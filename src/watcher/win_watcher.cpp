/// @file win_watcher.cpp
/// @brief Windows file watcher implementation using ReadDirectoryChangesW.

#include "win_watcher.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <string>
#include <utility>

namespace fs = std::filesystem;

namespace engram {

// ---------------------------------------------------------------------------
// Path filtering
// ---------------------------------------------------------------------------

bool WinFileWatcher::should_filter(const fs::path& path) {
    // Convert to generic string for consistent separator handling.
    const std::string generic = path.generic_string();

    // Directories to ignore (match as path components).
    static const char* const kIgnoredDirs[] = {
        ".git",
        "build",
        "node_modules",
        ".vs",
        ".vscode",
        "__pycache__",
        ".cache",
    };

    for (const auto* dir : kIgnoredDirs) {
        // Check for "/dir/" in the middle or "/dir" at the end,
        // or "dir/" at the start.
        const std::string slash_dir_slash = std::string("/") + dir + "/";
        const std::string slash_dir = std::string("/") + dir;
        const std::string dir_slash = std::string(dir) + "/";

        if (generic.find(slash_dir_slash) != std::string::npos) return true;
        // Check if path ends with /dir
        if (generic.size() >= slash_dir.size() &&
            generic.compare(generic.size() - slash_dir.size(), slash_dir.size(), slash_dir) == 0) {
            return true;
        }
        // Check if path starts with dir/
        if (generic.compare(0, dir_slash.size(), dir_slash) == 0) return true;
        // Exact match (single component)
        if (generic == dir) return true;
    }

    // File extensions to ignore (temp files, swap files, etc.).
    const std::string ext = path.extension().string();
    static const char* const kIgnoredExts[] = {
        ".swp", ".swo", ".swn",     // Vim swap files
        ".tmp", ".temp",             // Generic temp files
        ".bak",                      // Backup files
        ".orig",                     // Merge originals
    };

    for (const auto* ignored_ext : kIgnoredExts) {
        if (ext == ignored_ext) return true;
    }

    // Filename patterns to ignore.
    const std::string filename = path.filename().string();
    if (filename.empty()) return false;

    // Files starting with ~ (Office temp files, etc.)
    if (filename[0] == '~') return true;

    // Files starting with .# (Emacs lock files)
    if (filename.size() >= 2 && filename[0] == '.' && filename[1] == '#') return true;

    // Files ending with ~ (backup files from various editors)
    if (filename.back() == '~') return true;

    return false;
}

// ---------------------------------------------------------------------------
// Platform-specific implementation
// ---------------------------------------------------------------------------

#ifdef _WIN32

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

WinFileWatcher::WinFileWatcher()
    : config_{}
{
}

WinFileWatcher::WinFileWatcher(WinWatcherConfig config)
    : config_(std::move(config))
{
}

WinFileWatcher::~WinFileWatcher() {
    stop();
}

// ---------------------------------------------------------------------------
// Move operations
// ---------------------------------------------------------------------------

WinFileWatcher::WinFileWatcher(WinFileWatcher&& other) noexcept
    : config_(std::move(other.config_))
    , watch_dir_(std::move(other.watch_dir_))
    , callback_(std::move(other.callback_))
    , dir_handle_(other.dir_handle_)
    , stop_event_(other.stop_event_)
    , ready_event_(other.ready_event_)
    , watch_thread_(std::move(other.watch_thread_))
    , watching_(other.watching_.load())
    , pending_events_(std::move(other.pending_events_))
{
    other.dir_handle_ = INVALID_HANDLE_VALUE;
    other.stop_event_ = nullptr;
    other.ready_event_ = nullptr;
    other.watching_.store(false);
}

WinFileWatcher& WinFileWatcher::operator=(WinFileWatcher&& other) noexcept {
    if (this != &other) {
        stop();  // Clean up current state before moving.

        config_ = std::move(other.config_);
        watch_dir_ = std::move(other.watch_dir_);
        callback_ = std::move(other.callback_);
        dir_handle_ = other.dir_handle_;
        stop_event_ = other.stop_event_;
        ready_event_ = other.ready_event_;
        watch_thread_ = std::move(other.watch_thread_);
        watching_.store(other.watching_.load());
        pending_events_ = std::move(other.pending_events_);

        other.dir_handle_ = INVALID_HANDLE_VALUE;
        other.stop_event_ = nullptr;
        other.ready_event_ = nullptr;
        other.watching_.store(false);
    }
    return *this;
}

// ---------------------------------------------------------------------------
// start / stop / is_watching
// ---------------------------------------------------------------------------

bool WinFileWatcher::start(const fs::path& directory, WatchCallback callback) {
    if (watching_.load()) {
        spdlog::error("WinFileWatcher::start() called while already watching");
        return false;
    }

    // Verify the directory exists.
    std::error_code ec;
    if (!fs::is_directory(directory, ec)) {
        spdlog::error("WinFileWatcher::start(): '{}' is not a valid directory",
                       directory.string());
        return false;
    }

    // Canonicalize the path.
    watch_dir_ = fs::canonical(directory, ec);
    if (ec) {
        spdlog::error("WinFileWatcher::start(): failed to canonicalize '{}': {}",
                       directory.string(), ec.message());
        return false;
    }

    callback_ = std::move(callback);

    // Open the directory for overlapped I/O.
    dir_handle_ = CreateFileW(
        watch_dir_.wstring().c_str(),
        FILE_LIST_DIRECTORY,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED,
        nullptr
    );

    if (dir_handle_ == INVALID_HANDLE_VALUE) {
        spdlog::error("WinFileWatcher::start(): CreateFileW failed (error {})",
                       GetLastError());
        return false;
    }

    // Create a manual-reset event for shutdown signaling.
    stop_event_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (stop_event_ == nullptr) {
        spdlog::error("WinFileWatcher::start(): CreateEventW failed (error {})",
                       GetLastError());
        CloseHandle(dir_handle_);
        dir_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    // Create a manual-reset event that the watch loop signals once it has
    // issued the first ReadDirectoryChangesW call.  start() waits on this
    // so callers can safely create files right after start() returns.
    ready_event_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (ready_event_ == nullptr) {
        spdlog::error("WinFileWatcher::start(): CreateEventW (ready) failed (error {})",
                       GetLastError());
        CloseHandle(stop_event_);
        stop_event_ = nullptr;
        CloseHandle(dir_handle_);
        dir_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    watching_.store(true);

    // Spawn the background thread.
    watch_thread_ = std::thread([this]() { watch_loop(); });

    // Wait for the watch loop to issue its first ReadDirectoryChangesW.
    WaitForSingleObject(ready_event_, 5000);

    spdlog::info("WinFileWatcher: watching '{}'", watch_dir_.string());
    return true;
}

void WinFileWatcher::stop() {
    if (!watching_.load()) {
        return;
    }

    spdlog::debug("WinFileWatcher::stop(): signaling shutdown");

    // Signal the background thread to exit.
    if (stop_event_ != nullptr) {
        SetEvent(stop_event_);
    }

    // Cancel any pending I/O on the directory handle so the thread wakes up.
    if (dir_handle_ != INVALID_HANDLE_VALUE) {
        CancelIoEx(dir_handle_, nullptr);
    }

    // Wait for the thread to finish.
    if (watch_thread_.joinable()) {
        watch_thread_.join();
    }

    // Close handles.
    if (dir_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(dir_handle_);
        dir_handle_ = INVALID_HANDLE_VALUE;
    }

    if (stop_event_ != nullptr) {
        CloseHandle(stop_event_);
        stop_event_ = nullptr;
    }

    if (ready_event_ != nullptr) {
        CloseHandle(ready_event_);
        ready_event_ = nullptr;
    }

    watching_.store(false);

    spdlog::info("WinFileWatcher: stopped");
}

bool WinFileWatcher::is_watching() const {
    return watching_.load();
}

// ---------------------------------------------------------------------------
// Background watch loop
// ---------------------------------------------------------------------------

void WinFileWatcher::watch_loop() {
    // Buffer for ReadDirectoryChangesW results.
    // 64 KB is generous and avoids most overflow cases.
    constexpr DWORD kBufSize = 64 * 1024;
    std::vector<BYTE> buffer(kBufSize);

    OVERLAPPED overlapped{};
    overlapped.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (overlapped.hEvent == nullptr) {
        spdlog::error("WinFileWatcher: failed to create overlapped event");
        watching_.store(false);
        return;
    }

    const DWORD kFilter =
        FILE_NOTIFY_CHANGE_FILE_NAME |
        FILE_NOTIFY_CHANGE_LAST_WRITE |
        FILE_NOTIFY_CHANGE_DIR_NAME;

    bool first_call = true;

    while (watching_.load()) {
        ResetEvent(overlapped.hEvent);

        BOOL ok = ReadDirectoryChangesW(
            dir_handle_,
            buffer.data(),
            kBufSize,
            TRUE,       // watch subtree
            kFilter,
            nullptr,    // bytes returned (not used with overlapped)
            &overlapped,
            nullptr     // completion routine
        );

        if (!ok) {
            DWORD err = GetLastError();
            if (first_call && ready_event_) SetEvent(ready_event_);
            if (err == ERROR_OPERATION_ABORTED) {
                // Cancelled by stop() — normal shutdown path.
                break;
            }
            spdlog::error("WinFileWatcher: ReadDirectoryChangesW failed (error {})", err);
            break;
        }

        // Signal that monitoring is active so start() can return.
        if (first_call) {
            first_call = false;
            if (ready_event_) SetEvent(ready_event_);
        }

        // Wait for either the directory notification or the stop signal.
        HANDLE handles[] = { overlapped.hEvent, stop_event_ };
        DWORD wait_result = WaitForMultipleObjects(2, handles, FALSE, INFINITE);

        if (wait_result == WAIT_OBJECT_0) {
            // Directory change notification received.
            DWORD bytes_transferred = 0;
            if (GetOverlappedResult(dir_handle_, &overlapped, &bytes_transferred, FALSE)) {
                if (bytes_transferred == 0) {
                    // Buffer overflow (ERROR_NOTIFY_ENUM_DIR condition).
                    // The system could not fit all notifications into the buffer.
                    spdlog::warn("WinFileWatcher: notification buffer overflow — "
                                 "some events may have been lost. Re-watching.");
                    continue;
                }
                process_notifications(buffer.data(), bytes_transferred);

                // After processing, wait for the debounce window, then flush.
                // Use the stop event as an interruptible sleep.
                DWORD debounce_ms = static_cast<DWORD>(config_.debounce_ms.count());
                DWORD sleep_result = WaitForSingleObject(stop_event_, debounce_ms);
                if (sleep_result == WAIT_OBJECT_0) {
                    // Stop was signaled during debounce — flush and exit.
                    flush_pending();
                    break;
                }
                // Debounce window elapsed — flush accumulated events.
                flush_pending();
            } else {
                DWORD err = GetLastError();
                if (err == ERROR_OPERATION_ABORTED) {
                    break;  // Cancelled by stop().
                }
                spdlog::error("WinFileWatcher: GetOverlappedResult failed (error {})", err);
                break;
            }
        } else if (wait_result == WAIT_OBJECT_0 + 1) {
            // Stop event was signaled.
            CancelIoEx(dir_handle_, &overlapped);
            break;
        } else {
            // Unexpected wait result.
            spdlog::error("WinFileWatcher: WaitForMultipleObjects returned {}", wait_result);
            break;
        }
    }

    CloseHandle(overlapped.hEvent);
    spdlog::debug("WinFileWatcher: watch loop exited");
}

// ---------------------------------------------------------------------------
// Notification processing
// ---------------------------------------------------------------------------

void WinFileWatcher::process_notifications(const BYTE* buffer, DWORD bytes_transferred) {
    const auto* info = reinterpret_cast<const FILE_NOTIFY_INFORMATION*>(buffer);

    for (;;) {
        // Extract the filename from the wide-char buffer.
        // FileNameLength is in bytes, not characters.
        const DWORD name_chars = info->FileNameLength / sizeof(WCHAR);
        std::wstring wname(info->FileName, name_chars);

        // Build the full absolute path.
        fs::path full_path = watch_dir_ / fs::path(wname);

        // Apply filtering.
        // Use the relative path for filter matching (more reliable).
        fs::path rel_path = fs::path(wname);
        if (!should_filter(rel_path)) {
            FileChange change;
            change.path = std::move(full_path);
            change.event = map_action(info->Action);

            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_events_.push_back(std::move(change));
        }

        // Advance to the next entry.
        if (info->NextEntryOffset == 0) {
            break;
        }
        info = reinterpret_cast<const FILE_NOTIFY_INFORMATION*>(
            reinterpret_cast<const BYTE*>(info) + info->NextEntryOffset);
    }
}

FileEvent WinFileWatcher::map_action(DWORD action) {
    switch (action) {
        case FILE_ACTION_ADDED:            return FileEvent::Created;
        case FILE_ACTION_REMOVED:          return FileEvent::Deleted;
        case FILE_ACTION_MODIFIED:         return FileEvent::Modified;
        case FILE_ACTION_RENAMED_OLD_NAME: return FileEvent::Renamed;
        case FILE_ACTION_RENAMED_NEW_NAME: return FileEvent::Created;  // Treat new name as a creation.
        default:
            spdlog::warn("WinFileWatcher: unknown file action {}", action);
            return FileEvent::Modified;
    }
}

// ---------------------------------------------------------------------------
// Debounced flush
// ---------------------------------------------------------------------------

void WinFileWatcher::flush_pending() {
    std::vector<FileChange> batch;
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        if (pending_events_.empty()) {
            return;
        }
        batch.swap(pending_events_);
    }

    // Deduplicate: if the same path appears multiple times, keep the last event.
    // This handles the common case of a file being written multiple times during
    // a save operation.
    std::vector<FileChange> deduped;
    deduped.reserve(batch.size());

    // Walk backwards, keeping the first (latest) occurrence of each path.
    std::vector<std::string> seen;
    for (auto it = batch.rbegin(); it != batch.rend(); ++it) {
        std::string key = it->path.generic_string();
        if (std::find(seen.begin(), seen.end(), key) == seen.end()) {
            seen.push_back(key);
            deduped.push_back(std::move(*it));
        }
    }

    // Reverse to restore chronological order.
    std::reverse(deduped.begin(), deduped.end());

    if (!deduped.empty() && callback_) {
        spdlog::debug("WinFileWatcher: delivering {} events", deduped.size());
        callback_(deduped);
    }
}

#else // !_WIN32

// ---------------------------------------------------------------------------
// Non-Windows stub implementation
// ---------------------------------------------------------------------------

WinFileWatcher::WinFileWatcher() = default;

WinFileWatcher::WinFileWatcher(WinWatcherConfig /*config*/) {}

WinFileWatcher::~WinFileWatcher() = default;

WinFileWatcher::WinFileWatcher(WinFileWatcher&& other) noexcept
    : watching_(other.watching_.load())
{
    other.watching_.store(false);
}

WinFileWatcher& WinFileWatcher::operator=(WinFileWatcher&& other) noexcept {
    if (this != &other) {
        watching_.store(other.watching_.load());
        other.watching_.store(false);
    }
    return *this;
}

bool WinFileWatcher::start(const fs::path& /*directory*/, WatchCallback /*callback*/) {
    spdlog::error("WinFileWatcher is only supported on Windows");
    return false;
}

void WinFileWatcher::stop() {
    watching_.store(false);
}

bool WinFileWatcher::is_watching() const {
    return watching_.load();
}

#endif // _WIN32

} // namespace engram
