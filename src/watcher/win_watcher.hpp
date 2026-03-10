#pragma once

/// @file win_watcher.hpp
/// @brief Windows file watcher implementation using ReadDirectoryChangesW.
///
/// Monitors a directory tree for file creation, modification, deletion, and
/// renames using the Windows ReadDirectoryChangesW API.  Change events are
/// collected and debounced before delivery to a user-supplied callback on a
/// background thread.

#include "watcher.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace engram {

/// Configuration for the Windows file watcher.
struct WinWatcherConfig {
    /// Debounce window in milliseconds.  Events are batched over this period
    /// before the callback is invoked.
    std::chrono::milliseconds debounce_ms{200};
};

/// @class WinFileWatcher
/// @brief Concrete FileWatcher using ReadDirectoryChangesW (Windows only).
///
/// Runs a background thread that calls ReadDirectoryChangesW in a loop.
/// Uses an Event object for clean shutdown signaling.  Change events are
/// debounced and filtered before delivery to the user callback.
///
/// Thread-safety: start() and stop() must not be called concurrently.
/// The callback is invoked on the background thread.
///
/// Movable but not copyable.
class WinFileWatcher final : public FileWatcher {
public:
    /// @brief Construct a watcher with default configuration.
    WinFileWatcher();

    /// @brief Construct a watcher with custom configuration.
    /// @param config  Watcher parameters (debounce window, etc.).
    explicit WinFileWatcher(WinWatcherConfig config);

    /// @brief Destructor. Calls stop() if still watching.
    ~WinFileWatcher() override;

    // Non-copyable.
    WinFileWatcher(const WinFileWatcher&) = delete;
    WinFileWatcher& operator=(const WinFileWatcher&) = delete;

    // Movable.
    WinFileWatcher(WinFileWatcher&& other) noexcept;
    WinFileWatcher& operator=(WinFileWatcher&& other) noexcept;

    /// @brief Start watching the given directory recursively.
    ///
    /// Opens a directory handle and spawns a background thread that monitors
    /// for file changes.  Events are debounced and delivered to @p callback.
    ///
    /// @param directory  The root directory to watch. Must exist.
    /// @param callback   Function invoked with batched file change events.
    /// @return true if watching started successfully, false on error.
    bool start(const std::filesystem::path& directory,
               WatchCallback callback) override;

    /// @brief Stop watching and release all resources.
    ///
    /// Signals the background thread to exit and blocks until it joins.
    /// Safe to call multiple times and from the destructor.
    void stop() override;

    /// @brief Returns true if the watcher is currently active.
    bool is_watching() const override;

    /// @brief Check whether a path should be filtered (ignored).
    ///
    /// Returns true for paths containing common noise directories or
    /// temporary/swap files that should not trigger indexing.
    ///
    /// @param path  The path to check (relative or absolute).
    /// @return true if the path should be ignored.
    static bool should_filter(const std::filesystem::path& path);

private:
#ifdef _WIN32
    /// The background thread function that calls ReadDirectoryChangesW.
    void watch_loop();

    /// Process a FILE_NOTIFY_INFORMATION buffer and append events to the
    /// pending batch.
    void process_notifications(const BYTE* buffer, DWORD bytes_transferred);

    /// Map a FILE_ACTION_* constant to a FileEvent enum value.
    static FileEvent map_action(DWORD action);

    /// Flush the pending event batch to the user callback.
    void flush_pending();

    WinWatcherConfig config_;

    std::filesystem::path watch_dir_;
    WatchCallback callback_;

    /// Directory handle opened with CreateFileW.
    HANDLE dir_handle_ = INVALID_HANDLE_VALUE;

    /// Event object used to signal the background thread to stop.
    HANDLE stop_event_ = nullptr;

    /// Event object signaled by the watch loop once ReadDirectoryChangesW
    /// has been issued for the first time.  start() waits on this so that
    /// callers can safely create files immediately after start() returns.
    HANDLE ready_event_ = nullptr;

    /// Background watcher thread.
    std::thread watch_thread_;

    /// Flag indicating whether the watcher is active.
    std::atomic<bool> watching_{false};

    /// Pending events accumulated during the debounce window.
    std::vector<FileChange> pending_events_;

    /// Mutex protecting pending_events_.
    std::mutex pending_mutex_;
#else
    // On non-Windows platforms, the watcher is a no-op stub.
    std::atomic<bool> watching_{false};
#endif
};

} // namespace engram
