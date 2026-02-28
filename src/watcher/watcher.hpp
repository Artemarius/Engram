#pragma once

/// @file watcher.hpp
/// @brief Abstract filesystem watcher interface for Engram.
///
/// Defines the types and base class for monitoring directory trees for file
/// changes.  Concrete implementations use platform-specific APIs
/// (ReadDirectoryChangesW on Windows, inotify on Linux, etc.).

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace engram {

/// The kind of filesystem event observed.
enum class FileEvent : uint8_t {
    Created,   ///< A new file was created.
    Modified,  ///< An existing file's content was changed.
    Deleted,   ///< A file was deleted.
    Renamed    ///< A file was renamed (old path in `path`, new path unavailable).
};

/// A single observed filesystem change.
struct FileChange {
    std::filesystem::path path;  ///< Absolute path to the affected file.
    FileEvent             event; ///< What happened.
};

/// Callback type invoked by the watcher when file changes are detected.
///
/// The watcher batches changes and delivers them in groups.  Callbacks are
/// invoked on the watcher's background thread, so implementations must be
/// thread-safe or post work to another thread.
using WatchCallback = std::function<void(const std::vector<FileChange>&)>;

/// @class FileWatcher
/// @brief Pure-virtual interface for recursive directory watchers.
///
/// Implementations monitor a directory tree for file creation, modification,
/// deletion, and renames.  Only one directory can be watched at a time per
/// FileWatcher instance.
class FileWatcher {
public:
    virtual ~FileWatcher() = default;

    /// @brief Start watching the given directory recursively.
    ///
    /// File change events are delivered to @p callback on a background thread.
    /// Calling start() while already watching is an error and returns false;
    /// call stop() first.
    ///
    /// @param directory  The root directory to watch.  Must exist.
    /// @param callback   Function invoked with batched file change events.
    /// @return true if watching started successfully, false on error.
    virtual bool start(const std::filesystem::path& directory,
                       WatchCallback callback) = 0;

    /// @brief Stop watching and release all associated resources.
    ///
    /// Blocks until the background watcher thread has exited.  Safe to call
    /// even if not currently watching (no-op).
    virtual void stop() = 0;

    /// @brief Returns true if the watcher is currently active.
    virtual bool is_watching() const = 0;
};

} // namespace engram
