/// @file test_watcher.cpp
/// @brief Google Test suite for the file watcher module.

#include <gtest/gtest.h>

#include "../src/watcher/watcher.hpp"
#include "../src/watcher/win_watcher.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// RAII wrapper for a temporary directory.  Creates the directory on
/// construction and removes it (recursively) on destruction.
class TempDir {
public:
    TempDir() {
        auto base = fs::temp_directory_path();
        // Use a unique name to avoid collisions across parallel test runs.
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = base / ("engram_watcher_test_" + std::to_string(now));
        fs::create_directories(path_);
    }

    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }

    const fs::path& path() const { return path_; }

    // Non-copyable.
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

private:
    fs::path path_;
};

/// Write a file with the given content.
static void write_file(const fs::path& path, const std::string& content) {
    // Ensure parent directory exists.
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(content.data(), static_cast<std::streamsize>(content.size()));
    ofs.close();
}

/// Helper that collects watcher events with a condition variable for
/// synchronization, so tests can wait for events without busy-looping.
class EventCollector {
public:
    void callback(const std::vector<engram::FileChange>& changes) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& c : changes) {
            events_.push_back(c);
        }
        event_count_ += static_cast<int>(changes.size());
        cv_.notify_all();
    }

    /// Wait until at least @p count events have been collected, or @p timeout
    /// elapses. Returns true if the condition was met.
    bool wait_for_events(int count, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [&] {
            return event_count_ >= count;
        });
    }

    /// Return a copy of all collected events.
    std::vector<engram::FileChange> events() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_;
    }

    /// Return the number of collected events.
    int count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return event_count_;
    }

    /// Reset the collector.
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.clear();
        event_count_ = 0;
    }

    /// Make a WatchCallback bound to this collector.
    engram::WatchCallback make_callback() {
        return [this](const std::vector<engram::FileChange>& changes) {
            callback(changes);
        };
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<engram::FileChange> events_;
    int event_count_ = 0;
};

/// Check if any event in a list has a path whose filename matches.
static bool has_event_for(const std::vector<engram::FileChange>& events,
                          const std::string& filename) {
    return std::any_of(events.begin(), events.end(),
        [&](const engram::FileChange& c) {
            return c.path.filename().string() == filename;
        });
}

/// Check if any event in a list has a path whose filename matches and the
/// given event type.
static bool has_event_for(const std::vector<engram::FileChange>& events,
                          const std::string& filename,
                          engram::FileEvent expected_event) {
    return std::any_of(events.begin(), events.end(),
        [&](const engram::FileChange& c) {
            return c.path.filename().string() == filename &&
                   c.event == expected_event;
        });
}

// ===========================================================================
// Path filtering tests (these work on all platforms)
// ===========================================================================

TEST(WatcherFilter, GitDirectoryIsFiltered) {
    EXPECT_TRUE(engram::WinFileWatcher::should_filter(fs::path(".git/config")));
    EXPECT_TRUE(engram::WinFileWatcher::should_filter(fs::path("src/.git/objects/pack")));
    EXPECT_TRUE(engram::WinFileWatcher::should_filter(fs::path(".git")));
}

TEST(WatcherFilter, BuildDirectoryIsFiltered) {
    EXPECT_TRUE(engram::WinFileWatcher::should_filter(fs::path("build/Release/foo.obj")));
    EXPECT_TRUE(engram::WinFileWatcher::should_filter(fs::path("build")));
}

TEST(WatcherFilter, NodeModulesIsFiltered) {
    EXPECT_TRUE(engram::WinFileWatcher::should_filter(fs::path("node_modules/express/index.js")));
}

TEST(WatcherFilter, SwapFilesAreFiltered) {
    EXPECT_TRUE(engram::WinFileWatcher::should_filter(fs::path("src/main.cpp.swp")));
    EXPECT_TRUE(engram::WinFileWatcher::should_filter(fs::path("src/main.cpp.swo")));
}

TEST(WatcherFilter, TempFilesAreFiltered) {
    EXPECT_TRUE(engram::WinFileWatcher::should_filter(fs::path("data/output.tmp")));
    EXPECT_TRUE(engram::WinFileWatcher::should_filter(fs::path("file.bak")));
}

TEST(WatcherFilter, TildeFilesAreFiltered) {
    EXPECT_TRUE(engram::WinFileWatcher::should_filter(fs::path("~$document.docx")));
    EXPECT_TRUE(engram::WinFileWatcher::should_filter(fs::path("src/main.cpp~")));
}

TEST(WatcherFilter, EmacsLockFilesAreFiltered) {
    EXPECT_TRUE(engram::WinFileWatcher::should_filter(fs::path(".#main.cpp")));
}

TEST(WatcherFilter, NormalSourceFilesPassThrough) {
    EXPECT_FALSE(engram::WinFileWatcher::should_filter(fs::path("src/main.cpp")));
    EXPECT_FALSE(engram::WinFileWatcher::should_filter(fs::path("CMakeLists.txt")));
    EXPECT_FALSE(engram::WinFileWatcher::should_filter(fs::path("scripts/test.py")));
    EXPECT_FALSE(engram::WinFileWatcher::should_filter(fs::path("README.md")));
}

TEST(WatcherFilter, VsDirectoryIsFiltered) {
    EXPECT_TRUE(engram::WinFileWatcher::should_filter(fs::path(".vs/settings.json")));
}

TEST(WatcherFilter, PycacheIsFiltered) {
    EXPECT_TRUE(engram::WinFileWatcher::should_filter(fs::path("__pycache__/module.pyc")));
}

// ===========================================================================
// Watcher lifecycle tests (basic, platform-independent)
// ===========================================================================

TEST(WatcherLifecycle, DefaultConstruction) {
    engram::WinFileWatcher watcher;
    EXPECT_FALSE(watcher.is_watching());
}

TEST(WatcherLifecycle, StopWhenNotWatching) {
    engram::WinFileWatcher watcher;
    // Should be a no-op, not crash.
    watcher.stop();
    watcher.stop();
    EXPECT_FALSE(watcher.is_watching());
}

TEST(WatcherLifecycle, MoveConstruction) {
    engram::WinFileWatcher watcher1;
    engram::WinFileWatcher watcher2(std::move(watcher1));
    EXPECT_FALSE(watcher2.is_watching());
}

TEST(WatcherLifecycle, MoveAssignment) {
    engram::WinFileWatcher watcher1;
    engram::WinFileWatcher watcher2;
    watcher2 = std::move(watcher1);
    EXPECT_FALSE(watcher2.is_watching());
}

TEST(WatcherLifecycle, StartFailsOnNonexistentDirectory) {
    engram::WinFileWatcher watcher;
    EventCollector collector;
    bool ok = watcher.start(fs::path("this/does/not/exist/at/all"),
                            collector.make_callback());
    EXPECT_FALSE(ok);
    EXPECT_FALSE(watcher.is_watching());
}

TEST(WatcherLifecycle, PolymorphismThroughBasePointer) {
    std::unique_ptr<engram::FileWatcher> watcher =
        std::make_unique<engram::WinFileWatcher>();
    EXPECT_FALSE(watcher->is_watching());
    watcher->stop();  // No-op, should not crash.
}

// ===========================================================================
// Platform-specific tests: actual file watching (Windows only)
// ===========================================================================

#ifdef _WIN32

TEST(WatcherWin, StartAndStopSucceeds) {
    TempDir dir;
    engram::WinFileWatcher watcher;
    EventCollector collector;

    ASSERT_TRUE(watcher.start(dir.path(), collector.make_callback()));
    EXPECT_TRUE(watcher.is_watching());

    watcher.stop();
    EXPECT_FALSE(watcher.is_watching());
}

TEST(WatcherWin, DoubleStartFails) {
    TempDir dir;
    engram::WinFileWatcher watcher;
    EventCollector collector;

    ASSERT_TRUE(watcher.start(dir.path(), collector.make_callback()));

    // Starting again without stopping should fail.
    EXPECT_FALSE(watcher.start(dir.path(), collector.make_callback()));

    watcher.stop();
}

TEST(WatcherWin, DoubleStopIsSafe) {
    TempDir dir;
    engram::WinFileWatcher watcher;
    EventCollector collector;

    ASSERT_TRUE(watcher.start(dir.path(), collector.make_callback()));

    watcher.stop();
    watcher.stop();  // Should be a no-op, not crash.
    EXPECT_FALSE(watcher.is_watching());
}

TEST(WatcherWin, DetectsFileCreation) {
    TempDir dir;
    engram::WinWatcherConfig config;
    config.debounce_ms = std::chrono::milliseconds(50);
    engram::WinFileWatcher watcher(config);
    EventCollector collector;

    ASSERT_TRUE(watcher.start(dir.path(), collector.make_callback()));

    // Create a file.
    write_file(dir.path() / "hello.txt", "Hello, world!");

    // Wait for at least one event.
    ASSERT_TRUE(collector.wait_for_events(1, std::chrono::seconds(5)))
        << "Timed out waiting for file creation event";

    auto events = collector.events();
    EXPECT_TRUE(has_event_for(events, "hello.txt"))
        << "Expected an event for hello.txt";

    watcher.stop();
}

TEST(WatcherWin, DetectsFileModification) {
    TempDir dir;
    // Pre-create the file before watching so we can detect modification.
    write_file(dir.path() / "data.txt", "initial content");

    engram::WinWatcherConfig config;
    config.debounce_ms = std::chrono::milliseconds(50);
    engram::WinFileWatcher watcher(config);
    EventCollector collector;

    ASSERT_TRUE(watcher.start(dir.path(), collector.make_callback()));

    // Give the watcher a moment to set up.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Modify the file.
    write_file(dir.path() / "data.txt", "modified content");

    ASSERT_TRUE(collector.wait_for_events(1, std::chrono::seconds(5)))
        << "Timed out waiting for file modification event";

    auto events = collector.events();
    EXPECT_TRUE(has_event_for(events, "data.txt"))
        << "Expected an event for data.txt";

    watcher.stop();
}

TEST(WatcherWin, DetectsFileDeletion) {
    TempDir dir;
    write_file(dir.path() / "doomed.txt", "this will be deleted");

    engram::WinWatcherConfig config;
    config.debounce_ms = std::chrono::milliseconds(50);
    engram::WinFileWatcher watcher(config);
    EventCollector collector;

    ASSERT_TRUE(watcher.start(dir.path(), collector.make_callback()));

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Delete the file.
    std::error_code ec;
    fs::remove(dir.path() / "doomed.txt", ec);

    ASSERT_TRUE(collector.wait_for_events(1, std::chrono::seconds(5)))
        << "Timed out waiting for file deletion event";

    auto events = collector.events();
    EXPECT_TRUE(has_event_for(events, "doomed.txt", engram::FileEvent::Deleted))
        << "Expected a Deleted event for doomed.txt";

    watcher.stop();
}

TEST(WatcherWin, DetectsSubdirectoryChanges) {
    TempDir dir;
    fs::create_directories(dir.path() / "sub" / "deep");

    engram::WinWatcherConfig config;
    config.debounce_ms = std::chrono::milliseconds(50);
    engram::WinFileWatcher watcher(config);
    EventCollector collector;

    ASSERT_TRUE(watcher.start(dir.path(), collector.make_callback()));

    // Create a file in a subdirectory.
    write_file(dir.path() / "sub" / "deep" / "nested.cpp", "int main() {}");

    ASSERT_TRUE(collector.wait_for_events(1, std::chrono::seconds(5)))
        << "Timed out waiting for subdirectory file creation event";

    auto events = collector.events();
    EXPECT_TRUE(has_event_for(events, "nested.cpp"))
        << "Expected an event for nested.cpp in subdirectory";

    watcher.stop();
}

TEST(WatcherWin, FiltersGitDirectory) {
    TempDir dir;
    fs::create_directories(dir.path() / ".git" / "objects");

    engram::WinWatcherConfig config;
    config.debounce_ms = std::chrono::milliseconds(50);
    engram::WinFileWatcher watcher(config);
    EventCollector collector;

    ASSERT_TRUE(watcher.start(dir.path(), collector.make_callback()));

    // Create a file inside .git/ — should be filtered out.
    write_file(dir.path() / ".git" / "objects" / "abc123", "git object data");

    // Also create a normal file to verify the watcher is working.
    write_file(dir.path() / "source.cpp", "int x = 42;");

    ASSERT_TRUE(collector.wait_for_events(1, std::chrono::seconds(5)))
        << "Timed out waiting for source.cpp event";

    auto events = collector.events();

    // The .git file should NOT appear in events.
    EXPECT_FALSE(has_event_for(events, "abc123"))
        << "Events from .git/ should be filtered out";

    // The normal file should appear.
    EXPECT_TRUE(has_event_for(events, "source.cpp"))
        << "Expected an event for source.cpp";

    watcher.stop();
}

TEST(WatcherWin, FiltersTempFiles) {
    TempDir dir;
    engram::WinWatcherConfig config;
    config.debounce_ms = std::chrono::milliseconds(50);
    engram::WinFileWatcher watcher(config);
    EventCollector collector;

    ASSERT_TRUE(watcher.start(dir.path(), collector.make_callback()));

    // Create a swap file — should be filtered.
    write_file(dir.path() / ".main.cpp.swp", "vim swap data");

    // Create a normal file to confirm watcher works.
    write_file(dir.path() / "real.cpp", "int main() {}");

    ASSERT_TRUE(collector.wait_for_events(1, std::chrono::seconds(5)))
        << "Timed out waiting for events";

    auto events = collector.events();
    EXPECT_FALSE(has_event_for(events, ".main.cpp.swp"))
        << "Swap files should be filtered out";
    EXPECT_TRUE(has_event_for(events, "real.cpp"))
        << "Expected an event for real.cpp";

    watcher.stop();
}

TEST(WatcherWin, StopAndRestart) {
    TempDir dir;
    engram::WinWatcherConfig config;
    config.debounce_ms = std::chrono::milliseconds(50);
    engram::WinFileWatcher watcher(config);
    EventCollector collector;

    // First run.
    ASSERT_TRUE(watcher.start(dir.path(), collector.make_callback()));
    write_file(dir.path() / "first.txt", "first");
    ASSERT_TRUE(collector.wait_for_events(1, std::chrono::seconds(5)));
    watcher.stop();

    // Reset the collector.
    collector.reset();

    // Second run — should work again after stop.
    ASSERT_TRUE(watcher.start(dir.path(), collector.make_callback()));
    write_file(dir.path() / "second.txt", "second");
    ASSERT_TRUE(collector.wait_for_events(1, std::chrono::seconds(5)));

    auto events = collector.events();
    EXPECT_TRUE(has_event_for(events, "second.txt"))
        << "Expected event for second.txt after restart";

    watcher.stop();
}

TEST(WatcherWin, DestructorStopsWatching) {
    TempDir dir;
    EventCollector collector;

    {
        engram::WinFileWatcher watcher;
        ASSERT_TRUE(watcher.start(dir.path(), collector.make_callback()));
        EXPECT_TRUE(watcher.is_watching());
        // Destructor should call stop() automatically.
    }

    // If we get here without hanging, the destructor worked correctly.
    SUCCEED();
}

TEST(WatcherWin, MultipleFilesInBatch) {
    TempDir dir;
    engram::WinWatcherConfig config;
    config.debounce_ms = std::chrono::milliseconds(100);
    engram::WinFileWatcher watcher(config);
    EventCollector collector;

    ASSERT_TRUE(watcher.start(dir.path(), collector.make_callback()));

    // Create several files in quick succession — they should be batched.
    write_file(dir.path() / "a.cpp", "int a;");
    write_file(dir.path() / "b.cpp", "int b;");
    write_file(dir.path() / "c.cpp", "int c;");

    // Wait for all three events (may arrive as one batch or multiple).
    ASSERT_TRUE(collector.wait_for_events(3, std::chrono::seconds(5)))
        << "Timed out waiting for batch of file creation events";

    auto events = collector.events();
    EXPECT_TRUE(has_event_for(events, "a.cpp"));
    EXPECT_TRUE(has_event_for(events, "b.cpp"));
    EXPECT_TRUE(has_event_for(events, "c.cpp"));

    watcher.stop();
}

TEST(WatcherWin, EventPathsAreAbsolute) {
    TempDir dir;
    engram::WinWatcherConfig config;
    config.debounce_ms = std::chrono::milliseconds(50);
    engram::WinFileWatcher watcher(config);
    EventCollector collector;

    ASSERT_TRUE(watcher.start(dir.path(), collector.make_callback()));

    write_file(dir.path() / "absolute_test.txt", "data");

    ASSERT_TRUE(collector.wait_for_events(1, std::chrono::seconds(5)));

    auto events = collector.events();
    for (const auto& e : events) {
        EXPECT_TRUE(e.path.is_absolute())
            << "Event path should be absolute: " << e.path.string();
    }

    watcher.stop();
}

#endif // _WIN32
