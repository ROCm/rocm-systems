/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "top.hpp"
#include "utils/debug.hpp"
#include "os/os.hpp"

#if !defined(AMD_LOG_LEVEL)
#include "utils/flags.hpp"
#endif

#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <thread>
#include <sstream>
#include <iomanip>
#include <inttypes.h>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <memory>
#include <chrono>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif  // _WIN32

namespace amd {

FILE* outFile = stderr;

void truncate_log_file();
static void TruncateLogFileFlushPath();

static void crashFlushCallback();

// ================================================================================================
// Async logging infrastructure
// ================================================================================================

static std::atomic<bool> crashHandlersInstalled_{false};

struct LogEntry {
  LogLevel level;          //!< Log severity level
  const char* file;        //!< Source file name
  int line;                //!< Source line number
  std::string message;     //!< Formatted log message
  uint64_t timestamp;      //!< Timestamp in microseconds
  uint32_t pid;            //!< Process ID
  uint32_t tid;            //!< Thread ID (hash)
  uint64_t duration;       //!< Duration in microseconds (0 if not a duration log)
  bool hasDuration;        //!< True if this is a duration log entry
  std::atomic<bool> valid; //!< Valid flag for lock-free synchronization

  LogEntry()
      : level(LOG_NONE),
        file(""),
        line(0),
        timestamp(0),
        pid(0),
        duration(0),
        hasDuration(false),
        valid(false) {}
};

class AsyncLogger {
 public:
  AsyncLogger() : buffer_(kBufferSize) {
    if (!flagIsDefault(AMD_LOG_ASYNC) && AMD_LOG_ASYNC) {
      enable(true);
    }
  }

  ~AsyncLogger() {
    enable(false);
    stop();
  }

  void start() {
    if (!running_.load(std::memory_order_relaxed)) {
      running_.store(true, std::memory_order_relaxed);
      enabled_.store(true, std::memory_order_relaxed);
      workerThread_ = std::thread(&AsyncLogger::workerLoop, this);
    }
  }

  void stop() {
    if (running_.load(std::memory_order_relaxed)) {
      running_.store(false, std::memory_order_relaxed);
      flushCV_.notify_all();
      if (workerThread_.joinable()) {
        workerThread_.join();
      }
    }
  }

  void enable(bool enable) {
    enabled_.store(enable, std::memory_order_relaxed);
    if (enable && !running_.load(std::memory_order_relaxed)) {
      start();
    }

    if (enable) {
      bool expected = false;
      if (crashHandlersInstalled_.compare_exchange_strong(expected, true,
                                                          std::memory_order_acq_rel)) {
        if (!Os::installExceptionHandlers(crashFlushCallback)) {
          crashHandlersInstalled_.store(false, std::memory_order_release);
        }
      }
    } else {
      bool expected = true;
      if (crashHandlersInstalled_.compare_exchange_strong(expected, false,
                                                          std::memory_order_acq_rel)) {
        Os::uninstallExceptionHandlers();
      }
    }
  }

  bool isEnabled() const {
    return enabled_.load(std::memory_order_relaxed);
  }

  void log(LogLevel level, const char* file, int line, const char* message,
           uint64_t timestamp, uint64_t duration = 0, bool hasDuration = false) {
    if (!enabled_.load(std::memory_order_relaxed)) {
      return;  // Fall back to sync logging
    }

    size_t currentWrite = writeIndex_.fetch_add(1, std::memory_order_release);
    size_t currentRead = readIndex_.load(std::memory_order_acquire);

    // Check if buffer is full
    while (currentWrite - currentRead >= kBufferSize) {
      flush();
      currentRead = readIndex_.load(std::memory_order_acquire);
    }
    // Write to buffer (lock-free)
    LogEntry& entry = buffer_[currentWrite % kBufferSize];
    entry.level = level;
    entry.file = file ? file : "";
    entry.line = line;
    entry.message = message ? message : "";
    entry.timestamp = timestamp;
    entry.pid = Os::getProcessId();
    entry.tid = static_cast<uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFFF);
    entry.duration = duration;
    entry.hasDuration = hasDuration;
    entry.valid.store(true, std::memory_order_release);
  }

  void flush() {
    if (enabled_.load(std::memory_order_relaxed)) {
      flushCV_.notify_all();
    }
  }

  void flushInCurrentThread() {
    if (enabled_.load(std::memory_order_relaxed)) {
      flushPending();
    }
  }

  void FlushOnCrash() noexcept {
    if (!enabled_.load(std::memory_order_relaxed)) {
      return;
    }

    try {
      flushPending();
    } catch (...) {
    }
  }

 private:
  static constexpr size_t kBufferSize = 16 * 1024;//!< Circular buffer size
  static constexpr size_t kFlushIntervalMs = 1;  //!< Flush interval in milliseconds

  std::vector<LogEntry> buffer_;           //!< Circular buffer of log entries
  std::atomic<size_t> writeIndex_{0};      //!< Write position in circular buffer
  std::atomic<size_t> readIndex_{0};       //!< Read position in circular buffer
  std::atomic<bool> running_{false};       //!< Worker thread running flag
  std::atomic<bool> enabled_{false};       //!< Async logging enabled flag

  std::thread workerThread_;               //!< Background worker thread for flushing
  std::mutex flushMutex_;                  //!< Mutex for flush condition variable
  std::condition_variable flushCV_;        //!< Condition variable for worker wakeup

  void workerLoop() {
    while (running_.load(std::memory_order_relaxed)) {
      std::unique_lock<std::mutex> lock(flushMutex_);
      flushCV_.wait_for(lock, std::chrono::milliseconds(kFlushIntervalMs),
                       [this] { return !running_.load(std::memory_order_relaxed); });
      
      if (running_.load(std::memory_order_relaxed)) {
        flushPending();
      }
    }
    // Final flush on shutdown
    flushPending();
  }

  void flushPending() {
    TruncateLogFileFlushPath();
    
    size_t currentRead = readIndex_.load(std::memory_order_acquire);
    size_t currentWrite = writeIndex_.load(std::memory_order_acquire);
    size_t writeCount = 0;

    while (currentRead != currentWrite) {
      LogEntry& entry = buffer_[currentRead % kBufferSize];
      
      // Wait for valid flag
      while (!entry.valid.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      
      writeToFile(entry);
      entry.valid.store(false, std::memory_order_release);
      
      currentRead++;
      readIndex_.store(currentRead, std::memory_order_release);
      writeCount++;
      if (writeCount % 1024 == 0) {
        fflush(outFile);
      }
      currentWrite = writeIndex_.load(std::memory_order_acquire);
    }
    fflush(outFile);
  }

  void writeToFile(const LogEntry& entry) {
    char pidtid[64] = "";
    if (AMD_LOG_LEVEL >= 4) {
      snprintf(pidtid, sizeof(pidtid), "[pid:%u tid: 0x%05x]", entry.pid, entry.tid);
    }

    if (entry.hasDuration) {
      fprintf(outFile, ":%d:%-25s:%-4d: %010" PRIu64 " us: %s %s: duration: %" PRIu64 " us\n",
              entry.level, entry.file, entry.line, entry.timestamp,
              pidtid, entry.message.c_str(), entry.duration);
    } else {
      fprintf(outFile, ":%d:%-25s:%-4d: %010" PRIu64 " us: %s %s\n",
              entry.level, entry.file, entry.line, entry.timestamp,
              pidtid, entry.message.c_str());
    }
  }
};

// ================================================================================================
static AsyncLogger& getAsyncLogger();

static void crashFlushCallback() {
  getAsyncLogger().FlushOnCrash();
}

static AsyncLogger& getAsyncLogger() {
  static AsyncLogger instance;
  return instance;
}

// ================================================================================================
static void TruncateLogFileFlushPath() {
  if (outFile == stderr) {
    return;
  }

  const size_t maxLogSize = AMD_LOG_LEVEL_SIZE * Mi;

  fflush(outFile);
  if (fseek(outFile, 0, SEEK_END) != 0) {
    return;
  }

  long size = ftell(outFile);
  if (size < 0 || static_cast<size_t>(size) <= maxLogSize) {
    return;
  }

#ifdef _WIN32
  int fd = _fileno(outFile);
  if (fd < 0 || _chsize_s(fd, 0) != 0) {
#else
  int fd = fileno(outFile);
  if (fd < 0 || ftruncate(fd, 0) != 0) {
#endif
    return;
  }

  fseek(outFile, 0, SEEK_SET);
}

// ================================================================================================
void truncate_log_file() {
  if (outFile != stderr) {
    fseek(outFile, 0, SEEK_END);
    long size = ftell(outFile);

    const size_t maxLogSize = AMD_LOG_LEVEL_SIZE * Mi;
    if (size > maxLogSize) {
      if (nullptr == freopen(NULL, "w", outFile)) {
        outFile = stderr;
      }
    }
  }
}

// ================================================================================================
void report_warning(const char* message) {
  truncate_log_file();
  fprintf(outFile, "Warning: %s\n", message);
}

// ================================================================================================
void log_entry(LogLevel level, const char* file, int line, const char* message) {
  if (level == LOG_NONE) {
    return;
  }
  truncate_log_file();
  fprintf(outFile, ":%d:%s:%d: %s\n", level, file, line, message);
  fflush(outFile);
}

// ================================================================================================
void log_timestamped(LogLevel level, const char* file, int line, const char* message) {
  static bool gotstart = false;  // not thread-safe, but not scary if fails
  static uint64_t start;

  if (!gotstart) {
    start = Os::timeNanos();
    gotstart = true;
  }

  uint64_t time = Os::timeNanos() - start;
  if (level == LOG_NONE) {
    return;
  }

  truncate_log_file();
  fprintf(outFile, ":% 2d:%15s:% 5d: (%010lld) us %s\n", level, file, line, time / 1000ULL,
          message);
  fflush(outFile);
}

// ================================================================================================
void log_printf(LogLevel level, const char* file, int line, const char* format, ...) {
  va_list ap;
  char message[4096];
  va_start(ap, format);
  vsnprintf(message, sizeof(message), format, ap);
  va_end(ap);
  uint64_t timeUs = Os::timeNanos() / 1000ULL;

  // Try async logging first
  if (getAsyncLogger().isEnabled()) {
    getAsyncLogger().log(level, file, line, message, timeUs);
    return;
  }

  // Fall back to sync logging
  char pidtid[64] = "";
  if (AMD_LOG_LEVEL >= 4) {
    size_t tid_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
    snprintf(pidtid, sizeof(pidtid), "[pid:%u tid: 0x%05zx]",
             Os::getProcessId(), tid_hash & 0xFFFFF);
  }

  truncate_log_file();

  fprintf(outFile, ":%d:%-25s:%-4d: %010" PRIu64 " us: %s %s\n", level, file, line, timeUs,
          pidtid, message);

  fflush(outFile);
}

// ================================================================================================
void log_printf(LogLevel level, const char* file, int line, uint64_t* start, const char* format,
                ...) {
  va_list ap;
  char message[4096];
  va_start(ap, format);
  vsnprintf(message, sizeof(message), format, ap);
  va_end(ap);
  uint64_t timeUs = Os::timeNanos() / 1000ULL;

  bool isStartLog = (start == 0 || *start == 0);
  uint64_t duration = isStartLog ? 0 : (timeUs - *start);

  // Try async logging first
  if (getAsyncLogger().isEnabled()) {
    getAsyncLogger().log(level, file, line, message, timeUs, duration, !isStartLog);
    if (start != 0 && *start == 0) {
      *start = timeUs;
    }
    return;
  }

  // Fall back to sync logging
  char pidtid[64] = "";
  if (AMD_LOG_LEVEL >= 4) {
    size_t tid_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
    snprintf(pidtid, sizeof(pidtid), "[pid:%u tid: 0x%05zx]",
             Os::getProcessId(), tid_hash & 0xFFFFF);
  }

  truncate_log_file();

  if (isStartLog) {
    fprintf(outFile, ":%d:%-25s:%-4d: %010" PRIu64 " us: %s %s\n", level, file, line, timeUs,
            pidtid, message);
  } else {
    fprintf(outFile, ":%d:%-25s:%-4d: %010" PRIu64 " us: %s %s: duration: %" PRIu64 " us\n", level,
            file, line, timeUs, pidtid, message, duration);
  }
  fflush(outFile);
  if (start != 0 && *start == 0) {
    *start = timeUs;
  }
}

// ================================================================================================
void EnableAsyncLogging(bool enable) {
  getAsyncLogger().enable(enable);
}

// ================================================================================================
bool IsAsyncLoggingEnabled() {
  return getAsyncLogger().isEnabled();
}

// ================================================================================================
void FlushAsyncLogs() {
  getAsyncLogger().flush();
}

// ================================================================================================
void FlushAsyncLogsInCurrentThread() {
  getAsyncLogger().flushInCurrentThread();
}

}  // namespace amd
