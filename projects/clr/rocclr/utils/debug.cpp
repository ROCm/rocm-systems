/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

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
#endif  // _WIN32

namespace amd {

FILE* outFile = stderr;

void truncate_log_file();

// ================================================================================================
// Async logging infrastructure
// ================================================================================================

struct LogEntry {
  LogLevel level;          //!< Log severity level
  std::string file;        //!< Source file name
  int line;                //!< Source line number
  std::string message;     //!< Formatted log message
  uint64_t timestamp;      //!< Timestamp in microseconds
  uint32_t pid;            //!< Process ID
  std::thread::id tid;     //!< Thread ID
  uint64_t duration;       //!< Duration in microseconds (0 if not a duration log)
  bool hasDuration;        //!< True if this is a duration log entry

  LogEntry() : level(LOG_NONE), line(0), timestamp(0), pid(0), duration(0), hasDuration(false) {}
};

class AsyncLogger {
 public:
  AsyncLogger() : buffer_(kBufferSize) {}

  ~AsyncLogger() {
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
  }

  bool isEnabled() const {
    return enabled_.load(std::memory_order_relaxed);
  }

  void log(LogLevel level, const char* file, int line, const char* message,
           uint64_t timestamp, uint64_t duration = 0, bool hasDuration = false) {
    if (!enabled_.load(std::memory_order_relaxed)) {
      return;  // Fall back to sync logging
    }

    size_t currentWrite = writeIndex_.load(std::memory_order_relaxed);
    size_t currentRead = readIndex_.load(std::memory_order_acquire);

    // Check if buffer is full
    if (currentWrite - currentRead >= kBufferSize) {
      // Buffer full - force a sync flush or drop (we'll drop oldest for now)
      readIndex_.fetch_add(1, std::memory_order_release);
    }

    // Write to buffer (lock-free)
    LogEntry& entry = buffer_[currentWrite % kBufferSize];
    entry.level = level;
    entry.file = file ? file : "";
    entry.line = line;
    entry.message = message ? message : "";
    entry.timestamp = timestamp;
    entry.pid = Os::getProcessId();
    entry.tid = std::this_thread::get_id();
    entry.duration = duration;
    entry.hasDuration = hasDuration;

    writeIndex_.fetch_add(1, std::memory_order_release);
  }

  void flush() {
    if (enabled_.load(std::memory_order_relaxed)) {
      flushCV_.notify_all();
      // Give worker thread a moment to flush
      std::this_thread::sleep_for(std::chrono::milliseconds(kFlushIntervalMs + 10));
    }
  }

  void flushInCurrentThread() {
    if (enabled_.load(std::memory_order_relaxed)) {
      flushPending();
    }
  }

 private:
  static constexpr size_t kBufferSize = 8192;  //!< Circular buffer size
  static constexpr size_t kFlushIntervalMs = 100;  //!< Flush interval in milliseconds

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
    size_t currentRead = readIndex_.load(std::memory_order_acquire);
    size_t currentWrite = writeIndex_.load(std::memory_order_acquire);

    while (currentRead != currentWrite) {
      const LogEntry& entry = buffer_[currentRead % kBufferSize];
      writeToFile(entry);
      currentRead++;
      readIndex_.store(currentRead, std::memory_order_release);
      currentWrite = writeIndex_.load(std::memory_order_acquire);
    }
  }

  void writeToFile(const LogEntry& entry) {
    truncate_log_file();

    std::stringstream pidtid;
    if (AMD_LOG_LEVEL >= 4) {
      pidtid << "[pid:" << entry.pid << " tid: 0x";
      pidtid << std::hex << std::setw(5) << entry.tid << "]";
    }

    if (entry.hasDuration) {
      fprintf(outFile, ":%d:%-25s:%-4d: %010" PRIu64 " us: %s %s: duration: %" PRIu64 " us\n",
              entry.level, entry.file.c_str(), entry.line, entry.timestamp,
              pidtid.str().c_str(), entry.message.c_str(), entry.duration);
    } else {
      fprintf(outFile, ":%d:%-25s:%-4d: %010" PRIu64 " us: %s %s\n",
              entry.level, entry.file.c_str(), entry.line, entry.timestamp,
              pidtid.str().c_str(), entry.message.c_str());
    }
    fflush(outFile);
  }
};

static std::unique_ptr<AsyncLogger> g_asyncLogger;
static std::once_flag g_asyncLoggerInitFlag;

static AsyncLogger& getAsyncLogger() {
  std::call_once(g_asyncLoggerInitFlag, []() {
    g_asyncLogger = std::make_unique<AsyncLogger>();
  });
  return *g_asyncLogger;
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
  std::stringstream pidtid;
  if (AMD_LOG_LEVEL >= 4) {
    pidtid << "[pid:" << Os::getProcessId() << " tid: 0x";
    pidtid << std::hex << std::setw(5) << std::this_thread::get_id() << "]";
  }

  truncate_log_file();

  fprintf(outFile, ":%d:%-25s:%-4d: %010" PRIu64 " us: %s %s\n", level, file, line, timeUs,
          pidtid.str().c_str(), message);

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
  std::stringstream pidtid;
  if (AMD_LOG_LEVEL >= 4) {
    pidtid << "[pid:" << Os::getProcessId() << " tid: 0x";
    pidtid << std::hex << std::setw(5) << std::this_thread::get_id() << "]";
  }

  truncate_log_file();

  if (isStartLog) {
    fprintf(outFile, ":%d:%-25s:%-4d: %010" PRIu64 " us: %s %s\n", level, file, line, timeUs,
            pidtid.str().c_str(), message);
  } else {
    fprintf(outFile, ":%d:%-25s:%-4d: %010" PRIu64 " us: %s %s: duration: %" PRIu64 " us\n", level,
            file, line, timeUs, pidtid.str().c_str(), message, duration);
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
