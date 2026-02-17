/* Copyright (c) 2026 Advanced Micro Devices, Inc.

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
#include "platform/runtime.hpp"

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
#ifdef _WIN32
#include <windows.h>
#endif  // _WIN32

// Include async logging infrastructure
#include "utils/async_logger.hpp"

namespace amd {

FILE* outFile = stderr;

// Maximum log file size (used by both sync and async logging)
// Note: 'extern' is required to give const variable external linkage in C++
extern const size_t maxLogSize = AMD_LOG_LEVEL_SIZE * Mi;

// Async logging state
static std::atomic<bool> async_logging_enabled_(false);
static std::once_flag async_logging_init_flag_;

// ================================================================================================
void init_async_logging_once() {
  logging::AsyncLoggerConfig config;

  // Read configuration from flags
  config.enabled = AMD_LOG_INMEM;
  config.buffer_size = AMD_LOG_BUFFER_SIZE * Mi;
  config.flush_interval_ms = AMD_LOG_FLUSH_INTERVAL_MS;
  config.log_file_path = flagIsDefault(AMD_LOG_LEVEL_FILE) ? nullptr : AMD_LOG_LEVEL_FILE;

  if (config.enabled) {
    // Initialize async logger
    logging::AsyncLogger::GetInstance().Initialize(config);
    async_logging_enabled_.store(true, std::memory_order_release);

    // Register shutdown hook
    RuntimeTearDown::RegisterTearDownCallback("async_logging_shutdown", []() {
      shutdown_async_logging();
    });
  }
}

// ================================================================================================
void init_async_logging() {
  std::call_once(async_logging_init_flag_, init_async_logging_once);
}

// ================================================================================================
bool is_async_logging_enabled() {
  // Rely on std::call_once fast path (just atomic check after first call)
  init_async_logging();
  return async_logging_enabled_.load(std::memory_order_acquire);
}

// ================================================================================================
void shutdown_async_logging() {
  if (async_logging_enabled_.load(std::memory_order_acquire)) {
    logging::AsyncLogger::GetInstance().Shutdown();
    async_logging_enabled_.store(false, std::memory_order_release);
  }
}

// ================================================================================================
void flush_async_logs() {
  if (async_logging_enabled_.load(std::memory_order_acquire)) {
    logging::AsyncLogger::GetInstance().FlushSync();
  }
}

// ================================================================================================
void async_log_printf_impl(uint16_t level, const char* file,
                          int line, const char* format, ...) {
  va_list args;
  va_start(args, format);
  logging::AsyncLogger::GetInstance().WriteLog(level, file, line, format, args);
  va_end(args);
}

// ================================================================================================
void truncate_log_file() {
  if (outFile != stderr) {
    fseek(outFile, 0, SEEK_END);
    long size = ftell(outFile);

    if (size > static_cast<long>(maxLogSize)) {
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
  std::stringstream pidtid;
  if (AMD_LOG_LEVEL >= 4) {
    pidtid << "[pid:" << Os::getProcessId() << " tid: 0x";
    pidtid << std::hex << std::setw(5) << std::this_thread::get_id() << "]";
  }

  va_start(ap, format);
  char message[4096];
  vsnprintf(message, sizeof(message), format, ap);
  va_end(ap);
  uint64_t timeUs = Os::timeNanos() / 1000ULL;

  truncate_log_file();

  fprintf(outFile, ":%d:%-25s:%-4d: %010" PRIu64 " us: %s %s\n", level, file, line, timeUs,
          pidtid.str().c_str(), message);

  fflush(outFile);
}

// ================================================================================================
void log_printf(LogLevel level, const char* file, int line, uint64_t* start, const char* format,
                ...) {
  // Check if async logging is enabled - route to async path if available
  if (is_async_logging_enabled()) {
    va_list ap;
    va_start(ap, format);
    char message[4096];
    vsnprintf(message, sizeof(message), format, ap);
    va_end(ap);

    uint64_t timeUs = Os::timeNanos() / 1000ULL;

    // Log asynchronously with or without duration
    if (start == nullptr || *start == 0) {
      async_log_printf_impl(level, file, line, "%s", message);
      if (start != nullptr && *start == 0) {
        *start = timeUs;
      }
    } else {
      uint64_t duration = timeUs - *start;
      async_log_printf_impl(level, file, line, "%s: duration: %" PRIu64 " us", message,
                            duration);
    }
    return;
  }

  // Synchronous logging path (when async is disabled)
  va_list ap;
  std::stringstream pidtid;
  if (AMD_LOG_LEVEL >= 4) {
    pidtid << "[pid:" << Os::getProcessId() << " tid: 0x";
    pidtid << std::hex << std::setw(5) << std::this_thread::get_id() << "]";
  }
  va_start(ap, format);
  char message[4096];
  vsnprintf(message, sizeof(message), format, ap);
  va_end(ap);
  uint64_t timeUs = Os::timeNanos() / 1000ULL;

  truncate_log_file();

  if (start == 0 || *start == 0) {
    fprintf(outFile, ":%d:%-25s:%-4d: %010" PRIu64 " us: %s %s\n", level, file, line, timeUs,
            pidtid.str().c_str(), message);
  } else {
    fprintf(outFile, ":%d:%-25s:%-4d: %010" PRIu64 " us: %s %s: duration: %" PRIu64 " us\n", level,
            file, line, timeUs, pidtid.str().c_str(), message, timeUs - *start);
  }
  fflush(outFile);
  if (*start == 0) {
    *start = timeUs;
  }
}

}  // namespace amd
