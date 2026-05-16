////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/util/rocr_logging.h"
#include "core/util/os.h"

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <inttypes.h>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <process.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#endif

namespace rocr {

// Global logging state instance
RocrLoggingState g_rocr_log_state;

// Global async logger instance
RocrAsyncLogger g_rocr_async_logger;

// Flag to track if crash handlers are installed
static std::atomic<bool> crash_handlers_installed{false};

// Forward declaration of crash flush callback
static void CrashFlushCallback(int sig);

// ============================================================================
// Async Logger Implementation
// ============================================================================

RocrAsyncLogger::RocrAsyncLogger() : buffer_(kBufferSize) {}

RocrAsyncLogger::~RocrAsyncLogger() {
  Enable(false);
  Stop();
}

void RocrAsyncLogger::Start() {
  if (!running_.load(std::memory_order_relaxed)) {
    running_.store(true, std::memory_order_relaxed);
    enabled_.store(true, std::memory_order_relaxed);
    worker_thread_ = std::thread(&RocrAsyncLogger::WorkerLoop, this);
  }
}

void RocrAsyncLogger::Stop() {
  if (running_.load(std::memory_order_relaxed)) {
    running_.store(false, std::memory_order_relaxed);
    flush_cv_.notify_all();
    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
  }
}

void RocrAsyncLogger::Enable(bool enable) {
  enabled_.store(enable, std::memory_order_relaxed);
  if (enable && !running_.load(std::memory_order_relaxed)) {
    Start();
  }

#ifndef _WIN32
  if (enable) {
    bool expected = false;
    if (crash_handlers_installed.compare_exchange_strong(expected, true,
                                                         std::memory_order_acq_rel)) {
      // Install signal handlers for crash flushing
      struct sigaction sa;
      sa.sa_handler = CrashFlushCallback;
      sigemptyset(&sa.sa_mask);
      sa.sa_flags = SA_RESETHAND;  // Reset to default after handling

      sigaction(SIGSEGV, &sa, nullptr);
      sigaction(SIGABRT, &sa, nullptr);
      sigaction(SIGBUS, &sa, nullptr);
      sigaction(SIGFPE, &sa, nullptr);
    }
  }
#endif
}

void RocrAsyncLogger::Log(RocrLogLevel level, const char* file, int line,
                          const char* message, uint64_t timestamp,
                          uint64_t duration, bool has_duration) {
  if (!enabled_.load(std::memory_order_relaxed)) {
    return;  // Fall back to sync logging
  }

  size_t current_write = write_index_.fetch_add(1, std::memory_order_release);
  size_t current_read = read_index_.load(std::memory_order_acquire);

  // Check if buffer is full
  while (current_write - current_read >= kBufferSize) {
    Flush();
    current_read = read_index_.load(std::memory_order_acquire);
  }

  // Write to buffer (lock-free)
  RocrLogEntry& entry = buffer_[current_write % kBufferSize];
  entry.level = level;
  entry.file = file ? file : "";
  entry.line = line;
  if (message) {
    strncpy(entry.message, message, sizeof(entry.message) - 1);
    entry.message[sizeof(entry.message) - 1] = '\0';
  } else {
    entry.message[0] = '\0';
  }
  entry.timestamp = timestamp;
  entry.pid = static_cast<uint32_t>(os::GetProcessId());
  entry.tid = static_cast<uint32_t>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFFF);
  entry.duration = duration;
  entry.has_duration = has_duration;
  entry.valid.store(true, std::memory_order_release);
}

void RocrAsyncLogger::Flush() {
  if (enabled_.load(std::memory_order_relaxed)) {
    flush_cv_.notify_all();
  }
}

void RocrAsyncLogger::FlushInCurrentThread() {
  if (enabled_.load(std::memory_order_relaxed)) {
    FlushPending();
  }
}

void RocrAsyncLogger::FlushOnCrash() noexcept {
  // SIGNAL SAFETY: This function may be called from a signal handler.
  // Only use async-signal-safe functions: write(), _exit(), signal constants.
  // Do NOT use: fprintf, fflush, malloc, new, mutex, atomics with ordering.

  if (!enabled_.load(std::memory_order_relaxed)) {
    return;
  }

  // Get file descriptor for direct write (async-signal-safe)
  int fd = -1;
  if (g_rocr_log_state.log_file != nullptr && g_rocr_log_state.log_file != stderr) {
    fd = fileno(g_rocr_log_state.log_file);
  } else {
    fd = STDERR_FILENO;
  }

  if (fd < 0) return;

  // Write a simple crash marker using only write() (async-signal-safe)
  static const char crash_msg[] = "\n[rocr] CRASH: Async log buffer may contain unwritten entries\n";
  // Ignore return value - nothing we can do in signal handler
  (void)write(fd, crash_msg, sizeof(crash_msg) - 1);

  // Note: We intentionally do NOT call FlushPending() here because it uses
  // fprintf, fflush, std::mutex, and other non-async-signal-safe functions.
  // The pending log entries will be lost, but this is safer than undefined behavior.
}

void RocrAsyncLogger::WorkerLoop() {
  while (running_.load(std::memory_order_relaxed)) {
    std::unique_lock<std::mutex> lock(flush_mutex_);
    flush_cv_.wait_for(lock, std::chrono::milliseconds(kFlushIntervalMs),
                       [this] { return !running_.load(std::memory_order_relaxed); });

    if (running_.load(std::memory_order_relaxed)) {
      FlushPending();
    }
  }
  // Final flush on shutdown
  FlushPending();
}

void RocrAsyncLogger::FlushPending() {
  // Check for log file truncation
  if (g_rocr_log_state.log_file != stderr && g_rocr_log_state.log_max_size_mb > 0) {
    const size_t max_log_size = g_rocr_log_state.log_max_size_mb * 1024 * 1024;

    fflush(g_rocr_log_state.log_file);
    if (fseek(g_rocr_log_state.log_file, 0, SEEK_END) == 0) {
      long size = ftell(g_rocr_log_state.log_file);
      if (size > 0 && static_cast<size_t>(size) > max_log_size) {
#ifdef _WIN32
        int fd = _fileno(g_rocr_log_state.log_file);
        if (fd >= 0) {
          _chsize_s(fd, 0);
        }
#else
        int fd = fileno(g_rocr_log_state.log_file);
        if (fd >= 0) {
          ftruncate(fd, 0);
        }
#endif
        // Must fflush after ftruncate to sync stdio buffer with fd state
        fflush(g_rocr_log_state.log_file);
        fseek(g_rocr_log_state.log_file, 0, SEEK_SET);
      }
    }
  }

  size_t current_read = read_index_.load(std::memory_order_acquire);
  size_t current_write = write_index_.load(std::memory_order_acquire);
  size_t write_count = 0;

  while (current_read != current_write) {
    RocrLogEntry& entry = buffer_[current_read % kBufferSize];

    // Wait for valid flag with timeout to prevent infinite spin
    auto spin_start = std::chrono::steady_clock::now();
    while (!entry.valid.load(std::memory_order_acquire)) {
      std::this_thread::yield();
      // Timeout after 5 seconds - writer may have crashed
      if (std::chrono::steady_clock::now() - spin_start > std::chrono::seconds(5)) {
        // Skip this corrupted entry and continue
        entry.valid.store(false, std::memory_order_release);
        current_read++;
        read_index_.store(current_read, std::memory_order_release);
        continue;
      }
    }

    WriteToFile(entry);
    entry.valid.store(false, std::memory_order_release);

    current_read++;
    read_index_.store(current_read, std::memory_order_release);
    write_count++;

    // Periodic flush to avoid buffering too much
    if (write_count % 1024 == 0) {
      fflush(g_rocr_log_state.log_file);
    }
    current_write = write_index_.load(std::memory_order_acquire);
  }
  fflush(g_rocr_log_state.log_file);
}

void RocrAsyncLogger::WriteToFile(const RocrLogEntry& entry) {
  char pidtid[64] = "";
  if (g_rocr_log_state.log_level >= ROCR_LOG_DEBUG) {
    snprintf(pidtid, sizeof(pidtid), "[pid:%u tid:0x%05x]", entry.pid, entry.tid);
  }

  if (entry.has_duration) {
    fprintf(g_rocr_log_state.log_file,
            ":%d:%-25s:%-4d: %010" PRIu64 " us: %s [rocr] %s: duration: %" PRIu64 " us\n",
            entry.level, entry.file, entry.line, entry.timestamp,
            pidtid, entry.message, entry.duration);
  } else {
    fprintf(g_rocr_log_state.log_file,
            ":%d:%-25s:%-4d: %010" PRIu64 " us: %s [rocr] %s\n",
            entry.level, entry.file, entry.line, entry.timestamp,
            pidtid, entry.message);
  }
}

// ============================================================================
// Crash Handler
// ============================================================================

static void CrashFlushCallback(int sig) {
  g_rocr_async_logger.FlushOnCrash();

  // Re-raise signal to get default behavior (core dump, etc.)
#ifndef _WIN32
  signal(sig, SIG_DFL);
  raise(sig);
#endif
}

// ============================================================================
// Core Logging Functions
// ============================================================================

uint64_t rocr_get_timestamp_us() {
  return os::ReadAccurateClock() / 1000ULL;
}

// Print help message when HSA_LOG_LEVEL=help
static void rocr_log_print_help() {
  fprintf(stderr,
    "ROCR Runtime Logging System\n"
    "===========================\n"
    "\n"
    "Environment Variables:\n"
    "  HSA_LOG_LEVEL   0-6 or 'help'  Log verbosity level\n"
    "  HSA_LOG_MASK    hex            Category bitmask (default 0x7FFFFFFF)\n"
    "  HSA_LOG_FILE    path           Output file (PID appended)\n"
    "  HSA_LOG_SIZE    MB             Max file size (default 2048)\n"
    "  HSA_LOG_ASYNC   0/1            Async ring buffer mode\n"
    "  HSA_LOG_FORMAT  json           Structured JSON output\n"
    "\n"
    "Log Levels:\n"
    "  0  NONE     Disabled (default)\n"
    "  1  ERROR    Critical errors\n"
    "  2  WARNING  Warnings + errors\n"
    "  3  INFO     General info\n"
    "  4  DEBUG    Detailed debug\n"
    "  5  TRACE    Very detailed\n"
    "  6  VERBOSE  Function entry/exit\n"
    "\n"
    "Category Masks (OR together):\n"
    "  0x1    INIT       0x2    QUEUE      0x4    MEM\n"
    "  0x8    SIGNAL     0x10   IPC        0x20   AGENT\n"
    "  0x40   AQL        0x80   SDMA       0x100  COPY\n"
    "  0x200  BLIT       0x400  SCRATCH    0x800  POOL\n"
    "  0x2000 FAULT      0x8000 EXCEPT     0x10000 WAIT\n"
    "\n"
    "Examples:\n"
    "  HSA_LOG_LEVEL=1 ./app                    # Errors only\n"
    "  HSA_LOG_LEVEL=4 HSA_LOG_MASK=0x4 ./app   # Debug memory\n"
    "  HSA_LOG_LEVEL=5 HSA_LOG_FILE=/tmp/r ./app # Trace to file\n"
    "\n");
}

void rocr_log_init() {
  if (g_rocr_log_state.initialized) {
    return;
  }

  // Parse HSA_LOG_LEVEL - check for "help" first
  std::string var = os::GetEnvVar("HSA_LOG_LEVEL");
  if (var == "help" || var == "HELP" || var == "?") {
    rocr_log_print_help();
    g_rocr_log_state.log_level = 0;  // Don't enable logging after help
  } else if (!var.empty()) {
    g_rocr_log_state.log_level = atoi(var.c_str());
  }

  // Parse HSA_LOG_MASK
  var = os::GetEnvVar("HSA_LOG_MASK");
  if (!var.empty()) {
    g_rocr_log_state.log_mask = strtoull(var.c_str(), nullptr, 0);
  }

  // Parse HSA_LOG_FILE
  g_rocr_log_state.log_file_path = os::GetEnvVar("HSA_LOG_FILE");
  if (!g_rocr_log_state.log_file_path.empty()) {
    // Append PID to filename
    std::string filename = g_rocr_log_state.log_file_path;
    filename += ".";
    filename += std::to_string(os::GetProcessId());
    filename += ".log";

    g_rocr_log_state.log_file = fopen(filename.c_str(), "w");
    if (!g_rocr_log_state.log_file) {
      fprintf(stderr, "rocr: Failed to open log file '%s', falling back to stderr\n",
              filename.c_str());
      g_rocr_log_state.log_file = stderr;
    }
  }

  // Parse HSA_LOG_SIZE
  var = os::GetEnvVar("HSA_LOG_SIZE");
  if (!var.empty()) {
    g_rocr_log_state.log_max_size_mb = strtoul(var.c_str(), nullptr, 10);
  }

  // Parse HSA_LOG_ASYNC
  var = os::GetEnvVar("HSA_LOG_ASYNC");
  g_rocr_log_state.async_enabled = (var == "1");

  // Parse HSA_LOG_FORMAT (json for structured output)
  var = os::GetEnvVar("HSA_LOG_FORMAT");
  g_rocr_log_state.structured_output = (var == "json" || var == "JSON");

  // Enable async logger if requested
  if (g_rocr_log_state.async_enabled && g_rocr_log_state.log_level > 0) {
    g_rocr_async_logger.Enable(true);
  }

  g_rocr_log_state.initialized = true;

  // Log initialization if logging is enabled
  if (g_rocr_log_state.log_level > 0) {
    RocrLogInfo(ROCR_LOG_INIT, "ROCR logging initialized: level=%d mask=0x%llx async=%d",
                g_rocr_log_state.log_level,
                (unsigned long long)g_rocr_log_state.log_mask,
                g_rocr_log_state.async_enabled ? 1 : 0);
  }
}

void rocr_log_shutdown() {
  if (!g_rocr_log_state.initialized) {
    return;
  }

  // Log shutdown if logging is enabled
  if (g_rocr_log_state.log_level > 0) {
    RocrLogInfo(ROCR_LOG_INIT, "ROCR logging shutting down");
  }

  // Stop async logger
  if (g_rocr_log_state.async_enabled) {
    g_rocr_async_logger.FlushInCurrentThread();
    g_rocr_async_logger.Enable(false);
    g_rocr_async_logger.Stop();
  }

  // Close log file if it's not stderr
  if (g_rocr_log_state.log_file != stderr && g_rocr_log_state.log_file != nullptr) {
    fflush(g_rocr_log_state.log_file);
    fclose(g_rocr_log_state.log_file);
    g_rocr_log_state.log_file = stderr;
  }

  g_rocr_log_state.initialized = false;
}

// Escape special characters for JSON output to prevent injection
static std::string json_escape(const char* str) {
  std::string result;
  result.reserve(strlen(str));
  for (const char* p = str; *p; ++p) {
    switch (*p) {
      case '"':  result += "\\\""; break;
      case '\\': result += "\\\\"; break;
      case '\b': result += "\\b";  break;
      case '\f': result += "\\f";  break;
      case '\n': result += "\\n";  break;
      case '\r': result += "\\r";  break;
      case '\t': result += "\\t";  break;
      default:   result += *p;     break;
    }
  }
  return result;
}

void rocr_log_printf(RocrLogLevel level, uint64_t mask, const char* file,
                     int line, const char* format, ...) {
  va_list ap;
  va_start(ap, format);

  char message[4096];
  vsnprintf(message, sizeof(message), format, ap);
  va_end(ap);

  uint64_t timestamp = rocr_get_timestamp_us();

  // Use async logging if enabled
  if (g_rocr_log_state.async_enabled && g_rocr_async_logger.IsEnabled()) {
    g_rocr_async_logger.Log(level, file, line, message, timestamp);
    return;
  }

  // Synchronous logging
  std::lock_guard<std::mutex> lock(g_rocr_log_state.file_mutex);

  uint32_t pid = static_cast<uint32_t>(os::GetProcessId());
  uint32_t tid = static_cast<uint32_t>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFFF);

  if (g_rocr_log_state.structured_output) {
    // JSON structured output for analytics - escape special characters
    std::string escaped_msg = json_escape(message);
    std::string escaped_file = json_escape(file);
    fprintf(g_rocr_log_state.log_file,
            "{\"level\":%d,\"category\":\"0x%llx\",\"file\":\"%s\",\"line\":%d,"
            "\"timestamp\":%" PRIu64 ",\"pid\":%u,\"tid\":\"0x%x\",\"msg\":\"%s\"}\n",
            level, (unsigned long long)mask, escaped_file.c_str(), line, timestamp,
            pid, tid, escaped_msg.c_str());
  } else {
    char pidtid[64] = "";
    if (g_rocr_log_state.log_level >= ROCR_LOG_DEBUG) {
      snprintf(pidtid, sizeof(pidtid), "[pid:%u tid:0x%05x]", pid, tid);
    }

    fprintf(g_rocr_log_state.log_file,
            ":%d:%-25s:%-4d: %010" PRIu64 " us: %s [rocr] %s\n",
            level, file, line, timestamp, pidtid, message);
  }
  fflush(g_rocr_log_state.log_file);
}

void rocr_log_printf_duration(RocrLogLevel level, uint64_t mask, const char* file,
                              int line, uint64_t* start_time, const char* format, ...) {
  va_list ap;
  va_start(ap, format);

  char message[4096];
  vsnprintf(message, sizeof(message), format, ap);
  va_end(ap);

  uint64_t timestamp = rocr_get_timestamp_us();
  uint64_t duration = timestamp - *start_time;

  // Use async logging if enabled
  if (g_rocr_log_state.async_enabled && g_rocr_async_logger.IsEnabled()) {
    g_rocr_async_logger.Log(level, file, line, message, timestamp, duration, true);
    return;
  }

  // Synchronous logging
  std::lock_guard<std::mutex> lock(g_rocr_log_state.file_mutex);

  uint32_t pid = static_cast<uint32_t>(os::GetProcessId());
  uint32_t tid = static_cast<uint32_t>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFFF);

  if (g_rocr_log_state.structured_output) {
    // JSON structured output for analytics - escape special characters
    std::string escaped_msg = json_escape(message);
    std::string escaped_file = json_escape(file);
    fprintf(g_rocr_log_state.log_file,
            "{\"level\":%d,\"category\":\"0x%llx\",\"file\":\"%s\",\"line\":%d,"
            "\"timestamp\":%" PRIu64 ",\"pid\":%u,\"tid\":\"0x%x\","
            "\"duration_us\":%" PRIu64 ",\"msg\":\"%s\"}\n",
            level, (unsigned long long)mask, escaped_file.c_str(), line, timestamp,
            pid, tid, duration, escaped_msg.c_str());
  } else {
    char pidtid[64] = "";
    if (g_rocr_log_state.log_level >= ROCR_LOG_DEBUG) {
      snprintf(pidtid, sizeof(pidtid), "[pid:%u tid:0x%05x]", pid, tid);
    }

    fprintf(g_rocr_log_state.log_file,
            ":%d:%-25s:%-4d: %010" PRIu64 " us: %s [rocr] %s: duration: %" PRIu64 " us\n",
            level, file, line, timestamp, pidtid, message, duration);
  }
  fflush(g_rocr_log_state.log_file);
}

// ============================================================================
// HSA Status Name Helper
// ============================================================================

const char* rocr_hsa_status_name(uint32_t status) {
  switch (status) {
    case 0x0: return "HSA_STATUS_SUCCESS";
    case 0x1: return "HSA_STATUS_INFO_BREAK";
    case 0x1000: return "HSA_STATUS_ERROR";
    case 0x1001: return "HSA_STATUS_ERROR_INVALID_ARGUMENT";
    case 0x1002: return "HSA_STATUS_ERROR_INVALID_QUEUE_CREATION";
    case 0x1003: return "HSA_STATUS_ERROR_INVALID_ALLOCATION";
    case 0x1004: return "HSA_STATUS_ERROR_INVALID_AGENT";
    case 0x1005: return "HSA_STATUS_ERROR_INVALID_REGION";
    case 0x1006: return "HSA_STATUS_ERROR_INVALID_SIGNAL";
    case 0x1007: return "HSA_STATUS_ERROR_INVALID_QUEUE";
    case 0x1008: return "HSA_STATUS_ERROR_OUT_OF_RESOURCES";
    case 0x1009: return "HSA_STATUS_ERROR_INVALID_PACKET_FORMAT";
    case 0x100A: return "HSA_STATUS_ERROR_RESOURCE_FREE";
    case 0x100B: return "HSA_STATUS_ERROR_NOT_INITIALIZED";
    case 0x100C: return "HSA_STATUS_ERROR_REFCOUNT_OVERFLOW";
    case 0x100D: return "HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS";
    case 0x100E: return "HSA_STATUS_ERROR_INVALID_INDEX";
    case 0x100F: return "HSA_STATUS_ERROR_INVALID_ISA";
    case 0x1010: return "HSA_STATUS_ERROR_INVALID_ISA_NAME";
    case 0x1011: return "HSA_STATUS_ERROR_INVALID_CODE_OBJECT";
    case 0x1012: return "HSA_STATUS_ERROR_INVALID_EXECUTABLE";
    case 0x1013: return "HSA_STATUS_ERROR_FROZEN_EXECUTABLE";
    case 0x1014: return "HSA_STATUS_ERROR_INVALID_SYMBOL_NAME";
    case 0x1015: return "HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED";
    case 0x1016: return "HSA_STATUS_ERROR_VARIABLE_UNDEFINED";
    case 0x1017: return "HSA_STATUS_ERROR_EXCEPTION";
    case 0x1018: return "HSA_STATUS_ERROR_INVALID_CODE_SYMBOL";
    case 0x1019: return "HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL";
    case 0x101A: return "HSA_STATUS_ERROR_INVALID_FILE";
    case 0x101B: return "HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER";
    case 0x101C: return "HSA_STATUS_ERROR_INVALID_CACHE";
    case 0x101D: return "HSA_STATUS_ERROR_INVALID_WAVEFRONT";
    case 0x101E: return "HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP";
    case 0x101F: return "HSA_STATUS_ERROR_INVALID_RUNTIME_STATE";
    case 0x1020: return "HSA_STATUS_ERROR_FATAL";
    default: return "HSA_STATUS_UNKNOWN";
  }
}

}  // namespace rocr
