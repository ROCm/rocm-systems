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

// =============================================================================
// ROCR Runtime Comprehensive Logging System
// =============================================================================
//
// HIP-style logging with environment variable control. Disabled by default.
//
// ENVIRONMENT VARIABLES:
//   HSA_LOG_LEVEL   0-6     Log verbosity (0=none, 1=error, 2=warn, 3=info,
//                           4=debug, 5=trace, 6=verbose). Set to "help" for usage.
//   HSA_LOG_MASK    hex     Category bitmask (default 0x7FFFFFFF = all)
//   HSA_LOG_FILE    path    Output file (PID appended), empty = stderr
//   HSA_LOG_SIZE    MB      Max file size before truncation (default 2048)
//   HSA_LOG_ASYNC   0/1     Enable async ring buffer logging
//   HSA_LOG_FORMAT  string  Set to "json" for structured JSON output
//
// LOG LEVELS:
//   0 NONE     No logging (production default)
//   1 ERROR    Critical errors only
//   2 WARNING  Warnings + errors
//   3 INFO     General operational info
//   4 DEBUG    Detailed debug info
//   5 TRACE    Very detailed tracing
//   6 VERBOSE  Function entry/exit with args
//
// CATEGORY MASKS (combine with OR):
//   0x1    INIT      Runtime init/shutdown
//   0x2    QUEUE     Queue create/destroy
//   0x4    MEM       Memory alloc/free
//   0x8    SIGNAL    Signal operations
//   0x10   IPC       IPC create/attach/detach
//   0x20   AGENT     Agent/topology
//   0x40   AQL       AQL packets
//   0x80   SDMA      SDMA operations
//   0x100  COPY      Copy operations
//   0x200  BLIT      BlitKernel ops
//   0x400  SCRATCH   Scratch allocation
//   0x800  POOL      Memory/signal pools
//   0x8000 EXCEPT    Exception handlers
//
// QUICK START:
//   # Error logging only
//   HSA_LOG_LEVEL=1 ./app
//
//   # Debug memory issues
//   HSA_LOG_LEVEL=4 HSA_LOG_MASK=0x4 ./app
//
//   # Full trace to file
//   HSA_LOG_LEVEL=5 HSA_LOG_FILE=/tmp/rocr ./app
//
//   # Show help
//   HSA_LOG_LEVEL=help ./app
//
// =============================================================================

#ifndef HSA_RUNTIME_CORE_UTIL_ROCR_LOGGING_H_
#define HSA_RUNTIME_CORE_UTIL_ROCR_LOGGING_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rocr {

// ============================================================================
// Log Levels - Severity levels for log messages
// ============================================================================
enum RocrLogLevel {
  ROCR_LOG_NONE = 0,     // No logging
  ROCR_LOG_ERROR = 1,    // Errors only
  ROCR_LOG_WARNING = 2,  // Warnings + errors
  ROCR_LOG_INFO = 3,     // General info
  ROCR_LOG_DEBUG = 4,    // Detailed debug
  ROCR_LOG_TRACE = 5,    // Very detailed tracing
  ROCR_LOG_VERBOSE = 6   // Extremely detailed
};

// ============================================================================
// Log Masks - Category bitmasks for filtering log output
// ============================================================================
enum RocrLogMask : uint64_t {
  // Core Operations
  ROCR_LOG_INIT = 0x1,         // Runtime init/shutdown
  ROCR_LOG_QUEUE = 0x2,        // Queue create/destroy
  ROCR_LOG_MEM = 0x4,          // Memory alloc/free
  ROCR_LOG_SIGNAL = 0x8,       // Signal operations
  ROCR_LOG_IPC = 0x10,         // IPC create/attach/detach
  ROCR_LOG_AGENT = 0x20,       // Agent/topology discovery

  // GPU Operations
  ROCR_LOG_AQL = 0x40,         // AQL packet logging
  ROCR_LOG_SDMA = 0x80,        // SDMA operations
  ROCR_LOG_COPY = 0x100,       // Copy operations
  ROCR_LOG_BLIT = 0x200,       // BlitKernel operations

  // Resource Management
  ROCR_LOG_SCRATCH = 0x400,    // Scratch allocation
  ROCR_LOG_POOL = 0x800,       // Memory/signal pools
  ROCR_LOG_CACHE = 0x1000,     // Caching operations

  // Error/Debug
  ROCR_LOG_FAULT = 0x2000,     // VM faults/memory faults
  ROCR_LOG_HANG = 0x4000,      // Hang detection
  ROCR_LOG_EXCEPT = 0x8000,    // Exception handlers

  // Synchronization
  ROCR_LOG_WAIT = 0x10000,     // Signal waits
  ROCR_LOG_LOCK = 0x20000,     // Lock contention

  // Detailed Output
  ROCR_LOG_AQL2 = 0x40000,     // Raw AQL packet bytes
  ROCR_LOG_SDMA2 = 0x80000,    // Raw SDMA packet bytes
  ROCR_LOG_LOCATION = 0x100000,  // Include file:line in output
  ROCR_LOG_TIMESTAMP = 0x200000, // Detailed timestamps

  // Special Categories
  ROCR_LOG_PCS = 0x1000000,    // PC Sampling
  ROCR_LOG_API = 0x2000000,    // HSA API calls

  // Performance & Health Monitoring
  ROCR_LOG_PERF = 0x400000,       // Performance anomalies
  ROCR_LOG_HEALTH = 0x800000,     // Health monitoring
  ROCR_LOG_METRICS = 0x4000000,   // Aggregated metrics
  ROCR_LOG_THERMAL = 0x8000000,   // Thermal throttling
  ROCR_LOG_POWER = 0x10000000,    // Power management

  // Multi-GPU & Distributed
  ROCR_LOG_P2P = 0x20000000,      // Peer-to-peer transfers
  ROCR_LOG_TOPOLOGY = 0x40000000, // Topology changes
  ROCR_LOG_XGMI = 0x80000000,     // XGMI link stats

  // Inference-specific
  ROCR_LOG_BATCH = 0x100000000,   // Batch processing
  ROCR_LOG_KERNEL = 0x200000000,  // Kernel launches

  // Always log (matches all masks)
  ROCR_LOG_ALWAYS = 0xFFFFFFFFFFFFFFFFULL
};

// ============================================================================
// Log Entry Structure for Async Logging
// ============================================================================
struct RocrLogEntry {
  RocrLogLevel level;           // Log severity level
  const char* file;             // Source file name
  int line;                     // Source line number
  char message[4096];           // Formatted log message
  uint64_t timestamp;           // Timestamp in microseconds
  uint32_t pid;                 // Process ID
  uint32_t tid;                 // Thread ID (hash)
  uint64_t duration;            // Duration in microseconds (0 if not a duration log)
  bool has_duration;            // True if this is a duration log entry
  std::atomic<bool> valid;      // Valid flag for lock-free synchronization

  RocrLogEntry()
      : level(ROCR_LOG_NONE),
        file(""),
        line(0),
        timestamp(0),
        pid(0),
        tid(0),
        duration(0),
        has_duration(false),
        valid(false) {
    message[0] = '\0';
  }
};

// ============================================================================
// Async Logger Class - Ring buffer based async logging
// ============================================================================
class RocrAsyncLogger {
 public:
  RocrAsyncLogger();
  ~RocrAsyncLogger();

  // Start the background worker thread
  void Start();

  // Stop the background worker thread
  void Stop();

  // Enable/disable async logging
  void Enable(bool enable);

  // Check if async logging is enabled
  bool IsEnabled() const {
    return enabled_.load(std::memory_order_relaxed);
  }

  // Log a message (async path)
  void Log(RocrLogLevel level, const char* file, int line, const char* message,
           uint64_t timestamp, uint64_t duration = 0, bool has_duration = false);

  // Request flush (async)
  void Flush();

  // Flush in current thread (synchronous)
  void FlushInCurrentThread();

  // Emergency flush on crash
  void FlushOnCrash() noexcept;

 private:
  static constexpr size_t kBufferSize = 16 * 1024;  // Circular buffer size
  static constexpr size_t kFlushIntervalMs = 1;     // Flush interval in milliseconds

  std::vector<RocrLogEntry> buffer_;     // Circular buffer of log entries
  std::atomic<size_t> write_index_{0};   // Write position in circular buffer
  std::atomic<size_t> read_index_{0};    // Read position in circular buffer
  std::atomic<bool> running_{false};     // Worker thread running flag
  std::atomic<bool> enabled_{false};     // Async logging enabled flag

  std::thread worker_thread_;            // Background worker thread for flushing
  std::mutex flush_mutex_;               // Mutex for flush condition variable
  std::condition_variable flush_cv_;     // Condition variable for worker wakeup

  void WorkerLoop();
  void FlushPending();
  void WriteToFile(const RocrLogEntry& entry);
};

// ============================================================================
// Global Logging State
// ============================================================================
struct RocrLoggingState {
  int log_level;                  // Current log level (from HSA_LOG_LEVEL)
  uint64_t log_mask;              // Current log mask (from HSA_LOG_MASK)
  std::string log_file_path;      // Log file path (from HSA_LOG_FILE)
  size_t log_max_size_mb;         // Max log file size in MB (from HSA_LOG_SIZE)
  bool async_enabled;             // Async logging enabled (from HSA_LOG_ASYNC)
  bool structured_output;         // JSON output mode (from HSA_LOG_FORMAT=json)
  FILE* log_file;                 // Log file handle (or stderr)
  bool initialized;               // Whether logging is initialized
  std::mutex file_mutex;          // Mutex for file operations

  RocrLoggingState()
      : log_level(0),
        log_mask(0x7FFFFFFF),
        log_max_size_mb(2048),
        async_enabled(false),
        structured_output(false),
        log_file(stderr),
        initialized(false) {}
};

// Global logging state instance
extern RocrLoggingState g_rocr_log_state;

// Global async logger instance
extern RocrAsyncLogger g_rocr_async_logger;

// ============================================================================
// Core Logging Functions
// ============================================================================

// Initialize the logging system (called from Runtime::Load)
void rocr_log_init();

// Shutdown the logging system (called from Runtime::Unload)
void rocr_log_shutdown();

// Core logging function
void rocr_log_printf(RocrLogLevel level, uint64_t mask, const char* file,
                     int line, const char* format, ...);

// Duration logging function
void rocr_log_printf_duration(RocrLogLevel level, uint64_t mask, const char* file,
                              int line, uint64_t* start_time, const char* format, ...);

// Get current timestamp in microseconds
uint64_t rocr_get_timestamp_us();

// ============================================================================
// Logging Macros
// ============================================================================

// Check if logging is enabled for a given level and mask
#define ROCR_LOG_ENABLED(level, mask) \
  (rocr::g_rocr_log_state.log_level >= (level) && \
   (rocr::g_rocr_log_state.log_mask & (mask) || (mask) == rocr::ROCR_LOG_ALWAYS))

// Get just the filename from full path
#ifdef _WIN32
#define __ROCR_FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
#define __ROCR_FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

// Main logging macro
#define RocrLog(level, mask, format, ...)                                    \
  do {                                                                       \
    if (ROCR_LOG_ENABLED(level, mask)) {                                     \
      rocr::rocr_log_printf(level, mask, __ROCR_FILENAME__, __LINE__,        \
                            format, ##__VA_ARGS__);                          \
    }                                                                        \
  } while (false)

// Convenience macros for different log levels
#define RocrLogError(mask, format, ...) \
  RocrLog(rocr::ROCR_LOG_ERROR, mask, format, ##__VA_ARGS__)

#define RocrLogWarning(mask, format, ...) \
  RocrLog(rocr::ROCR_LOG_WARNING, mask, format, ##__VA_ARGS__)

#define RocrLogInfo(mask, format, ...) \
  RocrLog(rocr::ROCR_LOG_INFO, mask, format, ##__VA_ARGS__)

#define RocrLogDebug(mask, format, ...) \
  RocrLog(rocr::ROCR_LOG_DEBUG, mask, format, ##__VA_ARGS__)

#define RocrLogTrace(mask, format, ...) \
  RocrLog(rocr::ROCR_LOG_TRACE, mask, format, ##__VA_ARGS__)

#define RocrLogVerbose(mask, format, ...) \
  RocrLog(rocr::ROCR_LOG_VERBOSE, mask, format, ##__VA_ARGS__)

// Duration logging macro - calculates duration from start_time
#define RocrLogDuration(level, mask, start_time_ptr, format, ...)            \
  do {                                                                       \
    if (ROCR_LOG_ENABLED(level, mask)) {                                     \
      rocr::rocr_log_printf_duration(level, mask, __ROCR_FILENAME__,         \
                                     __LINE__, start_time_ptr,               \
                                     format, ##__VA_ARGS__);                 \
    }                                                                        \
  } while (false)

// Scoped duration logging - logs on scope exit
class RocrScopedLog {
 public:
  RocrScopedLog(RocrLogLevel level, uint64_t mask, const char* file, int line,
                const char* msg)
      : level_(level),
        mask_(mask),
        file_(file),
        line_(line),
        msg_(msg),
        start_time_(rocr_get_timestamp_us()),
        enabled_(ROCR_LOG_ENABLED(level, mask)) {}

  ~RocrScopedLog() {
    if (enabled_) {
      rocr_log_printf_duration(level_, mask_, file_, line_, &start_time_,
                               "%s", msg_);
    }
  }

 private:
  RocrLogLevel level_;
  uint64_t mask_;
  const char* file_;
  int line_;
  const char* msg_;
  uint64_t start_time_;
  bool enabled_;
};

#define ROCR_SCOPED_LOG(level, mask, msg) \
  rocr::RocrScopedLog PASTE(rocr_scoped_log_, __COUNTER__)(level, mask, \
    __ROCR_FILENAME__, __LINE__, msg)

// Helper macro for PASTE
#ifndef PASTE
#define PASTE2(x, y) x##y
#define PASTE(x, y) PASTE2(x, y)
#endif

// ============================================================================
// Sampled Logging - For high-frequency events to reduce overhead
// ============================================================================

// Log only 1 in N occurrences (for high-frequency events)
#define RocrLogSampled(level, mask, rate, format, ...)                        \
  do {                                                                        \
    static std::atomic<uint64_t> _rocr_sample_counter{0};                     \
    if ((_rocr_sample_counter.fetch_add(1, std::memory_order_relaxed) %       \
         (rate)) == 0) {                                                      \
      RocrLog(level, mask, format, ##__VA_ARGS__);                            \
    }                                                                         \
  } while (false)

// Sampled warning - logs 1 in N occurrences
#define RocrLogWarningSampled(mask, rate, format, ...) \
  RocrLogSampled(rocr::ROCR_LOG_WARNING, mask, rate, format, ##__VA_ARGS__)

// Sampled debug - logs 1 in N occurrences
#define RocrLogDebugSampled(mask, rate, format, ...) \
  RocrLogSampled(rocr::ROCR_LOG_DEBUG, mask, rate, format, ##__VA_ARGS__)

// ============================================================================
// Rate-Limited Logging - For events that should not flood logs
// ============================================================================

// Log at most once per interval_ms milliseconds
#define RocrLogRateLimited(level, mask, interval_ms, format, ...)             \
  do {                                                                        \
    static std::atomic<uint64_t> _rocr_last_log_time{0};                      \
    uint64_t _rocr_now = rocr::rocr_get_timestamp_us();                       \
    uint64_t _rocr_last = _rocr_last_log_time.load(std::memory_order_relaxed);\
    if (_rocr_now - _rocr_last > (interval_ms) * 1000ULL) {                   \
      if (_rocr_last_log_time.compare_exchange_strong(_rocr_last, _rocr_now)) {\
        RocrLog(level, mask, format, ##__VA_ARGS__);                          \
      }                                                                       \
    }                                                                         \
  } while (false)

// ============================================================================
// Function Tracing - Entry/Exit logging with arguments and return values
// ============================================================================

// Log function entry with input arguments (VERBOSE level)
#define ROCR_TRACE_ENTER(mask, format, ...)                                    \
  do {                                                                         \
    if (ROCR_LOG_ENABLED(rocr::ROCR_LOG_VERBOSE, mask)) {                      \
      rocr::rocr_log_printf(rocr::ROCR_LOG_VERBOSE, mask, __ROCR_FILENAME__,   \
                            __LINE__, "ENTER %s(" format ")",                  \
                            __func__, ##__VA_ARGS__);                          \
    }                                                                          \
  } while (false)

// Log function exit with return value (VERBOSE level)
#define ROCR_TRACE_EXIT(mask, format, ...)                                     \
  do {                                                                         \
    if (ROCR_LOG_ENABLED(rocr::ROCR_LOG_VERBOSE, mask)) {                      \
      rocr::rocr_log_printf(rocr::ROCR_LOG_VERBOSE, mask, __ROCR_FILENAME__,   \
                            __LINE__, "EXIT %s" format,                        \
                            __func__, ##__VA_ARGS__);                          \
    }                                                                          \
  } while (false)

// Log function exit with hsa_status_t return value
#define ROCR_TRACE_EXIT_STATUS(mask, status)                                   \
  ROCR_TRACE_EXIT(mask, " -> %s (0x%x)",                                       \
                  rocr::rocr_hsa_status_name(status), (unsigned)(status))

// Scoped function tracer - logs entry on construction, exit on destruction
// Captures and logs function duration at VERBOSE level
class RocrScopedTrace {
 public:
  RocrScopedTrace(uint64_t mask, const char* file, int line, const char* func,
                  const char* args)
      : mask_(mask),
        file_(file),
        line_(line),
        func_(func),
        start_time_(rocr_get_timestamp_us()),
        enabled_(ROCR_LOG_ENABLED(ROCR_LOG_VERBOSE, mask)) {
    if (enabled_) {
      rocr_log_printf(ROCR_LOG_VERBOSE, mask, file, line,
                      "ENTER %s(%s)", func, args ? args : "");
    }
  }

  // Set return value to be logged on scope exit
  void SetReturn(const char* ret_fmt, ...) {
    if (enabled_ && ret_fmt) {
      va_list args;
      va_start(args, ret_fmt);
      vsnprintf(return_value_, sizeof(return_value_), ret_fmt, args);
      va_end(args);
      has_return_ = true;
    }
  }

  ~RocrScopedTrace() {
    if (enabled_) {
      uint64_t duration = rocr_get_timestamp_us() - start_time_;
      if (has_return_) {
        rocr_log_printf(ROCR_LOG_VERBOSE, mask_, file_, line_,
                        "EXIT %s -> %s [%llu us]",
                        func_, return_value_, (unsigned long long)duration);
      } else {
        rocr_log_printf(ROCR_LOG_VERBOSE, mask_, file_, line_,
                        "EXIT %s [%llu us]", func_, (unsigned long long)duration);
      }
    }
  }

 private:
  uint64_t mask_;
  const char* file_;
  int line_;
  const char* func_;
  uint64_t start_time_;
  bool enabled_;
  char return_value_[256] = {0};
  bool has_return_ = false;
};

// Create a scoped tracer that logs entry with args and exit with duration
// Usage: ROCR_SCOPED_TRACE(ROCR_LOG_MEM, "ptr=%p size=%zu", ptr, size);
//        ... code ...
// Note: Uses __LINE__ for unique variable names (one invocation per line)
#define ROCR_SCOPED_TRACE(mask, format, ...)                                   \
  char PASTE(_rocr_trace_args_, __LINE__)[256];                                \
  snprintf(PASTE(_rocr_trace_args_, __LINE__),                                 \
           sizeof(PASTE(_rocr_trace_args_, __LINE__)),                         \
           format, ##__VA_ARGS__);                                             \
  rocr::RocrScopedTrace PASTE(_rocr_tracer_, __LINE__)(                        \
      mask, __ROCR_FILENAME__, __LINE__, __func__,                             \
      PASTE(_rocr_trace_args_, __LINE__))

// Simpler version without args (just function name)
#define ROCR_SCOPED_TRACE_FUNC(mask)                                           \
  rocr::RocrScopedTrace PASTE(_rocr_tracer_, __LINE__)(                        \
      mask, __ROCR_FILENAME__, __LINE__, __func__, nullptr)

// Helper to convert hsa_status_t to string (for ROCR_TRACE_EXIT_STATUS)
const char* rocr_hsa_status_name(uint32_t status);

}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_UTIL_ROCR_LOGGING_H_
