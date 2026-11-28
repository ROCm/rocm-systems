/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// =============================================================================
// ROCR Runtime Logging System
// =============================================================================
//
// Unified logging with CLR/HIP. Disabled by default.
//
// CLR/HIP INTEGRATION:
//   When AMD_LOG_LEVEL >= 6, CLR enables ROCR logging via hsa_amd_enable_logging().
//   Both CLR and ROCR logs go to the same file for unified debugging.
//
//   For HIP applications, use AMD_LOG_LEVEL for unified CLR+ROCR+Thunk logging:
//     AMD_LOG_LEVEL=6 ./hip_app
//
// STANDALONE HSA APPLICATIONS:
//   HSA_LOG_LEVEL=help ./app    Print usage help to stderr and keep running
//   HSA_LOG_LEVEL=3 ./app       Enable info-level logging
//
// =============================================================================

#ifndef HSA_RUNTIME_CORE_UTIL_LOGGING_H_
#define HSA_RUNTIME_CORE_UTIL_LOGGING_H_

#include <atomic>
#include <cinttypes>  // PRIu64/PRIx64/PRId64, used by callers' format strings
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#ifdef _WIN32
#include <io.h>
#define STDERR_FILENO 2
#else
#include <unistd.h>
#endif

namespace rocr {

// ============================================================================
// Log Levels - Severity levels for log messages
// ============================================================================
enum LogLevel {
  LOG_NONE = 0,     // No logging
  LOG_ERROR = 1,    // Errors only
  LOG_WARNING = 2,  // Warnings + errors
  LOG_INFO = 3,     // General info
  LOG_DEBUG = 4,    // Detailed debug
  LOG_TRACE = 5,    // Very detailed tracing
  LOG_VERBOSE = 6   // Extremely detailed
};

// ============================================================================
// Log Categories - Simplified per review feedback (MEM/INFO/ERROR)
// ============================================================================
enum LogMask : uint64_t {
  // Primary categories (per reviewer feedback)
  LOG_MEM = 0x1,        // Memory allocations/free
  LOG_INFO_CAT = 0x2,   // General info logs
  LOG_ERROR_CAT = 0x4,  // Error logs

  // Additional categories mapped to above for compatibility
  LOG_INIT = LOG_INFO_CAT,     // Runtime init/shutdown -> INFO
  LOG_QUEUE = LOG_INFO_CAT,    // Queue operations -> INFO
  LOG_SIGNAL = LOG_INFO_CAT,   // Signal operations -> INFO
  LOG_IPC = LOG_MEM,           // IPC -> MEM (memory sharing)
  LOG_AGENT = LOG_INFO_CAT,    // Agent/topology -> INFO
  LOG_AQL = LOG_INFO_CAT,      // AQL packets -> INFO
  LOG_SDMA = LOG_INFO_CAT,     // SDMA operations -> INFO
  LOG_COPY = LOG_MEM,          // Copy operations -> MEM
  LOG_BLIT = LOG_INFO_CAT,     // BlitKernel -> INFO
  LOG_SCRATCH = LOG_MEM,       // Scratch allocation -> MEM
  LOG_POOL = LOG_MEM,          // Memory/signal pools -> MEM
  LOG_EXCEPT = LOG_ERROR_CAT,  // Exception handlers -> ERROR

  // Always log (matches all masks)
  LOG_ALWAYS = 0xFFFFFFFFFFFFFFFFULL
};

// Additional category aliases for combined masks
constexpr uint64_t LOG_WAIT = LOG_INFO_CAT;   // Wait operations -> INFO
constexpr uint64_t LOG_HANG = LOG_ERROR_CAT;  // Hang detection -> ERROR
constexpr uint64_t LOG_HEALTH = LOG_INFO_CAT; // Health checks -> INFO

// ============================================================================
// Global Logging State
// ============================================================================
struct LoggingState {
  // Read on every log site without a lock, so these are atomic. Relaxed ordering
  // is sufficient: a reader that observes a stale level or mask only mis-filters
  // one message.
  std::atomic<int> log_level;       // Current log level (from HSA_LOG_LEVEL)
  std::atomic<uint64_t> log_mask;   // Current log mask (from HSA_LOG_MASK)
  std::atomic<bool> initialized;    // Whether logging is initialized

  // log_file and owns_log_file are only read or written while file_mutex is held.
  FILE* log_file;         // Log file handle (or stderr)
  bool owns_log_file;     // True if we opened log_file and should close it
  std::mutex file_mutex;  // Guards log_file, owns_log_file and the writes through them

  LoggingState()
      : log_level(0),
        log_mask(0xFFFFFFFFFFFFFFFFULL),
        initialized(false),
        log_file(stderr),
        owns_log_file(false) {}
};

// Global logging state instance
extern LoggingState g_log_state;

// ============================================================================
// Core Logging Functions
// ============================================================================

// Initialize the logging system (called from Runtime::Load)
void log_init();

// Shutdown the logging system (called from Runtime::Unload)
void log_shutdown();

// Core logging function
void log_printf(int level, uint64_t mask, const char* file, int line, const char* format, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 5, 6)))
#endif
    ;

// Replace the log output stream. Any stream previously opened by the logging
// system is closed first. The caller keeps ownership of 'file'; passing nullptr
// restores stderr. Safe to call while other threads are logging.
void log_set_output(FILE* file);

// Get current timestamp in microseconds
uint64_t get_timestamp_us();


// ============================================================================
// Logging Macros
// ============================================================================

// Check if logging is enabled for a given level and mask
#define LOG_ENABLED(level, mask)                                                                   \
  (rocr::g_log_state.log_level.load(std::memory_order_relaxed) >= (level) &&                       \
   (rocr::g_log_state.log_mask.load(std::memory_order_relaxed) & (mask)))


// Get just the filename from full path
#ifdef _WIN32
#define __LOG_FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
#define __LOG_FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif


// Main logging macro
#define Log(level, mask, format, ...)                                                              \
  do {                                                                                             \
    if (LOG_ENABLED(level, mask)) {                                                                \
      rocr::log_printf(level, mask, __LOG_FILENAME__, __LINE__, format, ##__VA_ARGS__);            \
    }                                                                                              \
  } while (false)

// Convenience macros for different log levels
#define LogError(mask, format, ...) Log(rocr::LOG_ERROR, mask, format, ##__VA_ARGS__)

#define LogWarning(mask, format, ...) Log(rocr::LOG_WARNING, mask, format, ##__VA_ARGS__)

#define LogInfo(mask, format, ...) Log(rocr::LOG_INFO, mask, format, ##__VA_ARGS__)

#define LogDebug(mask, format, ...) Log(rocr::LOG_DEBUG, mask, format, ##__VA_ARGS__)

// Trace and Verbose level macros
#define LogTrace(mask, format, ...) Log(rocr::LOG_TRACE, mask, format, ##__VA_ARGS__)
#define LogVerbose(mask, format, ...) Log(rocr::LOG_VERBOSE, mask, format, ##__VA_ARGS__)

// ============================================================================
// Function Tracing - Entry/Exit logging
// ============================================================================

#define ROCR_TRACE_ENTER(mask, format, ...)                                                        \
  do {                                                                                             \
    if (LOG_ENABLED(rocr::LOG_VERBOSE, mask)) {                                                    \
      rocr::log_printf(rocr::LOG_VERBOSE, mask, __LOG_FILENAME__, __LINE__,                        \
                       "ENTER %s(" format ")", __func__, ##__VA_ARGS__);                           \
    }                                                                                              \
  } while (false)

#define ROCR_TRACE_EXIT(mask, format, ...)                                                         \
  do {                                                                                             \
    if (LOG_ENABLED(rocr::LOG_VERBOSE, mask)) {                                                    \
      rocr::log_printf(rocr::LOG_VERBOSE, mask, __LOG_FILENAME__, __LINE__, "EXIT %s" format,      \
                       __func__, ##__VA_ARGS__);                                                   \
    }                                                                                              \
  } while (false)

#define ROCR_TRACE_EXIT_STATUS(mask, status) ROCR_TRACE_EXIT(mask, " -> 0x%x", (unsigned)(status))

}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_UTIL_LOGGING_H_
