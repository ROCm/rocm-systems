/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/util/logging.h"

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <inttypes.h>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#include <sys/syscall.h>
#endif

namespace rocr {

// Global logging state instance
LoggingState g_log_state;

// ============================================================================
// Helper Functions
// ============================================================================

static uint32_t GetPid() {
#ifdef _WIN32
  return static_cast<uint32_t>(_getpid());
#else
  return static_cast<uint32_t>(getpid());
#endif
}

static uint32_t GetTid() {
#ifdef _WIN32
  return static_cast<uint32_t>(GetCurrentThreadId());
#else
  return static_cast<uint32_t>(syscall(SYS_gettid));
#endif
}

static const char* LevelName(int level) {
  switch (level) {
    case LOG_ERROR:
      return "ERROR";
    case LOG_WARNING:
      return "WARN";
    case LOG_INFO:
      return "INFO";
    case LOG_DEBUG:
      return "DEBUG";
    case LOG_TRACE:
      return "TRACE";
    case LOG_VERBOSE:
      return "VERBOSE";
    default:
      return "?";
  }
}

// ============================================================================
// Timestamp
// ============================================================================

// Monotonic by construction. The only callers subtract two readings to report an
// elapsed wait, and high_resolution_clock aliases the non-monotonic system_clock
// on libstdc++, where a backward clock step would underflow that unsigned
// subtraction and report an absurd duration.
uint64_t get_timestamp_us() {
  auto now = std::chrono::steady_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

// ============================================================================
// Help Text
// ============================================================================

static void PrintHelp() {
  fprintf(stderr,
          "\n"
          "ROCR Runtime Logging - Environment Variables\n"
          "=============================================\n"
          "\n"
          "For HIP applications, use unified CLR+ROCR logging:\n"
          "  AMD_LOG_LEVEL=6 ./hip_app\n"
          "\n"
          "For standalone HSA applications:\n"
          "  HSA_LOG_LEVEL=<0-6>   Log verbosity (see below). 'help' prints this text.\n"
          "  HSA_LOG_MASK=<hex>    Category bitmask (default=all categories)\n"
          "  HSA_LOG_FILE=<path>   Output file prefix (default=stderr)\n"
          "\n"
          "Log Levels:\n"
          "  0 NONE     No logging (default)\n"
          "  1 ERROR    Critical errors only\n"
          "  2 WARNING  Warnings + errors\n"
          "  3 INFO     General operational info\n"
          "  4 DEBUG    Detailed debug info\n"
          "  5 TRACE    Very detailed tracing\n"
          "  6 VERBOSE  Function entry/exit with arguments\n"
          "\n"
          "Categories (HSA_LOG_MASK):\n"
          "  0x1  MEM    Memory allocation, free and IPC sharing\n"
          "  0x2  INFO   Init, queue, signal, agent, AQL and SDMA activity\n"
          "  0x4  ERROR  Error and exception paths\n"
          "\n"
          "Examples:\n"
          "  HSA_LOG_LEVEL=1 ./app                   # Errors only\n"
          "  HSA_LOG_LEVEL=3 ./app                   # Info level, all categories\n"
          "  HSA_LOG_LEVEL=4 HSA_LOG_MASK=0x1 ./app  # Debug level, memory only\n"
          "\n"
          "Thunk (libhsakmt) logging is separate and is controlled by\n"
          "HSAKMT_DEBUG_LEVEL. Thunk output always goes to stderr.\n"
          "\n");
}

// ============================================================================
// Initialization
// ============================================================================

void log_init() {
  // Claim initialization exactly once. Concurrent callers that lose the race
  // return immediately instead of re-parsing the environment.
  bool expected = false;
  if (!g_log_state.initialized.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                                       std::memory_order_acquire)) {
    return;
  }

  // When AMD_LOG_LEVEL selects level 6 or above, CLR configures ROCR logging
  // through hsa_amd_enable_logging() and the HSA_LOG_* variables are ignored.
  const char* amd_log_level = getenv("AMD_LOG_LEVEL");
  if (amd_log_level && atoi(amd_log_level) >= 6) {
    return;
  }

  // Parse HSA_LOG_LEVEL
  int level = 0;
  const char* level_str = getenv("HSA_LOG_LEVEL");
  if (level_str) {
    if (strcmp(level_str, "help") == 0 || strcmp(level_str, "HELP") == 0) {
      // Print usage and continue with logging disabled. A library must not
      // terminate the host process.
      PrintHelp();
    } else {
      level = atoi(level_str);
      if (level < LOG_NONE) level = LOG_NONE;
      if (level > LOG_VERBOSE) level = LOG_VERBOSE;
    }
  }

  // Parse HSA_LOG_MASK
  uint64_t mask = g_log_state.log_mask.load(std::memory_order_relaxed);
  const char* mask_str = getenv("HSA_LOG_MASK");
  if (mask_str) {
    mask = strtoull(mask_str, nullptr, 0);
  }

  // Parse HSA_LOG_FILE. The process id is appended so that concurrent processes
  // do not share one file.
  const char* file_str = getenv("HSA_LOG_FILE");
  if (file_str && strlen(file_str) > 0) {
    char filename[512];
    snprintf(filename, sizeof(filename), "%s.%u", file_str, GetPid());
    FILE* f = fopen(filename, "w");
    if (f) {
      std::lock_guard<std::mutex> lock(g_log_state.file_mutex);
      g_log_state.log_file = f;
      g_log_state.owns_log_file = true;
    } else {
      fprintf(stderr, "[ROCR] Could not open log file '%s', logging to stderr\n", filename);
    }
  }

  g_log_state.log_mask.store(mask, std::memory_order_relaxed);
  g_log_state.log_level.store(level, std::memory_order_relaxed);

  if (level > LOG_NONE) {
    fprintf(stderr, "[ROCR] Logging enabled: level=%d mask=0x%" PRIx64 "\n", level, mask);
  }
}

void log_shutdown() {
  g_log_state.log_level.store(LOG_NONE, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(g_log_state.file_mutex);
    if (g_log_state.owns_log_file && g_log_state.log_file && g_log_state.log_file != stderr &&
        g_log_state.log_file != stdout) {
      fclose(g_log_state.log_file);
    }
    g_log_state.log_file = stderr;
    g_log_state.owns_log_file = false;
  }
  g_log_state.initialized.store(false, std::memory_order_release);
}

void log_set_output(FILE* file) {
  std::lock_guard<std::mutex> lock(g_log_state.file_mutex);
  if (g_log_state.owns_log_file && g_log_state.log_file && g_log_state.log_file != stderr &&
      g_log_state.log_file != stdout) {
    fclose(g_log_state.log_file);
  }
  g_log_state.log_file = file ? file : stderr;
  g_log_state.owns_log_file = false;
}

// ============================================================================
// Core Logging Function
// ============================================================================

void log_printf(int level, uint64_t mask, const char* file, int line, const char* format, ...) {
  if (g_log_state.log_level.load(std::memory_order_relaxed) < level) return;
  if (!(g_log_state.log_mask.load(std::memory_order_relaxed) & mask)) return;

  // Format: [PID:TID] LEVEL file:line message
  // This format matches CLR logging format
  char buffer[4096];
  int offset = snprintf(buffer, sizeof(buffer), "[%u:%u] %s %s:%d ", GetPid(), GetTid(),
                        LevelName(level), file, line);

  if (offset > 0 && offset < (int)sizeof(buffer)) {
    va_list args;
    va_start(args, format);
    vsnprintf(buffer + offset, sizeof(buffer) - offset, format, args);
    va_end(args);
  }

  // Write to log file with mutex protection
  {
    std::lock_guard<std::mutex> lock(g_log_state.file_mutex);
    fprintf(g_log_state.log_file, "%s\n", buffer);
    fflush(g_log_state.log_file);
  }
}

}  // namespace rocr
