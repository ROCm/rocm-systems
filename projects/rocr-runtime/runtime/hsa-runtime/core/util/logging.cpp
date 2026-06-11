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

uint64_t get_timestamp_us() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

// ============================================================================
// Help Text
// ============================================================================

static void PrintHelpAndExit() {
  fprintf(stderr,
          "\n"
          "ROCR Runtime Logging - Environment Variables\n"
          "=============================================\n"
          "\n"
          "For HIP applications, use unified CLR+ROCR+Thunk logging:\n"
          "  AMD_LOG_LEVEL=6 ./hip_app\n"
          "\n"
          "For standalone HSA applications:\n"
          "  HSA_LOG_LEVEL=<0-6>   Log verbosity (0=none, 1=error, 2=warn, 3=info, 4=debug)\n"
          "  HSA_LOG_MASK=<hex>    Category bitmask (default=all)\n"
          "  HSA_LOG_FILE=<path>   Output file (default=stderr)\n"
          "\n"
          "Log Levels:\n"
          "  0 NONE     No logging (default)\n"
          "  1 ERROR    Critical errors only\n"
          "  2 WARNING  Warnings + errors\n"
          "  3 INFO     General operational info\n"
          "  4 DEBUG    Detailed debug info\n"
          "\n"
          "Categories (HSA_LOG_MASK):\n"
          "  0x1  MEM    Memory allocations/free\n"
          "  0x2  INFO   General info logs\n"
          "  0x4  ERROR  Error logs\n"
          "\n"
          "Examples:\n"
          "  HSA_LOG_LEVEL=1 ./app              # Errors only\n"
          "  HSA_LOG_LEVEL=3 ./app              # Info level\n"
          "  HSA_LOG_LEVEL=4 HSA_LOG_MASK=0x1 ./app  # Debug memory only\n"
          "\n");
  exit(0);
}

// ============================================================================
// Initialization
// ============================================================================

void log_init() {
  if (g_log_state.initialized) return;

  // Check if CLR is controlling logging (AMD_LOG_LEVEL set)
  const char* amd_log_level = getenv("AMD_LOG_LEVEL");
  if (amd_log_level && atoi(amd_log_level) >= 6) {
    // CLR will configure logging via hsa_amd_enable_logging()
    g_log_state.clr_controlled = true;
    g_log_state.initialized = true;
    return;
  }

  // Parse HSA_LOG_LEVEL
  const char* level_str = getenv("HSA_LOG_LEVEL");
  if (level_str) {
    if (strcmp(level_str, "help") == 0 || strcmp(level_str, "HELP") == 0) {
      PrintHelpAndExit();
    }
    g_log_state.log_level = atoi(level_str);
  }

  // Parse HSA_LOG_MASK
  const char* mask_str = getenv("HSA_LOG_MASK");
  if (mask_str) {
    g_log_state.log_mask = strtoull(mask_str, nullptr, 0);
  }

  // Parse HSA_LOG_FILE
  const char* file_str = getenv("HSA_LOG_FILE");
  if (file_str && strlen(file_str) > 0) {
    // Append PID to filename
    char filename[512];
    snprintf(filename, sizeof(filename), "%s.%u", file_str, GetPid());
    FILE* f = fopen(filename, "w");
    if (f) {
      g_log_state.log_file = f;
      g_log_state.owns_log_file = true;
    }
  }

  g_log_state.initialized = true;

  if (g_log_state.log_level > 0) {
    fprintf(stderr, "[ROCR] Logging enabled: level=%d mask=0x%" PRIx64 "\n", g_log_state.log_level,
            g_log_state.log_mask);
  }
}

void log_shutdown() {
  if (g_log_state.owns_log_file && g_log_state.log_file && g_log_state.log_file != stderr &&
      g_log_state.log_file != stdout) {
    fclose(g_log_state.log_file);
    g_log_state.log_file = stderr;
    g_log_state.owns_log_file = false;
  }
  g_log_state.initialized = false;
}

// ============================================================================
// Core Logging Function
// ============================================================================

void log_printf(int level, uint64_t mask, const char* file, int line, const char* format, ...) {
  if (g_log_state.log_level < level) return;
  if (!(g_log_state.log_mask & mask)) return;

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

// ============================================================================
// IPC Leak Warning
// ============================================================================

void log_ipc_warning(uint64_t active_count, uint64_t total_count, void* ptr) {
  // Warn when IPC attachment count exceeds threshold (potential leak)
  if (active_count > 1000 || (total_count > 10000 && active_count > total_count / 2)) {
    LogWarning(LOG_MEM,
               "High IPC attachment count: active=%" PRIu64 " total=%" PRIu64
               " ptr=%p POTENTIAL_LEAK",
               active_count, total_count, ptr);
  }
}

}  // namespace rocr
