/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See debug_fakes.h.

#include "debug_fakes.h"

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <utility>

#include "debug.h"

uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // INFO/TRACE always call in, any subsystem
uint64_t ncclDebugMask = 0;
thread_local int ncclDebugNoWarn = 0;

namespace {
std::mutex g_debugLogMutex;  // units log from their own threads too
std::vector<std::pair<ncclDebugLogLevel, std::string>> g_debugLog;
}  // namespace

void ncclDebugLog(ncclDebugLogLevel level, unsigned long /*flags*/, const char* /*filefunc*/, int /*line*/,
                  const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  std::lock_guard<std::mutex> lock(g_debugLogMutex);
  g_debugLog.emplace_back(level, buf);
}

std::vector<std::string> DebugLogLines(ncclDebugLogLevel level) {
  std::lock_guard<std::mutex> lock(g_debugLogMutex);
  std::vector<std::string> lines;
  for (const auto& l : g_debugLog) {
    if (l.first == level) lines.push_back(l.second);
  }
  return lines;
}

std::vector<std::string> DebugLogMatching(const std::string& needle) {
  std::lock_guard<std::mutex> lock(g_debugLogMutex);
  std::vector<std::string> lines;
  for (const auto& l : g_debugLog) {
    if (l.second.find(needle) != std::string::npos) lines.push_back(l.second);
  }
  return lines;
}

bool DebugLogged(ncclDebugLogLevel level, const std::string& needle) {
  for (const std::string& line : DebugLogLines(level)) {
    if (line.find(needle) != std::string::npos) return true;
  }
  return false;
}

void ResetDebugLog() {
  std::lock_guard<std::mutex> lock(g_debugLogMutex);
  g_debugLog.clear();
}
