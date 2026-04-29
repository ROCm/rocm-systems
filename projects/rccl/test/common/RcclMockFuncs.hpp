/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */

#ifndef RCCL_MOCK_FUNCS_HPP
#define RCCL_MOCK_FUNCS_HPP

#include "info.h"

void ncclDebugLog(ncclDebugLogLevel, unsigned long, char const*, int, char const*, ...) {};
ncclResult_t getHostName(char* hostname, int maxlen, const char delim) {
  return ncclSuccess;
}

// Mocked debug-level globals used by INFO/WARN/TRACE macros referenced by
// proxy_trace.cc and CollTraceUtils.cc when they're compiled into the test
// binary directly (Release build with hidden visibility on rccl).
int ncclDebugLevel = -1;
uint64_t ncclDebugMask = 0;

#endif  // RCCL_MOCK_FUNCS_HPP
