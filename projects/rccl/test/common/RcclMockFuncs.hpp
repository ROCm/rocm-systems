/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */

#ifndef RCCL_MOCK_FUNCS_HPP
#define RCCL_MOCK_FUNCS_HPP

#include "info.h"

// Old signature (pre-v2.30.7)
__attribute__((weak)) void ncclDebugLog(ncclDebugLogLevel, unsigned long, char const*, int, char const*, ...) {};
// New signature (v2.30.7+): file + func + line args added
__attribute__((weak)) void ncclDebugLogInternal(ncclDebugLogLevel, unsigned long, char const*, char const*, int, char const*, ...) {};
ncclResult_t getHostName(char* hostname, int maxlen, const char delim) {
  return ncclSuccess;
}

#endif  // RCCL_MOCK_FUNCS_HPP
