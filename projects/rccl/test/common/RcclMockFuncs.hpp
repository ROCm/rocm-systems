/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */

#ifndef RCCL_MOCK_FUNCS_HPP
#define RCCL_MOCK_FUNCS_HPP

#include "info.h"

void ncclDebugLog(ncclDebugLogLevel, unsigned long, char const*, int, char const*, ...) {};
ncclResult_t getHostName(char* hostname, int maxlen, const char delim) {
  return ncclSuccess;
}

#endif  // RCCL_MOCK_FUNCS_HPP
