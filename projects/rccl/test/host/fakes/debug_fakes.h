/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The shared ncclDebugLog recorder (src/debug.cc) for binaries that cannot link nccl_fakes.cc. It keeps every
// line at any level, under a lock, so a test can poll what a unit's worker threads logged.

#ifndef RCCL_TEST_HOST_DEBUG_FAKES_H_
#define RCCL_TEST_HOST_DEBUG_FAKES_H_

#include <string>
#include <vector>

#include "nccl.h"
#include "nccl_common.h"

// The text of every line logged at `level`, in order.
std::vector<std::string> DebugLogLines(ncclDebugLogLevel level);
// Every line, at any level, that contains `needle`.
std::vector<std::string> DebugLogMatching(const std::string& needle);
bool DebugLogged(ncclDebugLogLevel level, const std::string& needle);
void ResetDebugLog();

#endif  // RCCL_TEST_HOST_DEBUG_FAKES_H_
