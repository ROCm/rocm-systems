/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// src/plugin/profiler.cc's ncclProfilerProxyTraceDumpIfAny, recorded. The rest of that file's floor is still in
// nccl_stubs.cc and comm_fakes.cc.

#ifndef RCCL_TEST_HOST_PROFILER_FAKES_H_
#define RCCL_TEST_HOST_PROFILER_FAKES_H_

#include <vector>

// Every profilerContext passed to ncclProfilerProxyTraceDumpIfAny since the last reset, in order.
std::vector<void*> ProfilerProxyTraceDumps();
void ResetProfilerFakes();

#endif  // RCCL_TEST_HOST_PROFILER_FAKES_H_
