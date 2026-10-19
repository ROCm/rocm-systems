/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Externals src/include/alloc.h's helpers reach, for the binaries that compile them through their unit:
// rccl-UnitTestsMicroAlloc, rccl-UnitTestsMicroAllocator and rccl-UnitTestsMicroMnnvl. HIP comes from
// hip_fakes.cc. nccl_fakes.cc also defines the memory-manager calls, but without recording them, so these
// binaries do not link it and take its logging and cuMem globals from here.

#ifndef RCCL_TEST_HOST_ALLOC_MICRO_FAKES_H_
#define RCCL_TEST_HOST_ALLOC_MICRO_FAKES_H_

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "alloc.h"

// Every WARN, in order.
bool Warned(const char* needle);
extern std::vector<std::string> g_warns;

extern int g_cuMemEnable;            // what ncclCuMemEnable returns
extern ncclResult_t g_busIdResult;   // getBusId reports 0x100 * cudaDev with this result

// The memory-manager calls alloc.h makes, recorded.
struct TrackCall {
  void* ptr;
  size_t size;
  hipMemAllocationHandleType type;
  ncclMemType_t memType;
};
extern std::vector<TrackCall> g_tracked;
extern ncclResult_t g_trackResult;
extern ncclMemUntrackInfo g_untrackInfo;  // what ncclMemUntrackDynamic hands back
extern ncclResult_t g_untrackResult;
extern std::vector<std::pair<void*, size_t>> g_untrackedPersist;

// Backs every NCCL_PARAM a unit routes through param_redirect.h; the default returns the default.
extern std::function<int64_t(const char*, int64_t)> g_loadParam;

void ResetAllocMicroFakes();

#endif  // RCCL_TEST_HOST_ALLOC_MICRO_FAKES_H_
