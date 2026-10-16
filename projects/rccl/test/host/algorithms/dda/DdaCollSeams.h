/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams shared by the DDA collective launcher tests. Include before the launcher: it renames
// hipExtLaunchKernelGGL so each launch is recorded, with its typed arguments, instead of started.

#ifndef RCCL_TEST_HOST_ALGORITHMS_DDA_DDACOLLSEAMS_H_
#define RCCL_TEST_HOST_ALGORITHMS_DDA_DDACOLLSEAMS_H_

#include <any>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include "comm.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"  // g_loadParam, used by param_redirect.h

struct DdaLaunch {
  const void* kernel;
  dim3 grid;
  dim3 block;
  uint32_t sharedMemBytes;
  hipStream_t stream;
  hipEvent_t startEvent;
  hipEvent_t stopEvent;
  uint32_t flags;
  std::vector<std::any> args;
  // Throws std::bad_any_cast when the launcher passed a different type.
  template <typename T>
  T Arg(size_t i) const {
    return std::any_cast<T>(args.at(i));
  }
};

extern std::vector<DdaLaunch> g_ddaLaunches;

template <typename F, typename... Args>
void RecordDdaLaunch(F kernel, const dim3& grid, const dim3& block, uint32_t sharedMemBytes, hipStream_t stream,
                     hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags, Args... args) {
  g_ddaLaunches.push_back({reinterpret_cast<const void*>(kernel), grid, block, sharedMemBytes, stream, startEvent,
                           stopEvent, flags, {std::any(args)...}});
}

#define hipExtLaunchKernelGGL RecordDdaLaunch

// One hipMemcpyAsync, with how many launches had been recorded before it.
struct DdaCopy {
  void* dst;
  const void* src;
  size_t bytes;
  hipMemcpyKind kind;
  hipStream_t stream;
  size_t launchesBefore;
};

inline void RecordDdaCopies(std::vector<DdaCopy>* copies) {
  g_hipMemcpyAsync = [copies](void* dst, const void* src, size_t bytes, hipMemcpyKind kind, hipStream_t stream) {
    copies->push_back({dst, src, bytes, kind, stream, g_ddaLaunches.size()});
    return hipSuccess;
  };
}

template <typename K>
const void* KernelPtr(K kernel) {
  return reinterpret_cast<const void*>(kernel);
}

// One per State type, never destroyed: the destructor would pull in barrier resource teardown this binary does
// not link. The barrier bytes are filled so a launch can be checked to carry this exact barrier.
template <typename State>
State* SharedBarrierState(unsigned char fill) {
  static State* const state = new State();
  std::memset(static_cast<void*>(&state->barrierHost), fill, sizeof(state->barrierHost));
  return state;
}

template <typename Barrier>
bool SameBarrier(const Barrier& a, const Barrier& b) {
  return std::memcmp(&a, &b, sizeof(Barrier)) == 0;
}

// rccl_wrap.cc DDA thresholds; defaults match rccl_wrap.cc with no arch table and no env.
extern std::function<size_t(const ncclComm*, ncclFunc_t)> g_rcclDdaLLThreshold;
extern std::function<size_t(const ncclComm*, ncclFunc_t)> g_rcclDdaLL128Threshold;
extern std::function<size_t(const ncclComm*)> g_rcclDdaLLTwoShotThreshold;
extern std::function<size_t(const ncclComm*)> g_rcclDdaLL128TwoShotThreshold;
extern std::function<int(const ncclComm*)> g_rcclDdaFabricVmmBlocks;

void ResetDdaCollSeams();

#endif  // RCCL_TEST_HOST_ALGORITHMS_DDA_DDACOLLSEAMS_H_
