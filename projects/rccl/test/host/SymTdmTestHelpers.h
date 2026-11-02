/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared by sym-kernels-test.cc and sym-model-test.cc, which link into the same binary.

#ifndef RCCL_TEST_HOST_SYM_TDM_TEST_HELPERS_H_
#define RCCL_TEST_HOST_SYM_TDM_TEST_HELPERS_H_

#include <cstdint>
#include <functional>
#include <string>

#include "comm.h"
#include "tdm/tdmCopy.h"  // TDM_TOOLCHAIN_AVAILABLE

constexpr int kGfx1250CompCap = 125;  // HIP reports gfx1250 as compute capability 12.5

// g_loadParam stub: SYM_TMA_ENABLE reads as value; every other param keeps its default.
inline std::function<int64_t(const char*, int64_t)> SymTmaEnable(int64_t value) {
  return [value](const char* env, int64_t deftVal) -> int64_t {
    return std::string(env) == "SYM_TMA_ENABLE" ? value : deftVal;
  };
}

// ncclSymkTmaAvailable keys on archName under __HIP_PLATFORM_AMD__ and on minCompCap otherwise, and
// rcclSymkCalculateWarps divides the block width by WarpSize, so all three must describe one arch.
inline void SetSymkArch(ncclComm* comm, const char* arch, int compCap, int warpSize) {
  comm->archName = const_cast<char*>(arch);
  comm->compCap = comm->minCompCap = comm->maxCompCap = compCap;
  comm->WarpSize = warpSize;
}

inline void SetTdmCapableArch(ncclComm* comm) { SetSymkArch(comm, "gfx1250", kGfx1250CompCap, /*warpSize=*/32); }

// ROCm also needs the SDK's TDM header, so only cases asserting a Tma kernel survives need this.
inline bool TmaReachableOnThisBuild() {
#if defined(__HIP_PLATFORM_AMD__)
  return TDM_TOOLCHAIN_AVAILABLE;
#else
  return true;
#endif
}

#endif  // RCCL_TEST_HOST_SYM_TDM_TEST_HELPERS_H_
