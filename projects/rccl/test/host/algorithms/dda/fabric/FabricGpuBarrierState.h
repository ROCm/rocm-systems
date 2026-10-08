/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Mirrors FabricGpuBarrier's private layout so tests can read it.

#ifndef RCCL_TEST_HOST_ALGORITHMS_DDA_FABRIC_FABRICGPUBARRIERSTATE_H_
#define RCCL_TEST_HOST_ALGORITHMS_DDA_FABRIC_FABRICGPUBARRIERSTATE_H_

#include <cstring>
#include <type_traits>

#include "algorithms/dda/fabric/fabric_gpu_barrier.h"

struct FabricGpuBarrierState {
  int nBlocks;
  int selfRank;
  int nRanks;
  dda::common::FabricGpuBarrier::FlagType** peerFlags;
};
static_assert(sizeof(FabricGpuBarrierState) == sizeof(dda::common::FabricGpuBarrier),
              "FabricGpuBarrier layout changed");
static_assert(std::is_standard_layout<dda::common::FabricGpuBarrier>::value,
              "FabricGpuBarrier layout is no longer fixed");
static_assert(std::is_trivially_copyable<dda::common::FabricGpuBarrier>::value,
              "FabricGpuBarrier is no longer a plain value");

inline FabricGpuBarrierState StateOf(const dda::common::FabricGpuBarrier& barrier) {
  FabricGpuBarrierState state;
  std::memcpy(&state, &barrier, sizeof(state));
  return state;
}

#endif  // RCCL_TEST_HOST_ALGORITHMS_DDA_FABRIC_FABRICGPUBARRIERSTATE_H_
