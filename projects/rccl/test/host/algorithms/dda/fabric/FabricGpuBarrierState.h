/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Read access to a FabricGpuBarrier's private state for the fabric DDA tests.
//
// The barrier is the value the host hands the kernels; its fields are private
// and only read on the device. Mirror the layout to check what was handed over.
// Tests give the three ints distinct values, so a reordered field fails rather
// than passes. The size assert does not catch a field appended after nRanks_:
// it lands in the interior padding before peerFlags_ and sizeof stays the same.

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
