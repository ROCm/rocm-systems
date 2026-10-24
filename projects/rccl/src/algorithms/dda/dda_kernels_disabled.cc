/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Stand-in host entry points used when DDA kernels are compiled out
 * (install.sh --disable-dda / -DENABLE_DDA=OFF). Eligibility is always false,
 * so collective selection falls through to another algorithm. A launch that
 * still arrives here is a caller bug and returns an error.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_gather/dda_all_gather.h"
#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "algorithms/dda/alltoall/dda_alltoall.h"
#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"

#include "debug.h"

namespace {

ncclResult_t ddaKernelsDisabled() {
  WARN("DDA kernels were disabled at compile time (install.sh --disable-dda)");
  return ncclInternalError;
}

}  // namespace

#define DDA_OFF_ELIGIBLE_RED(name) \
  bool name(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t) { return false; }

#define DDA_OFF_LAUNCH_RED(name) \
  ncclResult_t name(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm*, cudaStream_t) { \
    return ddaKernelsDisabled(); \
  }

#define DDA_OFF_ELIGIBLE(name) bool name(ncclComm*, const void*, void*, size_t, ncclDataType_t) { return false; }

#define DDA_OFF_LAUNCH(name) \
  ncclResult_t name(const void*, void*, size_t, ncclDataType_t, ncclComm*, cudaStream_t) { \
    return ddaKernelsDisabled(); \
  }

#define DDA_OFF_BLOCKS(name) uint32_t name(ncclComm*, size_t, ncclDataType_t) { return 0; }

DDA_OFF_ELIGIBLE_RED(ncclAllReduceDdaIpcEligible)
DDA_OFF_LAUNCH_RED(ncclAllReduceDdaIpc)
DDA_OFF_ELIGIBLE_RED(ncclAllReduceDdaFabricEligible)
DDA_OFF_LAUNCH_RED(ncclAllReduceDdaFabric)
DDA_OFF_ELIGIBLE_RED(ncclAllReduceDdaFabricLLEligible)
DDA_OFF_ELIGIBLE_RED(ddaLLArOneShotEligible)
DDA_OFF_ELIGIBLE_RED(ddaLLArTwoShotEligible)
DDA_OFF_ELIGIBLE_RED(ddaLL128ArOneShotEligible)
DDA_OFF_ELIGIBLE_RED(ddaLL128ArTwoShotEligible)
DDA_OFF_LAUNCH_RED(ncclAllReduceDdaFabricLL)
DDA_OFF_ELIGIBLE_RED(ncclAllReduceDdaFabricLL128Eligible)
DDA_OFF_LAUNCH_RED(ncclAllReduceDdaFabricLL128)
DDA_OFF_BLOCKS(ncclAllReduceDdaIpcBlocks)
DDA_OFF_BLOCKS(ncclAllReduceDdaFabricBlocks)
DDA_OFF_BLOCKS(ncclAllReduceDdaFabricLLBlocks)
DDA_OFF_BLOCKS(ncclAllReduceDdaFabricLL128Blocks)

DDA_OFF_ELIGIBLE(ncclAllGatherDdaIpcEligible)
DDA_OFF_LAUNCH(ncclAllGatherDdaIpc)
DDA_OFF_BLOCKS(ncclAllGatherDdaIpcBlocks)
DDA_OFF_BLOCKS(ncclAllGatherDdaFabricBlocks)
DDA_OFF_BLOCKS(ncclAllGatherDdaFabricLLBlocks)
DDA_OFF_BLOCKS(ncclAllGatherDdaFabricLL128Blocks)
DDA_OFF_ELIGIBLE(ncclAllGatherDdaFabricEligible)
DDA_OFF_LAUNCH(ncclAllGatherDdaFabric)
DDA_OFF_ELIGIBLE(ncclAllGatherDdaFabricLLEligible)
DDA_OFF_LAUNCH(ncclAllGatherDdaFabricLL)
DDA_OFF_ELIGIBLE(ncclAllGatherDdaFabricLL128Eligible)
DDA_OFF_LAUNCH(ncclAllGatherDdaFabricLL128)

DDA_OFF_ELIGIBLE_RED(ncclReduceScatterDdaIpcEligible)
DDA_OFF_LAUNCH_RED(ncclReduceScatterDdaIpc)
DDA_OFF_ELIGIBLE_RED(ncclReduceScatterDdaFabricEligible)
DDA_OFF_LAUNCH_RED(ncclReduceScatterDdaFabric)
DDA_OFF_ELIGIBLE_RED(ncclReduceScatterDdaFabricLLEligible)
DDA_OFF_LAUNCH_RED(ncclReduceScatterDdaFabricLL)
DDA_OFF_ELIGIBLE_RED(ncclReduceScatterDdaFabricLL128Eligible)
DDA_OFF_LAUNCH_RED(ncclReduceScatterDdaFabricLL128)

DDA_OFF_ELIGIBLE(ncclAllToAllDdaIpcEligible)
DDA_OFF_LAUNCH(ncclAllToAllDdaIpc)
DDA_OFF_BLOCKS(ncclAllToAllDdaIpcBlocks)
DDA_OFF_BLOCKS(ncclAllToAllDdaFabricBlocks)
DDA_OFF_BLOCKS(ncclAllToAllDdaFabricLLBlocks)
DDA_OFF_BLOCKS(ncclAllToAllDdaFabricLL128Blocks)
DDA_OFF_ELIGIBLE(ncclAllToAllDdaFabricEligible)
DDA_OFF_LAUNCH(ncclAllToAllDdaFabric)
DDA_OFF_ELIGIBLE(ncclAllToAllDdaFabricLLEligible)
DDA_OFF_LAUNCH(ncclAllToAllDdaFabricLL)
DDA_OFF_ELIGIBLE(ncclAllToAllDdaFabricLL128Eligible)
DDA_OFF_LAUNCH(ncclAllToAllDdaFabricLL128)

#undef DDA_OFF_ELIGIBLE_RED
#undef DDA_OFF_LAUNCH_RED
#undef DDA_OFF_ELIGIBLE
#undef DDA_OFF_LAUNCH
#undef DDA_OFF_BLOCKS
