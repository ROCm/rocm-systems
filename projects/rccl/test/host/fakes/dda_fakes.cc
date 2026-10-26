/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the DDA eligibility and block-count entry points defined by the
// production DDA translation units.

#include "dda_fakes.h"

#include "algorithms/dda/all_gather/dda_all_gather.h"
#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "algorithms/dda/alltoall/dda_alltoall.h"
#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"
#include "fail_loud.h"
#include "signature-drift.h"

#define DEFINE_DDA_REDUCTION_ELIGIBLE(hook, prod)                                                        \
  std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t)> g_##hook =     \
      [](ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t) { return false; };          \
  bool prod(ncclComm* comm, const void* send, void* recv, size_t count, ncclDataType_t type,             \
            ncclRedOp_t op) {                                                                            \
    return g_##hook(comm, send, recv, count, type, op);                                                   \
  }

#define DEFINE_DDA_ALLGATHER_ELIGIBLE(hook, prod)                                                         \
  std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t)> g_##hook =                   \
      [](ncclComm*, const void*, void*, size_t, ncclDataType_t) { return false; };                        \
  bool prod(ncclComm* comm, const void* send, void* recv, size_t count, ncclDataType_t type) {            \
    return g_##hook(comm, send, recv, count, type);                                                       \
  }

#define DEFINE_DDA_BLOCKS(hook, prod, sentinel)                                                           \
  std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_##hook =                                  \
      [](ncclComm*, size_t, ncclDataType_t) { return sentinel; };                                        \
  uint32_t prod(ncclComm* comm, size_t count, ncclDataType_t type) {                                     \
    return g_##hook(comm, count, type);                                                                   \
  }

DEFINE_DDA_REDUCTION_ELIGIBLE(allReduceDdaIpcEligible, ncclAllReduceDdaIpcEligible)
DEFINE_DDA_REDUCTION_ELIGIBLE(allReduceDdaFabricEligible, ncclAllReduceDdaFabricEligible)
DEFINE_DDA_REDUCTION_ELIGIBLE(allReduceDdaFabricLLEligible, ncclAllReduceDdaFabricLLEligible)
DEFINE_DDA_REDUCTION_ELIGIBLE(allReduceDdaFabricLL128Eligible, ncclAllReduceDdaFabricLL128Eligible)
DEFINE_DDA_BLOCKS(allReduceDdaIpcBlocks, ncclAllReduceDdaIpcBlocks, 111)
DEFINE_DDA_BLOCKS(allReduceDdaFabricBlocks, ncclAllReduceDdaFabricBlocks, 112)
DEFINE_DDA_BLOCKS(allReduceDdaFabricLLBlocks, ncclAllReduceDdaFabricLLBlocks, 113)
DEFINE_DDA_BLOCKS(allReduceDdaFabricLL128Blocks, ncclAllReduceDdaFabricLL128Blocks, 114)

DEFINE_DDA_ALLGATHER_ELIGIBLE(allGatherDdaIpcEligible, ncclAllGatherDdaIpcEligible)
DEFINE_DDA_ALLGATHER_ELIGIBLE(allGatherDdaFabricEligible, ncclAllGatherDdaFabricEligible)
DEFINE_DDA_ALLGATHER_ELIGIBLE(allGatherDdaFabricLLEligible, ncclAllGatherDdaFabricLLEligible)
DEFINE_DDA_ALLGATHER_ELIGIBLE(allGatherDdaFabricLL128Eligible, ncclAllGatherDdaFabricLL128Eligible)
DEFINE_DDA_BLOCKS(allGatherDdaIpcBlocks, ncclAllGatherDdaIpcBlocks, 121)
DEFINE_DDA_BLOCKS(allGatherDdaFabricBlocks, ncclAllGatherDdaFabricBlocks, 122)
DEFINE_DDA_BLOCKS(allGatherDdaFabricLLBlocks, ncclAllGatherDdaFabricLLBlocks, 123)
DEFINE_DDA_BLOCKS(allGatherDdaFabricLL128Blocks, ncclAllGatherDdaFabricLL128Blocks, 124)

// AlltoAll takes the same operand shape as AllGather (no reduction op), so it
// reuses that macro.
DEFINE_DDA_ALLGATHER_ELIGIBLE(allToAllDdaIpcEligible, ncclAllToAllDdaIpcEligible)
DEFINE_DDA_ALLGATHER_ELIGIBLE(allToAllDdaFabricEligible, ncclAllToAllDdaFabricEligible)
DEFINE_DDA_ALLGATHER_ELIGIBLE(allToAllDdaFabricLLEligible, ncclAllToAllDdaFabricLLEligible)
DEFINE_DDA_ALLGATHER_ELIGIBLE(allToAllDdaFabricLL128Eligible, ncclAllToAllDdaFabricLL128Eligible)
DEFINE_DDA_BLOCKS(allToAllDdaIpcBlocks, ncclAllToAllDdaIpcBlocks, 141)
DEFINE_DDA_BLOCKS(allToAllDdaFabricBlocks, ncclAllToAllDdaFabricBlocks, 142)
DEFINE_DDA_BLOCKS(allToAllDdaFabricLLBlocks, ncclAllToAllDdaFabricLLBlocks, 143)
DEFINE_DDA_BLOCKS(allToAllDdaFabricLL128Blocks, ncclAllToAllDdaFabricLL128Blocks, 144)

DEFINE_DDA_REDUCTION_ELIGIBLE(reduceScatterDdaIpcEligible, ncclReduceScatterDdaIpcEligible)
DEFINE_DDA_REDUCTION_ELIGIBLE(reduceScatterDdaFabricEligible, ncclReduceScatterDdaFabricEligible)
DEFINE_DDA_REDUCTION_ELIGIBLE(reduceScatterDdaFabricLLEligible, ncclReduceScatterDdaFabricLLEligible)
DEFINE_DDA_REDUCTION_ELIGIBLE(reduceScatterDdaFabricLL128Eligible, ncclReduceScatterDdaFabricLL128Eligible)
// Unused today: rcclSelectReduceScatter does not publish DDA block counts.
DEFINE_DDA_BLOCKS(reduceScatterDdaIpcBlocks, ncclReduceScatterDdaIpcBlocks, 131)
DEFINE_DDA_BLOCKS(reduceScatterDdaFabricBlocks, ncclReduceScatterDdaFabricBlocks, 132)
DEFINE_DDA_BLOCKS(reduceScatterDdaFabricLLBlocks, ncclReduceScatterDdaFabricLLBlocks, 133)
DEFINE_DDA_BLOCKS(reduceScatterDdaFabricLL128Blocks, ncclReduceScatterDdaFabricLL128Blocks, 134)

#define DEFINE_DDA_LAUNCH(hook, prod)                                                                     \
  DdaLaunchHook g_##hook = [](const void*, void*, size_t, ncclDataType_t, ncclComm*,                     \
                              hipStream_t) -> ncclResult_t { FailLoudUnfaked("dda_fakes", #prod); };     \
  ncclResult_t prod(const void* send, void* recv, size_t count, ncclDataType_t type, ncclComm* comm,     \
                    hipStream_t stream) {                                                                \
    return g_##hook(send, recv, count, type, comm, stream);                                              \
  }                                                                                                      \
  ASSERT_HOOK_MATCHES_PROD(g_##hook, prod);

#define DEFINE_DDA_REDUCE_LAUNCH(hook, prod)                                                              \
  DdaReduceLaunchHook g_##hook = [](const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm*,  \
                                    hipStream_t) -> ncclResult_t { FailLoudUnfaked("dda_fakes", #prod); }; \
  ncclResult_t prod(const void* send, void* recv, size_t count, ncclDataType_t type, ncclRedOp_t op,     \
                    ncclComm* comm, hipStream_t stream) {                                                \
    return g_##hook(send, recv, count, type, op, comm, stream);                                          \
  }                                                                                                      \
  ASSERT_HOOK_MATCHES_PROD(g_##hook, prod);

DEFINE_DDA_LAUNCH(allGatherDdaIpc, ncclAllGatherDdaIpc)
DEFINE_DDA_LAUNCH(allGatherDdaFabric, ncclAllGatherDdaFabric)
DEFINE_DDA_LAUNCH(allGatherDdaFabricLL, ncclAllGatherDdaFabricLL)
DEFINE_DDA_LAUNCH(allGatherDdaFabricLL128, ncclAllGatherDdaFabricLL128)
DEFINE_DDA_LAUNCH(allToAllDdaIpc, ncclAllToAllDdaIpc)
DEFINE_DDA_LAUNCH(allToAllDdaFabric, ncclAllToAllDdaFabric)
DEFINE_DDA_LAUNCH(allToAllDdaFabricLL, ncclAllToAllDdaFabricLL)
DEFINE_DDA_LAUNCH(allToAllDdaFabricLL128, ncclAllToAllDdaFabricLL128)
DEFINE_DDA_REDUCE_LAUNCH(allReduceDdaIpc, ncclAllReduceDdaIpc)
DEFINE_DDA_REDUCE_LAUNCH(allReduceDdaFabric, ncclAllReduceDdaFabric)
DEFINE_DDA_REDUCE_LAUNCH(allReduceDdaFabricLL, ncclAllReduceDdaFabricLL)
DEFINE_DDA_REDUCE_LAUNCH(allReduceDdaFabricLL128, ncclAllReduceDdaFabricLL128)
DEFINE_DDA_REDUCE_LAUNCH(reduceScatterDdaIpc, ncclReduceScatterDdaIpc)
DEFINE_DDA_REDUCE_LAUNCH(reduceScatterDdaFabric, ncclReduceScatterDdaFabric)
DEFINE_DDA_REDUCE_LAUNCH(reduceScatterDdaFabricLL, ncclReduceScatterDdaFabricLL)
DEFINE_DDA_REDUCE_LAUNCH(reduceScatterDdaFabricLL128, ncclReduceScatterDdaFabricLL128)

#undef DEFINE_DDA_REDUCE_LAUNCH
#undef DEFINE_DDA_LAUNCH
#undef ASSERT_HOOK_MATCHES_PROD
#undef DEFINE_DDA_BLOCKS
#undef DEFINE_DDA_ALLGATHER_ELIGIBLE
#undef DEFINE_DDA_REDUCTION_ELIGIBLE
