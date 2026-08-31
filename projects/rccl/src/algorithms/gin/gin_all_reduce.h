/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the GIN-SDMA AllReduce path launched from ncclAllReduce
 * when symmetric windows are used. By default only messages >= 256 MiB
 * (GIN two-shot) take this path; smaller messages use DDA AllReduce.
 * RCCL_GIN_ALLREDUCE_FORCE_ENABLE=1 also enables LSA one-shot (<= 8 MiB) and
 * LSA two-shot ((8 MiB, 256 MiB)).
 * See LICENSE.txt for license information.
 ******************************************************************************/

#ifndef GIN_ALL_REDUCE_H_
#define GIN_ALL_REDUCE_H_

#include "gin_all_reduce_policy.h"
#include "nccl.h"
#include "nccl_device.h"

struct ncclComm;

// Size and CTA constants live in gin_all_reduce_policy.h so host unit tests and
// ncclAllReduceGinSdmaEligible() share one definition.
// LSA one-shot for messages <= kGinAllReduceLsaOneShotMaxBytes (force-enable only).
// LSA two-shot for (8 MiB, 256 MiB) (force-enable only).
// GIN two-shot for messages >= kGinAllReduceGinTwoShotMinBytes (default path).

// Lazily created on the first eligible AllReduce and torn down with the comm.
// Declared unconditionally: ncclComm embeds this even when ENABLE_ROCSHMEM_GIN is off.
//
// Deliberately holds no per-launch host state. All cross-rank synchronization lives on the
// device (LSA barrier epochs in the resource window, GIN signals and their shadows) and is
// re-read by the kernel on every launch, which is what lets these collectives be captured
// into a graph and replayed: a host-side counter baked into a kernel argument at capture time
// would freeze at its captured value while the device counters kept advancing.
struct ncclGinAllReduceState {
  bool initialized;
  struct ncclDevComm devComm;
};

#if defined(ENABLE_ROCSHMEM_GIN)

// RCCL_GIN_ALLREDUCE_FORCE_ENABLE. Defined in gin_all_reduce_sdma.cu.
int64_t rcclParamGinAllReduceForceEnable();

bool ncclAllReduceGinSdmaEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                  ncclDataType_t datatype, ncclRedOp_t op);

// True when this AllReduce is a GIN-SDMA candidate that the default size policy
// left for DDA (message < 256 MiB and RCCL_GIN_ALLREDUCE_FORCE_ENABLE is not 1).
bool ncclAllReduceGinSdmaYieldToDda(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                    ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceGinSdma(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                  ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

ncclResult_t ncclGinAllReduceFinalize(ncclComm* comm);

#else

inline ncclResult_t ncclGinAllReduceFinalize(ncclComm*) { return ncclSuccess; }

#endif

#endif
