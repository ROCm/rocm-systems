/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the GIN-SDMA AllReduce path selected by rcclSelectAllReduce
 * when symmetric windows are used. By default gfx950 takes GIN two-shot for
 * messages >= 256 MiB; gfx1250 with Anvil SDMA takes GinScatter from 64 MiB.
 * Smaller messages use DDA AllReduce.
 * RCCL_GIN_ALLREDUCE_FORCE_ENABLE=1 also enables LSA one-shot (<= 4 MiB) and
 * LSA two-shot ((4 MiB, GIN floor)).
 * See LICENSE.txt for license information.
 ******************************************************************************/

#ifndef GIN_ALL_REDUCE_H_
#define GIN_ALL_REDUCE_H_

#include "gin_all_reduce_policy.h"
#include "nccl.h"
#include "nccl_device.h"

#include <stdint.h>

struct ncclComm;
struct ncclDevrWindow;

// Size and CTA constants live in gin_all_reduce_policy.h so host unit tests and
// ncclAllReduceGinSdmaEligible() share one definition.
// LSA one-shot for messages <= kGinAllReduceLsaOneShotMaxBytes (force-enable only).
// LSA two-shot between 4 MiB and the GIN floor (force-enable only).
// gfx950 + Anvil SDMA: GIN two-shot for messages >= kGinAllReduceGinTwoShotMinBytes (256 MiB).
// gfx1250 + Anvil SDMA: GinScatter for messages >= kGinAllReduceGinScatterMinBytes (64 MiB).

// Lazily created on the first eligible AllReduce and torn down with the comm.
// Declared unconditionally: ncclComm embeds this even when ENABLE_ROCSHMEM_GIN is off.
//
// Deliberately holds no per-launch host epoch. All cross-rank synchronization lives on the
// device (LSA barrier epochs in the resource window, GIN signals and their shadows) and is
// re-read by the kernel on every launch, which is what lets these collectives be captured
// into a graph and replayed: a host-side counter baked into a kernel argument at capture time
// would freeze at its captured value while the device counters kept advancing.
// intraGpuCtaBar is two device uint32s (arrived, sense) for the GIN two-shot intra-GPU
// CTA barrier; the pointer is stable, the words themselves advance on the device.
// scratch / scratchWin are the symmetric staging buffer for GinScatter (512 MiB of
// CE-style ping-pong incoming slots). Allocated and window-registered once on
// gfx1250 before GIN connect.
struct ncclGinAllReduceState {
  bool initialized;
  struct ncclDevComm devComm;
  uint32_t* intraGpuCtaBar;
  void* scratch;
  size_t scratchBytes;
  struct ncclDevrWindow* scratchWin;
};

#if defined(ENABLE_ROCSHMEM_GIN)

// RCCL_GIN_ALLREDUCE_FORCE_ENABLE. Defined in gin_all_reduce_sdma.cu.
int64_t rcclParamGinAllReduceForceEnable();

bool ncclAllReduceGinSdmaEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                  ncclDataType_t datatype, ncclRedOp_t op);

// True when this AllReduce is a GIN-SDMA candidate that the default size policy
// left for DDA (message below the GIN floor and RCCL_GIN_ALLREDUCE_FORCE_ENABLE is not 1).
bool ncclAllReduceGinSdmaYieldToDda(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                    ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceGinSdma(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                  ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

ncclResult_t ncclGinAllReduceFinalize(ncclComm* comm);

#else

inline ncclResult_t ncclGinAllReduceFinalize(ncclComm*) { return ncclSuccess; }

#endif

#endif
