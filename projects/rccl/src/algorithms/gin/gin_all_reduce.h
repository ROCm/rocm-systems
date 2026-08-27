/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the GIN-SDMA AllReduce path (LSA one-shot <= 16 MiB,
 * LSA two-shot > 16 MiB, GIN two-shot >= 128 MiB) launched from ncclAllReduce
 * when symmetric windows are used.
 * See LICENSE.txt for license information.
 ******************************************************************************/

#ifndef GIN_ALL_REDUCE_H_
#define GIN_ALL_REDUCE_H_

#include "nccl.h"
#include "nccl_device.h"

struct ncclComm;

// LSA one-shot for messages <= kGinAllReduceLsaOneShotMaxBytes.
// LSA two-shot for (16 MiB, 128 MiB); GIN two-shot for messages >= kGinAllReduceGinTwoShotMinBytes.
constexpr int kGinAllReduceLsaCtas = 56;
constexpr int kGinAllReduceLsaTwoShotCtasPerPeer = 8;
constexpr int kGinAllReduceLsaTwoShotMaxCtas = kGinAllReduceLsaTwoShotCtasPerPeer * 16;

constexpr int kGinAllReduceMinBytes = 512ULL * 1024;
constexpr int kGinAllReduceLsaThreadsPerCta = 512;
constexpr size_t kGinAllReduceLsaOneShotMaxBytes = 8ULL * 1024 * 1024;
constexpr size_t kGinAllReduceLsaTwoShotMidBytes = 32ULL * 1024 * 1024;
constexpr size_t kGinAllReduceGinTwoShotMinBytes = 256ULL * 1024 * 1024;

constexpr size_t kGinAllReduceMinPutBytes = 128;

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

bool ncclAllReduceGinSdmaEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                  ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceGinSdma(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                  ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

ncclResult_t ncclGinAllReduceFinalize(ncclComm* comm);

#else

inline ncclResult_t ncclGinAllReduceFinalize(ncclComm*) { return ncclSuccess; }

#endif

#endif
