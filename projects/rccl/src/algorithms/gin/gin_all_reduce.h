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

constexpr size_t kGinAllReduceTwoShotSyncBytes = 16;
constexpr size_t kGinAllReduceMinPutBytes = 128;

// Lazily created on the first eligible AllReduce and torn down with the comm.
// Declared unconditionally: ncclComm embeds this even when ENABLE_ROCSHMEM_GIN is off.
struct ncclGinAllReduceState {
  bool initialized;
  struct ncclDevComm devComm;
  uint64_t* twoShotSync; // device [reduceDoneSync, agDoneSync]
  uint64_t twoShotReduceEpoch; // host shadow: next reduce-phase target base
  uint64_t twoShotAgEpoch;       // host shadow: next AG-phase target base
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
