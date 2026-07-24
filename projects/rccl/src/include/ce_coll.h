/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CE_COLL_H_
#define NCCL_CE_COLL_H_

#include "nccl.h"
#include "nccl_common.h"
#include "bitops.h"

// Memory operations per rank for different synchronization protocols
#define NCCL_CE_SYNC_OPS_PER_RANK_MC 2
#define NCCL_CE_SYNC_OPS_PER_RANK_UC 3
#define RCCL_CE_NUM_COPY_STREAMS 8

// CE AllReduce: maximum supported total AllReduce message size.
// The ceARTmpBuf is sized to hold nRanks scatter slots (totalBytes) plus
// one reduce-scratch slot (chunkBytes = totalBytes/nRanks), so the buffer
// must satisfy: totalBytes * (nRanks+1)/nRanks <= ceARTmpBufSize.
//
// Eligible message sizes are gated to the
// [NCCL_CE_AR_MIN_MSG_BYTES, NCCL_CE_AR_MAX_MSG_BYTES] range below
// (default 4 MiB .. 256 MiB).
#define NCCL_CE_AR_MAX_MSG_BYTES (256ull * 1024 * 1024)
#define NCCL_CE_AR_MIN_MSG_BYTES (4ull * 1024 * 1024)

// 0 (default): CE AllReduce reduce kernels are Sum-only.
// 1: also build/run Prod/Min/Max (BUILD_CE_REDUCE_ALL_OPS=ON).
#ifndef NCCL_CE_REDUCE_ALL_OPS
#define NCCL_CE_REDUCE_ALL_OPS 0
#endif

struct ncclCeColl {
  uint8_t* baseUCSymReadyPtr;
  uint8_t* baseUCSymComplPtr;
  size_t baseUCSymReadyOffset;
  size_t baseUCSymComplOffset;
  uint32_t ceSeqNum;
  bool useCompletePtr;
  uint32_t intraBatchSyncFreq;
  uint64_t intraBatchSyncMsgThreshold;
  struct ncclDevrWindow* ceSyncWin;
  int nCopyStreams;
  cudaStream_t copyStreams[RCCL_CE_NUM_COPY_STREAMS];
  cudaEvent_t copyEvents[RCCL_CE_NUM_COPY_STREAMS];
#ifdef ENABLE_FAULT_INJECTION
  uint32_t ceFaults;  // bitmask of CE_FAULT_* bits; see ce_fault_inject.h
#endif

  // CE AllReduce staging buffer (symmetric, size = nRanks*maxChunk + maxChunk).
  // Layout: [0 .. nRanks*chunkBytes) scatter staging,
  //         [nRanks*chunkBytes .. (nRanks+1)*chunkBytes) reduce scratch.
  uint8_t* ceARTmpBuf;
  struct ncclDevrWindow* ceARTmpWin;

  // Latched while this comm has live graph-captured plans. CE 2-shot AllReduce
  // can deadlock on eager calls that share a graph-mode comm, so we disable CE
  // AR during that period and re-enable it after captured plans are reclaimed.
  // Written only from rcclCeAllReduceGraphLatchTick(); no internal lock, same
  // single-writer-per-comm contract as localPersistentRefs (comm.h).
  bool graphModeSeen;
};

struct ncclCeInitTask {
  struct ncclCeInitTask* next;
  struct ncclComm* comm;
};

struct alignas(16) ncclCeCollArgs {
  ncclFunc_t func;
  int rootRank;
  ncclDataType_t datatype;
  size_t nElts;
  size_t eltSize;
  uint8_t* sendBuff;
  uint8_t* recvBuff;
  struct ncclDevrWindow* sendWin;
  struct ncclDevrWindow* recvWin;
  void* collApiEventHandle;  // Parent API event handle for profiler hierarchy
  void* ceCollProfHandle;    // CE collective profiler event handle
  bool useDda;
  void** ddaPeerBases;      // host-side table of every rank's DDA scratch base pointer
  void*
    ddaUserRecvBuff; // user recvbuff (using DDA staging) or NULL otherwise (if recvbuffer is using symmetric windows)
  size_t ddaCopyBackBytes; // bytes to copy scratch -> user recvbuff
  ncclRedOp_t redOp; // Only used for AllReduce
};

struct ncclCeBatchOpsParams {
  void** dsts;
  void** srcs;
  size_t* sizes;
  size_t numOps;
  bool intraBatchSync;
#ifdef CE_BATCH_ASYNC_SUPPORTED
  hipMemcpyAttributes* attrs;
  size_t* attrIdxs;
  size_t numAttrs;
#endif
};

bool ncclCeAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                     ncclSymRegType_t winRegType);

bool ncclCeScratchAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                            ncclSymRegType_t winRegType);

bool ncclCeImplemented(ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty);

ncclResult_t ncclCeInit(struct ncclComm* comm);

ncclResult_t ncclCeFinalize(struct ncclComm* comm);

ncclResult_t ncclMemOpSync(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclLaunchCeColl(struct ncclComm* comm, struct ncclKernelPlan* plan);

ncclResult_t ncclCeAllGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclCeScatter(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclCeGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclCeAlltoAll(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

// CE AllReduce: scatter → local-reduce → allgather (→ optional copy-to-user-recvbuff).
// Requires comm->ceColl.ceARTmpBuf != NULL (i.e. ncclCeInit has run).
ncclResult_t ncclCeAllReduce(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                             ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream,
                             struct ncclDevrWindow* recvWin = nullptr);
#endif /* NCCL_CE_COLL_H_ */
