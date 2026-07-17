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

// Default is <= 2 MiB (holds NUM_SLOTS * nRanks chunks (2 scatter slots), 
// and the reduced output goes to the user recvbuff)
#define NCCL_CE_AR_MAX_MSG_BYTES  (256ull * 1024 * 1024)

#ifndef NCCL_CE_REDUCE_MAX_BLOCKS
#define NCCL_CE_REDUCE_MAX_BLOCKS 46
#endif

#ifndef NCCL_CE_NUM_SLOTS
#define NCCL_CE_NUM_SLOTS 2
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

  // CE AllReduce staging buffer (symmetric), double-buffered scatter staging:
  // Layout: [slot 0: nRanks chunks][slot 1: nRanks chunks], slot stride = nRanks*chunkBytes.
  // The reduced result is written straight into the user recvbuff (no scratch).
  uint8_t*               ceARTmpBuf;
  struct ncclDevrWindow* ceARTmpWin;
  uint32_t* signalBuffer;
  struct ncclDevrWindow* signalWin;
  // Global counter barrier for regular launch: [0]=arrival, [1]=completed generation.
  uint32_t* d_barrierSync;
  cudaStream_t scatterStream;     // trails the reduce kernel: waits d_reduceDone, then all-gathers
  cudaEvent_t  synceEvent;  // join scatterStream back onto the caller's stream
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
  ncclRedOp_t redOp;         // Only used for AllReduce
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
ncclResult_t ncclCeAllReduce(struct ncclComm* comm, const void* sendbuff,
                              void* recvbuff, size_t count,
                              ncclDataType_t datatype, ncclRedOp_t op,
                              cudaStream_t stream,
                              struct ncclDevrWindow* recvWin = nullptr);
#endif /* NCCL_CE_COLL_H_ */
