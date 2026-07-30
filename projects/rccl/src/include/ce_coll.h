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
  void* ceCollProfHandle;     // CE collective profiler event handle
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

bool ncclHierCeAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                         ncclSymRegType_t winRegType);

ncclResult_t ncclCeInit(struct ncclComm* comm);

ncclResult_t ncclCeFinalize(struct ncclComm* comm);

// Intra-LSA-rank barrier.
ncclResult_t ncclMemOpSync(struct ncclComm* comm, cudaStream_t stream, struct ncclCeCollArgs* profilerArgs = nullptr);

// Allocate / free internal arrays for a batch-ops parameter struct.
ncclResult_t ncclCeInitBatchOpsParams(struct ncclCeBatchOpsParams* params, int capacity);
void ncclCeFreeBatchOpsParams(struct ncclCeBatchOpsParams* params);

// Launch a batch of cudaMemcpyAsync ops
ncclResult_t ncclCeLaunchBatchOps(struct ncclComm* comm, struct ncclCeBatchOpsParams* params, cudaStream_t stream,
                                  struct ncclCeCollArgs* profilerArgs = nullptr);

ncclResult_t ncclLaunchCeColl(struct ncclComm* comm, struct ncclKernelPlan* plan);

ncclResult_t scheduleCeCollTaskToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan);

ncclResult_t ncclCeAllGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclCeScatter(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclCeGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclCeAlltoAll(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclHierCeAllGather(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);

ncclResult_t ncclHierCeAlltoAll(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
#endif /* NCCL_CE_COLL_H_ */
