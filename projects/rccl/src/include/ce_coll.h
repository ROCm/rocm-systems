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
#include "sym_kernels.h"

// Memory operations per rank for different synchronization protocols
#define NCCL_CE_SYNC_OPS_PER_RANK_MC 2
#define NCCL_CE_SYNC_OPS_PER_RANK_UC 3
#define RCCL_CE_NUM_COPY_STREAMS 8

// Default ceARTmpBuf capacity. Independent of the 2-shot table cap
// (ceNonRegMax[AllReduce]): registered CE always needs a staging buffer, and
// 2-shot only grows this when its table/env size exceeds the default.
#ifndef NCCL_CE_AR_TMPBUF_DEFAULT_BYTES
#define NCCL_CE_AR_TMPBUF_DEFAULT_BYTES (256ULL * 1024 * 1024)
#endif

#ifndef NCCL_CE_REDUCE_MAX_BLOCKS
#define NCCL_CE_REDUCE_MAX_BLOCKS 46
#endif

#ifndef NCCL_CE_NUM_SLOTS
#define NCCL_CE_NUM_SLOTS 2
#endif

// Per-rank slot size in ceARTmpBuf. The host scatter addresses slots in bytes
// (rank * slotChunkBytes) while the reduce kernel addresses them in elements
// (rank * slotChunkElems), so a slot must hold a whole number of elements. 16 is
// a multiple of every supported element size and also keeps each rank boundary
// aligned for the kernel's 16B vector loads, so one round-down satisfies both.
inline size_t ncclCeAllReduceSlotChunkBytes(size_t maxChunkBytes) {
  return alignDown(maxChunkBytes, (size_t)16);
}

// Chunk size for the pipelined path, i.e. when a shard does not fit in one slot.
// A chunk must fit its slot, and needs the same 16B rounding as the slot itself:
// the host walks chunks in bytes (ch * chunkBytes) while the kernel walks them in
// elements (ch * baseChunkElems).
inline size_t ncclCeAllReduceChooseChunkBytes(size_t shardBytes, size_t slotChunkBytes) {
  const size_t MIN_CHUNK_BYTES = 4 * 1024 * 1024ULL;
  const size_t MAX_CHUNK_BYTES = 256 * 1024 * 1024ULL;
  size_t targetChunkBytes = shardBytes / 4;
  if (targetChunkBytes > MAX_CHUNK_BYTES) targetChunkBytes = MAX_CHUNK_BYTES;
  if (targetChunkBytes < MIN_CHUNK_BYTES) targetChunkBytes = MIN_CHUNK_BYTES;
  if (targetChunkBytes > slotChunkBytes) targetChunkBytes = slotChunkBytes;
  return alignDown(targetChunkBytes, (size_t)16);
}

struct ncclCeColl {
  uint8_t* baseUCSymReadyPtr;
  uint8_t* baseUCSymComplPtr;
  size_t baseUCSymReadyOffset;
  size_t baseUCSymComplOffset;
  uint32_t ceSeqNum;
  // Device buffer sourcing the UC barrier flag value. Slot [0]: running seq
  // (stored per non-capture barrier); slot [1]: constant GRAPH_SYNC_VALUE used
  // during graph capture. Peer writes memcpy from here so they can be issued
  // as a separate stream op ahead of the wait/reset batch (see ncclPrepUCSync).
  uint32_t* ceSeqNumDev;
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
  uint8_t* ceARTmpBuf;
  struct ncclDevrWindow* ceARTmpWin;
  size_t ceArMaxBytes;   // allocated ceARTmpBuf capacity (default, grown if 2-shot cap is larger)
  uint32_t* signalBuffer;
  struct ncclDevrWindow* signalWin;
  // Global counter barrier for regular launch: [0]=arrival, [1]=completed generation.
  uint32_t* d_barrierSync;
  cudaStream_t scatterStream;
  cudaEvent_t synceEvent;  // join scatterStream back onto the caller's stream
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

  // AlltoAllv: [sendSizes, sendDispls, recvSizes, recvDispls] x nRanks (bytes).
  size_t* sizes;

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

ncclResult_t ncclCeAlltoAllv(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclAlltoAllvValidatePeerSendSize(size_t sendBytes, size_t peerRecvBytes, int srcRank, int dstRank);

bool ncclCeAlltoAllvEligible(struct ncclComm* comm, ncclDataType_t datatype, ncclSymRegType_t winRegType,
                             bool hasSysmemSegment, bool capturing);

ncclResult_t ncclHierCeAllGather(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);

ncclResult_t ncclHierCeAlltoAll(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);

// CE AllReduce: scatter → local-reduce → allgather (→ optional copy-to-user-recvbuff).
// Requires comm->ceColl.ceARTmpBuf != NULL (i.e. ncclCeInit has run).
ncclResult_t ncclCeAllReduce(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                             ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream,
                             struct ncclDevrWindow* recvWin = nullptr);

// Reduce-kernel block count for a per-rank chunk of `chunkElems` elements
// (chunkElems = count / nRanks). Mirrors the geometry ncclCeLaunchLocalReduce
// launches; for host-side impl-selection reporting. Returns 0 if chunkElems==0.
int ncclCeLocalReduceBlocks(ncclDataType_t datatype, size_t chunkElems);
#endif /* NCCL_CE_COLL_H_ */
