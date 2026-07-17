/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "register_inline.h"
#include <algorithm>
#include <atomic>
#include <cuda.h>
#include "rocmwrap.h"
#include "ce_coll.h"
#include "alloc.h"
#include "ce_fault_inject.h"

#ifdef ENABLE_FAULT_INJECTION
// Common fault check helper
static ncclResult_t ceFaultCheck(struct ncclComm* comm, uint32_t bit, const char* fnName) {
  if (comm->ceColl.ceFaults & bit) {
    WARN("CE: fault injection: %s returning ncclSystemError (rank %d)", fnName, comm->rank);
    return ncclSystemError;
  }
  return ncclSuccess;
}
#endif

RCCL_PARAM(CeMultiStreams, "CE_MULTI_STREAMS", 0);
RCCL_PARAM(CeBatchAsyncEnable, "CE_BATCH_ASYNC_ENABLE", -2);

#ifdef CE_BATCH_ASYNC_SUPPORTED
// Runtime detection: does the running driver actually implement hipMemcpyBatchAsync?
// Window (native 7.12 OR 7.0.2.x backport) is defined once in rocmwrap.h.
static int ncclCeBatchAsyncSupported() {
  int driverVersion;
  if (ncclCudaDriverVersion(&driverVersion) != ncclSuccess) return 0;
  return NCCL_CE_BATCH_ASYNC_VERSION_SUPPORTED(driverVersion);
}
#endif

static int ncclCeBatchAsyncEnable() {
  // Called once per CE collective; warn at most once to avoid flooding the log.
  static std::atomic<bool> warnedUnsupported{false};
#ifdef CE_BATCH_ASYNC_SUPPORTED
  int param = rcclParamCeBatchAsyncEnable();
  int supported = ncclCeBatchAsyncSupported();
  if (param > 0 && !supported) {
    if (!warnedUnsupported.exchange(true))
      WARN("RCCL_CE_BATCH_ASYNC_ENABLE=1 is set but hipMemcpyBatchAsync is not supported at runtime; disabling CE "
           "batch path");
    return 0;
  }
  return param >= 0 ? param : (param == -2 && supported);
#else
  if (rcclParamCeBatchAsyncEnable() > 0 && !warnedUnsupported.exchange(true))
    WARN("RCCL_CE_BATCH_ASYNC_ENABLE=1 is set but CE batch API not available; disabling");
  return 0;
#endif
}
// Static constant for graph synchronization
static const uint32_t GRAPH_SYNC_VALUE = 1;

// Static constants for intra-batch synchronization to improve CE collective performance with large scale
// Frequency of intra-batch synchronization
static const uint32_t CE_COLL_INTRA_BATCH_SYNC_FREQ = 8;
// Message threshold for intra-batch synchronization
static const uint64_t CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD = 512 * 1024 * 1024;

static void ceDestroyCopyStreams(struct ncclComm* comm, int nPairs) {
  for (int j = 0; j < nPairs; j++) {
    CUDACHECKIGNORE(cudaEventDestroy(comm->ceColl.copyEvents[j]));
    CUDACHECKIGNORE(cudaStreamDestroy(comm->ceColl.copyStreams[j]));
  }
  comm->ceColl.nCopyStreams = 0;
}

ncclResult_t ncclCeInit(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

#ifdef ENABLE_FAULT_INJECTION
  NCCLCHECK(ceFaultCheck(comm, CE_FAULT_INIT, "ncclCeInit"));
#endif

  uint8_t* ceDevBase = nullptr;
  size_t ceDevBaseSize = alignUp(comm->nRanks * sizeof(uint32_t), 16) * 2;
  ncclWindow_vidmem* ceWinDev = nullptr;
  ncclWindow_vidmem* ceWinDevHost = nullptr;
  int i = 0;
  int targetStreams = 0;
  // Ensure symmetric memory runtime is initialized
  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);
  // Allocate and register memory for the symmetric memory
  NCCLCHECKGOTO(ncclMemAlloc((void**)&ceDevBase, ceDevBaseSize), ret, fail);
  NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, ceDevBase, ceDevBaseSize, NCCL_WIN_COLL_SYMMETRIC, &ceWinDev), ret,
                fail);
  NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, ceWinDev, &ceWinDevHost), ret, fail);
  // Get the ncclDevrWindow from the winHost field
  comm->ceColl.ceSyncWin = (struct ncclDevrWindow*)ceWinDevHost->winHost;

  comm->ceColl.baseUCSymReadyOffset = 0;
  comm->ceColl.baseUCSymComplOffset = alignUp(comm->nRanks * sizeof(uint32_t), 16);
  comm->ceColl.baseUCSymReadyPtr = (uint8_t*)comm->ceColl.ceSyncWin->userPtr + comm->ceColl.baseUCSymReadyOffset;
  comm->ceColl.baseUCSymComplPtr = (uint8_t*)comm->ceColl.ceSyncWin->userPtr + comm->ceColl.baseUCSymComplOffset;
  comm->ceColl.ceSeqNum = 0;
  comm->ceColl.useCompletePtr = false;
  comm->ceColl.intraBatchSyncFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  comm->ceColl.intraBatchSyncMsgThreshold = CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD;
  comm->ceColl.nCopyStreams = 0;
  INFO(NCCL_INIT, "Init CE, rank %d baseUCSymReadyPtr %p, baseUCSymComplPtr %p, seq num %d", comm->rank,
       comm->ceColl.baseUCSymReadyPtr, comm->ceColl.baseUCSymComplPtr, comm->ceColl.ceSeqNum);
  {
    int multiStreams = rcclParamCeMultiStreams();
    if (multiStreams > 0) {
      targetStreams = std::min(multiStreams, (int)RCCL_CE_NUM_COPY_STREAMS);
      INFO(NCCL_INIT, "CE multi-stream enabled: rank %d using %d streams (requested=%d)", comm->rank, targetStreams,
           multiStreams);
      for (i = 0; i < targetStreams; i++) {
        CUDACHECKGOTO(cudaStreamCreateWithFlags(&comm->ceColl.copyStreams[i], cudaStreamNonBlocking), ret,
                      fail_ce_stream);
        CUDACHECKGOTO(cudaEventCreateWithFlags(&comm->ceColl.copyEvents[i], cudaEventDisableTiming), ret,
                      fail_ce_event);
        comm->ceColl.nCopyStreams++;
      }
    }
  }

exit:
  return ret;
fail_ce_event:
  CUDACHECKIGNORE(cudaStreamDestroy(comm->ceColl.copyStreams[i]));
fail_ce_stream:
  INFO(NCCL_INIT, "CE init failed on rank %d after creating %d/%d copy streams", comm->rank, i, targetStreams);
  ceDestroyCopyStreams(comm, i);
  goto fail;
fail:
  if (ceWinDev != nullptr) ncclCommWindowDeregister(comm, ceWinDev);
  if (ceDevBase != nullptr) ncclMemFree(ceDevBase);
  goto exit;
}

ncclResult_t ncclCeFinalize(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  // Clean up ceInitTaskQueue
  while (!ncclIntruQueueEmpty(&comm->ceInitTaskQueue)) {
    struct ncclCeInitTask* task = ncclIntruQueueDequeue(&comm->ceInitTaskQueue);
    free(task);
  }

  // Clean up CE resources
  if (comm->ceColl.baseUCSymReadyPtr != NULL) {
    if (comm->ceColl.ceSyncWin && comm->ceColl.ceSyncWin->vidmem) {
      NCCLCHECKGOTO(ncclCommWindowDeregister(comm, comm->ceColl.ceSyncWin->vidmem), ret, fail);
      NCCLCHECKGOTO(ncclMemFree(comm->ceColl.baseUCSymReadyPtr), ret, fail);
    }
    comm->ceColl.baseUCSymReadyPtr = NULL;
    comm->ceColl.baseUCSymComplPtr = NULL;
    comm->ceColl.ceSyncWin = NULL;
  }
  // Clean up copy streams and events
  ceDestroyCopyStreams(comm, comm->ceColl.nCopyStreams);

exit:
  return ret;
fail:
  // [RCCL] In ncclCeFinalize there are no ceWinDev/ceDevBase locals, so the
  // cleanup uses the comm->ceColl.* members directly. The NCCLCHECKIGNORE
  // helpers tolerate null pointers safely.
  goto exit;
}

bool ncclCeImplemented(ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty);

bool ncclCeAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                     ncclSymRegType_t winRegType) {
  if (!ncclCeImplemented(coll, red, ty)) {
    TRACE(NCCL_TUNING, "Skipping CE collective: not implemented");
    return false;
  }
  if (comm->nNodes > 1) {
    TRACE(NCCL_TUNING, "Skipping CE collective: comm is not a single node");
    return false;
  }
  if (!comm->symmetricSupport) {
    TRACE(NCCL_TUNING, "Skipping CE collective: symmetric support is not enabled");
    return false;
  }
  if (winRegType != ncclSymSendRegRecvReg && winRegType != ncclSymSendNonregRecvReg) {
    TRACE(NCCL_TUNING, "Skipping CE collective: window registration type %d is not supported", winRegType);
    return false;
  }
  return true;
}

bool ncclCeImplemented(ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
  int driverVersion;
  if (ncclCudaDriverVersion(&driverVersion) != ncclSuccess) return false;

  // CE is supported in ROCm 7.12+ and the 7.0.2.x range [7.0.2.2, 7.0.3.0).
  // hipDriverGetVersion() encodes as MAJOR*10000000 + MINOR*100000 + PATCH*1000 + BUILD;
  //   ROCm 7.12.0   → 71200000
  //   ROCm 7.0.2.2  → 70051831  (lower bound of the 7.0.2.x backport range)
  //   ROCm 7.0.3.0  → 70060000  (exclusive upper bound)
  if (driverVersion >= 71200000 || (driverVersion >= 70051831 && driverVersion < 70060000)) {
    switch (coll) {
    case ncclFuncAllGather:
    case ncclFuncAlltoAll:
    case ncclFuncScatter:
    case ncclFuncGather:
      return true;
    default:
      return false;
    }
  }
  return false;
}

ncclResult_t ncclPrepMCSync(struct ncclComm* comm, bool isComplete, hipStreamBatchMemOpParams* batchParams,
                            size_t* opIdx, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  uint32_t* readyPtrs = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);
  uint32_t currentSeq = ++comm->ceColl.ceSeqNum;

  // Source pointer is either the constant graph sync value or the sequence number
  void* srcPtr = capturing ? (void*)&GRAPH_SYNC_VALUE : (void*)&currentSeq;
  // Wait value is either the constant graph sync value or the sequence number
  uint32_t waitValue = capturing ? GRAPH_SYNC_VALUE : currentSeq;

  // Use multi-cast address as destination pointer
  void* mcDstPtr;
  void* dstPtr = isComplete ? (void*)&completePtrs[comm->rank] : (void*)&readyPtrs[comm->rank];
  size_t offset = (uint8_t*)dstPtr - (uint8_t*)comm->ceColl.ceSyncWin->userPtr;
  NCCLCHECKGOTO(ncclDevrGetLsaTeamPtrMC(comm, comm->ceColl.ceSyncWin, offset, ncclTeamLsa(comm), &mcDstPtr), ret, fail);

  // Write our own ready/complete flag to the multi-cast address
  CUDACHECKGOTO(cudaMemcpyAsync(mcDstPtr, srcPtr, sizeof(uint32_t), cudaMemcpyHostToDevice, stream), ret, fail);

  // Add local wait operations for every other rank
  for (int r = 0; r < comm->nRanks; ++r) {
    if (r == comm->rank) continue;
    batchParams[*opIdx] = {};
    batchParams[*opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    batchParams[*opIdx].waitValue.address = (CUdeviceptr)(isComplete ? (void*)&completePtrs[r] : (void*)&readyPtrs[r]);
    batchParams[*opIdx].waitValue.value = waitValue;
    batchParams[*opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    (*opIdx)++;
  }

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclPrepUCSync(struct ncclComm* comm, bool isComplete, hipStreamBatchMemOpParams* batchParams,
                            size_t* opIdx) {
  ncclResult_t ret = ncclSuccess;

#ifdef ENABLE_FAULT_INJECTION
  NCCLCHECK(ceFaultCheck(comm, CE_FAULT_SYNC_PREP, "ncclPrepUCSync"));
#endif

  uint32_t* readyPtrs = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);
  uint32_t currentSeq = ++comm->ceColl.ceSeqNum;

  // Write our own ready/complete flag to remote ranks
  uint32_t waitValue = capturing ? GRAPH_SYNC_VALUE : currentSeq;
  for (int r = 0; r < comm->nRanks; ++r) {
    if (r == comm->rank) continue;
    void* peerDstPtr;
    void* dstPtr = isComplete ? (void*)&completePtrs[comm->rank] : (void*)&readyPtrs[comm->rank];
    size_t offset = (uint8_t*)dstPtr - (uint8_t*)comm->ceColl.ceSyncWin->userPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, r, &peerDstPtr), ret, fail);
    batchParams[*opIdx] = {};
    batchParams[*opIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    batchParams[*opIdx].writeValue.address = (CUdeviceptr)peerDstPtr;
    batchParams[*opIdx].writeValue.value = waitValue;
    batchParams[*opIdx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
    (*opIdx)++;
  }

  // Add local wait operations for every other rank
  for (int r = 0; r < comm->nRanks; ++r) {
    if (r == comm->rank) continue;
    batchParams[*opIdx] = {};
    batchParams[*opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    batchParams[*opIdx].waitValue.address = (CUdeviceptr)(isComplete ? (void*)&completePtrs[r] : (void*)&readyPtrs[r]);
    batchParams[*opIdx].waitValue.value = capturing ? GRAPH_SYNC_VALUE : currentSeq;
    batchParams[*opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    (*opIdx)++;
  }

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclMemOpSync(struct ncclComm* comm, cudaStream_t stream, struct ncclCeCollArgs* args) {
  ncclResult_t ret = ncclSuccess;
  void* ceSyncHandle = NULL;
  int lsaSize = comm->devrState.lsaSize;

  // Get pointers to the ready and complete synchronization arrays
  uint32_t* readyPtrs = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  // Allocate enough slots for all possible ops
  size_t batchSize = (comm->nvlsSupport ? NCCL_CE_SYNC_OPS_PER_RANK_MC : NCCL_CE_SYNC_OPS_PER_RANK_UC) * comm->nRanks;
  size_t opIdx = 0;

  // Prepare batch memory operations for synchronization
  hipStreamBatchMemOpParams* batchParams = nullptr;
  NCCLCHECKGOTO(ncclCalloc(&batchParams, batchSize), ret, fail);

  if (comm->nvlsSupport) {
    NCCLCHECKGOTO(ncclPrepMCSync(comm, comm->ceColl.useCompletePtr, batchParams, &opIdx, stream), ret, fail);
  } else {
    NCCLCHECKGOTO(ncclPrepUCSync(comm, comm->ceColl.useCompletePtr, batchParams, &opIdx), ret, fail);
  }

  // For CUDA graph capture, add reset operation
  if (ncclCudaGraphValid(comm->planner.capturingGraph)) {
    for (int i = 0; i < lsaSize; i++) {
      batchParams[opIdx] = {};
      batchParams[opIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
      batchParams[opIdx].writeValue.address =
        (CUdeviceptr)(comm->ceColl.useCompletePtr ? (void*)&completePtrs[i] : (void*)&readyPtrs[i]);
      batchParams[opIdx].writeValue.value = 0;
      // CU_STREAM_WRITE_VALUE_DEFAULT is a CUDA-specific constant with no HIP equivalent.
      // This field must be initialized to satisfy the CUDA-compatible struct definition,
      // but the HIP runtime does not use this flag and treats it as 0.
      batchParams[opIdx].writeValue.flags = 0;
      opIdx++;
    }
  }

  // Execute all memory operations in a single batch
  CUCHECKGOTO(hipStreamBatchMemOp(stream, opIdx, batchParams, 0), ret, fail);

  // Toggle the flag for next call
  comm->ceColl.useCompletePtr = !comm->ceColl.useCompletePtr;

exit:
  if (batchParams) free(batchParams);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeInitBatchOpsParams(struct ncclCeBatchOpsParams* params, int nRanks) {
  ncclResult_t ret = ncclSuccess;

  params->srcs = nullptr;
  params->dsts = nullptr;
  params->sizes = nullptr;
  params->numOps = 0;
  params->intraBatchSync = false;
#ifdef CE_BATCH_ASYNC_SUPPORTED
  params->attrs = nullptr;
  params->attrIdxs = nullptr;
  params->numAttrs = 0;
#endif

  NCCLCHECKGOTO(ncclCalloc(&params->srcs, nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&params->dsts, nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&params->sizes, nRanks), ret, fail);
#ifdef CE_BATCH_ASYNC_SUPPORTED
  NCCLCHECKGOTO(ncclCalloc(&params->attrs, nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&params->attrIdxs, nRanks), ret, fail);
#endif
exit:
  return ret;
fail:
  goto exit;
}

void ncclCeFreeBatchOpsParams(struct ncclCeBatchOpsParams* params) {
  if (params->srcs) free(params->srcs);
  if (params->dsts) free(params->dsts);
  if (params->sizes) free(params->sizes);
#ifdef CE_BATCH_ASYNC_SUPPORTED
  if (params->attrs) free(params->attrs);
  if (params->attrIdxs) free(params->attrIdxs);
#endif
}

ncclResult_t ncclCeLaunchBatchOps(struct ncclComm* comm, struct ncclCeBatchOpsParams* params, cudaStream_t stream,
                                  struct ncclCeCollArgs* args) {
  ncclResult_t ret = ncclSuccess;
  bool capturing;
  void* ceBatchHandle = NULL;

#ifdef ENABLE_FAULT_INJECTION
  NCCLCHECK(ceFaultCheck(comm, CE_FAULT_LAUNCH_OP, "ncclCeLaunchBatchOps"));
#endif

  // cudaMemcpyBatchAsync does not accept the legacy null stream (e.g. PyTorch null stream).
  // Fall back to cudaMemcpyAsync per-op when stream is NULL.
  bool isLegacyStream;
  NCCLCHECKGOTO(ncclCudaStreamIsLegacyNull(stream, &isLegacyStream), ret, fail);

  // Start CE batch profiling
  NCCLCHECKGOTO(ncclProfilerStartCeBatchEvent(comm, args, params, stream, &ceBatchHandle), ret, fail);

  // Check if there are any operations to perform
  if (params->numOps == 0) goto exit;

  // Check if we are in a CUDA graph capture
  capturing = ncclCudaGraphValid(comm->planner.capturingGraph);

  //--------------Graph capture / legacy stream--------------
  // cudaMemcpyBatchAsync is not supported during CUDA graph capture or with the
  // legacy null stream (e.g. PyTorch's null stream); fall back to per-op cudaMemcpyAsync.
  if (capturing || isLegacyStream) {
    for (int i = 0; i < params->numOps; i++) {
      CUDACHECKGOTO(cudaMemcpyAsync((void*)params->dsts[i], (void*)params->srcs[i], params->sizes[i],
                                    cudaMemcpyDeviceToDevice, stream),
                    ret, fail);

      if (params->intraBatchSync && ((i + 1) % comm->ceColl.intraBatchSyncFreq == 0) && ((i + 1) < params->numOps)) {
        NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);
      }
    }
  }
  //--------------No graph capture--------------
  else {
#ifdef CE_BATCH_ASYNC_SUPPORTED
    if (ncclCeBatchAsyncEnable()) {
      params->attrs[0] = {};
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
      params->attrs[0].srcAccessOrder = hipMemcpySrcAccessOrderStream;
      params->attrs[0].flags = hipMemcpyFlagPreferOverlapWithCompute;
#else
      params->attrs[0].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
      params->attrs[0].flags = cudaMemcpyFlagPreferOverlapWithCompute;
#endif
      params->attrIdxs[0] = 0;
      params->numAttrs = 1;

      if (params->intraBatchSync) {
      // Break into multiple batches with sync between them
        int batchSize = comm->ceColl.intraBatchSyncFreq;
        for (int i = 0; i < params->numOps; i += batchSize) {
          int currentBatchSize = (i + batchSize <= params->numOps) ? batchSize : params->numOps - i;
          INFO(NCCL_COLL,
               "CE: rank %d -> Batch path with intraBatchSync (hipMemcpyBatchAsync, intraBatchSync), numOps=%zu, "
               "batchSize=%d",
               comm->rank, params->numOps, currentBatchSize);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
          CUDACHECKGOTO(hipMemcpyBatchAsync(
#else
          CUDACHECKGOTO(cudaMemcpyBatchAsync(
#endif
                          (void**)&params->dsts[i], (void**)&params->srcs[i], &params->sizes[i], currentBatchSize,
                          params->attrs, params->attrIdxs, params->numAttrs, nullptr, stream),
                        ret, fail);
        // Sync after each batch
          if (i + batchSize < params->numOps) {
            NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);
          }
        }
      } else {
      // Use single batch for all operations
        INFO(NCCL_COLL, "CE: rank %d -> Batch path without intraBatchSync (hipMemcpyBatchAsync), numOps=%zu",
             comm->rank, params->numOps);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        CUDACHECKGOTO(hipMemcpyBatchAsync(
#else
        CUDACHECKGOTO(cudaMemcpyBatchAsync(
#endif
                        (void**)params->dsts, (void**)params->srcs, params->sizes, params->numOps, params->attrs,
                        params->attrIdxs, params->numAttrs, nullptr, stream),
                      ret, fail);
      }
    } else  // CE batch async disabled — fall through to non-batch paths below
#endif // CE_BATCH_ASYNC_SUPPORTED
      if (comm->ceColl.nCopyStreams > 0 && (int)params->numOps > 1 && !params->intraBatchSync) {
        int nStreams = comm->ceColl.nCopyStreams;
        int activeStreams = ((int)params->numOps < nStreams) ? (int)params->numOps : nStreams;
        INFO(NCCL_COLL, "CE: rank %d -> No-Batch Multi-Stream path (%d streams), numOps=%zu", comm->rank, activeStreams,
             params->numOps);

      // Make copy streams wait on the main stream
        for (int s = 0; s < activeStreams; s++) {
          CUDACHECKGOTO(cudaEventRecord(comm->ceColl.copyEvents[s], stream), ret, fail);
          CUDACHECKGOTO(cudaStreamWaitEvent(comm->ceColl.copyStreams[s], comm->ceColl.copyEvents[s], 0), ret, fail);
        }

      // Distribute copies round-robin across streams
        for (int i = 0; i < (int)params->numOps; i++) {
          int s = i % activeStreams;
          CUDACHECKGOTO(cudaMemcpyAsync((void*)params->dsts[i], (void*)params->srcs[i], params->sizes[i],
                                        cudaMemcpyDeviceToDevice, comm->ceColl.copyStreams[s]),
                        ret, fail);
        }

      // Make main stream wait on all copy streams
        for (int s = 0; s < activeStreams; s++) {
          CUDACHECKGOTO(cudaEventRecord(comm->ceColl.copyEvents[s], comm->ceColl.copyStreams[s]), ret, fail);
          CUDACHECKGOTO(cudaStreamWaitEvent(stream, comm->ceColl.copyEvents[s], 0), ret, fail);
        }
      } else {
      // For older ROCm versions, fall back to individual transfers
        INFO(NCCL_COLL, "CE: rank %d -> No-Batch Single-Stream path (cudaMemcpyAsync), numOps=%zu", comm->rank,
             params->numOps);
        for (int i = 0; i < params->numOps; i++) {
          CUDACHECKGOTO(cudaMemcpyAsync((void*)params->dsts[i], (void*)params->srcs[i], params->sizes[i],
                                        cudaMemcpyDeviceToDevice, stream),
                        ret, fail);

          if (params->intraBatchSync && ((i + 1) % comm->ceColl.intraBatchSyncFreq == 0) &&
              ((i + 1) < params->numOps)) {
            NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);
          }
        }
      }
  }

exit:
  // Stop CE batch profiling - always attempt if started, even on error
  ncclProfilerStopCeBatchEvent(comm, ceBatchHandle, stream);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeAllGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff + myLsaRank * chunkBytes;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};

  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, lsaSize), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  // Copy own data to receive buffer if operation is out-of-place
  if (myRecvBuff != mySendBuff) {
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)myRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // Copy data to other ranks
  for (int r = 1; r < lsaSize; r++) {
    int targetRank = (myLsaRank + r) % lsaSize;
    offset = myRecvBuff - (uint8_t*)args->recvWin->userPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, targetRank, &peerRecvBuff), ret, fail);
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // Check if we need to perform intra-batch synchronization
  batchOpsParams.intraBatchSync = (batchOpsParams.numOps > comm->ceColl.intraBatchSyncFreq &&
                                   chunkBytes * batchOpsParams.numOps >= comm->ceColl.intraBatchSyncMsgThreshold);

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream, args), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

// AlltoAll across the LSA team (intra-node only).
ncclResult_t ncclCeAlltoAll(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  // Calculate the size of data each rank sends to every other rank
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, lsaSize), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  // Copy data to other ranks: send data chunk for each destination rank
  for (int r = 0; r < lsaSize; r++) {
    int dstRank = (myLsaRank + r) % lsaSize;
    uint8_t* srcPtr = mySendBuff + dstRank * chunkBytes;
    uint8_t* dstPtr = myRecvBuff + myLsaRank * chunkBytes;

    if (dstRank == myLsaRank) {
      // Local copy for own data
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    } else {
      // Remote copy to other ranks: send to rank dstRank's receive buffer at position comm->rank
      offset = dstPtr - (uint8_t*)args->recvWin->userPtr;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, dstRank, &peerRecvBuff), ret, fail);
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  }

  // Check if we need to perform intra-batch synchronization
  batchOpsParams.intraBatchSync = (batchOpsParams.numOps > comm->ceColl.intraBatchSyncFreq &&
                                   chunkBytes * batchOpsParams.numOps >= comm->ceColl.intraBatchSyncMsgThreshold);

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream, args), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

// Scatter across the LSA team (intra-node only).
ncclResult_t ncclCeScatter(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  // Calculate the size of data each rank sends to every other rank
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  int rootLsaRank;
  void* peerDstPtr;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, lsaSize), ret, fail);
  NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, args->rootRank, &rootLsaRank), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  if (myLsaRank == rootLsaRank) {
    // Check if this is an in-place scatter operation
    bool isInPlace = (myRecvBuff == mySendBuff + myLsaRank * chunkBytes);

    // Copy root's own data first if not in-place
    if (!isInPlace) {
      uint8_t* srcPtr = mySendBuff + myLsaRank * chunkBytes;
      uint8_t* dstPtr = myRecvBuff;
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }

    // Root rank distributes data to other ranks
    for (int r = 1; r < lsaSize; r++) {
      int dstRank = (myLsaRank + r) % lsaSize;
      uint8_t* srcPtr = mySendBuff + dstRank * chunkBytes;
      uint8_t* dstPtr = isInPlace ? myRecvBuff + dstRank * chunkBytes : myRecvBuff;

      offset = dstPtr - (uint8_t*)args->recvWin->userPtr;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, dstRank, &peerDstPtr), ret, fail);
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerDstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  }
  // Non-root ranks don't need to perform any copy operations

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream, args), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

// Gather across the LSA team (intra-node only).
ncclResult_t ncclCeGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int myLsaRank = comm->devrState.lsaSelf;
  // Calculate the size of data each rank sends to every other rank
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  int rootLsaRank;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, 1), ret, fail);
  NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, args->rootRank, &rootLsaRank), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  if (myLsaRank == rootLsaRank) {
    // Root rank copies its own data to the correct position in receive buffer
    uint8_t* dstPtr = myRecvBuff + myLsaRank * chunkBytes;
    if (mySendBuff != dstPtr) {
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  } else {
    // Non-root ranks send their data to root's receive buffer
    uint8_t* rootRecvPtr = (uint8_t*)args->recvBuff + myLsaRank * chunkBytes;
    offset = rootRecvPtr - (uint8_t*)args->recvWin->userPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, rootLsaRank, &peerRecvBuff), ret, fail);
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream, args), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

bool ncclHierCeAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                         ncclSymRegType_t winRegType) {
  if (!ncclCeImplemented(coll, red, ty)) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: not implemented");
    return false;
  }
  if (coll != ncclFuncAllGather && coll != ncclFuncAlltoAll) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: only AllGather and AlltoAll are supported");
    return false;
  }

  // Must be multi-node (single-node uses the regular CE path)
  if (comm->nNodes <= 1) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: not multi-node");
    return false;
  }
  // If LSA already spans the whole comm, use CE path instead
  if (ncclDevrIsOneLsaTeam(comm)) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: LSA spans the comm; use CE path instead");
    return false;
  }
  // Intra-node CE scatter writes via LSA pointers
  if (ncclTeamLsa(comm).nRanks < comm->localRanks) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: LSA team does not cover all local ranks");
    return false;
  }
  // Need symmetric support
  if (!comm->symmetricSupport) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: symmetric support is not enabled");
    return false;
  }
  // Need RMA proxy for inter-node puts
  if (!comm->hostRmaSupport || comm->config.numRmaCtx == 0) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: RMA proxy not available");
    return false;
  }
  // Need registered windows for both send and recv buffers
  if (winRegType != ncclSymSendRegRecvReg) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: window registration type %d not supported", winRegType);
    return false;
  }
  return true;
}

// Per-(peer, chunk) chunking plan in flat form. Peer p's chunks
// span [chunkStart[p], chunkStart[p+1]); total chunks = chunkStart[nPeers].
struct ncclHierChunkPlan {
  int nPeers;
  int* chunkStart;   // [nPeers + 1]  -- prefix sums
  size_t* chunkBytes;   // [chunkStart[nPeers]]  -- per-chunk byte size
  size_t* chunkOff;     // [chunkStart[nPeers]]  -- per-chunk offset within
                         //                          peer's perRankBytes slice
};

// Maximum chunk size used when building hierarchical-collective chunk plans.
static constexpr size_t HIER_COLL_MAX_CHUNK_SIZE = 64 * 1024 * 1024;

// Build a uniform chunking plan
// Every peer gets the same chunk list, last chunk per peer absorbs the remainder.
static ncclResult_t ncclHierCollBuildChunk(size_t perRankBytes, int nPeers, size_t maxChunk,
                                           struct ncclHierChunkPlan* outPlan) {
  ncclResult_t ret = ncclSuccess;
  const size_t align = 8 * 1024;

  outPlan->nPeers = nPeers;
  outPlan->chunkStart = nullptr;
  outPlan->chunkBytes = nullptr;
  outPlan->chunkOff = nullptr;

  int numChunks;
  size_t uniformSize, lastChunk;
  if (perRankBytes == 0 || maxChunk == 0 || perRankBytes <= maxChunk) {
    numChunks = 1;
    uniformSize = perRankBytes;
    lastChunk = perRankBytes;
  } else {
    numChunks = (int)((perRankBytes + maxChunk - 1) / maxChunk);
    uniformSize = (perRankBytes / numChunks / align) * align;
    if (uniformSize < align) uniformSize = align;
    lastChunk = perRankBytes - uniformSize * (numChunks - 1);
  }

  NCCLCHECKGOTO(ncclCalloc(&outPlan->chunkStart, nPeers + 1), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&outPlan->chunkBytes, nPeers * numChunks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&outPlan->chunkOff, nPeers * numChunks), ret, fail);

  for (int p = 0; p <= nPeers; p++) {
    outPlan->chunkStart[p] = p * numChunks;
  }
  for (int p = 0; p < nPeers; p++) {
    size_t off = 0;
    for (int c = 0; c < numChunks; c++) {
      int idx = p * numChunks + c;
      size_t sz = (c == numChunks - 1) ? lastChunk : uniformSize;
      outPlan->chunkBytes[idx] = sz;
      outPlan->chunkOff[idx] = off;
      off += sz;
    }
  }
exit:
  return ret;
fail:
  free(outPlan->chunkStart);
  outPlan->chunkStart = nullptr;
  free(outPlan->chunkBytes);
  outPlan->chunkBytes = nullptr;
  free(outPlan->chunkOff);
  outPlan->chunkOff = nullptr;
  goto exit;
}

static void ncclHierCollFreeChunkPlan(struct ncclHierChunkPlan* plan) {
  if (plan == nullptr) return;
  free(plan->chunkStart);
  free(plan->chunkBytes);
  free(plan->chunkOff);
  plan->chunkStart = nullptr;
  plan->chunkBytes = nullptr;
  plan->chunkOff = nullptr;
  plan->nPeers = 0;
}

// Cross-node rail-sync entry barrier for the hierarchical CE collectives.
static ncclResult_t ncclRailSync(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                 struct ncclKernelPlan* plan, int ctx, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int localRank = comm->localRank;
  int nNodes = comm->nNodes;
  int nRemoteNodes = nNodes - 1;
  bool persistent = plan->persistent;

  // No remote nodes -> nothing to barrier across; fast-path no-op.
  if (nRemoteNodes <= 0) return ncclSuccess;

  int* railPeers = nullptr;
  int* railSigOnes = nullptr;
  // One signal-only put op per rail peer, packed into a single group desc.
  struct ncclRmaPutSignalOp* groupOps = nullptr;
  struct ncclRmaProxyDesc* groupDesc = nullptr;
  struct ncclRmaProxyDesc* waitDesc = nullptr;
  CUstreamBatchMemOpParams* putBatch = nullptr;
  CUstreamBatchMemOpParams* waitBatch = nullptr;

  NCCLCHECKGOTO(ncclCalloc(&railPeers, nRemoteNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&railSigOnes, nRemoteNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupOps, nRemoteNodes), ret, fail);

  // Build one signal-only put op per rail peer
  {
    int idx = 0;
    for (int n = 0; n < nNodes; n++) {
      if (n == comm->node) continue;
      int railPeer = comm->nodeRanks[n].localRankToRank[localRank];
      railPeers[idx] = railPeer;
      railSigOnes[idx] = 1;

      NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, rmaProxyCtx, ctx, persistent,
                                           /*srcWin=*/nullptr, /*srcOff=*/0,
                                           /*peerWin=*/nullptr, /*peerOff=*/0,
                                           /*size=*/0, railPeer, NCCL_SIGNAL, &groupOps[idx]),
                    ret, fail);
      idx++;
    }
  }

  // Build the group put desc
  NCCLCHECKGOTO(ncclCalloc(&groupDesc, 1), ret, fail);
  NCCLCHECKGOTO(ncclRmaProxyPutGroupBuildDesc(comm, rmaProxyCtx, plan, nRemoteNodes, &groupOps, ctx, groupDesc), ret,
                fail);

  // Build one wait descriptor that covers all nRemoteNodes inbound signals.
  NCCLCHECKGOTO(ncclCalloc(&waitDesc, 1), ret, fail);
  NCCLCHECKGOTO(ncclRmaProxyWaitBuildDesc(comm, rmaProxyCtx, plan, nRemoteNodes, &railPeers, &railSigOnes, waitDesc),
                ret, fail);

  // ------------------------------------------------------------------
  // Stage 1: issue the group put (start + done) as one batch.
  // ------------------------------------------------------------------
  {
    int startOps = ncclRmaProxyPutGroupStartNumOps(persistent);
    int doneOps = ncclRmaProxyPutGroupDoneNumOps(persistent);
    int putBatchOps = startOps + doneOps;

    NCCLCHECKGOTO(ncclCalloc(&putBatch, putBatchOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupStartParams(groupDesc, &putBatch[0]), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupDoneParams(groupDesc, &putBatch[startOps]), ret, fail);

    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &groupDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, putBatchOps, putBatch), ret, fail);
  }

  // ------------------------------------------------------------------
  // Stage 2: issue the inbound-signal wait as a separate batch.
  // ------------------------------------------------------------------
  {
    int waitOps = ncclRmaProxyWaitNumStreamOps(waitDesc);
    NCCLCHECKGOTO(ncclCalloc(&waitBatch, waitOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitParams(rmaProxyCtx, waitDesc, waitBatch), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &waitDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitOps, waitBatch), ret, fail);
  }

exit:
  free(putBatch);
  free(waitBatch);
  if (groupDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &groupDesc);
  if (waitDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &waitDesc);
  free(groupOps);
  free(railPeers);
  free(railSigOnes);
  return ret;
fail:
  goto exit;
}

// Helper function to wait for a single peer's signals.
static ncclResult_t ncclProxyWaitOnePeer(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                         struct ncclKernelPlan* plan, int ctx, cudaStream_t stream, int peer,
                                         int nsignals) {
  ncclResult_t ret = ncclSuccess;

  int* waitPeers = nullptr;
  int* waitSigCounts = nullptr;
  struct ncclRmaProxyDesc* waitDesc = nullptr;
  CUstreamBatchMemOpParams* waitBatch = nullptr;

  NCCLCHECKGOTO(ncclCalloc(&waitPeers, 1), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitSigCounts, 1), ret, fail);
  waitPeers[0] = peer;
  waitSigCounts[0] = nsignals;

  NCCLCHECKGOTO(ncclCalloc(&waitDesc, 1), ret, fail);
  NCCLCHECKGOTO(ncclRmaProxyWaitBuildDesc(comm, rmaProxyCtx, plan, 1, &waitPeers, &waitSigCounts, waitDesc), ret, fail);

  {
    int waitOps = ncclRmaProxyWaitNumStreamOps(waitDesc);
    NCCLCHECKGOTO(ncclCalloc(&waitBatch, waitOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitParams(rmaProxyCtx, waitDesc, waitBatch), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &waitDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitOps, waitBatch), ret, fail);
  }

exit:
  free(waitBatch);
  if (waitDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &waitDesc);
  free(waitPeers);
  free(waitSigCounts);
  return ret;
fail:
  goto exit;
}

// Hierarchical AllGather: railed all-to-all inter-node + intra-node CE scatter.
// Each per-rank slice is split into chunks. A single PutGroup descriptor
// bundles all nRemoteNodes * nChunks puts.
//
// DAG on the user stream:
//   RailSync                    // cross-node entry barrier (net + wait)
//   PutGroupSubmit              // one memop fires all network puts in parallel
//   IntraNodeBarrier            // gates LSA peers' recvbuf writes; runs while proxy is in flight
//   SelfBcast                   // CE scatter of own slice to LSA peers
//   for (peer, chunk) in shift order:
//     wait for chunk's signal; CE-scatter it to local peers via LSA
//   PutGroupDone                // one memop blocks until all network puts complete
//   IntraNodeBarrier            // gates user code reading recvbuf

ncclResult_t ncclHierCeAllGather(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  int ctx = 0;
  int myRank = comm->rank;
  int localRank = comm->localRank;
  int nNodes = comm->nNodes;
  int nRemoteNodes = nNodes - 1;
  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  bool persistent = plan->persistent;

  struct ncclCeCollArgs* args = plan->ceCollArgs;
  const void* sendbuff = args->sendBuff;
  void* recvbuff = args->recvBuff;
  struct ncclDevrWindow* sendWin = args->sendWin;
  struct ncclDevrWindow* recvWin = args->recvWin;
  size_t perRankBytes = args->nElts * args->eltSize;

  struct ncclRmaProxyCtx* rmaProxyCtx = (struct ncclRmaProxyCtx*)comm->rmaState.rmaProxyState.rmaProxyCtxs[ctx];

  // Per-(peer, chunk) plan.
  struct ncclHierChunkPlan chunkPlan = {};
  // Inter-node put-signal-group descriptor.
  struct ncclRmaProxyDesc* groupDesc = nullptr;
  struct ncclRmaPutSignalOp* groupOps = nullptr;
  CUstreamBatchMemOpParams* groupStartParam = nullptr;
  CUstreamBatchMemOpParams* groupDoneParam = nullptr;
  // Batch-ops scratch for intra-node broadcast.
  struct ncclCeBatchOpsParams ceBcastOps = {};
  // Batch-ops scratch for per-chunk intra-node CE scatter.
  struct ncclCeBatchOpsParams ceScatterOps = {};

  // ====================================================================
  // Phase 1: Rail sync (cross-node entry barrier)
  // ====================================================================
  NCCLCHECKGOTO(ncclRailSync(comm, rmaProxyCtx, plan, ctx, stream), ret, fail);

  // ====================================================================
  // Phase 2: Start all inter-node puts (one group descriptor, chunked)
  // ====================================================================
  {
    NCCLCHECKGOTO(ncclHierCollBuildChunk(perRankBytes, nRemoteNodes, HIER_COLL_MAX_CHUNK_SIZE, &chunkPlan), ret, fail);
    int totalOps = chunkPlan.chunkStart[chunkPlan.nPeers];

    int startOps = ncclRmaProxyPutGroupStartNumOps(persistent);
    int doneOps = ncclRmaProxyPutGroupDoneNumOps(persistent);
    NCCLCHECKGOTO(ncclCalloc(&groupStartParam, startOps), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&groupDoneParam, doneOps), ret, fail);

    // Window-relative offsets
    size_t srcWinOffset = (const uint8_t*)sendbuff - (const uint8_t*)sendWin->userPtr;
    size_t peerWinOffset = ((const uint8_t*)recvbuff + myRank * perRankBytes) - (const uint8_t*)recvWin->userPtr;

    // Allocate desc + ops array
    NCCLCHECKGOTO(ncclCalloc(&groupDesc, 1), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&groupOps, totalOps), ret, fail);

    for (int s = 1; s < nNodes; s++) {
      int p = s - 1;                                 // peer index in plan
      int n = (comm->node + s) % nNodes;
      int railPeer = comm->nodeRanks[n].localRankToRank[localRank];

      for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
        size_t subBytes = chunkPlan.chunkBytes[c];
        size_t off = chunkPlan.chunkOff[c];

        NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, rmaProxyCtx, ctx, persistent, sendWin, srcWinOffset + off, recvWin,
                                             peerWinOffset + off, subBytes, railPeer, NCCL_SIGNAL, &groupOps[c]),
                      ret, fail);
      }
    }

    // Build the group desc
    NCCLCHECKGOTO(ncclRmaProxyPutGroupBuildDesc(comm, rmaProxyCtx, plan, totalOps, &groupOps, ctx, groupDesc), ret,
                  fail);

    NCCLCHECKGOTO(ncclRmaProxyPutGroupStartParams(groupDesc, groupStartParam), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupDoneParams(groupDesc, groupDoneParam), ret, fail);

    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &groupDesc), ret, fail);

    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, startOps, groupStartParam), ret, fail);
  }

  // ====================================================================
  // Phase 3: Initial intra-node barrier
  // ====================================================================
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  // ====================================================================
  // Phase 4: Self-broadcast (intra-node CE Broadcast of own chunk)
  // ====================================================================
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceBcastOps, lsaSize), ret, fail);
  {
    uint8_t* myRecvSlot = (uint8_t*)recvbuff + myRank * perRankBytes;
    size_t offset = myRecvSlot - (uint8_t*)recvWin->userPtr;

    // Out-of-place: copy own data to own recvbuf slot
    if (myRecvSlot != (const uint8_t*)sendbuff) {
      ceBcastOps.srcs[ceBcastOps.numOps] = (void*)sendbuff;
      ceBcastOps.dsts[ceBcastOps.numOps] = (void*)myRecvSlot;
      ceBcastOps.sizes[ceBcastOps.numOps] = perRankBytes;
      ceBcastOps.numOps++;
    }

    // Broadcast to all other LSA peers
    for (int r = 1; r < lsaSize; r++) {
      int targetLsaRank = (myLsaRank + r) % lsaSize;
      void* peerBuf;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, offset, targetLsaRank, &peerBuf), ret, fail);
      ceBcastOps.srcs[ceBcastOps.numOps] = (void*)sendbuff;
      ceBcastOps.dsts[ceBcastOps.numOps] = peerBuf;
      ceBcastOps.sizes[ceBcastOps.numOps] = perRankBytes;
      ceBcastOps.numOps++;
    }

    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceBcastOps, stream, args), ret, fail);
  }

  // ====================================================================
  // Phase 5: Wait for each (peer, chunk) + intra-node CE scatter (pipelined)
  // ====================================================================
  {
    for (int s = 1; s < nNodes; s++) {
      int p = s - 1;                                 // peer index in plan
      int n = (comm->node - s + nNodes) % nNodes;
      int railPeer = comm->nodeRanks[n].localRankToRank[localRank];
      size_t peerSliceOffset = railPeer * perRankBytes;

      for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
        size_t subBytes = chunkPlan.chunkBytes[c];
        size_t off = chunkPlan.chunkOff[c];

        uint8_t* chunkSlot = (uint8_t*)recvbuff + peerSliceOffset + off;
        size_t winOffset = chunkSlot - (uint8_t*)recvWin->userPtr;

        // ----- Wait for this sub-chunk's signal from railPeer -----
        NCCLCHECKGOTO(ncclProxyWaitOnePeer(comm, rmaProxyCtx, plan, ctx, stream, railPeer, /*nsignals=*/1), ret, fail);

        // ----- CE scatter this sub-chunk to all other LSA peers -----
        NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceScatterOps, lsaSize), ret, fail);
        for (int r = 1; r < lsaSize; r++) {
          int targetLsaRank = (myLsaRank + r) % lsaSize;
          void* peerBuf;
          NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, winOffset, targetLsaRank, &peerBuf), ret, fail);
          ceScatterOps.srcs[ceScatterOps.numOps] = chunkSlot;
          ceScatterOps.dsts[ceScatterOps.numOps] = peerBuf;
          ceScatterOps.sizes[ceScatterOps.numOps] = subBytes;
          ceScatterOps.numOps++;
        }

        NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceScatterOps, stream, args), ret, fail);
        ncclCeFreeBatchOpsParams(&ceScatterOps);
      }
    }
  }

  // ====================================================================
  // Phase 6: Wait for all outgoing data puts to complete
  // ====================================================================
  {
    int doneOps = ncclRmaProxyPutGroupDoneNumOps(persistent);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, doneOps, groupDoneParam), ret, fail);
  }

  // ====================================================================
  // Phase 7: Final intra-node barrier
  // ====================================================================
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&ceBcastOps);
  ncclCeFreeBatchOpsParams(&ceScatterOps);
  free(groupStartParam);
  free(groupDoneParam);
  free(groupOps);
  if (groupDesc != nullptr) {
    (void)ncclRmaProxyDestroyDesc(comm, &groupDesc);
  }
  ncclHierCollFreeChunkPlan(&chunkPlan);
  return ret;
fail:
  goto exit;
}

// Hierarchical AlltoAll: alltoall inter-node + intra-node CE alltoall.
// DAG on the user stream:
//   RailSync                    // rail-only entry barrier
//   IntraNodeBarrier #1         // all ranks in sync
//   PutGroupSubmit              // one memop fires all put operations
//   IntraNodeAlltoAll           // batched CE alltoall
//   AggregateWait               // single multi-peer wait descriptor covering all remote peers
//   PutGroupDone                // one memop blocks until outbound puts done
//   IntraNodeBarrier #2         // all ranks in sync

ncclResult_t ncclHierCeAlltoAll(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  int ctx = 0;
  int myRank = comm->rank;
  int myNode = comm->node;
  int nNodes = comm->nNodes;
  int localRanks = comm->localRanks;
  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  int numRemotePeers = (nNodes - 1) * localRanks;
  bool persistent = plan->persistent;

  struct ncclCeCollArgs* args = plan->ceCollArgs;
  const void* sendbuff = args->sendBuff;
  void* recvbuff = args->recvBuff;
  struct ncclDevrWindow* sendWin = args->sendWin;
  struct ncclDevrWindow* recvWin = args->recvWin;
  size_t perPeerBytes = args->nElts * args->eltSize;
  bool inPlace = (sendbuff == recvbuff);

  struct ncclRmaProxyCtx* rmaProxyCtx = (struct ncclRmaProxyCtx*)comm->rmaState.rmaProxyState.rmaProxyCtxs[ctx];

  // Chunk plan for the inter-node put-signal-group.
  struct ncclHierChunkPlan chunkPlan = {};
  // Inter-node put-signal-group descriptor.
  struct ncclRmaProxyDesc* groupDesc = nullptr;
  struct ncclRmaPutSignalOp* groupOps = nullptr;
  CUstreamBatchMemOpParams* groupStartParam = nullptr;
  CUstreamBatchMemOpParams* groupDoneParam = nullptr;
  // Aggregate inbound wait descriptor (covers all remote peers).
  int* waitPeers = nullptr;
  int* waitSigCounts = nullptr;
  struct ncclRmaProxyDesc* waitDesc = nullptr;
  CUstreamBatchMemOpParams* waitBatch = nullptr;
  // Intra-node alltoall scratch.
  struct ncclCeBatchOpsParams ceLocalA2A = {};

  // ====================================================================
  // Phase 1: Rail sync (rail-only cross-node entry barrier)
  // ====================================================================
  NCCLCHECKGOTO(ncclRailSync(comm, rmaProxyCtx, plan, ctx, stream), ret, fail);

  // ====================================================================
  // Phase 2: Intra-node barrier
  // ====================================================================
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  // ====================================================================
  // Phase 3: Build & submit put-signal-group (start memop).
  // ====================================================================
  {
    NCCLCHECKGOTO(ncclHierCollBuildChunk(perPeerBytes, numRemotePeers, HIER_COLL_MAX_CHUNK_SIZE, &chunkPlan), ret,
                  fail);
    int totalOps = chunkPlan.chunkStart[chunkPlan.nPeers];

    int startOps = ncclRmaProxyPutGroupStartNumOps(persistent);
    int doneOps = ncclRmaProxyPutGroupDoneNumOps(persistent);
    NCCLCHECKGOTO(ncclCalloc(&groupStartParam, startOps), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&groupDoneParam, doneOps), ret, fail);

    NCCLCHECKGOTO(ncclCalloc(&groupDesc, 1), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&groupOps, totalOps), ret, fail);

    int p = 0;  // chunk plan slot index
    for (int s = 1; s < nNodes; s++) {
      int n = (myNode + s) % nNodes;
      for (int lr = 0; lr < localRanks; lr++) {
        int peer = comm->nodeRanks[n].localRankToRank[lr];
        size_t srcWinOffset =
          ((const uint8_t*)sendbuff + (size_t)peer * perPeerBytes) - (const uint8_t*)sendWin->userPtr;
        size_t peerWinOffset =
          ((const uint8_t*)recvbuff + (size_t)myRank * perPeerBytes) - (const uint8_t*)recvWin->userPtr;

        for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
          size_t subBytes = chunkPlan.chunkBytes[c];
          size_t off = chunkPlan.chunkOff[c];

          NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, rmaProxyCtx, ctx, persistent, sendWin, srcWinOffset + off, recvWin,
                                               peerWinOffset + off, subBytes, peer, NCCL_SIGNAL, &groupOps[c]),
                        ret, fail);
        }
        p++;
      }
    }

    NCCLCHECKGOTO(ncclRmaProxyPutGroupBuildDesc(comm, rmaProxyCtx, plan, totalOps, &groupOps, ctx, groupDesc), ret,
                  fail);

    NCCLCHECKGOTO(ncclRmaProxyPutGroupStartParams(groupDesc, groupStartParam), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupDoneParams(groupDesc, groupDoneParam), ret, fail);

    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &groupDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, startOps, groupStartParam), ret, fail);
  }

  // ====================================================================
  // Phase 4: Intra-node alltoall (batched CE memcpy over LSA).
  // ====================================================================
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceLocalA2A, lsaSize), ret, fail);
  {
    size_t myRecvOffset = ((const uint8_t*)recvbuff + (size_t)myRank * perPeerBytes) - (const uint8_t*)recvWin->userPtr;

    for (int k = 0; k < lsaSize; k++) {
      int targetLsa = (myLsaRank + k) % lsaSize;
      int targetWorldRank = comm->nodeRanks[myNode].localRankToRank[targetLsa];

      if (inPlace && targetLsa == myLsaRank) continue;

      void* peerRecvSlot;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, myRecvOffset, targetLsa, &peerRecvSlot), ret, fail);

      ceLocalA2A.srcs[ceLocalA2A.numOps] = (void*)((const uint8_t*)sendbuff + (size_t)targetWorldRank * perPeerBytes);
      ceLocalA2A.dsts[ceLocalA2A.numOps] = peerRecvSlot;
      ceLocalA2A.sizes[ceLocalA2A.numOps] = perPeerBytes;
      ceLocalA2A.numOps++;
    }

    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceLocalA2A, stream, args), ret, fail);
  }

  // ====================================================================
  // Phase 5: Aggregate wait for all remote peers.
  // ====================================================================
  {
    NCCLCHECKGOTO(ncclCalloc(&waitPeers, numRemotePeers), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&waitSigCounts, numRemotePeers), ret, fail);

    int p = 0;
    for (int s = 1; s < nNodes; s++) {
      int n = (myNode - s + nNodes) % nNodes;
      for (int lr = 0; lr < localRanks; lr++) {
        waitPeers[p] = comm->nodeRanks[n].localRankToRank[lr];
        waitSigCounts[p] = chunkPlan.chunkStart[p + 1] - chunkPlan.chunkStart[p];
        p++;
      }
    }

    NCCLCHECKGOTO(ncclCalloc(&waitDesc, 1), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitBuildDesc(comm, rmaProxyCtx, plan, numRemotePeers, &waitPeers, &waitSigCounts,
                                            waitDesc),
                  ret, fail);

    int waitOps = ncclRmaProxyWaitNumStreamOps(waitDesc);
    NCCLCHECKGOTO(ncclCalloc(&waitBatch, waitOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitParams(rmaProxyCtx, waitDesc, waitBatch), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &waitDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitOps, waitBatch), ret, fail);
  }

  // ====================================================================
  // Phase 6: PutGroupDone memop (outbound puts complete on the wire).
  // ====================================================================
  {
    int doneOps = ncclRmaProxyPutGroupDoneNumOps(persistent);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, doneOps, groupDoneParam), ret, fail);
  }

  // ====================================================================
  // Phase 7: Intra-node barrier
  // ====================================================================
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&ceLocalA2A);
  free(groupStartParam);
  free(groupDoneParam);
  free(groupOps);
  if (groupDesc != nullptr) {
    (void)ncclRmaProxyDestroyDesc(comm, &groupDesc);
  }
  free(waitBatch);
  if (waitDesc != nullptr) {
    (void)ncclRmaProxyDestroyDesc(comm, &waitDesc);
  }
  free(waitPeers);
  free(waitSigCounts);
  ncclHierCollFreeChunkPlan(&chunkPlan);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclLaunchCeColl(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclResult_t ret = ncclSuccess;
  cudaStream_t stream = comm->planner.streams->stream;
  struct ncclCeCollArgs* args = plan->ceCollArgs;

  // Start CE collective profiling
  NCCLCHECKGOTO(ncclProfilerStartCeCollEvent(comm, args, stream), ret, fail);

  switch (args->func) {
  case ncclFuncAllGather:
    NCCLCHECKGOTO(ncclCeAllGather(comm, args, stream), ret, fail);
    break;
  case ncclFuncAlltoAll:
    NCCLCHECKGOTO(ncclCeAlltoAll(comm, args, stream), ret, fail);
    break;
  case ncclFuncScatter:
    NCCLCHECKGOTO(ncclCeScatter(comm, args, stream), ret, fail);
    break;
  case ncclFuncGather:
    NCCLCHECKGOTO(ncclCeGather(comm, args, stream), ret, fail);
    break;
  default:
    ret = ncclInvalidUsage;
  }

exit:
  // Stop CE collective profiling - always attempt if started, even on error
  ncclProfilerStopCeCollEvent(comm, args, stream);
  return ret;
fail:
  goto exit;
}

ncclResult_t scheduleCeCollTaskToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collCeTaskQueue);

  plan->isCeColl = true;
  plan->ceCollArgs = ncclMemoryStackAlloc<struct ncclCeCollArgs>(&comm->memScoped);
  plan->ceCollArgs->rootRank = task->root;
  plan->ceCollArgs->datatype = task->datatype;
  plan->ceCollArgs->nElts = task->count;
  plan->ceCollArgs->eltSize = ncclTypeSize(task->datatype);
  plan->ceCollArgs->sendBuff = (uint8_t*)task->sendbuff;
  plan->ceCollArgs->recvBuff = (uint8_t*)task->recvbuff;
  plan->ceCollArgs->func = task->func;
  plan->ceCollArgs->sendWin = task->sendWin;
  plan->ceCollArgs->recvWin = task->recvWin;
  plan->ceCollArgs->collApiEventHandle = task->collApiEventHandle;

  if (comm->rank == 0) {
    if (!ncclDevrIsOneLsaTeam(comm)) {
      INFO(NCCL_TUNING, "%s [Hierarchical CE]: %ld Bytes -> RMA proxy + CE", ncclFuncToString(task->func),
           task->count * ncclTypeSize(task->datatype));
    } else {
      const char* nvlsSync = comm->nvlsSupport ? "; CE synchronization with NVLS" : "";
      INFO(NCCL_TUNING, "%s [Copy Engine]: %ld Bytes -> cudaMemcpy%s", ncclFuncToString(task->func),
           task->count * ncclTypeSize(task->datatype), nvlsSync);
    }
  }

  ncclIntruQueueEnqueue(&planner->planQueue, plan);
  ncclIntruQueueDequeue(&planner->collCeTaskQueue);
  ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, task);

  return ncclSuccess;
}
