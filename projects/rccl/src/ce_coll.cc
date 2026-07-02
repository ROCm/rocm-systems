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
#include "dev_runtime.h" 


ncclResult_t ncclCeLaunchPersistentReduce(
  const void* in, void* out, int nRanks,
  size_t baseChunkElems, size_t tailChunkElems, size_t chunksPerShard,
  uint32_t* signalBuffer, uint32_t* d_quitFlag, uint32_t* d_kernelReady,
  uint32_t* d_reduceDone, uint32_t readyValue, size_t totalSteps,
  ncclDataType_t datatype, ncclRedOp_t op, hipStream_t stream);

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
  const size_t NUM_SLOTS = 2;

  // Declare every variable that is live at a goto-target label up-front, so no
  // NCCLCHECKGOTO jumps over a variable initialization (ill-formed in C++).
  uint8_t*           ceDevBase     = nullptr;
  uint8_t*           signalBuf     = nullptr;
  uint8_t*           ceARTmpBuf    = nullptr;
  ncclWindow_vidmem* ceWinDev      = nullptr;
  ncclWindow_vidmem* ceWinDevHost  = nullptr;
  ncclWindow_vidmem* sigWinDev     = nullptr;
  ncclWindow_vidmem* sigWinDevHost = nullptr;
  ncclWindow_vidmem* arWinDev      = nullptr;
  ncclWindow_vidmem* arWinDevHost  = nullptr;
  size_t ceDevBaseSize  = alignUp(comm->nRanks*sizeof(uint32_t), 16) * 2;
  size_t sigBufferSize  = NUM_SLOTS * comm->nRanks * sizeof(uint32_t);
  size_t maxChunkBytes  = NCCL_CE_AR_MAX_MSG_BYTES / comm->nRanks;
  size_t ceARTmpBufSize = alignUp(NUM_SLOTS * comm->nRanks * maxChunkBytes, 16);
  int i = 0;
  int targetStreams = 0;

  // Symmetric memory runtime must be initialized before any window registration.
  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);

  // Local-only control words (no peer access -> no window registration needed).
  NCCLCHECKGOTO(ncclMemAlloc((void**)&comm->ceColl.d_kernelReady, sizeof(uint32_t)), ret, fail);
  NCCLCHECKGOTO(ncclMemAlloc((void**)&comm->ceColl.d_quitFlag,    sizeof(uint32_t)), ret, fail);
  NCCLCHECKGOTO(ncclMemAlloc((void**)&comm->ceColl.d_reduceDone,  sizeof(uint32_t)), ret, fail);
  CUDACHECKGOTO(hipMemset(comm->ceColl.d_kernelReady, 0, sizeof(uint32_t)), ret, fail);
  CUDACHECKGOTO(hipMemset(comm->ceColl.d_quitFlag,    0, sizeof(uint32_t)), ret, fail);
  CUDACHECKGOTO(hipMemset(comm->ceColl.d_reduceDone,  0, sizeof(uint32_t)), ret, fail);
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&comm->ceColl.computeStream, cudaStreamNonBlocking), ret, fail);
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&comm->ceColl.gatherStream, cudaStreamNonBlocking), ret, fail);
  CUDACHECKGOTO(cudaEventCreateWithFlags(&comm->ceColl.gatherDoneEvent, cudaEventDisableTiming), ret, fail);
  // Signal buffer: [NUM_SLOTS][nRanks], indexed slot*nRanks + r. Symmetric window
  // so peers can ring each other's doorbells via LSA (same pattern as ceARTmpWin).
  NCCLCHECKGOTO(ncclMemAlloc((void**)&signalBuf, sigBufferSize), ret, fail);
  NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, signalBuf, sigBufferSize,
                                              NCCL_WIN_COLL_SYMMETRIC, &sigWinDev), ret, fail);
  NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, sigWinDev, &sigWinDevHost), ret, fail);
  comm->ceColl.signalWin    = (struct ncclDevrWindow*)sigWinDevHost->winHost;
  comm->ceColl.signalBuffer = (uint32_t*)comm->ceColl.signalWin->userPtr;
  CUDACHECKGOTO(hipMemset(comm->ceColl.signalBuffer, 0, sigBufferSize), ret, fail);

  // ceSync window (ready/complete flag arrays).
  NCCLCHECKGOTO(ncclMemAlloc((void**)&ceDevBase, ceDevBaseSize), ret, fail);
  NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, ceDevBase, ceDevBaseSize, NCCL_WIN_COLL_SYMMETRIC, &ceWinDev), ret,
                fail);
  NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, ceWinDev, &ceWinDevHost), ret, fail);
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

  // CE AllReduce staging buffer (double-buffered scatter staging, no scratch):
  //   [slot 0: nRanks chunks][slot 1: nRanks chunks].
  NCCLCHECKGOTO(ncclMemAlloc((void**)&ceARTmpBuf, ceARTmpBufSize), ret, fail_ar);
  NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, ceARTmpBuf, ceARTmpBufSize,
                                              NCCL_WIN_COLL_SYMMETRIC, &arWinDev), ret, fail_ar);
  NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, arWinDev, &arWinDevHost), ret, fail_ar);
  comm->ceColl.ceARTmpWin = (struct ncclDevrWindow*)arWinDevHost->winHost;
  comm->ceColl.ceARTmpBuf = (uint8_t*)comm->ceColl.ceARTmpWin->userPtr;
  INFO(NCCL_INIT, "Init CE AllReduce, rank %d ceARTmpBuf %p size %zu",
       comm->rank, comm->ceColl.ceARTmpBuf, ceARTmpBufSize);

exit:
  return ret;
fail_ar:
  ceDestroyCopyStreams(comm, comm->ceColl.nCopyStreams);
  goto fail;
fail_ce_event:
  CUDACHECKIGNORE(cudaStreamDestroy(comm->ceColl.copyStreams[i]));
fail_ce_stream:
  INFO(NCCL_INIT, "CE init failed on rank %d after creating %d/%d copy streams", comm->rank, i, targetStreams);
  ceDestroyCopyStreams(comm, i);
  goto fail;
fail:
  if (arWinDev != nullptr) ncclCommWindowDeregister(comm, arWinDev);
  if (ceARTmpBuf != nullptr) ncclMemFree(ceARTmpBuf);
  if (ceWinDev != nullptr) ncclCommWindowDeregister(comm, ceWinDev);
  if (ceDevBase != nullptr) ncclMemFree(ceDevBase);
  if (sigWinDev != nullptr) ncclCommWindowDeregister(comm, sigWinDev);
  if (signalBuf != nullptr) ncclMemFree(signalBuf);
  if (comm->ceColl.d_kernelReady != nullptr) ncclMemFree(comm->ceColl.d_kernelReady);
  if (comm->ceColl.d_quitFlag != nullptr) ncclMemFree(comm->ceColl.d_quitFlag);
  if (comm->ceColl.d_reduceDone != nullptr) ncclMemFree(comm->ceColl.d_reduceDone);
  if (comm->ceColl.computeStream != nullptr) cudaStreamDestroy(comm->ceColl.computeStream);
  if (comm->ceColl.gatherStream != nullptr) cudaStreamDestroy(comm->ceColl.gatherStream);
  if (comm->ceColl.gatherDoneEvent != nullptr) cudaEventDestroy(comm->ceColl.gatherDoneEvent);
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

  // Clean up CE AllReduce staging buffer
  if (comm->ceColl.ceARTmpBuf != NULL) {
    if (comm->ceColl.ceARTmpWin && comm->ceColl.ceARTmpWin->vidmem ) {
      NCCLCHECKGOTO(ncclCommWindowDeregister(comm, comm->ceColl.ceARTmpWin->vidmem), ret, fail);
      NCCLCHECKGOTO(ncclMemFree(comm->ceColl.ceARTmpBuf), ret, fail);
    }
    comm->ceColl.ceARTmpBuf = NULL;
    comm->ceColl.ceARTmpWin = NULL;
  }
  if(comm->ceColl.signalBuffer != NULL) {
    if (comm->ceColl.signalWin && comm->ceColl.signalWin->vidmem ) {
      NCCLCHECKGOTO(ncclCommWindowDeregister(comm, comm->ceColl.signalWin->vidmem), ret, fail);
      NCCLCHECKGOTO(ncclMemFree(comm->ceColl.signalBuffer), ret, fail);
    }
    comm->ceColl.signalBuffer = NULL;
    comm->ceColl.signalWin = NULL;
  }

  // Clean up copy streams and events
  ceDestroyCopyStreams(comm, comm->ceColl.nCopyStreams);
  if (comm->ceColl.d_kernelReady != nullptr) ncclMemFree(comm->ceColl.d_kernelReady);
  if (comm->ceColl.d_quitFlag != nullptr) ncclMemFree(comm->ceColl.d_quitFlag);
  if (comm->ceColl.d_reduceDone != nullptr) ncclMemFree(comm->ceColl.d_reduceDone);
  if (comm->ceColl.computeStream != nullptr) cudaStreamDestroy(comm->ceColl.computeStream);
  if (comm->ceColl.gatherStream != nullptr) cudaStreamDestroy(comm->ceColl.gatherStream);
  if (comm->ceColl.gatherDoneEvent != nullptr) cudaEventDestroy(comm->ceColl.gatherDoneEvent);


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
    case ncclFuncAllReduce:
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

ncclResult_t ncclMemOpSync(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  void* ceSyncHandle = NULL;

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
    for (int i = 0; i < comm->nRanks; i++) {
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

ncclResult_t ncclCeLaunchBatchOps(struct ncclComm* comm, struct ncclCeCollArgs* args,
                                  struct ncclCeBatchOpsParams* params, cudaStream_t stream) {
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
        NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);
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
            NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);
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
            NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);
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
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff + comm->rank * chunkBytes;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};

  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

  // Copy own data to receive buffer if operation is out-of-place
  if (myRecvBuff != mySendBuff) {
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)myRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // Copy data to other ranks
  for (int r = 1; r < comm->nRanks; r++) {
    int targetRank = (comm->rank + r) % comm->nRanks;
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
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, args, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeAlltoAll(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Calculate the size of data each rank sends to every other rank
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

  // Copy data to other ranks: send data chunk for each destination rank
  for (int r = 0; r < comm->nRanks; r++) {
    int dstRank = (comm->rank + r) % comm->nRanks;
    uint8_t* srcPtr = mySendBuff + dstRank * chunkBytes;
    uint8_t* dstPtr = myRecvBuff + comm->rank * chunkBytes;

    if (dstRank == comm->rank) {
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
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, args, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeScatter(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Calculate the size of data each rank sends to every other rank
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  int rootRank = args->rootRank;
  void* peerDstPtr;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

  if (comm->rank == rootRank) {
    // Check if this is an in-place scatter operation
    bool isInPlace = (myRecvBuff == mySendBuff + comm->rank * chunkBytes);

    // Copy root's own data first if not in-place
    if (!isInPlace) {
      uint8_t* srcPtr = mySendBuff + comm->rank * chunkBytes;
      uint8_t* dstPtr = myRecvBuff;
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }

    // Root rank distributes data to other ranks
    for (int r = 1; r < comm->nRanks; r++) {
      int dstRank = (comm->rank + r) % comm->nRanks;
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
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, args, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Calculate the size of data each rank sends to every other rank
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  int rootRank = args->rootRank;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, 1), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

  if (comm->rank == rootRank) {
    // Root rank copies its own data to the correct position in receive buffer
    uint8_t* dstPtr = myRecvBuff + comm->rank * chunkBytes;
    if (mySendBuff != dstPtr) {
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  } else {
    // Non-root ranks send their data to root's receive buffer
    uint8_t* rootRecvPtr = (uint8_t*)args->recvBuff + comm->rank * chunkBytes;
    offset = rootRecvPtr - (uint8_t*)args->recvWin->userPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, rootRank, &peerRecvBuff), ret, fail);
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, args, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}
// =============================================================================
// CE AllReduce pipeline: single-op stream memory-op helpers.
// Thin wrappers over hipStreamBatchMemOp, matching the CUDA-compatible batch
// memop struct already used by ncclMemOpSync / ncclPrepUCSync in this file.
//   - WaitGte / WaitEq : stall the stream until *addr >= / == value
//   - WriteValue       : write value to a local address on the stream
//   - QueueDoorbellWrite : write value to a peer (LSA) address on the stream
// =============================================================================
static ncclResult_t ncclMemOpWaitValue(struct ncclComm* /*comm*/, uint32_t* addr,
  uint32_t value, uint32_t flags,
  cudaStream_t stream) {
  hipStreamBatchMemOpParams op = {};
  op.waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
  op.waitValue.address   = (CUdeviceptr)addr;
  op.waitValue.value     = value;
  op.waitValue.flags     = flags;
  CUCHECK(hipStreamBatchMemOp(stream, 1, &op, 0));
  return ncclSuccess;
}

ncclResult_t ncclMemOpWaitGte(struct ncclComm* comm, uint32_t* addr,
uint32_t value, cudaStream_t stream) {
  return ncclMemOpWaitValue(comm, addr, value, CU_STREAM_WAIT_VALUE_GEQ, stream);
}

ncclResult_t ncclMemOpWaitEq(struct ncclComm* comm, uint32_t* addr,
uint32_t value, cudaStream_t stream) {
  return ncclMemOpWaitValue(comm, addr, value, CU_STREAM_WAIT_VALUE_EQ, stream);
}

ncclResult_t ncclMemOpWriteValue(struct ncclComm* /*comm*/, uint32_t* addr,
uint32_t value, cudaStream_t stream) {
  hipStreamBatchMemOpParams op = {};
  op.writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
  op.writeValue.address   = (CUdeviceptr)addr;
  op.writeValue.value     = value;
  op.writeValue.flags     = CU_STREAM_WRITE_VALUE_DEFAULT;
  CUCHECK(hipStreamBatchMemOp(stream, 1, &op, 0));
  return ncclSuccess;
}

// A cross-rank doorbell is just a WriteValue32 to a peer (LSA) pointer.
ncclResult_t ncclCeQueueDoorbellWrite(struct ncclComm* comm, void* peerAddr,
 uint32_t value, cudaStream_t stream) {
 return ncclMemOpWriteValue(comm, (uint32_t*)peerAddr, value, stream);
}

inline size_t chooseChunkBytes(size_t shardBytes) {
  const size_t MIN_CHUNK_BYTES = 4 * 1024 * 1024ULL;
  const size_t MAX_CHUNK_BYTES = 512 * 1024 * 1024ULL;
  const size_t CHUNK_BYTES_STEP = 128 * 1024ULL;
  if (shardBytes <= MIN_CHUNK_BYTES) return MIN_CHUNK_BYTES;
  size_t targetChunkBytes = shardBytes / 4;
  targetChunkBytes = (targetChunkBytes + 4095U) & ~4095U;
  if (targetChunkBytes > MAX_CHUNK_BYTES) return MAX_CHUNK_BYTES; 
  if (targetChunkBytes < MIN_CHUNK_BYTES) return MIN_CHUNK_BYTES;
  return targetChunkBytes;
}


ncclResult_t ncclCeAllReduce(struct ncclComm* comm, const void* sendbuff,
  void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op,
  cudaStream_t stream,
  struct ncclDevrWindow* recvWin) {
  ncclResult_t ret = ncclSuccess;
  

  const size_t eltSize         = ncclTypeSize(datatype);
  const size_t totalBytes      = count * eltSize;
  const size_t shardBytes      = totalBytes / comm->nRanks;
  const size_t shardElems      = count / comm->nRanks;
  const size_t NUM_SLOTS       = 2;

  struct ncclCeColl* ceColl    = &comm->ceColl;
  uint8_t* tmpBuf              = ceColl->ceARTmpBuf;
  uint8_t* outShard           = (uint8_t*)recvbuff + (size_t)comm->rank * shardBytes; // kernel writes reduced shard here
  uint32_t* signalBuffer       = ceColl->signalBuffer;
  uint32_t* d_kernelReady      = ceColl->d_kernelReady;
  uint32_t* d_reduceDone       = ceColl->d_reduceDone;
  cudaStream_t reduceStream    = ceColl->computeStream;
  cudaStream_t gatherStream = ceColl->gatherStream;

  // 1. Calculate optimized alignment chunk sizes 
  size_t chunkBytes     = chooseChunkBytes(shardBytes);
  size_t baseChunkElems = chunkBytes / eltSize;
  size_t chunksPerShard = shardElems / baseChunkElems;
  size_t tailChunkElems = shardElems % baseChunkElems;
  if (tailChunkElems != 0) chunksPerShard++;

  size_t totalSteps = chunksPerShard;
  const size_t slotStrideBytes = chunkBytes * (size_t)comm->nRanks;

  // Unique call verification tracking sequence
  static uint32_t callCounter = 0;
  callCounter++;
  uint32_t kReadyValue = callCounter;

  struct ncclCeBatchOpsParams batchOpsParams = {};
  struct ncclCeCollArgs collArgs = {
  .func = ncclFuncAllReduce, .datatype = datatype, .nElts = count,
  .eltSize = eltSize, .redOp = op
  };

  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), ret, fail);
  // Phase 3 (all-gather) writes the reduced shard into peers' recvbuff via recvWin.
  // Without a registered symmetric recv window there is no peer address to target.
  if (recvWin == nullptr) {
    WARN("CE AllReduce requires a registered recv window (recvWin==NULL); rank %d", comm->rank);
    ret = ncclInvalidArgument;
    goto exit;
  }
  NCCLCHECKGOTO(ncclMemOpSync(comm, &collArgs, stream), ret, fail);
  // =========================================================================
  // STEP 1: LAUNCH PERSISTENT REDUCTION KERNEL (ONCE, BEFORE THE LOOP)
  // =========================================================================
  {
    // Reset the per-chunk completion counter before (re)launching. Ordered on
    // reduceStream ahead of the kernel, so the kernel's increments start from 0
    // and the CE stream's WaitGte(ch+1) can't match a stale value from a prior call.
    CUDACHECKGOTO(hipMemsetAsync(d_reduceDone, 0, sizeof(uint32_t), reduceStream), ret, fail);
    NCCLCHECKGOTO(ncclCeLaunchPersistentReduce(
      tmpBuf, outShard, comm->nRanks,
      baseChunkElems, tailChunkElems, chunksPerShard,
      signalBuffer, ceColl->d_quitFlag, d_kernelReady,
      d_reduceDone, kReadyValue, totalSteps,
      datatype, op, reduceStream), ret, fail);
  }

  // =========================================================================
  // STEP 2: STALL COPY STREAM UNTIL PERSISTENT KERNEL CO-OPERATIVE LAUNCH READY
  // =========================================================================
  NCCLCHECKGOTO(ncclMemOpWaitGte(comm, d_kernelReady, kReadyValue, stream), ret, fail);
  NCCLCHECKGOTO(ncclMemOpWaitGte(comm, d_kernelReady, kReadyValue, gatherStream), ret, fail);
  // =========================================================================
  // STEP 3: CHUNK EXECUTION LOOP (SCATTER -> REDUCE -> GATHER)
  // =========================================================================
  for (size_t ch = 0; ch < chunksPerShard; ch++) {
  const int slot = (int)(ch % NUM_SLOTS);
  const bool isTail = (ch == chunksPerShard - 1) && (tailChunkElems > 0);
  const size_t currentChunkBytes = isTail ? tailChunkElems * eltSize : chunkBytes;

  // A. Verify that the persistent kernel has fully consumed this slot from the prior loop
  if (ch >= NUM_SLOTS) {
    size_t mySlotOffset = ((size_t)slot * comm->nRanks + comm->rank) * sizeof(uint32_t);
    for (int r = 0; r < comm->nRanks; r++) {
      if (r == comm->rank) {
        NCCLCHECKGOTO(ncclMemOpWaitEq(comm,
            &signalBuffer[(size_t)slot * comm->nRanks + comm->rank], 0, stream), ret, fail);
      } else {
        void* peerSig;
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceColl->signalWin, mySlotOffset, r, &peerSig), ret, fail);
        NCCLCHECKGOTO(ncclMemOpWaitEq(comm, (uint32_t*)peerSig, 0, stream), ret, fail);
      }
    }
  }

  // -------------------------------------------------------------------------
  // PHASE 1: SCATTER (CE Stream)
  // -------------------------------------------------------------------------
  batchOpsParams.numOps = 0;
  const size_t dstSlotOffsetBytes = (size_t)slot * slotStrideBytes + (size_t)comm->rank * chunkBytes;
  const size_t srcChunkOffsetBytes = ch * chunkBytes;

  // Batched doorbell: all peers' + local slot signals in ONE stream memop.
  
    //hipStreamBatchMemOpParams sigOps[64]; // nRanks is single-node small; assert nRanks<=64
    //size_t nOps = 0;
    //const size_t peerSlotOffset = ((size_t)slot * comm->nRanks + comm->rank) * sizeof(uint32_t);
  for (int r = 0; r < comm->nRanks; r++) {
    void* dstPtr;
    const uint8_t* srcShard = (const uint8_t*)sendbuff + (r * shardBytes) + srcChunkOffsetBytes;  
    if (r == comm->rank) {
      dstPtr = tmpBuf + dstSlotOffsetBytes;
    } else {
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceColl->ceARTmpWin, dstSlotOffsetBytes, r, &dstPtr), ret, fail);
    }
    batchOpsParams.srcs[batchOpsParams.numOps]  = (void*)srcShard;
    batchOpsParams.dsts[batchOpsParams.numOps]  = dstPtr;
    batchOpsParams.sizes[batchOpsParams.numOps] = currentChunkBytes;
    batchOpsParams.numOps++;
  }
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &collArgs, &batchOpsParams, stream), ret, fail);
  INFO(NCCL_COLL, "CE AllReduce: Scatter launched");
  // B. Fire cross-rank fabric doorbells to signal peer compute engines
  for (int r = 0; r < comm->nRanks; r++) {
    if (r == comm->rank) continue;
    void* peerSignalAddr;
    size_t peerSlotOffset = ((size_t)slot * comm->nRanks + comm->rank) * sizeof(uint32_t);
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceColl->signalWin,
                    peerSlotOffset, r, &peerSignalAddr), ret, fail);
    NCCLCHECKGOTO(ncclCeQueueDoorbellWrite(comm, peerSignalAddr, 1, stream), ret, fail);
  }
  size_t localSlotIndex = (size_t)slot * comm->nRanks + comm->rank;
  NCCLCHECKGOTO(ncclMemOpWriteValue(comm, &signalBuffer[localSlotIndex], 1, stream), ret, fail);
  INFO(NCCL_COLL, "CE AllReduce: Scatter Signal written");
  // -------------------------------------------------------------------------
  // PHASE 2: ASYNCHRONOUS COMPUTE INTERLOCK INTERCONNECT
  // -------------------------------------------------------------------------
  NCCLCHECKGOTO(ncclMemOpWaitGte(comm, d_reduceDone, (uint32_t)(ch + 1), gatherStream), ret, fail);
  INFO(NCCL_COLL, "CE AllReduce: Reduce completed");
  // -------------------------------------------------------------------------
  // PHASE 3: ASYNCHRONOUS ALL-GATHER (CE Stream)
  // -------------------------------------------------------------------------
  // The kernel already wrote the reduced chunk into our recvbuff shard, so there
  // is no self-copy: we just broadcast that chunk to every peer's recvbuff.
  INFO(NCCL_COLL, "CE AllReduce: All-gather launched");
    batchOpsParams.numOps = 0;
    size_t globalRecvOffsetBytes = srcChunkOffsetBytes + ((size_t)comm->rank * shardBytes);
    uint8_t* myReducedChunk = (uint8_t*)recvbuff + globalRecvOffsetBytes;
    for (int r = 1; r < comm->nRanks; r++) {
      int targetRank = (comm->rank + r) % comm->nRanks;
      void* peerRecvBuff;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, globalRecvOffsetBytes,
                      targetRank, &peerRecvBuff), ret, fail);
      batchOpsParams.srcs[batchOpsParams.numOps]  = (void*)myReducedChunk;
      batchOpsParams.dsts[batchOpsParams.numOps]  = peerRecvBuff;
      batchOpsParams.sizes[batchOpsParams.numOps] = currentChunkBytes;
      batchOpsParams.numOps++;
    }
    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &collArgs, &batchOpsParams, gatherStream), ret, fail);
    INFO(NCCL_COLL, "CE AllReduce: All-gather done");
  }
    // Join: the caller synchronizes on `stream`, so make `stream` wait for the
  // trailing all-gather to complete before the collective is considered done.
  CUDACHECKGOTO(cudaEventRecord(ceColl->gatherDoneEvent, gatherStream), ret, fail);
  CUDACHECKGOTO(cudaStreamWaitEvent(stream, ceColl->gatherDoneEvent, 0), ret, fail);
  NCCLCHECKGOTO(ncclMemOpSync(comm, &collArgs, stream), ret, fail);
  exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
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
    case ncclFuncAllReduce:
      // CE init runs (in ncclCommGroupRegisterSymmetric) before doLaunches, so
      // ceARTmpBuf is guaranteed non-NULL by the time we get here.
      if (comm->ceColl.ceARTmpBuf == NULL) {
        WARN("CE AllReduce invoked before CE init; this should not happen");
        ret = ncclInvalidUsage;
        break;
      }
      // Pass args->recvWin so ncclCeAllReduce can take the fast path
      // (AG written directly into user recvbuff, no final D2D copy).
      NCCLCHECKGOTO(ncclCeAllReduce(comm, args->sendBuff, args->recvBuff,
                                    args->nElts, args->datatype, args->redOp,
                                    stream, args->recvWin), ret, fail);
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
