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
#include "alltoallv_meta.h"
#include "group.h"
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

ncclResult_t ncclCeLaunchPersistentReduce(const void* in, void* out, int nRanks, size_t baseChunkElems,
                                          size_t tailChunkElems, size_t chunksPerShard, size_t slotChunkElems,
                                          uint32_t* signalBuffer, size_t totalSteps, uint32_t* d_barrierSync,
                                          ncclDataType_t datatype, ncclRedOp_t op, hipStream_t stream, int coopLaunch);

RCCL_PARAM(CeMultiStreams, "CE_MULTI_STREAMS", 0);
RCCL_PARAM(CeBatchAsyncEnable, "CE_BATCH_ASYNC_ENABLE", -2);
RCCL_PARAM(CeCoopLaunch, "CE_COOP_LAUNCH", 0);
RCCL_PARAM_DECLARE(CeAllReduce);

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
  const size_t NUM_SLOTS = NCCL_CE_NUM_SLOTS;

  // Declare every variable that is live at a goto-target label up-front, so no
  // NCCLCHECKGOTO jumps over a variable initialization (ill-formed in C++).
  uint8_t* ceDevBase = nullptr;
  uint8_t* signalBuf = nullptr;
  uint8_t* ceARTmpBuf = nullptr;
  ncclWindow_vidmem* ceWinDev = nullptr;
  ncclWindow_vidmem* ceWinDevHost = nullptr;
  ncclWindow_vidmem* sigWinDev = nullptr;
  ncclWindow_vidmem* sigWinDevHost = nullptr;
  ncclWindow_vidmem* arWinDev = nullptr;
  ncclWindow_vidmem* arWinDevHost = nullptr;
  size_t ceDevBaseSize = alignUp(comm->nRanks * sizeof(uint32_t), 16) * 2;
  size_t sigBufferSize = NUM_SLOTS * comm->nRanks * sizeof(uint32_t);
  size_t maxChunkBytes = ncclCeAllReduceMaxChunkBytes(comm->nRanks);
  size_t ceARTmpBufSize = alignUp(NUM_SLOTS * comm->nRanks * maxChunkBytes, 16);
  int i = 0;
  int targetStreams = 0;
  uint32_t graphSyncValue = GRAPH_SYNC_VALUE;
  // Symmetric memory runtime must be initialized before any window registration.
  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);

  // Local-only control words (no peer access -> no window registration needed).
  CUDACHECKGOTO(hipExtMallocWithFlags((void**)&comm->ceColl.d_barrierSync, 2 * sizeof(uint32_t),
                                      hipDeviceMallocUncached),
                ret, fail);
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&comm->ceColl.scatterStream, cudaStreamNonBlocking), ret, fail);
  CUDACHECKGOTO(cudaEventCreateWithFlags(&comm->ceColl.synceEvent, cudaEventDisableTiming), ret, fail);
  // Signal buffer: [NUM_SLOTS][nRanks], indexed slot*nRanks + r. Symmetric window
  // so peers can ring each other's doorbells via LSA (same pattern as ceARTmpWin).
  NCCLCHECKGOTO(ncclMemAlloc((void**)&signalBuf, sigBufferSize), ret, fail);
  NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, signalBuf, sigBufferSize, NCCL_WIN_COLL_SYMMETRIC, &sigWinDev), ret,
                fail);
  NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, sigWinDev, &sigWinDevHost), ret, fail);
  comm->ceColl.signalWin = (struct ncclDevrWindow*)sigWinDevHost->winHost;
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
  // Allocate the UC barrier flag-value buffer: slot [0] tracks the running seq,
  // slot [1] holds the constant GRAPH_SYNC_VALUE for graph capture.
  NCCLCHECKGOTO(ncclCudaCalloc(&comm->ceColl.ceSeqNumDev, 2, comm->memManager), ret, fail);
  NCCLCHECKGOTO(ncclCudaMemcpy(comm->ceColl.ceSeqNumDev + 1, &graphSyncValue, 1), ret, fail);
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
  if (rcclParamCeAllReduce()) {
    NCCLCHECKGOTO(ncclMemAlloc((void**)&ceARTmpBuf, ceARTmpBufSize), ret, fail_ar);
    NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, ceARTmpBuf, ceARTmpBufSize, NCCL_WIN_COLL_SYMMETRIC, &arWinDev),
                  ret, fail_ar);
    NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, arWinDev, &arWinDevHost), ret, fail_ar);
    comm->ceColl.ceARTmpWin = (struct ncclDevrWindow*)arWinDevHost->winHost;
    comm->ceColl.ceARTmpBuf = (uint8_t*)comm->ceColl.ceARTmpWin->userPtr;
    INFO(NCCL_INIT, "Init CE AllReduce, rank %d ceARTmpBuf %p size %zu", comm->rank, comm->ceColl.ceARTmpBuf,
         ceARTmpBufSize);
  }

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
  // ncclCeFinalize() runs unconditionally from commFree(), including after a failed
  // init, and keys its cleanup off these members being non-NULL. Clear everything we
  // release here so finalize does not deregister/free it a second time or dereference
  // a window that is already gone.
  if (arWinDev != nullptr) ncclCommWindowDeregister(comm, arWinDev);
  if (ceARTmpBuf != nullptr) ncclMemFree(ceARTmpBuf);
  comm->ceColl.ceARTmpBuf = nullptr;
  comm->ceColl.ceARTmpWin = nullptr;
  if (ceWinDev != nullptr) ncclCommWindowDeregister(comm, ceWinDev);
  if (ceDevBase != nullptr) ncclMemFree(ceDevBase);
  comm->ceColl.baseUCSymReadyPtr = nullptr;
  comm->ceColl.baseUCSymComplPtr = nullptr;
  comm->ceColl.ceSyncWin = nullptr;
  if (sigWinDev != nullptr) ncclCommWindowDeregister(comm, sigWinDev);
  if (signalBuf != nullptr) ncclMemFree(signalBuf);
  comm->ceColl.signalBuffer = nullptr;
  comm->ceColl.signalWin = nullptr;
  if (comm->ceColl.d_barrierSync != nullptr) {
    CUDACHECKIGNORE(hipFree(comm->ceColl.d_barrierSync));
    comm->ceColl.d_barrierSync = nullptr;
  }
  if (comm->ceColl.scatterStream != nullptr) {
    CUDACHECKIGNORE(cudaStreamDestroy(comm->ceColl.scatterStream));
    comm->ceColl.scatterStream = nullptr;
  }
  if (comm->ceColl.synceEvent != nullptr) {
    CUDACHECKIGNORE(cudaEventDestroy(comm->ceColl.synceEvent));
    comm->ceColl.synceEvent = nullptr;
  }
  if (comm->ceColl.ceSeqNumDev != nullptr) {
    ncclCudaFree(comm->ceColl.ceSeqNumDev, comm->memManager);
    comm->ceColl.ceSeqNumDev = nullptr;
  }
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
    if (comm->ceColl.ceARTmpWin && comm->ceColl.ceARTmpWin->vidmem) {
      NCCLCHECKGOTO(ncclCommWindowDeregister(comm, comm->ceColl.ceARTmpWin->vidmem), ret, fail);
      NCCLCHECKGOTO(ncclMemFree(comm->ceColl.ceARTmpBuf), ret, fail);
    }
    comm->ceColl.ceARTmpBuf = NULL;
    comm->ceColl.ceARTmpWin = NULL;
  }
  if (comm->ceColl.signalBuffer != NULL) {
    if (comm->ceColl.signalWin && comm->ceColl.signalWin->vidmem) {
      NCCLCHECKGOTO(ncclCommWindowDeregister(comm, comm->ceColl.signalWin->vidmem), ret, fail);
      NCCLCHECKGOTO(ncclMemFree(comm->ceColl.signalBuffer), ret, fail);
    }
    comm->ceColl.signalBuffer = NULL;
    comm->ceColl.signalWin = NULL;
  }

  // Free the UC barrier flag-value device buffer
  if (comm->ceColl.ceSeqNumDev != NULL) {
    NCCLCHECKGOTO(ncclCudaFree(comm->ceColl.ceSeqNumDev, comm->memManager), ret, fail);
    comm->ceColl.ceSeqNumDev = NULL;
  }

  // Clean up copy streams and events
  ceDestroyCopyStreams(comm, comm->ceColl.nCopyStreams);
  if (comm->ceColl.d_barrierSync != nullptr) CUDACHECKIGNORE(hipFree(comm->ceColl.d_barrierSync));
  if (comm->ceColl.scatterStream != nullptr) CUDACHECKIGNORE(cudaStreamDestroy(comm->ceColl.scatterStream));
  if (comm->ceColl.synceEvent != nullptr) CUDACHECKIGNORE(cudaEventDestroy(comm->ceColl.synceEvent));

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

bool ncclCeAlltoAllvEligible(struct ncclComm* comm, ncclDataType_t datatype, ncclSymRegType_t winRegType,
                             bool hasSysmemSegment, bool capturing) {
  if (ncclGroupDepth != 0) return false;
  if (!(comm->config.CTAPolicy & NCCL_CTA_POLICY_ZERO)) return false;
  if (hasSysmemSegment || capturing) return false;
  return ncclCeAvailable(comm, ncclFuncAlltoAllv, ncclDevSum, datatype, winRegType);
}

bool ncclCeScratchAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
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
    case ncclFuncAlltoAllv:
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

  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  uint32_t* readyPtrs = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);
  uint32_t currentSeq = ++comm->ceColl.ceSeqNum;

  // Wait value is either the constant graph sync value or the sequence number
  uint32_t waitValue = capturing ? GRAPH_SYNC_VALUE : currentSeq;

  // Use multi-cast address as destination pointer
  void* mcDstPtr;
  void* dstPtr = isComplete ? (void*)&completePtrs[myLsaRank] : (void*)&readyPtrs[myLsaRank];
  size_t offset = (uint8_t*)dstPtr - (uint8_t*)comm->ceColl.ceSyncWin->userPtr;
  NCCLCHECKGOTO(ncclDevrGetLsaTeamPtrMC(comm, comm->ceColl.ceSyncWin, offset, ncclTeamLsa(comm), &mcDstPtr), ret, fail);

  // Store the updated sequence number in the device buffer.
  if (!capturing) {
    CUCHECKGOTO(cuStreamWriteValue32(stream, (CUdeviceptr)comm->ceColl.ceSeqNumDev, currentSeq,
                                    CU_STREAM_WRITE_VALUE_DEFAULT),
              ret, fail);
  }

  // Write our own ready/complete flag to the multi-cast address
  CUDACHECKGOTO(cudaMemcpyAsync(mcDstPtr, comm->ceColl.ceSeqNumDev + capturing, sizeof(uint32_t),
                              cudaMemcpyDeviceToDevice, stream),
              ret, fail);

  // Add local wait operations for every other rank
  for (int r = 0; r < lsaSize; ++r) {
    if (r == myLsaRank) continue;
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
// Capture: issue peer writes as a separate stream batch so the captured graph
// orders write → wait → reset (reset is a second batch in ncclMemOpSync). Fusing
// write+wait+reset in one HIP batch can hang on replay.
static ncclResult_t ncclPrepUCSyncCapture(struct ncclComm* comm, bool isComplete, uint32_t* readyPtrs,
                                         uint32_t* completePtrs, uint32_t waitValue, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  hipStreamBatchMemOpParams* writeParams = nullptr;
  size_t writeIdx = 0;
  void* dstPtr = isComplete ? (void*)&completePtrs[comm->rank] : (void*)&readyPtrs[comm->rank];
  size_t offset = (uint8_t*)dstPtr - (uint8_t*)comm->ceColl.ceSyncWin->userPtr;

  NCCLCHECKGOTO(ncclCalloc(&writeParams, comm->nRanks), ret, fail);
  for (int r = 0; r < comm->nRanks; ++r) {
    if (r == comm->rank) continue;
    void* peerDstPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, r, &peerDstPtr), ret, fail);
    writeParams[writeIdx] = {};
    writeParams[writeIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    writeParams[writeIdx].writeValue.address = (CUdeviceptr)peerDstPtr;
    writeParams[writeIdx].writeValue.value = waitValue;
    writeParams[writeIdx].writeValue.flags = 0;
    writeIdx++;
  }
  if (writeIdx) {
    CUCHECKGOTO(hipStreamBatchMemOp(stream, writeIdx, writeParams, 0), ret, fail);
  }

exit:
  free(writeParams);
  return ret;
fail:
  goto exit;
}

// Non-capture: fuse peer writes into batchParams with waits (one submit, lower latency).
static ncclResult_t ncclPrepUCSyncNonCapture(struct ncclComm* comm, bool isComplete, uint32_t* readyPtrs,
                                            uint32_t* completePtrs, uint32_t waitValue,
                                            hipStreamBatchMemOpParams* batchParams, size_t* opIdx) {
  ncclResult_t ret = ncclSuccess;
  void* dstPtr = isComplete ? (void*)&completePtrs[comm->rank] : (void*)&readyPtrs[comm->rank];
  size_t offset = (uint8_t*)dstPtr - (uint8_t*)comm->ceColl.ceSyncWin->userPtr;

  for (int r = 0; r < comm->nRanks; ++r) {
    if (r == comm->rank) continue;
    void* peerDstPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, r, &peerDstPtr), ret, fail);
    batchParams[*opIdx] = {};
    batchParams[*opIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    batchParams[*opIdx].writeValue.address = (CUdeviceptr)peerDstPtr;
    batchParams[*opIdx].writeValue.value = waitValue;
    batchParams[*opIdx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
    (*opIdx)++;
  }

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclPrepUCSync(struct ncclComm* comm, bool isComplete, hipStreamBatchMemOpParams* batchParams,
                            size_t* opIdx, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  #ifdef ENABLE_FAULT_INJECTION
    NCCLCHECK(ceFaultCheck(comm, CE_FAULT_SYNC_PREP, "ncclPrepUCSync"));
  #endif

  uint32_t* readyPtrs = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);
  uint32_t currentSeq = ++comm->ceColl.ceSeqNum;
  uint32_t waitValue = capturing ? GRAPH_SYNC_VALUE : currentSeq;

  if (capturing) {
    NCCLCHECKGOTO(ncclPrepUCSyncCapture(comm, isComplete, readyPtrs, completePtrs, waitValue, stream), ret, fail);
  } else {
    NCCLCHECKGOTO(ncclPrepUCSyncNonCapture(comm, isComplete, readyPtrs, completePtrs, waitValue, batchParams, opIdx),
                  ret, fail);
  }

  // Waits always go in batchParams (submitted by ncclMemOpSync; reset follows if capturing).
  for (int r = 0; r < comm->nRanks; ++r) {
    if (r == comm->rank) continue;
    batchParams[*opIdx] = {};
    batchParams[*opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    batchParams[*opIdx].waitValue.address =
      (CUdeviceptr)(isComplete ? (void*)&completePtrs[r] : (void*)&readyPtrs[r]);
    batchParams[*opIdx].waitValue.value = waitValue;
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
    NCCLCHECKGOTO(ncclPrepUCSync(comm, comm->ceColl.useCompletePtr, batchParams, &opIdx, stream), ret, fail);
  }

  // Execute the wait batch first and let it fully drain. The reset-to-0 ops
  // MUST run strictly after every wait is satisfied: if they share one batch
  // with the waits, ROCm's hipStreamBatchMemOp does not guarantee the resets
  // wait for the wait ops, so a reset can clobber a peer's flag mid-barrier
  // and hang graph replay. Issuing them as a separate, stream-ordered batch
  // forces reset-after-wait.
  CUCHECKGOTO(hipStreamBatchMemOp(stream, opIdx, batchParams, 0), ret, fail);

  // For graph capture, reset our flag array to 0 in a separate batch so the
  // fixed-value barrier can be replayed.
  if (ncclCudaGraphValid(comm->planner.capturingGraph)) {
    size_t resetIdx = 0;
    for (int i = 0; i < comm->nRanks; i++) {
      batchParams[resetIdx] = {};
      batchParams[resetIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
      batchParams[resetIdx].writeValue.address =
        (CUdeviceptr)(comm->ceColl.useCompletePtr ? (void*)&completePtrs[i] : (void*)&readyPtrs[i]);
      batchParams[resetIdx].writeValue.value = 0;
      // CU_STREAM_WRITE_VALUE_DEFAULT is a CUDA-specific constant with no HIP equivalent.
      // This field must be initialized to satisfy the CUDA-compatible struct definition,
      // but the HIP runtime does not use this flag and treats it as 0.
      batchParams[resetIdx].writeValue.flags = 0;
      resetIdx++;
    }
    CUCHECKGOTO(hipStreamBatchMemOp(stream, resetIdx, batchParams, 0), ret, fail);
  }

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
    // Force an even sync count so useCompletePtr returns to its pre-launch
    // state; an odd count leaves the toggle flipped and desyncs the next
    // ready/complete barrier on graph replay. (ported from NCCL)
    if (params->intraBatchSync &&
        ((params->numOps + comm->ceColl.intraBatchSyncFreq - 1) / comm->ceColl.intraBatchSyncFreq) % 2 == 0) {
      NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);
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
    offset = myRecvBuff - (uint8_t*)args->recvBuff;
    if (args->useDda) {
      peerRecvBuff = (uint8_t*)args->ddaPeerBases[targetRank] + offset;
    } else {
      size_t winOff = offset + ((uint8_t*)args->recvBuff - (uint8_t*)args->recvWin->userPtr);
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, winOff, targetRank, &peerRecvBuff), ret, fail);
    }
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
      offset = dstPtr - (uint8_t*)args->recvBuff;
      if (args->useDda) {
        peerRecvBuff = (uint8_t*)args->ddaPeerBases[dstRank] + offset;
      } else {
        size_t winOff = offset + ((uint8_t*)args->recvBuff - (uint8_t*)args->recvWin->userPtr);
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, winOff, dstRank, &peerRecvBuff), ret, fail);
      }
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

ncclResult_t ncclAlltoAllvValidatePeerSendSize(size_t sendBytes, size_t peerRecvBytes, int srcRank, int dstRank) {
  if (sendBytes != peerRecvBytes) {
    WARN("CE AlltoAllv: size mismatch rank %d -> %d: sendBytes=%zu peerRecvBytes=%zu", srcRank, dstRank, sendBytes,
         peerRecvBytes);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t ncclCeAlltoAllv(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  if (args->sizes == nullptr) {
    WARN("CE AlltoAllv: missing size metadata");
    return ncclInvalidUsage;
  }

  size_t* sendSizes = ncclAlltoAllvSendSizes(args->sizes, comm->rank, comm->nRanks);
  size_t* sendDispls = ncclAlltoAllvSendDispls(args->sizes, comm->rank, comm->nRanks);
  size_t* recvDispls = ncclAlltoAllvRecvDispls(args->sizes, comm->rank, comm->nRanks);
  size_t peerDispls = 0;

  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  void* peerRecvBuff;
  size_t offset, winOff;
  size_t totalBytes = 0;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), ret, fail);

  NCCLCHECKGOTO(ncclAlltoAllvValidateSizeMatrix(args->sizes, comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  // Copy data to other ranks: send variable-sized chunk for each destination rank
  for (int r = 0; r < comm->nRanks; r++) {
    int dstRank = (comm->rank + r) % comm->nRanks;
    const size_t chunkBytes = sendSizes[dstRank];
    if (chunkBytes == 0) {
      continue;
    }

    uint8_t* srcPtr = mySendBuff + sendDispls[dstRank];
    uint8_t* dstPtr = myRecvBuff + recvDispls[comm->rank];
    totalBytes += chunkBytes;

    if (dstRank == comm->rank) {
      // Local copy for own data
      if (srcPtr != dstPtr) {
        batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
        batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
        batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
        batchOpsParams.numOps++;
      }
    } else {
      // Remote copy to other ranks: send to rank dstRank's receive buffer at position comm->rank
      size_t* peerRecvDispls = ncclAlltoAllvRecvDispls(args->sizes, dstRank, comm->nRanks);
      peerDispls = peerRecvDispls[comm->rank];

      uint8_t* dstPtr = (uint8_t*)myRecvBuff + peerDispls;
      offset = dstPtr - (uint8_t*)args->recvBuff;
      winOff = offset + ((uint8_t*)args->recvBuff - (uint8_t*)args->recvWin->userPtr);

      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, winOff, dstRank, &peerRecvBuff), ret, fail);

      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  }

  // Check if we need to perform intra-batch synchronization
  batchOpsParams.intraBatchSync =
    (batchOpsParams.numOps > comm->ceColl.intraBatchSyncFreq && totalBytes >= comm->ceColl.intraBatchSyncMsgThreshold);

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

      offset = dstPtr - (uint8_t*)args->recvBuff;
      if (args->useDda) {
        peerDstPtr = (uint8_t*)args->ddaPeerBases[dstRank] + offset;
      } else {
        size_t winOff = offset + ((uint8_t*)args->recvBuff - (uint8_t*)args->recvWin->userPtr);
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, winOff, dstRank, &peerDstPtr), ret, fail);
      }
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
    offset = rootRecvPtr - (uint8_t*)args->recvBuff;
    if (args->useDda) {
      peerRecvBuff = (uint8_t*)args->ddaPeerBases[rootLsaRank] + offset;
    } else {
      size_t winOff = offset + ((uint8_t*)args->recvBuff - (uint8_t*)args->recvWin->userPtr);
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, winOff, rootLsaRank, &peerRecvBuff), ret, fail);
    }
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

ncclResult_t ncclCeAllReduce(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                             ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream,
                             struct ncclDevrWindow* recvWin) {
  ncclResult_t ret = ncclSuccess;

  const size_t eltSize = ncclTypeSize(datatype);
  const size_t totalBytes = count * eltSize;
  const size_t shardElems = count / comm->nRanks;
  const size_t shardBytes = shardElems * eltSize;
  const size_t NUM_SLOTS = NCCL_CE_NUM_SLOTS;
  const size_t slotChunkBytes = ncclCeAllReduceSlotChunkBytes(ncclCeAllReduceMaxChunkBytes(comm->nRanks));
  if (shardElems == 0 || slotChunkBytes < eltSize) {
    WARN("CE AllReduce: no valid chunk layout (count=%zu eltSize=%zu nRanks=%d)", count, eltSize, comm->nRanks);
    return ncclInvalidArgument;
  }
  const size_t chunkBytes =
    (shardBytes <= slotChunkBytes) ? shardBytes : ncclCeAllReduceChooseChunkBytes(shardBytes, slotChunkBytes);
  size_t baseChunkElems = chunkBytes / eltSize;
  size_t slotChunkElems = slotChunkBytes / eltSize;
  size_t chunksPerShard = shardElems / baseChunkElems;
  size_t tailChunkElems = shardElems % baseChunkElems;
  if (tailChunkElems != 0) {
    chunksPerShard++;
  }
  size_t totalSteps = chunksPerShard;
  // ceARTmpBuf slots are spaced by slotChunkBytes (<= maxChunkBytes at init).
  const size_t slotStrideBytes = slotChunkBytes * (size_t)comm->nRanks;
  int startCh = (int)chunksPerShard - (int)NUM_SLOTS;
  // Unique call verification tracking sequence
  std::vector<uint32_t*> basePeerSignalAddr(comm->nRanks);
  INFO(NCCL_COLL,
       "CE AllReduce: rank %d totalBytes %zu chunkBytes=%zu baseChunkElems=%zu tailChunkElems=%zu chunksPerShard=%zu",
       comm->rank, totalBytes, chunkBytes, baseChunkElems, tailChunkElems, chunksPerShard);
  std::vector<hipStreamBatchMemOpParams> waits(comm->nRanks);
  std::vector<hipStreamBatchMemOpParams> writes(comm->nRanks);
  for (int r = 0; r < comm->nRanks; r++) {
    waits[r] = {};
    waits[r].operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    waits[r].waitValue.value = 0;
    waits[r].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;

    writes[r] = {};
    writes[r].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    writes[r].writeValue.value = 1;
    writes[r].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
  }

  struct ncclCeColl* ceColl = &comm->ceColl;
  uint8_t* tmpBuf = ceColl->ceARTmpBuf;
  uint8_t* outShard = (uint8_t*)recvbuff + (size_t)comm->rank * shardBytes; // kernel writes reduced shard here
  uint32_t* signalBuffer = ceColl->signalBuffer;
  void* peerSig = nullptr;
  void* peerSignalAddr = nullptr;
  bool fastPath = false;
  size_t mySlotOffset = 0;
  uint8_t* myRecvSlot = nullptr;
  size_t recvSlotOffset = 0;

  struct ncclCeBatchOpsParams batchOpsParams = {};
  struct ncclCeCollArgs collArgs = {};
  const int coopLaunch = rcclParamCeCoopLaunch();
  collArgs.func = ncclFuncAllReduce;
  collArgs.datatype = datatype;
  collArgs.redOp = op;
  collArgs.nElts = count;
  collArgs.eltSize = eltSize;
  collArgs.sendBuff = (uint8_t*)sendbuff;
  collArgs.recvBuff = (uint8_t*)recvbuff;
  collArgs.recvWin = recvWin;
  cudaStream_t ceStream;
  if (totalSteps > 1) {
    ceStream = ceColl->scatterStream;
  } else {
    ceStream = stream;
  }
  if (recvWin == nullptr) {
    NCCLCHECKGOTO(ncclDevrFindWindow(comm, recvbuff, &recvWin), ret, fail);
  }
  if (recvWin != nullptr && (recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC)) {
    const uintptr_t winStart = (uintptr_t)recvWin->userPtr;
    const uintptr_t recvStart = (uintptr_t)recvbuff;
    // A pointer-only window lookup is insufficient: Phase 3 writes the complete
    // receive range through peer mappings. Fall back to the temporary CE window
    // unless [recvbuff, recvbuff + totalBytes) is contained in recvWin.
    fastPath =
      recvStart >= winStart && totalBytes <= recvWin->size && recvStart - winStart <= recvWin->size - totalBytes;
  }
  collArgs.recvWin = recvWin;

  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), ret, fail);

  if (totalSteps > 1) {
    // Pipelined path uses stream for the reduce kernel and copyStream for
    // scatter/allgather. Synchronize them here so copyStream does not start until
    // prior work on the caller stream has finished (pairs with the synceEvent at exit).
    CUDACHECKGOTO(cudaEventRecord(ceColl->synceEvent, stream), ret, fail);
    CUDACHECKGOTO(cudaStreamWaitEvent(ceStream, ceColl->synceEvent, 0), ret, fail);
  }
  // Synchronize all ranks/GPUs before launching the collective operation.
  NCCLCHECKGOTO(ncclMemOpSync(comm, ceStream, &collArgs), ret, fail);
  // =========================================================================
  // STEP 1: LAUNCH PERSISTENT REDUCTION KERNEL (ONCE, BEFORE THE LOOP), IF TOTAL STEPS > 1
  // =========================================================================

  if (totalSteps > 1) {
    CUDACHECKGOTO(cudaMemsetAsync(ceColl->d_barrierSync, 0, 2 * sizeof(uint32_t), stream), ret, fail);
    NCCLCHECKGOTO(ncclCeLaunchPersistentReduce(tmpBuf, outShard, comm->nRanks, baseChunkElems, tailChunkElems,
                                               chunksPerShard, slotChunkElems, signalBuffer, totalSteps,
                                               ceColl->d_barrierSync, datatype, op, stream, coopLaunch),
                  ret, fail);
  }

  for (int ch = 0; ch < chunksPerShard; ch++) {
    INFO(NCCL_COLL, "CE AllReduce: Phase 1: rank %d totalBytes=%zu chunkBytes=%zu ch %d", comm->rank, totalBytes,
         chunkBytes, ch);
    const int slot = (int)(ch % NUM_SLOTS);
    const bool isTail = (ch == chunksPerShard - 1) && (tailChunkElems > 0);
    const size_t currentChunkBytes = isTail ? tailChunkElems * eltSize : chunkBytes;
    if (totalSteps > 1) {
      mySlotOffset = (slot * (size_t)comm->nRanks + comm->rank) * sizeof(uint32_t);
      for (int r = 0; r < comm->nRanks; r++) {
        if (r != comm->rank) {
          NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceColl->signalWin, mySlotOffset, r, &peerSig), ret, fail);
          basePeerSignalAddr[r] = (uint32_t*)peerSig;
        }
      }
    }
    // A. Verify that the persistent kernel has fully consumed this slot from the prior loop
    //
    // ceARTmpBuf is double-buffered (NUM_SLOTS slots); chunk ch uses slot ch % NUM_SLOTS.
    // The first NUM_SLOTS chunks get fresh slots — no drain wait. From chunk NUM_SLOTS on,
    // scatterStream waits for the persistent reduce kernel to clear the slot (signal == 0)
    // on all ranks before overwriting it (e.g. chunk 2 reuses slot 0 when NUM_SLOTS == 2).
    if (ch >= NUM_SLOTS) {
      for (int r = 0; r < comm->nRanks; r++) {
        if (r == comm->rank) {
          waits[r].waitValue.address = &signalBuffer[slot * comm->nRanks + comm->rank];
        } else {
          waits[r].waitValue.address = basePeerSignalAddr[r];
        }
      }
      CUCHECK(hipStreamBatchMemOp(ceStream, comm->nRanks, waits.data(), 0));
    }
    // Batched doorbell: all peers' + local slot signals stream memop.
    const size_t dstSlotOffsetBytes = (size_t)slot * slotStrideBytes + (size_t)comm->rank * slotChunkBytes;
    const size_t srcChunkOffsetBytes = ch * chunkBytes;
    batchOpsParams.numOps = 0;
    for (int r = 0; r < comm->nRanks; r++) {
      void* dstPtr;
      const uint8_t* srcShard = (const uint8_t*)sendbuff + (r * shardBytes) + srcChunkOffsetBytes;
      if (r == comm->rank) {
        dstPtr = tmpBuf + dstSlotOffsetBytes;
      } else {
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceColl->ceARTmpWin, dstSlotOffsetBytes, r, &dstPtr), ret, fail);
      }
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcShard;
      batchOpsParams.dsts[batchOpsParams.numOps] = dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = currentChunkBytes;
      batchOpsParams.numOps++;
    }
    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, ceStream, &collArgs), ret, fail);
    if (totalSteps > 1) {
      // Phase 2: write signal for reduce to start after all scatter operations are complete
      for (int r = 0; r < comm->nRanks; r++) {
        if (r == comm->rank) {
          writes[r].writeValue.address = &signalBuffer[slot * comm->nRanks + comm->rank];
        } else {
          writes[r].writeValue.address = basePeerSignalAddr[r];
        }
      }
      CUCHECK(hipStreamBatchMemOp(ceStream, comm->nRanks, writes.data(), 0));
    }
    if (totalSteps == 1) {
      NCCLCHECKGOTO(ncclMemOpSync(comm, ceStream, &collArgs), ret, fail);
      NCCLCHECKGOTO(ncclCeLaunchPersistentReduce(tmpBuf, outShard, comm->nRanks, baseChunkElems, tailChunkElems,
                                                 chunksPerShard, slotChunkElems, signalBuffer, totalSteps,
                                                 ceColl->d_barrierSync, datatype, op, ceStream, coopLaunch),
                    ret, fail);
    }
  }
  if (totalSteps > 1) {
    // Phase 3: wait for NUM_SLOTS chunks to be reduced before starting allgather
    // startch is chunkPerShard - NUM_SLOTS
    if (startCh < 0) {
      startCh = 0;
    }
    for (int chunkIndex = startCh; chunkIndex < (int)chunksPerShard; chunkIndex++) {
      const int drainSlot = chunkIndex % NUM_SLOTS;
      const size_t slotOffset = (drainSlot * (size_t)comm->nRanks + comm->rank) * sizeof(uint32_t);
      // wait signal[drainSlot] == 0 on all ranks using slotOffset for peers
      for (int r = 0; r < comm->nRanks; r++) {
        if (r == comm->rank) {
          waits[r].waitValue.address = &signalBuffer[drainSlot * comm->nRanks + comm->rank];
        } else {
          NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceColl->signalWin, slotOffset, r, &peerSig), ret, fail);
          waits[r].waitValue.address = (uint32_t*)peerSig;
        }
      }
      CUCHECK(hipStreamBatchMemOp(ceStream, comm->nRanks, waits.data(), 0));
    }
  }
  if (fastPath) {
    // Phase 3: allgather directly into the user's receive window.
    batchOpsParams.numOps = 0;
    myRecvSlot = (uint8_t*)recvbuff + (size_t)comm->rank * shardBytes;
    size_t recvWindowOffset = (size_t)((uint8_t*)recvbuff - (uint8_t*)recvWin->userPtr);
#ifdef ENABLE_FAULT_INJECTION
    if (comm->ceColl.ceFaults & CE_FAULT_LEGACY_RECV_OFFSET) {
      recvWindowOffset = 0;
    }
#endif
    recvSlotOffset = recvWindowOffset + (size_t)comm->rank * shardBytes;

    // The reduce kernel writes the reduced shard straight into recvbuff, so outShard and
    // myRecvSlot alias; copying would be a self-overlapping D2D transfer.
    if (myRecvSlot != outShard) {
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)outShard;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)myRecvSlot;
      batchOpsParams.sizes[batchOpsParams.numOps] = shardBytes;
      batchOpsParams.numOps++;
    }
    // copy from outShard to peerRecvBuff
    for (int r = 1; r < comm->nRanks; r++) {
      int targetRank = (comm->rank + r) % comm->nRanks;
      void* peerRecvBuff;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, recvSlotOffset, targetRank, &peerRecvBuff), ret, fail);
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)outShard;
      batchOpsParams.dsts[batchOpsParams.numOps] = peerRecvBuff;
      batchOpsParams.sizes[batchOpsParams.numOps] = shardBytes;
      batchOpsParams.numOps++;
    }
    batchOpsParams.intraBatchSync = (batchOpsParams.numOps > comm->ceColl.intraBatchSyncFreq &&
                                     chunkBytes * batchOpsParams.numOps >= comm->ceColl.intraBatchSyncMsgThreshold);
    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, ceStream, &collArgs), ret, fail);

    // Cross-rank barrier so each rank knows all peers' allgather writes into its
    // recvbuff are visible before returning to the user.
    NCCLCHECKGOTO(ncclMemOpSync(comm, ceStream, &collArgs), ret, fail);
  } else {
    struct ncclCeCollArgs agArgs = {};
    agArgs.func = ncclFuncAllGather;
    agArgs.nElts = shardBytes / eltSize;
    agArgs.eltSize = eltSize;
    agArgs.sendBuff = outShard;
    agArgs.recvBuff = tmpBuf;
    agArgs.sendWin = comm->ceColl.ceARTmpWin;
    agArgs.recvWin = comm->ceColl.ceARTmpWin;
    NCCLCHECKGOTO(ncclCeAllGather(comm, &agArgs, ceStream), ret, fail);

    // Phase 5 (slow path only): Local Copy — move assembled result from
    // tmpBuf to user recvbuff.
    CUDACHECKGOTO(cudaMemcpyAsync(recvbuff, tmpBuf, totalBytes, cudaMemcpyDeviceToDevice, ceStream), ret, fail);
  }

  if (totalSteps > 1) {
    CUDACHECKGOTO(cudaEventRecord(ceColl->synceEvent, ceStream), ret, fail);
    CUDACHECKGOTO(cudaStreamWaitEvent(stream, ceColl->synceEvent, 0), ret, fail);
  }
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
  case ncclFuncAlltoAllv:
    NCCLCHECKGOTO(ncclCeAlltoAllv(comm, args, stream), ret, fail);
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
    NCCLCHECKGOTO(ncclCeAllReduce(comm, args->sendBuff, args->recvBuff, args->nElts, args->datatype, args->redOp,
                                  stream, args->recvWin),
                  ret, fail);
    break;
  default:
    ret = ncclInvalidUsage;
  }
  // DDA path: results were staged in scratch (args->recvBuff). Copy them back to
  // the user's recv buffer. Copy-back semantics are collective-specific:
  //   AllGather / AlltoAll: every rank holds the full nRanks*chunk result.
  //   Gather:               only the root holds the full nRanks*chunk result.
  //   Scatter:              every rank receives a single chunk at scratch offset 0.
  if (args->useDda && args->ddaUserRecvBuff != NULL) {
    const size_t chunkBytes = args->nElts * args->eltSize;
    const size_t fullBytes = (size_t)comm->nRanks * chunkBytes;
    switch (args->func) {
    case ncclFuncGather:
      if (comm->rank == args->rootRank) {
        CUDACHECKGOTO(cudaMemcpyAsync(args->ddaUserRecvBuff, args->recvBuff /*scratch*/, fullBytes,
                                      cudaMemcpyDeviceToDevice, stream),
                      ret, fail);
      }
      break;
    case ncclFuncScatter:
      CUDACHECKGOTO(cudaMemcpyAsync(args->ddaUserRecvBuff, args->recvBuff /*scratch*/, chunkBytes,
                                    cudaMemcpyDeviceToDevice, stream),
                    ret, fail);
      break;
    default: // AllGather, AlltoAll
      CUDACHECKGOTO(cudaMemcpyAsync(args->ddaUserRecvBuff, args->recvBuff /*scratch*/, fullBytes,
                                    cudaMemcpyDeviceToDevice, stream),
                    ret, fail);
      break;
    }
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
