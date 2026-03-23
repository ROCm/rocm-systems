/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "register_inline.h"
#include <cuda.h>
#include "rocmwrap.h"
#include "ce_coll.h"
#include "alloc.h"

// Compile-time check for CUDART_VERSION
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#ifdef CUDART_VERSION
#pragma message "CUDART_VERSION = " TOSTRING(CUDART_VERSION)
#else
#pragma message "CUDART_VERSION is NOT defined (defaulting to 0)"
#define CUDART_VERSION 0
#endif

// Static constant for graph synchronization
static const uint32_t GRAPH_SYNC_VALUE = 1;

// Static constants for intra-batch synchronization to improve CE collective performance with large scale
// Frequency of intra-batch synchronization
static const uint32_t CE_COLL_INTRA_BATCH_SYNC_FREQ = 8;
// Message threshold for intra-batch synchronization
static const uint64_t CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD = 512*1024*1024;

// RCCL parameter for enabling batch memcpy in CE collectives
RCCL_PARAM(CeBatchMemcpy, "CE_BATCH_MEMCPY", 0);

ncclResult_t ncclCeInit(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  uint8_t* ceDevBase;
  size_t ceDevBaseSize = alignUp(comm->nRanks*sizeof(uint32_t), 16) * 2;
  ncclWindow_vidmem* ceWinDev;
  ncclWindow_vidmem* ceWinDevHost;
  int ceDriverVersion = 0;

  // Ensure symmetric memory runtime is initialized
  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);
  // Allocate and register memory for the symmetric memory
  NCCLCHECKGOTO(ncclMemAlloc((void**)&ceDevBase, ceDevBaseSize), ret, fail);
  NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, ceDevBase, ceDevBaseSize, NCCL_WIN_COLL_SYMMETRIC, &ceWinDev), ret, fail);
  NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, ceWinDev, &ceWinDevHost), ret, fail);
  // Get the ncclDevrWindow from the winHost field
  comm->ceColl.ceSyncWin = (struct ncclDevrWindow*)ceWinDevHost->winHost;

  comm->ceColl.baseUCSymReadyOffset = 0;
  comm->ceColl.baseUCSymComplOffset = alignUp(comm->nRanks*sizeof(uint32_t), 16);
  comm->ceColl.baseUCSymReadyPtr = (uint8_t*)comm->ceColl.ceSyncWin->userPtr + comm->ceColl.baseUCSymReadyOffset;
  comm->ceColl.baseUCSymComplPtr = (uint8_t*)comm->ceColl.ceSyncWin->userPtr + comm->ceColl.baseUCSymComplOffset;
  comm->ceColl.ceSeqNum = 0;
  comm->ceColl.useCompletePtr = false;
  comm->ceColl.intraBatchSyncFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  comm->ceColl.intraBatchSyncMsgThreshold = CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD;

<<<<<<< Updated upstream
  comm->ceColl.useBatchMemcpy = (rcclParamCeBatchMemcpy() == 1);

  // Initialize multi-stream support for non-batch path
  comm->ceColl.numStreams = 8;  // Use 8 streams
  comm->ceColl.streams = nullptr;
  if (!comm->ceColl.useBatchMemcpy) {
    // Only create streams if we're not using batch memcpy
    comm->ceColl.streams = (cudaStream_t*)malloc(comm->ceColl.numStreams * sizeof(cudaStream_t));
    if (comm->ceColl.streams == nullptr) {
      WARN("Failed to allocate memory for CE streams");
      ret = ncclSystemError;
      goto fail;
    }
    for (int i = 0; i < comm->ceColl.numStreams; i++) {
      CUDACHECKGOTO(cudaStreamCreateWithFlags(&comm->ceColl.streams[i], cudaStreamNonBlocking), ret, fail);
    }
  }

  ncclCudaDriverVersion(&ceDriverVersion);
  INFO(NCCL_INIT, "Init CE, rank %d CUDART_VERSION=%d driverVersion=%d useBatchMemcpy=%d numStreams=%d",
       comm->rank, CUDART_VERSION, ceDriverVersion, comm->ceColl.useBatchMemcpy, comm->ceColl.numStreams);
=======
  comm->ceColl.nCopyStreams = NCCL_CE_NUM_COPY_STREAMS;
  for (int i = 0; i < comm->ceColl.nCopyStreams; i++) {
    CUDACHECKGOTO(cudaStreamCreateWithFlags(&comm->ceColl.copyStreams[i], cudaStreamNonBlocking), ret, fail);
    CUDACHECKGOTO(cudaEventCreateWithFlags(&comm->ceColl.copyEvents[i], cudaEventDisableTiming), ret, fail);
  }

  INFO(NCCL_INIT, "Init CE, rank %d baseUCSymReadyPtr %p, baseUCSymComplPtr %p, seq num %d, nCopyStreams %d",
       comm->rank, comm->ceColl.baseUCSymReadyPtr, comm->ceColl.baseUCSymComplPtr, comm->ceColl.ceSeqNum, comm->ceColl.nCopyStreams);
>>>>>>> Stashed changes

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeFinalize(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  
  // Clean up ceInitTaskQueue
  while (!ncclIntruQueueEmpty(&comm->ceInitTaskQueue)) {
    struct ncclCeInitTask* task = ncclIntruQueueDequeue(&comm->ceInitTaskQueue);
    free(task);
  }
  
<<<<<<< Updated upstream
  // Clean up CE streams
  if (comm->ceColl.streams != nullptr) {
    for (int i = 0; i < comm->ceColl.numStreams; i++) {
      CUDACHECKGOTO(cudaStreamDestroy(comm->ceColl.streams[i]), ret, fail);
    }
    free(comm->ceColl.streams);
    comm->ceColl.streams = nullptr;
  }
  
=======
  // Clean up copy streams and events
  for (int i = 0; i < comm->ceColl.nCopyStreams; i++) {
    CUDACHECKIGNORE(cudaEventDestroy(comm->ceColl.copyEvents[i]));
    CUDACHECKIGNORE(cudaStreamDestroy(comm->ceColl.copyStreams[i]));
  }
  comm->ceColl.nCopyStreams = 0;

>>>>>>> Stashed changes
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

exit:
  return ret;
fail:
  goto exit;
}

bool ncclCeImplemented(ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
  int driverVersion;
  if (ncclCudaDriverVersion(&driverVersion) != ncclSuccess) return false;

  // CE is supported in CUDA 12.5 and later
  if (driverVersion >= 12050) {
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

ncclResult_t ncclPrepMCSync(struct ncclComm* comm, bool isComplete, hipStreamBatchMemOpParams* batchParams, size_t* opIdx, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  uint32_t* readyPtrs    = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
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
  CUDACHECKGOTO(cudaMemcpyAsync(
    mcDstPtr,
    srcPtr,
    sizeof(uint32_t),
    cudaMemcpyHostToDevice,
    stream), ret, fail);

  // Add local wait operations for every other rank
  for (int r = 0; r < comm->nRanks; ++r) {
    if (r == comm->rank) continue;
    batchParams[*opIdx] = {};
    // batchParams[*opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
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

ncclResult_t ncclPrepUCSync(struct ncclComm* comm, bool isComplete,
                               hipStreamBatchMemOpParams* batchParams,
                               size_t* opIdx) {
  ncclResult_t ret = ncclSuccess;

  uint32_t* readyPtrs    = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);
  uint32_t currentSeq = ++comm->ceColl.ceSeqNum;

  // Write our own ready/complete flag to remote ranks
  uint32_t waitValue = capturing ? GRAPH_SYNC_VALUE : currentSeq;
  for (int r = 0; r < comm->nRanks; ++r) {
    if (r == comm->rank) continue;
    void * peerDstPtr;
    void* dstPtr = isComplete ? (void*)&completePtrs[comm->rank] : (void*)&readyPtrs[comm->rank];
    size_t offset = (uint8_t*)dstPtr - (uint8_t*)comm->ceColl.ceSyncWin->userPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, r, &peerDstPtr), ret, fail);
    batchParams[*opIdx] = {};
    batchParams[*opIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    batchParams[*opIdx].writeValue.address  = (CUdeviceptr)peerDstPtr;
    batchParams[*opIdx].writeValue.value = waitValue;
    batchParams[*opIdx].writeValue.flags = 0;
    (*opIdx)++;
  }

  // Add local wait operations for every other rank
  for (int r = 0; r < comm->nRanks; ++r) {
    if (r == comm->rank) continue;
    batchParams[*opIdx] = {};
    batchParams[*opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    batchParams[*opIdx].waitValue.address  = (CUdeviceptr)(isComplete ? (void*)&completePtrs[r] : (void*)&readyPtrs[r]);
    batchParams[*opIdx].waitValue.value = waitValue;
    batchParams[*opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    (*opIdx)++;
  }

exit:
  return ret;
fail:
  goto exit;
}


ncclResult_t ncclMemOpSync(struct ncclComm* comm, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

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
      // batchParams[opIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
      batchParams[opIdx].writeValue.address = (CUdeviceptr)(comm->ceColl.useCompletePtr ? (void*)&completePtrs[i] : (void*)&readyPtrs[i]);
      batchParams[opIdx].writeValue.value = 0;
      // batchParams[opIdx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
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
#ifdef RCCL_ENABLE_BATCH_MEMCPY
  params->attrs = nullptr;
  params->attrIdxs = nullptr;
  params->numAttrs = 0;
#endif
  
  NCCLCHECKGOTO(ncclCalloc(&params->srcs, nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&params->dsts, nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&params->sizes, nRanks), ret, fail);
#ifdef RCCL_ENABLE_BATCH_MEMCPY
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
#ifdef RCCL_ENABLE_BATCH_MEMCPY
  if (params->attrs) free(params->attrs);
  if (params->attrIdxs) free(params->attrIdxs);
#endif
}

ncclResult_t ncclCeLaunchBatchOps(struct ncclComm* comm, struct ncclCeBatchOpsParams* params, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Check if there are any operations to perform
  if (params->numOps == 0) {
    return ncclSuccess;
  }

  // Check if we are in a CUDA graph capture
  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);
<<<<<<< Updated upstream
  int driverVersion;
  size_t totalBytes = 0;
=======
  int driverVersion = 0;
#ifdef RCCL_ENABLE_BATCH_MEMCPY
  // Check if batch memcpy is requested
  bool useBatchMemcpy = (rcclParamCeBatchMemcpy() == 1);
#endif
>>>>>>> Stashed changes

  NCCLCHECKGOTO(ncclCudaDriverVersion(&driverVersion), ret, fail);

  for (int i = 0; i < (int)params->numOps; i++) totalBytes += params->sizes[i];

  INFO(NCCL_COLL, "CE LaunchBatchOps: rank %d numOps=%d totalBytes=%zu useBatchMemcpy=%d capturing=%d intraBatchSync=%d",
       comm->rank, (int)params->numOps, totalBytes, comm->ceColl.useBatchMemcpy, capturing, params->intraBatchSync);

  //--------------Graph capture--------------
  // cudaMemcpyBatchAsync is not supported during CUDA graph capture
  if (capturing) {
<<<<<<< Updated upstream
    INFO(NCCL_COLL, "CE LaunchBatchOps: rank %d -> graph-capture path (cudaMemcpyAsync), numOps=%d", comm->rank, params->numOps);
    for (int i =0; i < params->numOps; i++) {
=======
    INFO(NCCL_COLL, "CE LaunchBatchOps: rank %d using graph-capture path (cudaMemcpyAsync), numOps=%lu", comm->rank, params->numOps);
    for (int i =0; i < (int)params->numOps; i++) {
>>>>>>> Stashed changes
      CUDACHECKGOTO(cudaMemcpyAsync(
        (void*)params->dsts[i],
        (void*)params->srcs[i],
        params->sizes[i],
        cudaMemcpyDeviceToDevice,
        stream), ret, fail);

      if (params->intraBatchSync && ((i+1) % comm->ceColl.intraBatchSyncFreq == 0) && ((i+1) < (int)params->numOps)) {
        NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);
      }
    }
  }
<<<<<<< Updated upstream
  //--------------No graph capture--------------
#ifdef RCCL_ENABLE_BATCH_MEMCPY
  else if (comm->ceColl.useBatchMemcpy) {
    //--------------Batch memcpy path (RCCL_CE_BATCH_MEMCPY=1)--------------
    if (params->intraBatchSync) {
      INFO(NCCL_COLL, "CE LaunchBatchOps: rank %d -> BATCH path WITH intraBatchSync, numOps=%d totalBytes=%zu", comm->rank, (int)params->numOps, totalBytes);
      // Break into multiple batches with sync between them
      int batchSize = comm->ceColl.intraBatchSyncFreq;
      for (int i = 0; i < (int)params->numOps; i += batchSize) {
        int currentBatchSize = (i + batchSize <= (int)params->numOps) ? batchSize : (int)params->numOps - i;
       
        CUDACHECKGOTO(cudaMemcpyBatchAsync(
          &params->dsts[i], &params->srcs[i], &params->sizes[i], currentBatchSize,
          params->attrs, params->attrIdxs, params->numAttrs, nullptr, 0, stream), ret, fail);
      
        // Sync after each batch except the last one
        if (i + batchSize < (int)params->numOps) {
          NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);
        }
      }
    } else {
      INFO(NCCL_COLL, "CE LaunchBatchOps: rank %d -> BATCH path WITHOUT intraBatchSync, numOps=%d totalBytes=%zu", comm->rank, (int)params->numOps, totalBytes);
      // Use single batch for all operations
      CUDACHECKGOTO(cudaMemcpyBatchAsync(
        params->dsts, params->srcs, params->sizes, params->numOps,
        params->attrs, params->attrIdxs, params->numAttrs, nullptr, stream), ret, fail);
    }
  }
#endif
  else {
    //--------------Non-batch path: individual cudaMemcpyAsync (default)--------------
    if (comm->ceColl.streams != nullptr && comm->ceColl.numStreams > 1) {
      // Multi-stream path: distribute operations across 8 streams
      INFO(NCCL_COLL, "CE LaunchBatchOps: rank %d -> NON-BATCH MULTI-STREAM path (%d streams), numOps=%d totalBytes=%zu", 
           comm->rank, comm->ceColl.numStreams, (int)params->numOps, totalBytes);
      
      for (int i = 0; i < (int)params->numOps; i++) {
        // Round-robin distribution across streams
        int streamIdx = i % comm->ceColl.numStreams;
        CUDACHECKGOTO(cudaMemcpyAsync(
          (void*)params->dsts[i],
          (void*)params->srcs[i],
          params->sizes[i],
          cudaMemcpyDeviceToDevice,
          comm->ceColl.streams[streamIdx]), ret, fail);
      }
      
      // Synchronize all streams at the end
      for (int s = 0; s < comm->ceColl.numStreams; s++) {
        CUDACHECKGOTO(cudaStreamSynchronize(comm->ceColl.streams[s]), ret, fail);
      }
    } else {
      // Single-stream fallback
      INFO(NCCL_COLL, "CE LaunchBatchOps: rank %d -> NON-BATCH SINGLE-STREAM path (cudaMemcpyAsync), numOps=%d totalBytes=%zu", comm->rank, (int)params->numOps, totalBytes);
=======
  //--------------Batch memcpy path--------------
#ifdef RCCL_ENABLE_BATCH_MEMCPY
  else if (useBatchMemcpy && driverVersion >= 12080) {
    INFO(NCCL_COLL, "CE LaunchBatchOps: rank %d using BATCH memcpy path (hipMemcpyBatchAsync), numOps=%lu", comm->rank, params->numOps);
    
    // Use hipMemcpyBatchAsync for better performance
    const size_t maxBatchSize = 1024;
    for (size_t batchStart = 0; batchStart < params->numOps; batchStart += maxBatchSize) {
      size_t currentBatchSize = (batchStart + maxBatchSize > params->numOps) ? 
                                (params->numOps - batchStart) : maxBatchSize;
      
      CUDACHECKGOTO(hipMemcpyBatchAsync(
        (void**)&params->dsts[batchStart],
        (void**)&params->srcs[batchStart],
        &params->sizes[batchStart],
        currentBatchSize,
        nullptr,        // attrs (nullptr for default D2D)
        nullptr,        // attrsIdxs
        0,              // numAttrs
        nullptr,        // failIdx
        stream), ret, fail);
      
      if (params->intraBatchSync && ((batchStart + currentBatchSize) % comm->ceColl.intraBatchSyncFreq == 0) && 
          ((batchStart + currentBatchSize) < params->numOps)) {
        NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);
      }
    }
  }
#endif
  //--------------No graph capture: multi-stream parallel copies--------------
  else {
    int nStreams = comm->ceColl.nCopyStreams;
    if (nStreams <= 0 || (int)params->numOps <= 1) {
      INFO(NCCL_COLL, "CE LaunchBatchOps: rank %d using single-stream cudaMemcpyAsync, numOps=%lu", comm->rank, params->numOps);
>>>>>>> Stashed changes
      for (int i = 0; i < (int)params->numOps; i++) {
        CUDACHECKGOTO(cudaMemcpyAsync(
          (void*)params->dsts[i],
          (void*)params->srcs[i],
          params->sizes[i],
          cudaMemcpyDeviceToDevice,
          stream), ret, fail);
      }
    } else {
      int activeStreams = ((int)params->numOps < nStreams) ? (int)params->numOps : nStreams;
      INFO(NCCL_COLL, "CE LaunchBatchOps: rank %d using MULTI-STREAM cudaMemcpyAsync, numOps=%lu, nStreams=%d", comm->rank, params->numOps, activeStreams);

      // Make copy streams wait on the main stream
      for (int s = 0; s < activeStreams; s++) {
        CUDACHECKGOTO(cudaEventRecord(comm->ceColl.copyEvents[s], stream), ret, fail);
        CUDACHECKGOTO(cudaStreamWaitEvent(comm->ceColl.copyStreams[s], comm->ceColl.copyEvents[s], 0), ret, fail);
      }

      // Distribute copies round-robin across streams
      for (int i = 0; i < (int)params->numOps; i++) {
        int s = i % activeStreams;
        CUDACHECKGOTO(cudaMemcpyAsync(
          (void*)params->dsts[i],
          (void*)params->srcs[i],
          params->sizes[i],
          cudaMemcpyDeviceToDevice,
          comm->ceColl.copyStreams[s]), ret, fail);
      }

<<<<<<< Updated upstream
        if (params->intraBatchSync && ((i+1) % comm->ceColl.intraBatchSyncFreq == 0) && ((i+1) < (int)params->numOps)) {
          NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);
        }
=======
      // Make main stream wait on all copy streams
      for (int s = 0; s < activeStreams; s++) {
        CUDACHECKGOTO(cudaEventRecord(comm->ceColl.copyEvents[s], comm->ceColl.copyStreams[s]), ret, fail);
        CUDACHECKGOTO(cudaStreamWaitEvent(stream, comm->ceColl.copyEvents[s], 0), ret, fail);
>>>>>>> Stashed changes
      }
    }
  }

exit:
  return ret;
fail:
  goto exit;
}


ncclResult_t ncclCeAllGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  
  // Calculate the size of each rank's data chunk
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff + comm->rank * chunkBytes;
  void* peerRecvBuff;
  size_t offset;

  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);

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
  batchOpsParams.intraBatchSync = (batchOpsParams.numOps > comm->ceColl.intraBatchSyncFreq && chunkBytes*batchOpsParams.numOps >= comm->ceColl.intraBatchSyncMsgThreshold);

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);

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
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks * comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);

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
  batchOpsParams.intraBatchSync = (batchOpsParams.numOps > comm->ceColl.intraBatchSyncFreq && chunkBytes*batchOpsParams.numOps >= comm->ceColl.intraBatchSyncMsgThreshold);

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeScatter(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  
  // Calculate the size of data root sends to each rank
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  int rootRank = args->rootRank;
  void* peerDstPtr;
  size_t offset;

  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);

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
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  
  // Calculate the size of data each rank sends to root
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  int rootRank = args->rootRank;
  void* peerRecvBuff;
  size_t offset;

  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, 1), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);

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
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream), ret, fail);

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
  return ret;
fail:
  goto exit;
}
