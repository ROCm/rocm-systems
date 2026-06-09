/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <assert.h>
#include "nccl.h"
#include "comm.h"
#include "gin/gin_host.h"
#include "alloc.h"
#include "checks.h"
#include "gdrwrap.h"
#include "nccl_device/gin/proxy/gin_proxy_device_host_common.h"
#include "compiler.h"

NCCL_PARAM(GinProxyQueueSize, "GIN_PROXY_QUEUE_SIZE", -1);
extern int64_t ncclParamIbDataDirect();
extern int64_t ncclParamDmaBufEnable();

struct ginProxyGfdState {
  ncclGinProxyOp_t op;
  uint16_t counterId;
  int done;
  void *request;
};

// a member might be on the GPU, if it has a *GdrHandle counterpart
struct ginProxyHostGpuCtx {
  int contextId;
  size_t queueSize;

  // size = nRanks * queueSize
  ncclGinProxyGfd_t *queues;
  void *cisGdrHandle;
  // Consumed Indices, one per rank
  uint32_t *cis;
  // to decrease the number of reads/writes to cis which might be on the GPU
  uint32_t *cisShadow;
  // Seen Indices one per rank
  uint32_t *sis;

  // same size as queues
  struct ginProxyGfdState *states;
  // same size as queues
  uint64_t *inlines;
  // inlines is registered as a memory region with the GIN plugin
  void *inlinesMhandle;
  void *inlinesGinHandle;
};

struct ginProxyCtx {
  struct ncclComm *comm;
  void *collComm;
  ncclNetDeviceHandle_t *devHandle;
  ncclNetProperties_t props;

  // GPU queues, if GDR on the GPU, else on the CPU
  // Queue size, must be a power of 2
  struct ginProxyHostGpuCtx *hostGpuCtx;

  void *countersGdrHandle;
  uint64_t *counters;
  uint64_t *countersDev;
  CUmemGenericAllocationHandle signalsCumemhandle;
  void *signalsMhandle;
  void *signalsGinHandle;
  uint64_t *signalsDev;
  int hasError;
  int nContexts;
  int nCountersPerContext;
  int nSignalsPerContext;
};

// Depending on GDR, allocate memory on the CPU or GPU.
// host_flags is not used for now, but it is here for future use.
template <typename T>
static ncclResult_t allocMemCPUAccessible(T **ptr, T **devPtr, size_t nelem, int host_flags,
                                          void **gdrHandle, bool forceHost = false) {
  if (ncclGdrCopy && !forceHost) {
    NCCLCHECK(ncclGdrCudaCalloc(ptr, devPtr, nelem, gdrHandle, NULL));
  } else {
    NCCLCHECK(ncclCuMemHostAlloc((void **)ptr, NULL, nelem * sizeof(T)));
    memset((void *)*ptr, 0, nelem * sizeof(T));
    *devPtr = *ptr;
    if (gdrHandle) *gdrHandle = NULL;  // Mark as host allocated by nulling GDR handle
  }
  return ncclSuccess;
}

// [RCCL] Manager-aware overload added by the NCCL 2.29.7 sync; the manager
// argument is currently ignored (memory tracking lives in mem_manager.cc and
// isn't wired through this path yet).
template <typename T>
static ncclResult_t allocMemCPUAccessible(T **ptr, T **devPtr, size_t nelem, int host_flags,
                                          void **gdrHandle, struct ncclMemManager* /*manager*/,
                                          bool forceHost = false) {
  return allocMemCPUAccessible(ptr, devPtr, nelem, host_flags, gdrHandle, forceHost);
}

// Depending on GDR, free memory on the CPU or GPU.
template <typename T>
static ncclResult_t freeMemCPUAccessible(T *ptr, void *gdrHandle) {
  if (gdrHandle != NULL) {  // If a GDR handle exists, it was GDR memory
    NCCLCHECK(ncclGdrCudaFree(gdrHandle, NULL));
  } else {  // Otherwise, it was host memory (or GDR was off)
    NCCLCHECK(ncclCuMemHostFree(ptr));
  }
  return ncclSuccess;
}

// [RCCL] Manager-aware overload added by the NCCL 2.29.7 sync; the manager
// argument is currently ignored.
template <typename T>
static ncclResult_t freeMemCPUAccessible(T *ptr, void *gdrHandle, struct ncclMemManager* /*manager*/) {
  return freeMemCPUAccessible(ptr, gdrHandle);
}

static ncclResult_t getDmaBufFd(void *addr, size_t length, int *fd,
                                bool forceNonDataDirect = false) {
  if (ncclParamDmaBufEnable() == 0) return ncclInvalidUsage;

  // GIN's symmetric windows are cuMem/VMM allocations registered with the NIC via ibv_reg_dmabuf_mr.
  // the cuMem/hipMemGetHandleForAddressRange that exports the DMA-BUF FD requires this HIP version.
#if CUDA_VERSION >= 11070 || HIP_VERSION >= 71260540
  static size_t hostPageSize = sysconf(_SC_PAGESIZE);
  size_t alignedSize = length;
  ALIGN_SIZE(alignedSize, hostPageSize);

#if CUDA_VERSION >= 12080
  if (ncclParamIbDataDirect() && !forceNonDataDirect) {
    CUresult status = pfn_cuMemGetHandleForAddressRange(
      (void *)fd, (CUdeviceptr)addr, alignedSize, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
      CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE);
    if (status == CUDA_SUCCESS) return ncclSuccess;
  }
#endif

#if defined(__HIP_PLATFORM_AMD__)
  // Direct call: hipified to hipMemGetHandleForAddressRange on HIP at build time.
  // Same pattern as transport/net.cc and transport/coll_net.cc.
  CUresult status = cuMemGetHandleForAddressRange((void *)fd, (CUdeviceptr)addr, alignedSize,
                                                  CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
#else
  CUresult status = pfn_cuMemGetHandleForAddressRange((void *)fd, (CUdeviceptr)addr, alignedSize,
                                                      CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
#endif
  if (status == CUDA_SUCCESS) return ncclSuccess;
#endif

  return ncclInvalidUsage;
}

static ncclResult_t proxyGinPollCompletions(ncclGin_t *ginComm, void *collComm,
                                            struct ginProxyCtx *ctx,
                                            struct ginProxyHostGpuCtx *hostGpuCtx) {
  for (int targetRank = 0; targetRank < ctx->comm->nRanks; targetRank++) {
    // loop on all seen but unconsumed GFDs
    for (uint32_t i = hostGpuCtx->cisShadow[targetRank]; i < hostGpuCtx->sis[targetRank]; i++) {
      uint32_t idx = i & (hostGpuCtx->queueSize - 1);
      struct ginProxyGfdState *state =
        &hostGpuCtx->states[targetRank * hostGpuCtx->queueSize + idx];
      // no need to poll if already done
      if (!state->done) {
        ginComm->test(collComm, state->request, &state->done);
        if (state->done) {
          TRACE(NCCL_NET, "GFD completed - contextId: %d, stateIdx: %lu, request: %p", hostGpuCtx->contextId, state - hostGpuCtx->states,
                state->request);
          // update the counter specified in the GFD
          if (state->op & ncclGinProxyOpWithCounter) {
            int contextId = hostGpuCtx->contextId;
            uint64_t* counterPtr = &ctx->counters[contextId * ctx->nCountersPerContext + state->counterId];
            COMPILER_ATOMIC_STORE(counterPtr, *counterPtr + 1,
                              std::memory_order_relaxed);
            TRACE(NCCL_NET, "Updated counter %d to %ld for context %d", state->counterId,
                  *counterPtr, contextId);
          }
        }
      }
      // allow holes in the CI space to get resolved
      if (state->done && i == hostGpuCtx->cisShadow[targetRank]) {
        // tell the GPU that we have consumed the GFD
        COMPILER_ATOMIC_STORE(&hostGpuCtx->cis[targetRank], ++hostGpuCtx->cisShadow[targetRank],
                          std::memory_order_relaxed);
        TRACE(NCCL_NET, "Updated cis[%u] to %u for context %d", targetRank, hostGpuCtx->cisShadow[targetRank], hostGpuCtx->contextId);
      }
    }
  }

  return ncclSuccess;
}

static int proxyGinPollGfd(struct ginProxyCtx *ctx, ginProxyHostGpuCtx *hostGpuCtx, int targetRank,
                           ncclGinProxyGfd_t *gfd, struct ginProxyGfdState **state) {
  ncclGinProxyGfd_t *q = hostGpuCtx->queues + targetRank * hostGpuCtx->queueSize;
  uint32_t idx = hostGpuCtx->sis[targetRank] & (hostGpuCtx->queueSize - 1);
  ncclGinProxyQword_t qword;
#if defined(__HIP_PLATFORM_AMD__)
  uint64_t *headerPtr = (uint64_t *)__builtin_assume_aligned(&q[idx].qword[ncclGinProxyGfdHeader].raw, alignof(uint64_t));
  qword.raw = __atomic_load_n(headerPtr, __ATOMIC_RELAXED);
#else
  COMPILER_ATOMIC_LOAD_DEST(&q[idx].qword[ncclGinProxyGfdHeader].raw, &qword.raw, std::memory_order_relaxed);
#endif
  if (qword.flag.v == 0) {
    return 0;
  }

  // We know for sure that the first qword is there, copy it.
  gfd->qword[ncclGinProxyGfdHeader] = q[idx].qword[ncclGinProxyGfdHeader];
  // Wait for and copy the other qwords.
  for (int k = 1; k < ncclGinProxyGfdQwords; k++) {
#if defined(__HIP_PLATFORM_AMD__)
    uint64_t *qwordPtr = (uint64_t *)__builtin_assume_aligned(&q[idx].qword[k].raw, alignof(uint64_t));
    do {
      qword.raw = __atomic_load_n(qwordPtr, __ATOMIC_RELAXED);
    } while (qword.flag.v == 0);
#else
    do {
      COMPILER_ATOMIC_LOAD_DEST(&q[idx].qword[k].raw, &qword.raw, std::memory_order_relaxed);
    } while (qword.flag.v == 0);
#endif
    gfd->qword[k] = qword;
  }
  // Now we have the full GFD in the local struct.

  // Reset the GFD in the queue. This lets the producer know that the GFD is consumed.
  // On HIP, NT-stores avoid an RFO stall behind in-flight PCIe writes from the GPU
  // (can cost 80-200us under multi-GPU contention).
#if defined(__HIP_PLATFORM_AMD__)
  for (int k = 0; k < ncclGinProxyGfdQwords; k++) {
    __builtin_nontemporal_store((uint64_t)0, &q[idx].qword[k].raw);
  }
  // Drain WC buffers so the NT-zero stores are visible before the credit (ci) advance,
  // otherwise the GPU producer could overwrite the slot before the zeros land.
  wc_store_fence();
#else
  for (int k = 0; k < ncclGinProxyGfdQwords; k++) {
    COMPILER_ATOMIC_STORE(&q[idx].qword[k].raw, 0, std::memory_order_relaxed);
  }
#endif

  // set the counter_id into the state
  uint32_t stateIdx = targetRank * hostGpuCtx->queueSize + idx;
  *state = &hostGpuCtx->states[stateIdx];
  (*state)->op = (ncclGinProxyOp_t)(gfd->qword[ncclGinProxyGfdHeader].header.op);
  (*state)->counterId = gfd->qword[ncclGinProxyGfdCompletion].completion.counterId;
  (*state)->done = 0;
  (*state)->request = NULL;

  TRACE(NCCL_NET,
        "GFD on context %d to target PE %d raw idx: %u, idx: %u - op: %#lx, size: %lu, srcOff: %lu, dstOff: %lu, "
        "srcHandle: %lu, dstHandle: %lu, counterId: %u, signalId: %u, stateIdx: %u",
        hostGpuCtx->contextId, targetRank, hostGpuCtx->sis[targetRank], idx, gfd->qword[ncclGinProxyGfdHeader].header.op,
        gfd->qword[ncclGinProxyGfdHeader].header.size,
        gfd->qword[ncclGinProxyGfdSrcOff].srcOff.srcOff,
        gfd->qword[ncclGinProxyGfdDstOff].dstOff.dstOff,
        gfd->qword[ncclGinProxyGfdSrcHandle].srcHandle.srcHandle,
        gfd->qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle,
        gfd->qword[ncclGinProxyGfdCompletion].completion.counterId,
        gfd->qword[ncclGinProxyGfdCompletion].completion.signalId, stateIdx);

  hostGpuCtx->sis[targetRank]++;

  return 1;
}

static int mapGfdOpToSignalOp(ncclGinProxyGfd_t *gfd) {
  // Mask down to the signal-op bits only. WithInline and WithCounter are
  // orthogonal modifiers and must be excluded -- otherwise an inline put
  // with SignalInc (op = Put|WithInline|WithSignalInc = 0x0b) masks to
  // 0x0a and falls through to default, silently demoting iputSignal to a
  // plain iput so the remote signal cell never gets bumped.
  uint8_t op = gfd->qword[ncclGinProxyGfdHeader].header.op;
  uint8_t signalOp = op & (ncclGinProxyOpWithSignalInc | ncclGinProxyOpWithSignalAdd);
  switch (signalOp) {
    case ncclGinProxyOpWithSignalInc:
      return NCCL_NET_SIGNAL_OP_INC;
    case ncclGinProxyOpWithSignalAdd:
      return NCCL_NET_SIGNAL_OP_ADD;
    default:
      return -1;
  }
}

static inline uint64_t extractSignalVal(ncclGinProxyGfd_t *gfd) {
  uint64_t signalVal = gfd->qword[ncclGinProxyGfdCompletion].completion.signalValLow;
  signalVal |= (uint64_t)gfd->qword[ncclGinProxyGfdSignalVal].signalVal.signalValLow2 << 16;
  signalVal |= (uint64_t)gfd->qword[ncclGinProxyGfdSignalVal].signalVal.signalValHigh << 32;
  return signalVal;
}

static ncclResult_t proxyGinProcessGfd(ncclGin_t *ginComm, void *collComm, struct ginProxyCtx *ctx,
                                       struct ginProxyHostGpuCtx *hostGpuCtx, int targetRank,
                                       ncclGinProxyGfd_t *gfd, struct ginProxyGfdState *state) {
  int signalOp;
  uint64_t signalVal;

  // Handle VA Signal operations (signal-only, no PUT)
  if (gfd->qword[ncclGinProxyGfdHeader].header.op & ncclGinProxyOpVASignal) {
    uint64_t signalOff = gfd->qword[ncclGinProxyGfdVASignalOff].vaSignalOff.vaSignalOff;
    void *signalHandle = (void *)(uint64_t)gfd->qword[ncclGinProxyGfdVASignalHandle].vaSignalHandle.vaSignalHandle;
    signalVal = extractSignalVal(gfd);
    signalOp = mapGfdOpToSignalOp(gfd);
    NCCLCHECK(ginComm->iputSignal(collComm, 0, nullptr, 0, 0, nullptr,
                                  targetRank, signalOff, signalHandle, signalVal,
                                  signalOp, hostGpuCtx->contextId, &state->request));
    return ncclSuccess;
  }

  uint64_t size = gfd->qword[ncclGinProxyGfdHeader].header.size;
  uint64_t srcOff;
  void *srcHandle;
  if (gfd->qword[ncclGinProxyGfdHeader].header.op & ncclGinProxyOpWithInline) {
    // `gfd` is a stack-local copy filled by proxyGinPollGfd, not a pointer
    // into hostGpuCtx->queues, so we cannot do `gfd - hostGpuCtx->queues` to
    // recover its slot index. Recover it from `state` instead, which is a
    // real heap pointer (`*state = &hostGpuCtx->states[stateIdx]`).
    size_t slotIdx = state - hostGpuCtx->states;
    uint64_t *inlineVal = &hostGpuCtx->inlines[slotIdx];
    srcOff = slotIdx * sizeof(uint64_t);
    // reconstruct the inline value from the two qwords
    *inlineVal = gfd->qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow;
    if (size > 4)
      *inlineVal |= (uint64_t)gfd->qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow2 << 32;
    if (size > 6)
      *inlineVal |= (uint64_t)gfd->qword[ncclGinProxyGfdInlineHigh].inlineHigh.inlineValHigh << 48;
    srcHandle = hostGpuCtx->inlinesMhandle;
  } else {
    srcOff = gfd->qword[ncclGinProxyGfdSrcOff].srcOff.srcOff;
    srcHandle = (void *)(uint64_t)gfd->qword[ncclGinProxyGfdSrcHandle].srcHandle.srcHandle;
  }
  uint64_t dstOff = gfd->qword[ncclGinProxyGfdDstOff].dstOff.dstOff;
  void *dstHandle = (void *)(uint64_t)gfd->qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle;

  switch (gfd->qword[ncclGinProxyGfdHeader].header.op & ncclGinProxyOpBaseMask) {
    case ncclGinProxyOpPut:
      signalOp = mapGfdOpToSignalOp(gfd);
      if (signalOp == -1) {
        // First cast from 63 bits to 64 bits and then to void * to avoid warnings
        NCCLCHECK(ginComm->iput(collComm, srcOff, srcHandle, size, dstOff, dstHandle,
                                targetRank, hostGpuCtx->contextId, &state->request));
      } else {
        // Reconstruct the signal value
        signalVal = extractSignalVal(gfd);
        uint64_t signalOff = (gfd->qword[ncclGinProxyGfdCompletion].completion.signalId +
                              hostGpuCtx->contextId * ctx->nSignalsPerContext) * sizeof(uint64_t);
        NCCLCHECK(ginComm->iputSignal(collComm, srcOff, srcHandle, size, dstOff, dstHandle,
                                      targetRank, signalOff, ctx->signalsGinHandle, signalVal,
                                      signalOp, hostGpuCtx->contextId, &state->request));
      }
      break;
    default:
      // this error should already have been checked in pollGfd
      assert(0);
  }
  TRACE(NCCL_NET, "GFD submitted into GIN plugin - contextId: %d, stateIdx: %lu, request: %p",
        hostGpuCtx->contextId, state - hostGpuCtx->states, state->request);
  return ncclSuccess;
}

static uint64_t isPowerOfTwo(uint64_t n) { return (n > 0) && ((n & (n - 1)) == 0); }

// Check if the GIN plugin supports DMA-BUF, if so we can try to get the DMA-BUF handle from CUDA,
// if that fails we fallback to non-DMA-BUF
static ncclResult_t ncclGinProxyRegMrSym(ncclGin_t *ginComm, struct ginProxyCtx *ctx, void *addr,
                                         size_t size, int type, int mr_flags, void **mhandle,
                                         void **ginHandle) {
  if (type == NCCL_PTR_HOST) {
    NCCLCHECK(ginComm->regMrSym(ctx->collComm, addr, size, type, mr_flags, mhandle, ginHandle));
  } else if (type == NCCL_PTR_CUDA) {
    ncclResult_t dmabufResult = ncclInvalidUsage;
    if (ncclParamDmaBufEnable() && (ctx->props.ptrSupport & NCCL_PTR_DMABUF)) {
      ncclResult_t registrationResult = ncclSuccess;
      int dmabufFd = -1;
      dmabufResult = getDmaBufFd(addr, size, &dmabufFd);
      if (dmabufResult == ncclSuccess) {
        registrationResult = ginComm->regMrSymDmaBuf(ctx->collComm, addr, size, type, 0, dmabufFd,
                                                     mr_flags, mhandle, ginHandle);
        close(dmabufFd);
      }
      if (registrationResult != ncclSuccess) {
        dmabufFd = -1;
        dmabufResult = getDmaBufFd(addr, size, &dmabufFd, true);
        if (dmabufResult == ncclSuccess) {
          NCCLCHECK(ginComm->regMrSymDmaBuf(ctx->collComm, addr, size, type, 0, dmabufFd,
                                            mr_flags, mhandle, ginHandle));
          close(dmabufFd);
        }
      }
    }
    // Fallback to non-DMA-BUF if the DMA-BUF handle is not supported
    if (dmabufResult != ncclSuccess) {
      NCCLCHECK(ginComm->regMrSym(ctx->collComm, addr, size, type, mr_flags, mhandle, ginHandle));
    }
  } else {
    return ncclInvalidUsage;
  }

  return ncclSuccess;
}

ncclResult_t ncclGinProxyCreateContext(struct ncclComm *comm, void *collComm, int devId,
                                       int nSignals, int nCounters, int nContexts, void **outGinCtx,
                                       ncclNetDeviceHandle_t **outDevHandle) {
  ncclGin_t *ginComm = (ncclGin_t *)comm->sharedRes->ginState.ncclGin;
  ncclGinProxyGpuCtx_t *devGpuCtxArray_h = nullptr;

  if (!ncclGdrCopy)
    INFO(NCCL_NET, "GIN Proxy will not be using GDRCopy");

  struct ginProxyCtx *proxyCtx = NULL;
  NCCLCHECK(ncclCalloc(&proxyCtx, 1));

  proxyCtx->comm = comm;
  proxyCtx->collComm = collComm;
  proxyCtx->nContexts = nContexts;

  // Sanitize the queue size
  NCCLCHECK(ginComm->getProperties(devId, &proxyCtx->props));
  uint64_t queueSize = ncclParamGinProxyQueueSize();
  uint32_t maxRequests = NCCL_NET_MAX_REQUESTS * proxyCtx->props.maxRecvs;
  if (queueSize == -1) {
    queueSize = maxRequests;
  }
  if (queueSize > maxRequests) {
    INFO(NCCL_NET,
         "NCCL_GIN_PROXY_QUEUE_SIZE is greater than the maximum outstanding requests in the GIN "
         "plugin (%d), using the default/maximum value instead",
         maxRequests);
    queueSize = maxRequests;
  }
  if (queueSize < 1) {
    INFO(NCCL_NET,
         "NCCL_GIN_PROXY_QUEUE_SIZE is less than 1, using the default/maximum value instead");
    queueSize = maxRequests;
  }
  if (!isPowerOfTwo(queueSize)) {
    INFO(
      NCCL_NET,
      "NCCL_GIN_PROXY_QUEUE_SIZE is not a power of two, using the default/maximum value instead");
    queueSize = maxRequests;
  }

  // Allocate the counters on the GPU or CPU depending on GDR
  NCCLCHECK(allocMemCPUAccessible(&proxyCtx->counters, &proxyCtx->countersDev,
                                  nCounters * nContexts, CU_MEMHOSTALLOC_WRITECOMBINED,
                                  &proxyCtx->countersGdrHandle, comm->memManager));
  proxyCtx->nCountersPerContext = nCounters;

  // Allocate the signals on the GPU and then register the memory region with the GIN plugin.
  // Enforcing strong ordering on the signals mr is vital to ensure ordering between puts and
  // signals.
  size_t signalsBufSize = nSignals * nContexts * sizeof(uint64_t);
  NCCLCHECK(ncclCuMemAlloc((void **)&proxyCtx->signalsDev, &proxyCtx->signalsCumemhandle,
                           CU_MEM_HANDLE_TYPE_NONE, signalsBufSize, comm->memManager));
  CUDACHECK(cudaMemset(proxyCtx->signalsDev, 0, signalsBufSize));
  NCCLCHECK(ncclGinProxyRegMrSym(ginComm, proxyCtx, proxyCtx->signalsDev, signalsBufSize,
                                 NCCL_PTR_CUDA, NCCL_NET_MR_FLAG_FORCE_SO,
                                 &proxyCtx->signalsMhandle, &proxyCtx->signalsGinHandle));
  proxyCtx->nSignalsPerContext = nSignals;

  NCCLCHECK(ncclCalloc(&proxyCtx->hostGpuCtx, nContexts));
  NCCLCHECK(ncclCalloc(&devGpuCtxArray_h, nContexts));
  for (int contextId = 0; contextId < nContexts; contextId++) {
    struct ginProxyHostGpuCtx *hostGpuCtx = proxyCtx->hostGpuCtx + contextId;
    hostGpuCtx->contextId = contextId;
    hostGpuCtx->queueSize = queueSize;
    size_t queuesLength = hostGpuCtx->queueSize * comm->nRanks;
    NCCLCHECK(ncclCalloc(&hostGpuCtx->states, queuesLength));
    NCCLCHECK(ncclCalloc(&hostGpuCtx->cisShadow, comm->nRanks));
    NCCLCHECK(ncclCalloc(&hostGpuCtx->sis, comm->nRanks));
    NCCLCHECK(ncclCalloc(&hostGpuCtx->inlines, queuesLength));
    NCCLCHECK(ncclGinProxyRegMrSym(ginComm, proxyCtx, hostGpuCtx->inlines,
                                   queuesLength * sizeof(uint64_t), NCCL_PTR_HOST, 0,
                                   &hostGpuCtx->inlinesMhandle, &hostGpuCtx->inlinesGinHandle));

    ncclGinProxyGpuCtx_t *devGpuCtx_h = devGpuCtxArray_h + contextId;
    devGpuCtx_h->nranks = comm->nRanks;
    devGpuCtx_h->queueSize = hostGpuCtx->queueSize;
    devGpuCtx_h->counters = proxyCtx->countersDev + contextId * nCounters;
    devGpuCtx_h->signals = proxyCtx->signalsDev + contextId * nSignals;
    NCCLCHECK(ncclCudaCalloc(&devGpuCtx_h->pis, comm->nRanks, comm->memManager));

    // Allocate the GFD queues, CIs, counters, signals and test/wait variables on the either the CPU
    // or GPU.
    NCCLCHECK(allocMemCPUAccessible(&hostGpuCtx->queues, &devGpuCtx_h->queues, queuesLength, 0, NULL,
                                    comm->memManager, true /*forceHost*/));
    NCCLCHECK(allocMemCPUAccessible(&hostGpuCtx->cis, &devGpuCtx_h->cis, comm->nRanks,
                                    CU_MEMHOSTALLOC_WRITECOMBINED, &hostGpuCtx->cisGdrHandle, comm->memManager));
  }

  ncclGinProxyGpuCtx_t *devGpuCtx_d = NULL;
  NCCLCHECK(ncclCudaCalloc(&devGpuCtx_d, nContexts, comm->memManager));
  // Copy the proxy's devGpuCtx to the GPU
  NCCLCHECK(ncclCudaMemcpy(devGpuCtx_d, devGpuCtxArray_h, nContexts));

  ncclNetDeviceHandle_t *devHandle = NULL;
  NCCLCHECK(ncclCalloc(&devHandle, 1));
  devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  devHandle->netDeviceVersion = NCCL_GIN_PROXY_VERSION;
  devHandle->handle = (void *)devGpuCtx_d;
  devHandle->size = 0;
  devHandle->needsProxyProgress = 1;

  proxyCtx->devHandle = devHandle;

  *outDevHandle = devHandle;
  *outGinCtx = proxyCtx;

  free(devGpuCtxArray_h);

  return ncclSuccess;
}

ncclResult_t ncclGinProxyRegister(ncclGin_t *ginComm, void *ginCtx, void *addr, size_t size,
                                  int type, int mr_flags, void **mhandle, void **ginHandle) {
  struct ginProxyCtx *ctx = (struct ginProxyCtx *)ginCtx;
  // Register the memory region with the GIN plugin
  NCCLCHECK(ncclGinProxyRegMrSym(ginComm, ctx, addr, size, type, mr_flags, mhandle, ginHandle));
  return ncclSuccess;
}

ncclResult_t ncclGinProxyDeregister(ncclGin_t *ginComm, void *ginCtx, void *mhandle) {
  struct ginProxyCtx *ctx = (struct ginProxyCtx *)ginCtx;
  // Deregister the memory region with the GIN plugin
  NCCLCHECK(ginComm->deregMrSym(ctx->collComm, mhandle));
  return ncclSuccess;
}

ncclResult_t ncclGinProxyDestroyContext(ncclGin_t *ginComm, void *ginCtx) {
  if (!ginCtx) return ncclSuccess;
  struct ginProxyCtx *ctx = (struct ginProxyCtx *)ginCtx;

  // Free counters
  if (ctx) {
    if (ctx->counters || ctx->countersGdrHandle)
      freeMemCPUAccessible(ctx->counters, ctx->countersGdrHandle, ctx->comm->memManager);

    // Free signals
    if (ginComm && ctx->collComm && ctx->signalsMhandle)
      ginComm->deregMrSym(ctx->collComm, ctx->signalsMhandle);
    if (ctx->signalsDev) ncclCudaFree(ctx->signalsDev, ctx->comm->memManager);

    // Free hostGpuCtx and its allocations
    if (ctx->hostGpuCtx) {
      for (int contextId = 0; contextId < ctx->nContexts; contextId++) {
        struct ginProxyHostGpuCtx *hostGpuCtx = ctx->hostGpuCtx + contextId;
        if (hostGpuCtx->cisShadow) free(hostGpuCtx->cisShadow);
        if (hostGpuCtx->sis) free(hostGpuCtx->sis);
        if (hostGpuCtx->states) free(hostGpuCtx->states);
        if (hostGpuCtx->inlines) free(hostGpuCtx->inlines);
        if (ginComm && ctx->collComm && hostGpuCtx->inlinesMhandle)
          ginComm->deregMrSym(ctx->collComm, hostGpuCtx->inlinesMhandle);
        if (hostGpuCtx->queues) freeMemCPUAccessible(hostGpuCtx->queues, NULL, ctx->comm->memManager);
        if (hostGpuCtx->cis || hostGpuCtx->cisGdrHandle)
          freeMemCPUAccessible(hostGpuCtx->cis, hostGpuCtx->cisGdrHandle, ctx->comm->memManager);
      }
      free(ctx->hostGpuCtx);
    }

    ncclNetDeviceHandle_t *devHandle = (ncclNetDeviceHandle_t *)ctx->devHandle;
    if (devHandle) {
      if (devHandle->handle) ncclCudaFree((void *)devHandle->handle, ctx->comm->memManager);
      free(devHandle);
    }

    free(ctx);
  }

  return ncclSuccess;
}

ncclResult_t ncclGinProxyProgress(ncclGin_t *ginComm, void *ginCtx) {
  struct ginProxyCtx *ctx = (struct ginProxyCtx *)ginCtx;

  for (int contextId = 0; contextId < ctx->nContexts; contextId++) {
    struct ginProxyHostGpuCtx *hostGpuCtx = ctx->hostGpuCtx + contextId;
    NCCLCHECK(proxyGinPollCompletions(ginComm, ctx->collComm, ctx, hostGpuCtx));
    for (int targetRank = 0; targetRank < ctx->comm->nRanks; targetRank++) {
      // Poll on the GFD queue
      ncclGinProxyGfd_t gfd;
      struct ginProxyGfdState *state = NULL;
      if (proxyGinPollGfd(ctx, hostGpuCtx, targetRank, &gfd, &state)) {
        ncclResult_t ret =
          proxyGinProcessGfd(ginComm, ctx->collComm, ctx, hostGpuCtx, targetRank, &gfd, state);
        if (ret) ctx->hasError = ret;
        NCCLCHECK(ret);
      }
      if (ginComm->ginProgress) ginComm->ginProgress(ctx->collComm);
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclGinProxyQueryLastError(ncclGin_t *ginComm, void *ginCtx, bool *hasError) {
  struct ginProxyCtx *ctx = (struct ginProxyCtx *)ginCtx;
  *hasError = ctx->hasError;
  return ncclSuccess;
}
