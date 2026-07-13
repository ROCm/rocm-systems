/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

/**
 * GIN plugin: SDMA Anvil device path (NCCL_GIN_TYPE=5).
 * Small messages use inlined IPC flat stores via GIN-owned device-memory peer table in GPU context.
 * Large messages use standalone Anvil SDMA (gin_anvil_sdma_factory).
 */

#include "gin/gin_host_anvil_sdma.h"
#include "comm.h"
#include "dev_runtime.h"
#include "bootstrap.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_ipc_table.h"
#include <gin_anvil/sdma_factory.h>
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>
#include <map>

static std::map<void*, int> bufferRegRefcount;

struct ginAnvilInitCtx {
  struct ncclComm* comm;
};

struct ginAnvilCollCtx {
  int nranks;
  int rank;
  struct ncclComm* comm;
  gin_anvil_sdma_handle_t sdma;
  void** gpu_queue_handles;
  uint64_t* sdma_dirty_d;
  int numChannels;
  int sdmaChannelStride;
};

struct ginAnvilGinCtx {
  ncclNetDeviceHandle_v11_t* devHandle;
  ncclGinAnvilSdmaGPUContext* gpuCtxDev;
  ncclGinAnvilSdmaGPUContext gpuCtxHost;
  struct ncclComm* comm;
  int nRanks;
  int rank;
  int nSignals;
  int nCounters;
  int signalSlot;
  bool hasError;
  bool signalsBound;
  void** gpu_queue_handles;
  uint64_t* sdma_dirty_d;
  int numChannels;
  int sdmaChannelStride;
};

struct GinAnvilPendingEntry {
  ginAnvilGinCtx* ctx;
  GinAnvilPendingEntry* next;
};

static std::map<struct ncclComm*, GinAnvilPendingEntry*> g_pendingByComm;
static std::map<struct ncclComm*, int> g_nextSignalSlot;

static void ginAnvilPendingAdd(struct ncclComm* comm, ginAnvilGinCtx* ctx) {
  auto* e = new GinAnvilPendingEntry{ctx, g_pendingByComm[comm]};
  g_pendingByComm[comm] = e;
}

static void ginAnvilPendingRemove(struct ncclComm* comm, ginAnvilGinCtx* ctx) {
  GinAnvilPendingEntry** prev = &g_pendingByComm[comm];
  for (GinAnvilPendingEntry* e = g_pendingByComm[comm]; e != nullptr; e = e->next) {
    if (e->ctx == ctx) {
      *prev = e->next;
      delete e;
      return;
    }
    prev = &e->next;
  }
}

static void ginAnvilPendingClear(struct ncclComm* comm) {
  for (GinAnvilPendingEntry* e = g_pendingByComm[comm]; e != nullptr;) {
    GinAnvilPendingEntry* next = e->next;
    delete e;
    e = next;
  }
  g_pendingByComm.erase(comm);
  g_nextSignalSlot.erase(comm);
}

struct ginAnvilMemHandle {
  ncclGinAnvilSdmaMemHandle* devHandle;
  void* addr;
  void* lsaSelfAddr;
  size_t size;
};

struct ginAnvilListenCtx {
  int dev;
};

static int ginAnvilBootstrapAllgather(void* ctx, void* buf, size_t perRankSize) {
  return (bootstrapAllGather(ctx, buf, (int)perRankSize) == ncclSuccess) ? 0 : -1;
}

void ncclGinAnvilSetInitContext(void* initCtx, struct ncclComm* comm) {
  static_cast<ginAnvilInitCtx*>(initCtx)->comm = comm;
}

void ncclGinAnvilPluginTestResetHostState(void) {
  while (!g_pendingByComm.empty()) {
    ginAnvilPendingClear(g_pendingByComm.begin()->first);
  }
  g_nextSignalSlot.clear();
  bufferRegRefcount.clear();
}

static ncclResult_t ginAnvilInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  const char* gin_type = getenv("NCCL_GIN_TYPE");
  if (gin_type && atoi(gin_type) != NCCL_NET_DEVICE_GIN_ANVIL_SDMA) return ncclInternalError;
  if (gin_anvil_sdma_probe() <= 0) return ncclInternalError;
  auto* ictx = new ginAnvilInitCtx{};
  *ctx = ictx;
  return ncclSuccess;
}

static ncclResult_t ginAnvilDevices(int* ndev) {
  *ndev = 1;
  return ncclSuccess;
}

static ncclResult_t ginAnvilGetProperties(int dev, ncclNetProperties_v12_t* props) {
  memset(props, 0, sizeof(*props));
  props->name = const_cast<char*>("gin-anvil-sdma");
  props->pciPath = nullptr;
  props->guid = 0;
  props->ptrSupport = NCCL_PTR_CUDA;
  props->netDeviceType = NCCL_NET_DEVICE_GIN_ANVIL_SDMA;
  props->netDeviceVersion = NCCL_GIN_ANVIL_SDMA_NET_VERSION;
  props->maxP2pBytes = 1ULL << 30;
  props->maxCollBytes = 1ULL << 30;
  return ncclSuccess;
}

static ncclResult_t ginAnvilListen(void* ctx, int dev, void* handle, void** listenComm) {
  auto* lctx = new ginAnvilListenCtx;
  lctx->dev = dev;
  *listenComm = lctx;
  memset(handle, 0, NCCL_NET_HANDLE_MAXSIZE);
  return ncclSuccess;
}

static int ginAnvilEnvInt(const char* name, int defaultVal) {
  const char* e = getenv(name);
  if (e && e[0]) {
    int v = atoi(e);
    return v > 0 ? v : defaultVal;
  }
  return defaultVal;
}

static int ginAnvilSdmaThresholdFromEnv() {
  return ginAnvilEnvInt("NCCL_GIN_ANVIL_SDMA_THRESHOLD", (int)NCCL_GIN_ANVIL_SDMA_THRESHOLD_DEFAULT);
}

static int ginAnvilSdmaNumChannelsFromEnv() {
  int v = ginAnvilEnvInt("NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS", 1);
  return v >= 1 && v <= 8 ? v : 1;
}

static uint32_t ginAnvilFusedSignalFromEnv() {
  const char* e = getenv("NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL");
  if (!e || !e[0]) return NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL_DEFAULT;
  if (e[0] == '0' && e[1] == '\0') return 0;
  return atoi(e) != 0 ? 1u : 0u;
}

static ncclResult_t ginAnvilConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm,
                                    void** collComm) {
  auto* ictx = (ginAnvilInitCtx*)ctx;
  auto* cctx = new ginAnvilCollCtx{};
  cctx->nranks = nranks;
  cctx->rank = rank;
  cctx->comm = ictx->comm;

  int localDev = 0;
  if (hipGetDevice(&localDev) != hipSuccess) {
    delete cctx;
    return ncclSystemError;
  }

  int numCh = ginAnvilSdmaNumChannelsFromEnv();

  gin_anvil_sdma_handle_t h = nullptr;
  void* gpu_handles = nullptr;
  uint64_t* dirty = nullptr;
  if (gin_anvil_sdma_create(nranks, rank, localDev, ginAnvilBootstrapAllgather, cctx->comm->bootstrap, numCh,
                            &h, &gpu_handles, &dirty) != 0) {
    WARN("GIN anvil-sdma: gin_anvil_sdma_create failed");
    delete cctx;
    return ncclSystemError;
  }

  cctx->sdma = h;
  cctx->gpu_queue_handles = (void**)gpu_handles;
  cctx->sdma_dirty_d = dirty;
  cctx->numChannels = gin_anvil_sdma_get_num_channels(h);
  cctx->sdmaChannelStride = gin_anvil_sdma_get_channel_stride(h);

  INFO(NCCL_INIT, "GIN anvil-sdma: standalone SDMA queues (%d ranks, %d ch, spread=%d)", nranks,
       cctx->numChannels, cctx->sdmaChannelStride);
  *collComm = cctx;
  return ncclSuccess;
}

static ncclResult_t ginAnvilCloseListen(void* listenComm) {
  delete (ginAnvilListenCtx*)listenComm;
  return ncclSuccess;
}

static ncclResult_t ginAnvilCloseColl(void* collComm) {
  ginAnvilCollCtx* cctx = (ginAnvilCollCtx*)collComm;
  if (cctx) {
    if (cctx->sdma) gin_anvil_sdma_destroy(cctx->sdma);
    delete cctx;
  }
  return ncclSuccess;
}

static ncclResult_t ginAnvilFinalize(void* ctx) {
  delete (ginAnvilInitCtx*)ctx;
  return ncclSuccess;
}

static ncclResult_t ginAnvilRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mrFlags,
                                     void** mhandle, void** ginHandle) {
  ginAnvilCollCtx* cctx = (ginAnvilCollCtx*)collComm;
  struct ncclDevrState* devr = &cctx->comm->devrState;

  ginAnvilMemHandle* mh = nullptr;
  NCCLCHECK(ncclCalloc(&mh, 1));

  auto& refcount = bufferRegRefcount[data];

  void* lsaSelfAddr = nullptr;
  NCCLCHECK(ncclDevrGetLsaSelfAddr(devr, data, &lsaSelfAddr));
  if (lsaSelfAddr == nullptr) {
    WARN("GIN anvil-sdma: could not resolve LSA flat addr for %p", data);
    bufferRegRefcount.erase(data);
    free(mh);
    return ncclSystemError;
  }

  if (refcount == 0) {
    int rc = ncclGinAnvilIpcTableRegisterVmm(lsaSelfAddr, size, devr->lsaSelf, devr->lsaSize,
                                             (ptrdiff_t)devr->bigSize);
    if (rc != 0) {
      WARN("GIN anvil-sdma: IPC table register failed for %p (lsaSelf=%p) size %zu", data, lsaSelfAddr,
           size);
      bufferRegRefcount.erase(data);
      free(mh);
      return ncclSystemError;
    }
    INFO(NCCL_INIT, "GIN anvil-sdma: registered addr=%p lsaSelf=%p +%zu", data, lsaSelfAddr, size);
  }
  refcount++;

  mh->addr = data;
  mh->lsaSelfAddr = lsaSelfAddr;
  mh->size = size;

  if (hipMalloc(&mh->devHandle, sizeof(ncclGinAnvilSdmaMemHandle)) != hipSuccess) {
    free(mh);
    return ncclSystemError;
  }

  ncclGinAnvilSdmaMemHandle hostMh;
  hostMh.baseAddr = (uintptr_t)lsaSelfAddr;
  (void)hipMemcpy(mh->devHandle, &hostMh, sizeof(ncclGinAnvilSdmaMemHandle), hipMemcpyHostToDevice);

  *mhandle = mh;
  *ginHandle = mh->devHandle;
  return ncclSuccess;
}

static ncclResult_t ginAnvilRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset,
                                           int fd, uint64_t mrFlags, void** mhandle, void** ginHandle) {
  return ginAnvilRegMrSym(collComm, data, size, type, mrFlags, mhandle, ginHandle);
}

static ncclResult_t ginAnvilDeregMrSym(void* collComm, void* mhandle) {
  ginAnvilMemHandle* mh = (ginAnvilMemHandle*)mhandle;
  if (!mh) return ncclSuccess;

  if (mh->addr) {
    auto it = bufferRegRefcount.find(mh->addr);
    if (it != bufferRegRefcount.end()) {
      it->second--;
      if (it->second <= 0) {
        if (mh->lsaSelfAddr) (void)ncclGinAnvilIpcTableUnregister(mh->lsaSelfAddr);
        bufferRegRefcount.erase(it);
      }
    }
  }
  if (mh->devHandle) (void)hipFree(mh->devHandle);
  free(mh);
  return ncclSuccess;
}

static ncclResult_t ginAnvilRegisterLsaSignals(ginAnvilGinCtx* ctx, void* lsaSelf, size_t bytes) {
  struct ncclDevrState* devr = &ctx->comm->devrState;
  int rc = ncclGinAnvilIpcTableRegisterVmm(lsaSelf, bytes, devr->lsaSelf, devr->lsaSize,
                                           (ptrdiff_t)devr->bigSize);
  if (rc != 0) {
    WARN("GIN anvil-sdma: LSA signal IPC table register failed for %p size %zu", lsaSelf, bytes);
    return ncclSystemError;
  }
  ctx->gpuCtxHost.signals = (uint64_t*)lsaSelf;
  ctx->signalsBound = true;
  if (hipMemcpy(ctx->gpuCtxDev, &ctx->gpuCtxHost, sizeof(ncclGinAnvilSdmaGPUContext),
                hipMemcpyHostToDevice) != hipSuccess) {
    return ncclSystemError;
  }
  INFO(NCCL_INIT,
       "GIN anvil-sdma: bound LSA signals slot=%d lsaSelf=%p +%zu (rank %d, lsaSize=%d, bigSize=%zu)",
       ctx->signalSlot, lsaSelf, bytes, devr->lsaSelf, devr->lsaSize, (size_t)devr->bigSize);
  return ncclSuccess;
}

ncclResult_t ncclGinAnvilBindResourceWindowSignals(struct ncclComm* comm, void* resourceUserPtr,
                                                   size_t arenaByteOffset, int nContexts,
                                                   int nSignalsPerContext) {
  if (!comm || !resourceUserPtr || nContexts < 1 || nSignalsPerContext < 1) return ncclInvalidArgument;

  ncclResult_t ret = ncclSuccess;
  for (GinAnvilPendingEntry* e = g_pendingByComm[comm]; e != nullptr; e = e->next) {
    ginAnvilGinCtx* ctx = e->ctx;
    if (ctx->nSignals <= 0) continue;
    if (ctx->signalSlot < 0 || ctx->signalSlot >= nContexts) {
      WARN("GIN anvil-sdma: signal slot %d out of range (nContexts=%d)", ctx->signalSlot, nContexts);
      ginAnvilPendingClear(comm);
      return ncclInvalidArgument;
    }

    size_t off = arenaByteOffset + (size_t)ctx->signalSlot * (size_t)nSignalsPerContext * sizeof(uint64_t);
    void* localPtr = (char*)resourceUserPtr + off;
    void* lsaSelf = nullptr;
    NCCLCHECK(ncclDevrGetLsaSelfAddr(&comm->devrState, localPtr, &lsaSelf));
    if (lsaSelf == nullptr) {
      WARN("GIN anvil-sdma: could not resolve LSA flat addr for resource-window signals at %p", localPtr);
      ginAnvilPendingClear(comm);
      return ncclSystemError;
    }

    size_t bytes = (size_t)ctx->nSignals * sizeof(uint64_t);
    NCCLCHECKGOTO(ginAnvilRegisterLsaSignals(ctx, lsaSelf, bytes), ret, fail);
  }

fail:
  ginAnvilPendingClear(comm);
  return ret;
}

static ncclResult_t ginAnvilCreateContext(void* collComm, ncclGinConfig_v13_t* config, void** outGinCtx,
                                          ncclNetDeviceHandle_v11_t** outDevHandle) {
  ginAnvilCollCtx* cctx = (ginAnvilCollCtx*)collComm;
  ncclResult_t ret = ncclSuccess;
  auto* ctx = new ginAnvilGinCtx{};
  ctx->nRanks = cctx->nranks;
  ctx->rank = cctx->rank;
  ctx->nSignals = config->nSignals;
  ctx->nCounters = config->nCounters;
  ctx->comm = cctx->comm;
  ctx->signalSlot = g_nextSignalSlot[cctx->comm]++;
  ctx->hasError = false;
  ctx->signalsBound = false;
  ctx->gpu_queue_handles = cctx->gpu_queue_handles;
  ctx->sdma_dirty_d = cctx->sdma_dirty_d;
  ctx->numChannels = cctx->numChannels;
  ctx->sdmaChannelStride = cctx->sdmaChannelStride;

  if (!cctx->gpu_queue_handles || !cctx->sdma_dirty_d) {
    WARN("GIN anvil-sdma: missing SDMA infrastructure (handles=%p dirty=%p)", cctx->gpu_queue_handles,
         (void*)cctx->sdma_dirty_d);
    ret = ncclSystemError;
    goto fail;
  }

  NCCLCHECK(ncclCalloc(&ctx->devHandle, 1));
  ctx->devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_ANVIL_SDMA;
  ctx->devHandle->netDeviceVersion = NCCL_GIN_ANVIL_SDMA_NET_VERSION;
  ctx->devHandle->needsProxyProgress = 0;

  if (hipMalloc(&ctx->gpuCtxDev, sizeof(ncclGinAnvilSdmaGPUContext)) != hipSuccess) {
    ret = ncclSystemError;
    goto fail;
  }

  memset(&ctx->gpuCtxHost, 0, sizeof(ncclGinAnvilSdmaGPUContext));
  ctx->gpuCtxHost.layoutMagic = NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC;
  ctx->gpuCtxHost.nRanks = ctx->nRanks;
  ctx->gpuCtxHost.rank = ctx->rank;
  ctx->gpuCtxHost.nSignals = config->nSignals;
  ctx->gpuCtxHost.nCounters = config->nCounters;
  ctx->gpuCtxHost.numChannels = ctx->numChannels;
  ctx->gpuCtxHost.sdmaChannel = 0;
  ctx->gpuCtxHost.sdmaChannelStride = ctx->sdmaChannelStride;
  ctx->gpuCtxHost.queueHandles = ctx->gpu_queue_handles;
  ctx->gpuCtxHost.sdmaDirty = ctx->sdma_dirty_d;
  ctx->gpuCtxHost.sdmaThreshold = (uint32_t)ginAnvilSdmaThresholdFromEnv();
  ctx->gpuCtxHost.fusedSdmaSignal = ginAnvilFusedSignalFromEnv();
  ctx->gpuCtxHost.signals = nullptr;

  if (config->nCounters > 0) {
    if (hipExtMallocWithFlags((void**)&ctx->gpuCtxHost.counters, sizeof(uint64_t) * config->nCounters,
                              hipDeviceMallocFinegrained) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }
    if (hipMemset(ctx->gpuCtxHost.counters, 0, sizeof(uint64_t) * config->nCounters) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }
  }

  ncclGinAnvilIpcTableGetDevice(&ctx->gpuCtxHost.ipcTable, &ctx->gpuCtxHost.ipcTableCount);

  if (hipMemcpy(ctx->gpuCtxDev, &ctx->gpuCtxHost, sizeof(ncclGinAnvilSdmaGPUContext), hipMemcpyHostToDevice) !=
      hipSuccess) {
    WARN("GIN anvil-sdma: hipMemcpy gpu context failed");
    ret = ncclSystemError;
    goto fail;
  }

  ncclGinAnvilIpcTableTrackContext(&ctx->gpuCtxHost, ctx->gpuCtxDev);

  if (config->nSignals > 0) {
    ginAnvilPendingAdd(cctx->comm, ctx);
  }

  ctx->devHandle->handle = ctx->gpuCtxDev;
  ctx->devHandle->size = sizeof(ncclGinAnvilSdmaGPUContext);

  *outGinCtx = ctx;
  *outDevHandle = ctx->devHandle;
  INFO(NCCL_INIT,
       "GIN anvil-sdma: context created (v%d, %d signals, %d counters, signalSlot=%d, sdmaThreshold=%u, "
       "spread=%d, fusedSignal=%u)",
       NCCL_GIN_ANVIL_SDMA_NET_VERSION, config->nSignals, config->nCounters, ctx->signalSlot,
       ctx->gpuCtxHost.sdmaThreshold, ctx->gpuCtxHost.sdmaChannelStride, ctx->gpuCtxHost.fusedSdmaSignal);
  return ncclSuccess;

fail:
  if (ctx) {
    if (ctx->comm) ginAnvilPendingRemove(ctx->comm, ctx);
    if (ctx->signalsBound && ctx->gpuCtxHost.signals) {
      (void)ncclGinAnvilIpcTableUnregister(ctx->gpuCtxHost.signals);
    }
    if (ctx->gpuCtxHost.counters) (void)hipFree(ctx->gpuCtxHost.counters);
    if (ctx->gpuCtxDev) (void)hipFree(ctx->gpuCtxDev);
    free(ctx->devHandle);
    delete ctx;
  }
  return ret;
}

static ncclResult_t ginAnvilDestroyContext(void* ginCtx) {
  ginAnvilGinCtx* ctx = (ginAnvilGinCtx*)ginCtx;
  if (!ctx) return ncclSuccess;
  if (ctx->comm) ginAnvilPendingRemove(ctx->comm, ctx);
  ncclGinAnvilIpcTableUntrackContext(&ctx->gpuCtxHost);
  if (ctx->signalsBound && ctx->gpuCtxHost.signals) {
    (void)ncclGinAnvilIpcTableUnregister(ctx->gpuCtxHost.signals);
  }
  if (ctx->gpuCtxHost.counters) (void)hipFree(ctx->gpuCtxHost.counters);
  if (ctx->gpuCtxDev) (void)hipFree(ctx->gpuCtxDev);
  free(ctx->devHandle);
  delete ctx;
  return ncclSuccess;
}

static ncclResult_t ginAnvilGinProgress(void* ginCtx) { return ncclSuccess; }

static ncclResult_t ginAnvilQueryLastError(void* ginCtx, bool* hasError) {
  *hasError = false;
  return ncclSuccess;
}

__attribute__((visibility("default"))) ncclGin_t ncclGinAnvilSdmaPlugin = {
    .name = "gin-anvil-sdma",
    .init = ginAnvilInit,
    .devices = ginAnvilDevices,
    .getProperties = ginAnvilGetProperties,
    .listen = ginAnvilListen,
    .connect = ginAnvilConnect,
    .createContext = ginAnvilCreateContext,
    .regMrSym = ginAnvilRegMrSym,
    .regMrSymDmaBuf = ginAnvilRegMrSymDmaBuf,
    .deregMrSym = ginAnvilDeregMrSym,
    .destroyContext = ginAnvilDestroyContext,
    .closeColl = ginAnvilCloseColl,
    .closeListen = ginAnvilCloseListen,
    .iput = NULL,
    .iputSignal = NULL,
    .iget = NULL,
    .iflush = NULL,
    .test = NULL,
    .ginProgress = ginAnvilGinProgress,
    .queryLastError = ginAnvilQueryLastError,
    .finalize = ginAnvilFinalize,
};

#endif  // ENABLE_ROCSHMEM_GIN
