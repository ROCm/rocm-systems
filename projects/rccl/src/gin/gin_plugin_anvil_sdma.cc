/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

/**
 * GIN plugin: SDMA Anvil device path (NCCL_GIN_TYPE=6).
 * Small messages use inlined IPC flat stores via GIN-owned device-memory peer table in GPU context.
 * Large messages use standalone Anvil SDMA (gin_anvil_sdma_factory).
 */

#include "gin/gin_host_anvil_sdma.h"
#include "algorithms/dda/fabric/fabric_init.h"
#include "algorithms/dda/fabric/fabric_mem_handler.h"
#include "alloc.h"
#include "comm.h"
#include "dev_runtime.h"
#include "bootstrap.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_ipc_table.h"
#include <gin_anvil/sdma_factory.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <new>

static std::map<void*, int> bufferRegRefcount;
static std::map<void*, ncclFabricMemHandler*> fabricBufferHandlers;
static std::map<void*, CUmemGenericAllocationHandle> fabricBufferHandles;
static std::map<void*, int> fabricBufferRefcount;
static std::mutex pluginMutex;

static void ginAnvilCuMemRelease(CUmemGenericAllocationHandle handle) {
  if (rcclSkipCuMemFree()) return;
  (void)cuMemRelease(handle);
}

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
  uintptr_t* signal_remote_addrs_dev;
};

struct GinAnvilPendingEntry {
  ginAnvilGinCtx* ctx;
  GinAnvilPendingEntry* next;
};

static std::map<struct ncclComm*, GinAnvilPendingEntry*> g_pendingByComm;

static void ginAnvilPendingAdd(struct ncclComm* comm, ginAnvilGinCtx* ctx) {
  std::lock_guard<std::mutex> lock(pluginMutex);
  auto* e = new GinAnvilPendingEntry{ctx, g_pendingByComm[comm]};
  g_pendingByComm[comm] = e;
}

static void ginAnvilPendingRemove(struct ncclComm* comm, ginAnvilGinCtx* ctx) {
  std::lock_guard<std::mutex> lock(pluginMutex);
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
  std::lock_guard<std::mutex> lock(pluginMutex);
  for (GinAnvilPendingEntry* e = g_pendingByComm[comm]; e != nullptr;) {
    GinAnvilPendingEntry* next = e->next;
    delete e;
    e = next;
  }
  g_pendingByComm.erase(comm);
}

struct ginAnvilMemHandle {
  ncclGinAnvilSdmaMemHandle* devHandle;
  void* addr;
  void* lsaSelfAddr;
  size_t size;
  uintptr_t* remote_vas_dev;
  bool fabricMem;
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
  std::lock_guard<std::mutex> lock(pluginMutex);
  while (!g_pendingByComm.empty()) {
    struct ncclComm* comm = g_pendingByComm.begin()->first;
    for (GinAnvilPendingEntry* e = g_pendingByComm[comm]; e != nullptr;) {
      GinAnvilPendingEntry* next = e->next;
      delete e;
      e = next;
    }
    g_pendingByComm.erase(comm);
  }
  bufferRegRefcount.clear();
  for (auto& entry : fabricBufferHandlers) {
    delete entry.second;
  }
  fabricBufferHandlers.clear();
  for (auto& entry : fabricBufferHandles) {
    ginAnvilCuMemRelease(entry.second);
  }
  fabricBufferHandles.clear();
  fabricBufferRefcount.clear();
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

// v14 GIN plugins expose GIN capability flags via getGinProperties. The Anvil
// SDMA backend uses intra-node LSA (flat) signals, which behave as both strong
// and VA-addressable signals; report both as supported (matches the behavior
// previously injected by the v13->v14 shim for internal plugins).
static ncclResult_t ginAnvilGetGinProperties(ncclGinProperties_t* ginProps) {
  ginProps->supportsStrongSignals = true;
  ginProps->supportsVASignals = true;
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

// Backend gin.put inline-vs-copy-engine threshold. Parsed as 64-bit so a value
// >= 2 GiB does not wrap through int (atoi) and silently fall back to the
// compiled default. Explicit 0 is honored. The GPU context field is uint32_t, so
// values above UINT32_MAX saturate rather than wrapping.
static size_t ginAnvilSdmaThresholdFromEnv() {
  const char* e = getenv("NCCL_GIN_ANVIL_SDMA_THRESHOLD");
  if (e && e[0] && *e != '-') {
    char* end = nullptr;
    unsigned long long v = strtoull(e, &end, 10);
    if (end != e && *end == '\0') return (size_t)v;
  }
  return (size_t)NCCL_GIN_ANVIL_SDMA_THRESHOLD_DEFAULT;
}

static int ginAnvilSdmaNumChannels() {
  // Forced to a single SDMA channel. Multi-channel (>=4 channels) at
  // >=256 MiB/peer aborts with a fail-loud "unhandled system error" on
  // 8x MI355X, and the collective GIN-put paths issue their puts from a single
  // warp anyway (channel 0), so additional channels provide no benefit.
  return 1;
}

static uint32_t ginAnvilFusedSignalFromEnv() {
  const char* e = getenv("NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL");
  if (!e || !e[0]) return NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL_DEFAULT;
  if (e[0] == '0' && e[1] == '\0') return 0;
  return atoi(e) != 0 ? 1u : 0u;
}

static uint32_t ginAnvilIpcAgentFenceFromEnv() {
  const char* e = getenv("NCCL_GIN_ANVIL_SDMA_IPC_AGENT_FENCE");
  if (!e || !e[0]) return 0;
  if (e[0] == '0' && e[1] == '\0') return 0;
  return atoi(e) != 0 ? 1u : 0u;
}

static uint32_t ginAnvilIpcSignalPeerFromEnv() {
  const char* e = getenv("NCCL_GIN_ANVIL_SDMA_SIGNAL_IPC");
  if (!e || !e[0]) return 0;
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

  int numCh = ginAnvilSdmaNumChannels();

  gin_anvil_sdma_handle_t h = nullptr;
  void* gpu_handles = nullptr;
  uint64_t* dirty = nullptr;
  if (gin_anvil_sdma_create(nranks, rank, localDev, ginAnvilBootstrapAllgather, cctx->comm->bootstrap, numCh, &h,
                            &gpu_handles, &dirty) != 0) {
    WARN("GIN anvil-sdma: gin_anvil_sdma_create failed");
    delete cctx;
    return ncclSystemError;
  }

  cctx->sdma = h;
  cctx->gpu_queue_handles = (void**)gpu_handles;
  cctx->sdma_dirty_d = dirty;
  cctx->numChannels = gin_anvil_sdma_get_num_channels(h);
  cctx->sdmaChannelStride = gin_anvil_sdma_get_channel_stride(h);

  INFO(NCCL_INIT, "GIN anvil-sdma: standalone SDMA queues (%d ranks, %d ch, spread=%d)", nranks, cctx->numChannels,
       cctx->sdmaChannelStride);
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

static ncclResult_t ginAnvilRegMrSymFabric(ginAnvilCollCtx* cctx, void* data, size_t size, ginAnvilMemHandle* mh,
                                           void** mhandle, void** ginHandle) {
  struct ncclComm* comm = cctx->comm;
  ncclFabricMemHandler* handler = nullptr;
  CUmemGenericAllocationHandle memHandle{};
  CUdeviceptr memAddr = 0;
  size_t memSize = 0;
  int numSegments = 0;
  ncclResult_t ret = ncclSuccess;
  uintptr_t* remote_vas_host = nullptr;
  bool localRetainedHandle = false;

  NCCLCHECK(ncclCuMemGetAddressRange(reinterpret_cast<CUdeviceptr>(data), size, &memAddr, &memSize, &numSegments));
  if (numSegments != 1) {
    WARN("GIN anvil-sdma fabric: multi-segment MR not supported yet (segments=%d data=%p size=%zu)", numSegments, data,
         size);
    return ncclSystemError;
  }
  if (size > memSize) {
    WARN("GIN anvil-sdma fabric: registration size %zu exceeds mapped segment size %zu for %p", size, memSize, data);
    return ncclSystemError;
  }

  const uintptr_t offset = reinterpret_cast<uintptr_t>(data) - reinterpret_cast<uintptr_t>(memAddr);

  {
    std::lock_guard<std::mutex> lock(pluginMutex);
    auto& refcount = fabricBufferRefcount[data];
    if (refcount == 0) {
      handler = new (std::nothrow) ncclFabricMemHandler(comm->bootstrap, comm->rank, cctx->nranks, comm->memManager);
      if (handler == nullptr) {
        return ncclSystemError;
      }
      CUresult cuRes = cuMemRetainAllocationHandle(&memHandle, reinterpret_cast<void*>(memAddr));
      if (cuRes != CUDA_SUCCESS) {
        delete handler;
        return ncclSystemError;
      }
      localRetainedHandle = true;
      ret = handler->addSelfDeviceMem(reinterpret_cast<void*>(memAddr), memHandle, memSize);
      if (ret != ncclSuccess) {
        delete handler;
        (void)cuMemRelease(memHandle);
        return ret;
      }
      ret = handler->exchangeMemPtrs();
      if (ret != ncclSuccess) {
        delete handler;
        (void)cuMemRelease(memHandle);
        return ret;
      }
      fabricBufferHandlers[data] = handler;
      fabricBufferHandles[data] = memHandle;
      localRetainedHandle = false;
      INFO(NCCL_INIT, "GIN anvil-sdma fabric: exported addr=%p memBase=%p size=%zu offset=%zu", data,
           reinterpret_cast<void*>(memAddr), memSize, static_cast<size_t>(offset));
    } else {
      handler = fabricBufferHandlers[data];
    }
    refcount++;
  }

  mh->addr = data;
  mh->lsaSelfAddr = data;
  mh->size = size;
  mh->fabricMem = true;
  mh->remote_vas_dev = nullptr;

  if (hipMalloc(&mh->devHandle, sizeof(ncclGinAnvilSdmaMemHandle)) != hipSuccess) {
    ret = ncclSystemError;
    goto failFabricRef;
  }

  remote_vas_host = (uintptr_t*)malloc(sizeof(uintptr_t) * (size_t)cctx->nranks);
  if (!remote_vas_host) {
    ret = ncclSystemError;
    goto failFabricDevHandle;
  }
  for (int pe = 0; pe < cctx->nranks; pe++) {
    void* peerPtr = nullptr;
    NCCLCHECKGOTO(handler->getPeerDeviceMemPtr(pe, &peerPtr), ret, failFabricRemote);
    remote_vas_host[pe] = reinterpret_cast<uintptr_t>(peerPtr) + offset;
  }
  if (hipMalloc(&mh->remote_vas_dev, sizeof(uintptr_t) * (size_t)cctx->nranks) != hipSuccess ||
      hipMemcpy(mh->remote_vas_dev, remote_vas_host, sizeof(uintptr_t) * (size_t)cctx->nranks, hipMemcpyHostToDevice) !=
        hipSuccess) {
    ret = ncclSystemError;
    goto failFabricRemote;
  }
  free(remote_vas_host);
  remote_vas_host = nullptr;

  {
    ncclGinAnvilSdmaMemHandle hostMh;
    hostMh.baseAddr = reinterpret_cast<uintptr_t>(data);
    hostMh.remote_vas = mh->remote_vas_dev;
    hostMh.vmmStride = 0;
    (void)hipMemcpy(mh->devHandle, &hostMh, sizeof(ncclGinAnvilSdmaMemHandle), hipMemcpyHostToDevice);
  }

  *mhandle = mh;
  *ginHandle = mh->devHandle;
  return ncclSuccess;

failFabricRemote:
  if (remote_vas_host) free(remote_vas_host);
failFabricDevHandle:
  if (mh->devHandle) CUDACHECKIGNORE(hipFree(mh->devHandle));
failFabricRef:
  {
    std::lock_guard<std::mutex> lock(pluginMutex);
    auto it = fabricBufferRefcount.find(data);
    if (it != fabricBufferRefcount.end()) {
      it->second--;
      if (it->second <= 0) {
        auto hit = fabricBufferHandlers.find(data);
        if (hit != fabricBufferHandlers.end()) {
          delete hit->second;
          fabricBufferHandlers.erase(hit);
        }
        auto handleIt = fabricBufferHandles.find(data);
        if (handleIt != fabricBufferHandles.end()) {
          ginAnvilCuMemRelease(handleIt->second);
          fabricBufferHandles.erase(handleIt);
        }
        fabricBufferRefcount.erase(it);
      }
    }
  }
  if (localRetainedHandle) {
    ginAnvilCuMemRelease(memHandle);
  }
  return ret;
}

static ncclResult_t ginAnvilRegMrSymLsa(ginAnvilCollCtx* cctx, void* data, size_t size, ginAnvilMemHandle* mh,
                                        void** mhandle, void** ginHandle) {
  struct ncclDevrState* devr = &cctx->comm->devrState;

  void* lsaSelfAddr = nullptr;
  NCCLCHECK(ncclDevrGetLsaSelfAddr(devr, data, &lsaSelfAddr));
  if (lsaSelfAddr == nullptr) {
    WARN("GIN anvil-sdma: could not resolve LSA flat addr for %p", data);
    return ncclSystemError;
  }

  {
    std::lock_guard<std::mutex> lock(pluginMutex);
    auto& refcount = bufferRegRefcount[data];
    if (refcount == 0) {
      int rc =
        ncclGinAnvilIpcTableRegisterVmm(lsaSelfAddr, size, devr->lsaSelf, devr->lsaSize, (ptrdiff_t)devr->bigSize);
      if (rc != 0) {
        WARN("GIN anvil-sdma: IPC table register failed for %p (lsaSelf=%p) size %zu", data, lsaSelfAddr, size);
        bufferRegRefcount.erase(data);
        return ncclSystemError;
      }
      INFO(NCCL_INIT, "GIN anvil-sdma: registered addr=%p lsaSelf=%p +%zu", data, lsaSelfAddr, size);
    }
    refcount++;
  }

  mh->addr = data;
  mh->lsaSelfAddr = lsaSelfAddr;
  mh->size = size;
  mh->fabricMem = false;
  mh->remote_vas_dev = nullptr;

  if (hipMalloc(&mh->devHandle, sizeof(ncclGinAnvilSdmaMemHandle)) != hipSuccess) {
    return ncclSystemError;
  }

  const ptrdiff_t stride = (ptrdiff_t)devr->bigSize;
  uintptr_t* remote_vas_host = (uintptr_t*)malloc(sizeof(uintptr_t) * (size_t)cctx->nranks);
  if (!remote_vas_host) {
    CUDACHECKIGNORE(hipFree(mh->devHandle));
    return ncclSystemError;
  }
  for (int pe = 0; pe < cctx->nranks; pe++) {
    remote_vas_host[pe] = (uintptr_t)lsaSelfAddr + static_cast<ptrdiff_t>(pe - devr->lsaSelf) * stride;
  }
  if (hipMalloc(&mh->remote_vas_dev, sizeof(uintptr_t) * (size_t)cctx->nranks) != hipSuccess ||
      hipMemcpy(mh->remote_vas_dev, remote_vas_host, sizeof(uintptr_t) * (size_t)cctx->nranks, hipMemcpyHostToDevice) !=
        hipSuccess) {
    free(remote_vas_host);
    CUDACHECKIGNORE(hipFree(mh->devHandle));
    return ncclSystemError;
  }
  free(remote_vas_host);

  ncclGinAnvilSdmaMemHandle hostMh;
  hostMh.baseAddr = (uintptr_t)lsaSelfAddr;
  hostMh.remote_vas = mh->remote_vas_dev;
  hostMh.vmmStride = stride;
  (void)hipMemcpy(mh->devHandle, &hostMh, sizeof(ncclGinAnvilSdmaMemHandle), hipMemcpyHostToDevice);

  *mhandle = mh;
  *ginHandle = mh->devHandle;
  return ncclSuccess;
}

static ncclResult_t ginAnvilRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mrFlags,
                                     void** mhandle, void** ginHandle) {
  ginAnvilCollCtx* cctx = (ginAnvilCollCtx*)collComm;
  ginAnvilMemHandle* mh = nullptr;
  NCCLCHECK(ncclCalloc(&mh, 1));

  ncclResult_t ret = ncclSuccess;
  if (ginAnvilUseFabricMem(cctx->comm)) {
    ret = ginAnvilRegMrSymFabric(cctx, data, size, mh, mhandle, ginHandle);
  } else {
    ret = ginAnvilRegMrSymLsa(cctx, data, size, mh, mhandle, ginHandle);
  }
  if (ret != ncclSuccess) {
    free(mh);
  }
  return ret;
}

static ncclResult_t ginAnvilRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                           uint64_t mrFlags, void** mhandle, void** ginHandle) {
  return ginAnvilRegMrSym(collComm, data, size, type, mrFlags, mhandle, ginHandle);
}

static ncclResult_t ginAnvilDeregMrSym(void* collComm, void* mhandle) {
  ginAnvilMemHandle* mh = (ginAnvilMemHandle*)mhandle;
  if (!mh) return ncclSuccess;

  if (mh->addr) {
    std::lock_guard<std::mutex> lock(pluginMutex);
    if (mh->fabricMem) {
      auto it = fabricBufferRefcount.find(mh->addr);
      if (it != fabricBufferRefcount.end()) {
        it->second--;
        if (it->second <= 0) {
          auto hit = fabricBufferHandlers.find(mh->addr);
          if (hit != fabricBufferHandlers.end()) {
            delete hit->second;
            fabricBufferHandlers.erase(hit);
          }
          auto handleIt = fabricBufferHandles.find(mh->addr);
          if (handleIt != fabricBufferHandles.end()) {
            ginAnvilCuMemRelease(handleIt->second);
            fabricBufferHandles.erase(handleIt);
          }
          fabricBufferRefcount.erase(it);
        }
      }
    } else {
      auto it = bufferRegRefcount.find(mh->addr);
      if (it != bufferRegRefcount.end()) {
        it->second--;
        if (it->second <= 0) {
          if (mh->lsaSelfAddr) (void)ncclGinAnvilIpcTableUnregister(mh->lsaSelfAddr);
          bufferRegRefcount.erase(it);
        }
      }
    }
  }
  if (mh->devHandle) CUDACHECKIGNORE(hipFree(mh->devHandle));
  if (mh->remote_vas_dev) CUDACHECKIGNORE(hipFree(mh->remote_vas_dev));
  free(mh);
  return ncclSuccess;
}

static bool ginAnvilSignalDebugEnabled() {
  const char* v = getenv("RCCL_GIN_ANVIL_SDMA_SIGNAL_DEBUG");
  return v && atoi(v) != 0;
}

static ncclResult_t ginAnvilRegisterLsaSignals(ginAnvilGinCtx* ctx, void* lsaSelf, size_t bytes) {
  struct ncclDevrState* devr = &ctx->comm->devrState;
  struct ncclComm* comm = ctx->comm;
  const ptrdiff_t stride = (ptrdiff_t)devr->bigSize;

  if (ctx->nRanks != devr->lsaSize) {
    WARN("GIN anvil-sdma: signal nRanks=%d != devr->lsaSize=%d (rank %d)", ctx->nRanks, devr->lsaSize, ctx->rank);
  }
  if (ctx->rank != devr->lsaSelf) {
    WARN("GIN anvil-sdma: ctx->rank=%d != devr->lsaSelf=%d (signal peer indexing may be wrong)", ctx->rank,
         devr->lsaSelf);
  }

  ctx->gpuCtxHost.signals = (uint64_t*)lsaSelf;

  uintptr_t* hostAddrs = (uintptr_t*)calloc((size_t)ctx->nRanks, sizeof(uintptr_t));
  if (!hostAddrs) return ncclSystemError;
  for (int pe = 0; pe < ctx->nRanks; pe++) {
    hostAddrs[pe] = (uintptr_t)lsaSelf + static_cast<ptrdiff_t>(pe - ctx->rank) * stride;
  }

  uintptr_t selfExpected = (uintptr_t)lsaSelf;
  if (hostAddrs[ctx->rank] != selfExpected) {
    WARN("GIN anvil-sdma: signal_remote_addrs[self=%d]=%#lx != signals=%#lx (lsaSelf=%#lx stride=%zd)", ctx->rank,
         (unsigned long)hostAddrs[ctx->rank], (unsigned long)selfExpected, (unsigned long)lsaSelf, (long)stride);
  }

  uintptr_t* gatheredLocalBases = (uintptr_t*)calloc((size_t)ctx->nRanks, sizeof(uintptr_t));
  if (gatheredLocalBases) {
    gatheredLocalBases[ctx->rank] = (uintptr_t)lsaSelf;
    if (ginAnvilBootstrapAllgather(comm->bootstrap, gatheredLocalBases, sizeof(uintptr_t)) != 0) {
      WARN("GIN anvil-sdma: signal local-base allgather failed");
    } else if (gatheredLocalBases[ctx->rank] != (uintptr_t)lsaSelf) {
      WARN("GIN anvil-sdma: signal local-base allgather mismatch rank=%d got=%#lx expect=%#lx", ctx->rank,
           (unsigned long)gatheredLocalBases[ctx->rank], (unsigned long)lsaSelf);
    }
    free(gatheredLocalBases);
  }

  int rc = ncclGinAnvilIpcTableRegisterExplicit(lsaSelf, hostAddrs, ctx->nRanks, bytes);
  if (rc != 0) {
    WARN("GIN anvil-sdma: LSA signal IPC table register failed for %p size %zu", lsaSelf, bytes);
    free(hostAddrs);
    return ncclSystemError;
  }

  if (ctx->signal_remote_addrs_dev) CUDACHECKIGNORE(hipFree(ctx->signal_remote_addrs_dev));
  if (hipMalloc(&ctx->signal_remote_addrs_dev, sizeof(uintptr_t) * (size_t)ctx->nRanks) != hipSuccess ||
      hipMemcpy(ctx->signal_remote_addrs_dev, hostAddrs, sizeof(uintptr_t) * (size_t)ctx->nRanks,
                hipMemcpyHostToDevice) != hipSuccess) {
    free(hostAddrs);
    return ncclSystemError;
  }
  ctx->gpuCtxHost.signal_remote_addrs = ctx->signal_remote_addrs_dev;

  if (ginAnvilSignalDebugEnabled()) {
    for (int pe = 0; pe < ctx->nRanks; pe++) {
      INFO(NCCL_INIT, "GIN anvil-sdma: signal_remote_addrs rank=%d peer=%d -> %#lx (local signals %#lx)", ctx->rank, pe,
           (unsigned long)hostAddrs[pe], (unsigned long)lsaSelf);
    }
  }

  uintptr_t remote0 = hostAddrs[0];
  uintptr_t remoteSelf = hostAddrs[ctx->rank];
  free(hostAddrs);
  ctx->signalsBound = true;
  if (hipMemcpy(ctx->gpuCtxDev, &ctx->gpuCtxHost, sizeof(ncclGinAnvilSdmaGPUContext), hipMemcpyHostToDevice) !=
      hipSuccess) {
    return ncclSystemError;
  }
  INFO(NCCL_INIT,
       "GIN anvil-sdma: bound LSA signals slot=%d signals=%p bytes=%zu rank=%d lsaSelf=%d lsaSize=%d "
       "stride=%zu remote[0]=%#lx remote[self]=%#lx",
       ctx->signalSlot, lsaSelf, bytes, ctx->rank, devr->lsaSelf, devr->lsaSize, (size_t)devr->bigSize,
       (unsigned long)remote0, (unsigned long)remoteSelf);

  return ncclSuccess;
}

ncclResult_t ncclGinAnvilBindResourceWindowSignals(struct ncclComm* comm, void* resourceUserPtr, size_t arenaByteOffset,
                                                   int nContexts, int nSignalsPerContext) {
  if (!comm || !resourceUserPtr || nContexts < 1 || nSignalsPerContext < 1) return ncclInvalidArgument;

  ncclResult_t ret = ncclSuccess;
  int slot = 0;
  for (GinAnvilPendingEntry* e = g_pendingByComm[comm]; e != nullptr; e = e->next) {
    ginAnvilGinCtx* ctx = e->ctx;
    if (ctx->nSignals <= 0) continue;
    ctx->signalSlot = slot++;
    if (ctx->signalSlot >= nContexts) {
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

static ncclResult_t ginAnvilCreateContext(void* collComm, ncclGinConfig_t* config, void** outGinCtx,
                                          ncclNetDeviceHandle_v11_t** outDevHandle) {
  ginAnvilCollCtx* cctx = (ginAnvilCollCtx*)collComm;
  ncclResult_t ret = ncclSuccess;
  auto* ctx = new ginAnvilGinCtx{};
  ctx->nRanks = cctx->nranks;
  ctx->rank = cctx->rank;
  ctx->nSignals = config->nSignals;
  ctx->nCounters = config->nCounters;
  ctx->comm = cctx->comm;
  ctx->signalSlot = -1;  // assigned during ncclGinAnvilBindResourceWindowSignals
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
  {
    const size_t thr = ginAnvilSdmaThresholdFromEnv();
    ctx->gpuCtxHost.sdmaThreshold =
        (thr > (size_t)std::numeric_limits<uint32_t>::max()) ? std::numeric_limits<uint32_t>::max()
                                                             : (uint32_t)thr;
  }
  ctx->gpuCtxHost.fusedSdmaSignal = ginAnvilFusedSignalFromEnv();
  ctx->gpuCtxHost.ipcAgentFence = ginAnvilIpcAgentFenceFromEnv();
  ctx->gpuCtxHost.ipcSignalPeer = ginAnvilIpcSignalPeerFromEnv();
  ctx->gpuCtxHost.signals = nullptr;
  ctx->gpuCtxHost.signal_remote_addrs = nullptr;
  ctx->signal_remote_addrs_dev = nullptr;

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
    if (ctx->signal_remote_addrs_dev) CUDACHECKIGNORE(hipFree(ctx->signal_remote_addrs_dev));
    if (ctx->gpuCtxHost.counters) CUDACHECKIGNORE(hipFree(ctx->gpuCtxHost.counters));
    if (ctx->gpuCtxDev) CUDACHECKIGNORE(hipFree(ctx->gpuCtxDev));
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
  if (ctx->signal_remote_addrs_dev) CUDACHECKIGNORE(hipFree(ctx->signal_remote_addrs_dev));
  if (ctx->gpuCtxHost.counters) CUDACHECKIGNORE(hipFree(ctx->gpuCtxHost.counters));
  if (ctx->gpuCtxDev) CUDACHECKIGNORE(hipFree(ctx->gpuCtxDev));
  free(ctx->devHandle);
  delete ctx;
  return ncclSuccess;
}

static ncclResult_t ginAnvilGinProgress(void* ginCtx) {
  return ncclSuccess;
}

static ncclResult_t ginAnvilQueryLastError(void* ginCtx, bool* hasError) {
  *hasError = false;
  return ncclSuccess;
}

__attribute__((visibility("default"))) ncclGin_t ncclGinAnvilSdmaPlugin = {
  .name = "gin-anvil-sdma",
  .init = ginAnvilInit,
  .devices = ginAnvilDevices,
  .getGinProperties = ginAnvilGetGinProperties,
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
  .ginProgress = ginAnvilGinProgress,
  .queryLastError = ginAnvilQueryLastError,
  .finalize = ginAnvilFinalize,
};

#endif  // ENABLE_ROCSHMEM_GIN
