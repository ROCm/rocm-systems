/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

/**
 * GIN plugin: SDMA Anvil device path (NCCL_GIN_TYPE=7).
 * Small messages use inlined IPC flat stores via GIN-owned device-memory peer table in GPU context.
 * Large messages use standalone Anvil SDMA (gin_anvil_sdma_factory).
 */

#include "gin/gin_host_anvil_sdma.h"
#include "gin/gin_bootstrap_agree.h"
#include "algorithms/dda/fabric/fabric_init.h"
#include "algorithms/dda/fabric/fabric_mem_handler.h"
#include "alloc.h"
#include "gin/gin_anvil_conn_check.h"
#include "comm.h"
#include "dev_runtime.h"
#include "bootstrap.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_ipc_table.h"
#include "nccl_device/gin/anvil_sdma/gin_fabric_lsa_policy.h"
#include "gin/gin_anvil_sdma_factory.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <set>
#include <unordered_map>
#include <vector>

#if defined(GIN_ANVIL_PLUGIN_UNIT_TEST)
namespace GinAnvilPluginTestHooks {
ncclResult_t queryVmmRange(void* data, size_t size, CUdeviceptr* base, size_t* memSize, int* numSegments);
CUresult retainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr);
}  // namespace GinAnvilPluginTestHooks
#endif

static ncclResult_t ginAnvilQueryFabricVmmRange(void* data, size_t size, CUdeviceptr* base, size_t* memSize,
                                                int* numSegments) {
#if defined(GIN_ANVIL_PLUGIN_UNIT_TEST)
  return GinAnvilPluginTestHooks::queryVmmRange(data, size, base, memSize, numSegments);
#else
  return ncclCuMemGetAddressRange(reinterpret_cast<CUdeviceptr>(data), size, base, memSize, numSegments);
#endif
}

static ncclResult_t ginAnvilRetainFabricHandle(CUmemGenericAllocationHandle* handle, void* addr) {
#if defined(GIN_ANVIL_PLUGIN_UNIT_TEST)
  return GinAnvilPluginTestHooks::retainAllocationHandle(handle, addr) == CUDA_SUCCESS ? ncclSuccess
                                                                                       : ncclSystemError;
#else
  CUresult cuRes = cuMemRetainAllocationHandle(handle, addr);
  return cuRes == CUDA_SUCCESS ? ncclSuccess : ncclSystemError;
#endif
}

struct GinAnvilFabricKey {
  void* collComm;
  void* data;
  bool operator<(GinAnvilFabricKey const& o) const {
    if (collComm != o.collComm) return collComm < o.collComm;
    return data < o.data;
  }
};

static std::map<void*, int> bufferRegRefcount;
static std::map<GinAnvilFabricKey, ncclFabricMemHandler*> fabricBufferHandlers;
static std::map<GinAnvilFabricKey, CUmemGenericAllocationHandle> fabricBufferHandles;
static std::map<GinAnvilFabricKey, int> fabricBufferRefcount;
static std::unordered_map<uintptr_t, int> g_signalSpanRefcount;
static std::mutex pluginMutex;

static void ginAnvilCuMemRelease(CUmemGenericAllocationHandle handle) {
  // Owner cuMemRelease is never skipped (alloc.h skip-free is peer-unmap only).
  (void)cuMemRelease(handle);
}

static void ginAnvilFabricRefDropLocked(GinAnvilFabricKey const& key) {
  auto it = fabricBufferRefcount.find(key);
  if (it == fabricBufferRefcount.end()) return;
  it->second--;
  if (it->second > 0) return;
  auto hit = fabricBufferHandlers.find(key);
  if (hit != fabricBufferHandlers.end()) {
    delete hit->second;
    fabricBufferHandlers.erase(hit);
  }
  auto handleIt = fabricBufferHandles.find(key);
  if (handleIt != fabricBufferHandles.end()) {
    ginAnvilCuMemRelease(handleIt->second);
    fabricBufferHandles.erase(handleIt);
  }
  fabricBufferRefcount.erase(it);
}

// "Did every rank succeed?" over comm->bootstrap. ncclSystemError means the
// allgather ran and at least one rank voted no; any other non-success return is
// the allgather's own failure. Buffer sizing/indexing lives in
// ncclGinBootstrapAgree: GIN rail teams can have cctx->nranks != comm->nRanks,
// and bootstrapAllGather always indexes by comm->rank.
static ncclResult_t ginAnvilAllgatherOk(struct ncclComm* comm, int localOk) {
  int allOk = 1;
  NCCLCHECK(ncclGinBootstrapAgreeAll(comm, localOk, &allOk));
  return allOk ? ncclSuccess : ncclSystemError;
}

struct GinAnvilFabricVote {
  int ok;
  int published;
  uint64_t offset;
};

// Same agreement as ginAnvilAllgatherOk, plus two fabric-specific reductions in
// the one round trip: every rank must have resolved the same offset into the
// VMM allocation, and *allPublished reports whether every rank already holds a
// reference to this key (the re-registration path).
static ncclResult_t ginAnvilAllgatherFabricVote(struct ncclComm* comm, GinAnvilFabricVote local,
                                                int* allPublished) {
  int allOk = 1;
  int published = 1;
  ncclResult_t ret =
      ncclGinBootstrapAgree(comm, local, [&allOk, &published](GinAnvilFabricVote const* votes, int n) {
        for (int i = 0; i < n; i++) {
          if (!votes[i].ok) allOk = 0;
        }
        if (!allOk) return;
        for (int i = 1; i < n; i++) {
          if (votes[i].offset != votes[0].offset) {
            WARN("GIN anvil-sdma fabric: VMM offset mismatch (rank0=%llu rank%d=%llu)",
                 (unsigned long long)votes[0].offset, i, (unsigned long long)votes[i].offset);
            allOk = 0;
            return;
          }
        }
        for (int i = 0; i < n; i++) {
          if (!votes[i].published) published = 0;
        }
      });
  if (allPublished) *allPublished = (ret == ncclSuccess) ? published : 0;
  if (ret != ncclSuccess) return ret;
  return allOk ? ncclSuccess : ncclSystemError;
}

// Arms the fabric LSA AllToAll lane only if every rank opted in with the same
// threshold. The device kernel picks LSA or gin.put from that threshold, and
// LSA's barriers are collective, so ranks with different values would split
// between the two bodies and hang. *allEnabled is 0 unless the allgather
// succeeded and every vote matched; a non-success return is the allgather's own
// failure, not a negative vote.
static ncclResult_t ginAnvilAgreeLsaA2A(struct ncclComm* comm, int localEnabled, size_t localThreshold,
                                        int* allEnabled) {
  *allEnabled = 0;
  struct Vote {
    size_t threshold;
    int enabled;
  };
  const Vote local{localThreshold, localEnabled ? 1 : 0};
  int agreed = 1;
  NCCLCHECK(ncclGinBootstrapAgree(comm, local, [&agreed](Vote const* votes, int n) {
    for (int i = 0; i < n; ++i) {
      if (!votes[i].enabled || votes[i].threshold != votes[0].threshold) agreed = 0;
    }
  }));
  *allEnabled = agreed;
  return ncclSuccess;
}

// Per-rank comm objects whose LSA-signal peer connectivity has already been
// validated once. A process may own multiple ranks with the same commHash.
static std::set<struct ncclComm*> ginAnvilConnCheckedComms;

struct ginAnvilInitCtx {
  struct ncclComm* comm;
};

// Live ginAnvilInitCtx allocations. Only observable surface a test has for the
// finalize path, which is otherwise reached through ncclCommDestroy.
static std::atomic<int> ginAnvilLiveInitCtxs{0};

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
  ncclGinAnvilSdmaGPUContext* gpuCtxHost;
  struct ncclComm* comm;
  int nRanks;
  int rank;
  int nContexts;
  int nSignals;
  int nCounters;
  bool hasError;
  uint64_t* counters;
  void** gpu_queue_handles;
  uint64_t* sdma_dirty_d;
  int numChannels;
  int sdmaChannelStride;
  uintptr_t** signal_remote_addrs_dev;
  void* signalSpanLsaSelf;
};

static void ginAnvilSignalSpanRefInc(void* spanLsaSelf) {
  if (!spanLsaSelf) return;
  std::lock_guard<std::mutex> lock(pluginMutex);
  g_signalSpanRefcount[(uintptr_t)spanLsaSelf]++;
}

static void ginAnvilSignalSpanRefDec(void* spanLsaSelf) {
  if (!spanLsaSelf) return;
  std::lock_guard<std::mutex> lock(pluginMutex);
  auto it = g_signalSpanRefcount.find((uintptr_t)spanLsaSelf);
  if (it == g_signalSpanRefcount.end()) return;
  if (--it->second <= 0) {
    (void)ncclGinAnvilIpcTableUnregister(spanLsaSelf);
    g_signalSpanRefcount.erase(it);
  }
}

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

static void ginAnvilFreeGinCtx(ginAnvilGinCtx* ctx) {
  if (!ctx) return;
  if (ctx->comm) ginAnvilPendingRemove(ctx->comm, ctx);
  for (int contextId = 0; contextId < ctx->nContexts; contextId++) {
    if (ctx->gpuCtxHost) ncclGinAnvilIpcTableUntrackContext(&ctx->gpuCtxHost[contextId]);
    if (ctx->signal_remote_addrs_dev && ctx->signal_remote_addrs_dev[contextId]) {
      CUDACHECKIGNORE(hipFree(ctx->signal_remote_addrs_dev[contextId]));
    }
  }
  ginAnvilSignalSpanRefDec(ctx->signalSpanLsaSelf);
  if (ctx->counters) CUDACHECKIGNORE(hipFree(ctx->counters));
  if (ctx->gpuCtxDev) CUDACHECKIGNORE(hipFree(ctx->gpuCtxDev));
  free(ctx->signal_remote_addrs_dev);
  delete[] ctx->gpuCtxHost;
  free(ctx->devHandle);
  delete ctx;
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
  g_signalSpanRefcount.clear();
  ginAnvilConnCheckedComms.clear();
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

int ncclGinAnvilPluginTestLiveInitCtxCount(void) {
  return ginAnvilLiveInitCtxs.load(std::memory_order_relaxed);
}

void ncclGinAnvilPluginTestFabricState(void* collComm, void* data, struct ncclGinAnvilFabricTestState* out) {
  if (out == nullptr) return;
  std::lock_guard<std::mutex> lock(pluginMutex);
  out->handlers = fabricBufferHandlers.size();
  out->handles = fabricBufferHandles.size();
  out->refcounts = fabricBufferRefcount.size();
  auto it = fabricBufferRefcount.find(GinAnvilFabricKey{collComm, data});
  out->refcount = (it == fabricBufferRefcount.end()) ? 0 : it->second;
}

static ncclResult_t ginAnvilInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  const char* gin_type = getenv("NCCL_GIN_TYPE");
  if (gin_type && atoi(gin_type) != NCCL_NET_DEVICE_GIN_ANVIL_SDMA) return ncclInternalError;
  if (gin_anvil_sdma_probe() <= 0) return ncclInternalError;
  auto* ictx = new ginAnvilInitCtx{};
  ginAnvilLiveInitCtxs.fetch_add(1, std::memory_order_relaxed);
  *ctx = ictx;
  return ncclSuccess;
}

static ncclResult_t ginAnvilDevices(int* ndev) {
  *ndev = 1;
  return ncclSuccess;
}

// v14 GIN plugins expose GIN capability flags via getGinProperties.
// Strong signals are honoured per Put/PutValue by draining every SDMA queue
// to the peer before the signal (quietPeerQueues in gin_anvil_sdma.h). VA
// signals resolve through their window like a put destination and take the
// same strong/weak and fused paths as indexed signals.
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
  // strtoull skips leading whitespace before it sees a sign, so " -1" would
  // pass a bare *e != '-' check and come back as ULLONG_MAX. An overflowing
  // value does the same when errno is ignored. Both fall back to the default.
  if (e && e[0] && !std::isspace(static_cast<unsigned char>(e[0])) && *e != '-') {
    errno = 0;
    char* end = nullptr;
    unsigned long long v = strtoull(e, &end, 10);
    if (end != e && *end == '\0' && errno != ERANGE) return (size_t)v;
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

static uint32_t ginAnvilIpcSignalPeerFromEnv() {
  const char* e = getenv("NCCL_GIN_ANVIL_SDMA_SIGNAL_IPC");
  if (!e || !e[0]) return 0;
  if (e[0] == '0' && e[1] == '\0') return 0;
  return atoi(e) != 0 ? 1u : 0u;
}

static bool ginAnvilConnCheckEnabledFromEnv() {
  const char* e = getenv("NCCL_GIN_ANVIL_SDMA_CONN_CHECK");
  if (!e || !e[0]) return true;
  return !(e[0] == '0' && e[1] == '\0');
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
    // Erase here, not in finalize: closeColl also runs on backend-drop paths
    // (plugin/gin.cc) that never reach plugin finalize, and a recycled
    // ncclComm* would otherwise skip the gate. Host finalize keeps the backend
    // records for ncclGinFinalize (AICOMRCCL-2739), so finalize still runs for
    // a live connected backend and is not the only erase site.
    if (cctx->comm) {
      std::lock_guard<std::mutex> lock(pluginMutex);
      ginAnvilConnCheckedComms.erase(cctx->comm);
    }
    if (cctx->sdma) gin_anvil_sdma_destroy(cctx->sdma);
    delete cctx;
  }
  return ncclSuccess;
}

static ncclResult_t ginAnvilFinalize(void* ctx) {
  if (ctx) ginAnvilLiveInitCtxs.fetch_sub(1, std::memory_order_relaxed);
  delete (ginAnvilInitCtx*)ctx;
  return ncclSuccess;
}

static ncclResult_t ginAnvilRegMrSymFabric(ginAnvilCollCtx* cctx, void* data, size_t size, ginAnvilMemHandle* mh,
                                           void** mhandle, void** ginHandle) {
  struct ncclComm* comm = cctx->comm;
  const GinAnvilFabricKey key{cctx, data};
  ncclFabricMemHandler* handler = nullptr;
  CUdeviceptr memAddr = 0;
  size_t memSize = 0;
  int numSegments = 0;
  ncclResult_t ret = ncclSuccess;
  uintptr_t* remote_vas_host = nullptr;
  int setupOk = 1;
  int allPublished = 0;

  struct FabricSetupGuard {
    ncclFabricMemHandler* handler = nullptr;
    CUmemGenericAllocationHandle memHandle{};
    bool retained = false;
    bool published = false;
    ~FabricSetupGuard() {
      if (published) return;
      delete handler;
      if (retained) ginAnvilCuMemRelease(memHandle);
    }
  } guard;

  ret = ginAnvilQueryFabricVmmRange(data, size, &memAddr, &memSize, &numSegments);
  const uintptr_t offset =
      (ret == ncclSuccess && numSegments == 1)
          ? (reinterpret_cast<uintptr_t>(data) - reinterpret_cast<uintptr_t>(memAddr))
          : 0;
  if (ret != ncclSuccess) {
    setupOk = 0;
  } else if (numSegments != 1) {
    WARN("GIN anvil-sdma fabric: multi-segment MR not supported yet (segments=%d data=%p size=%zu)", numSegments, data,
         size);
    setupOk = 0;
    ret = ncclSystemError;
  } else if (offset + size > memSize) {
    WARN("GIN anvil-sdma fabric: registration [%p, %p) exceeds mapped segment [%p, %zu) for %p",
         reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(data)),
         reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(data) + size), reinterpret_cast<void*>(memAddr), memSize,
         data);
    setupOk = 0;
    ret = ncclSystemError;
  }

  int publishedLocal = 0;
  {
    std::lock_guard<std::mutex> lock(pluginMutex);
    auto it = fabricBufferRefcount.find(key);
    if (it != fabricBufferRefcount.end() && it->second > 0) publishedLocal = 1;
  }
  GinAnvilFabricVote vote{setupOk, publishedLocal, static_cast<uint64_t>(offset)};
  {
    ncclResult_t ag = ginAnvilAllgatherFabricVote(comm, vote, &allPublished);
    if (ag != ncclSuccess) return ag;
  }

  // Everything from here to the publish vote below is collective, so a local
  // failure is carried in publishOk rather than returned: a one-sided return
  // leaves the peers blocked in ginAnvilAllgatherOk instead of failing with us.
  int publishOk = 1;
  bool refTaken = false;

  if (allPublished) {
    // Re-acquire a reference on the buffer every rank just voted as published.
    // This can still miss if a concurrent ginAnvilDeregMrSym dropped the last
    // reference for this key between the vote and this recheck.
    std::lock_guard<std::mutex> lock(pluginMutex);
    auto it = fabricBufferRefcount.find(key);
    auto hit = fabricBufferHandlers.find(key);
    if (it == fabricBufferRefcount.end() || it->second <= 0 || hit == fabricBufferHandlers.end()) {
      publishOk = 0;
    } else {
      it->second++;
      refTaken = true;
      handler = hit->second;
    }
  } else {
    guard.handler = new (std::nothrow) ncclFabricMemHandler(comm->bootstrap, cctx->rank, cctx->nranks, comm->memManager);
    if (guard.handler == nullptr) {
      setupOk = 0;
      ret = ncclSystemError;
    } else {
      ret = ginAnvilRetainFabricHandle(&guard.memHandle, reinterpret_cast<void*>(memAddr));
      if (ret != ncclSuccess) {
        setupOk = 0;
      } else {
        guard.retained = true;
        ret = guard.handler->addSelfDeviceMem(reinterpret_cast<void*>(memAddr), guard.memHandle, memSize);
        if (ret != ncclSuccess) setupOk = 0;
      }
    }

    {
      ncclResult_t ag = ginAnvilAllgatherOk(comm, setupOk);
      if (ag != ncclSuccess) return ag;
    }

    if (guard.handler == nullptr || !guard.retained || !setupOk) return ncclSystemError;

    // Do not hold pluginMutex across exchangeMemPtrs (bootstrapAllGather).
    ret = guard.handler->exchangeMemPtrs();
    {
      const int exchangeOk = (ret == ncclSuccess) ? 1 : 0;
      ncclResult_t ag = ginAnvilAllgatherOk(comm, exchangeOk);
      if (ag != ncclSuccess) return ag;
    }
    if (ret != ncclSuccess) return ret;

    {
      std::lock_guard<std::mutex> lock(pluginMutex);
      auto refIt = fabricBufferRefcount.emplace(key, 0).first;
      auto& refcount = refIt->second;
      if (refcount == 0) {
        fabricBufferHandlers[key] = guard.handler;
        fabricBufferHandles[key] = guard.memHandle;
        handler = guard.handler;
        guard.published = true;
        INFO(NCCL_INIT, "GIN anvil-sdma fabric: exported addr=%p memBase=%p size=%zu offset=%zu", data,
             reinterpret_cast<void*>(memAddr), memSize, static_cast<size_t>(offset));
      } else {
        handler = fabricBufferHandlers[key];
      }
      refcount++;
      refTaken = true;
    }
  }

  if (handler == nullptr) publishOk = 0;

  mh->addr = data;
  mh->lsaSelfAddr = data;
  mh->size = size;
  mh->fabricMem = true;
  mh->remote_vas_dev = nullptr;
  mh->devHandle = nullptr;

  if (publishOk && hipMalloc(&mh->devHandle, sizeof(ncclGinAnvilSdmaMemHandle)) != hipSuccess) {
    mh->devHandle = nullptr;
    publishOk = 0;
  }
  if (publishOk) {
    remote_vas_host = (uintptr_t*)malloc(sizeof(uintptr_t) * (size_t)cctx->nranks);
    if (!remote_vas_host) publishOk = 0;
  }
  if (publishOk) {
    for (int pe = 0; pe < cctx->nranks; pe++) {
      void* peerPtr = nullptr;
      if (handler->getPeerDeviceMemPtr(pe, &peerPtr) != ncclSuccess || peerPtr == nullptr) {
        publishOk = 0;
        break;
      }
      remote_vas_host[pe] = reinterpret_cast<uintptr_t>(peerPtr) + offset;
    }
  }
  if (publishOk) {
    if (hipMalloc(&mh->remote_vas_dev, sizeof(uintptr_t) * (size_t)cctx->nranks) != hipSuccess ||
        hipMemcpy(mh->remote_vas_dev, remote_vas_host, sizeof(uintptr_t) * (size_t)cctx->nranks,
                  hipMemcpyHostToDevice) != hipSuccess) {
      publishOk = 0;
    }
  }
  {
    ncclResult_t ag = ginAnvilAllgatherOk(comm, publishOk);
    if (ag != ncclSuccess) publishOk = 0;
  }
  if (!publishOk) {
    if (mh->remote_vas_dev) {
      CUDACHECKIGNORE(hipFree(mh->remote_vas_dev));
      mh->remote_vas_dev = nullptr;
    }
    if (remote_vas_host) free(remote_vas_host);
    if (mh->devHandle) {
      CUDACHECKIGNORE(hipFree(mh->devHandle));
      mh->devHandle = nullptr;
    }
    if (refTaken) {
      std::lock_guard<std::mutex> lock(pluginMutex);
      ginAnvilFabricRefDropLocked(key);
    }
    return ncclSystemError;
  }
  free(remote_vas_host);

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
    if (mh->remote_vas_dev) {
      CUDACHECKIGNORE(hipFree(mh->remote_vas_dev));
      mh->remote_vas_dev = nullptr;
    }
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
  // Fabric MRs allgather on comm->bootstrap (world size). Rail GIN teams
  // (NCCL_GIN_CONNECTION_RAIL) have cctx->nranks != comm->nRanks; use LSA.
  if (ginAnvilUseFabricMem(cctx->comm) && cctx->comm && cctx->nranks == cctx->comm->nRanks) {
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
      ginAnvilFabricRefDropLocked(GinAnvilFabricKey{collComm, mh->addr});
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

// [GIN-CONN-CHECK] Validate that every peer can actually reach this rank's LSA
// signal buffer (and vice versa) over the coherent fabric. This probes the
// direct remote-address mapping used as SignalInc's fallback path. On gfx950
// the force-enabled cuMem VMM peer mapping
// intermittently comes back silently wrong for one rank, which otherwise turns
// into a first-collective hang (~1 run in 6). Detect it here and fail loudly.
// The pass/fail decision is made collectively (bootstrap allgather) so all ranks
// abort together rather than one rank aborting while the rest hang.
enum GinAnvilConnCheckStep {
  kConnCheckNone = 0,
  kConnCheckWrite,
  kConnCheckBarrierAfterWrite,
  kConnCheckVerify,
  kConnCheckD2H,
  kConnCheckReset,
  kConnCheckAllgather,
  kConnCheckBarrierAfterAllgather,
};

static const char* ginAnvilConnCheckStepName(GinAnvilConnCheckStep step) {
  switch (step) {
    case kConnCheckNone: return "none";
    case kConnCheckWrite: return "write";
    case kConnCheckBarrierAfterWrite: return "barrier-after-write";
    case kConnCheckVerify: return "verify";
    case kConnCheckD2H: return "d2h";
    case kConnCheckReset: return "reset";
    case kConnCheckAllgather: return "allgather";
    case kConnCheckBarrierAfterAllgather: return "barrier-after-allgather";
    default: return "unknown";
  }
}

static ncclResult_t ginAnvilCheckSignalConnectivity(ginAnvilGinCtx* ctx, void* lsaSelf,
                                                    uintptr_t* signalRemoteAddrsDev) {
  struct ncclComm* comm = ctx->comm;
  struct ncclDevrState* devr = &comm->devrState;
  const int nRanks = ctx->nRanks;
  const int lsaTeamSize = devr->lsaSize;
  const int rank = ctx->rank;
  constexpr int kMaxConnCheckRanks = gin_anvil::conn_check::kMaxConnCheckHostRanks;
  if (nRanks < 2) return ncclSuccess;
  if (nRanks > kMaxConnCheckRanks) {
    WARN("GIN anvil-sdma: conn-check supports at most %d ranks (got %d)", kMaxConnCheckRanks, nRanks);
    return ncclSystemError;
  }
  // Conn-check collectives must use the LSA team (lsaRankList/lsaSelf/lsaSize),
  // not the GIN team's ctx->rank/nRanks on comm->bootstrap. Under
  // NCCL_GIN_CONNECTION_RAIL those spaces differ: bootstrapAllGather writes
  // comm->nRanks entries and bootstrapBarrier addresses world ranks 0..nRanks-1.
  // An invalid team cannot be folded into setupState: that fold is itself an
  // allgather over this team. lsaSize/lsaSelf are derived identically on every
  // rank by ncclDevrInit, so this bails on all ranks together.
  if (lsaTeamSize < 1 || lsaTeamSize > kMaxConnCheckRanks || devr->lsaRankList == nullptr || devr->lsaSelf < 0 ||
      devr->lsaSelf >= lsaTeamSize) {
    WARN("GIN anvil-sdma: conn-check setup has invalid LSA team (rank %d, lsaSelf=%d, lsaSize=%d, lsaRankList=%p)",
         rank, devr->lsaSelf, lsaTeamSize, (void*)devr->lsaRankList);
    return ncclSystemError;
  }
  ncclResult_t ret = ncclSuccess;
  bool markedComm = false;
  hipStream_t connStream = nullptr;
  int* missingDev = nullptr;
  int missingHost[kMaxConnCheckRanks] = {};
  int gathered[kMaxConnCheckRanks];
  constexpr int MAX_ATTEMPTS = 3;
  int injRank = -1;
  bool ok = false;
  bool hasMissingSnapshot = false;
  int localMissing = 0;
  GinAnvilConnCheckStep failedStep = kConnCheckNone;

  // Setup and bypass are collective decisions. No rank may leave while peers
  // are about to enter the first conn-check barrier.
  enum SetupState {
    kSetupReady = 0,
    kSetupBypass = 1,
    kSetupFailed = 2,
    kSetupSkip = 3,
  };
  int setupState = kSetupReady;
  if (!ginAnvilConnCheckEnabledFromEnv()) {
    setupState = kSetupBypass;
  } else if (comm->globalGinSupport == NCCL_GIN_CONNECTION_RAIL || nRanks != devr->lsaSize ||
             ctx->nSignals < nRanks) {
    setupState = kSetupSkip;
  }
  if (setupState == kSetupReady && (signalRemoteAddrsDev == nullptr || lsaSelf == nullptr)) {
    WARN("GIN anvil-sdma: conn-check setup has null address (rank %d, remote-addrs=%p, lsaSelf=%p)",
         rank, (void*)signalRemoteAddrsDev, lsaSelf);
    setupState = kSetupFailed;
  }
  if (setupState == kSetupReady &&
      hipStreamCreateWithFlags(&connStream, hipStreamNonBlocking) != hipSuccess) {
    WARN("GIN anvil-sdma: conn-check hipStreamCreate failed (rank %d)", rank);
    setupState = kSetupFailed;
  }
  if (setupState == kSetupReady &&
      hipMalloc(&missingDev, sizeof(int) * (size_t)nRanks) != hipSuccess) {
    WARN("GIN anvil-sdma: conn-check hipMalloc failed (rank %d)", rank);
    setupState = kSetupFailed;
  }

  {
  const int lsaTeamRank = devr->lsaSelf;
  auto lsaAllgatherInts = [&](int* buf) -> ncclResult_t {
    return bootstrapIntraNodeAllGather(comm->bootstrap, devr->lsaRankList, lsaTeamRank, lsaTeamSize, buf,
                                       sizeof(int));
  };
  auto lsaBarrier = [&](int tag) -> ncclResult_t {
    return bootstrapIntraNodeBarrier(comm->bootstrap, devr->lsaRankList, lsaTeamRank, lsaTeamSize, tag);
  };

  std::fill_n(gathered, lsaTeamSize, -1);
  gathered[lsaTeamRank] = setupState;
  ret = lsaAllgatherInts(gathered);
  if (ret != ncclSuccess) {
    WARN("GIN anvil-sdma: conn-check setup allgather failed (rank %d)", rank);
    goto cleanup;
  }
  {
    int bypassRanks = 0;
    int failedRanks = 0;
    for (int r = 0; r < lsaTeamSize; r++) {
      bypassRanks += gathered[r] == kSetupBypass;
      failedRanks += gathered[r] == kSetupFailed;
    }
    if (failedRanks != 0) {
      WARN("GIN anvil-sdma: conn-check setup failed on %d rank(s)", failedRanks);
      ret = ncclSystemError;
      goto cleanup;
    }
    if (bypassRanks != 0) {
      if (bypassRanks != lsaTeamSize) {
        WARN("GIN anvil-sdma: conn-check bypass differs across ranks (%d/%d disabled)", bypassRanks, lsaTeamSize);
        ret = ncclSystemError;
      } else {
        INFO(NCCL_INIT,
             "GIN anvil-sdma: LSA signal conn-check disabled by NCCL_GIN_ANVIL_SDMA_CONN_CHECK=0");
      }
      goto cleanup;
    }
    int skipRanks = 0;
    for (int r = 0; r < lsaTeamSize; r++) skipRanks += gathered[r] == kSetupSkip;
    if (skipRanks != 0) {
      if (skipRanks != lsaTeamSize) {
        WARN("GIN anvil-sdma: conn-check skip differs across ranks (%d/%d ineligible)", skipRanks, lsaTeamSize);
        ret = ncclSystemError;
        goto cleanup;
      }
      if (comm->globalGinSupport == NCCL_GIN_CONNECTION_RAIL) {
        INFO(NCCL_INIT,
             "GIN anvil-sdma: skipping LSA signal conn-check (NCCL_GIN_CONNECTION_RAIL, rank %d, "
             "ginRank=%d, lsaSelf=%d, nRanks=%d, lsaSize=%d)",
             rank, rank, devr->lsaSelf, nRanks, devr->lsaSize);
      } else if (nRanks != devr->lsaSize) {
        INFO(NCCL_INIT,
             "GIN anvil-sdma: skipping LSA signal conn-check (nRanks=%d != lsaSize=%d, rank %d)", nRanks,
             devr->lsaSize, rank);
      } else {
        INFO(NCCL_INIT,
             "GIN anvil-sdma: skipping LSA signal conn-check (nSignals=%d < nRanks=%d, rank %d)",
             ctx->nSignals, nRanks, rank);
      }
      goto cleanup;
    }
  }

  {
    std::lock_guard<std::mutex> lock(pluginMutex);
    if (!ginAnvilConnCheckedComms.insert(comm).second) goto cleanup;
    markedComm = true;
  }

#ifdef ENABLE_FAULT_INJECTION
  if (const char* injEnv = getenv("NCCL_GIN_ANVIL_SDMA_CONN_INJECT_FAIL_RANK")) injRank = atoi(injEnv);
#endif

  int lastLocalMissing = 0;
  int lastGlobalMissing = 0;
  bool lastLocalFail = false;
  for (int attempt = 0; attempt < MAX_ATTEMPTS && !ok; attempt++) {
    unsigned long long stamp = 0xC0FFEE00ULL + (unsigned long long)(attempt + 1);
    bool localFail = false;
    failedStep = kConnCheckNone;
    hasMissingSnapshot = false;
    auto failAt = [&](GinAnvilConnCheckStep step) {
      localFail = true;
      failedStep = step;
    };
    auto warnStep = [&](GinAnvilConnCheckStep step) {
      WARN("GIN anvil-sdma: conn-check step '%s' failed (rank %d, attempt %d/%d)",
           ginAnvilConnCheckStepName(step), rank, attempt + 1, MAX_ATTEMPTS);
    };

    if (rank == injRank) {
      WARN("GIN anvil-sdma: [TEST] injecting connectivity fault on rank %d (skipping signal writes)", rank);
    } else if (ginAnvilConnWrite(signalRemoteAddrsDev, nRanks, rank, stamp, connStream) != 0) {
      failAt(kConnCheckWrite);
    }

    if (!localFail && hipStreamSynchronize(connStream) != hipSuccess) {
      failAt(kConnCheckWrite);
    }

    ret = lsaBarrier(0x51611);
    if (ret != ncclSuccess) {
      warnStep(kConnCheckBarrierAfterWrite);
      goto cleanup;
    }

    localMissing = 0;
    if (!localFail) {
      if (ginAnvilConnCheck(lsaSelf, nRanks, stamp, missingDev, connStream) != 0) {
        failAt(kConnCheckVerify);
      } else if (hipStreamSynchronize(connStream) != hipSuccess) {
        failAt(kConnCheckVerify);
      } else if (hipMemcpy(missingHost, missingDev, sizeof(int) * (size_t)nRanks, hipMemcpyDeviceToHost) !=
                 hipSuccess) {
        failAt(kConnCheckD2H);
      } else {
        hasMissingSnapshot = true;
        for (int x = 0; x < nRanks; x++) localMissing += missingHost[x];
        if (hipMemsetAsync(lsaSelf, 0, sizeof(uint64_t) * (size_t)nRanks, connStream) != hipSuccess) {
          failAt(kConnCheckReset);
        } else if (hipStreamSynchronize(connStream) != hipSuccess) {
          failAt(kConnCheckReset);
        }
      }
    }

    if (localFail) localMissing = lsaTeamSize;
    std::fill_n(gathered, lsaTeamSize, -1);
    gathered[lsaTeamRank] = localMissing;
    ret = lsaAllgatherInts(gathered);
    if (ret != ncclSuccess) {
      warnStep(kConnCheckAllgather);
      goto cleanup;
    }
    if (localFail) warnStep(failedStep);
    int globalMissing = 0;
    for (int r = 0; r < lsaTeamSize; r++) globalMissing += gathered[r];
    lastLocalMissing = localMissing;
    lastGlobalMissing = globalMissing;
    lastLocalFail = localFail;
    ret = lsaBarrier(0x51612);
    if (ret != ncclSuccess) {
      warnStep(kConnCheckBarrierAfterAllgather);
      goto cleanup;
    }
    if (globalMissing == 0 && !localFail) {
      ok = true;
      break;
    }
    if (rank == 0) {
      const char* missingKind =
          (!localFail && localMissing == 0 && globalMissing > 0) ? "peer-reported" : "global";
      WARN("GIN anvil-sdma: LSA signal connectivity attempt %d/%d incomplete "
           "(%s missing increments=%d); retrying",
           attempt + 1, MAX_ATTEMPTS, missingKind, globalMissing);
    }
  }

  if (!ok) {
    if (hasMissingSnapshot) {
      for (int x = 0; x < nRanks; x++) {
        if (!missingHost[x]) continue;
        WARN("GIN anvil-sdma: LSA signal connectivity FAILED: rank %d cannot receive from src %d "
             "(cuMem VMM peer mapping broken)",
             rank, x);
      }
    }
    if (!lastLocalFail && lastLocalMissing == 0 && lastGlobalMissing > 0) {
      WARN("GIN anvil-sdma: LSA signal connectivity gate failed after %d attempts on rank %d "
           "(peer-reported missing increments=%d). Re-launch the job, or set "
           "NCCL_GIN_ANVIL_SDMA_CONN_CHECK=0 to bypass.",
           MAX_ATTEMPTS, rank, lastGlobalMissing);
    } else if (lastLocalFail) {
      // localMissing is lsaTeamSize here only so the allgather aborts the team.
      // That sentinel is not a measured missing-stamp count.
      WARN("GIN anvil-sdma: LSA signal connectivity gate failed after %d attempts on rank %d "
           "(local step failure, last step='%s'). Re-launch the job, or set "
           "NCCL_GIN_ANVIL_SDMA_CONN_CHECK=0 to bypass.",
           MAX_ATTEMPTS, rank, ginAnvilConnCheckStepName(failedStep));
    } else {
      WARN("GIN anvil-sdma: LSA signal connectivity gate failed after %d attempts on rank %d "
           "(local missing=%d, last step='%s'). Re-launch the job, or set "
           "NCCL_GIN_ANVIL_SDMA_CONN_CHECK=0 to bypass.",
           MAX_ATTEMPTS, rank, lastLocalMissing, ginAnvilConnCheckStepName(failedStep));
    }
    ret = ncclSystemError;
  } else {
    INFO(NCCL_INIT, "GIN anvil-sdma: LSA signal connectivity OK (rank %d, nRanks %d)", rank, nRanks);
  }
  }

cleanup:
  if (markedComm && ret != ncclSuccess) {
    std::lock_guard<std::mutex> lock(pluginMutex);
    ginAnvilConnCheckedComms.erase(comm);
  }
  if (missingDev) CUDACHECKIGNORE(hipFree(missingDev));
  if (connStream) (void)hipStreamDestroy(connStream);
  return ret;
}

// Peer VAs for a signal region at local flat address `base`. The table is indexed by
// GIN-team peer, matching resolveRemotePeerVa's (peer - rsCtx->rank); lsaSelf is a
// different space under NCCL_GIN_CONNECTION_RAIL. Caller frees the result.
static uintptr_t* ginAnvilBuildSignalPeerAddrs(void* base, int nRanks, int ginRank, ptrdiff_t stride) {
  uintptr_t* addrs = (uintptr_t*)calloc((size_t)nRanks, sizeof(uintptr_t));
  if (!addrs) return nullptr;
  for (int pe = 0; pe < nRanks; pe++) {
    addrs[pe] = (uintptr_t)base + static_cast<ptrdiff_t>(pe - ginRank) * stride;
  }
  return addrs;
}

// The signal peer table is indexed by GIN peer, but each entry is a slot in this
// rank's LSA window. Returns the first GIN peer that is not the LSA team member in
// the slot its entry points at, or -1 if every entry is a mapped peer.
static int ginAnvilFirstSignalPeerOutsideLsa(const struct ncclComm* comm, int nRanks, int ginRank) {
  const struct ncclDevrState* devr = &comm->devrState;
  const long long ginStride =
    comm->globalGinSupport == NCCL_GIN_CONNECTION_RAIL ? comm->contiguousRanksPerHost : 1;
  for (int pe = 0; pe < nRanks; pe++) {
    const int slot = devr->lsaSelf + (pe - ginRank);
    const long long worldPeer = comm->rank + (long long)(pe - ginRank) * ginStride;
    if (devr->lsaRankList == nullptr || slot < 0 || slot >= devr->lsaSize || devr->lsaRankList[slot] != worldPeer) {
      return pe;
    }
  }
  return -1;
}

static ncclResult_t ginAnvilRegisterSignalSpan(struct ncclComm* comm, void* spanLsaSelf, size_t spanBytes, int nRanks,
                                               int ginRank) {
  struct ncclDevrState* devr = &comm->devrState;

  const int outsidePeer = ginAnvilFirstSignalPeerOutsideLsa(comm, nRanks, ginRank);
  if (outsidePeer >= 0) {
    WARN("GIN anvil-sdma: GIN peer %d is outside the LSA team of rank %d (ginRank=%d, ginRanks=%d, lsaSelf=%d, "
         "lsaSize=%d). Anvil SDMA can only signal LSA team peers; set NCCL_LSA_TEAM_SIZE to cover the GIN team "
         "or select another NCCL_GIN_TYPE.",
         outsidePeer, comm->rank, ginRank, nRanks, devr->lsaSelf, devr->lsaSize);
    return ncclInvalidUsage;
  }

  uintptr_t* hostAddrs = ginAnvilBuildSignalPeerAddrs(spanLsaSelf, nRanks, ginRank, (ptrdiff_t)devr->bigSize);
  if (!hostAddrs) return ncclSystemError;

  int rc = ncclGinAnvilIpcTableRegisterExplicit(spanLsaSelf, hostAddrs, nRanks, spanBytes);
  free(hostAddrs);
  if (rc != 0) {
    WARN("GIN anvil-sdma: LSA signal span IPC table register failed for %p size %zu", spanLsaSelf, spanBytes);
    return ncclSystemError;
  }

  // Probe the span base once. Every logical context's slot is a constant offset
  // from it, so running this per context would cost one bootstrap round trip per
  // context inside ncclDevCommCreate.
  //
  // bootstrapAllGather writes one slot per world rank, so the buffer is sized and
  // indexed in world space. nRanks/ginRank are the GIN team, which is smaller
  // under NCCL_GIN_CONNECTION_RAIL.
  const int worldRanks = comm->nRanks;
  uintptr_t* gatheredLocalBases =
    (worldRanks > 0 && comm->rank >= 0 && comm->rank < worldRanks)
      ? (uintptr_t*)calloc((size_t)worldRanks, sizeof(uintptr_t))
      : nullptr;
  if (gatheredLocalBases) {
    gatheredLocalBases[comm->rank] = (uintptr_t)spanLsaSelf;
    if (ginAnvilBootstrapAllgather(comm->bootstrap, gatheredLocalBases, sizeof(uintptr_t)) != 0) {
      WARN("GIN anvil-sdma: signal local-base allgather failed");
    } else if (gatheredLocalBases[comm->rank] != (uintptr_t)spanLsaSelf) {
      WARN("GIN anvil-sdma: signal local-base allgather mismatch rank=%d got=%#lx expect=%#lx", comm->rank,
           (unsigned long)gatheredLocalBases[comm->rank], (unsigned long)spanLsaSelf);
    }
    free(gatheredLocalBases);
  }
  return ncclSuccess;
}

static ncclResult_t ginAnvilBindContextSignals(ginAnvilGinCtx* ctx, int contextId, int signalSlot, void* lsaSelf,
                                               size_t bytes) {
  struct ncclDevrState* devr = &ctx->comm->devrState;
  ncclGinAnvilSdmaGPUContext* gpuCtxHost = &ctx->gpuCtxHost[contextId];

  gpuCtxHost->signals = (uint64_t*)lsaSelf;

  uintptr_t* hostAddrs = ginAnvilBuildSignalPeerAddrs(lsaSelf, ctx->nRanks, ctx->rank, (ptrdiff_t)devr->bigSize);
  if (!hostAddrs) {
    gpuCtxHost->signals = nullptr;
    return ncclSystemError;
  }

  if (hipMalloc(&ctx->signal_remote_addrs_dev[contextId], sizeof(uintptr_t) * (size_t)ctx->nRanks) != hipSuccess ||
      hipMemcpy(ctx->signal_remote_addrs_dev[contextId], hostAddrs, sizeof(uintptr_t) * (size_t)ctx->nRanks,
                hipMemcpyHostToDevice) != hipSuccess) {
    free(hostAddrs);
    if (ctx->signal_remote_addrs_dev[contextId]) {
      CUDACHECKIGNORE(hipFree(ctx->signal_remote_addrs_dev[contextId]));
      ctx->signal_remote_addrs_dev[contextId] = nullptr;
    }
    gpuCtxHost->signals = nullptr;
    return ncclSystemError;
  }
  gpuCtxHost->signal_remote_addrs = ctx->signal_remote_addrs_dev[contextId];

  if (ginAnvilSignalDebugEnabled()) {
    for (int pe = 0; pe < ctx->nRanks; pe++) {
      INFO(NCCL_INIT, "GIN anvil-sdma: signal_remote_addrs rank=%d peer=%d -> %#lx (local signals %#lx)", ctx->rank, pe,
           (unsigned long)hostAddrs[pe], (unsigned long)lsaSelf);
    }
  }

  uintptr_t remote0 = hostAddrs[0];
  uintptr_t remoteSelf = hostAddrs[ctx->rank];
  free(hostAddrs);
  if (hipMemcpy(ctx->gpuCtxDev + contextId, gpuCtxHost, sizeof(ncclGinAnvilSdmaGPUContext), hipMemcpyHostToDevice) !=
      hipSuccess) {
    CUDACHECKIGNORE(hipFree(ctx->signal_remote_addrs_dev[contextId]));
    ctx->signal_remote_addrs_dev[contextId] = nullptr;
    gpuCtxHost->signal_remote_addrs = nullptr;
    gpuCtxHost->signals = nullptr;
    return ncclSystemError;
  }
  INFO(NCCL_INIT,
       "GIN anvil-sdma: bound LSA signals context=%d slot=%d signals=%p bytes=%zu rank=%d lsaSelf=%d lsaSize=%d "
       "stride=%zu remote[0]=%#lx remote[self]=%#lx",
       contextId, signalSlot, lsaSelf, bytes, ctx->rank, devr->lsaSelf, devr->lsaSize, (size_t)devr->bigSize,
       (unsigned long)remote0, (unsigned long)remoteSelf);

  return ncclSuccess;
}

ncclResult_t ncclGinAnvilBindResourceWindowSignals(struct ncclComm* comm, void* resourceUserPtr, size_t arenaByteOffset,
                                                   int nSignalSlots, int nSignalsPerContext) {
  if (!comm || !resourceUserPtr || nSignalSlots < 1 || nSignalsPerContext < 1) return ncclInvalidArgument;

  void* spanLocalPtr = (char*)resourceUserPtr + arenaByteOffset;
  void* spanLsaSelf = nullptr;
  NCCLCHECK(ncclDevrGetLsaSelfAddr(&comm->devrState, spanLocalPtr, &spanLsaSelf));
  if (spanLsaSelf == nullptr) {
    WARN("GIN anvil-sdma: could not resolve LSA flat addr for resource-window signal span at %p", spanLocalPtr);
    ginAnvilPendingClear(comm);
    return ncclSystemError;
  }

  // Snapshot the pending list under pluginMutex. ginAnvilPendingAdd / Remove /
  // Clear all take that lock and it is not recursive, so the walks below cannot
  // hold it across ginAnvilRegisterSignalSpan or ginAnvilPendingClear. find()
  // also avoids default-inserting an empty list for a comm with nothing pending.
  std::vector<ginAnvilGinCtx*> pending;
  {
    std::lock_guard<std::mutex> lock(pluginMutex);
    auto it = g_pendingByComm.find(comm);
    if (it != g_pendingByComm.end()) {
      for (GinAnvilPendingEntry* e = it->second; e != nullptr; e = e->next) {
        if (e->ctx) pending.push_back(e->ctx);
      }
    }
  }

  const size_t spanBytes = (size_t)nSignalSlots * (size_t)nSignalsPerContext * sizeof(uint64_t);
  int spanRanks = 0;
  int spanGinRank = 0;
  for (ginAnvilGinCtx* ctx : pending) {
    if (ctx->nSignals > 0) {
      spanRanks = ctx->nRanks;
      spanGinRank = ctx->rank;
      break;
    }
  }
  if (spanRanks <= 0) {
    // Non-anvil GIN backends still call this bind from ncclDevCommCreate whenever
    // signals and a resource window exist. An empty pending list means nothing
    // anvil-owned needs binding; succeed so type-5 (and similar) bring-up keeps working.
    ginAnvilPendingClear(comm);
    return ncclSuccess;
  }

  ncclResult_t ret = ncclSuccess;
  int slot = 0;
  ginAnvilGinCtx* connCheckCtx = nullptr;
  void* connCheckLsaSelf = nullptr;
  NCCLCHECKGOTO(ginAnvilRegisterSignalSpan(comm, spanLsaSelf, spanBytes, spanRanks, spanGinRank), ret, fail);
  for (ginAnvilGinCtx* ctx : pending) {
    if (ctx->nSignals <= 0) continue;
    ctx->signalSpanLsaSelf = spanLsaSelf;
    ginAnvilSignalSpanRefInc(spanLsaSelf);
    for (int contextId = 0; contextId < ctx->nContexts; contextId++, slot++) {
      if (slot >= nSignalSlots) {
        WARN("GIN anvil-sdma: signal slot %d out of range (nSignalSlots=%d)", slot, nSignalSlots);
        ret = ncclInvalidArgument;
        goto fail;
      }

      size_t off = arenaByteOffset + (size_t)slot * (size_t)nSignalsPerContext * sizeof(uint64_t);
      void* localPtr = (char*)resourceUserPtr + off;
      void* lsaSelf = nullptr;
      NCCLCHECKGOTO(ncclDevrGetLsaSelfAddr(&comm->devrState, localPtr, &lsaSelf), ret, fail);
      if (lsaSelf == nullptr) {
        WARN("GIN anvil-sdma: could not resolve LSA flat addr for resource-window signals at %p", localPtr);
        ret = ncclSystemError;
        goto fail;
      }

      size_t bytes = (size_t)ctx->nSignals * sizeof(uint64_t);
      NCCLCHECKGOTO(ginAnvilBindContextSignals(ctx, contextId, slot, lsaSelf, bytes), ret, fail);
      if (connCheckCtx == nullptr) {
        connCheckCtx = ctx;
        connCheckLsaSelf = lsaSelf;
      }
    }
  }

  // [GIN-CONN-CHECK] Validate peer signal connectivity once per comm. Detects the
  // intermittent gfx950 cuMem-VMM peer-map fault and fails loudly here instead of
  // letting the first collective hang forever. This runs per span, not per logical
  // context: its setup allgather is collective over the LSA team, and the bypass
  // and skip paths return before the comm is marked, so a per-context call would
  // repeat that allgather nContexts times.
  if (connCheckCtx != nullptr) {
    bool alreadyChecked = false;
    {
      std::lock_guard<std::mutex> lock(pluginMutex);
      alreadyChecked = ginAnvilConnCheckedComms.count(comm) != 0;
    }
    if (!alreadyChecked) {
      NCCLCHECKGOTO(
        ginAnvilCheckSignalConnectivity(connCheckCtx, connCheckLsaSelf, connCheckCtx->signal_remote_addrs_dev[0]),
        ret, fail);
    }
  }

fail:
  if (ret != ncclSuccess) {
    for (ginAnvilGinCtx* ctx : pending) {
      if (ctx->signalSpanLsaSelf == spanLsaSelf) {
        ginAnvilSignalSpanRefDec(spanLsaSelf);
        ctx->signalSpanLsaSelf = nullptr;
      }
    }
  }
  ginAnvilPendingClear(comm);
  return ret;
}

static ncclResult_t ginAnvilCreateContext(void* collComm, ncclGinConfig_t* config, void** outGinCtx,
                                          ncclNetDeviceHandle_v11_t** outDevHandle) {
  ginAnvilCollCtx* cctx = (ginAnvilCollCtx*)collComm;
  ncclResult_t ret = ncclSuccess;
  auto* ctx = new ginAnvilGinCtx{};
  size_t threshold = 0;
  ctx->signalSpanLsaSelf = nullptr;
  ctx->nRanks = cctx->nranks;
  ctx->rank = cctx->rank;
  ctx->nContexts = config->nContexts > 0 ? config->nContexts : 1;
  ctx->nSignals = config->nSignals;
  ctx->nCounters = config->nCounters;
  ctx->comm = cctx->comm;
  ctx->hasError = false;
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

  ctx->gpuCtxHost = new ncclGinAnvilSdmaGPUContext[(size_t)ctx->nContexts]{};
  ctx->signal_remote_addrs_dev = (uintptr_t**)calloc((size_t)ctx->nContexts, sizeof(uintptr_t*));
  if (!ctx->signal_remote_addrs_dev) {
    ret = ncclSystemError;
    goto fail;
  }
  if (hipMalloc(&ctx->gpuCtxDev, (size_t)ctx->nContexts * sizeof(ncclGinAnvilSdmaGPUContext)) != hipSuccess) {
    ret = ncclSystemError;
    goto fail;
  }

  if (config->nCounters > 0) {
    const size_t countersBytes = sizeof(uint64_t) * (size_t)config->nCounters * (size_t)ctx->nContexts;
    if (hipExtMallocWithFlags((void**)&ctx->counters, countersBytes, hipDeviceMallocFinegrained) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }
    if (hipMemset(ctx->counters, 0, countersBytes) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }
  }

  threshold = ginAnvilSdmaThresholdFromEnv();
  for (int contextId = 0; contextId < ctx->nContexts; contextId++) {
    ncclGinAnvilSdmaGPUContext* gpuCtx = &ctx->gpuCtxHost[contextId];
    gpuCtx->layoutMagic = NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC;
    gpuCtx->nRanks = ctx->nRanks;
    gpuCtx->rank = ctx->rank;
    gpuCtx->nSignals = config->nSignals;
    gpuCtx->nCounters = config->nCounters;
    gpuCtx->numChannels = ctx->numChannels;
    gpuCtx->sdmaChannel = 0;
    gpuCtx->sdmaChannelStride = ctx->sdmaChannelStride;
    gpuCtx->queueHandles = ctx->gpu_queue_handles;
    gpuCtx->sdmaDirty = ctx->sdma_dirty_d;
    gpuCtx->sdmaThreshold = (threshold > (size_t)std::numeric_limits<uint32_t>::max())
                              ? std::numeric_limits<uint32_t>::max()
                              : (uint32_t)threshold;
    gpuCtx->fusedSdmaSignal = ginAnvilFusedSignalFromEnv();
    gpuCtx->ipcSignalPeer = ginAnvilIpcSignalPeerFromEnv();
    gpuCtx->counters =
        ctx->counters != nullptr ? ctx->counters + (size_t)contextId * (size_t)config->nCounters : nullptr;
    ncclGinAnvilIpcTableGetDevice(&gpuCtx->ipcTable, &gpuCtx->ipcTableCount);
  }

  // Only context 0 (what devComm.ginHandles[0] points at) gets the fabric LSA
  // lane, since that is the context the device-API AllToAll reads. The others
  // use gin.put.
  if (ginAnvilUseFabricMem(cctx->comm)) {
    ncclGinAnvilSdmaGPUContext* gpuCtx = &ctx->gpuCtxHost[0];
    const size_t lsaThreshold = gin::fabric::resolveGinFabricLsaThresholdAlltoAll(ctx->nRanks);
    const bool lsaCapable = ginFabricLsaA2ACapable(cctx->comm, ctx->nRanks) && lsaThreshold > 0;
    const int localLsaEnabled = gin::fabric::ginFabricLsaA2AWantEnabled(lsaCapable) ? 1 : 0;
    int allLsaEnabled = 0;
    NCCLCHECKGOTO(ginAnvilAgreeLsaA2A(cctx->comm, localLsaEnabled, lsaThreshold, &allLsaEnabled), ret, fail);
    if (allLsaEnabled) {
      gpuCtx->fabricA2ALsaEnabled = 1;
      gpuCtx->fabricA2ALsaThreshold = lsaThreshold;
    }
  }

  if (hipMemcpy(ctx->gpuCtxDev, ctx->gpuCtxHost,
                (size_t)ctx->nContexts * sizeof(ncclGinAnvilSdmaGPUContext), hipMemcpyHostToDevice) != hipSuccess) {
    WARN("GIN anvil-sdma: hipMemcpy gpu context failed");
    ret = ncclSystemError;
    goto fail;
  }

  for (int contextId = 0; contextId < ctx->nContexts; contextId++) {
    ncclGinAnvilIpcTableTrackContext(&ctx->gpuCtxHost[contextId], &ctx->gpuCtxDev[contextId]);
  }

  if (config->nSignals > 0) {
    ginAnvilPendingAdd(cctx->comm, ctx);
  }

  ctx->devHandle->handle = ctx->gpuCtxDev;
  ctx->devHandle->size = (size_t)ctx->nContexts * sizeof(ncclGinAnvilSdmaGPUContext);

  *outGinCtx = ctx;
  *outDevHandle = ctx->devHandle;
  INFO(NCCL_INIT,
       "GIN anvil-sdma: context created (v%d, %d logical contexts, %d signals, %d counters, sdmaThreshold=%u, "
       "%d channels, spread=%d, fusedSignal=%u)",
       NCCL_GIN_ANVIL_SDMA_NET_VERSION, ctx->nContexts, config->nSignals, config->nCounters,
       ctx->gpuCtxHost[0].sdmaThreshold, ctx->numChannels, ctx->gpuCtxHost[0].sdmaChannelStride,
       ctx->gpuCtxHost[0].fusedSdmaSignal);
  return ncclSuccess;

fail:
  if (ctx) ginAnvilFreeGinCtx(ctx);
  return ret;
}

static ncclResult_t ginAnvilDestroyContext(void* ginCtx) {
  ginAnvilGinCtx* ctx = (ginAnvilGinCtx*)ginCtx;
  if (!ctx) return ncclSuccess;
  ginAnvilFreeGinCtx(ctx);
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
