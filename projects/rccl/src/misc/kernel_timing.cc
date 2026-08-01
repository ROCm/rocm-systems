/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/* Per-dispatch kernel timing taken from the dispatch packet itself.
 *
 * RCCL's other timing path brackets a launch with two stream-marker events,
 * which costs ~9 us per dispatch and reports the marker interval rather than
 * the kernel's. Attaching a stop event to the dispatch instead costs nothing
 * measurable and carries the packet processor's own start/end timestamps.
 *
 * Those timestamps originate in ROCr, which writes them into the dispatch's
 * completion signal; ROCclr converts them to the system clock domain and caches
 * the pair on the event. HIP exposes neither, so the pair is read out of the
 * runtime's own structures. Nothing about that layout is assumed: on the first
 * timed dispatch, hsa_amd_profiling_get_dispatch_time is used as an oracle to
 * establish ground truth, and only a location whose contents match the oracle
 * exactly is trusted afterwards. If no such location is found, timing stays off
 * for the process rather than reporting numbers we cannot vouch for.
 */

#include "kernel_timing.h"

#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <set>
#include <unistd.h>
#include <vector>

#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "param.h"

#if defined(__HIP_PLATFORM_AMD__)
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif

RCCL_PARAM(KernelTiming, "KERNEL_TIMING", 0);
RCCL_PARAM(KernelTimingInflight, "KERNEL_TIMING_INFLIGHT", 1024);
RCCL_PARAM(KernelTimingCapacity, "KERNEL_TIMING_CAPACITY", 65536);

namespace {

/* ---------- reading the dispatch timestamps out of an event ---------- */

/* Walking runtime-owned memory means occasionally following a value that is not
 * a pointer. Going through /proc/self/mem turns that into an error return
 * instead of a fault. Only discovery pays for this; the steady-state path
 * dereferences the validated chain directly. */
int probeFd() {
  static int fd = open("/proc/self/mem", O_RDONLY);
  return fd;
}

bool probeRead(uint64_t addr, void* out, size_t n) {
  if (addr < 0x10000 || (addr & 7)) return false;
  return pread(probeFd(), out, n, (off_t)addr) == (ssize_t)n;
}

struct Chain {
  int hop[4];
  int nhop = 0;
  int leaf = -1;
  bool valid() const { return leaf >= 0; }
};

bool followChain(void* event, const Chain& c, uint64_t* startNs, uint64_t* endNs) {
  uint64_t p = (uint64_t)event;
  for (int i = 0; i < c.nhop; ++i) {
    p = *(uint64_t*)(p + c.hop[i]);
    if (p < 0x10000 || (p & 7)) return false;
  }
  *startNs = *(uint64_t*)(p + c.leaf);
  *endNs = *(uint64_t*)(p + c.leaf + 8);
  return *endNs >= *startNs;
}

constexpr int kScanWords = 32; /* 256 bytes per visited block */
constexpr int kScanDepth = 3;

#if defined(__HIP_PLATFORM_AMD__)
hsa_agent_t g_gpuAgents[16];
int g_nGpuAgents = 0;

hsa_status_t collectGpuAgent(hsa_agent_t a, void*) {
  hsa_device_type_t t;
  if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t) == HSA_STATUS_SUCCESS && t == HSA_DEVICE_TYPE_GPU &&
      g_nGpuAgents < (int)(sizeof(g_gpuAgents) / sizeof(g_gpuAgents[0]))) {
    g_gpuAgents[g_nGpuAgents++] = a;
  }
  return HSA_STATUS_SUCCESS;
}

/* A ROCr signal handle is a pointer to amd_signal_t, so a candidate can be
 * screened before it is handed to the runtime -- passing a bogus handle to
 * hsa_amd_profiling_get_dispatch_time would fault inside ROCr. */
bool dispatchTimeFromSignal(uint64_t handle, uint64_t* startNs, uint64_t* endNs) {
  struct {
    int64_t kind;
    int64_t value;
    uint64_t mailbox;
    uint32_t id, reserved;
    uint64_t startTs, endTs;
  } hdr;
  if (!probeRead(handle, &hdr, sizeof(hdr))) return false;
  if (hdr.kind != AMD_SIGNAL_KIND_USER) return false;
  if (hdr.startTs == 0 || hdr.endTs <= hdr.startTs) return false;

  hsa_signal_t sig;
  sig.handle = handle;
  for (int i = 0; i < g_nGpuAgents; ++i) {
    hsa_amd_profiling_dispatch_time_t t;
    if (hsa_amd_profiling_get_dispatch_time(g_gpuAgents[i], sig, &t) == HSA_STATUS_SUCCESS && t.end > t.start) {
      *startNs = t.start;
      *endNs = t.end;
      return true;
    }
  }
  return false;
}
#else
bool dispatchTimeFromSignal(uint64_t, uint64_t*, uint64_t*) { return false; }
#endif

void walk(uint64_t base, int depth, int* hop, int nhop, std::set<uint64_t>& seen,
          void (*visit)(uint64_t base, const uint64_t* words, const int* hop, int nhop, void* arg), void* arg) {
  if (depth > kScanDepth || !seen.insert(base).second) return;
  uint64_t w[kScanWords];
  if (!probeRead(base, w, sizeof(w))) return;
  visit(base, w, hop, nhop, arg);
  if (nhop == (int)(sizeof(Chain::hop) / sizeof(int))) return;
  for (int i = 0; i < kScanWords; ++i) {
    hop[nhop] = i * 8;
    walk(w[i], depth + 1, hop, nhop + 1, seen, visit, arg);
  }
}

struct OracleSearch {
  uint64_t startNs, endNs;
  bool found;
};

void visitForOracle(uint64_t, const uint64_t* w, const int*, int, void* arg) {
  OracleSearch* s = (OracleSearch*)arg;
  if (s->found) return;
  for (int i = 0; i < kScanWords; ++i) {
    if (dispatchTimeFromSignal(w[i], &s->startNs, &s->endNs)) {
      s->found = true;
      return;
    }
  }
}

struct ChainSearch {
  uint64_t startNs, endNs;
  Chain chain;
};

void visitForChain(uint64_t, const uint64_t* w, const int* hop, int nhop, void* arg) {
  ChainSearch* s = (ChainSearch*)arg;
  if (s->chain.valid()) return;
  for (int i = 0; i + 1 < kScanWords; ++i) {
    if (w[i] == s->startNs && w[i + 1] == s->endNs) {
      for (int h = 0; h < nhop; ++h) s->chain.hop[h] = hop[h];
      s->chain.nhop = nhop;
      s->chain.leaf = i * 8;
      return;
    }
  }
}

/* Locates where the runtime keeps this dispatch's timestamps, using ROCr's own
 * answer as the reference. */
Chain discoverChain(void* event) {
  Chain none;
#if defined(__HIP_PLATFORM_AMD__)
  if (g_nGpuAgents == 0) hsa_iterate_agents(collectGpuAgent, nullptr);
  if (g_nGpuAgents == 0) return none;

  int hop[4];
  OracleSearch oracle = {0, 0, false};
  {
    std::set<uint64_t> seen;
    walk((uint64_t)event, 0, hop, 0, seen, visitForOracle, &oracle);
  }
  if (!oracle.found) return none;

  ChainSearch cs;
  cs.startNs = oracle.startNs;
  cs.endNs = oracle.endNs;
  {
    std::set<uint64_t> seen;
    walk((uint64_t)event, 0, hop, 0, seen, visitForChain, &cs);
  }
  /* A cached copy is preferred over the signal itself: the signal may be reused
   * by a later packet before we harvest, which would silently yield another
   * dispatch's times, whereas the copy is a snapshot valid for the event's
   * lifetime. */
  if (cs.chain.valid()) {
    uint64_t s = 0, e = 0;
    if (followChain(event, cs.chain, &s, &e) && s == oracle.startNs && e == oracle.endNs) return cs.chain;
  }
#endif
  return none;
}

} // namespace

/* ---------- per-communicator state ---------- */

struct ncclKernelTimingCtx {
  std::mutex lock;
  Chain chain;
  bool discoveryDone = false;
  bool disabled = false;

  /* Dispatches launched but not yet harvested, in launch order. A slot is
   * reserved before the launch and armed only once it has been issued. */
  std::vector<cudaEvent_t> inflightEvent;
  std::vector<ncclKernelTimingRecord> inflightRec;
  std::vector<bool> inflightArmed;
  uint64_t inflightHead = 0, inflightTail = 0;

  /* Completed records awaiting a drain. */
  std::vector<ncclKernelTimingRecord> ring;
  uint64_t ringHead = 0, ringTail = 0;
  uint64_t dropped = 0;

  uint64_t seq = 0;
};

bool ncclKernelTimingEnabled() {
#if defined(__HIP_PLATFORM_AMD__)
  return rcclParamKernelTiming() != 0;
#else
  return false;
#endif
}

ncclResult_t ncclKernelTimingCommInit(struct ncclComm* comm) {
  comm->kernelTiming = nullptr;
  if (!ncclKernelTimingEnabled()) return ncclSuccess;

  int inflight = (int)rcclParamKernelTimingInflight();
  int capacity = (int)rcclParamKernelTimingCapacity();
  if (inflight < 8) inflight = 8;
  if (capacity < inflight) capacity = inflight;

  ncclKernelTimingCtx* ctx = new ncclKernelTimingCtx();
  ctx->inflightEvent.resize(inflight, nullptr);
  ctx->inflightRec.resize(inflight);
  ctx->inflightArmed.resize(inflight, false);
  ctx->ring.resize(capacity);
  for (int i = 0; i < inflight; i++) {
    /* Timing must stay enabled on these events; they are the timestamp source. */
    if (cudaEventCreateWithFlags(&ctx->inflightEvent[i], cudaEventDefault) != cudaSuccess) {
      for (int j = 0; j < i; j++) (void)cudaEventDestroy(ctx->inflightEvent[j]);
      delete ctx;
      WARN("KERNEL_TIMING: could not create events, kernel timing disabled");
      return ncclSuccess;
    }
  }
  comm->kernelTiming = ctx;
  INFO(NCCL_INIT, "KERNEL_TIMING: enabled (inflight %d, capacity %d)", inflight, capacity);
  return ncclSuccess;
}

ncclResult_t ncclKernelTimingCommFree(struct ncclComm* comm) {
  ncclKernelTimingCtx* ctx = comm->kernelTiming;
  if (ctx == nullptr) return ncclSuccess;
  comm->kernelTiming = nullptr;
  for (size_t i = 0; i < ctx->inflightEvent.size(); i++) {
    if (ctx->inflightEvent[i]) (void)cudaEventDestroy(ctx->inflightEvent[i]);
  }
  delete ctx;
  return ncclSuccess;
}

namespace {

void pushRecord(ncclKernelTimingCtx* ctx, const ncclKernelTimingRecord& rec) {
  uint64_t capacity = ctx->ring.size();
  if (ctx->ringHead - ctx->ringTail == capacity) {
    ctx->ringTail++; /* overwrite oldest */
    ctx->dropped++;
  }
  ctx->ring[ctx->ringHead % capacity] = rec;
  ctx->ringHead++;
}

/* Moves every completed dispatch from the in-flight queue into the ring. Called
 * from the launch path, so it must never block: it stops at the first dispatch
 * still running. */
void harvest(ncclKernelTimingCtx* ctx) {
  uint64_t inflight = ctx->inflightEvent.size();
  while (ctx->inflightTail != ctx->inflightHead) {
    uint64_t slot = ctx->inflightTail % inflight;
    if (!ctx->inflightArmed[slot]) {
      /* Reserved but never issued: its event holds whatever the previous use
       * left behind, so drop it. */
      ctx->inflightTail++;
      continue;
    }
    cudaEvent_t ev = ctx->inflightEvent[slot];
    if (cudaEventQuery(ev) != cudaSuccess) break;

    if (!ctx->discoveryDone) {
      ctx->discoveryDone = true;
      ctx->chain = discoverChain(ev);
      if (!ctx->chain.valid()) {
        ctx->disabled = true;
        WARN("KERNEL_TIMING: could not locate dispatch timestamps in this ROCm runtime; timing disabled");
      } else {
        INFO(NCCL_INIT, "KERNEL_TIMING: dispatch timestamps located (%d hops, leaf +%d)", ctx->chain.nhop,
             ctx->chain.leaf);
      }
    }

    ncclKernelTimingRecord rec = ctx->inflightRec[slot];
    if (!ctx->disabled && followChain(ev, ctx->chain, &rec.startNs, &rec.endNs)) pushRecord(ctx, rec);
    ctx->inflightArmed[slot] = false;
    ctx->inflightTail++;
  }
}

} // namespace

cudaEvent_t ncclKernelTimingBeginLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, uint32_t nChannels,
                                        uint32_t nThreads, uint64_t* slotOut) {
  ncclKernelTimingCtx* ctx = comm->kernelTiming;
  if (ctx == nullptr) return nullptr;
  /* Graph-captured launches accept the event but replay never populates it. */
  if (plan->persistent) return nullptr;

  std::lock_guard<std::mutex> guard(ctx->lock);
  harvest(ctx);
  if (ctx->disabled) return nullptr;

  uint64_t inflight = ctx->inflightEvent.size();
  if (ctx->inflightHead - ctx->inflightTail == inflight) {
    /* Nothing has completed and every slot is held; skip timing this dispatch
     * rather than stalling the launch. */
    ctx->dropped++;
    return nullptr;
  }

  uint64_t ticket = ctx->inflightHead++;
  *slotOut = ticket;
  uint64_t slot = ticket % inflight;
  ctx->inflightArmed[slot] = false;
  ncclKernelTimingRecord& rec = ctx->inflightRec[slot];
  memset(&rec, 0, sizeof(rec));
  rec.seq = ctx->seq++;
  rec.commHash = comm->commHash;
  rec.rank = comm->rank;
  rec.nChannels = nChannels;
  rec.nThreads = nThreads;
  for (struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue); ct != nullptr; ct = ct->next) {
    if (rec.nColls == 0) {
      rec.func = (uint32_t)ct->func;
      rec.datatype = (uint32_t)ct->datatype;
      rec.count = (uint64_t)ct->count;
    }
    rec.nColls++;
  }
  return ctx->inflightEvent[slot];
}

void ncclKernelTimingCommitLaunch(struct ncclComm* comm, uint64_t slot) {
  ncclKernelTimingCtx* ctx = comm->kernelTiming;
  if (ctx == nullptr) return;
  std::lock_guard<std::mutex> guard(ctx->lock);
  ctx->inflightArmed[slot % ctx->inflightEvent.size()] = true;
}

NCCL_API(ncclResult_t, ncclKernelTimingDrain, ncclComm_t comm, ncclKernelTimingRecord* out, int max, int* got,
         uint64_t* dropped);
ncclResult_t ncclKernelTimingDrain(ncclComm_t comm, ncclKernelTimingRecord* out, int max, int* got,
                                   uint64_t* dropped) {
  if (comm == nullptr || out == nullptr || got == nullptr || max < 0) return ncclInvalidArgument;
  ncclKernelTimingCtx* ctx = comm->kernelTiming;
  if (ctx == nullptr) return ncclInvalidUsage;

  std::lock_guard<std::mutex> guard(ctx->lock);
  harvest(ctx);

  uint64_t capacity = ctx->ring.size();
  int n = 0;
  while (n < max && ctx->ringTail != ctx->ringHead) {
    out[n++] = ctx->ring[ctx->ringTail % capacity];
    ctx->ringTail++;
  }
  *got = n;
  if (dropped) *dropped = ctx->dropped;
  return ncclSuccess;
}
