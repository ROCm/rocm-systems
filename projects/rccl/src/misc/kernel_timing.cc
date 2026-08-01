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
 * completion signal. HIP does not expose that signal, so on the first timed
 * dispatch its handle is located inside the runtime's own structures: memory
 * reachable from the event is scanned for something ROCr accepts as a signal it
 * has timed. Only the location is discovered. The timestamps themselves come
 * from hsa_amd_profiling_get_dispatch_time on every harvest, so nothing depends
 * on how HIP happens to cache or convert them. If no signal is found, timing
 * stays off for the process rather than reporting numbers we cannot vouch for.
 */

#include "kernel_timing.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <set>
#include <time.h>
#include <unistd.h>
#include <utility>
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
RCCL_PARAM(KernelTimingHarvestOnLaunch, "KERNEL_TIMING_HARVEST_ON_LAUNCH", 0);
RCCL_PARAM(KernelTimingScanWords, "KERNEL_TIMING_SCAN_WORDS", 32);
RCCL_PARAM(KernelTimingScanDepth, "KERNEL_TIMING_SCAN_DEPTH", 3);

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

/* Guards the per-dispatch reads. A word that held a pointer for one dispatch
 * may hold anything for the next, and a bad dereference would take the
 * application down, so every address is checked against the process's mappings
 * first. The map is cached: a hit costs a binary search, and only a miss goes
 * back to the kernel. */
class MappedRanges {
 public:
  bool readable(uint64_t addr, size_t n) {
    if (addr < 0x10000 || (addr & 7)) return false;
    std::lock_guard<std::mutex> guard(lock_);
    if (lookup(addr, n)) return true;
    /* The runtime maps more memory as it goes, so a miss usually just means the
     * cache is stale and is worth re-reading for. Only once a refresh has
     * failed to explain a miss -- the address is simply bad -- is the next
     * refresh held off, so that a broken chain cannot re-read the map on every
     * dispatch. */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
    uint64_t now = (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
    if (lastRefreshMissed_ && now - lastRefreshNs_ < 10000000ull) return false;
    lastRefreshNs_ = now;
    refresh();
    bool ok = lookup(addr, n);
    lastRefreshMissed_ = !ok;
    return ok;
  }

 private:
  void refresh() {
    ranges_.clear();
    FILE* f = fopen("/proc/self/maps", "r");
    if (f == nullptr) return;
    char line[512];
    while (fgets(line, sizeof(line), f) != nullptr) {
      uint64_t lo, hi;
      char perms[8];
      if (sscanf(line, "%lx-%lx %7s", &lo, &hi, perms) != 3 || perms[0] != 'r') continue;
      if (!ranges_.empty() && ranges_.back().second == lo) {
        ranges_.back().second = hi;
      } else {
        ranges_.emplace_back(lo, hi);
      }
    }
    fclose(f);
  }

  bool lookup(uint64_t addr, size_t n) const {
    auto it = std::upper_bound(ranges_.begin(), ranges_.end(), addr,
                               [](uint64_t v, const std::pair<uint64_t, uint64_t>& r) { return v < r.first; });
    if (it == ranges_.begin()) return false;
    --it;
    return addr >= it->first && addr + n <= it->second;
  }

  std::mutex lock_;
  std::vector<std::pair<uint64_t, uint64_t>> ranges_;
  uint64_t lastRefreshNs_ = 0;
  bool lastRefreshMissed_ = false;
};

MappedRanges& mappedRanges() {
  static MappedRanges m;
  return m;
}

/* Where the event keeps the handle of the completion signal its dispatch was
 * issued with: a sequence of pointer hops from the event object, then the byte
 * offset of the handle word. */
struct Chain {
  int hop[4];
  int nhop = 0;
  int leaf = -1;
  bool valid() const { return leaf >= 0; }
};

bool followChain(void* event, const Chain& c, uint64_t* handle) {
  uint64_t p = (uint64_t)event;
  for (int i = 0; i < c.nhop; ++i) {
    if (!mappedRanges().readable(p + c.hop[i], 8)) return false;
    p = *(uint64_t*)(p + c.hop[i]);
  }
  if (!mappedRanges().readable(p + c.leaf, 8)) return false;
  *handle = *(uint64_t*)(p + c.leaf);
  return true;
}

/* How much of the runtime's object graph discovery is willing to look at. The
 * defaults reach the signal in the runtimes tried so far; a layout that keeps
 * it further away can be reached by raising them, at the cost of a longer
 * one-time scan. */
constexpr int kMaxScanWords = 128;
int scanWords() {
  int n = (int)rcclParamKernelTimingScanWords();
  return n < 8 ? 8 : (n > kMaxScanWords ? kMaxScanWords : n);
}
int scanDepth() { return (int)rcclParamKernelTimingScanDepth(); }

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

/* ROCr converts a dispatch's ticks with the clock calibration of the agent it
 * is asked about, and it does not check that the signal belongs to that agent:
 * asking the wrong one returns a plausible but skewed answer. The agent is
 * therefore matched to the communicator's device by PCI address rather than by
 * enumeration order, which HIP and HSA need not share. */
bool agentForDevice(int dev, hsa_agent_t* out) {
  if (g_nGpuAgents == 0) hsa_iterate_agents(collectGpuAgent, nullptr);

  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;

  for (int i = 0; i < g_nGpuAgents; ++i) {
    uint32_t bdfid = 0;
    if (hsa_agent_get_info(g_gpuAgents[i], (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdfid) != HSA_STATUS_SUCCESS)
      continue;
    if ((int)((bdfid >> 8) & 0xff) == prop.pciBusID && (int)((bdfid >> 3) & 0x1f) == prop.pciDeviceID) {
      *out = g_gpuAgents[i];
      return true;
    }
  }
  return false;
}

/* Why a discovery attempt ended where it did, for the failure log. */
struct DiscoveryStats {
  int blocks = 0;    /* memory blocks reachable from the event */
  int signals = 0;   /* words that look like an amd_signal_t handle */
  int stamped = 0;   /* those carrying a plausible timestamp pair */
  int oracleOk = 0;  /* those ROCr agreed to report a dispatch time for */
};

bool dispatchTimeFromSignal(uint64_t handle, hsa_agent_t agent, uint64_t* startNs, uint64_t* endNs) {
  hsa_signal_t sig;
  sig.handle = handle;
  hsa_amd_profiling_dispatch_time_t t;
  if (hsa_amd_profiling_get_dispatch_time(agent, sig, &t) != HSA_STATUS_SUCCESS) return false;
  if (t.end <= t.start) return false;
  *startNs = t.start;
  *endNs = t.end;
  return true;
}

/* Screens a candidate word before it is handed to the runtime -- passing a
 * bogus handle to hsa_amd_profiling_get_dispatch_time would fault inside ROCr.
 * A ROCr signal handle is a pointer to amd_signal_t, so the header can be read
 * through /proc/self/mem first and rejected without risk. */
bool searchSignal(uint64_t handle, hsa_agent_t agent, uint64_t* startNs, uint64_t* endNs, DiscoveryStats* st) {
  struct {
    int64_t kind;
    int64_t value;
    uint64_t mailbox;
    uint32_t id, reserved;
    uint64_t startTs, endTs;
  } hdr;
  if (!probeRead(handle, &hdr, sizeof(hdr))) return false;
  if (hdr.kind != AMD_SIGNAL_KIND_USER) return false;
  st->signals++;
  if (hdr.startTs == 0 || hdr.endTs <= hdr.startTs) return false;
  st->stamped++;

  if (!dispatchTimeFromSignal(handle, agent, startNs, endNs)) return false;
  st->oracleOk++;
  return true;
}

/* ROCr reads the whole amd_signal_t, so the handle has to be known-good before
 * it is passed in, not merely non-null. */
bool signalLooksLive(uint64_t handle) {
  if (!mappedRanges().readable(handle, sizeof(amd_signal_t))) return false;
  return *(const int64_t*)handle == AMD_SIGNAL_KIND_USER;
}
#else
struct DiscoveryStats {
  int blocks = 0, signals = 0, stamped = 0, oracleOk = 0;
};
struct hsa_agent_t {
  uint64_t handle;
};
bool dispatchTimeFromSignal(uint64_t, hsa_agent_t, uint64_t*, uint64_t*) { return false; }
bool signalLooksLive(uint64_t) { return false; }
#endif

void walk(uint64_t base, int depth, int* hop, int nhop, std::set<uint64_t>& seen,
          void (*visit)(uint64_t base, const uint64_t* words, const int* hop, int nhop, void* arg), void* arg) {
  if (depth > scanDepth() || !seen.insert(base).second) return;
  const int nw = scanWords();
  uint64_t w[kMaxScanWords];
  if (!probeRead(base, w, nw * sizeof(uint64_t))) return;
  visit(base, w, hop, nhop, arg);
  if (nhop == (int)(sizeof(Chain::hop) / sizeof(int))) return;
  for (int i = 0; i < nw; ++i) {
    hop[nhop] = i * 8;
    walk(w[i], depth + 1, hop, nhop + 1, seen, visit, arg);
  }
}

struct SignalSearch {
  uint64_t startNs, endNs;
  bool found;
  Chain chain;
  hsa_agent_t agent;
  DiscoveryStats stats;
};

void visitForSignal(uint64_t, const uint64_t* w, const int* hop, int nhop, void* arg) {
  SignalSearch* s = (SignalSearch*)arg;
  if (s->found) return;
  s->stats.blocks++;
  const int nw = scanWords();
  for (int i = 0; i < nw; ++i) {
    if (searchSignal(w[i], s->agent, &s->startNs, &s->endNs, &s->stats)) {
      for (int h = 0; h < nhop; ++h) s->chain.hop[h] = hop[h];
      s->chain.nhop = nhop;
      s->chain.leaf = i * 8;
      s->found = true;
      return;
    }
  }
}

/* Locates the completion signal of a dispatch that has already run, reachable
 * from its event. Only the location is discovered; the timestamps themselves
 * come from ROCr on every harvest, so nothing depends on how the HIP runtime
 * caches or converts them. */
Chain discoverChain(void* event, int dev, hsa_agent_t* agent, DiscoveryStats* stats) {
  Chain none;
#if defined(__HIP_PLATFORM_AMD__)
  SignalSearch ss;
  ss.found = false;
  if (!agentForDevice(dev, &ss.agent)) return none;

  int hop[4];
  {
    std::set<uint64_t> seen;
    walk((uint64_t)event, 0, hop, 0, seen, visitForSignal, &ss);
  }
  *stats = ss.stats;
  if (!ss.found) return none;

  /* The chain is only trusted if walking it from the event reaches the same
   * signal ROCr just answered for. */
  uint64_t handle = 0, s = 0, e = 0;
  if (followChain(event, ss.chain, &handle) && dispatchTimeFromSignal(handle, ss.agent, &s, &e) &&
      s == ss.startNs && e == ss.endNs) {
    *agent = ss.agent;
    return ss.chain;
  }
#endif
  return none;
}

} // namespace

/* ---------- per-communicator state ---------- */

struct ncclKernelTimingCtx {
  std::mutex lock;
  Chain chain;
  hsa_agent_t agent;
  /* A chain is believed only once it has produced a plausible, fresh window for
   * a second dispatch; a location that merely happens to hold some signal would
   * not survive that. */
  enum { kDiscover, kConfirm, kReady } phase = kDiscover;
  uint64_t confirmStartNs = 0;
  bool disabled = false;

  /* Dispatches launched but not yet harvested, in launch order. A slot is
   * reserved before the launch and armed only once it has been issued. */
  std::vector<cudaEvent_t> inflightEvent;
  std::vector<ncclKernelTimingRecord> inflightRec;
  std::vector<bool> inflightArmed;
  std::vector<uint64_t> inflightArmNs;
  uint64_t inflightHead = 0, inflightTail = 0;

  /* Completed records awaiting a drain. */
  std::vector<ncclKernelTimingRecord> ring;
  uint64_t ringHead = 0, ringTail = 0;
  uint64_t dropped = 0;
  /* Why harvests were discarded, for the teardown summary. */
  uint64_t dropChain = 0, dropDead = 0, dropUntimed = 0, dropStale = 0, dropBusy = 0;

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
  ctx->inflightArmNs.resize(inflight, 0);
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
  if (ctx->dropped) {
    INFO(NCCL_INIT, "KERNEL_TIMING: %lu dispatches went untimed (chain %lu, dead %lu, untimed %lu, stale %lu, busy %lu)",
         ctx->dropped, ctx->dropChain, ctx->dropDead, ctx->dropUntimed, ctx->dropStale, ctx->dropBusy);
  }
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
void harvest(ncclKernelTimingCtx* ctx, int dev) {
  if (ctx->disabled) return;
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

    if (ctx->phase == ncclKernelTimingCtx::kDiscover) {
      DiscoveryStats st;
      ctx->chain = discoverChain(ev, dev, &ctx->agent, &st);
      if (!ctx->chain.valid()) {
        ctx->disabled = true;
        WARN("KERNEL_TIMING: could not locate the dispatch completion signal in this ROCm runtime; timing "
             "disabled (blocks %d, signals %d, stamped %d, timed %d)",
             st.blocks, st.signals, st.stamped, st.oracleOk);
      } else {
        ctx->phase = ncclKernelTimingCtx::kConfirm;
        INFO(NCCL_INIT, "KERNEL_TIMING: dispatch completion signal located (%d hops, leaf +%d)", ctx->chain.nhop,
             ctx->chain.leaf);
      }
    }

    if (!ctx->disabled) {
      ncclKernelTimingRecord rec = ctx->inflightRec[slot];
      uint64_t handle = 0;
      struct timespec ts;
      clock_gettime(CLOCK_BOOTTIME, &ts);
      uint64_t nowNs = (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
      /* A signal that outlived its dispatch and was handed to some later packet
       * would report a window that cannot belong to this launch, so anything
       * outside the interval between the launch call and now is discarded
       * rather than reported. */
      if (!followChain(ev, ctx->chain, &handle)) {
        ctx->dropped++, ctx->dropChain++;
      } else if (!signalLooksLive(handle)) {
        ctx->dropped++, ctx->dropDead++;
      } else if (!dispatchTimeFromSignal(handle, ctx->agent, &rec.startNs, &rec.endNs)) {
        ctx->dropped++, ctx->dropUntimed++;
      } else if (rec.startNs < ctx->inflightArmNs[slot] || rec.endNs > nowNs) {
        ctx->dropped++, ctx->dropStale++;
      } else if (ctx->phase == ncclKernelTimingCtx::kConfirm) {
        /* First use of the chain outside discovery. Repeating the window it was
         * discovered with would mean the location is some fixed signal rather
         * than this dispatch's. */
        if (ctx->confirmStartNs == 0) {
          ctx->confirmStartNs = rec.startNs;
        } else if (rec.startNs == ctx->confirmStartNs) {
          ctx->disabled = true;
          WARN("KERNEL_TIMING: dispatch timestamps do not track the dispatch; timing disabled");
        } else {
          ctx->phase = ncclKernelTimingCtx::kReady;
        }
        pushRecord(ctx, rec);
      } else {
        pushRecord(ctx, rec);
      }
    }
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
  /* Harvesting here would query an event on every launch, which makes the
   * runtime flush the queue and costs far more than the timing is worth. The
   * queue is drained instead, and a launch only gives up its timing if every
   * slot is still held. */
  if (rcclParamKernelTimingHarvestOnLaunch()) harvest(ctx, comm->cudaDev);
  if (ctx->disabled) return nullptr;

  /* Every dispatch is counted whether or not it ends up timed, so a record's
   * seq says which dispatch it was and a gap says one went unmeasured. */
  uint64_t seq = ctx->seq++;

  uint64_t inflight = ctx->inflightEvent.size();
  if (ctx->inflightHead - ctx->inflightTail == inflight) {
    /* Every slot is held, which means the queue is not being drained. Harvest
     * here to keep going: it costs an event query on this one launch and
     * reclaims the whole run of completed dispatches. */
    harvest(ctx, comm->cudaDev);
    if (ctx->disabled) return nullptr;
    if (ctx->inflightHead - ctx->inflightTail == inflight) {
      /* Genuinely nothing has completed; skip timing rather than stall. */
      ctx->dropped++, ctx->dropBusy++;
      return nullptr;
    }
  }

  uint64_t ticket = ctx->inflightHead++;
  *slotOut = ticket;
  uint64_t slot = ticket % inflight;
  ctx->inflightArmed[slot] = false;
  /* Taken before the dispatch is issued, so the kernel cannot legitimately
   * report a start earlier than this; that is what makes a signal reused by a
   * later packet detectable at harvest. */
  {
    struct timespec ts;
    clock_gettime(CLOCK_BOOTTIME, &ts);
    ctx->inflightArmNs[slot] = (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
  }
  ncclKernelTimingRecord& rec = ctx->inflightRec[slot];
  memset(&rec, 0, sizeof(rec));
  rec.seq = seq;
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
  harvest(ctx, comm->cudaDev);

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
