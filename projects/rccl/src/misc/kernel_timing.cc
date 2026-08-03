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
 * completion signal. HIP does not expose that signal, so it is reached by three
 * pointer hops through the runtime's own event objects. Where those hops go
 * differs between ROCm releases, and the version number cannot be used to tell
 * the releases apart, so the offsets are settled by measurement at startup: a
 * dispatch is issued between two clock readings and each known layout is tried
 * against it, keeping the one whose timestamps land inside that bracket. Only
 * the offsets are chosen this way. The timestamps themselves come from
 * hsa_amd_profiling_get_dispatch_time on every harvest, so nothing depends on
 * how HIP happens to cache or convert them. If no layout fits, timing stays off
 * for the process rather than reporting numbers we cannot vouch for.
 */

#include "kernel_timing.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <time.h>
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

namespace {

/* ---------- reading the dispatch timestamps out of an event ---------- */

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

/* The clock ROCr reports dispatch timestamps in, so that records can be read
 * against the host's view of when a launch happened. */
uint64_t bootNs() {
  struct timespec ts;
  clock_gettime(CLOCK_BOOTTIME, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* Where the event keeps the handle of the completion signal its dispatch was
 * issued with. A hipEvent_t is a hip::Event, which holds an amd::Event, which
 * holds the hardware event carrying the signal, so three byte offsets describe
 * the whole route.
 *
 * These come from compiling that traversal against the headers of every ROCm 7
 * and later tag and reading the displacements out of the generated code, which
 * tools/kernel-timing/dispatch-time/layout_offsets.sh does. Sixty-odd tags
 * produce three distinct answers. They are not ordered by version and cannot be
 * selected by it: TheRock 7.10 and 7.12 both report HIP 7.2.0 and disagree.
 * Which one is right here is therefore established by trying them. */
struct Layout {
  int event;    /* hip::Event  -> amd::Event*        */
  int amdEvent; /* amd::Event  -> hardware event*    */
  int signal;   /* hardware event -> signal handle   */
  const char* releases;
};

const Layout kLayouts[] = {
  {176, 280, 16, "ROCm 7.1-7.3, TheRock 7.9-7.11"},
  {168, 280, 16, "TheRock 7.12"},
  {88, 248, 16, "TheRock 7.13 and later, ROCm 10"},
};
constexpr int kNumLayouts = (int)(sizeof(kLayouts) / sizeof(kLayouts[0]));

/* The layout in use, once one has been established. A property of the runtime,
 * so it is settled once for the process rather than per communicator. */
const Layout* g_layout = nullptr;

bool followLayout(void* event, const Layout& l, uint64_t* handle) {
  uint64_t p = (uint64_t)event;
  if (!mappedRanges().readable(p + l.event, 8)) return false;
  p = *(uint64_t*)(p + l.event);
  if (!mappedRanges().readable(p + l.amdEvent, 8)) return false;
  p = *(uint64_t*)(p + l.amdEvent);
  if (!mappedRanges().readable(p + l.signal, 8)) return false;
  *handle = *(uint64_t*)(p + l.signal);
  return true;
}

#if defined(__HIP_PLATFORM_AMD__)
hsa_agent_t g_gpuAgents[16];
int g_nGpuAgents = 0;
std::once_flag g_gpuAgentsOnce;

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
  std::call_once(g_gpuAgentsOnce, []() { (void)hsa_iterate_agents(collectGpuAgent, nullptr); });

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

/* ROCr reads the whole amd_signal_t, so the handle has to be known-good before
 * it is passed in, not merely non-null. */
bool signalLooksLive(uint64_t handle) {
  if (!mappedRanges().readable(handle, sizeof(amd_signal_t))) return false;
  return *(const int64_t*)handle == AMD_SIGNAL_KIND_USER;
}

/* RCCL_KERNEL_TIMING_LAYOUT: unset tries each layout in turn, "none" skips the
 * probe entirely, and 1..N pins one -- which still has to pass the same check,
 * so a pin that has gone stale fails the way a bad probe does. Read here rather
 * than declared as a parameter because those carry integers only. */
enum {
  kLayoutProbeAll = -1,
  kLayoutOff = -2
};

int requestedLayout() {
  const char* s = ncclGetEnv("RCCL_KERNEL_TIMING_LAYOUT");
  if (s == nullptr || s[0] == '\0') return kLayoutProbeAll;
  if (strcmp(s, "none") == 0) return kLayoutOff;
  char* end = nullptr;
  long v = strtol(s, &end, 10);
  if (end != s && *end == '\0' && v >= 1 && v <= kNumLayouts) return (int)(v - 1);
  WARN("KERNEL_TIMING: RCCL_KERNEL_TIMING_LAYOUT=%s is not \"none\" or 1-%d; probing instead", s, kNumLayouts);
  return kLayoutProbeAll;
}

/* Settles which layout this runtime uses, on a dispatch issued here so that
 * nothing depends on a collective having run yet. A candidate has to survive
 * the mapping checks, be a signal ROCr recognises, and report a window inside
 * the bracket taken around the dispatch -- offsets that address the wrong words
 * do not produce a window a few microseconds wide in the right place. */
bool resolveLayout(int dev, hsa_agent_t agent) {
  const int want = requestedLayout();
  if (want == kLayoutOff) {
    INFO(NCCL_INIT, "KERNEL_TIMING: layout probe disabled by RCCL_KERNEL_TIMING_LAYOUT=none");
    return false;
  }

  int previous = -1;
  if (cudaGetDevice(&previous) != cudaSuccess) return false;
  if (cudaSetDevice(dev) != cudaSuccess) return false;

  cudaStream_t stream = nullptr;
  cudaEvent_t event = nullptr;
  uint64_t before = 0, after = 0;
  bool dispatched = false;

  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess &&
      cudaEventCreateWithFlags(&event, cudaEventDefault | hipEventDisableSystemFence) == cudaSuccess) {
    before = bootNs();
    if (cudaEventRecord(event, stream) == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess) {
      after = bootNs();
      dispatched = true;
    }
  }

  if (dispatched) {
    for (int i = 0; i < kNumLayouts; i++) {
      if (want != kLayoutProbeAll && want != i) continue;
      const Layout& l = kLayouts[i];
      uint64_t handle = 0, start = 0, end = 0;
      if (followLayout(event, l, &handle) && signalLooksLive(handle) &&
          dispatchTimeFromSignal(handle, agent, &start, &end) && start >= before && end <= after) {
        g_layout = &l;
        INFO(NCCL_INIT, "KERNEL_TIMING: layout %d (+%d, +%d, +%d), as built for %s", i + 1, l.event, l.amdEvent,
             l.signal, l.releases);
        break;
      }
    }
  }

  if (event != nullptr) (void)cudaEventDestroy(event);
  if (stream != nullptr) (void)cudaStreamDestroy(stream);
  (void)cudaSetDevice(previous);

  if (g_layout == nullptr) {
    WARN("KERNEL_TIMING: no known event layout fits this ROCm (tried %d), timing disabled; "
         "tools/kernel-timing/dispatch-time/layout_offsets.sh derives the offsets for a new release",
         want == kLayoutProbeAll ? kNumLayouts : 1);
  }
  return g_layout != nullptr;
}
#else
struct hsa_agent_t {
  uint64_t handle;
};
bool agentForDevice(int, hsa_agent_t*) {
  return false;
}
bool dispatchTimeFromSignal(uint64_t, hsa_agent_t, uint64_t*, uint64_t*) {
  return false;
}
bool signalLooksLive(uint64_t) {
  return false;
}
bool resolveLayout(int, hsa_agent_t) {
  return false;
}
#endif

} // namespace

/* ---------- per-communicator state ---------- */

struct ncclKernelTimingCtx {
  /* Launchers only touch the ticket counters and their uniquely owned slot.
   * Drains are serialized separately, so timestamp extraction never holds a
   * lock needed by a launch. */
  std::mutex drainLock;
  /* Which agent ROCr should convert this device's ticks with. The layout is a
   * property of the runtime and is held once for the process; the agent is per
   * device and so lives here. */
  hsa_agent_t agent;

  enum SlotState : uint64_t {
    kFree = 0,
    kReserved = 1,
    kArmed = 2,
    kCancelled = 3
  };
  struct Slot {
    std::atomic<uint64_t> state;
    cudaEvent_t event;
    ncclKernelTimingRecord rec;
    Slot() : state(0), event(nullptr) {}
  };
  Slot* inflightSlot = nullptr;
  uint64_t inflight = 0;
  std::atomic<uint64_t> inflightHead{0};
  std::atomic<uint64_t> inflightTail{0};

  /* Completed records awaiting a drain. */
  std::vector<ncclKernelTimingRecord> ring;
  uint64_t ringHead = 0, ringTail = 0;
  std::atomic<uint64_t> dropped{0};
  /* Why harvests were discarded, for the teardown summary. */
  std::atomic<uint64_t> dropUnread{0}, dropDead{0}, dropUntimed{0}, dropQuery{0}, dropBusy{0}, dropOverflow{0},
    dropCancelled{0};

  std::atomic<uint64_t> seq{0};
};

constexpr uint64_t kSlotStateBits = 2;
constexpr uint64_t kSlotStateMask = (1ull << kSlotStateBits) - 1;

uint64_t slotWord(uint64_t ticket, ncclKernelTimingCtx::SlotState state) {
  return (ticket << kSlotStateBits) | (uint64_t)state;
}

uint64_t slotTicket(uint64_t word) {
  return word >> kSlotStateBits;
}

ncclKernelTimingCtx::SlotState slotState(uint64_t word) {
  return (ncclKernelTimingCtx::SlotState)(word & kSlotStateMask);
}

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

  hsa_agent_t agent;
  if (!agentForDevice(comm->cudaDev, &agent)) {
    WARN("KERNEL_TIMING: no HSA agent matches device %d, kernel timing disabled", comm->cudaDev);
    return ncclSuccess;
  }

  /* One probe serves the process: every communicator here is talking to the
   * same HIP runtime, and it is the runtime that decides the layout. */
  static std::once_flag probed;
  std::call_once(probed, [&]() { (void)resolveLayout(comm->cudaDev, agent); });
  if (g_layout == nullptr) return ncclSuccess;

  int inflight = (int)rcclParamKernelTimingInflight();
  int capacity = (int)rcclParamKernelTimingCapacity();
  if (inflight < 8) inflight = 8;
  if (capacity < inflight) capacity = inflight;

  ncclKernelTimingCtx* ctx = new ncclKernelTimingCtx();
  ctx->agent = agent;
  ctx->inflight = (uint64_t)inflight;
  ctx->inflightSlot = new ncclKernelTimingCtx::Slot[inflight];
  ctx->ring.resize(capacity);
  for (int i = 0; i < inflight; i++) {
    ctx->inflightSlot[i].state.store(slotWord((uint64_t)i, ncclKernelTimingCtx::kFree), std::memory_order_relaxed);
    /* Timing must stay enabled on these events; they are the timestamp source.
     * hipEventDisableSystemFence: this event is never inspected for data-visibility
     * (nothing outside this file ever sees it), only polled for its HSA dispatch
     * timestamps, so it doesn't need the runtime's implicit system-scope release
     * fence. Without this flag, binding the event to the dispatch upgrades the AQL
     * packet's acquire/release fence scope from AGENT to SYSTEM (see ROCclr's
     * ihipLaunchKernelCommand / addSystemScope_), which is a real GPU-side stall on
     * every timed dispatch. */
    if (cudaEventCreateWithFlags(&ctx->inflightSlot[i].event,
                                 cudaEventDefault | hipEventDisableSystemFence) != cudaSuccess) {
      for (int j = 0; j < i; j++) (void)cudaEventDestroy(ctx->inflightSlot[j].event);
      delete[] ctx->inflightSlot;
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
  uint64_t dropped = ctx->dropped.load(std::memory_order_relaxed);
  if (dropped != 0) {
    INFO(NCCL_INIT,
         "KERNEL_TIMING: %lu dispatches went untimed "
         "(unread %lu, dead %lu, untimed %lu, query %lu, busy %lu, overflow %lu, cancelled %lu)",
         dropped, ctx->dropUnread.load(std::memory_order_relaxed), ctx->dropDead.load(std::memory_order_relaxed),
         ctx->dropUntimed.load(std::memory_order_relaxed), ctx->dropQuery.load(std::memory_order_relaxed),
         ctx->dropBusy.load(std::memory_order_relaxed), ctx->dropOverflow.load(std::memory_order_relaxed),
         ctx->dropCancelled.load(std::memory_order_relaxed));
  }
  for (uint64_t i = 0; i < ctx->inflight; i++) {
    if (ctx->inflightSlot[i].event) (void)cudaEventDestroy(ctx->inflightSlot[i].event);
  }
  delete[] ctx->inflightSlot;
  delete ctx;
  return ncclSuccess;
}

namespace {

void pushRecord(ncclKernelTimingCtx* ctx, const ncclKernelTimingRecord& rec) {
  uint64_t capacity = ctx->ring.size();
  if (ctx->ringHead - ctx->ringTail == capacity) {
    ctx->ringTail++; /* overwrite oldest */
    ctx->dropped.fetch_add(1, std::memory_order_relaxed);
    ctx->dropOverflow.fetch_add(1, std::memory_order_relaxed);
  }
  ctx->ring[ctx->ringHead % capacity] = rec;
  ctx->ringHead++;
}

/* Moves completed dispatches into the output ring. Only a drain calls this:
 * event queries and ROCr timestamp conversion must never run on a launch
 * thread. A reserved slot is still between Begin and Commit, so stop rather
 * than recycling its event underneath the launcher. */
void harvest(ncclKernelTimingCtx* ctx) {
  uint64_t ticket = ctx->inflightTail.load(std::memory_order_relaxed);
  while (ticket != ctx->inflightHead.load(std::memory_order_acquire)) {
    ncclKernelTimingCtx::Slot& slot = ctx->inflightSlot[ticket % ctx->inflight];
    uint64_t word = slot.state.load(std::memory_order_acquire);
    if (slotTicket(word) != ticket || slotState(word) == ncclKernelTimingCtx::kFree ||
        slotState(word) == ncclKernelTimingCtx::kReserved)
      break;

    if (slotState(word) == ncclKernelTimingCtx::kCancelled) {
      ctx->dropped.fetch_add(1, std::memory_order_relaxed);
      ctx->dropCancelled.fetch_add(1, std::memory_order_relaxed);
    } else {
      cudaError_t query = cudaEventQuery(slot.event);
      if (query == cudaErrorNotReady) break;

      ncclKernelTimingRecord rec = slot.rec;
      if (query != cudaSuccess) {
        ctx->dropped.fetch_add(1, std::memory_order_relaxed);
        ctx->dropQuery.fetch_add(1, std::memory_order_relaxed);
      } else {
        uint64_t handle = 0;
        if (!followLayout(slot.event, *g_layout, &handle)) {
          ctx->dropped.fetch_add(1, std::memory_order_relaxed);
          ctx->dropUnread.fetch_add(1, std::memory_order_relaxed);
        } else if (!signalLooksLive(handle)) {
          ctx->dropped.fetch_add(1, std::memory_order_relaxed);
          ctx->dropDead.fetch_add(1, std::memory_order_relaxed);
        } else if (!dispatchTimeFromSignal(handle, ctx->agent, &rec.startNs, &rec.endNs)) {
          ctx->dropped.fetch_add(1, std::memory_order_relaxed);
          ctx->dropUntimed.fetch_add(1, std::memory_order_relaxed);
        } else {
          pushRecord(ctx, rec);
        }
      }
    }

    ticket++;
    slot.state.store(slotWord(ticket + ctx->inflight - 1, ncclKernelTimingCtx::kFree), std::memory_order_release);
    ctx->inflightTail.store(ticket, std::memory_order_release);
  }
}

} // namespace

namespace {

/* Claims a ticket without querying the GPU. A full queue drops timing for this
 * dispatch immediately; only the consumer is allowed to harvest. */
cudaEvent_t reserve(ncclKernelTimingCtx* ctx, struct ncclComm* comm, uint64_t* ticketOut,
                    ncclKernelTimingRecord** recOut) {
  /* Every dispatch is counted whether or not it ends up timed, so a record's
   * seq says which dispatch it was and a gap says one went unmeasured. */
  uint64_t seq = ctx->seq.fetch_add(1, std::memory_order_relaxed);

  uint64_t ticket = ctx->inflightHead.load(std::memory_order_relaxed);
  for (;;) {
    uint64_t tail = ctx->inflightTail.load(std::memory_order_acquire);
    if (ticket - tail >= ctx->inflight) {
      ctx->dropped.fetch_add(1, std::memory_order_relaxed);
      ctx->dropBusy.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    if (ctx->inflightHead.compare_exchange_weak(ticket, ticket + 1, std::memory_order_acq_rel,
                                                std::memory_order_relaxed))
      break;
  }

  ncclKernelTimingCtx::Slot& slot = ctx->inflightSlot[ticket % ctx->inflight];
  uint64_t expected = slotWord(ticket, ncclKernelTimingCtx::kFree);
  while (!slot.state.compare_exchange_weak(expected, slotWord(ticket, ncclKernelTimingCtx::kReserved),
                                           std::memory_order_acq_rel, std::memory_order_relaxed))
    expected = slotWord(ticket, ncclKernelTimingCtx::kFree);

  ncclKernelTimingRecord& rec = slot.rec;
  rec.seq = seq;
  rec.commHash = comm->commHash;
  rec.rank = comm->rank;
  *ticketOut = ticket;
  *recOut = &rec;
  return slot.event;
}

} // namespace

cudaEvent_t ncclKernelTimingBeginDispatch(struct ncclComm* comm, cudaStream_t stream, uint64_t* slotOut) {
  ncclKernelTimingCtx* ctx = comm->kernelTiming;
  if (ctx == nullptr) return nullptr;

  /* Captured ext launches accept an event, but replay never populates it. */
  cudaStreamCaptureStatus captureStatus;
  if (cudaStreamIsCapturing(stream, &captureStatus) != cudaSuccess || captureStatus != cudaStreamCaptureStatusNone)
    return nullptr;

  ncclKernelTimingRecord* rec = nullptr;
  return reserve(ctx, comm, slotOut, &rec);
}

cudaEvent_t ncclKernelTimingBeginLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, uint64_t* slotOut) {
  ncclKernelTimingCtx* ctx = comm->kernelTiming;
  if (ctx == nullptr) return nullptr;
  /* Graph-captured launches accept the event but replay never populates it. */
  if (plan->persistent) return nullptr;

  ncclKernelTimingRecord* rec = nullptr;
  return reserve(ctx, comm, slotOut, &rec);
}

/* Both Commit variants below write the describing fields on a still-kReserved
 * slot, then CAS it to kArmed -- the same ordering reserve() used to write
 * seq/commHash/rank, just deferred past the launch call. harvest() will not
 * touch a kReserved slot (see its comment), so this is safe without any extra
 * synchronization. */

void ncclKernelTimingCommitLaunch(struct ncclComm* comm, uint64_t ticket, struct ncclKernelPlan* plan,
                                  uint32_t nChannels, uint32_t nThreads) {
  ncclKernelTimingCtx* ctx = comm->kernelTiming;
  if (ctx == nullptr) return;
  ncclKernelTimingCtx::Slot& slot = ctx->inflightSlot[ticket % ctx->inflight];

  ncclKernelTimingRecord& rec = slot.rec;
  rec.nChannels = nChannels;
  rec.nThreads = nThreads;
  rec.nColls = plan->collOpCount;
  struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
  if (ct == nullptr) {
    rec.func = 0;
    rec.datatype = 0;
    rec.count = 0;
  } else {
    rec.func = (uint32_t)ct->func;
    rec.datatype = (uint32_t)ct->datatype;
    rec.count = (uint64_t)ct->count;
  }

  uint64_t expected = slotWord(ticket, ncclKernelTimingCtx::kReserved);
  (void)slot.state.compare_exchange_strong(expected, slotWord(ticket, ncclKernelTimingCtx::kArmed),
                                           std::memory_order_release, std::memory_order_relaxed);
}

void ncclKernelTimingCommitDispatch(struct ncclComm* comm, uint64_t ticket, uint32_t func, uint32_t datatype,
                                    uint64_t count, uint32_t nChannels, uint32_t nThreads) {
  ncclKernelTimingCtx* ctx = comm->kernelTiming;
  if (ctx == nullptr) return;
  ncclKernelTimingCtx::Slot& slot = ctx->inflightSlot[ticket % ctx->inflight];

  ncclKernelTimingRecord& rec = slot.rec;
  rec.func = func;
  rec.datatype = datatype;
  rec.count = count;
  rec.nChannels = nChannels;
  rec.nThreads = nThreads;
  rec.nColls = 1;

  uint64_t expected = slotWord(ticket, ncclKernelTimingCtx::kReserved);
  (void)slot.state.compare_exchange_strong(expected, slotWord(ticket, ncclKernelTimingCtx::kArmed),
                                           std::memory_order_release, std::memory_order_relaxed);
}

void ncclKernelTimingCancelLaunch(struct ncclComm* comm, uint64_t ticket) {
  ncclKernelTimingCtx* ctx = comm->kernelTiming;
  if (ctx == nullptr) return;
  ncclKernelTimingCtx::Slot& slot = ctx->inflightSlot[ticket % ctx->inflight];
  uint64_t expected = slotWord(ticket, ncclKernelTimingCtx::kReserved);
  (void)slot.state.compare_exchange_strong(expected, slotWord(ticket, ncclKernelTimingCtx::kCancelled),
                                           std::memory_order_release, std::memory_order_relaxed);
}

NCCL_API(ncclResult_t, ncclKernelTimingDrain, ncclComm_t comm, ncclKernelTimingRecord* out, int max, int* got,
         uint64_t* dropped);
ncclResult_t ncclKernelTimingDrain(ncclComm_t comm, ncclKernelTimingRecord* out, int max, int* got, uint64_t* dropped) {
  if (comm == nullptr || out == nullptr || got == nullptr || max < 0) return ncclInvalidArgument;
  ncclKernelTimingCtx* ctx = comm->kernelTiming;
  if (ctx == nullptr) return ncclInvalidUsage;

  std::lock_guard<std::mutex> guard(ctx->drainLock);
  harvest(ctx);

  uint64_t capacity = ctx->ring.size();
  int n = 0;
  while (n < max && ctx->ringTail != ctx->ringHead) {
    out[n++] = ctx->ring[ctx->ringTail % capacity];
    ctx->ringTail++;
  }
  *got = n;
  if (dropped) *dropped = ctx->dropped.load(std::memory_order_relaxed);
  return ncclSuccess;
}
