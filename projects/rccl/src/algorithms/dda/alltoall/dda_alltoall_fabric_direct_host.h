/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host side of the DDA fabric AllToAll ZERO-COPY direct path
 * (kernels: alltoall_dda_fabric_direct.h, alltoall_dda_fabric_direct_hybrid.h). Included at the
 * end of dda_alltoall_fabric.cu so it shares that translation unit's device-linker build rule.
 *
 * Owns a small fabric-exported flag buffer per rank, created together with the push buffer by the
 * first AllToAll that can use either (ncclDdaA2aLazyInit). Two ways in:
 *
 * Window: the receive buffer lies in a symmetric window (ncclCommWindowRegister with
 * NCCL_WIN_COLL_SYMMETRIC, LSA registration): window registration is collective, and the symmetric
 * contract has every rank pass its buffer at the same window offset, so whether a call is windowed is
 * the same on every rank and each peer's recvbuff comes from this rank's LSA mapping. Every pair is
 * written directly (window kernel).
 *
 * Implicit (RCCL_DDA_A2A_IMPLICIT=1 on every rank, opt-in): any other out-of-place call in the Direct
 * window launches the hybrid kernel on every rank. The receive buffer's mapping in each peer is
 * per-rank state, so it only changes what the RECEIVER posts per pair at run time (ADDR cell: its
 * recvbuff in the sender's VA, or "staged": the sender stages through the receiver's Direct-owned
 * staging buffer in rounds, the receiver copies out). The mapping comes from:
 *   - an ncclCommRegister record of recvbuff that reached every peer (NCCL IPC registration, cached
 *     by the record); a record that did not reach every peer and has no legacy (hipIpc) import is
 *     handed to the implicit cache instead (a second legacy import of one allocation breaks when the
 *     first is closed);
 *   - otherwise the implicit cache: the receive allocation is IPC-registered with every peer on first
 *     sight (cuMem POSIX fd / FABRIC for VMM memory, legacy hipIpc for hipMalloc memory; private and
 *     asynchronous, p2p.cc ncclIpcImplicitReg*), keyed by (allocation base, size, HIP buffer id),
 *     at most RCCL_DDA_A2A_IMPLICIT_CACHE allocations (LRU; an entry used by a graph capture is
 *     pinned until that graph is destroyed; when every entry is pinned a new allocation is staged
 *     instead of growing the cache). hipMalloc allocations (legacy imports, which can only be closed
 *     at comm destroy) are imported up to RCCL_DDA_A2A_IMPLICIT_LEGACY_BYTES per comm, staged beyond.
 *     Failed imports are cached with the entry (no per-call retry).
 * Registration state therefore never has to agree across ranks, and every rank launches the same
 * kernel for every call in the window.
 *
 * Env (all RCCL_*):
 *   DDA_A2A_DIRECT              0 disables (default 1; no flag buffer)
 *   DDA_A2A_IMPLICIT            1 enables the implicit path (default 0; +64 MiB staging buffer per
 *                               rank; off on every rank unless every rank sets it)
 *   DDA_A2A_IMPLICIT_CACHE      implicitly registered allocations kept (default 8)
 *   DDA_A2A_IMPLICIT_LEGACY_BYTES  bytes of hipMalloc allocations imported per comm (default 4 GiB;
 *                               0: never import hipMalloc memory; < 0: no limit)
 * Selection window: rcclArchThresholds::a2aDirectPerPeerCut / a2aDirectMin / a2aDirectMax
 * (rcclSelectAlltoAllFabric). Tuning builds (RCCL_DDA_A2A_TUNING, dda_alltoall.h) also read the
 * geometry values DDA_A2A_DIRECT_BPP (0 = rank-count table), _SELF_BLOCKS (12), _THREADS (256, max
 * 512), _UNROLL (0 = size table; 8 or 16), and DDA_A2A_IMPLICIT_DEBUG (test controls, below).
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/alltoall/alltoall_dda_fabric_direct.h"
#include "algorithms/dda/alltoall/alltoall_dda_fabric_direct_hybrid.h"
#include "algorithms/dda/alltoall/dda_alltoall_implicit_policy.h"
#include "algorithms/dda/alltoall/dda_alltoall_lazy.h"
#include "alloc.h"
#include "dev_runtime.h"
#include "p2p.h"
#include "param.h"
#include "register.h"
#include "register_inline.h"
#include "strongstream.h"

#include <sched.h>
#include <memory>
#include <vector>

RCCL_PARAM(DdaA2aDirect, "DDA_A2A_DIRECT", 1);
// BPP / THREADS (tuning builds) must be the same on every rank: they partition each pair's chunk
// (and staged rounds) between the sender's and the receiver's blocks.
RCCL_A2A_KNOB(DdaA2aDirectBpp, "DDA_A2A_DIRECT_BPP", 0);  // 0 = ddaA2aDirectBppAuto(nRanks)
RCCL_A2A_KNOB(DdaA2aDirectSelfBlocks, "DDA_A2A_DIRECT_SELF_BLOCKS", 12);
RCCL_A2A_KNOB(DdaA2aDirectThreads, "DDA_A2A_DIRECT_THREADS", 256);
RCCL_A2A_KNOB(DdaA2aDirectUnroll, "DDA_A2A_DIRECT_UNROLL", 0);  // 0 = size table
// Implicit zero-copy for receive buffers outside a symmetric window (opt-in): registration cache of
// unregistered receive allocations (dda_alltoall_implicit_policy.h, ncclIpcImplicitReg* in p2p.cc).
RCCL_PARAM(DdaA2aImplicit, "DDA_A2A_IMPLICIT", 0);
RCCL_PARAM(DdaA2aImplicitCache, "DDA_A2A_IMPLICIT_CACHE", 8);
RCCL_PARAM(DdaA2aImplicitLegacyBytes, "DDA_A2A_IMPLICIT_LEGACY_BYTES", 4ll << 30);
// Tests only: 1 ignore the buffer id (stale maps, negative control), 2/4/32 force staged pairs, 8 do
// not wait for imports at registration, 16 no closes before destroy.
RCCL_A2A_KNOB(DdaA2aImplicitDebug, "DDA_A2A_IMPLICIT_DEBUG", 0);

namespace {

// One implicitly registered recvbuff allocation (private, asynchronous registration; see p2p.cc).
struct DdaA2aImplicitEntry {
  uintptr_t base = 0;
  size_t size = 0;
  unsigned long long bufId = 0;
  ncclIpcImplicitReg* reg = nullptr;
  int pending = 0;  // peer imports still in flight (their pairs are staged meanwhile)
  int pollCalls = 0;
  int64_t t0 = 0;
  uintptr_t peerAddr[dda::common::kDdaA2aDirectMaxRanks] = {};  // allocation base in peer r's VA (0: not mapped)
  uint64_t lastUse = 0;
  uint64_t uid = 0;  // identity for graph unpins (base/id can be reused by a new entry)
  int pins = 0;      // graph captures that use this entry and are still alive (not evictable, not closable)
  bool legacy = false;
  // A legacy import still in flight when the first-sight wait gave up: the peer may still import the
  // handle, which hangs inside ROCr if the allocation is freed first. Never evicted or closed (the
  // closes would wait for that import); its host state is freed at Fini only.
  bool stuck = false;
};

// An entry on its way out: the peers' imports are closed once `ev` (recorded on the stream of
// the call that retired it, i.e. after every earlier kernel that advertised the mapping) completed.
struct DdaA2aImplicitRetiring {
  ncclIpcImplicitReg* reg = nullptr;
  cudaEvent_t ev = nullptr;
};

struct DdaA2aDirectState {
  std::vector<void*> peerMaps;  // imported peer flag buffers (self slot = base), freed at Fini
  void* base = nullptr;
  char** peerFlagsDev = nullptr;
  // ---- implicit (only with the agreed dda_a2a_lazy::kImplicit part) ----
  std::vector<void*> scratchMaps;  // imported peer staging buffers (self slot = scratch)
  void* scratch = nullptr;         // this rank's staging buffer: [nRanks slots of slotBytes]
  char** scratchPeerDev = nullptr;
  size_t scratchBytes = 0;  // agreed (same on every rank)
  std::vector<DdaA2aImplicitEntry*> cache;
  std::vector<DdaA2aImplicitRetiring> retiring;
  // Retired but kept mapped: graph-pinned entries (until their graphs are destroyed), entries
  // retired while capturing (until the next uncaptured call), and every legacy (hipIpc) entry until
  // comm destroy -- closing a legacy import while a newer import of the same allocation exists in
  // that peer breaks the newer one (measured: a11stress free+realloc fails with closes, passes
  // without). Legacy entries are therefore bounded by refusing new ones (legacyBytes), not by
  // eviction. A zombie whose (base, size, buffer id) shows up again is revived instead of
  // re-registered.
  std::vector<DdaA2aImplicitEntry*> zombies;
  std::shared_ptr<dda_a2a_implicit::UnpinQueue> unpinQ = std::make_shared<dda_a2a_implicit::UnpinQueue>();
  size_t legacyBytes = 0;  // open legacy imports (cache + zombies), never decreases before destroy
  uint64_t nextUid = 0;
  bool legacyBudgetNoted = false, capFullNoted = false;
  // ncclCommRegister records whose IPC registration did not reach every peer (e.g. a peer on another
  // host and an allocation without FABRIC export): handled by the implicit cache from then on, so the
  // failing registration is not retried every call. Key = (record, range); a stale key only routes a
  // later record at the same address through the implicit cache, which is correct for any buffer.
  struct RegIncomplete {
    const void* reg;
    uintptr_t beg, end;
  };
  std::vector<RegIncomplete> regIncomplete;
  uint64_t clock = 0;
  uint64_t nRegister = 0, nEvict = 0, nInvalidate = 0, nClosed = 0, nRefusedLegacy = 0, nRefusedFull = 0;
};

static int ddaA2aDirectClamp(int64_t v, int lo, int hi, int dflt) {
  return (v >= lo && v <= hi) ? (int)v : dflt;
}

// Blocks per peer row before the residency cap. Small comms need more blocks per row to fill the
// fabric: P05 (R0 OOP busBW, mt6t, 16M/64M/256M) 2r cross-chip (GPUs 0,2) 12 -> 48: 109/131/129 ->
// 270/455/377 (sibling 0,1: within -6%..+0%); 4r 12 -> 20: 0-3 240/335/340 -> 319/450/517 (0-3 dips to
// 170-280 at 32-64M with 24/28), 0,2,4,6 245/340/358 -> 324/505/550. 8r keeps 12 (A08), 16r is capped
// to 5 by residency anyway. n = 3 is interpolated (unmeasured).
static int ddaA2aDirectBppAuto(int n) { return n <= 2 ? 48 : n <= 4 ? 20 : 12; }

// Grid geometry (residency, see the kernel header): the grid must fit the device at once, so it is
// capped at comm->ddaFabricMaxBlocks (comm-wide min CU count, which RCCL_DDA_FABRIC_MAXBLOCKS
// lowers). All inputs are comm-uniform, so every rank launches the same grid and the same B (the
// DONE[j] / stride partition requires B to match on both sides of a pair).
// 8 ranks x 96 CUs: 12 + 7 x 12 = 96; 16 ranks: 12 + 15 x 5 = 87; 4 ranks 12 + 3 x 20 = 72;
// 2 ranks 12 + 48 = 60.
static bool ddaA2aDirectGeom(const ncclComm* comm, int* bppOut, int* selfOut) {
  const int n = comm->nRanks;
  const int maxBlk = comm->ddaFabricMaxBlocks;
  int sb = ddaA2aDirectClamp(rcclParamDdaA2aDirectSelfBlocks(), 1, 1024, 12);
  int bpp = ddaA2aDirectClamp(rcclParamDdaA2aDirectBpp(), 1, dda::common::kDdaA2aDirectMaxBpp, ddaA2aDirectBppAuto(n));
  if (n < 2 || maxBlk < n) return false;  // not even one block per peer + one self block
  if (sb > maxBlk - (n - 1)) sb = maxBlk - (n - 1);
  const int cap = (maxBlk - sb) / (n - 1);
  if (bpp > cap) bpp = cap;
  if (bppOut) *bppOut = bpp;
  if (selfOut) *selfOut = sb;
  return bpp >= 1 && sb >= 1;
}

// Every peer's recvbuff in this rank's VA, from the symmetric window that holds recvbuff (same
// offset on every rank). False if any peer has no LSA mapping or a misaligned one.
static bool ddaA2aDirectPeerRecv(ncclComm* comm, void* recvbuff, uintptr_t* out) {
  struct ncclDevrWindow* win = nullptr;
  if (ncclDevrFindWindow(comm, recvbuff, &win) != ncclSuccess || win == nullptr) return false;
  const size_t off = (uintptr_t)recvbuff - (uintptr_t)win->userPtr;
  for (int r = 0; r < comm->nRanks; ++r) {
    out[r] = 0;
    if (r == comm->rank) continue;
    int lsa = -1;
    void* ptr = nullptr;
    if (ncclDevrWorldToLsaRank(comm, r, &lsa) != ncclSuccess || lsa < 0 ||
        ncclDevrGetLsaRankPtr(comm, win, off, lsa, &ptr) != ncclSuccess || ptr == nullptr ||
        ((uintptr_t)ptr & 15) != 0)
      return false;
    out[r] = (uintptr_t)ptr;
  }
  return true;
}

// ---- implicit registration cache ----

static DdaA2aImplicitEntry* ddaA2aImplicitRegister(ncclComm* comm, uintptr_t base, size_t size,
                                                   unsigned long long id) {
  auto* e = new (std::nothrow) DdaA2aImplicitEntry();
  if (e == nullptr) return nullptr;
  e->base = base;
  e->size = size;
  e->bufId = id;
  e->t0 = clockNano();
  int peers[dda::common::kDdaA2aDirectMaxRanks];
  int nPeers = 0;
  for (int r = 0; r < comm->nRanks; ++r) {
    if (r != comm->rank) peers[nPeers++] = r;
  }
  // Exports now (before any older import at this VA is closed: HIP's legacy export of a
  // reallocation fails once a stale import was closed first); peers import asynchronously.
  if (ncclIpcImplicitRegStart(comm, (void*)base, size, peers, nPeers, &e->reg) != ncclSuccess) e->reg = nullptr;
  (void)cudaGetLastError();
  if (e->reg != nullptr) (void)ncclIpcImplicitRegPoll(comm, e->reg, e->peerAddr, &e->pending);
  // Wait for the peers' imports here, inside the call that first sees the buffer:
  // - a legacy (hipIpc) handle is only valid while the allocation lives (a peer importing it after
  //   the user freed it hangs inside ROCr's IPC import), and the caller owns the buffer during the call;
  // - left asynchronous, imports (~1 ms each, serialized per peer proxy: ~6 ms, up to ~60 ms) land
  //   after the host has enqueued many more calls, which then stage those pairs (measured).
  // The wait is bounded: imports never wait for the importer's device, and import closes never run
  // on a proxy thread during the run (helper thread for VMM, destroy-time for legacy).
  const bool legacy = ncclIpcImplicitIsLegacy(e->reg) != 0;
  e->legacy = legacy;
  // Coupling with RCCL_DDA_A2A_TIMEOUT: the peers' kernels of this call already wait for this rank's
  // ENTRY meanwhile, so a timeout shorter than this wait can trip them (imports normally take ms).
  const bool wait = !(rcclParamDdaA2aImplicitDebug() & 8);
  const uint64_t tw = clockNano();
  while (wait && e->pending > 0) {
    sched_yield();
    (void)ncclIpcImplicitRegPoll(comm, e->reg, e->peerAddr, &e->pending);
    if (clockNano() - tw > 30ull * 1000 * 1000 * 1000) {
      // Pairs still importing are staged. A legacy import in flight must never meet a freed
      // allocation: the entry stays pinned for the communicator's lifetime.
      e->stuck = legacy;
      WARN("DDA A2A implicit: rank %d: %d peer imports of %p not done after 30 s; staging those pairs%s", comm->rank,
           e->pending, (void*)base,
           legacy ? "; this hipMalloc allocation must not be freed before the communicator is destroyed" : "");
      break;
    }
  }
  if (e->stuck) ++e->pins;
  INFO(NCCL_REG, "DDA A2A implicit: rank %d registered recv allocation %p size %zu id %llu (%s) in %.3f ms, %d imports in flight",
       comm->rank, (void*)base, size, id, legacy ? "legacy" : "vmm", (clockNano() - e->t0) / 1e6, e->pending);
  return e;
}

// Non-blocking progress of retiring entries (dda_a2a_implicit::progressRetiring: never while the
// call's stream is capturing -- hipEventQuery during a capture invalidates it -- and only a
// successful query counts as complete).
static void ddaA2aImplicitProgress(ncclComm* comm, DdaA2aDirectState* st, bool capturing) {
  dda_a2a_implicit::progressRetiring(
    st->retiring, capturing,
    [](DdaA2aImplicitRetiring& r) {
      const cudaError_t q = cudaEventQuery(r.ev);
      (void)cudaGetLastError();  // NotReady (or a capture error) is recorded as the last error
      if (q != cudaSuccess) return false;
      CUDACHECKIGNORE(cudaEventDestroy(r.ev));
      r.ev = nullptr;
      return true;
    },
    [comm, st](DdaA2aImplicitRetiring& r) {
      int pending = 0;
      (void)ncclIpcImplicitDeregPoll(comm, r.reg, &pending, /*asyncClose=*/true);
      if (pending != 0) return false;
      ncclIpcImplicitFree(r.reg);
      st->nClosed++;
      return true;
    });
}

static void ddaA2aImplicitRetire(DdaA2aDirectState* st, DdaA2aImplicitEntry* e, cudaStream_t stream,
                                 bool capturing) {
  if (e->reg == nullptr) {
    delete e;
    return;
  }
  cudaEvent_t ev = nullptr;
  if (e->pins > 0 || capturing || e->legacy || (rcclParamDdaA2aImplicitDebug() & 16) ||
      cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess || cudaEventRecord(ev, stream) != cudaSuccess) {
    (void)cudaGetLastError();
    if (ev != nullptr) CUDACHECKIGNORE(cudaEventDestroy(ev));
    st->zombies.push_back(e);
    return;
  }
  st->retiring.push_back({e->reg, ev});
  delete e;
}

static void ddaA2aImplicitCloseBlocking(ncclComm* comm, ncclIpcImplicitReg* reg) {
  int pending = 1;
  while (pending) {
    if (ncclIpcImplicitDeregPoll(comm, reg, &pending, /*asyncClose=*/false) != ncclSuccess) break;
    if (pending) sched_yield();
  }
  ncclIpcImplicitFree(reg);
}

// Runs from commDestroySync's legacy-IPC cleanup drain, i.e. while the proxies (which close the
// peer imports) are still up; ncclDdaA2aDirectCommFini runs after they stop. Enqueued by
// ncclDdaA2aLazyInit when the implicit part was agreed.
// After an abort or a revoke the peers may be gone or stuck: nothing is waited for, the host state is
// freed by ncclDdaA2aDirectCommFini (the peer imports die with the peers' processes).
static ncclResult_t ddaA2aImplicitCleanup(ncclComm* comm, ncclCommCallback* cb) {
  free(cb);
  auto* st = static_cast<DdaA2aDirectState*>(comm->ddaA2aDirectState);
  if (st == nullptr) return ncclSuccess;
  if (st->cache.empty() && st->retiring.empty() && st->zombies.empty()) return ncclSuccess;
  if ((comm->abortFlag != nullptr && __atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE) != 0) || comm->revokedFlag) {
    INFO(NCCL_REG, "DDA A2A implicit: rank %d: communicator aborted / revoked, peer imports not closed", comm->rank);
    return ncclSuccess;
  }
  // The comm's own streams are synchronized by commDestroySync; a retired mapping may still be in use
  // by the call that retired it (user stream): wait for that call only. Closing a cached entry's
  // imports waits, in each peer's proxy, for that peer's device to idle.
  for (auto& r : st->retiring) {
    if (r.ev != nullptr) CUDACHECKIGNORE(cudaEventSynchronize(r.ev));
  }
  INFO(NCCL_REG, "DDA A2A implicit: rank %d cache %zu entries, %llu registrations, %llu evictions, %llu invalidations, %llu closed, %zu zombies, %zu legacy bytes, %llu staged by the legacy budget, %llu staged by a full cache",
       comm->rank, st->cache.size(), (unsigned long long)st->nRegister, (unsigned long long)st->nEvict,
       (unsigned long long)st->nInvalidate, (unsigned long long)st->nClosed, st->zombies.size(), st->legacyBytes,
       (unsigned long long)st->nRefusedLegacy, (unsigned long long)st->nRefusedFull);
  for (auto& r : st->retiring) {
    if (r.ev != nullptr) CUDACHECKIGNORE(cudaEventDestroy(r.ev));
    ddaA2aImplicitCloseBlocking(comm, r.reg);
  }
  // Stuck entries (legacy import still in flight) are left to ncclDdaA2aDirectCommFini (host memory only).
  std::vector<DdaA2aImplicitEntry*> keep;
  for (auto* v : {&st->cache, &st->zombies}) {
    for (auto* e : *v) {
      if (e->stuck) {
        keep.push_back(e);
        continue;
      }
      if (e->reg != nullptr) ddaA2aImplicitCloseBlocking(comm, e->reg);
      delete e;
    }
  }
  st->retiring.clear();
  st->cache.swap(keep);
  st->zombies.clear();
  return ncclSuccess;
}

// Graph-destructor unpins (queued by unpinDestructor), then zombies that nothing holds any more:
// VMM ones are retired (closed once this call's stream passes the point where they were last
// advertised); legacy ones stay open until destroy (counted in legacyBytes).
static void ddaA2aImplicitUnpin(DdaA2aDirectState* st, cudaStream_t stream, bool capturing) {
  dda_a2a_implicit::applyUnpins(*st->unpinQ, st->cache, st->zombies);
  if (capturing || st->zombies.empty()) return;
  std::vector<DdaA2aImplicitEntry*> keep, release;
  for (auto* z : st->zombies) {
    (z->pins == 0 && !z->legacy && !(rcclParamDdaA2aImplicitDebug() & 16) ? release : keep).push_back(z);
  }
  if (release.empty()) return;
  st->zombies.swap(keep);
  for (auto* z : release) ddaA2aImplicitRetire(st, z, stream, false);  // -> retiring (zombie again if no event)
}

// Whether a registration of this allocation would use legacy (hipIpc) imports; mirrors
// ncclIpcImplicitRegStart's choice (VMM handle available -> POSIX fd / FABRIC, else hipIpc).
static bool ddaA2aImplicitIsLegacyAlloc(uintptr_t base) {
#if ROCM_VERSION >= 70000
  CUmemGenericAllocationHandle h;
  if (ncclCuMemEnable() && cuMemRetainAllocationHandle(&h, (void*)base) == CUDA_SUCCESS) {
    (void)cuMemRelease(h);
    return false;
  }
  (void)cudaGetLastError();
#endif
  (void)base;
  return true;
}

// Fills rmt[r] = recvbuff in peer r's VA, or 0 (that pair goes through my staging buffer).
// Per call (hit): one address-range + buffer-id query, a capture check and a linear scan of <= cap entries.
// Bounded: the cache never holds more than RCCL_DDA_A2A_IMPLICIT_CACHE entries (all pinned -> the new
// allocation is staged), and legacy imports never exceed RCCL_DDA_A2A_IMPLICIT_LEGACY_BYTES (-> staged).
static void ddaA2aImplicitLookup(ncclComm* comm, DdaA2aDirectState* st, void* recvbuff, size_t total,
                                 cudaStream_t stream, bool capturing, uintptr_t* rmt) {
  for (int r = 0; r < comm->nRanks; ++r) rmt[r] = 0;
  if (!st->retiring.empty()) ddaA2aImplicitProgress(comm, st, capturing);
  CUdeviceptr b = 0;
  size_t sz = 0;
  unsigned long long id = 0;
  if (cuMemGetAddressRange(&b, &sz, (CUdeviceptr)recvbuff) != CUDA_SUCCESS || b == 0 ||
      (uintptr_t)recvbuff + total > (uintptr_t)b + sz ||
      hipPointerGetAttribute(&id, HIP_POINTER_ATTRIBUTE_BUFFER_ID, (hipDeviceptr_t)recvbuff) != hipSuccess) {
    (void)cudaGetLastError();
    return;  // not a single device allocation we can identify: stage every pair
  }
  if (rcclParamDdaA2aImplicitDebug() & 1) id = 0;
  const uintptr_t base = (uintptr_t)b;
  ddaA2aImplicitUnpin(st, stream, capturing);
  DdaA2aImplicitEntry* e = nullptr;
  size_t slot = st->cache.size();
  for (size_t i = 0; i < st->cache.size(); ++i) {
    if (st->cache[i]->base == base) {
      slot = i;
      break;
    }
  }
  if (slot < st->cache.size() && st->cache[slot]->bufId == id && st->cache[slot]->size == sz) {
    e = st->cache[slot];
    if (e->pending > 0) {
      const int before = e->pending;
      (void)ncclIpcImplicitRegPoll(comm, e->reg, e->peerAddr, &e->pending);
      ++e->pollCalls;
      if (e->pending != before) {
        INFO(NCCL_REG, "DDA A2A implicit: rank %d allocation %p: %d imports still in flight after %d calls, %.3f ms",
             comm->rank, (void*)e->base, e->pending, e->pollCalls, (clockNano() - e->t0) / 1e6);
      }
    }
  } else {
    int64_t cap = rcclParamDdaA2aImplicitCache();
    if (cap < 1) cap = 1;
    // Room: a reallocation at a cached base takes that slot; otherwise the cache must be below its cap
    // or hold an unpinned entry to evict.
    size_t lru = st->cache.size();
    if (slot == st->cache.size() && (int64_t)st->cache.size() >= cap) {
      lru = dda_a2a_implicit::lruUnpinned(st->cache);
      if (lru == st->cache.size()) {
        st->nRefusedFull++;
        if (!st->capFullNoted) {
          st->capFullNoted = true;
          INFO(NCCL_REG, "DDA A2A implicit: rank %d: all %lld cached allocations are held by live graphs; new "
               "receive allocations are staged until a graph is destroyed (RCCL_DDA_A2A_IMPLICIT_CACHE)",
               comm->rank, (long long)cap);
        }
        return;
      }
    }
    // Miss: revive a zombie of this exact allocation, else register (waits for the peer imports;
    // a peer whose import failed stays staged).
    for (size_t i = 0; i < st->zombies.size() && e == nullptr; ++i) {
      DdaA2aImplicitEntry* z = st->zombies[i];
      if (z->base == base && z->size == sz && z->bufId == id && id != 0) {
        e = z;
        st->zombies.erase(st->zombies.begin() + i);
      }
    }
    if (e == nullptr) {
      // hipMalloc memory: legacy imports cannot be closed before destroy, so they are budgeted.
      const int64_t budget = rcclParamDdaA2aImplicitLegacyBytes();
      if (budget >= 0 && ddaA2aImplicitIsLegacyAlloc(base) &&
          !dda_a2a_implicit::legacyAllowed(budget, st->legacyBytes, sz)) {
        st->nRefusedLegacy++;
        if (!st->legacyBudgetNoted) {
          st->legacyBudgetNoted = true;
          INFO(NCCL_REG, "DDA A2A implicit: rank %d: hipMalloc receive buffers beyond %lld bytes per communicator "
               "are staged (RCCL_DDA_A2A_IMPLICIT_LEGACY_BYTES; %zu imported); ncclMemAlloc buffers take "
               "zero-copy without this budget",
               comm->rank, (long long)budget, st->legacyBytes);
        }
        return;
      }
      e = ddaA2aImplicitRegister(comm, base, sz, id);
      if (e == nullptr) return;
      e->uid = ++st->nextUid;
      if (e->legacy) st->legacyBytes += sz;
      st->nRegister++;
    }
    if (slot < st->cache.size()) {
      DdaA2aImplicitEntry* old = st->cache[slot];  // freed + reallocated at the same VA
      st->cache[slot] = e;
      st->nInvalidate++;
      ddaA2aImplicitRetire(st, old, stream, capturing);
    } else {
      if (lru < st->cache.size()) {
        DdaA2aImplicitEntry* old = st->cache[lru];
        st->cache.erase(st->cache.begin() + lru);
        st->nEvict++;
        ddaA2aImplicitRetire(st, old, stream, capturing);
      }
      st->cache.push_back(e);
    }
  }
  if (capturing) {
    // Pinned until the graph is destroyed: its replays advertise this mapping.
    ++e->pins;
    struct ncclCudaGraph g;
    auto* arg = new (std::nothrow) dda_a2a_implicit::UnpinArg{st->unpinQ, e->uid};
    if (arg != nullptr &&
        (ncclCudaGetCapturingGraph(&g, stream, comm->config.graphUsageMode) != ncclSuccess || !ncclCudaGraphValid(g) ||
         ncclCudaGraphAddDestructor(g, dda_a2a_implicit::unpinDestructor, arg) != ncclSuccess)) {
      delete arg;  // no destructor: pinned for the comm's lifetime (as without graph tracking)
    }
    (void)cudaGetLastError();
  }
  e->lastUse = ++st->clock;
  const uintptr_t off = (uintptr_t)recvbuff - base;
  for (int r = 0; r < comm->nRanks; ++r) {
    if (r != comm->rank && e->peerAddr[r] != 0 && ((e->peerAddr[r] + off) & 15) == 0) rmt[r] = e->peerAddr[r] + off;
  }
}

static bool ddaA2aRegIsIncomplete(const DdaA2aDirectState* st, const struct ncclReg* reg) {
  for (const auto& r : st->regIncomplete) {
    if (r.reg == reg && r.beg == reg->begAddr && r.end == reg->endAddr) return true;
  }
  return false;
}

// rmt[r] = this rank's recvbuff in peer r's VA, or 0 (r stages into my staging buffer). Per rank:
// changes only what this rank posts in its ADDR cells, never the kernel.
// The caller runs this in relaxed capture mode (ncclAllToAllDdaFabricDirect's hybrid launch).
static void ddaA2aImplicitRecvInPeers(ncclComm* comm, DdaA2aDirectState* st, void* recvbuff, size_t total,
                                      cudaStream_t stream, uintptr_t* rmt) {
  for (int r = 0; r < comm->nRanks; ++r) rmt[r] = 0;
  hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
  const bool capturing = stream != nullptr && hipStreamIsCapturing(stream, &cs) == hipSuccess &&
                         cs != hipStreamCaptureStatusNone;
  (void)cudaGetLastError();
  struct ncclReg* reg = nullptr;
  bool valid = false;
  if (ncclRegFind(comm, recvbuff, total, &reg) != ncclSuccess) reg = nullptr;
  if (reg != nullptr && (ncclRegLocalIsValid(reg, &valid) != ncclSuccess || !valid)) reg = nullptr;
  if (reg != nullptr) {
    // The implicit cache already holds a legacy (hipIpc) import of this allocation (it saw the buffer
    // before ncclCommRegister did): keep using it. A second legacy import of the same allocation breaks
    // when either is closed.
    CUdeviceptr b = 0;
    size_t sz = 0;
    if (cuMemGetAddressRange(&b, &sz, (CUdeviceptr)recvbuff) == CUDA_SUCCESS && b != 0 &&
        dda_a2a_implicit::hasLegacyAt(st->cache, st->zombies, (uintptr_t)b))
      reg = nullptr;
    (void)cudaGetLastError();
  }
  if (reg != nullptr && !ddaA2aRegIsIncomplete(st, reg)) {
    // ncclCommRegister record: its NCCL IPC registration (cached by the record) gives the addresses.
    int peers[dda::common::kDdaA2aDirectMaxRanks];
    int nPeers = 0;
    for (int r = 0; r < comm->nRanks; ++r) {
      if (r != comm->rank) peers[nPeers++] = r;
    }
    int regFlag = 0;
    uintptr_t offset = 0;
    uintptr_t* rmtDev = nullptr;
    const ncclResult_t rr =
      ncclIpcLocalRegisterBuffer(comm, recvbuff, total, peers, nPeers, NCCL_IPC_COLLECTIVE, &regFlag, &offset, &rmtDev);
    // A first registration probes a FABRIC export that is refused before the POSIX-fd fallback; that
    // leaves a sticky last error which the launch's error check would otherwise report.
    (void)cudaGetLastError();
    bool all = rr == ncclSuccess && regFlag && reg->regIpcAddrs.hostPeerRmtAddrs != nullptr;
    for (int r = 0; r < comm->nRanks; ++r) {
      if (r == comm->rank || reg->regIpcAddrs.hostPeerRmtAddrs == nullptr) continue;
      const int idx = comm->p2pCrossClique ? r : comm->rankToLocalRank[r];
      const uintptr_t a = reg->regIpcAddrs.hostPeerRmtAddrs[idx];
      if (a == 0 || ((a + offset) & 15) != 0) {
        all = false;
      } else {
        rmt[r] = a + offset;
      }
    }
    if (all) return;
    // Incomplete (ipcRegisterBuffer stops at the first failing peer, e.g. one on another host when
    // the allocation has no FABRIC export): it would be retried every call and leave even reachable
    // peers staged. Hand the record to the implicit cache (per-peer results, negative results
    // cached), unless one of its peer imports is a legacy hipIpc one: a second legacy import of the
    // same allocation breaks when the first is closed (A11). Then the missing pairs stay staged.
    for (int i = 0; i < reg->ipcInfosSize && reg->ipcInfos != nullptr; ++i) {
      if (reg->ipcInfos[i] != nullptr && reg->ipcInfos[i]->impInfo.legacyIpcCap) return;
    }
    if (st->regIncomplete.size() >= 64) st->regIncomplete.erase(st->regIncomplete.begin());  // a drop costs one retry
    st->regIncomplete.push_back({reg, reg->begAddr, reg->endAddr});
    INFO(NCCL_REG, "DDA A2A direct: rank %d registration of %p not complete for every peer; using the implicit cache for it",
         comm->rank, recvbuff);
  }
  ddaA2aImplicitLookup(comm, st, recvbuff, total, stream, capturing, rmt);
}

} // namespace

ncclResult_t ncclDdaA2aDirectCommFini(ncclComm* comm) {
  if (comm == nullptr || comm->ddaA2aDirectState == nullptr) return ncclSuccess;
  auto* st = static_cast<DdaA2aDirectState*>(comm->ddaA2aDirectState);
  // Normally empty (released by ddaA2aImplicitCleanup, before the proxies stop); after an abort the
  // peer imports die with the peers' processes: only free host memory here.
  for (auto& r : st->retiring) {
    if (r.ev != nullptr) CUDACHECKIGNORE(cudaEventDestroy(r.ev));
    ncclIpcImplicitFree(r.reg);
  }
  for (auto* e : st->cache) {
    ncclIpcImplicitFree(e->reg);
    delete e;
  }
  for (auto* z : st->zombies) {
    ncclIpcImplicitFree(z->reg);
    delete z;
  }
  st->retiring.clear();
  st->cache.clear();
  st->zombies.clear();
  if (st->scratch != nullptr) {
    CUDACHECKIGNORE(cudaFree(st->scratchPeerDev));
    for (size_t i = 0; i < st->scratchMaps.size(); ++i) {
      if ((int)i != comm->rank && st->scratchMaps[i] != nullptr) (void)ncclCuMemFreeAddr(st->scratchMaps[i], comm->memManager);
    }
    (void)ncclCuMemFree(st->scratch, comm->memManager);
  }
  CUDACHECKIGNORE(cudaFree(st->peerFlagsDev));
  for (size_t i = 0; i < st->peerMaps.size(); ++i) {
    if ((int)i != comm->rank && st->peerMaps[i] != nullptr) (void)ncclCuMemFreeAddr(st->peerMaps[i], comm->memManager);
  }
  (void)ncclCuMemFree(st->base, comm->memManager);
  delete st;
  comm->ddaA2aDirectState = nullptr;
  return ncclSuccess;
}

// Shape + window, independent of the path's state (the lazy trigger uses it before the flag buffer
// exists). Every input is the same on every rank under the collective contract: count, datatype,
// nRanks, localRanks; out of place, 16B alignment and "recvbuff in a symmetric window" are the
// caller's to keep the same on every rank (as for the symmetric kernels). No side effects.
bool ncclAllToAllDdaFabricDirectWindowed(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                         ncclDataType_t datatype) {
  if (comm == nullptr || count == 0 || sendbuff == recvbuff) return false;
  if (comm->nRanks > dda::common::kDdaA2aDirectMaxRanks || comm->localRanks != comm->nRanks) return false;
  const size_t perPeer = count * ncclTypeSize(datatype);
  if (perPeer % 16 != 0) return false;
  if (((uintptr_t)sendbuff | (uintptr_t)recvbuff) % 16 != 0) return false;
  struct ncclDevrWindow* win = nullptr;
  if (ncclDevrFindWindow(comm, recvbuff, &win) != ncclSuccess || win == nullptr) return false;
  if (!(win->winFlags & NCCL_WIN_COLL_SYMMETRIC) || !ncclDevrWinRegEnabled(win->winFlags, ncclDevrRegisterLsa))
    return false;
  return (uintptr_t)recvbuff - (uintptr_t)win->userPtr + perPeer * comm->nRanks <= win->size;
}

// Windowed + the agreed flag buffer + geometry + every peer's mapping. No side effects.
bool ncclAllToAllDdaFabricDirectEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                         ncclDataType_t datatype) {
  if (comm == nullptr || comm->ddaA2aDirectState == nullptr) return false;
  if (!ncclAllToAllDdaFabricDirectWindowed(comm, sendbuff, recvbuff, count, datatype)) return false;
  if (!ddaA2aDirectGeom(comm, nullptr, nullptr)) return false;
  uintptr_t peerRecv[dda::common::kDdaA2aDirectMaxRanks];
  return ddaA2aDirectPeerRecv(comm, recvbuff, peerRecv);
}

// Implicit (hybrid kernel) for a call that is not windowed: the agreed implicit part, shape,
// geometry and the staging round bound. Every input is comm-uniform (the agreed parts and staging
// size, count, datatype, nRanks, localRanks, comm-wide CU cap) except in-place-ness and alignment,
// which the caller keeps the same on every rank. Nothing about recvbuff's registration is an input.
// No side effects.
bool ncclAllToAllDdaFabricDirectImplicitEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                                 ncclDataType_t datatype) {
  if (comm == nullptr || comm->ddaA2aDirectState == nullptr || count == 0 || sendbuff == recvbuff) return false;
  const auto* st = static_cast<const DdaA2aDirectState*>(comm->ddaA2aDirectState);
  return dda_a2a_implicit::implicitEligible(st->scratch != nullptr, st->scratchBytes, comm->nRanks, comm->localRanks,
                                            dda::common::kDdaA2aDirectMaxRanks, count * ncclTypeSize(datatype),
                                            (uintptr_t)sendbuff, (uintptr_t)recvbuff,
                                            ddaA2aDirectGeom(comm, nullptr, nullptr),
                                            (size_t)dda::common::kDdaA2aDirectMaxRounds);
}

uint32_t ncclAllToAllDdaFabricDirectBlocks(ncclComm* comm) {
  int bpp = 1, sb = 1;
  (void)ddaA2aDirectGeom(comm, &bpp, &sb);
  return (uint32_t)(sb + (comm->nRanks - 1) * bpp);
}

static dim3 ddaA2aDirectBlock() {
  const int64_t thr = rcclParamDdaA2aDirectThreads();
  return dim3((thr >= 64 && thr <= 512 && thr % 64 == 0) ? (unsigned)thr : 256u);
}

static int64_t ddaA2aDirectUnroll(size_t total) {
  int64_t unroll = rcclParamDdaA2aDirectUnroll();
  if (unroll != 8 && unroll != 16) unroll = total > (256u << 20) ? 16 : 8;
  return unroll;
}

static ncclResult_t ddaA2aDirectLaunch(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                       ncclComm* comm, cudaStream_t stream) {
  auto* st = static_cast<DdaA2aDirectState*>(comm->ddaA2aDirectState);
  const size_t perPeer = count * ncclTypeSize(datatype);
  const size_t total = perPeer * comm->nRanks;
  dda::common::DdaA2aDirectArgs a = {};
  // The selector checked all of these on this call; a failure here is internal.
  if (st == nullptr || !ddaA2aDirectGeom(comm, &a.bpp, &a.selfBlocks) ||
      !ddaA2aDirectPeerRecv(comm, recvbuff, a.peerRecv))
    return ncclInternalError;
  a.sendbuff = static_cast<const char*>(sendbuff);
  a.recvbuff = static_cast<char*>(recvbuff);
  a.peerFlags = st->peerFlagsDev;
  a.selfFlags = static_cast<char*>(st->base);
  a.wd = ncclDdaA2aWdCtx(comm);
  a.chunkBytes = perPeer;
  a.rank = comm->rank;
  a.nRanks = comm->nRanks;
  const dim3 grid(ncclAllToAllDdaFabricDirectBlocks(comm));
  const dim3 block = ddaA2aDirectBlock();
  const int64_t unroll = ddaA2aDirectUnroll(total);

  INFO(NCCL_COLL, "DDA fabric AllToAll DIRECT (zero-copy): nRanks=%d perPeer=%zu grid=%u bpp=%d self=%d block=%u unroll=%lld",
       comm->nRanks, perPeer, grid.x, a.bpp, a.selfBlocks, block.x, (long long)unroll);

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  (void)cudaGetLastError();  // drop sticky errors of handled calls (refused FABRIC export in p2p IPC registration)
  if (unroll == 16) {
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricDirect<16>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, a);
  } else {
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricDirect<8>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, a);
  }
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

static ncclResult_t ddaA2aDirectHybridLaunch(const void* sendbuff, void* recvbuff, size_t count,
                                             ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  auto* st = static_cast<DdaA2aDirectState*>(comm->ddaA2aDirectState);
  const size_t perPeer = count * ncclTypeSize(datatype);
  const size_t total = perPeer * comm->nRanks;
  dda::common::DdaA2aDirectHybridArgs a = {};
  if (st == nullptr || st->scratch == nullptr || !ddaA2aDirectGeom(comm, &a.bpp, &a.selfBlocks)) return ncclInternalError;
  a.slotBytes = dda_a2a_implicit::stagingSlot(st->scratchBytes, comm->nRanks, perPeer);
  if (a.slotBytes == 0 || dda_a2a_implicit::stagingRounds(st->scratchBytes, comm->nRanks, perPeer) >
                            (size_t)dda::common::kDdaA2aDirectMaxRounds)
    return ncclInternalError;
  // Per rank (host only): which of my receive pairs are written directly. Relaxed capture mode: the
  // lookup's runtime calls (event record / query, registration) must not invalidate a capture of this
  // or another thread (as ncclDdaA2aLazyInit).
  {
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    const bool modeSet = cudaThreadExchangeStreamCaptureMode(&mode) == cudaSuccess;
    (void)cudaGetLastError();
    ddaA2aImplicitRecvInPeers(comm, st, recvbuff, total, stream, a.myRecvInPeer);
    if (modeSet) (void)cudaThreadExchangeStreamCaptureMode(&mode);
  }
  {
    // Test-only (DDA_A2A_IMPLICIT_DEBUG): force pairs through staging (2: all, 4: asymmetric subset
    // (distance % 3 == 1), 32: only rank 0 receives staged, from rank 1).
    const int64_t dbg = rcclParamDdaA2aImplicitDebug();
    for (int r = 0; r < comm->nRanks; ++r) {
      const int d = (r - comm->rank + comm->nRanks) % comm->nRanks;
      if ((dbg & 2) || ((dbg & 4) && d % 3 == 1) || ((dbg & 32) && comm->rank == 0 && r == 1)) a.myRecvInPeer[r] = 0;
    }
  }
  a.sendbuff = static_cast<const char*>(sendbuff);
  a.recvbuff = static_cast<char*>(recvbuff);
  a.peerFlags = st->peerFlagsDev;
  a.selfFlags = static_cast<char*>(st->base);
  a.wd = ncclDdaA2aWdCtx(comm);
  a.scratchPeer = st->scratchPeerDev;
  a.scratchSelf = static_cast<char*>(st->scratch);
  a.chunkBytes = perPeer;
  a.rank = comm->rank;
  a.nRanks = comm->nRanks;
  int nStaged = 0;
  for (int r = 0; r < comm->nRanks; ++r) nStaged += (r != comm->rank && a.myRecvInPeer[r] == 0);
  const dim3 grid(ncclAllToAllDdaFabricDirectBlocks(comm));
  const dim3 block = ddaA2aDirectBlock();
  const int64_t unroll = ddaA2aDirectUnroll(total);

  INFO(NCCL_COLL, "DDA fabric AllToAll DIRECT implicit (zero-copy, per-pair staging): nRanks=%d perPeer=%zu grid=%u bpp=%d self=%d block=%u unroll=%lld stagedRecvPairs=%d slot=%zu",
       comm->nRanks, perPeer, grid.x, a.bpp, a.selfBlocks, block.x, (long long)unroll, nStaged, a.slotBytes);

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  (void)cudaGetLastError();  // drop sticky errors of handled calls (refused exports / imports above)
  if (unroll == 16) {
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricDirectHybrid<16>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, a);
  } else {
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricDirectHybrid<8>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, a);
  }
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

ncclResult_t ncclAllToAllDdaFabricDirect(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                         ncclComm* comm, cudaStream_t stream) {
  // The selector took this call either windowed (window kernel) or, not windowed, with the implicit
  // part agreed (hybrid kernel); "windowed" is the same on every rank, so is the kernel.
  // Any failure poisons the comm: peers may already run this call, so this rank's device epoch would
  // be one behind theirs; later AllToAlls are refused instead (as for push, rcclDdaA2aPushRun).
  const ncclResult_t r = dda_a2a_implicit::dispatchDirect(
    ncclAllToAllDdaFabricDirectWindowed(comm, sendbuff, recvbuff, count, datatype),
    [&] { return ddaA2aDirectLaunch(sendbuff, recvbuff, count, datatype, comm, stream); },
    [&] { return ddaA2aDirectHybridLaunch(sendbuff, recvbuff, count, datatype, comm, stream); });
  if (r != ncclSuccess) ncclDdaA2aPoison(comm, kDdaA2aErrLaunch);
  return r;
}
