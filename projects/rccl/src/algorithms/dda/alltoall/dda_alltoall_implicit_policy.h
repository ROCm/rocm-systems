/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Bounds of the DDA AllToAll implicit registration cache (dda_alltoall_fabric_direct_host.h),
 * independent of HIP so host tests can check them: graph-destructor unpins, the cache cap with
 * pinned entries, the byte budget of legacy (hipMalloc / hipIpc) imports, and the staging slot of
 * pairs that are not mapped.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_ALLTOALL_IMPLICIT_POLICY_H_
#define DDA_ALLTOALL_IMPLICIT_POLICY_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace dda_a2a_implicit {

// Graph destructor -> comm: uids of cache entries whose capturing graph was destroyed. Shared
// (shared_ptr) with every destructor object, so a graph destroyed after the comm, or a comm
// destroyed with graphs alive, never touches freed memory. The destructor runs on a runtime
// thread and may not call HIP: it only queues; the next lookup drains.
struct UnpinQueue {
  std::mutex mu;
  std::vector<uint64_t> uids;
};
struct UnpinArg {
  std::shared_ptr<UnpinQueue> q;
  uint64_t uid;
};
inline void unpinDestructor(void* p) {
  auto* a = static_cast<UnpinArg*>(p);
  {
    std::lock_guard<std::mutex> lk(a->q->mu);
    a->q->uids.push_back(a->uid);
  }
  delete a;
}

// Drains the queue: each uid releases one pin of the entry carrying it, in either list
// (E: uid, pins). Unknown uids (entry already gone at destroy) are ignored.
template <class E>
void applyUnpins(UnpinQueue& q, std::vector<E*>& cache, std::vector<E*>& zombies) {
  std::vector<uint64_t> uids;
  {
    std::lock_guard<std::mutex> lk(q.mu);
    uids.swap(q.uids);
  }
  for (uint64_t uid : uids) {
    bool done = false;
    for (auto* v : {&cache, &zombies}) {
      for (E* e : *v) {
        if (e->uid == uid && e->pins > 0) {
          --e->pins;
          done = true;
          break;
        }
      }
      if (done) break;
    }
  }
}

// Eviction victim for a new allocation when the cache is at its cap: the least recently used entry
// that no live graph pins (E: pins, lastUse). cache.size() if every entry is pinned: the caller
// must then not register (stage the call) rather than grow past the cap.
template <class E>
size_t lruUnpinned(const std::vector<E*>& cache) {
  size_t lru = cache.size();
  for (size_t i = 0; i < cache.size(); ++i) {
    if (cache[i]->pins > 0) continue;
    if (lru == cache.size() || cache[i]->lastUse < cache[lru]->lastUse) lru = i;
  }
  return lru;
}

// Legacy (hipIpc) imports can only be closed at comm destroy (closing one while a newer import of a
// reallocation exists breaks the newer one), so they are bounded by refusing new ones: budget < 0
// unlimited, 0 never.
inline bool legacyAllowed(int64_t budget, size_t legacyBytes, size_t size) {
  return budget < 0 || (legacyBytes <= (uint64_t)budget && size <= (uint64_t)budget - legacyBytes);
}

// Staging (pairs whose receiver has no mapping): the receiver's staging buffer of scratchBytes holds
// one slot per source, a 4 KiB multiple, no larger than the per-peer chunk rounded up to 4 KiB; a
// staged pair moves its chunk in ceil(perPeer / slot) rounds. 0 when the buffer is too small for a
// 4 KiB slot per rank. Every input is comm-uniform (agreed scratchBytes, nRanks, count), so every rank
// computes the same slot and round count.
inline size_t stagingSlot(size_t scratchBytes, int nRanks, size_t perPeer) {
  if (nRanks <= 0) return 0;
  size_t slot = scratchBytes / (size_t)nRanks / 4096 * 4096;
  const size_t pr = (perPeer + 4095) / 4096 * 4096;
  if (slot > pr) slot = pr;
  return slot;
}
// SIZE_MAX when there is no slot.
inline size_t stagingRounds(size_t scratchBytes, int nRanks, size_t perPeer) {
  const size_t slot = stagingSlot(scratchBytes, nRanks, perPeer);
  return slot == 0 ? SIZE_MAX : (perPeer + slot - 1) / slot;
}

// Retiring entries (R: ev, plus whatever close() needs): an entry's peer imports are closed once the
// event recorded when it was retired completed. Nothing is queried while the calling stream is
// capturing (an event query during a capture invalidates it -- and, in HIP, every global-mode
// capture of the process); query(r) returns true only for a definite completion (it then clears
// r.ev), anything else (not ready, an error) is "not yet". close(r) returns true once the closes
// are done; the entry is then dropped.
template <class R, class Query, class Close>
void progressRetiring(std::vector<R>& v, bool capturing, Query query, Close close) {
  if (capturing) return;
  for (size_t i = 0; i < v.size();) {
    if (v[i].ev != nullptr && !query(v[i])) {
      ++i;
      continue;
    }
    if (close(v[i])) {
      v.erase(v.begin() + i);
    } else {
      ++i;
    }
  }
}

// Whether the cache or the retired-but-mapped list holds a legacy (hipIpc) import at this allocation
// base (E: base, legacy). An ncclCommRegister record of such a buffer must not import it again.
template <class E>
bool hasLegacyAt(const std::vector<E*>& cache, const std::vector<E*>& zombies, uintptr_t base) {
  for (const auto* v : {&cache, &zombies}) {
    for (const E* e : *v) {
      if (e->legacy && e->base == base) return true;
    }
  }
  return false;
}

// The implicit (hybrid kernel) predicate for a call that is not windowed. Every input is
// comm-uniform (agreed part and staging size, nRanks, localRanks, per-peer bytes, the comm-wide
// geometry) or the caller's to keep uniform (in-place-ness, alignment). Nothing about the receive
// buffer's registration is an input, by construction.
inline bool implicitEligible(bool agreed, size_t scratchBytes, int nRanks, int localRanks, int maxRanks,
                             size_t perPeer, uintptr_t sendbuff, uintptr_t recvbuff, bool geomOk, size_t maxRounds) {
  if (!agreed || perPeer == 0 || sendbuff == recvbuff) return false;
  if (nRanks > maxRanks || localRanks != nRanks) return false;
  if (perPeer % 16 != 0 || ((sendbuff | recvbuff) % 16) != 0) return false;
  if (!geomOk) return false;
  return stagingRounds(scratchBytes, nRanks, perPeer) <= maxRounds;
}

// The Direct launcher's kernel choice: window kernel for a windowed call, hybrid kernel otherwise.
template <class Window, class Hybrid>
auto dispatchDirect(bool windowed, Window window, Hybrid hybrid) -> decltype(window()) {
  return windowed ? window() : hybrid();
}

}  // namespace dda_a2a_implicit

#endif  // DDA_ALLTOALL_IMPLICIT_POLICY_H_
