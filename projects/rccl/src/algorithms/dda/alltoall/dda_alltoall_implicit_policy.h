/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Bounds of the DDA AllToAll implicit registration cache (dda_alltoall_fabric_direct_host.h),
 * independent of HIP so host tests can check them: graph-destructor unpins, the cache cap with
 * pinned entries, and the byte budget of legacy (hipMalloc / hipIpc) imports.
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

}  // namespace dda_a2a_implicit

#endif  // DDA_ALLTOALL_IMPLICIT_POLICY_H_
