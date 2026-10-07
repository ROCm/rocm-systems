#!/usr/bin/env python3
"""Apply the test-only shims that let run_proxy_freelist_tsan_test.sh observe the
proxy-op free-list handoff.

No shim changes the publish/consume protocol under test; they change how the pool
is mapped, how many ops it hands out, and whether the take path reports itself.
Without them the test cannot observe anything at all, so it would pass no matter
what the code does.

single mapping
    The pool is mapped separately by the proxy thread (proxyProgressInit) and by
    every connection (ncclProxyConnect), so producer and consumer reach the same
    physical memory through different virtual addresses. ThreadSanitizer keys its
    shadow state on the virtual address and therefore cannot correlate the two
    accesses. Reusing one mapping per shm object inside the process makes the
    real protocol observable.

execution proof
    A run that never reaches the shared free list would report no race for the
    trivial reason that it never looked. The take path says once that it ran, and
    the test requires that line.

short free chain
    MAX_OPS_PER_PEER is in the thousands, so in a short run the main thread never
    drains its private chain and never comes back to the shared free list - the
    site under test simply does not execute. Handing each peer a few ops forces
    the handoff to run while the proxy thread recycles into it.

The input is normally the hipified proxy.cc inside the build directory, so the
source tree is left untouched.

Usage:
  proxy_tsan_shims.py --in build/hipify/src/proxy.cc --out /tmp/proxy_tsan.cc
"""
import argparse
import sys

# Anything near MAX_OPS_PER_PEER defeats the shim, so the accepted range is kept
# small on purpose; see the check in main().
MAX_TEST_OPS_PER_PEER = 256

POOL_INIT_OLD = """    for (int r = 0; r < proxyState->tpLocalnRanks; r++) {
      pool->freeOps[r] = r * MAX_OPS_PER_PEER;
      for (int i = 0; i < MAX_OPS_PER_PEER - 1; i++)
        pool->ops[r * MAX_OPS_PER_PEER + i].next = r * MAX_OPS_PER_PEER + i + 1;
      pool->ops[(r + 1) * MAX_OPS_PER_PEER - 1].next = -1;
    }
"""

POOL_INIT_NEW = """    // TEST-ONLY (short free chain): see proxy_tsan_shims.py. Each peer starts with
    // a few ops instead of MAX_OPS_PER_PEER so the main thread runs out and has to
    // come back to the shared pool->freeOps list while the proxy thread recycles
    // into it. Capacity only; the pool keeps its size and the handoff protocol is
    // untouched.
    for (int r = 0; r < proxyState->tpLocalnRanks; r++) {
      const int testOpsPerPeer = %d;
      // A chain longer than one peer's partition would link ops that belong to
      // the next peer, so two peers could hand out the same op. The bound is
      // checked here because only the compiler knows MAX_OPS_PER_PEER, which
      // depends on the target.
      static_assert(testOpsPerPeer <= MAX_OPS_PER_PEER,
                    "test free chain does not fit in one peer's partition of pool->ops");
      pool->freeOps[r] = r * MAX_OPS_PER_PEER;
      for (int i = 0; i < testOpsPerPeer - 1; i++)
        pool->ops[r * MAX_OPS_PER_PEER + i].next = r * MAX_OPS_PER_PEER + i + 1;
      pool->ops[r * MAX_OPS_PER_PEER + testOpsPerPeer - 1].next = -1;
    }
"""

SINGLE_MAPPING_REGISTRY = """
// ---- TEST-ONLY (single shm mapping per pool) -------------------------------
// See proxy_tsan_shims.py: the proxy-op pool is mapped separately by the proxy
// thread and by each connection, so a sanitizer cannot correlate the producer
// and consumer accesses (different virtual addresses, same physical memory).
// Reusing one mapping per shm object inside the process makes the real free-list
// protocol observable. Mapping only; no protocol change.
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
static std::mutex g_testShmMutex;
static std::map<std::string, void*> g_testShmCache;
// Producer and consumer spell the same object differently but both end in the
// mkstemp suffix of /dev/shm/nccl-XXXXXX, which is what lets one key serve both.
// That shape is checked rather than assumed: if it ever changes, the lookup would
// quietly miss, the pool would be mapped twice again, and the test would observe
// nothing and pass.
static std::string testShmKey(const char* p) {
  std::string s(p);
  size_t at = s.rfind("nccl-");
  if (at == std::string::npos || s.size() - (at + 5) != 6) {
    fprintf(stderr, "[test] proxy-op shm path '%s' is not .../nccl-XXXXXX, "
                    "so the single-mapping shim cannot key on it\\n", p);
    abort();
  }
  return s.substr(at + 5);
}
static void testShmRegister(const char* p, void* addr) {
  std::lock_guard<std::mutex> lk(g_testShmMutex);
  g_testShmCache[testShmKey(p)] = addr;
}
static void* testShmLookup(const char* p) {
  std::lock_guard<std::mutex> lk(g_testShmMutex);
  auto it = g_testShmCache.find(testShmKey(p));
  return it == g_testShmCache.end() ? nullptr : it->second;
}
// ---------------------------------------------------------------------------
"""

REGISTRY_ANCHOR = "#define NCCL_MAX_PROXY_CONNECTIONS"

CONSUMER_OLD = """    if (proxyOps->pool == NULL) {
      NCCLCHECK(ncclShmOpen(poolPath, sizeof(poolPath), sizeof(struct ncclProxyOpsPool), (void**)(&proxyOps->pool),
                            NULL, -1, &proxyOps->handle));
      proxyOps->nextOps = proxyOps->nextOpsEnd = proxyOps->freeOp = -1;
    }
"""

CONSUMER_NEW = """    if (proxyOps->pool == NULL) {
      void* shared = testShmLookup(poolPath);
      if (shared != nullptr) {
        proxyOps->pool = (struct ncclProxyOpsPool*)shared;
      } else {
        NCCLCHECK(ncclShmOpen(poolPath, sizeof(poolPath), sizeof(struct ncclProxyOpsPool), (void**)(&proxyOps->pool),
                              NULL, -1, &proxyOps->handle));
        testShmRegister(poolPath, proxyOps->pool);
      }
      proxyOps->nextOps = proxyOps->nextOpsEnd = proxyOps->freeOp = -1;
    }
"""

PRODUCER_OLD = """    NCCLCHECK(ncclShmOpen(shmPath, sizeof(shmPath), size, (void**)&pool, NULL, proxyState->tpLocalnRanks,
                          &state->handle));
"""

PRODUCER_NEW = """    NCCLCHECK(ncclShmOpen(shmPath, sizeof(shmPath), size, (void**)&pool, NULL, proxyState->tpLocalnRanks,
                          &state->handle));
    testShmRegister(shmPath, pool);
"""

TAKE_PROOF_OLD = """    opIndex = freeOp;
"""

TAKE_PROOF_NEW = """    opIndex = freeOp;
    // TEST-ONLY (execution proof): see proxy_tsan_shims.py. A run that never
    // reaches the shared free list reports no race for the trivial reason that it
    // never looked, so say once that this path did run.
    {
      static unsigned long testSharedTakes = 0;
      if (__atomic_fetch_add(&testSharedTakes, 1UL, __ATOMIC_RELAXED) == 0)
        fprintf(stderr, "[test] main thread took the shared free list\\n");
    }
"""


def replace_once(text, old, new, what):
    n = text.count(old)
    if n != 1:
        sys.exit(f"[proxy_tsan_shims] expected exactly one occurrence of {what}, found {n}")
    return text.replace(old, new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--ops-per-peer", type=int, default=32,
                    help="ops handed to each peer at pool init (default 32, keep it small)")
    args = ap.parse_args()

    # The value has to stay far below MAX_OPS_PER_PEER, and not merely inside it:
    # a chain the main thread never drains means the handoff under test is never
    # reached, and the run then comes back clean without having checked anything.
    # The generated code also carries a static_assert against the real
    # MAX_OPS_PER_PEER, which is the only place its value is known.
    if not 1 <= args.ops_per_peer <= MAX_TEST_OPS_PER_PEER:
        ap.error(f"--ops-per-peer must be between 1 and {MAX_TEST_OPS_PER_PEER}; the point of the shim is a"
                 " chain short enough that the main thread runs out and takes the shared free list")

    s = open(args.src).read()
    s = replace_once(s, POOL_INIT_OLD, POOL_INIT_NEW % args.ops_per_peer, "the free-list pool init")
    s = replace_once(s, REGISTRY_ANCHOR, SINGLE_MAPPING_REGISTRY + "\n" + REGISTRY_ANCHOR,
                     "the registry anchor")
    s = replace_once(s, CONSUMER_OLD, CONSUMER_NEW, "the connection-side shm open")
    s = replace_once(s, TAKE_PROOF_OLD, TAKE_PROOF_NEW, "the shared free-list take")
    s = replace_once(s, PRODUCER_OLD, PRODUCER_NEW, "the proxy-thread-side shm open")

    open(args.dst, "w").write(s)
    print(f"[proxy_tsan_shims] wrote {args.dst} (ops_per_peer={args.ops_per_peer})")


if __name__ == "__main__":
    main()
