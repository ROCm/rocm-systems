# HotSwap content cache evidence

This directory contains the executable and formal evidence for the process-wide,
content-exact retarget cache:

- `hotswap_cache_test.cc` covers cross-reader reuse, mutation between loads,
  forced hash collisions, synchronization, reentrancy rejection, failure
  recovery, and weak lifetime.
- `model/ContentRetargetCache.tla` specifies the exact-content per-transform
  state machine, including two different contents forced into one hash bucket.
- `hotswap_cache_benchmark.cc` compares no caching, the lookup/copy behavior
  introduced by #8274, and content-exact single-flight sharing.

## Required behavior

The HSA reader APIs require an application-owned memory buffer to outlive its
reader, but do not make the buffer immutable. ROCR also stores the application
pointer and reparses it for each load. The cache must therefore satisfy both of
these cases:

1. Distinct readers over byte-identical code objects and the same transform
   share one rewrite. This is the CLR multi-device path exercised by RCCL.
2. If bytes change between loads, the later load cannot observe an output
   produced from the earlier bytes, even if the pointer, size, reader, and ISA
   are unchanged.

Concurrent application writes to a reader buffer during an HSA load remain
unsupported. A miss takes an immutable snapshot before invoking COMGR, so every
supported call rewrites one stable byte sequence.

## Algorithm and identity

The cache is process-wide. A lookup first selects a bucket with:

```text
(content hash, source size, source ISA, target ISA,
 entry-trampoline mode, strict mode)
```

The content hash is only an index. A weak source-intern table requires an exact
`size + memcmp` match against an immutable snapshot before returning a source
identity. Transform ready entries and flights then compare that exact source
identity. Hash collision cannot change the result selected by a request.
Reader identity is not part of either key. Each reader receives a monotonic
numeric ID used only for cross-reader metrics, so cache metadata never retains
or compares a potentially expired reader pointer.

Each content-hash bucket and hash/transform bucket has its own mutex. Candidate
collection and source snapshot publication are coordinated within one content
bucket; exact comparisons run outside its mutex, followed by a generation
recheck before publication. Transform coordination is serialized only within
one hash/transform bucket. COMGR runs outside all cache mutexes. A process-level
table mutex protects bucket creation and metadata sweeping.

The first content miss creates one source snapshot. The first miss for an exact
source and transform creates one flight, and its caller runs COMGR on the
calling thread. Matching callers wait on that flight and receive the same
immutable result. Different transforms share the interned source snapshot but
have independent flights and outputs. Failures are published to existing
waiters but are never inserted into the ready set, so a later request retries.

## Ownership, eviction, and fragmentation

The rewritten ELF owns its source snapshot. Loaded code objects retain an
aliasing `shared_ptr` to that ELF, while ready entries are `weak_ptr`s. The
following implications are intentional:

- Destroying one reader or executable cannot invalidate another executable's
  code-object metadata.
- The last owner of an output releases that output immediately. The source
  snapshot is released after the last output/flight using that content drops.
- The cache never strongly retains payload merely to improve a future hit.
- Expired weak entries are removed on bucket access. Empty bucket metadata is
  swept periodically as new buckets are inserted.

There is no byte-capacity LRU, TTL, or disk eviction policy. Such a policy is
not needed to bound payload memory because the cache owns no payload after the
last consumer releases it. There is one source allocation per distinct live
content and one COMGR output allocation per distinct live transform. No
separate cache arena is used, so internal cache-arena fragmentation is absent;
ordinary allocator fragmentation is outside the live-byte counters.

File-backed readers retain their existing private read-only mapping. They use
the same in-process content cache as memory readers.

## Disk-persistent tier

The disk tier layers cross-process/cross-run persistence under the single-flight
producer: on a cold miss the leader reads disk before running COMGR, and on a
COMGR success it enqueues an asynchronous write. Because this runs only on the
single-flight leader, disk I/O is coalesced across waiters and never serializes
the in-memory cache.

Its trust model, key strength, and eviction policy are deliberate and defined
here (per the requirement above that a disk cache define its own boundary):

- **Trust boundary — per-user, non-adversarial.** The cache root defaults to
  `$XDG_CACHE_HOME/rocm/hotswap` else `$HOME/.cache/rocm/hotswap`, inside the
  user's own trust boundary. Entries written by the owning user are trusted; a
  user who can already run code in their own process gains nothing by planting a
  cache entry, so this is not a defended threat. `HSA_HOTSWAP_CACHE_DIR` is an
  opt-in override; pointing it at a shared/world-writable location steps outside
  the per-user model by explicit choice.
- **Toolchain salt.** Entries are namespaced by a salt derived from the loaded
  COMGR library identity (path + size + mtime + format version). A toolchain
  change yields a fresh salt subtree, so entries built by a different COMGR are
  never read back.
- **Entry identity — 64-bit key, no on-read content compare.** The lookup key is
  a 64-bit hash over (source bytes, source ISA, target ISA, flags), used as the
  filename; reads validate magic/version/salt/payload-size but do not re-verify
  payload against source bytes. Unlike the in-memory tier — whose exact `memcmp`
  makes correctness independent of hash strength — the disk tier's correctness is
  probabilistic in the key. This is an intentional cost/benefit choice: a true
  exact-compare would require storing and re-reading the full source blob on every
  hit (hundreds of MB for large code objects), and merely widening the digest
  would only shift the collision exponent without providing the exact-compare
  invariant. For a per-user cache the 64-bit birthday-collision probability is
  well below ambient hardware error rates at any realistic cache size.
- **Atomic publication.** Writes go to a unique temp file, are `fsync`'d, then
  `rename`'d into place, so a reader never observes a partially written entry. A
  crash can lose a not-yet-published entry (a future miss), never corrupt one.
- **No eviction.** There is no LRU/TTL/size cap; stale salt subtrees from prior
  toolchains are not swept. Footprint is bounded per toolchain by the set of
  objects actually loaded. A best-effort stale-salt sweep is possible future
  work if disk footprint becomes an operational concern.

## Memory and latency bounds

Let `B` be source bytes, `S_i` the output bytes for live transform `i`, `N`
matching requests, and `C` COMGR latency.

While a rewritten ELF is reusable, any implementation must retain `S_i` bytes.
For deterministic exact reuse when application memory may change, it must also
retain enough immutable information to distinguish every possible `B`-byte
input. In the worst case that requires `B` bytes. This implementation retains
exactly one shared `B`-byte snapshot and one `S_i`-byte output per distinct live
transform, plus bounded object/map metadata. It therefore meets the worst-case
payload lower bound `B + sum(S_i)` even when multiple transforms use the same
source; it does not claim allocator-overhead optimality.

A cold overlapping batch for one transform must perform at least one rewrite
and cannot complete before `C`. This implementation performs one producer, one
shared source snapshot, and one output allocation. A warm exact hit performs no
COMGR work and no payload allocation or output copy. It reads the source for
hash selection and exact verification. Because deterministic
collision-independent identity requires inspecting changed input bytes,
source-size-dependent hit latency is unavoidable; the second exact verification
pass is the explicit cost of making hash collisions irrelevant.

Expected payload behavior for matching requests is:

| Strategy | Cold producers | Live output | Cache source snapshot | Warm output copy |
|---|---:|---:|---:|---:|
| no cache | `N` | `N * S` | `0` | `S` per load |
| #8274 lookup/copy | up to `N` | at least `(N + 1) * S` | `0` | `S` per hit |
| content single-flight | `1` | `S` | one shared `B` | none |

## Synchronization argument

All ready entries, flights, waiter counts, and flight results are accessed
under their bucket mutex. Exact source comparison happens before selecting a
ready value or flight. Insertion of the first flight under that mutex is the
miss linearization point, so at most one producer exists for one exact content
and transform.

The producer publishes its result and `completed = true` while holding the same
mutex used by predicate waits, then calls `notify_all`. This establishes the
happens-before edge from result publication to waiter consumption. Every
waiter owns a `shared_ptr` to its flight, so failure removal and a subsequent
retry cannot destroy or overwrite the generation still observed by old
waiters. Ready values and returned values are immutable shared objects.

The lock order is table mutex before bucket mutex. Normal lookup releases the
table mutex before taking a bucket mutex, source and transform bucket mutexes
are never held together, and no path acquires the table mutex while holding a
bucket mutex. COMGR and producer callbacks run without cache locks. Predicate
waits atomically release their bucket mutex. These rules remove every reverse
lock edge needed to form a cache deadlock cycle.

Each flight records its producer thread. If that thread recursively requests
the same exact source and transform, the nested request returns
`kReentrantRequest` rather than joining and waiting on itself. Other threads
continue to join the flight normally.

The process cache uses `std::call_once` rather than a function-local static.
This matters because the Unix ROCR build uses `-fno-threadsafe-statics`.
Construction failure is propagated to the load path, which rewrites without
caching, and `call_once` permits a later construction retry.

`ContentRetargetCache.tla` abstracts the source-intern step as an exact content
identity, then checks one leader per exact key, waiter-generation association,
success-only ready publication, exact-content result selection, collision
isolation, failure/retry ordering, weak expiry, and agreement among callers
observing one generation. Reader identity is intentionally not part of the
exact key. The model excludes the leader from its waiter set, matching the
implementation's typed reentrancy rejection.

The safety proof assumes application bytes are not concurrently modified during
an HSA call. Liveness additionally depends on COMGR returning, the producer
thread not being asynchronously terminated, and eventual scheduler progress.
A stalled COMGR call can stall matching waiters without creating a cache lock
cycle; the cache does not claim wait-freedom or scheduler fairness.

## Runtime metrics

The counters are present in normal runtime builds and are emitted with
`HSA_HOTSWAP_VERBOSE=1`:

- producer calls/failures, ready hits, cross-reader results, coalesced results,
  and reentrant rejections;
- bytes and time spent hashing and exact-comparing;
- wait and cache-lock time;
- source snapshot allocations and cumulative/live/peak source bytes;
- cumulative/live/peak output bytes; and
- current bucket, ready, and in-flight entry counts.

These counters distinguish the required production outcome, one producer plus
`N-1` cross-reader results, from same-reader microbenchmark reuse.

## Running the evidence

Build and run the focused tests and benchmark:

```console
cmake --build <rocrtst-build> \
  --target hotswap_rewrite hotswap_cache hotswap_cache_benchmark -j
ctest --test-dir <rocrtst-build> \
  -R '^hotswap_(rewrite|cache)$' --output-on-failure
<rocrtst-build>/hotswap_cache_benchmark \
  --threads=1,2,4,8 --bytes=65536,1048576,8388608 \
  --producer-us=2000 --iterations=5
```

Use an optimized build for latency numbers. The benchmark uses `payload_bytes`
for both synthetic source and output size so source-snapshot and output costs
remain directly comparable.

The benchmark is a behavioral model, not a substitute for running RCCL. It
reports output and source-snapshot allocation/live/peak bytes separately. On a
four-caller 1 MiB synthetic cold batch, the expected distinguishing result is
four producers for the lookup/copy model and one producer plus three coalesced
results for content single-flight.

The current evidence run produced:

- focused rewrite/cache CTests: pass;
- cache suite: 13 tests pass under GCC ThreadSanitizer, including forced hash
  collision and randomized schedules through 64 threads;
- TLC: 510,325 states generated, 121,729 distinct states, complete depth 17,
  with no invariant violation; and
- optimized four-caller 1 MiB benchmark: one producer, three coalesced results,
  one source snapshot, one output, and no warm output allocation/copy.

Run the formal model with a TLA+ tools distribution:

```console
cd projects/rocr-runtime/rocrtst/suites/functional/hotswap/model
java -cp /path/to/tla2tools.jar tlc2.TLC \
  -config ContentRetargetCache.cfg ContentRetargetCache.tla
```

For ThreadSanitizer, configure a separate rocrtst build with both compile and
link instrumentation:

```console
cmake -S projects/rocr-runtime/rocrtst/suites/test_common \
  -B build-rocrtst-hotswap-tsan \
  -DCMAKE_CXX_FLAGS="-O1 -g -fsanitize=thread -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" \
  <the same LLVM/ROCR dependency arguments as the normal rocrtst build>
cmake --build build-rocrtst-hotswap-tsan --target hotswap_cache -j
ctest --test-dir build-rocrtst-hotswap-tsan \
  -R '^hotswap_cache$' --output-on-failure
```

GCC TSAN on WSL2 may abort before `main` with `unexpected memory mapping`.
Run the non-PIE test binary with address randomization disabled on that host:

```console
setarch x86_64 -R env TSAN_OPTIONS=halt_on_error=1 \
  build-rocrtst-hotswap-tsan/hotswap_cache
```

A pre-`main` mapping abort is a host limitation rather than a passing race
result.

## Production acceptance

The merge gate for DCGPUBU-1871 is the original four-GPU RCCL initialization
path with normal CLR reader creation. With verbose metrics enabled it must show:

```text
producer_calls = 1
cross_reader_results = 3
reentrant_rejections = 0
produced_output_bytes = one rewritten ELF
live_source_snapshot_bytes = one source code object
live_output_bytes = one rewritten ELF
```

Kernel execution, debugger/profiler code-object inspection, coredump behavior,
and repeated executable destruction must also pass on a supported GPU system.
These hardware checks cannot be replaced by the synthetic benchmark.
