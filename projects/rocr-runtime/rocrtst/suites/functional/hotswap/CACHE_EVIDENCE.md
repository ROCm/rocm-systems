# HotSwap Retarget Cache Evidence

This directory contains three complementary forms of evidence for the
reader-scoped retarget cache:

- `hotswap_cache_test.cc` exercises synchronization, failure recovery,
  ownership, exact-key separation, and randomized schedules up to 64 threads.
- `model/ReaderRetargetCache.tla` specifies the per-key state machine for TLC.
- `hotswap_cache_benchmark.cc` compares no caching, the copy-cache behavior
  merged in #8274, and reader-scoped single-flight sharing.

## Cost model and lower bounds

For `N` overlapping requests for one exact transform key, let `S` be the
rewritten ELF size and `C` the producer latency. A successful cold request must
run the producer at least once, materialize at least one `S`-byte result, and
cannot complete before that producer finishes. While any caller or loaded
executable uses the result, at least `S` bytes must remain live. These are lower
bounds, independent of cache implementation.

The reader single-flight path meets the bounds: one producer, one immutable
payload shared by all callers, cold latency `C + O(cache synchronization)`, and
`S` retained payload bytes. A warm hit runs no producer and makes no payload
copy. This is exact count optimality for this cost model, not a claim that no
alternative map or mutex implementation can improve a constant number of
nanoseconds.

The cache is reader-scoped and stores weak payload references. It therefore
does not extend a code object's lifetime: the entry expires when the last
active load or loader object releases its `shared_ptr`. Expired entries are
swept during successful publication. There is no size-based eviction policy or
cache arena to fragment; payload allocation remains COMGR's single output
allocation, and ordinary allocator fragmentation is outside the payload-byte
metric.

## Synchronization argument

All `ready`, `in_flight`, and mutable flight state is accessed under the cache
mutex. The miss linearization point is insertion of one flight under that
mutex, so at most one caller becomes producer for a key. The producer runs
without the mutex, allowing unrelated keys to progress concurrently. Success
or failure is published to the flight under the mutex before `notify_all`;
waiters use a predicate wait and copy that exact published result.

Each waiter owns a `shared_ptr` to its flight. A failed generation may be
removed and a retry may start before an old waiter wakes, but the old flight and
its result remain alive until those old waiters finish. Payloads are immutable
and shared by `shared_ptr`, while the ready map holds only `weak_ptr`, so one
caller releasing a result cannot invalidate another caller's view.

The TLC model checks these statements over all bounded interleavings,
including weak-entry expiry, successful and failed publication, and a retry
starting before old waiters wake. ThreadSanitizer and randomized stress test the
C++ memory synchronization. These checks establish safety for the modeled
state machine; they do not prove scheduler fairness or lock-freedom.

## Runtime invariants and fault injection

`hotswap_cache_test.cc` injects producer exceptions and `std::bad_alloc`, then
verifies typed errors, wakeup of every waiter, and successful retry. It also
checks same-key single-flight, different-key parallelism, exact-key separation,
weak ownership, and producer/ready/coalesced/wait/lock/live-byte metrics.

`hotswap_rewrite_test.cc` retains the ROCR load-path contract: optional rewrite
or rewritten-load failure falls back to the original code object; required A0
or strict rewrites fail closed; disable and non-candidate paths load the
original; and successful rewritten buffers remain owned through loader use.

The benchmark is a behavioral model rather than a checkout-and-build of each
PR. It models retarget cost separately from loader parsing and implements the
#8274 lookup-only miss race and per-hit output copy directly. It reports CSV
rows for cold and warm batches. `payload_bytes` is the rewritten ELF size;
`producer_us` is synthetic COMGR time.

```console
cmake --build <rocrtst-build> --target hotswap_cache_benchmark -j
<rocrtst-build>/hotswap_cache_benchmark \
  --threads=1,2,4,8 --bytes=65536,1048576,8388608 \
  --producer-us=2000 --iterations=5
```

Expected invariants for `N` concurrent loads and output size `S`:

| Strategy | Cold producers | Retained payload | Warm cache-hit copy |
|---|---:|---:|---:|
| no cache | `N` | `N * S` | `S` per load |
| #8274 copy cache | up to `N` | `(N + 1) * S` | `S` per load |
| reader single-flight | `1` | `S` | none |

The TLC model is bounded by `Clients`, `Keys`, and `MaxGenerations` in the
configuration file. Run it with a `tla2tools.jar` distribution:

```console
cd projects/rocr-runtime/rocrtst/suites/functional/hotswap/model
java -cp /path/to/tla2tools.jar tlc2.TLC -config ReaderRetargetCache.cfg \
  ReaderRetargetCache.tla
```

The checked invariants cover one leader per in-flight key, waiter/generation
association, success-only ready publication, and agreement among callers that
observe the same generation. The model permits a failure to start a new
generation before old waiters wake, matching the implementation's
`shared_ptr<RetargetFlight>` lifetime.

For ThreadSanitizer, configure a separate rocrtst build with both compile and
link flags so the isolated cache binary is instrumented:

```console
cmake -S projects/rocr-runtime/rocrtst/suites/test_common \
  -B build-rocrtst-hotswap-tsan \
  -DCMAKE_CXX_FLAGS="-O1 -g -fsanitize=thread -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" \
  <the same LLVM/ROCR dependency arguments as the normal rocrtst build>
cmake --build build-rocrtst-hotswap-tsan --target hotswap_cache -j
ctest --test-dir build-rocrtst-hotswap-tsan -R '^hotswap_cache$' --output-on-failure
```

GCC TSAN on WSL2 may abort before `main` with `unexpected memory mapping`.
For that host, add `-fno-pie` to `CMAKE_CXX_FLAGS`, add `-no-pie` to the
linker flags, and run the binary with address randomization disabled:

```console
setarch x86_64 -R env TSAN_OPTIONS=halt_on_error=1 \
  build-rocrtst-hotswap-tsan/hotswap_cache
```
