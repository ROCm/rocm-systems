# Concurrent dispatch fuzzer

`rj_concurrent_dispatch_fuzz` sends an input-carried kernel through a real AQL queue. Its input
header is:

- byte 0: GPU target from the execution-fuzzer registry;
- byte 1: cache priming, barrier, acquire, release, and XCD-fan-out bits 0 through 4;
- bytes 2 through 5: workgroup count, workgroup size, packet count, and queue owner;
- byte 6: random-generator kernel-class choice (not consumed by replay);
- byte 7: number of cache lines to prime;
- bytes 8 through 15: reserved.

The remaining bytes are the raw kernel body. The harness appends a target-specific `s_endpgm` and
bounds simulation time; decode rejection, stopped waves, and programs that exhaust that budget are
ordinary outcomes. A bounded wave address space exposes only the kernel and 64 KiB data island, so
raw kernels cannot overwrite the AQL ring, completion signal, or other harness-owned state.

Representative multi-XCD targets use two XCDs. Other targets use one XCD, and the engine partition
count is always capped by the topology's actual XCD count. Each XCD has three CUs so the pooled mode
really creates three shared CPU dispatch workers even on a one-XCD target; the harness rejects a
configuration that silently caps that pool. A wave-dispatch observer initializes several scalar and
vector register pairs with addresses in a shared 64 KiB data island so arbitrary memory instructions
have a useful path into L1/L2 and backing memory.

Multi-XCD inputs are run three ways:

1. one simulation thread and one CPU dispatch worker;
2. up to one simulation thread per XCD and one CPU dispatch worker;
3. up to one simulation thread per XCD and three shared CPU dispatch workers.

Single-XCD inputs omit the identical second mode and run the distinct serial and pooled modes.

The target checks queue, signal, workgroup, and XCD-dispatch invariants in every execution. Its
canonical target-specific kernel copies the kernarg address, performs a flat atomic add, waits for
the memory operation, and has a strong differential oracle: every mode must produce the exact
expected data image and identical instruction, wave, completion, and flushed-memory results. A
kernel gets this oracle only when its program bytes exactly match the target's canonical program.
Random campaigns regularly emit that program so target, topology, dispatch, fence, and cache
controls remain fuzzed under the oracle.

The harness disables simulator wait diagnostics so arbitrary instruction bytes remain ordinary fuzz
outcomes. The exact image oracle therefore cannot independently detect a canonical kernel that
waits on the wrong counter; the seed regression pins the complete atomic-and-wait suffix to
independently assembled LLVM instruction words for every target.

The canonical atomic checks shared-data visibility and completion across modes, but it is not an
independent acquire/release-fence oracle. Cache priming starts from backing memory that already
contains the same bytes, the atomic updates backing through the L2 path, and the final image is read
after a harness flush. Acquire and release controls therefore remain covered by crash, sanitizer,
bookkeeping, and cross-mode checks; a stale-load/publication kernel would be needed to verify their
cache semantics directly.

Other inputs carry arbitrary target-specific instruction bytes after the same useful cache prefix.
Those kernels retain crash, sanitizer, bookkeeping, and cross-mode scheduling-outcome oracles, but
deliberately skip the memory comparison: mixed atomics and other racy GPU programs can legitimately
produce different final bytes when CPU dispatch order changes. Modes must agree on whether a
terminal dispatch fault occurred, but post-fault retirement is ignored because in-flight pooled
work can still change the completed-workgroup count and publish another packet's completion. The
faulting packet itself must remain pending, and a fault must not strand an active engine primary.
When every mode completes, modes agree on signal and workgroup retirement. If any mode stalls or
reaches the tick budget, raw-kernel progress is not compared: racy data-dependent control flow can
legitimately change whether a mode finishes and how many workgroups retire first. After a fatal
exit, only terminal-fault, primary-lifecycle, and structured exit classification are compared for
the same in-flight-work reason. Stopped, quiescent, and budget-limited programs remain ordinary
coverage outcomes. A racy raw program that changes whether a fault occurs across scheduling modes
is reported for triage.

## Build and test

From this directory's project root (`emulation/rocjitsu`):

```sh
cmake -S . -B build-concurrent-dispatch -GNinja -DBUILD_TESTING=ON
ninja -C build-concurrent-dispatch rj_concurrent_dispatch_fuzz rj_concurrent_dispatch_fuzz_test
ctest --test-dir build-concurrent-dispatch -j4 \
  -R '^(ConcurrentDispatchFuzzTest\.|ConcurrentDispatchFuzzCliTest$)' --output-on-failure

build-concurrent-dispatch/fuzz/concurrent_dispatch/rj_concurrent_dispatch_fuzz \
  --random 1000 --seed 0 --output findings
```

## AFL++ with sanitizers

Deterministic random inputs reuse the execution fuzzer's target-specific decoded-instruction bias.
Configure the whole simulator with the AFL++ compiler wrappers so coverage reaches the execution
backend, then emit the structured corpus and mutate the raw body directly:

```sh
AFL_ROOT=/path/to/AFLplusplus
cmake -S . -B build-concurrent-dispatch-afl -GNinja \
  -DCMAKE_C_COMPILER="$AFL_ROOT/afl-clang-fast" \
  -DCMAKE_CXX_COMPILER="$AFL_ROOT/afl-clang-fast++" \
  -DBUILD_TESTING=ON \
  -DRJ_ENABLE_ASAN=ON -DRJ_ENABLE_UBSAN=ON
ninja -C build-concurrent-dispatch-afl rj_concurrent_dispatch_fuzz

build-concurrent-dispatch-afl/fuzz/concurrent_dispatch/rj_concurrent_dispatch_fuzz \
  --emit-seeds concurrent-dispatch-seeds
"$AFL_ROOT/afl-fuzz" -i concurrent-dispatch-seeds -o concurrent-dispatch-findings \
  -m none -g 1 -G 1024 -- \
  build-concurrent-dispatch-afl/fuzz/concurrent_dispatch/rj_concurrent_dispatch_fuzz --afl
```

Selector 3 exposes the same target through `rj_fuzz`.

## ThreadSanitizer

Build the focused target with ThreadSanitizer to detect host races in engine partitions, the CPU
dispatch pool, cache state, and completion paths:

```sh
cmake -S . -B build-concurrent-dispatch-tsan -GNinja -DRJ_ENABLE_TSAN=ON
ninja -C build-concurrent-dispatch-tsan \
  rj_concurrent_dispatch_fuzz rj_concurrent_dispatch_fuzz_test
ctest --test-dir build-concurrent-dispatch-tsan -j4 \
  -R '^ConcurrentDispatchFuzzTest\.' --output-on-failure
```
