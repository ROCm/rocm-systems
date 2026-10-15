# ABCE test plan

ABCE is pure logic over plain memory: it validates a request, picks engines, decomposes it into SDMA
packets and publishes them to a ring. Nearly all of that can be tested without a GPU by emitting a
frame, walking it with `abce_decode.h`, and asserting the packet sequence and byte count. Only the
last tiers need hardware.

## Running

```
cmake -S runtimes/api-headers/tests/abce -B build-abce-tests -DCMAKE_BUILD_TYPE=Release
cmake --build build-abce-tests -j && ctest --test-dir build-abce-tests
```

Or pass `-DABCE_BUILD_TESTS=ON` to a parent project that `add_subdirectory`s `runtimes/api-headers`
(then run ctest on `<build>/.../api-headers`). Needs GTest and pthreads; use the `ctest` that matches
the `cmake` that configured the tree.

Sanitizers: `-DABCE_TESTS_SANITIZER=thread|address|undefined`. On this machine's kernel TSan needs
ASLR off: `setarch "$(uname -m)" -R ./abce_unit_tests`.

## Layout

| Path | Purpose |
|---|---|
| `harness/abce_test_harness.hpp` | ISA table, frame walker, fake ring (plain memory, instant or manual engine, mapped-words or queue-ops backend), `Mapper` (builder + orchestrator + rings) |
| `unit/packet_fields_test.cpp` | Packet bit-field regressions |
| `unit/validation_test.cpp` | Request validation and plan lifecycle |
| `unit/profiling_test.cpp` | Timestamp ordering for profiled copies |
| `unit/ring_test.cpp` | Ring protocol, both queue-control backends, concurrency |
| `unit/orchestrator_test.cpp` | Golden frames, placement, sizing, submit-vs-map |
| `CMakeLists.txt` | Test binary plus per-header C11/C++17 self-containment |

The harness keeps its expected per-ISA values (`swap_supported`, `is_gfx125plus`) independent of the
library so a library regression cannot also move the expectation.

## Tiers

### T0. Build hygiene (done)
Every public header compiles standalone as C11 and as C++17 with `-Wall -Wextra -Werror`
(`abce_cxx.h` C++ only; `abce_device.h` excluded, it needs a device compiler).

### T1. Packet fields (done)
Regression pins for bug classes already hit in ROCr's SDMA path:
64-bit fence/poll carry `sys` and system scope; 32-bit fence scope only on scope-field parts; fused
wait/signal blocks are system scope for every fused kind (linear, swap, multicast, indirect); the
indirect packet is compacted when wait/signal are absent and does not rely on a zeroed buffer.

Profiling (`unit/profiling_test.cpp`) pins the ordering ROCr fixed in ROCm/rocm-systems#12615: with a
timestamp surface set, no fused body signals completion inline (the gfx125+ signal block is gone),
the end timestamp follows every body (and the fan-in poll on fan-out), and the completion write is
the last packet; a timestamp alone re-introduces the start gate and epilogue on the gate-free
fan-out; completion is an atomic or a 64-bit fence depending on `device_atomic_support`.

### T2. Validation (done)
Batch-level arguments, per-op rejection with `failed_op` attribution, size/alignment/null checks, ISA
gating (indirect, swap on all seven ISAs), `NO_ENGINE`, coordination-word requirement for fan-out,
oversized requests (`OUT_OF_RANGE` at map vs failure at submit), plan lifecycle (submit once,
unmapped plan, failed map leaves rings untouched).

### T3. Ring protocol (done, both backends)
Publish/doorbell, zeroed acquire over dirty memory, attach seeding from an in-use queue, padded-size
rules, oversized request, wrap padding as a separate NOP region (README example), exact fill,
back-pressure, drain, 4-producer stress (monotone doorbell, no torn or reordered records).

### T4. Orchestrator frames (done)
README golden frames (gfx942 and gfx1250: single, back-to-back, multicast, gated and gate-free
fan-out), placement thresholds, and two ISA-wide invariants: **sizing matches emission** (every
frame decodes to exactly its reserved bytes) and **submit matches map** (ring bytes identical to
the mapped frames), both across gfx90a/942/950/1030/1100/1201/1250.

### T5. Known defects (done, disabled)
`KnownDefects.DISABLED_*` encodes a failing expectation for each open bug so a fix flips it on.
Currently: gfx90a large multicast with several engines returns `NO_LEGAL_ENGINE`.

### T6. Planned: property/fuzz (GPU-free)
- Randomised request batches over all ISAs; invariants: map either fails cleanly or yields a frame
  that decodes with `clean == true` and has exactly the reserved size; submit never writes outside
  its reservation; no UNKNOWN/TRUNCATED packets.
- Randomised ring schedules (sizes, record lengths, producer counts, engine pacing) with the TSan
  build; invariant: records appear once, in reservation order.
- libFuzzer on `abce_decode.h` against truncated/garbage streams: must terminate, never read past
  the end.

### T7. Planned: C++ and HSA wrappers (GPU-free)
`abce_cxx.h` RAII/ownership and error mapping; `abce_hsa.h` with a stubbed HSA table (queue
creation, signal handling, pinned-memory lock failure paths).

### T8. Planned: device header
Compile `abce_device.h` for gfx942 and gfx1250 (hipcc, device-only) and check it is
self-contained; run the device frame-builder against `abce_decode.h` through a HIP kernel writing
into host-visible memory. Needs a ROCm toolchain, optionally a GPU.

### T9. Planned: hardware (needs a GPU, separate target)
Round-trip data checks per op kind and size class (H2D, D2H, D2D, fill, swap, multicast, rect,
indirect); multi-queue fan-out; wrap under real back-pressure; dependency signals; error paths
(bad address must fault the queue, not hang). Run on gfx942 and gfx1250 where available.

## Mutation checks (manual, before landing test changes)
Apply each to a scratch copy of `include/`, rebuild, expect red:

| Mutation | Caught by |
|---|---|
| drop `wait_scope` on a fused wait | `PacketFields.Fused*` |
| drop `sys` on 64-bit fence / poll | `PacketFields.Fence64*`, `Poll64*` |
| always emit the wait block in the indirect builder | `PacketFields.IndirectCopyIsCompacted*` |
| drop swap alignment / support check | `Validation.SwapIsGated*` |
| drop indirect gfx125 gate | `Validation.IndirectCopyNeedsGfx125Plus` |
| `bytes >= size` -> `bytes > size` in reserve | `Ring.ARequestAtOrAbove*` (hangs; times out) |
| fused size `1+6` -> `1+5` | `SizingMatchesEmission/gfx1250`, golden frames |
| drop ring zeroing | `Ring.AcquiredRegionIsZeroed*` |
| timestamps no longer force an epilogue (inline fused signal returns) | `Profiling*/gfx1250`, `ProfilingGfx125Plus.*` |
| end timestamp emitted after the completion write | `Profiling.*` (all ISAs) |
| timestamps no longer count as gate-requiring ordering work | `ProfilingGfx125Plus.ATimestampAlone*` |

In debug builds the builder asserts also fire for the swap-alignment mutation; in release builds
(asserts compiled out) only the validation test catches it.

## Out of scope
Hardware ordering of the read pointer (loaded relaxed by design: the engine is hardware). TSan runs
model it with `__tsan_acquire/__tsan_release` in the harness; the raw mapped-words concurrency test
is skipped under TSan because the library cannot carry those annotations.
