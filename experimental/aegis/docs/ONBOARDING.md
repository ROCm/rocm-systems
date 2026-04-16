# AegisBit Onboarding (Read First)

Fast-start reference for anyone (human or AI agent) picking up work on
AegisBit. Scope: vocabulary, conceptual model, key data types,
build/test commands, common task recipes, and non-obvious invariants.

For the layered architecture map, component composition diagrams, and
shutdown ordering rationale, see `docs/ARCHITECTURE.md`. For deferred
follow-ups, see `docs/future-work.md`.

---

## 1. TL;DR (30-second version)

AegisBit is a HIP/HSA profiler that attaches via `LD_PRELOAD`,
intercepts every kernel dispatch, rewrites the dispatched GPU code
object to capture memory-access metadata per instrumented site, and
emits per-kernel VMEM coalescing + LDS bank-conflict summaries to JSON
on process exit.

Entry points:

- `src/intercept/HSAInterceptor.cpp` — LD_PRELOAD of `hsa_*` functions.
- `src/intercept/DispatchInterceptor.cpp` — hooks queue writes, calls
  `TracingEngine::onDispatch` with each AQL packet.
- `src/runtime/TracingEngine.cpp` — singleton coordinator for the
  whole pipeline.
- `src/runtime/KernelPatcher.cpp` — produces a patched ELF for any
  given (kernel, mode, trace config).

---

## 2. Build, test, run

### Build

```bash
./build.sh configure   # one-time (cmake)
./build.sh build       # incremental rebuild
./build.sh clean       # rm -rf build/
```

Under the hood: standard CMake + Ninja/Make under `build/`. After
`configure` you can rebuild directly with:

```bash
cmake --build build -j$(nproc)
```

Requires a nearby LLVM build (looks for `../llvm-project-amd-staging/build`
or `../llvm-project/build`).

### Test

```bash
./build.sh test        # build + ctest (unit + integration)
./build.sh e2e         # Python-driven end-to-end (GPU required)
./build.sh test-all    # both
```

Raw `ctest` for faster iteration:

```bash
cd build
ctest --output-on-failure                     # all
ctest -R TrampolineBridgeGTest                # one
ctest -E 'HSAPoolGTest|HSAPoolGPUGTest|ZeroSGPRSpillGPUTest' # skip known-bad
```

### Known pre-existing test failures (ignore these)

On hosts without a fully configured MI350X the following **always** fail
and are unrelated to current work:

- `HSAPoolGTest`
- `HSAPoolGPUGTest`
- `ZeroSGPRSpillGPUTest`

They failed on `main` before the modularity refactor started and
continue to fail identically; root cause is environmental (no kernarg
pool reported / `named symbol not found` on MI350X). **Do not chase
them unless you own the GPU host config.**

Everything else should pass; if a previously-passing test regresses,
that's on you.

### Run the profiler on a workload

```bash
AEGISBIT_LOG=1 \
LD_PRELOAD=build/lib/libaegisbit.so \
./your_hip_program
```

Output JSON is controlled by `AEGISBIT_OUTPUT_DIR` (see
`src/runtime/RuntimeConfig.cpp` for the full list of env vars; every
one is declared in `include/aegisbit/RuntimeConfig.h` under `DebugFlags`
or `TransformFlags`).

---

## 3. Glossary

### GPU / AMD-specific

| Term | Meaning |
|---|---|
| CDNA / gfx942 / gfx950 | AMD compute GPU architectures we target (MI300X, MI350X). |
| VALU / SALU | Vector / Scalar ALU instructions. |
| VMEM | Vector memory instructions (global/buffer loads + stores). |
| SMEM | Scalar memory instructions. |
| LDS | Local Data Share — on-chip shared memory, accessed with `ds_*` instructions. |
| VGPR / SGPR | Vector / Scalar general-purpose registers. |
| AccVGPR | Accumulator VGPRs used by MFMA (matrix FMA) instructions. |
| MFMA | Matrix Fused Multiply-Add. |
| Wave / Wavefront | 64-lane SIMD execution unit (on gfx942/gfx950). |
| Kernarg | Kernel arguments — passed to the GPU via a separate buffer. |
| Kernel descriptor (KD) | 64-byte `.rodata` struct describing resources + entry offset. |
| AQL packet | Architecture Queueing Language dispatch packet (HSA queue entry). |
| HSA | Heterogeneous System Architecture — AMD's low-level GPU runtime API. |
| Scratch | Per-workitem private memory (backed by VRAM). |

### DBI / AegisBit-specific

| Term | Meaning |
|---|---|
| Site | An instrumentation point — one VMEM or LDS instruction in the kernel we want to observe. See `InstrumentationSite` / `SiteInfo`. |
| Trampoline | Injected code that replaces an instruction with a jump to a payload, which samples metadata then jumps back. |
| Island | A contiguous region of `.text` holding trampoline code. Multiple islands per kernel when trampolines can't reach the site with a short jump. |
| Payload | The metadata-capture code executed at each site (VMEM reduction, LDS bank-conflict count, or full address dump). |
| Relay stub | A tiny "double-hop" trampoline used when the payload body is too far away for a direct `s_branch`. Site → relay → body → site. |
| Direct body | The alternative: the payload body inlined right next to the site. |
| Shared body | One copy of the payload body used by many sites, reached via `s_call_b64` (link-register call) or `s_swappc_b64` (swap-PC). |
| Long jump | 64-bit PC-relative jump sequence used when `s_branch`'s ±1MB range is insufficient. |
| OnGpuReduce | Payload strategy that reduces per-wave metadata on-GPU and writes small per-site counters. |
| FullCapture | Payload strategy that dumps every lane's address to a big ring buffer. |
| Re-entrancy guard | `thread_local bool InsideAegisBit` in `onDispatch` that prevents our own patched-kernel dispatches from being patched again. |

---

## 4. Conceptual model

### What gets profiled

Every HSA kernel dispatch the process issues goes through
`TracingEngine::onDispatch`. A dispatch is **traced** when all of:

- `RuntimeConfig::shouldTraceKernel(name)` returns true (filter).
- Kernel name doesn't start with `__amd_rocclr_` (internal HIP kernels
  — skipped to avoid re-entrancy).
- Kernel hasn't previously failed to patch (in `FailedKernels` set).
- The re-entrancy guard isn't set (dispatch didn't come from our own
  patched kernel launching a child kernel).

Traced dispatches get:

1. A patched ELF built once (cached in `KernelPatcher::PatchedCache`).
2. A loaded HSA kernel handle (cached in `LoadedKernelCache`).
3. A persistent GPU-visible trace buffer (cached in
   `PersistentBufferCache`, one per kernel name).
4. `packet->kernel_object` swapped to the patched kernel.
5. `packet->completion_signal` swapped to our signal (unless
   `AEGISBIT_SKIP_SIGNAL` is set).
6. An `ActiveDispatch` record in `DispatchRegistry`, monitored by
   `SignalMonitor`.

### Mode × Strategy matrix

Today there is exactly **one** `InstrumentationMode`:

- `MEMORY_ONLY` — per-site memory-access profiler.

Within that mode, the trace payload has two `PayloadStrategy`
variants:

| Strategy | What the buffer contains | When it's used |
|---|---|---|
| `OnGpuReduce` (default) | Per-site 8-byte counters. For VMEM: `{total_cache_lines_touched, total_samples}`. For LDS: `{total_unique_banks, total_samples}`. Aggregated on-GPU with wave intrinsics. | Small memory footprint, fast. Selected unless `AEGISBIT_STRATEGY=full_capture`. |
| `FullCapture` | `MemoryTraceRecord[]` — per-wave, per-site records each with 64 per-lane addresses. | Debugging / ground-truth. Much higher memory + bandwidth overhead. Set via `AEGISBIT_STRATEGY=full_capture`. |

The strategy drives:

- Trampoline payload codegen (`TrampolineEmitter::emitBodyForPath` selects
  the `OnGpuReduce` vs. `FullCapture` body).
- Buffer interpretation in `ProfilingResultsSink::ingest` — the two
  strategies have completely different buffer layouts and analyzer
  entry points (`analyzeOnGpuCounters*` vs. `analyzeBuffer` +
  `analyzeLDSBuffer`).

When editing the analysis or codegen path, **always check which branch
you're in**. It's easy to break one and have the other's tests still
pass.

### Trampoline strategies (orthogonal dimension)

Independent of `PayloadStrategy`, there is a **trampoline strategy**
that chooses how to reach the payload from each site. Three concrete
strategies live under `src/transform/`:

| Strategy | Jump mechanism | When it's chosen |
|---|---|---|
| `SharedBodyStrategy` | `s_call_b64` + shared body | Default for gfx942 without free SGPR pair. |
| `SwapPCSharedBodyStrategy` | `s_swappc_b64` + shared body | When there's a free SGPR pair for the link and the body fits. |
| `AdaptiveStrategy` | Per-site decision between direct body and relay stub | When island placement is tight; handles overflow site-by-site. |

Choice is made by `KernelPatcher::maybeUpgradeToSwapPC` + the
`AEGISBIT_STRATEGY` / `AEGISBIT_FORCE_SWAPPC` / `AEGISBIT_FORCE_RELAY`
env knobs.

---

## 5. Core data types cheat-sheet

All live under `include/aegisbit/`. Jump straight to the header for
the authoritative field list.

### Interception / capture (`DispatchInterceptor.h`)

| Type | Purpose | Key fields |
|---|---|---|
| `CapturedCodeObject` | ELF bytes + metadata captured when an HSA code object is loaded. | `CodeObjectId`, `Bytes`, `URI`, `LoadBase`. |
| `CapturedKernelSymbol` | One kernel within a code object; captured when a symbol resolves. | `KernelId`, `CodeObjectId`, `KernelName`, `KernelObject`, `KernargSegmentSize`. |

### Instrumentation input (`Types.h`, `CoalescingAnalyzer.h`)

| Type | Purpose | Key fields |
|---|---|---|
| `DecodedInstruction` | One disassembled instruction. | `Inst` (MCInst), `Address`, `Size`, `Category` (VMEM/LDS/…). |
| `BasicBlock` / `ControlFlowGraph` | CFG. | `Instructions`, `Successors`, `Predecessors`; CFG has O(1) block lookup via `BlockIndex`. |
| `InstrumentationSite` | A chosen site to instrument. | `Address`, `Offset`, `OrigInst`, `IsLoad`, `IsGlobal`, `AddrVGPRIndex`, `Addr64`, LDS offsets, pre-spill wait counts. |
| `SiteInfo` | Post-patch per-site metadata used by the analyzer + JSON output. | `SiteID`, `PC`, `InstrName`, `IsLoad`, `ElemSize`, `IsLDS`, `DSOffset0/1`, `SourceFile`, `Line`. |
| `InstrumentationMode` | enum: `MEMORY_ONLY`. | — |
| `PayloadStrategy` | enum: `OnGpuReduce`, `FullCapture`. | — |
| `TraceConfig` | Trace buffer + counter addresses baked into the trampoline as immediates. | `BufferAddr`, `CounterAddr`, `BufferSize`, `Strategy`, `SupportsGPUAtomics`; static `RecordSize=520`, `CounterSize=8`. |

### Patching output (`KernelPatcher.h`, `TrampolineTypes.h`)

| Type | Purpose | Key fields |
|---|---|---|
| `PatchedKernel` | Output of `KernelPatcher::getOrPatch`. | `PatchedELF`, `KernelObject`, `KernelName`, `GPUArch`, `AdditionalVGPRs`/`SGPRs`/`ScratchBytes`, `Trace`, `Plan`, `SiteMap`, `NumMemorySites`. |
| `InstrumentationPlan` | Typed summary of how this kernel will be instrumented. | Produced by `InstrumentationPlanner`; selects strategy, reports register usage. |
| `TrampolineSlot` | One site's patch — displaced bytes + trampoline bytes. | `OriginalPC`, `PatchBytes`, `TrampolineBytes`, `DisplacedSize`, `UsedLongJump`. |
| `TrampolineIsland` | Contiguous block of trampoline code at a specific `.text` offset. | `Bytes`, `Offset`. |
| `BridgeResult` | All slots + islands + prologue bytes for a kernel. | `Slots`, `Islands`, `PatchedCount`, `LongJumpCount`, `PrologueBytes`. |
| `ScratchRegisters` | Register allocation decision for the trampoline prologue. | Return-address SGPR pair, scratch VGPR, AccVGPR spill config. |
| `PatchCacheKey` | Cache key for patched kernels. | `CodeObjectId`, `KernelId`, `Mode`. |

### Launch + runtime (`KernelLauncher.h`, `ActiveDispatch.h`, `PersistentBufferCache.h`)

| Type | Purpose | Key fields |
|---|---|---|
| `LoadedKernel` | HSA handles for a loaded patched ELF. | `CodeObjectHandle`, `ExecutableHandle`, `KernelSymbol`, `KernelName`. |
| `DispatchParams` | Captured from the AQL packet. | Workgroup + grid sizes per dim. |
| `ActiveDispatch` | In-flight dispatch we're monitoring. | `DispatchID`, `KernelName`, `Params`, `StartTime`, `OriginalKernelObject`, `PatchedKernelObject`, `CompletionSignalHandle`, `OriginalSignalHandle`, `SiteMap`. |
| `PersistentTraceBuffer` | GPU-visible buffer owned by `PersistentBufferCache`. | `BufferPtr`, `CounterPtr`, `BufferSize`, `Config` (TraceConfig with addresses). |

### Analysis output (`CoalescingAnalyzer.h`)

| Type | Purpose |
|---|---|
| `CoalescingSummary` | Per-kernel VMEM summary (sites, overall efficiency, per-site metrics). Written to JSON. |
| `LDSSummary` | Per-kernel LDS bank-conflict summary. |

### Configuration (`RuntimeConfig.h`)

| Type | Purpose |
|---|---|
| `DebugFlags` | Dumps, logging, site caps, diagnostics — read-only behavior. |
| `TransformFlags` | Codegen knobs — affect patched output (relay choice, body jump, strategy forcing). |
| `RuntimeConfig` | Top-level singleton, holds `Debug`, `Transform`, `Mode`, kernel-name filter, output dir. |

---

## 6. Common task recipes

### Add a new `AEGISBIT_*` environment variable

1. Declare the field in `DebugFlags` or `TransformFlags`
   (`include/aegisbit/RuntimeConfig.h`). Pick the struct by whether
   the flag changes codegen (`Transform`) or is diagnostic-only (`Debug`).
2. Parse it once in `RuntimeConfig::initialize()`
   (`src/runtime/RuntimeConfig.cpp`). Reuse `parseBool` /
   `parseIntOrDefault` helpers already there.
3. Read at call sites as `RuntimeConfig::getInstance().Debug.X` or
   `.Transform.X`. Never `::getenv` at a call site.
4. If you want to test both branches, use `RuntimeConfigOverride` in
   `test/unit/support/` (see existing usage in
   `test/unit/TrampolineBridgeGTest.cpp`).
5. Document it in the header comment for the new field and in
   `RuntimeConfig.cpp`'s parse function.

### Add a new trampoline strategy

1. Implement the `TrampolineStrategy` interface (see
   `src/transform/TrampolineStrategy.h`) in a new file under
   `src/transform/`.
2. Add the source to `src/CMakeLists.txt`.
3. Extend `TrampolineBridge::createStrategyForPlan` (and the
   `InstrumentationPlanner`) to pick it when appropriate.
4. Reuse `BridgeHelpers` for island alignment, dispatch-table
   construction, etc.
5. Add a gtest to `test/unit/TrampolineBridgeGTest.cpp` with synthetic
   `InstrumentationSite` arrays.

### Add a new HSA call path

1. Today every HSA call is made directly by `HSAPoolManager`,
   `TraceBufferAllocator`, `KernargPool`, `SignalMonitor`, or
   `TracingEngine::launchProfilerDispatch`/`cleanupDispatch`. Add it
   to whichever owns the relevant responsibility.
2. Check `SignalMonitor` / `TracingEngine::finalize` shutdown
   ordering — if your call runs during `__cxa_finalize`, it **must not
   call back into HSA** (see "Gotchas", below).
3. Once `IHSARuntime` lands (`docs/future-work.md`) this workflow
   becomes "add a method to the interface" instead.

### Add a new per-site metric

1. Extend `SiteInfo` (`CoalescingAnalyzer.h`) with the new field.
2. Extend `CoalescingSummary` / `LDSSummary` so `writeJSON` emits it.
3. Populate in the analyzer:
   - `OnGpuReduce`: `CoalescingAnalyzer::analyzeOnGpuCounters*`.
   - `FullCapture`: `CoalescingAnalyzer::analyzeBuffer` /
     `analyzeLDSBuffer`.
4. If it needs on-GPU collection, extend the payload codegen in
   `src/codegen/PayloadCompiler.cpp` or the emitter in
   `src/transform/TrampolineEmitter.cpp` (different paths for
   OnGpuReduce vs. FullCapture; don't forget both).

### Add a new unit test

Recorded ELF fixtures live in `test/unit/fixtures/`. Load them with
`CodeObjectHandler::loadFromBytes` to exercise `codeobj` / `disasm` /
`analysis` code without a GPU. See
`test/unit/KernelPatcherGTest.cpp` for the canonical pattern.

For HSA-touching code today there is no mock — either make the code
HSA-free (refactor the HSA call out) or drop the test under
`test/integration/` with a `gpu` CTest label (gets skipped when no GPU
is available). The `IHSARuntime` refactor (`docs/future-work.md`)
will remove this friction.

### Investigate a new kernel regression

1. Set `AEGISBIT_LOG=2 AEGISBIT_DUMP_ELF=/tmp/patched.elf
   AEGISBIT_DUMP_INPUT_ELF=/tmp/input` and rerun.
2. Diff the input vs. patched ELFs:
   - `llvm-objdump --disassemble-symbols=<kernel>.kd /tmp/patched.elf`.
3. Look at the `SiteInfo` array for the kernel in the JSON output —
   compare with the disassembly to see which site corresponds to
   which instruction.
4. If patching failed, the kernel is in `FailedKernels`; the
   `AEGISBIT_LOG` output will tell you which stage (`planInstrumentation`,
   `buildBridge`, `assemblePatchedText`) raised the error.
5. `docs/body-island-bug-investigation.md` and
   `docs/investigation-zerosGPR-flash-attn-errors.md` are worth
   reading before starting a hard investigation — most patched-kernel
   bugs are variations on those two themes.

---

## 7. Gotchas / invariants

These are things the code depends on that aren't obvious from
structure alone. Breaking them is how you corrupt GPU state or glibc
heap.

### Never call HSA during `__cxa_finalize`

When `TracingEngine::finalize()` runs via the LD_PRELOAD dtor, the HSA
runtime is partway through its own teardown. Calls like
`hsa_amd_memory_pool_free`, `hsa_executable_destroy`, etc. corrupt
glibc heap metadata and segfault on process exit. Therefore:

- `LoadedKernelCache::clear()` only drops handles; does not call
  `hsa_executable_destroy`.
- `PersistentBufferCache::clear()` only drops pointers; does not call
  `hsa_amd_memory_pool_free`.
- `KernargPool::clear()` ditto.
- `cleanupDispatch(d, DuringFinalize=true)` skips `hsa_signal_destroy`.

The OS reclaims all GPU allocations at process exit. This is an
**intentional leak**, not a bug.

### Shutdown ordering is exact

`finalize()` order must stay:

1. `Kernargs.joinInit()` — background thread finishes *before* we tear
   anything down.
2. `SignalMon->stop()` — monitor thread finishes *before* we drain
   dispatches.
3. `Dispatches.drainAll()` → `cleanupDispatch(DuringFinalize=true)`.
4. `KernelCache.clear()` / `PersistBuffers.clear()` / `Kernargs.clear()`.
5. `Results.flush(path)`.

Reordering these corrupts state. The reasons are commented in
`TracingEngine::finalize()`; don't short-circuit them.

### The re-entrancy guard is load-bearing

`TracingEngine::onDispatch` starts with a `thread_local bool
InsideAegisBit` guard. Without it: if our patched kernel launches a
child kernel (some library kernels do), that child hits `onDispatch`,
gets patched, launches a grandchild, and so on. **Don't remove it**,
and when adding any new dispatch-path call, make sure you go through
`onDispatch` not around it.

### Buffer addresses are baked as immediates

The instrumented trampoline encodes `TraceConfig::BufferAddr` and
`CounterAddr` as 64-bit immediates in the patched code. Therefore:

- The buffer can't be reallocated during a kernel's lifetime.
- `PersistentBufferCache` reuses one buffer per kernel across
  dispatches; the counter is reset per dispatch but the buffer
  pointer is stable.
- **Don't extend kernargs** on the profiler fast path — the original
  kernarg passes through unchanged. If you need to pass data from CPU,
  use immediates or a side-channel buffer.

### LDS addresses are 32-bit, VMEM addresses are 64-bit

`InstrumentationSite::Addr64` + `InstrumentationSite::IsGlobal`
disambiguate. `GLOBAL_*` with `saddr=off` uses a `VReg_64` pair (the
VGPR pair `v[Idx : Idx+1]`). `GLOBAL_*` with `saddr` uses a single
32-bit offset VGPR. `DS_*` uses a single 32-bit VGPR plus constant
offsets `DSOffset0`/`DSOffset1`. Getting this wrong in the payload
codegen produces addresses that look plausible but are off by a
factor of 2 or wildly wrong.

### `AccVGPR` spill has three fallback paths

If the kernel uses all arch-VGPRs for MFMA, the trampoline needs to
spill an AccVGPR to scratch. `ScratchRegisters::setupAccVGPRSpill`
tries three fallbacks in order; do not reorder them without
understanding
`docs/investigation-zerosGPR-flash-attn-errors.md`.

### Pre-kernel padding matters for long jumps

Long-jump islands are placed before the kernel entry point. If the
kernel descriptor's `KernelCodeEntryByteOffset` isn't adjusted in
lockstep (`DescriptorUpdater::applyPreKernelPadding`), the GPU jumps
into the middle of an island on entry. Always keep
`applyPreKernelPadding` in sync with `maybeUpgradeToSwapPC` and
`buildBridge`.

### Signal forwarding vs. `AEGISBIT_SKIP_SIGNAL`

Normally we swap the dispatch's completion signal with our own, then
forward to the original signal in `cleanupDispatch` after analysis.
`AEGISBIT_SKIP_SIGNAL=1` disables the swap — the app's signal fires
directly and we lose the completion callback, so no analysis happens.
This exists to isolate patched-code bugs from signal-forwarding bugs
during debugging. **Not a production flag.**

---

## 8. Navigation index

### "Where does X happen?"

| Question | File(s) |
|---|---|
| How does an HSA call reach us? | `src/intercept/HSAInterceptor.cpp` → `src/intercept/DispatchInterceptor.cpp` |
| Where is the re-entrancy guard? | `TracingEngine::onDispatch` (top of function) |
| Where is the kernel filtered? | `RuntimeConfig::shouldTraceKernel` + inline check for `__amd_rocclr_` |
| How is the patched ELF built? | `KernelPatcher::patchKernel` stages → `TrampolineBridge` → strategy `build()` |
| Where are sites chosen? | `src/analysis/SiteAnalyzer.cpp` |
| Where is scratch allocated? | `src/analysis/ScratchRegisters.cpp` (`fromDescriptor*`, `refineScratchVGPRs`, `setupAccVGPRSpill`) |
| Where is the patched kernel loaded? | `src/launch/KernelLauncher.cpp` (wrapped by `LoadedKernelCache`) |
| Where is the trace buffer allocated? | `src/runtime/TraceBufferAllocator.cpp` (pools from `HSAPoolManager`) |
| Where does a dispatch become "active"? | `TracingEngine::launchProfilerDispatch` → `DispatchRegistry::insert` |
| How do we notice a dispatch finished? | `src/runtime/SignalMonitor.cpp` polls; fires `CompletionCallback` on engine |
| Where is per-dispatch analysis? | `ProfilingResultsSink::ingest` |
| Where is final JSON written? | `ProfilingResultsSink::flush` → `CoalescingAnalyzer::writeJSON` |
| Where is an env var parsed? | `src/runtime/RuntimeConfig.cpp` — `RuntimeConfig::initialize` |

### Files by line count (rough complexity signal)

Top 10 by size (see `ARCHITECTURE.md` §2 for layer assignment):

```
2178  src/transform/TrampolineEmitter.cpp   — ISA-level codegen
 926  src/analysis/CoalescingAnalyzer.cpp   — VMEM/LDS analysis + JSON writer
 848  src/disasm/InstructionBuilder.cpp     — MC-layer instruction assembly
 683  src/runtime/KernelPatcher.cpp         — patch pipeline orchestration
 622  src/runtime/TracingEngine.cpp         — dispatch coordinator
 588  src/intercept/DispatchInterceptor.cpp — AQL packet interception
 583  src/transform/RelayEmitter.cpp        — relay stub construction
 542  src/intercept/HSAInterceptor.cpp      — LD_PRELOAD + HSA hooks
 433  src/codegen/PayloadCompiler.cpp       — payload snippet codegen
 430  src/analysis/SiteAnalyzer.cpp         — site discovery
```

Below these, each file is well under 400 lines and maps 1:1 with its
responsibility.

---

## 9. What this doc doesn't cover

- ISA-level details of the trampoline emitter (see
  `docs/PAD_ISLAND_ARCHITECTURE.md` and the source comments in
  `TrampolineEmitter.cpp`).
- Concrete coalescing metric definitions (see the doc comments at the
  top of `CoalescingAnalyzer.h` and `CoalescingAnalyzer.cpp`).
- Hardware-level MFMA / AccVGPR semantics — learn these from the
  CDNA ISA manual when needed.
- User-facing CLI / Python tooling — see `README.md` and `test/run_e2e.py`.
- History of past bugs — `docs/body-island-bug-investigation.md`,
  `docs/investigation-zerosGPR-flash-attn-errors.md`. Read before
  touching the patcher.
