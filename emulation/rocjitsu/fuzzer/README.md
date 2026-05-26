# RocFuzz: rocjitsu fuzzer intercept

Coverage-guided fuzzing repeatedly mutates inputs, runs the target, and keeps
inputs that expose new execution behavior. For GPU fuzzing, the underlying
infrastructure has to collect CFG branch or edge information from device
execution into memory that the fuzzing engine can observe, so the engine can use
that feedback to explore program state space. This prototype uses rocjitsu to
patch supported AMDGPU code objects and merges device counters into AFL++'s
coverage map.

The product goal is to fuzz host and device code across ROCm libraries and user
applications, including prebuilt kernels that cannot use compiler-generated
device coverage. Writing every ROCm library fuzz target is a separate effort;
this tree is about proving that the core intercept, code-object rewriting, and
harness workflow can work end to end.

## Executive Summary

This prototype demonstrates that ROCm device-side branch coverage can be
collected from prebuilt AMDGPU code objects without recompiling the target
library. It does that by using rocjitsu as a dynamic binary instrumentation
substrate: intercept loader and launch APIs, classify the code object, decode
and patch AMDGCN text, run the patched kernel, and merge device counters into
AFL++ feedback.

The production question is no longer whether the idea is plausible. The hard
part is turning a defensive, report-heavy prototype into reusable rocjitsu DBI
infrastructure with clear support boundaries. The largest investments are
loader/container coverage, transactional code-object mutation, relocation and
special-state preservation, resource-aware probe planning, stable device-state
handoff, concurrency/lifetime hardening, and CI/performance validation across a
declared gfxip matrix.

The same infrastructure would benefit rocjitsu beyond fuzzing. A product-grade
fuzzer forces rocjitsu to own real-world code-object ingestion, safe instruction
insertion, CFG/liveness quality, descriptor/resource mutation, loader lifetime
models, and hardware-backed regression tests. Those are foundational services
for any future rocjitsu DBI or DBT client, not fuzzer-only plumbing.

For the productization framing, MVP scope, current evidence snapshot, and
risk criteria, see the [productization readiness note](docs/productization-readiness.md).

## What This Proves

On a limited but representative set of loader and kernel paths, the preload can:

- patch raw AMDGPU ELF/HSACO images loaded through HIP module APIs;
- parse clang offload bundles, HIP fatbin wrappers, and CCOB (`.co`) code-object
  containers;
- patch lower HSA code-object readers used by compiler-registered HIP fatbins
  and KPACK-backed ROCm library payloads;
- lazily patch large CCOB HIP modules after `hipModuleGetFunction` identifies
  the kernel being launched;
- redirect later HIP runtime launches through per-kernel shadow modules cached
  from the active fatbin/KPACK registration;
- merge device counters into AFL after common synchronization points and
  explicit persistent-mode boundaries.

The main conclusion is that AFL++ plus rocjitsu DBI is a workable shape for
ROCm host/device fuzzing when the interesting kernels are prebuilt and cannot
rely on compiler-generated device coverage. The default coverage policy should
stay adaptive: prefer previous-BB edges, fall back to fixed branch counters
where that is safer, and report skipped sites explicitly.

## Library Coverage Proof Point

The example targets are best read as capability probes, not as a claim that
ROCm libraries are already broadly fuzzed. They show that the preload can reach
real library kernels through several packaging and launch paths:

| Example | Capability demonstrated | Remaining gap |
| --- | --- | --- |
| `rocblas-sgemm` | Real GEMM/Tensile launch, lazy CCOB exact-kernel shadowing, and default hybrid previous-BB/fixed branch coverage. | Coverage is tied to the current local `Type_SS` path; broader Tensile coverage needs stronger relocation and temporary-register planning. |
| `rocfft-c2c` | Non-GEMM generated kernels with device-edge deltas through ROCm library module loading. | The retained seed still exposes a known unpatched `twiddle_gen_*` payload. |
| `rocrand-uniform` | KPACK/fatbin registration, launch-scoped HSA-reader patching, and runtime shadow modules. | Coverage is shallow and currently uses loader-scoped fixed branch counters. |
| `rocsparse-spmv` | HSA-reader patching plus AFL-visible device-feedback differences between seeds. | The AFL wrapper remains non-persistent because the rocSPARSE handle path is not stable after AFL deferred forkserver setup. |
| `rocsolver-getrf` | Solver-stack HSA-reader/runtime-shadow patching with nonzero device-edge deltas. | Current coverage is loader-scoped fixed branch coverage and intentionally avoids numeric validation. |
| `miopen-activation` | Entry-unsafe kernel handling through self-contained fixed branch counters without entry redirection. | Richer previous-BB coverage needs more liveness and relocation support for that payload shape. |

The useful proof point is therefore coverage-path diversity: GEMM and non-GEMM
targets, direct and lower-level loader paths, KPACK/fatbin registration, CCOB
payloads, runtime shadows, persistent harnesses, and AFL-visible device feedback
all have at least one checked example. Turning that into product breadth means
expanding the supported matrix and failure policy, not merely adding more
one-off harnesses.

## Gaps Exposed by the Examples

The examples point to these production gaps, in priority order:

- Coverage depth across real kernels. rocBLAS gets useful hybrid
  previous-BB/fixed coverage on the local Tensile path, but rocRAND, rocSPARSE,
  and rocSOLVER mostly prove loader-scoped fixed branch counters.
- Relocation and resource planning. More edges require safer handling of VOPD,
  PC-relative code, `EXEC`/`SCC` state, temporary SGPR/VGPR pressure, scratch,
  descriptor growth, and trampoline placement.
- Loader and container hardening. CCOB, KPACK, HSA-reader, runtime-shadow, and
  lazy module paths work on checked cases, but product support needs explicit
  lifetime, concurrency, version, and fallback contracts.
- Persistent/forkserver robustness. Most examples have persistent variants, but
  the AFL wrapper still runs rocSPARSE non-persistently because that path is not
  stable after AFL deferred forkserver setup.
- Device-state ABI and scaling. Device counters and `previous_bb` state need a
  durable handoff and sizing model for streams, grids, cleanup, and concurrent
  launches.
- Measurement and support matrix. The prototype needs systematic overhead
  numbers, refreshed campaigns, CI hardware coverage for declared gfxip targets,
  and generated per-example reports from patch JSONL.

## Architecture

```text
AFL++ target process
  -> crash-focused HIP/ROCm harness
  -> librocjitsu_afl_preload.so intercepts HIP/HSA loader and launch calls
  -> rocjitsu parses supported code-object containers and AMDGPU ELFs
  -> rocjitsu decodes AMDGPU text, recovers CFGs, and plans safe probe sites
  -> patched kernels update device coverage counters
  -> preload synchronizes, quantizes, and merges counters into AFL trace_bits
```

The public enablement surface is deliberately small:

- preload `librocjitsu_afl_preload.so` into the target process;
- set only the supported `ROCJITSU_AFL_*` environment variables documented in
  the user manual for reports, diagnostics, and optional gates;
- optionally call persistent iteration hooks from the harness;
- use `ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1` for short wiring checks that are
  expected to produce real device branch feedback.

Everything below that surface is an implementation detail of the preload.
rocjitsu supplies the AMDGPU ELF utilities, generated gfxip-specific decoders,
CFG recovery, descriptor patching, cave insertion, instruction relocation
helpers, and AMDGCN probe emission that make the intercept possible.

The preload is a DBI client, not the full rocjitsu DBT pipeline. It does not
translate whole kernels into a new IR and re-emit them. Instead, it consumes the
same lower-level building blocks a DBT needs: code-object parsing, generated
instruction decoding, CFG/liveness analysis, kernel descriptor mutation,
resource accounting, trampoline placement, relocation checks, and target-specific
probe builders. Productionizing RocFuzz should therefore push these pieces
toward stable rocjitsu APIs instead of leaving them embedded in the preload.

The implementation is split as follows:

- `afl-dbi/runtime/`: preload, loader interposition, image parsing, patch
  reporting, instrumentation planning, relocation helpers, and AFL merge logic;
- `afl-dbi/smoke/`: synthetic HIP, fatbin, HSA-reader, CCOB, wave64, and
  high-lane smoke targets;
- `afl-dbi/tests/`: host-only parser, probe, edge-model, and relocation units;
- `afl-dbi/tools/`: CCOB packaging, code-object inspection, and report checks;
- `examples/`: crash-focused ROCm library fuzz targets and campaign reports;
- `manual-coverage-poc/`: older manual coverage reference, not the product path;
- [docs/user-manual.md](docs/user-manual.md): build, preload, environment,
  persistent-mode, and validation commands for people running the prototype.

## DBI/DBT Design References

The short version from surveying mature non-LLVM DBI/DBT systems is that
rocjitsu does not need a large compiler IR to productize RocFuzz. DynamoRIO,
QEMU TCG, and Frida Gum/Stalker all point toward a smaller native instruction
and block model with disciplined pass boundaries, explicit relocation and state
contracts, managed patch artifacts, and structured observer metrics.

For rocjitsu, the key architectural implication is that RocFuzz should become a
coverage-policy client on top of reusable DBI services: decoded AMDGCN block
mutation, resource-aware probe planning, relocator/writer APIs, transactional
code-object publication, device-state handoff, and reporting for selected,
degraded, and skipped sites. The detailed notes and source pointers are in the
[DBI/DBT literature survey](docs/dbi-literature-survey.md). That survey also covers
CUDA-side cuFuzz, the lack of a public ROCm/HIP equivalent, and why PC sampling
is useful diagnostic evidence but not a primary AFL feedback path.

## Why AFL++

AFL++ is the better default engine for a productized ROCm device-code fuzzer
because the coverage source has to be external to the target binary. Many ROCm
libraries and kernels we care about are prebuilt, generated, or hand-written
device payloads, and some will never be built by LLVM in a mode where
compiler-generated coverage instrumentation is available.

That makes an out-of-process fuzzing model with an externally writable coverage
map the right product shape. The DBI runtime can collect device coverage however
rocjitsu can obtain it, then merge it into AFL++'s `trace_bits` before the
engine decides whether an input is interesting. This is also the same basic
shape cuFuzz uses with NVBit.

libFuzzer can still be useful for source-available unit-style harnesses,
especially when compiler coverage is available. It should not be the primary
product assumption for ROCm library and device-code fuzzing, because its biggest
advantages come from in-process execution and compiler-inserted coverage.

## Intercept Coverage

The preload currently interposes these public HIP/HSA entrypoints:

- module loading: `hipModuleLoad`, `hipModuleLoadData`,
  `hipModuleLoadDataEx`, `hipModuleLoadFatBinary`, `hipModuleUnload`,
  `hipModuleGetFunction`;
- kernel launch: `hipModuleLaunchKernel`, `hipExtModuleLaunchKernel`,
  `hipLaunchKernel`;
- compiler registration: `__hipRegisterFatBinary`,
  `__hipRegisterFunction`, `__hipUnregisterFatBinary`;
- HSA code-object readers: `hsa_code_object_reader_create_from_memory`,
  `hsa_code_object_reader_create_from_file`,
  `hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size`,
  `hsa_code_object_reader_destroy`;
- AMD loader extension table lookup through `hsa_system_get_extension_table`,
  where the v1.02+ file-offset reader slot is rewritten to the same wrapper;
- synchronization and merge points: `hipDeviceSynchronize`,
  `hipStreamSynchronize`, `hipStreamSynchronize_spt`,
  `hipEventSynchronize`, blocking `hipMemcpy`, persistent iteration end, and
  process teardown.

HIP registration hooks provide fatbin/KPACK context and runtime-function names.
The HSA-reader hooks are the main patching path for compiler-registered fatbins
and several ROCm libraries. Direct HIP module-load hooks still matter for raw
HSACO tests, user code using the module API, and CCOB-backed Tensile payloads
that are loadable through HIP.

## Code-Object Flow

The patching flow is:

1. Intercept a direct HIP module load, compiler fatbin registration, lower HSA
   code-object reader creation, or AMD loader-table file-offset reader call.
2. Copy the loader image and classify it as raw AMDGPU ELF, clang offload
   bundle, HIP fatbin wrapper, CCOB, or adjacent container sequence.
3. Extract device images and order compatible AMDGPU ELFs by the current HIP
   device target.
4. For compiler-registered fatbins/KPACK payloads, scope HSA-reader patching to
   the active runtime launch when no explicit kernel filter is configured.
5. For large HIP-loaded CCOB payloads, remember the original module and defer
   patching until `hipModuleGetFunction` identifies the requested kernel.
6. Decode the selected ELF, find `.kd` kernel descriptors, build reachable CFGs,
   and choose safe block-entry and/or direct-branch sites.
7. Patch descriptors and text, append trampolines/probes into a reachable cave,
   then load the patched image or patched reader bytes.
8. On synchronization or persistent end, copy device counters, quantize them,
   and merge them into AFL's `trace_bits`.

The direct HIP CCOB path intentionally uses a `raw_exact_kernel_elf` shadow
policy today. That trades sibling-payload preservation for launch stability:
after `hipModuleGetFunction` names a kernel, the preload builds a patched raw
ELF for that exact kernel and returns a function handle from the shadow module.
The lower HSA-reader path uses the stronger sibling-preserving policy: it
rebuilds the parsed bundle/CCOB graph after replacing the selected device ELF,
tries that rebuilt container first, and keeps a raw patched ELF fallback if the
loader rejects the rebuilt image.

## Main Sources of Complexity

The fatbin/KPACK and CCOB work was the point where this stopped being a simple
raw-HSACO patcher. The same issues will show up in any product design:

- The bytes at the public HIP API boundary are often not the bytes the GPU
  loader finally consumes. ROCm libraries may pass HIP fatbins, KPACK-backed
  compiler registrations, clang bundles, nested CCOB containers, or raw ELFs
  with `.co` file names.
- Kernel identity arrives at different times on different paths. Registration
  exposes host/device function names, HSA readers expose raw code bytes, and
  HIP module APIs may not name the target kernel until `hipModuleGetFunction`.
  The implementation had to correlate those events instead of patching every
  payload eagerly.
- Patching a whole library container is both expensive and less stable. The
  prototype now prefers launch-scoped HSA-reader patching and per-kernel shadow
  modules for KPACK/fatbin paths, and lazy exact-kernel shadows for HIP CCOB
  modules.
- Multi-payload containers cannot always be collapsed to one raw ELF. The
  HSA-reader CCOB path preserves sibling payloads by rebuilding the container
  graph; the HIP lazy shadow path deliberately does not, because rebuilt CCOB
  shadow modules were less launch-stable in the smoke target.
- Loader lifetime and concurrency are part of correctness. The runtime tracks
  shadow-module caches, duplicate publication races, module unload, fatbin
  unregister, destructor-time cleanup, and sync-point attribution.
- Structured JSONL reports are required for debugging. Stderr is not enough to
  explain which image was selected, whether a CCOB rebuild preserved siblings,
  why a site was skipped, whether a shadow function was a cache hit, or which
  launch path produced device-edge deltas.
- Coverage insertion is a resource-allocation problem. Every site consumes
  SGPR/VGPR state, may need scratch backing, can require descriptor and metadata
  growth, and has to preserve architectural state such as `EXEC`, `SCC`, and
  branch operands.
- Relocation is a correctness boundary, not an optimization. Some overwritten
  instructions can be replayed safely, while PC-relative operands, VOPD,
  `EXEC` transitions, and SCC/VCC-sensitive instructions need explicit
  relocation rules before they can be used as general insertion points.
- Coverage fidelity competes with patchability. Previous-BB edge probes are the
  preferred signal, but fixed branch counters are a deliberate semantic fallback
  when previous-BB state, live `EXEC`, relocation, scratch, or cave placement
  would make a richer probe unsafe.

These are product-relevant lessons: a real ROCm fuzzer needs a loader-aware DBI
runtime and a resource-aware AMDGCN mutation backend, not only an instruction
patcher.

## Coverage Semantics

The AFL-visible map remains the normal 64K byte array:

```text
trace_bits[0..32767]      host coverage
trace_bits[32768..65535]  device coverage
```

Device hooks update raw 32-bit counters first. For previous-BB edges the
logical slot is:

```text
(previous_bb[workitem] ^ bb_id) & 32767
```

The emitted hook follows the cuFuzz-compatible first-active-lane policy: one
active lane increments the counter at a hook, while active lanes update their
own `previous_bb`. The host-side model documents the logical extension point
for counting distinct previous-BB groups, active-lane popcounts, or compact
`EXEC` state when we want more divergence-sensitive feedback.

Full previous-BB edge instrumentation is enabled for supported gfx11/gfx12
code objects. Older rocjitsu-decodable targets currently get a conservative
entry-counter tier; implementing pre-gfx11 previous-BB edge coverage is a TODO,
not current scope.

## User Manual

The public runtime surface is intentionally small: preload
`librocjitsu_afl_preload.so`, optionally enable structured patch reports, use
the persistent hooks for persistent harnesses, and gate short wiring tests with
`ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1`.

The detailed build, run, environment, persistent-mode, report, and validation
commands live in the [RocFuzz user manual](docs/user-manual.md). The README
keeps only the product and architecture story so it stays readable for reviewers
who are evaluating whether the intercept is worth productionizing.

## rocjitsu <> Fuzzer Intercept

cuFuzz relies on NVBit for services that ROCm does not provide as a ready-made
runtime API. The rocjitsu intercept has to replace those services explicitly.

| cuFuzz/NVBit service | rocjitsu fuzzer equivalent today | Gap to parity |
| --- | --- | --- |
| Per-kernel instrumentation event | HIP module-load hooks, HIP registration diagnostics, lower HSA-reader hooks, loader-table fallback, lazy CCOB function lookup, and launch interception | API-table coverage and lifetime handling need broader library validation. |
| CFG enumeration | `rocjitsu::BasicBlock::build(...)` over decoded AMDGPU text | CFG quality depends on generated decoder coverage and opaque-instruction handling. |
| Basic-block hook insertion | Text patching plus trampolines in executable caves | Only selected block entries and direct branch terminators are patched safely today. |
| Device helper call | Inline AMDGCN probe sequences | Probe register use is liveness-selected; gfx11/gfx12 VGPR scratch-spill fallback is part of adaptive planning where loader-visible private-segment growth is proven; SGPR scratch forms are still guarded by relocation, opaque-instruction, `EXEC`, and loader-safety constraints. |
| Instrumented-code activation | Patched HIP module bytes, patched HSA reader bytes, lazy CCOB shadows, or KPACK runtime shadows | Product path needs stronger concurrency, lifetime, and loader-table coverage. |
| Device state handoff | Process-local pointer embedded into probes, or loaded by self-contained edge probes | Product path needs an ABI-stable state mechanism. |
| Coverage merge | Preload copies device counters and merges into AFL `trace_bits` | Needs larger-grid state management, CI-produced reports, and crash repro metadata. |

## Product Questions

These questions need targeted spikes before committing to a product support
matrix:

| Question | Why it matters | Current status | How cuFuzz handles it |
| --- | --- | --- | --- |
| What performance penalty is acceptable? | A product fuzzer needs enough exec/sec to discover crashes, especially when persistent mode is unavailable or GPU launches are expensive. | We have smoke results and short campaign data, but no systematic patch-time, launch-time, shadow-module, or steady-state overhead model. | cuFuzz uses AFL++ persistent mode to amortize CUDA startup and launch overhead and carries benchmark/ablation modes for coverage and sanitizer combinations. |
| How much execution-mask feedback should be exposed? | Plain previous-BB edges may miss useful SIMT divergence information; richer `EXEC` feedback may improve discovery but increases probe cost and map pressure. | The host model documents active-mask semantics. Emitted probes use the cuFuzz-compatible first-active-lane policy. | cuFuzz uses CUDA active-lane primitives to pick one active lane for the edge increment and keeps `previous_bb` per work item; it does not expose the full lane mask as separate AFL feedback. |
| How should sanitizers integrate? | Crash-focused fuzzing finds process failures, but memory/race/API bugs may need ASan/UBSan, ROCm sanitizers, or library-specific checkers. | AFL++ can compose with source-available host sanitizers; device-side sanitizer composition with DBI instrumentation is not designed yet. | cuFuzz decouples coverage collection from expensive checking with AFL++ SAND wrappers and sanitizer-style modes. |

## Productionization Work

This should be treated as a rocjitsu DBI productization effort, not primarily as
a fuzz-target-writing effort. More library harnesses are useful, but they are
orthogonal; the core tooling is production-ready only when the same preload can
instrument supported ROCm libraries and user HIP code with predictable coverage,
diagnostics, overhead, and failure modes.

The remaining work breaks down into concrete engineering axes rather than a
credible single calendar estimate:

| Axis | Product work | rocjitsu payoff |
| --- | --- | --- |
| Loader and container contract | Turn the current intercept matrix into a supported loader contract with version checks, lifetime ownership rules, concurrency tests, and fallback behavior per path. | Hardens rocjitsu code-object ingestion against real ROCm library packaging instead of synthetic HSACO-only tests. |
| Transactional mutation backend | Move mutation planning into reusable rocjitsu APIs: patch transactions, cave allocation, rollback, symbol/offset remapping, descriptor/resource updates, and reportable preflight. | Gives rocjitsu a general DBI insertion substrate usable by fuzzing, tracing, profiling, and future DBT experiments. |
| Relocation and special state | Implement operand-aware relocation for PC-relative instructions, VOPD, multiword encodings, SCC/VCC/EXEC dependencies, and branch-condition preservation. | Improves rocjitsu's ability to transform arbitrary AMDGCN safely instead of only decoding and replaying narrow cases. |
| Resource-aware probe planning | Make the planner resource-complete: SGPR/VGPR allocation, scratch spill policy, private-segment growth, map-pressure accounting, cave reachability, runtime overhead accounting, and deterministic degradation. | Exercises and matures rocjitsu liveness, register models, kernel descriptor patching, and target-specific encoding support. |
| Device-state ABI | Define a durable state handoff through dispatch metadata, a code-object global, kernarg extension, loader-managed symbol, or another ABI that handles streams, grids, and process lifetime. | Establishes a reusable host/device side-channel for DBI clients that need runtime state. |
| Coverage semantics | Decide the supported feedback contract for previous-BB edges, fixed fallbacks, wave32/wave64, map pressure, optional active-mask summaries, and cross-kernel attribution. | Provides a precise semantic testbed for SIMT control-flow instrumentation in rocjitsu. |
| Support matrix | Publish supported gfxip/loader/library combinations, add required hardware runners, and keep pre-gfx11 previous-BB coverage explicitly out of scope until it is funded. | Makes rocjitsu target support evidence-based and visible to downstream users. |
| Persistent and forkserver runtime | Audit HIP/ROCm object lifetime across forkserver, streams, handles, lazy module loads, retries, and teardown; add repro tooling for lifecycle bugs. | Finds loader/runtime assumptions that matter to any long-running rocjitsu client, not just fuzzing. |
| Performance and reporting | Add systematic patch-time, launch-time, steady-state overhead, crash repro metadata, sanitizer composition, and per-example result generation in CI. | Turns rocjitsu DBI behavior into measurable artifacts instead of ad hoc stderr debugging. |
| Packaging and ergonomics | Package the preload, AFL++ setup, support matrix, example templates, and report tools so teams can add harnesses without understanding AMDGCN patching. | Forces rocjitsu APIs and diagnostics to be consumable by non-authors. |

## Prototype Results

The retained campaign report is
[examples/reports/campaign-20260523-153851.md](examples/reports/campaign-20260523-153851.md).
That run used short AFL++
campaigns with stagnation-based early stopping. No target produced crash or
hang artifacts.

That report predates the synthetic fatbin/CCOB loader smokes and the
rocRAND/rocSPARSE/rocSOLVER default-coverage promotions, so the library table
above is the better summary of current capability. The next measurement step is
rerunning short AFL campaigns for the updated branch-coverage paths and
refreshing the per-example report from structured JSONL data.
