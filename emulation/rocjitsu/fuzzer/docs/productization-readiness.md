# RocFuzz Productization Readiness

This note summarizes the current RocFuzz prototype, the evidence behind it, and
the remaining work required to turn it into a production-quality ROCm
device-code fuzzing tool. The detailed technical background lives in
[../README.md](../README.md) and
[dbi-literature-survey.md](dbi-literature-survey.md).

## Productization Summary

A focused one-quarter MVP staffed by two experienced compiler/DBI engineers is
a realistic next step if the goal is to make ROCm device-code fuzzing viable
for prebuilt libraries and user code. The prototype has moved the question from
"is ROCm device branch coverage possible without compiler instrumentation?" to
"can we turn the current defensive DBI prototype into a reusable rocjitsu
product surface with clear support boundaries?"

The recommended MVP should have a narrow definition of done: gfx11/gfx12,
crash-focused fuzzing, AFL++ preload integration, real device branch feedback,
structured patch reports, and explicit unsupported-path reporting. Broader ROCm
library target writing, sanitizer integration, numerics, and pre-gfx11
previous-BB coverage should remain out of scope for this phase.

## Current Evidence

The maintained bounded evidence gate passed on May 25, 2026:

```sh
cmake --build emulation/rocjitsu/fuzzer/examples/build --target \
  rocfuzz_example_coverage_baseline_report -j1
```

The lower-level gfx11/gfx12 instrumentation matrix also passed:

```sh
cmake --build emulation/rocjitsu/build --target \
  rocjitsu_afl_gfx11_gfx12_coverage_matrix
```

The regenerated report is
[../examples/reports/coverage-baseline-20260525.md](../examples/reports/coverage-baseline-20260525.md).
It is a coverage-quality report, not an AFL crash-campaign result.

| Example | Current evidence | Main limitation |
| --- | --- | --- |
| `rocblas-sgemm` | 4/4 patch events succeeded, 1149 patched sites, previous-BB plus fixed fallback coverage, 67 observed device slots. | Good proof of real Tensile/GEMM coverage, but more sites still need safer relocation, opaque-instruction modeling, and temporary-register planning. |
| `rocfft-c2c` | Generated FFT kernels produce previous-BB device deltas. | The retained `twiddle_gen_*` payload is still reported as no-patchable-sites. |
| `rocrand-uniform` | KPACK/HSA-reader runtime-shadow path produces fixed-branch device feedback and AFL-visible showmap differences. | Coverage is shallow and mostly proves loader/runtime-shadow plumbing. |
| `rocsparse-spmv` | 2/3 patch events succeeded, 23 fixed-counter sites, 12 observed device slots, AFL-visible showmap differences. | Persistent AFL wrapper remains blocked by rocSPARSE handle/forkserver stability. |
| `rocsolver-getrf` | 6/6 patch events succeeded, 46 fixed-counter sites, 33 observed device slots. | Coverage is through short helper kernels; richer previous-BB identity needs more safe sites. |
| `miopen-activation` | Entry-unsafe activation kernels use self-contained fixed branch counters and produce device deltas. | Small kernels; value is proving the entry-unsafe fallback path. |

This is enough to justify a focused productionization phase. It is not enough
to claim full ROCm library fuzzing support.

## What the Prototype Proves

- AFL++ can consume device-side branch feedback from prebuilt AMDGPU code
  objects through an external preload, without compiler-generated device
  coverage.
- rocjitsu can decode and patch supported AMDGCN kernels through multiple real
  loader paths: raw HSACO, HIP module load, HSA reader, fatbin/KPACK, runtime
  shadows, and lazy CCOB shadows.
- The adaptive coverage policy is the right product shape: prefer previous-BB
  edges, degrade to fixed branch counters when special-state or placement
  proofs are missing, and report skipped sites.
- Structured patch JSONL is necessary for explaining low edge counts, selected
  sites, fallback reasons, loader behavior, and device-edge deltas.
- The infrastructure needed for RocFuzz overlaps strongly with reusable
  rocjitsu DBI services: mutable AMDGCN blocks, relocation, state preservation,
  patch transactions, descriptor/resource updates, and device-state handoff.

## What It Does Not Prove

- Arbitrary ROCm library coverage. The examples are capability probes, not a
  complete library support matrix.
- Full relocation correctness for every AMDGCN instruction shape, especially
  PC-relative code, VOPD, multiword encodings, and SCC/VCC/EXEC-sensitive
  sequences.
- Stable overhead or throughput. We have smoke and short-campaign data, not a
  systematic patch-time, launch-time, and steady-state overhead model.
- Sanitizer composition. Host sanitizers may compose with source-available
  harnesses, but device-side sanitizer interaction with DBI coverage is not
  designed.
- Persistent mode across every library. rocSPARSE is the current known example
  where the retained AFL path remains non-persistent.
- pre-gfx11 previous-BB coverage. Older rocjitsu-decodable targets remain a
  conservative entry/fixed-counter tier until explicitly funded.

## MVP Scope

The productionization MVP should support:

- gfx11/gfx12 AMDGPU targets with wave32 and forced-wave64 coverage checks;
- crash-focused fuzzing, not numeric correctness checking;
- AFL++ preload integration with host and device coverage merged into the AFL
  bitmap;
- supported loader paths for direct HIP modules, HSA reader memory/file paths,
  compiler fatbins/KPACK, runtime shadows, and lazy CCOB shadows;
- adaptive branch/edge coverage with previous-BB where safe and fixed counters
  as the default degradation path;
- structured patch reports and per-example generated reports;
- CI hardware coverage for the declared support matrix;
- ergonomic harness templates for teams adding ROCm library or user-code fuzz
  targets.

The MVP should explicitly exclude:

- broad ROCm library harness ownership;
- pre-gfx11 previous-BB edge coverage;
- sanitizer integration beyond host-side source-available composition;
- numeric validation;
- LLVM/compiler-inserted device coverage assumptions;
- a public matrix of coverage modes. The user-facing policy should remain
  adaptive best-effort coverage with diagnostics.

## Risks and Decision Criteria

| Risk | Current evidence | Decision criterion |
| --- | --- | --- |
| Relocation and special-state correctness | Current probes work on selected gfx11/gfx12 paths and report EXEC/SCC/VOPD/placement degradation. | Resize if too many real kernels require unmodeled PC-relative, VOPD, SCC/VCC, or EXEC transformations to get useful coverage. Defer broader product claims if legality decisions cannot be made explainable and fail-closed. |
| Device-state ABI | Current prototype can allocate device counters and `previous_bb` state and make them visible to patched code. | Resize if pointer embedding and current launch-scoped state cannot support streams, concurrent launches, and persistent reuse. Defer supported-loader claims if no ABI-stable handoff is feasible. |
| Loader/container lifetime | Raw HSACO, HSA reader, fatbin/KPACK, runtime shadow, and CCOB paths have checked examples. | Resize if ROCm library loaders require too many path-specific hooks. Defer broad loader support if shadow publication cannot be made transactional with clear unload/unregister cleanup. |
| Persistent/forkserver robustness | Several examples have persistent variants; rocSPARSE remains non-persistent. | Resize if major libraries cannot run under AFL persistent/deferred forkserver without corrupting ROCm handles. Keep non-persistent support as a fallback, but require explicit throughput data. |
| Runtime overhead | Coverage gates prove functionality but not systematic overhead. | Resize if patching or launch overhead prevents useful exec/sec on representative crash harnesses. Require baseline overhead measurement before expanding library targets. |
| CI and hardware availability | Local gfx1201/gfx1100 checks pass for maintained matrices. | Defer product claims if there is no owned CI hardware for the declared gfxip/wave-mode support matrix. |
| Support-scope creep | The prototype can reach several real libraries, but each has different loader and kernel behavior. | Resize if the effort turns into writing and owning many library fuzz targets. Core tooling and harness authoring must remain separate workstreams. |

## Recommended Framing

The strongest framing is:

> RocFuzz has proven that rocjitsu can fill the missing ROCm equivalent of an
> NVBit-like device instrumentation layer for prebuilt AMDGPU kernels. A
> quarter of focused compiler/DBI investment should determine whether this can
> become reusable production infrastructure with credible coverage semantics,
> support boundaries, and diagnostics.

The weakest framing would be:

> RocFuzz already fuzzes ROCm libraries broadly.

That overstates the evidence. The prototype shows feasibility and a clear
technical path; productionization is the work needed to make that path safe,
ergonomic, measured, and supportable.
