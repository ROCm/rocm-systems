# rocjitsu fuzzer library examples

This directory contains crash-focused AFL++ targets for real ROCm library
entrypoints.  The targets are intentionally small: AFL input bytes select
bounded operation parameters and data patterns, then the harness calls the
library and lets process crashes surface naturally.  They do not use numeric
reference checks.

The active smoke targets exercise standard ROCm library entrypoints:

- `rocblas_sgemm` on bounded single-precision GEMM shapes.  This exercises a
  real GEMM provider and real rocBLAS/Tensile kernel launches.  Its smoke
  targets require real device branch coverage.  The preload detects the
  entry-unsafe Tensile `Cijk_*` payloads and uses the default hybrid branch
  planner because redirecting those kernel entries still trips the local
  payload.
- `rocfft_execute` on a bounded single-precision C2C transform.  This path
  exercises real ROCm library module loading and kernel launches without
  depending on GEMM selection.  The default smokes now require device branch
  coverage from the generated FFT kernels even though the retained seed also
  launches an unpatched twiddle kernel.  The high-edge report gate keeps that
  `twiddle_gen_*` gap explicit.
- `rocrand_generate_uniform` on a bounded pseudorandom generator.  The TheRock
  rocRAND package dispatches through HIP KPACK/fatbin registration.  The preload
  records HIP runtime-function registrations and scopes lower HSA-reader DBI to
  the launched kernel.  Later named launches in the same HIP fatbin registration
  can be patched from the cached raw ELF through default runtime shadow modules
  without KPACK-wide patching of unlaunched rocRAND variants.  The checked-in
  smoke now observes device branch coverage through the default loader-scoped
  fixed branch policy.
- `rocsparse_scsrmv` on bounded CSR sparse matrix/vector products.  The input
  mutates dimensions, transpose mode, row-density patterns, column indices, and
  scalar/data values.  The smoke targets require device branch coverage through
  the default HSA-reader loader policy, which uses self-contained fixed branch
  counters for this loader-scoped payload.
- `rocsolver_sgetrf` on bounded single-precision LU factorizations.  The input
  mutates rectangular dimensions, leading dimension padding, and matrix values
  while treating singular pivots as ordinary library results, not harness
  failures.  The smoke targets require device branch coverage through
  HSA-reader/runtime-shadow patching and the default loader-scoped fixed branch
  policy.
- `miopenActivationForward` on bounded NCHW single-precision tensors.  The input
  mutates tensor shapes, activation mode, activation parameters, and data values.
  Its smoke targets require device branch coverage through the default hybrid
  planner: entry-safe kernels keep entry probes, while the launched
  entry-unsafe activation kernel falls back to a self-contained fixed branch
  counter when previous-BB branch probes have no liveness-safe site.

The exploratory `hipblasLtMatrixTransform` target loads the TheRock
`hipblasltTransform_gfx*.hsaco` payload, but it is not the default smoke path
because the current TheRock package aborts in that code path before we get a
clean end-to-end fuzz iteration.  The exploratory `rocblas_saxpy` target is a
smaller non-GEMM rocBLAS path kept around for quick rocBLAS launch checks.

Some GEMM paths can select Tensile `.co` payloads in the CCOB container format.
The preload has parser coverage, a HIP-accepted synthetic CCOB-wrapped branchy
smoke, and a lazy HIP-module path for large CCOB payloads: it remembers the
original module, waits for `hipModuleGetFunction` to name the launched kernel,
then loads a second raw ELF patched only for that kernel.  On the local gfx1100
path, rocBLAS SGEMM launches a `Type_SS` VALU/FMAC kernel and reports nonzero
required device branch edges through local text-cave trampolines.  This path
does not depend on WMMA decoding.

The examples use TheRock wheels in a local uv-managed virtual environment so
they do not depend on a system `/opt/rocm` install.

## Setup

Install the newest multi-arch TheRock ROCm SDK into the local venv:

```sh
cd emulation/rocjitsu/fuzzer/examples
./scripts/setup-therock.sh
```

By default this installs library, development, `gfx1201`, and `gfx1100` packages
from `https://rocm.nightlies.amd.com/whl-multi-arch/`.  Override the defaults
with environment variables:

```sh
ROCFUZZ_THEROCK_DEVICES=gfx1201,gfx1100 ./scripts/setup-therock.sh
ROCFUZZ_THEROCK_VERSION=7.14.0a20260523 ./scripts/setup-therock.sh
```

The script writes `therock.env` with the resolved SDK paths.  Source it before
configuring examples by hand:

```sh
. ./therock.env
cmake -S . -B build -G Ninja
cmake --build build --target rocfuzz_example_rocblas_sgemm
cmake --build build --target rocfuzz_example_rocfft_c2c
cmake --build build --target rocfuzz_example_rocrand_uniform
cmake --build build --target rocfuzz_example_rocsparse_spmv
cmake --build build --target rocfuzz_example_rocsolver_getrf
cmake --build build --target rocfuzz_example_miopen_activation
```

The smoke and report targets rebuild the parent rocjitsu preload target before
running when `ROCFUZZ_BUILD_AFL_PRELOAD=ON` and `ROCFUZZ_ROCJITSU_BUILD` points
at a configured rocjitsu build, which is the default for this checkout.  To
build it manually or use an external preload path:

```sh
cmake --build ../../build --target rocjitsu_afl_preload
cmake -S . -B build -G Ninja -DROCFUZZ_AFL_PRELOAD=/path/to/librocjitsu_afl_preload.so -DROCFUZZ_BUILD_AFL_PRELOAD=OFF
```

Run the smoke path with the DBI preload:

```sh
cmake --build build --target rocfuzz_example_rocblas_sgemm_smoke
cmake --build build --target rocfuzz_example_rocblas_sgemm_persistent_smoke
cmake --build build --target rocfuzz_example_rocfft_c2c_smoke
cmake --build build --target rocfuzz_example_rocfft_c2c_persistent_smoke
cmake --build build --target rocfuzz_example_rocrand_uniform_smoke
cmake --build build --target rocfuzz_example_rocrand_uniform_persistent_smoke
cmake --build build --target rocfuzz_example_rocsparse_spmv_smoke
cmake --build build --target rocfuzz_example_rocsparse_spmv_persistent_smoke
cmake --build build --target rocfuzz_example_rocsolver_getrf_smoke
cmake --build build --target rocfuzz_example_rocsolver_getrf_persistent_smoke
cmake --build build --target rocfuzz_example_miopen_activation_smoke
cmake --build build --target rocfuzz_example_miopen_activation_persistent_smoke
cmake --build build --target rocfuzz_example_device_showmap_checks
cmake --build build --target rocfuzz_example_high_edge_report_smoke
cmake --build build --target rocfuzz_example_rocblas_sgemm_default_hybrid_report_smoke
cmake --build build --target rocfuzz_example_rocblas_sgemm_vopd_fresh_growth_report_smoke
cmake --build build --target rocfuzz_example_coverage_baseline_report
cmake --build build --target rocfuzz_example_smoke_policy_gates
```

## Coverage Gate Policy

`rocfuzz_example_smoke_policy_gates` is the maintained bounded validation
target for example coverage.  It intentionally does not run AFL campaigns; it
checks that the preload still patches real library code objects, observes
device edge deltas where we claim branch coverage, and reports the known
launch-only gaps explicitly.

| Example | Gate |
| --- | --- |
| `rocblas-sgemm` | Required device edges in normal and persistent smokes; default hybrid report requires previous-BB branch probes where safe, liveness-selected registers, fixed fallback slots, and a device-edge delta. A separate diagnostic smoke gates the bounded VOPD/Tensile fresh-register growth proof. |
| `rocfft-c2c` | Normal and persistent smokes require device edges from generated FFT kernels; the high-edge report also requires the known `twiddle_gen_*` no-patchable-sites record. |
| `rocrand-uniform` | Patch report requires loader-scoped fixed branch counters, stable fixed-slot scope, and a device-edge delta; persistent smoke checks persistent hook wiring. |
| `rocsparse-spmv` | Required device edges plus a patch-report check for loader-scoped fixed branch counters and nonzero device-edge deltas.  AFL-visible showmap comparison must differ for two seeds when `afl-showmap` is available. |
| `rocsolver-getrf` | Required device edges plus a patch-report check for loader/runtime-shadow fixed branch counters and nonzero device-edge deltas. |
| `miopen-activation` | Required device edges plus a report check that proves entry-unsafe activation kernels use self-contained fixed branch counters without entry redirection. |

`rocfuzz_example_high_edge_report_smoke` is report-driven.  It raises the
edge budgets on rocBLAS SGEMM, rocFFT C2C, and MIOpen activation, then checks
the generated JSONL patch reports.  rocBLAS and MIOpen also require an observed
device-edge delta for the current seed.  rocFFT also requires deltas for the
patched FFT kernels, while preserving a report check for the retained seed's
unpatched twiddle kernel.

`rocfuzz_example_rocblas_sgemm_default_hybrid_report_smoke` gates the default
best-effort Tensile path. It requires the preload to keep entry probes for
entry-safe helper kernels, choose self-contained previous-BB branch probes where
they are safe, degrade EXEC-conditioned or over-budget branch sites to fixed
counters, use liveness-selected probe registers, select branch-edge sites, and
observe a device-edge delta without spelling out the old branch-only env-var
recipe. The report also pins fixed-counter fallback cause accounting and the
candidate-derived fixed fallback budget, so regressions distinguish aggregate-cap
fallback from EXEC-safety, liveness, and placement fallback.

`rocfuzz_example_rocblas_sgemm_vopd_fresh_growth_report_smoke` is not a public
coverage mode. It is a bounded regression for the VOPD/Tensile safety proof:
the target scopes rocBLAS SGEMM to MT128x64 kernels, enables the debug-only
opaque fresh-register path, forces SGPR and VGPR growth, disables scratch
avoidance, and requires previous-BB branch sites, descriptor/metadata growth,
zero patch failures, and a device-edge delta.

The MIOpen activation smokes also use the default planner. They currently prove
the entry-unsafe fallback path: activation kernels leave their descriptors'
entry points unchanged and use self-contained fixed branch counters. The report
check pins `MIOpenActiveFwdLite` with same-item JSON assertions, so this
regression keeps covering the payload that previously faulted under entry
redirection.

The DBI smoke coverage also exercises the finer-grained adaptive fallback: a
self-contained branch kernel can keep previous-BB branch probes at safe sites
and degrade only the EXEC-conditioned or liveness-expensive branch edges to
fixed counters.

`rocfuzz_example_device_showmap_checks` runs AFL++ `afl-showmap` against
rocRAND and rocSPARSE input pairs, filters out host coverage and the device
launch counter, and compares only AFL-visible device branch slots. This catches
regressions where patching and `device_edge_delta` still work but input changes
no longer produce different device feedback for the fuzzer.

`rocfuzz_example_coverage_baseline_report` depends on the maintained smoke
policy gates and writes `build/coverage-baseline.md`. The report classifies each
active example, summarizes patch events, coverage mix, hashed/fixed edge sites,
device-edge deltas, sampled selected edge forms, AFL-visible showmap evidence
where available, and low-edge blocker buckets from structured patch-report data.
Loader-scoped kernel filters are reported separately from unsupported
instrumentation so scoped HSA/KPACK paths do not look like generic failures. It
is a coverage-quality report, not a crash campaign report.

For an AFL++ run, use the checked-in wrapper.  It configures a separate
`build-afl` tree with `afl-clang-fast++`, builds the selected harness, sets the
rocjitsu preload and target-specific ROCm library environment, and then launches
`afl-fuzz`.  Most targets use AFL++ persistent shared-memory testcase delivery;
`rocsparse-spmv` currently consumes one shared-memory testcase per process
because the rocSPARSE handle path is not stable after AFL's deferred forkserver.
The wrapper gives `rocsparse-spmv` a larger default AFL timeout because the
first rocSPARSE launch is slow.  Branch-covered CMake smoke targets set
`ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1`; report-only or launch-only smokes leave
that gate off.  The AFL wrapper also leaves the gate off by default so
mutations that skip all patched device edges are not reported as crashes.  Set
`ROCFUZZ_REQUIRE_DEVICE_EDGES=1` only for short wiring checks.

```sh
scripts/run-afl-example.sh rocblas-sgemm -V 60
scripts/run-afl-example.sh rocfft-c2c -V 60
scripts/run-afl-example.sh rocsparse-spmv -V 60
scripts/run-afl-example.sh rocsolver-getrf -V 60
scripts/run-afl-example.sh miopen-activation -V 60
```

Generate a markdown report from an AFL output root and the current patch-report
smoke files with:

```sh
scripts/summarize-afl-campaign.py afl-out/campaign-YYYYMMDD-HHMMSS \
  --output reports/campaign-YYYYMMDD-HHMMSS.md
```

For a single bounded wrapper run, set `ROCFUZZ_AFL_REPORT` and the wrapper will
write the report after `afl-fuzz` exits:

```sh
ROCFUZZ_AFL_REPORT=reports/rocblas-sgemm.md \
  scripts/run-afl-example.sh rocblas-sgemm -V 60
```

Omit `-V 60` for an open-ended fuzzing run.  The wrapper also accepts
`rocrand-uniform`, which exercises launch-scoped KPACK/HSA-reader coverage plus
default runtime shadow modules for later launches in the same registered
fatbin.

Generated venvs, build directories, AFL queues, and findings are ignored by git.
Campaign summaries that are meant to be kept live under `reports/`.
