# ConSan CDNA4 (`gfx950`) status

This is the native `gfx950` workload × instrumentation evidence ledger.  It
follows the acceptance standard of [STATUS_RDNA4.md](STATUS_RDNA4.md) and the
expanded corpus structure of [STATUS_GFX1250.md](STATUS_GFX1250.md), but it
inherits no coverage denominator, machine-code identity, fault expectation,
timing, provenance, or green cell from another architecture.

The executable authority is
[`consan_validation.py`](../../tests/dbi/consan/consan_validation.py), with the
experiment contract described by [VALIDATION.md](VALIDATION.md).  Porting work
and dependencies are tracked in [PLAN_GFX950.md](PLAN_GFX950.md).

End-to-end evidence is the primary project metric.  Focused decoder, builder,
spill, and resource tests are prerequisites and debugging tools; they cannot
promote a workload cell by themselves.

All benchmark and workload measurements use exactly one repetition
(`--benchmark_repetitions=1`, or `--repetitions 1` for the PyTorch runner).

## Status legend

Every cell uses the same maturity scale as the other architecture ledgers:

- 🩶 **unseen / unassessed:** no useful current-tip gfx950 execution evidence
  yet;
- 🟥 **does not work:** current execution fails before establishing useful
  workload/profile behavior;
- 🟧 **some things work:** useful execution or instrumentation behavior is
  demonstrated, but substantial correctness, coverage, completion, or
  acceptance gaps remain;
- 🟨 **most things work:** the clean workload and important instrumentation
  path work, with limited coverage or final-acceptance evidence still missing;
- 🟩 **everything works:** every required gate is retained at one frozen
  revision, including clean, coverage, oracle, fault, containment, overhead,
  memory, timeout, health, and provenance evidence.

`N/A` is used only when a fresh gfx950 inventory proves semantic absence and
records a typed reason.

## Current matrix

Every cell began gray at the current branch tip because the implementation and
workload changed substantially since the first gfx950 campaign.  Text
following a gray heart records historical evidence or a known prerequisite
only.  The first current-tip D128-block run below promotes only the profiles it
actually revalidated; every other profile must still be revalidated from
scratch before promotion.

| Workload | SuperCollider | Record/Replay | Sampled | Inline Shadow |
|---|---|---|---|---|
| **P0 Qwen3-0.6B prefill** | 🩶 Unknown | 🩶 Unknown | 🩶 Unknown | 🩶 Unknown |
| **P1 Sharktank TP1 prefill** | 🩶 historical: clean 120/120 native LDS access accepted | 🩶 historical: clean 104/104 access + 31/31 barrier accepted | 🩶 historical: clean 104/104 access + 7/7 qualified barrier accepted | 🩶 historical: clean 104/104 access + 31/31 barrier accepted |
| **P1 Sharktank TP1 decode/combined** | 🩶 historical: clean 240/240 native LDS access accepted | 🩶 historical: clean 208/208 access + 62/62 barrier accepted | 🩶 historical: clean 208/208 access + 14/14 qualified barrier accepted | 🩶 historical: clean 208/208 access + 62/62 barrier accepted across six sequential runs |
| **P2 Sharktank TP2 family** | 🩶 historical: clean 936/936 native LDS access accepted | 🩶 historical: clean 840/840 access + 168/168 barrier accepted | 🩶 historical: aggregate rejected: one 140-access + 10-barrier slice executes 0 | 🩶 historical: aggregate rejected: one 140-access + 28-barrier slice lowers 0 |
| **P3 CLIP BF16** | 🩶 historical: clean 45/45 native LDS access accepted | 🩶 historical: clean 39/39 access + 24/24 barrier accepted | 🩶 historical: clean 39/39 access + 20/20 qualified barrier accepted | 🩶 historical: clean 39/39 access + 24/24 barrier accepted |
| **P4 hip-moi D128 block attention** | 🟨 current clean: both exact oracles pass; 12/12 accesses; complete analysis | 🟥 current planner rejects persistent state at the ordinary-VGPR/AccVGPR boundary | 🟥 current planner rejects persistent state at the ordinary-VGPR/AccVGPR boundary | 🟥 current scalar fallback lacks entry-local scratch for one connected component |
| **P4 hip-moi D128 pressure attention** | 🩶 historical: clean 12/12 group-FLAT accepted | 🩶 historical: clean 12/12 access + 4/4 barrier accepted | 🩶 historical: clean 12/12 access accepted | 🩶 historical: clean 12/12 access + 4/4 barrier accepted |
| **P4 hip-moi MFMA attention** | 🩶 historical: clean 12/12 group-FLAT accepted | 🩶 historical: clean 12/12 access + 4/4 barrier accepted | 🩶 historical: clean 12/12 access accepted | 🩶 historical: clean 12/12 access + 4/4 barrier accepted |
| **P4 hip-moi Stream-K arrival** | 🩶 historical: clean 4/4 access accepted | 🩶 historical: clean 4/4 access + 10/10 atomic + 4/4 barrier + 16/16 fence accepted | 🩶 historical: clean 4/4 access + 10/10 atomic accepted | 🩶 historical: planner safely rejected persistent identity at a full ordinary-VGPR/AccVGPR boundary; scalar persistent state open |
| **P4 hip-moi tree atomic-OR** | 🩶 historical: clean 4/4 access accepted | 🩶 historical: clean 4/4 access + 10/10 atomic + 4/4 barrier + 16/16 fence accepted | 🩶 historical: clean 4/4 access + 10/10 atomic accepted | 🩶 historical: clean 4/4 access + 10/10 atomic + 4/4 barrier accepted |
| **P4 hip-moi Jakub attention** | 🩶 Applicability inventory pending | 🩶 Applicability inventory pending | 🩶 Applicability inventory pending | 🩶 Applicability inventory pending |

### Current-matrix executable audit

The exact validation IDs below are the workload definitions used by
`consan_validation.py --target gfx950`.  A workload-scoped `doctor` check was
run for every row on 2026-07-22.  “Runnable” means that all required sources,
assets, target-native executables, and workspace tools resolve now; it is not
an instrumentation acceptance claim.

| Validation ID | Current gfx950 availability | Exact definition |
|---|---|---|
| `qwen-prefill` | **Blocked** | Input, expected output, and parameters exist, but `iree-test-suites-build/torch_models/qwen3-600m/gfx950/qwen3-600m.vmfb` is missing. |
| `tp1-prefill` | **Runnable** | Sharktank `toy_llama.mlir` + `toy_llama.irpa`, prefill mode; workload-scoped doctor passes. |
| `tp1-decode-combined` | **Runnable** | The same TP1 assets, decode and combined modes; workload-scoped doctor passes. |
| `tp2-family` | **Runnable** | Sharktank `toy_llama_tp2.mlir` with common, rank-0, and rank-1 parameters; workload-scoped doctor passes. |
| `clip-bf16` | **Runnable** | Sharktank toy CLIP BF16 MLIR, parameters, input, and expected result; workload-scoped doctor passes. |
| `d128-block` | **Runnable and smoke-passed** | `hip_moi_instrumented_cdna4_d128_attention_block_test`, `HipMoiCdna4D128AttentionBlock.SampledFastContextMatchesHostReference`; physical gfx950 oracle passes in 120 ms. |
| `d128-pressure` | **Runnable and smoke-passed** | `hip_moi_instrumented_cdna4_d128_attention_pressure_test`, `HipMoiCdna4D128AttentionPressure.FullKvDoubleBufferedExactContextMatchesHostReference`; physical gfx950 oracle passes in 188 ms. |
| `wmma-attention` | **Runnable and smoke-passed** | `hip_moi_instrumented_cdna4_mfma_attention_block_test`, `HipMoiCdna4MfmaAttentionBlock.ExactContextMatchesHostReference`; physical gfx950 oracle passes in 142 ms.  The historical validation ID is retained, but the native operation is MFMA rather than WMMA. |
| `streamk-arrival` | **Runnable and smoke-passed** | `hip_moi_instrumented_cdna4_mfma_streamk_arrival_counter_test`, `HipMoiCdna4MfmaStreamKArrivalCounter.AcqRelFetchAddOrdersMfmaPartials`; physical gfx950 oracle passes in 92 ms. |
| `tree-atomic-or` | **Runnable and smoke-passed** | `hip_moi_instrumented_cdna4_mfma_streamk_tree_atomic_or_test`, `HipMoiCdna4MfmaStreamKTreeAtomicOr.AcqRelBitmaskOrdersMfmaPartials`; physical gfx950 oracle passes in 101 ms. |
| `jakub-attention` | **Blocked** | The registry deliberately requires `hip_moi_reference_cdna4_jakub_matmul`; that target-native executable does not yet exist. |

The two blocked rows remain visible so their missing deliverable is explicit.
They must not be scheduled as executable validation until the named artifact
exists.  All smoke commands use the physical device through the workspace
TheRock runtime, with software-model environment variables unset.

## RocJitsu test-corpus expansion

The workspace corpus was surveyed at revision `aa54cc86c9eb`.  Its packaged
Tensile artifacts are gfx1250-only, so none may be relabeled or executed as
gfx950 evidence.  The corpus does contain several source-built gfx950 kernel
cases.  Every cell below remains gray until the case has an independent
oracle, a retained target-native inventory, and a standard-profile clean run
for that flavor.

The gfx950 corpus configuration enables HIP matmul, HipKittens, and rocBLAS.
It has no run-time skip list.  It currently skips compilation of the large
4096³ HIP matmul case and both four-wave FP8 HipKittens cases; those are useful
planned rows, not runnable evidence.

### Corpus executable audit

| Tracking unit | Current gfx950 availability | Exact source/build contract |
|---|---|---|
| `hip_matmul_matmul::m128_n128_k128` | **Runnable and smoke-passed** | `corpus/kernels/cases/hip-matmul/matmul/case.json`; executable `rocjitsu-test-corpus-build/kernels-gfx950-hip-matmul/cases/hip-matmul/hip_matmul_matmul`; `-m 128 -n 128 -k 128`, `FIXED_ITERATIONS=1`.  All three selected MFMA/shared-memory kernels pass correctness on the physical gfx950. |
| `hipkittens_gemm_bf16fp32_16x32::m256_n256_k256` | **Source-defined, build required** | The case declares gfx950 support and an exit-code oracle, but the current `kernels-gfx950` directory has no completed Ninja build or executable. |
| FP8/MXFP8 four-wave HipKittens rows | **Compile-disabled** | Both exact case JSON files exist, but `corpus/kernels/configs/gfx950.json` lists them in `skip_compile_tests`. |
| `hip_streamk_simple::m256_n256_k256` and `hip_streamk_two_tile::m256_n256_k256` | **Not gfx950-enabled** | Exact source/oracle definitions exist, but both case manifests currently declare only gfx942. |
| `rocblas_sgemm` exact cases | **Source-defined, build required** | `corpus/kernels/cases/rocblas/sgemm/case.json` declares gfx950 and exact named tests, but the current gfx950 corpus build has no `rocblas_sgemm` executable. |

Only the first row is runnable today.  The other rows are retained as explicit
enablement work and may not be counted as current gfx950 execution evidence.

| Priority | Tracking unit | SuperCollider | Record/Replay | Sampled | Inline Shadow | Why it matters and next proof |
|---|---|---|---|---|---|---|
| P0 | `hip_matmul_matmul::m128_n128_k128` | 🩶 Unassessed | 🩶 Unassessed | 🩶 Unassessed | 🩶 Unassessed | Native gfx950 MFMA kernels use shared-memory tiling, repeated workgroup barriers, and a double-buffered LDS path.  Preserve the upstream numeric check as a machine-readable oracle, inventory every selected kernel, then run the four clean profiles. |
| P0 | `hipkittens_gemm_bf16fp32_16x32::m256_n256_k256` | 🩶 Unassessed | 🩶 Unassessed | 🩶 Unassessed | 🩶 Unassessed | Explicit gfx950 case with dynamic LDS, wide DS reads/writes, direct global-to-LDS traffic, a deep barrier schedule, and MFMA.  This is the highest-signal packaged gfx950 synchronization case. |
| P1 | `hipkittens_gemm_fp8fp32_4wave::m256_n256_k256` | 🩶 Compile assessment pending | 🩶 Compile assessment pending | 🩶 Compile assessment pending | 🩶 Compile assessment pending | Explicit gfx950 four-wave FP8 case; currently listed in `skip_compile_tests`.  Remove that corpus-level blocker only after recording the compiler failure or confirming a current toolchain build, then inventory its LDS/barrier shapes. |
| P1 | `hipkittens_gemm_mxfp8_4wave::m256_n256_k256` | 🩶 Compile assessment pending | 🩶 Compile assessment pending | 🩶 Compile assessment pending | 🩶 Compile assessment pending | Explicit gfx950 four-wave microscaling GEMM; also currently compile-skipped.  It is the closest packaged low-precision companion to the BF16 row. |
| P1 | `hip_streamk_simple::m256_n256_k256` gfx950 port | 🩶 Target enablement pending | 🩶 Target enablement pending | 🩶 Target enablement pending | 🩶 Target enablement pending | Source contains double-buffered LDS, workgroup barriers, and release-store/acquire-load cross-workgroup publication.  Metadata currently admits only gfx942, so gfx950 compilation and numeric equivalence must be established before this becomes a runnable row. |
| P1 | `hip_streamk_two_tile::m256_n256_k256` gfx950 port | 🩶 Target enablement pending | 🩶 Target enablement pending | 🩶 Target enablement pending | 🩶 Target enablement pending | Adds two-tile ownership and repeated global publication/consumption to the same LDS and ordered-atomic structure.  Treat it as a separate denominator after target enablement. |
| P2 | `rocblas_sgemm` compact exact cases | 🩶 Unassessed | 🩶 Unassessed | 🩶 Unassessed | 🩶 Unassessed | The gfx950 corpus exposes exact small and rectangular SGEMMs through rocBLAS.  First freeze a deterministic selected solution and prove its synchronization inventory; library dispatch alone is not ConSan signal. |

### gfx950 Tensile follow-on

There is not yet a packaged gfx950 Tensile manifest in
`rocjitsu-test-corpus`.  The workspace rocm-libraries checkout at revision
`c2fafc16393d` contains 36 gfx950 GEMM YAML files.  The following bounded
subset is the proposed gfx950 equivalent of the high-signal gfx1250 Tensile
rows; it must be generated and added to the corpus before validation.  All
statuses are deliberately gray.

| Priority | Proposed gfx950 config | SuperCollider | Record/Replay | Sampled | Inline Shadow | Selection rationale |
|---|---|---|---|---|---|---|
| P0 | `gemm/gfx950/xfp32.yaml`, reduced Stream-K exact case | 🩶 Not packaged | 🩶 Not packaged | 🩶 Not packaged | 🩶 Not packaged | Combines `StreamK: 3`, `DirectToLds: 1`, optional cluster-local reads, XCC mapping, and exact problem sizes.  This is the first candidate for ordered global publication plus LDS synchronization. |
| P1 | `gemm/gfx950/general_wgm.yaml`, reduced Stream-K matrix | 🩶 Not packaged | 🩶 Not packaged | 🩶 Not packaged | 🩶 Not packaged | Exercises Stream-K modes 1/2/3, LDS transpose, `StoreSyncOpt`, and workgroup/XCC mapping.  Reduce it to one numerically checked representative per distinct synchronization shape. |
| P1 | `gemm/gfx950/lds_tr.yaml` | 🩶 Not packaged | 🩶 Not packaged | 🩶 Not packaged | 🩶 Not packaged | Explicit LDS-transpose instructions, single- and multiple-buffer layouts, and exact odd-size cases provide target-specific DS-shape breadth. |
| P1 | `gemm/gfx950/lds160K.yaml` | 🩶 Not packaged | 🩶 Not packaged | 🩶 Not packaged | 🩶 Not packaged | Single/double-buffer transpose variants near the 160 KiB gfx950 LDS capacity stress descriptor growth and Inline Shadow placement without inventing a synthetic workload. |
| P1 | `gemm/gfx950/subtile_mxfp8.yaml`, reduced Stream-K case | 🩶 Not packaged | 🩶 Not packaged | 🩶 Not packaged | 🩶 Not packaged | Low-precision Stream-K with direct-to-LDS broadens access widths and high-pressure placement beyond FP32. |
| P2 | `gemm/gfx950/i8_gsu_gfx950.yaml` | 🩶 Not packaged | 🩶 Not packaged | 🩶 Not packaged | 🩶 Not packaged | GlobalSplitU multiple-buffer execution supplies a non-Stream-K split-reduction control; retain it only if inventory shows synchronization beyond already selected rows. |

Static YAML features are selection evidence only.  Generated code-object
inventory determines whether a candidate actually contains admitted barriers,
ordered atomics, fences, or LDS accesses, and numeric execution determines
which generated solution enters the denominator.

## PyTorch expansion

The existing PyTorch runner has portable exact-oracle operators that are
appropriate gfx950 candidates.  The installed PyTorch build reports the
physical MI355X and includes gfx950 in `torch.cuda.get_arch_list()`, but an
ordinary `torch.arange` device operation and the first `torch.mode` oracle both
fail with `hipErrorInvalidImage`.  Therefore this PyTorch installation is not
runnable on gfx950 despite its advertised architecture list.  Every profile
cell is red at the shared uninstrumented-runtime prerequisite; ConSan has not
yet been invoked.  A gfx950-compatible PyTorch installation and a passing
one-repetition baseline are required before profile assessment.

The tensor-descriptor-add and cluster-synchronization rows from the gfx1250
ledger are intentionally not copied.  Their runner implementations call
target-specific descriptor or cluster APIs.  A gfx950 row must instead come
from a native gfx950 lowering (for example the Tensile cluster-local-read rows
above), not from renaming architecture-specific source.

| Priority | Tracking unit | SuperCollider | Record/Replay | Sampled | Inline Shadow | Why it matters and next proof |
|---|---|---|---|---|---|---|
| P0 | `torch.mode`, large rows | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | Exact values/indices; historically produces a dense LDS/barrier object plus ordered LDS operations on gfx1250.  Inventory the independently generated gfx950 object rather than reusing those counts. |
| P0 | `torch.topk`, FP64 spill and BF16 coverage cases | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | Exact values/indices across FP64 register pressure and BF16 access-width breadth.  This is the primary PyTorch spill and dense-placement stress row. |
| P1 | `torch.sort` over segmented rows | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | Exact values/indices and a dense synchronized sorting kernel; useful after `topk` to distinguish operation-specific from generic placement failures. |
| P1 | `torch.histc` with a shared-memory-sized bin count | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | Exact bin counts; expected to exercise LDS atomics and barriers.  Fresh gfx950 inventory must classify atomic access and ordering roles independently. |
| P2 | Collision-heavy `torch.scatter_reduce` (`sum`, BF16 and FP32) | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | Exact collision sums and atomic-heavy execution.  Keep only if gfx950 inventory finds an applicable synchronization role or useful atomic-access coverage. |
| P2 | `torch.linalg.vector_norm` and large-row `torch.softmax` | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | 🩶 Baseline blocked: invalid device image | Exact 3-4-5 norms and CPU-referenced softmax reductions.  Provides smaller reduction/barrier objects than sorting and top-k. |

The first PyTorch step is to install or build a wheel whose ordinary device
kernels execute on this gfx950, then pass one uninstrumented repetition of
each exact oracle.  Registry enablement and a single all-profile clean pass to
inventory target-native objects follow.  Do not carry gfx1250 site counts,
fault selectors, timing ratios, or applicability decisions into this table.

## Evidence baseline

The following establishes machine access and corpus availability.  It is not
instrumentation acceptance evidence.

| Item | Current evidence |
|---|---|
| Working branch | `users/bjacob/sanitizers` |
| Survey base | `4495672ad45f3b90d0e367916e3231420a5be579`; this refresh records candidates above that committed implementation state |
| Retained first-campaign provenance | The detailed native evidence below was produced on the former `users/bjacob/consan-gfx950-take2` line.  It remains useful compatibility evidence, but is not current-tip acceptance evidence. |
| Device | AMD Instinct MI355X, `gfx950`, wave64 |
| ISA | `amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-` |
| Driver/runtime | ROCk 6.14.14; workspace TheRock HSA runtime 1.21 |
| ROCm distribution | `$WORKSPACE_ROOT/TheRock/build/dist/rocm` |
| Physical dispatch smoke | On 2026-07-22, workspace TheRock `rocminfo` reports MI355X / `gfx950:sramecc+:xnack-`.  Five native CDNA4 hip-moi host-reference tests pass in 92--188 ms, and corpus `hip_matmul` m128³ passes correctness for all three selected MFMA/shared-memory kernels. |
| Validation corpus | `iree-test-suites` `49f46d6d4370e5aa0a6367751474e20c6c4e95c0`; required Sharktank assets present; LFS fsck clean |
| Validation doctor | `f5c91c6d1d`: target-aware registry and workload-scoped doctor are active. The gfx950 `d128-block` doctor passes with the current hook, CDNA4 executable, workspace TheRock `rocminfo`, and hip-moi source/build. The all-workload doctor now resolves all five available hip-moi roles to CDNA4/MFMA artifacts and isolates the remaining missing inputs to the Qwen build tree plus a true CDNA4 Jakub counterpart. |
| RocJitsu test corpus | `rocjitsu-test-corpus` `aa54cc86c9ebff3eb840743b36ff8d9b3b2d43c4`; gfx950 enables source-built HIP matmul, HipKittens, and rocBLAS cases.  Its packaged Tensile artifacts are gfx1250-only. |
| gfx950 Tensile source pool | `rocm-libraries` `c2fafc16393d0ce47a0a5801d827d43f0d3714a4`; 36 gfx950 GEMM YAMLs are available for reduction and packaging, but none is presently a validation-registry row. |
| PyTorch discovery | The available PyTorch build advertises gfx950 support and sees the MI355X, but both a basic device operation and `torch.mode` fail with `hipErrorInvalidImage`.  Portable operator rows remain gray pending revalidation, with this baseline blocker recorded explicitly; existing validation-registry entries are also gfx1250-only. |
| Registry boundary | Current gfx950 target-specific resolution covers the hip-moi gtest roles only.  The corpus and PyTorch tables above are a planned expansion, not claims that `consan_validation.py` can run those rows today. |

## Implementation evidence

Every row below is gray because it is an inventory of earlier implementation
evidence, not a current-tip acceptance result.  Commit IDs and artifacts are
retained to guide focused revalidation; labels such as “host-proven” and
“native-proven” describe what the earlier campaign established only.

This table records prerequisite implementation gates.  They do not promote a
workload/profile cell without the end-to-end evidence above.

| Area | Status | Evidence |
|---|---|---|
| CDNA4 fixed scratch encoding | 🩶 Host-proven | `319950aacd`: exact two-word `scratch_store_dword` / `scratch_load_dword` encodings decode as 8-byte CDNA4 instructions; aligned offsets through 4092 accepted and 4096 rejected. |
| CDNA4 private wait | 🩶 Host-proven | `319950aacd`: `s_waitcnt vmcnt(0)` exact word `0xbf8c0f70`; wrong-architecture requests rejected. |
| CDNA4 scalar control builders | 🩶 Host-proven | `38cf90d7c2`: 13 exact LLVM gfx950 words and decoder round trips cover EXEC/VCC/SCC preservation, compares, and conditional branches in the separate CDNA4 backend. |
| CDNA4 vector builders | 🩶 Host-proven | `113cd36228`: exact LLVM bytes and decoder round trips cover arithmetic, comparisons, lane identity, and address updates. Literal add/multiply use materialization because gfx950 rejects VOP3 literal operands; in-place multiply exposes its extra-VGPR requirement. |
| CDNA4 SMEM/FLAT builders | 🩶 Host-proven | `4c51920eb2`: exact LLVM bytes and decoder round trips cover dispatch loads and B32 publication load/store, including CDNA4 pair alignment and immediate boundaries. |
| CDNA4 DS/atomic builders | 🩶 Host-proven | `73241ff07f`: exact LLVM bytes and decoder round trips cover LDS store/exchange and B32/B64 FLAT atomics; tuple alignment, return forms, and device scope fail closed. |
| CDNA4 synchronization/cache builders | 🩶 Host-proven | `f34be1ae72`: exact LLVM bytes and decoder round trips cover VM/LGKM drains, pre-drained workgroup barrier, trap, SALU dependency NOP, and scalar I/D-cache controls. |
| Architecture-neutral builder dispatch | 🩶 Host-proven | `b21df67176` and `a0b96f50ce`: scalar control, EXEC/VCC/SCC, vector arithmetic, literal recipes, address arithmetic, SMEM/FLAT/DS publication, atomics, private scratch, semantic waits, and workgroup barriers select separate CDNA4/RDNA4 backends. Five dispatch tests cover fixed and variable instruction lengths plus unsupported-architecture rejection. Engine-level gfx950 emission remains the separate active node `B5B`. |
| Native-shape CDNA4 analysis corpus | 🩶 Initial slice | `0cb03769ea`: a synthetic gfx950 ELF carries exact LLVM CDNA4 `ds_write_b32`/`ds_read_b32` encodings. ConSan decodes both eight-byte instructions, extracts the expected VGPR operands, classifies both as supported LDS sites, and marks the kernel as a preflight candidate. Wider DS, FLAT, atomic, barrier, and exclusion coverage remains in `A0-A4`. |
| CDNA4 FLAT atomic inventory/address plan | 🩶 Initial slice | `1806f7c90d` and `089fdf0585`: a native two-word gfx950 `flat_atomic_add` retains exact address/data/result VGPRs, offset, return mode, normalized B32 width, and device scope. Its flat guest-pair address plan is supported without materialization, and the exact compiler `buffer_wbl2`/wait/returning-atomic/wait/`buffer_inv` shape associates as one conservative acquire-release event. Wider operations, global forms, fences, and fault mutation remain open in `A4`. |
| CDNA4 group-FLAT access inventory/emission | 🩶 Native vertical | `6e997a6a39` and `31552da206`: CDNA4 raw FLAT decoding retains op, segment, address/data/result VGPRs, offset, and cache fields; explicit `SRC_SHARED_BASE` provenance admits native two-word D16/D32/D64/D128 group accesses without weakening strict provenance. The retained `d128-block` Record/Replay run at `consan-validation/gfx950-take2-d128-block-record-007` executes successfully with all 12/12 admitted accesses and all 4/4 shared-function barriers patched and dynamically complete. The newly exercised many-candidate path uses target-aware CDNA4 `s_getpc_b64`/`s_setpc_b64` indirect islands. Broader FLAT forms and campaign-level fault/performance evidence remain open, so `A2` is active and no workload cell is promoted. |
| CDNA4 far shared-function ownership | 🩶 Native clean vertical | `b58799fda0` retains execution-owner descriptors directly on ordinary FLAT sites, allowing SuperCollider to construct far local indirect relays independently of MOI resource plans. A CDNA4 host regression exercises a recovered local call and liveness-proven dead-SGPR fallback; all 591 ConSan host tests pass. Retained artifact `consan-validation/gfx950-take2-d128-pressure-clean-005` accepts all four profiles with 12/12 accesses under each engine, 4/4 barriers under Record/Replay and Inline Shadow, passing workload oracles, and zero dynamic-incomplete encounters. `V5` is active pending the remaining application matrix, faults, overhead, and frozen-tip rerun. |
| CDNA4 SuperCollider native LDS checks | 🩶 Native-proven | `afbca95de0` and `3e7c5bdf02` port native LDS check/trap emission without routing gfx950 through RDNA4 encodings. Exact host coverage includes B32/B64/B128 reads and writes, transpose B16, read2/read2st64, non-contiguous write2st64 operands, and even-aligned automatic-report address tuples. All 601 ConSan host tests pass. Retained CLIP, TP1 prefill, TP1 decode/combined, and TP2 artifacts accept 45/45, 120/120, 240/240, and 936/936 accesses respectively with passing workload oracles, analysis/static/dynamic completeness, and zero mismatches. `SC0` is green/DONE; campaign cells remain orange pending faults, overhead, and frozen-tip evidence. |
| CDNA4 SuperCollider group-FLAT checks | 🩶 Native clean vertical | `c7ca36d375`: SuperCollider handles gfx950's two-word group-FLAT D16/D32/D64/D128 loads and stores, uses U16 comparison for short values, emits the CDNA4 VM/LGKM completion wait, preserves RDNA4's established bytes, and even-aligns the report address tuple. Retained artifact `consan-validation/gfx950-take2-d128-block-supercollider-006` is analysis-, static-, and dynamic-complete with 12/12 supported accesses patched and zero report mismatches. The coverage denominator excludes global, private, and provenance-unknown FLAT operations outside this shared-memory detector's semantics. Racy/fault detection, trap breadth, and campaign evidence remain open in `SC1`. |
| CDNA4 singleton-barrier fault mutation | 🩶 First contained campaign | `8c8480e68f` makes the reviewed selector target-aware: gfx950's single full `s_barrier` is selected by exact site identity, while gfx1201 retains exact site-plus-sequence selection for its associated signal/wait pair. Retained artifact `consan-validation/gfx950-take2-d128-block-fault-003` is accepted for all four profiles with exact-one `requested=1`, `planned=1`, `applied=1` mutation accounting, the precommitted `not_detected` qualified miss and workload-oracle failure, bounded termination, and healthy before/after probes. This activates `A5` and the D128 slice of `V6`; other mutation families and workload campaigns remain open. |
| D128 paired overhead | 🩶 First accepted timing row | `ecd529c017` selects the gfx950 `SampledFastContext` specialization whose native execution produces all admitted group-FLAT evidence; the superseded Exact-only attempt produced no Record/Replay records and was rejected. In retained artifact `consan-validation/gfx950-take2-d128-block-overhead-002`, baseline-before, all four three-process profile rows, and baseline-after are accepted. The paired baseline is 130.5 ms; process slowdowns are 6.61x SuperCollider, 8.35x Record/Replay, 8.31x Sampled, and 9.29x Inline Shadow. `V7` remains active pending peak-memory evidence and the broader matrix. |
| CDNA4 Record/Replay access emission | 🩶 Native vertical | `4fdaee5298`, `44e7767b9b`, and `3a52aaee88`: a gfx950 DS-store candidate emits a `ModifiedValid` first-light access-record patch and passes final re-decode. The body retains the guest DS access and contains CDNA4 FLAT publication load/atomics, VM/LGKM completion waits, and two-word lane-rank recipes. A production-hook native dispatch now writes and validates the access record; replay breadth remains open in `RR0`. |
| CDNA4 Sampled access emission | 🩶 Native vertical | `ec84cd7af7` and `2f9cc947b0`: a gfx950 DS-store candidate emits a `ModifiedValid` Sampled patch and passes final re-decode. The body contains CDNA4 FLAT swap-x2 publication, compare-swap claim, counter atomics, VM/LGKM completion waits, and the displaced guest DS access. A production-hook forced-spill dispatch preserves all live values and publishes a valid write entry plus ready causal window. Runtime selection and immediate/host-scan agreement remain open in `SA0-SA1`. |
| CDNA4 Sampled barrier lowering | 🩶 Native-proven vertical | The report-buffer target gate now admits both supported native architectures, with a CDNA4 host fixture proving a selected LDS causal window followed by singleton `s_barrier` metadata publication and final validation. Retained native artifacts `gfx950-take2-tp1-prefill-sampled-clean-004`, `gfx950-take2-tp1-decode-sampled-clean-002`, and `gfx950-take2-clip-bf16-sampled-clean-002` accept 104/104 + 7/7, 208/208 + 14/14, and 39/39 + 20/20 access/qualified-barrier coverage respectively, with passing workload oracles and zero dynamic-incomplete state. TP2 admission remains open in `SA0`. |
| CDNA4 sampled scalar-state boundary | 🩶 Native-proven | `2da1350b8e`: gfx950 dump inspection proved that encoded `s102:s103` is architectural `flat_scratch`, not ordinary scalar storage. Automatic owner, dispatch, and temporary-state placement now stops below CDNA4's special aliases; explicit overlap fails closed. Removing the obsolete runtime-sample counter and unused sampled-atomic tail reservation fits the high-pressure Stream-K helper in `s92:s101`. All 594 ConSan host tests pass, and retained artifact `consan-validation/gfx950-take2-streamk-arrival-clean-005` accepts native Sampled execution with 4/4 accesses and 10/10 atomics instead of the prior `0x1000` GPU fault. |
| CDNA4 Sampled forced spill | 🩶 Native-proven | `2f9cc947b0`: the production HSA hook forces a three-VGPR Sampled spill around native `ds_write_b32`, grows dispatch-private storage, and completes native AQL dispatch. Eight live values survive for all 64 lanes; the sampled entry and causal window agree on kind, epoch, generation, range, and 1D workgroup identity. |
| CDNA4 InlineShadow access/barrier vertical | 🩶 Native-proven TP1 lifecycle | A gfx950 DS-store candidate emits `ModifiedValid` and passes final re-decode plus exact-shadow semantic validation; workgroup-local forced-spill publication and global versioned transactions both have host byte proof. The global table hashes dispatch plus stable workgroup identity across 32 banks, and metadata-distinct lanes sharing one address are serialized by the pending-mask loop. Sequential decode+combined stress localized the residual loss to stable-version snapshots of stale nonempty payload. CDNA4 retries now issue `buffer_inv sc1`, consume a coherent opening version, and leave the ordinary fast path unchanged. Artifacts `gfx950-take2-tp1-decode-combined-inline-invalidate-retry-026` through `-031` provide six consecutive accepted runs at 208/208 accesses, 62/62 barriers, passing oracles, and zero dynamic-incomplete state. Prefill artifact `gfx950-take2-tp1-prefill-inline-serialized-lanes-037` remains accepted at 104/104 + 31/31, and `gfx950-take2-d128-block-inline-004` at 12/12 + 4/4. Atomic semantics, faults, and broader campaign evidence remain open in `IS0-IS1` and validation nodes. |
| Mixed CDNA4 AccVGPR/VCC persistent identity | 🩶 Native-proven | Moving compiler `ACCUM_OFFSET` remains prohibited because it changes existing MFMA operand meaning. Per-owner/component tuples avoid AccVGPR banks; private-state entry relays preserve kernarg preloads; persistent epoch/owner/workgroup-key slots remain disjoint from ephemeral spill storage; and appended probes snapshot displaced DS addresses. Automatic scalar state also excludes the original physical VCC pair at the top of each CDNA4 kernel's allocated user-SGPR bank. Full TP1 artifact `gfx950-take2-tp1-prefill-inline-vcc-safe-029` moves the attention kernel's Inline state from overlapping s70:s71 to s72 and changes a 533,757-incomplete oracle failure into a passing oracle with 27 broader undercoverage publications. A direct native run filtered to `prefill_bs1$async_dispatch_12_attention_4x2xDx32x32xD` accepts 4/4 accesses and 6/6 barriers with `dynamic_complete=true`, zero incomplete publications, and a passing oracle. Together with retained CLIP artifact `gfx950-take2-clip-bf16-inline-clean-021` at 39/39 + 24/24 and zero incomplete state, this closes `IS0A` and `I4`; the formerly separate full-workload publication residue is closed by the TP1 access/barrier row. |
| CDNA4 InlineShadow forced spill | 🩶 Native-proven | `54673f205e`: a full-VGPR-file fixture forces the 16-register Inline Shadow window into private spill storage. The local exact-shadow path uses even CDNA4 `ds_wrxchg_rtn_b64` and diagnostic CAS tuples, passes final semantic validation, and completes native AQL dispatch with all live values intact and zero diagnostic, overflow, unsupported, or malformed counts. |
| CDNA4 InlineShadow atomic ordering | 🩶 Native Stream-K vertical, AcqRel token import open | `089fdf0585` established compiler-shaped gfx950 acquire-release emission and structural validation. Checkpoint `255749430d` extends the CDNA4 reservation to 28 VGPRs so token producer state, scalar persistent owner/epoch materialization, and the even final address pair are disjoint; all 622 ConSan host tests pass there. Artifact `gfx950-take2-streamk-scalar-state-011` executes 4/4 accesses, 4/4 barriers, and 10/10 atomics but exposes the empty-release-slot race. Artifacts `gfx950-take2-streamk-claimed-acqrel-013` through `-016` localize lost VGPR and SGPR journals. In `-017`, scratch+8 seeding plus retention of the outer claim mask in the dead retry pair lets wave 1 stage and commit version 2; wave 2 then leaves a fully staged version-3 reservation. The remaining rejection is now the nonempty-predecessor token transaction: one malformed token, one overflow, no visible acquired token, and the old false diagnostic. Successor commit, repeated clean acceptance, fault detection, and full host semantic regression remain open in `IS1`. |
| Transactional static VGPR spill plan | 🩶 Host-proven | `319950aacd`: stable four-byte slots, pre-save drain, 16-byte required-private alignment, 4 KiB bound, failure rollback, and unchanged RDNA4 behavior. |
| CDNA4 MOI resource alignment/growth | 🩶 Host-proven | `44e7767b9b`: gfx950 descriptors decode and grow in eight-VGPR granules throughout planning and emission. Automatic first-light planning grows 8 to 16 VGPRs, all CDNA4 scratch windows use an even FLAT-address base, and odd explicit bases fail closed as `ExplicitMisaligned`. |
| CDNA4 Record/Replay forced spill | 🩶 Native-proven | `44e7767b9b` and `3a52aaee88`: an integrated first-light patch spills three VGPRs around the emitted probe using exact native CDNA4 scratch save/restore words and VM_CNT waits, preserves the displaced guest DS instruction, and passes final validation. The production hook adds the CDNA4 owner/epoch entry prologue, grows the live kernel object's private segment from 0 to 16 bytes, and completes native AQL dispatch with all live values intact and a valid LDS-write record. |
| Shared/private layout policy | 🩶 Native-proven | `97f978941d` and `c4b0c3a1a4`: architecture-typed 8 MiB/4 KiB limits and gfx950 16-byte final normalization feed one shared-helper layout. A native noinline LDS helper has two full-VGPR-file owners; the production hook selects one three-VGPR spill window, grows both live kernel objects from 0 to 16 private bytes, and both AQL dispatches preserve output while publishing a valid write record. |
| CDNA4 Record/Replay barrier forced spill | 🩶 Native-proven | `33075e7aac`: CDNA4 barrier-record lowering uses the target-neutral builder dispatch and passes final validation. A full-VGPR-file native kernel emits separate three-VGPR access and six-VGPR barrier spill trampolines, grows live dispatch-private storage from 0 to 32 bytes, preserves eight values for all 64 lanes, and publishes both access and barrier records. |
| CDNA4 Record/Replay atomic forced spill | 🩶 Native-proven | `6f42088f3d`: qualified CDNA4 cache-release/atomic/cache-acquire ordering no longer requires an RDNA-only `TH` field. A native full-VGPR-file kernel emits and final-validates a three-VGPR atomic-record spill, grows live private storage from 0 to 16 bytes, preserves eight values for all 64 lanes, executes exactly 64 increments, and publishes the expected address, scope, and raw-zero CDNA semantics. Companion-fence parity is recorded separately below. |
| CDNA4 Record/Replay fence spill parity | 🩶 Typed fail-closed | `af03dfa4c4`: CDNA4 reaches the same deliberate second-text-growth limitation as RDNA4. Both qualified companion fences retain spill plans but emit no guessed patch; each final ledger entry is `placement_or_lowering_failed` / `instrumentation_patch_missing`, accompanied by the explicit second-growth spill warning. |
| CDNA4 dynamic-stack spill policy | 🩶 Native-proven / typed fail-closed | `8068d7d018` and `4dcba398f9`: Inline Shadow uses a site-local, SCC-preserving frame around the compiler `s32:s33` stack convention, with exact explicit-`saddr` CDNA4 scratch words and VM_CNT waits. Host integration final-validates a 16-VGPR spill and 64-byte private requirement. A production-hook gfx950 dynamic-allocation kernel forces the same recipe and preserves all 64 private-stack values without diagnostics. Record/Replay deliberately remains unpatched with typed `DynamicStack` / `ResourceFailed` / `UnsupportedResourcePlan` evidence. |
| Instrumented group-segment dispatch growth | 🩶 Native-proven | The production HSA hook now propagates each instrumented kernel's required group bytes through cached symbol metadata and copied AQL dispatch packets, independently of the existing private-segment transaction. Retained TP1 artifact `consan-validation/gfx950-take2-tp1-prefill-inline-clean-027` proves the affected attention dispatch grows from 768 to 2304 group bytes. Its unchanged oracle failure and 533,757 dynamic-incomplete publications ruled out underallocated LDS as the identity root cause; the subsequent physical-VCC repair is recorded in the mixed-identity row. |
| Focused host gate | 🩶 606/606 focused | The complete 604-test `ConSan*.*` host suite plus two CDNA4 cache-control builder tests pass in the canonical `/home/ossci/xx/rocjitsu-build`. Exact encoding/decoder proof pins `buffer_inv sc1`, and the CDNA4 Inline emitted-code regression requires it in both versioned retry paths. Earlier physical-VCC, private epoch/owner/workgroup-key, entry-save, and displaced-address snapshot regressions remain covered. |
| Live CDNA4 scratch round trip | 🩶 Native-proven | `0ad99bd3ac`: full-wave64 and divergent even-lane native MI355X round trips pass. Code-object inspection retains the exact scratch store/load pairs, `0xbf8c0f70` waits, partial-`EXEC` save/restore, wave64 metadata, and 32 private bytes. |

## Frozen campaign provenance

Not established.  Before the first green cell this section must name one
committed executable revision, freshly rebuilt hook SHA-256, target-aware
manifest and audit hashes, workload source revisions, exact artifact root, and
the controlled runtime/toolchain environment.  The final campaign must rerun
every accepted row at the same frozen tip.

## Promotion requirements

A workload/profile cell remains non-green until its retained evidence proves:

1. the standard-v1 profile is used, with no forbidden coverage-limiting or
   manually selected register controls;
2. the independent numerical or semantic oracle passes on the clean workload;
3. ConSan sees an applicable gfx950 code object, records the complete typed
   exclusion ledger, and patches every admitted supported access, barrier,
   atomic, and fence site;
4. the selected engine executes dynamically without incomplete state,
   forbidden overflow, or unexpected clean diagnostics;
5. each admitted fault has a freshly inventoried and reviewed gfx950 selector,
   exact-one final-byte mutation proof, and a precommitted detector/oracle
   outcome;
6. every fault terminates within its bound and retains before/after device
   health evidence; and
7. one paired baseline/profile repetition, peak memory, commands, hashes,
   timeout state, and retained artifact paths are recorded at a frozen
   campaign revision.

Clean execution alone is compatibility evidence, not a green cell.  A timeout,
trap, crash, output mismatch, or GPU reset is an execution outcome rather than
a ConSan detection.

## Progress log

- 2026-07-22: Began current-tip physical revalidation with D128 block artifact
  `consan-validation-gfx950-d128-block-clean-tip-20260722-001`.  The baseline
  passes both exact host-reference oracles.  SuperCollider passes both oracles
  with complete 12/12 access coverage and static/dynamic analysis.  Record/
  Replay and Sampled safely reject persistent VGPR state at the kernels'
  ordinary-VGPR/AccVGPR boundary; Inline Shadow separately selects scalar
  state for one connected component but fails to assign entry-local scratch
  for another.  Provenance is clean at `2deee22c30`, and postflight physical
  gfx950 discovery remains healthy.  These are current placement regressions,
  so the three failing cells are red rather than inheriting old acceptance.

- 2026-07-22: Rechecked the actual physical device rather than relying on the
  retained campaign.  Workspace TheRock `rocminfo` reports MI355X / gfx950,
  five exact native CDNA4 hip-moi tests pass, and the source-built corpus
  `hip_matmul` m128³ case passes correctness for all three selected kernels at
  one iteration.  Workload-scoped doctor checks define every current-matrix
  row: TP1, TP2, CLIP, and the five native hip-moi roles are runnable; Qwen is
  blocked on its gfx950 VMFB and Jakub on its target-native executable.
- 2026-07-22: Executable audits now separate corpus rows that run today from
  source-only, compile-disabled, and wrong-target cases.  The available
  PyTorch installation is not usable evidence: although it advertises gfx950
  and sees the physical device, even a basic device kernel and `torch.mode`
  fail with `hipErrorInvalidImage`.  All portable PyTorch profile cells remain
  gray pending revalidation, with the shared baseline blocker called out in
  each cell until a compatible build exists.
- 2026-07-22: Refreshed this ledger to use the same four-profile matrix and
  red/orange/yellow/green/gray maturity scale as the expanded gfx1250 ledger.
  Every current status is gray because the substantially changed current tip
  requires complete revalidation; old campaign results remain only as
  historical annotations.
- 2026-07-22: Surveyed `rocjitsu-test-corpus` at `aa54cc86c9eb`.  Native
  gfx950 HIP matmul and HipKittens BF16 cases are the first packaged corpus
  candidates.  The two four-wave low-precision cases are compile-skipped,
  hip-stream-k metadata is gfx942-only, and the packaged Tensile artifacts
  are gfx1250-only.  These distinctions are now explicit rather than silently
  treating source availability as runnable validation support.
- 2026-07-22: Surveyed the 36 gfx950 GEMM YAMLs in `rocm-libraries` at
  `c2fafc16393d` and selected a bounded Stream-K, LDS-transpose, 160-KiB LDS,
  low-precision, and split-reduction follow-on set.  All remain gray until
  reduced, generated, numerically checked, packaged, inventoried, and added to
  the validation registry.
- 2026-07-22: Mapped only architecture-neutral PyTorch operator rows to the
  proposed gfx950 expansion.  Existing PyTorch validation rows are currently
  target-restricted to gfx1250, and target-specific descriptor/cluster source
  was deliberately excluded.  Registry enablement and fresh gfx950 code-object
  inventories precede any profile assessment.

- 2026-07-18: Stream-K Inline Shadow on the current safe planner is rejected
  before installation, not accepted as the older matrix wording implied.
  Both dynamic-stack kernel owners reach the same full CDNA4 ordinary-VGPR
  boundary, and their connected helper needs owner/epoch/workgroup identity;
  allocating the old v112:v114 tuple would move `ACCUM_OFFSET` semantics and
  corrupt MFMA operands.  Artifact
  `gfx950-take2-streamk-inline-after-retry-001` records the typed planning
  rejection.  `IS1` remains orange/ACTIVE while a scalar per-wave persistent
  representation is implemented; no unsafe VGPR growth is being restored.
- 2026-07-18: The TP1 Inline Shadow lifecycle gap is closed while `IS0` and
  `IS1` remain orange/ACTIVE for atomic breadth and campaign evidence.  Split
  telemetry proved every residual sequential decode+combined miss was a
  stable-version rejection of stale nonempty payload; no claim, CAS, commit,
  changed-version, odd-version, or terminal-version failure was involved.
  CDNA4 retry paths now execute `buffer_inv sc1`, retain the coherent atomic
  opening version, and preserve ordinary-load performance on the uncontended
  fast path.  Six consecutive retained authoritative runs (`-026` through
  `-031`) accept 208/208 accesses and 62/62 barriers with passing oracles and
  `dynamic_incomplete=0`.  The synchronized plan box remains orange, and 606/606
  focused host tests pass.
- 2026-07-17: CDNA4 Sampled barrier lowering is native-proven.  The prior
  architecture gate was the direct cause of TP1/CLIP's missing supported
  barriers; a focused CDNA4 singleton-barrier patch test now passes alongside
  the existing CDNA4 atomic test.  Retained native TP1 prefill, TP1
  decode/combined, and CLIP artifacts accept 104/104 + 7/7, 208/208 + 14/14,
  and 39/39 + 20/20 access/qualified-barrier coverage with passing workload
  oracles and no dynamic-incomplete state.  The TP2 aggregate still rejects
  one 140-access slice because Sampled selects no applicable patch there, so
  `SA0A` is green/DONE while `SA0` and `V5` remain orange/ACTIVE.  The Mermaid
  class assignments are synchronized with those labels.
- 2026-07-17: Commit `3e7c5bdf02` completes the native-LDS application
  breadth exposed after the CLIP vertical.  Automatic B32 reporting now pads
  to an even CDNA4 FLAT address tuple, while `ds_read2_b64` and
  `ds_read2st64_b64` duplicate both 64-bit addresses into a contiguous result
  tuple.  All 601 ConSan host tests pass.  Retained artifacts
  `consan-validation/gfx950-take2-tp1-prefill-sc-clean-005`,
  `consan-validation/gfx950-take2-tp1-decode-combined-sc-clean-003`, and
  `consan-validation/gfx950-take2-tp2-family-sc-clean-003` are accepted at
  120/120, 240/240, and 936/936 native accesses with passing oracles and full
  static/dynamic coverage.  The synchronized `SC0` DAG node is green/DONE;
  workload cells stay orange pending their remaining campaign requirements.
- 2026-07-17: Immediate native Sharktank reruns after `afbca95de0` retain the
  earlier 0/120, 0/240, and 0/936 SuperCollider coverage.  They now expose a
  precise error rather than an architecture gate: a B32 check places the
  automatic report address pair at an odd CDNA4 VGPR.  The whole code object
  correctly rolls back on the unencodable report action.  The CLIP 45/45 row
  remains accepted evidence, while synchronized `SC0` returns to orange/ACTIVE
  until report-scratch parity is fixed and the broader reruns pass.
- 2026-07-17: Commit `afbca95de0` establishes the initial CDNA4 native-LDS
  SuperCollider CLIP vertical.  Exact host tests cover ordinary reads/writes,
  transpose B16 reads, and the non-contiguous data operands of
  `ds_write2st64_b64`; all 599 ConSan tests pass.  Retained artifact
  `consan-validation/gfx950-take2-clip-bf16-sc-clean-005` accepts CLIP BF16
  with its independent oracle passing, all 45/45 supported native LDS accesses
  patched, analysis/static/dynamic completeness, and zero mismatches.  The
  CLIP cell remains orange until its fault, overhead, and frozen-tip evidence is
  complete.  The immediate Sharktank audit above subsequently broadened its
  implementation exit criterion.
- 2026-07-17: Native TP1 decode/combined, TP2 family, and CLIP BF16
  reconnaissance is retained in `gfx950-take2-tp1-decode-combined-clean-001`,
  `gfx950-take2-tp2-family-clean-001`, and `gfx950-take2-clip-bf16-clean-001`.
  Record/Replay accepts all three with 208/208 + 62/62, 840/840 + 168/168,
  and 39/39 + 24/24 access/barrier coverage.  SuperCollider consistently
  patches 0 supported accesses; Sampled consistently omits supported barriers
  and additionally misses one TP2 aggregate slice; Inline reports either
  undercoverage or dynamic-incomplete state.  All cells and synchronized DAG
  nodes remain orange/ACTIVE pending fixes and authoritative reruns.
- 2026-07-17: Retained artifact
  `consan-validation/gfx950-take2-tp1-prefill-clean-002` runs native TP1
  prefill through the workspace's IREE Python venv.  Record/Replay is accepted
  with 104/104 accesses and 31/31 barriers.  SuperCollider passes the workload
  but patches 0/120 supported accesses; Sampled passes and patches 104/104
  accesses but 0/10 supported barriers; Inline patches 104/104 accesses and
  31/31 barriers but records 28 dynamic-incomplete encounters and exits 90.
  These are orange active evidence cells, not green campaign completion; the
  synchronized `V1` and `V5` DAG nodes are orange/ACTIVE.
- 2026-07-17: Retained artifact
  `consan-validation/gfx950-take2-tree-atomic-or-clean-001` accepts tree
  atomic-OR under all four profiles with passing workload oracles and no
  unexpected diagnostics.  All profiles patch 4/4 accesses; Record/Replay
  additionally patches 10/10 atomics, 4/4 barriers, and 16/16 fences, Sampled
  patches 10/10 atomics, and Inline patches 10/10 atomics plus 4/4 barriers.
  The four cells remain orange pending fault, overhead, and frozen-tip evidence,
  and the synchronized `V5` node remains orange/ACTIVE.
- 2026-07-17: Stream-K Inline now lowers and dynamically executes all 4/4
  accesses alongside 10/10 atomics and 4/4 barriers, with zero ConSan
  diagnostics or incomplete state in accepted individual trials.  Sequential
  stress is not yet accepted: hip-moi intermittently reports its own
  `0x1c05`/`0x1c06` access conflict because its RMW release bookkeeping occurs
  after the hardware atomic and permits only four acquire retries; ConSan is
  instrumenting that nested atomic before control returns to the bookkeeping.
  The application output remains correct and ConSan remains clean.  The cell
  and synchronized `V5` DAG node therefore stay orange/ACTIVE, not green.
- 2026-07-17: Commit `2da1350b8e` fixes the native sampled Stream-K fault by
  excluding CDNA4 `flat_scratch`/`xnack_mask` aliases from ordinary scalar
  instrumentation state and removing two stale scalar reservations.  All 594
  ConSan host tests pass.  In retained artifact
  `consan-validation/gfx950-take2-streamk-arrival-clean-005`, SuperCollider,
  Record/Replay, and Sampled are accepted with 4/4 access coverage;
  Record/Replay additionally covers 10/10 atomics, 4/4 barriers, and 16/16
  fences, while Sampled covers 10/10 atomics.  Inline Shadow passes the
  workload oracle and covers 10/10 atomics plus 4/4 barriers but is rejected
  at 0/4 accesses.  All four cells are orange to distinguish three retained
  clean acceptances and one active coverage gap from frozen green campaign
  completion; the orange/ACTIVE `V5` Mermaid box names that same state.
- 2026-07-17: Retained artifact
  `consan-validation/gfx950-take2-mfma-attention-clean-001` accepts native
  MFMA attention under all four profiles.  Every row passes its independent
  workload oracle, covers 12/12 accesses, and records zero dynamic-incomplete
  encounters; Record/Replay and Inline Shadow also cover 4/4 barriers.  The
  four MFMA-attention cells are orange pending faults, overhead, and frozen-tip
  evidence, and the orange/ACTIVE `V5` box text names this added vertical.
- 2026-07-17: Commit `b58799fda0` closes the D128-pressure SuperCollider
  placement gap by retaining execution-owner descriptors on ordinary FLAT
  sites.  The focused CDNA4 far-call regression and all 591 ConSan host tests
  pass.  In retained artifact
  `consan-validation/gfx950-take2-d128-pressure-clean-005`, SuperCollider,
  Record/Replay, Sampled, and Inline Shadow all pass the workload oracle and
  patch 12/12 accesses; Record/Replay and Inline Shadow also patch 4/4
  barriers, and every row has zero dynamic-incomplete encounters.  The four
  pressure cells and `V5` are orange/ACTIVE; faults, overhead, the remaining
  application matrix, and frozen-tip evidence remain open.
- 2026-07-17: Commit `ecd529c017` corrects the gfx950-only D128 overhead
  workload selection.  The initial retained `overhead-001` experiment proved
  that the ExactContext specialization executes no admitted ConSan access
  probes on this target; its Record/Replay rows correctly failed the required
  dynamic-evidence gate.  The replacement
  `consan-validation/gfx950-take2-d128-block-overhead-002` uses the native
  `SampledFastContext` specialization and accepts baseline-before, all four
  profiles, and baseline-after across three processes each.  The paired
  baseline is 130.5 ms, with 6.61x, 8.35x, 8.31x, and 9.29x process slowdown
  for SuperCollider, Record/Replay, Sampled, and Inline Shadow respectively.
  The cells and `V7` remain orange pending peak-memory and frozen-tip evidence.
- 2026-07-17: Commit `8c8480e68f` corrects the gfx950 barrier-drop
  specification contract for CDNA4's single full `s_barrier`; RDNA4's paired
  signal/wait selector is unchanged.  The reviewed D128 selector was fixed
  before rerunning the campaign, retaining its precommitted detector and
  oracle policies.  Artifact
  `consan-validation/gfx950-take2-d128-block-fault-003` is accepted for
  SuperCollider, Record/Replay, Sampled, and Inline Shadow.  Every row applies
  exactly one mutation, produces the expected independent oracle failure and
  qualified detector miss, terminates within its bound, and passes both GPU
  health gates.  The D128 matrix cells remain orange pending overhead and a
  frozen-tip rerun; `A5` and `V6` are now orange/ACTIVE in the plan.
- 2026-07-17: Commit `c7ca36d375` ports SuperCollider's redundant
  group-FLAT check to CDNA4 without changing the established RDNA4 sequence.
  It supports two-word D16/D32/D64/D128 loads and stores, U16 short-value
  comparisons, the CDNA4 VM/LGKM completion wait, and even report-address
  tuples.  Retained artifact
  `consan-validation/gfx950-take2-d128-block-supercollider-006` is accepted
  with analysis/static/dynamic completeness, 12/12 accesses patched, and
  zero mismatches.  A permissive earlier 0/0 result is superseded rather than
  treated as evidence.  The focused, ConSan, and full gates pass 413/413,
  590/590, and 2145/2145.  `SC1` is orange/ACTIVE pending a retained racy/fault
  detection proof and the wider acceptance campaign.
- 2026-07-17: Commit `125419056d` fixes the CDNA4 Inline Shadow identity
  lifetime exposed by group-FLAT D128 attention.  Workgroup identity needed by
  shared helpers is now entry-persistent and allocated above each owning
  descriptor's full VGPR allocation, rather than relying on site-local
  liveness across arbitrary guest code.  Retained artifact
  `consan-validation/gfx950-take2-d128-block-inline-004` is fully accepted:
  12/12 accesses and 4/4 barriers are patched, analysis/static/dynamic
  coverage are complete, and dynamic-incomplete is zero.  The focused,
  ConSan, and full gates pass 413/413, 588/588, and 2143/2143.  This advances
  `IS1` to orange/ACTIVE; it does not yet complete multidimensional identity,
  fault, overhead, or frozen-campaign requirements.
- 2026-07-17: Retained four-profile artifact
  `consan-validation/gfx950-take2-d128-block-all-001` classifies the next
  slices.  Record/Replay repeats its 12/12 access and 4/4 barrier clean
  vertical.  Sampled executes with 12/12 admitted accesses.  Inline Shadow is
  static-complete at 12/12 plus 4/4 but rejects the run after accounting
  47,360 unsupported dynamic lane-site encounters: CDNA4 function-time
  workgroup SGPR sources are not stable without the entry snapshot/restoration
  transaction.  Identity nodes `I0-I3` are therefore active (orange), not TODO.
  SuperCollider still finds no applicable gfx950 site and remains an open
  implementation node; its runner-side zero-site acceptance is not treated as
  evidence.
- 2026-07-17: Commit `31552da206` completed the first native Record/Replay
  group-FLAT slice with full supported-site coverage.  D16 `flat_store_short` and
  `flat_load_ushort` now normalize to two-byte accesses.  Enabling all twelve
  workload sites exposed the previously dormant many-candidate indirect
  island path, whose RDNA-specific raw `s_getpc_b64`/`s_setpc_b64` words are
  now architecture-aware builders.  The same commit enables CDNA4 persistent
  epoch advancement at barriers.  Retained artifact
  `consan-validation/gfx950-take2-d128-block-record-007` runs successfully
  with 12/12 access and 4/4 shared-function barrier sites patched and dynamic
  evidence complete.  Nodes `A3`, `RR1`, and `Q0` are now active (orange) in
  the DAG.  The focused, ConSan, and full gates pass 413/413, 588/588, and
  2143/2143.
- 2026-07-17: Commit `6e997a6a39` started access-normalization node `A2`.
  CDNA4 two-word FLAT decode, raw-field retention, explicit shared-base
  provenance, candidate admission, and Record/Replay/Inline first-light
  emission now have exact host coverage.  In retained native artifact
  `consan-validation/gfx950-take2-d128-block-record-003`, Record/Replay runs
  successfully and patches all 8/8 admitted D32/D64 group-FLAT accesses, with
  all eight dynamically visible.  The workload remains deliberately
  non-green: four D16 short/ushort accesses and four shared-function barriers
  are still missing.  The focused, ConSan, and full rocJITsu gates pass
  410/410, 586/586, and 2140/2140 respectively.
- 2026-07-17: Commit `f5c91c6d1d` started validation-environment nodes `E1`
  and `V0`.  The manifest no longer emits RDNA4 executable or test-suite
  spellings for gfx950 hip-moi rows: D128 block/pressure, MFMA attention,
  Stream-K arrival, and tree atomic-OR all resolve to present CDNA4 binaries.
  Workload-scoped doctor checks let those ready rows proceed independently;
  `d128-block` passes its complete preflight using the current hook and
  workspace TheRock `rocminfo`.  The full doctor honestly retains two gaps:
  Qwen artifacts and a semantically equivalent CDNA4 Jakub build.
- 2026-07-17: Commits `8068d7d018` and `4dcba398f9` completed dynamic-stack
  spill node `S8C`.  Exact host tests prove CDNA4's explicit-`saddr` scratch
  encodings, SCC preservation, stack advance/restore, final-valid Inline
  Shadow integration, and Record/Replay's typed rejection.  The production
  hook also instruments a real compiler-generated gfx950 dynamic-allocation
  kernel: its forced 16-VGPR spill completes native dispatch, preserves every
  lane's private-stack value, and reports no Inline Shadow error counters.
  All nine native spill cases, the 409/409 focused gate, the 584/584 ConSan
  suite, and the full 2138/2138 rocJitsu suite pass.
- 2026-07-17: Commit `af03dfa4c4` completed fence parity node `S8D`.
  CDNA4 no longer stops at an RDNA-only architecture gate: both qualified
  fence sites reach the intentional second-text-growth spill limitation,
  retain spill resource evidence, emit no unsafe patch, and receive typed
  `placement_or_lowering_failed` / `instrumentation_patch_missing` outcomes.
  The native atomic case and the 406/406 plus 582/582 host gates remain clean.
- 2026-07-17: Commit `6f42088f3d` completed atomic spill node `S8B`.
  CDNA4's compiler-shaped acquire-release atomic sequence is accepted without
  fabricating an RDNA `TH` encoding.  The production hook forces its
  three-VGPR record probe into private scratch, grows the live kernel object
  from 0 to 16 bytes, and completes native dispatch with all 512 live values
  intact, the counter at exactly 64, and a valid atomic record.  All eight
  native spill cases, the 406/406 focused host gate, and all 582 ConSan host
  tests pass.  Fence second-growth rejection remains open in `S8D` and is not
  claimed by this row.
- 2026-07-17: Commit `33075e7aac` completed barrier spill node `S8A`.
  The native gfx950 fixture forces both its LDS access and `s_barrier` through
  production-hook private spills.  The access uses three VGPRs, the barrier
  uses six, both trampolines pass final validation, and dispatch-private
  storage grows from 0 to 32 bytes.  All eight live values survive across the
  barrier for every lane and both access and barrier records are visible.  All
  seven native spill cases, the 405/405 focused host gate, and all 581 ConSan
  host tests pass.
- 2026-07-17: Commit `c4b0c3a1a4` completed shared-layout node `S3`.
  One native noinline LDS helper is owned by two full-VGPR-file gfx950
  kernels.  The production hook reports a common three-register spill plan
  with `owners=2`, assigns private-backed persistent epoch state, grows both
  live kernel objects from 0 to 16 private bytes, and successfully dispatches
  both callers.  Their distinct 64-lane outputs survive and the report contains
  a valid LDS-write record.  All six native spill cases, the 404/404 focused
  host gate, and all 580 ConSan host tests pass.
- 2026-07-17: Commit `54673f205e` completed Inline Shadow spill node
  `S7B`.  CDNA4's workgroup-local path now rotates only its B64 DS-exchange
  return and first-diagnostic CAS tuples to even VGPR pairs, leaving the global
  versioned transaction unchanged.  Host final validation recognizes and
  proves the native instruction shapes.  The production-hook native dispatch
  spills 16 VGPRs, preserves all live output, publishes required evidence, and
  leaves every diagnostic/error counter zero.  The focused host gate is
  404/404, all ConSan host tests are 580/580, and all five native spill cases
  pass serially.
- 2026-07-17: Commit `2f9cc947b0` completed Sampled spill node `S7A`.
  The production-hook gfx950 fixture forces three spilled VGPRs, preserves
  eight live values for all 64 lanes through native AQL dispatch, and
  publishes a valid LDS-write sampled entry with a ready, matching causal
  window.  All four native spill cases—standalone full-wave and
  partial-`EXEC`, Record/Replay, and Sampled—pass serially on the MI355X.
- 2026-07-17: Commit `3a52aaee88` completed native spill nodes `S4` and
  `S6B`.  A dedicated gfx950 HIP fixture runs through the production HSA hook,
  forcing a three-VGPR Record/Replay spill around a native `ds_write_b32`.
  The patch includes a newly enabled CDNA4 owner/epoch entry prologue, passes
  final validation, grows the live dispatch-private requirement from 0 to 16
  bytes, preserves eight live values for all 64 lanes, and publishes a valid
  LDS-write access record.  The native full-wave, partial-`EXEC`, and
  hook-driven cases pass serially; the focused host gate is 403/403 and all
  ConSan host tests are 579/579.
- 2026-07-17: Commit `44e7767b9b` completed host-integrated spill node
  `S6A`.  It fixed two resource assumptions exposed only by gfx950 integration:
  descriptor VGPR granularity is eight, and every ConSan scratch window must
  begin at an even VGPR for CDNA4 FLAT addresses.  Automatic 8-to-16 VGPR
  growth, typed odd-base rejection, and an exact three-VGPR Record/Replay
  spill around a retained native DS instruction all pass.  The spill grows a
  preexisting 32-byte private segment to 48 bytes and survives final
  validation.  The focused gate is 402/402 and all ConSan host tests are
  578/578.  Its then-open native dispatch/AQL evidence was completed by
  `3a52aaee88`.
- 2026-07-17: Commit `089fdf0585` completed the first CDNA4 InlineShadow
  atomic-ordering vertical.  The exact gfx950 compiler acquire-release shape
  is associated, admitted, emitted, and structurally revalidated; exact
  instruction assertions retain the guest atomic, two release CAS operations,
  fifteen acquired-token CAS operations, and semantic waits.  CDNA4 scratch
  rotation makes every FLAT address and CAS tuple even-aligned while leaving
  RDNA4 allocation unchanged.  The focused gate is 400/400 and all ConSan
  host tests are 576/576.  This advances `A4`/`IS0`; automatic-resource,
  forced-spill, broader synchronization, and native evidence remain open.
- 2026-07-17: Commit `1806f7c90d` added the first native CDNA4 FLAT-atomic
  inventory and supported address plan.  Exact ordered synchronization and
  inline-atomic emission remain open.  All ConSan host tests are 574/574.
- 2026-07-17: Commit `b0ddc85877` produced a fully validated CDNA4
  InlineShadow access patch.  It also routed scalar subtraction through the
  target-neutral builder and taught exact-shadow final validation to inspect
  both in-place and trampoline bodies.  The focused gate is 399/399 and all
  ConSan host tests are 573/573.  This starts `IS0`; atomic/synchronization,
  automatic-resource, forced-spill, and native evidence remain open.
- 2026-07-17: Commit `ec84cd7af7` produced a fully validated CDNA4 Sampled
  access patch and removed the last embedded gfx12 LDS-wait word from engine
  emission.  The focused gate is 398/398 and all ConSan host tests are
  572/572.  This starts `SA0`; runtime/native and broader synchronization
  evidence remain open.
- 2026-07-17: Commit `6f654e596d` removed the last hard-coded gfx12
  `s_wait_loadcnt 0` words from ConSan engine emission.  Every publication,
  compare-swap, guest-atomic, and snapshot path now obtains its completion
  wait from the target-semantic builder; all 571 ConSan host tests pass.
- 2026-07-17: Commit `4fdaee5298` produced the first fully validated CDNA4
  engine patch: Record/Replay first-light access emission on a native DS
  instruction shape.  It also moved conditional branch fixups and publication
  waits onto target-semantic builders after the new test caught their hidden
  RDNA assumptions.  The focused gate is 397/397 and all ConSan host tests are
  571/571.  This starts `RR0`; native execution and full replay evidence are
  still required.
- 2026-07-17: Commit `0cb03769ea` established the first target-native analysis
  fixture.  Exact CDNA4 DS read/write instruction shapes survive the synthetic
  gfx950 ELF path and reach supported ConSan LDS candidates with correct
  operands.  This is analysis evidence only; no engine or workload cell is
  promoted yet.
- 2026-07-17: Commit `a0b96f50ce` completed neutral dispatch for ConSan's
  variable-length emission recipes: literal materialization, vector/address
  arithmetic, FLAT/DS/private memory, atomics, workgroup barriers, and their
  target-semantic waits.  CDNA4 multiply recipes explicitly receive a
  nonaliasing materialization VGPR, and private scratch uses VM_CNT completion
  rather than the broader general-FLAT drain.  The dispatch gate is 5/5, the
  focused builder/resource/spill/engine gate is 396/396, and the complete
  ConSan host suite is 569/569.  The plan now separates completed neutral
  routing (`B5A`) from active gfx950 engine-emission/shape proof (`B5B`).
- 2026-07-17: Commit `b21df67176` introduced the neutral instrumentation
  facade and routed ConSan's fixed-width scalar/vector control operations
  through it.  The facade has exact CDNA4/RDNA4 dispatch tests and typed
  rejection for unsupported architectures.  The focused gate is 394/394 and
  the complete ConSan host suite is 569/569; variable-length emission remains
  open in `B5`.
- 2026-07-17: Commit `f34be1ae72` completed the CDNA4 synchronization/cache
  primitive slice.  The wait immediates reflect gfx950's actual counter model:
  general FLAT drains VM_CNT and LGKM_CNT, LDS/SMEM drain LGKM_CNT, and scratch
  drains VM_CNT.  `s_barrier` is paired with a conservative memory pre-drain;
  gfx950 uses `s_nop 0`, never gfx12 `s_delay_alu`, for the one-cycle SALU
  separation.  The focused builder/resource/spill gate is 391/391.
- 2026-07-17: Commit `73241ff07f` completed the CDNA4 DS/atomic builder slice.
  Seven assembler-authoritative encodings and decoder mnemonics pass together
  with negative tuple/scope/return tests; the focused builder/spill gate is
  89/89.
- 2026-07-17: Commit `4c51920eb2` completed the non-atomic CDNA4 memory-builder
  slice.  Dispatch SMEM loads and FLAT publication accesses match LLVM and
  decode correctly with target-specific pair and offset rejection.  The
  focused builder/spill gate passes 87/87.
- 2026-07-17: Commit `113cd36228` completed the CDNA4 vector-builder slice and
  documented a new resource-planning requirement: in-place multiply by a
  literal needs a distinct materialization VGPR on gfx950.  The implementation
  uses only assembler-accepted sequences and the focused scalar/vector/spill
  gate passes 85/85.  Engine integration remains pending.
- 2026-07-17: Commit `38cf90d7c2` completed the separate CDNA4 scalar-control
  builder slice.  Exact gfx950 words and decoder mnemonics cover EXEC/VCC/SCC
  save/restore, scalar compares, and all branch predicates needed by probes;
  the RDNA4 builder remains CDNA-free.  This is builder evidence, not yet a
  workload promotion.
- 2026-07-17: Commit `97f978941d` made ConSan private-layout planning
  architecture-typed instead of inheriting RDNA4's 8 MiB range.  CDNA4
  persistent and spill extents are now bounded at 4 KiB and their final
  descriptor/AQL requirement is rounded to 16 bytes without moving dword
  slots.  The expanded focused gate passes 379/379; integrated shared-owner
  CDNA4 evidence is still pending, so this prerequisite remains orange.
- 2026-07-17: Commit `0ad99bd3ac` completed the standalone live CDNA4 scratch
  proof.  Both full-wave64 and partial-`EXEC` round trips pass serially on the
  MI355X using only the workspace TheRock runtime.  Disassembly confirms the
  intended scratch instructions and waits survived compilation, the partial
  kernel narrows and restores `EXEC`, and both kernel descriptors provision 32
  private bytes.  This promotes implementation node `S5`, not an e2e workload
  cell.
- 2026-07-17: Commit `319950aacd` established the separate CDNA4 fixed-scratch
  backend and transactional static VGPR spill plan.  Exact encodings, decoder
  round trips, limits, alignment, waits, rollback, and RDNA4 preservation pass
  the 377-test focused host gate.  This is prerequisite evidence only; all e2e
  cells remain unknown until native workload qualification.
- 2026-07-17: Created the CDNA4 ledger.  Verified native device discovery and
  an actual HIP dispatch using only the workspace TheRock runtime.  Verified
  the new `iree-test-suites` checkout and all directly consumed Sharktank
  assets.  All workload/profile cells intentionally begin unknown.
