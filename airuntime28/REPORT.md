# AIRUNTIME-28: non-temporal stores in the blit copy kernel

Does a non-temporal store hint in `__amd_rocclr_copyBuffer` make device-to-device copies
faster on MI450?

**Not by itself. It is worth having only for the concurrent case, and only if someone can
name a workload that runs a cache-sensitive kernel alongside 100-300 MiB copies.**

Take it default-off, gated to gfx12.

## Overview

| | |
|---|---|
| **What it buys** | Nothing measurable on an isolated copy below 96 MiB or above 256 MiB. **Roughly 3-5%** on copies in the 96-192 MiB band. **2.4% to 4.7%** off a co-running cache-sensitive kernel's runtime when its working set is 8-128 MiB, peaking around 32 MiB. |
| **What it costs** | One compiler builtin. Byte-exact at every size and alignment tested, no hip-tests regressions, no new failure mode, off unless asked for. |

## Background

`__amd_rocclr_copyBuffer` is the kernel behind every device-to-device `hipMemcpy`: read
16 bytes, write 16 bytes, grid-stride. A large copy pushes its whole footprint through GL2,
evicting whatever else was cached, though the copy never re-reads a byte. A non-temporal store
hint tells the cache not to retain those lines, so in principle the copy stops evicting its
neighbours.

The ticket assumed the hint required dropping the access width from 128-bit to 64-bit. It does
not. `__builtin_nontemporal_store` rejects HIP's `ulong2` because that type is a struct, not
because 128 bits is too wide — the builtin takes "a pointer to integer, float, pointer, or a
vector of such types", and any native 128-bit vector satisfies it. Change the type, not the
width. No width is given up, which matters, because width turns out to dominate everything else
measured here.

## Results

One result set, `results/20260828_062133`, on `heliosr-1b114-a07-4` at a 1100 MHz clock
ceiling. The machine has since been reconfigured to 2400 MHz, which moves every absolute figure
below; re-measure before quoting these against current hardware.

Negative is faster. `(noise)` means smaller than that run's resolution limit — the spread the
rig shows when it measures the same kernel twice — so: no effect found.

### Isolated streaming copy, 1 GiB

Each row changes one thing from the variant it is measured against. Where that is not
`plain-128`, the variant differs from production in two ways and comparing it to production
would credit the wrong one.

| change | measured against | effect [95% CI] |
|---|---|---|
| NT store hint, 128-bit (**the shipped change**) | `plain-128` | -0.47% [-0.83, -0.11] (noise) |
| also hinting the load | `nt-store-128` | +0.58% [+0.16, +0.71] (noise) |
| NT store hint at 64-bit width | `plain-64` | +0.23% [-0.04, +0.65] (noise) |
| NT store hint at 32-bit width | `plain-32` | -0.18% [-0.34, +1.30] (noise) |
| 64-bit instead of 128-bit | `plain-128` | **+77.26% [+76.75, +77.69]** |
| 32-bit instead of 128-bit | `plain-128` | **+220.10% [+218.77, +220.62]** |

Resolution limit 0.74 percentage points (pp); an effect must beat that in magnitude, it is not
a plus-or-minus band. Baseline 0.5253 ms, 4088 GB/s.

No hint is separable from noise; width dominates by two orders of magnitude. That is why
PR 2616 reads as evidence against non-temporal stores when it is evidence about width.

`TH_STORE_NT_RT` was tried and rejected: it needs hand-written gfx12 asm, and net of that
hand-writing it measures -0.29%, noise.

### Where in the size range the hint does anything

| copy size | nt-store-128 vs plain-128 | resolution limit |
|---|---|---|
| 64 KiB - 4 MiB | -1.3% to +0.8%, all (noise) | 6.8 - 15.0 pp |
| 16 - 64 MiB | -0.4% to +0.0%, all (noise) | 0.4 - 1.3 pp |
| **96 MiB** | **-4.43% [-6.17, -3.84]** | 3.7 pp |
| **128 MiB** | **-3.07% [-4.18, -2.25]** | 2.4 pp |
| **192 MiB** | **-3.20% [-4.32, -1.51]** | 3.1 pp |
| 256 - 512 MiB | -1.5%, -1.4%, both (noise) | 2.5, 2.4 pp |
| 1 GiB | -0.38% (noise) | 1.2 pp |

The band that pays is where the copy straddles GL2: 96 MiB copied touches 192 MiB against a
~96-128 MiB cache. Below 64 MiB there is no eviction pressure to relieve, and below 4 MiB the
dispatch costs more than the copy. Read it as roughly 3-5% from 96 to 192 MiB: across four runs
96 and 128 MiB were always significant, 192 and 256 MiB moved in and out.

### A copy alongside a cache-sensitive kernel

Metric is the **victim kernel's** time, not the copy's. A 128 MiB copy repeats to cover the
victim's ~5 ms run; the victim's working set is swept.

| victim working set | victim time vs plain-128 | codegen control |
|---|---|---|
| 2 MiB | +0.14% (noise) | (noise) |
| 8 MiB | **-2.56%** | (noise) |
| 16 MiB | **-2.70%** | (noise) |
| **32 MiB** | **-4.73%** | (noise) |
| 48 MiB | **-2.88%** | (noise) |
| 64 MiB | **-2.73%** | (noise) |
| 96 MiB | **-2.40%** | (noise) |
| 128 MiB | **-2.67%** | (noise) |

**The only scenario where the change pays.** The control arm stays in noise throughout, so the
effect is the hint and not codegen. It is also the most repeatable result here: four runs put
the 32 MiB peak at -4.7% to -5.0% and the shelf at -2.4% to -2.9%.

2 MiB is the one size with no effect — it survives in a ~96 MiB GL2 whatever the copy does, and
it is the size the first version of this measurement used.

### Nothing survives a kernel dispatch

Dependent-load latency is the same whether the previous dispatch flushed the cache or walked
the identical addresses (worst ratio 1.012x within GL2), while four laps inside one dispatch
run 2.2x faster per hop. None of six allocation kinds changes it, and sweeping the flush over a
16x range moves cold latency by 0.06%.

So a copy cannot evict what a later kernel needs, because nothing survives to be evicted. The
hint can neither help nor hurt a sequential consumer — which is why the adversarial search
below comes back empty, and why only concurrent work benefits. Mechanism:
[FINDING-gl2-residency.md](FINDING-gl2-residency.md).

### Attempts to make it lose

Nine mechanisms that could plausibly cost something — reader of the destination, destination
reused as a source, fan-out from one hot source, repeated overwrite, staging-buffer reuse, four
concurrent copies, the narrow fallback path — each at three footprints: half of GL2, all of
GL2, and 2.7x GL2. Twenty-seven cases.

**None where the hint is significantly worse.** Most adverse: +1.19%, at a case whose own
resolution limit was 6.1 pp. Five cases at the largest footprint show it significantly better,
by 1.8% to 4.7%.

The limit of that claim: the least sensitive case could not have detected a regression under
6.1 pp. The cases that matter most — a reader of the destination at a GL2-resident footprint —
resolve to about 0.5 pp and show under 0.3% either way.

## Risks and limits

- **Absolute copy times in the 16-48 MiB band are not a smooth function of size.** They sit on
  flat plateaus (~103 us for 16-20 MiB, ~56 us for 24-48 MiB) independent of bytes moved, and a
  run occasionally lands on the faster plateau at a size that usually takes the slower one.
  Within-run repeatability is under 1% and both arms of a comparison always sit on the same
  plateau, so the paired results are unaffected — but the ms and GB/s columns in that band
  describe a run rather than the hardware. Cause not established.
- **The isolated-copy gain is the narrowest claim here.** It is significant at 96 and 128 MiB in
  every run, but the band's upper edge moves between runs, so its width is less certain than its
  existence. It is the claim most worth re-checking on production silicon.

## The change

Commits `81e65d6bbb` and `ac583d3369` on `users/victzhan/AIRUNTIME-28-nt-blit` — 6 files, 60
insertions, 4 deletions. `blitcl.cpp` gains `__amd_rocclr_copyBufferNT`, identical to
`__amd_rocclr_copyBuffer` except the store is `__builtin_nontemporal_store` and it still tests
`aligned_size == sizeof(ulong2)` so it reaches its own wide path; `flags.hpp` gains
`DEBUG_CLR_BLIT_NONTEMPORAL`, default **false**; the ROC and PAL blit managers select it when
set. The PR has the rest. `airuntime28-nt-blit.patch` is generated from the branch by
`remote/clr_patch.sh` — regenerate, never hand-edit.

**Provenance caveat.** Measurements were taken against a CLR built at `563095dbca`, before the
branch was rebased 335 commits onto `develop`. The diff content is byte-identical and
`remote/validate_kernel.sh` confirms the shipped kernel still emits
`global_store_b128 ... th:TH_STORE_NT` at full width, but the numbers have not been re-run on
the new base.

**For whoever owns [PR 2616](https://github.com/ROCm/clr/pull/2616): it never executes its own
64-bit path.** Its kernel branches on `aligned_size == sizeof(ulong)` (8) while
`shaderCopyBuffer` passes `kMaxAlignment` (16), so every aligned copy falls through to the
`uint` branch and runs a *32-bit* non-temporal copy — 3.2x slower than baseline at 1 GiB. Any
measurement of that PR as-is shows non-temporal stores as catastrophic, for unrelated reasons.

## Validation

**ISA.** All nine variants emit the instruction width and temporal hint they claim, checked
against expectations declared in `src/common/variants.h` so the check cannot drift from the
code (`remote/isa_check.sh`). Support kernels carry no temporal hints, so they are neutral
probes. Separately, the real `BlitLinearSourceCode` blob is extracted from `blitcl.cpp` and
compiled for gfx1250, so the shipped kernel is checked too, not only the transcription.

**Byte-exactness.** All nine variants verified over a size that is not a multiple of 16, so
the scalar remainder tail is exercised, before any timing is trusted — the check runs inside
`isolated_copy` and fails the run if it fails.

**Correctness through the real path.** `hipMemcpyAsync` D2D over 10 sizes x 3 base offsets,
including sizes exercising the remainder tail (1, 7, 255, 4 KiB+3, 3 MiB+17), offsets forcing
the unaligned `uint` path, and a guard byte past the end to catch overruns. 0 failures with
the flag off and on.

**Kernel selection.** Under `AMD_LOG_LEVEL=4` the same workload dispatches
`__amd_rocclr_copyBuffer` with the flag off and `__amd_rocclr_copyBufferNT` with it on.

**hip-tests memory suites**, patched runtime via `LD_LIBRARY_PATH`: MemoryTest1, MemoryTest2
and DeviceMemoryTest abort at the same pre-existing points with the flag off and on, and
MemoryTest2's failing sites are identical. The single MemoryTest1 delta is
`hipHostRegister.cc:754`, whose assertion depends on transient free memory; host-registered
memory is host-direct-access, which `useShaderCopyBufferPath` excludes from the shader copy
path, so this change cannot reach it. Six runs each under flag-off, flag-on and stock ROCm
give 0 failures every time.
