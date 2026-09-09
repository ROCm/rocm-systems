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

Negative is faster. `(noise)` = smaller than that run's resolution limit, the spread the rig
shows measuring the same kernel twice, so: no effect found.

### Isolated streaming copy, 1 GiB

Each row changes one thing from the variant it is measured against.

| change | measured against | effect [95% CI] |
|---|---|---|
| NT store hint, 128-bit (**the shipped change**) | `plain-128` | -0.47% [-0.83, -0.11] (noise) |
| also hinting the load | `nt-store-128` | +0.58% [+0.16, +0.71] (noise) |
| NT store hint at 64-bit width | `plain-64` | +0.23% [-0.04, +0.65] (noise) |
| NT store hint at 32-bit width | `plain-32` | -0.18% [-0.34, +1.30] (noise) |
| 64-bit instead of 128-bit | `plain-128` | **+77.26% [+76.75, +77.69]** |
| 32-bit instead of 128-bit | `plain-128` | **+220.10% [+218.77, +220.62]** |

Resolution limit 0.74 percentage points (pp). Baseline 0.5253 ms, 4088 GB/s.

No hint beats noise; width dominates by two orders of magnitude. That is why
[PR 2616](https://github.com/ROCm/clr/pull/2616) reads as evidence against non-temporal stores
when it is really evidence about width — it branches on `aligned_size == sizeof(ulong)` while
the host passes 16, so every aligned copy silently takes the 32-bit path. The other hint,
`TH_STORE_NT_RT`, needs hand-written asm and measures -0.29% net of it: rejected.

### By copy size

| copy size | nt-store-128 vs plain-128 | resolution limit |
|---|---|---|
| 64 KiB - 4 MiB | -1.3% to +0.8%, all (noise) | 6.8 - 15.0 pp |
| 16 - 64 MiB | -0.4% to +0.0%, all (noise) | 0.4 - 1.3 pp |
| **96 MiB** | **-4.43% [-6.17, -3.84]** | 3.7 pp |
| **128 MiB** | **-3.07% [-4.18, -2.25]** | 2.4 pp |
| **192 MiB** | **-3.20% [-4.32, -1.51]** | 3.1 pp |
| 256 - 512 MiB | -1.5%, -1.4%, both (noise) | 2.5, 2.4 pp |
| 1 GiB | -0.38% (noise) | 1.2 pp |

The band that pays is where the copy straddles GL2 — 96 MiB copied touches 192 MiB against a
~96-128 MiB cache. Call it 3-5% from 96 to 192 MiB: over four runs 96 and 128 MiB were always
significant, 192 and 256 MiB moved in and out.

### Alongside a cache-sensitive kernel

Metric is the **victim kernel's** time, not the copy's, with a 128 MiB copy running against it.

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

**The only scenario where the change pays**, and the most repeatable result here: four runs put
the peak at -4.7% to -5.0% and the shelf at -2.4% to -2.9%. The control stays in noise, so this
is the hint and not codegen. 2 MiB is the one size with nothing to protect — and the size the
first version of this measurement used.

### Nothing survives a kernel dispatch

Dependent-load latency is unchanged whether the previous dispatch flushed the cache or walked
the identical addresses, while four laps inside one dispatch run 2.2x faster per hop, and no
allocation kind changes that. So a copy cannot evict what a later kernel needs — nothing
survives to be evicted — which is why the hint can neither help nor hurt a sequential consumer,
and why only concurrent work benefits. Mechanism:
[FINDING-gl2-residency.md](FINDING-gl2-residency.md).

### Attempts to make it lose

Nine mechanisms that could plausibly cost something — a reader of the destination, reuse as a
source, fan-out, repeated overwrite, staging reuse, concurrent copies, the narrow path — each
at half, all, and 2.7x of GL2. Twenty-seven cases, **none where the hint is significantly
worse**; most adverse +1.19%. The least sensitive case could not have caught a regression under
6.1 pp, but the ones that matter most resolve to ~0.5 pp and show under 0.3% either way.

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
- **The numbers predate the current base.** They were taken before the branch was rebased
  onto current `develop` and have not been re-run on it. The diff is byte-identical and
  `remote/validate_kernel.sh` still confirms the shipped kernel emits
  `global_store_b128 ... th:TH_STORE_NT` at full width.

## The change
https://github.com/ROCm/rocm-systems/pull/11054

`blitcl.cpp` gains `__amd_rocclr_copyBufferNT`: `__amd_rocclr_copyBuffer` with
`__builtin_nontemporal_store` on the store, still `ulong2`, selected by a new
`DEBUG_CLR_BLIT_NONTEMPORAL` flag that defaults to **false**.

## Validation

Every variant compiles to the instruction and cache hint it claims — verified on the real
shipped kernel, not just the benchmark's copy — and copies byte-for-byte correctly, both
checked before any timing is trusted. Copies through the real runtime pass at every size and
alignment tried, and the hip-tests memory suites show no new failures with the flag on.