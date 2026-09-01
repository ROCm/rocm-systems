# AIRUNTIME-28: non-temporal stores in the blit copy kernel

Does a non-temporal store hint in `__amd_rocclr_copyBuffer` make device-to-device copies
faster on MI450?

**Not by itself. It is worth having only for the concurrent case, and only if someone can
name a workload that runs a cache-sensitive kernel alongside 100-300 MiB copies.**
Recommendation: take the change default-off, gated to gfx12.

All numbers below are from one result set, `results/20260828_062133`, produced by
`remote/run_all.sh` on gfx1250. Superseded numbers from earlier revisions of this report are
in [CHANGELOG.md](CHANGELOG.md), not here. Method and controls are in [METHOD.md](METHOD.md);
commands in [REPRODUCE.md](REPRODUCE.md).

## Recommendation

| | |
|---|---|
| **What it buys** | Nothing measurable on an isolated copy below 96 MiB or above 256 MiB. **Roughly 3-5%** on copies in the 96-192 MiB band. **2.4% to 4.7%** off a co-running cache-sensitive kernel's runtime when its working set is 8-128 MiB, peaking around 32 MiB. |
| **What it costs** | One compiler builtin. Byte-exact at every size and alignment tested, no hip-tests regressions, no new failure mode, off unless asked for. |
| **Fix before merge** | The flag is not architecture-gated. On MI200 and Navi the same builtin emits *coherence* bits, not a cache hint. Everything here was measured on gfx1250 only. |
| **Not my call** | Nobody has shown that 100-300 MiB device-to-device copies, or concurrent cache-sensitive kernels, are hot for a workload we care about. That decides whether it is ever worth enabling. |

## Background

`__amd_rocclr_copyBuffer` is the kernel behind every device-to-device `hipMemcpy`: read
16 bytes, write 16 bytes, grid-stride. A large copy pushes its whole footprint through GL2,
evicting whatever else was cached, though the copy never re-reads a byte. A non-temporal store
hint tells the cache not to retain those lines, so in principle the copy stops evicting its
neighbours.

The ticket assumed the hint required dropping the access width from 128-bit to 64-bit. It does
not, in either language.

`__builtin_nontemporal_store` does reject HIP's `ulong2`, but because of what that type *is*,
not how wide it is: HIP's `ulong2` is a struct, `HIP_vector_type<unsigned long, 2>`, and the
builtin takes "a pointer to integer, float, pointer, or a vector of such types". A native
128-bit vector satisfies that. So the fix is to change the type, not to narrow the access — the
benchmark variants in this investigation are HIP, declare their 128-bit type as
`ext_vector_type(2)`, and emit `global_store_b128 ... th:TH_STORE_NT`, confirmed in the ISA.

The production kernel does not need even that substitution, because `blitcl.cpp` is OpenCL,
where `ulong2` is already a native vector the builtin accepts. Either way, no width is given
up — which matters, because width turns out to dominate everything else measured here.

## Results

System: `heliosr-1b114-a07-4`, gfx1250 (MI450 A0 engineering sample, `REV_ID 0x00`), 256 CU,
432 GiB HBM, SPX / NPS1. Negative is faster throughout. `(ns)` means the effect does not
exceed the resolution limit — the same variant measured twice in the same run — and so cannot
be distinguished from the rig's own noise, whatever its confidence interval says.

### Isolated streaming copy, 1 GiB

Each row compares one variant against the control that differs from it in exactly one
respect, which is not always the production baseline.

| question | effect [95% CI] |
|---|---|
| NT store hint, 128-bit (**the shipped change**) | -0.47% [-0.83, -0.11] (ns) |
| also hinting the load, 128-bit | +0.58% [+0.16, +0.71] (ns) |
| hand-written store carrying the *default* hint (codegen only) | **-1.26% [-1.71, -0.95]** |
| `TH_STORE_NT_RT` hint, net of that codegen effect | -0.29% [-0.59, +0.05] (ns) |
| NT store hint at 64-bit width | +0.23% [-0.04, +0.65] (ns) |
| NT store hint at 32-bit width | -0.18% [-0.34, +1.30] (ns) |
| 64-bit instead of 128-bit (width only) | **+77.26% [+76.75, +77.69]** |
| 32-bit instead of 128-bit (width only) | **+220.10% [+218.77, +220.62]** |

Resolution limit: ±0.74 pp. Baseline 0.5253 ms, 4088 GB/s read+write.

Three things to take from this. No temporal hint is separable from noise on an isolated 1 GiB
copy. Access width dominates by two orders of magnitude. And the largest sub-1% effect in the
table belongs to *hand-writing the store* rather than to any temporal hint — which is why the
`TH_STORE_NT_RT` variant, which looked like a 1.1% win against the production baseline, is
worth nothing once compared against its own codegen control.

### Where in the size range the hint does anything

| copy size | nt-store-128 vs plain-128 | resolution limit |
|---|---|---|
| 64 KiB - 4 MiB | between -1.3% and +0.8%, all (ns) | 6.8 - 15.0 pp |
| 16 - 64 MiB | between -0.4% and +0.0%, all (ns) | 0.4 - 1.3 pp |
| **96 MiB** | **-4.43% [-6.17, -3.84]** | 3.7 pp |
| **128 MiB** | **-3.07% [-4.18, -2.25]** | 2.4 pp |
| **192 MiB** | **-3.20% [-4.32, -1.51]** | 3.1 pp |
| 256 - 512 MiB | -1.5%, -1.4%, both (ns) | 2.5, 2.4 pp |
| 1 GiB | -0.38% (ns) | 1.2 pp |

The band that pays is where the copy's footprint straddles GL2 capacity: at 96 MiB the copy
touches 192 MiB against a ~96-128 MiB cache. Below 64 MiB there is no eviction pressure to
relieve; above 256 MiB the copy overwhelms the cache regardless and the hint can only shave
the margin. Sizes below 4 MiB cannot show anything at all, because the dispatch costs more
than the copy (`small_copy`: a 16 KiB copy takes 0.9 empty dispatches, and nothing under
8 MiB exceeds 1.5).

The band's *upper* edge is the least stable part of this table. Across four full runs the
96 MiB and 128 MiB points were significant every time, while 192 MiB and 256 MiB moved in and
out of significance between -1.5% and -3.2%. Read the claim as "roughly 3-5% from 96 to
192 MiB, fading out by 256 MiB", not as the exact figures in any one row.

### A copy running alongside a cache-sensitive kernel

The metric is the **victim kernel's** time, not the copy's: a 128 MiB copy is repeated to
cover the victim's whole ~5 ms run, and the victim's working set is swept.

| victim working set | victim time, nt-store-128 vs plain-128 | codegen control |
|---|---|---|
| 2 MiB | +0.14% (ns) | (ns) |
| 8 MiB | **-2.56%** | (ns) |
| 16 MiB | **-2.70%** | (ns) |
| **32 MiB** | **-4.73%** | (ns) |
| 48 MiB | **-2.88%** | (ns) |
| 64 MiB | **-2.73%** | (ns) |
| 96 MiB | **-2.40%** | (ns) |
| 128 MiB | **-2.67%** | (ns) |

The control arm — the same copy with a hand-written store carrying the default hint — sits
within noise at every size, so what is being measured is the temporal hint and not codegen.
This is the most repeatable result here: four full runs put the 32 MiB peak between -4.7% and
-5.0% and the shelf between -2.4% and -2.9%, and the earlier eight-point table from the two
programs this experiment replaced agrees within 0.2 pp at seven of eight sizes.

The shape follows the measured hierarchy. A 2 MiB working set survives in a ~96 MiB GL2
whatever the copy does, so there is nothing to protect; that is the one size where the effect
vanishes, and it is the size the first version of this measurement used. From 8 to 32 MiB the
victim genuinely depends on GL2 and is maximally exposed. Above 48 MiB the victim's own
footprint is a large fraction of GL2, so it is partly memory-bound regardless and protection
matters relatively less — hence the flat ~2.5% shelf.

**This is the only scenario in which the change pays**, and the reason is the next section.

### Nothing survives a kernel dispatch

Cache state does not persist across a dispatch boundary on this part: dependent-load latency
is identical whether the previous dispatch flushed the cache or walked the exact same
addresses (largest ratio 1.012x at any footprint within GL2), while four laps *inside one
dispatch* run up to 2.2x faster per hop. Caching works; it does not survive leaving a kernel.
None of six allocation kinds changes this, and sweeping the cold-cache flush across a 16x
range moves cold latency by 0.06% — the flush has nothing to do, because the dispatch boundary
already did it.

Two consequences for this ticket, neither depending on the mechanism. "The copy evicts data
the next kernel needs" cannot happen, because nothing survives to be evicted — so the hint
cannot help a sequential consumer. Equally it cannot *hurt* one, which is most of why the
adversarial search below comes back empty. Only genuinely concurrent work can benefit.

The mechanism is a separate investigation with a larger prize attached, and it is written up
in [FINDING-gl2-residency.md](FINDING-gl2-residency.md). It also means the driver-reported
cache size is wrong by 24x, which invalidated the footprint sizing in the first round of this
work; see [CHANGELOG.md](CHANGELOG.md).

### Attempts to make it lose

Nine mechanisms by which a non-temporal store could plausibly cost something — a reader of the
destination, reuse of the destination as a source, fan-out from one hot source, repeated
overwrite, staging-buffer reuse, four concurrent copies, the narrow fallback path — each run
at three footprints: half of GL2, all of GL2, and 2.7x GL2. Twenty-seven cases.

**No case found where the hint is significantly worse.** The most adverse effect anywhere was
+1.19%, at a case whose own resolution limit was 6.1 pp. Five cases at the largest footprint
show the hint significantly *better*, by 1.8% to 4.7%.

The limit of that claim: the least sensitive case in the suite could not have detected a
regression smaller than 6.1 pp. The cases that matter most for this question — a reader of the
destination at a GL2-resident footprint — resolve to about 0.5 pp and show effects under 0.3%
either way.

## Risks and limits

- **One machine, one GPU, an A0 engineering sample.** Nothing here speaks to production
  silicon.
- **The flag is not architecture-gated.** This is the one thing that must be fixed before
  merge: on MI200 and Navi the same builtin emits coherence bits rather than a cache hint.
- **The victim is synthetic** — a loop sweeping a cache-resident buffer. It is a plausible
  stand-in for a cache-sensitive kernel, but "up to 4.8%" describes this access pattern, not
  any application.
- **Absolute copy times in the 16-48 MiB band are not a smooth function of size.** They sit on
  flat plateaus (~103 us for 16-20 MiB, ~56 us for 24-48 MiB) independent of bytes moved, and a
  run occasionally lands on the faster plateau at a size that usually takes the slower one.
  Within-run repeatability is under 1% and both arms of a comparison always sit on the same
  plateau, so the paired results are unaffected — but the ms and GB/s columns in that band
  describe a run rather than the hardware. Cause not established.
- **PAL is untested.** The PAL mirror is compile-consistent only; PAL is Windows-only.
- **The isolated-copy gain is the narrowest claim here.** It is significant at 96 and 128 MiB in
  every run, but the band's upper edge moves between runs, so its width is less certain than its
  existence. It is the claim most worth re-checking on production silicon.

## The change

Commit `81e65d6bbb` on `users/victzhan/AIRUNTIME-28-nt-blit`. 59 insertions, 4 deletions,
6 files. The branch tip may sit ahead of that commit — pressing "Update branch" on the PR adds
a merge from `develop` — but the change itself is that one commit, and the PR diff stays these
six files.

`airuntime28-nt-blit.patch` is generated by `remote/clr_patch.sh`, which diffs against the
merge-base with `develop` (not `HEAD^`, which stops being the base as soon as such a merge
lands) and then verifies that applying the result to that base reproduces the branch's change
exactly. Regenerate it after any branch update rather than editing it.

One provenance caveat. The measurements in this report were taken against a CLR built at
`563095dbca`; the commit was later rebased 335 commits forward onto `develop`. The rebase was
textually clean and the diff content is byte-identical, but two of those upstream commits do
touch files this change touches (`rocblit.cpp`, `flags.hpp`), so the rebase was re-verified
rather than assumed: the new flag still sits in the blit-flag block, `shaderCopyBuffer` still
selects on it with `kMaxAlignment` unchanged at 16, and `remote/validate_kernel.sh` confirms
the shipped `__amd_rocclr_copyBufferNT` still emits `global_store_b128 ... th:TH_STORE_NT` at
full width with an unhinted load. The performance numbers themselves have not been re-run on
the new base.

- `blitcl.cpp` gains `__amd_rocclr_copyBufferNT`: identical to `__amd_rocclr_copyBuffer`
  except the store is `__builtin_nontemporal_store`. It keeps `ulong2` and tests
  `aligned_size == sizeof(ulong2)`, so it reaches its own wide path.
- `flags.hpp` gains `DEBUG_CLR_BLIT_NONTEMPORAL`, default **false**.
- `rocblit.hpp` / `palblit.hpp` gain `BlitCopyBufferNT` in the kernel enum and `BlitName`.
- `rocblit.cpp` `shaderCopyBuffer` and `palblit.cpp` `copyBuffer` select it when set.

Worth passing to whoever owns [PR 2616](https://github.com/ROCm/clr/pull/2616): **it never
executes its own 64-bit path.** Its kernel branches on `aligned_size == sizeof(ulong)` (8)
while `shaderCopyBuffer` still passes `kMaxAlignment` (16), so every aligned copy falls
through to the `uint` branch and runs a *32-bit* non-temporal copy — the `pr2616-actual-32`
row above, 3.2x slower than the baseline on a 1 GiB copy. Any measurement of that PR as-is
would have shown non-temporal stores to be catastrophic, for reasons unrelated to
non-temporal stores.

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
