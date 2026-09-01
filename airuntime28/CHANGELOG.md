# Supersession log

What earlier revisions of this investigation claimed, what replaced it, and why. This file
exists so that [REPORT.md](REPORT.md) can state current conclusions without retracting itself
in place — an earlier revision presented a concurrency table, asserted "four of four agree at
about -1%", and then explained two paragraphs later that the -1% was an artefact. A reader who
stopped at the table left with the wrong number.

Anything here is **withdrawn**. It is kept because knowing which way a number moved, and why,
is what tells you how much to trust the ones that are left.

---

## 2026-08-28 — consolidation and re-measurement

Thirteen near-duplicate benchmark programs became a shared measurement core plus eight thin
experiments, and every claim was re-measured through it. Result set
`results/20260828_062133`.

### The headline table came from the harness the report itself called flawed

**Was:** the "isolated streaming copy, 1 GiB" table was produced by `nt_blit_bench isolated`,
which used `for (variant) { for (iteration) }` block ordering with no noise floor and no
duplicate baseline slot — while the same document's method section listed block ordering as a
confounder, and its traps section documented a mid-investigation regression that turned out to
be exactly that artefact.

**Now:** produced by `isolated_copy` on the shared harness: one sample per arm per iteration,
order reshuffled every iteration, every arm duplicated so the rig's resolution is measured in
the same run.

**What moved:** the width penalties survived, expressed the other way round. `plain-64` was
reported as "-44% bandwidth", now +77.4% time; `plain-32` as "-68.8%", now +221.0% time. Those
are the same measurement — 1/(1-0.44) = 1.79 — and they agree to within a point. The shipped
change's figure moved from -0.1% to -0.28% [-0.71, +0.50] and is still not separable from
noise, now against a stated 0.93 pp resolution limit rather than against nothing.

### `TH_STORE_NT_RT` was reported as a win. It is codegen.

**Was:** `ntrt-store-128` at -1.4% on a 1 GiB copy, with a note that the codegen control sat at
-1.3% and so "that gain is the hand-written store, not the hint" — correct, but buried under a
table that credited the variant with the gain.

**Now:** reported only as the controlled comparison, `ntrt-store-128` against
`asm-plain-128`: **+0.09% [-0.41, +0.20] (ns)**. The hint is worth nothing net of hand-writing
the store. The codegen effect itself, -1.26% [-1.44, -0.76], is the largest sub-1% effect in
the isolated table and has nothing to do with temporal hints.

### The concurrency benefit was 1%. It is 2.4-4.8%.

**Was:** "-1%, four of four runs agree", from a victim with a **2 MiB** working set.

**Why it was wrong:** 2 MiB was chosen believing `l2CacheSize` = 4 MiB. GL2 is ~96-128 MiB, so a
continuously re-touched 2 MiB working set is nearly untouchable — a streaming copy has to
displace ~94 MiB before it costs the victim anything. That was the wrong regime to measure in,
and 2 MiB is the single working-set size at which the effect vanishes.

**Now:** the working set is swept. +0.30% (ns) at 2 MiB, then significant at every size from 8
to 128 MiB, peaking at **-4.77%** at 32 MiB and settling on a **-2.4% to -2.6%** shelf from
48 MiB out past GL2 capacity.

### Two programs measured the concurrency result and nobody could tell which produced the table

**Was:** `victim_sweep.hip` and `victim_ws_sweep.hip`, 110 differing lines, both publishing an
eight-point working-set sweep. The report contained **both tables**, presented as two separate
results agreeing with each other, when they were the same experiment run twice.

**Now:** one `concurrency` experiment. It reproduced the published eight-point table before
either was deleted: seven of eight points within 0.2 pp, the eighth (16 MiB) differing by
0.8 pp at a footprint that sits exactly on the near-cache knee where the calibrated pass count
is most sensitive. The duplicated table is gone.

### "No adversarial case found" had not been tested

**Was:** ten adversarial cases, all at a 128 MiB footprint, concluding no regression exists.

**Why it was weak:** a 128 MiB copy touches 256 MiB, nearly three times the measured GL2. Nothing
it wrote could have stayed resident under any store policy, so the cases meant to punish the
hint — a reader of the destination, reuse of the destination as a source — could not have
detected damage even if there were some. The whole suite was reasoned inside a 4 MiB mental
model.

**Now:** nine cases at three footprints — half of GL2, all of GL2, and 2.7x GL2 — for
twenty-seven in total, plus the staging-buffer case folded in from what was a separate program.
Still no case where the hint is significantly worse; most adverse effect anywhere +1.60%. The
suite now also prints its own coarsest resolution limit (8.5 pp) so the negative claim carries
its own bound, which the earlier version did not.

### The cold-cache flush was calibrated against the wrong cache size

**Was:** `flushBytes = 256 * kMiB` in six programs, chosen believing L2 was 4 MiB — a 64x
margin. Against the measured ~96 MiB it is 2.7x, which is thin and had never been checked.

**Now:** swept from 128 MiB to 2 GiB. Results are insensitive to it: cold dependent-load
latency is 738-742 ns/hop at every size and equal to the warm reference, and copy time varies by
2%. That is not because 128 MiB is comfortably adequate — it is because the dispatch boundary
already invalidates GL2, so the flush has nothing left to do. 1 GiB is used anyway (10.7x
measured GL2, ~0.2 ms per sample) so that no result depends on that finding continuing to hold.

### The bootstrap used two different estimators

**Was:** 2000 resamples in `order_control.hip` and 3000 in every other program, so the
confidence intervals in the concurrency table were computed differently from the ones in the
size curve. Pure copy-paste damage from the statistics core existing in seven copies.

**Now:** one definition in `src/common/stats.h`, 3000 resamples, fixed seed, used by every
experiment.

### The small-copy range was described using the wrong cache size

**Was:** 16 KiB - 8 MiB, characterised as "small enough to have stayed cached".

**Now:** swept to 64 MiB, and the framing is dispatch overhead rather than cache residency,
because that is what the data says: a 16 KiB copy costs 0.9 empty dispatches and nothing below
8 MiB exceeds 1.5. Cache residency was never the reason nothing happens down there, and against
a ~96 MiB GL2 the original range was nowhere near the interesting boundary anyway.

### A 1 MiB regression was reported and then withdrawn

**Was:** a mid-investigation revision reported a regression at 1 MiB copies.

**Why it was wrong:** it was the block-ordering artefact described above — the 1 MiB
measurements ran in a block whose position in the run differed from the baseline's.

**Now:** 1 MiB shows -0.43% [-2.63, +0.46] against a 4.6 pp resolution limit. No effect, and
the harness that produced the false one no longer exists.

### The wrong explanation for the missing cache residency

**Was:** the GL2 invalidation was attributed to the `isGfx12` branch added by
[PR #966](https://github.com/ROCm/rocm-systems/pull/966).

**Why it was wrong:** that predicate is `major==12 && minor==0 && stepping in {0,1}`. gfx1250 is
minor 5, so it does not match, and reading the decoded dispatch packets confirms most
dispatches carry agent rather than system scope.

**Now:** [FINDING-gl2-residency.md](FINDING-gl2-residency.md) rules out fence scope by
measurement (forcing system scope everywhere changes nothing) and memory type by measurement
(none of six allocation kinds retains), and states plainly that the mechanism is open.

### Overstated: "AMD's own coherence docs predict the opposite"

**Was:** phrased as though a full flush contradicted the specification, implying a defect.

**Now:** it does not. The documentation permits either a selective `INVL2_NC` or a full
`WBINVL2`, and a full writeback-invalidate wipes RW lines too, so the measurement is consistent
with spec-permitted behaviour. What remains unexplained is narrower and stated as such: the
invalidation is not gated by scope, and it is not selective by memory type.

### The driver's cache size was quoted as a small number

**Was:** treated as "the driver reports 4 MiB, the measurement says 96 MiB, so the driver is
wrong".

**Now:** the KFD record behind it has all its geometry fields zeroed — `cache_line_size`,
association, latency. It is an unpopulated stub, so the 4 MiB should be read as *absent* rather
than as an incorrect measurement, which is a different kind of bug and a different fix.

### Nine dead scripts and a duplicated program

**Removed:** three abandoned clock-pinning attempts (`clock_pin.sh`, `clock_set.sh`,
`clock_try.sh` — this part cannot be clock-pinned, so the clock is measured instead),
`waitcnt_check.sh` (probed a hypothesis that was disproven), `regression_repeat.sh`,
`hiptests_diff.sh`, `verify_doc_commands.sh`, `run.sh` and an older `build.sh` (duplicate entry
points), plus `flaky_check.sh`, `rebuild_all.sh`, `concurrency_repeat.sh`,
`victim_sweep_repeat.sh`, `size_curve_run.sh` and `staging_run.sh`, all superseded by the single
`build.sh` / `run_all.sh` pair. `fence_scope_check.sh` and `fence_root_cause.sh` — one question
in two halves, the second re-running the first — became `fence_check.sh`.
`explain_isa_pipeline.sh` explained a grep pipeline that `isa_check.sh` no longer uses, so it
had started to mislead.

**Also removed:** `regression_check.hip` (superseded by the size curve), `order_control.hip`
(its slot-ordering logic became the shared harness), `staging_test.hip` (folded into the
adversarial suite as a case) and `nt_blit_bench.hip` (its cache-latency scan became
`cache_capacity`, its correctness check became `verifyVariant` in `geometry.h`, and its
null-result scenarios became adversarial cases at GL2-aware footprints).

The old `run_all.sh` omitted `victim_sweep`, `mtype_probe`, `l2_capacity` and the fence probes —
four of the five most important measurements — so "I ran run_all.sh" and "I ran the experiments"
were different statements. The new one runs everything and fails loudly.

### The ISA expectations were duplicated in shell

**Was:** the expected opcode and hint for each of the nine variants written out in a Python
block inside `isa_dump.sh`, separate from the variants themselves.

**Now:** declared in `src/common/variants.h` next to the code they describe, printed by
`isolated_copy --print-isa-expectations`, and read from there by `isa_check.sh`. Adding a
variant needs no change to the script. The script also classifies kernels from their demangled
signatures rather than from mangled-name regexes, which had silently stopped matching when the
code moved into a namespace.

---

## Newly observed, not previously reported

**Absolute copy times in the 16-48 MiB band are not a smooth function of size.** They sit on
flat plateaus — ~103 us for 16-20 MiB, ~56 us for 24-48 MiB — independent of how much data is
moved, and a run occasionally lands on the faster plateau at a size that usually takes the
slower one. Reproduced across two independent programs and eight consecutive process
invocations; not explained by buffer addresses, first-touch cost, or the flush. Within-run
repeatability is under 1% and both arms of a comparison always sit on the same plateau, so no
paired result is affected, but absolute ms and GB/s figures in that band describe a run rather
than the hardware. Recorded here because an unexplained non-monotonicity in a published curve
should be stated rather than smoothed over.
