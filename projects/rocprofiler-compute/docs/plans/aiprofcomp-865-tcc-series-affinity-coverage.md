# AIPROFCOMP-865 — TCC series affinity + coverage (approved packing policy)

**Status:** Approved for design (not yet implemented in the allocator)
**JIRA:** AIPROFCOMP-865
**Date:** 2026-10-01
**Related:**
- [Single-pass packable plan](aiprofcomp-865-single-pass-packable-plan.md) (Phase 1 SPP)
- [Phase 2 WEIGHTED_AVG design](aiprofcomp-865-phase2-weighted-avg-design.md) (**not** this policy)
- [AQL / SQ slot ticket](aiprofcomp-865-aql4096-sdk-ticket.md) (SQ:8 context)
- Evidence: `validation-artifacts/spp-health-261001-gfx942-500med/reports/delta_ge100_investigation.md`

---

## 0. Naming — what this is / is not

Call this **TCC series affinity + coverage**.

| This policy | Not this policy |
|-------------|-----------------|
| Harden how **TCC channel series** are packed under SPP | Phase 2 **`SLOT_LIMIT` / `WEIGHTED_AVG`** |
| Fill-like only by analogy: pack by **series base** and keep ratio pairs together | Parent metrics that cannot fit one HW bucket |
| Applies to gfx942 default SPP (and arches with the same TCC:4 model) | Changing the 14-pass SPP acceptance gate unless evaluation says so |

SLOT_LIMIT-like “fill” is only an analogy: we expand a series to all collectable channels in a pass the way SLOT fill places missing PMCs into existing buckets — **without** introducing `WEIGHTED_AVG` or demoting packable metrics to Phase 2.

---

## 1. Approved policy (normative)

### 1.1 Packing unit = series base; expand all collectable channels

- The TCC HW limit is **4 event bases per pass** (`perfmon_config["TCC"] == 4` on gfx942). That matches AQL / block register accounting, **not** a soft product preference. **SQ is 8** (`perfmon_config["SQ"] == 8`).
- Channel instances (`TCC_*[i]`) are **dimensions of one event base**, not extra register slots. `LimitedSet.add` already admits further `[i]` of a base already present without charging TCC capacity (`soc_base.LimitedSet`).
- When a TCC channel series is selected for a profile, place **all collectable channel instances** of that base in the pass that holds the base (same file / same channel map). Do not leave a partial channel subset of a selected series.

### 1.2 Keep LEVEL + matching REQ / ATOMIC in the same pass

Affinity pairs (same pass, same channel index set):

| LEVEL | Denominator (REQ / ATOMIC) | Example panel |
|-------|----------------------------|---------------|
| `TCC_EA0_RDREQ_LEVEL` | `TCC_EA0_RDREQ` | 18.6 read latency |
| `TCC_EA0_WRREQ_LEVEL` | `TCC_EA0_WRREQ` | 18.7 WR+atomic latency |
| `TCC_EA0_ATOMIC_LEVEL` | `TCC_EA0_ATOMIC` | 18.8 atomic latency |

**Reason:** Latency (and similar) formulas are ratios. Joining LEVEL from one app replay with REQ/ATOMIC from another is wrong: each `pmc_perf_*.yaml` pass **re-executes** the workload, so L2 channel hashing / buffer VA can remap hot channels. Cross-pass ratio joins invent per-index outliers even when aggregate traffic matches.

### 1.3 Cover every selected series across passes; do not prune to runtime-nonzero

- Selection is from the **profile / YAML metric set**, not from a prior run’s nonzero channels.
- Do not drop “cold” channels at pack time. Zero is a valid observation; pruning biases panels and breaks channel-complete views (block 18).

### 1.4 Do not duplicate the same REQ series into two passes with different channel maps

- Ban placing the same per-channel EA REQ family (e.g. `TCC_EA0_WRREQ[*]`) in two passes when those passes can disagree on the per-index fingerprint.
- Even when both YAMLs list the same index set `0…N-1`, **runtime channel maps diverge across replays** (261001 mega_kernel: WRREQ medians differed by up to ~12× on some channels between SPP pass 1 and pass 2).
- Practical rule: if LEVEL lives with REQ in pass *A*, do **not** also emit that REQ series in pass *B* as an “orphan” copy for some other packable union.

`*_sum` aggregate counters in other passes are out of scope for this channel-map rule (they are not per-channel series with a remap fingerprint).

---

## 2. Reasons (discussion + AQL / HW facts)

1. **AQL / HW:** TCC block advertises **4** simultaneous event selects; SQ advertises **8**. rocprof-compute’s `mi_gpu_specs.get_perfmon_config("gfx942")` encodes `TCC: 4`, `SQ: 8`. Channel instances do not multiply the TCC select count.
2. **Ratio integrity:** Legacy already splits some LEVEL/REQ pairs across passes (261001: RD LEVEL vs RDREQ; ATOMIC vs ATOMIC_LEVEL). SPP fixed several of those, but still **duplicates** RDREQ/WRREQ into the atomic pass without LEVEL — a hazard if analyze ever picks the wrong copy.
3. **Remap evidence:** All eight mega_kernel median \|rel\| ≥ 100% block-18 deltas in `delta_ge100_investigation.md` are consistent with L2 channel-index remapping across independent passes; packing still must not invite cross-pass ratio joins or ambiguous REQ duplicates.
4. **Coverage:** Per-channel panels assume a full channel vector; runtime-nonzero pruning would silently drop channels and invalidate comparisons.

---

## 3. Pass-count evaluation (gfx942 default SPP)

**Method:** Offline `try_allocate_single_pass_packable` via `tools/eval_single_pass_packable.py --arch gfx942`, plus TCC layout from `validation-artifacts/spp-health-261001-gfx942-500med/.../mega_kernel_*_spp/.../perfmon/pmc_perf_*.yaml`. Simulated policy by stripping orphan REQ series that lack co-resident LEVEL when a LEVEL+REQ home pass already exists, then re-checking packable coverage.

### 3.1 Current SPP baseline

| | Value |
|--|------:|
| SPP passes (offline eval) | **14** |
| `packable_multi` | **0** |
| SLOT_LIMIT / SLOT +passes | 16 / **0** |
| TCC / SQ limits | **4** / **8** |

**TCC channel-event bases in current SPP (offline + mega_kernel agree on structure):**

| Pass | TCC channel bases (4/4) |
|------|-------------------------|
| `pmc_perf_0` | `TCC_ATOMIC`, `TCC_BUBBLE`, `TCC_READ`, `TCC_WRITE` |
| `pmc_perf_1` | `TCC_EA0_ATOMIC`, `TCC_EA0_ATOMIC_LEVEL`, `TCC_EA0_RDREQ`, `TCC_EA0_WRREQ` |
| `pmc_perf_2` | `TCC_EA0_RDREQ`, `TCC_EA0_RDREQ_LEVEL`, `TCC_EA0_WRREQ`, `TCC_EA0_WRREQ_LEVEL` |
| `pmc_perf_3` | `TCC_HIT`, `TCC_MISS`, `TCC_REQ`, `TCC_TOO_MANY_EA_WRREQS_STALL` |

Passes 4–13 hold non-channel / `_sum` PMCs only (no extra TCC channel series).

**Already compliant today:**

- LEVEL↔REQ for RD and WR in `pmc_perf_2`
- LEVEL↔ATOMIC in `pmc_perf_1`
- Full collectable channel expansion per base in a pass (mega_kernel: **16** instances/base)
- TCC occupancy uses series bases (4 bases/pass), not instance count

**Current violations of §1.4:**

- `TCC_EA0_RDREQ[*]` in **both** `pmc_perf_1` and `pmc_perf_2`
- `TCC_EA0_WRREQ[*]` in **both** `pmc_perf_1` and `pmc_perf_2`

Pass 1 copies are orphans relative to RD/WR LEVEL (LEVEL only in pass 2). They exist so packable union `1800.1805` (`TCC_EA0_ATOMIC` + `RDREQ` + `WRREQ` as one YAML row group) gets a single bucket.

### 3.2 After enforcing affinity + no orphan REQ dups

| Change | Effect |
|--------|--------|
| Remove orphan `RDREQ` / `WRREQ` from pass 1 | Pass 1 → `ATOMIC` + `ATOMIC_LEVEL` only (2/4 TCC) |
| Latency unions 1806 / 1807 / 1808 | Still co-resident (unchanged) |
| Total collection passes | **Still 14** |
| TCC channel lower bound (greedy affinity units, 14 selected series) | **Still 4** TCC channel passes |

**Conclusion: enforcing this policy does not increase collection passes vs current SPP (14 → 14, flat).**

Which passes change (layout only, not count):

- **`pmc_perf_1`:** drop orphan `TCC_EA0_RDREQ[*]` and `TCC_EA0_WRREQ[*]`
- **`pmc_perf_2`:** unchanged (canonical RD/WR LEVEL+REQ home)
- **`pmc_perf_0` / `pmc_perf_3` / passes 4–13:** unchanged for this policy

### 3.3 Packable-coverage caveat (1805), not a pass bump

Under today’s `_iter_metric_groups`, panel **1805** (“L2-Fabric Requests”) is one packable union of **three independent columns** (read=`RDREQ`, write=`WRREQ`, atomic=`ATOMIC`). After stripping orphan REQs, that combined union is no longer in one bucket → offline `packable_multi` would read **1** unless grouping is fixed.

That does **not** require a 15th pass:

- Each column alone is already covered after the strip (ATOMIC in pass 1; RDREQ/WRREQ in pass 2).
- Putting ATOMIC+RDREQ+WRREQ **and** both RD/WR LEVEL pairs in one pass needs **5–6** TCC bases &gt; **TCC:4** — impossible; the only way SPP “solved” 1805 was the banned REQ duplicate.
- Preferred follow-up (implementation later): treat 1805 columns as separate packable groups (or accept multi-bucket for the combined row without opening a new pass). Cross-pass **sums** for independent count columns are far less hazardous than LEVEL/REQ **ratios**.

### 3.4 Summary table

| Scenario | Passes | Notes |
|----------|-------:|-------|
| Current default SPP | **14** | Orphan RDREQ/WRREQ in pass 1 |
| SPP + TCC series affinity + coverage | **14** | **No increase**; strip orphans from pass 1 |
| TCC channel-series lower bound under affinity | **4** | Same as today |

**Pass-count verdict: no increase.**

---

## 4. Implementation notes (out of scope for this doc’s delivery)

Do **not** implement packing changes in the same change as this design/eval. When implementing later:

1. Encode affinity pairs as hard co-residence constraints in `try_allocate_single_pass_packable` (or a thin post-pass repair that strips orphan REQ dups and re-checks coverage).
2. Keep series-base packing / full channel expand (already largely true via `LimitedSet` + TCC file map).
3. Decide 1805 column grouping vs `packable_multi` gate before claiming `packable_multi == 0` under the new constraints.
4. Add an inspector / eval check: no REQ series in two channel passes; every LEVEL base co-resident with its REQ/ATOMIC.
5. Re-run `eval_single_pass_packable.py --arch gfx942` and a mega_kernel spot layout diff.

---

## 5. Cross-links to update when implementing

- [Single-pass packable plan](aiprofcomp-865-single-pass-packable-plan.md) §2 / §3.2 — reference this policy as a packing harden on top of SPP.
- Phase 2 WEIGHTED_AVG design — explicitly **exclude** this work (not SLOT_LIMIT).
- Metric validation / health gates — prefer aggregate + sorted-channel compares for block 18 (see investigation); affinity reduces packing-induced ratio risk, not remap noise.

---

## 6. End-user value evaluation (AIPROFCOMP-865)

**Question:** Does TCC series affinity + coverage help an end user **detect TCC problems**
(imbalance, latency, bandwidth, wrong ratios), or is it mainly internal packing hygiene?

### 6.1 What user-visible failures it prevents

| Failure mode | User-visible symptom | Policy fix |
|--------------|----------------------|------------|
| LEVEL and matching REQ/ATOMIC in **different** passes | Fabric latency panels (18.6–18.8) join numerator/denominator from **two app replays** with divergent L2 channel maps → invented per-channel outliers even when aggregate traffic is fine | §1.2 co-residence |
| Same EA REQ series in **two** passes (SPP today: RDREQ/WRREQ in p1 and p2) | Analyze/inspector may pick the orphan copy; joining LEVEL(p2) with WRREQ(p1) can yield absurd ratios (261001: ~19900 vs correct ~1641 on one channel) | §1.4 no orphan REQ dups |
| Partial channel subset of a selected series | Block-18 panels look incomplete / biased if cold channels are pruned at pack time | §1.1 / §1.3 full series expand + no nonzero prune |

These are **real correctness** risks for anyone reading per-channel latency or relying on a single REQ series identity. Affinity makes the default SPP layout match what a careful user already assumes: “ratio parts come from the same collection pass.”

### 6.2 What it does **not** fix

- **Cross-run / cross-pass channel reshuffle:** Each `pmc_perf_*.yaml` pass (and each legacy vs SPP profile) **re-executes** the app → new buffer VAs → L2 hash remaps hot traffic to different channel indices. Affinity does not stabilize indices across independent runs.
- **SPP-vs-legacy fixed-index health deltas:** The eight mega_kernel median \|rel\| ≥ 100% block-18 rows in `delta_ge100_investigation.md` are **comparison artifacts** (same hotspot fingerprint, permuted indices). Aggregates and sorted-channel pairs already agree (Σ REQ identical; sorted max ~1.5%). Affinity would not have cleared those gates.
- **Need for better compares:** End users (and health gates) still need **aggregates (18.1)** and **sorted / assignment-cost channel compares** to detect true imbalance regressions. Affinity does not replace that.

### 6.3 Were ≥100% health deltas “real TCC bugs”?

**No — not for end-user TCC health.** All eight were remapping / fixed-index noise. Secondary packing notes from the same investigation (legacy LEVEL/REQ splits; SPP orphan REQ dups) **are** real hazards, but they did not drive those eight median deltas under the current analyze join path (analyze used the co-resident pair where available).

### 6.4 Net verdict — ship for default SPP?

| Lens | Verdict |
|------|---------|
| Pass cost | **Free** (14 → 14) |
| End-user TCC *detection* of imbalance/BW | **Weak direct win** — does not surface new imbalance signals; remap noise remains |
| End-user TCC *correctness* of latency ratios | **Yes** — prevents packing-induced wrong ratios and ambiguous REQ copies |
| Primary character | **Packing hygiene / correctness insurance**, not a new “find hot channels” feature |

**Recommendation:** Worth shipping for **default SPP** as a low-cost correctness harden (especially with legacy still splitting some EA pairs). Do **not** sell it as the fix for block-18 health flake; pair with aggregate + sorted-channel gates and (optionally) alloc-once / pinned health workloads for within-run VA stability ([health workload note](aiprofcomp-865-health-workload-pinned-alloc-once.md)).
