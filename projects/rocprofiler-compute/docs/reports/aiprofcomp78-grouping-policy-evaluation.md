# AIPROFCOMP-78 — Counter Grouping Policy Evaluation

**Branch:** `users/feizheng10/cpx_fix`  
**Date:** 2026-08-27  
**Arch:** gfx942 (MI300X)

**Related:** [Single-pass grouping evaluation (design)](../design/single-pass-counter-grouping-evaluation.md) · [Root cause report](aiprofcomp78-multipass-root-cause-report.md)

---

## 1. Question

Can `profiling_counter_grouping_policy.yaml` entries for affected metrics **eliminate** bound violations without unacceptable profile cost?

---

## 2. Current gfx942 policy

```yaml
gfx942:
  same_bucket_priority_metric_ids: {}
```

No same-bucket priorities. Default greedy packing splits ratio partners across accumulator passes (see root-cause report §6).

---

## 3. Measured cost — full-panel CPX profiles

| Workload | Perfmon passes (measured) | Profile scope |
|----------|---------------------------|---------------|
| occupancy_cpx | 13 | Full panel |
| mat_exp | 13 | Full panel |
| rocflop | 13 | Full panel |

Profile time ∝ pass count × kernel runtime. CPX (`cu_per_gpu=38`) reduces compute vs SPX but pass count is unchanged.

---

## 4. Allocator simulation — pass count vs policy

Tool: `OmniSoC_Base.detect_counters()` + `_allocate_perfmon_counter_files()` on shipped gfx942 YAML.

### 4.1 Scope reduction (no policy change)

| User selection | Simulated passes | Counters | vs full panel |
|----------------|------------------|----------|---------------|
| Full panel | 12 | 439 | baseline |
| `--block 17` | **8** | 34 | **−33%** |
| `--block 5,6` | **5** | 39 | **−58%** |
| `--block 15` | **8** | 28 | **−33%** |
| `--block 5,6,15,17` | **8** | 97 | **−33%** |

Block-only profiling is the **cheapest** way to get co-temporal counters for validation without policy edits.

### 4.2 Adding grouping policy (full panel) — typically increases passes

Adding `same_bucket_priority_metric_ids` on gfx942 full panel **typically increases** pass count in simulation. Metric-aware coalesce opens new buckets when priority groups cannot fit existing ones.

| `same_bucket_priority_metric_ids` | Simulated passes | Δ |
|-----------------------------------|------------------|---|
| `{}` (current) | 12 | — |
| `17.2.1` (HBM Read Traffic) | **17** | **+42%** |
| `6.1.2` (Workgroup Manager Utilization) | 17 | +42% |
| `17.2.1`, `6.1.2`, `5.1.0`, `15.4.0` | 17 | +42% |

Do **not** expect grouping policy to reduce passes on full-panel gfx942 profiles. The trade-off is co-location accuracy vs longer profile time.

### 4.3 Co-location effect (full panel)

| Pair | Separate passes (no policy) | Same pass (with priority) |
|------|----------------------------|---------------------------|
| HBM DRAM + RDREQ_sum | VMEM_ACCUM vs WAVES_ACCUM | Both in bucket `0` with `17.2.1` |
| GRBM_SPI + GRBM_GUI | LDS_ACCUM vs IFETCH_ACCUM | Both in bucket `0` with `6.1.2` |
| CPF busy + idle | IFETCH vs LDS | Both in bucket `0` with cap-metric priorities |

**Trade-off:** Co-location achievable, but **+5 passes** on full panel in simulation.

### 4.4 Block 17 only — policy adds no co-location benefit for HBM

| Policy | Passes | DRAM pass | RDREQ_sum pass | Same? |
|--------|--------|-----------|----------------|-------|
| `{}` | 8 | bucket `2` | bucket `2` | **Yes** |
| `17.2.1` | 8 | bucket `0` | bucket `0` | **Yes** |

For HBM validation, **`--block 17` alone** already co-locates partners; grouping policy is unnecessary for this metric subset.

---

## 5. Does grouping solve the problem?

| Scenario | Solves bound violations? | Cost | Recommendation |
|----------|-------------------------|------|----------------|
| Full panel + empty policy (today) | **No** — partners split | 12–13 passes | Caps needed for display |
| Full panel + priority policy | **Likely yes** (co-located partners) | **~17 passes (+42%)** | Only if accuracy > time |
| `--block 17` validation | **Likely yes** for HBM | **8 passes** | **Preferred validation path** |
| `--block 5,6,15,17` | **Likely yes** for all cap metrics | **8 passes** | Best validation bundle |
| Analyze-time caps (#10717) | Display only | 0 extra profile cost | Keep for default full panel |

**Not yet measured on hardware:** Re-profile mat_exp with `--block 17` and confirm `a > b` rows disappear in raw PMC. Simulation predicts co-location; validation run is outstanding.

---

## 6. Proposed gfx942 policy entries (if full-panel co-location is required)

Example entries (metric ids from panel YAML):

```yaml
gfx942:
  same_bucket_priority_metric_ids:
    "17.2.1":
      name: "HBM Read Traffic"
    "17.2.6":
      name: "HBM Write and Atomic Traffic"
    "6.1.2":
      name: "Workgroup Manager Utilization"
    "5.1.0":
      name: "CPF Utilization"
    "15.4.0":
      name: "Data-Return Busy"
```

**Before merge:** Confirm measured pass count and wall time on representative workloads (mat_exp, occupancy).

---

## 7. Summary

| Finding | Detail |
|---------|--------|
| Grouping is the accurate fix | When partners share a pass |
| Empty gfx942 policy | Contributes to partner splitting on full panel |
| Policy cost on full panel | **+42% passes** in simulation — not free |
| Block-only profiling | **8 passes**, HBM partners already co-located |
| Caps in #10717 | Correct stopgap for default multi-pass users |

---

## 8. Reproduce simulation

From `projects/rocprofiler-compute/`:

```bash
python3 tools/counter_grouping_inspector.py --arch gfx942
python3 tools/counter_grouping_inspector.py --arch gfx942 --block 17
grep -E 'Summary:|Workgroup Manager Utilization|HBM Read Traffic' /tmp/plan.txt  # after --output /tmp/plan.txt
```

See [single-pass-counter-grouping-evaluation.md §7](../design/single-pass-counter-grouping-evaluation.md#7-how-to-reproduce-pass-count-simulation) for full usage.
