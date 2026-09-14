# AIPROFCOMP-78 — Hardware Validation Runbook (Conductor + CPX)

**JIRA:** AIPROFCOMP-78  
**Related:** [Grouping evaluation §8](../design/single-pass-counter-grouping-evaluation.md#8-hardware-validation-conductor--cpx)

---

## Target system (gfx942 CPX)

| Field | Value |
|-------|-------|
| **Booked node** | `hpe-darkstar-ccs-aus-e12-03` (MI300X, gfx942) |
| **Conductor pool** | `MI300X-AIG-SW-ML-LIBRARIES` |
| **Entity ID** | `dff206e0-2724-4bd1-9d67-f21cf3158a45` |
| **Hostname** | `hpe-darkstar-ccs-aus-e12-03.cs-aus.dcgpu` |
| **SSH** | `feizheng@hpe-darkstar-ccs-aus-e12-03.cs-aus.dcgpu` |
| **Reservation** | `06a90967-5dce-7ed0-8000-95ffbeeefaa0` |
| **Window** | 2026-08-27 **21:00–01:00 CDT** (4 h) |
| **Partition mode** | CPX on 5/8 cards (`amd-smi set --compute-partition CPX`; 3 cards busy/SPX) |

**Alternates** (dry-run OK with `rocprof-compute` team, same pool class):

| Node | Pool | Entity ID |
|------|------|-----------|
| `splinter-odcdh4-wbc1-c` | `MI300X-AIG-SW-Shared-Pool` | `17c3f3ec-89a2-4da7-b48d-f86b8c1b8e06` |
| `dell300x-ccs-aus-f03-19` | `MI300X-AIG-SW-Shared-Pool` | `21c999cd-4a73-4149-8142-8c65fa9a30e8` |

Access via Conductor reservation (not Alola Slurm). If SSH fails, run `conductor system sut-auth-remediate --system hpe-darkstar-ccs-aus-e12-03` or use Conductor batch jobs against the reserved entity.

---

## Environment (validated 2026-08-27)

Use **ROCm 7.15** on the node — default `/opt/rocm` (7.2.1) is too old for current `sdk_config` and crashes on gfx1250 accumulate counters.

```bash
ROCM=/cluster/apps/ubuntu-24/rocm/rocm-7.15.0.dev.df64a75
export ROCM_PATH=$ROCM PATH=$ROCM/bin:$PATH \
  LD_LIBRARY_PATH=$ROCM/lib:$ROCM/lib/rocprofiler-sdk:$ROCM/lib/rocprofiler-compute
export HIP_VISIBLE_DEVICES=9   # idle CPX logical GPU (BDF 0000:53:00.0)

rocprof-compute --version   # 3.8.0 (bundled with ROCm 7.15)
```

**Note:** `sysinfo.csv` may report **SPX / 304 CUs** because amd-smi queries the first visible device. The application runs on the GPU selected by `HIP_VISIBLE_DEVICES`.

**Workload substitute (historical):** Full-panel pass-count validation used `./sample/mat_mul_max` when `mat_exp` was not yet built. **`mat_exp` runs with ROCm 7.15** — build with `make CXX=hipcc` under `HPCTrainingExamples/.../streams_sync/hip/` (see Follow-up 5). Default `/opt/rocm` (7.2.1) lacks the bundled rocprof-compute 3.8.0 path used here; use `/cluster/apps/ubuntu-24/rocm/rocm-7.15.0.dev.df64a75` explicitly.

---

## Follow-up 1 — Policy pass count vs inspector

1. Baseline inspector (empty policy):

   ```bash
   python3 tools/counter_grouping_inspector.py --arch gfx942 | rg '^Summary:'
   # Expect: Summary: 13 bucket(s)
   ```

2. Add proposed gfx942 entries to `profiling_counter_grouping_policy.yaml`, re-run inspector:

   ```bash
   # Expect: Summary: 19 bucket(s) for 6.1.2, 17.2.1, or all cap metrics (+6 passes)
   ```

3. On hardware (CPX, full panel), profile twice with empty vs policy YAML:

   ```bash
   rocprof-compute profile --name matmul_full_empty -VV -- ./sample/mat_mul_max
   # patch policy YAML (see grouping eval doc), then:
   rocprof-compute profile --name matmul_full_policy -VV -- ./sample/mat_mul_max
   ```

4. Record from each run:
   - Kernel re-runs: `grep -c 'Current input file' profile.log`
   - Wall time from log `START:` / `END:` timestamps
   - Compare to inspector `Summary` (13 vs 19)

### Results (2026-08-27, darkstar)

| Run | Policy | Kernel passes | Wall time | Inspector |
|-----|--------|---------------|-----------|-----------|
| `aiprofcomp78_matmul_full_empty` | `{}` | **13** | **108 s** | 13 ✓ |
| `aiprofcomp78_matmul_full_policy` | `6.1.2`, `17.2.1`, `5.1.0`, `15.4.0` | **13** | **104 s** | 19 (3.10+ inspector only) |

**Finding:** Bundled **rocprof-compute 3.8.0** does not apply gfx942 `same_bucket_priority_metric_ids` — `perfmon/pmc_perf_*.yaml` files are **byte-identical** between empty and patched-policy runs. Inspector **13 → 19** prediction requires **rocprof-compute 3.10+** (not yet validated on hardware; 3.10 source + ROCm 7.15 hits SQG SDK abort during counter detection).

---

## Follow-up 2 — Block 17 single-pass HBM validation

1. Profile with L2 panel only (partners co-located; inspector predicts **8 passes**):

   ```bash
   rocprof-compute profile --name occupancy_b17 -VV --block 17 -- ./sample/occupancy   # test case occupancy sample
   ```

2. Inspect raw PMC before merge (per-pass CSV or analyze):

   ```bash
   # Per-dispatch HBM: 100 * TCC_EA0_RDREQ_DRAM_sum / TCC_EA0_RDREQ_sum
   python3 ~/Downloads/aiprofcomp78-darkstar-data/analyze_darkstar_hardware.py
   ```

3. Compare to merged full-panel CPX data (`~/Downloads/aiprofcomp78-cpx-data/`):

   | Dataset | Dispatches | `a > b` rate | Max ratio |
   |---------|------------|--------------|-----------|
   | occupancy CPX full panel (merged) | 8 | **25%** (2/8) | 103.9% |
   | mat_exp CPX full panel (merged) | 208 | **7.2%** (15/208) | 188.2% |
   | **darkstar `--block 17` occupancy (pass 2, co-located)** | 8 | **0%** (0/8) | **100.0%** |

### Results (2026-08-27, darkstar)

| Metric | Inspector | Hardware |
|--------|-----------|----------|
| Pass count | **8** | **8** ✓ |
| HBM partners same pass | `pmc_perf_2.yaml` | `pmc_perf_2.yaml` ✓ |
| Wall time | — | **85 s** |
| Raw `a > b` dispatches | expect ~0% | **0%** ✓ |

---

## Follow-up 3 — Block 5 CPF Utilization (split partners)

Unlike block 17 HBM, **`--block 5` does not co-locate** CPF Utilization partners. Inspector predicts **5 passes** with `CPF_CPF_STAT_BUSY` in bucket **0** and `CPF_CPF_STAT_IDLE` in bucket **1**. This run confirms pass layout on hardware and measures stitched per-dispatch ratios — it does **not** prove co-located single-pass collection (that requires grouping policy `5.1.0` on rocprof-compute **3.10+**).

1. Profile with CPC/CPF panel only:

   ```bash
   rocprof-compute profile --name occupancy_b5 -VV --block 5 -- ./sample/occupancy   # test case occupancy sample
   ```

2. Confirm pass split in perfmon YAML:

   ```bash
   WL=workloads/aiprofcomp78_occupancy_b5/MI300X_A1
   rg -l CPF_CPF_STAT_BUSY  "$WL/perfmon/"   # expect pmc_perf_0.yaml
   rg -l CPF_CPF_STAT_IDLE  "$WL/perfmon/"   # expect pmc_perf_1.yaml
   ```

3. Per-dispatch CPF Util proxy (stitched across passes, like analyze):

   ```python
   # 100 * busy / (busy + idle)  per Dispatch_ID after merging all results_pmc_perf_*.csv
   python3 ~/Downloads/aiprofcomp78-darkstar-data/analyze_darkstar_hardware.py
   ```

4. Compare to full-panel merged CPX baseline (`~/Downloads/aiprofcomp78-cpx-data/`):

   | Dataset | Passes | Partners co-located? | Dispatches | `busy/(busy+idle) > 100%` | Max ratio |
   |---------|--------|----------------------|------------|---------------------------|-----------|
   | mat_exp CPX full panel (merged analyze avg) | 13 | No | 208 | 0 per-row* | 100% (avg **101.05%** in uncapped analyze) |
   | occupancy CPX full panel (merged) | 13 | No | 8 | 0 | 100% |
   | **darkstar `--block 5` occupancy (stitched)** | **5** | **No** (pass 0 vs 1) | 8 | **0%** (0/8) | **100.0%** |

   \*Per-dispatch `busy ≤ busy+idle` can still hold while **aggregate** `SUM(busy)/SUM(busy+idle)` exceeds 100% on full-panel mat_exp (uncapped analyze).

### Results (2026-08-27, darkstar)

| Check | Inspector | Hardware |
|-------|-----------|----------|
| Pass count | **5** | **5** ✓ |
| `CPF_CPF_STAT_BUSY` pass | bucket `0` | `pmc_perf_0.yaml` ✓ |
| `CPF_CPF_STAT_IDLE` pass | bucket `1` | `pmc_perf_1.yaml` ✓ |
| Partners same pass | **No** | **No** ✓ |
| Wall time | — | **~50 s** |
| Stitched `busy/(busy+idle) > 100%` | expect possible | **0%** on occupancy (8 dispatches) |

**Finding:** Hardware confirms the inspector split. Block-only `--block 5` remains **multi-pass for CPF Utilization** (same structural pattern as Workgroup Manager Utilization). Occupancy shows no bound violations after stitching; mat_exp full-panel **avg** inflation (101.05%) is not reproduced on this workload — co-location validation for CPF still requires policy **3.10+** or a workload that shows per-row violations.

---

## Follow-up 4 — rocflop CPX single-pass (HBM + WGM)

§3.1 rows 7–8 cover two metrics on **`./sample/rocflop`** (7 dispatches on CPX). Run separate profiles — HBM partners co-locate on `--block 17`; WGM uses metric-id **`--block 6.1.2`** (**1 pass**).

1. Build and profile:

   ```bash
   cd sample && hipcc -O3 rocflop.cpp -o rocflop && cd ..
   rocprof-compute profile --name rocflop_b17 -VV --block 17 -- ./sample/rocflop
   rocprof-compute profile --name rocflop_m612 -VV --block 6.1.2 -- ./sample/rocflop
   ```

2. Compare to merged full-panel rocflop CPX (`~/Downloads/aiprofcomp78-cpx-data/rocflop/0/`):

   | Metric | Merged CPX (§3.1) | `--block` | Co-located? | `a > b` (darkstar) | Max ratio |
   |--------|-------------------|-----------|-------------|-------------------|-----------|
   | HBM Read Traffic | 99.70% avg; 0/7 per-row merged | 17 | **Yes** (`pmc_perf_2.yaml`) | **0/7 (0%)** | **100.0%** |
   | Workgroup Manager Utilization | 100.08% avg; 1/7 per-row merged | **6.1.2** | **Yes** (**1 pass**) | **0/7 (0%)** | **100.0%** |

### Results (2026-08-27, darkstar)

| Run | Passes | Wall time |
|-----|--------|-----------|
| `aiprofcomp78_rocflop_b17` | **8** | **101 s** |
| `aiprofcomp78_rocflop_m612` | **1** | **~19 s** |

**Finding:** HBM and WGM on rocflop **validated** with co-located collection (`--block 17` and **`--block 6.1.2`**). Panel `--block 6` (split) remains a structural check only — see historical `aiprofcomp78_rocflop_b6`.

---

## Follow-up 5 — mat_exp (streams_sync) with ROCm 7.15

`mat_exp` is **not** in `./sample/`; it lives under HPCTrainingExamples. **rocblas is present** in `/cluster/apps/ubuntu-24/rocm/rocm-7.15.0.dev.df64a75` — set `ROCM_PATH` and build with **`hipcc`** (the Makefile defaults to `g++`, which fails).

1. Build:

   ```bash
   ROCM=/cluster/apps/ubuntu-24/rocm/rocm-7.15.0.dev.df64a75
   export ROCM_PATH=$ROCM PATH=$ROCM/bin:$PATH \
     LD_LIBRARY_PATH=$ROCM/lib:$ROCM/lib/rocprofiler-sdk:$ROCM/lib/rocprofiler-compute
   MAT=~/aiprofcomp78/HPCTrainingExamples/Libraries/matrix_exponential/streams_sync/hip
   cd "$MAT" && make clean && make CXX=hipcc && ./mat_exp   # expect PASSED!
   ```

2. Profile (208 dispatches — matches CPX full-panel mat_exp):

   ```bash
   rocprof-compute profile --name mat_exp_b17 -VV --block 17 -- "$MAT/mat_exp"
   rocprof-compute profile --name mat_exp_b6 -VV --block 6 -- "$MAT/mat_exp"
   ```

3. Compare to §3.1 / §3.3 merged CPX baseline:

   | Metric | Merged CPX | `--block` | Co-located? | darkstar `a > b` | Max ratio |
   |--------|------------|-----------|-------------|------------------|-----------|
   | HBM Read Traffic | 15/208 (**7.2%**); max **188.2%** | 17 | **Yes** | **0/208 (0%)** | **100.0%** |
   | Workgroup Manager Utilization | 2/206 (**1.0%**); max **739.6%** | 6 | **No** | **3/208 (1.4%)** | **605.9%** |

### Results (2026-08-27, darkstar)

| Run | Passes | Wall time |
|-----|--------|-----------|
| `aiprofcomp78_mat_exp_b17` | **8** | **~42 s** |
| `aiprofcomp78_mat_exp_b6` | **3** | **~21 s** |

**Finding:** mat_exp HBM **validated** on the real workload (same 208 dispatches as CPX). WGM/CPF/Data-Return co-located validation uses metric-id `--block` (Follow-up 7).

---

## Follow-up 7 — Metric-id single-pass validation (§3.5 rows 3–5, rocflop WGM)

Panel `--block 5/6/15` collects whole hardware panels and may still split ratio partners. Filter by **metric id** so only formula counters are collected — typically **1 kernel pass**, partners in one `pmc_perf_0.yaml`.

| Metric | Metric id | mat_exp darkstar | rocflop darkstar |
|--------|-----------|------------------|------------------|
| Workgroup Manager Utilization | `6.1.2` | **1 pass**, 0/208 `a > b`, max **100.0%** | **1 pass**, 0/7, max **100.0%** |
| CPF Utilization | `5.1.0` | **1 pass**, 0/208, max **100.0%** | — |
| Data-Return Busy | `15.4.0` | **1 pass**, 0/208, max **0.04%** | — |

1. Profile (bundled 3.8.0 accepts metric-id `--block`):

   ```bash
   MAT=~/aiprofcomp78/HPCTrainingExamples/Libraries/matrix_exponential/streams_sync/hip/mat_exp
   rocprof-compute profile --name mat_exp_m612 -VV --block 6.1.2 -- "$MAT/mat_exp"
   rocprof-compute profile --name mat_exp_m510 -VV --block 5.1.0 -- "$MAT/mat_exp"
   rocprof-compute profile --name mat_exp_m1540 -VV --block 15.4.0 -- "$MAT/mat_exp"
   rocprof-compute profile --name rocflop_m612 -VV --block 6.1.2 -- ./sample/rocflop
   ```

   Or one combined profile (same 1-pass layout per inspector):

   ```bash
   rocprof-compute profile --set aiprofcomp78_bounds --name mat_exp_bounds -- "$MAT/mat_exp"
   ```

   (`aiprofcomp78_bounds` set ships in `profile_configs/sets/gfx942_sets.yaml`.)

2. Offline pass plan:

   ```bash
   python3 tools/counter_grouping_inspector.py --arch gfx942 --block 6.1.2
   python3 tools/counter_grouping_inspector.py --arch gfx942 --block 5.1.0 15.4.0
   # Expect: Summary: 1 bucket(s); target metric in single-bucket table
   ```

3. Per-dispatch ratios: merge `results_pmc_perf_*.csv` (single file when 1 pass) or run `analyze_darkstar_hardware.py`.

### Results (2026-08-27, darkstar)

| Run | Passes | Wall time |
|-----|--------|-----------|
| `aiprofcomp78_mat_exp_m612` | **1** | **~10 s** |
| `aiprofcomp78_mat_exp_m510` | **1** | **~10 s** |
| `aiprofcomp78_mat_exp_m1540` | **1** | **~10 s** |
| `aiprofcomp78_rocflop_m612` | **1** | **~7 s** |

**Finding:** Merged full-panel max violations (WGM **739.6%**, Data-Return **345.38%**) **do not reproduce** under co-located metric-id collection. Root cause for §3.1 bound violations is **multi-pass stitching**, not HW counter error on stable mat_exp/rocflop workloads.

---

## Follow-up 6 — mat_exp block 5 (CPF) + block 15 (Data-Return Busy)

§3.1 rows 4–5 use **mat_exp CPX** merged baselines. Block profiles confirm inspector partner splits on the **same workload** (208 dispatches) — they do **not** co-locate partners without grouping policy.

1. Profile:

   ```bash
   MAT=~/aiprofcomp78/HPCTrainingExamples/Libraries/matrix_exponential/streams_sync/hip/mat_exp
   rocprof-compute profile --name mat_exp_b5 -VV --block 5 -- "$MAT/mat_exp"
   rocprof-compute profile --name mat_exp_b15 -VV --block 15 -- "$MAT/mat_exp"
   ```

2. Confirm pass splits:

   ```bash
   # CPF (--block 5): busy pass 0, idle pass 1
   rg -l CPF_CPF_STAT_BUSY  workloads/aiprofcomp78_mat_exp_b5/MI300X_A1/perfmon/   # pmc_perf_0.yaml
   rg -l CPF_CPF_STAT_IDLE  workloads/aiprofcomp78_mat_exp_b5/MI300X_A1/perfmon/   # pmc_perf_1.yaml

   # Data-Return (--block 15): GRBM pass 0, TD pass 3
   rg -l GRBM_GUI_ACTIVE workloads/aiprofcomp78_mat_exp_b15/MI300X_A1/perfmon/     # pmc_perf_0.yaml
   rg -l TD_TD_BUSY_sum  workloads/aiprofcomp78_mat_exp_b15/MI300X_A1/perfmon/      # pmc_perf_3.yaml
   ```

3. Per-dispatch stitched ratios (after merging all `results_pmc_perf_*.csv`):

   | Metric | Formula | Merged CPX (§3.1) | darkstar `--block` | Co-located? | `a > b` | Max ratio |
   |--------|---------|-------------------|--------------------|-------------|---------|-----------|
   | CPF Utilization | `100 × busy / (busy + idle)` | avg **101.05%** | **5** mat_exp | **No** | **0/208 (0%)** | **100.0%** |
   | Data-Return Busy | `100 × TD_TD_BUSY_sum / (GRBM_GUI_ACTIVE × cu_per_gpu)` | max **345.38%** | **15** mat_exp | **No** | **0/208 (0%)** | **64.4%** |

4. Local analysis:

   ```bash
   python3 ~/Downloads/aiprofcomp78-darkstar-data/analyze_darkstar_hardware.py
   ```

### Results (2026-08-27, darkstar)

| Run | Passes | Wall time |
|-----|--------|-----------|
| `aiprofcomp78_mat_exp_b5` | **5** | **~32 s** |
| `aiprofcomp78_mat_exp_b15` | **8** | **~47 s** |

**Finding:** Hardware confirms inspector splits on mat_exp. Stitched block-only merges show **no bound violations** for CPF or Data-Return (unlike full-panel merged max **345.38%**). True co-located validation still requires grouping policy **`5.1.0`** / **`15.4.0`** on rocprof-compute **3.10+**.

---

## CPX setup (Conductor reservation)

After Conductor reservation is active on the SUT:

```bash
sudo amd-smi set --compute-partition CPX || rocm-smi --setcomputepartition CPX
rocm-smi --showcomputepartition
# Expect CPX on idle cards; busy cards return AMDSMI_STATUS_BUSY
```

On a shared 8-GPU node, pick an idle CPX logical GPU:

```bash
rocm-smi --showpids
export HIP_VISIBLE_DEVICES=9   # example: CPX partition 0 on BDF 0000:53:00.0
```

Verify compute partition with `rocm-smi -d $HIP_VISIBLE_DEVICES --showcomputepartition` (not only `sysinfo.csv`).

---

## Local artifacts

Data copied to **`~/Downloads/aiprofcomp78-darkstar-data/`**:

| Path | Content |
|------|---------|
| `aiprofcomp78_occupancy_b17/` | Block 17 occupancy workload |
| `aiprofcomp78_occupancy_b5/` | Block 5 CPF panel (split partners) |
| `aiprofcomp78_rocflop_b17/` | Block 17 rocflop (HBM co-located) |
| `aiprofcomp78_rocflop_b6/` | Block 6 rocflop (WGM split) |
| `aiprofcomp78_mat_exp_b17/` | Block 17 mat_exp streams_sync (HBM co-located) |
| `aiprofcomp78_mat_exp_b6/` | Block 6 mat_exp (WGM split) |
| `aiprofcomp78_mat_exp_b5/` | Block 5 mat_exp (CPF split) |
| `aiprofcomp78_mat_exp_b15/` | Block 15 mat_exp (Data-Return split) |
| `aiprofcomp78_mat_exp_m612/` | Metric `6.1.2` mat_exp (WGM co-located) |
| `aiprofcomp78_mat_exp_m510/` | Metric `5.1.0` mat_exp (CPF co-located) |
| `aiprofcomp78_mat_exp_m1540/` | Metric `15.4.0` mat_exp (Data-Return co-located) |
| `aiprofcomp78_rocflop_m612/` | Metric `6.1.2` rocflop (WGM co-located) |
| `profile_mat_exp_b17.log` | mat_exp block 17 profile log |
| `profile_mat_exp_b5.log` / `profile_mat_exp_b15.log` | mat_exp block 5 / 15 profile logs |
| `profile_occupancy_b5.log` | Block 5 profile log |
| `profile_rocflop_b17.log` / `profile_rocflop_b6.log` | rocflop profile logs |
| `aiprofcomp78_matmul_full_empty/` | Full panel, empty policy |
| `aiprofcomp78_matmul_full_policy/` | Full panel, patched policy (no perfmon change on 3.8.0) |
| `profile_*.log` | Profile logs with timestamps |
| `analyze_darkstar_hardware.py` | Local pass-count + HBM ratio script |
| `analyze_darkstar_hardware.log` | Script output |
