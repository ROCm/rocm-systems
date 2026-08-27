# AIPROFCOMP-78 — Hardware Validation Runbook (Conductor + CPX)

**JIRA:** AIPROFCOMP-78  
**Related:** [Grouping evaluation §8](../design/single-pass-counter-grouping-evaluation.md#8-hardware-validation-conductor--cpx)

---

## Target system (gfx942 CPX)

| Field | Value |
|-------|-------|
| **Recommended node** | `ctr-cx71-mi300x-01` (MI300X, gfx942) |
| **Conductor pool** | `AIG-SW-Alola` (entity `6e2c399f-b1af-40fa-8e19-3004d3930c72`) |
| **Alola Slurm** | `--gres=gpu:gfx942-mi300x:1 -w ctr-cx71-mi300x-01` |
| **Partition mode** | CPX (`amd-smi set --compute-partition CPX` — requires **reservation/sudo** on Alola) |
| **Window** | 4 h (policy full-panel + block-17 re-profiles) |

**Conductor access note:** `fei.zheng@amd.com` is on the active team reservation **`rocprof-compute`** (`06a87e2c-7744-783e-8000-198db5aebe85`, `asrock-1w300-f2-1`, MI350X) but does **not** currently have permission to create new reservations on `AIG-SW-Alola` MI300X nodes. Request pool access or ask a pool admin to book `ctr-cx71-mi300x-01` for 4 h.

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
   rocprof-compute profile --name mat_exp_policy_empty -VV --overwrite -- ./mat_exp
   # restore policy entries
   rocprof-compute profile --name mat_exp_policy_on -VV --overwrite -- ./mat_exp
   ```

4. Record from each run:
   - `ls workloads/*/0/perfmon/pmc_perf*.yaml | wc -l` (pass count)
   - Wall time from profile log (`real` or log timestamps)
   - Compare to inspector `Summary` (13 vs 19)

---

## Follow-up 2 — Block 17 single-pass HBM validation

1. Profile with L2 panel only (partners co-located; inspector predicts **8 passes**):

   ```bash
   rocprof-compute profile --name occupancy_b17 -VV --block 17 --overwrite -- ./sample/occupancy
   rocprof-compute profile --name mat_exp_b17   -VV --block 17 --overwrite -- ./mat_exp
   ```

2. Analyze and inspect raw PMC before merge:

   ```bash
   rocprof-compute analyze -p workloads/occupancy_b17/0 --block 17
   rocprof-compute analyze -p workloads/mat_exp_b17/0   --block 17
   ```

3. Compare to merged full-panel CPX data (`aiprofcomp78-cpx-data/`):

   ```python
   # Per-dispatch HBM: 100 * TCC_EA0_RDREQ_DRAM_sum / TCC_EA0_RDREQ_sum
   # Expect: a > b row rate drops vs multi-pass full panel (occupancy 25% -> ~0%; mat_exp 7.2% -> ~0%)
   ```

---

## CPX setup (Alola reservation mode)

After Conductor/Slurm reservation with sudo:

```bash
amd-smi set --compute-partition CPX || rocm-smi --setcomputepartition CPX
rocm-smi --showcomputepartition
# Expect CPX, cu_per_gpu=38, num_xcd=1 in sysinfo after profile
```

Verify in `workloads/*/0/sysinfo.csv` before analyze.
