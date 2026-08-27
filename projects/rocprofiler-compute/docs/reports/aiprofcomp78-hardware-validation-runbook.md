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
| **Reservation** | `06a90967-5dce-7ed0-8000-95ffbeeefaa0` |
| **Window** | 2026-08-27 **21:00–01:00 CDT** (4 h) |
| **Partition mode** | CPX (`amd-smi set --compute-partition CPX` — typically available with Conductor reservation/sudo) |

**Alternates** (dry-run OK with `rocprof-compute` team, same pool class):

| Node | Pool | Entity ID |
|------|------|-----------|
| `splinter-odcdh4-wbc1-c` | `MI300X-AIG-SW-Shared-Pool` | `17c3f3ec-89a2-4da7-b48d-f86b8c1b8e06` |
| `dell300x-ccs-aus-f03-19` | `MI300X-AIG-SW-Shared-Pool` | `21c999cd-4a73-4149-8142-8c65fa9a30e8` |

Access via Conductor reservation (not Alola Slurm). If SSH fails, run `conductor system sut-auth-remediate --system hpe-darkstar-ccs-aus-e12-03` or use Conductor batch jobs against the reserved entity.

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

## CPX setup (Conductor reservation)

After Conductor reservation is active on the SUT:

```bash
amd-smi set --compute-partition CPX || rocm-smi --setcomputepartition CPX
rocm-smi --showcomputepartition
# Expect CPX, cu_per_gpu=38, num_xcd=1 in sysinfo after profile
```

Verify in `workloads/*/0/sysinfo.csv` before analyze.
