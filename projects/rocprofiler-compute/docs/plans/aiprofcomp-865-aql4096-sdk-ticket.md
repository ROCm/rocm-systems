# SQ accumulate slot accounting / AQL 4096 — SDK ticket draft

**Component:** rocprofiler-sdk / AQLProfile (rocprofv3 PMC collection)
**Observed with:** ROCm 7.1 / 7.14 / TheRock 7.14.0a20260612 / TheRock 7.15.0a20260728
**GPUs:** MI300A/X (gfx942 SPX), MI350 (gfx950 SPX), MI250 (gfx90a)
**Reporter context:** Found while profiling with rocprof-compute; reproducible with standalone `rocprofv3`

## Summary

On gfx942, AQLProfile advertises **8** SQ block counter registers
(`SqCounterBlockNumCounters = 8`). Collecting **7 plain SQ PMCs** succeeds.
Adding one stock derived metric that uses `accumulate(..., HIGH_RES)`
(`VmemLatency`) fails.

That is consistent with **accumulate consuming more than one SQ register**
(e.g. base level event + HIGH_RES / `SQ_ACCUM_PREV_HIRES` pairing → **2 slots**),
so the pack becomes **7 + 2 = 9 > 8**.

Depending on the path, the failure surfaces as either:

1. SDK `ROCPROFILER_STATUS_ERROR_EXCEEDS_HW_LIMIT` (error **38**):
   *“Request exceeds the capabilities of the hardware to collect”* — correct
   class of error, but no detail (block, used vs max, accumulate cost).
2. Mid-run fatal **`AQLProfile Return Code: 4096`** (opaque `HSA_STATUS_ERROR`)
   when using named `accumulate(...)` / `*_ACCUM` via custom metrics — should
   have been rejected as error 38 before packet create.

## Ask for SDK

1. **Confirm accumulate HW cost:** For
   `accumulate(SQ_INST_LEVEL_*, HIGH_RES)` (and stock metrics such as
   `VmemLatency` / `SmemLatency`), does AQL/SDK count this as **two** SQ
   counter registers (base + accum) against the block limit of 8 on gfx942?
   Please confirm the exact accounting (including any injected dummy/base
   events in `CounterMemoryManager::CopyEvents`).
2. **If yes (accum = 2):** Fail early with a clear
   `ROCPROFILER_STATUS_ERROR_EXCEEDS_HW_LIMIT` (error 38) message that
   includes block name, used count, and max — and **do not** fall through to
   fatal `AQLProfile Return Code: 4096` when the pack simply exceeds the SQ
   register limit.

(Docs already say a single `--pmc` / `pmc` group must fit one HW pass and to
use multi-pass when it does not. The gap is confirming accumulate’s slot cost
and replacing AQL 4096 with that capability error.)

## Minimal reproduction

Tree: `scripts/aql4096_rocprofv3_repro/`

```bash
export ROCM_PATH=/path/to/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}
./scripts/aql4096_rocprofv3_repro/run_repro.sh          # CASE=stock → error 38
CASE=accum ./scripts/aql4096_rocprofv3_repro/run_repro.sh  # → AQL 4096 path
```

### Stock set (error 38) — `pmc_fail_stock_vmemlatency.yaml`

```text
SQ_ACTIVE_INST_ANY
SQ_INSTS
SQ_INSTS_MFMA
SQ_INSTS_SMEM
SQ_INSTS_VALU
SQ_VALU_MFMA_BUSY_CYCLES
SQ_WAVES          # 7 SQ events
VmemLatency       # accumulate(SQ_INST_LEVEL_VMEM, HIGH_RES) / SQ_INSTS_VMEM
```

| Case | Result |
|------|--------|
| 7 SQ only | OK (≤ 8) |
| `VmemLatency` alone | OK |
| 7 SQ + `VmemLatency` | **error 38** |
| 7 SQ + `SmemLatency` | **error 38** (same accumulate pattern) |

Stock `VmemLatency`:

```yaml
expression: reduce(accumulate(SQ_INST_LEVEL_VMEM, HIGH_RES),sum)/reduce(SQ_INSTS_VMEM,sum)
```

### Metrics-path set (AQL 4096) — `CASE=accum`

Seven SQ + `SQ_INST_LEVEL_SMEM_ACCUM` with
`expression: accumulate(SQ_INST_LEVEL_SMEM, HIGH_RES)` under
`ROCPROFILER_METRICS_PATH` → mid-run:

```text
Could not create PMC packets! AQLProfile Return Code: 4096
Events: [0,61,6],[0,26,6],[0,73,6],...
```

Without metrics path, `*_ACCUM` is missing/skipped, so the same names look OK.

## Evidence

- Alola MI300A + TheRock 7.15: 7 SQ + `VmemLatency` → error 38; halves OK
- Same: 7 SQ + `SmemLatency` → error 38
- Conductor MI300X + TheRock 7.15: 7 SQ + `*_ACCUM` (metrics path) → AQL 4096
- gfx942 SQ limit from AQL: `SqCounterBlockNumCounters = 8` (also
  rocprofiler-compute `mi_gpu_spec.yaml` `perfmon_config.SQ: 8`)

## Attachments when filing

- `scripts/aql4096_rocprofv3_repro/`
- Host: GPU model, partition mode, `rocprofv3 --version`, `ROCM_PATH`
- Logs: error 38 case and AQL 4096 `Events: [...]` line
