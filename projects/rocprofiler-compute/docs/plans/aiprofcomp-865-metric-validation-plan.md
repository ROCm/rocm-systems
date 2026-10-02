# AIPROFCOMP-865 — Multi-arch metric validation (SPP)

**Status:** In progress
**Branch:** `users/feizheng10/aiprofcomp-865-cp-sat-local`
**Goal:** Validate default Single-pass packable (SPP) packing with full
rocprof-compute profiles (including roofline) across target GPUs, and produce
metric health reports comparable to prior MI450 / gfx1151 campaigns.

---

## Locked decisions

| # | Decision |
|---|----------|
| 1 | **gfx115x** primary SKU = **gfx1151** (Strix Halo). |
| 2 | **MI100** — run only if a TheRock (or equivalent usable) stack exists on the node; otherwise **skip**. |
| 3 | Profiles are **full** (all default blocks) **with roofline**. Prefer native counter tool (`rocprofiler-sdk` default); build native tool if needed. Do **not** pass `--no-roof` / `--no-native-tool` unless a host forces a documented workaround. |
| 4 | Health report generators are ported into this tree; artifacts and refined reports live under `validation-artifacts/`. |

### Arch order

1. **MI300** (`gfx942`)
2. **MI350** (`gfx950`)
3. **gfx1151**
4. **MI250** (`gfx90a`)
5. **MI100** (`gfx908`) — optional (see #2)

### Reserved follow-up

- [ ] **CPX mode on MI300** — after SPX health is signed off; separate partition run + health/delta (not in the main loop).

---

## How health reports were produced before

Prior agent campaigns used HTML generators under the EMU checkout (not always
present in sparse trees):

| Campaign | Generator | Panel config dir |
|----------|-----------|------------------|
| MI450 / gfx1250 | `tools/generate_gfx1250_metric_health_report.py` | `analysis_configs/gfx1250` |
| gfx1151 / gfx1153 | `tools/generate_gfx1153_metric_health_report.py` | `analysis_configs/gfx115x` |

Orchestration pattern (`run_health_*_local_driver.sh` + remote `run_health_<tag>.sh`):

1. Sync branch → GPU host (TheRock container or bare TheRock install).
2. Build apps → `rocprof-compute profile` → `analyze … --view table` into logs.
3. scp workloads + logs locally.
4. Run generator with `name:path` workload logs; optional `--baseline` for before/after.

**Analyze requirement:** `--view table` is mandatory so chart-only panels emit
plain metric tables. Default TTY analyze omits metrics and under-counts health.

This plan ports a unified generator:

```text
tools/generate_metric_health_report.py --arch <gfx…> --out <html> …
```

---

## Phase A — Tooling

1. Add `tools/generate_metric_health_report.py` (arch-selectable; baseline deltas).
2. Add `scripts/run_spp_health_<arch>.sh` (remote worker) + thin local driver
   (sync / wait / fetch / report), modeled on `run_health_260714_local_driver.sh`.
3. Artifact root:

```text
validation-artifacts/spp-health-<TAG>/
  <arch>/{vcopy,nbody,mega_kernel}_{spp,legacy}/
  logs/<arch>/*.log
  reports/<arch>_health_spp.html
  reports/<arch>_delta_spp_vs_legacy.html   # or baseline section in one HTML
  FAILURE_SUMMARY.md
```

---

## Phase B — Per-arch gate 0: mega_kernel smoke

Before any profiling on an arch:

```bash
# TheRock env on host
export PATH=$THEROCK/bin:$PATH
export LD_LIBRARY_PATH=$THEROCK/lib   # prefer TheRock-only; avoid mixing /opt/rocm

cd sample/mega_kernel
# build for offload-arch (Makefile targets or hipcc --offload-arch=…)
./mega_kernel_test …   # expect OVERALL PASSED (bypasses OK)
```

Fix small mega_kernel / Makefile issues if smoke fails. Do not start profiling
until smoke is green.

| Arch | Offload | Notes |
|------|---------|-------|
| MI300 | gfx942 | Conductor / darkstar TheRock 7.15 |
| MI350 | gfx950 | MI350 TheRock host |
| gfx1151 | gfx1151 | e.g. gorgon / Strix Halo |
| MI250 | gfx90a | alola `gfx90a-mi250` |
| MI100 | gfx908 | **skip if no TheRock** |

---

## Phase C — Apps under profile

| App | Build | Run args |
|-----|-------|----------|
| **vcopy** | `hipcc --offload-arch=<arch> -O3 sample/vcopy.cpp -o sample/vc` | `./sample/vc -n 81920 -b 256` |
| **nbody** | From [HIP-Examples nbody-block.cpp](https://github.com/ROCm/HIP-Examples/blob/master/mini-nbody/hip/nbody-block.cpp): `hipcc -DSHMOO --offload-arch=<arch> sample/mini-nbody-block.cpp -o sample/mini-nbody-block` | `./sample/mini-nbody-block 131072` |
| **mega_kernel** | Arch-specific (see sample README) | CDNA: e.g. `-b 65536`; gfx1151: README defaults; do **not** use gfx1250’s mistaken `-t 32` wave-size pattern |

### Profile / analyze commands

```bash
# SPP (default)
./src/rocprof-compute profile -n ${name}_${TAG}_spp --overwrite -- ./app …

# Legacy packing
ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC=1 \
  ./src/rocprof-compute profile -n ${name}_${TAG}_legacy --overwrite -- ./app …

# Analyze (both)
./src/rocprof-compute analyze -p workloads/<dir> -k 0 --view table \
  > workloads/logs/<arch>/${name}_${mode}.log
```

- **Full profile:** no block filter; **roofline on** (omit `--no-roof`).
- **Native tool:** default path (omit `--no-native-tool`). If cmake/native build
  fails, document the host workaround; only then fall back to `--no-native-tool`.

---

## Phase D — Reports after scp

For each arch:

**(a) Failure summary** — profile/analyze OK|FAIL, AQL 4096 / err 38, SIGSEGV,
rocpd schema, native-tool build failures, duration.

**(b) SPP vs legacy metric deltas** — generator `--baseline` = legacy logs,
primary = SPP; Old/New/unit table for changed metrics; note pass-count from
`tools/eval_single_pass_packable.py --arch <arch>`.

**(c) SPP metric health HTML** — Good / >100% / N/A / zero_* buckets; refine
arch-specific overflow allowlists (intentional dual-issue, etc.).

---

## Phase E — Rollup

Cross-arch `FAILURE_SUMMARY.md` + short narrative: packing still within HW
limits; health deltas attributable to SPP vs noise/env.

Then schedule **CPX MI300** as a separate validation ticket/run.

---

## Execution status

| Step | Status |
|------|--------|
| Plan doc (this file) | Done |
| Port `generate_metric_health_report.py` | Done |
| MI300 mega_kernel smoke | **Done** (60/60 PASS, 17 bypassed) — 2026-09-30 |
| MI300 full+roof SPP+legacy profiles | **In progress** (`TAG=260930-gfx942` on darkstar) |
| MI300 health + delta reports | Pending (after `ALL_DONE`) |
| MI350 → gfx1151 → MI250 → (MI100?) | Pending |
| CPX MI300 | Reserved |

### MI300 run notes

- Host: `hpe-darkstar-ccs-aus-e12-03` + TheRock `7.15.0a20260728`
- Log: `/home/AMD/feizheng/aiprofcomp78/spp_health_260930-gfx942.log`
- Worker: `scripts/run_spp_health_gfx942.sh`
- Local driver: `scripts/run_spp_health_gfx942_local_driver.sh --wait --fetch --report`
- `cmake` via `/tmp/rpc_venv` (pip) so native-tool path can build; if cmake absent script falls back to `--no-native-tool`
- Profile: full blocks + roofline (`--overwrite`, no `--no-roof`)

### Health report Total / L2 channels (gfx942)

- **Total** in the HTML Summary is analyze-table row count (`--view table`), not
  the ~408 YAML defs. Block 18 expands with `$total_l2_chan`:
  SPX `16×8=128` → `387 + 1 + 8×128 = 1412`. On a single-XCD (CPX) die,
  `total_l2_chan=16` → Total ≈ **516**.
- Conductor CPX die (`cu_per_gpu=38`, `se_per_gpu=4`) must not keep SPX
  `num_xcd=8` / `total_l2_chan=128` — that yields 896 N/A channel rows
  (channels 16–127 × 8 tables). Root cause is partition mislabel / visibility,
  not mega_kernel/vcopy batch size: profiler only returns `TCC_*[0]…[15]`.
- Fix: `MachineSpecsCDNA` down-corrects to CPX when amd-smi full-chip CUs are
  `num_xcd`× rocminfo CUs **or** when `se_per_gpu % num_xcd != 0`. Analyze also
  applies `parser.reconcile_sysinfo_l2_channels()` so existing workloads re-analyze
  cleanly without inventing channel data.
- Regen report (after logs exist):

```bash
python3 tools/generate_metric_health_report.py --arch gfx942 \
  --out validation-artifacts/spp-health-<TAG>/reports/gfx942_health_spp.html \
  --baseline-label "legacy heuristic" \
  mega_kernel:…/logs/gfx942/mega_kernel_spp.log \
  vcopy:…/logs/gfx942/vcopy_spp.log \
  nbody:…/logs/gfx942/nbody_spp.log \
  --baseline mega_kernel:…/legacy.log vcopy:…/legacy.log nbody:…/legacy.log
# or: scripts/run_spp_health_gfx942_local_driver.sh --report
```
