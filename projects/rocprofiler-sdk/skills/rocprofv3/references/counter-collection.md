# Hardware counter collection with rocprofv3

Contents:

1. [Interpreting the counters](#interpreting-the-counters)
2. [Counter availability by architecture](#counter-availability-by-architecture)
3. [What the counters mean](#what-the-counters-mean)
4. [Planning passes](#planning-passes)
5. [Kernel filtering](#kernel-filtering)
6. [Input files](#input-files)
7. [Kernel replay (beta)](#kernel-replay-beta)
8. [Counter group rotation](#counter-group-rotation)
9. [Extra counter definitions](#extra-counter-definitions)
10. [Output layout and SQL](#output-layout-and-sql)
11. [SPM sampling (beta)](#spm-sampling-beta)
12. [Peak DRAM bandwidth](#peak-dram-bandwidth)
13. [Troubleshooting](#troubleshooting)

Sources: rocprofiler-sdk counter definitions (`share/rocprofiler-sdk/config.yaml`), the
rocprofiler-sdk how-to guides, PerfXpert's knowledge base (`pmc_limits.yaml`,
`rocprofv3_counter_limits.yaml`, `metric_thresholds.yaml`, `memory_hierarchy.yaml`,
`vgpr_occupancy_tables.yaml`, `gpu_specs.yaml`), and runs on an AMD Instinct MI325X (gfx942,
rocprofv3 1.4.1).

## Interpreting the counters

`scripts/analyze_counters.py` reads `.db` files, `*counter_collection*.csv` files, or
directories (it merges every `pass_N/`). Options: `--kernel NAME`, `--top N`,
`--peak-hbm-tbps X`, `-o report.md`.

| Signal | Meaning | Advice |
| --- | --- | --- |
| DRAM bandwidth at 70% of peak or more | Memory-bandwidth-bound | Move fewer bytes: fuse kernels, reuse data in LDS/registers, use smaller types. Faster math will not help. |
| DRAM bytes per VMEM wave-instruction above wave size x 16 B (1024 B on wave64) | Over-fetch: strided or scattered access, each lane pulls its own cache line | Make consecutive lanes touch consecutive addresses (layout change, transpose through LDS, swap loop order). 256 B is the coalesced ideal for 4-byte elements on wave64. |
| `FETCH_SIZE` much larger than the bytes the algorithm must read | Redundant traffic or cache thrashing | Compute the ideal from the source (elements x size) and compare. |
| L2 hit rate under 50% | Working set exceeds L2 or poor locality | Tile or block for reuse. Pure streaming kernels miss by design. |
| `OccupancyPercent` under 25% (with low bandwidth and low VALU) | Latency-bound: too few waves | Reduce VGPR/AGPR or LDS per workgroup, launch more workgroups, check `__launch_bounds__`. Very short kernels read low because of ramp-up. |
| `VALUBusy` high, low DRAM bandwidth | Compute-bound on vector ALU | Cut instruction count (strength reduction, fast intrinsics, lower precision) or use MFMA. |
| `MfmaUtil` 50% or more | Matrix-core-bound | Lower precision formats or less matrix work. |
| `LDSBankConflict` 10% or more | LDS bank conflicts | Pad shared arrays (`[N][N+1]`) or swizzle indices. |
| Scratch above 0 | Register spills or stack arrays | Reduce register pressure, avoid dynamically indexed local arrays. |

Caveats to state when relevant:

- Counter collection serializes kernels and brackets each dispatch, so `GRBM_GUI_ACTIVE` equals
  `GRBM_COUNT` per dispatch. Do not report it as "GPU utilization"; take utilization from a trace.
- Durations come from the counter run and are slightly inflated; use a trace for timing.
- Some SDK-derived percentages exceed 100% on multi-XCD GPUs (for example `VALUBusy` on MI300).
  Compare them between kernels, not against 100%.
- Counters are aggregates over the whole dispatch. For which source line causes the problem, follow
  up with PC sampling or thread trace.

Register use caps occupancy. On CDNA2 to CDNA4 (gfx90a, gfx942, gfx950) VGPRs and AGPRs share one
512-entry file per SIMD lane, so add the kernel's VGPR and AGPR counts (both in the report) and
read the waves per SIMD (the hardware maximum is 8, that is 32 per CU):

| VGPR + AGPR per thread | 64 or fewer | 80 | 96 | 128 | 160 | 256 |
| --- | --- | --- | --- | --- | --- | --- |
| Waves per SIMD | 8 | 6 | 5 | 4 | 3 | 2 |

LDS per workgroup and the workgroup size cap occupancy the same way. If the register count is
the cap and occupancy is the limiter, reduce live values or set `__launch_bounds__`.

Report, per hot kernel: the limiter, the evidence (two or three numbers), the code change, and the
expected effect. If the data is inconclusive, say which counter group to add.

## Counter availability by architecture

"Y" means the SDK defines the counter for that target. Always confirm on the machine with
`rocprofv3-avail -d 0 list --pmc`.

| Counter | gfx90a | gfx942 | gfx950 | gfx1100 | gfx1151 | gfx1201 |
| --- | --- | --- | --- | --- | --- | --- |
| GRBM_COUNT, GRBM_GUI_ACTIVE, SQ_WAVES, SQ_BUSY_CYCLES, SQ_WAVE_CYCLES | Y | Y | Y | Y | Y | Y |
| SQ_INSTS_VALU, SQ_INSTS_SALU, SQ_INSTS_LDS, SQ_INSTS_SMEM | Y | Y | Y | Y | Y | Y |
| FETCH_SIZE | Y | Y | Y | Y | Y | Y |
| WRITE_SIZE | Y | Y | Y | Y | Y | - |
| OccupancyPercent, MeanOccupancyPerActiveCU, LDSBankConflict | Y | Y | Y | Y | Y | Y |
| TCC_HIT_sum, TCC_MISS_sum (CDNA L2) | Y | Y | Y | - | - | - |
| GL2C_HIT_sum, GL2C_MISS_sum (RDNA L2) | - | - | - | Y | Y | Y |
| SQ_INSTS_VMEM_RD, SQ_INSTS_VMEM_WR | Y | Y | Y | - | - | - |
| VALUBusy | Y | Y | Y | Y | - | Y |
| MemUnitStalled, MfmaUtil, SQ_ACCUM_PREV_HIRES | Y | Y | Y | - | - | - |
| MemUnitBusy, L2CacheHit, GPUBusy | Y | - | - | Y | Y | Y |

## What the counters mean

Raw counters are summed over every hardware instance (all XCDs, shader engines, and L2 channels)
for one dispatch.

| Counter | Meaning |
| --- | --- |
| `SQ_WAVES` | Waves launched. Equals grid size / wave size for 1D launches; a cheap sanity check. |
| `SQ_INSTS_VALU`, `SQ_INSTS_SALU`, `SQ_INSTS_LDS`, `SQ_INSTS_SMEM` | Instructions issued per class, per wave when divided by `SQ_WAVES`. |
| `SQ_INSTS_VMEM_RD`, `SQ_INSTS_VMEM_WR` | Vector memory instructions (global, buffer, flat, scratch). One instruction moves wave size x bytes per lane. |
| `SQ_BUSY_CYCLES`, `SQ_WAVE_CYCLES` | Cycles with active waves; wave-cycles (quad cycles) summed over waves. Inputs to occupancy. |
| `GRBM_GUI_ACTIVE`, `GRBM_COUNT` | GPU-active and total cycles. In counter mode each dispatch is bracketed, so they are equal; use them as a cycle count, not utilization. |
| `TCC_HIT_sum`, `TCC_MISS_sum` | L2 hits and misses over all channels (uncached reads count as misses). |
| `FETCH_SIZE` | KiB read from DRAM (HBM), including over-fetch. |
| `WRITE_SIZE` | KiB written to DRAM. Not defined on RDNA4 (gfx1200, gfx1201); there only read bandwidth is available. |
| `OccupancyPercent` | Average resident waves as % of the hardware maximum (32 waves/CU on CDNA3). |
| `MeanOccupancyPerActiveCU` | Average resident waves per CU while the CU is busy. |
| `VALUBusy` | % of GPU time vector ALUs issue. Can exceed 100% on multi-XCD parts; compare between kernels. |
| `MfmaUtil` | % of time matrix cores are busy. |
| `MemUnitStalled` | % of GPU time the vector memory unit is stalled by the cache. |
| `LDSBankConflict` | % of LDS active cycles lost to bank conflicts. |

Look up the exact formula of any derived metric with `rocprofv3-avail -d 0 info --pmc`.

Derived numbers the analysis script computes:

- L2 hit rate = hits / (hits + misses).
- DRAM bandwidth = `FETCH_SIZE` x 1024 / duration + `WRITE_SIZE` x 1024 / duration, each using the
  durations of its own pass. Bytes per nanosecond equals GB/s.
- DRAM bytes per VMEM wave-instruction = (FETCH + WRITE bytes) / (`SQ_INSTS_VMEM_RD` + `SQ_INSTS_VMEM_WR`).
  Coalesced 4-byte accesses on wave64 with no reuse give 256 B; float4 gives 1024 B. Above
  wave size x 16 B the access pattern over-fetches. Far below 256 B means caches serve most bytes.

Measured example on MI325X (16M floats): a coalesced copy showed 256 B per instruction, 33% L2 hit,
63% of peak bandwidth; a copy reading with stride 33 showed 4224 B per instruction, 3% L2 hit,
2 GiB fetched for 64 MiB of useful data, and 86% of peak bandwidth spent mostly on waste.

## Planning passes

- One `--pmc` group is collected per application run. Multiple `--pmc` flags make multiple runs,
  each in its own `pass_N` directory. Combining `-i input.txt` with `--pmc` adds the CLI groups as
  extra passes.
- Per-pass hardware limits on gfx9 (CDNA): SQ 8, TCP 4, TCC 4, SPI 6, TA 2, TD 2, CPC 2, CPF 2,
  GRBM 2, GDS 4 counters. RDNA3.5 (gfx1151): SQ 8, TCP 4, SPI 6, GL1A/GL1C/GL2A/GL2C 4, TA 2,
  GRBM 2, CPC 2, GCEA 2.
- Derived metrics expand to their raw inputs: `FETCH_SIZE` uses three TCC counters and
  `WRITE_SIZE` two, which together exceed the TCC limit. Give each of them its own pass with no
  other counters (the isolation rule PerfXpert's pass planner enforces); some combinations
  pass `pmc-check` on gfx942, but isolation is safe on every architecture. Keep other derived
  metrics in small groups and confirm with `rocprofv3-avail -d <gpu> pmc-check C1 C2 ...`. Use
  `C:device=1` to check a counter on another GPU.
- Counters with dimensions (for example per-channel `TCC_HIT[0:15]`) are collected as the sum.
  Bracket notation is not accepted; use JSON output for per-instance values.
- Error 38, "Request exceeds the capabilities of the hardware to collect", means a group does
  not fit; split it.

## Kernel filtering

- `--kernel-include-regex REGEX` and `--kernel-exclude-regex REGEX` are searched (substring
  match) in the formatted kernel name: demangled by default, mangled with `-M`, truncated with `-T`.
  Exclude is applied after include.
- `--kernel-iteration-range` selects instances of each matching kernel, counted from 1:
  `2`, `2-4`, `"[1-2],[5-8]"`, or `"[1, 3, [5-8]]"`. Without it every instance is collected.
- Filters apply to counter collection and thread trace only, not to tracing.

## Input files

Text (`-i input.txt`), one pass per line:

```text
pmc: GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES
pmc: FETCH_SIZE
```

JSON (`-i input.json`; YAML works too but needs `pyyaml`). Each job is a separate run with its own
output and filters:

```json
{
  "jobs": [
    { "pmc": ["SQ_WAVES", "TCC_HIT_sum", "TCC_MISS_sum"], "kernel_include_regex": "gemm",
      "kernel_iteration_range": "[2-4]", "output_directory": "pmc_out", "output_file": "run" },
    { "pmc": ["FETCH_SIZE"], "kernel_include_regex": "gemm", "output_directory": "pmc_out",
      "output_file": "run" }
  ]
}
```

Other job keys mirror CLI options with underscores (`output_format`, `truncate_kernels`,
`mangled_kernels`, `kernel_exclude_regex`, `log_level`, `preload`).

## Kernel replay (beta)

`--replay-mode kernel --kernel-replay-beta-enabled` collects every `--pmc` group in one
application run by replaying each targeted dispatch once per group, snapshotting and restoring
device memory between passes.

- Use it when application replay is slow or non-deterministic.
- Each group must still fit in one hardware pass; the number of passes is the number of groups.
- Only coarse-grained device allocations and module-scope `__device__`/`__constant__` variables
  are restored. Unified, managed, `hipMallocAsync`, host, and fine-grained memory are not.
- HIP graph launches and multi-packet submissions run once without replay (with a warning).
- Host RAM use scales with the agent's whole tracked device footprint.
- Cannot be combined with `--att`, PC sampling, or `--spm`. No MPI coordination.
- Output is one database (no `pass_N`). JSON adds `replay_pass`; CSV has no pass column, so give
  each group distinct counters.

## Counter group rotation

For one run that samples different groups on successive dispatches (different dispatches, not
the same one), use an input file:

```json
{ "jobs": [ { "pmc_groups": [["SQ_WAVES", "GRBM_COUNT"], ["FETCH_SIZE"]], "pmc_group_interval": 4 } ] }
```

The interval counts dispatches per device. `pmc_groups` cannot be combined with `--pmc`.

## Extra counter definitions

`-E extra.yaml --pmc MY_COUNTER` adds user-defined counters:

```yaml
rocprofiler-sdk:
  counters-schema-version: 1
  counters:
    - name: GRBM_GUI_ACTIVE_SUM
      description: "Unit: cycles"
      properties: []
      definitions:
        - architectures: [gfx942, gfx90a]
          expression: reduce(GRBM_GUI_ACTIVE,max)*CU_NUM
```

## Output layout and SQL

- Multi-pass runs write `DIR/pass_1/NAME_results.db`, `DIR/pass_2/...`. A single group writes
  `DIR/NAME_results.db` directly.
- The `counters_collection` view has one row per dispatch and counter: `dispatch_id`,
  `kernel_name`, `counter_name`, `value`, `duration`, `start`, `end`, `grid_size`,
  `workgroup_size`, `lds_block_size`, `scratch_size`, `vgpr_count`, `accum_vgpr_count`,
  `sgpr_count`, `agent_abs_index`, `is_derived`, `expression`.

```sql
SELECT kernel_name, counter_name, COUNT(*) AS dispatches, AVG(value) AS mean_value
FROM counters_collection GROUP BY kernel_name, counter_name ORDER BY kernel_name;
```

- `rocpd convert -i DB -f csv` (needs pandas) writes `<out>_counter_collection_trace.csv` with
  columns `Dispatch_Id, Kernel_Name, Counter_Name, Counter_Value, Start_Timestamp,
  End_Timestamp, Vgpr_Count, ...`. Direct `--output-format csv` writes
  `pass_N/<prefix>_counter_collection.csv` with the same information. The analysis script reads both.
- `--output-format json` adds per-dimension values and, with kernel replay, `replay_pass`.

## SPM sampling (beta)

Streaming Performance Monitor samples counters over time within a kernel instead of per dispatch.
It needs amdgpu driver 6.19.14 or later.

```bash
rocprofv3-avail -d 0 list --spm          # counters that support SPM
rocprofv3 --spm-beta-enabled --spm SQ_WAVES --spm-sample-interval 1200 \
  --spm-sample-interval-unit sclk_cycles --output-format json -- ./app
```

SPM cannot be combined with `--pmc` or PC sampling, and all SPM counters must fit in one pass.
Samples land in the `spm_counters` view.

## Peak DRAM bandwidth

Used by the script for "% of peak" (decimal TB/s, AMD Instinct spec sheets; also in PerfXpert's
`gpu_specs.yaml`). Pass `--peak-hbm-tbps` for other GPUs.

| GPU | TB/s |
| --- | --- |
| MI100 | 1.23 |
| MI210 | 1.6 |
| MI250, MI250X | 3.2 |
| MI300A, MI300X | 5.3 |
| MI325X | 6.0 |
| MI350X, MI355X | 8.0 |

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Request exceeds the capabilities of the hardware to collect` (error 38) or `not collected on agent` | Split the group; check with `pmc-check`; never put `FETCH_SIZE` and `WRITE_SIZE` together. |
| `Invalid counter name` | Not defined on this architecture; look it up with `rocprofv3-avail -d 0 list --pmc`. |
| No `pass_N` output or empty report | The kernel filter matched nothing (the regex is searched in the demangled name, or the mangled/truncated name with `-M`/`-T`), or the app failed in a later pass. |
| All counters zero on Radeon | Set the performance level to `STABLE_STD`. |
| Counter values vary between passes | The workload is not deterministic; fix the input or use `--replay-mode kernel --kernel-replay-beta-enabled`. |
| `--pmc` rejected together with `--spm`, `--att-activity`, or `--att-perfcounters` | Collect those in a separate run. Kernel replay also refuses `--att`, PC sampling, and `--spm`. |
| Multi-pass with `--pid` or `--collection-period` rejected | Not supported; use a launched run, or `pmc_groups` rotation in an input file. |
| YAML input file fails to load | `pip install pyyaml`, or use a JSON input file. |
