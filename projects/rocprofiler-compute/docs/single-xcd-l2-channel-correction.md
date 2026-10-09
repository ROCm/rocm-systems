# Single-XCD L2 channel correction

Problem statement for a workaround that stays off the single-pass packable
design. Health test report generation still needs it. The proper fix is in
amd-smi.

## Problem

`ROCR_VISIBLE_DEVICES` can isolate one XCD while amd-smi still reports SPX and
the full-chip CU count. rocprof-compute reads `compute_partition` and
`num_compute_units` from amd-smi, looks up `num_xcd` from that partition, and
saves `num_xcd` and `total_l2_chan` for the whole chip.

Block 18 then expands past the TCC channels that were actually collected. On an
MI300X single-XCD profile (`cu_per_gpu=38`, `se_per_gpu=4`) that shows up as
N/A for channels 16–127.

The same inflation is in a saved workload. Re-analyze and the health test
report median recompute read `sysinfo.csv`. If that file still says SPX with
`num_xcd=8`, L2 channel expansion uses the full-chip count.

rocminfo already reports the visible `cu_per_gpu` and `se_per_gpu`. amd-smi
does not report the visible partition.

## Not “already in CPX”

The check does not require `compute_partition` to already be CPX. It detects a
one-XCD visible set while `num_xcd > 1`, then writes `num_xcd=1` and
`compute_partition=CPX`.

It runs only when CPX mode for that part is one XCD: MI300 (gfx940, gfx941,
gfx942), MI350 (gfx950), and gfx1250 on the profile path. Re-analyze also
requires `gfx9`, so gfx1250 is profile-only. Parts whose `num_xcd` is already 1
never reach it.

Either signal is enough after that gate:

- The amd-smi CU count is greater than the visible rocminfo CUs, divides them
  evenly, and the quotient equals `num_xcd`.
- `se_per_gpu` is set and is not divisible by `num_xcd`. A real multi-XCD SPX
  reports an aggregate shader-engine count. A one-XCD die mislabeled as SPX
  does not.

Re-analyze has no amd-smi CU count, so only the shader-engine check can fire
there. It then rewrites `total_l2_chan` through `totall2_banks(..., "CPX")`.

## Proper fix

amd-smi should report the partition and CU count visible to the process. If it
did, `num_xcd` would already be 1 and this rewrite would not run.

## Why the health test report still uses it

Single-XCD health workloads are profiled with `ROCR_VISIBLE_DEVICES` and then
re-analyzed for the health test report. Median recompute in
`tools/compare_spp_legacy_medians.py` loads `sysinfo.csv` through
`reconcile_sysinfo_l2_channels`. Without that, L2 metrics in the report are
N/A or compared on the wrong channel count.

Profile applies the same correction in
`MachineSpecsCDNA.finalize_soc_fields`, before `total_l2_chan` is derived, so a
new single-XCD profile stores the down-corrected counts. Post-analysis applies
it in `OmniAnalyze_Base.initalize_runs` while loading `sysinfo.csv`.

## Scope

This is outside the single-pass packable and unpackable design. Pull requests
[#12709](https://github.com/ROCm/rocm-systems/pull/12709),
[#12830](https://github.com/ROCm/rocm-systems/pull/12830),
[#12712](https://github.com/ROCm/rocm-systems/pull/12712), and
[#12848](https://github.com/ROCm/rocm-systems/pull/12848) leave `num_xcd` and
`total_l2_chan` as amd-smi and the saved sysinfo report them.

This branch is based on the health test report branch and keeps the workaround
for report generation only:

- Profile: `MachineSpecsCDNA._reconcile_num_xcd_with_visible_cus`
- Post-analysis: `reconcile_sysinfo_l2_channels` in `parser.py`
- Health test report: `tools/compare_spp_legacy_medians.py`
