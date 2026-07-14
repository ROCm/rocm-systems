# Change Log for RDC

Full documentation for RDC is available at [ROCm DataCenter Tool User Guide](https://rocm.docs.amd.com/projects/rdc/en/latest/).

## RDC 1.3.1 for ROCm 7.14.0

### Added

- **Added 59 new telemetry fields to close the gap with Device Metrics Exporter (DME)**.
  - Energy: `RDC_FI_GPU_ENERGY` — total energy consumed via `amdsmi_get_energy_count()`.
  - Temperature: `RDC_FI_GPU_JUNCTION_TEMP` — dedicated junction/hotspot temperature field.
  - Clock ranges: `RDC_FI_GPU_CLOCK_MIN`, `RDC_FI_GPU_CLOCK_MAX` — min/max GPU clock frequencies. Additional clock types: `RDC_FI_SOC_CLOCK`, `RDC_FI_VCLK0`, `RDC_FI_DCLK0`.
  - Memory: `RDC_FI_GPU_MEMORY_FREE` (free VRAM), visible VRAM (`RDC_FI_GPU_VIS_VRAM_TOTAL/USED/FREE`), GTT memory (`RDC_FI_GPU_GTT_TOTAL/USED/FREE`).
  - PCIe: `RDC_FI_PCIE_SPEED`, `RDC_FI_PCIE_MAX_SPEED`, `RDC_FI_PCIE_REPLAY_ROLLOVER`, `RDC_FI_PCIE_BANDWIDTH_BIDIR` with sentinel value handling for APU platforms.
  - Instantaneous activity: `RDC_FI_GPU_GFX_BUSY_INST`, `RDC_FI_GPU_VCN_BUSY_INST`, `RDC_FI_GPU_JPEG_BUSY_INST` from `gpu_metrics.xcp_stats`.
  - ECC deferred errors: 19 per-block deferred error fields (`RDC_FI_ECC_*_DE`) plus `RDC_FI_ECC_DEFERRED_TOTAL`, reading `deferred_count` from `amdsmi_error_count_t`.
  - Violation/throttle metrics: 19 new `RDC_HEALTH_*` fields covering accumulated counts and percentages for processor hot, PPT power, socket/VR/HBM thermal, gfx clock host limits, and low utilization violations via `amdsmi_get_violation_status()`. Driver 1.8 XCP/XCC fields return NOT_SUPPORTED on older platforms.

- **Added automated DME-RDC metric sync check**.
  - New script `tools/dme_rdc_metric_sync_check.py` parses DME's protobuf metric definitions and compares against RDC field enums via a curated mapping file (`tools/dme_rdc_metric_mapping.json`).
  - New GitHub Action (`.github/workflows/rdc-dme-sync-check.yml`) runs weekly and on PRs touching metric definitions. Automatically creates GitHub issues when DME adds metrics not yet tracked in RDC.

### Changed

- Bumped gRPC from 1.67.1 to 1.78.1. See [ROCm/TheRock#4172](https://github.com/ROCm/TheRock/pull/4172).

### Removed

- Removed RVS integration. [RVS](https://github.com/ROCm/ROCmValidationSuite) is built independently of RDC and TheRock, so its integration has been disabled.
  - `BUILD_RVS` now defaults to `OFF` (#7116).

### Resolved Issues

- The `Failed to insert module: N3amd3rdc10RdcRVSLibE` error no longer occurs.

## RDC 1.3.0 for ROCm 7.13.0

### Added

- **Added GFX and memory accumulated activity metrics**.
  - New fields `RDC_FI_GFX_ACTIVITY_ACC` (509), `RDC_FI_MEM_ACTIVITY_ACC` (510), and `RDC_FI_ACCUMULATION_COUNTER` (511) expose the accumulated GFX/memory activity counters and the accumulation cycle counter from amdsmi gpu_metrics. Use `accumulation_counter` as the normalization denominator to compute utilization: `(activity_acc_n - activity_acc_n-1) / (accumulation_counter_n - accumulation_counter_n-1) * 100`.

### Resolved Issues

- **Fixed broken partition metrics**.  
  - Regardless if GPU was partitioned, RDC only saw the GPU index and no instances due to upstream gpu_metrics changes

## RDC 1.2.0 for ROCm 7.1.0

### Added

- CPU monitoring support with 30+ CPU field definitions through AMD SMI integration.
- CPU partition format support (c0.0, c1.0) for monitoring AMD EPYC processors.
- Mixed GPU/CPU monitoring in single `rdci dmon` command.

### Optimized

- Improved profiler metrics path detection for counter definitions.

### Resolved issues

- Group management issues with listing created/non-created groups.
- ECC_UNCORRECT field behavior.

## RDC for ROCm 7.0.0

### Added

- More profiling and monitoring metrics, especially for AMD Instinct MI300 and newer GPUs.
- Advanced logging and debugging options, including new log levels and troubleshooting guidance.

### Changed

- Completed migration from legacy [ROCProfiler](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/) to [ROCprofiler-SDK](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/).
- Reorganized the configuration files internally and improved [README/installation](https://github.com/ROCm/rocm-systems/blob/develop/projects/rdc/README.md) instructions.
- Updated metrics and monitoring support for the latest AMD data center GPUs.

### Optimized

- Integration with [ROCprofiler-SDK](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/) for performance metrics collection.
- Standalone and embedded operating modes, including streamlined authentication and configuration options.
- Support and documentation for diagnostic commands and GPU group management.
- [RVS](https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/latest/) test integration and reporting.
## RDC for ROCm 6.4.0

### Added

- RDC policy feature
- Power and thermal throttling metrics
- RVS [IET](https://github.com/ROCm/ROCmValidationSuite/tree/a6177fc5e3f2679f98bbbc80dc536d535a43fb69/iet.so), [PEBB](https://github.com/ROCm/ROCmValidationSuite/tree/a6177fc5e3f2679f98bbbc80dc536d535a43fb69/pebb.so), and [memory bandwidth tests](https://github.com/ROCm/ROCmValidationSuite/tree/a6177fc5e3f2679f98bbbc80dc536d535a43fb69/babel.so)
- Link status
- RDC_FI_PROF_SM_ACTIVE metric

### Changed

- Migrated from [rocprofiler v1](https://github.com/ROCm/rocprofiler) to [rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk)
- Improved README.md for better usability
- Moved `rdc_options` into `share/rdc/conf/`
- Fixed ABSL in clang18+

## RDC for ROCm 6.3.0

### Added

- [RVS](https://github.com/ROCm/ROCmValidationSuite) integration
- Real time logging for diagnostic command
- `--version` command
- `XGMI_TOTAL_READ_KB` and `XGMI_TOTAL_WRITE_KB` monitoring metrics

## RDC for ROCm 6.2.0

- Added [rocprofiler](https://github.com/ROCm/rocprofiler) dmon metrics
- Added new ECC metrics
- Added [ROCmValidationSuite](https://github.com/ROCm/ROCmValidationSuite) diagnostic command
- Fully migrated to [AMDSMI](https://github.com/ROCm/amdsmi)
  - Removed RASLIB dependency and blobs
  - Removed [rocm_smi_lib](https://github.com/ROCm/rocm_smi_lib) dependency

## RDC for ROCm 6.1.0

- Added `--address` flag to rdcd
- Upgraded from C++11 to C++17
- Upgraded gRPC

## RDC for ROCm 5.5.0

- Added new profiling metrics for RDC dmon module.
