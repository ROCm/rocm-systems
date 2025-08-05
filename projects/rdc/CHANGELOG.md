# Change Log for RDC

Full documentation for RDC is available at [ROCm DataCenter Tool User Guide](https://rocm.docs.amd.com/projects/rdc/en/latest/).

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
