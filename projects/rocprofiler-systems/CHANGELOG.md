<!-- markdownlint-disable MD024 -->

# Changelog for ROCm Systems Profiler

Full documentation for ROCm Systems Profiler is available at [https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/).

## ROCm Systems Profiler 1.5.0 for ROCm x.y.z (unreleased)

### Changed

- Simplify categorizing like pmc_info events by removing the _<idx> from the "symbol" field. ie., "JpegAct_0" -> "JpegAct".
- Added `libhsa-runtime64.so` and `libomp.so` to the internal library exclusion list for runtime instrumentation to prevent instrumenting of runtime library internals.
- RCCL implementation refactored with `production_pmc_registrar` for improved testability and separation of concerns.
- Unsupported RCCL datatypes now gracefully return 0 with `LOG_WARNING` instead of aborting profiler, allowing continued profiling with newer RCCL versions.

### Added

- Per-GPU RCCL communication data counters (Send/Recv) in `rocpd` output with multi-GPU device attribution using `ncclCommCuDevice` API.
- Presets profiles that configure the rocprofiler-system tools for common profiling scenarios, offering optimized configurations for specific use cases.

### Resolved issues

- Fixed an issue where JPEG engine activity PMC events were not being collected for MI35X systems. Only the first 32 JPEG engines were being collected.

### Resolved issues

- Fixed MPI perfetto trace file merging when using trace cache mode with `ROCPROFSYS_PERFETTO_COMBINE_TRACES=ON`. Previously, each MPI rank would produce a separate trace file; now all ranks' traces are correctly merged into a single output file.

## ROCm Systems Profiler 1.4.0 for ROCm 7.11.0

### Added

- Support for UCX (Unified Communication X) API tracing.
- Profiling and metric collection capabilities for XGMI and PCIe data.
- How-to document for XGMI and PCIe sampling and monitoring.
- Documentation for `--trace-legacy` / `-L` CLI flag for direct tracing mode.
- Added dependency to `spdlog` library.
- Added environment variable `ROCPROFSYS_LOG_LEVEL` which control level of logging.
  - Available log levels: `critical`, `error`, `warning`, `info`(default), `debug`, `trace` and `off`.
- Added cmake option `ROCPROFSYS_GFX_TARGETS` which controls GFX targets used to build example binaries.

### Changed

- `ROCPROFSYS_TRACE` now controls whether perfetto tracing is enabled (default: true when tracing mode).
- `ROCPROFSYS_TRACE_LEGACY` controls whether to use legacy direct mode (true) or cached mode (false, default).
- By default, tracing uses deferred trace generation (cached mode) for improved performance and minimal runtime overhead.
- `--trace` / `-T` CLI flag enables tracing with cached mode by default.
- `--trace-legacy` / `-L` CLI flag enables legacy direct mode for tracing.
- Changed thread storage allocation from a hard-coded 4096-element array to a compile-time computed size derived from the ROCPROFSYS_MAX_THREADS configuration flag.
- Changed logging module to use `spdlog` library.

### Resolved issues

- Fixed application termination with segfault when thread creation surpasses ROCPROFSYS_MAX_THREADS configuration.
- Fixed how `roctxRange` markers are handled in the `rocpd` output. The "push" and "pop" markers are now shown as a single event.

### Removed

- `ROCPROFSYS_TRACE_CACHED` environment variable (tracing now uses cached mode by default when `ROCPROFSYS_TRACE_LEGACY=false`).

### Deprecated

- `ROCPROFSYS_USE_PERFETTO` environment variable (use `ROCPROFSYS_TRACE`).
- `ROCPROFSYS_VERBOSE` and `ROCPROFSYS_DEBUG` environment variables (use `ROCPROFSYS_LOG_LEVEL`).

## ROCm Systems Profiler 1.3.0 for ROCm 7.2.0

### Added

- Added a `ROCPROFSYS_PERFETTO_FLUSH_PERIOD_MS` configuration setting to set the flush period for Perfetto traces. The default value is 10000 ms (10 seconds).
- Added fetching of the `rocpd` schema from rocprofiler-sdk-rocpd

### Changed

- Improved Fortran main function detection to ensure `rocprof-sys-instrument` uses the Fortran program main function instead of the C wrapper.

### Resolved issues

- Fixed a crash when running `rocprof-sys-python` with ROCPROFSYS_USE_ROCPD enabled.
- Fixed an issue where kernel/memory-copy events could appear on the wrong Perfetto track (e.g., queue track when stream grouping was requested) because _group_by_queue state leaked between records.

## ROCm Systems Profiler 1.2.1 for ROCm 7.1.1

### Resolved issues

- Fixed an issue of OpenMP Tools (OMPT) events, GPU performance counters, VA-API, MPI, and host events failing to be collected in the `rocpd` output.

## ROCm Systems Profiler 1.2.0 for ROCm 7.1.0

### Added

- ``ROCPROFSYS_ROCM_GROUP_BY_QUEUE`` configuration setting to allow grouping of events by hardware queue, instead of the default grouping.
- Support for `rocpd` database output with the `ROCPROFSYS_USE_ROCPD` configuration setting.
- Support for profiling PyTorch workloads using the `rocpd` output database.
- Support for tracing OpenMP API in Fortran applications.
- An error warning that is triggered if the profiler application fails due to SELinux enforcement being enabled. The warning includes steps to disable SELinux enforcement.

### Changed

- Updated the grouping of "kernel dispatch" and "memory copy" events in Perfetto traces. They are now grouped together by HIP Stream rather than separately and by hardware queue.
- Updated PAPI module to v7.2.0b2.
- ROCprofiler-SDK is now used for tracing OMPT API calls.

## ROCm Systems Profiler 1.1.1 for ROCm 7.0.2

### Resolved issues

- Fixed an issue where ROC-TX ranges were displayed as two separate events instead of a single spanning event.

## ROCm Systems Profiler 1.1.0 for ROCm 7.0

### Added

- Profiling and metric collection capabilities for VCN engine activity, JPEG engine activity, and API tracing for rocDecode, rocJPEG, and VA-APIs.
- How-to document for VCN and JPEG activity sampling and tracing.
- Support for tracing Fortran applications.
- Support for tracing MPI API in Fortran.

### Changed

- Replaced ROCm SMI backend with AMD SMI backend for collecting GPU metrics.
- ROCprofiler-SDK is now used to trace RCCL API and collect communication counters.
  - Use the setting `ROCPROFSYS_USE_RCCLP = ON` to enable profiling and tracing of RCCL application data.
- Updated the Dyninst submodule to v13.0.
- Set the default value of `ROCPROFSYS_SAMPLING_CPUS` to `none`.

### Resolved issues

- Fixed GPU metric collection settings with `ROCPROFSYS_AMD_SMI_METRICS`.
- Fixed a build issue with CMake 4.
- Fixed incorrect kernel names shown for kernel dispatch tracks in Perfetto.
- Fixed formatting of some output logs.
- Fixed an issue where ROC-TX ranges were displayed as two separate events instead of a single spanning event.

## ROCm Systems Profiler 1.0.2 for ROCm 6.4.2

### Optimized

- Improved readability of the OpenMP target offload traces by showing on a single Perfetto track.

### Resolved issues

- Fixed the file path to the script that merges Perfetto files from multi-process MPI runs. The script has also been renamed from `merge-multiprocess-output.sh` to `rocprof-sys-merge-output.sh`.

## ROCm Systems Profiler 1.0.1 for ROCm 6.4.1

### Added

- How-to document for [network performance profiling](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/amd-staging/how-to/nic-profiling.html) for standard Network Interface Cards (NICs).

### Resolved issues

- Fixed a build issue with Dyninst on GCC 13.

## ROCm Systems Profiler 1.0.0 for ROCm 6.4.0

### Added

- Support for VA-API and rocDecode tracing.

- Aggregation of MPI data collected across distributed nodes and ranks. The data is concatenated into a single proto file.

### Changed

- Backend refactored to use ROCprofiler-SDK rather than ROCProfiler and ROCTracer.

### Resolved issues

- Fixed hardware counter summary files not being generated after profiling.

- Fixed an application crash when collecting performance counters with ROCProfiler.

- Fixed interruption in config file generation.

- Fixed segmentation fault while running `rocprof-sys-instrument`.

- Fixed an issue where running `rocprof-sys-causal` or using the `-I all` option with `rocprof-sys-sample` caused the system to become non-responsive.

- Fixed an issue where sampling multi-GPU Python workloads caused the system to stop responding.

## ROCm Systems Profiler 0.1.1 for ROCm 6.3.2

### Resolved issues

- Fixed an error when building from source on some SUSE and RHEL systems when using the `ROCPROFSYS_BUILD_DYNINST` option.

## ROCm Systems Profiler 0.1.0 for ROCm 6.3.1

### Added

- Improvements to support OMPT target offload.

### Resolved issues

- Fixed an issue with generated Perfetto files.

- Fixed an issue with merging multiple `.proto` files.

- Fixed an issue causing GPU resource data to be missing from traces of Instinct MI300A systems.

- Fixed a minor issue for users upgrading to ROCm 6.3 from 6.2 post-rename from `omnitrace`.

## ROCm Systems Profiler 0.1.0 for ROCm 6.3.0

### Changed

- Renamed Omnitrace to ROCm Systems Profiler.

## Omnitrace 1.11.2 for ROCm 6.2.1

### Known issues

- Perfetto can no longer open Omnitrace proto files. Loading the Perfetto trace output `.proto` file in `ui.perfetto.dev` can
  result in a dialog with the message, "Oops, something went wrong! Please file a bug." The information in the dialog will
  refer to an "Unknown field type." The workaround is to open the files with the previous version of the Perfetto UI found
  at <https://ui.perfetto.dev/v46.0-35b3d9845/#!/>.
