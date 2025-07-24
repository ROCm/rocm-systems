# Changelog for ROCm Systems Profiler

Full documentation for ROCm Systems Profiler is available at [https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/).

## ROCm Systems Profiler 1.1.0 for ROCm 7.0

### Added

- Profiling and metric collection capabilities for VCN engine activity, JPEG engine activity, and API tracing for rocDecode, rocJPEG, and VA-APIs.
- How-to document for VCN and JPEG activity sampling and tracing.
- Support for tracing Fortran applications.
- Support for tracing MPI API in Fortran.
- By default, group "kernel dispatch" and "memory copy" events by HIP stream ID in Perfetto traces.
  - Add the "ROCPROFSYS_ROCM_GROUP_BY_QUEUE" configuration setting to group events by queue, instead.

### Changed

- Replaced ROCm SMI backend with AMD SMI backend for collecting GPU metrics.
- ROCprofiler-SDK is now used to trace RCCL API and collect communication counters.
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
  at https://ui.perfetto.dev/v46.0-35b3d9845/#!/.
