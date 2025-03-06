# Changelog for ROCm Systems Profiler

Full documentation for ROCm Systems Profiler is available at [https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/).

## ROCm Systems Profiler 1.1.0 for ROCm 6.5

### Added

- Added profiling and metric collection capabilities for VCN engine activity, JPEG engine activity and API tracing for rocDecode, rocJPEG and VA-APIs.

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
