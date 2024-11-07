# Changelog for ROCm Systems Profiler

Full documentation for ROCm Systems Profiler is available at [https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/).

## ROCm Systems Profiler 0.1.0 for ROCm 6.3.0

### Changed

- Renamed Omnitrace to ROCm Systems Profiler.

## Omnitrace 1.11.2 for ROCm 6.2.1

### Known issues

- Perfetto can no longer open Omnitrace proto files. Loading the Perfetto trace output `.proto` file in `ui.perfetto.dev` can
  result in a dialog with the message, "Oops, something went wrong! Please file a bug." The information in the dialog will
  refer to an "Unknown field type." The workaround is to open the files with the previous version of the Perfetto UI found
  at https://ui.perfetto.dev/v46.0-35b3d9845/#!/.
