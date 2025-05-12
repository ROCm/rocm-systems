# Changelog for ROCprofiler-SDK

Full documentation for ROCprofiler-SDK is available at [rocm.docs.amd.com/projects/rocprofiler-sdk](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/index.html)

## ROCprofiler-SDK for AFAR I

### Added

- HSA API tracing
- Kernel dispatch tracing
- Kernel dispatch counter collection
  - Instances reported as single dimension
  - No serialization

## ROCprofiler-SDK for AFAR II

### Added

- HIP API tracing
- ROCTx tracing
- Tracing ROCProf Tool V3
- Documentation packaging
- ROCTx control (start and stop)
- Memory copy tracing

## ROCprofiler-SDK for AFAR III

### Added

- Kernel dispatch counter collection. This includes serialization and multidimensional instances.
- Kernel serialization.
- Serialization control (on and off).
- ROCprof tool plugin interface V3 for counters and dimensions.
- Support to list metrics.
- Correlation-Id retirement
- HIP and HSA trace distinction:
  - --hip-runtime-trace          For collecting HIP Runtime API traces
  - --hip-compiler-trace         For collecting HIP compiler-generated code traces
  - --hsa-core-trace             For collecting HSA API traces (core API)
  - --hsa-amd-trace              For collecting HSA API traces (AMD-extension API)
  - --hsa-image-trace            For collecting HSA API traces (image-extension API)
  - --hsa-finalizer-trace        For collecting HSA API traces (finalizer-extension API)

## ROCprofiler-SDK for AFAR IV

### Added

**API:**

- Page migration reporting
- Scratch memory reporting
- Kernel dispatch callback tracing
- External correlation Id request service
- Buffered counter collection record headers
- Option to remove HSA dependency from counter collection

**Tool:**

- `rocprofv3` multi-GPU support in a single-process

## ROCprofiler-SDK for AFAR V

### Added

**API:**

- Agent or device counter collection
- PC sampling (beta)

**Tool:**

- Single JSON output format support
- Perfetto output format support (.pftrace)
- Input YAML support for counter collection
- Input JSON support for counter collection
- Application replay in counter collection
- `rocprofv3` multi-GPU support:
  - Multiprocess (multiple files)

### Changed

- `rocprofv3` tool now requires mentioning `--` before the application. For detailed use, see [Using rocprofv3](source/docs/how-to/using-rocprofv3.rst)

### Resolved issues

- Fixed `SQ_ACCUM_PREV` and `SQ_ACCUM_PREV_HIRE` overwriting issue

## ROCprofiler-SDK 0.4.0 for ROCm release 6.2 (AFAR VI)

### Added

- OTF2 tool support
- Kernel and range filtering
- Counter collection definitions in YAML
- Documentation updates (SQ block, counter collection, tracing, tool usage)
- `rocprofv3` option `--kernel-rename`
- `rocprofv3` options for Perfetto settings (buffer size and so on)
- CSV columns for kernel trace
  - `Thread_Id`
  - `Dispatch_Id`
- CSV column for counter collection

## ROCprofiler-SDK 0.5.0 for ROCm release 6.3 (AFAR VII)

### Added

- Start and end timestamp columns to the counter collection csv output
- Check to force tools to initialize context id with zero
- Support to specify hardware counters for collection using rocprofv3 as `rocprofv3 --pmc [COUNTER [COUNTER ...]]`
- Memory Allocation Tracing
- PC sampling tool support with CSV and JSON output formats
- List supported PC Sampling Configurations

### Changed

- `--marker-trace` option for `rocprofv3` now supports the legacy ROCTx library `libroctx64.so` when the application is linked against the new library `librocprofiler-sdk-roctx.so`.
- Replaced deprecated `hipHostMalloc` and `hipHostFree` functions with `hipExtHostAlloc` and `hipFreeHost` for ROCm versions starting 6.3.
- Updated `rocprofv3` `--help` options.
- Changed naming of "agent profiling" to a more descriptive "device counting service". To convert existing tool or user code to the new name, use the following sed:
`find . -type f -exec sed -i 's/rocprofiler_agent_profile_callback_t/rocprofiler_device_counting_service_callback_t/g; s/rocprofiler_configure_agent_profile_counting_service/rocprofiler_configure_device_counting_service/g; s/agent_profile.h/device_counting_service.h/g; s/rocprofiler_sample_agent_profile_counting_service/rocprofiler_sample_device_counting_service/g' {} +`
- Changed naming of "dispatch profiling service" to a more descriptive "dispatch counting service". To convert existing tool or user code to the new names, the following sed can be used: `-type f -exec sed -i -e 's/dispatch_profile_counting_service/dispatch_counting_service/g' -e 's/dispatch_profile.h/dispatch_counting_service.h/g' -e 's/rocprofiler_profile_counting_dispatch_callback_t/rocprofiler_dispatch_counting_service_callback_t/g' -e 's/rocprofiler_profile_counting_dispatch_data_t/rocprofiler_dispatch_counting_service_data_t/g'  -e 's/rocprofiler_profile_counting_dispatch_record_t/rocprofiler_dispatch_counting_service_record_t/g' {} +`
- `FETCH_SIZE` metric on gfx94x now uses `TCC_BUBBLE` for 128B reads.
- PMC dispatch-based counter collection serialization is now per-device instead of being global across all devices.
- Added output return functionality to rocprofiler_sample_device_counting_service
- Added rocprofiler_load_counter_definition.

### Resolved issues

- Create subdirectory when `rocprofv3 --output-file` includes a folder path
- Fixed misaligned stores (undefined behavior) for buffer records
- Fixed crash when only scratch reporting is enabled
- Fixed `MeanOccupancy` metrics
- Fixed aborted-application validation test to properly check for `hipExtHostAlloc` command
- Fixed implicit reduction of SQ and GRBM metrics
- Fixed support for derived counters in reduce operation
- Bug fixed in max-in-reduce operation
- Introduced fix to handle a range of values for `select()` dimension in expressions parser
- Conditional `aql::set_profiler_active_on_queue` only when counter collection is registered (resolves Navi3 kernel tracing issues)

### Removed

- Removed gfx8 metric definitions
- Removed `rocprofv3` installation to sbin directory

## ROCprofiler-SDK 0.6.0 for ROCm release 6.4

### Added

- Support for `select()` operation in counter expression.
- `reduce()` operation for counter expression with respect to dimension.
- `--collection-period` feature in `rocprofv3` to enable filtering using time.
- `--collection-period-unit` feature in `rocprofv3` to control time units used in collection period option.
- Deprecation notice for ROCProfiler and ROCProfilerV2.
- Support for rocDecode API Tracing
- Usage documentation for ROCTx
- Usage documentation for MPI applications
- SDK: `rocprofiler_agent_v0_t` support for agent UUIDs
- SDK: `rocprofiler_agent_v0_t` support for agent visibility based on gpu isolation environment variables such as `ROCR_VISIBLE_DEVICES` and so on.
- Accumulation VGPR support for `rocprofv3`.
- Host-trap based PC sampling support for rocprofv3.
- Support for OpenMP tool.

## ROCprofiler-SDK 1.0.0 for ROCm release 7.0

### Added

- Added support for rocJPEG API Tracing
- Added MI350X/MI355X support
- Added rocprofiler_create_counter to allow for adding custom derived counters at runtime.
- Added support for iteration based counter multiplexing to rocprofv3 (see documentation)
- Added perfetto support for counter collection.
- Added support for negating rocprofv3 tracing options when using aggregate options, e.g. `--sys-trace --hsa-trace=no`
- Added `--agent-index` option in rocprofv3 to specify the agent naming convention in the output
  - absolute == node_id
  - relative == logical_node_id
  - type-relative == logical_node_type_id
- Added MI300 stochastic (hardware-based) PC sampling support in ROCProfiler-SDK and ROCProfV3

### Changed

- SDK no longer creates a background thread when every tool returns a nullptr from `rocprofiler_configure`.
- Updated disassembly.hpp's vaddr-to-file-offset mapping to use the dedicated comgr API.
- rocprofv3 shorthand argument for `--collection-period` is now `-P` (upper-case) as `-p` (lower-case) is reserved for later use

### Resolved issues

- Fixed missing callbacks around internal thread creation within counter collection service

### Removed

- Support of gfx940 and gfx941 targets from compilation
