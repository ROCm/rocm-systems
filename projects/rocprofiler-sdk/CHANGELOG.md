# Changelog for ROCprofiler-SDK

Full documentation for ROCprofiler-SDK is available at [rocm.docs.amd.com/projects/rocprofiler-sdk](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/index.html)

## Unreleased

### Added

**API:**

  - Advanced Thread Trace (ATT) support in the live attach workflow:
    - On attach, the SDK registers for code-object iteration and creation callbacks so thread trace operates correctly on code objects that were loaded before the attach occurred.
    - Makes ATT usable on already-running production workloads without an application restart.
  - Experimental SQTT quick scan mode for thread trace, enabled through a new CMake flag:
    - Collects thread trace data without packet insertion or HSA signal manipulation, removing the queue interception overhead required by the standard ATT path.
    - Individual kernels can be traced without serialization, and the path is independent of the ROCm runtime version.
    - Experimental: intended to validate the new collection path and to enable out-of-process thread trace and long-kernel tracing in future releases.

**rocprof-trace-decoder:**

  - Python API for decoding Advanced Thread Trace (ATT) / SQTT data directly from Python, without writing a C++ consumer:
    - Wraps the decoder library and exposes thread trace decoding as a first-class Python interface, with samples demonstrating common workflows.
    - Useful for analysis scripts, Jupyter notebooks, and custom profiling tools that process ATT output programmatically.
    - Decoder integration tests have been migrated to Python, simplifying test authoring and making it easier for downstream tools to validate their trace-decoding pipelines.

### Changed

- Improved `rocpd` capability to convert sampling data from `rocprofiler-systems` database to Perfetto trace.

### Resolved issues

  - Fixed `rocprofv3` crashing during output generation when a second tool subscribed to code object tracing in the same process, which blocked profiling PyTorch and Triton workloads through rocprofiler-compute.
  - Fixed `rocprofv3` hanging instead of exiting when a fatal signal arrives while it is already handling one, for example when output generation aborts. It previously left GPU child processes running and required killing the process manually.

### Removed

## ROCprofiler-SDK 1.3.5 for ROCm release 10.0.0

### Added

**API:**

  - rocSHMEM host-stream API interception for the rocSHMEM tracing domain introduced in 1.3.0:
    - `rocshmem_putmem_on_stream`, `rocshmem_getmem_on_stream`, and `rocshmem_alltoallmem_on_stream` are intercepted and emitted as per-call trace records.
    - Records are interleaved with HIP, HSA, RCCL, and other runtime traces so rocSHMEM communication activity can be viewed on the same timeline as GPU compute.
  - hipFile API tracing as a first-class tracing domain:
    - hipFile API calls are intercepted through dispatch-table wrapping and emitted as per-call trace records alongside HIP, HSA, and other runtime activity.
    - Enables file I/O operations to be correlated with GPU kernels and memory copies in a single profiling timeline.
  - Streaming Performance Monitor (SPM) counter data in the rocpd output format:
    - SPM records are stored as `rocpd_track` rows labelled `SPM`, with counter values grouped by timestamp into `rocpd_sample` rows and per-dimension data in `rocpd_pmc_event` rows.
    - The rocpd schema gains the `sample_id`, `xcc`, `shader_engine`, and `instance` columns.
    - SPM data is consumable by any tool that reads the rocpd database and is convertible to CSV via `rocpd convert`. Conversion to the other output formats, such as Perfetto and OTF2, is not yet supported.

**rocprofv3 (CLI):**

  - OpenMP (OMPT) tracing via the new `--ompt-trace` flag:
    - Accepts a bare boolean or a space-separated category list (`all` `thread` `parallel` `task` `sync` `mutex` `target` `device` `error`), following the same style as `--pmc` and `--output-format`; for example, `--ompt-trace parallel task target sync`. Categories must be space-separated; comma-separated tokens are rejected. Also folded into `--sys-trace`/`--runtime-trace`.
    - rocpd-only trace: records go to the rocpd database (the default output format) and are exported via `rocpd convert`.
    - The OMPT callback layer is already supported by ROCprofiler-SDK; this flag makes it accessible without writing a custom tool.
  - hipFile API tracing via the new `--hipfile-trace` flag (or the `ROCPROF_HIPFILE_API_TRACE` environment variable):
    - Automatically included in `--runtime-trace` and `--sys-trace`.
    - Records are emitted across all supported output backends: CSV, JSON, Perfetto, OTF2, and rocpd.
  - Container-aware `rocattach` symbol resolution: attach entry points are resolved directly from the target process mapped ELF, and tool paths are validated from the target's perspective before injection. This allows attaching from a host to a containerized process without manually copying `.so` files.

### Changed

- Previously, `rocattach` calculated symbol offsets from the host's `librocprofiler-register.so` and applied them to the target's mapping, which failed when the host and container libraries differ in ELF layout or path. Offsets are now resolved from the target process itself.
- Idle inline queues with no active profiling consumers now bypass queue interposition entirely, reducing overhead for applications that create queues but do not immediately dispatch work.
- DWARF information is now parsed lazily, reducing startup overhead for attach and tracing sessions on large binaries.
- Disabled autoflush in thread trace to prevent premature buffer flushes that produced incomplete or corrupted traces.
- Bump rocpd schema to version 3.0.1 which supports NIC agent types.
- Bump rocpd schema to version 3.0.2 for HIP graph per-node attribution (`graph_exec_id`/`graph_node_id` columns on `rocpd_kernel_dispatch`/`rocpd_memory_copy` and the new `rocpd_graph_launch` table). The pre-graph-attribution 3.0.1 schema is now frozen under `versions/3.0.1/` per the rocpd schema versioning scheme.
- Bump rocpd schema to version 3.0.3 for SPM support. The pre-spm-support 3.0.2 schema is now frozen under `versions/3.0.2/` per the rocpd schema versioning scheme.

### Resolved issues

- Fixed a GPU stall in device thread trace that occurred when thread trace was started before `hsa_init()`.
- Fixed a counter-collection stall caused by an `InterceptQueue` ordering bug, and fixed an out-of-bounds write in `Submit()`.
- Fixed `roctxMark` calls propagating as kernel rename labels, which caused spurious kernel name changes in traces containing ROCTx markers.
- Fixed SQ aliasing on harvested WGPs and multi-counter desync on gfx11xx targets in AQLprofile, and corrected the `GcEaSeCounterBlockMaxEvent` value.
- Added a guard to prevent double-initialization of the PC sampling service.
- Fixed `rocprofv3` attach sessions exiting before all buffered output was flushed; attach sessions now block until the flush completes.
- Fixed the ordering of code object callbacks during attach, which could race with tools that depend on ordered delivery.
- Fixed the `fmt/format.h` include path, the `fpic` flag for samples, OMP lookup in CI, and clang-tidy quickscan enablement.

### Known issues

- SPM sessions can remain in a stale state after abrupt termination. See [GitHub issue #6489](https://github.com/ROCm/rocm-systems/issues/6489) for details.

### Removed

- Dependency on `libatomic`. The library was previously linked unconditionally through the `rocprofiler-sdk-atomic` interface target, which caused link failures on toolchains and container images where `libatomic1` is not installed. The single `std::atomic` use that required it has been replaced with explicit memory-ordering synchronization; behavior is unchanged.


## ROCprofiler-SDK 1.3.2 for ROCm release 7.14

### Added

**API:**

- Streaming Performance Monitor (SPM) counter collection support (beta):
    - New experimental API in `rocprofiler-sdk/experimental/spm.h`:
    - GPU-timestamped counter values alongside kernel dispatch information.
  - Added `spm_support` along with reserved padding to `rocprofiler_counter_info_v1_t`
  - Anytime initialization support.
    - Tools can call `rocprofiler_force_configure` after one or more other tools have configured rocprofiler-sdk.
      - NOTE: during the initialization of another tool, there is a small window where previously existing tools will not receive records generated from application background threads.

**rocprofv3(CLI):**
  - SPM counter collection support in `rocprofv3` (beta):
    - `--spm <counter>` flag to specify counters for SPM collection.
    - `--spm-sample-interval` and `--spm-sample-interval-unit` parameters to configure sampling rate.
    - `--spm-beta-enabled` flag to opt in to the beta SPM feature.
    - `--spm-config` option in `rocprofv3-avail` to list available SPM configurations.
  - JSON output format support for SPM.

**Documentation:**

  - SPM API reference guide (`api-reference/spm.rst`).
  - SPM usage guide for `rocprofv3` (`how-to/using-spm.rst`).
  - `--spm-config` documentation to `rocprofv3-avail` usage guide.

### Changed
- Bump rocpd schema to version 3.0.1 which supports NIC agent types.
- Bump rocpd schema to version 3.0.2 for HIP graph per-node attribution (`graph_exec_id`/`graph_node_id` columns on `rocpd_kernel_dispatch`/`rocpd_memory_copy` and the new `rocpd_graph_launch` table). The pre-graph-attribution 3.0.1 schema is now frozen under `versions/3.0.1/` per the rocpd schema versioning scheme.
- Bump rocpd schema to version 3.0.3 for SPM support. The pre-spm-support 3.0.2 schema is now frozen under `versions/3.0.2/` per the rocpd schema versioning scheme.
- Improved `rocpd` capability to convert sampling data from `rocprofiler-systems` database to Perfetto trace.

### Removed


## ROCprofiler-SDK 1.3.5

### Added
- HipFile API tracing support


## ROCprofiler-SDK 1.3.0 for ROCm release 7.2.4

### Optimized

- Reduced ROCprofiler-sdk profiling overhead: Improved profiling stability for vLLM workloads traced with PyTorch torch.profiler using the rocprofiler-sdk backend.

## ROCprofiler-SDK 1.3.0 for ROCm release 7.13

### Added

**API:**

- Late-start profiling support: Added automatic late‑start profiling support, enabling profiling when `rocprofiler-sdk` is loaded after HSA/HIP runtimes have already initialized.
  - `rocprofiler_force_configure()` now automatically detects and profiles runtimes initialized before the SDK loads.
  - Integrates with `rocprofiler-register` to retrieve the registered API tables.
  - Supports all runtime types (HSA, HIP, ROCTX, RCCL, ROCDecode, ROCJpeg, and more) automatically.
  - No explicit late-start API calls required; works transparently.

- KFD (Kernel Fusion Driver) event tracing support:
  - Buffer service configurations for each KFD buffer tracing type.
  - New type `tool_buffer_tracing_kfd_record_t` using `std::variant` to wrap 8 different KFD buffer tracing types.
  - Each KFD event generates `rocpd_info_pmc`, `rocpd_event`, `rocpd_region`, and `rocpd_pmc_event` rows
  - Fixed handling for special SVM location in KFD prefetch location reporting
  - Fixed parsing for queue restore events to handle both correct format (character '0') and broken driver format (NULL character '\0')

- Per-graph-node attribution for HIP graph operations:
  - New callback tracing kind `ROCPROFILER_CALLBACK_TRACING_HIP_GRAPH` fires for `hipGraphInstantiate*`, `hipGraphExecDestroy`, and `hipGraphLaunch{,_spt}` lifecycle events. Payload carries a process-monotonic `graph_exec_id` and the raw `hipGraphExec_t` handle. Mirrors the `HIP_STREAM` design.
  - New buffer tracing kind `ROCPROFILER_BUFFER_TRACING_HIP_GRAPH` emits a summary record per successful `hipGraphLaunch` invocation with `graph_exec_id`, `kernel_dispatch_count`, agent, and launch timestamps.
  - Tools can associate kernel dispatches and memory copies with their producing graph node by maintaining a per-thread attribution stack (push on EXEC_LAUNCH ENTER, pop on EXIT) and capturing the top of stack into the external correlation id at request time. See the doc comment on `rocprofiler_callback_tracing_hip_graph_data_t` for the recipe.
  - Attribution accuracy across launches requires segmented scheduling (default), `AMD_DIRECT_DISPATCH=1`, and single-threaded launching of one `hipGraphExec_t`.

**rocprofv3 (CLI):**

- Multi-pass counter collection support: Support for multiple `--pmc` flags to define separate counter groups for different profiling passes.
  - Ability to combine command-line `--pmc` flags with input file counter groups.
  - Each pass generates output in a separate `pass_n` subdirectory.
  - Example: `rocprofv3 --pmc SQ_WAVES --pmc GRBM_COUNT -- <app>` creates two profiling passes.

- KFD (Kernel Fusion Driver) event tracing support:
  - KFD record dumping to `rocpd` with support for 8 main KFD event types.
  - Support for `rocpd` to Perfetto conversion for KFD events.
  - `--kfd-trace` flag to enable KFD event tracing.

- ROCTx Support for ATT: Added ROCtx support to device thread trace when using `--att --selected-regions`.
  - Allows `roctxProfilerPause` and `roctxProfilerResume` to explicitly control when ATT data collection starts and stops.
  - Enables more precise, region-focused ATT tracing with reduced overhead and noise.
  - Supports multiple resume/pause cycles, each producing separate trace output files.
  - Incompatible with `--att-consecutive-kernels`.

- PC Sampling Support for Dynamic Attach: Added PC Sampling support for Dynamic Attach feature, allowing users to attach to a running application and collect PC samples without restarting the workload.
  - Enables profiling long-running or production-style jobs at the point of interest.
  - Results integrate with the existing PC sampling analysis flow.

- HIP graph attribution fields and trace:
  - Kernel and memory-copy records gain `graph_exec_id` and `graph_node_id` fields in JSON and rocpd output. rocpd conversion exposes these fields in CSV, OTF2, and Perfetto output.
  - New `--hip-graph-trace` CLI flag emits per-launch summary records (one row per successful `hipGraphLaunch`) in JSON and rocpd output, including `graph_exec_id` and `kernel_dispatch_count`. Automatically enabled by `--hip-trace` / `--hip-runtime-trace` since HIP graphs are part of the HIP runtime.
- rocSHMEM API tracing support:
  - `--rocshmem-trace` flag to enable tracing of rocSHMEM host-stream APIs.
  - Included in the `--sys-trace` and `--runtime-trace` aggregate tracing options.
  - Emitted directly to the JSON and rocpd (default, `.db`) output formats; CSV, Perfetto (`.pftrace`), and OTF2 output are produced from the rocpd database via `rocpd convert`.

**Documentation:**

- Added marker-controlled thread tracing section to the thread trace how-to guide.
- Added cross-reference from ROCTx documentation to ATT with `selected-regions`.
- Added HIP graph attribution section to the rocprofv3 how-to guide covering the new output columns, the `--hip-graph-trace` flag, the determinism contract, and v1 limitations.

### Changed

**Implementation:**

- Late-start architecture redesign: Removed direct runtime symbol access in favor of proper rocprofiler-register integration.
  - Replaced ~600 lines of `dlopen`/`dlsym` bypass logic with ~80 lines by using `rocprofiler_register_invoke_all_registrations()`.
  - Late-start now works by requesting `rocprofiler-register` to repropagate stored API tables.
  - Extensible design. Automatically supports new runtimes without SDK code changes.
  - Provides a proper separation of concerns. `rocprofiler-register` manages the table storage while SDK manages the table wrapping.
- Counter dimension encoding changed from fixed-width to variable-width allocation per dimension type.
- Dimension selection and reduction logic now uses explicit dimension masks and single-index selection.
- HSA queue interception extended to handle AMD extended kernel dispatch packets.

### Removed

- Counter collection support for plain text (`.txt`) input files has been deprecated due to lack of schema validation and input sanitization. Only structured file formats (JSON and YAML) with schema validation are supported.

### Resolved issues

- Fixed rocpd OTF2 output to add `ACCELERATOR_DEVICE` as system tree node domain for AMD devices.
- Fixed `rocprofv3` input file parsing where comment lines containing `pmc:` were incorrectly processed as valid counter collection directives, causing unintended profiling passes.

**Internal APIs (non-public):**

- Removed internal functions (were never in public headers):
  - `rocprofiler_start_late_internal()`
  - `rocprofiler_is_late_start_internal()`
  - `rocprofiler_stop_late_internal()`
- Replaced with single internal function: `rocprofiler::late_start::invoke_register_propagation()`

**Note:** Public API (`rocprofiler_force_configure()`) remains unchanged - no breaking changes for users

## ROCprofiler-SDK 1.1.0 for ROCm release 7.2

### Added
- Counter collection support for `gfx1150` and `gfx1151`.
- HSA Extension API v8 support.
- `hipStreamCopyAttributes` API implementation.
- `--profile-mpi-ranks` option in `rocprofv3` to selectively profile specific MPI ranks. Supports comma-separated ranges and individual ranks (e.g., `--profile-mpi-ranks 0-3,8,10-15`).

### Optimized

- Improved process attachment and updated the corresponding [documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3-process-attachment.html).
- Improved [Quick reference guide for rocprofv3] (https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick_guide.html).
- Updated installation documentation with links to the latest repository (https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/install/installation.html).

### Resolved issues
- Fixed multi-GPU dimension mismatch.
- Fixed device lock issue for dispatch counters.
- Addressed OpenMP Tools task scheduling null pointer exception.
- Fixed stream ID errors arising during process attachment.
- Fixed issues arising during dynamic code object loading.

## ROCprofiler-SDK 1.1.0 for ROCm release 7.1

### Added
- Dynamic process attachment- ROCprofiler-sdk and `rocprofv3` now facilitate dynamic profiling of a running GPU applications by attaching to its process ID (PID), rather than launching the application through the profiler itself.
- Scratch-memory trace information to the Perfetto output in `rocprofv3`.
- New capabilities to the thread trace support in `rocprofv3`, including real-time clock support for thread trace alignment on gfx9 architecture. This enables high-resolution clock computation and better synchronization across shader engines. Additionally, `MultiKernelDispatch` thread trace support is now available across all ASICs.
- Documentation for dynamic process attachment.
- Documentation for `rocpd` summaries.


### Optimized
- Improved the stability and robustness of the `rocpd` output.

## ROCprofiler-SDK 1.0.0 for ROCm release 7.0

### Added

- Support for [rocJPEG](https://rocm.docs.amd.com/projects/rocJPEG/en/latest/index.html) API Tracing.
- Support for AMD Instinct MI350X and MI355X accelerators.
- `rocprofiler_create_counter` to facilitate adding custom derived counters at runtime.
- Support in `rocprofv3` for iteration based counter multiplexing.
- Perfetto support for counter collection.
- Support for negating `rocprofv3` tracing options when using aggregate options such as `--sys-trace --hsa-trace=no`.
- `--agent-index` option in `rocprofv3` to specify the agent naming convention in the output:
  - absolute == node_id
  - relative == logical_node_id
  - type-relative == logical_node_type_id
- MI300 and MI350 stochastic (hardware-based) PC sampling support in ROCProfiler-SDK and `rocprofv3`.
- Python bindings for `rocprofiler-sdk-roctx`
- SQLite3 output support for `rocprofv3` using `--output-format rocpd`.
- `rocprofiler-sdk-rocpd` package:
  - Public API in `include/rocprofiler-sdk-rocpd/rocpd.h`.
  - Library implementation in `librocprofiler-sdk-rocpd.so`.
  - Support for `find_package(rocprofiler-sdk-rocpd)`.
  - `rocprofiler-sdk-rocpd` DEB and RPM packages.
- `--version` option in `rocprofv3`.
- `rocpd` Python package.
- Thread trace as experimental API.
- ROCprof Trace Decoder as experimental API:
  - Requires [ROCprof Trace Decoder plugin](https://github.com/rocm/rocprof-trace-decoder).
- Thread trace option in the `rocprofv3` tool under the `--att` parameters:
  - See [using thread trace with rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/amd-mainline/how-to/using-thread-trace.html)
  - Requires [ROCprof Trace Decoder plugin](https://github.com/rocm/rocprof-trace-decoder).
- `rocpd` output format documentation:
  - Requires [ROCprof Trace Decoder plugin](https://github.com/rocm/rocprof-trace-decoder).
- Perfetto support for scratch memory.
- Support in the `rocprofv3` avail tool for command-line arguments.
- Documentation for `rocprofv3` advanced options.
- Support for multi dispatch ATT file added

### Changed

- SDK to NOT to create a background thread when every tool returns a nullptr from `rocprofiler_configure`.
- `vaddr-to-file-offset` mapping in `disassembly.hpp` to use the dedicated comgr API.
- `rocprofiler_uuid_t` ABI to hold 128 bit value.
- `rocprofv3` shorthand argument for `--collection-period` to `-P` (upper-case) while `-p` (lower-case) is reserved for later use.
- Default output format for `rocprofv3` to `rocpd` (SQLite3 database).
- `rocprofv3` avail tool to be renamed from `rocprofv3_avail` to `rocprofv3-avail` tool.
- `rocprofv3` tool to facilitate thread trace and PC sampling on the same agent.

### Resolved issues

- Fixed missing callbacks around internal thread creation within counter collection service.
- Fixed potential data race in the ROCprofiler-SDK double buffering scheme.
- Fixed usage of std::regex in the core ROCprofiler-SDK library that caused segfaults or exceptions when used under dual ABI.
- Fixed Perfetto counter collection by introducing accumulation per dispatch.
- Fixed code object disassembly for missing function inlining information.
- Fixed queue preemption error and `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` error for stochastic PC-sampling in MI300X, leading to stabler runs.
- Fixed the system hang issue for host-trap PC-sampling on MI300X.
- Fixed `rocpd` counter collection issue when counter collection alone is enabled. `rocpd_kernel_dispatch` table is updated to be populated by counters data instead of kernel_dispatch data.
- Fixed `rocprofiler_*_id_t` structs for inconsistency related to a "null" handle:
  - The correct definition for a null handle is `.handle = 0` while some definitions previously used `UINT64_MAX`.
- Fixed kernel trace csv output generated by `rocpd`.

### Removed

- Support for compilation of gfx940 and gfx941 targets.

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

## ROCprofiler-SDK for AFAR II

### Added

- HIP API tracing
- ROCTx tracing
- Tracing ROCProf Tool V3
- Documentation packaging
- ROCTx control (start and stop)
- Memory copy tracing

## ROCprofiler-SDK for AFAR I

### Added

- HSA API tracing
- Kernel dispatch tracing
- Kernel dispatch counter collection
  - Instances reported as single dimension
  - No serialization
