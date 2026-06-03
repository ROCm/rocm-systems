<!-- markdownlint-disable MD024 -->

# Changelog for ROCm Systems Profiler

Full documentation for ROCm Systems Profiler is available at [https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/).

## ROCm Systems Profiler 1.7.0 for ROCm 7.14.0

### Added

- Unified-memory profiling reports (`unified_memory.txt` and
  `unified_memory.json`) summarizing KFD page-fault and page-migration events,
  including per-GPU counts, trigger breakdown (`gpu_page_fault`,
  `cpu_page_fault`, `prefetch`), and Host-to-Device / Device-to-Host migration
  bandwidth. Enable with `ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON`; requires
  `HSA_XNACK=1` on an XNACK-capable AMD GPU and ROCProfiler-SDK 1.2.2 or
  later. The required KFD tracing domains are enabled automatically.
- MPI-rank-based console output filtering features controlled with CLI arguments:
  `--rank-filter-logs` and `--rank-filter-id`.
- GPU Hardware Performance Counter (PMC) sampling via the ROCProfiler-SDK device
  counting service. Periodic per-GPU hardware counters are collected alongside
  existing PMC sources and exposed in both Perfetto and RocPD outputs. Specify
  counters with `ROCPROFSYS_GPU_PERF_COUNTERS` (comma-separated; suffix
  `:device=N` to target a specific GPU). Requires ROCProfiler-SDK 0.6.0 or
  later (ROCm 6.4.0+).
- GPU graphics and memory clock frequency metrics (`gfx_clock`, `mem_clock`) via
  AMD SMI, exposing `current_gfxclk` and `current_uclk` in MHz as PMC samples.
  Configure via `ROCPROFSYS_AMD_SMI_METRICS=gfx_clock,mem_clock`.
- Progress bars during trace cache post-processing: perfetto generation
  (`sequential dispatch`) shows one bar per buffered_storage file in turn; rocpd
  generation (`multithreaded dispatch`) shows a single aggregate bar accumulating
  updates from all worker threads.
- Per-stream Perfetto tracks (`HIP Activity Stream {N}`) for kernel dispatch,
  scratch memory, and memory copy events in the trace-cache path, matching the
  buffered tracing behavior. Controlled via `ROCPROFSYS_ROCM_GROUP_BY_QUEUE`
  (default: `false` — group by HIP stream).
- Add `--list-domains` and `--list-operations <domain>` to `rocprof-sys-avail`.
  There new options allow the user to query more information about available
  ROCm domains (used in `ROCPROFSYS_ROCM_DOMAINS`) and their operations.

### Changed

- Remove Boost as a Dyninst dependency by replacing Boost usage with in-tree
  `dyncompat` shims and C++17 standard library equivalents; Bundled Dyninst now
  requires **GCC ≥ 10**.
- The `trace-openmp` configuration preset no longer includes `HSA_API`,
  by default.
- `rocprof-sys-sample` - Aligned flags with `rocprof-sys-run`. Renamed `--freq`,
  `--cputime` and `--realtime` to `--sampling-freq`, `--sampling-cputime` and
  `--sampling-realtime`, respectively. Old flags are still handled as a part of
  backward compatibility.
- Allow presets to use `--gpus`/`--cpu`/`--ai-nics` flags without
  `--device`/`--host` flags.

### Resolved issues

- Fix ElfUtils build on GCC 15.
- Fix output directory of `rocpd` files when re-attaching to the same process
  with `rocprof-sys-attach`. Now, each session will have a unique output folder.
- Fix CPU related counters (like CPU frequency) missing from `rocpd` output.
- Fixed the handling of "group-by-queue" option in the Perfetto generator.
- Fix visualization of GPU counters, which made it look like there was activity
  between kernel dispatches.
- Fixed hang due to mismatched versions of `binutils` between system and bundled
  versions. Ensure that the vendored version of `binutils`'s symbols are hidden.
- Fix for ASAN build on TheRock.
- Fix issue that could cause certain events to appear in trace, when the should
  have been excluded due to roctx region filtering.
- Fix cmake issue that caused the wrong version of `elfutils` to be linked when
  building for TheRock. The system version of `elfutils` was used, rather than
  the vendored version causing package install failures.

## ROCm Systems Profiler 1.6.0 for ROCm 7.13.0

### Added

- Kernel Fusion Driver (KFD) event tracing support to capture page faults, page
  migrations, queue evictions, GPU unmap events, and dropped events. Requires
  ROCProfiler-SDK 1.2.2 or later. Enable with
  `ROCPROFSYS_ROCM_DOMAINS=kfd_events`.
- Support for pause and resume of profiling via `roctxProfilerPause` and
  `roctxProfilerResume`.
- Support for selective region tracing via the `ROCPROFSYS_SELECTED_REGIONS`
  environment variable, limiting tracing to specified regions.
- `--selected-regions` CLI argument to `rocprof-sys-sample`, `rocprof-sys-run`,
  and `rocprof-sys-instrument` for specifying selective region tracing from the
  command line.
- Support for re-attaching to a previously profiled process. After detaching,
  `rocprof-sys-attach` can re-attach to the same PID for a new profiling session.
- MPI-rank-based file output filtering features controlled with new CLI arguments:
  `--rank-filter-output` and `--rank-filter-id`.
- JSON-based configurable preset system with `--preset=<name>` flag, replacing the
  old `--<preset-name>` flags. Presets are now loaded from JSON files in
  `source/bin/common/presets/`, making them extensible and exportable. Use
  `--list-presets` to see available presets and `--explain=<name>` for detailed
  preset information.
- Domain flags for composable configuration: `--gpu[=metrics]`, `--rocm[=domains]`,
`--cpu[=hz]`, `--parallel[=runtimes]`. Domain flags can be combined with presets
  to customize profiling without editing configuration files.
- Configuration export via `--export-config[=file]` to save resolved settings as
  reusable JSON configuration files. Exported configs can be loaded back with
  `--preset=./config.json`.
- Topic-based help system: `--help` now shows a compact summary with essential
  options and a list of help topics. Use `--help=<topic>` (e.g., `--help=sampling`,
  `--help=gpu`, `--help=tracing`) to see only relevant options. Use `--help=all`
  for the full option listing.
- Post-run output summary during library finalization showing result file locations.
- JSON schema file (`share/rocprofiler-systems/presets/schema.json`) for preset
  validation.
- Documentation (`docs/how-to/instrumenting-rewriting-binary-application.rst`)
  describing what to do when Dyninst reports a "Failed to transform trace" error
  during instrumentation.

### Changed

- `rocprof-sys-avail` no longer queries GPU devices or hardware counters unless
  `--hw-counters` or `--all` is requested, reducing startup time and allowing
  settings/component queries in environments without GPU/ROCm.
- `rocprof-sys-instrument` diagnostic file dumps (available, instrumented,
  excluded, coverage, overlapping) are now gated behind the `--dump-info` flag
  instead of being generated unconditionally.
- Preset flags changed from `--balanced` to `--preset=balanced` syntax. The old
  `--<preset-name>` flags are still supported and handled within `preset_registry`.
- Removed the `ROCPROFSYS_USE_ROCM` CMake option. ROCm is now required for
  building the ROCm Systems Profiler.

### Resolved issues

- Fixed an issue where the `--rocm-domains` CLI option for `rocprof-sys-run` was not recognized.

## ROCm Systems Profiler 1.5.0 for ROCm 7.12.0

### Added

- Per-GPU RCCL communication data counters (Send/Recv) in `rocpd` output with multi-GPU device attribution using `ncclCommCuDevice` API.
- Presets profiles that configure the rocprofiler-system tools for common profiling scenarios, offering optimized configurations for specific use cases.
- SDMA (System Direct Memory Access) utilization metrics support via AMD SMI, showing device-level SDMA usage percentage aggregated from all processes. Configure via `ROCPROFSYS_AMD_SMI_METRICS=sdma_usage`.
- `rocprof-sys-attach` CLI tool for attaching to and profiling running processes via rocprofiler-sdk rocattach API (experimental).
- Support for OpenSHMEM API tracing via `ROCPROFSYS_USE_SHMEM=ON` configuration setting.

### Changed

- Simplify categorizing like pmc_info events by removing the "_<idx>" from the "symbol" field. ie., "JpegAct_0" -> "JpegAct".
- Added `libhsa-runtime64.so` and `libomp.so` to the internal library exclusion list for runtime instrumentation to prevent instrumenting of runtime library internals.
- RCCL implementation refactored with `production_pmc_registrar` for improved testability and separation of concerns.
- Unsupported RCCL datatypes now gracefully return 0 with `LOG_WARNING` instead of aborting profiler, allowing continued profiling with newer RCCL versions.
- Added AI NIC support.

### Resolved issues

- Fixed an issue where JPEG engine activity PMC events were not being collected for MI35X systems. Only the first 32 JPEG engines were being collected.
- Fixed MPI perfetto trace file merging when using trace cache mode with `ROCPROFSYS_PERFETTO_COMBINE_TRACES=ON`.
  Previously, each MPI rank would produce a separate trace file; now all ranks' traces are correctly merged into a single output file.

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
