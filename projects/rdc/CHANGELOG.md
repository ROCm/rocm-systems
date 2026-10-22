# Change Log for RDC

Full documentation for RDC is available at [ROCm DataCenter Tool User Guide](https://rocm.docs.amd.com/projects/rdc/en/latest/).

## RDC 1.4.0 for ROCm 10.1.0

### Changed

- **`rdc_health_get()` returns `RDC_ST_GROUP_NOT_FOUND` for a GPU group that does not exist**.
  - It returned `RDC_ST_OK` with no health watches, so `rdci health -f` showed every watch as Off and exited successfully, and `rdci health -c` failed asking for watches to be enabled. Both commands now fail with a message that the group does not exist.

- `rdci diag` reports a check as skipped, instead of passed, when it or one GPU in it compared nothing (a group without GPUs, a group of one GPU for the topology check, a GPU whose sensors cannot be read, processes whose GPUs cannot be read); `rdc_diagnostic_run()`, and `rdc_test_case_run()` in embedded mode, also report a check whose module returned without writing a result as skipped.

- `rdci diag`'s GPU parameters line gives one sentence per GPU (limits in range, or how many crossed), and its per-GPU lines state each crossed limit, for example `Edge temperature 105 is at or above its maximum 100.`

- `rdci diag`'s compute process lines cover only the GPUs of the group, and leave out the process that runs the check (`rdcd`, or the program that embeds RDC), which can stay listed on a GPU after the ROCr diagnostics have used it.

- `rdci diag` cuts a compute process, node topology, GPU parameters, Compute Queue or System memory detail too long to show after a whole sentence, GPU pair or process ID and ends it with "(N more not shown)".

- `rdci diag` skips the Compute Queue and System memory checks on a GPU it cannot match to a GPU that ROCr lists, for example when AMD SMI cannot report the GPU's KFD node or ROCR_VISIBLE_DEVICES hides it, instead of testing whichever GPU ROCr lists in that position.

- `rdci diag`'s Compute Queue and System memory checks no longer spend about 0.75 seconds of CPU time per GPU in an unused timer (more on CPUs with a lower base clock).

- `rdci diag`'s Compute Queue and System memory checks report a GPU, or a part of a check, that they could not run as skipped, with the reason, instead of passed.

- `rdci diag`'s System memory check uses much less GPU memory and time.
  - It allocates and frees up to 64 MiB in each GPU memory pool that allows allocation instead of half of the pool, has the CPU write and read back 8 KiB of each GPU's memory instead of about 2 MB, and reports a pool without free memory as skipped instead of failed.

- The GPU memory that ROCr reserves for each GPU's queue in `rdci diag`'s Compute Queue and System memory checks, by default about 140 MiB, is now held until the check has finished the whole group, instead of being released after each GPU, where no other part of RDC is using ROCr.

### Removed

- Removed the unused `GetBlockNameStr()` test helper and its GPU block name map. Nothing called it, and its `static_assert` on `AMDSMI_GPU_BLOCK_LAST` broke the RDC build whenever AMD SMI added an IP block (#10949).

### Resolved Issues

- `rdci health -s` on a GPU group that does not exist no longer leaves an unused field group in rdcd. Repeated failures filled the field group table, and commands that need a new field group, such as `rdci dmon`, then failed until rdcd was restarted.

- `rdc_status_string()` now describes `RDC_ST_DISABLED_MODULE`. It returned "Unknown", so the rdcd log reported a disabled module as an unknown error.

- `RDC_FI_POWER_USAGE` now reports the socket power of AMD Instinct MI300 and newer GPUs. It reported 4294967295000000 µW, so power policies on these GPUs triggered on every check.
  - Partitions that have no socket power reading of their own now report N/A instead.

- `rdci topo` without `-i` now always reports that the GPU index is missing. It could instead skip that check and print the topology of an arbitrary GPU or an unrelated error.

- RDC no longer reports another GPU's XGMI link states, or garbage values that could crash `rdcd`, for a GPU whose link status AMD SMI cannot read. That GPU now reports no links.
  - `rdci link` now prints an error and exits with a nonzero status when it cannot get the link status, instead of printing nothing and exiting with 0.

- `RDC_FI_GPU_MEMORY_CUR_BANDWIDTH` no longer reports zero under DMA or copy-only memory traffic on GPUs whose instantaneous UMC activity does not reflect DMA transfers. It now derives memory activity from the `mem_activity_acc` accumulator over firmware time, and falls back to the instantaneous `umc_activity` reading when the accumulator or firmware timestamp is unavailable.

- `rdci topo` and `rdc_device_topology_get()` now report the links of the requested GPU. They reported links between other GPU pairs, which showed the wrong link types on systems where GPU pairs are not all linked the same way, such as XGMI hives joined by PCIe.
  - `rdci topo` now exits with an error code when the GPU index is out of range or the topology query fails, instead of exiting 0, and `rdc_device_topology_get()` reports link types RDC does not define as `RDC_IOLINK_TYPE_UNDEFINED`.

- `RDC_FI_ECC_CORRECT_TOTAL`, `RDC_FI_ECC_UNCORRECT_TOTAL`, and `RDC_FI_ECC_DEFERRED_TOTAL` no longer hang when AMD SMI defines GPU blocks above bit 31. The block iteration used a 32-bit counter that wrapped to zero instead of terminating.

- Opening or stopping an RDC handle no longer crashes the process when another handle is in use, and concurrent `rdc_connect()` calls no longer fail at random.

- Certificates signed with the RDC certificate authority setup in `authentication/openssl.cnf` can no longer sign other certificates. A certificate request could ask to become a signing certificate, and its holder could then create client and server certificates that `rdcd` and `rdci` trust.
  - If you signed requests that other people created, check the certificates in the `newcerts` directory of your CA for `CA:TRUE`. If you find one, create a new root certificate and reissue all certificates.

- `rdci discovery` now reports an error instead of exiting successfully with an incomplete GPU list when `rdcd` cannot read a GPU's attributes. Failed `rdc_device_get_attributes()` calls now return zeroed attributes, and `rdcd` replies to failed requests no longer contain garbage values.

- `rdci dmon` and `rdci health -s` name the missing group again, for example `Cannot find the group 4242`. Since ROCm 7.12 they printed only `Group not found` (or `Fieldgroup not found` for a missing field group in `rdci dmon`).

- `RDC_FI_GPU_PARTITION_COUNT` now reports how many compute partitions the GPU is split into, for example 1 in SPX mode. It reported 65535 or N/A in every partition mode on most GPUs.
  - A GPU that cannot be partitioned now reports 1.

- Stopping `rdcd`, or embedded RDC with `rdc_stop_embedded()`, no longer hangs. Rarely, the stop never returned and the process had to be killed.

- `rdcd` exits with an error when it cannot listen on its port, for example when another process uses the port. It printed "Server listening" and then crashed.

- `rdc_stop_embedded()` after `rdc_config_set()` no longer aborts with "terminate called without an active exception".

- `rdcd` names the owner of a lock file that it cannot open, instead of reporting "Numerical result out of range". It no longer reports "Bad file descriptor" for a lock file that does not exist, and prints the path of a lock file that it cannot read.

- `rdcd` exits at once, with status 1, on a second SIGTERM or SIGINT, or on one that arrives while it shuts down, so another stop request ends a stalled shutdown.

- A process that uses AMD SMI itself keeps working AMD SMI handles after it starts RDC in embedded mode. Loading RDC shut AMD SMI down and initialized it again, which invalidated the handles of the host process.

- `rdc_start_embedded()` returns `RDC_ST_FAIL_LOAD_MODULE`, and `rdcd` exits with an error, when AMD SMI fails to initialize. Both aborted the process.

- GPU indexes no longer shift when AMD SMI fails to identify a GPU, unless AMD SMI also lists CPUs and no identified GPU shares that GPU's socket. That GPU keeps its index and is reported as unavailable until AMD SMI identifies it: `rdci discovery` lists it as `unavailable`, `rdci link` shows no links for it, `rdci topo` shows no link to it, and its fields read N/A instead of another GPU's values.

- The node topology diagnostic checks each pair of GPUs, instead of each GPU against itself, and names the GPUs by index.

- The GPU parameters diagnostic reports each GPU's messages under that GPU and compares the critical voltage limits.

- The compute process diagnostic finds the GPUs of each process: it asked AMD SMI for them with a NULL list, which current AMD SMI versions reject, so it passed while processes ran; it also retries when processes start while it reads the list.

- `rdcd` and `rdci` no longer need about 7 MB of stack per diagnostic run.

- `rdci diag` no longer shows the results of an earlier run for a check that did not run.

- `rdcd` no longer logs an error for each diagnostic run with a custom configuration.

- `rdci diag`'s Compute Queue and System memory checks run on the GPU each result names, report a result and a line for each GPU, and name each GPU in their progress lines.

- `rdci diag` reports correct per-GPU results for groups with GPU partitions.

- `rdci diag`'s Compute Queue and System memory checks keep working after a check on one GPU stops early, for example on a GPU without a compute queue kernel, and no longer leave GPU memory allocated after it.
  - Before, they failed on every later run until `rdcd` restarted, and `rdcd`'s profiler fields, when first read after such a check, had no values until then; in an `rdcd` that had read a profiler field first, each such check instead left GPU memory allocated until `rdcd` stopped.

- `rdcd` no longer stops when a Compute Queue or System memory check fails to clean up.

- `rdci diag`'s Compute Queue and System memory checks no longer clear the `HSA_ENABLE_INTERRUPT` environment variable of the process that runs them.

- Fixed memory-safety issues in `rdci diag`'s Compute Queue and System memory checks.

- `rdci diag`'s Compute Queue check fails a GPU whose kernel returns a wrong search result; before, it reported such a run as passed.

- A diagnostic GPU kernel or memory copy that does not finish in 10 seconds fails its GPU's check instead of holding the diagnostic run, and a diagnostic queue that cannot be destroyed fails its GPU's check, where its failure was ignored.
  - Until the process that runs them (`rdcd`, or a program that embeds RDC) restarts, later runs report that GPU as failed and every other GPU of the Compute Queue and System memory checks as skipped, and the GPU memory the stopped check had reserved, by default about 140 MiB on each GPU it had reached, stays reserved.

- RDC no longer loads diagnostic kernels from the working directory of the process that runs them (`rdcd`, or a program that embeds RDC).

- `rdci diag`'s System memory check reports a value mismatch as its GPU's failure; `rdcd` logs it at debug level instead of always printing it to its standard output.

- The `rdc_diag_kernel_tests` program of the RDC test package starts from its installed location.
  - It now also checks the diagnostic kernels installed under `lib/rdc/hsaco`; before, it checked only the build machine's source tree and reported its checks as skipped when that tree was absent.

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

### Changed

- **Since ROCm 7.12 (RDC 1.3.0), the group APIs report a missing group with a dedicated status**.
  - For a group that does not exist, the GPU group get-info, add, and destroy calls return `RDC_ST_GROUP_NOT_FOUND`, and the field group get-info and destroy calls return `RDC_ST_FLDGROUP_NOT_FOUND`, instead of `RDC_ST_NOT_FOUND`. `rdc_field_watch()` passes both statuses through, and `rdc_health_set()` returns `RDC_ST_GROUP_NOT_FOUND` for a missing GPU group.

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
