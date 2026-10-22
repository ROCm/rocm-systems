# Change Log for RDC

Full documentation for RDC is available at [ROCm DataCenter Tool User Guide](https://rocm.docs.amd.com/projects/rdc/en/latest/).

## RDC 1.4.0 for ROCm 10.1.0

### Changed

- **`rdc_health_get()` returns `RDC_ST_GROUP_NOT_FOUND` for a GPU group that does not exist**.
  - It returned `RDC_ST_OK` with no health watches, so `rdci health -f` showed every watch as Off and exited successfully, and `rdci health -c` failed asking for watches to be enabled. Both commands now fail with a message that the group does not exist.

- RDC stops listening for a GPU's process start and end events once no job or watch on that GPU uses them. Before, they stayed enabled after a job stopped.

- **RDC keeps at most 256 jobs, running or stopped**.
  - A job with a new ID that starts when RDC already keeps 256 jobs removes the job that stopped first, and `rdc_job_get_stats()` returns `RDC_ST_NOT_FOUND` for it. If no job has stopped, a start on a group whose job fields are not watched first stops the running job, started first, whose GPU group was destroyed or runs a later job, and that job makes room (it stays stopped if the start then fails); if there is none, `rdc_job_start_stats()` returns `RDC_ST_MAX_LIMIT`. A start that fails removes no other job. Every job used to stay in memory until `rdc_job_remove()` or `rdc_job_remove_all()` removed it.

- **`rdc_field_watch()` documents that a `max_keep_samples` or `max_keep_age` of 0 keeps no samples**.
  - The documentation said 0 meant no limit. To keep every sample, pass a large value, such as `UINT32_MAX`.

- **A job lists at most 64 processes**.
  - `rdc.h` documents this limit of `rdc_job_get_stats()`, and rdcd now logs once when a job has more processes.

- `rdc.h` documents field value timestamps (`rdc_field_value.ts`) and the timestamps of `rdc_field_get_value_since()` in milliseconds. It said microseconds.

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

- `rdci dmon` no longer hangs when it watches a GPU event and a GPU sends two events of that kind at once. RDC now gives every GPU event its own timestamp.

- `rdcd` no longer drops every GPU event read in one poll when one of them comes from a GPU that RDC does not list. It skips that event.

- RDC skips GPU events of a kind it does not know, instead of recording them as field 0. With assertions enabled, `rdcd` aborted.

- Watching a GPU event field, or starting a job, on several GPUs no longer stops at the first GPU whose AMD SMI handle lookup, capability or event mask write fails. RDC enables the events on the other GPUs.

- Watching a GPU event field, for example with `rdci dmon -e 2000`, no longer replaces the GPU events that other watches and running jobs on that GPU use, such as the process start and end events that job statistics record, and ending the watch no longer stops them. After AMD SMI fails to stop a GPU's event notification, the next watch of the same events gets them.

- A job start that fails, for example on a GPU group that runs another job, no longer leaves a running job behind. Its job ID returned `RDC_ST_ALREADY_EXIST` until the job was removed, and stopping it stopped the stats of the group's other job. A start that failed to set up process events, whose job kept collecting stats although `rdc_job_start_stats()` returned the error, no longer leaves a job either.

- `rdc_job_remove()` and `rdc_job_remove_all()` now free the ID of a job that cannot stop, for example after a client unwatched the job fields of its group. Starting a job with that ID returned `RDC_ST_ALREADY_EXIST`.

- Two `rdc_job_start_stats()` calls with the same job ID at the same time no longer both succeed. One of the two GPU groups then kept a watch that failed its later jobs with `RDC_ST_CONFLICT`.

- Field watches with a `max_keep_samples` above 2147483647, such as `UINT32_MAX`, no longer crash `rdcd` or the process that embeds RDC.

- Watches of a destroyed GPU group or field group no longer make every later unwatch fail: `rdci stats -x` and `rdci health --clear` failed with "Group not found", and `rdci dmon` left its fields being read after it ended. RDC also stops reading the fields of the destroyed group.

- Two jobs, or two health watches, on GPU groups that share a GPU both get that GPU's samples. Only one of them did, so `rdci stats -j` showed no samples for that GPU in the other job.

- `rdci stats -j` no longer reports a huge energy after a power sample with an earlier timestamp or a negative power reading, and no longer drops energy below one joule per sample, which made it 0 at low power or short update intervals.

- Job stats no longer list a process with ID 0 for a process event that names no process.

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
