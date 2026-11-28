# Change Log for RDC

Full documentation for RDC is available at [ROCm DataCenter Tool User Guide](https://rocm.docs.amd.com/projects/rdc/en/latest/).

## RDC 1.5.0 for ROCm 10.3.0

### Changed

- **`rdc_policy_register()` returns `RDC_ST_GROUP_NOT_FOUND` for a GPU group that does not exist**, and through `rdcd` it returns `RDC_ST_MAX_LIMIT` when `rdcd` already serves 64 policy registrations from all of its clients. Through `rdcd`, it also waits up to 5 seconds for `rdcd` to admit the registration; with an `rdcd` older than this release, the first registration of a connection takes up to 5 seconds.

- `rdcd` pings the connection of each client every 60 seconds and closes a connection whose client does not answer within 20 seconds.

- `rdcd` runs at most two diagnostics at a time, `rdc_diagnostic_run()` and `rdc_test_case_run()` together for all of its clients, and returns `RDC_ST_MAX_LIMIT` for another, so that clients cannot keep the GPUs busy with diagnostics without limit. A run keeps its place until it ends, also after its client has gone away.

- **`rdc_health_get()` returns `RDC_ST_GROUP_NOT_FOUND` for a GPU group that does not exist**.
  - It returned `RDC_ST_OK` with no health watches, so `rdci health -f` showed every watch as Off and exited successfully, and `rdci health -c` failed asking for watches to be enabled. Both commands now fail with a message that the group does not exist.

- **`rdc_field_watch()` documents that a `max_keep_samples` or `max_keep_age` of 0 keeps no samples**.
  - The documentation said 0 meant no limit. To keep every sample, pass a large value, such as `UINT32_MAX`.

- **RDC keeps at most 256 jobs, running or stopped**.
  - A job with a new ID that starts when RDC already keeps 256 jobs removes the job that stopped first, and `rdc_job_get_stats()` returns `RDC_ST_NOT_FOUND` for it. If no job has stopped, a start on a group whose job fields are not watched first stops the running job, started first, whose GPU group was destroyed or runs a later job, and that job makes room (it stays stopped if the start then fails); if there is none, `rdc_job_start_stats()` returns `RDC_ST_MAX_LIMIT`. A start that fails removes no other job. Every job used to stay in memory until `rdc_job_remove()` or `rdc_job_remove_all()` removed it.

- `rdc_job_start_stats()` returns `RDC_ST_MAX_LIMIT` for a group of more than `RDC_MAX_NUM_GPUS_PER_JOB` (16) GPUs, the most that `rdc_job_info_t` holds.

- `rdc_job_get_stats()` fills `gpus[0]` to `gpus[num_gpus - 1]`, ordered by `gpu_id`, which names the GPU or GPU partition of each entry. It used to place each GPU at the index of its GPU ID. The summary averages each statistic over the GPUs that have samples for it.

- `rdci stats -j` labels each GPU by its ID, such as `GPU 3`, or `GPU g0.1` for partition 1 of GPU 0, instead of by its position in the job. The JSON keys follow, such as `gpu_3` and `gpu_g0.1`. A statistic without samples prints as `N/A` in the text output and as `null` in JSON, instead of zeros and a minimum of 18446744073709551615. PCIe standard deviations print with two decimals, like the other statistics.

- `rdcd` exits with status 1 for an unknown option or a missing option argument, instead of ignoring it and starting, and whenever it finds another `rdcd` running; it exited with status 0 when two started at the same time.

- A non-root `rdcd` no longer refuses to start when it cannot read the lock file of the root `rdcd`, `/var/run/rdcd.lock`. It still stops when it can read that file and the root `rdcd` holds it.

### Resolved Issues

- `rdcd` no longer keeps a thread and memory for each policy registration whose client went away, and no longer hangs at shutdown while policy registrations arrive. Unregistering the policies of a group also ends a registration whose client stays connected but stopped reading notifications.

- All clients registered for the policies of one GPU group receive every notification. Each notification went to only one of them, and a client could miss some when several GPUs met a condition at once.
  - `rdcd` holds up to 1,024 notifications for a client that is slow to read them, and drops the oldest beyond that.

- `rdc_policy_register()` through `rdcd` works again for a group whose registration ended by itself, for example because `rdcd` restarted. It returned `RDC_ST_CONFLICT` for that group until the connection was closed.

- `rdci diag`, and `rdc_diagnostic_run()` and `rdc_test_case_run()` with `rdcd`, report the error of a diagnostic that `rdcd` cannot run, for example for a GPU group that does not exist, and return `RDC_ST_CLIENT_ERROR` when `rdcd` cannot be reached. They reported success with an empty or stale result.

- `rdc_diagnostic_run()` and `rdc_test_case_run()` with `rdcd` no longer crash when `config` is NULL.

- `rdci health -s` on a GPU group that does not exist no longer leaves an unused field group in rdcd. Repeated failures filled the field group table, and commands that need a new field group, such as `rdci dmon`, then failed until rdcd was restarted.

- `rdc_status_string()` now describes `RDC_ST_DISABLED_MODULE`. It returned "Unknown", so the rdcd log reported a disabled module as an unknown error.

- `RDC_FI_POWER_USAGE` now reports the socket power of AMD Instinct MI300 and newer GPUs. It reported 4294967295000000 µW, so power policies on these GPUs triggered on every check.
  - Partitions that have no socket power reading of their own now report N/A instead.

- `rdci topo` without `-i` now always reports that the GPU index is missing. It could instead skip that check and print the topology of an arbitrary GPU or an unrelated error.

- RDC no longer reports another GPU's XGMI link states, or garbage values that could crash `rdcd`, for a GPU whose link status AMD SMI cannot read. That GPU now reports no links.
  - `rdci link` now prints an error and exits with a nonzero status when it cannot get the link status, instead of printing nothing and exiting with 0.

- `rdcd` no longer crashes reading `RDC_FI_GPU_CLOCK` or `RDC_FI_MEM_CLOCK` when AMD SMI reports no current clock level, as it does for a power-gated clock domain on some APUs. The field is N/A then.

- Fields that AMD SMI cannot read are N/A instead of AMD SMI's "not available" value, such as 4294967295 for `RDC_FI_GPU_GFX_BUSY_INST` or 65535 for `RDC_FI_GPU_VCN_BUSY_INST` on Radeon GPUs. This covers the utilization and clock fields, the `RDC_FI_GPU_*_BUSY_INST` fields, and the `RDC_HEALTH_*` throttle and residency counters.

- `RDC_FI_ECC_CORRECT_TOTAL`, `RDC_FI_ECC_UNCORRECT_TOTAL`, and `RDC_FI_ECC_DEFERRED_TOTAL` are N/A on GPUs without ECC, like the per-block ECC fields, instead of 0. The memory health check still checks such a GPU's retired pages.

- `RDC_FI_GPU_MEMORY_CUR_BANDWIDTH` is N/A instead of 0 when the GPU reports no memory activity.

- `rdci group -l` lists 64 groups, the most RDC holds; it reported `The max limit reached`.

- `rdc_job_get_stats()` and `rdc_policy_get()` with `rdcd` no longer report more GPUs, processes, or policies than the `rdcd` reply carries, or read past the entries it carries.

- `rdci topo` and `rdc_device_topology_get()` now report the links of the requested GPU. They reported links between other GPU pairs, which showed the wrong link types on systems where GPU pairs are not all linked the same way, such as XGMI hives joined by PCIe.
  - `rdci topo` now exits with an error code when the GPU index is out of range or the topology query fails, instead of exiting 0, and `rdc_device_topology_get()` reports link types RDC does not define as `RDC_IOLINK_TYPE_UNDEFINED`.

- RDC checks GPU policy conditions every 500 ms, as intended. It checked them every 0.5 ms, which read every GPU in a policy group thousands of times per second and kept `rdcd` busy on a CPU core.

- Deleting a GPU group stops its policies. RDC kept checking the policies of a deleted group on whatever GPUs an uninitialized group record held, and with the GPU reset action could reset those GPUs.

- Setting or deleting policies while RDC checks them, and registering for policy notifications with `rdcd`, are now thread-safe.

- `rdc_policy_set()` returns `RDC_ST_GROUP_NOT_FOUND` for a GPU group that does not exist, instead of accepting the policy, and `rdci policy` exits with the status of a failed set.

- A process that uses AMD SMI itself keeps working AMD SMI handles after it starts RDC in embedded mode. Loading RDC shut AMD SMI down and initialized it again, which invalidated the handles of the host process. Such a process must pass `AMDSMI_INIT_AMD_CPUS` in its first `amdsmi_init()` call for RDC to read CPU fields.

- `rdc_start_embedded()` returns `RDC_ST_FAIL_LOAD_MODULE`, and `rdcd` exits with an error, when AMD SMI fails to initialize. Both aborted the process.

- GPU indexes no longer shift when AMD SMI fails to identify a GPU, unless AMD SMI also lists CPUs and no identified GPU shares that GPU's socket. That GPU keeps its index and is reported as unavailable until AMD SMI identifies it: `rdci discovery` lists it as `unavailable`, `rdci link` shows no links for it, `rdci topo` shows no link to it, and its fields read N/A instead of another GPU's values. `rdci config` sets and clears the settings of a group that holds such a GPU on the other GPUs, instead of stopping at it; after a set, that GPU gets the settings once AMD SMI identifies it.

- The node topology check of `rdci diag` reports the link weight between each pair of GPUs in the group. It reported each GPU's weight to itself.

- Opening or stopping an RDC handle no longer crashes the process when another handle is in use, and concurrent `rdc_connect()` calls no longer fail at random.

- Certificates signed with the RDC certificate authority setup in `authentication/openssl.cnf` can no longer sign other certificates. A certificate request could ask to become a signing certificate, and its holder could then create client and server certificates that `rdcd` and `rdci` trust.
  - If you signed requests that other people created, check the certificates in the `newcerts` directory of your CA for `CA:TRUE`. If you find one, create a new root certificate and reissue all certificates.

- A compute partition's `RDC_FI_GPU_CLOCK` no longer reports 65535 MHz or another partition's clock when the GPU lists unavailable clock readings before the available ones. It stops updating, instead of reporting a wrong clock, when the GPU's partition count drops to the partition's index or below while the partition is watched (for example, when AMD SMI reports the partition profile as not supported).
  - A reading of 0 MHz now counts like any other. RDC skipped it, so partitions could report another partition's clock, or no clock.

- `rdci discovery` now reports an error instead of exiting successfully with an incomplete GPU list when `rdcd` cannot read a GPU's attributes. Failed `rdc_device_get_attributes()` calls now return zeroed attributes, and `rdcd` replies to failed requests no longer contain garbage values.

- `rdc_job_get_stats()` and `rdci stats -j` report every GPU of a job. Only the GPU with the lowest ID was summarized, and the entries of the other GPUs were not filled in. Job statistics for a GPU partition, or for a GPU whose ID is above 15, no longer crash `rdcd`. A GPU without a total memory reading reports memory utilization as unavailable instead of failing the call, and memory utilization is no longer inflated on GPUs with 400 GiB of memory or more.

- A job start that fails, for example on a GPU group that runs another job, no longer leaves a running job behind. Its job ID returned `RDC_ST_ALREADY_EXIST` until the job was removed, and stopping it stopped the stats of the group's other job. A start that failed to set up process events, whose job kept collecting stats although `rdc_job_start_stats()` returned the error, no longer leaves a job either.

- `rdc_job_remove()` and `rdc_job_remove_all()` now free the ID of a job that cannot stop, for example after a client unwatched the job fields of its group. Starting a job with that ID returned `RDC_ST_ALREADY_EXIST`.

- Two `rdc_job_start_stats()` calls with the same job ID at the same time no longer both succeed. One of the two GPU groups then kept a watch that failed its later jobs with `RDC_ST_CONFLICT`.

- `rdcd` no longer crashes during `rdci diag`, and later diagnostic runs and profiler fields keep working. Repeated `rdci diag` runs also no longer use up GPU memory in `rdcd`.
  - After the first `rdci diag`, `rdcd` keeps the ROCm runtime loaded until it stops, as it already does after the first profiler field read, so changing the GPU partition mode needs `rdcd` stopped. `rdci diag` requests now run one at a time.

- `rdci diag` now runs the Compute Queue and System memory checks on AMD Instinct MI455X GPUs instead of skipping them.

- `field_id_string()` returns `INVALID` for a field ID that RDC does not know, for example one that a newer `rdcd` reports. It read past the end of RDC's field table, which could crash the program.

- Log lines that several threads write at the same time no longer run into each other.

- `rdc_group_gpu_create()` with `RDC_GROUP_DEFAULT` fails, and creates no group, when it cannot add one of the GPUs. It returned success with that GPU missing, or failed and left the group behind.

- A path that `get_rocm_path()` returned no longer changes when the function is called again, and the function is safe to call from several threads.

- `rdc_device_get_component_version()` no longer crashes `rdcd` when AMD SMI gives no build string. It then reports AMD SMI's version as major.minor.release.

- RDC no longer writes past its list of supported fields when its telemetry libraries support more than 8192 fields together. The list then holds the first 8192, and a watch keeps all the fields that it asks for.

- `RDC_FI_GPU_PARTITION_COUNT` now reports how many compute partitions the GPU is split into, for example 1 in SPX mode. It reported 65535 or N/A in every partition mode on most GPUs.
  - A GPU that cannot be partitioned now reports 1.

- Stopping `rdcd`, or embedded RDC with `rdc_stop_embedded()`, no longer hangs. Rarely, the stop never returned and the process had to be killed.

- Field watches with a `max_keep_samples` above 2147483647, such as `UINT32_MAX`, no longer crash `rdcd` or the process that embeds RDC.

- `rdcd` exits with an error when it cannot listen on its port, for example when another process uses the port, and the message asks whether `rdcd` is already running. It printed "Server listening" and then crashed.

- `rdc_stop_embedded()` after `rdc_config_set()` no longer aborts with "terminate called without an active exception".

- `rdcd` names the owner of a lock file that it cannot open, instead of reporting "Numerical result out of range". It no longer reports "Bad file descriptor" for a lock file that does not exist, and prints the path of a lock file that it cannot read.

- `rdcd` no longer follows a symbolic link at its lock file path or writes into a file that belongs to another user or has another name, and a root `rdcd` no longer refuses to start because of a `/tmp/rdcd.lock` that another user created and locked. For a lock file it refuses to use, `rdcd` exits with an error that names it.

- When `rdcd` takes more than 5 seconds to stop, another SIGTERM or SIGINT ends it at once, with status 1.

- `rdcd` no longer crashes when the watches due in one update read more than 8,192 GPU-field pairs from AMD SMI, for example two `rdci dmon` sessions of 128 fields on 64 GPUs each.

- `rdcd` no longer crashes listing more than 128 (`RDC_MAX_NUM_DEVICES`) GPUs and GPU partitions; `rdc_device_get_all()` lists the first 128.

- `rdci policy --reg` no longer uses a whole CPU core once rdcd goes away, and no longer reads freed memory as it exits. With a connection to rdcd, `rdc_policy_unregister()` and `rdc_disconnect()` now wait for a running policy callback to return, and `rdc_policy_register()` works again for a group that was unregistered; it used to return `RDC_ST_CONFLICT` for that group until the connection was closed.

## RDC 1.4.0 for ROCm 10.2.0

### Resolved Issues

- The `RDC_FI_GPU_MEMORY_CUR_BANDWIDTH` fix for zero readings under DMA or copy-only memory traffic is now included.

- `rdci dmon` and `rdci health -s` name the missing group again, for example `Cannot find the group 4242`. Since ROCm 7.12 they printed only `Group not found` (or `Fieldgroup not found` for a missing field group in `rdci dmon`).

## RDC 1.3.1 for ROCm 10.1.0

### Removed

- Removed the unused `GetBlockNameStr()` test helper and its GPU block name map. Nothing called it, and its `static_assert` on `AMDSMI_GPU_BLOCK_LAST` broke the RDC build whenever AMD SMI added an IP block (#10949).

### Resolved Issues

- `RDC_FI_ECC_CORRECT_TOTAL`, `RDC_FI_ECC_UNCORRECT_TOTAL`, and `RDC_FI_ECC_DEFERRED_TOTAL` no longer hang when AMD SMI defines GPU blocks above bit 31. The block iteration used a 32-bit counter that wrapped to zero instead of terminating.

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
