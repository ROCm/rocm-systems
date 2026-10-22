# Change Log for RDC

Full documentation for RDC is available at [ROCm DataCenter Tool User Guide](https://rocm.docs.amd.com/projects/rdc/en/latest/).

## RDC 1.4.0 for ROCm 10.1.0

### Changed

- **`rdc_policy_register()` returns `RDC_ST_GROUP_NOT_FOUND` for a GPU group that does not exist**, and through `rdcd` it returns `RDC_ST_MAX_LIMIT` when `rdcd` already serves 64 policy registrations from all of its clients. Through `rdcd`, it also waits up to 5 seconds for `rdcd` to admit the registration; with an `rdcd` older than this release, the first registration of a connection takes up to 5 seconds.

- `rdcd` pings the connection of each client every 60 seconds and closes a connection whose client does not answer within 20 seconds.

- `rdcd` runs at most two diagnostics at a time, `rdc_diagnostic_run()` and `rdc_test_case_run()` together for all of its clients, and returns `RDC_ST_MAX_LIMIT` for another, so that clients cannot keep the GPUs busy with diagnostics without limit. A run keeps its place until it ends, also after its client has gone away.

- **`rdc_health_get()` returns `RDC_ST_GROUP_NOT_FOUND` for a GPU group that does not exist**.
  - It returned `RDC_ST_OK` with no health watches, so `rdci health -f` showed every watch as Off and exited successfully, and `rdci health -c` failed asking for watches to be enabled. Both commands now fail with a message that the group does not exist.

- **The RDC job and group calls reject IDs and names that do not fit their buffers.** `rdc_job_start_stats()`, `rdc_job_get_stats()`, `rdc_job_stop_stats()` and `rdc_job_remove()` return `RDC_ST_BAD_PARAMETER` for a job ID of 64 bytes or more or with a control character, and `rdc_group_gpu_create()` and `rdc_group_field_create()` for a name of 256 bytes or more; `rdcd` rejects the same, and IDs or names with a NUL inside, from any gRPC client.
  - The job ID parameter is a `char[64]`, but RDC accepted longer job IDs and stored them whole, and cut longer names. `rdci stats` with such a job ID, and `rdci group -c` and `rdci fieldgroup -c` with such a name, now fail.

### Removed

- Removed the unused `GetBlockNameStr()` test helper and its GPU block name map. Nothing called it, and its `static_assert` on `AMDSMI_GPU_BLOCK_LAST` broke the RDC build whenever AMD SMI added an IP block (#10949).

### Resolved Issues

- `rdcd` no longer keeps a thread and memory for each policy registration whose client went away, and no longer hangs at shutdown while policy registrations arrive.

- All clients registered for the policies of one GPU group receive every notification. Each notification went to only one of them.

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

- `rdci group -l` lists 64 groups, the most RDC holds; it reported `The max limit reached`.

- `rdc_job_get_stats()` and `rdc_policy_get()` with `rdcd` no longer report more GPUs, processes, or policies than the `rdcd` reply carries, or read past the entries it carries.

- `RDC_FI_GPU_MEMORY_CUR_BANDWIDTH` no longer reports zero under DMA or copy-only memory traffic on GPUs whose instantaneous UMC activity does not reflect DMA transfers. It now derives memory activity from the `mem_activity_acc` accumulator over firmware time, and falls back to the instantaneous `umc_activity` reading when the accumulator or firmware timestamp is unavailable.

- `rdci topo` and `rdc_device_topology_get()` now report the links of the requested GPU. They reported links between other GPU pairs, which showed the wrong link types on systems where GPU pairs are not all linked the same way, such as XGMI hives joined by PCIe.
  - `rdci topo` now exits with an error code when the GPU index is out of range or the topology query fails, instead of exiting 0, and `rdc_device_topology_get()` reports link types RDC does not define as `RDC_IOLINK_TYPE_UNDEFINED`.

- RDC checks GPU policy conditions every 500 ms, as intended. It checked them every 0.5 ms, which read every GPU in a policy group thousands of times per second and kept `rdcd` busy on a CPU core.

- Deleting a GPU group stops its policies. RDC kept checking the policies of a deleted group on whatever GPUs an uninitialized group record held, and with the GPU reset action could reset those GPUs.

- Setting or deleting policies while RDC checks them, and registering for policy notifications with `rdcd`, are now thread-safe.

- `rdc_policy_set()` returns `RDC_ST_GROUP_NOT_FOUND` for a GPU group that does not exist, instead of accepting the policy, and `rdci policy` exits with the status of a failed set.

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

- `rdci policy --reg` no longer uses a whole CPU core once rdcd goes away, and no longer reads freed memory as it exits. With a connection to rdcd, `rdc_policy_unregister()` and `rdc_disconnect()` now wait for a running policy callback to return, and `rdc_policy_register()` works again for a group that was unregistered; it used to return `RDC_ST_CONFLICT` for that group until the connection was closed.

- `rdcd` no longer sends uninitialized memory, or memory past a buffer, in its replies when the RDC library returns less than a reply holds: device names, component versions, group and field group information, field values, job statistics, policies, health checks, topology, link status and diagnostic results. It sends counts no larger than the entries it sends, and strings no longer than their buffers.

- `rdc_field_get_latest_value()` and `rdc_field_get_value_since()` set the `status` of the value they return, and `rdc_field_get_value_since()` returns all of a `BLOB` value instead of its first 8 bytes. Through `rdcd`, a string or `BLOB` value since a time is sent without its trailing spaces and without a NUL in place of the first of them.

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
