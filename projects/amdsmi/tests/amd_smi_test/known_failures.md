# Known Test Failures

Tests listed here are skipped because the underlying API misbehaves on all tested
hardware. Each entry names the symptom, the affected test, and the file holding
the skip. Root causes are under investigation unless stated otherwise.

Paths in the **Location** column are relative to `tests/amd_smi_test/`. The skip
is the `AMDSMI_SKIP_KNOWN_FAILURE()` call at the top of the named test's body.
To retire an entry, delete that call and this row.

## Running the skipped tests

Set `AMDSMI_RUN_KNOWN_FAILURES=1` to run every test in this file instead of
skipping it, so its current behavior can be checked without editing source:

```shell
AMDSMI_RUN_KNOWN_FAILURES=1 ./amdsmitst
```

Entries here are skipped on **every** ASIC. Where a bug is known to affect only
some, prefer a per-ASIC `FILTER[...]` row in `amdsmitst.exclude` over an entry
here, so the coverage is kept everywhere else. The three
`AMDSMI_STATUS_UNEXPECTED_DATA` clock and thermal entries below are still global
pending a full per-ASIC sweep; re-scoping them needs a
`AMDSMI_RUN_KNOWN_FAILURES=1` run on each supported ASIC to find which ones pass.

Measured so far, on gfx950 (16 GPUs), all three still fail, but not with the
status recorded below: `GetClkFreq_AllGpusAllTypes` and
`GetClockInfo_AllGpusAllTypes` return `AMDSMI_STATUS_INVAL` and
`GetTempMetric_AllGpusAllTypesMetrics` returns
`AMDSMI_STATUS_INTERNAL_EXCEPTION`. The `UNEXPECTED_DATA` symptom below is from a
different ASIC, so the underlying cause may not be shared.

A test that now passes is a candidate to retire from this file; one that still
fails prints its real assertion. Results differ by ASIC, so re-verify on your
target hardware before removing an entry.

The variable does not override `amdsmitst.exclude`. A test filtered there stays
filtered no matter what the source or this variable says.

## AMDSMI_STATUS_UNEXPECTED_DATA (error 43)

These APIs return `AMDSMI_STATUS_UNEXPECTED_DATA` where `AMDSMI_STATUS_SUCCESS` is
expected. Root cause is unknown for all entries; they likely share a driver-side
data-format issue.

| API | Skipped test(s) | Location |
|-----|-----------------|----------|
| `amdsmi_get_clk_freq` | `GpuIntegration.GetClkFreq_AllGpusAllTypes` | `integration/gpu/clock/clock_freq_test.cc` |
| `amdsmi_get_clk_freq` | `GpuFunctionalReadOnly.TestFrequenciesRead` | `main.cc` |
| `amdsmi_set_clk_freq` | `GpuFunctionalReadWrite.TestFrequenciesReadWrite` | `main.cc` |
| `amdsmi_get_clk_info` | `GpuIntegration.GetClockInfo_AllGpusAllTypes` | `integration/gpu/clock/clock_freq_test.cc` |
| `amdsmi_get_violation_status` | `GpuIntegration.GetViolationStatus_AllGpus` | `integration/gpu/events/event_ptl_test.cc` |
| `amdsmi_get_gpu_xcd_counter` | `GpuIntegration.GetXcdCounter_AllGpus` | `integration/gpu/identity/id_info_test.cc` |
| `amdsmi_get_gpu_metrics_info` | `GpuIntegration.GetMetricsInfo_AllGpus` | `integration/gpu/metrics/metrics_test.cc` |
| `amdsmi_get_utilization_count` | `GpuIntegration.GetUtilizationCount_AllGpus` | `integration/gpu/perf/perf_overdrive_test.cc` |
| `amdsmi_get_gpu_activity` | `GpuIntegration.GetActivity_AllGpus` | `integration/gpu/perf/perf_overdrive_test.cc` |
| `amdsmi_get_energy_count` | `GpuIntegration.GetEnergyCount_AllGpus` | `integration/gpu/power/power_test.cc` |
| `amdsmi_get_pcie_bandwidth` | `GpuIntegration.GetPciBandwidth_AllGpus` | `integration/gpu/pci/pci_test.cc` |
| `amdsmi_get_pcie_info` | `GpuIntegration.GetPcieInfo_AllGpus` | `integration/gpu/pci/pci_test.cc` |
| `amdsmi_get_xgmi_info` | `SystemIntegration.GetGpuXgmiLinkStatus_AllGpus` | `integration/system/topology_test.cc` |
| `amdsmi_get_link_metrics` | `SystemIntegration.GetLinkMetrics_AllGpus` | `integration/system/topology_test.cc` |
| `amdsmi_get_afids_from_cper` | `GpuIntegration.GetAfidsFromCper_DummyBuffer` | `integration/gpu/ras/ras_ecc_test.cc` |
| `amdsmi_get_temp_metric` | `GpuIntegration.GetTempMetric_AllGpusAllTypesMetrics` | `integration/gpu/thermal/thermal_fan_test.cc` |
| `amdsmi_get_temp_metric` | `GpuFunctionalReadOnly.TempRead` | `main.cc` |
| `amdsmi_xgmi_*` (error injection) | `GpuFunctionalReadWrite.TestXGMIReadWrite` | `main.cc` |

## AMDSMI_STATUS_UNEXPECTED_SIZE (error 42)

| API | Skipped test(s) | Location |
|-----|-----------------|----------|
| counter lifecycle flow | `GpuFunctionalReadOnly.Counter_LifecycleWorkflow` | `functional/gpu/perf/counter_test.cc` |

## Setter reports success without taking effect

| API | Bug | Skipped test(s) | Location |
|-----|-----|-----------------|----------|
| `amdsmi_set_clk_freq` | Returns `AMDSMI_STATUS_SUCCESS`, but the GFX domain keeps reporting its previous level, so the readback check fails. The bitmask only limits which levels are *allowed*, so an idle domain need not move | `GpuFunctionalReadWrite.ClkFreq_SetVerifyRestore` | `functional/gpu/clock/clock_read_write_test.cc` |

## Library Input-Validation Bugs

These tests are skipped because the library crashes (segfault/abort) or returns
an undocumented status instead of the expected `AMDSMI_STATUS_INVAL`.

| API | Bug | Skipped test(s) | Location |
|-----|-----|-----------------|----------|
| `amdsmi_get_gpu_cper_entries` | Returns `AMDSMI_STATUS_OUT_OF_RESOURCES` for `nullptr` output instead of `AMDSMI_STATUS_INVAL` | `GpuIntegration.GetCperEntries_NullOutput` | `integration/gpu/ras/ras_ecc_test.cc` |
| `amdsmi_get_gpu_pci_throughput` | Returns `AMDSMI_STATUS_SUCCESS` for `nullptr` output pointers; should return `AMDSMI_STATUS_INVAL` | `GpuIntegration.GetPciThroughput_NullOutput` | `integration/gpu/pci/pci_test.cc` |
| `amdsmi_get_link_topology_nearest` | Returns `AMDSMI_STATUS_SUCCESS` for an invalid handle; should return `AMDSMI_STATUS_INVAL` | `SystemIntegration.GetLinkTopologyNearest_InvalidHandle` | `integration/system/topology_test.cc` |
| `amdsmi_get_processor_handle_from_bdf` | Returns `AMDSMI_STATUS_API_FAILED` for zero BDF; should return `NOT_FOUND` or `INVAL` | `SystemIntegration.GetProcessorHandleFromBdf_ZeroBdf` | `integration/system/enumeration_test.cc` |
| `amdsmi_gpu_xgmi_error_status` | Returns `AMDSMI_STATUS_INVAL` for valid arguments on every GPU; `amdsmi.h` reserves `INVAL` for a `nullptr` status pointer. Re-enable once an absent XGMI sysfs node maps to `NOT_SUPPORTED` | `GpuIntegration.XgmiErrorStatus_AllGpus` | `integration/gpu/xgmi/xgmi_test.cc` |
| `amdsmi_shut_down` | Returns `AMDSMI_STATUS_SUCCESS` when the init refcount is already zero; the test expects `AMDSMI_STATUS_INIT_ERROR` | `SystemFunctionalReadOnly.TestConcurrentInit` | `main.cc` |

## Re-enabling a test

Once the underlying issue is fixed:
1. Drop the test from `amdsmitst.exclude` if it is listed there. That file is
   what CI sources, so a test left in it stays filtered out no matter what the
   source says.
2. Delete the `AMDSMI_SKIP_KNOWN_FAILURE()` call from the test body, at the
   location given above.
3. Uncomment the reproduction stub (if present).
4. Remove the entry from this file.

## Setters with no getter to restore from

These are not failures. AMD SMI exposes no getter for these values, so the tests
below cannot record a prior value and put it back. A root run leaves them at the
documented default rather than at whatever was configured before.

| API | Test | Location |
|-----|------|----------|
| `amdsmi_set_cpu_xgmi_width` | `CpuFunctionalReadWrite.LinkSetters_Set` | `functional/cpu/link/link_read_write_test.cc` |
| `amdsmi_set_cpu_gmi3_link_width_range` | `CpuFunctionalReadWrite.LinkSetters_Set` | `functional/cpu/link/link_read_write_test.cc` |
| `amdsmi_set_cpu_core_msr_floor_freq_limit` | `CpuFunctionalReadWrite.MsrFloorFreqLimit_Set` | `functional/cpu/power/boostlimit_read_write_test.cc` |
| `amdsmi_reset_gpu_fan` | `GpuFunctionalReadWrite.FanSpeed_SetVerifyRestore` | `functional/gpu/thermal/fan_speed_read_write_test.cc` |
