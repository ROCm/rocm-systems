# Known Test Failures

Tests listed here are skipped because the underlying API misbehaves on all tested
hardware. Each entry names the symptom, the affected test, and the file holding
the skip. Root causes are under investigation unless stated otherwise.

Paths in the **Location** column are relative to `tests/amd_smi_test/`. The skip
is the `AMDSMI_SKIP_KNOWN_FAILURE()` call in the named test, at the top of the
body unless the entry says otherwise.

## Running the skipped tests

Set `AMDSMI_RUN_KNOWN_FAILURES` to run every test in this file instead of
skipping it, so its current behavior can be checked without editing source:

```shell
AMDSMI_RUN_KNOWN_FAILURES=1 ./amdsmitst
```

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
| `amdsmi_get_clk_freq` | `GpuFunctionalReadOnly.TestFrequenciesRead` | `functional/gpu/clock/frequencies_read_test.cc` |
| `amdsmi_set_clk_freq` | `GpuFunctionalReadWrite.TestFrequenciesReadWrite` | `functional/gpu/clock/frequencies_read_write_test.cc` |
| `amdsmi_get_clk_info` | `GpuIntegration.GetClockInfo_AllGpusAllTypes` | `integration/gpu/clock/clock_freq_test.cc` |
| `amdsmi_get_violation_status` | `GpuIntegration.GetViolationStatus_AllGpus` | `integration/gpu/events/event_ptl_test.cc` |
| `amdsmi_get_gpu_xcd_counter` | `GpuIntegration.GetXcdCounter_AllGpus` | `integration/gpu/identity/id_info_test.cc` |
| `amdsmi_get_gpu_metrics_info` | `GpuIntegration.GetMetricsInfo_AllGpus` | `integration/gpu/metrics/metrics_test.cc` |
| `amdsmi_gpu_create_event`/`amdsmi_gpu_control_counter` | `GpuIntegration.CounterLifecycle_AllGpus` | `integration/gpu/perf/counter_test.cc` |
| `amdsmi_get_utilization_count` | `GpuIntegration.GetUtilizationCount_AllGpus` | `integration/gpu/perf/perf_overdrive_test.cc` |
| `amdsmi_get_gpu_activity` | `GpuIntegration.GetActivity_AllGpus` | `integration/gpu/perf/perf_overdrive_test.cc` |
| `amdsmi_get_energy_count` | `GpuIntegration.GetEnergyCount_AllGpus` | `integration/gpu/power/power_test.cc` |
| `amdsmi_get_pcie_bandwidth` | `GpuIntegration.GetPciBandwidth_AllGpus` | `integration/gpu/pci/pci_test.cc` |
| `amdsmi_get_pcie_info` | `GpuIntegration.GetPcieInfo_AllGpus` | `integration/gpu/pci/pci_test.cc` |
| `amdsmi_get_xgmi_info` | `SystemIntegration.GetGpuXgmiLinkStatus_AllGpus` | `integration/system/topology_test.cc` |
| `amdsmi_get_link_metrics` | `SystemIntegration.GetLinkMetrics_AllGpus` | `integration/system/topology_test.cc` |
| `amdsmi_get_afids_from_cper` | `GpuIntegration.GetAfidsFromCper_DummyBuffer` | `integration/gpu/ras/ras_ecc_test.cc` |
| `amdsmi_get_temp_metric` | `GpuIntegration.GetTempMetric_AllGpusAllTypesMetrics` | `integration/gpu/thermal/thermal_fan_test.cc` |
| `amdsmi_get_temp_metric` | `GpuFunctionalReadOnly.TempRead` | `functional/gpu/thermal/temp_read_test.cc` |
| `amdsmi_xgmi_*` (error injection) | `GpuFunctionalReadWrite.TestXGMIReadWrite` | `functional/gpu/xgmi/xgmi_read_write_test.cc` |

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
| `amdsmi_status_code_to_string` | Crashes on `nullptr` output pointer; should return `AMDSMI_STATUS_INVAL` | `SystemIntegration.StatusCodeToString_NullOutput` | `integration/system/init_test.cc` |
| `amdsmi_get_gpu_xcd_counter` | Crashes on `nullptr` output pointer; should return `AMDSMI_STATUS_INVAL` | `GpuIntegration.GetXcdCounter_NullOutput` | `integration/gpu/identity/id_info_test.cc` |
| `amdsmi_gpu_control_counter` | Crashes on invalid processor handle; should return `AMDSMI_STATUS_INVAL` | `GpuIntegration.ControlCounter_InvalidHandle` | `integration/gpu/perf/counter_test.cc` |
| `amdsmi_get_gpu_cper_entries` | Returns `AMDSMI_STATUS_OUT_OF_RESOURCES` for `nullptr` output instead of `AMDSMI_STATUS_INVAL` | `GpuIntegration.GetCperEntries_NullOutput` | `integration/gpu/ras/ras_ecc_test.cc` |
| `amdsmi_get_gpu_metrics_header_info` | Crashes on `nullptr` output pointer; should return `AMDSMI_STATUS_INVAL` | `GpuIntegration.GetMetricsHeaderInfo_NullOutput` | `integration/gpu/metrics/metrics_test.cc` |
| `amdsmi_get_gpu_pci_throughput` | Returns `AMDSMI_STATUS_SUCCESS` for `nullptr` output pointers; should return `AMDSMI_STATUS_INVAL` | `GpuIntegration.GetPciThroughput_NullOutput` | `integration/gpu/pci/pci_test.cc` |
| `amdsmi_topo_get_numa_node_number` | Crashes on `nullptr` output pointer; should return `AMDSMI_STATUS_INVAL` | `SystemIntegration.TopoGetNumaNodeNumber_NullOutput` | `integration/system/topology_test.cc` |
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
