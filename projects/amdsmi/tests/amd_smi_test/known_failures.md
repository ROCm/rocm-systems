# Known Test Failures

Tests listed here are unconditionally skipped because the underlying API returns
an unexpected status on all tested hardware. Each entry notes the symptom and the
affected test(s); root causes are under investigation unless stated otherwise.

## AMDSMI_STATUS_UNEXPECTED_DATA (error 43)

These APIs return `AMDSMI_STATUS_UNEXPECTED_DATA` where `AMDSMI_STATUS_SUCCESS` is
expected. Root cause is unknown for all entries; they likely share a driver-side
data-format issue.

| API | Skipped test(s) |
|-----|-----------------|
| `amdsmi_get_clk_freq` | `GpuIntegration.GetClkFreq_AllGpusAllTypes`, `GpuFunctionalReadOnly.TestFrequenciesRead` |
| `amdsmi_set_clk_freq` | `GpuFunctionalReadWrite.TestFrequenciesReadWrite` |
| `amdsmi_get_clk_info` | `GpuIntegration.GetClockInfo_AllGpusAllTypes` |
| `amdsmi_get_violation_status` | `GpuIntegration.GetViolationStatus_AllGpus` |
| `amdsmi_get_gpu_xcd_counter` | `GpuIntegration.GetXcdCounter_AllGpus` |
| `amdsmi_get_gpu_metrics_info` | `GpuIntegration.GetMetricsInfo_AllGpus` |
| `amdsmi_gpu_create_event`/`amdsmi_gpu_control_counter` | `GpuIntegration.CounterLifecycle_AllGpus` |
| `amdsmi_get_utilization_count` | `GpuIntegration.GetUtilizationCount_AllGpus` |
| `amdsmi_get_gpu_activity` | `GpuIntegration.GetActivity_AllGpus` |
| `amdsmi_get_energy_count` | `GpuIntegration.GetEnergyCount_AllGpus` |
| `amdsmi_get_pcie_bandwidth` | `GpuIntegration.GetPciBandwidth_AllGpus` |
| `amdsmi_set_gpu_pci_bandwidth` | `GpuFunctionalReadWrite.PciBandwidth_SetVerifyRestore` |
| `amdsmi_get_pcie_info` | `GpuIntegration.GetPcieInfo_AllGpus` |
| `amdsmi_get_xgmi_info` | `SystemIntegration.GetGpuXgmiLinkStatus_AllGpus` |
| `amdsmi_get_link_metrics` | `SystemIntegration.GetLinkMetrics_AllGpus` |
| `amdsmi_get_afids_from_cper` | `GpuIntegration.GetAfidsFromCper_DummyBuffer` |
| `amdsmi_get_temp_metric` | `GpuFunctionalReadOnly.TempRead` |
| `amdsmi_xgmi_*` (error injection) | `GpuFunctionalReadWrite.TestXGMIReadWrite` |

## AMDSMI_STATUS_UNEXPECTED_SIZE (error 42)

| API | Skipped test(s) |
|-----|-----------------|
| counter lifecycle flow | `GpuFunctionalReadOnly.Counter_LifecycleWorkflow` |

## AMDSMI_STATUS_INVAL (error 1)

| API | Skipped test(s) |
|-----|-----------------|
| `amdsmi_set_clk_freq` | `GpuFunctionalReadWrite.ClkFreq_SetVerifyRestore` |

## Library Input-Validation Bugs

These tests are skipped because the library crashes (segfault/abort) or returns
an undocumented status instead of the expected `AMDSMI_STATUS_INVAL`.

| API | Bug | Skipped test(s) |
|-----|-----|-----------------|
| `amdsmi_status_code_to_string` | Crashes on `nullptr` output pointer; should return `AMDSMI_STATUS_INVAL` | `SystemIntegration.StatusCodeToString_NullOutput` |
| `amdsmi_get_gpu_xcd_counter` | Crashes on `nullptr` output pointer; should return `AMDSMI_STATUS_INVAL` | `GpuIntegration.GetXcdCounter_NullOutput` |
| `amdsmi_gpu_control_counter` | Crashes on invalid processor handle; should return `AMDSMI_STATUS_INVAL` | `GpuIntegration.ControlCounter_InvalidHandle`, `GpuFunctionalReadOnly.ControlCounter_InvalidHandle` |
| `amdsmi_get_gpu_cper_entries` | Returns `AMDSMI_STATUS_OUT_OF_RESOURCES` for `nullptr` output instead of `AMDSMI_STATUS_INVAL` | `GpuIntegration.GetCperEntries_NullOutput` |
| `amdsmi_get_gpu_metrics_header_info` | Crashes on `nullptr` output pointer; should return `AMDSMI_STATUS_INVAL` | `GpuIntegration.GetMetricsHeaderInfo_NullOutput` |
| `amdsmi_topo_get_numa_node_number` | Crashes on `nullptr` output pointer; should return `AMDSMI_STATUS_INVAL` | `SystemIntegration.TopoGetNumaNodeNumber_NullOutput` |
| `amdsmi_get_processor_handle_from_bdf` | Returns `AMDSMI_STATUS_API_FAILED` for zero BDF; should return `NOT_FOUND` or `INVAL` | `SystemIntegration.GetProcessorHandleFromBdf_ZeroBdf` |
| `amdsmi_gpu_xgmi_error_status` | Returns `AMDSMI_STATUS_INVAL` for valid arguments on every GPU; `amdsmi.h` reserves `INVAL` for a `nullptr` status pointer. Re-enable once an absent XGMI sysfs node maps to `NOT_SUPPORTED` | `GpuIntegration.XgmiErrorStatus_AllGpus` |
| `amdsmi_shut_down` | Returns `AMDSMI_STATUS_SUCCESS` when the init refcount is already zero; the test expects `AMDSMI_STATUS_INIT_ERROR` | `SystemFunctionalReadOnly.TestConcurrentInit` |

## Re-enabling a test

Once the underlying issue is fixed:
1. Drop the test from `amdsmitst.exclude` if it is listed there. That file is
   what CI sources, so a test left in it stays filtered out no matter what the
   source says.
2. Delete the `GTEST_SKIP()` line from the test body.
3. Uncomment the reproduction stub (if present).
4. Remove the entry from this file.
