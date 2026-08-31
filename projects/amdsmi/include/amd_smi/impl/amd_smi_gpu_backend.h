// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_GPU_BACKEND_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_GPU_BACKEND_H_

#include <vector>

#include "amd_smi/amdsmi.h"

namespace amd::smi {

// Per-GPU backend abstraction. amd_smi.cc dispatches to this instead of
// branching on platform identity. Non-pure virtuals return NOT_SUPPORTED so
// each backend only overrides what it actually implements.
class IGPUBackend {
 public:
  virtual ~IGPUBackend() = default;

  virtual amdsmi_status_t GetAsicInfo(amdsmi_asic_info_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t GetBoardInfo(amdsmi_board_info_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t GetKfdInfo(amdsmi_kfd_info_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t GetVramInfo(amdsmi_vram_info_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t GetMemoryTotal(amdsmi_memory_type_t, uint64_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetMemoryUsage(amdsmi_memory_type_t, uint64_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetTempMetric(amdsmi_temperature_type_t, amdsmi_temperature_metric_t,
                                        int64_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetVoltMetric(amdsmi_voltage_type_t, amdsmi_voltage_metric_t, int64_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetPowerInfo(amdsmi_power_info_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t GetGpuActivity(amdsmi_engine_usage_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetBusyPercent(uint32_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t GetClockInfo(amdsmi_clk_type_t, amdsmi_clk_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetPcieInfo(amdsmi_pcie_info_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t GetDriverInfo(amdsmi_driver_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetVbiosInfo(amdsmi_vbios_info_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t GetUuid(unsigned int*, char*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t GetGpuCacheInfo(amdsmi_gpu_cache_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetFwInfo(amdsmi_fw_info_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t GetGpuMetricsInfo(amdsmi_gpu_metrics_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetPowerCapInfo(amdsmi_power_cap_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetFanRpms(uint32_t, int64_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t GetFanSpeed(uint32_t, int64_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t GetFanSpeedMax(uint32_t, uint64_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetVcnBusyPercent(uint32_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t AllocFabricTelemetry(uint32_t, amdsmi_fabric_telemetry_t**) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetFabricTelemetryData(amdsmi_fabric_telemetry_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t FreeFabricTelemetry(amdsmi_fabric_telemetry_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetGpuFabricInfo(amdsmi_fabric_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  IGPUBackend(const IGPUBackend&) = delete;
  IGPUBackend& operator=(const IGPUBackend&) = delete;

 protected:
  IGPUBackend() = default;
};

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_GPU_BACKEND_H_
