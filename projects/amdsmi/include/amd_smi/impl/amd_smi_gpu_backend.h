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

  virtual amdsmi_status_t get_asic_info(amdsmi_asic_info_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t get_board_info(amdsmi_board_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t get_kfd_info(amdsmi_kfd_info_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t get_vram_info(amdsmi_vram_info_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t get_memory_total(amdsmi_memory_type_t, uint64_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t get_memory_usage(amdsmi_memory_type_t, uint64_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t get_temp_metric(amdsmi_temperature_type_t, amdsmi_temperature_metric_t,
                                          int64_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t get_volt_metric(amdsmi_voltage_type_t, amdsmi_voltage_metric_t,
                                          int64_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t get_power_info(amdsmi_power_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t get_gpu_activity(amdsmi_engine_usage_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t get_busy_percent(uint32_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t get_clock_info(amdsmi_clk_type_t, amdsmi_clk_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t get_pcie_info(amdsmi_pcie_info_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t get_driver_info(amdsmi_driver_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t get_vbios_info(amdsmi_vbios_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t get_uuid(unsigned int*, char*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t get_gpu_cache_info(amdsmi_gpu_cache_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t get_fw_info(amdsmi_fw_info_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t get_gpu_metrics_info(amdsmi_gpu_metrics_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t get_power_cap_info(amdsmi_power_cap_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t get_fan_rpms(uint32_t, int64_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t get_fan_speed(uint32_t, int64_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t get_fan_speed_max(uint32_t, uint64_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t get_vcn_busy_percent(uint32_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t alloc_fabric_telemetry(uint32_t, amdsmi_fabric_telemetry_t**) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t get_fabric_telemetry_data(amdsmi_fabric_telemetry_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t free_fabric_telemetry(amdsmi_fabric_telemetry_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t get_gpu_fabric_info(amdsmi_fabric_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  IGPUBackend(const IGPUBackend&) = delete;
  IGPUBackend& operator=(const IGPUBackend&) = delete;

 protected:
  IGPUBackend() = default;
};

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_GPU_BACKEND_H_
