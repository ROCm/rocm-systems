// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_WSL_DEVICE_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_WSL_DEVICE_H_

#ifdef ENABLE_WSL_BACKEND

#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_gpu_backend.h"
#include "amd_smi/impl/amd_smi_processor.h"
#include "amd_smi/impl/amd_smi_socket.h"
#include "hsakmt/rocdxg_smi.h"

typedef struct _HsaNodeProperties HsaNodeProperties;

namespace amd::smi {

class AMDSmiSocket;

// WSL GPU backend: implements IGPUBackend by calling rocdxg_smi_* functions
// resolved at runtime via dlopen. One instance per GPU device.
class WSLGPUBackend : public IGPUBackend {
 public:
  // Checks /dev/dxg, loads librocdxg, enumerates GPU nodes, creates
  // AMDSmiGPUDevice + WSLGPUBackend pairs, and populates sockets/processors.
  // Returns NOT_SUPPORTED if not a WSL environment.
  // Returns other error codes on WSL init failure.
  static amdsmi_status_t try_populate(std::vector<AMDSmiSocket*>& sockets,
                                      std::set<AMDSmiProcessor*>& processors);

  // Returns true if try_populate succeeded (WSL devices are in use).
  static bool is_active();

  // Closes the KFD channel opened by try_populate. No-op if never populated.
  static amdsmi_status_t shutdown();

  // IGPUBackend overrides
  amdsmi_status_t get_asic_info(amdsmi_asic_info_t*) override;
  amdsmi_status_t get_board_info(amdsmi_board_info_t*) override;
  amdsmi_status_t get_kfd_info(amdsmi_kfd_info_t*) override;
  amdsmi_status_t get_vram_info(amdsmi_vram_info_t*) override;
  amdsmi_status_t get_memory_total(amdsmi_memory_type_t, uint64_t*) override;
  amdsmi_status_t get_memory_usage(amdsmi_memory_type_t, uint64_t*) override;
  amdsmi_status_t get_temp_metric(amdsmi_temperature_type_t, amdsmi_temperature_metric_t,
                                  int64_t*) override;
  amdsmi_status_t get_volt_metric(amdsmi_voltage_type_t, amdsmi_voltage_metric_t,
                                  int64_t*) override;
  amdsmi_status_t get_power_info(amdsmi_power_info_t*) override;
  amdsmi_status_t get_gpu_activity(amdsmi_engine_usage_t*) override;
  amdsmi_status_t get_busy_percent(uint32_t*) override;
  amdsmi_status_t get_clock_info(amdsmi_clk_type_t, amdsmi_clk_info_t*) override;
  amdsmi_status_t get_pcie_info(amdsmi_pcie_info_t*) override;
  amdsmi_status_t get_driver_info(amdsmi_driver_info_t*) override;
  amdsmi_status_t get_vbios_info(amdsmi_vbios_info_t*) override;
  amdsmi_status_t get_uuid(unsigned int*, char*) override;
  amdsmi_status_t get_gpu_cache_info(amdsmi_gpu_cache_info_t*) override;
  amdsmi_status_t get_fw_info(amdsmi_fw_info_t*) override;
  amdsmi_status_t get_gpu_metrics_info(amdsmi_gpu_metrics_t*) override;
  amdsmi_status_t get_power_cap_info(amdsmi_power_cap_info_t*) override;
  amdsmi_status_t get_fan_rpms(uint32_t sensor_ind, int64_t* speed) override;
  amdsmi_status_t get_fan_speed(uint32_t sensor_ind, int64_t* speed) override;
  amdsmi_status_t get_fan_speed_max(uint32_t sensor_ind, uint64_t* max_speed) override;

  // Device identity (populated from HsaNodeProperties at construction).
  uint32_t node_id() const { return node_id_; }
  amdsmi_bdf_t bdf() const { return bdf_; }

 private:
  explicit WSLGPUBackend(uint32_t gpu_id, uint32_t node_id, const HsaNodeProperties& props);

  uint32_t gpu_id_;
  uint32_t node_id_;
  uint16_t vendor_id_;
  uint16_t device_id_;
  uint32_t family_id_;
  uint32_t num_compute_units_;
  uint32_t num_xcc_;
  uint64_t unique_id_;
  uint64_t local_mem_size_;
  amdsmi_bdf_t bdf_;
  std::string marketing_name_;

  // Lazily-loaded aggregate static device info from rocdxg_smi_get_device_info().
  mutable rocdxg_smi_device_info_t device_info_ = {};
  mutable std::once_flag device_info_once_;
  mutable amdsmi_status_t device_info_status_ = AMDSMI_STATUS_NOT_INIT;

  amdsmi_status_t load_device_info() const;
};

}  // namespace amd::smi

#endif  // ENABLE_WSL_BACKEND
#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_WSL_DEVICE_H_
