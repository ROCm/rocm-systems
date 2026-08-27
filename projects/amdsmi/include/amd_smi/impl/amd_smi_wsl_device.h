/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

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
#include "amd_smi/impl/wsl/rocdxg_abi.h"

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

  // Device identity, cached from rocdxg_smi_get_device_info() at construction.
  uint32_t node_id() const { return node_id_; }
  amdsmi_bdf_t bdf() const { return bdf_; }

 private:
  WSLGPUBackend(uint32_t gpu_id, uint32_t node_id, const rocdxg_smi_device_info_t& info);

  uint32_t gpu_id_;
  uint32_t node_id_;
  uint64_t device_id_;
  uint64_t unique_id_;
  amdsmi_bdf_t bdf_;

  // Aggregate static device info, fetched once during enumeration.
  rocdxg_smi_device_info_t device_info_ = {};
};

}  // namespace amd::smi

#endif  // ENABLE_WSL_BACKEND
#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_WSL_DEVICE_H_
