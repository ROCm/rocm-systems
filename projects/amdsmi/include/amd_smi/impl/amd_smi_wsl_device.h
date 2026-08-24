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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
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
#include "amd_smi/impl/wsl/wsl_adapter.h"

namespace amd::smi {

class AMDSmiSocket;

// WSL GPU backend: implements IGPUBackend on top of the amdsmi-owned
// D3DKMT/DXCore + libwkmi.a interop (see amd_smi/impl/wsl/). One instance per
// WSL adapter.
class WSLGPUBackend : public IGPUBackend {
 public:
  // Checks /dev/dxg, initializes DXCore, enumerates+opens WSL adapters, and
  // creates AMDSmiGPUDevice + WSLGPUBackend pairs, populating sockets/processors.
  // Returns NOT_SUPPORTED if not a WSL environment.
  // Returns other error codes on WSL init failure.
  static amdsmi_status_t try_populate(std::vector<AMDSmiSocket*>& sockets,
                                      std::set<AMDSmiProcessor*>& processors);

  // Returns true if try_populate succeeded (WSL devices are in use).
  static bool is_active();

  // Closes the adapters opened by try_populate. No-op if never populated.
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

  // Device identity (derived from adapter_ at construction).
  amdsmi_bdf_t bdf() const { return bdf_; }

 private:
  explicit WSLGPUBackend(uint32_t gpu_id, const wsl::WslAdapterInfo& adapter);

  uint32_t gpu_id_;
  amdsmi_bdf_t bdf_;
  wsl::WslAdapterInfo adapter_;

  // One-time static device info snapshot -- everything here is queried once and
  // cached, as opposed to per-call dynamic sensor data (temperatures, clocks,
  // power, activity, fan speed, ...) which is always re-queried via PMLog on
  // every Get* call and therefore intentionally excluded from this struct.
  // Fields that are already cached on adapter_ (or adapter_.device_info) --
  // e.g. board_product_name, driver_version/driver_desc, memory_bus_width --
  // are read directly from adapter_ instead of being duplicated here.
  struct WslDeviceSnapshot {
    // Asic (adapter_.device_info has no vendor/subvendor/subsystem/serial
    // fields, so these are derived/defaulted here once).
    uint32_t subvendor_id = 0;
    uint64_t asic_serial = 0;
    uint64_t target_graphics_version = 0;  // packed maj/min/stepping nibbles
    uint32_t subsystem_id = 0;

    // VBIOS -- only obtainable via a dedicated wsl:: query, not cached
    // anywhere else on WslAdapterInfo.
    std::string vbios_name;
    std::string vbios_build_date;
    std::string vbios_part_number;
    std::string vbios_version;

    // Cache, shaped from adapter_.device_info.l1/l2/l3_cache_size (bytes).
    uint32_t num_cache_types = 0;
    struct CacheEntry {
      uint32_t cache_size_kb = 0;
      uint32_t cache_level = 0;
      uint32_t cache_properties = 0;
      uint32_t max_num_cu_shared = 0;
      uint32_t num_cache_instance = 0;
    } cache[AMDSMI_MAX_CACHE_TYPES];

    // Firmware, shaped from adapter_.device_info.mec_fw_version/sdma_fw_version.
    uint32_t num_fw_info = 0;
    struct FwEntry {
      amdsmi_fw_block_t fw_id = static_cast<amdsmi_fw_block_t>(0);
      uint64_t fw_version = 0;
    } fw_info_list[AMDSMI_FW_ID__MAX];

    // PCIe static capabilities (CWDDECI_CHIPSETIDENTIFICATION escape).
    uint32_t max_pcie_lane_width = 0;
    uint32_t pcie_gen = 0;  // 1-5, 0 if unknown

    // UUID seed (asic_serial if available, else bdf_.as_uint).
    uint64_t uuid_seed = 0;
  };

  mutable WslDeviceSnapshot device_info_ = {};
  mutable std::once_flag device_info_once_;
  mutable amdsmi_status_t device_info_status_ = AMDSMI_STATUS_NOT_INIT;

  amdsmi_status_t load_device_info() const;
};

}  // namespace amd::smi

#endif  // ENABLE_WSL_BACKEND
#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_WSL_DEVICE_H_
