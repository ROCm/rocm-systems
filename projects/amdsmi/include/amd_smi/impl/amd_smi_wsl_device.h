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

#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_gpu_device.h"

typedef struct _HsaNodeProperties HsaNodeProperties;

namespace amd::smi {

class AMDSmiWslGPUDevice : public AMDSmiGPUDevice {
 public:
  AMDSmiWslGPUDevice(uint32_t gpu_id, uint32_t node_id, const HsaNodeProperties& props,
                     AMDSmiDrm& drm);

  bool is_wsl_device() const override { return true; }
  uint32_t node_id() const { return node_id_; }
  uint64_t unique_id() const { return unique_id_; }
  uint32_t num_compute_units() const { return num_compute_units_; }
  uint32_t num_xcc() const { return num_xcc_; }
  uint64_t local_mem_size() const { return local_mem_size_; }
  const std::string& marketing_name() const { return marketing_name_; }

  amdsmi_status_t get_kfd_info(amdsmi_kfd_info_t* info) const;
  amdsmi_status_t get_asic_info(amdsmi_asic_info_t* info) const;
  amdsmi_status_t get_board_info(amdsmi_board_info_t* info) const;
  amdsmi_status_t get_vram_info(amdsmi_vram_info_t* info) const;
  amdsmi_status_t get_memory_total(amdsmi_memory_type_t mem_type, uint64_t* total) const;
  amdsmi_status_t get_memory_usage(amdsmi_memory_type_t mem_type, uint64_t* used) const;
  amdsmi_status_t get_temp_metric(amdsmi_temperature_type_t sensor_type,
                                  amdsmi_temperature_metric_t metric, int64_t* temperature) const;
  amdsmi_status_t get_volt_metric(amdsmi_voltage_type_t sensor_type,
                                  amdsmi_voltage_metric_t metric, int64_t* voltage) const;
  amdsmi_status_t get_power_info(amdsmi_power_info_t* info) const;
  amdsmi_status_t get_busy_percent(uint32_t* gpu_busy_percent) const;
  amdsmi_status_t get_activity(amdsmi_engine_usage_t* info) const;
  amdsmi_status_t get_clock_info(amdsmi_clk_type_t clk_type, amdsmi_clk_info_t* info) const;
  amdsmi_status_t get_pcie_info(amdsmi_pcie_info_t* info) const;
  amdsmi_status_t get_driver_info(amdsmi_driver_info_t* info) const;
  amdsmi_status_t get_vbios_info(amdsmi_vbios_info_t* info) const;
  amdsmi_status_t get_process_list(std::vector<amdsmi_proc_info_t>* processes) const;
  amdsmi_status_t get_uuid(unsigned int* uuid_length, char* uuid) const;

 private:
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
};

bool is_wsl_gpu_device(const AMDSmiGPUDevice* device);
AMDSmiWslGPUDevice* as_wsl_gpu_device(AMDSmiGPUDevice* device);
const AMDSmiWslGPUDevice* as_wsl_gpu_device(const AMDSmiGPUDevice* device);

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_WSL_DEVICE_H_
