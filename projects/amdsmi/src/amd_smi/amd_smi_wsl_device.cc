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

#include "amd_smi/impl/amd_smi_wsl_device.h"
#include "amd_smi/impl/amd_smi_wsl_syms.h"

#if __has_include(<hsakmt/rocdxg_smi.h>)
#define AMDSMI_HAS_ROCDXG_SMI 1
#endif

#include <cstdio>
#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>
#include <unistd.h>

#include "amd_smi/impl/amd_smi_uuid.h"

namespace amd::smi {
namespace {

amdsmi_bdf_t make_bdf(const HsaNodeProperties& props) {
  amdsmi_bdf_t bdf = {};
  bdf.bdf.domain_number = props.Domain;
  bdf.bdf.bus_number = (props.LocationId >> 8) & 0xff;
  bdf.bdf.device_number = (props.LocationId >> 3) & 0x1f;
  bdf.bdf.function_number = props.LocationId & 0x7;
  return bdf;
}

std::string marketing_name_from_hsa(const HsaNodeProperties& props) {
  std::string name;
  for (auto ch : props.MarketingName) {
    if (ch == 0) break;
    name.push_back(static_cast<char>(ch));
  }
  if (!name.empty()) return name;

  for (auto ch : props.AMDName) {
    if (ch == 0) break;
    name.push_back(static_cast<char>(ch));
  }
  return name;
}

void copy_string(char* dst, const std::string& src) {
  if (dst == nullptr) return;
  std::memset(dst, 0, AMDSMI_MAX_STRING_LENGTH);
  std::snprintf(dst, AMDSMI_MAX_STRING_LENGTH, "%s", src.c_str());
}

#ifdef AMDSMI_HAS_ROCDXG_SMI
amdsmi_status_t rocdxg_to_amdsmi_status(HSAKMT_STATUS status) {
  switch (status) {
    case HSAKMT_STATUS_SUCCESS:
      return AMDSMI_STATUS_SUCCESS;
    case HSAKMT_STATUS_INVALID_PARAMETER:
    case HSAKMT_STATUS_INVALID_HANDLE:
    case HSAKMT_STATUS_INVALID_NODE_UNIT:
      return AMDSMI_STATUS_INVAL;
    case HSAKMT_STATUS_NO_MEMORY:
      return AMDSMI_STATUS_OUT_OF_RESOURCES;
    case HSAKMT_STATUS_BUFFER_TOO_SMALL:
      return AMDSMI_STATUS_INSUFFICIENT_SIZE;
    case HSAKMT_STATUS_NOT_IMPLEMENTED:
      return AMDSMI_STATUS_NOT_YET_IMPLEMENTED;
    case HSAKMT_STATUS_NOT_SUPPORTED:
      return AMDSMI_STATUS_NOT_SUPPORTED;
    case HSAKMT_STATUS_UNAVAILABLE:
      return AMDSMI_STATUS_SETTING_UNAVAILABLE;
    case HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED:
    case HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR:
      return AMDSMI_STATUS_DRIVER_NOT_LOADED;
    default:
      return AMDSMI_STATUS_API_FAILED;
  }
}

void copy_rocdxg_string(char* dst, const char* src) {
  if (dst == nullptr) return;
  std::memset(dst, 0, AMDSMI_MAX_STRING_LENGTH);
  if (src != nullptr) std::snprintf(dst, AMDSMI_MAX_STRING_LENGTH, "%s", src);
}
#endif

}  // namespace

AMDSmiWslGPUDevice::AMDSmiWslGPUDevice(uint32_t gpu_id, uint32_t node_id,
                                       const HsaNodeProperties& props, AMDSmiDrm& drm)
    : AMDSmiGPUDevice(gpu_id, "wsl_node" + std::to_string(node_id), make_bdf(props), drm),
      node_id_(node_id),
      vendor_id_(props.VendorId),
      device_id_(props.DeviceId),
      family_id_(props.FamilyID),
      num_compute_units_(props.NumCUPerArray * props.NumArrays),
      num_xcc_(props.NumXcc),
      unique_id_(props.UniqueID),
      local_mem_size_(props.LocalMemSize),
      bdf_(make_bdf(props)),
      marketing_name_(marketing_name_from_hsa(props)) {}

amdsmi_status_t AMDSmiWslGPUDevice::get_kfd_info(amdsmi_kfd_info_t* info) const {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  std::memset(info, 0, sizeof(*info));
  info->kfd_id = unique_id_ ? unique_id_ : std::numeric_limits<uint64_t>::max();
  info->node_id = node_id_;
  info->current_partition_id = std::numeric_limits<uint32_t>::max();
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiWslGPUDevice::get_asic_info(amdsmi_asic_info_t* info) const {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
#ifdef AMDSMI_HAS_ROCDXG_SMI
  rocdxg_smi_asic_info_t rocdxg_info = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_asic_info(node_id_, &rocdxg_info);
  if (hstatus == HSAKMT_STATUS_SUCCESS) {
    std::memset(info, 0, sizeof(*info));
    copy_rocdxg_string(info->market_name, rocdxg_info.market_name);
    info->vendor_id = rocdxg_info.vendor_id;
    if (info->vendor_id == 0x1002) {
      copy_string(info->vendor_name, "Advanced Micro Devices, Inc. [AMD/ATI]");
    }
    info->subvendor_id = rocdxg_info.subvendor_id;
    info->device_id = rocdxg_info.device_id;
    info->rev_id = rocdxg_info.rev_id;
    std::snprintf(info->asic_serial, AMDSMI_MAX_STRING_LENGTH, "%016lx",
                  rocdxg_info.asic_serial);
    info->oam_id = std::numeric_limits<uint32_t>::max();
    info->num_of_compute_units = rocdxg_info.num_of_compute_units;
    info->target_graphics_version = rocdxg_info.target_graphics_version;
    info->subsystem_id = rocdxg_info.subsystem_id;
    return AMDSMI_STATUS_SUCCESS;
  }
  if (hstatus != HSAKMT_STATUS_NOT_SUPPORTED) return rocdxg_to_amdsmi_status(hstatus);
#endif
  std::memset(info, 0, sizeof(*info));
  copy_string(info->market_name, marketing_name_);
  info->vendor_id = vendor_id_;
  if (vendor_id_ == 0x1002) {
    copy_string(info->vendor_name, "Advanced Micro Devices, Inc. [AMD/ATI]");
  }
  info->subvendor_id = std::numeric_limits<uint32_t>::max();
  info->device_id = device_id_;
  info->rev_id = std::numeric_limits<uint32_t>::max();
  copy_string(info->asic_serial, "ffffffffffffffff");
  info->oam_id = std::numeric_limits<uint32_t>::max();
  info->num_of_compute_units =
      num_compute_units_ ? num_compute_units_ : std::numeric_limits<uint32_t>::max();
  info->target_graphics_version = std::numeric_limits<uint64_t>::max();
  info->subsystem_id = std::numeric_limits<uint32_t>::max();
  info->flags = family_id_;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiWslGPUDevice::get_board_info(amdsmi_board_info_t* info) const {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
#ifdef AMDSMI_HAS_ROCDXG_SMI
  rocdxg_smi_board_info_t rocdxg_info = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_board_info(node_id_, &rocdxg_info);
  if (hstatus == HSAKMT_STATUS_SUCCESS) {
    std::memset(info, 0, sizeof(*info));
    copy_rocdxg_string(info->product_name, rocdxg_info.product_name);
    copy_rocdxg_string(info->manufacturer_name, rocdxg_info.manufacturer_name);
    return AMDSMI_STATUS_SUCCESS;
  }
  if (hstatus != HSAKMT_STATUS_NOT_SUPPORTED) return rocdxg_to_amdsmi_status(hstatus);
#endif
  std::memset(info, 0, sizeof(*info));
  copy_string(info->product_name, marketing_name_);
  copy_string(info->manufacturer_name, "Advanced Micro Devices, Inc. [AMD/ATI]");
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiWslGPUDevice::get_vram_info(amdsmi_vram_info_t* info) const {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
#ifdef AMDSMI_HAS_ROCDXG_SMI
  rocdxg_smi_vram_info_t rocdxg_info = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_vram_info(node_id_, &rocdxg_info);
  if (hstatus == HSAKMT_STATUS_SUCCESS) {
    std::memset(info, 0, sizeof(*info));
    info->vram_type = AMDSMI_VRAM_TYPE_UNKNOWN;
    copy_string(info->vram_vendor, "UNKNOWN");
    info->vram_size = rocdxg_info.vram_size_mb;
    info->vram_bit_width = rocdxg_info.vram_bit_width;
    info->vram_max_bandwidth = std::numeric_limits<decltype(info->vram_max_bandwidth)>::max();
    return AMDSMI_STATUS_SUCCESS;
  }
  if (hstatus != HSAKMT_STATUS_NOT_SUPPORTED) return rocdxg_to_amdsmi_status(hstatus);
#endif
  std::memset(info, 0, sizeof(*info));
  info->vram_type = AMDSMI_VRAM_TYPE_UNKNOWN;
  copy_string(info->vram_vendor, "UNKNOWN");
  info->vram_size = local_mem_size_ / (1024 * 1024);
  info->vram_bit_width = std::numeric_limits<decltype(info->vram_bit_width)>::max();
  info->vram_max_bandwidth = std::numeric_limits<decltype(info->vram_max_bandwidth)>::max();
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiWslGPUDevice::get_memory_total(amdsmi_memory_type_t mem_type,
                                                     uint64_t* total) const {
  if (total == nullptr) return AMDSMI_STATUS_INVAL;
  if (mem_type != AMDSMI_MEM_TYPE_VRAM && mem_type != AMDSMI_MEM_TYPE_VIS_VRAM) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

#ifdef AMDSMI_HAS_ROCDXG_SMI
  rocdxg_smi_vram_info_t rocdxg_info = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_vram_info(node_id_, &rocdxg_info);
  if (hstatus == HSAKMT_STATUS_SUCCESS) {
    *total = rocdxg_info.vram_size_mb * 1024 * 1024;
    return AMDSMI_STATUS_SUCCESS;
  }
  if (hstatus != HSAKMT_STATUS_NOT_SUPPORTED) return rocdxg_to_amdsmi_status(hstatus);
#endif
  *total = local_mem_size_;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiWslGPUDevice::get_memory_usage(amdsmi_memory_type_t mem_type,
                                                     uint64_t* used) const {
  if (used == nullptr) return AMDSMI_STATUS_INVAL;
  if (mem_type != AMDSMI_MEM_TYPE_VRAM && mem_type != AMDSMI_MEM_TYPE_VIS_VRAM) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

#ifdef AMDSMI_HAS_ROCDXG_SMI
  rocdxg_smi_vram_usage_t usage = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_vram_usage(node_id_, &usage);
  if (hstatus == HSAKMT_STATUS_SUCCESS) {
    *used = usage.vram_used_mb * 1024 * 1024;
    return AMDSMI_STATUS_SUCCESS;
  }
  return rocdxg_to_amdsmi_status(hstatus);
#else
  (void)mem_type;
  return AMDSMI_STATUS_NOT_SUPPORTED;
#endif
}

amdsmi_status_t AMDSmiWslGPUDevice::get_temp_metric(amdsmi_temperature_type_t sensor_type,
                                                    amdsmi_temperature_metric_t metric,
                                                    int64_t* temperature) const {
  if (temperature == nullptr) return AMDSMI_STATUS_INVAL;
#ifdef AMDSMI_HAS_ROCDXG_SMI
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_temperature(
      node_id_, static_cast<uint32_t>(sensor_type), static_cast<uint32_t>(metric), temperature);
  return rocdxg_to_amdsmi_status(hstatus);
#else
  (void)sensor_type;
  (void)metric;
  return AMDSMI_STATUS_NOT_SUPPORTED;
#endif
}

amdsmi_status_t AMDSmiWslGPUDevice::get_volt_metric(amdsmi_voltage_type_t sensor_type,
                                                    amdsmi_voltage_metric_t metric,
                                                    int64_t* voltage) const {
  if (voltage == nullptr) return AMDSMI_STATUS_INVAL;
#ifdef AMDSMI_HAS_ROCDXG_SMI
  if (metric != AMDSMI_VOLT_CURRENT) return AMDSMI_STATUS_NOT_SUPPORTED;
  rocdxg_smi_power_info_t power = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_power_info(node_id_, &power);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return rocdxg_to_amdsmi_status(hstatus);
  switch (sensor_type) {
    case AMDSMI_VOLT_TYPE_VDDGFX:
      *voltage = power.gfx_voltage;
      return AMDSMI_STATUS_SUCCESS;
    default:
      return AMDSMI_STATUS_NOT_SUPPORTED;
  }
#else
  (void)sensor_type;
  (void)metric;
  return AMDSMI_STATUS_NOT_SUPPORTED;
#endif
}

amdsmi_status_t AMDSmiWslGPUDevice::get_power_info(amdsmi_power_info_t* info) const {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  std::memset(info, 0, sizeof(*info));
  info->socket_power = std::numeric_limits<decltype(info->socket_power)>::max();
  info->current_socket_power =
      std::numeric_limits<decltype(info->current_socket_power)>::max();
  info->average_socket_power =
      std::numeric_limits<decltype(info->average_socket_power)>::max();
  info->gfx_voltage = std::numeric_limits<decltype(info->gfx_voltage)>::max();
  info->soc_voltage = std::numeric_limits<decltype(info->soc_voltage)>::max();
  info->mem_voltage = std::numeric_limits<decltype(info->mem_voltage)>::max();
  info->power_limit = std::numeric_limits<decltype(info->power_limit)>::max();
  info->ubb_power = std::numeric_limits<decltype(info->ubb_power)>::max();
#ifdef AMDSMI_HAS_ROCDXG_SMI
  rocdxg_smi_power_info_t power = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_power_info(node_id_, &power);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return rocdxg_to_amdsmi_status(hstatus);
  info->current_socket_power = power.current_socket_power;
  info->average_socket_power = power.current_socket_power;
  info->socket_power = power.current_socket_power;
  info->gfx_voltage = power.gfx_voltage;
  info->soc_voltage = power.soc_voltage;
  info->mem_voltage = power.mem_voltage;
  info->power_limit = power.power_limit;
  return AMDSMI_STATUS_SUCCESS;
#else
  return AMDSMI_STATUS_NOT_SUPPORTED;
#endif
}

amdsmi_status_t AMDSmiWslGPUDevice::get_busy_percent(uint32_t* gpu_busy_percent) const {
  if (gpu_busy_percent == nullptr) return AMDSMI_STATUS_INVAL;
#ifdef AMDSMI_HAS_ROCDXG_SMI
  rocdxg_smi_gpu_metrics_info_t metrics = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_gpu_metrics_info(node_id_, &metrics);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return rocdxg_to_amdsmi_status(hstatus);
  *gpu_busy_percent = metrics.average_gfx_activity;
  return AMDSMI_STATUS_SUCCESS;
#else
  return AMDSMI_STATUS_NOT_SUPPORTED;
#endif
}

amdsmi_status_t AMDSmiWslGPUDevice::get_activity(amdsmi_engine_usage_t* info) const {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
#ifdef AMDSMI_HAS_ROCDXG_SMI
  rocdxg_smi_gpu_metrics_info_t metrics = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_gpu_metrics_info(node_id_, &metrics);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return rocdxg_to_amdsmi_status(hstatus);
  std::memset(info, 0, sizeof(*info));
  info->gfx_activity = metrics.average_gfx_activity;
  info->umc_activity = metrics.average_umc_activity;
  return AMDSMI_STATUS_SUCCESS;
#else
  return AMDSMI_STATUS_NOT_SUPPORTED;
#endif
}

amdsmi_status_t AMDSmiWslGPUDevice::get_clock_info(amdsmi_clk_type_t clk_type,
                                                   amdsmi_clk_info_t* info) const {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
#ifdef AMDSMI_HAS_ROCDXG_SMI
  rocdxg_smi_clock_info_t rocdxg_info = {};
  HSAKMT_STATUS hstatus =
      g_wsl_syms.rocdxg_smi_get_clock_info(node_id_, static_cast<uint32_t>(clk_type), &rocdxg_info);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return rocdxg_to_amdsmi_status(hstatus);
  std::memset(info, 0, sizeof(*info));
  info->clk = rocdxg_info.clk;
  info->min_clk = rocdxg_info.min_clk;
  info->max_clk = rocdxg_info.max_clk;
  info->clk_locked = rocdxg_info.clk_locked;
  info->clk_deep_sleep = rocdxg_info.clk_deep_sleep;
  return AMDSMI_STATUS_SUCCESS;
#else
  (void)clk_type;
  return AMDSMI_STATUS_NOT_SUPPORTED;
#endif
}

amdsmi_status_t AMDSmiWslGPUDevice::get_pcie_info(amdsmi_pcie_info_t* info) const {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
#ifdef AMDSMI_HAS_ROCDXG_SMI
  rocdxg_smi_pcie_info_t rocdxg_info = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_pcie_info(node_id_, &rocdxg_info);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return rocdxg_to_amdsmi_status(hstatus);
  std::memset(info, 0, sizeof(*info));
  info->pcie_static.max_pcie_width = rocdxg_info.max_pcie_width;
  info->pcie_static.max_pcie_speed = rocdxg_info.max_pcie_speed;
  info->pcie_static.pcie_interface_version = rocdxg_info.pcie_interface_version;
  info->pcie_static.slot_type = static_cast<amdsmi_card_form_factor_t>(rocdxg_info.slot_type);
  info->pcie_metric.pcie_width = rocdxg_info.pcie_width;
  info->pcie_metric.pcie_speed = rocdxg_info.pcie_speed;
  info->pcie_metric.pcie_bandwidth = rocdxg_info.pcie_bandwidth;
  info->pcie_metric.pcie_replay_count = rocdxg_info.pcie_replay_count;
  info->pcie_metric.pcie_l0_to_recovery_count = rocdxg_info.pcie_l0_to_recovery_count;
  info->pcie_metric.pcie_replay_roll_over_count = rocdxg_info.pcie_replay_roll_over_count;
  info->pcie_metric.pcie_nak_sent_count = rocdxg_info.pcie_nak_sent_count;
  info->pcie_metric.pcie_nak_received_count = rocdxg_info.pcie_nak_received_count;
  return AMDSMI_STATUS_SUCCESS;
#else
  return AMDSMI_STATUS_NOT_SUPPORTED;
#endif
}

amdsmi_status_t AMDSmiWslGPUDevice::get_driver_info(amdsmi_driver_info_t* info) const {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
#ifdef AMDSMI_HAS_ROCDXG_SMI
  rocdxg_smi_driver_info_t rocdxg_info = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_driver_info(node_id_, &rocdxg_info);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return rocdxg_to_amdsmi_status(hstatus);
  std::memset(info, 0, sizeof(*info));
  copy_rocdxg_string(info->driver_version, rocdxg_info.driver_version);
  copy_rocdxg_string(info->driver_date, rocdxg_info.driver_date);
  copy_rocdxg_string(info->driver_name, rocdxg_info.driver_name);
  return AMDSMI_STATUS_SUCCESS;
#else
  return AMDSMI_STATUS_NOT_SUPPORTED;
#endif
}

amdsmi_status_t AMDSmiWslGPUDevice::get_vbios_info(amdsmi_vbios_info_t* info) const {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
#ifdef AMDSMI_HAS_ROCDXG_SMI
  rocdxg_smi_vbios_info_t rocdxg_info = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_vbios_info(node_id_, &rocdxg_info);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return rocdxg_to_amdsmi_status(hstatus);
  std::memset(info, 0, sizeof(*info));
  copy_rocdxg_string(info->name, rocdxg_info.name);
  copy_rocdxg_string(info->build_date, rocdxg_info.build_date);
  copy_rocdxg_string(info->part_number, rocdxg_info.part_number);
  copy_rocdxg_string(info->version, rocdxg_info.version);
  copy_rocdxg_string(info->boot_firmware, rocdxg_info.boot_firmware);
  return AMDSMI_STATUS_SUCCESS;
#else
  return AMDSMI_STATUS_NOT_SUPPORTED;
#endif
}

amdsmi_status_t AMDSmiWslGPUDevice::get_process_list(
    std::vector<amdsmi_proc_info_t>* processes) const {
  if (processes == nullptr) return AMDSMI_STATUS_INVAL;
  processes->clear();
#ifdef AMDSMI_HAS_ROCDXG_SMI
  uint32_t count = 0;
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_enum_processes(node_id_, &count, nullptr);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return rocdxg_to_amdsmi_status(hstatus);
  if (count == 0) return AMDSMI_STATUS_SUCCESS;

  std::vector<rocdxg_smi_process_info_t> rocdxg_processes(count);
  hstatus = g_wsl_syms.rocdxg_smi_enum_processes(node_id_, &count, rocdxg_processes.data());
  if (hstatus != HSAKMT_STATUS_SUCCESS) return rocdxg_to_amdsmi_status(hstatus);

  processes->reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    amdsmi_proc_info_t out = {};
    out.pid = rocdxg_processes[i].process_id;
    out.mem = rocdxg_processes[i].vram_usage_bytes;
    out.memory_usage.vram_mem = rocdxg_processes[i].vram_usage_bytes;
    out.sdma_usage = rocdxg_processes[i].sdma_usage;
    out.cu_occupancy = static_cast<uint32_t>(
        std::min<uint64_t>(rocdxg_processes[i].cu_occupancy,
                           std::numeric_limits<uint32_t>::max()));
    out.evicted_time = static_cast<uint32_t>(
        std::min<uint64_t>(rocdxg_processes[i].evicted_time,
                           std::numeric_limits<uint32_t>::max()));

    char exe_path[64] = {};
    std::snprintf(exe_path, sizeof(exe_path), "/proc/%u/exe", out.pid);
    ssize_t len = readlink(exe_path, out.name, AMDSMI_MAX_STRING_LENGTH - 1);
    if (len > 0) out.name[len] = '\0';
    processes->push_back(out);
  }
  return AMDSMI_STATUS_SUCCESS;
#else
  return AMDSMI_STATUS_NOT_SUPPORTED;
#endif
}

amdsmi_status_t AMDSmiWslGPUDevice::get_uuid(unsigned int* uuid_length, char* uuid) const {
  if (uuid_length == nullptr || uuid == nullptr || *uuid_length < AMDSMI_GPU_UUID_SIZE) {
    return AMDSMI_STATUS_INVAL;
  }

  const uint64_t id = unique_id_ ? unique_id_ : bdf_.as_uint;
  amdsmi_status_t status = amdsmi_uuid_gen(uuid, id, device_id_, 0xff);
  if (status == AMDSMI_STATUS_SUCCESS) *uuid_length = AMDSMI_GPU_UUID_SIZE;
  return status;
}

bool is_wsl_gpu_device(const AMDSmiGPUDevice* device) {
  return device != nullptr && device->is_wsl_device();
}

AMDSmiWslGPUDevice* as_wsl_gpu_device(AMDSmiGPUDevice* device) {
  return is_wsl_gpu_device(device) ? static_cast<AMDSmiWslGPUDevice*>(device) : nullptr;
}

const AMDSmiWslGPUDevice* as_wsl_gpu_device(const AMDSmiGPUDevice* device) {
  return is_wsl_gpu_device(device) ? static_cast<const AMDSmiWslGPUDevice*>(device) : nullptr;
}

}  // namespace amd::smi
