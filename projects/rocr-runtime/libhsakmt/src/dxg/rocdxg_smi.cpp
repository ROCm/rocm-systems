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

#include "hsakmt/rocdxg_smi.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "impl/wddm/device.h"
#include "impl/wddm/thunks.h"
#include "librocdxg.h"
#include "wkmi.h"

namespace {

wsl::thunk::WDDMDevice* checked_device(uint32_t node_id) {
  if (dxg_runtime == nullptr || dxg_runtime->dxg_open_count == 0 || dxg_runtime->is_forked) {
    return nullptr;
  }
  return get_wddmdev(node_id);
}

// Map NTSTATUS → HSAKMT_STATUS
HSAKMT_STATUS nt_to_hsa(NTSTATUS status) {
  switch (status) {
    case STATUS_SUCCESS:           return HSAKMT_STATUS_SUCCESS;
    case STATUS_NOT_SUPPORTED:
    case STATUS_NOT_IMPLEMENTED:   return HSAKMT_STATUS_NOT_SUPPORTED;
    case STATUS_NO_MEMORY:         return HSAKMT_STATUS_NO_MEMORY;
    case STATUS_BUFFER_TOO_SMALL:  return HSAKMT_STATUS_BUFFER_TOO_SMALL;
    case STATUS_INVALID_PARAMETER: return HSAKMT_STATUS_INVALID_PARAMETER;
    default:                       return HSAKMT_STATUS_ERROR;
  }
}

// Look up a PMLog sensor value by sensor ID.
//
// QueryPMLogSupport returns sensor_ids[slot] = sensor_id for each occupied slot.
// QueryPMLogData returns pmlog.supported[sensor_id] / pmlog.value[sensor_id] — it
// uses the sensor_id directly as the array index, NOT the slot position.
// So we index pmlog directly by the sensor_id enum value.
static uint32_t pmlog_sensor(const Wkmi::PmlogQueryResult& pmlog,
                              const uint16_t /* unused_slot_map */[Wkmi::kPmlogMaxSensors],
                              Wkmi::PmlogSensorId id) {
  uint32_t idx = static_cast<uint32_t>(id);
  if (idx >= Wkmi::kPmlogMaxSensors) return Wkmi::kSensorUnavailable;
  if (!pmlog.supported[idx]) return Wkmi::kSensorUnavailable;
  return pmlog.value[idx];
}

template <typename T>
HSAKMT_STATUS clear_out(T* out) {
  if (out == nullptr) return HSAKMT_STATUS_INVALID_PARAMETER;
  std::memset(out, 0, sizeof(*out));
  return HSAKMT_STATUS_SUCCESS;
}

void copy_string(char* dst, const char* src) {
  if (dst == nullptr) return;
  std::memset(dst, 0, ROCDXG_SMI_MAX_STRING_LENGTH);
  if (src != nullptr) {
    std::snprintf(dst, ROCDXG_SMI_MAX_STRING_LENGTH, "%s", src);
  }
}

struct D3DKMT_ENUMPROCESSES {
  LUID AdapterLuid;
  uint64_t Buffer;
  uint64_t BufferCount;
};

uint64_t target_graphics_version(wsl::thunk::WDDMDevice& device) {
  return (static_cast<uint64_t>(device.Major()) << 16) |
         (static_cast<uint64_t>(device.Minor()) << 8) |
         static_cast<uint64_t>(device.Stepping());
}

HSAKMT_STATUS query_process_vram(wsl::thunk::WDDMDevice& device, uint32_t pid,
                                 uint64_t* vram_bytes) {
  if (vram_bytes == nullptr) return HSAKMT_STATUS_INVALID_PARAMETER;
  if (!wsl::thunk::d3dthunk::QueryVideoMemoryInfoAvailable()) {
    return HSAKMT_STATUS_NOT_SUPPORTED;
  }

  D3DKMT_QUERYVIDEOMEMORYINFO args = {};
  args.hProcess = reinterpret_cast<HANDLE>(static_cast<uintptr_t>(pid));
  args.hAdapter = device.GetAdapter();
  args.MemorySegmentGroup = D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL;
  args.PhysicalAdapterIndex = 0;

  auto code = wsl::thunk::d3dthunk::QueryVideoMemoryInfo(&args);
  if (code != ErrorCode::Success) {
    return HSAKMT_STATUS_ERROR;
  }
  *vram_bytes = args.CurrentUsage;
  return HSAKMT_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_device_count(uint32_t* count) {
  if (count == nullptr) return HSAKMT_STATUS_INVALID_PARAMETER;
  CHECK_DXG_OPEN();
  *count = get_num_wddmdev();
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_bdf_info(uint32_t node_id,
                                                rocdxg_smi_bdf_info_t* info) {
  HSAKMT_STATUS status = clear_out(info);
  if (status != HSAKMT_STATUS_SUCCESS) return status;

  auto* device = checked_device(node_id);
  if (device == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  const uint32_t location = device->PciBusAddr();
  info->domain_number = device->Domain();
  info->bus_number = (location >> 8) & 0xff;
  info->device_number = (location >> 3) & 0x1f;
  info->function_number = location & 0x7;
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_asic_info(uint32_t node_id,
                                                 rocdxg_smi_asic_info_t* info) {
  HSAKMT_STATUS status = clear_out(info);
  if (status != HSAKMT_STATUS_SUCCESS) return status;

  auto* device = checked_device(node_id);
  if (device == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  info->device_id = device->DeviceId();
  info->vendor_id = 0x1002;
  info->subvendor_id = std::numeric_limits<uint32_t>::max();
  info->subsystem_id = std::numeric_limits<uint32_t>::max();
  info->rev_id = device->AsicRevision();
  info->asic_serial = device->Uuid();
  copy_string(info->market_name, device->ProductName());
  info->num_of_compute_units = device->ComputeUnitCount();
  info->target_graphics_version = target_graphics_version(*device);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_board_info(uint32_t node_id,
                                                  rocdxg_smi_board_info_t* info) {
  HSAKMT_STATUS status = clear_out(info);
  if (status != HSAKMT_STATUS_SUCCESS) return status;

  auto* wdev = checked_device(node_id);
  if (wdev == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  Wkmi::DriverRegInfo reg = {};
  if (Wkmi::QueryDriverRegInfo(wdev->GetAdapter(), &reg) == STATUS_SUCCESS &&
      reg.adapter_string[0] != '\0') {
    copy_string(info->product_name, reg.adapter_string);
  } else {
    copy_string(info->product_name, wdev->ProductName());
  }
  copy_string(info->manufacturer_name, "Advanced Micro Devices, Inc. [AMD/ATI]");
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_vram_info(uint32_t node_id,
                                                 rocdxg_smi_vram_info_t* info) {
  HSAKMT_STATUS status = clear_out(info);
  if (status != HSAKMT_STATUS_SUCCESS) return status;

  auto* device = checked_device(node_id);
  if (device == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  info->vram_type = 0;
  info->vram_bit_width = device->MemoryBusWidth();
  info->vram_size_mb = device->LocalHeapSize() / (1024 * 1024);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_vram_usage(uint32_t node_id,
                                                  rocdxg_smi_vram_usage_t* usage) {
  HSAKMT_STATUS status = clear_out(usage);
  if (status != HSAKMT_STATUS_SUCCESS) return status;

  auto* device = checked_device(node_id);
  if (device == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  const uint64_t total = device->LocalHeapSize();
  uint64_t available = 0;
  if (device->VramAvail(&available) != HSA_STATUS_SUCCESS) {
    return HSAKMT_STATUS_ERROR;
  }

  usage->vram_total_mb = total / (1024 * 1024);
  usage->vram_used_mb = (available >= total) ? 0 : ((total - available) / (1024 * 1024));
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_power_info(uint32_t node_id,
                                                  rocdxg_smi_power_info_t* info) {
  HSAKMT_STATUS status = clear_out(info);
  if (status != HSAKMT_STATUS_SUCCESS) return status;

  auto* wdev = checked_device(node_id);
  if (wdev == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  // Query sensor IDs first so we can map sensor index → id
  uint16_t sensor_ids[Wkmi::kPmlogMaxSensors] = {};
  Wkmi::QueryPMLogSupport(wdev->GetAdapter(), wdev->DeviceHandle(), sensor_ids);

  Wkmi::PmlogQueryResult pmlog = {};
  NTSTATUS ret = Wkmi::QueryPMLogData(wdev->GetAdapter(), wdev->DeviceHandle(), &pmlog);
  if (ret != STATUS_SUCCESS) return nt_to_hsa(ret);

  auto v = [&](Wkmi::PmlogSensorId id) {
    return pmlog_sensor(pmlog, sensor_ids, id);
  };

  info->current_socket_power = v(Wkmi::kPmlogBoardPower) != Wkmi::kSensorUnavailable
                                   ? v(Wkmi::kPmlogBoardPower)
                                   : v(Wkmi::kPmlogAsicPower);
  info->gfx_voltage = v(Wkmi::kPmlogGfxVoltage);
  info->soc_voltage = v(Wkmi::kPmlogSocVoltage);
  info->mem_voltage = v(Wkmi::kPmlogMemVoltage);

  // power_limit from sensor limits (max of asic/board power sensor)
  Wkmi::PmlogSensorLimits limits = {};
  if (Wkmi::QueryPMLogSensorLimits(wdev->GetAdapter(), wdev->DeviceHandle(), &limits) ==
      STATUS_SUCCESS) {
    // Find the limit for kPmlogBoardPower or kPmlogAsicPower
    for (uint32_t i = 0; i < Wkmi::kPmlogMaxSensors; ++i) {
      if (sensor_ids[i] == static_cast<uint16_t>(Wkmi::kPmlogBoardPower) ||
          sensor_ids[i] == static_cast<uint16_t>(Wkmi::kPmlogAsicPower)) {
        info->power_limit = limits.limits[i][1]; // max
        break;
      }
    }
  }
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_temperature(uint32_t node_id,
                                                   uint32_t sensor_type,
                                                   uint32_t metric,
                                                   int64_t* temperature) {
  if (temperature == nullptr) return HSAKMT_STATUS_INVALID_PARAMETER;

  auto* wdev = checked_device(node_id);
  if (wdev == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  // Only AMDSMI_TEMP_CURRENT (metric==0) is available via PMLog
  if (metric != 0) return HSAKMT_STATUS_NOT_SUPPORTED;

  // Primary PMLog sensor IDs per amdsmi sensor_type:
  //   EDGE(0) → TEMP_EDGE(8)     fallback: TEMP_GFX(28)
  //   HOTSPOT(1) → TEMP_HOTSPOT(27) fallback: TEMP_SOC(29)
  //   VRAM(2) → TEMP_MEM(9)     fallback: none
  // On APUs without discrete edge/hotspot sensors, TEMP_GFX and TEMP_SOC carry
  // equivalent readings.
  Wkmi::PmlogSensorId primary, fallback;
  switch (sensor_type) {
    case 0: primary = Wkmi::kPmlogTempEdge;    fallback = Wkmi::kPmlogTempGfx; break;
    case 1: primary = Wkmi::kPmlogTempHotspot; fallback = Wkmi::kPmlogTempSoc; break;
    case 2: primary = Wkmi::kPmlogTempMem;     fallback = Wkmi::kPmlogTempMem; break;
    default: return HSAKMT_STATUS_NOT_SUPPORTED;
  }

  uint16_t sensor_ids[Wkmi::kPmlogMaxSensors] = {};
  Wkmi::QueryPMLogSupport(wdev->GetAdapter(), wdev->DeviceHandle(), sensor_ids);

  Wkmi::PmlogQueryResult pmlog = {};
  NTSTATUS ret = Wkmi::QueryPMLogData(wdev->GetAdapter(), wdev->DeviceHandle(), &pmlog);
  if (ret != STATUS_SUCCESS) return nt_to_hsa(ret);

  uint32_t val = pmlog_sensor(pmlog, sensor_ids, primary);
  if (val == Wkmi::kSensorUnavailable)
    val = pmlog_sensor(pmlog, sensor_ids, fallback);
  if (val == Wkmi::kSensorUnavailable) return HSAKMT_STATUS_NOT_SUPPORTED;

  *temperature = static_cast<int64_t>(val);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_clock_info(uint32_t node_id,
                                                  uint32_t clk_type,
                                                  rocdxg_smi_clock_info_t* info) {
  HSAKMT_STATUS status = clear_out(info);
  if (status != HSAKMT_STATUS_SUCCESS) return status;

  auto* wdev = checked_device(node_id);
  if (wdev == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  // Map amdsmi clk_type to PMLog sensor IDs:
  // 0=GFX/SYS → kPmlogGfxClk(1), 4=MEM → kPmlogMemClk(2), 3=SOC → kPmlogSocClk(3)
  Wkmi::PmlogSensorId target;
  switch (clk_type) {
    case 0: target = Wkmi::kPmlogGfxClk; break;
    case 3: target = Wkmi::kPmlogSocClk; break;
    case 4: target = Wkmi::kPmlogMemClk; break;
    default: return HSAKMT_STATUS_NOT_SUPPORTED;
  }

  uint16_t sensor_ids[Wkmi::kPmlogMaxSensors] = {};
  Wkmi::QueryPMLogSupport(wdev->GetAdapter(), wdev->DeviceHandle(), sensor_ids);

  Wkmi::PmlogQueryResult pmlog = {};
  NTSTATUS ret = Wkmi::QueryPMLogData(wdev->GetAdapter(), wdev->DeviceHandle(), &pmlog);
  if (ret == STATUS_SUCCESS) {
    uint32_t cur = pmlog_sensor(pmlog, sensor_ids, target);
    if (cur != Wkmi::kSensorUnavailable) {
      info->clk = cur;
      // Static max clocks from adapter info
      info->max_clk = (clk_type == 4) ? wdev->MaxMemoryClockMhz() : wdev->MaxEngineClockMhz();
      return HSAKMT_STATUS_SUCCESS;
    }
  }

  // Fallback: static max only for GFX and MEM
  if (clk_type == 0) { info->max_clk = wdev->MaxEngineClockMhz(); return HSAKMT_STATUS_SUCCESS; }
  if (clk_type == 4) { info->max_clk = wdev->MaxMemoryClockMhz(); return HSAKMT_STATUS_SUCCESS; }
  return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_pcie_info(uint32_t node_id,
                                                 rocdxg_smi_pcie_info_t* info) {
  HSAKMT_STATUS status = clear_out(info);
  if (status != HSAKMT_STATUS_SUCCESS) return status;

  auto* wdev = checked_device(node_id);
  if (wdev == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  // Static PCIe capabilities from CWDDECI_CHIPSETIDENTIFICATION
  Wkmi::ChipsetIdInfo ci = {};
  NTSTATUS ret = Wkmi::QueryChipsetId(wdev->GetAdapter(), wdev->DeviceHandle(), &ci);
  if (ret != STATUS_SUCCESS) return nt_to_hsa(ret);

  info->max_pcie_width         = static_cast<uint16_t>(ci.max_pcie_lane_width);
  info->pcie_interface_version = ci.pcie_gen;
  // speed in MT/s: gen1=2500, gen2=5000, gen3=8000, gen4=16000, gen5=32000
  static const uint32_t kGenSpeed[] = {0, 2500, 5000, 8000, 16000, 32000};
  uint32_t gen = (ci.pcie_gen < 6) ? ci.pcie_gen : 0;
  info->max_pcie_speed = kGenSpeed[gen];

  // Dynamic current width/speed from PMLog BUS_LANES and BUS_SPEED
  uint16_t sensor_ids[Wkmi::kPmlogMaxSensors] = {};
  Wkmi::QueryPMLogSupport(wdev->GetAdapter(), wdev->DeviceHandle(), sensor_ids);
  Wkmi::PmlogQueryResult pmlog = {};
  if (Wkmi::QueryPMLogData(wdev->GetAdapter(), wdev->DeviceHandle(), &pmlog) == STATUS_SUCCESS) {
    uint32_t lanes = pmlog_sensor(pmlog, sensor_ids, Wkmi::kPmlogBusLanes);
    uint32_t speed = pmlog_sensor(pmlog, sensor_ids, Wkmi::kPmlogBusSpeed);
    if (lanes != Wkmi::kSensorUnavailable)
      info->pcie_width = static_cast<uint16_t>(lanes);
    if (speed != Wkmi::kSensorUnavailable)
      info->pcie_speed = speed;
  }
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_driver_info(uint32_t node_id,
                                                   rocdxg_smi_driver_info_t* info) {
  HSAKMT_STATUS status = clear_out(info);
  if (status != HSAKMT_STATUS_SUCCESS) return status;

  auto* wdev = checked_device(node_id);
  if (wdev == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  Wkmi::DriverRegInfo reg = {};
  NTSTATUS ret = Wkmi::QueryDriverRegInfo(wdev->GetAdapter(), &reg);
  if (ret != STATUS_SUCCESS) return nt_to_hsa(ret);

  // Match upstream libthunk_proxy.a field assignments (verified by disassembly):
  //   driver_version = ReleaseVersion  (full string, e.g. "26.10.21.04-260623a-...")
  //   driver_date    = 6 chars after first '-' in ReleaseVersion (e.g. "260623")
  //   driver_name    = DriverDesc      (e.g. "AMD Radeon(TM) 8060S Graphics")
  copy_string(info->driver_version, reg.release_version);
  copy_string(info->driver_name,    reg.driver_desc);

  // Extract date: up to 6 chars starting right after the first '-'
  const char* dash = std::strchr(reg.release_version, '-');
  if (dash && *(dash + 1) != '\0') {
    char date_buf[ROCDXG_SMI_MAX_STRING_LENGTH] = {};
    std::strncpy(date_buf, dash + 1, 6);
    copy_string(info->driver_date, date_buf);
  } else {
    copy_string(info->driver_date, "N/A");
  }
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_vbios_info(uint32_t node_id,
                                                  rocdxg_smi_vbios_info_t* info) {
  HSAKMT_STATUS status = clear_out(info);
  if (status != HSAKMT_STATUS_SUCCESS) return status;

  auto* wdev = checked_device(node_id);
  if (wdev == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  Wkmi::VideoBiosInfo vbios = {};
  NTSTATUS ret = Wkmi::QueryVideoBiosInfo(wdev->GetAdapter(), wdev->DeviceHandle(), &vbios);
  if (ret != STATUS_SUCCESS) return nt_to_hsa(ret);

  copy_string(info->version,    vbios.version);
  copy_string(info->part_number, vbios.part_number);
  copy_string(info->build_date, vbios.date);
  // name: use adapter string from registry as product name
  Wkmi::DriverRegInfo reg = {};
  if (Wkmi::QueryDriverRegInfo(wdev->GetAdapter(), &reg) == STATUS_SUCCESS)
    copy_string(info->name, reg.adapter_string);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_gpu_metrics_info(
    uint32_t node_id, rocdxg_smi_gpu_metrics_info_t* info) {
  HSAKMT_STATUS status = clear_out(info);
  if (status != HSAKMT_STATUS_SUCCESS) return status;

  auto* wdev = checked_device(node_id);
  if (wdev == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  uint16_t sensor_ids[Wkmi::kPmlogMaxSensors] = {};
  Wkmi::QueryPMLogSupport(wdev->GetAdapter(), wdev->DeviceHandle(), sensor_ids);

  Wkmi::PmlogQueryResult pmlog = {};
  NTSTATUS ret = Wkmi::QueryPMLogData(wdev->GetAdapter(), wdev->DeviceHandle(), &pmlog);
  if (ret != STATUS_SUCCESS) return nt_to_hsa(ret);

  auto v = [&](Wkmi::PmlogSensorId id) -> uint32_t {
    return pmlog_sensor(pmlog, sensor_ids, id);
  };

  info->temperature_edge          = v(Wkmi::kPmlogTempEdge);
  info->temperature_hotspot       = v(Wkmi::kPmlogTempHotspot);
  info->temperature_mem           = v(Wkmi::kPmlogTempMem);
  info->average_gfx_activity      = v(Wkmi::kPmlogGfxActivity);
  info->average_umc_activity      = v(Wkmi::kPmlogMemActivity);
  uint32_t bp = v(Wkmi::kPmlogBoardPower);
  info->current_socket_power      = (bp != Wkmi::kSensorUnavailable) ? bp : v(Wkmi::kPmlogAsicPower);
  info->current_gfxclk            = v(Wkmi::kPmlogGfxClk);
  info->current_socclk            = v(Wkmi::kPmlogSocClk);
  info->current_fan_speed         = v(Wkmi::kPmlogFanRpm);
  info->current_fan_speed_percent = v(Wkmi::kPmlogFanPercent);
  info->voltage_soc               = v(Wkmi::kPmlogSocVoltage);
  info->voltage_gfx               = v(Wkmi::kPmlogGfxVoltage);
  info->voltage_mem               = v(Wkmi::kPmlogMemVoltage);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_enum_processes(uint32_t node_id,
                                                  uint32_t* num_processes,
                                                  rocdxg_smi_process_info_t* processes) {
  if (num_processes == nullptr) return HSAKMT_STATUS_INVALID_PARAMETER;

  auto* device = checked_device(node_id);
  if (device == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;
  if (!wsl::thunk::d3dthunk::EnumProcessesAvailable()) {
    return HSAKMT_STATUS_NOT_SUPPORTED;
  }

  uint64_t capacity = *num_processes;
  if (capacity == 0) capacity = 256;
  std::vector<uint32_t> pids(static_cast<size_t>(capacity), 0);

  D3DKMT_ENUMPROCESSES args = {};
  args.AdapterLuid = device->GetLuid();
  args.Buffer = reinterpret_cast<uint64_t>(pids.data());
  args.BufferCount = capacity;

  auto code = wsl::thunk::d3dthunk::EnumProcesses(&args);
  if (code != ErrorCode::Success) return HSAKMT_STATUS_ERROR;

  const uint32_t found = static_cast<uint32_t>(args.BufferCount);
  if (processes == nullptr) {
    *num_processes = found;
    return HSAKMT_STATUS_SUCCESS;
  }

  const uint32_t output_capacity = *num_processes;
  *num_processes = found;
  const uint32_t copy_count = std::min(found, output_capacity);
  for (uint32_t i = 0; i < copy_count; ++i) {
    processes[i] = {};
    processes[i].process_id = pids[i];
    uint64_t vram = 0;
    if (query_process_vram(*device, pids[i], &vram) == HSAKMT_STATUS_SUCCESS) {
      processes[i].vram_usage_bytes = vram;
    }
  }

  return output_capacity >= found ? HSAKMT_STATUS_SUCCESS : HSAKMT_STATUS_BUFFER_TOO_SMALL;
}

}  // extern "C"
