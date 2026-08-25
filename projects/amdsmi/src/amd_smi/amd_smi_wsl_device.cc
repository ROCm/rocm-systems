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

#ifdef ENABLE_WSL_BACKEND

#include "amd_smi/impl/amd_smi_wsl_device.h"

#include <fcntl.h>
#include <ntstatus.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_socket.h"
#include "amd_smi/impl/amd_smi_uuid.h"
#include "amd_smi/impl/wsl/dxcore_loader.h"
#include "amd_smi/impl/wsl/wsl_pmlog.h"
#include "amd_smi/impl/wsl/wsl_status.h"
#include "amd_smi/impl/wsl/wsl_vram.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace amd::smi {
namespace {

static constexpr const char* kDxgDevPath = "/dev/dxg";

// PCIe generation (1-5) -> max link speed in MT/s. Index 0 (unknown gen) maps to 0.
static constexpr uint32_t kGenSpeed[] = {0, 2500, 5000, 8000, 16000, 32000};

static amdsmi_bdf_t make_bdf(const Wkmi::DeviceInfo& dev) {
  amdsmi_bdf_t bdf = {};
  bdf.bdf.domain_number = dev.domain;
  bdf.bdf.bus_number = (dev.pci_bus_addr >> 8) & 0xff;
  bdf.bdf.device_number = (dev.pci_bus_addr >> 3) & 0x1f;
  bdf.bdf.function_number = dev.pci_bus_addr & 0x7;
  return bdf;
}

static void copy_string(char* dst, const std::string& src) {
  if (dst == nullptr) return;
  std::memset(dst, 0, AMDSMI_MAX_STRING_LENGTH);
  std::snprintf(dst, AMDSMI_MAX_STRING_LENGTH, "%s", src.c_str());
}

static void copy_c_string(char* dst, const char* src) {
  if (dst == nullptr) return;
  std::memset(dst, 0, AMDSMI_MAX_STRING_LENGTH);
  if (src != nullptr) std::snprintf(dst, AMDSMI_MAX_STRING_LENGTH, "%s", src);
}

// Derives a short driver "date" field (up to 6 chars after the first '-') from
// a driver_version string such as "31.0.24027-250101a-...". Returns "N/A" if
// no '-' is present. There is no dedicated driver-date field on WslAdapterInfo
// (only driver_version/driver_desc registry strings are available), so this
// reimplements the old rocdxg driver_date convention locally.
static std::string derive_driver_date(const char* driver_version) {
  if (driver_version == nullptr) return "N/A";
  const char* dash = std::strchr(driver_version, '-');
  if (dash == nullptr || *(dash + 1) == '\0') return "N/A";
  std::string date(dash + 1, std::min<size_t>(6, std::strlen(dash + 1)));
  return date;
}

// Placeholder DRM instance for WSL devices — WSL has no /dev/dri.
static AMDSmiDrm g_wsl_drm;

// Adapters opened by try_populate, torn down by shutdown(). Idempotent: cleared
// after a successful shutdown() so a second call is a no-op.
static std::vector<wsl::WslAdapterInfo> g_wsl_adapters;
static bool g_wsl_active = false;

}  // namespace

// -----------------------------------------------------------------------------
// WSLGPUBackend constructor and try_populate
// -----------------------------------------------------------------------------

WSLGPUBackend::WSLGPUBackend(uint32_t gpu_id, const wsl::WslAdapterInfo& adapter)
    : gpu_id_(gpu_id), bdf_(make_bdf(adapter.device_info)), adapter_(adapter) {}

amdsmi_status_t WSLGPUBackend::try_populate(std::vector<AMDSmiSocket*>& sockets,
                                            std::set<AMDSmiProcessor*>& processors) {
  // Not WSL if /dev/dxg is absent.
  if (access(kDxgDevPath, F_OK) != 0) return AMDSMI_STATUS_NOT_SUPPORTED;

  if (!wsl::thunk::dxcore::DxcoreLoader::Instance().Initialize())
    return AMDSMI_STATUS_DRIVER_NOT_LOADED;

  std::vector<wsl::WslAdapterInfo> adapters;
  NTSTATUS status = wsl::EnumerateWslAdapters(&adapters);
  if (status != STATUS_SUCCESS) return wsl::ToAmdsmiStatus(status);

  uint32_t gpu_index = 0;
  for (auto& adapter : adapters) {
    status = wsl::OpenWslAdapter(&adapter);
    if (status != STATUS_SUCCESS) {
      std::ostringstream ss;
      ss << __PRETTY_FUNCTION__ << " | wsl::OpenWslAdapter failed: 0x" << std::hex << status;
      LOG_ERROR(ss);
      continue;
    }
    g_wsl_adapters.push_back(adapter);

    auto* backend = new WSLGPUBackend(gpu_index, adapter);

    std::string path = "wsl_adapter" + std::to_string(gpu_index);
    auto* device = new AMDSmiGPUDevice(gpu_index++, path, backend->bdf(), g_wsl_drm);
    device->set_backend(backend);

    std::string socket_id = "wsl:";
    socket_id += std::to_string(static_cast<int>(device->get_processor_type()));
    socket_id += ":";
    socket_id += std::to_string(backend->bdf().as_uint);

    auto* socket = new AMDSmiSocket(socket_id);
    socket->add_processor(device);
    sockets.push_back(socket);
    processors.insert(device);
  }

  if (gpu_index == 0) {
    wsl::thunk::dxcore::DxcoreLoader::Instance().Shutdown();
    return AMDSMI_STATUS_NOT_FOUND;
  }
  g_wsl_active = true;
  return AMDSMI_STATUS_SUCCESS;
}

bool WSLGPUBackend::is_active() { return g_wsl_active; }

amdsmi_status_t WSLGPUBackend::shutdown() {
  if (!g_wsl_active) return AMDSMI_STATUS_SUCCESS;
  g_wsl_active = false;
  for (auto& adapter : g_wsl_adapters) wsl::CloseWslAdapter(&adapter);
  g_wsl_adapters.clear();
  wsl::thunk::dxcore::DxcoreLoader::Instance().Shutdown();
  return AMDSMI_STATUS_SUCCESS;
}

// -----------------------------------------------------------------------------
// IGPUBackend method implementations
// -----------------------------------------------------------------------------

amdsmi_status_t WSLGPUBackend::load_device_info() const {
  std::call_once(device_info_once_, [this]() {
    const auto& dev = adapter_.device_info;
    WslDeviceSnapshot snap;

    snap.subvendor_id = adapter_.sub_vendor_id;
    snap.subsystem_id = adapter_.sub_system_id;

    snap.asic_serial = dev.uuid;

    // Pack major/minor/stepping (three separate ints on Wkmi::DeviceInfo) using
    // the same major<<16|minor<<8|stepping encoding as
    // rocdxg_smi.cpp's target_graphics_version(wsl::thunk::WDDMDevice&).
    snap.target_graphics_version = (static_cast<uint64_t>(dev.major) << 16) |
                                   (static_cast<uint64_t>(dev.minor) << 8) |
                                   static_cast<uint64_t>(dev.stepping);

    // Cache hierarchy: l1/l2/l3_cache_size are in bytes; amdsmi reports KB.
    // level/properties/max_num_cu_shared/num_cache_instance have no equivalent
    // on Wkmi::DeviceInfo, so use reasonable single-instance-per-level defaults
    // (matches the shape rocdxg reported for these ASICs: one CU-shared L1 per
    // CU, one L2 shared across the whole ASIC, one L3/MALL if present).
    uint32_t idx = 0;
    if (dev.l1_cache_size > 0 && idx < AMDSMI_MAX_CACHE_TYPES) {
      snap.cache[idx].cache_size_kb = dev.l1_cache_size / 1024;
      snap.cache[idx].cache_level = 1;
      snap.cache[idx].cache_properties = AMDSMI_CACHE_PROPERTY_DATA_CACHE;
      snap.cache[idx].max_num_cu_shared = 2;
      snap.cache[idx].num_cache_instance = 1;
      ++idx;
    }
    if (dev.l2_cache_size > 0 && idx < AMDSMI_MAX_CACHE_TYPES) {
      snap.cache[idx].cache_size_kb = dev.l2_cache_size / 1024;
      snap.cache[idx].cache_level = 2;
      snap.cache[idx].cache_properties =
          AMDSMI_CACHE_PROPERTY_ENABLED | AMDSMI_CACHE_PROPERTY_DATA_CACHE;
      snap.cache[idx].max_num_cu_shared = dev.compute_unit_count;
      snap.cache[idx].num_cache_instance = 1;
      ++idx;
    }
    if (dev.l3_cache_size > 0 && idx < AMDSMI_MAX_CACHE_TYPES) {
      snap.cache[idx].cache_size_kb = dev.l3_cache_size / 1024;
      snap.cache[idx].cache_level = 3;
      snap.cache[idx].cache_properties =
          AMDSMI_CACHE_PROPERTY_ENABLED | AMDSMI_CACHE_PROPERTY_DATA_CACHE;
      snap.cache[idx].max_num_cu_shared = dev.compute_unit_count;
      snap.cache[idx].num_cache_instance = 1;
      ++idx;
    }
    snap.num_cache_types = idx;

    // Firmware: only MEC/SDMA versions are available on Wkmi::DeviceInfo.
    uint32_t fw_idx = 0;
    if (fw_idx < AMDSMI_FW_ID__MAX) {
      snap.fw_info_list[fw_idx].fw_id = AMDSMI_FW_ID_CP_MEC1;
      snap.fw_info_list[fw_idx].fw_version = dev.mec_fw_version;
      ++fw_idx;
    }
    if (fw_idx < AMDSMI_FW_ID__MAX) {
      snap.fw_info_list[fw_idx].fw_id = AMDSMI_FW_ID_SDMA0;
      snap.fw_info_list[fw_idx].fw_version = dev.sdma_fw_version;
      ++fw_idx;
    }
    snap.num_fw_info = fw_idx;

    // PCIe static capabilities (best-effort; non-fatal if the escape fails).
    Wkmi::ChipsetIdInfo chipset = {};
    if (wsl::QueryChipsetInfo(adapter_, &chipset)) {
      snap.max_pcie_lane_width = chipset.max_pcie_lane_width;
      snap.pcie_gen = chipset.pcie_gen;
    }

    // VBIOS info (best-effort; non-fatal if the escape fails).
    Wkmi::VideoBiosInfo vbios = {};
    if (wsl::QueryVideoBios(adapter_, &vbios)) {
      // Wkmi::VideoBiosInfo has no "name" field; fall back to the product name.
      snap.vbios_name = dev.product_name;
      snap.vbios_build_date = vbios.date;
      snap.vbios_part_number = vbios.part_number;
      snap.vbios_version = vbios.version;
    }

    snap.uuid_seed = snap.asic_serial != 0 ? snap.asic_serial : bdf_.as_uint;

    device_info_ = snap;
    device_info_status_ = AMDSMI_STATUS_SUCCESS;
  });
  return device_info_status_;
}

amdsmi_status_t WSLGPUBackend::get_kfd_info(amdsmi_kfd_info_t* info) {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  // No KFD in WSL — the KFD ID and node ID concepts don't apply. Preserved
  // verbatim from the old implementation (there was never a rocdxg-backed
  // KFD path either).
  std::memset(info, 0xFF, sizeof(*info));
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_asic_info(amdsmi_asic_info_t* info) {
  amdsmi_status_t r = load_device_info();
  if (r != AMDSMI_STATUS_SUCCESS) return r;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  const auto& dev = adapter_.device_info;

  std::memset(info, 0, sizeof(*info));
  copy_string(info->market_name, dev.product_name);
  info->vendor_id = adapter_.vendor_id;
  if (info->vendor_id == 0x1002)
    copy_string(info->vendor_name, "Advanced Micro Devices, Inc. [AMD/ATI]");
  info->subvendor_id = device_info_.subvendor_id;
  info->device_id = adapter_.device_id;
  info->rev_id = dev.asic_revision;
  std::snprintf(info->asic_serial, AMDSMI_MAX_STRING_LENGTH, "%016lx",
                static_cast<unsigned long>(device_info_.asic_serial));
  info->oam_id = gpu_id_;
  info->num_of_compute_units = dev.compute_unit_count;
  info->target_graphics_version = device_info_.target_graphics_version;
  info->subsystem_id = device_info_.subsystem_id;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_board_info(amdsmi_board_info_t* info) {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  std::memset(info, 0, sizeof(*info));
  copy_c_string(info->product_name, adapter_.board_product_name);
  copy_string(info->manufacturer_name, "Advanced Micro Devices, Inc. [AMD/ATI]");
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_vram_info(amdsmi_vram_info_t* info) {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  std::memset(info, 0, sizeof(*info));
  info->vram_type = AMDSMI_VRAM_TYPE_UNKNOWN;
  copy_string(info->vram_vendor, "UNKNOWN");
  info->vram_size = wsl::VramTotal(adapter_) / (1024 * 1024);
  info->vram_bit_width = adapter_.device_info.memory_bus_width;
  info->vram_max_bandwidth = std::numeric_limits<decltype(info->vram_max_bandwidth)>::max();
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_memory_total(amdsmi_memory_type_t mem_type, uint64_t* total) {
  if (mem_type != AMDSMI_MEM_TYPE_VRAM && mem_type != AMDSMI_MEM_TYPE_VIS_VRAM)
    return AMDSMI_STATUS_NOT_SUPPORTED;
  if (total == nullptr) return AMDSMI_STATUS_INVAL;
  *total = wsl::VramTotal(adapter_);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_memory_usage(amdsmi_memory_type_t mem_type, uint64_t* used) {
  if (mem_type != AMDSMI_MEM_TYPE_VRAM && mem_type != AMDSMI_MEM_TYPE_VIS_VRAM)
    return AMDSMI_STATUS_NOT_SUPPORTED;
  if (used == nullptr) return AMDSMI_STATUS_INVAL;
  if (!wsl::QueryVramUsage(adapter_, used)) return AMDSMI_STATUS_API_FAILED;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_temp_metric(amdsmi_temperature_type_t sensor_type,
                                               amdsmi_temperature_metric_t metric,
                                               int64_t* temperature) {
  if (metric != AMDSMI_TEMP_CURRENT) return AMDSMI_STATUS_NOT_SUPPORTED;
  if (temperature == nullptr) return AMDSMI_STATUS_INVAL;

  wsl::PmlogSnapshot snap;
  if (!wsl::QueryPmlogSnapshot(adapter_, &snap)) return AMDSMI_STATUS_API_FAILED;

  uint32_t value = Wkmi::kSensorUnavailable;
  switch (sensor_type) {
    case AMDSMI_TEMPERATURE_TYPE_EDGE:
      value = wsl::PmlogValue(snap, Wkmi::kPmlogTempEdge);
      if (value == Wkmi::kSensorUnavailable)
        value = wsl::PmlogValue(snap, Wkmi::kPmlogTempGfx);  // APU fallback
      break;
    case AMDSMI_TEMPERATURE_TYPE_HOTSPOT:  // == AMDSMI_TEMPERATURE_TYPE_JUNCTION
      value = wsl::PmlogValue(snap, Wkmi::kPmlogTempHotspot);
      if (value == Wkmi::kSensorUnavailable)
        value = wsl::PmlogValue(snap, Wkmi::kPmlogTempSoc);  // APU fallback
      break;
    case AMDSMI_TEMPERATURE_TYPE_VRAM:
      value = wsl::PmlogValue(snap, Wkmi::kPmlogTempMem);  // no fallback
      break;
    default:
      return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  if (value == Wkmi::kSensorUnavailable) return AMDSMI_STATUS_NOT_SUPPORTED;
  *temperature = static_cast<int64_t>(value);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_volt_metric(amdsmi_voltage_type_t sensor_type,
                                               amdsmi_voltage_metric_t metric, int64_t* voltage) {
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL,
  // matching old behavior.
  if (metric != AMDSMI_VOLT_CURRENT) return AMDSMI_STATUS_NOT_SUPPORTED;
  if (sensor_type != AMDSMI_VOLT_TYPE_VDDGFX) return AMDSMI_STATUS_NOT_SUPPORTED;
  if (voltage == nullptr) return AMDSMI_STATUS_INVAL;

  wsl::PmlogSnapshot snap;
  if (!wsl::QueryPmlogSnapshot(adapter_, &snap)) return AMDSMI_STATUS_API_FAILED;
  uint32_t value = wsl::PmlogValue(snap, Wkmi::kPmlogGfxVoltage);
  if (value == Wkmi::kSensorUnavailable) return AMDSMI_STATUS_NOT_SUPPORTED;
  *voltage = static_cast<int64_t>(value);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_power_info(amdsmi_power_info_t* info) {
  wsl::PmlogSnapshot snap;
  if (!wsl::QueryPmlogSnapshot(adapter_, &snap)) return AMDSMI_STATUS_API_FAILED;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;

  std::memset(info, 0, sizeof(*info));
  info->socket_power = std::numeric_limits<decltype(info->socket_power)>::max();
  info->current_socket_power = std::numeric_limits<decltype(info->current_socket_power)>::max();
  info->average_socket_power = std::numeric_limits<decltype(info->average_socket_power)>::max();
  info->gfx_voltage = std::numeric_limits<decltype(info->gfx_voltage)>::max();
  info->soc_voltage = std::numeric_limits<decltype(info->soc_voltage)>::max();
  info->mem_voltage = std::numeric_limits<decltype(info->mem_voltage)>::max();
  info->power_limit = std::numeric_limits<decltype(info->power_limit)>::max();
  info->ubb_power = std::numeric_limits<decltype(info->ubb_power)>::max();

  uint32_t power = wsl::PmlogValue(snap, Wkmi::kPmlogBoardPower);
  if (power == Wkmi::kSensorUnavailable)
    power = wsl::PmlogValue(snap, Wkmi::kPmlogAsicPower);  // board power -> asic power fallback
  if (power != Wkmi::kSensorUnavailable) {
    info->current_socket_power = power;
    info->average_socket_power = power;
    info->socket_power = power;
  }

  uint32_t gfx_v = wsl::PmlogValue(snap, Wkmi::kPmlogGfxVoltage);
  if (gfx_v != Wkmi::kSensorUnavailable) info->gfx_voltage = gfx_v;
  uint32_t soc_v = wsl::PmlogValue(snap, Wkmi::kPmlogSocVoltage);
  if (soc_v != Wkmi::kSensorUnavailable) info->soc_voltage = soc_v;
  uint32_t mem_v = wsl::PmlogValue(snap, Wkmi::kPmlogMemVoltage);
  if (mem_v != Wkmi::kSensorUnavailable) info->mem_voltage = mem_v;

  uint32_t limit = wsl::PmlogMaxLimit(snap, Wkmi::kPmlogBoardPower);
  if (limit == Wkmi::kSensorUnavailable) limit = wsl::PmlogMaxLimit(snap, Wkmi::kPmlogAsicPower);
  if (limit != Wkmi::kSensorUnavailable) info->power_limit = limit;

  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_busy_percent(uint32_t* gpu_busy_percent) {
  wsl::PmlogSnapshot snap;
  if (!wsl::QueryPmlogSnapshot(adapter_, &snap)) return AMDSMI_STATUS_API_FAILED;
  uint32_t value = wsl::PmlogValue(snap, Wkmi::kPmlogGfxActivity);
  if (value == Wkmi::kSensorUnavailable) return AMDSMI_STATUS_NOT_SUPPORTED;
  if (gpu_busy_percent == nullptr) return AMDSMI_STATUS_INVAL;
  *gpu_busy_percent = value;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_gpu_activity(amdsmi_engine_usage_t* info) {
  wsl::PmlogSnapshot snap;
  if (!wsl::QueryPmlogSnapshot(adapter_, &snap)) return AMDSMI_STATUS_API_FAILED;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  std::memset(info, 0, sizeof(*info));
  uint32_t gfx = wsl::PmlogValue(snap, Wkmi::kPmlogGfxActivity);
  uint32_t mem = wsl::PmlogValue(snap, Wkmi::kPmlogMemActivity);
  info->gfx_activity = gfx != Wkmi::kSensorUnavailable ? gfx : 0x0000FFFF;
  info->umc_activity = mem != Wkmi::kSensorUnavailable ? mem : std::numeric_limits<uint32_t>::max();
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_clock_info(amdsmi_clk_type_t clk_type, amdsmi_clk_info_t* info) {
  Wkmi::PmlogSensorId sensor;
  switch (clk_type) {
    case AMDSMI_CLK_TYPE_SYS:  // == AMDSMI_CLK_TYPE_GFX
      sensor = Wkmi::kPmlogGfxClk;
      break;
    case AMDSMI_CLK_TYPE_MEM:
      sensor = Wkmi::kPmlogMemClk;
      break;
    case AMDSMI_CLK_TYPE_SOC:
      sensor = Wkmi::kPmlogSocClk;
      break;
    default:
      return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  if (info == nullptr) return AMDSMI_STATUS_INVAL;

  wsl::PmlogSnapshot snap;
  uint32_t clk = Wkmi::kSensorUnavailable;
  if (wsl::QueryPmlogSnapshot(adapter_, &snap)) clk = wsl::PmlogValue(snap, sensor);

  uint32_t max_clk = clk_type == AMDSMI_CLK_TYPE_MEM ? adapter_.device_info.max_memory_clock_mhz
                                                     : adapter_.device_info.max_engine_clock_mhz;

  std::memset(info, 0, sizeof(*info));
  // max_clk of 0 means Wkmi::ParseAdapterInfo never populated this field on
  // this ASIC/driver; report NOT_SUPPORTED rather than a SUCCESS the caller
  // can't trust (max_clk is otherwise guaranteed non-zero on SUCCESS).
  if (clk != Wkmi::kSensorUnavailable && max_clk > 0) {
    info->clk = clk;
    info->max_clk = max_clk;
    return AMDSMI_STATUS_SUCCESS;
  }

  // Fallback: static max only, for GFX and MEM (matches
  // rocdxg_smi_get_clock_info's behavior when the PMLog sensor is unavailable).
  if ((clk_type == AMDSMI_CLK_TYPE_SYS || clk_type == AMDSMI_CLK_TYPE_MEM) && max_clk > 0) {
    info->max_clk = max_clk;
    return AMDSMI_STATUS_SUCCESS;
  }
  return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t WSLGPUBackend::get_pcie_info(amdsmi_pcie_info_t* info) {
  amdsmi_status_t r = load_device_info();
  if (r != AMDSMI_STATUS_SUCCESS) return r;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;

  std::memset(info, 0, sizeof(*info));
  // pcie_gen == 0 means Wkmi::QueryChipsetId failed; leave static fields at
  // their zero default instead of aborting before the dynamic PMLog query below.
  info->pcie_static.max_pcie_width = static_cast<uint16_t>(device_info_.max_pcie_lane_width);
  uint32_t gen = std::min<uint32_t>(device_info_.pcie_gen, 5);
  info->pcie_static.max_pcie_speed = kGenSpeed[gen];
  info->pcie_static.pcie_interface_version = device_info_.pcie_gen;
  // No equivalent on Wkmi::ChipsetIdInfo for these two fields.
  info->pcie_static.slot_type = AMDSMI_CARD_FORM_FACTOR_UNKNOWN;
  info->pcie_static.max_pcie_interface_version = std::numeric_limits<uint32_t>::max();

  wsl::PmlogSnapshot snap;
  if (!wsl::QueryPmlogSnapshot(adapter_, &snap)) return AMDSMI_STATUS_API_FAILED;
  // Static chipset fields never populated on this hardware/driver (pcie_gen ==
  // 0): report the same API_FAILED the caller already tolerates instead of a
  // SUCCESS the required-non-zero static fields can't back up.
  if (device_info_.pcie_gen == 0) return AMDSMI_STATUS_API_FAILED;
  uint32_t width = wsl::PmlogValue(snap, Wkmi::kPmlogBusLanes);
  uint32_t speed_gen = wsl::PmlogValue(snap, Wkmi::kPmlogBusSpeed);
  info->pcie_metric.pcie_width =
      width != Wkmi::kSensorUnavailable ? static_cast<uint16_t>(width) : 0;
  if (speed_gen != Wkmi::kSensorUnavailable) {
    uint32_t cur_gen = std::min<uint32_t>(speed_gen, 5);
    info->pcie_metric.pcie_speed = kGenSpeed[cur_gen];
  } else {
    info->pcie_metric.pcie_speed = 0;
  }
  // No PMLog sensor equivalents exist for these counters/bandwidth.
  info->pcie_metric.pcie_bandwidth = 0;
  info->pcie_metric.pcie_replay_count = 0;
  info->pcie_metric.pcie_l0_to_recovery_count = 0;
  info->pcie_metric.pcie_replay_roll_over_count = 0;
  info->pcie_metric.pcie_nak_sent_count = 0;
  info->pcie_metric.pcie_nak_received_count = 0;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_driver_info(amdsmi_driver_info_t* info) {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  std::memset(info, 0, sizeof(*info));
  copy_c_string(info->driver_version, adapter_.driver_version);
  copy_string(info->driver_date, derive_driver_date(adapter_.driver_version));
  copy_c_string(info->driver_name, adapter_.driver_desc);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_vbios_info(amdsmi_vbios_info_t* info) {
  amdsmi_status_t r = load_device_info();
  if (r != AMDSMI_STATUS_SUCCESS) return r;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  std::memset(info, 0, sizeof(*info));
  copy_string(info->name, device_info_.vbios_name);
  copy_string(info->build_date, device_info_.vbios_build_date);
  copy_string(info->part_number, device_info_.vbios_part_number);
  copy_string(info->version, device_info_.vbios_version);
  // No boot_firmware equivalent on Wkmi::VideoBiosInfo.
  copy_string(info->boot_firmware, "N/A");
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_gpu_cache_info(amdsmi_gpu_cache_info_t* info) {
  amdsmi_status_t r = load_device_info();
  if (r != AMDSMI_STATUS_SUCCESS) return r;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  std::memset(info, 0, sizeof(*info));
  info->num_cache_types =
      std::min(device_info_.num_cache_types, static_cast<uint32_t>(AMDSMI_MAX_CACHE_TYPES));
  for (uint32_t i = 0; i < info->num_cache_types; ++i) {
    info->cache[i].cache_size = device_info_.cache[i].cache_size_kb;
    info->cache[i].cache_level = device_info_.cache[i].cache_level;
    info->cache[i].cache_properties = device_info_.cache[i].cache_properties;
    info->cache[i].max_num_cu_shared = device_info_.cache[i].max_num_cu_shared;
    info->cache[i].num_cache_instance = device_info_.cache[i].num_cache_instance;
  }
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_fw_info(amdsmi_fw_info_t* info) {
  amdsmi_status_t r = load_device_info();
  if (r != AMDSMI_STATUS_SUCCESS) return r;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  std::memset(info, 0, sizeof(*info));
  info->num_fw_info = std::min(device_info_.num_fw_info, static_cast<uint32_t>(AMDSMI_FW_ID__MAX));
  for (uint32_t i = 0; i < info->num_fw_info; ++i) {
    info->fw_info_list[i].fw_id = device_info_.fw_info_list[i].fw_id;
    info->fw_info_list[i].fw_version = device_info_.fw_info_list[i].fw_version;
  }
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_fan_rpms(uint32_t /* sensor_ind */, int64_t* speed) {
  // Old code always returned NOT_SUPPORTED here even with rocdxg present. The
  // new PMLog stack does expose a Wkmi::kPmlogFanRpm sensor, so this is
  // implemented rather than preserved as an unconditional stub -- flagged as
  // a deliberate improvement/deviation from old behavior.
  wsl::PmlogSnapshot snap;
  if (!wsl::QueryPmlogSnapshot(adapter_, &snap)) return AMDSMI_STATUS_API_FAILED;
  uint32_t value = wsl::PmlogValue(snap, Wkmi::kPmlogFanRpm);
  if (value == Wkmi::kSensorUnavailable) return AMDSMI_STATUS_NOT_SUPPORTED;
  if (speed == nullptr) return AMDSMI_STATUS_INVAL;
  *speed = static_cast<int64_t>(value);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_fan_speed(uint32_t /* sensor_ind */, int64_t* speed) {
  wsl::PmlogSnapshot snap;
  if (!wsl::QueryPmlogSnapshot(adapter_, &snap)) return AMDSMI_STATUS_API_FAILED;
  uint32_t value = wsl::PmlogValue(snap, Wkmi::kPmlogFanPercent);
  if (value == Wkmi::kSensorUnavailable) return AMDSMI_STATUS_NOT_SUPPORTED;
  if (speed == nullptr) return AMDSMI_STATUS_INVAL;
  *speed = static_cast<int64_t>(value);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_fan_speed_max(uint32_t /* sensor_ind */, uint64_t* max_speed) {
  if (max_speed == nullptr) return AMDSMI_STATUS_INVAL;
  *max_speed = 100;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_power_cap_info(amdsmi_power_cap_info_t* info) {
  amdsmi_power_info_t power = {};
  amdsmi_status_t r = get_power_info(&power);
  if (r != AMDSMI_STATUS_SUCCESS) return r;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;

  std::memset(info, 0, sizeof(*info));
  // Only power_cap has a PMLog-derived source (matches old rocdxg behavior);
  // default/min/max/dpm caps have no equivalent and are left zeroed.
  if (power.power_limit != std::numeric_limits<uint32_t>::max())
    info->power_cap = power.power_limit;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_gpu_metrics_info(amdsmi_gpu_metrics_t* info) {
  wsl::PmlogSnapshot snap;
  if (!wsl::QueryPmlogSnapshot(adapter_, &snap)) return AMDSMI_STATUS_API_FAILED;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;

  // Init all numeric fields to sentinel (0xFF = max for all uint types); keep the pointer null.
  std::memset(info, 0xFF, sizeof(*info));
  info->apu_metrics = nullptr;

  auto set_u16_if_avail = [&](uint16_t* dst, Wkmi::PmlogSensorId id) {
    uint32_t v = wsl::PmlogValue(snap, id);
    if (v != Wkmi::kSensorUnavailable) *dst = static_cast<uint16_t>(v);
  };

  set_u16_if_avail(&info->temperature_edge, Wkmi::kPmlogTempEdge);
  set_u16_if_avail(&info->temperature_hotspot, Wkmi::kPmlogTempHotspot);
  set_u16_if_avail(&info->temperature_mem, Wkmi::kPmlogTempMem);
  set_u16_if_avail(&info->average_gfx_activity, Wkmi::kPmlogGfxActivity);
  set_u16_if_avail(&info->average_umc_activity, Wkmi::kPmlogMemActivity);
  set_u16_if_avail(&info->current_fan_speed, Wkmi::kPmlogFanRpm);
  set_u16_if_avail(&info->voltage_soc, Wkmi::kPmlogSocVoltage);
  set_u16_if_avail(&info->voltage_gfx, Wkmi::kPmlogGfxVoltage);
  set_u16_if_avail(&info->voltage_mem, Wkmi::kPmlogMemVoltage);

  uint32_t power = wsl::PmlogValue(snap, Wkmi::kPmlogBoardPower);
  if (power == Wkmi::kSensorUnavailable) power = wsl::PmlogValue(snap, Wkmi::kPmlogAsicPower);
  if (power != Wkmi::kSensorUnavailable) {
    info->current_socket_power = static_cast<uint16_t>(power);
    info->average_socket_power = static_cast<uint16_t>(power);
  }

  uint32_t gfxclk = wsl::PmlogValue(snap, Wkmi::kPmlogGfxClk);
  if (gfxclk != Wkmi::kSensorUnavailable) {
    // All XCCs run at the same GFX clock; propagate to all XCC slots.
    uint32_t n =
        std::min(adapter_.device_info.num_xcc, static_cast<uint32_t>(AMDSMI_MAX_NUM_GFX_CLKS));
    n = std::max(n, 1U);
    for (uint32_t i = 0; i < n; ++i) info->current_gfxclks[i] = static_cast<uint16_t>(gfxclk);
  }
  set_u16_if_avail(&info->current_socclk, Wkmi::kPmlogSocClk);

  // VCN/JPEG decoders report 0% activity in WSL (no video workloads on this path).
  std::fill(info->vcn_activity, info->vcn_activity + AMDSMI_MAX_NUM_VCN, static_cast<uint16_t>(0));
  std::fill(info->jpeg_activity, info->jpeg_activity + AMDSMI_MAX_NUM_JPEG,
            static_cast<uint16_t>(0));
  // GFX clocks are not locked in WSL; report 0 (all DISABLED bits).
  info->gfxclk_lock_status = 0;
  // No throttle telemetry in WSL; report unthrottled.
  info->throttle_status = 0;

  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_uuid(unsigned int* uuid_length, char* uuid) {
  amdsmi_status_t r = load_device_info();
  if (r != AMDSMI_STATUS_SUCCESS) return r;
  if (uuid_length == nullptr || uuid == nullptr || *uuid_length < AMDSMI_GPU_UUID_SIZE)
    return AMDSMI_STATUS_INVAL;

  amdsmi_status_t status = amdsmi_uuid_gen(uuid, device_info_.uuid_seed, adapter_.device_id, 0xff);
  if (status == AMDSMI_STATUS_SUCCESS) *uuid_length = AMDSMI_GPU_UUID_SIZE;
  return status;
}

}  // namespace amd::smi

#endif  // ENABLE_WSL_BACKEND
