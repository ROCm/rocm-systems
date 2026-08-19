// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

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

#ifndef REG_SZ
#define REG_SZ 1ul  // Unicode nul terminated string (from winnt.h)
#endif

namespace {

// Read a REG_SZ value from the adapter driver registry key.
static bool query_adapter_reg_str(D3DKMT_HANDLE adapter, const char* key_name, char* buf_out,
                                  size_t buf_len) {
  static constexpr uint32_t kMaxOutputSize = 512;
  struct RegQuery {
    D3DDDI_QUERYREGISTRY_INFO info;
    wchar_t output[kMaxOutputSize];
  } q = {};
  q.info.QueryType = D3DDDI_QUERYREGISTRY_ADAPTERKEY;
  q.info.QueryFlags.TranslatePath = 0;
  q.info.ValueType = REG_SZ;
  if (mbstowcs(q.info.ValueName, key_name, MAX_PATH) == static_cast<size_t>(-1)) return false;
  D3DKMT_QUERYADAPTERINFO args = {};
  args.hAdapter = adapter;
  args.Type = KMTQAITYPE_QUERYREGISTRY;
  args.pPrivateDriverData = &q;
  args.PrivateDriverDataSize = sizeof(q);
  if (DXCORE_CALL(D3DKMTQueryAdapterInfo(&args)) != STATUS_SUCCESS) return false;
  if (q.info.Status != D3DDDI_QUERYREGISTRY_STATUS_SUCCESS) return false;
  if (buf_out && buf_len > 0) {
    wcstombs(buf_out, q.info.OutputString, buf_len - 1);
    buf_out[buf_len - 1] = '\0';
  }
  return true;
}


wsl::thunk::WDDMDevice* checked_device(uint32_t node_id) {
  if (dxg_runtime == nullptr || dxg_runtime->dxg_open_count == 0 || dxg_runtime->is_forked) {
    return nullptr;
  }
  return get_wddmdev(node_id);
}

// Map NTSTATUS → HSAKMT_STATUS
HSAKMT_STATUS nt_to_hsa(NTSTATUS status) {
  switch (status) {
    case STATUS_SUCCESS:
      return HSAKMT_STATUS_SUCCESS;
    case STATUS_NOT_SUPPORTED:
    case STATUS_NOT_IMPLEMENTED:
      return HSAKMT_STATUS_NOT_SUPPORTED;
    case STATUS_NO_MEMORY:
      return HSAKMT_STATUS_NO_MEMORY;
    case STATUS_BUFFER_TOO_SMALL:
      return HSAKMT_STATUS_BUFFER_TOO_SMALL;
    case STATUS_INVALID_PARAMETER:
      return HSAKMT_STATUS_INVALID_PARAMETER;
    default:
      return HSAKMT_STATUS_ERROR;
  }
}

// Look up a PMLog sensor value by sensor ID.
//
// QueryPMLogData returns pmlog.supported[sensor_id] / pmlog.value[sensor_id] — it
// uses the sensor_id directly as the array index, NOT the slot position that
// QueryPMLogSupport's sensor_ids[] map would give. So we index pmlog directly
// by the sensor_id enum value; no slot map is needed here.
static uint32_t pmlog_sensor(const Wkmi::PmlogQueryResult& pmlog, Wkmi::PmlogSensorId id) {
  uint32_t idx = static_cast<uint32_t>(id);
  if (idx >= Wkmi::kPmlogMaxSensors) return Wkmi::kSensorUnavailable;
  if (!pmlog.supported[idx]) return Wkmi::kSensorUnavailable;
  return pmlog.value[idx];
}

template <typename T> HSAKMT_STATUS clear_out(T* out) {
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
// Size must match the dxgkrnl ioctl ABI: LX_DXENUMPROCESSES (nr=72, size=0x18 = 24 bytes).
// Verified by decoding ioctl code 0xc0184748 from libdxcore.so D3DKMTEnumProcesses.
static_assert(sizeof(D3DKMT_ENUMPROCESSES) == 0x18,
              "D3DKMT_ENUMPROCESSES size mismatch vs dxgkrnl ioctl ABI");

uint64_t target_graphics_version(wsl::thunk::WDDMDevice& device) {
  return (static_cast<uint64_t>(device.Major()) << 16) |
      (static_cast<uint64_t>(device.Minor()) << 8) | static_cast<uint64_t>(device.Stepping());
}

HSAKMT_STATUS query_process_vram(wsl::thunk::WDDMDevice& device, uint32_t pid,
                                 uint64_t* vram_bytes) {
  if (vram_bytes == nullptr) return HSAKMT_STATUS_INVALID_PARAMETER;
  if (!wsl::thunk::d3dthunk::QueryVideoMemoryInfoAvailable()) {
    return HSAKMT_STATUS_NOT_SUPPORTED;
  }

  D3DKMT_QUERYVIDEOMEMORYINFO args = {};
  // NOTE: hProcess normally expects a real process handle (e.g. from
  // OpenProcess); this passes the raw PID instead. Unconfirmed whether
  // WSL2's dxgkrnl path accepts a PID here — pending wkmi team confirmation.
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

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_power_info(uint32_t node_id, rocdxg_smi_power_info_t* info) {
  HSAKMT_STATUS status = clear_out(info);
  if (status != HSAKMT_STATUS_SUCCESS) return status;

  auto* wdev = checked_device(node_id);
  if (wdev == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  Wkmi::PmlogQueryResult pmlog = {};
  NTSTATUS ret = Wkmi::QueryPMLogData(wdev->GetAdapter(), wdev->DeviceHandle(), &pmlog);
  if (ret != STATUS_SUCCESS) return nt_to_hsa(ret);

  auto v = [&](Wkmi::PmlogSensorId id) { return pmlog_sensor(pmlog, id); };

  info->current_socket_power = v(Wkmi::kPmlogBoardPower) != Wkmi::kSensorUnavailable
      ? v(Wkmi::kPmlogBoardPower)
      : v(Wkmi::kPmlogAsicPower);
  info->gfx_voltage = v(Wkmi::kPmlogGfxVoltage);
  info->soc_voltage = v(Wkmi::kPmlogSocVoltage);
  info->mem_voltage = v(Wkmi::kPmlogMemVoltage);

  // power_limit from sensor limits (max of asic/board power sensor). Like
  // PmlogQueryResult, PmlogSensorLimits is id-indexed, not slot-indexed:
  // QueryPMLogSensorLimits (wkmi.cpp) memcpy's the raw KMD struct straight
  // into limits[] with no slot remapping, same layout as QueryPMLogData.
  Wkmi::PmlogSensorLimits limits = {};
  if (Wkmi::QueryPMLogSensorLimits(wdev->GetAdapter(), wdev->DeviceHandle(), &limits) ==
      STATUS_SUCCESS) {
    uint32_t board_id = static_cast<uint32_t>(Wkmi::kPmlogBoardPower);
    uint32_t asic_id = static_cast<uint32_t>(Wkmi::kPmlogAsicPower);
    if (v(Wkmi::kPmlogBoardPower) != Wkmi::kSensorUnavailable && board_id < Wkmi::kPmlogMaxSensors)
      info->power_limit = limits.limits[board_id][1];  // max
    else if (asic_id < Wkmi::kPmlogMaxSensors)
      info->power_limit = limits.limits[asic_id][1];  // max
  }
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_temperature(uint32_t node_id, uint32_t sensor_type,
                                                   uint32_t metric, int64_t* temperature) {
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
    case 0:
      primary = Wkmi::kPmlogTempEdge;
      fallback = Wkmi::kPmlogTempGfx;
      break;
    case 1:
      primary = Wkmi::kPmlogTempHotspot;
      fallback = Wkmi::kPmlogTempSoc;
      break;
    case 2:
      primary = Wkmi::kPmlogTempMem;
      fallback = Wkmi::kPmlogTempMem;
      break;
    default:
      return HSAKMT_STATUS_NOT_SUPPORTED;
  }

  Wkmi::PmlogQueryResult pmlog = {};
  NTSTATUS ret = Wkmi::QueryPMLogData(wdev->GetAdapter(), wdev->DeviceHandle(), &pmlog);
  if (ret != STATUS_SUCCESS) return nt_to_hsa(ret);

  uint32_t val = pmlog_sensor(pmlog, primary);
  if (val == Wkmi::kSensorUnavailable) val = pmlog_sensor(pmlog, fallback);
  if (val == Wkmi::kSensorUnavailable) return HSAKMT_STATUS_NOT_SUPPORTED;

  *temperature = static_cast<int64_t>(val);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_clock_info(uint32_t node_id, uint32_t clk_type,
                                                  rocdxg_smi_clock_info_t* info) {
  HSAKMT_STATUS status = clear_out(info);
  if (status != HSAKMT_STATUS_SUCCESS) return status;

  auto* wdev = checked_device(node_id);
  if (wdev == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  // Map amdsmi clk_type to PMLog sensor IDs:
  // 0=GFX/SYS → kPmlogGfxClk(1), 4=MEM → kPmlogMemClk(2), 3=SOC → kPmlogSocClk(3)
  Wkmi::PmlogSensorId target;
  switch (clk_type) {
    case 0:
      target = Wkmi::kPmlogGfxClk;
      break;
    case 3:
      target = Wkmi::kPmlogSocClk;
      break;
    case 4:
      target = Wkmi::kPmlogMemClk;
      break;
    default:
      return HSAKMT_STATUS_NOT_SUPPORTED;
  }

  Wkmi::PmlogQueryResult pmlog = {};
  NTSTATUS ret = Wkmi::QueryPMLogData(wdev->GetAdapter(), wdev->DeviceHandle(), &pmlog);
  if (ret == STATUS_SUCCESS) {
    uint32_t cur = pmlog_sensor(pmlog, target);
    if (cur != Wkmi::kSensorUnavailable) {
      info->clk = cur;
      // Static max clocks from adapter info
      info->max_clk = (clk_type == 4) ? wdev->MaxMemoryClockMhz() : wdev->MaxEngineClockMhz();
      return HSAKMT_STATUS_SUCCESS;
    }
  }

  // Fallback: static max only for GFX and MEM
  if (clk_type == 0) {
    info->max_clk = wdev->MaxEngineClockMhz();
    return HSAKMT_STATUS_SUCCESS;
  }
  if (clk_type == 4) {
    info->max_clk = wdev->MaxMemoryClockMhz();
    return HSAKMT_STATUS_SUCCESS;
  }
  return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_pcie_info(uint32_t node_id, rocdxg_smi_pcie_info_t* info) {
  HSAKMT_STATUS status = clear_out(info);
  if (status != HSAKMT_STATUS_SUCCESS) return status;

  auto* wdev = checked_device(node_id);
  if (wdev == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  // Static PCIe capabilities from CWDDECI_CHIPSETIDENTIFICATION. Not fatal if
  // this escape is unsupported by the driver: dynamic width/speed below is
  // queried independently via PMLog, so we still return what we can.
  Wkmi::ChipsetIdInfo ci = {};
  NTSTATUS ret = Wkmi::QueryChipsetId(wdev->GetAdapter(), wdev->DeviceHandle(), &ci);
  if (ret == STATUS_SUCCESS) {
    info->max_pcie_width = static_cast<uint16_t>(ci.max_pcie_lane_width);
    info->pcie_interface_version = ci.pcie_gen;
    // speed in MT/s: gen1=2500, gen2=5000, gen3=8000, gen4=16000, gen5=32000
    static const uint32_t kGenSpeed[] = {0, 2500, 5000, 8000, 16000, 32000};
    uint32_t gen = (ci.pcie_gen < 6) ? ci.pcie_gen : 0;
    info->max_pcie_speed = kGenSpeed[gen];
  } else {
    info->max_pcie_width = std::numeric_limits<uint16_t>::max();
    info->max_pcie_speed = std::numeric_limits<uint32_t>::max();
    info->pcie_interface_version = std::numeric_limits<uint32_t>::max();
  }

  // Dynamic current width/speed from PMLog BUS_LANES and BUS_SPEED
  Wkmi::PmlogQueryResult pmlog = {};
  if (Wkmi::QueryPMLogData(wdev->GetAdapter(), wdev->DeviceHandle(), &pmlog) == STATUS_SUCCESS) {
    uint32_t lanes = pmlog_sensor(pmlog, Wkmi::kPmlogBusLanes);
    uint32_t speed = pmlog_sensor(pmlog, Wkmi::kPmlogBusSpeed);
    if (lanes != Wkmi::kSensorUnavailable) info->pcie_width = static_cast<uint16_t>(lanes);
    if (speed != Wkmi::kSensorUnavailable) info->pcie_speed = speed;
  }
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_gpu_metrics_info(uint32_t node_id,
                                                        rocdxg_smi_gpu_metrics_info_t* info) {
  HSAKMT_STATUS status = clear_out(info);
  if (status != HSAKMT_STATUS_SUCCESS) return status;

  auto* wdev = checked_device(node_id);
  if (wdev == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  Wkmi::PmlogQueryResult pmlog = {};
  NTSTATUS ret = Wkmi::QueryPMLogData(wdev->GetAdapter(), wdev->DeviceHandle(), &pmlog);
  if (ret != STATUS_SUCCESS) return nt_to_hsa(ret);

  auto v = [&](Wkmi::PmlogSensorId id) -> uint32_t { return pmlog_sensor(pmlog, id); };

  // Fields below propagate Wkmi::kSensorUnavailable (all-ones uint32) verbatim
  // when a sensor is absent, matching this codebase's all-ones "unset"
  // convention (e.g. amdsmi's UINTx_MAX). Callers narrowing to a smaller type
  // must compare against their own type's max before truncating, the way
  // amd_smi_wsl_device.cc::GetGpuMetricsInfo does for these fields.
  info->temperature_edge = v(Wkmi::kPmlogTempEdge);
  info->temperature_hotspot = v(Wkmi::kPmlogTempHotspot);
  info->temperature_mem = v(Wkmi::kPmlogTempMem);
  info->average_gfx_activity = v(Wkmi::kPmlogGfxActivity);
  info->average_umc_activity = v(Wkmi::kPmlogMemActivity);
  uint32_t bp = v(Wkmi::kPmlogBoardPower);
  info->current_socket_power = (bp != Wkmi::kSensorUnavailable) ? bp : v(Wkmi::kPmlogAsicPower);
  info->current_gfxclk = v(Wkmi::kPmlogGfxClk);
  info->current_socclk = v(Wkmi::kPmlogSocClk);
  info->current_fan_speed = v(Wkmi::kPmlogFanRpm);
  info->current_fan_speed_percent = v(Wkmi::kPmlogFanPercent);
  info->voltage_soc = v(Wkmi::kPmlogSocVoltage);
  info->voltage_gfx = v(Wkmi::kPmlogGfxVoltage);
  info->voltage_mem = v(Wkmi::kPmlogMemVoltage);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_enum_processes(uint32_t node_id, uint32_t* num_processes,
                                                  rocdxg_smi_process_info_t* processes) {
  if (num_processes == nullptr) return HSAKMT_STATUS_INVALID_PARAMETER;

  auto* device = checked_device(node_id);
  if (device == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;
  if (!wsl::thunk::d3dthunk::EnumProcessesAvailable()) {
    return HSAKMT_STATUS_NOT_SUPPORTED;
  }

  // capacity 0 (the "how many?" query form) is capped at 256; if more than
  // 256 processes have the adapter open, the reported count is bounded by
  // this buffer size rather than the true total.
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

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_device_info(uint32_t node_id,
                                                   rocdxg_smi_device_info_t* info) {
  if (info == nullptr) return HSAKMT_STATUS_INVALID_PARAMETER;
  // Rejects a caller built against a different layout before anything is written.
  if (info->struct_size != sizeof(*info)) return HSAKMT_STATUS_BUFFER_TOO_SMALL;
  auto* wdev = checked_device(node_id);
  if (wdev == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  std::memset(info, 0, sizeof(*info));

  // BDF
  {
    const uint32_t loc = wdev->PciBusAddr();
    info->bdf.domain_number = wdev->Domain();
    info->bdf.bus_number = (loc >> 8) & 0xff;
    info->bdf.device_number = (loc >> 3) & 0x1f;
    info->bdf.function_number = loc & 0x7;
  }

  // ASIC
  {
    info->asic.device_id = wdev->DeviceId();
    info->asic.vendor_id = 0x1002;
    info->asic.subvendor_id = std::numeric_limits<uint32_t>::max();
    info->asic.subsystem_id = std::numeric_limits<uint32_t>::max();
    D3DKMT_QUERY_DEVICE_IDS dev_ids = {};
    D3DKMT_QUERYADAPTERINFO qi_ids = {};
    qi_ids.hAdapter = wdev->GetAdapter();
    qi_ids.Type = KMTQAITYPE_PHYSICALADAPTERDEVICEIDS;
    qi_ids.pPrivateDriverData = &dev_ids;
    qi_ids.PrivateDriverDataSize = sizeof(dev_ids);
    if (DXCORE_CALL(D3DKMTQueryAdapterInfo(&qi_ids)) == STATUS_SUCCESS) {
      // Empirically verified against the reference amd-smi tool on the same
      // hardware: D3DKMT's SubSystemID/SubVendorID map to amd-smi's
      // subvendor_id/subsystem_id in swapped order.
      info->asic.subvendor_id = dev_ids.DeviceIds.SubSystemID;
      info->asic.subsystem_id = dev_ids.DeviceIds.SubVendorID;
    }
    info->asic.rev_id = wdev->AsicRevision();
    info->asic.asic_serial = wdev->Uuid();
    info->asic.num_of_compute_units = wdev->ComputeUnitCount();
    info->asic.num_xcc = wdev->NumXcc();
    info->asic.target_graphics_version = target_graphics_version(*wdev);
    copy_string(info->asic.market_name, wdev->ProductName());
  }

  // Board + VBIOS name: query AdapterString once
  {
    char adapter_string[MAX_PATH] = {};
    D3DKMT_ADAPTERREGISTRYINFO ri = {};
    D3DKMT_QUERYADAPTERINFO qi = {};
    qi.hAdapter = wdev->GetAdapter();
    qi.Type = KMTQAITYPE_ADAPTERREGISTRYINFO;
    qi.pPrivateDriverData = &ri;
    qi.PrivateDriverDataSize = sizeof(ri);
    if (DXCORE_CALL(D3DKMTQueryAdapterInfo(&qi)) == STATUS_SUCCESS && ri.AdapterString[0])
      wcstombs(adapter_string, ri.AdapterString, sizeof(adapter_string) - 1);

    copy_string(info->board.product_name, adapter_string[0] ? adapter_string : wdev->ProductName());
    copy_string(info->board.manufacturer_name, "Advanced Micro Devices, Inc. [AMD/ATI]");
    copy_string(info->vbios.name, adapter_string[0] ? adapter_string : wdev->ProductName());
  }

  // VRAM
  {
    info->vram.vram_type = 0;
    info->vram.vram_bit_width = wdev->MemoryBusWidth();
    info->vram.vram_size_mb = wdev->LocalHeapSize() / (1024 * 1024);
  }

  // Driver (registry)
  {
    char release_version[256] = {};
    char driver_desc[256] = {};
    query_adapter_reg_str(wdev->GetAdapter(), "ReleaseVersion", release_version,
                          sizeof(release_version));
    query_adapter_reg_str(wdev->GetAdapter(), "DriverDesc", driver_desc, sizeof(driver_desc));
    copy_string(info->driver.driver_version, release_version);
    copy_string(info->driver.driver_name, driver_desc);
    const char* dash = std::strchr(release_version, '-');
    if (dash && *(dash + 1) != '\0') {
      char date_buf[ROCDXG_SMI_MAX_STRING_LENGTH] = {};
      std::strncpy(date_buf, dash + 1, 6);
      copy_string(info->driver.driver_date, date_buf);
    } else {
      copy_string(info->driver.driver_date, "N/A");
    }
  }

  // VBIOS (escape)
  {
    Wkmi::VideoBiosInfo vbios = {};
    if (Wkmi::QueryVideoBiosInfo(wdev->GetAdapter(), wdev->DeviceHandle(), &vbios) ==
        STATUS_SUCCESS) {
      copy_string(info->vbios.version, vbios.version);
      copy_string(info->vbios.part_number, vbios.part_number);
      copy_string(info->vbios.build_date, vbios.date);
    }
  }

  // Cache. cache_properties bits match amdsmi's AMDSMI_CACHE_PROPERTY_*
  // (amdsmi.h): ENABLED=0x1, DATA_CACHE=0x2, INST_CACHE=0x4, CPU_CACHE=0x8,
  // SIMD_CACHE=0x10. L1 below omits ENABLED (0x2, not 0x3) unlike L2/L3 —
  // unclear if intentional; flagging rather than silently changing it.
  {
    constexpr uint32_t kCacheDataOnly = 0x2;     // DATA_CACHE
    constexpr uint32_t kCacheEnabledData = 0x3;  // ENABLED | DATA_CACHE
    uint32_t idx = 0;
    if (wdev->GetL1CacheSize() > 0)
      info->cache.cache[idx++] = {wdev->GetL1CacheSize() / 1024, 1, kCacheDataOnly, 2, 1};
    if (wdev->GetL2CacheSize() > 0)
      info->cache.cache[idx++] = {wdev->GetL2CacheSize() / 1024, 2, kCacheEnabledData,
                                  wdev->ComputeUnitCount(), 1};
    if (wdev->GetL3CacheSize() > 0)
      info->cache.cache[idx++] = {wdev->GetL3CacheSize() / 1024, 3, kCacheEnabledData,
                                  wdev->ComputeUnitCount(), 1};
    info->cache.num_cache_types = idx;
  }

  // FW ids match amdsmi's AMDSMI_FW_ID_* enum (amdsmi.h); this lower-level
  // lib doesn't depend on that header directly, so the values are inlined.
  {
    constexpr uint32_t kFwIdMec1 = 7;    // AMDSMI_FW_ID_CP_MEC1
    constexpr uint32_t kFwIdSdma0 = 10;  // AMDSMI_FW_ID_SDMA0
    uint32_t idx = 0;
    if (wdev->GetMecFwVersion()) info->fw.entries[idx++] = {kFwIdMec1, wdev->GetMecFwVersion()};
    if (wdev->GetSdmaFwVersion()) info->fw.entries[idx++] = {kFwIdSdma0, wdev->GetSdmaFwVersion()};
    info->fw.num_fw_info = idx;
  }

  return HSAKMT_STATUS_SUCCESS;
}

}  // extern "C"
