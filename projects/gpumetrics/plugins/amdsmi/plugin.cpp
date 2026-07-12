// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// gpumetrics backend plugin for AMD SMI (libamd_smi). Exposes AMD GPU telemetry
// (temperature, power, clocks, activity, memory, ECC, PCIe, fans, energy, and a
// few gpu_metrics-derived fields) through the plugin ABI.
//
// Design notes:
//   * Topology is enumerated ONCE in init() and cached; read() never re-walks it.
//   * A single declarative table (kMetricTable) couples each descriptor with its
//     reader, so list_metrics() and read() share one source of truth.
//   * Backend "not supported" maps to GPUM_ERR_UNSUPPORTED; every per-field
//     sentinel (UINT32_MAX, 0xFFFF, UINT64_MAX) is treated as unsupported.

#include <amd_smi/amdsmi.h>

#include <cctype>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "gpumetrics/plugin_abi.h"
#include "gpumetrics/types.h"

namespace {

// Wall-clock timestamp in nanoseconds.
uint64_t NowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
         static_cast<uint64_t>(ts.tv_nsec);
}

gpum_status TranslateStatus(amdsmi_status_t s) {
  switch (s) {
    case AMDSMI_STATUS_SUCCESS:
      return GPUM_OK;
    case AMDSMI_STATUS_NOT_SUPPORTED:
      return GPUM_ERR_UNSUPPORTED;
    case AMDSMI_STATUS_NOT_INIT:
      return GPUM_ERR_NOT_INITIALIZED;
    case AMDSMI_STATUS_INVAL:
      return GPUM_ERR_INVALID_ARG;
    case AMDSMI_STATUS_NO_DATA:
      return GPUM_ERR_NO_DATA;
    default:
      return GPUM_ERR_BACKEND;
  }
}

// Per-field sentinels amdsmi uses for "unsupported/unavailable"; never surface
// these as real values.
bool IsSentinelU16(uint16_t v) { return v == 0xFFFFu; }
bool IsSentinelU32(uint32_t v) { return v == 0xFFFFFFFFu; }
bool IsSentinelU64(uint64_t v) { return v == 0xFFFFFFFFFFFFFFFFull; }

void SetU64(gpum_value* v, uint64_t x) {
  v->type = GPUM_TYPE_U64;
  v->u64 = x;
}
void SetI64(gpum_value* v, int64_t x) {
  v->type = GPUM_TYPE_I64;
  v->i64 = x;
}
void SetF64(gpum_value* v, double x) {
  v->type = GPUM_TYPE_F64;
  v->f64 = x;
}

// One device the plugin serves: the amdsmi processor handle plus the socket
// ordinal it was enumerated under, so read() dispatches without re-walking.
struct DeviceEntry {
  amdsmi_processor_handle handle;
  uint32_t socket_index;
};

}  // namespace

// The opaque ctx type declared in the ABI, defined and owned by the plugin.
struct gpum_plugin_ctx {
  bool amdsmi_initialized = false;
  std::vector<DeviceEntry> devices;
  // Storage handed to the core; valid until shutdown().
  std::vector<gpum_device_desc> device_descs;
};

namespace {

// A reader fills *out for one device handle and returns the amdsmi status;
// per-field sentinels are converted to AMDSMI_STATUS_NOT_SUPPORTED by the reader.
using ReaderFn = amdsmi_status_t (*)(amdsmi_processor_handle h, gpum_value* out);

// A declarative registry entry: descriptor metadata + its reader. Single source
// of truth shared by list_metrics() and read().
struct MetricEntry {
  const char* key;
  const char* unit;
  const char* description;
  gpum_value_type type;
  uint32_t scope;
  ReaderFn reader;
};

// --- Temperature (whole degrees Celsius, int64) ---
amdsmi_status_t ReadTemp(amdsmi_processor_handle h, amdsmi_temperature_type_t sensor,
                         gpum_value* out) {
  int64_t t = 0;
  amdsmi_status_t s = amdsmi_get_temp_metric(h, sensor, AMDSMI_TEMP_CURRENT, &t);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  SetI64(out, t);
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t ReadTempEdge(amdsmi_processor_handle h, gpum_value* o) {
  return ReadTemp(h, AMDSMI_TEMPERATURE_TYPE_EDGE, o);
}
amdsmi_status_t ReadTempHotspot(amdsmi_processor_handle h, gpum_value* o) {
  return ReadTemp(h, AMDSMI_TEMPERATURE_TYPE_HOTSPOT, o);
}
amdsmi_status_t ReadTempMem(amdsmi_processor_handle h, gpum_value* o) {
  return ReadTemp(h, AMDSMI_TEMPERATURE_TYPE_VRAM, o);
}

// --- Power ---
// *_socket_power fields are in Watts on bare-metal Linux, but which is populated
// depends on ASIC generation, and both 0xFFFF and 0xFFFFFFFF appear as
// "unavailable" sentinels:
//   - socket_power (u64):           older/general instantaneous
//   - current_socket_power (u32):   MI300+ instantaneous
//   - average_socket_power (u32):   Navi / MI200 and earlier
// A u32 field can even carry the 16-bit 0xFFFF sentinel (seen on MI350X), so
// treat any all-ones value as unavailable.
bool PowerValid16or32(uint32_t v) { return v != 0 && v != 0xFFFFu && v != 0xFFFFFFFFu; }

// Instantaneous socket power: prefer MI300+ current field, then socket_power,
// else average.
amdsmi_status_t ReadPowerCurrentSocket(amdsmi_processor_handle h, gpum_value* out) {
  amdsmi_power_info_t info{};
  amdsmi_status_t s = amdsmi_get_power_info(h, &info);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  if (PowerValid16or32(info.current_socket_power)) {
    SetU64(out, info.current_socket_power);
    return AMDSMI_STATUS_SUCCESS;
  }
  if (!IsSentinelU64(info.socket_power) && info.socket_power != 0 &&
      info.socket_power != 0xFFFFu) {
    SetU64(out, info.socket_power);
    return AMDSMI_STATUS_SUCCESS;
  }
  if (PowerValid16or32(info.average_socket_power)) {
    SetU64(out, info.average_socket_power);
    return AMDSMI_STATUS_SUCCESS;
  }
  return AMDSMI_STATUS_NOT_SUPPORTED;
}
// Average socket power: prefer the average field, else any populated
// instantaneous field.
amdsmi_status_t ReadPowerAverageSocket(amdsmi_processor_handle h, gpum_value* out) {
  amdsmi_power_info_t info{};
  amdsmi_status_t s = amdsmi_get_power_info(h, &info);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  if (PowerValid16or32(info.average_socket_power)) {
    SetU64(out, info.average_socket_power);
    return AMDSMI_STATUS_SUCCESS;
  }
  if (PowerValid16or32(info.current_socket_power)) {
    SetU64(out, info.current_socket_power);
    return AMDSMI_STATUS_SUCCESS;
  }
  if (!IsSentinelU64(info.socket_power) && info.socket_power != 0 &&
      info.socket_power != 0xFFFFu) {
    SetU64(out, info.socket_power);
    return AMDSMI_STATUS_SUCCESS;
  }
  return AMDSMI_STATUS_NOT_SUPPORTED;
}
amdsmi_status_t ReadPowerCap(amdsmi_processor_handle h, gpum_value* out) {
  amdsmi_power_cap_info_t info{};
  amdsmi_status_t s = amdsmi_get_power_cap_info(h, 0, &info);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  if (info.power_cap == 0 || IsSentinelU64(info.power_cap))
    return AMDSMI_STATUS_NOT_SUPPORTED;
  // power_cap is in microwatts on linux_bm; convert to Watts.
  SetF64(out, static_cast<double>(info.power_cap) / 1e6);
  return AMDSMI_STATUS_SUCCESS;
}

// --- Clocks ---
amdsmi_status_t ReadClock(amdsmi_processor_handle h, amdsmi_clk_type_t type,
                          gpum_value* out) {
  amdsmi_clk_info_t info{};
  amdsmi_status_t s = amdsmi_get_clock_info(h, type, &info);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  SetU64(out, info.clk);
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t ReadClockGfx(amdsmi_processor_handle h, gpum_value* o) {
  return ReadClock(h, AMDSMI_CLK_TYPE_GFX, o);
}
amdsmi_status_t ReadClockMem(amdsmi_processor_handle h, gpum_value* o) {
  return ReadClock(h, AMDSMI_CLK_TYPE_MEM, o);
}
amdsmi_status_t ReadClockSoc(amdsmi_processor_handle h, gpum_value* o) {
  return ReadClock(h, AMDSMI_CLK_TYPE_SOC, o);
}

// --- Activity ---
amdsmi_status_t ReadActivityGfx(amdsmi_processor_handle h, gpum_value* out) {
  amdsmi_engine_usage_t u{};
  amdsmi_status_t s = amdsmi_get_gpu_activity(h, &u);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  if (IsSentinelU32(u.gfx_activity)) return AMDSMI_STATUS_NOT_SUPPORTED;
  SetU64(out, u.gfx_activity);
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t ReadActivityUmc(amdsmi_processor_handle h, gpum_value* out) {
  amdsmi_engine_usage_t u{};
  amdsmi_status_t s = amdsmi_get_gpu_activity(h, &u);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  if (IsSentinelU32(u.umc_activity)) return AMDSMI_STATUS_NOT_SUPPORTED;
  SetU64(out, u.umc_activity);
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t ReadActivityMm(amdsmi_processor_handle h, gpum_value* out) {
  amdsmi_engine_usage_t u{};
  amdsmi_status_t s = amdsmi_get_gpu_activity(h, &u);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  if (IsSentinelU32(u.mm_activity)) return AMDSMI_STATUS_NOT_SUPPORTED;
  SetU64(out, u.mm_activity);
  return AMDSMI_STATUS_SUCCESS;
}

// --- Memory ---
amdsmi_status_t ReadVramUsed(amdsmi_processor_handle h, gpum_value* out) {
  uint64_t used = 0;
  amdsmi_status_t s = amdsmi_get_gpu_memory_usage(h, AMDSMI_MEM_TYPE_VRAM, &used);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  SetU64(out, used);
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t ReadVramTotal(amdsmi_processor_handle h, gpum_value* out) {
  uint64_t total = 0;
  amdsmi_status_t s = amdsmi_get_gpu_memory_total(h, AMDSMI_MEM_TYPE_VRAM, &total);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  SetU64(out, total);
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t ReadVramFree(amdsmi_processor_handle h, gpum_value* out) {
  uint64_t total = 0, used = 0;
  amdsmi_status_t s = amdsmi_get_gpu_memory_total(h, AMDSMI_MEM_TYPE_VRAM, &total);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  s = amdsmi_get_gpu_memory_usage(h, AMDSMI_MEM_TYPE_VRAM, &used);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  SetU64(out, total >= used ? total - used : 0);
  return AMDSMI_STATUS_SUCCESS;
}

// --- ECC ---
amdsmi_status_t ReadEccCorrectable(amdsmi_processor_handle h, gpum_value* out) {
  amdsmi_error_count_t ec{};
  amdsmi_status_t s = amdsmi_get_gpu_total_ecc_count(h, &ec);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  SetU64(out, ec.correctable_count);
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t ReadEccUncorrectable(amdsmi_processor_handle h, gpum_value* out) {
  amdsmi_error_count_t ec{};
  amdsmi_status_t s = amdsmi_get_gpu_total_ecc_count(h, &ec);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  SetU64(out, ec.uncorrectable_count);
  return AMDSMI_STATUS_SUCCESS;
}

// --- PCIe (from the pcie_metric sub-struct) ---
amdsmi_status_t ReadPcieSpeed(amdsmi_processor_handle h, gpum_value* out) {
  amdsmi_pcie_info_t info{};
  amdsmi_status_t s = amdsmi_get_pcie_info(h, &info);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  if (IsSentinelU32(info.pcie_metric.pcie_speed)) return AMDSMI_STATUS_NOT_SUPPORTED;
  SetU64(out, info.pcie_metric.pcie_speed);
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t ReadPcieWidth(amdsmi_processor_handle h, gpum_value* out) {
  amdsmi_pcie_info_t info{};
  amdsmi_status_t s = amdsmi_get_pcie_info(h, &info);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  if (IsSentinelU16(info.pcie_metric.pcie_width)) return AMDSMI_STATUS_NOT_SUPPORTED;
  SetU64(out, info.pcie_metric.pcie_width);
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t ReadPcieBandwidth(amdsmi_processor_handle h, gpum_value* out) {
  amdsmi_pcie_info_t info{};
  amdsmi_status_t s = amdsmi_get_pcie_info(h, &info);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  if (info.pcie_metric.pcie_bandwidth == 0 ||
      IsSentinelU32(info.pcie_metric.pcie_bandwidth))
    return AMDSMI_STATUS_NOT_SUPPORTED;
  SetU64(out, info.pcie_metric.pcie_bandwidth);  // Mb/s
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t ReadPcieReplayCount(amdsmi_processor_handle h, gpum_value* out) {
  amdsmi_pcie_info_t info{};
  amdsmi_status_t s = amdsmi_get_pcie_info(h, &info);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  if (IsSentinelU64(info.pcie_metric.pcie_replay_count))
    return AMDSMI_STATUS_NOT_SUPPORTED;
  SetU64(out, info.pcie_metric.pcie_replay_count);
  return AMDSMI_STATUS_SUCCESS;
}

// --- Fans ---
amdsmi_status_t ReadFanRpm(amdsmi_processor_handle h, gpum_value* out) {
  int64_t rpm = 0;
  amdsmi_status_t s = amdsmi_get_gpu_fan_rpms(h, 0, &rpm);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  SetI64(out, rpm);
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t ReadFanSpeed(amdsmi_processor_handle h, gpum_value* out) {
  int64_t speed = 0;
  amdsmi_status_t s = amdsmi_get_gpu_fan_speed(h, 0, &speed);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  SetI64(out, speed);
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t ReadFanSpeedMax(amdsmi_processor_handle h, gpum_value* out) {
  uint64_t max_speed = 0;
  amdsmi_status_t s = amdsmi_get_gpu_fan_speed_max(h, 0, &max_speed);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  SetU64(out, max_speed);
  return AMDSMI_STATUS_SUCCESS;
}

// --- Energy ---
amdsmi_status_t ReadEnergy(amdsmi_processor_handle h, gpum_value* out) {
  uint64_t acc = 0;
  float resolution = 0.0f;
  uint64_t ts = 0;
  amdsmi_status_t s = amdsmi_get_energy_count(h, &acc, &resolution, &ts);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  if (IsSentinelU64(acc)) return AMDSMI_STATUS_NOT_SUPPORTED;
  // energy = accumulator * resolution, in microjoules.
  SetF64(out, static_cast<double>(acc) * static_cast<double>(resolution));
  return AMDSMI_STATUS_SUCCESS;
}

// --- gpu_metrics-derived fields ---
amdsmi_status_t ReadXgmiRead0(amdsmi_processor_handle h, gpum_value* out) {
  amdsmi_gpu_metrics_t m{};
  amdsmi_status_t s = amdsmi_get_gpu_metrics_info(h, &m);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  if (IsSentinelU64(m.xgmi_read_data_acc[0])) return AMDSMI_STATUS_NOT_SUPPORTED;
  SetU64(out, m.xgmi_read_data_acc[0]);
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t ReadXgmiWrite0(amdsmi_processor_handle h, gpum_value* out) {
  amdsmi_gpu_metrics_t m{};
  amdsmi_status_t s = amdsmi_get_gpu_metrics_info(h, &m);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  if (IsSentinelU64(m.xgmi_write_data_acc[0])) return AMDSMI_STATUS_NOT_SUPPORTED;
  SetU64(out, m.xgmi_write_data_acc[0]);
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t ReadThrottleStatus(amdsmi_processor_handle h, gpum_value* out) {
  amdsmi_gpu_metrics_t m{};
  amdsmi_status_t s = amdsmi_get_gpu_metrics_info(h, &m);
  if (s != AMDSMI_STATUS_SUCCESS) return s;
  if (IsSentinelU32(m.throttle_status)) return AMDSMI_STATUS_NOT_SUPPORTED;
  SetU64(out, m.throttle_status);
  return AMDSMI_STATUS_SUCCESS;
}

// The metric registry: one entry per key, coupling descriptor and reader.
// list_metrics() derives descriptors from it; read() dispatches through it.
constexpr uint32_t kScopeGpu = GPUM_SCOPE_GPU;

const MetricEntry kMetricTable[] = {
    // Temperature
    {"temp.edge", "C", "Edge (die) temperature", GPUM_TYPE_I64, kScopeGpu, ReadTempEdge},
    {"temp.hotspot", "C", "Hotspot/junction temperature", GPUM_TYPE_I64, kScopeGpu,
     ReadTempHotspot},
    {"temp.mem", "C", "Memory (VRAM) temperature", GPUM_TYPE_I64, kScopeGpu, ReadTempMem},
    // Power
    {"power.current_socket", "W", "Current socket power draw", GPUM_TYPE_U64,
     GPUM_SCOPE_GPU | GPUM_SCOPE_SOCKET, ReadPowerCurrentSocket},
    {"power.average_socket", "W", "Average socket power draw", GPUM_TYPE_U64,
     GPUM_SCOPE_GPU | GPUM_SCOPE_SOCKET, ReadPowerAverageSocket},
    {"power.cap", "W", "Configured power cap", GPUM_TYPE_F64, kScopeGpu, ReadPowerCap},
    // Clocks
    {"clock.gfx", "MHz", "Graphics engine clock", GPUM_TYPE_U64, kScopeGpu, ReadClockGfx},
    {"clock.mem", "MHz", "Memory clock", GPUM_TYPE_U64, kScopeGpu, ReadClockMem},
    {"clock.soc", "MHz", "SoC clock", GPUM_TYPE_U64, kScopeGpu, ReadClockSoc},
    // Activity
    {"activity.gfx", "%", "Graphics engine utilization", GPUM_TYPE_U64, kScopeGpu,
     ReadActivityGfx},
    {"activity.umc", "%", "Memory (UMC) controller utilization", GPUM_TYPE_U64, kScopeGpu,
     ReadActivityUmc},
    {"activity.mm", "%", "Multimedia engine utilization", GPUM_TYPE_U64, kScopeGpu,
     ReadActivityMm},
    // Memory
    {"mem.vram.used", "bytes", "VRAM currently used", GPUM_TYPE_U64, kScopeGpu, ReadVramUsed},
    {"mem.vram.total", "bytes", "Total VRAM", GPUM_TYPE_U64, kScopeGpu, ReadVramTotal},
    {"mem.vram.free", "bytes", "Free VRAM (total-used)", GPUM_TYPE_U64, kScopeGpu, ReadVramFree},
    // ECC
    {"ecc.total.correctable", "", "Total correctable ECC errors", GPUM_TYPE_U64, kScopeGpu,
     ReadEccCorrectable},
    {"ecc.total.uncorrectable", "", "Total uncorrectable ECC errors", GPUM_TYPE_U64, kScopeGpu,
     ReadEccUncorrectable},
    // PCIe
    {"pcie.speed", "MT/s", "Current PCIe link speed", GPUM_TYPE_U64, kScopeGpu, ReadPcieSpeed},
    {"pcie.width", "", "Current PCIe link width (lanes)", GPUM_TYPE_U64, kScopeGpu,
     ReadPcieWidth},
    {"pcie.bandwidth", "Mb/s", "Instantaneous PCIe bandwidth", GPUM_TYPE_U64, kScopeGpu,
     ReadPcieBandwidth},
    {"pcie.replay_count", "", "PCIe replay count", GPUM_TYPE_U64, kScopeGpu, ReadPcieReplayCount},
    // Fans
    {"fan.rpm", "rpm", "Fan speed in RPM", GPUM_TYPE_I64, kScopeGpu, ReadFanRpm},
    {"fan.speed", "", "Fan speed (0..max)", GPUM_TYPE_I64, kScopeGpu, ReadFanSpeed},
    {"fan.speed_max", "", "Maximum fan speed", GPUM_TYPE_U64, kScopeGpu, ReadFanSpeedMax},
    // Energy
    {"power.energy", "uJ", "Accumulated energy consumption", GPUM_TYPE_F64, kScopeGpu, ReadEnergy},
    // gpu_metrics-derived
    {"xgmi.read_kb.0", "KB", "XGMI link 0 accumulated read data", GPUM_TYPE_U64, kScopeGpu,
     ReadXgmiRead0},
    {"xgmi.write_kb.0", "KB", "XGMI link 0 accumulated write data", GPUM_TYPE_U64, kScopeGpu,
     ReadXgmiWrite0},
    {"throttle.status", "", "Throttle status bitmask (0 = unthrottled)", GPUM_TYPE_U64, kScopeGpu,
     ReadThrottleStatus},
};

constexpr uint32_t kMetricCount = sizeof(kMetricTable) / sizeof(kMetricTable[0]);

// Descriptor array handed to the core, derived once from the table.
gpum_metric_desc g_metric_descs[kMetricCount];
bool g_metric_descs_built = false;

void BuildMetricDescs() {
  if (g_metric_descs_built) return;
  for (uint32_t i = 0; i < kMetricCount; ++i) {
    gpum_metric_desc& d = g_metric_descs[i];
    std::memset(&d, 0, sizeof(d));
    std::snprintf(d.key, sizeof(d.key), "%s", kMetricTable[i].key);
    std::snprintf(d.unit, sizeof(d.unit), "%s", kMetricTable[i].unit);
    std::snprintf(d.description, sizeof(d.description), "%s", kMetricTable[i].description);
    d.type = kMetricTable[i].type;
    d.scope = kMetricTable[i].scope;
  }
  g_metric_descs_built = true;
}

const MetricEntry* FindMetric(const char* key) {
  if (key == nullptr) return nullptr;
  for (uint32_t i = 0; i < kMetricCount; ++i) {
    if (std::strcmp(key, kMetricTable[i].key) == 0) return &kMetricTable[i];
  }
  return nullptr;
}

// Parse the amdsmi UUID hex string into 16 bytes, ignoring non-hex characters.
// Leaves the buffer all-zero if fewer than 32 hex digits are found.
void ParseUuid(const char* text, uint8_t out[16]) {
  std::memset(out, 0, 16);
  if (text == nullptr) return;
  int nibbles = 0;
  for (const char* p = text; *p && nibbles < 32; ++p) {
    int v;
    char c = *p;
    if (c >= '0' && c <= '9')
      v = c - '0';
    else if (c >= 'a' && c <= 'f')
      v = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F')
      v = c - 'A' + 10;
    else
      continue;
    if ((nibbles & 1) == 0)
      out[nibbles / 2] = static_cast<uint8_t>(v << 4);
    else
      out[nibbles / 2] |= static_cast<uint8_t>(v);
    ++nibbles;
  }
  if (nibbles < 32) std::memset(out, 0, 16);
}

// Fill a device_desc identity + name for one processor handle.
void FillDeviceDesc(amdsmi_processor_handle h, uint32_t socket_index,
                    uint32_t plugin_local_index, gpum_device_desc* desc) {
  std::memset(desc, 0, sizeof(*desc));
  gpum_device_identity& id = desc->identity;
  id.plugin_local_index = plugin_local_index;
  id.socket_id = socket_index;
  // Report raw identity; the core does grouping. On CPX each partition is a
  // distinct PCIe function with its own BDF/kfd/uuid. Partition index = PCIe
  // function (0 = whole-GPU handle).
  amdsmi_bdf_t bdf{};
  uint32_t bdf_function = 0;
  if (amdsmi_get_gpu_device_bdf(h, &bdf) == AMDSMI_STATUS_SUCCESS) {
    bdf_function = static_cast<uint32_t>(bdf.bdf.function_number);
    id.bdf = gpum_bdf_pack(static_cast<uint32_t>(bdf.bdf.domain_number),
                           static_cast<uint8_t>(bdf.bdf.bus_number),
                           static_cast<uint8_t>(bdf.bdf.device_number),
                           static_cast<uint8_t>(bdf.bdf.function_number));
  }
  id.partition_index = (bdf_function > 0) ? static_cast<int32_t>(bdf_function) : -1;

  // oam_id: per-board physical id, valid only on the whole-GPU handle
  // (partitions report the sentinel); the core's strongest grouping key.
  id.oam_id = GPUM_ID_UNKNOWN;
  amdsmi_asic_info_t asic{};
  bool have_asic = amdsmi_get_gpu_asic_info(h, &asic) == AMDSMI_STATUS_SUCCESS;
  if (have_asic && !IsSentinelU32(asic.oam_id)) id.oam_id = asic.oam_id;

  amdsmi_kfd_info_t kfd{};
  if (amdsmi_get_gpu_kfd_info(h, &kfd) == AMDSMI_STATUS_SUCCESS) {
    if (!IsSentinelU32(kfd.node_id)) id.kfd_node_id = kfd.node_id;
  }

  char uuid_str[AMDSMI_GPU_UUID_SIZE + 1] = {0};
  unsigned int uuid_len = AMDSMI_GPU_UUID_SIZE;
  if (amdsmi_get_gpu_device_uuid(h, &uuid_len, uuid_str) == AMDSMI_STATUS_SUCCESS) {
    ParseUuid(uuid_str, id.uuid);
  }

  amdsmi_enumeration_info_t en{};
  if (amdsmi_get_gpu_enumeration_info(h, &en) == AMDSMI_STATUS_SUCCESS) {
    if (!IsSentinelU32(en.drm_render)) id.drm_render_minor = en.drm_render;
  }

  // Name: prefer ASIC market name, fall back to board product name.
  if (have_asic && asic.market_name[0] != '\0') {
    std::snprintf(desc->name, sizeof(desc->name), "%s", asic.market_name);
  } else {
    amdsmi_board_info_t board{};
    if (amdsmi_get_gpu_board_info(h, &board) == AMDSMI_STATUS_SUCCESS &&
        board.product_name[0] != '\0') {
      std::snprintf(desc->name, sizeof(desc->name), "%s", board.product_name);
    } else {
      std::snprintf(desc->name, sizeof(desc->name), "AMD GPU");
    }
  }
}

// --- vtable implementation ---

gpum_status PluginInit(gpum_plugin_ctx** out_ctx) {
  if (out_ctx == nullptr) return GPUM_ERR_INVALID_ARG;
  *out_ctx = nullptr;

  amdsmi_status_t s = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  if (s != AMDSMI_STATUS_SUCCESS) return TranslateStatus(s);

  gpum_plugin_ctx* ctx = new (std::nothrow) gpum_plugin_ctx();
  if (ctx == nullptr) {
    amdsmi_shut_down();
    return GPUM_ERR_INTERNAL;
  }
  ctx->amdsmi_initialized = true;

  // Enumerate topology ONCE and cache it (two-call size-query pattern).
  uint32_t socket_count = 0;
  s = amdsmi_get_socket_handles(&socket_count, nullptr);
  if (s != AMDSMI_STATUS_SUCCESS) {
    amdsmi_shut_down();
    delete ctx;
    return TranslateStatus(s);
  }
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  if (socket_count > 0) {
    s = amdsmi_get_socket_handles(&socket_count, sockets.data());
    if (s != AMDSMI_STATUS_SUCCESS) {
      amdsmi_shut_down();
      delete ctx;
      return TranslateStatus(s);
    }
  }

  for (uint32_t si = 0; si < socket_count; ++si) {
    uint32_t proc_count = 0;
    if (amdsmi_get_processor_handles(sockets[si], &proc_count, nullptr) !=
        AMDSMI_STATUS_SUCCESS)
      continue;
    if (proc_count == 0) continue;
    std::vector<amdsmi_processor_handle> procs(proc_count);
    if (amdsmi_get_processor_handles(sockets[si], &proc_count, procs.data()) !=
        AMDSMI_STATUS_SUCCESS)
      continue;

    for (uint32_t pi = 0; pi < proc_count; ++pi) {
      // Unprefixed processor_type_t: primary name in older amd_smi (<=26.3) and
      // a back-compat alias in newer (>=26.5), so it compiles across versions.
      processor_type_t ptype = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
      if (amdsmi_get_processor_type(procs[pi], &ptype) != AMDSMI_STATUS_SUCCESS)
        continue;
      if (ptype != AMDSMI_PROCESSOR_TYPE_AMD_GPU) continue;
      ctx->devices.push_back({procs[pi], si});
    }
  }

  // Precompute device descriptors for enumerate().
  ctx->device_descs.resize(ctx->devices.size());
  for (uint32_t i = 0; i < ctx->devices.size(); ++i) {
    FillDeviceDesc(ctx->devices[i].handle, ctx->devices[i].socket_index, i,
                   &ctx->device_descs[i]);
  }

  BuildMetricDescs();
  *out_ctx = ctx;
  return GPUM_OK;
}

void PluginShutdown(gpum_plugin_ctx* ctx) {
  if (ctx == nullptr) return;
  if (ctx->amdsmi_initialized) amdsmi_shut_down();
  delete ctx;
}

gpum_status PluginEnumerate(gpum_plugin_ctx* ctx, const gpum_device_desc** out_devices,
                            uint32_t* out_count) {
  if (ctx == nullptr || out_devices == nullptr || out_count == nullptr)
    return GPUM_ERR_INVALID_ARG;
  *out_devices = ctx->device_descs.empty() ? nullptr : ctx->device_descs.data();
  *out_count = static_cast<uint32_t>(ctx->device_descs.size());
  return GPUM_OK;
}

gpum_status PluginListMetrics(gpum_plugin_ctx* ctx, const gpum_metric_desc** out_metrics,
                              uint32_t* out_count) {
  if (ctx == nullptr || out_metrics == nullptr || out_count == nullptr)
    return GPUM_ERR_INVALID_ARG;
  BuildMetricDescs();
  *out_metrics = g_metric_descs;
  *out_count = kMetricCount;
  return GPUM_OK;
}

gpum_status PluginRead(gpum_plugin_ctx* ctx, const gpum_read_req* reqs, uint32_t n,
                       gpum_sample* out_samples) {
  if (ctx == nullptr || (n > 0 && (reqs == nullptr || out_samples == nullptr)))
    return GPUM_ERR_INVALID_ARG;

  for (uint32_t i = 0; i < n; ++i) {
    gpum_sample& out = out_samples[i];
    std::memset(&out, 0, sizeof(out));
    out.timestamp_ns = NowNs();

    const gpum_read_req& req = reqs[i];

    if (req.plugin_local_index >= ctx->devices.size()) {
      out.status = GPUM_ERR_NOT_FOUND;
      continue;
    }
    amdsmi_processor_handle h = ctx->devices[req.plugin_local_index].handle;

    const MetricEntry* entry = FindMetric(req.key);
    if (entry == nullptr) {
      out.status = GPUM_ERR_NOT_FOUND;
      continue;
    }

    gpum_value value{};
    amdsmi_status_t s = entry->reader(h, &value);
    if (s != AMDSMI_STATUS_SUCCESS) {
      out.status = TranslateStatus(s);
      out.type = entry->type;
      continue;
    }
    out.status = GPUM_OK;
    out.type = value.type;
    out.value = value;
  }

  return GPUM_OK;
}

// The immortal vtable returned by the entry point.
const gpum_plugin_v1 g_vtable = {
    /*abi_version=*/GPUM_PLUGIN_ABI_V1,
    /*name=*/"amdsmi",
    /*init=*/PluginInit,
    /*shutdown=*/PluginShutdown,
    /*enumerate=*/PluginEnumerate,
    /*list_metrics=*/PluginListMetrics,
    /*read=*/PluginRead,
};

}  // namespace

extern "C" const gpum_plugin_v1* gpum_plugin_entry_v1(void) { return &g_vtable; }
