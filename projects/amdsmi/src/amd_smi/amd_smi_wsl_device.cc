// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifdef ENABLE_WSL_BACKEND

#include "amd_smi/impl/amd_smi_wsl_device.h"

#include <dlfcn.h>
#include <fcntl.h>
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
#include "amd_smi/impl/amd_smi_utils.h"
#include "amd_smi/impl/amd_smi_uuid.h"
#include "amd_smi/impl/amd_smi_wsl_syms.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace amd::smi {
namespace {

static amdsmi_bdf_t make_bdf(const HsaNodeProperties& props) {
  amdsmi_bdf_t bdf = {};
  bdf.bdf.domain_number = props.Domain;
  bdf.bdf.bus_number = (props.LocationId >> 8) & 0xff;
  bdf.bdf.device_number = (props.LocationId >> 3) & 0x1f;
  bdf.bdf.function_number = props.LocationId & 0x7;
  return bdf;
}

static std::string marketing_name_from_hsa(const HsaNodeProperties& props) {
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

static void copy_string(char* dst, const std::string& src) {
  if (dst == nullptr) return;
  std::memset(dst, 0, AMDSMI_MAX_STRING_LENGTH);
  std::snprintf(dst, AMDSMI_MAX_STRING_LENGTH, "%s", src.c_str());
}

static void copy_rocdxg_string(char* dst, const char* src) {
  if (dst == nullptr) return;
  std::memset(dst, 0, AMDSMI_MAX_STRING_LENGTH);
  if (src != nullptr) std::snprintf(dst, AMDSMI_MAX_STRING_LENGTH, "%s", src);
}

static constexpr const char* kDxgDevPath = "/dev/dxg";
static constexpr const char* kRocdxgSoV = "librocdxg.so.1";
static constexpr const char* kRocdxgSo = "librocdxg.so";

// librocdxg handle — owned by try_populate, valid for the process lifetime.
static void* g_rocdxg_handle = nullptr;

// Helper: resolve one dlsym symbol; print and return false on failure.
template <typename T>
static bool bind_sym(void* handle, const char* name, T& fn) {
  fn = reinterpret_cast<T>(dlsym(handle, name));
  if (!fn) {
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | missing symbol: " << name;
    LOG_ERROR(ss);
    return false;
  }
  return true;
}

// dlopen librocdxg.so and bind all required symbols into g_wsl_syms.
static bool load_rocdxg() {
  if (g_rocdxg_handle) return true;
  g_rocdxg_handle = dlopen(kRocdxgSoV, RTLD_NOW | RTLD_LOCAL);
  if (!g_rocdxg_handle) g_rocdxg_handle = dlopen(kRocdxgSo, RTLD_NOW | RTLD_LOCAL);
  if (!g_rocdxg_handle) {
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | dlopen librocdxg failed: " << dlerror();
    LOG_ERROR(ss);
    return false;
  }

  bool ok = true;
  ok &= bind_sym(g_rocdxg_handle, "hsaKmtOpenKFD", g_wsl_syms.hsaKmtOpenKFD);
  ok &= bind_sym(g_rocdxg_handle, "hsaKmtCloseKFD", g_wsl_syms.hsaKmtCloseKFD);
  ok &= bind_sym(g_rocdxg_handle, "hsaKmtAcquireSystemProperties",
                 g_wsl_syms.hsaKmtAcquireSystemProperties);
  ok &= bind_sym(g_rocdxg_handle, "hsaKmtReleaseSystemProperties",
                 g_wsl_syms.hsaKmtReleaseSystemProperties);
  ok &= bind_sym(g_rocdxg_handle, "hsaKmtGetNodeProperties", g_wsl_syms.hsaKmtGetNodeProperties);
  ok &= bind_sym(g_rocdxg_handle, "rocdxg_smi_get_device_info",
                 g_wsl_syms.rocdxg_smi_get_device_info);
  ok &=
      bind_sym(g_rocdxg_handle, "rocdxg_smi_get_vram_usage", g_wsl_syms.rocdxg_smi_get_vram_usage);
  ok &=
      bind_sym(g_rocdxg_handle, "rocdxg_smi_get_power_info", g_wsl_syms.rocdxg_smi_get_power_info);
  ok &= bind_sym(g_rocdxg_handle, "rocdxg_smi_get_temperature",
                 g_wsl_syms.rocdxg_smi_get_temperature);
  ok &=
      bind_sym(g_rocdxg_handle, "rocdxg_smi_get_clock_info", g_wsl_syms.rocdxg_smi_get_clock_info);
  ok &= bind_sym(g_rocdxg_handle, "rocdxg_smi_get_pcie_info", g_wsl_syms.rocdxg_smi_get_pcie_info);
  ok &= bind_sym(g_rocdxg_handle, "rocdxg_smi_get_gpu_metrics_info",
                 g_wsl_syms.rocdxg_smi_get_gpu_metrics_info);
  ok &=
      bind_sym(g_rocdxg_handle, "rocdxg_smi_enum_processes", g_wsl_syms.rocdxg_smi_enum_processes);

  if (!ok) {
    dlclose(g_rocdxg_handle);
    g_rocdxg_handle = nullptr;
    g_wsl_syms = WslSyms{};
    return false;
  }
  return true;
}

static amdsmi_status_t hsakmt_to_amdsmi(HSAKMT_STATUS s) {
  switch (s) {
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

// Placeholder DRM instance for WSL devices — WSL has no /dev/dri.
static AMDSmiDrm g_wsl_drm;

}  // namespace

// Definition of the global WSL symbol table (declared extern in amd_smi_wsl_syms.h).
WslSyms g_wsl_syms;

// True iff try_populate succeeded — tracks whether hsaKmtCloseKFD is needed.
static bool g_wsl_active = false;

// -----------------------------------------------------------------------------
// WSLGPUBackend constructor and try_populate
// -----------------------------------------------------------------------------

WSLGPUBackend::WSLGPUBackend(uint32_t gpu_id, uint32_t node_id, const HsaNodeProperties& props)
    : gpu_id_(gpu_id),
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

amdsmi_status_t WSLGPUBackend::try_populate(std::vector<AMDSmiSocket*>& sockets,
                                            std::set<AMDSmiProcessor*>& processors) {
  // Not WSL if /dev/dxg is absent.
  if (access(kDxgDevPath, F_OK) != 0) return AMDSMI_STATUS_NOT_SUPPORTED;

  if (!load_rocdxg()) return AMDSMI_STATUS_DRIVER_NOT_LOADED;

  HSAKMT_STATUS hstatus = g_wsl_syms.hsaKmtOpenKFD();
  if (hstatus != HSAKMT_STATUS_SUCCESS && hstatus != HSAKMT_STATUS_KERNEL_ALREADY_OPENED) {
    return hsakmt_to_amdsmi(hstatus);
  }

  HsaSystemProperties system_props = {};
  hstatus = g_wsl_syms.hsaKmtAcquireSystemProperties(&system_props);
  if (hstatus != HSAKMT_STATUS_SUCCESS) {
    g_wsl_syms.hsaKmtCloseKFD();
    return hsakmt_to_amdsmi(hstatus);
  }

  uint32_t gpu_index = 0;
  for (uint32_t node_id = 0; node_id < system_props.NumNodes; ++node_id) {
    HsaNodeProperties node_props = {};
    hstatus = g_wsl_syms.hsaKmtGetNodeProperties(node_id, &node_props);
    if (hstatus != HSAKMT_STATUS_SUCCESS) {
      g_wsl_syms.hsaKmtReleaseSystemProperties();
      g_wsl_syms.hsaKmtCloseKFD();
      return hsakmt_to_amdsmi(hstatus);
    }

    if (node_props.NumFComputeCores == 0) continue;

    auto* backend = new WSLGPUBackend(gpu_index, node_id, node_props);

    std::string path = "wsl_node" + std::to_string(node_id);
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

  // Don't release system properties here — topology_drop_snapshot() clears
  // wdevices_, which rocdxg_smi_* functions need for the process lifetime.
  // hsaKmtReleaseSystemProperties() is called in shutdown().

  if (gpu_index == 0) {
    g_wsl_syms.hsaKmtReleaseSystemProperties();
    g_wsl_syms.hsaKmtCloseKFD();
    return AMDSMI_STATUS_NOT_FOUND;
  }
  g_wsl_active = true;
  return AMDSMI_STATUS_SUCCESS;
}

bool WSLGPUBackend::is_active() { return g_wsl_active; }

amdsmi_status_t WSLGPUBackend::shutdown() {
  if (!g_wsl_active) return AMDSMI_STATUS_SUCCESS;
  g_wsl_active = false;
  g_wsl_syms.hsaKmtReleaseSystemProperties();
  HSAKMT_STATUS hret = g_wsl_syms.hsaKmtCloseKFD();
  if (hret != HSAKMT_STATUS_SUCCESS && hret != HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED)
    return hsakmt_to_amdsmi(hret);
  if (g_rocdxg_handle) {
    dlclose(g_rocdxg_handle);
    g_rocdxg_handle = nullptr;
    g_wsl_syms = WslSyms{};
  }
  return AMDSMI_STATUS_SUCCESS;
}

// -----------------------------------------------------------------------------
// IGPUBackend method implementations
// -----------------------------------------------------------------------------

amdsmi_status_t WSLGPUBackend::load_device_info() const {
  std::call_once(device_info_once_, [this]() {
    device_info_.struct_size = sizeof(device_info_);
    HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_device_info(node_id_, &device_info_);
    device_info_status_ = hsakmt_to_amdsmi(hstatus);
  });
  return device_info_status_;
}

amdsmi_status_t WSLGPUBackend::get_kfd_info(amdsmi_kfd_info_t* info) {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  // No KFD in WSL — the KFD ID and node ID concepts don't apply.
  std::memset(info, 0xFF, sizeof(*info));
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_asic_info(amdsmi_asic_info_t* info) {
  amdsmi_status_t r = load_device_info();
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL.
  if (r != AMDSMI_STATUS_SUCCESS && r != AMDSMI_STATUS_NOT_SUPPORTED) return r;
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  if (r == AMDSMI_STATUS_SUCCESS) {
    const auto& a = device_info_.asic;
    init_asic_info_defaults(info);
    copy_rocdxg_string(info->market_name, a.market_name);
    info->vendor_id = a.vendor_id;
    if (info->vendor_id == 0x1002)
      copy_string(info->vendor_name, "Advanced Micro Devices, Inc. [AMD/ATI]");
    info->subvendor_id = a.subvendor_id;
    info->device_id = a.device_id;
    info->rev_id = a.rev_id;
    std::snprintf(info->asic_serial, AMDSMI_MAX_STRING_LENGTH, "%016lx", a.asic_serial);
    // oam_id and physical_acc_id stay N/A from the defaults; gpu_id_ is only an
    // enumeration index, not real hardware data.
    info->num_of_compute_units = a.num_of_compute_units;
    // Convert IP version (major<<16|minor<<8|stepping) to the nibble-packed
    // hex format Python expects: hex(value)[2:] == "MMSS" (e.g. 0x1100 → "gfx1100").
    {
      uint32_t maj = (a.target_graphics_version >> 16) & 0xFF;
      uint32_t min = (a.target_graphics_version >> 8) & 0xFF;
      uint32_t stp = a.target_graphics_version & 0xFF;
      info->target_graphics_version = ((maj / 10) << 12) | ((maj % 10) << 8) | (min << 4) | stp;
    }
    info->subsystem_id = a.subsystem_id;
    return AMDSMI_STATUS_SUCCESS;
  }
  if (r != AMDSMI_STATUS_NOT_SUPPORTED) return r;
  init_asic_info_defaults(info);
  copy_string(info->market_name, marketing_name_);
  info->vendor_id = vendor_id_;
  if (vendor_id_ == 0x1002)
    copy_string(info->vendor_name, "Advanced Micro Devices, Inc. [AMD/ATI]");
  info->device_id = device_id_;
  // oam_id and physical_acc_id stay N/A from the defaults; gpu_id_ is only an
  // enumeration index, not real hardware data.
  info->num_of_compute_units =
      num_compute_units_ ? num_compute_units_ : std::numeric_limits<uint32_t>::max();
  info->flags = family_id_;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_board_info(amdsmi_board_info_t* info) {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  amdsmi_status_t r = load_device_info();
  if (r == AMDSMI_STATUS_SUCCESS) {
    std::memset(info, 0, sizeof(*info));
    copy_rocdxg_string(info->product_name, device_info_.board.product_name);
    copy_rocdxg_string(info->manufacturer_name, device_info_.board.manufacturer_name);
    return AMDSMI_STATUS_SUCCESS;
  }
  if (r != AMDSMI_STATUS_NOT_SUPPORTED) return r;
  std::memset(info, 0, sizeof(*info));
  copy_string(info->product_name, marketing_name_);
  copy_string(info->manufacturer_name, "Advanced Micro Devices, Inc. [AMD/ATI]");
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_vram_info(amdsmi_vram_info_t* info) {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  amdsmi_status_t r = load_device_info();
  if (r == AMDSMI_STATUS_SUCCESS) {
    std::memset(info, 0, sizeof(*info));
    info->vram_type = AMDSMI_VRAM_TYPE_UNKNOWN;
    copy_string(info->vram_vendor, "UNKNOWN");
    info->vram_size = device_info_.vram.vram_size_mb;
    info->vram_bit_width = device_info_.vram.vram_bit_width;
    info->vram_max_bandwidth = std::numeric_limits<decltype(info->vram_max_bandwidth)>::max();
    return AMDSMI_STATUS_SUCCESS;
  }
  if (r != AMDSMI_STATUS_NOT_SUPPORTED) return r;
  std::memset(info, 0, sizeof(*info));
  info->vram_type = AMDSMI_VRAM_TYPE_UNKNOWN;
  copy_string(info->vram_vendor, "UNKNOWN");
  info->vram_size = local_mem_size_ / (1024 * 1024);
  info->vram_bit_width = std::numeric_limits<decltype(info->vram_bit_width)>::max();
  info->vram_max_bandwidth = std::numeric_limits<decltype(info->vram_max_bandwidth)>::max();
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_memory_total(amdsmi_memory_type_t mem_type, uint64_t* total) {
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL.
  if (mem_type != AMDSMI_MEM_TYPE_VRAM && mem_type != AMDSMI_MEM_TYPE_VIS_VRAM)
    return AMDSMI_STATUS_NOT_SUPPORTED;
  if (total == nullptr) return AMDSMI_STATUS_INVAL;
  amdsmi_status_t r = load_device_info();
  if (r == AMDSMI_STATUS_SUCCESS) {
    *total = device_info_.vram.vram_size_mb * 1024 * 1024;
    return AMDSMI_STATUS_SUCCESS;
  }
  if (r != AMDSMI_STATUS_NOT_SUPPORTED) return r;
  *total = local_mem_size_;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_memory_usage(amdsmi_memory_type_t mem_type, uint64_t* used) {
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL.
  if (mem_type != AMDSMI_MEM_TYPE_VRAM && mem_type != AMDSMI_MEM_TYPE_VIS_VRAM)
    return AMDSMI_STATUS_NOT_SUPPORTED;
  if (used == nullptr) return AMDSMI_STATUS_INVAL;
  rocdxg_smi_vram_usage_t usage = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_vram_usage(node_id_, &usage);
  if (hstatus == HSAKMT_STATUS_SUCCESS) {
    *used = usage.vram_used_mb * 1024 * 1024;
    return AMDSMI_STATUS_SUCCESS;
  }
  return hsakmt_to_amdsmi(hstatus);
}

amdsmi_status_t WSLGPUBackend::get_temp_metric(amdsmi_temperature_type_t sensor_type,
                                               amdsmi_temperature_metric_t metric,
                                               int64_t* temperature) {
  // Native rsmi_dev_temp_metric_get checks nullptr unconditionally, before support
  // determination, for standard sensor types — match that contract here.
  if (temperature == nullptr) return AMDSMI_STATUS_INVAL;
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_temperature(
      node_id_, static_cast<uint32_t>(sensor_type), static_cast<uint32_t>(metric), temperature);
  return hsakmt_to_amdsmi(hstatus);
}

amdsmi_status_t WSLGPUBackend::get_volt_metric(amdsmi_voltage_type_t sensor_type,
                                               amdsmi_voltage_metric_t metric, int64_t* voltage) {
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL.
  if (metric != AMDSMI_VOLT_CURRENT) return AMDSMI_STATUS_NOT_SUPPORTED;
  if (sensor_type != AMDSMI_VOLT_TYPE_VDDGFX) return AMDSMI_STATUS_NOT_SUPPORTED;
  if (voltage == nullptr) return AMDSMI_STATUS_INVAL;
  rocdxg_smi_power_info_t power = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_power_info(node_id_, &power);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return hsakmt_to_amdsmi(hstatus);
  *voltage = power.gfx_voltage;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_power_info(amdsmi_power_info_t* info) {
  rocdxg_smi_power_info_t power = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_power_info(node_id_, &power);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return hsakmt_to_amdsmi(hstatus);
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL.
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
  info->current_socket_power = power.current_socket_power;
  info->average_socket_power = power.current_socket_power;
  info->socket_power = power.current_socket_power;
  info->gfx_voltage = power.gfx_voltage;
  info->soc_voltage = power.soc_voltage;
  info->mem_voltage = power.mem_voltage;
  info->power_limit = power.power_limit;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_busy_percent(uint32_t* gpu_busy_percent) {
  rocdxg_smi_gpu_metrics_info_t metrics = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_gpu_metrics_info(node_id_, &metrics);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return hsakmt_to_amdsmi(hstatus);
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL.
  if (gpu_busy_percent == nullptr) return AMDSMI_STATUS_INVAL;
  *gpu_busy_percent = metrics.average_gfx_activity;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_gpu_activity(amdsmi_engine_usage_t* info) {
  rocdxg_smi_gpu_metrics_info_t metrics = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_gpu_metrics_info(node_id_, &metrics);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return hsakmt_to_amdsmi(hstatus);
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL.
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  std::memset(info, 0, sizeof(*info));
  info->gfx_activity = metrics.average_gfx_activity;
  info->umc_activity = metrics.average_umc_activity;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_clock_info(amdsmi_clk_type_t clk_type, amdsmi_clk_info_t* info) {
  rocdxg_smi_clock_info_t rocdxg_info = {};
  HSAKMT_STATUS hstatus =
      g_wsl_syms.rocdxg_smi_get_clock_info(node_id_, static_cast<uint32_t>(clk_type), &rocdxg_info);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return hsakmt_to_amdsmi(hstatus);
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL.
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  std::memset(info, 0, sizeof(*info));
  info->clk = rocdxg_info.clk;
  info->min_clk = rocdxg_info.min_clk;
  info->max_clk = rocdxg_info.max_clk;
  info->clk_locked = rocdxg_info.clk_locked;
  info->clk_deep_sleep = rocdxg_info.clk_deep_sleep;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_pcie_info(amdsmi_pcie_info_t* info) {
  rocdxg_smi_pcie_info_t rocdxg_info = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_pcie_info(node_id_, &rocdxg_info);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return hsakmt_to_amdsmi(hstatus);
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL.
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
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
}

amdsmi_status_t WSLGPUBackend::get_driver_info(amdsmi_driver_info_t* info) {
  amdsmi_status_t r = load_device_info();
  if (r != AMDSMI_STATUS_SUCCESS) return r;
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL.
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  std::memset(info, 0, sizeof(*info));
  copy_rocdxg_string(info->driver_version, device_info_.driver.driver_version);
  copy_rocdxg_string(info->driver_date, device_info_.driver.driver_date);
  copy_rocdxg_string(info->driver_name, device_info_.driver.driver_name);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_vbios_info(amdsmi_vbios_info_t* info) {
  amdsmi_status_t r = load_device_info();
  if (r != AMDSMI_STATUS_SUCCESS) return r;
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL.
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  std::memset(info, 0, sizeof(*info));
  copy_rocdxg_string(info->name, device_info_.vbios.name);
  copy_rocdxg_string(info->build_date, device_info_.vbios.build_date);
  copy_rocdxg_string(info->part_number, device_info_.vbios.part_number);
  copy_rocdxg_string(info->version, device_info_.vbios.version);
  copy_rocdxg_string(info->boot_firmware, device_info_.vbios.boot_firmware);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_gpu_cache_info(amdsmi_gpu_cache_info_t* info) {
  amdsmi_status_t r = load_device_info();
  if (r != AMDSMI_STATUS_SUCCESS) return r;
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL.
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  std::memset(info, 0, sizeof(*info));
  const auto& c = device_info_.cache;
  static_assert(ROCDXG_SMI_MAX_CACHE_TYPES <= AMDSMI_MAX_CACHE_TYPES,
                "rocdxg cache array no longer fits amdsmi's cache array");
  info->num_cache_types =
      std::min(c.num_cache_types, static_cast<uint32_t>(ROCDXG_SMI_MAX_CACHE_TYPES));
  for (uint32_t i = 0; i < info->num_cache_types; ++i) {
    info->cache[i].cache_size = c.cache[i].cache_size_kb;
    info->cache[i].cache_level = c.cache[i].cache_level;
    info->cache[i].cache_properties = c.cache[i].cache_properties;
    info->cache[i].max_num_cu_shared = c.cache[i].max_num_cu_shared;
    info->cache[i].num_cache_instance = c.cache[i].num_cache_instance;
  }
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_fw_info(amdsmi_fw_info_t* info) {
  amdsmi_status_t r = load_device_info();
  if (r != AMDSMI_STATUS_SUCCESS) return r;
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL.
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  std::memset(info, 0, sizeof(*info));
  const auto& fw = device_info_.fw;
  const uint32_t src_max =
      std::min(fw.num_fw_info, static_cast<uint32_t>(ROCDXG_SMI_MAX_FW_ENTRIES));
  info->num_fw_info = std::min(src_max, static_cast<uint32_t>(AMDSMI_FW_ID__MAX));
  for (uint32_t i = 0; i < info->num_fw_info; ++i) {
    info->fw_info_list[i].fw_id = static_cast<amdsmi_fw_block_t>(fw.entries[i].fw_id);
    info->fw_info_list[i].fw_version = fw.entries[i].fw_version;
  }
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_fan_rpms(uint32_t /* sensor_ind */, int64_t* /* speed */) {
  return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t WSLGPUBackend::get_fan_speed(uint32_t /* sensor_ind */, int64_t* speed) {
  rocdxg_smi_gpu_metrics_info_t metrics = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_gpu_metrics_info(node_id_, &metrics);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return hsakmt_to_amdsmi(hstatus);
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL.
  if (speed == nullptr) return AMDSMI_STATUS_INVAL;
  *speed = static_cast<int64_t>(metrics.current_fan_speed_percent);
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
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL.
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  std::memset(info, 0, sizeof(*info));
  if (power.power_limit != std::numeric_limits<uint32_t>::max())
    info->power_cap = power.power_limit;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WSLGPUBackend::get_gpu_metrics_info(amdsmi_gpu_metrics_t* info) {
  rocdxg_smi_gpu_metrics_info_t metrics = {};
  HSAKMT_STATUS hstatus = g_wsl_syms.rocdxg_smi_get_gpu_metrics_info(node_id_, &metrics);
  if (hstatus != HSAKMT_STATUS_SUCCESS) return hsakmt_to_amdsmi(hstatus);
  // Feature support checked before nullptr so NOT_SUPPORTED takes priority over INVAL.
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  // Init all numeric fields to sentinel (0xFF = max for all uint types); keep the pointer null.
  std::memset(info, 0xFF, sizeof(*info));
  info->apu_metrics = nullptr;

  // rocdxg temperatures are in degrees C (uint32_t); amdsmi_gpu_metrics_t uses uint16_t degrees C.
  // Guard against the UINT32_MAX sentinel rocdxg returns for an absent sensor
  // (see rocdxg_smi_get_gpu_metrics_info) so it isn't silently truncated into
  // a bogus reading instead of leaving the memset 0xFF "N/A" sentinel above.
  if (metrics.temperature_edge <= 0xFFFEU)
    info->temperature_edge = static_cast<uint16_t>(metrics.temperature_edge);
  if (metrics.temperature_hotspot <= 0xFFFEU)
    info->temperature_hotspot = static_cast<uint16_t>(metrics.temperature_hotspot);
  if (metrics.average_gfx_activity <= 0xFFFEU)
    info->average_gfx_activity = static_cast<uint16_t>(metrics.average_gfx_activity);
  if (metrics.average_umc_activity <= 0xFFFEU)
    info->average_umc_activity = static_cast<uint16_t>(metrics.average_umc_activity);
  if (metrics.current_socket_power <= 0xFFFEU) {
    info->current_socket_power = static_cast<uint16_t>(metrics.current_socket_power);
    info->average_socket_power = static_cast<uint16_t>(metrics.current_socket_power);
  }
  // Populate current clocks from rocdxg. Guard against UINT32_MAX sentinel
  // (rocdxg returns UINT32_MAX when a field is unsupported on this version).
  if (metrics.current_gfxclk <= 0xFFFEU) {
    // All XCCs run at the same GFX clock on MI300X; propagate to all XCC slots.
    uint32_t n = std::min(num_xcc_, static_cast<uint32_t>(AMDSMI_MAX_NUM_GFX_CLKS));
    for (uint32_t i = 0; i < n; ++i)
      info->current_gfxclks[i] = static_cast<uint16_t>(metrics.current_gfxclk);
  }
  if (metrics.current_socclk <= 0xFFFEU)
    info->current_socclk = static_cast<uint16_t>(metrics.current_socclk);
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
  if (uuid_length == nullptr || uuid == nullptr || *uuid_length < AMDSMI_GPU_UUID_SIZE)
    return AMDSMI_STATUS_INVAL;

  const uint64_t id = unique_id_ ? unique_id_ : bdf_.as_uint;
  amdsmi_status_t status = amdsmi_uuid_gen(uuid, id, device_id_, 0xff);
  if (status == AMDSMI_STATUS_SUCCESS) *uuid_length = AMDSMI_GPU_UUID_SIZE;
  return status;
}

}  // namespace amd::smi

#endif  // ENABLE_WSL_BACKEND
