////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/amd_xdna_driver.h"

#include <array>
#include <cassert>
#include <cerrno>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <fcntl.h>
#include <libdrm/drm.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "inc/hsa_ext_amd_aie.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/runtime.h"
#include "core/inc/signal.h"
#include "core/util/memory.h"
#include "core/util/utils.h"
#include "uapi/amdxdna_accel.h"

namespace rocr {
namespace AMD {

namespace {
using DriverMemoryHandleWord = decltype(std::declval<core::DriverMemoryHandle>().handle);
}

static_assert((sizeof(DriverMemoryHandleWord) >= sizeof(uint32_t)) &&
                  (alignof(DriverMemoryHandleWord) >= alignof(uint32_t)),
              "DriverMemoryHandle cannot store a XDNA handle");

/// @brief Opcode types for commands.
///
/// This should match the opcode types defined in xdna-driver and XRT ERT (ert_cmd_opcode).
enum ert_cmd_opcode {
  /// @brief Invalid command.
  ERT_INVALID_CMD = ~0U,
  /// @brief Start a workgroup on a CU.
  ERT_START_CU = 0,
  /// @brief Command chain.
  ERT_CMD_CHAIN = 19,
  /// @brief Instruction buffer command format on NPU format.
  ERT_START_NPU = 20,
  /// @brief Instruction buffer command with preemption format on NPU.
  ERT_START_NPU_PREEMPT = 21,
  /// @brief Instruction buffer command with preemption format on NPU using ELF.
  ERT_START_NPU_PREEMPT_ELF = 22,
};

/// @brief Command state.
///
/// This should match the command states defined in xdna-driver and XRT ERT (ert_cmd_state).
enum ert_cmd_state {
  /// @brief Invalid state.
  ERT_CMD_STATE_INVALID,
  /// @brief Set by host before submitting a command to scheduler.
  ERT_CMD_STATE_NEW,
  /// @brief Internal scheduler state.
  ERT_CMD_STATE_QUEUED,
  /// @brief Internal scheduler state.
  ERT_CMD_STATE_RUNNING,
  /// @brief Set by scheduler when command completes.
  ERT_CMD_STATE_COMPLETED,
  /// @brief Set by scheduler if command failed.
  ERT_CMD_STATE_ERROR,
  /// @brief Set by scheduler if command abort.
  ERT_CMD_STATE_ABORT,
  /// @brief Internal scheduler state.
  ERT_CMD_STATE_SUBMITTED,
  /// @brief Set by scheduler if command timeout and reset.
  ERT_CMD_STATE_TIMEOUT,
  /// @brief Set by scheduler if command timeout and fail to reset.
  ERT_CMD_STATE_NORESPONSE,
};

/// @brief Start kernel command packet.
///
/// This should match the command format defined in xdna-driver (amdxdna_cmd) and XRT ERT
/// (ert_start_kernel_cmd).
struct ert_start_kernel_cmd {
  union {
    struct {
      /// @brief Current state of a command. Should be one of the values in @ref ert_cmd_state.
      uint32_t state : 4;
      uint32_t unused : 6;
      /// @brief Extra CU masks in addition to mandatory mask. The number of extra CU masks is
      /// determined by the value of this field, and the actual masks are included in the payload
      /// after the mandatory cu_mask.
      uint32_t extra_cu_masks : 2;
      /// @brief Number of words following header for cmd data. Not include stat data. The actual
      /// number of CU masks in the payload is (1 + extra_cu_masks) based on the header fields, and
      /// the rest of the payload is data.
      uint32_t count : 11;
      /// @brief Opcode for the command. Should be one of the values in @ref ert_cmd_opcode.
      uint32_t opcode : 5;
      /// @brief Reserved. Must be 0.
      uint32_t reserved : 4;
    };
    uint32_t header;
  };
  /// @brief 1 mandatory CU mask, up to 4 optional CU masks, determined by @ref extra_cu_masks. Rest
  /// of data.
  uint32_t data[];
};

/// @brief Command chain packet.
///
/// This should match the command format defined in xdna-driver (amdxdna_cmd_chain) and XRT ERT
/// (ert_cmd_chain).
struct ert_cmd_chain_data {
  /// @brief Number of commands in the chain.
  uint32_t command_count;
  /// @brief Index of last successfully submitted command in chain.
  uint32_t submit_index;
  /// @brief Index of failing command if cmd status is not completed.
  uint32_t error_index;
  /// @brief Reserved. Must be 0.
  uint32_t reserved[3];
  /// @brief BO handles of each command in the chain. The number of BO handles is determined by @ref
  /// command_count.
  uint64_t data[];
};

/// @brief XDNA device type.
enum class XDNADeviceType {
  Phx,
  Stx,     // Strix Halo / Krackan
  Unknown  // Unknown device
};

/// @brief XDNA device ID.
struct XDNADeviceId {
  uint16_t device;

  bool operator<(const XDNADeviceId& other) const { return device < other.device; }
};

/// @brief Supported XDNA devices.
static const std::map<XDNADeviceId, XDNADeviceType> supported_xdna_devices = {
    {{0x1502}, XDNADeviceType::Phx},  // Phoenix
    {{0x17f0}, XDNADeviceType::Stx},  // Strix Halo / Krackan
};

/// @brief Devnode path for XDNA devices.
static constexpr std::string_view devnodes_path = "/dev/accel";
/// @brief Sysfs path for XDNA devices.
static constexpr std::string_view sysfs_path = "/sys/class/accel";
/// @brief Devnode prefix for XDNA devices.
static constexpr std::string_view devnode_prefix = "accel";
/// @brief Maximum devnode minor number for XDNA devices.
constexpr uint32_t devnode_max_minor_num = 64;

/// @brief Used to transform an address into a device address
constexpr uint32_t DEV_ADDR_BASE = 0x04000000;
constexpr uint32_t DEV_ADDR_OFFSET_MASK = 0x02FFFFFF;

/// @brief The driver places a structure before each command in a command chain.
/// Need to increase the size of the command by the size of this structure.
/// In the following xdna driver source can see where this is implemented:
/// https://github.com/amd/xdna-driver/blob/eddd92c0f61592c576a500f16efa24eb23667c23/src/driver/amdxdna/aie2_msg_priv.h#L387
/// https://github.com/amd/xdna-driver/blob/eddd92c0f61592c576a500f16efa24eb23667c23/src/driver/amdxdna/aie2_message.c#L637
constexpr uint32_t CMD_COUNT_SIZE_INCREASE = 3;

/// @brief Default amdxdna_cu_config::cu_func when configuring a CU.
constexpr uint32_t default_cu_func = 0;

/// @brief Calls ioctl with the given request and argument, and retries if the call is interrupted
/// by a signal or if it returns EAGAIN.
///
/// @param[in] fd file descriptor
/// @param[in] request ioctl request code
/// @param[in] arg pointer to the argument for the ioctl call
static hsa_status_t xdna_ioctl(int fd, unsigned long request, void* arg) {
  int ret;
  do {
    ret = ioctl(fd, request, arg);
    if (ret >= 0) {
      return HSA_STATUS_SUCCESS;
    }
  } while (errno == EINTR || errno == EAGAIN);

  // Map errno to appropriate HSA status code.
  switch (errno) {
    case EINVAL:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    case ENOENT:
      return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    case ENOMEM:
    case ENOSPC:
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    default:
      return HSA_STATUS_ERROR;
  }
}

/// @brief Per hardware context PDI cache.
class PDICache {
 private:
  /// @brief CU mask size.
  constexpr static size_t cu_mask_size = sizeof(uint32_t) * CHAR_BIT;

 public:
  using size_type = uint32_t;

 private:
  std::array<uint32_t, cu_mask_size> entries = {};
  size_type entry_count = 0;

 public:
  /// @brief Sentinel value for entries not found.
  constexpr static size_type NotFound = cu_mask_size;

  /// @brief Returns if the cache is empty.
  constexpr bool empty() const { return entry_count == 0; }

  /// @brief Returns the size of the cache.
  constexpr size_type size() const { return entry_count; }

  /// @brief Returns the index of the BO handle if it is the cache, otherwise @ref NotFound.
  ///
  /// This function does a linear search because the mask is small (32 elements).
  size_type GetIndex(uint32_t pdi_handle) const {
    for (size_type i = 0; i < entry_count; ++i) {
      if (entries[i] == pdi_handle) {
        return i;
      }
    }
    return NotFound;
  }

  /// @brief Sets the next cache entry.
  hsa_status_t SetNext(uint32_t pdi_bo_handle, size_type& index) {
    if (entry_count == entries.size()) {
      // cache is full
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    index = entry_count++;
    entries[index] = pdi_bo_handle;
    return HSA_STATUS_SUCCESS;
  }

  constexpr uint32_t operator[](size_type index) const { return entries[index]; }
};

/// @brief Metadata for a Kernel Mode Queue (KMQ).
struct KmqMetadata {
  uint32_t hw_ctx_handle = AMDXDNA_INVALID_CTX_HANDLE;
  uint32_t syncobj_handle = 0;
  PDICache pdi_cache;
};

/**
 * @brief Flushes the CPU cache for the packet's arguments.
 *
 * The sizes of the arguments are after the pointers of the arguments.
 *
 * @param pkt pointer to the packet
 */
static void FlushArguments(const hsa_amd_aie_kernel_dispatch_packet_t* pkt) {
  auto* kernarg_address = static_cast<uint64_t*>(pkt->kernarg_address);
  for (uint32_t kernarg_idx = 0; kernarg_idx < pkt->num_kernargs; ++kernarg_idx) {
    void* ptr = reinterpret_cast<void*>(kernarg_address[kernarg_idx]);
    size_t size = kernarg_address[kernarg_idx + pkt->num_kernargs];
    FlushCpuCache(ptr, 0, size);
  }
}

/**
 * @brief Destroys the amdxdna_hwctx with the given handle.
 *
 * @param[in] fd driver file descriptor
 * @param[in] hw_ctx_handle handle of the hardware context to destroy
 */
static hsa_status_t DestroyHwCtx(int fd, uint32_t hw_ctx_handle) {
  assert(hw_ctx_handle != AMDXDNA_INVALID_CTX_HANDLE);

  amdxdna_drm_destroy_hwctx args = {};
  args.handle = hw_ctx_handle;
  return xdna_ioctl(fd, DRM_IOCTL_AMDXDNA_DESTROY_HWCTX, &args);
}

/// @brief Creates and configures a hardware context for the KMQ, and updates the KMQ metadata.
///
/// @param[in] fd driver file descriptor
/// @param[in] num_core_tiles number of core tiles to configure the hardware context with
/// @param[in,out] kmq_metadata KMQ metadata to update with the hardware context handle and syncobj
/// handle
static hsa_status_t CreateHwCtx(int fd, uint32_t num_core_tiles, KmqMetadata* kmq_metadata) {
  // Create QoS information; we don't leverage any external Qos hints.
  amdxdna_qos_info qos_info = {};
  qos_info.user_start_col = USER_START_COL_NOT_REQUESTED;

  // Create the new hardware context.
  amdxdna_drm_create_hwctx create_hwctx_args = {};
  create_hwctx_args.qos_p = reinterpret_cast<uintptr_t>(&qos_info);
  create_hwctx_args.max_opc = 0x800;
  create_hwctx_args.num_tiles = num_core_tiles;
  hsa_status_t err = xdna_ioctl(fd, DRM_IOCTL_AMDXDNA_CREATE_HWCTX, &create_hwctx_args);
  if (err != HSA_STATUS_SUCCESS) {
    assert(false && "Failed to create hardware context for KMQ");
    return err;
  }

  // Create hardware context configuration.
  const size_t num_cus = kmq_metadata->pdi_cache.empty() ? 1 : kmq_metadata->pdi_cache.size();
  const size_t config_cu_param_size =
      sizeof(amdxdna_hwctx_param_config_cu) + num_cus * sizeof(amdxdna_cu_config);

  auto* xdna_config_cu_param =
      static_cast<amdxdna_hwctx_param_config_cu*>(malloc(config_cu_param_size));
  if (xdna_config_cu_param == nullptr) {
    DestroyHwCtx(fd, create_hwctx_args.handle);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  MAKE_SCOPE_GUARD([xdna_config_cu_param] { free(xdna_config_cu_param); });
  memset(xdna_config_cu_param, 0, config_cu_param_size);

  if (!kmq_metadata->pdi_cache.empty()) {
    xdna_config_cu_param->num_cus = kmq_metadata->pdi_cache.size();
    for (size_t i = 0; i < kmq_metadata->pdi_cache.size(); i++) {
      xdna_config_cu_param->cu_configs[i].cu_bo = kmq_metadata->pdi_cache[i];
      xdna_config_cu_param->cu_configs[i].cu_func = default_cu_func;
    }
  } else {
    // If the PDI cache is empty, it means we have not allocated any CU configuration BOs yet. Still
    // need to configure the hardware context with at least 1 CU, so we set the cu_bo of the first
    // CU config to 0, which is an invalid BO handle but indicates to the driver that we want to use
    // the default CU configuration.
    xdna_config_cu_param->num_cus = 1;
    xdna_config_cu_param->cu_configs[0].cu_bo = 0;
    xdna_config_cu_param->cu_configs[0].cu_func = default_cu_func;
  }

  // Configure the new hardware context.
  amdxdna_drm_config_hwctx config_hw_ctx_args = {};
  config_hw_ctx_args.handle = create_hwctx_args.handle;
  config_hw_ctx_args.param_type = DRM_AMDXDNA_HWCTX_CONFIG_CU;
  config_hw_ctx_args.param_val = reinterpret_cast<uint64_t>(xdna_config_cu_param);
  config_hw_ctx_args.param_val_size = static_cast<uint32_t>(config_cu_param_size);
  err = xdna_ioctl(fd, DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, &config_hw_ctx_args);
  if (err != HSA_STATUS_SUCCESS) {
    DestroyHwCtx(fd, create_hwctx_args.handle);
    assert(false && "Failed to configure hardware context for KMQ");
    return err;
  }

  kmq_metadata->hw_ctx_handle = create_hwctx_args.handle;
  kmq_metadata->syncobj_handle = create_hwctx_args.syncobj_handle;

  return HSA_STATUS_SUCCESS;
}

/**
 * @brief Submits a command for execution.
 *
 * @param[in] fd driver file descriptor
 * @param[in] cmd_bo_handle BO handle of the command to execute
 * @param[in] bo_handles handles associated with the command
 * @param[in] hw_ctx_handle hardware context handle
 * @param[out] seq_out sequence number of the command
 */
static hsa_status_t SubmitCommand(int fd, uint32_t cmd_bo_handle,
                                  const std::vector<uint32_t>& bo_handles, uint32_t hw_ctx_handle,
                                  uint64_t& seq_out) {
  assert(hw_ctx_handle != AMDXDNA_INVALID_CTX_HANDLE);

  amdxdna_drm_exec_cmd exec_cmd = {};
  exec_cmd.hwctx = hw_ctx_handle;
  exec_cmd.type = AMDXDNA_CMD_SUBMIT_EXEC_BUF;
  exec_cmd.cmd_handles = cmd_bo_handle;
  exec_cmd.args = reinterpret_cast<uint64_t>(bo_handles.data());
  exec_cmd.cmd_count = 1;
  exec_cmd.arg_count = static_cast<uint32_t>(bo_handles.size());
  hsa_status_t err = xdna_ioctl(fd, DRM_IOCTL_AMDXDNA_EXEC_CMD, &exec_cmd);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  seq_out = exec_cmd.seq;
  return HSA_STATUS_SUCCESS;
}

/**
 * @brief Waits for a command to finish.
 *
 * @param[in] fd driver file descriptor
 * @param[in] cmd command to wait for
 * @param[in] hw_ctx_handle hardware context handle
 * @param[in] syncobj_handle DRM syncobj handle for timeline wait
 * @param[in] seq sequence number of the command
 */
static hsa_status_t WaitCommand(int fd, ert_start_kernel_cmd* cmd, uint32_t hw_ctx_handle,
                                uint32_t syncobj_handle, uint64_t seq) {
  assert(hw_ctx_handle != AMDXDNA_INVALID_CTX_HANDLE);

  // Check command status before waiting to avoid unnecessary ioctl if the command has already
  // completed.
  auto& cmd_ref = *static_cast<volatile ert_start_kernel_cmd*>(cmd);
  switch (cmd_ref.state) {
    case ERT_CMD_STATE_NEW:
    case ERT_CMD_STATE_QUEUED:
    case ERT_CMD_STATE_RUNNING:
      // Command is still in progress, need to wait.
      break;
    case ERT_CMD_STATE_COMPLETED:
      // Command has completed, no need to wait.
      return HSA_STATUS_SUCCESS;
    default:
      // Command is in an error state.
      return HSA_STATUS_ERROR;
  }

  // Prefer DRM syncobj timeline wait when available.
  if (syncobj_handle != 0) {
    drm_syncobj_timeline_wait timeline_wait = {};
    timeline_wait.handles = reinterpret_cast<uintptr_t>(&syncobj_handle);
    timeline_wait.points = reinterpret_cast<uintptr_t>(&seq);
    timeline_wait.count_handles = 1;
    timeline_wait.timeout_nsec = INT64_MAX;
    timeline_wait.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT;
    hsa_status_t err = xdna_ioctl(fd, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &timeline_wait);
    if (err != HSA_STATUS_SUCCESS) {
      return err;
    }
  } else {
    // Fallback: XDNA-specific wait.
    amdxdna_drm_wait_cmd wait_cmd = {};
    wait_cmd.hwctx = hw_ctx_handle;
    wait_cmd.timeout = 0;  // no timeout, wait until the command finishes
    wait_cmd.seq = seq;
    hsa_status_t err = xdna_ioctl(fd, DRM_IOCTL_AMDXDNA_WAIT_CMD, &wait_cmd);
    if (err != HSA_STATUS_SUCCESS) {
      return err;
    }
  }

  // Check if command failed.
  if (cmd_ref.state != ERT_CMD_STATE_COMPLETED) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}


XdnaDriver::XdnaDriver(std::string devnode_name)
    : core::Driver(core::DriverType::XDNA, std::move(devnode_name)) {}

hsa_status_t XdnaDriver::DiscoverDriver(std::unique_ptr<core::Driver>& driver) {
  for (uint32_t i = 0; i < devnode_max_minor_num; ++i) {
    auto tmp_driver = std::make_unique<XdnaDriver>(std::string(devnode_prefix) + std::to_string(i));
    if (tmp_driver->Open() == HSA_STATUS_SUCCESS) {
      if (tmp_driver->QueryKernelModeDriver(core::DriverQuery::GET_DRIVER_VERSION) ==
          HSA_STATUS_SUCCESS) {
        // XDNADriver supports only one XDNA device. Once found, the driver is initialized.
        driver = std::move(tmp_driver);
        return HSA_STATUS_SUCCESS;
      } else {
        tmp_driver->Close();
      }
    }
  }

  return HSA_STATUS_ERROR;
}

uint64_t XdnaDriver::GetDevHeapByteSize() {
  return dev_heap_size;
}

hsa_status_t XdnaDriver::Init() { return InitDeviceHeap(); }

hsa_status_t XdnaDriver::ShutDown() { return FreeDeviceHeap(); }

hsa_status_t XdnaDriver::QueryKernelModeDriver(core::DriverQuery query) {
  switch (query) {
  case core::DriverQuery::GET_DRIVER_VERSION:
    return QueryDriverVersion();
  default:
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t XdnaDriver::Open() {
  const std::string devnode_path = std::string(devnodes_path) + "/" + devnode_name_;
  fd_ = open(devnode_path.c_str(), O_RDWR | O_CLOEXEC);
  if (fd_ < 0) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::Close() {
  int ret(0);
  if (fd_ > 0) {
    ret = close(fd_);
    fd_ = -1;
  }
  if (ret) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::GetSystemProperties(HsaSystemProperties& sys_props) const {
  sys_props.NumNodes = 1;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::GetNodeProperties(HsaNodeProperties& node_props, uint32_t node_id) const {
  amdxdna_drm_query_aie_metadata aie_metadata = {};
  amdxdna_drm_get_info get_info_args = {};
  get_info_args.param = DRM_AMDXDNA_QUERY_AIE_METADATA;
  get_info_args.buffer_size = sizeof(aie_metadata);
  get_info_args.buffer = reinterpret_cast<uintptr_t>(&aie_metadata);

  hsa_status_t err = xdna_ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info_args);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  const std::string sysfs_device_path = std::string(sysfs_path) + "/" + devnode_name_ + "/device";

  // Find device type.
  XDNADeviceType device_type = XDNADeviceType::Unknown;
  {
    const std::string device_id_file = sysfs_device_path + "/device";
    std::ifstream is(device_id_file);
    if (!is.good()) {
      assert(false && "Device file not found in sysfs.");
      return HSA_STATUS_ERROR;
    }

    XDNADeviceId device_id = {};
    // Device ID is in hex.
    if (!(is >> std::hex >> device_id.device)) {
      assert(false && "Failed to read device ID from sysfs.");
      return HSA_STATUS_ERROR;
    }

    const auto device_type_it = supported_xdna_devices.find(device_id);
    if (device_type_it == supported_xdna_devices.end()) {
      assert(false && "Unsupported XDNA device.");
      return HSA_STATUS_ERROR;
    }
    device_type = device_type_it->second;
  }

  // Fill in node properties that depend on device type.
  std::fill_n(node_props.AMDName, HSA_PUBLIC_NAME_SIZE, 0);
  switch (device_type) {
    case XDNADeviceType::Phx: {
      constexpr std::string_view name("aie2");
      assert(name.size() < HSA_PUBLIC_NAME_SIZE);
      std::copy(name.begin(), name.end(), node_props.AMDName);
      // Only target N-1 columns as that is the number of shim DMAs in NPU1 devices.
      node_props.NumNeuralCores = (aie_metadata.cols - 1) * aie_metadata.core.row_count;
    } break;

    case XDNADeviceType::Stx: {
      constexpr std::string_view name("aie2p");
      assert(name.size() < HSA_PUBLIC_NAME_SIZE);
      std::copy(name.begin(), name.end(), node_props.AMDName);
      node_props.NumNeuralCores = aie_metadata.cols * aie_metadata.core.row_count;
    } break;

    default:
      assert(false && "Unsupported XDNA device.");
      return HSA_STATUS_ERROR;
  }

  // Read device name from sysfs.
  {
    const std::string device_name_file = sysfs_device_path + "/vbnv";
    std::ifstream is(device_name_file);
    if (!is.good()) {
      assert(false && "Device file name not found in sysfs.");
      return HSA_STATUS_ERROR;
    }
    std::array<char, HSA_PUBLIC_NAME_SIZE> device_name = {};
    if (!is.getline(device_name.data(), device_name.size() - 1)) {
      assert(false && "Failed to read device name from sysfs.");
      return HSA_STATUS_ERROR;
    }
    // Convert device name from ASCII to UTF-16 for MarketingName.
    std::copy(device_name.begin(), device_name.end(), node_props.MarketingName);
  }

  /// @todo XDNA driver currently only supports single-node AIE
  /// devices over PCIe. Update this once we can get topology
  /// information dynamically from the sysfs.
  node_props.NumIOLinks = 0;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::GetEdgeProperties(std::vector<HsaIoLinkProperties>& io_link_props,
                                           uint32_t node_id) const {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::GetMemoryProperties(uint32_t node_id,
                                             std::vector<HsaMemoryProperties>& mem_props) const {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::GetCacheProperties(uint32_t node_id, uint32_t processor_id,
                                            std::vector<HsaCacheProperties>& cache_props) const {
  // AIE currently has no caches.
  return HSA_STATUS_ERROR_INVALID_CACHE;
}

hsa_status_t
XdnaDriver::AllocateMemory(const core::MemoryRegion &mem_region,
                           core::MemoryRegion::AllocateFlags alloc_flags,
                           void **mem, size_t size, uint32_t node_id) {
  const MemoryRegion& m_region = static_cast<const MemoryRegion&>(mem_region);

  if (!m_region.IsSystem()) {
    return HSA_STATUS_ERROR_INVALID_REGION;
  }

  amdxdna_drm_create_bo create_bo_args = {};
  create_bo_args.size = size;
  const bool use_bo_share = !m_region.IsDeviceSVM();
  if (use_bo_share) {
    create_bo_args.type = AMDXDNA_BO_SHARE;
  } else {
    // While this is already checked in MemoryRegion::AllocateImpl, the max size is
    // MemoryRegion::max_sysmem_alloc_size_ for HSA_HEAPTYPE_DEVICE_SVM which is incorrect
    // for dev heap.
    if (size > dev_heap_size) {
      return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    }

    create_bo_args.type = AMDXDNA_BO_DEV;
  }

  hsa_status_t err = xdna_ioctl(fd_, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_bo_args);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  BOHandle bo_handle;
  bo_handle.handle = create_bo_args.handle;
  bo_handle.size = size;

  // Close the BO in case of error.
  MAKE_NAMED_SCOPE_GUARD(bo_guard, [&] { DestroyBOHandle(bo_handle); });

  amdxdna_drm_get_bo_info get_bo_info_args = {};
  get_bo_info_args.handle = create_bo_args.handle;
  err = xdna_ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info_args);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  if (use_bo_share) {
    if (alloc_flags & core::MemoryRegion::AllocateMemoryOnly) {
      /// TODO: We create an anonymous mapping to get a unique virtual address since the memory
      /// handle mapping, i.e., Runtime::memory_handle_map_, is indexed using DriverHandle which is
      /// driver-agnostic and just a pointer to the virtual address space. We waste a page, but it
      /// ensures uniqueness across drivers.
      bo_handle.vaddr =
          mmap(nullptr, MemoryRegion::GetPageSize(), PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (bo_handle.vaddr == MAP_FAILED) {
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
      }
    } else {
      bo_handle.vaddr =
          mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, get_bo_info_args.map_offset);
      if (bo_handle.vaddr == MAP_FAILED) {
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
      }
    }
    bo_handle.unmap_vaddr = true;
  } else {
    /// This is dev heap and is already mapped. See InitDeviceHeap().
    bo_handle.vaddr = reinterpret_cast<void*>(get_bo_info_args.vaddr);
    bo_handle.unmap_vaddr = false;
  }

  // We keep a mapping from VA memory addresses to BO handles because some operations, e.g.,
  // FreeMemory, pass only the memory address and not a BO handle.
  vmem_addr_mappings.emplace(bo_handle.vaddr, bo_handle);

  bo_guard.Dismiss();

  *mem = bo_handle.vaddr;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::FreeMemory(void* mem, size_t size) {
  auto it = vmem_addr_mappings.find(mem);
  if (it == vmem_addr_mappings.end()) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }

  auto& bo_handle = it->second;
  hsa_status_t err = DestroyBOHandle(bo_handle);
  vmem_addr_mappings.erase(it);
  return err;
}

hsa_status_t XdnaDriver::CreateQueue(uint32_t node_id, HSA_QUEUE_TYPE type, uint32_t queue_pct,
                                     HSA::hsa_amd_queue_priority_internal_t priority, uint32_t sdma_engine_id,
                                     void* queue_addr, uint64_t queue_size_bytes, uint64_t queue_metadata_size_bytes,
                                     HsaEvent* event, HsaQueueResource& queue_resource) const {
  // Driver doesn't support user-mode queues.
  return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
}

hsa_status_t XdnaDriver::UpdateQueue(HSA_QUEUEID queue_id, uint32_t queue_pct,
                                     HSA::hsa_amd_queue_priority_internal_t priority,
                                     void* queue_addr, uint64_t queue_size, HsaEvent* event) const {
  // Driver doesn't support queue updates.
  return HSA_STATUS_ERROR_INVALID_QUEUE;
}

hsa_status_t XdnaDriver::DestroyQueue(HSA_QUEUEID queue_id) const {
  // Driver doesn't support user-mode queues.
  return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
}

hsa_status_t XdnaDriver::CreateKernelModeQueue(size_t queue_size, void** queue_metadata) const {
  auto kmq_metadata = std::make_unique<KmqMetadata>();
  const uint32_t num_core_tiles = 1;
  hsa_status_t err = CreateHwCtx(fd_, num_core_tiles, kmq_metadata.get());
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }
  *queue_metadata = kmq_metadata.release();
  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::DestroyKernelModeQueue(void* queue_metadata) const {
  if (queue_metadata == nullptr ||
      (static_cast<KmqMetadata*>(queue_metadata)->hw_ctx_handle == AMDXDNA_INVALID_CTX_HANDLE)) {
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  // Create a unique_ptr to ensure cleanup.
  std::unique_ptr<KmqMetadata> kmq_metadata;
  kmq_metadata.reset(static_cast<KmqMetadata*>(queue_metadata));

  // Destroy hardware context associated with the queue.
  hsa_status_t err = DestroyHwCtx(fd_, kmq_metadata->hw_ctx_handle);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }
  kmq_metadata->hw_ctx_handle = AMDXDNA_INVALID_CTX_HANDLE;
  kmq_metadata->syncobj_handle = 0;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::SetQueueCUMask(HSA_QUEUEID queue_id, uint32_t cu_mask_count,
                                        uint32_t* queue_cu_mask) const {
  // AIE doesn't support queue CU masks.
  return HSA_STATUS_ERROR_INVALID_QUEUE;
}

hsa_status_t XdnaDriver::AllocQueueGWS(HSA_QUEUEID queue_id, uint32_t num_gws,
                                       uint32_t* first_gws) const {
  // AIE doesn't support GWS.
  return HSA_STATUS_ERROR_INVALID_QUEUE;
}

hsa_status_t XdnaDriver::ExportMemoryHandle(const core::Agent& agent, const core::DriverMemoryHandle& handle,
                                            core::ShareType type, uint32_t flags, void* export_handle,
                                            uint64_t* export_offset) {
  (void)agent;
  (void)flags;
  (void)export_offset;
  if (export_handle == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  switch (type) {
  case core::ShareType::DMABUF_FD: {
    // handle.handle is the kernel BO handle populated by CreateShareableHandle.
    if (!handle.IsValid()) {
      return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    }

    drm_prime_handle export_params = {};
    export_params.handle = handle.handle;
    export_params.flags = DRM_RDWR;
    export_params.fd = -1;
    hsa_status_t err = xdna_ioctl(fd_, DRM_IOCTL_PRIME_HANDLE_TO_FD, &export_params);
    if (err != HSA_STATUS_SUCCESS) {
      return err;
    }

    *static_cast<int*>(export_handle) = export_params.fd;
    return HSA_STATUS_SUCCESS;
  }
  case core::ShareType::FABRIC_HANDLE:
    return HSA_STATUS_ERROR;
  default:
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

hsa_status_t XdnaDriver::ImportMemoryHandle(const core::Agent& agent, core::DriverMemoryHandle* handle,
                                            core::ShareType type, void* import_handle,
                                            void* mem) {
  (void)agent;
  (void)mem;
  if (handle == nullptr || import_handle == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  switch (type) {
  case core::ShareType::DMABUF_FD: {
    const int dmabuf_fd = *static_cast<int*>(import_handle);

    drm_prime_handle import_params = {};
    import_params.handle = AMDXDNA_INVALID_BO_HANDLE;
    import_params.fd = dmabuf_fd;
    hsa_status_t err = xdna_ioctl(fd_, DRM_IOCTL_PRIME_FD_TO_HANDLE, &import_params);
    if (err != HSA_STATUS_SUCCESS) {
      return err;
    }

    *handle = core::DriverMemoryHandle{import_params.handle};
    handle->size = lseek(dmabuf_fd, 0, SEEK_END);
    return HSA_STATUS_SUCCESS;
  }
  case core::ShareType::FABRIC_HANDLE:
    return HSA_STATUS_ERROR;
  default:
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

hsa_status_t XdnaDriver::DestroyImportedMemoryHandle(core::DriverMemoryHandle* handle) {
  // Nothing to do for XDNA since we have a single, non-ref counted handle.
  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::Map(const core::DriverMemoryHandle& handle, void *mem,
                             size_t offset, size_t size,
                             hsa_access_permission_t perms, uint32_t node_id) {
  (void)node_id;
  // Get fd associated with the handle.
  drm_prime_handle params = {};
  params.handle = handle.handle;
  params.fd = -1;
  hsa_status_t err = xdna_ioctl(fd_, DRM_IOCTL_PRIME_HANDLE_TO_FD, &params);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  // Change permissions.
  void *mapped_ptr = mmap(mem, size, PermissionsToMmapFlags(perms),
                          MAP_FIXED | MAP_SHARED, params.fd, offset);
  close(params.fd);
  if (mapped_ptr == MAP_FAILED) {
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::Unmap(const core::DriverMemoryHandle& handle, void *mem,
                               size_t offset, size_t size, uint32_t node_id) {
  (void)node_id;
  if (munmap(mem, size) != 0) {
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::CreateShareableHandle(void* va, void* mem, size_t size,
                                               const core::Agent& agent,
                                               core::DriverMemoryHandle* handle, uint64_t* offset) {
  (void)va;
  (void)agent;

  // Find BO handle; mem is the BO handle; see AllocateMemory.
  auto bo_handle = FindBOHandle(mem);
  if (!bo_handle.IsValid()) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }

  // Get offset.
  amdxdna_drm_get_bo_info get_bo_info_args = {};
  get_bo_info_args.handle = bo_handle.handle;
  hsa_status_t err = xdna_ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info_args);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  // Get fd associated with the handle.
  drm_prime_handle params = {};
  params.handle = bo_handle.handle;
  params.flags = DRM_RDWR;
  params.fd = -1;
  err = xdna_ioctl(fd_, DRM_IOCTL_PRIME_HANDLE_TO_FD, &params);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  handle->handle = bo_handle.handle;
  handle->dmabuf_fd = params.fd;
  handle->mmap_offset = get_bo_info_args.map_offset;
  handle->size = size;
  *offset = 0;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::DestroyMemoryHandle(core::DriverMemoryHandle* handle) {
  // Close the dmabuf_fd.
  if (handle->dmabuf_fd >= 0) {
    close(handle->dmabuf_fd);
  }

  // Close the BO handle.
  drm_gem_close close_params = {};
  close_params.handle = handle->handle;
  hsa_status_t err = xdna_ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_params);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  *handle = {};

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::QueryDriverVersion() {
  amdxdna_drm_query_aie_version aie_version = {};
  amdxdna_drm_get_info args{DRM_AMDXDNA_QUERY_AIE_VERSION, sizeof(aie_version),
                            reinterpret_cast<uintptr_t>(&aie_version)};

  hsa_status_t err = xdna_ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_INFO, &args);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  version_.KernelInterfaceMajorVersion = aie_version.major;
  version_.KernelInterfaceMinorVersion = aie_version.minor;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::InitDeviceHeap() {
  amdxdna_drm_create_bo create_bo_args = {};
  create_bo_args.size = dev_heap_size;
  create_bo_args.type = AMDXDNA_BO_DEV_HEAP;
  hsa_status_t err = xdna_ioctl(fd_, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_bo_args);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  dev_heap_handle.handle = create_bo_args.handle;

  // Unmap memory and close the BO in case of error.
  MAKE_NAMED_SCOPE_GUARD(dev_heap_handle_guard, [&] { DestroyBOHandle(dev_heap_handle); });

  amdxdna_drm_get_bo_info get_bo_info_args = {};
  get_bo_info_args.handle = dev_heap_handle.handle;
  err = xdna_ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info_args);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  const size_t size = dev_heap_align * 2 - 1;
  dev_heap_handle.vaddr =
      mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (dev_heap_handle.vaddr == MAP_FAILED) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  dev_heap_handle.unmap_vaddr = true;
  dev_heap_handle.size = size;

  void* addr_aligned = reinterpret_cast<void*>(
      AlignUp(reinterpret_cast<uintptr_t>(dev_heap_handle.vaddr), dev_heap_align));

  dev_heap_aligned =
      mmap(addr_aligned, dev_heap_size, PROT_READ | PROT_WRITE,
           MAP_SHARED | MAP_FIXED, fd_, get_bo_info_args.map_offset);
  if (dev_heap_aligned == MAP_FAILED) {
    dev_heap_aligned = nullptr;
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  dev_heap_handle_guard.Dismiss();

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::FreeDeviceHeap() {
  hsa_status_t err = DestroyBOHandle(dev_heap_handle);
  assert(err == HSA_STATUS_SUCCESS && "Failed to destroy device heap BO handle.");
  dev_heap_aligned = nullptr;
  return err;
}

hsa_status_t XdnaDriver::CreateCmdBO(uint32_t size, BOHandle& cmd_bo_handle) const {
  amdxdna_drm_create_bo create_cmd_bo = {};
  create_cmd_bo.type = AMDXDNA_BO_CMD;
  create_cmd_bo.size = size;
  hsa_status_t err = xdna_ioctl(fd_, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_cmd_bo);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  BOHandle tmp_cmd_bo_handle;
  tmp_cmd_bo_handle.handle = create_cmd_bo.handle;
  tmp_cmd_bo_handle.size = size;

  // Unmap and close the command BO in case of error.
  MAKE_NAMED_SCOPE_GUARD(tmp_cmd_bo_handle_guard, [&] { DestroyBOHandle(tmp_cmd_bo_handle); });

  amdxdna_drm_get_bo_info cmd_bo_get_bo_info = {};
  cmd_bo_get_bo_info.handle = tmp_cmd_bo_handle.handle;
  err = xdna_ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &cmd_bo_get_bo_info);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  void* mem = mmap(nullptr, tmp_cmd_bo_handle.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_,
                   cmd_bo_get_bo_info.map_offset);
  if (mem == MAP_FAILED) {
    return HSA_STATUS_ERROR;
  }
  tmp_cmd_bo_handle.vaddr = mem;
  tmp_cmd_bo_handle.unmap_vaddr = true;

  tmp_cmd_bo_handle_guard.Dismiss();

  cmd_bo_handle = tmp_cmd_bo_handle;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::SubmitCmdChain(hsa_queue_t& q, void* queue_metadata,
                                        uint64_t first_pkt_idx, uint64_t num_pkts,
                                        uint32_t num_core_tiles) {
  auto kmq_metadata = static_cast<KmqMetadata*>(queue_metadata);

  // Instruction and arguments BOs (performance hint: up to 3 argument BOs per packet).
  std::vector<uint32_t> bo_handles;
  bo_handles.reserve(num_pkts * 4);

  // Commands to be submitted.
  std::vector<BOHandle> cmd_bo_handles;
  cmd_bo_handles.reserve(num_pkts);
  // Unmap and close the command BOs in case of an error.
  MAKE_NAMED_SCOPE_GUARD(cmd_bo_handles_guard, [&] {
    for (auto& bo_handle : cmd_bo_handles) {
      DestroyBOHandle(bo_handle);
    }
  });

  // Flag to reconfigure the hardware context because of a new PDI.
  bool reconfigure_queue = false;

  // Process all packets in a single command chain.
  auto* queue = static_cast<hsa_amd_aie_kernel_dispatch_packet_t*>(q.base_address);
  const uint64_t mask = q.size - 1;
  for (uint64_t i = 0; i < num_pkts; ++i) {
    const auto pkt_idx = (first_pkt_idx + i) & mask;
    auto* pkt = queue + pkt_idx;

    // Determine if the PDI is cached, if not it will be added to the PDI cache and the hardware
    // context will be reconfigured.
    auto pdi_bo_handle = FindBOHandle(pkt->pdi_addr);
    if (!pdi_bo_handle.IsValid()) {
      return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    }
    auto cached_pdi_index = kmq_metadata->pdi_cache.GetIndex(pdi_bo_handle.handle);
    if (cached_pdi_index == PDICache::NotFound) {
      FlushCpuCache(pdi_bo_handle.vaddr, 0, pdi_bo_handle.size);
      hsa_status_t err = kmq_metadata->pdi_cache.SetNext(pdi_bo_handle.handle, cached_pdi_index);
      if (err != HSA_STATUS_SUCCESS) {
        assert(false && "Failed to set PDI in cache.");
        return err;
      }
      reconfigure_queue = true;
    }

    // Add the instruction sequence BO handle to bo_handles and flush cache.
    void* insts_addr =
        reinterpret_cast<void*>(Concat<uint64_t>(pkt->insts_addr_high, pkt->insts_addr_low));
    auto instr_bo_handle = FindBOHandle(insts_addr);
    if (!instr_bo_handle.IsValid()) {
      assert(false && "Failed to find instruction sequence BO for command packet.");
      return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    }
    bo_handles.push_back(instr_bo_handle.handle);
    FlushCpuCache(insts_addr, 0, pkt->insts_size);

    // Add the argument BO handles to bo_handles.
    auto* kernarg_address = static_cast<uint64_t*>(pkt->kernarg_address);
    for (uint32_t kernarg_idx = 0; kernarg_idx < pkt->num_kernargs; ++kernarg_idx) {
      void* ptr = reinterpret_cast<void*>(kernarg_address[kernarg_idx]);
      auto bo_handle = FindBOHandle(ptr);
      if (!bo_handle.IsValid()) {
        assert(false && "Failed to find argument BO for command packet.");
        return HSA_STATUS_ERROR_INVALID_ALLOCATION;
      }
      bo_handles.push_back(bo_handle.handle);
    }

    // Create command for the kernel.
    const uint32_t cmd_dwords = (1 +  // CU mask
                                 2 +  // txn opcode
                                 3 +  // instruction sequence (address lo/hi + size)
                                 2 * pkt->num_kernargs);  // arguments (address lo/hi)
    const uint32_t cmd_data_bytesize = cmd_dwords * sizeof(uint32_t);
    const uint32_t cmd_bytesize = sizeof(ert_start_kernel_cmd) + cmd_data_bytesize;
    BOHandle cmd_bo_handle;
    hsa_status_t err = CreateCmdBO(cmd_bytesize, cmd_bo_handle);
    if (err != HSA_STATUS_SUCCESS) {
      assert(false && "Failed to create command BO.");
      return err;
    }
    cmd_bo_handles.push_back(cmd_bo_handle);

    auto* cmd = static_cast<ert_start_kernel_cmd*>(cmd_bo_handle.vaddr);
    memset(cmd, 0, cmd_bytesize);
    cmd->state = ERT_CMD_STATE_NEW;
    cmd->extra_cu_masks = 0;
    // The driver places a structure before each command in a command chain.
    // Need to increase the size of the command by the size of this structure.
    cmd->count = cmd_dwords + CMD_COUNT_SIZE_INCREASE;
    cmd->opcode = pkt->opcode;               // HSA_AMD_AIE_PACKET_OPCODE_KMQ == ERT_START_CU == 0x0
    cmd->data[0] = 0x1 << cached_pdi_index;  // CU mask bit
    cmd->data[1] = 0x3;                      // txn opcode
    cmd->data[2] = 0x0;                      // txn opcode
    cmd->data[3] = (DEV_ADDR_BASE |
                    (reinterpret_cast<uintptr_t>(insts_addr) &
                     DEV_ADDR_OFFSET_MASK));              // instruction sequence address (lo)
    cmd->data[4] = 0x0;                                   // instruction sequence address (hi)
    cmd->data[5] = (pkt->insts_size / sizeof(uint32_t));  // instruction sequence dword count
    for (uint32_t kernarg_idx = 0; kernarg_idx < pkt->num_kernargs; ++kernarg_idx) {
      const auto kernarg = kernarg_address[kernarg_idx];
      cmd->data[6 + 2 * kernarg_idx] = (kernarg & 0xFFFFFFFF);  // argument address (lo)
      cmd->data[6 + 2 * kernarg_idx + 1] = (kernarg >> 32);     // argument address (hi)
    }
  }

  // Reconfigure hardware context.
  if (reconfigure_queue) {
    // Destroy the existing hardware context.
    // Note: we can do this because we have forced synchronization between command chains. If we
    // move to a more asynchronous model, we will need to figure out how hardware context
    // destruction works while applications are running.
    hsa_status_t err = DestroyHwCtx(fd_, kmq_metadata->hw_ctx_handle);
    if (err != HSA_STATUS_SUCCESS) {
      assert(false && "Failed to destroy hardware context for queue.");
      return err;
    }
    kmq_metadata->hw_ctx_handle = AMDXDNA_INVALID_CTX_HANDLE;
    kmq_metadata->syncobj_handle = 0;

    // Create a new hardware context.
    err = CreateHwCtx(fd_, num_core_tiles, kmq_metadata);
    if (err != HSA_STATUS_SUCCESS) {
      assert(false && "Failed to configure hardware context for queue.");
      return err;
    }
  }

  // Remove duplicate BOs, since the driver reports an error if the same BO is provided multiple
  // times.
  std::sort(bo_handles.begin(), bo_handles.end());
  bo_handles.erase(std::unique(bo_handles.begin(), bo_handles.end()), bo_handles.end());

  // Flush cache for the arguments.
  for (uint64_t i = 0; i < num_pkts; ++i) {
    const auto pkt_idx = (first_pkt_idx + i) & mask;
    auto* pkt = queue + pkt_idx;
    FlushArguments(pkt);
  }

  if (num_pkts == 1) {
    // Single packet: submit the per-kernel cmd BO directly, no chain wrapper.
    uint64_t seq = 0;
    hsa_status_t status =
        SubmitCommand(fd_, cmd_bo_handles[0].handle, bo_handles, kmq_metadata->hw_ctx_handle, seq);
    if (status != HSA_STATUS_SUCCESS) {
      assert(false && "Failed to submit command.");
      return status;
    }
    status = WaitCommand(fd_, static_cast<ert_start_kernel_cmd*>(cmd_bo_handles[0].vaddr),
                         kmq_metadata->hw_ctx_handle, kmq_metadata->syncobj_handle, seq);
    if (status != HSA_STATUS_SUCCESS) {
      assert(false && "Failed waiting for command.");
      return status;
    }
  } else {
    // Create command chain for multi-packet dispatches.
    const size_t cmd_chain_data_bytesize = cmd_bo_handles.size() * sizeof(uint64_t);
    const size_t cmd_data_bytesize = sizeof(ert_cmd_chain_data) + cmd_chain_data_bytesize;
    const size_t cmd_bytesize = sizeof(ert_start_kernel_cmd) + cmd_data_bytesize;
    BOHandle cmd_bo_handle;
    hsa_status_t status = CreateCmdBO(cmd_bytesize, cmd_bo_handle);
    if (status != HSA_STATUS_SUCCESS) {
      assert(false && "Failed to create command chain BO.");
      return status;
    }
    MAKE_NAMED_SCOPE_GUARD(cmd_bo_handle_guard, [&] { DestroyBOHandle(cmd_bo_handle); });

    auto* cmd = static_cast<ert_start_kernel_cmd*>(cmd_bo_handle.vaddr);
    memset(cmd, 0, cmd_bytesize);
    cmd->state = ERT_CMD_STATE_NEW;
    cmd->count = static_cast<uint32_t>(cmd_data_bytesize / sizeof(uint32_t));
    cmd->opcode = ERT_CMD_CHAIN;
    auto* cmd_chain = reinterpret_cast<ert_cmd_chain_data*>(cmd->data);
    cmd_chain->command_count = static_cast<uint32_t>(cmd_bo_handles.size());
    for (size_t i = 0; i < cmd_bo_handles.size(); i++) {
      cmd_chain->data[i] = cmd_bo_handles[i].handle;
    }

    // Execute all commands in the command chain.
    uint64_t seq = 0;
    status = SubmitCommand(fd_, cmd_bo_handle.handle, bo_handles, kmq_metadata->hw_ctx_handle, seq);
    if (status != HSA_STATUS_SUCCESS) {
      assert(false && "Failed to submit command chain.");
      return status;
    }
    status = WaitCommand(fd_, static_cast<ert_start_kernel_cmd*>(cmd_bo_handle.vaddr),
                         kmq_metadata->hw_ctx_handle, kmq_metadata->syncobj_handle, seq);
    if (status != HSA_STATUS_SUCCESS) {
      assert(false && "Failed waiting for command chain.");
      return status;
    }
  }

  // Flush cache for the arguments again to ensure visibility of any changes made by the AIE kernels
  // and fire completion signal for each packet.
  for (uint64_t i = 0; i < num_pkts; ++i) {
    const auto pkt_idx = (first_pkt_idx + i) & mask;
    auto* pkt = queue + pkt_idx;

    // Flush cache.
    FlushArguments(pkt);

    // Fire completion signal.
    if (pkt->completion_signal.handle != 0) {
      core::Signal* sig = core::Signal::Convert(pkt->completion_signal);
      sig->SubRelease(1);
    }
  }

  // Guards will unmap and close cmd BOs and cmd_chain BO.

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::SPMAcquire(uint32_t preferred_node_id) const {
  // AIE does not support streaming performance monitor.
  return HSA_STATUS_ERROR_INVALID_AGENT;
}

hsa_status_t XdnaDriver::SPMRelease(uint32_t preferred_node_id) const {
  // AIE does not support streaming performance monitor.
  return HSA_STATUS_ERROR_INVALID_AGENT;
}

hsa_status_t XdnaDriver::SPMSetDestBuffer(uint32_t preferred_node_id, uint32_t size_bytes,
                                          uint32_t* timeout, uint32_t* size_copied,
                                          void* dest_mem_addr, bool* is_spm_data_loss) const {
  // AIE does not support streaming performance monitor.
  return HSA_STATUS_ERROR_INVALID_AGENT;
}

hsa_status_t XdnaDriver::IsModelEnabled(bool* enable) const {
  // AIE does not support a driver model.
  *enable = false;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::DestroyBOHandle(BOHandle& bo_handle) const {
  if (!bo_handle.IsValid()) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t unmap_err = HSA_STATUS_SUCCESS;

  // Unmap the memory.
  if (bo_handle.unmap_vaddr) {
    if (munmap(bo_handle.vaddr, bo_handle.size) != 0) {
      unmap_err = HSA_STATUS_ERROR;
      assert(false && "Failed to unmap BO memory.");
    } else {
      bo_handle.unmap_vaddr = false;
      bo_handle.vaddr = nullptr;
      bo_handle.size = 0;
    }
  }

  // Close the BO handle.
  drm_gem_close close_bo_args = {};
  close_bo_args.handle = bo_handle.handle;
  hsa_status_t ioctl_err = xdna_ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_bo_args);
  bo_handle.handle = AMDXDNA_INVALID_BO_HANDLE;

  if (ioctl_err != HSA_STATUS_SUCCESS) {
    return ioctl_err;
  }
  return unmap_err;
}

XdnaDriver::BOHandle XdnaDriver::FindBOHandle(void* mem) const {
  auto it = vmem_addr_mappings.lower_bound(mem);
  if (it == vmem_addr_mappings.cend()) {
    // Exact address not found or is larger than the largest address.
    return BOHandle{};
  }

  if (it->first == mem) {
    // Exact address found.
    return it->second;
  }

  if (it == vmem_addr_mappings.cbegin()) {
    // Address is smaller than the smallest registered address.
    return BOHandle{};
  }

  // Go back one element, since lower_bound returns an iterator to the element that is equal or
  // greater.
  --it;

  assert(it->first < mem);
  if (mem >= (static_cast<char*>(it->first) + it->second.size)) {
    // Address is not from this allocation.
    return BOHandle{};
  }

  return it->second;
}

hsa_status_t XdnaDriver::SetTrapHandler(uint32_t node_id, const void* base, uint64_t base_size,
                                        const void* buffer_base, uint64_t buffer_base_size) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t XdnaDriver::AllocateScratchMemory(uint32_t node_id, uint64_t size, void** mem) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t XdnaDriver::GetDeviceHandle(uint32_t node_id, void** device_handle) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t XdnaDriver::GetDeviceFd(uint32_t node_id, int *fd) const {
  *fd = fd_;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::GetClockCounters(uint32_t node_id, HsaClockCounters* clock_counter) const {
  return HSA_STATUS_ERROR;
}


hsa_status_t XdnaDriver::GetTileConfig(uint32_t node_id, HsaGpuTileConfig* config) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t XdnaDriver::GetWallclockFrequency(uint32_t node_id, uint64_t* frequency) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t XdnaDriver::AvailableMemory(uint32_t node_id, uint64_t* available_size) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t XdnaDriver::RegisterMemory(void* ptr, uint64_t size, HsaMemFlags mem_flags) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t XdnaDriver::DeregisterMemory(void* ptr) const { return HSA_STATUS_ERROR; }

hsa_status_t XdnaDriver::MakeMemoryResident(const void* mem, size_t size, uint64_t* alternate_va,
                                            const HsaMemMapFlags* mem_flags, uint32_t num_nodes,
                                            const uint32_t* nodes) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t XdnaDriver::GetQueueSaveAreaInfo(HSA_QUEUEID queue_id, void** address, size_t* size) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t XdnaDriver::MakeMemoryUnresident(const void* mem) const { return HSA_STATUS_ERROR; }

}  // namespace AMD
}  // namespace rocr
