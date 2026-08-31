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

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <libdrm/drm.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "inc/hsa_ext_amd_aie.h"
#include "core/inc/amd_aie_agent.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/runtime.h"
#include "core/inc/signal.h"
#include "core/util/memory.h"
#include "core/util/os.h"
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

/// @brief Payload of an @ref ERT_START_NPU_PREEMPT_ELF command, placed after the CU mask.
///
/// This should match the command format defined in xdna-driver (amdxdna_cmd_preempt_data) and XRT
/// ERT (ert_npu_preempt_data). The save and restore buffers are for preemption; a design without
/// preemption sections leaves them null, which the aie2p firmware accepts.
struct ert_npu_preempt_data {
  /// @brief Device address of the control code.
  uint64_t instruction_buffer;
  /// @brief Device address of the preemption save buffer, or 0.
  uint64_t save_buffer;
  /// @brief Device address of the preemption restore buffer, or 0.
  uint64_t restore_buffer;
  /// @brief Size of the control code in bytes.
  uint32_t instruction_buffer_size;
  /// @brief Size of the save buffer in bytes, or 0.
  uint32_t save_buffer_size;
  /// @brief Size of the restore buffer in bytes, or 0.
  uint32_t restore_buffer_size;
  /// @brief Number of dwords of properties following this structure. Reserved; must be 0.
  uint32_t instruction_prop_count;
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
  /// @brief Unknown device.
  Unknown = 0,
  /// @brief Phoenix (npu1), aie2. PDI + instruction sequence dispatch only.
  Phx,
  /// @brief Strix / Strix Halo / Krackan (npu4/5/6), aie2p. Also supports full-ELF dispatch.
  Stx,
};

/// @brief XDNA device ID.
struct XDNADeviceId {
  uint16_t device;

  bool operator<(const XDNADeviceId& other) const { return device < other.device; }
};

/// @brief Supported XDNA devices.
static const std::map<XDNADeviceId, XDNADeviceType> supported_xdna_devices = {
    {{0x1502}, XDNADeviceType::Phx},  // Phoenix (npu1, aie2)
    {{0x17f0}, XDNADeviceType::Stx},  // Strix / Strix Halo / Krackan (npu4/5/6, aie2p)
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

/// @brief Dwords of padding declared on a PDI + instruction sequence command's payload.
///
/// The firmware aborts a command chain whose slot carries an odd argument count below 15. The
/// driver takes that count straight from this command: amdxdna_cmd_get_payload() reports
/// (count - 1) dwords and aie2_cmdlist_fill_one_slot_cf() uses it as arg_cnt. This command's real
/// payload after the CU mask is an odd number of dwords, so it needs an odd amount of padding to
/// come out even, and one dword is the least that does it.
constexpr uint32_t CMD_COUNT_SIZE_INCREASE = 1;

/// @brief Size of the driver's per-command chain buffer, from MAX_CHAIN_CMDBUF_SIZE in
/// xdna-driver (aie2_msg_priv.h). Every command in a chain has to fit in one of these.
constexpr uint32_t MAX_CHAIN_CMDBUF_SIZE = 4096;

/// @brief Size of the driver's per-command chain slot header, from cmd_chain_slot_npu in
/// xdna-driver (aie2_msg_priv.h). The slot is the header plus the command's argument dwords.
constexpr uint32_t CHAIN_SLOT_HEADER_BYTESIZE = 52;

/// @brief Returns the bytes a command with @p arg_cnt argument dwords occupies in a chain.
static constexpr uint32_t ChainSlotBytesize(uint32_t arg_cnt) {
  return CHAIN_SLOT_HEADER_BYTESIZE + arg_cnt * sizeof(uint32_t);
}

/// @brief Largest value ert_start_kernel_cmd::count can hold; it is an 11-bit field.
constexpr uint32_t MAX_CMD_COUNT = (1u << 11) - 1;

/// @brief Required alignment of a full-ELF control code's device address.
///
/// Not documented anywhere; established by sweeping the control code across offsets within its
/// allocation on npu4 and observing which dispatches complete. Every 16 KiB-aligned address
/// works and every other one hangs, so the runtime rejects the misaligned ones up front. The
/// PDI has no such requirement -- its address is a plain 64-bit store into the control code.
constexpr uint32_t CTRL_CODE_DEV_ADDR_ALIGNMENT = 16384;

/// @brief Number of argument dwords a full-ELF command carries.
///
/// The arguments themselves are patched into the control code, so all the command has to carry
/// is the kernel opcode, which the driver overwrites with its own TXN constant.
constexpr uint32_t ELF_CMD_ARG_DWORDS = sizeof(uint64_t) / sizeof(uint32_t);

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

  /// @brief Drops every entry added since the cache held @p count of them.
  void Truncate(size_type count) {
    assert(count <= entry_count);
    entry_count = count;
  }

  constexpr uint32_t operator[](size_type index) const { return entries[index]; }
};

/// @brief Which dispatch ABI a queue is using.
///
/// A queue is homogeneous. The two modes need incompatible hardware contexts -- PDI + instruction
/// sequence needs CU configuration, full-ELF must not have any -- and a hardware context's CU
/// configuration cannot be changed once set, so switching a live queue between them is not
/// possible.
enum class QueueMode {
  /// @brief No packet submitted yet; the first one decides.
  Undecided,
  /// @brief PDI plus a separate instruction sequence, dispatched as ERT_START_CU.
  PdiInsts,
  /// @brief Full ELF, dispatched as ERT_START_NPU_PREEMPT_ELF.
  FullElf,
};

/// @brief Returns the dispatch shape @p pkt is asking for.
///
/// A packet asks for a full-ELF dispatch by giving the offset at which its control code takes the
/// PDI's device address; a PDI plus instruction sequence has no such patch site and leaves the
/// field zero.
static QueueMode PacketMode(const hsa_amd_aie_kernel_dispatch_packet_t* pkt) {
  return (pkt->pdi_patch_offset != 0) ? QueueMode::FullElf : QueueMode::PdiInsts;
}

/// @brief Metadata for a Kernel Mode Queue (KMQ).
struct KmqMetadata {
  uint32_t hw_ctx_handle = AMDXDNA_INVALID_CTX_HANDLE;
  uint32_t syncobj_handle = 0;
  PDICache pdi_cache;
  QueueMode mode = QueueMode::Undecided;
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
/// The context is given a CU configuration only if the KMQ has PDIs to configure it with. A
/// full-ELF queue never does, and configuring one would be wrong: the driver rejects a second
/// CONFIG_CU on the same context, so a CU configuration set now could not be undone later.
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

  if (!kmq_metadata->pdi_cache.empty()) {
    // Create hardware context configuration.
    const size_t num_cus = kmq_metadata->pdi_cache.size();
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

    xdna_config_cu_param->num_cus = static_cast<uint16_t>(num_cus);
    for (size_t i = 0; i < num_cus; i++) {
      xdna_config_cu_param->cu_configs[i].cu_bo = kmq_metadata->pdi_cache[i];
      xdna_config_cu_param->cu_configs[i].cu_func = default_cu_func;
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

/**
 * @brief Resolves a virtual address to the BO handle, along with its allocation base and size.
 *
 * @param[in] mem virtual address to resolve
 * @param[in] agent agent that owns the memory pool associated with the virtual address
 * @param[out] bo_handle pointer to store the BO handle associated with the allocation
 * @param[out] base pointer to store the base address of the allocation (optional)
 * @param[out] size pointer to store the size of the allocation (optional)
 *
 * @return @c HSA_STATUS_SUCCESS on success, or an error status if the address does not belong to
 * an allocation accessible to the agent
 */
static hsa_status_t ResolveBOHandle(void* mem, const core::Agent& agent, uint32_t* bo_handle,
                                    void** base, size_t* size) {
  void* local_base = nullptr;
  core::DriverMemoryHandle handle{};
  if (core::Runtime::runtime_singleton_->FindDriverMemoryHandle(mem, &agent, &local_base,
                                                                &handle) != HSA_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }
  if (base != nullptr) *base = local_base;
  if (size != nullptr) *size = handle.size;
  *bo_handle = static_cast<uint32_t>(handle.handle);
  return HSA_STATUS_SUCCESS;
}


/// @brief BO handle information.
struct BOHandle {
  /// Mapped address.
  void* vaddr = nullptr;
  /// Handle returned by xdna, or AMDXDNA_INVALID_BO_HANDLE when this holds no BO.
  uint32_t handle = AMDXDNA_INVALID_BO_HANDLE;
  /// Size in bytes.
  size_t size = 0;

  constexpr BOHandle() = default;
  constexpr BOHandle(void* vaddr, uint32_t handle, size_t size)
      : vaddr{vaddr}, handle{handle}, size{size} {}
  constexpr bool IsValid() const { return handle != AMDXDNA_INVALID_BO_HANDLE; }
};

/// @brief Returns true if @p vaddr lies within the device heap mapping at @p heap_base.
///
/// Dev heap BOs carve their VA out of the device heap and borrow its mapping, so DestroyBOHandle
/// must not unmap them. This lets it distinguish those from BO_SHAREs (which own an independent
/// mmap) by the VA alone, without the caller tracking mapping ownership.
static bool IsDevHeapVA(const void* heap_base, const void* vaddr) {
  if (heap_base == nullptr) return false;
  const auto addr = reinterpret_cast<uintptr_t>(vaddr);
  const auto base = reinterpret_cast<uintptr_t>(heap_base);
  return (addr >= base) && ((addr - base) < XdnaDriver::GetDevHeapByteSize());
}

/// @brief Destroys @p bo_handle.
///
/// @note Unmaps the virtual address and closes the BO, even if the former fails.
///
/// @param[in] fd driver file descriptor
/// @param[in] heap_base base of the device heap mapping, which dev heap BOs borrow
/// @param[in,out] bo_handle BO handle to destroy
static hsa_status_t DestroyBOHandle(int fd, const void* heap_base, BOHandle& bo_handle) {
  if (!bo_handle.IsValid()) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t unmap_err = HSA_STATUS_SUCCESS;

  // Unmap only non-null BO_SHAREs, which own an independent mmap. Dev heap allocations carve
  // their VA out of the shared device heap and must leave that mapping intact; they
  // are recognized by the VA falling inside the device-heap range.
  if ((bo_handle.vaddr != nullptr) && !IsDevHeapVA(heap_base, bo_handle.vaddr)) {
    if (munmap(bo_handle.vaddr, bo_handle.size) != 0) {
      unmap_err = HSA_STATUS_ERROR;
      assert(false && "Failed to unmap BO memory.");
    } else {
      bo_handle.vaddr = nullptr;
      bo_handle.size = 0;
    }
  }

  // Close the BO.
  drm_gem_close close_bo_args = {};
  close_bo_args.handle = bo_handle.handle;
  hsa_status_t ioctl_err = xdna_ioctl(fd, DRM_IOCTL_GEM_CLOSE, &close_bo_args);
  if (ioctl_err != HSA_STATUS_SUCCESS) {
    return ioctl_err;
  }
  bo_handle.handle = AMDXDNA_INVALID_BO_HANDLE;

  return unmap_err;
}

/// @brief Creates a command BO of @p size bytes and returns it in @p cmd_bo_handle.
static hsa_status_t CreateCmdBO(int fd, const void* heap_base, uint32_t size,
                                BOHandle& cmd_bo_handle) {
  amdxdna_drm_create_bo create_cmd_bo = {};
  create_cmd_bo.type = AMDXDNA_BO_CMD;
  create_cmd_bo.size = size;
  hsa_status_t err = xdna_ioctl(fd, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_cmd_bo);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  BOHandle tmp_cmd_bo_handle;
  tmp_cmd_bo_handle.handle = create_cmd_bo.handle;
  tmp_cmd_bo_handle.size = size;

  // Unmap and close the command BO in case of error.
  MAKE_NAMED_SCOPE_GUARD(tmp_cmd_bo_handle_guard,
                         [&] { DestroyBOHandle(fd, heap_base, tmp_cmd_bo_handle); });

  amdxdna_drm_get_bo_info cmd_bo_get_bo_info = {};
  cmd_bo_get_bo_info.handle = tmp_cmd_bo_handle.handle;
  err = xdna_ioctl(fd, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &cmd_bo_get_bo_info);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  void* mem = mmap(nullptr, tmp_cmd_bo_handle.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                   cmd_bo_get_bo_info.map_offset);
  if (mem == MAP_FAILED) {
    return HSA_STATUS_ERROR;
  }
  tmp_cmd_bo_handle.vaddr = mem;

  tmp_cmd_bo_handle_guard.Dismiss();

  cmd_bo_handle = tmp_cmd_bo_handle;

  return HSA_STATUS_SUCCESS;
}

/// @brief Creates a command BO with @p declared_dwords of payload and fills in its header.
///
/// @p declared_dwords is what the command reports in its count field, and the driver reads that
/// straight back: amdxdna_cmd_get_payload() hands the chain fill path (declared_dwords - 1) dwords
/// starting at data[1], and the fill path memcpy()s all of them into the chain slot. So the whole
/// declared payload is allocated and zeroed here, including any padding a caller declares but does
/// not write -- otherwise the driver copies uninitialised memory to the device.
///
/// @param[in] cu_mask value for data[0]. The driver derives a CU index from it with ffs() - 1 and
/// rejects a command with no bit set, even on the paths that then ignore the index.
/// @param[out] cmd_bo the command BO, owned by the caller on success.
/// @param[out] cmd the mapped command, header filled in and payload zeroed.
static hsa_status_t CreateCommand(int fd, const void* heap_base, uint32_t declared_dwords,
                                  uint32_t opcode, uint32_t cu_mask, BOHandle* cmd_bo,
                                  ert_start_kernel_cmd** cmd) {
  if (declared_dwords > MAX_CMD_COUNT) {
    // count is an 11-bit field and the chain slot is sized from it, so letting it wrap would
    // under-size the slot and overflow the chain buffer.
    return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
  }

  const uint32_t cmd_bytesize = sizeof(ert_start_kernel_cmd) + declared_dwords * sizeof(uint32_t);
  const hsa_status_t err = CreateCmdBO(fd, heap_base, cmd_bytesize, *cmd_bo);
  if (err != HSA_STATUS_SUCCESS) {
    assert(false && "Failed to create command BO.");
    return err;
  }

  auto* new_cmd = static_cast<ert_start_kernel_cmd*>(cmd_bo->vaddr);
  memset(new_cmd, 0, cmd_bytesize);
  new_cmd->state = ERT_CMD_STATE_NEW;
  new_cmd->extra_cu_masks = 0;
  new_cmd->count = declared_dwords;
  new_cmd->opcode = opcode;
  new_cmd->data[0] = cu_mask;

  *cmd = new_cmd;
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

uint64_t XdnaDriver::GetDevHeapByteSize() { return dev_heap_size; }

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


/// @brief Reads the PCI device ID of @p devnode_name from sysfs.
static hsa_status_t ReadDeviceId(const std::string& devnode_name, uint16_t* device_id) {
  const std::string device_id_file =
      std::string(sysfs_path) + "/" + devnode_name + "/device/device";

  // Device ID is in hex, with a "0x" prefix.
  std::ifstream is(device_id_file);
  if (!is.good() || !(is >> std::hex >> *device_id)) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

/// @brief Queries the device address the NPU sees for @p bo_handle.
///
/// @param[out] dev_addr device address, or 0 if the BO has none. A BO shared with the host has
/// none: the NPU walks the same page tables, so its host VA is already the address the hardware
/// uses.
static hsa_status_t GetBODevAddr(int fd, uint32_t bo_handle, uint64_t* dev_addr) {
  amdxdna_drm_get_bo_info get_bo_info_args = {};
  get_bo_info_args.handle = bo_handle;
  const hsa_status_t err = xdna_ioctl(fd, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info_args);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  *dev_addr = (get_bo_info_args.xdna_addr == AMDXDNA_INVALID_ADDR) ? 0 : get_bo_info_args.xdna_addr;
  return HSA_STATUS_SUCCESS;
}

/// @brief Resolves @p ptr to its BO handle, the device address the NPU sees for it, and how many
/// bytes of the allocation follow it.
///
/// @param[out] dev_addr device address of @p ptr, or 0 if the allocation has none. A buffer
/// shared with the host has none, which also means the NPU cannot fetch from it directly.
/// @param[out] bytes_from_ptr bytes between @p ptr and the end of its allocation.
static hsa_status_t ResolveDeviceBuffer(int fd, void* ptr, const core::Agent& agent,
                                        uint32_t* bo_handle, uint64_t* dev_addr,
                                        size_t* bytes_from_ptr) {
  void* base = nullptr;
  size_t alloc_size = 0;
  hsa_status_t err = ResolveBOHandle(ptr, agent, bo_handle, &base, &alloc_size);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  err = GetBODevAddr(fd, *bo_handle, dev_addr);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  // ResolveBOHandle reports the allocation; the packet may point part-way into it.
  const size_t offset = static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(base);
  if (offset > alloc_size) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }
  *bytes_from_ptr = alloc_size - offset;
  if (*dev_addr != 0) *dev_addr += offset;
  return HSA_STATUS_SUCCESS;
}

/// @brief Resolves each of the packet's kernel arguments and adds its BO to @p bo_handles.
///
/// The hardware reaches argument buffers through DMA descriptors rather than fetching them, so
/// they need no device address -- only to be listed so the driver keeps them resident.
static hsa_status_t AddKernargBOs(const hsa_amd_aie_kernel_dispatch_packet_t* pkt,
                                  const core::Agent& agent, std::vector<uint32_t>& bo_handles) {
  if (pkt->num_kernargs != 0 && pkt->kernarg_address == nullptr) {
    return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
  }

  auto* kernarg_address = static_cast<uint64_t*>(pkt->kernarg_address);
  for (uint32_t kernarg_idx = 0; kernarg_idx < pkt->num_kernargs; ++kernarg_idx) {
    void* ptr = reinterpret_cast<void*>(kernarg_address[kernarg_idx]);
    uint32_t arg_handle = AMDXDNA_INVALID_BO_HANDLE;
    const hsa_status_t err = ResolveBOHandle(ptr, agent, &arg_handle, nullptr, nullptr);
    if (err != HSA_STATUS_SUCCESS) {
      assert(false && "Failed to find argument BO for command packet.");
      return err;
    }
    bo_handles.push_back(arg_handle);
  }
  return HSA_STATUS_SUCCESS;
}

/// @brief Maps a PCI device ID to its device type, or @ref XDNADeviceType::Unknown if this
/// runtime does not support it.
static XDNADeviceType DeviceTypeOf(uint16_t device_id) {
  const auto it = supported_xdna_devices.find({device_id});
  return (it == supported_xdna_devices.end()) ? XDNADeviceType::Unknown : it->second;
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

  uint16_t device_id = 0;
  err = ReadDeviceId(devnode_name_, &device_id);
  if (err != HSA_STATUS_SUCCESS) {
    assert(false && "Failed to read the XDNA device ID from sysfs.");
    return err;
  }
  // Record it on the node: it is what identifies the device to anything holding the agent, and
  // the dispatch path derives the device type from it rather than re-reading sysfs.
  node_props.DeviceId = device_id;

  // Fill in node properties that depend on device type.
  std::fill_n(node_props.AMDName, HSA_PUBLIC_NAME_SIZE, 0);
  switch (DeviceTypeOf(device_id)) {
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

hsa_status_t XdnaDriver::AllocateMemory(const core::MemoryRegion& mem_region,
                                        core::MemoryRegion::AllocateFlags alloc_flags, size_t size,
                                        uint32_t node_id, core::DriverMemoryHandle* handle) {
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
  MAKE_NAMED_SCOPE_GUARD(bo_guard, [&] { DestroyBOHandle(fd_, dev_heap_vaddr, bo_handle); });

  amdxdna_drm_get_bo_info get_bo_info_args = {};
  get_bo_info_args.handle = create_bo_args.handle;
  err = xdna_ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info_args);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  if (use_bo_share) {
    if (alloc_flags & core::MemoryRegion::AllocateMemoryOnly) {
      bo_handle.vaddr = nullptr;
    } else {
      bo_handle.vaddr =
          mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, get_bo_info_args.map_offset);
      if (bo_handle.vaddr == MAP_FAILED) {
        // bo_guard must not try to unmap MAP_FAILED.
        bo_handle.vaddr = nullptr;
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
      }
    }
  } else {
    /// This is dev heap and is already mapped. See InitDeviceHeap().
    bo_handle.vaddr = reinterpret_cast<void*>(get_bo_info_args.vaddr);
  }

  bo_guard.Dismiss();

  // The handle word is the driver-native BO id. vaddr is the allocation's real VA (the VA for dev
  // heap BOs, the mmap'd VA for share BOs). It is nullptr only for AllocateMemoryOnly SHARE BOs.
  // FreeMemory decides whether to unmap by the VA itself (dev heap range vs. an owned mmap), so no
  // ownership flag is carried. Export-only fields stay at defaults.
  handle->handle = bo_handle.handle;
  handle->vaddr = bo_handle.vaddr;
  handle->size = size;
  handle->owner = this;
  handle->owns_allocation = true;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::FreeMemory(const core::DriverMemoryHandle& handle) {
  if (handle.handle == AMDXDNA_INVALID_BO_HANDLE) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }

  BOHandle bo_handle;
  bo_handle.handle = static_cast<uint32_t>(handle.handle);
  bo_handle.size = handle.size;
  bo_handle.vaddr = handle.vaddr;

  return DestroyBOHandle(fd_, dev_heap_vaddr, bo_handle);
}

hsa_status_t XdnaDriver::CreateQueue(uint32_t node_id, HSA_QUEUE_TYPE type, uint32_t queue_pct,
                                     HSA::hsa_amd_queue_priority_internal_t priority,
                                     uint32_t sdma_engine_id, void* queue_addr,
                                     uint64_t queue_size_bytes, uint64_t queue_metadata_size_bytes,
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

hsa_status_t XdnaDriver::CreateKernelModeQueue(size_t queue_size, uint32_t num_core_tiles,
                                               void** queue_metadata) const {
  auto kmq_metadata = std::make_unique<KmqMetadata>();
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

hsa_status_t XdnaDriver::ExportMemoryHandle(const core::Agent& agent,
                                            const core::DriverMemoryHandle& handle,
                                            core::ShareType type, void* export_handle) {
  (void)agent;
  if (export_handle == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  switch (type) {
    case core::ShareType::DMABUF_FD: {
      if (handle.handle == AMDXDNA_INVALID_BO_HANDLE) {
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

hsa_status_t XdnaDriver::ImportMemoryHandle(const core::Agent& agent,
                                            core::DriverMemoryHandle* handle, core::ShareType type,
                                            void* import_handle, void* mem) {
  (void)agent;
  (void)mem;
  if (handle == nullptr || import_handle == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  switch (type) {
    case core::ShareType::DMABUF_FD: {
      const auto* source = static_cast<const core::DriverMemoryHandle*>(import_handle);

      const int dmabuf_fd = source->dmabuf_fd;

      drm_prime_handle import_params = {};
      import_params.handle = AMDXDNA_INVALID_BO_HANDLE;
      import_params.fd = dmabuf_fd;
      hsa_status_t err = xdna_ioctl(fd_, DRM_IOCTL_PRIME_FD_TO_HANDLE, &import_params);
      if (err != HSA_STATUS_SUCCESS) {
        return err;
      }

      *handle = core::DriverMemoryHandle{import_params.handle};
      handle->owner = this;

      // A drm_file holds at most one GEM handle per object, so importing an allocation this
      // driver already owns hands back the owner's own handle instead of creating one, and
      // releasing that in DestroyMemoryHandle would destroy the allocation.
      handle->owns_allocation = (source->owner != this) ||
          (source->handle != static_cast<uint64_t>(import_params.handle));

      // Establish the size from the dma-buf.
      struct stat dmabuf_stat = {};
      if (fstat(dmabuf_fd, &dmabuf_stat) != 0) {
        const hsa_status_t rollback_err = DestroyMemoryHandle(handle);
        assert(rollback_err == HSA_STATUS_SUCCESS && "Failed to release the imported BO.");
        (void)rollback_err;
        return HSA_STATUS_ERROR_INVALID_ALLOCATION;
      }
      handle->size = static_cast<size_t>(dmabuf_stat.st_size);

      return HSA_STATUS_SUCCESS;
    }
    case core::ShareType::FABRIC_HANDLE:
      return HSA_STATUS_ERROR;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

hsa_status_t XdnaDriver::Map(const core::DriverMemoryHandle& handle, void* mem, size_t offset,
                             size_t size, hsa_access_permission_t perms, uint32_t node_id) {
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
  void* mapped_ptr =
      mmap(mem, size, PermissionsToMmapFlags(perms), MAP_FIXED | MAP_SHARED, params.fd, offset);
  close(params.fd);
  if (mapped_ptr == MAP_FAILED) {
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::Unmap(const core::DriverMemoryHandle& handle, void* mem, size_t offset,
                               size_t size, uint32_t node_id) {
  (void)node_id;
  if (munmap(mem, size) != 0) {
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::CreateShareableHandle(core::DriverMemoryHandle* handle,
                                               const core::Agent& agent, uint64_t* offset) {
  (void)agent;

  // On input, handle is the allocation handle whose native id is the BO handle (see
  // AllocateMemory).
  const auto bo_handle = static_cast<uint32_t>(handle->handle);
  if (bo_handle == AMDXDNA_INVALID_BO_HANDLE) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }

  // Get offset.
  amdxdna_drm_get_bo_info get_bo_info_args = {};
  get_bo_info_args.handle = bo_handle;
  hsa_status_t err = xdna_ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info_args);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  // Get fd associated with the handle.
  drm_prime_handle params = {};
  params.handle = bo_handle;
  params.flags = DRM_RDWR;
  params.fd = -1;
  err = xdna_ioctl(fd_, DRM_IOCTL_PRIME_HANDLE_TO_FD, &params);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  // handle->handle and handle->size carry over from allocation
  handle->dmabuf_fd = params.fd;
  handle->mmap_offset = get_bo_info_args.map_offset;

  *offset = 0;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::DestroyMemoryHandle(core::DriverMemoryHandle* handle) {
  // Attempt every release even if an earlier one fails, and clear the handle either way: a
  // handle left holding an fd that was already closed would close it a second time, by which
  // point the descriptor may name something else entirely.
  hsa_status_t err = HSA_STATUS_SUCCESS;

  // Close the dmabuf_fd.
  if (handle->dmabuf_fd >= 0 && close(handle->dmabuf_fd) != 0) {
    err = HSA_STATUS_ERROR;
  }

  // Close the BO handle, unless there is none or it is borrowed from the handle that owns it,
  // which ImportMemoryHandle decided when it created this one.
  if (handle->owns_allocation &&
      static_cast<uint32_t>(handle->handle) != AMDXDNA_INVALID_BO_HANDLE) {
    drm_gem_close close_params = {};
    close_params.handle = handle->handle;
    hsa_status_t ioctl_err = xdna_ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_params);
    if (ioctl_err != HSA_STATUS_SUCCESS) {
      err = ioctl_err;
    }
  }
  *handle = {};

  return err;
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
  dev_heap_bo = create_bo_args.handle;

  // Unmap memory and close the dev heap BO in case of error.
  MAKE_NAMED_SCOPE_GUARD(dev_heap_guard, [&] { FreeDeviceHeap(); });

  amdxdna_drm_get_bo_info get_bo_info_args = {};
  get_bo_info_args.handle = dev_heap_bo;
  err = xdna_ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info_args);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  // ReserveMemory over-allocates, aligns, and trims the slack, leaving exactly one
  // dev_heap_size mapping to own.
  void* heap = os::ReserveMemory(nullptr, dev_heap_size, dev_heap_alignment, os::MEM_PROT_NONE);
  if (heap == nullptr) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  if (!os::MapMemory(heap, dev_heap_size, os::MEM_PROT_RW, fd_, get_bo_info_args.map_offset)) {
    os::ReleaseMemory(heap, dev_heap_size);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  dev_heap_vaddr = heap;

  dev_heap_guard.Dismiss();

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::FreeDeviceHeap() {
  if (dev_heap_bo == AMDXDNA_INVALID_BO_HANDLE) {
    return HSA_STATUS_SUCCESS;
  }

  // Unmap dev heap.
  hsa_status_t munmap_err = HSA_STATUS_SUCCESS;
  if (dev_heap_vaddr != nullptr && !os::ReleaseMemory(dev_heap_vaddr, dev_heap_size)) {
    munmap_err = HSA_STATUS_ERROR;
    assert(false && "Failed to unmap device heap BO.");
  }

  // dev_heap_vaddr is deliberately left set: DestroyBOHandle uses IsDevHeapVA() to
  // recognize dev heap allocations that carve their VA out of the heap and must not be
  // unmapped. Clearing it here would make a dev heap freed after teardown munmap a VA
  // range that has since been recycled. InitDeviceHeap() overwrites it on re-init.

  // Close the BO.
  drm_gem_close close_bo_args = {};
  close_bo_args.handle = dev_heap_bo;
  hsa_status_t ioctl_err = xdna_ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_bo_args);
  assert(ioctl_err == HSA_STATUS_SUCCESS && "Failed to destroy device heap BO handle.");
  if (ioctl_err != HSA_STATUS_SUCCESS) {
    return ioctl_err;
  }

  dev_heap_bo = AMDXDNA_INVALID_BO_HANDLE;

  return munmap_err;
}

/// @brief Builds an ERT_START_CU command for a PDI + instruction sequence packet.
///
/// Adds the packet's instruction and argument BOs to @p bo_handles, and sets @p reconfigure if the
/// packet introduced a PDI the hardware context does not know about yet.
///
/// @param[out] cmd_bo the command BO, owned by the caller on success.
/// @param[out] arg_cnt number of argument dwords the driver will see, which determines how many of
/// these commands fit in one chain.
static hsa_status_t BuildPdiInstsCommand(int fd, const void* heap_base, const void* packet,
                                         const core::Agent& agent, KmqMetadata* kmq_metadata,
                                         std::vector<uint32_t>& bo_handles, bool* reconfigure,
                                         BOHandle* cmd_bo, uint32_t* arg_cnt) {
  const auto* pkt = static_cast<const hsa_amd_aie_kernel_dispatch_packet_t*>(packet);

  // Determine if the PDI is cached, if not it will be added to the PDI cache and the hardware
  // context will be reconfigured.
  void* pdi_base = nullptr;
  size_t pdi_size = 0;
  uint32_t pdi_handle = AMDXDNA_INVALID_BO_HANDLE;
  hsa_status_t err = ResolveBOHandle(pkt->pdi_addr, agent, &pdi_handle, &pdi_base, &pdi_size);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }
  auto cached_pdi_index = kmq_metadata->pdi_cache.GetIndex(pdi_handle);
  if (cached_pdi_index == PDICache::NotFound) {
    FlushCpuCache(pdi_base, 0, pdi_size);
    err = kmq_metadata->pdi_cache.SetNext(pdi_handle, cached_pdi_index);
    if (err != HSA_STATUS_SUCCESS) {
      assert(false && "Failed to set PDI in cache.");
      return err;
    }
    *reconfigure = true;
  }

  // Add the instruction sequence BO handle to bo_handles and flush cache.
  void* insts_addr =
      reinterpret_cast<void*>(Concat<uint64_t>(pkt->insts_addr_high, pkt->insts_addr_low));
  uint32_t instr_handle = AMDXDNA_INVALID_BO_HANDLE;
  err = ResolveBOHandle(insts_addr, agent, &instr_handle, nullptr, nullptr);
  if (err != HSA_STATUS_SUCCESS) {
    assert(false && "Failed to find instruction sequence BO for command packet.");
    return err;
  }
  bo_handles.push_back(instr_handle);
  FlushCpuCache(insts_addr, 0, pkt->insts_size);

  err = AddKernargBOs(pkt, agent, bo_handles);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  const uint32_t cmd_dwords = (1 +  // CU mask
                               2 +  // txn opcode
                               3 +  // instruction sequence (address lo/hi + size)
                               2 * pkt->num_kernargs);  // arguments (address lo/hi)
  ert_start_kernel_cmd* cmd = nullptr;
  err = CreateCommand(fd, heap_base, cmd_dwords + CMD_COUNT_SIZE_INCREASE, ERT_START_CU,
                      0x1 << cached_pdi_index, cmd_bo, &cmd);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  cmd->data[1] = 0x3;  // txn opcode
  cmd->data[2] = 0x0;  // txn opcode
  cmd->data[3] = (DEV_ADDR_BASE |
                  (reinterpret_cast<uintptr_t>(insts_addr) &
                   DEV_ADDR_OFFSET_MASK));              // instruction sequence address (lo)
  cmd->data[4] = 0x0;                                   // instruction sequence address (hi)
  cmd->data[5] = (pkt->insts_size / sizeof(uint32_t));  // instruction sequence dword count
  // On this path the hardware patches the arguments into the instruction sequence itself, so
  // their addresses go in the command's register map.
  const auto* kernarg_address = static_cast<const uint64_t*>(pkt->kernarg_address);
  for (uint32_t kernarg_idx = 0; kernarg_idx < pkt->num_kernargs; ++kernarg_idx) {
    const auto kernarg = kernarg_address[kernarg_idx];
    cmd->data[6 + 2 * kernarg_idx] = (kernarg & 0xFFFFFFFF);  // argument address (lo)
    cmd->data[6 + 2 * kernarg_idx + 1] = (kernarg >> 32);     // argument address (hi)
  }

  // The driver reads count - 1 - extra_cu_masks dwords of payload into the chain slot.
  *arg_cnt = cmd->count - 1;
  return HSA_STATUS_SUCCESS;
}

/// @brief Builds an ERT_START_NPU_PREEMPT_ELF command for a full-ELF packet.
///
/// The packet's control code arrives with the application's argument addresses already patched in;
/// this writes the PDI's device address into it, which the application cannot know, and points the
/// command at it.
///
/// @param[out] cmd_bo the command BO, owned by the caller on success.
/// @param[out] arg_cnt number of argument dwords the driver will see.
static hsa_status_t BuildFullElfCommand(int fd, const void* heap_base, const void* packet,
                                        const core::Agent& agent, std::vector<uint32_t>& bo_handles,
                                        BOHandle* cmd_bo, uint32_t* arg_cnt) {
  const auto* pkt = static_cast<const hsa_amd_aie_kernel_dispatch_packet_t*>(packet);

  // The control code comes from the application already patched with its argument addresses; all
  // this path adds is the PDI's device address, which only the runtime can know. A packet only
  // reaches here when it asked for that patch, so pdi_patch_offset is non-zero.
  void* ctrl_code =
      reinterpret_cast<void*>(Concat<uint64_t>(pkt->insts_addr_high, pkt->insts_addr_low));

  // Validate what can be validated from the packet alone, before spending any ioctls on it.
  //
  // The PDI patch site is a 64-bit address and has to lie wholly inside the control code. Check
  // the size first: both operands are unsigned, so subtracting from an unvalidated insts_size or
  // adding to an unvalidated offset could wrap and let the bound pass. This also rejects a zero
  // insts_size.
  if (ctrl_code == nullptr || pkt->pdi_addr == nullptr || pkt->insts_size < sizeof(uint64_t) ||
      pkt->pdi_patch_offset > pkt->insts_size - sizeof(uint64_t) ||
      (pkt->pdi_patch_offset % sizeof(uint32_t)) != 0) {
    return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
  }

  uint32_t ctrl_code_handle = AMDXDNA_INVALID_BO_HANDLE;
  uint64_t ctrl_code_dev_addr = 0;
  size_t ctrl_code_avail = 0;
  hsa_status_t err = ResolveDeviceBuffer(fd, ctrl_code, agent, &ctrl_code_handle,
                                         &ctrl_code_dev_addr, &ctrl_code_avail);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }
  if (ctrl_code_dev_addr == 0) {
    // The control code has to be somewhere the NPU can fetch it from directly, which means the
    // device heap. A host-only buffer has no device address and would not be reachable.
    log_warning_n(10, "AIE full-ELF control code must be allocated from device memory.\n");
    return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
  }
  if ((ctrl_code_dev_addr % CTRL_CODE_DEV_ADDR_ALIGNMENT) != 0) {
    // Measured on npu4 (Strix, aie2p): the dispatch only completes when the control code's device
    // address is 16 KiB aligned. At 2 KiB, 4 KiB, 8 KiB, 12 KiB, 20 KiB and 24 KiB offsets the
    // packet is accepted and then never signals, so the failure surfaces as a hang rather than an
    // error. Reject it here, where it can still be reported.
    log_warning_n(10, "AIE full-ELF control code must be %u-byte aligned in device memory.\n",
                  CTRL_CODE_DEV_ADDR_ALIGNMENT);
    return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
  }
  // insts_size comes from the packet, so it has to be shown to fit before anything reads or
  // writes that many bytes.
  if (pkt->insts_size > ctrl_code_avail) {
    return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
  }
  bo_handles.push_back(ctrl_code_handle);

  // Write the PDI's device address into the control code where the application asked.
  uint32_t pdi_handle = AMDXDNA_INVALID_BO_HANDLE;
  uint64_t pdi_dev_addr = 0;
  size_t pdi_avail = 0;
  err = ResolveDeviceBuffer(fd, pkt->pdi_addr, agent, &pdi_handle, &pdi_dev_addr, &pdi_avail);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }
  if (pdi_dev_addr == 0) {
    log_warning_n(10, "AIE full-ELF PDI must be allocated from device memory.\n");
    return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
  }

  auto* site =
      reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(ctrl_code) + pkt->pdi_patch_offset);
  site[0] = static_cast<uint32_t>(pdi_dev_addr & 0xFFFFFFFF);
  site[1] = static_cast<uint32_t>(pdi_dev_addr >> 32);

  // The PDI is reached only through the address just written, so the command still has to list it
  // for the driver to keep it resident.
  FlushCpuCache(pkt->pdi_addr, 0, pdi_avail);
  bo_handles.push_back(pdi_handle);

  FlushCpuCache(ctrl_code, 0, pkt->insts_size);

  // The arguments are already patched into the control code, so they are not passed to the
  // hardware -- but they still have to be resident for the duration of the dispatch.
  err = AddKernargBOs(pkt, agent, bo_handles);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  const uint32_t cmd_dwords = 1 +  // CU mask
      sizeof(ert_npu_preempt_data) / sizeof(uint32_t) + ELF_CMD_ARG_DWORDS;
  ert_start_kernel_cmd* cmd = nullptr;
  // The ELF fill path ignores the CU mask, but the driver still derives a CU index from it before
  // dispatching and rejects the command if no bit is set.
  err = CreateCommand(fd, heap_base, cmd_dwords, ERT_START_NPU_PREEMPT_ELF, 0x1, cmd_bo, &cmd);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  // The payload starts after the CU mask, which is data[0]; that puts it at byte offset 8 in a
  // page-aligned command BO, so the 64-bit fields below are aligned.
  auto* npu = reinterpret_cast<ert_npu_preempt_data*>(&cmd->data[1]);
  npu->instruction_buffer = ctrl_code_dev_addr;
  npu->instruction_buffer_size = static_cast<uint32_t>(pkt->insts_size);
  // The save and restore buffers and the property count are left zero. This design has no
  // preemption sections, and aie2p firmware accepts null preemption buffers.
  //
  // A zeroed 64-bit kernel opcode follows. The driver overwrites its low dword with its own TXN
  // constant, so there is nothing for this code to fill in -- but the space has to be there,
  // because the driver sizes the chain slot from it.

  *arg_cnt = ELF_CMD_ARG_DWORDS;
  return HSA_STATUS_SUCCESS;
}

/// @brief Returns a name for an @ref ert_cmd_state value, for diagnostics.
static const char* ErtStateName(uint32_t state) {
  switch (state) {
    case ERT_CMD_STATE_INVALID:
      return "invalid";
    case ERT_CMD_STATE_NEW:
      return "new";
    case ERT_CMD_STATE_QUEUED:
      return "queued";
    case ERT_CMD_STATE_RUNNING:
      return "running";
    case ERT_CMD_STATE_COMPLETED:
      return "completed";
    case ERT_CMD_STATE_ERROR:
      return "error";
    case ERT_CMD_STATE_ABORT:
      return "abort";
    case ERT_CMD_STATE_SUBMITTED:
      return "submitted";
    case ERT_CMD_STATE_TIMEOUT:
      return "timeout";
    case ERT_CMD_STATE_NORESPONSE:
      return "no response";
    default:
      return "unrecognised";
  }
}

/// @brief Reports which command in a failed chain went wrong.
///
/// The firmware records how far it got in the chain itself. Without this a failed chain is only
/// a status code: the caller cannot tell whether the first command was malformed, whether the chain
/// was too long, or whether one dispatch in the middle timed out.
static void LogChainFailure(const volatile ert_start_kernel_cmd* chain_cmd,
                            const volatile ert_cmd_chain_data* chain, const BOHandle* cmd_bos,
                            size_t num_commands) {
  const uint32_t error_index = chain->error_index;
  const uint32_t submit_index = chain->submit_index;

  // error_index comes from the device, so it is only trustworthy as an index once it has been
  // checked against the chain this code built. Both indices stay zero if the firmware failed
  // before it got as far as recording them, so they are reported as its claim, not as fact.
  // The firmware updates the chain command's state but not each sub-command's, so a
  // sub-command that never ran still reads 'new'.
  const char* failed_state = "unavailable";
  if (error_index < num_commands) {
    failed_state = ErtStateName(
        static_cast<volatile ert_start_kernel_cmd*>(cmd_bos[error_index].vaddr)->state);
  }

  log_warning_n(10,
                "AIE command chain of %zu failed: chain state '%s', device reports last submitted "
                "command %u and failing command %u (state '%s').\n",
                num_commands, ErtStateName(chain_cmd->state), submit_index, error_index,
                failed_state);
}

/// @brief Submits @p num_commands commands as one chain (or directly, if there is only one) and
/// waits for them to complete.
static hsa_status_t SubmitAndWaitChain(int fd, const void* heap_base, const BOHandle* cmd_bos,
                                       size_t num_commands, const std::vector<uint32_t>& bo_handles,
                                       KmqMetadata* kmq_metadata) {
  if (num_commands == 1) {
    // Single command: submit the per-kernel cmd BO directly, no chain wrapper.
    uint64_t seq = 0;
    hsa_status_t status =
        SubmitCommand(fd, cmd_bos[0].handle, bo_handles, kmq_metadata->hw_ctx_handle, seq);
    if (status != HSA_STATUS_SUCCESS) {
      assert(false && "Failed to submit command.");
      return status;
    }
    status = WaitCommand(fd, static_cast<ert_start_kernel_cmd*>(cmd_bos[0].vaddr),
                         kmq_metadata->hw_ctx_handle, kmq_metadata->syncobj_handle, seq);
    if (status != HSA_STATUS_SUCCESS) {
      log_warning_n(
          10, "AIE command failed: state '%s'.\n",
          ErtStateName(static_cast<volatile ert_start_kernel_cmd*>(cmd_bos[0].vaddr)->state));
      assert(false && "Failed waiting for command.");
      return status;
    }
    return HSA_STATUS_SUCCESS;
  }

  // Create command chain for multi-command dispatches.
  const size_t cmd_chain_data_bytesize = num_commands * sizeof(uint64_t);
  const size_t cmd_data_bytesize = sizeof(ert_cmd_chain_data) + cmd_chain_data_bytesize;
  const size_t cmd_bytesize = sizeof(ert_start_kernel_cmd) + cmd_data_bytesize;
  BOHandle cmd_bo_handle;
  hsa_status_t status = CreateCmdBO(fd, heap_base, cmd_bytesize, cmd_bo_handle);
  if (status != HSA_STATUS_SUCCESS) {
    assert(false && "Failed to create command chain BO.");
    return status;
  }
  MAKE_NAMED_SCOPE_GUARD(cmd_bo_handle_guard,
                         [&] { DestroyBOHandle(fd, heap_base, cmd_bo_handle); });

  auto* cmd = static_cast<ert_start_kernel_cmd*>(cmd_bo_handle.vaddr);
  memset(cmd, 0, cmd_bytesize);
  cmd->state = ERT_CMD_STATE_NEW;
  cmd->count = static_cast<uint32_t>(cmd_data_bytesize / sizeof(uint32_t));
  cmd->opcode = ERT_CMD_CHAIN;
  auto* cmd_chain = reinterpret_cast<ert_cmd_chain_data*>(cmd->data);
  cmd_chain->command_count = static_cast<uint32_t>(num_commands);
  for (size_t i = 0; i < num_commands; i++) {
    cmd_chain->data[i] = cmd_bos[i].handle;
  }

  // Execute all commands in the command chain.
  uint64_t seq = 0;
  status = SubmitCommand(fd, cmd_bo_handle.handle, bo_handles, kmq_metadata->hw_ctx_handle, seq);
  if (status != HSA_STATUS_SUCCESS) {
    assert(false && "Failed to submit command chain.");
    return status;
  }
  status = WaitCommand(fd, static_cast<ert_start_kernel_cmd*>(cmd_bo_handle.vaddr),
                       kmq_metadata->hw_ctx_handle, kmq_metadata->syncobj_handle, seq);
  if (status != HSA_STATUS_SUCCESS) {
    // Re-read through volatile: the firmware writes these while the wait above is blocked.
    LogChainFailure(static_cast<volatile ert_start_kernel_cmd*>(cmd_bo_handle.vaddr),
                    reinterpret_cast<volatile ert_cmd_chain_data*>(cmd->data), cmd_bos,
                    num_commands);
    assert(false && "Failed waiting for command chain.");
    return status;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::SubmitCmdChain(hsa_queue_t& q, void* queue_metadata,
                                        uint64_t first_pkt_idx, uint64_t num_pkts,
                                        uint32_t num_core_tiles, const core::Agent& agent) {
  auto kmq_metadata = static_cast<KmqMetadata*>(queue_metadata);

  auto* queue = static_cast<hsa_amd_aie_kernel_dispatch_packet_t*>(q.base_address);
  const uint64_t mask = q.size - 1;

  // Nothing to submit. Returning early matters because the mode below is read out of the first
  // packet: with no packets there is none to read, and committing a mode from an unwritten slot
  // would pin the queue to whatever happened to be there.
  if (num_pkts == 0) return HSA_STATUS_SUCCESS;

  // A queue is homogeneous: the first packet fixes the dispatch mode, because the two modes need
  // hardware contexts whose CU configuration cannot be reconciled after the fact.
  //
  // The mode is only committed to the queue once every packet has been accepted, so a malformed
  // submission cannot pin an otherwise-unused queue to a mode nothing ever ran in.
  const QueueMode mode = PacketMode(&queue[first_pkt_idx & mask]);
  if (kmq_metadata->mode == QueueMode::Undecided) {
    // Full ELF is an aie2p feature. This reports what the silicon supports, not the running
    // firmware: the firmware gate (management protocol minor >= 15) is not exposed through any
    // ioctl, so too-old firmware only shows up as EOPNOTSUPP when a command is submitted.
    const auto& aie_agent = static_cast<const AieAgent&>(agent);
    if (mode == QueueMode::FullElf &&
        DeviceTypeOf(aie_agent.properties().DeviceId) != XDNADeviceType::Stx) {
      return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
    }
  } else if (kmq_metadata->mode != mode) {
    log_warning_n(10, "AIE queue cannot mix full-ELF and PDI dispatches.\n");
    return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
  }

  // Instruction and arguments BOs (performance hint: up to 3 argument BOs per packet).
  std::vector<uint32_t> bo_handles;
  bo_handles.reserve(num_pkts * 4);

  // Commands to be submitted, and how many argument dwords each occupies in a chain slot.
  std::vector<BOHandle> cmd_bo_handles;
  std::vector<uint32_t> cmd_arg_cnts;
  cmd_bo_handles.reserve(num_pkts);
  cmd_arg_cnts.reserve(num_pkts);
  // Unmap and close the command BOs in case of an error.
  MAKE_NAMED_SCOPE_GUARD(cmd_bo_handles_guard, [&] {
    for (auto& bo_handle : cmd_bo_handles) {
      DestroyBOHandle(fd_, dev_heap_vaddr, bo_handle);
    }
  });

  // Flag to reconfigure the hardware context because of a new PDI.
  bool reconfigure_queue = false;

  // Building a command adds its PDI to the cache, but the hardware context is only reconfigured
  // to match once every packet has been built. If the batch fails in between, those entries would
  // claim compute units the context was never given, and the next submission would find them
  // cached, skip the reconfigure, and dispatch against a context that cannot run them. Roll them
  // back unless the cache and the context end up agreeing.
  const auto pdi_cache_watermark = kmq_metadata->pdi_cache.size();
  MAKE_NAMED_SCOPE_GUARD(pdi_cache_guard,
                         [&] { kmq_metadata->pdi_cache.Truncate(pdi_cache_watermark); });

  for (uint64_t i = 0; i < num_pkts; ++i) {
    const auto pkt_idx = (first_pkt_idx + i) & mask;
    auto* pkt = queue + pkt_idx;

    if (static_cast<hsa_amd_aie_packet_opcode_t>(pkt->opcode) != HSA_AMD_AIE_PACKET_OPCODE_KMQ ||
        PacketMode(pkt) != mode) {
      return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
    }

    BOHandle cmd_bo_handle;
    uint32_t arg_cnt = 0;
    const hsa_status_t err = (mode == QueueMode::FullElf)
        ? BuildFullElfCommand(fd_, dev_heap_vaddr, pkt, agent, bo_handles, &cmd_bo_handle, &arg_cnt)
        : BuildPdiInstsCommand(fd_, dev_heap_vaddr, pkt, agent, kmq_metadata, bo_handles,
                               &reconfigure_queue, &cmd_bo_handle, &arg_cnt);
    if (err != HSA_STATUS_SUCCESS) return err;

    cmd_bo_handles.push_back(cmd_bo_handle);
    cmd_arg_cnts.push_back(arg_cnt);
  }

  // Every packet was accepted, so the queue is committed to this mode from here on.
  kmq_metadata->mode = mode;

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

  // The context now matches the cache, so the new entries stand even if the submission below
  // fails: the commands they describe are still the ones this context can run.
  pdi_cache_guard.Dismiss();

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

  // Split into chains the driver's 4 KiB chain buffer can hold. The driver rejects an oversized
  // chain outright, so the split has to happen here; the commands are independent, so submitting
  // them as several chains back to back is equivalent to one long chain.
  size_t chunk_start = 0;
  while (chunk_start < cmd_bo_handles.size()) {
    // Every command in a chain shares one buffer, so take commands until the next one would not
    // fit -- the same accounting the driver does as it packs the slots.
    size_t chunk_len = 0;
    uint32_t chunk_bytesize = 0;
    for (size_t i = chunk_start; i < cmd_bo_handles.size(); ++i) {
      const uint32_t slot_bytesize = ChainSlotBytesize(cmd_arg_cnts[i]);
      if (chunk_bytesize + slot_bytesize > MAX_CHAIN_CMDBUF_SIZE) break;
      chunk_bytesize += slot_bytesize;
      ++chunk_len;
    }
    if (chunk_len == 0) {
      // A single command does not fit in the chain buffer at all. Nothing to split further.
      assert(false && "Command exceeds the driver's chain buffer.");
      return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
    }

    const hsa_status_t status = SubmitAndWaitChain(
        fd_, dev_heap_vaddr, &cmd_bo_handles[chunk_start], chunk_len, bo_handles, kmq_metadata);
    if (status != HSA_STATUS_SUCCESS) return status;

    chunk_start += chunk_len;
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

hsa_status_t XdnaDriver::GetDeviceFd(uint32_t node_id, int* fd) const {
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
                                            const HsaMemFlags* mem_flags, uint32_t num_nodes,
                                            const uint32_t* nodes) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t XdnaDriver::GetQueueSaveAreaInfo(HSA_QUEUEID queue_id, void** address,
                                              size_t* size) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t XdnaDriver::MakeMemoryUnresident(const void* mem) const { return HSA_STATUS_ERROR; }

hsa_status_t XdnaDriver::CheckAcceleratorReadiness(core::Agent& agent, bool* ready) const {
  (void)agent;
  (void)ready;
  return HSA_STATUS_ERROR;
}

}  // namespace AMD
}  // namespace rocr
