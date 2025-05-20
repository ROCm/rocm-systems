////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
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

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <memory>
#include <string>

#include "core/inc/amd_aie_aql_queue.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/runtime.h"
#include "core/util/memory.h"
#include "core/util/utils.h"
#include "uapi/amdxdna_accel.h"

namespace rocr {
namespace AMD {

static_assert((sizeof(core::ShareableHandle::handle) >= sizeof(uint32_t)) &&
                  (alignof(core::ShareableHandle::handle) >= alignof(uint32_t)),
              "ShareableHandle cannot store a XDNA handle");

/// @brief Index of the first operand in a command.
///
/// Before the operands there are:
/// - 2 dwords for transaction op code
/// - 2 dwords for the instructions BO address
/// - 1 dword for the size of the instructions BO size
constexpr uint32_t operand_starting_index = 5;

/// @brief Default amdxdna_cu_config::cu_func when configuring a CU.
constexpr uint32_t default_cu_func = 0;

/// @brief Calculates the number of operands in a packet given the number of arguments in the
///        packet.
///
/// Each operand is 3 dwords (hi, lo address, and size). The op code is not counted in @p arg_count
/// but the instructions are.
///
/// @param arg_count number of arguments in the packet
/// @return number of operands in the packet
constexpr uint32_t GetOperandCount(uint32_t arg_count) { return (arg_count / 3) - 1; }

/// @brief Flushes operands.
static void FlushOperands(uint32_t count, hsa_amd_aie_ert_start_kernel_data_t* cmd_pkt_payload) {
  // Going through all of the operands in the command and flushing them.
  const uint32_t num_operands = GetOperandCount(count);
  for (uint32_t operand_iter = 0; operand_iter < num_operands; operand_iter++) {
    const uint32_t operand_index = operand_starting_index + 2 * operand_iter;
    const uint64_t operand_addr = Concat<uint64_t>(cmd_pkt_payload->data[operand_index + 1],
                                                   cmd_pkt_payload->data[operand_index]);
    const uint32_t operand_size_starting_index = operand_starting_index + 2 * num_operands;
    const uint32_t operand_bo_size =
        cmd_pkt_payload->data[operand_size_starting_index + operand_iter];
    FlushCpuCache(reinterpret_cast<void*>(operand_addr), 0, operand_bo_size);
  }
}

XdnaDriver::XdnaDriver(std::string devnode_name)
    : core::Driver(core::DriverType::XDNA, std::move(devnode_name)) {}

hsa_status_t XdnaDriver::DiscoverDriver(std::unique_ptr<core::Driver>& driver) {
  const int max_minor_num(64);
  static const std::string devnode_prefix("/dev/accel/accel");

  for (int i = 0; i < max_minor_num; ++i) {
    auto tmp_driver = std::unique_ptr<Driver>(new XdnaDriver(devnode_prefix + std::to_string(i)));
    if (tmp_driver->Open() == HSA_STATUS_SUCCESS) {
      if (tmp_driver->QueryKernelModeDriver(core::DriverQuery::GET_DRIVER_VERSION) ==
          HSA_STATUS_SUCCESS) {
        driver = std::move(tmp_driver);
        return HSA_STATUS_SUCCESS;
      } else {
        tmp_driver->Close();
      }
    }
  }

  return HSA_STATUS_ERROR;
}

uint64_t XdnaDriver::GetSystemMemoryByteSize() {
  const long pagesize = sysconf(_SC_PAGESIZE);
  const long page_count = sysconf(_SC_PHYS_PAGES);
  return pagesize * page_count;
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
  fd_ = open(devnode_name_.c_str(), O_RDWR | O_CLOEXEC);
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
  /// @todo XDNA driver currently only supports single-node AIE
  /// devices over PCIe. Update this once we can get topology
  /// information dynamically from the sysfs.
  node_props.NumNeuralCores = 1;
  node_props.NumIOLinks = 0;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::GetEdgeProperties(std::vector<HsaIoLinkProperties>& io_link_props,
                                           uint32_t node_id) const {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::GetAgentProperties(core::Agent &agent) const {
  if (agent.device_type() != core::Agent::DeviceType::kAmdAieDevice) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  auto& aie_agent = static_cast<AieAgent&>(agent);

  amdxdna_drm_query_aie_metadata aie_metadata = {};
  amdxdna_drm_get_info get_info_args = {};
  get_info_args.param = DRM_AMDXDNA_QUERY_AIE_METADATA;
  get_info_args.buffer_size = sizeof(aie_metadata);
  get_info_args.buffer = reinterpret_cast<uintptr_t>(&aie_metadata);

  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info_args) < 0) {
    return HSA_STATUS_ERROR;
  }

  // Right now can only target N-1 columns as that is the
  // number of shim DMAs in npu1 devices.
  aie_agent.SetNumCols(aie_metadata.cols - 1);
  aie_agent.SetNumCoreRows(aie_metadata.core.row_count);

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
  const bool use_bo_shmem = !m_region.IsDeviceSVM();
  if (use_bo_shmem) {
    create_bo_args.type = AMDXDNA_BO_SHMEM;
  } else {
    create_bo_args.type = AMDXDNA_BO_DEV;
  }

  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_bo_args) < 0) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  BOHandle bo_handle;
  bo_handle.handle = create_bo_args.handle;
  bo_handle.size = size;

  // Close the BO in case of error.
  MAKE_NAMED_SCOPE_GUARD(bo_guard, [&] { DestroyBOHandle(bo_handle); });

  amdxdna_drm_get_bo_info get_bo_info_args = {};
  get_bo_info_args.handle = create_bo_args.handle;
  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info_args) < 0) {
    return HSA_STATUS_ERROR;
  }

  /// TODO: For now we always map the memory and keep a mapping from handles
  /// to VA memory addresses. Once we can support the separate VMEM call to
  /// map handles we can fix this.
  if (use_bo_shmem) {
    bo_handle.vaddr =
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, get_bo_info_args.map_offset);
    if (bo_handle.vaddr == MAP_FAILED) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
  } else {
    bo_handle.vaddr = reinterpret_cast<void*>(get_bo_info_args.vaddr);
  }

  if (alloc_flags & core::MemoryRegion::AllocateMemoryOnly) {
    *mem = reinterpret_cast<void *>(create_bo_args.handle);
  } else {
    *mem = bo_handle.vaddr;
  }

  vmem_handle_mappings.emplace(bo_handle.handle, bo_handle.vaddr);
  vmem_addr_mappings.emplace(bo_handle.vaddr, bo_handle);

  bo_guard.Dismiss();

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::FreeMemory(void *mem, size_t size) {
  auto it = vmem_addr_mappings.find(mem);
  if (it == vmem_addr_mappings.end()) return HSA_STATUS_ERROR_INVALID_ALLOCATION;

  auto handle = it->second.handle;

  drm_gem_close close_args = {};
  close_args.handle = handle;
  if (ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_args) < 0) {
    return HSA_STATUS_ERROR;
  }

  vmem_handle_mappings.erase(handle);
  vmem_addr_mappings.erase(it);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::CreateQueue(core::Queue &queue) const {
  if (!AieAqlQueue::IsType(&queue)) {
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  // Set the hw ctx handle of the queue to invalid to avoid incorrect destruction.
  auto& aie_queue = static_cast<AieAqlQueue&>(queue);
  aie_queue.SetHwCtxHandle(AMDXDNA_INVALID_BO_HANDLE);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::DestroyQueue(core::Queue &queue) const {
  if (!AieAqlQueue::IsType(&queue)) {
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  auto& aie_queue = static_cast<AieAqlQueue&>(queue);
  if (aie_queue.GetHwCtxHandle() != AMDXDNA_INVALID_BO_HANDLE) {
    amdxdna_drm_destroy_hwctx destroy_hwctx_args = {};
    destroy_hwctx_args.handle = aie_queue.GetHwCtxHandle();
    if (ioctl(fd_, DRM_IOCTL_AMDXDNA_DESTROY_HWCTX, &destroy_hwctx_args) < 0) {
      return HSA_STATUS_ERROR;
    }
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::ExportDMABuf(void *mem, size_t size, int *dmabuf_fd,
                                      size_t *offset) {
  // Not implemented yet.
  return HSA_STATUS_ERROR;
}

hsa_status_t XdnaDriver::ImportDMABuf(int dmabuf_fd, core::Agent &agent,
                                      core::ShareableHandle &handle) {
  drm_prime_handle import_params = {};
  import_params.handle = AMDXDNA_INVALID_BO_HANDLE;
  import_params.fd = dmabuf_fd;
  if (ioctl(fd_, DRM_IOCTL_PRIME_FD_TO_HANDLE, &import_params) < 0)
    return HSA_STATUS_ERROR;

  handle.handle = import_params.handle;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::Map(core::ShareableHandle handle, void *mem,
                             size_t offset, size_t size,
                             hsa_access_permission_t perms) {
  // Get fd associated with the handle.
  drm_prime_handle params = {};
  params.handle = handle.handle;
  params.fd = -1;
  if (ioctl(fd_, DRM_IOCTL_PRIME_HANDLE_TO_FD, &params) < 0)
    return HSA_STATUS_ERROR;

  // Change permissions.
  void *mapped_ptr = mmap(mem, size, PermissionsToMmapFlags(perms),
                          MAP_FIXED | MAP_SHARED, params.fd, offset);
  if (mapped_ptr == MAP_FAILED)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::Unmap(core::ShareableHandle handle, void *mem,
                               size_t offset, size_t size) {
  if (munmap(mem, size) != 0)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::ReleaseShareableHandle(core::ShareableHandle &handle) {
  drm_gem_close close_params = {};
  close_params.handle = handle.handle;
  if (ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_params) < 0)
    return HSA_STATUS_ERROR;

  handle = {};

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::QueryDriverVersion() {
  amdxdna_drm_query_aie_version aie_version{0, 0};
  amdxdna_drm_get_info args{DRM_AMDXDNA_QUERY_AIE_VERSION, sizeof(aie_version),
                            reinterpret_cast<uintptr_t>(&aie_version)};

  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_INFO, &args) < 0) {
    return HSA_STATUS_ERROR;
  }

  version_.KernelInterfaceMajorVersion = aie_version.major;
  version_.KernelInterfaceMinorVersion = aie_version.minor;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::InitDeviceHeap() {
  amdxdna_drm_create_bo create_bo_args = {};
  create_bo_args.size = dev_heap_size;
  create_bo_args.type = AMDXDNA_BO_DEV_HEAP;
  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_bo_args) < 0) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  dev_heap_handle.handle = create_bo_args.handle;

  // Unmap memory and close the BO in case of error.
  MAKE_NAMED_SCOPE_GUARD(dev_heap_handle_guard, [&] { DestroyBOHandle(dev_heap_handle); });

  amdxdna_drm_get_bo_info get_bo_info_args = {};
  get_bo_info_args.handle = dev_heap_handle.handle;
  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info_args) < 0) {
    return HSA_STATUS_ERROR;
  }

  const size_t size = dev_heap_align * 2 - 1;
  dev_heap_handle.vaddr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (dev_heap_handle.vaddr == MAP_FAILED) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
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
  hsa_status_t status = HSA_STATUS_SUCCESS;

  if (dev_heap_aligned) {
    if (munmap(dev_heap_aligned, dev_heap_size) != 0) {
      status = HSA_STATUS_ERROR;
    }
    dev_heap_aligned = nullptr;
  }

  if (dev_heap_handle.IsValid()) {
    if (munmap(dev_heap_handle.vaddr, dev_heap_handle.size) != 0) {
      status = HSA_STATUS_ERROR;
    }
    drm_gem_close close_bo_args = {};
    close_bo_args.handle = dev_heap_handle.handle;
    ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_bo_args);
    dev_heap_handle = BOHandle{};
  }

  return status;
}

hsa_status_t XdnaDriver::ExecCmdAndWait(const BOHandle& cmd_chain_bo_handle,
                                        const std::vector<uint32_t>& bo_handles,
                                        AieAqlQueue& aie_queue) {
  // Submit command chain.
  amdxdna_drm_exec_cmd exec_cmd = {};
  exec_cmd.hwctx = aie_queue.GetHwCtxHandle();
  exec_cmd.type = AMDXDNA_CMD_SUBMIT_EXEC_BUF;
  exec_cmd.cmd_handles = cmd_chain_bo_handle.handle;
  exec_cmd.args = reinterpret_cast<uint64_t>(bo_handles.data());
  exec_cmd.cmd_count = 1;
  exec_cmd.arg_count = bo_handles.size();

  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_EXEC_CMD, &exec_cmd) < 0) return HSA_STATUS_ERROR;

  // Waiting for command chain to finish.
  amdxdna_drm_wait_cmd wait_cmd = {};
  wait_cmd.hwctx = aie_queue.GetHwCtxHandle();
  wait_cmd.timeout = DEFAULT_TIMEOUT_VAL;
  wait_cmd.seq = exec_cmd.seq;

  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_WAIT_CMD, &wait_cmd) < 0) return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::PrepareBOs(uint32_t count,
                                    hsa_amd_aie_ert_start_kernel_data_t* cmd_pkt_payload,
                                    std::vector<uint32_t>& bo_handles) {
  const uint64_t instr_addr =
      Concat<uint64_t>(cmd_pkt_payload->data[CMD_PKT_PAYLOAD_INSTRUCTION_SEQUENCE_IDX + 1],
                       cmd_pkt_payload->data[CMD_PKT_PAYLOAD_INSTRUCTION_SEQUENCE_IDX]);
  auto instr_bo_handle = FindBOHandle(reinterpret_cast<void*>(instr_addr));
  if (!instr_bo_handle.IsValid()) {
    return HSA_STATUS_ERROR;
  }

  // Keep track of the instruction sequence BO.
  bo_handles.push_back(instr_bo_handle.handle);

  // Flush the instruction sequence. The packet contains the number of instructions.
  const uint32_t instr_bo_size =
      cmd_pkt_payload->data[CMD_PKT_PAYLOAD_INSTRUCTION_SEQUENCE_SIZE_IDX] * INSTR_SIZE_BYTES;
  FlushCpuCache(reinterpret_cast<void*>(instr_addr), 0, instr_bo_size);

  // Going through all of the operands in the command, keeping track of the
  // addresses and turning the addresses into handles. The starting index of
  // the operands in a command is `operand_starting_index` and the fields
  // are 32-bits we need to iterate over every two
  const uint32_t num_operands = GetOperandCount(count);
  bo_handles.reserve(num_operands);
  for (uint32_t operand_iter = 0; operand_iter < num_operands; operand_iter++) {
    const uint32_t operand_index = operand_starting_index + 2 * operand_iter;
    const uint64_t operand_addr = Concat<uint64_t>(cmd_pkt_payload->data[operand_index + 1],
                                                   cmd_pkt_payload->data[operand_index]);
    auto operand_bo_handle = FindBOHandle(reinterpret_cast<void*>(operand_addr));
    if (!operand_bo_handle.IsValid()) {
      return HSA_STATUS_ERROR;
    }

    // Keep track of the operand BO.
    bo_handles.push_back(operand_bo_handle.handle);

    // Flush the operand.
    const uint32_t operand_size_starting_index = operand_starting_index + 2 * num_operands;
    const uint32_t operand_bo_size =
        cmd_pkt_payload->data[operand_size_starting_index + operand_iter];
    FlushCpuCache(reinterpret_cast<void*>(operand_addr), 0, operand_bo_size);
  }

  // Transform the instruction sequence address into device address
  cmd_pkt_payload->data[CMD_PKT_PAYLOAD_INSTRUCTION_SEQUENCE_IDX] =
      DEV_ADDR_BASE | (instr_addr & DEV_ADDR_OFFSET_MASK);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::CreateCmdBO(uint32_t size, BOHandle& cmd_bo_handle) {
  amdxdna_drm_create_bo create_cmd_bo = {};
  create_cmd_bo.type = AMDXDNA_BO_CMD;
  create_cmd_bo.size = size;
  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_cmd_bo) < 0) {
    return HSA_STATUS_ERROR;
  }

  // Close the BO in case of error.
  MAKE_NAMED_SCOPE_GUARD(cmd_bo_handle_guard, [&] {
    drm_gem_close close_bo_args = {};
    close_bo_args.handle = create_cmd_bo.handle;
    ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_bo_args);
  });

  amdxdna_drm_get_bo_info cmd_bo_get_bo_info = {};
  cmd_bo_get_bo_info.handle = create_cmd_bo.handle;
  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &cmd_bo_get_bo_info) < 0) {
    return HSA_STATUS_ERROR;
  }

  void* mem = static_cast<amdxdna_cmd*>(mmap(nullptr, create_cmd_bo.size, PROT_READ | PROT_WRITE,
                                             MAP_SHARED, fd_, cmd_bo_get_bo_info.map_offset));
  if (mem == MAP_FAILED) {
    return HSA_STATUS_ERROR;
  }

  cmd_bo_handle = BOHandle{mem, create_cmd_bo.handle, size};

  cmd_bo_handle_guard.Dismiss();

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::SubmitCmdChain(hsa_amd_aie_ert_packet_t* first_pkt, uint32_t num_pkts,
                                        AieAqlQueue& aie_queue) {
  // Stores instruction and operand BOs.
  std::vector<uint32_t> bo_handles;

  // Stores commands that we are going to submit and the corresponding metadata.
  std::vector<BOHandle> cmd_bo_handles;
  cmd_bo_handles.reserve(num_pkts);
  // Unmap and close the command BOs in case of an error.
  MAKE_NAMED_SCOPE_GUARD(cmd_bo_handles_guard, [&] {
    for (auto& bo_handle : cmd_bo_handles) {
      DestroyBOHandle(bo_handle);
    }
  });

  // PDI cache. If the cache is updated, a new hardware context will be created for the queue.
  auto pdi_cache_it = hw_ctx_pdi_cache_map.find(aie_queue.GetHwCtxHandle());
  auto pdi_cache = (pdi_cache_it != hw_ctx_pdi_cache_map.end()) ? pdi_cache_it->second : PDICache{};
  bool reconfigure_queue = false;

  // Iterating over all the contiguous HSA_AMD_AIE_ERT_CMD_CHAIN packets
  for (uint32_t pkt_iter = 0; pkt_iter < num_pkts; pkt_iter++) {
    // Getting the current command packet
    hsa_amd_aie_ert_packet_t* pkt = first_pkt + pkt_iter;
    hsa_amd_aie_ert_start_kernel_data_t* cmd_pkt_payload =
        reinterpret_cast<hsa_amd_aie_ert_start_kernel_data_t*>(pkt->payload_data);

    // Add the handles for all of the BOs to bo_handles as well as rewrite
    // the instruction handle to contain the device address
    hsa_status_t status = PrepareBOs(pkt->count, cmd_pkt_payload, bo_handles);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }

    // Creating a packet that contains the command to execute the kernel
    const uint32_t cmd_size = sizeof(amdxdna_cmd) + pkt->count * sizeof(uint32_t);
    BOHandle cmd_bo_handle;
    status = CreateCmdBO(cmd_size, cmd_bo_handle);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
    // Unmap and close the command BO in case of an error.
    MAKE_NAMED_SCOPE_GUARD(cmd_bo_handle_guard, [&] { DestroyBOHandle(cmd_bo_handle); });

    auto* cmd = static_cast<amdxdna_cmd*>(cmd_bo_handle.vaddr);

    // Filling in the fields of the command
    cmd->state = pkt->state;
    cmd->extra_cu_masks = 0;

    // The driver places a structure before each command in a command chain.
    // Need to increase the size of the command by the size of this structure.
    cmd->count = pkt->count + CMD_COUNT_SIZE_INCREASE;
    cmd->opcode = pkt->opcode;

    // Find if the PDI is cached in the queues PDI cache. If even one PDI is not found, the hardware
    // context will need to be reconfigured and the cache updated.
    auto pdi_bo_handle = FindBOHandle(cmd_pkt_payload->pdi_addr);
    if (!pdi_bo_handle.IsValid()) return HSA_STATUS_ERROR_INVALID_ALLOCATION;

    // Determine if the PDI is cached, if not it will be added to the PDI cache.
    auto cached_pdi_index = pdi_cache.GetIndex(pdi_bo_handle.handle);
    if (cached_pdi_index == PDICache::NotFound) {
      FlushCpuCache(pdi_bo_handle.vaddr, 0, pdi_bo_handle.size);
      status = pdi_cache.SetNext(pdi_bo_handle, cached_pdi_index);
      if (status != HSA_STATUS_SUCCESS) {
        return status;
      }
      reconfigure_queue = true;
    }

    cmd->data[0] = 0x1 << static_cast<uint32_t>(cached_pdi_index);
    memcpy((cmd->data + 1), cmd_pkt_payload->data, 4 * pkt->count);

    // Keeping track of the command
    cmd_bo_handles.push_back(cmd_bo_handle);
    cmd_bo_handle_guard.Dismiss();
  }

  // If there were PDIs that were not cached, the hardware context needs to be reconfigured.
  // The cache map will be update with the new hardware context.
  if (reconfigure_queue) {
    if (pdi_cache_it != hw_ctx_pdi_cache_map.end()) {
      hw_ctx_pdi_cache_map.erase(pdi_cache_it);
    }

    hsa_status_t status = ConfigHwCtx(pdi_cache, aie_queue);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }

    // Update cache mapping.
    hw_ctx_pdi_cache_map.emplace(aie_queue.GetHwCtxHandle(), pdi_cache);
  }

  // Creating a packet that contains the command chain
  const uint32_t cmd_chain_size = (cmd_bo_handles.size() + 1) * sizeof(uint32_t);
  BOHandle cmd_chain_bo_handle;
  hsa_status_t status = CreateCmdBO(cmd_chain_size, cmd_chain_bo_handle);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  // Unmap and close the command chain BO in case of an error.
  MAKE_NAMED_SCOPE_GUARD(cmd_chain_bo_handle_guard, [&] { DestroyBOHandle(cmd_chain_bo_handle); });

  auto* cmd_chain = static_cast<amdxdna_cmd*>(cmd_chain_bo_handle.vaddr);

  // Writing information to the command buffer
  amdxdna_cmd_chain* cmd_chain_payload = reinterpret_cast<amdxdna_cmd_chain*>(cmd_chain->data);

  // Creating a command chain
  cmd_chain->state = HSA_AMD_AIE_ERT_STATE_NEW;
  cmd_chain->extra_cu_masks = 0;
  cmd_chain->count = sizeof(amdxdna_cmd_chain) + cmd_bo_handles.size() * sizeof(uint64_t);
  cmd_chain->opcode = HSA_AMD_AIE_ERT_CMD_CHAIN;
  cmd_chain_payload->command_count = cmd_bo_handles.size();
  cmd_chain_payload->submit_index = 0;
  cmd_chain_payload->error_index = 0;
  for (size_t i = 0; i < cmd_bo_handles.size(); i++) {
    cmd_chain_payload->data[i] = cmd_bo_handles[i].handle;
  }

  // Removing duplicates in the bo container. The driver will report
  // an error if we provide the same BO handle multiple times.
  // This can happen if any of the BOs are the same across jobs
  std::sort(bo_handles.begin(), bo_handles.end());
  bo_handles.erase(std::unique(bo_handles.begin(), bo_handles.end()), bo_handles.end());

  // Executing all commands in the command chain
  status = ExecCmdAndWait(cmd_chain_bo_handle, bo_handles, aie_queue);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  for (uint32_t pkt_iter = 0; pkt_iter < num_pkts; pkt_iter++) {
    hsa_amd_aie_ert_packet_t* pkt = first_pkt + pkt_iter;
    auto* cmd_pkt_payload =
        reinterpret_cast<hsa_amd_aie_ert_start_kernel_data_t*>(pkt->payload_data);
    FlushOperands(pkt->count, cmd_pkt_payload);
  }

  // Unmapping and closing the cmd BOs
  cmd_bo_handles_guard.Dismiss();
  for (auto& command_bo_handle : cmd_bo_handles) {
    if (munmap(command_bo_handle.vaddr, command_bo_handle.size) != 0) {
      status = HSA_STATUS_ERROR;
    }
    drm_gem_close close_bo_args = {};
    close_bo_args.handle = command_bo_handle.handle;
    ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_bo_args);
  }

  // Unmapping and closing the cmd_chain BO
  cmd_chain_bo_handle_guard.Dismiss();
  if (munmap(cmd_chain, cmd_chain_size) != 0) {
    status = HSA_STATUS_ERROR;
  }
  drm_gem_close close_bo_args = {};
  close_bo_args.handle = cmd_chain_bo_handle.handle;
  ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_bo_args);

  return status;
}

hsa_status_t XdnaDriver::SPMAcquire(uint32_t preferred_node_id) const {
  // AIE does not support streaming performance monitor.
  return HSA_STATUS_ERROR_INVALID_AGENT;
}

hsa_status_t XdnaDriver::SPMRelease(uint32_t preferred_node_id) const {
  // AIE does not support streaming performance monitor.
  return HSA_STATUS_ERROR_INVALID_AGENT;
};

hsa_status_t XdnaDriver::SPMSetDestBuffer(uint32_t preferred_node_id, uint32_t size_bytes,
                                          uint32_t* timeout, uint32_t* size_copied,
                                          void* dest_mem_addr, bool* is_spm_data_loss) const {
  // AIE does not support streaming performance monitor.
  return HSA_STATUS_ERROR_INVALID_AGENT;
}

hsa_status_t XdnaDriver::IsModelEnabled(bool* enable) const {
  // AIE does not support streaming performance monitor.
  *enable = false;
  return HSA_STATUS_SUCCESS;
}

void XdnaDriver::DestroyBOHandle(BOHandle& handle) {
  munmap(handle.vaddr, handle.size);
  drm_gem_close close_bo_args = {};
  close_bo_args.handle = handle.handle;
  ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_bo_args);
  handle = {};
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

hsa_status_t XdnaDriver::ConfigHwCtx(const PDICache& pdi_bo_handles, AieAqlQueue& aie_queue) {
  const size_t config_cu_param_size =
      sizeof(amdxdna_hwctx_param_config_cu) + pdi_bo_handles.size() * sizeof(amdxdna_cu_config);

  auto* xdna_config_cu_param =
      static_cast<amdxdna_hwctx_param_config_cu*>(malloc(config_cu_param_size));
  if (xdna_config_cu_param == nullptr) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  MAKE_SCOPE_GUARD([xdna_config_cu_param] { free(xdna_config_cu_param); });

  xdna_config_cu_param->num_cus = pdi_bo_handles.size();

  for (size_t i = 0; i < pdi_bo_handles.size(); i++) {
    xdna_config_cu_param->cu_configs[i].cu_bo = pdi_bo_handles[i].handle;
    xdna_config_cu_param->cu_configs[i].cu_func = default_cu_func;
  }

  if (aie_queue.GetHwCtxHandle() != AMDXDNA_INVALID_BO_HANDLE) {
    // Destroy the hardware context
    // Note: we can do this because we have forced synchronization between
    // command chains. If we move to a more asynchronous model, we will need to
    // figure out how hardware context destruction works while applications
    // are running
    amdxdna_drm_destroy_hwctx destroy_hwctx_args = {};
    destroy_hwctx_args.handle = aie_queue.GetHwCtxHandle();
    if (ioctl(fd_, DRM_IOCTL_AMDXDNA_DESTROY_HWCTX, &destroy_hwctx_args) < 0) {
      return HSA_STATUS_ERROR;
    }
  }

  // Create the new hardware context
  // Currently we do not leverage QoS information.
  amdxdna_qos_info qos_info = {};
  amdxdna_drm_create_hwctx create_hwctx_args = {};
  create_hwctx_args.qos_p = reinterpret_cast<uintptr_t>(&qos_info);
  create_hwctx_args.max_opc = 0x800;
  create_hwctx_args.num_tiles = aie_queue.GetAgent().GetNumCores();

  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_CREATE_HWCTX, &create_hwctx_args) < 0) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  // Configure the new hardware context
  amdxdna_drm_config_hwctx config_hw_ctx_args = {};
  config_hw_ctx_args.handle = create_hwctx_args.handle;
  config_hw_ctx_args.param_type = DRM_AMDXDNA_HWCTX_CONFIG_CU;
  config_hw_ctx_args.param_val = reinterpret_cast<uint64_t>(xdna_config_cu_param);
  config_hw_ctx_args.param_val_size = static_cast<uint32_t>(config_cu_param_size);

  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, &config_hw_ctx_args) < 0) {
    return HSA_STATUS_ERROR;
  }

  aie_queue.SetHwCtxHandle(create_hwctx_args.handle);

  return HSA_STATUS_SUCCESS;
}

} // namespace AMD
} // namespace rocr
