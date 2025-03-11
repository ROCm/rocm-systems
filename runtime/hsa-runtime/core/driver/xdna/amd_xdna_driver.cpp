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

XdnaDriver::XdnaDriver(std::string devnode_name)
    : core::Driver(core::DriverType::XDNA, devnode_name) {}

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
  return HSA_STATUS_SUCCESS;
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

  auto &aie_agent(static_cast<AieAgent &>(agent));

  amdxdna_drm_query_aie_metadata aie_metadata{0};
  amdxdna_drm_get_info get_info_args{
      .param = DRM_AMDXDNA_QUERY_AIE_METADATA,
      .buffer_size = sizeof(aie_metadata),
      .buffer = reinterpret_cast<uintptr_t>(&aie_metadata)};

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
  const MemoryRegion &m_region(static_cast<const MemoryRegion &>(mem_region));

  amdxdna_drm_create_bo create_bo_args{0};
  create_bo_args.size = size;

  amdxdna_drm_get_bo_info get_bo_info_args{0};
  drm_gem_close close_bo_args{0};
  void *mapped_mem(nullptr);

  if (!m_region.IsSystem()) {
    return HSA_STATUS_ERROR_INVALID_REGION;
  }

  const bool use_bo_shmem = !m_region.IsDeviceSVM();
  if (use_bo_shmem) {
    create_bo_args.type = AMDXDNA_BO_SHMEM;
  } else {
    create_bo_args.type = AMDXDNA_BO_DEV;
  }

  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_bo_args) < 0) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  get_bo_info_args.handle = create_bo_args.handle;
  // In case we need to close this BO to avoid leaks due to some error after
  // creation.
  close_bo_args.handle = create_bo_args.handle;

  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info_args) < 0) {
    // Close the BO in the case we can't get info about it.
    ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_bo_args);
    return HSA_STATUS_ERROR;
  }

  /// TODO: For now we always map the memory and keep a mapping from handles
  /// to VA memory addresses. Once we can support the separate VMEM call to
  /// map handles we can fix this.
  if (use_bo_shmem) {
    mapped_mem = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_,
                      get_bo_info_args.map_offset);
    if (mapped_mem == MAP_FAILED) {
      // Close the BO in the case when a mapping fails and we got a BO handle.
      ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_bo_args);
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
  } else {
    mapped_mem = reinterpret_cast<void *>(get_bo_info_args.vaddr);
  }

  if (alloc_flags & core::MemoryRegion::AllocateMemoryOnly) {
    *mem = reinterpret_cast<void *>(create_bo_args.handle);
  } else {
    *mem = mapped_mem;
  }

  vmem_handle_mappings.emplace(create_bo_args.handle, mapped_mem);
  vmem_addr_mappings.emplace(mapped_mem, create_bo_args.handle);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::FreeMemory(void *mem, size_t size) {
  auto it = vmem_addr_mappings.find(mem);
  if (it == vmem_addr_mappings.end()) return HSA_STATUS_ERROR_INVALID_ALLOCATION;

  auto handle = it->second;

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

  auto &aie_queue(static_cast<AieAqlQueue &>(queue));
  auto &aie_agent(aie_queue.GetAgent());

  // Currently we do not leverage QoS information.
  amdxdna_qos_info qos_info{0};
  amdxdna_drm_create_hwctx create_hwctx_args = {};
  create_hwctx_args.qos_p = reinterpret_cast<uintptr_t>(&qos_info);
  create_hwctx_args.max_opc = 0x800;
  create_hwctx_args.num_tiles = static_cast<uint32_t>(aie_agent.GetNumCores());

  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_CREATE_HWCTX, &create_hwctx_args) < 0) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  aie_queue.SetHwCtxHandle(create_hwctx_args.handle);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::DestroyQueue(core::Queue &queue) const {
  if (!AieAqlQueue::IsType(&queue)) {
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  auto &aie_queue(static_cast<AieAqlQueue &>(queue));
  amdxdna_drm_destroy_hwctx destroy_hwctx_args{.handle =
                                                   aie_queue.GetHwCtxHandle()};

  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_DESTROY_HWCTX, &destroy_hwctx_args) < 0) {
    return HSA_STATUS_ERROR;
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

  amdxdna_drm_get_bo_info get_bo_info_args{0};
  drm_gem_close close_bo_args{0};

  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_bo_args) < 0) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  get_bo_info_args.handle = create_bo_args.handle;
  // In case we need to close this BO to avoid leaks due to some error after
  // creation.
  close_bo_args.handle = create_bo_args.handle;

  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info_args) < 0) {
    // Close the BO in the case we can't get info about it.
    ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_bo_args);
    return HSA_STATUS_ERROR;
  }

  dev_heap_parent = mmap(0, dev_heap_align * 2 - 1, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

  if (dev_heap_parent == MAP_FAILED) {
    // Close the BO in the case when a mapping fails and we got a BO handle.
    ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_bo_args);
    dev_heap_parent = nullptr;
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  void *addr_aligned(reinterpret_cast<void *>(
      AlignUp(reinterpret_cast<uintptr_t>(dev_heap_parent), dev_heap_align)));

  dev_heap_aligned =
      mmap(addr_aligned, dev_heap_size, PROT_READ | PROT_WRITE,
           MAP_SHARED | MAP_FIXED, fd_, get_bo_info_args.map_offset);

  if (dev_heap_aligned == MAP_FAILED) {
    // Close the BO in the case when a mapping fails and we got a BO handle.
    ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_bo_args);
    // Unmap the dev_heap_parent.
    dev_heap_aligned = nullptr;
    FreeDeviceHeap();
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  return HSA_STATUS_SUCCESS;
}

std::unordered_map<uint32_t, void*>& XdnaDriver::GetHandleMappings() {
  return vmem_handle_mappings;
}

std::unordered_map<void*, uint32_t>& XdnaDriver::GetAddrMappings() { return vmem_addr_mappings; }

hsa_status_t XdnaDriver::FreeDeviceHeap() {
  if (dev_heap_parent) {
    if (munmap(dev_heap_parent, dev_heap_align * 2 - 1) != 0) return HSA_STATUS_ERROR;
    dev_heap_parent = nullptr;
  }

  if (dev_heap_aligned) {
    if (munmap(dev_heap_aligned, dev_heap_size) != 0) return HSA_STATUS_ERROR;
    dev_heap_aligned = nullptr;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::SyncBos(const std::vector<uint64_t>& bo_addrs,
                                 const std::vector<uint32_t>& bo_sizes) {
  if (bo_addrs.size() != bo_sizes.size()) return HSA_STATUS_ERROR;

  for (int i = 0; i < bo_addrs.size(); i++) {
    FlushCpuCache(reinterpret_cast<void*>(bo_addrs[i]), 0, bo_sizes[i]);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::ExecCmdAndWait(amdxdna_drm_exec_cmd* exec_cmd, uint32_t hw_ctx_handle) {
  // Submit the cmd
  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_EXEC_CMD, exec_cmd)) return HSA_STATUS_ERROR;

  // Waiting for command to finish
  amdxdna_drm_wait_cmd wait_cmd = {};
  wait_cmd.hwctx = hw_ctx_handle;
  wait_cmd.timeout = DEFAULT_TIMEOUT_VAL;
  wait_cmd.seq = exec_cmd->seq;

  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_WAIT_CMD, &wait_cmd)) return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::RegisterCmdBOs(
    uint32_t count, std::vector<uint32_t>& bo_args, std::vector<uint32_t>& bo_sizes,
    std::vector<uint64_t>& bo_addrs, hsa_amd_aie_ert_start_kernel_data_t* cmd_pkt_payload,
    const std::unordered_map<void*, uint32_t>& vmem_addr_mappings) {
  // This is the index where the operand addresses start in a command
  const int operand_starting_index = 5;

  // Counting the number of operands in the command payload.
  uint32_t num_operands = GetOperandCount(count);

  uint64_t instr_addr = Concat<uint64_t, uint32_t>(
      cmd_pkt_payload->data[CMD_PKT_PAYLOAD_INSTRUCTION_SEQUENCE_IDX + 1],
      cmd_pkt_payload->data[CMD_PKT_PAYLOAD_INSTRUCTION_SEQUENCE_IDX]);
  auto instr_handle = vmem_addr_mappings.find(reinterpret_cast<void*>(instr_addr));

  if (instr_handle == vmem_addr_mappings.end()) return HSA_STATUS_ERROR;

  // Keep track of the handles and addresses before we submit the packet
  bo_args.push_back(instr_handle->second);
  bo_addrs.push_back(instr_addr);

  // Adding the instruction sequence size. The packet contains the number of
  // instructions.
  uint32_t instr_bo_size =
      cmd_pkt_payload->data[CMD_PKT_PAYLOAD_INSTRUCTION_SEQUENCE_SIZE_IDX] * INSTR_SIZE_BYTES;
  bo_sizes.push_back(instr_bo_size);

  // Going through all of the operands in the command, keeping track of the
  // addresses and turning the addresses into handles. The starting index of
  // the operands in a command is `operand_starting_index` and the fields
  // are 32-bits we need to iterate over every two
  for (int operand_iter = 0; operand_iter < num_operands; operand_iter++) {
    uint32_t operand_index = operand_starting_index + 2 * operand_iter;
    uint64_t operand_addr = Concat<uint64_t, uint32_t>(cmd_pkt_payload->data[operand_index + 1],
                                                       cmd_pkt_payload->data[operand_index]);
    auto operand_handle = vmem_addr_mappings.find(reinterpret_cast<void*>(operand_addr));
    if (operand_handle == vmem_addr_mappings.end()) return HSA_STATUS_ERROR;
    bo_args.push_back(operand_handle->second);
    bo_addrs.push_back(operand_addr);
  }

  // Going through all of the operands in the command, keeping track of
  // the sizes of each operand. The size is used to sync the buffer
  uint32_t operand_size_starting_index = operand_starting_index + 2 * num_operands;
  for (int operand_iter = 0; operand_iter < num_operands; operand_iter++) {
    bo_sizes.push_back(cmd_pkt_payload->data[operand_size_starting_index + operand_iter]);
  }

  // Transform the instruction sequence address into device address
  cmd_pkt_payload->data[CMD_PKT_PAYLOAD_INSTRUCTION_SEQUENCE_IDX] =
      DEV_ADDR_BASE | instr_addr & DEV_ADDR_OFFSET_MASK;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::CreateCmd(uint32_t size, uint32_t* handle, amdxdna_cmd** cmd) {
  // Creating the command
  amdxdna_drm_create_bo create_cmd_bo = {};
  create_cmd_bo.type = AMDXDNA_BO_CMD, create_cmd_bo.size = size;
  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_cmd_bo)) return HSA_STATUS_ERROR;

  amdxdna_drm_get_bo_info cmd_bo_get_bo_info = {};
  cmd_bo_get_bo_info.handle = create_cmd_bo.handle;
  if (ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &cmd_bo_get_bo_info)) return HSA_STATUS_ERROR;

  *cmd = static_cast<amdxdna_cmd*>(mmap(nullptr, create_cmd_bo.size, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fd_, cmd_bo_get_bo_info.map_offset));

  if (cmd == MAP_FAILED) return HSA_STATUS_ERROR;

  *handle = create_cmd_bo.handle;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t XdnaDriver::SubmitCmdChain(hsa_amd_aie_ert_packet_t* first_pkt, uint32_t num_pkts,
                                        uint32_t num_operands, uint32_t hw_ctx_handle) {
  // Storing the metadata of the BOs that store the operands and metadata
  // of the commands we are going to submit
  std::vector<uint32_t> bo_args;
  std::vector<uint32_t> bo_sizes;
  std::vector<uint64_t> bo_addrs;
  bo_args.reserve(num_operands);
  bo_sizes.reserve(num_operands);
  bo_addrs.reserve(num_operands);

  // Storing the commands that we are going to submit and the
  // corresponding metadata
  std::vector<uint32_t> cmd_handles;
  std::vector<uint32_t> cmd_sizes;
  std::vector<amdxdna_cmd*> cmds;
  cmd_handles.reserve(num_pkts);
  cmd_sizes.reserve(num_pkts);
  cmds.reserve(num_pkts);

  // Iterating over all the contiguous HSA_AMD_AIE_ERT_CMD_CHAIN packets
  for (int pkt_iter = 0; pkt_iter < num_pkts; pkt_iter++) {
    // Getting the current command packet
    hsa_amd_aie_ert_packet_t* pkt = first_pkt + pkt_iter;
    hsa_amd_aie_ert_start_kernel_data_t* cmd_pkt_payload =
        reinterpret_cast<hsa_amd_aie_ert_start_kernel_data_t*>(pkt->payload_data);

    // Add the handles for all of the BOs to bo_args as well as rewrite
    // the command payload handles to contain the actual virtual addresses
    if (RegisterCmdBOs(pkt->count, bo_args, bo_sizes, bo_addrs, cmd_pkt_payload,
                       vmem_addr_mappings) != HSA_STATUS_SUCCESS)
      return HSA_STATUS_ERROR;

    // Creating a packet that contains the command to execute the kernel
    uint32_t cmd_bo_handle = 0;
    amdxdna_cmd* cmd = nullptr;
    uint32_t cmd_size = sizeof(amdxdna_cmd) + pkt->count * sizeof(uint32_t);
    if (CreateCmd(cmd_size, &cmd_bo_handle, &cmd)) return HSA_STATUS_ERROR;

    // Filling in the fields of the command
    cmd->state = pkt->state;
    cmd->extra_cu_masks = 0;

    // The driver places a structure before each command in a command chain.
    // Need to increase the size of the command by the size of this structure.
    cmd->count = pkt->count + CMD_COUNT_SIZE_INCREASE;
    cmd->opcode = pkt->opcode;
    cmd->data[0] = cmd_pkt_payload->cu_mask;
    memcpy((cmd->data + 1), cmd_pkt_payload->data, 4 * pkt->count);

    // Keeping track of the handle
    cmd_handles.push_back(cmd_bo_handle);
    cmds.push_back(cmd);
    cmd_sizes.push_back(cmd_size);
  }

  // Creating a packet that contains the command chain
  uint32_t cmd_chain_bo_handle = 0;
  amdxdna_cmd* cmd_chain = nullptr;
  int cmd_chain_size = (cmd_handles.size() + 1) * sizeof(uint32_t);
  if (CreateCmd(cmd_chain_size, &cmd_chain_bo_handle, &cmd_chain)) return HSA_STATUS_ERROR;

  // Writing information to the command buffer
  amdxdna_cmd_chain* cmd_chain_payload = reinterpret_cast<amdxdna_cmd_chain*>(cmd_chain->data);

  // Creating a command chain
  cmd_chain->state = HSA_AMD_AIE_ERT_STATE_NEW;
  cmd_chain->extra_cu_masks = 0;
  cmd_chain->count = sizeof(amdxdna_cmd_chain) + cmd_handles.size() * sizeof(uint64_t);
  cmd_chain->opcode = HSA_AMD_AIE_ERT_CMD_CHAIN;
  cmd_chain_payload->command_count = cmd_handles.size();
  cmd_chain_payload->submit_index = 0;
  cmd_chain_payload->error_index = 0;
  for (int i = 0; i < cmd_handles.size(); i++) {
    cmd_chain_payload->data[i] = cmd_handles[i];
  }

  // Syncing BOs before we execute the command
  if (SyncBos(bo_addrs, bo_sizes)) return HSA_STATUS_ERROR;

  // Removing duplicates in the bo container. The driver will report
  // an error if we provide the same BO handle multiple times.
  // This can happen if any of the BOs are the same across jobs
  std::sort(bo_args.begin(), bo_args.end());
  bo_args.erase(std::unique(bo_args.begin(), bo_args.end()), bo_args.end());

  // Filling in the fields to execute the command chain
  amdxdna_drm_exec_cmd exec_cmd_0 = {};
  exec_cmd_0.hwctx = hw_ctx_handle;
  exec_cmd_0.type = AMDXDNA_CMD_SUBMIT_EXEC_BUF;
  exec_cmd_0.cmd_handles = cmd_chain_bo_handle;
  exec_cmd_0.args = reinterpret_cast<uint64_t>(bo_args.data());
  exec_cmd_0.cmd_count = 1;
  exec_cmd_0.arg_count = bo_args.size();

  // Executing all commands in the command chain
  ExecCmdAndWait(&exec_cmd_0, hw_ctx_handle);

  // Unmapping and closing the cmd BOs
  drm_gem_close close_bo_args{0};
  for (int i = 0; i < cmd_handles.size(); i++) {
    if (munmap(cmds[i], cmd_sizes[i]) != 0) return HSA_STATUS_ERROR;
    close_bo_args.handle = cmd_handles[i];
    ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_bo_args);
  }

  // Unmapping and closing the cmd_chain BO
  if (munmap(cmd_chain, cmd_chain_size) != 0) return HSA_STATUS_ERROR;
  close_bo_args.handle = cmd_chain_bo_handle;
  ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close_bo_args);

  // Syncing BOs after we execute the command
  if (SyncBos(bo_addrs, bo_sizes)) return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
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

} // namespace AMD
} // namespace rocr
