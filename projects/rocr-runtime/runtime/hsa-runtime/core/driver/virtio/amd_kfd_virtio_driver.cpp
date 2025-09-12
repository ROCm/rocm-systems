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

#include "core/inc/amd_virtio_driver.h"
#include "hsakmt/hsakmt_virtio.h"

#include <link.h>
#include <vector>

#include "core/inc/amd_gpu_agent.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/runtime.h"

extern r_debug _amdgpu_r_debug;

namespace rocr {
namespace AMD {

KfdVirtioDriver::KfdVirtioDriver(std::string devnode_name)
    : core::Driver(core::DriverType::KFD_VIRTIO, std::move(devnode_name)) {}

hsa_status_t KfdVirtioDriver::DiscoverDriver(std::unique_ptr<core::Driver>& driver) {
  auto tmp_driver = std::unique_ptr<core::Driver>(new KfdVirtioDriver(""));

  if (tmp_driver->Open() == HSA_STATUS_SUCCESS) {
    driver = std::move(tmp_driver);
    return HSA_STATUS_SUCCESS;
  }

  return HSA_STATUS_ERROR;
}

hsa_status_t KfdVirtioDriver::Open() {
  return vhsaKmtOpenKFD() == HSAKMT_STATUS_SUCCESS ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

hsa_status_t KfdVirtioDriver::Close() {
  return vhsaKmtCloseKFD() == HSAKMT_STATUS_SUCCESS ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

hsa_status_t KfdVirtioDriver::Init() {
  HSAKMT_STATUS ret =
      vhsaKmtRuntimeEnable(&_amdgpu_r_debug, core::Runtime::runtime_singleton_->flag().debug());
  uint32_t caps_mask = 0;

  if (ret != HSAKMT_STATUS_SUCCESS && ret != HSAKMT_STATUS_NOT_SUPPORTED) return HSA_STATUS_ERROR;

  if (vhsaKmtGetRuntimeCapabilities(&caps_mask) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  core::Runtime::runtime_singleton_->KfdVersion(
      ret != HSAKMT_STATUS_NOT_SUPPORTED,
      !!(caps_mask & HSA_RUNTIME_ENABLE_CAPS_SUPPORTS_CORE_DUMP_MASK));

  if (vhsaKmtGetVersion(&version_) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  core::Runtime::runtime_singleton_->KfdVersion(version_);

  if (version_.KernelInterfaceMajorVersion == 1 && version_.KernelInterfaceMinorVersion == 0)
    core::g_use_interrupt_wait = false;

  /* Force disable interrupt wait in VIRTIO driver temporarily */
  core::g_use_interrupt_wait = false;

  /* Force disable XNACK in VIRTIO driver temporarily */
  core::Runtime::runtime_singleton_->XnackEnabled(false);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::ShutDown() {
  HSAKMT_STATUS ret = vhsaKmtRuntimeDisable();
  if (ret != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  ret = vhsaKmtReleaseSystemProperties();

  if (ret != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  return Close();
}

hsa_status_t KfdVirtioDriver::QueryKernelModeDriver(core::DriverQuery query) {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::GetSystemProperties(HsaSystemProperties& sys_props) const {
  if (vhsaKmtAcquireSystemProperties(&sys_props) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::GetNodeProperties(HsaNodeProperties& node_props,
                                                uint32_t node_id) const {
  if (vhsaKmtGetNodeProperties(node_id, &node_props) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::GetEdgeProperties(std::vector<HsaIoLinkProperties>& io_link_props,
                                                uint32_t node_id) const {
  if (vhsaKmtGetNodeIoLinkProperties(node_id, io_link_props.size(), io_link_props.data()) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::GetMemoryProperties(
    uint32_t node_id, std::vector<HsaMemoryProperties>& mem_props) const {
  if (mem_props.empty()) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (vhsaKmtGetNodeMemoryProperties(node_id, mem_props.size(), mem_props.data()) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::GetCacheProperties(
    uint32_t node_id, uint32_t processor_id, std::vector<HsaCacheProperties>& cache_props) const {
  if (cache_props.empty()) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (vhsaKmtGetNodeCacheProperties(node_id, 0, cache_props.size(), cache_props.data()) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::GetDeviceHandle(uint32_t node_id, void** device_handle) const {
  assert(device_handle != nullptr);

  if (vhsaKmtGetAMDGPUDeviceHandle(node_id, device_handle) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::GetClockCounters(uint32_t node_id,
                                               HsaClockCounters* clock_counter) const {
  assert(clock_counter != nullptr);

  if (vhsaKmtGetClockCounters(node_id, clock_counter) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::SetTrapHandler(uint32_t node_id, const void* base, uint64_t base_size,
                                             const void* buffer_base,
                                             uint64_t buffer_base_size) const {
  if (vhsaKmtSetTrapHandler(node_id, const_cast<void*>(base), base_size,
                            const_cast<void*>(buffer_base),
                            buffer_base_size) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::AllocateMemory(const core::MemoryRegion& mem_region,
                                             core::MemoryRegion::AllocateFlags alloc_flags,
                                             void** mem, size_t size, uint32_t agent_node_id) {
  const MemoryRegion& m_region(static_cast<const MemoryRegion&>(mem_region));
  HsaMemFlags kmt_alloc_flags(m_region.mem_flags());
  HSAKMT_STATUS ret;

  kmt_alloc_flags.ui32.ExecuteAccess =
      (alloc_flags & core::MemoryRegion::AllocateExecutable ? 1 : 0);
  kmt_alloc_flags.ui32.AQLQueueMemory =
      (alloc_flags & core::MemoryRegion::AllocateDoubleMap ? 1 : 0);

  if (m_region.IsSystem() && (alloc_flags & core::MemoryRegion::AllocateNonPaged)) {
    kmt_alloc_flags.ui32.NonPaged = 1;
  }

  if (!m_region.IsLocalMemory() && (alloc_flags & core::MemoryRegion::AllocateMemoryOnly)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Allocating a memory handle for virtual memory
  kmt_alloc_flags.ui32.NoAddress = !!(alloc_flags & core::MemoryRegion::AllocateMemoryOnly);

  // Allocate pseudo fine grain memory
  kmt_alloc_flags.ui32.CoarseGrain =
      (alloc_flags & core::MemoryRegion::AllocatePCIeRW ? 0 : kmt_alloc_flags.ui32.CoarseGrain);

  kmt_alloc_flags.ui32.NoSubstitute =
      (alloc_flags & core::MemoryRegion::AllocatePinned ? 1 : kmt_alloc_flags.ui32.NoSubstitute);

  kmt_alloc_flags.ui32.GTTAccess =
      (alloc_flags & core::MemoryRegion::AllocateGTTAccess ? 1 : kmt_alloc_flags.ui32.GTTAccess);

  kmt_alloc_flags.ui32.Uncached =
      (alloc_flags & core::MemoryRegion::AllocateUncached ? 1 : kmt_alloc_flags.ui32.Uncached);

  if (m_region.IsLocalMemory()) {
    // Allocate physically contiguous memory. AllocateKfdMemory function call
    // will fail if this flag is not supported in KFD.
    kmt_alloc_flags.ui32.Contiguous =
        (alloc_flags & core::MemoryRegion::AllocateContiguous ? 1
                                                              : kmt_alloc_flags.ui32.Contiguous);
  }

  //// Only allow using the suballocator for ordinary VRAM.
  if (m_region.IsLocalMemory() && !kmt_alloc_flags.ui32.NoAddress) {
    bool subAllocEnabled = !core::Runtime::runtime_singleton_->flag().disable_fragment_alloc();
    // Avoid modifying executable or queue allocations.
    bool useSubAlloc = subAllocEnabled;
    useSubAlloc &= ((alloc_flags & (~core::MemoryRegion::AllocateRestrict)) == 0);

    if (useSubAlloc) {
      *mem = m_region.fragment_alloc(size);

      if ((alloc_flags & core::MemoryRegion::AllocateAsan)) {
        // TODO: Implement ASAN support for VIRTIO driver
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
      }

      return HSA_STATUS_SUCCESS;
    }
  }

  const uint32_t node_id = (alloc_flags & core::MemoryRegion::AllocateGTTAccess)
      ? agent_node_id
      : m_region.owner()->node_id();

  //// Allocate memory.
  //// If it fails attempt to release memory from the block allocator and retry.
  ret = vhsaKmtAllocMemory(node_id, size, kmt_alloc_flags, mem);
  if (ret != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  if (*mem == nullptr) {
    m_region.owner()->Trim();
    ret = vhsaKmtAllocMemory(node_id, size, kmt_alloc_flags, mem);
    if (ret != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
  }

  if (*mem != nullptr) {
    if (kmt_alloc_flags.ui32.NoAddress) return HSA_STATUS_SUCCESS;

    // Commit the memory.
    // For system memory, on non-restricted allocation, map it to all GPUs. On
    // restricted allocation, only CPU is allowed to access by default, so
    // no need to map
    // For local memory, only map it to the owning GPU. Mapping to other GPU,
    // if the access is allowed, is performed on AllowAccess.
    HsaMemMapFlags map_flag = m_region.map_flags();
    size_t map_node_count = 1;
    const uint32_t owner_node_id = m_region.owner()->node_id();
    const uint32_t* map_node_id = &owner_node_id;

    if (m_region.IsSystem()) {
      if ((alloc_flags & core::MemoryRegion::AllocateRestrict) == 0) {
        // Map to all GPU agents.
        map_node_count = core::Runtime::runtime_singleton_->gpu_ids().size();

        if (map_node_count == 0) {
          // No need to pin since no GPU in the platform.
          return HSA_STATUS_SUCCESS;
        }

        map_node_id = &core::Runtime::runtime_singleton_->gpu_ids()[0];
      } else {
        // No need to pin it for CPU exclusive access.
        return HSA_STATUS_SUCCESS;
      }
    }

    uint64_t alternate_va = 0;
    const bool is_resident =
        (MakeMemoryResident(*mem, size, &alternate_va, &map_flag, map_node_count, map_node_id) ==
         HSA_STATUS_SUCCESS);

    const bool require_pinning =
        (!m_region.full_profile() || m_region.IsLocalMemory() || m_region.IsScratch());

    if (require_pinning && !is_resident) {
      vhsaKmtFreeMemory(*mem, size);
      *mem = nullptr;
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    if ((alloc_flags & core::MemoryRegion::AllocateAsan)) {
      // TODO: Implement ASAN support for VIRTIO driver
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    return HSA_STATUS_SUCCESS;
  }

  return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
}

hsa_status_t KfdVirtioDriver::FreeMemory(void* mem, size_t size) {
  MakeMemoryUnresident(mem);
  return vhsaKmtFreeMemory(mem, size) == HSAKMT_STATUS_SUCCESS ? HSA_STATUS_SUCCESS
                                                               : HSA_STATUS_ERROR;
}

hsa_status_t KfdVirtioDriver::AllocateScratchMemory(uint32_t node_id, uint64_t size,
                                                    void** mem) const {
  assert(mem != nullptr);
  assert(size != 0);

  HsaMemFlags flags = {};
  flags.ui32.Scratch = 1;
  flags.ui32.HostAccess = 1;
  void* ptr = nullptr;

  HSAKMT_STATUS ret = vhsaKmtAllocMemory(node_id, size, flags, &ptr);
  if (ret != HSAKMT_STATUS_SUCCESS || ptr == nullptr) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  *mem = ptr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::RegisterMemory(void* ptr, uint64_t size,
                                             HsaMemFlags mem_flags) const {
  assert(ptr != nullptr);
  assert(size != 0);

  if (vhsaKmtRegisterMemoryWithFlags(ptr, size, mem_flags) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::DeregisterMemory(void* ptr) const {
  if (vhsaKmtDeregisterMemory(ptr) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::AvailableMemory(uint32_t node_id, uint64_t* available_size) const {
  assert(available_size != nullptr);

  if (vhsaKmtAvailableMemory(node_id, available_size) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::MakeMemoryResident(const void* mem, size_t size,
                                                 uint64_t* alternate_va,
                                                 const HsaMemMapFlags* mem_flags,
                                                 uint32_t num_nodes, const uint32_t* nodes) const {
  assert(mem != nullptr);
  assert(size != 0);

  if (mem_flags == nullptr && nodes == nullptr) {
    if (vhsaKmtMapMemoryToGPU(const_cast<void*>(mem), size, alternate_va) != HSAKMT_STATUS_SUCCESS)
      return HSA_STATUS_ERROR;
  } else if (mem_flags != nullptr && nodes != nullptr) {
    if (vhsaKmtMapMemoryToGPUNodes(const_cast<void*>(mem), size, alternate_va, *mem_flags,
                                   num_nodes,
                                   const_cast<uint32_t*>(nodes)) != HSAKMT_STATUS_SUCCESS)
      return HSA_STATUS_ERROR;
  } else {
    debug_print("Invalid memory flags ptr:%p nodes ptr:%p\n", mem_flags, nodes);
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::MakeMemoryUnresident(const void* mem) const {
  vhsaKmtUnmapMemoryToGPU(const_cast<void*>(mem));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::CreateQueue(uint32_t node_id, HSA_QUEUE_TYPE type, uint32_t queue_pct,
                                          HSA_QUEUE_PRIORITY priority, uint32_t sdma_engine_id,
                                          void* queue_addr, uint64_t queue_size_bytes,
                                          HsaEvent* event, HsaQueueResource& queue_resource) const {
  if (vhsaKmtCreateQueueExt(node_id, type, queue_pct, priority, sdma_engine_id, queue_addr,
                            queue_size_bytes, event, &queue_resource) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::DestroyQueue(HSA_QUEUEID queue_id) const {
  if (vhsaKmtDestroyQueue(queue_id) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::UpdateQueue(HSA_QUEUEID queue_id, uint32_t queue_percentage,
                                          HSA_QUEUE_PRIORITY priority, void* queue_mem,
                                          uint64_t queue_size, HsaEvent* event) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t KfdVirtioDriver::SetQueueCUMask(HSA_QUEUEID queue_id, uint32_t num_cu_mask,
                                             uint32_t* cu_mask) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t KfdVirtioDriver::AllocQueueGWS(HSA_QUEUEID queue_id, uint32_t num_GWS,
                                            uint32_t* GWS) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t KfdVirtioDriver::ExportDMABuf(void* mem, size_t size, int* dmabuf_fd, size_t* offset) {
  return HSA_STATUS_ERROR;
}

hsa_status_t KfdVirtioDriver::ImportDMABuf(int dmabuf_fd, core::Agent& agent,
                                           core::ShareableHandle& handle) {
  return HSA_STATUS_ERROR;
}

hsa_status_t KfdVirtioDriver::Map(core::ShareableHandle handle, void* mem, size_t offset,
                                  size_t size, hsa_access_permission_t perms) {
  return HSA_STATUS_ERROR;
}

hsa_status_t KfdVirtioDriver::Unmap(core::ShareableHandle handle, void* mem, size_t offset,
                                    size_t size) {
  return HSA_STATUS_ERROR;
}

hsa_status_t KfdVirtioDriver::ReleaseShareableHandle(core::ShareableHandle& handle) {
  return HSA_STATUS_ERROR;
}

hsa_status_t KfdVirtioDriver::GetTileConfig(uint32_t node_id, HsaGpuTileConfig* config) const {
  if (vhsaKmtGetTileConfig(node_id, config) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::SPMAcquire(uint32_t node_id) const { return HSA_STATUS_ERROR; }

hsa_status_t KfdVirtioDriver::SPMRelease(uint32_t node_id) const { return HSA_STATUS_ERROR; }

hsa_status_t KfdVirtioDriver::SPMSetDestBuffer(uint32_t node_id, uint32_t size, uint32_t* timeout,
                                               uint32_t* size_copied, void* dest,
                                               bool* is_data_loss) const {
  return HSA_STATUS_ERROR;
}


hsa_status_t KfdVirtioDriver::OpenSMI(uint32_t node_id, int* fd) const { return HSA_STATUS_ERROR; }

hsa_status_t KfdVirtioDriver::GetWallclockFrequency(uint32_t node_id, uint64_t* frequency) const {
  assert(frequency != nullptr);

  amdgpu_gpu_info info;
  amdgpu_device_handle handle;
  if (GetDeviceHandle(node_id, reinterpret_cast<void**>(&handle)) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  if (vamdgpu_query_gpu_info(handle, &info) < 0) return HSA_STATUS_ERROR;

  // Reported by libdrm in KHz.
  *frequency = uint64_t(info.gpu_counter_freq) * 1000ull;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdVirtioDriver::IsModelEnabled(bool* enable) const {
  *enable = false;
  return HSA_STATUS_SUCCESS;
}

}  // namespace AMD
}  // namespace rocr
