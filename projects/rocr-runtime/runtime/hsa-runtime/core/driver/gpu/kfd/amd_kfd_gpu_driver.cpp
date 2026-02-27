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

#include "core/inc/amd_gpu_driver.h"

#include <memory>
#include <string>

#if defined(__linux__)
#include <amdgpu_drm.h>
#include <link.h>
#include <sys/ioctl.h>
#endif

#include "hsakmt/hsakmt.h"

#include "core/inc/amd_gpu_agent.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/runtime.h"

#if defined(_WIN32)
#include "loader/executable.hpp"
#endif

extern r_debug _amdgpu_r_debug;

namespace rocr {
namespace AMD {

hsa_status_t GpuDriver::Init() {
  HSAKMT_STATUS ret =
      HSAKMT_CALL(hsaKmtRuntimeEnable(&_amdgpu_r_debug, core::Runtime::runtime_singleton_->flag().debug()));

  if (ret != HSAKMT_STATUS_SUCCESS && ret != HSAKMT_STATUS_NOT_SUPPORTED) return HSA_STATUS_ERROR;

  uint32_t caps_mask = 0;
  if (HSAKMT_CALL(hsaKmtGetRuntimeCapabilities(&caps_mask)) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  core::Runtime::runtime_singleton_->KfdVersion(
      ret != HSAKMT_STATUS_NOT_SUPPORTED,
      !!(caps_mask & HSA_RUNTIME_ENABLE_CAPS_SUPPORTS_CORE_DUMP_MASK));

  if (HSAKMT_CALL(hsaKmtGetVersion(&version_)) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  if (version_.KernelInterfaceMajorVersion == kfd_version_major_min &&
      version_.KernelInterfaceMinorVersion < kfd_version_major_min)
    return HSA_STATUS_ERROR;

  core::Runtime::runtime_singleton_->KfdVersion(version_);

  if (version_.KernelInterfaceMajorVersion == 1 && version_.KernelInterfaceMinorVersion == 0)
    core::g_use_interrupt_wait = false;

  bool xnack_mode = BindXnackMode();
  core::Runtime::runtime_singleton_->XnackEnabled(xnack_mode);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::ShutDown() {
  HSAKMT_STATUS ret = HSAKMT_CALL(hsaKmtRuntimeDisable());
  if (ret != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  ret = HSAKMT_CALL(hsaKmtReleaseSystemProperties());

  if (ret != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  return Close();
}

hsa_status_t GpuDriver::DiscoverDriver(std::unique_ptr<core::Driver>& driver) {
  auto tmp_driver = std::unique_ptr<core::Driver>(new GpuDriver("/dev/kfd"));

  if (tmp_driver->Open() == HSA_STATUS_SUCCESS) {
    driver = std::move(tmp_driver);
    return HSA_STATUS_SUCCESS;
  }

  return HSA_STATUS_ERROR;
}

hsa_status_t GpuDriver::QueryKernelModeDriver(core::DriverQuery query) {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::Open() {
  return HSAKMT_CALL(hsaKmtOpenKFD()) == HSAKMT_STATUS_SUCCESS ? HSA_STATUS_SUCCESS
                                                  : HSA_STATUS_ERROR;
}

hsa_status_t GpuDriver::Close() {
  return HSAKMT_CALL(hsaKmtCloseKFD()) == HSAKMT_STATUS_SUCCESS ? HSA_STATUS_SUCCESS
                                                   : HSA_STATUS_ERROR;
}

hsa_status_t GpuDriver::GetSystemProperties(HsaSystemProperties& sys_props) const {
  if (HSAKMT_CALL(hsaKmtReleaseSystemProperties()) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  if (HSAKMT_CALL(hsaKmtAcquireSystemProperties(&sys_props)) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetNodeProperties(HsaNodeProperties& node_props, uint32_t node_id) const {
  if (HSAKMT_CALL(hsaKmtGetNodeProperties(node_id, &node_props)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetEdgeProperties(std::vector<HsaIoLinkProperties>& io_link_props,
                                          uint32_t node_id) const {
  if (HSAKMT_CALL(hsaKmtGetNodeIoLinkProperties(node_id, io_link_props.size(), io_link_props.data())) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetMemoryProperties(uint32_t node_id,
                                            std::vector<HsaMemoryProperties>& mem_props) const {
  if (!mem_props.data()) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (HSAKMT_CALL(hsaKmtGetNodeMemoryProperties(node_id, mem_props.size(), mem_props.data())) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetCacheProperties(uint32_t node_id, uint32_t processor_id,
                                           std::vector<HsaCacheProperties>& cache_props) const {
  if (!cache_props.data()) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (HSAKMT_CALL(hsaKmtGetNodeCacheProperties(node_id, processor_id, cache_props.size(), cache_props.data())) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t
GpuDriver::AllocateMemory(const core::MemoryRegion &mem_region,
                          core::MemoryRegion::AllocateFlags alloc_flags,
                          void **mem, size_t size, uint32_t agent_node_id) {
  const MemoryRegion &m_region(static_cast<const MemoryRegion &>(mem_region));
  HsaMemFlags kmt_alloc_flags(m_region.mem_flags());

  kmt_alloc_flags.ui32.ExecuteAccess =
      (alloc_flags & core::MemoryRegion::AllocateExecutable ? 1 : 0);

  if (m_region.IsSystem() &&
      (alloc_flags & core::MemoryRegion::AllocateNonPaged)) {
    kmt_alloc_flags.ui32.NonPaged = 1;
  }

  if (!m_region.IsLocalMemory() &&
      (alloc_flags & core::MemoryRegion::AllocateMemoryOnly)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Allocating a memory handle for virtual memory
  kmt_alloc_flags.ui32.NoAddress =
      !!(alloc_flags & core::MemoryRegion::AllocateMemoryOnly);

  // Allocate pseudo fine grain memory
  kmt_alloc_flags.ui32.CoarseGrain =
      (alloc_flags & core::MemoryRegion::AllocatePCIeRW
           ? 0
           : kmt_alloc_flags.ui32.CoarseGrain);

  kmt_alloc_flags.ui32.NoSubstitute =
      (alloc_flags & core::MemoryRegion::AllocatePinned
           ? 1
           : kmt_alloc_flags.ui32.NoSubstitute);

  kmt_alloc_flags.ui32.GTTAccess =
      (alloc_flags & core::MemoryRegion::AllocateGTTAccess
           ? 1
           : kmt_alloc_flags.ui32.GTTAccess);

  kmt_alloc_flags.ui32.Uncached =
      (alloc_flags & core::MemoryRegion::AllocateUncached
            ? 1
            : kmt_alloc_flags.ui32.Uncached);

  kmt_alloc_flags.ui32.QueueObject =
      (alloc_flags & core::MemoryRegion::AllocateQueueObject ? 1
                                                             : kmt_alloc_flags.ui32.QueueObject);
  if (kmt_alloc_flags.ui32.Uncached) {
    /* Uncached overwrites CoarseGrain and ExtendedCoherent */
    kmt_alloc_flags.ui32.CoarseGrain = 0;
    kmt_alloc_flags.ui32.ExtendedCoherent = 0;
  }

  kmt_alloc_flags.ui32.ExecuteBlit =
    !!(alloc_flags & core::MemoryRegion::AllocateExecutableBlitKernelObject);

  if (m_region.IsLocalMemory()) {
    // Allocate physically contiguous memory. AllocateKfdMemory function call
    // will fail if this flag is not supported in KFD.
    kmt_alloc_flags.ui32.Contiguous =
        (alloc_flags & core::MemoryRegion::AllocateContiguous
             ? 1
             : kmt_alloc_flags.ui32.Contiguous);
  }

  //// Only allow using the suballocator for ordinary VRAM.
  if (m_region.IsLocalMemory() && !kmt_alloc_flags.ui32.NoAddress) {
    bool subAllocEnabled =
        !core::Runtime::runtime_singleton_->flag().disable_fragment_alloc();
    // Avoid modifying executable or queue allocations.
    bool useSubAlloc = subAllocEnabled;
    useSubAlloc &=
        ((alloc_flags & (~core::MemoryRegion::AllocateRestrict)) == 0);

    if (useSubAlloc) {
      *mem = m_region.fragment_alloc(size);

      if ((alloc_flags & core::MemoryRegion::AllocateAsan) &&
          HSAKMT_CALL(hsaKmtReplaceAsanHeaderPage(*mem)) != HSAKMT_STATUS_SUCCESS) {
        m_region.fragment_free(*mem);
        *mem = nullptr;
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
      }

      return HSA_STATUS_SUCCESS;
    }
  }

  const uint32_t node_id =
      (alloc_flags & core::MemoryRegion::AllocateGTTAccess)
          ? agent_node_id
          : m_region.owner()->node_id();

  //// Allocate memory.
  //// If it fails attempt to release memory from the block allocator and retry.
  *mem = AllocateKfdMemory(kmt_alloc_flags, node_id, size);
  if (*mem == nullptr) {
    m_region.owner()->Trim();
    *mem = AllocateKfdMemory(kmt_alloc_flags, node_id, size);
  }

  if (*mem != nullptr) {
    if (kmt_alloc_flags.ui32.NoAddress)
      return HSA_STATUS_SUCCESS;

    // Commit the memory.
    // For system memory, on non-restricted allocation, map it to all GPUs. On
    // restricted allocation, only CPU is allowed to access by default, so
    // no need to map
    // For local memory, only map it to the owning GPU. Mapping to other GPU,
    // if the access is allowed, is performed on AllowAccess.
    HsaMemMapFlags map_flag = m_region.map_flags();
    size_t map_node_count = 1;
    const uint32_t owner_node_id = m_region.owner()->node_id();
    const uint32_t *map_node_id = &owner_node_id;

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
    const bool is_resident = MakeKfdMemoryResident(
        map_node_count, map_node_id, *mem, size, &alternate_va, map_flag);

    const bool require_pinning =
        (!m_region.full_profile() || m_region.IsLocalMemory() ||
         m_region.IsScratch());

    if (require_pinning && !is_resident) {
      FreeKfdMemory(*mem, size);
      *mem = nullptr;
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    if ((alloc_flags & core::MemoryRegion::AllocateAsan) &&
        HSAKMT_CALL(hsaKmtReplaceAsanHeaderPage(*mem)) != HSAKMT_STATUS_SUCCESS) {
      FreeKfdMemory(*mem, size);
      *mem = nullptr;
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    return HSA_STATUS_SUCCESS;
  }

  return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
}

hsa_status_t GpuDriver::FreeMemory(void *mem, size_t size) {
  MakeKfdMemoryUnresident(mem);
  return FreeKfdMemory(mem, size) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

hsa_status_t GpuDriver::CreateQueue(uint32_t node_id, HSA_QUEUE_TYPE type, uint32_t queue_pct,
                                    HSA::hsa_amd_queue_priority_internal_t priority, uint32_t sdma_engine_id,
                                    void* queue_addr, uint64_t queue_size_bytes, HsaEvent* event,
                                    HsaQueueResource& queue_resource) const {
  // Convert from ROCR internal priority type to KFD type
  HSA_QUEUE_PRIORITY kfd_priority = HsaInternalToKfdPriority(priority);

  if (HSAKMT_CALL(hsaKmtCreateQueueExt(node_id, type, queue_pct, kfd_priority, sdma_engine_id,
                                       queue_addr, queue_size_bytes, event, &queue_resource)) !=
      HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::DestroyQueue(HSA_QUEUEID queue_id) const {
  if (HSAKMT_CALL(hsaKmtDestroyQueue(queue_id)) != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::UpdateQueue(HSA_QUEUEID queue_id, uint32_t queue_pct,
                                    HSA::hsa_amd_queue_priority_internal_t priority, void* queue_addr,
                                    uint64_t queue_size, HsaEvent* event) const {
  // Convert from ROCR internal priority type to KFD type
  HSA_QUEUE_PRIORITY kfd_priority = HsaInternalToKfdPriority(priority);

  if (HSAKMT_CALL(hsaKmtUpdateQueue(queue_id, queue_pct, kfd_priority, queue_addr, queue_size,
                                    event)) != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::SetQueueCUMask(HSA_QUEUEID queue_id, uint32_t cu_mask_count,
                                       uint32_t* queue_cu_mask) const {
  if (HSAKMT_CALL(hsaKmtSetQueueCUMask(queue_id, cu_mask_count, queue_cu_mask)) !=
      HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::AllocQueueGWS(HSA_QUEUEID queue_id, uint32_t num_gws,
                                      uint32_t* first_gws) const {
  if (HSAKMT_CALL(hsaKmtAllocQueueGWS(queue_id, num_gws, first_gws)) != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetShareableHandle(void* va, void* mem, size_t size,
                                           core::ShareableHandle* handle) {
#if defined(_WIN32)
  uint64_t mem_handle;
  HSAKMT_STATUS status = HSAKMT_CALL(hsaKmtGetMemoryHandle(va, mem, size, &mem_handle));
  if (status != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  handle->handle = mem_handle;
  return HSA_STATUS_SUCCESS;
#else
  return HSA_STATUS_ERROR;
#endif
}

hsa_status_t GpuDriver::ExportDMABuf(void *mem, size_t size, int *dmabuf_fd,
                                     size_t *offset) {
  int dmabuf_fd_res = -1;
  size_t offset_res = 0;
  HSAKMT_STATUS status =
      HSAKMT_CALL(hsaKmtExportDMABufHandle(mem, size, &dmabuf_fd_res, &offset_res));
  if (status != HSAKMT_STATUS_SUCCESS) {
    if (status == HSAKMT_STATUS_INVALID_PARAMETER) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  *dmabuf_fd = dmabuf_fd_res;
  *offset = offset_res;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::ImportDMABuf(int dmabuf_fd, core::Agent &agent,
                                     core::ShareableHandle &handle) {
  auto &gpu_agent = static_cast<GpuAgent &>(agent);
  HsaExternalHandleDesc desc;
  desc.device_handle = gpu_agent.libThunkDev();
  desc.fd = static_cast<HSAint32>(dmabuf_fd);
  desc.type = HSA_EXTERNAL_HANDLE_DMA_BUF;
  desc.metadata = 0;
  HsaHandleImportFlags hflags = {0};
  HsaHandleImportResult res;
  HSAKMT_STATUS status = HSAKMT_CALL(hsaKmtHandleImport(&desc, &res, &hflags));
  if (status != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  handle.handle = reinterpret_cast<uint64_t>(res.buf_handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::Map(core::ShareableHandle handle, void *mem,
                            size_t offset, size_t size,
                            hsa_access_permission_t perms) {
  HsaMemoryObjectHandle memhandle = reinterpret_cast<HsaMemoryObjectHandle>(handle.handle);
  HSAKMT_STATUS status = HSAKMT_CALL(hsaKmtMemoryVaMap(memhandle, static_cast<HSAuint64>(offset),
                                     static_cast<HSAuint64>(size), reinterpret_cast<HSAuint64>(mem),
                                     mem_perm(perms)));
  if (status != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::Unmap(core::ShareableHandle handle, void *mem,
                              size_t offset, size_t size) {
  HsaMemoryObjectHandle memhandle = reinterpret_cast<HsaMemoryObjectHandle>(handle.handle);
  HSAKMT_STATUS status = HSAKMT_CALL(hsaKmtMemoryVaUnmap(memhandle, (HSAuint64)offset, (HSAuint64)size,
                                     reinterpret_cast<HSAuint64>(mem)));
  if (status != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::ReleaseShareableHandle(core::ShareableHandle &handle) {
  auto memhandle = reinterpret_cast<HsaMemoryObjectHandle>(handle.handle);
  HSAKMT_STATUS status = HSAKMT_CALL(hsaKmtMemHandleFree(memhandle));
  if (status != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  handle = {};
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::SPMAcquire(uint32_t preferred_node_id) const {
  if (HSAKMT_CALL(hsaKmtSPMAcquire(preferred_node_id)) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::SPMRelease(uint32_t preferred_node_id) const {
  if (HSAKMT_CALL(hsaKmtSPMRelease(preferred_node_id)) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::SPMSetDestBuffer(uint32_t preferred_node_id, uint32_t size_bytes,
                                         uint32_t* timeout, uint32_t* size_copied,
                                         void* dest_mem_addr, bool* is_spm_data_loss) const {
  if (HSAKMT_CALL(hsaKmtSPMSetDestBuffer(preferred_node_id, size_bytes, timeout, size_copied, dest_mem_addr,
                             is_spm_data_loss)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::OpenSMI(uint32_t node_id, int* fd) const {
  if (HSAKMT_CALL(hsaKmtOpenSMI(node_id, fd)) != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::SetTrapHandler(uint32_t node_id, const void* base, uint64_t base_size,
                                       const void* buffer_base, uint64_t buffer_base_size) const {
  if (HSAKMT_CALL(hsaKmtSetTrapHandler(node_id, const_cast<void*>(base), base_size,
                                       const_cast<void*>(buffer_base), buffer_base_size)) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::AllocateScratchMemory(uint32_t node_id, uint64_t size, void** mem) const {
  assert(mem);
  assert(size > 0);

  HsaMemFlags flags = {};
  flags.ui32.Scratch = 1;
  flags.ui32.HostAccess = 1;

  void* ptr = AllocateKfdMemory(flags, node_id, size);
  if (ptr == nullptr) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  *mem = ptr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetDeviceHandle(uint32_t node_id, void** device_handle) const {
  assert(device_handle);

  if (HSAKMT_CALL(hsaKmtGetAMDGPUDeviceHandle(node_id, reinterpret_cast<HsaAMDGPUDeviceHandle*>(device_handle))) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetClockCounters(uint32_t node_id, HsaClockCounters* clock_counter) const {
  assert(clock_counter);

  if (HSAKMT_CALL(hsaKmtGetClockCounters(node_id, clock_counter)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetTileConfig(uint32_t node_id, HsaGpuTileConfig* config) const {
  assert(config);

  if (HSAKMT_CALL(hsaKmtGetTileConfig(node_id, config)) != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::AvailableMemory(uint32_t node_id, uint64_t* available_size) const {
  assert(available_size);

  if (HSAKMT_CALL(hsaKmtAvailableMemory(node_id, available_size)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::RegisterMemory(void* ptr, uint64_t size, HsaMemFlags mem_flags) const {
  assert(ptr);
  assert(size > 0);

  if (HSAKMT_CALL(hsaKmtRegisterMemoryWithFlags(ptr, size, mem_flags)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::DeregisterMemory(void* ptr) const {
  if (HSAKMT_CALL(hsaKmtDeregisterMemory(ptr)) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::MakeMemoryResident(const void* mem, size_t size, uint64_t* alternate_va,
                                           const HsaMemMapFlags* mem_flags, uint32_t num_nodes,
                                           const uint32_t* nodes) const {
  if (mem_flags == nullptr && nodes == nullptr) {
    if (HSAKMT_CALL(hsaKmtMapMemoryToGPU(const_cast<void*>(mem), size, alternate_va)) !=
        HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }
  } else if (mem_flags != nullptr && nodes != nullptr) {
    if (!MakeKfdMemoryResident(num_nodes, nodes, mem, size, alternate_va, *mem_flags)) {
      return HSA_STATUS_ERROR;
    }
  } else {
    debug_print("Invalid memory flags ptr:%p nodes ptr:%p\n", mem_flags, nodes);
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::MakeMemoryUnresident(const void* mem) const {
  HSAKMT_CALL(hsaKmtUnmapMemoryToGPU(const_cast<void*>(mem)));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::IsModelEnabled(bool* enable) const {
  // AIE does not support streaming performance monitor.
  HSAKMT_STATUS status = HSAKMT_STATUS_ERROR;
  status = HSAKMT_CALL(hsaKmtModelEnabled(enable));
  if (status != HSAKMT_STATUS_SUCCESS)
     return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetWallclockFrequency(uint32_t node_id, uint64_t* frequency) const {
  assert(frequency);

  HSAKMT_CALL(hsaKmtGetNodeWallclockFrequency(node_id, frequency));

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetQueueSaveAreaInfo(HSA_QUEUEID queue_id, void** address, size_t* size) const {
  assert(address);
  assert(size);

  HsaQueueInfo queue_info = {};

  HSAKMT_STATUS status = HSAKMT_CALL(hsaKmtGetQueueInfo(queue_id, &queue_info));
  if (status != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }

  *address = queue_info.SaveAreaHeader;
  *size = queue_info.SaveAreaSizeInBytes;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::CreateEvent(HsaEventDescriptor& event_descriptor, bool manual_reset,
                                    HsaEvent** event) const {
  if (HSAKMT_CALL(hsaKmtCreateEvent(&event_descriptor, manual_reset, false, event)) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::DestroyEvent(HsaEvent* event) const {
  HSAKMT_CALL(hsaKmtDestroyEvent(event));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::WaitOnEvent(HsaEvent* event, uint32_t timeout_ms,
                                    uint64_t* event_age) const {
  HSAKMT_CALL(hsaKmtWaitOnEvent_Ext(event, timeout_ms, event_age));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::WaitOnMultipleEvents(HsaEvent** events, uint32_t num_events,
                                             bool wait_on_all, uint32_t timeout_ms,
                                             uint64_t* event_age) const {
  HSAKMT_CALL(
      hsaKmtWaitOnMultipleEvents_Ext(events, num_events, wait_on_all, timeout_ms, event_age));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::SetEvent(HsaEvent* event) const {
  HSAKMT_CALL(hsaKmtSetEvent(event));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::RingDoorbell(HSA_QUEUEID queue_id, uint64_t value) const {
  HSAKMT_CALL(hsaKmtQueueRingDoorbell(queue_id, value));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::QueryPointerInfo(const void* ptr, HsaPointerInfo* info) const {
  if (HSAKMT_CALL(hsaKmtQueryPointerInfo(ptr, info)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::SetMemoryUserData(const void* ptr, void* user_data) const {
  if (HSAKMT_CALL(hsaKmtSetMemoryUserData(ptr, user_data)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::FreeMemoryHandle(HsaMemoryObjectHandle handle) const {
  if (HSAKMT_CALL(hsaKmtMemHandleFree(handle)) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::ReturnAsanHeaderPage(void* addr) const {
  if (HSAKMT_CALL(hsaKmtReturnAsanHeaderPage(addr)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::MapMemoryToGPU(const void* mem, size_t size, uint64_t* alternate_va) const {
  if (HSAKMT_CALL(hsaKmtMapMemoryToGPU(const_cast<void*>(mem), size, alternate_va)) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::MapMemoryToGPUNodes(const void* mem, size_t size, uint64_t* alternate_va,
                                            HsaMemMapFlags flags, uint32_t num_nodes,
                                            const uint32_t* nodes) const {
  if (HSAKMT_CALL(hsaKmtMapMemoryToGPUNodes(const_cast<void*>(mem), size, alternate_va, flags,
                                            num_nodes, const_cast<uint32_t*>(nodes))) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetMemoryCpuAddr(void* device_handle, void* mem_handle, int* drm_fd,
                                         uint64_t* cpu_addr) const {
  if (HSAKMT_CALL(hsaKmtMemoryGetCpuAddr(static_cast<HsaMemoryObjectHandle>(device_handle),
                                         static_cast<HsaMemoryObjectHandle>(mem_handle), drm_fd,
                                         cpu_addr)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::AllocateMemoryAlign(uint32_t node, size_t size, size_t alignment,
                                            HsaMemFlags flags, void** mem) const {
  if (HSAKMT_CALL(hsaKmtAllocMemoryAlign(node, size, alignment, flags, mem)) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::MemoryCpuMap(HsaMemoryObjectHandle handle, void** cpu_ptr) const {
  if (HSAKMT_CALL(hsaKmtMemoryCpuMap(handle, cpu_ptr)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::ExportDMABufHandle(const void* mem, size_t size, int* dmabuf_fd,
                                           uint64_t* offset) const {
  if (HSAKMT_CALL(hsaKmtExportDMABufHandle(const_cast<void*>(mem), size, dmabuf_fd, offset)) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::HandleImport(const HsaExternalHandleDesc* desc,
                                     HsaHandleImportResult* result,
                                     HsaHandleImportFlags* flags) const {
  if (HSAKMT_CALL(hsaKmtHandleImport(desc, result, flags)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::ShareMemory(void* mem, size_t size, HsaSharedMemoryHandle* handle) const {
  if (HSAKMT_CALL(hsaKmtShareMemory(mem, size, handle)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::RegisterSharedHandle(const HsaSharedMemoryHandle* handle, void** address,
                                             HSAuint64* size) const {
  if (HSAKMT_CALL(hsaKmtRegisterSharedHandle(handle, address, size)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::RegisterGraphicsHandleToNodes(int dmabuf_fd, HsaGraphicsResourceInfo* info,
                                                      uint32_t num_nodes, uint32_t* nodes) const {
  if (HSAKMT_CALL(hsaKmtRegisterGraphicsHandleToNodes(dmabuf_fd, info, num_nodes, nodes)) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::RegisterGraphicsHandleToNodesExt(HSAuint64 dmabuf_fd,
                                                         HsaGraphicsResourceInfo* info,
                                                         HSAuint64 num_nodes, uint32_t* nodes,
                                                         HSA_REGISTER_MEM_FLAGS flags) const {
  if (HSAKMT_CALL(hsaKmtRegisterGraphicsHandleToNodesExt(dmabuf_fd, info, num_nodes, nodes,
                                                         flags)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::MemoryVaMap(HsaMemoryObjectHandle handle, uint64_t offset, uint64_t size,
                                    uint64_t va, uint32_t access) const {
  if (HSAKMT_CALL(
          hsaKmtMemoryVaMap(handle, offset, size, va, static_cast<HsaMemoryMapFlags>(access))) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::MemoryVaUnmap(HsaMemoryObjectHandle handle, uint64_t offset, uint64_t size,
                                      uint64_t va) const {
  if (HSAKMT_CALL(hsaKmtMemoryVaUnmap(handle, offset, size, va)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::SVMSetAttr(void* addr, size_t size, uint32_t count,
                                   HSA_SVM_ATTRIBUTE* attrs) const {
  if (HSAKMT_CALL(hsaKmtSVMSetAttr(addr, size, count, attrs)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::SVMGetAttr(void* addr, size_t size, uint32_t count,
                                   HSA_SVM_ATTRIBUTE* attrs) const {
  if (HSAKMT_CALL(hsaKmtSVMGetAttr(addr, size, count, attrs)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::PcSamplingQueryCapabilities(uint32_t node_id, void* sample_info,
                                                    size_t size, uint32_t* count) const {
  if (HSAKMT_CALL(hsaKmtPcSamplingQueryCapabilities(node_id, sample_info, size, count)) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::PcSamplingCreate(uint32_t node_id, HsaPcSamplingInfo* sample_info,
                                         HsaPcSamplingTraceId* trace_id) const {
  HSAKMT_STATUS status = HSAKMT_CALL(hsaKmtPcSamplingCreate(node_id, sample_info, trace_id));
  if (status == HSAKMT_STATUS_KERNEL_ALREADY_OPENED)
    return (hsa_status_t)HSA_STATUS_ERROR_RESOURCE_BUSY;
  if (status != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::PcSamplingDestroy(uint32_t node_id, HsaPcSamplingTraceId trace_id) const {
  if (HSAKMT_CALL(hsaKmtPcSamplingDestroy(node_id, trace_id)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::PcSamplingStart(uint32_t node_id, HsaPcSamplingTraceId trace_id) const {
  if (HSAKMT_CALL(hsaKmtPcSamplingStart(node_id, trace_id)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::PcSamplingStop(uint32_t node_id, HsaPcSamplingTraceId trace_id) const {
  if (HSAKMT_CALL(hsaKmtPcSamplingStop(node_id, trace_id)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::DbgEnable(void** runtime_ptr, uint32_t* runtime_size) const {
  if (HSAKMT_CALL(hsaKmtDbgEnable(runtime_ptr, runtime_size)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::DbgDisable() const {
  if (HSAKMT_CALL(hsaKmtDbgDisable()) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::DbgGetDeviceData(void** data, uint32_t* count, uint32_t* entry_size) const {
  if (HSAKMT_CALL(hsaKmtDbgGetDeviceData(data, count, entry_size)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::DbgGetQueueData(void** data, uint32_t* count, uint32_t* entry_size,
                                        bool suspend) const {
  if (HSAKMT_CALL(hsaKmtDbgGetQueueData(data, count, entry_size, suspend)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::AisReadWriteFile(void* device_ptr, size_t size, int fd, int64_t file_offset,
                                         HsaAisFlags operation, uint64_t* size_copied,
                                         int32_t* status) const {
  if (HSAKMT_CALL(hsaKmtAisReadWriteFile(device_ptr, size, fd, file_offset, operation, size_copied,
                                         status)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

} // namespace AMD
} // namespace rocr
