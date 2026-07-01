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

#include "core/inc/amd_kfd_driver.h"

#if defined(__linux__)
#include <amdgpu_drm.h>
#include <link.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#endif

#include "hsakmt/hsakmt.h"

#include "core/inc/amd_gpu_agent.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/runtime.h"
#include "core/util/os.h"

#if defined(_WIN32)
#include "loader/executable.hpp"
#endif

extern r_debug _amdgpu_r_debug;

namespace rocr {

/// @brief Mapping between priority type used internally within ROCR to the type used by KFD

// Highest queue priority allowed for HSA user is HSA_QUEUE_PRIORITY_HIGH
// HSA_QUEUE_PRIORITY_MAXIMUM is reserved for PC Sampling and can only be allocated internally
// in ROCR
__forceinline HSA_QUEUE_PRIORITY HsaInternalToKfdPriority(
    rocr::HSA::hsa_amd_queue_priority_internal_t priority) {
  switch (priority) {
    case rocr::HSA::HSA_AMD_QUEUE_PRIORITY_LOW:
      return HSA_QUEUE_PRIORITY_MINIMUM;
    case rocr::HSA::HSA_AMD_QUEUE_PRIORITY_NORMAL:
      return HSA_QUEUE_PRIORITY_NORMAL;
    case rocr::HSA::HSA_AMD_QUEUE_PRIORITY_HIGH:
      return HSA_QUEUE_PRIORITY_HIGH;
    case rocr::HSA::HSA_AMD_QUEUE_PRIORITY_MAXIMUM:
      return HSA_QUEUE_PRIORITY_MAXIMUM;
    default:
      return HSA_QUEUE_PRIORITY_NORMAL;
  }
}

namespace AMD {
static_assert(
    (sizeof(decltype(core::DriverMemoryHandle::handle)) >= sizeof(HsaMemoryObjectHandle)) &&
        (alignof(decltype(core::DriverMemoryHandle::handle)) >= alignof(HsaMemoryObjectHandle)),
    "DriverMemoryHandle cannot store a HsaMemoryObjectHandle");
namespace {

__forceinline HsaMemoryMapFlags mem_perm(hsa_access_permission_t perm) {
  switch (perm) {
  case HSA_ACCESS_PERMISSION_RO:
    return HSA_MEMORY_ACCESS_RO;
  case HSA_ACCESS_PERMISSION_WO:
    return HSA_MEMORY_ACCESS_WO;
  case HSA_ACCESS_PERMISSION_RW:
    return HSA_MEMORY_ACCESS_RW;
  case HSA_ACCESS_PERMISSION_NONE:
  default:
    return HSA_MEMORY_ACCESS_NONE;
  }
}

} // namespace

KfdDriver::KfdDriver(std::string devnode_name)
    : core::Driver(core::DriverType::KFD, std::move(devnode_name)) {}

hsa_status_t KfdDriver::Init() {
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

hsa_status_t KfdDriver::ShutDown() {
  HSAKMT_STATUS ret = HSAKMT_CALL(hsaKmtRuntimeDisable());
  if (ret != HSAKMT_STATUS_SUCCESS && ret != HSAKMT_STATUS_NOT_SUPPORTED) return HSA_STATUS_ERROR;

  ret = HSAKMT_CALL(hsaKmtReleaseSystemProperties());

  if (ret != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  return Close();
}

hsa_status_t KfdDriver::DiscoverDriver(std::unique_ptr<core::Driver>& driver) {
  auto tmp_driver = std::unique_ptr<core::Driver>(new KfdDriver("/dev/kfd"));

  if (tmp_driver->Open() == HSA_STATUS_SUCCESS) {
    driver = std::move(tmp_driver);
    return HSA_STATUS_SUCCESS;
  }

  return HSA_STATUS_ERROR;
}

hsa_status_t KfdDriver::QueryKernelModeDriver(core::DriverQuery query) {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::Open() {
  return HSAKMT_CALL(hsaKmtOpenKFD()) == HSAKMT_STATUS_SUCCESS ? HSA_STATUS_SUCCESS
                                                  : HSA_STATUS_ERROR;
}

hsa_status_t KfdDriver::Close() {
  return HSAKMT_CALL(hsaKmtCloseKFD()) == HSAKMT_STATUS_SUCCESS ? HSA_STATUS_SUCCESS
                                                   : HSA_STATUS_ERROR;
}

hsa_status_t KfdDriver::GetSystemProperties(HsaSystemProperties& sys_props) const {
  // Note: We intentionally do NOT call hsaKmtReleaseSystemProperties() here.
  // hsaKmtRuntimeEnable (called from Init) already acquired system properties.
  // Releasing and re-acquiring would tear down FMM apertures and fail to
  // re-acquire the VM because the kernel-side VM binding persists.
  // hsaKmtAcquireSystemProperties handles the cached-snapshot case internally.
  if (HSAKMT_CALL(hsaKmtAcquireSystemProperties(&sys_props)) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::GetNodeProperties(HsaNodeProperties& node_props, uint32_t node_id) const {
  if (HSAKMT_CALL(hsaKmtGetNodeProperties(node_id, &node_props)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::GetEdgeProperties(std::vector<HsaIoLinkProperties>& io_link_props,
                                          uint32_t node_id) const {
  if (HSAKMT_CALL(hsaKmtGetNodeIoLinkProperties(node_id, io_link_props.size(), io_link_props.data())) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::GetMemoryProperties(uint32_t node_id,
                                            std::vector<HsaMemoryProperties>& mem_props) const {
  if (!mem_props.data()) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (HSAKMT_CALL(hsaKmtGetNodeMemoryProperties(node_id, mem_props.size(), mem_props.data())) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::GetCacheProperties(uint32_t node_id, uint32_t processor_id,
                                           std::vector<HsaCacheProperties>& cache_props) const {
  if (!cache_props.data()) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (HSAKMT_CALL(hsaKmtGetNodeCacheProperties(node_id, processor_id, cache_props.size(), cache_props.data())) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t
KfdDriver::AllocateMemory(const core::MemoryRegion &mem_region,
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
    // Allocate physically contiguous memory. hsaKmtAllocMemory function call
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

  // agent_node_id uses 0 as the allocator default/unspecified sentinel.
  const bool has_agent_node_id = agent_node_id != 0;
  const bool allocation_uses_agent_node =
      (alloc_flags & (core::MemoryRegion::AllocateGTTAccess |
                      core::MemoryRegion::AllocateQueueObject)) != 0;
  const uint32_t node_id = has_agent_node_id && allocation_uses_agent_node
      ? agent_node_id
      : m_region.owner()->node_id();

  //// Allocate memory.
  //// If it fails attempt to release memory from the block allocator and retry.

  auto status = HSAKMT_CALL(hsaKmtAllocMemory(node_id, size, kmt_alloc_flags, mem));
  if (status == HSAKMT_STATUS_NO_MEMORY) {
    m_region.owner()->Trim();
    status = HSAKMT_CALL(hsaKmtAllocMemory(node_id, size, kmt_alloc_flags, mem));
  }
  if (status == HSAKMT_STATUS_SUCCESS) {
    if (kmt_alloc_flags.ui32.NoAddress) {
      // returns mem
      return HSA_STATUS_SUCCESS;
    }

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

    MAKE_NAMED_SCOPE_GUARD(memoryGuard, [&]() {
      if (*mem != nullptr) {
        HSAKMT_CALL(hsaKmtFreeMemory(*mem, size));
        *mem = nullptr;
      }
    });

    uint64_t alternate_va = 0;

    const bool is_resident =
      (HSAKMT_CALL(hsaKmtMapMemoryToGPUNodes(*mem, size, &alternate_va, map_flag,
                                             map_node_count, const_cast<uint32_t*>(map_node_id))) == HSAKMT_STATUS_SUCCESS);

    // On Windows/DXG, allow allocations to succeed even if MakeResident
    // is best-effort; WDDM will demand-page on GPU access.
    const bool is_windxg =
        core::Runtime::runtime_singleton_->thunkLoader()->IsWinDxg();
    const bool require_pinning =
        !is_windxg &&
        (!m_region.full_profile() || m_region.IsLocalMemory() ||
         m_region.IsScratch());

    if (require_pinning && !is_resident) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    if ((alloc_flags & core::MemoryRegion::AllocateAsan) &&
        HSAKMT_CALL(hsaKmtReplaceAsanHeaderPage(*mem)) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    memoryGuard.Dismiss();
    return HSA_STATUS_SUCCESS;
  }

  return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
}

hsa_status_t KfdDriver::FreeMemory(void *mem, size_t size) {
  HSAKMT_CALL(hsaKmtUnmapMemoryToGPU(const_cast<void *>(mem)));
  return (HSAKMT_CALL(hsaKmtFreeMemory(mem, size)) == HSAKMT_STATUS_SUCCESS) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

hsa_status_t KfdDriver::CreateQueue(uint32_t node_id, HSA_QUEUE_TYPE type, uint32_t queue_pct,
                                    HSA::hsa_amd_queue_priority_internal_t priority, uint32_t sdma_engine_id,
                                    void* queue_addr, uint64_t queue_size_bytes, uint64_t queue_metadata_size_bytes,
                                    HsaEvent* event, HsaQueueResource& queue_resource) const {
  // Convert from ROCR internal priority type to KFD type
  HSA_QUEUE_PRIORITY kfd_priority = HsaInternalToKfdPriority(priority);

  if (HSAKMT_CALL(hsaKmtCreateQueueV2(node_id, type, queue_pct, kfd_priority, sdma_engine_id,
                                         queue_addr, queue_size_bytes, queue_metadata_size_bytes,
                                         event, &queue_resource)) != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::DestroyQueue(HSA_QUEUEID queue_id) const {
  if (HSAKMT_CALL(hsaKmtDestroyQueue(queue_id)) != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::UpdateQueue(HSA_QUEUEID queue_id, uint32_t queue_pct,
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

hsa_status_t KfdDriver::SetQueueCUMask(HSA_QUEUEID queue_id, uint32_t cu_mask_count,
                                       uint32_t* queue_cu_mask) const {
  if (HSAKMT_CALL(hsaKmtSetQueueCUMask(queue_id, cu_mask_count, queue_cu_mask)) !=
      HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::AllocQueueGWS(HSA_QUEUEID queue_id, uint32_t num_gws,
                                      uint32_t* first_gws) const {
  if (HSAKMT_CALL(hsaKmtAllocQueueGWS(queue_id, num_gws, first_gws)) != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::ExportMemoryHandle(const core::Agent& agent, const core::DriverMemoryHandle& handle,
                                           core::ShareType type, uint32_t flags, void* export_handle,
                                           uint64_t* export_offset) {
  if (export_handle == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  switch (type) {
  case core::ShareType::DMABUF_FD: {
    auto* dmabuf_fd = static_cast<int*>(export_handle);
    if (flags & core::EXPORT_MEMORY_FLAGS_KFD_DMABUF) {
      if (export_offset == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      void* mem = reinterpret_cast<void*>(handle.handle);
      if (HSAKMT_CALL(hsaKmtExportDMABufHandle(mem, handle.size, dmabuf_fd, export_offset)) !=
          HSAKMT_STATUS_SUCCESS) {
        return HSA_STATUS_ERROR;
      }
      return HSA_STATUS_SUCCESS;
    }
#if defined(__linux__)
    if (handle.dmabuf_fd != -1) {
      *dmabuf_fd = handle.dmabuf_fd;
      return HSA_STATUS_SUCCESS;
    }
#endif
    (void)export_offset;
    const auto& gpu_agent = static_cast<const GpuAgent&>(agent);

    HsaHandleExportDesc desc = {};
    desc.device_handle = gpu_agent.libThunkDev();
    desc.type = HSA_EXTERNAL_HANDLE_DMA_BUF;
    desc.buf_handle = reinterpret_cast<HsaMemoryObjectHandle>(handle.handle);
    desc.size = handle.size;

    HsaHandleExportFlags export_flags = {};
    HsaMemoryExportResult res = {};

    if (HSAKMT_CALL(hsaKmtHandleExport(&desc, &res, &export_flags)) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }
    *dmabuf_fd = res.dmabuf_fd;
    return HSA_STATUS_SUCCESS;
  }
  case core::ShareType::FABRIC_HANDLE: {
    (void)export_offset;
#if !defined(__linux__)
    assert(!"Unimplemented!");
    return HSA_STATUS_ERROR;
#else
    auto* fabric_handle = static_cast<hsa_fabric_handle_t*>(export_handle);
    const auto& gpu_agent = static_cast<const GpuAgent&>(agent);

    HsaHandleExportDesc desc = {};
    desc.device_handle = gpu_agent.libThunkDev();
    desc.type = HSA_EXTERNAL_HANDLE_FABRIC;
    desc.buf_handle = reinterpret_cast<HsaMemoryObjectHandle>(handle.handle);
    desc.size = handle.size;

    HsaHandleExportFlags export_flags = {};
    HsaMemoryExportResult res = {};

    if (HSAKMT_CALL(hsaKmtHandleExport(&desc, &res, &export_flags)) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }

    memcpy(fabric_handle, reinterpret_cast<void*>(&res.fabric), sizeof(hsa_fabric_handle_t));
    return HSA_STATUS_SUCCESS;
#endif
  }
  default:
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

hsa_status_t KfdDriver::ImportMemoryHandle(const core::Agent& agent, core::DriverMemoryHandle* handle,
                                           core::ShareType type, void* import_handle,
                                           void* mem) {
  if (handle == nullptr || import_handle == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  switch (type) {
  case core::ShareType::DMABUF_FD: {
    const auto& gpu_agent = static_cast<const GpuAgent&>(agent);
    const int dmabuf_fd = static_cast<const core::DriverMemoryHandle*>(import_handle)->dmabuf_fd;

    HsaHandleImportDesc desc = {};
    desc.device_handle = gpu_agent.libThunkDev();
    desc.dmabuf_fd = static_cast<HSAint32>(dmabuf_fd);
    desc.type = HSA_EXTERNAL_HANDLE_DMA_BUF;
    desc.mem = mem;
    desc.metadata = 0;
    HsaHandleImportFlags hflags = {0};
    HsaHandleImportResult res = {};
    HSAKMT_STATUS status = HSAKMT_CALL(hsaKmtHandleImport(&desc, &res, &hflags));
    if (status != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }
    handle->handle = reinterpret_cast<uint64_t>(res.buf_handle);
    handle->size = res.alloc_size;
    return HSA_STATUS_SUCCESS;
  }
  case core::ShareType::FABRIC_HANDLE: {
#if !defined(__linux__)
    assert(!"Unimplemented!");
    return HSA_STATUS_ERROR;
#endif
    const auto& gpu_agent = static_cast<const GpuAgent&>(agent);
    const auto fabric_handle = static_cast<const core::DriverMemoryHandle*>(import_handle)->fabric_handle;

    HsaHandleImportDesc desc = {};
    desc.device_handle = gpu_agent.libThunkDev();
    desc.type = HSA_EXTERNAL_HANDLE_FABRIC;
    memcpy(&desc.fabric, &fabric_handle, sizeof(fabric_handle));

    HsaHandleImportFlags hflags = {};
    HsaHandleImportResult res = {};

    HSAKMT_STATUS status = HSAKMT_CALL(hsaKmtHandleImport(&desc, &res, &hflags));
    if (status != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

    handle->handle = reinterpret_cast<uint64_t>(res.buf_handle);
    rocr::os::DmaBufClose(res.dmabuf_fd);
    handle->size = res.alloc_size;
    return HSA_STATUS_SUCCESS;
  }
  default:
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

hsa_status_t KfdDriver::Map(const core::DriverMemoryHandle& handle, void* mem, size_t offset, size_t size,
                            hsa_access_permission_t perms, uint32_t node_id) {
  HsaMemoryObjectHandle memhandle = reinterpret_cast<HsaMemoryObjectHandle>(handle.handle);
  HSAKMT_STATUS status = HSAKMT_CALL(hsaKmtMemoryVaMap(memhandle,
                                     static_cast<HSAuint64>(offset),
                                     static_cast<HSAuint64>(size), reinterpret_cast<HSAuint64>(mem),
                                     mem_perm(perms), node_id));
  if (status != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::Unmap(const core::DriverMemoryHandle& handle, void *mem,
                              size_t offset, size_t size, uint32_t node_id) {
  HsaMemoryObjectHandle memhandle = reinterpret_cast<HsaMemoryObjectHandle>(handle.handle);
  HSAKMT_STATUS status = HSAKMT_CALL(hsaKmtMemoryVaUnmap(memhandle,
                                     static_cast<HSAuint64>(offset),
                                     static_cast<HSAuint64>(size), reinterpret_cast<HSAuint64>(mem), node_id));
  if (status != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::CreateShareableHandle(void* va, void* mem, size_t size,
                                              const core::Agent& agent,
                                              core::DriverMemoryHandle* handle, uint64_t* offset) {
  // Create handle by exporting and importing the memory from the owning agent.
  (void)va;

  int source_fd = -1;

  /*
   * On Linux, export via KFD first (EXPORT_MEMORY_FLAGS_KFD_DMABUF) so the KFD section of the
   * AMDGPU driver has a BO entry, then import into DRM. The shareable dmabuf_fd itself is created
   * lazily when access is set (see Runtime::VMemorySetAccessPerHandle), so it is not re-exported
   * here. On Windows, KFD export and DRM export are equivalent.
   */

  core::DriverMemoryHandle kfd_alloc = {};
  kfd_alloc.handle = reinterpret_cast<uint64_t>(mem);
  kfd_alloc.size = size;
  if (ExportMemoryHandle(agent, kfd_alloc, core::ShareType::DMABUF_FD,
                         core::EXPORT_MEMORY_FLAGS_KFD_DMABUF, &source_fd, offset) !=
      HSA_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }

  core::DriverMemoryHandle source_handle = {};
  source_handle.dmabuf_fd = source_fd;

  core::DriverMemoryHandle targetHandle = {};
  hsa_status_t ret = ImportMemoryHandle(agent, &targetHandle, core::ShareType::DMABUF_FD,
                                        &source_handle, mem);
#if defined(__linux__)
  rocr::os::DmaBufClose(source_fd);
#endif
  if (ret != HSA_STATUS_SUCCESS)
    return ret;
  assert(targetHandle.size == size);

#if defined(__linux__)
  /*
   * We converted mem into a driver handle. The driver handle will keep the reference count
   * inside the KMD so we can free the original KFD allocation.
   */
  if (HSAKMT_CALL(hsaKmtFreeMemory(mem, size)) != HSAKMT_STATUS_SUCCESS) {
    DestroyMemoryHandle(&targetHandle);
    return HSA_STATUS_ERROR;
  }
#endif

  const auto devhandle = static_cast<const GpuAgent&>(agent).libThunkDev();
  const auto memhandle = reinterpret_cast<HsaMemoryObjectHandle>(targetHandle.handle);
  if (HSAKMT_CALL(hsaKmtMemoryGetCpuAddr(devhandle, memhandle, &handle->mmap_offset)) != HSAKMT_STATUS_SUCCESS) {
    DestroyMemoryHandle(&targetHandle);
    return HSA_STATUS_ERROR;
  }

  handle->handle = targetHandle.handle;
  /*
   * Do not hold a shareable dmabuf_fd open for the lifetime of the handle. It is created lazily
   * (and closed again) when access is set in Runtime::VMemorySetAccessPerHandle.
   */
  handle->dmabuf_fd = -1;
  handle->size = size;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::DestroyMemoryHandle(core::DriverMemoryHandle* handle) {
  if (handle->dmabuf_fd >= 0) {
    hsa_status_t status = rocr::os::DmaBufClose(handle->dmabuf_fd);
    handle->dmabuf_fd = -1;
    if (status != HSA_STATUS_SUCCESS) return status;
  }

  auto memhandle = reinterpret_cast<HsaMemoryObjectHandle>(handle->handle);
  if (memhandle != nullptr) {
    HSAKMT_STATUS status = HSAKMT_CALL(hsaKmtMemHandleFree(memhandle));
    if (status != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }
  }
  *handle = {};
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::SPMAcquire(uint32_t preferred_node_id) const {
  if (HSAKMT_CALL(hsaKmtSPMAcquire(preferred_node_id)) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::SPMRelease(uint32_t preferred_node_id) const {
  if (HSAKMT_CALL(hsaKmtSPMRelease(preferred_node_id)) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::SPMSetDestBuffer(uint32_t preferred_node_id, uint32_t size_bytes,
                                         uint32_t* timeout, uint32_t* size_copied,
                                         void* dest_mem_addr, bool* is_spm_data_loss) const {
  if (HSAKMT_CALL(hsaKmtSPMSetDestBuffer(preferred_node_id, size_bytes, timeout, size_copied, dest_mem_addr,
                             is_spm_data_loss)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::OpenSMI(uint32_t node_id, int* fd) const {
  if (HSAKMT_CALL(hsaKmtOpenSMI(node_id, fd)) != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::ImportExternalSemaphore(uint32_t node_id, void* nt_handle,
                                                hsa_amd_external_semaphore_handle_type_t type,
                                                hsa_amd_external_semaphore_t* out_sem) const {
  // hsa_amd_external_semaphore_handle_type_t maps 1:1 to
  // HSA_EXTERNAL_SEMAPHORE_HANDLE_TYPE by design (see hsa_ext_amd.h).
  HSA_EXTERNAL_SEMAPHORE_HANDLE_TYPE kmt_type =
      static_cast<HSA_EXTERNAL_SEMAPHORE_HANDLE_TYPE>(type);

  HSA_EXTERNAL_SEMAPHORE_HANDLE kmt_handle = {};
  // Require both thunks up front: importing without destroy would leak the
  // handle. Missing either -> NOT_SUPPORTED, not a null call.
  auto* thunk_loader = core::Runtime::runtime_singleton_->thunkLoader();
  const bool loaded =
      thunk_loader->HSAKMT_PFN(hsaKmtImportExternalSemaphore) != nullptr &&
      thunk_loader->HSAKMT_PFN(hsaKmtDestroyExternalSemaphore) != nullptr;
  HSAKMT_STATUS s =
      loaded ? HSAKMT_CALL(hsaKmtImportExternalSemaphore(node_id, nt_handle, kmt_type, &kmt_handle))
             : HSAKMT_STATUS_NOT_SUPPORTED;

  // libhsakmt distinguishes invalid input (null handle, unknown type)
  // from "no node for this agent" and from generic KMD failures.
  // Surface those distinctions to the public API instead of folding
  // every non-success code into HSA_STATUS_ERROR.
  switch (s) {
    case HSAKMT_STATUS_SUCCESS:
      break;
    case HSAKMT_STATUS_INVALID_PARAMETER:  // e.g. null nt_handle
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    case HSAKMT_STATUS_NOT_SUPPORTED:      // missing thunk / unsupported type / Linux stub
      return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
    case HSAKMT_STATUS_INVALID_NODE_UNIT:  // no WDDM device for node
      return HSA_STATUS_ERROR_INVALID_AGENT;
    default:
      return HSA_STATUS_ERROR;
  }

  out_sem->handle = kmt_handle.handle;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::DestroyExternalSemaphore(hsa_amd_external_semaphore_t sem) const {
  HSA_EXTERNAL_SEMAPHORE_HANDLE kmt_handle = {sem.handle};
  // No export -> not this driver's handle. INVALID_AGENT (base-class
  // contract) lets handle_close keep polling other drivers.
  if (core::Runtime::runtime_singleton_->thunkLoader()->HSAKMT_PFN(hsaKmtDestroyExternalSemaphore) == nullptr)
    return HSA_STATUS_ERROR_INVALID_AGENT;
  if (HSAKMT_CALL(hsaKmtDestroyExternalSemaphore(kmt_handle)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

namespace {
// Preserve the libhsakmt distinctions a caller can act on (bad handle vs.
// wrong node vs. generic failure) instead of folding everything into ERROR.
hsa_status_t MapQueueExtSemStatus(HSAKMT_STATUS s) {
  switch (s) {
    case HSAKMT_STATUS_SUCCESS:
      return HSA_STATUS_SUCCESS;
    case HSAKMT_STATUS_INVALID_HANDLE:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    case HSAKMT_STATUS_INVALID_NODE_UNIT:
      return HSA_STATUS_ERROR_INVALID_AGENT;
    case HSAKMT_STATUS_NOT_SUPPORTED:  // missing thunk / platform stub
      return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
    default:
      return HSA_STATUS_ERROR;
  }
}
}  // namespace

hsa_status_t KfdDriver::SignalExternalSemaphore(uint64_t queue_id,
                                                hsa_amd_external_semaphore_t sem,
                                                uint64_t value) const {
  HSA_EXTERNAL_SEMAPHORE_HANDLE kmt_handle = {sem.handle};
  // Optional thunk: missing export maps to NOT_SUPPORTED, not a null call.
  if (core::Runtime::runtime_singleton_->thunkLoader()->HSAKMT_PFN(hsaKmtQueueSignalExternalSemaphore) == nullptr)
    return MapQueueExtSemStatus(HSAKMT_STATUS_NOT_SUPPORTED);
  return MapQueueExtSemStatus(
      HSAKMT_CALL(hsaKmtQueueSignalExternalSemaphore(queue_id, kmt_handle, value)));
}

hsa_status_t KfdDriver::WaitExternalSemaphore(uint64_t queue_id,
                                              hsa_amd_external_semaphore_t sem,
                                              uint64_t value) const {
  HSA_EXTERNAL_SEMAPHORE_HANDLE kmt_handle = {sem.handle};
  // Optional thunk: missing export maps to NOT_SUPPORTED, not a null call.
  if (core::Runtime::runtime_singleton_->thunkLoader()->HSAKMT_PFN(hsaKmtQueueWaitExternalSemaphore) == nullptr)
    return MapQueueExtSemStatus(HSAKMT_STATUS_NOT_SUPPORTED);
  return MapQueueExtSemStatus(
      HSAKMT_CALL(hsaKmtQueueWaitExternalSemaphore(queue_id, kmt_handle, value)));
}

bool KfdDriver::BindXnackMode() {
  // Get users' preference for Xnack mode of ROCm platform.
  HSAint32 mode = core::Runtime::runtime_singleton_->flag().xnack();
  bool config_xnack = (mode != Flag::XNACK_REQUEST::XNACK_UNCHANGED);

  // Indicate to driver users' preference for Xnack mode
  // Call to driver can fail and is a supported feature
  HSAKMT_STATUS status = HSAKMT_STATUS_ERROR;
  if (config_xnack) {
    status = HSAKMT_CALL(hsaKmtSetXNACKMode(mode));
    if (status == HSAKMT_STATUS_SUCCESS) {
      return (mode != Flag::XNACK_DISABLE);
    }
  }

  // Get Xnack mode of devices bound by driver. This could happen
  // when a call to SET Xnack mode fails or user has no particular
  // preference
  status = HSAKMT_CALL(hsaKmtGetXNACKMode(&mode));
  if (status != HSAKMT_STATUS_SUCCESS) {
    debug_print(
        "KFD does not support xnack mode query.\nROCr must assume "
        "xnack is disabled.\n");
    return false;
  }
  return (mode != Flag::XNACK_DISABLE);
}

hsa_status_t KfdDriver::SetTrapHandler(uint32_t node_id, const void* base, uint64_t base_size,
                                       const void* buffer_base, uint64_t buffer_base_size) const {
  if (HSAKMT_CALL(hsaKmtSetTrapHandler(node_id, const_cast<void*>(base), base_size,
                                       const_cast<void*>(buffer_base), buffer_base_size)) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::SetSigbusDelay(uint32_t node_id, uint32_t delay_ms) const {
  if (HSAKMT_CALL(hsaKmtSetSigbusDelay(node_id, delay_ms)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::AllocateScratchMemory(uint32_t node_id, uint64_t size, void** mem) const {
  assert(mem);
  assert(size > 0);

  HsaMemFlags flags = {};
  flags.ui32.Scratch = 1;
  flags.ui32.HostAccess = 1;

  *mem = nullptr;
  auto status = HSAKMT_CALL(hsaKmtAllocMemory(node_id, size, flags, mem));
  if (status != HSAKMT_STATUS_SUCCESS || *mem == nullptr) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::GetDeviceHandle(uint32_t node_id, void** device_handle) const {
  assert(device_handle);

  if (HSAKMT_CALL(hsaKmtGetAMDGPUDeviceHandle(node_id, reinterpret_cast<HsaAMDGPUDeviceHandle*>(device_handle))) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::GetDeviceFd(uint32_t node_id, int *fd) const {
  HsaAMDGPUDeviceHandle device_handle;

  if (HSAKMT_CALL(hsaKmtGetAMDGPUDeviceHandle(node_id, &device_handle)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  if (HSAKMT_CALL(hsaKmtGetAmdGPUDeviceFd(device_handle, fd)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::GetClockCounters(uint32_t node_id, HsaClockCounters* clock_counter) const {
  assert(clock_counter);

  if (HSAKMT_CALL(hsaKmtGetClockCounters(node_id, clock_counter)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::GetTileConfig(uint32_t node_id, HsaGpuTileConfig* config) const {
  assert(config);

  if (HSAKMT_CALL(hsaKmtGetTileConfig(node_id, config)) != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::AvailableMemory(uint32_t node_id, uint64_t* available_size) const {
  assert(available_size);

  if (HSAKMT_CALL(hsaKmtAvailableMemory(node_id, available_size)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::RegisterMemory(void* ptr, uint64_t size, HsaMemFlags mem_flags) const {
  assert(ptr);
  assert(size > 0);

  if (HSAKMT_CALL(hsaKmtRegisterMemoryWithFlags(ptr, size, mem_flags)) != HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::DeregisterMemory(void* ptr) const {
  if (HSAKMT_CALL(hsaKmtDeregisterMemory(ptr)) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::MakeMemoryResident(const void* mem, size_t size, uint64_t* alternate_va,
                                           const HsaMemMapFlags* mem_flags, uint32_t num_nodes,
                                           const uint32_t* nodes) const {
  if (mem_flags == nullptr && nodes == nullptr) {
    if (HSAKMT_CALL(hsaKmtMapMemoryToGPU(const_cast<void*>(mem), size, alternate_va)) !=
        HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }
  } else if (mem_flags != nullptr && nodes != nullptr) {
    if (HSAKMT_CALL(hsaKmtMapMemoryToGPUNodes(const_cast<void*>(mem), size, alternate_va,
                                              *mem_flags, num_nodes, const_cast<uint32_t *>(nodes))) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }
  } else {
    debug_print("Invalid memory flags ptr:%p nodes ptr:%p\n", mem_flags, nodes);
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::MakeMemoryUnresident(const void* mem) const {
  HSAKMT_CALL(hsaKmtUnmapMemoryToGPU(const_cast<void*>(mem)));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::IsModelEnabled(bool* enable) const {
  // AIE does not support streaming performance monitor.
  HSAKMT_STATUS status = HSAKMT_STATUS_ERROR;
  status = HSAKMT_CALL(hsaKmtModelEnabled(enable));
  if (status != HSAKMT_STATUS_SUCCESS)
     return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::GetWallclockFrequency(uint32_t node_id, uint64_t* frequency) const {
  assert(frequency);

  HSAKMT_STATUS status = HSAKMT_CALL(hsaKmtGetNodeWallclockFrequency(node_id, frequency));
  if (status != HSAKMT_STATUS_SUCCESS)
     return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::GetQueueSaveAreaInfo(HSA_QUEUEID queue_id, void** address, size_t* size) const {
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

} // namespace AMD
} // namespace rocr
