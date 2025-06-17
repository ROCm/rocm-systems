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

#include "core/inc/amd_kfd_driver.h"

#include <memory>
#include <string>

#include <amdgpu_drm.h>
#include <link.h>
#include <sys/ioctl.h>

#include "hsakmt/hsakmt.h"

#include "core/inc/amd_gpu_agent.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/runtime.h"

extern r_debug _amdgpu_r_debug;

namespace rocr {
namespace AMD {

static_assert(
    (sizeof(core::ShareableHandle::handle) >= sizeof(amdgpu_bo_handle)) &&
        (alignof(core::ShareableHandle::handle) >= alignof(amdgpu_bo_handle)),
    "ShareableHandle cannot store a amdgpu_bo_handle");

namespace {

__forceinline uint64_t drm_perm(hsa_access_permission_t perm) {
  switch (perm) {
  case HSA_ACCESS_PERMISSION_RO:
    return AMDGPU_VM_PAGE_READABLE;
  case HSA_ACCESS_PERMISSION_WO:
    return AMDGPU_VM_PAGE_WRITEABLE;
  case HSA_ACCESS_PERMISSION_RW:
    return AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  case HSA_ACCESS_PERMISSION_NONE:
  default:
    return 0;
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
  if (ret != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

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
  if (HSAKMT_CALL(hsaKmtReleaseSystemProperties()) != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

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

hsa_status_t KfdDriver::FreeMemory(void *mem, size_t size) {
  MakeKfdMemoryUnresident(mem);
  return FreeKfdMemory(mem, size) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

hsa_status_t KfdDriver::CreateQueue(uint32_t node_id, HSA_QUEUE_TYPE type, uint32_t queue_pct,
                                    HSA_QUEUE_PRIORITY priority, uint32_t sdma_engine_id,
                                    void* queue_addr, uint64_t queue_size_bytes, HsaEvent* event,
                                    HsaQueueResource& queue_resource) const {
  if (HSAKMT_CALL(hsaKmtCreateQueueExt(node_id, type, queue_pct, priority, sdma_engine_id,
                                       queue_addr, queue_size_bytes, event, &queue_resource)) !=
      HSAKMT_STATUS_SUCCESS) {
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
                                    HSA_QUEUE_PRIORITY priority, void* queue_addr,
                                    uint64_t queue_size, HsaEvent* event) const {
  if (HSAKMT_CALL(hsaKmtUpdateQueue(queue_id, queue_pct, priority, queue_addr, queue_size,
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

hsa_status_t KfdDriver::ExportDMABuf(void *mem, size_t size, int *dmabuf_fd,
                                     size_t *offset) {
  int dmabuf_fd_res = -1;
  size_t offset_res = 0;
  if (HSAKMT_CALL(hsaKmtExportDMABufHandle(mem, size, &dmabuf_fd_res, &offset_res)) !=
      HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  *dmabuf_fd = dmabuf_fd_res;
  *offset = offset_res;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::ImportDMABuf(int dmabuf_fd, core::Agent &agent,
                                     core::ShareableHandle &handle) {
  auto &gpu_agent = static_cast<GpuAgent &>(agent);
  amdgpu_bo_import_result res;
  auto ret = DRM_CALL(amdgpu_bo_import(
      gpu_agent.libDrmDev(), amdgpu_bo_handle_type_dma_buf_fd, dmabuf_fd, &res));
  if (ret)
    return HSA_STATUS_ERROR;

  handle.handle = reinterpret_cast<uint64_t>(res.buf_handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::Map(core::ShareableHandle handle, void *mem,
                            size_t offset, size_t size,
                            hsa_access_permission_t perms) {
  const auto ldrm_bo = reinterpret_cast<amdgpu_bo_handle>(handle.handle);
  if (!ldrm_bo)
    return HSA_STATUS_ERROR;

  if (DRM_CALL(amdgpu_bo_va_op(ldrm_bo, offset, size, reinterpret_cast<uint64_t>(mem),
                      drm_perm(perms), AMDGPU_VA_OP_MAP)) != 0)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::Unmap(core::ShareableHandle handle, void *mem,
                              size_t offset, size_t size) {
  const auto ldrm_bo = reinterpret_cast<amdgpu_bo_handle>(handle.handle);
  if (!ldrm_bo)
    return HSA_STATUS_ERROR;

  if (DRM_CALL(amdgpu_bo_va_op(ldrm_bo, offset, size, reinterpret_cast<uint64_t>(mem), 0,
                      AMDGPU_VA_OP_UNMAP)) != 0)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t KfdDriver::ReleaseShareableHandle(core::ShareableHandle &handle) {
  const auto ldrm_bo = reinterpret_cast<amdgpu_bo_handle>(handle.handle);
  if (!ldrm_bo)
    return HSA_STATUS_ERROR;

  const auto ret = DRM_CALL(amdgpu_bo_free(ldrm_bo));
  if (ret)
    return HSA_STATUS_ERROR;

  handle = {};
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

void *KfdDriver::AllocateKfdMemory(const HsaMemFlags &flags, uint32_t node_id,
                                   size_t size) {
  void *mem = nullptr;
  const HSAKMT_STATUS status = HSAKMT_CALL(hsaKmtAllocMemory(node_id, size, flags, &mem));
  return (status == HSAKMT_STATUS_SUCCESS) ? mem : nullptr;
}

bool KfdDriver::FreeKfdMemory(void *mem, size_t size) {
  if (mem == nullptr || size == 0) {
    debug_print("Invalid free ptr:%p size:%lu\n", mem, size);
    return false;
  }

  if (HSAKMT_CALL(hsaKmtFreeMemory(mem, size)) != HSAKMT_STATUS_SUCCESS) {
    debug_print("Failed to free ptr:%p size:%lu\n", mem, size);
    return false;
  }
  return true;
}

bool KfdDriver::MakeKfdMemoryResident(size_t num_node, const uint32_t *nodes,
                                      const void *mem, size_t size,
                                      uint64_t *alternate_va,
                                      HsaMemMapFlags map_flag) {
  assert(num_node > 0);
  assert(nodes);

  *alternate_va = 0;

  HSAKMT_STATUS kmt_status(HSAKMT_CALL(hsaKmtMapMemoryToGPUNodes(
      const_cast<void *>(mem), size, alternate_va, map_flag, num_node,
      const_cast<uint32_t *>(nodes))));

  return (kmt_status == HSAKMT_STATUS_SUCCESS);
}

void KfdDriver::MakeKfdMemoryUnresident(const void *mem) {
  HSAKMT_CALL(hsaKmtUnmapMemoryToGPU(const_cast<void *>(mem)));
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

hsa_status_t KfdDriver::IsModelEnabled(bool* enable) const {
  // AIE does not support streaming performance monitor.
  HSAKMT_STATUS status = HSAKMT_STATUS_ERROR;
  status = HSAKMT_CALL(hsaKmtModelEnabled(enable));
  if (status != HSAKMT_STATUS_SUCCESS)
     return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

} // namespace AMD
} // namespace rocr
