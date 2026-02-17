/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <sys/stat.h>
#include <cinttypes>
#include <cassert>
#include "impl/wddm/gpu_memory.h"
#include "impl/wddm/device.h"
#include "util/utils.h"

using namespace std;

namespace wsl {
namespace thunk {

size_t GpuMemory::CalcChunkNumbers(gpusize size) {
  const auto chunk_size = WDDMDevice::GpuMemoryChunkSize;
  return (size + chunk_size - 1) / chunk_size;
}

gpusize GpuMemory::AdjustSize(gpusize size) const {
  const auto &device_info = device_->DeviceInfo();

  if (device_info.enable_big_page_alignment && desc_.domain == thunk_proxy::kLocal) {
    uint32_t alignment = device_info.big_page_alignment_size;
    // BigPage is only supported for allocations > bigPageMinAlignment.
    // Also, if bigPageMinAlignment == 0, BigPage optimization is not supported per KMD.
    // We do either LargePage or BigPage alignment, whichever has a higher value.
    if ((device_info.hw_big_page_min_alignment_size > 0) && (size > device_info.hw_big_page_min_alignment_size)) {
      alignment = std::max(alignment, device_info.hw_big_page_min_alignment_size);
      if (size > device_info.hw_big_page_alignment_size)
        alignment = std::max(alignment, device_info.hw_big_page_alignment_size);
    }
    if (alignment > 0)
      size = rocr::AlignUp(size, alignment);
  } else {
    const size_t min_size = 4096;
    size = rocr::AlignUp(size, min_size);
  }
  return size;
}

GpuMemory::GpuMemory(WDDMDevice *device) : device_(device) {
  num_allocations_ = 0;
  // alloc_handle_ is intentionally zero-initialized so that alloc_handles_ptr_
  // can safely point to it as a default, always-valid single allocation handle.
  alloc_handle_ = 0;
  alloc_handles_ptr_ = &alloc_handle_;
  resource_ = 0;
  mem_fd_ = -1;
}

GpuMemory::~GpuMemory() {
  FreeGpuVirtualAddress(GpuAddress(), Size());
  FreePhysicalMemory();
  if (desc_.handle_ape_addr > 0)
    dxg_runtime->HandleApertureFree(desc_.handle_ape_addr);
}

ErrorCode GpuMemory::Init(const GpuMemoryCreateInfo &create_info) {
  desc_.domain = create_info.domain;
  desc_.adapter_luid = device_->GetLuid();
  desc_.client_size = create_info.size;
  desc_.alignment = create_info.alignment;
  desc_.mem_flags = create_info.mem_flags;
  desc_.engine_flag = create_info.engine_flag;
  desc_.flags.is_virtual = create_info.flags.virtual_alloc;
  desc_.flags.is_physical_only = create_info.flags.physical_only;
  desc_.flags.is_physical_contiguous = create_info.flags.physical_contiguous;
  desc_.flags.is_imported_sys_memfd = create_info.flags.sysmem_ipc_sig_importer;
  desc_.flags.is_sysmem_exporter = create_info.flags.sysmem_ipc_sig_exporter;
  desc_.flags.is_va_required = create_info.flags.alloc_va;
  desc_.flags.is_blit_kernel_object = create_info.flags.blit_kernel_object;

  /* we can't tell the allocation is regular vmm or ipc mem at creation stage,
     they share same creation parameters, so forcing all vram allocations to
     sharable to support IPC mem */
  if (create_info.flags.interprocess ||
      desc_.domain == thunk_proxy::AllocDomain::kLocal)
    desc_.flags.is_shared = true;

  desc_.flags.is_locked = create_info.flags.locked;
  desc_.size = AdjustSize(desc_.client_size);

  if (IsUserMemory() || IsSystem())
    desc_.cpu_addr = create_info.user_ptr;

  num_allocations_ = CalcChunkNumbers(Size());
  if (num_allocations_ > 1)
    alloc_handles_ptr_ = new WinAllocationHandle[num_allocations_];

  memset(alloc_handles_ptr_, 0, num_allocations_ * sizeof(WinAllocationHandle));

  auto code = ErrorCode::Success;

  if (IsPhysicalOnly()) {
    code = dxg_runtime->HandleApertureAlloc(desc_.size, &desc_.handle_ape_addr);
    return code;
  }

  code = ReserveGpuVirtualAddress(create_info.va_hint, Size(), create_info.alignment);
  if (IsVirtual() || (code != ErrorCode::Success))
      return code;

  auto guard = rocr::MakeScopeGuard([this, &code]() {
    if (code != ErrorCode::Success) {

      if (is_phymem_created) {
        FreePhysicalMemory();
      }
      FreeGpuVirtualAddress(GpuAddress(), Size());
    }
  });
  (void)guard;

  code = CreatePhysicalMemory();
  if (code != ErrorCode::Success)
    return code;

  code = MapGpuVirtualAddress(GpuAddress(), Size());
  if (code != ErrorCode::Success)
    return code;

  code = MakeResident();
  if (code != ErrorCode::Success)
    return code;

  if (!GetDevice()->WaitOnPagingFenceFromCpu())
    code = ErrorCode::Unknown;

  return code;
}
// ================================================================================================
ErrorCode GpuMemory::MapMemoryToVirtualAddress(bool create_phys_mem) {
  auto code = ErrorCode::Success;
  auto guard = rocr::MakeScopeGuard([this, &code]() {
    if (code != ErrorCode::Success) {
      if (is_phymem_created) {
        FreePhysicalMemory();
      }
    }
  });
  (void)guard;
  // Physical mapping it's no longer a virtual address allocation
  desc_.flags.is_virtual = 0;
  if (create_phys_mem) {
    code = CreatePhysicalMemory();
    if (code != ErrorCode::Success) return code;
  }

  code = MapGpuVirtualAddress(GpuAddress(), Size());
  if (code != ErrorCode::Success) return code;

  code = MakeResident();
  if (code != ErrorCode::Success) return code;

  if (!GetDevice()->WaitOnPagingFenceFromCpu()) code = ErrorCode::Unknown;
  return code;
}

// ================================================================================================
ErrorCode GpuMemory::UnmapGpuVirtualAddress(const gpusize addr, const gpusize size, gpusize offset) {
  auto code = ErrorCode::Success;
  size_t i = 0;
  auto map_addr = addr;
  auto map_size = size;

  while (offset >= WDDMDevice::GpuMemoryChunkSize) {
    offset -= WDDMDevice::GpuMemoryChunkSize;
    i += 1;
  }

  while (map_size > 0) {
    auto block_size = std::min(map_size, WDDMDevice::GpuMemoryChunkSize);

    D3DDDI_MAPGPUVIRTUALADDRESS args{};

    args.hPagingQueue = device_->PagingQueue();
    args.BaseAddress = map_addr;
    args.hAllocation = GetAllocationHandle(i);
    args.SizeInPages = block_size / 0x1000;
    args.Protection.NoAccess = 1;

    code = d3dthunk::MapGpuVirtualAddress(&args);

    if (code == ErrorCode::NotReady)
      device_->UpdatePageFence(args.PagingFenceValue);
    else if (code != ErrorCode::Success)
      break;

    map_addr += block_size;
    map_size -= block_size;
    offset = 0;   // reset second unmapped allocation offset to zero
    i += 1;
  }

  if (dxg_runtime->hsakmt_debug_sysmem && code == ErrorCode::Success) {
    SetGpuAddress(0);
    SetCpuAddress(nullptr);
  }

  return code;
}

ErrorCode GpuMemory::MapGpuVirtualAddress(const gpusize addr, const gpusize size, gpusize offset) {

  auto code = ErrorCode::Success;
  size_t i = 0;
  auto map_addr = addr;
  auto map_size = size;
  const size_t _4K = 0x1000;

  while (offset >= WDDMDevice::GpuMemoryChunkSize) {
    offset -= WDDMDevice::GpuMemoryChunkSize;
    i += 1;
  }
  const size_t first_chunk = i;
  const auto first_chunk_offset = offset;
  /* Found two limitation for local vram:
   * 1. invisible vram va has to be 64K aligned, otherwise map gpu va fail
   * 2. visible vram can not be cpu mapped when command submission or after gpu mapped
   */
  while (map_size > 0) {
    auto block_size = std::min(map_size, WDDMDevice::GpuMemoryChunkSize);

    D3DDDI_MAPGPUVIRTUALADDRESS args{};

    args.hPagingQueue = device_->PagingQueue();
    args.BaseAddress = map_addr;
    args.hAllocation = GetAllocationHandle(i);
    args.OffsetInPages = offset / _4K;
    args.SizeInPages = block_size / _4K;
    args.Protection.Write = 1;

    code = d3dthunk::MapGpuVirtualAddress(&args);

    if (code != ErrorCode::Success) {
      if (code == ErrorCode::NotReady) {
        const uint64_t fence_value = args.PagingFenceValue;
        device_->UpdatePageFence(fence_value);
        code = ErrorCode::Success;
      } else
        break;
    }

    map_addr += block_size;
    map_size -= block_size;
    offset = 0;  // reset second mapped allocation offset to zero
    i++;
  }

  if (code != ErrorCode::Success) {
    // Map failed, unmap partial mapped block
    offset = first_chunk_offset;
    map_addr = addr;
    map_size = size;
    for (size_t j = first_chunk; j < i; j++) {
      auto block_size = std::min(map_size, WDDMDevice::GpuMemoryChunkSize);

      D3DDDI_MAPGPUVIRTUALADDRESS args{};

      args.hPagingQueue = device_->PagingQueue();
      args.BaseAddress = map_addr;
      args.hAllocation = 0;
      args.OffsetInPages = offset / _4K;
      args.SizeInPages = block_size / _4K;
      args.Protection.NoAccess = 1;

      auto unmap_code = d3dthunk::MapGpuVirtualAddress(&args);
      if (unmap_code == ErrorCode::NotReady)
        device_->UpdatePageFence(args.PagingFenceValue);

      map_addr += block_size;
      map_size -= block_size;
    }
  }

  return code;
}

ErrorCode GpuMemory::ReserveGpuVirtualAddress(gpusize base_virt_addr, gpusize size, gpusize alignment) {
  ErrorCode status;
  gpusize gpu_virt_addr = 0;
  if ((desc_.flags.is_sysmem_exporter || desc_.flags.is_imported_sys_memfd)
      && desc_.domain == thunk_proxy::AllocDomain::kSystem) {
    int mfd = (mem_fd_ > -1)? mem_fd_ : -1;
    status = dxg_runtime->ReserveIPCSysMem(Size(), &gpu_virt_addr, desc_.alignment, mfd, desc_.flags.is_locked);
    if (status == ErrorCode::Success)
      mem_fd_ = mfd;
  } else {
    status = dxg_runtime->ReserveGpuVirtualAddress(desc_.domain, base_virt_addr, size, &gpu_virt_addr, alignment,
        desc_.flags.is_locked);
  }

  if (status == ErrorCode::Success) {
    desc_.gpu_addr = gpu_virt_addr;

    if (IsSystem())
      desc_.cpu_addr = reinterpret_cast<void *>(desc_.gpu_addr);
  }
  return status;
}

ErrorCode GpuMemory::FreeGpuVirtualAddress(gpusize base_addr, gpusize size) {
  if (mem_fd_ > -1)
    return dxg_runtime->FreeIPCSysMem(GpuAddress(), Size(), mem_fd_);

  return base_addr != 0 ?
         dxg_runtime->FreeGpuVirtualAddress(desc_.domain, base_addr, size) :
         ErrorCode::Success;
}

ErrorCode GpuMemory::CreatePhysicalMemory() {

  assert(!IsVirtual() && NumChunks() > 0);

  const auto num_allocations = NumChunks();
  void *priv_drv_data;
  void *priv_alloc_data;
  int priv_drv_data_size;
  int priv_alloc_data_size;

  thunk_proxy::GetAllocPrivDataSize(&priv_drv_data_size, &priv_alloc_data_size);
  int total_size = priv_drv_data_size +
    num_allocations * priv_alloc_data_size +
    num_allocations * sizeof(D3DDDI_ALLOCATIONINFO2);
  priv_drv_data = malloc(total_size);
  if (!priv_drv_data)
    return ErrorCode::OutOfMemory;

  memset(priv_drv_data, 0, total_size);
  thunk_proxy::FillinAllocPrivDrvData(priv_drv_data, priv_alloc_data_size);

  priv_alloc_data = static_cast<unsigned char*>(priv_drv_data) + priv_drv_data_size;
  auto alloc_info = reinterpret_cast<D3DDDI_ALLOCATIONINFO2*>(
       static_cast<unsigned char*>(priv_alloc_data) + priv_alloc_data_size * num_allocations);

  size_t size = desc_.size;
  uint64_t addr = desc_.gpu_addr;
  char *cpu_addr = static_cast<char *>(desc_.cpu_addr);
  const auto &device_info = GetDevice()->DeviceInfo();

  for (size_t i = 0; i < num_allocations; i++) {

    void* priv_data = (void*)((char*)priv_alloc_data + priv_alloc_data_size * i);
    size_t block_size = std::min(size, WDDMDevice::GpuMemoryChunkSize);

    if (IsUserMemory() || IsSystem()) {
      thunk_proxy::SetAllocationInfo(priv_data, block_size, desc_.domain, 0, desc_.mem_flags, desc_.engine_flag, device_info);
      alloc_info[i].pSystemMem = static_cast<void *>(cpu_addr);
      cpu_addr += block_size;
    } else {
      thunk_proxy::SetAllocationInfo(priv_data, block_size, desc_.domain, addr, desc_.mem_flags, desc_.engine_flag, device_info);
    }

    size -= block_size;
    addr += block_size;

    alloc_info[i].pPrivateDriverData = priv_data;
    alloc_info[i].PrivateDriverDataSize = priv_alloc_data_size;
    alloc_info[i].VidPnSourceId = D3DDDI_ID_UNINITIALIZED;
  }

  D3DKMT_CREATEALLOCATION args = {};
  args.hDevice = device_->DeviceHandle();
  args.pPrivateDriverData = priv_drv_data;
  args.PrivateDriverDataSize = priv_drv_data_size;
  args.NumAllocations = num_allocations;
  args.pAllocationInfo2 = alloc_info;

  /* The PhysicallyContiguous flag causes allocation failure
   * args.Flags.PhysicallyContiguous = IsPhysicalContiguous();
   */

  SharedHandleInfo shared_info;
  if (IsShared()) {
    shared_info.size = desc_.size;
    shared_info.client_size = desc_.client_size;
    shared_info.domain = desc_.domain;
    shared_info.adapter_luid = desc_.adapter_luid;
    shared_info.flags = static_cast<uint32_t>(desc_.flags.reserved);
    shared_info.mem_flags = desc_.mem_flags;
    shared_info.pid = dxg_runtime->parent_pid;
    shared_info.gpu_addr = desc_.gpu_addr;
    args.pPrivateRuntimeData = &shared_info;
    args.PrivateRuntimeDataSize = sizeof(shared_info);
    args.Flags.NtSecuritySharing = 1;
    args.Flags.CreateShared = 1;
    args.Flags.CreateResource = 1;
  }

  auto status = d3dthunk::CreateAllocation(&args);
  if (status == ErrorCode::Success) {
    for (size_t i = 0; i < num_allocations; i++)
      alloc_handles_ptr_[i] = alloc_info[i].hAllocation;

    resource_ = args.hResource;
    is_phymem_created = 1;
  }
  free(priv_drv_data);
  return status;
}

ErrorCode GpuMemory::FreePhysicalMemory() {
  auto code = ErrorCode::Success;

  if (resource_ || (NumChunks() > 0 && alloc_handles_ptr_ != nullptr && *alloc_handles_ptr_ != 0))
    code = d3dthunk::DestroyAllocation(device_->DeviceHandle(), resource_, NumChunks(),
                                       alloc_handles_ptr_);

  if (alloc_handles_ptr_ != &alloc_handle_) delete[] alloc_handles_ptr_;

  alloc_handle_ = 0;
  alloc_handles_ptr_ = &alloc_handle_;

  is_phymem_created = 0;
  return code;
}

ErrorCode GpuMemory::MakeResident() {

  D3DDDI_MAKERESIDENT args = {};
  args.hPagingQueue = device_->PagingQueue();
  args.NumAllocations = NumChunks();
  args.AllocationList = alloc_handles_ptr_;
  args.Flags.CantTrimFurther = 1;

  auto code = d3dthunk::MakeResident(&args);
  if (code == ErrorCode::NotReady) {
    const auto fence_value = args.PagingFenceValue;
    device_->UpdatePageFence(fence_value);
    code = ErrorCode::Success;
  }
  return code;
}

ErrorCode GpuMemory::Evict() {

  D3DKMT_EVICT args = {};
  args.hDevice = device_->DeviceHandle();
  args.NumAllocations = NumChunks();
  args.AllocationList = alloc_handles_ptr_;

  return d3dthunk::Evict(&args);
}

ErrorCode GpuMemory::OpenResourceFromKMTHandle(D3DKMT_HANDLE buffer_handle,
                                               D3DKMT_HANDLE device_handle,
                                               D3DKMT_OPENRESOURCE** out_open_resource) {
#if defined(WIN32)
  D3DKMT_QUERYRESOURCEINFO query_args{};
  query_args.hDevice = device_handle;
  query_args.hGlobalShare = buffer_handle;
  auto ret = d3dthunk::QueryResourceInfo(&query_args);

  if (ret != ErrorCode::Success) {
    pr_err("query resource info from KMT handle failed %d\n", static_cast<int>(ret));
    return ret;
  }

  const size_t data_size = sizeof(D3DKMT_OPENRESOURCE) + query_args.PrivateRuntimeDataSize +
      query_args.TotalPrivateDriverDataSize + query_args.ResourcePrivateDriverDataSize +
      sizeof(D3DDDI_OPENALLOCATIONINFO) * query_args.NumAllocations +
      thunk_proxy::GetProxyResourceInfoSize();  // for extra room for pTotalPrivateDriverDataBuffer
  D3DKMT_OPENRESOURCE* open_resource = reinterpret_cast<D3DKMT_OPENRESOURCE*>(calloc(1, data_size));

  if (open_resource == nullptr) {
    pr_err("alloc data failed, size=%zu\n", data_size);
    return ErrorCode::OutOfMemory;
  }

  *out_open_resource = reinterpret_cast<D3DKMT_OPENRESOURCE*>(open_resource);

  open_resource->hGlobalShare = query_args.hGlobalShare;
  open_resource->hDevice = query_args.hDevice;
  open_resource->PrivateRuntimeDataSize = query_args.PrivateRuntimeDataSize;
  open_resource->TotalPrivateDriverDataBufferSize = query_args.TotalPrivateDriverDataSize;
  open_resource->ResourcePrivateDriverDataSize = query_args.ResourcePrivateDriverDataSize;
  open_resource->NumAllocations = query_args.NumAllocations;

  size_t offset = sizeof(D3DKMT_OPENRESOURCE);

  if (query_args.PrivateRuntimeDataSize != 0) {
    open_resource->pPrivateRuntimeData = reinterpret_cast<char*>(open_resource) + offset;
    offset += open_resource->PrivateRuntimeDataSize;
  }

  if (query_args.TotalPrivateDriverDataSize != 0) {
    open_resource->pTotalPrivateDriverDataBuffer = reinterpret_cast<char*>(open_resource) + offset;
    offset += open_resource->TotalPrivateDriverDataBufferSize;

    // NOTE: We need to trick the function calls into allocating enough room for the
    // proxy resource info structure, which is tacked onto the end of pTotalPrivateDriverDataBuffer.
    offset += thunk_proxy::GetProxyResourceInfoSize();
  }

  if (query_args.ResourcePrivateDriverDataSize != 0) {
    open_resource->pResourcePrivateDriverData = reinterpret_cast<char*>(open_resource) + offset;
    offset += open_resource->ResourcePrivateDriverDataSize;
  }

  if (query_args.NumAllocations != 0) {
    open_resource->pOpenAllocationInfo = reinterpret_cast<D3DDDI_OPENALLOCATIONINFO*>(
        reinterpret_cast<char*>(open_resource) + offset);
  }

  ret = d3dthunk::OpenResource(open_resource);
  if (ret != ErrorCode::Success) {
    pr_err("open resource info from KMT handle failed %d\n", static_cast<int>(ret));
  }

  return ret;
#else
  assert(!"Unimplemented!");
  return ErrorCode::UnSupported;
#endif
}

ErrorCode GpuMemory::OpenResourceFromNTHandle(HANDLE buffer_handle, D3DKMT_HANDLE device_handle,
                                              D3DKMT_OPENRESOURCEFROMNTHANDLE** out_open_resource) {
  D3DKMT_QUERYRESOURCEINFOFROMNTHANDLE query_args{};
  query_args.hDevice = device_handle;
  query_args.hNtHandle = buffer_handle;
  auto ret = d3dthunk::QueryResourceInfoFromNtHandle(&query_args);

  if (ret != ErrorCode::Success) {
    pr_err("query resource info from NT handle failed %d\n", static_cast<int>(ret));
    return ret;
  }

  const size_t data_size = sizeof(D3DKMT_OPENRESOURCEFROMNTHANDLE) +
      query_args.PrivateRuntimeDataSize + query_args.TotalPrivateDriverDataSize +
      query_args.ResourcePrivateDriverDataSize +
      sizeof(D3DDDI_OPENALLOCATIONINFO2) * query_args.NumAllocations;
  D3DKMT_OPENRESOURCEFROMNTHANDLE* open_resource =
      reinterpret_cast<D3DKMT_OPENRESOURCEFROMNTHANDLE*>(calloc(1, data_size));

  if (open_resource == nullptr) {
    pr_err("alloc data failed, size=%zu\n", data_size);
    return ErrorCode::OutOfMemory;
  }

  *out_open_resource = reinterpret_cast<D3DKMT_OPENRESOURCEFROMNTHANDLE*>(open_resource);

  open_resource->hNtHandle = query_args.hNtHandle;
  open_resource->hDevice = query_args.hDevice;
  open_resource->PrivateRuntimeDataSize = query_args.PrivateRuntimeDataSize;
  open_resource->TotalPrivateDriverDataBufferSize = query_args.TotalPrivateDriverDataSize;
  open_resource->ResourcePrivateDriverDataSize = query_args.ResourcePrivateDriverDataSize;
  open_resource->NumAllocations = query_args.NumAllocations;

  size_t offset = sizeof(D3DKMT_OPENRESOURCEFROMNTHANDLE);

  if (query_args.PrivateRuntimeDataSize != 0) {
    open_resource->pPrivateRuntimeData = reinterpret_cast<char*>(open_resource) + offset;
    offset += open_resource->PrivateRuntimeDataSize;
  }

  if (query_args.TotalPrivateDriverDataSize != 0) {
    open_resource->pTotalPrivateDriverDataBuffer = reinterpret_cast<char*>(open_resource) + offset;
    offset += open_resource->TotalPrivateDriverDataBufferSize;
  }

  if (query_args.ResourcePrivateDriverDataSize != 0) {
    open_resource->pResourcePrivateDriverData = reinterpret_cast<char*>(open_resource) + offset;
    offset += open_resource->ResourcePrivateDriverDataSize;
  }

  if (query_args.NumAllocations != 0) {
    open_resource->pOpenAllocationInfo2 = reinterpret_cast<D3DDDI_OPENALLOCATIONINFO2*>(
        reinterpret_cast<char*>(open_resource) + offset);
  }

  ret = d3dthunk::OpenResourceFromNtHandle(open_resource);
  if (ret != ErrorCode::Success) {
    pr_err("open resource info from NT handle failed %d\n", static_cast<int>(ret));
  }

  return ret;
}

ErrorCode GpuMemory::ExportPhysicalHandle(int* dmabuf_fd, uint32_t flags) {
  if (mem_fd_ > -1) {
    *dmabuf_fd = mem_fd_;
    return ErrorCode::Success;
  }

  if (IsShared())
    return d3dthunk::ShareObjects(1, resource_, flags, dmabuf_fd);
  else
    return ErrorCode::UnSupported;
}

ErrorCode GpuMemory::ImportPhysicalFD(const GpuMemoryCreateInfo& create_info, gpusize* gpu_addr) {
  // the ipc signal sys mem fd will be closed in Runtime::IPCClientImport, dup to hold a reference
#if defined(__linux__)
  mem_fd_ = dup(create_info.dmabuf_fd);
#else
  assert(!"Unimplemented!");
  mem_fd_ = 1;
#endif

  desc_.client_size = create_info.size;
  desc_.size = AdjustSize(desc_.client_size);
  desc_.domain = thunk_proxy::AllocDomain::kSystem;
  desc_.adapter_luid = device_->GetLuid();
  desc_.alignment = 0x1000;
  desc_.mem_flags = create_info.mem_flags;
  desc_.engine_flag = create_info.engine_flag;
  desc_.flags.is_imported_sys_memfd = create_info.flags.sysmem_ipc_sig_importer;
  desc_.flags.is_va_required = create_info.flags.alloc_va;
  desc_.flags.is_virtual = create_info.flags.virtual_alloc;
  desc_.flags.is_physical_only = create_info.flags.physical_only;
  desc_.flags.is_physical_contiguous = create_info.flags.physical_contiguous;
  desc_.flags.is_locked = create_info.flags.locked;

  auto code = ReserveGpuVirtualAddress(create_info.va_hint, Size(), create_info.alignment);
  if (code != ErrorCode::Success) return code;

  bool physical_created = false;
  auto guard = rocr::MakeScopeGuard([this, &physical_created, &code]() {
    if (code != ErrorCode::Success) {
      if (physical_created) FreePhysicalMemory();
      FreeGpuVirtualAddress(GpuAddress(), Size());
    }
  });
  (void)guard;

  num_allocations_ = CalcChunkNumbers(Size());
  if (num_allocations_ > 1) alloc_handles_ptr_ = new WinAllocationHandle[num_allocations_];

  memset(alloc_handles_ptr_, 0, num_allocations_ * sizeof(WinAllocationHandle));

  code = CreatePhysicalMemory();
  if (code != ErrorCode::Success) return code;

  physical_created = true;

  code = MapGpuVirtualAddress(GpuAddress(), Size());
  if (code != ErrorCode::Success) return code;

  code = MakeResident();
  if (code != ErrorCode::Success) return code;

  if (!GetDevice()->WaitOnPagingFenceFromCpu()) code = ErrorCode::Unknown;

  return code;
}

ErrorCode GpuMemory::ImportPhysicalKMTHandle(const GpuMemoryCreateInfo& create_info,
                                             gpusize* gpu_addr) {
#if defined(WIN32)                                            
  D3DKMT_OPENRESOURCE* open_resource = nullptr;
  ErrorCode ret = OpenResourceFromKMTHandle(static_cast<D3DKMT_HANDLE>(create_info.dmabuf_fd),
                                            device_->DeviceHandle(), &open_resource);
  auto guard_data = rocr::MakeScopeGuard([open_resource]() { free(open_resource); });

  if (ret != ErrorCode::Success) {
    pr_err("open resource failed %d\n", static_cast<int>(ret));
    return ErrorCode::InvalidateParams;
  }

  SharedHandleInfo shared_info{};
  SharedHandleInfo* shared_info_ptr = &shared_info;
  if (open_resource->PrivateRuntimeDataSize > 0)
    shared_info_ptr = static_cast<SharedHandleInfo*>(open_resource->pPrivateRuntimeData);

  if (open_resource->NumAllocations > 1)
    alloc_handles_ptr_ = new WinAllocationHandle[open_resource->NumAllocations];

  // Update shared_info_ptr if OpenResourceFromNtHandle skips populating it.
  if (open_resource->PrivateRuntimeDataSize == 0) {
    for (auto alloc_index = 0U; alloc_index < open_resource->NumAllocations; alloc_index++) {
      const auto* const pPrivateDriverData =
          open_resource->pOpenAllocationInfo[alloc_index].pPrivateDriverData;
      auto alloc_size = thunk_proxy::GetMemoryAllocationSize(pPrivateDriverData);
      shared_info_ptr->size += alloc_size;
      shared_info_ptr->client_size += alloc_size;
    }
  }

  if (shared_info_ptr->pid == dxg_runtime->parent_pid && create_info.flags.alloc_va &&
      IsSameAdapter(shared_info_ptr->adapter_luid) && shared_info_ptr->gpu_addr) {
    pr_info(
        "import from same device and same process, va is required. "
        "a buffer can't be mapped to 2 va. delete the imported buffer, use the existing one.\n");
    if (gpu_addr) *gpu_addr = shared_info_ptr->gpu_addr;
    return ErrorCode::SameProcessSameDevice;
  }

  desc_.size = shared_info_ptr->size;
  desc_.client_size = shared_info_ptr->client_size;
  desc_.domain = shared_info_ptr->domain;
  desc_.flags.reserved = shared_info_ptr->flags;
  desc_.mem_flags = shared_info_ptr->mem_flags;
  desc_.adapter_luid = shared_info_ptr->adapter_luid;
  resource_ = open_resource->hResource;
  num_allocations_ = open_resource->NumAllocations;
  for (auto i = 0; i < num_allocations_; i++)
    alloc_handles_ptr_[i] = open_resource->pOpenAllocationInfo[i].hAllocation;

  desc_.flags.is_va_required = create_info.flags.alloc_va;
  if (desc_.flags.is_va_required) {
    desc_.flags.is_imported_vram_ipc = 1;
    ret = ReserveGpuVirtualAddress(create_info.va_hint, desc_.size, create_info.alignment);
    if (ret != ErrorCode::Success)
      pr_err("failed to allocate svm range, error:%d\n", static_cast<int>(ret));

    return ret;
  } else {
    desc_.flags.is_imported_vram_vmem = 1;
    return dxg_runtime->HandleApertureAlloc(desc_.size, &desc_.handle_ape_addr);
  }
#else
  assert(!"Unimplemented!");
  return ErrorCode::UnSupported;
#endif
}

ErrorCode GpuMemory::ImportPhysicalNTHandle(const GpuMemoryCreateInfo& create_info,
                                            gpusize* gpu_addr) {
  D3DKMT_OPENRESOURCEFROMNTHANDLE* open_resource = nullptr;
  ErrorCode ret = OpenResourceFromNTHandle(reinterpret_cast<HANDLE>(create_info.dmabuf_fd),
                                           device_->DeviceHandle(), &open_resource);
  auto guard_data = rocr::MakeScopeGuard([open_resource]() { free(open_resource); });

  if (ret != ErrorCode::Success) {
    pr_err("open resource failed %d\n", static_cast<int>(ret));
    return ErrorCode::InvalidateParams;
  }

  SharedHandleInfo shared_info{};
  SharedHandleInfo* shared_info_ptr = &shared_info;
  if (open_resource->PrivateRuntimeDataSize > 0)
    shared_info_ptr = static_cast<SharedHandleInfo*>(open_resource->pPrivateRuntimeData);

  if (open_resource->NumAllocations > 1)
    alloc_handles_ptr_ = new WinAllocationHandle[open_resource->NumAllocations];

  // Update shared_info if OpenResourceFromNtHandle skips populating it.
  if (open_resource->PrivateRuntimeDataSize == 0) {
#if defined(WIN32)
    for (auto alloc_index = 0U; alloc_index < open_resource->NumAllocations; alloc_index++) {
      const auto* const pPrivateDriverData =
          open_resource->pOpenAllocationInfo2[alloc_index].pPrivateDriverData;
      auto alloc_size = thunk_proxy::GetMemoryAllocationSize(pPrivateDriverData);
      shared_info_ptr->size += alloc_size;
      shared_info_ptr->client_size += alloc_size;
    }
#else
    assert(!"Unimplemented!");
#endif
  }

  if (shared_info_ptr->pid == dxg_runtime->parent_pid && create_info.flags.alloc_va &&
      IsSameAdapter(shared_info_ptr->adapter_luid) && shared_info_ptr->gpu_addr) {
    pr_info(
        "import from same device and same process, va is required. "
        "a buffer can't be mapped to 2 va. delete the imported buffer, use the existing one.\n");
    if (gpu_addr) *gpu_addr = shared_info_ptr->gpu_addr;
    return ErrorCode::SameProcessSameDevice;
  }

  desc_.size = shared_info_ptr->size;
  desc_.client_size = shared_info_ptr->client_size;
  desc_.domain = shared_info_ptr->domain;
  desc_.flags.reserved = shared_info_ptr->flags;
  desc_.mem_flags = shared_info_ptr->mem_flags;
  desc_.adapter_luid = shared_info_ptr->adapter_luid;
  resource_ = open_resource->hResource;
  num_allocations_ = open_resource->NumAllocations;
  for (auto i = 0; i < num_allocations_; i++)
    alloc_handles_ptr_[i] = open_resource->pOpenAllocationInfo2[i].hAllocation;

  desc_.flags.is_va_required = create_info.flags.alloc_va;
  if (desc_.flags.is_va_required) {
    desc_.flags.is_imported_vram_ipc = 1;
    ret = ReserveGpuVirtualAddress(create_info.va_hint, desc_.size, create_info.alignment);
    if (ret != ErrorCode::Success)
      pr_err("failed to allocate svm range, error:%d\n", static_cast<int>(ret));

    return ret;
  } else {
    desc_.flags.is_imported_vram_vmem = 1;
    return dxg_runtime->HandleApertureAlloc(desc_.size, &desc_.handle_ape_addr);
  }
}

ErrorCode GpuMemory::ImportPhysicalHandle(const GpuMemoryCreateInfo& create_info,
                                          gpusize* gpu_addr) {
  const auto dmabuf_fd = create_info.dmabuf_fd;

  if (dmabuf_fd == 0 || dmabuf_fd == INVALID_DMABUF_FD) return ErrorCode::InvalidateParams;

  if (create_info.flags.sysmem_ipc_sig_importer) return ImportPhysicalFD(create_info, gpu_addr);
  if (create_info.flags.kmt_handle_importer) return ImportPhysicalKMTHandle(create_info, gpu_addr);
  return ImportPhysicalNTHandle(create_info, gpu_addr);
}

} // namespace thunk
} // namespace wsl
