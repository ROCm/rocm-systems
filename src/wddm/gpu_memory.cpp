#include <cinttypes>
#include <cassert>
#include "inc/wddm/gpu_memory.h"
#include "inc/wddm/device.h"
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
      size = AlignUp(size, alignment);
  } else {
    const size_t min_size = 4096;
    size = AlignUp(size, min_size);
  }
  return size;
}

GpuMemory::GpuMemory(WDDMDevice *device) : device_(device) {
  num_allocations_ = 0;
  alloc_handles_ptr_ = nullptr;
  alloc_handle_ = 0;
  resource_ = 0;
}

GpuMemory::~GpuMemory() {
  FreeGpuVirtualAddress(GpuAddress(), Size());
  FreePhysicalMemory();
  if (desc_.handle_ape_addr > 0)
    device_->HandleApertureFree(desc_.handle_ape_addr);
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
  desc_.flags.is_shared = create_info.flags.interprocess;
  desc_.flags.is_locked = create_info.flags.locked;

  desc_.size = AdjustSize(desc_.client_size);

  if (IsUserMemory() || IsSystem())
    desc_.cpu_addr = create_info.user_ptr;

  num_allocations_ = CalcChunkNumbers(Size());
  if (num_allocations_ == 1)
    alloc_handles_ptr_ = &alloc_handle_;
  else
    alloc_handles_ptr_ = new WinAllocationHandle[num_allocations_];

  memset(alloc_handles_ptr_, 0, num_allocations_ * sizeof(WinAllocationHandle));

  auto code = ErrorCode::Success;

  if (IsPhysicalOnly()) {
    code = CreatePhysicalMemory();
    if (code == ErrorCode::Success)
      code = device_->HandleApertureAlloc(desc_.size, &desc_.handle_ape_addr);
    return code;
  }

  code = ReserveGpuVirtualAddress(create_info.va_hint, Size(), create_info.alignment);
  if (IsVirtual() || (code != ErrorCode::Success))
      return code;

  bool physical_created = false;

  auto guard = MakeScopeGuard([this, &physical_created, &code]() {
    if (code != ErrorCode::Success) {

      if (physical_created) {
        FreePhysicalMemory();
      }
      FreeGpuVirtualAddress(GpuAddress(), Size());
    }
  });
  (void)guard;

  code = CreatePhysicalMemory();
  if (code != ErrorCode::Success)
    return code;

  physical_created = true;

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
  gpusize gpu_virt_addr = 0;
  auto status = device_->ReserveGpuVirtualAddress(desc_.domain, base_virt_addr, size, &gpu_virt_addr, alignment,
    desc_.flags.is_locked);
  if (status == ErrorCode::Success) {
    desc_.gpu_addr = gpu_virt_addr;

    if (IsSystem())
      desc_.cpu_addr = reinterpret_cast<void *>(desc_.gpu_addr);
  }
  return status;
}

ErrorCode GpuMemory::FreeGpuVirtualAddress(gpusize base_addr, gpusize size) {
  return base_addr != 0 ?
         device_->FreeGpuVirtualAddress(desc_.domain, base_addr, size) :
         ErrorCode::Success;
}

ErrorCode GpuMemory::CreatePhysicalMemory() {

  assert(!IsVirtual() && NumChunks() > 0);

  const auto num_allocations = NumChunks();
  void *priv_drv_data;
  void *alloc_priv;
  int priv_drv_data_size;
  int alloc_priv_data_size;

  if (!thunk_proxy::CreatePrivateAllocInfo(NumChunks(), &priv_drv_data, &alloc_priv,
                                          &priv_drv_data_size, &alloc_priv_data_size))
    return ErrorCode::OutOfMemory;

  auto alloc_info = reinterpret_cast<D3DDDI_ALLOCATIONINFO2*>(
       static_cast<unsigned char*>(priv_drv_data) + priv_drv_data_size * num_allocations);

  size_t size = desc_.size;
  uint64_t addr = desc_.gpu_addr;
  char *cpu_addr = static_cast<char *>(desc_.cpu_addr);
  const auto &device_info = GetDevice()->DeviceInfo();

  for (size_t i = 0; i < num_allocations; i++) {

    void* priv_data = (void*)((char*)priv_drv_data + priv_drv_data_size * i);
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
    alloc_info[i].PrivateDriverDataSize = priv_drv_data_size;
    alloc_info[i].VidPnSourceId = D3DDDI_ID_UNINITIALIZED;
  }

  D3DKMT_CREATEALLOCATION args = {};
  args.hDevice = device_->DeviceHandle();
  args.pPrivateDriverData = alloc_priv;
  args.PrivateDriverDataSize = alloc_priv_data_size;
  args.NumAllocations = num_allocations;
  args.pAllocationInfo2 = alloc_info;

  SharedHandleInfo shared_info;
  if (IsShared()) {
    shared_info.size = desc_.size;
    shared_info.client_size = desc_.client_size;
    shared_info.domain = desc_.domain;
    shared_info.adapter_luid = desc_.adapter_luid;
    shared_info.flags = reinterpret_cast<uint32_t>(desc_.flags.reserved);
    shared_info.mem_flags = desc_.mem_flags;
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
  }
  thunk_proxy::DestroyPrivateAllocInfo(priv_drv_data, alloc_priv);
  return status;
}

ErrorCode GpuMemory::FreePhysicalMemory() {
  auto code = ErrorCode::Success;

  if (alloc_handles_ptr_ == nullptr || (NumChunks() == 1 && *alloc_handles_ptr_ == 0))
      return code;

  code = d3dthunk::DestroyAllocation(device_->DeviceHandle(),
                                  resource_,
                                  NumChunks(),
                                  alloc_handles_ptr_);
  if (NumChunks() > 1)
    delete[] alloc_handles_ptr_;

  alloc_handles_ptr_ = nullptr;
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

ErrorCode GpuMemory::ExportPhysicalHandle(int* dmabuf_fd, uint32_t flags) {
  if (IsShared())
    return d3dthunk::ShareObjects(num_allocations_, resource_, flags, dmabuf_fd);
  else
    return ErrorCode::UnSupported;
}


ErrorCode GpuMemory::ImportPhysicalHandle(int dmabuf_fd) {
  D3DKMT_QUERYRESOURCEINFOFROMNTHANDLE query_args;

  if (dmabuf_fd <= 0)
    return ErrorCode::InvalidateParams;

  memset(&query_args, 0, sizeof(query_args));
  query_args.hDevice = device_->DeviceHandle();
  query_args.hNtHandle = reinterpret_cast<HANDLE>(dmabuf_fd);
  auto ret = d3dthunk::QueryResourceInfoFromNtHandle(&query_args);
  if (ret != ErrorCode::Success) {
    pr_err("query resource info from nt handle failed %d\n", static_cast<int>(ret));
    return ErrorCode::InvalidateParams;
  }
  pr_debug("wsl-thunk: import from nt handle %d, get allocation number %d,"
           " runtime data size %#x total driver data size %#x resource data size=%#x\n",
           dmabuf_fd,
           query_args.NumAllocations,
           query_args.PrivateRuntimeDataSize,
           query_args.TotalPrivateDriverDataSize,
           query_args.ResourcePrivateDriverDataSize);

  SharedHandleInfo shared_info;
  if(sizeof(shared_info) != query_args.PrivateRuntimeDataSize) {
    pr_err("shared hanle info size mismatch:%d vs %ld\n",
           query_args.PrivateRuntimeDataSize, sizeof(shared_info));
    return ErrorCode::UnSupported;
  }

  uint32_t total_size = query_args.NumAllocations * sizeof(D3DDDI_OPENALLOCATIONINFO2) +
    query_args.TotalPrivateDriverDataSize +
    query_args.ResourcePrivateDriverDataSize;
  D3DDDI_OPENALLOCATIONINFO2 *open_info =
    reinterpret_cast<D3DDDI_OPENALLOCATIONINFO2*> (calloc(1, total_size));
  if (!open_info) {
    pr_err("alloc open_info failed, NumAllocations:%d\n",
           query_args.NumAllocations);
    return ErrorCode::OutOfMemory;
  }

  alloc_handles_ptr_ = new WinAllocationHandle[query_args.NumAllocations];

  D3DKMT_OPENRESOURCEFROMNTHANDLE open_args;
  memset(&open_args, 0, sizeof(open_args));
  open_args.hDevice = query_args.hDevice;
  open_args.hNtHandle = query_args.hNtHandle;
  open_args.NumAllocations = query_args.NumAllocations;
  open_args.pOpenAllocationInfo2 = open_info;
  open_args.TotalPrivateDriverDataBufferSize = query_args.TotalPrivateDriverDataSize;
  open_args.pTotalPrivateDriverDataBuffer = reinterpret_cast<void*>
    (open_args.pOpenAllocationInfo2 + open_args.NumAllocations);
  open_args.ResourcePrivateDriverDataSize = query_args.ResourcePrivateDriverDataSize;
  open_args.pResourcePrivateDriverData = reinterpret_cast<void*>
    (((uint64_t)open_args.pTotalPrivateDriverDataBuffer) +
     open_args.TotalPrivateDriverDataBufferSize);
  open_args.PrivateRuntimeDataSize = query_args.PrivateRuntimeDataSize;
  open_args.pPrivateRuntimeData = reinterpret_cast<void*> (&shared_info);

  ret = d3dthunk::OpenResourceFromNtHandle(&open_args);
  if (ret != ErrorCode::Success) {
    ret = ErrorCode::InvalidateParams;
    pr_err("open resource failed %d\n", static_cast<int>(ret));
    goto err_out;
  }

  desc_.size = shared_info.size;
  desc_.client_size = shared_info.client_size;
  desc_.domain = shared_info.domain;
  desc_.flags.reserved = shared_info.flags;
  desc_.mem_flags = shared_info.mem_flags;
  desc_.adapter_luid = shared_info.adapter_luid;
  resource_ = open_args.hResource;
  num_allocations_ = open_args.NumAllocations;
  for (int i = 0; i < num_allocations_; i++)
    alloc_handles_ptr_[i] = open_info[i].hAllocation;

  free(open_info);
  return device_->HandleApertureAlloc(desc_.size, &desc_.handle_ape_addr);

err_out:
  delete[] alloc_handles_ptr_;
  alloc_handles_ptr_ = nullptr;
  free(open_info);
  return ret;
}

} // namespace thunk
} // namespace wsl
