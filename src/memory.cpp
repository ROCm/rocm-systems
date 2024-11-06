/*
 * Copyright © 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "libhsakmt.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <fcntl.h>
#include "inc/wddm/gpu_memory.h"
#include "util/simple_heap.h"

struct Allocation {
  Allocation()
      : handle(0), cpu_addr(0), gpu_addr(0), size(0), userptr(false),
        user_data(nullptr), size_requested(0), node_id(0), mem_flags_value(0) {}
  Allocation(wsl::thunk::GpuMemoryHandle handle_arg, void *cpu_addr_arg,
             uint64_t gpu_addr_arg, size_t size_arg, bool userptr_arg = false,
             void *user_data_arg = nullptr, size_t user_size_arg = 0,
             HSAuint32 node_id_arg = 0, HSAuint32 mem_flags_value_arg = 0)
      : handle(handle_arg), cpu_addr(cpu_addr_arg), gpu_addr(gpu_addr_arg),
        size(size_arg), userptr(userptr_arg), user_data(user_data_arg),
        size_requested(user_size_arg), node_id(node_id_arg),
        mem_flags_value(mem_flags_value_arg) {}

  wsl::thunk::GpuMemoryHandle handle;
  void *cpu_addr;
  uint64_t gpu_addr;
  bool userptr;
  size_t size; /* actual size = align_up(size_requested, granularity) */
  void *user_data;
  size_t size_requested; /* size requested by user */
  HSAuint32 node_id;
  HSAuint32 mem_flags_value;
};

static std::map<const void *, Allocation> allocation_map_;
static std::unique_ptr<std::mutex> allocation_map_lock_ = std::make_unique<std::mutex>();

void clear_allocation_map(void)
{
  allocation_map_lock_ = std::make_unique<std::mutex>();
  std::lock_guard<std::mutex> lock(*allocation_map_lock_);
  allocation_map_.clear();
}

HSAKMT_STATUS HSAKMTAPI hsaKmtSetMemoryPolicy(HSAuint32 Node,
                                              HSAuint32 DefaultPolicy,
                                              HSAuint32 AlternatePolicy,
                                              void *MemoryAddressAlternate,
                                              HSAuint64 MemorySizeInBytes) {
  CHECK_DXG_OPEN();
  pr_warn_once("not implemented\n");
  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAuint32 PageSizeFromFlags(unsigned int pageSizeFlags) {
  switch (pageSizeFlags) {
  case HSA_PAGE_SIZE_4KB:
    return 4 * 1024;
  case HSA_PAGE_SIZE_64KB:
    return 64 * 1024;
  case HSA_PAGE_SIZE_2MB:
    return 2 * 1024 * 1024;
  case HSA_PAGE_SIZE_1GB:
    return 1024 * 1024 * 1024;
  default:
    assert(false);
    return 4 * 1024;
  }
}

HSAKMT_STATUS HSAKMTAPI hsaKmtAllocMemory(HSAuint32 PreferredNode,
                                          HSAuint64 SizeInBytes,
                                          HsaMemFlags MemFlags,
                                          void **MemoryAddress) {
  return hsaKmtAllocMemoryAlign(PreferredNode, SizeInBytes, 0, MemFlags,
                                MemoryAddress);
}

#define POWER_OF_2(x) ((x && (!(x & (x - 1)))) ? 1 : 0)

bool isSystemMemoryAvailable(HSAuint64 SizeInBytes) {
  struct sysinfo info;
  if (sysinfo(&info) != 0)
    return false;
  return SizeInBytes <= info.freeram;
}

void* BlockAllocator::alloc(size_t request_size, size_t& allocated_size) const {
  void *address;
  HsaMemFlags MemFlags;

  MemFlags.Value = 0;
  MemFlags.ui32.CoarseGrain = 1;
  MemFlags.ui32.NoSubstitute = 1;
  allocated_size = wsl::AlignUp(request_size, block_size());
  if (HSAKMT_STATUS_SUCCESS == hsaKmtAllocMemoryAlignInternal(1, allocated_size, 0, MemFlags, &address, true))
    return address;

  return nullptr;
}

void BlockAllocator::free(void* ptr, size_t length) const {
  if (HSAKMT_STATUS_SUCCESS != hsaKmtFreeMemoryInternal(ptr, length, true))
    pr_err("wsl-thunk: BlockAllocator::free() err, address %p, length:%zu\n", ptr, length);
}

static wsl::SimpleHeap<BlockAllocator> fragment_allocator_;
static std::unique_ptr<std::mutex> fragment_allocator_lock_ = std::make_unique<std::mutex>();

void reset_suballocator(void) {
  fragment_allocator_lock_ = std::make_unique<std::mutex>();
  std::lock_guard<std::mutex> lock(*fragment_allocator_lock_);
  fragment_allocator_.reset();
}

void trim_suballocator(void) {
  std::lock_guard<std::mutex> lock(*fragment_allocator_lock_);
  fragment_allocator_.trim();
}

HSAKMT_STATUS hsaKmtAllocMemoryAlignInternal(HSAuint32 PreferredNode,
                                             HSAuint64 SizeInBytes,
                                             HSAuint64 Alignment,
                                             HsaMemFlags MemFlags,
                                             void **MemoryAddress,
                                             bool SkipSubAlloc) {
  CHECK_DXG_OPEN();

  if (!MemoryAddress)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  if (MemFlags.ui32.FixedAddress) {
    if (*MemoryAddress == nullptr)
      return HSAKMT_STATUS_INVALID_PARAMETER;
  } else
    *MemoryAddress = nullptr;

  wsl::thunk::WDDMDevice *dev = get_wddmdev(1);
  if (!dev)
    return HSAKMT_STATUS_ERROR;

  wsl::thunk::GpuMemory *gpu_mem = nullptr;
  wsl::thunk::GpuMemoryCreateInfo create_info{};
  create_info.size = SizeInBytes;

  /* If initialize scratch pool of GpuAgent, treat it as SVM reserve */
  if (MemFlags.ui32.Scratch && MemFlags.ui32.HostAccess && SizeInBytes > 0x80000000)
    MemFlags.ui32.OnlyAddress = 1;

  create_info.alignment = Alignment;
  create_info.va_hint = reinterpret_cast<gpusize>(*MemoryAddress);
  if ((PreferredNode == 0 && !MemFlags.ui32.NonPaged)
    || zfb_support || MemFlags.ui32.GTTAccess) {
    if (SizeInBytes > max_single_alloc_size)
      return HSAKMT_STATUS_NO_MEMORY;

    if (check_avail_sysram && !isSystemMemoryAvailable(SizeInBytes))
      return HSAKMT_STATUS_NO_MEMORY;

    /* If allocate VRAM under ZFB mode */
    if (zfb_support && MemFlags.ui32.NonPaged == 1)
      MemFlags.ui32.CoarseGrain = 1;

    create_info.domain = thunk_proxy::AllocDomain::kSystem;
  } else {
    create_info.domain = thunk_proxy::AllocDomain::kLocal;
  }

  if (!MemFlags.ui32.CoarseGrain)
    create_info.mem_flags = thunk_proxy::kFineGrain;

  //In hsa-runtime, only kernarg region set Uncached.
  if (MemFlags.ui32.Uncached)
    create_info.mem_flags |= thunk_proxy::kKernarg;

  create_info.flags.physical_only = MemFlags.ui32.NoAddress;
  create_info.flags.interprocess = MemFlags.ui32.NoAddress;
  create_info.flags.locked = MemFlags.ui32.NoSubstitute;//AllocatePinned
  create_info.flags.virtual_alloc = MemFlags.ui32.OnlyAddress;
  /*when only alloc virtual or only physical, it's vmm allocation, force to local*/
  if (create_info.flags.virtual_alloc || create_info.flags.physical_only)
    create_info.domain = thunk_proxy::AllocDomain::kLocal;

  /* Only allow using the suballocator for ordinary VRAM.*/
  bool trim_safe = false;
  if (!SkipSubAlloc &&
    create_info.domain == thunk_proxy::AllocDomain::kLocal &&
    !(create_info.flags.virtual_alloc || create_info.flags.physical_only)) {
    std::lock_guard<std::mutex> gard(*fragment_allocator_lock_);

    /* just quickly skip SA if size is bigger than SA block size.*/
    gpusize real_size;
    if (create_info.size > GPU_HUGE_PAGE_SIZE)
      real_size = wsl::AlignUp(create_info.size, GPU_HUGE_PAGE_SIZE);
    else
      real_size = wsl::AlignUp(create_info.size, getpagesize());

    if (real_size < fragment_allocator_.default_block_size()) {
      *MemoryAddress = fragment_allocator_.alloc(real_size);
      if (*MemoryAddress)
        return HSAKMT_STATUS_SUCCESS;
    }

    /* SA might keep a lot of free blocks as *cache*.
       * We can trim them if direct allocation fails at first time.
       */
    trim_safe = true;
  }

after_trim:
  auto code = dev->CreateGpuMemory(create_info, &gpu_mem);
  if (code == ErrorCode::Success) {
    std::lock_guard<std::mutex> gard(*allocation_map_lock_);

    /* For these physical allcations, use GpuMemory object's address as thunk handle*/
    if (create_info.flags.physical_only || create_info.dmabuf_fd > 0)
      *MemoryAddress = reinterpret_cast<void*>(gpu_mem->HandleApeAddress());
    else
      *MemoryAddress = reinterpret_cast<void *>(gpu_mem->GpuAddress());

    allocation_map_[*MemoryAddress] = Allocation(
        gpu_mem->GetGpuMemoryHandle(), *MemoryAddress, (uint64_t)*MemoryAddress,
        create_info.size, false, nullptr, SizeInBytes,
        MemFlags.ui32.GTTAccess ? 0 : PreferredNode, MemFlags.Value);
    return HSAKMT_STATUS_SUCCESS;
  } else if (trim_safe) {
    /* attempt to release memory from the block allocator and retry */
    std::lock_guard<std::mutex> gard(*fragment_allocator_lock_);
    fragment_allocator_.trim();
    trim_safe = false;
    goto after_trim;
  }

  return HSAKMT_STATUS_ERROR;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtAllocMemoryAlign(HSAuint32 PreferredNode,
                                               HSAuint64 SizeInBytes,
                                               HSAuint64 Alignment,
                                               HsaMemFlags MemFlags,
                                               void **MemoryAddress) {
  return hsaKmtAllocMemoryAlignInternal(PreferredNode, SizeInBytes,
                                        Alignment, MemFlags,
                                        MemoryAddress);
}

HSAKMT_STATUS hsaKmtFreeMemoryInternal(void *MemoryAddress,
                                       HSAuint64 SizeInBytes,
                                       bool SkipSubAlloc) {
  CHECK_DXG_OPEN();

  if (!MemoryAddress)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  if (!SkipSubAlloc) {
    std::lock_guard<std::mutex> gard(*fragment_allocator_lock_);
    if (fragment_allocator_.free(MemoryAddress))
      return HSAKMT_STATUS_SUCCESS;
  }

  wsl::thunk::GpuMemory *gpu_mem = nullptr;
  {
    std::lock_guard<std::mutex> gard(*allocation_map_lock_);
    auto it = allocation_map_.find(MemoryAddress);
    if (it == allocation_map_.end()) {
      return HSAKMT_STATUS_ERROR;
    }

    gpu_mem = wsl::thunk::GpuMemory::Convert(it->second.handle);
    allocation_map_.erase(it);
  }

  if (gpu_mem->IsQueueReferenced())
    return HSAKMT_STATUS_ERROR;

  delete gpu_mem;
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtFreeMemory(void *MemoryAddress,
                     HSAuint64 SizeInBytes) {
  return hsaKmtFreeMemoryInternal(MemoryAddress, SizeInBytes);
}

bool queue_acquire_buffer(void *MemoryAddress) {
  if (!MemoryAddress)
  return false;

  wsl::thunk::GpuMemory *gpu_mem = nullptr;
  {
  std::lock_guard<std::mutex> gard(*allocation_map_lock_);
  auto it = allocation_map_.find(MemoryAddress);
  if (it == allocation_map_.end()) {
    return HSAKMT_STATUS_ERROR;
  }

  gpu_mem = wsl::thunk::GpuMemory::Convert(it->second.handle);
  gpu_mem->GetQueueReference();
  }
  if (gpu_mem == nullptr)
  return false;

  return true;
}

bool queue_release_buffer(void *MemoryAddress) {
  if (!MemoryAddress)
    return false;

  wsl::thunk::GpuMemory *gpu_mem = nullptr;
  {
    std::lock_guard<std::mutex> gard(*allocation_map_lock_);
    auto it = allocation_map_.find(MemoryAddress);
    if (it == allocation_map_.end()) {
      return HSAKMT_STATUS_ERROR;
    }

    gpu_mem = wsl::thunk::GpuMemory::Convert(it->second.handle);
    gpu_mem->PutQueueReference();
  }
  if (gpu_mem == nullptr)
    return false;

  return true;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtAvailableMemory(HSAuint32 Node,
                                              HSAuint64 *AvailableBytes) {
  CHECK_DXG_OPEN();

  if (!AvailableBytes)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  wsl::thunk::WDDMDevice *dev = get_wddmdev(Node);
  if (!dev)
    return HSAKMT_STATUS_ERROR;

  *AvailableBytes = dev->VramAvail();
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterMemory(void *MemoryAddress,
                                             HSAuint64 MemorySizeInBytes) {
  CHECK_DXG_OPEN();
  pr_warn_once("not implemented\n");
  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterMemoryToNodes(void *MemoryAddress,
                                                    HSAuint64 MemorySizeInBytes,
                                                    HSAuint64 NumberOfNodes,
                                                    HSAuint32 *NodeArray) {
  CHECK_DXG_OPEN();

  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterMemoryWithFlags(
    void *MemoryAddress, HSAuint64 MemorySizeInBytes, HsaMemFlags MemFlags) {
  CHECK_DXG_OPEN();

  if (!MemoryAddress)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  pr_debug("address %p\n", MemoryAddress);

  if (MemFlags.ui32.ExtendedCoherent && MemFlags.ui32.CoarseGrain)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  // Registered memory should be ordinary paged host memory.
  if ((MemFlags.ui32.HostAccess != 1) || (MemFlags.ui32.NonPaged == 1))
    return HSAKMT_STATUS_NOT_SUPPORTED;

  if (!is_dgpu)
    /* TODO: support mixed APU and dGPU configurations */
    return HSAKMT_STATUS_NOT_SUPPORTED;

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterGraphicsHandleToNodes(HSAuint64 GraphicsResourceHandle,
                                                            HsaGraphicsResourceInfo *GraphicsResourceInfo,
                                                            HSAuint64 NumberOfNodes,
                                                            HSAuint32 *NodeArray) {
  HSA_REGISTER_MEM_FLAGS regFlags;
  regFlags.Value = 0;

  return hsaKmtRegisterGraphicsHandleToNodesExt(GraphicsResourceHandle,
            GraphicsResourceInfo,
            NumberOfNodes,
            NodeArray,
            regFlags);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterGraphicsHandleToNodesExt(HSAuint64 GraphicsResourceHandle,
							       HsaGraphicsResourceInfo *GraphicsResourceInfo,
							       HSAuint64 NumberOfNodes,
							       HSAuint32 *NodeArray,
							       HSA_REGISTER_MEM_FLAGS RegisterFlags) {
  CHECK_DXG_OPEN();
  uint32_t *gpu_id_array = NULL;
  HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

  pr_debug("number of nodes %lu\n", NumberOfNodes);

  GraphicsResourceInfo->NodeId = 1;
  return hsaKmtImportDMABufHandle(GraphicsResourceHandle, &GraphicsResourceInfo->MemoryAddress);
}


HSAKMT_STATUS HSAKMTAPI hsaKmtExportDMABufHandle(void *MemoryAddress,
                                                 HSAuint64 MemorySizeInBytes,
                                                 int *DMABufFd,
                                                 HSAuint64 *Offset) {
  CHECK_DXG_OPEN();

  std::lock_guard<std::mutex> gard(*allocation_map_lock_);
  auto it = allocation_map_.find(MemoryAddress);
  if (it == allocation_map_.end())
    return HSAKMT_STATUS_ERROR;

  auto gpu_mem = wsl::thunk::GpuMemory::Convert(it->second.handle);
  auto code = gpu_mem->ExportPhysicalHandle(DMABufFd);
  if (code != ErrorCode::Success)
    return HSAKMT_STATUS_ERROR;

  return HSAKMT_STATUS_SUCCESS;
}


HSAKMT_STATUS hsaKmtImportDMABufHandle(int DMABufFd,
                                                 void **MemoryAddress) {

  CHECK_DXG_OPEN();

  wsl::thunk::WDDMDevice* dev = get_wddmdev(1);
  wsl::thunk::GpuMemory *gpu_mem = nullptr;
  wsl::thunk::GpuMemoryCreateInfo create_info{};
  create_info.dmabuf_fd = DMABufFd;

  auto code = dev->CreateGpuMemory(create_info, &gpu_mem);
  if (code == ErrorCode::Success) {
    *MemoryAddress = reinterpret_cast<void *>(gpu_mem);
    std::lock_guard<std::mutex> gard(*allocation_map_lock_);
    /*
     * the gpu_mem->Flags() need convert back from GpuMemoryCreateFlags to
     * HsaMemFlags, reference hsaKmtAllocMemoryAlign
     * */
    allocation_map_[*MemoryAddress] = Allocation(
      gpu_mem->GetGpuMemoryHandle(), *MemoryAddress, (uint64_t)*MemoryAddress,
      gpu_mem->Size(), false, nullptr, gpu_mem->ClientSize(),
      1, gpu_mem->Flags());

    return HSAKMT_STATUS_SUCCESS;
  }

  return HSAKMT_STATUS_ERROR;

}


HSAKMT_STATUS HSAKMTAPI
hsaKmtShareMemory(void *MemoryAddress, HSAuint64 SizeInBytes,
                  HsaSharedMemoryHandle *SharedMemoryHandle) {
  CHECK_DXG_OPEN();
  pr_warn_once("not implemented\n");
  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtRegisterSharedHandle(const HsaSharedMemoryHandle *SharedMemoryHandle,
                           void **MemoryAddress, HSAuint64 *SizeInBytes) {
  CHECK_DXG_OPEN();
  pr_warn_once("not implemented\n");
  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterSharedHandleToNodes(
    const HsaSharedMemoryHandle *SharedMemoryHandle, void **MemoryAddress,
    HSAuint64 *SizeInBytes, HSAuint64 NumberOfNodes, HSAuint32 *NodeArray) {
  CHECK_DXG_OPEN();
  pr_warn_once("not implemented\n");
  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtProcessVMRead(HSAuint32 Pid,
                                            HsaMemoryRange *LocalMemoryArray,
                                            HSAuint64 LocalMemoryArrayCount,
                                            HsaMemoryRange *RemoteMemoryArray,
                                            HSAuint64 RemoteMemoryArrayCount,
                                            HSAuint64 *SizeCopied) {
  CHECK_DXG_OPEN();
  pr_warn_once("has been deprecated\n");
  assert(false);
  return HSAKMT_STATUS_NOT_IMPLEMENTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtProcessVMWrite(HSAuint32 Pid,
                                             HsaMemoryRange *LocalMemoryArray,
                                             HSAuint64 LocalMemoryArrayCount,
                                             HsaMemoryRange *RemoteMemoryArray,
                                             HSAuint64 RemoteMemoryArrayCount,
                                             HSAuint64 *SizeCopied) {
  CHECK_DXG_OPEN();
  pr_warn_once("has been deprecated\n");
  assert(false);
  return HSAKMT_STATUS_NOT_IMPLEMENTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDeregisterMemory(void *MemoryAddress) {
  CHECK_DXG_OPEN();

  if (!MemoryAddress)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  pr_debug("address %p\n", MemoryAddress);

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMapMemoryToGPU(void *MemoryAddress,
                                             HSAuint64 MemorySizeInBytes,
                                             HSAuint64 *AlternateVAGPU) {
  CHECK_DXG_OPEN();

  if (!MemoryAddress || !AlternateVAGPU) {
    pr_err("FIXME: mapping NULL pointer\n");
    return HSAKMT_STATUS_ERROR;
  }

  uint64_t start = wsl::AlignDown((uint64_t)MemoryAddress, 4096);
  uint64_t end =
      wsl::AlignUp((uint64_t)MemoryAddress + MemorySizeInBytes, 4096);

  void *aligned_ptr = (void *)start;
  size_t aligned_size = end - start;

  {
    std::lock_guard<std::mutex> gard(*fragment_allocator_lock_);
    if (nullptr != fragment_allocator_.block_base(aligned_ptr))
      return HSAKMT_STATUS_SUCCESS;
  }

  {
    std::lock_guard<std::mutex> gard(*allocation_map_lock_);
    // GTT mem
    auto it_gtt = allocation_map_.find(aligned_ptr);
    if (it_gtt != allocation_map_.end()) {
      if (!it_gtt->second.userptr) {
        if (it_gtt->second.size >= MemorySizeInBytes) {
          *AlternateVAGPU = (uint64_t)MemoryAddress;
          return HSAKMT_STATUS_SUCCESS;
        } else {
          return HSAKMT_STATUS_ERROR;
        }
      }
    }

    // userptr mem
    auto it = allocation_map_.find(MemoryAddress);
    if (it != allocation_map_.end()) {
      if (it->second.userptr && it->second.size >= MemorySizeInBytes) {
        *AlternateVAGPU =
            (uintptr_t)it->second.gpu_addr +
            ((uintptr_t)MemoryAddress - (uintptr_t)it->second.cpu_addr);
        return HSAKMT_STATUS_SUCCESS;
      }
    }
  }

  wsl::thunk::WDDMDevice *dev = get_wddmdev(1);
  if (!dev)
    return HSAKMT_STATUS_ERROR;

  wsl::thunk::GpuMemory *gpu_mem = nullptr;
  wsl::thunk::GpuMemoryHandle handle = 0;
  uint64_t addr;
  wsl::thunk::GpuMemoryCreateInfo create_info{};
  create_info.domain = thunk_proxy::kUserMemory;
  create_info.size = aligned_size;
  create_info.user_ptr = aligned_ptr;

  auto code = dev->CreateGpuMemory(create_info, &gpu_mem);
  if (code == ErrorCode::Success) {
    addr = gpu_mem->GpuAddress();
    handle = gpu_mem->GetGpuMemoryHandle();
  } else {
    return HSAKMT_STATUS_ERROR;
  }

  {
    std::lock_guard<std::mutex> guard(*allocation_map_lock_);
    allocation_map_[MemoryAddress] =
        Allocation(handle, aligned_ptr, addr, aligned_size, true, MemoryAddress,
                   MemorySizeInBytes);
    allocation_map_[(void *)addr] =
        Allocation(handle, aligned_ptr, addr, aligned_size, true, nullptr,
                   MemorySizeInBytes);
  }

  *AlternateVAGPU = addr + ((uintptr_t)MemoryAddress - (uintptr_t)aligned_ptr);

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMapMemoryToGPUNodes(
    void *MemoryAddress, HSAuint64 MemorySizeInBytes, HSAuint64 *AlternateVAGPU,
    HsaMemMapFlags MemMapFlags, HSAuint64 NumberOfNodes, HSAuint32 *NodeArray) {
  return hsaKmtMapMemoryToGPU(MemoryAddress, MemorySizeInBytes, AlternateVAGPU);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtUnmapMemoryToGPU(void *MemoryAddress) {
  CHECK_DXG_OPEN();

  if (!MemoryAddress) {
    /* Workaround for runtime bug */
    pr_err("FIXME: Unmapping NULL pointer\n");
    return HSAKMT_STATUS_SUCCESS;
  }

  pr_debug("address %p\n", MemoryAddress);

  {
    std::lock_guard<std::mutex> gard(*fragment_allocator_lock_);
    if (nullptr != fragment_allocator_.block_base(MemoryAddress))
      return HSAKMT_STATUS_SUCCESS;
  }

  wsl::thunk::GpuMemoryHandle handle = nullptr;
  {
    std::lock_guard<std::mutex> gard(*allocation_map_lock_);
    auto it = allocation_map_.find(MemoryAddress);
    if (it == allocation_map_.end()) {
      return HSAKMT_STATUS_ERROR;
    }

    if (!it->second.userptr) {
      return HSAKMT_STATUS_SUCCESS;
    }

    handle = it->second.handle;

    allocation_map_.erase((void *)it->second.gpu_addr);
    allocation_map_.erase(it);
  }
  auto gpu_mem = wsl::thunk::GpuMemory::Convert(handle);
  if (gpu_mem->IsQueueReferenced())
    return HSAKMT_STATUS_ERROR;

  delete gpu_mem;
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMapGraphicHandle(HSAuint32 NodeId,
                                               HSAuint64 GraphicDeviceHandle,
                                               HSAuint64 GraphicResourceHandle,
                                               HSAuint64 GraphicResourceOffset,
                                               HSAuint64 GraphicResourceSize,
                                               HSAuint64 *FlatMemoryAddress) {
  CHECK_DXG_OPEN();
  pr_warn_once("not implemented\n");
  /* This API was only ever implemented in KFD for Kaveri and
   * was never upstreamed. There are no open-source users of
   * this interface. It has been superseded by
   * RegisterGraphicsHandleToNodes.
   */
  return HSAKMT_STATUS_NOT_IMPLEMENTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtUnmapGraphicHandle(HSAuint32 NodeId,
                                                 HSAuint64 FlatMemoryAddress,
                                                 HSAuint64 SizeInBytes) {
  CHECK_DXG_OPEN();
  pr_warn_once("not implemented\n");
  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetTileConfig(HSAuint32 NodeId,
                                            HsaGpuTileConfig *config) {
  CHECK_DXG_OPEN();
  pr_warn_once("not implemented\n");
  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtQueryPointerInfo(const void *Pointer,
                                               HsaPointerInfo *PointerInfo) {
  CHECK_DXG_OPEN();

  if (!Pointer || !PointerInfo)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  pr_debug("pointer %p\n", Pointer);
  void *ptr = const_cast<void*>(Pointer);

  memset(PointerInfo, 0, sizeof(HsaPointerInfo));
  void* block_base = nullptr;
  {
    std::lock_guard<std::mutex> gard(*fragment_allocator_lock_);
      block_base = fragment_allocator_.block_base(ptr);
    if (block_base != nullptr)
      ptr = block_base;
  }

  Allocation allocation_info;
  bool found = false;
  {
    std::lock_guard<std::mutex> gard(*allocation_map_lock_);
    auto it = allocation_map_.upper_bound(ptr);
    if (it != allocation_map_.begin()) {
      --it;
      if (ptr >= it->first &&
        (ptr < reinterpret_cast<const uint8_t*>(it->first) + it->second.size_requested)) {
        allocation_info = it->second;
        found = true;
      }
    }
  }

  if (!found) {
    pr_debug("can't found allocation for %p\n", Pointer);
    PointerInfo->Type = HSA_POINTER_UNKNOWN;
    return HSAKMT_STATUS_ERROR;
  }

  if (allocation_info.userptr) {
    PointerInfo->Type = HSA_POINTER_REGISTERED_USER;
    PointerInfo->SizeInBytes = allocation_info.size;
  } else {
    PointerInfo->Type = HSA_POINTER_ALLOCATED;
    PointerInfo->SizeInBytes = allocation_info.size_requested;
  }

  PointerInfo->Node = allocation_info.node_id;
  PointerInfo->MemFlags.Value = allocation_info.mem_flags_value;
  PointerInfo->CPUAddress = allocation_info.cpu_addr;
  PointerInfo->GPUAddress = allocation_info.gpu_addr;
  if (block_base != nullptr) {
    uint64_t offset = reinterpret_cast<uint64_t>(Pointer) -
      reinterpret_cast<uint64_t>(block_base);
    PointerInfo->CPUAddress = reinterpret_cast<void*>(
      reinterpret_cast<uint64_t>(PointerInfo->CPUAddress) + offset);
    PointerInfo->GPUAddress += offset;
  }
  PointerInfo->UserData = allocation_info.user_data;

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtSetMemoryUserData(const void *Pointer,
                                                void *UserData) {
  CHECK_DXG_OPEN();
  pr_warn_once("not implemented\n");
  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtReplaceAsanHeaderPage(void *addr) {
  CHECK_DXG_OPEN();
  pr_warn_once("not supported\n");
  assert(false);
#ifdef SANITIZER_AMDGPU
  pr_debug("address %p\n", addr);
  CHECK_DXG_OPEN();

  return HSAKMT_STATUS_SUCCESS;
#else
  return HSAKMT_STATUS_NOT_SUPPORTED;
#endif
}

HSAKMT_STATUS HSAKMTAPI hsaKmtReturnAsanHeaderPage(void *addr) {
  CHECK_DXG_OPEN();
  pr_warn_once("not supported\n");
  assert(false);
#ifdef SANITIZER_AMDGPU
  pr_debug("address %p\n", addr);
  CHECK_DXG_OPEN();

  return HSAKMT_STATUS_SUCCESS;
#else
  return HSAKMT_STATUS_NOT_SUPPORTED;
#endif
}
