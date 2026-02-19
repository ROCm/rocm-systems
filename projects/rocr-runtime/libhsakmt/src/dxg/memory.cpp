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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/types.h>
#if defined(__linux__)
#include <sys/mman.h>
#include <sys/sysinfo.h>
#endif
#include <sys/stat.h>
#include <fcntl.h>
#include "impl/wddm/gpu_memory.h"
#include "util/simple_heap.h"
#include "util/os.h"

struct Allocation {
  Allocation()
      : handle(0), cpu_addr(0), gpu_addr(0), size(0), userptr(false),
        user_data(nullptr), size_requested(0), node_id(0), mem_flags_value(0),
        dmabuf_fd(-1), rocr_userdata(nullptr) {}
  Allocation(wsl::thunk::GpuMemoryHandle handle_arg, void *cpu_addr_arg,
             uint64_t gpu_addr_arg, size_t size_arg, bool userptr_arg = false,
             void *user_data_arg = nullptr, size_t user_size_arg = 0,
             HSAuint32 node_id_arg = 0, HSAuint32 mem_flags_value_arg = 0)
      : handle(handle_arg), cpu_addr(cpu_addr_arg), gpu_addr(gpu_addr_arg),
        size(size_arg), userptr(userptr_arg), user_data(user_data_arg),
        size_requested(user_size_arg), node_id(node_id_arg),
        mem_flags_value(mem_flags_value_arg), dmabuf_fd(-1), rocr_userdata(nullptr) {}

  wsl::thunk::GpuMemoryHandle handle;
  void *cpu_addr;
  uint64_t gpu_addr;
  bool userptr;
  size_t size; /* actual size = align_up(size_requested, granularity) */
  void *user_data;
  size_t size_requested; /* size requested by user */
  HSAuint32 node_id;
  HSAuint32 mem_flags_value;
  int dmabuf_fd;
  void *rocr_userdata;
};

static std::map<const void *, Allocation>* allocation_map_ = new std::map<const void *, Allocation>();
static std::mutex* allocation_map_lock_ = new std::mutex();
static rocr::SimpleHeap<BlockAllocator>* fragment_allocator_ = new rocr::SimpleHeap<BlockAllocator>();

// ================================================================================================
static Allocation* FindAllocation(const void* ptr, size_t size) {
  auto it = allocation_map_->upper_bound(ptr);
  if (it == allocation_map_->begin()) {
    return nullptr;
  }
  --it;
  auto& alloc = it->second;
  if (ptr >= it->first &&
      (reinterpret_cast<const char*>(ptr) + size) <=
          (reinterpret_cast<const char*>(it->first) + alloc.size)) {
    return &alloc;
  } else {
    return nullptr;
  }
}

// ================================================================================================
void clear_allocation_map(void) {
  //delete allocation_map_lock_;
  allocation_map_lock_ = new std::mutex();
  std::lock_guard<std::mutex> lock(*allocation_map_lock_);
  delete allocation_map_;
  allocation_map_ = new std::map<const void *, Allocation>();
}

// ================================================================================================
wsl::thunk::GpuMemory* GetGpuMemoryFromAddress(void* memory_address) {
  std::lock_guard<std::mutex> guard(*allocation_map_lock_);
  auto it = allocation_map_->find(memory_address);
  if (it == allocation_map_->end()) {
    return nullptr;
  }

  auto gpu_mem = wsl::thunk::GpuMemory::Convert(it->second.handle);
  return gpu_mem;
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
#if defined(__linux__)
  struct sysinfo info;
  if (sysinfo(&info) != 0)
    return false;
  return SizeInBytes <= info.freeram;
#else
  return true;
#endif
}

void* BlockAllocator::alloc(size_t request_size, size_t& allocated_size) const {
  void *address;
  HsaMemFlags MemFlags;

  MemFlags.Value = 0;
  MemFlags.ui32.CoarseGrain = 1;
  MemFlags.ui32.NoSubstitute = 1;
  allocated_size = rocr::AlignUp(request_size, block_size());
  if (HSAKMT_STATUS_SUCCESS == hsaKmtAllocMemoryAlignInternal(1, allocated_size, 0, MemFlags, &address, true))
    return address;

  return nullptr;
}

void BlockAllocator::free(void* ptr, size_t length) const {
  if (HSAKMT_STATUS_SUCCESS != hsaKmtFreeMemoryInternal(ptr, length, true))
    pr_err("wsl-thunk: BlockAllocator::free() err, address %p, length:%zu\n", ptr, length);
}

void reset_suballocator(void) {
  fragment_allocator_->reset();
}

void trim_suballocator(void) {
  fragment_allocator_->trim();
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

  uint32_t node = (PreferredNode < CpuNodes()) ? dxg_runtime->default_node : PreferredNode;
  wsl::thunk::WDDMDevice *dev = get_wddmdev(node);
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
  if ((PreferredNode < CpuNodes() && MemFlags.ui32.HostAccess)
    || dxg_runtime->zfb_support || MemFlags.ui32.GTTAccess) {
    if (SizeInBytes > dxg_runtime->max_single_alloc_size)
      return HSAKMT_STATUS_NO_MEMORY;

    if (dxg_runtime->check_avail_sysram && !isSystemMemoryAvailable(SizeInBytes))
      return HSAKMT_STATUS_NO_MEMORY;

    /* If allocate VRAM under ZFB mode */
    if (dxg_runtime->zfb_support && MemFlags.ui32.NonPaged == 1)
      MemFlags.ui32.CoarseGrain = 1;

    // AllocateNonPaged == AllocateIPC
    // @todo it requires a proper solution. In Windows GPU accessible memory isn't pageable
    // and IPC detection is wrong in general
#if defined(__linux__)
    create_info.flags.sysmem_ipc_sig_exporter = !!(MemFlags.ui32.NonPaged && !MemFlags.ui32.GTTAccess);
#endif
    create_info.domain = thunk_proxy::AllocDomain::kSystem;
  } else {
    create_info.domain = thunk_proxy::AllocDomain::kLocal;
  }

  if (!MemFlags.ui32.CoarseGrain)
    create_info.mem_flags = thunk_proxy::kFineGrain;

  //In hsa-runtime, only kernarg region set Uncached.
  if (MemFlags.ui32.Uncached)
    create_info.mem_flags |= thunk_proxy::kKernarg;

  if (MemFlags.ui32.QueueObject) {
    create_info.mem_flags |= thunk_proxy::kQueueObject;
  }

  create_info.flags.physical_only = MemFlags.ui32.NoAddress;
  create_info.flags.alloc_va = !create_info.flags.physical_only;
  create_info.flags.interprocess = MemFlags.ui32.NoAddress;
  create_info.flags.interprocess |= MemFlags.ui32.Contiguous;
  create_info.flags.physical_contiguous = MemFlags.ui32.Contiguous;
  create_info.flags.locked = MemFlags.ui32.NoSubstitute;//AllocatePinned
  create_info.flags.virtual_alloc = MemFlags.ui32.OnlyAddress;
  create_info.flags.blit_kernel_object =
      (MemFlags.ui32.ExecuteBlit && MemFlags.ui32.ExecuteAccess &&
      (create_info.domain == thunk_proxy::AllocDomain::kSystem));
  /*when only alloc virtual or only physical, it's vmm allocation, force to local*/
  if (create_info.flags.virtual_alloc || create_info.flags.physical_only
        || create_info.flags.physical_contiguous) {
    if (dxg_runtime->hsakmt_debug_sysmem) {
      create_info.domain = thunk_proxy::AllocDomain::kSystem;
    } else {
      create_info.domain = thunk_proxy::AllocDomain::kLocal;
    }
    SkipSubAlloc = true;
  }

  /* Only allow using the suballocator for ordinary VRAM.*/
  bool trim_safe = false;
  if (!SkipSubAlloc && create_info.domain == thunk_proxy::AllocDomain::kLocal) {
    /* just quickly skip SA if size is bigger than SA block size.*/
    gpusize real_size;
    if (create_info.size > GPU_HUGE_PAGE_SIZE)
      real_size = rocr::AlignUp(create_info.size, GPU_HUGE_PAGE_SIZE);
    else {
#if defined(__linux__)
      auto page_size = getpagesize();
#else
      gpusize page_size = 4096ULL;
#endif
      real_size = rocr::AlignUp(create_info.size, page_size);
    }
    if (real_size < fragment_allocator_->default_block_size()) {
      *MemoryAddress = fragment_allocator_->alloc(real_size);
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
    if (create_info.flags.physical_only ||
        (create_info.dmabuf_fd != 0 && create_info.dmabuf_fd != INVALID_DMABUF_FD))
      *MemoryAddress = reinterpret_cast<void*>(gpu_mem->HandleApeAddress());
    else
      *MemoryAddress = reinterpret_cast<void *>(gpu_mem->GpuAddress());

    (*allocation_map_)[*MemoryAddress] = Allocation(
        gpu_mem->GetGpuMemoryHandle(), *MemoryAddress, (uint64_t)*MemoryAddress,
        create_info.size, false, nullptr, SizeInBytes,
        MemFlags.ui32.GTTAccess ? 0 : PreferredNode, MemFlags.Value);
    return HSAKMT_STATUS_SUCCESS;
  } else if (trim_safe) {
    /* attempt to release memory from the block allocator and retry */
    fragment_allocator_->trim();
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
                                        MemoryAddress,
                                        !dxg_runtime->enable_thunk_sub_allocator);
}

HSAKMT_STATUS hsaKmtFreeMemoryInternal(void *MemoryAddress,
                                       HSAuint64 SizeInBytes,
                                       bool SkipSubAlloc) {
  CHECK_DXG_OPEN();

  if (!MemoryAddress)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  if (!SkipSubAlloc) {
    if (fragment_allocator_->free(MemoryAddress))
      return HSAKMT_STATUS_SUCCESS;
  }

  wsl::thunk::GpuMemory *gpu_mem = nullptr;
  {
    std::lock_guard<std::mutex> gard(*allocation_map_lock_);
    auto it = allocation_map_->find(MemoryAddress);
    if (it == allocation_map_->end()) {
      return HSAKMT_STATUS_ERROR;
    }

    gpu_mem = wsl::thunk::GpuMemory::Convert(it->second.handle);
    if (gpu_mem->IsQueueReferenced())
      return HSAKMT_STATUS_ERROR;

    wsl::thunk::GpuMemoryDescFlags flags;
    flags.reserved = gpu_mem->Flags();
    if (flags.is_imported_vram_ipc &&
      gpu_mem->DecSharedReference()) {
      pr_info("memory is still referenced\n");
      return HSAKMT_STATUS_SUCCESS;
    }

    if (it->second.dmabuf_fd >= 0) {
#if defined(__linux__)
      close(it->second.dmabuf_fd);
#endif
      it->second.dmabuf_fd = -1;
    }
    allocation_map_->erase(it);
  }

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
  auto it = allocation_map_->find(MemoryAddress);
  if (it == allocation_map_->end()) {
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
    auto it = allocation_map_->find(MemoryAddress);
    if (it == allocation_map_->end()) {
      return HSAKMT_STATUS_ERROR;
    }

    gpu_mem = wsl::thunk::GpuMemory::Convert(it->second.handle);
    gpu_mem->PutQueueReference();
  }
  if (gpu_mem == nullptr)
    return false;

  return true;
}

wsl::thunk::GpuMemory *get_gpu_mem(void *MemoryAddress) {
  std::lock_guard<std::mutex> gard(*allocation_map_lock_);
  auto it = allocation_map_->find(MemoryAddress);
  if (it == allocation_map_->end()) {
    return nullptr;
  }

  return wsl::thunk::GpuMemory::Convert(it->second.handle);
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

  if (!dxg_runtime->hsakmt_is_dgpu)
    /* TODO: support mixed APU and dGPU configurations */
    return HSAKMT_STATUS_NOT_SUPPORTED;

  return HSAKMT_STATUS_SUCCESS;
}

bool is_ipc_sysmemfd(uint64_t fd) {
#if defined(__linux__)
  std::string fdPath = "/proc/self/fd/" + std::to_string(fd);
  char linkTarget[256];
  size_t bytes = readlink(fdPath.c_str(), linkTarget, sizeof(linkTarget) - 1);
  if (bytes == -1)
    return false;
  linkTarget[bytes] = '\0';
  return strstr(linkTarget, "rocr4wsl_gtt") != nullptr;
#else
  assert(!"Unimplemeted!");
  return true;
#endif
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

#if defined(__linux__)
  if (is_ipc_sysmemfd(GraphicsResourceHandle)) {
    GraphicsResourceInfo->NodeId = dxg_runtime->default_node;
    pr_info("skip register sysmemfd. It would be released in next step\n");
    return HSAKMT_STATUS_SUCCESS;
  }
#endif

  if (NumberOfNodes == 0) {
    RegisterFlags.ui32.requiresVAddr = 0;
    NumberOfNodes = 1;
    NodeArray = (HSAuint32*)&(dxg_runtime->default_node);
  }
  else {
    RegisterFlags.ui32.requiresVAddr = 1;
  }

  pr_debug("number of nodes %lu\n", NumberOfNodes);
  wsl::thunk::GpuMemoryHandle mem_handle;

  ret = import_dmabuf_fd(GraphicsResourceHandle, NodeArray[0], RegisterFlags.ui32.requiresVAddr,
                         false, &mem_handle, RegisterFlags.ui32.kmtHandle);
  if (ret != HSAKMT_STATUS_SUCCESS) {
    pr_err("hsaKmtRegisterGraphicsHandleToNodesExt: import_dmabuf_fd failed, "
           "GraphicsResourceHandle: %lu, NodeId: %u\n",
           GraphicsResourceHandle, NodeArray[0]);
    return ret;
  }
  wsl::thunk::GpuMemory *gpu_mem = wsl::thunk::GpuMemory::Convert(mem_handle);
  GraphicsResourceInfo->NodeId = gpu_mem->GetDevice()->NodeId();
  GraphicsResourceInfo->SizeInBytes = gpu_mem->ClientSize();
  GraphicsResourceInfo->MemoryAddress = RegisterFlags.ui32.requiresVAddr ?
                                          reinterpret_cast<void *>(gpu_mem->GpuAddress()):
                                          reinterpret_cast<void*>(gpu_mem->HandleApeAddress());

  return ret;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtExportDMABufHandle(void *MemoryAddress,
                                                 HSAuint64 MemorySizeInBytes,
                                                 int *DMABufFd,
                                                 HSAuint64 *Offset) {
  CHECK_DXG_OPEN();

  std::lock_guard<std::mutex> gard(*allocation_map_lock_);

  auto it = allocation_map_->upper_bound(MemoryAddress);
  if (it != allocation_map_->begin()) {
    --it;
    auto gpu_mem = wsl::thunk::GpuMemory::Convert(it->second.handle);
    if (!gpu_mem->IsPhysicalCreated()) {
      auto code = gpu_mem->CreatePhysicalMemory();
      if (code != ErrorCode::Success) {
        return HSAKMT_STATUS_OUT_OF_RESOURCES;
      }
    }
    if (it->second.dmabuf_fd == -1) {
      auto code = gpu_mem->ExportPhysicalHandle(DMABufFd);
      if (code != ErrorCode::Success)
        return HSAKMT_STATUS_ERROR;
      it->second.dmabuf_fd = *DMABufFd;
      *Offset = reinterpret_cast<uint64_t>(MemoryAddress) - it->second.gpu_addr;
    }
#if defined(__linux__)
    *DMABufFd = dup(it->second.dmabuf_fd);
#else
    *DMABufFd = it->second.dmabuf_fd;
#endif
    return HSAKMT_STATUS_SUCCESS;
  }

  return HSAKMT_STATUS_ERROR;
}


HSAKMT_STATUS import_dmabuf_fd(uint64_t DMABufFd, uint32_t NodeId, bool alloc_va, bool is_ipc_memfd,
                               wsl::thunk::GpuMemoryHandle* GpuMemHandle, bool is_kmt_handle) {
  CHECK_DXG_OPEN();

  *GpuMemHandle = nullptr;
  wsl::thunk::WDDMDevice* dev = get_wddmdev(NodeId);
  wsl::thunk::GpuMemory *gpu_mem = nullptr;
  wsl::thunk::GpuMemoryCreateInfo create_info{};
  create_info.dmabuf_fd = DMABufFd;
  create_info.flags.alloc_va = alloc_va;
  create_info.flags.kmt_handle_importer = is_kmt_handle ? 1 : 0;

#if defined(__linux__)
  if (is_ipc_memfd) {
    struct stat st;
    fstat(DMABufFd, &st);
    uint64_t sz = st.st_size;
    if (4096 <= sz && sz < dxg_runtime->SystemHeapSize() && (sz & 0xfff) == 0) {
      pr_debug("DMABufFd %lu is sys mem fd(IPC signal), get size:%ld from it\n", DMABufFd, st.st_size);
      create_info.flags.sysmem_ipc_sig_importer = 1;        // set to 1 when backend is system memory
      create_info.size = st.st_size;
    }
  }
#endif

  gpusize gpu_va = 0;
  auto code = dev->CreateGpuMemory(create_info, &gpu_mem, &gpu_va);
  if (code == ErrorCode::SameProcessSameDevice) {
    /* Unit_hipMemPoolExportToShareableHandle_SameProc */
    pr_info("imported from same process, use the old one\n");
    std::lock_guard<std::mutex> gard(*allocation_map_lock_);
    auto it = allocation_map_->find((void*)gpu_va);
    if (it == allocation_map_->end()) {
      pr_err("where's the conflict buffer? va %#lx\n", create_info.va_hint);
      return HSAKMT_STATUS_ERROR;
    }
    wsl::thunk::GpuMemory *conflict_mem = wsl::thunk::GpuMemory::Convert(it->second.handle);
    conflict_mem->IncSharedReference();
    *GpuMemHandle = it->second.handle;
    return HSAKMT_STATUS_SUCCESS;
  } else if (code != ErrorCode::Success) {
    pr_err("fail to import fd, ret %d\n", (int)code);
    return HSAKMT_STATUS_ERROR;
  }

  void *MemoryAddress;
  if (alloc_va)
    MemoryAddress = reinterpret_cast<void *>(gpu_mem->GpuAddress());
  else
    MemoryAddress = reinterpret_cast<void*>(gpu_mem->HandleApeAddress());

  *GpuMemHandle = gpu_mem->GetGpuMemoryHandle();

  std::lock_guard<std::mutex> gard(*allocation_map_lock_);
  /*
   * the gpu_mem->Flags() need convert back from GpuMemoryCreateFlags to
   * HsaMemFlags, reference hsaKmtAllocMemoryAlign
   * */
  (*allocation_map_)[MemoryAddress] = Allocation(
    *GpuMemHandle, MemoryAddress, (uint64_t)MemoryAddress,
    gpu_mem->Size(), false, nullptr, gpu_mem->ClientSize(),
    NodeId, gpu_mem->Flags());

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetMemoryHandle(void* va, void* MemoryAddress, HSAuint64 SizeInBytes,
                      uint64_t* SharedMemoryHandle) {
  CHECK_DXG_OPEN();
  // find allocation map for MemorAddress
  auto phys_mem = GetGpuMemoryFromAddress(MemoryAddress);
  if (!phys_mem) return HSAKMT_STATUS_INVALID_HANDLE;

  auto va_mem = GetGpuMemoryFromAddress(va);
  // Check is this va is between range (old va : old va + size)
  if (!va_mem) {
    auto alloc = FindAllocation(va, SizeInBytes);
    if (alloc == nullptr) {
      return HSAKMT_STATUS_INVALID_HANDLE;
    }
  }

  if (dxg_runtime->hsakmt_debug_sysmem) {
    phys_mem->forceSysMem();
    phys_mem->SetCpuAddress(va);
    phys_mem->SetGpuAddress(reinterpret_cast<uint64_t>(va));
  }
  bool alloc_phys_mem = !phys_mem->IsPhysicalCreated();
  if (alloc_phys_mem) { // If phys mem handle is already created
    auto code =  phys_mem->CreatePhysicalMemory();
    if (code != ErrorCode::Success) {
      if (dxg_runtime->hsakmt_debug_sysmem) {
        phys_mem->SetGpuAddress(0);
        phys_mem->SetCpuAddress(nullptr);
      }
      return HSAKMT_STATUS_OUT_OF_RESOURCES;
    }
  }
  *SharedMemoryHandle = reinterpret_cast<uint64_t>(phys_mem->GetGpuMemoryHandle());
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtShareMemory(void *MemoryAddress, HSAuint64 SizeInBytes,
                  HsaSharedMemoryHandle *SharedMemoryHandle) {
  CHECK_DXG_OPEN();
  return HSAKMT_STATUS_NOT_IMPLEMENTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtRegisterSharedHandle(const HsaSharedMemoryHandle *SharedMemoryHandle,
                           void **MemoryAddress, HSAuint64 *SizeInBytes) {
  CHECK_DXG_OPEN();
  return HSAKMT_STATUS_NOT_IMPLEMENTED;
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

  {
    std::lock_guard<std::mutex> gard(*allocation_map_lock_);

    auto it = allocation_map_->find(MemoryAddress);
    if (it == allocation_map_->end()) {
      return HSAKMT_STATUS_SUCCESS;
    }

    auto *gpu_mem = wsl::thunk::GpuMemory::Convert(it->second.handle);
    wsl::thunk::GpuMemoryDescFlags flags;
    flags.reserved = gpu_mem->Flags();
    // IPC mem(vram)
    if (flags.is_imported_vram_ipc &&
      gpu_mem->DecSharedReference() == 0) {
      allocation_map_->erase(it);
      delete gpu_mem;
      return HSAKMT_STATUS_SUCCESS;
    }
    if (it->second.userptr) {
      allocation_map_->erase((void*)it->second.gpu_addr);
      allocation_map_->erase(it);
      delete gpu_mem;
      return HSAKMT_STATUS_SUCCESS;
    }
  }
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMapMemoryToGPU(void *MemoryAddress,
                                             HSAuint64 MemorySizeInBytes,
                                             HSAuint64 *AlternateVAGPU) {

  HSAuint64 NumberOfNodes = 1;
  HSAuint32 NodeArray[] = {dxg_runtime->default_node};
  HsaMemMapFlags MemMapFlags;
  MemMapFlags.Value = 0;

  return hsaKmtMapMemoryToGPUNodes(MemoryAddress, MemorySizeInBytes, AlternateVAGPU,
    MemMapFlags, NumberOfNodes, NodeArray);
}
HSAKMT_STATUS HSAKMTAPI hsaKmtMapMemoryToGPUNodes(
    void *MemoryAddress, HSAuint64 MemorySizeInBytes, HSAuint64 *AlternateVAGPU,
    HsaMemMapFlags MemMapFlags, HSAuint64 NumberOfNodes, HSAuint32 *NodeArray) {
  CHECK_DXG_OPEN();

  if (!MemoryAddress || !AlternateVAGPU) {
    pr_err("FIXME: mapping NULL pointer\n");
    return HSAKMT_STATUS_ERROR;
  }

  uint64_t start = rocr::AlignDown((uint64_t)MemoryAddress, 4096);
  uint64_t end =
      rocr::AlignUp((uint64_t)MemoryAddress + MemorySizeInBytes, 4096);

  void *aligned_ptr = (void *)start;
  size_t aligned_size = end - start;

  {
    if (nullptr != fragment_allocator_->block_base(aligned_ptr))
      return HSAKMT_STATUS_SUCCESS;
  }

  {
    std::lock_guard<std::mutex> gard(*allocation_map_lock_);
    auto it = FindAllocation(aligned_ptr, aligned_size);
    if (it != nullptr) {
      wsl::thunk::GpuMemory *gpu_mem = wsl::thunk::GpuMemory::Convert(it->handle);
      wsl::thunk::GpuMemoryDescFlags flags;
      flags.reserved = gpu_mem->Flags();
      // IPC mem
      if (flags.is_imported_vram_ipc) {

        auto code = gpu_mem->MapGpuVirtualAddress(gpu_mem->GpuAddress(), gpu_mem->Size());
        if (code != ErrorCode::Success)
          return HSAKMT_STATUS_ERROR;

        code = gpu_mem->MakeResident();
        if (code != ErrorCode::Success)
          return HSAKMT_STATUS_ERROR;

        wsl::thunk::WDDMDevice *dev = gpu_mem->GetDevice();
        if (!dev->WaitOnPagingFenceFromCpu())
          return HSAKMT_STATUS_ERROR;

        return HSAKMT_STATUS_SUCCESS;
      }
      if (gpu_mem->IsVirtual()) {
        gpu_mem->MapMemoryToVirtualAddress();
      }
      if (!it->userptr) {
      // GTT/Local mem
        if (it->size >= MemorySizeInBytes) {
          *AlternateVAGPU = (uint64_t)MemoryAddress;
          return HSAKMT_STATUS_SUCCESS;
        } else {
          return HSAKMT_STATUS_ERROR;
        }
      }
    }

    // userptr mem
    it = FindAllocation(MemoryAddress, aligned_size);
    if (it != nullptr) {
      if (it->userptr && it->size >= MemorySizeInBytes) {
        *AlternateVAGPU =
            (uintptr_t)it->gpu_addr +
            ((uintptr_t)MemoryAddress - (uintptr_t)it->cpu_addr);
        return HSAKMT_STATUS_SUCCESS;
      }
    }
  }

  // map userptr
  wsl::thunk::WDDMDevice *dev = get_wddmdev(NodeArray[0]);
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
   (*allocation_map_)[MemoryAddress] =
        Allocation(handle, aligned_ptr, addr, aligned_size, true, MemoryAddress,
                   MemorySizeInBytes);
    (*allocation_map_)[(void *)addr] =
        Allocation(handle, aligned_ptr, addr, aligned_size, true, nullptr,
                   MemorySizeInBytes);
  }

  *AlternateVAGPU = addr + ((uintptr_t)MemoryAddress - (uintptr_t)aligned_ptr);

  return HSAKMT_STATUS_SUCCESS;
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
    if (nullptr != fragment_allocator_->block_base(MemoryAddress))
      return HSAKMT_STATUS_SUCCESS;
  }

  wsl::thunk::GpuMemory *gpu_mem = nullptr;
  {
    std::lock_guard<std::mutex> gard(*allocation_map_lock_);

    auto it = allocation_map_->find(MemoryAddress);
    if (it == allocation_map_->end()) {
      return HSAKMT_STATUS_ERROR;
    }

    gpu_mem = wsl::thunk::GpuMemory::Convert(it->second.handle);
    if (gpu_mem->IsQueueReferenced())
      return HSAKMT_STATUS_ERROR;

    // IPC mem
    wsl::thunk::GpuMemoryDescFlags flags;
    flags.reserved = gpu_mem->Flags();
    if (flags.is_imported_vram_ipc &&
        !gpu_mem->IsSharedFromSameProcess()) {
      auto code = gpu_mem->UnmapGpuVirtualAddress(gpu_mem->GpuAddress(), gpu_mem->Size());
      if (code != ErrorCode::Success)
        return HSAKMT_STATUS_ERROR;
      gpu_mem->Evict();

      return HSAKMT_STATUS_SUCCESS;
    }
  }

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
  wsl::thunk::WDDMDevice* dev = get_wddmdev(NodeId);
  // Gfx9+ only need GbAddrConfig
  config->GbAddrConfig = dev->GbAddrConfig();
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtQueryPointerInfo(const void *Pointer,
                                               HsaPointerInfo *PointerInfo) {
  CHECK_DXG_OPEN();

  if (!Pointer || !PointerInfo)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  pr_debug("pointer %p\n", Pointer);

  memset(PointerInfo, 0, sizeof(HsaPointerInfo));

  wsl::thunk::GpuMemory *gpu_mem = nullptr;
  Allocation allocation_info;
  bool found = false;
  {
    std::lock_guard<std::mutex> gard(*allocation_map_lock_);
    auto it = allocation_map_->upper_bound(Pointer);
    if (it != allocation_map_->begin()) {
      --it;
      if (Pointer >= it->first &&
        (Pointer < reinterpret_cast<const uint8_t*>(it->first) + it->second.size_requested)) {
        allocation_info = it->second;
        gpu_mem = wsl::thunk::GpuMemory::Convert(it->second.handle);
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
  } else if (gpu_mem->IsVirtual()) {
    PointerInfo->Type = HSA_POINTER_RESERVED_ADDR;
  } else {
    PointerInfo->Type = HSA_POINTER_ALLOCATED;
    PointerInfo->SizeInBytes = allocation_info.size_requested;
  }

  PointerInfo->Node = allocation_info.node_id;
  PointerInfo->MemFlags.Value = allocation_info.mem_flags_value;
  PointerInfo->CPUAddress = allocation_info.cpu_addr;
  PointerInfo->GPUAddress = allocation_info.gpu_addr;
  PointerInfo->UserData = allocation_info.rocr_userdata;

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtSetMemoryUserData(const void *Pointer,
                                                void *UserData) {
  CHECK_DXG_OPEN();

  uint64_t aligned_ptr = rocr::AlignDown((uint64_t)Pointer, 4096);

  std::lock_guard<std::mutex> gard(*allocation_map_lock_);
  auto it = allocation_map_->find((void *)aligned_ptr);
  if (it != allocation_map_->end()) {
    it->second.rocr_userdata = UserData;
    return HSAKMT_STATUS_SUCCESS;
  }

  return HSAKMT_STATUS_ERROR;
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


HSAKMT_STATUS HSAKMTAPI hsaKmtHandleImport(const HsaExternalHandleDesc* import_desc,
    					HsaHandleImportResult* import_res, HsaHandleImportFlags* flags)
{
	CHECK_DXG_OPEN();
  if (import_desc->type != HSA_EXTERNAL_HANDLE_DMA_BUF) {
    assert(!"not supported\n");
    return HSAKMT_STATUS_NOT_SUPPORTED;
  }

  if (static_cast<int>(import_desc->fd) == -1) {
    import_res->buf_handle = 0;
    return HSAKMT_STATUS_ERROR;
  }

  wsl::thunk::GpuMemoryHandle mem_handle;
  wsl::thunk::WDDMDevice *pDevice = reinterpret_cast<wsl::thunk::WDDMDevice *>(import_desc->device_handle);
  bool is_ipc_memfd = is_ipc_sysmemfd(import_desc->fd);
  bool alloc_va = is_ipc_memfd;

  // kmt handle importer is false for dma_buf_fd
  HSAKMT_STATUS ret = import_dmabuf_fd(import_desc->fd, pDevice->NodeId(), alloc_va, is_ipc_memfd,
                                       &mem_handle, false);
  if (ret == HSAKMT_STATUS_SUCCESS) {
    //use GpuMemory object handle as drm buf handle
    import_res->buf_handle = reinterpret_cast<HsaMemoryObjectHandle>(mem_handle);
    return HSAKMT_STATUS_SUCCESS;
  }
  return HSAKMT_STATUS_ERROR;
}


HSAKMT_STATUS HSAKMTAPI hsaKmtMemoryVaMap(HsaMemoryObjectHandle Handle,
              HSAuint64 offset, HSAuint64 size, HSAuint64 addr,
              HsaMemoryMapFlags flags)
{
	CHECK_DXG_OPEN();
  wsl::thunk::GpuMemory* gpu_mem = reinterpret_cast<wsl::thunk::GpuMemory*>(Handle);
  assert(gpu_mem != nullptr);

  auto code = gpu_mem->MapGpuVirtualAddress(static_cast<gpusize>(addr), size, offset);
  if (code != ErrorCode::Success)
    return HSAKMT_STATUS_ERROR;

  code = gpu_mem->MakeResident();
  if (code != ErrorCode::Success)
    return HSAKMT_STATUS_ERROR;

  // Wait on paging fence to ensure map and residency are completed. MakeResident()
  // updates the same paging fence as Map(), hence it's safe to wait for the last one.
  if (!gpu_mem->GetDevice()->WaitOnPagingFenceFromCpu()) {
    return HSAKMT_STATUS_ERROR;
  }

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMemoryVaUnmap(HsaMemoryObjectHandle Handle,
              HSAuint64 offset, HSAuint64 size, HSAuint64 addr)
{
	CHECK_DXG_OPEN();
  wsl::thunk::GpuMemory* gpu_mem = reinterpret_cast<wsl::thunk::GpuMemory*>(Handle);
  assert(gpu_mem != nullptr);

  auto code = gpu_mem->UnmapGpuVirtualAddress(static_cast<gpusize>(addr), size, offset);
  if (code != ErrorCode::Success)
    return HSAKMT_STATUS_ERROR;
  // Wait on paging fence to ensure unmap is completed. Evict() doesn't
  // update the fence.
  if (!gpu_mem->GetDevice()->WaitOnPagingFenceFromCpu()) {
    return HSAKMT_STATUS_ERROR;
  }
  gpu_mem->Evict();
	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMemHandleFree(HsaMemoryObjectHandle Handle)
{
	CHECK_DXG_OPEN();
  wsl::thunk::GpuMemory* gpu_mem = reinterpret_cast<wsl::thunk::GpuMemory*>(Handle);
  delete gpu_mem;
	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMemoryCpuMap(HsaMemoryObjectHandle Handle,
              void** out_cpu_ptr)
{
	CHECK_DXG_OPEN();
  wsl::thunk::GpuMemory *gpu_mem = reinterpret_cast<wsl::thunk::GpuMemory *>(Handle);
  *out_cpu_ptr = nullptr;
  if (gpu_mem->IsSysMemFd()) {
    *out_cpu_ptr = gpu_mem->CpuAddress();
  }
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMemoryGetCpuAddr(HsaAMDGPUDeviceHandle DeviceHandle,
              HsaMemoryObjectHandle MemoryHandle, HSAint32* fd, HSAuint64* cpu_addr)
{
	CHECK_DXG_OPEN();
  wsl::thunk::GpuMemory* gpu_mem = reinterpret_cast<wsl::thunk::GpuMemory*>(MemoryHandle);
  assert(gpu_mem != nullptr);
  cpu_addr =  static_cast<HSAuint64*>(gpu_mem->CpuAddress());
  return HSAKMT_STATUS_SUCCESS;
}