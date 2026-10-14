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
#include <cinttypes>
#include <sys/types.h>
#if defined(__linux__)
#include <sys/mman.h>
#include <sys/sysinfo.h>
#endif
#include <sys/stat.h>
#include <fcntl.h>
#include <utility>
#include <vector>
#include "impl/wddm/gpu_memory.h"
#include "util/simple_heap.h"
#include "util/os.h"

struct Allocation {
  Allocation()
      : handle(0), cpu_addr(0), gpu_addr(0), size(0), userptr(false),
        user_data(nullptr), size_requested(0), node_id(0), mem_flags_value(0),
        dmabuf_fd(-1), rocr_userdata(nullptr), owner_node(0) {}
  Allocation(wsl::thunk::GpuMemoryHandle handle_arg, void *cpu_addr_arg,
             uint64_t gpu_addr_arg, size_t size_arg, bool userptr_arg = false,
             void *user_data_arg = nullptr, size_t user_size_arg = 0,
             HSAuint32 node_id_arg = 0, HSAuint32 mem_flags_value_arg = 0,
             HSAuint32 owner_node_arg = 0)
      : handle(handle_arg), cpu_addr(cpu_addr_arg), gpu_addr(gpu_addr_arg),
        size(size_arg), userptr(userptr_arg), user_data(user_data_arg),
        size_requested(user_size_arg), node_id(node_id_arg),
        mem_flags_value(mem_flags_value_arg), dmabuf_fd(-1), rocr_userdata(nullptr),
        owner_node(owner_node_arg) {}

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
  /* WDDM adapter that owns the primary GpuMemory. Distinct from node_id, which
   * stays the caller's PreferredNode (0 for GTT). */
  HSAuint32 owner_node;
  /* Peer GpuMemory views, indexed by WDDM adapter. The owner slot holds handle. */
  std::vector<wsl::thunk::GpuMemoryHandle> mapped;
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

static void set_mapped_handle(Allocation &alloc, HSAuint32 node,
                              wsl::thunk::GpuMemoryHandle handle) {
  if (is_cpu_hsa_node(node))
    return;

  const size_t idx = gpu_node_to_wddm_index(node);
  if (alloc.mapped.size() <= idx)
    alloc.mapped.resize(idx + 1, nullptr);
  alloc.mapped[idx] = handle;
}

static wsl::thunk::GpuMemoryHandle mapped_handle(const Allocation &alloc,
                                                 HSAuint32 node) {
  if (is_cpu_hsa_node(node))
    return nullptr;

  const size_t idx = gpu_node_to_wddm_index(node);
  return idx < alloc.mapped.size() ? alloc.mapped[idx] : nullptr;
}

/* Caller holds allocation_map_lock_. The gpu-address mirror is the second
 * userptr entry; keep its peer list in step with the cpu-address entry. */
static void sync_userptr_mirror_mapped(Allocation &alloc, HSAuint32 node,
                                       wsl::thunk::GpuMemoryHandle handle) {
  if (!alloc.userptr)
    return;

  auto it = allocation_map_->find(reinterpret_cast<void *>(alloc.gpu_addr));
  if (it == allocation_map_->end())
    return;

  if (handle) {
    set_mapped_handle(it->second, node, handle);
    return;
  }

  if (is_cpu_hsa_node(node))
    return;

  const size_t idx = gpu_node_to_wddm_index(node);
  if (idx < it->second.mapped.size())
    it->second.mapped[idx] = nullptr;
}

static void teardown_mapped_node(Allocation &alloc, HSAuint32 node,
                                 wsl::thunk::GpuMemory *primary) {
  if (is_cpu_hsa_node(node) || primary == nullptr)
    return;

  const size_t idx = gpu_node_to_wddm_index(node);
  if (idx >= alloc.mapped.size() || !alloc.mapped[idx])
    return;

  if (node != alloc.owner_node) {
    auto *peer_mem = wsl::thunk::GpuMemory::Convert(alloc.mapped[idx]);
    auto code = peer_mem->UnmapGpuVirtualAddress(primary->GpuAddress(),
                                                primary->Size());
    if (code != ErrorCode::Success)
      pr_err("teardown_mapped_node: unmap failed for node %u\n", node);
    peer_mem->Evict();
    delete peer_mem;
  }

  alloc.mapped[idx] = nullptr;
  sync_userptr_mirror_mapped(alloc, node, nullptr);
}

static HSAKMT_STATUS map_memory_to_nodes(wsl::thunk::GpuMemoryHandle handle,
                                         Allocation &alloc_info,
                                         HSAuint64 NumberOfNodes,
                                         HSAuint32 *NodeArray) {
  wsl::thunk::GpuMemory *gpu_mem = wsl::thunk::GpuMemory::Convert(handle);
  std::vector<HSAuint32> mapped_this_call;

  auto rollback = [&]() {
    for (auto mapped_node : mapped_this_call)
      teardown_mapped_node(alloc_info, mapped_node, gpu_mem);
    return HSAKMT_STATUS_ERROR;
  };

  for (HSAuint64 idx = 0; idx < NumberOfNodes; ++idx) {
    HSAuint32 node = NodeArray[idx];
    if (mapped_handle(alloc_info, node))
      continue;

    wsl::thunk::WDDMDevice *dev = get_wddmdev(node);
    if (!dev)
      return rollback();

    if (node == alloc_info.owner_node) {
      set_mapped_handle(alloc_info, node, handle);
      sync_userptr_mirror_mapped(alloc_info, node, handle);
      mapped_this_call.push_back(node);
      continue;
    }

    wsl::thunk::GpuMemory *my_mem = nullptr;
    if (gpu_mem->IsSystem() || gpu_mem->IsUserMemory()) {
      auto code = dev->MapFromPeerDev(gpu_mem, &my_mem);
      if (code != ErrorCode::Success || !my_mem) {
        pr_err("hsaKmtMapMemoryToGPUNodes: failed to map memory to node %u\n",
               node);
        return rollback();
      }
    } else {
      pr_err("not implemented\n");
      return rollback();
    }

    set_mapped_handle(alloc_info, node, my_mem->GetGpuMemoryHandle());
    sync_userptr_mirror_mapped(alloc_info, node, my_mem->GetGpuMemoryHandle());
    mapped_this_call.push_back(node);
    pr_debug("hsaKmtMapMemoryToGPUNodes: mapped memory %#" PRIx64 " to node %u\n",
             alloc_info.gpu_addr, node);
  }

  return HSAKMT_STATUS_SUCCESS;
}

static Allocation* FindAllocationByGpuMem(wsl::thunk::GpuMemory* gpu_mem) {
  const auto handle = gpu_mem->GetGpuMemoryHandle();
  for (auto& entry : *allocation_map_) {
    if (entry.second.handle == handle)
      return &entry.second;
    for (auto mapped : entry.second.mapped) {
      if (mapped == handle)
        return &entry.second;
    }
  }
  return nullptr;
}

/* Record a peer view on every allocation entry that owns this primary handle.
 * Caller holds allocation_map_lock_. */
static void note_peer_view(wsl::thunk::GpuMemoryHandle primary, HSAuint32 node,
                           wsl::thunk::GpuMemoryHandle view) {
  for (auto& entry : *allocation_map_) {
    if (entry.second.handle == primary)
      set_mapped_handle(entry.second, node, view);
  }
}

static bool validate_nodes(HSAuint64 NumberOfNodes, HSAuint32 *NodeArray) {
  if (NumberOfNodes == 0 || !NodeArray)
    return false;

  for (HSAuint64 i = 0; i < NumberOfNodes; i++) {
    if (get_wddmdev(NodeArray[i]) == nullptr) {
      pr_err("Invalid node id %u\n", NodeArray[i]);
      return false;
    }
  }
  return true;
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

  uint32_t node;
  if (PreferredNode < static_cast<HSAuint32>(CpuNodes())) {
    if (dxg_runtime->default_node.empty())
      return HSAKMT_STATUS_ERROR;
    node = dxg_runtime->default_node[0];
  } else {
    node = PreferredNode;
  }
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
    create_info.domain = Wkmi::AllocDomain::kSystem;
  } else {
    create_info.domain = Wkmi::AllocDomain::kLocal;
  }

  if (!MemFlags.ui32.CoarseGrain)
    create_info.mem_flags = Wkmi::kFineGrain;

  //In hsa-runtime, only kernarg region set Uncached.
  if (MemFlags.ui32.Uncached)
    create_info.mem_flags |= Wkmi::kKernarg;

  if (MemFlags.ui32.QueueObject) {
    create_info.mem_flags |= Wkmi::kQueueObject;
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
      (create_info.domain == Wkmi::AllocDomain::kSystem));
  /*when only alloc virtual or only physical, it's vmm allocation, force to local*/
  if (create_info.flags.virtual_alloc || create_info.flags.physical_only
        || create_info.flags.physical_contiguous) {
    if (dxg_runtime->hsakmt_debug_sysmem) {
      create_info.domain = Wkmi::AllocDomain::kSystem;
    } else {
      create_info.domain = Wkmi::AllocDomain::kLocal;
    }
    SkipSubAlloc = true;
  }

  /* Only allow using the suballocator for ordinary VRAM.*/
  bool trim_safe = false;
  if (!SkipSubAlloc && create_info.domain == Wkmi::AllocDomain::kLocal) {
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

    Allocation alloc(
        gpu_mem->GetGpuMemoryHandle(), *MemoryAddress, (uint64_t)*MemoryAddress,
        create_info.size, false, nullptr, SizeInBytes,
        MemFlags.ui32.GTTAccess ? 0 : PreferredNode, MemFlags.Value,
        static_cast<HSAuint32>(gpu_mem->GetDevice()->NodeId()));
    set_mapped_handle(alloc, alloc.owner_node, gpu_mem->GetGpuMemoryHandle());
    (*allocation_map_)[*MemoryAddress] = std::move(alloc);
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

    /* Drop peer views before the primary VA reservation goes away. */
    for (size_t idx = 0; idx < it->second.mapped.size(); ++idx) {
      const HSAuint32 node = wddm_index_to_gpu_node(static_cast<uint32_t>(idx));
      teardown_mapped_node(it->second, node, gpu_mem);
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

  if (dev->VramAvail(AvailableBytes) != HSA_STATUS_SUCCESS) {
    return HSAKMT_STATUS_ERROR;
  }

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
  return false;
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
    if (dxg_runtime->default_node.empty())
      return HSAKMT_STATUS_ERROR;
    GraphicsResourceInfo->NodeId = dxg_runtime->default_node[0];

    struct stat st;
    if (fstat(static_cast<int>(GraphicsResourceHandle), &st) == 0)
      GraphicsResourceInfo->SizeInBytes = st.st_size;

    pr_info("skip register sysmemfd. It would be released in next step\n");
    return HSAKMT_STATUS_SUCCESS;
  }
#endif

  if (NumberOfNodes == 0) {
    RegisterFlags.ui32.requiresVAddr = 0;
    if (dxg_runtime->default_node.empty())
      return HSAKMT_STATUS_ERROR;
    NumberOfNodes = dxg_runtime->default_node.size();
    NodeArray = dxg_runtime->default_node.data();
  }
  else {
    RegisterFlags.ui32.requiresVAddr = 1;
  }

  pr_debug("number of nodes %" PRIu64 "\n", NumberOfNodes);
  wsl::thunk::GpuMemoryHandle mem_handle;

  ret = import_dmabuf_fd(GraphicsResourceHandle, NumberOfNodes, NodeArray,
                         RegisterFlags.ui32.requiresVAddr,
                         false, &mem_handle, RegisterFlags.ui32.kmtHandle,
                         GraphicsResourceInfo->SizeHintInBytes);
  if (ret != HSAKMT_STATUS_SUCCESS) {
    pr_err("hsaKmtRegisterGraphicsHandleToNodesExt: import_dmabuf_fd failed, "
           "GraphicsResourceHandle: %" PRIu64 ", NodeId: %u\n",
           GraphicsResourceHandle, NodeArray[0]);
    return ret;
  }
  wsl::thunk::GpuMemory *gpu_mem = wsl::thunk::GpuMemory::Convert(mem_handle);
  GraphicsResourceInfo->NodeId = gpu_mem->GetDevice()->NodeId();
  GraphicsResourceInfo->SizeInBytes = gpu_mem->ClientSize();
  GraphicsResourceInfo->MemoryAddress = RegisterFlags.ui32.requiresVAddr ?
                                          reinterpret_cast<void *>(gpu_mem->GpuAddress()):
                                          reinterpret_cast<void*>(gpu_mem->HandleApeAddress());
  // Populate swizzle metadata for the ROCr interop layer to use as a fallback when
  // the DXX extension (CLQueryResource11/CLQueryResource) or GL extension(wglResourceAttachAMD)
  // is unavailable.
  const HsaWddmSurfaceMetadata* meta = gpu_mem->GetSurfaceMetadata();
  GraphicsResourceInfo->Metadata = meta;
  GraphicsResourceInfo->MetadataSizeInBytes = meta ? sizeof(HsaWddmSurfaceMetadata) : 0;

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
    }
    *Offset = reinterpret_cast<uint64_t>(MemoryAddress) - it->second.gpu_addr;
#if defined(__linux__)
    *DMABufFd = dup(it->second.dmabuf_fd);
#else
    *DMABufFd = it->second.dmabuf_fd;
#endif
    return HSAKMT_STATUS_SUCCESS;
  }

  return HSAKMT_STATUS_ERROR;
}


HSAKMT_STATUS import_dmabuf_fd(uint64_t DMABufFd, HSAuint64 NumberOfNodes, HSAuint32* NodeArray,
                               bool alloc_va, bool is_ipc_memfd,
                               wsl::thunk::GpuMemoryHandle* GpuMemHandle, bool is_kmt_handle,
                               uint64_t size_hint) {
  CHECK_DXG_OPEN();

  if (!validate_nodes(NumberOfNodes, NodeArray))
    return HSAKMT_STATUS_ERROR;

  *GpuMemHandle = nullptr;
  wsl::thunk::WDDMDevice* dev = get_wddmdev(NodeArray[0]);
  wsl::thunk::GpuMemory *gpu_mem = nullptr;
  wsl::thunk::GpuMemoryCreateInfo create_info{};
  create_info.dmabuf_fd = DMABufFd;
  create_info.flags.alloc_va = alloc_va;
  create_info.flags.kmt_handle_importer = is_kmt_handle ? 1 : 0;
  create_info.size = size_hint;

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
      pr_err("where's the conflict buffer? va %#" PRIx64 "\n", create_info.va_hint);
      return HSAKMT_STATUS_ERROR;
    }
    wsl::thunk::GpuMemory *conflict_mem = wsl::thunk::GpuMemory::Convert(it->second.handle);
    conflict_mem->IncSharedReference();
    *GpuMemHandle = it->second.handle;
    if (NumberOfNodes > 1)
      return map_memory_to_nodes(*GpuMemHandle, it->second, NumberOfNodes, NodeArray);
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
  Allocation alloc(
    *GpuMemHandle, MemoryAddress, (uint64_t)MemoryAddress,
    gpu_mem->Size(), false, nullptr, gpu_mem->ClientSize(),
    NodeArray[0], gpu_mem->Flags(),
    static_cast<HSAuint32>(gpu_mem->GetDevice()->NodeId()));
  set_mapped_handle(alloc, alloc.owner_node, *GpuMemHandle);
  if (NumberOfNodes > 1) {
    auto ret = map_memory_to_nodes(*GpuMemHandle, alloc, NumberOfNodes, NodeArray);
    if (ret != HSAKMT_STATUS_SUCCESS) {
      delete gpu_mem;
      return ret;
    }
  }
  (*allocation_map_)[MemoryAddress] = std::move(alloc);

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
  }
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMapMemoryToGPU(void *MemoryAddress,
                                             HSAuint64 MemorySizeInBytes,
                                             HSAuint64 *AlternateVAGPU) {

  HsaMemFlags MemFlags;
  MemFlags.Value = 0;
  MemFlags.ui32.CoarseGrain = 1;

  if (dxg_runtime->default_node.empty())
    return HSAKMT_STATUS_ERROR;

  return hsaKmtMapMemoryToGPUNodes(MemoryAddress, MemorySizeInBytes, AlternateVAGPU,
    MemFlags, dxg_runtime->default_node.size(), dxg_runtime->default_node.data());
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMapMemoryToGPUNodes(
    void *MemoryAddress, HSAuint64 MemorySizeInBytes, HSAuint64 *AlternateVAGPU,
    HsaMemFlags MemFlags, HSAuint64 NumberOfNodes, HSAuint32 *NodeArray) {
  CHECK_DXG_OPEN();

  if (!MemoryAddress || !AlternateVAGPU) {
    pr_err("FIXME: mapping NULL pointer\n");
    return HSAKMT_STATUS_ERROR;
  }

  if (!validate_nodes(NumberOfNodes, NodeArray)) {
    pr_err("invalid node list\n");
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

        wsl::thunk::WDDMDevice *dev = get_wddmdev(it->owner_node);
        if (!dev || !dev->WaitOnPagingFenceFromCpu())
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
          return map_memory_to_nodes(it->handle, *it, NumberOfNodes, NodeArray);
        } else {
          return HSAKMT_STATUS_ERROR;
        }
      }
    }

    // Reuse the user-pointer mapping already registered for this address.
    auto *allocation = FindAllocation(MemoryAddress, aligned_size);
    if (allocation != nullptr && allocation->userptr) {
      wsl::thunk::GpuMemory::Convert(allocation->handle)->IncMappingCount();
      *AlternateVAGPU = (uintptr_t)allocation->gpu_addr +
          ((uintptr_t)MemoryAddress - (uintptr_t)allocation->cpu_addr);
      return map_memory_to_nodes(allocation->handle, *allocation, NumberOfNodes,
                                 NodeArray);
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
  create_info.domain = Wkmi::kUserMemory;
  create_info.size = aligned_size;
  create_info.user_ptr = aligned_ptr;
  // create_info.mem_flags = 0 means coarse grain by default
  if (!MemFlags.ui32.CoarseGrain) {
    create_info.mem_flags = Wkmi::kFineGrain;
  }
  auto code = dev->CreateGpuMemory(create_info, &gpu_mem);
  if (code == ErrorCode::Success) {
    addr = gpu_mem->GpuAddress();
    handle = gpu_mem->GetGpuMemoryHandle();
  } else {
    return HSAKMT_STATUS_ERROR;
  }

  {
    std::lock_guard<std::mutex> guard(*allocation_map_lock_);
    Allocation cpu_entry(handle, aligned_ptr, addr, aligned_size, true,
                         MemoryAddress, MemorySizeInBytes, 0, 0, NodeArray[0]);
    Allocation gpu_entry(handle, aligned_ptr, addr, aligned_size, true, nullptr,
                         MemorySizeInBytes, 0, 0, NodeArray[0]);
    set_mapped_handle(cpu_entry, cpu_entry.owner_node, handle);
    set_mapped_handle(gpu_entry, gpu_entry.owner_node, handle);
    auto ret = map_memory_to_nodes(handle, cpu_entry, NumberOfNodes, NodeArray);
    if (ret != HSAKMT_STATUS_SUCCESS) {
      delete gpu_mem;
      return ret;
    }
    gpu_entry.mapped = cpu_entry.mapped;
    (*allocation_map_)[MemoryAddress] = std::move(cpu_entry);
    (*allocation_map_)[(void *)addr] = std::move(gpu_entry);
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
    if (flags.is_imported_vram_ipc) {
      if (gpu_mem->IsSharedFromSameProcess())
        return HSAKMT_STATUS_SUCCESS;

      auto code = gpu_mem->UnmapGpuVirtualAddress(gpu_mem->GpuAddress(), gpu_mem->Size());
      if (code != ErrorCode::Success)
        return HSAKMT_STATUS_ERROR;
      gpu_mem->Evict();
      if (!gpu_mem->GetDevice()->WaitOnPagingFenceFromCpu())
        return HSAKMT_STATUS_ERROR;
    }

    if (it->second.userptr) {
      if (gpu_mem->DecMappingCount() == 0) {
        for (size_t idx = 0; idx < it->second.mapped.size(); ++idx) {
          const HSAuint32 node = wddm_index_to_gpu_node(static_cast<uint32_t>(idx));
          if (it->second.mapped[idx] && node != it->second.owner_node)
            teardown_mapped_node(it->second, node, gpu_mem);
        }
        allocation_map_->erase((void*)it->second.gpu_addr);
        allocation_map_->erase(it);
        delete gpu_mem;
      }
      return HSAKMT_STATUS_SUCCESS;
    }

    for (size_t idx = 0; idx < it->second.mapped.size(); ++idx) {
      const HSAuint32 node = wddm_index_to_gpu_node(static_cast<uint32_t>(idx));
      if (it->second.mapped[idx] && node != it->second.owner_node)
        teardown_mapped_node(it->second, node, gpu_mem);
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


HSAKMT_STATUS HSAKMTAPI hsaKmtHandleImport(const HsaHandleImportDesc* import_desc,
    					HsaHandleImportResult* import_res, HsaHandleImportFlags* flags)
{
  CHECK_DXG_OPEN();
#ifdef WIN32
  if (import_desc->mem != nullptr) {
    void *memaddr = import_desc->mem;
    auto phys_mem = GetGpuMemoryFromAddress(memaddr);
    if (phys_mem) {
      if (!phys_mem->IsPhysicalCreated()) {
        auto code = phys_mem->CreatePhysicalMemory();
        if (code != ErrorCode::Success) {
          return HSAKMT_STATUS_OUT_OF_RESOURCES;
        }
      }
      import_res->buf_handle = reinterpret_cast<HsaMemoryObjectHandle>(
                               phys_mem->GetGpuMemoryHandle());
      import_res->alloc_size = phys_mem->Size();
      return HSAKMT_STATUS_SUCCESS;
    }
  }
#endif

  if (import_desc->type != HSA_EXTERNAL_HANDLE_DMA_BUF) {
    assert(!"not supported\n");
    return HSAKMT_STATUS_NOT_SUPPORTED;
  }

  if (static_cast<int>(import_desc->dmabuf_fd) == -1) {
    import_res->buf_handle = 0;
    return HSAKMT_STATUS_ERROR;
  }

  wsl::thunk::GpuMemoryHandle mem_handle;
  wsl::thunk::WDDMDevice *pDevice = reinterpret_cast<wsl::thunk::WDDMDevice *>(import_desc->device_handle);
  HSAuint32 node_id = static_cast<HSAuint32>(pDevice->NodeId());
  bool is_ipc_memfd = is_ipc_sysmemfd(import_desc->dmabuf_fd);
  bool alloc_va = is_ipc_memfd;

  // kmt handle importer is false for dma_buf_fd
  HSAKMT_STATUS ret = import_dmabuf_fd(import_desc->dmabuf_fd, 1, &node_id, alloc_va, is_ipc_memfd,
                                       &mem_handle, false);
  if (ret == HSAKMT_STATUS_SUCCESS) {
    //use GpuMemory object handle as drm buf handle
    import_res->buf_handle = reinterpret_cast<HsaMemoryObjectHandle>(mem_handle);
    import_res->alloc_size =
        wsl::thunk::GpuMemory::Convert(mem_handle)->Size();
    return HSAKMT_STATUS_SUCCESS;
  }
  return HSAKMT_STATUS_ERROR;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtHandleExport(const HsaHandleExportDesc* desc,
                                          HsaMemoryExportResult* res,
                                          HsaHandleExportFlags* flags)
{
  (void)flags;
  CHECK_DXG_OPEN();
  if (!desc || !res || desc->device_handle == NULL)
    return HSAKMT_STATUS_INVALID_HANDLE;

  if (desc->type != HSA_EXTERNAL_HANDLE_DMA_BUF)
    return HSAKMT_STATUS_NOT_SUPPORTED;

  wsl::thunk::GpuMemory* gpu_mem = wsl::thunk::GpuMemory::Convert(
      reinterpret_cast<wsl::thunk::GpuMemoryHandle>(desc->buf_handle));
  if (!gpu_mem)
    return HSAKMT_STATUS_INVALID_HANDLE;

  if (!gpu_mem->IsPhysicalCreated()) {
    auto code = gpu_mem->CreatePhysicalMemory();
    if (code != ErrorCode::Success)
      return HSAKMT_STATUS_OUT_OF_RESOURCES;
  }

  int dmabuf_fd = -1;
  auto code = gpu_mem->ExportPhysicalHandle(&dmabuf_fd);
  if (code != ErrorCode::Success)
    return HSAKMT_STATUS_ERROR;

  res->dmabuf_fd = dmabuf_fd;
  return HSAKMT_STATUS_SUCCESS;
}


HSAKMT_STATUS HSAKMTAPI hsaKmtMemoryVaMap(HsaMemoryObjectHandle Handle,
              HSAuint64 offset, HSAuint64 size, HSAuint64 addr,
              HsaMemoryMapFlags flags, HSAuint32 NodeId)
{
	CHECK_DXG_OPEN();
  (void)flags;
  wsl::thunk::GpuMemory* gpu_mem = reinterpret_cast<wsl::thunk::GpuMemory*>(Handle);
  assert(gpu_mem != nullptr);

  wsl::thunk::WDDMDevice* target = get_wddmdev(NodeId);
  if (!target)
    return HSAKMT_STATUS_INVALID_NODE_UNIT;

  /* Page-table update is submitted to NodeId's device. That is the only
   * paging fence this call advances, so the owner device is not waited on
   * unless it is the target. */
  wsl::thunk::GpuMemory* view = gpu_mem;
  if (gpu_mem->GetDevice() != target) {
    std::lock_guard<std::mutex> guard(*allocation_map_lock_);
    Allocation* alloc = FindAllocationByGpuMem(gpu_mem);
    if (!alloc)
      return HSAKMT_STATUS_ERROR;

    auto peer_handle = mapped_handle(*alloc, NodeId);
    if (peer_handle) {
      view = wsl::thunk::GpuMemory::Convert(peer_handle);
    } else {
      auto* primary = wsl::thunk::GpuMemory::Convert(alloc->handle);
      wsl::thunk::GpuMemory* peer = nullptr;
      auto code = target->MapFromPeerDev(primary, &peer);
      if (code != ErrorCode::Success || !peer)
        return HSAKMT_STATUS_ERROR;
      if (peer->GpuAddress() != addr) {
        delete peer;
        return HSAKMT_STATUS_ERROR;
      }
      note_peer_view(alloc->handle, NodeId, peer->GetGpuMemoryHandle());
      /* MapFromPeerDev already mapped this VA and waited on the target. */
      return HSAKMT_STATUS_SUCCESS;
    }
  }

  if (view->GpuAddress() == addr) {
    pr_info("bo is mapped already\n");
    return HSAKMT_STATUS_SUCCESS;
  } else if (view->GpuAddress()) {
    pr_err("amdgpu_bo_va_op: GPU memory already mapped at %p, but requested to map at %p\n",
           reinterpret_cast<void*>(view->GpuAddress()), reinterpret_cast<void*>(addr));
    return HSAKMT_STATUS_ERROR;
  }

  auto code = view->MapGpuVirtualAddress(static_cast<gpusize>(addr), size, offset);
  if (code != ErrorCode::Success)
    return HSAKMT_STATUS_ERROR;

  code = view->MakeResident();
  if (code != ErrorCode::Success)
    return HSAKMT_STATUS_ERROR;

  if (!target->WaitOnPagingFenceFromCpu())
    return HSAKMT_STATUS_ERROR;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMemoryVaUnmap(HsaMemoryObjectHandle Handle,
              HSAuint64 offset, HSAuint64 size, HSAuint64 addr, HSAuint32 NodeId)
{
	CHECK_DXG_OPEN();
  wsl::thunk::GpuMemory* gpu_mem = reinterpret_cast<wsl::thunk::GpuMemory*>(Handle);
  assert(gpu_mem != nullptr);

  wsl::thunk::WDDMDevice* target = get_wddmdev(NodeId);
  if (!target)
    return HSAKMT_STATUS_INVALID_NODE_UNIT;

  wsl::thunk::GpuMemory* view = gpu_mem;
  if (gpu_mem->GetDevice() != target) {
    std::lock_guard<std::mutex> guard(*allocation_map_lock_);
    Allocation* alloc = FindAllocationByGpuMem(gpu_mem);
    auto peer_handle = alloc ? mapped_handle(*alloc, NodeId) : nullptr;
    if (!peer_handle)
      return HSAKMT_STATUS_SUCCESS;
    view = wsl::thunk::GpuMemory::Convert(peer_handle);
  }

  auto code = view->UnmapGpuVirtualAddress(static_cast<gpusize>(addr), size, offset);
  if (code != ErrorCode::Success)
    return HSAKMT_STATUS_ERROR;
  /* Unmap updates the target device's paging fence. Evict() does not. */
  if (!target->WaitOnPagingFenceFromCpu())
    return HSAKMT_STATUS_ERROR;
  view->Evict();
	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMemHandleFree(HsaMemoryObjectHandle Handle)
{
	CHECK_DXG_OPEN();
	// On DXG, handle cleanup is managed through the DXG memory management path.
	// Validate the handle and return success.
	wsl::thunk::GpuMemory* gpu_mem = reinterpret_cast<wsl::thunk::GpuMemory*>(Handle);
	if (!gpu_mem) {
		return HSAKMT_STATUS_INVALID_HANDLE;
	}
	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMemHandleFreePreserveMetadata(HsaMemoryObjectHandle Handle)
{
	CHECK_DXG_OPEN();
	// On DXG, this behaves the same as hsaKmtMemHandleFree since the DXG
	// implementation doesn't manage metadata separately. Used by IPC exporter
	// path (IPCCreate) to release handle references without affecting metadata.
	wsl::thunk::GpuMemory* gpu_mem = reinterpret_cast<wsl::thunk::GpuMemory*>(Handle);
	if (!gpu_mem) {
		return HSAKMT_STATUS_INVALID_HANDLE;
	}
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
              HsaMemoryObjectHandle MemoryHandle, HSAuint64* cpu_addr)
{
	CHECK_DXG_OPEN();
  wsl::thunk::GpuMemory* gpu_mem = reinterpret_cast<wsl::thunk::GpuMemory*>(MemoryHandle);
  assert(gpu_mem != nullptr);
  *cpu_addr = reinterpret_cast<HSAuint64>(gpu_mem->CpuAddress());
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetAmdGPUDeviceFd(HsaAMDGPUDeviceHandle DeviceHandle, HSAint32* fd)
{
  return HSAKMT_STATUS_NOT_SUPPORTED;
}

// Persisting GL2 cache sizing is an amdgpu DRM ioctl; no DXG/WDDM equivalent.
HSAKMT_STATUS HSAKMTAPI hsaKmtSetPersistingCacheSize(HSAuint32 Node, HSAuint64 CacheSize)
{
  return HSAKMT_STATUS_NOT_SUPPORTED;
}
