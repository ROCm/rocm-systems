/* Copyright (c) 2025 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "vmheap.hpp"
#include "command.hpp"

namespace amd {

// ================================================================================================
address VmHeap::ReserveAddressRange(address start, size_t size, size_t alignment) {
  // Reserve a virtual address range on the device
  void* ptr = device_->virtualAlloc(start, size, alignment);
  // Save base memory object to accelerate access in the future
  base_memory_ = MemObjMap::FindVirtualMemObj(ptr);
  return reinterpret_cast<address>(ptr);
}

// ================================================================================================
bool VmHeap::ReleaseAddressRange(void* addr) {
  Memory* memObj = MemObjMap::FindVirtualMemObj(addr);
  assert(memObj != nullptr && "Cannot find the Virtual MemObj entry");

  // Frees address range on the device
  device_->virtualFree(addr);
  memObj->release();

  return true;
}

// ================================================================================================
bool VmHeap::CommitMemory(void* addr, size_t size) {
  const auto& dev_info = device_->info();
  size_t granularity = dev_info.virtualMemAllocGranularity_;
  auto padded_size = alignUp(size, granularity);

  // Allocate physical memory
  void* ptr = SvmBuffer::malloc(device_->context(), ROCCLR_MEM_PHYMEM, padded_size,
                                dev_info.memBaseAddrAlign_, nullptr);
  if (ptr == nullptr) {
    LogPrintfError("Failed to allocate physical memory %zd", padded_size);
    return false;
  }

  size_t offset = 0;  // this is ignored
  // Find physical memory in the map of all objects
  Memory* phys_mem_obj = MemObjMap::FindMemObj(ptr, &offset);

  // Map the physical memory to a virtual address
  Command* cmd = new VirtualMapCommand(GetVmQueue(), Command::EventWaitList{}, addr, padded_size,
                                       phys_mem_obj);
  cmd->enqueue();
  cmd->awaitCompletion();
  cmd->release();

  // Enable memory access
  if (!device_->SetMemAccess(addr, padded_size, Device::VmmAccess::kReadWrite)) {
    LogError("SetAccess failed for the commited memory in VmHeap!");
  }
  return true;
}

// ================================================================================================
bool VmHeap::UncommitMemory(void* addr, size_t size) {
  Memory* vaddr_sub_obj = MemObjMap::FindMemObj(addr);
  Memory* phys_mem_obj = vaddr_sub_obj->getUserData().phys_mem_obj;

  // Unmap the physical memory from a virtual address
  Command* cmd = new VirtualMapCommand(GetVmQueue(), Command::EventWaitList{}, addr, size, nullptr);
  cmd->enqueue();
  cmd->awaitCompletion();
  cmd->release();
  vaddr_sub_obj->release();
  SvmBuffer::free(device_->context(), phys_mem_obj->getSvmPtr());
  return true;
}

// ================================================================================================
VmHeap::VmHeap(Device* device, size_t va_size, size_t chunk_size, GetQueueFunc get_queue)
    : block_alignment_(kMinBlockAlignment),
      chunk_size_(chunk_size),
      lock_(true),
      device_(device),
      get_vm_queue_(get_queue) {
  va_size_ = alignUp(va_size, chunk_size);
  free_size_ = va_size_;
}

// ================================================================================================
VmHeap::~VmHeap() {
  if (created_) {
    ScopedLock k(lock_);

    // Release all heap blocks
    HeapBlock *walk, *next;
    walk = busy_list_;
    while (walk) {
      next = walk->next_;
      FreeBlock(walk);
      walk = next;
    }

    walk = free_list_;
    while (walk) {
      next = walk->next_;
      delete walk;
      walk = next;
    }

    if (mapped_mem_.size() > 0) {
      // Unmap the entire memory range
      UnmapPhysMemory(0, va_size_);
    }
    // Destroy virtual address space
    if (base_address_ != nullptr) {
      ReleaseAddressRange(base_address_);
    }
  }
}

// ================================================================================================
bool VmHeap::Create() {
  // Create a new GPU resource
  base_address_ = ReserveAddressRange(0, va_size_, kChunkSize);
  if (base_address_ == nullptr) {
    return false;
  }
  free_size_ = va_size_;
  // Set up initial free list
  free_list_ = new HeapBlock(this, va_size_, 0);
  if (free_list_ == nullptr) {
    return false;
  }
  mapped_mem_.resize(va_size_ / chunk_size_);
  return true;
}

// ================================================================================================
bool VmHeap::MapPhysMemory(size_t offset, size_t size) {
  auto start_chunk = offset / chunk_size_;
  auto end_chunk = alignUp(offset + size, chunk_size_) / chunk_size_;

  for (auto i = start_chunk; i < end_chunk; ++i) {
    if (!mapped_mem_[i]) {
      auto address = base_address_ + i * chunk_size_;
      if (CommitMemory(address, chunk_size_)) {
        mapped_size_ += chunk_size_;
        if (mapped_size_ > max_mapped_size_) {
          ClPrint(LOG_INFO, LOG_MEM_POOL, "VM heap grows in physical alloc to %d GB\n",
                  static_cast<int>(mapped_size_ / Gi));
        }
        max_mapped_size_ = std::max(max_mapped_size_, mapped_size_);
        mapped_mem_[i] = true;
      } else {
        assert(false);
        return false;
      }
    }
  }
  return true;
}

// ================================================================================================
void VmHeap::UnmapPhysMemory(size_t offset, size_t size) {
  auto busy_size = va_size_ - free_size_;
  uint64_t free_mapped = alignDown(mapped_size_ - busy_size, kChunkSize);

  int start_chunk = alignUp(offset, chunk_size_) / chunk_size_;
  int end_chunk = alignDown(offset + size, chunk_size_) / chunk_size_;

  for (int i = end_chunk - 1; i >= start_chunk; i--) {
    // If free mapped memory lower than the threshold, then stop unmapping
    if (free_mapped <= unmap_threshold_) {
      return;
    }
    if (i >= mapped_mem_.size()) {
      assert(false);
      LogError("VM heap allocation is beyond the range!");
      return;
    }
    if (mapped_mem_[i]) {
      auto address = base_address_ + i * chunk_size_;
      if (UncommitMemory(address, chunk_size_)) {
        mapped_size_ -= chunk_size_;
        free_mapped -= chunk_size_;
        mapped_mem_[i] = false;
      } else {
        assert(false);
      }
    }
  }
}

// ================================================================================================
void VmHeap::TrimPhysMemory(size_t unmap_threshold) {
  ScopedLock k(lock_);
  auto current = free_list_;
  auto unmap_org = unmap_threshold_;
  unmap_threshold_ = unmap_threshold;
  while (current != nullptr) {
    UnmapPhysMemory(current->offset_, current->size_);
    current = current->next_;
  }
  unmap_threshold_ = unmap_org;
}

// ================================================================================================
address VmHeap::Alloc(size_t size) {
  ScopedLock k(lock_);

  if (!created_) {
    // Create VM heap if it's not created
    created_ = Create();
    if (!created_) {
      return nullptr;
    }
  }
  address ptr = nullptr;
  size_t offset = 0;
  auto hb = AllocBlock(size + block_alignment_);
  if (hb != nullptr) {
    offset = ((hb->Offset() & ~kChunkSize) == 0) ? hb->Offset() + block_alignment_ : hb->Offset();
    ptr = base_address_ + offset;
  } else {
    return nullptr;
  }
  auto memory =
      new (device_->context()) Buffer(*base_memory_, 0, offset, size, &device_->context());
  if (nullptr == memory || !memory->create(nullptr)) {
    FreeBlock(hb);
    return nullptr;
  }
  MemObjMap::AddMemObj(ptr, memory);
  if (memory->getUserData().data == nullptr) {
    memory->getUserData().data = hb;
  }
  ClPrint(LOG_INFO, LOG_MEM_POOL, "VmHeap Alloc: %p offset(%zx + %zx) hb(%p)", ptr, hb->Offset(),
          memory->getSize(), hb);
  return ptr;
}

// ================================================================================================
void VmHeap::Free(Memory* memory) {
  const device::Memory* dev_mem = memory->getDeviceMemory(*device_);
  void* addr = reinterpret_cast<void*>(dev_mem->virtualAddress());
  if (addr == nullptr) {
    addr = memory->getSvmPtr();
  }

  if (!created_ || (addr < base_address_)) {
    return;
  }
  ScopedLock k(lock_);
  if (memory->getUserData().data != nullptr) {
    auto hb = reinterpret_cast<HeapBlock*>(memory->getUserData().data);
    ClPrint(LOG_INFO, LOG_MEM_POOL, "VmHeap Free: %p offset(%zx + %zx) hb(%p)", addr, hb->Offset(),
            memory->getSize(), hb);
    FreeBlock(hb);
  }
  MemObjMap::RemoveMemObj(addr);
  memory->release();
}

// ================================================================================================
HeapBlock* VmHeap::AllocBlock(size_t un_size) {
  assert(un_size != 0);
  ScopedLock k(lock_);
  HeapBlock* walk = free_list_;
  HeapBlock* best = nullptr;

  // Round size
  auto size = alignUp(un_size, block_alignment_);

  // Walk the free list looking for a suitable block (currently best-fit)
  while (walk) {
    if ((walk->size_ > size) && (best == nullptr || walk->size_ < best->size_)) {
      best = walk;
    } else if (walk->size_ == size) {
      // No need to split, just move to busy list
      DetachBlock(&free_list_, walk);
      walk->busy_ = true;
      InsertBlock(&busy_list_, walk);
      free_size_ -= size;
      if (!MapPhysMemory(walk->Offset(), size)) {
        free(walk);
        return nullptr;
      }
      return walk;
    }
    walk = walk->next_;
  }

  if (best != nullptr) {
    // Got one, but need to split it. Keep first part in free list,
    // put second part into busy list
    HeapBlock* newblock = SplitBlock(best, size);
    newblock->busy_ = true;
    InsertBlock(&busy_list_, newblock);
    free_size_ -= size;
    if (!MapPhysMemory(newblock->Offset(), size)) {
      free(newblock);
      return nullptr;
    }
    return newblock;
  }

  return nullptr;
}

// ================================================================================================
void VmHeap::FreeBlock(HeapBlock* blk) {
  DetachBlock(&busy_list_, blk);
  blk->busy_ = false;
  free_size_ += blk->size_;
  UnmapPhysMemory(blk->offset_, blk->size_);
  MergeBlock(&free_list_, blk);
}

// ================================================================================================
void VmHeap::DetachBlock(HeapBlock** list, HeapBlock* blk) {
  if (*list == blk) {
    *list = blk->next_;
  }
  if (blk->prev_) {
    blk->prev_->next_ = blk->next_;
  }
  if (blk->next_) {
    blk->next_->prev_ = blk->prev_;
  }
}

// ================================================================================================
void VmHeap::InsertBlock(HeapBlock** head, HeapBlock* blk) {
  if (nullptr == *head) {
    *head = blk;
    blk->prev_ = nullptr;
    blk->next_ = nullptr;
    return;
  }

  // Find the place to insert it at
  HeapBlock* walk = *head;
  while (walk->next_ && walk->next_->offset_ < blk->offset_) {
    walk = walk->next_;
  }

  // Insert it
  if (walk == *head) {
    if (walk->offset_ >= blk->offset_) {
      *head = blk;
      blk->prev_ = nullptr;
      blk->next_ = walk;
      walk->prev_ = *head;
      return;
    }
  }

  blk->next_ = walk->next_;
  blk->prev_ = walk;
  if (walk->next_) {
    walk->next_->prev_ = blk;
  }
  walk->next_ = blk;
}

// ================================================================================================
HeapBlock* VmHeap::SplitBlock(HeapBlock* blk, size_t tailsize) {
  // Create a new block from the beginning of the current
  HeapBlock* nb = new HeapBlock(blk->owner_, tailsize, blk->offset_);

  // Resize the old block
  blk->offset_ += tailsize;
  blk->size_ -= tailsize;
  return nb;
}

// ================================================================================================
void VmHeap::Join2Blocks(HeapBlock* first, HeapBlock* second) const {
  // Do the join
  first->size_ = first->size_ + second->size_;
  first->next_ = second->next_;
  if (second->next_) {
    second->next_->prev_ = first;
  }
  delete second;
}

// ================================================================================================
void VmHeap::MergeBlock(HeapBlock** head, HeapBlock* blk) {
  InsertBlock(head, blk);

  // Merge with successor if possible
  if ((blk->next_ != nullptr) && (blk->offset_ + blk->size_ == blk->next_->offset_)) {
    Join2Blocks(blk, blk->next_);
  }

  // Merge with predecessor if possible
  if ((blk->prev_ != nullptr) && (blk->prev_->offset_ + blk->prev_->size_ == blk->offset_)) {
    Join2Blocks(blk->prev_, blk);
  }
}

// ================================================================================================
address VmHeapArray::Alloc(size_t size) {
  address addr = nullptr;
  for (uint32_t i = 0; i < kMaxArraySize; ++i) {
    if (vm_heaps_[i]->free_size_ > (size + VmHeap::kChunkSize)) {
      addr = vm_heaps_[i]->Alloc(size);
      if (addr != nullptr) {
        break;
      }
    }
  }
  return addr;
}

// ================================================================================================
void VmHeapArray::Free(amd::Memory* memory) {
  const device::Memory* dev_mem = memory->getDeviceMemory(*device_);
  void* addr = reinterpret_cast<void*>(dev_mem->virtualAddress());
  if (addr == nullptr) {
    addr = memory->getSvmPtr();
  }
  for (uint32_t i = 0; i < kMaxArraySize; ++i) {
    if (vm_heaps_[i]->created_ && vm_heaps_[i]->InRange(addr)) {
      vm_heaps_[i]->Free(memory);
      break;
    }
  }
  uint64_t freed = 0;
  for (uint32_t i = 0; i < kMaxArraySize; ++i) {
    freed += vm_heaps_[i]->FreeMappedSize();
  }
  if (freed > unmap_threshold_) {
    uint64_t extra = freed - unmap_threshold_;
    uint64_t trim = (extra < unmap_threshold_) ? (unmap_threshold_ - extra) : 0;
    TrimPhysMemory(trim);
  }
}

// ================================================================================================
void VmHeapArray::TrimPhysMemory(size_t unmap_threshold) {
  for (uint i = 0; i < kMaxArraySize; ++i) {
    // Check the threshold against the accumulated sizes in all heaps
    if (vm_heaps_[i]->created_ && [this]() {
          uint64_t size = 0;
          for (uint i = 0; i < kMaxArraySize; ++i) {
            size += vm_heaps_[i]->FreeMappedSize();
          }
          return size;
        }() > unmap_threshold) {
      vm_heaps_[i]->TrimPhysMemory(unmap_threshold);
    } else {
      break;
    }
  }
}

// ================================================================================================
void VmHeapArray::SetUnmapThreshold(uint64_t threshold) {
  for (uint i = 0; i < kMaxArraySize; ++i) {
    // Note: it's not precisely correct to use the same threshold in all heaps,
    // but the logic will trim heaps in Free()
    if (vm_heaps_[i]->created_) {
      vm_heaps_[i]->SetUnmapThreshold(threshold);
    }
  }
  unmap_threshold_ = threshold;
}

// ================================================================================================
uint64_t VmHeapArray::MappedSize() const {
  uint64_t size = 0;
  for (uint i = 0; i < kMaxArraySize; ++i) {
    size += vm_heaps_[i]->MappedSize();
  }
  return size;
}

// ================================================================================================
uint64_t VmHeapArray::FreeMappedSize() const {
  uint64_t size = 0;
  for (uint i = 0; i < kMaxArraySize; ++i) {
    size += vm_heaps_[i]->FreeMappedSize();
  }
  return size;
}

// ================================================================================================
uint64_t VmHeapArray::MaxMappedSize() const {
  uint64_t size = 0;
  for (uint i = 0; i < kMaxArraySize; ++i) {
    size += vm_heaps_[i]->max_mapped_size_;
  }
  return size;
}

// ================================================================================================
void VmHeapArray::ResetMaxMappedSize() {
  for (uint i = 0; i < kMaxArraySize; ++i) {
    vm_heaps_[i]->max_mapped_size_ = 0;
  }
}

}  // namespace amd
