/* Copyright (c) 2022-2025 Advanced Micro Devices, Inc.

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

#include "hip_mempool_impl.hpp"
#include "hip_vm.hpp"
#include "platform/command.hpp"

namespace hip {

// ================================================================================================
void Heap::AddMemory(amd::Memory* memory, Stream* stream) {
  auto mem_size = memory->getSize();
  allocations_.insert({{mem_size, memory}, {stream}});
  total_size_ += mem_size;
  max_total_size_ = std::max(max_total_size_, total_size_);
}

// ================================================================================================
void Heap::AddMemory(amd::Memory* memory, const MemoryTimestamp& ts) {
  auto mem_size = memory->getSize();
  allocations_.insert({{mem_size, memory}, ts});
  total_size_ += mem_size;
  max_total_size_ = std::max(max_total_size_, total_size_);
}

// ================================================================================================
amd::Memory* Heap::FindMemory(size_t size, Stream* stream, bool opportunistic, void* dptr,
                              MemoryTimestamp* ts) {
  amd::Memory* memory = nullptr;
  auto start = allocations_.lower_bound({size, nullptr});
  for (auto it = start; it != allocations_.end();) {
    bool check_address = (dptr == nullptr);
    if (it->first.second->getSvmPtr() == dptr) {
      // If the search is done for the specified address then runtime must wait
      it->second.Wait();
      check_address = true;
    }
    // Runtime can accept an allocation with 12.5% on the size threshold
    if (it->first.first > (size / 8.0) * 9) {
      return nullptr;
    }
    // Check if size can match and it's safe to use this resource.
    if (check_address && (it->second.IsSafeFind(stream, opportunistic))) {
      memory = it->first.second;
      total_size_ -= memory->getSize();
      // Preserve event, since the logic could skip GPU wait on reuse
      ts->event_ = it->second.event_;
      // Remove found allocation from the map
      it = allocations_.erase(it);
      break;
    } else {
      ++it;
    }
  }
  return memory;
}

// ================================================================================================
bool Heap::RemoveMemory(amd::Memory* memory, MemoryTimestamp* ts) {
  auto mem_size = memory->getSize();
  if (auto it = allocations_.find({mem_size, memory}); it != allocations_.end()) {
    if (ts != nullptr) {
      // Preserve timestamp info for possible reuse later
      *ts = it->second;
    } else {
      it->second.SetEvent(nullptr);
    }
    total_size_ -= mem_size;
    allocations_.erase(it);
    return true;
  }
  return false;
}

// ================================================================================================
Heap::SortedMap::iterator Heap::EraseAllocation(Heap::SortedMap::iterator& it) {
  auto memory = it->first.second;
  const device::Memory* dev_mem = memory->getDeviceMemory(*device_->devices()[0]);
  void* dev_mem_vaddr = reinterpret_cast<void*>(dev_mem->virtualAddress());
  total_size_ -= it->first.first;
  if (dev_mem_vaddr == nullptr) {
    dev_mem_vaddr = memory->getSvmPtr();
  }

  if (use_vm_heap_) {
    vm_heap_.Free(memory);
  } else {
    amd::SvmBuffer::free(memory->getContext(), dev_mem_vaddr);
  }
  // Clear HIP event
  it->second.SetEvent(nullptr);
  // Remove the allocation from the map
  return allocations_.erase(it);
}

// ================================================================================================
bool Heap::ReleaseAllMemory(size_t min_bytes_to_hold, bool safe_release) {
  for (auto it = allocations_.begin(); it != allocations_.end();) {
    // Make sure the heap is smaller than the minimum value to hold
    if (total_size_ <= min_bytes_to_hold) {
      return true;
    }
    // Safe release forces unconditional wait for memory
    if (safe_release) {
      it->second.Wait();
    }
    if (it->second.IsSafeRelease()) {
      it = EraseAllocation(it);
    } else {
      ++it;
    }
  }
  // Handle managed pool with trim
  if (vm_heap_.FreeMappedSize() > min_bytes_to_hold) {
    vm_heap_.TrimPhysMemory(min_bytes_to_hold);
  }
  return true;
}

// ================================================================================================
bool Heap::ReleaseAllMemory() {
  for (auto it = allocations_.begin(); it != allocations_.end();) {
    // Make sure the heap holds the minimum number of bytes
    // @note: Managed memory controls the threshold on its own
    if (!use_vm_heap_ && (total_size_ <= release_threshold_)) {
      return true;
    }
    if (it->second.IsSafeRelease()) {
      it = EraseAllocation(it);
    } else {
      ++it;
    }
  }
  return true;
}

// ================================================================================================
void Heap::RemoveStream(Stream* stream) {
  for (auto it : allocations_) {
    it.second.safe_streams_.erase(stream);
  }
}

// ================================================================================================
void Heap::SetAccess(hip::Device* device, bool enable) {
  for (const auto& it : allocations_) {
    auto peer_device = device->asContext()->devices()[0];
    device::Memory* mem = it.first.second->getDeviceMemory(*peer_device);
    if (mem != nullptr) {
      if (!mem->getAllowedPeerAccess() && enable) {
        // Enable p2p access for the specified device
        peer_device->allowPeerAccess(mem);
        mem->setAllowedPeerAccess(true);
      } else if (mem->getAllowedPeerAccess() && !enable) {
        mem->setAllowedPeerAccess(false);
      }
    } else {
      LogError("Couldn't find device memory for P2P access");
    }
  }
}

// ================================================================================================
void* MemoryPool::AllocateMemory(size_t size, Stream* stream, void* dptr) {
  amd::ScopedLock lock(lock_pool_ops_);

  void* dev_ptr = nullptr;
  MemoryTimestamp ts;
  amd::Memory* memory = free_heap_.FindMemory(size, stream, Opportunistic(), dptr, &ts);
  if (memory == nullptr) {
    if (Properties().maxSize != 0 && (max_total_size_ + size) > Properties().maxSize) {
      return nullptr;
    }
    amd::Context* context = device_->asContext();
    const auto& dev_info = context->devices()[0]->info();
    if (dev_info.maxMemAllocSize_ < size) {
      return nullptr;
    }
    cl_svm_mem_flags flags = (state_.interprocess_) ? ROCCLR_MEM_INTERPROCESS : 0;
    flags |= (state_.phys_mem_) ? ROCCLR_MEM_PHYMEM : 0;
    if (state_.use_vm_heap_) {
      dev_ptr = Alloc(size);
    } else {
      dev_ptr = amd::SvmBuffer::malloc(*context, flags, size, dev_info.memBaseAddrAlign_, nullptr);
    }
    if (dev_ptr == nullptr) {
      size_t free = 0, total = 0;
      hipError_t err = hipMemGetInfo(&free, &total);
      if (err == hipSuccess) {
        LogPrintfError(
            "Allocation failed : Device memory : required :\
          %zu | free :%zu | total :%zu",
            size, free, total);
      }
      return nullptr;
    }

    size_t offset = 0;
    memory = getMemoryObject(dev_ptr, offset);
    // Saves the current device id so that it can be accessed later
    memory->getUserData().deviceId = device_->deviceId();

    // Update access for the new allocation from other devices
    for (const auto& it : access_map_) {
      auto vdi_device = it.first->asContext()->devices()[0];
      device::Memory* mem = memory->getDeviceMemory(*vdi_device);
      if ((mem != nullptr) && (it.second != hipMemAccessFlagsProtNone)) {
        vdi_device->allowPeerAccess(mem);
        mem->setAllowedPeerAccess(true);
      }
    }
  } else {
    dev_ptr = memory->getSvmPtr();
  }
  // Place the allocated memory into the busy heap
  ts.AddSafeStream(stream);
  busy_heap_.AddMemory(memory, ts);

  max_total_size_ =
      std::max(max_total_size_, busy_heap_.GetTotalSize() + free_heap_.GetTotalSize());
  // Increment the reference counter on the pool
  retain();

  ClPrint(amd::LOG_INFO, amd::LOG_MEM_POOL, "Pool AllocMem: %p, %p", memory->getSvmPtr(), memory);

  return dev_ptr;
}

// ================================================================================================
bool MemoryPool::FreeMemory(amd::Memory* memory, Stream* stream, Event* event) {
  {
    amd::ScopedLock lock(lock_pool_ops_);

    if (!state_.use_vm_heap_ && memory->getUserData().phys_mem_obj != nullptr) {
      memory = memory->getUserData().phys_mem_obj;
    }

    // If the free heap grows over the busy heap, then force release
    if (AMD_DIRECT_DISPATCH && (free_heap_.GetTotalSize() > busy_heap_.GetTotalSize())) {
      // Use event base release to reduce memory pressure
      constexpr size_t kBytesToHold = 0;
      free_heap_.ReleaseAllMemory(kBytesToHold);

      // If free mmeory is less than 12.5% of total, then force wait release
      size_t free = 0;
      size_t total = 0;
      hipError_t err = hipMemGetInfo(&free, &total);
      if ((err == hipSuccess) && (free < (total >> 3))) {
        constexpr bool kSafeRelease = true;
        free_heap_.ReleaseAllMemory(free_heap_.GetTotalSize() >> 1, kSafeRelease);
      }
    }

    MemoryTimestamp ts;
    // Remove memory object from the busy pool
    if (!busy_heap_.RemoveMemory(memory, &ts)) {
      // This pool doesn't contain memory
      return false;
    }
    ClPrint(amd::LOG_INFO, amd::LOG_MEM_POOL, "Pool FreeMem: %p, %p", memory->getSvmPtr(), memory);

    if (memory->getUserData().vaddr_mem_obj != nullptr) {
      auto va_mem = memory->getUserData().vaddr_mem_obj;
      if (stream == nullptr) {
        stream = g_devices[memory->getUserData().deviceId]->NullStream();
      }
      // Unmap virtual address from memory
      auto cmd = new amd::VirtualMapCommand(*stream, amd::Command::EventWaitList{},
                                            va_mem->getSvmPtr(), va_mem->getSize(), nullptr);
      cmd->enqueue();
      cmd->release();
    }

    if (stream != nullptr) {
      // The stream of destruction is a safe stream, because the app must handle sync
      ts.AddSafeStream(stream);

      if (event == nullptr) {
        // Add a marker to the stream to trace availability of this memory
        Event* e = new hip::Event(0);
        if (e != nullptr) {
          if (hipSuccess == e->addMarker(stream, nullptr)) {
            ts.SetEvent(e);
            // Make sure runtime sends a notification
            auto result = e->ready();
          }
        }
      } else {
        ts.SetEvent(event);
      }
    } else {
      // Assume a safe release from hipFree() if stream is nullptr
      ts.SetEvent(nullptr);
    }
    free_heap_.AddMemory(memory, ts);
  }

  // Decrement the reference counter on the pool.
  // Note: It may delete memory pool for the last allocation. Thus, the scope lock can't include
  // this call.
  release();

  return true;
}

// ================================================================================================
void MemoryPool::ReleaseAllMemory() {
  constexpr bool kSafeRelease = true;
  free_heap_.ReleaseAllMemory(0, kSafeRelease);
  busy_heap_.ReleaseAllMemory(0, kSafeRelease);
}

// ================================================================================================
void MemoryPool::ReleaseFreedMemory() {
  amd::ScopedLock lock(lock_pool_ops_);

  free_heap_.ReleaseAllMemory();
}

// ================================================================================================
void MemoryPool::RemoveStream(Stream* stream) {
  amd::ScopedLock lock(lock_pool_ops_);

  free_heap_.RemoveStream(stream);
}

// ================================================================================================
void MemoryPool::TrimTo(size_t min_bytes_to_hold) {
  amd::ScopedLock lock(lock_pool_ops_);

  free_heap_.ReleaseAllMemory(min_bytes_to_hold);
}

// ================================================================================================
hipError_t MemoryPool::SetAttribute(hipMemPoolAttr attr, void* value) {
  amd::ScopedLock lock(lock_pool_ops_);
  uint64_t reset;

  switch (attr) {
    case hipMemPoolReuseFollowEventDependencies:
      // Enable/disable HIP events tracking from the app's dependencies
      state_.event_dependencies_ = *reinterpret_cast<int32_t*>(value);
      break;
    case hipMemPoolReuseAllowOpportunistic:
      // Enable/disable HIP event check for freed memory
      state_.opportunistic_ = *reinterpret_cast<int32_t*>(value);
      break;
    case hipMemPoolReuseAllowInternalDependencies:
      // Enable/disable internal extra dependencies introduced in runtime
      state_.internal_dependencies_ = *reinterpret_cast<int32_t*>(value);
      break;
    case hipMemPoolAttrReleaseThreshold:
      free_heap_.SetReleaseThreshold(*reinterpret_cast<uint64_t*>(value));
      break;
    case hipMemPoolAttrReservedMemCurrent:
      // Should be GetAttribute only
      return hipErrorInvalidValue;
      break;
    case hipMemPoolAttrReservedMemHigh:
      reset = *reinterpret_cast<uint64_t*>(value);
      // Only 0 is accepted
      if (reset != 0) {
        return hipErrorInvalidValue;
      }
      max_total_size_ = reset;
      ResetMaxMappedSize();
      break;
    case hipMemPoolAttrUsedMemCurrent:
      // Should be GetAttribute only
      return hipErrorInvalidValue;
      break;
    case hipMemPoolAttrUsedMemHigh:
      reset = *reinterpret_cast<uint64_t*>(value);
      // Only 0 is accepted
      if (reset != 0) {
        return hipErrorInvalidValue;
      }
      busy_heap_.SetMaxTotalSize(reset);
      break;
    default:
      return hipErrorInvalidValue;
  }
  return hipSuccess;
}

// ================================================================================================
hipError_t MemoryPool::GetAttribute(hipMemPoolAttr attr, void* value) {
  amd::ScopedLock lock(lock_pool_ops_);

  switch (attr) {
    case hipMemPoolReuseFollowEventDependencies:
      // Enable/disable HIP events tracking from the app's dependencies
      *reinterpret_cast<int32_t*>(value) = EventDependencies();
      break;
    case hipMemPoolReuseAllowOpportunistic:
      // Enable/disable HIP event check for freed memory
      *reinterpret_cast<int32_t*>(value) = Opportunistic();
      break;
    case hipMemPoolReuseAllowInternalDependencies:
      // Enable/disable internal extra dependencies introduced in runtime
      *reinterpret_cast<int32_t*>(value) = InternalDependencies();
      break;
    case hipMemPoolAttrReleaseThreshold:
      *reinterpret_cast<uint64_t*>(value) = free_heap_.GetReleaseThreshold();
      break;
    case hipMemPoolAttrReservedMemCurrent:
      // All allocated memory by the pool in OS
      *reinterpret_cast<uint64_t*>(value) =
          (state_.use_vm_heap_) ? MappedSize()
                                : (busy_heap_.GetTotalSize() + free_heap_.GetTotalSize());
      break;
    case hipMemPoolAttrReservedMemHigh:
      // High watermark of all allocated memory in OS, since the last reset
      *reinterpret_cast<uint64_t*>(value) =
          (state_.use_vm_heap_) ? MaxMappedSize() : max_total_size_;
      break;
    case hipMemPoolAttrUsedMemCurrent:
      // Total currently used memory by the pool
      *reinterpret_cast<uint64_t*>(value) = busy_heap_.GetTotalSize();
      break;
    case hipMemPoolAttrUsedMemHigh:
      // High watermark of all used memoryS, since the last reset
      *reinterpret_cast<uint64_t*>(value) = busy_heap_.GetMaxTotalSize();
      break;
    default:
      return hipErrorInvalidValue;
  }
  return hipSuccess;
}

// ================================================================================================
void MemoryPool::SetAccess(hip::Device* device, hipMemAccessFlags flags) {
  amd::ScopedLock lock(lock_pool_ops_);

  // Check if the requested device is the pool device where memory was allocated
  if (device == device_) {
    return;
  }

  hipMemAccessFlags current_flags = hipMemAccessFlagsProtNone;

  // Check if access was enabled before
  if (access_map_.find(device) != access_map_.end()) {
    current_flags = access_map_[device];
  }

  if (current_flags != flags) {
    bool enable_access = false;
    // Save the access state  in the device map
    access_map_[device] = flags;
    // Check if access is enabled
    if ((flags == hipMemAccessFlagsProtRead) || (flags == hipMemAccessFlagsProtReadWrite)) {
      enable_access = true;
    }
    // Update device access on the both pools
    busy_heap_.SetAccess(device, enable_access);
    free_heap_.SetAccess(device, enable_access);
  }
}

// ================================================================================================
void MemoryPool::GetAccess(hip::Device* device, hipMemAccessFlags* flags) {
  amd::ScopedLock lock(lock_pool_ops_);

  // Current pool device has full access to memory allocation
  *flags = (device == device_) ? hipMemAccessFlagsProtReadWrite : hipMemAccessFlagsProtNone;

  // Check if access was enabled before
  if (access_map_.find(device) != access_map_.end()) {
    *flags = access_map_[device];
  }
}

// ================================================================================================
void MemoryPool::FreeAllMemory(Stream* stream) {
  while (!busy_heap_.Allocations().empty()) {
    FreeMemory(busy_heap_.Allocations().begin()->first.second, stream);
  }
}

// ================================================================================================
amd::Os::FileDesc MemoryPool::Export() {
  amd::ScopedLock lock(lock_pool_ops_);
  if (shared_ != nullptr) {
    return shared_->handle_;
  }

  constexpr uint32_t kFileNameSize = 20;
  char file_name[kFileNameSize];
  // Generate a unique name from the mempool pointer
  // Note: Windows can accept an unnamed allocation
  snprintf(file_name, kFileNameSize, "%p", this);
  amd::Os::FileDesc handle{};
  shared_ = reinterpret_cast<SharedMemPool*>(
      amd::Os::CreateIpcMemory(file_name, sizeof(SharedMemPool), &handle));
  if (shared_ != nullptr) {
    shared_->handle_ = handle;
    shared_->state_ = state_.value_;
    shared_->access_size_ = 0;
    memset(shared_->access_, 0, sizeof(SharedAccess) * kMaxMgpuAccess);
    assert((access_map_.size() <= kMaxMgpuAccess) && "Can't support more GPU(s) in shared access");
    for (auto it : access_map_) {
      shared_->access_[shared_->access_size_] = SharedAccess{it.first->deviceId(), it.second};
      shared_->access_size_++;
    }
  }
  return handle;
}

// ================================================================================================
bool MemoryPool::Import(amd::Os::FileDesc handle) {
  amd::ScopedLock lock(lock_pool_ops_);
  bool result = false;
  auto shared = reinterpret_cast<SharedMemPool*>(
      amd::Os::OpenIpcMemory(nullptr, handle, sizeof(SharedMemPool)));

  if (shared != nullptr) {
    state_.value_ = shared->state_;
    for (uint32_t i = 0; i < shared->access_size_; ++i) {
      access_map_[g_devices[shared->access_[i].device_id_]] = shared->access_[i].flags_;
    }
    result = true;
  }
  return result;
}
}  // namespace hip
