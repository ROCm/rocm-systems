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

#pragma once

#include <hip/hip_runtime.h>
#include "hip_event.hpp"
#include "hip_internal.hpp"
#include "platform/vmheap.hpp"
#include <unordered_map>
#include <unordered_set>

namespace hip {

class Device;
class Stream;

struct SharedMemPointer {
  size_t offset_;
  size_t size_;
  char handle_[AMD_IPC_MEM_HANDLE_SIZE];
};

struct MemoryTimestamp {
  MemoryTimestamp(hip::Stream* stream = nullptr) {
    if (stream != nullptr) {
      safe_streams_.insert(stream);
    }
  }
  /// Adds a safe stream to the list of stream for possible reuse
  void AddSafeStream(Stream* event_stream, Stream* wait_stream = nullptr) {
    if (wait_stream == nullptr) {
      if (safe_streams_.find(event_stream) == safe_streams_.end()) {
        safe_streams_.insert(event_stream);
      }
    } else {
      if (safe_streams_.find(event_stream) != safe_streams_.end()) {
        safe_streams_.insert(wait_stream);
      }
    }
  }
  /// Changes last known valid event asociated with memory
  void SetEvent(hip::Event* event) {
    // Runtime will delete the HIP event, hence make sure GPU is done with it
    Wait();
    delete event_;
    event_ = event;
  }
  /// Wait for memory to be available
  void Wait() {
    if (event_ != nullptr) {
      auto hip_error = event_->synchronize();
    }
  }
  /// Returns if memory object is safe for reuse
  bool IsSafeFind(hip::Stream* stream = nullptr, bool opportunistic = true) {
    bool result = false;
    if (safe_streams_.find(stream) != safe_streams_.end()) {
      // A safe stream doesn't require TS validation
      result = true;
    } else if (opportunistic && (event_ != nullptr)) {
      // Check HIP event for a retired status
      result = (event_->query() == hipSuccess) ? true : false;
    } else if (event_ == nullptr) {
      // Event doesn't exist. It was a safe release with explicit wait
      return true;
    }
    return result;
  }
  /// Returns if memory object is safe for reuse
  bool IsSafeRelease() {
    bool result = true;
    if (event_ != nullptr) {
      // Check HIP event for a retired status
      result = (event_->query() == hipSuccess) ? true : false;
    }
    return result;
  }

  std::unordered_set<hip::Stream*> safe_streams_;  //!< Safe streams for memory reuse
  hip::Event* event_ = nullptr;  //!< Last known HIP event, associated with the memory object
};

class Heap : public amd::EmbeddedObject {
 public:
  typedef std::map<std::pair<size_t, amd::Memory*>, MemoryTimestamp> SortedMap;

  Heap(hip::Device* device, amd::VmHeapArray& vm_heap)
      : total_size_(0),
        max_total_size_(0),
        release_threshold_(0),
        device_(device),
        vm_heap_(vm_heap) {}
  ~Heap() {}

  /// Adds allocation into the heap on a specific stream
  void AddMemory(amd::Memory* memory, Stream* stream);

  /// Adds allocation into the heap with specific TS
  void AddMemory(amd::Memory* memory, const MemoryTimestamp& ts);

  /// Finds memory object with the specified size
  amd::Memory* FindMemory(size_t size, Stream* stream, bool opportunistic, void* dptr,
                          MemoryTimestamp* ts);

  /// Removes allocation from the map
  bool RemoveMemory(amd::Memory* memory, MemoryTimestamp* ts = nullptr);

  /// Releases all memory, until the threshold value is met
  bool ReleaseAllMemory(size_t min_bytes_to_hold, bool safe_release = false);

  /// Releases all memory, safe to the provided stream, until the threshold value is met
  bool ReleaseAllMemory();

  /// Remove the provided stream from the safe list
  void RemoveStream(Stream* stream);

  /// Enables P2P access to the provided device
  void SetAccess(hip::Device* device, bool enable);

  /// Heap doesn't have any allocations
  bool IsEmpty() const { return (allocations_.size() == 0) ? true : false; }

  /// Set the memory release threshold
  void SetReleaseThreshold(uint64_t value) {
    release_threshold_ = value;
    vm_heap_.SetUnmapThreshold(value);
  }

  /// Set the memory release threshold
  uint64_t GetReleaseThreshold() const { return release_threshold_; }

  /// Get the size of all allocations in the heap
  uint64_t GetTotalSize() const { return total_size_; }

  /// Get the size of all allocations in the heap
  uint64_t GetMaxTotalSize() const { return max_total_size_; }

  /// Set maximum total, allocated by the heap
  void SetMaxTotalSize(uint64_t value) { max_total_size_ = value; }

  /// Erases single allocation form the heap's map
  SortedMap::iterator EraseAllocation(SortedMap::iterator& it);

  /// Add a safe stream for  quick looks-ups in all allocations
  void AddSafeStream(Stream* event_stream, Stream* wait_stream) {
    for (auto& it : allocations_) {
      it.second.AddSafeStream(event_stream, wait_stream);
    }
  }

  /// Checks if memory belongs to this heap
  bool IsActiveMemory(amd::Memory* memory) const {
    return (allocations_.find({memory->getSize(), memory}) != allocations_.end());
  }

  /// Enabled VM heap for memory, instead of direct allocations
  void EnableVmHeap() { use_vm_heap_ = true; }

  /// Returns true if heap uses virtual memory
  bool UseVmHeap() const { return use_vm_heap_; }

  const auto& Allocations() { return allocations_; }

 private:
  Heap() = delete;
  Heap(const Heap&) = delete;
  Heap& operator=(const Heap&) = delete;

  SortedMap allocations_;       //!< Map of allocations on a specific stream
  uint64_t total_size_;         //!< Size of all allocations in the heap
  uint64_t max_total_size_;     //!< Maximum heap allocation size
  uint64_t release_threshold_;  //!< Threshold size in bytes for memory release from heap, default 0

  hip::Device* device_;        //!< Hip device the allocations will reside
  amd::VmHeapArray& vm_heap_;  //!< Managed heap for memory allocaitons
  bool use_vm_heap_ = false;   //!< Use virtual heap or direct allocations
};

/// Allocates memory in the pool on the specified stream and places the allocation into busy_heap_
/// @note: the logic also will look in free_heap for possible reuse.
/// hipMemPoolReuseAllowOpportunistic option will validate if HIP event,
/// associated with memory is done, then reuse can be performed.
class MemoryPool : public amd::ReferenceCountedObject, amd::VmHeapArray {
 public:
  struct SharedAccess {
    int device_id_;            //!< Device ID for access with a specified shared resource
    hipMemAccessFlags flags_;  //!< Flags which define access type
  };

  static constexpr uint32_t kMaxMgpuAccess = 32;
  struct SharedMemPool {
    amd::Os::FileDesc handle_;             //!< File descriptor for shared memory
    uint32_t state_;                       //!< Memory pool state
    uint32_t access_size_;                 //!< The number of entries in access array
    SharedAccess access_[kMaxMgpuAccess];  //!< The list of devices for access
  };

  MemoryPool(hip::Device* device, const hipMemPoolProps* props = nullptr, bool phys_mem = false)
      : VmHeapArray(device->devices()[0],
                    [this]() -> amd::HostQueue& { return *device_->NullStream(); }),
        busy_heap_(device, *this),
        free_heap_(device, *this),
        lock_pool_ops_(true),
        device_(device),
        shared_(nullptr),
        max_total_size_(0) {
    device_->AddMemoryPool(this);
    state_.value_ = 0;
    state_.event_dependencies_ = 1;
    state_.opportunistic_ = 1;
    state_.internal_dependencies_ = 1;
    state_.phys_mem_ = HIP_MEM_POOL_USE_VM && phys_mem;
    if (props != nullptr) {
      properties_ = *props;
    } else {
      properties_ = {.allocType = hipMemAllocationTypePinned,
                     .handleTypes = hipMemHandleTypeNone,
                     .location = {.type = hipMemLocationTypeDevice, .id = device_->deviceId()},
                     .win32SecurityAttributes = nullptr,
                     .maxSize = 0,
                     .reserved = {}};
    }
    state_.interprocess_ = properties_.handleTypes != hipMemHandleTypeNone;
    // Check if VM heap can be enabled
    if (DEBUG_HIP_MEM_POOL_VMHEAP && !state_.phys_mem_ && !state_.interprocess_) {
      state_.use_vm_heap_ = true;
      busy_heap_.EnableVmHeap();
      free_heap_.EnableVmHeap();
    }
  }

  virtual ~MemoryPool() override {
    if (!busy_heap_.IsEmpty()) {
      LogError("Shouldn't destroy pool with busy allocations!");
    }
    ReleaseAllMemory();
    // Remove memory pool from the list of all pool on the current device
    device_->RemoveMemoryPool(this);
    if (shared_ != nullptr) {
      // Note: The app supposes to close the handle... Double close in Windows will cause a crash
      amd::Os::CloseIpcMemory(0, shared_, sizeof(SharedMemPool));
    }
  }

  /// The same stream can reuse memory without HIP event validation
  void* AllocateMemory(size_t size, Stream* stream, void* dptr = nullptr);

  /// Frees memory by placing memory object with HIP event into free_heap_
  bool FreeMemory(amd::Memory* memory, Stream* stream, Event* event = nullptr);

  /// Check if memory is active and belongs to the busy heap
  bool IsBusyMemory(amd::Memory* memory) const { return busy_heap_.IsActiveMemory(memory); }

  /// Releases all allocations from free_heap_. It can be called on Stream or Device synchronization
  /// @note The caller must make sure it's safe to release memory
  void ReleaseFreedMemory();

  /// Removes a stream from tracking
  void RemoveStream(hip::Stream* stream);

  /// Releases all allocations in MemoryPool
  void ReleaseAllMemory();

  /// Place the allocated memory into the busy heap
  void AddBusyMemory(amd::Memory* memory) { busy_heap_.AddMemory(memory, nullptr); }

  /// Add a safe stream for quick looks-ups if event dependencies option is enabled
  void AddSafeStream(Stream* event_stream, Stream* wait_stream) {
    amd::ScopedLock lock(lock_pool_ops_);
    if (EventDependencies()) {
      free_heap_.AddSafeStream(event_stream, wait_stream);
    }
  }

  /// Trims the pool until it has only min_bytes_to_hold
  void TrimTo(size_t min_bytes_to_hold);

  /// Trims the pool until it has only min_bytes_to_hold
  hip::Device* Device() const { return device_; }

  /// Set memory pool control attributes
  hipError_t SetAttribute(hipMemPoolAttr attr, void* value);

  /// Get memory pool control attributes
  hipError_t GetAttribute(hipMemPoolAttr attr, void* value);

  /// Set memory pool access by different devices
  void SetAccess(hip::Device* device, hipMemAccessFlags flags);

  /// Set memory pool access by different devices
  void GetAccess(hip::Device* device, hipMemAccessFlags* flags);

  /// Frees all busy memory
  void FreeAllMemory(hip::Stream* stream = nullptr);

  /// Exports memory pool into an OS specific handle
  amd::Os::FileDesc Export();

  /// Imports memory pool from an OS specific handle
  bool Import(amd::Os::FileDesc handle);

  /// Returns properties of this memory pool
  const hipMemPoolProps& Properties() const { return properties_; }

  /// Accessors for the pool state
  bool EventDependencies() const { return (state_.event_dependencies_) ? true : false; }
  bool Opportunistic() const { return (state_.opportunistic_) ? true : false; }
  bool InternalDependencies() const { return (state_.internal_dependencies_) ? true : false; }
  bool GraphInUse() const { return (state_.graph_in_use_) ? true : false; }
  void SetGraphInUse() { state_.graph_in_use_ = true; }

 private:
  MemoryPool() = delete;
  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;

  Heap busy_heap_;  //!< Heap of busy allocations
  Heap free_heap_;  //!< Heap of freed allocations
  union {
    struct {
      uint32_t event_dependencies_ : 1;     //!< Event dependencies tracking is enabled
      uint32_t opportunistic_ : 1;          //!< HIP event check is enabled
      uint32_t internal_dependencies_ : 1;  //!< Runtime adds internal events to handle memory
                                            //!< dependencies
      uint32_t interprocess_ : 1;  //!< Memory pool can be used in interprocess communications
      uint32_t graph_in_use_ : 1;  //!< Memory pool was used in a graph execution
      uint32_t phys_mem_ : 1;     //!< Mempool is used for graphs and will have physical allocations
      uint32_t use_vm_heap_ : 1;  //!< Use VM heap or direct allocations
    };
    uint32_t value_;
  } state_;

  hipMemPoolProps properties_;  //!< Properties of the memory pool
  amd::Monitor lock_pool_ops_;  //!< Access to the pool must be lock protected
  std::map<hip::Device*, hipMemAccessFlags>
      access_map_;  //!< Map of access to the pool from devices

  hip::Device* device_;      //!< Hip device the heap will reside
  SharedMemPool* shared_;    //!< Pointer to shared memory for IPC
  uint64_t max_total_size_;  //!< Max of total reserved memory in the pool since last reset
};


}  // namespace hip
