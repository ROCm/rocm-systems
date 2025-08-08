////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
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

// HSA runtime C++ interface file.

#ifndef HSA_RUNTME_CORE_INC_COMMAND_QUEUE_H_
#define HSA_RUNTME_CORE_INC_COMMAND_QUEUE_H_

#include <sstream>

#include "core/common/shared.h"
#include "core/inc/checked.h"
#include "core/inc/memory_region.h"
#include "core/util/utils.h"
#include "inc/amd_hsa_queue.h"
#include "inc/hsa_ext_amd.h"
#include "hsakmt/hsakmt.h"

namespace rocr {
namespace core {
struct AqlPacket {

  union {
    struct {
      uint16_t header;
      struct {
        uint8_t user_data[62];
      } body;
     } packet;
    struct {
      uint16_t header;
      uint8_t format;
      uint8_t rest[61];
    } amd_vendor;
    hsa_kernel_dispatch_packet_t dispatch;
    hsa_barrier_and_packet_t barrier_and;
    hsa_barrier_or_packet_t barrier_or;
    hsa_agent_dispatch_packet_t agent;
  };

  // Access the type field from a packet header. The caller is responsible for
  // loading the header using an atomic or ordinary load as appropriate.
  static uint8_t type(uint16_t header) {
    return ((header >> HSA_PACKET_HEADER_TYPE) & ((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1));
  }

  // Determine if a packet is valid. The caller is responsible for loading the
  // header using an atomic or ordinary load as appropriate.
  static bool IsValid(uint16_t header) {
    return ((type(header) <= HSA_PACKET_TYPE_BARRIER_OR) &&
            (type(header) != HSA_PACKET_TYPE_INVALID));
  }

  bool __forceinline IsDispatchAndNeedsScratch() const {
    assert(IsValid(packet.header) && "Invalid packet in dynamic scratch handler.");

    if (type(packet.header) != HSA_PACKET_TYPE_KERNEL_DISPATCH ||
        dispatch.private_segment_size == 0)
      return false;

    assert((dispatch.workgroup_size_x != 0) && (dispatch.workgroup_size_y != 0) &&
           (dispatch.workgroup_size_z != 0) && "Invalid dispatch dimension.");

    return true;
  }

  std::string string() const {
    std::stringstream string;
    uint8_t t = type(packet.header);

    static const char* type_names[] = {
        "HSA_PACKET_TYPE_VENDOR_SPECIFIC", "HSA_PACKET_TYPE_INVALID",
        "HSA_PACKET_TYPE_KERNEL_DISPATCH", "HSA_PACKET_TYPE_BARRIER_AND",
        "HSA_PACKET_TYPE_AGENT_DISPATCH",  "HSA_PACKET_TYPE_BARRIER_OR"};

    if (t >= sizeof(type_names) / sizeof(const char*)) {
      string << "type: UNKNOWN#" << t;
      return string.str();
    }

    string << "type: " << type_names[t]
           << "\nbarrier: " << ((dispatch.header >> HSA_PACKET_HEADER_BARRIER) &
                                ((1 << HSA_PACKET_HEADER_WIDTH_BARRIER) - 1))
           << "\nacquire: " << ((dispatch.header >> HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) &
                                ((1 << HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE) - 1))
           << "\nrelease: " << ((dispatch.header >> HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE) &
                                ((1 << HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE) - 1));

    if (t == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      string << "\nDim: " << dispatch.setup
             << "\nworkgroup_size: " << dispatch.workgroup_size_x << ", "
             << dispatch.workgroup_size_y << ", " << dispatch.workgroup_size_z
             << "\ngrid_size: " << dispatch.grid_size_x << ", "
             << dispatch.grid_size_y << ", " << dispatch.grid_size_z
             << "\nprivate_size: " << dispatch.private_segment_size
             << "\ngroup_size: " << dispatch.group_segment_size
             << "\nkernel_object: " << dispatch.kernel_object
             << "\nkern_arg: " << dispatch.kernarg_address
             << "\nsignal: " << dispatch.completion_signal.handle;
    }

    if ((t == HSA_PACKET_TYPE_BARRIER_AND) ||
        (t == HSA_PACKET_TYPE_BARRIER_OR)) {
      for (int i = 0; i < 5; i++)
        string << "\ndep[" << i << "]: " << barrier_and.dep_signal[i].handle;
      string << "\nsignal: " << barrier_and.completion_signal.handle;
    }

    return string.str();
  }
};

class Queue;

/// @brief Helper structure to simplify conversion of amd_queue_v2_t and
/// core::Queue object.
struct SharedQueue {
  amd_queue_v2_t amd_queue;
  Queue* core_queue;
};

/// @brief Class Queue which encapsulate user mode queues and
/// provides Api to access its Read, Write indices using Acquire,
/// Release and Relaxed semantics.
/*
Queue is intended to be an pure interface class and may be wrapped or replaced
by tools.
All funtions other than Convert and public_handle must be virtual.
*/
class Queue : public Checked<0xFA3906A679F9DB49> {
 public:
  Queue(SharedQueue* shared_queue, uint64_t queue_flags)
      : Queue(shared_queue, queue_flags, false) {}

  Queue(SharedQueue* shared_queue, uint64_t queue_flags, bool pcie_write_ordering)
      : amd_queue_(shared_queue->amd_queue),
        shared_queue_(shared_queue),
        flags_(queue_flags),
        pcie_write_ordering_(pcie_write_ordering) {
    public_handle_ = Convert(this);
    shared_queue->core_queue = this;
  }

  virtual ~Queue() {}

  virtual void Destroy() { delete this; }

  /// @brief Returns the handle of Queue's public data type
  ///
  /// @param queue Pointer to an instance of Queue implementation object
  ///
  /// @return hsa_queue_t * Pointer to the public data type of a queue
  static __forceinline hsa_queue_t* Convert(Queue* queue) {
    return (queue != nullptr) ? &queue->amd_queue_.hsa_queue : nullptr;
  }

  /// @brief Transform the public data type of a Queue's data type into an
  //  instance of it Queue class object
  ///
  /// @param queue Handle of public data type of a queue
  ///
  /// @return Queue * Pointer to the Queue's implementation object
  static __forceinline Queue* Convert(const hsa_queue_t* queue) {
    return (queue != nullptr)
        ? reinterpret_cast<SharedQueue*>(reinterpret_cast<uintptr_t>(queue) -
                                         offsetof(SharedQueue, amd_queue.hsa_queue))->core_queue
        : nullptr;
  }

  /// @brief Inactivate the queue object. Once inactivate a
  /// queue cannot be used anymore and must be destroyed
  ///
  /// @return hsa_status_t Status of request
  virtual hsa_status_t Inactivate() = 0;

  /// @brief Change the scheduling priority of the queue
  virtual hsa_status_t SetPriority(HSA_QUEUE_PRIORITY priority) = 0;

  /// @brief Reads the Read Index of Queue using Acquire semantics
  ///
  /// @return uint64_t Value of Read index
  virtual uint64_t LoadReadIndexAcquire() = 0;

  /// @brief Reads the Read Index of Queue using Relaxed semantics
  ///
  /// @return uint64_t Value of Read index
  virtual uint64_t LoadReadIndexRelaxed() = 0;

  /// @brief Reads the Write Index of Queue using Acquire semantics
  ///
  /// @return uint64_t Value of Write index
  virtual uint64_t LoadWriteIndexAcquire() = 0;

  /// Reads the Write Index of Queue using Relaxed semantics
  ///
  /// @return uint64_t Value of Write index
  virtual uint64_t LoadWriteIndexRelaxed() = 0;

  /// @brief Updates the Read Index of Queue using Relaxed semantics
  ///
  /// @param value New value of Read index to update
  virtual void StoreReadIndexRelaxed(uint64_t value) = 0;

  /// @brief Updates the Read Index of Queue using Release semantics
  ///
  /// @param value New value of Read index to update
  virtual void StoreReadIndexRelease(uint64_t value) = 0;

  /// @brief Updates the Write Index of Queue using Relaxed semantics
  ///
  /// @param value New value of Write index to update
  virtual void StoreWriteIndexRelaxed(uint64_t value) = 0;

  /// @brief Updates the Write Index of Queue using Release semantics
  ///
  /// @param value New value of Write index to update
  virtual void StoreWriteIndexRelease(uint64_t value) = 0;

  /// @brief Compares and swaps Write index using Acquire and Release semantics
  ///
  /// @param expected Current value of write index
  ///
  /// @param value Value of new write index
  ///
  /// @return uint64_t Value of write index before the update
  virtual uint64_t CasWriteIndexAcqRel(uint64_t expected, uint64_t value) = 0;

  /// @brief Compares and swaps Write index using Acquire semantics
  ///
  /// @param expected Current value of write index
  ///
  /// @param value Value of new write index
  ///
  /// @return uint64_t Value of write index before the update
  virtual uint64_t CasWriteIndexAcquire(uint64_t expected, uint64_t value) = 0;

  /// @brief Compares and swaps Write index using Relaxed semantics
  ///
  /// @param expected Current value of write index
  ///
  /// @param value Value of new write index
  ///
  /// @return uint64_t Value of write index before the update
  virtual uint64_t CasWriteIndexRelaxed(uint64_t expected, uint64_t value) = 0;

  /// @brief Compares and swaps Write index using Release semantics
  ///
  /// @param expected Current value of write index
  ///
  /// @param value Value of new write index
  ///
  /// @return uint64_t Value of write index before the update
  virtual uint64_t CasWriteIndexRelease(uint64_t expected, uint64_t value) = 0;

  /// @brief Updates the Write index using Acquire and Release semantics
  ///
  /// @param value Value of new write index
  ///
  /// @return uint64_t Value of write index before the update
  virtual uint64_t AddWriteIndexAcqRel(uint64_t value) = 0;

  /// @brief Updates the Write index using Acquire semantics
  ///
  /// @param value Value of new write index
  ///
  /// @return uint64_t Value of write index before the update
  virtual uint64_t AddWriteIndexAcquire(uint64_t value) = 0;

  /// @brief Updates the Write index using Relaxed semantics
  ///
  /// @param value Value of new write index
  ///
  /// @return uint64_t Value of write index before the update
  virtual uint64_t AddWriteIndexRelaxed(uint64_t value) = 0;

  /// @brief Updates the Write index using Release semantics
  ///
  /// @param value Value of new write index
  ///
  /// @return uint64_t Value of write index before the update
  virtual uint64_t AddWriteIndexRelease(uint64_t value) = 0;

  /// @brief Set CU Masking
  ///
  /// @param num_cu_mask_count size of mask bit array
  ///
  /// @param cu_mask pointer to cu mask
  ///
  /// @return hsa_status_t
  virtual hsa_status_t SetCUMasking(uint32_t num_cu_mask_count, const uint32_t* cu_mask) = 0;

  /// @brief Get CU Masking
  ///
  /// @param num_cu_mask_count size of mask bit array
  ///
  /// @param cu_mask pointer to cu mask
  ///
  /// @return hsa_status_t
  virtual hsa_status_t GetCUMasking(uint32_t num_cu_mask_count, uint32_t* cu_mask) = 0;

  /// @brief Submits a block of PM4.
  ///
  /// @param cmd_data pointer to command buffer
  ///
  /// @param cmd_size_b command buffer size in bytes
  ///
  /// @param acquireFence acquire-fence type
  ///
  /// @param releaseFence acquire-fence type
  ///
  /// @param signal optional wait signal
  ///
  /// if @p signal is provided, function will return without waiting for commands to be executed
  /// if @p signal is NULL, waits until commands have been executed.
  virtual void ExecutePM4(uint32_t* cmd_data, size_t cmd_size_b,
                          hsa_fence_scope_t acquireFence = HSA_FENCE_SCOPE_NONE,
                          hsa_fence_scope_t releaseFence = HSA_FENCE_SCOPE_NONE,
                          hsa_signal_t* signal = NULL) = 0;

  virtual void SetProfiling(bool enabled) {
    AMD_HSA_BITS_SET(amd_queue_.queue_properties, AMD_QUEUE_PROPERTIES_ENABLE_PROFILING,
                     (enabled != 0));
  }

  /// @ brief Returns queue queries about the queue
  virtual hsa_status_t GetInfo(hsa_queue_info_attribute_t attribute, void* value) = 0;

  /// @ brief Reports async queue errors to stderr if no other error handler was registered.
  static void DefaultErrorHandler(hsa_status_t status, hsa_queue_t* source, void* data);

  // Handle of AMD Queue struct
  amd_queue_v2_t& amd_queue_;

  hsa_queue_t* public_handle() const { return public_handle_; }

  typedef void* rtti_t;

  bool IsType(rtti_t id) { return _IsA(id); }

  /// @brief Used to determine if the queue's packet buffer was allocated
  /// in the agent's local device memory.
  bool IsDeviceMemRingBuf() const {
    return (flags_ & HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF) != 0;
  }
  /// @brief Used to determine if the queue descriptor was allocated in
  /// the agent's local device memory.
  bool IsDeviceMemQueueDescriptor() const {
    return (flags_ & HSA_AMD_QUEUE_CREATE_DEVICE_MEM_QUEUE_DESCRIPTOR) != 0;
  }

  bool needsPcieOrdering() const { return pcie_write_ordering_; }

 protected:
  static void set_public_handle(Queue* ptr, hsa_queue_t* handle) {
    ptr->do_set_public_handle(handle);
  }
  virtual void do_set_public_handle(hsa_queue_t* handle) {
    public_handle_ = handle;
  }

  virtual bool _IsA(rtti_t id) const = 0;

  SharedQueue* shared_queue_;

  hsa_queue_t* public_handle_;

  /// Next available queue id.
  uint64_t GetQueueId() { return hsa_queue_counter_++; }

 private:

  // HSA Queue ID - used to bind a unique ID
  static std::atomic<uint64_t> hsa_queue_counter_;

  const uint64_t flags_;
  bool pcie_write_ordering_ = false;

  DISALLOW_COPY_AND_ASSIGN(Queue);
};
}   //  namespace core
}   //  namespace rocr

#endif  // header guard
