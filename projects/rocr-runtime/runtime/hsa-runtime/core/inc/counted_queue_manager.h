/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTIME_CORE_INC_COUNTED_QUEUE_MANAGER_H_
#define HSA_RUNTIME_CORE_INC_COUNTED_QUEUE_MANAGER_H_


#include "hsa.h"
#include "hsa_ext_amd.h"
#include "core/inc/agent.h"
#include "core/inc/runtime.h"
#include <map>
#include <mutex>
#include <vector>
#include <memory>

namespace rocr {
namespace core {

// Wrapper for a logical counted queue (unique handle + callback)
struct CountedQueue {
  core::Queue* hw_queue; // this will store the public handle of HW Queue (hsa_queue_t)
  void (*callback)(hsa_status_t, hsa_queue_t*, void*);
  void* callback_data;

  CountedQueue(core::Queue* hw, void (*cb)(hsa_status_t, hsa_queue_t*, void*), void* data)
      : hw_queue(hw), callback(cb), callback_data(data) {}
};

// Manages the pool of counted queues for a single GPU agent
class CountedQueuePoolManager {
 public:
  explicit CountedQueuePoolManager(core::Agent*);

  // Acquire a queue (either reuse or create new)
  hsa_status_t AcquireQueue(hsa_queue_type_t type, HSA::hsa_amd_queue_priority_internal_t priority,
                            void (*callback)(hsa_status_t, hsa_queue_t*, void*), void* data,
                            uint64_t flags, hsa_queue_t** out_queue);

  // Release a logical queue
  hsa_status_t ReleaseQueue(hsa_queue_t* queue);

  // Called during hsa_shutdown to remove all user and CP queues
  void Cleanup();

 private:
  core::Queue* FindOrCreateHardwareQueue(hsa_queue_type_t type, HSA::hsa_amd_queue_priority_internal_t priority,
                                           void (*callback)(hsa_status_t, hsa_queue_t*, void*),
                                           void* data, uint64_t flags);
  
  core::Agent* agent_; // pointer to the gpu agent that owns this pool
  uint32_t max_hw_queues_;
  size_t counted_queue_size_;
  std::mutex mutex_;

  // Pool of hw queues by priority on the agent
  std::map<HSA::hsa_amd_queue_priority_internal_t, std::vector<core::Queue*>> hw_queue_pools_;

  // Map from unique handle to CountedQueue (hw queue, metadata per acquire request)
  std::map<hsa_queue_t*, std::unique_ptr<CountedQueue>> counted_queues_;
};

}  // namespace core
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_COUNTED_QUEUE_MANAGER_H_
