/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates. 
 * 
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTME_CORE_INC_COUNTED_QUEUE_MANAGER_H_
#define HSA_RUNTME_CORE_INC_COUNTED_QUEUE_MANAGER_H_


#include "hsa.h"
#include "hsa_ext_amd.h"
#include "core/inc/agent.h"
#include "core/inc/queue.h"
#include "core/inc/runtime.h"

#include "map"

namespace rocr {
namespace core {

// Wrapper for real hardware queue tracking and its references
// Each entry in queue pool is of this type
struct HardwareQueue {
  hsa_queue_t* hw_queue;  // Actual hardware queue
  uint32_t use_count;     // Internal reference count
  hsa_amd_queue_priority_t priority;
  hsa_agent_t agent;

  HardwareQueue(hsa_queue_t* q, hsa_amd_queue_priority_t prio, hsa_agent_t ag)
      : hw_queue(q), use_count(0), priority(prio), agent(ag) {}
};

// Wrapper around HW queue to provide unique logical handles to multiple users, even when same HW
// queue is used. Also store callbacks per logical handle, not HW handle
struct CountedQueue {
  HardwareQueue* hw_queue;  // Pointer to shared hardware queue
  // callback per unique logical handle, not hardware handle
  void (*callback)(hsa_status_t, hsa_queue_t*, void*);
  void* callback_data;

  CountedQueue(HardwareQueue* hw, void (*cb)(hsa_status_t, hsa_queue_t*, void*), void* data)
      : hw_queue(hw), callback(cb), callback_data(data) {}
};

// Singleton manager for a pool of counted queues
class CountedQueuePoolManager {
 public:
  static CountedQueuePoolManager& Instance();

  hsa_status_t AcquireQueue(hsa_agent_t agent, hsa_queue_type_t type,
                            hsa_amd_queue_priority_t priority,
                            void (*callback)(hsa_status_t, hsa_queue_t*, void*), void* data,
                            uint64_t flags, hsa_queue_t** out_queue);

  hsa_status_t ReleaseQueue(hsa_queue_t* queue);

  hsa_status_t GetQueueInfo(hsa_queue_t* queue, hsa_counted_queue_info_attribute_t attribute,
                            void* value);

  static bool IsInstanceCreated();

  bool IsCountedQueue(hsa_queue_t* queue);

 private:
  CountedQueuePoolManager() : max_hw_queues_(4) {}
  ~CountedQueuePoolManager();

  // Disable copy and assignment
  CountedQueuePoolManager(const CountedQueuePoolManager&) = delete;
  CountedQueuePoolManager& operator=(const CountedQueuePoolManager&) = delete;

  HardwareQueue* FindOrCreateHardwareQueue(hsa_agent_t agent, hsa_queue_type_t type,
                                           hsa_amd_queue_priority_t priority,
                                           void (*callback)(hsa_status_t, hsa_queue_t*, void*),
                                           void* data, uint64_t flags);

  // Map from (agent+priority) to the list of hardware queues each combination of (agent,priority) has
  std::map<uint64_t, std::vector<std::unique_ptr<HardwareQueue>>> hw_queue_pools_;

  // Map from unique queue handle to CountedQueue metadata (includes the hw_queue and callbacks used)
  std::map<hsa_queue_t*, std::unique_ptr<CountedQueue>> counted_queues_;

  uint32_t max_hw_queues_;
  std::mutex mutex_;
  static std::atomic<bool> instance_created_;
};


}  // namespace core
}  // namespace rocr

#endif // HSA_RUNTME_CORE_INC_COUNTED_QUEUE_MANAGER_H_