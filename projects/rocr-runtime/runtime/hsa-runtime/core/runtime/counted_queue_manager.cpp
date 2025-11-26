/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/inc/counted_queue_manager.h"
#include "core/inc/agent.h"
#include "core/inc/runtime.h"
#include <cassert>

namespace rocr {
namespace core {

constexpr size_t DEFAULT_QUEUE_SIZE = 16384;

static std::map<hsa_amd_queue_priority_t, HSA_QUEUE_PRIORITY> priomap = {
    {HSA_AMD_QUEUE_PRIORITY_LOW, HSA_QUEUE_PRIORITY_MINIMUM},
    {HSA_AMD_QUEUE_PRIORITY_NORMAL, HSA_QUEUE_PRIORITY_NORMAL},
    {HSA_AMD_QUEUE_PRIORITY_HIGH, HSA_QUEUE_PRIORITY_HIGH},
};

CountedQueuePoolManager::CountedQueuePoolManager(core::Agent* agent) : agent_(agent) {
  // Read in GPU_MAX_HW_QUEUES flag value
  max_hw_queues_ = core::Runtime::runtime_singleton_->flag().cp_queues_limit();
}

hsa_status_t CountedQueuePoolManager::AcquireQueue(
    hsa_queue_type_t type, hsa_amd_queue_priority_t priority,
    void (*callback)(hsa_status_t, hsa_queue_t*, void*), void* data, uint64_t flags,
    hsa_queue_t** out_queue) {
  std::lock_guard<std::mutex> lock(mutex_);

  core::Queue* core_queue = FindOrCreateHardwareQueue(type, priority, callback, data, flags);
  if (!core_queue) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  // Create unique SharedQueue structure and store the unique handle in it
  SharedQueue* shared_queue = new SharedQueue();
  if (!shared_queue) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  // Copy amd_queue from HW queue
  shared_queue->amd_queue = core_queue->amd_queue_;

  // Point to the SAME underlying core::Queue (shared HW queue)
  shared_queue->core_queue = core_queue;

  // Create a unique handle from this new SharedQueue
  hsa_queue_t* unique_handle = &shared_queue->amd_queue.hsa_queue;

  // Track metadata
  CountedQueue* counted_q = new CountedQueue(core_queue, callback, data);
  counted_queues_[unique_handle] = counted_q;

  // Increment use count
  core_queue->use_count++;
  
  // Mark as a counted queue, if not already set
  if (!core_queue->is_counted_queue) {
    core_queue->is_counted_queue = true;
  }

  *out_queue = unique_handle;
  return HSA_STATUS_SUCCESS;
}

core::Queue* CountedQueuePoolManager::FindOrCreateHardwareQueue(
    hsa_queue_type_t type, hsa_amd_queue_priority_t priority,
    void (*callback)(hsa_status_t, hsa_queue_t*, void*), void* data, uint64_t flags) {
  auto& pool = hw_queue_pools_[priority];

  // Reuse least-used queue if max reached
  if (pool.size() >= max_hw_queues_) {
    core::Queue* least_used = nullptr;
    uint32_t min_count = UINT32_MAX;

    for (auto* q : pool) {
      if (q->use_count < min_count) {
        min_count = q->use_count;
        least_used = q;
      }
    }
    return least_used;
  }

  // Create a new hardware queue
  core::Queue* cmd_queue = nullptr;
  hsa_status_t status =
      agent_->QueueCreate(DEFAULT_QUEUE_SIZE, type, 0, callback, data, 0, 0, &cmd_queue);
  if (status != HSA_STATUS_SUCCESS) return nullptr;

  status = cmd_queue->SetPriority(priomap[priority]);
  if (status != HSA_STATUS_SUCCESS) return nullptr;

  cmd_queue->SetProfiling(true);

  // Add to pool
  pool.push_back(cmd_queue);
  return cmd_queue;
}

hsa_status_t CountedQueuePoolManager::ReleaseQueue(hsa_queue_t* queue) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = counted_queues_.find(queue);
  if (it == counted_queues_.end()) return HSA_STATUS_ERROR;

  CountedQueue* counted_q = it->second;

  // Decrement internal ref count inside core::Queue object
  if (counted_q->hw_queue->use_count > 0) {
    counted_q->hw_queue->use_count--;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t CountedQueuePoolManager::GetQueueInfo(hsa_queue_t* queue,
                                                   hsa_queue_info_attribute_t attribute,
                                                   void* value) {
  if (!queue || !value) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  switch (attribute) {
    case HSA_QUEUE_INFO_USE_COUNT: {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = counted_queues_.find(queue);
      if (it == counted_queues_.end()) {
        // Queue has not been created using hsa_amd_counted_queue_acquire API
        *static_cast<int32_t*>(value) = -1;
      } else {
        if (it->second->hw_queue->use_count == 0) {
          // Queue was created using hsa_amd_counted_queue_acquire API but has been released 
          // it is not in use anymore by any application
          return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        }
        *static_cast<uint32_t*>(value) = it->second->hw_queue->use_count;
      }
      return HSA_STATUS_SUCCESS;
    }

    case HSA_QUEUE_INFO_HW_ID: {
      std::lock_guard<std::mutex> lock(mutex_);
      // Check counted queues map which contains HardwareQueue*
      auto it = counted_queues_.find(queue);
      if (it != counted_queues_.end()) {
        *static_cast<uint32_t*>(value) = it->second->hw_queue->public_handle()->id;
        return HSA_STATUS_SUCCESS;
      }
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

void CountedQueuePoolManager::Cleanup() {
  std::lock_guard<std::mutex> lock(mutex_);

  // Destroy hardware queues
  for (auto& priority_pool : hw_queue_pools_) {
    for (auto* hw_queue : priority_pool.second) {
      if (hw_queue) {
        hw_queue->Destroy();
      }
    }
    priority_pool.second.clear();
  }
  hw_queue_pools_.clear();

  // Clean up counted and shared queues
  for (auto& cq : counted_queues_) {
    CountedQueue* counted_q = cq.second;
    delete counted_q;

    // Recover SharedQueue from unique handle and free memory
    hsa_queue_t* queue_handle = cq.first;
    SharedQueue* shared = reinterpret_cast<SharedQueue*>(
        reinterpret_cast<char*>(queue_handle) - offsetof(SharedQueue, amd_queue.hsa_queue));
    delete shared;
  }
  counted_queues_.clear();
}

CountedQueuePoolManager::~CountedQueuePoolManager() {}


}  // namespace core
}  // namespace rocr
