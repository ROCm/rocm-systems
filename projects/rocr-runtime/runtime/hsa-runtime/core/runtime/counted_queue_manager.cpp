/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/inc/counted_queue_manager.h"
#include "core/inc/agent.h"
#include "core/inc/runtime.h"

namespace rocr {
namespace core {

CountedQueuePoolManager::CountedQueuePoolManager(core::Agent* agent) : agent_(agent) {
  // Read in GPU_MAX_HW_QUEUES and HSA_COUNTED_QUEUE_SIZE flags
  max_hw_queues_ = core::Runtime::runtime_singleton_->flag().cp_queues_limit();
  counted_queue_size_ = core::Runtime::runtime_singleton_->flag().counted_queue_size();
}

hsa_status_t CountedQueuePoolManager::AcquireQueue(
    hsa_queue_type_t type, HSA::hsa_amd_queue_priority_internal_t priority,
    void (*callback)(hsa_status_t, hsa_queue_t*, void*), void* data, uint64_t flags,
    hsa_queue_t** out_queue) {
  std::lock_guard<std::mutex> lock(mutex_);

  core::Queue* core_queue = FindOrCreateHardwareQueue(type, priority, callback, data, flags);
  if (!core_queue) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  // Create unique SharedQueue structure and store the unique handle in it
  SharedQueue* shared_queue = new (std::nothrow) SharedQueue();
  if (!shared_queue) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  // Copy amd_queue from HW queue
  shared_queue->amd_queue = core_queue->amd_queue_;

  // Point to the SAME underlying core::Queue (shared HW queue)
  shared_queue->core_queue = core_queue;

  // Create a unique handle from this new SharedQueue
  hsa_queue_t* unique_handle = &shared_queue->amd_queue.hsa_queue;

  // Track metadata
  auto counted_q = std::make_unique<CountedQueue>(core_queue, callback, data);
  counted_queues_[unique_handle] = std::move(counted_q);

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
    hsa_queue_type_t type, HSA::hsa_amd_queue_priority_internal_t priority,
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
      agent_->QueueCreate(counted_queue_size_, type, 0, callback, data, 0, 0, &cmd_queue);
  if (status != HSA_STATUS_SUCCESS) return nullptr;

  status = cmd_queue->SetPriority(priority);
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

  CountedQueue* counted_q = it->second.get();

  // Decrement internal ref count inside core::Queue object
  if (counted_q->hw_queue->use_count > 0) {
    counted_q->hw_queue->use_count--;
    
    // Remove unique handle from map when it is no longer in use by an application
    if (counted_q->hw_queue->use_count == 0) {
      counted_queues_.erase(queue);

      // free the associated shared_queue when removing the counted_queue
      SharedQueue* shared = reinterpret_cast<SharedQueue*>(
        reinterpret_cast<char*>(queue) - offsetof(SharedQueue, amd_queue.hsa_queue));
      delete shared;
    }
  }

  return HSA_STATUS_SUCCESS;
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
    // Recover SharedQueue from unique handle and free memory
    hsa_queue_t* queue_handle = cq.first;
    SharedQueue* shared = reinterpret_cast<SharedQueue*>(
        reinterpret_cast<char*>(queue_handle) - offsetof(SharedQueue, amd_queue.hsa_queue));
    delete shared;
  }
  counted_queues_.clear();
}

}  // namespace core
}  // namespace rocr
