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

// Static member definitions
std::unordered_set<hsa_queue_t*> CountedQueuePoolManager::released_handles_;
std::deque<hsa_queue_t*> CountedQueuePoolManager::released_handles_order_;
std::mutex CountedQueuePoolManager::released_handles_mutex_;

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

bool CountedQueuePoolManager::IsValidQueueHandle(hsa_queue_t* queue) {
  std::lock_guard<std::mutex> lock(mutex_);
  return counted_queues_.find(queue) != counted_queues_.end();
}

bool CountedQueuePoolManager::IsReleasedHandle(hsa_queue_t* handle) {
  std::lock_guard<std::mutex> lock(released_handles_mutex_);
  return released_handles_.find(handle) != released_handles_.end();
}

void CountedQueuePoolManager::TrackReleasedHandle(hsa_queue_t* handle) {
  std::lock_guard<std::mutex> lock(released_handles_mutex_);

  released_handles_.insert(handle);
  released_handles_order_.push_back(handle);

  // Limit to 32K handles (~256KB memory) for use-after-release detection
  constexpr size_t kMaxReleasedHandles = 32768;
  if (released_handles_order_.size() > kMaxReleasedHandles) {
    hsa_queue_t* oldest = released_handles_order_.front();
    released_handles_order_.pop_front();
    released_handles_.erase(oldest);
  }
}

void CountedQueuePoolManager::ClearReleasedHandles() {
  std::lock_guard<std::mutex> lock(released_handles_mutex_);
  released_handles_.clear();
  released_handles_order_.clear();
}

hsa_status_t CountedQueuePoolManager::ReleaseQueue(hsa_queue_t* queue) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = counted_queues_.find(queue);
  if (it == counted_queues_.end()) return HSA_STATUS_ERROR;

  CountedQueue* counted_q = it->second.get();

  // Decrement internal ref count inside core::Queue object
  if (counted_q->hw_queue->use_count > 0) {
    counted_q->hw_queue->use_count--;
  }

  // Track the handle address for use-after-release detection (~8 bytes per handle)
  TrackReleasedHandle(queue);

  // Delete the SharedQueue immediately (no longer kept alive)
  SharedQueue* shared = reinterpret_cast<SharedQueue*>(reinterpret_cast<char*>(queue) -
                                                       offsetof(SharedQueue, amd_queue.hsa_queue));
  delete shared;

  counted_queues_.erase(it);

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

  for (auto& cq : counted_queues_) {
    hsa_queue_t* queue_handle = cq.first;
    SharedQueue* shared = reinterpret_cast<SharedQueue*>(
        reinterpret_cast<char*>(queue_handle) - offsetof(SharedQueue, amd_queue.hsa_queue));
    delete shared;
  }
  counted_queues_.clear();

  // Clear released handle tracking (static, so clear once during shutdown)
  ClearReleasedHandles();
}

}  // namespace core
}  // namespace rocr
