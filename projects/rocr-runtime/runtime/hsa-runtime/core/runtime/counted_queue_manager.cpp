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

// HSA C to C++ interface implementation.
// This file does argument checking and conversion to C++.

#include "core/inc/counted_queue_manager.h"

namespace rocr {

namespace core {

static std::map<hsa_amd_queue_priority_t, HSA_QUEUE_PRIORITY> priomap = {
    {HSA_AMD_QUEUE_PRIORITY_LOW, HSA_QUEUE_PRIORITY_MINIMUM},
    {HSA_AMD_QUEUE_PRIORITY_NORMAL, HSA_QUEUE_PRIORITY_NORMAL},
    {HSA_AMD_QUEUE_PRIORITY_HIGH, HSA_QUEUE_PRIORITY_HIGH},
};

// Validate priority enum value
static bool IsValidPriority(hsa_amd_queue_priority_t priority) {
  return priority == HSA_AMD_QUEUE_PRIORITY_LOW || priority == HSA_AMD_QUEUE_PRIORITY_NORMAL ||
      priority == HSA_AMD_QUEUE_PRIORITY_HIGH;
}

// Generate a 64-bit unique key using agent+priority combination for hw queue pool lookup
static uint64_t MakePoolKey(hsa_agent_t agent, hsa_amd_queue_priority_t priority) {
  return (static_cast<uint64_t>(agent.handle) << 32) | static_cast<uint64_t>(priority);
}

// Singleton accessor
CountedQueuePoolManager& CountedQueuePoolManager::Instance() {
  static CountedQueuePoolManager instance;
  return instance;
}

hsa_status_t CountedQueuePoolManager::AcquireQueue(
    hsa_agent_t agent, hsa_queue_type_t type, hsa_amd_queue_priority_t priority,
    void (*callback)(hsa_status_t, hsa_queue_t*, void*), void* data, uint64_t flags,
    hsa_queue_t** out_queue) {
  // Validate parameters
  if (!out_queue || !IsValidPriority(priority)) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  // support only multi-producer queues
  if (type != HSA_QUEUE_TYPE_MULTI) return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;

  // Validate agent
  core::Agent* gpu_agent = core::Agent::Convert(agent);
  if (!gpu_agent || !gpu_agent->IsValid()) return HSA_STATUS_ERROR_INVALID_AGENT;

  // Find existing or create a new hardware queue
  std::lock_guard<std::mutex> lock(mutex_);
  HardwareQueue* hw_queue = FindOrCreateHardwareQueue(agent, type, priority, callback, data, flags);
  if (!hw_queue) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  // Create a new CountedQueue wrapper to generate unique handles for same HW queues
  hsa_queue_t* unique_handle = new hsa_queue_t(*hw_queue->hw_queue);
  auto counted_queue = std::make_unique<CountedQueue>(hw_queue, callback, data);
  counted_queues_[unique_handle] = std::move(counted_queue);

  // Increment internal ref count
  hw_queue->use_count++;

  // Store wrapper to track all queues created via this API and return the new logical queue handle
  *out_queue = unique_handle;

  return HSA_STATUS_SUCCESS;
}

HardwareQueue* CountedQueuePoolManager::FindOrCreateHardwareQueue(
    hsa_agent_t agent, hsa_queue_type_t type, hsa_amd_queue_priority_t priority,
    void (*callback)(hsa_status_t, hsa_queue_t*, void*), void* data, uint64_t flags) {
  // Create pool key using agent and priority values and get the queue pool for the given agent
  uint64_t pool_key = MakePoolKey(agent, priority);
  auto& pool = hw_queue_pools_[pool_key];

  // Check if the number of queues on this agent has hit the limit
  max_hw_queues_ = core::Runtime::runtime_singleton_->flag().cp_queues_limit();
  if (pool.size() >= max_hw_queues_) {
    // Get the least used hw queue
    HardwareQueue* leastSharedQueue = nullptr;
    uint32_t min_count = UINT32_MAX;

    for (auto& q : pool) {
      if (q->use_count < min_count) {
        min_count = q->use_count;
        leastSharedQueue = q.get();
      }
    }
    return leastSharedQueue;
  }

  // Within limit, can create a new hardware queue
  hsa_queue_t* new_hw_queue = nullptr;
  core::Queue* cmd_queue = nullptr;
  core::Agent* gpu_agent = core::Agent::Convert(agent);
  hsa_status_t status =
      gpu_agent->QueueCreate(0, type, 0, callback, data, 0, 0, &cmd_queue);  // size of queue??
  if (status != HSA_STATUS_SUCCESS) return nullptr;
  assert(cmd_queue != nullptr);

  status = cmd_queue->SetPriority(priomap[priority]);
  // is this how queue priority is set?
  // what else needs to be set after above API call?
  if (status != HSA_STATUS_SUCCESS) return nullptr;

  new_hw_queue = core::Queue::Convert(cmd_queue);

  // Create HardwareQueue wrapper for newly created hwQueue
  auto hw_queue = std::make_unique<HardwareQueue>(new_hw_queue, priority, agent);
  HardwareQueue* result = hw_queue.get();

  pool.push_back(std::move(hw_queue));
  return result;
}

hsa_status_t CountedQueuePoolManager::ReleaseQueue(hsa_queue_t* queue) {
  if (!queue) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  // free up counted_queues_ first
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = counted_queues_.find(queue);
  if (it == counted_queues_.end()) {
    return HSA_STATUS_ERROR;
  }

  // valid queue
  CountedQueue* counted_q = it->second.get();
  HardwareQueue* hw_queue = counted_q->hw_queue;

  // decrement internal ref count
  // Do not destroy underlying HW Queue even if ref count = 0
  if (hw_queue->use_count > 0) {
    hw_queue->use_count--;
  }

  // remove counted queue unique handle entry
  counted_queues_.erase(it);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t CountedQueuePoolManager::GetQueueInfo(hsa_queue_t* queue,
                                                   hsa_counted_queue_info_attribute_t attribute,
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
        *static_cast<uint32_t*>(value) = it->second->hw_queue->use_count;
      }
      return HSA_STATUS_SUCCESS;
    }

    case HSA_QUEUE_INFO_HW_ID: {
      std::lock_guard<std::mutex> lock(mutex_);
      // Check counted queues map which contains HardwareQueue*
      auto it = counted_queues_.find(queue);
      if (it != counted_queues_.end()) {
        *static_cast<uint32_t*>(value) = it->second->hw_queue->hw_queue->id;
        return HSA_STATUS_SUCCESS;
      }
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

void CountedQueuePoolManager::TriggerCallback(hsa_queue_t* queue, hsa_status_t status) {
  void (*callback)(hsa_status_t, hsa_queue_t*, void*) = nullptr;
  void* callback_data = nullptr;

  {
    // Use mutex while searching counted_queues_ only
    // release and then execute the callback function
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = counted_queues_.find(queue);
    if (it == counted_queues_.end()) {
      return;
    }

    CountedQueue* cq = it->second.get();
    callback = cq->callback;
    callback_data = cq->callback_data;
  }

  // Execute the found callback, if any
  if (callback) {
    callback(status, queue, callback_data);
  }
}

CountedQueuePoolManager::~CountedQueuePoolManager() {
  std::lock_guard<std::mutex> lock(mutex_);

  // Delete all logical handles created for users
  for (auto& q : counted_queues_) {
    hsa_queue_t* handle = q.first;
    delete handle;  // free the copy we created with new
  }
  counted_queues_.clear();
}


}  // namespace core
}  // namespace rocr