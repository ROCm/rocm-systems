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
  CountedQueuePoolManager() : max_hw_queues_(0) {}
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