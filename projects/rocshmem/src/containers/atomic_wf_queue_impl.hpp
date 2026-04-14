/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "atomic_wf_queue.hpp"
#include <iostream>

#include <hip/hip_runtime.h>
#include <cassert>

namespace rocshmem {

/*****************************************************************************
 ******************************* WAVE FREE LIST ******************************
 *****************************************************************************/

template <typename TYPE, typename ALLOCATOR>
AtomicWFQueue<TYPE, ALLOCATOR>::~AtomicWFQueue() {}

template <typename TYPE, typename ALLOCATOR>
__host__ void AtomicWFQueue<TYPE, ALLOCATOR>::deallocate_queue() {
  if (queue_ != nullptr) {
    allocator_.deallocate((void*)queue_);
    queue_ = nullptr;
  }
  size_ = 0;
  curr_size_ = 0;
  head_ = 0;
  tail_ = 0;
}

template <typename TYPE, typename ALLOCATOR>
AtomicWFQueue<TYPE, ALLOCATOR>::AtomicWFQueue(const ALLOCATOR& allocator)
    : allocator_{allocator}, size_{0}, curr_size_{0}, head_{0}, tail_{0} {}

template <typename TYPE, typename ALLOCATOR>
__host__ void AtomicWFQueue<TYPE, ALLOCATOR>::allocate_queue(
  unsigned int size) {

  size_ = size;
  head_ = 0;
  tail_ = 0;
  curr_size_ = 0;
  allocator_.allocate(reinterpret_cast<void**>(&queue_),
                      sizeof(TYPE) * size_);
}

template <typename TYPE, typename ALLOCATOR>
__host__ void AtomicWFQueue<TYPE, ALLOCATOR>::push(const TYPE& val) {
  if (curr_size_ < size_) {
    queue_[tail_] = val;
    tail_ = (tail_ + 1) % size_;
    curr_size_++;
  }
  else {
    std::cerr << "AtomicWfQueue is full: " << curr_size_
              << " elements" << std::endl;
  }
}

template <typename TYPE, typename ALLOCATOR>
__device__ unsigned int
AtomicWFQueue<TYPE, ALLOCATOR>::active_logical_lane_id() {
  uint64_t ballot{__ballot(1)};
  uint64_t my_physical_lane_id{__lane_id()};
  uint64_t all_ones_mask = -1;
  uint64_t lane_mask{all_ones_mask << my_physical_lane_id};
  uint64_t inverted_mask{~lane_mask};
  uint64_t lower_active_lanes{ballot & inverted_mask};
  unsigned int my_logical_lane_id{__popcll(lower_active_lanes)};
  return my_logical_lane_id; 
}

template <typename TYPE, typename ALLOCATOR>
__device__ TYPE AtomicWFQueue<TYPE, ALLOCATOR>::broadcast_lds(
  bool lowest_active, TYPE value) {

  /**
   * Shared array to broadcast data within each wavefront
   * Max threads per block = 1024, wavefront size = 64 (in most GPUs)
   * Maximum array size required = 1024/64 = 16
   */
  constexpr size_t SIZE = 1024 / WF_SIZE;
  __shared__ TYPE value_per_warp[SIZE];
  auto wavefront_id {get_flat_block_id() / WF_SIZE};
  if (lowest_active) {
    value_per_warp[wavefront_id] = value;
    __threadfence_block();
  }
  return value_per_warp[wavefront_id];
}

template <typename TYPE, typename ALLOCATOR>
__device__ void AtomicWFQueue<TYPE, ALLOCATOR>::enqueue(const TYPE& val) {
  unsigned int my_active_lane_id {active_logical_lane_id()};
  bool is_lowest_active_lane {my_active_lane_id == 0};
  if (is_lowest_active_lane) {
    /**
     * Prevents multiple wavefronts from simultaneously entering the enqueue
     * operation. Ensures a first-come, first-serve execution order
     */
    TicketLockGuard<MutexType> guard(*enqueue_mutex_.get());

    /**
     * There should always be space available.
     * If the queue is full, it indicates an unexpected issue.
     */
    assert(!is_full());

    int next_tail = (tail_ + 1) % size_;
    queue_[tail_] = val;

    tail_ = next_tail;
    atomic_add(&curr_size_, 1);
  }
}

template <typename TYPE, typename ALLOCATOR>
__device__ TYPE AtomicWFQueue<TYPE, ALLOCATOR>::dequeue() {
  TYPE ret_val {TYPE()};
  unsigned int my_active_lane_id {active_logical_lane_id()};
  bool is_lowest_active_lane {my_active_lane_id == 0};
  if (is_lowest_active_lane) {
    /**
     * Prevents multiple wavefronts from simultaneously entering the dequeue
     * operation. Ensures a first-come, first-serve execution order
     */
    TicketLockGuard<MutexType> guard(*dequeue_mutex_.get());

    // queue is empty, wait until data is available
    while (is_empty()) {}

    int next_head = (head_ + 1) % size_;

    ret_val = queue_[head_];

    head_ = next_head;
    atomic_sub(&curr_size_, 1);
  }

  ret_val = broadcast_lds(is_lowest_active_lane, ret_val);
  // TYPE should support + operation
  ret_val += my_active_lane_id;

  return ret_val;
}

}  // namespace rocshmem
