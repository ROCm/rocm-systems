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

#ifndef ROCSHMEM_ATOMIC_WF_QUEUE_GTEST_HPP
#define ROCSHMEM_ATOMIC_WF_QUEUE_GTEST_HPP

#include "../src/containers/atomic_wf_queue_impl.hpp"
#include "gtest/gtest.h"
#include "../src/memory/hip_allocator.hpp"
#include "../src/util.hpp"
#include <iostream>

namespace rocshmem {

template <typename AWFQueue>
__global__ void wf_lane_ids(AWFQueue* awf_queue, unsigned int* device_array,
                           int wf_size) {

  int t_id {get_flat_id()};
  int lane_id {t_id % wf_size};
  device_array[t_id] = awf_queue->active_logical_lane_id();
}

template <typename AWFQueue>
__global__ void concurrent_enqueue_dequeue(
    AWFQueue* awf_queue,
    unsigned int* device_array) {

  int t_id {get_flat_id()};
  int val {awf_queue->dequeue()};
  device_array[t_id] = val;
  awf_queue->enqueue(val);
}

class AtomicWFQueueTestFixture : public ::testing::Test {
 public:
  AtomicWFQueueTestFixture() {
    awf_queue = awf_queue_proxy.get();

    int device_id {};
    hipDeviceProp_t device_props;
    CHECK_HIP(hipGetDevice(&device_id));
    CHECK_HIP(hipGetDeviceProperties(&device_props, device_id));
    
    wf_size = device_props.warpSize;
  }

  ~AtomicWFQueueTestFixture() {}

  void get_thread_lane_ids(unsigned int num_threads = 4) {
    
    unsigned int *device_array {nullptr};

    hip_allocator_.allocate(reinterpret_cast<void**>(&device_array),
                            sizeof(unsigned int) * num_threads);
  

    hipLaunchKernelGGL(wf_lane_ids, 1, num_threads, 0, nullptr,
                       awf_queue, device_array, wf_size);

    CHECK_HIP(hipDeviceSynchronize());

    hip_allocator_.deallocate(device_array);

    for (unsigned int i{0}; i < num_threads; i++) {
      EXPECT_EQ(device_array[i], i % wf_size);
    }
  }

  void init_queue(int num_blocks, int block_size) {
    int num_elems = num_blocks * ((block_size - 1) / wf_size + 1);

    awf_queue->allocate_queue(num_elems);

    for (int i{0}; i < num_elems; i++) {
      awf_queue->push(i);
    }

    EXPECT_EQ(awf_queue->get_curr_size(), num_elems);
    EXPECT_EQ(awf_queue->get_queue_size(), num_elems);
    EXPECT_EQ(awf_queue->get_tail(), awf_queue->get_head());

    awf_queue->deallocate_queue();
  }

  void verify(unsigned int *arr, int num_blocks, int block_size) {
    unsigned int expected_val {};
    unsigned int lane_id {};
    unsigned int idx {};
    for (unsigned int i{0}; i < num_blocks; i++) {
      for (unsigned int j{0}; j < block_size; j++) {
        idx = i * block_size + j;
        lane_id = j % wf_size;
        if (!lane_id) {
          expected_val = arr[idx];
        }
        EXPECT_EQ(arr[idx], expected_val + lane_id);
      }
    }
  }

  void dequeue_enqueue(int num_blocks, int block_size, int queue_size) {

    int num_threads {num_blocks * block_size};
    unsigned int *device_array {nullptr};

    hip_allocator_.allocate(reinterpret_cast<void**>(&device_array),
                            sizeof(unsigned int) * num_threads);

    awf_queue->allocate_queue(queue_size);

    for (int i{0}; i < queue_size; i++) {
      awf_queue->push(i);
    }

    EXPECT_EQ(awf_queue->get_curr_size(), queue_size);
    EXPECT_EQ(awf_queue->get_queue_size(), queue_size);
    EXPECT_EQ(awf_queue->get_tail(), awf_queue->get_head());

    hipLaunchKernelGGL(concurrent_enqueue_dequeue, num_blocks, block_size, 0,
                       nullptr, awf_queue, device_array);
    CHECK_HIP(hipDeviceSynchronize());

    EXPECT_EQ(awf_queue->get_curr_size(), queue_size);
    EXPECT_EQ(awf_queue->get_tail(), awf_queue->get_head());

    verify(device_array, num_blocks, block_size);

    hip_allocator_.deallocate(device_array);
    awf_queue->deallocate_queue();
  }

  protected:

    AtomicWFQueueProxy<HIPAllocator, int> awf_queue_proxy{};
    AtomicWFQueue<int, HIPAllocator>* awf_queue{};

    /**
     * @brief An allocator to create objects in device memory.
     */
    HIPAllocator hip_allocator_ {};

    /**
     * @brief Wavefront size.
     */
    int wf_size {};
};

}  // namespace rocshmem

#endif  // ROCSHMEM_ATOMIC_WF_QUEUE_GTEST_HPP
