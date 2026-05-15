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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_HPP_

#include "hdp_proxy.hpp"
#include "queue_proxy.hpp"
#include "queue_desc_proxy.hpp"

namespace rocshmem {

class MPITransport;

class Queue {
 public:
  Queue();

  Queue(size_t max_queues, size_t queue_size);

  bool process(uint64_t queue_index, MPITransport* transport);

  uint64_t get_read_index(uint64_t queue_index);

  void increment_read_index(uint64_t queue_index);

  void flush_hdp();

  void sfence_flush_hdp();

  void notify(volatile char *status);

  uint64_t size();

  __host__ __device__ queue_desc_t* descriptor(uint64_t index);

  queue_element_t* elements(uint64_t index);

 private:
  queue_element* next_element(uint64_t queue_index);

  void copy_element_to_cache(uint64_t queue_index);

  QueueProxyT queue_proxy_{};

  QueueDescProxyT queue_desc_proxy_{};

  QueueElementProxyT queue_element_cache_proxy_{};

  HdpProxy<HIPHostAllocator> hdp_proxy_{};

  size_t max_queues_{};

  size_t queue_size_{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_HPP_
