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

#include "queue.hpp"
#include "envvar.hpp"
#include "mpi_transport.hpp"

namespace rocshmem {

Queue::Queue() { }

Queue::Queue(size_t max_queues, size_t queue_size)
    : max_queues_{max_queues},
      queue_size_{queue_size},
      queue_proxy_{max_queues, queue_size},
      queue_desc_proxy_{max_queues} { }

uint64_t Queue::get_read_index(uint64_t queue_index) {
  return descriptor(queue_index)->read_index % queue_size_;
}

void Queue::increment_read_index(uint64_t queue_index) {
  descriptor(queue_index)->read_index++;
}

bool Queue::process(uint64_t queue_index, MPITransport* transport) {
  auto next_elem{next_element(queue_index)};
  if (next_elem->notify_cpu.valid) {
    transport->insertRequest(next_elem, queue_index);
    auto queues{queue_proxy_.get()};
    queues[queue_index][get_read_index(queue_index)].notify_cpu.valid = 0;
    increment_read_index(queue_index);
    return true;
  }
  return false;
}

queue_element* Queue::next_element(uint64_t queue_index) {
  queue_element *next_elem{nullptr};
  if (!envvar::ro::net_cpu_queue) {
    hdp_proxy_.get()->hdp_flush();
    copy_element_to_cache(queue_index);
    next_elem = queue_element_cache_proxy_.get();
  } else {
    auto queues {queue_proxy_.get()};
    auto read_slot{get_read_index(queue_index)};
    next_elem = &queues[queue_index][read_slot];
  }
  return next_elem;
}

void Queue::copy_element_to_cache(uint64_t queue_index) {
    auto element{queue_element_cache_proxy_.get()};
    auto read_slot{get_read_index(queue_index)};
    auto queues {queue_proxy_.get()};
    ::memcpy(element, &queues[queue_index][read_slot], sizeof(queue_element_t));
}

void Queue::flush_hdp() {
  if (envvar::ro::net_cpu_queue) {
    hdp_proxy_.get()->hdp_flush();
  }
}

void Queue::sfence_flush_hdp() {
  if (envvar::ro::net_cpu_queue) {
    asm volatile("sfence" ::: "memory");
    hdp_proxy_.get()->hdp_flush();
  }
}

void Queue::notify(volatile char* status) {
  *status = 1;
}

uint64_t Queue::size() {
  return queue_size_;
}

__host__ __device__ queue_desc_t* Queue::descriptor(uint64_t index) {
  auto queue_descs{queue_desc_proxy_.get()};
  return &queue_descs[index];
}

queue_element_t* Queue::elements(uint64_t index) {
  auto queue{queue_proxy_.get()};
  return queue[index];
}

}  // namespace rocshmem
