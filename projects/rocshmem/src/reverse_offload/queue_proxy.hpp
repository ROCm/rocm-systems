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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_PROXY_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_PROXY_HPP_

#include "atomic_return.hpp"
#include "device_proxy.hpp"
#include "hdp_policy.hpp"
#include "ipc_policy.hpp"
#include "commands_types.hpp"
#include "profiler.hpp"
#include "sync/abql_block_mutex.hpp"

namespace rocshmem {

struct cacheline_t {
  volatile char valid;
  volatile char padding[63];
} __attribute__((__aligned__(64)));

typedef struct queue_element {
  /**
   * Polled by the CPU to determine when a command is ready. Set by the GPU
   * once a queue element has been completely filled out. This is padded
   * from the actual data to prevent thrashing on an APU when the GPU is
   * trying to fill out a packet and the CPU is reading the valid bit.
   */
  cacheline_t notify_cpu;

  /**
   * All following fields written by the GPU and read by the CPU.
   */
  ro_net_cmds type{};
  int PE{-1};
  void *src{nullptr};
  void *dst{nullptr};
  int ro_net_win_id{-1};
  int logPE_stride{-1};
  int PE_size{-1};
  long *pSync{nullptr};
  int op{-1};
  int datatype{-1};
  int PE_root{-1};
  volatile char *status{nullptr};
  intptr_t team_comm{};
  union {
    size_t size;
    unsigned long long atomic_value;
  } ol1;
  union {
    void *pWrk;
    unsigned long long atomic_cond;
  } ol2;
} __attribute__((__aligned__(64))) queue_element_t;

template <typename ALLOCATOR>
class QueueElementProxy {
  using ProxyT = DeviceProxy<ALLOCATOR, queue_element_t>;

 public:
  QueueElementProxy(size_t num_elems = 1) : proxy_{num_elems} {
    new (proxy_.get()) queue_element_t();
  }

  QueueElementProxy(const QueueElementProxy& other) = delete;

  QueueElementProxy& operator=(const QueueElementProxy& other) = delete;

  QueueElementProxy(QueueElementProxy&& other) = default;

  QueueElementProxy& operator=(QueueElementProxy&& other) = default;

  ~QueueElementProxy() { proxy_.get()->~queue_element_t(); }

  __host__ __device__ queue_element_t* get() { return proxy_.get(); }

 private:
  ProxyT proxy_{};
};

using QueueElementProxyT = QueueElementProxy<PosixAligned64Allocator>;

template <typename ALLOCATOR>
class QueueProxy {
  using ProxyT = DeviceProxy<ALLOCATOR, queue_element_t *>;
  using ProxyPerBlockT = DeviceProxy<ALLOCATOR, queue_element_t>;

 public:
  /**
   * @brief Initializes a c-style array of circular queues.
   *
   * The circular queues are indexed using the device block-id so that each
   * each block has its own queue.
   */
  QueueProxy() = default;

  QueueProxy(size_t max_queues, size_t queue_size)
    : max_queues_{max_queues}, queue_size_{queue_size},
      total_queue_elements_{queue_size * max_queues},
      queue_proxy_{max_queues},
      per_block_queue_proxy_{queue_size * max_queues} {

    auto **queue_array{queue_proxy_.get()};
    auto *per_block_queue{per_block_queue_proxy_.get()};
    for (size_t i{0}; i < max_queues_; i++) {
      queue_array[i] = per_block_queue + i * queue_size;
    }
    size_t total_queue_element_bytes{sizeof(queue_element_t) *
                                     total_queue_elements_};
    memset(per_block_queue, 0, total_queue_element_bytes);
  }

  QueueProxy(const QueueProxy& other) = delete;

  QueueProxy& operator=(const QueueProxy& other) = delete;

  QueueProxy(QueueProxy&& other) = default;

  QueueProxy& operator=(QueueProxy&& other) = default;

  __host__ __device__ queue_element_t **get() { return queue_proxy_.get(); }

 private:
  ProxyT queue_proxy_{};

  ProxyPerBlockT per_block_queue_proxy_{};

  size_t max_queues_{};

  size_t queue_size_{};

  size_t total_queue_elements_{};
};

using QueueProxyT = QueueProxy<HIPHostAllocator>;

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_PROXY_HPP_
