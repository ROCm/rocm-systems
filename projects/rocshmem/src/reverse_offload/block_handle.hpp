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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_BLOCK_HANDLE_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_BLOCK_HANDLE_HPP_

#include "containers/atomic_wf_queue_impl.hpp"
#include "hdp_policy.hpp"
#include "ipc_policy.hpp"
#include "profiler.hpp"
#include "queue.hpp"

namespace rocshmem {

using AWF_Queue_statusT = AtomicWFQueue<volatile char*, HIPAllocator>;
using AWF_Queue_ret_buffT = AtomicWFQueue<uint64_t*, HIPAllocator>;

struct BlockHandle {
  ROStats profiler{};
  queue_element_t *queue{nullptr};
  uint64_t queue_size{};
  volatile uint64_t read_index{};
  volatile uint64_t write_index{};
  volatile uint64_t *host_read_index{};
  volatile char *status{nullptr};
  void *g_ret{nullptr};
  void *atomic_ret{nullptr};
  volatile uint64_t lock{};
  AWF_Queue_statusT *default_ctx_status{nullptr};
  AWF_Queue_ret_buffT *default_ctx_g_ret{nullptr};
  AWF_Queue_ret_buffT *default_ctx_atomic_ret{nullptr};
};

template <typename ALLOCATOR>
class DefaultBlockHandleProxy {
  using ProxyT = DeviceProxy<ALLOCATOR, BlockHandle>;

 public:
  DefaultBlockHandleProxy() = default;

  DefaultBlockHandleProxy(void *g_ret, void *atomic_ret, Queue *queue,
                          volatile char *status,
                          AWF_Queue_statusT *default_ctx_status,
                          AWF_Queue_ret_buffT *default_ctx_g_ret,
                          AWF_Queue_ret_buffT *default_ctx_atomic_ret,
                          size_t num_elems = 1)
    : proxy_{num_elems} {

    // TODO(bpotter): create a default queue for this queue descriptor
    auto queue_descriptor{queue->descriptor(0)};
    auto block_handle{proxy_.get()};
    block_handle->profiler.resetStats();
    block_handle->queue = queue->elements(0);
    block_handle->queue_size = queue->size();
    block_handle->read_index = queue_descriptor->read_index;
    block_handle->write_index = queue_descriptor->write_index;
    block_handle->host_read_index = &queue_descriptor->read_index;
    block_handle->status = status;
    block_handle->g_ret = g_ret;
    block_handle->atomic_ret = atomic_ret;
    block_handle->lock = 0;
    block_handle->default_ctx_status = default_ctx_status;
    block_handle->default_ctx_g_ret = default_ctx_g_ret;
    block_handle->default_ctx_atomic_ret = default_ctx_atomic_ret;
  }

  DefaultBlockHandleProxy(const DefaultBlockHandleProxy& other) = delete;

  DefaultBlockHandleProxy& operator=(const DefaultBlockHandleProxy& other) = delete;

  DefaultBlockHandleProxy(DefaultBlockHandleProxy&& other) = default;

  DefaultBlockHandleProxy& operator=(DefaultBlockHandleProxy&& other) = default;

  __host__ __device__ BlockHandle *get() { return proxy_.get(); }

 private:
  ProxyT proxy_{};
};

using DefaultBlockHandleProxyT = DefaultBlockHandleProxy<HIPDefaultFinegrainedAllocator>;

template <typename ALLOCATOR>
class BlockHandleProxy {
  using ProxyT = DeviceProxy<ALLOCATOR, BlockHandle>;

 public:
  BlockHandleProxy() = default;

  BlockHandleProxy(void *g_ret, void *atomic_ret, Queue *queue, size_t offset,
                   volatile char *status, size_t max_blocks)
    : proxy_{max_blocks} {

    for (size_t i{0}; i < max_blocks; i++) {
      auto queue_descriptor{queue->descriptor(i)};
      auto block_handle{&proxy_.get()[i]};
      size_t block_offset{i * offset};
      block_handle->profiler.resetStats();
      block_handle->queue = queue->elements(i);
      block_handle->queue_size = queue->size();
      block_handle->read_index = queue_descriptor->read_index;
      block_handle->write_index = queue_descriptor->write_index;
      block_handle->host_read_index = &queue_descriptor->read_index;
      block_handle->status = status + block_offset;
      block_handle->g_ret = reinterpret_cast<uint64_t*>(g_ret) + block_offset;
      block_handle->atomic_ret = reinterpret_cast<uint64_t*>(atomic_ret) +
                                 block_offset;
      block_handle->lock = 0;
    }
  }

  BlockHandleProxy(const BlockHandleProxy& other) = delete;

  BlockHandleProxy& operator=(const BlockHandleProxy& other) = delete;

  BlockHandleProxy(BlockHandleProxy&& other) = default;

  BlockHandleProxy& operator=(BlockHandleProxy&& other) = default;

  __host__ __device__ BlockHandle *get() { return proxy_.get(); }

 private:
  ProxyT proxy_{};

  size_t num_blocks_{};
};

using BlockHandleProxyT = BlockHandleProxy<HIPDefaultFinegrainedAllocator>;

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_BLOCK_HANDLE_HPP_
