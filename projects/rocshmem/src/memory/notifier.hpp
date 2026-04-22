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

#ifndef LIBRARY_SRC_MEMORY_NOTIFIER_HPP_
#define LIBRARY_SRC_MEMORY_NOTIFIER_HPP_

#include "device_proxy.hpp"
#include "util.hpp"
#include "atomic.hpp"

namespace rocshmem {

template<detail::atomic::rocshmem_memory_scope scope>
class Notifier {

 public:
  __device__ uint64_t load() {
    return detail::atomic::load<uint64_t, scope>(&value_, orders_);
  }

  __device__ void store(uint64_t val) {
    detail::atomic::store<uint64_t, scope>(&value_, val, orders_);
  }

  __device__ void fence() {
    detail::atomic::threadfence<scope>();
  }

  __device__ void sync() {
    if constexpr (scope == detail::atomic::memory_scope_thread ||
                  scope == detail::atomic::memory_scope_wavefront) {
      return;
    }
    if constexpr (scope == detail::atomic::memory_scope_workgroup) {
      __syncthreads();
      return;
    }
    if constexpr (scope == detail::atomic::memory_scope_system) {
      static_assert(false);
      return;
    }

    uint32_t done {signal_ + 1};
    __syncthreads();

    uint32_t retval {0};
    bool executor {!threadIdx.x && !threadIdx.y && !threadIdx.z};
    if (executor) {
      retval = detail::atomic::fetch_add<uint32_t, uint32_t, scope>(&count_, 1, orders_);
      fence();
    }
    __syncthreads();

    if (retval == ((gridDim.x * gridDim.y * gridDim.z) - 1)) {
      if (executor) {
        detail::atomic::store<uint32_t, scope>(&count_, 0, orders_);
        fence();
        detail::atomic::fetch_add<uint32_t, uint32_t, scope>(&signal_, 1, orders_);
      }
    }

    if (executor) {
      while (detail::atomic::load<uint32_t, scope>(&signal_, orders_) != done) {
        ;
      }
    }
    __syncthreads();
  }

 private:
  detail::atomic::rocshmem_memory_orders orders_{};

  uint64_t value_{};

  uint32_t signal_ {};

  uint32_t count_ {};
};

template <typename ALLOCATOR, detail::atomic::rocshmem_memory_scope scope>
class NotifierProxy {
  using ProxyT = DeviceProxy<ALLOCATOR, Notifier<scope>>;

 public:
  NotifierProxy(size_t num_elems = 1) : proxy_{num_elems} {
    new (proxy_.get()) Notifier<scope>();
  }

  NotifierProxy(const NotifierProxy& other) = delete;

  NotifierProxy& operator=(const NotifierProxy& other) = delete;

  NotifierProxy(NotifierProxy&& other) = default;

  NotifierProxy& operator=(NotifierProxy&& other) = default;

  ~NotifierProxy() {
    proxy_.get()->~Notifier<scope>();
  }

  __host__ __device__ Notifier<scope>* get() { return proxy_.get(); }

 private:
  ProxyT proxy_{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_NOTIFIER_HPP_
