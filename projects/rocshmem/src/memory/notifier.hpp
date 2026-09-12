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
#include "hip_allocator.hpp"

namespace rocshmem {

namespace atomic = detail::atomic;

template<atomic::memory_scope scope>
class Notifier {

 public:
  __device__ uint64_t load() {
    return atomic::load<scope, atomic::memory_order::acquire>(&value_);
  }

  __device__ void store(uint64_t val) {
    atomic::store<scope, atomic::memory_order::release>(&value_, val);
  }

  __device__ void fence() {
    atomic::threadfence<scope>();
  }

  __device__ void sync() {
    if constexpr (scope == atomic::memory_scope::single ||
                  scope == atomic::memory_scope::wavefront) {
      return;
    }
    if constexpr (scope == atomic::memory_scope::workgroup) {
      __syncthreads();
      return;
    }
    if constexpr (scope == atomic::memory_scope::system) {
      static_assert(false);
      return;
    }

    uint32_t done {signal_ + 1};
    __syncthreads();

    uint32_t retval {0};
    bool executor {!threadIdx.x && !threadIdx.y && !threadIdx.z};
    if (executor) {
      retval = atomic::fetch_add<scope, atomic::memory_order::acq_rel>(&count_, 1);
      fence();
    }
    __syncthreads();

    if (retval == ((gridDim.x * gridDim.y * gridDim.z) - 1)) {
      if (executor) {
        atomic::store<scope, atomic::memory_order::release>(&count_, 0);
        fence();
        atomic::fetch_add<scope, atomic::memory_order::acq_rel>(&signal_, 1);
      }
    }

    if (executor) {
      while (atomic::load<scope, atomic::memory_order::acquire>(
               &signal_) != done) {
        ;
      }
    }
    __syncthreads();
  }

 private:
  uint64_t value_{};

  uint32_t signal_ {};

  uint32_t count_ {};
};

template <atomic::memory_scope scope>
class NotifierProxy {
  using ProxyT = DeviceProxy<Notifier<scope>>;

 public:
  NotifierProxy(const HIPAllocator& alloc = HIPAllocator(),
                size_t num_elems = 1)
    : alloc_{alloc}, proxy_{num_elems, alloc_} {
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
  HIPAllocator alloc_{};
  ProxyT proxy_{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_NOTIFIER_HPP_
