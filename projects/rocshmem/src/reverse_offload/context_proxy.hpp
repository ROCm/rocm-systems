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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_CONTEXT_PROXY_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_CONTEXT_PROXY_HPP_

#include "rocshmem/rocshmem.hpp"
#include "device_proxy.hpp"
#include "memory/hip_allocator.hpp"
#include "context_ro_device.hpp"

namespace rocshmem {

class ROBackend;

template <typename ALLOCATOR>
class DefaultContextProxy {
  using ProxyT = DeviceProxy<ALLOCATOR, ROContext>;

 public:
  DefaultContextProxy() = default;

  /*
   * Placement new the memory which is allocated by proxy_
   */
  explicit DefaultContextProxy(ROBackend* backend, TeamInfo *tinfo,
                               size_t num_elems = 1)
  : constructed_{true}, proxy_{num_elems} {
    auto ctx{proxy_.get()};
    new (ctx) ROContext(reinterpret_cast<Backend*>(backend), -1, true);
    rocshmem_ctx_t local{ctx, tinfo};
    set_internal_ctx(&local);
  }

  /*
   * Since placement new is called in the constructor, then
   * delete must be called manually.
   */
  ~DefaultContextProxy() {
    if (constructed_) {
      proxy_.get()->~ROContext();
    }
  }

  DefaultContextProxy(const DefaultContextProxy& other) = delete;

  DefaultContextProxy& operator=(const DefaultContextProxy& other) = delete;

  DefaultContextProxy(DefaultContextProxy&& other) = default;

  DefaultContextProxy& operator=(DefaultContextProxy&& other) = default;

  /*
   * @brief Provide access to the memory referenced by the proxy
   */
  __host__ __device__ Context* get() { return proxy_.get(); }

 private:
  /*
   * @brief Memory managed by the lifetime of this object
   */
  ProxyT proxy_{};

  /*
   * @brief denotes if an objects was constructed in proxy
   */
  bool constructed_{false};
};

using DefaultContextProxyT = DefaultContextProxy<HIPAllocator>;

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_CONTEXT_PROXY_HPP_
