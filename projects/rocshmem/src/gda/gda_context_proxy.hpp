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

#ifndef LIBRARY_SRC_GDA_CONTEXT_PROXY_HPP_
#define LIBRARY_SRC_GDA_CONTEXT_PROXY_HPP_


#include "device_proxy.hpp"
#include "backend_gda.hpp"

namespace rocshmem {

class GDABackend;

template <typename ALLOCATOR>
class GDADefaultContextProxy {
  using ProxyT = DeviceProxy<ALLOCATOR, GDAContext>;

 public:
  GDADefaultContextProxy() = default;

  /*
   * Placement new the memory which is allocated by proxy_
   */
  explicit GDADefaultContextProxy(GDABackend* backend, TeamInfo *tinfo,
                                  int gda_provider,
                                  size_t num_elems = 1)
  : constructed_{true}, proxy_{num_elems} {
    auto ctx{proxy_.get()};
    new (ctx) GDAContext(reinterpret_cast<Backend*>(backend), 0, gda_provider);
    ctx->tinfo = tinfo;
    rocshmem_ctx_t local{ctx, tinfo};
    set_internal_ctx(&local);
  }

  /*
   * Since placement new is called in the constructor, then
   * delete must be called manually.
   */
  ~GDADefaultContextProxy() {
    if (constructed_) {
      proxy_.get()->~GDAContext();
    }
  }

  GDADefaultContextProxy(const GDADefaultContextProxy& other) = delete;

  GDADefaultContextProxy& operator=(const GDADefaultContextProxy& other) = delete;

  GDADefaultContextProxy(GDADefaultContextProxy&& other) = default;

  GDADefaultContextProxy& operator=(GDADefaultContextProxy&& other) {
    if (this != &other) {
      proxy_ = std::move(other.proxy_);
      constructed_ = true;
      other.constructed_ = false;
    }
    return *this;
  }

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

using GDADefaultContextProxyT = GDADefaultContextProxy<HIPAllocator>;

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_CONTEXT_PROXY_HPP_
