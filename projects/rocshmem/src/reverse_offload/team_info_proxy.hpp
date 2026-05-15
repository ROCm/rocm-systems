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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_TEAM_INFO_PROXY_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_TEAM_INFO_PROXY_HPP_

#include "device_proxy.hpp"
#include "team.hpp"

namespace rocshmem {

template <typename ALLOCATOR>
class TeamInfoProxy {
  using ProxyT = DeviceProxy<ALLOCATOR, TeamInfo>;

 public:
  /*
   * Placement new the memory which is allocated by proxy_
   */
  TeamInfoProxy(Team* parent_team, int pe_start, int stride, int size,
    size_t num_elems = 1) : proxy_{num_elems} {
    new (proxy_.get()) TeamInfo(parent_team, pe_start, stride, size);
  }

  TeamInfoProxy(const TeamInfoProxy& other) = delete;

  TeamInfoProxy& operator=(const TeamInfoProxy& other) = delete;

  TeamInfoProxy(TeamInfoProxy&& other) = default;

  TeamInfoProxy& operator=(TeamInfoProxy&& other) = default;

  /*
   * Since placement new is called in the constructor, then
   * delete must be called manually.
   */
  ~TeamInfoProxy() { proxy_.get()->~TeamInfo(); }

  /*
   * @brief Provide access to the memory referenced by the proxy
   */
  __host__ __device__ TeamInfo* get() { return proxy_.get(); }

 private:
  /*
   * @brief Memory managed by the lifetime of this object
   */
  ProxyT proxy_{};
};

using TeamInfoProxyT = TeamInfoProxy<HIPAllocator>;

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_TEAM_INFO_PROXY_HPP_
