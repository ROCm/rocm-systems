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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_WINDOW_PROXY_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_WINDOW_PROXY_HPP_

#include "device_proxy.hpp"
#include "memory/window_info.hpp"
#include "mpi_transport.hpp"

namespace rocshmem {

template <typename ALLOCATOR>
class WindowProxy {
 private:
  using ProxyT = DeviceProxy<ALLOCATOR, WindowInfoMPI *>;

 public:
  /*
   * Placement new the memory which is allocated by proxy_
   */
  WindowProxy(SymmetricHeap *heap, MPI_Comm comm, size_t num_windows)
    : num_windows_{num_windows}, proxy_{num_windows} {

    WindowInfoMPI** window_info{proxy_.get()};

    for (size_t i{0}; i < num_windows_; i++) {
      window_info[i] =
          new WindowInfoMPI(comm, heap->get_local_heap_base(), heap->get_size());
    }
  }

  WindowProxy(const WindowProxy& other) = delete;

  WindowProxy& operator=(const WindowProxy& other) = delete;

  WindowProxy(WindowProxy&& other) = default;

  WindowProxy& operator=(WindowProxy&& other) = default;

  /*
   * Since placement new is called in the constructor, then
   * delete must be called manually.
   */
  ~WindowProxy() {
    auto *window_info{proxy_.get()};

    for (size_t i{0}; i < num_windows_; i++) {
      delete window_info[i];
    }
  }

  /*
   * @brief Provide access to the memory referenced by the proxy
   */
  __host__ __device__ WindowInfoMPI **get() { return proxy_.get(); }

  __host__ size_t get_num_MPI_windows() { return num_windows_; }
 private:
  /*
   * @brief Memory managed by the lifetime of this object
   */
  ProxyT proxy_{};

  /**
   * @brief Number of MPI windows used for device contexts in RO Backend
   */
  size_t num_windows_{32};
};

using WindowProxyT = WindowProxy<HostAllocator>;

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_WINDOW_PROXY_HPP_
