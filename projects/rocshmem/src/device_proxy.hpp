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

#ifndef LIBRARY_SRC_DEVICE_PROXY_HPP_
#define LIBRARY_SRC_DEVICE_PROXY_HPP_

#include <hip/hip_runtime.h>
#include <memory>
#include <utility>
#include <cstring>
#include <cassert>

namespace rocshmem {

template <typename ALLOCATOR, typename T>
class DeviceProxy {
 public:
  DeviceProxy() = default;

  DeviceProxy(size_t num_elems) : num_elems_ {num_elems} {
    /**
     * @brief The allocation size of the internal memory
     *
    */
    size_t size_bytes = sizeof(T) * num_elems_;
    /*
     * Allocate memory and verify that the allocation worked.
     */
    T* temp{nullptr};
    allocator_.allocate(reinterpret_cast<void**>(&temp), size_bytes);
    assert(temp);

    /*
     * Default memory provided by the allocation to recognizable bytes.
     */
    memset(static_cast<void*>(temp), 0, size_bytes);

    /*
     * Pass the memory into a unique ptr for tracking.
     */
    std::unique_ptr<T, Deleter> up{temp};
    up_ = std::move(up);

    /*
     * Set a c-style ptr for access by the device.
     */
    ptr_ = up_.get();
  }

  DeviceProxy(const DeviceProxy& other) = delete;

  DeviceProxy& operator=(const DeviceProxy& other) = delete;

  DeviceProxy(DeviceProxy&& other) = default;

  DeviceProxy& operator=(DeviceProxy&& other) = default;

  /**
   * @brief Return internal storage tracked by the Proxy.
   *
   * @note Do not try to free this memory yourself. The proxy maintains
   * the lifetime of the data itself.
   */
  __host__ __device__ T* get() { return ptr_; }

 private:
  /**
   * @brief Internal Deleter functor is required by up_ member
   */
  class Deleter {
   public:
    void operator()(void* x) { a_.deallocate(x); }

   private:
    ALLOCATOR a_;
  };

  /**
   * @brief Externally provided allocator type.
   */
  ALLOCATOR allocator_{};

  /**
   * @brief Unique pointer for tracking the proxy.
   */
  std::unique_ptr<T, Deleter> up_{nullptr};

  /**
   * @brief A handle to access the internal memory.
   *
   * In general, device code cannot access standard library routines
   * like std::unique_ptr::get(). Circumvent this problem by caching
   * the pointer manually in this class.
   */
  T* ptr_{nullptr};

  /**
   * @brief Number of elements of type T to be allocated
   */
  size_t num_elems_{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_DEVICE_PROXY_HPP_
