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

#ifndef LIBRARY_SRC_FENCE_POLICY_HPP_
#define LIBRARY_SRC_FENCE_POLICY_HPP_

#include "rocshmem/rocshmem.hpp"

namespace rocshmem {

/**
 * @brief Controls the behavior of device code which may need to stall
 */
class Fence {
 public:
  /**
   * Secondary constructor
   */
  __host__ __device__ Fence() = default;

  /**
   * Primary constructor
   *
   * @param[in] options interpreted as a bitfield using bitwise operations
   */
  __host__ __device__ Fence(long option) {
    if (option & ROCSHMEM_CTX_NOSTORE) {
      flush_ = false;
    }
  }

  /**
   * @brief Wait for outstanding memory operations to complete
   *
   * This can be useful when code needs guarantees about visibility
   * before moving past the flush.
   *
   * @return void
   */
  __device__ void flush() {
    if (flush_) {
      __threadfence();
    }
  }

 private:
  /**
   * @brief Used to toggle flushes behavior on and off
   *
   * @note By default, flushing is enabled.
   */
  bool flush_{true};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_FENCE_POLICY_HPP_
