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

#ifndef LIBRARY_SRC_CONTAINERS_SHARE_STRATEGY_HPP_
#define LIBRARY_SRC_CONTAINERS_SHARE_STRATEGY_HPP_

#include <hip/hip_runtime.h>

#include "index_strategy.hpp"

namespace rocshmem {

class Global {
 public:
  /**
   * @brief
   *
   * @param out_ptr
   *
   * @return
   */
  template <typename T>
  __host__ T fetch_incr(T *out_ptr) {}
};

class Grid {
 public:
  /**
   * @brief
   *
   * @param out_ptr
   *
   * @return
   */
  template <typename T>
  __device__ T fetch_incr(T *out_ptr) {}
};

class Block {
 private:
  friend class BitwiseDeviceMethods;

 public:
  /**
   * @brief
   *
   * @param out_ptr
   *
   * @return
   */
  template <typename T>
  __device__ T fetch_incr(T *out_ptr) {
    T fetch_value = 0;

    auto num_active_lanes = number_active_lanes();

    if (is_lowest_active_lane()) {
      fetch_value = atomicAdd(out_ptr, num_active_lanes);
    }

    fetch_value = broadcast_up(fetch_value);

    return fetch_value + active_logical_lane_id();
  }

 private:
  /**
   * @brief
   *
   * @return
   */
  __device__ uint64_t lane_id();

  /**
   * @brief
   *
   * @return
   */
  __device__ uint64_t number_active_lanes();

  /**
   * @brief
   *
   * @return
   */
  __device__ bool is_lowest_active_lane();

  /**
   * @brief
   *
   * @return
   */
  __device__ uint64_t lowest_active_lane();

  /**
   * @brief
   *
   * @return
   */
  __device__ uint64_t active_logical_lane_id();

  /**
   * @brief
   *
   * @return
   */
  __device__ uint64_t broadcast_up(uint64_t fetch_value);
};

class Private {
 public:
  /**
   * @brief
   *
   * @param out_ptr
   *
   * @return
   */
  template <typename T>
  __device__ T fetch_incr(T *out_ptr) {
    auto orig_value = *out_ptr;
    *out_ptr = orig_value + 1;
    return orig_value;
  }
};

enum class ShareStrategyEnum { PRIVATE = 0, BLOCK = 1, DEVICE = 2, UNUSED = 3 };

class ShareStrategy {
 public:
  __host__ __device__ ShareStrategy() = default;

  __host__ __device__ ShareStrategy(ShareStrategyEnum sse) : _sse(sse) {}

  /**
   * @brief
   *
   * @return
   */
  __device__ void syncthreads() const;

  /**
   * @brief
   *
   * @return
   */
  template <typename T>
  __device__ T fetch_incr(T *out_ptr, size_t my_pe, size_t num_pes) {
    T value = 0;
    switch (_sse) {
      case ShareStrategyEnum::PRIVATE:
        return _private.fetch_incr(out_ptr);
      case ShareStrategyEnum::BLOCK:
        for (size_t i = 0; i < num_pes; i++) {
          if (i == my_pe) {
            value = _block.fetch_incr(out_ptr);
          }
        }
        return value;
      case ShareStrategyEnum::DEVICE:
        abort();
        return 0;
      case ShareStrategyEnum::UNUSED:
        abort();
        return 0;
    }
  }

  /**
   * @brief
   *
   * @return
   */
  __device__ bool uses_designated_send_thread() {
    switch (_sse) {
      case ShareStrategyEnum::PRIVATE:
        return false;
      case ShareStrategyEnum::BLOCK:
        return true;
      case ShareStrategyEnum::DEVICE:
        abort();
        return false;
      case ShareStrategyEnum::UNUSED:
        abort();
        return false;
    }
  }

 private:
  Private _private{};
  Block _block{};
  ShareStrategyEnum _sse{ShareStrategyEnum::UNUSED};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTAINERS_SHARE_STRATEGY_HPP_
