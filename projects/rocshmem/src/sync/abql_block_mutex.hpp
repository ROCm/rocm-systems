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

#ifndef LIBRARY_SRC_SYNC_ABQL_BLOCK_MUTEX_HPP_
#define LIBRARY_SRC_SYNC_ABQL_BLOCK_MUTEX_HPP_

#include "device_proxy.hpp"

#include <hip/hip_runtime.h>

namespace rocshmem {

class ABQLBlockMutex {
  using TicketT = uint64_t;

  /**
   * @brief Supports access on a per channel basis.
   *
   * Readers spin on the field on different memory channels to
   * avoid hotspots on a single memory channel.
   *
   * Writers update the turn upon releasing their lock.
   */
  struct Turn {
    volatile TicketT vol_{0};
    int8_t padding[248];
  };

  static_assert(sizeof(Turn) == 256);

 public:
  /**
   * @brief Primary constructor
   */
  ABQLBlockMutex() = default;

  /**
   * @brief locks the device mutex
   *
   * @return ticket acquired during lock
   */
  __device__ TicketT lock();

  /**
   * @brief unlocks the device mutex
   *
   * @param ticket needed to signal next user
   *
   * @return void
   */
  __device__ void unlock(TicketT ticket);

  /**
   * @brief grab the next ticket
   *
   * @return ticket
   */
  __device__ TicketT grab_ticket_();

  __device__ bool is_turn_(TicketT ticket);

  /**
   * @brief wait for my turn
   *
   * @param ticket value to wait for
   */
  __device__ void wait_turn_(TicketT ticket);

  /**
   * @brief signal turn for next ticket holder
   *
   * @param ticket value of next turn
   */
  __device__ void signal_next_(TicketT ticket);

 private:
  /**
   * @brief Holds the current ticket count
   */
  TicketT ticket_{0};

  /**
   * @brief Specifies how many slots in turns_.
   */
  static constexpr unsigned memory_channels_{32};

  /**
   * @brief Holds the current turn
   */
  Turn turns_[memory_channels_]{};
};

template <typename ALLOCATOR>
class ABQLBlockMutexProxy {
  using ProxyT = DeviceProxy<ALLOCATOR, ABQLBlockMutex>;

 public:
  ABQLBlockMutexProxy(size_t num_elems = 1) : proxy_{num_elems} {}

  ABQLBlockMutexProxy(const ABQLBlockMutexProxy& other) = delete;

  ABQLBlockMutexProxy& operator=(const ABQLBlockMutexProxy& other) = delete;

  ABQLBlockMutexProxy(ABQLBlockMutexProxy&& other) = default;

  ABQLBlockMutexProxy& operator=(ABQLBlockMutexProxy&& other) = default;

  __host__ __device__ ABQLBlockMutex* get() { return proxy_.get(); }

 private:
  ProxyT proxy_{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_SYNC_ABQL_BLOCK_MUTEX_HPP_
