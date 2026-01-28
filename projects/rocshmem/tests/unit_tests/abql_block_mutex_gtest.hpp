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

#ifndef ROCSHMEM_ABQL_BLOCK_MUTEX_GTEST_HPP
#define ROCSHMEM_ABQL_BLOCK_MUTEX_GTEST_HPP

#include "gtest/gtest.h"

#include "../src/memory/hip_allocator.hpp"
#include "../src/sync/abql_block_mutex.hpp"
#include "../src/util.hpp"

#include <hip/hip_runtime.h>
#include <cassert>

namespace rocshmem {

inline __device__
void
increment_counter(ABQLBlockMutex *mutex,
                  size_t *counter) {
    auto ticket {mutex->lock()};
    (*counter)++;
    __threadfence();
    mutex->unlock(ticket);
}

__global__
void
all_threads_once(ABQLBlockMutex *mutex,
                 size_t *counter) {
    increment_counter(mutex, counter);
}

__global__
void
block_leader_once(ABQLBlockMutex *mutex,
                  size_t *counter) {
    if (is_thread_zero_in_block()) {
        increment_counter(mutex, counter);
    }
}

__global__
void
warp_leader_once(ABQLBlockMutex *mutex,
                 size_t *counter) {
    if (is_thread_zero_in_wave()) {
        increment_counter(mutex, counter);
    }
}

class ABQLBlockMutexTestFixture : public ::testing::Test {
  public:
    ABQLBlockMutexTestFixture() {
        assert(mutex_ == nullptr);
        hip_allocator_.allocate((void**)&mutex_, sizeof(ABQLBlockMutex));

        assert(mutex_);
        new (mutex_) ABQLBlockMutex();

        assert(counter_ == nullptr);
        hip_allocator_.allocate((void**)&counter_, sizeof(int));

        assert(counter_);
        *counter_ = 0;
    }

    ~ABQLBlockMutexTestFixture() {
        if (mutex_) {
            hip_allocator_.deallocate(mutex_);
        }

        if (counter_) {
            hip_allocator_.deallocate(counter_);
        }
    }

    void
    run_all_threads_once(uint32_t x_block_dim,
                         uint32_t x_grid_dim) {
        const dim3 hip_blocksize(x_block_dim, 1, 1);
        const dim3 hip_gridsize(x_grid_dim, 1, 1);

        hipLaunchKernelGGL(all_threads_once,
                           hip_gridsize,
                           hip_blocksize,
                           0,
                           nullptr,
                           mutex_,
                           counter_);

        CHECK_HIP(hipStreamSynchronize(nullptr));

        size_t number_threads {x_block_dim * x_grid_dim};

        ASSERT_EQ(*counter_, number_threads);
    }

  protected:
    /**
     * @brief An allocator to create objects in device memory.
     */
    HIPAllocator hip_allocator_ {};

    /**
     * @brief A mutex to prevent data races.
     */
    ABQLBlockMutex *mutex_ {nullptr};

    /**
     * @brief A monotonically increasing counter to track accesses.
     */
    size_t *counter_ {nullptr};
};


} // namespace rocshmem

#endif  // ROCSHMEM_ABQL_BLOCK_MUTEX_GTEST_HPP
