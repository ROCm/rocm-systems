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

#ifndef ROCSHMEM_NOTIFIER_GTEST_HPP
#define ROCSHMEM_NOTIFIER_GTEST_HPP

#include "gtest/gtest.h"

#include "../src/memory/hip_allocator.hpp"
#include "../src/memory/notifier.hpp"
#include "../src/util.hpp"

#include <hip/hip_runtime.h>
#include <cassert>

namespace rocshmem {

/**
 * @brief The bit pattern written to memory by each thread.
 */
static const uint8_t THREAD_VALUE {0xF9};

/**
 * @brief The bit pattern written to memory by each thread.
 */
static const uint64_t NOTIFIER_OFFSET {0x100B00};

inline __device__
void
write_to_memory(uint8_t* raw_memory) {
    auto thread_idx {get_flat_id()};
    raw_memory[thread_idx] = THREAD_VALUE;
    __threadfence();
}

template <typename NotifierT>
__global__
void
all_threads_once(uint8_t* raw_memory,
                 NotifierT * notifier) {
    if (!get_flat_id()) {
      notifier->store(NOTIFIER_OFFSET);
      notifier->fence();
    }
    notifier->sync();
    uint64_t offset_u64 {notifier->load()};
    uint64_t raw_memory_u64 {reinterpret_cast<uint64_t>(raw_memory)};
    uint64_t address_u64 {raw_memory_u64 + offset_u64};
    uint8_t* address {reinterpret_cast<uint8_t*>(address_u64)};
    write_to_memory(address);
}

class NotifierBase : public ::testing::Test {
  public:
    NotifierBase() {
        assert(raw_memory_ == nullptr);
        hip_allocator_.allocate((void**)&raw_memory_, GIBIBYTE_);
        assert(raw_memory_);
    }

    ~NotifierBase() {
        if (raw_memory_) {
            hip_allocator_.deallocate(raw_memory_);
        }
    }

    void
    verify(size_t number_threads) {
        uint8_t* offset_addr {compute_offset_addr()};
        for (size_t i {0}; i < number_threads; i++) {
            ASSERT_EQ(offset_addr[i], THREAD_VALUE);
        }
    }

  protected:
    /**
     * @brief Helper function to reconstruct device calculation.
     */
    uint8_t*
    compute_offset_addr() {
        uint64_t raw_memory_u64 {reinterpret_cast<uint64_t>(raw_memory_)};
        uint64_t address_u64 {raw_memory_u64 + NOTIFIER_OFFSET};
        uint8_t* address {reinterpret_cast<uint8_t*>(address_u64)};
        return address;
    }

    /**
     * @brief An allocator to create objects in device memory.
     */
    HIPAllocator hip_allocator_ {};

    /**
     * @brief The size of the raw memory block below.
     */
    static const size_t GIBIBYTE_ {1 << 30};

    /**
     * @brief A block of memory used to hold individual writes from threads.
     */
    uint8_t *raw_memory_ {nullptr};

};

class NotifierBlockTestFixture : public NotifierBase {
    using NotifierT = Notifier<detail::atomic::memory_scope_workgroup>;
    using NotifierProxyT = NotifierProxy<HIPAllocator, detail::atomic::memory_scope_workgroup>;

  public:
    void
    run_all_threads_once(uint32_t x_block_dim,
                         uint32_t x_grid_dim) {
        new (notifier_.get()) NotifierT();
        const dim3 block(x_block_dim, 1, 1);
        const dim3 grid(x_grid_dim, 1, 1);
        all_threads_once<NotifierT><<<grid, block>>>(raw_memory_, notifier_.get());
        CHECK_HIP(hipStreamSynchronize(nullptr));
        verify(x_block_dim * x_grid_dim);
    }

    /**
     * @brief Used to broadcast base offset for writing.
     */
    NotifierProxyT notifier_ {};
};

class NotifierAgentTestFixture : public NotifierBase {
    using NotifierT = Notifier<detail::atomic::memory_scope_agent>;
    using NotifierProxyT = NotifierProxy<HIPAllocator, detail::atomic::memory_scope_agent>;

  public:
    void
    run_all_threads_once(uint32_t x_block_dim,
                         uint32_t x_grid_dim) {
        new (notifier_.get()) NotifierT();
        const dim3 block(x_block_dim, 1, 1);
        const dim3 grid(x_grid_dim, 1, 1);
        all_threads_once<NotifierT><<<grid, block>>>(raw_memory_, notifier_.get());
        CHECK_HIP(hipStreamSynchronize(nullptr));
        verify(x_block_dim * x_grid_dim);
    }

    /**
     * @brief Used to broadcast base offset for writing.
     */
    NotifierProxyT notifier_ {};
};

} // namespace rocshmem

#endif  // ROCSHMEM_NOTIFIER_GTEST_HPP
