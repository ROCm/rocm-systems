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

#ifndef ROCSHMEM_INDEX_STRATEGY_GTEST_HPP
#define ROCSHMEM_INDEX_STRATEGY_GTEST_HPP

#include "gtest/gtest.h"

#include "../src/containers/index_strategy.hpp"
#include "../src/memory/hip_allocator.hpp"
#include "../src/util.hpp"

#include <hip/hip_runtime_api.h>
#include <cassert>

namespace rocshmem {

template <typename INDEX_STRAT>
__global__
void
memory_set(int *raw_mem, size_t num_elems)
{
    assert(raw_mem);

    INDEX_STRAT idx_strat(num_elems);

    for (size_t i = idx_strat.start();
         i < idx_strat.end();
         i = idx_strat.next(i)) {
        raw_mem[i]++;
    }
}

class IndexStrategyTestFixture : public ::testing::Test
{
  public:
    IndexStrategyTestFixture()
    {
        assert(_raw_mem == nullptr);
        size_t raw_mem_size_bytes = sizeof(int) * _mem_elements;
        _hip_allocator.allocate(reinterpret_cast<void**>(&_raw_mem),
                                raw_mem_size_bytes);

        assert(_raw_mem);
        for (size_t i = 0; i < _mem_elements; i++) {
            _raw_mem[i] = 0;
        }
    }

    ~IndexStrategyTestFixture()
    {
        if (_raw_mem) {
            _hip_allocator.deallocate(_raw_mem);
        }
    }

    template <typename INDEX_STRAT>
    void
    run_memory_set_test(const dim3 grid_dim, const dim3 block_dim)
    {
        hipLaunchKernelGGL(memory_set<INDEX_STRAT>,
                           grid_dim,
                           block_dim,
                           0,
                           nullptr,
                           _raw_mem,
                           _mem_elements);

        CHECK_HIP(hipStreamSynchronize(nullptr));

        for(size_t i = 0; i < _mem_elements; i++) {
            EXPECT_EQ(_raw_mem[i], 1);
        }
    }

  protected:
    int *_raw_mem = nullptr;
    HIPAllocator _hip_allocator {};
    static constexpr size_t _mem_elements = 262144;
};

} // namespace rocshmem

#endif  // ROCSHMEM_INDEX_STRATEGY_GTEST_HPP
