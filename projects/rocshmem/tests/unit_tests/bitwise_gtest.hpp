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

#ifndef ROCSHMEM_BITWISE_GTEST_HPP
#define ROCSHMEM_BITWISE_GTEST_HPP

#define HIP_ENABLE_PRINTF

#include "gtest/gtest.h"
#include "wf_size.hpp"

#include "../src/memory/hip_allocator.hpp"
#include "containers/matrix.hpp"
#include "containers/share_strategy.hpp"
#include "containers/strategies.hpp"

#include <hip/hip_runtime.h>
#include <cassert>

namespace rocshmem {

/*****************************************************************************
 ************************ WarpMatrix Type Helpers ****************************
 *****************************************************************************/
typedef Matrix<uint64_t> WarpMatrix;

/*****************************************************************************
 ***************************** Device Methods ********************************
 *****************************************************************************/
class BitwiseDeviceMethods
{
  public:
    /*************************************************************************
     ************************* Block Strategy Methods ************************
     *************************************************************************/
    __device__
    void
    lowest_active_lane(WarpMatrix *warp_matrix,
                       size_t lanes_bitfield)
    {
        Block block {};
        if (activate_lane_helper(lanes_bitfield)) {
            auto low_lane = block.lowest_active_lane();
            size_t warp_index = hipThreadIdx_x / WF_SIZE;
            size_t block_index = hipBlockIdx_x;
            auto *elem = warp_matrix->access(warp_index, block_index);
            *elem = low_lane;
        }
    }

    __device__
    void
    is_lowest_active_lane(WarpMatrix *warp_matrix,
                          size_t lanes_bitfield)
    {
        Block block {};
        if (activate_lane_helper(lanes_bitfield)) {
            if (block.is_lowest_active_lane()) {
                size_t warp_index = hipThreadIdx_x / WF_SIZE;
                size_t block_index = hipBlockIdx_x;
                auto *elem = warp_matrix->access(warp_index, block_index);
                *elem = block.lane_id();
            }
        }
    }

    __device__
    void
    active_logical_lane_id_2(WarpMatrix *warp_matrix,
                             size_t lanes_bitfield)
    {
        Block block {};
        if (activate_lane_helper(lanes_bitfield)) {
            if (block.active_logical_lane_id() == 2) {
                size_t warp_index = hipThreadIdx_x / WF_SIZE;
                size_t block_index = hipBlockIdx_x;
                auto *elem = warp_matrix->access(warp_index, block_index);
                *elem = block.lane_id();
            }
        }
    }

    __device__
    void
    lane_id(WarpMatrix *warp_matrix,
            size_t lanes_bitfield)
    {
        Block block {};
        if (activate_lane_helper(lanes_bitfield)) {
            auto lane_id = block.lane_id();
            size_t warp_index = hipThreadIdx_x / WF_SIZE;
            size_t block_index = hipBlockIdx_x;
            auto *elem = warp_matrix->access(warp_index, block_index);
            *elem = lane_id;
        }
    }

    __device__
    void
    number_active_lanes(WarpMatrix *warp_matrix,
                        size_t lanes_bitfield)
    {
        Block block {};
        if (activate_lane_helper(lanes_bitfield)) {
            auto number_active_lanes = block.number_active_lanes();
            size_t warp_index = hipThreadIdx_x / WF_SIZE;
            size_t block_index = hipBlockIdx_x;
            auto *elem = warp_matrix->access(warp_index, block_index);
            *elem = number_active_lanes;
        }
    }

    __device__
    void
    broadcast_up_value_42(WarpMatrix *warp_matrix,
                          size_t lanes_bitfield)
    {
        Block block {};
        if (activate_lane_helper(lanes_bitfield)) {
            uint64_t value = 1;
            if (block.is_lowest_active_lane()) {
                value = 42;
            }
            value = block.broadcast_up(value);
            size_t warp_index = hipThreadIdx_x / WF_SIZE;
            size_t block_index = hipBlockIdx_x;
            auto *elem = warp_matrix->access(warp_index, block_index);
            *elem = value;
        }
    }

    __device__
    void
    fetch_incr_lowest_active_lane(WarpMatrix *warp_matrix,
                                  size_t lanes_bitfield)
    {
        Block block {};
        if (activate_lane_helper(lanes_bitfield)) {
            auto orig = block.fetch_incr(_fetch_value);
            if (block.is_lowest_active_lane()) {
                size_t warp_index = hipThreadIdx_x / WF_SIZE;
                size_t block_index = hipBlockIdx_x;
                auto *elem = warp_matrix->access(warp_index, block_index);
                *elem = orig;
            }
        }
    }

    __device__
    void
    fetch_incr_active_logical_lane_1(WarpMatrix *warp_matrix,
                                     size_t lanes_bitfield)
    {
        Block block {};
        if (activate_lane_helper(lanes_bitfield)) {
            auto orig = block.fetch_incr(_fetch_value);
            if (block.active_logical_lane_id() == 1) {
                size_t warp_index = hipThreadIdx_x / WF_SIZE;
                size_t block_index = hipBlockIdx_x;
                auto *elem = warp_matrix->access(warp_index, block_index);
                *elem = orig;
            }
        }
    }

    /*************************************************************************
     ************************* Helper Methods ********************************
     *************************************************************************/
    __device__
    bool
    activate_lane_helper(uint64_t lanes_bitfield)
    {
        /*
         * In the following example, assume the following values:
         *     hipThreadIdx_x := 66
         *     _warp_size := 64.
         *
         *     index (tens):    0 0 0 0 0 0 0 ... 6 6 . .
         *           (ones):    0 1 2 3 4 5 6 ... 2 3 . .
         *     lanes_bitfield: [1 0 1 0 1 0 1 ... 1 0 . .]
         *
         * Example:
         * warp_bit_id := hipThreadIdx_x % _warp_size;
         * warp_bit_id := 66 % 64
         * warp_bit_id := 2
         */
        uint64_t warp_bit_id = hipThreadIdx_x % WF_SIZE;

        /*
         * Example (continued):
         * warp_bitmask := 1 << 2
         * index (tens):  0 0 0 0 0 0 0 ... 6 6 . .
         *       (ones):  0 1 2 3 4 5 6 ... 2 3 . .
         * warp_bitmask: [0 0 1 0 0 0 0 ... 0 0 . .]
         */
        uint64_t my_warp_bitmask_id = 1UL << warp_bit_id;

        /*
         * Example (continued):
         * index (tens):    0 0 0 0 0 0 0 ... 6 6 . .
         *       (ones):    0 1 2 3 4 5 6 ... 2 3 . .
         * lanes_bitfield: [1 0 1 0 1 0 1 ... 1 0 . .]
         * warp_bitmask:   [0 0 1 0 0 0 0 ... 0 0 . .]
         */
        bool is_an_active_lane = lanes_bitfield & my_warp_bitmask_id;

        return is_an_active_lane;
    }

    long long unsigned *_fetch_value = nullptr;
};

/*****************************************************************************
 ***************************** Test Fixture **********************************
 *****************************************************************************/
class BitwiseTestFixture : public ::testing::Test
{
  public:
    BitwiseTestFixture() = default;

    ~BitwiseTestFixture()
    {
        if (_device_methods) {
            if (_device_methods->_fetch_value) {
                _hip_allocator.deallocate(_device_methods->_fetch_value);
            }
            _hip_allocator.deallocate(_device_methods);
        }
        if (_warp_matrix) {
            _hip_allocator.deallocate(_warp_matrix);
        }
    }

    /*************************************************************************
     **************************** Setup Methods ******************************
     *************************************************************************/
    void
    setup_fixture(dim3 block_dim, dim3 grid_dim)
    {
        _hip_block_dim = block_dim;
        _hip_grid_dim = grid_dim;
        _wf_size = get_wf_size();

        assert(_device_methods == nullptr);
        _hip_allocator.allocate(reinterpret_cast<void**>(&_device_methods),
                                sizeof(BitwiseDeviceMethods));

        assert(_device_methods);

        _hip_allocator.allocate(reinterpret_cast<void**>(&_device_methods->_fetch_value),
                                sizeof(long long unsigned));

        assert(_device_methods->_fetch_value);

        *_device_methods->_fetch_value = 0;

        assert(_warp_matrix == nullptr);
        _hip_allocator.allocate(reinterpret_cast<void**>(&_warp_matrix),
                                sizeof(WarpMatrix));

        size_t warps_per_block = ceil(float(_hip_block_dim.x) / WF_SIZE);

        const ObjectStrategy *default_object_strategy =
                DefaultObjectStrategy::instance()->get();

        assert(_warp_matrix);
        new (_warp_matrix) WarpMatrix(warps_per_block,
                                      _hip_grid_dim.x,
                                      _hip_allocator,
                                      *default_object_strategy);
    }

    void
    zero_warp_matrix()
    {
        for (size_t row = 0; row < _warp_matrix->rows(); row++) {
            for (size_t col = 0; col < _warp_matrix->columns(); col++) {
                auto *entry = _warp_matrix->access(row, col);
                *entry = 0;
            }
        }
    }

    void
    verify_zeroed_warp_matrix()
    {
        for (size_t row = 0; row < _warp_matrix->rows(); row++) {
            for (size_t col = 0; col < _warp_matrix->columns(); col++) {
                auto *entry = _warp_matrix->access(row, col);
                ASSERT_EQ(*entry, 0);
            }
        }
    }

    /*************************************************************************
     *********************** Kernel Launch Methods ***************************
     *************************************************************************/
    void
    host_run_device_kernel(void(*fn)(BitwiseDeviceMethods*,
                                     WarpMatrix*,
                                     size_t),
                           size_t activate_lanes_bitfield)
    {
        hipLaunchKernelGGL(fn,
                           _hip_grid_dim,
                           _hip_block_dim,
                           0,
                           nullptr,
                           _device_methods,
                           _warp_matrix,
                           activate_lanes_bitfield);

        CHECK_HIP(hipStreamSynchronize(nullptr));
    }

  protected:
    /*************************************************************************
     ********************** Implementation Variables *************************
     *************************************************************************/
    dim3 _hip_block_dim {};
    dim3 _hip_grid_dim {};
    HIPAllocator _hip_allocator {};
    WarpMatrix *_warp_matrix = nullptr;
    BitwiseDeviceMethods *_device_methods = nullptr;
    int _wf_size;
};

} // namespace rocshmem

#endif  // ROCSHMEM_BITWISE_GTEST_HPP
