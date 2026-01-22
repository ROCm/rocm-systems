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

#include "ipc_impl_simple_coarse_gtest.hpp"

using namespace rocshmem;


TEST_P(DegenerateSimpleCoarse, ptr_check) {
    ASSERT_NE(heap_mem_.get_ptr(), nullptr);
}

TEST_P(DegenerateSimpleCoarse, MPI_num_pes) {
    ASSERT_EQ(mpi_->num_pes(), 2);
}

TEST_P(DegenerateSimpleCoarse, IPC_bases) {
    ASSERT_EQ(mpi_->num_pes(), 2);
    ASSERT_NE(ipc_impl_.ipc_bases, nullptr);
    for(int i{0}; i < mpi_->num_pes(); i++) {
        ASSERT_NE(ipc_impl_.ipc_bases[i], nullptr);
    }
}

TEST_P(DegenerateSimpleCoarse, golden_1048576_int) {
    iota_golden(1048576);
    validate_golden(1048576);
}

INSTANTIATE_TEST_SUITE_P(
    IPCImplSimpleCoarseTestFixture,
    DegenerateSimpleCoarse,
    ::testing::Values(
        std::make_tuple(1,  1,   1))
);

//=============================================================================

TEST_P(ParameterizedBlockSimpleCoarse, write) {
    dim3 grid = dim3(std::get<0>(GetParam()), 1, 1);
    dim3 block = dim3(std::get<1>(GetParam()), 1, 1);
    size_t size = std::get<2>(GetParam());
    write(grid, block, size);
}

TEST_P(ParameterizedBlockSimpleCoarse, read) {
    dim3 grid = dim3(std::get<0>(GetParam()), 1, 1);
    dim3 block = dim3(std::get<1>(GetParam()), 1, 1);
    size_t size = std::get<2>(GetParam());
    read(grid, block, size);
}

INSTANTIATE_TEST_SUITE_P(
    IPCImplSimpleCoarseTestFixture,
    ParameterizedBlockSimpleCoarse,
    ::testing::Values(
        std::make_tuple(1, 1024, 32),      // 0
        std::make_tuple(1, 1,    1048576), // 1
        std::make_tuple(1, 2,    1048576), // 2
        std::make_tuple(1, 4,    1048576), // 3
        std::make_tuple(1, 8,    1048576), // 4
        std::make_tuple(1, 16,   1048576), // 5
        std::make_tuple(1, 32,   1048576), // 6
        std::make_tuple(1, 64,   1048576), // 7
        std::make_tuple(1, 128,  1048576), // 8
        std::make_tuple(1, 256,  1048576), // 9
        std::make_tuple(1, 512,  1048576), // 10
        std::make_tuple(1, 768,  1048576), // 11
        std::make_tuple(1, 1024, 1048576)) // 12
);

//=============================================================================

TEST_P(ParameterizedWarpSimpleCoarse, write) {
    dim3 grid = dim3(std::get<0>(GetParam()), 1, 1);
    dim3 block = dim3(std::get<1>(GetParam()), 1, 1);
    size_t size = std::get<2>(GetParam());
    write(grid, block, size);
}

TEST_P(ParameterizedWarpSimpleCoarse, read) {
    dim3 grid = dim3(std::get<0>(GetParam()), 1, 1);
    dim3 block = dim3(std::get<1>(GetParam()), 1, 1);
    size_t size = std::get<2>(GetParam());
    read(grid, block, size);
}

INSTANTIATE_TEST_SUITE_P(
    IPCImplSimpleCoarseTestFixture,
    ParameterizedWarpSimpleCoarse,
    ::testing::Values(
        std::make_tuple(1, 64,   1),       // 0
        std::make_tuple(1, 64,   32),      // 1
        std::make_tuple(1,  1,   1048576), // 2
        std::make_tuple(1,  2,   1048576), // 3
        std::make_tuple(1,  3,   1048576), // 4
        std::make_tuple(1,  4,   1048576), // 5
        std::make_tuple(1,  5,   1048576), // 6
        std::make_tuple(1,  6,   1048576), // 7
        std::make_tuple(1,  7,   1048576), // 8
        std::make_tuple(1,  8,   1048576), // 9
        std::make_tuple(1,  9,   1048576), // 10
        std::make_tuple(1, 10,   1048576), // 11
        std::make_tuple(1, 11,   1048576), // 12
        std::make_tuple(1, 12,   1048576), // 13
        std::make_tuple(1, 13,   1048576), // 14
        std::make_tuple(1, 14,   1048576), // 15
        std::make_tuple(1, 15,   1048576), // 16
        std::make_tuple(1, 16,   1048576), // 17
        std::make_tuple(1, 17,   1048576), // 18
        std::make_tuple(1, 18,   1048576), // 19
        std::make_tuple(1, 19,   1048576), // 20
        std::make_tuple(1, 20,   1048576), // 21
        std::make_tuple(1, 21,   1048576), // 22
        std::make_tuple(1, 22,   1048576), // 23
        std::make_tuple(1, 23,   1048576), // 24
        std::make_tuple(1, 24,   1048576), // 25
        std::make_tuple(1, 25,   1048576), // 26
        std::make_tuple(1, 26,   1048576), // 27
        std::make_tuple(1, 27,   1048576), // 28
        std::make_tuple(1, 28,   1048576), // 29
        std::make_tuple(1, 28,   1048576), // 30
        std::make_tuple(1, 29,   1048576), // 31
        std::make_tuple(1, 30,   1048576), // 32
        std::make_tuple(1, 31,   1048576), // 33
        std::make_tuple(1, 32,   1048576), // 34
        std::make_tuple(1, 33,   1048576), // 35
        std::make_tuple(1, 34,   1048576), // 36
        std::make_tuple(1, 35,   1048576), // 37
        std::make_tuple(1, 36,   1048576), // 38
        std::make_tuple(1, 37,   1048576), // 39
        std::make_tuple(1, 38,   1048576), // 40
        std::make_tuple(1, 39,   1048576), // 41
        std::make_tuple(1, 40,   1048576), // 42
        std::make_tuple(1, 41,   1048576), // 43
        std::make_tuple(1, 42,   1048576), // 44
        std::make_tuple(1, 43,   1048576), // 45
        std::make_tuple(1, 44,   1048576), // 46
        std::make_tuple(1, 45,   1048576), // 47
        std::make_tuple(1, 46,   1048576), // 48
        std::make_tuple(1, 47,   1048576), // 49
        std::make_tuple(1, 48,   1048576), // 50
        std::make_tuple(1, 49,   1048576), // 51
        std::make_tuple(1, 50,   1048576), // 52
        std::make_tuple(1, 51,   1048576), // 53
        std::make_tuple(1, 52,   1048576), // 54
        std::make_tuple(1, 53,   1048576), // 55
        std::make_tuple(1, 54,   1048576), // 56
        std::make_tuple(1, 55,   1048576), // 57
        std::make_tuple(1, 56,   1048576), // 58
        std::make_tuple(1, 57,   1048576), // 59
        std::make_tuple(1, 58,   1048576), // 60
        std::make_tuple(1, 59,   1048576), // 61
        std::make_tuple(1, 60,   1048576), // 62
        std::make_tuple(1, 61,   1048576), // 63
        std::make_tuple(1, 62,   1048576), // 64
        std::make_tuple(1, 63,   1048576), // 65
        std::make_tuple(1, 64,   1048576)) // 66
);

//=============================================================================

TEST_P(ParameterizedThreadSimpleCoarse, write) {
    dim3 grid = dim3(std::get<0>(GetParam()), 1, 1);
    dim3 block = dim3(std::get<1>(GetParam()), 1, 1);
    size_t size = std::get<2>(GetParam());
    write(grid, block, size);
}

TEST_P(ParameterizedThreadSimpleCoarse, read) {
    dim3 grid = dim3(std::get<0>(GetParam()), 1, 1);
    dim3 block = dim3(std::get<1>(GetParam()), 1, 1);
    size_t size = std::get<2>(GetParam());
    read(grid, block, size);
}

INSTANTIATE_TEST_SUITE_P(
    IPCImplSimpleCoarseTestFixture,
    ParameterizedThreadSimpleCoarse,
    ::testing::Values(
        std::make_tuple(1,  1,   1048576))
);
