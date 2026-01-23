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

#include "ipc_impl_tiled_fine_gtest.hpp"

using namespace rocshmem;

//=============================================================================

TEST_F(DegenerateTiledFine, ptr_check) {
    ASSERT_NE(heap_mem_.get_ptr(), nullptr);
}

TEST_F(DegenerateTiledFine, MPI_num_pes) {
    ASSERT_EQ(mpi_->num_pes(), 2);
}

TEST_F(DegenerateTiledFine, IPC_bases) {
    ASSERT_EQ(mpi_->num_pes(), 2);
    ASSERT_NE(ipc_impl_.ipc_bases, nullptr);
    for(int i{0}; i < mpi_->num_pes(); i++) {
        ASSERT_NE(ipc_impl_.ipc_bases[i], nullptr);
    }
}

TEST_F(DegenerateTiledFine, golden_1048576_int) {
    iota_golden(1048576);
    validate_golden(1048576);
}

//=============================================================================

int block_signals_calculation(int grid_dim_x, int block_dim_x, size_t size) {
    size_t bytes = size * sizeof(int);
    int total_num_threads {grid_dim_x * block_dim_x};
    int one_grid_iteration_data_size {total_num_threads * THREAD_TRANSFER_GRANULARITY};
    int num_grid_iterations = ((bytes - THREAD_TRANSFER_GRANULARITY) + one_grid_iteration_data_size) / one_grid_iteration_data_size;
    int partial_grid_last_iteration_data_size = bytes % one_grid_iteration_data_size;

    int bytes_per_block {block_dim_x * THREAD_TRANSFER_GRANULARITY};

    int num_signals_for_one_full_iteration = grid_dim_x;
    int num_signals_for_partial_last_iteration = ((partial_grid_last_iteration_data_size - THREAD_TRANSFER_GRANULARITY) + bytes_per_block) / bytes_per_block;

    int num_signals = 0;
    if (partial_grid_last_iteration_data_size) {
        num_signals = num_signals_for_one_full_iteration * (num_grid_iterations - 1);
	num_signals += num_signals_for_partial_last_iteration;
    } else {
        num_signals = num_signals_for_one_full_iteration * num_grid_iterations;
    }

    return num_signals;
}

TEST_P(ParameterizedBlockTiledFine, write) {
    dim3 grid = dim3(std::get<0>(GetParam()), 1, 1);
    dim3 block = dim3(std::get<1>(GetParam()), 1, 1);
    size_t size = std::get<2>(GetParam());
    auto number_of_signals_required = block_signals_calculation(std::get<0>(GetParam()), std::get<1>(GetParam()), size);
    write(grid, block, size, number_of_signals_required);
}

TEST_P(ParameterizedBlockTiledFine, read) {
    dim3 grid = dim3(std::get<0>(GetParam()), 1, 1);
    dim3 block = dim3(std::get<1>(GetParam()), 1, 1);
    size_t size = std::get<2>(GetParam());
    read(grid, block, size);
}

INSTANTIATE_TEST_SUITE_P(
    IPCImplTiledFineTestFixture,
    ParameterizedBlockTiledFine,
    ::testing::Values(
        std::make_tuple(1,  1024, 32),       // 0
        std::make_tuple(1,  1024, 2048),     // 1
        std::make_tuple(1,  1024, 65536),    // 2
        std::make_tuple(1,  1,    1048576),  // 3
        std::make_tuple(1,  2,    1048576),  // 4
        std::make_tuple(1,  4,    1048576),  // 5
        std::make_tuple(1,  8,    1048576),  // 6
        std::make_tuple(1,  16,   1048576),  // 7
        std::make_tuple(1,  32,   1048576),  // 8
        std::make_tuple(1,  64,   1048576),  // 9
        std::make_tuple(1,  128,  1048576),  // 10
        std::make_tuple(1,  256,  1048576),  // 11
        std::make_tuple(1,  512,  1048576),  // 12
        std::make_tuple(1,  768,  1048576),  // 13
        std::make_tuple(1,  1024, 1048576),  // 14
        std::make_tuple(2,  1,    1048576),  // 15
        std::make_tuple(2,  2,    1048576),  // 16
        std::make_tuple(2,  4,    1048576),  // 17
        std::make_tuple(2,  8,    1048576),  // 18
        std::make_tuple(2,  16,   1048576),  // 19
        std::make_tuple(2,  32,   1048576),  // 20
        std::make_tuple(2,  64,   1048576),  // 21
        std::make_tuple(2,  128,  1048576),  // 22
        std::make_tuple(2,  256,  1048576),  // 23
        std::make_tuple(2,  512,  1048576),  // 24
        std::make_tuple(2,  768,  1048576),  // 25
        std::make_tuple(2,  1024, 1048576),  // 26
        std::make_tuple(4,  1024, 1048576),  // 27
        std::make_tuple(8,  1024, 1048576),  // 28
        std::make_tuple(16, 1024, 1048576),  // 29
        std::make_tuple(32, 1024, 1048576),  // 30
        std::make_tuple(38, 1024, 1048576),  // 31
        std::make_tuple(38, 1024, 2097152),  // 32
        std::make_tuple(38, 1024, 4194304),  // 33
        std::make_tuple(38, 1024, 8388608),  // 34
        std::make_tuple(38, 1024, 16777216), // 35
        std::make_tuple(38, 1024, 33554432)) // 36
);

//=============================================================================

int warp_signals_calculation(int grid_dim_x, int block_dim_x, size_t size) {
    size_t bytes = size * sizeof(int);
    int total_num_threads {grid_dim_x * block_dim_x};
    int one_grid_iteration_data_size {total_num_threads * THREAD_TRANSFER_GRANULARITY};
    int num_grid_iterations = ((bytes - THREAD_TRANSFER_GRANULARITY) + one_grid_iteration_data_size) / one_grid_iteration_data_size;
    int partial_grid_last_iteration_data_size = bytes % one_grid_iteration_data_size;

    int warps_per_block {warpsPerBlock(block_dim_x)};
    int total_num_warps_in_grid {grid_dim_x * warps_per_block};
    int bytes_per_warp {min(block_dim_x, WARP_SIZE) * THREAD_TRANSFER_GRANULARITY};

    int num_signals_for_one_full_iteration = total_num_warps_in_grid;
    int num_signals_for_partial_last_iteration = ((partial_grid_last_iteration_data_size - THREAD_TRANSFER_GRANULARITY) + bytes_per_warp) / bytes_per_warp;

    int num_signals = 0;
    if (partial_grid_last_iteration_data_size) {
        num_signals = num_signals_for_one_full_iteration * (num_grid_iterations - 1);
	num_signals += num_signals_for_partial_last_iteration;
    } else {
        num_signals = num_signals_for_one_full_iteration * num_grid_iterations;
    }

    return num_signals;
}

TEST_P(ParameterizedWarpTiledFine, write) {
    dim3 grid = dim3(std::get<0>(GetParam()), 1, 1);
    dim3 block = dim3(std::get<1>(GetParam()), 1, 1);
    size_t size = std::get<2>(GetParam());
    auto number_of_signals_required = warp_signals_calculation(std::get<0>(GetParam()), std::get<1>(GetParam()), size);
    write(grid, block, size, number_of_signals_required);
}

TEST_P(ParameterizedWarpTiledFine, read) {
    dim3 grid = dim3(std::get<0>(GetParam()), 1, 1);
    dim3 block = dim3(std::get<1>(GetParam()), 1, 1);
    size_t size = std::get<2>(GetParam());
    read(grid, block, size);
}

INSTANTIATE_TEST_SUITE_P(
    IPCImplTiledFineTestFixture,
    ParameterizedWarpTiledFine,
    ::testing::Values(
        std::make_tuple(1, 64,           8), // 0
        std::make_tuple(1, 64,          32), // 1
        std::make_tuple(1,  1,     1048576), // 2
        std::make_tuple(1,  2,     1048576), // 3
        std::make_tuple(1,  3,     1048576), // 4
        std::make_tuple(1,  4,     1048576), // 5
        std::make_tuple(1,  5,     1048576), // 6
        std::make_tuple(1,  6,     1048576), // 7
        std::make_tuple(1,  7,     1048576), // 8
        std::make_tuple(1,  8,     1048576), // 9
        std::make_tuple(1,  9,     1048576), // 10
        std::make_tuple(1,  10,    1048576), // 11
        std::make_tuple(1,  11,    1048576), // 12
        std::make_tuple(1,  12,    1048576), // 13
        std::make_tuple(1,  13,    1048576), // 14
        std::make_tuple(1,  14,    1048576), // 15
        std::make_tuple(1,  15,    1048576), // 16
        std::make_tuple(1,  16,    1048576), // 17
        std::make_tuple(1,  17,    1048576), // 18
        std::make_tuple(1,  18,    1048576), // 19
        std::make_tuple(1,  19,    1048576), // 20
        std::make_tuple(1,  20,    1048576), // 21
        std::make_tuple(1,  21,    1048576), // 22
        std::make_tuple(1,  22,    1048576), // 23
        std::make_tuple(1,  23,    1048576), // 24
        std::make_tuple(1,  24,    1048576), // 25
        std::make_tuple(1,  25,    1048576), // 26
        std::make_tuple(1,  26,    1048576), // 27
        std::make_tuple(1,  27,    1048576), // 28
        std::make_tuple(1,  28,    1048576), // 29
        std::make_tuple(1,  28,    1048576), // 30
        std::make_tuple(1,  29,    1048576), // 31
        std::make_tuple(1,  30,    1048576), // 32
        std::make_tuple(1,  31,    1048576), // 33
        std::make_tuple(1,  32,    1048576), // 34
        std::make_tuple(1,  33,    1048576), // 35
        std::make_tuple(1,  34,    1048576), // 36
        std::make_tuple(1,  35,    1048576), // 37
        std::make_tuple(1,  36,    1048576), // 38
        std::make_tuple(1,  37,    1048576), // 39
        std::make_tuple(1,  38,    1048576), // 40
        std::make_tuple(1,  39,    1048576), // 41
        std::make_tuple(1,  40,    1048576), // 42
        std::make_tuple(1,  41,    1048576), // 43
        std::make_tuple(1,  42,    1048576), // 44
        std::make_tuple(1,  43,    1048576), // 45
        std::make_tuple(1,  44,    1048576), // 46
        std::make_tuple(1,  45,    1048576), // 47
        std::make_tuple(1,  46,    1048576), // 48
        std::make_tuple(1,  47,    1048576), // 49
        std::make_tuple(1,  48,    1048576), // 50
        std::make_tuple(1,  49,    1048576), // 51
        std::make_tuple(1,  50,    1048576), // 52
        std::make_tuple(1,  51,    1048576), // 53
        std::make_tuple(1,  52,    1048576), // 54
        std::make_tuple(1,  53,    1048576), // 55
        std::make_tuple(1,  54,    1048576), // 56
        std::make_tuple(1,  55,    1048576), // 57
        std::make_tuple(1,  56,    1048576), // 58
        std::make_tuple(1,  57,    1048576), // 59
        std::make_tuple(1,  58,    1048576), // 60
        std::make_tuple(1,  59,    1048576), // 61
        std::make_tuple(1,  60,    1048576), // 62
        std::make_tuple(1,  61,    1048576), // 63
        std::make_tuple(1,  62,    1048576), // 64
        std::make_tuple(1,  63,    1048576), // 65
        std::make_tuple(1,  64,    1048576), // 66
        std::make_tuple(1,  256,   1048576), // 67
        std::make_tuple(1,  512,   1048576), // 68
        std::make_tuple(1,  768,   1048576), // 69
        std::make_tuple(1,  1024,  1048576), // 70
        std::make_tuple(2,  32,    1048576), // 71
        std::make_tuple(2,  64,    1048576), // 72
        std::make_tuple(2,  128,   1048576), // 73
        std::make_tuple(2,  256,   1048576), // 74
        std::make_tuple(2,  512,   1048576), // 75
        std::make_tuple(2,  1024,  1048576), // 76
        std::make_tuple(4,  32,    1048576), // 77
        std::make_tuple(4,  64,    1048576), // 78
        std::make_tuple(4,  128,   1048576), // 79
        std::make_tuple(4,  256,   1048576), // 80
        std::make_tuple(4,  512,   1048576), // 81
        std::make_tuple(4,  1024,  1048576), // 82
        std::make_tuple(8,  32,    1048576), // 83
        std::make_tuple(8,  64,    1048576), // 84
        std::make_tuple(8,  128,   1048576), // 85
        std::make_tuple(8,  256,   1048576), // 86
        std::make_tuple(8,  512,   1048576), // 87
        std::make_tuple(8,  1024,  1048576), // 88
        std::make_tuple(16, 32,    1048576), // 89
        std::make_tuple(16, 64,    1048576), // 90
        std::make_tuple(16, 128,   1048576), // 91
        std::make_tuple(16, 256,   1048576), // 92
        std::make_tuple(16, 512,   1048576), // 93
        std::make_tuple(16, 1024,  1048576), // 94
        std::make_tuple(32, 32,    1048576), // 95
        std::make_tuple(32, 64,    1048576), // 96
        std::make_tuple(32, 128,   1048576), // 97
        std::make_tuple(32, 256,   1048576), // 98
        std::make_tuple(32, 512,   1048576), // 99
        std::make_tuple(32, 1024,  1048576), // 100
        std::make_tuple(38, 32,    1048576), // 101
        std::make_tuple(38, 64,    1048576), // 102
        std::make_tuple(38, 128,   1048576), // 103
        std::make_tuple(38, 256,   1048576), // 104
        std::make_tuple(38, 512,   1048576), // 105
        std::make_tuple(38, 1024,  1048576), // 106
        std::make_tuple(38, 1024,  2097152), // 107
        std::make_tuple(38, 1024,  4194304), // 108
        std::make_tuple(38, 1024,  8388608), // 109
        std::make_tuple(38, 1024, 16777216), // 110
        std::make_tuple(38, 1024, 33554432)) // 111
);

//=============================================================================

TEST_P(ParameterizedThreadTiledFine, write) {
    dim3 grid = dim3(std::get<0>(GetParam()), 1, 1);
    dim3 block = dim3(std::get<1>(GetParam()), 1, 1);
    size_t size = std::get<2>(GetParam());
    write(grid, block, size, 1);
}

TEST_P(ParameterizedThreadTiledFine, read) {
    dim3 grid = dim3(std::get<0>(GetParam()), 1, 1);
    dim3 block = dim3(std::get<1>(GetParam()), 1, 1);
    size_t size = std::get<2>(GetParam());
    read(grid, block, size);
}

INSTANTIATE_TEST_SUITE_P(
    IPCImplTiledFineTestFixture,
    ParameterizedThreadTiledFine,
    ::testing::Values(
        std::make_tuple(1,  1,   1048576))
);
