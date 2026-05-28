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

#include "index_strategy_gtest.hpp"

#define HIP_ENABLE_PRINTF

using namespace rocshmem;

TEST_F(IndexStrategyTestFixture,
       run_TCBA_memory_set_test_grid_1_1_1_block_1_1_1) {
  using IndexStrategy = Thread_Contiguous_Block_Agnostic;
  run_memory_set_test<IndexStrategy>({1, 1, 1}, {1, 1, 1});
}

//=============================================================================
TEST_F(IndexStrategyTestFixture,
       run_TDBD_memory_set_test_grid_1_1_1_block_64_1_1) {
  using IndexStrategy = Thread_Discontiguous_Block_Discontiguous;
  run_memory_set_test<IndexStrategy>({1, 1, 1}, {64, 1, 1});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBD_memory_set_test_grid_1_1_1_block_64_2_1) {
  using IndexStrategy = Thread_Discontiguous_Block_Discontiguous;
  run_memory_set_test<IndexStrategy>({1, 1, 1}, {64, 2, 1});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBD_memory_set_test_grid_1_1_1_block_64_4_1) {
  using IndexStrategy = Thread_Discontiguous_Block_Discontiguous;
  run_memory_set_test<IndexStrategy>({1, 1, 1}, {64, 4, 1});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBD_memory_set_test_grid_1_1_1_block_64_16_1) {
  using IndexStrategy = Thread_Discontiguous_Block_Discontiguous;
  run_memory_set_test<IndexStrategy>({1, 1, 1}, {64, 16, 1});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBD_memory_set_test_grid_1_1_1_block_16_16_4) {
  using IndexStrategy = Thread_Discontiguous_Block_Discontiguous;
  run_memory_set_test<IndexStrategy>({1, 1, 1}, {16, 16, 4});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBD_memory_set_test_grid_4_1_1_block_16_16_4) {
  using IndexStrategy = Thread_Discontiguous_Block_Discontiguous;
  run_memory_set_test<IndexStrategy>({4, 1, 1}, {16, 16, 4});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBD_memory_set_test_grid_4_4_4_block_16_16_4) {
  using IndexStrategy = Thread_Discontiguous_Block_Discontiguous;
  run_memory_set_test<IndexStrategy>({4, 4, 4}, {16, 16, 4});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBD_memory_set_test_grid_1024_8_8_block_1024_1_1) {
  using IndexStrategy = Thread_Discontiguous_Block_Discontiguous;
  run_memory_set_test<IndexStrategy>({1024, 8, 8}, {1024, 1, 1});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBD_memory_set_test_grid_1024_8_8_block_1_1024_1) {
  using IndexStrategy = Thread_Discontiguous_Block_Discontiguous;
  run_memory_set_test<IndexStrategy>({1024, 8, 8}, {1, 1024, 1});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBD_memory_set_test_grid_1024_8_8_block_1_1_1024) {
  using IndexStrategy = Thread_Discontiguous_Block_Discontiguous;
  run_memory_set_test<IndexStrategy>({1024, 8, 8}, {1, 1, 1024});
}

//=============================================================================
TEST_F(IndexStrategyTestFixture,
       run_TDBC_memory_set_test_grid_1_1_1_block_64_1_1) {
  using IndexStrategy = Thread_Discontiguous_Block_Contiguous;
  run_memory_set_test<IndexStrategy>({1, 1, 1}, {64, 1, 1});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBC_memory_set_test_grid_1_1_1_block_64_2_1) {
  using IndexStrategy = Thread_Discontiguous_Block_Contiguous;
  run_memory_set_test<IndexStrategy>({1, 1, 1}, {64, 2, 1});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBC_memory_set_test_grid_1_1_1_block_64_4_1) {
  using IndexStrategy = Thread_Discontiguous_Block_Contiguous;
  run_memory_set_test<IndexStrategy>({1, 1, 1}, {64, 4, 1});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBC_memory_set_test_grid_1_1_1_block_64_16_1) {
  using IndexStrategy = Thread_Discontiguous_Block_Contiguous;
  run_memory_set_test<IndexStrategy>({1, 1, 1}, {64, 16, 1});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBC_memory_set_test_grid_1_1_1_block_16_16_4) {
  using IndexStrategy = Thread_Discontiguous_Block_Contiguous;
  run_memory_set_test<IndexStrategy>({1, 1, 1}, {16, 16, 4});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBC_memory_set_test_grid_4_1_1_block_16_16_4) {
  using IndexStrategy = Thread_Discontiguous_Block_Contiguous;
  run_memory_set_test<IndexStrategy>({4, 1, 1}, {16, 16, 4});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBC_memory_set_test_grid_4_4_4_block_16_16_4) {
  using IndexStrategy = Thread_Discontiguous_Block_Contiguous;
  run_memory_set_test<IndexStrategy>({4, 4, 4}, {16, 16, 4});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBC_memory_set_test_grid_1024_8_8_block_1024_1_1) {
  using IndexStrategy = Thread_Discontiguous_Block_Contiguous;
  run_memory_set_test<IndexStrategy>({1024, 8, 8}, {1024, 1, 1});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBC_memory_set_test_grid_1024_8_8_block_1_1024_1) {
  using IndexStrategy = Thread_Discontiguous_Block_Contiguous;
  run_memory_set_test<IndexStrategy>({1024, 8, 8}, {1, 1024, 1});
}

TEST_F(IndexStrategyTestFixture,
       run_TDBC_memory_set_test_grid_1024_8_8_block_1_1_1024) {
  using IndexStrategy = Thread_Discontiguous_Block_Contiguous;
  run_memory_set_test<IndexStrategy>({1024, 8, 8}, {1, 1, 1024});
}
