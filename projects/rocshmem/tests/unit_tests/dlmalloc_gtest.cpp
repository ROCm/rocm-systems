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

#include "dlmalloc_gtest.hpp"
#include <cstdint>

using namespace rocshmem;

// a small portion of the heap is not available due to cost of dlmalloc bookkeeping
#define DLMALLOC_BOOKKEEPING static_cast<size_t>(128 * ALIGNMENT)

TEST_F(DLMallocTestFixture, used_0_bytes) {
  size_t heap_size{1 << 30};
  ASSERT_LE(strat_.get_used(), DLMALLOC_BOOKKEEPING);
  ASSERT_EQ(strat_.get_used() + strat_.get_avail(), heap_size);
}

TEST_F(DLMallocTestFixture, alloc_0_bytes) {
  size_t initial_used{strat_.get_used()};
  char* c_ptr{nullptr};
  size_t size{0};
  strat_.alloc(&c_ptr, size);
  ASSERT_EQ(c_ptr, nullptr);
  ASSERT_EQ(strat_.get_used(), initial_used);
}

TEST_F(DLMallocTestFixture, alloc_1_byte) {
  char* c_ptr{nullptr};
  size_t size{1};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);
}

TEST_F(DLMallocTestFixture, alloc_128_bytes) {
  char* c_ptr{nullptr};
  size_t size{128};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);
}

TEST_F(DLMallocTestFixture, alloc_256_bytes) {
  char* c_ptr{nullptr};
  size_t size{256};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);
}

TEST_F(DLMallocTestFixture, alloc_512_bytes) {
  char* c_ptr{nullptr};
  size_t size{512};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);
}

TEST_F(DLMallocTestFixture, alloc_513_bytes) {
  char* c_ptr{nullptr};
  size_t size{513};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);
}

TEST_F(DLMallocTestFixture, alloc_4KB) {
  char* c_ptr{nullptr};
  size_t size{4096};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);
}

TEST_F(DLMallocTestFixture, alloc_128KB) {
  char* c_ptr{nullptr};
  size_t size{131072};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);
}

TEST_F(DLMallocTestFixture, alloc_1GB) {
  char* c_ptr{nullptr};
  size_t heap_size{1 << 30};
  size_t size{heap_size - DLMALLOC_BOOKKEEPING};

  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);
  ASSERT_EQ(strat_.get_used() + strat_.get_avail(), heap_size);
}

TEST_F(DLMallocTestFixture, alloc_256_bytes_X2_free_256_bytes_X2) {
  char* c_ptr_1{nullptr};
  char* c_ptr_2{nullptr};
  size_t size{256};
  strat_.alloc(&c_ptr_1, size);
  ASSERT_NE(c_ptr_1, nullptr);
  strat_.alloc(&c_ptr_2, size);
  ASSERT_NE(c_ptr_2, nullptr);

  strat_.free(c_ptr_1);
  strat_.free(c_ptr_2);
}

TEST_F(DLMallocTestFixture, alloc_1GB_free_1GB) {
  char* c_ptr{nullptr};
  size_t heap_size{1 << 30};
  size_t size{heap_size - DLMALLOC_BOOKKEEPING};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);
  ASSERT_EQ(strat_.get_used() + strat_.get_avail(), heap_size);

  strat_.free(c_ptr);
}

