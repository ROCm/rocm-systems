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

#include "single_heap_gtest.hpp"

using namespace rocshmem;

TEST_F(SingleHeapTestFixture, unallocated_size_check) {
  ASSERT_EQ(single_heap_.get_size(), 1 << 30);
}

TEST_F(SingleHeapTestFixture, free_null) {
  void* ptr{nullptr};
  single_heap_.free(ptr);
}

TEST_F(SingleHeapTestFixture, alloc_0) {
  // some allocators (e.g. dlmalloc) use memory for internal bookkeeping
  size_t initial_used{single_heap_.get_used()};
  size_t request_size{0};
  void* ptr{nullptr};

  single_heap_.malloc(&ptr, request_size);
  ASSERT_EQ(ptr, nullptr);
  ASSERT_EQ(single_heap_.get_used(), initial_used);

  single_heap_.free(ptr);
  ASSERT_EQ(single_heap_.get_used(), initial_used);
}

TEST_F(SingleHeapTestFixture, alloc_1) {
  size_t initial_used{single_heap_.get_used()};
  size_t request_size{1};
  void* ptr{nullptr};

  single_heap_.malloc(&ptr, request_size);
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr) & (ALIGNMENT-1), 0);

  single_heap_.free(ptr);
  ASSERT_EQ(single_heap_.get_used(), initial_used);
}

TEST_F(SingleHeapTestFixture, alloc_256) {
  size_t initial_used{single_heap_.get_used()};
  size_t request_size{256};
  void* ptr{nullptr};

  single_heap_.malloc(&ptr, request_size);
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr) & (ALIGNMENT-1), 0);

  single_heap_.free(ptr);
  ASSERT_EQ(single_heap_.get_used(), initial_used);
}

TEST_F(SingleHeapTestFixture, alloc_1024) {
  size_t initial_used{single_heap_.get_used()};
  size_t request_size{1024};
  void* ptr{nullptr};

  single_heap_.malloc(&ptr, request_size);
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr) & (ALIGNMENT-1), 0);

  single_heap_.free(ptr);
  ASSERT_EQ(single_heap_.get_used(), initial_used);
}

TEST_F(SingleHeapTestFixture, alloc_1MB) {
  size_t initial_used{single_heap_.get_used()};
  size_t request_size{1 << 20};
  void* ptr{nullptr};

  single_heap_.malloc(&ptr, request_size);
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr) & (ALIGNMENT-1), 0);

  single_heap_.free(ptr);
  ASSERT_EQ(single_heap_.get_used(), initial_used);
}

TEST_F(SingleHeapTestFixture, alloc_4097) {
  size_t initial_used{single_heap_.get_used()};
  size_t request_size{4097};
  void* ptr{nullptr};

  single_heap_.malloc(&ptr, request_size);
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr) & (ALIGNMENT-1), 0);

  single_heap_.free(ptr);
  ASSERT_EQ(single_heap_.get_used(), initial_used);
}

TEST_F(SingleHeapTestFixture, alloc_X2_8191) {
  size_t initial_used{single_heap_.get_used()};
  size_t request_size{8191};
  void* ptr_1{nullptr};
  void* ptr_2{nullptr};

  single_heap_.malloc(&ptr_1, request_size);
  ASSERT_NE(ptr_1, nullptr);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr_1) & (ALIGNMENT-1), 0);

  single_heap_.malloc(&ptr_2, request_size);
  ASSERT_NE(ptr_2, nullptr);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr_2) & (ALIGNMENT-1), 0);

  single_heap_.free(ptr_1);
  single_heap_.free(ptr_2);
  ASSERT_EQ(single_heap_.get_used(), initial_used);
}

TEST_F(SingleHeapTestFixture, alloc_X2_free_alloc_free_X2_1MB) {
  size_t initial_used{single_heap_.get_used()};
  void* ptr_1{nullptr};
  void* ptr_2{nullptr};
  void* ptr_3{nullptr};
  size_t request_size{1 << 20};

  single_heap_.malloc(&ptr_1, request_size);
  ASSERT_NE(ptr_1, nullptr);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr_1) & (ALIGNMENT-1), 0);

  single_heap_.malloc(&ptr_2, request_size);
  ASSERT_NE(ptr_1, nullptr);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr_2) & (ALIGNMENT-1), 0);

  single_heap_.free(ptr_1);

  single_heap_.malloc(&ptr_3, request_size);
  ASSERT_NE(ptr_3, nullptr);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr_3) & (ALIGNMENT-1), 0);

  single_heap_.free(ptr_3);
  single_heap_.free(ptr_2);
  ASSERT_EQ(single_heap_.get_used(), initial_used);
}

TEST_F(SingleHeapTestFixture, alloc_X2_free_alloc_free_X2_63) {
  size_t initial_used{single_heap_.get_used()};
  void* ptr_1{nullptr};
  void* ptr_2{nullptr};
  void* ptr_3{nullptr};
  size_t request_size{63};

  single_heap_.malloc(&ptr_1, request_size);
  ASSERT_NE(ptr_1, nullptr);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr_1) & (ALIGNMENT-1), 0);

  single_heap_.malloc(&ptr_2, request_size);
  ASSERT_NE(ptr_1, nullptr);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr_2) & (ALIGNMENT-1), 0);

  single_heap_.free(ptr_1);

  single_heap_.malloc(&ptr_3, request_size);
  ASSERT_NE(ptr_3, nullptr);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr_3) & (ALIGNMENT-1), 0);

  single_heap_.free(ptr_3);
  single_heap_.free(ptr_2);
  ASSERT_EQ(single_heap_.get_used(), initial_used);
}
