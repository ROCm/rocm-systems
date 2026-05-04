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

#include "backend_bc_buffer_register_gtest.hpp"

using namespace rocshmem;

// ============================================================================
// buffer_register tests
// ============================================================================

TEST_F(BufferRegisterTestFixture, register_success) {
  char buffer[1024];
  int result = backend_->buffer_register(buffer, sizeof(buffer));
  EXPECT_EQ(result, ROCSHMEM_SUCCESS);
  EXPECT_EQ(backend_->user_buffer_regions.size(), 1u);
}

TEST_F(BufferRegisterTestFixture, register_null_address) {
  int result = backend_->buffer_register(nullptr, 1024);
  EXPECT_EQ(result, ROCSHMEM_ERROR);
  EXPECT_EQ(backend_->user_buffer_regions.size(), 0u);
}

TEST_F(BufferRegisterTestFixture, register_zero_length) {
  char buffer[1024];
  int result = backend_->buffer_register(buffer, 0);
  EXPECT_EQ(result, ROCSHMEM_ERROR);
  EXPECT_EQ(backend_->user_buffer_regions.size(), 0u);
}

TEST_F(BufferRegisterTestFixture, register_overflow) {
  void* addr = reinterpret_cast<void*>(UINTPTR_MAX - 100);
  int result = backend_->buffer_register(addr, 200);
  EXPECT_EQ(result, ROCSHMEM_ERROR);
  EXPECT_EQ(backend_->user_buffer_regions.size(), 0u);
}

TEST_F(BufferRegisterTestFixture, register_multiple_non_overlapping) {
  char buffer1[1024];
  char buffer2[1024];
  char buffer3[1024];

  int result1 = backend_->buffer_register(buffer1, sizeof(buffer1));
  int result2 = backend_->buffer_register(buffer2, sizeof(buffer2));
  int result3 = backend_->buffer_register(buffer3, sizeof(buffer3));

  EXPECT_EQ(result1, ROCSHMEM_SUCCESS);
  EXPECT_EQ(result2, ROCSHMEM_SUCCESS);
  EXPECT_EQ(result3, ROCSHMEM_SUCCESS);
  EXPECT_EQ(backend_->user_buffer_regions.size(), 3u);
}

TEST_F(BufferRegisterTestFixture, register_overlap_exact_same) {
  char buffer[1024];
  int result1 = backend_->buffer_register(buffer, sizeof(buffer));
  EXPECT_EQ(result1, ROCSHMEM_SUCCESS);

  int result2 = backend_->buffer_register(buffer, sizeof(buffer));
  EXPECT_EQ(result2, ROCSHMEM_ERROR);
  EXPECT_EQ(backend_->user_buffer_regions.size(), 1u);
}

TEST_F(BufferRegisterTestFixture, register_overlap_partial_start) {
  char buffer[2048];
  int result1 = backend_->buffer_register(buffer, 1024);
  EXPECT_EQ(result1, ROCSHMEM_SUCCESS);

  int result2 = backend_->buffer_register(buffer + 512, 1024);
  EXPECT_EQ(result2, ROCSHMEM_ERROR);
  EXPECT_EQ(backend_->user_buffer_regions.size(), 1u);
}

TEST_F(BufferRegisterTestFixture, register_overlap_partial_end) {
  char buffer[2048];
  int result1 = backend_->buffer_register(buffer + 512, 1024);
  EXPECT_EQ(result1, ROCSHMEM_SUCCESS);

  int result2 = backend_->buffer_register(buffer, 1024);
  EXPECT_EQ(result2, ROCSHMEM_ERROR);
  EXPECT_EQ(backend_->user_buffer_regions.size(), 1u);
}

TEST_F(BufferRegisterTestFixture, register_overlap_contained) {
  char buffer[2048];
  int result1 = backend_->buffer_register(buffer, sizeof(buffer));
  EXPECT_EQ(result1, ROCSHMEM_SUCCESS);

  int result2 = backend_->buffer_register(buffer + 256, 512);
  EXPECT_EQ(result2, ROCSHMEM_ERROR);
  EXPECT_EQ(backend_->user_buffer_regions.size(), 1u);
}

TEST_F(BufferRegisterTestFixture, register_overlap_containing) {
  char buffer[2048];
  int result1 = backend_->buffer_register(buffer + 256, 512);
  EXPECT_EQ(result1, ROCSHMEM_SUCCESS);

  int result2 = backend_->buffer_register(buffer, sizeof(buffer));
  EXPECT_EQ(result2, ROCSHMEM_ERROR);
  EXPECT_EQ(backend_->user_buffer_regions.size(), 1u);
}

TEST_F(BufferRegisterTestFixture, register_adjacent_no_overlap) {
  char buffer[2048];
  int result1 = backend_->buffer_register(buffer, 1024);
  EXPECT_EQ(result1, ROCSHMEM_SUCCESS);

  int result2 = backend_->buffer_register(buffer + 1024, 1024);
  EXPECT_EQ(result2, ROCSHMEM_SUCCESS);
  EXPECT_EQ(backend_->user_buffer_regions.size(), 2u);
}

// ============================================================================
// buffer_unregister tests
// ============================================================================

TEST_F(BufferRegisterTestFixture, unregister_success_by_start) {
  char buffer[1024];
  backend_->buffer_register(buffer, sizeof(buffer));
  ASSERT_EQ(backend_->user_buffer_regions.size(), 1u);

  int result = backend_->buffer_unregister(buffer);
  EXPECT_EQ(result, ROCSHMEM_SUCCESS);
  EXPECT_EQ(backend_->user_buffer_regions.size(), 0u);
}

TEST_F(BufferRegisterTestFixture, unregister_success_by_interior_address) {
  char buffer[1024];
  backend_->buffer_register(buffer, sizeof(buffer));
  ASSERT_EQ(backend_->user_buffer_regions.size(), 1u);

  int result = backend_->buffer_unregister(buffer + 512);
  EXPECT_EQ(result, ROCSHMEM_SUCCESS);
  EXPECT_EQ(backend_->user_buffer_regions.size(), 0u);
}

TEST_F(BufferRegisterTestFixture, unregister_null_address) {
  int result = backend_->buffer_unregister(nullptr);
  EXPECT_EQ(result, ROCSHMEM_ERROR);
}

TEST_F(BufferRegisterTestFixture, unregister_not_registered) {
  char buffer[1024];
  int result = backend_->buffer_unregister(buffer);
  EXPECT_EQ(result, ROCSHMEM_ERROR);
}

TEST_F(BufferRegisterTestFixture, unregister_past_end) {
  char buffer[1024];
  backend_->buffer_register(buffer, sizeof(buffer));
  ASSERT_EQ(backend_->user_buffer_regions.size(), 1u);

  int result = backend_->buffer_unregister(buffer + sizeof(buffer));
  EXPECT_EQ(result, ROCSHMEM_ERROR);
  EXPECT_EQ(backend_->user_buffer_regions.size(), 1u);
}

TEST_F(BufferRegisterTestFixture, unregister_one_of_multiple) {
  char buffer1[1024];
  char buffer2[1024];
  char buffer3[1024];

  backend_->buffer_register(buffer1, sizeof(buffer1));
  backend_->buffer_register(buffer2, sizeof(buffer2));
  backend_->buffer_register(buffer3, sizeof(buffer3));
  ASSERT_EQ(backend_->user_buffer_regions.size(), 3u);

  int result = backend_->buffer_unregister(buffer2);
  EXPECT_EQ(result, ROCSHMEM_SUCCESS);
  EXPECT_EQ(backend_->user_buffer_regions.size(), 2u);

  EXPECT_EQ(backend_->buffer_unregister(buffer2), ROCSHMEM_ERROR);
}

TEST_F(BufferRegisterTestFixture, register_after_unregister) {
  char buffer[1024];
  backend_->buffer_register(buffer, sizeof(buffer));
  ASSERT_EQ(backend_->user_buffer_regions.size(), 1u);

  backend_->buffer_unregister(buffer);
  ASSERT_EQ(backend_->user_buffer_regions.size(), 0u);

  int result = backend_->buffer_register(buffer, sizeof(buffer));
  EXPECT_EQ(result, ROCSHMEM_SUCCESS);
  EXPECT_EQ(backend_->user_buffer_regions.size(), 1u);
}
