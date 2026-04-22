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

#include "symmetric_heap_gtest.hpp"

using namespace rocshmem;

TEST_F(SymmetricHeapTestFixture, malloc_free) {
  void *ptr{nullptr};
  size_t request_bytes{48};

  symmetric_heap_->malloc(&ptr, request_bytes);
  ASSERT_NE(ptr, nullptr);
  ASSERT_NO_FATAL_FAILURE(symmetric_heap_->free(ptr));
}

TEST_F(SymmetricHeapTestFixture, window_info) {
  auto win_info_ptr{symmetric_heap_->get_window_info()};

  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(win_info_ptr);
  if (window_info_mpi) {
    void *window_base_addr{nullptr};
    int flag{0};
    MPI_Win_get_attr(window_info_mpi->get_win(), MPI_WIN_BASE, &window_base_addr,
                     &flag);
    ASSERT_NE(0, flag);
    ASSERT_NE(nullptr, window_base_addr);
  }
}

TEST_F(SymmetricHeapTestFixture, heap_bases) {
  auto heap_bases{symmetric_heap_->get_heap_bases()};
  for (const auto &base : heap_bases) {
    ASSERT_NE(nullptr, base);
  }
}
