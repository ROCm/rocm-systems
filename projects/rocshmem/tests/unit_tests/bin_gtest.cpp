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

#include "bin_gtest.hpp"

using namespace rocshmem;

TEST_F(BinTestFixture, is_empty_check) { ASSERT_TRUE(bin_.empty()); }

TEST_F(BinTestFixture, is_not_empty_check) {
  bin_.put(nullptr);

  ASSERT_FALSE(bin_.empty());
}

TEST_F(BinTestFixture, size_check) {
  ASSERT_EQ(bin_.size(), 0);
  bin_.put(nullptr);
  ASSERT_EQ(bin_.size(), 1);
  bin_.put(nullptr);
  ASSERT_EQ(bin_.size(), 2);
}

TEST_F(BinTestFixture, retrieval_check) {
  char* p_xa{reinterpret_cast<char*>(0xa)};
  char* p_xb{reinterpret_cast<char*>(0xb)};
  bin_.put(p_xa);
  bin_.put(p_xb);
  auto g_xb = bin_.get();
  auto g_xa = bin_.get();
  ASSERT_EQ(p_xa, g_xa);
  ASSERT_EQ(p_xb, g_xb);
}
