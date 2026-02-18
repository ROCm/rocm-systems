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

#include "binner_gtest.hpp"

using namespace rocshmem;

TEST(Binner, ffs_0) {
  size_t val{0x0};
  auto bit_pos{find_first_set_one(val)};
  ASSERT_EQ(bit_pos, UINT_MAX);
}

TEST(Binner, ffs_1) {
  size_t val{0x1};
  auto bit_pos{find_first_set_one(val)};
  ASSERT_EQ(bit_pos, 0);
}

TEST(Binner, ffs_2) {
  size_t val{0x2};
  auto bit_pos{find_first_set_one(val)};
  ASSERT_EQ(bit_pos, 1);
}

TEST(Binner, ffs_4) {
  size_t val{0x4};
  auto bit_pos{find_first_set_one(val)};
  ASSERT_EQ(bit_pos, 2);
}

TEST(Binner, ffs_8) {
  size_t val{0x8};
  auto bit_pos{find_first_set_one(val)};
  ASSERT_EQ(bit_pos, 3);
}

TEST(Binner, ffs_100) {
  size_t val{0x100};
  auto bit_pos{find_first_set_one(val)};
  ASSERT_EQ(bit_pos, 8);
}

TEST(Binner, ffs_8000) {
  size_t val{0x8000};
  auto bit_pos{find_first_set_one(val)};
  ASSERT_EQ(bit_pos, 15);
}

TEST(Binner, ffs_ff80) {
  size_t val{0xFF80};
  auto bit_pos{find_first_set_one(val)};
  ASSERT_EQ(bit_pos, 15);
}

TEST(Binner, ffs_4000_0000) {
  size_t val{0x40000000};
  auto bit_pos{find_first_set_one(val)};
  ASSERT_EQ(bit_pos, 30);
}

TEST(Binner, ffs_0100_0000_0000_0000) {
  size_t val{0x0100000000000000};
  auto bit_pos{find_first_set_one(val)};
  ASSERT_EQ(bit_pos, 56);
}

TEST(Binner, ffs_0200_0000_0000_0000) {
  size_t val{0x0200000000000000};
  auto bit_pos{find_first_set_one(val)};
  ASSERT_EQ(bit_pos, 57);
}

TEST(Binner, ffs_4000_0000_0000_0000) {
  size_t val{0x4000000000000000};
  auto bit_pos{find_first_set_one(val)};
  ASSERT_EQ(bit_pos, 62);
}

TEST(Binner, ffs_8000_0000_0000_0000) {
  size_t val{0x8000000000000000};
  auto bit_pos{find_first_set_one(val)};
  ASSERT_EQ(bit_pos, 63);
}

TEST(Binner, ffs_max) {
  size_t val = -1;
  auto bit_pos{find_first_set_one(val)};
  ASSERT_EQ(bit_pos, 63);
}

TEST_F(BinnerTestFixture, correct_bin_sizes_one_gig) {
  auto bins{binner_.get_bins()};
  ASSERT_EQ(bins->size(), 24);

  std::array<unsigned, 24> a;
  for (unsigned i = 0; i < 24; i++) {
    a[i] = i + 7;
  }

  for (auto e : a) {
    bool found = bins->count(std::pow(2, e));
    ASSERT_TRUE(found);
  }
}

TEST_F(BinnerTestFixture, bin_get_one_gig) {
  binner_.assign_heap_to_bins();

  auto bins{binner_.get_bins()};
  ASSERT_EQ(bins->size(), 24);

  size_t gibibyte = std::pow(2, 30);
  auto bin{(*bins)[gibibyte]};

  ASSERT_EQ(bin.size(), 1);

  auto address_record = bin.get();
  ASSERT_EQ(address_record.get_size(), gibibyte);
}
