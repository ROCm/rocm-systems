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

#include "address_record_gtest.hpp"

using namespace rocshmem;

TEST_F(AddressRecordTestFixture, split_entry) {
  auto [ar1, ar2] = split_.split();

  char* a1{reinterpret_cast<char*>(0x200)};
  size_t size1{0x40};
  ASSERT_EQ(ar1.get_address(), a1);
  ASSERT_EQ(ar1.get_size(), size1);

  char* a2{reinterpret_cast<char*>(0x240)};
  size_t size2{0x40};
  ASSERT_EQ(ar2.get_address(), a2);
  ASSERT_EQ(ar2.get_size(), size2);
}

#ifdef NDEBUG
TEST_F(AddressRecordTestFixture, DISABLED_split_bad_address)
#else
TEST_F(AddressRecordTestFixture, split_bad_address)
#endif
{
  ASSERT_DEATH({ bad_addr_.split(); }, "");
}

#ifdef NDEBUG
TEST_F(AddressRecordTestFixture, DISABLED_split_bad_size)
#else
TEST_F(AddressRecordTestFixture, split_bad_size)
#endif
{
  ASSERT_DEATH({ bad_size_.split(); }, "");
}

TEST_F(AddressRecordTestFixture, combine_1_into_combine_2) {
  AddressRecord ar{combine_1_.combine(combine_2_)};
  ASSERT_EQ(ar.get_address(), combine_1_.get_address());
  ASSERT_EQ(ar.get_size(), combine_1_.get_size() << 1);
}

#ifdef NDEBUG
TEST_F(AddressRecordTestFixture, DISABLED_combine_nullptr_record)
#else
TEST_F(AddressRecordTestFixture, combine_nullptr_record)
#endif
{
  ASSERT_DEATH({ AddressRecord ar = bad_addr_.combine(combine_2_); }, "");
}

#ifdef NDEBUG
TEST_F(AddressRecordTestFixture, DISABLED_combine_differerent_sizes)
#else
TEST_F(AddressRecordTestFixture, combine_different_sizes)
#endif
{
  char* a1{reinterpret_cast<char*>(0x120)};
  char* a2{reinterpret_cast<char*>(0x140)};
  size_t size1{0x20};
  size_t size2{0x40};

  AddressRecord ar1{a1, size1};
  AddressRecord ar2{a2, size2};
  ASSERT_DEATH({ AddressRecord ar = ar1.combine(ar2); }, "");
}
