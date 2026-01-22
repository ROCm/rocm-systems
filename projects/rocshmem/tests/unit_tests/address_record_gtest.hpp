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

#ifndef ROCSHMEM_ADDRESS_RECORD_GTEST_HPP
#define ROCSHMEM_ADDRESS_RECORD_GTEST_HPP

#include "gtest/gtest.h"

#include "../src/memory/address_record.hpp"

namespace rocshmem {

class AddressRecordTestFixture : public ::testing::Test
{
  protected:
    AddressRecord split_ {reinterpret_cast<char*>(0x200), 0x80};
    AddressRecord combine_1_ {reinterpret_cast<char*>(0x120), 0x20};
    AddressRecord combine_2_ {reinterpret_cast<char*>(0x140), 0x20};
    AddressRecord bad_addr_ {nullptr, 0x20};
    AddressRecord bad_size_ {reinterpret_cast<char*>(0x800), 0x0};
};

} // namespace rocshmem

#endif // ROCSHMEM_ADDRESS_RECORD_GTEST_HPP
