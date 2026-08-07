/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <gtest/gtest.h>

namespace hipFileTest {

inline constexpr unsigned char kByteEntry     = 0xFF; // Initial value
inline constexpr unsigned char kByteModified  = 0x22;
inline constexpr unsigned char kByteDevSlack  = 0xAA;
inline constexpr unsigned char kByteFileSlack = 0x55;

// Asserts the every-Nth-byte modify policy over n Data bytes: byte i equals
// kByteModified iff (i % modify_stride == 0), otherwise it still holds kByteEntry.
inline void
assertBytesModified(const unsigned char *arr, size_t n, size_t modify_stride)
{
    assert(modify_stride != 0 && "modify_stride must be >= 1");
    for (size_t i = 0; i < n; ++i) {
        const unsigned char want = (i % modify_stride == 0) ? kByteModified : kByteEntry;
        ASSERT_EQ(want, arr[i]) << "byte modify-policy mismatch at index " << i;
    }
}

// Asserts bytes in [from, to) all equal `value` to verify untouched data was truly untouched.
inline void
assertBytesConstant(const unsigned char *arr, size_t from, size_t to, unsigned char value)
{
    for (size_t i = from; i < to; ++i) {
        ASSERT_EQ(value, arr[i]) << "byte constant region changed at index " << i;
    }
}

} // namespace hipFileTest
