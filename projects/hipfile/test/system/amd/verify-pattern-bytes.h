/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

namespace hipFileTest {

inline constexpr uint8_t kByteEntry     = 0xFF; // Initial value
inline constexpr uint8_t kByteModified  = 0x22;
inline constexpr uint8_t kByteDevSlack  = 0xAA;
inline constexpr uint8_t kByteFileSlack = 0x55;

// Asserts the every-Nth-byte modify policy over n Data bytes: byte i equals
// kByteModified iff (i % modify_stride == 0), otherwise it still holds kByteEntry.
inline void
assertBytesModified(const uint8_t *arr, size_t n, size_t modify_stride)
{
    ASSERT_NE(0U, modify_stride) << "modify_stride must be >= 1";
    for (size_t i = 0; i < n; ++i) {
        const uint8_t want = (i % modify_stride == 0) ? kByteModified : kByteEntry;
        ASSERT_EQ(want, arr[i]) << "byte modify-policy mismatch at index " << i;
    }
}

// Asserts bytes in [from, to) all equal `value` to verify untouched data was truly untouched.
inline void
assertBytesConstant(const uint8_t *arr, size_t from, size_t to, uint8_t value)
{
    for (size_t i = from; i < to; ++i) {
        ASSERT_EQ(value, arr[i]) << "byte constant region changed at index " << i;
    }
}

} // namespace hipFileTest
