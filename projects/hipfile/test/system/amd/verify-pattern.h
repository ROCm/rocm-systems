/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

namespace hipFileTest {

inline constexpr int32_t kPatternBase = 1;

inline void
fillIndexPattern(int32_t *arr, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        arr[i] = kPatternBase + static_cast<int32_t>(i);
    }
}

// Asserts every element was doubled: arr[i] == 2 * (kPatternBase + i).
inline void
assertDoubledPattern(const int32_t *arr, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        ASSERT_EQ(2 * (kPatternBase + static_cast<int32_t>(i)), arr[i]) << "doubled mismatch at index " << i;
    }
}

// Asserts the every-Nth-element modify policy: element i is doubled iff
// (i % modify_stride == 0), otherwise it still holds the un-doubled seed.
inline void
assertModifiedPattern(const int32_t *arr, size_t n, size_t modify_stride)
{
    ASSERT_NE(0U, modify_stride) << "modify_stride must be >= 1";
    for (size_t i = 0; i < n; ++i) {
        const int32_t seed = kPatternBase + static_cast<int32_t>(i);
        const int32_t want = (i % modify_stride == 0) ? 2 * seed : seed;
        ASSERT_EQ(want, arr[i]) << "modify-policy mismatch at index " << i;
    }
}

} // namespace hipFileTest
