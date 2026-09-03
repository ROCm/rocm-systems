// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/range_replay/digest.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace digest = ::rocprofiler::range_replay::digest;

TEST(range_replay_digest, identical_buffers_hash_identically)
{
    const auto lhs = std::array<uint8_t, 8>{1, 2, 3, 4, 5, 6, 7, 8};
    const auto rhs = std::array<uint8_t, 8>{1, 2, 3, 4, 5, 6, 7, 8};

    EXPECT_EQ(digest::hash_bytes(lhs.data(), lhs.size()),
              digest::hash_bytes(rhs.data(), rhs.size()));
}

TEST(range_replay_digest, a_single_changed_byte_changes_the_hash)
{
    auto       buffer = std::array<uint8_t, 32>{};
    const auto before = digest::hash_bytes(buffer.data(), buffer.size());

    buffer[17] = 1;

    EXPECT_NE(before, digest::hash_bytes(buffer.data(), buffer.size()));
}

TEST(range_replay_digest, length_is_part_of_the_hash)
{
    const auto buffer = std::array<uint8_t, 4>{0, 0, 0, 0};

    EXPECT_NE(digest::hash_bytes(buffer.data(), 2), digest::hash_bytes(buffer.data(), 4));
}

TEST(range_replay_digest, empty_and_null_inputs_are_well_defined)
{
    const auto buffer = std::array<uint8_t, 1>{7};

    EXPECT_EQ(digest::hash_bytes(nullptr, 0), digest::hash_bytes(buffer.data(), 0));
}

TEST(range_replay_digest, matching_digests_report_no_divergence)
{
    const auto lhs = digest::region_digests_t{1, 2, 3};
    const auto rhs = digest::region_digests_t{1, 2, 3};

    EXPECT_EQ(digest::count_divergent(lhs, rhs), 0U);
}

TEST(range_replay_digest, each_differing_region_is_counted)
{
    const auto lhs = digest::region_digests_t{1, 2, 3, 4};
    const auto rhs = digest::region_digests_t{1, 9, 3, 9};

    EXPECT_EQ(digest::count_divergent(lhs, rhs), 2U);
}

TEST(range_replay_digest, a_region_count_mismatch_is_fully_divergent)
{
    // Nothing can be concluded region by region when the two digests do not describe the same set
    // of regions, so the larger count is reported rather than a partial comparison.
    EXPECT_EQ(digest::count_divergent(digest::region_digests_t{1, 2},
                                      digest::region_digests_t{1, 2, 3, 4, 5}),
              5U);
}
