// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "kernel_filter_range.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <unordered_set>

namespace
{
using set_t = std::unordered_set<size_t>;
using rocprofiler::tool::get_kernel_filter_range;
}  // namespace

TEST(kernel_filter_range, mixed_singles_and_ranges)
{
    // singles and an inclusive nested range
    EXPECT_EQ(get_kernel_filter_range("[1, 3, 5, [8-12]]"), (set_t{1, 3, 5, 8, 9, 10, 11, 12}));
}

TEST(kernel_filter_range, bare_range_without_brackets)
{
    EXPECT_EQ(get_kernel_filter_range("3-6"), (set_t{3, 4, 5, 6}));
}

TEST(kernel_filter_range, single_bracketed_range)
{
    EXPECT_EQ(get_kernel_filter_range("[8-12]"), (set_t{8, 9, 10, 11, 12}));
}

TEST(kernel_filter_range, single_integer) { EXPECT_EQ(get_kernel_filter_range("3"), (set_t{3})); }

TEST(kernel_filter_range, empty_string_means_all_iterations)
{
    // empty set is the sentinel callers treat as "profile all iterations"
    EXPECT_TRUE(get_kernel_filter_range("").empty());
}

TEST(kernel_filter_range, comma_join_form_matches_bracket_form)
{
    // the CLI joins multi-token ranges (e.g. `--kernel-iteration-range 1 2`)
    // with ", " before parsing; assert that form is equivalent to "[1-2]"
    EXPECT_EQ(get_kernel_filter_range("1, 2"), get_kernel_filter_range("[1-2]"));
    EXPECT_EQ(get_kernel_filter_range("1, 2"), (set_t{1, 2}));
}

TEST(kernel_filter_range, duplicate_indices_collapse)
{
    // overlapping specifications collapse to a set (no duplicates)
    EXPECT_EQ(get_kernel_filter_range("[2, [1-3]]"), (set_t{1, 2, 3}));
}

TEST(kernel_filter_range, malformed_range_aborts)
{
    EXPECT_DEATH(get_kernel_filter_range("[1-]"), "bad range format");
}

TEST(kernel_filter_range, non_integer_token_aborts)
{
    EXPECT_DEATH(get_kernel_filter_range("[a]"), "Non-integer value detected");
}
