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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "extra_counters.hpp"

#include <gtest/gtest.h>

#include <string>

namespace
{
using rocprofiler::tool::reuse_extra_counters_snapshot;
}

TEST(extra_counters, same_path_reuses_immutable_snapshot)
{
    auto refreshed_contents = std::string{};

    EXPECT_TRUE(reuse_extra_counters_snapshot(
        "/definitions.yaml", refreshed_contents, "/definitions.yaml", "original contents"));
    EXPECT_EQ(refreshed_contents, "original contents");
}

TEST(extra_counters, same_path_reuses_empty_snapshot)
{
    auto refreshed_contents = std::string{};

    EXPECT_TRUE(reuse_extra_counters_snapshot(
        "/empty-definitions.yaml", refreshed_contents, "/empty-definitions.yaml", ""));
    EXPECT_TRUE(refreshed_contents.empty());
}

TEST(extra_counters, explicit_contents_are_not_replaced)
{
    auto refreshed_contents = std::string{"changed contents"};

    EXPECT_FALSE(reuse_extra_counters_snapshot(
        "", refreshed_contents, "/definitions.yaml", "original contents"));
    EXPECT_EQ(refreshed_contents, "changed contents");
    EXPECT_NE(refreshed_contents, "original contents");
}

TEST(extra_counters, changed_path_is_loaded_normally)
{
    auto refreshed_contents = std::string{};

    EXPECT_FALSE(reuse_extra_counters_snapshot(
        "/new-definitions.yaml", refreshed_contents, "/definitions.yaml", "original contents"));
    EXPECT_TRUE(refreshed_contents.empty());
}
