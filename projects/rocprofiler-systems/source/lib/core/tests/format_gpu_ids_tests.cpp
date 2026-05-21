// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/text_layout.hpp"

#include <cstddef>
#include <string>
#include <vector>

TEST(format_gpu_ids, empty_returns_empty_string)
{
    EXPECT_EQ(rocprofsys::output::format_gpu_ids({}), "");
}

TEST(format_gpu_ids, single_id_renders_as_colon_n)
{
    EXPECT_EQ(rocprofsys::output::format_gpu_ids({ 0 }), ":0");
    EXPECT_EQ(rocprofsys::output::format_gpu_ids({ 7 }), ":7");
}

TEST(format_gpu_ids, two_ids_render_comma_separated)
{
    EXPECT_EQ(rocprofsys::output::format_gpu_ids({ 0, 1 }), ":0,1");
    EXPECT_EQ(rocprofsys::output::format_gpu_ids({ 0, 3 }), ":0,3");
}

TEST(format_gpu_ids, three_contiguous_compresses_to_range)
{
    EXPECT_EQ(rocprofsys::output::format_gpu_ids({ 0, 1, 2 }), ":0-2");
    EXPECT_EQ(rocprofsys::output::format_gpu_ids({ 0, 1, 2, 3, 4, 5, 6, 7 }), ":0-7");
}

TEST(format_gpu_ids, mixed_contiguous_and_singleton)
{
    EXPECT_EQ(rocprofsys::output::format_gpu_ids({ 0, 1, 2, 5 }), ":0-2,5");
    EXPECT_EQ(rocprofsys::output::format_gpu_ids({ 0, 2, 3, 4, 7 }), ":0,2-4,7");
}

TEST(format_gpu_ids, unsorted_input_is_sorted_first)
{
    EXPECT_EQ(rocprofsys::output::format_gpu_ids({ 3, 0, 1, 2 }), ":0-3");
    EXPECT_EQ(rocprofsys::output::format_gpu_ids({ 5, 0 }), ":0,5");
}

TEST(format_gpu_ids, duplicates_are_uniqued)
{
    EXPECT_EQ(rocprofsys::output::format_gpu_ids({ 0, 0, 1, 1, 2, 2 }), ":0-2");
}

TEST(format_gpu_ids, truncates_pathological_list)
{
    std::vector<int> ids;
    for(int i = 0; i < 64; i += 2)
        ids.push_back(i);
    ASSERT_EQ(ids.size(), 32u);

    const auto rendered = rocprofsys::output::format_gpu_ids(ids);
    EXPECT_NE(rendered.find("...(+16 more)"), std::string::npos);
    EXPECT_NE(rendered.find(":0,2,4,6,"), std::string::npos);
}
