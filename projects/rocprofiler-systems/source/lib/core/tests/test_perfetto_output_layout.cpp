// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/output_layout.hpp"

#include <gtest/gtest.h>

namespace
{
using rocprofsys::core::parse_perfetto_output_layout;
using rocprofsys::core::perfetto_output_layout;
using rocprofsys::core::writes_merged_file;
using rocprofsys::core::writes_per_process_files;
}  // namespace

TEST(perfetto_output_layout, parses_canonical_values)
{
    EXPECT_EQ(parse_perfetto_output_layout("single_file_only"),
              perfetto_output_layout::single_file_only);
    EXPECT_EQ(parse_perfetto_output_layout("per_process_only"),
              perfetto_output_layout::per_process_only);
    EXPECT_EQ(parse_perfetto_output_layout("full"), perfetto_output_layout::full);
}

TEST(perfetto_output_layout, parses_legacy_values)
{
    EXPECT_EQ(parse_perfetto_output_layout("single_file"),
              perfetto_output_layout::single_file_only);
    EXPECT_EQ(parse_perfetto_output_layout("per_process"),
              perfetto_output_layout::per_process_only);
}

TEST(perfetto_output_layout, falls_back_to_full_for_unknown_values)
{
    EXPECT_EQ(parse_perfetto_output_layout(""), perfetto_output_layout::full);
    EXPECT_EQ(parse_perfetto_output_layout("unexpected"), perfetto_output_layout::full);
}

TEST(perfetto_output_layout, exposes_file_write_intent)
{
    EXPECT_FALSE(writes_per_process_files(perfetto_output_layout::single_file_only));
    EXPECT_TRUE(writes_merged_file(perfetto_output_layout::single_file_only));

    EXPECT_TRUE(writes_per_process_files(perfetto_output_layout::per_process_only));
    EXPECT_FALSE(writes_merged_file(perfetto_output_layout::per_process_only));

    EXPECT_TRUE(writes_per_process_files(perfetto_output_layout::full));
    EXPECT_TRUE(writes_merged_file(perfetto_output_layout::full));
}
