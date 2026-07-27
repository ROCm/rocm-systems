// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/argparse.hpp"

using rocprofsys::argparse::resolve_output_format;
using rocprofsys::argparse::strset_t;

TEST(output_format, proto_enables_only_perfetto)
{
    const auto sel = resolve_output_format(strset_t{ "proto" });
    EXPECT_TRUE(sel.perfetto);
    EXPECT_FALSE(sel.rocpd);
    EXPECT_FALSE(sel.profile());
    EXPECT_FALSE(sel.json);
    EXPECT_FALSE(sel.text);
}

TEST(output_format, rocpd_disables_perfetto_and_profile)
{
    const auto sel = resolve_output_format(strset_t{ "rocpd" });
    EXPECT_FALSE(sel.perfetto);
    EXPECT_TRUE(sel.rocpd);
    EXPECT_FALSE(sel.profile());
    EXPECT_FALSE(sel.json);
    EXPECT_FALSE(sel.text);
}

TEST(output_format, json_selects_timemory_json)
{
    const auto sel = resolve_output_format(strset_t{ "json" });
    EXPECT_TRUE(sel.profile());
    EXPECT_TRUE(sel.json);
    EXPECT_FALSE(sel.text);
    EXPECT_FALSE(sel.perfetto);
    EXPECT_FALSE(sel.rocpd);
}

TEST(output_format, text_selects_timemory_text)
{
    const auto sel = resolve_output_format(strset_t{ "text" });
    EXPECT_TRUE(sel.profile());
    EXPECT_TRUE(sel.text);
    EXPECT_FALSE(sel.json);
    EXPECT_FALSE(sel.perfetto);
    EXPECT_FALSE(sel.rocpd);
}

TEST(output_format, json_and_text_both_enable_profile)
{
    // two profile sub-formats together; profile() stays true, both flags set
    const auto sel = resolve_output_format(strset_t{ "json", "text" });
    EXPECT_TRUE(sel.json);
    EXPECT_TRUE(sel.text);
    EXPECT_TRUE(sel.profile());
    EXPECT_FALSE(sel.perfetto);
    EXPECT_FALSE(sel.rocpd);
}

TEST(output_format, txt_aliases_text)
{
    const auto from_txt  = resolve_output_format(strset_t{ "txt" });
    const auto from_text = resolve_output_format(strset_t{ "text" });
    EXPECT_TRUE(from_txt.text);
    EXPECT_TRUE(from_txt.profile());
    EXPECT_FALSE(from_txt.json);
    EXPECT_EQ(from_txt.text, from_text.text);
    EXPECT_EQ(from_txt.profile(), from_text.profile());
}

TEST(output_format, proto_and_rocpd_combine)
{
    const auto sel = resolve_output_format(strset_t{ "proto", "rocpd" });
    EXPECT_TRUE(sel.perfetto);
    EXPECT_TRUE(sel.rocpd);
    EXPECT_FALSE(sel.profile());
    EXPECT_FALSE(sel.json);
    EXPECT_FALSE(sel.text);
}

TEST(output_format, proto_and_json_combine)
{
    const auto sel = resolve_output_format(strset_t{ "proto", "json" });
    EXPECT_TRUE(sel.perfetto);
    EXPECT_TRUE(sel.profile());
    EXPECT_TRUE(sel.json);
    EXPECT_FALSE(sel.text);
    EXPECT_FALSE(sel.rocpd);
}

TEST(output_format, empty_selection_disables_all)
{
    const auto sel = resolve_output_format(strset_t{});
    EXPECT_FALSE(sel.perfetto);
    EXPECT_FALSE(sel.rocpd);
    EXPECT_FALSE(sel.profile());
    EXPECT_FALSE(sel.json);
    EXPECT_FALSE(sel.text);
}
