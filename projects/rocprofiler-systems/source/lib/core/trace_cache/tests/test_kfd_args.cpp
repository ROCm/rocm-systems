// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/common_types.hpp"

#include <gtest/gtest.h>

#include <string>

using namespace rocprofsys;

TEST(kfd_args_test, page_fault_args_roundtrip)
{
    auto args_str = std::string("0;;uint64_t;;address;;0x7f4a00001000;;"
                                "1;;string;;agent;;5;;");

    auto args = process_arguments_string(args_str);
    ASSERT_EQ(args.size(), 2u);

    EXPECT_EQ(args[0].arg_number, 0u);
    EXPECT_EQ(args[0].arg_type, "uint64_t");
    EXPECT_EQ(args[0].arg_name, "address");
    EXPECT_EQ(args[0].arg_value, "0x7f4a00001000");

    EXPECT_EQ(args[1].arg_number, 1u);
    EXPECT_EQ(args[1].arg_type, "string");
    EXPECT_EQ(args[1].arg_name, "agent");
    EXPECT_EQ(args[1].arg_value, "5");

    auto rebuilt  = get_args_string(args);
    auto reparsed = process_arguments_string(rebuilt);
    ASSERT_EQ(reparsed.size(), 2u);
    EXPECT_EQ(reparsed[0].arg_name, "address");
    EXPECT_EQ(reparsed[1].arg_name, "agent");
}

TEST(kfd_args_test, page_migrate_args_parse)
{
    auto args_str = std::string("0;;uint64_t;;start_address;;0x7fb100000000;;"
                                "1;;uint64_t;;end_address;;0x7fb100200000;;"
                                "2;;string;;src_agent;;1;;"
                                "3;;string;;dst_agent;;2;;"
                                "4;;string;;prefetch_agent;;null;;"
                                "5;;string;;preferred_agent;;null;;"
                                "6;;int;;error_code;;0;;");

    auto args = process_arguments_string(args_str);
    ASSERT_EQ(args.size(), 7u);

    EXPECT_EQ(args[0].arg_name, "start_address");
    EXPECT_EQ(args[0].arg_type, "uint64_t");

    EXPECT_EQ(args[1].arg_name, "end_address");
    EXPECT_EQ(args[1].arg_type, "uint64_t");

    EXPECT_EQ(args[2].arg_name, "src_agent");
    EXPECT_EQ(args[2].arg_value, "1");

    EXPECT_EQ(args[3].arg_name, "dst_agent");
    EXPECT_EQ(args[3].arg_value, "2");

    EXPECT_EQ(args[4].arg_name, "prefetch_agent");
    EXPECT_EQ(args[4].arg_value, "null");

    EXPECT_EQ(args[5].arg_name, "preferred_agent");
    EXPECT_EQ(args[5].arg_value, "null");

    EXPECT_EQ(args[6].arg_name, "error_code");
    EXPECT_EQ(args[6].arg_type, "int");
    EXPECT_EQ(args[6].arg_value, "0");
}

TEST(kfd_args_test, queue_args_parse)
{
    auto args_str = std::string("0;;string;;agent;;3;;");

    auto args = process_arguments_string(args_str);
    ASSERT_EQ(args.size(), 1u);

    EXPECT_EQ(args[0].arg_number, 0u);
    EXPECT_EQ(args[0].arg_type, "string");
    EXPECT_EQ(args[0].arg_name, "agent");
    EXPECT_EQ(args[0].arg_value, "3");
}

TEST(kfd_args_test, unmap_from_gpu_args_parse)
{
    auto args_str = std::string("0;;string;;agent;;5;;"
                                "1;;uint64_t;;start_address;;0x7f0000000000;;"
                                "2;;uint64_t;;end_address;;0x7f0000100000;;");

    auto args = process_arguments_string(args_str);
    ASSERT_EQ(args.size(), 3u);

    EXPECT_EQ(args[0].arg_name, "agent");
    EXPECT_EQ(args[1].arg_name, "start_address");
    EXPECT_EQ(args[2].arg_name, "end_address");
}

TEST(kfd_args_test, dropped_events_args_parse)
{
    auto args_str = std::string("0;;uint64_t;;count;;42;;");

    auto args = process_arguments_string(args_str);
    ASSERT_EQ(args.size(), 1u);

    EXPECT_EQ(args[0].arg_number, 0u);
    EXPECT_EQ(args[0].arg_type, "uint64_t");
    EXPECT_EQ(args[0].arg_name, "count");
    EXPECT_EQ(args[0].arg_value, "42");
}

TEST(kfd_args_test, empty_args_string)
{
    auto args = process_arguments_string("");
    EXPECT_EQ(args.size(), 0u);
}

TEST(kfd_args_test, malformed_args_string_throws)
{
    EXPECT_THROW(process_arguments_string("0;;uint64_t;;address;;"),
                 std::invalid_argument);
}

TEST(kfd_args_test, null_agent_value_preserved)
{
    auto args_str = std::string("0;;string;;agent;;null;;");

    auto args = process_arguments_string(args_str);
    ASSERT_EQ(args.size(), 1u);
    EXPECT_EQ(args[0].arg_value, "null");
}
