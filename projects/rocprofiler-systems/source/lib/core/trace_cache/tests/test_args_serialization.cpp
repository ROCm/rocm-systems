// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/common_types.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>

using namespace rocprofsys;

// Builds the args the way the instrumentor does for a single source_object arg
// (see module_function::operator()).
namespace
{
std::string
serialize_source_object(const std::string& source_object)
{
    function_args_t args{};
    if(!source_object.empty())
        args.push_back({ 0U, "string", "source_object", source_object });
    return get_args_string(args);
}
}  // namespace

TEST(args_serialization_test, empty_args_yields_empty_string)
{
    EXPECT_TRUE(get_args_string(function_args_t{}).empty());
}

TEST(args_serialization_test, single_arg_exact_wire_format)
{
    function_args_t args{ { 0U, "string", "source_object", "libfoo.so" } };
    EXPECT_EQ(get_args_string(args), "0;;string;;source_object;;libfoo.so;;");
}

TEST(args_serialization_test, multiple_args_are_concatenated)
{
    function_args_t args{ { 0U, "string", "object", "libomp.so" },
                          { 1U, "int", "count", "42" } };
    EXPECT_EQ(get_args_string(args), "0;;string;;object;;libomp.so;;1;;int;;count;;42;;");
}

TEST(args_serialization_test, get_args_string_roundtrips_through_parser)
{
    function_args_t args{ { 0U, "string", "source_object", "libfoo.so" },
                          { 1U, "std::uint64_t", "addr", "0x7f00" } };

    auto parsed = process_arguments_string(get_args_string(args));

    ASSERT_EQ(parsed.size(), args.size());
    for(size_t i = 0; i < args.size(); ++i)
    {
        EXPECT_EQ(parsed[i].arg_number, args[i].arg_number);
        EXPECT_EQ(parsed[i].arg_type, args[i].arg_type);
        EXPECT_EQ(parsed[i].arg_name, args[i].arg_name);
        EXPECT_EQ(parsed[i].arg_value, args[i].arg_value);
    }
}

// --- instrumentor producer pattern (module_function::operator()) ------------

TEST(args_serialization_test, source_object_present_serializes)
{
    auto out    = serialize_source_object("minimal-recursion");
    auto parsed = process_arguments_string(out);

    ASSERT_EQ(parsed.size(), 1u);
    EXPECT_EQ(parsed[0].arg_number, 0u);
    EXPECT_EQ(parsed[0].arg_type, "string");
    EXPECT_EQ(parsed[0].arg_name, "source_object");
    EXPECT_EQ(parsed[0].arg_value, "minimal-recursion");
}

TEST(args_serialization_test, empty_source_object_yields_empty_string)
{
    // An empty object name (e.g. the main binary / unresolved module) must
    // produce no args so the instrumentor falls back to rocprofsys_push_trace.
    EXPECT_TRUE(serialize_source_object("").empty());
}

// --- process_arguments_string robustness (malformed index guards) -----------

TEST(args_serialization_test, non_numeric_index_throws)
{
    EXPECT_THROW(process_arguments_string("x;;string;;name;;value;;"),
                 std::invalid_argument);
}

TEST(args_serialization_test, trailing_garbage_after_index_throws)
{
    EXPECT_THROW(process_arguments_string("0abc;;string;;name;;value;;"),
                 std::invalid_argument);
}

TEST(args_serialization_test, negative_index_throws)
{
    EXPECT_THROW(process_arguments_string("-1;;string;;name;;value;;"),
                 std::invalid_argument);
}

TEST(args_serialization_test, overflowing_index_throws)
{
    auto too_big = std::to_string(
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1);
    EXPECT_THROW(process_arguments_string(too_big + ";;string;;name;;value;;"),
                 std::invalid_argument);
}

TEST(args_serialization_test, wrong_token_count_throws)
{
    EXPECT_THROW(process_arguments_string("0;;string;;name;;"), std::invalid_argument);
}

TEST(args_serialization_test, max_uint32_index_is_accepted)
{
    auto max_idx = std::to_string(std::numeric_limits<std::uint32_t>::max());
    auto parsed  = process_arguments_string(max_idx + ";;string;;name;;value;;");

    ASSERT_EQ(parsed.size(), 1u);
    EXPECT_EQ(parsed[0].arg_number, std::numeric_limits<std::uint32_t>::max());
}
