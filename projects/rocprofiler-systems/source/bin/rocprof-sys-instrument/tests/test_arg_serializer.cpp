// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "arg_serializer.hpp"
#include "core/common_types.hpp"  // process_arguments_string
#include "core/demangler.hpp"     // for platform-independent type name lookup

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <string_view>

namespace
{
// process_arguments_string reverses the serialization process
rocprofsys::function_args_t
parse(const std::string& s)
{
    return rocprofsys::process_arguments_string(s);
}
}  // namespace

class arg_serializer_test : public ::testing::Test
{};

TEST_F(arg_serializer_test, EmptyWhenNoArgs)
{
    EXPECT_EQ(rocprofsys_get_serialized_args(), std::string{});
}

TEST_F(arg_serializer_test, SingleEnabledStringArg)
{
    auto out    = rocprofsys_get_serialized_args("object", "libfoo.so", true);
    auto parsed = parse(out);

    ASSERT_EQ(parsed.size(), 1u);
    EXPECT_EQ(parsed[0].arg_number, 0u);
    EXPECT_EQ(parsed[0].arg_type, "string");
    EXPECT_EQ(parsed[0].arg_name, "object");
    EXPECT_EQ(parsed[0].arg_value, "libfoo.so");
}

TEST_F(arg_serializer_test, DisabledArgIsSkipped)
{
    auto out = rocprofsys_get_serialized_args("object", "anything", false);
    EXPECT_TRUE(out.empty());
}

TEST_F(arg_serializer_test, EmptyNameIsSkipped)
{
    auto out = rocprofsys_get_serialized_args("", "value", true);
    EXPECT_TRUE(out.empty());
}

TEST_F(arg_serializer_test, NullptrNameIsSkippedAndDoesNotCrash)
{
    const char* null_name = nullptr;
    auto        out       = rocprofsys_get_serialized_args(null_name, "value", true);
    EXPECT_TRUE(out.empty());
}

TEST_F(arg_serializer_test, NullptrCStringValueYieldsEmptyValueField)
{
    const char* null_value = nullptr;
    auto        out        = rocprofsys_get_serialized_args("name", null_value, true);
    auto        parsed     = parse(out);

    ASSERT_EQ(parsed.size(), 1u);
    EXPECT_EQ(parsed[0].arg_name, "name");
    EXPECT_EQ(parsed[0].arg_type, "string");
    EXPECT_TRUE(parsed[0].arg_value.empty());
}

TEST_F(arg_serializer_test, IntegralValueIsStringifiedWithDemangledType)
{
    auto out    = rocprofsys_get_serialized_args("count", 42, true);
    auto parsed = parse(out);

    ASSERT_EQ(parsed.size(), 1u);
    EXPECT_EQ(parsed[0].arg_name, "count");
    EXPECT_EQ(parsed[0].arg_value, "42");
    EXPECT_EQ(parsed[0].arg_type, "int");
}

TEST_F(arg_serializer_test, FloatingPointValueIsStringified)
{
    auto out    = rocprofsys_get_serialized_args("ratio", 3.5, true);
    auto parsed = parse(out);

    ASSERT_EQ(parsed.size(), 1u);
    EXPECT_EQ(parsed[0].arg_name, "ratio");
    EXPECT_EQ(parsed[0].arg_value, "3.5");
    EXPECT_EQ(parsed[0].arg_type, "double");
}

TEST_F(arg_serializer_test, MixedTriplesIndexesOnlyEnabled)
{
    auto out    = rocprofsys_get_serialized_args(  //
        "a", "alpha", true,                     // expected idx 0
        "b", "beta", false,                     // skipped, idx not consumed
        "c", "gamma", true);                    // expected idx 1
    auto parsed = parse(out);

    ASSERT_EQ(parsed.size(), 2u);
    EXPECT_EQ(parsed[0].arg_number, 0u);
    EXPECT_EQ(parsed[0].arg_name, "a");
    EXPECT_EQ(parsed[0].arg_value, "alpha");
    EXPECT_EQ(parsed[1].arg_number, 1u);
    EXPECT_EQ(parsed[1].arg_name, "c");
    EXPECT_EQ(parsed[1].arg_value, "gamma");
}

TEST_F(arg_serializer_test, AcceptsStdStringAndStringView)
{
    std::string      s_name  = "s_name";
    std::string      s_val   = "s_val";
    std::string_view sv_name = "sv_name";
    std::string_view sv_val  = "sv_val";

    auto out    = rocprofsys_get_serialized_args(  //
        s_name, s_val, true,                    //
        sv_name, sv_val, true);
    auto parsed = parse(out);

    ASSERT_EQ(parsed.size(), 2u);
    EXPECT_EQ(parsed[0].arg_type, "string");
    EXPECT_EQ(parsed[0].arg_name, "s_name");
    EXPECT_EQ(parsed[0].arg_value, "s_val");
    EXPECT_EQ(parsed[1].arg_type, "string");
    EXPECT_EQ(parsed[1].arg_name, "sv_name");
    EXPECT_EQ(parsed[1].arg_value, "sv_val");
}

TEST_F(arg_serializer_test, ExactDelimiterFormatForSingleArg)
{
    // Implementation-detail check: the produced format is
    //   <idx>;;<type>;;<name>;;<value>;;
    // If this fails because the delimiter changed, also update
    // process_arguments_string in core/common_types.hpp and any
    // downstream consumers.
    auto out = rocprofsys_get_serialized_args("k", "v", true);
    EXPECT_EQ(out, "0;;string;;k;;v;;");
}

TEST_F(arg_serializer_test, AllDisabledYieldsEmpty)
{
    auto out = rocprofsys_get_serialized_args(  //
        "a", "alpha", false,                    //
        "b", 1, false,                          //
        "c", 2.0, false);
    EXPECT_TRUE(out.empty());
}
