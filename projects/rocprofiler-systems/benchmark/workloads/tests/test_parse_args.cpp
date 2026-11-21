// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace
{
workload::args
parse_size(std::string value)
{
    std::string        name = "prog";
    std::string        key  = "--size";
    std::vector<char*> argv = { name.data(), key.data(), value.data() };
    return workload::parse_args(static_cast<int>(argv.size()), argv.data(), {}, "prog",
                                "");
}
}  // namespace

class parse_args_bad_value : public ::testing::TestWithParam<const char*>
{};

TEST_P(parse_args_bad_value, exits_with_code_2)
{
    EXPECT_EXIT(parse_size(GetParam()), ::testing::ExitedWithCode(2), "usage:");
}

INSTANTIATE_TEST_SUITE_P(parse_args, parse_args_bad_value,
                         ::testing::Values("", "-1", "+3", " 3", "3x",
                                           "99999999999999999999999"));

TEST(parse_args, accepts_largest_value)
{
    EXPECT_EQ(parse_size("18446744073709551615").size, UINT64_MAX);
}
