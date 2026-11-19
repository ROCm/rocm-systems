// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/result.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

using namespace rocprofsys::common;

namespace
{
enum class test_error
{
    failed
};

using int_result = result<int, test_error>;
}  // namespace

TEST(result_test, success_is_ok_and_holds_value)
{
    const auto res = int_result{ 42 };

    EXPECT_TRUE(res.is_ok());
    EXPECT_FALSE(res.is_error());
    EXPECT_EQ(res.value(), 42);
}

TEST(result_test, failure_is_error_and_value_access_throws)
{
    const auto res = int_result{ failure{ test_error::failed } };

    EXPECT_TRUE(res.is_error());
    EXPECT_FALSE(res.is_ok());
    EXPECT_EQ(res.error(), test_error::failed);
    EXPECT_THROW(static_cast<void>(res.value()), std::logic_error);
}

TEST(result_test, error_access_on_success_throws)
{
    const auto res = int_result{ 42 };

    EXPECT_THROW(static_cast<void>(res.error()), std::logic_error);
}

TEST(result_test, value_or_returns_value_on_success_and_fallback_on_failure)
{
    const auto success = int_result{ 42 };
    const auto bad     = int_result{ failure{ test_error::failed } };

    EXPECT_EQ(success.value_or(7), 42);
    EXPECT_EQ(bad.value_or(7), 7);
}

TEST(result_test, same_value_and_error_type_stays_unambiguous)
{
    using string_result = result<std::string, std::string>;

    const auto success = string_result{ std::string{ "value" } };
    const auto bad     = string_result{ failure{ std::string{ "error" } } };

    ASSERT_TRUE(success.is_ok());
    EXPECT_EQ(success.value(), "value");
    ASSERT_TRUE(bad.is_error());
    EXPECT_EQ(bad.error(), "error");
}
