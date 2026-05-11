// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "lib/aqlprofile/core/last_error.hpp"
#include "lib/aqlprofile/core/logger.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <string_view>
#include <thread>

namespace aql_profile
{
TEST(ErrorApiTest, SetAndGetLastError)
{
    set_last_error("hello world");
    EXPECT_STREQ(get_last_error(), "hello world");
}

TEST(ErrorApiTest, TruncatesLongMessage)
{
    std::string long_msg(2048, 'A');
    set_last_error(long_msg);
    EXPECT_LT(std::strlen(get_last_error()), long_msg.size());
    EXPECT_GT(std::strlen(get_last_error()), 0u);
}

TEST(ErrorApiTest, SetLastErrorOverwrites)
{
    set_last_error("first message");
    EXPECT_STREQ(get_last_error(), "first message");

    set_last_error("second message");
    EXPECT_STREQ(get_last_error(), "second message");
}

TEST(ErrorApiTest, EmptyMessage)
{
    set_last_error("something");
    set_last_error("");
    EXPECT_STREQ(get_last_error(), "");
}

TEST(ErrorApiTest, ThreadLocal)
{
    set_last_error("main_thread");

    std::string child_msg;
    std::thread t([&]() {
        set_last_error("child_thread");
        child_msg = std::string(get_last_error());
    });
    t.join();

    EXPECT_STREQ(get_last_error(), "main_thread");
    EXPECT_EQ(child_msg, "child_thread");
}

TEST(ErrorApiTest, ErrLoggingMacro)
{
    ERR_LOGGING("test error {}", 42);

    auto msg = std::string(get_last_error());
    EXPECT_THAT(msg, testing::HasSubstr("test error 42"));
}

TEST(ErrorApiTest, ErrLoggingPlainString)
{
    ERR_LOGGING("simple error message");
    EXPECT_STREQ(get_last_error(), "simple error message");
}

}  // namespace aql_profile
