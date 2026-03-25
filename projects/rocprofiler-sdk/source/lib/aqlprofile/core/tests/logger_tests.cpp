// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "lib/aqlprofile/core/logger.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <string_view>
#include <thread>

namespace aql_profile
{
class ErrLoggingTest : public ::testing::Test
{
protected:
    void SetUp() override { std::memset(last_error_msg().data(), 0, MSG_BUF_LEN); }

    std::string_view msg() { return {last_error_msg().data()}; }
};

TEST_F(ErrLoggingTest, CapturesStringLiteral)
{
    ERR_LOGGING << "something broke";

    EXPECT_NE(msg().find("something broke"), std::string_view::npos);
}

TEST_F(ErrLoggingTest, ChainedFmtFormat)
{
    ERR_LOGGING << fmt::format("SE({}) ", 3) << ":: " << fmt::format("size({}/{})", 100, 200);

    EXPECT_NE(msg().find("SE(3) :: size(100/200)"), std::string_view::npos);
}

TEST_F(ErrLoggingTest, CapturesFmtFormatMultipleArgs)
{
    ERR_LOGGING << fmt::format("SE({}) size({}/{})", 3, 100, 200);

    EXPECT_NE(msg().find("SE(3) size(100/200)"), std::string_view::npos);
}

TEST_F(ErrLoggingTest, IncludesFunctionName)
{
    ERR_LOGGING << "error";

    EXPECT_NE(msg().find("TestBody():"), std::string_view::npos);
}

TEST_F(ErrLoggingTest, OverwritesPreviousError)
{
    ERR_LOGGING << "first";
    EXPECT_NE(msg().find("first"), std::string_view::npos);

    ERR_LOGGING << "second";
    EXPECT_NE(msg().find("second"), std::string_view::npos);
    EXPECT_EQ(msg().find("first"), std::string_view::npos);
}

TEST_F(ErrLoggingTest, ThreadLocal)
{
    ERR_LOGGING << "main thread";

    std::string other;
    std::thread t([&other]() {
        ERR_LOGGING << "other thread";
        other = last_error_msg().data();
    });
    t.join();

    EXPECT_NE(msg().find("main thread"), std::string_view::npos);
    EXPECT_NE(other.find("other thread"), std::string::npos);
}

TEST_F(ErrLoggingTest, TruncatesLongMessage)
{
    std::string big(MSG_BUF_LEN + 100, 'X');
    ERR_LOGGING << std::string_view{big};

    EXPECT_EQ(std::strlen(last_error_msg().data()), MSG_BUF_LEN - 1);
}

TEST_F(ErrLoggingTest, NullTerminated)
{
    ERR_LOGGING << "test";

    auto len = std::strlen(last_error_msg().data());
    EXPECT_EQ(last_error_msg()[len], '\0');
}

TEST_F(ErrLoggingTest, ExceptionWhat)
{
    try
    {
        throw std::runtime_error("throwaway");
    } catch(const std::exception& e)
    {
        ERR_LOGGING << e.what();
    }

    EXPECT_NE(msg().find("throwaway"), std::string_view::npos);
}

}  // namespace aql_profile
