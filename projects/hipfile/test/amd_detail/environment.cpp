/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "environment.h"
#include "hipfile-warnings.h"
#include "msys.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <limits>
#include <memory>

using namespace hipFile;
using namespace testing;
using namespace std;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

TEST(HipFileEnvironment, GetBoolReturnsNulloptIfNotSet)
{
    StrictMock<MSys> msys;

    EXPECT_CALL(msys, getenv).WillOnce(Return(nullptr));
    ASSERT_EQ(hipFile::Environment::get<bool>(""), nullopt);
}

TEST(HipFileEnvironment, GetBoolReturnsNulloptIfValueIsInvalid)
{
    StrictMock<MSys> msys;

    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("blah")));
    ASSERT_EQ(hipFile::Environment::get<bool>(""), nullopt);
    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("")));
    ASSERT_EQ(hipFile::Environment::get<bool>(""), nullopt);
}

TEST(HipFileEnvironment, GetBoolReturnsOptionalTrueIfTrue)
{
    StrictMock<MSys> msys;

    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("true")));
    ASSERT_EQ(hipFile::Environment::get<bool>(""), make_optional<>(true));

    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("TRUE")));
    ASSERT_EQ(hipFile::Environment::get<bool>(""), make_optional<>(true));

    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("TrUe")));
    ASSERT_EQ(hipFile::Environment::get<bool>(""), make_optional<>(true));
}

TEST(HipFileEnvironment, GetBoolReturnsOptionalFalseIfFalse)
{
    StrictMock<MSys> msys;

    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("false")));
    ASSERT_EQ(hipFile::Environment::get<bool>(""), make_optional<>(false));

    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("FALSE")));
    ASSERT_EQ(hipFile::Environment::get<bool>(""), make_optional<>(false));

    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("FaLsE")));
    ASSERT_EQ(hipFile::Environment::get<bool>(""), make_optional<>(false));
}

TEST(HipFileEnvironment, GetIntReturnsNulloptIfNotSet)
{
    StrictMock<MSys> msys;

    EXPECT_CALL(msys, getenv).WillOnce(Return(nullptr));
    ASSERT_EQ(hipFile::Environment::get<unsigned int>(""), nullopt);
}

TEST(HipFileEnvironment, GetIntReturnsNulloptIfValueIsInvalid)
{
    StrictMock<MSys> msys;

    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("blah")));
    ASSERT_EQ(hipFile::Environment::get<unsigned int>(""), nullopt);
    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("")));
    ASSERT_EQ(hipFile::Environment::get<unsigned int>(""), nullopt);
    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("1abc")));
    ASSERT_EQ(hipFile::Environment::get<unsigned int>(""), nullopt);
}

TEST(HipFileEnvironment, GetIntReturnsIntIfValueIsInt)
{
    StrictMock<MSys> msys;

    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("0")));
    ASSERT_EQ(hipFile::Environment::get<unsigned int>(""), make_optional<>(0));
}

TEST(HipFileEnvironment, GetIntReturnsResultOutOfRange)
{
    StrictMock<MSys> msys;

    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("999999999999999999999999")));
    ASSERT_EQ(hipFile::Environment::get<unsigned int>(""), std::numeric_limits<unsigned int>::max());
}

TEST(HipFileEnvironment, GetIntReturnsNegative)
{
    StrictMock<MSys> msys;

    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("-1")));
    ASSERT_EQ(hipFile::Environment::get<int>(""), make_optional<>(-1));
}

TEST(HipFileEnvironment, GetIntegralOverloads)
{
    StrictMock<MSys> msys;

    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("1")));
    ASSERT_EQ(hipFile::Environment::get<unsigned int>(""), make_optional<>(1));
    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("2")));
    ASSERT_EQ(hipFile::Environment::get<unsigned long>(""), make_optional<>(2ul));
    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("3")));
    ASSERT_EQ(hipFile::Environment::get<unsigned short>(""), make_optional<>(static_cast<unsigned short>(3)));
    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("4")));
    ASSERT_EQ(hipFile::Environment::get<int>(""), make_optional<>(4));
    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("5")));
    ASSERT_EQ(hipFile::Environment::get<long>(""), make_optional<>(5l));
    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("6")));
    ASSERT_EQ(hipFile::Environment::get<short>(""), make_optional<>(static_cast<short>(6)));
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
