/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "environment.h"
#include "hipfile-warnings.h"
#include "msys.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
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

TEST(HipFileEnvironment, GetUnsignedIntReturnsNulloptIfNotSet)
{
    StrictMock<MSys> msys;

    EXPECT_CALL(msys, getenv).WillOnce(Return(nullptr));
    ASSERT_EQ(hipFile::Environment::get<unsigned int>(""), nullopt);
}

TEST(HipFileEnvironment, GetUnsignedIntReturnsNulloptIfValueIsInvalid)
{
    StrictMock<MSys> msys;

    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("blah")));
    ASSERT_EQ(hipFile::Environment::get<unsigned int>(""), nullopt);
    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("")));
    ASSERT_EQ(hipFile::Environment::get<unsigned int>(""), nullopt);
    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("1abc")));
    ASSERT_EQ(hipFile::Environment::get<unsigned int>(""), nullopt);
}

TEST(HipFileEnvironment, GetUnsignedIntReturnsIntIfValueIsInt)
{
    StrictMock<MSys> msys;

    EXPECT_CALL(msys, getenv).WillOnce(Return(const_cast<char *>("0")));
    ASSERT_EQ(hipFile::Environment::get<unsigned int>(""), make_optional<>(0));
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
