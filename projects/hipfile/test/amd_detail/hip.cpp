/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip.h"
#include "hipfile-warnings.h"
#include "mhip.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

using namespace hipFile;
using namespace testing;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

TEST(Hip, getHipAmdFileRead)
{
    StrictMock<MHip> mhip;

    int   hip_version{1234567890};
    void *hip_pointer{reinterpret_cast<void *>(0x0BADF00D)};

    EXPECT_CALL(mhip, hipRuntimeGetVersion).WillOnce(Return(hip_version));
    EXPECT_CALL(mhip, hipGetProcAddress(StrEq("hipAmdFileRead"), hip_version, 0, nullptr))
        .WillOnce(Return(hip_pointer));

    ASSERT_EQ(getHipAmdFileReadPtr(), hip_pointer);
}

TEST(Hip, getHipAmdFileReadHandlesHipRuntimeGetVersionFailure)
{
    StrictMock<MHip> mhip;

    EXPECT_CALL(mhip, hipRuntimeGetVersion).WillOnce(Throw(Hip::RuntimeError(hipErrorTbd)));

    ASSERT_EQ(getHipAmdFileReadPtr(), nullptr);
}

TEST(Hip, getHipAmdFileReadHandlesHipGetProcAddressFailure)
{
    StrictMock<MHip> mhip;

    int hip_version{1234567890};

    EXPECT_CALL(mhip, hipRuntimeGetVersion).WillOnce(Return(hip_version));
    EXPECT_CALL(mhip, hipGetProcAddress(StrEq("hipAmdFileRead"), hip_version, 0, nullptr))
        .WillOnce(Throw(Hip::RuntimeError(hipErrorTbd)));

    ASSERT_EQ(getHipAmdFileReadPtr(), nullptr);
}

TEST(Hip, getHipAmdFileWrite)
{
    StrictMock<MHip> mhip;

    int   hip_version{1234567890};
    void *hip_pointer{reinterpret_cast<void *>(0x0BADF00D)};

    EXPECT_CALL(mhip, hipRuntimeGetVersion).WillOnce(Return(hip_version));
    EXPECT_CALL(mhip, hipGetProcAddress(StrEq("hipAmdFileWrite"), hip_version, 0, nullptr))
        .WillOnce(Return(hip_pointer));

    ASSERT_EQ(getHipAmdFileWritePtr(), hip_pointer);
}

TEST(Hip, getHipAmdFileWriteHandlesHipRuntimeGetVersionFailure)
{
    StrictMock<MHip> mhip;

    EXPECT_CALL(mhip, hipRuntimeGetVersion).WillOnce(Throw(Hip::RuntimeError(hipErrorTbd)));

    ASSERT_EQ(getHipAmdFileWritePtr(), nullptr);
}

TEST(Hip, getHipAmdFileWriteHandlesHipGetProcAddressFailure)
{
    StrictMock<MHip> mhip;

    int hip_version{1234567890};

    EXPECT_CALL(mhip, hipRuntimeGetVersion).WillOnce(Return(hip_version));
    EXPECT_CALL(mhip, hipGetProcAddress(StrEq("hipAmdFileWrite"), hip_version, 0, nullptr))
        .WillOnce(Throw(Hip::RuntimeError(hipErrorTbd)));

    ASSERT_EQ(getHipAmdFileWritePtr(), nullptr);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
