/*
 * Copyright (C) 2014-2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "KFDOpenCloseKFDTest.hpp"
#include "KFDTestUtil.hpp"

// Before every test from this class fixture, open KFD
void KFDOpenCloseKFDTest::SetUp() {
    ROUTINE_START

    ASSERT_SUCCESS(hsaKmtOpenKFD() );

    ROUTINE_END
}

// After every test from this class fixture, close KFD
void KFDOpenCloseKFDTest::TearDown() {
    ROUTINE_START

    EXPECT_SUCCESS(hsaKmtCloseKFD() );

    ROUTINE_END
}

/* This test does not use class KFDOpenCloseKFDTest but is placed here
 * since it's testing same topic as other test
 * Verify that calling hsaKmtCloseKFD on a closed KFD will return right status
 */
TEST(KFDCloseKFDTest, CloseAClosedKfd ) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_EQ(HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED, hsaKmtCloseKFD());

    TEST_END
}

// Verify that calling hsaKmtCloseKFD on an already opened KFD will return right status
TEST_F(KFDOpenCloseKFDTest, OpenAlreadyOpenedKFD ) {
    TEST_START(TESTPROFILE_RUNALL)

    EXPECT_EQ(HSAKMT_STATUS_KERNEL_ALREADY_OPENED, hsaKmtOpenKFD());

    EXPECT_SUCCESS(hsaKmtCloseKFD());

    TEST_END
}

// Testing the normal scenario: open followed by close (done in the setup and teardown functions)
TEST_F(KFDOpenCloseKFDTest, OpenCloseKFD ) {
}

TEST_F(KFDOpenCloseKFDTest, InvalidKFDHandleTest ) {
    TEST_START(TESTPROFILE_RUNALL)

    HsaVersionInfo  m_VersionInfo;
    pid_t m_ChildPid = fork();
    if (m_ChildPid == 0) {
        EXPECT_EQ(HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED, hsaKmtGetVersion(&m_VersionInfo));
        exit(0);
    } else {
        int childStatus;
        EXPECT_EQ(m_ChildPid, waitpid(m_ChildPid, &childStatus, 0));
        EXPECT_NE(0, WIFEXITED(childStatus));
        EXPECT_EQ(0, WEXITSTATUS(childStatus));
    }
    TEST_END
}
