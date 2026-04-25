/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hipfile.h"

#include "test-common.h"

#include <gtest/gtest.h>

// hipFile APIs that trigger driver init/deinit
struct DriverInit : public ::testing::Test {

    // Ensure the driver is not initialized/open before the test
    void SetUp() override
    {
        ASSERT_EQ(hipFileUseCount(), 0);
    }

    // Ensure the driver is deinitialized/closed after the test
    void TearDown() override
    {
        while (0 < hipFileUseCount()) {
            ASSERT_EQ(hipFileDriverClose(), HIPFILE_SUCCESS);
        }
    }
};

// hipFile APIs that do not trigger driver init/deinit
struct DriverNoInit : public ::testing::Test {

    // Ensure the driver has not been opened
    void SetUp() override
    {
        ASSERT_EQ(hipFileUseCount(), 0);
        ASSERT_EQ(hipFileDriverClose(), HipFileOpError(hipFileDriverNotInitialized));
    }

    // Ensure the driver has not been opened
    void TearDown() override
    {
        ASSERT_EQ(hipFileUseCount(), 0);
        ASSERT_EQ(hipFileDriverClose(), HipFileOpError(hipFileDriverNotInitialized));
    }
};
