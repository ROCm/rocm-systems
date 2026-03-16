// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "rocprof-sys/library/rocprofiler-sdk/roctx_client.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

class roctx_client_test : public ::testing::Test
{
protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(roctx_client_test, test_roctx_client_constructor)
{
    using namespace rocprofsys::rocprofiler_sdk;

    const roctx_client_config config{ true, true, true, "TestRegion" };
    roctx_client              client(config);
    EXPECT_NE(client.get_controller(), nullptr);
}
