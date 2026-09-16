// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/hip/runtime_api.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

// TODO: placeholder test file, expand coverage once hip::runtime_api gains real
// behavior.

namespace rocprofsys::domains::callback::hip
{
namespace
{

using test_support::externals;
using test_support::mock_sdk;

}  // namespace

TEST(runtime_api_test, descriptor_reports_correct_metadata)
{
    constexpr const auto& k_domain = k_runtime_api<mock_sdk, externals>;

    EXPECT_EQ(k_domain.meta.name, "runtime_api");
    EXPECT_EQ(k_domain.meta.id, mock_sdk::CALLBACK_TRACING_HIP_RUNTIME_API);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::callback);
    ASSERT_TRUE(k_domain.meta.group.has_value());
    EXPECT_EQ(k_domain.meta.group->name, "hip_api");
}

}  // namespace rocprofsys::domains::callback::hip
