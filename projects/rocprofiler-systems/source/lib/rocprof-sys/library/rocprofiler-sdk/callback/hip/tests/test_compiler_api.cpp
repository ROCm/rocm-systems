// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/hip/compiler_api.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gtest/gtest.h>

// TODO: placeholder test file, expand coverage once hip::compiler_api gains real
// behavior.

namespace rocprofsys::domains::callback::hip
{
namespace
{

using test_support::expect_domain_uses_category;
using test_support::externals;
using test_support::externals_with_tracing;
using test_support::mock_sdk;
using test_support::mock_sdk_with_tracing;

}  // namespace

TEST(compiler_api_test, descriptor_reports_correct_metadata)
{
    constexpr const auto& k_domain = k_compiler_api<mock_sdk, externals>;

    EXPECT_EQ(k_domain.meta.name, "hip_compiler_api");
    EXPECT_EQ(k_domain.meta.id, mock_sdk::CALLBACK_TRACING_HIP_COMPILER_API);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::callback);
    ASSERT_TRUE(k_domain.meta.group.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access) checked by ASSERT_TRUE above
    EXPECT_EQ(k_domain.meta.group->name, "hip_api");
}

// Regression guard: hip::compiler_api must push/pop timemory and stamp buffer-storage
// records with "rocm_hip_api", not "rocm_hsa_api" or any other domain's category.
TEST(compiler_api_test, uses_rocm_hip_api_category)
{
    constexpr const auto& k_domain =
        k_compiler_api<mock_sdk_with_tracing, externals_with_tracing>;

    expect_domain_uses_category(k_domain, "rocm_hip_api");
}

}  // namespace rocprofsys::domains::callback::hip
