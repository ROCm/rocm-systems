// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/hsa/finalize_ext_api.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gtest/gtest.h>

// TODO: placeholder test file, expand coverage once hsa::finalize_ext_api gains real
// behavior.

namespace rocprofsys::domains::callback::hsa
{
namespace
{

using test_support::expect_domain_uses_category;
using test_support::externals;
using test_support::externals_with_tracing;
using test_support::mock_sdk;
using test_support::mock_sdk_with_tracing;

}  // namespace

TEST(finalize_ext_api_test, descriptor_reports_correct_metadata)
{
    constexpr const auto& k_domain = k_finalize_ext_api<mock_sdk, externals>;

    EXPECT_EQ(k_domain.meta.name, "hsa_finalize_ext_api");
    EXPECT_EQ(k_domain.meta.id, mock_sdk::CALLBACK_TRACING_HSA_FINALIZE_EXT_API);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::callback);
    ASSERT_TRUE(k_domain.meta.group.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access) checked by ASSERT_TRUE above
    EXPECT_EQ(k_domain.meta.group->name, "hsa_api");
}

// Regression guard: hsa::finalize_ext_api must push/pop timemory and stamp
// buffer-storage records with "rocm_hsa_api", not "rocm_hip_api" or any other
// domain's category.
TEST(finalize_ext_api_test, uses_rocm_hsa_api_category)
{
    constexpr const auto& k_domain =
        k_finalize_ext_api<mock_sdk_with_tracing, externals_with_tracing>;

    expect_domain_uses_category(k_domain, "rocm_hsa_api");
}

}  // namespace rocprofsys::domains::callback::hsa
