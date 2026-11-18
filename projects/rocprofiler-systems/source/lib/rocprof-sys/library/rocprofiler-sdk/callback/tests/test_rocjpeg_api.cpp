// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/rocjpeg_api.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gtest/gtest.h>

#include <optional>

namespace rocprofsys::domains::callback
{
namespace
{

using test_support::expect_domain_uses_category;
using test_support::externals;
using test_support::externals_with_tracing;
using test_support::mock_sdk;
using test_support::mock_sdk_with_tracing;

}  // namespace

TEST(rocjpeg_api_test, descriptor_reports_correct_metadata)
{
    constexpr const auto& k_domain = k_rocjpeg_api<mock_sdk, externals>;

    EXPECT_EQ(k_domain.meta.name, "rocjpeg_api");
    EXPECT_EQ(k_domain.meta.id, mock_sdk::CALLBACK_TRACING_ROCJPEG_API);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::callback);
    ASSERT_FALSE(k_domain.meta.group.has_value());
    EXPECT_EQ(k_domain.meta.group, std::nullopt);
}

// Regression guard: rocjpeg_api must push/pop timemory and stamp buffer-storage
// records with "rocm_rocjpeg_api", not any other domain's category.
TEST(rocjpeg_api_test, uses_rocm_rocjpeg_api_category)
{
    constexpr const auto& k_domain =
        k_rocjpeg_api<mock_sdk_with_tracing, externals_with_tracing>;

    expect_domain_uses_category(k_domain, "rocm_rocjpeg_api");
}

}  // namespace rocprofsys::domains::callback
