// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/hsa/core_api.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

// TODO: placeholder test file, expand coverage once hsa::core_api gains real behavior.

namespace rocprofsys::domains::callback::hsa
{
namespace
{

using test_support::externals;
using test_support::mock_sdk;

}  // namespace

TEST(core_api_test, descriptor_reports_correct_metadata)
{
    constexpr const auto& k_domain = k_core_api<mock_sdk, externals>;

    EXPECT_EQ(k_domain.meta.name, "core_api");
    EXPECT_EQ(k_domain.meta.id, mock_sdk::CALLBACK_TRACING_HSA_CORE_API);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::callback);
    ASSERT_TRUE(k_domain.meta.group.has_value());
    EXPECT_EQ(k_domain.meta.group->name, "hsa_api");
}

}  // namespace rocprofsys::domains::callback::hsa
