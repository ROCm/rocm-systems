// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/rocshmem_api.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

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

TEST(rocshmem_api_test, descriptor_reports_correct_metadata)
{
    constexpr const auto& k_domain = k_rocshmem_api<mock_sdk, externals>;

    EXPECT_EQ(k_domain.meta.name, "rocshmem_api");
    EXPECT_EQ(k_domain.meta.id, mock_sdk::CALLBACK_TRACING_ROCSHMEM_API);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::callback);
    ASSERT_FALSE(k_domain.meta.group.has_value());
    EXPECT_EQ(k_domain.meta.group, std::nullopt);
}

TEST(rocshmem_api_test, on_record_dispatches_by_phase_without_crashing)
{
    constexpr const auto& k_domain = k_rocshmem_api<mock_sdk, externals>;
    mock_sdk::user_data_t user_data{};

    auto enter_record  = mock_sdk::callback_tracing_record_t{};
    enter_record.phase = mock_sdk::CALLBACK_PHASE_ENTER;
    k_domain.on_record(enter_record, &user_data, nullptr);

    auto exit_record  = mock_sdk::callback_tracing_record_t{};
    exit_record.phase = mock_sdk::CALLBACK_PHASE_EXIT;
    k_domain.on_record(exit_record, &user_data, nullptr);

    auto none_record  = mock_sdk::callback_tracing_record_t{};
    none_record.phase = mock_sdk::CALLBACK_PHASE_NONE;
    k_domain.on_record(none_record, &user_data, nullptr);
}

// Regression guard: rocshmem_api must push/pop timemory and stamp buffer-storage
// records with "rocm_rocshmem_api", not any other domain's category.
TEST(rocshmem_api_test, uses_rocm_rocshmem_api_category)
{
    constexpr const auto& k_domain =
        k_rocshmem_api<mock_sdk_with_tracing, externals_with_tracing>;

    expect_domain_uses_category(k_domain, "rocm_rocshmem_api");
}

}  // namespace rocprofsys::domains::callback
