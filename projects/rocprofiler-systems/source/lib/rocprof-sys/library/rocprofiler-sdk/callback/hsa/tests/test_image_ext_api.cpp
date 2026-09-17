// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/hsa/image_ext_api.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

// TODO: placeholder test file, expand coverage once hsa::image_ext_api gains real
// behavior.

namespace rocprofsys::domains::callback::hsa
{
namespace
{

using test_support::externals;
using test_support::mock_sdk;

}  // namespace

TEST(image_ext_api_test, descriptor_reports_correct_metadata)
{
    constexpr const auto& k_domain = k_image_ext_api<mock_sdk, externals>;

    EXPECT_EQ(k_domain.meta.name, "hsa_image_ext_api");
    EXPECT_EQ(k_domain.meta.id, mock_sdk::CALLBACK_TRACING_HSA_IMAGE_EXT_API);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::callback);
    ASSERT_TRUE(k_domain.meta.group.has_value());
    EXPECT_EQ(k_domain.meta.group->name, "hsa_api");
}

TEST(image_ext_api_test, on_record_dispatches_by_phase_without_crashing)
{
    constexpr const auto& k_domain = k_image_ext_api<mock_sdk, externals>;
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

}  // namespace rocprofsys::domains::callback::hsa
