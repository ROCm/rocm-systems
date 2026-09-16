// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/code_object.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

namespace rocprofsys::domains::callback
{
namespace
{

using ::testing::StrictMock;

using test_support::externals;
using test_support::g_externals_mock;
using test_support::gmock_externals;
using test_support::mock_sdk;

}  // namespace

TEST(code_object_test, descriptor_reports_correct_metadata)
{
    constexpr const auto& k_domain = k_code_object<mock_sdk, externals>;

    EXPECT_EQ(k_domain.meta.name, "code_object");
    EXPECT_EQ(k_domain.meta.id, mock_sdk::CALLBACK_TRACING_CODE_OBJECT);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::callback);
    EXPECT_FALSE(k_domain.meta.group.has_value());

    const auto expected_on_record = on_code_object<mock_sdk, externals>;
    EXPECT_EQ(k_domain.on_record, expected_on_record);
}

TEST(code_object_test, on_code_object_handles_call_without_crashing)
{
    const mock_sdk::callback_tracing_record_t record{};
    mock_sdk::user_data_t                     user_data{};

    on_code_object<mock_sdk, externals>(record, &user_data, nullptr);
}

// on_code_object_configure() is a no-op: it must not touch any Externals member.
// StrictMock<gmock_externals> fails the test if add_string/get_agents_by_type/
// add_pmc_info are called here.
TEST(code_object_test, on_configure_is_a_noop)
{
    g_externals_mock = std::make_unique<StrictMock<gmock_externals>>();

    on_code_object_configure<externals>();

    g_externals_mock.reset();
}

}  // namespace rocprofsys::domains::callback
