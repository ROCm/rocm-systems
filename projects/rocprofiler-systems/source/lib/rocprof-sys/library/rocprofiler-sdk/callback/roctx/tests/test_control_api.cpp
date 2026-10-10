// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/roctx/control_api.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

namespace rocprofsys::domains::callback::roctx
{
namespace
{

using ::testing::InSequence;
using ::testing::Return;
using ::testing::StrictMock;

using test_support::g_roctx_trigger_mock;
using test_support::g_tracing_backend_mock;
using test_support::gmock_roctx_trigger;
using test_support::gmock_tracing_backend;

using sdk = test_support::mock_sdk_with_tracing;
using ext = test_support::externals_with_marker;

constexpr std::uint64_t k_enter_ts = 1500;
constexpr std::uint64_t k_exit_ts  = 2000;

sdk::callback_tracing_record_t
make_control_record(std::size_t operation)
{
    sdk::callback_tracing_record_t record{};
    record.kind      = sdk::CALLBACK_TRACING_MARKER_CONTROL_API;
    record.operation = static_cast<std::uint32_t>(operation);
    return record;
}

class roctx_control_api_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_roctx_trigger_mock   = std::make_unique<StrictMock<gmock_roctx_trigger>>();
        g_tracing_backend_mock = std::make_unique<StrictMock<gmock_tracing_backend>>();
    }

    void TearDown() override
    {
        g_roctx_trigger_mock.reset();
        g_tracing_backend_mock.reset();
    }

    sdk::user_data_t m_user_data{};
};

}  // namespace

TEST_F(roctx_control_api_test, control_api_descriptor_reports_marker_control_metadata)
{
    constexpr const auto& k_domain = k_control_api<sdk, ext>;

    EXPECT_EQ(k_domain.meta.name, "marker_control_api");
    EXPECT_EQ(k_domain.meta.id, sdk::CALLBACK_TRACING_MARKER_CONTROL_API);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::callback);
    EXPECT_FALSE(k_domain.meta.group.has_value());
    EXPECT_NE(k_domain.on_record, nullptr);
    EXPECT_EQ(k_domain.on_configure, nullptr);
}

TEST_F(roctx_control_api_test, control_enter_pause_calls_on_pause)
{
    EXPECT_CALL(*g_roctx_trigger_mock, on_pause()).Times(1);

    on_roctx_control_enter<sdk, ext>(
        make_control_record(sdk::MARKER_CONTROL_API_ID_roctxProfilerPause), &m_user_data,
        nullptr, k_enter_ts);
}

TEST_F(roctx_control_api_test, control_enter_resume_does_nothing)
{
    on_roctx_control_enter<sdk, ext>(
        make_control_record(sdk::MARKER_CONTROL_API_ID_roctxProfilerResume), &m_user_data,
        nullptr, k_enter_ts);
}

TEST_F(roctx_control_api_test, control_exit_resume_calls_on_resume)
{
    EXPECT_CALL(*g_roctx_trigger_mock, on_resume()).Times(1);

    on_roctx_control_exit<sdk, ext>(
        make_control_record(sdk::MARKER_CONTROL_API_ID_roctxProfilerResume), &m_user_data,
        nullptr, k_exit_ts);
}

TEST_F(roctx_control_api_test, control_exit_pause_does_nothing)
{
    on_roctx_control_exit<sdk, ext>(
        make_control_record(sdk::MARKER_CONTROL_API_ID_roctxProfilerPause), &m_user_data,
        nullptr, k_exit_ts);
}

TEST_F(roctx_control_api_test, control_without_trigger_is_noop)
{
    g_roctx_trigger_mock.reset();

    on_roctx_control_enter<sdk, ext>(
        make_control_record(sdk::MARKER_CONTROL_API_ID_roctxProfilerPause), &m_user_data,
        nullptr, k_enter_ts);
    on_roctx_control_exit<sdk, ext>(
        make_control_record(sdk::MARKER_CONTROL_API_ID_roctxProfilerResume), &m_user_data,
        nullptr, k_exit_ts);
}

TEST_F(roctx_control_api_test,
       control_on_record_dispatches_pause_on_enter_and_resume_on_exit)
{
    constexpr const auto& k_domain = k_control_api<sdk, ext>;

    {
        const InSequence sequence;
        EXPECT_CALL(*g_tracing_backend_mock, get_timestamp()).WillOnce(Return(1));
        EXPECT_CALL(*g_roctx_trigger_mock, on_pause()).Times(1);
        EXPECT_CALL(*g_tracing_backend_mock, get_timestamp()).WillOnce(Return(2));
        EXPECT_CALL(*g_roctx_trigger_mock, on_resume()).Times(1);
    }

    auto enter  = make_control_record(sdk::MARKER_CONTROL_API_ID_roctxProfilerPause);
    enter.phase = sdk::CALLBACK_PHASE_ENTER;
    k_domain.on_record(enter, &m_user_data, nullptr);

    auto exit_record =
        make_control_record(sdk::MARKER_CONTROL_API_ID_roctxProfilerResume);
    exit_record.phase = sdk::CALLBACK_PHASE_EXIT;
    k_domain.on_record(exit_record, &m_user_data, nullptr);
}

}  // namespace rocprofsys::domains::callback::roctx
