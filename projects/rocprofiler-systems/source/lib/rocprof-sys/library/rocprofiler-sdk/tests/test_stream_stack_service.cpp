// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/stream_stack_service.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include <cstdint>

#include <gtest/gtest.h>

namespace
{
using rocprofsys::domains::test_support::mock_sdk;
using service_t   = rocprofsys::rocprofiler_sdk::stream_stack_service<mock_sdk>;
using stream_id_t = mock_sdk::stream_id_t;

// The thread_local stack backing service_t is shared across every test in this
// binary; drain it back to its implicit default frame so tests do not leak
// state into each other.
class StreamStackServiceTest : public ::testing::Test
{
protected:
    void TearDown() override
    {
        while(service_t::top().handle != 0)
        {
            service_t::pop();
        }
    }
};

TEST_F(StreamStackServiceTest, top_returns_default_frame_when_nothing_pushed)
{
    EXPECT_EQ(service_t::top().handle, 0u);
}

TEST_F(StreamStackServiceTest, push_then_top_returns_pushed_value)
{
    service_t::push(stream_id_t{ 42 });

    EXPECT_EQ(service_t::top().handle, 42u);
}

TEST_F(StreamStackServiceTest, pop_restores_previous_frame)
{
    service_t::push(stream_id_t{ 7 });
    service_t::push(stream_id_t{ 9 });
    ASSERT_EQ(service_t::top().handle, 9u);

    service_t::pop();
    EXPECT_EQ(service_t::top().handle, 7u);

    service_t::pop();
    EXPECT_EQ(service_t::top().handle, 0u);
}

TEST_F(StreamStackServiceTest, request_stream_correlation_id_returns_success)
{
    service_t::push(stream_id_t{ 123 });

    mock_sdk::user_data_t external_corr_id{};
    const auto            result = service_t::request_stream_correlation_id(
        mock_sdk::thread_id_t{}, mock_sdk::context_id_t{},
        mock_sdk::external_correlation_request_kind_t{}, mock_sdk::tracing_operation_t{},
        std::uint64_t{ 0 }, &external_corr_id, nullptr);

    EXPECT_EQ(result, 0);
    EXPECT_NE(external_corr_id.ptr, nullptr);
}

TEST_F(StreamStackServiceTest,
       get_stream_id_returns_default_when_correlation_pointer_is_null)
{
    mock_sdk::kernel_dispatch_record_t record{};
    record.correlation_id.external.ptr = nullptr;

    EXPECT_EQ(service_t::get_stream_id(&record).handle, 0u);
}

TEST_F(StreamStackServiceTest,
       get_stream_id_recovers_stream_captured_at_request_time_and_frees_correlation)
{
    service_t::push(stream_id_t{ 55 });

    mock_sdk::user_data_t external_corr_id{};
    const auto            result = service_t::request_stream_correlation_id(
        mock_sdk::thread_id_t{}, mock_sdk::context_id_t{},
        mock_sdk::external_correlation_request_kind_t{}, mock_sdk::tracing_operation_t{},
        std::uint64_t{ 0 }, &external_corr_id, nullptr);
    ASSERT_EQ(result, 0);
    ASSERT_NE(external_corr_id.ptr, nullptr);

    mock_sdk::kernel_dispatch_record_t record{};
    record.correlation_id.external.ptr = external_corr_id.ptr;

    service_t::pop();

    const auto stream_id = service_t::get_stream_id(&record);

    EXPECT_EQ(stream_id.handle, 55u);
    EXPECT_EQ(record.correlation_id.external.ptr, nullptr);
}

TEST_F(StreamStackServiceTest, get_stream_id_is_idempotent_after_consuming_correlation)
{
    service_t::push(stream_id_t{ 77 });

    mock_sdk::user_data_t external_corr_id{};
    ASSERT_EQ(service_t::request_stream_correlation_id(
                  mock_sdk::thread_id_t{}, mock_sdk::context_id_t{},
                  mock_sdk::external_correlation_request_kind_t{},
                  mock_sdk::tracing_operation_t{}, std::uint64_t{ 0 }, &external_corr_id,
                  nullptr),
              0);

    mock_sdk::kernel_dispatch_record_t record{};
    record.correlation_id.external.ptr = external_corr_id.ptr;
    service_t::pop();

    ASSERT_EQ(service_t::get_stream_id(&record).handle, 77u);
    EXPECT_EQ(service_t::get_stream_id(&record).handle, 0u);
}

}  // namespace
