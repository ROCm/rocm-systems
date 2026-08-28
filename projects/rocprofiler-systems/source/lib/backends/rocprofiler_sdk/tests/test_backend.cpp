// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "backends/rocprofiler_sdk/backend.hpp"
#include "backends/rocprofiler_sdk/tests/mock_sdk.hpp"
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>

namespace rocprofsys::backends::rocprofiler_sdk::testing
{

namespace gm = ::testing;

using sut = backend<mock_sdk>;

// ─── Fixture ─────────────────────────────────────────────────────────────────

class backend_test : public ::testing::Test
{
protected:
    void SetUp() override { g_mock_sdk = std::make_unique<gmock_sdk>(); }
    void TearDown() override { g_mock_sdk.reset(); }
};

// ─── sdk_check ────────────────────────────────────────────────────────────────

TEST_F(backend_test, sdk_check_on_success_does_not_throw)
{
    EXPECT_NO_THROW(sdk_check<mock_sdk>(mock_sdk::STATUS_SUCCESS));
}

TEST_F(backend_test, sdk_check_on_error_throws_runtime_error)
{
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("some error"));

    EXPECT_THROW(sdk_check<mock_sdk>(mock_sdk::STATUS_ERROR), std::runtime_error);
}

TEST_F(backend_test, sdk_check_error_message_contains_sdk_string)
{
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("custom sdk message"));

    try
    {
        sdk_check<mock_sdk>(mock_sdk::STATUS_ERROR);
        FAIL() << "Expected std::runtime_error";
    } catch(const std::runtime_error& e)
    {
        EXPECT_THAT(e.what(), gm::HasSubstr("custom sdk message"));
    }
}

// ─── make_agent_id ────────────────────────────────────────────────────────────

TEST_F(backend_test, make_agent_id_constructs_from_handle)
{
    const auto result = sut::make_agent_id(99);
    EXPECT_EQ(result.handle, 99u);
}

// ─── Pure SDK delegations (return status_t) ────────────────────────────────────

TEST_F(backend_test, create_context_returns_sdk_status)
{
    context_id ctx{};
    EXPECT_CALL(*g_mock_sdk, create_context(&ctx))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_EQ(sut::create_context(&ctx), mock_sdk::STATUS_SUCCESS);
}

TEST_F(backend_test, start_context_returns_sdk_status)
{
    const context_id ctx{ 3 };
    EXPECT_CALL(*g_mock_sdk, start_context(ctx))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_EQ(sut::start_context(ctx), mock_sdk::STATUS_SUCCESS);
}

TEST_F(backend_test, stop_context_returns_sdk_status)
{
    const context_id ctx{ 3 };
    EXPECT_CALL(*g_mock_sdk, stop_context(ctx))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_EQ(sut::stop_context(ctx), mock_sdk::STATUS_SUCCESS);
}

TEST_F(backend_test, sample_device_counting_service_returns_sdk_status)
{
    const context_id ctx{ 1 };
    const user_data  ud{};
    size_t           count = 0;
    EXPECT_CALL(*g_mock_sdk, sample_device_counting_service(ctx, ud, 0, nullptr, &count))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_EQ(sut::sample_device_counting_service(ctx, ud, 0, nullptr, &count),
              mock_sdk::STATUS_SUCCESS);
}

TEST_F(backend_test, iterate_agent_supported_counters_returns_sdk_status)
{
    const agent_id ag{ 5 };
    EXPECT_CALL(*g_mock_sdk, iterate_agent_supported_counters(ag, nullptr, nullptr))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_EQ(sut::iterate_agent_supported_counters(ag, nullptr, nullptr),
              mock_sdk::STATUS_SUCCESS);
}

TEST_F(backend_test, create_counter_config_returns_sdk_status)
{
    const agent_id    ag{ 5 };
    counter_config_id cfg{};
    EXPECT_CALL(*g_mock_sdk, create_counter_config(ag, nullptr, 0, &cfg))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_EQ(sut::create_counter_config(ag, nullptr, 0, &cfg), mock_sdk::STATUS_SUCCESS);
}

TEST_F(backend_test, configure_device_counting_service_returns_sdk_status)
{
    const context_id ctx{ 1 };
    const buffer_id  buf{ 2 };
    const agent_id   ag{ 3 };
    EXPECT_CALL(*g_mock_sdk,
                configure_device_counting_service(ctx, buf, ag, nullptr, nullptr))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_EQ(sut::configure_device_counting_service(ctx, buf, ag, nullptr, nullptr),
              mock_sdk::STATUS_SUCCESS);
}

// ─── query_record_counter_id ─────────────────────────────────────────────────

TEST_F(backend_test, query_record_counter_id_extracts_instance_id_from_record)
{
    // For SDK v1+ (compile_time_version >= 10000), backend.hpp directly writes
    // record.id into counter_id->handle without making an SDK call.
    const counter_record rec{ counter_instance_id{ 42 }, 0.0 };
    counter_id           out_id{};

    EXPECT_EQ(sut::query_record_counter_id(rec, &out_id), sut::status_success);
    EXPECT_EQ(out_id.handle, std::uint64_t{ 42 });
}

TEST_F(backend_test, query_record_counter_id_returns_error_for_null_output)
{
    const counter_record rec{ counter_instance_id{ 42 }, 0.0 };

    EXPECT_EQ(sut::query_record_counter_id(rec, nullptr),
              mock_sdk::STATUS_ERROR_INVALID_ARGUMENT);
}

// ─── query_counter_details ────────────────────────────────────────────────────

TEST_F(backend_test, query_counter_details_returns_empty_on_sdk_error)
{
    const counter_id cid{ 7 };
    EXPECT_CALL(*g_mock_sdk,
                query_counter_info(cid, mock_sdk::COUNTER_INFO_VERSION_1, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));

    EXPECT_THAT(sut::query_counter_details(cid), gm::IsEmpty());
}

TEST_F(backend_test, query_counter_details_returns_empty_when_name_is_null)
{
    // SDK returns SUCCESS but leaves info zero-initialised; name stays nullptr.
    const counter_id cid{ 7 };
    EXPECT_CALL(*g_mock_sdk,
                query_counter_info(cid, mock_sdk::COUNTER_INFO_VERSION_1, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_THAT(sut::query_counter_details(cid), gm::IsEmpty());
}

TEST_F(backend_test, query_counter_details_returns_metadata_with_one_instance_and_one_dim)
{
    const counter_id cid{ 7 };

    dim_info          dim{ "index", 0 };
    dim_info*         dims_arr[] = { &dim };
    dim_instance      inst{ 1, 1, dims_arr };
    dim_instance*     insts_arr[] = { &inst };
    counter_info_v1_t fill{ "SQ_WAVES", "Wave count", "SQ", "", 0, 1, 1, insts_arr };

    EXPECT_CALL(*g_mock_sdk,
                query_counter_info(cid, mock_sdk::COUNTER_INFO_VERSION_1, gm::_))
        .WillOnce([&fill](counter_id, counter_info_ver, void* out) -> status_t {
            *static_cast<counter_info_v1_t*>(out) = fill;
            return k_status_success;
        });

    const auto result = sut::query_counter_details(cid);

    ASSERT_THAT(result, gm::SizeIs(1));
    EXPECT_EQ(result[0].name, "SQ_WAVES");
    EXPECT_EQ(result[0].counter_id, 1u);
    EXPECT_EQ(result[0].is_derived, true);
    ASSERT_THAT(result[0].dimensions, gm::SizeIs(1));
    EXPECT_EQ(result[0].dimensions[0].name, "index");
    EXPECT_EQ(result[0].dimensions[0].position, 0u);
}

TEST_F(backend_test, query_counter_details_returns_empty_dims_when_instance_has_none)
{
    const counter_id cid{ 8 };

    dim_instance      inst{ 2, 0, nullptr };
    dim_instance*     insts_arr[] = { &inst };
    counter_info_v1_t fill{ "SQ_BUSY", nullptr, nullptr, nullptr, 0, 0, 1, insts_arr };

    EXPECT_CALL(*g_mock_sdk,
                query_counter_info(cid, mock_sdk::COUNTER_INFO_VERSION_1, gm::_))
        .WillOnce([&fill](counter_id, counter_info_ver, void* out) -> status_t {
            *static_cast<counter_info_v1_t*>(out) = fill;
            return k_status_success;
        });

    const auto result = sut::query_counter_details(cid);

    ASSERT_THAT(result, gm::SizeIs(1));
    EXPECT_EQ(result[0].name, "SQ_BUSY");
    EXPECT_THAT(result[0].dimensions, gm::IsEmpty());
}

// ─── flush_buffer ─────────────────────────────────────────────────────────────

TEST_F(backend_test, flush_buffer_on_success_does_not_throw)
{
    const buffer_id buf{ 42 };
    EXPECT_CALL(*g_mock_sdk, flush_buffer(buf))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_NO_THROW(sut::flush_buffer(buf));
}

TEST_F(backend_test, flush_buffer_on_buffer_busy_silently_ignored)
{
    const buffer_id buf{ 42 };
    EXPECT_CALL(*g_mock_sdk, flush_buffer(buf))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR_BUFFER_BUSY));

    EXPECT_NO_THROW(sut::flush_buffer(buf));
}

TEST_F(backend_test, flush_buffer_on_other_error_throws)
{
    const buffer_id buf{ 42 };
    EXPECT_CALL(*g_mock_sdk, flush_buffer(buf))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("flush failed"));

    EXPECT_THROW(sut::flush_buffer(buf), std::runtime_error);
}

// ─── destroy_buffer ───────────────────────────────────────────────────────────

TEST_F(backend_test, destroy_buffer_succeeds_on_first_call)
{
    const buffer_id buf{ 7 };
    EXPECT_CALL(*g_mock_sdk, destroy_buffer(buf))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_NO_THROW(sut::destroy_buffer(buf));
}

TEST_F(backend_test, destroy_buffer_retries_until_not_busy)
{
    const buffer_id buf{ 7 };
    EXPECT_CALL(*g_mock_sdk, destroy_buffer(buf))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR_BUFFER_BUSY))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR_BUFFER_BUSY))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_NO_THROW(sut::destroy_buffer(buf));
}

// ─── context_is_active ────────────────────────────────────────────────────────

TEST_F(backend_test, context_is_active_returns_true_when_active)
{
    const context_id ctx{ 1 };
    EXPECT_CALL(*g_mock_sdk, context_is_active(ctx, gm::_))
        .WillOnce(
            gm::DoAll(gm::SetArgPointee<1>(1), gm::Return(mock_sdk::STATUS_SUCCESS)));

    EXPECT_TRUE(sut::context_is_active(ctx));
}

TEST_F(backend_test, context_is_active_returns_false_when_out_is_zero)
{
    const context_id ctx{ 1 };
    EXPECT_CALL(*g_mock_sdk, context_is_active(ctx, gm::_))
        .WillOnce(
            gm::DoAll(gm::SetArgPointee<1>(0), gm::Return(mock_sdk::STATUS_SUCCESS)));

    EXPECT_FALSE(sut::context_is_active(ctx));
}

TEST_F(backend_test, context_is_active_returns_false_on_sdk_error)
{
    const context_id ctx{ 1 };
    EXPECT_CALL(*g_mock_sdk, context_is_active(ctx, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));

    EXPECT_FALSE(sut::context_is_active(ctx));
}

// ─── context_is_valid ────────────────────────────────────────────────────────

TEST_F(backend_test, context_is_valid_returns_true_when_valid)
{
    const context_id ctx{ 2 };
    EXPECT_CALL(*g_mock_sdk, context_is_valid(ctx, gm::_))
        .WillOnce(
            gm::DoAll(gm::SetArgPointee<1>(1), gm::Return(mock_sdk::STATUS_SUCCESS)));

    EXPECT_TRUE(sut::context_is_valid(ctx));
}

TEST_F(backend_test, context_is_valid_returns_false_when_out_is_zero)
{
    const context_id ctx{ 2 };
    EXPECT_CALL(*g_mock_sdk, context_is_valid(ctx, gm::_))
        .WillOnce(
            gm::DoAll(gm::SetArgPointee<1>(0), gm::Return(mock_sdk::STATUS_SUCCESS)));

    EXPECT_FALSE(sut::context_is_valid(ctx));
}

TEST_F(backend_test, context_is_valid_returns_false_on_sdk_error)
{
    const context_id ctx{ 2 };
    EXPECT_CALL(*g_mock_sdk, context_is_valid(ctx, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));

    EXPECT_FALSE(sut::context_is_valid(ctx));
}

// ─── get_timestamp ────────────────────────────────────────────────────────────

TEST_F(backend_test, get_timestamp_returns_value_written_by_sdk)
{
    const timestamp expected{ 12345 };
    EXPECT_CALL(*g_mock_sdk, get_timestamp(gm::_))
        .WillOnce(gm::DoAll(gm::SetArgPointee<0>(expected),
                            gm::Return(mock_sdk::STATUS_SUCCESS)));

    EXPECT_EQ(sut::get_timestamp(), expected);
}

// ─── get_status_string ────────────────────────────────────────────────────────

TEST_F(backend_test, get_status_string_delegates_to_sdk)
{
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("error string"));

    EXPECT_STREQ(sut::get_status_string(mock_sdk::STATUS_ERROR), "error string");
}

// ─── Void forwarders — success path ──────────────────────────────────────────

TEST_F(backend_test, create_buffer_succeeds)
{
    buffer_id buf{};
    EXPECT_CALL(*g_mock_sdk,
                create_buffer(gm::_, gm::_, gm::_, gm::_, gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_NO_THROW(sut::create_buffer({ 1 }, 4096, 2048, 0, nullptr, nullptr, &buf));
}

TEST_F(backend_test, create_callback_thread_succeeds)
{
    callback_thread_id thread{};
    EXPECT_CALL(*g_mock_sdk, create_callback_thread(&thread))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_NO_THROW(sut::create_callback_thread(&thread));
}

TEST_F(backend_test, assign_callback_thread_succeeds)
{
    const buffer_id          buf{ 3 };
    const callback_thread_id thread{ 9 };
    EXPECT_CALL(*g_mock_sdk, assign_callback_thread(buf, thread))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_NO_THROW(sut::assign_callback_thread(buf, thread));
}

TEST_F(backend_test, configure_callback_tracing_service_succeeds)
{
    EXPECT_CALL(*g_mock_sdk, configure_callback_tracing_service(gm::_, gm::_, gm::_,
                                                                gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_NO_THROW(
        sut::configure_callback_tracing_service({}, 0, nullptr, 0, nullptr, nullptr));
}

TEST_F(backend_test, configure_buffer_tracing_service_succeeds)
{
    EXPECT_CALL(*g_mock_sdk,
                configure_buffer_tracing_service(gm::_, gm::_, gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_NO_THROW(sut::configure_buffer_tracing_service({}, 0, nullptr, 0, {}));
}

TEST_F(backend_test, configure_external_correlation_id_request_service_succeeds)
{
    EXPECT_CALL(*g_mock_sdk, configure_external_correlation_id_request_service(
                                 gm::_, gm::_, gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_NO_THROW(sut::configure_external_correlation_id_request_service(
        {}, nullptr, 0, nullptr, nullptr));
}

TEST_F(backend_test, configure_external_correlation_id_request_service_throws_on_error)
{
    EXPECT_CALL(*g_mock_sdk, configure_external_correlation_id_request_service(
                                 gm::_, gm::_, gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("ext corr failed"));

    EXPECT_THROW(sut::configure_external_correlation_id_request_service({}, nullptr, 0,
                                                                        nullptr, nullptr),
                 std::runtime_error);
}

TEST_F(backend_test, configure_callback_dispatch_counting_service_succeeds)
{
    EXPECT_CALL(*g_mock_sdk, configure_callback_dispatch_counting_service(
                                 gm::_, gm::_, gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_NO_THROW(sut::configure_callback_dispatch_counting_service(
        {}, nullptr, nullptr, nullptr, nullptr));
}

TEST_F(backend_test, configure_callback_dispatch_counting_service_throws_on_error)
{
    EXPECT_CALL(*g_mock_sdk, configure_callback_dispatch_counting_service(
                                 gm::_, gm::_, gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("dispatch failed"));

    EXPECT_THROW(sut::configure_callback_dispatch_counting_service({}, nullptr, nullptr,
                                                                   nullptr, nullptr),
                 std::runtime_error);
}

TEST_F(backend_test, at_internal_thread_create_succeeds)
{
    EXPECT_CALL(*g_mock_sdk, at_internal_thread_create(gm::_, gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_NO_THROW(sut::at_internal_thread_create(nullptr, nullptr, 0, nullptr));
}

TEST_F(backend_test, query_callback_op_name_succeeds)
{
    const char*   name{};
    std::uint64_t name_len{};
    EXPECT_CALL(*g_mock_sdk, query_callback_op_name(gm::_, gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_NO_THROW(sut::query_callback_op_name(0, 0, &name, &name_len));
}

TEST_F(backend_test, query_buffer_op_name_succeeds)
{
    const char*   name{};
    std::uint64_t name_len{};
    EXPECT_CALL(*g_mock_sdk, query_buffer_op_name(gm::_, gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_NO_THROW(sut::query_buffer_op_name(0, 0, &name, &name_len));
}

TEST_F(backend_test, query_buffer_op_name_throws_on_sdk_error)
{
    const char*   name{};
    std::uint64_t name_len{};
    EXPECT_CALL(*g_mock_sdk, query_buffer_op_name(gm::_, gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("buf op failed"));

    EXPECT_THROW(sut::query_buffer_op_name(0, 0, &name, &name_len), std::runtime_error);
}

TEST_F(backend_test, iterate_callback_tracing_kind_operation_args_succeeds)
{
    EXPECT_CALL(*g_mock_sdk,
                iterate_callback_tracing_kind_operation_args(gm::_, gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_NO_THROW(
        sut::iterate_callback_tracing_kind_operation_args({}, nullptr, 0, nullptr));
}

TEST_F(backend_test, iterate_callback_tracing_kind_operation_args_throws_on_error)
{
    EXPECT_CALL(*g_mock_sdk,
                iterate_callback_tracing_kind_operation_args(gm::_, gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("args failed"));

    EXPECT_THROW(
        sut::iterate_callback_tracing_kind_operation_args({}, nullptr, 0, nullptr),
        std::runtime_error);
}

TEST_F(backend_test, iterate_counter_dimensions_succeeds)
{
    const counter_id cid{ 5 };
    EXPECT_CALL(*g_mock_sdk, iterate_counter_dimensions(cid, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_NO_THROW(sut::iterate_counter_dimensions(cid, nullptr, nullptr));
}

TEST_F(backend_test, query_counter_info_succeeds)
{
    const counter_id cid{ 6 };
    EXPECT_CALL(*g_mock_sdk, query_counter_info(cid, 0, nullptr))
        .WillOnce(gm::Return(mock_sdk::STATUS_SUCCESS));

    EXPECT_NO_THROW(sut::query_counter_info(cid, 0, nullptr));
}

TEST_F(backend_test, query_counter_info_throws_on_sdk_error)
{
    const counter_id cid{ 6 };
    EXPECT_CALL(*g_mock_sdk, query_counter_info(cid, 0, nullptr))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("info failed"));

    EXPECT_THROW(sut::query_counter_info(cid, 0, nullptr), std::runtime_error);
}

// ─── Void forwarders — error path ────────────────────────────────────────────

TEST_F(backend_test, create_buffer_throws_on_sdk_error)
{
    EXPECT_CALL(*g_mock_sdk,
                create_buffer(gm::_, gm::_, gm::_, gm::_, gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("create_buffer failed"));

    buffer_id buf{};
    EXPECT_THROW(sut::create_buffer({ 1 }, 4096, 2048, 0, nullptr, nullptr, &buf),
                 std::runtime_error);
}

TEST_F(backend_test, create_callback_thread_throws_on_sdk_error)
{
    EXPECT_CALL(*g_mock_sdk, create_callback_thread(gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("thread failed"));

    callback_thread_id thread{};
    EXPECT_THROW(sut::create_callback_thread(&thread), std::runtime_error);
}

TEST_F(backend_test, assign_callback_thread_throws_on_sdk_error)
{
    const buffer_id          buf{ 3 };
    const callback_thread_id thread{ 9 };
    EXPECT_CALL(*g_mock_sdk, assign_callback_thread(buf, thread))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("assign failed"));

    EXPECT_THROW(sut::assign_callback_thread(buf, thread), std::runtime_error);
}

TEST_F(backend_test, configure_callback_tracing_service_throws_on_sdk_error)
{
    EXPECT_CALL(*g_mock_sdk, configure_callback_tracing_service(gm::_, gm::_, gm::_,
                                                                gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("cb tracing failed"));

    EXPECT_THROW(
        sut::configure_callback_tracing_service({}, 0, nullptr, 0, nullptr, nullptr),
        std::runtime_error);
}

TEST_F(backend_test, configure_buffer_tracing_service_throws_on_sdk_error)
{
    EXPECT_CALL(*g_mock_sdk,
                configure_buffer_tracing_service(gm::_, gm::_, gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("buf tracing failed"));

    EXPECT_THROW(sut::configure_buffer_tracing_service({}, 0, nullptr, 0, {}),
                 std::runtime_error);
}

TEST_F(backend_test, at_internal_thread_create_throws_on_sdk_error)
{
    EXPECT_CALL(*g_mock_sdk, at_internal_thread_create(gm::_, gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("thread hook failed"));

    EXPECT_THROW(sut::at_internal_thread_create(nullptr, nullptr, 0, nullptr),
                 std::runtime_error);
}

TEST_F(backend_test, query_callback_op_name_throws_on_sdk_error)
{
    EXPECT_CALL(*g_mock_sdk, query_callback_op_name(gm::_, gm::_, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("op name failed"));

    const char*   name{};
    std::uint64_t name_len{};
    EXPECT_THROW(sut::query_callback_op_name(0, 0, &name, &name_len), std::runtime_error);
}

TEST_F(backend_test, iterate_counter_dimensions_throws_on_sdk_error)
{
    const counter_id cid{ 5 };
    EXPECT_CALL(*g_mock_sdk, iterate_counter_dimensions(cid, gm::_, gm::_))
        .WillOnce(gm::Return(mock_sdk::STATUS_ERROR));
    EXPECT_CALL(*g_mock_sdk, get_status_string(mock_sdk::STATUS_ERROR))
        .WillOnce(gm::Return("dim failed"));

    EXPECT_THROW(sut::iterate_counter_dimensions(cid, nullptr, nullptr),
                 std::runtime_error);
}

// ─── Cached SDK-query methods (get_version / tracing names) ──────────────────
//
// Each cached method needs its own Sdk type per test — sharing mock_sdk (and
// therefore `sut`) would let one test's cached result leak into another.

template <int Tag>
struct tagged_mock_sdk : mock_sdk
{};

constexpr int version_cache_tag_id        = 1;
constexpr int callback_names_cache_tag_id = 2;
constexpr int buffer_names_cache_tag_id   = 3;
constexpr int version_match_tag_id        = 4;
constexpr int version_mismatch_tag_id     = 5;

using version_cache_tag        = tagged_mock_sdk<version_cache_tag_id>;
using callback_names_cache_tag = tagged_mock_sdk<callback_names_cache_tag_id>;
using buffer_names_cache_tag   = tagged_mock_sdk<buffer_names_cache_tag_id>;
using version_match_tag        = tagged_mock_sdk<version_match_tag_id>;
using version_mismatch_tag     = tagged_mock_sdk<version_mismatch_tag_id>;

TEST_F(backend_test, get_version_caches_after_first_call)
{
    EXPECT_CALL(*g_mock_sdk, get_version(gm::_, gm::_, gm::_))
        .Times(1)
        .WillOnce(gm::DoAll(gm::SetArgPointee<0>(1), gm::SetArgPointee<1>(2),
                            gm::SetArgPointee<2>(3),
                            gm::Return(mock_sdk::STATUS_SUCCESS)));

    using sut_v = backend<version_cache_tag>;

    std::uint32_t major = 0;
    std::uint32_t minor = 0;
    std::uint32_t patch = 0;
    EXPECT_EQ(sut_v::get_version(&major, &minor, &patch), mock_sdk::STATUS_SUCCESS);
    EXPECT_EQ(major, 1u);
    EXPECT_EQ(minor, 2u);
    EXPECT_EQ(patch, 3u);

    // Second call must not hit the SDK again — the EXPECT_CALL above is Times(1).
    major = 0;
    minor = 0;
    patch = 0;
    EXPECT_EQ(sut_v::get_version(&major, &minor, &patch), mock_sdk::STATUS_SUCCESS);
    EXPECT_EQ(major, 1u);
    EXPECT_EQ(minor, 2u);
    EXPECT_EQ(patch, 3u);
}

TEST_F(backend_test, get_callback_tracing_names_caches_after_first_call)
{
    auto table = name_info<>{};
    table.emplace(1, "HIP_RUNTIME_API");
    EXPECT_CALL(*g_mock_sdk, get_callback_tracing_names())
        .Times(1)
        .WillOnce(gm::Return(table));

    using sut_cb = backend<callback_names_cache_tag>;

    const auto& first  = sut_cb::get_callback_tracing_names();
    const auto& second = sut_cb::get_callback_tracing_names();

    EXPECT_EQ(&first, &second);
    EXPECT_EQ(first[1].name, "HIP_RUNTIME_API");
}

TEST_F(backend_test, get_buffer_tracing_names_caches_after_first_call)
{
    auto table = name_info<>{};
    table.emplace(2, "MEMORY_COPY");
    EXPECT_CALL(*g_mock_sdk, get_buffer_tracing_names())
        .Times(1)
        .WillOnce(gm::Return(table));

    using sut_buf = backend<buffer_names_cache_tag>;

    const auto& first  = sut_buf::get_buffer_tracing_names();
    const auto& second = sut_buf::get_buffer_tracing_names();

    EXPECT_EQ(&first, &second);
    EXPECT_EQ(first[2].name, "MEMORY_COPY");
}

TEST_F(backend_test, name_info_emplace_operation_overload_fills_gap_with_default_value)
{
    auto table = name_info<>{};
    table.emplace(1, 0, "OP_A");
    table.emplace(1, 2, "OP_C");

    const auto& operations = table[1].operations;
    ASSERT_EQ(operations.size(), 3u);
    EXPECT_EQ(operations.at(0), "OP_A");
    EXPECT_EQ(operations.at(1), std::string_view{});
    EXPECT_EQ(operations.at(2), "OP_C");
}

TEST_F(backend_test, check_version_compatibility_accepts_matching_runtime_version)
{
    // mock_sdk::compile_time_version == 10100u -> 1.1.0
    EXPECT_CALL(*g_mock_sdk, get_version(gm::_, gm::_, gm::_))
        .WillOnce(gm::DoAll(gm::SetArgPointee<0>(1), gm::SetArgPointee<1>(1),
                            gm::SetArgPointee<2>(0),
                            gm::Return(mock_sdk::STATUS_SUCCESS)));

    using sut_v = backend<version_match_tag>;
    EXPECT_NO_THROW(sut_v::check_version_compatibility());
}

TEST_F(backend_test, check_version_compatibility_throws_on_runtime_version_mismatch)
{
    // mock_sdk::compile_time_version == 10100u -> 1.1.0, runtime reports 1.2.0
    EXPECT_CALL(*g_mock_sdk, get_version(gm::_, gm::_, gm::_))
        .WillOnce(gm::DoAll(gm::SetArgPointee<0>(1), gm::SetArgPointee<1>(2),
                            gm::SetArgPointee<2>(0),
                            gm::Return(mock_sdk::STATUS_SUCCESS)));

    using sut_v = backend<version_mismatch_tag>;
    EXPECT_THROW(sut_v::check_version_compatibility(), std::runtime_error);
}

}  // namespace rocprofsys::backends::rocprofiler_sdk::testing
