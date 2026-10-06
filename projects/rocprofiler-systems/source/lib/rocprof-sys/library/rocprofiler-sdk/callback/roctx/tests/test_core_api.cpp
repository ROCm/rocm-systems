// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/roctx/core_api.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace rocprofsys::domains::callback::roctx
{
namespace
{

using ::testing::A;
using ::testing::Eq;
using ::testing::InSequence;
using ::testing::NotNull;
using ::testing::Return;
using ::testing::SaveArg;
using ::testing::StrictMock;

using test_support::g_buffer_storage_mock;
using test_support::g_externals_mock;
using test_support::g_metadata_registry_mock;
using test_support::g_roctx_trigger_mock;
using test_support::g_session_mock;
using test_support::g_tracing_backend_mock;
using test_support::gmock_buffer_storage;
using test_support::gmock_externals;
using test_support::gmock_metadata_registry;
using test_support::gmock_roctx_trigger;
using test_support::gmock_session;
using test_support::gmock_tracing_backend;
using test_support::thread_info_data_t;
using test_support::tracing_names_t;

using sdk = test_support::mock_sdk_with_tracing;
using ext = test_support::externals_with_marker;

constexpr std::uint64_t k_enter_ts             = 1500;
constexpr std::uint64_t k_exit_ts              = 2000;
constexpr std::uint64_t k_thread_id            = 7;
constexpr std::uint64_t k_correlation_internal = 31;
constexpr std::uint64_t k_correlation_external = 32;
constexpr std::int32_t  k_ppid                 = 100;
constexpr std::int32_t  k_pid                  = 200;

constexpr std::string_view k_trigger_name  = "roctx";
constexpr std::string_view k_category_name = "rocm_marker_api";
// The shared mock backend resolves every (kind, operation) to this name.
constexpr std::string_view k_table_name = "operation";
// Not any marker core/control operation id.
constexpr std::size_t k_unmapped_operation = 99;

// Serialization of the single argument fed through iterate_args by populate_one_arg.
constexpr std::string_view k_serialized_arg = "0;;int;;x;;42;;";

// Gives the SDK-side argument iteration one populated argument, as the real SDK does.
void
populate_one_arg(std::uint64_t kind, std::uint32_t operation,
                 sdk::callback_tracing_operation_args_cb_t callback, void* data)
{
    callback(kind, static_cast<std::int32_t>(operation), 0, nullptr, 0, "int", "x", "42",
             0, data);
}

sdk::callback_tracing_record_t
make_record(std::size_t kind, std::size_t operation, void* payload)
{
    sdk::callback_tracing_record_t record{};
    record.kind                          = kind;
    record.operation                     = static_cast<std::uint32_t>(operation);
    record.thread_id                     = k_thread_id;
    record.correlation_id.internal       = k_correlation_internal;
    record.correlation_id.external.value = k_correlation_external;
    record.payload                       = payload;
    return record;
}

sdk::callback_tracing_record_t
make_core_record(std::size_t operation, void* payload)
{
    return make_record(sdk::CALLBACK_TRACING_MARKER_CORE_API, operation, payload);
}

sdk::callback_tracing_record_t
make_control_record(std::size_t operation)
{
    return make_record(sdk::CALLBACK_TRACING_MARKER_CONTROL_API, operation, nullptr);
}

// Runs @p body on a fresh thread so the thread_local range stacks start empty and
// nothing a test leaves on them reaches another test.
template <typename Body>
void
on_fresh_thread(Body&& body)
{
    std::thread worker{ std::forward<Body>(body) };
    worker.join();
}

void
expect_writes_allowed()
{
    EXPECT_CALL(*g_roctx_trigger_mock, should_write_markers()).WillOnce(Return(true));
    EXPECT_CALL(*g_session_mock, is_active_without(Eq(k_trigger_name)))
        .WillOnce(Return(true));
}

void
expect_writes_denied_by_trigger()
{
    EXPECT_CALL(*g_roctx_trigger_mock, should_write_markers()).WillOnce(Return(false));
}

void
expect_writes_denied_by_session()
{
    EXPECT_CALL(*g_roctx_trigger_mock, should_write_markers()).WillOnce(Return(true));
    EXPECT_CALL(*g_session_mock, is_active_without(Eq(k_trigger_name)))
        .WillOnce(Return(false));
}

void
expect_timemory_push(std::string_view name)
{
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(true));
    EXPECT_CALL(*g_externals_mock, tracing_push_timemory(Eq(name)));
}

// Everything write_end() sends to the metadata registry, the SDK and the buffer
// storage regardless of the timemory switch.
void
expect_end_sinks(std::size_t operation, std::string_view name, std::uint64_t begin_ts,
                 std::string_view args = {})
{
    EXPECT_CALL(*g_externals_mock, get_ppid()).WillOnce(Return(k_ppid));
    EXPECT_CALL(*g_externals_mock, get_pid()).WillOnce(Return(k_pid));
    EXPECT_CALL(*g_metadata_registry_mock,
                add_thread_info(
                    Eq(thread_info_data_t{ k_ppid, k_pid, k_thread_id, 0, 0, "{}" })));

    auto& iterate_expectation = EXPECT_CALL(
        *g_tracing_backend_mock,
        iterate_args(
            Eq(static_cast<std::uint64_t>(sdk::CALLBACK_TRACING_MARKER_CORE_API)),
            Eq(static_cast<std::uint32_t>(operation)), NotNull(), NotNull()));
    if(!args.empty())
    {
        iterate_expectation.WillOnce(populate_one_arg);
    }

    EXPECT_CALL(*g_buffer_storage_mock,
                store_region_sample(
                    Eq(k_thread_id), Eq(std::string{ name }), Eq(k_correlation_internal),
                    Eq(k_correlation_external), Eq(begin_ts), Eq(k_exit_ts),
                    Eq(std::string{ args }), Eq(std::string{ k_category_name })));
}

// write_end() with timemory on.
void
expect_write_end(std::size_t operation, std::string_view name, std::uint64_t begin_ts)
{
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(true));
    EXPECT_CALL(*g_externals_mock, tracing_pop_timemory(Eq(name)));
    expect_end_sinks(operation, name, begin_ts);
}

class roctx_core_api_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_externals_mock = std::make_unique<StrictMock<gmock_externals>>();
        g_metadata_registry_mock =
            std::make_unique<StrictMock<gmock_metadata_registry>>();
        g_buffer_storage_mock  = std::make_unique<StrictMock<gmock_buffer_storage>>();
        g_roctx_trigger_mock   = std::make_unique<StrictMock<gmock_roctx_trigger>>();
        g_session_mock         = std::make_unique<StrictMock<gmock_session>>();
        g_tracing_backend_mock = std::make_unique<StrictMock<gmock_tracing_backend>>();
    }

    void TearDown() override
    {
        g_externals_mock.reset();
        g_metadata_registry_mock.reset();
        g_buffer_storage_mock.reset();
        g_roctx_trigger_mock.reset();
        g_session_mock.reset();
        g_tracing_backend_mock.reset();
    }

    sdk::marker_payload_t m_payload{};
    sdk::user_data_t      m_user_data{};
};

}  // namespace

// ─── Descriptors / configure ────────────────────────────────────────────────────

TEST_F(roctx_core_api_test, core_api_descriptor_reports_marker_core_metadata)
{
    constexpr const auto& k_domain = k_core_api<sdk, ext>;

    EXPECT_EQ(k_domain.meta.name, "marker_core_api");
    EXPECT_EQ(k_domain.meta.id, sdk::CALLBACK_TRACING_MARKER_CORE_API);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::callback);
    EXPECT_FALSE(k_domain.meta.group.has_value());
    EXPECT_NE(k_domain.on_record, nullptr);
    EXPECT_NE(k_domain.on_configure, nullptr);
}

TEST_F(roctx_core_api_test, on_configure_registers_marker_category_string_exactly_once)
{
    constexpr const auto& k_domain = k_core_api<sdk, ext>;

    EXPECT_CALL(*g_metadata_registry_mock, add_string(Eq(k_category_name))).Times(1);

    k_domain.on_configure();
}

// ─── Core enter ─────────────────────────────────────────────────────────────────

TEST_F(roctx_core_api_test,
       range_push_enter_opens_range_and_writes_begin_when_writes_allowed)
{
    m_payload.args.roctxRangePushA.message = "range_a";
    const auto record =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangePushA, &m_payload);
    std::uint64_t range_id = 0;

    EXPECT_CALL(*g_roctx_trigger_mock,
                on_range_start(A<std::uint64_t>(), Eq(std::string_view{ "range_a" })))
        .Times(1)
        .WillOnce(SaveArg<0>(&range_id));
    expect_writes_allowed();
    EXPECT_CALL(*g_externals_mock, intern_string(Eq(std::string_view{ "range_a" })))
        .WillOnce(Return(11));
    expect_timemory_push("range_a");

    on_fresh_thread([&] {
        on_roctx_core_enter<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                          k_enter_ts);

        const auto& stack = detail::open_ranges<ext>::s_pushed;
        ASSERT_EQ(stack.size(), 1u);
        EXPECT_EQ(stack.top().name_id, 11u);
        EXPECT_EQ(stack.top().begin_timestamp, k_enter_ts);
        EXPECT_TRUE(stack.top().write_enabled);
        EXPECT_EQ(stack.top().range_id, range_id);
    });

    EXPECT_EQ(m_user_data.value, k_enter_ts);
}

TEST_F(roctx_core_api_test, range_push_enter_ids_decrement_from_previous_push)
{
    m_payload.args.roctxRangePushA.message = "range_a";
    sdk::marker_payload_t second_payload{};
    second_payload.args.roctxRangePushA.message = "range_b";
    const auto first =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangePushA, &m_payload);
    const auto second =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangePushA, &second_payload);
    std::uint64_t first_id  = 0;
    std::uint64_t second_id = 0;

    EXPECT_CALL(*g_roctx_trigger_mock,
                on_range_start(A<std::uint64_t>(), Eq(std::string_view{ "range_a" })))
        .Times(1)
        .WillOnce(SaveArg<0>(&first_id));
    EXPECT_CALL(*g_roctx_trigger_mock,
                on_range_start(A<std::uint64_t>(), Eq(std::string_view{ "range_b" })))
        .Times(1)
        .WillOnce(SaveArg<0>(&second_id));
    EXPECT_CALL(*g_roctx_trigger_mock, should_write_markers())
        .Times(2)
        .WillRepeatedly(Return(false));
    EXPECT_CALL(*g_externals_mock, intern_string(Eq(std::string_view{ "range_a" })))
        .WillOnce(Return(1));
    EXPECT_CALL(*g_externals_mock, intern_string(Eq(std::string_view{ "range_b" })))
        .WillOnce(Return(2));

    on_fresh_thread([&] {
        on_roctx_core_enter<sdk, ext, roctx_api_category>(first, &m_user_data, nullptr,
                                                          k_enter_ts);
        on_roctx_core_enter<sdk, ext, roctx_api_category>(second, &m_user_data, nullptr,
                                                          k_enter_ts);

        EXPECT_EQ(detail::open_ranges<ext>::s_pushed.size(), 2u);
    });

    EXPECT_EQ(second_id, first_id - 1);
}

TEST_F(roctx_core_api_test, range_push_enter_skips_begin_when_trigger_denies_writes)
{
    m_payload.args.roctxRangePushA.message = "range_a";
    const auto record =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangePushA, &m_payload);

    EXPECT_CALL(*g_roctx_trigger_mock,
                on_range_start(A<std::uint64_t>(), Eq(std::string_view{ "range_a" })))
        .Times(1);
    expect_writes_denied_by_trigger();
    EXPECT_CALL(*g_externals_mock, intern_string(Eq(std::string_view{ "range_a" })))
        .WillOnce(Return(11));

    on_fresh_thread([&] {
        on_roctx_core_enter<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                          k_enter_ts);

        const auto& stack = detail::open_ranges<ext>::s_pushed;
        ASSERT_EQ(stack.size(), 1u);
        EXPECT_FALSE(stack.top().write_enabled);
    });
}

TEST_F(roctx_core_api_test,
       range_push_enter_skips_begin_when_session_paused_by_other_trigger)
{
    m_payload.args.roctxRangePushA.message = "range_a";
    const auto record =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangePushA, &m_payload);

    EXPECT_CALL(*g_roctx_trigger_mock,
                on_range_start(A<std::uint64_t>(), Eq(std::string_view{ "range_a" })))
        .Times(1);
    expect_writes_denied_by_session();
    EXPECT_CALL(*g_externals_mock, intern_string(Eq(std::string_view{ "range_a" })))
        .WillOnce(Return(11));

    on_fresh_thread([&] {
        on_roctx_core_enter<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                          k_enter_ts);

        const auto& stack = detail::open_ranges<ext>::s_pushed;
        ASSERT_EQ(stack.size(), 1u);
        EXPECT_FALSE(stack.top().write_enabled);
    });
}

TEST_F(roctx_core_api_test,
       range_push_enter_tracks_range_without_timemory_push_when_disabled)
{
    m_payload.args.roctxRangePushA.message = "range_a";
    const auto record =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangePushA, &m_payload);

    EXPECT_CALL(*g_roctx_trigger_mock,
                on_range_start(A<std::uint64_t>(), Eq(std::string_view{ "range_a" })))
        .Times(1);
    expect_writes_allowed();
    EXPECT_CALL(*g_externals_mock, intern_string(Eq(std::string_view{ "range_a" })))
        .WillOnce(Return(11));
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(false));

    on_fresh_thread([&] {
        on_roctx_core_enter<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                          k_enter_ts);

        const auto& stack = detail::open_ranges<ext>::s_pushed;
        ASSERT_EQ(stack.size(), 1u);
        EXPECT_TRUE(stack.top().write_enabled);
    });
}

TEST_F(roctx_core_api_test, mark_enter_interns_name_and_writes_begin)
{
    m_payload.args.roctxMarkA.message = "mark_a";
    const auto record = make_core_record(sdk::MARKER_CORE_API_ID_roctxMarkA, &m_payload);

    EXPECT_CALL(*g_externals_mock, intern_string(Eq(std::string_view{ "mark_a" })))
        .WillOnce(Return(3));
    expect_writes_allowed();
    expect_timemory_push("mark_a");

    on_roctx_core_enter<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                      k_enter_ts);

    EXPECT_EQ(m_user_data.value, k_enter_ts);
}

TEST_F(roctx_core_api_test, mark_enter_interns_name_but_skips_begin_when_writes_denied)
{
    m_payload.args.roctxMarkA.message = "mark_a";
    const auto record = make_core_record(sdk::MARKER_CORE_API_ID_roctxMarkA, &m_payload);

    EXPECT_CALL(*g_externals_mock, intern_string(Eq(std::string_view{ "mark_a" })))
        .WillOnce(Return(3));
    expect_writes_denied_by_trigger();

    on_roctx_core_enter<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                      k_enter_ts);

    EXPECT_EQ(m_user_data.value, k_enter_ts);
}

TEST_F(roctx_core_api_test, range_start_pop_and_stop_enter_only_stamp_timestamp)
{
    for(const auto operation :
        { sdk::MARKER_CORE_API_ID_roctxRangeStartA, sdk::MARKER_CORE_API_ID_roctxRangePop,
          sdk::MARKER_CORE_API_ID_roctxRangeStop })
    {
        SCOPED_TRACE("operation: " + std::to_string(operation));

        const auto record = make_core_record(operation, &m_payload);
        m_user_data.value = 0;

        on_roctx_core_enter<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                          k_enter_ts);

        EXPECT_EQ(m_user_data.value, k_enter_ts);
    }
}

TEST_F(roctx_core_api_test, other_op_enter_writes_begin_with_table_name)
{
    const auto record = make_core_record(k_unmapped_operation, &m_payload);

    expect_writes_allowed();
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    expect_timemory_push(k_table_name);

    on_roctx_core_enter<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                      k_enter_ts);

    EXPECT_EQ(m_user_data.value, k_enter_ts);
}

TEST_F(roctx_core_api_test, other_op_enter_skips_begin_when_writes_denied)
{
    const auto record = make_core_record(k_unmapped_operation, &m_payload);

    expect_writes_denied_by_trigger();

    on_roctx_core_enter<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                      k_enter_ts);

    EXPECT_EQ(m_user_data.value, k_enter_ts);
}

TEST_F(roctx_core_api_test, enter_without_trigger_is_noop)
{
    g_roctx_trigger_mock.reset();
    m_payload.args.roctxRangePushA.message = "range_a";
    const auto record =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangePushA, &m_payload);
    constexpr std::uint64_t k_untouched = 42;
    m_user_data.value                   = k_untouched;

    on_fresh_thread([&] {
        on_roctx_core_enter<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                          k_enter_ts);

        EXPECT_TRUE(detail::open_ranges<ext>::s_pushed.empty());
    });

    EXPECT_EQ(m_user_data.value, k_untouched);
}

// ─── Core exit ──────────────────────────────────────────────────────────────────

TEST_F(roctx_core_api_test,
       range_pop_exit_writes_end_region_and_stops_range_with_pushed_id)
{
    const auto record =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangePop, &m_payload);
    constexpr std::uint64_t k_range_id = 555;
    constexpr std::uint64_t k_name_id  = 11;
    // The pushed range carries its own begin timestamp; user_data must be ignored.
    m_user_data.value = 9999;

    EXPECT_CALL(*g_externals_mock, lookup_string(Eq(k_name_id)))
        .WillOnce(Return("range_a"));
    expect_write_end(sdk::MARKER_CORE_API_ID_roctxRangePop, "range_a", k_enter_ts);
    EXPECT_CALL(*g_roctx_trigger_mock, on_range_stop(Eq(k_range_id))).Times(1);

    on_fresh_thread([&] {
        detail::open_ranges<ext>::s_pushed.push(
            { k_name_id, k_enter_ts, true, k_range_id });

        on_roctx_core_exit<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                         k_exit_ts);

        EXPECT_TRUE(detail::open_ranges<ext>::s_pushed.empty());
    });
}

TEST_F(roctx_core_api_test, range_pop_exit_with_empty_stack_does_not_call_trigger)
{
    const auto record =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangePop, &m_payload);

    on_fresh_thread([&] {
        on_roctx_core_exit<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                         k_exit_ts);

        EXPECT_TRUE(detail::open_ranges<ext>::s_pushed.empty());
    });
}

TEST_F(roctx_core_api_test,
       range_pop_exit_skips_write_when_range_opened_with_writes_disabled)
{
    const auto record =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangePop, &m_payload);
    constexpr std::uint64_t k_range_id = 556;
    constexpr std::uint64_t k_name_id  = 12;

    EXPECT_CALL(*g_externals_mock, lookup_string(Eq(k_name_id))).WillOnce(Return("x"));
    EXPECT_CALL(*g_roctx_trigger_mock, on_range_stop(Eq(k_range_id))).Times(1);

    on_fresh_thread([&] {
        detail::open_ranges<ext>::s_pushed.push(
            { k_name_id, k_enter_ts, false, k_range_id });

        on_roctx_core_exit<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                         k_exit_ts);

        EXPECT_TRUE(detail::open_ranges<ext>::s_pushed.empty());
    });
}

TEST_F(roctx_core_api_test, range_pop_exit_skips_write_when_name_lookup_fails)
{
    const auto record =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangePop, &m_payload);
    constexpr std::uint64_t k_range_id = 557;
    constexpr std::uint64_t k_name_id  = 13;

    EXPECT_CALL(*g_externals_mock, lookup_string(Eq(k_name_id)))
        .WillOnce(Return(nullptr));
    EXPECT_CALL(*g_roctx_trigger_mock, on_range_stop(Eq(k_range_id))).Times(1);

    on_fresh_thread([&] {
        detail::open_ranges<ext>::s_pushed.push(
            { k_name_id, k_enter_ts, true, k_range_id });

        on_roctx_core_exit<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                         k_exit_ts);

        EXPECT_TRUE(detail::open_ranges<ext>::s_pushed.empty());
    });
}

TEST_F(roctx_core_api_test, range_start_exit_registers_range_with_sdk_id_and_writes_begin)
{
    m_payload.args.roctxRangeStartA.message  = "start_a";
    m_payload.retval.roctx_range_id_t_retval = 77;
    const auto record =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangeStartA, &m_payload);
    m_user_data.value = k_enter_ts;

    EXPECT_CALL(*g_roctx_trigger_mock,
                on_range_start(Eq(77u), Eq(std::string_view{ "start_a" })))
        .Times(1);
    expect_writes_allowed();
    EXPECT_CALL(*g_externals_mock, intern_string(Eq(std::string_view{ "start_a" })))
        .WillOnce(Return(21));
    expect_timemory_push("start_a");

    on_fresh_thread([&] {
        on_roctx_core_exit<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                         k_exit_ts);

        const auto& stack = detail::open_ranges<ext>::s_started;
        ASSERT_EQ(stack.size(), 1u);
        EXPECT_EQ(stack.top().name_id, 21u);
        EXPECT_EQ(stack.top().begin_timestamp, k_enter_ts);
        EXPECT_TRUE(stack.top().write_enabled);
    });
}

TEST_F(roctx_core_api_test, range_start_exit_registers_range_without_write_when_denied)
{
    m_payload.args.roctxRangeStartA.message  = "start_a";
    m_payload.retval.roctx_range_id_t_retval = 77;
    const auto record =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangeStartA, &m_payload);
    m_user_data.value = k_enter_ts;

    EXPECT_CALL(*g_roctx_trigger_mock,
                on_range_start(Eq(77u), Eq(std::string_view{ "start_a" })))
        .Times(1);
    expect_writes_denied_by_trigger();
    EXPECT_CALL(*g_externals_mock, intern_string(Eq(std::string_view{ "start_a" })))
        .WillOnce(Return(21));

    on_fresh_thread([&] {
        on_roctx_core_exit<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                         k_exit_ts);

        const auto& stack = detail::open_ranges<ext>::s_started;
        ASSERT_EQ(stack.size(), 1u);
        EXPECT_FALSE(stack.top().write_enabled);
    });
}

TEST_F(roctx_core_api_test,
       range_stop_exit_writes_end_region_and_stops_range_with_payload_id)
{
    m_payload.args.roctxRangeStop.id = 77;
    const auto record =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangeStop, &m_payload);
    constexpr std::uint64_t k_name_id = 21;

    EXPECT_CALL(*g_externals_mock, lookup_string(Eq(k_name_id)))
        .WillOnce(Return("start_a"));
    expect_write_end(sdk::MARKER_CORE_API_ID_roctxRangeStop, "start_a", k_enter_ts);
    EXPECT_CALL(*g_roctx_trigger_mock, on_range_stop(Eq(77u))).Times(1);

    on_fresh_thread([&] {
        detail::open_ranges<ext>::s_started.push({ k_name_id, k_enter_ts, true });

        on_roctx_core_exit<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                         k_exit_ts);

        EXPECT_TRUE(detail::open_ranges<ext>::s_started.empty());
    });
}

TEST_F(roctx_core_api_test, range_stop_exit_with_empty_stack_does_not_call_trigger)
{
    m_payload.args.roctxRangeStop.id = 77;
    const auto record =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangeStop, &m_payload);

    on_fresh_thread([&] {
        on_roctx_core_exit<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                         k_exit_ts);

        EXPECT_TRUE(detail::open_ranges<ext>::s_started.empty());
    });
}

TEST_F(roctx_core_api_test, mark_exit_writes_end_with_message_and_enter_timestamp)
{
    m_payload.args.roctxMarkA.message = "mark_a";
    const auto record = make_core_record(sdk::MARKER_CORE_API_ID_roctxMarkA, &m_payload);
    m_user_data.value = k_enter_ts;

    expect_writes_allowed();
    expect_write_end(sdk::MARKER_CORE_API_ID_roctxMarkA, "mark_a", k_enter_ts);

    on_roctx_core_exit<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                     k_exit_ts);
}

TEST_F(roctx_core_api_test, mark_exit_skips_when_writes_denied)
{
    m_payload.args.roctxMarkA.message = "mark_a";
    const auto record = make_core_record(sdk::MARKER_CORE_API_ID_roctxMarkA, &m_payload);
    m_user_data.value = k_enter_ts;

    expect_writes_denied_by_trigger();

    on_roctx_core_exit<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                     k_exit_ts);
}

TEST_F(roctx_core_api_test, range_push_exit_does_nothing)
{
    const auto record =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangePushA, &m_payload);

    on_fresh_thread([&] {
        on_roctx_core_exit<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                         k_exit_ts);

        EXPECT_TRUE(detail::open_ranges<ext>::s_pushed.empty());
    });
}

TEST_F(roctx_core_api_test, other_op_exit_writes_end_with_table_name)
{
    const auto record = make_core_record(k_unmapped_operation, &m_payload);
    m_user_data.value = k_enter_ts;

    expect_writes_allowed();
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    expect_write_end(k_unmapped_operation, k_table_name, k_enter_ts);

    on_roctx_core_exit<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                     k_exit_ts);
}

TEST_F(roctx_core_api_test, other_op_exit_skips_when_writes_denied)
{
    const auto record = make_core_record(k_unmapped_operation, &m_payload);
    m_user_data.value = k_enter_ts;

    expect_writes_denied_by_trigger();

    on_roctx_core_exit<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                     k_exit_ts);
}

TEST_F(roctx_core_api_test, exit_without_trigger_is_noop)
{
    g_roctx_trigger_mock.reset();
    const auto record =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangePop, &m_payload);

    on_fresh_thread([&] {
        detail::open_ranges<ext>::s_pushed.push({ 11, k_enter_ts, true, 555 });

        on_roctx_core_exit<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                         k_exit_ts);

        EXPECT_EQ(detail::open_ranges<ext>::s_pushed.size(), 1u);
    });
}

// ─── write_end outputs (driven through the mark exit) ───────────────────────────

TEST_F(roctx_core_api_test, mark_exit_skips_timemory_pop_when_timemory_disabled)
{
    m_payload.args.roctxMarkA.message = "mark_a";
    const auto record = make_core_record(sdk::MARKER_CORE_API_ID_roctxMarkA, &m_payload);
    m_user_data.value = k_enter_ts;

    expect_writes_allowed();
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(false));
    expect_end_sinks(sdk::MARKER_CORE_API_ID_roctxMarkA, "mark_a", k_enter_ts);

    on_roctx_core_exit<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                     k_exit_ts);
}

TEST_F(roctx_core_api_test, mark_exit_stores_region_with_serialized_sdk_arguments)
{
    m_payload.args.roctxMarkA.message = "mark_a";
    const auto record = make_core_record(sdk::MARKER_CORE_API_ID_roctxMarkA, &m_payload);
    m_user_data.value = k_enter_ts;

    expect_writes_allowed();
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(false));
    expect_end_sinks(sdk::MARKER_CORE_API_ID_roctxMarkA, "mark_a", k_enter_ts,
                     k_serialized_arg);

    on_roctx_core_exit<sdk, ext, roctx_api_category>(record, &m_user_data, nullptr,
                                                     k_exit_ts);
}

// ─── Core dispatch ──────────────────────────────────────────────────────────────

TEST_F(roctx_core_api_test, core_on_record_dispatches_enter_exit_and_ignores_none_phase)
{
    constexpr const auto&   k_domain      = k_core_api<sdk, ext>;
    constexpr std::uint64_t k_dispatch_ts = 1234;

    // Exactly two timestamps are taken: one for ENTER, one for EXIT. NONE takes none.
    EXPECT_CALL(*g_tracing_backend_mock, get_timestamp())
        .WillOnce(Return(k_dispatch_ts))
        .WillOnce(Return(k_dispatch_ts + 1));

    auto enter  = make_core_record(sdk::MARKER_CORE_API_ID_roctxRangeStartA, &m_payload);
    enter.phase = sdk::CALLBACK_PHASE_ENTER;
    k_domain.on_record(enter, &m_user_data, nullptr);
    EXPECT_EQ(m_user_data.value, k_dispatch_ts);

    auto exit_record =
        make_core_record(sdk::MARKER_CORE_API_ID_roctxRangePushA, &m_payload);
    exit_record.phase = sdk::CALLBACK_PHASE_EXIT;
    on_fresh_thread([&] { k_domain.on_record(exit_record, &m_user_data, nullptr); });

    auto none  = make_core_record(sdk::MARKER_CORE_API_ID_roctxMarkA, &m_payload);
    none.phase = sdk::CALLBACK_PHASE_NONE;
    k_domain.on_record(none, &m_user_data, nullptr);
}

}  // namespace rocprofsys::domains::callback::roctx
