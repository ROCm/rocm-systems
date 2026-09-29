// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/ompt.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace rocprofsys::domains::callback
{
namespace
{

using ::testing::_;
using ::testing::Return;
using ::testing::StrictMock;

using test_support::externals_with_tracing;
using test_support::g_externals_mock;
using test_support::g_tracing_backend_mock;
using test_support::gmock_externals;
using test_support::gmock_tracing_backend;
using test_support::mock_sdk_with_tracing;
using test_support::tracing_names_t;

using sdk = mock_sdk_with_tracing;
using ext = externals_with_tracing;

sdk::callback_tracing_record_t
make_record(sdk::ompt_operation_t operation,
            sdk::callback_phase_t phase = sdk::CALLBACK_PHASE_NONE,
            std::uint64_t thread_id = 1, std::uint64_t correlation = 1)
{
    sdk::callback_tracing_record_t record{};
    record.operation               = static_cast<std::uint32_t>(operation);
    record.phase                   = phase;
    record.thread_id               = thread_id;
    record.correlation_id.internal = correlation;
    return record;
}

// clang-tidy misclassifies GTest fixtures (SetUp/TearDown are virtual, TestBody is
// pure-virtual in the generated subclass) as an abstract class requiring an
// "_interface" suffix.
// NOLINTNEXTLINE(readability-identifier-naming)
class ompt_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_tracing_backend_mock = std::make_unique<StrictMock<gmock_tracing_backend>>();
        g_externals_mock       = std::make_unique<StrictMock<gmock_externals>>();
        detail::get_ompt_standard_cb_storage<sdk>().clear();
        detail::get_ompt_parallel_cb_storage<sdk>().clear();
    }

    void TearDown() override
    {
        g_tracing_backend_mock.reset();
        g_externals_mock.reset();
        detail::get_ompt_standard_cb_storage<sdk>().clear();
        detail::get_ompt_parallel_cb_storage<sdk>().clear();
    }
};

}  // namespace

// ─── save_args ───────────────────────────────────────────────────────────────

TEST(ompt_save_args_test, appends_name_and_value_pair)
{
    detail::callback_arg_array_t args;
    const int                    rc =
        detail::save_args<sdk>(0, 0, 0, nullptr, 0, "int", "x", "42", 0, &args);

    EXPECT_EQ(rc, 0);
    ASSERT_EQ(args.size(), 1U);
    EXPECT_EQ(args[0].first, "x");
    EXPECT_EQ(args[0].second, "42");
}

// ─── ompt_get_unified_name ───────────────────────────────────────────────────

TEST_F(ompt_test, unified_name_uses_operation_table_for_non_parallel_operation)
{
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));

    const auto record = make_record(sdk::OMPT_ID_task_create);
    EXPECT_EQ(detail::ompt_get_unified_name<sdk>(record), "operation");
}

TEST_F(ompt_test, unified_name_is_omp_parallel_for_parallel_begin)
{
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));

    const auto record = make_record(sdk::OMPT_ID_parallel_begin);
    EXPECT_EQ(detail::ompt_get_unified_name<sdk>(record), "omp_parallel");
}

TEST_F(ompt_test, unified_name_is_omp_parallel_for_parallel_end)
{
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));

    const auto record = make_record(sdk::OMPT_ID_parallel_end);
    EXPECT_EQ(detail::ompt_get_unified_name<sdk>(record), "omp_parallel");
}

// ─── should_skip ─────────────────────────────────────────────────────────────

TEST(ompt_should_skip_test, null_payload_skips)
{
    auto record    = make_record(sdk::OMPT_ID_task_create);
    record.payload = nullptr;

    EXPECT_TRUE(detail::should_skip<sdk>(record));
}

TEST(ompt_should_skip_test, implicit_task_with_initial_flag_skips)
{
    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.implicit_task.flags =
        static_cast<int>(detail::external_types::ompt_task_flag_t::ompt_task_initial);

    auto record    = make_record(sdk::OMPT_ID_implicit_task);
    record.payload = &payload;

    EXPECT_TRUE(detail::should_skip<sdk>(record));
}

TEST(ompt_should_skip_test, implicit_task_without_initial_flag_does_not_skip)
{
    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.implicit_task.flags =
        static_cast<int>(detail::external_types::ompt_task_flag_t::ompt_task_implicit);

    auto record    = make_record(sdk::OMPT_ID_implicit_task);
    record.payload = &payload;

    EXPECT_FALSE(detail::should_skip<sdk>(record));
}

TEST(ompt_should_skip_test, thread_begin_with_initial_thread_skips)
{
    auto payload                          = sdk::callback_tracing_ompt_data_t{};
    payload.args.thread_begin.thread_type = sdk::ompt_thread_type_t::ompt_thread_initial;

    auto record    = make_record(sdk::OMPT_ID_thread_begin);
    record.payload = &payload;

    EXPECT_TRUE(detail::should_skip<sdk>(record));
}

TEST(ompt_should_skip_test, thread_begin_with_worker_thread_does_not_skip)
{
    auto payload                          = sdk::callback_tracing_ompt_data_t{};
    payload.args.thread_begin.thread_type = sdk::ompt_thread_type_t::ompt_thread_worker;

    auto record    = make_record(sdk::OMPT_ID_thread_begin);
    record.payload = &payload;

    EXPECT_FALSE(detail::should_skip<sdk>(record));
}

TEST(ompt_should_skip_test, unrelated_operation_does_not_skip)
{
    auto payload   = sdk::callback_tracing_ompt_data_t{};
    auto record    = make_record(sdk::OMPT_ID_parallel_begin);
    record.payload = &payload;

    EXPECT_FALSE(detail::should_skip<sdk>(record));
}

// ─── ompt_iterate_operation_args ─────────────────────────────────────────────

TEST_F(ompt_test, iterate_args_returns_early_for_operation_without_flags)
{
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload   = sdk::callback_tracing_ompt_data_t{};
    auto record    = make_record(sdk::OMPT_ID_dispatch);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    EXPECT_TRUE(args.empty());
}

TEST_F(ompt_test, iterate_args_returns_early_when_payload_null_for_flagged_operation)
{
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto record    = make_record(sdk::OMPT_ID_parallel_begin);
    record.payload = nullptr;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    EXPECT_TRUE(args.empty());
}

TEST_F(ompt_test, iterate_args_parallel_begin_program_invoker_and_league_cause)
{
    using detail::external_types::ompt_parallel_flag_t;
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.parallel_begin.flags =
        static_cast<int>(ompt_parallel_flag_t::ompt_parallel_invoker_program) |
        static_cast<int>(ompt_parallel_flag_t::ompt_parallel_league);

    auto record    = make_record(sdk::OMPT_ID_parallel_begin);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    ASSERT_EQ(args.size(), 2U);
    EXPECT_EQ(args[0], (std::pair<std::string, std::string>{ "invoker", "program" }));
    EXPECT_EQ(args[1], (std::pair<std::string, std::string>{ "invoker_cause",
                                                             "teams_construct" }));
}

TEST_F(ompt_test, iterate_args_parallel_end_runtime_invoker_and_team_cause)
{
    using detail::external_types::ompt_parallel_flag_t;
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.parallel_end.flags =
        static_cast<int>(ompt_parallel_flag_t::ompt_parallel_invoker_runtime) |
        static_cast<int>(ompt_parallel_flag_t::ompt_parallel_team);

    auto record    = make_record(sdk::OMPT_ID_parallel_end);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    ASSERT_EQ(args.size(), 2U);
    EXPECT_EQ(args[0], (std::pair<std::string, std::string>{ "invoker", "runtime" }));
    EXPECT_EQ(args[1], (std::pair<std::string, std::string>{ "invoker_cause",
                                                             "parallel_construct" }));
}

TEST_F(ompt_test, iterate_args_parallel_begin_without_flags_appends_nothing)
{
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload   = sdk::callback_tracing_ompt_data_t{};
    auto record    = make_record(sdk::OMPT_ID_parallel_begin);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    EXPECT_TRUE(args.empty());
}

TEST_F(ompt_test, iterate_args_task_create_initial_classification_with_untied_property)
{
    using detail::external_types::ompt_task_flag_t;
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.task_create.flags =
        static_cast<int>(ompt_task_flag_t::ompt_task_initial) |
        static_cast<int>(ompt_task_flag_t::ompt_task_untied);

    auto record    = make_record(sdk::OMPT_ID_task_create);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    ASSERT_EQ(args.size(), 2U);
    EXPECT_EQ(args[0],
              (std::pair<std::string, std::string>{ "classification", "initial" }));
    EXPECT_EQ(args[1], (std::pair<std::string, std::string>{ "properties", "untied" }));
}

TEST_F(ompt_test, iterate_args_task_create_implicit_classification_no_properties)
{
    using detail::external_types::ompt_task_flag_t;
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.task_create.flags =
        static_cast<int>(ompt_task_flag_t::ompt_task_implicit);

    auto record    = make_record(sdk::OMPT_ID_task_create);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    ASSERT_EQ(args.size(), 2U);
    EXPECT_EQ(args[0],
              (std::pair<std::string, std::string>{ "classification", "implicit" }));
    EXPECT_EQ(args[1], (std::pair<std::string, std::string>{ "properties", "none" }));
}

TEST_F(ompt_test, iterate_args_task_create_explicit_classification)
{
    using detail::external_types::ompt_task_flag_t;
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.task_create.flags =
        static_cast<int>(ompt_task_flag_t::ompt_task_explicit);

    auto record    = make_record(sdk::OMPT_ID_task_create);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    ASSERT_EQ(args.size(), 2U);
    EXPECT_EQ(args[0],
              (std::pair<std::string, std::string>{ "classification", "explicit" }));
}

TEST_F(ompt_test, iterate_args_task_create_target_classification)
{
    using detail::external_types::ompt_task_flag_t;
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload                   = sdk::callback_tracing_ompt_data_t{};
    payload.args.task_create.flags = static_cast<int>(ompt_task_flag_t::ompt_task_target);

    auto record    = make_record(sdk::OMPT_ID_task_create);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    ASSERT_EQ(args.size(), 2U);
    EXPECT_EQ(args[0],
              (std::pair<std::string, std::string>{ "classification", "target" }));
}

TEST_F(ompt_test, iterate_args_task_create_all_properties_and_no_classification)
{
    using detail::external_types::ompt_task_flag_t;
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.task_create.flags =
        static_cast<int>(ompt_task_flag_t::ompt_task_undeferred) |
        static_cast<int>(ompt_task_flag_t::ompt_task_untied) |
        static_cast<int>(ompt_task_flag_t::ompt_task_final) |
        static_cast<int>(ompt_task_flag_t::ompt_task_mergeable) |
        static_cast<int>(ompt_task_flag_t::ompt_task_merged);

    auto record    = make_record(sdk::OMPT_ID_task_create);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    // No classification bit set -> only the properties entry is appended.
    ASSERT_EQ(args.size(), 1U);
    EXPECT_EQ(args[0],
              (std::pair<std::string, std::string>{
                  "properties", "undeferred, untied, final, mergeable, merged" }));
}

TEST_F(ompt_test, iterate_args_implicit_task_initial_kind)
{
    using detail::external_types::ompt_task_flag_t;
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.implicit_task.flags =
        static_cast<int>(ompt_task_flag_t::ompt_task_initial);

    auto record    = make_record(sdk::OMPT_ID_implicit_task);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    ASSERT_EQ(args.size(), 1U);
    EXPECT_EQ(args[0], (std::pair<std::string, std::string>{ "kind", "initial" }));
}

TEST_F(ompt_test, iterate_args_implicit_task_implicit_kind)
{
    using detail::external_types::ompt_task_flag_t;
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.implicit_task.flags =
        static_cast<int>(ompt_task_flag_t::ompt_task_implicit);

    auto record    = make_record(sdk::OMPT_ID_implicit_task);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    ASSERT_EQ(args.size(), 1U);
    EXPECT_EQ(args[0], (std::pair<std::string, std::string>{ "kind", "implicit" }));
}

TEST_F(ompt_test, iterate_args_implicit_task_neither_kind_appends_nothing)
{
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload   = sdk::callback_tracing_ompt_data_t{};
    auto record    = make_record(sdk::OMPT_ID_implicit_task);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    EXPECT_TRUE(args.empty());
}

TEST_F(ompt_test, iterate_args_cancel_parallel_construct_and_activated_state)
{
    using detail::external_types::ompt_cancel_flag_t;
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.cancel.flags =
        static_cast<int>(ompt_cancel_flag_t::ompt_cancel_parallel) |
        static_cast<int>(ompt_cancel_flag_t::ompt_cancel_activated);

    auto record    = make_record(sdk::OMPT_ID_cancel);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    ASSERT_EQ(args.size(), 2U);
    EXPECT_EQ(args[0], (std::pair<std::string, std::string>{ "construct", "parallel" }));
    EXPECT_EQ(args[1], (std::pair<std::string, std::string>{ "state", "activated" }));
}

TEST_F(ompt_test, iterate_args_cancel_sections_construct_and_detected_state)
{
    using detail::external_types::ompt_cancel_flag_t;
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.cancel.flags =
        static_cast<int>(ompt_cancel_flag_t::ompt_cancel_sections) |
        static_cast<int>(ompt_cancel_flag_t::ompt_cancel_detected);

    auto record    = make_record(sdk::OMPT_ID_cancel);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    ASSERT_EQ(args.size(), 2U);
    EXPECT_EQ(args[0], (std::pair<std::string, std::string>{ "construct", "sections" }));
    EXPECT_EQ(args[1], (std::pair<std::string, std::string>{ "state", "detected" }));
}

TEST_F(ompt_test, iterate_args_cancel_loop_construct_and_discarded_task_state)
{
    using detail::external_types::ompt_cancel_flag_t;
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.cancel.flags =
        static_cast<int>(ompt_cancel_flag_t::ompt_cancel_loop) |
        static_cast<int>(ompt_cancel_flag_t::ompt_cancel_discarded_task);

    auto record    = make_record(sdk::OMPT_ID_cancel);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    ASSERT_EQ(args.size(), 2U);
    EXPECT_EQ(args[0], (std::pair<std::string, std::string>{ "construct", "loop" }));
    EXPECT_EQ(args[1],
              (std::pair<std::string, std::string>{ "state", "discarded_task" }));
}

TEST_F(ompt_test, iterate_args_cancel_taskgroup_construct_without_state)
{
    using detail::external_types::ompt_cancel_flag_t;
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.cancel.flags =
        static_cast<int>(ompt_cancel_flag_t::ompt_cancel_taskgroup);

    auto record    = make_record(sdk::OMPT_ID_cancel);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    ASSERT_EQ(args.size(), 1U);
    EXPECT_EQ(args[0], (std::pair<std::string, std::string>{ "construct", "taskgroup" }));
}

TEST_F(ompt_test, iterate_args_cancel_without_flags_appends_nothing)
{
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload   = sdk::callback_tracing_ompt_data_t{};
    auto record    = make_record(sdk::OMPT_ID_cancel);
    record.payload = &payload;

    detail::callback_arg_array_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    EXPECT_TRUE(args.empty());
}

TEST_F(ompt_test, iterate_args_supports_function_args_t_container)
{
    using detail::external_types::ompt_task_flag_t;
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.task_create.flags =
        static_cast<int>(ompt_task_flag_t::ompt_task_explicit);

    auto record    = make_record(sdk::OMPT_ID_task_create);
    record.payload = &payload;

    function_args_t args;
    detail::ompt_iterate_operation_args<sdk>(record, args);

    ASSERT_EQ(args.size(), 2U);
    EXPECT_EQ(args[0].arg_name, "classification");
    EXPECT_EQ(args[0].arg_value, "explicit");
    EXPECT_EQ(args[1].arg_name, "properties");
    EXPECT_EQ(args[1].arg_value, "none");
}

// ─── ompt_emit_region ────────────────────────────────────────────────────────

TEST_F(ompt_test, emit_region_builds_expected_region_sample)
{
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_tracing_backend_mock, get_parent_stack_id(_))
        .WillOnce(Return(std::uint64_t{ 55 }));
    EXPECT_CALL(*g_externals_mock, metadata_add_string("rocm_ompt_api"));
    EXPECT_CALL(*g_externals_mock, get_ppid()).WillOnce(Return(2));
    EXPECT_CALL(*g_externals_mock, get_pid()).WillOnce(Return(3));
    EXPECT_CALL(*g_externals_mock, metadata_add_thread_info(2, 3, 7));
    EXPECT_CALL(*g_externals_mock, region_sample_buffer_storage_store(
                                       7, std::string{ "operation" }, 9, 55, 10, 20,
                                       std::string{}, std::string{ "rocm_ompt_api" }));

    auto record = make_record(sdk::OMPT_ID_task_create, sdk::CALLBACK_PHASE_NONE, 7, 9);
    auto backtrace_data  = std::optional<int>{};
    function_args_t args = {};

    detail::ompt_emit_region<sdk, ext, ompt_api_category>(record, 10, 20, backtrace_data,
                                                          args);
}

// ─── ompt_cache_instant_event / ompt_cache_orphan_event ─────────────────────

TEST_F(ompt_test, cache_instant_event_uses_same_begin_and_end_timestamp)
{
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_tracing_backend_mock, get_parent_stack_id(_)).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_string(_));
    EXPECT_CALL(*g_externals_mock, get_ppid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, get_pid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_thread_info(_, _, _));
    EXPECT_CALL(*g_externals_mock,
                region_sample_buffer_storage_store(_, _, _, _, 15, 15, _, _));

    auto payload   = sdk::callback_tracing_ompt_data_t{};
    auto record    = make_record(sdk::OMPT_ID_lock_init);
    record.payload = &payload;

    auto backtrace_data = std::optional<int>{};
    detail::ompt_cache_instant_event<sdk, ext, ompt_api_category>(record, 15,
                                                                  backtrace_data);
}

TEST_F(ompt_test, cache_orphan_event_reuses_stored_begin_timestamp_for_both_sides)
{
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_tracing_backend_mock, get_parent_stack_id(_)).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_string(_));
    EXPECT_CALL(*g_externals_mock, get_ppid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, get_pid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_thread_info(_, _, _));
    EXPECT_CALL(*g_externals_mock,
                region_sample_buffer_storage_store(_, _, _, _, 25, 25, _, _));

    const auto record = make_record(sdk::OMPT_ID_lock_init);
    const auto stored =
        detail::rocprofsys_ompt_data_storage_t<sdk>{ record, 25, function_args_t{} };

    auto backtrace_data = std::optional<int>{};
    detail::ompt_cache_orphan_event<sdk, ext, ompt_api_category>(stored, backtrace_data);
}

// ─── push/pop standard callback ──────────────────────────────────────────────

TEST_F(ompt_test, push_then_pop_standard_callback_found_path_uses_stored_begin_timestamp)
{
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    auto record = make_record(sdk::OMPT_ID_task_create, sdk::CALLBACK_PHASE_ENTER, 4, 42);
    record.payload = &payload;

    detail::ompt_push_standard_callback<sdk>(record, 100);

    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_tracing_backend_mock, get_parent_stack_id(_)).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_string(_));
    EXPECT_CALL(*g_externals_mock, get_ppid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, get_pid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_thread_info(_, _, _));
    EXPECT_CALL(*g_externals_mock,
                region_sample_buffer_storage_store(_, _, _, _, 100, 200, _, _));

    auto backtrace_data = std::optional<int>{};
    detail::ompt_pop_standard_callback<sdk, ext, ompt_api_category>(record, 200,
                                                                    backtrace_data);

    EXPECT_TRUE(detail::get_ompt_standard_cb_storage<sdk>().empty());
}

TEST_F(ompt_test, pop_standard_callback_without_matching_push_emits_orphan_instant_event)
{
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_tracing_backend_mock, get_parent_stack_id(_)).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_string(_));
    EXPECT_CALL(*g_externals_mock, get_ppid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, get_pid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_thread_info(_, _, _));
    EXPECT_CALL(*g_externals_mock,
                region_sample_buffer_storage_store(_, _, _, _, 300, 300, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    auto record = make_record(sdk::OMPT_ID_task_create, sdk::CALLBACK_PHASE_EXIT, 4, 999);
    record.payload = &payload;

    auto backtrace_data = std::optional<int>{};
    detail::ompt_pop_standard_callback<sdk, ext, ompt_api_category>(record, 300,
                                                                    backtrace_data);
}

// ─── push/pop parallel callback ──────────────────────────────────────────────

TEST_F(ompt_test, push_then_pop_parallel_callback_found_path)
{
    int fake_parallel_data = 0;

    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto push_payload                              = sdk::callback_tracing_ompt_data_t{};
    push_payload.args.parallel_begin.parallel_data = &fake_parallel_data;
    auto push_record =
        make_record(sdk::OMPT_ID_parallel_begin, sdk::CALLBACK_PHASE_NONE, 1, 1);
    push_record.payload = &push_payload;

    detail::ompt_push_parallel_callback<sdk>(push_record, 10);

    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_tracing_backend_mock, get_parent_stack_id(_)).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_string(_));
    EXPECT_CALL(*g_externals_mock, get_ppid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, get_pid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_thread_info(_, _, _));
    EXPECT_CALL(*g_externals_mock,
                region_sample_buffer_storage_store(_, _, _, _, 10, 20, _, _));

    auto pop_payload                            = sdk::callback_tracing_ompt_data_t{};
    pop_payload.args.parallel_end.parallel_data = &fake_parallel_data;
    auto pop_record =
        make_record(sdk::OMPT_ID_parallel_end, sdk::CALLBACK_PHASE_NONE, 1, 1);
    pop_record.payload = &pop_payload;

    auto backtrace_data = std::optional<int>{};
    detail::ompt_pop_parallel_callback<sdk, ext, ompt_api_category>(pop_record, 20,
                                                                    backtrace_data);

    EXPECT_TRUE(detail::get_ompt_parallel_cb_storage<sdk>().empty());
}

TEST_F(ompt_test, pop_parallel_callback_without_matching_push_emits_orphan)
{
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_tracing_backend_mock, get_parent_stack_id(_)).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_string(_));
    EXPECT_CALL(*g_externals_mock, get_ppid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, get_pid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_thread_info(_, _, _));
    EXPECT_CALL(*g_externals_mock,
                region_sample_buffer_storage_store(_, _, _, _, 50, 50, _, _));

    int  unmatched                          = 0;
    auto payload                            = sdk::callback_tracing_ompt_data_t{};
    payload.args.parallel_end.parallel_data = &unmatched;
    auto record = make_record(sdk::OMPT_ID_parallel_end, sdk::CALLBACK_PHASE_NONE, 1, 1);
    record.payload = &payload;

    auto backtrace_data = std::optional<int>{};
    detail::ompt_pop_parallel_callback<sdk, ext, ompt_api_category>(record, 50,
                                                                    backtrace_data);
}

// ─── ompt_finalize_orphan_events ─────────────────────────────────────────────

TEST_F(ompt_test, finalize_orphan_events_emits_and_clears_both_storages)
{
    const auto record1 =
        make_record(sdk::OMPT_ID_task_create, sdk::CALLBACK_PHASE_ENTER, 1, 11);
    detail::get_ompt_standard_cb_storage<sdk>().emplace(
        11U,
        detail::rocprofsys_ompt_data_storage_t<sdk>{ record1, 5, function_args_t{} });

    const auto record2 =
        make_record(sdk::OMPT_ID_parallel_begin, sdk::CALLBACK_PHASE_NONE, 1, 1);
    detail::get_ompt_parallel_cb_storage<sdk>().emplace(
        uintptr_t{ 0x1234 },
        detail::rocprofsys_ompt_data_storage_t<sdk>{ record2, 6, function_args_t{} });

    EXPECT_CALL(*g_externals_mock, get_backtrace_data(false))
        .WillOnce(Return(std::nullopt));
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .Times(2)
        .WillRepeatedly(Return(tracing_names_t{}));
    EXPECT_CALL(*g_tracing_backend_mock, get_parent_stack_id(_))
        .Times(2)
        .WillRepeatedly(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_string(_)).Times(2);
    EXPECT_CALL(*g_externals_mock, get_ppid()).Times(2).WillRepeatedly(Return(0));
    EXPECT_CALL(*g_externals_mock, get_pid()).Times(2).WillRepeatedly(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_thread_info(_, _, _)).Times(2);
    EXPECT_CALL(*g_externals_mock,
                region_sample_buffer_storage_store(_, _, _, _, _, _, _, _))
        .Times(2);

    detail::ompt_finalize_orphan_events<sdk, ext, ompt_api_category>();

    EXPECT_TRUE(detail::get_ompt_standard_cb_storage<sdk>().empty());
    EXPECT_TRUE(detail::get_ompt_parallel_cb_storage<sdk>().empty());
}

// ─── ompt_tracing_callback_start / ompt_tracing_callback_stop ───────────────

TEST_F(ompt_test, tracing_callback_start_pushes_timemory_when_enabled)
{
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(true));
    EXPECT_CALL(*g_externals_mock, tracing_push_timemory("operation"));

    const auto       record = make_record(sdk::OMPT_ID_task_create);
    sdk::user_data_t user_data{};
    detail::ompt_tracing_callback_start<sdk, ext, ompt_api_category>(record, &user_data,
                                                                     0);
}

TEST_F(ompt_test, tracing_callback_start_skips_timemory_when_disabled)
{
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(false));

    const auto       record = make_record(sdk::OMPT_ID_task_create);
    sdk::user_data_t user_data{};
    detail::ompt_tracing_callback_start<sdk, ext, ompt_api_category>(record, &user_data,
                                                                     0);
}

TEST_F(ompt_test, tracing_callback_stop_pops_timemory_when_enabled)
{
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(true));
    EXPECT_CALL(*g_externals_mock, tracing_pop_timemory("operation"));

    const auto       record = make_record(sdk::OMPT_ID_task_create);
    sdk::user_data_t user_data{};
    detail::ompt_tracing_callback_stop<sdk, ext, ompt_api_category>(record, &user_data,
                                                                    0);
}

TEST_F(ompt_test, tracing_callback_stop_skips_timemory_when_disabled)
{
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(false));

    const auto       record = make_record(sdk::OMPT_ID_task_create);
    sdk::user_data_t user_data{};
    detail::ompt_tracing_callback_stop<sdk, ext, ompt_api_category>(record, &user_data,
                                                                    0);
}

// ─── on_ompt_configure ───────────────────────────────────────────────────────

TEST(ompt_configure_test, is_a_noop) { on_ompt_configure<ext>(); }

// ─── on_ompt_enter ───────────────────────────────────────────────────────────

TEST_F(ompt_test, enter_skips_when_should_skip_true)
{
    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.implicit_task.flags =
        static_cast<int>(detail::external_types::ompt_task_flag_t::ompt_task_initial);
    auto record    = make_record(sdk::OMPT_ID_implicit_task, sdk::CALLBACK_PHASE_ENTER);
    record.payload = &payload;

    sdk::user_data_t user_data{};
    on_ompt_enter<sdk, ext, ompt_api_category>(record, &user_data, nullptr, 0);
}

TEST_F(ompt_test, enter_starts_tracing_and_pushes_standard_callback_when_not_skipped)
{
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(false));
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    auto record = make_record(sdk::OMPT_ID_task_create, sdk::CALLBACK_PHASE_ENTER, 1, 55);
    record.payload = &payload;

    sdk::user_data_t user_data{};
    on_ompt_enter<sdk, ext, ompt_api_category>(record, &user_data, nullptr, 123);

    EXPECT_TRUE(detail::get_ompt_standard_cb_storage<sdk>().contains(55U));
}

// ─── on_ompt_exit ────────────────────────────────────────────────────────────

TEST_F(ompt_test, exit_skips_when_should_skip_true)
{
    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.implicit_task.flags =
        static_cast<int>(detail::external_types::ompt_task_flag_t::ompt_task_initial);
    auto record    = make_record(sdk::OMPT_ID_implicit_task, sdk::CALLBACK_PHASE_EXIT);
    record.payload = &payload;

    sdk::user_data_t user_data{};
    on_ompt_exit<sdk, ext, ompt_api_category>(record, &user_data, nullptr, 0);
}

TEST_F(ompt_test, exit_pops_found_standard_callback_and_emits_span)
{
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    auto enter_record =
        make_record(sdk::OMPT_ID_task_create, sdk::CALLBACK_PHASE_ENTER, 1, 77);
    enter_record.payload = &payload;
    detail::ompt_push_standard_callback<sdk>(enter_record, 10);

    EXPECT_CALL(*g_externals_mock, check_backtrace_operations(_, _))
        .WillOnce(Return(false));
    EXPECT_CALL(*g_externals_mock, get_backtrace_data(false))
        .WillOnce(Return(std::nullopt));
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .Times(2)
        .WillRepeatedly(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(false));
    EXPECT_CALL(*g_tracing_backend_mock, get_parent_stack_id(_)).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_string(_));
    EXPECT_CALL(*g_externals_mock, get_ppid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, get_pid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_thread_info(_, _, _));
    EXPECT_CALL(*g_externals_mock,
                region_sample_buffer_storage_store(_, _, _, _, 10, 20, _, _));

    auto exit_record =
        make_record(sdk::OMPT_ID_task_create, sdk::CALLBACK_PHASE_EXIT, 1, 77);
    exit_record.payload = &payload;
    sdk::user_data_t user_data{};
    on_ompt_exit<sdk, ext, ompt_api_category>(exit_record, &user_data, nullptr, 20);
}

TEST_F(ompt_test, exit_without_matching_enter_emits_orphan_event)
{
    EXPECT_CALL(*g_externals_mock, check_backtrace_operations(_, _))
        .WillOnce(Return(false));
    EXPECT_CALL(*g_externals_mock, get_backtrace_data(false))
        .WillOnce(Return(std::nullopt));
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .Times(2)
        .WillRepeatedly(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(false));
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));
    EXPECT_CALL(*g_tracing_backend_mock, get_parent_stack_id(_)).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_string(_));
    EXPECT_CALL(*g_externals_mock, get_ppid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, get_pid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_thread_info(_, _, _));
    EXPECT_CALL(*g_externals_mock,
                region_sample_buffer_storage_store(_, _, _, _, 60, 60, _, _));

    auto payload = sdk::callback_tracing_ompt_data_t{};
    auto record = make_record(sdk::OMPT_ID_task_create, sdk::CALLBACK_PHASE_EXIT, 1, 321);
    record.payload = &payload;

    sdk::user_data_t user_data{};
    on_ompt_exit<sdk, ext, ompt_api_category>(record, &user_data, nullptr, 60);
}

// ─── on_ompt_none ────────────────────────────────────────────────────────────

TEST_F(ompt_test, none_skips_when_should_skip_true)
{
    auto payload = sdk::callback_tracing_ompt_data_t{};
    payload.args.implicit_task.flags =
        static_cast<int>(detail::external_types::ompt_task_flag_t::ompt_task_initial);
    auto record    = make_record(sdk::OMPT_ID_implicit_task);
    record.payload = &payload;

    sdk::user_data_t user_data{};
    on_ompt_none<sdk, ext, ompt_api_category>(record, &user_data, nullptr, 0);
}

TEST_F(ompt_test, none_ignores_callback_functions_marker)
{
    auto payload   = sdk::callback_tracing_ompt_data_t{};
    auto record    = make_record(sdk::OMPT_ID_callback_functions);
    record.payload = &payload;

    sdk::user_data_t user_data{};
    on_ompt_none<sdk, ext, ompt_api_category>(record, &user_data, nullptr, 0);
}

TEST_F(ompt_test, none_ignores_thread_end)
{
    auto payload   = sdk::callback_tracing_ompt_data_t{};
    auto record    = make_record(sdk::OMPT_ID_thread_end);
    record.payload = &payload;

    sdk::user_data_t user_data{};
    on_ompt_none<sdk, ext, ompt_api_category>(record, &user_data, nullptr, 0);
}

TEST_F(ompt_test, none_dispatches_parallel_begin_and_pushes_parallel_callback)
{
    EXPECT_CALL(*g_externals_mock, check_backtrace_operations(_, _))
        .WillOnce(Return(false));
    EXPECT_CALL(*g_externals_mock, get_backtrace_data(false))
        .WillOnce(Return(std::nullopt));
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(false));
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));

    int  fake_parallel_data                   = 0;
    auto payload                              = sdk::callback_tracing_ompt_data_t{};
    payload.args.parallel_begin.parallel_data = &fake_parallel_data;
    auto record                               = make_record(sdk::OMPT_ID_parallel_begin);
    record.payload                            = &payload;

    sdk::user_data_t user_data{};
    on_ompt_none<sdk, ext, ompt_api_category>(record, &user_data, nullptr, 40);

    EXPECT_TRUE(detail::get_ompt_parallel_cb_storage<sdk>().contains(
        reinterpret_cast<uintptr_t>(&fake_parallel_data)));
}

TEST_F(ompt_test, none_dispatches_parallel_end_and_pops_parallel_callback)
{
    int fake_parallel_data = 0;

    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));
    auto push_payload                              = sdk::callback_tracing_ompt_data_t{};
    push_payload.args.parallel_begin.parallel_data = &fake_parallel_data;
    auto push_record    = make_record(sdk::OMPT_ID_parallel_begin);
    push_record.payload = &push_payload;
    detail::ompt_push_parallel_callback<sdk>(push_record, 5);

    EXPECT_CALL(*g_externals_mock, check_backtrace_operations(_, _))
        .WillOnce(Return(false));
    EXPECT_CALL(*g_externals_mock, get_backtrace_data(false))
        .WillOnce(Return(std::nullopt));
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .Times(2)
        .WillRepeatedly(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(false));
    EXPECT_CALL(*g_tracing_backend_mock, get_parent_stack_id(_)).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_string(_));
    EXPECT_CALL(*g_externals_mock, get_ppid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, get_pid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_thread_info(_, _, _));
    EXPECT_CALL(*g_externals_mock,
                region_sample_buffer_storage_store(_, _, _, _, 5, 15, _, _));

    auto pop_payload                            = sdk::callback_tracing_ompt_data_t{};
    pop_payload.args.parallel_end.parallel_data = &fake_parallel_data;
    auto pop_record                             = make_record(sdk::OMPT_ID_parallel_end);
    pop_record.payload                          = &pop_payload;

    sdk::user_data_t user_data{};
    on_ompt_none<sdk, ext, ompt_api_category>(pop_record, &user_data, nullptr, 15);
}

TEST_F(ompt_test, none_dispatches_instant_event_for_lock_init)
{
    EXPECT_CALL(*g_externals_mock, check_backtrace_operations(_, _))
        .WillOnce(Return(false));
    EXPECT_CALL(*g_externals_mock, get_backtrace_data(false))
        .WillOnce(Return(std::nullopt));
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .Times(3)
        .WillRepeatedly(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory())
        .Times(2)
        .WillRepeatedly(Return(false));
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));
    EXPECT_CALL(*g_tracing_backend_mock, get_parent_stack_id(_)).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_string(_));
    EXPECT_CALL(*g_externals_mock, get_ppid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, get_pid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_thread_info(_, _, _));
    EXPECT_CALL(*g_externals_mock,
                region_sample_buffer_storage_store(_, _, _, _, 30, 30, _, _));

    auto payload   = sdk::callback_tracing_ompt_data_t{};
    auto record    = make_record(sdk::OMPT_ID_lock_init);
    record.payload = &payload;

    sdk::user_data_t user_data{};
    on_ompt_none<sdk, ext, ompt_api_category>(record, &user_data, nullptr, 30);
}

TEST_F(ompt_test, none_dispatches_instant_event_for_thread_begin)
{
    EXPECT_CALL(*g_externals_mock, check_backtrace_operations(_, _))
        .WillOnce(Return(false));
    EXPECT_CALL(*g_externals_mock, get_backtrace_data(false))
        .WillOnce(Return(std::nullopt));
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .Times(3)
        .WillRepeatedly(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory())
        .Times(2)
        .WillRepeatedly(Return(false));
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _));
    EXPECT_CALL(*g_tracing_backend_mock, get_parent_stack_id(_)).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_string(_));
    EXPECT_CALL(*g_externals_mock, get_ppid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, get_pid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_thread_info(_, _, _));
    EXPECT_CALL(*g_externals_mock,
                region_sample_buffer_storage_store(_, _, _, _, 12, 12, _, _));

    auto payload                          = sdk::callback_tracing_ompt_data_t{};
    payload.args.thread_begin.thread_type = sdk::ompt_thread_type_t::ompt_thread_worker;
    auto record                           = make_record(sdk::OMPT_ID_thread_begin);
    record.payload                        = &payload;

    sdk::user_data_t user_data{};
    on_ompt_none<sdk, ext, ompt_api_category>(record, &user_data, nullptr, 12);
}

TEST_F(ompt_test, none_logs_warning_for_unhandled_operation)
{
    EXPECT_CALL(*g_externals_mock, check_backtrace_operations(_, _))
        .WillOnce(Return(false));
    EXPECT_CALL(*g_externals_mock, get_backtrace_data(false))
        .WillOnce(Return(std::nullopt));

    auto payload   = sdk::callback_tracing_ompt_data_t{};
    auto record    = make_record(9999);
    record.payload = &payload;

    sdk::user_data_t user_data{};
    on_ompt_none<sdk, ext, ompt_api_category>(record, &user_data, nullptr, 0);
}

// ─── k_ompt_api domain descriptor ────────────────────────────────────────────

TEST(ompt_domain_test, metadata_matches_ompt_domain)
{
    constexpr const auto& domain = k_ompt_api<sdk, ext>;

    EXPECT_EQ(domain.meta.name, "ompt");
    EXPECT_EQ(domain.meta.id, sdk::CALLBACK_TRACING_OMPT);
    EXPECT_EQ(domain.meta.mode, collection_mode::callback);
    EXPECT_FALSE(domain.meta.group.has_value());
}

}  // namespace rocprofsys::domains::callback
