// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/common_tracing_callbacks.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

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

// Mirrors the per-domain Category traits (e.g. hip::runtime_api_category,
// hsa::core_api_category): bundles the timemory tag type and the buffer-storage
// category name that on_tracing_api_enter/exit look up via Category<Externals>. Reuses
// externals_with_tracing::rocm_hip_api_category / rocm_hip_api_category_name, same as
// production domain files do for their own Externals.
template <typename ExternalsT>
struct test_category
{
    using type = ExternalsT::rocm_hip_api_category;

    static constexpr std::string_view k_name = ExternalsT::rocm_hip_api_category_name;
};

// clang-tidy misclassifies GTest fixtures (SetUp/TearDown are virtual, TestBody is
// pure-virtual in the generated subclass) as an abstract class requiring an
// "_interface" suffix.
// NOLINTNEXTLINE(readability-identifier-naming)
class common_tracing_callbacks_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_tracing_backend_mock = std::make_unique<StrictMock<gmock_tracing_backend>>();
        g_externals_mock       = std::make_unique<StrictMock<gmock_externals>>();
    }

    void TearDown() override
    {
        g_tracing_backend_mock.reset();
        g_externals_mock.reset();
    }
};

}  // namespace

TEST_F(common_tracing_callbacks_test, enter_short_circuits_when_externals_are_inactive)
{
    EXPECT_CALL(*g_externals_mock, is_active()).WillOnce(Return(false));

    const test_support::callback_tracing_record_t record{};
    test_support::user_data_t                     user_data{};

    on_tracing_api_enter<mock_sdk_with_tracing, externals_with_tracing, test_category>(
        record, &user_data, nullptr);

    EXPECT_EQ(user_data.value, 0);
}

TEST_F(common_tracing_callbacks_test,
       enter_stores_timestamp_and_pushes_timemory_when_active)
{
    constexpr std::uint64_t k_timestamp = 42;

    EXPECT_CALL(*g_externals_mock, is_active()).WillOnce(Return(true));
    EXPECT_CALL(*g_tracing_backend_mock, get_timestamp()).WillOnce(Return(k_timestamp));
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(true));
    EXPECT_CALL(*g_externals_mock, tracing_push_timemory("operation"));

    const test_support::callback_tracing_record_t record{};
    test_support::user_data_t                     user_data{};

    on_tracing_api_enter<mock_sdk_with_tracing, externals_with_tracing, test_category>(
        record, &user_data, nullptr);

    EXPECT_EQ(user_data.value, k_timestamp);
}

TEST_F(common_tracing_callbacks_test, enter_skips_timemory_push_when_disabled)
{
    EXPECT_CALL(*g_externals_mock, is_active()).WillOnce(Return(true));
    EXPECT_CALL(*g_tracing_backend_mock, get_timestamp()).WillOnce(Return(1));
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(false));
    // No tracing_push_timemory expectation: StrictMock fails the test if it is
    // called anyway.

    const test_support::callback_tracing_record_t record{};
    test_support::user_data_t                     user_data{};

    on_tracing_api_enter<mock_sdk_with_tracing, externals_with_tracing, test_category>(
        record, &user_data, nullptr);
}

TEST_F(common_tracing_callbacks_test,
       exit_reads_timestamp_before_checking_active_and_returns_when_inactive)
{
    constexpr std::uint64_t k_timestamp = 7;

    EXPECT_CALL(*g_tracing_backend_mock, get_timestamp()).WillOnce(Return(k_timestamp));
    EXPECT_CALL(*g_externals_mock, is_active()).WillOnce(Return(false));

    const test_support::callback_tracing_record_t record{};
    test_support::user_data_t                     user_data{ .value = 1 };

    on_tracing_api_exit<mock_sdk_with_tracing, externals_with_tracing, test_category>(
        record, &user_data, nullptr);
}

TEST_F(common_tracing_callbacks_test,
       exit_stores_region_sample_built_from_backend_and_externals_values)
{
    constexpr std::uint64_t k_begin_timestamp = 10;
    constexpr std::uint64_t k_end_timestamp   = 20;
    constexpr std::uint64_t k_parent_stack_id = 99;
    constexpr std::int32_t  k_pid             = 111;
    constexpr std::int32_t  k_ppid            = 222;
    constexpr std::uint64_t k_thread_id       = 5;
    constexpr std::uint64_t k_kind            = 3;
    constexpr std::uint32_t k_operation       = 4;
    constexpr std::uint64_t k_correlation_id  = 77;

    test_support::callback_tracing_record_t record{};
    record.kind                    = k_kind;
    record.operation               = k_operation;
    record.thread_id               = k_thread_id;
    record.correlation_id.internal = k_correlation_id;

    test_support::user_data_t user_data{ .value = k_begin_timestamp };

    EXPECT_CALL(*g_tracing_backend_mock, get_timestamp())
        .WillOnce(Return(k_end_timestamp));
    EXPECT_CALL(*g_externals_mock, is_active()).WillOnce(Return(true));
    EXPECT_CALL(*g_externals_mock, check_backtrace_operations(k_kind, k_operation))
        .WillOnce(Return(true));
    EXPECT_CALL(*g_externals_mock, get_backtrace_data(true))
        .WillOnce(Return(std::optional<int>{ 1 }));
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(true));
    EXPECT_CALL(*g_externals_mock, tracing_pop_timemory("operation"));
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(k_kind, k_operation, _, _));
    EXPECT_CALL(*g_externals_mock, metadata_add_string("rocm_hip_api"));
    EXPECT_CALL(*g_externals_mock, get_ppid()).WillOnce(Return(k_ppid));
    EXPECT_CALL(*g_externals_mock, get_pid()).WillOnce(Return(k_pid));
    EXPECT_CALL(*g_externals_mock, metadata_add_thread_info(k_ppid, k_pid, k_thread_id));
    EXPECT_CALL(*g_tracing_backend_mock, get_parent_stack_id(_))
        .WillOnce(Return(k_parent_stack_id));
    EXPECT_CALL(*g_externals_mock,
                region_sample_buffer_storage_store(
                    k_thread_id, std::string{ "operation" }, k_correlation_id,
                    k_parent_stack_id, k_begin_timestamp, k_end_timestamp, std::string{},
                    std::string{ "rocm_hip_api" }));

    on_tracing_api_exit<mock_sdk_with_tracing, externals_with_tracing, test_category>(
        record, &user_data, nullptr);
}

// on_tracing_api_exit() never invokes iterate_args_callback itself: it hands the
// callback function pointer to SdkBackend::iterate_callback_tracing_kind_operation_args
// and lets the SDK invoke it per argument. Every other test in this file leaves
// iterate_args() as a no-op (matching how the shared mock behaves everywhere else),
// so iterate_args_callback's body never runs. This test invokes the real callback
// pointer via WillOnce(Invoke(...)) to exercise both its branches: a fully-populated
// argument (arg_type/arg_name/arg_value_str all non-null) that gets serialized into
// args_str, and one with a null field that must be skipped.
TEST_F(common_tracing_callbacks_test, exit_serializes_args_populated_via_iterate_callback)
{
    const test_support::callback_tracing_record_t record{};
    test_support::user_data_t                     user_data{};

    EXPECT_CALL(*g_tracing_backend_mock, get_timestamp()).WillOnce(Return(1));
    EXPECT_CALL(*g_externals_mock, is_active()).WillOnce(Return(true));
    EXPECT_CALL(*g_externals_mock, check_backtrace_operations(_, _))
        .WillOnce(Return(false));
    EXPECT_CALL(*g_externals_mock, get_backtrace_data(false))
        .WillOnce(Return(std::nullopt));
    EXPECT_CALL(*g_tracing_backend_mock, get_callback_tracing_names())
        .WillOnce(Return(tracing_names_t{}));
    EXPECT_CALL(*g_externals_mock, get_use_timemory()).WillOnce(Return(false));
    EXPECT_CALL(*g_tracing_backend_mock, iterate_args(_, _, _, _))
        .WillOnce([](std::uint64_t kind, std::uint32_t operation,
                     mock_sdk_with_tracing::callback_tracing_operation_args_cb_t callback,
                     void*                                                       data) {
            // Populated argument: exercises the "record it" branch.
            callback(kind, static_cast<std::int32_t>(operation), 0, nullptr, 0, "int",
                     "x", "42", 0, data);
            // A null field: exercises the "skip it" branch.
            callback(kind, static_cast<std::int32_t>(operation), 1, nullptr, 0, nullptr,
                     nullptr, nullptr, 0, data);
        });
    EXPECT_CALL(*g_externals_mock, metadata_add_string("rocm_hip_api"));
    EXPECT_CALL(*g_externals_mock, get_ppid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, get_pid()).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock, metadata_add_thread_info(_, _, _));
    EXPECT_CALL(*g_tracing_backend_mock, get_parent_stack_id(_)).WillOnce(Return(0));
    EXPECT_CALL(*g_externals_mock,
                region_sample_buffer_storage_store(_, _, _, _, _, _,
                                                   std::string{ "0;;int;;x;;42;;" }, _));

    on_tracing_api_exit<mock_sdk_with_tracing, externals_with_tracing, test_category>(
        record, &user_data, nullptr);
}

TEST_F(common_tracing_callbacks_test, on_tracing_api_configure_is_a_noop)
{
    // Not part of StrictMock<gmock_externals>: on_tracing_api_configure<Externals>()
    // must not touch any Externals member, so no expectations means any call fails.
    on_tracing_api_configure<externals_with_tracing>();
}

}  // namespace rocprofsys::domains::callback
