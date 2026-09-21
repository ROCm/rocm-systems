// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gtest/gtest.h>

namespace rocprofsys::domains
{
namespace
{

using test_support::callback_tracing_record_t;
using test_support::mock_sdk;
using test_support::user_data_t;

int g_enter_calls = 0;
int g_exit_calls  = 0;
int g_none_calls  = 0;

void
reset_counters()
{
    g_enter_calls = 0;
    g_exit_calls  = 0;
    g_none_calls  = 0;
}

void
record_enter(callback_tracing_record_t, user_data_t*, void*)
{
    ++g_enter_calls;
}

void
record_exit(callback_tracing_record_t, user_data_t*, void*)
{
    ++g_exit_calls;
}

void
record_none(callback_tracing_record_t, user_data_t*, void*)
{
    ++g_none_calls;
}

using sut_t =
    tracing_callback_dispatcher<mock_sdk, record_enter, record_exit, record_none>;

}  // namespace

TEST(tracing_callback_dispatcher_test, dispatches_enter_phase_to_on_enter_only)
{
    reset_counters();
    auto record  = callback_tracing_record_t{};
    record.phase = mock_sdk::CALLBACK_PHASE_ENTER;

    sut_t::callback(record, nullptr, nullptr);

    EXPECT_EQ(g_enter_calls, 1);
    EXPECT_EQ(g_exit_calls, 0);
    EXPECT_EQ(g_none_calls, 0);
}

TEST(tracing_callback_dispatcher_test, dispatches_exit_phase_to_on_exit_only)
{
    reset_counters();
    auto record  = callback_tracing_record_t{};
    record.phase = mock_sdk::CALLBACK_PHASE_EXIT;

    sut_t::callback(record, nullptr, nullptr);

    EXPECT_EQ(g_enter_calls, 0);
    EXPECT_EQ(g_exit_calls, 1);
    EXPECT_EQ(g_none_calls, 0);
}

TEST(tracing_callback_dispatcher_test, dispatches_none_phase_to_on_none_only)
{
    reset_counters();
    auto record  = callback_tracing_record_t{};
    record.phase = mock_sdk::CALLBACK_PHASE_NONE;

    sut_t::callback(record, nullptr, nullptr);

    EXPECT_EQ(g_enter_calls, 0);
    EXPECT_EQ(g_exit_calls, 0);
    EXPECT_EQ(g_none_calls, 1);
}

TEST(tracing_callback_dispatcher_test,
     missing_callbacks_default_to_nullptr_and_are_skipped)
{
    reset_counters();
    using partial_t = tracing_callback_dispatcher<mock_sdk, record_enter>;

    auto enter_record  = callback_tracing_record_t{};
    enter_record.phase = mock_sdk::CALLBACK_PHASE_ENTER;
    partial_t::callback(enter_record, nullptr, nullptr);
    EXPECT_EQ(g_enter_calls, 1);

    auto exit_record  = callback_tracing_record_t{};
    exit_record.phase = mock_sdk::CALLBACK_PHASE_EXIT;
    partial_t::callback(exit_record, nullptr, nullptr);
    EXPECT_EQ(g_exit_calls, 0);

    auto none_record  = callback_tracing_record_t{};
    none_record.phase = mock_sdk::CALLBACK_PHASE_NONE;
    partial_t::callback(none_record, nullptr, nullptr);
    EXPECT_EQ(g_none_calls, 0);
}

}  // namespace rocprofsys::domains
