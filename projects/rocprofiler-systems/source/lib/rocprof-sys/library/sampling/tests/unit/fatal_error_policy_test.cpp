// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for fatal error policy — NFR-FM-1, NFR-FM-2.
// Three fatal sites: pthread_sigmask failure, offload file-state failures,
// perf_event_open failure.

#include <gtest/gtest.h>

#include "doubles/throwing_fatal_error_policy.hpp"

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;

// ─── NFR-FM-1/2: throwing_fatal_error_policy throws sampling_fatal_error ───────────

TEST(fatal_error_policy, throwing_policy_throws_sampling_fatal_error)
{
    throwing_fatal_error_policy policy;

    EXPECT_THROW(policy.fatal(__FILE__, __LINE__, "test fatal message"),
                 sampling_fatal_error)
        << "throwing_fatal_error_policy must throw sampling_fatal_error";
}

TEST(fatal_error_policy, sampling_fatal_error_carries_message)
{
    throwing_fatal_error_policy policy;

    try
    {
        policy.fatal(__FILE__, __LINE__, "expected error text");
        FAIL() << "Expected sampling_fatal_error to be thrown";
    } catch(sampling_fatal_error const& ex)
    {
        EXPECT_NE(std::string(ex.what()).find("expected error text"), std::string::npos)
            << "sampling_fatal_error must carry the formatted message";
    }
}

TEST(fatal_error_policy, sampling_fatal_error_carries_file_and_line)
{
    throwing_fatal_error_policy policy;

    try
    {
        policy.fatal("myfile.cpp", 42, "a message");
        FAIL() << "Expected sampling_fatal_error to be thrown";
    } catch(sampling_fatal_error const& ex)
    {
        EXPECT_STREQ(ex.file(), "myfile.cpp");
        EXPECT_EQ(ex.line(), 42);
    }
}

// ─── NFR-FM-2: sigmask failure path — tested via signal_mask_guard ──────────────
// The signal_mask_guard routes pthread_sigmask errors through fatal_error_policy.
// We test this with a recording_signal_dispatcher that returns EINVAL on demand.

#include "doubles/test_sampling_policies.hpp"
#include "sampling/sampling_service.hpp"

#ifdef sigmask
#    undef sigmask
#endif

using test_service = rocprofsys::sampling::sampling_service<test_sampling_policies>;

TEST(fatal_error_policy, sigmask_failure_calls_fatal_policy_through_service)
{
    test_service svc;

    // Populate signal_types_[0] so block_signals() does not return early on empty set.
    svc.setup(0);

    // Force the next sigmask call to fail.
    svc.signal_dispatcher_ref().set_fail_next();

    // block_signals() internally calls signal_dispatcher_.sigmask().
    // If the production code correctly wraps this in signal_mask_guard which calls
    // fatal_.fatal() on non-zero return, the throwing_fatal_error_policy will throw.
    EXPECT_THROW(svc.block_signals(), sampling_fatal_error)
        << "pthread_sigmask failure must route through fatal_error_policy";
}
