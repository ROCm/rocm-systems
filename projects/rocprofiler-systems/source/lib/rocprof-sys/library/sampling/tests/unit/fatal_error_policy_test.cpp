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

#include "doubles/in_memory_emitter.hpp"
#include "doubles/mock_overflow_trigger.hpp"
#include "doubles/test_sampling_policies.hpp"
#include "doubles/throwing_fatal_error_policy.hpp"
#include "sampling/sampling_service.hpp"
#include "sampling/src/sample_ring_buffer.hpp"

// test_service alias lives in test_sampling_policies.hpp.
using rocprofsys::sampling::test::test_service;

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

// ─── NFR-FM-2: overflow trigger configure failure → fatal_error_policy ───────

TEST(fatal_error_policy, overflow_trigger_configure_failure_calls_fatal_policy)
{
    // The overflow trigger policy must accept a FatalErrorPolicy& and call fatal()
    // on failure instead of just LOG_CRITICAL.
    mock_overflow_trigger       trigger;
    throwing_fatal_error_policy fatal;
    trigger.fail_next_configure = true;

    EXPECT_THROW(trigger.configure(0, 0, SIGRTMIN, nullptr, fatal), sampling_fatal_error)
        << "overflow trigger configure failure must route through fatal_error_policy";
}

// ─── NFR-FM-2: offload write failure → fatal_error_policy ───────────────────

TEST(fatal_error_policy, offload_write_failure_calls_fatal_policy)
{
    // The offload write() must accept a FatalErrorPolicy& and call fatal() on failure.
    in_memory_emitter           offload;
    throwing_fatal_error_policy fatal;
    offload.fail_next_write = true;

    rocprofsys::sampling::sample_ring_buffer<8> ring;
    EXPECT_THROW(offload.write(0, ring, fatal), sampling_fatal_error)
        << "offload write failure must route through fatal_error_policy";
}
