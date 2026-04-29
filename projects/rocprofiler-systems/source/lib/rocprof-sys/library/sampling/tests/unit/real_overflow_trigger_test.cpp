// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Smoke tests for real_overflow_trigger — the Linux perf_event_open-backed
// OverflowTriggerPolicy production implementation.
//
// Only the default-constructed state and stop()-without-configure path can be
// tested without a real perf event fd: configure() calls perf_event_open(2)
// which requires CAP_PERFMON or perf_event_paranoid <= 2.  Full signal-delivery
// coverage lives in the stress test (signal_handler_stress_test.cpp).
//
// rocprofsys_sampling_signal_handler is provided by librocprofiler-systems-static
// (services_accessor.cpp).  No stub needed here.

#include <gtest/gtest.h>

#include "sampling/policies/real_overflow_trigger.hpp"

using namespace rocprofsys::sampling;

// ── Default-constructed state ─────────────────────────────────────────────────

TEST(real_overflow_trigger_smoke, default_constructed_is_not_open)
{
    real_overflow_trigger trigger;
    EXPECT_FALSE(trigger.is_open())
        << "real_overflow_trigger must not report is_open() before configure()";
}

// ── stop() without configure() is a no-op ────────────────────────────────────

TEST(real_overflow_trigger_smoke, stop_without_configure_does_not_crash)
{
    real_overflow_trigger trigger;
    EXPECT_NO_FATAL_FAILURE(trigger.stop())
        << "stop() on an unconfigured real_overflow_trigger must be a safe no-op";
    EXPECT_FALSE(trigger.is_open()) << "is_open() must remain false after stop() no-op";
}

// ── start() without configure() is a no-op ───────────────────────────────────

TEST(real_overflow_trigger_smoke, start_without_configure_does_not_crash)
{
    real_overflow_trigger trigger;
    EXPECT_NO_FATAL_FAILURE(trigger.start())
        << "start() on an unconfigured real_overflow_trigger must be a safe no-op";
    EXPECT_FALSE(trigger.is_open());
}

// ── Destructor on default-constructed does not crash ─────────────────────────

TEST(real_overflow_trigger_smoke, destructor_on_default_constructed_does_not_crash)
{
    EXPECT_NO_FATAL_FAILURE({
        real_overflow_trigger trigger;
        (void) trigger;
    }) << "destructor of default-constructed real_overflow_trigger must not crash";
}

// ── Type properties ───────────────────────────────────────────────────────────

static_assert(!std::is_copy_constructible_v<real_overflow_trigger>,
              "real_overflow_trigger must be non-copyable (owns perf fd)");
static_assert(!std::is_copy_assignable_v<real_overflow_trigger>,
              "real_overflow_trigger must be non-copy-assignable");
static_assert(std::is_default_constructible_v<real_overflow_trigger>,
              "real_overflow_trigger must be default-constructible");

// OverflowTriggerPolicy contract: is_open() must be noexcept.
static_assert(noexcept(std::declval<real_overflow_trigger>().is_open()),
              "is_open() must be noexcept per OverflowTriggerPolicy");
static_assert(noexcept(std::declval<real_overflow_trigger>().stop()),
              "stop() must be noexcept per OverflowTriggerPolicy");
static_assert(noexcept(std::declval<real_overflow_trigger>().start()),
              "start() must be noexcept per OverflowTriggerPolicy");
