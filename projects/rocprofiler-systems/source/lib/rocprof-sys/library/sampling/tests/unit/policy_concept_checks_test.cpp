// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for C++17 detection-idiom policy concept checks (Task #15).
//
// Verifies that:
//   - is_*_policy<GoodType>::value == true for conforming test doubles
//   - is_*_policy<BadType>::value == false for non-conforming types
//
// The static_assert enforcement itself is exercised by the sampling_service
// constructor (sampling_service_impl.hpp); those fire at compile time and are
// not runtime-testable.  The trait predicates are tested here as runtime
// boolean assertions.

#include <gtest/gtest.h>

// Policy check traits
#include "sampling/policies/policy_checks.hpp"

// Conforming test doubles
#include "doubles/fake_clock.hpp"
#include "doubles/in_memory_emitter.hpp"
#include "doubles/mock_overflow_trigger.hpp"
#include "doubles/mock_timer_trigger.hpp"
#include "doubles/mock_unwinder.hpp"
#include "doubles/noop_perfetto_sink.hpp"
#include "doubles/recording_signal_dispatcher.hpp"
#include "doubles/recording_trace_sink.hpp"
#include "doubles/throwing_fatal_error_policy.hpp"

using namespace rocprofsys::sampling::detail;

// ── Conforming types: trait must be true ──────────────────────────────────────

TEST(policy_concept_checks, clock_policy_satisfied_by_fake_clock)
{
    static_assert(is_clock_policy_v<rocprofsys::sampling::test::fake_clock>,
                  "fake_clock must satisfy ClockPolicy");
    EXPECT_TRUE((is_clock_policy_v<rocprofsys::sampling::test::fake_clock>) );
}

TEST(policy_concept_checks, timer_trigger_policy_satisfied_by_mock_timer_trigger)
{
    static_assert(
        is_timer_trigger_policy_v<rocprofsys::sampling::test::mock_timer_trigger>,
        "mock_timer_trigger must satisfy TimerTriggerPolicy");
    EXPECT_TRUE(
        (is_timer_trigger_policy_v<rocprofsys::sampling::test::mock_timer_trigger>) );
}

TEST(policy_concept_checks, overflow_trigger_policy_satisfied_by_mock_overflow_trigger)
{
    static_assert(
        is_overflow_trigger_policy_v<rocprofsys::sampling::test::mock_overflow_trigger>,
        "mock_overflow_trigger must satisfy OverflowTriggerPolicy");
    EXPECT_TRUE((is_overflow_trigger_policy_v<
                 rocprofsys::sampling::test::mock_overflow_trigger>) );
}

TEST(policy_concept_checks,
     signal_dispatcher_policy_satisfied_by_recording_signal_dispatcher)
{
    static_assert(is_signal_dispatcher_policy_v<
                      rocprofsys::sampling::test::recording_signal_dispatcher>,
                  "recording_signal_dispatcher must satisfy SignalDispatcherPolicy");
    EXPECT_TRUE((is_signal_dispatcher_policy_v<
                 rocprofsys::sampling::test::recording_signal_dispatcher>) );
}

TEST(policy_concept_checks, unwinder_policy_satisfied_by_mock_unwinder)
{
    static_assert(is_unwinder_policy_v<rocprofsys::sampling::test::mock_unwinder>,
                  "mock_unwinder must satisfy UnwinderPolicy");
    EXPECT_TRUE((is_unwinder_policy_v<rocprofsys::sampling::test::mock_unwinder>) );
}

TEST(policy_concept_checks, emitter_policy_satisfied_by_in_memory_emitter)
{
    static_assert(is_emitter_policy_v<rocprofsys::sampling::test::in_memory_emitter>,
                  "in_memory_emitter must satisfy EmitterPolicy");
    EXPECT_TRUE((is_emitter_policy_v<rocprofsys::sampling::test::in_memory_emitter>) );
}

TEST(policy_concept_checks, trace_sink_policy_satisfied_by_recording_trace_sink)
{
    static_assert(
        is_trace_sink_policy_v<rocprofsys::sampling::test::recording_trace_sink>,
        "recording_trace_sink must satisfy TraceSinkPolicy");
    EXPECT_TRUE(
        (is_trace_sink_policy_v<rocprofsys::sampling::test::recording_trace_sink>) );
}

TEST(policy_concept_checks, perfetto_sink_policy_satisfied_by_noop_perfetto_sink)
{
    static_assert(
        is_perfetto_sink_policy_v<rocprofsys::sampling::test::noop_perfetto_sink>,
        "noop_perfetto_sink must satisfy PerfettoSinkPolicy");
    EXPECT_TRUE(
        (is_perfetto_sink_policy_v<rocprofsys::sampling::test::noop_perfetto_sink>) );
}

TEST(policy_concept_checks, fatal_error_policy_satisfied_by_throwing_policy)
{
    static_assert(
        is_fatal_error_policy_v<rocprofsys::sampling::test::throwing_fatal_error_policy>,
        "throwing_fatal_error_policy must satisfy FatalErrorPolicy");
    EXPECT_TRUE((is_fatal_error_policy_v<
                 rocprofsys::sampling::test::throwing_fatal_error_policy>) );
}

// ── Non-conforming types: trait must be false ─────────────────────────────────

namespace
{

struct bad_clock
{
    // missing now_ns() and now_steady()
};

struct bad_timer_trigger
{
    // missing start(), stop(), is_armed()
};

struct bad_overflow_trigger
{
    // missing start(), stop(), is_open()
};

struct bad_signal_dispatcher
{
    // missing sigmask(int, void const*, void*)
};

struct bad_unwinder
{
    // missing unwind() and valid_pc()
};

struct bad_emitter
{
    // missing read(), tids(), reset(), erase()
};

struct bad_trace_sink
{
    // missing store_timer() and store_overflow()
};

struct bad_perfetto_sink
{
    // missing emit_timer() and emit_overflow()
};

}  // namespace

TEST(policy_concept_checks, clock_policy_not_satisfied_by_bad_type)
{
    static_assert(!is_clock_policy_v<bad_clock>,
                  "bad_clock must NOT satisfy ClockPolicy");
    EXPECT_FALSE((is_clock_policy_v<bad_clock>) );
}

TEST(policy_concept_checks, timer_trigger_policy_not_satisfied_by_bad_type)
{
    static_assert(!is_timer_trigger_policy_v<bad_timer_trigger>,
                  "bad_timer_trigger must NOT satisfy TimerTriggerPolicy");
    EXPECT_FALSE((is_timer_trigger_policy_v<bad_timer_trigger>) );
}

TEST(policy_concept_checks, overflow_trigger_policy_not_satisfied_by_bad_type)
{
    static_assert(!is_overflow_trigger_policy_v<bad_overflow_trigger>,
                  "bad_overflow_trigger must NOT satisfy OverflowTriggerPolicy");
    EXPECT_FALSE((is_overflow_trigger_policy_v<bad_overflow_trigger>) );
}

TEST(policy_concept_checks, signal_dispatcher_policy_not_satisfied_by_bad_type)
{
    static_assert(!is_signal_dispatcher_policy_v<bad_signal_dispatcher>,
                  "bad_signal_dispatcher must NOT satisfy SignalDispatcherPolicy");
    EXPECT_FALSE((is_signal_dispatcher_policy_v<bad_signal_dispatcher>) );
}

TEST(policy_concept_checks, unwinder_policy_not_satisfied_by_bad_type)
{
    static_assert(!is_unwinder_policy_v<bad_unwinder>,
                  "bad_unwinder must NOT satisfy UnwinderPolicy");
    EXPECT_FALSE((is_unwinder_policy_v<bad_unwinder>) );
}

TEST(policy_concept_checks, emitter_policy_not_satisfied_by_bad_type)
{
    static_assert(!is_emitter_policy_v<bad_emitter>,
                  "bad_emitter must NOT satisfy EmitterPolicy");
    EXPECT_FALSE((is_emitter_policy_v<bad_emitter>) );
}

TEST(policy_concept_checks, trace_sink_policy_not_satisfied_by_bad_type)
{
    static_assert(!is_trace_sink_policy_v<bad_trace_sink>,
                  "bad_trace_sink must NOT satisfy TraceSinkPolicy");
    EXPECT_FALSE((is_trace_sink_policy_v<bad_trace_sink>) );
}

TEST(policy_concept_checks, perfetto_sink_policy_not_satisfied_by_bad_type)
{
    static_assert(!is_perfetto_sink_policy_v<bad_perfetto_sink>,
                  "bad_perfetto_sink must NOT satisfy PerfettoSinkPolicy");
    EXPECT_FALSE((is_perfetto_sink_policy_v<bad_perfetto_sink>) );
}
