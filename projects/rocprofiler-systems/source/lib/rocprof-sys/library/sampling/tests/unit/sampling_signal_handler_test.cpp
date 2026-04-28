// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for the sampling signal handler invariants.
//
// The production handler (rocprofsys_sampling_signal_handler) is defined in
// services_accessor.cpp (production-only TU) and is not callable from this
// binary. These tests exercise the handler's invariants through the policy
// seams it uses:
//
//   - thread_sampler_state::try_enter_handler() / exit_handler() (re-entry guard)
//   - sample_ring_buffer::try_push() + dropped_count() (ring-full drop path)
//   - thread_sampler_state::increment_dropped() / dropped_count() (state counter)
//   - thread_sampler_state::is_running() (null-state guard path)
//
// Tests #2-#4 from the team-lead spec map to tests below.
// Direct invocation of rocprofsys_sampling_signal_handler requires the full
// production link (services_accessor.cpp + main library).
//
// Criteria: DEC-15 (re-entry guard), DEC-4 (in-flight count), AC-21 (drop, not crash).

#include <gtest/gtest.h>

#include "doubles/test_sampling_policies.hpp"
#include "sampling/data/backtrace_record.hpp"
#include "sampling/src/sample_ring_buffer.hpp"
#include "sampling/src/thread_sampler_state.hpp"

#include <cstdint>
#include <vector>

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;

// Use the small-registry test policy (MaxThreads=8 from test_sampling_policies).
using state_t = thread_sampler_state<test_sampling_policies>;

// ── Helpers ───────────────────────────────────────────────────────────────────

static backtrace_record
make_record(int64_t tid = 0, uint64_t ts_ns = 1000)
{
    backtrace_record r{};
    r.tid          = tid;
    r.timestamp_ns = ts_ns;
    r.trigger      = trigger_type::TIMER;
    r.pc_count     = 2;
    r.raw_pcs[0]   = 0x1000;
    r.raw_pcs[1]   = 0x2000;
    return r;
}

// Fill a ring buffer to capacity so the next try_push returns false.
template <size_t N>
static void
fill_ring(sample_ring_buffer<N>& ring)
{
    for(size_t i = 0; i < N; ++i)
    {
        backtrace_record r = make_record(0, static_cast<uint64_t>(i + 1));
        ring.try_push(r);
    }
}

// ── Re-entry guard (DEC-15) ───────────────────────────────────────────────────

TEST(sampling_signal_handler, reentry_guard_first_call_returns_false)
{
    // try_enter_handler() returns FALSE when not yet entered (safe to proceed).
    // The handler only drops when it returns TRUE (already entered).
    state_t state;
    bool    already_entered = state.try_enter_handler();
    EXPECT_FALSE(already_entered) << "try_enter_handler() must return false on first "
                                     "call (handler is safe to proceed)";
    state.exit_handler();  // cleanup
}

TEST(sampling_signal_handler, reentry_guard_second_call_returns_true)
{
    // After try_enter_handler() returns false (handler entered), a second call
    // must return true (re-entry detected — drop the sample).
    state_t state;
    bool    first  = state.try_enter_handler();
    bool    second = state.try_enter_handler();

    EXPECT_FALSE(first) << "First try_enter_handler() must return false";
    EXPECT_TRUE(second)
        << "Second try_enter_handler() must return true (re-entry detected)";

    state.exit_handler();
}

TEST(sampling_signal_handler, reentry_guard_cleared_after_exit)
{
    // After exit_handler(), the guard must be clear so a subsequent call is allowed.
    state_t state;
    (void) state.try_enter_handler();  // enter
    state.exit_handler();              // exit

    bool after_exit = state.try_enter_handler();
    EXPECT_FALSE(after_exit) << "After exit_handler(), try_enter_handler() must return "
                                "false again (guard cleared)";
    state.exit_handler();
}

// ── Ring-full drop path (AC-21, DEC-15) ──────────────────────────────────────

TEST(sampling_signal_handler, ring_buffer_returns_false_when_full)
{
    // Precondition: ring at capacity → try_push returns false (sample dropped).
    // This is the ring-full check the signal handler uses before calling
    // increment_dropped().
    constexpr size_t            SMALL_N = 4;
    sample_ring_buffer<SMALL_N> ring;

    fill_ring(ring);
    ASSERT_EQ(ring.count(), SMALL_N) << "ring must be at capacity after fill";

    bool pushed = ring.try_push(make_record());
    EXPECT_FALSE(pushed)
        << "try_push() must return false when ring is at capacity (sample is dropped)";
}

TEST(sampling_signal_handler, ring_buffer_dropped_count_increments_on_full)
{
    // The ring buffer itself tracks drops internally via its own dropped_count_.
    // Handler code calls state->increment_dropped() separately — but the ring's
    // own counter also increments, giving an independent audit trail.
    constexpr size_t            SMALL_N = 4;
    sample_ring_buffer<SMALL_N> ring;

    fill_ring(ring);

    size_t before = ring.dropped_count();
    ring.try_push(make_record());
    size_t after = ring.dropped_count();

    EXPECT_EQ(after, before + 1)
        << "ring.dropped_count() must increment by 1 when try_push fails on full ring";
}

TEST(sampling_signal_handler, ring_buffer_capacity_unchanged_after_full_push)
{
    // Existing records must NOT be overwritten when the ring is full.
    // The handler must preserve oldest samples — not evict them.
    constexpr size_t            SMALL_N = 4;
    sample_ring_buffer<SMALL_N> ring;

    fill_ring(ring);
    size_t count_before = ring.count();
    ring.try_push(make_record(0, 999999U));
    size_t count_after = ring.count();

    EXPECT_EQ(count_before, count_after) << "ring capacity must not change after a "
                                            "dropped push (oldest records preserved)";
}

// ── Per-state dropped counter (DEC-15) ────────────────────────────────────────

TEST(sampling_signal_handler, state_dropped_count_starts_at_zero)
{
    state_t state;
    EXPECT_EQ(state.dropped_count(), 0U)
        << "dropped_count() must be 0 for a freshly constructed state";
}

TEST(sampling_signal_handler, state_increment_dropped_increments_counter)
{
    state_t state;
    state.increment_dropped();
    EXPECT_EQ(state.dropped_count(), 1U)
        << "increment_dropped() must increment dropped_count() by 1";
    state.increment_dropped();
    EXPECT_EQ(state.dropped_count(), 2U)
        << "two increment_dropped() calls must produce dropped_count() == 2";
}

// ── is_running guard (null-state early return path) ────────────────────────────

TEST(sampling_signal_handler, state_not_running_after_construction)
{
    // The handler checks is_running() before proceeding.
    // A state that has not been started must report not-running, so the handler
    // returns early (no sample pushed, no dropped increment).
    state_t state;
    EXPECT_FALSE(state.is_running())
        << "is_running() must be false for a freshly constructed state "
        << "(handler should return early, matching the null-state early-return path)";
}

TEST(sampling_signal_handler, state_is_running_after_start)
{
    state_t state;
    state.start();
    EXPECT_TRUE(state.is_running()) << "is_running() must be true after start()";
}

TEST(sampling_signal_handler, state_not_running_after_stop)
{
    state_t state;
    state.start();
    state.stop();
    EXPECT_FALSE(state.is_running()) << "is_running() must be false after stop()";
}

// ── In-flight count (DEC-4) ───────────────────────────────────────────────────

TEST(sampling_signal_handler, in_flight_count_starts_at_zero)
{
    state_t state;
    EXPECT_EQ(state.in_flight_count(), 0)
        << "in_flight_count() must be 0 before any handler invocation";
}

TEST(sampling_signal_handler, enter_in_flight_increments_count)
{
    state_t state;
    state.enter_in_flight();
    EXPECT_EQ(state.in_flight_count(), 1)
        << "enter_in_flight() must increment in_flight_count() to 1";
}

TEST(sampling_signal_handler, exit_in_flight_decrements_count)
{
    state_t state;
    state.enter_in_flight();
    state.exit_in_flight();
    EXPECT_EQ(state.in_flight_count(), 0)
        << "exit_in_flight() must decrement in_flight_count() back to 0";
}

// ── Full handler invariant sequence ───────────────────────────────────────────
// Simulate the exact sequence the production handler executes:
//   1. check is_running() → proceed
//   2. try_enter_handler() → false (not re-entrant) → proceed
//   3. enter_in_flight()
//   4. try_push(rec) → succeeds
//   5. exit_in_flight()
//   6. exit_handler()
// Then verify ring buffer has the expected record.

TEST(sampling_signal_handler, simulated_handler_sequence_pushes_record)
{
    state_t state;
    state.start();

    bool blocked = false;  // service.is_blocked() — false in unit test context

    // Simulate handler entry.
    ASSERT_TRUE(state.is_running());
    ASSERT_FALSE(blocked);

    bool re_entered = state.try_enter_handler();
    ASSERT_FALSE(re_entered) << "handler must not be re-entered in this sequence";

    state.enter_in_flight();

    backtrace_record rec    = make_record(0, 42000U);
    bool             pushed = state.ring_buffer().try_push(rec);
    EXPECT_TRUE(pushed) << "try_push must succeed on an empty ring";

    state.exit_in_flight();
    state.exit_handler();

    // Verify the record landed.
    EXPECT_EQ(state.ring_buffer().count(), 1U)
        << "ring must contain 1 record after the simulated handler sequence";
    auto popped = state.ring_buffer().pop();
    ASSERT_TRUE(popped.has_value());
    EXPECT_EQ(popped->timestamp_ns, 42000U);
}

TEST(sampling_signal_handler, simulated_reentry_drops_sample_and_increments_counter)
{
    // Simulate a re-entrant call: handler is already entered (try_enter_handler == true).
    // Handler must: increment_dropped() and return without pushing.
    state_t state;
    state.start();

    // Simulate handler already entered.
    bool first = state.try_enter_handler();
    ASSERT_FALSE(first) << "first entry must succeed";

    // Now simulate a re-entrant signal delivery.
    bool re_entered = state.try_enter_handler();
    ASSERT_TRUE(re_entered) << "second try_enter_handler must detect re-entry";

    // Handler body: increment dropped and return early.
    state.increment_dropped();

    // Verify: ring must be empty (handler returned without pushing).
    EXPECT_EQ(state.ring_buffer().count(), 0U)
        << "ring must be empty — re-entrant handler must not push a record";
    EXPECT_EQ(state.dropped_count(), 1U)
        << "dropped_count() must be 1 after one re-entrant drop";

    state.exit_handler();
}

TEST(sampling_signal_handler, simulated_ring_full_increments_dropped_on_state)
{
    // Simulate handler execution when ring is full:
    //   try_push fails → handler calls state.increment_dropped().
    constexpr size_t SMALL_N = 4;

    // We cannot easily test with thread_sampler_state's internal ring (capacity=2048)
    // being full, so use sample_ring_buffer<4> directly to exercise the same logic.
    sample_ring_buffer<SMALL_N> ring;
    fill_ring(ring);

    state_t state;
    state.start();
    (void) state.try_enter_handler();
    state.enter_in_flight();

    backtrace_record rec    = make_record(0, 99000U);
    bool             pushed = ring.try_push(rec);
    if(!pushed)
    {
        // Mirrors: state->increment_dropped() in handler.
        state.increment_dropped();
    }

    state.exit_in_flight();
    state.exit_handler();

    EXPECT_FALSE(pushed) << "try_push must fail on a full ring";
    EXPECT_EQ(state.dropped_count(), 1U)
        << "state.dropped_count() must be 1 after one ring-full drop";
    EXPECT_EQ(ring.count(), SMALL_N)
        << "ring contents must be unchanged after a failed push";
}
