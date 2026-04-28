// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for thread_sampler_state<Policies> and thread_sampler_state_registry<Policies,
// N>. AC-1 through AC-7: setup/shutdown per-thread state management. DEC-3 (optional
// triggers), DEC-4 (registry array), DEC-15 (in_handler flag).

#include <gtest/gtest.h>

#include "doubles/test_sampling_policies.hpp"
#include "sampling/src/thread_sampler_state.hpp"

#include <atomic>
#include <cstdint>
#include <optional>

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;

using test_state    = thread_sampler_state<test_sampling_policies>;
using test_registry = thread_sampler_state_registry<test_sampling_policies, 8>;

// ─── thread_sampler_state: construction ──────────────────────────────────────

TEST(thread_sampler_state, default_constructed_is_not_running)
{
    test_state state;
    EXPECT_FALSE(state.is_running()) << "thread_sampler_state must start not-running";
}

TEST(thread_sampler_state, default_constructed_has_no_timer_trigger)
{
    test_state state;
    EXPECT_FALSE(state.timer_trigger().has_value())
        << "timer_trigger must be absent before configure";
}

TEST(thread_sampler_state, default_constructed_has_no_overflow_trigger)
{
    test_state state;
    EXPECT_FALSE(state.overflow_trigger().has_value())
        << "overflow_trigger must be absent before configure";
}

TEST(thread_sampler_state, default_constructed_ring_buffer_is_empty)
{
    test_state state;
    EXPECT_EQ(state.ring_buffer().count(), 0U)
        << "ring_buffer must be empty on construction";
}

// ─── thread_sampler_state: in_handler flag (DEC-15) ──────────────────────────

TEST(thread_sampler_state, in_handler_flag_starts_clear)
{
    test_state state;
    // test_and_set on a cleared flag must return false (was clear)
    EXPECT_FALSE(state.try_enter_handler())
        << "in_handler flag must start clear — first try_enter_handler returns false";
}

TEST(thread_sampler_state, in_handler_flag_blocks_reentry)
{
    test_state            state;
    [[maybe_unused]] auto first = state.try_enter_handler();  // enters handler
    EXPECT_TRUE(state.try_enter_handler())
        << "second try_enter_handler must return true (already in handler)";
    state.exit_handler();
}

TEST(thread_sampler_state, exit_handler_clears_flag)
{
    test_state            state;
    [[maybe_unused]] auto entered = state.try_enter_handler();
    state.exit_handler();
    EXPECT_FALSE(state.try_enter_handler())
        << "after exit_handler, try_enter_handler must return false again";
    state.exit_handler();
}

// ─── thread_sampler_state: start / stop ──────────────────────────────────────

TEST(thread_sampler_state, start_sets_running)
{
    test_state state;
    state.start();
    EXPECT_TRUE(state.is_running()) << "start() must set running to true";
    state.stop();
}

TEST(thread_sampler_state, stop_clears_running)
{
    test_state state;
    state.start();
    state.stop();
    EXPECT_FALSE(state.is_running()) << "stop() must clear running flag";
}

// ─── thread_sampler_registry: construction ───────────────────────────────────

TEST(thread_sampler_state_registry, default_constructed_all_slots_null)
{
    test_registry reg;
    for(size_t i = 0; i < 8; ++i)
    {
        EXPECT_EQ(reg.at(static_cast<int64_t>(i)), nullptr)
            << "all registry slots must be null on construction";
    }
}

// ─── thread_sampler_state_registry: emplace / at ─────────────────────────────

TEST(thread_sampler_state_registry, emplace_creates_state_for_tid)
{
    test_registry reg;
    reg.emplace(0);
    EXPECT_NE(reg.at(0), nullptr) << "emplace(0) must create a non-null state";
}

TEST(thread_sampler_state_registry, emplace_is_idempotent_for_same_tid)
{
    test_registry reg;
    reg.emplace(1);
    auto* first = reg.at(1);
    reg.emplace(1);
    EXPECT_EQ(reg.at(1), first)
        << "second emplace for the same tid must not replace the state";
}

TEST(thread_sampler_state_registry, at_returns_null_for_unregistered_tid)
{
    test_registry reg;
    EXPECT_EQ(reg.at(5), nullptr)
        << "at() must return null for a tid that was never emplaced";
}

// ─── thread_sampler_state_registry: reset ─────────────────────────────────────

TEST(thread_sampler_state_registry, reset_clears_all_slots)
{
    test_registry reg;
    reg.emplace(0);
    reg.emplace(3);
    reg.reset();
    EXPECT_EQ(reg.at(0), nullptr) << "reset() must clear slot 0";
    EXPECT_EQ(reg.at(3), nullptr) << "reset() must clear slot 3";
}

// ─── thread_sampler_state_registry: out-of-range tid ─────────────────────────

TEST(thread_sampler_state_registry, at_returns_null_for_out_of_range_tid)
{
    test_registry reg;
    EXPECT_EQ(reg.at(100), nullptr)
        << "at() with tid >= MaxThreads must return null (no UB)";
}

TEST(thread_sampler_state_registry, emplace_is_noop_for_out_of_range_tid)
{
    test_registry reg;
    reg.emplace(100);  // must not crash or corrupt
    EXPECT_EQ(reg.at(0), nullptr) << "emplace() OOB must not corrupt slot 0";
}

TEST(thread_sampler_state_registry, emplace_is_noop_for_negative_tid)
{
    test_registry reg;
    reg.emplace(-1);  // must not crash or corrupt
    SUCCEED() << "emplace(-1) must not crash";
}

// ─── thread_sampler_state_registry: capacity() ────────────────────────────────

TEST(thread_sampler_state_registry, capacity_returns_max_threads)
{
    test_registry reg;
    EXPECT_EQ(reg.capacity(), 8U)
        << "capacity() must return the MaxThreads template value";
}

// ─── thread_sampler_state_registry: each() visits non-null slots ──────────────

TEST(thread_sampler_state_registry, each_visits_only_emplaced_slots)
{
    test_registry reg;
    reg.emplace(2);
    reg.emplace(5);

    std::vector<int64_t> visited;
    reg.each([&](int64_t tid, test_state&) { visited.push_back(tid); });

    ASSERT_EQ(visited.size(), 2U);
    EXPECT_EQ(visited.at(0), 2);
    EXPECT_EQ(visited.at(1), 5);
}

TEST(thread_sampler_state_registry, each_does_not_visit_empty_registry)
{
    test_registry reg;
    int           count = 0;
    reg.each([&](int64_t, test_state&) { ++count; });
    EXPECT_EQ(count, 0) << "each() on empty registry must visit zero slots";
}

// ─── thread_sampler_state: dropped_count ─────────────────────────────────────

TEST(thread_sampler_state, dropped_count_starts_at_zero)
{
    test_state state;
    EXPECT_EQ(state.dropped_count(), 0U) << "dropped_count must start at 0";
}

TEST(thread_sampler_state, increment_dropped_increases_count)
{
    test_state state;
    state.increment_dropped();
    state.increment_dropped();
    EXPECT_EQ(state.dropped_count(), 2U)
        << "dropped_count must equal number of increments";
}

// ─── thread_sampler_state: timer_trigger emplace-in-optional ─────────────────

TEST(thread_sampler_state, timer_trigger_can_be_emplaced)
{
    test_state state;
    state.timer_trigger().emplace();
    EXPECT_TRUE(state.timer_trigger().has_value())
        << "timer_trigger must be present after emplace";
}

TEST(thread_sampler_state, overflow_trigger_can_be_emplaced)
{
    test_state state;
    state.overflow_trigger().emplace();
    EXPECT_TRUE(state.overflow_trigger().has_value())
        << "overflow_trigger must be present after emplace";
}

// ─── thread_sampler_state: signal_types ──────────────────────────────────────

TEST(thread_sampler_state, signal_types_empty_on_construction)
{
    test_state state;
    EXPECT_TRUE(state.signal_types().empty())
        << "signal_types must be empty before set_signal_types";
}

TEST(thread_sampler_state, set_signal_types_stores_set)
{
    test_state state;
    state.set_signal_types({ 34, 40 });
    EXPECT_EQ(state.signal_types().count(34), 1U);
    EXPECT_EQ(state.signal_types().count(40), 1U);
}

// ─── thread_sampler_state: in_flight_count ───────────────────────────────────

TEST(thread_sampler_state, in_flight_count_starts_at_zero)
{
    test_state state;
    EXPECT_EQ(state.in_flight_count(), 0) << "in_flight_count must start at 0";
}

TEST(thread_sampler_state, enter_exit_in_flight_balances)
{
    test_state state;
    state.enter_in_flight();
    EXPECT_EQ(state.in_flight_count(), 1);
    state.exit_in_flight();
    EXPECT_EQ(state.in_flight_count(), 0);
}

TEST(thread_sampler_state, wait_for_in_flight_zero_returns_true_when_already_zero)
{
    test_state state;
    EXPECT_TRUE(state.wait_for_in_flight_zero(10))
        << "wait_for_in_flight_zero must return true immediately when count is 0";
}

// ─── thread_sampler_state_registry: erase ────────────────────────────────────

TEST(thread_sampler_state_registry, erase_removes_slot)
{
    test_registry reg;
    reg.emplace(2);
    ASSERT_NE(reg.at(2), nullptr);
    reg.erase(2);
    EXPECT_EQ(reg.at(2), nullptr) << "erase(2) must remove the state for tid 2";
}

TEST(thread_sampler_state_registry, erase_oob_is_noop)
{
    test_registry reg;
    reg.erase(100);  // must not crash
    SUCCEED() << "erase() OOB must not crash";
}

// ─── thread_sampler_state_registry: const overloads (I-11) ───────────────────

TEST(thread_sampler_state_registry, at_const_returns_const_ptr_to_existing_state)
{
    test_registry reg;
    reg.emplace(0);

    const test_registry& creg = reg;
    test_state const*    ptr  = creg.at(0);

    ASSERT_NE(ptr, nullptr) << "const at(0) must return non-null after emplace(0)";
    EXPECT_FALSE(ptr->is_running()) << "const at() must give readable state";
}

TEST(thread_sampler_state_registry, each_const_iterates_all_active_tids)
{
    test_registry reg;
    reg.emplace(1);
    reg.emplace(4);
    reg.emplace(7);

    const test_registry& creg = reg;

    std::vector<int64_t> visited;
    creg.each([&](int64_t tid, test_state const&) { visited.push_back(tid); });

    ASSERT_EQ(visited.size(), 3U) << "const each() must visit exactly 3 emplaced tids";
    EXPECT_EQ(visited.at(0), 1);
    EXPECT_EQ(visited.at(1), 4);
    EXPECT_EQ(visited.at(2), 7);
}
