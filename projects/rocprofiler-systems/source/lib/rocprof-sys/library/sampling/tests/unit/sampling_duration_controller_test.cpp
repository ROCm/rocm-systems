// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for sampling_duration_controller — AC-14.
// Uses fake_clock to avoid real sleeps.

#include <gtest/gtest.h>

#include "doubles/fake_clock.hpp"
#include "sampling/src/sampling_duration_controller.hpp"

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;

// ─── AC-14: fires shutdown callback after deadline ────────────────────────────

TEST(sampling_duration_controller, fires_shutdown_after_deadline)
{
    bool shutdown_fired = false;
    auto on_shutdown    = [&] { shutdown_fired = true; };

    fake_clock::reset(0);

    // Controller with a 1-second duration (1e9 ns)
    sampling_duration_controller<fake_clock> ctrl(on_shutdown);

    ASSERT_FALSE(shutdown_fired);

    // Start the controller — it should NOT fire yet (time hasn't advanced).
    ctrl.start(1.0);  // 1.0 second duration

    // Advance time past the deadline and tick the controller.
    fake_clock::advance_ns(1'100'000'000U);  // 1.1 seconds
    ctrl.tick_for_test();                    // poke the controller to check deadline

    EXPECT_TRUE(shutdown_fired) << "sampling_duration_controller must fire the shutdown "
                                   "callback after the deadline";
}

// ─── AC-14: spurious wakeup loops without firing early ────────────────────────

TEST(sampling_duration_controller, spurious_wakeup_loops_without_firing_early)
{
    bool shutdown_fired = false;
    fake_clock::reset(0);

    sampling_duration_controller<fake_clock> ctrl([&] { shutdown_fired = true; });
    ctrl.start(1.0);

    // Advance time to only 0.5 seconds — deadline not yet reached.
    fake_clock::advance_ns(500'000'000U);
    ctrl.tick_for_test();

    EXPECT_FALSE(shutdown_fired)
        << "Spurious wakeup must NOT trigger shutdown before deadline";
}

// ─── AC-14: already finalized state breaks the loop ──────────────────────────

TEST(sampling_duration_controller, finalized_state_breaks_loop)
{
    bool shutdown_fired = false;
    fake_clock::reset(0);

    sampling_duration_controller<fake_clock> ctrl([&] { shutdown_fired = true; });
    ctrl.start(1.0);

    // Mark process as finalizing — controller should stop waiting.
    ctrl.set_finalized_for_test(true);
    ctrl.tick_for_test();

    EXPECT_FALSE(shutdown_fired)
        << "Finalized state must break the loop without firing the shutdown callback";
}
