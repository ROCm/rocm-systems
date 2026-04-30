// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for sampling_duration_controller — AC-14.
// Uses fake_clock to avoid real sleeps.

#include <gtest/gtest.h>

#include "doubles/fake_clock.hpp"
#include "sampling/src/sampling_duration_controller.hpp"

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;

TEST(sampling_duration_controller, fires_shutdown_after_deadline)
{
    bool shutdown_fired = false;

    fake_clock::reset(0);

    sampling_duration_controller<fake_clock> ctrl([&] { shutdown_fired = true; });

    ASSERT_FALSE(shutdown_fired);

    ctrl.start(1.0);

    fake_clock::advance_ns(1'100'000'000U);
    ctrl.check_deadline();

    EXPECT_TRUE(shutdown_fired) << "sampling_duration_controller must fire the shutdown "
                                   "callback after the deadline";
}

TEST(sampling_duration_controller, spurious_wakeup_loops_without_firing_early)
{
    bool shutdown_fired = false;
    fake_clock::reset(0);

    sampling_duration_controller<fake_clock> ctrl([&] { shutdown_fired = true; });
    ctrl.start(1.0);

    fake_clock::advance_ns(500'000'000U);
    ctrl.check_deadline();

    EXPECT_FALSE(shutdown_fired)
        << "check_deadline() before deadline must NOT trigger shutdown";
}

TEST(sampling_duration_controller, stop_breaks_loop_without_firing)
{
    bool shutdown_fired = false;
    fake_clock::reset(0);

    sampling_duration_controller<fake_clock> ctrl([&] { shutdown_fired = true; });
    ctrl.start(1.0);

    ctrl.stop();

    EXPECT_FALSE(shutdown_fired)
        << "stop() must break the loop without firing the shutdown callback";
}
