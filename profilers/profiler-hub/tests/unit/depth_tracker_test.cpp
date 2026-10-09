// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "depth_tracker.hpp"

#include <gtest/gtest.h>

namespace
{

using profiler_hub::depth_tracker;

TEST(depth_tracker_test, a_new_tracker_has_depth_zero)
{
    const depth_tracker tracker;

    EXPECT_EQ(tracker.depth(), 0U);
}

TEST(depth_tracker_test, events_one_after_another_have_depth_one)
{
    depth_tracker tracker;

    EXPECT_EQ(tracker.add(0, 10), 1U);
    EXPECT_EQ(tracker.add(20, 30), 1U);
    EXPECT_EQ(tracker.add(40, 50), 1U);

    EXPECT_EQ(tracker.depth(), 1U);
}

TEST(depth_tracker_test, an_event_starting_when_another_ends_does_not_overlap_it)
{
    depth_tracker tracker;

    tracker.add(0, 10);
    EXPECT_EQ(tracker.add(10, 20), 1U);

    EXPECT_EQ(tracker.depth(), 1U);
}

TEST(depth_tracker_test, nested_events_add_up)
{
    depth_tracker tracker;

    EXPECT_EQ(tracker.add(0, 10), 1U);
    EXPECT_EQ(tracker.add(2, 5), 2U);
    EXPECT_EQ(tracker.add(3, 4), 3U);
    EXPECT_EQ(tracker.add(6, 8), 2U);

    EXPECT_EQ(tracker.depth(), 3U);
}

TEST(depth_tracker_test, events_that_overlap_without_nesting_count_as_concurrent)
{
    depth_tracker tracker;

    tracker.add(0, 5);
    EXPECT_EQ(tracker.add(3, 8), 2U);
    EXPECT_EQ(tracker.add(6, 9), 2U);

    EXPECT_EQ(tracker.depth(), 2U);
}

TEST(depth_tracker_test, events_starting_together_all_overlap)
{
    depth_tracker tracker;

    tracker.add(5, 10);
    tracker.add(5, 10);
    EXPECT_EQ(tracker.add(5, 7), 3U);

    EXPECT_EQ(tracker.depth(), 3U);
}

TEST(depth_tracker_test, an_event_without_duration_still_has_depth_one)
{
    depth_tracker tracker;

    EXPECT_EQ(tracker.add(5, 5), 1U);
    EXPECT_EQ(tracker.add(5, 5), 1U);

    EXPECT_EQ(tracker.depth(), 1U);
}

TEST(depth_tracker_test, the_depth_stays_at_its_maximum_after_events_end)
{
    depth_tracker tracker;

    tracker.add(0, 10);
    tracker.add(1, 9);
    tracker.add(100, 110);

    EXPECT_EQ(tracker.depth(), 2U);
}

TEST(depth_tracker_test, reset_starts_a_new_track)
{
    depth_tracker tracker;
    tracker.add(0, 10);
    tracker.add(1, 9);

    tracker.reset();

    EXPECT_EQ(tracker.depth(), 0U);
    EXPECT_EQ(tracker.add(2, 3), 1U);
    EXPECT_EQ(tracker.depth(), 1U);
}

}  // namespace
