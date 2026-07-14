// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/trace_suppression.hpp"

#include <gtest/gtest.h>
#include <thread>

using rocprofsys::trace_suppression;

TEST(TraceSuppressionTest, InactiveByDefault)
{
    EXPECT_FALSE(trace_suppression::is_active());
}

TEST(TraceSuppressionTest, EnterExitTogglesActiveState)
{
    trace_suppression::enter();
    EXPECT_TRUE(trace_suppression::is_active());

    trace_suppression::exit();
    EXPECT_FALSE(trace_suppression::is_active());
}

TEST(TraceSuppressionTest, NestedEnterExitTracksDepth)
{
    trace_suppression::enter();
    trace_suppression::enter();
    EXPECT_TRUE(trace_suppression::is_active());

    trace_suppression::exit();
    EXPECT_TRUE(trace_suppression::is_active());  // one level still active

    trace_suppression::exit();
    EXPECT_FALSE(trace_suppression::is_active());
}

TEST(TraceSuppressionTest, StateIsPerThread)
{
    EXPECT_FALSE(trace_suppression::is_active());

    std::thread other([]() {
        EXPECT_FALSE(trace_suppression::is_active());
        trace_suppression::enter();
        EXPECT_TRUE(trace_suppression::is_active());
    });
    other.join();

    EXPECT_FALSE(trace_suppression::is_active());
}
