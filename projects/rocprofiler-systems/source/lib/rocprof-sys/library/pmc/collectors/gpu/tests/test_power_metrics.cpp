// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//
// Unit tests for the shared socket-power selection / labeling helpers declared in
// library/pmc/collectors/gpu/types.hpp.
//
// These helpers are the single source of truth used by the live Perfetto path
// (perfetto_policy.hpp) and the replay paths (perfetto_processor.cpp /
// rocpd_processor.cpp), so exercising them here covers all output formats.
//

#include "library/pmc/collectors/gpu/types.hpp"
#include <cstdint>

#include <gtest/gtest.h>

using namespace rocprofsys::pmc::collectors::gpu;

namespace
{

enabled_metrics
make_enabled(bool current, bool average)
{
    enabled_metrics em{};
    em.bits.current_socket_power = current ? 1u : 0u;
    em.bits.average_socket_power = average ? 1u : 0u;
    return em;
}

metrics
make_power_metrics(std::uint32_t current, std::uint32_t average)
{
    metrics m{};
    m.current_socket_power = current;
    m.average_socket_power = average;
    return m;
}

}  // namespace

// Current available (with or without average) -> current value is used.
TEST(socket_power, PrefersCurrentWhenAvailable)
{
    const auto m = make_power_metrics(150, 140);

    EXPECT_DOUBLE_EQ(select_socket_power(make_enabled(true, false), m), 150.0);
    EXPECT_DOUBLE_EQ(select_socket_power(make_enabled(true, true), m), 150.0);
}

// Only average available -> falls back to the averaged reading.
TEST(socket_power, FallsBackToAverageWhenCurrentUnavailable)
{
    const auto m = make_power_metrics(150, 140);

    EXPECT_DOUBLE_EQ(select_socket_power(make_enabled(false, true), m), 140.0);
}

TEST(socket_power, HasCurrentReflectsBit)
{
    EXPECT_TRUE(has_current_socket_power(make_enabled(true, false)));
    EXPECT_TRUE(has_current_socket_power(make_enabled(true, true)));
    EXPECT_FALSE(has_current_socket_power(make_enabled(false, true)));
    EXPECT_FALSE(has_current_socket_power(make_enabled(false, false)));
}

// Track label matches the reading select_socket_power() emits.
TEST(socket_power, TrackLabelMatchesSelectedReading)
{
    EXPECT_STREQ(socket_power_track_label(make_enabled(true, false)), "Current Power");
    EXPECT_STREQ(socket_power_track_label(make_enabled(true, true)), "Current Power");
    EXPECT_STREQ(socket_power_track_label(make_enabled(false, true)), "Avg. Power");
}
