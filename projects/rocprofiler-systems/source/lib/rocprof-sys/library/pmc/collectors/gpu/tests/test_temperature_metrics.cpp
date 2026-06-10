// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//
// Unit tests for GPU die temperature selection / labeling helpers in
// library/pmc/collectors/gpu/types.hpp (has_gpu_temperature_output, select_gpu_temperature,
// gpu_temperature_track_label).
//
// Same helpers are used by perfetto_policy.hpp, perfetto_processor.cpp, and
// rocpd_processor.cpp.
//

#include "library/pmc/collectors/gpu/types.hpp"

#include <cstdint>

#include <gtest/gtest.h>

using namespace rocprofsys::pmc::collectors::gpu;

namespace
{

enabled_metrics
make_temp_enabled(bool hotspot, bool edge)
{
    enabled_metrics em{};
    em.bits.hotspot_temperature = hotspot ? 1u : 0u;
    em.bits.edge_temperature    = edge ? 1u : 0u;
    return em;
}

metrics
make_temp_metrics(std::uint32_t hotspot_c, std::uint32_t edge_c)
{
    metrics m{};
    m.hotspot_temperature = hotspot_c;
    m.edge_temperature    = edge_c;
    return m;
}

}  // namespace

// Hotspot enabled (with or without edge) -> hotspot value is used.
TEST(gpu_temperature, PrefersHotspotWhenAvailable)
{
    const auto m = make_temp_metrics(72, 55);

    EXPECT_DOUBLE_EQ(select_gpu_temperature(make_temp_enabled(true, false), m), 72.0);
    EXPECT_DOUBLE_EQ(select_gpu_temperature(make_temp_enabled(true, true), m), 72.0);
}

// Only edge enabled -> edge reading is used.
TEST(gpu_temperature, FallsBackToEdgeWhenHotspotDisabled)
{
    const auto m = make_temp_metrics(72, 55);

    EXPECT_DOUBLE_EQ(select_gpu_temperature(make_temp_enabled(false, true), m), 55.0);
}

TEST(gpu_temperature, HasGpuTemperatureOutputReflectsBits)
{
    EXPECT_TRUE(has_gpu_temperature_output(make_temp_enabled(true, false)));
    EXPECT_TRUE(has_gpu_temperature_output(make_temp_enabled(false, true)));
    EXPECT_TRUE(has_gpu_temperature_output(make_temp_enabled(true, true)));
    EXPECT_FALSE(has_gpu_temperature_output(make_temp_enabled(false, false)));
}

// Track label matches the reading select_gpu_temperature() emits.
TEST(gpu_temperature, TrackLabelMatchesSelectedReading)
{
    EXPECT_STREQ(gpu_temperature_track_label(make_temp_enabled(true, false)),
                 "Hotspot Temp");
    EXPECT_STREQ(gpu_temperature_track_label(make_temp_enabled(true, true)),
                 "Hotspot Temp");
    EXPECT_STREQ(gpu_temperature_track_label(make_temp_enabled(false, true)), "Edge Temp");
}
