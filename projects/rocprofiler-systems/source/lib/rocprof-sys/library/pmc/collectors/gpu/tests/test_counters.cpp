// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for the AMD SMI GPU counter table (library/pmc/collectors/gpu/counters.hpp):
// the names, units, enable rules and value readers every output uses.

#include "core/agent.hpp"
#include "library/pmc/collectors/gpu/counters.hpp"
#include "library/pmc/collectors/gpu/types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <vector>

using namespace rocprofsys::pmc::collectors::gpu;

namespace
{

const counter*
find_counter(std::string_view perfetto_label)
{
    const auto& all   = counters();
    const auto  found = std::ranges::find(all, perfetto_label, &counter::perfetto_label);
    return found == all.end() ? nullptr : &*found;
}

std::vector<std::string>
collected_labels(std::uint32_t enabled_mask)
{
    std::vector<std::string> labels;
    for(const auto* item : collected_counters(enabled_mask))
    {
        labels.push_back(item->perfetto_label);
    }
    return labels;
}

}  // namespace

TEST(gpu_counters, power_prefers_current_and_falls_back_to_average)
{
    constexpr auto k_both =
        metric_bit::k_current_socket_power | metric_bit::k_average_socket_power;
    EXPECT_EQ(collected_labels(k_both), std::vector<std::string>{ "Current Power" });
    EXPECT_EQ(collected_labels(metric_bit::k_average_socket_power),
              std::vector<std::string>{ "Avg. Power" });

    constexpr std::uint32_t k_current = 150;
    constexpr std::uint32_t k_average = 140;
    metrics                 values{};
    values.current_socket_power = k_current;
    values.average_socket_power = k_average;
    EXPECT_DOUBLE_EQ(collected_counters(k_both).front()->read(values), k_current);
    EXPECT_DOUBLE_EQ(
        collected_counters(metric_bit::k_average_socket_power).front()->read(values),
        k_average);
}

TEST(gpu_counters, temperature_prefers_hotspot_and_falls_back_to_edge)
{
    constexpr std::int64_t k_hotspot = 80;
    constexpr std::int64_t k_edge    = 60;
    metrics                values{};
    values.hotspot_temperature = k_hotspot;
    values.edge_temperature    = k_edge;

    const auto hotspot = collected_counters(metric_bit::k_hotspot_temperature |
                                            metric_bit::k_edge_temperature);
    const auto edge    = collected_counters(metric_bit::k_edge_temperature);
    ASSERT_EQ(hotspot.size(), 1U);
    ASSERT_EQ(edge.size(), 1U);
    EXPECT_DOUBLE_EQ(hotspot.front()->read(values), k_hotspot);
    EXPECT_DOUBLE_EQ(edge.front()->read(values), k_edge);
    EXPECT_EQ(hotspot.front()->pmc_name, edge.front()->pmc_name);
}

TEST(gpu_counters, metric_bits_match_the_enabled_metrics_bitfield)
{
    enabled_metrics enabled{};
    enabled.bits.jpeg_busy = 1;
    EXPECT_EQ(enabled.value, metric_bit::k_jpeg_busy);
    enabled.value          = 0;
    enabled.bits.mem_clock = 1;
    EXPECT_EQ(enabled.value, metric_bit::k_mem_clock);
    enabled.value                     = 0;
    enabled.bits.current_socket_power = 1;
    EXPECT_EQ(enabled.value, metric_bit::k_current_socket_power);
}

// develop registered 419 pmc_info rows and tracks per GPU; the fallbacks share them.
TEST(gpu_counters, one_unique_pmc_name_per_counter_except_fallbacks)
{
    std::set<std::string> names;
    std::size_t           primary = 0;
    for(const auto& item : counters())
    {
        if(!is_fallback(item))
        {
            ++primary;
            names.insert(item.pmc_name);
        }
    }
    EXPECT_EQ(primary, 419U);
    EXPECT_EQ(names.size(), primary);
}

TEST(gpu_counters, xcp_counter_reads_its_element_or_nan)
{
    const auto* item = find_counter("VCN Busy XCP_1: [02]");
    ASSERT_NE(item, nullptr);
    EXPECT_EQ(item->pmc_name, "device_vcn_activity_1_2");
    EXPECT_EQ(item->rocpd_track, item->pmc_name);

    constexpr std::uint16_t k_busy = 37;
    metrics                 values{};
    values.xcp_stats[1].vcn_busy[2] = k_busy;
    EXPECT_DOUBLE_EQ(item->read(values), k_busy);
    values.xcp_stats[1].vcn_busy[2] = METRIC_VALUE_NOT_SUPPORTED_16;
    EXPECT_TRUE(std::isnan(item->read(values)));
}

TEST(gpu_counters, xgmi_link_counter_names_and_absent_value)
{
    const auto* item = find_counter("XGMI Read Data [03]");
    ASSERT_NE(item, nullptr);
    EXPECT_EQ(item->pmc_name, "device_xgmi_read_data_link3");
    EXPECT_EQ(item->rocpd_track, "device_xgmi_read_data [Link 3]");
    EXPECT_EQ(item->symbol, "XGMI Read [Link 3]");

    metrics values{};
    values.xgmi.data_acc.read[3] = std::numeric_limits<std::uint64_t>::max();
    EXPECT_TRUE(std::isnan(item->read(values)));
}

TEST(gpu_counters, schema_names_the_device_only_in_perfetto_tracks)
{
    const auto schema = make_schema(1, collected_counters(metric_bit::k_gfx_activity));

    EXPECT_EQ(schema.type, rocprofsys::agent_type::gpu);
    EXPECT_EQ(schema.agent_type_index, 1U);
    EXPECT_EQ(schema.event_category, "amd_smi");
    ASSERT_EQ(schema.series.size(), 1U);
    EXPECT_EQ(schema.series.front().perfetto_track, "GPU [1] GFX Busy (S)");
    EXPECT_EQ(schema.series.front().pmc_name, "device_busy_gfx");
    EXPECT_EQ(schema.series.front().rocpd_track, "device_busy_gfx");
    EXPECT_EQ(schema.series.front().units, "%");
}
