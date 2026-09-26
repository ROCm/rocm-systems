// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/gpu_counters.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace rocprofsys::avail
{
namespace
{
device_record
make_device(std::uint64_t handle, std::size_t index, std::string name)
{
    auto record         = device_record{};
    record.agent_handle = handle;
    record.index        = index;
    record.name         = std::move(name);
    return record;
}

counter_record
make_counter(std::uint64_t id, std::string name, std::string block)
{
    auto record        = counter_record{};
    record.id          = id;
    record.name        = std::move(name);
    record.description = "description of " + record.name;
    record.block       = std::move(block);
    return record;
}

struct fake_counters
{
    std::map<std::uint64_t, std::vector<std::uint64_t>> ids_by_agent = {};
    std::map<std::uint64_t, counter_record>             records      = {};
    std::set<std::uint64_t>                             undescribed  = {};
    std::set<std::uint64_t>                             failing      = {};
    std::set<std::uint64_t>                             throwing_ids = {};

    std::vector<std::uint64_t> counter_ids(std::uint64_t agent_handle)
    {
        if(failing.count(agent_handle) != 0)
            throw std::runtime_error{ "counter listing failed" };
        const auto found = ids_by_agent.find(agent_handle);
        return found == ids_by_agent.end() ? std::vector<std::uint64_t>{}
                                           : found->second;
    }

    std::vector<counter_record> describe(std::uint64_t counter_id)
    {
        if(throwing_ids.count(counter_id) != 0)
            throw std::runtime_error{ "counter query failed" };
        if(undescribed.count(counter_id) != 0) return {};
        const auto found = records.find(counter_id);
        if(found == records.end()) return {};
        return { found->second };
    }
};

static_assert(counter_inventory_source<fake_counters>);

TEST(avail_gpu_counters, enumerates_counters_for_a_single_agent)
{
    auto source                = fake_counters{};
    source.ids_by_agent[10]    = { 1, 2 };
    source.records[1]          = make_counter(1, "GRBM_GUI_ACTIVE", "GRBM");
    source.records[2]          = make_counter(2, "SQ_WAVES", "SQ");

    const auto result =
        collect_gpu_counters(source, { make_device(10, 0, "gfx942") });

    ASSERT_EQ(result.devices.size(), 1U);
    EXPECT_EQ(result.devices[0].agent_handle, 10U);
    EXPECT_EQ(result.devices[0].index, 0U);
    EXPECT_EQ(result.devices[0].device_name, "gfx942");
    ASSERT_EQ(result.devices[0].counters.size(), 2U);
    EXPECT_TRUE(result.diagnostics.empty());
}

TEST(avail_gpu_counters, preserves_metadata_and_dimensions)
{
    auto counter        = make_counter(1, "SQ_WAVES", "SQ");
    counter.expression  = "SQ_WAVES_sum";
    counter.is_derived  = true;
    counter.is_constant = false;
    counter.dimensions  = { { "SHADER_ENGINE", 4 }, { "SIMD", 1 } };

    auto source             = fake_counters{};
    source.ids_by_agent[10] = { 1 };
    source.records[1]       = counter;

    const auto result =
        collect_gpu_counters(source, { make_device(10, 0, "gfx942") });

    ASSERT_EQ(result.devices.size(), 1U);
    ASSERT_EQ(result.devices[0].counters.size(), 1U);

    const auto& out = result.devices[0].counters[0];
    EXPECT_EQ(out.name, "SQ_WAVES");
    EXPECT_EQ(out.description, "description of SQ_WAVES");
    EXPECT_EQ(out.block, "SQ");
    EXPECT_EQ(out.expression, "SQ_WAVES_sum");
    EXPECT_TRUE(out.is_derived);
    EXPECT_FALSE(out.is_constant);
    ASSERT_EQ(out.dimensions.size(), 2U);
    EXPECT_EQ(out.dimensions[0].name, "SHADER_ENGINE");
    EXPECT_EQ(out.dimensions[0].position, 4U);
    EXPECT_EQ(out.dimensions[1].name, "SIMD");
}

TEST(avail_gpu_counters, counters_are_sorted_by_name)
{
    auto source             = fake_counters{};
    source.ids_by_agent[10] = { 3, 1, 2 };
    source.records[1]       = make_counter(1, "SQ_WAVES", "SQ");
    source.records[2]       = make_counter(2, "GRBM_COUNT", "GRBM");
    source.records[3]       = make_counter(3, "TCC_HIT", "TCC");

    const auto result =
        collect_gpu_counters(source, { make_device(10, 0, "gfx942") });

    ASSERT_EQ(result.devices[0].counters.size(), 3U);
    EXPECT_EQ(result.devices[0].counters[0].name, "GRBM_COUNT");
    EXPECT_EQ(result.devices[0].counters[1].name, "SQ_WAVES");
    EXPECT_EQ(result.devices[0].counters[2].name, "TCC_HIT");
}

TEST(avail_gpu_counters, groups_counters_per_gpu)
{
    auto source             = fake_counters{};
    source.ids_by_agent[10] = { 1 };
    source.ids_by_agent[20] = { 2, 3 };
    source.records[1]       = make_counter(1, "SQ_WAVES", "SQ");
    source.records[2]       = make_counter(2, "SQ_WAVES", "SQ");
    source.records[3]       = make_counter(3, "TCC_HIT", "TCC");

    const auto result = collect_gpu_counters(
        source, { make_device(10, 0, "gfx942"), make_device(20, 1, "gfx90a") });

    ASSERT_EQ(result.devices.size(), 2U);
    EXPECT_EQ(result.devices[0].index, 0U);
    EXPECT_EQ(result.devices[0].counters.size(), 1U);
    EXPECT_EQ(result.devices[1].index, 1U);
    EXPECT_EQ(result.devices[1].counters.size(), 2U);
}

TEST(avail_gpu_counters, a_counter_name_shared_by_two_gpus_is_kept_on_both)
{
    auto source             = fake_counters{};
    source.ids_by_agent[10] = { 1 };
    source.ids_by_agent[20] = { 1 };
    source.records[1]       = make_counter(1, "SQ_WAVES", "SQ");

    const auto result = collect_gpu_counters(
        source, { make_device(10, 0, "gfx942"), make_device(20, 1, "gfx942") });

    ASSERT_EQ(result.devices.size(), 2U);
    ASSERT_EQ(result.devices[0].counters.size(), 1U);
    ASSERT_EQ(result.devices[1].counters.size(), 1U);
    EXPECT_EQ(result.devices[0].counters[0].name, "SQ_WAVES");
    EXPECT_EQ(result.devices[1].counters[0].name, "SQ_WAVES");
}

TEST(avail_gpu_counters, no_devices_is_an_empty_success)
{
    auto source = fake_counters{};

    const auto result = collect_gpu_counters(source, {});

    EXPECT_TRUE(result.devices.empty());
    EXPECT_TRUE(result.diagnostics.empty());
}

TEST(avail_gpu_counters, agent_reporting_no_counters_yields_an_empty_group)
{
    auto source = fake_counters{};

    const auto result =
        collect_gpu_counters(source, { make_device(10, 0, "gfx942") });

    ASSERT_EQ(result.devices.size(), 1U);
    EXPECT_TRUE(result.devices[0].counters.empty());
    EXPECT_TRUE(result.diagnostics.empty());
}

TEST(avail_gpu_counters, a_failing_agent_does_not_discard_the_other_agents)
{
    auto source             = fake_counters{};
    source.failing          = { 10 };
    source.ids_by_agent[20] = { 1 };
    source.records[1]       = make_counter(1, "SQ_WAVES", "SQ");

    const auto result = collect_gpu_counters(
        source, { make_device(10, 0, "gfx942"), make_device(20, 1, "gfx90a") });

    ASSERT_EQ(result.devices.size(), 2U);
    EXPECT_TRUE(result.devices[0].counters.empty());
    EXPECT_EQ(result.devices[1].counters.size(), 1U);

    ASSERT_EQ(result.diagnostics.size(), 1U);
    EXPECT_EQ(result.diagnostics[0].source, source_id::rocprofiler_sdk);
    EXPECT_EQ(result.diagnostics[0].message, "counter listing failed");
}

TEST(avail_gpu_counters, undescribable_counters_are_skipped_without_a_warning)
{
    auto source             = fake_counters{};
    source.ids_by_agent[10] = { 1, 2 };
    source.records[1]       = make_counter(1, "SQ_WAVES", "SQ");
    source.undescribed      = { 2 };

    const auto result =
        collect_gpu_counters(source, { make_device(10, 0, "gfx942") });

    ASSERT_EQ(result.devices[0].counters.size(), 1U);
    EXPECT_EQ(result.devices[0].counters[0].name, "SQ_WAVES");
    EXPECT_TRUE(result.diagnostics.empty());
}

TEST(avail_gpu_counters, a_throwing_description_warns_and_keeps_the_rest)
{
    auto source             = fake_counters{};
    source.ids_by_agent[10] = { 1, 2 };
    source.records[1]       = make_counter(1, "SQ_WAVES", "SQ");
    source.throwing_ids     = { 2 };

    const auto result =
        collect_gpu_counters(source, { make_device(10, 0, "gfx942") });

    ASSERT_EQ(result.devices[0].counters.size(), 1U);
    EXPECT_EQ(result.devices[0].counters[0].name, "SQ_WAVES");
    ASSERT_EQ(result.diagnostics.size(), 1U);
    EXPECT_EQ(result.diagnostics[0].message, "counter query failed");
}
}  // namespace
}  // namespace rocprofsys::avail
