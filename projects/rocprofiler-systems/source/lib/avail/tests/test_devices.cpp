// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/devices.hpp"
#include <cstdint>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace rocprofsys::avail
{
namespace
{
sdk_agent_entry
make_agent(std::uint64_t handle, std::size_t index, std::string name, std::string bdf)
{
    auto entry            = sdk_agent_entry{};
    entry.handle          = handle;
    entry.index           = index;
    entry.name            = std::move(name);
    entry.product_name    = "Instinct";
    entry.vendor_name     = "AMD";
    entry.pci_bdf         = std::move(bdf);
    entry.runtime_visible = true;
    return entry;
}

/// Records whether it was consulted so the tests can assert that a backend is
/// never initialized when there is nothing for it to do.
struct fake_agents
{
    std::vector<sdk_agent_entry> agents  = {};
    std::string                  failure = {};
    mutable int                  calls   = 0;

    std::vector<sdk_agent_entry> gpu_agents() const
    {
        ++calls;
        if(!failure.empty()) throw std::runtime_error{ failure };
        return agents;
    }
};

struct fake_smi
{
    std::vector<smi_device_entry> devices = {};
    std::string                   failure = {};
    mutable int                   calls   = 0;

    std::vector<smi_device_entry> gpu_devices() const
    {
        ++calls;
        if(!failure.empty()) throw std::runtime_error{ failure };
        return devices;
    }
};

static_assert(agent_inventory_source<fake_agents>);
static_assert(smi_inventory_source<fake_smi>);

TEST(avail_devices, sdk_only_yields_agents_without_smi_fields)
{
    auto agents = fake_agents{ { make_agent(1, 0, "gfx942", "0000:0c:00.0") } };
    auto smi    = fake_smi{};

    const auto result = collect_devices(agents, smi);

    ASSERT_EQ(result.devices.size(), 1U);
    EXPECT_EQ(result.devices[0].agent_handle, 1U);
    EXPECT_EQ(result.devices[0].name, "gfx942");
    EXPECT_EQ(result.devices[0].pci_bdf, "0000:0c:00.0");
    EXPECT_FALSE(result.devices[0].has_smi_data());
    EXPECT_TRUE(result.diagnostics.empty());
}

TEST(avail_devices, smi_enriches_agent_matched_by_bdf)
{
    auto agents = fake_agents{ { make_agent(1, 0, "gfx942", "0000:0c:00.0"),
                                 make_agent(2, 1, "gfx942", "0000:22:00.0") } };
    auto smi    = fake_smi{ { { "0000:22:00.0", "AMD Instinct MI300X", "AMD Inc." } } };

    const auto result = collect_devices(agents, smi);

    ASSERT_EQ(result.devices.size(), 2U);
    EXPECT_FALSE(result.devices[0].has_smi_data());

    ASSERT_TRUE(result.devices[1].smi_market_name.has_value());
    EXPECT_EQ(*result.devices[1].smi_market_name, "AMD Instinct MI300X");
    ASSERT_TRUE(result.devices[1].smi_vendor_name.has_value());
    EXPECT_EQ(*result.devices[1].smi_vendor_name, "AMD Inc.");
    EXPECT_TRUE(result.diagnostics.empty());
}

TEST(avail_devices, bdf_correlation_ignores_hex_case)
{
    auto agents = fake_agents{ { make_agent(1, 0, "gfx90a", "0000:0A:00.0") } };
    auto smi    = fake_smi{ { { "0000:0a:00.0", "AMD Instinct MI250X", "AMD" } } };

    const auto result = collect_devices(agents, smi);

    ASSERT_EQ(result.devices.size(), 1U);
    ASSERT_TRUE(result.devices[0].smi_market_name.has_value());
    EXPECT_EQ(*result.devices[0].smi_market_name, "AMD Instinct MI250X");
}

TEST(avail_devices, smi_fields_stay_unset_when_smi_reports_nothing_for_the_bdf)
{
    auto agents = fake_agents{ { make_agent(1, 0, "gfx942", "0000:0c:00.0") } };
    auto smi    = fake_smi{ { { "0000:ff:00.0", "Other GPU", "AMD" } } };

    const auto result = collect_devices(agents, smi);

    ASSERT_EQ(result.devices.size(), 1U);
    EXPECT_FALSE(result.devices[0].smi_market_name.has_value());
    EXPECT_FALSE(result.devices[0].smi_vendor_name.has_value());
    EXPECT_TRUE(result.diagnostics.empty());
}

TEST(avail_devices, smi_device_without_a_matching_agent_is_dropped)
{
    auto agents = fake_agents{ { make_agent(1, 0, "gfx942", "0000:0c:00.0") } };
    auto smi    = fake_smi{ { { "0000:0c:00.0", "MI300X", "AMD" },
                              { "0000:99:00.0", "Unprofilable", "AMD" } } };

    const auto result = collect_devices(agents, smi);

    EXPECT_EQ(result.devices.size(), 1U);
}

TEST(avail_devices, empty_smi_strings_do_not_populate_optional_fields)
{
    auto agents = fake_agents{ { make_agent(1, 0, "gfx942", "0000:0c:00.0") } };
    auto smi    = fake_smi{ { { "0000:0c:00.0", "", "AMD" } } };

    const auto result = collect_devices(agents, smi);

    ASSERT_EQ(result.devices.size(), 1U);
    EXPECT_FALSE(result.devices[0].smi_market_name.has_value());
    EXPECT_TRUE(result.devices[0].smi_vendor_name.has_value());
}

TEST(avail_devices, no_agents_is_an_empty_success_and_never_touches_smi)
{
    auto agents = fake_agents{};
    auto smi    = fake_smi{};

    const auto result = collect_devices(agents, smi);

    EXPECT_TRUE(result.devices.empty());
    EXPECT_TRUE(result.diagnostics.empty());
    EXPECT_EQ(smi.calls, 0);
}

TEST(avail_devices, sdk_failure_reports_a_diagnostic_and_skips_smi)
{
    auto agents    = fake_agents{};
    agents.failure = "sdk unavailable";
    auto smi       = fake_smi{ { { "0000:0c:00.0", "MI300X", "AMD" } } };

    const auto result = collect_devices(agents, smi);

    EXPECT_TRUE(result.devices.empty());
    ASSERT_EQ(result.diagnostics.size(), 1U);
    EXPECT_EQ(result.diagnostics[0].source, source_id::rocprofiler_sdk);
    EXPECT_EQ(result.diagnostics[0].message, "sdk unavailable");
    EXPECT_EQ(smi.calls, 0);
}

TEST(avail_devices, smi_failure_still_returns_the_sdk_devices)
{
    auto agents = fake_agents{ { make_agent(1, 0, "gfx942", "0000:0c:00.0") } };
    auto smi    = fake_smi{};
    smi.failure = "amd-smi init failed";

    const auto result = collect_devices(agents, smi);

    ASSERT_EQ(result.devices.size(), 1U);
    EXPECT_EQ(result.devices[0].name, "gfx942");
    EXPECT_FALSE(result.devices[0].has_smi_data());
    ASSERT_EQ(result.diagnostics.size(), 1U);
    EXPECT_EQ(result.diagnostics[0].source, source_id::amd_smi);
    EXPECT_EQ(result.diagnostics[0].message, "amd-smi init failed");
}

TEST(avail_devices, records_are_ordered_by_device_index)
{
    auto agents = fake_agents{ { make_agent(7, 2, "gfx942", "0000:0c:00.0"),
                                 make_agent(3, 0, "gfx942", "0000:0d:00.0"),
                                 make_agent(5, 1, "gfx942", "0000:0e:00.0") } };
    auto smi    = fake_smi{};

    const auto result = collect_devices(agents, smi);

    ASSERT_EQ(result.devices.size(), 3U);
    EXPECT_EQ(result.devices[0].index, 0U);
    EXPECT_EQ(result.devices[1].index, 1U);
    EXPECT_EQ(result.devices[2].index, 2U);
}

TEST(avail_devices, runtime_visibility_is_carried_through)
{
    auto agent            = make_agent(1, 0, "gfx942", "0000:0c:00.0");
    agent.runtime_visible = false;

    auto agents = fake_agents{ { agent } };
    auto smi    = fake_smi{};

    const auto result = collect_devices(agents, smi);

    ASSERT_EQ(result.devices.size(), 1U);
    EXPECT_FALSE(result.devices[0].runtime_visible);
}

TEST(avail_devices, agent_without_a_bdf_is_reported_without_smi_data)
{
    auto agents = fake_agents{ { make_agent(1, 0, "gfx942", "") } };
    auto smi    = fake_smi{ { { "", "Bogus", "AMD" } } };

    const auto result = collect_devices(agents, smi);

    ASSERT_EQ(result.devices.size(), 1U);
    EXPECT_FALSE(result.devices[0].has_smi_data());
}
}  // namespace
}  // namespace rocprofsys::avail
