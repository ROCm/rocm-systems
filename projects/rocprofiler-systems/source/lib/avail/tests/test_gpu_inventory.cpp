// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/catalog.hpp"
#include "avail/devices.hpp"
#include "avail/gpu_counters.hpp"
#include "avail/records.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace rocprofsys::avail
{
namespace
{

struct fake_sdk
{
    using status_t        = int;
    using agent_version_t = int;
    using agent_id        = struct
    {
        std::uint64_t handle;
    };
    using counter_id = struct
    {
        std::uint64_t handle;
    };

    struct agent_t
    {
        agent_id      id{};
        int           type                 = 0;
        std::uint32_t logical_node_type_id = 0;
        std::uint32_t location_id          = 0;
        std::uint32_t domain               = 0;
        std::uint32_t node_id              = 0;
        const char*   name                 = "";
        const char*   vendor_name          = "";
        const char*   product_name         = "";
    };

    struct counter_info_v0_t
    {
        const char*   name        = nullptr;
        const char*   description = "";
        const char*   block       = "";
        const char*   expression  = "";
        std::uint32_t is_constant = 0;
        std::uint32_t is_derived  = 0;
    };

    struct dimension_info_t
    {
        const char*   name          = "";
        std::uint64_t instance_size = 0;
    };

    using query_available_agents_cb_t = status_t (*)(agent_version_t, const void**,
                                                     std::size_t, void*);
    using available_counters_cb_t     = status_t (*)(agent_id, counter_id*, std::size_t,
                                                 void*);
    using available_dimensions_cb_t   = status_t (*)(counter_id, const dimension_info_t*,
                                                   std::size_t, void*);

    // Names match rocprofiler_sdk::wrapper (C/SDK identifiers).
    // NOLINTBEGIN(readability-identifier-naming)
    static constexpr status_t        STATUS_SUCCESS         = 0;
    static constexpr status_t        STATUS_ERROR           = 1;
    static constexpr agent_version_t AGENT_INFO_VERSION_0   = 0;
    static constexpr int             AGENT_TYPE_GPU         = 1;
    static constexpr int             AGENT_TYPE_CPU         = 0;
    static constexpr int             COUNTER_INFO_VERSION_0 = 0;
    // NOLINTEND(readability-identifier-naming)

    inline static std::vector<agent_t> agents{};
    inline static status_t             agent_query_status = STATUS_SUCCESS;
    inline static std::map<std::uint64_t, std::vector<counter_id>> counters_by_agent{};
    inline static std::map<std::uint64_t, counter_info_v0_t>       counter_info{};
    inline static std::map<std::uint64_t, std::vector<dimension_info_t>> dimensions{};

    static void reset()
    {
        agents.clear();
        agent_query_status = STATUS_SUCCESS;
        counters_by_agent.clear();
        counter_info.clear();
        dimensions.clear();
    }

    static const char* get_status_string(status_t) { return "sdk-error"; }

    static status_t query_available_agents(agent_version_t,
                                           query_available_agents_cb_t callback,
                                           std::size_t, void* user_data)
    {
        if(agent_query_status != STATUS_SUCCESS)
        {
            return agent_query_status;
        }
        std::vector<const void*> pointers;
        pointers.reserve(agents.size());
        for(const auto& agent : agents)
        {
            pointers.push_back(&agent);
        }
        return callback(AGENT_INFO_VERSION_0,
                        pointers.empty() ? nullptr : pointers.data(), pointers.size(),
                        user_data);
    }

    static status_t iterate_agent_supported_counters(agent_id                agent,
                                                     available_counters_cb_t callback,
                                                     void*                   user_data)
    {
        auto& ids = counters_by_agent[agent.handle];
        return callback(agent, ids.data(), ids.size(), user_data);
    }

    static status_t iterate_counter_dimensions(counter_id                counter,
                                               available_dimensions_cb_t callback,
                                               void*                     user_data)
    {
        auto& dims = dimensions[counter.handle];
        return callback(counter, dims.data(), dims.size(), user_data);
    }

    static status_t query_counter_info(counter_id counter, int, void* info)
    {
        const auto found = counter_info.find(counter.handle);
        if(found == counter_info.end())
        {
            return STATUS_ERROR;
        }
        *static_cast<counter_info_v0_t*>(info) = found->second;
        return STATUS_SUCCESS;
    }
};

constexpr std::uint64_t k_cpu_agent_handle     = 11;
constexpr std::uint64_t k_gpu_agent_handle     = 42;
constexpr std::uint32_t k_gpu_location_id      = 0x300;
constexpr std::uint64_t k_counter_agent_handle = 7;

[[nodiscard]] std::string
issue_message(const gpu_agents_listing_result& listed)
{
    if(!listed.issue.has_value())
    {
        return {};
    }
    return listed.issue->message;
}

[[nodiscard]] source_id
issue_source(const gpu_agents_listing_result& listed)
{
    if(!listed.issue.has_value())
    {
        return source_id::catalog;
    }
    return listed.issue->source;
}

TEST(avail_gpu_inventory, gpu_agents_skip_cpu_and_map_fields)
{
    fake_sdk::reset();
    fake_sdk::agents = {
        fake_sdk::agent_t{
            .id                   = { .handle = k_cpu_agent_handle },
            .type                 = fake_sdk::AGENT_TYPE_CPU,
            .logical_node_type_id = 0,
            .name                 = "CPU",
        },
        fake_sdk::agent_t{
            .id                   = { .handle = k_gpu_agent_handle },
            .type                 = fake_sdk::AGENT_TYPE_GPU,
            .logical_node_type_id = 1,
            .location_id          = k_gpu_location_id,
            .domain               = 0,
            .node_id              = 2,
            .name                 = "gfx90a",
            .vendor_name          = "AMD",
            .product_name         = "MI210",
        },
    };

    const auto listed = inventory::gpu_agents<fake_sdk>();
    ASSERT_FALSE(listed.issue.has_value());
    ASSERT_EQ(listed.agents.size(), 1u);
    EXPECT_EQ(listed.agents[0].handle, k_gpu_agent_handle);
    EXPECT_EQ(listed.agents[0].device.id, "gpu-1");
    EXPECT_EQ(listed.agents[0].device.name, "MI210");
    EXPECT_EQ(listed.agents[0].device.kind, device_kind::gpu);
    EXPECT_EQ(listed.agents[0].device.vendor, std::optional<std::string>{ "AMD" });
    EXPECT_EQ(listed.agents[0].device.pci_bdf,
              std::optional<std::string>{ "0000:03:00.0" });
    EXPECT_EQ(listed.agents[0].device.numa_node, std::optional<std::size_t>{ 2 });
}

TEST(avail_gpu_inventory, gpu_agents_query_failure_is_not_a_stub)
{
    fake_sdk::reset();
    fake_sdk::agent_query_status = fake_sdk::STATUS_ERROR;

    const auto listed = inventory::gpu_agents<fake_sdk>();
    EXPECT_TRUE(listed.agents.empty());
    EXPECT_TRUE(listed.issue.has_value());
    EXPECT_EQ(issue_source(listed), source_id::rocprofiler_sdk);
    EXPECT_EQ(issue_message(listed), "sdk-error");
    EXPECT_NE(issue_message(listed), k_not_implemented_message);
}

TEST(avail_gpu_inventory, gpu_counters_skip_constants_and_keep_derived)
{
    fake_sdk::reset();
    gpu_agent_info agent;
    agent.handle    = k_counter_agent_handle;
    agent.device.id = "gpu-0";

    fake_sdk::counters_by_agent[k_counter_agent_handle] = { { .handle = 1 },
                                                            { .handle = 2 },
                                                            { .handle = 3 } };
    fake_sdk::counter_info[1]                           = {
                                  .name = "GRBM_COUNT", .description = "busy", .block = "GRBM", .is_constant = 0
    };
    fake_sdk::counter_info[2] = { .name = "MAX_WAVE_SIZE", .is_constant = 1 };
    fake_sdk::counter_info[3] = { .name        = "TCC_HIT_SUM",
                                  .expression  = "TCC_HIT[0]+TCC_HIT[1]",
                                  .is_constant = 0,
                                  .is_derived  = 1 };
    fake_sdk::dimensions[1]   = { { .name = "SE", .instance_size = 4 } };

    const auto records = inventory::gpu_counters<fake_sdk>({ agent });
    ASSERT_EQ(records.size(), 2u);
    EXPECT_EQ(records[0].name, "GRBM_COUNT");
    EXPECT_EQ(records[0].device_id, "gpu-0");
    EXPECT_EQ(records[0].block, std::optional<std::string>{ "GRBM" });
    ASSERT_EQ(records[0].dimensions.size(), 1u);
    EXPECT_EQ(records[0].dimensions[0].name, "SE");
    EXPECT_EQ(records[0].dimensions[0].extent, 4u);
    EXPECT_EQ(records[1].name, "TCC_HIT_SUM");
    EXPECT_EQ(records[1].expression,
              std::optional<std::string>{ "TCC_HIT[0]+TCC_HIT[1]" });
}

}  // namespace
}  // namespace rocprofsys::avail
