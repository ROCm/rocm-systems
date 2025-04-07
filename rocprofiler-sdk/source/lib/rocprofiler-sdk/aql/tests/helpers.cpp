// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

#include <gtest/gtest.h>
#include <map>
#include <unordered_set>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

using namespace rocprofiler;
using namespace rocprofiler::counters::test_constants;

namespace
{
auto
findDeviceMetrics(const rocprofiler_agent_t& agent, const std::unordered_set<std::string>& metrics)
{
    std::vector<counters::Metric> ret;
    auto                          all_counters = counters::loadMetrics()->arch_to_metric;

    ROCP_INFO << "Looking up counters for " << std::string(agent.name);

    auto* gfx_metrics = common::get_val(all_counters, std::string(agent.name));
    if(!gfx_metrics)
    {
        ROCP_ERROR << "No counters found for " << std::string(agent.name);
        return ret;
    }

    for(auto& counter : *gfx_metrics)
    {
        if((metrics.count(counter.name()) > 0 || metrics.empty()) && !counter.block().empty())
        {
            ret.push_back(counter);
        }
    }
    return ret;
}

hsa_ven_amd_aqlprofile_id_query_t
v1_get_query_info(hsa_agent_t agent, const counters::Metric& metric)
{
    hsa_ven_amd_aqlprofile_profile_t  profile = {.agent  = agent,
                                                .type   = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC,
                                                .events = nullptr,
                                                .event_count     = 0,
                                                .parameters      = nullptr,
                                                .parameter_count = 0,
                                                .output_buffer   = {nullptr, 0},
                                                .command_buffer  = {nullptr, 0}};
    hsa_ven_amd_aqlprofile_id_query_t query   = {metric.block().c_str(), 0, 0};
    if(hsa_ven_amd_aqlprofile_get_info(&profile, HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID, &query) !=
       HSA_STATUS_SUCCESS)
    {
        DLOG(FATAL) << fmt::format("AQL failed to query info for counter {}", metric);
        throw std::runtime_error(fmt::format("AQL failed to query info for counter {}", metric));
    }
    return query;
}

uint32_t
v1_get_block_counters(hsa_agent_t agent, const hsa_ven_amd_aqlprofile_event_t& event)
{
    hsa_ven_amd_aqlprofile_profile_t query              = {.agent       = agent,
                                              .type        = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC,
                                              .events      = &event,
                                              .event_count = 1,
                                              .parameters  = nullptr,
                                              .parameter_count = 0,
                                              .output_buffer   = {nullptr, 0},
                                              .command_buffer  = {nullptr, 0}};
    uint32_t                         max_block_counters = 0;
    if(hsa_ven_amd_aqlprofile_get_info(&query,
                                       HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS,
                                       &max_block_counters) != HSA_STATUS_SUCCESS)
    {
        throw std::runtime_error(fmt::format("AQL failed to max block info for counter {}",
                                             static_cast<int64_t>(event.block_name)));
    }
    return max_block_counters;
}

rocprofiler_status_t
v1_get_dim_info(hsa_agent_t                    agent,
                hsa_ven_amd_aqlprofile_event_t event,
                uint32_t                       sample_id,
                std::map<int, uint64_t>&       dims)
{
    auto callback = [](int, int id, int extent, int, const char*, void* userdata) -> hsa_status_t {
        auto& map = *static_cast<std::map<int, uint64_t>*>(userdata);
        map.emplace(id, extent);
        return HSA_STATUS_SUCCESS;
    };

    if(hsa_ven_amd_aqlprofile_iterate_event_coord(
           agent, event, sample_id, callback, static_cast<void*>(&dims)) != HSA_STATUS_SUCCESS)
    {
        return ROCPROFILER_STATUS_ERROR_AQL_NO_EVENT_COORD;
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

void
test_init()
{
    HsaApiTable table;
    table.amd_ext_ = &get_ext_table();
    table.core_    = &get_api_table();
    agent::construct_agent_cache(&table);
    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
    hsa::get_queue_controller()->init(get_api_table(), get_ext_table());
}
}  // namespace

TEST(aql_helpers, get_query_info)
{
    auto agents = agent::get_agents();
    ASSERT_FALSE(agents.empty());

    for(auto agent : agents)
    {
        // auto aql_agent = *CHECK_NOTNULL(agent::get_aql_agent(agent->id));
        if(agent->type == ROCPROFILER_AGENT_TYPE_CPU) continue;
        auto metrics = findDeviceMetrics(*agent, {});
        ASSERT_FALSE(metrics.empty());

        for(auto& metric : metrics)
        {
            auto query = aql::get_query_info(agent->id, metric);
            ROCP_INFO << fmt::format("{},{},{}", query.id, query.name, query.instance_count);
            EXPECT_TRUE(query.name != nullptr);
            EXPECT_TRUE(query.instance_count != 0);
            EXPECT_TRUE(query.id < std::numeric_limits<uint32_t>().max());
        }
    }
}

TEST(aql_helpers, get_query_info_compare_v1)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();
    auto agents = agent::get_agents();

    ASSERT_FALSE(agents.empty());

    for(auto agent : agents)
    {
        if(agent->type == ROCPROFILER_AGENT_TYPE_CPU) continue;
        auto metrics = findDeviceMetrics(*agent, {});
        ASSERT_FALSE(metrics.empty());

        for(auto& metric : metrics)
        {
            auto query = aql::get_query_info(agent->id, metric);
            auto query_v1 =
                v1_get_query_info(agent::get_agent_cache(agent)->get_hsa_agent(), metric);
            // v1 query with hsa_agent

            EXPECT_EQ(query.id, query_v1.id);
            EXPECT_EQ(std::string(query.name), std::string(query_v1.name));
        }
    }
    hsa_shut_down();
}

TEST(aql_helpers, get_block_counters)
{
    auto agents = agent::get_agents();
    ASSERT_FALSE(agents.empty());

    for(auto agent : agents)
    {
        if(agent->type == ROCPROFILER_AGENT_TYPE_CPU) continue;
        auto metrics = findDeviceMetrics(*agent, {});
        ASSERT_FALSE(metrics.empty());

        for(auto& metric : metrics)
        {
            auto query = aql::get_query_info(agent->id, metric);
            for(unsigned block_index = 0; block_index < query.instance_count; ++block_index)
            {
                aqlprofile_pmc_event_t event = {
                    .block_index = block_index,
                    .event_id    = static_cast<uint32_t>(std::atoi(metric.event().c_str())),
                    .flags       = aqlprofile_pmc_event_flags_t{0},
                    .block_name  = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(query.id)};
                auto max_block_counters = aql::get_block_counters(agent->id, event);
                EXPECT_GT(max_block_counters, 0);
            }
        }
    }
}

TEST(aql_helpers, get_block_counters_compare_v1)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();
    auto agents = agent::get_agents();

    ASSERT_FALSE(agents.empty());

    for(auto agent : agents)
    {
        if(agent->type == ROCPROFILER_AGENT_TYPE_CPU) continue;
        auto metrics = findDeviceMetrics(*agent, {});
        ASSERT_FALSE(metrics.empty());

        for(auto& metric : metrics)
        {
            auto query = aql::get_query_info(agent->id, metric);
            for(unsigned block_index = 0; block_index < query.instance_count; ++block_index)
            {
                aqlprofile_pmc_event_t event = {
                    .block_index = block_index,
                    .event_id    = static_cast<uint32_t>(std::atoi(metric.event().c_str())),
                    .flags       = aqlprofile_pmc_event_flags_t{0},
                    .block_name  = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(query.id)};

                hsa_ven_amd_aqlprofile_event_t event_v1 = {
                    .block_name  = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(query.id),
                    .block_index = block_index,
                    .counter_id  = static_cast<uint32_t>(std::atoi(metric.event().c_str()))};
                EXPECT_EQ(aql::get_block_counters(agent->id, event),
                          v1_get_block_counters(agent::get_agent_cache(agent)->get_hsa_agent(),
                                                event_v1));
            }
        }
    }
    hsa_shut_down();
}

TEST(aql_helpers, get_dim_info)
{
    auto agents = agent::get_agents();
    ASSERT_FALSE(agents.empty());

    for(auto agent : agents)
    {
        if(agent->type == ROCPROFILER_AGENT_TYPE_CPU) continue;
        auto metrics = findDeviceMetrics(*agent, {});
        ASSERT_FALSE(metrics.empty());

        for(auto& metric : metrics)
        {
            auto query = aql::get_query_info(agent->id, metric);
            for(unsigned block_index = 0; block_index < query.instance_count; ++block_index)
            {
                aqlprofile_pmc_event_t event = {
                    .block_index = block_index,
                    .event_id    = static_cast<uint32_t>(std::atoi(metric.event().c_str())),
                    .flags       = aqlprofile_pmc_event_flags_t{0},
                    .block_name  = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(query.id)};
                std::map<int, uint64_t> dims;
                EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, aql::get_dim_info(agent->id, event, 0, dims));
                EXPECT_GT(dims.size(), 0);
            }
        }
    }
}

TEST(aql_helpers, get_dim_info_compare_v1)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();
    auto agents = agent::get_agents();

    ASSERT_FALSE(agents.empty());

    for(auto agent : agents)
    {
        if(agent->type == ROCPROFILER_AGENT_TYPE_CPU) continue;
        auto metrics = findDeviceMetrics(*agent, {});
        ASSERT_FALSE(metrics.empty());

        for(auto& metric : metrics)
        {
            std::map<int, uint64_t> dims;
            std::map<int, uint64_t> dims_v1;
            auto                    query = aql::get_query_info(agent->id, metric);
            for(unsigned block_index = 0; block_index < query.instance_count; ++block_index)
            {
                aqlprofile_pmc_event_t event = {
                    .block_index = block_index,
                    .event_id    = static_cast<uint32_t>(std::atoi(metric.event().c_str())),
                    .flags       = aqlprofile_pmc_event_flags_t{0},
                    .block_name  = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(query.id)};

                hsa_ven_amd_aqlprofile_event_t event_v1 = {
                    .block_name  = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(query.id),
                    .block_index = block_index,
                    .counter_id  = static_cast<uint32_t>(std::atoi(metric.event().c_str()))};
                EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, aql::get_dim_info(agent->id, event, 0, dims));
                EXPECT_EQ(
                    ROCPROFILER_STATUS_SUCCESS,
                    v1_get_dim_info(
                        agent::get_agent_cache(agent)->get_hsa_agent(), event_v1, 0, dims_v1));
                EXPECT_EQ(dims.size(), dims_v1.size());
                EXPECT_EQ(dims, dims_v1);
            }
        }
    }
    hsa_shut_down();
}
