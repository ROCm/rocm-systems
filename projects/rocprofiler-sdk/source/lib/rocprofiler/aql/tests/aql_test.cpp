// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <functional>
#include <map>
#include <unordered_set>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

#include "lib/rocprofiler/aql/helpers.hpp"
#include "lib/rocprofiler/aql/packet_construct.hpp"
#include "lib/rocprofiler/counters/metrics.hpp"
#include "lib/rocprofiler/hsa/agent_cache.hpp"
#include "lib/rocprofiler/hsa/queue.hpp"
#include "lib/rocprofiler/hsa/queue_controller.hpp"

namespace rocprofiler
{
AmdExtTable
get_ext_table()
{
    static auto _v = []() {
        auto val                                  = AmdExtTable{};
        val.hsa_amd_memory_pool_get_info_fn       = hsa_amd_memory_pool_get_info;
        val.hsa_amd_agent_iterate_memory_pools_fn = hsa_amd_agent_iterate_memory_pools;
        val.hsa_amd_memory_pool_allocate_fn       = hsa_amd_memory_pool_allocate;
        val.hsa_amd_memory_pool_free_fn           = hsa_amd_memory_pool_free;
        val.hsa_amd_agent_memory_pool_get_info_fn = hsa_amd_agent_memory_pool_get_info;
        val.hsa_amd_agents_allow_access_fn        = hsa_amd_agents_allow_access;
        return val;
    }();
    return _v;
}

auto
findDeviceMetrics(const hsa::AgentCache& agent, const std::unordered_set<std::string>& metrics)
{
    std::vector<counters::Metric> ret;
    auto                          all_counters = counters::getBaseHardwareMetrics();

    auto gfx_metrics = common::get_val(all_counters, std::string(agent.name()));
    if(!gfx_metrics) return ret;

    for(auto& counter : *gfx_metrics)
    {
        if(metrics.count(counter.name()) > 0 || metrics.empty())
        {
            ret.push_back(counter);
        }
    }
    return ret;
}

}  // namespace rocprofiler

using namespace rocprofiler::aql;

TEST(aql_profile, construct_packets)
{
    hsa_init();
    try
    {
        auto agents = rocprofiler::hsa::get_queue_controller().get_supported_agents();
        for(const auto& [_, agent] : agents)
        {
            LOG(WARNING) << fmt::format("Found Agent: {}", agent.get_hsa_agent().handle);
            auto metrics = rocprofiler::findDeviceMetrics(agent, {"SQ_WAVES"});
            ASSERT_EQ(metrics.size(), 1);
            AQLPacketConstruct(agent, metrics);
        }
    } catch(std::runtime_error&)
    {
        LOG(WARNING) << "Could not fetch agents on host, skipping test";
        return;
    }
    hsa_shut_down();
}

TEST(aql_profile, too_many_counters)
{
    hsa_init();
    try
    {
        auto agents = rocprofiler::hsa::get_queue_controller().get_supported_agents();

        for(const auto& [_, agent] : agents)
        {
            LOG(WARNING) << fmt::format("Found Agent: {}", agent.get_hsa_agent().handle);

            auto metrics = rocprofiler::findDeviceMetrics(agent, {});
            EXPECT_THROW(
                {
                    try
                    {
                        AQLPacketConstruct(agent, metrics);
                    } catch(const std::exception& e)
                    {
                        EXPECT_NE(e.what(), nullptr) << e.what();
                        throw;
                    }
                },
                std::runtime_error);
        }
    } catch(std::runtime_error&)
    {
        LOG(WARNING) << "Could not fetch agents on host, skipping test";
        return;
    }
    hsa_shut_down();
}

TEST(aql_profile, packet_generation_single)
{
    hsa_init();
    try
    {
        auto agents = rocprofiler::hsa::get_queue_controller().get_supported_agents();
        for(const auto& [_, agent] : agents)
        {
            auto               metrics = rocprofiler::findDeviceMetrics(agent, {"SQ_WAVES"});
            AQLPacketConstruct pkt(agent, metrics);
            auto               test_pkt = pkt.construct_packet(rocprofiler::get_ext_table());
            EXPECT_TRUE(test_pkt);
        }
    } catch(std::runtime_error&)
    {
        LOG(WARNING) << "Could not fetch agents on host, skipping test";
        return;
    }

    hsa_shut_down();
}

TEST(aql_profile, packet_generation_multi)
{
    hsa_init();
    try
    {
        auto agents = rocprofiler::hsa::get_queue_controller().get_supported_agents();
        for(const auto& [_, agent] : agents)
        {
            auto metrics =
                rocprofiler::findDeviceMetrics(agent, {"SQ_WAVES", "TA_FLAT_READ_WAVEFRONTS"});
            AQLPacketConstruct pkt(agent, metrics);
            auto               test_pkt = pkt.construct_packet(rocprofiler::get_ext_table());
            EXPECT_TRUE(test_pkt);
        }
    } catch(std::runtime_error&)
    {
        LOG(WARNING) << "Could not fetch agents on host, skipping test";
        return;
    }
    hsa_shut_down();
}
