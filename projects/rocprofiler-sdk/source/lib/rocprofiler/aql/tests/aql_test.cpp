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
    return {.hsa_amd_memory_pool_get_info_fn       = hsa_amd_memory_pool_get_info,
            .hsa_amd_agent_iterate_memory_pools_fn = hsa_amd_agent_iterate_memory_pools,
            .hsa_amd_memory_pool_allocate_fn       = hsa_amd_memory_pool_allocate,
            .hsa_amd_memory_pool_free_fn           = hsa_amd_memory_pool_free,
            .hsa_amd_agent_memory_pool_get_info_fn = hsa_amd_agent_memory_pool_get_info,
            .hsa_amd_agents_allow_access_fn        = hsa_amd_agents_allow_access};
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
