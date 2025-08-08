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

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <unordered_set>

using namespace rocprofiler::counters::test_constants;

namespace rocprofiler
{
AmdExtTable&
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
        val.hsa_amd_memory_fill_fn                = hsa_amd_memory_fill;
        return val;
    }();
    return _v;
}

CoreApiTable&
get_api_table()
{
    static auto _v = []() {
        auto val                       = CoreApiTable{};
        val.hsa_iterate_agents_fn      = hsa_iterate_agents;
        val.hsa_agent_get_info_fn      = hsa_agent_get_info;
        val.hsa_queue_create_fn        = hsa_queue_create;
        val.hsa_queue_destroy_fn       = hsa_queue_destroy;
        val.hsa_signal_wait_relaxed_fn = hsa_signal_wait_relaxed;
        return val;
    }();
    return _v;
}

auto
findDeviceMetrics(const hsa::AgentCache& agent, const std::unordered_set<std::string>& metrics)
{
    std::vector<counters::Metric> ret;
    auto                          all_counters = counters::loadMetrics()->arch_to_metric;

    ROCP_INFO << "Looking up counters for " << std::string(agent.name());
    auto* gfx_metrics = common::get_val(all_counters, std::string(agent.name()));
    if(!gfx_metrics)
    {
        ROCP_ERROR << "No counters found for " << std::string(agent.name());
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

}  // namespace rocprofiler

using namespace rocprofiler::aql;

TEST(aql_profile, construct_packets)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    rocprofiler::test_init();
    auto agents = rocprofiler::hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);
    for(const auto& [_, agent] : agents)
    {
        ROCP_INFO << fmt::format("Found Agent: {}", agent.get_hsa_agent().handle);
        auto metrics = rocprofiler::findDeviceMetrics(agent, {"SQ_WAVES"});
        ASSERT_EQ(metrics.size(), 1);
        CounterPacketConstruct(agent.get_rocp_agent()->id, metrics);
    }
    hsa_shut_down();
}

TEST(aql_profile, too_many_counters)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    rocprofiler::test_init();
    auto agents = rocprofiler::hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);
    for(const auto& [_, agent] : agents)
    {
        ROCP_INFO << fmt::format("Found Agent: {}", agent.get_hsa_agent().handle);

        auto metrics = rocprofiler::findDeviceMetrics(agent, {});
        EXPECT_NE(CounterPacketConstruct(agent.get_rocp_agent()->id, metrics).can_collect(),
                  ROCPROFILER_STATUS_SUCCESS);
    }
    hsa_shut_down();
}

TEST(aql_profile, packet_generation_single)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    rocprofiler::test_init();

    auto agents = rocprofiler::hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);
    for(const auto& [_, agent] : agents)
    {
        auto                   metrics = rocprofiler::findDeviceMetrics(agent, {"SQ_WAVES"});
        CounterPacketConstruct pkt(agent.get_rocp_agent()->id, metrics);
        auto                   test_pkt =
            pkt.construct_packet(rocprofiler::get_api_table(), rocprofiler::get_ext_table());

        EXPECT_TRUE(test_pkt);
    }

    hsa_shut_down();
}

TEST(aql_profile, packet_generation_multi)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    rocprofiler::test_init();

    auto agents = rocprofiler::hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);
    for(const auto& [_, agent] : agents)
    {
        auto metrics =
            rocprofiler::findDeviceMetrics(agent, {"SQ_WAVES", "TA_FLAT_READ_WAVEFRONTS"});
        CounterPacketConstruct pkt(agent.get_rocp_agent()->id, metrics);
        auto                   test_pkt =
            pkt.construct_packet(rocprofiler::get_api_table(), rocprofiler::get_ext_table());
        EXPECT_TRUE(test_pkt);
    }

    hsa_shut_down();
}

/*
class TestAqlPacket : public rocprofiler::hsa::CounterAQLPacket
{
public:
    TestAqlPacket(bool mallocd)
    : rocprofiler::hsa::CounterAQLPacket([](void* x) -> hsa_status_t {
        ::free(x);
        return HSA_STATUS_SUCCESS;
    })
    {
        this->profile.output_buffer.ptr  = malloc(sizeof(double));
        this->profile.command_buffer.ptr = malloc(sizeof(double));
        this->command_buf_mallocd        = mallocd;
        this->output_buffer_malloced     = mallocd;
    }
};

TEST(aql_profile, test_aql_packet)
{
    auto check_null = [](auto& val) {
        hsa_ext_amd_aql_pm4_packet_t null_val = {
            .header = 0, .pm4_command = {0}, .completion_signal = {.handle = 0}};
        return val.header == null_val.header &&
               memcmp(val.pm4_command, null_val.pm4_command, sizeof(null_val.pm4_command)) == 0 &&
               val.completion_signal.handle == null_val.completion_signal.handle;
    };

    TestAqlPacket test_pkt(true);
    EXPECT_TRUE(check_null(test_pkt.start)) << "Start packet not null";
    EXPECT_TRUE(check_null(test_pkt.stop)) << "Stop packet not null";
    EXPECT_TRUE(check_null(test_pkt.read)) << "Read packet not null";

    // test custom destructor as well
    // Why is this valid?
    TestAqlPacket test_pkt2(false);
}
*/
