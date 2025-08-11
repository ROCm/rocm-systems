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
#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

using namespace rocprofiler::counters::test_constants;

#define ROCPROFILER_CALL(ARG, MSG)                                                                 \
    {                                                                                              \
        auto _status = (ARG);                                                                      \
        EXPECT_EQ(_status, ROCPROFILER_STATUS_SUCCESS) << MSG << " :: " << #ARG;                   \
    }

namespace rocprofiler
{
void
test_init()
{
    auto init = []() -> bool {
        HsaApiTable table;
        table.amd_ext_ = &get_ext_table();
        table.core_    = &get_api_table();
        rocprofiler::hsa::copy_table(table.core_, 0);
        rocprofiler::hsa::copy_table(table.amd_ext_, 0);
        agent::construct_agent_cache(&table);
        hsa::get_queue_controller()->init(get_api_table(), get_ext_table());
        return true;
    };
    [[maybe_unused]] static bool run_once = init();
}

}  // namespace rocprofiler

using namespace rocprofiler;

TEST(thread_trace, resource_creation)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);

    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);

    for(const auto& [_, agent] : agents)
    {
        auto params = thread_trace::thread_trace_parameter_pack{};

        aql::ThreadTraceAQLPacketFactory factory(agent, params, get_api_table(), get_ext_table());

        auto packet = factory.construct_control_packet();
        packet->populate_before();
        packet->populate_after();

        ASSERT_TRUE(packet->before_krn_pkt.size() > 0);
        ASSERT_TRUE(packet->after_krn_pkt.size() > 0);
    }

    {
        thread_trace::thread_trace_parameter_pack params{};
        thread_trace::DispatchThreadTracer        tracer{};

        for(const auto& [_, agent] : agents)
            tracer.add_agent(agent.get_rocp_agent()->id, params);

        tracer.resource_init();

        for(auto& [_, agenttracer] : tracer.get_agents())
        {
            agenttracer->load_codeobj(1, 0x1000, 0x1000);
            agenttracer->load_codeobj(2, 0x3000, 0x1000);
            agenttracer->unload_codeobj(1);
        }

        tracer.resource_deinit();
    }
}

TEST(thread_trace, configure_test)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    rocprofiler_context_id_t ctx{0};
    ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");

    std::vector<rocprofiler_thread_trace_parameter_t> params;
    params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, {1}});
    params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {0xF}});
    params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, {0x1000000}});
    params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, {0xF}});

    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);

    for(auto& [_, agent] : agents)
    {
        rocprofiler_configure_dispatch_thread_trace_service(
            ctx,
            agent.get_rocp_agent()->id,
            params.data(),
            params.size(),
            [](rocprofiler_agent_id_t,
               rocprofiler_queue_id_t,
               rocprofiler_async_correlation_id_t,
               rocprofiler_kernel_id_t,
               rocprofiler_dispatch_id_t,
               void*,
               rocprofiler_user_data_t*) { return ROCPROFILER_THREAD_TRACE_CONTROL_NONE; },
            [](rocprofiler_agent_id_t, int64_t, void*, size_t, rocprofiler_user_data_t) {},
            nullptr);
    }

    ROCPROFILER_CALL(rocprofiler_start_context(ctx), "context start failed");
    ROCPROFILER_CALL(rocprofiler_stop_context(ctx), "context stop failed");
    context::pop_client(1);
}

TEST(thread_trace, perfcounters_configure_test)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    rocprofiler_context_id_t ctx{0};
    ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");

    // Only GFX9 SQ Block counters are supported
    std::vector<std::pair<std::string, uint64_t>> perf_counters = {
        {"SQ_WAVES", 0x1}, {"SQ_WAVES", 0x2}, {"SQ_WAVES", 0x2}, {"GRBM_COUNT", 0x3}};
    std::set<std::pair<uint32_t, uint32_t>>           expected;
    std::vector<rocprofiler_thread_trace_parameter_t> params;
    params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTERS_CTRL, {1}});
    auto metrics = rocprofiler::counters::getMetricsForAgent("gfx90a");

    for(auto& [counter_name, simd_mask] : perf_counters)
        for(auto& metric : metrics)
            if(metric.name() == counter_name)
            {
                rocprofiler_thread_trace_parameter_t att_param;
                att_param.type       = ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTER;
                att_param.counter_id = rocprofiler_counter_id_t{.handle = metric.id()};
                att_param.simd_mask  = simd_mask;
                params.push_back(att_param);
                expected.insert({std::atoi(metric.event().c_str()), simd_mask});
            }

    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);

    for(auto& [_, agent] : agents)
    {
        rocprofiler_configure_dispatch_thread_trace_service(
            ctx,
            agent.get_rocp_agent()->id,
            params.data(),
            params.size(),
            [](rocprofiler_agent_id_t,
               rocprofiler_queue_id_t,
               rocprofiler_async_correlation_id_t,
               rocprofiler_kernel_id_t,
               rocprofiler_dispatch_id_t,
               void*,
               rocprofiler_user_data_t*) { return ROCPROFILER_THREAD_TRACE_CONTROL_NONE; },
            [](rocprofiler_agent_id_t, int64_t, void*, size_t, rocprofiler_user_data_t) {},
            nullptr);
    }

    auto* context = rocprofiler::context::get_mutable_registered_context(ctx);
    auto* tracer  = context->dispatch_thread_trace.get();

    ASSERT_NE(tracer, nullptr);
    for(auto& [id, agent] : tracer->get_agents())
    {
        ASSERT_EQ(agent->params.perfcounter_ctrl, 1);
        ASSERT_EQ(agent->params.perfcounters.size(), 3);
        for(const auto& param : agent->params.perfcounters)
            EXPECT_TRUE(expected.find(param) != expected.end())
                << "valid AQLprofile mask not generated for perfcounters";
    }
    context::pop_client(1);
}

TEST(thread_trace, perfcounters_aql_options_test)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    const std::uint8_t sqtt_default_num_options = 5;
    auto               agents = hsa::get_queue_controller()->get_supported_agents();

    thread_trace::thread_trace_parameter_pack _params = {};
    auto metrics = rocprofiler::counters::getMetricsForAgent("gfx90a");
    std::vector<std::pair<std::string, uint64_t>> perf_counters = {
        {"SQ_WAVES", 0x1}, {"SQ_WAVES", 0x2}, {"GRBM_COUNT", 0x3}};
    for(auto& [counter_name, simd_mask] : perf_counters)
        for(auto& metric : metrics)
            if(metric.name() == counter_name)
                _params.perfcounters.push_back({std::atoi(metric.event().c_str()), simd_mask});
    _params.perfcounter_ctrl = 2;
    auto new_tracer          = std::make_unique<thread_trace::ThreadTracerQueue>(
        _params, begin(agents)->second.get_rocp_agent()->id);

    ASSERT_EQ(new_tracer->factory->aql_params.size(),
              sqtt_default_num_options + perf_counters.size());
    context::pop_client(1);
}

rocprofiler_status_t
query_available_agents(rocprofiler_agent_version_t /* version */,
                       const void** agents,
                       size_t       num_agents,
                       void*        ctx_ptr)
{
    for(size_t idx = 0; idx < num_agents; idx++)
    {
        const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[idx]);
        if(agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;

        std::vector<rocprofiler_thread_trace_parameter_t> params;
        params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, {1}});
        params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {0xF}});
        params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, {0x1000000}});
        params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, {0xF}});
        params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTERS_CTRL, {1}});

        {
            auto metrics = rocprofiler::counters::getMetricsForAgent("gfx90a");

            rocprofiler_thread_trace_parameter_t att_param;
            att_param.type      = ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTER;
            att_param.simd_mask = 0xF;
            for(auto& metric : metrics)
                if(metric.name() == "SQ_WAVES") rocprofiler_counter_id_t{.handle = metric.id()};

            params.push_back(att_param);
        }

        rocprofiler_configure_device_thread_trace_service(
            *reinterpret_cast<rocprofiler_context_id_t*>(ctx_ptr),
            agent->id,
            params.data(),
            params.size(),
            [](rocprofiler_agent_id_t, int64_t, void*, size_t, rocprofiler_user_data_t) {},
            rocprofiler_user_data_t{});
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

TEST(thread_trace, agent_configure_test)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    rocprofiler_context_id_t ctx{0};
    ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");

    ROCPROFILER_CALL(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                                        &query_available_agents,
                                                        sizeof(rocprofiler_agent_t),
                                                        &ctx),
                     "Failed to find GPU agents");
}
