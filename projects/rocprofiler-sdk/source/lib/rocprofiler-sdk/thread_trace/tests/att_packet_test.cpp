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

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/thread_trace/att_core.hpp"

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
        agent::construct_agent_cache(&table);
        hsa::get_queue_controller()->init(get_api_table(), get_ext_table());
        return true;
    };
    [[maybe_unused]] static bool run_ince = init();
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
        thread_trace::DispatchThreadTracer        tracer(std::move(params));

        for(const auto& [_, agent] : agents)
        {
            // Init twice to simulate two queues
            tracer.resource_init(agent, get_api_table(), get_ext_table());
            tracer.resource_init(agent, get_api_table(), get_ext_table());
        }

        for(auto& [_, agenttracer] : tracer.agents)
        {
            agenttracer->load_codeobj(1, 0x1000, 0x1000);
            agenttracer->load_codeobj(2, 0x3000, 0x1000);
            agenttracer->unload_codeobj(1);
        }

        for(const auto& [_, agent] : agents)
        {
            // Deinit twice to remove both queues
            tracer.resource_deinit(agent);
            tracer.resource_deinit(agent);
        }
    }
    hsa_shut_down();
}

TEST(thread_trace, configure_test)
{
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    rocprofiler_context_id_t ctx{0};
    ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");

    std::vector<rocprofiler_att_parameter_t> params;
    params.push_back({ROCPROFILER_ATT_PARAMETER_TARGET_CU, {1}});
    params.push_back({ROCPROFILER_ATT_PARAMETER_SHADER_ENGINE_MASK, {0xF}});
    params.push_back({ROCPROFILER_ATT_PARAMETER_BUFFER_SIZE, {0x1000000}});
    params.push_back({ROCPROFILER_ATT_PARAMETER_SIMD_SELECT, {0xF}});

    rocprofiler_configure_dispatch_thread_trace_service(
        ctx,
        params.data(),
        params.size(),
        [](rocprofiler_queue_id_t,
           const rocprofiler_agent_t*,
           rocprofiler_correlation_id_t,
           rocprofiler_kernel_id_t,
           rocprofiler_dispatch_id_t,
           rocprofiler_user_data_t*,
           void*) { return ROCPROFILER_ATT_CONTROL_NONE; },
        [](int64_t, void*, size_t, rocprofiler_user_data_t) {},
        nullptr);

    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    ROCPROFILER_CALL(rocprofiler_start_context(ctx), "context start failed");
    ROCPROFILER_CALL(rocprofiler_stop_context(ctx), "context stop failed");
    context::pop_client(1);
    hsa_shut_down();
}

TEST(thread_trace, perfcounters_configure_test)
{
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    rocprofiler_context_id_t ctx{0};
    ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");

    // Only GFX9 SQ Block counters are supported
    std::vector<std::pair<std::string, uint64_t>> perf_counters = {
        {"SQ_WAVES", 0x1}, {"SQ_WAVES", 0x2}, {"SQ_WAVES", 0x2}, {"GRBM_COUNT", 0x3}};
    std::set<std::pair<uint32_t, uint32_t>>  expected;
    std::vector<rocprofiler_att_parameter_t> params;
    params.push_back({ROCPROFILER_ATT_PARAMETER_PERFCOUNTERS_CTRL, {1}});
    auto metrics = rocprofiler::counters::getMetricsForAgent("gfx90a");

    for(auto& [counter_name, simd_mask] : perf_counters)
        for(auto& metric : metrics)
            if(metric.name() == counter_name)
            {
                rocprofiler_att_parameter_t att_param;
                att_param.type       = ROCPROFILER_ATT_PARAMETER_PERFCOUNTER;
                att_param.counter_id = rocprofiler_counter_id_t{.handle = metric.id()};
                att_param.simd_mask  = simd_mask;
                params.push_back(att_param);
                expected.insert({std::atoi(metric.event().c_str()), simd_mask});
            }

    rocprofiler_configure_dispatch_thread_trace_service(
        ctx,
        params.data(),
        params.size(),
        [](rocprofiler_queue_id_t,
           const rocprofiler_agent_t*,
           rocprofiler_correlation_id_t,
           rocprofiler_kernel_id_t,
           rocprofiler_dispatch_id_t,
           rocprofiler_user_data_t*,
           void*) { return ROCPROFILER_ATT_CONTROL_NONE; },
        [](int64_t, void*, size_t, rocprofiler_user_data_t) {},
        nullptr);

    auto* context = rocprofiler::context::get_mutable_registered_context(ctx);
    auto* tracer  = context->dispatch_thread_trace.get();

    ASSERT_NE(tracer, nullptr);
    ASSERT_EQ(tracer->params.perfcounter_ctrl, 1);
    ASSERT_EQ(tracer->params.perfcounters.size(), 3);
    for(const auto& param : tracer->params.perfcounters)
        EXPECT_TRUE(expected.find(param) != expected.end())
            << "valid AQLprofile mask not generated for perfcounters";
    context::pop_client(1);
    hsa_shut_down();
}

TEST(thread_trace, perfcounters_aql_options_test)
{
    hsa_init();
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
        _params, begin(agents)->second, get_api_table(), get_ext_table());

    ASSERT_EQ(new_tracer->factory->aql_params.size(),
              sqtt_default_num_options + perf_counters.size());
    context::pop_client(1);
    hsa_shut_down();
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

        std::vector<rocprofiler_att_parameter_t> params;
        params.push_back({ROCPROFILER_ATT_PARAMETER_TARGET_CU, {1}});
        params.push_back({ROCPROFILER_ATT_PARAMETER_SHADER_ENGINE_MASK, {0xF}});
        params.push_back({ROCPROFILER_ATT_PARAMETER_BUFFER_SIZE, {0x1000000}});
        params.push_back({ROCPROFILER_ATT_PARAMETER_SIMD_SELECT, {0xF}});
        params.push_back({ROCPROFILER_ATT_PARAMETER_PERFCOUNTERS_CTRL, {1}});

        {
            auto metrics = rocprofiler::counters::getMetricsForAgent("gfx90a");

            rocprofiler_att_parameter_t att_param;
            att_param.type      = ROCPROFILER_ATT_PARAMETER_PERFCOUNTER;
            att_param.simd_mask = 0xF;
            for(auto& metric : metrics)
                if(metric.name() == "SQ_WAVES") rocprofiler_counter_id_t{.handle = metric.id()};

            params.push_back(att_param);
        }

        rocprofiler_configure_agent_thread_trace_service(
            *reinterpret_cast<rocprofiler_context_id_t*>(ctx_ptr),
            params.data(),
            params.size(),
            agent->id,
            [](int64_t, void*, size_t, rocprofiler_user_data_t) {},
            nullptr);
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

TEST(thread_trace, agent_configure_test)
{
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
