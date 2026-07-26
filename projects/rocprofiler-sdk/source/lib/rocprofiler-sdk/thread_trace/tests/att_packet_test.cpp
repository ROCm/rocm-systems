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
#include "lib/rocprofiler-sdk/counters/tests/metrics_test_helpers.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"
#include "lib/rocprofiler-sdk/thread_trace/shared_trace_resources.hpp"

#include <gtest/gtest.h>
#include "lib/common/logging.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <thread>
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

rocprofiler_status_t
get_sq_waves_counter(rocprofiler_agent_id_t /* id */,
                     rocprofiler_counter_id_t* counters,
                     size_t                    num_counters,
                     void*                     userdata)
{
    for(size_t i = 0; i < num_counters; ++i)
    {
        auto _info = rocprofiler_counter_info_v1_t{};
        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(counters[i], ROCPROFILER_COUNTER_INFO_VERSION_1, &_info),
            "query counter");

        if(_info.name && std::string_view(_info.name).find("SQ_WAVES") == 0)
        {
            static_cast<rocprofiler_thread_trace_parameter_t*>(userdata)->counter_id = counters[i];
            return ROCPROFILER_STATUS_SUCCESS;
        }
    }
    return ROCPROFILER_STATUS_ERROR;
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
        thread_trace::register_shared_trace_requirements(agent.get_rocp_agent()->id,
                                                         agent.get_hsa_agent(),
                                                         params.buffer_size,
                                                         params.num_buffers);
        auto resources = thread_trace::acquire_shared_trace_resources(agent);

        aql::ThreadTraceAQLPacketFactory factory(
            agent, params, get_api_table(), get_ext_table(), resources);

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
    thread_trace::free_shared_trace_resources();
}

// Shared buffers are reused across contexts (same slot -> same pointer) and
// distinct per ring slot.
TEST(thread_trace, shared_buffer_reuse)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);

    // Isolate from any prior test that may have populated the manager.
    thread_trace::free_shared_trace_resources();

    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);

    const auto& agent = begin(agents)->second;

    // Build a TraceMemoryPool the same way ThreadTraceAQLPacketFactory does.
    auto make_pool = [&agent](const thread_trace::agent_trace_resources_ptr_t& resources) {
        hsa::TraceMemoryPool pool{};
        pool.allocate_fn     = get_ext_table().hsa_amd_memory_pool_allocate_fn;
        pool.allow_access_fn = get_ext_table().hsa_amd_agents_allow_access_fn;
        pool.free_fn         = get_ext_table().hsa_amd_memory_pool_free_fn;
        pool.gpu_agent       = agent.get_hsa_agent();
        pool.gpu_pool_       = agent.gpu_pool();
        pool.resources       = resources;
        return pool;
    };

    // Two contexts with different sizes; shared buffer is sized to the max.
    thread_trace::register_shared_trace_requirements(
        agent.get_rocp_agent()->id, agent.get_hsa_agent(), 0x1000000, 1);
    thread_trace::register_shared_trace_requirements(
        agent.get_rocp_agent()->id, agent.get_hsa_agent(), 0x2000000, 1);
    auto resources      = thread_trace::acquire_shared_trace_resources(agent);
    auto same_resources = thread_trace::acquire_shared_trace_resources(agent);
    EXPECT_EQ(resources.get(), same_resources.get());

    auto pool_a = make_pool(resources);
    auto pool_b = make_pool(resources);

    constexpr uint64_t kSize = 0x2000000;

    // First context requests two ring slots (e.g. triple buffering).
    void* a0 = pool_a.allocate_output(kSize);
    void* a1 = pool_a.allocate_output(kSize);
    ASSERT_NE(a0, nullptr);
    ASSERT_NE(a1, nullptr);
    EXPECT_NE(a0, a1) << "distinct ring slots must be distinct buffers";

    // Second context reuses the same physical buffers per slot.
    void* b0 = pool_b.allocate_output(kSize);
    void* b1 = pool_b.allocate_output(kSize);
    EXPECT_EQ(a0, b0) << "same ring slot must be shared across contexts";
    EXPECT_EQ(a1, b1) << "same ring slot must be shared across contexts";

    EXPECT_TRUE(resources->owns_output_buffer(a0));
    EXPECT_TRUE(resources->owns_output_buffer(a1));
    EXPECT_FALSE(resources->owns_output_buffer(nullptr));

    // Arbitrary nonzero ID: verifies nested traces from the same context share ownership.
    constexpr auto test_context_id = rocprofiler_context_id_t{42};
    resources->begin_trace(test_context_id);
    resources->begin_trace(test_context_id);
    resources->end_trace(test_context_id);
    resources->end_trace(test_context_id);

    pool_a.resources.reset();
    pool_b.resources.reset();
    same_resources.reset();
    resources.reset();
    thread_trace::free_shared_trace_resources();
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
    params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTERS_CTRL, {0}});
    params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTER_EXCLUDE_MASK, {0}});
    params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_NO_DETAIL, {0}});

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
            [](rocprofiler_thread_trace_shader_data_t, rocprofiler_user_data_t) {},
            nullptr);
    }

    ROCPROFILER_CALL(rocprofiler_start_context(ctx), "context start failed");
    ROCPROFILER_CALL(rocprofiler_stop_context(ctx), "context stop failed");
    context::pop_client(1);
}

TEST(thread_trace, disjoint_dispatch_contexts_own_independent_callbacks)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);

    auto agents = hsa::get_queue_controller()->get_supported_agents();
    if(agents.size() < 2) GTEST_SKIP() << "requires two GPU agents";
    context::push_client(1);

    auto       itr     = agents.begin();
    const auto agent_a = (itr++)->second.get_rocp_agent()->id;
    const auto agent_b = itr->second.get_rocp_agent()->id;

    auto dispatch_callback = [](rocprofiler_agent_id_t,
                                rocprofiler_queue_id_t,
                                rocprofiler_async_correlation_id_t,
                                rocprofiler_kernel_id_t,
                                rocprofiler_dispatch_id_t,
                                void*,
                                rocprofiler_user_data_t*) {
        return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;
    };
    auto shader_callback = [](rocprofiler_thread_trace_shader_data_t, rocprofiler_user_data_t) {};

    rocprofiler_context_id_t context_a{0};
    rocprofiler_context_id_t context_b{0};
    ROCPROFILER_CALL(rocprofiler_create_context(&context_a), "context A creation");
    ROCPROFILER_CALL(rocprofiler_create_context(&context_b), "context B creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_dispatch_thread_trace_service(
            context_a, agent_a, nullptr, 0, dispatch_callback, shader_callback, nullptr),
        "context A ATT configuration");
    ROCPROFILER_CALL(
        rocprofiler_configure_dispatch_thread_trace_service(
            context_b, agent_b, nullptr, 0, dispatch_callback, shader_callback, nullptr),
        "context B ATT configuration");

    auto callback_count = []() {
        size_t count = 0;
        hsa::get_queue_controller()->iterate_callbacks([&](auto, const auto&) { ++count; });
        return count;
    };

    const auto initial_callbacks = callback_count();
    EXPECT_EQ(rocprofiler_start_context(context_a), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_start_context(context_b), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(callback_count(), initial_callbacks + 2);
    EXPECT_TRUE(hsa::get_queue_controller()->serialization_enabled());

    EXPECT_EQ(rocprofiler_stop_context(context_a), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(callback_count(), initial_callbacks + 1);
    EXPECT_TRUE(hsa::get_queue_controller()->serialization_enabled());
    EXPECT_EQ(rocprofiler_stop_context(context_b), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(callback_count(), initial_callbacks);
    EXPECT_FALSE(hsa::get_queue_controller()->serialization_enabled());

    context::pop_client(1);
}

TEST(thread_trace, concurrent_dispatch_stops_release_serialization_once)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto callback_count = []() {
        size_t count = 0;
        hsa::get_queue_controller()->iterate_callbacks([&](auto, const auto&) { ++count; });
        return count;
    };

    thread_trace::DispatchThreadTracer tracer{};
    const auto                         initial_callbacks = callback_count();
    tracer.start_context();
    EXPECT_EQ(callback_count(), initial_callbacks + 1);
    EXPECT_TRUE(hsa::get_queue_controller()->serialization_enabled());

    auto stop_a = std::thread{[&]() { tracer.stop_context(); }};
    auto stop_b = std::thread{[&]() { tracer.stop_context(); }};
    stop_a.join();
    stop_b.join();

    EXPECT_EQ(callback_count(), initial_callbacks);
    EXPECT_FALSE(hsa::get_queue_controller()->serialization_enabled());
}

TEST(thread_trace, perfcounters_configure_test)
{
    constexpr int NUM_COUNTERS = 3;
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    rocprofiler_context_id_t ctx{0};
    ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");

    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);

    for(auto& [_, agent] : agents)
    {
        auto params = std::vector<rocprofiler_thread_trace_parameter_t>{};
        params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTERS_CTRL, {1}});

        auto sq_waves = rocprofiler_thread_trace_parameter_t{};
        sq_waves.type = ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTER;
        ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(
                             agent.get_rocp_agent()->id, get_sq_waves_counter, &sq_waves),
                         "iterate counters");

        for(int i = 0; i < NUM_COUNTERS; i++)
        {
            sq_waves.simd_mask = 1 << i;
            params.emplace_back(sq_waves);
        }

        ROCPROFILER_CALL(
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
                [](rocprofiler_thread_trace_shader_data_t, rocprofiler_user_data_t) {},
                nullptr),
            "configure");
    }

    auto* context = rocprofiler::context::get_mutable_registered_context(ctx);
    auto* tracer  = context->dispatch_thread_trace.get();

    ASSERT_NE(tracer, nullptr);
    for(auto& [id, agent] : tracer->get_agents())
    {
        // We expect perfcounters.size() to match the number of counters we added
        ASSERT_EQ(agent->params.perfcounter_ctrl, 1);
        ASSERT_EQ(agent->params.perfcounters.size(), NUM_COUNTERS);
        for(const auto& param : agent->params.perfcounters)
        {
            // We expect a nonzero event id (.first) and nonzero simd mask (.second)
            EXPECT_TRUE(param.first != 0);
            EXPECT_TRUE(param.second != 0)
                << "valid AQLprofile mask not generated for perfcounters";
        }
    }
    context::pop_client(1);
}

TEST(thread_trace, perfcounters_configure_fail_test)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    rocprofiler_context_id_t ctx{0};
    ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");

    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);

    for(auto& [_, agent] : agents)
    {
        auto params = std::vector<rocprofiler_thread_trace_parameter_t>{};
        params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTERS_CTRL, {1}});

        auto sq_waves = rocprofiler_thread_trace_parameter_t{};
        sq_waves.type = ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTER;
        // We are not initializing the counter, so we expect the configuration to fail
        params.emplace_back(sq_waves);

        auto status = rocprofiler_configure_dispatch_thread_trace_service(
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
            [](rocprofiler_thread_trace_shader_data_t, rocprofiler_user_data_t) {},
            nullptr);

        EXPECT_NE(status, ROCPROFILER_STATUS_SUCCESS);
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
    const auto& agent        = begin(agents)->second;
    thread_trace::register_shared_trace_requirements(agent.get_rocp_agent()->id,
                                                     agent.get_hsa_agent(),
                                                     _params.buffer_size,
                                                     _params.num_buffers);
    auto new_tracer =
        std::make_unique<thread_trace::ThreadTracerAgent>(_params, agent.get_rocp_agent()->id);

    ASSERT_EQ(new_tracer->factory->aql_params.size(),
              sqtt_default_num_options + perf_counters.size());
    new_tracer.reset();
    thread_trace::free_shared_trace_resources();
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
        params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTERS_CTRL, {0}});
        params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTER_EXCLUDE_MASK, {0}});
        params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_NO_DETAIL, {0}});

        {
            auto metrics = rocprofiler::counters::getMetricsForAgent(agent);

            rocprofiler_thread_trace_parameter_t att_param;
            att_param.type      = ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTER;
            att_param.simd_mask = 0xF;
            for(auto& metric : metrics)
                if(metric.name() == "SQ_WAVES")
                    att_param.counter_id = rocprofiler_counter_id_t{.handle = metric.id()};

            params.push_back(att_param);
        }

        rocprofiler_configure_device_thread_trace_service(
            *reinterpret_cast<rocprofiler_context_id_t*>(ctx_ptr),
            agent->id,
            params.data(),
            params.size(),
            [](rocprofiler_thread_trace_shader_data_t, rocprofiler_user_data_t) {},
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

    context::pop_client(1);
}

TEST(thread_trace, triple_buffer_multiple_shader)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    rocprofiler_context_id_t ctx{0};
    ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");

    auto configure_agents = [](rocprofiler_agent_version_t /* version */,
                               const void** agents,
                               size_t       num_agents,
                               void*        ctx_ptr) {
        for(size_t idx = 0; idx < num_agents; idx++)
        {
            const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[idx]);
            if(agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;

            auto parameters = std::vector<rocprofiler_thread_trace_parameter_t>{};
            parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_NUM_BUFFERS, {3}});
            parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {0x3}});

            auto status = rocprofiler_configure_device_thread_trace_service(
                *reinterpret_cast<rocprofiler_context_id_t*>(ctx_ptr),
                agent->id,
                parameters.data(),
                parameters.size(),
                [](rocprofiler_thread_trace_shader_data_t, rocprofiler_user_data_t) {},
                rocprofiler_user_data_t{});

            return status;
        }
        return ROCPROFILER_STATUS_ERROR;
    };

    auto status = rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0, configure_agents, sizeof(rocprofiler_agent_t), &ctx);
    ASSERT_EQ(status, ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);

    context::pop_client(1);
}

TEST(thread_trace, triple_buffer_dispatch_mode)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    rocprofiler_context_id_t ctx{0};
    ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");

    auto configure_agents = [](rocprofiler_agent_version_t /* version */,
                               const void** agents,
                               size_t       num_agents,
                               void*        ctx_ptr) {
        for(size_t idx = 0; idx < num_agents; idx++)
        {
            const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[idx]);
            if(agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;

            auto parameters = std::vector<rocprofiler_thread_trace_parameter_t>{};
            parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_NUM_BUFFERS, {3}});

            auto status = rocprofiler_configure_dispatch_thread_trace_service(
                *reinterpret_cast<rocprofiler_context_id_t*>(ctx_ptr),
                agent->id,
                parameters.data(),
                parameters.size(),
                [](rocprofiler_agent_id_t,
                   rocprofiler_queue_id_t,
                   rocprofiler_async_correlation_id_t,
                   rocprofiler_kernel_id_t,
                   rocprofiler_dispatch_id_t,
                   void*,
                   rocprofiler_user_data_t*) { return ROCPROFILER_THREAD_TRACE_CONTROL_NONE; },
                [](rocprofiler_thread_trace_shader_data_t, rocprofiler_user_data_t) {},
                nullptr);

            return status;
        }
        return ROCPROFILER_STATUS_ERROR;
    };

    auto status = rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0, configure_agents, sizeof(rocprofiler_agent_t), &ctx);
    ASSERT_EQ(status, ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);

    context::pop_client(1);
}
