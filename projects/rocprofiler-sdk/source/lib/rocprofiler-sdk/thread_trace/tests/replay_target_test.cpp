// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Per-pass thread trace targets for kernel replay: a replay pass may trace a different target CU,
// SIMD or shader engine mask than its context was configured with. These tests cover the three
// layers involved -- validating a selection, building packets for it, and the dispatch path that
// picks the packet -- against real agents, without submitting any GPU work.

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_hooks/client_ids.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_info_session.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/local_context.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"

#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <bitset>
#include <cstdint>
#include <optional>
#include <vector>

using namespace rocprofiler::counters::test_constants;
using namespace rocprofiler;

namespace rocprofiler
{
void
test_init();  // att_packet_test.cpp
}

namespace
{
class ReplayTargetFakeQueue : public hsa::Queue
{
public:
    ReplayTargetFakeQueue(const hsa::AgentCache& a, rocprofiler_queue_id_t id)
    : hsa::Queue(a, get_api_table())
    , _agent(a)
    , _id(id)
    {}
    const hsa::AgentCache& get_agent() const final { return _agent; }
    rocprofiler_queue_id_t get_id() const final { return _id; }

    ~ReplayTargetFakeQueue() override = default;

private:
    const hsa::AgentCache& _agent;
    rocprofiler_queue_id_t _id = {};
};

// Small enough to keep several packets cheap, large enough for every shader engine of the
// agents the SDK supports to get its 128 KiB.
constexpr uint64_t k_buffer_size = 0x1000000;

rocprofiler_thread_trace_control_flags_t
start_stop_dispatch_cb(rocprofiler_agent_id_t,
                       rocprofiler_queue_id_t,
                       rocprofiler_async_correlation_id_t,
                       rocprofiler_kernel_id_t,
                       rocprofiler_dispatch_id_t,
                       void*,
                       rocprofiler_user_data_t*)
{
    return ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP;
}

void ignore_shader_data(rocprofiler_thread_trace_shader_data_t, rocprofiler_user_data_t) {}

thread_trace::thread_trace_parameter_pack
make_pack()
{
    auto pack               = thread_trace::thread_trace_parameter_pack{};
    pack.dispatch_cb_fn     = start_stop_dispatch_cb;
    pack.shader_cb_fn       = ignore_shader_data;
    pack.buffer_size        = k_buffer_size;
    pack.target_cu          = 1;
    pack.simd_select        = 0xF;
    pack.shader_engine_mask = 0x1;
    return pack;
}

uint64_t
num_shader_engines(rocprofiler_agent_id_t agent_id)
{
    const auto* agent = agent::get_agent(agent_id);
    return (agent != nullptr) ? agent->num_shader_banks : 0;
}

// The aql_params value the factory emits for `name`, or nullopt when it emits none.
std::optional<uint32_t>
aql_param(const aql::ThreadTraceAQLPacketFactory& factory,
          hsa_ven_amd_aqlprofile_parameter_name_t name)
{
    for(const auto& param : factory.aql_params)
        if(param.parameter_name == name) return param.value;
    return std::nullopt;
}

// Run one dispatch through the tracer's pre-dispatch hook and complete it the way the exit hook
// does, returning the aqlprofile handle of the trace packet that would have been injected.
std::optional<uint64_t>
trace_one_dispatch(thread_trace::DispatchThreadTracer& tracer,
                   hsa::Queue&                         queue,
                   rocprofiler_dispatch_id_t           dispatch_id)
{
    auto user_data = rocprofiler_user_data_t{};
    auto written   = tracer.pre_kernel_call(queue, 1, dispatch_id, &user_data, nullptr);
    if(!written.packet) return std::nullopt;

    auto* control = dynamic_cast<hsa::TraceControlAQLPacket*>(written.packet.get());
    if(control == nullptr) return std::nullopt;
    const auto handle = control->GetHandle().handle;

    auto inst_pkt = hsa::inst_pkt_t{};
    inst_pkt.emplace_back(std::move(written.packet), hsa::queue_hooks::THREAD_TRACE_CLIENT_ID);
    auto session     = hsa::queue_info_session_t{.queue = queue};
    auto packet_data = hsa::packet_data_t{};
    tracer.post_kernel_call(inst_pkt, session, packet_data);
    return handle;
}

void
init_runtime()
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();
    registration::init_logging();
    registration::set_init_status(-1);
}
}  // namespace

// A selection is resolved against the configured parameters of the replayed agent: parameters
// left out keep their configured value, only the three target parameters are accepted, and every
// value is checked against the limits aqlprofile enforces when it builds the packets.
TEST(thread_trace_replay_target, resolve_target_rules)
{
    init_runtime();

    const auto& agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_FALSE(agents.empty());

    for(const auto& [_, cache] : agents)
    {
        const auto agent_id = CHECK_NOTNULL(cache.get_rocp_agent())->id;
        SCOPED_TRACE(::testing::Message() << cache.get_rocp_agent()->name);

        auto tracer = thread_trace::DispatchThreadTracer{};
        tracer.add_agent(agent_id, make_pack());

        using param_t = rocprofiler_thread_trace_parameter_t;
        auto target   = thread_trace::dispatch_target_t{};
        auto resolve  = [&](std::vector<param_t> params) {
            target = thread_trace::dispatch_target_t{};
            return tracer.resolve_target(agent_id, params.data(), params.size(), target);
        };

        EXPECT_EQ(tracer.resolve_target(
                      rocprofiler_agent_id_t{.handle = agent_id.handle + 1000}, nullptr, 0, target),
                  ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT)
            << "an agent the context does not trace on has nothing to resolve against";

        ASSERT_EQ(resolve({}), ROCPROFILER_STATUS_SUCCESS);
        EXPECT_EQ(target, (thread_trace::dispatch_target_t{1, 0xF, 0x1}));

        ASSERT_EQ(resolve({{ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, {2}}}),
                  ROCPROFILER_STATUS_SUCCESS);
        EXPECT_EQ(target, (thread_trace::dispatch_target_t{1, 2, 0x1}));

        ASSERT_EQ(resolve({{ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, {0}},
                           {ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, {3}},
                           {ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {0x1}}}),
                  ROCPROFILER_STATUS_SUCCESS);
        EXPECT_EQ(target, (thread_trace::dispatch_target_t{0, 3, 0x1}));

        EXPECT_EQ(resolve({{ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, {16}}}),
                  ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
        EXPECT_EQ(resolve({{ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, {16}}}),
                  ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
        EXPECT_EQ(resolve({{ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {0}}}),
                  ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
        EXPECT_EQ(resolve({{ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, {k_buffer_size}}}),
                  ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT)
            << "only the target parameters can change between passes";
        EXPECT_EQ(tracer.resolve_target(agent_id, nullptr, 1, target),
                  ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);

        const auto num_se = num_shader_engines(agent_id);
        if(num_se == 0 || num_se >= 64) continue;

        EXPECT_EQ(resolve({{ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK,
                            {uint64_t{1} << num_se}}}),
                  ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT)
            << "a mask bit without a shader engine behind it would silently trace less";

        const uint64_t all_engines = (uint64_t{1} << num_se) - 1;
        ASSERT_EQ(resolve({{ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {all_engines}}}),
                  ROCPROFILER_STATUS_SUCCESS);
        EXPECT_EQ(target.shader_engine_mask, all_engines);

        // The smallest buffer configuration accepts leaves less than 128 KiB per engine once it
        // is split between enough of them; that has to be rejected here rather than throw when
        // aqlprofile builds the packets.
        auto small_pack        = make_pack();
        small_pack.buffer_size = 0x100000;
        tracer.add_agent(agent_id, small_pack);
        const uint64_t per_engine =
            ((small_pack.buffer_size - (64 - num_se) * 0x1000) / num_se) & ~uint64_t{0xFFF};
        EXPECT_EQ(resolve({{ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {all_engines}}}),
                  (per_engine < 0x20000) ? ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT
                                         : ROCPROFILER_STATUS_SUCCESS);
    }
}

// Packets built for another target carry that target to aqlprofile and write into the configured
// packet's trace buffer instead of allocating their own.
TEST(thread_trace_replay_target, target_packets_share_the_trace_buffer)
{
    init_runtime();

    const auto& agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_FALSE(agents.empty());

    for(const auto& [_, cache] : agents)
    {
        const auto agent_id = CHECK_NOTNULL(cache.get_rocp_agent())->id;
        SCOPED_TRACE(::testing::Message() << cache.get_rocp_agent()->name);

        const auto base_pack    = make_pack();
        auto       base_factory = aql::ThreadTraceAQLPacketFactory{agent_id, base_pack};
        auto       base         = base_factory.construct_control_packet();
        ASSERT_NE(base->GetOutputBuffer(), nullptr);
        EXPECT_GE(base->GetOutputSize(), k_buffer_size);

        auto target_pack        = base_pack;
        target_pack.target_cu   = 0;
        target_pack.simd_select = 2;
        auto target_factory     = aql::ThreadTraceAQLPacketFactory{agent_id, target_pack};
        EXPECT_EQ(
            aql_param(target_factory, HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET),
            0u);
        EXPECT_EQ(aql_param(target_factory, HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION),
                  2u);

        {
            auto target = target_factory.construct_control_packet(*base);
            EXPECT_NE(target->GetHandle().handle, base->GetHandle().handle)
                << "each target needs its own start and stop packets";
            EXPECT_EQ(target->GetOutputBuffer(), base->GetOutputBuffer());

            target->populate_before();
            target->populate_after();
            EXPECT_FALSE(target->before_krn_pkt.empty());
            EXPECT_FALSE(target->after_krn_pkt.empty());
        }

        // The configured packet still owns its buffer after a packet that borrowed it is gone.
        EXPECT_NE(base->GetOutputBuffer(), nullptr);
    }
}

// End to end through the dispatch hook: inside a replay loop, a selection made in the PASS-enter
// window changes which packet the next dispatch gets; repeating a target reuses its packets;
// dropping the selection, or leaving the loop, goes back to the configured packet.
TEST(thread_trace_replay_target, selection_changes_the_injected_packet)
{
    init_runtime();

    const hsa::AgentCache* cache = nullptr;
    for(const auto& [_, itr] : hsa::get_queue_controller()->get_supported_agents())
    {
        if(itr.get_rocp_agent() == nullptr || !itr.get_rocp_agent()->runtime_visibility.hsa)
            continue;
        cache = &itr;
        break;
    }
    if(cache == nullptr) GTEST_SKIP() << "no ATT-capable GPU agent available";
    const auto agent_id = cache->get_rocp_agent()->id;

    context::push_client(1);
    auto ctx = rocprofiler_context_id_t{0};
    ASSERT_EQ(rocprofiler_create_context(&ctx), ROCPROFILER_STATUS_SUCCESS);
    auto parameters = std::vector<rocprofiler_thread_trace_parameter_t>{
        {ROCPROFILER_THREAD_TRACE_PARAMETER_RESOURCE_MODE,
         {ROCPROFILER_THREAD_TRACE_PARAMETER_RESOURCE_MODE_HSA}},
        {ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, {k_buffer_size}},
        {ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {0x1}}};
    ASSERT_EQ(rocprofiler_configure_dispatch_thread_trace_service(ctx,
                                                                  agent_id,
                                                                  parameters.data(),
                                                                  parameters.size(),
                                                                  start_stop_dispatch_cb,
                                                                  ignore_shader_data,
                                                                  nullptr),
              ROCPROFILER_STATUS_SUCCESS);
    context::pop_client(1);

    auto* ctx_p = context::get_mutable_registered_context(ctx);
    ASSERT_TRUE(ctx_p != nullptr && ctx_p->dispatch_thread_trace);
    auto& tracer = *ctx_p->dispatch_thread_trace;
    tracer.resource_init();
    if(tracer.get_agents().count(agent_id) == 0)
        GTEST_SKIP() << "the tracer did not build resources for this agent";
    tracer.start_context();

    auto queue = ReplayTargetFakeQueue{*cache, {.handle = 17}};

    const auto configured = trace_one_dispatch(tracer, queue, 1);
    ASSERT_TRUE(configured.has_value());

    using param_t  = rocprofiler_thread_trace_parameter_t;
    const auto set = [&](std::vector<param_t> params) {
        kernel_replay::set_toggles_armed(true);
        auto status = kernel_replay::replay_local_set_thread_trace_parameters(
            ctx, params.data(), params.size());
        kernel_replay::set_toggles_armed(false);
        return status;
    };
    const auto simd = [](uint64_t value) {
        return param_t{ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, {value}};
    };

    EXPECT_EQ(trace_one_dispatch(tracer, queue, 2), configured)
        << "a dispatch outside a replay loop traces the configured target";

    {
        auto active = context::context_array_t{};
        active.emplace_back(ctx_p);
        auto loop = kernel_replay::scoped_local_context_control{active, agent_id};

        EXPECT_EQ(kernel_replay::replay_local_set_thread_trace_parameters(ctx, nullptr, 0),
                  ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR)
            << "a selection is only legal in the PASS-enter window";

        ASSERT_EQ(set({simd(1)}), ROCPROFILER_STATUS_SUCCESS);
        const auto simd1 = trace_one_dispatch(tracer, queue, 3);
        ASSERT_TRUE(simd1.has_value());
        EXPECT_NE(simd1, configured);
        EXPECT_EQ(trace_one_dispatch(tracer, queue, 3), simd1)
            << "the selection sticks across passes and its packets are built once";

        ASSERT_EQ(set({simd(2)}), ROCPROFILER_STATUS_SUCCESS);
        const auto simd2 = trace_one_dispatch(tracer, queue, 3);
        ASSERT_TRUE(simd2.has_value());
        EXPECT_NE(simd2, configured);
        EXPECT_NE(simd2, simd1);

        EXPECT_EQ(set({simd(16)}), ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
        EXPECT_EQ(trace_one_dispatch(tracer, queue, 3), simd2)
            << "a rejected selection keeps the previous one";

        ASSERT_EQ(set({simd(1)}), ROCPROFILER_STATUS_SUCCESS);
        EXPECT_EQ(trace_one_dispatch(tracer, queue, 3), simd1);

        ASSERT_EQ(set({}), ROCPROFILER_STATUS_SUCCESS);
        EXPECT_EQ(trace_one_dispatch(tracer, queue, 3), configured)
            << "an empty selection goes back to the configured target";

        ASSERT_EQ(set({simd(2)}), ROCPROFILER_STATUS_SUCCESS);
    }

    EXPECT_FALSE(kernel_replay::local_thread_trace_target(ctx).has_value());
    EXPECT_EQ(trace_one_dispatch(tracer, queue, 4), configured)
        << "the selection must not outlive the replay loop";

    EXPECT_EQ(tracer.pending_post_moves(), 0);
    tracer.stop_context();
    tracer.resource_deinit();
}
