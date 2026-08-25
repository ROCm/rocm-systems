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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"
#include "lib/rocprofiler-sdk/thread_trace/queue_hooks.hpp"

#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>
#include <hsa/hsa.h>

using namespace rocprofiler::counters::test_constants;
using namespace rocprofiler;

namespace rocprofiler
{
namespace hsa
{
class PerAgentFakeQueue : public Queue
{
public:
    PerAgentFakeQueue(const AgentCache& a, rocprofiler_queue_id_t id)
    : Queue(a, get_api_table())
    , _agent(a)
    , _id(id)
    {}
    const AgentCache&      get_agent() const final { return _agent; };
    rocprofiler_queue_id_t get_id() const final { return _id; };

    ~PerAgentFakeQueue() override = default;

private:
    const AgentCache&      _agent;
    rocprofiler_queue_id_t _id = {};
};
}  // namespace hsa
}  // namespace rocprofiler

namespace
{
struct agent_pair
{
    const hsa::AgentCache* first  = nullptr;
    const hsa::AgentCache* second = nullptr;
};

void
test_init()
{
    static bool _initialized = false;
    if(_initialized) return;
    _initialized = true;

    HsaApiTable table;
    table.amd_ext_ = &get_ext_table();
    table.core_    = &get_api_table();
    agent::construct_agent_cache(&table);
    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
    hsa::get_queue_controller()->init(get_api_table(), get_ext_table());
    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
}

agent_pair
get_two_agents()
{
    auto  result = agent_pair{};
    auto& agents = hsa::get_queue_controller()->get_supported_agents();
    for(const auto& [_, agent] : agents)
    {
        if(!agent.get_rocp_agent()) continue;
        if(!result.first)
            result.first = &agent;
        else if(!result.second)
            result.second = &agent;
        else
            break;
    }
    return result;
}

rocprofiler_thread_trace_control_flags_t
noop_dispatch_cb(rocprofiler_agent_id_t,
                 rocprofiler_queue_id_t,
                 rocprofiler_async_correlation_id_t,
                 rocprofiler_kernel_id_t,
                 rocprofiler_dispatch_id_t,
                 void*,
                 rocprofiler_user_data_t*)
{
    return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;
}

void noop_shader_cb(rocprofiler_thread_trace_shader_data_t, rocprofiler_user_data_t) {}

rocprofiler_context_id_t
make_att_context(rocprofiler_agent_id_t agent_id)
{
    auto ctx = rocprofiler_context_id_t{0};
    EXPECT_EQ(rocprofiler_create_context(&ctx), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_configure_dispatch_thread_trace_service(
                  ctx, agent_id, nullptr, 0, noop_dispatch_cb, noop_shader_cb, nullptr),
              ROCPROFILER_STATUS_SUCCESS);
    return ctx;
}

TEST(thread_trace_per_agent, collects_on_and_intersects_match_configured_agents)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.second) GTEST_SKIP() << "fewer than two GPU agents available";

    auto agent_a = agents.first->get_rocp_agent()->id;
    auto agent_b = agents.second->get_rocp_agent()->id;

    auto left  = thread_trace::DispatchThreadTracer{};
    auto right = thread_trace::DispatchThreadTracer{};
    left.add_agent(agent_a, {});
    right.add_agent(agent_b, {});

    EXPECT_TRUE(left.collects_on(agent_a));
    EXPECT_FALSE(left.collects_on(agent_b));
    EXPECT_FALSE(left.intersects(right));

    right.add_agent(agent_a, {});
    EXPECT_TRUE(left.intersects(right));

    auto configured = left.configured_agents();
    EXPECT_EQ(configured.size(), 1U);
    EXPECT_EQ(configured.count(agent_a), 1U);
}

TEST(thread_trace_per_agent, disjoint_agent_contexts_can_be_active_together)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.second) GTEST_SKIP() << "fewer than two GPU agents available";

    auto ctx_a = make_att_context(agents.first->get_rocp_agent()->id);
    auto ctx_b = make_att_context(agents.second->get_rocp_agent()->id);

    EXPECT_EQ(rocprofiler_start_context(ctx_a), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_start_context(ctx_b), ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(rocprofiler_stop_context(ctx_b), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_stop_context(ctx_a), ROCPROFILER_STATUS_SUCCESS);
}

TEST(thread_trace_per_agent, overlapping_agent_contexts_still_conflict)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.first) GTEST_SKIP() << "no GPU agent available";

    auto agent_a = agents.first->get_rocp_agent()->id;
    auto ctx_a   = make_att_context(agent_a);
    auto ctx_b   = make_att_context(agent_a);

    ASSERT_EQ(rocprofiler_start_context(ctx_a), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_start_context(ctx_b), ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT);
    ASSERT_EQ(rocprofiler_stop_context(ctx_a), ROCPROFILER_STATUS_SUCCESS);
}

TEST(thread_trace_per_agent, serialization_is_scoped_to_the_contexts_agents)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.second) GTEST_SKIP() << "fewer than two GPU agents available";

    auto  agent_a    = agents.first->get_rocp_agent()->id;
    auto  agent_b    = agents.second->get_rocp_agent()->id;
    auto* controller = hsa::get_queue_controller();

    auto queue_a = hsa::PerAgentFakeQueue{*agents.first, {.handle = 501}};
    auto queue_b = hsa::PerAgentFakeQueue{*agents.second, {.handle = 502}};
    controller->serializer(&queue_a);
    controller->serializer(&queue_b);

    auto ctx = make_att_context(agent_b);
    ASSERT_EQ(rocprofiler_start_context(ctx), ROCPROFILER_STATUS_SUCCESS);

    EXPECT_TRUE(controller->is_serialization_enabled(agent_b));
    EXPECT_FALSE(controller->is_serialization_enabled(agent_a));

    ASSERT_EQ(rocprofiler_stop_context(ctx), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_FALSE(controller->is_serialization_enabled(agent_b));
}

TEST(thread_trace_per_agent, write_hook_ignores_dispatches_on_other_agents)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.second) GTEST_SKIP() << "fewer than two GPU agents available";

    auto ctx = make_att_context(agents.second->get_rocp_agent()->id);
    ASSERT_EQ(rocprofiler_start_context(ctx), ROCPROFILER_STATUS_SUCCESS);

    auto packet      = hsa::rocprofiler_packet{};
    auto corr_id     = context::correlation_id{};
    corr_id.internal = 4242;

    {
        auto queue_a       = hsa::PerAgentFakeQueue{*agents.first, {.handle = 601}};
        auto inst_pkt      = hsa::inst_pkt_t{};
        bool is_serialized = false;
        auto user_data     = rocprofiler_user_data_t{.value = corr_id.internal};

        thread_trace::write_hook(
            queue_a, packet, 42, 1, &user_data, {}, &corr_id, inst_pkt, is_serialized);

        EXPECT_FALSE(is_serialized)
            << "a dispatch on an agent outside the context must not be serialized";
        EXPECT_TRUE(inst_pkt.empty())
            << "a dispatch on an agent outside the context must not be instrumented";
    }

    ASSERT_EQ(rocprofiler_stop_context(ctx), ROCPROFILER_STATUS_SUCCESS);
}
}  // namespace
