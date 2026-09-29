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

#include "lib/rocprofiler-sdk/kernel_replay/local_context.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"

#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/fwd.h>

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

#include <atomic>

using namespace rocprofiler::counters::test_constants;
using namespace rocprofiler;

namespace
{
class FakeQueue : public hsa::Queue
{
public:
    FakeQueue(const hsa::AgentCache& a, rocprofiler_queue_id_t id)
    : hsa::Queue(a, get_api_table())
    , _agent(a)
    , _id(id)
    {}
    const hsa::AgentCache& get_agent() const final { return _agent; }
    rocprofiler_queue_id_t get_id() const final { return _id; }

    ~FakeQueue() override = default;

private:
    const hsa::AgentCache& _agent;
    rocprofiler_queue_id_t _id = {};
};

}  // namespace

namespace rocprofiler
{
void
test_init();  // att_packet_test.cpp
}

namespace
{
rocprofiler_thread_trace_control_flags_t
counting_dispatch_cb(rocprofiler_agent_id_t,
                     rocprofiler_queue_id_t,
                     rocprofiler_async_correlation_id_t,
                     rocprofiler_kernel_id_t,
                     rocprofiler_dispatch_id_t,
                     void* userdata,
                     rocprofiler_user_data_t*)
{
    static_cast<std::atomic<int>*>(userdata)->fetch_add(1);
    return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;
}

context::context_array_t
as_active(const context::context& ctx)
{
    context::context_array_t active{};
    active.emplace_back(&ctx);
    return active;
}

// resource_init() keys the tracer's agent map on agent::get_hsa_agent(rocp_agent), but
// pre_kernel_call() looks it up with queue.get_agent().get_hsa_agent(). Those are two different
// routes to "the HSA agent for this device", and picking get_supported_agents().begin() gave the
// fake queue an agent the tracer had not keyed, so pre_kernel_call took its agents.end() branch and
// the dispatch callback never ran -- the test saw zero hits.
//
// Register every supported agent (as resource_creation does) and then let the tracer report which
// AgentCache it actually mapped. That is precisely the invariant pre_kernel_call needs, read off
// the tracer's own state rather than re-derived here, so the test does not depend on the two
// lookup routes agreeing.
const hsa::AgentCache*
register_agents_and_find_traceable(thread_trace::DispatchThreadTracer&              tracer,
                                   const thread_trace::thread_trace_parameter_pack& params)
{
    const auto& supported = hsa::get_queue_controller()->get_supported_agents();

    for(const auto& [_, cache] : supported)
    {
        if(cache.get_rocp_agent() != nullptr) tracer.add_agent(cache.get_rocp_agent()->id, params);
    }

    tracer.resource_init();

    for(const auto& [_, cache] : supported)
    {
        if(tracer.get_agents().count(cache.get_rocp_agent()->id) == 1) return &cache;
    }

    return nullptr;
}

// pre_kernel_call() only reaches dispatch_cb_fn once the tracer's agent map and the fake queue's
// AgentCache resolve to the same agent and the agent admits a dispatch. Neither is something
// the override decides, so establish that a dispatch lands before asserting anything about the
// override. Without this the forced-off expectation below passes on a node that never dispatches
// at all, which is indistinguishable from the skip it is meant to prove.
bool
baseline_reaches_dispatch_cb(thread_trace::DispatchThreadTracer& tracer,
                             const hsa::Queue&                   queue,
                             std::atomic<int>&                   hits,
                             rocprofiler_user_data_t*            user_data)
{
    hits.store(0);
    tracer.pre_kernel_call(queue, /*kernel_id=*/1, /*dispatch_id=*/0, user_data, nullptr);
    const bool reached = hits.load() == 1;
    hits.store(0);
    return reached;
}
}  // namespace

TEST(thread_trace, local_context_override_skips_pre_kernel_call)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);

    ASSERT_FALSE(hsa::get_queue_controller()->get_supported_agents().empty());

    std::atomic<int> hits{0};

    thread_trace::thread_trace_parameter_pack params{};
    params.context_id            = {.handle = 62};
    params.dispatch_cb_fn        = counting_dispatch_cb;
    params.callback_userdata.ptr = &hits;

    thread_trace::DispatchThreadTracer tracer{};

    const auto* agent_ptr = register_agents_and_find_traceable(tracer, params);
    if(agent_ptr == nullptr)
        GTEST_SKIP() << "no supported agent is reachable through the tracer's agent map";
    const auto& agent = *agent_ptr;

    rocprofiler_queue_id_t  qid{.handle = 1};
    FakeQueue               fq(agent, qid);
    rocprofiler_user_data_t user_data{};

    context::context dummy{};
    dummy.context_idx = params.context_id.handle;

    // Separate the two ways a missing baseline can arise: an override left behind outside a replay
    // loop is a defect in the code under test and must fail here, whereas an agent that cannot
    // deliver a dispatch at all is an environment limit and skips below.
    ASSERT_FALSE(kernel_replay::local_context_override(params.context_id).has_value())
        << "no replay loop is active, so this context must carry no override";

    if(!baseline_reaches_dispatch_cb(tracer, fq, hits, &user_data))
    {
        tracer.resource_deinit();
        GTEST_SKIP() << "no dispatch reaches the ATT callback through a fake queue on this agent, "
                        "so a forced-off skip cannot be told apart from no dispatch at all";
    }

    {
        auto                                        active = as_active(dummy);
        kernel_replay::scoped_local_context_control loop{active};
        kernel_replay::set_toggles_armed(true);
        EXPECT_EQ(kernel_replay::replay_local_disable_context(params.context_id),
                  ROCPROFILER_STATUS_SUCCESS);
        kernel_replay::set_toggles_armed(false);

        hits.store(0);
        tracer.pre_kernel_call(fq, /*kernel_id=*/1, /*dispatch_id=*/2, &user_data, nullptr);
        EXPECT_EQ(hits.load(), 0) << "ATT dispatch callback must not run when locally stopped";
    }

    hits.store(0);
    tracer.pre_kernel_call(fq, /*kernel_id=*/1, /*dispatch_id=*/3, &user_data, nullptr);
    EXPECT_EQ(hits.load(), 1) << "the local stop must not outlive the replay loop";

    tracer.resource_deinit();
}

TEST(thread_trace, local_context_override_forced_on_still_invokes_dispatch_cb)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);

    ASSERT_FALSE(hsa::get_queue_controller()->get_supported_agents().empty());

    std::atomic<int> hits{0};

    thread_trace::thread_trace_parameter_pack params{};
    params.context_id            = {.handle = 63};
    params.dispatch_cb_fn        = counting_dispatch_cb;
    params.callback_userdata.ptr = &hits;

    thread_trace::DispatchThreadTracer tracer{};

    const auto* agent_ptr = register_agents_and_find_traceable(tracer, params);
    if(agent_ptr == nullptr)
        GTEST_SKIP() << "no supported agent is reachable through the tracer's agent map";
    const auto& agent = *agent_ptr;

    rocprofiler_queue_id_t  qid{.handle = 1};
    FakeQueue               fq(agent, qid);
    rocprofiler_user_data_t user_data{};

    context::context dummy{};
    dummy.context_idx = params.context_id.handle;

    ASSERT_FALSE(kernel_replay::local_context_override(params.context_id).has_value())
        << "no replay loop is active, so this context must carry no override";

    if(!baseline_reaches_dispatch_cb(tracer, fq, hits, &user_data))
    {
        tracer.resource_deinit();
        GTEST_SKIP() << "no dispatch reaches the ATT callback through a fake queue on this agent, "
                        "so a forced-on dispatch would prove nothing";
    }

    {
        auto                                        active = as_active(dummy);
        kernel_replay::scoped_local_context_control loop{active};
        kernel_replay::set_toggles_armed(true);
        EXPECT_EQ(kernel_replay::replay_local_enable_context(params.context_id),
                  ROCPROFILER_STATUS_SUCCESS);
        kernel_replay::set_toggles_armed(false);

        hits.store(0);
        tracer.pre_kernel_call(fq, /*kernel_id=*/1, /*dispatch_id=*/1, &user_data, nullptr);
        EXPECT_EQ(hits.load(), 1) << "ATT only skips when forced off; a local start is not a skip";
    }

    tracer.resource_deinit();
}
