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

// DispatchThreadTracer::resource_init() keys its agent map on agent::get_hsa_agent(), while
// pre_kernel_call() looks that map up with the handle the queue's AgentCache carries. Both are
// filled by construct_agent_cache(), but only agents where the two agree can be driven through
// pre_kernel_call -- for the rest it takes the agents.end() branch and never reaches the dispatch
// callback. Pick an agent that satisfies that invariant instead of whichever one happens to hash
// first, which is arbitrary on a multi-GPU node.
const hsa::AgentCache*
find_traceable_agent()
{
    const auto& agents = hsa::get_queue_controller()->get_supported_agents();

    for(const auto& [_, cache] : agents)
    {
        const auto* rocp_agent = cache.get_rocp_agent();
        if(rocp_agent == nullptr) continue;

        auto mapped = agent::get_hsa_agent(rocp_agent);
        if(mapped && mapped->handle == cache.get_hsa_agent().handle) return &cache;
    }

    return nullptr;
}
}  // namespace

TEST(thread_trace, local_context_override_skips_pre_kernel_call)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);

    ASSERT_FALSE(hsa::get_queue_controller()->get_supported_agents().empty());

    const auto* agent_ptr = find_traceable_agent();
    if(agent_ptr == nullptr)
        GTEST_SKIP() << "no supported agent maps back to its own HSA handle on this system";
    const auto& agent = *agent_ptr;

    std::atomic<int> hits{0};

    thread_trace::thread_trace_parameter_pack params{};
    params.context_id            = {.handle = 62};
    params.dispatch_cb_fn        = counting_dispatch_cb;
    params.callback_userdata.ptr = &hits;

    thread_trace::DispatchThreadTracer tracer{};
    tracer.add_agent(agent.get_rocp_agent()->id, params);
    tracer.resource_init();
    ASSERT_EQ(tracer.get_agents().count(agent.get_hsa_agent()), 1U)
        << "tracer did not register the agent backing the fake queue";

    rocprofiler_queue_id_t  qid{.handle = 1};
    FakeQueue               fq(agent, qid);
    rocprofiler_user_data_t user_data{};

    context::context dummy{};
    dummy.context_idx = params.context_id.handle;

    tracer.pre_kernel_call(fq, /*kernel_id=*/1, /*dispatch_id=*/1, &user_data, nullptr);
    EXPECT_EQ(hits.load(), 1);

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
    EXPECT_EQ(hits.load(), 1);

    tracer.resource_deinit();
}

TEST(thread_trace, local_context_override_forced_on_still_invokes_dispatch_cb)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);

    ASSERT_FALSE(hsa::get_queue_controller()->get_supported_agents().empty());

    const auto* agent_ptr = find_traceable_agent();
    if(agent_ptr == nullptr)
        GTEST_SKIP() << "no supported agent maps back to its own HSA handle on this system";
    const auto& agent = *agent_ptr;

    std::atomic<int> hits{0};

    thread_trace::thread_trace_parameter_pack params{};
    params.context_id            = {.handle = 63};
    params.dispatch_cb_fn        = counting_dispatch_cb;
    params.callback_userdata.ptr = &hits;

    thread_trace::DispatchThreadTracer tracer{};
    tracer.add_agent(agent.get_rocp_agent()->id, params);
    tracer.resource_init();
    ASSERT_EQ(tracer.get_agents().count(agent.get_hsa_agent()), 1U)
        << "tracer did not register the agent backing the fake queue";

    rocprofiler_queue_id_t  qid{.handle = 1};
    FakeQueue               fq(agent, qid);
    rocprofiler_user_data_t user_data{};

    context::context dummy{};
    dummy.context_idx = params.context_id.handle;

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
