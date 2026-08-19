// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_hooks/client_ids.hpp"
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
class HookTestFakeQueue : public Queue
{
public:
    HookTestFakeQueue(const AgentCache& a, rocprofiler_queue_id_t id)
    : Queue(a, get_api_table())
    , _agent(a)
    , _id(id)
    {}

    const AgentCache&      get_agent() const final { return _agent; };
    rocprofiler_queue_id_t get_id() const final { return _id; };

    ~HookTestFakeQueue() override = default;

private:
    const AgentCache&      _agent;
    rocprofiler_queue_id_t _id = {};
};
}  // namespace hsa
}  // namespace rocprofiler

namespace
{
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

TEST(ThreadTraceQueueHooks, IsAnyActiveReturnsFalseWhenNoContextActive)
{
    EXPECT_FALSE(rocprofiler::thread_trace::is_any_active());
}

// A dispatch instrumented while the context is active must still complete via
// signal_completion_hook after stop_context clears the active slot.
TEST(ThreadTraceQueueHooks, StopContextInFlightCompletionRoutesViaHookPath)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto& agents = hsa::get_queue_controller()->get_supported_agents();
    const hsa::AgentCache* att_agent = nullptr;
    for(const auto& [_, agent] : agents)
    {
        if(!agent.get_rocp_agent()) continue;
        if(!agent.get_rocp_agent()->runtime_visibility.hsa) continue;
        att_agent = &agent;
        break;
    }
    if(!att_agent) GTEST_SKIP() << "no ATT-capable GPU agent available";

    auto ctx = rocprofiler_context_id_t{0};
    ASSERT_EQ(rocprofiler_create_context(&ctx), ROCPROFILER_STATUS_SUCCESS);
    ASSERT_EQ(rocprofiler_configure_dispatch_thread_trace_service(
                  ctx,
                  att_agent->get_rocp_agent()->id,
                  nullptr,
                  0,
                  start_stop_dispatch_cb,
                  [](rocprofiler_thread_trace_shader_data_t, rocprofiler_user_data_t) {},
                  nullptr),
              ROCPROFILER_STATUS_SUCCESS);
    ASSERT_EQ(rocprofiler_start_context(ctx), ROCPROFILER_STATUS_SUCCESS);

    auto* ctx_p = context::get_mutable_registered_context(ctx);
    ASSERT_TRUE(ctx_p && ctx_p->dispatch_thread_trace);
    auto& tracer = *ctx_p->dispatch_thread_trace;

    hsa::HookTestFakeQueue fq(*att_agent, {.handle = 901});
    hsa::rocprofiler_packet pkt{};
    context::correlation_id corr_id{};
    corr_id.internal = 42;
    auto user_data = rocprofiler_user_data_t{.value = corr_id.internal};

    hsa::inst_pkt_t inst_pkt;
    bool            is_serialized = false;
    thread_trace::write_hook(fq,
                             pkt,
                             1,
                             1,
                             &user_data,
                             {},
                             &corr_id,
                             inst_pkt,
                             is_serialized);

    ASSERT_FALSE(inst_pkt.empty()) << "write_hook must inject ATT control packet";
    EXPECT_GE(tracer.post_move_data.load(), 1);

    ASSERT_EQ(rocprofiler_stop_context(ctx), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_FALSE(thread_trace::is_any_active());

    auto sess = std::make_shared<hsa::queue_info_session_t>(hsa::queue_info_session_t{.queue = fq});
    auto packet_data = hsa::packet_data_t{};
    packet_data.user_data = user_data;

    thread_trace::signal_completion_hook(fq, pkt, sess, packet_data, inst_pkt, {});

    EXPECT_EQ(tracer.post_move_data.load(), 0)
        << "post_move_data must drain via signal_completion_hook after stop_context";

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
}
}  // namespace
