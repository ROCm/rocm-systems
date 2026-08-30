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

#include "lib/rocprofiler-sdk/counters/queue_hooks.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/core.hpp"
#include "lib/rocprofiler-sdk/counters/dispatch_handlers.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_hooks/client_ids.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>
#include <hsa/hsa.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

using namespace rocprofiler::counters::test_constants;
using namespace rocprofiler;

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        ASSERT_EQ(CHECKSTATUS, ROCPROFILER_STATUS_SUCCESS)                                         \
            << msg << ": " << rocprofiler_get_status_string(CHECKSTATUS);                          \
    }

namespace rocprofiler
{
namespace hsa
{
class QueueHooksFakeQueue : public Queue
{
public:
    QueueHooksFakeQueue(const AgentCache& a, rocprofiler_queue_id_t id)
    : Queue(a, get_api_table())
    , _agent(a)
    , _id(id)
    {}
    const AgentCache&      get_agent() const final { return _agent; };
    rocprofiler_queue_id_t get_id() const final { return _id; };

    ~QueueHooksFakeQueue() override = default;

private:
    const AgentCache&      _agent;
    rocprofiler_queue_id_t _id = {};
};
}  // namespace hsa
}  // namespace rocprofiler

namespace
{
rocprofiler_context_id_t&
get_client_ctx()
{
    static rocprofiler_context_id_t ctx{0};
    return ctx;
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

struct expected_dispatch
{
    rocprofiler_counter_config_id_t id          = {.handle = 0};
    rocprofiler_queue_id_t          queue_id    = {.handle = 0};
    rocprofiler_agent_id_t          agent_id    = {.handle = 0};
    uint64_t                        kernel_id   = 0;
    uint64_t                        dispatch_id = 0;
};

void
user_dispatch_cb(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                 rocprofiler_counter_config_id_t*             config,
                 rocprofiler_user_data_t*,
                 void* callback_data_args)
{
    auto& expected = *static_cast<expected_dispatch*>(callback_data_args);
    ASSERT_NE(config, nullptr);
    config->handle = expected.id.handle;
    (void) dispatch_data;
}

// Records delivered to the tool, keyed by dispatch id. Guarded because before stop_context the
// consumer thread delivers these, and after it consumer_thread_t::add() consumes inline on the
// caller -- so the writer can be either thread depending on when the completion lands.
struct delivered_records
{
    std::mutex          mtx{};
    std::vector<size_t> dispatch_ids{};

    void add(size_t dispatch_id)
    {
        auto lk = std::unique_lock{mtx};
        dispatch_ids.emplace_back(dispatch_id);
    }

    std::vector<size_t> snapshot()
    {
        auto lk = std::unique_lock{mtx};
        return dispatch_ids;
    }

    // Reset so a repeated run of the test does not see the previous run's deliveries.
    void clear()
    {
        auto lk = std::unique_lock{mtx};
        dispatch_ids.clear();
    }
};

delivered_records&
get_delivered()
{
    static delivered_records _v{};
    return _v;
}

void
user_record_cb(rocprofiler_dispatch_counting_service_data_t dispatch_data,
               rocprofiler_counter_record_t*,
               size_t,
               rocprofiler_user_data_t,
               void*)
{
    get_delivered().add(dispatch_data.dispatch_info.dispatch_id);
}

// is_any_active replaces the queue.get_notifiers() signal that the old per-queue
// callback registration used to provide to the HSA write interceptor gate. With
// no dispatch-counter-collection context active, it must report inactive. This
// requires no GPU / HSA runtime.
TEST(counters_queue_hooks, is_any_active_false_when_no_context_active)
{
    EXPECT_FALSE(rocprofiler::counters::is_any_active());
}

TEST(counters_queue_hooks, exit_hook_skips_when_inst_pkt_has_no_counter_client_id)
{
    hsa::inst_pkt_t inst_pkt;
    inst_pkt.emplace_back(
        std::make_pair(std::make_unique<rocprofiler::hsa::AQLPacket>(),
                       rocprofiler::hsa::queue_hooks::THREAD_TRACE_CLIENT_ID));

    auto sess    = std::make_shared<rocprofiler::hsa::queue_info_session_t>();
    auto packet  = rocprofiler::hsa::packet_data_t{};
    auto fq_pkt  = rocprofiler::hsa::rocprofiler_packet{};

    // Must return without touching registered counter contexts (no init required).
    rocprofiler::counters::kernel_dispatch_phase_exit_hook(
        *reinterpret_cast<rocprofiler::hsa::Queue*>(nullptr),
        fq_pkt,
        sess,
        packet,
        inst_pkt,
        rocprofiler::kernel_dispatch::profiling_time{});
    SUCCEED();
}

TEST(counters_queue_hooks, is_any_active_true_while_context_started)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");
    ROCPROFILER_CALL(rocprofiler_configure_callback_dispatch_counting_service(
                         get_client_ctx(), user_dispatch_cb, nullptr, nullptr, nullptr),
                     "Could not setup counting service");
    ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()), "start context");

    EXPECT_TRUE(rocprofiler::counters::is_any_active());

    ROCPROFILER_CALL(rocprofiler_stop_context(get_client_ctx()), "stop context");
    EXPECT_FALSE(rocprofiler::counters::is_any_active());

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
    set_client_ctx(get_client_ctx());
}

// Regression for callback-registry removal, per the review on #8891: dispatches enqueued while the
// context is active must still complete when stop_context runs before their GPU completions land.
//
// Two invariants are checked, because they fail independently:
//
//  1. Routing. Each dispatch's packet_return_map entry is erased by its own completion, so the map
//     is asserted to drop by exactly one per completion rather than only to reach zero at the end.
//     An aggregate check would pass even if one completion erased another dispatch's entry.
//  2. Delivery. Every dispatch reaches the tool's record callback. Erasure happens in completed_cb
//     before the hand-off to the consumer, so a test that only checks the map would pass even if
//     nothing were ever delivered -- which is the failure the review actually described.
//
// Several dispatches are in flight at once, since a single one cannot distinguish "routed the one
// entry" from "routed by provenance".
//
// No kernel executes here, so the counter values are whatever the profile's output buffer holds;
// the assertion is about delivery, not values.
TEST(counters_queue_hooks, stop_context_in_flight_completion_routes_via_hook_path)
{
    constexpr size_t num_dispatches = 3;

    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");
    // A record callback is supplied so delivery is observable; the base test passed nullptr, which
    // is why it could not detect a dropped completion.
    ROCPROFILER_CALL(rocprofiler_configure_callback_dispatch_counting_service(
                         get_client_ctx(), user_dispatch_cb, nullptr, user_record_cb, nullptr),
                     "Could not setup counting service");
    ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()), "start context");

    auto* ctx_p = context::get_mutable_registered_context(get_client_ctx());
    ASSERT_TRUE(ctx_p);
    ASSERT_TRUE(ctx_p->dispatch_counter_collection);
    ASSERT_EQ(ctx_p->dispatch_counter_collection->callbacks.size(), 1);
    auto cb_info = ctx_p->dispatch_counter_collection->callbacks.front();

    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
    auto agents = hsa::get_queue_controller()->get_supported_agents();
    ASSERT_GT(agents.size(), 0);

    const auto& [_, agent] = *agents.begin();
    ASSERT_TRUE(agent.get_rocp_agent());

    rocprofiler_counter_id_t counter_id = {.handle = 0};
    {
        auto        mets = rocprofiler::counters::loadMetrics();
        const auto* gfx_metrics =
            rocprofiler::common::get_val(mets->arch_to_metric, std::string(agent.name()));
        ASSERT_TRUE(gfx_metrics);
        ASSERT_FALSE(gfx_metrics->empty());
        for(const auto& metric : *gfx_metrics)
        {
            if(metric.expression().empty())
            {
                counter_id.handle = metric.id();
                break;
            }
        }
    }
    ASSERT_NE(counter_id.handle, 0U);

    expected_dispatch expected = {};
    ROCPROFILER_CALL(
        rocprofiler_create_counter_config(agent.get_rocp_agent()->id, &counter_id, 1, &expected.id),
        "Unable to create profile");
    cb_info->callback_args = &expected;
    expected.queue_id      = {.handle = 1};
    expected.agent_id      = agent.get_rocp_agent()->id;
    expected.kernel_id     = 42;
    expected.dispatch_id   = 7;

    hsa::QueueHooksFakeQueue fq(agent, expected.queue_id);
    hsa::rocprofiler_packet  pkt{};
    // correlation_id declares constructors and holds private ref counters, so it is not an
    // aggregate and cannot take a designated initializer.
    context::correlation_id corr_id{};
    corr_id.internal = 99;

    get_delivered().clear();

    // Enqueue several dispatches while the context is active. Each queue_cb call must register its
    // own entry; if the implementation ever pooled one packet across concurrent dispatches, the
    // size assertion below would catch it rather than the test silently degenerating to one.
    std::vector<std::unique_ptr<hsa::AQLPacket>> in_flight;
    std::vector<size_t>                          dispatch_ids;
    for(size_t i = 0; i < num_dispatches; ++i)
    {
        const size_t dispatch_id = expected.dispatch_id + i;
        auto         user_data   = rocprofiler_user_data_t{.value = corr_id.internal};
        auto         ret_pkt     = rocprofiler::counters::queue_cb(
            ctx_p, cb_info, fq, pkt, expected.kernel_id, dispatch_id, &user_data, {}, &corr_id);
        ASSERT_TRUE(ret_pkt.packet) << "queue_cb produced no instrumentation for dispatch " << i;
        in_flight.emplace_back(std::move(ret_pkt.packet));
        dispatch_ids.emplace_back(dispatch_id);
    }

    size_t map_size_before_stop = 0;
    cb_info->packet_return_map.rlock([&](const auto& data) { map_size_before_stop = data.size(); });
    ASSERT_EQ(map_size_before_stop, num_dispatches)
        << "queue_cb should register one packet_return_map entry per in-flight dispatch";

    // Stop while every dispatch is still in flight on the GPU. This is the window the review
    // described: the context leaves the active list, so a completion hook that routed by
    // activeness would find nothing to deliver to.
    ROCPROFILER_CALL(rocprofiler_stop_context(get_client_ctx()), "stop context");
    EXPECT_FALSE(rocprofiler::counters::is_any_active());

    // Complete them one at a time, checking the map drops by exactly one each time.
    for(size_t i = 0; i < num_dispatches; ++i)
    {
        auto sess =
            std::make_shared<hsa::queue_info_session_t>(hsa::queue_info_session_t{.queue = fq});
        auto& packet_data                                     = sess->packet_data.emplace_back();
        sess->correlation_id                                  = &corr_id;
        packet_data.callback_record.dispatch_info.dispatch_id = dispatch_ids[i];

        rocprofiler::counters::inst_pkt_t inst_pkt;
        // Construct the pair in place rather than via make_pair: make_pair would deduce the tag's
        // enum type and rely on a pair-to-pair conversion, where piecewise construction converts
        // the tag to ClientID directly. Also matches how the production hook emplaces.
        inst_pkt.emplace_back(std::move(in_flight[i]), hsa::queue_hooks::COUNTERS_CLIENT_ID);

        rocprofiler::counters::kernel_dispatch_phase_exit_hook(
            fq, pkt, sess, packet_data, inst_pkt, rocprofiler::kernel_dispatch::profiling_time{});

        size_t remaining = num_dispatches;
        cb_info->packet_return_map.rlock([&](const auto& data) { remaining = data.size(); });
        EXPECT_EQ(remaining, num_dispatches - (i + 1))
            << "completion " << i
            << " should erase exactly its own packet_return_map entry after stop_context";
    }

    size_t map_size_after_completion = num_dispatches;
    cb_info->packet_return_map.rlock(
        [&](const auto& data) { map_size_after_completion = data.size(); });
    EXPECT_EQ(map_size_after_completion, 0U)
        << "packet_return_map must drain via kernel_dispatch_phase_exit_hook after stop_context";

    // Delivery. callback_thread_stop() has already run as part of stop_context, so these arrive
    // through the inline path in consumer_thread_t::add() rather than the consumer thread -- which
    // is precisely the lifetime property the review asked to be kept intact.
    auto delivered = get_delivered().snapshot();
    EXPECT_EQ(delivered.size(), num_dispatches)
        << "every in-flight dispatch must reach the tool's record callback after stop_context";
    for(auto dispatch_id : dispatch_ids)
    {
        EXPECT_NE(std::find(delivered.begin(), delivered.end(), dispatch_id), delivered.end())
            << "no record delivered for dispatch " << dispatch_id;
    }

    registration::set_init_status(1);
    registration::finalize();
}
}  // namespace
