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
#include "lib/rocprofiler-sdk/counters/core.hpp"
#include "lib/rocprofiler-sdk/counters/queue_hooks.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>
#include <hsa/hsa.h>

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

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
// Two GPU agents are required to say anything about scoping: with one agent, "scoped to agent
// A" and "unscoped" are indistinguishable.
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

void
noop_dispatch_cb(rocprofiler_dispatch_counting_service_data_t,
                 rocprofiler_counter_config_id_t*,
                 rocprofiler_user_data_t*,
                 void*)
{}

void
noop_record_cb(rocprofiler_dispatch_counting_service_data_t,
               rocprofiler_counter_record_t*,
               size_t,
               rocprofiler_user_data_t,
               void*)
{}

// Creates a context with a dispatch counting service configured on it.
rocprofiler_context_id_t
make_counter_context()
{
    auto ctx = rocprofiler_context_id_t{0};
    EXPECT_EQ(rocprofiler_create_context(&ctx), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_configure_callback_dispatch_counting_service(
                  ctx, noop_dispatch_cb, nullptr, noop_record_cb, nullptr),
              ROCPROFILER_STATUS_SUCCESS);
    return ctx;
}

// ---------------------------------------------------------------------------------------------
// rocprofiler_dispatch_counting_service_set_agents argument handling
// ---------------------------------------------------------------------------------------------

TEST(counters_per_agent, set_agents_requires_a_configured_service)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto ctx = rocprofiler_context_id_t{0};
    ASSERT_EQ(rocprofiler_create_context(&ctx), ROCPROFILER_STATUS_SUCCESS);

    auto agents = get_two_agents();
    if(!agents.first) GTEST_SKIP() << "no GPU agent available";
    auto agent_id = agents.first->get_rocp_agent()->id;

    // The agent set lives on the counter-collection service, so there is nothing to restrict
    // until the service exists. Silently succeeding here would let a tool believe it had
    // scoped collection when it had not.
    EXPECT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx, &agent_id, 1),
              ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND);
}

TEST(counters_per_agent, set_agents_rejects_bad_arguments)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.first) GTEST_SKIP() << "no GPU agent available";

    auto ctx      = make_counter_context();
    auto agent_id = agents.first->get_rocp_agent()->id;

    EXPECT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx, nullptr, 1),
              ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);

    // A null array with no entries is the documented way to clear the restriction.
    EXPECT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx, nullptr, 0),
              ROCPROFILER_STATUS_SUCCESS);

    auto bogus = rocprofiler_agent_id_t{.handle = 0xDEADBEEF};
    EXPECT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx, &bogus, 1),
              ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND);

    // The rejected call must not have partially applied: a failed set leaves the previous
    // (unrestricted) set in place rather than a half-built one.
    auto* ctx_p = context::get_mutable_registered_context(ctx);
    ASSERT_TRUE(ctx_p);
    ASSERT_TRUE(ctx_p->dispatch_counter_collection);
    EXPECT_TRUE(ctx_p->dispatch_counter_collection->agents.empty());

    EXPECT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx, &agent_id, 1),
              ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(ctx_p->dispatch_counter_collection->agents.size(), 1U);
    EXPECT_TRUE(ctx_p->dispatch_counter_collection->collects_on(agent_id));
}

TEST(counters_per_agent, set_agents_is_locked_while_the_context_is_active)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.first) GTEST_SKIP() << "no GPU agent available";

    auto ctx      = make_counter_context();
    auto agent_id = agents.first->get_rocp_agent()->id;

    ASSERT_EQ(rocprofiler_start_context(ctx), ROCPROFILER_STATUS_SUCCESS);

    // The dispatch path reads the agent set without a lock, and start_context has already used
    // it to scope serialization, so changing it mid-flight would silently desynchronize the two.
    EXPECT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx, &agent_id, 1),
              ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED);

    ASSERT_EQ(rocprofiler_stop_context(ctx), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx, &agent_id, 1),
              ROCPROFILER_STATUS_SUCCESS);
}

// The HSA write interceptor gates on is_active_on_agent(), so this is what decides whether a
// queue on an unrelated GPU stays on the fast path and keeps packet batching. An unrestricted
// context claims every agent; one narrowed with set_agents() must claim only its own.
TEST(counters_per_agent, is_active_on_agent_follows_the_scoped_agent_set)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.first || !agents.second) GTEST_SKIP() << "needs two GPU agents";

    const auto agent_0 = agents.first->get_rocp_agent()->id;
    const auto agent_1 = agents.second->get_rocp_agent()->id;

    auto ctx = make_counter_context();

    // Registered but never started, so it claims nothing.
    EXPECT_FALSE(counters::is_active_on_agent(agent_0));

    ASSERT_EQ(rocprofiler_start_context(ctx), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_TRUE(counters::is_active_on_agent(agent_0));
    EXPECT_TRUE(counters::is_active_on_agent(agent_1));
    ASSERT_EQ(rocprofiler_stop_context(ctx), ROCPROFILER_STATUS_SUCCESS);

    ASSERT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx, &agent_1, 1),
              ROCPROFILER_STATUS_SUCCESS);
    ASSERT_EQ(rocprofiler_start_context(ctx), ROCPROFILER_STATUS_SUCCESS);

    EXPECT_FALSE(counters::is_active_on_agent(agent_0));
    EXPECT_TRUE(counters::is_active_on_agent(agent_1));

    // is_any_active() cannot draw that distinction, which is the reason the gate moved off it.
    EXPECT_TRUE(counters::is_any_active());

    ASSERT_EQ(rocprofiler_stop_context(ctx), ROCPROFILER_STATUS_SUCCESS);
}

// ---------------------------------------------------------------------------------------------
// Context conflict: the scenario from the #9586 review
// ---------------------------------------------------------------------------------------------

// Before per-agent scoping this was impossible -- any second counter-collection context was
// rejected outright, so "context A on GPU-0, context B on GPU-1" could not even be expressed.
TEST(counters_per_agent, disjoint_agent_contexts_can_be_active_together)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.second) GTEST_SKIP() << "fewer than two GPU agents available";

    auto agent_a = agents.first->get_rocp_agent()->id;
    auto agent_b = agents.second->get_rocp_agent()->id;

    auto ctx_a = make_counter_context();
    auto ctx_b = make_counter_context();

    ASSERT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx_a, &agent_a, 1),
              ROCPROFILER_STATUS_SUCCESS);
    ASSERT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx_b, &agent_b, 1),
              ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(rocprofiler_start_context(ctx_a), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_start_context(ctx_b), ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(rocprofiler_stop_context(ctx_b), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_stop_context(ctx_a), ROCPROFILER_STATUS_SUCCESS);
}

TEST(counters_per_agent, overlapping_agent_contexts_still_conflict)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.first) GTEST_SKIP() << "no GPU agent available";
    auto agent_a = agents.first->get_rocp_agent()->id;

    auto ctx_a = make_counter_context();
    auto ctx_b = make_counter_context();

    ASSERT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx_a, &agent_a, 1),
              ROCPROFILER_STATUS_SUCCESS);
    ASSERT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx_b, &agent_a, 1),
              ROCPROFILER_STATUS_SUCCESS);

    ASSERT_EQ(rocprofiler_start_context(ctx_a), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_start_context(ctx_b), ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT);

    ASSERT_EQ(rocprofiler_stop_context(ctx_a), ROCPROFILER_STATUS_SUCCESS);
}

// An unrestricted context claims every agent, so it must keep conflicting with everything --
// otherwise existing single-context tools would start silently sharing agents.
TEST(counters_per_agent, unrestricted_context_conflicts_with_a_scoped_one)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.second) GTEST_SKIP() << "fewer than two GPU agents available";
    auto agent_b = agents.second->get_rocp_agent()->id;

    auto ctx_all    = make_counter_context();
    auto ctx_scoped = make_counter_context();

    ASSERT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx_scoped, &agent_b, 1),
              ROCPROFILER_STATUS_SUCCESS);

    ASSERT_EQ(rocprofiler_start_context(ctx_all), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_start_context(ctx_scoped), ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT);

    ASSERT_EQ(rocprofiler_stop_context(ctx_all), ROCPROFILER_STATUS_SUCCESS);
}

// ---------------------------------------------------------------------------------------------
// Serialization scope and reference counting
// ---------------------------------------------------------------------------------------------

// The actual complaint in the review: a context that only wants GPU-1 must not serialize GPU-0.
TEST(counters_per_agent, serialization_is_scoped_to_the_contexts_agents)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.second) GTEST_SKIP() << "fewer than two GPU agents available";

    auto  agent_a    = agents.first->get_rocp_agent()->id;
    auto  agent_b    = agents.second->get_rocp_agent()->id;
    auto* controller = hsa::get_queue_controller();

    // Touch both serializers so the agents are known to the controller; otherwise "not
    // serialized" would be trivially true because the entry does not exist yet.
    auto queue_a = hsa::PerAgentFakeQueue{*agents.first, {.handle = 101}};
    auto queue_b = hsa::PerAgentFakeQueue{*agents.second, {.handle = 102}};
    controller->serializer(&queue_a);
    controller->serializer(&queue_b);

    auto ctx = make_counter_context();
    ASSERT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx, &agent_b, 1),
              ROCPROFILER_STATUS_SUCCESS);

    auto* ctx_p = context::get_mutable_registered_context(ctx);
    ASSERT_TRUE(ctx_p && ctx_p->dispatch_counter_collection);
    ASSERT_EQ(ctx_p->dispatch_counter_collection->agents.size(), 1U) << "agent set not applied";
    ASSERT_TRUE(ctx_p->dispatch_counter_collection->collects_on(agent_b));
    ASSERT_FALSE(ctx_p->dispatch_counter_collection->collects_on(agent_a));

    EXPECT_FALSE(controller->is_serialization_enabled(agent_a));
    EXPECT_FALSE(controller->is_serialization_enabled(agent_b));

    ASSERT_EQ(rocprofiler_start_context(ctx), ROCPROFILER_STATUS_SUCCESS);

    EXPECT_TRUE(controller->is_serialization_enabled(agent_b))
        << "the context's own agent must be serialized";
    EXPECT_FALSE(controller->is_serialization_enabled(agent_a))
        << "an agent outside the context's agent set must not be serialized";

    ASSERT_EQ(rocprofiler_stop_context(ctx), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_FALSE(controller->is_serialization_enabled(agent_b));
}

// counters::start_context() acquires serialization and stop_context() releases it once per
// enabled -> disabled transition, so a repeated start must not keep a second acquisition.
// rocprofiler_start_context() already returns early for an active context, so this drives the
// service directly.
TEST(counters_per_agent, repeated_service_start_does_not_pin_serialization)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.first) GTEST_SKIP() << "no GPU agent available";

    auto  agent_a    = agents.first->get_rocp_agent()->id;
    auto* controller = hsa::get_queue_controller();
    auto  queue_a    = hsa::PerAgentFakeQueue{*agents.first, {.handle = 151}};
    controller->serializer(&queue_a);

    auto ctx = make_counter_context();
    ASSERT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx, &agent_a, 1),
              ROCPROFILER_STATUS_SUCCESS);
    const auto* ctx_p = context::get_registered_context(ctx);
    ASSERT_TRUE(ctx_p && ctx_p->dispatch_counter_collection);

    ASSERT_FALSE(controller->is_serialization_enabled(agent_a));
    counters::start_context(ctx_p);
    counters::start_context(ctx_p);
    EXPECT_TRUE(controller->is_serialization_enabled(agent_a));

    counters::stop_context(ctx_p);
    EXPECT_FALSE(controller->is_serialization_enabled(agent_a))
        << "a repeated start must not leave the agent serialized after one stop";

    // An unmatched stop must not release a reference it does not hold.
    counters::stop_context(ctx_p);
    EXPECT_FALSE(controller->is_serialization_enabled(agent_a));
}

TEST(counters_per_agent, concurrent_context_start_stop_does_not_pin_serialization)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.first) GTEST_SKIP() << "no GPU agent available";

    auto  agent_a    = agents.first->get_rocp_agent()->id;
    auto* controller = hsa::get_queue_controller();
    auto  queue_a    = hsa::PerAgentFakeQueue{*agents.first, {.handle = 152}};
    controller->serializer(&queue_a);

    auto ctx = make_counter_context();
    ASSERT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx, &agent_a, 1),
              ROCPROFILER_STATUS_SUCCESS);

    auto unexpected = std::atomic<int>{0};
    auto go         = std::atomic<bool>{false};
    auto worker     = [&]() {
        while(!go.load(std::memory_order_acquire))
            std::this_thread::yield();

        for(int i = 0; i < 64; ++i)
        {
            if(rocprofiler_start_context(ctx) != ROCPROFILER_STATUS_SUCCESS)
                unexpected.fetch_add(1, std::memory_order_relaxed);

            auto status = rocprofiler_stop_context(ctx);
            if(status != ROCPROFILER_STATUS_SUCCESS &&
               status != ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND)
                unexpected.fetch_add(1, std::memory_order_relaxed);
        }
    };

    auto first  = std::thread{worker};
    auto second = std::thread{worker};
    go.store(true, std::memory_order_release);
    first.join();
    second.join();

    (void) rocprofiler_stop_context(ctx);
    EXPECT_EQ(unexpected.load(std::memory_order_relaxed), 0);
    EXPECT_FALSE(controller->is_serialization_enabled(agent_a));

    const auto* ctx_p = context::get_registered_context(ctx);
    ASSERT_TRUE(ctx_p && ctx_p->dispatch_counter_collection);
    bool enabled = true;
    ctx_p->dispatch_counter_collection->enabled.rlock([&](const auto& value) { enabled = value; });
    EXPECT_FALSE(enabled);
}

TEST(counters_per_agent, concurrent_overlapping_context_starts_admit_exactly_one)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.first) GTEST_SKIP() << "no GPU agent available";

    auto  agent_a    = agents.first->get_rocp_agent()->id;
    auto* controller = hsa::get_queue_controller();
    auto  queue_a    = hsa::PerAgentFakeQueue{*agents.first, {.handle = 153}};
    controller->serializer(&queue_a);

    auto first_ctx  = make_counter_context();
    auto second_ctx = make_counter_context();
    ASSERT_EQ(rocprofiler_dispatch_counting_service_set_agents(first_ctx, &agent_a, 1),
              ROCPROFILER_STATUS_SUCCESS);
    ASSERT_EQ(rocprofiler_dispatch_counting_service_set_agents(second_ctx, &agent_a, 1),
              ROCPROFILER_STATUS_SUCCESS);

    for(int iteration = 0; iteration < 32; ++iteration)
    {
        auto ready         = std::atomic<int>{0};
        auto go            = std::atomic<bool>{false};
        auto first_status  = ROCPROFILER_STATUS_ERROR;
        auto second_status = ROCPROFILER_STATUS_ERROR;
        auto start         = [&](rocprofiler_context_id_t context, rocprofiler_status_t& status) {
            ready.fetch_add(1, std::memory_order_release);
            while(!go.load(std::memory_order_acquire))
                std::this_thread::yield();
            status = rocprofiler_start_context(context);
        };

        auto first  = std::thread{start, first_ctx, std::ref(first_status)};
        auto second = std::thread{start, second_ctx, std::ref(second_status)};
        while(ready.load(std::memory_order_acquire) != 2)
            std::this_thread::yield();
        go.store(true, std::memory_order_release);
        first.join();
        second.join();

        const auto first_won  = first_status == ROCPROFILER_STATUS_SUCCESS;
        const auto second_won = second_status == ROCPROFILER_STATUS_SUCCESS;
        EXPECT_NE(first_won, second_won);
        EXPECT_TRUE(first_status == ROCPROFILER_STATUS_SUCCESS ||
                    first_status == ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT);
        EXPECT_TRUE(second_status == ROCPROFILER_STATUS_SUCCESS ||
                    second_status == ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT);

        auto winner = first_won ? first_ctx : second_ctx;
        ASSERT_EQ(rocprofiler_stop_context(winner), ROCPROFILER_STATUS_SUCCESS);
        EXPECT_FALSE(controller->is_serialization_enabled(agent_a));
        EXPECT_FALSE(counters::is_any_active());
    }
}

// Counter collection, thread trace and SPM each enable serialization independently. Without
// reference counting the first one to stop unserializes the others, which is a pre-existing
// bug that per-agent scoping would otherwise make easier to hit.
TEST(counters_per_agent, serialization_is_reference_counted_across_subsystems)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.first) GTEST_SKIP() << "no GPU agent available";

    auto  agent_a    = agents.first->get_rocp_agent()->id;
    auto* controller = hsa::get_queue_controller();

    auto queue_a = hsa::PerAgentFakeQueue{*agents.first, {.handle = 201}};
    controller->serializer(&queue_a);

    ASSERT_FALSE(controller->is_serialization_enabled(agent_a));

    controller->enable_serialization();
    controller->enable_serialization();
    EXPECT_TRUE(controller->is_serialization_enabled(agent_a));

    controller->disable_serialization();
    EXPECT_TRUE(controller->is_serialization_enabled(agent_a))
        << "one subsystem stopping must not unserialize an agent another still needs";

    controller->disable_serialization();
    EXPECT_FALSE(controller->is_serialization_enabled(agent_a));

    // Over-releasing must not drive the count negative, or the next enable would be swallowed.
    controller->disable_serialization();
    controller->enable_serialization();
    EXPECT_TRUE(controller->is_serialization_enabled(agent_a));
    controller->disable_serialization();
    EXPECT_FALSE(controller->is_serialization_enabled(agent_a));
}

// A scoped request and an unscoped request overlap on the scoped agent; releasing one must not
// release the other.
TEST(counters_per_agent, scoped_and_unscoped_serialization_compose)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.second) GTEST_SKIP() << "fewer than two GPU agents available";

    auto  agent_a    = agents.first->get_rocp_agent()->id;
    auto  agent_b    = agents.second->get_rocp_agent()->id;
    auto* controller = hsa::get_queue_controller();

    auto queue_a = hsa::PerAgentFakeQueue{*agents.first, {.handle = 301}};
    auto queue_b = hsa::PerAgentFakeQueue{*agents.second, {.handle = 302}};
    controller->serializer(&queue_a);
    controller->serializer(&queue_b);

    controller->enable_serialization({agent_b});
    EXPECT_FALSE(controller->is_serialization_enabled(agent_a));
    EXPECT_TRUE(controller->is_serialization_enabled(agent_b));

    controller->enable_serialization();
    EXPECT_TRUE(controller->is_serialization_enabled(agent_a));
    EXPECT_TRUE(controller->is_serialization_enabled(agent_b));

    controller->disable_serialization();
    EXPECT_FALSE(controller->is_serialization_enabled(agent_a));
    EXPECT_TRUE(controller->is_serialization_enabled(agent_b))
        << "the scoped request still holds agent B";

    controller->disable_serialization({agent_b});
    EXPECT_FALSE(controller->is_serialization_enabled(agent_b));
}

// ---------------------------------------------------------------------------------------------
// Dispatch path
// ---------------------------------------------------------------------------------------------

// queue_cb returns serialize=true even when it declines to instrument, so a tool cannot avoid
// serializing a GPU by returning a null config -- the filtering has to happen before the call.
// This is the end-to-end form of the review's example.
TEST(counters_per_agent, enter_hook_ignores_dispatches_on_other_agents)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    auto agents = get_two_agents();
    if(!agents.second) GTEST_SKIP() << "fewer than two GPU agents available";

    auto agent_b = agents.second->get_rocp_agent()->id;

    auto ctx = make_counter_context();
    ASSERT_EQ(rocprofiler_dispatch_counting_service_set_agents(ctx, &agent_b, 1),
              ROCPROFILER_STATUS_SUCCESS);
    ASSERT_EQ(rocprofiler_start_context(ctx), ROCPROFILER_STATUS_SUCCESS);

    auto packet      = hsa::rocprofiler_packet{};
    auto corr_id     = context::correlation_id{};
    corr_id.internal = 4242;

    // A dispatch on the agent the context does not collect on.
    {
        auto queue_a       = hsa::PerAgentFakeQueue{*agents.first, {.handle = 401}};
        auto inst_pkt      = hsa::inst_pkt_t{};
        bool is_serialized = false;
        auto user_data     = rocprofiler_user_data_t{.value = corr_id.internal};

        counters::kernel_dispatch_phase_enter_hook(
            queue_a, packet, 42, 1, &user_data, {}, &corr_id, inst_pkt, is_serialized);

        EXPECT_FALSE(is_serialized)
            << "a dispatch on an agent outside the context's agent set must not be serialized";
        EXPECT_TRUE(inst_pkt.empty())
            << "a dispatch on an agent outside the context's agent set must not be instrumented";
    }

    // The same dispatch on the agent the context does collect on still goes through, so the
    // test cannot pass just because the hook stopped doing anything at all.
    {
        auto queue_b       = hsa::PerAgentFakeQueue{*agents.second, {.handle = 402}};
        auto inst_pkt      = hsa::inst_pkt_t{};
        bool is_serialized = false;
        auto user_data     = rocprofiler_user_data_t{.value = corr_id.internal};

        counters::kernel_dispatch_phase_enter_hook(
            queue_b, packet, 42, 2, &user_data, {}, &corr_id, inst_pkt, is_serialized);

        EXPECT_TRUE(is_serialized)
            << "a dispatch on the context's own agent must still be serialized";
    }

    ASSERT_EQ(rocprofiler_stop_context(ctx), ROCPROFILER_STATUS_SUCCESS);
}
}  // namespace
