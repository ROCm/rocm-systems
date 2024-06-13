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

#include "lib/rocprofiler-sdk/counters/agent_profiling.hpp"
#include <cstdint>

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/controller.hpp"
#include "lib/rocprofiler-sdk/counters/core.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
#include "rocprofiler-sdk/fwd.h"

namespace rocprofiler
{
namespace counters
{
std::atomic<bool>&
hsa_inited()
{
    static std::atomic<bool> inited{false};
    return inited;
}

uint64_t
submitPacket(const CoreApiTable& table, hsa_queue_t* queue, const void* packet)
{
    const uint32_t pkt_size = 0x40;

    // advance command queue
    const uint64_t write_idx = table.hsa_queue_add_write_index_scacq_screl_fn(queue, 1);
    while((write_idx - table.hsa_queue_load_read_index_relaxed_fn(queue)) >= queue->size)
    {
        sched_yield();
    }

    const uint32_t slot_idx = (uint32_t)(write_idx % queue->size);
    // NOLINTBEGIN(performance-no-int-to-ptr)
    uint32_t* queue_slot =
        reinterpret_cast<uint32_t*>((uintptr_t)(queue->base_address) + (slot_idx * pkt_size));
    // NOLINTEND(performance-no-int-to-ptr)

    const uint32_t* slot_data = reinterpret_cast<const uint32_t*>(packet);

    // Copy buffered commands into the queue slot.
    // Overwrite the AQL invalid header (first dword) last.
    // This prevents the slot from being read until it's fully written.
    memcpy(&queue_slot[1], &slot_data[1], pkt_size - sizeof(uint32_t));
    std::atomic<uint32_t>* header_atomic_ptr =
        reinterpret_cast<std::atomic<uint32_t>*>(&queue_slot[0]);
    header_atomic_ptr->store(slot_data[0], std::memory_order_release);

    // ringdoor bell
    table.hsa_signal_store_relaxed_fn(queue->doorbell_signal, write_idx);

    return write_idx;
}

namespace
{
uint16_t
header_pkt(hsa_packet_type_t type)
{
    uint16_t header = type << HSA_PACKET_HEADER_TYPE;
    header |= 1 << HSA_PACKET_HEADER_BARRIER;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    return header;
}

std::unique_ptr<hsa::CounterAQLPacket>
construct_aql_pkt(const hsa::AgentCache& agent, std::shared_ptr<profile_config>& profile)
{
    if(counter_callback_info::setup_profile_config(agent, profile) != ROCPROFILER_STATUS_SUCCESS)
    {
        return nullptr;
    }

    auto pkts = profile->pkt_generator->construct_packet(
        CHECK_NOTNULL(hsa::get_queue_controller())->get_ext_table());

    pkts->start.header                   = header_pkt(HSA_PACKET_TYPE_VENDOR_SPECIFIC);
    pkts->start.completion_signal.handle = 0;
    pkts->stop.header                    = header_pkt(HSA_PACKET_TYPE_VENDOR_SPECIFIC);
    pkts->read.header                    = header_pkt(HSA_PACKET_TYPE_VENDOR_SPECIFIC);
    return pkts;
}

bool
agent_async_handler(hsa_signal_value_t /*signal_v*/, void* data)
{
    const auto* ctx = context::get_registered_context({.handle = (uint64_t) data});
    if(!ctx) return false;

    const auto& agent_ctx   = *ctx->agent_counter_collection;
    const auto& prof_config = agent_ctx.profile;

    // Decode the AQL packet data
    auto decoded_pkt =
        EvaluateAST::read_pkt(prof_config->pkt_generator.get(), *agent_ctx.callback_data.packet);
    EvaluateAST::read_special_counters(
        *prof_config->agent, prof_config->required_special_counters, decoded_pkt);

    auto* buf = buffer::get_buffer(agent_ctx.buffer.handle);
    if(!buf)
    {
        ROCP_FATAL << fmt::format("Buffer {} destroyed before record was written",
                                  agent_ctx.buffer.handle);
        return false;
    }

    // Write out the AQL data to the buffer
    for(auto& ast : prof_config->asts)
    {
        std::vector<std::unique_ptr<std::vector<rocprofiler_record_counter_t>>> cache;
        auto* ret = CHECK_NOTNULL(ast.evaluate(decoded_pkt, cache));
        ast.set_out_id(*ret);
        for(auto& val : *ret)
        {
            val.user_data = agent_ctx.callback_data.user_data;
            buf->emplace(
                ROCPROFILER_BUFFER_CATEGORY_COUNTERS, ROCPROFILER_COUNTER_RECORD_VALUE, val);
        }
    }

    // reset the signal to allow another sample to start
    agent_ctx.callback_data.table.hsa_signal_store_relaxed_fn(agent_ctx.callback_data.completion,
                                                              1);
    return true;
}

void
init_callback_data(const rocprofiler::context::context& ctx, const hsa::AgentCache& agent)
{
    // Note: Calls to this function should be protected by agent_ctx.status being set
    // to LOCKED by the caller. This is to prevent multiple threads from trying to
    // setup the same agent at the same time.
    auto& agent_ctx = *ctx.agent_counter_collection;
    if(agent_ctx.callback_data.packet) return;

    agent_ctx.callback_data.packet = construct_aql_pkt(agent, agent_ctx.profile);

    if(agent_ctx.callback_data.completion.handle != 0) return;

    // If we do not have a completion handle, this is our first time profiling this agent.
    // Setup our shared data structures.
    agent_ctx.callback_data.queue = agent.profile_queue();

    agent_ctx.callback_data.table = CHECK_NOTNULL(hsa::get_queue_controller())->get_core_table();

    // Tri-state signal
    //   1: allow next sample to start
    //   0: sample in progress
    //  -1: sample complete
    CHECK_EQ(agent_ctx.callback_data.table.hsa_signal_create_fn(
                 1, 0, nullptr, &agent_ctx.callback_data.completion),
             HSA_STATUS_SUCCESS);

    // Signal to manage the startup of the context. Allows us to ensure that
    // the AQL packet we inject with start_context() completes before returning
    CHECK_EQ(
        agent_ctx.callback_data.table.hsa_signal_create_fn(1, 0, nullptr, &agent_ctx.start_signal),
        HSA_STATUS_SUCCESS);

    // Setup callback
    // NOLINTBEGIN(performance-no-int-to-ptr)
    CHECK_EQ(CHECK_NOTNULL(hsa::get_queue_controller())
                 ->get_ext_table()
                 .hsa_amd_signal_async_handler_fn(agent_ctx.callback_data.completion,
                                                  HSA_SIGNAL_CONDITION_LT,
                                                  0,
                                                  agent_async_handler,
                                                  (void*) ctx.context_idx),
             HSA_STATUS_SUCCESS);
    // NOLINTEND(performance-no-int-to-ptr)

    // Set state of the queue to allow profiling (may not be needed since AQL
    // may do this in the future).
    CHECK(agent.cpu_pool().handle != 0);
    CHECK(agent.get_hsa_agent().handle != 0);

    aql::set_profiler_active_on_queue(
        CHECK_NOTNULL(hsa::get_queue_controller())->get_ext_table(),
        agent.cpu_pool(),
        agent.get_hsa_agent(),
        [&](hsa::rocprofiler_packet pkt) {
            pkt.ext_amd_aql_pm4.completion_signal = agent_ctx.callback_data.completion;
            submitPacket(
                agent_ctx.callback_data.table, agent_ctx.callback_data.queue, (void*) &pkt);
            if(agent_ctx.callback_data.table.hsa_signal_wait_relaxed_fn(
                   agent_ctx.callback_data.completion,
                   HSA_SIGNAL_CONDITION_EQ,
                   0,
                   20000000,
                   HSA_WAIT_STATE_ACTIVE) != 0)
            {
                ROCP_FATAL << "Could not set agent to be profiled";
            }
            agent_ctx.callback_data.table.hsa_signal_store_relaxed_fn(
                agent_ctx.callback_data.completion, 1);
        });
}
}  // namespace

rocprofiler_status_t
read_agent_ctx(const context::context*    ctx,
               rocprofiler_user_data_t    user_data,
               rocprofiler_counter_flag_t flags)
{
    if(!ctx->agent_counter_collection || !ctx->agent_counter_collection->profile)
    {
        if(!ctx->agent_counter_collection)
        {
            ROCP_ERROR << fmt::format("Context {} has no agent counter collection",
                                      ctx->context_idx);
        }
        else
        {
            ROCP_ERROR << fmt::format("Context {} has no profile", ctx->context_idx);
        }
        return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;
    }

    auto& agent_ctx = *ctx->agent_counter_collection;

    if(hsa_inited().load() == false)
    {
        return ROCPROFILER_STATUS_ERROR;
    }

    const auto* agent = agent::get_agent_cache(agent_ctx.profile->agent);

    // If the agent no longer exists or we don't have a profile queue, reading is an error
    if(!agent || !agent->profile_queue()) return ROCPROFILER_STATUS_ERROR;

    // Set the state to LOCKED to prevent other calls to start/stop/read.
    auto expected = rocprofiler::context::agent_counter_collection_service::state::ENABLED;
    if(!agent_ctx.status.compare_exchange_strong(
           expected, rocprofiler::context::agent_counter_collection_service::state::LOCKED))
    {
        return ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR;
    }

    CHECK(agent_ctx.callback_data.packet);

    ROCP_TRACE << fmt::format("Agent Infor for Running Counter: Name = {}, XCC = {}, "
                              "SE = {}, CU = {}, SIMD = {}",
                              agent->get_rocp_agent()->name,
                              agent->get_rocp_agent()->num_xcc,
                              agent->get_rocp_agent()->num_shader_banks,
                              agent->get_rocp_agent()->cu_count,
                              agent->get_rocp_agent()->simd_arrays_per_engine);

    // Submit the read packet to the queue
    submitPacket(agent_ctx.callback_data.table,
                 agent->profile_queue(),
                 (void*) &agent_ctx.callback_data.packet->read);

    // Submit a barrier packet. This is needed to flush hardware caches. Without this
    // the read packet may not have the correct data.
    rocprofiler::hsa::rocprofiler_packet barrier{};
    barrier.barrier_and.header            = header_pkt(HSA_PACKET_TYPE_BARRIER_AND);
    barrier.barrier_and.completion_signal = agent_ctx.callback_data.completion;
    agent_ctx.callback_data.table.hsa_signal_store_relaxed_fn(agent_ctx.callback_data.completion,
                                                              0);
    agent_ctx.callback_data.user_data = user_data;
    submitPacket(
        agent_ctx.callback_data.table, agent->profile_queue(), (void*) &barrier.barrier_and);

    // Wait for the barrier/read packet to complete
    if(flags != ROCPROFILER_COUNTER_FLAG_ASYNC)
    {
        // Wait for any inprogress samples to complete before returning
        agent_ctx.callback_data.table.hsa_signal_wait_relaxed_fn(agent_ctx.callback_data.completion,
                                                                 HSA_SIGNAL_CONDITION_EQ,
                                                                 1,
                                                                 UINT64_MAX,
                                                                 HSA_WAIT_STATE_ACTIVE);
    }

    agent_ctx.status.exchange(
        rocprofiler::context::agent_counter_collection_service::state::ENABLED);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
start_agent_ctx(const context::context* ctx)
{
    auto status = ROCPROFILER_STATUS_SUCCESS;
    if(!ctx->agent_counter_collection)
    {
        return status;
    }

    auto& agent_ctx = *ctx->agent_counter_collection;

    if(hsa_inited().load() == false)
    {
        return ROCPROFILER_STATUS_SUCCESS;
    }

    const auto* agent = agent::get_agent_cache(agent::get_agent(agent_ctx.agent_id));
    // Note: we may not have an AgentCache yet if HSA is not started.
    // This is not an error and the startup will happen on hsa registration.
    if(!agent) return ROCPROFILER_STATUS_ERROR;

    // But if we have an agent cache, we need a profile queue.
    if(!agent->profile_queue())
    {
        return ROCPROFILER_STATUS_ERROR_NO_PROFILE_QUEUE;
    }

    // Set the state to LOCKED to prevent other calls to start/stop/read.
    auto expected = rocprofiler::context::agent_counter_collection_service::state::DISABLED;
    if(!agent_ctx.status.compare_exchange_strong(
           expected, rocprofiler::context::agent_counter_collection_service::state::LOCKED))
    {
        return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;
    }

    // Ask the tool what profile we should use for this agent
    agent_ctx.cb(
        {.handle = ctx->context_idx},
        agent_ctx.agent_id,
        [](rocprofiler_context_id_t        context_id,
           rocprofiler_profile_config_id_t config_id) -> rocprofiler_status_t {
            auto* cb_ctx = rocprofiler::context::get_mutable_registered_context(context_id);
            if(!cb_ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;

            auto config = rocprofiler::counters::get_profile_config(config_id);
            if(!config) return ROCPROFILER_STATUS_ERROR_PROFILE_NOT_FOUND;

            if(!cb_ctx->agent_counter_collection)
            {
                return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;
            }

            // Only allow profiles to be set in the locked state
            if(cb_ctx->agent_counter_collection->status.load() !=
               rocprofiler::context::agent_counter_collection_service::state::LOCKED)
            {
                return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;
            }

            // Only update the profile if it has changed. Avoids packet regeneration.
            if(!cb_ctx->agent_counter_collection->profile ||
               cb_ctx->agent_counter_collection->profile->id.handle != config_id.handle)
            {
                if(cb_ctx->agent_counter_collection->agent_id.handle != config->agent->id.handle)
                {
                    return ROCPROFILER_STATUS_ERROR_AGENT_MISMATCH;
                }

                cb_ctx->agent_counter_collection->profile = config;
                cb_ctx->agent_counter_collection->callback_data.packet.reset();
            }
            return ROCPROFILER_STATUS_SUCCESS;
        },
        agent_ctx.user_data);

    // User didn't set a profile
    if(!agent_ctx.profile)
    {
        agent_ctx.status.exchange(
            rocprofiler::context::agent_counter_collection_service::state::DISABLED);
        return status;
    }

    // Generate necessary structures in the context (packet gen, etc) to process
    // this packet.
    init_callback_data(*ctx, *agent);

    // No hardware counters were actually asked for (i.e. all constants)
    if(agent_ctx.profile->reqired_hw_counters.empty())
    {
        agent_ctx.status.exchange(
            rocprofiler::context::agent_counter_collection_service::state::DISABLED);
        return ROCPROFILER_STATUS_ERROR_NO_HARDWARE_COUNTERS;
    }

    // We could not generate AQL packets for some reason
    if(!agent_ctx.callback_data.packet)
    {
        agent_ctx.status.exchange(
            rocprofiler::context::agent_counter_collection_service::state::DISABLED);
        return ROCPROFILER_STATUS_ERROR_AST_GENERATION_FAILED;
    }

    agent_ctx.callback_data.packet->start.completion_signal = agent_ctx.start_signal;
    agent_ctx.callback_data.table.hsa_signal_store_relaxed_fn(agent_ctx.start_signal, 1);
    submitPacket(agent_ctx.callback_data.table,
                 agent->profile_queue(),
                 (void*) &agent_ctx.callback_data.packet->start);

    // Wait for startup to finish before continuing
    agent_ctx.callback_data.table.hsa_signal_wait_relaxed_fn(
        agent_ctx.start_signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);

    agent_ctx.status.exchange(
        rocprofiler::context::agent_counter_collection_service::state::ENABLED);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
stop_agent_ctx(const context::context* ctx)
{
    auto status = ROCPROFILER_STATUS_SUCCESS;
    if(!ctx->agent_counter_collection || !ctx->agent_counter_collection->profile)
    {
        return status;
    }

    auto& agent_ctx = *ctx->agent_counter_collection;

    if(hsa_inited().load() == false)
    {
        return ROCPROFILER_STATUS_SUCCESS;
    }

    const auto* agent = agent::get_agent_cache(agent_ctx.profile->agent);
    if(!agent || !agent->profile_queue()) return status;

    auto expected = rocprofiler::context::agent_counter_collection_service::state::ENABLED;
    if(!agent_ctx.status.compare_exchange_strong(
           expected, rocprofiler::context::agent_counter_collection_service::state::LOCKED))
    {
        // Status is already stopped or being enabled elsewhere.
        return ROCPROFILER_STATUS_SUCCESS;
    }

    CHECK(agent_ctx.callback_data.packet);

    submitPacket(agent_ctx.callback_data.table,
                 agent->profile_queue(),
                 (void*) &agent_ctx.callback_data.packet->stop);

    // Wait for any inprogress samples to complete before returning
    agent_ctx.callback_data.table.hsa_signal_wait_relaxed_fn(agent_ctx.callback_data.completion,
                                                             HSA_SIGNAL_CONDITION_EQ,
                                                             1,
                                                             UINT64_MAX,
                                                             HSA_WAIT_STATE_ACTIVE);

    return status;
}

// If we have ctx's that were started before HSA was initialized, we need to
// actually start those contexts now.
rocprofiler_status_t
agent_profile_hsa_registration()
{
    hsa_inited().store(true);

    for(auto& ctx : context::get_active_contexts())
    {
        if(!ctx->agent_counter_collection) continue;
        start_agent_ctx(ctx);
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

agent_callback_data::~agent_callback_data()
{
    if(completion.handle != 0) table.hsa_signal_destroy_fn(completion);
}
}  // namespace counters
}  // namespace rocprofiler