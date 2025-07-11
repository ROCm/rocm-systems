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

#include "lib/rocprofiler-sdk/counters/device_counting.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/controller.hpp"
#include "lib/rocprofiler-sdk/counters/core.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/details/fmt.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <chrono>
#include <cstdint>
#include <unordered_map>

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
submitPacket(hsa_queue_t* queue, const void* packet)
{
    const uint32_t pkt_size = 0x40;

    // advance command queue
    const uint64_t write_idx =
        hsa::get_core_table()->hsa_queue_add_write_index_scacq_screl_fn(queue, 1);
    while((write_idx - hsa::get_core_table()->hsa_queue_load_read_index_relaxed_fn(queue)) >=
          queue->size)
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
    hsa::get_core_table()->hsa_signal_store_relaxed_fn(queue->doorbell_signal, write_idx);

    ROCP_TRACE << fmt::format("SLOT_IDX: {} WRITE_IDX: {} PKT: {}",
                              slot_idx,
                              write_idx,
                              *static_cast<const hsa::rocprofiler_packet*>(packet));
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

/**
 * Construct the packet or grab it from the cache.
 * Note this function is not thread safe and is only called from
 * init_callback_data which can only be called when the context is in the LOCKED state
 * with only a single thread active.
 */
std::shared_ptr<hsa::CounterAQLPacket>
construct_aql_pkt(std::shared_ptr<counter_config>& profile)
{
    static std::atomic<bool> has_thread{false};
    static std::unordered_map<rocprofiler_profile_config_id_t,
                              std::shared_ptr<hsa::CounterAQLPacket>>
        pkt_cache;
    // Asserts if there are two threads in this function at the same time.
    auto _ = common::assert_single_threaded(has_thread);

    // If we have a packet in the cache, return it.
    if(pkt_cache.find(profile->id) != pkt_cache.end())
    {
        return pkt_cache[profile->id];
    }

    // If we do not have a packet in the cache, create it.
    if(counter_callback_info::setup_counter_config(profile) != ROCPROFILER_STATUS_SUCCESS)
    {
        return nullptr;
    }

    auto pkts = profile->pkt_generator->construct_packet(
        CHECK_NOTNULL(hsa::get_queue_controller())->get_core_table(),
        CHECK_NOTNULL(hsa::get_queue_controller())->get_ext_table());

    pkts->packets.start_packet.header = header_pkt(HSA_PACKET_TYPE_VENDOR_SPECIFIC);
    pkts->packets.stop_packet.header  = header_pkt(HSA_PACKET_TYPE_VENDOR_SPECIFIC);
    pkts->packets.read_packet.header  = header_pkt(HSA_PACKET_TYPE_VENDOR_SPECIFIC);

    pkts->packets.start_packet.completion_signal.handle = 0;

    pkt_cache[profile->id] = std::move(pkts);
    return pkt_cache[profile->id];
}

bool
agent_async_handler(hsa_signal_value_t /*signal_v*/, void* data)
{
    if(!data) return false;
    const auto& callback_data = *static_cast<rocprofiler::counters::agent_callback_data*>(data);

    const auto& prof_config = callback_data.profile;

    // Decode the AQL packet data
    auto decoded_pkt =
        EvaluateAST::read_pkt(prof_config->pkt_generator.get(), *callback_data.packet);
    EvaluateAST::read_special_counters(
        *prof_config->agent, prof_config->required_special_counters, decoded_pkt);

    auto* buf = buffer::get_buffer(callback_data.buffer.handle);
    if(!buf && callback_data.buffer != rocprofiler_buffer_id_t{.handle = 0})
    {
        ROCP_FATAL << fmt::format("Buffer {} destroyed before record was written",
                                  callback_data.buffer.handle);
        return false;
    }

    if(decoded_pkt.empty())
    {
        // reset the signal to allow another sample to start
        hsa::get_core_table()->hsa_signal_store_relaxed_fn(callback_data.completion, 1);
        return true;
    }

    // Write out the AQL data to the buffer
    for(auto& ast : prof_config->asts)
    {
        std::vector<std::unique_ptr<std::vector<rocprofiler_counter_record_t>>> cache;
        auto* ret = CHECK_NOTNULL(ast.evaluate(decoded_pkt, cache));
        ast.set_out_id(*ret);
        for(auto& val : *ret)
        {
            val.user_data = callback_data.user_data;
            val.agent_id  = prof_config->agent->id;
            if(callback_data.cached_counters)
            {
                callback_data.cached_counters->push_back(val);
            }
            if(buf)
                buf->emplace(
                    ROCPROFILER_BUFFER_CATEGORY_COUNTERS, ROCPROFILER_COUNTER_RECORD_VALUE, val);
        }
    }

    // reset the signal to allow another sample to start
    hsa::get_core_table()->hsa_signal_store_relaxed_fn(callback_data.completion, 1);
    return true;
}

/**
 * Setup the agent for handling profiling. This includes setting up the AQL packet,
 * setting up the async handler, and (if this is the first time profiling) setting
 * the profiling register on the queue. This function should only be called when
 * the context is in the LOCKED status.
 */
void
init_callback_data(rocprofiler::counters::agent_callback_data& callback_data,
                   const hsa::AgentCache&                      agent)
{
    // we have already setup this ctx
    if(callback_data.packet) return;

    callback_data.packet = construct_aql_pkt(callback_data.profile);
    callback_data.queue  = agent.profile_queue();

    if(callback_data.completion.handle != 0) return;

    CHECK(hsa::get_core_table() != nullptr);
    CHECK(hsa::get_amd_ext_table() != nullptr);
    CHECK(hsa::get_core_table()->hsa_signal_create_fn != nullptr);
    CHECK(hsa::get_core_table()->hsa_signal_wait_relaxed_fn != nullptr);
    CHECK(hsa::get_core_table()->hsa_signal_store_relaxed_fn != nullptr);
    CHECK(hsa::get_amd_ext_table()->hsa_amd_signal_async_handler_fn != nullptr);

    // Tri-state signal
    //   1: allow next sample to start
    //   0: sample in progress
    //  -1: sample complete
    CHECK_EQ(hsa::get_core_table()->hsa_signal_create_fn(1, 0, nullptr, &callback_data.completion),
             HSA_STATUS_SUCCESS);

    // Signal to manage the startup of the context. Allows us to ensure that
    // the AQL packet we inject with start_context() completes before returning
    CHECK_EQ(
        hsa::get_core_table()->hsa_signal_create_fn(1, 0, nullptr, &callback_data.start_signal),
        HSA_STATUS_SUCCESS);

    // Setup callback
    // NOLINTBEGIN(performance-no-int-to-ptr)
    CHECK_EQ(hsa::get_amd_ext_table()->hsa_amd_signal_async_handler_fn(callback_data.completion,
                                                                       HSA_SIGNAL_CONDITION_LT,
                                                                       0,
                                                                       agent_async_handler,
                                                                       &callback_data),
             HSA_STATUS_SUCCESS);
    // NOLINTEND(performance-no-int-to-ptr)

    // If we do not have a completion handle, this is our first time profiling this agent.
    // Setup our shared data structures.
    static std::unordered_set<hsa_queue_t*> queues_init;
    if(queues_init.find(callback_data.queue) != queues_init.end()) return;
    queues_init.insert(callback_data.queue);

    // Set state of the queue to allow profiling (may not be needed since AQL
    // may do this in the future).
    CHECK(agent.cpu_pool().handle != 0);
    CHECK(agent.get_hsa_agent().handle != 0);

    aql::set_profiler_active_on_queue(
        agent.cpu_pool(), agent.get_hsa_agent(), [&](hsa::rocprofiler_packet pkt) {
            pkt.ext_amd_aql_pm4.completion_signal = callback_data.completion;
            submitPacket(callback_data.queue, (void*) &pkt);
            constexpr auto timeout_hint =
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds{1});
            if(hsa::get_core_table()->hsa_signal_wait_relaxed_fn(callback_data.completion,
                                                                 HSA_SIGNAL_CONDITION_EQ,
                                                                 0,
                                                                 timeout_hint.count(),
                                                                 HSA_WAIT_STATE_ACTIVE) != 0)
            {
                ROCP_FATAL << "Could not set agent to be profiled";
            }
            hsa::get_core_table()->hsa_signal_store_relaxed_fn(callback_data.completion, 1);
        });
}
}  // namespace

/**
 * Read the previously started profiling registers for each agent. Injects both the read packet
 * and the stop packet (a sidestep to the AQL issues) into the queue and optionally waits for the
 * return. A small note here is that this function should avoid allocations to be signal safe.
 *
 * Special Case: If the counters the user requests are purely constants, skip packet injection
 * and trigger the async handler manually.
 */
rocprofiler_status_t
read_agent_ctx(const context::context*                    ctx,
               rocprofiler_user_data_t                    user_data,
               rocprofiler_counter_flag_t                 flags,
               std::vector<rocprofiler_counter_record_t>* out_counters)
{
    rocprofiler_status_t status = ROCPROFILER_STATUS_SUCCESS;
    if(!ctx->device_counter_collection)
    {
        ROCP_ERROR << fmt::format("Context {} has no agent counter collection", ctx->context_idx);
        return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;
    }

    auto& agent_ctx = *ctx->device_counter_collection;

    // If we have not initiualized HSA yet, nothing to read, return;
    if(hsa_inited().load() == false)
    {
        return ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED;
    }

    // Set the state to LOCKED to prevent other calls to start/stop/read.
    auto expected = rocprofiler::context::device_counting_service::state::ENABLED;
    if(!agent_ctx.status.compare_exchange_strong(
           expected, rocprofiler::context::device_counting_service::state::LOCKED))
    {
        return ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR;
    }

    for(auto& callback_data : agent_ctx.agent_data)
    {
        auto wait_if_sync = [&]() {
            if((flags & ROCPROFILER_COUNTER_FLAG_ASYNC) == 0)
            {
                // Wait for any inprogress samples to complete before returning
                hsa::get_core_table()->hsa_signal_wait_relaxed_fn(callback_data.completion,
                                                                  HSA_SIGNAL_CONDITION_EQ,
                                                                  1,
                                                                  UINT64_MAX,
                                                                  HSA_WAIT_STATE_ACTIVE);
            }
        };

        if(!callback_data.profile || !callback_data.set_profile) continue;
        const auto* agent = agent::get_agent_cache(callback_data.profile->agent);

        // If the agent no longer exists or we don't have a profile queue, reading is an error
        if(!agent || !agent->profile_queue())
        {
            status = ROCPROFILER_STATUS_ERROR;
            break;
        }

        // No AQL packet, nothing to do here.
        if(!callback_data.packet) continue;

        wait_if_sync();

        if((flags & ROCPROFILER_COUNTER_FLAG_ASYNC) == 0)
            callback_data.cached_counters = out_counters;

        // If we have no hardware counters but a packet. The caller is expecting
        // non-hardware based counter values to be returned. We can skip packet injection
        // and trigger the async handler directly
        if(callback_data.profile->reqired_hw_counters.empty())
        {
            callback_data.user_data = user_data;
            hsa::get_core_table()->hsa_signal_store_relaxed_fn(callback_data.completion, -1);
            wait_if_sync();
            continue;
        }

        ROCP_TRACE << fmt::format("Agent Info for Running Counter: Name = {}, XCC = {}, "
                                  "SE = {}, CU = {}, SIMD = {}",
                                  agent->get_rocp_agent()->name,
                                  agent->get_rocp_agent()->num_xcc,
                                  agent->get_rocp_agent()->num_shader_banks,
                                  agent->get_rocp_agent()->cu_count,
                                  agent->get_rocp_agent()->simd_arrays_per_engine);

        // Submit the read packet to the queue
        submitPacket(agent->profile_queue(), &callback_data.packet->packets.read_packet);

        // Submit a barrier packet. This is needed to flush hardware caches. Without this
        // the read packet may not have the correct data.
        rocprofiler::hsa::rocprofiler_packet barrier{};
        barrier.barrier_and.header            = header_pkt(HSA_PACKET_TYPE_BARRIER_AND);
        barrier.barrier_and.completion_signal = callback_data.completion;
        hsa::get_core_table()->hsa_signal_store_relaxed_fn(callback_data.completion, 0);
        callback_data.user_data = user_data;
        submitPacket(agent->profile_queue(), &barrier.barrier_and);
        wait_if_sync();
        if((flags & ROCPROFILER_COUNTER_FLAG_ASYNC) == 0) callback_data.cached_counters = nullptr;
    }

    agent_ctx.status.exchange(rocprofiler::context::device_counting_service::state::ENABLED);
    return status;
}

/**
 * Start the agent profiling for the context. For each agent that this context is
 * enabled for, we will call the tool to get the profile config. This config will
 * will then be used to generate the AQL packet (if it differs from the previous
 * profile used). init_callback_data does this initialization. If a tool does not
 * supply a profile, we skip this agent. We then submit the start packet to the
 * profile queue. This call is synchronous.
 *
 * Special Case: if constants are the only counters being collected, we skip
 * packet injection.
 */
rocprofiler_status_t
start_agent_ctx(const context::context* ctx)
{
    auto status = ROCPROFILER_STATUS_SUCCESS;
    if(!ctx->device_counter_collection)
    {
        return status;
    }

    auto& agent_ctx = *ctx->device_counter_collection;

    if(hsa_inited().load() == false)
    {
        return ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED;
    }

    // Set the state to LOCKED to prevent other calls to start/stop/read.
    auto expected = rocprofiler::context::device_counting_service::state::DISABLED;
    if(!agent_ctx.status.compare_exchange_strong(
           expected, rocprofiler::context::device_counting_service::state::LOCKED))
    {
        return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;
    }

    for(auto& callback_data : agent_ctx.agent_data)
    {
        const auto* agent = agent::get_agent_cache(agent::get_agent(callback_data.agent_id));

        if(!agent)
        {
            ROCP_ERROR << "No agent found for context: " << ctx->context_idx;
            status = ROCPROFILER_STATUS_ERROR;
            break;
        }

        // But if we have an agent cache, we need a profile queue.
        if(!agent->profile_queue())
        {
            ROCP_ERROR << "No profile queue found for context: " << ctx->context_idx;
            status = ROCPROFILER_STATUS_ERROR_NO_PROFILE_QUEUE;
            break;
        }

        callback_data.set_profile = false;

        // Ask the tool what profile we should use for this agent
        callback_data.cb(
            {.handle = ctx->context_idx},
            callback_data.agent_id,
            [](rocprofiler_context_id_t        context_id,
               rocprofiler_counter_config_id_t config_id) -> rocprofiler_status_t {
                auto* cb_ctx = rocprofiler::context::get_mutable_registered_context(context_id);
                if(!cb_ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;

                auto config = rocprofiler::counters::get_counter_config(config_id);
                if(!config) return ROCPROFILER_STATUS_ERROR_PROFILE_NOT_FOUND;

                if(!cb_ctx->device_counter_collection)
                {
                    return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;
                }

                // Only allow profiles to be set in the locked state
                if(cb_ctx->device_counter_collection->status.load() !=
                   rocprofiler::context::device_counting_service::state::LOCKED)
                {
                    return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;
                }

                for(auto& agent_data : cb_ctx->device_counter_collection->agent_data)
                {
                    // Find the agent that this profile is for and set it.
                    if(agent_data.agent_id.handle == config->agent->id.handle)
                    {
                        // If the profile config has changed, reset the packet
                        // and swap the profile.
                        if(agent_data.profile != config)
                        {
                            agent_data.profile = config;
                            agent_data.packet.reset();
                        }
                        // A flag to state that we set a profile
                        agent_data.set_profile = true;
                        return ROCPROFILER_STATUS_SUCCESS;
                    }
                }

                return ROCPROFILER_STATUS_ERROR_AGENT_MISMATCH;
            },
            callback_data.callback_data.ptr);

        // If we did not set a profile, we have nothing to do.
        if(!callback_data.set_profile)
        {
            callback_data.packet.reset();
            continue;
        }

        CHECK(callback_data.profile);

        // Generate necessary structures in the context (packet gen, etc) to process
        // this packet.
        init_callback_data(callback_data, *agent);

        // No hardware counters were actually asked for (i.e. all constants)
        if(callback_data.profile->reqired_hw_counters.empty())
        {
            continue;
        }

        callback_data.packet->packets.start_packet.completion_signal = callback_data.start_signal;
        hsa::get_core_table()->hsa_signal_store_relaxed_fn(callback_data.start_signal, 1);
        submitPacket(agent->profile_queue(), &callback_data.packet->packets.start_packet);

        // Wait for startup to finish before continuing
        hsa::get_core_table()->hsa_signal_wait_relaxed_fn(callback_data.start_signal,
                                                          HSA_SIGNAL_CONDITION_EQ,
                                                          0,
                                                          UINT64_MAX,
                                                          HSA_WAIT_STATE_ACTIVE);
    }

    agent_ctx.status.exchange(rocprofiler::context::device_counting_service::state::ENABLED);
    return status;
}

/**
 * Issue the stop packet for all active agents in this context. This call is
 * synchronous.
 *
 * Special Case: if no hardware counters are being collected, skip issuing the
 * stop packet.
 */
rocprofiler_status_t
stop_agent_ctx(const context::context* ctx)
{
    auto status = ROCPROFILER_STATUS_SUCCESS;
    if(!ctx->device_counter_collection)
    {
        return status;
    }

    auto& agent_ctx = *ctx->device_counter_collection;

    if(hsa_inited().load() == false)
    {
        return ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED;
    }

    auto expected = rocprofiler::context::device_counting_service::state::ENABLED;
    if(!agent_ctx.status.compare_exchange_strong(
           expected, rocprofiler::context::device_counting_service::state::LOCKED))
    {
        // Status is already stopped or being enabled elsewhere.
        return ROCPROFILER_STATUS_SUCCESS;
    }

    for(auto& callback_data : agent_ctx.agent_data)
    {
        if(!callback_data.packet) continue;

        const auto* agent = agent::get_agent_cache(callback_data.profile->agent);
        if(!agent || !agent->profile_queue()) continue;

        if(!callback_data.profile->reqired_hw_counters.empty())
        {
            // Remove when AQL is updated to not require stop to be called first
            submitPacket(agent->profile_queue(), &callback_data.packet->packets.stop_packet);
        }

        // Wait for the stop packet to complete
        hsa::get_core_table()->hsa_signal_wait_relaxed_fn(callback_data.completion,
                                                          HSA_SIGNAL_CONDITION_EQ,
                                                          1,
                                                          UINT64_MAX,
                                                          HSA_WAIT_STATE_ACTIVE);
    }

    agent_ctx.status.exchange(rocprofiler::context::device_counting_service::state::DISABLED);
    return status;
}

// Stop all contexts and prevent any further requests to start/stop/read.
// Waits until any current operation is complete before exiting.
rocprofiler_status_t
device_counting_service_finalize()
{
    for(auto& ctx : context::get_registered_contexts())
    {
        std::vector<rocprofiler::context::device_counting_service::state> expected = {
            rocprofiler::context::device_counting_service::state::DISABLED,
            rocprofiler::context::device_counting_service::state::ENABLED,
            rocprofiler::context::device_counting_service::state::EXIT};
        if(!ctx->device_counter_collection) continue;
        while(!ctx->device_counter_collection->status.compare_exchange_strong(
                  expected[0], rocprofiler::context::device_counting_service::state::EXIT) &&
              !ctx->device_counter_collection->status.compare_exchange_strong(
                  expected[1], rocprofiler::context::device_counting_service::state::EXIT) &&
              !ctx->device_counter_collection->status.compare_exchange_strong(
                  expected[2], rocprofiler::context::device_counting_service::state::EXIT))
        {
            // Note: Compare Exchange can modify expected even if the exchange fails
            expected = {rocprofiler::context::device_counting_service::state::DISABLED,
                        rocprofiler::context::device_counting_service::state::ENABLED,
                        rocprofiler::context::device_counting_service::state::EXIT};
        };
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

// If we have ctx's that were started before HSA was initialized, we need to
// actually start those contexts now that we have an HSA instance.
rocprofiler_status_t
device_counting_service_hsa_registration()
{
    hsa_inited().store(true);

    for(auto& ctx : context::get_active_contexts())
    {
        if(!ctx->device_counter_collection) continue;
        start_agent_ctx(ctx);
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

agent_callback_data::~agent_callback_data()
{
    if(completion.handle != 0) hsa::get_core_table()->hsa_signal_destroy_fn(completion);
}
}  // namespace counters
}  // namespace rocprofiler
