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

#include "lib/rocprofiler-sdk/spm/device_counting.hpp"
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/fwd.h>
#include "lib/common/logging.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/ioctl.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/spm/core.hpp"

#include <string.h>
#include <thread>

namespace rocprofiler
{
namespace SPM
{
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
    memcpy(&queue_slot[1], &slot_data[1], sizeof(hsa_ext_amd_aql_pm4_packet_t) - sizeof(uint32_t));
    std::atomic<uint32_t>* header_atomic_ptr =
        reinterpret_cast<std::atomic<uint32_t>*>(&queue_slot[0]);
    header_atomic_ptr->store(slot_data[0], std::memory_order_release);

    // ringdoor bell
    hsa::get_core_table()->hsa_signal_store_relaxed_fn(queue->doorbell_signal, write_idx);

    return write_idx;
}

namespace
{
constexpr auto rocprofiler_context_none = ROCPROFILER_CONTEXT_NONE;
}
std::atomic<bool>&
hsa_inited()
{
    static std::atomic<bool> inited{false};
    return inited;
}

namespace
{
/**
 * Construct the SPM AQL packet for the given profile.
 * Note this function is not thread safe and is only called from
 * init_callback_data which can only be called when the context is in the LOCKED state
 * with only a single thread active.
 */
std::unique_ptr<hsa::SPMPacket>
construct_aql_pkt(std::shared_ptr<spm::spm_counter_config>& profile)
{
    auto pkt = rocprofiler::aql::spm_construct_packet(
        CHECK_NOTNULL(profile->agent)->id,
        std::vector<counters::Metric>{profile->metrics.begin(), profile->metrics.end()},
        profile->spm_parameters);
    if(!pkt)
    {
        ROCP_ERROR << "SPM device counting packet construction failed";
        return nullptr;
    }

    return pkt;
}

/**
 * Setup the agent for handling profiling. This includes setting up the AQL packet,
 * setting up the async handler, and (if this is the first time profiling) setting
 * the profiling register on the queue. This function should only be called when
 * the context is in the LOCKED status.
 */
void
init_callback_data(rocprofiler::SPM::spm_agent_callback_data& callback_data,
                   const hsa::AgentCache&                     agent)
{
    // we have already setup this ctx
    if(callback_data.packet) return;

    callback_data.packet = std::move(construct_aql_pkt(callback_data.profile));

    callback_data.queue             = agent.profile_queue();
    callback_data.packet->cb.buffer = callback_data.buffer;

    if(callback_data.stop_signal.handle != 0) return;

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
    CHECK_EQ(hsa::get_core_table()->hsa_signal_create_fn(1, 0, nullptr, &callback_data.stop_signal),
             HSA_STATUS_SUCCESS);

    // Signal to manage the startup of the context. Allows us to ensure that
    // the AQL packet we inject with start_context() completes before returning
    CHECK_EQ(
        hsa::get_core_table()->hsa_signal_create_fn(1, 0, nullptr, &callback_data.start_signal),
        HSA_STATUS_SUCCESS);

    CHECK(agent.cpu_pool().handle != 0);
    CHECK(agent.get_hsa_agent().handle != 0);
}
}  // namespace

rocprofiler_status_t
spm_start_agent_ctx(const context::context* ctx)
{
    auto status = ROCPROFILER_STATUS_SUCCESS;
    if(!ctx->device_spm)
    {
        return status;
    }

    auto& agent_ctx = *ctx->device_spm;

    if(hsa_inited().load() == false)
    {
        return ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED;
    }

    // Set the state to LOCKED to prevent other calls to start/stop/read.
    auto expected = rocprofiler::context::spm_device_counting_service::state::DISABLED;
    if(!agent_ctx.status.compare_exchange_strong(
           expected, rocprofiler::context::spm_device_counting_service::state::LOCKED))
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

        if(hsa::use_ondemand_queue())
        {
            agent->init_device_counting_service_queue(*hsa::get_core_table(),
                                                      *hsa::get_amd_ext_table());
        }

        // But if we have an agent cache, we need a profile queue.
        if(!agent->profile_queue())
        {
            ROCP_ERROR << "No profile queue found for context: " << ctx->context_idx;
            status = ROCPROFILER_STATUS_ERROR_NO_PROFILE_QUEUE;
            break;
        }

        // Lock the device for profiling
        if(counters::counter_collection_has_device_lock())
        {
            counters::counter_collection_device_lock(agent->get_rocp_agent(), true);
        }

        counters::counter_collection_ptl_disable(agent->get_rocp_agent());

        callback_data.set_profile = false;

        // Ask the tool what profile we should use for this agent
        callback_data.cb(
            {.handle = ctx->context_idx},
            callback_data.agent_id,
            [](rocprofiler_context_id_t        context_id,
               rocprofiler_counter_config_id_t config_id) -> rocprofiler_status_t {
                auto* cb_ctx = rocprofiler::context::get_mutable_registered_context(context_id);
                if(!cb_ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;

                auto config = rocprofiler::spm::get_spm_counter_config(config_id);
                if(!config) return ROCPROFILER_STATUS_ERROR_PROFILE_NOT_FOUND;

                if(!cb_ctx->device_spm)
                {
                    return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;
                }

                // Only allow profiles to be set in the locked state
                if(cb_ctx->device_spm->status.load() !=
                   rocprofiler::context::spm_device_counting_service::state::LOCKED)
                {
                    return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;
                }

                for(auto& agent_data : cb_ctx->device_spm->agent_data)
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

        auto* spm_pkt = CHECK_NOTNULL(callback_data.packet.get());

        // Populate start/stop packet vectors
        spm_pkt->clear();
        spm_pkt->cb.buffer = callback_data.buffer;
        spm_pkt->populate_before();
        spm_pkt->populate_after();

        if(!spm_pkt->kfd_start())
        {
            ROCP_ERROR << "SPM KFD start failed for device counting";
            status = ROCPROFILER_STATUS_ERROR;
            break;
        }

        // Set completion signal on start packet
        spm_pkt->profile.packets.start_packet.completion_signal = callback_data.start_signal;
        hsa::get_core_table()->hsa_signal_store_screlease_fn(callback_data.start_signal, 1);

        // Submit start packet to profile queue
        submitPacket(callback_data.queue, &spm_pkt->profile.packets.start_packet);

        // Wait for start packet to complete
        while(hsa::get_core_table()->hsa_signal_wait_scacquire_fn(callback_data.start_signal,
                                                                  HSA_SIGNAL_CONDITION_EQ,
                                                                  0,
                                                                  UINT64_MAX,
                                                                  HSA_WAIT_STATE_BLOCKED) != 0)
        {};
    }

    agent_ctx.status.store(rocprofiler::context::spm_device_counting_service::state::ENABLED);
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
spm_stop_agent_ctx(const context::context* ctx)
{
    auto status = ROCPROFILER_STATUS_SUCCESS;
    if(!ctx->device_spm)
    {
        return status;
    }

    auto& agent_ctx = *ctx->device_spm;

    if(hsa_inited().load() == false)
    {
        return ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED;
    }

    auto expected = rocprofiler::context::spm_device_counting_service::state::ENABLED;
    if(!agent_ctx.status.compare_exchange_strong(
           expected, rocprofiler::context::spm_device_counting_service::state::LOCKED))
    {
        // Status is already stopped or being enabled elsewhere.
        return ROCPROFILER_STATUS_SUCCESS;
    }

    for(auto& callback_data : agent_ctx.agent_data)
    {
        if(!callback_data.packet) continue;

        const auto* agent = agent::get_agent_cache(callback_data.profile->agent);
        if(!agent || !agent->profile_queue()) continue;

        callback_data.packet->profile.packets.stop_packet.completion_signal =
            callback_data.stop_signal;
        hsa::get_core_table()->hsa_signal_store_screlease_fn(callback_data.stop_signal, 1);
        submitPacket(callback_data.queue, &callback_data.packet->profile.packets.stop_packet);
        while(hsa::get_core_table()->hsa_signal_wait_scacquire_fn(callback_data.stop_signal,
                                                                  HSA_SIGNAL_CONDITION_EQ,
                                                                  0,
                                                                  UINT64_MAX,
                                                                  HSA_WAIT_STATE_BLOCKED) != 0)
        {};

        callback_data.packet->kfd_stop();

        counters::counter_collection_ptl_enable(agent->get_rocp_agent());
        counters::counter_collection_device_unlock(agent->get_rocp_agent());

        if(hsa::use_ondemand_queue())
        {
            if(callback_data.stop_signal.handle != 0)
            {
                hsa::get_core_table()->hsa_signal_destroy_fn(callback_data.stop_signal);
                callback_data.stop_signal.handle = 0;
            }
            if(callback_data.start_signal.handle != 0)
            {
                hsa::get_core_table()->hsa_signal_destroy_fn(callback_data.start_signal);
                callback_data.start_signal.handle = 0;
            }
            callback_data.packet.reset();
            callback_data.queue = nullptr;
            agent->destroy_device_counting_service_queue();
        }
    }

    agent_ctx.status.exchange(rocprofiler::context::spm_device_counting_service::state::DISABLED);
    return status;
}

// Stop all contexts and prevent any further requests to start/stop/read.
// Unlike counters device counting, SPM has continuous hardware streaming
// that must be explicitly stopped before marking EXIT.
rocprofiler_status_t
spm_device_counting_service_finalize()
{
    for(auto& ctx : context::get_registered_contexts())
    {
        if(!ctx->device_spm) continue;
        spm_stop_agent_ctx(ctx);

        std::vector<rocprofiler::context::spm_device_counting_service::state> expected = {
            rocprofiler::context::spm_device_counting_service::state::DISABLED,
            rocprofiler::context::spm_device_counting_service::state::ENABLED,
            rocprofiler::context::spm_device_counting_service::state::EXIT};
        while(!ctx->device_spm->status.compare_exchange_strong(
                  expected[0], rocprofiler::context::spm_device_counting_service::state::EXIT) &&
              !ctx->device_spm->status.compare_exchange_strong(
                  expected[1], rocprofiler::context::spm_device_counting_service::state::EXIT) &&
              !ctx->device_spm->status.compare_exchange_strong(
                  expected[2], rocprofiler::context::spm_device_counting_service::state::EXIT))
        {
            expected = {rocprofiler::context::spm_device_counting_service::state::DISABLED,
                        rocprofiler::context::spm_device_counting_service::state::ENABLED,
                        rocprofiler::context::spm_device_counting_service::state::EXIT};
        };
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
spm_device_counting_service_hsa_registration()
{
    hsa_inited().store(true);

    for(auto& ctx : context::get_active_contexts())
    {
        if(!ctx->device_spm) continue;
        spm_start_agent_ctx(ctx);
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

spm_agent_callback_data::~spm_agent_callback_data()
{
    if(start_signal.handle != 0) hsa::get_core_table()->hsa_signal_destroy_fn(start_signal);
    if(stop_signal.handle != 0) hsa::get_core_table()->hsa_signal_destroy_fn(stop_signal);
}

}  // namespace SPM
}  // namespace rocprofiler