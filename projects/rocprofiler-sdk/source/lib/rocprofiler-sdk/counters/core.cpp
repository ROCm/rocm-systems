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

#include "lib/rocprofiler-sdk/counters/core.hpp"

#include "lib/common/container/small_vector.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/dispatch_handlers.hpp"
#include "lib/rocprofiler-sdk/counters/sample_processing.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/fwd.h>

namespace rocprofiler
{
namespace counters
{
rocprofiler_status_t
counter_callback_info::setup_counter_config(std::shared_ptr<counter_config>& profile)
{
    if(profile->pkt_generator || !profile->reqired_hw_counters.empty())
    {
        return ROCPROFILER_STATUS_SUCCESS;
    }

    // Sets up the packet generator for the profile. This must be delayed until after HSA is loaded.
    // This call needs to be thread protected in that only one thread must be setting up profile at
    // the same time.

    auto& config     = *profile;
    auto  agent_name = std::string(config.agent->name);
    for(const auto& metric : config.metrics)
    {
        const auto asts = get_ast_map();
        auto       req_counters =
            get_required_hardware_counters(asts->arch_to_counter_asts, agent_name, metric);

        if(!req_counters)
        {
            ROCP_ERROR << fmt::format("Could not find counter {}", metric.name());
            return ROCPROFILER_STATUS_ERROR_PROFILE_COUNTER_NOT_FOUND;
        }

        // Special metrics are those that are not hw counters but other
        // constants like MAX_WAVE_SIZE
        for(const auto& req_metric : *req_counters)
        {
            if(req_metric.constant().empty())
            {
                config.reqired_hw_counters.insert(req_metric);
            }
            else
            {
                config.required_special_counters.insert(req_metric);
            }
        }

        const auto* agent_map =
            rocprofiler::common::get_val(asts->arch_to_counter_asts, agent_name);
        if(!agent_map)
        {
            ROCP_ERROR << fmt::format("Could not build AST for {}", agent_name);
            return ROCPROFILER_STATUS_ERROR_AST_GENERATION_FAILED;
        }

        const auto* counter_ast = rocprofiler::common::get_val(*agent_map, metric.name());
        if(!counter_ast)
        {
            ROCP_ERROR << fmt::format("Could not find AST for {}", metric.name());
            return ROCPROFILER_STATUS_ERROR_AST_NOT_FOUND;
        }
        config.asts.push_back(*counter_ast);

        try
        {
            config.asts.back().set_dimensions(config.agent->id);
        } catch(std::runtime_error& e)
        {
            ROCP_ERROR << metric.name() << " has improper dimensions"
                       << " " << e.what();
            return ROCPROFILER_STATUS_ERROR_AST_NOT_FOUND;
        }
    }

    profile->pkt_generator = std::make_unique<rocprofiler::aql::CounterPacketConstruct>(
        config.agent->id,
        std::vector<counters::Metric>{profile->reqired_hw_counters.begin(),
                                      profile->reqired_hw_counters.end()});
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
counter_callback_info::get_packet(std::unique_ptr<rocprofiler::hsa::AQLPacket>& ret_pkt,
                                  std::shared_ptr<counter_config>&              profile)
{
    rocprofiler_status_t status;
    // Check packet cache
    profile->packets.wlock([&](auto& pkt_vector) {
        status = counter_callback_info::setup_counter_config(profile);
        if(!pkt_vector.empty() && status == ROCPROFILER_STATUS_SUCCESS)
        {
            ret_pkt = std::move(pkt_vector.back());
            pkt_vector.pop_back();
        }
    });

    if(status != ROCPROFILER_STATUS_SUCCESS) return status;
    if(!ret_pkt)
    {
        // If we do not have a packet in the cache, create one.
        ret_pkt = profile->pkt_generator->construct_packet(
            CHECK_NOTNULL(hsa::get_queue_controller())->get_core_table(),
            CHECK_NOTNULL(hsa::get_queue_controller())->get_ext_table());
    }

    ret_pkt->clear();
    packet_return_map.wlock([&](auto& data) { data.emplace(ret_pkt.get(), profile); });

    return ROCPROFILER_STATUS_SUCCESS;
}

void
start_context(const context::context* ctx)
{
    if(!ctx || !ctx->dispatch_counter_collection) return;

    auto* controller = hsa::get_queue_controller();

    bool already_enabled = true;
    // Scope serialization to the agents this context collects on. An empty set still means
    // every agent, so an unrestricted context serializes the whole machine as before.
    CHECK_NOTNULL(controller)->enable_serialization(ctx->dispatch_counter_collection->agents);
    ctx->dispatch_counter_collection->enabled.wlock([&](auto& enabled) {
        if(enabled) return;
        already_enabled = false;
        enabled         = true;
    });

    if(!already_enabled)
    {
        // Counter collection no longer registers a per-queue callback with the queue
        // controller; the HSA write interceptor calls counters::kernel_dispatch_phase_enter_hook /
        // kernel_dispatch_phase_exit_hook directly (see hsa/queue.cpp). Keep the callback thread.
        callback_thread_start();
    }
}

void
stop_context(const context::context* ctx)
{
    if(!ctx || !ctx->dispatch_counter_collection) return;

    auto* controller = hsa::get_queue_controller();

    ctx->dispatch_counter_collection->enabled.wlock([&](auto& enabled) {
        if(!enabled) return;
        enabled = false;
    });

    if(controller)
    {
        // Drain in-flight dispatches before anything else is torn down. The review of #8891
        // accepted provenance-based completion routing on the condition that the callback thread
        // and the counter_callback_info objects stay alive until in-flight dispatches drain, so the
        // drain is what makes that condition hold literally rather than by argument.
        //
        // context::stop_context calls this function while the context is still in the active list,
        // which is what keeps the service visible for the duration of the drain: the enter hook
        // still reaches queue_cb, and queue_cb's disabled path returns serialize=true so the
        // serialized->unserialized transition stays coordinated until disable_serialization() runs
        // below. That ordering is why no separate "draining" flag is needed -- but it only works
        // because the drain happens here, before the slot is cleared.
        hsa::queue_controller_sync();
        controller->disable_serialization(ctx->dispatch_counter_collection->agents);
        // No per-queue callback to remove; counters::kernel_dispatch_phase_enter_hook no-ops once
        // dispatch_counter_collection is disabled above.
    }

    // After the drain. consumer_thread_t::exit() also waits until its queue is empty before
    // joining, and add() consumes inline once the thread is gone, so a late completion is still
    // processed -- but that is a backstop, not the guarantee the review asked for.
    callback_thread_stop();
}

rocprofiler_status_t
set_dispatch_agents(rocprofiler_context_id_t      context_id,
                    const rocprofiler_agent_id_t* agents,
                    size_t                        num_agents)
{
    if(num_agents > 0 && agents == nullptr) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    auto* ctx_p = context::get_mutable_registered_context(context_id);
    if(!ctx_p) return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;
    if(!ctx_p->dispatch_counter_collection) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    // The agent set is read without a lock on the dispatch path and is what scopes
    // serialization at start, so it may only change while the context is stopped. Hold the
    // contexts mutex so this check and the assignment are atomic with start_context: a
    // concurrent start could otherwise activate the context after the scan (using the old
    // set for serialization) and then race with the unordered-set assignment below while
    // dispatch hooks read it.
    auto _lk = std::unique_lock<std::mutex>{context::get_contexts_mutex()};
    for(const auto* itr : context::get_active_contexts())
    {
        if(itr && itr->context_idx == ctx_p->context_idx)
            return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;
    }

    auto selected = std::unordered_set<rocprofiler_agent_id_t>{};
    selected.reserve(num_agents);
    for(size_t i = 0; i < num_agents; ++i)
    {
        const auto* agent = rocprofiler::agent::get_agent(agents[i]);
        if(!agent || agent->type != ROCPROFILER_AGENT_TYPE_GPU)
            return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;
        selected.emplace(agents[i]);
    }

    ctx_p->dispatch_counter_collection->agents = std::move(selected);

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
configure_agent_collection(rocprofiler_context_id_t                 context_id,
                           rocprofiler_buffer_id_t                  buffer_id,
                           rocprofiler_agent_id_t                   agent_id,
                           rocprofiler_device_counting_service_cb_t cb,
                           void*                                    user_data)
{
    return get_controller().configure_agent_collection(
        context_id, buffer_id, agent_id, cb, user_data);
}

rocprofiler_status_t
configure_buffered_dispatch(rocprofiler_context_id_t                   context_id,
                            rocprofiler_buffer_id_t                    buffer,
                            rocprofiler_dispatch_counting_service_cb_t callback,
                            void*                                      callback_args)
{
    CHECK_NE(buffer.handle, 0ULL);
    return get_controller().configure_dispatch(
        context_id, buffer, callback, callback_args, nullptr, nullptr);
}

rocprofiler_status_t
configure_callback_dispatch(rocprofiler_context_id_t                   context_id,
                            rocprofiler_dispatch_counting_service_cb_t callback,
                            void*                                      callback_data_args,
                            rocprofiler_dispatch_counting_record_cb_t  record_callback,
                            void*                                      record_callback_args)
{
    return get_controller().configure_dispatch(context_id,
                                               {.handle = 0},
                                               callback,
                                               callback_data_args,
                                               record_callback,
                                               record_callback_args);
}

}  // namespace counters
}  // namespace rocprofiler
