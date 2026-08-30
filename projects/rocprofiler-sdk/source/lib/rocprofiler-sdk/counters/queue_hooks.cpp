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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/counters/queue_hooks.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/dispatch_handlers.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_hooks/client_ids.hpp"

namespace rocprofiler
{
namespace counters
{
namespace
{
auto
counter_contexts_filter()
{
    return [](const context::context* ctx) -> bool {
        return ctx && ctx->dispatch_counter_collection != nullptr;
    };
}
}  // namespace

void
kernel_dispatch_phase_enter_hook(
    const hsa::Queue&                                        queue,
    const hsa::rocprofiler_packet&                           kernel_packet,
    rocprofiler_kernel_id_t                                  kernel_id,
    rocprofiler_dispatch_id_t                                dispatch_id,
    rocprofiler_user_data_t*                                 user_data,
    const hsa::queue_info_session_t::external_corr_id_map_t& ext_corr_ids,
    const context::correlation_id*                           correlation_id,
    hsa::inst_pkt_t&                                         inst_pkt,
    bool&                                                    is_serialized)
{
    const auto agent_id = CHECK_NOTNULL(queue.get_agent().get_rocp_agent())->id;

    auto active = context::get_active_contexts(counter_contexts_filter());
    for(const auto* ctx : active)
    {
        // Skip contexts that do not collect on this dispatch's agent. This is what keeps a
        // GPU-1-only context from serializing GPU-0: queue_cb is what returns serialize=true,
        // including on its disabled path, so filtering has to happen before the call rather
        // than inside it.
        if(!ctx->dispatch_counter_collection->collects_on(agent_id)) continue;

        for(auto& cb : ctx->dispatch_counter_collection->callbacks)
        {
            auto [packet, bSerial] = queue_cb(ctx,
                                              cb,
                                              queue,
                                              kernel_packet,
                                              kernel_id,
                                              dispatch_id,
                                              user_data,
                                              ext_corr_ids,
                                              correlation_id);
            if(packet)
                inst_pkt.emplace_back(std::move(packet), hsa::queue_hooks::COUNTERS_CLIENT_ID);
            is_serialized |= bSerial;
        }
    }
}

void
kernel_dispatch_phase_exit_hook(const hsa::Queue& /*queue*/,
                                const hsa::rocprofiler_packet& /*kernel_packet*/,
                                std::shared_ptr<hsa::queue_info_session_t>& session,
                                hsa::packet_data_t&                         packet,
                                hsa::inst_pkt_t&                            inst_pkt,
                                kernel_dispatch::profiling_time             dispatch_time)
{
    bool has_counter_packets = false;
    for(const auto& tagged_pkt : inst_pkt)
    {
        if(tagged_pkt.second == hsa::queue_hooks::COUNTERS_CLIENT_ID)
        {
            has_counter_packets = true;
            break;
        }
    }
    if(!has_counter_packets) return;

    // Route by packet provenance, not current activeness: completed_cb self-filters via
    // packet_return_map, so in-flight dispatches still complete after stop_context removes the
    // context from the active list. This is what guarantees a completion that arrives is delivered;
    // the drain in stop_context is what bounds when completions arrive. Kernel replay needs the
    // same property, since each pass completes separately.
    auto contexts = context::get_registered_contexts(counter_contexts_filter());
    for(const auto* ctx : contexts)
    {
        for(auto& cb : ctx->dispatch_counter_collection->callbacks)
        {
            completed_cb(ctx, cb, session, packet, inst_pkt, dispatch_time);
        }
    }
}

bool
is_any_active()
{
    return !context::get_active_contexts(counter_contexts_filter()).empty();
}
}  // namespace counters
}  // namespace rocprofiler
