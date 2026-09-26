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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/spm/queue_hooks.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_hooks/client_ids.hpp"
#include "lib/rocprofiler-sdk/spm/dispatch_handlers.hpp"

namespace rocprofiler
{
namespace spm
{
namespace
{
auto
spm_contexts_filter()
{
    return [](const context::context* ctx) -> bool { return ctx && ctx->dispatch_spm != nullptr; };
}
}  // namespace

void
write_hook(const hsa::Queue&                                        queue,
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

    auto active = context::get_active_contexts(spm_contexts_filter());
    for(const auto* ctx : active)
    {
        if(!ctx->dispatch_spm->collects_on(agent_id)) continue;

        for(auto& cb : ctx->dispatch_spm->callbacks)
        {
            auto [packet, bSerial] = pre_kernel_call(ctx,
                                                     cb,
                                                     queue,
                                                     kernel_packet,
                                                     kernel_id,
                                                     dispatch_id,
                                                     user_data,
                                                     ext_corr_ids,
                                                     correlation_id);
            if(packet) inst_pkt.emplace_back(std::move(packet), hsa::queue_hooks::SPM_CLIENT_ID);
            is_serialized |= bSerial;
        }
    }
}

void
signal_completion_hook(const hsa::Queue& /*queue*/,
                       const hsa::rocprofiler_packet& /*kernel_packet*/,
                       std::shared_ptr<hsa::queue_info_session_t>& session,
                       hsa::packet_data_t& /*packet*/,
                       hsa::inst_pkt_t&                inst_pkt,
                       kernel_dispatch::profiling_time dispatch_time)
{
    bool has_spm_packets = false;
    for(const auto& tagged_pkt : inst_pkt)
    {
        if(tagged_pkt.second == hsa::queue_hooks::SPM_CLIENT_ID)
        {
            has_spm_packets = true;
            break;
        }
    }
    if(!has_spm_packets) return;

    // Route by packet provenance, not current activeness: post_kernel_call self-filters via
    // packet_return_map, so in-flight dispatches still complete after stop_context removes the
    // context from the active list. Without this, a dispatch in flight at stop skips the
    // DISPATCH_END record, kfd_stop() and the barrier-signal cleanup, and leaks its map entry.
    //
    // Iterating registered contexts widens the candidate set, but routing stays per-origin because
    // each context's post_kernel_call only claims packets present in its own packet_return_map.
    // Disjoint agent sets allow multiple concurrent SPM contexts when scoped via set_agents().
    auto contexts = context::get_registered_contexts(spm_contexts_filter());
    for(const auto* ctx : contexts)
    {
        for(auto& cb : ctx->dispatch_spm->callbacks)
        {
            post_kernel_call(ctx, cb, session, inst_pkt, dispatch_time);
        }
    }
}

bool
is_any_active()
{
    return !context::get_active_contexts(spm_contexts_filter()).empty();
}
}  // namespace spm
}  // namespace rocprofiler
