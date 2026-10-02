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

#include "lib/rocprofiler-sdk/thread_trace/queue_hooks.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_hooks/client_ids.hpp"
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"

namespace rocprofiler
{
namespace thread_trace
{
namespace
{
auto
thread_trace_contexts_filter()
{
    return [](const context::context* ctx) -> bool {
        return ctx && ctx->dispatch_thread_trace != nullptr;
    };
}
}  // namespace

void
write_hook(const hsa::Queue& queue,
           const hsa::rocprofiler_packet& /*kernel_packet*/,
           rocprofiler_kernel_id_t   kernel_id,
           rocprofiler_dispatch_id_t dispatch_id,
           rocprofiler_user_data_t*  user_data,
           const hsa::queue_info_session_t::external_corr_id_map_t& /*ext_corr_ids*/,
           const context::correlation_id* correlation_id,
           hsa::inst_pkt_t&               inst_pkt,
           bool&                          is_serialized)
{
    const auto agent_id = CHECK_NOTNULL(queue.get_agent().get_rocp_agent())->id;

    auto active = context::get_active_contexts(thread_trace_contexts_filter());
    for(auto* ctx : active)
    {
        auto& tracer = *ctx->dispatch_thread_trace;
        if(!tracer.collects_on(agent_id)) continue;

        auto [packet, bSerial] =
            tracer.pre_kernel_call(queue, kernel_id, dispatch_id, user_data, correlation_id);
        if(packet)
            inst_pkt.emplace_back(std::move(packet), hsa::queue_hooks::THREAD_TRACE_CLIENT_ID);
        is_serialized |= bSerial;
    }
}

void
signal_completion_hook(const hsa::Queue& /*queue*/,
                       const hsa::rocprofiler_packet& /*kernel_packet*/,
                       std::shared_ptr<hsa::queue_info_session_t>& session,
                       hsa::packet_data_t&                         packet_data,
                       hsa::inst_pkt_t&                            inst_pkt,
                       kernel_dispatch::profiling_time /*dispatch_time*/)
{
    // Completion routing follows packet provenance rather than current activeness so work
    // submitted before stop_context can still retire after the context leaves the active set.
    auto contexts = context::get_registered_contexts(thread_trace_contexts_filter());
    for(auto* ctx : contexts)
    {
        auto& tracer = *ctx->dispatch_thread_trace;
        tracer.post_kernel_call(inst_pkt, *session, packet_data);
    }
}

bool
is_any_active()
{
    return !context::get_active_contexts(thread_trace_contexts_filter()).empty();
}
}  // namespace thread_trace
}  // namespace rocprofiler
