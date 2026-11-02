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

#pragma once

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_info_session.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"

#include <memory>

namespace rocprofiler
{
namespace thread_trace
{
// Explicit replacement for the thread-trace per-queue callback that used to be
// registered with the HSA queue controller. Iterates active dispatch_thread_trace
// contexts and calls each tracer's pre_kernel_call; appends produced packets to
// inst_pkt, OR-folding each tracer's serialize flag into is_serialized.
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
    bool&                                                    is_serialized);

// Explicit replacement for the thread-trace completion callback. Iterates registered
// dispatch_thread_trace contexts so post_kernel_call can still drain packets that were
// injected before stop_context removed the tracer from the active set.
void
kernel_dispatch_phase_exit_hook(const hsa::Queue&                           queue,
                                const hsa::rocprofiler_packet&              kernel_packet,
                                std::shared_ptr<hsa::queue_info_session_t>& session,
                                hsa::packet_data_t&                         packet,
                                hsa::inst_pkt_t&                            inst_pkt,
                                kernel_dispatch::profiling_time             dispatch_time);

// True if any context currently has dispatch thread trace active.
bool
is_any_active();

// True if an active dispatch thread trace context is configured for `agent_id`. Callers on a
// per-queue path should prefer this over is_any_active(): a tracer is configured per agent, so
// one set up for a single GPU must not pull queues on the other GPUs off the write interceptor's
// fast path.
bool
is_active_on_agent(rocprofiler_agent_id_t agent_id);
}  // namespace thread_trace
}  // namespace rocprofiler
