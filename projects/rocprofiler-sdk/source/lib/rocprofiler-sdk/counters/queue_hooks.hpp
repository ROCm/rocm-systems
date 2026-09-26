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

#pragma once

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_info_session.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"

#include <memory>

namespace rocprofiler
{
namespace counters
{
// Explicit replacement for the dispatch-counter per-queue callback that used to
// be registered with the HSA queue controller. Iterates active
// dispatch_counter_collection contexts and calls each callback's queue_cb;
// appends produced packets to inst_pkt, OR-folding each callback's serialize flag
// into is_serialized.
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

// Explicit replacement for the dispatch-counter completion callback. Iterates
// registered dispatch_counter_collection contexts (not only active ones) and calls
// each callback's completed_cb; completed_cb self-filters via packet_return_map so
// in-flight dispatches still complete after stop_context.
void
kernel_dispatch_phase_exit_hook(const hsa::Queue&                           queue,
                                const hsa::rocprofiler_packet&              kernel_packet,
                                std::shared_ptr<hsa::queue_info_session_t>& session,
                                hsa::packet_data_t&                         packet,
                                hsa::inst_pkt_t&                            inst_pkt,
                                kernel_dispatch::profiling_time             dispatch_time);

// True if any context currently has dispatch counter collection active.
bool
is_any_active();
}  // namespace counters
}  // namespace rocprofiler
