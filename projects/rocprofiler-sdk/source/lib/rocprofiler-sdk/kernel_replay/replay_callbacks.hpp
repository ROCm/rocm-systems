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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"

#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstdint>

namespace rocprofiler
{
namespace hsa
{
class Queue;
}
namespace kernel_replay
{
struct replay_plan_t
{
    bool                    replay_requested = false;
    uint64_t                total_passes     = 1;
    bool                    indefinite       = false;
    rocprofiler_user_data_t user_data        = {.value = 0};

    rocprofiler_callback_tracing_kernel_replay_data_t config_data = {};
    tracing::callback_context_data_vec_t              config_contexts{};
    tracing::external_correlation_id_map_t            external_correlation_ids{};

    uint64_t (*pass_count_cb)(rocprofiler_kernel_dispatch_info_t dispatch_info,
                              rocprofiler_user_data_t            user_data) = nullptr;
    int (*replay_continue_cb)(rocprofiler_kernel_dispatch_info_t dispatch_info,
                              uint64_t                           current_pass,
                              uint64_t                           total_passes,
                              rocprofiler_user_data_t            user_data) = nullptr;
};

// Per-pass callback context state. Populated during PASS PHASE_ENTER and reused for PASS
// PHASE_EXIT so the exit record carries the same thread id, correlation ids, operation, and
// tool-written user_data captured at enter (the exit callback derives these from the record
// stored during enter).
struct pass_context_state_t
{
    tracing::callback_context_data_vec_t   contexts{};
    tracing::external_correlation_id_map_t external_correlation_ids{};
};

// Set once when a tool configures a KERNEL_REPLAY callback-tracing service. Acts as a cheap
// process-global fast-path gate so has_active_replay_contexts() (and thus WriteInterceptor) does
// not walk the active-context list on every dispatch when replay is never used.
void
set_replay_service_configured(bool enabled);

bool
has_active_replay_contexts();

rocprofiler_kernel_dispatch_info_t
make_dispatch_info(const hsa::Queue& queue, const hsa::rocprofiler_packet& pkt);

replay_plan_t
execute_config_phase_enter(const hsa::Queue&              queue,
                           const hsa::rocprofiler_packet& pkt,
                           rocprofiler_thread_id_t        thr_id,
                           uint64_t                       internal_corr_id,
                           uint64_t                       ancestor_corr_id);

void
execute_config_phase_exit(const replay_plan_t&    plan,
                          rocprofiler_thread_id_t thr_id,
                          uint64_t                internal_corr_id,
                          uint64_t                ancestor_corr_id);

void
execute_pass_phase_enter(const replay_plan_t&    plan,
                         uint64_t                current_pass,
                         rocprofiler_thread_id_t thr_id,
                         uint64_t                internal_corr_id,
                         uint64_t                ancestor_corr_id,
                         pass_context_state_t&   out_pass_state);

void
execute_pass_phase_exit(const replay_plan_t&  plan,
                        uint64_t              current_pass,
                        pass_context_state_t& pass_state);

bool
should_continue_replay(const replay_plan_t& plan, uint64_t current_pass, bool is_final_pass);
}  // namespace kernel_replay
}  // namespace rocprofiler
