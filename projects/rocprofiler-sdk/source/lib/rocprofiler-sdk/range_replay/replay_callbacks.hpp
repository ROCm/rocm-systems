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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

// Delivery of the RANGE_REPLAY callback-tracing operations (CONFIG, PASS, CLOSE) and the pass plan
// derived from the tool's CONFIG callback. Mirrors kernel_replay/replay_callbacks.hpp; the
// differences follow from a range spanning application code:
//  - CONFIG's ENTER and EXIT phases both run inside rocprofiler_range_replay_begin (a synchronous
//    configuration exchange), rather than bracketing the replay loop.
//  - CLOSE reports the outcome, because a range can be declined long after it was configured.

#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstdint>

namespace rocprofiler
{
namespace range_replay
{
using range_replay_data_t = rocprofiler_callback_tracing_range_replay_data_t;

// What the tool asked for at CONFIG, plus the contexts to deliver this range's later callbacks to.
struct range_plan_t
{
    tracing::callback_context_data_vec_t   config_contexts          = {};
    tracing::external_correlation_id_map_t external_correlation_ids = {};
    range_replay_data_t                    config_data              = {};

    // CLOSE's contexts, resolved at CONFIG rather than at close. A range is admitted while its
    // context is active, and CLOSE is that range's terminal notification, so it must be delivered
    // even if the tool stopped the context while the range was open -- the same rule that lets an
    // in-flight dispatch complete after its context stops. Re-resolving at close would silently
    // drop it.
    tracing::callback_context_data_vec_t   close_contexts        = {};
    tracing::external_correlation_id_map_t close_correlation_ids = {};

    // Sequence-wide user_data, captured from the first CONFIG context so every PASS and the CLOSE
    // callback for this range see what the tool wrote at CONFIG.
    rocprofiler_user_data_t user_data = {.value = 0};

    uint64_t (*pass_count_cb)(uint64_t, rocprofiler_user_data_t)                     = nullptr;
    int (*replay_continue_cb)(uint64_t, uint64_t, uint64_t, rocprofiler_user_data_t) = nullptr;

    uint64_t total_passes = 0;
    bool     indefinite   = false;
    // True when the tool asked for more than the application's own execution. A range with
    // replay_requested == false is still recorded-and-declined so the tool gets a CLOSE status.
    bool replay_requested = false;
};

// Per-pass callback state, populated during PASS PHASE_ENTER and reused for PASS PHASE_EXIT so the
// exit record carries the thread id, correlation ids, and tool-written user_data captured at enter.
struct pass_context_state_t
{
    tracing::callback_context_data_vec_t   contexts                 = {};
    tracing::external_correlation_id_map_t external_correlation_ids = {};
};

// Set once when a tool configures a RANGE_REPLAY callback-tracing service. A cheap process-global
// gate so the dispatch path does not walk the active-context list when range replay is never used.
void
set_range_replay_service_configured(bool enabled);

bool
has_active_range_replay_contexts();

// True if any registered context already configured a RANGE_REPLAY service. Checked at
// configuration time to reject a second subscriber: a range runs a single plan, so exactly one
// context may own range replay process-wide.
bool
has_registered_range_replay_context();

// CONFIG: ask the tool how many passes this range should run. Both callback phases run here.
range_plan_t
execute_config_callback(uint64_t                range_id,
                        rocprofiler_thread_id_t thr_id,
                        uint64_t                internal_corr_id,
                        uint64_t                ancestor_corr_id);

void
execute_pass_phase_enter(const range_plan_t&     plan,
                         uint64_t                current_pass,
                         rocprofiler_agent_id_t  agent_id,
                         uint64_t                dispatch_count,
                         rocprofiler_thread_id_t thr_id,
                         uint64_t                internal_corr_id,
                         uint64_t                ancestor_corr_id,
                         pass_context_state_t&   out_pass_state);

void
execute_pass_phase_exit(const range_plan_t&    plan,
                        uint64_t               current_pass,
                        rocprofiler_agent_id_t agent_id,
                        uint64_t               dispatch_count,
                        pass_context_state_t&  pass_state);

// CLOSE: report the outcome of the range. Delivered exactly once per opened range, including for
// ranges that were declined before any pass ran.
void
execute_close_callback(const range_plan_t&               plan,
                       rocprofiler_range_replay_status_t status,
                       rocprofiler_agent_id_t            agent_id,
                       uint64_t                          dispatch_count,
                       uint64_t                          divergence_count,
                       rocprofiler_thread_id_t           thr_id,
                       uint64_t                          internal_corr_id,
                       uint64_t                          ancestor_corr_id);

// Whether another pass should run after `current_pass`.
bool
should_continue_replay(const range_plan_t& plan, uint64_t current_pass, bool is_final_pass);
}  // namespace range_replay
}  // namespace rocprofiler
