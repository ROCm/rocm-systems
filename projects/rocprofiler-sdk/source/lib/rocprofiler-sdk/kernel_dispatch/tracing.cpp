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

#include "lib/rocprofiler-sdk/kernel_dispatch/tracing.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_correlation.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>

#include <string_view>

namespace rocprofiler
{
namespace kernel_dispatch
{
profiling_time
get_dispatch_time(const queue_info_session_t& session, packet_data_t& packet_data)
{
    const auto& callback_record = packet_data.callback_record;
    const auto* _rocp_agent     = agent::get_agent(callback_record.dispatch_info.agent_id);
    auto        _hsa_agent      = agent::get_hsa_agent(_rocp_agent);

    // --- KFD dispatch-log path (preferred timestamp source) ---
    // Only attempted when the correlation key was captured at enqueue. Firmware
    // records carry raw GPU-clock ticks; hsa_amd_profiling_convert_tick_to_system_
    // domain() rebases them onto the same CLOCK_BOOTTIME domain HSA uses (it handles
    // both the GPU<->system rate ratio and epoch, via the runtime's own clock-sync
    // machinery). If anything is missing (no record, no agent, convert error) we
    // fall through to the unconditional HSA path below.
    //
    // PHASE 1 = option (b): kfd_selection_enabled() is false, so this block is
    // skipped entirely and every eligible dispatch deterministically reports HSA
    // timestamps -- the per-dispatch "KFD if the record happened to arrive in time,
    // HSA otherwise" race is gone. The fallback is counted either way so the
    // substitution is visible. Phase 2 flips the gate.
    if(packet_data.kfd_correlation_key_valid && kfd::kfd_selection_enabled())
    {
        auto corr_key = kfd::correlation_key{packet_data.kfd_doorbell_off,
                                             packet_data.kfd_dispatch_idx_low32,
                                             packet_data.kfd_generation,
                                             packet_data.kfd_gpu_id};

        // Rendezvous rather than a one-shot take(): wait for the reader to deposit
        // this dispatch's record instead of silently using HSA because the record
        // was merely late. The wait ends at the batch's single absolute deadline,
        // when the reader is declared dead, or on deposit. No lock is held across
        // it. The reader thread's evict_stale reclaims any result never taken here,
        // so the results map stays bounded without an erase on this path.
        auto kfd_result = kfd::results_map().wait_take(corr_key, session.kfd_deadline_ns);

        // Converting firmware ticks needs the agent; without it we cannot emit KFD
        // timestamps, so fall through to the HSA path (which also handles null agent).
        if(kfd_result && _hsa_agent)
        {
            const auto* _ext          = hsa::get_amd_ext_table();
            uint64_t    kfd_start_sys = 0;
            uint64_t    kfd_end_sys   = 0;
            auto        _s1           = _ext->hsa_amd_profiling_convert_tick_to_system_domain_fn(
                *_hsa_agent, kfd_result->start_gpu_ticks, &kfd_start_sys);
            auto _s2 = _ext->hsa_amd_profiling_convert_tick_to_system_domain_fn(
                *_hsa_agent, kfd_result->end_gpu_ticks, &kfd_end_sys);

            // Sanity guard against a mis-correlated / stale firmware record: the
            // converted times must form a positive interval that fits inside this
            // dispatch's own CPU window [enqueue, now]. A record pulled for the wrong
            // dispatch would land outside these bounds, so reject it and fall back to
            // HSA. This checks correlation, not the conversion (which is HSA's own).
            const uint64_t _now      = common::timestamp_ns();
            const bool     _kfd_sane = _s1 == HSA_STATUS_SUCCESS && _s2 == HSA_STATUS_SUCCESS &&
                                   kfd::kfd_time_is_sane(
                                       kfd_start_sys, kfd_end_sys, session.enqueue_ts, _now);

            // Emit the bounds-checked firmware timestamps as-is: they are already
            // system-domain, and adjust_profiling_time() would re-scale them under
            // ROCPROFILER_CI_FREQ_SCALE_TIMESTAMPS and widen the tighter HW interval.
            if(_kfd_sane) return profiling_time{HSA_STATUS_SUCCESS, kfd_start_sys, kfd_end_sys};
            // convert failed or record failed the sanity guard: fall through to HSA.
        }
    }

    // This dispatch was eligible for a firmware timestamp but reports an HSA one.
    if(packet_data.kfd_correlation_key_valid) kfd::results_map().note_hsa_fallback();

    // --- HSA fallback (unchanged; unconditional) ---
    auto _signal  = packet_data.kernel_packet.kernel_dispatch.completion_signal;
    auto _kern_id = callback_record.dispatch_info.kernel_id;

    return (_hsa_agent) ? get_dispatch_time(*_hsa_agent, _signal, _kern_id, session.enqueue_ts)
                        : profiling_time{.status = HSA_STATUS_ERROR_INVALID_AGENT};
}

void
emit_kernel_dispatch_record(tracing::tracing_data&                               tracing_data_v,
                            rocprofiler_callback_tracing_kernel_dispatch_data_t& callback_record,
                            context::correlation_id*                             _corr_id,
                            rocprofiler_thread_id_t                              _tid,
                            uint64_t                                             start_timestamp,
                            uint64_t                                             end_timestamp)
{
    using kernel_dispatch_record_t = rocprofiler_buffer_tracing_kernel_dispatch_record_t;

    // get the contexts that were active when the dispatch was traced
    if(tracing_data_v.callback_contexts.empty() && tracing_data_v.buffered_contexts.empty()) return;

    const auto& _extern_corr_ids  = tracing_data_v.external_correlation_ids;
    auto        _internal_corr_id = (_corr_id) ? _corr_id->internal : 0;
    auto        _ancestor_corr_id = (_corr_id) ? _corr_id->ancestor : 0;

    {
        callback_record.start_timestamp = start_timestamp;
        callback_record.end_timestamp   = end_timestamp;

        if(!tracing_data_v.callback_contexts.empty())
        {
            auto tracer_data = callback_record;
            tracing::execute_phase_none_callbacks(tracing_data_v.callback_contexts,
                                                  _tid,
                                                  _internal_corr_id,
                                                  _extern_corr_ids,
                                                  _ancestor_corr_id,
                                                  ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                                  ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                                                  tracer_data);
        }

        if(!tracing_data_v.buffered_contexts.empty())
        {
            auto record = kernel_dispatch_record_t{sizeof(kernel_dispatch_record_t),
                                                   ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                   ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                                                   rocprofiler_async_correlation_id_t{},
                                                   _tid,
                                                   callback_record.start_timestamp,
                                                   callback_record.end_timestamp,
                                                   callback_record.dispatch_info};

            tracing::execute_buffer_record_emplace(tracing_data_v.buffered_contexts,
                                                   _tid,
                                                   _internal_corr_id,
                                                   _extern_corr_ids,
                                                   _ancestor_corr_id,
                                                   ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                   ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                                                   record);
        }
    }
}

void
dispatch_complete(queue_info_session_t& session,
                  packet_data_t&        packet_data,
                  profiling_time        dispatch_time)
{
    if(dispatch_time.status != HSA_STATUS_SUCCESS) return;

    emit_kernel_dispatch_record(packet_data.tracing_data,
                                packet_data.callback_record,
                                session.correlation_id,
                                session.tid,
                                dispatch_time.start,
                                dispatch_time.end);
}
}  // namespace kernel_dispatch
}  // namespace rocprofiler
