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

#include <atomic>
#include <fmt/format.h>
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
    if(packet_data.kfd_correlation_key_valid)
    {
        auto corr_key = kfd::correlation_key{packet_data.kfd_doorbell_off,
                                             packet_data.kfd_dispatch_idx_low32,
                                             packet_data.kfd_generation};

        // Take the firmware result for this dispatch. A miss (no record yet, or
        // never deposited) falls through to the unconditional HSA path below. The
        // reader thread's evict_stale reclaims any result never taken here (e.g.
        // dispatch completed via HSA fallback before its record arrived), so the
        // results map stays bounded without an erase on this path.
        auto kfd_result = kfd::results_map().take(corr_key);

        {
            static std::atomic<uint64_t> s_hit{0}, s_miss{0};
            (kfd_result ? s_hit : s_miss).fetch_add(1, std::memory_order_relaxed);
            uint64_t h = s_hit.load(), m = s_miss.load();
            if((h + m) % 256 == 0)
                ROCP_WARNING << fmt::format("DBGPROV hit={} miss={} rate={:.1f}%", h, m,
                                            100.0 * static_cast<double>(h) /
                                                static_cast<double>(h + m));
        }

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
            // dispatch would land outside these bounds; rather than let
            // adjust_profiling_time() clamp it (or FATAL under CI-strict) into a
            // plausible-but-wrong value, we reject and fall back to HSA. This checks
            // correlation, not the conversion (which is HSA's own).
            const uint64_t _now      = common::timestamp_ns();
            const bool     _kfd_sane = _s1 == HSA_STATUS_SUCCESS && _s2 == HSA_STATUS_SUCCESS &&
                                   kfd_start_sys < kfd_end_sys &&
                                   kfd_start_sys >= session.enqueue_ts && kfd_end_sys <= _now;

            if(_kfd_sane)
            {
                // Emit firmware timestamps. The firmware interval (end-start) is
                // deliberately kept: it is measured at the true HW dispatch
                // boundaries and is tighter/earlier than HSA's signal-machinery
                // timing, so we do NOT force it to match the HSA duration.
                auto kfd_time =
                    tracing::profiling_time{HSA_STATUS_SUCCESS, kfd_start_sys, kfd_end_sys};
                return tracing::adjust_profiling_time(
                    "dispatch",
                    "kfd_dispatch_log",
                    kfd_time,
                    tracing::profiling_time{HSA_STATUS_SUCCESS, session.enqueue_ts, _now});
            }
            // convert failed or record failed the sanity guard: fall through to HSA.
        }
    }

    // --- HSA fallback (unchanged; unconditional) ---
    auto _signal  = packet_data.kernel_packet.kernel_dispatch.completion_signal;
    auto _kern_id = callback_record.dispatch_info.kernel_id;

    return (_hsa_agent) ? get_dispatch_time(*_hsa_agent, _signal, _kern_id, session.enqueue_ts)
                        : profiling_time{.status = HSA_STATUS_ERROR_INVALID_AGENT};
}

void
dispatch_complete(queue_info_session_t& session,
                  packet_data_t&        packet_data,
                  profiling_time        dispatch_time)
{
    using kernel_dispatch_record_t = rocprofiler_buffer_tracing_kernel_dispatch_record_t;

    // get the contexts that were active when the signal was created
    auto& tracing_data_v = packet_data.tracing_data;
    if(tracing_data_v.callback_contexts.empty() && tracing_data_v.buffered_contexts.empty()) return;

    // we need to decrement this reference count at the end of the functions
    auto* _corr_id = session.correlation_id;

    // only do the following work if there are contexts that require this info
    auto&       callback_record   = packet_data.callback_record;
    const auto& _extern_corr_ids  = packet_data.tracing_data.external_correlation_ids;
    auto        _tid              = session.tid;
    auto        _internal_corr_id = (_corr_id) ? _corr_id->internal : 0;
    auto        _ancestor_corr_id = (_corr_id) ? _corr_id->ancestor : 0;

    if(dispatch_time.status == HSA_STATUS_SUCCESS)
    {
        callback_record.start_timestamp = dispatch_time.start;
        callback_record.end_timestamp   = dispatch_time.end;

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
}  // namespace kernel_dispatch
}  // namespace rocprofiler
