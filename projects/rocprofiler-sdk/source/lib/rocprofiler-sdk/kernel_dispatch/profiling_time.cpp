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

#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/tracing/profiling_time.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>

#if defined(__x86_64__)
#    include <immintrin.h>
#endif

#include <atomic>
#include <cstddef>
#include <string_view>

namespace rocprofiler
{
namespace kernel_dispatch
{
namespace
{
profiling_time
finish_dispatch_time(std::string_view                  _responsible,
                     hsa_status_t                      _status,
                     hsa_amd_profiling_dispatch_time_t _dispatch_time,
                     uint64_t                          _ts,
                     hsa_agent_t                       _hsa_agent,
                     rocprofiler_kernel_id_t           _kernel_id,
                     std::optional<uint64_t>           _baseline)
{
    auto _profile_time = tracing::profiling_time{_status, _dispatch_time.start, _dispatch_time.end};

    if(_profile_time.status == HSA_STATUS_SUCCESS)
    {
        _profile_time = tracing::adjust_profiling_time(
            "dispatch",
            _responsible,
            _profile_time,
            tracing::profiling_time{
                HSA_STATUS_SUCCESS, _baseline.value_or(_dispatch_time.start), _ts});
    }
    else
    {
        ROCP_CI_LOG(ERROR) << fmt::format(
            "{} for kernel id={} on agent-{} returned status={} :: {}",
            _responsible,
            _kernel_id,
            CHECK_NOTNULL(agent::get_rocprofiler_agent(_hsa_agent))->node_id,
            static_cast<int>(_status),
            hsa::get_hsa_status_string(_status));
    }

    return _profile_time;
}

static_assert(offsetof(amd_signal_t, value) == 8);
static_assert(offsetof(amd_signal_t, start_ts) == 32);
static_assert(offsetof(amd_signal_t, end_ts) == 40);
static_assert(alignof(amd_signal_t) == 64);

// The value and both raw timestamps, with two streaming loads of the one cache line holding them.
// On a write-combined mapping (how the host maps large-BAR device memory) a plain load is a
// serialized ~1 us PCIe round trip, but streaming loads of a line are served by one fill; on other
// memory types a streaming load is an ordinary load. Weakly ordered: see device_signal_read_fence.
#if defined(__x86_64__)
__attribute__((target("sse4.1"))) device_signal_snapshot
load_device_signal(hsa_signal_t _signal)
{
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto*      _line = reinterpret_cast<__m128i*>(_signal.handle);
    const auto _head = _mm_stream_load_si128(_line);
    const auto _ts   = _mm_stream_load_si128(_line + 2);
    return device_signal_snapshot{static_cast<hsa_signal_value_t>(_mm_extract_epi64(_head, 1)),
                                  static_cast<uint64_t>(_mm_cvtsi128_si64(_ts)),
                                  static_cast<uint64_t>(_mm_extract_epi64(_ts, 1))};
}
#else
device_signal_snapshot
load_device_signal(hsa_signal_t _signal)
{
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    const volatile auto* _amd_signal =
        reinterpret_cast<const volatile amd_signal_t*>(_signal.handle);
    return device_signal_snapshot{_amd_signal->value, _amd_signal->start_ts, _amd_signal->end_ts};
}
#endif

profiling_time
convert_device_signal_time(hsa_agent_t                        _hsa_agent,
                           const device_signal_snapshot&      _snapshot,
                           hsa_amd_profiling_dispatch_time_t& _dispatch_time)
{
    _dispatch_time = hsa_amd_profiling_dispatch_time_t{};
    // zero ticks are what hsa_amd_profiling_get_dispatch_time reports as 0..0
    if(_snapshot.start_tick == 0 || _snapshot.end_tick == 0)
        return profiling_time{HSA_STATUS_SUCCESS, 0, 0};

    const auto* _ext = hsa::get_amd_ext_table();
    if(!_ext || !_ext->hsa_amd_profiling_convert_tick_to_system_domain_fn)
        return profiling_time{HSA_STATUS_ERROR_NOT_INITIALIZED, 0, 0};

    // end first, as ROCr's GpuAgent::TranslateTime does, so clock-read jitter does not land in the
    // duration
    auto status = _ext->hsa_amd_profiling_convert_tick_to_system_domain_fn(
        _hsa_agent, _snapshot.end_tick, &_dispatch_time.end);
    if(status == HSA_STATUS_SUCCESS)
        status = _ext->hsa_amd_profiling_convert_tick_to_system_domain_fn(
            _hsa_agent, _snapshot.start_tick, &_dispatch_time.start);
    return profiling_time{status, _dispatch_time.start, _dispatch_time.end};
}

// Whether raw reads agree with hsa_amd_profiling_get_dispatch_time. Checked on the first
// completed device signal, so a change in ROCr's signal layout or time translation turns the raw
// path off rather than producing wrong timestamps.
enum class raw_read_state : int
{
    unchecked = 0,
    agrees,
    disagrees
};

std::atomic<raw_read_state>&
get_raw_read_state()
{
    static auto _v = std::atomic<raw_read_state>{raw_read_state::unchecked};
    return _v;
}

// ROCr re-syncs its clocks while translating, so translations of one tick can differ slightly
constexpr uint64_t raw_read_tolerance_ns = 1000;

bool
within_tolerance(uint64_t _lhs, uint64_t _rhs)
{
    return ((_lhs > _rhs) ? _lhs - _rhs : _rhs - _lhs) <= raw_read_tolerance_ns;
}
}  // namespace

device_signal_snapshot
read_device_signal(hsa_signal_t signal)
{
    return load_device_signal(signal);
}

void
device_signal_read_fence()
{
#if defined(__x86_64__)
    _mm_mfence();
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

profiling_time
get_dispatch_time(hsa_agent_t             _hsa_agent,
                  hsa_signal_t            _signal,
                  rocprofiler_kernel_id_t _kernel_id,
                  std::optional<uint64_t> _baseline)  // NOLINT(performance-unnecessary-value-param)
{
    auto ts                   = common::timestamp_ns();
    auto dispatch_time        = hsa_amd_profiling_dispatch_time_t{};
    auto dispatch_time_status = hsa::get_amd_ext_table()->hsa_amd_profiling_get_dispatch_time_fn(
        _hsa_agent, _signal, &dispatch_time);

    return finish_dispatch_time("hsa_amd_profiling_get_dispatch_time",
                                dispatch_time_status,
                                dispatch_time,
                                ts,
                                _hsa_agent,
                                _kernel_id,
                                _baseline);
}

profiling_time
get_device_signal_dispatch_time(
    hsa_agent_t                   _hsa_agent,
    hsa_signal_t                  _signal,
    const device_signal_snapshot& _snapshot,
    rocprofiler_kernel_id_t       _kernel_id,
    std::optional<uint64_t>       _baseline)  // NOLINT(performance-unnecessary-value-param)
{
    auto& _state = get_raw_read_state();
    if(_state.load(std::memory_order_relaxed) == raw_read_state::disagrees)
        return get_dispatch_time(_hsa_agent, _signal, _kernel_id, _baseline);

    auto ts            = common::timestamp_ns();
    auto dispatch_time = hsa_amd_profiling_dispatch_time_t{};
    auto converted     = convert_device_signal_time(_hsa_agent, _snapshot, dispatch_time);

    if(_state.load(std::memory_order_relaxed) == raw_read_state::unchecked &&
       converted.status == HSA_STATUS_SUCCESS && dispatch_time.start != 0)
    {
        auto reference = hsa_amd_profiling_dispatch_time_t{};
        auto status    = hsa::get_amd_ext_table()->hsa_amd_profiling_get_dispatch_time_fn(
            _hsa_agent, _signal, &reference);
        const bool agrees = status == HSA_STATUS_SUCCESS &&
                            within_tolerance(dispatch_time.start, reference.start) &&
                            within_tolerance(dispatch_time.end, reference.end);
        _state.store(agrees ? raw_read_state::agrees : raw_read_state::disagrees,
                     std::memory_order_relaxed);
        if(!agrees)
        {
            ROCP_WARNING << fmt::format(
                "device signal timestamps read directly ({}..{}) disagree with "
                "hsa_amd_profiling_get_dispatch_time ({}..{}); using the latter from now on",
                dispatch_time.start,
                dispatch_time.end,
                reference.start,
                reference.end);
            return get_dispatch_time(_hsa_agent, _signal, _kernel_id, _baseline);
        }
    }

    return finish_dispatch_time("hsa_amd_profiling_convert_tick_to_system_domain",
                                converted.status,
                                dispatch_time,
                                ts,
                                _hsa_agent,
                                _kernel_id,
                                _baseline);
}
}  // namespace kernel_dispatch
}  // namespace rocprofiler
