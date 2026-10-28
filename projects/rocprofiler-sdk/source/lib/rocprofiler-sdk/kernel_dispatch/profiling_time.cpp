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

// Both raw timestamps in one streaming load. The host maps device memory write-combined, where a
// plain load is a serialized ~1 us PCIe round trip but streaming loads overlap; on other memory
// types a streaming load is an ordinary load. Weakly ordered: see device_signal_read_fence.
#if defined(__x86_64__)
__attribute__((target("sse4.1"))) void
load_raw_timestamps(hsa_signal_t _signal, uint64_t& _start, uint64_t& _end)
{
    static_assert(offsetof(amd_signal_t, end_ts) == offsetof(amd_signal_t, start_ts) + 8);
    static_assert(offsetof(amd_signal_t, start_ts) % 16 == 0);

    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto*      _ts = &reinterpret_cast<amd_signal_t*>(_signal.handle)->start_ts;
    const auto _v  = _mm_stream_load_si128(reinterpret_cast<__m128i*>(_ts));
    _start         = static_cast<uint64_t>(_mm_cvtsi128_si64(_v));
    _end           = static_cast<uint64_t>(_mm_extract_epi64(_v, 1));
}
#else
void
load_raw_timestamps(hsa_signal_t _signal, uint64_t& _start, uint64_t& _end)
{
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    const auto* _amd_signal = reinterpret_cast<const amd_signal_t*>(_signal.handle);
    _start                  = _amd_signal->start_ts;
    _end                    = _amd_signal->end_ts;
}
#endif
}  // namespace

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
    hsa_agent_t             _hsa_agent,
    hsa_signal_t            _signal,
    rocprofiler_kernel_id_t _kernel_id,
    std::optional<uint64_t> _baseline)  // NOLINT(performance-unnecessary-value-param)
{
    auto ts         = common::timestamp_ns();
    auto start_tick = uint64_t{0};
    auto end_tick   = uint64_t{0};
    load_raw_timestamps(_signal, start_tick, end_tick);

    auto dispatch_time = hsa_amd_profiling_dispatch_time_t{};
    auto status        = HSA_STATUS_SUCCESS;
    // zero ticks are what hsa_amd_profiling_get_dispatch_time reports as 0..0
    if(start_tick != 0 && end_tick != 0)
    {
        const auto* _ext = hsa::get_amd_ext_table();
        // end first, as ROCr does, so clock-read jitter does not land in the duration
        status = _ext->hsa_amd_profiling_convert_tick_to_system_domain_fn(
            _hsa_agent, end_tick, &dispatch_time.end);
        if(status == HSA_STATUS_SUCCESS)
            status = _ext->hsa_amd_profiling_convert_tick_to_system_domain_fn(
                _hsa_agent, start_tick, &dispatch_time.start);
    }

    return finish_dispatch_time("hsa_amd_profiling_convert_tick_to_system_domain",
                                status,
                                dispatch_time,
                                ts,
                                _hsa_agent,
                                _kernel_id,
                                _baseline);
}
}  // namespace kernel_dispatch
}  // namespace rocprofiler
