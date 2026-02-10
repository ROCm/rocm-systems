// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/cpu/types.hpp"

#include <cstdint>
#include <vector>

namespace rocprofsys
{
namespace pmc
{
namespace cpu
{

/**
 * @brief CPU PMC sample type.
 *
 * This struct represents a sample of CPU performance metrics collected by the PMC.
 */
struct sample : trace_cache::cacheable_t
{
    static constexpr trace_cache::type_identifier_t type_identifier{
        trace_cache::type_identifier_t::cpu_pmc_sample
    };

    sample() = default;
    sample(enabled_metrics _settings, uint64_t _timestamp, process_metrics _process_data,
           std::vector<uint8_t> _freqs, std::vector<uint8_t> _loads)
    : enabled_metric(_settings)
    , timestamp(_timestamp)
    , process_data(_process_data)
    , freqs(std::move(_freqs))
    , loads(std::move(_loads))
    {}

    enabled_metrics      enabled_metric;
    uint64_t             timestamp;
    process_metrics      process_data;
    std::vector<uint8_t> freqs;  // serialized cpu_id+freq pairs
    std::vector<uint8_t> loads;  // serialized cpu_id+load pairs
};

}  // namespace cpu
}  // namespace pmc

namespace trace_cache
{

template <>
inline void
serialize(uint8_t* buffer, const pmc::cpu::sample& item)
{
    utility::store_value(buffer, static_cast<uint32_t>(item.enabled_metric.value),
                         item.timestamp, item.process_data.page_rss,
                         item.process_data.virt_mem, item.process_data.peak_rss,
                         item.process_data.context_switches,
                         item.process_data.page_faults,
                         item.process_data.user_mode_time,
                         item.process_data.kernel_mode_time, item.freqs, item.loads);
}

template <>
inline pmc::cpu::sample
deserialize(uint8_t*& buffer)
{
    pmc::cpu::sample item;
    utility::parse_value(buffer, item.enabled_metric.value, item.timestamp,
                         item.process_data.page_rss, item.process_data.virt_mem,
                         item.process_data.peak_rss,
                         item.process_data.context_switches,
                         item.process_data.page_faults,
                         item.process_data.user_mode_time,
                         item.process_data.kernel_mode_time, item.freqs, item.loads);
    return item;
}

template <>
inline size_t
get_size(const pmc::cpu::sample& item)
{
    return utility::get_size(static_cast<uint32_t>(item.enabled_metric.value),
                             item.timestamp, item.process_data.page_rss,
                             item.process_data.virt_mem, item.process_data.peak_rss,
                             item.process_data.context_switches,
                             item.process_data.page_faults,
                             item.process_data.user_mode_time,
                             item.process_data.kernel_mode_time, item.freqs, item.loads);
}

/// @brief CPU PMC sample type - legacy alias for backward compatibility
using cpu_pmc_sample = pmc::cpu::sample;

}  // namespace trace_cache
}  // namespace rocprofsys
