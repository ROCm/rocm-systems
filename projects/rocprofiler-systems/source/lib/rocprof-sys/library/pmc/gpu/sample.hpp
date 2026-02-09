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
#include "library/pmc/gpu/types.hpp"

#include <cstdint>

namespace rocprofsys
{
namespace pmc
{
namespace gpu
{

/**
 * @brief GPU PMC sample type.
 *
 * This struct represents a sample of GPU performance metrics collected by the PMC.
 */
struct sample : trace_cache::cacheable_t
{
    static constexpr trace_cache::type_identifier_t type_identifier{
        trace_cache::type_identifier_t::gpu_pmc_sample
    };

    sample() = default;
    sample(enabled_metrics _settings, uint32_t _device_id, size_t _timestamp,
           metrics _metric_values)
    : enabled_metric(_settings)
    , device_id(_device_id)
    , timestamp(_timestamp)
    , metric_values(_metric_values)
    {}

    enabled_metrics enabled_metric;
    size_t          device_id;
    uint64_t        timestamp;
    metrics         metric_values;
};

}  // namespace gpu
}  // namespace pmc

namespace trace_cache
{

template <>
inline void
serialize(uint8_t* buffer, const pmc::gpu::sample& item)
{
    utility::store_value(
        buffer, static_cast<uint32_t>(item.enabled_metric.value), item.device_id,
        item.timestamp, item.metric_values.average_socket_power,
        item.metric_values.current_socket_power, item.metric_values.memory_usage,
        item.metric_values.hotspot_temperature, item.metric_values.edge_temperature,
        item.metric_values.gfx_activity, item.metric_values.umc_activity,
        item.metric_values.mm_activity, item.metric_values.xcp_stats,
        item.metric_values.xgmi.link.width, item.metric_values.xgmi.link.speed,
        item.metric_values.xgmi.data_acc.read, item.metric_values.xgmi.data_acc.write,
        item.metric_values.pcie.link.width, item.metric_values.pcie.link.speed,
        item.metric_values.pcie.bandwidth.acc, item.metric_values.pcie.bandwidth.inst);
}

template <>
inline pmc::gpu::sample
deserialize(uint8_t*& buffer)
{
    pmc::gpu::sample item;
    utility::parse_value(
        buffer, item.enabled_metric.value, item.device_id, item.timestamp,
        item.metric_values.average_socket_power, item.metric_values.current_socket_power,
        item.metric_values.memory_usage, item.metric_values.hotspot_temperature,
        item.metric_values.edge_temperature, item.metric_values.gfx_activity,
        item.metric_values.umc_activity, item.metric_values.mm_activity,
        item.metric_values.xcp_stats, item.metric_values.xgmi.link.width,
        item.metric_values.xgmi.link.speed, item.metric_values.xgmi.data_acc.read,
        item.metric_values.xgmi.data_acc.write, item.metric_values.pcie.link.width,
        item.metric_values.pcie.link.speed, item.metric_values.pcie.bandwidth.acc,
        item.metric_values.pcie.bandwidth.inst);
    return item;
}

template <>
inline size_t
get_size(const pmc::gpu::sample& item)
{
    return utility::get_size(
        item.enabled_metric.value, item.device_id, item.timestamp,
        item.metric_values.average_socket_power, item.metric_values.current_socket_power,
        item.metric_values.memory_usage, item.metric_values.hotspot_temperature,
        item.metric_values.edge_temperature, item.metric_values.gfx_activity,
        item.metric_values.umc_activity, item.metric_values.mm_activity,
        item.metric_values.xcp_stats, item.metric_values.xgmi.link.width,
        item.metric_values.xgmi.link.speed, item.metric_values.xgmi.data_acc.read,
        item.metric_values.xgmi.data_acc.write, item.metric_values.pcie.link.width,
        item.metric_values.pcie.link.speed, item.metric_values.pcie.bandwidth.acc,
        item.metric_values.pcie.bandwidth.inst);
}

/// @brief GPU PMC sample type - legacy alias for backward compatibility
using gpu_pmc_sample = pmc::gpu::sample;

}  // namespace trace_cache
}  // namespace rocprofsys
