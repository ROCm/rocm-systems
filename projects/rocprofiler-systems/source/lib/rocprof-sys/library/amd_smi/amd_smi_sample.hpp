#pragma once

#include "core/trace_cache/cache_type_traits.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/amd_smi/common.hpp"
#include <cstdint>

namespace rocprofsys
{

namespace trace_cache
{

struct amd_smi_sample
{
    static constexpr trace_cache::type_identifier_t type_identifier{
        trace_cache::type_identifier_t::amd_smi_sample
    };

    amd_smi_sample() = default;
    amd_smi_sample(amd_smi::enabled_metric _settings, uint32_t _device_id,
                   size_t _timestamp, amd_smi::smi_metrics _metrics)
    : enabled_metric(_settings)
    , device_id(_device_id)
    , timestamp(_timestamp)
    , metrics(_metrics)
    {}

    amd_smi::enabled_metric enabled_metric;
    size_t                  device_id;
    uint64_t                timestamp;
    amd_smi::smi_metrics    metrics;
};

template <>
inline void
serialize(uint8_t* buffer, const amd_smi_sample& item)
{
    trace_cache::utility::store_value(buffer, item.enabled_metric.value, item.device_id,
                                      item.timestamp, item.metrics);
}

template <>
inline amd_smi_sample
deserialize(uint8_t*& /*buffer*/)
{
    amd_smi_sample item;
    return item;
}

template <>
inline size_t
get_size(const amd_smi_sample& /*item*/)
{
    return 32;
}

}  // namespace trace_cache
}  // namespace rocprofsys
