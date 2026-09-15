// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/cache_type_traits.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/sample_type.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace rocprofsys::rocprofiler_sdk::spm
{
/**
 * @brief One SPM counter instance carried by a sample batch.
 *
 * `counter_instance_id` is the per-dimension-instance id used to resolve the
 * Perfetto track. `counter_id` is the base SDK counter id from
 * `rocprofiler_query_record_counter_id()`; the two are distinct id spaces and
 * neither can be derived from the other.
 */
struct counter_info
{
    // Currently unused by Perfetto export. Serialized so future RocPD SPM export can
    // map each instance back to the base SDK counter id after SDK shutdown.
    std::uint64_t counter_id          = 0;
    std::uint64_t counter_instance_id = 0;
};
static_assert(std::is_trivially_copyable<counter_info>::value,
              "spm::counter_info must remain a lightweight value type");

/// @brief One counter reading, indexed into `sample::counters`.
struct counter_value
{
    std::uint32_t counter_info_index = 0;
    double        value              = 0.0;
};
static_assert(std::is_trivially_copyable<counter_value>::value,
              "spm::counter_value must remain a lightweight value type");

/// @brief All counter readings sharing one hardware timestamp.
struct timestamp_sample
{
    std::uint64_t              timestamp = 0;
    std::vector<counter_value> values;
};

/**
 * @brief SPM sample type for trace cache serialization.
 *
 * One trace-cache record per non-empty SDK data callback batch. Each batch belongs
 * to one dispatch and may contain multiple counter instances and timestamps. The
 * model does not rely on SDK record ordering: counter metadata is interned once in
 * `counters`, and `timestamp_sample::values` index into it.
 */
struct sample : trace_cache::cacheable_t
{
    // Match the trace-cache serialization protocol used by every sample type.
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr trace_cache::type_identifier_t type_identifier{
        trace_cache::type_identifier_t::spm_sample
    };

    std::uint64_t agent_id_handle         = 0;
    std::uint64_t dispatch_id             = 0;
    std::uint64_t kernel_id               = 0;
    std::uint64_t queue_id_handle         = 0;
    std::uint64_t correlation_id_internal = 0;
    std::uint64_t correlation_id_ancestor = 0;
    // Reserved for HIP stream correlation; the producer writes zero until that
    // callback-to-stream mapping is available.
    std::uint64_t                 stream_handle = 0;
    bool                          data_loss     = false;
    std::vector<counter_info>     counters;
    std::vector<timestamp_sample> samples;
};

}  // namespace rocprofsys::rocprofiler_sdk::spm

namespace rocprofsys::trace_cache
{
/// @brief SPM sample type aliases used by the trace-cache storage and processors.
using spm_counter_info     = rocprofiler_sdk::spm::counter_info;
using spm_counter_value    = rocprofiler_sdk::spm::counter_value;
using spm_timestamp_sample = rocprofiler_sdk::spm::timestamp_sample;
using spm_sample           = rocprofiler_sdk::spm::sample;

template <>
inline void
serialize(std::uint8_t* buffer, const rocprofiler_sdk::spm::sample& item)
{
    size_t     pos          = 0;
    const auto num_counters = static_cast<std::uint32_t>(item.counters.size());
    const auto num_samples  = static_cast<std::uint32_t>(item.samples.size());
    utility::store_value(item.agent_id_handle, buffer, pos);
    utility::store_value(item.dispatch_id, buffer, pos);
    utility::store_value(item.kernel_id, buffer, pos);
    utility::store_value(item.queue_id_handle, buffer, pos);
    utility::store_value(item.correlation_id_internal, buffer, pos);
    utility::store_value(item.correlation_id_ancestor, buffer, pos);
    utility::store_value(item.stream_handle, buffer, pos);
    utility::store_value(item.data_loss, buffer, pos);
    utility::store_value(num_counters, buffer, pos);
    utility::store_value(num_samples, buffer, pos);
    for(const auto& counter : item.counters)
    {
        utility::store_value(counter.counter_id, buffer, pos);
        utility::store_value(counter.counter_instance_id, buffer, pos);
    }
    for(const auto& sample : item.samples)
    {
        utility::store_value(sample.timestamp, buffer, pos);
        utility::store_value(static_cast<std::uint32_t>(sample.values.size()), buffer,
                             pos);
        for(const auto& value : sample.values)
        {
            utility::store_value(value.counter_info_index, buffer, pos);
            utility::store_value(value.value, buffer, pos);
        }
    }
}

template <>
[[nodiscard]] inline rocprofiler_sdk::spm::sample
deserialize(std::uint8_t*& buffer)
{
    rocprofiler_sdk::spm::sample item;
    std::uint32_t                num_counters = 0;
    std::uint32_t                num_samples  = 0;
    utility::parse_value(buffer, item.agent_id_handle, item.dispatch_id, item.kernel_id,
                         item.queue_id_handle, item.correlation_id_internal,
                         item.correlation_id_ancestor, item.stream_handle, item.data_loss,
                         num_counters, num_samples);
    item.counters.resize(num_counters);
    for(auto& counter : item.counters)
    {
        utility::parse_value(buffer, counter.counter_id, counter.counter_instance_id);
    }
    item.samples.resize(num_samples);
    for(auto& sample : item.samples)
    {
        std::uint32_t num_values = 0;
        utility::parse_value(buffer, sample.timestamp, num_values);
        sample.values.resize(num_values);
        for(auto& value : sample.values)
        {
            utility::parse_value(buffer, value.counter_info_index, value.value);
        }
    }
    return item;
}

template <>
[[nodiscard]] inline size_t
get_size(const rocprofiler_sdk::spm::sample& item)
{
    size_t total_size = utility::get_size(
        item.agent_id_handle, item.dispatch_id, item.kernel_id, item.queue_id_handle,
        item.correlation_id_internal, item.correlation_id_ancestor, item.stream_handle,
        item.data_loss, static_cast<std::uint32_t>(item.counters.size()),
        static_cast<std::uint32_t>(item.samples.size()));
    for(const auto& counter : item.counters)
    {
        total_size += utility::get_size(counter.counter_id, counter.counter_instance_id);
    }
    for(const auto& sample : item.samples)
    {
        total_size += utility::get_size(sample.timestamp,
                                        static_cast<std::uint32_t>(sample.values.size()));
        for(const auto& value : sample.values)
        {
            total_size += utility::get_size(value.counter_info_index, value.value);
        }
    }
    return total_size;
}

}  // namespace rocprofsys::trace_cache
