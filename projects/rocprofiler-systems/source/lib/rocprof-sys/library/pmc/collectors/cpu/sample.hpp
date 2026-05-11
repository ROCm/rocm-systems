// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/collectors/cpu/types.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace rocprofsys::pmc::collectors::cpu
{

/**
 * @brief CPU PMC sample type for trace cache.
 *
 * Stores enabled metrics, timestamp, process-level data, and serialized
 * per-CPU frequency/load data as byte vectors.
 */
struct sample : trace_cache::cacheable_t
{
    static constexpr trace_cache::type_identifier_t type_identifier{
        trace_cache::type_identifier_t::cpu_pmc_sample
    };

    sample() = default;
    sample(enabled_metrics _settings, std::uint32_t _device_id, std::uint64_t _timestamp,
           const process_metrics& _process_data, std::vector<std::uint8_t> _freqs,
           std::vector<std::uint8_t> _loads)
    : enabled_metric(_settings)
    , device_id(_device_id)
    , timestamp(_timestamp)
    , process_data(_process_data)
    , freqs(std::move(_freqs))
    , loads(std::move(_loads))
    {}

    enabled_metrics           enabled_metric{};
    std::uint32_t             device_id = 0;
    std::uint64_t             timestamp = 0;
    process_metrics           process_data{};
    std::vector<std::uint8_t> freqs;  // serialized cpu_id+freq pairs
    std::vector<std::uint8_t> loads;  // serialized cpu_id+load pairs

    template <class Archive>
    ROCPROFSYS_INLINE void serialize(Archive& ar)
    {
        // The wire format encodes the packed freqs / loads vectors only; the
        // upstream per_cpu source vector lives only in the live process_metrics
        // and is not stored on disk. Mirror the legacy field order exactly.
        auto enabled_value = static_cast<std::uint32_t>(enabled_metric.value);
        ar(enabled_value, device_id, timestamp, process_data.page_rss,
           process_data.virt_mem, process_data.peak_rss, process_data.context_switches,
           process_data.page_faults, process_data.user_mode_time,
           process_data.kernel_mode_time, freqs, loads);
        if constexpr(std::is_same_v<Archive, trace_cache::input_archive>)
        {
            enabled_metric.value = enabled_value;
        }
    }
};

inline std::vector<std::uint8_t>
serialize_frequencies(const std::vector<per_cpu_metrics>& cpu_data)
{
    constexpr size_t idx_size   = sizeof(size_t);
    constexpr size_t value_size = sizeof(float);

    std::vector<std::uint8_t> result;
    result.resize(cpu_data.size() * (idx_size + value_size));

    size_t offset = 0;
    for(const auto& cpu : cpu_data)
    {
        std::memcpy(result.data() + offset, &cpu.cpu_id, idx_size);
        offset += idx_size;
        std::memcpy(result.data() + offset, &cpu.frequency, value_size);
        offset += value_size;
    }
    return result;
}

inline std::vector<std::uint8_t>
serialize_loads(const std::vector<per_cpu_metrics>& cpu_data)
{
    constexpr size_t idx_size   = sizeof(size_t);
    constexpr size_t value_size = sizeof(double);

    std::vector<std::uint8_t> result;
    result.resize(cpu_data.size() * (idx_size + value_size));

    size_t offset = 0;
    for(const auto& cpu : cpu_data)
    {
        std::memcpy(result.data() + offset, &cpu.cpu_id, idx_size);
        offset += idx_size;
        std::memcpy(result.data() + offset, &cpu.load, value_size);
        offset += value_size;
    }
    return result;
}

}  // namespace rocprofsys::pmc::collectors::cpu

namespace rocprofsys::trace_cache
{

/// @brief CPU PMC sample type alias
using cpu_pmc_sample = pmc::collectors::cpu::sample;

}  // namespace rocprofsys::trace_cache
