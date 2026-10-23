// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/collectors/cpu/types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::collectors::cpu
{

/** @brief Frequency of one CPU core in MHz. */
struct core_frequency
{
    std::size_t cpu_id = 0;
    float       value  = 0.0f;

    bool operator==(const core_frequency&) const = default;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(cpu_id, value);
    }
};

/** @brief Load of one CPU core in percent. */
struct core_load
{
    std::size_t cpu_id = 0;
    double      value  = 0.0;

    bool operator==(const core_load&) const = default;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(cpu_id, value);
    }
};

/**
 * @brief CPU PMC sample type for trace cache.
 *
 * Stores enabled metrics, timestamp, process-level data, and per-core frequency
 * and load entries.
 */
struct sample : trace_cache::cacheable_t
{
    static constexpr trace_cache::type_identifier_t type_identifier{
        trace_cache::type_identifier_t::cpu_pmc_sample
    };

    sample() = default;
    sample(enabled_metrics settings, std::uint32_t socket_id, std::uint64_t timestamp_ns,
           const process_metrics&      process,
           std::vector<core_frequency> per_core_frequencies,
           std::vector<core_load>      per_core_loads)
    : enabled_metric(settings)
    , device_id(socket_id)
    , timestamp(timestamp_ns)
    , process_data(process)
    , freqs(std::move(per_core_frequencies))
    , loads(std::move(per_core_loads))
    {}

    enabled_metrics             enabled_metric{};
    std::uint32_t               device_id = 0;
    std::uint64_t               timestamp = 0;
    process_metrics             process_data{};
    std::vector<core_frequency> freqs;
    std::vector<core_load>      loads;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(enabled_metric, device_id, timestamp, process_data, freqs, loads);
    }
};

inline std::vector<core_frequency>
core_frequencies(const std::vector<per_cpu_metrics>& cpu_data)
{
    std::vector<core_frequency> result(cpu_data.size());
    std::ranges::transform(cpu_data, result.begin(), [](const per_cpu_metrics& cpu) {
        return core_frequency{ .cpu_id = cpu.cpu_id, .value = cpu.frequency };
    });
    return result;
}

inline std::vector<core_load>
core_loads(const std::vector<per_cpu_metrics>& cpu_data)
{
    std::vector<core_load> result(cpu_data.size());
    std::ranges::transform(cpu_data, result.begin(), [](const per_cpu_metrics& cpu) {
        return core_load{ .cpu_id = cpu.cpu_id, .value = cpu.load };
    });
    return result;
}

}  // namespace rocprofsys::pmc::collectors::cpu

namespace rocprofsys::trace_cache
{

/// @brief CPU PMC sample type alias
using cpu_pmc_sample = pmc::collectors::cpu::sample;

}  // namespace rocprofsys::trace_cache
