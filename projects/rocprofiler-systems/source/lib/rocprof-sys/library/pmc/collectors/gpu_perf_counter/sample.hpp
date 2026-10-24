// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/cache_type_traits.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{
/**
 * @brief SDK PMC sample type for trace cache serialization.
 *
 * Stores a snapshot of GPU hardware counter values collected via
 * rocprofiler_sample_device_counting_service(). Each entry carries its
 * own qualified counter name including dimension positions.
 */
struct sample : trace_cache::cacheable_t
{
    static constexpr trace_cache::type_identifier_t type_identifier{
        trace_cache::type_identifier_t::gpu_perf_counter_sample
    };

    sample() = default;
    sample(std::uint32_t dev_id, std::uint64_t time_ns, std::vector<counter_value> ent)
    : device_id(dev_id)
    , timestamp(time_ns)
    , entries(std::move(ent))
    {}

    std::uint32_t              device_id = 0;
    std::uint64_t              timestamp = 0;
    std::vector<counter_value> entries;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(device_id, timestamp, entries);
    }
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter

namespace rocprofsys::trace_cache::type_traits
{

// counter_value has no padding (see its static_assert), so its bytes are the format.
template <>
struct explicitly_raw<pmc::collectors::gpu_perf_counter::counter_value> : std::true_type
{};

}  // namespace rocprofsys::trace_cache::type_traits

namespace rocprofsys::trace_cache
{

/// @brief SDK PMC sample type alias
using gpu_perf_counter_sample = pmc::collectors::gpu_perf_counter::sample;

}  // namespace rocprofsys::trace_cache
