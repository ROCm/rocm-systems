// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/categories.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/collectors/sdk_pmc/types.hpp"
#include "logger/debug.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace rocprofsys::pmc::collectors::sdk_pmc
{

/**
 * @brief Output policy for writing SDK PMC samples to the trace cache.
 *
 * Writes pmc_event_with_sample records to the trace cache buffer storage,
 * reusing the rocm_counter_collection category so that SDK PMC counters
 * appear in the same output pipeline as dispatch-mode counters.
 */
struct cache_policy
{
    /**
     * @brief Initialize trace cache category metadata for SDK PMC.
     */
    static void initialize_category_metadata()
    {
        trace_cache::get_metadata_registry().add_string(
            trait::name<category::rocm_counter_collection>::value);
    }

    /**
     * @brief Initialize track metadata for SDK PMC counters.
     *
     * Tracks are registered dynamically in store_sample when counter names
     * are first seen, since SDK PMC counter names are not known at init time.
     */
    static void initialize_tracks_metadata() {}

    /**
     * @brief Initialize per-device PMC metadata.
     * @param gpu_id GPU device identifier.
     */
    static void initialize_pmc_metadata(size_t gpu_id)
    {
        LOG_DEBUG("Registered PMC metadata for device {}", gpu_id);
    }

    /**
     * @brief Store an SDK PMC sample to the trace cache.
     *
     * Writes one pmc_event_with_sample per counter value to the trace cache
     * buffer storage.
     */
    static void store_sample(size_t device_id, const std::string& /*device_name*/,
                             const enabled_metrics& /*enabled_metrics_cfg*/,
                             const enabled_metrics& /*supported_metrics*/,
                             const metrics& metric_values, uint64_t timestamp)
    {
        for(const auto& cv : metric_values.counters)
        {
            auto track_name = fmt::format(" GPU SDK_PMC {} [{}]", cv.name, device_id);

            trace_cache::get_buffer_storage().store(trace_cache::pmc_event_with_sample{
                static_cast<size_t>(
                    category_enum_id<category::rocm_counter_collection>::value),
                track_name.c_str(), timestamp, "{}", 0, 0, 0, "{}", "{}",
                static_cast<uint32_t>(device_id), static_cast<uint8_t>(agent_type::GPU),
                track_name.c_str(), cv.value, std::nullopt });
        }
    }
};

}  // namespace rocprofsys::pmc::collectors::sdk_pmc
