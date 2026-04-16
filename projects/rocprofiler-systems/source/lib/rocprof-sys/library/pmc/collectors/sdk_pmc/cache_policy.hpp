// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/categories.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/collectors/sdk_pmc/sample.hpp"
#include "library/pmc/collectors/sdk_pmc/types.hpp"
#include "logger/debug.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rocprofsys::pmc::collectors::sdk_pmc
{

/**
 * @brief Output policy for writing SDK PMC samples to the trace cache.
 *
 * Registers pmc_info and track metadata during config(), then stores
 * batched sdk_pmc_sample records during sampling. Processors resolve
 * counter names via metadata_registry at post-processing time.
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
     * Tracks are registered per-device in initialize_pmc_metadata() because
     * track names include the device index.
     */
    static void initialize_tracks_metadata() {}

    /**
     * @brief Initialize per-device PMC metadata for SDK PMC counters.
     *
     * Registers pmc_info and tracks for each counter name provided by the
     * device, and stores the ordered name-to-track mapping in the
     * metadata_registry for processor use.
     *
     * @param gpu_id GPU device identifier.
     * @param counter_names Ordered list of counter names from the device.
     */
    static void initialize_pmc_metadata(size_t                          gpu_id,
                                        const std::vector<std::string>& counter_names)
    {
        constexpr size_t      EVENT_CODE       = 0;
        constexpr size_t      INSTANCE_ID      = 0;
        constexpr const char* LONG_DESCRIPTION = "";
        constexpr const char* COMPONENT        = "";
        constexpr const char* BLOCK            = "";
        constexpr const char* EXPRESSION       = "";
        constexpr const char* TARGET_ARCH      = "GPU";

        auto& registry = trace_cache::get_metadata_registry();

        std::vector<trace_cache::info::sdk_pmc_name_entry> name_entries;
        name_entries.reserve(counter_names.size());

        for(const auto& name : counter_names)
        {
            auto track_name = fmt::format("sdk_pmc_{} [GPU {}]", name, gpu_id);

            // Register pmc_info
            registry.add_pmc_info({ agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE,
                                    INSTANCE_ID, name.c_str(), name.c_str(),
                                    "SDK PMC hardware counter", LONG_DESCRIPTION,
                                    COMPONENT, "count", rocprofsys::trace_cache::ABSOLUTE,
                                    BLOCK, EXPRESSION, 0, 0 });

            // Register track
            registry.add_track({ track_name.c_str(), std::nullopt, "{}" });

            name_entries.push_back({ name, std::move(track_name) });
        }

        // Store ordered name mapping for processor use
        registry.set_sdk_pmc_counter_names(static_cast<uint32_t>(gpu_id),
                                           std::move(name_entries));

        LOG_DEBUG("Registered {} SDK PMC counters for device {}", counter_names.size(),
                  gpu_id);
    }

    /**
     * @brief Store an SDK PMC sample to the trace cache.
     *
     * Writes one sdk_pmc_sample per device per tick with all counter
     * entries (qualified name + value) batched.
     */
    static void store_sample(size_t device_id, const std::string& /*device_name*/,
                             const enabled_metrics& /*enabled_metrics_cfg*/,
                             const enabled_metrics& /*supported_metrics*/,
                             const metrics& metric_values, uint64_t timestamp)
    {
        std::vector<sample_entry> entries;
        entries.reserve(metric_values.counters.size());
        for(const auto& counter : metric_values.counters)
        {
            entries.push_back(sample_entry{ counter.name, counter.value });
        }

        trace_cache::get_buffer_storage().store(trace_cache::sdk_pmc_sample{
            static_cast<uint32_t>(device_id), timestamp, std::move(entries) });
    }
};

}  // namespace rocprofsys::pmc::collectors::sdk_pmc
