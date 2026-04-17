// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "core/categories.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/pmc/collectors/gpu_perf_counter/sample.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/pmc/device_providers/rocprofiler_sdk/provider.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{

/**
 * @brief Output policy for writing SDK PMC samples to the trace cache.
 *
 * Registers pmc_info and track metadata during config(), then stores
 * batched gpu_perf_counter_sample records during sampling. Processors resolve
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
     * Static tracks cannot be registered here because track names include the
     * device index and counter name. Per-device tracks are registered via
     * initialize_device_tracks().
     */
    static void initialize_tracks_metadata() {}

    /**
     * @brief Format a track name for an SDK PMC counter on a given device.
     */
    static std::string format_track_name(size_t gpu_id, const std::string& qname)
    {
        return fmt::format("GPU [{}] {} (S)", gpu_id, qname);
    }

    /**
     * @brief Register per-device tracks for SDK PMC counters.
     *
     * Called once per device during config(). Registers a track per qualified
     * counter name, mirroring the gpu collector's track registration pattern.
     *
     * @param gpu_id GPU device identifier.
     * @param qualified_names Ordered list of qualified counter names from the device.
     */
    static void initialize_device_tracks(size_t                          gpu_id,
                                         const std::vector<std::string>& qualified_names)
    {
        auto& registry = trace_cache::get_metadata_registry();

        for(const auto& qname : qualified_names)
        {
            auto track_name = format_track_name(gpu_id, qname);
            registry.add_track({ track_name, std::nullopt, "{}" });
        }
    }

    using counter_metadata = device_providers::rocprofiler_sdk::counter_metadata;

    /**
     * @brief Initialize per-device PMC metadata for SDK PMC counters.
     *
     * Registers pmc_info (with block, expression, is_constant, is_derived from
     * SDK counter info) and tracks for each qualified counter name. Stores the
     * ordered name-to-track mapping in the metadata_registry for processor use.
     *
     * @param gpu_id GPU device identifier.
     * @param qualified_names Ordered list of qualified counter names from the device.
     * @param counter_meta Per-counter metadata from rocprofiler_counter_info_v1_t.
     */
    static void initialize_pmc_metadata(size_t                          gpu_id,
                                        const std::vector<std::string>& qualified_names,
                                        const std::vector<counter_metadata>& counter_meta)
    {
        constexpr size_t      EVENT_CODE       = 0;
        constexpr size_t      INSTANCE_ID      = 0;
        constexpr const char* LONG_DESCRIPTION = "";
        constexpr const char* COMPONENT        = "";
        constexpr const char* TARGET_ARCH      = "GPU";

        auto& registry = trace_cache::get_metadata_registry();

        std::vector<trace_cache::info::gpu_perf_counter_name_entry> name_entries;
        name_entries.reserve(qualified_names.size());

        for(const auto& qname : qualified_names)
        {
            // Find matching counter metadata by base name prefix
            auto base_name = std::string_view{ qname };
            auto bracket   = base_name.find('[');
            if(bracket != std::string_view::npos)
            {
                base_name = base_name.substr(0, bracket);
            }

            auto meta_it = std::find_if(
                counter_meta.begin(), counter_meta.end(),
                [&](const counter_metadata& m) { return m.name == base_name; });

            auto     description = std::string{ "SDK PMC hardware counter" };
            auto     block       = std::string{};
            auto     expression  = std::string{};
            uint32_t is_constant = 0;
            uint32_t is_derived  = 0;

            if(meta_it != counter_meta.end())
            {
                if(!meta_it->description.empty()) description = meta_it->description;
                block       = meta_it->block;
                expression  = meta_it->expression;
                is_constant = meta_it->is_constant ? 1 : 0;
                is_derived  = meta_it->is_derived ? 1 : 0;
            }

            registry.add_pmc_info({ agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE,
                                    INSTANCE_ID, qname, qname, description,
                                    LONG_DESCRIPTION, COMPONENT, "count",
                                    rocprofsys::trace_cache::ABSOLUTE, block, expression,
                                    is_constant, is_derived, "{}" });

            name_entries.push_back({ qname, format_track_name(gpu_id, qname) });
        }

        registry.set_gpu_perf_counter_counter_names(static_cast<uint32_t>(gpu_id),
                                                    std::move(name_entries));

        LOG_DEBUG("Registered {} SDK PMC counters for device {}", qualified_names.size(),
                  gpu_id);
    }

    /**
     * @brief Store an SDK PMC sample to the trace cache.
     *
     * Filters metric_values to only include counters that are in the
     * enabled ∩ supported set, then writes one gpu_perf_counter_sample
     * per device per tick.
     */
    static void store_sample(size_t device_id, const std::string& /*device_name*/,
                             const enabled_metrics& /*enabled_metrics_cfg*/,
                             const enabled_metrics& /*supported_metrics*/,
                             const metrics& metric_values, uint64_t timestamp)
    {
        const auto& counters = metric_values.counters;
        if(counters.empty()) return;

        std::vector<sample_entry> entries;
        entries.reserve(counters.size());
        for(const auto& ctr : counters)
            entries.push_back({ ctr.name, ctr.value });

        trace_cache::get_buffer_storage().store(trace_cache::gpu_perf_counter_sample{
            static_cast<uint32_t>(device_id), timestamp, std::move(entries) });
    }
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
