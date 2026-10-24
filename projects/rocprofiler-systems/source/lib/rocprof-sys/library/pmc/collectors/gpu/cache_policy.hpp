// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "core/categories.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/collectors/gpu/counters.hpp"
#include "library/pmc/collectors/gpu/types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu
{

/**
 * @brief Output policy for writing AMD SMI GPU samples to the trace cache.
 *
 * Tracks and pmc_info rows come from counters(), for every counter (as on develop: 419
 * per GPU). Each GPU registers one PMC schema for the counters it collects, at its first
 * sample, and every sample is a pmc_values_sample with those values.
 */
struct cache_policy
{
    static void initialize_category_metadata()
    {
        trace_cache::get_metadata_registry().add_string(
            trait::name<category::amd_smi>::value);
    }

    static void initialize_tracks_metadata()
    {
        for(const auto& item : counters())
        {
            if(!is_fallback(item))
            {
                trace_cache::get_metadata_registry().add_track(
                    { .track_name = item.rocpd_track,
                      .thread_id  = std::nullopt,
                      .extdata    = "{}" });
            }
        }
    }

    /**
     * @brief Register the pmc_info rows of one GPU.
     *
     * @param gpu_id AMD SMI index of the GPU (the agent's GPU type index)
     */
    static void initialize_pmc_metadata(size_t gpu_id)
    {
        for(const auto& item : counters())
        {
            if(!is_fallback(item))
            {
                register_pmc_info(gpu_id, item);
            }
        }
    }

    /**
     * @brief Store one GPU sample as a pmc_values_sample.
     *
     * Runs under the sampler's amd_smi lock (sample and pause), so the per-GPU layouts
     * need no lock of their own.
     *
     * @param device_id AMD SMI index of the GPU
     * @param enabled_metrics_cfg Metrics requested by the user
     * @param supported_metrics Metrics the GPU supports
     * @param metric_values Collected metric values
     * @param timestamp Sample timestamp in nanoseconds
     */
    static void store_sample(size_t                              device_id,
                             [[maybe_unused]] const std::string& device_name,
                             const enabled_metrics&              enabled_metrics_cfg,
                             const enabled_metrics&              supported_metrics,
                             const metrics& metric_values, std::uint64_t timestamp)
    {
        auto& layout =
            layout_for(device_id, enabled_metrics_cfg.value & supported_metrics.value);
        if(layout.collected.empty())
        {
            return;
        }

        std::ranges::transform(
            layout.collected, layout.values.begin(),
            [&metric_values](const counter* item) { return item->read(metric_values); });
        trace_cache::get_buffer_storage().store(
            trace_cache::pmc_values_sample{ layout.schema_id, timestamp, layout.values });
    }

private:
    /** The schema of one GPU and the counters it collects, in schema order. */
    struct device_layout
    {
        std::uint32_t               enabled_mask = 0;
        std::uint32_t               schema_id    = 0;
        std::vector<const counter*> collected;
        std::vector<double>         values;  ///< scratch for one sample
    };

    static device_layout& layout_for(std::size_t device_id, std::uint32_t enabled_mask)
    {
        static std::unordered_map<std::size_t, device_layout> s_layouts;

        auto found = s_layouts.find(device_id);
        if(found == s_layouts.end() || found->second.enabled_mask != enabled_mask)
        {
            found = s_layouts
                        .insert_or_assign(device_id, make_layout(device_id, enabled_mask))
                        .first;
        }
        return found->second;
    }

    static device_layout make_layout(std::size_t device_id, std::uint32_t enabled_mask)
    {
        device_layout layout{ .enabled_mask = enabled_mask,
                              .schema_id    = 0,
                              .collected    = collected_counters(enabled_mask),
                              .values       = {} };
        layout.values.resize(layout.collected.size());
        layout.schema_id = trace_cache::get_metadata_registry().add_pmc_schema(
            make_schema(device_id, layout.collected));
        return layout;
    }

    static void register_pmc_info(std::size_t gpu_id, const counter& item)
    {
        trace_cache::get_metadata_registry().add_pmc_info(
            { .type             = agent_type::gpu,
              .agent_type_index = gpu_id,
              .target_arch      = "GPU",
              .event_code       = 0,
              .instance_id      = 0,
              .name             = item.pmc_name,
              .symbol           = item.symbol,
              .description      = item.description,
              .long_description = "",
              .component        = "",
              .units            = item.rocpd_units,
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = "",
              .expression       = "",
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });
    }
};

}  // namespace rocprofsys::pmc::collectors::gpu
