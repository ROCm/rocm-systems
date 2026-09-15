// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/categories.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "library/pmc/collectors/hipfile/sample.hpp"
#include "library/pmc/collectors/hipfile/types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace rocprofsys::pmc::collectors::hipfile
{

/**
 * @brief Output policy for writing hipFile samples to the trace cache.
 *
 * Every track is GPU-indexed. hipFile also exposes process-global counters (file and
 * buffer registrations), but those are not GPU measurements and the profiler has no
 * process-scoped device track to put them on; parking them on GPU 0, as an earlier
 * revision did, made them read as GPU 0's activity. They are omitted rather than
 * misattributed.
 */
struct cache_policy
{
    static void initialize_category_metadata()
    {
        trace_cache::get_metadata_registry().add_string(
            trait::name<category::hipfile>::value);
    }

    /// Tracks are per GPU, so they are registered in initialize_pmc_metadata() once the
    /// device set is known rather than as a fixed global list.
    static void initialize_tracks_metadata() {}

    /**
     * @brief Register every hipFile track and its PMC metadata for one GPU.
     *
     * Runs at config() time for the whole enumerated device set, so track metadata no
     * longer appears lazily on first activity.
     *
     * @param gpu_id Profiler GPU index (@c device_type_index).
     */
    static void initialize_pmc_metadata(std::size_t gpu_id)
    {
        constexpr std::size_t k_event_code       = 0;
        constexpr std::size_t k_instance_id      = 0;
        constexpr const char* k_long_description = "";
        constexpr const char* k_component        = "";
        constexpr const char* k_block            = "";
        constexpr const char* k_expression       = "";
        constexpr const char* k_target_arch      = "GPU";

        for(const auto& metric : k_metric_table)
        {
            const auto name = track_name(gpu_id, metric.suffix);

            trace_cache::get_metadata_registry().add_track({ name, std::nullopt, "{}" });

            // ABSOLUTE is accurate for both shapes here: the counters are cumulative
            // totals and the bandwidths are instantaneous rates. Neither is a delta.
            trace_cache::get_metadata_registry().add_pmc_info(
                { agent_type::gpu, gpu_id, k_target_arch, k_event_code, k_instance_id,
                  pmc_name(metric.suffix), metric.suffix,
                  trait::name<category::hipfile>::description, k_long_description,
                  k_component, metric.unit, rocprofsys::trace_cache::ABSOLUTE, k_block,
                  k_expression, 0, 0, "{}" });
        }
    }

    /**
     * @brief Store one GPU's hipFile sample.
     *
     * @param device_id Profiler GPU index (@c device_type_index).
     * @param enabled_metrics_cfg Metrics enabled by configuration.
     * @param supported_metrics Metrics this device can supply.
     * @param metric_values Collected values.
     * @param timestamp Sample timestamp in nanoseconds.
     */
    // needs a refactor of every collector's store_sample
    // NOLINTNEXTLINE(readability-function-size)
    static void store_sample(std::size_t                         device_id,
                             [[maybe_unused]] const std::string& device_name,
                             const enabled_metrics&              enabled_metrics_cfg,
                             const enabled_metrics&              supported_metrics,
                             const metrics& metric_values, std::uint64_t timestamp)
    {
        // A failed hipFile query is not a measurement of zero, so it produces no sample.
        // Note this is not the paused path: pause() emits a value-initialized metrics,
        // which has query_failed false and so still drops the tracks to zero.
        if(metric_values.query_failed)
        {
            return;
        }

        enabled_metrics active;
        active.value = enabled_metrics_cfg.value & supported_metrics.value;

        // The whole per-GPU snapshot goes out as one record. Track names and PMC
        // identifiers are derived from k_metric_table during post-processing, so this
        // path - which runs on the sampling thread shared with every other collector -
        // builds no strings and allocates nothing.
        trace_cache::get_buffer_storage().store(sample{
            active, static_cast<std::uint32_t>(device_id), timestamp, metric_values });
    }
};

}  // namespace rocprofsys::pmc::collectors::hipfile
