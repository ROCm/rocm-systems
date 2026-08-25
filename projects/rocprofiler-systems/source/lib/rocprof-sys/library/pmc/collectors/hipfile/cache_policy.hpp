// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "core/categories.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
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
        constexpr std::size_t EVENT_CODE       = 0;
        constexpr std::size_t INSTANCE_ID      = 0;
        constexpr const char* LONG_DESCRIPTION = "";
        constexpr const char* COMPONENT        = "";
        constexpr const char* BLOCK            = "";
        constexpr const char* EXPRESSION       = "";
        constexpr const char* TARGET_ARCH      = "GPU";

        for(const auto& metric : METRIC_TABLE)
        {
            const auto name = track_name(gpu_id, metric.suffix);

            trace_cache::get_metadata_registry().add_track({ name, std::nullopt, "{}" });

            // ABSOLUTE is accurate for both shapes here: the counters are cumulative
            // totals and the bandwidths are instantaneous rates. Neither is a delta.
            trace_cache::get_metadata_registry().add_pmc_info(
                { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
                  pmc_name(metric.suffix), metric.suffix,
                  trait::name<category::hipfile>::description, LONG_DESCRIPTION,
                  COMPONENT, metric.unit, rocprofsys::trace_cache::ABSOLUTE, BLOCK,
                  EXPRESSION, 0, 0, "{}" });
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
    static void store_sample(std::size_t                         device_id,
                             [[maybe_unused]] const std::string& device_name,
                             const enabled_metrics&              enabled_metrics_cfg,
                             const enabled_metrics&              supported_metrics,
                             const metrics& metric_values, std::uint64_t timestamp)
    {
        // A failed hipFile query is not a measurement of zero, so it produces no sample.
        // Note this is not the paused path: pause() emits a value-initialized metrics,
        // which has query_failed false and so still drops the tracks to zero.
        if(metric_values.query_failed) return;

        const std::uint32_t active = enabled_metrics_cfg.value & supported_metrics.value;

        for(const auto& metric : METRIC_TABLE)
        {
            if((active & (1U << metric.bit)) == 0U) continue;

            // Display label for the counter track, and the device-independent identifier
            // RocPD joins on. Perfetto shows the former; rocpd_info_pmc stores the
            // latter.
            const auto name = track_name(device_id, metric.suffix);

            trace_cache::get_buffer_storage().store(trace_cache::pmc_event_with_sample{
                static_cast<std::size_t>(category_enum_id<category::hipfile>::value),
                name, timestamp, "{}", 0, 0, 0, "{}", "{}",
                static_cast<std::uint32_t>(device_id),
                static_cast<std::uint8_t>(agent_type::GPU), pmc_name(metric.suffix),
                metric.value(metric_values), std::nullopt });
        }
    }
};

}  // namespace rocprofsys::pmc::collectors::hipfile
