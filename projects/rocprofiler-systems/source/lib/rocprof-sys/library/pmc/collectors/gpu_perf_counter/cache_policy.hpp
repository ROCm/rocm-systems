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
#include "logger/debug.hpp"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{

struct cache_policy
{
    static void initialize_category_metadata()
    {
        trace_cache::get_metadata_registry().add_string(
            trait::name<category::rocm_counter_collection>::value);
    }

    static void initialize_tracks_metadata() { /* no-op */ }

    static void initialize_pmc_metadata(size_t                               gpu_id,
                                        const std::vector<counter_metadata>& counter_meta)
    {
        constexpr size_t      event_code       = 0;
        constexpr size_t      instance_id      = 0;
        constexpr const char* long_description = "";
        constexpr const char* component        = "";
        constexpr const char* target_arch      = "GPU";

        auto& registry = trace_cache::get_metadata_registry();

        std::vector<trace_cache::info::gpu_perf_counter_name_entry> name_entries;
        name_entries.reserve(counter_meta.size());

        for(const auto& meta : counter_meta)
        {
            auto qname      = make_qualified_name(meta);
            auto track_name = format_track_name(gpu_id, qname);

            registry.add_track({ track_name, std::nullopt, "{}" });

            registry.add_pmc_info({ agent_type::GPU, gpu_id, target_arch, event_code,
                                    instance_id, qname, qname,
                                    meta.description.empty() ? "SDK PMC hardware counter"
                                                             : meta.description,
                                    long_description, component, "count",
                                    rocprofsys::trace_cache::ABSOLUTE, meta.block,
                                    meta.expression, meta.is_constant ? 1U : 0U,
                                    meta.is_derived ? 1U : 0U, "{}" });

            name_entries.push_back(
                { meta.counter_id, std::move(qname), std::move(track_name) });
        }

        registry.set_gpu_perf_counter_counter_names(static_cast<std::uint32_t>(gpu_id),
                                                    std::move(name_entries));

        LOG_DEBUG("Registered {} SDK PMC counters for device {}", counter_meta.size(),
                  gpu_id);
    }

    static void store_sample(size_t device_id, const std::string& /*device_name*/,
                             const enabled_metrics& /*enabled_metrics_cfg*/,
                             const enabled_metrics& /*supported_metrics*/,
                             const metrics& metric_values, std::uint64_t timestamp)
    {
        if(metric_values.empty()) return;

        trace_cache::get_buffer_storage().store(trace_cache::gpu_perf_counter_sample{
            static_cast<std::uint32_t>(device_id), timestamp, metric_values });
    }
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
