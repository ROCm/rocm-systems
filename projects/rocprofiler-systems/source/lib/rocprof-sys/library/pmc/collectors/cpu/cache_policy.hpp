// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/config.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/pmc/collectors/cpu/sample.hpp"
#include "library/pmc/collectors/cpu/types.hpp"

#include <spdlog/fmt/fmt.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace rocprofsys::pmc::collectors::cpu
{

/**
 * @brief Output policy for writing CPU PMC samples to the trace cache.
 *
 * Handles registration of CPU-specific metadata (tracks, PMC info)
 * and serialization of CPU metric samples into the trace cache.
 */
struct cache_policy
{
    static void initialize_category_metadata()
    {
        trace_cache::get_metadata_registry().add_string(
            trait::name<category::cpu_freq>::value);
    }

    static void initialize_tracks_metadata()
    {
        const auto thread_id = std::nullopt;

        trace_cache::get_metadata_registry().add_track(
            { "cpu_frequency", thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track({ "cpu_load", thread_id, "{}" });
    }

    /**
     * @brief Initialize per-CPU PMC metadata entries.
     *
     * @param socket_id Socket (physical package) ID for agent registration.
     * @param monitored_cpus Set of CPU IDs being monitored on this socket.
     * @param is_first_socket True if this is the first selected socket —
     *        process-level metrics are registered only once, under this socket.
     */
    static void initialize_pmc_metadata(size_t                  socket_id,
                                        const std::set<size_t>& monitored_cpus,
                                        bool                    is_first_socket)
    {
        constexpr size_t      EVENT_CODE       = 0;
        constexpr size_t      INSTANCE_ID      = 0;
        constexpr const char* LONG_DESCRIPTION = "";
        constexpr const char* COMPONENT        = "";
        constexpr const char* BLOCK            = "";
        constexpr const char* EXPRESSION       = "";
        constexpr const char* TARGET_ARCH      = "CPU";

        for(const auto cpu_id : monitored_cpus)
        {
            const auto freq_name = fmt::format("cpu{}_frequency", cpu_id);
            trace_cache::get_metadata_registry().add_pmc_info(
                { agent_type::CPU, socket_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
                  freq_name.c_str(), freq_name.c_str(), "CPU Core Frequency",
                  LONG_DESCRIPTION, COMPONENT, "MHz", rocprofsys::trace_cache::ABSOLUTE,
                  BLOCK, EXPRESSION, 0, 0, "{}" });

            const auto load_name = fmt::format("cpu{}_load", cpu_id);
            trace_cache::get_metadata_registry().add_pmc_info(
                { agent_type::CPU, socket_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
                  load_name.c_str(), load_name.c_str(), "CPU Core Load Percentage",
                  LONG_DESCRIPTION, COMPONENT, trace_cache::PERCENTAGE,
                  rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0, "{}" });
        }

        // Process-level metrics are process-wide; register under first selected socket
        if(!is_first_socket) return;

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::CPU, socket_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              "process_page_rss", "Page RSS", "Process Physical Memory (RSS)",
              LONG_DESCRIPTION, COMPONENT, "bytes", rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0, "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::CPU, socket_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              "process_virt_mem", "Virt Mem", "Process Virtual Memory", LONG_DESCRIPTION,
              COMPONENT, "bytes", rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0,
              0, "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::CPU, socket_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              "process_peak_rss", "Peak RSS", "Process Peak Memory (HWM)",
              LONG_DESCRIPTION, COMPONENT, "bytes", rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0, "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::CPU, socket_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              "process_ctx_switches", "Ctx Switches", "Context Switches",
              LONG_DESCRIPTION, COMPONENT, "count", rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0, "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::CPU, socket_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              "process_page_faults", "Page Faults", "Page Faults", LONG_DESCRIPTION,
              COMPONENT, "count", rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0,
              0, "{}" });
    }

    /**
     * @brief Store a CPU PMC sample to the trace cache.
     *
     * Thread-safety: store_sample is called only from the sampler thread,
     * serialized by type_mutex<category::amd_smi> in pmc::sample/pause.
     * The static s_zero_entries is safe under this single-writer invariant.
     *
     * @param device_id Socket (physical package) ID.
     * @param device_name Device name (unused for CPU).
     * @param enabled_metrics_cfg Metrics enabled by configuration.
     * @param supported_metrics Metrics supported by the device.
     * @param metric_values Collected metric values.
     * @param timestamp Sample timestamp in nanoseconds.
     */
    static void store_sample(size_t device_id, const std::string& /*device_name*/,
                             const enabled_metrics& enabled_metrics_cfg,
                             const enabled_metrics& supported_metrics,
                             const metrics& metric_values, std::uint64_t timestamp)
    {
        enabled_metrics effective;
        effective.value = enabled_metrics_cfg.value & supported_metrics.value;

        const auto& cpu_data = get_effective_cpu_data(metric_values);

        trace_cache::get_buffer_storage().store(trace_cache::cpu_pmc_sample{
            effective, static_cast<std::uint32_t>(device_id), timestamp,
            metric_values.process_data, serialize_frequencies(cpu_data),
            serialize_loads(cpu_data) });
    }

private:
    /**
     * @brief Return cpu_data from the metrics, or cached zero entries on pause.
     *
     * During normal sampling, cpu_data is non-empty and we cache the CPU IDs.
     * During pause, base::collector creates metrics_t{} which has empty cpu_data.
     * In that case, we return zero entries for all previously-seen CPU IDs so
     * that Perfetto counter tracks drop to zero.
     */
    static const std::vector<per_cpu_metrics>& get_effective_cpu_data(
        const metrics& metric_values)
    {
        static std::vector<per_cpu_metrics> s_zero_entries;

        if(!metric_values.cpu_data.empty())
        {
            s_zero_entries.clear();
            s_zero_entries.reserve(metric_values.cpu_data.size());
            for(const auto& cpu : metric_values.cpu_data)
                s_zero_entries.push_back({ cpu.cpu_id, 0.0f, 0.0 });
            return metric_values.cpu_data;
        }
        return s_zero_entries;
    }
};

}  // namespace rocprofsys::pmc::collectors::cpu
