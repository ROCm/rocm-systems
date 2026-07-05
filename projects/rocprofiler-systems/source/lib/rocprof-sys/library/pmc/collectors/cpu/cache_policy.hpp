// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/categories.hpp"
#include "core/config.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/pmc/collectors/cpu/sample.hpp"
#include "library/pmc/collectors/cpu/types.hpp"

#include <spdlog/fmt/fmt.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
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

    // Required by base::collector contract; per-cpu tracks need monitored_cpus
    // and are registered in initialize_pmc_metadata instead.
    static void initialize_tracks_metadata() {}

    /**
     * @brief Initialize per-CPU PMC metadata entries.
     *
     * @param socket_id Socket (physical package) ID for agent registration.
     * @param monitored_cpus Set of CPU IDs being monitored on this socket.
     * @param is_first_socket True if this is the first selected socket;
     *        process-level metrics are registered only once, under this socket.
     */
    static void initialize_pmc_metadata(size_t                  socket_id,
                                        const std::set<size_t>& monitored_cpus,
                                        bool                    is_first_socket)
    {
        constexpr size_t        event_code       = 0;
        constexpr size_t        instance_id      = 0;
        constexpr const char*   long_description = "";
        constexpr const char*   component        = "";
        constexpr const char*   block            = "";
        constexpr const char*   expression       = "";
        constexpr const char*   target_arch      = "CPU";
        constexpr std::uint32_t is_constant      = 0;
        constexpr std::uint32_t is_derived       = 0;
        constexpr const char*   extdata          = "{}";

        using ::tim::trait::name;
        auto& registry = trace_cache::get_metadata_registry();

        const std::string freq_base = name<category::cpu_freq>::value;
        const std::string load_base = name<category::cpu_load>::value;

        for(const size_t cpu_id : monitored_cpus)
        {
            const auto freq_name =
                fmt::format("{} [{}] Core [{}]", freq_base, socket_id, cpu_id);
            registry.add_pmc_info(
                { /* type             = */ agent_type::CPU,
                  /* agent_type_index = */ socket_id,
                  /* target_arch      = */ target_arch,
                  /* event_code       = */ event_code,
                  /* instance_id      = */ instance_id,
                  /* name             = */ freq_name.c_str(),
                  /* symbol           = */ freq_name.c_str(),
                  /* description      = */ "CPU Core Frequency",
                  /* long_description = */ long_description,
                  /* component        = */ component,
                  /* units            = */ "MHz",
                  /* value_type       = */ rocprofsys::trace_cache::ABSOLUTE,
                  /* block            = */ block,
                  /* expression       = */ expression,
                  /* is_constant      = */ is_constant,
                  /* is_derived       = */ is_derived,
                  /* extdata          = */ extdata });
            registry.add_track({ freq_name, std::nullopt, extdata });

            const auto load_name =
                fmt::format("{} [{}] Core [{}]", load_base, socket_id, cpu_id);
            registry.add_pmc_info(
                { /* type             = */ agent_type::CPU,
                  /* agent_type_index = */ socket_id,
                  /* target_arch      = */ target_arch,
                  /* event_code       = */ event_code,
                  /* instance_id      = */ instance_id,
                  /* name             = */ load_name.c_str(),
                  /* symbol           = */ load_name.c_str(),
                  /* description      = */ "CPU Core Load Percentage",
                  /* long_description = */ long_description,
                  /* component        = */ component,
                  /* units            = */ trace_cache::PERCENTAGE,
                  /* value_type       = */ rocprofsys::trace_cache::ABSOLUTE,
                  /* block            = */ block,
                  /* expression       = */ expression,
                  /* is_constant      = */ is_constant,
                  /* is_derived       = */ is_derived,
                  /* extdata          = */ extdata });
            registry.add_track({ load_name, std::nullopt, extdata });
        }

        if(!is_first_socket) return;

        auto add_process_pmc = [&, socket_id](const char* metric_name, const char* symbol,
                                              const char* description, const char* units,
                                              const char* value_type) {
            registry.add_pmc_info({ /* type             = */ agent_type::CPU,
                                    /* agent_type_index = */ socket_id,
                                    /* target_arch      = */ target_arch,
                                    /* event_code       = */ event_code,
                                    /* instance_id      = */ instance_id,
                                    /* name             = */ metric_name,
                                    /* symbol           = */ symbol,
                                    /* description      = */ description,
                                    /* long_description = */ long_description,
                                    /* component        = */ component,
                                    /* units            = */ units,
                                    /* value_type       = */ value_type,
                                    /* block            = */ block,
                                    /* expression       = */ expression,
                                    /* is_constant      = */ is_constant,
                                    /* is_derived       = */ is_derived,
                                    /* extdata          = */ extdata });
            registry.add_track({ metric_name, std::nullopt, extdata });
        };

        add_process_pmc(name<category::process_page>::value, "Page RSS",
                        "Process Physical Memory (RSS)", "MB",
                        rocprofsys::trace_cache::ABSOLUTE);

        add_process_pmc(name<category::process_virt>::value, "Virt Mem",
                        "Process Virtual Memory", "MB",
                        rocprofsys::trace_cache::ABSOLUTE);

        add_process_pmc(name<category::process_peak>::value, "Peak RSS",
                        "Process Peak Memory (HWM)", "MB",
                        rocprofsys::trace_cache::ABSOLUTE);

        add_process_pmc(name<category::process_context_switch>::value, "Ctx Switches",
                        "Context Switches", "count", rocprofsys::trace_cache::ABSOLUTE);

        add_process_pmc(name<category::process_page_fault>::value, "Page Faults",
                        "Page Faults", "count", rocprofsys::trace_cache::ABSOLUTE);

        add_process_pmc(name<category::process_user_mode_time>::value, "User Time",
                        "Process CPU Time in User Mode", "sec",
                        rocprofsys::trace_cache::ABSOLUTE);

        add_process_pmc(name<category::process_kernel_mode_time>::value, "Kernel Time",
                        "Process CPU Time in Kernel Mode", "sec",
                        rocprofsys::trace_cache::ABSOLUTE);
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

        assert(device_id <= std::numeric_limits<std::uint32_t>::max() &&
               "socket id exceeds cpu_pmc_sample::device_id width");
        trace_cache::get_buffer_storage().store(trace_cache::cpu_pmc_sample{
            effective, static_cast<std::uint32_t>(device_id), timestamp,
            metric_values.process_data, serialize_frequencies(cpu_data),
            serialize_loads(cpu_data) });
    }

private:
    /**
     * @brief Return cpu_data from the metrics, or cached zero entries on pause.
     *
     * During normal sampling, cpu_data is non-empty and we refresh the cached
     * zero set only when the CPU set changes (size or first/last cpu_id
     * differs). During pause, base::collector creates metrics_t{} with empty
     * cpu_data; we return the cached zero set so Perfetto counter tracks drop
     * to zero.
     *
     * @warning The returned reference is valid only until the next call to
     *          get_effective_cpu_data on this thread.
     */
    static const std::vector<per_cpu_metrics>& get_effective_cpu_data(
        const metrics& metric_values)
    {
        static std::vector<per_cpu_metrics> s_zero_entries;

        if(!metric_values.cpu_data.empty())
        {
            const auto& src = metric_values.cpu_data;
            const bool  set_changed =
                s_zero_entries.size() != src.size() ||
                (!s_zero_entries.empty() &&
                 (s_zero_entries.front().cpu_id != src.front().cpu_id ||
                  s_zero_entries.back().cpu_id != src.back().cpu_id));
            if(set_changed)
            {
                s_zero_entries.clear();
                s_zero_entries.reserve(src.size());
                for(const auto& cpu : src)
                    s_zero_entries.push_back({ cpu.cpu_id, 0.0f, 0.0 });
            }
            return src;
        }
        return s_zero_entries;
    }
};

}  // namespace rocprofsys::pmc::collectors::cpu
