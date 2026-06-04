// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/perfetto.hpp"
#include "library/pmc/collectors/cpu/types.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"

#include <spdlog/fmt/fmt.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace rocprofsys::pmc::collectors::cpu
{

namespace detail
{

struct cpu_perfetto_sample
{
    std::uint64_t timestamp;
    metrics       metric_values;
};

struct cpu_track_set
{
    std::map<size_t, size_t> freq_tracks;
    std::map<size_t, size_t> load_tracks;
};

inline std::unique_ptr<std::vector<cpu_perfetto_sample>>&
get_cpu_samples()
{
    static std::unique_ptr<std::vector<cpu_perfetto_sample>> data;
    return data;
}

inline cpu_track_set&
get_cpu_tracks()
{
    static cpu_track_set tracks;
    return tracks;
}

}  // namespace detail

/**
 * @brief Output policy for writing CPU PMC samples to Perfetto traces.
 *
 * Creates per-CPU counter tracks for frequency and load, buffers samples,
 * and flushes them during post-processing.
 */
struct perfetto_policy
{
    using counter_track = perfetto_counter_track<metrics>;

    static void init_storage()
    {
        detail::get_cpu_samples() =
            std::make_unique<std::vector<detail::cpu_perfetto_sample>>();
    }

    static void setup_counter_tracks(size_t                  socket_id,
                                     const std::set<size_t>& monitored_cpus,
                                     const enabled_metrics&  enabled)
    {
        auto& tracks = detail::get_cpu_tracks();

        for(const auto cpu_id : monitored_cpus)
        {
            if(enabled.bits.frequency)
            {
                auto name =
                    fmt::format("CPU [{}] Core [{}] Frequency (S)", socket_id, cpu_id);
                auto track_id = counter_track::emplace(socket_id, name, "MHz");
                tracks.freq_tracks[cpu_id] = track_id;
            }

            if(enabled.bits.load)
            {
                auto name = fmt::format("CPU [{}] Core [{}] Load (S)", socket_id, cpu_id);
                auto track_id              = counter_track::emplace(socket_id, name, "%");
                tracks.load_tracks[cpu_id] = track_id;
            }
        }
    }

    static void store_sample([[maybe_unused]] size_t device_index,
                             const metrics& metric_values, std::uint64_t timestamp)
    {
        if(detail::get_cpu_samples())
        {
            detail::get_cpu_samples()->emplace_back(
                detail::cpu_perfetto_sample{ timestamp, metric_values });
        }
    }

    static void post_process(size_t socket_id, const std::set<size_t>& /*monitored_cpus*/,
                             const enabled_metrics& enabled)
    {
        if(!detail::get_cpu_samples()) return;

        auto& samples = *detail::get_cpu_samples();

        LOG_DEBUG(
            "[CPU perfetto_policy] Post-processing {} CPU PMC samples for socket {}",
            samples.size(), socket_id);

        const auto& thread_info = thread_info::get(0, InternalTID);
        if(!thread_info) return;

        auto& tracks = detail::get_cpu_tracks();

        for(const auto& sample : samples)
        {
            const auto ts = sample.timestamp;
            if(!thread_info->is_valid_time(ts)) continue;

            for(const auto& cpu : sample.metric_values.cpu_data)
            {
                if(enabled.bits.frequency)
                {
                    const auto it = tracks.freq_tracks.find(cpu.cpu_id);
                    if(it != tracks.freq_tracks.end())
                    {
                        TRACE_COUNTER("cpu_frequency",
                                      counter_track::at(socket_id, it->second), ts,
                                      static_cast<double>(cpu.frequency));
                    }
                }

                if(enabled.bits.load)
                {
                    const auto it = tracks.load_tracks.find(cpu.cpu_id);
                    if(it != tracks.load_tracks.end())
                    {
                        TRACE_COUNTER("cpu_load",
                                      counter_track::at(socket_id, it->second), ts,
                                      cpu.load);
                    }
                }
            }
        }

        samples.clear();
    }
};

}  // namespace rocprofsys::pmc::collectors::cpu
