// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/perfetto.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{

namespace detail
{

struct gpu_perf_counter_perfetto_sample
{
    std::uint64_t timestamp = 0;
    metrics       values;
};

struct gpu_perf_counter_perfetto_device_data
{
    std::unique_ptr<std::vector<gpu_perf_counter_perfetto_sample>> samples;
    std::unordered_map<counter_id_t, size_t>                       counter_tracks;
};

inline std::unordered_map<size_t, gpu_perf_counter_perfetto_device_data>&
get_perfetto_data()
{
    static std::unordered_map<size_t, gpu_perf_counter_perfetto_device_data> data;
    return data;
}

}  // namespace detail

struct perfetto_policy
{
    using counter_track = perfetto_counter_track<metrics>;

    template <typename DeviceEntryVector>
    static void init_storage(const DeviceEntryVector& device_entries)
    {
        for(const auto& entry : device_entries)
        {
            auto idx                         = entry.device->get_index();
            detail::get_perfetto_data()[idx] = {
                std::make_unique<std::vector<detail::gpu_perf_counter_perfetto_sample>>(),
                {}
            };
        }
    }

    static void setup_counter_tracks(size_t                               device_index,
                                     const std::vector<counter_metadata>& counter_meta)
    {
        auto it = detail::get_perfetto_data().find(device_index);
        if(it == detail::get_perfetto_data().end()) return;

        for(const auto& meta : counter_meta)
        {
            auto qname      = make_qualified_name(meta);
            auto track_name = format_track_name(device_index, qname);
            auto track_id   = counter_track::emplace(device_index, track_name, "count");
            it->second.counter_tracks[meta.counter_id] = track_id;
            LOG_DEBUG("Created Perfetto counter track: {}", track_name);
        }
    }

    static void store_sample(size_t device_index, const metrics& metric_values,
                             std::uint64_t timestamp)
    {
        auto it = detail::get_perfetto_data().find(device_index);
        if(it == detail::get_perfetto_data().end()) return;

        it->second.samples->emplace_back(
            detail::gpu_perf_counter_perfetto_sample{ timestamp, metric_values });
    }

    static void post_process(const enabled_metrics& /*enabled*/)
    {
        const auto& thread_info = thread_info::get(0, InternalTID);
        if(!thread_info) return;

        for(const auto& entry : detail::get_perfetto_data())
        {
            const auto  device_index = entry.first;
            const auto& data         = entry.second;
            if(!data.samples) continue;

            LOG_DEBUG("Post-processing {} samples for device {}", data.samples->size(),
                      device_index);

            for(const auto& sample : *data.samples)
            {
                if(!thread_info->is_valid_time(sample.timestamp)) continue;

                for(const auto& cv : sample.values)
                {
                    auto track_it = data.counter_tracks.find(cv.counter_id);
                    if(track_it == data.counter_tracks.end()) continue;

                    TRACE_COUNTER("rocm_counter_collection",
                                  counter_track::at(device_index, track_it->second),
                                  sample.timestamp, cv.value);
                }
            }
        }
    }
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
