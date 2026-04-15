// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/perfetto.hpp"
#include "library/pmc/collectors/sdk_pmc/types.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rocprofsys::pmc::collectors::sdk_pmc
{

namespace detail
{

struct sdk_pmc_perfetto_sample
{
    uint64_t timestamp = 0;
    metrics  metric_values;
};

struct sdk_pmc_perfetto_device_data
{
    std::unique_ptr<std::vector<sdk_pmc_perfetto_sample>> samples;
    // track_index per counter name (counter_name -> track index in
    // perfetto_counter_track)
    std::map<std::string, size_t> counter_tracks;
};

inline std::map<size_t, sdk_pmc_perfetto_device_data>&
get_perfetto_data()
{
    static std::map<size_t, sdk_pmc_perfetto_device_data> data;
    return data;
}

}  // namespace detail

/**
 * @brief Output policy for writing SDK PMC samples to Perfetto traces.
 *
 * Buffers samples during collection and flushes them as counter tracks
 * during post_process(). Counter tracks are created dynamically based on
 * the counter names discovered at runtime.
 */
struct perfetto_policy
{
    using counter_track = perfetto_counter_track<metrics>;

    /**
     * @brief Initialize Perfetto storage for devices.
     */
    template <typename DeviceEntryVector>
    static void init_storage(const DeviceEntryVector& device_entries)
    {
        for(const auto& entry : device_entries)
        {
            auto idx                         = entry.device->get_index();
            detail::get_perfetto_data()[idx] = {
                std::make_unique<std::vector<detail::sdk_pmc_perfetto_sample>>(), {}
            };
        }
    }

    /**
     * @brief Set up Perfetto counter tracks for a device.
     *
     * Since SDK PMC counter names are dynamic, tracks are created lazily
     * during store_sample when new counter names are first seen.
     */
    static void setup_counter_tracks(size_t /*device_index*/,
                                     const enabled_metrics& /*enabled*/)
    {}

    /**
     * @brief Buffer a PMC sample for later Perfetto serialization.
     */
    static void store_sample(size_t device_index, const metrics& metric_values,
                             uint64_t timestamp)
    {
        auto it = detail::get_perfetto_data().find(device_index);
        if(it == detail::get_perfetto_data().end())
        {
            return;
        }

        // Create tracks for any new counter names
        for(const auto& cv : metric_values.counters)
        {
            if(it->second.counter_tracks.find(cv.name) == it->second.counter_tracks.end())
            {
                auto track_name =
                    fmt::format("GPU SDK PMC [{}] {} (S)", device_index, cv.name);
                auto track_id = counter_track::emplace(device_index, track_name, "count");
                it->second.counter_tracks[cv.name] = track_id;
                LOG_DEBUG("Created Perfetto counter track: {}", track_name);
            }
        }

        it->second.samples->emplace_back(
            detail::sdk_pmc_perfetto_sample{ timestamp, metric_values });
    }

    /**
     * @brief Post-process buffered samples and write to Perfetto trace.
     */
    static void post_process(const enabled_metrics& /*enabled*/)
    {
        const auto& thread_info = thread_info::get(0, InternalTID);
        if(!thread_info)
        {
            return;
        }

        for(const auto& [device_index, data] : detail::get_perfetto_data())
        {
            if(!data.samples)
            {
                continue;
            }

            LOG_DEBUG("Post-processing {} samples for device {}", data.samples->size(),
                      device_index);

            for(const auto& sample : *data.samples)
            {
                if(!thread_info->is_valid_time(sample.timestamp))
                {
                    continue;
                }

                for(const auto& cv : sample.metric_values.counters)
                {
                    auto track_it = data.counter_tracks.find(cv.name);
                    if(track_it == data.counter_tracks.end())
                    {
                        continue;
                    }

                    TRACE_COUNTER("rocm_counter_collection",
                                  counter_track::at(device_index, track_it->second),
                                  sample.timestamp, cv.value);
                }
            }
        }
    }
};

}  // namespace rocprofsys::pmc::collectors::sdk_pmc
