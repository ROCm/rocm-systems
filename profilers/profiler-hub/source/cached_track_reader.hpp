// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/thread_pool.hpp"
#include "connection_source.hpp"
#include "profiler-hub/c/profiler_hub_types.h"
#include "track_read_options.hpp"

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace profiler_hub
{

/**
 * Serves the events and samples of a track as C API lists. A request for a whole track
 * (both bounds 0) is read once, sorted, kept and answered with the same storage on every
 * later call; a request with a time window gets storage of its own. Everything handed out
 * stays valid until the reader is destroyed. Events are only served for tracks that are
 * not PMC tracks and samples only for PMC tracks; any other request yields an empty list.
 *
 * All members are safe to call from several threads. Large thread tracks are read in id
 * ranges on several connections, using @p workers for the helpers.
 */
class cached_track_reader
{
public:
    cached_track_reader(connection_source&   source,
                        common::thread_pool& workers,
                        track_read_options   options);

    [[nodiscard]] ph_event_list_t events(const reader_types::track_info_ptr_t& track,
                                         uint64_t                              start_ts,
                                         uint64_t                              end_ts);

    [[nodiscard]] ph_sample_list_t samples(const reader_types::track_info_ptr_t& track,
                                           uint64_t                              start_ts,
                                           uint64_t                              end_ts);

private:
    template <typename Item>
    struct cache_entry
    {
        std::once_flag    once;
        std::vector<Item> items;
    };

    template <typename Item>
    using cache_t = std::unordered_map<size_t, cache_entry<Item>>;

    template <typename Item, typename Build>
    std::vector<Item>& cached(cache_t<Item>&                        cache,
                              const reader_types::track_info_ptr_t& track,
                              Build&&                               build);

    [[nodiscard]] std::vector<ph_event_t> build_sorted_events(
        track_row_reader&                     reader,
        const reader_types::track_info_ptr_t& track);

    [[nodiscard]] std::vector<ph_event_t> read_thread_track_in_parts(
        track_row_reader&                     reader,
        const reader_types::track_info_ptr_t& track);

    connection_source&   m_source;
    common::thread_pool& m_workers;
    track_read_options   m_options;

    std::mutex           m_cache_mutex;
    cache_t<ph_event_t>  m_events_cache;
    cache_t<ph_sample_t> m_samples_cache;

    std::mutex                           m_windowed_mutex;
    std::deque<std::vector<ph_event_t>>  m_windowed_events;
    std::deque<std::vector<ph_sample_t>> m_windowed_samples;
};

}  // namespace profiler_hub
