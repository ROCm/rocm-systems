// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/thread_pool.hpp"
#include "connection_source.hpp"
#include "profiler-hub/c/profiler_hub_types.h"
#include "track_read_options.hpp"

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace profiler_hub
{

class track_file_store;

/**
 * Serves the events and samples of a track as C API lists. A request for a whole track
 * (both bounds 0) is read once, sorted, kept and answered with the same storage on every
 * later call; a request with a time window gets storage of its own, filled from that
 * whole-track read. A sample window also includes the sample before the window and the
 * sample after it, and repeats the last value at the track end when nothing follows.
 * Everything handed out stays valid until the reader is destroyed.
 * Events are only served for tracks that are not PMC tracks and samples only for PMC
 * tracks; any other request yields an empty list.
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

    /**
     * The largest ph_event_t::nesting_depth of the whole track; 0 for a null track, a PMC
     * track or a track without events. Reads and caches the whole track if needed.
     */
    [[nodiscard]] uint32_t nesting_depth(const reader_types::track_info_ptr_t& track);

    /**
     * Loads the whole track if needed and returns the nesting depth of the event
     * with @p id and @p type. Empty when the track is null, a PMC track, or does
     * not contain that event.
     */
    [[nodiscard]] std::optional<uint32_t> nesting_depth_of(
        const reader_types::track_info_ptr_t& track,
        uint64_t                              id,
        ph_event_type_t                       type);

    [[nodiscard]] ph_sample_list_t samples(const reader_types::track_info_ptr_t& track,
                                           uint64_t                              start_ts,
                                           uint64_t                              end_ts);

    /** Builds one track and returns it. Does not keep the events. @p keep_name
     *  owns each display name; its pointer is stored on the event. */
    [[nodiscard]] std::vector<ph_event_t> load_events(
        const reader_types::track_info_ptr_t&                track,
        const std::function<const char*(std::string_view)>& keep_name);

    [[nodiscard]] std::vector<ph_sample_t> load_samples(
        const reader_types::track_info_ptr_t& track);

    /** When set, nesting_depth_of reads the store instead of the event cache. */
    void set_track_store(track_file_store* store, uint32_t source_index);

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
        track_row_reader&                                    reader,
        const reader_types::track_info_ptr_t&                track,
        const std::function<const char*(std::string_view)>& keep_name);

    [[nodiscard]] std::vector<ph_event_t> read_thread_track_in_parts(
        track_row_reader&                                    reader,
        const reader_types::track_info_ptr_t&                track,
        const std::function<const char*(std::string_view)>& keep_name);

    // ph_event_t.name has to outlive the row that produced it. The deque is
    // append-only, so a c_str() taken after emplace stays valid.
    [[nodiscard]] const char* keep_name(std::string_view text);
    [[nodiscard]] std::function<const char*(std::string_view)> name_keeper();

    connection_source&   m_source;
    common::thread_pool& m_workers;
    track_read_options   m_options;

    struct event_depth_index
    {
        std::once_flag                                        once;
        std::array<std::unordered_map<uint64_t, uint32_t>, 5> by_type;
    };

    std::mutex           m_cache_mutex;
    cache_t<ph_event_t>  m_events_cache;
    cache_t<ph_sample_t> m_samples_cache;

    std::mutex                                    m_depth_mutex;
    std::unordered_map<size_t, event_depth_index> m_event_depths;

    std::mutex                           m_windowed_mutex;
    std::deque<std::vector<ph_event_t>>  m_windowed_events;
    std::deque<std::vector<ph_sample_t>> m_windowed_samples;

    std::mutex              m_name_mutex;
    std::deque<std::string> m_names;

    track_file_store* m_store{ nullptr };
    uint32_t          m_store_source{ 0 };
};

}  // namespace profiler_hub
