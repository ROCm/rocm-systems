// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader.hpp"

#include <cstddef>
#include <optional>
#include <utility>

namespace profiler_hub
{

/**
 * The reads of one trace connection that the track read path needs. The object
 * owns its connection: destroying it gives the connection back to its source.
 */
class track_row_reader
{
public:
    track_row_reader()          = default;
    virtual ~track_row_reader() = default;

    track_row_reader(const track_row_reader&)            = delete;
    track_row_reader& operator=(const track_row_reader&) = delete;
    track_row_reader(track_row_reader&&)                 = delete;
    track_row_reader& operator=(track_row_reader&&)      = delete;

    [[nodiscard]] virtual reader_types::timeline_event_list_t events_for_track(
        const reader_types::track_info_ptr_t& track,
        const reader_types::event_filter_t&   filter) = 0;

    [[nodiscard]] virtual reader_types::counter_timeline_event_list_t
    counter_events_for_track(const reader_types::track_info_ptr_t& track,
                             const reader_types::event_filter_t&   filter) = 0;

    [[nodiscard]] virtual std::optional<std::pair<size_t, size_t>> event_id_span(
        reader_types::event_type_t type) = 0;

    virtual void visit_events_in_id_range(const reader_types::track_info_ptr_t& track,
                                          reader_types::event_type_t            type,
                                          size_t                                id_begin,
                                          size_t                                id_end,
                                          reader_t::event_visitor_t             visitor,
                                          void* context) = 0;
};

}  // namespace profiler_hub
