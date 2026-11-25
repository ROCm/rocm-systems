// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "pooled_connection_source.hpp"

#include <utility>

namespace profiler_hub
{

namespace
{

class pooled_track_row_reader final : public track_row_reader
{
public:
    explicit pooled_track_row_reader(common::connection_pool::lease lease)
    : m_lease{ std::move(lease) }
    {}

    reader_types::timeline_event_list_t events_for_track(
        const reader_types::track_info_ptr_t& track,
        const reader_types::event_filter_t&   filter) override
    {
        return m_lease->reader().get_events_for_track(track, filter);
    }

    reader_types::counter_timeline_event_list_t counter_events_for_track(
        const reader_types::track_info_ptr_t& track,
        const reader_types::event_filter_t&   filter) override
    {
        return m_lease->reader().get_counter_events_for_track(track, filter);
    }

    std::optional<std::pair<size_t, size_t>> event_id_span(
        reader_types::event_type_t type) override
    {
        return m_lease->reader().get_event_id_span(type);
    }

    void visit_events_in_id_range(const reader_types::track_info_ptr_t& track,
                                  reader_types::event_type_t            type,
                                  size_t                                id_begin,
                                  size_t                                id_end,
                                  reader_t::event_visitor_t             visitor,
                                  void*                                 context) override
    {
        m_lease->reader().visit_track_events_in_id_range(
            track, type, id_begin, id_end, visitor, context);
    }

private:
    common::connection_pool::lease m_lease;
};

}  // namespace

pooled_connection_source::pooled_connection_source(common::connection_pool& pool) noexcept
: m_pool{ pool }
{}

std::unique_ptr<track_row_reader>
pooled_connection_source::acquire()
{
    return std::make_unique<pooled_track_row_reader>(m_pool.acquire());
}

std::unique_ptr<track_row_reader>
pooled_connection_source::try_acquire()
{
    auto lease = m_pool.try_acquire();
    if(!lease.has_value()) return nullptr;
    return std::make_unique<pooled_track_row_reader>(std::move(*lease));
}

}  // namespace profiler_hub
