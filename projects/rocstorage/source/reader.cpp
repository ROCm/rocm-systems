// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "rocstorage/reader.hpp"
#include "reader_impl.hpp"
#include "rocstorage/storage.hpp"

#include <memory>
#include <utility>

namespace rocstorage
{

reader_t::reader_t(std::unique_ptr<rocstorage::storage_t> storage)
: m_impl(std::make_unique<impl>(std::move(storage)))
{}

reader_t::~reader_t() = default;

reader_types::node_info_list_t
reader_t::get_all_nodes() const
{
    return m_impl->get_all_nodes();
}

reader_types::process_info_list_t
reader_t::get_all_processes() const
{
    return m_impl->get_all_processes();
}

reader_types::thread_info_list_t
reader_t::get_all_threads() const
{
    return m_impl->get_all_threads();
}

reader_types::agent_info_list_t
reader_t::get_all_agents() const
{
    return m_impl->get_all_agents();
}

reader_types::queue_info_list_t
reader_t::get_all_queues() const
{
    return m_impl->get_all_queues();
}

reader_types::stream_info_list_t
reader_t::get_all_streams() const
{
    return m_impl->get_all_streams();
}

reader_types::pmc_info_list_t
reader_t::get_all_pmc_info() const
{
    return m_impl->get_all_pmc_infos();
}

reader_types::code_object_info_list_t
reader_t::get_all_code_objects() const
{
    return m_impl->get_all_code_objects();
}

reader_types::kernel_symbol_info_list_t
reader_t::get_all_kernel_symbols() const
{
    return m_impl->get_all_kernel_symbols();
}

reader_types::track_info_list_t
reader_t::get_all_tracks() const
{
    return m_impl->get_all_tracks();
}

reader_types::timeline_event_list_t
reader_t::get_events_for_track(reader_types::track_info_ptr_t      track,
                               const reader_types::event_filter_t& filter) const
{
    (void) track;
    (void) filter;
    return {};
}

reader_types::timeline_event_list_t
reader_t::get_events(const reader_types::event_filter_t& filter) const
{
    (void) filter;
    return {};
}

size_t
reader_t::get_event_count(const reader_types::event_filter_t& filter) const
{
    (void) filter;
    return 0;
}

std::optional<reader_types::region_data_t>
reader_t::get_region_details(const reader_types::timeline_event_t& event) const
{
    (void) event;
    return std::nullopt;
}

std::optional<reader_types::kernel_dispatch_data_t>
reader_t::get_kernel_dispatch_details(const reader_types::timeline_event_t& event) const
{
    (void) event;
    return std::nullopt;
}

std::optional<reader_types::memory_copy_data_t>
reader_t::get_memory_copy_details(const reader_types::timeline_event_t& event) const
{
    (void) event;
    return std::nullopt;
}

std::optional<reader_types::memory_alloc_data_t>
reader_t::get_memory_alloc_details(const reader_types::timeline_event_t& event) const
{
    (void) event;
    return std::nullopt;
}

std::optional<reader_types::sample_data_t>
reader_t::get_sample_details(const reader_types::timeline_event_t& event) const
{
    (void) event;
    return std::nullopt;
}

std::optional<reader_types::pmc_event_data_t>
reader_t::get_pmc_event_details(const reader_types::timeline_event_t& event) const
{
    (void) event;
    return std::nullopt;
}

shared_types::call_stack_t
reader_t::get_call_stack(const reader_types::timeline_event_t& event) const
{
    (void) event;
    return {};
}

shared_types::source_context_list_t
reader_t::get_source_context(const reader_types::timeline_event_t& event) const
{
    (void) event;
    return {};
}

reader_types::arg_data_list_t
reader_t::get_arguments(const reader_types::timeline_event_t& event) const
{
    (void) event;
    return {};
}

reader_types::timeline_event_list_t
reader_t::get_correlated_events(const reader_types::timeline_event_t& event) const
{
    (void) event;
    return {};
}

reader_types::event_summary_list_t
reader_t::get_kernel_summary(const reader_types::time_window_t& window) const
{
    (void) window;
    return {};
}

reader_types::event_summary_list_t
reader_t::get_region_summary(const reader_types::time_window_t& window) const
{
    (void) window;
    return {};
}

reader_types::time_window_t
reader_t::get_data_time_range() const
{
    return {};
}

reader_types::event_counts_t
reader_t::get_event_counts(const reader_types::time_window_t& window) const
{
    (void) window;
    return {};
}

}  // namespace rocstorage
