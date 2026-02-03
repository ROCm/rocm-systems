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

data_types::node_info_list_t
reader_t::get_all_nodes() const
{
    return m_impl->get_all_nodes();
}

data_types::process_info_list_t
reader_t::get_all_processes() const
{
    return m_impl->get_process_list();
}

data_types::thread_info_list_t
reader_t::get_all_threads() const
{
    return m_impl->get_thread_list();
}

data_types::agent_info_list_t
reader_t::get_all_agents() const
{
    return m_impl->get_agent_list();
}

data_types::queue_info_list_t
reader_t::get_all_queues() const
{
    return m_impl->get_queue_list();
}

data_types::stream_info_list_t
reader_t::get_all_streams() const
{
    return m_impl->get_stream_list();
}

data_types::pmc_info_list_t
reader_t::get_all_pmc_info() const
{
    return m_impl->get_pmc_info_list();
}

data_types::code_object_info_list_t
reader_t::get_all_code_objects() const
{
    return m_impl->get_code_object_list();
}

data_types::kernel_symbol_info_list_t
reader_t::get_all_kernel_symbols() const
{
    return m_impl->get_kernel_symbol_list();
}

data_types::node_info_ptr_t
reader_t::get_node_by_id(data_types::node_id_t id) const
{
    (void) id;
    return nullptr;
}

data_types::process_info_ptr_t
reader_t::get_process_by_id(data_types::process_id_t id) const
{
    (void) id;
    return nullptr;
}

data_types::thread_info_ptr_t
reader_t::get_thread_by_id(data_types::thread_id_t id) const
{
    (void) id;
    return nullptr;
}

data_types::agent_info_ptr_t
reader_t::get_agent_by_id(const data_types::agent_unique_id_t& id) const
{
    (void) id;
    return nullptr;
}

data_types::queue_info_ptr_t
reader_t::get_queue_by_id(data_types::queue_id_t id) const
{
    (void) id;
    return nullptr;
}

data_types::stream_info_ptr_t
reader_t::get_stream_by_id(data_types::stream_id_t id) const
{
    (void) id;
    return nullptr;
}

data_types::pmc_info_ptr_t
reader_t::get_pmc_by_id(const data_types::pmc_info_unique_id_t& id) const
{
    (void) id;
    return nullptr;
}

data_types::code_object_info_ptr_t
reader_t::get_code_object_by_id(data_types::code_object_id_t id) const
{
    (void) id;
    return nullptr;
}

data_types::kernel_symbol_info_ptr_t
reader_t::get_kernel_symbol_by_id(data_types::kernel_symbol_id_t id) const
{
    (void) id;
    return nullptr;
}

data_types::track_info_list_t
reader_t::get_all_tracks() const
{
    return m_impl->get_track_list();
}

data_types::track_info_ptr_t
reader_t::get_track_by_name(data_types::track_name_t name) const
{
    (void) name;
    return nullptr;
}

data_types::timeline_event_list_t
reader_t::get_events_for_track(data_types::track_info_ptr_t      track,
                               const data_types::event_filter_t& filter) const
{
    (void) track;
    (void) filter;
    return {};
}

data_types::timeline_event_list_t
reader_t::get_events(const data_types::event_filter_t& filter) const
{
    (void) filter;
    return {};
}

size_t
reader_t::get_event_count(const data_types::event_filter_t& filter) const
{
    (void) filter;
    return 0;
}

std::optional<data_types::region_data_t>
reader_t::get_region_details(const data_types::timeline_event_t& event) const
{
    (void) event;
    return std::nullopt;
}

std::optional<data_types::kernel_dispatch_data_t>
reader_t::get_kernel_dispatch_details(const data_types::timeline_event_t& event) const
{
    (void) event;
    return std::nullopt;
}

std::optional<data_types::memory_copy_data_t>
reader_t::get_memory_copy_details(const data_types::timeline_event_t& event) const
{
    (void) event;
    return std::nullopt;
}

std::optional<data_types::memory_alloc_data_t>
reader_t::get_memory_alloc_details(const data_types::timeline_event_t& event) const
{
    (void) event;
    return std::nullopt;
}

std::optional<data_types::sample_data_t>
reader_t::get_sample_details(const data_types::timeline_event_t& event) const
{
    (void) event;
    return std::nullopt;
}

std::optional<data_types::pmc_event_data_t>
reader_t::get_pmc_event_details(const data_types::timeline_event_t& event) const
{
    (void) event;
    return std::nullopt;
}

data_types::call_stack_t
reader_t::get_call_stack(const data_types::timeline_event_t& event) const
{
    (void) event;
    return {};
}

data_types::source_context_list_t
reader_t::get_source_context(const data_types::timeline_event_t& event) const
{
    (void) event;
    return {};
}

data_types::arg_data_list_t
reader_t::get_arguments(const data_types::timeline_event_t& event) const
{
    (void) event;
    return {};
}

data_types::timeline_event_list_t
reader_t::get_correlated_events(const data_types::timeline_event_t& event) const
{
    (void) event;
    return {};
}

data_types::event_summary_list_t
reader_t::get_kernel_summary(const data_types::time_window_t& window) const
{
    (void) window;
    return {};
}

data_types::event_summary_list_t
reader_t::get_region_summary(const data_types::time_window_t& window) const
{
    (void) window;
    return {};
}

data_types::time_window_t
reader_t::get_data_time_range() const
{
    return {};
}

data_types::event_counts_t
reader_t::get_event_counts(const data_types::time_window_t& window) const
{
    (void) window;
    return {};
}

}  // namespace rocstorage
