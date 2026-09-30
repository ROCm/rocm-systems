// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/writer.hpp"
#include "profiler-hub/storage.hpp"
#include "profiler-hub/writer_types.hpp"

#include "writer_impl.hpp"

#include <memory>
#include <utility>

namespace profiler_hub
{

writer_t::writer_t(std::unique_ptr<profiler_hub::storage_t> storage)
: m_impl(std::make_unique<impl>(std::move(storage)))
{}

writer_t::~writer_t() = default;

void
writer_t::register_node_info(const writer_types::node_info_t& node_info)
{
    m_impl->register_node_info(node_info);
}

void
writer_t::register_process_info(const writer_types::process_info_t& process_info)
{
    m_impl->register_process_info(process_info);
}

void
writer_t::register_agent_info(const writer_types::agent_info_t& agent)
{
    m_impl->register_agent_info(agent);
}

void
writer_t::register_pmc_info(const writer_types::pmc_info_t& pmc_info)
{
    m_impl->register_pmc_info(pmc_info);
}

void
writer_t::register_thread_info(const writer_types::thread_info_t& thread_info)
{
    m_impl->register_thread_info(thread_info);
}

void
writer_t::register_stream_info(const writer_types::stream_info_t& stream_info)
{
    m_impl->register_stream_info(stream_info);
}

void
writer_t::register_queue_info(const writer_types::queue_info_t& queue_info)
{
    m_impl->register_queue_info(queue_info);
}

void
writer_t::register_code_object_info(const writer_types::code_object_info_t& code_object)
{
    m_impl->register_code_object_info(code_object);
}

void
writer_t::register_kernel_symbol_info(
    const writer_types::kernel_symbol_info_t& kernel_symbol)
{
    m_impl->register_kernel_symbol_info(kernel_symbol);
}

void
writer_t::register_track_info(const writer_types::track_info_t& track)
{
    m_impl->register_track_info(track);
}

void
writer_t::register_string(std::string_view str)
{
    m_impl->register_string(str);
}

void
writer_t::insert_region_data(const writer_types::region_data_t&       region_data,
                             const writer_types::trace_environment_t& trace_environment)
{
    m_impl->insert_region_data(region_data, trace_environment);
}

void
writer_t::insert_pmc_event_data(const writer_types::pmc_event_data_t&     pmc_event_data,
                                const writer_types::pmc_info_unique_id_t& pmc_unique_id)
{
    m_impl->insert_pmc_event_data(pmc_event_data, pmc_unique_id);
}

void
writer_t::insert_kernel_dispatch_data(
    const writer_types::kernel_dispatch_data_t& kernel_dispatch_data,
    const writer_types::trace_environment_t&    trace_environment)
{
    m_impl->insert_kernel_dispatch_data(kernel_dispatch_data, trace_environment);
}

void
writer_t::insert_memory_copy_data(
    const writer_types::memory_copy_data_t&  memory_copy_data,
    const writer_types::trace_environment_t& trace_environment)
{
    m_impl->insert_memory_copy_data(memory_copy_data, trace_environment);
}

void
writer_t::insert_memory_alloc_data(
    const writer_types::memory_alloc_data_t& memory_alloc_data,
    const writer_types::trace_environment_t& trace_environment)
{
    m_impl->insert_memory_alloc_data(memory_alloc_data, trace_environment);
}

void
writer_t::flush_in_memory_data_to_disk()
{
    m_impl->flush_in_memory_data_to_disk();
}

}  // namespace profiler_hub
