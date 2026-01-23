// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <rocstorage/writer.hpp>

#include "writer_impl.hpp"

namespace rocstorage
{

writer::writer(std::shared_ptr<data_storage::database> database, std::string uuid)
: m_impl(std::make_unique<impl>(std::move(database), std::move(uuid)))
{}

writer::~writer() = default;

void
writer::register_node_info(const writer_api::node_info_t& node_info)
{
    m_impl->register_node_info(node_info);
}

void
writer::register_process_info(const writer_api::process_info_t& process_info)
{
    m_impl->register_process_info(process_info);
}

void
writer::register_agent_info(const writer_api::agent_info_t& agent)
{
    m_impl->register_agent_info(agent);
}

void
writer::register_pmc_info(const writer_api::pmc_info_t& pmc_info)
{
    m_impl->register_pmc_info(pmc_info);
}

void
writer::register_thread_info(const writer_api::thread_info_t& thread_info)
{
    m_impl->register_thread_info(thread_info);
}

void
writer::register_stream_info(const writer_api::stream_info_t& stream_info)
{
    m_impl->register_stream_info(stream_info);
}

void
writer::register_queue_info(const writer_api::queue_info_t& queue_info)
{
    m_impl->register_queue_info(queue_info);
}

void
writer::register_code_object_info(const writer_api::code_object_info_t& code_object)
{
    m_impl->register_code_object_info(code_object);
}

void
writer::register_kernel_symbol_info(const writer_api::kernel_symbol_info_t& kernel_symbol)
{
    m_impl->register_kernel_symbol_info(kernel_symbol);
}

void
writer::register_track_info(const writer_api::track_info_t& track)
{
    m_impl->register_track_info(track);
}

void
writer::register_string(const char* str)
{
    m_impl->register_string(str);
}

void
writer::insert_region_data(const writer_api::region_data_t&       region_data,
                           const writer_api::trace_environment_t& trace_environment)
{
    m_impl->insert_region_data(region_data, trace_environment);
}

void
writer::insert_pmc_event_data(const writer_api::pmc_event_data_t&     pmc_event_data,
                              const writer_api::pmc_info_unique_id_t& pmc_unique_id)
{
    m_impl->insert_pmc_event_data(pmc_event_data, pmc_unique_id);
}

void
writer::insert_kernel_dispatch_data(
    const writer_api::kernel_dispatch_data_t& kernel_dispatch_data,
    const writer_api::trace_environment_t&    trace_environment)
{
    m_impl->insert_kernel_dispatch_data(kernel_dispatch_data, trace_environment);
}

void
writer::insert_memory_copy_data(const writer_api::memory_copy_data_t&  memory_copy_data,
                                const writer_api::trace_environment_t& trace_environment)
{
    m_impl->insert_memory_copy_data(memory_copy_data, trace_environment);
}

void
writer::insert_memory_alloc_data(const writer_api::memory_alloc_data_t& memory_alloc_data,
                                 const writer_api::trace_environment_t& trace_environment)
{
    m_impl->insert_memory_alloc_data(memory_alloc_data, trace_environment);
}

void
writer::flush_in_memory_data_to_disk()
{
    m_impl->flush_in_memory_data_to_disk();
}

}  // namespace rocstorage
