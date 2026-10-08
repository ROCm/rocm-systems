// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "policies/profiler-hub/writer.hpp"

#include <profiler-hub/shared_types.hpp>
#include <profiler-hub/storage.hpp>
#include <profiler-hub/writer.hpp>
#include <profiler-hub/writer_types.hpp>

#include "logger/debug.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace rocprofsys::trace_cache
{

/**
 * Production `policies::profiler_hub::writer_policy`: owns a `profiler_hub::writer_t`
 * backed by a freshly created database at `database_path`. An existing file at that
 * path is removed first so repeated init/finalize cycles start from a clean state.
 */
class profiler_hub_writer
{
public:
    using node_id_t              = ::profiler_hub::writer_types::node_id_t;
    using process_id_t           = ::profiler_hub::writer_types::process_id_t;
    using thread_id_t            = ::profiler_hub::writer_types::thread_id_t;
    using code_object_id_t       = ::profiler_hub::writer_types::code_object_id_t;
    using kernel_symbol_id_t     = ::profiler_hub::writer_types::kernel_symbol_id_t;
    using stream_id_t            = ::profiler_hub::writer_types::stream_id_t;
    using queue_id_t             = ::profiler_hub::writer_types::queue_id_t;
    using timestamp_ns_t         = ::profiler_hub::writer_types::timestamp_ns_t;
    using pmc_description_name_t = ::profiler_hub::writer_types::pmc_description_name_t;
    using track_name_t           = ::profiler_hub::writer_types::track_name_t;
    using agent_unique_id_t      = ::profiler_hub::writer_types::agent_unique_id_t;
    using trace_environment_t    = ::profiler_hub::writer_types::trace_environment_t;
    using node_info_t            = ::profiler_hub::writer_types::node_info_t;
    using process_info_t         = ::profiler_hub::writer_types::process_info_t;
    using agent_info_t           = ::profiler_hub::writer_types::agent_info_t;
    using pmc_info_unique_id_t   = ::profiler_hub::writer_types::pmc_info_unique_id_t;
    using pmc_info_t             = ::profiler_hub::writer_types::pmc_info_t;
    using thread_info_t          = ::profiler_hub::writer_types::thread_info_t;
    using stream_info_t          = ::profiler_hub::writer_types::stream_info_t;
    using queue_info_t           = ::profiler_hub::writer_types::queue_info_t;
    using code_object_info_t     = ::profiler_hub::writer_types::code_object_info_t;
    using kernel_symbol_info_t   = ::profiler_hub::writer_types::kernel_symbol_info_t;
    using track_info_t           = ::profiler_hub::writer_types::track_info_t;
    using arg_data_t             = ::profiler_hub::writer_types::arg_data_t;
    using event_data_t           = ::profiler_hub::writer_types::event_data_t;
    using region_data_t          = ::profiler_hub::writer_types::region_data_t;
    using sample_data_t          = ::profiler_hub::writer_types::sample_data_t;
    using pmc_event_data_t       = ::profiler_hub::writer_types::pmc_event_data_t;
    using kernel_dispatch_data_t = ::profiler_hub::writer_types::kernel_dispatch_data_t;
    using memory_copy_data_t     = ::profiler_hub::writer_types::memory_copy_data_t;
    using memory_alloc_data_t    = ::profiler_hub::writer_types::memory_alloc_data_t;
    using call_stack_t           = ::profiler_hub::shared_types::call_stack_t;
    using source_context_list_t  = ::profiler_hub::shared_types::source_context_list_t;

    profiler_hub_writer(const std::string& database_path, const std::string& uuid);

    void register_node_info(const node_info_t& value)
    {
        m_writer.register_node_info(value);
    }
    void register_process_info(const process_info_t& value)
    {
        m_writer.register_process_info(value);
    }
    void register_agent_info(const agent_info_t& value)
    {
        m_writer.register_agent_info(value);
    }
    void register_pmc_info(const pmc_info_t& value) { m_writer.register_pmc_info(value); }
    void register_thread_info(const thread_info_t& value)
    {
        m_writer.register_thread_info(value);
    }
    void register_stream_info(const stream_info_t& value)
    {
        m_writer.register_stream_info(value);
    }
    void register_queue_info(const queue_info_t& value)
    {
        m_writer.register_queue_info(value);
    }
    void register_code_object_info(const code_object_info_t& value)
    {
        m_writer.register_code_object_info(value);
    }
    void register_kernel_symbol_info(const kernel_symbol_info_t& value)
    {
        m_writer.register_kernel_symbol_info(value);
    }
    void register_track_info(const track_info_t& value)
    {
        m_writer.register_track_info(value);
    }
    void register_string(std::string_view value) { m_writer.register_string(value); }

    void insert_region_data(const region_data_t& data, const trace_environment_t& extra)
    {
        m_writer.insert_region_data(data, extra);
    }
    void insert_pmc_event_data(const pmc_event_data_t&     data,
                               const pmc_info_unique_id_t& extra)
    {
        m_writer.insert_pmc_event_data(data, extra);
    }
    void insert_kernel_dispatch_data(const kernel_dispatch_data_t& data,
                                     const trace_environment_t&    extra)
    {
        m_writer.insert_kernel_dispatch_data(data, extra);
    }
    void insert_memory_copy_data(const memory_copy_data_t&  data,
                                 const trace_environment_t& extra)
    {
        m_writer.insert_memory_copy_data(data, extra);
    }
    void insert_memory_alloc_data(const memory_alloc_data_t& data,
                                  const trace_environment_t& extra)
    {
        m_writer.insert_memory_alloc_data(data, extra);
    }

    void flush_in_memory_data_to_disk() { m_writer.flush_in_memory_data_to_disk(); }

private:
    ::profiler_hub::writer_t m_writer;
};

inline profiler_hub_writer::profiler_hub_writer(const std::string& database_path,
                                                const std::string& uuid)
: m_writer([&] {
    // A leftover file from a previous session would make the writer fail with
    // "Database already initialized!" (e.g. ROCPROFSYS_USE_PID=OFF re-init cycles).
    if(std::filesystem::exists(database_path))
    {
        LOG_WARNING("rocpd output file already exists and will be overwritten: {}. "
                    "Previous profiling data in that file will be lost. "
                    "Set ROCPROFSYS_USE_PID=ON to give each session a unique path.",
                    database_path);
        std::filesystem::remove(database_path);
    }
    return std::make_unique<::profiler_hub::storage_t>(database_path, uuid);
}())
{}

static_assert(policies::profiler_hub::writer_policy<profiler_hub_writer>);

}  // namespace rocprofsys::trace_cache
