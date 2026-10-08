// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "mock_profiler_hub_types.hpp"

#include <gmock/gmock.h>

#include <memory>
#include <string>
#include <string_view>

namespace rocprofsys::trace_cache::test
{

/// Non-copyable gmock object; instantiate as `StrictMock<gmock_profiler_hub_writer>`.
struct gmock_profiler_hub_writer
{
    MOCK_METHOD(void, register_node_info, (const writer_types::node_info_t&) );
    MOCK_METHOD(void, register_process_info, (const writer_types::process_info_t&) );
    MOCK_METHOD(void, register_agent_info, (const writer_types::agent_info_t&) );
    MOCK_METHOD(void, register_pmc_info, (const writer_types::pmc_info_t&) );
    MOCK_METHOD(void, register_thread_info, (const writer_types::thread_info_t&) );
    MOCK_METHOD(void, register_stream_info, (const writer_types::stream_info_t&) );
    MOCK_METHOD(void, register_queue_info, (const writer_types::queue_info_t&) );
    MOCK_METHOD(void, register_code_object_info,
                (const writer_types::code_object_info_t&) );
    MOCK_METHOD(void, register_kernel_symbol_info,
                (const writer_types::kernel_symbol_info_t&) );
    MOCK_METHOD(void, register_track_info, (const writer_types::track_info_t&) );
    MOCK_METHOD(void, register_string, (std::string_view));

    MOCK_METHOD(void, insert_region_data,
                (const writer_types::region_data_t&,
                 const writer_types::trace_environment_t&) );
    MOCK_METHOD(void, insert_pmc_event_data,
                (const writer_types::pmc_event_data_t&,
                 const writer_types::pmc_info_unique_id_t&) );
    MOCK_METHOD(void, insert_kernel_dispatch_data,
                (const writer_types::kernel_dispatch_data_t&,
                 const writer_types::trace_environment_t&) );
    MOCK_METHOD(void, insert_memory_copy_data,
                (const writer_types::memory_copy_data_t&,
                 const writer_types::trace_environment_t&) );
    MOCK_METHOD(void, insert_memory_alloc_data,
                (const writer_types::memory_alloc_data_t&,
                 const writer_types::trace_environment_t&) );

    MOCK_METHOD(void, flush_in_memory_data_to_disk, ());
};

/// Set by the test fixture before constructing the processor under test.
inline std::unique_ptr<gmock_profiler_hub_writer> g_mock_profiler_hub_writer;

/**
 * Copyable-by-construction policy wrapper satisfying
 * `policies::profiler_hub::writer_policy`. The processor owns this by value; every call
 * is forwarded to `g_mock_profiler_hub_writer`, which the test sets up with
 * `StrictMock`. The constructor arguments (database path, uuid) are ignored.
 *
 * Matchers must evaluate inside the call: `writer_types` fields are non-owning views.
 */
struct mock_profiler_hub_writer
{
    using node_id_t              = writer_types::node_id_t;
    using process_id_t           = writer_types::process_id_t;
    using thread_id_t            = writer_types::thread_id_t;
    using code_object_id_t       = writer_types::code_object_id_t;
    using kernel_symbol_id_t     = writer_types::kernel_symbol_id_t;
    using stream_id_t            = writer_types::stream_id_t;
    using queue_id_t             = writer_types::queue_id_t;
    using timestamp_ns_t         = writer_types::timestamp_ns_t;
    using pmc_description_name_t = writer_types::pmc_description_name_t;
    using track_name_t           = writer_types::track_name_t;
    using agent_unique_id_t      = writer_types::agent_unique_id_t;
    using trace_environment_t    = writer_types::trace_environment_t;
    using node_info_t            = writer_types::node_info_t;
    using process_info_t         = writer_types::process_info_t;
    using agent_info_t           = writer_types::agent_info_t;
    using pmc_info_unique_id_t   = writer_types::pmc_info_unique_id_t;
    using pmc_info_t             = writer_types::pmc_info_t;
    using thread_info_t          = writer_types::thread_info_t;
    using stream_info_t          = writer_types::stream_info_t;
    using queue_info_t           = writer_types::queue_info_t;
    using code_object_info_t     = writer_types::code_object_info_t;
    using kernel_symbol_info_t   = writer_types::kernel_symbol_info_t;
    using track_info_t           = writer_types::track_info_t;
    using arg_data_t             = writer_types::arg_data_t;
    using event_data_t           = writer_types::event_data_t;
    using region_data_t          = writer_types::region_data_t;
    using sample_data_t          = writer_types::sample_data_t;
    using pmc_event_data_t       = writer_types::pmc_event_data_t;
    using kernel_dispatch_data_t = writer_types::kernel_dispatch_data_t;
    using memory_copy_data_t     = writer_types::memory_copy_data_t;
    using memory_alloc_data_t    = writer_types::memory_alloc_data_t;
    using call_stack_t           = shared_types::call_stack_t;
    using source_context_list_t  = shared_types::source_context_list_t;

    mock_profiler_hub_writer(const std::string& /*database_path*/,
                             const std::string& /*uuid*/)
    {}

    void register_node_info(const writer_types::node_info_t& v)
    {
        g_mock_profiler_hub_writer->register_node_info(v);
    }
    void register_process_info(const writer_types::process_info_t& v)
    {
        g_mock_profiler_hub_writer->register_process_info(v);
    }
    void register_agent_info(const writer_types::agent_info_t& v)
    {
        g_mock_profiler_hub_writer->register_agent_info(v);
    }
    void register_pmc_info(const writer_types::pmc_info_t& v)
    {
        g_mock_profiler_hub_writer->register_pmc_info(v);
    }
    void register_thread_info(const writer_types::thread_info_t& v)
    {
        g_mock_profiler_hub_writer->register_thread_info(v);
    }
    void register_stream_info(const writer_types::stream_info_t& v)
    {
        g_mock_profiler_hub_writer->register_stream_info(v);
    }
    void register_queue_info(const writer_types::queue_info_t& v)
    {
        g_mock_profiler_hub_writer->register_queue_info(v);
    }
    void register_code_object_info(const writer_types::code_object_info_t& v)
    {
        g_mock_profiler_hub_writer->register_code_object_info(v);
    }
    void register_kernel_symbol_info(const writer_types::kernel_symbol_info_t& v)
    {
        g_mock_profiler_hub_writer->register_kernel_symbol_info(v);
    }
    void register_track_info(const writer_types::track_info_t& v)
    {
        g_mock_profiler_hub_writer->register_track_info(v);
    }
    void register_string(std::string_view v)
    {
        g_mock_profiler_hub_writer->register_string(v);
    }

    void insert_region_data(const writer_types::region_data_t&       data,
                            const writer_types::trace_environment_t& env)
    {
        g_mock_profiler_hub_writer->insert_region_data(data, env);
    }
    void insert_pmc_event_data(const writer_types::pmc_event_data_t&     data,
                               const writer_types::pmc_info_unique_id_t& uid)
    {
        g_mock_profiler_hub_writer->insert_pmc_event_data(data, uid);
    }
    void insert_kernel_dispatch_data(const writer_types::kernel_dispatch_data_t& data,
                                     const writer_types::trace_environment_t&    env)
    {
        g_mock_profiler_hub_writer->insert_kernel_dispatch_data(data, env);
    }
    void insert_memory_copy_data(const writer_types::memory_copy_data_t&  data,
                                 const writer_types::trace_environment_t& env)
    {
        g_mock_profiler_hub_writer->insert_memory_copy_data(data, env);
    }
    void insert_memory_alloc_data(const writer_types::memory_alloc_data_t& data,
                                  const writer_types::trace_environment_t& env)
    {
        g_mock_profiler_hub_writer->insert_memory_alloc_data(data, env);
    }

    void flush_in_memory_data_to_disk()
    {
        g_mock_profiler_hub_writer->flush_in_memory_data_to_disk();
    }
};

}  // namespace rocprofsys::trace_cache::test
