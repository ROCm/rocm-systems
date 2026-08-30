// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/writer.hpp"
#include "profiler-hub/writer_types.hpp"

#include "profiler-hub/storage.hpp"

#include "writers/writer_context.hpp"
#include "writers/writer_policy_traits.hpp"

#include "writers/schema_v3/writer_policy.hpp"

#include <memory>

namespace profiler_hub
{

template <typename Policy>
class writer_impl_core
{
    static_assert(is_valid_writer_policy_v<Policy>,
                  "Policy must satisfy writer policy requirements: "
                  "provide all required type aliases and writer types "
                  "must inherit from their CRTP interfaces");

    using stmts_t      = typename Policy::insert_statements_t;
    using common_ops_t = typename Policy::common_ops_t;

public:
    explicit writer_impl_core(std::shared_ptr<writer_context> ctx);

    void register_node_info(const writer_types::node_info_t& node_info)
    {
        m_info_writer->register_node_info(node_info);
    }

    void register_process_info(const writer_types::process_info_t& process_info)
    {
        m_info_writer->register_process_info(process_info);
    }

    void register_agent_info(const writer_types::agent_info_t& agent_info)
    {
        m_info_writer->register_agent_info(agent_info);
    }

    void register_pmc_info(const writer_types::pmc_info_t& pmc_info)
    {
        m_info_writer->register_pmc_info(pmc_info);
    }

    void register_thread_info(const writer_types::thread_info_t& thread_info)
    {
        m_info_writer->register_thread_info(thread_info);
    }

    void register_stream_info(const writer_types::stream_info_t& stream_info)
    {
        m_info_writer->register_stream_info(stream_info);
    }

    void register_queue_info(const writer_types::queue_info_t& queue_info)
    {
        m_info_writer->register_queue_info(queue_info);
    }

    void register_code_object_info(
        const writer_types::code_object_info_t& code_object_info)
    {
        m_info_writer->register_code_object_info(code_object_info);
    }

    void register_kernel_symbol_info(
        const writer_types::kernel_symbol_info_t& kernel_symbol_info)
    {
        m_info_writer->register_kernel_symbol_info(kernel_symbol_info);
    }

    void register_track_info(const writer_types::track_info_t& track)
    {
        m_info_writer->register_track_info(track);
    }

    void register_string(std::string_view str) { m_info_writer->register_string(str); }

    void insert_region_data(const writer_types::region_data_t&       region_data,
                            const writer_types::trace_environment_t& trace_environment)
    {
        m_region_writer->insert(region_data, trace_environment);
    }

    void insert_pmc_event_data(const writer_types::pmc_event_data_t&     pmc_event_data,
                               const writer_types::pmc_info_unique_id_t& pmc_unique_id)
    {
        m_pmc_event_writer->insert(pmc_event_data, pmc_unique_id);
    }

    void insert_kernel_dispatch_data(
        const writer_types::kernel_dispatch_data_t& kernel_dispatch_data,
        const writer_types::trace_environment_t&    trace_environment)
    {
        m_kernel_dispatch_writer->insert(kernel_dispatch_data, trace_environment);
    }

    void insert_memory_copy_data(
        const writer_types::memory_copy_data_t&  memory_copy_data,
        const writer_types::trace_environment_t& trace_environment)
    {
        m_memory_copy_writer->insert(memory_copy_data, trace_environment);
    }

    void insert_memory_alloc_data(
        const writer_types::memory_alloc_data_t& memory_alloc_data,
        const writer_types::trace_environment_t& trace_environment)
    {
        m_memory_alloc_writer->insert(memory_alloc_data, trace_environment);
    }

    void flush_in_memory_data_to_disk() { m_ctx->backend->flush(); }

private:
    std::shared_ptr<writer_context>                            m_ctx;
    std::shared_ptr<stmts_t>                                   m_stmts;
    std::shared_ptr<common_ops_t>                              m_common_ops;
    std::unique_ptr<typename Policy::info_writer_t>            m_info_writer;
    std::unique_ptr<typename Policy::kernel_dispatch_writer_t> m_kernel_dispatch_writer;
    std::unique_ptr<typename Policy::memory_copy_writer_t>     m_memory_copy_writer;
    std::unique_ptr<typename Policy::memory_alloc_writer_t>    m_memory_alloc_writer;
    std::unique_ptr<typename Policy::region_writer_t>          m_region_writer;
    std::unique_ptr<typename Policy::pmc_event_writer_t>       m_pmc_event_writer;
};

using active_policy_t = writer_policy_v3;

struct writer_t::impl : writer_impl_core<active_policy_t>
{
    explicit impl(std::unique_ptr<profiler_hub::storage_t> storage);

private:
    static std::shared_ptr<writer_context> create_writer_context(
        const std::unique_ptr<storage_t>& storage);

    std::unique_ptr<profiler_hub::storage_t> m_storage;
};

}  // namespace profiler_hub
