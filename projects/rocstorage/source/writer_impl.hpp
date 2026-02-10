// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "rocstorage/writer.hpp"
#include "rocstorage/writer_types.hpp"

#include "data_storage/backends/sqlite_backend.hpp"
#include "data_storage/insert_statements.hpp"
#include "entity_registry.hpp"
#include "insert_validator.hpp"
#include "primary_key_providers.hpp"
#include "rocstorage/storage.hpp"

#include <memory>
#include <string>

namespace rocstorage
{

struct writer_t::impl
{
public:
    explicit impl(std::unique_ptr<rocstorage::storage_t> storage);

    void register_node_info(const writer_types::node_info_t& node_info);
    void register_process_info(const writer_types::process_info_t& process_info);
    void register_agent_info(const writer_types::agent_info_t& agent_info);
    void register_pmc_info(const writer_types::pmc_info_t& pmc_info);
    void register_thread_info(const writer_types::thread_info_t& thread_info);
    void register_stream_info(const writer_types::stream_info_t& stream_info);
    void register_queue_info(const writer_types::queue_info_t& queue_info);
    void register_code_object_info(
        const writer_types::code_object_info_t& code_object_info);
    void register_kernel_symbol_info(
        const writer_types::kernel_symbol_info_t& kernel_symbol_info);
    void register_track_info(const writer_types::track_info_t& track);
    void register_string(const char* str);

    void insert_region_data(const writer_types::region_data_t&       region_data,
                            const writer_types::trace_environment_t& trace_environment);
    void insert_pmc_event_data(const writer_types::pmc_event_data_t&     pmc_event_data,
                               const writer_types::pmc_info_unique_id_t& pmc_unique_id);
    void insert_kernel_dispatch_data(
        const writer_types::kernel_dispatch_data_t& kernel_dispatch_data,
        const writer_types::trace_environment_t&    trace_environment);
    void insert_memory_copy_data(
        const writer_types::memory_copy_data_t&  memory_copy_data,
        const writer_types::trace_environment_t& trace_environment);
    void insert_memory_alloc_data(
        const writer_types::memory_alloc_data_t& memory_alloc_data,
        const writer_types::trace_environment_t& trace_environment);

    void flush_in_memory_data_to_disk();

private:
    primary_key_t insert_event(const writer_types::event_data_t& event_data);
    void          insert_sample(const writer_types::sample_data_t& sample_data,
                                const primary_key_t&               event_primary_key);
    void insert_arg(const writer_types::arg_data_t& arg_data, primary_key_t event_id);

    std::unique_ptr<rocstorage::storage_t> m_storage;

    std::shared_ptr<data_storage::sqlite_backend>               m_backend;
    std::string                                                 m_uuid;
    std::unique_ptr<data_storage::schema_v3::insert_statements> m_insert_statements;
    std::shared_ptr<entity_registry>                            m_entity_registry;
    std::shared_ptr<primary_key_providers>                      m_key_providers;
    std::shared_ptr<insert_validator>                           m_validator;
};

}  // namespace rocstorage
