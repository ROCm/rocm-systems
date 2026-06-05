// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "queries/insert/table_insert_query.hpp"
#include "spdlog/fmt/bundled/core.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace profiler_hub::data_storage::schema_v3
{
using integer_primary_key_t = size_t;
using integer_foreign_key_t = size_t;

template <typename Backend>
struct insert_statements
{
    explicit insert_statements(std::shared_ptr<Backend> backend, std::string uuid)
    : m_backend(std::move(backend))
    , m_uuid(std::move(uuid))
    {
        initialize_string_statement();
        initialize_node_info_statement();
        initialize_process_info_statement();
        initialize_agent_info_statement();
        initialize_pmc_info_statement();
        initialize_thread_info_statement();
        initialize_stream_info_statement();
        initialize_queue_info_statement();
        initialize_kernel_symbol_info_statement();
        initialize_code_object_info_statement();
        initialize_track_info_statement();
        initialize_event_statement();
        initialize_arg_statement();
        initialize_pmc_event_statement();
        initialize_region_statement();
        initialize_sample_statement();
        initialize_kernel_dispatch_statement();
        initialize_memory_copy_statement();
        initialize_memory_alloc_statement();
    }

    insert_statements()                                    = delete;
    insert_statements(const insert_statements&)            = delete;
    insert_statements(insert_statements&&)                 = delete;
    insert_statements& operator=(const insert_statements&) = delete;
    insert_statements& operator=(insert_statements&&)      = delete;

    template <typename... Ts>
    using statement_t = typename Backend::template prepared_insert_statement<Ts...>;

    using string_statement_func_t = statement_t<size_t, std::string_view>;

    using node_info_statement_func_t = statement_t<integer_primary_key_t,
                                                   size_t,
                                                   std::string_view,
                                                   std::optional<std::string_view>,
                                                   std::optional<std::string_view>,
                                                   std::optional<std::string_view>,
                                                   std::optional<std::string_view>,
                                                   std::optional<std::string_view>,
                                                   std::optional<std::string_view>>;

    using process_info_statement_func_t = statement_t<integer_primary_key_t,
                                                      integer_foreign_key_t,
                                                      std::optional<size_t>,
                                                      size_t,
                                                      std::optional<size_t>,
                                                      std::optional<size_t>,
                                                      std::optional<size_t>,
                                                      std::optional<size_t>,
                                                      std::optional<std::string_view>,
                                                      std::string_view,
                                                      std::string_view>;

    using thread_info_statement_func_t = statement_t<integer_primary_key_t,
                                                     integer_foreign_key_t,
                                                     std::optional<size_t>,
                                                     integer_foreign_key_t,
                                                     size_t,
                                                     std::optional<std::string_view>,
                                                     std::optional<size_t>,
                                                     std::optional<size_t>,
                                                     std::string_view>;

    // Set type to nullptr if type is not available
    using agent_info_statement_func_t = statement_t<integer_primary_key_t,
                                                    integer_foreign_key_t,
                                                    integer_foreign_key_t,
                                                    std::optional<std::string_view>,
                                                    std::optional<size_t>,
                                                    std::optional<size_t>,
                                                    size_t,
                                                    std::optional<size_t>,
                                                    std::optional<std::string_view>,
                                                    std::optional<std::string_view>,
                                                    std::optional<std::string_view>,
                                                    std::optional<std::string_view>,
                                                    std::optional<std::string_view>,
                                                    std::string_view>;

    using queue_info_statement_func_t = statement_t<integer_primary_key_t,
                                                    integer_foreign_key_t,
                                                    integer_foreign_key_t,
                                                    std::optional<std::string_view>,
                                                    std::string_view>;

    using stream_info_statement_func_t = statement_t<integer_primary_key_t,
                                                     integer_foreign_key_t,
                                                     integer_foreign_key_t,
                                                     std::optional<std::string_view>,
                                                     std::string_view>;

    using pmc_info_statement_func_t = statement_t<integer_primary_key_t,
                                                  integer_foreign_key_t,
                                                  integer_foreign_key_t,
                                                  std::optional<integer_foreign_key_t>,
                                                  std::optional<std::string_view>,
                                                  std::optional<size_t>,
                                                  std::optional<size_t>,
                                                  std::string_view,
                                                  std::optional<std::string_view>,
                                                  std::optional<std::string_view>,
                                                  std::optional<std::string_view>,
                                                  std::optional<std::string_view>,
                                                  std::optional<std::string_view>,
                                                  std::optional<std::string_view>,
                                                  std::optional<std::string_view>,
                                                  std::optional<std::string_view>,
                                                  std::optional<size_t>,
                                                  std::optional<size_t>,
                                                  std::string_view>;

    using code_object_info_statement_func_t =
        statement_t<integer_primary_key_t,
                    integer_foreign_key_t,
                    integer_foreign_key_t,
                    std::optional<integer_foreign_key_t>,
                    std::optional<std::string_view>,
                    std::optional<size_t>,
                    std::optional<size_t>,
                    std::optional<size_t>,
                    std::optional<std::string_view>,
                    std::string_view>;

    using kernel_symbol_info_statement_func_t =
        statement_t<integer_primary_key_t,
                    integer_foreign_key_t,
                    integer_foreign_key_t,
                    integer_foreign_key_t,
                    std::optional<std::string_view>,
                    std::optional<std::string_view>,
                    std::optional<size_t>,
                    std::optional<size_t>,
                    std::optional<size_t>,
                    std::optional<size_t>,
                    std::optional<size_t>,
                    std::optional<size_t>,
                    std::optional<size_t>,
                    std::optional<size_t>,
                    std::string_view>;

    using track_info_statement_func_t = statement_t<integer_primary_key_t,
                                                    integer_foreign_key_t,
                                                    std::optional<integer_foreign_key_t>,
                                                    std::optional<integer_foreign_key_t>,
                                                    std::optional<integer_foreign_key_t>,
                                                    std::string_view>;

    using event_statement_func_t = statement_t<integer_primary_key_t,
                                               std::optional<integer_foreign_key_t>,
                                               std::optional<size_t>,
                                               std::optional<size_t>,
                                               std::optional<size_t>,
                                               std::string_view,
                                               std::string_view,
                                               std::string_view>;

    using arg_statement_func_t = statement_t<integer_primary_key_t,
                                             integer_foreign_key_t,
                                             size_t,
                                             std::string_view,
                                             std::string_view,
                                             std::optional<std::string_view>,
                                             std::string_view>;

    using pmc_event_statement_func_t = statement_t<integer_primary_key_t,
                                                   std::optional<integer_foreign_key_t>,
                                                   integer_foreign_key_t,
                                                   double,
                                                   std::string_view>;

    using region_statement_func_t = statement_t<integer_primary_key_t,
                                                integer_foreign_key_t,
                                                integer_foreign_key_t,
                                                integer_foreign_key_t,
                                                uint64_t,
                                                uint64_t,
                                                integer_foreign_key_t,
                                                std::optional<integer_foreign_key_t>,
                                                std::string_view>;

    using sample_statement_func_t = statement_t<integer_primary_key_t,
                                                integer_foreign_key_t,
                                                uint64_t,
                                                std::optional<integer_foreign_key_t>,
                                                std::string_view>;

    using kernel_dispatch_statement_func_t =
        statement_t<integer_primary_key_t,
                    integer_foreign_key_t,
                    integer_foreign_key_t,
                    std::optional<integer_foreign_key_t>,
                    integer_foreign_key_t,
                    integer_foreign_key_t,
                    size_t,
                    integer_foreign_key_t,
                    integer_foreign_key_t,
                    uint64_t,
                    uint64_t,
                    std::optional<size_t>,
                    std::optional<size_t>,
                    size_t,
                    size_t,
                    size_t,
                    size_t,
                    size_t,
                    size_t,
                    std::optional<integer_foreign_key_t>,
                    std::optional<integer_foreign_key_t>,
                    std::string_view>;

    using memory_copy_statement_func_t = statement_t<integer_primary_key_t,
                                                     integer_foreign_key_t,
                                                     integer_foreign_key_t,
                                                     std::optional<integer_foreign_key_t>,
                                                     uint64_t,
                                                     uint64_t,
                                                     integer_foreign_key_t,
                                                     std::optional<integer_foreign_key_t>,
                                                     std::optional<size_t>,
                                                     std::optional<integer_foreign_key_t>,
                                                     std::optional<size_t>,
                                                     size_t,
                                                     std::optional<integer_foreign_key_t>,
                                                     std::optional<integer_foreign_key_t>,
                                                     std::optional<integer_foreign_key_t>,
                                                     std::optional<integer_foreign_key_t>,
                                                     std::string_view>;

    using memory_alloc_statement_func_t =
        statement_t<integer_primary_key_t,
                    integer_foreign_key_t,
                    integer_foreign_key_t,
                    std::optional<integer_foreign_key_t>,
                    std::optional<integer_foreign_key_t>,
                    std::optional<std::string_view>,
                    std::optional<std::string_view>,
                    uint64_t,
                    uint64_t,
                    std::optional<size_t>,
                    size_t,
                    std::optional<integer_foreign_key_t>,
                    std::optional<integer_foreign_key_t>,
                    std::optional<integer_foreign_key_t>,
                    std::string_view>;

public:
    [[nodiscard]] const string_statement_func_t& string_statement() const
    {
        return m_string_statement;
    }

    [[nodiscard]] const node_info_statement_func_t& node_info_statement() const
    {
        return m_node_info_statement;
    }

    [[nodiscard]] const process_info_statement_func_t& process_info_statement() const
    {
        return m_process_info_statement;
    }

    [[nodiscard]] const agent_info_statement_func_t& agent_info_statement() const
    {
        return m_agent_info_statement;
    }

    [[nodiscard]] const pmc_info_statement_func_t& pmc_info_statement() const
    {
        return m_pmc_info_statement;
    }

    [[nodiscard]] const thread_info_statement_func_t& thread_info_statement() const
    {
        return m_thread_info_statement;
    }

    [[nodiscard]] const stream_info_statement_func_t& stream_info_statement() const
    {
        return m_stream_info_statement;
    }

    [[nodiscard]] const queue_info_statement_func_t& queue_info_statement() const
    {
        return m_queue_info_statement;
    }

    [[nodiscard]] const kernel_symbol_info_statement_func_t&
    kernel_symbol_info_statement() const
    {
        return m_kernel_symbol_info_statement;
    }

    [[nodiscard]] const code_object_info_statement_func_t& code_object_info_statement()
        const
    {
        return m_code_object_info_statement;
    }

    [[nodiscard]] const track_info_statement_func_t& track_info_statement() const
    {
        return m_track_info_statement;
    }

    [[nodiscard]] const event_statement_func_t& event_statement() const
    {
        return m_event_statement;
    }

    [[nodiscard]] const arg_statement_func_t& arg_statement() const
    {
        return m_arg_statement;
    }

    [[nodiscard]] const pmc_event_statement_func_t& pmc_event_statement() const
    {
        return m_pmc_event_statement;
    }

    [[nodiscard]] const region_statement_func_t& region_statement() const
    {
        return m_region_statement;
    }

    [[nodiscard]] const sample_statement_func_t& sample_statement() const
    {
        return m_sample_statement;
    }

    [[nodiscard]] const kernel_dispatch_statement_func_t& kernel_dispatch_statement()
        const
    {
        return m_kernel_dispatch_statement;
    }

    [[nodiscard]] const memory_copy_statement_func_t& memory_copy_statement() const
    {
        return m_memory_copy_statement;
    }

    [[nodiscard]] const memory_alloc_statement_func_t& memory_alloc_statement() const
    {
        return m_memory_alloc_statement;
    }

private:
    template <typename... Ts>
    void create_statement(statement_t<Ts...>& target, const std::string& query)
    {
        target = m_backend->template create_write_statement_executor<Ts...>(query);
    }

    void initialize_string_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto query = query_builder.set_table_name(fmt::format("rocpd_string_{}", m_uuid))
                         .set_columns("id", "string")
                         .set_values('?', '?')
                         .get_query_string();
        create_statement(m_string_statement, query);
    }

    void initialize_node_info_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto                                              query =
            query_builder.set_table_name(fmt::format("rocpd_info_node_{}", m_uuid))
                .set_columns("id",
                             "hash",
                             "machine_id",
                             "system_name",
                             "hostname",
                             "release",
                             "version",
                             "hardware_name",
                             "domain_name")
                .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_node_info_statement, query);
    }

    void initialize_process_info_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto                                              query =
            query_builder.set_table_name(fmt::format("rocpd_info_process_{}", m_uuid))
                .set_columns("id",
                             "nid",
                             "ppid",
                             "pid",
                             "init",
                             "fini",
                             "start",
                             "end",
                             "command",
                             "environment",
                             "extdata")
                .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_process_info_statement, query);
    }

    void initialize_thread_info_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto                                              query =
            query_builder.set_table_name(fmt::format("rocpd_info_thread_{}", m_uuid))
                .set_columns(
                    "id", "nid", "ppid", "pid", "tid", "name", "start", "end", "extdata")
                .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_thread_info_statement, query);
    }

    void initialize_agent_info_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto                                              query =
            query_builder.set_table_name(fmt::format("rocpd_info_agent_{}", m_uuid))
                .set_columns("id",
                             "nid",
                             "pid",
                             "type",
                             "absolute_index",
                             "logical_index",
                             "type_index",
                             "uuid",
                             "name",
                             "model_name",
                             "vendor_name",
                             "product_name",
                             "user_name",
                             "extdata")
                .set_values(
                    '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_agent_info_statement, query);
    }

    void initialize_pmc_info_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto                                              query =
            query_builder.set_table_name(fmt::format("rocpd_info_pmc_{}", m_uuid))
                .set_columns("id",
                             "nid",
                             "pid",
                             "agent_id",
                             "target_arch",
                             "event_code",
                             "instance_id",
                             "name",
                             "symbol",
                             "description",
                             "long_description",
                             "component",
                             "units",
                             "value_type",
                             "block",
                             "expression",
                             "is_constant",
                             "is_derived",
                             "extdata")
                .set_values('?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?')
                .get_query_string();
        create_statement(m_pmc_info_statement, query);
    }

    void initialize_stream_info_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto                                              query =
            query_builder.set_table_name(fmt::format("rocpd_info_stream_{}", m_uuid))
                .set_columns("id", "nid", "pid", "name", "extdata")
                .set_values('?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_stream_info_statement, query);
    }

    void initialize_queue_info_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto                                              query =
            query_builder.set_table_name(fmt::format("rocpd_info_queue_{}", m_uuid))
                .set_columns("id", "nid", "pid", "name", "extdata")
                .set_values('?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_queue_info_statement, query);
    }

    void initialize_kernel_symbol_info_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto                                              query =
            query_builder
                .set_table_name(fmt::format("rocpd_info_kernel_symbol_{}", m_uuid))
                .set_columns("id",
                             "nid",
                             "pid",
                             "code_object_id",
                             "kernel_name",
                             "display_name",
                             "kernel_object",
                             "kernarg_segment_size",
                             "kernarg_segment_alignment",
                             "group_segment_size",
                             "private_segment_size",
                             "sgpr_count",
                             "arch_vgpr_count",
                             "accum_vgpr_count",
                             "extdata")
                .set_values('?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?')
                .get_query_string();
        create_statement(m_kernel_symbol_info_statement, query);
    }

    void initialize_code_object_info_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto                                              query =
            query_builder.set_table_name(fmt::format("rocpd_info_code_object_{}", m_uuid))
                .set_columns("id",
                             "nid",
                             "pid",
                             "agent_id",
                             "uri",
                             "load_base",
                             "load_size",
                             "load_delta",
                             "storage_type",
                             "extdata")
                .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_code_object_info_statement, query);
    }

    void initialize_track_info_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto query = query_builder.set_table_name(fmt::format("rocpd_track_{}", m_uuid))
                         .set_columns("id", "nid", "pid", "tid", "name_id", "extdata")
                         .set_values('?', '?', '?', '?', '?', '?')
                         .get_query_string();
        create_statement(m_track_info_statement, query);
    }

    void initialize_event_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto query = query_builder.set_table_name(fmt::format("rocpd_event_{}", m_uuid))
                         .set_columns("id",
                                      "category_id",
                                      "stack_id",
                                      "parent_stack_id",
                                      "correlation_id",
                                      "call_stack",
                                      "line_info",
                                      "extdata")
                         .set_values('?', '?', '?', '?', '?', '?', '?', '?')
                         .get_query_string();
        create_statement(m_event_statement, query);
    }

    void initialize_arg_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto                                              query =
            query_builder.set_table_name(fmt::format("rocpd_arg_{}", m_uuid))
                .set_columns(
                    "id", "event_id", "position", "type", "name", "value", "extdata")
                .set_values('?', '?', '?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_arg_statement, query);
    }

    void initialize_pmc_event_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto                                              query =
            query_builder.set_table_name(fmt::format("rocpd_pmc_event_{}", m_uuid))
                .set_columns("id", "event_id", "pmc_id", "value", "extdata")
                .set_values('?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_pmc_event_statement, query);
    }

    void initialize_region_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto query = query_builder.set_table_name(fmt::format("rocpd_region_{}", m_uuid))
                         .set_columns("id",
                                      "nid",
                                      "pid",
                                      "tid",
                                      "start",
                                      "end",
                                      "name_id",
                                      "event_id",
                                      "extdata")
                         .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?')
                         .get_query_string();
        create_statement(m_region_statement, query);
    }

    void initialize_sample_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto                                              query =
            query_builder.set_table_name(fmt::format("rocpd_sample_{}", m_uuid))
                .set_columns("id", "track_id", "timestamp", "event_id", "extdata")
                .set_values('?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_sample_statement, query);
    }

    void initialize_kernel_dispatch_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto                                              query =
            query_builder.set_table_name(fmt::format("rocpd_kernel_dispatch_{}", m_uuid))
                .set_columns("id",
                             "nid",
                             "pid",
                             "tid",
                             "agent_id",
                             "kernel_id",
                             "dispatch_id",
                             "queue_id",
                             "stream_id",
                             "start",
                             "end",
                             "private_segment_size",
                             "group_segment_size",
                             "workgroup_size_x",
                             "workgroup_size_y",
                             "workgroup_size_z",
                             "grid_size_x",
                             "grid_size_y",
                             "grid_size_z",
                             "region_name_id",
                             "event_id",
                             "extdata")
                .set_values('?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?')
                .get_query_string();
        create_statement(m_kernel_dispatch_statement, query);
    }

    void initialize_memory_copy_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto                                              query =
            query_builder.set_table_name(fmt::format("rocpd_memory_copy_{}", m_uuid))
                .set_columns("id",
                             "nid",
                             "pid",
                             "tid",
                             "start",
                             "end",
                             "name_id",
                             "dst_agent_id",
                             "dst_address",
                             "src_agent_id",
                             "src_address",
                             "size",
                             "queue_id",
                             "stream_id",
                             "region_name_id",
                             "event_id",
                             "extdata")
                .set_values('?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?')
                .get_query_string();
        create_statement(m_memory_copy_statement, query);
    }

    void initialize_memory_alloc_statement()
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        auto                                              query =
            query_builder.set_table_name(fmt::format("rocpd_memory_allocate_{}", m_uuid))
                .set_columns("id",
                             "nid",
                             "pid",
                             "tid",
                             "agent_id",
                             "type",
                             "level",
                             "start",
                             "end",
                             "address",
                             "size",
                             "queue_id",
                             "stream_id",
                             "event_id",
                             "extdata")
                .set_values('?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?')
                .get_query_string();
        create_statement(m_memory_alloc_statement, query);
    }

    std::shared_ptr<Backend> m_backend;
    std::string              m_uuid;

    string_statement_func_t             m_string_statement;
    node_info_statement_func_t          m_node_info_statement;
    process_info_statement_func_t       m_process_info_statement;
    agent_info_statement_func_t         m_agent_info_statement;
    pmc_info_statement_func_t           m_pmc_info_statement;
    thread_info_statement_func_t        m_thread_info_statement;
    stream_info_statement_func_t        m_stream_info_statement;
    queue_info_statement_func_t         m_queue_info_statement;
    kernel_symbol_info_statement_func_t m_kernel_symbol_info_statement;
    code_object_info_statement_func_t   m_code_object_info_statement;
    track_info_statement_func_t         m_track_info_statement;
    event_statement_func_t              m_event_statement;
    arg_statement_func_t                m_arg_statement;
    pmc_event_statement_func_t          m_pmc_event_statement;
    region_statement_func_t             m_region_statement;
    sample_statement_func_t             m_sample_statement;
    kernel_dispatch_statement_func_t    m_kernel_dispatch_statement;
    memory_copy_statement_func_t        m_memory_copy_statement;
    memory_alloc_statement_func_t       m_memory_alloc_statement;
};

}  // namespace profiler_hub::data_storage::schema_v3
