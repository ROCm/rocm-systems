// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "insert_statements.hpp"
#include "database.hpp"
#include "queries/insert/table_insert_query.hpp"
#include "spdlog/fmt/bundled/core.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace rocstorage::data_storage::schema_v3
{

insert_statements::insert_statements(std::shared_ptr<database> database, std::string uuid)
: m_database(std::move(database))
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

void
insert_statements::initialize_string_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto query = query_builder.set_table_name(fmt::format("rocpd_string_{}", m_uuid))
                     .set_columns("id", "string")
                     .set_values('?', '?')
                     .get_query_string();

    m_string_statement =
        m_database->create_statement_executor<size_t, const char*>(query);
}

void
insert_statements::initialize_node_info_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto query = query_builder.set_table_name(fmt::format("rocpd_info_node_{}", m_uuid))
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

    m_node_info_statement = m_database->create_statement_executor<integer_primary_key_t,
                                                                  size_t,
                                                                  const char*,
                                                                  const char*,
                                                                  const char*,
                                                                  const char*,
                                                                  const char*,
                                                                  const char*,
                                                                  const char*>(query);
}

void
insert_statements::initialize_process_info_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto                                            query =
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

    m_process_info_statement =
        m_database->create_statement_executor<integer_primary_key_t,
                                              integer_foreign_key_t,
                                              std::optional<size_t>,
                                              size_t,
                                              std::optional<size_t>,
                                              std::optional<size_t>,
                                              std::optional<size_t>,
                                              std::optional<size_t>,
                                              const char*,
                                              const char*,
                                              const char*>(query);
}

void
insert_statements::initialize_thread_info_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto                                            query =
        query_builder.set_table_name(fmt::format("rocpd_info_thread_{}", m_uuid))
            .set_columns(
                "id", "nid", "ppid", "pid", "tid", "name", "start", "end", "extdata")
            .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?')
            .get_query_string();
    m_thread_info_statement = m_database->create_statement_executor<integer_primary_key_t,
                                                                    integer_foreign_key_t,
                                                                    std::optional<size_t>,
                                                                    integer_foreign_key_t,
                                                                    size_t,
                                                                    const char*,
                                                                    std::optional<size_t>,
                                                                    std::optional<size_t>,
                                                                    const char*>(query);
}

void
insert_statements::initialize_agent_info_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto                                            query =
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

    m_agent_info_statement = m_database->create_statement_executor<integer_primary_key_t,
                                                                   integer_foreign_key_t,
                                                                   integer_foreign_key_t,
                                                                   const char*,
                                                                   std::optional<size_t>,
                                                                   std::optional<size_t>,
                                                                   size_t,
                                                                   std::optional<size_t>,
                                                                   const char*,
                                                                   const char*,
                                                                   const char*,
                                                                   const char*,
                                                                   const char*,
                                                                   const char*>(query);
}

void
insert_statements::initialize_pmc_info_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto query = query_builder.set_table_name(fmt::format("rocpd_info_pmc_{}", m_uuid))
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
    m_pmc_info_statement =
        m_database->create_statement_executor<integer_primary_key_t,
                                              integer_foreign_key_t,
                                              integer_foreign_key_t,
                                              std::optional<integer_foreign_key_t>,
                                              const char*,
                                              std::optional<size_t>,
                                              std::optional<size_t>,
                                              const char*,
                                              const char*,
                                              const char*,
                                              const char*,
                                              const char*,
                                              const char*,
                                              const char*,
                                              const char*,
                                              const char*,
                                              std::optional<size_t>,
                                              std::optional<size_t>,
                                              const char*>(query);
}

void
insert_statements::initialize_stream_info_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto query = query_builder.set_table_name(fmt::format("rocpd_info_stream_{}", m_uuid))
                     .set_columns("id", "nid", "pid", "name", "extdata")
                     .set_values('?', '?', '?', '?', '?')
                     .get_query_string();
    m_stream_info_statement = m_database->create_statement_executor<integer_primary_key_t,
                                                                    integer_foreign_key_t,
                                                                    integer_foreign_key_t,
                                                                    const char*,
                                                                    const char*>(query);
}

void
insert_statements::initialize_queue_info_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto query = query_builder.set_table_name(fmt::format("rocpd_info_queue_{}", m_uuid))
                     .set_columns("id", "nid", "pid", "name", "extdata")
                     .set_values('?', '?', '?', '?', '?')
                     .get_query_string();
    m_queue_info_statement = m_database->create_statement_executor<integer_primary_key_t,
                                                                   integer_foreign_key_t,
                                                                   integer_foreign_key_t,
                                                                   const char*,
                                                                   const char*>(query);
}

void
insert_statements::initialize_kernel_symbol_info_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto                                            query =
        query_builder.set_table_name(fmt::format("rocpd_info_kernel_symbol_{}", m_uuid))
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
            .set_values(
                '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
            .get_query_string();
    m_kernel_symbol_info_statement =
        m_database->create_statement_executor<integer_primary_key_t,
                                              integer_foreign_key_t,
                                              integer_foreign_key_t,
                                              integer_foreign_key_t,
                                              const char*,
                                              const char*,
                                              std::optional<size_t>,
                                              std::optional<size_t>,
                                              std::optional<size_t>,
                                              std::optional<size_t>,
                                              std::optional<size_t>,
                                              std::optional<size_t>,
                                              std::optional<size_t>,
                                              std::optional<size_t>,
                                              const char*>(query);
}

void
insert_statements::initialize_code_object_info_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto                                            query =
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
    m_code_object_info_statement =
        m_database->create_statement_executor<integer_primary_key_t,
                                              integer_foreign_key_t,
                                              integer_foreign_key_t,
                                              std::optional<integer_foreign_key_t>,
                                              const char*,
                                              std::optional<size_t>,
                                              std::optional<size_t>,
                                              std::optional<size_t>,
                                              const char*,
                                              const char*>(query);
}

void
insert_statements::initialize_track_info_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto query = query_builder.set_table_name(fmt::format("rocpd_track_{}", m_uuid))
                     .set_columns("id", "nid", "pid", "tid", "name_id", "extdata")
                     .set_values('?', '?', '?', '?', '?', '?')
                     .get_query_string();
    m_track_info_statement =
        m_database->create_statement_executor<integer_primary_key_t,
                                              integer_foreign_key_t,
                                              std::optional<integer_foreign_key_t>,
                                              std::optional<integer_foreign_key_t>,
                                              std::optional<integer_foreign_key_t>,
                                              const char*>(query);
}

void
insert_statements::initialize_event_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
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
    m_event_statement =
        m_database->create_statement_executor<integer_primary_key_t,
                                              std::optional<integer_foreign_key_t>,
                                              std::optional<size_t>,
                                              std::optional<size_t>,
                                              std::optional<size_t>,
                                              const char*,
                                              const char*,
                                              const char*>(query);
}

void
insert_statements::initialize_arg_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto                                            query =
        query_builder.set_table_name(fmt::format("rocpd_arg_{}", m_uuid))
            .set_columns("id", "event_id", "position", "type", "name", "value", "extdata")
            .set_values('?', '?', '?', '?', '?', '?', '?')
            .get_query_string();
    m_arg_statement = m_database->create_statement_executor<integer_primary_key_t,
                                                            integer_foreign_key_t,
                                                            size_t,
                                                            const char*,
                                                            const char*,
                                                            const char*,
                                                            const char*>(query);
}

void
insert_statements::initialize_pmc_event_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto query = query_builder.set_table_name(fmt::format("rocpd_pmc_event_{}", m_uuid))
                     .set_columns("id", "event_id", "pmc_id", "value", "extdata")
                     .set_values('?', '?', '?', '?', '?')
                     .get_query_string();
    m_pmc_event_statement =
        m_database->create_statement_executor<integer_primary_key_t,
                                              std::optional<integer_foreign_key_t>,
                                              integer_foreign_key_t,
                                              double,
                                              const char*>(query);
}

void
insert_statements::initialize_region_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
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
    m_region_statement =
        m_database->create_statement_executor<integer_primary_key_t,
                                              integer_foreign_key_t,
                                              integer_foreign_key_t,
                                              integer_foreign_key_t,
                                              uint64_t,
                                              uint64_t,
                                              integer_foreign_key_t,
                                              std::optional<integer_foreign_key_t>,
                                              const char*>(query);
}

void
insert_statements::initialize_sample_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto query = query_builder.set_table_name(fmt::format("rocpd_sample_{}", m_uuid))
                     .set_columns("id", "track_id", "timestamp", "event_id", "extdata")
                     .set_values('?', '?', '?', '?', '?')
                     .get_query_string();
    m_sample_statement =
        m_database->create_statement_executor<integer_primary_key_t,
                                              integer_foreign_key_t,
                                              uint64_t,
                                              std::optional<integer_foreign_key_t>,
                                              const char*>(query);
}

void
insert_statements::initialize_kernel_dispatch_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto                                            query =
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
    m_kernel_dispatch_statement =
        m_database->create_statement_executor<integer_primary_key_t,
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
                                              const char*>(query);
}

void
insert_statements::initialize_memory_copy_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto query = query_builder.set_table_name(fmt::format("rocpd_memory_copy_{}", m_uuid))
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
    m_memory_copy_statement =
        m_database->create_statement_executor<integer_primary_key_t,
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
                                              const char*>(query);
}

void
insert_statements::initialize_memory_alloc_statement()
{
    rocstorage::queries::insert::table_insert_query query_builder;
    auto                                            query =
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
            .set_values(
                '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
            .get_query_string();
    m_memory_alloc_statement =
        m_database->create_statement_executor<integer_primary_key_t,
                                              integer_foreign_key_t,
                                              integer_foreign_key_t,
                                              std::optional<integer_foreign_key_t>,
                                              std::optional<integer_foreign_key_t>,
                                              const char*,
                                              const char*,
                                              uint64_t,
                                              uint64_t,
                                              std::optional<size_t>,
                                              size_t,
                                              std::optional<integer_foreign_key_t>,
                                              std::optional<integer_foreign_key_t>,
                                              std::optional<integer_foreign_key_t>,
                                              const char*>(query);
}

}  // namespace rocstorage::data_storage::schema_v3
