// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "insert_statements.hpp"
#include "table_insert_query.hpp"

namespace rocstorage
{
namespace data_storage
{

insert_statements::insert_statements(std::shared_ptr<database> database, std::string uuid)
: m_database(std::move(database))
, m_uuid(std::move(uuid))
{
    initialize_event_stmt();
    initialize_pmc_event_stmt();
    initialize_sample_stmt();
    initialize_region_stmt();
    initialize_kernel_dispatch_stmt();
    initialize_memory_copy_stmt();
    initialize_kernel_symbol_stmt();
    initialize_code_object_stmt();
    initialize_args_stmt();
    initialize_memory_alloc_stmt();
}

void
insert_statements::initialize_event_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_event_" + m_uuid)
                     .set_columns("guid", "category_id", "stack_id", "parent_stack_id",
                                  "correlation_id", "call_stack", "line_info", "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?', '?')
                     .get_query_string();
    m_insert_event_statement =
        m_database->create_statement_executor<const char*, size_t, size_t, size_t, size_t,
                                              const char*, const char*, const char*>(
            query);
}

void
insert_statements::initialize_pmc_event_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_pmc_event_" + m_uuid)
                     .set_columns("guid", "event_id", "pmc_id", "value", "extdata")
                     .set_values('?', '?', '?', '?', '?')
                     .get_query_string();
    m_insert_pmc_event_statement =
        m_database
            ->create_statement_executor<const char*, size_t, size_t, double, const char*>(
                query);
}

void
insert_statements::initialize_sample_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_sample_" + m_uuid)
                     .set_columns("guid", "track_id", "timestamp", "event_id", "extdata")
                     .set_values('?', '?', '?', '?', '?')
                     .get_query_string();
    m_insert_sample_statement =
        m_database->create_statement_executor<const char*, size_t, uint64_t, size_t,
                                              const char*>(query);
}

void
insert_statements::initialize_region_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_region_" + m_uuid)
                     .set_columns("guid", "nid", "pid", "tid", "start", "end", "name_id",
                                  "event_id", "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?')
                     .get_query_string();
    m_insert_region_statement =
        m_database
            ->create_statement_executor<const char*, size_t, size_t, size_t, uint64_t,
                                        uint64_t, size_t, size_t, const char*>(query);
}

void
insert_statements::initialize_kernel_dispatch_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_kernel_dispatch_" + m_uuid)
                     .set_columns("guid", "nid", "pid", "tid", "agent_id", "kernel_id",
                                  "dispatch_id", "queue_id", "stream_id", "start", "end",
                                  "private_segment_size", "group_segment_size",
                                  "workgroup_size_x", "workgroup_size_y",
                                  "workgroup_size_z", "grid_size_x", "grid_size_y",
                                  "grid_size_z", "region_name_id", "event_id", "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?',
                                 '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
                     .get_query_string();
    m_insert_kernel_dispatch_statement = m_database->create_statement_executor<
        const char*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t,
        uint64_t, uint64_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t,
        size_t, size_t, size_t, const char*>(query);
}

void
insert_statements::initialize_memory_copy_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_memory_copy_" + m_uuid)
                     .set_columns("guid", "nid", "pid", "tid", "start", "end", "name_id",
                                  "dst_agent_id", "dst_address", "src_agent_id",
                                  "src_address", "size", "queue_id", "stream_id",
                                  "region_name_id", "event_id", "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?',
                                 '?', '?', '?', '?', '?', '?')
                     .get_query_string();
    m_insert_memory_copy_statement = m_database->create_statement_executor<
        const char*, size_t, size_t, size_t, uint64_t, uint64_t, size_t, size_t, size_t,
        size_t, size_t, size_t, size_t, size_t, size_t, size_t, const char*>(query);
}

void
insert_statements::initialize_kernel_symbol_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto                                      query =
        query_builder.set_table_name("rocpd_info_kernel_symbol_" + m_uuid)
            .set_columns("id", "guid", "nid", "pid", "code_object_id", "kernel_name",
                         "display_name", "kernel_object", "kernarg_segment_size",
                         "kernarg_segment_alignment", "group_segment_size",
                         "private_segment_size", "sgpr_count", "arch_vgpr_count",
                         "accum_vgpr_count", "extdata")
            .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?',
                        '?', '?', '?')
            .get_query_string();
    m_insert_kernel_symbol_statement = m_database->create_statement_executor<
        size_t, const char*, size_t, size_t, uint64_t, const char*, const char*, uint64_t,
        uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
        const char*>(query);
}

void
insert_statements::initialize_code_object_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto                                      query =
        query_builder.set_table_name("rocpd_info_code_object_" + m_uuid)
            .set_columns("id", "guid", "nid", "pid", "agent_id", "uri", "load_base",
                         "load_size", "load_delta", "storage_type", "extdata")
            .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
            .get_query_string();
    m_insert_code_object_statement =
        m_database->create_statement_executor<size_t, const char*, size_t, size_t, size_t,
                                              const char*, uint64_t, uint64_t, uint64_t,
                                              const char*, const char*>(query);
}

void
insert_statements::initialize_args_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_arg_" + m_uuid)
                     .set_columns("guid", "event_id", "position", "type", "name", "value",
                                  "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?')
                     .get_query_string();
    m_insert_args_statement =
        m_database->create_statement_executor<const char*, size_t, size_t, const char*,
                                              const char*, const char*, const char*>(
            query);
}

void
insert_statements::initialize_memory_alloc_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_memory_allocate_" + m_uuid)
                     .set_columns("guid", "nid", "pid", "tid", "agent_id", "type",
                                  "level", "start", "end", "address", "size", "queue_id",
                                  "stream_id", "event_id", "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?',
                                 '?', '?', '?', '?')
                     .get_query_string();
    m_insert_memory_alloc_statement = m_database->create_statement_executor<
        const char*, size_t, size_t, size_t, size_t, const char*, const char*, uint64_t,
        uint64_t, size_t, size_t, size_t, size_t, size_t, const char*>(query);

    // Statement without agent_id
    query = query_builder.set_table_name("rocpd_memory_allocate_" + m_uuid)
                .set_columns("guid", "nid", "pid", "tid", "type", "level", "start", "end",
                             "address", "size", "queue_id", "stream_id", "event_id",
                             "extdata")
                .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?',
                            '?', '?')
                .get_query_string();
    m_insert_memory_alloc_no_agent_statement = m_database->create_statement_executor<
        const char*, size_t, size_t, size_t, const char*, const char*, uint64_t, uint64_t,
        size_t, size_t, size_t, size_t, size_t, const char*>(query);
}
}  // namespace data_storage
}  // namespace rocstorage
