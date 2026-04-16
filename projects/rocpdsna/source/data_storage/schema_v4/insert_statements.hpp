// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

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

namespace rocpdsna::data_storage::schema_v4
{
using integer_primary_key_t = size_t;
using integer_foreign_key_t = size_t;

/**
 * @brief Schema v4 (latest) insert statements
 *
 * Key differences from schema_v3:
 * - rocpd_event: NO call_stack or line_info columns
 * - rocpd_info_agent: NO user_name, HAS generic_name
 * - rocpd_region: Uses track_id, start_id, end_id (references rocpd_timestamp)
 * - rocpd_kernel_dispatch: Uses track_id, start_id, end_id
 * - rocpd_memory_copy: Uses track_id, start_id, end_id
 * - rocpd_memory_allocate: Uses track_id, start_id, end_id, name_id, region_name_id
 * - rocpd_sample: Uses name_id, timestamp_id
 * - New tables: rocpd_timestamp, rocpd_call_stack, rocpd_line_info, rocpd_info_category
 */
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
        initialize_timestamp_statement();
        initialize_category_info_statement();
        initialize_address_range_info_statement();
        initialize_source_code_info_statement();
        initialize_pc_info_statement();
        initialize_event_statement();
        initialize_arg_statement();
        initialize_call_stack_statement();
        initialize_line_info_statement();
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

    using string_statement_func_t = statement_t<size_t,             // id
                                                std::string_view>;  // string

    using node_info_statement_func_t =
        statement_t<integer_primary_key_t,             // id
                    size_t,                            // hash
                    std::string_view,                  // machine_id
                    std::optional<std::string_view>,   // name  V4 addition
                    std::optional<std::string_view>,   // system_name
                    std::optional<std::string_view>,   // hostname
                    std::optional<std::string_view>,   // release
                    std::optional<std::string_view>,   // version
                    std::optional<std::string_view>,   // hardware_name
                    std::optional<std::string_view>>;  // domain_name

    using process_info_statement_func_t =
        statement_t<integer_primary_key_t,            // id
                    integer_foreign_key_t,            // nid
                    std::optional<size_t>,            // ppid
                    size_t,                           // pid
                    std::optional<std::string_view>,  // name  v4 addition
                    std::optional<size_t>,            // init
                    std::optional<size_t>,            // fini
                    std::optional<size_t>,            // start
                    std::optional<size_t>,            // end
                    std::optional<std::string_view>,  // command
                    std::string_view,                 // environment
                    std::string_view>;                // extdata

    using thread_info_statement_func_t =
        statement_t<integer_primary_key_t,            // id
                    integer_foreign_key_t,            // nid
                    std::optional<size_t>,            // ppid
                    integer_foreign_key_t,            // pid
                    size_t,                           // tid
                    std::optional<std::string_view>,  // name
                    std::optional<size_t>,            // start
                    std::optional<size_t>,            // end
                    std::string_view>;                // extdata

    // Schema v4: agent_info WITHOUT user_name, WITH generic_name
    using agent_info_statement_func_t =
        statement_t<integer_primary_key_t,            // id
                    integer_foreign_key_t,            // nid
                    integer_foreign_key_t,            // pid
                    std::optional<std::string_view>,  // type
                    std::optional<size_t>,            // absolute_index
                    std::optional<size_t>,            // logical_index
                    size_t,                           // type_index
                    std::optional<size_t>,            // uuid
                    std::optional<std::string_view>,  // name
                    std::optional<std::string_view>,  // generic_name
                    std::optional<std::string_view>,  // model_name
                    std::optional<std::string_view>,  // vendor_name
                    std::optional<std::string_view>,  // product_name
                    std::string_view>;                // extdata

    using queue_info_statement_func_t =
        statement_t<integer_primary_key_t,            // id
                    integer_foreign_key_t,            // nid
                    integer_foreign_key_t,            // pid
                    std::optional<std::string_view>,  // name
                    std::string_view>;                // extdata

    using stream_info_statement_func_t =
        statement_t<integer_primary_key_t,            // id
                    integer_foreign_key_t,            // nid
                    integer_foreign_key_t,            // pid
                    std::optional<std::string_view>,  // name
                    std::string_view>;                // extdata

    using pmc_info_statement_func_t =
        statement_t<integer_primary_key_t,                 // id
                    integer_foreign_key_t,                 // nid
                    integer_foreign_key_t,                 // pid
                    std::optional<integer_foreign_key_t>,  // agent_id
                    std::optional<std::string_view>,       // target_arch
                    std::optional<size_t>,                 // event_code
                    std::optional<size_t>,                 // instance_id
                    std::string_view,                      // name
                    std::optional<std::string_view>,       // symbol
                    std::optional<std::string_view>,       // qualifier (V4 ONLY)
                    std::optional<std::string_view>,       // description
                    std::optional<std::string_view>,       // long_description
                    std::optional<std::string_view>,       // component
                    std::optional<std::string_view>,       // units
                    std::optional<std::string_view>,       // value_type
                    std::optional<std::string_view>,       // block
                    std::optional<std::string_view>,       // expression
                    std::optional<size_t>,                 // is_constant
                    std::optional<size_t>,                 // is_derived
                    std::string_view>;                     // extdata

    using code_object_info_statement_func_t =
        statement_t<integer_primary_key_t,                 // id
                    integer_foreign_key_t,                 // nid
                    integer_foreign_key_t,                 // pid
                    std::optional<integer_foreign_key_t>,  // agent_id
                    std::optional<std::string_view>,       // uri
                    std::optional<size_t>,                 // load_base
                    std::optional<size_t>,                 // load_size
                    std::optional<size_t>,                 // load_delta
                    std::optional<std::string_view>,       // storage_type
                    std::string_view>;                     // extdata

    using kernel_symbol_info_statement_func_t =
        statement_t<integer_primary_key_t,            // id
                    integer_foreign_key_t,            // nid
                    integer_foreign_key_t,            // pid
                    integer_foreign_key_t,            // code_object_id
                    std::optional<std::string_view>,  // kernel_name
                    std::optional<std::string_view>,  // display_name
                    std::optional<size_t>,            // kernel_object
                    std::optional<size_t>,            // kernarg_segment_size
                    std::optional<size_t>,            // kernarg_segment_alignment
                    std::optional<size_t>,            // group_segment_size
                    std::optional<size_t>,            // private_segment_size
                    std::optional<size_t>,            // sgpr_count
                    std::optional<size_t>,            // arch_vgpr_count
                    std::optional<size_t>,            // accum_vgpr_count
                    std::string_view>;                // extdata

    // v4: track has extended columns (agent_id, queue_id, stream_id)
    using track_info_statement_func_t =
        statement_t<integer_primary_key_t,                 // id
                    integer_foreign_key_t,                 // nid
                    std::optional<integer_foreign_key_t>,  // ppid
                    std::optional<integer_foreign_key_t>,  // pid
                    std::optional<integer_foreign_key_t>,  // tid
                    std::optional<integer_foreign_key_t>,  // agent_id
                    std::optional<integer_foreign_key_t>,  // queue_id
                    std::optional<integer_foreign_key_t>,  // stream_id
                    std::optional<integer_foreign_key_t>,  // name_id
                    std::string_view>;                     // extdata

    // NEW in v4: rocpd_timestamp table
    using timestamp_statement_func_t =
        statement_t<integer_primary_key_t,                  // id
                    uint64_t,                               // value
                    std::optional<int>,                     // phase
                    std::optional<integer_foreign_key_t>>;  // track_id

    // NEW in v4: rocpd_info_category table
    using category_info_statement_func_t = statement_t<integer_primary_key_t,  // id
                                                       std::string_view,       // name
                                                       std::string_view>;      // extdata

    // NEW in v4: rocpd_info_address_range table
    using address_range_info_statement_func_t =
        statement_t<integer_primary_key_t,  // id
                    integer_foreign_key_t,  // nid
                    integer_foreign_key_t,  // pid
                    std::optional<size_t>,  // address_base
                    std::optional<size_t>,  // address_low
                    std::optional<size_t>,  // address_high
                    std::string_view>;      // extdata

    // NEW in v4: rocpd_info_source_code table
    using source_code_info_statement_func_t =
        statement_t<integer_primary_key_t,                 // id
                    integer_foreign_key_t,                 // nid
                    integer_foreign_key_t,                 // pid
                    std::optional<integer_foreign_key_t>,  // address_id
                    std::optional<std::string_view>,       // file
                    std::optional<size_t>,                 // line_number
                    std::string_view,                      // lines (JSON)
                    std::string_view,                      // instructions (JSON)
                    std::string_view>;                     // extdata

    // NEW in v4: rocpd_info_pc table
    using pc_info_statement_func_t =
        statement_t<integer_primary_key_t,                 // id
                    integer_foreign_key_t,                 // nid
                    integer_foreign_key_t,                 // pid
                    std::string_view,                      // function (NOT NULL)
                    std::optional<integer_foreign_key_t>,  // address_id
                    std::optional<std::string_view>,       // file
                    std::optional<size_t>,                 // line
                    std::string_view>;                     // extdata

    // v4: event WITHOUT call_stack and line_info columns
    using event_statement_func_t =
        statement_t<integer_primary_key_t,                 // id
                    std::optional<integer_foreign_key_t>,  // category_id
                    std::optional<size_t>,                 // stack_id
                    std::optional<size_t>,                 // parent_stack_id
                    std::optional<size_t>,                 // correlation_id
                    std::string_view>;                     // extdata

    using arg_statement_func_t = statement_t<integer_primary_key_t,            // id
                                             integer_foreign_key_t,            // event_id
                                             size_t,                           // position
                                             std::string_view,                 // type
                                             std::string_view,                 // name
                                             std::optional<std::string_view>,  // value
                                             std::string_view>;                // extdata

    // NEW in v4: rocpd_call_stack table (replaces call_stack JSONB column)
    using call_stack_statement_func_t =
        statement_t<integer_primary_key_t,                 // id
                    integer_foreign_key_t,                 // event_id
                    std::optional<integer_foreign_key_t>,  // pc_id
                    size_t,                                // depth
                    std::string_view>;                     // extdata

    // NEW in v4: rocpd_line_info table (replaces line_info JSONB column)
    using line_info_statement_func_t =
        statement_t<integer_primary_key_t,                 // id
                    integer_foreign_key_t,                 // event_id
                    std::optional<integer_foreign_key_t>,  // source_code_id
                    std::optional<integer_foreign_key_t>,  // pc_id
                    std::string_view>;                     // extdata

    using pmc_event_statement_func_t =
        statement_t<integer_primary_key_t,                 // id
                    std::optional<integer_foreign_key_t>,  // event_id
                    integer_foreign_key_t,                 // pmc_id
                    double,                                // value
                    std::string_view>;                     // extdata

    // v4: region uses track_id, start_id, end_id
    using region_statement_func_t =
        statement_t<integer_primary_key_t,                 // id
                    integer_foreign_key_t,                 // track_id
                    integer_foreign_key_t,                 // name_id
                    integer_foreign_key_t,                 // start_id
                    integer_foreign_key_t,                 // end_id
                    std::optional<integer_foreign_key_t>,  // event_id
                    std::string_view>;                     // extdata

    // v4: sample uses name_id, timestamp_id
    using sample_statement_func_t =
        statement_t<integer_primary_key_t,                 // id
                    integer_foreign_key_t,                 // track_id
                    integer_foreign_key_t,                 // name_id
                    integer_foreign_key_t,                 // timestamp_id
                    std::optional<integer_foreign_key_t>,  // event_id
                    std::string_view>;                     // extdata

    // v4: kernel_dispatch uses track_id, start_id, end_id
    using kernel_dispatch_statement_func_t =
        statement_t<integer_primary_key_t,                 // id
                    integer_foreign_key_t,                 // track_id
                    integer_foreign_key_t,                 // kernel_id
                    size_t,                                // dispatch_id
                    integer_foreign_key_t,                 // start_id
                    integer_foreign_key_t,                 // end_id
                    std::optional<size_t>,                 // private_segment_size
                    std::optional<size_t>,                 // group_segment_size
                    size_t,                                // workgroup_size_x
                    size_t,                                // workgroup_size_y
                    size_t,                                // workgroup_size_z
                    size_t,                                // grid_size_x
                    size_t,                                // grid_size_y
                    size_t,                                // grid_size_z
                    std::optional<integer_foreign_key_t>,  // region_name_id
                    std::optional<integer_foreign_key_t>,  // event_id
                    std::string_view>;                     // extdata

    // v4: memory_copy uses track_id, start_id, end_id
    using memory_copy_statement_func_t =
        statement_t<integer_primary_key_t,                 // id
                    integer_foreign_key_t,                 // track_id
                    integer_foreign_key_t,                 // start_id
                    integer_foreign_key_t,                 // end_id
                    integer_foreign_key_t,                 // name_id
                    std::optional<integer_foreign_key_t>,  // dst_agent_id
                    std::optional<size_t>,                 // dst_address
                    std::optional<integer_foreign_key_t>,  // src_agent_id
                    std::optional<size_t>,                 // src_address
                    size_t,                                // size
                    std::optional<integer_foreign_key_t>,  // region_name_id
                    std::optional<integer_foreign_key_t>,  // event_id
                    std::string_view>;                     // extdata

    // v4: memory_allocate uses track_id, start_id, end_id, name_id
    using memory_alloc_statement_func_t =
        statement_t<integer_primary_key_t,                 // id
                    integer_foreign_key_t,                 // track_id
                    std::optional<std::string_view>,       // type
                    std::optional<std::string_view>,       // level
                    integer_foreign_key_t,                 // start_id
                    integer_foreign_key_t,                 // end_id
                    integer_foreign_key_t,                 // name_id
                    std::optional<size_t>,                 // address
                    size_t,                                // size
                    std::optional<integer_foreign_key_t>,  // region_name_id
                    std::optional<integer_foreign_key_t>,  // event_id
                    std::string_view>;                     // extdata

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

    [[nodiscard]] const timestamp_statement_func_t& timestamp_statement() const
    {
        return m_timestamp_statement;
    }

    [[nodiscard]] const category_info_statement_func_t& category_info_statement() const
    {
        return m_category_info_statement;
    }

    [[nodiscard]] const address_range_info_statement_func_t&
    address_range_info_statement() const
    {
        return m_address_range_info_statement;
    }

    [[nodiscard]] const source_code_info_statement_func_t& source_code_info_statement()
        const
    {
        return m_source_code_info_statement;
    }

    [[nodiscard]] const pc_info_statement_func_t& pc_info_statement() const
    {
        return m_pc_info_statement;
    }

    [[nodiscard]] const event_statement_func_t& event_statement() const
    {
        return m_event_statement;
    }

    [[nodiscard]] const arg_statement_func_t& arg_statement() const
    {
        return m_arg_statement;
    }

    [[nodiscard]] const call_stack_statement_func_t& call_stack_statement() const
    {
        return m_call_stack_statement;
    }

    [[nodiscard]] const line_info_statement_func_t& line_info_statement() const
    {
        return m_line_info_statement;
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
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto query = query_builder.set_table_name(fmt::format("rocpd_string_{}", m_uuid))
                         .set_columns("id", "string")
                         .set_values('?', '?')
                         .get_query_string();
        create_statement(m_string_statement, query);
    }

    void initialize_node_info_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder.set_table_name(fmt::format("rocpd_info_node_{}", m_uuid))
                .set_columns("id",
                             "hash",
                             "machine_id",
                             "name",  // V4 addition
                             "system_name",
                             "hostname",
                             "release",
                             "version",
                             "hardware_name",
                             "domain_name")
                .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_node_info_statement, query);
    }

    void initialize_process_info_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder.set_table_name(fmt::format("rocpd_info_process_{}", m_uuid))
                .set_columns("id",
                             "nid",
                             "ppid",
                             "pid",
                             "name",
                             "init",
                             "fini",
                             "start",
                             "end",
                             "command",
                             "environment",
                             "extdata")
                .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_process_info_statement, query);
    }

    void initialize_thread_info_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder.set_table_name(fmt::format("rocpd_info_thread_{}", m_uuid))
                .set_columns(
                    "id", "nid", "ppid", "pid", "tid", "name", "start", "end", "extdata")
                .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_thread_info_statement, query);
    }

    // Schema v4: agent_info with generic_name, without user_name
    void initialize_agent_info_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
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
                             "generic_name",
                             "model_name",
                             "vendor_name",
                             "product_name",
                             "extdata")
                .set_values(
                    '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_agent_info_statement, query);
    }

    void initialize_pmc_info_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
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
                             "qualifier",
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
                            '?',
                            '?')
                .get_query_string();
        create_statement(m_pmc_info_statement, query);
    }

    void initialize_stream_info_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder.set_table_name(fmt::format("rocpd_info_stream_{}", m_uuid))
                .set_columns("id", "nid", "pid", "name", "extdata")
                .set_values('?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_stream_info_statement, query);
    }

    void initialize_queue_info_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder.set_table_name(fmt::format("rocpd_info_queue_{}", m_uuid))
                .set_columns("id", "nid", "pid", "name", "extdata")
                .set_values('?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_queue_info_statement, query);
    }

    void initialize_kernel_symbol_info_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
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
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
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

    // v4: track has extended columns
    void initialize_track_info_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto query = query_builder.set_table_name(fmt::format("rocpd_track_{}", m_uuid))
                         .set_columns("id",
                                      "nid",
                                      "ppid",
                                      "pid",
                                      "tid",
                                      "agent_id",
                                      "queue_id",
                                      "stream_id",
                                      "name_id",
                                      "extdata")
                         .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
                         .get_query_string();
        create_statement(m_track_info_statement, query);
    }

    // NEW in v4: rocpd_timestamp
    void initialize_timestamp_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder.set_table_name(fmt::format("rocpd_timestamp_{}", m_uuid))
                .set_columns("id", "value", "phase", "track_id")
                .set_values('?', '?', '?', '?')
                .get_query_string();
        create_statement(m_timestamp_statement, query);
    }

    // NEW in v4: rocpd_info_category
    void initialize_category_info_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder.set_table_name(fmt::format("rocpd_info_category_{}", m_uuid))
                .set_columns("id", "name", "extdata")
                .set_values('?', '?', '?')
                .get_query_string();
        create_statement(m_category_info_statement, query);
    }

    // NEW in v4: rocpd_info_address_range
    void initialize_address_range_info_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder
                .set_table_name(fmt::format("rocpd_info_address_range_{}", m_uuid))
                .set_columns("id",
                             "nid",
                             "pid",
                             "address_base",
                             "address_low",
                             "address_high",
                             "extdata")
                .set_values('?', '?', '?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_address_range_info_statement, query);
    }

    // NEW in v4: rocpd_info_source_code
    void initialize_source_code_info_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder.set_table_name(fmt::format("rocpd_info_source_code_{}", m_uuid))
                .set_columns("id",
                             "nid",
                             "pid",
                             "address_id",
                             "file",
                             "line_number",
                             "lines",
                             "instructions",
                             "extdata")
                .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_source_code_info_statement, query);
    }

    // NEW in v4: rocpd_info_pc
    void initialize_pc_info_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto query = query_builder.set_table_name(fmt::format("rocpd_info_pc_{}", m_uuid))
                         .set_columns("id",
                                      "nid",
                                      "pid",
                                      "function",
                                      "address_id",
                                      "file",
                                      "line",
                                      "extdata")
                         .set_values('?', '?', '?', '?', '?', '?', '?', '?')
                         .get_query_string();
        create_statement(m_pc_info_statement, query);
    }

    // v4: event WITHOUT call_stack and line_info
    void initialize_event_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto query = query_builder.set_table_name(fmt::format("rocpd_event_{}", m_uuid))
                         .set_columns("id",
                                      "category_id",
                                      "stack_id",
                                      "parent_stack_id",
                                      "correlation_id",
                                      "extdata")
                         .set_values('?', '?', '?', '?', '?', '?')
                         .get_query_string();
        create_statement(m_event_statement, query);
    }

    void initialize_arg_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder.set_table_name(fmt::format("rocpd_arg_{}", m_uuid))
                .set_columns(
                    "id", "event_id", "position", "type", "name", "value", "extdata")
                .set_values('?', '?', '?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_arg_statement, query);
    }

    // NEW in v4: rocpd_call_stack
    void initialize_call_stack_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder.set_table_name(fmt::format("rocpd_call_stack_{}", m_uuid))
                .set_columns("id", "event_id", "pc_id", "depth", "extdata")
                .set_values('?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_call_stack_statement, query);
    }

    // NEW in v4: rocpd_line_info
    void initialize_line_info_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder.set_table_name(fmt::format("rocpd_line_info_{}", m_uuid))
                .set_columns("id", "event_id", "source_code_id", "pc_id", "extdata")
                .set_values('?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_line_info_statement, query);
    }

    void initialize_pmc_event_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder.set_table_name(fmt::format("rocpd_pmc_event_{}", m_uuid))
                .set_columns("id", "event_id", "pmc_id", "value", "extdata")
                .set_values('?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_pmc_event_statement, query);
    }

    // v4: region uses track_id, start_id, end_id
    void initialize_region_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto query = query_builder.set_table_name(fmt::format("rocpd_region_{}", m_uuid))
                         .set_columns("id",
                                      "track_id",
                                      "name_id",
                                      "start_id",
                                      "end_id",
                                      "event_id",
                                      "extdata")
                         .set_values('?', '?', '?', '?', '?', '?', '?')
                         .get_query_string();
        create_statement(m_region_statement, query);
    }

    // v4: sample uses name_id, timestamp_id
    void initialize_sample_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder.set_table_name(fmt::format("rocpd_sample_{}", m_uuid))
                .set_columns(
                    "id", "track_id", "name_id", "timestamp_id", "event_id", "extdata")
                .set_values('?', '?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_sample_statement, query);
    }

    // v4: kernel_dispatch uses track_id, start_id, end_id
    void initialize_kernel_dispatch_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder.set_table_name(fmt::format("rocpd_kernel_dispatch_{}", m_uuid))
                .set_columns("id",
                             "track_id",
                             "kernel_id",
                             "dispatch_id",
                             "start_id",
                             "end_id",
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
                            '?')
                .get_query_string();
        create_statement(m_kernel_dispatch_statement, query);
    }

    // v4: memory_copy uses track_id, start_id, end_id
    void initialize_memory_copy_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder.set_table_name(fmt::format("rocpd_memory_copy_{}", m_uuid))
                .set_columns("id",
                             "track_id",
                             "start_id",
                             "end_id",
                             "name_id",
                             "dst_agent_id",
                             "dst_address",
                             "src_agent_id",
                             "src_address",
                             "size",
                             "region_name_id",
                             "event_id",
                             "extdata")
                .set_values(
                    '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
                .get_query_string();
        create_statement(m_memory_copy_statement, query);
    }

    // v4: memory_allocate uses track_id, start_id, end_id, name_id
    void initialize_memory_alloc_statement()
    {
        rocpdsna::queries::insert::table_insert_query query_builder;
        auto                                          query =
            query_builder.set_table_name(fmt::format("rocpd_memory_allocate_{}", m_uuid))
                .set_columns("id",
                             "track_id",
                             "type",
                             "level",
                             "start_id",
                             "end_id",
                             "name_id",
                             "address",
                             "size",
                             "region_name_id",
                             "event_id",
                             "extdata")
                .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
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
    timestamp_statement_func_t          m_timestamp_statement;
    category_info_statement_func_t      m_category_info_statement;
    address_range_info_statement_func_t m_address_range_info_statement;
    source_code_info_statement_func_t   m_source_code_info_statement;
    pc_info_statement_func_t            m_pc_info_statement;
    event_statement_func_t              m_event_statement;
    arg_statement_func_t                m_arg_statement;
    call_stack_statement_func_t         m_call_stack_statement;
    line_info_statement_func_t          m_line_info_statement;
    pmc_event_statement_func_t          m_pmc_event_statement;
    region_statement_func_t             m_region_statement;
    sample_statement_func_t             m_sample_statement;
    kernel_dispatch_statement_func_t    m_kernel_dispatch_statement;
    memory_copy_statement_func_t        m_memory_copy_statement;
    memory_alloc_statement_func_t       m_memory_alloc_statement;
};

}  // namespace rocpdsna::data_storage::schema_v4
