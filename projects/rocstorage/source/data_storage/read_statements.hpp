// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "database.hpp"

#include "rocstorage/writer_types.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "queries/select/table_select_query.hpp"

namespace rocstorage::data_storage::schema_v3
{

struct process_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<size_t>      ppid;
    std::optional<size_t>      init;
    std::optional<size_t>      fini;
    std::optional<size_t>      start;
    std::optional<size_t>      end;
    std::optional<const char*> command;
    const char*                environment{};
    const char*                extdata{};
};

struct string_result
{
    size_t      id{};
    const char* value;
};

struct stream_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<const char*> name;
    const char*                extdata{};
};

struct queue_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<const char*> name;
    const char*                extdata{};
};

struct thread_info_result
{
    size_t                     id{};
    size_t                     nid{};
    std::optional<size_t>      ppid;
    size_t                     pid{};
    size_t                     tid{};
    std::optional<const char*> name;
    std::optional<size_t>      start;
    std::optional<size_t>      end;
    const char*                extdata{};
};

struct agent_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<const char*> type;
    std::optional<size_t>      absolute_index;
    std::optional<size_t>      logical_index;
    std::optional<size_t>      type_index;
    std::optional<size_t>      uuid;
    std::optional<const char*> name;
    std::optional<const char*> model_name;
    std::optional<const char*> vendor_name;
    std::optional<const char*> product_name;
    std::optional<const char*> user_name;
    const char*                extdata{};
};

struct track_info_result
{
    size_t                id{};
    size_t                nid{};
    std::optional<size_t> pid;
    std::optional<size_t> tid;
    std::optional<size_t> name_id;
    const char*           extdata{};
};

struct kernel_symbol_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    size_t                     code_object_id{};
    std::optional<const char*> kernel_name;
    std::optional<const char*> display_name;
    std::optional<size_t>      kernel_object;
    std::optional<size_t>      kernarg_segment_size;
    std::optional<size_t>      kernarg_segment_alignment;
    std::optional<size_t>      group_segment_size;
    std::optional<size_t>      private_segment_size;
    std::optional<size_t>      sgpr_count;
    std::optional<size_t>      arch_vgpr_count;
    std::optional<size_t>      accum_vgpr_count;
    const char*                extdata{};
};

struct code_object_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<size_t>      agent_id;
    std::optional<const char*> uri;
    std::optional<size_t>      load_base;
    std::optional<size_t>      load_size;
    std::optional<size_t>      load_delta;
    std::optional<const char*> storage_type;
    const char*                extdata{};
};

struct pmc_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<size_t>      agent_id;
    std::optional<const char*> target_arch;
    std::optional<size_t>      event_code;
    std::optional<size_t>      instance_id;
    const char*                name{};
    const char*                symbol{};
    std::optional<const char*> description;
    std::optional<const char*> long_description;
    std::optional<const char*> component;
    std::optional<const char*> units;
    std::optional<const char*> value_type;
    std::optional<const char*> block;
    std::optional<const char*> expression;
    std::optional<size_t>      is_constant;
    std::optional<size_t>      is_derived;
    const char*                extdata{};
};

struct read_statements
{
    explicit read_statements(std::shared_ptr<database> db, std::string uuid)
    : m_database{ std::move(db) }
    , m_uuid{ std::move(uuid) }
    {
        initialize_string_statement();
        initialize_node_info_statement();
        initialize_process_info_statement();
        initialize_stream_info_statement();
        initialize_queue_info_statement();
        initialize_thread_info_statement();
        initialize_agent_info_statement();
        initialize_track_info_statement();
        initialize_kernel_symbol_info_statement();
        initialize_code_object_info_statement();
        initialize_pmc_info_statement();
    }
    read_statements()                                  = delete;
    read_statements(const read_statements&)            = delete;
    read_statements(read_statements&&)                 = delete;
    read_statements& operator=(const read_statements&) = delete;
    read_statements& operator=(read_statements&&)      = delete;
    virtual ~read_statements()                         = default;

    using string_statement_func_t = std::function<statement_result<string_result>()>;

    using node_info_statement_func_t =
        std::function<statement_result<writer_types::node_info_t>()>;

    using process_info_statement_func_t =
        std::function<statement_result<process_info_result>()>;

    using stream_info_statement_func_t =
        std::function<statement_result<stream_info_result>()>;

    using queue_info_statement_func_t =
        std::function<statement_result<queue_info_result>()>;

    using thread_info_statement_func_t =
        std::function<statement_result<thread_info_result>()>;

    using agent_info_statement_func_t =
        std::function<statement_result<agent_info_result>()>;

    using track_info_statement_func_t =
        std::function<statement_result<track_info_result>()>;

    using kernel_symbol_info_statement_func_t =
        std::function<statement_result<kernel_symbol_info_result>()>;

    using code_object_info_statement_func_t =
        std::function<statement_result<code_object_info_result>()>;

    using pmc_info_statement_func_t = std::function<statement_result<pmc_info_result>()>;

    [[nodiscard]] string_statement_func_t string_statement() const
    {
        return m_string_statement;
    }

    [[nodiscard]] node_info_statement_func_t node_info_statement() const
    {
        return m_node_info_statement;
    }

    [[nodiscard]] process_info_statement_func_t process_info_statement() const
    {
        return m_process_info_statement;
    }

    [[nodiscard]] stream_info_statement_func_t stream_info_statement() const
    {
        return m_stream_info_statement;
    }

    [[nodiscard]] queue_info_statement_func_t queue_info_statement() const
    {
        return m_queue_info_statement;
    }

    [[nodiscard]] thread_info_statement_func_t thread_info_statement() const
    {
        return m_thread_info_statement;
    }

    [[nodiscard]] agent_info_statement_func_t agent_info_statement() const
    {
        return m_agent_info_statement;
    }

    [[nodiscard]] track_info_statement_func_t track_info_statement() const
    {
        return m_track_info_statement;
    }

    [[nodiscard]] kernel_symbol_info_statement_func_t kernel_symbol_info_statement() const
    {
        return m_kernel_symbol_info_statement;
    }

    [[nodiscard]] code_object_info_statement_func_t code_object_info_statement() const
    {
        return m_code_object_info_statement;
    }

    [[nodiscard]] pmc_info_statement_func_t pmc_info_statement() const
    {
        return m_pmc_info_statement;
    }

private:
    void initialize_string_statement()
    {
        const auto uuid = m_database->get_uuid();

        queries::select::table_select_query query_builder = {};
        const auto                          query = query_builder.select("id", "string")
                               .from(fmt::format("rocpd_string_{}", uuid))
                               .get_query_string();

        m_string_statement = m_database->create_read_statement_executor<string_result>(
            query, &string_result::id, &string_result::value);
    }

    void initialize_node_info_statement()
    {
        auto query = queries::select::table_select_query{}
                         .select("id",
                                 "hash",
                                 "machine_id",
                                 "system_name",
                                 "hostname",
                                 "release",
                                 "version",
                                 "hardware_name",
                                 "domain_name")
                         .from(fmt::format("rocpd_info_node_{}", m_uuid))
                         .get_query_string();

        m_node_info_statement =
            m_database->create_read_statement_executor<writer_types::node_info_t>(
                query,
                &writer_types::node_info_t::node_id,
                &writer_types::node_info_t::hash,
                &writer_types::node_info_t::machine_id,
                &writer_types::node_info_t::system_name,
                &writer_types::node_info_t::hostname,
                &writer_types::node_info_t::release,
                &writer_types::node_info_t::version,
                &writer_types::node_info_t::hardware_name,
                &writer_types::node_info_t::domain_name);
    }

    void initialize_process_info_statement()
    {
        auto query = queries::select::table_select_query{}
                         .select("id",
                                 "nid",
                                 "pid",
                                 "ppid",
                                 "init",
                                 "fini",
                                 "start",
                                 "end",
                                 "command",
                                 "environment",
                                 "extdata")
                         .from(fmt::format("rocpd_info_process_{}", m_uuid))
                         .get_query_string();

        m_process_info_statement =
            m_database->create_read_statement_executor<process_info_result>(
                query,
                &process_info_result::id,
                &process_info_result::nid,
                &process_info_result::pid,
                &process_info_result::ppid,
                &process_info_result::init,
                &process_info_result::fini,
                &process_info_result::start,
                &process_info_result::end,
                &process_info_result::command,
                &process_info_result::environment,
                &process_info_result::extdata);
    }

    void initialize_stream_info_statement()
    {
        auto query = queries::select::table_select_query{}
                         .select("id", "nid", "pid", "name", "extdata")
                         .from(fmt::format("rocpd_info_stream_{}", m_uuid))
                         .get_query_string();

        m_stream_info_statement =
            m_database->create_read_statement_executor<stream_info_result>(
                query,
                &stream_info_result::id,
                &stream_info_result::nid,
                &stream_info_result::pid,
                &stream_info_result::name,
                &stream_info_result::extdata);
    }

    void initialize_queue_info_statement()
    {
        auto query = queries::select::table_select_query{}
                         .select("id", "nid", "pid", "name", "extdata")
                         .from(fmt::format("rocpd_info_queue_{}", m_uuid))
                         .get_query_string();

        m_queue_info_statement =
            m_database->create_read_statement_executor<queue_info_result>(
                query,
                &queue_info_result::id,
                &queue_info_result::nid,
                &queue_info_result::pid,
                &queue_info_result::name,
                &queue_info_result::extdata);
    }

    void initialize_thread_info_statement()
    {
        auto query =
            queries::select::table_select_query{}
                .select(
                    "id", "nid", "ppid", "pid", "tid", "name", "start", "end", "extdata")
                .from(fmt::format("rocpd_info_thread_{}", m_uuid))
                .get_query_string();

        m_thread_info_statement =
            m_database->create_read_statement_executor<thread_info_result>(
                query,
                &thread_info_result::id,
                &thread_info_result::nid,
                &thread_info_result::ppid,
                &thread_info_result::pid,
                &thread_info_result::tid,
                &thread_info_result::name,
                &thread_info_result::start,
                &thread_info_result::end,
                &thread_info_result::extdata);
    }

    void initialize_agent_info_statement()
    {
        auto query = queries::select::table_select_query{}
                         .select("id",
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
                         .from(fmt::format("rocpd_info_agent_{}", m_uuid))
                         .get_query_string();

        m_agent_info_statement =
            m_database->create_read_statement_executor<agent_info_result>(
                query,
                &agent_info_result::id,
                &agent_info_result::nid,
                &agent_info_result::pid,
                &agent_info_result::type,
                &agent_info_result::absolute_index,
                &agent_info_result::logical_index,
                &agent_info_result::type_index,
                &agent_info_result::uuid,
                &agent_info_result::name,
                &agent_info_result::model_name,
                &agent_info_result::vendor_name,
                &agent_info_result::product_name,
                &agent_info_result::user_name,
                &agent_info_result::extdata);
    }

    void initialize_track_info_statement()
    {
        auto query = queries::select::table_select_query{}
                         .select("id", "nid", "pid", "tid", "name_id", "extdata")
                         .from(fmt::format("rocpd_track_{}", m_uuid))
                         .get_query_string();

        m_track_info_statement =
            m_database->create_read_statement_executor<track_info_result>(
                query,
                &track_info_result::id,
                &track_info_result::nid,
                &track_info_result::pid,
                &track_info_result::tid,
                &track_info_result::name_id,
                &track_info_result::extdata);
    }

    void initialize_kernel_symbol_info_statement()
    {
        auto query = queries::select::table_select_query{}
                         .select("id",
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
                         .from(fmt::format("rocpd_info_kernel_symbol_{}", m_uuid))
                         .get_query_string();

        m_kernel_symbol_info_statement =
            m_database->create_read_statement_executor<kernel_symbol_info_result>(
                query,
                &kernel_symbol_info_result::id,
                &kernel_symbol_info_result::nid,
                &kernel_symbol_info_result::pid,
                &kernel_symbol_info_result::code_object_id,
                &kernel_symbol_info_result::kernel_name,
                &kernel_symbol_info_result::display_name,
                &kernel_symbol_info_result::kernel_object,
                &kernel_symbol_info_result::kernarg_segment_size,
                &kernel_symbol_info_result::kernarg_segment_alignment,
                &kernel_symbol_info_result::group_segment_size,
                &kernel_symbol_info_result::private_segment_size,
                &kernel_symbol_info_result::sgpr_count,
                &kernel_symbol_info_result::arch_vgpr_count,
                &kernel_symbol_info_result::accum_vgpr_count,
                &kernel_symbol_info_result::extdata);
    }

    void initialize_code_object_info_statement()
    {
        auto query = queries::select::table_select_query{}
                         .select("id",
                                 "nid",
                                 "pid",
                                 "agent_id",
                                 "uri",
                                 "load_base",
                                 "load_size",
                                 "load_delta",
                                 "storage_type",
                                 "extdata")
                         .from(fmt::format("rocpd_info_code_object_{}", m_uuid))
                         .get_query_string();

        m_code_object_info_statement =
            m_database->create_read_statement_executor<code_object_info_result>(
                query,
                &code_object_info_result::id,
                &code_object_info_result::nid,
                &code_object_info_result::pid,
                &code_object_info_result::agent_id,
                &code_object_info_result::uri,
                &code_object_info_result::load_base,
                &code_object_info_result::load_size,
                &code_object_info_result::load_delta,
                &code_object_info_result::storage_type,
                &code_object_info_result::extdata);
    }

    void initialize_pmc_info_statement()
    {
        auto query = queries::select::table_select_query{}
                         .select("id",
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
                         .from(fmt::format("rocpd_info_pmc_{}", m_uuid))
                         .get_query_string();

        m_pmc_info_statement =
            m_database->create_read_statement_executor<pmc_info_result>(
                query,
                &pmc_info_result::id,
                &pmc_info_result::nid,
                &pmc_info_result::pid,
                &pmc_info_result::agent_id,
                &pmc_info_result::target_arch,
                &pmc_info_result::event_code,
                &pmc_info_result::instance_id,
                &pmc_info_result::name,
                &pmc_info_result::symbol,
                &pmc_info_result::description,
                &pmc_info_result::long_description,
                &pmc_info_result::component,
                &pmc_info_result::units,
                &pmc_info_result::value_type,
                &pmc_info_result::block,
                &pmc_info_result::expression,
                &pmc_info_result::is_constant,
                &pmc_info_result::is_derived,
                &pmc_info_result::extdata);
    }

    std::shared_ptr<database> m_database;
    std::string               m_uuid;

    string_statement_func_t             m_string_statement;
    node_info_statement_func_t          m_node_info_statement;
    process_info_statement_func_t       m_process_info_statement;
    stream_info_statement_func_t        m_stream_info_statement;
    queue_info_statement_func_t         m_queue_info_statement;
    thread_info_statement_func_t        m_thread_info_statement;
    agent_info_statement_func_t         m_agent_info_statement;
    track_info_statement_func_t         m_track_info_statement;
    kernel_symbol_info_statement_func_t m_kernel_symbol_info_statement;
    code_object_info_statement_func_t   m_code_object_info_statement;
    pmc_info_statement_func_t           m_pmc_info_statement;
};
}  // namespace rocstorage::data_storage::schema_v3
