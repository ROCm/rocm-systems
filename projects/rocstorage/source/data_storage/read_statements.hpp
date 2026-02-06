// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "database.hpp"

#include "rocstorage/reader_types.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "queries/select/table_select_query.hpp"

namespace rocstorage::data_storage::schema_v3
{

struct node_info_result
{
    size_t      node_id;
    size_t      hash;
    std::string machine_id;
    std::string system_name;
    std::string hostname;
    std::string release;
    std::string version;
    std::string hardware_name;
    std::string domain_name;
};

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
    std::optional<std::string> command;
    std::string                environment;
    std::string                extdata;
};

struct string_result
{
    size_t      id{};
    std::string value;
};

struct stream_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<std::string> name;
    std::string                extdata;
};

struct queue_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<std::string> name;
    std::string                extdata;
};

struct thread_info_result
{
    size_t                     id{};
    size_t                     nid{};
    std::optional<size_t>      ppid;
    size_t                     pid{};
    size_t                     tid{};
    std::optional<std::string> name;
    std::optional<size_t>      start;
    std::optional<size_t>      end;
    std::string                extdata;
};

struct agent_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<std::string> type;
    std::optional<size_t>      absolute_index;
    std::optional<size_t>      logical_index;
    std::optional<size_t>      type_index;
    std::optional<size_t>      uuid;
    std::optional<std::string> name;
    std::optional<std::string> model_name;
    std::optional<std::string> vendor_name;
    std::optional<std::string> product_name;
    std::optional<std::string> user_name;
    std::string                extdata;
};

struct track_info_result
{
    size_t                id{};
    size_t                nid{};
    std::optional<size_t> pid;
    std::optional<size_t> tid;
    std::optional<size_t> name_id;
    std::string           extdata;
};

struct kernel_symbol_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    size_t                     code_object_id{};
    std::optional<std::string> kernel_name;
    std::optional<std::string> display_name;
    std::optional<size_t>      kernel_object;
    std::optional<size_t>      kernarg_segment_size;
    std::optional<size_t>      kernarg_segment_alignment;
    std::optional<size_t>      group_segment_size;
    std::optional<size_t>      private_segment_size;
    std::optional<size_t>      sgpr_count;
    std::optional<size_t>      arch_vgpr_count;
    std::optional<size_t>      accum_vgpr_count;
    std::string                extdata;
};

struct code_object_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<size_t>      agent_id;
    std::optional<std::string> uri;
    std::optional<size_t>      load_base;
    std::optional<size_t>      load_size;
    std::optional<size_t>      load_delta;
    std::optional<std::string> storage_type;
    std::string                extdata;
};

struct pmc_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<size_t>      agent_id;
    std::optional<std::string> target_arch;
    std::optional<size_t>      event_code;
    std::optional<size_t>      instance_id;
    std::string                name{};
    std::string                symbol{};
    std::optional<std::string> description;
    std::optional<std::string> long_description;
    std::optional<std::string> component;
    std::optional<std::string> units;
    std::optional<std::string> value_type;
    std::optional<std::string> block;
    std::optional<std::string> expression;
    std::optional<size_t>      is_constant;
    std::optional<size_t>      is_derived;
    std::string                extdata;
};

struct timeline_event_result
{
    size_t id{};

    size_t start_timestamp{};
    size_t end_timestamp{};

    std::optional<size_t> display_name_id;
    std::optional<size_t> category_id;

    size_t                nid{};
    std::optional<size_t> pid;
    std::optional<size_t> tid;
    std::optional<size_t> track_id;
};

struct sample_timeline_event_result
{
    size_t                id{};
    size_t                timestamp{};
    std::optional<size_t> category_id;
    size_t                track_id{};
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

        initialize_region_timeline_event_statements();
        initialize_kernel_dispatch_timeline_event_statements();
        initialize_memory_allocate_timeline_event_statements();
        initialize_memory_copy_timeline_event_statements();
    }
    read_statements()                                  = delete;
    read_statements(const read_statements&)            = delete;
    read_statements(read_statements&&)                 = delete;
    read_statements& operator=(const read_statements&) = delete;
    read_statements& operator=(read_statements&&)      = delete;
    virtual ~read_statements()                         = default;

    using string_statement_func_t = std::function<statement_result<string_result>()>;

    using node_info_statement_func_t =
        std::function<statement_result<node_info_result>()>;

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

    using timeline_event_statement_func_t =
        std::function<statement_result<timeline_event_result>()>;

    using timeline_event_time_filtered_func_t =
        std::function<statement_result<timeline_event_result>(size_t, size_t)>;

    using timeline_event_track_filtered_func_t = std::function<
        statement_result<timeline_event_result>(size_t, size_t, size_t, size_t)>;

    using timeline_event_track_and_time_filtered_func_t = std::function<statement_result<
        timeline_event_result>(size_t, size_t, size_t, size_t, size_t, size_t)>;

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

    struct timeline_event_statement_set
    {
        timeline_event_statement_func_t               base;
        timeline_event_time_filtered_func_t           time_filtered;
        timeline_event_track_filtered_func_t          track_filtered;
        timeline_event_track_and_time_filtered_func_t track_and_time_filtered;
    };

    [[nodiscard]] const timeline_event_statement_set& region_statements() const
    {
        return m_region_statements;
    }

    [[nodiscard]] const timeline_event_statement_set& kernel_dispatch_statements() const
    {
        return m_kernel_dispatch_statements;
    }

    [[nodiscard]] const timeline_event_statement_set& memory_allocate_statements() const
    {
        return m_memory_allocate_statements;
    }

    [[nodiscard]] const timeline_event_statement_set& memory_copy_statements() const
    {
        return m_memory_copy_statements;
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
            m_database->create_read_statement_executor<node_info_result>(
                query,
                &node_info_result::node_id,
                &node_info_result::hash,
                &node_info_result::machine_id,
                &node_info_result::system_name,
                &node_info_result::hostname,
                &node_info_result::release,
                &node_info_result::version,
                &node_info_result::hardware_name,
                &node_info_result::domain_name);
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

    // Helper: initialize all 4 variants for a timeline event type using
    // query builder reusability (base query reused for each WHERE variant)
    template <typename JoinBuilder>
    void initialize_timeline_event_variants(JoinBuilder&                  base,
                                            std::string_view              alias,
                                            timeline_event_statement_set& out)
    {
        const auto a = std::string(alias);

        out.base = m_database->create_read_statement_executor<timeline_event_result>(
            base.get_query_string(),
            &timeline_event_result::id,
            &timeline_event_result::start_timestamp,
            &timeline_event_result::end_timestamp,
            &timeline_event_result::display_name_id,
            &timeline_event_result::category_id,
            &timeline_event_result::nid,
            &timeline_event_result::pid,
            &timeline_event_result::tid,
            &timeline_event_result::track_id);

        out.time_filtered =
            m_database->create_read_statement_executor<timeline_event_result,
                                                       bind_types<size_t, size_t>>(
                base.where(a + ".start <= ?")
                    .and_where(a + ".end >= ?")
                    .get_query_string(),
                &timeline_event_result::id,
                &timeline_event_result::start_timestamp,
                &timeline_event_result::end_timestamp,
                &timeline_event_result::display_name_id,
                &timeline_event_result::category_id,
                &timeline_event_result::nid,
                &timeline_event_result::pid,
                &timeline_event_result::tid,
                &timeline_event_result::track_id);

        const auto track_where = "(" + a + ".nid = ? AND " + a + ".pid = ? AND " + a +
                                 ".tid = ?) OR S.track_id = ?";

        out.track_filtered = m_database->create_read_statement_executor<
            timeline_event_result,
            bind_types<size_t, size_t, size_t, size_t>>(
            base.where(track_where).get_query_string(),
            &timeline_event_result::id,
            &timeline_event_result::start_timestamp,
            &timeline_event_result::end_timestamp,
            &timeline_event_result::display_name_id,
            &timeline_event_result::category_id,
            &timeline_event_result::nid,
            &timeline_event_result::pid,
            &timeline_event_result::tid,
            &timeline_event_result::track_id);

        out.track_and_time_filtered = m_database->create_read_statement_executor<
            timeline_event_result,
            bind_types<size_t, size_t, size_t, size_t, size_t, size_t>>(
            base.where("(" + track_where + ")")
                .and_where(a + ".start <= ?")
                .and_where(a + ".end >= ?")
                .get_query_string(),
            &timeline_event_result::id,
            &timeline_event_result::start_timestamp,
            &timeline_event_result::end_timestamp,
            &timeline_event_result::display_name_id,
            &timeline_event_result::category_id,
            &timeline_event_result::nid,
            &timeline_event_result::pid,
            &timeline_event_result::tid,
            &timeline_event_result::track_id);
    }

    void initialize_region_timeline_event_statements()
    {
        queries::select::table_select_query query;
        auto&                               base = query
                         .select("R.id",
                                 "R.start",
                                 "R.end",
                                 "R.name_id",
                                 "E.category_id",
                                 "R.nid",
                                 "R.pid",
                                 "R.tid",
                                 "S.track_id")
                         .from(fmt::format("rocpd_region_{}", m_uuid), "R")
                         .inner_join("rocpd_event", "E", "R.event_id = E.id")
                         .left_join("rocpd_sample", "S", "S.event_id = R.event_id");

        initialize_timeline_event_variants(base, "R", m_region_statements);
    }

    void initialize_kernel_dispatch_timeline_event_statements()
    {
        queries::select::table_select_query query;
        auto&                               base = query
                         .select("K.id",
                                 "K.start",
                                 "K.end",
                                 "K.region_name_id",
                                 "E.category_id",
                                 "K.nid",
                                 "K.pid",
                                 "K.tid",
                                 "S.track_id")
                         .from(fmt::format("rocpd_kernel_dispatch_{}", m_uuid), "K")
                         .inner_join("rocpd_event", "E", "E.id = K.event_id")
                         .left_join("rocpd_sample", "S", "S.event_id = K.event_id");

        initialize_timeline_event_variants(base, "K", m_kernel_dispatch_statements);
    }

    void initialize_memory_allocate_timeline_event_statements()
    {
        queries::select::table_select_query query;
        auto&                               base = query
                         .select("MA.id",
                                 "MA.start",
                                 "MA.end",
                                 "E.category_id",
                                 "E.category_id",
                                 "MA.nid",
                                 "MA.pid",
                                 "MA.tid",
                                 "S.track_id")
                         .from(fmt::format("rocpd_memory_allocate_{}", m_uuid), "MA")
                         .inner_join("rocpd_event", "E", "E.id = MA.event_id")
                         .left_join("rocpd_sample", "S", "S.event_id = MA.event_id");

        initialize_timeline_event_variants(base, "MA", m_memory_allocate_statements);
    }

    void initialize_memory_copy_timeline_event_statements()
    {
        queries::select::table_select_query query;
        auto&                               base = query
                         .select("MC.id",
                                 "MC.start",
                                 "MC.end",
                                 "MC.region_name_id",
                                 "E.category_id",
                                 "MC.nid",
                                 "MC.pid",
                                 "MC.tid",
                                 "S.track_id")
                         .from(fmt::format("rocpd_memory_copy_{}", m_uuid), "MC")
                         .inner_join("rocpd_event", "E", "MC.event_id = E.id")
                         .left_join("rocpd_sample", "S", "S.event_id = MC.event_id");

        initialize_timeline_event_variants(base, "MC", m_memory_copy_statements);
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

    timeline_event_statement_set m_region_statements;
    timeline_event_statement_set m_kernel_dispatch_statements;
    timeline_event_statement_set m_memory_allocate_statements;
    timeline_event_statement_set m_memory_copy_statements;
};
}  // namespace rocstorage::data_storage::schema_v3
