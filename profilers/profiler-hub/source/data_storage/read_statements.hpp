// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/sqlite_backend.hpp"

#include "profiler-hub/reader_types.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "queries/select/table_select_query.hpp"

namespace profiler_hub::data_storage::schema_v3
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

// ----- Event detail result structs -----

struct region_detail_result
{
    size_t                id{};
    size_t                start{};
    size_t                end{};
    std::optional<size_t> name_id;
    std::optional<size_t> event_id;
    size_t                nid{};
    std::optional<size_t> pid;
    std::optional<size_t> tid;
    std::string           extdata;
};

struct kernel_dispatch_detail_result
{
    size_t                id{};
    size_t                dispatch_id{};
    size_t                start{};
    size_t                end{};
    std::optional<size_t> kernel_id;
    std::optional<size_t> private_segment_size;
    std::optional<size_t> group_segment_size;
    size_t                workgroup_size_x{};
    size_t                workgroup_size_y{};
    size_t                workgroup_size_z{};
    size_t                grid_size_x{};
    size_t                grid_size_y{};
    size_t                grid_size_z{};
    std::optional<size_t> region_name_id;
    std::optional<size_t> event_id;
    size_t                nid{};
    std::optional<size_t> pid;
    std::optional<size_t> tid;
    std::string           extdata;
};

struct memory_copy_detail_result
{
    size_t                id{};
    size_t                start{};
    size_t                end{};
    std::optional<size_t> name_id;
    std::optional<size_t> dst_agent_id;
    std::optional<size_t> dst_address;
    std::optional<size_t> src_agent_id;
    std::optional<size_t> src_address;
    size_t                size{};
    std::optional<size_t> region_name_id;
    std::optional<size_t> event_id;
    size_t                nid{};
    std::optional<size_t> pid;
    std::optional<size_t> tid;
    std::string           extdata;
};

struct memory_alloc_detail_result
{
    size_t                     id{};
    std::optional<std::string> type;
    std::optional<std::string> level;
    size_t                     start{};
    size_t                     end{};
    std::optional<size_t>      address;
    size_t                     size{};
    std::optional<size_t>      event_id;
    size_t                     nid{};
    std::optional<size_t>      pid;
    std::optional<size_t>      tid;
    std::string                extdata;
};

struct event_detail_result
{
    size_t                id{};
    std::optional<size_t> category_id;
    std::optional<size_t> stack_id;
    std::optional<size_t> parent_stack_id;
    std::optional<size_t> correlation_id;
    std::string           call_stack;
    std::string           line_info;
    std::string           extdata;
};

struct arg_detail_result
{
    size_t      position{};
    std::string type;
    std::string name;
    std::string value;
    std::string extdata;
};

/// Lightweight result for resolving event metadata from event-specific tables.
/// JOINs event-specific table with rocpd_event to get both event_id and event metadata.
struct event_id_result
{
    std::optional<size_t> event_id;
    std::optional<size_t> category_id;
    std::optional<size_t> stack_id;
    std::optional<size_t> parent_stack_id;
    std::optional<size_t> correlation_id;
    std::string           call_stack;
    std::string           line_info;
    std::string           event_extdata;
};

struct count_result
{
    size_t count{};
};

struct time_range_result
{
    std::optional<size_t> min_start;
    std::optional<size_t> max_end;
};

struct read_statements
{
    explicit read_statements(std::shared_ptr<sqlite_backend> backend, std::string uuid)
    : m_backend{ std::move(backend) }
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

        initialize_detail_statements();
        initialize_event_id_statements();
        initialize_correlated_event_statements();
        initialize_count_statements();
        initialize_time_range_statements();
    }
    read_statements()                                  = delete;
    read_statements(const read_statements&)            = delete;
    read_statements(read_statements&&)                 = delete;
    read_statements& operator=(const read_statements&) = delete;
    read_statements& operator=(read_statements&&)      = delete;
    virtual ~read_statements()                         = default;

    using string_statement_func_t =
        std::function<sqlite_backend::result_set<string_result>()>;

    using node_info_statement_func_t =
        std::function<sqlite_backend::result_set<node_info_result>()>;

    using process_info_statement_func_t =
        std::function<sqlite_backend::result_set<process_info_result>()>;

    using stream_info_statement_func_t =
        std::function<sqlite_backend::result_set<stream_info_result>()>;

    using queue_info_statement_func_t =
        std::function<sqlite_backend::result_set<queue_info_result>()>;

    using thread_info_statement_func_t =
        std::function<sqlite_backend::result_set<thread_info_result>()>;

    using agent_info_statement_func_t =
        std::function<sqlite_backend::result_set<agent_info_result>()>;

    using track_info_statement_func_t =
        std::function<sqlite_backend::result_set<track_info_result>()>;

    using kernel_symbol_info_statement_func_t =
        std::function<sqlite_backend::result_set<kernel_symbol_info_result>()>;

    using code_object_info_statement_func_t =
        std::function<sqlite_backend::result_set<code_object_info_result>()>;

    using pmc_info_statement_func_t =
        std::function<sqlite_backend::result_set<pmc_info_result>()>;

    using timeline_event_statement_func_t =
        std::function<sqlite_backend::result_set<timeline_event_result>()>;

    using timeline_event_time_filtered_func_t =
        std::function<sqlite_backend::result_set<timeline_event_result>(size_t, size_t)>;

    using timeline_event_track_filtered_func_t = std::function<sqlite_backend::result_set<
        timeline_event_result>(size_t, size_t, size_t, size_t)>;

    using timeline_event_track_and_time_filtered_func_t =
        std::function<sqlite_backend::result_set<
            timeline_event_result>(size_t, size_t, size_t, size_t, size_t, size_t)>;

    // Detail statement func types (parameterized by id)
    using region_detail_func_t =
        std::function<sqlite_backend::result_set<region_detail_result>(size_t)>;
    using kernel_dispatch_detail_func_t =
        std::function<sqlite_backend::result_set<kernel_dispatch_detail_result>(size_t)>;
    using memory_copy_detail_func_t =
        std::function<sqlite_backend::result_set<memory_copy_detail_result>(size_t)>;
    using memory_alloc_detail_func_t =
        std::function<sqlite_backend::result_set<memory_alloc_detail_result>(size_t)>;
    using event_detail_func_t =
        std::function<sqlite_backend::result_set<event_detail_result>(size_t)>;
    using arg_detail_func_t =
        std::function<sqlite_backend::result_set<arg_detail_result>(size_t)>;
    using event_id_func_t =
        std::function<sqlite_backend::result_set<event_id_result>(size_t)>;
    using count_func_t = std::function<sqlite_backend::result_set<count_result>()>;
    using count_time_filtered_func_t =
        std::function<sqlite_backend::result_set<count_result>(size_t, size_t)>;
    using time_range_func_t =
        std::function<sqlite_backend::result_set<time_range_result>()>;

    // Correlated events: bind (stack_id, excluded_event_id)
    using correlated_event_func_t =
        std::function<sqlite_backend::result_set<timeline_event_result>(size_t, size_t)>;

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

    // Detail query accessors
    [[nodiscard]] const region_detail_func_t& region_detail() const
    {
        return m_region_detail;
    }
    [[nodiscard]] const kernel_dispatch_detail_func_t& kernel_dispatch_detail() const
    {
        return m_kernel_dispatch_detail;
    }
    [[nodiscard]] const memory_copy_detail_func_t& memory_copy_detail() const
    {
        return m_memory_copy_detail;
    }
    [[nodiscard]] const memory_alloc_detail_func_t& memory_alloc_detail() const
    {
        return m_memory_alloc_detail;
    }
    [[nodiscard]] const event_detail_func_t& event_detail() const
    {
        return m_event_detail;
    }
    [[nodiscard]] const arg_detail_func_t& arg_detail() const { return m_arg_detail; }

    // Event ID resolution accessors (one per event type)
    [[nodiscard]] const event_id_func_t& region_event_id() const
    {
        return m_region_event_id;
    }
    [[nodiscard]] const event_id_func_t& kernel_dispatch_event_id() const
    {
        return m_kernel_dispatch_event_id;
    }
    [[nodiscard]] const event_id_func_t& memory_copy_event_id() const
    {
        return m_memory_copy_event_id;
    }
    [[nodiscard]] const event_id_func_t& memory_alloc_event_id() const
    {
        return m_memory_alloc_event_id;
    }

    // Correlated event accessors
    struct correlated_event_statement_set
    {
        correlated_event_func_t region;
        correlated_event_func_t kernel_dispatch;
        correlated_event_func_t memory_copy;
        correlated_event_func_t memory_allocate;
    };

    [[nodiscard]] const correlated_event_statement_set& correlated_event_statements()
        const
    {
        return m_correlated_event_statements;
    }

    // Count and time range accessors
    [[nodiscard]] const count_func_t& region_count() const { return m_region_count; }
    [[nodiscard]] const count_func_t& kernel_dispatch_count() const
    {
        return m_kernel_dispatch_count;
    }
    [[nodiscard]] const count_func_t& memory_copy_count() const
    {
        return m_memory_copy_count;
    }
    [[nodiscard]] const count_func_t& memory_alloc_count() const
    {
        return m_memory_alloc_count;
    }
    [[nodiscard]] const count_time_filtered_func_t& region_count_time_filtered() const
    {
        return m_region_count_time_filtered;
    }
    [[nodiscard]] const count_time_filtered_func_t& kernel_dispatch_count_time_filtered()
        const
    {
        return m_kernel_dispatch_count_time_filtered;
    }
    [[nodiscard]] const count_time_filtered_func_t& memory_copy_count_time_filtered()
        const
    {
        return m_memory_copy_count_time_filtered;
    }
    [[nodiscard]] const count_time_filtered_func_t& memory_alloc_count_time_filtered()
        const
    {
        return m_memory_alloc_count_time_filtered;
    }
    [[nodiscard]] const time_range_func_t& region_time_range() const
    {
        return m_region_time_range;
    }
    [[nodiscard]] const time_range_func_t& kernel_dispatch_time_range() const
    {
        return m_kernel_dispatch_time_range;
    }
    [[nodiscard]] const time_range_func_t& memory_copy_time_range() const
    {
        return m_memory_copy_time_range;
    }
    [[nodiscard]] const time_range_func_t& memory_alloc_time_range() const
    {
        return m_memory_alloc_time_range;
    }

private:
    void initialize_string_statement()
    {
        const auto uuid = m_backend->get_uuid();

        queries::select::table_select_query query_builder = {};
        const auto                          query = query_builder.select("id", "string")
                               .from(fmt::format("rocpd_string_{}", uuid))
                               .get_query_string();

        m_string_statement = m_backend->create_read_statement_executor<string_result>(
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
            m_backend->create_read_statement_executor<node_info_result>(
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
            m_backend->create_read_statement_executor<process_info_result>(
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
            m_backend->create_read_statement_executor<stream_info_result>(
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
            m_backend->create_read_statement_executor<queue_info_result>(
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
            m_backend->create_read_statement_executor<thread_info_result>(
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
            m_backend->create_read_statement_executor<agent_info_result>(
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
            m_backend->create_read_statement_executor<track_info_result>(
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
            m_backend->create_read_statement_executor<kernel_symbol_info_result>(
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
            m_backend->create_read_statement_executor<code_object_info_result>(
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

        m_pmc_info_statement = m_backend->create_read_statement_executor<pmc_info_result>(
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

    template <typename JoinBuilder>
    void initialize_timeline_event_variants(JoinBuilder&                  base,
                                            std::string_view              alias,
                                            timeline_event_statement_set& out)
    {
        const auto a = std::string(alias);

        out.base = m_backend->create_read_statement_executor<timeline_event_result>(
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
            m_backend->create_read_statement_executor<timeline_event_result,
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

        out.track_filtered = m_backend->create_read_statement_executor<
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

        out.track_and_time_filtered = m_backend->create_read_statement_executor<
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

    void initialize_detail_statements()
    {
        // Region detail by id
        auto region_q = queries::select::table_select_query{}
                            .select("id",
                                    "start",
                                    "end",
                                    "name_id",
                                    "event_id",
                                    "nid",
                                    "pid",
                                    "tid",
                                    "extdata")
                            .from(fmt::format("rocpd_region_{}", m_uuid))
                            .where("id = ?")
                            .get_query_string();

        m_region_detail = m_backend->create_read_statement_executor<region_detail_result,
                                                                    bind_types<size_t>>(
            region_q,
            &region_detail_result::id,
            &region_detail_result::start,
            &region_detail_result::end,
            &region_detail_result::name_id,
            &region_detail_result::event_id,
            &region_detail_result::nid,
            &region_detail_result::pid,
            &region_detail_result::tid,
            &region_detail_result::extdata);

        // Kernel dispatch detail by id
        auto kd_q = queries::select::table_select_query{}
                        .select("id",
                                "dispatch_id",
                                "start",
                                "end",
                                "kernel_id",
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
                                "nid",
                                "pid",
                                "tid",
                                "extdata")
                        .from(fmt::format("rocpd_kernel_dispatch_{}", m_uuid))
                        .where("id = ?")
                        .get_query_string();

        m_kernel_dispatch_detail =
            m_backend->create_read_statement_executor<kernel_dispatch_detail_result,
                                                      bind_types<size_t>>(
                kd_q,
                &kernel_dispatch_detail_result::id,
                &kernel_dispatch_detail_result::dispatch_id,
                &kernel_dispatch_detail_result::start,
                &kernel_dispatch_detail_result::end,
                &kernel_dispatch_detail_result::kernel_id,
                &kernel_dispatch_detail_result::private_segment_size,
                &kernel_dispatch_detail_result::group_segment_size,
                &kernel_dispatch_detail_result::workgroup_size_x,
                &kernel_dispatch_detail_result::workgroup_size_y,
                &kernel_dispatch_detail_result::workgroup_size_z,
                &kernel_dispatch_detail_result::grid_size_x,
                &kernel_dispatch_detail_result::grid_size_y,
                &kernel_dispatch_detail_result::grid_size_z,
                &kernel_dispatch_detail_result::region_name_id,
                &kernel_dispatch_detail_result::event_id,
                &kernel_dispatch_detail_result::nid,
                &kernel_dispatch_detail_result::pid,
                &kernel_dispatch_detail_result::tid,
                &kernel_dispatch_detail_result::extdata);

        // Memory copy detail by id
        auto mc_q = queries::select::table_select_query{}
                        .select("id",
                                "start",
                                "end",
                                "name_id",
                                "dst_agent_id",
                                "dst_address",
                                "src_agent_id",
                                "src_address",
                                "size",
                                "region_name_id",
                                "event_id",
                                "nid",
                                "pid",
                                "tid",
                                "extdata")
                        .from(fmt::format("rocpd_memory_copy_{}", m_uuid))
                        .where("id = ?")
                        .get_query_string();

        m_memory_copy_detail =
            m_backend->create_read_statement_executor<memory_copy_detail_result,
                                                      bind_types<size_t>>(
                mc_q,
                &memory_copy_detail_result::id,
                &memory_copy_detail_result::start,
                &memory_copy_detail_result::end,
                &memory_copy_detail_result::name_id,
                &memory_copy_detail_result::dst_agent_id,
                &memory_copy_detail_result::dst_address,
                &memory_copy_detail_result::src_agent_id,
                &memory_copy_detail_result::src_address,
                &memory_copy_detail_result::size,
                &memory_copy_detail_result::region_name_id,
                &memory_copy_detail_result::event_id,
                &memory_copy_detail_result::nid,
                &memory_copy_detail_result::pid,
                &memory_copy_detail_result::tid,
                &memory_copy_detail_result::extdata);

        // Memory alloc detail by id
        auto ma_q = queries::select::table_select_query{}
                        .select("id",
                                "type",
                                "level",
                                "start",
                                "end",
                                "address",
                                "size",
                                "event_id",
                                "nid",
                                "pid",
                                "tid",
                                "extdata")
                        .from(fmt::format("rocpd_memory_allocate_{}", m_uuid))
                        .where("id = ?")
                        .get_query_string();

        m_memory_alloc_detail =
            m_backend->create_read_statement_executor<memory_alloc_detail_result,
                                                      bind_types<size_t>>(
                ma_q,
                &memory_alloc_detail_result::id,
                &memory_alloc_detail_result::type,
                &memory_alloc_detail_result::level,
                &memory_alloc_detail_result::start,
                &memory_alloc_detail_result::end,
                &memory_alloc_detail_result::address,
                &memory_alloc_detail_result::size,
                &memory_alloc_detail_result::event_id,
                &memory_alloc_detail_result::nid,
                &memory_alloc_detail_result::pid,
                &memory_alloc_detail_result::tid,
                &memory_alloc_detail_result::extdata);

        // Event detail by id (from rocpd_event)
        auto ev_q = queries::select::table_select_query{}
                        .select("id",
                                "category_id",
                                "stack_id",
                                "parent_stack_id",
                                "correlation_id",
                                "call_stack",
                                "line_info",
                                "extdata")
                        .from(fmt::format("rocpd_event_{}", m_uuid))
                        .where("id = ?")
                        .get_query_string();

        m_event_detail =
            m_backend
                ->create_read_statement_executor<event_detail_result, bind_types<size_t>>(
                    ev_q,
                    &event_detail_result::id,
                    &event_detail_result::category_id,
                    &event_detail_result::stack_id,
                    &event_detail_result::parent_stack_id,
                    &event_detail_result::correlation_id,
                    &event_detail_result::call_stack,
                    &event_detail_result::line_info,
                    &event_detail_result::extdata);

        // Arg detail by event_id
        auto arg_q = queries::select::table_select_query{}
                         .select("position", "type", "name", "value", "extdata")
                         .from(fmt::format("rocpd_arg_{}", m_uuid))
                         .where("event_id = ?")
                         .order_by("position")
                         .get_query_string();

        m_arg_detail =
            m_backend
                ->create_read_statement_executor<arg_detail_result, bind_types<size_t>>(
                    arg_q,
                    &arg_detail_result::position,
                    &arg_detail_result::type,
                    &arg_detail_result::name,
                    &arg_detail_result::value,
                    &arg_detail_result::extdata);
    }

    void initialize_event_id_statements()
    {
        auto make_event_id_stmt = [&](const std::string& table) {
            auto q =
                fmt::format("SELECT E.id, E.category_id, E.stack_id, E.parent_stack_id, "
                            "E.correlation_id, E.call_stack, E.line_info, E.extdata "
                            "FROM {} T INNER JOIN rocpd_event_{} E ON T.event_id = E.id "
                            "WHERE T.id = ?",
                            fmt::format("{}_{}", table, m_uuid),
                            m_uuid);

            return m_backend
                ->create_read_statement_executor<event_id_result, bind_types<size_t>>(
                    q,
                    &event_id_result::event_id,
                    &event_id_result::category_id,
                    &event_id_result::stack_id,
                    &event_id_result::parent_stack_id,
                    &event_id_result::correlation_id,
                    &event_id_result::call_stack,
                    &event_id_result::line_info,
                    &event_id_result::event_extdata);
        };

        m_region_event_id          = make_event_id_stmt("rocpd_region");
        m_kernel_dispatch_event_id = make_event_id_stmt("rocpd_kernel_dispatch");
        m_memory_copy_event_id     = make_event_id_stmt("rocpd_memory_copy");
        m_memory_alloc_event_id    = make_event_id_stmt("rocpd_memory_allocate");
    }

    void initialize_correlated_event_statements()
    {
        auto make_correlated_stmt = [&](const std::string& table,
                                        const std::string& alias,
                                        const std::string& display_name_col) {
            auto q =
                fmt::format("SELECT {a}.id, {a}.start, {a}.end, {dn}, E.category_id, "
                            "{a}.nid, {a}.pid, {a}.tid, S.track_id "
                            "FROM {t} {a} "
                            "INNER JOIN rocpd_event_{u} E ON {a}.event_id = E.id "
                            "LEFT JOIN rocpd_sample_{u} S ON S.event_id = {a}.event_id "
                            "WHERE E.stack_id = ? AND E.id != ?",
                            fmt::arg("a", alias),
                            fmt::arg("t", fmt::format("{}_{}", table, m_uuid)),
                            fmt::arg("u", m_uuid),
                            fmt::arg("dn", display_name_col));

            return m_backend->create_read_statement_executor<timeline_event_result,
                                                             bind_types<size_t, size_t>>(
                q,
                &timeline_event_result::id,
                &timeline_event_result::start_timestamp,
                &timeline_event_result::end_timestamp,
                &timeline_event_result::display_name_id,
                &timeline_event_result::category_id,
                &timeline_event_result::nid,
                &timeline_event_result::pid,
                &timeline_event_result::tid,
                &timeline_event_result::track_id);
        };

        m_correlated_event_statements.region =
            make_correlated_stmt("rocpd_region", "R", "R.name_id");
        m_correlated_event_statements.kernel_dispatch =
            make_correlated_stmt("rocpd_kernel_dispatch", "K", "K.region_name_id");
        m_correlated_event_statements.memory_copy =
            make_correlated_stmt("rocpd_memory_copy", "MC", "MC.region_name_id");
        m_correlated_event_statements.memory_allocate =
            make_correlated_stmt("rocpd_memory_allocate", "MA", "E.category_id");
    }

    void initialize_count_statements()
    {
        auto make_count_stmt = [&](const std::string& table) {
            auto q = fmt::format("SELECT COUNT(*) FROM {}_{}", table, m_uuid);
            return m_backend->create_read_statement_executor<count_result>(
                q, &count_result::count);
        };
        auto make_count_time_filtered_stmt = [&](const std::string& table) {
            auto q = fmt::format(
                "SELECT COUNT(*) FROM {}_{} WHERE start <= ? AND \"end\" >= ?",
                table,
                m_uuid);
            return m_backend->create_read_statement_executor<count_result,
                                                             bind_types<size_t, size_t>>(
                q, &count_result::count);
        };

        m_region_count          = make_count_stmt("rocpd_region");
        m_kernel_dispatch_count = make_count_stmt("rocpd_kernel_dispatch");
        m_memory_copy_count     = make_count_stmt("rocpd_memory_copy");
        m_memory_alloc_count    = make_count_stmt("rocpd_memory_allocate");

        m_region_count_time_filtered = make_count_time_filtered_stmt("rocpd_region");
        m_kernel_dispatch_count_time_filtered =
            make_count_time_filtered_stmt("rocpd_kernel_dispatch");
        m_memory_copy_count_time_filtered =
            make_count_time_filtered_stmt("rocpd_memory_copy");
        m_memory_alloc_count_time_filtered =
            make_count_time_filtered_stmt("rocpd_memory_allocate");
    }

    void initialize_time_range_statements()
    {
        auto make_time_range_stmt = [&](const std::string& table) {
            auto q = fmt::format("SELECT MIN(start), MAX(end) FROM {}_{}", table, m_uuid);
            return m_backend->create_read_statement_executor<time_range_result>(
                q, &time_range_result::min_start, &time_range_result::max_end);
        };

        m_region_time_range          = make_time_range_stmt("rocpd_region");
        m_kernel_dispatch_time_range = make_time_range_stmt("rocpd_kernel_dispatch");
        m_memory_copy_time_range     = make_time_range_stmt("rocpd_memory_copy");
        m_memory_alloc_time_range    = make_time_range_stmt("rocpd_memory_allocate");
    }

    std::shared_ptr<sqlite_backend> m_backend;
    std::string                     m_uuid;

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

    // Detail query members
    region_detail_func_t          m_region_detail;
    kernel_dispatch_detail_func_t m_kernel_dispatch_detail;
    memory_copy_detail_func_t     m_memory_copy_detail;
    memory_alloc_detail_func_t    m_memory_alloc_detail;
    event_detail_func_t           m_event_detail;
    arg_detail_func_t             m_arg_detail;

    // Event ID resolution (per event type)
    event_id_func_t m_region_event_id;
    event_id_func_t m_kernel_dispatch_event_id;
    event_id_func_t m_memory_copy_event_id;
    event_id_func_t m_memory_alloc_event_id;

    // Correlated events
    correlated_event_statement_set m_correlated_event_statements;

    // Count statements
    count_func_t m_region_count;
    count_func_t m_kernel_dispatch_count;
    count_func_t m_memory_copy_count;
    count_func_t m_memory_alloc_count;

    count_time_filtered_func_t m_region_count_time_filtered;
    count_time_filtered_func_t m_kernel_dispatch_count_time_filtered;
    count_time_filtered_func_t m_memory_copy_count_time_filtered;
    count_time_filtered_func_t m_memory_alloc_count_time_filtered;

    // Time range statements
    time_range_func_t m_region_time_range;
    time_range_func_t m_kernel_dispatch_time_range;
    time_range_func_t m_memory_copy_time_range;
    time_range_func_t m_memory_alloc_time_range;
};
}  // namespace profiler_hub::data_storage::schema_v3
