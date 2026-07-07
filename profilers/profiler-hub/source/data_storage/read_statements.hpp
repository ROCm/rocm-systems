// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/sqlite_backend.hpp"
#include "read_statements_base.hpp"

#include "../json_serializers.hpp"
#include "profiler-hub/reader_types.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "queries/select/table_select_query.hpp"

namespace profiler_hub::data_storage::schema_v3
{

// All result structs, func typedefs, and set structs are defined in
// read_statements_base.hpp (namespace profiler_hub::data_storage) and are
// inherited here. This is the v3 backend: same SQL as before, now overriding
// the abstract read_statements_base accessors.
struct read_statements : public read_statements_base
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

        initialize_track_synthesis_statements();
        initialize_interval_track_statements();
        initialize_track_stats_statements();
        initialize_scalar_track_statements();
        initialize_scalar_detail_statement();
        initialize_flow_statements();
    }
    read_statements()                                  = delete;
    read_statements(const read_statements&)            = delete;
    read_statements(read_statements&&)                 = delete;
    read_statements& operator=(const read_statements&) = delete;
    read_statements& operator=(read_statements&&)      = delete;
    ~read_statements() override                        = default;

    [[nodiscard]] string_statement_func_t string_statement() const override
    {
        return m_string_statement;
    }

    [[nodiscard]] node_info_statement_func_t node_info_statement() const override
    {
        return m_node_info_statement;
    }

    [[nodiscard]] process_info_statement_func_t process_info_statement() const override
    {
        return m_process_info_statement;
    }

    [[nodiscard]] stream_info_statement_func_t stream_info_statement() const override
    {
        return m_stream_info_statement;
    }

    [[nodiscard]] queue_info_statement_func_t queue_info_statement() const override
    {
        return m_queue_info_statement;
    }

    [[nodiscard]] thread_info_statement_func_t thread_info_statement() const override
    {
        return m_thread_info_statement;
    }

    [[nodiscard]] agent_info_statement_func_t agent_info_statement() const override
    {
        return m_agent_info_statement;
    }

    [[nodiscard]] track_info_statement_func_t track_info_statement() const override
    {
        return m_track_info_statement;
    }

    [[nodiscard]] kernel_symbol_info_statement_func_t kernel_symbol_info_statement()
        const override
    {
        return m_kernel_symbol_info_statement;
    }

    [[nodiscard]] code_object_info_statement_func_t code_object_info_statement()
        const override
    {
        return m_code_object_info_statement;
    }

    [[nodiscard]] pmc_info_statement_func_t pmc_info_statement() const override
    {
        return m_pmc_info_statement;
    }

    [[nodiscard]] const timeline_event_statement_set& region_statements() const override
    {
        return m_region_statements;
    }

    [[nodiscard]] const timeline_event_statement_set& kernel_dispatch_statements()
        const override
    {
        return m_kernel_dispatch_statements;
    }

    [[nodiscard]] const timeline_event_statement_set& memory_allocate_statements()
        const override
    {
        return m_memory_allocate_statements;
    }

    [[nodiscard]] const timeline_event_statement_set& memory_copy_statements()
        const override
    {
        return m_memory_copy_statements;
    }

    // Detail query accessors
    [[nodiscard]] const region_detail_func_t& region_detail() const override
    {
        return m_region_detail;
    }
    [[nodiscard]] const kernel_dispatch_detail_func_t& kernel_dispatch_detail()
        const override
    {
        return m_kernel_dispatch_detail;
    }
    [[nodiscard]] const memory_copy_detail_func_t& memory_copy_detail() const override
    {
        return m_memory_copy_detail;
    }
    [[nodiscard]] const memory_alloc_detail_func_t& memory_alloc_detail() const override
    {
        return m_memory_alloc_detail;
    }
    [[nodiscard]] const event_detail_func_t& event_detail() const override
    {
        return m_event_detail;
    }
    [[nodiscard]] const arg_detail_func_t& arg_detail() const override
    {
        return m_arg_detail;
    }

    // Event ID resolution accessors (one per event type)
    [[nodiscard]] const event_id_func_t& region_event_id() const override
    {
        return m_region_event_id;
    }
    [[nodiscard]] const event_id_func_t& kernel_dispatch_event_id() const override
    {
        return m_kernel_dispatch_event_id;
    }
    [[nodiscard]] const event_id_func_t& memory_copy_event_id() const override
    {
        return m_memory_copy_event_id;
    }
    [[nodiscard]] const event_id_func_t& memory_alloc_event_id() const override
    {
        return m_memory_alloc_event_id;
    }

    [[nodiscard]] const correlated_event_statement_set& correlated_event_statements()
        const override
    {
        return m_correlated_event_statements;
    }

    // Count and time range accessors
    [[nodiscard]] const count_func_t& region_count() const override
    {
        return m_region_count;
    }
    [[nodiscard]] const count_func_t& kernel_dispatch_count() const override
    {
        return m_kernel_dispatch_count;
    }
    [[nodiscard]] const count_func_t& memory_copy_count() const override
    {
        return m_memory_copy_count;
    }
    [[nodiscard]] const count_func_t& memory_alloc_count() const override
    {
        return m_memory_alloc_count;
    }
    [[nodiscard]] const count_time_filtered_func_t& region_count_time_filtered()
        const override
    {
        return m_region_count_time_filtered;
    }
    [[nodiscard]] const count_time_filtered_func_t& kernel_dispatch_count_time_filtered()
        const override
    {
        return m_kernel_dispatch_count_time_filtered;
    }
    [[nodiscard]] const count_time_filtered_func_t& memory_copy_count_time_filtered()
        const override
    {
        return m_memory_copy_count_time_filtered;
    }
    [[nodiscard]] const count_time_filtered_func_t& memory_alloc_count_time_filtered()
        const override
    {
        return m_memory_alloc_count_time_filtered;
    }
    [[nodiscard]] const time_range_func_t& region_time_range() const override
    {
        return m_region_time_range;
    }
    [[nodiscard]] const time_range_func_t& kernel_dispatch_time_range() const override
    {
        return m_kernel_dispatch_time_range;
    }
    [[nodiscard]] const time_range_func_t& memory_copy_time_range() const override
    {
        return m_memory_copy_time_range;
    }
    [[nodiscard]] const time_range_func_t& memory_alloc_time_range() const override
    {
        return m_memory_alloc_time_range;
    }

    // Track synthesis accessors
    [[nodiscard]] const distinct_gpu_queue_func_t& distinct_gpu_queue_tracks()
        const override
    {
        return m_distinct_gpu_queue_tracks;
    }
    [[nodiscard]] const distinct_dma_func_t& distinct_dma_tracks() const override
    {
        return m_distinct_dma_tracks;
    }
    [[nodiscard]] const distinct_region_func_t& distinct_region_tracks() const override
    {
        return m_distinct_region_tracks;
    }
    [[nodiscard]] const sample_track_id_func_t& distinct_sample_track_ids() const override
    {
        return m_distinct_sample_track_ids;
    }
    [[nodiscard]] const max_track_id_func_t& max_track_id() const override
    {
        return m_max_track_id;
    }
    [[nodiscard]] const counter_track_name_func_t& counter_track_names() const override
    {
        return m_counter_track_names;
    }

    // Interval-track accessors
    [[nodiscard]] const interval_track_3_func_t& region_interval_track_main()
        const override
    {
        return m_region_interval_track_main;
    }
    [[nodiscard]] const interval_track_3_func_t& region_interval_track_sample()
        const override
    {
        return m_region_interval_track_sample;
    }
    [[nodiscard]] const interval_track_4_func_t& kernel_dispatch_interval_track()
        const override
    {
        return m_kernel_dispatch_interval_track;
    }
    [[nodiscard]] const interval_track_4_func_t& memory_copy_interval_qs() const override
    {
        return m_memory_copy_interval_qs;
    }
    [[nodiscard]] const interval_track_3_func_t& memory_copy_interval_q_only()
        const override
    {
        return m_memory_copy_interval_q_only;
    }
    [[nodiscard]] const interval_track_3_func_t& memory_copy_interval_s_only()
        const override
    {
        return m_memory_copy_interval_s_only;
    }
    [[nodiscard]] const interval_track_2_func_t& memory_copy_interval_neither()
        const override
    {
        return m_memory_copy_interval_neither;
    }

    // Track-stats accessors (aggregates matching the interval-track scoping above)
    [[nodiscard]] const stats_track_3_func_t& region_stats_track_main() const override
    {
        return m_region_stats_track_main;
    }
    [[nodiscard]] const stats_track_3_func_t& region_stats_track_sample() const override
    {
        return m_region_stats_track_sample;
    }
    [[nodiscard]] const stats_track_4_func_t& kernel_dispatch_stats_track() const override
    {
        return m_kernel_dispatch_stats_track;
    }
    [[nodiscard]] const stats_track_4_func_t& memory_copy_stats_qs() const override
    {
        return m_memory_copy_stats_qs;
    }
    [[nodiscard]] const stats_track_3_func_t& memory_copy_stats_q_only() const override
    {
        return m_memory_copy_stats_q_only;
    }
    [[nodiscard]] const stats_track_3_func_t& memory_copy_stats_s_only() const override
    {
        return m_memory_copy_stats_s_only;
    }
    [[nodiscard]] const stats_track_2_func_t& memory_copy_stats_neither() const override
    {
        return m_memory_copy_stats_neither;
    }
    [[nodiscard]] const stats_track_1_func_t& scalar_stats() const override
    {
        return m_scalar_stats;
    }

    // Scalar-track accessors
    [[nodiscard]] const scalar_track_func_t& scalar_track() const override
    {
        return m_scalar_track;
    }
    [[nodiscard]] const scalar_detail_func_t& scalar_detail() const override
    {
        return m_scalar_detail;
    }
    [[nodiscard]] const scalar_detail_func_t& pmc_event_detail() const override
    {
        return m_pmc_event_detail;
    }

    [[nodiscard]] const flow_statement_set& region_to_kernel_dispatch_flows()
        const override
    {
        return m_region_to_kernel_dispatch_flows;
    }
    [[nodiscard]] const flow_statement_set& region_to_memory_copy_flows() const override
    {
        return m_region_to_memory_copy_flows;
    }
    [[nodiscard]] const flow_statement_set& region_to_memory_allocate_flows()
        const override
    {
        return m_region_to_memory_allocate_flows;
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
            &timeline_event_result::category_name,
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
                &timeline_event_result::category_name,
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
            &timeline_event_result::category_name,
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
            &timeline_event_result::category_name,
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
                                 "CS.string",
                                 "R.nid",
                                 "R.pid",
                                 "R.tid",
                                 "S.track_id")
                         .from(fmt::format("rocpd_region_{}", m_uuid), "R")
                         .inner_join("rocpd_event", "E", "R.event_id = E.id")
                         .left_join("rocpd_string", "CS", "CS.id = E.category_id")
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
                                 "CS.string",
                                 "K.nid",
                                 "K.pid",
                                 "K.tid",
                                 "S.track_id")
                         .from(fmt::format("rocpd_kernel_dispatch_{}", m_uuid), "K")
                         .inner_join("rocpd_event", "E", "E.id = K.event_id")
                         .left_join("rocpd_string", "CS", "CS.id = E.category_id")
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
                                 "CS.string",
                                 "MA.nid",
                                 "MA.pid",
                                 "MA.tid",
                                 "S.track_id")
                         .from(fmt::format("rocpd_memory_allocate_{}", m_uuid), "MA")
                         .inner_join("rocpd_event", "E", "E.id = MA.event_id")
                         .left_join("rocpd_string", "CS", "CS.id = E.category_id")
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
                                 "CS.string",
                                 "MC.nid",
                                 "MC.pid",
                                 "MC.tid",
                                 "S.track_id")
                         .from(fmt::format("rocpd_memory_copy_{}", m_uuid), "MC")
                         .inner_join("rocpd_event", "E", "MC.event_id = E.id")
                         .left_join("rocpd_string", "CS", "CS.id = E.category_id")
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
        // v3 stores call_stack / line_info as JSON blob strings on rocpd_event. The
        // raw executor reads those strings; the wrapper deserializes them into the
        // version-neutral event_id_result the reader consumes.
        auto make_event_id_stmt = [&](const std::string& table) -> event_id_func_t {
            auto q = fmt::format(
                "SELECT E.id, CS.string, E.stack_id, E.parent_stack_id, "
                "E.correlation_id, E.call_stack, E.line_info, E.extdata "
                "FROM {t} T INNER JOIN rocpd_event_{u} E ON T.event_id = E.id "
                "LEFT JOIN rocpd_string_{u} CS ON CS.id = E.category_id "
                "WHERE T.id = ?",
                fmt::arg("t", fmt::format("{}_{}", table, m_uuid)),
                fmt::arg("u", m_uuid));

            auto raw_exec = m_backend->create_read_statement_executor<event_id_raw_result,
                                                                      bind_types<size_t>>(
                q,
                &event_id_raw_result::event_id,
                &event_id_raw_result::category_name,
                &event_id_raw_result::stack_id,
                &event_id_raw_result::parent_stack_id,
                &event_id_raw_result::correlation_id,
                &event_id_raw_result::call_stack,
                &event_id_raw_result::line_info,
                &event_id_raw_result::event_extdata);

            return [raw_exec](size_t id) -> std::vector<event_id_result> {
                auto                         raws = raw_exec(id).to_vector();
                std::vector<event_id_result> out;
                out.reserve(raws.size());
                for(auto& r : raws)
                {
                    event_id_result e;
                    e.event_id        = r.event_id;
                    e.category_name   = std::move(r.category_name);
                    e.stack_id        = r.stack_id;
                    e.parent_stack_id = r.parent_stack_id;
                    e.correlation_id  = r.correlation_id;
                    e.call_stack = json_serializers::deserialize_call_stack(r.call_stack);
                    e.line_info =
                        json_serializers::deserialize_source_context(r.line_info);
                    e.event_extdata = std::move(r.event_extdata);
                    out.push_back(std::move(e));
                }
                return out;
            };
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
                fmt::format("SELECT {a}.id, {a}.start, {a}.end, {dn}, CS.string, "
                            "{a}.nid, {a}.pid, {a}.tid, S.track_id "
                            "FROM {t} {a} "
                            "INNER JOIN rocpd_event_{u} E ON {a}.event_id = E.id "
                            "LEFT JOIN rocpd_string_{u} CS ON CS.id = E.category_id "
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
                &timeline_event_result::category_name,
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

    void initialize_track_synthesis_statements()
    {
        const auto& u = m_uuid;

        m_distinct_gpu_queue_tracks =
            m_backend->create_read_statement_executor<distinct_gpu_queue_result>(
                fmt::format("SELECT DISTINCT nid, pid, agent_id, queue_id "
                            "FROM rocpd_kernel_dispatch_{}",
                            u),
                &distinct_gpu_queue_result::nid,
                &distinct_gpu_queue_result::pid,
                &distinct_gpu_queue_result::agent_id,
                &distinct_gpu_queue_result::queue_id);

        m_distinct_dma_tracks =
            m_backend->create_read_statement_executor<distinct_dma_result>(
                fmt::format("SELECT DISTINCT nid, pid, queue_id, stream_id "
                            "FROM rocpd_memory_copy_{}",
                            u),
                &distinct_dma_result::nid,
                &distinct_dma_result::pid,
                &distinct_dma_result::queue_id,
                &distinct_dma_result::stream_id);

        // Region (cpu_thread) tracks are synthesized from rocpd_region rather than
        // rocpd_track (v3 rocpd_track is not a reliable thread registry). Each distinct
        // (nid, pid, tid, is_sample) is a track: is_sample separates region events that
        // have a rocpd_sample (sample) from those that do not (main), matching
        // roc-optiq's region-main / region-sample split.
        m_distinct_region_tracks =
            m_backend->create_read_statement_executor<distinct_region_result>(
                fmt::format("SELECT r.nid, r.pid, r.tid, "
                            "CASE WHEN s.event_id IS NULL THEN 0 ELSE 1 END AS is_sample "
                            "FROM rocpd_region_{u} r "
                            "LEFT JOIN rocpd_sample_{u} s ON s.event_id = r.event_id "
                            "GROUP BY r.nid, r.pid, r.tid, is_sample",
                            fmt::arg("u", u)),
                &distinct_region_result::nid,
                &distinct_region_result::pid,
                &distinct_region_result::tid,
                &distinct_region_result::is_sample);

        m_distinct_sample_track_ids =
            m_backend->create_read_statement_executor<sample_track_id_result>(
                fmt::format("SELECT DISTINCT track_id FROM rocpd_sample_{}", u),
                &sample_track_id_result::track_id);

        m_max_track_id = m_backend->create_read_statement_executor<max_track_id_result>(
            fmt::format("SELECT MAX(id) FROM rocpd_track_{}", u),
            &max_track_id_result::max_id);

        m_counter_track_names =
            m_backend->create_read_statement_executor<counter_track_name_result>(
                fmt::format("SELECT s.track_id, ip.name "
                            "FROM rocpd_sample_{u} s "
                            "JOIN rocpd_pmc_event_{u} pe ON pe.event_id = s.event_id "
                            "JOIN rocpd_info_pmc_{u} ip ON ip.id = pe.pmc_id "
                            "GROUP BY s.track_id",
                            fmt::arg("u", u)),
                &counter_track_name_result::track_id,
                &counter_track_name_result::name);
    }

    void initialize_interval_track_statements()
    {
        const auto& u = m_uuid;

        // cpu_thread: region events keyed on (nid, pid, tid), split main vs. sample.
        // main = regions whose event has no rocpd_sample; sample = regions whose event
        // does. Together they partition the thread's regions, matching the two
        // synthesized region tracks (region_track_kind_t main / sample).
        // Category is resolved in-SQL to its display string (rocpd_string via
        // rocpd_event.category_id), mirroring the timeline-event category_name pattern.
        // rocpd_event / rocpd_string are LEFT JOINed so the returned row set stays
        // identical to the pre-category query (and to get_track_stats' count) — category
        // is purely additive, NULL when an event or category is absent.
        m_region_interval_track_main =
            m_backend->create_read_statement_executor<interval_row_result,
                                                      bind_types<size_t, size_t, size_t>>(
                fmt::format(
                    "SELECT r.id, r.start, r.\"end\", r.name_id, CS.string "
                    "FROM rocpd_region_{u} r "
                    "LEFT JOIN rocpd_sample_{u} s ON s.event_id = r.event_id "
                    "LEFT JOIN rocpd_event_{u} E ON E.id = r.event_id "
                    "LEFT JOIN rocpd_string_{u} CS ON CS.id = E.category_id "
                    "WHERE r.nid = ? AND r.pid = ? AND r.tid = ? AND s.event_id IS NULL "
                    "ORDER BY r.start",
                    fmt::arg("u", u)),
                &interval_row_result::id,
                &interval_row_result::start,
                &interval_row_result::end,
                &interval_row_result::name_ref,
                &interval_row_result::category);

        m_region_interval_track_sample =
            m_backend->create_read_statement_executor<interval_row_result,
                                                      bind_types<size_t, size_t, size_t>>(
                fmt::format(
                    "SELECT DISTINCT r.id, r.start, r.\"end\", r.name_id, CS.string "
                    "FROM rocpd_region_{u} r "
                    "INNER JOIN rocpd_sample_{u} s ON s.event_id = r.event_id "
                    "LEFT JOIN rocpd_event_{u} E ON E.id = r.event_id "
                    "LEFT JOIN rocpd_string_{u} CS ON CS.id = E.category_id "
                    "WHERE r.nid = ? AND r.pid = ? AND r.tid = ? ORDER BY r.start",
                    fmt::arg("u", u)),
                &interval_row_result::id,
                &interval_row_result::start,
                &interval_row_result::end,
                &interval_row_result::name_ref,
                &interval_row_result::category);

        // gpu_queue: kernel dispatches keyed on (nid, pid, agent_id, queue_id)
        m_kernel_dispatch_interval_track = m_backend->create_read_statement_executor<
            interval_row_result,
            bind_types<size_t, size_t, size_t, size_t>>(
            fmt::format("SELECT id, start, \"end\", kernel_id "
                        "FROM rocpd_kernel_dispatch_{} "
                        "WHERE nid = ? AND pid = ? AND agent_id = ? AND queue_id = ? "
                        "ORDER BY start",
                        u),
            &interval_row_result::id,
            &interval_row_result::start,
            &interval_row_result::end,
            &interval_row_result::name_ref);

        // dma: memory copies keyed on (nid, pid, queue_id, stream_id). NULL queue_id /
        // stream_id are distinct group values, so prepare one variant per NULL pattern.
        auto make_mc_interval = [&](const char* qs_clause, auto bind_tag) {
            using bt = decltype(bind_tag);
            return m_backend->create_read_statement_executor<interval_row_result, bt>(
                fmt::format(
                    "SELECT id, start, \"end\", name_id FROM rocpd_memory_copy_{} "
                    "WHERE nid = ? AND pid = ? AND {} ORDER BY start",
                    u,
                    qs_clause),
                &interval_row_result::id,
                &interval_row_result::start,
                &interval_row_result::end,
                &interval_row_result::name_ref);
        };

        m_memory_copy_interval_qs =
            make_mc_interval("queue_id = ? AND stream_id = ?",
                             bind_types<size_t, size_t, size_t, size_t>{});
        m_memory_copy_interval_q_only = make_mc_interval(
            "queue_id = ? AND stream_id IS NULL", bind_types<size_t, size_t, size_t>{});
        m_memory_copy_interval_s_only = make_mc_interval(
            "queue_id IS NULL AND stream_id = ?", bind_types<size_t, size_t, size_t>{});
        m_memory_copy_interval_neither = make_mc_interval(
            "queue_id IS NULL AND stream_id IS NULL", bind_types<size_t, size_t>{});
    }

    void initialize_track_stats_statements()
    {
        const auto& u = m_uuid;

        // MIN(start)/MAX(end)/COUNT over exactly the rows the matching interval-track
        // query returns, so per-track bounds/count agree with a full slice load.

        // cpu_thread region main: regions on (nid,pid,tid) whose event has no sample.
        m_region_stats_track_main = m_backend->create_read_statement_executor<
            track_stats_result,
            bind_types<size_t, size_t, size_t>>(
            fmt::format(
                "SELECT MIN(r.start), MAX(r.\"end\"), COUNT(*) FROM rocpd_region_{u} r "
                "LEFT JOIN rocpd_sample_{u} s ON s.event_id = r.event_id "
                "WHERE r.nid = ? AND r.pid = ? AND r.tid = ? AND s.event_id IS NULL",
                fmt::arg("u", u)),
            &track_stats_result::min_ts,
            &track_stats_result::max_ts,
            &track_stats_result::count);

        // cpu_thread region sample: regions on (nid,pid,tid) whose event has a sample.
        // COUNT(DISTINCT r.id) matches the interval query's SELECT DISTINCT r.id (a
        // region event may have multiple samples).
        m_region_stats_track_sample =
            m_backend->create_read_statement_executor<track_stats_result,
                                                      bind_types<size_t, size_t, size_t>>(
                fmt::format("SELECT MIN(r.start), MAX(r.\"end\"), COUNT(DISTINCT r.id) "
                            "FROM rocpd_region_{u} r "
                            "INNER JOIN rocpd_sample_{u} s ON s.event_id = r.event_id "
                            "WHERE r.nid = ? AND r.pid = ? AND r.tid = ?",
                            fmt::arg("u", u)),
                &track_stats_result::min_ts,
                &track_stats_result::max_ts,
                &track_stats_result::count);

        // gpu_queue: kernel dispatches on (nid,pid,agent_id,queue_id).
        m_kernel_dispatch_stats_track = m_backend->create_read_statement_executor<
            track_stats_result,
            bind_types<size_t, size_t, size_t, size_t>>(
            fmt::format("SELECT MIN(start), MAX(\"end\"), COUNT(*) "
                        "FROM rocpd_kernel_dispatch_{} "
                        "WHERE nid = ? AND pid = ? AND agent_id = ? AND queue_id = ?",
                        u),
            &track_stats_result::min_ts,
            &track_stats_result::max_ts,
            &track_stats_result::count);

        // dma: memory copies on (nid,pid,queue_id,stream_id), one variant per NULL
        // pattern.
        auto make_mc_stats = [&](const char* qs_clause, auto bind_tag) {
            using bt = decltype(bind_tag);
            return m_backend->create_read_statement_executor<track_stats_result, bt>(
                fmt::format("SELECT MIN(start), MAX(\"end\"), COUNT(*) "
                            "FROM rocpd_memory_copy_{} WHERE nid = ? AND pid = ? AND {}",
                            u,
                            qs_clause),
                &track_stats_result::min_ts,
                &track_stats_result::max_ts,
                &track_stats_result::count);
        };

        m_memory_copy_stats_qs =
            make_mc_stats("queue_id = ? AND stream_id = ?",
                          bind_types<size_t, size_t, size_t, size_t>{});
        m_memory_copy_stats_q_only  = make_mc_stats("queue_id = ? AND stream_id IS NULL",
                                                   bind_types<size_t, size_t, size_t>{});
        m_memory_copy_stats_s_only  = make_mc_stats("queue_id IS NULL AND stream_id = ?",
                                                   bind_types<size_t, size_t, size_t>{});
        m_memory_copy_stats_neither = make_mc_stats(
            "queue_id IS NULL AND stream_id IS NULL", bind_types<size_t, size_t>{});

        // counter: samples on track_id joined to their pmc value (matches scalar_track).
        m_scalar_stats =
            m_backend
                ->create_read_statement_executor<track_stats_result, bind_types<size_t>>(
                    fmt::format("SELECT MIN(s.timestamp), MAX(s.timestamp), COUNT(*) "
                                "FROM rocpd_sample_{u} s "
                                "JOIN rocpd_pmc_event_{u} p ON p.event_id = s.event_id "
                                "WHERE s.track_id = ?",
                                fmt::arg("u", u)),
                    &track_stats_result::min_ts,
                    &track_stats_result::max_ts,
                    &track_stats_result::count);
    }

    void initialize_scalar_track_statements()
    {
        const auto& u = m_uuid;

        m_scalar_track =
            m_backend
                ->create_read_statement_executor<scalar_row_result, bind_types<size_t>>(
                    fmt::format("SELECT s.id, s.timestamp, p.value "
                                "FROM rocpd_sample_{u} s "
                                "JOIN rocpd_pmc_event_{u} p ON p.event_id = s.event_id "
                                "WHERE s.track_id = ? ORDER BY s.timestamp",
                                fmt::arg("u", u)),
                    &scalar_row_result::id,
                    &scalar_row_result::timestamp,
                    &scalar_row_result::value);
    }

    void initialize_scalar_detail_statement()
    {
        const auto& u = m_uuid;

        // Keyed on rocpd_sample.id (scalar_event_t::opaque_id).
        m_scalar_detail = m_backend->create_read_statement_executor<scalar_detail_result,
                                                                    bind_types<size_t>>(
            fmt::format("SELECT s.id, s.track_id, s.timestamp, p.value, s.event_id "
                        "FROM rocpd_sample_{u} s "
                        "JOIN rocpd_pmc_event_{u} p ON p.event_id = s.event_id "
                        "WHERE s.id = ?",
                        fmt::arg("u", u)),
            &scalar_detail_result::id,
            &scalar_detail_result::track_id,
            &scalar_detail_result::timestamp,
            &scalar_detail_result::value,
            &scalar_detail_result::event_id);

        // Keyed on rocpd_pmc_event.id.
        m_pmc_event_detail =
            m_backend->create_read_statement_executor<scalar_detail_result,
                                                      bind_types<size_t>>(
                fmt::format("SELECT s.id, s.track_id, s.timestamp, p.value, s.event_id "
                            "FROM rocpd_pmc_event_{u} p "
                            "JOIN rocpd_sample_{u} s ON s.event_id = p.event_id "
                            "WHERE p.id = ?",
                            fmt::arg("u", u)),
                &scalar_detail_result::id,
                &scalar_detail_result::track_id,
                &scalar_detail_result::timestamp,
                &scalar_detail_result::value,
                &scalar_detail_result::event_id);
    }

    void initialize_flow_statements()
    {
        const auto& u = m_uuid;

        // source = CPU region; dest = a GPU-side event sharing the same stack_id.
        auto make_flow_set = [&](const std::string& dest_table,
                                 const std::string& dest_alias) -> flow_statement_set {
            const auto base_sql =
                fmt::format("SELECT R.id, {d}.id "
                            "FROM rocpd_region_{u} R "
                            "JOIN rocpd_event_{u} ER ON R.event_id = ER.id "
                            "JOIN rocpd_event_{u} E{d} "
                            "  ON E{d}.stack_id = ER.stack_id AND E{d}.id != ER.id "
                            "JOIN {dt}_{u} {d} ON {d}.event_id = E{d}.id "
                            "WHERE ER.stack_id IS NOT NULL AND ER.stack_id != 0",
                            fmt::arg("u", u),
                            fmt::arg("d", dest_alias),
                            fmt::arg("dt", dest_table));

            flow_statement_set out;
            out.base = m_backend->create_read_statement_executor<flow_row_result>(
                base_sql, &flow_row_result::source_id, &flow_row_result::dest_id);

            // Optional time window applied to the SOURCE event's start timestamp.
            out.time_filtered =
                m_backend->create_read_statement_executor<flow_row_result,
                                                          bind_types<size_t, size_t>>(
                    base_sql + " AND R.start >= ? AND R.start <= ?",
                    &flow_row_result::source_id,
                    &flow_row_result::dest_id);
            return out;
        };

        m_region_to_kernel_dispatch_flows = make_flow_set("rocpd_kernel_dispatch", "K");
        m_region_to_memory_copy_flows     = make_flow_set("rocpd_memory_copy", "MC");
        m_region_to_memory_allocate_flows = make_flow_set("rocpd_memory_allocate", "MA");
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

    // Track synthesis statements
    distinct_gpu_queue_func_t m_distinct_gpu_queue_tracks;
    distinct_dma_func_t       m_distinct_dma_tracks;
    distinct_region_func_t    m_distinct_region_tracks;
    sample_track_id_func_t    m_distinct_sample_track_ids;
    max_track_id_func_t       m_max_track_id;
    counter_track_name_func_t m_counter_track_names;

    // Interval-track statements
    interval_track_3_func_t m_region_interval_track_main;
    interval_track_3_func_t m_region_interval_track_sample;
    interval_track_4_func_t m_kernel_dispatch_interval_track;
    interval_track_4_func_t m_memory_copy_interval_qs;
    interval_track_3_func_t m_memory_copy_interval_q_only;
    interval_track_3_func_t m_memory_copy_interval_s_only;
    interval_track_2_func_t m_memory_copy_interval_neither;

    // Track-stats statements
    stats_track_3_func_t m_region_stats_track_main;
    stats_track_3_func_t m_region_stats_track_sample;
    stats_track_4_func_t m_kernel_dispatch_stats_track;
    stats_track_4_func_t m_memory_copy_stats_qs;
    stats_track_3_func_t m_memory_copy_stats_q_only;
    stats_track_3_func_t m_memory_copy_stats_s_only;
    stats_track_2_func_t m_memory_copy_stats_neither;
    stats_track_1_func_t m_scalar_stats;

    // Scalar-track statements
    scalar_track_func_t  m_scalar_track;
    scalar_detail_func_t m_scalar_detail;
    scalar_detail_func_t m_pmc_event_detail;

    // Flow statements
    flow_statement_set m_region_to_kernel_dispatch_flows;
    flow_statement_set m_region_to_memory_copy_flows;
    flow_statement_set m_region_to_memory_allocate_flows;
};
}  // namespace profiler_hub::data_storage::schema_v3
