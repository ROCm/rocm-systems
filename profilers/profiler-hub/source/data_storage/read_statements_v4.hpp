// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/sqlite_backend.hpp"
#include "read_statements_base.hpp"

#include "profiler-hub/reader_types.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace profiler_hub::data_storage::schema_v4
{

// v4.0 read backend, implementing the track-scoped reader subset against the v4.0
// rocpd schema.
//
// v4.0 differs from v3 in two structural ways that drive the SQL here:
//   * rocpd_track is the universal identity anchor — every event table carries a
//     single track_id FK, so all track-scoped queries reduce to WHERE track_id = ?.
//   * timestamps are normalized into the rocpd_timestamp spine — interval tables
//     carry start_id/end_id FKs and rocpd_sample carries timestamp_id, so reading
//     an actual time requires a JOIN onto rocpd_timestamp.value.
//
// The timeline-event and count accessors are implemented here against the v4.0
// schema (spine JOINs + rocpd_track + rocpd_info_category). The remaining legacy
// detail/call-stack/correlated/time-range surface is not yet implemented; those
// accessors inherit the default-empty stubs from read_statements_base and the
// reader still guards those paths.
//
// Table naming: this backend reuses the v3 reader convention `rocpd_<name>_<uuid>`
// (underscore separator supplied by the reader). The `{{uuid}}` placeholder in the
// v4.0 DDL template is not substituted here.
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

        initialize_track_synthesis_statements();
        initialize_interval_track_statements();
        initialize_track_stats_statements();
        initialize_scalar_track_statements();
        initialize_scalar_detail_statement();
        initialize_ambiguous_pmc_ids_statement();
        initialize_flow_statements();

        initialize_timeline_event_statements();
        initialize_count_statements();

        initialize_event_id_statements();
        initialize_arg_detail_statement();
        initialize_correlated_event_statements();
        initialize_detail_statements();
        initialize_time_range_statements();
        initialize_summary_statements();
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

    [[nodiscard]] const sample_track_id_func_t& distinct_sample_track_ids() const override
    {
        return m_distinct_sample_track_ids;
    }
    // Stream tracks are the one v4.0 track type NOT read 1:1 from rocpd_track: multiple
    // rocpd_track rows can share a stream_id, so stream tracks are synthesized by
    // distinct (nid, pid, stream_id) with a non-null stream_id.
    [[nodiscard]] const distinct_stream_func_t& distinct_stream_tracks() const override
    {
        return m_distinct_stream_tracks;
    }
    [[nodiscard]] const distinct_memory_func_t& distinct_memory_tracks() const override
    {
        return m_distinct_memory_tracks;
    }
    [[nodiscard]] const memory_alloc_track_ids_func_t& memory_alloc_track_ids()
        const override
    {
        return m_memory_alloc_track_ids;
    }
    [[nodiscard]] const distinct_kd_pmc_func_t& distinct_kd_pmc_tracks() const override
    {
        return m_distinct_kd_pmc_tracks;
    }
    [[nodiscard]] const distinct_mem_activity_func_t& distinct_mem_activity_tracks()
        const override
    {
        return m_distinct_mem_activity_tracks;
    }
    [[nodiscard]] const mem_activity_raw_func_t& mem_activity_raw_track() const override
    {
        return m_mem_activity_raw_track;
    }
    [[nodiscard]] const sample_track_id_func_t& kd_pmc_track_ids() const override
    {
        return m_kd_pmc_track_ids;
    }
    [[nodiscard]] const counter_track_name_func_t& counter_track_names() const override
    {
        return m_counter_track_names;
    }
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
    [[nodiscard]] const ambiguous_pmc_ids_func_t& ambiguous_pmc_ids() const override
    {
        return m_ambiguous_pmc_ids;
    }

    // ----- flow accessors (same stack_id semantics as v3) -----
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
    [[nodiscard]] const flow_statement_set& region_to_region_flows() const override
    {
        return m_region_to_region_flows;
    }
    [[nodiscard]] const flow_statement_set& kernel_dispatch_sibling_flows() const override
    {
        return m_kernel_dispatch_sibling_flows;
    }
    [[nodiscard]] const flow_statement_set& memory_copy_sibling_flows() const override
    {
        return m_memory_copy_sibling_flows;
    }
    [[nodiscard]] const flow_statement_set& memory_allocate_sibling_flows() const override
    {
        return m_memory_allocate_sibling_flows;
    }

    [[nodiscard]] const interval_track_1_func_t& region_interval_track_v4() const override
    {
        return m_region_interval_track_v4;
    }
    [[nodiscard]] const interval_track_1_func_t& kernel_dispatch_interval_track_v4()
        const override
    {
        return m_kernel_dispatch_interval_track_v4;
    }
    [[nodiscard]] const interval_track_1_func_t& memory_copy_interval_track_v4()
        const override
    {
        return m_memory_copy_interval_track_v4;
    }
    [[nodiscard]] const interval_track_1_func_t& memory_alloc_interval_track_v4()
        const override
    {
        return m_memory_alloc_interval_track_v4;
    }
    [[nodiscard]] const interval_track_4_func_t& kd_pmc_interval_track() const override
    {
        return m_kd_pmc_interval_track;
    }
    [[nodiscard]] const interval_track_3_func_t& stream_interval_track() const override
    {
        return m_stream_interval_track;
    }

    [[nodiscard]] const stats_track_1_func_t& region_stats_track_v4() const override
    {
        return m_region_stats_track_v4;
    }
    [[nodiscard]] const stats_track_1_func_t& kernel_dispatch_stats_track_v4()
        const override
    {
        return m_kernel_dispatch_stats_track_v4;
    }
    [[nodiscard]] const stats_track_1_func_t& memory_copy_stats_track_v4() const override
    {
        return m_memory_copy_stats_track_v4;
    }
    [[nodiscard]] const stats_track_1_func_t& memory_alloc_stats_track_v4() const override
    {
        return m_memory_alloc_stats_track_v4;
    }
    [[nodiscard]] const stats_track_4_func_t& kd_pmc_stats_track() const override
    {
        return m_kd_pmc_stats_track;
    }
    [[nodiscard]] const stats_track_3_func_t& stream_stats_track() const override
    {
        return m_stream_stats_track;
    }
    [[nodiscard]] const stats_track_1_func_t& scalar_stats() const override
    {
        return m_scalar_stats;
    }

    // ----- legacy timeline-event accessors -----
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

    // ----- legacy count accessors -----
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

    // ----- legacy event-metadata / property accessors -----
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

    [[nodiscard]] const arg_detail_func_t& arg_detail() const override
    {
        return m_arg_detail;
    }

    [[nodiscard]] const correlated_event_statement_set& correlated_event_statements()
        const override
    {
        return m_correlated_event_statements;
    }

    // ----- legacy per-event detail accessors -----
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

    // ----- legacy per-event-type time-range accessors -----
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

    [[nodiscard]] const summary_func_t& kernel_summary() const override
    {
        return m_kernel_summary;
    }
    [[nodiscard]] const summary_time_filtered_func_t& kernel_summary_time_filtered()
        const override
    {
        return m_kernel_summary_time_filtered;
    }
    [[nodiscard]] const summary_func_t& region_summary() const override
    {
        return m_region_summary;
    }
    [[nodiscard]] const summary_time_filtered_func_t& region_summary_time_filtered()
        const override
    {
        return m_region_summary_time_filtered;
    }

private:
    void initialize_string_statement()
    {
        m_string_statement = m_backend->create_read_statement_executor<string_result>(
            fmt::format("SELECT id, string FROM rocpd_string_{}", m_uuid),
            &string_result::id,
            &string_result::value);
    }

    void initialize_node_info_statement()
    {
        m_node_info_statement =
            m_backend->create_read_statement_executor<node_info_result>(
                fmt::format("SELECT id, hash, machine_id, system_name, hostname, "
                            "release, version, hardware_name, domain_name "
                            "FROM rocpd_info_node_{}",
                            m_uuid),
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
        m_process_info_statement =
            m_backend->create_read_statement_executor<process_info_result>(
                fmt::format("SELECT id, nid, pid, ppid, init, fini, start, end, "
                            "command, environment, extdata FROM rocpd_info_process_{}",
                            m_uuid),
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
        m_stream_info_statement =
            m_backend->create_read_statement_executor<stream_info_result>(
                fmt::format(
                    "SELECT id, nid, pid, name, extdata FROM rocpd_info_stream_{}",
                    m_uuid),
                &stream_info_result::id,
                &stream_info_result::nid,
                &stream_info_result::pid,
                &stream_info_result::name,
                &stream_info_result::extdata);
    }

    void initialize_queue_info_statement()
    {
        m_queue_info_statement =
            m_backend->create_read_statement_executor<queue_info_result>(
                fmt::format("SELECT id, nid, pid, name, extdata FROM rocpd_info_queue_{}",
                            m_uuid),
                &queue_info_result::id,
                &queue_info_result::nid,
                &queue_info_result::pid,
                &queue_info_result::name,
                &queue_info_result::extdata);
    }

    void initialize_thread_info_statement()
    {
        m_thread_info_statement =
            m_backend->create_read_statement_executor<thread_info_result>(
                fmt::format("SELECT id, nid, ppid, pid, tid, name, start, end, extdata "
                            "FROM rocpd_info_thread_{}",
                            m_uuid),
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
        // v4.0 dropped user_name and added generic_name; map generic_name onto the
        // user_name result field so downstream consumers see a stable shape.
        m_agent_info_statement =
            m_backend->create_read_statement_executor<agent_info_result>(
                fmt::format("SELECT id, nid, pid, type, absolute_index, logical_index, "
                            "type_index, uuid, name, model_name, vendor_name, "
                            "product_name, generic_name, extdata "
                            "FROM rocpd_info_agent_{}",
                            m_uuid),
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
        // v4.0 rocpd_track carries the full identity tuple; select the extended
        // columns so track_info_result's agent_id/queue_id/stream_id are populated.
        m_track_info_statement =
            m_backend->create_read_statement_executor<track_info_result>(
                fmt::format("SELECT id, nid, pid, tid, agent_id, queue_id, stream_id, "
                            "name_id, extdata FROM rocpd_track_{}",
                            m_uuid),
                &track_info_result::id,
                &track_info_result::nid,
                &track_info_result::pid,
                &track_info_result::tid,
                &track_info_result::agent_id,
                &track_info_result::queue_id,
                &track_info_result::stream_id,
                &track_info_result::name_id,
                &track_info_result::extdata);
    }

    void initialize_kernel_symbol_info_statement()
    {
        m_kernel_symbol_info_statement =
            m_backend->create_read_statement_executor<kernel_symbol_info_result>(
                fmt::format("SELECT id, nid, pid, code_object_id, kernel_name, "
                            "display_name, kernel_object, kernarg_segment_size, "
                            "kernarg_segment_alignment, group_segment_size, "
                            "private_segment_size, sgpr_count, arch_vgpr_count, "
                            "accum_vgpr_count, extdata "
                            "FROM rocpd_info_kernel_symbol_{}",
                            m_uuid),
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
        m_code_object_info_statement =
            m_backend->create_read_statement_executor<code_object_info_result>(
                fmt::format("SELECT id, nid, pid, agent_id, uri, load_base, load_size, "
                            "load_delta, storage_type, extdata "
                            "FROM rocpd_info_code_object_{}",
                            m_uuid),
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
        m_pmc_info_statement = m_backend->create_read_statement_executor<pmc_info_result>(
            fmt::format("SELECT id, nid, pid, agent_id, target_arch, event_code, "
                        "instance_id, name, symbol, description, long_description, "
                        "component, units, value_type, block, expression, is_constant, "
                        "is_derived, extdata FROM rocpd_info_pmc_{}",
                        m_uuid),
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

    void initialize_track_synthesis_statements()
    {
        const auto& u = m_uuid;

        // A track_id is a counter iff at least one of its rocpd_sample rows joins
        // rocpd_pmc_event on event_id (sample tracks with zero pmc_event rows, e.g.
        // region timer-samples, are not counters). Matches counter_track_names below
        // so discovery and metadata agree on the same counter set.
        m_distinct_sample_track_ids =
            m_backend->create_read_statement_executor<sample_track_id_result>(
                fmt::format("SELECT DISTINCT s.track_id FROM rocpd_sample_{u} s "
                            "JOIN rocpd_pmc_event_{u} pe ON pe.event_id = s.event_id",
                            fmt::arg("u", u)),
                &sample_track_id_result::track_id);

        // Stream tracks: v4 stream_id lives on rocpd_track, and multiple tracks can share
        // a stream_id, so synthesize one stream track per distinct (nid, pid, stream_id)
        // rather than 1:1 per rocpd_track row. NULL stream_id excluded (a stream track
        // needs a concrete identity). These stream tracks are ADDITIVE — the same
        // rocpd_track rows still yield their existing gpu_queue/dma/cpu_thread tracks;
        // this matches Optiq, which builds queue AND stream tracks from the same events.
        m_distinct_stream_tracks =
            m_backend->create_read_statement_executor<distinct_stream_result>(
                fmt::format("SELECT DISTINCT nid, pid, stream_id FROM rocpd_track_{} "
                            "WHERE stream_id IS NOT NULL",
                            u),
                &distinct_stream_result::nid,
                &distinct_stream_result::pid,
                &distinct_stream_result::stream_id);

        m_counter_track_names =
            m_backend->create_read_statement_executor<counter_track_name_result>(
                fmt::format("SELECT s.track_id, pe.pmc_id, ip.name "
                            "FROM rocpd_sample_{u} s "
                            "JOIN rocpd_pmc_event_{u} pe ON pe.event_id = s.event_id "
                            "JOIN rocpd_info_pmc_{u} ip ON ip.id = pe.pmc_id "
                            "GROUP BY s.track_id",
                            fmt::arg("u", u)),
                &counter_track_name_result::track_id,
                &counter_track_name_result::pmc_id,
                &counter_track_name_result::name);

        // memory tracks: one per distinct (nid, agent_id, queue_id, pid) in
        // rocpd_memory_allocate JOIN rocpd_track (agent_id / queue_id from rocpd_track).
        // NULL agent_id / queue_id are distinct group values.
        m_distinct_memory_tracks =
            m_backend->create_read_statement_executor<distinct_memory_result>(
                fmt::format("SELECT DISTINCT T.nid, T.agent_id, T.queue_id, T.pid "
                            "FROM rocpd_memory_allocate_{u} ma "
                            "JOIN rocpd_track_{u} T ON T.id = ma.track_id",
                            fmt::arg("u", u)),
                &distinct_memory_result::nid,
                &distinct_memory_result::agent_id,
                &distinct_memory_result::queue_id,
                &distinct_memory_result::pid);

        // Collect the rocpd_track.id values referenced by rocpd_memory_allocate so the
        // generic classification loop can check memory before gpu_queue (both may have
        // agent_id + queue_id on their rocpd_track row).
        m_memory_alloc_track_ids =
            m_backend->create_read_statement_executor<sample_track_id_result>(
                fmt::format("SELECT DISTINCT track_id FROM rocpd_memory_allocate_{}", u),
                &sample_track_id_result::track_id);

        // Matches Optiq's GetRocprofPerformanceCountersTrackQuery v4 GROUP BY exactly.
        m_distinct_kd_pmc_tracks =
            m_backend->create_read_statement_executor<distinct_kd_pmc_result>(
                fmt::format("SELECT DISTINCT T.nid, T.agent_id, PMC_E.pmc_id, T.pid "
                            "FROM rocpd_pmc_event_{u} PMC_E "
                            "INNER JOIN rocpd_kernel_dispatch_{u} K "
                            "ON K.event_id = PMC_E.event_id "
                            "INNER JOIN rocpd_track_{u} T ON T.id = K.track_id",
                            fmt::arg("u", u)),
                &distinct_kd_pmc_result::nid,
                &distinct_kd_pmc_result::agent_id,
                &distinct_kd_pmc_result::pmc_id,
                &distinct_kd_pmc_result::pid);

        // agent_id (from rocpd_track) is reliable in v4: track_id is always present on
        // alloc events.
        m_distinct_mem_activity_tracks =
            m_backend->create_read_statement_executor<distinct_mem_activity_result>(
                fmt::format("SELECT DISTINCT T.nid, T.pid, T.agent_id "
                            "FROM rocpd_memory_allocate_{u} ma "
                            "INNER JOIN rocpd_track_{u} T ON T.id = ma.track_id",
                            fmt::arg("u", u)),
                &distinct_mem_activity_result::nid,
                &distinct_mem_activity_result::pid,
                &distinct_mem_activity_result::agent_id);

        // Per-agent running sums are computed downstream in C++, not here.
        m_mem_activity_raw_track =
            m_backend->create_read_statement_executor<mem_activity_raw_result,
                                                      bind_types<size_t, size_t>>(
                fmt::format("SELECT ma.id, ts_s.value, ma.address, ma.size, T.agent_id, "
                            "ma.type "
                            "FROM rocpd_memory_allocate_{u} ma "
                            "INNER JOIN rocpd_track_{u} T ON T.id = ma.track_id "
                            "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = ma.start_id "
                            "WHERE T.nid = ? AND T.pid = ? "
                            "ORDER BY ts_s.value",
                            fmt::arg("u", u)),
                &mem_activity_raw_result::id,
                &mem_activity_raw_result::start,
                &mem_activity_raw_result::address,
                &mem_activity_raw_result::size,
                &mem_activity_raw_result::agent_id,
                &mem_activity_raw_result::type);

        m_kd_pmc_track_ids =
            m_backend->create_read_statement_executor<sample_track_id_result>(
                fmt::format("SELECT DISTINCT K.track_id "
                            "FROM rocpd_pmc_event_{u} PMC_E "
                            "INNER JOIN rocpd_kernel_dispatch_{u} K "
                            "ON K.event_id = PMC_E.event_id",
                            fmt::arg("u", u)),
                &sample_track_id_result::track_id);
    }

    void initialize_interval_track_statements()
    {
        const auto& u = m_uuid;

        // Category resolved in-SQL via rocpd_info_category; rocpd_event /
        // rocpd_info_category are LEFT JOINed so the row set is unchanged — additive.
        m_region_interval_track_v4 =
            m_backend
                ->create_read_statement_executor<interval_row_result, bind_types<size_t>>(
                    fmt::format(
                        "SELECT r.id, ts_s.value, ts_e.value, r.name_id, IC.name "
                        "FROM rocpd_region_{u} r "
                        "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = r.start_id "
                        "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = r.end_id "
                        "LEFT JOIN rocpd_event_{u} E ON E.id = r.event_id "
                        "LEFT JOIN rocpd_info_category_{u} IC ON IC.id = E.category_id "
                        "WHERE r.track_id = ? ORDER BY ts_s.value",
                        fmt::arg("u", u)),
                    &interval_row_result::id,
                    &interval_row_result::start,
                    &interval_row_result::end,
                    &interval_row_result::name_ref,
                    &interval_row_result::category);

        // kernel dispatch intervals: name_ref is the kernel_symbol id.
        m_kernel_dispatch_interval_track_v4 =
            m_backend
                ->create_read_statement_executor<interval_row_result, bind_types<size_t>>(
                    fmt::format(
                        "SELECT k.id, ts_s.value, ts_e.value, k.kernel_id, IC.name "
                        "FROM rocpd_kernel_dispatch_{u} k "
                        "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = k.start_id "
                        "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = k.end_id "
                        "LEFT JOIN rocpd_event_{u} E ON E.id = k.event_id "
                        "LEFT JOIN rocpd_info_category_{u} IC ON IC.id = E.category_id "
                        "WHERE k.track_id = ? ORDER BY ts_s.value",
                        fmt::arg("u", u)),
                    &interval_row_result::id,
                    &interval_row_result::start,
                    &interval_row_result::end,
                    &interval_row_result::name_ref,
                    &interval_row_result::category);

        m_memory_copy_interval_track_v4 =
            m_backend
                ->create_read_statement_executor<interval_row_result, bind_types<size_t>>(
                    fmt::format(
                        "SELECT mc.id, ts_s.value, ts_e.value, mc.name_id, IC.name "
                        "FROM rocpd_memory_copy_{u} mc "
                        "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = mc.start_id "
                        "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = mc.end_id "
                        "LEFT JOIN rocpd_event_{u} E ON E.id = mc.event_id "
                        "LEFT JOIN rocpd_info_category_{u} IC ON IC.id = E.category_id "
                        "WHERE mc.track_id = ? ORDER BY ts_s.value",
                        fmt::arg("u", u)),
                    &interval_row_result::id,
                    &interval_row_result::start,
                    &interval_row_result::end,
                    &interval_row_result::name_ref,
                    &interval_row_result::category);

        // memory allocate intervals: v4 has a native name_id on rocpd_memory_allocate,
        // so name_ref is populated here (unlike v3).
        m_memory_alloc_interval_track_v4 =
            m_backend
                ->create_read_statement_executor<interval_row_result, bind_types<size_t>>(
                    fmt::format(
                        "SELECT ma.id, ts_s.value, ts_e.value, ma.name_id, IC.name "
                        "FROM rocpd_memory_allocate_{u} ma "
                        "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = ma.start_id "
                        "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = ma.end_id "
                        "LEFT JOIN rocpd_event_{u} E ON E.id = ma.event_id "
                        "LEFT JOIN rocpd_info_category_{u} IC ON IC.id = E.category_id "
                        "WHERE ma.track_id = ? ORDER BY ts_s.value",
                        fmt::arg("u", u)),
                    &interval_row_result::id,
                    &interval_row_result::start,
                    &interval_row_result::end,
                    &interval_row_result::name_ref,
                    &interval_row_result::category);

        // kernel_dispatch_pmc intervals. Each event = one kernel_dispatch with PMC data.
        // name_ref is kernel_id (for kernel symbol resolution). Keyed by (K.nid, K.pid,
        // K.agent_id, PMC_E.pmc_id) via rocpd_track JOIN.
        m_kd_pmc_interval_track = m_backend->create_read_statement_executor<
            interval_row_result,
            bind_types<size_t, size_t, size_t, size_t>>(
            fmt::format("SELECT K.id, ts_s.value, ts_e.value, K.kernel_id, IC.name "
                        "FROM rocpd_pmc_event_{u} PMC_E "
                        "INNER JOIN rocpd_kernel_dispatch_{u} K "
                        "ON K.event_id = PMC_E.event_id "
                        "INNER JOIN rocpd_track_{u} T ON T.id = K.track_id "
                        "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = K.start_id "
                        "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = K.end_id "
                        "LEFT JOIN rocpd_event_{u} E ON E.id = K.event_id "
                        "LEFT JOIN rocpd_info_category_{u} IC ON IC.id = E.category_id "
                        "WHERE T.nid = ? AND T.pid = ? AND T.agent_id = ? "
                        "AND PMC_E.pmc_id = ? "
                        "ORDER BY ts_s.value",
                        fmt::arg("u", u)),
            &interval_row_result::id,
            &interval_row_result::start,
            &interval_row_result::end,
            &interval_row_result::name_ref,
            &interval_row_result::category);

        // stream: aggregates kernel_dispatch + memory_copy + memory_allocate sharing a
        // stream. v4 stream_id is on rocpd_track, so each leg joins its event table to
        // rocpd_track and filters T.stream_id = ? (bound three times). start/end resolve
        // through the timestamp spine. op_kind literal per leg (kernel_dispatch=1,
        // memory_copy=2, memory_allocate=3) drives the reader's per-event name lookup
        // and get_*_details() dispatch. Unlike v3, memory_allocate carries a name_id in
        // v4, so its name_ref is populated.
        m_stream_interval_track =
            m_backend->create_read_statement_executor<interval_row_result,
                                                      bind_types<size_t, size_t, size_t>>(
                fmt::format(
                    "SELECT k.id, ts_s.value, ts_e.value, k.kernel_id, IC.name, 1 "
                    "FROM rocpd_kernel_dispatch_{u} k "
                    "JOIN rocpd_track_{u} T ON T.id = k.track_id "
                    "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = k.start_id "
                    "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = k.end_id "
                    "LEFT JOIN rocpd_event_{u} E ON E.id = k.event_id "
                    "LEFT JOIN rocpd_info_category_{u} IC ON IC.id = E.category_id "
                    "WHERE T.stream_id = ? "
                    "UNION ALL "
                    "SELECT mc.id, ts_s.value, ts_e.value, mc.name_id, IC.name, 2 "
                    "FROM rocpd_memory_copy_{u} mc "
                    "JOIN rocpd_track_{u} T ON T.id = mc.track_id "
                    "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = mc.start_id "
                    "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = mc.end_id "
                    "LEFT JOIN rocpd_event_{u} E ON E.id = mc.event_id "
                    "LEFT JOIN rocpd_info_category_{u} IC ON IC.id = E.category_id "
                    "WHERE T.stream_id = ? "
                    "UNION ALL "
                    "SELECT ma.id, ts_s.value, ts_e.value, ma.name_id, IC.name, 3 "
                    "FROM rocpd_memory_allocate_{u} ma "
                    "JOIN rocpd_track_{u} T ON T.id = ma.track_id "
                    "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = ma.start_id "
                    "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = ma.end_id "
                    "LEFT JOIN rocpd_event_{u} E ON E.id = ma.event_id "
                    "LEFT JOIN rocpd_info_category_{u} IC ON IC.id = E.category_id "
                    "WHERE T.stream_id = ? "
                    "ORDER BY 2",
                    fmt::arg("u", u)),
                &interval_row_result::id,
                &interval_row_result::start,
                &interval_row_result::end,
                &interval_row_result::name_ref,
                &interval_row_result::category,
                &interval_row_result::op_kind);
    }

    void initialize_track_stats_statements()
    {
        const auto& u = m_uuid;

        // MIN(start)/MAX(end)/COUNT over exactly the rows the matching interval-track
        // query returns, so per-track bounds/count agree with a full slice load.

        m_region_stats_track_v4 =
            m_backend
                ->create_read_statement_executor<track_stats_result, bind_types<size_t>>(
                    fmt::format("SELECT MIN(ts_s.value), MAX(ts_e.value), COUNT(*) "
                                "FROM rocpd_region_{u} r "
                                "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = r.start_id "
                                "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = r.end_id "
                                "WHERE r.track_id = ?",
                                fmt::arg("u", u)),
                    &track_stats_result::min_ts,
                    &track_stats_result::max_ts,
                    &track_stats_result::count);

        m_kernel_dispatch_stats_track_v4 =
            m_backend
                ->create_read_statement_executor<track_stats_result, bind_types<size_t>>(
                    fmt::format("SELECT MIN(ts_s.value), MAX(ts_e.value), COUNT(*) "
                                "FROM rocpd_kernel_dispatch_{u} k "
                                "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = k.start_id "
                                "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = k.end_id "
                                "WHERE k.track_id = ?",
                                fmt::arg("u", u)),
                    &track_stats_result::min_ts,
                    &track_stats_result::max_ts,
                    &track_stats_result::count);

        m_memory_copy_stats_track_v4 =
            m_backend
                ->create_read_statement_executor<track_stats_result, bind_types<size_t>>(
                    fmt::format("SELECT MIN(ts_s.value), MAX(ts_e.value), COUNT(*) "
                                "FROM rocpd_memory_copy_{u} mc "
                                "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = mc.start_id "
                                "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = mc.end_id "
                                "WHERE mc.track_id = ?",
                                fmt::arg("u", u)),
                    &track_stats_result::min_ts,
                    &track_stats_result::max_ts,
                    &track_stats_result::count);

        m_memory_alloc_stats_track_v4 =
            m_backend
                ->create_read_statement_executor<track_stats_result, bind_types<size_t>>(
                    fmt::format("SELECT MIN(ts_s.value), MAX(ts_e.value), COUNT(*) "
                                "FROM rocpd_memory_allocate_{u} ma "
                                "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = ma.start_id "
                                "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = ma.end_id "
                                "WHERE ma.track_id = ?",
                                fmt::arg("u", u)),
                    &track_stats_result::min_ts,
                    &track_stats_result::max_ts,
                    &track_stats_result::count);

        m_kd_pmc_stats_track = m_backend->create_read_statement_executor<
            track_stats_result,
            bind_types<size_t, size_t, size_t, size_t>>(
            fmt::format("SELECT MIN(ts_s.value), MAX(ts_e.value), COUNT(*) "
                        "FROM rocpd_pmc_event_{u} PMC_E "
                        "INNER JOIN rocpd_kernel_dispatch_{u} K "
                        "ON K.event_id = PMC_E.event_id "
                        "INNER JOIN rocpd_track_{u} T ON T.id = K.track_id "
                        "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = K.start_id "
                        "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = K.end_id "
                        "WHERE T.nid = ? AND T.pid = ? AND T.agent_id = ? "
                        "AND PMC_E.pmc_id = ?",
                        fmt::arg("u", u)),
            &track_stats_result::min_ts,
            &track_stats_result::max_ts,
            &track_stats_result::count);

        // stream: MIN(start)/MAX(end)/COUNT over the same 3-way UNION as the v4 stream
        // interval query (T.stream_id bound three times), through the timestamp spine.
        m_stream_stats_track =
            m_backend->create_read_statement_executor<track_stats_result,
                                                      bind_types<size_t, size_t, size_t>>(
                fmt::format("SELECT MIN(s), MAX(e), COUNT(*) FROM ("
                            "SELECT ts_s.value AS s, ts_e.value AS e "
                            "FROM rocpd_kernel_dispatch_{u} k "
                            "JOIN rocpd_track_{u} T ON T.id = k.track_id "
                            "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = k.start_id "
                            "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = k.end_id "
                            "WHERE T.stream_id = ? "
                            "UNION ALL "
                            "SELECT ts_s.value, ts_e.value "
                            "FROM rocpd_memory_copy_{u} mc "
                            "JOIN rocpd_track_{u} T ON T.id = mc.track_id "
                            "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = mc.start_id "
                            "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = mc.end_id "
                            "WHERE T.stream_id = ? "
                            "UNION ALL "
                            "SELECT ts_s.value, ts_e.value "
                            "FROM rocpd_memory_allocate_{u} ma "
                            "JOIN rocpd_track_{u} T ON T.id = ma.track_id "
                            "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = ma.start_id "
                            "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = ma.end_id "
                            "WHERE T.stream_id = ?)",
                            fmt::arg("u", u)),
                &track_stats_result::min_ts,
                &track_stats_result::max_ts,
                &track_stats_result::count);

        m_scalar_stats =
            m_backend
                ->create_read_statement_executor<track_stats_result, bind_types<size_t>>(
                    fmt::format("SELECT MIN(ts.value), MAX(ts.value), COUNT(*) "
                                "FROM rocpd_sample_{u} s "
                                "JOIN rocpd_timestamp_{u} ts ON ts.id = s.timestamp_id "
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
                    fmt::format("SELECT s.id, ts.value, p.value "
                                "FROM rocpd_sample_{u} s "
                                "JOIN rocpd_timestamp_{u} ts ON ts.id = s.timestamp_id "
                                "JOIN rocpd_pmc_event_{u} p ON p.event_id = s.event_id "
                                "WHERE s.track_id = ? ORDER BY ts.value",
                                fmt::arg("u", u)),
                    &scalar_row_result::id,
                    &scalar_row_result::timestamp,
                    &scalar_row_result::value);
    }

    void initialize_scalar_detail_statement()
    {
        const auto& u = m_uuid;

        // Keyed on rocpd_sample.id (scalar_sample_t::opaque_id).
        m_scalar_detail = m_backend->create_read_statement_executor<scalar_detail_result,
                                                                    bind_types<size_t>>(
            fmt::format("SELECT s.id, s.track_id, ts.value, p.value, s.event_id "
                        "FROM rocpd_sample_{u} s "
                        "JOIN rocpd_timestamp_{u} ts ON ts.id = s.timestamp_id "
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
                fmt::format("SELECT s.id, s.track_id, ts.value, p.value, s.event_id "
                            "FROM rocpd_pmc_event_{u} p "
                            "JOIN rocpd_sample_{u} s ON s.event_id = p.event_id "
                            "JOIN rocpd_timestamp_{u} ts ON ts.id = s.timestamp_id "
                            "WHERE p.id = ?",
                            fmt::arg("u", u)),
                &scalar_detail_result::id,
                &scalar_detail_result::track_id,
                &scalar_detail_result::timestamp,
                &scalar_detail_result::value,
                &scalar_detail_result::event_id);
    }

    void initialize_ambiguous_pmc_ids_statement()
    {
        const auto& u = m_uuid;
        m_ambiguous_pmc_ids =
            m_backend->create_read_statement_executor<ambiguous_pmc_id_result>(
                fmt::format("SELECT DISTINCT pmc_id "
                            "FROM rocpd_pmc_event_{u} "
                            "GROUP BY event_id, pmc_id HAVING COUNT(*) > 1",
                            fmt::arg("u", u)),
                &ambiguous_pmc_id_result::pmc_id);
    }

    void initialize_flow_statements()
    {
        const auto& u = m_uuid;

        // A flow leg links a SOURCE event to a DEST event sharing the same non-zero
        // stack_id (the stack-clique join, E{s}.id != E{d}.id). source/dest may be the
        // same table (region->region, same-type siblings) or different (region->GPU).
        auto make_flow_set = [&](const std::string& source_table,
                                 const std::string& source_alias,
                                 const std::string& dest_table,
                                 const std::string& dest_alias) -> flow_statement_set {
            // Surface each endpoint's start + parent_stack_id and the shared clique
            // stack_id so get_flows can orient the directed edge (parent lineage else
            // start-ts) and derive its flow_id. Column order matches the member-pointer
            // binding order below.
            const auto base_sql = fmt::format(
                "SELECT {s}.id, {d}.id, "
                "(SELECT value FROM rocpd_timestamp_{u} WHERE id = {s}.start_id), "
                "(SELECT value FROM rocpd_timestamp_{u} WHERE id = {d}.start_id), "
                "E{s}.stack_id, E{s}.parent_stack_id, E{d}.parent_stack_id "
                "FROM {st}_{u} {s} "
                "JOIN rocpd_event_{u} E{s} ON {s}.event_id = E{s}.id "
                "JOIN rocpd_event_{u} E{d} "
                "  ON E{d}.stack_id = E{s}.stack_id AND E{d}.id != E{s}.id "
                "JOIN {dt}_{u} {d} ON {d}.event_id = E{d}.id "
                "WHERE E{s}.stack_id IS NOT NULL AND E{s}.stack_id != 0",
                fmt::arg("u", u),
                fmt::arg("s", source_alias),
                fmt::arg("st", source_table),
                fmt::arg("d", dest_alias),
                fmt::arg("dt", dest_table));

            flow_statement_set out;
            out.base = m_backend->create_read_statement_executor<flow_row_result>(
                base_sql,
                &flow_row_result::source_id,
                &flow_row_result::dest_id,
                &flow_row_result::source_start,
                &flow_row_result::dest_start,
                &flow_row_result::stack_id,
                &flow_row_result::source_parent,
                &flow_row_result::dest_parent);

            const auto time_sql =
                base_sql +
                fmt::format(" AND {s}.start_id IN (SELECT id FROM rocpd_timestamp_{u} "
                            "WHERE value >= ? AND value <= ?)",
                            fmt::arg("u", u),
                            fmt::arg("s", source_alias));

            out.time_filtered =
                m_backend->create_read_statement_executor<flow_row_result,
                                                          bind_types<size_t, size_t>>(
                    time_sql,
                    &flow_row_result::source_id,
                    &flow_row_result::dest_id,
                    &flow_row_result::source_start,
                    &flow_row_result::dest_start,
                    &flow_row_result::stack_id,
                    &flow_row_result::source_parent,
                    &flow_row_result::dest_parent);
            return out;
        };

        m_region_to_kernel_dispatch_flows =
            make_flow_set("rocpd_region", "R", "rocpd_kernel_dispatch", "K");
        m_region_to_memory_copy_flows =
            make_flow_set("rocpd_region", "R", "rocpd_memory_copy", "MC");
        m_region_to_memory_allocate_flows =
            make_flow_set("rocpd_region", "R", "rocpd_memory_allocate", "MA");
        m_region_to_region_flows =
            make_flow_set("rocpd_region", "R", "rocpd_region", "R2");
        m_kernel_dispatch_sibling_flows =
            make_flow_set("rocpd_kernel_dispatch", "K", "rocpd_kernel_dispatch", "K2");
        m_memory_copy_sibling_flows =
            make_flow_set("rocpd_memory_copy", "MC", "rocpd_memory_copy", "MC2");
        m_memory_allocate_sibling_flows =
            make_flow_set("rocpd_memory_allocate", "MA", "rocpd_memory_allocate", "MA2");
    }

    void initialize_timeline_event_statements()
    {
        const auto& u = m_uuid;

        // Build all four timeline variants for one interval table. v4.0 differs
        // from v3 structurally: start/end resolved through the rocpd_timestamp spine
        // (start_id/end_id); nid/pid/tid come from the rocpd_track identity anchor,
        // not the event table (v4 event tables carry only track_id). The track-scoped
        // variants filter on track_id alone (the universal v4 anchor); the three
        // leading nid/pid/tid binds are accepted for signature parity with v3 and
        // consumed as always-true `? IS NOT NULL` no-ops so the anonymous-`?` count
        // matches bind_types.
        auto make_timeline_set =
            [&](const std::string& table,
                const std::string& alias,
                const std::string& display_col) -> timeline_event_statement_set {
            const auto a = alias;

            const auto select_from = fmt::format(
                "SELECT {a}.id, ts_s.value, ts_e.value, {dn}, IC.name, "
                "TR.nid, TR.pid, TR.tid, {a}.track_id "
                "FROM {tbl}_{u} {a} "
                "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = {a}.start_id "
                "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = {a}.end_id "
                "JOIN rocpd_track_{u} TR ON TR.id = {a}.track_id "
                "LEFT JOIN rocpd_event_{u} E ON E.id = {a}.event_id "
                "LEFT JOIN rocpd_info_category_{u} IC ON IC.id = E.category_id",
                fmt::arg("a", a),
                fmt::arg("dn", display_col),
                fmt::arg("tbl", table),
                fmt::arg("u", u));

            timeline_event_statement_set out;

            out.base = m_backend->create_read_statement_executor<timeline_event_result>(
                select_from,
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
                    select_from + " WHERE ts_s.value <= ? AND ts_e.value >= ?",
                    &timeline_event_result::id,
                    &timeline_event_result::start_timestamp,
                    &timeline_event_result::end_timestamp,
                    &timeline_event_result::display_name_id,
                    &timeline_event_result::category_name,
                    &timeline_event_result::nid,
                    &timeline_event_result::pid,
                    &timeline_event_result::tid,
                    &timeline_event_result::track_id);

            const auto track_where =
                " WHERE ? IS NOT NULL AND ? IS NOT NULL AND ? IS NOT NULL AND " + a +
                ".track_id = ?";

            out.track_filtered = m_backend->create_read_statement_executor<
                timeline_event_result,
                bind_types<size_t, size_t, size_t, size_t>>(
                select_from + track_where,
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
                select_from + track_where + " AND ts_s.value <= ? AND ts_e.value >= ?",
                &timeline_event_result::id,
                &timeline_event_result::start_timestamp,
                &timeline_event_result::end_timestamp,
                &timeline_event_result::display_name_id,
                &timeline_event_result::category_name,
                &timeline_event_result::nid,
                &timeline_event_result::pid,
                &timeline_event_result::tid,
                &timeline_event_result::track_id);

            return out;
        };

        m_region_statements = make_timeline_set("rocpd_region", "R", "R.name_id");
        m_kernel_dispatch_statements =
            make_timeline_set("rocpd_kernel_dispatch", "K", "K.region_name_id");
        // v4.0 memory_allocate has a native NOT NULL name_id (v3 lacked one).
        m_memory_allocate_statements =
            make_timeline_set("rocpd_memory_allocate", "MA", "MA.name_id");
        m_memory_copy_statements =
            make_timeline_set("rocpd_memory_copy", "MC", "MC.region_name_id");
    }

    void initialize_count_statements()
    {
        const auto& u = m_uuid;

        auto make_count_stmt = [&](const std::string& table) {
            return m_backend->create_read_statement_executor<count_result>(
                fmt::format("SELECT COUNT(*) FROM {}_{}", table, u),
                &count_result::count);
        };
        auto make_count_time_filtered_stmt = [&](const std::string& table) {
            return m_backend->create_read_statement_executor<count_result,
                                                             bind_types<size_t, size_t>>(
                fmt::format("SELECT COUNT(*) FROM {tbl}_{u} T "
                            "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = T.start_id "
                            "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = T.end_id "
                            "WHERE ts_s.value <= ? AND ts_e.value >= ?",
                            fmt::arg("tbl", table),
                            fmt::arg("u", u)),
                &count_result::count);
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
        const auto& u = m_uuid;

        auto make_time_range_stmt = [&](const std::string& table) {
            return m_backend->create_read_statement_executor<time_range_result>(
                fmt::format("SELECT MIN(ts_s.value), MAX(ts_e.value) FROM {tbl}_{u} T "
                            "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = T.start_id "
                            "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = T.end_id",
                            fmt::arg("tbl", table),
                            fmt::arg("u", u)),
                &time_range_result::min_start,
                &time_range_result::max_end);
        };

        m_region_time_range          = make_time_range_stmt("rocpd_region");
        m_kernel_dispatch_time_range = make_time_range_stmt("rocpd_kernel_dispatch");
        m_memory_copy_time_range     = make_time_range_stmt("rocpd_memory_copy");
        m_memory_alloc_time_range    = make_time_range_stmt("rocpd_memory_allocate");
    }

    // GROUP-BY-name aggregates (v4.0), grouped by name_col (kernel_id for kernels,
    // name_id for regions). The time-filtered variant keeps events overlapping
    // [start,end] (bind order end,start).
    void initialize_summary_statements()
    {
        const auto& u = m_uuid;

        auto make_summary_stmt = [&](const std::string& table,
                                     const std::string& name_col) {
            return m_backend->create_read_statement_executor<summary_result>(
                fmt::format("SELECT T.{col}, COUNT(*), "
                            "SUM(ts_e.value - ts_s.value), MIN(ts_e.value - ts_s.value), "
                            "MAX(ts_e.value - ts_s.value) FROM {tbl}_{u} T "
                            "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = T.start_id "
                            "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = T.end_id "
                            "GROUP BY T.{col}",
                            fmt::arg("col", name_col),
                            fmt::arg("tbl", table),
                            fmt::arg("u", u)),
                &summary_result::name_ref,
                &summary_result::count,
                &summary_result::total_duration,
                &summary_result::min_duration,
                &summary_result::max_duration);
        };
        auto make_summary_time_filtered_stmt = [&](const std::string& table,
                                                   const std::string& name_col) {
            return m_backend->create_read_statement_executor<summary_result,
                                                             bind_types<size_t, size_t>>(
                fmt::format("SELECT T.{col}, COUNT(*), "
                            "SUM(ts_e.value - ts_s.value), MIN(ts_e.value - ts_s.value), "
                            "MAX(ts_e.value - ts_s.value) FROM {tbl}_{u} T "
                            "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = T.start_id "
                            "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = T.end_id "
                            "WHERE ts_s.value <= ? AND ts_e.value >= ? "
                            "GROUP BY T.{col}",
                            fmt::arg("col", name_col),
                            fmt::arg("tbl", table),
                            fmt::arg("u", u)),
                &summary_result::name_ref,
                &summary_result::count,
                &summary_result::total_duration,
                &summary_result::min_duration,
                &summary_result::max_duration);
        };

        m_kernel_summary = make_summary_stmt("rocpd_kernel_dispatch", "kernel_id");
        m_kernel_summary_time_filtered =
            make_summary_time_filtered_stmt("rocpd_kernel_dispatch", "kernel_id");
        m_region_summary = make_summary_stmt("rocpd_region", "name_id");
        m_region_summary_time_filtered =
            make_summary_time_filtered_stmt("rocpd_region", "name_id");
    }

    // Matches v3 deserialize_source_context: only string elements are kept; malformed
    // JSON yields an empty vector.
    static std::vector<std::string> parse_json_string_array(const std::string& s)
    {
        std::vector<std::string> out;
        if(s.empty()) return out;
        auto j = nlohmann::json::parse(s, nullptr, /*allow_exceptions=*/false);
        if(j.is_array())
        {
            for(const auto& el : j)
            {
                if(el.is_string()) out.push_back(el.get<std::string>());
            }
        }
        return out;
    }

    void initialize_event_id_statements()
    {
        const auto& u = m_uuid;

        // v4.0 has no JSON blobs on rocpd_event: call stack and line info are
        // relational. For each event we run three queries and assemble the
        // version-neutral event_id_result the reader consumes:
        //   * meta   — the scalar event row (category/stack/correlation/extdata)
        //   * frames — rocpd_call_stack -> rocpd_info_pc -> rocpd_info_address_range
        //   * lines  — rocpd_line_info  -> rocpd_info_source_code / rocpd_info_pc /
        //              rocpd_info_address_range
        // A single lazy result_set cannot express the one-to-many frame/line fan-out,
        // so the accessor materializes and returns a fully-built vector.
        struct meta_row
        {
            std::optional<size_t>      event_id;
            std::optional<std::string> category_name;
            std::optional<size_t>      stack_id;
            std::optional<size_t>      parent_stack_id;
            std::optional<size_t>      correlation_id;
            std::string                extdata;
        };
        struct frame_row
        {
            std::optional<std::string> function;
            std::optional<std::string> file;
            std::optional<size_t>      line;
            std::optional<size_t>      address_base;
            std::optional<size_t>      address_low;
            std::optional<size_t>      address_high;
        };
        struct line_row
        {
            std::optional<std::string> sc_file;
            std::optional<size_t>      sc_line_number;
            std::string                sc_lines;
            std::string                sc_instructions;
            std::optional<std::string> pc_function;
            std::optional<std::string> pc_file;
            std::optional<size_t>      pc_line;
            std::optional<size_t>      address_base;
            std::optional<size_t>      address_low;
            std::optional<size_t>      address_high;
        };

        auto make_event_id_stmt = [&](const std::string& table) -> event_id_func_t {
            auto meta_exec =
                m_backend->create_read_statement_executor<meta_row, bind_types<size_t>>(
                    fmt::format(
                        "SELECT E.id, IC.name, E.stack_id, "
                        "E.parent_stack_id, E.correlation_id, E.extdata "
                        "FROM {tbl}_{u} T "
                        "INNER JOIN rocpd_event_{u} E ON E.id = T.event_id "
                        "LEFT JOIN rocpd_info_category_{u} IC ON IC.id = E.category_id "
                        "WHERE T.id = ?",
                        fmt::arg("tbl", table),
                        fmt::arg("u", u)),
                    &meta_row::event_id,
                    &meta_row::category_name,
                    &meta_row::stack_id,
                    &meta_row::parent_stack_id,
                    &meta_row::correlation_id,
                    &meta_row::extdata);

            auto frame_exec =
                m_backend->create_read_statement_executor<frame_row, bind_types<size_t>>(
                    fmt::format("SELECT pc.function, pc.file, pc.line, "
                                "ar.address_base, ar.address_low, ar.address_high "
                                "FROM rocpd_call_stack_{u} cs "
                                "LEFT JOIN rocpd_info_pc_{u} pc ON pc.id = cs.pc_id "
                                "LEFT JOIN rocpd_info_address_range_{u} ar "
                                "  ON ar.id = pc.address_id "
                                "WHERE cs.event_id = ? ORDER BY cs.depth",
                                fmt::arg("u", u)),
                    &frame_row::function,
                    &frame_row::file,
                    &frame_row::line,
                    &frame_row::address_base,
                    &frame_row::address_low,
                    &frame_row::address_high);

            auto line_exec =
                m_backend->create_read_statement_executor<line_row, bind_types<size_t>>(
                    fmt::format(
                        "SELECT sc.file, sc.line_number, sc.lines, sc.instructions, "
                        "pc.function, pc.file, pc.line, "
                        "ar.address_base, ar.address_low, ar.address_high "
                        "FROM rocpd_line_info_{u} li "
                        "LEFT JOIN rocpd_info_source_code_{u} sc "
                        "  ON sc.id = li.source_code_id "
                        "LEFT JOIN rocpd_info_pc_{u} pc ON pc.id = li.pc_id "
                        "LEFT JOIN rocpd_info_address_range_{u} ar "
                        "  ON ar.id = COALESCE(pc.address_id, sc.address_id) "
                        "WHERE li.event_id = ?",
                        fmt::arg("u", u)),
                    &line_row::sc_file,
                    &line_row::sc_line_number,
                    &line_row::sc_lines,
                    &line_row::sc_instructions,
                    &line_row::pc_function,
                    &line_row::pc_file,
                    &line_row::pc_line,
                    &line_row::address_base,
                    &line_row::address_low,
                    &line_row::address_high);

            return [meta_exec, frame_exec, line_exec](
                       size_t id) -> std::vector<event_id_result> {
                auto                         metas = meta_exec(id).to_vector();
                std::vector<event_id_result> out;
                out.reserve(metas.size());
                for(auto& m : metas)
                {
                    event_id_result e;
                    e.event_id        = m.event_id;
                    e.category_name   = std::move(m.category_name);
                    e.stack_id        = m.stack_id;
                    e.parent_stack_id = m.parent_stack_id;
                    e.correlation_id  = m.correlation_id;
                    e.event_extdata   = std::move(m.extdata);

                    if(m.event_id)
                    {
                        const auto eid = *m.event_id;

                        for(auto& f : frame_exec(eid).to_vector())
                        {
                            reader_types::stack_frame_t frame;
                            if(f.function || f.file || f.line)
                            {
                                reader_types::program_counter_info_t pc;
                                pc.function           = f.function.value_or("");
                                pc.filename           = f.file.value_or("");
                                pc.line_number        = f.line;
                                frame.program_counter = std::move(pc);
                            }
                            if(f.address_base)
                            {
                                reader_types::address_range_info_t ar;
                                ar.address_base     = f.address_base.value_or(0);
                                ar.address_low      = f.address_low.value_or(0);
                                ar.address_high     = f.address_high.value_or(0);
                                frame.address_range = std::move(ar);
                            }
                            e.call_stack.push_back(std::move(frame));
                        }

                        for(auto& l : line_exec(eid).to_vector())
                        {
                            reader_types::line_info_entry_t entry;
                            if(l.sc_file || l.sc_line_number || l.sc_lines != "[]" ||
                               l.sc_instructions != "[]")
                            {
                                reader_types::source_code_info_t sc;
                                sc.filename             = l.sc_file;
                                sc.starting_line_number = l.sc_line_number;
                                sc.source_code_lines =
                                    parse_json_string_array(l.sc_lines);
                                sc.assembly_instruction_lines =
                                    parse_json_string_array(l.sc_instructions);
                                entry.source_code = std::move(sc);
                            }
                            if(l.pc_function || l.pc_file || l.pc_line)
                            {
                                reader_types::program_counter_info_t pc;
                                pc.function           = l.pc_function.value_or("");
                                pc.filename           = l.pc_file.value_or("");
                                pc.line_number        = l.pc_line;
                                entry.program_counter = std::move(pc);
                            }
                            if(l.address_base)
                            {
                                reader_types::address_range_info_t ar;
                                ar.address_base     = l.address_base.value_or(0);
                                ar.address_low      = l.address_low.value_or(0);
                                ar.address_high     = l.address_high.value_or(0);
                                entry.address_range = std::move(ar);
                            }
                            e.line_info.push_back(std::move(entry));
                        }
                    }

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

    void initialize_arg_detail_statement()
    {
        const auto& u = m_uuid;

        // rocpd_arg is structurally identical to v3 (event_id/position/type/name/
        // value/extdata); the only difference is the uuid-suffixed table name.
        m_arg_detail =
            m_backend
                ->create_read_statement_executor<arg_detail_result, bind_types<size_t>>(
                    fmt::format("SELECT position, type, name, value, extdata "
                                "FROM rocpd_arg_{} WHERE event_id = ? ORDER BY position",
                                u),
                    &arg_detail_result::position,
                    &arg_detail_result::type,
                    &arg_detail_result::name,
                    &arg_detail_result::value,
                    &arg_detail_result::extdata);
    }

    void initialize_correlated_event_statements()
    {
        const auto& u = m_uuid;

        // Row shape matches the v4 timeline set (spine JOINs + rocpd_track identity +
        // rocpd_info_category); only the WHERE clause differs, selecting stack_id
        // siblings and excluding the event itself.
        auto make_correlated_stmt =
            [&](const std::string& table,
                const std::string& alias,
                const std::string& display_col) -> correlated_event_func_t {
            const auto a = alias;
            auto       q = fmt::format(
                "SELECT {a}.id, ts_s.value, ts_e.value, {dn}, IC.name, "
                      "TR.nid, TR.pid, TR.tid, {a}.track_id "
                      "FROM {tbl}_{u} {a} "
                      "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = {a}.start_id "
                      "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = {a}.end_id "
                      "JOIN rocpd_track_{u} TR ON TR.id = {a}.track_id "
                      "INNER JOIN rocpd_event_{u} E ON E.id = {a}.event_id "
                      "LEFT JOIN rocpd_info_category_{u} IC ON IC.id = E.category_id "
                      "WHERE E.stack_id = ? AND E.id != ?",
                fmt::arg("a", a),
                fmt::arg("dn", display_col),
                fmt::arg("tbl", table),
                fmt::arg("u", u));

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
        // v4.0 memory_allocate has a native NOT NULL name_id (see timeline set).
        m_correlated_event_statements.memory_allocate =
            make_correlated_stmt("rocpd_memory_allocate", "MA", "MA.name_id");
    }

    void initialize_detail_statements()
    {
        const auto& u = m_uuid;

        m_region_detail = m_backend->create_read_statement_executor<region_detail_result,
                                                                    bind_types<size_t>>(
            fmt::format("SELECT r.id, ts_s.value, ts_e.value, r.name_id, r.event_id, "
                        "TR.nid, TR.pid, TR.tid, r.extdata "
                        "FROM rocpd_region_{u} r "
                        "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = r.start_id "
                        "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = r.end_id "
                        "JOIN rocpd_track_{u} TR ON TR.id = r.track_id "
                        "WHERE r.id = ?",
                        fmt::arg("u", u)),
            &region_detail_result::id,
            &region_detail_result::start,
            &region_detail_result::end,
            &region_detail_result::name_id,
            &region_detail_result::event_id,
            &region_detail_result::nid,
            &region_detail_result::pid,
            &region_detail_result::tid,
            &region_detail_result::extdata);

        m_kernel_dispatch_detail =
            m_backend->create_read_statement_executor<kernel_dispatch_detail_result,
                                                      bind_types<size_t>>(
                fmt::format(
                    "SELECT k.id, k.dispatch_id, ts_s.value, ts_e.value, k.kernel_id, "
                    "k.private_segment_size, k.group_segment_size, k.workgroup_size_x, "
                    "k.workgroup_size_y, k.workgroup_size_z, k.grid_size_x, "
                    "k.grid_size_y, k.grid_size_z, k.region_name_id, k.event_id, "
                    "TR.nid, TR.pid, TR.tid, k.extdata "
                    "FROM rocpd_kernel_dispatch_{u} k "
                    "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = k.start_id "
                    "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = k.end_id "
                    "JOIN rocpd_track_{u} TR ON TR.id = k.track_id "
                    "WHERE k.id = ?",
                    fmt::arg("u", u)),
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

        m_memory_copy_detail =
            m_backend->create_read_statement_executor<memory_copy_detail_result,
                                                      bind_types<size_t>>(
                fmt::format(
                    "SELECT mc.id, ts_s.value, ts_e.value, mc.name_id, mc.dst_agent_id, "
                    "mc.dst_address, mc.src_agent_id, mc.src_address, mc.size, "
                    "mc.region_name_id, mc.event_id, TR.nid, TR.pid, TR.tid, mc.extdata "
                    "FROM rocpd_memory_copy_{u} mc "
                    "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = mc.start_id "
                    "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = mc.end_id "
                    "JOIN rocpd_track_{u} TR ON TR.id = mc.track_id "
                    "WHERE mc.id = ?",
                    fmt::arg("u", u)),
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

        m_memory_alloc_detail = m_backend->create_read_statement_executor<
            memory_alloc_detail_result,
            bind_types<size_t>>(
            fmt::format(
                "SELECT ma.id, ma.type, ma.level, ts_s.value, ts_e.value, "
                "ma.address, ma.size, ma.event_id, TR.nid, TR.pid, TR.tid, ma.extdata "
                "FROM rocpd_memory_allocate_{u} ma "
                "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = ma.start_id "
                "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = ma.end_id "
                "JOIN rocpd_track_{u} TR ON TR.id = ma.track_id "
                "WHERE ma.id = ?",
                fmt::arg("u", u)),
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

    sample_track_id_func_t        m_distinct_sample_track_ids;
    distinct_stream_func_t        m_distinct_stream_tracks;
    distinct_memory_func_t        m_distinct_memory_tracks;
    memory_alloc_track_ids_func_t m_memory_alloc_track_ids;
    distinct_kd_pmc_func_t        m_distinct_kd_pmc_tracks;
    distinct_mem_activity_func_t  m_distinct_mem_activity_tracks;
    mem_activity_raw_func_t       m_mem_activity_raw_track;
    sample_track_id_func_t        m_kd_pmc_track_ids;
    counter_track_name_func_t     m_counter_track_names;

    interval_track_1_func_t m_region_interval_track_v4;
    interval_track_1_func_t m_kernel_dispatch_interval_track_v4;
    interval_track_1_func_t m_memory_copy_interval_track_v4;
    interval_track_1_func_t m_memory_alloc_interval_track_v4;
    interval_track_4_func_t m_kd_pmc_interval_track;
    interval_track_3_func_t m_stream_interval_track;

    stats_track_1_func_t m_region_stats_track_v4;
    stats_track_1_func_t m_kernel_dispatch_stats_track_v4;
    stats_track_1_func_t m_memory_copy_stats_track_v4;
    stats_track_1_func_t m_memory_alloc_stats_track_v4;
    stats_track_4_func_t m_kd_pmc_stats_track;
    stats_track_3_func_t m_stream_stats_track;
    stats_track_1_func_t m_scalar_stats;

    scalar_track_func_t      m_scalar_track;
    scalar_detail_func_t     m_scalar_detail;
    scalar_detail_func_t     m_pmc_event_detail;
    ambiguous_pmc_ids_func_t m_ambiguous_pmc_ids;

    flow_statement_set m_region_to_kernel_dispatch_flows;
    flow_statement_set m_region_to_memory_copy_flows;
    flow_statement_set m_region_to_memory_allocate_flows;
    flow_statement_set m_region_to_region_flows;
    flow_statement_set m_kernel_dispatch_sibling_flows;
    flow_statement_set m_memory_copy_sibling_flows;
    flow_statement_set m_memory_allocate_sibling_flows;

    timeline_event_statement_set m_region_statements;
    timeline_event_statement_set m_kernel_dispatch_statements;
    timeline_event_statement_set m_memory_allocate_statements;
    timeline_event_statement_set m_memory_copy_statements;

    count_func_t m_region_count;
    count_func_t m_kernel_dispatch_count;
    count_func_t m_memory_copy_count;
    count_func_t m_memory_alloc_count;

    count_time_filtered_func_t m_region_count_time_filtered;
    count_time_filtered_func_t m_kernel_dispatch_count_time_filtered;
    count_time_filtered_func_t m_memory_copy_count_time_filtered;
    count_time_filtered_func_t m_memory_alloc_count_time_filtered;

    event_id_func_t m_region_event_id;
    event_id_func_t m_kernel_dispatch_event_id;
    event_id_func_t m_memory_copy_event_id;
    event_id_func_t m_memory_alloc_event_id;

    arg_detail_func_t m_arg_detail;

    correlated_event_statement_set m_correlated_event_statements;

    region_detail_func_t          m_region_detail;
    kernel_dispatch_detail_func_t m_kernel_dispatch_detail;
    memory_copy_detail_func_t     m_memory_copy_detail;
    memory_alloc_detail_func_t    m_memory_alloc_detail;

    time_range_func_t m_region_time_range;
    time_range_func_t m_kernel_dispatch_time_range;
    time_range_func_t m_memory_copy_time_range;
    time_range_func_t m_memory_alloc_time_range;

    summary_func_t               m_kernel_summary;
    summary_time_filtered_func_t m_kernel_summary_time_filtered;
    summary_func_t               m_region_summary;
    summary_time_filtered_func_t m_region_summary_time_filtered;
};

}  // namespace profiler_hub::data_storage::schema_v4
