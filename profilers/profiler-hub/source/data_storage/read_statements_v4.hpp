// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/sqlite_backend.hpp"
#include "read_statements_base.hpp"

#include "profiler-hub/reader_types.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace profiler_hub::data_storage::schema_v4
{

// v4.0 read backend. Implements the track-scoped reader subset
// (get_interval_track / get_scalar_track / get_flows + scalar detail overloads,
// plus the info tables and counter-name/scalar-track statements the reader
// needs) against the v4.0 rocpd schema.
//
// v4.0 differs from v3 in two structural ways that drive the SQL here:
//   * rocpd_track is the universal identity anchor — every event table carries a
//     single track_id FK, so all track-scoped queries reduce to WHERE track_id = ?.
//   * timestamps are normalized into the rocpd_timestamp spine — interval tables
//     carry start_id/end_id FKs and rocpd_sample carries timestamp_id, so reading
//     an actual time requires a JOIN onto rocpd_timestamp.value.
//
// The legacy timeline/detail/call-stack/correlated/count/time-range surface is
// intentionally NOT implemented here (task 002C). Those accessors inherit the
// default-empty stubs from read_statements_base; the reader guards every v4 path
// so an empty stub is never invoked.
//
// Table naming: this backend reuses the v3 reader convention `rocpd_<name>_<uuid>`
// (underscore separator supplied by the reader). See the open question in
// tasks/002B-result.md about the v4.0 DDL template placeholder `{{uuid}}`.
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

    // ----- info table accessors (return by value, matching base) -----
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

    // ----- counter / scalar accessors -----
    [[nodiscard]] const sample_track_id_func_t& distinct_sample_track_ids() const override
    {
        return m_distinct_sample_track_ids;
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

    // ----- track_id-anchored interval accessors (v4.0-specific) -----
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

        // Counter tracks: any track_id referenced by a rocpd_sample row.
        m_distinct_sample_track_ids =
            m_backend->create_read_statement_executor<sample_track_id_result>(
                fmt::format("SELECT DISTINCT track_id FROM rocpd_sample_{}", u),
                &sample_track_id_result::track_id);

        // Map each counter track_id to its PMC name.
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

        // region intervals: start/end resolved through the timestamp spine.
        m_region_interval_track_v4 =
            m_backend
                ->create_read_statement_executor<interval_row_result, bind_types<size_t>>(
                    fmt::format("SELECT r.id, ts_s.value, ts_e.value, r.name_id "
                                "FROM rocpd_region_{u} r "
                                "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = r.start_id "
                                "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = r.end_id "
                                "WHERE r.track_id = ? ORDER BY ts_s.value",
                                fmt::arg("u", u)),
                    &interval_row_result::id,
                    &interval_row_result::start,
                    &interval_row_result::end,
                    &interval_row_result::name_ref);

        // kernel dispatch intervals: name_ref is the kernel_symbol id.
        m_kernel_dispatch_interval_track_v4 =
            m_backend
                ->create_read_statement_executor<interval_row_result, bind_types<size_t>>(
                    fmt::format("SELECT k.id, ts_s.value, ts_e.value, k.kernel_id "
                                "FROM rocpd_kernel_dispatch_{u} k "
                                "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = k.start_id "
                                "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = k.end_id "
                                "WHERE k.track_id = ? ORDER BY ts_s.value",
                                fmt::arg("u", u)),
                    &interval_row_result::id,
                    &interval_row_result::start,
                    &interval_row_result::end,
                    &interval_row_result::name_ref);

        // memory copy intervals.
        m_memory_copy_interval_track_v4 =
            m_backend
                ->create_read_statement_executor<interval_row_result, bind_types<size_t>>(
                    fmt::format("SELECT mc.id, ts_s.value, ts_e.value, mc.name_id "
                                "FROM rocpd_memory_copy_{u} mc "
                                "JOIN rocpd_timestamp_{u} ts_s ON ts_s.id = mc.start_id "
                                "JOIN rocpd_timestamp_{u} ts_e ON ts_e.id = mc.end_id "
                                "WHERE mc.track_id = ? ORDER BY ts_s.value",
                                fmt::arg("u", u)),
                    &interval_row_result::id,
                    &interval_row_result::start,
                    &interval_row_result::end,
                    &interval_row_result::name_ref);
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

        // Keyed on rocpd_sample.id (scalar_event_t::opaque_id).
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

    void initialize_flow_statements()
    {
        const auto& u = m_uuid;

        // source = CPU region; dest = a GPU-side event sharing the same stack_id.
        // Structurally identical to v3 (rocpd_event still carries stack_id); the
        // only v4 difference is the time filter, which must resolve the source
        // start time through the rocpd_timestamp spine.
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

            // Optional time window applied to the SOURCE event's start timestamp,
            // resolved via the rocpd_timestamp spine.
            const auto time_sql =
                fmt::format("SELECT R.id, {d}.id "
                            "FROM rocpd_region_{u} R "
                            "JOIN rocpd_event_{u} ER ON R.event_id = ER.id "
                            "JOIN rocpd_event_{u} E{d} "
                            "  ON E{d}.stack_id = ER.stack_id AND E{d}.id != ER.id "
                            "JOIN {dt}_{u} {d} ON {d}.event_id = E{d}.id "
                            "JOIN rocpd_timestamp_{u} TSR ON TSR.id = R.start_id "
                            "WHERE ER.stack_id IS NOT NULL AND ER.stack_id != 0 "
                            "  AND TSR.value >= ? AND TSR.value <= ?",
                            fmt::arg("u", u),
                            fmt::arg("d", dest_alias),
                            fmt::arg("dt", dest_table));

            out.time_filtered =
                m_backend->create_read_statement_executor<flow_row_result,
                                                          bind_types<size_t, size_t>>(
                    time_sql, &flow_row_result::source_id, &flow_row_result::dest_id);
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

    sample_track_id_func_t    m_distinct_sample_track_ids;
    counter_track_name_func_t m_counter_track_names;

    interval_track_1_func_t m_region_interval_track_v4;
    interval_track_1_func_t m_kernel_dispatch_interval_track_v4;
    interval_track_1_func_t m_memory_copy_interval_track_v4;

    scalar_track_func_t  m_scalar_track;
    scalar_detail_func_t m_scalar_detail;
    scalar_detail_func_t m_pmc_event_detail;

    flow_statement_set m_region_to_kernel_dispatch_flows;
    flow_statement_set m_region_to_memory_copy_flows;
    flow_statement_set m_region_to_memory_allocate_flows;
};

}  // namespace profiler_hub::data_storage::schema_v4
