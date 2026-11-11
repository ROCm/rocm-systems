// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// The only kernel-dispatch change in 3.0.2: graph_exec_id and graph_node_id.
// Validation, key lookup, and the event/sample rows stay in the 3.0.0 writer.
// Schemas 3.0.3, 3.0.4, and 3.0.5 reuse this writer. The table did not change again.

#include "writers/common/kernel_dispatch_writer.hpp"
#include "writers/kernel_dispatch_writer.hpp"

#include "data_storage/schema_version.hpp"
#include "queries/insert/table_insert_query.hpp"

#include <fmt/format.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace profiler_hub
{

template <>
class kernel_dispatch_writer<data_storage::schema_3_0_2_tag>
: public kernel_dispatch_common<kernel_dispatch_writer<data_storage::schema_3_0_2_tag>>
{
public:
    using base_t =
        kernel_dispatch_common<kernel_dispatch_writer<data_storage::schema_3_0_2_tag>>;

    using graph_statement_t =
        data_storage::sqlite_backend::prepared_insert_statement<size_t,
                                                                size_t,
                                                                size_t,
                                                                std::optional<size_t>,
                                                                size_t,
                                                                size_t,
                                                                size_t,
                                                                size_t,
                                                                size_t,
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
                                                                size_t,
                                                                size_t,
                                                                std::optional<size_t>,
                                                                std::optional<size_t>,
                                                                std::string_view>;

    explicit kernel_dispatch_writer(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                               stmts,
        std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> common_ops)
    : base_t(std::move(ctx), std::move(stmts), std::move(common_ops))
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        const auto                                        query =
            query_builder
                .set_table_name(fmt::format("rocpd_kernel_dispatch_{}", m_ctx->uuid))
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
                             "graph_exec_id",
                             "graph_node_id",
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
                            '?',
                            '?',
                            '?')
                .get_query_string();
        m_graph_statement =
            m_ctx->backend
                ->template create_write_statement_executor<size_t,
                                                           size_t,
                                                           size_t,
                                                           std::optional<size_t>,
                                                           size_t,
                                                           size_t,
                                                           size_t,
                                                           size_t,
                                                           size_t,
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
                                                           size_t,
                                                           size_t,
                                                           std::optional<size_t>,
                                                           std::optional<size_t>,
                                                           std::string_view>(query);
    }

    void reject_unsupported_fields(const writer_types::kernel_dispatch_data_t&) const {}

    void write_kernel_dispatch(const writer_types::kernel_dispatch_data_t& data,
                               const writer_types::trace_environment_t&    trace_env,
                               const kernel_dispatch_insert&               row)
    {
        m_graph_statement(row.pk,
                          trace_env.node_id.value(),
                          row.process_pk,
                          row.thread_pk,
                          row.agent_pk,
                          data.kernel_symbol_id,
                          data.dispatch_id,
                          row.queue_pk,
                          row.stream_pk,
                          data.start_timestamp,
                          data.end_timestamp,
                          data.private_segment_size,
                          data.group_segment_size,
                          data.workgroup_size_x,
                          data.workgroup_size_y,
                          data.workgroup_size_z,
                          data.grid_size_x,
                          data.grid_size_y,
                          data.grid_size_z,
                          data.graph_exec_id.value_or(0),
                          data.graph_node_id.value_or(0),
                          row.name_pk,
                          row.event_pk,
                          data.extdata);
    }

private:
    graph_statement_t m_graph_statement;
};

}  // namespace profiler_hub
