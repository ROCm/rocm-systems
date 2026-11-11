// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// 3.0.2 memory copy stores graph_exec_id and graph_node_id. Later schemas reuse this.

#include "writers/common/memory_copy_writer.hpp"
#include "writers/memory_copy_writer.hpp"

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
class memory_copy_writer<data_storage::schema_3_0_2_tag>
: public memory_copy_common<memory_copy_writer<data_storage::schema_3_0_2_tag>>
{
public:
    using base_t = memory_copy_common<memory_copy_writer<data_storage::schema_3_0_2_tag>>;

    using graph_statement_t =
        data_storage::sqlite_backend::prepared_insert_statement<size_t,
                                                                size_t,
                                                                size_t,
                                                                std::optional<size_t>,
                                                                uint64_t,
                                                                uint64_t,
                                                                size_t,
                                                                std::optional<size_t>,
                                                                std::optional<size_t>,
                                                                std::optional<size_t>,
                                                                std::optional<size_t>,
                                                                size_t,
                                                                std::optional<size_t>,
                                                                std::optional<size_t>,
                                                                size_t,
                                                                size_t,
                                                                std::optional<size_t>,
                                                                std::optional<size_t>,
                                                                std::string_view>;

    explicit memory_copy_writer(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                               stmts,
        std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> common_ops)
    : base_t(std::move(ctx), std::move(stmts), std::move(common_ops))
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        const auto                                        query =
            query_builder.set_table_name(fmt::format("rocpd_memory_copy_{}", m_ctx->uuid))
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
                            '?')
                .get_query_string();
        m_graph_statement =
            m_ctx->backend
                ->template create_write_statement_executor<size_t,
                                                           size_t,
                                                           size_t,
                                                           std::optional<size_t>,
                                                           uint64_t,
                                                           uint64_t,
                                                           size_t,
                                                           std::optional<size_t>,
                                                           std::optional<size_t>,
                                                           std::optional<size_t>,
                                                           std::optional<size_t>,
                                                           size_t,
                                                           std::optional<size_t>,
                                                           std::optional<size_t>,
                                                           size_t,
                                                           size_t,
                                                           std::optional<size_t>,
                                                           std::optional<size_t>,
                                                           std::string_view>(query);
    }

    void reject_unsupported_fields(const writer_types::memory_copy_data_t&) const {}

    void write_memory_copy(const writer_types::memory_copy_data_t&  data,
                           const writer_types::trace_environment_t& trace_env,
                           const memory_copy_insert&                row)
    {
        m_graph_statement(row.pk,
                          trace_env.node_id.value(),
                          row.process_pk,
                          row.thread_pk,
                          data.start_timestamp,
                          data.end_timestamp,
                          row.name_pk,
                          row.dst_agent_pk,
                          data.dst_address,
                          row.src_agent_pk,
                          data.src_address,
                          data.size,
                          row.queue_pk,
                          row.stream_pk,
                          data.graph_exec_id.value_or(0),
                          data.graph_node_id.value_or(0),
                          row.region_name_pk,
                          row.event_pk,
                          data.extdata);
    }

private:
    graph_statement_t m_graph_statement;
};

}  // namespace profiler_hub
