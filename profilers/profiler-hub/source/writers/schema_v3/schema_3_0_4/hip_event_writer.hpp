// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// rocpd_hip_event, added in 3.0.4. Schema 3.0.5 reuses this writer.

#include "writers/hip_event_writer.hpp"
#include "writers/schema_v3/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_version.hpp"
#include "queries/insert/table_insert_query.hpp"

#include <fmt/format.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace profiler_hub
{

template <>
class hip_event_writer<data_storage::schema_3_0_4_tag>
: public hip_event_writer_interface<hip_event_writer<data_storage::schema_3_0_4_tag>>
{
public:
    using statement_t =
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
                                                                size_t,
                                                                std::optional<size_t>,
                                                                std::optional<size_t>,
                                                                std::string_view>;

    using stmts_t =
        data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>;
    using common_ops_t = common_insert_operations<data_storage::schema_v3_tag>;

    explicit hip_event_writer(std::shared_ptr<writer_context> ctx,
                              std::shared_ptr<stmts_t>,
                              std::shared_ptr<common_ops_t> common_ops)
    : m_ctx(std::move(ctx))
    , m_common_ops(std::move(common_ops))
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        const auto                                        query =
            query_builder.set_table_name(fmt::format("rocpd_hip_event_{}", m_ctx->uuid))
                .set_columns("id",
                             "nid",
                             "pid",
                             "tid",
                             "start",
                             "end",
                             "name_id",
                             "agent_id",
                             "queue_id",
                             "stream_id",
                             "hip_event_handle",
                             "source_queue_id",
                             "event_id",
                             "extdata")
                .set_values(
                    '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
                .get_query_string();
        m_statement =
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
                                                           size_t,
                                                           std::optional<size_t>,
                                                           std::optional<size_t>,
                                                           std::string_view>(query);
    }

    void insert_impl(const writer_types::hip_event_data_t&    data,
                     const writer_types::trace_environment_t& trace_env)
    {
        auto transaction_block = m_ctx->backend->begin_transaction();

        m_ctx->validator->require_node(trace_env.node_id)
            .require_process(trace_env.process_id)
            .validate_optional_thread(trace_env.thread_id)
            .validate_optional_agent(trace_env.agent_id)
            .validate_optional_queue(trace_env.queue_id)
            .validate_optional_stream(trace_env.stream_id);

        m_common_ops->ensure_string_registered(data.name);

        const auto process_pk =
            m_ctx->validator->resolve_process_key(trace_env.process_id);
        const auto thread_pk =
            m_ctx->validator->resolve_optional_thread_key(trace_env.thread_id);
        const auto agent_pk =
            m_ctx->validator->resolve_optional_agent_key(trace_env.agent_id);
        const auto queue_pk =
            m_ctx->validator->resolve_optional_queue_key(trace_env.queue_id);
        const auto stream_pk =
            m_ctx->validator->resolve_optional_stream_key(trace_env.stream_id);
        const auto name_pk =
            m_ctx->registry->string_info().get_primary_key_value_for_entity(data.name);

        std::optional<primary_key_t> event_pk;
        if(data.event.has_value())
        {
            event_pk = m_common_ops->insert_event(data.event.value());
        }

        const auto pk = m_ctx->key_providers->hip_event_data().get_primary_key_value();
        m_statement(pk,
                    trace_env.node_id.value(),
                    process_pk,
                    thread_pk,
                    data.start_timestamp,
                    data.end_timestamp,
                    name_pk,
                    agent_pk,
                    queue_pk,
                    stream_pk,
                    data.hip_event_handle,
                    data.source_queue_id,
                    event_pk,
                    data.extdata);
    }

private:
    std::shared_ptr<writer_context>                                        m_ctx;
    std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> m_common_ops;
    statement_t                                                            m_statement;
};

}  // namespace profiler_hub
