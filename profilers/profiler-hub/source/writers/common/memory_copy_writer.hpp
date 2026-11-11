// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "writers/interfaces/memory_copy_writer_interface.hpp"
#include "writers/schema_v3/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v3/insert_statements.hpp"
#include "profiler-hub/writer_types.hpp"

#include <memory>
#include <optional>

namespace profiler_hub
{

struct memory_copy_insert
{
    primary_key_t                pk{};
    primary_key_t                process_pk{};
    std::optional<primary_key_t> thread_pk;
    std::optional<primary_key_t> src_agent_pk;
    std::optional<primary_key_t> dst_agent_pk;
    std::optional<primary_key_t> queue_pk;
    std::optional<primary_key_t> stream_pk;
    primary_key_t                name_pk{};
    std::optional<primary_key_t> region_name_pk;
    std::optional<primary_key_t> event_pk;
};

// Schema 3.0.0 memory copy. 3.0.2 adds graph_exec_id and graph_node_id.
template <typename Derived>
class memory_copy_common : public memory_copy_writer_interface<Derived>
{
public:
    explicit memory_copy_common(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                               stmts,
        std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> common_ops)
    : m_ctx(std::move(ctx))
    , m_stmts(std::move(stmts))
    , m_common_ops(std::move(common_ops))
    {}

    void insert_impl(const writer_types::memory_copy_data_t&  data,
                     const writer_types::trace_environment_t& trace_env)
    {
        this->self().reject_unsupported_fields(data);

        auto transaction_block = m_ctx->backend->begin_transaction();

        m_ctx->validator->require_node(trace_env.node_id)
            .require_process(trace_env.process_id)
            .validate_optional_thread(trace_env.thread_id)
            .validate_optional_agent(data.src_agent_id, "Source agent")
            .validate_optional_agent(data.dst_agent_id, "Destination agent")
            .validate_optional_queue(trace_env.queue_id)
            .validate_optional_stream(trace_env.stream_id);

        m_common_ops->ensure_string_registered(data.name);
        m_common_ops->ensure_optional_string_registered(data.region_name);

        memory_copy_insert row;
        row.process_pk = m_ctx->validator->resolve_process_key(trace_env.process_id);
        row.thread_pk = m_ctx->validator->resolve_optional_thread_key(trace_env.thread_id);
        row.src_agent_pk = m_ctx->validator->resolve_optional_agent_key(data.src_agent_id);
        row.dst_agent_pk = m_ctx->validator->resolve_optional_agent_key(data.dst_agent_id);
        row.queue_pk  = m_ctx->validator->resolve_optional_queue_key(trace_env.queue_id);
        row.stream_pk = m_ctx->validator->resolve_optional_stream_key(trace_env.stream_id);
        row.name_pk =
            m_ctx->registry->string_info().get_primary_key_value_for_entity(data.name);

        if(data.event.has_value())
        {
            row.event_pk = m_common_ops->insert_event(data.event.value());
        }
        if(data.region_name.has_value())
        {
            row.region_name_pk =
                m_ctx->registry->string_info().get_primary_key_value_for_entity(
                    data.region_name.value());
        }

        row.pk = m_ctx->key_providers->memory_copy_data().get_primary_key_value();
        this->self().write_memory_copy(data, trace_env, row);
        m_common_ops->maybe_insert_sample(trace_env, data.start_timestamp, row.event_pk);
    }

protected:
    std::shared_ptr<writer_context> m_ctx;
    std::shared_ptr<
        data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                           m_stmts;
    std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> m_common_ops;
};

}  // namespace profiler_hub
