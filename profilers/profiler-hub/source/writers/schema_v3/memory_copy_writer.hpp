// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "writers/memory_copy_writer.hpp"
#include "writers/schema_v3/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v3/insert_statements.hpp"
#include "data_storage/schema_version.hpp"
#include "profiler-hub/writer_types.hpp"

#include <memory>
#include <optional>

namespace profiler_hub
{

template <>
class memory_copy_writer<data_storage::schema_v3_tag>
: public memory_copy_writer_interface<memory_copy_writer<data_storage::schema_v3_tag>>
{
public:
    explicit memory_copy_writer(
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

        const auto process_pk =
            m_ctx->validator->resolve_process_key(trace_env.process_id);
        const auto thread_pk =
            m_ctx->validator->resolve_optional_thread_key(trace_env.thread_id);
        const auto src_agent_pk =
            m_ctx->validator->resolve_optional_agent_key(data.src_agent_id);
        const auto dst_agent_pk =
            m_ctx->validator->resolve_optional_agent_key(data.dst_agent_id);
        const auto queue_pk =
            m_ctx->validator->resolve_optional_queue_key(trace_env.queue_id);
        const auto stream_pk =
            m_ctx->validator->resolve_optional_stream_key(trace_env.stream_id);
        const auto name_pk =
            m_ctx->registry->string_info().get_primary_key_value_for_entity(data.name);

        std::optional<primary_key_t> event_pk = std::nullopt;
        if(data.event.has_value())
        {
            event_pk = m_common_ops->insert_event(data.event.value());
        }

        std::optional<primary_key_t> region_name_pk = std::nullopt;
        if(data.region_name.has_value())
        {
            region_name_pk =
                m_ctx->registry->string_info().get_primary_key_value_for_entity(
                    data.region_name.value());
        }

        const auto pk = m_ctx->key_providers->memory_copy_data().get_primary_key_value();

        m_stmts->memory_copy_statement()(pk,
                                         trace_env.node_id.value(),
                                         process_pk,
                                         thread_pk,
                                         data.start_timestamp,
                                         data.end_timestamp,
                                         name_pk,
                                         dst_agent_pk,
                                         data.dst_address,
                                         src_agent_pk,
                                         data.src_address,
                                         data.size,
                                         queue_pk,
                                         stream_pk,
                                         region_name_pk,
                                         event_pk,
                                         data.extdata);

        m_common_ops->maybe_insert_sample(trace_env, data.start_timestamp, event_pk);
    }

private:
    std::shared_ptr<writer_context> m_ctx;
    std::shared_ptr<
        data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                           m_stmts;
    std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> m_common_ops;
};

}  // namespace profiler_hub
