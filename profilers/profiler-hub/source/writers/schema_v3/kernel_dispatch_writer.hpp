// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "writers/kernel_dispatch_writer.hpp"
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
class kernel_dispatch_writer<data_storage::schema_v3_tag>
: public kernel_dispatch_writer_interface<
      kernel_dispatch_writer<data_storage::schema_v3_tag>>
{
public:
    explicit kernel_dispatch_writer(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                               stmts,
        std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> common_ops)
    : m_ctx(std::move(ctx))
    , m_stmts(std::move(stmts))
    , m_common_ops(std::move(common_ops))
    {}

    void insert_impl(const writer_types::kernel_dispatch_data_t& data,
                     const writer_types::trace_environment_t&    trace_env)
    {
        auto transaction_block = m_ctx->backend->begin_transaction();

        m_ctx->validator->require_node(trace_env.node_id)
            .require_process(trace_env.process_id)
            .require_thread(trace_env.thread_id)
            .require_agent(trace_env.agent_id)
            .require_queue(trace_env.queue_id)
            .require_stream(trace_env.stream_id)
            .require_kernel_symbol(data.kernel_symbol_id);

        m_common_ops->ensure_optional_string_registered(data.name);

        const auto process_pk =
            m_ctx->validator->resolve_process_key(trace_env.process_id);
        const auto thread_pk =
            m_ctx->validator->resolve_optional_thread_key(trace_env.thread_id);
        const auto agent_pk  = m_ctx->validator->resolve_agent_key(trace_env.agent_id);
        const auto queue_pk  = m_ctx->validator->resolve_queue_key(trace_env.queue_id);
        const auto stream_pk = m_ctx->validator->resolve_stream_key(trace_env.stream_id);
        const auto name_pk =
            data.name.has_value()
                ? std::make_optional(
                      m_ctx->registry->string_info().get_primary_key_value_for_entity(
                          data.name.value()))
                : std::nullopt;

        std::optional<primary_key_t> event_pk = std::nullopt;
        if(data.event.has_value())
        {
            event_pk = m_common_ops->insert_event(data.event.value());
        }

        const auto pk =
            m_ctx->key_providers->kernel_dispatch_data().get_primary_key_value();

        m_stmts->kernel_dispatch_statement()(pk,
                                             trace_env.node_id.value(),
                                             process_pk,
                                             thread_pk,
                                             agent_pk,
                                             data.kernel_symbol_id,
                                             data.dispatch_id,
                                             queue_pk,
                                             stream_pk,
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
                                             name_pk,
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
