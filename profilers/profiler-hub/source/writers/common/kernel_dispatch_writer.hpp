// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "writers/interfaces/kernel_dispatch_writer_interface.hpp"
#include "writers/schema_v3/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v3/insert_statements.hpp"
#include "profiler-hub/writer_types.hpp"

#include <memory>
#include <optional>

namespace profiler_hub
{

// Resolved keys shared by every schema's kernel-dispatch insert. A version
// writer only decides which columns to bind.
struct kernel_dispatch_insert
{
    primary_key_t                pk{};
    primary_key_t                process_pk{};
    std::optional<primary_key_t> thread_pk;
    primary_key_t                agent_pk{};
    primary_key_t                queue_pk{};
    primary_key_t                stream_pk{};
    std::optional<primary_key_t> name_pk;
    std::optional<primary_key_t> event_pk;
};

// Schema 3.0.0 kernel-dispatch insert. Later schemas reuse this and override
// only the column bind.
template <typename Derived>
class kernel_dispatch_common : public kernel_dispatch_writer_interface<Derived>
{
public:
    explicit kernel_dispatch_common(
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
        this->self().reject_unsupported_fields(data);

        auto transaction_block = m_ctx->backend->begin_transaction();

        m_ctx->validator->require_node(trace_env.node_id)
            .require_process(trace_env.process_id)
            .require_thread(trace_env.thread_id)
            .require_agent(trace_env.agent_id)
            .require_queue(trace_env.queue_id)
            .require_stream(trace_env.stream_id)
            .require_kernel_symbol(data.kernel_symbol_id);

        m_common_ops->ensure_optional_string_registered(data.name);

        kernel_dispatch_insert row;
        row.process_pk = m_ctx->validator->resolve_process_key(trace_env.process_id);
        row.thread_pk =
            m_ctx->validator->resolve_optional_thread_key(trace_env.thread_id);
        row.agent_pk  = m_ctx->validator->resolve_agent_key(trace_env.agent_id);
        row.queue_pk  = m_ctx->validator->resolve_queue_key(trace_env.queue_id);
        row.stream_pk = m_ctx->validator->resolve_stream_key(trace_env.stream_id);
        row.name_pk =
            data.name.has_value()
                ? std::make_optional(
                      m_ctx->registry->string_info().get_primary_key_value_for_entity(
                          data.name.value()))
                : std::nullopt;

        if(data.event.has_value())
        {
            row.event_pk = m_common_ops->insert_event(data.event.value());
        }

        row.pk = m_ctx->key_providers->kernel_dispatch_data().get_primary_key_value();

        this->self().write_kernel_dispatch(data, trace_env, row);

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
