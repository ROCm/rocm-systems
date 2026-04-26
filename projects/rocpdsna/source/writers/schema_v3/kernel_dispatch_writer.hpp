// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "writers/kernel_dispatch_writer.hpp"
#include "writers/schema_v3/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v3/insert_statements.hpp"
#include "data_storage/schema_version.hpp"
#include "data_storage/vtable/kernel_dispatch_buffer.hpp"
#include "rocpdsna/writer_types.hpp"

#include "spdlog/fmt/bundled/core.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>

namespace rocpdsna
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
    {
        // Cache the buffer pointer at construction via the static registry
        // keyed by real table name. The buffer is owned by the vtable
        // instance and lives as long as the connection. Schema is
        // initialized before the writer is built, so the buffer must be
        // present; a missing entry is a wiring bug, not a runtime path.
        const auto real_table_name = fmt::format("rocpd_kernel_dispatch_{}", m_ctx->uuid);
        m_buffer = data_storage::vtable::kernel_dispatch_buffer::get_active_instance(
            real_table_name);
        if(m_buffer == nullptr)
        {
            throw std::runtime_error("kernel_dispatch buffer not registered for table " +
                                     real_table_name);
        }
    }

    void insert_impl(const writer_types::kernel_dispatch_data_t& data,
                     const writer_types::trace_environment_t&    trace_env)
    {
        // Guard the entire insert with a transaction so insert_event,
        // maybe_insert_sample, and any other intermediate writes commit
        // atomically (or roll back together) with the buffered dispatch row.
        // The vtable buffer's flush(), which actually persists the dispatch
        // row, runs against this same transaction's connection later.
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
                          std::string(data.name.value())))
                : std::nullopt;

        std::optional<primary_key_t> event_pk = std::nullopt;
        if(data.event.has_value())
        {
            event_pk = m_common_ops->insert_event(data.event.value());
        }

        const auto pk =
            m_ctx->key_providers->kernel_dispatch_data().get_primary_key_value();

        // Run all throwing operations first so the row commits to the buffer
        // only when the surrounding transaction would have committed too.
        // The buffer bypasses SQLite transactions, so any push that happens
        // before a throw cannot be rolled back.
        m_common_ops->maybe_insert_sample(trace_env, data.start_timestamp, event_pk);

        // Push columns straight into the buffer's per-column vectors,
        // skipping the SQLite vtable trampoline.
        auto to_opt_int64 = [](const std::optional<size_t>& v) {
            return v.has_value() ? std::make_optional(static_cast<int64_t>(*v))
                                 : std::nullopt;
        };

        m_buffer->push(data_storage::vtable::kernel_dispatch_row{
            .id                   = static_cast<int64_t>(pk),
            .nid                  = static_cast<int64_t>(trace_env.node_id.value()),
            .pid                  = static_cast<int64_t>(process_pk),
            .tid                  = to_opt_int64(thread_pk),
            .agent_id             = static_cast<int64_t>(agent_pk),
            .kernel_id            = static_cast<int64_t>(data.kernel_symbol_id),
            .dispatch_id          = static_cast<int64_t>(data.dispatch_id),
            .queue_id             = static_cast<int64_t>(queue_pk),
            .stream_id            = static_cast<int64_t>(stream_pk),
            .start                = static_cast<int64_t>(data.start_timestamp),
            .end                  = static_cast<int64_t>(data.end_timestamp),
            .private_segment_size = to_opt_int64(data.private_segment_size),
            .group_segment_size   = to_opt_int64(data.group_segment_size),
            .workgroup_size_x     = static_cast<int64_t>(data.workgroup_size_x),
            .workgroup_size_y     = static_cast<int64_t>(data.workgroup_size_y),
            .workgroup_size_z     = static_cast<int64_t>(data.workgroup_size_z),
            .grid_size_x          = static_cast<int64_t>(data.grid_size_x),
            .grid_size_y          = static_cast<int64_t>(data.grid_size_y),
            .grid_size_z          = static_cast<int64_t>(data.grid_size_z),
            .region_name_id       = to_opt_int64(name_pk),
            .event_id             = to_opt_int64(event_pk),
            .extdata              = data.extdata });
    }

private:
    std::shared_ptr<writer_context> m_ctx;
    std::shared_ptr<
        data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                           m_stmts;
    std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> m_common_ops;
    data_storage::vtable::kernel_dispatch_buffer* m_buffer = nullptr;
};

}  // namespace rocpdsna
