// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "writers/memory_copy_writer.hpp"
#include "writers/schema_v3/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v3/insert_statements.hpp"
#include "data_storage/schema_version.hpp"
#include "data_storage/vtable/memory_copy_buffer.hpp"
#include "rocpdsna/writer_types.hpp"

#include "spdlog/fmt/bundled/core.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>

namespace rocpdsna
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
    {
        // Cache the per-connection buffer pointer to skip the vtable
        // trampoline on each insert. Schema is initialized before the
        // writer is built; a missing entry is a wiring bug.
        const auto real_table_name = fmt::format("rocpd_memory_copy_{}", m_ctx->uuid);
        m_buffer = data_storage::vtable::memory_copy_buffer::get_active_instance(
            real_table_name);
        if(m_buffer == nullptr)
        {
            throw std::runtime_error("memory_copy buffer not registered for table " +
                                     real_table_name);
        }
    }

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
            m_ctx->registry->string_info().get_primary_key_value_for_entity(
                std::string(data.name));

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
                    std::string(data.region_name.value()));
        }

        const auto pk = m_ctx->key_providers->memory_copy_data().get_primary_key_value();

        // Run all throwing operations first so the row commits to the buffer
        // only when the surrounding transaction would have committed too.
        // The buffer bypasses SQLite transactions, so any push that happens
        // before a throw cannot be rolled back.
        m_common_ops->maybe_insert_sample(trace_env, data.start_timestamp, event_pk);

        auto to_opt_int64 = [](const std::optional<size_t>& v) {
            return v.has_value() ? std::make_optional(static_cast<int64_t>(*v))
                                 : std::nullopt;
        };

        m_buffer->push(data_storage::vtable::memory_copy_row{
            .id             = static_cast<int64_t>(pk),
            .nid            = static_cast<int64_t>(trace_env.node_id.value()),
            .pid            = static_cast<int64_t>(process_pk),
            .tid            = to_opt_int64(thread_pk),
            .start          = static_cast<int64_t>(data.start_timestamp),
            .end            = static_cast<int64_t>(data.end_timestamp),
            .name_id        = static_cast<int64_t>(name_pk),
            .dst_agent_id   = to_opt_int64(dst_agent_pk),
            .dst_address    = to_opt_int64(data.dst_address),
            .src_agent_id   = to_opt_int64(src_agent_pk),
            .src_address    = to_opt_int64(data.src_address),
            .size           = static_cast<int64_t>(data.size),
            .queue_id       = to_opt_int64(queue_pk),
            .stream_id      = to_opt_int64(stream_pk),
            .region_name_id = to_opt_int64(region_name_pk),
            .event_id       = to_opt_int64(event_pk),
            .extdata        = data.extdata });
    }

private:
    std::shared_ptr<writer_context> m_ctx;
    std::shared_ptr<
        data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                           m_stmts;
    std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> m_common_ops;
    data_storage::vtable::memory_copy_buffer* m_buffer = nullptr;
};

}  // namespace rocpdsna
