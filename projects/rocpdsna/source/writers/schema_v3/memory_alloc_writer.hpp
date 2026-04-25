// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "writers/memory_alloc_writer.hpp"
#include "writers/schema_v3/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v3/insert_statements.hpp"
#include "data_storage/schema_version.hpp"
#include "data_storage/vtable/memory_alloc_buffer.hpp"
#include "rocpdsna/writer_types.hpp"

#include "spdlog/fmt/bundled/core.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace rocpdsna
{

template <>
class memory_alloc_writer<data_storage::schema_v3_tag>
: public memory_alloc_writer_interface<memory_alloc_writer<data_storage::schema_v3_tag>>
{
public:
    explicit memory_alloc_writer(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                               stmts,
        std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> common_ops)
    : m_ctx(std::move(ctx))
    , m_stmts(std::move(stmts))
    , m_common_ops(std::move(common_ops))
    {
        const auto real_table_name = fmt::format("rocpd_memory_allocate_{}", m_ctx->uuid);
        m_buffer = data_storage::vtable::memory_alloc_buffer::get_active_instance(
            real_table_name);
    }

    void insert_impl(const writer_types::memory_alloc_data_t& data,
                     const writer_types::trace_environment_t& trace_env)
    {
        auto transaction_block = m_ctx->backend->begin_transaction();

        m_ctx->validator->require_node(trace_env.node_id)
            .require_process(trace_env.process_id)
            .validate_optional_thread(trace_env.thread_id)
            .validate_optional_agent(trace_env.agent_id)
            .validate_optional_queue(trace_env.queue_id)
            .validate_optional_stream(trace_env.stream_id);

        if(data.type.has_value())
        {
            constexpr std::array<std::string_view, 4> allowed_types = {
                "ALLOC", "FREE", "REALLOC", "RECLAIM"
            };
            if(std::find(allowed_types.begin(), allowed_types.end(), data.type.value()) ==
               allowed_types.end())
            {
                throw std::runtime_error(fmt::format(
                    "Invalid type value for Memory Alloc Data: type: {}. Allowed: "
                    "ALLOC, FREE, REALLOC, RECLAIM",
                    data.type.value()));
            }
        }

        if(data.level.has_value())
        {
            constexpr std::array<std::string_view, 3> allowed_levels = { "REAL",
                                                                         "VIRTUAL",
                                                                         "SCRATCH" };
            if(std::find(allowed_levels.begin(),
                         allowed_levels.end(),
                         data.level.value()) == allowed_levels.end())
            {
                throw std::runtime_error(fmt::format(
                    "Invalid level value for Memory Alloc Data: level: {}. Allowed: "
                    "REAL, VIRTUAL, SCRATCH",
                    data.level.value()));
            }
        }

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

        std::optional<primary_key_t> event_pk = std::nullopt;
        if(data.event.has_value())
        {
            event_pk = m_common_ops->insert_event(data.event.value());
        }

        const auto pk = m_ctx->key_providers->memory_alloc_data().get_primary_key_value();

        if(m_buffer != nullptr)
        {
            auto to_opt_int64 = [](const std::optional<size_t>& v) {
                return v.has_value() ? std::make_optional(static_cast<int64_t>(*v))
                                     : std::nullopt;
            };

            m_buffer->push(static_cast<int64_t>(pk),
                           static_cast<int64_t>(trace_env.node_id.value()),
                           static_cast<int64_t>(process_pk),
                           to_opt_int64(thread_pk),
                           to_opt_int64(agent_pk),
                           data.type,
                           data.level,
                           static_cast<int64_t>(data.start_timestamp),
                           static_cast<int64_t>(data.end_timestamp),
                           to_opt_int64(data.address),
                           static_cast<int64_t>(data.size),
                           to_opt_int64(queue_pk),
                           to_opt_int64(stream_pk),
                           to_opt_int64(event_pk),
                           data.extdata);
        }
        else
        {
            m_stmts->memory_alloc_statement()(pk,
                                              trace_env.node_id.value(),
                                              process_pk,
                                              thread_pk,
                                              agent_pk,
                                              data.type,
                                              data.level,
                                              data.start_timestamp,
                                              data.end_timestamp,
                                              data.address,
                                              data.size,
                                              queue_pk,
                                              stream_pk,
                                              event_pk,
                                              data.extdata);
        }

        m_common_ops->maybe_insert_sample(trace_env, data.start_timestamp, event_pk);
    }

private:
    std::shared_ptr<writer_context> m_ctx;
    std::shared_ptr<
        data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                           m_stmts;
    std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> m_common_ops;
    data_storage::vtable::memory_alloc_buffer* m_buffer = nullptr;
};

}  // namespace rocpdsna
