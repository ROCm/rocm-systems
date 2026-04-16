// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "writers/memory_copy_writer.hpp"
#include "writers/schema_v4/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v4/insert_statements.hpp"
#include "data_storage/schema_version.hpp"
#include "rocpdsna/writer_types.hpp"

#include "spdlog/fmt/bundled/core.h"

#include <memory>
#include <optional>

namespace rocpdsna
{

/**
 * @brief Memory copy writer for schema v4 (latest)
 *
 * Key differences from schema_v3:
 * - Uses track_id instead of nid/pid/tid columns
 * - Uses start_id/end_id (references rocpd_timestamp) instead of direct timestamps
 * - Uses name_id, region_name_id for string references
 */
template <>
class memory_copy_writer<data_storage::schema_v4_tag>
: public memory_copy_writer_interface<memory_copy_writer<data_storage::schema_v4_tag>>
{
public:
    explicit memory_copy_writer(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v4::insert_statements<data_storage::sqlite_backend>>
                                                                               stmts,
        std::shared_ptr<common_insert_operations<data_storage::schema_v4_tag>> common_ops)
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

        // Get track_id from trace environment
        const auto track_pk = m_common_ops->resolve_track_id(trace_env);

        // Insert timestamps and get their IDs
        // Phase: 1 = start/enter/load, 2 = end/exit/unload (per SQL CHECK constraint)
        const auto start_id =
            m_common_ops->insert_timestamp(data.start_timestamp, 1, track_pk);
        const auto end_id =
            m_common_ops->insert_timestamp(data.end_timestamp, 2, track_pk);

        // Get name_id
        const auto name_pk =
            m_ctx->registry->string_info().get_primary_key_value_for_entity(
                std::string(data.name));

        // Get agent keys for src/dst
        const auto src_agent_pk =
            m_ctx->validator->resolve_optional_agent_key(data.src_agent_id);
        const auto dst_agent_pk =
            m_ctx->validator->resolve_optional_agent_key(data.dst_agent_id);

        std::optional<primary_key_t> event_pk = std::nullopt;
        if(data.event.has_value())
        {
            event_pk = m_common_ops->insert_event(
                data.event.value(), trace_env.node_id, trace_env.process_id);
        }

        std::optional<primary_key_t> region_name_pk = std::nullopt;
        if(data.region_name.has_value())
        {
            region_name_pk =
                m_ctx->registry->string_info().get_primary_key_value_for_entity(
                    std::string(data.region_name.value()));
        }

        const auto pk = m_ctx->key_providers->memory_copy_data().get_primary_key_value();

        // v4 memory_copy: id, track_id, start_id, end_id, name_id,
        //                 dst_agent_id, dst_address, src_agent_id, src_address,
        //                 size, region_name_id, event_id, extdata
        m_stmts->memory_copy_statement()(pk,
                                         track_pk,
                                         start_id,
                                         end_id,
                                         name_pk,
                                         dst_agent_pk,
                                         data.dst_address,
                                         src_agent_pk,
                                         data.src_address,
                                         data.size,
                                         region_name_pk,
                                         event_pk,
                                         data.extdata);

        m_common_ops->maybe_insert_sample(trace_env, data.start_timestamp, event_pk);
    }

private:
    std::shared_ptr<writer_context> m_ctx;
    std::shared_ptr<
        data_storage::schema_v4::insert_statements<data_storage::sqlite_backend>>
                                                                           m_stmts;
    std::shared_ptr<common_insert_operations<data_storage::schema_v4_tag>> m_common_ops;
};

}  // namespace rocpdsna
