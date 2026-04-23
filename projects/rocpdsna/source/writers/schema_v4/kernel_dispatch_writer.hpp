// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "writers/kernel_dispatch_writer.hpp"
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
 * @brief Kernel dispatch writer for schema v4 (latest)
 *
 * Key differences from schema_v3:
 * - Uses track_id instead of nid/pid/tid/agent_id/queue_id/stream_id columns
 * - Uses start_id/end_id (references rocpd_timestamp) instead of direct timestamps
 * - Uses region_name_id instead of name
 */
template <>
class kernel_dispatch_writer<data_storage::schema_v4_tag>
: public kernel_dispatch_writer_interface<
      kernel_dispatch_writer<data_storage::schema_v4_tag>>
{
public:
    explicit kernel_dispatch_writer(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v4::insert_statements<data_storage::sqlite_backend>>
                                                                               stmts,
        std::shared_ptr<common_insert_operations<data_storage::schema_v4_tag>> common_ops)
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

        // Get track_id from trace environment
        const auto track_pk = m_common_ops->resolve_track_id(trace_env);

        // Insert timestamps and get their IDs
        // Phase: 1 = start/enter/load, 2 = end/exit/unload (per SQL CHECK constraint)
        const auto start_id =
            m_common_ops->insert_timestamp(data.start_timestamp, 1, track_pk);
        const auto end_id =
            m_common_ops->insert_timestamp(data.end_timestamp, 2, track_pk);

        // Get region_name_id if name is provided
        std::optional<primary_key_t> region_name_pk =
            data.name.has_value()
                ? std::make_optional(
                      m_ctx->registry->string_info().get_primary_key_value_for_entity(
                          std::string(data.name.value())))
                : std::nullopt;

        std::optional<primary_key_t> event_pk = std::nullopt;
        if(data.event.has_value())
        {
            event_pk = m_common_ops->insert_event(
                data.event.value(), trace_env.node_id, trace_env.process_id);
        }

        const auto pk =
            m_ctx->key_providers->kernel_dispatch_data().get_primary_key_value();

        // v4 kernel_dispatch: id, track_id, kernel_id, dispatch_id, start_id, end_id,
        //                     private_segment_size, group_segment_size,
        //                     workgroup_size_x/y/z, grid_size_x/y/z,
        //                     region_name_id, event_id, extdata
        m_stmts->kernel_dispatch_statement()(pk,
                                             track_pk,
                                             data.kernel_symbol_id,
                                             data.dispatch_id,
                                             start_id,
                                             end_id,
                                             data.private_segment_size,
                                             data.group_segment_size,
                                             data.workgroup_size_x,
                                             data.workgroup_size_y,
                                             data.workgroup_size_z,
                                             data.grid_size_x,
                                             data.grid_size_y,
                                             data.grid_size_z,
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
