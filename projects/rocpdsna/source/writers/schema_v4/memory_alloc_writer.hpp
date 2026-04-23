// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "writers/memory_alloc_writer.hpp"
#include "writers/schema_v4/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v4/insert_statements.hpp"
#include "data_storage/schema_version.hpp"
#include "rocpdsna/writer_types.hpp"

#include "spdlog/fmt/bundled/core.h"

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace rocpdsna
{

/**
 * @brief Memory allocation writer for schema v4 (latest)
 *
 * Key differences from schema_v3:
 * - Uses track_id instead of nid/pid/tid columns
 * - Uses start_id/end_id (references rocpd_timestamp) instead of direct timestamps
 * - Uses name_id, region_name_id for string references
 */
template <>
class memory_alloc_writer<data_storage::schema_v4_tag>
: public memory_alloc_writer_interface<memory_alloc_writer<data_storage::schema_v4_tag>>
{
public:
    explicit memory_alloc_writer(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v4::insert_statements<data_storage::sqlite_backend>>
                                                                               stmts,
        std::shared_ptr<common_insert_operations<data_storage::schema_v4_tag>> common_ops)
    : m_ctx(std::move(ctx))
    , m_stmts(std::move(stmts))
    , m_common_ops(std::move(common_ops))
    {}

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

        // Get track_id from trace environment
        const auto track_pk = m_common_ops->resolve_track_id(trace_env);

        // Insert timestamps and get their IDs
        // Phase: 1 = start/enter/load, 2 = end/exit/unload (per SQL CHECK constraint)
        const auto start_id =
            m_common_ops->insert_timestamp(data.start_timestamp, 1, track_pk);
        const auto end_id =
            m_common_ops->insert_timestamp(data.end_timestamp, 2, track_pk);

        // name_id is required: prefer user-provided name, fall back to type, then generic
        std::string_view name_str = data.name.has_value()
                                        ? data.name.value()
                                        : data.type.value_or("memory_allocate");
        const auto       name_pk  = m_common_ops->insert_string(name_str);

        // region_name_id: resolve from user-provided region_name if present
        m_common_ops->ensure_optional_string_registered(data.region_name);
        std::optional<primary_key_t> region_name_pk = std::nullopt;
        if(data.region_name.has_value())
        {
            region_name_pk =
                m_ctx->registry->string_info().get_primary_key_value_for_entity(
                    std::string(data.region_name.value()));
        }

        std::optional<primary_key_t> event_pk = std::nullopt;
        if(data.event.has_value())
        {
            event_pk = m_common_ops->insert_event(
                data.event.value(), trace_env.node_id, trace_env.process_id);
        }

        const auto pk = m_ctx->key_providers->memory_alloc_data().get_primary_key_value();

        // v4 memory_allocate: id, track_id, type, level, start_id, end_id, name_id,
        //                     address, size, region_name_id, event_id, extdata
        m_stmts->memory_alloc_statement()(pk,
                                          track_pk,
                                          data.type,
                                          data.level,
                                          start_id,
                                          end_id,
                                          name_pk,
                                          data.address,
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
