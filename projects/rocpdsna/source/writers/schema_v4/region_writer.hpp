// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "writers/region_writer.hpp"
#include "writers/schema_v4/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v4/insert_statements.hpp"
#include "data_storage/schema_version.hpp"
#include "rocpdsna/writer_types.hpp"

#include "spdlog/fmt/bundled/core.h"

#include <memory>
#include <optional>
#include <stdexcept>

namespace rocpdsna
{

/**
 * @brief Region writer for schema v4 (latest)
 *
 * Key differences from schema_v3:
 * - Uses track_id instead of nid/pid/tid columns
 * - Uses inline start/end BIGINT + phase instead of rocpd_timestamp FK references
 * - Uses name_id instead of direct name reference
 */
template <>
class region_writer<data_storage::schema_v4_tag>
: public region_writer_interface<region_writer<data_storage::schema_v4_tag>>
{
public:
    explicit region_writer(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v4::insert_statements<data_storage::sqlite_backend>>
                                                                               stmts,
        std::shared_ptr<common_insert_operations<data_storage::schema_v4_tag>> common_ops)
    : m_ctx(std::move(ctx))
    , m_stmts(std::move(stmts))
    , m_common_ops(std::move(common_ops))
    {}

    void insert_impl(const writer_types::region_data_t&       data,
                     const writer_types::trace_environment_t& trace_env)
    {
        auto transaction_block = m_ctx->backend->begin_transaction();
        // 1. Validate trace context
        m_ctx->validator->require_node(trace_env.node_id)
            .require_process(trace_env.process_id)
            .require_thread(trace_env.thread_id);
        // 2. Validate args require event data for correlation
        if(!data.event.has_value() && !data.args.empty())
        {
            throw std::runtime_error(fmt::format(
                "Writing args require providing event data for correlation: name: {}",
                data.name));
        }

        // 3. Ensure name is registered and get name_id
        m_common_ops->ensure_string_registered(data.name);
        const auto name_pk =
            m_ctx->registry->string_info().get_primary_key_value_for_entity(
                std::string(data.name));

        // 4. Get track_id from trace environment
        const auto track_pk = m_common_ops->resolve_track_id(trace_env);

        // 5. Insert event data and get event_id
        std::optional<primary_key_t> event_pk = std::nullopt;
        if(data.event.has_value())
        {
            event_pk = m_common_ops->insert_event(
                data.event.value(), trace_env.node_id, trace_env.process_id);
        }

        // 6. Insert region data with inline timestamps + phase
        const auto pk = m_ctx->key_providers->region_data().get_primary_key_value();

        // Phase: 1 = start/enter/load, 2 = end/exit/unload
        // v4 region: id, track_id, name_id, start, start_phase, end, end_phase,
        //            event_id, extdata
        m_stmts->region_statement()(pk,
                                    track_pk,
                                    name_pk,
                                    data.start_timestamp,
                                    1,
                                    data.end_timestamp,
                                    2,
                                    event_pk,
                                    data.extdata);

        if(event_pk.has_value())
        {
            for(const auto& arg : data.args)
            {
                m_common_ops->insert_arg(arg, event_pk.value());
            }
        }

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
