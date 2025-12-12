// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "writers/region_writer.hpp"
#include "writers/schema_v3/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v3/insert_statements.hpp"
#include "data_storage/schema_version.hpp"
#include "profiler-hub/writer_types.hpp"

#include <fmt/core.h>

#include <memory>
#include <optional>
#include <stdexcept>

namespace profiler_hub
{

template <>
class region_writer<data_storage::schema_v3_tag>
: public region_writer_interface<region_writer<data_storage::schema_v3_tag>>
{
public:
    explicit region_writer(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                               stmts,
        std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> common_ops)
    : m_ctx(std::move(ctx))
    , m_stmts(std::move(stmts))
    , m_common_ops(std::move(common_ops))
    {}

    void insert_impl(const writer_types::region_data_t&       data,
                     const writer_types::trace_environment_t& trace_env)
    {
        auto transaction_block = m_ctx->backend->begin_transaction();

        m_ctx->validator->require_node(trace_env.node_id)
            .require_process(trace_env.process_id)
            .require_thread(trace_env.thread_id);

        if(!data.event.has_value() && !data.args.empty())
        {
            throw std::runtime_error(fmt::format(
                "Writing args require providing event data for correlation: name: {}",
                data.name));
        }

        m_common_ops->ensure_string_registered(data.name);

        const auto process_pk =
            m_ctx->validator->resolve_process_key(trace_env.process_id);
        const auto thread_pk = m_ctx->validator->resolve_thread_key(trace_env.thread_id);
        const auto name_pk =
            m_ctx->registry->string_info().get_primary_key_value_for_entity(data.name);

        std::optional<primary_key_t> event_pk = std::nullopt;
        if(data.event.has_value())
        {
            event_pk = m_common_ops->insert_event(data.event.value());
        }

        const auto pk = m_ctx->key_providers->region_data().get_primary_key_value();

        m_stmts->region_statement()(pk,
                                    trace_env.node_id.value(),
                                    process_pk,
                                    thread_pk,
                                    data.start_timestamp,
                                    data.end_timestamp,
                                    name_pk,
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
        data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                           m_stmts;
    std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> m_common_ops;
};

}  // namespace profiler_hub
