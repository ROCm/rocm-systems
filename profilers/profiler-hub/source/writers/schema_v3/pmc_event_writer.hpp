// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "writers/pmc_event_writer.hpp"
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
class pmc_event_writer<data_storage::schema_v3_tag>
: public pmc_event_writer_interface<pmc_event_writer<data_storage::schema_v3_tag>>
{
public:
    explicit pmc_event_writer(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                               stmts,
        std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> common_ops)
    : m_ctx(std::move(ctx))
    , m_stmts(std::move(stmts))
    , m_common_ops(std::move(common_ops))
    {}

    void insert_impl(const writer_types::pmc_event_data_t&     data,
                     const writer_types::pmc_info_unique_id_t& pmc_unique_id)
    {
        auto transaction_block = m_ctx->backend->begin_transaction();

        m_ctx->validator->require_pmc(pmc_unique_id);

        const auto pmc_pk = m_ctx->validator->resolve_pmc_key(pmc_unique_id);

        std::optional<primary_key_t> event_pk = std::nullopt;
        if(data.event.has_value())
        {
            event_pk = m_common_ops->insert_event(data.event.value());
        }

        const auto pk = m_ctx->key_providers->pmc_event_data().get_primary_key_value();

        m_stmts->pmc_event_statement()(pk, event_pk, pmc_pk, data.value, data.extdata);

        if(event_pk.has_value())
        {
            m_common_ops->insert_sample(data.sample, event_pk.value());
        }
    }

private:
    std::shared_ptr<writer_context> m_ctx;
    std::shared_ptr<
        data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                           m_stmts;
    std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> m_common_ops;
};

}  // namespace profiler_hub
