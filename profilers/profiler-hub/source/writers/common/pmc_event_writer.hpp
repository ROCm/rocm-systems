// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "writers/interfaces/pmc_event_writer_interface.hpp"
#include "writers/schema_v3/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v3/insert_statements.hpp"
#include "profiler-hub/writer_types.hpp"

#include <memory>
#include <optional>

namespace profiler_hub
{

struct pmc_event_insert
{
    primary_key_t                pk{};
    primary_key_t                pmc_pk{};
    std::optional<primary_key_t> event_pk;
    std::optional<primary_key_t> sample_pk;
};

// Schema 3.0.0 PMC event. 3.0.3 adds sample_id and a location.
// 3.0.5 keeps sample_id and moves the location onto rocpd_info_pmc.
template <typename Derived>
class pmc_event_common : public pmc_event_writer_interface<Derived>
{
public:
    explicit pmc_event_common(
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
        this->self().reject_unsupported_fields(data);

        auto transaction_block = m_ctx->backend->begin_transaction();

        m_ctx->validator->require_pmc(pmc_unique_id);

        pmc_event_insert row;
        row.pmc_pk = m_ctx->validator->resolve_pmc_key(pmc_unique_id);
        if(data.event.has_value())
        {
            row.event_pk = m_common_ops->insert_event(data.event.value());
        }
        if(data.sample_id.has_value())
        {
            // The caller's number is not stored. The sample row's id is.
            row.sample_pk = m_common_ops->insert_sample(data.sample, row.event_pk);
        }
        else if(row.event_pk.has_value())
        {
            m_common_ops->insert_sample(data.sample, row.event_pk);
        }

        row.pk = m_ctx->key_providers->pmc_event_data().get_primary_key_value();
        this->self().write_pmc_event(data, row);
    }

protected:
    std::shared_ptr<writer_context> m_ctx;
    std::shared_ptr<
        data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                           m_stmts;
    std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> m_common_ops;
};

}  // namespace profiler_hub
