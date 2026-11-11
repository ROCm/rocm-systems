// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// 3.0.3 PMC event stores sample_id, xcc, shader_engine, and instance.
// 3.0.4 reuses this writer. 3.0.5 drops the location columns.

#include "writers/common/pmc_event_writer.hpp"
#include "writers/pmc_event_writer.hpp"

#include "data_storage/schema_version.hpp"
#include "queries/insert/table_insert_query.hpp"

#include <fmt/format.h>

#include <optional>
#include <string>
#include <string_view>

namespace profiler_hub
{

template <>
class pmc_event_writer<data_storage::schema_3_0_3_tag>
: public pmc_event_common<pmc_event_writer<data_storage::schema_3_0_3_tag>>
{
public:
    using base_t = pmc_event_common<pmc_event_writer<data_storage::schema_3_0_3_tag>>;

    using location_statement_t =
        data_storage::sqlite_backend::prepared_insert_statement<size_t,
                                                                std::optional<size_t>,
                                                                std::optional<size_t>,
                                                                size_t,
                                                                double,
                                                                std::optional<size_t>,
                                                                std::optional<size_t>,
                                                                std::optional<size_t>,
                                                                std::string_view>;

    explicit pmc_event_writer(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                               stmts,
        std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> common_ops)
    : base_t(std::move(ctx), std::move(stmts), std::move(common_ops))
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        const auto                                        query =
            query_builder.set_table_name(fmt::format("rocpd_pmc_event_{}", m_ctx->uuid))
                .set_columns("id",
                             "event_id",
                             "sample_id",
                             "pmc_id",
                             "value",
                             "xcc",
                             "shader_engine",
                             "instance",
                             "extdata")
                .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?')
                .get_query_string();
        m_location_statement =
            m_ctx->backend
                ->template create_write_statement_executor<size_t,
                                                           std::optional<size_t>,
                                                           std::optional<size_t>,
                                                           size_t,
                                                           double,
                                                           std::optional<size_t>,
                                                           std::optional<size_t>,
                                                           std::optional<size_t>,
                                                           std::string_view>(query);
    }

    void reject_unsupported_fields(const writer_types::pmc_event_data_t&) const {}

    void write_pmc_event(const writer_types::pmc_event_data_t& data,
                         const pmc_event_insert&               row)
    {
        m_location_statement(row.pk,
                             row.event_pk,
                             row.sample_pk,
                             row.pmc_pk,
                             data.value,
                             data.xcc,
                             data.shader_engine,
                             data.instance,
                             data.extdata);
    }

private:
    location_statement_t m_location_statement;
};

}  // namespace profiler_hub
