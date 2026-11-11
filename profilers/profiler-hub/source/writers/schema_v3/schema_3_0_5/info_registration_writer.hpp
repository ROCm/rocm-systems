// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// 3.0.5 keeps spm_support and stores the PMC location on rocpd_info_pmc.

#include "writers/common/info_registration_writer.hpp"
#include "writers/info_registration_writer.hpp"

#include "data_storage/schema_version.hpp"
#include "queries/insert/table_insert_query.hpp"

#include <fmt/format.h>

#include <optional>
#include <string>
#include <string_view>

namespace profiler_hub
{

template <>
class info_registration_writer<data_storage::schema_3_0_5_tag>
: public info_registration_common<
      info_registration_writer<data_storage::schema_3_0_5_tag>>
{
public:
    using base_t = info_registration_common<
        info_registration_writer<data_storage::schema_3_0_5_tag>>;

    using location_statement_t = data_storage::sqlite_backend::prepared_insert_statement<
        size_t,
        size_t,
        size_t,
        std::optional<size_t>,
        std::optional<std::string_view>,
        std::optional<size_t>,
        std::optional<size_t>,
        std::string_view,
        std::optional<std::string_view>,
        std::optional<std::string_view>,
        std::optional<std::string_view>,
        std::optional<std::string_view>,
        std::optional<std::string_view>,
        std::optional<std::string_view>,
        std::optional<std::string_view>,
        std::optional<std::string_view>,
        std::optional<size_t>,
        std::optional<size_t>,
        std::optional<size_t>,
        std::optional<size_t>,
        std::optional<size_t>,
        std::optional<size_t>,
        std::optional<size_t>,
        std::optional<size_t>,
        std::string_view>;

    explicit info_registration_writer(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                               stmts,
        std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> common_ops)
    : base_t(std::move(ctx), std::move(stmts), std::move(common_ops))
    {
        profiler_hub::queries::insert::table_insert_query query_builder;
        const auto                                        query =
            query_builder.set_table_name(fmt::format("rocpd_info_pmc_{}", m_ctx->uuid))
                .set_columns("id",
                             "nid",
                             "pid",
                             "agent_id",
                             "target_arch",
                             "event_code",
                             "instance_id",
                             "name",
                             "symbol",
                             "description",
                             "long_description",
                             "component",
                             "units",
                             "value_type",
                             "block",
                             "expression",
                             "is_constant",
                             "is_derived",
                             "spm_support",
                             "xcc",
                             "shader_engine",
                             "shader_array",
                             "wgp",
                             "instance",
                             "extdata")
                .set_values('?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?',
                            '?')
                .get_query_string();
        m_location_statement = m_ctx->backend->template create_write_statement_executor<
            size_t,
            size_t,
            size_t,
            std::optional<size_t>,
            std::optional<std::string_view>,
            std::optional<size_t>,
            std::optional<size_t>,
            std::string_view,
            std::optional<std::string_view>,
            std::optional<std::string_view>,
            std::optional<std::string_view>,
            std::optional<std::string_view>,
            std::optional<std::string_view>,
            std::optional<std::string_view>,
            std::optional<std::string_view>,
            std::optional<std::string_view>,
            std::optional<size_t>,
            std::optional<size_t>,
            std::optional<size_t>,
            std::optional<size_t>,
            std::optional<size_t>,
            std::optional<size_t>,
            std::optional<size_t>,
            std::optional<size_t>,
            std::string_view>(query);
    }

    void reject_agent_type(const writer_types::agent_info_t& agent_info) const
    {
        if(!agent_info.unique_id.agent_type.has_value()) return;
        reject_unknown_agent_type(*agent_info.unique_id.agent_type);
    }

    void reject_pmc_fields(const writer_types::pmc_info_t& pmc_info) const
    {
        if(pmc_info.target_arch.has_value())
        {
            reject_unknown_target_arch(*pmc_info.target_arch);
        }
    }

    void write_pmc_info(const writer_types::pmc_info_t& pmc_info,
                        primary_key_t                   primary_key,
                        primary_key_t                   process_pk,
                        std::optional<primary_key_t>    agent_pk)
    {
        const auto& id = pmc_info.unique_id;
        m_location_statement(primary_key,
                             pmc_info.node_id,
                             process_pk,
                             agent_pk,
                             pmc_info.target_arch,
                             pmc_info.event_code,
                             pmc_info.instance_id,
                             id.name,
                             pmc_info.symbol,
                             pmc_info.description,
                             pmc_info.long_description,
                             pmc_info.component,
                             pmc_info.units,
                             pmc_info.value_type,
                             pmc_info.block,
                             pmc_info.expression,
                             pmc_info.is_constant,
                             pmc_info.is_derived,
                             pmc_info.spm_support,
                             id.xcc,
                             id.shader_engine,
                             id.shader_array,
                             id.wgp,
                             id.instance,
                             pmc_info.extdata);
    }

private:
    location_statement_t m_location_statement;
};

}  // namespace profiler_hub
