// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Schema 3.0.0 PMC event, also used by 3.0.1 and 3.0.2.

#include "writers/common/pmc_event_writer.hpp"
#include "writers/pmc_event_writer.hpp"

#include "data_storage/schema_version.hpp"

#include <stdexcept>
#include <string>

namespace profiler_hub
{

template <>
class pmc_event_writer<data_storage::schema_v3_tag>
: public pmc_event_common<pmc_event_writer<data_storage::schema_v3_tag>>
{
public:
    using pmc_event_common<
        pmc_event_writer<data_storage::schema_v3_tag>>::pmc_event_common;

    void reject_unsupported_fields(const writer_types::pmc_event_data_t& data) const
    {
        const auto version = m_ctx->schema_version.to_string();
        const auto reject =
            [&](bool set, std::string_view field, std::string_view since) {
                if(!set) return;
                throw std::runtime_error("pmc_event." + std::string(field) +
                                         " is not stored by schema " + version +
                                         ". It is stored by " + std::string(since) + ".");
            };
        reject(data.sample_id.has_value(), "sample_id", "3.0.3");
        reject(data.xcc.has_value(), "xcc", "3.0.3");
        reject(data.shader_engine.has_value(), "shader_engine", "3.0.3");
        reject(data.instance.has_value(), "instance", "3.0.3");
    }

    void write_pmc_event(const writer_types::pmc_event_data_t& data,
                         const pmc_event_insert&               row)
    {
        m_stmts->pmc_event_statement()(
            row.pk, row.event_pk, row.pmc_pk, data.value, data.extdata);
    }
};

}  // namespace profiler_hub
