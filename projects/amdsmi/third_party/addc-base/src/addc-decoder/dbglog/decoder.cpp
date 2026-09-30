// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/decoder/dbglog/decoder.hpp"

#include "addc/common.hpp"
#include "addc/decoder/dbglog/epyc.hpp"
#ifdef ADDC_HAS_MI450
#include "addc/decoder/dbglog/mi450.hpp"
#endif
#include "addc/detail/format.hpp"
#include "addc/product.hpp"

namespace addc::decoder
{

std::vector<DecodedDbgLog> decode_generic_dbglog(
    const std::vector<addc::pipeline::DbgLogGroup>& groups)
{
    std::vector<DecodedDbgLog> results;
    results.reserve(groups.size());

    for (const addc::pipeline::DbgLogGroup& g : groups)
    {
        DecodedDbgLog entry;
        entry.id = std::to_string(g.block_id);
        entry.data_array = nlohmann::json(nullptr);
        entry.afid = addc::afid_list(addc::kAfidSentinel);
        entry.corrupted = false;
        entry.possible_non_mca_fatal = false;
        entry.event_report_extra = nlohmann::ordered_json::object();
        entry.mca_bank = std::nullopt;

        results.push_back(std::move(entry));
    }

    return results;
}

std::vector<DecodedDbgLog> decode_base_dbglog(
    std::string_view project,
    const std::vector<addc::pipeline::DbgLogGroup>& groups)
{
    const auto* definition = addc::product::find_compiled_product(project);
    if (definition == nullptr)
    {
        return decode_epyc_dbglog(groups);
    }
    switch (definition->identity.id)
    {
#ifdef ADDC_HAS_MI450
        case addc::product::ProductId::mi450:
            return decode_mi450_dbglog(groups);
#endif
#if defined(ADDC_HAS_VENICE) || defined(ADDC_HAS_TURIN) || \
    defined(ADDC_HAS_GENOA) || defined(ADDC_HAS_FIRERANGE)
        case addc::product::ProductId::venice:
        case addc::product::ProductId::turin:
        case addc::product::ProductId::genoa:
        case addc::product::ProductId::firerange:
            return decode_epyc_dbglog(groups);
#endif
        default:
            // Historical registry lookup fell back to the generic EPYC
            // operation for structural crashdump records whose platform did
            // not own a specialized DbgLog decoder.
            return decode_epyc_dbglog(groups);
    }
}

nlohmann::ordered_json to_json(const DecodedDbgLog& d)
{
    nlohmann::ordered_json j;
    j["name"] = d.key_name.empty() ? nlohmann::json(nullptr)
                                   : nlohmann::json(d.key_name);
    j["id"] = d.id;
    j["data_array"] = d.data_array;
    j["afid"] = d.afid;
    j["corrupted"] =
        d.corrupted ? nlohmann::json(true) : nlohmann::json(nullptr);
    j["possible_non_mca_fatal"] = d.possible_non_mca_fatal
                                      ? nlohmann::json(true)
                                      : nlohmann::json(nullptr);
    return j;
}

} // namespace addc::decoder
