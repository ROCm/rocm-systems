// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/base_report.hpp"

#include <utility>

namespace addc::base
{

ResolvedReport resolve_report(pipeline::PipelineOutput output)
{
    return ResolvedReport{std::move(output)};
}

nlohmann::ordered_json to_json(const ResolvedReport& report)
{
    return pipeline::to_json(report.output);
}

CperErrorSummary summarize_events(const pipeline::PipelineOutput& output)
{
    CperErrorSummary result;
    for (const auto& event : output.events)
    {
        const auto afids = event.analysis.find("afid");
        if (afids == event.analysis.end() || !afids->is_array())
        {
            continue;
        }

        std::string fru_id;
        const auto event_fru_id = event.platform.find("fru_id");
        if (event_fru_id != event.platform.end() && event_fru_id->is_string())
        {
            fru_id = event_fru_id->get<std::string>();
        }

        std::string fru_text;
        const auto event_fru_text = event.platform.find("fru");
        if (event_fru_text != event.platform.end() && event_fru_text->is_string())
        {
            fru_text = event_fru_text->get<std::string>();
        }

        for (const auto& afid : *afids)
        {
            if (afid.is_number_integer())
            {
                result.push_back(CperErrorSummaryEntry{
                    .fru_id = fru_id,
                    .fru_text = fru_text,
                    .afid = afid.get<int>(),
                    .additional_context = {},
                });
            }
        }
    }
    return result;
}

CperErrorSummary summarize(const ResolvedReport& report)
{
    return summarize_events(report.output);
}

nlohmann::ordered_json summary_to_json(const CperErrorSummary& summary)
{
    auto result = nlohmann::ordered_json::array();
    for (const auto& entry : summary)
    {
        result.push_back({
            {"fru_id", entry.fru_id},
            {"fru_text", entry.fru_text},
            {"afid", entry.afid},
            {"additional_context", entry.additional_context},
        });
    }
    return result;
}

} // namespace addc::base
