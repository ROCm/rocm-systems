// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/diagnostics.hpp"
#include "addc/schema_version.hpp"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace addc::pipeline
{

struct PipelineEvent
{
    nlohmann::ordered_json metadata = nlohmann::ordered_json::object();
    nlohmann::ordered_json platform = nlohmann::ordered_json::object();
    nlohmann::ordered_json event_report = nlohmann::ordered_json::object();
    nlohmann::ordered_json data_array = nlohmann::ordered_json(nullptr);
    nlohmann::ordered_json validation = nlohmann::ordered_json::object();
    nlohmann::ordered_json analysis = nlohmann::ordered_json::object();
    nlohmann::ordered_json ppr = nlohmann::ordered_json(nullptr);
    nlohmann::ordered_json firmware = nlohmann::ordered_json(nullptr);
};

struct PipelineOutput
{
    std::string tool_version;
    std::string schema_version = std::string{addc::kSchemaVersion};
    std::string timestamp;
    std::string filename;
    std::string platform_id;
    std::optional<std::string> partition_id;
    uint64_t record_id = 0;
    std::string creator_id;
    std::vector<PipelineEvent> events;
    nlohmann::ordered_json recovery = nullptr;
    addc::Diagnostics diagnostics = {};
};

[[nodiscard]] inline nlohmann::ordered_json to_json(const PipelineOutput& out)
{
    nlohmann::ordered_json j;
    j["schema_version"] = out.schema_version;
    j["tool_version"] = out.tool_version;
    j["source"] = nlohmann::ordered_json{
        {"type", "cper"},
        {"filename", out.filename},
        {"timestamp", out.timestamp},
        {"platform_id", out.platform_id},
        {"partition_id", out.partition_id.has_value()
                             ? nlohmann::ordered_json(out.partition_id.value())
                             : nlohmann::ordered_json(nullptr)},
        {"record_id", out.record_id},
        {"creator_id", out.creator_id},
    };

    if (!out.recovery.is_null())
        j["recovery"] = out.recovery;

    auto& events = j["events"];
    events = nlohmann::ordered_json::array();
    for (const auto& ev : out.events)
    {
        nlohmann::ordered_json e;
        e["metadata"] = ev.metadata;
        e["platform"] = ev.platform;
        e["event_report"] = ev.event_report;
        e["data_array"] = ev.data_array;
        e["validation"] = ev.validation;
        e["analysis"] = ev.analysis;
        e["ppr"] = ev.ppr;
        e["firmware"] = ev.firmware;
        events.push_back(std::move(e));
    }

    if (!out.diagnostics.empty())
        j[addc::kDiagnosticsKey] = out.diagnostics.to_json();
    return j;
}

} // namespace addc::pipeline
