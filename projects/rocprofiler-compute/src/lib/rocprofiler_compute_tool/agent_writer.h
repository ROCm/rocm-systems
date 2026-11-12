// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "csv/csv.h"
#include "output_registry.h"
#include "sdk_callbacks.h"

#include <string_view>

namespace rocprofiler_compute_tool
{

/// Formats the agents CSV; separate from the file for testing.
bool format_agents_csv(const tool_data_t& tool_data, const csv::Sink& sink);

/// Writes one row per GPU agent to tool_data.agents_filename.
class AgentWriter : public OutputWriter
{
public:
    /// Artifact suffix this writer produces, gzip container included. Owned by
    /// the writer so the name and the format cannot drift apart.
    static constexpr std::string_view kFileSuffix = "_agents.csv.gz";

    void write(tool_data_t& tool_data) override;

    std::string_view name() const override { return "agents"; }
};

}  // namespace rocprofiler_compute_tool
