// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "agent_writer.h"

#include "compression/gzip_output_stream.h"
#include "csv/csv.h"

#include <iostream>
#include <ostream>
#include <string>
#include <utility>

namespace rocprofiler_compute_tool
{
namespace
{
constexpr std::string_view kHeader = "node_id,name,product_name\n";

// The filename advertises gzip, so the writer must actually produce it.
static_assert(AgentWriter::kFileSuffix.size() >= compression::kGzipSuffix.size() &&
                  AgentWriter::kFileSuffix.substr(AgentWriter::kFileSuffix.size() -
                                                  compression::kGzipSuffix.size()) == compression::kGzipSuffix,
              "AgentWriter::kFileSuffix must end in the gzip suffix");

void write_row(std::ostream& out, const std::pair<const uint64_t, agent_record_t>& entry)
{
    const auto& agent = entry.second;
    out << agent.node_id << ',' << csv::quote(agent.name) << ',' << csv::quote(agent.product_name);
}
}  // namespace

bool format_agents_csv(const tool_data_t& tool_data, const csv::Sink& sink)
{
    return csv::format(kHeader, tool_data.agents, write_row, sink);
}

void AgentWriter::write(tool_data_t& tool_data)
{
    if (tool_data.agents.empty() || tool_data.agents_filename.empty())
        return;

    compression::GzipFileOutputStream stream(tool_data.agents_filename);
    if (!stream.is_open())
    {
        std::cerr << "Failed to open output file: " << tool_data.agents_filename << std::endl;
        return;
    }

    const auto wrote = format_agents_csv(tool_data,
                                         [&stream](std::string_view text)
                                         { return stream.write(text); });

    if (!stream.close() || !wrote)
    {
        std::cerr << "Failed to write output file: " << tool_data.agents_filename << std::endl;
        return;
    }

    std::clog << "[rocprofiler-compute] GPU agents have been written to: " << tool_data.agents_filename
              << std::endl;
}

}  // namespace rocprofiler_compute_tool
