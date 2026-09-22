// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "dispatch_writer.h"

#include "compression/gzip_output_stream.h"
#include "csv_gz_writer.h"

#include <sstream>
#include <string>

namespace rocprofiler_compute_tool
{
namespace
{
constexpr std::string_view kHeader = "dispatch_id,gpu_id,kernel_id,grid_size,workgroup_size,"
                                     "lds_per_workgroup,scratch_per_workitem,start_timestamp,"
                                     "end_timestamp,correlation_id\n";

// Amortizes the sink indirection and the gzwrite call over many rows rather
// than paying both per row. Not tuned; any size well above a row works.
constexpr std::size_t kBatchBytes = 256 * 1024;

// The filename advertises gzip, so the writer must actually produce it.
static_assert(DispatchWriter::kFileSuffix.size() >= compression::kGzipSuffix.size() &&
                  DispatchWriter::kFileSuffix.substr(DispatchWriter::kFileSuffix.size() -
                                                     compression::kGzipSuffix.size()) ==
                      compression::kGzipSuffix,
              "DispatchWriter::kFileSuffix must end in the gzip suffix");
}  // namespace

bool format_dispatch_csv(const tool_data_t& tool_data, const std::function<bool(std::string_view)>& sink)
{
    std::ostringstream batch;
    batch << kHeader;

    const auto flush_batch = [&sink, &batch]()
    {
        const auto text = batch.str();
        batch.str(std::string{});
        return text.empty() || sink(text);
    };

    for (const auto& r : tool_data.dispatch_records)
    {
        batch << r.dispatch_id << ',' << r.agent_id << ',' << r.kernel_id << ',' << r.grid_size << ','
              << r.workgroup_size << ',' << r.lds_per_workgroup << ',' << r.scratch_per_workitem << ','
              << r.start_timestamp << ',' << r.end_timestamp << ',' << r.correlation_id << '\n';

        if (static_cast<std::size_t>(batch.tellp()) >= kBatchBytes && !flush_batch())
            return false;
    }

    return flush_batch();
}

void DispatchWriter::write(tool_data_t& tool_data)
{
    if (tool_data.dispatch_records.empty() || tool_data.dispatch_filename.empty())
        return;

    write_csv_gz(tool_data.dispatch_filename,
                 "Kernel dispatch data",
                 [&tool_data](const auto& sink) { return format_dispatch_csv(tool_data, sink); });
}

}  // namespace rocprofiler_compute_tool
