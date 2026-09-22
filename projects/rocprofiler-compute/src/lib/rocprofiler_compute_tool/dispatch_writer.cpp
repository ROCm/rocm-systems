// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "dispatch_writer.h"

#include "compression/gzip_output_stream.h"
#include "csv/csv.h"

#include <iostream>
#include <ostream>
#include <string>

namespace rocprofiler_compute_tool
{
namespace
{
constexpr std::string_view kHeader = "dispatch_id,gpu_id,kernel_id,grid_size,workgroup_size,"
                                     "lds_per_workgroup,scratch_per_workitem,start_timestamp,"
                                     "end_timestamp,correlation_id\n";

// The filename advertises gzip, so the writer must actually produce it.
static_assert(DispatchWriter::kFileSuffix.size() >= compression::kGzipSuffix.size() &&
                  DispatchWriter::kFileSuffix.substr(DispatchWriter::kFileSuffix.size() -
                                                     compression::kGzipSuffix.size()) ==
                      compression::kGzipSuffix,
              "DispatchWriter::kFileSuffix must end in the gzip suffix");

void write_row(std::ostream& out, const dispatch_record_t& record)
{
    out << record.dispatch_id << ',' << record.agent_id << ',' << record.kernel_id << ','
        << record.grid_size << ',' << record.workgroup_size << ',' << record.lds_per_workgroup
        << ',' << record.scratch_per_workitem << ',' << record.start_timestamp << ','
        << record.end_timestamp << ',' << record.correlation_id;
}
}  // namespace

bool format_dispatch_csv(const tool_data_t& tool_data, const csv::Sink& sink)
{
    return csv::format(kHeader, tool_data.dispatch_records, write_row, sink);
}

void DispatchWriter::write(tool_data_t& tool_data)
{
    if (tool_data.dispatch_records.empty() || tool_data.dispatch_filename.empty())
        return;

    compression::GzipFileOutputStream stream(tool_data.dispatch_filename);
    if (!stream.is_open())
    {
        std::cerr << "Failed to open output file: " << tool_data.dispatch_filename << std::endl;
        return;
    }

    const auto wrote = format_dispatch_csv(tool_data,
                                           [&stream](std::string_view text)
                                           { return stream.write(text); });

    if (!stream.close() || !wrote)
    {
        std::cerr << "Failed to write output file: " << tool_data.dispatch_filename << std::endl;
        return;
    }

    std::clog << "[rocprofiler-compute] Kernel dispatch data has been written to: "
              << tool_data.dispatch_filename << std::endl;
}

}  // namespace rocprofiler_compute_tool
