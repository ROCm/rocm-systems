// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "counters_writer.h"

#include "compression/gzip_output_stream.h"
#include "csv/csv.h"

#include <algorithm>
#include <iostream>
#include <ostream>
#include <string>

namespace rocprofiler_compute_tool
{
namespace
{
constexpr std::string_view kHeader = "dispatch_id,gpu_id,kernel_id,lds_per_workgroup,"
                                     "counter_id,counter_name,counter_value\n";

// The filename advertises gzip, so the writer must actually produce it.
static_assert(CsvCountersWriter::kFileSuffix.size() >= compression::kGzipSuffix.size() &&
                  CsvCountersWriter::kFileSuffix.substr(CsvCountersWriter::kFileSuffix.size() -
                                                        compression::kGzipSuffix.size()) ==
                      compression::kGzipSuffix,
              "CsvCountersWriter::kFileSuffix must end in the gzip suffix");

// ostream keeps counter_value formatting the readers already parse.
void write_row(std::ostream& out, const counter_info_record_t& record)
{
    out << record.dispatch_id << ',' << record.agent_id << ',' << record.kernel_id << ','
        << record.LDS_memory_size << ',' << record.counter_id << ',' << record.counter_name << ','
        << record.counter_value;
}
}  // namespace

bool format_counters_csv(const tool_data_t& tool_data, const csv::Sink& sink)
{
    return csv::format(kHeader, tool_data.counter_records, write_row, sink);
}

void CountersWriter::write(tool_data_t& tool_data)
{
    // Dispatches before the kernel to be filtered was registered may have been
    // profiled. Remove any records whose kernel id does not match the
    // target_kernel_ids
    if (!tool_data.target_kernel_ids.empty())
    {
        auto& records = tool_data.counter_records;
        records.erase(std::remove_if(records.begin(),
                                     records.end(),
                                     [&tool_data](const counter_info_record_t& record)
                                     {
                                         return tool_data.target_kernel_ids.find(record.kernel_id) ==
                                                tool_data.target_kernel_ids.end();
                                     }),
                      records.end());
    }

    if (!tool_data.counter_records.empty() && !tool_data.output_filename.empty())
        write_counters(&tool_data);
}

void CsvCountersWriter::write_counters(tool_data_t* tool_data)
{
    compression::GzipFileOutputStream stream(tool_data->output_filename);
    if (!stream.is_open())
    {
        std::cerr << "Failed to open output file: " << tool_data->output_filename << std::endl;
        return;
    }

    const auto wrote = format_counters_csv(*tool_data,
                                           [&stream](std::string_view text)
                                           { return stream.write(text); });

    if (!stream.close() || !wrote)
    {
        std::cerr << "Failed to write output file: " << tool_data->output_filename << std::endl;
        return;
    }

    std::clog << "[rocprofiler-compute] Counter collection data has been written to: "
              << tool_data->output_filename << std::endl;
}

}  // namespace rocprofiler_compute_tool
