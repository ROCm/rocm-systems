// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "counters_writer.h"

#include "compression/gzip_output_stream.h"
#include "csv_gz_writer.h"

#include <algorithm>
#include <sstream>
#include <string>

namespace rocprofiler_compute_tool
{
namespace
{
constexpr std::string_view kHeader = "dispatch_id,gpu_id,kernel_id,lds_per_workgroup,"
                                     "counter_id,counter_name,counter_value\n";

// Amortizes the sink indirection and the gzwrite call over many rows rather
// than paying both per row. Not tuned; any size well above a row works.
constexpr std::size_t kBatchBytes = 256 * 1024;

// The filename advertises gzip, so the writer must actually produce it.
static_assert(CsvCountersWriter::kFileSuffix.size() >= compression::kGzipSuffix.size() &&
                  CsvCountersWriter::kFileSuffix.substr(CsvCountersWriter::kFileSuffix.size() -
                                                        compression::kGzipSuffix.size()) ==
                      compression::kGzipSuffix,
              "CsvCountersWriter::kFileSuffix must end in the gzip suffix");
}  // namespace

bool format_counters_csv(const tool_data_t& tool_data, const std::function<bool(std::string_view)>& sink)
{
    // ostringstream keeps counter_value formatting the readers already parse.
    std::ostringstream batch;
    batch << kHeader;

    const auto flush_batch = [&sink, &batch]()
    {
        const auto text = batch.str();
        batch.str(std::string{});
        return text.empty() || sink(text);
    };

    for (const auto& r : tool_data.counter_records)
    {
        batch << r.dispatch_id << ',' << r.agent_id << ',' << r.kernel_id << ',' << r.LDS_memory_size
              << ',' << r.counter_id << ',' << r.counter_name << ',' << r.counter_value << '\n';

        if (static_cast<std::size_t>(batch.tellp()) >= kBatchBytes && !flush_batch())
            return false;
    }

    return flush_batch();
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
    write_csv_gz(tool_data->output_filename,
                 "Counter collection data",
                 [tool_data](const auto& sink) { return format_counters_csv(*tool_data, sink); });
}

}  // namespace rocprofiler_compute_tool
