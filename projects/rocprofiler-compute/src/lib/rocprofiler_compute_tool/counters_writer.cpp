// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "counters_writer.h"

#include "compression/gzip_output_stream.h"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>

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

void CsvCountersWriter::write_counters(tool_data_t* tool_data)
{
    const std::string& final_path = tool_data->output_filename;
    const std::string  temp_path  = final_path + ".tmp";

    compression::GzipFileOutputStream stream(temp_path);
    if (!stream.is_open())
    {
        std::cerr << "Failed to open output file: " << final_path << std::endl;
        return;
    }

    const auto wrote = format_counters_csv(*tool_data,
                                           [&stream](std::string_view text)
                                           { return stream.write(text); });

    std::error_code ec;
    if (!stream.close() || !wrote)
    {
        std::filesystem::remove(temp_path, ec);
        std::cerr << "Failed to write output file: " << final_path << std::endl;
        return;
    }

    std::filesystem::rename(temp_path, final_path, ec);
    if (ec)
    {
        std::filesystem::remove(temp_path, ec);
        std::cerr << "Failed to write output file: " << final_path << std::endl;
        return;
    }

    std::clog << "[rocprofiler-compute] [" << __FUNCTION__
              << "] Counter collection data has been written to: " << final_path << std::endl;
}

}  // namespace rocprofiler_compute_tool
