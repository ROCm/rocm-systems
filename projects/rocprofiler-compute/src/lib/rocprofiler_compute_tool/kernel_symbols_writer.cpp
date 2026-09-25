// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "kernel_symbols_writer.h"

#include "compression/gzip_output_stream.h"
#include "csv/csv.h"

#include <algorithm>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

namespace rocprofiler_compute_tool
{
namespace
{
constexpr std::string_view kHeader = "kernel_id,kernel_name,kernel_short_name,arch_vgpr,"
                                     "accum_vgpr,sgpr\n";

// The filename advertises gzip, so the writer must actually produce it.
static_assert(KernelSymbolsWriter::kFileSuffix.size() >= compression::kGzipSuffix.size() &&
                  KernelSymbolsWriter::kFileSuffix.substr(KernelSymbolsWriter::kFileSuffix.size() -
                                                          compression::kGzipSuffix.size()) ==
                      compression::kGzipSuffix,
              "KernelSymbolsWriter::kFileSuffix must end in the gzip suffix");
}  // namespace

bool format_kernel_symbols_csv(const tool_data_t& tool_data, const csv::Sink& sink)
{
    // The symbols live in a hash map, so sort to keep the artifact stable
    // across runs.
    std::vector<uint64_t> kernel_ids;
    kernel_ids.reserve(tool_data.kernel_symbols.size());
    for (const auto& [kernel_id, _] : tool_data.kernel_symbols)
        kernel_ids.push_back(kernel_id);
    std::sort(kernel_ids.begin(), kernel_ids.end());

    const auto write_row = [&tool_data](std::ostream& out, uint64_t kernel_id)
    {
        const auto& symbol = tool_data.kernel_symbols.at(kernel_id);
        out << kernel_id << ',' << csv::quote(symbol.kernel_name) << ','
            << csv::quote(symbol.kernel_short_name) << ',' << symbol.arch_vgpr_count << ','
            << symbol.accum_vgpr_count << ',' << symbol.sgpr_count;
    };

    return csv::format(kHeader, kernel_ids, write_row, sink);
}

void KernelSymbolsWriter::write(tool_data_t& tool_data)
{
    if (tool_data.kernel_symbols.empty() || tool_data.kernel_symbols_filename.empty())
        return;

    compression::GzipFileOutputStream stream(tool_data.kernel_symbols_filename);
    if (!stream.is_open())
    {
        std::cerr << "Failed to open output file: " << tool_data.kernel_symbols_filename << std::endl;
        return;
    }

    const auto wrote = format_kernel_symbols_csv(tool_data,
                                                 [&stream](std::string_view text)
                                                 { return stream.write(text); });

    if (!stream.close() || !wrote)
    {
        std::cerr << "Failed to write output file: " << tool_data.kernel_symbols_filename << std::endl;
        return;
    }

    std::clog << "[rocprofiler-compute] Kernel symbols have been written to: "
              << tool_data.kernel_symbols_filename << std::endl;
}

}  // namespace rocprofiler_compute_tool
