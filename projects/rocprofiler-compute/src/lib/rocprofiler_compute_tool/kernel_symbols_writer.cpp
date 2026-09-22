// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "kernel_symbols_writer.h"

#include "compression/gzip_output_stream.h"
#include "csv_gz_writer.h"

#include <algorithm>
#include <sstream>
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

bool format_kernel_symbols_csv(const tool_data_t&                           tool_data,
                               const std::function<bool(std::string_view)>& sink)
{
    // The symbols live in a hash map, so sort to keep the artifact stable
    // across runs.
    std::vector<uint64_t> kernel_ids;
    kernel_ids.reserve(tool_data.kernel_symbols.size());
    for (const auto& [kernel_id, _] : tool_data.kernel_symbols)
        kernel_ids.push_back(kernel_id);
    std::sort(kernel_ids.begin(), kernel_ids.end());

    // One row per kernel, so the whole artifact fits in a single batch.
    std::ostringstream out;
    out << kHeader;
    for (auto kernel_id : kernel_ids)
    {
        const auto& symbol = tool_data.kernel_symbols.at(kernel_id);
        out << kernel_id << ',' << csv_quote(symbol.kernel_name) << ','
            << csv_quote(symbol.kernel_short_name) << ',' << symbol.arch_vgpr_count << ','
            << symbol.accum_vgpr_count << ',' << symbol.sgpr_count << '\n';
    }

    return sink(out.str());
}

void KernelSymbolsWriter::write(tool_data_t& tool_data)
{
    if (tool_data.kernel_symbols.empty() || tool_data.kernel_symbols_filename.empty())
        return;

    write_csv_gz(tool_data.kernel_symbols_filename,
                 "Kernel symbols",
                 [&tool_data](const auto& sink)
                 { return format_kernel_symbols_csv(tool_data, sink); });
}

}  // namespace rocprofiler_compute_tool
