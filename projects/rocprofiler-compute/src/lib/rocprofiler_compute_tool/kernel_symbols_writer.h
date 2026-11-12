// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "csv/csv.h"
#include "output_registry.h"
#include "sdk_callbacks.h"

#include <string_view>

namespace rocprofiler_compute_tool
{

/// Formats the kernel symbols CSV; separate from the file for testing.
bool format_kernel_symbols_csv(const tool_data_t& tool_data, const csv::Sink& sink);

/// Writes one row per kernel to tool_data.kernel_symbols_filename, so these
/// properties do not repeat on every dispatch row.
class KernelSymbolsWriter : public OutputWriter
{
public:
    /// Artifact suffix this writer produces, gzip container included. Owned by
    /// the writer so the name and the format cannot drift apart.
    static constexpr std::string_view kFileSuffix = "_kernel_symbols.csv.gz";

    void write(tool_data_t& tool_data) override;

    std::string_view name() const override { return "kernel_symbols"; }
};

}  // namespace rocprofiler_compute_tool
