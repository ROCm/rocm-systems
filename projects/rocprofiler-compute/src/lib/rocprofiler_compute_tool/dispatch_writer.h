// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "csv/csv.h"
#include "output_registry.h"
#include "sdk_callbacks.h"

#include <string_view>

namespace rocprofiler_compute_tool
{

/// Formats the dispatch CSV; separate from the file for testing.
bool format_dispatch_csv(const tool_data_t& tool_data, const csv::Sink& sink);

/// Writes one row per profiled dispatch to tool_data.dispatch_filename.
/// Kernel properties live in the kernel symbols artifact, joined by kernel id.
class DispatchWriter : public OutputWriter
{
public:
    /// Artifact suffix this writer produces, gzip container included. Owned by
    /// the writer so the name and the format cannot drift apart.
    static constexpr std::string_view kFileSuffix = "_dispatch.csv.gz";

    void write(tool_data_t& tool_data) override;

    std::string_view name() const override { return "dispatch"; }
};

}  // namespace rocprofiler_compute_tool
