// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "csv/csv.h"
#include "output_registry.h"
#include "sdk_callbacks.h"

#include <string_view>

namespace rocprofiler_compute_tool
{

class CountersWriter : public OutputWriter
{
public:
    /// Drops records for kernels that are not being targeted, then writes what
    /// is left. Dispatches profiled before the targeted kernel was registered
    /// leave records behind that do not belong in the output.
    void write(tool_data_t& tool_data) override;

    std::string_view name() const override { return "counters"; }

    virtual void write_counters(tool_data_t* tool_data) = 0;
};

/// Formats the counter CSV; separate from the file for testing.
bool format_counters_csv(const tool_data_t& tool_data, const csv::Sink& sink);

/// Writes gzip-compressed CSV to tool_data->output_filename.
class CsvCountersWriter : public CountersWriter
{
public:
    /// Artifact suffix this writer produces, gzip container included. Owned by
    /// the writer so the name and the format cannot drift apart.
    static constexpr std::string_view kFileSuffix = "_native_counter_collection.csv.gz";

    void write_counters(tool_data_t* tool_data) override;
};
}  // namespace rocprofiler_compute_tool
