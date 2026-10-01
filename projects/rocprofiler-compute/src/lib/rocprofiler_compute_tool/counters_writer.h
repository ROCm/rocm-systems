// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "sdk_callbacks.h"

#include <functional>
#include <string_view>

namespace rocprofiler_compute_tool
{

class CountersWriter
{
public:
    virtual ~CountersWriter()                           = default;
    virtual void write_counters(tool_data_t* tool_data) = 0;
};

/// Formats counter CSV in batches; separate from the file for testing.
bool format_counters_csv(const tool_data_t& tool_data, const std::function<bool(std::string_view)>& sink);

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
