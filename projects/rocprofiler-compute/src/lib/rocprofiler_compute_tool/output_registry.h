// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include <memory>
#include <string_view>
#include <vector>

namespace rocprofiler_compute_tool
{
// Declared, not included: sdk_callbacks.h reaches this header through
// pc_sampling_feature.h, and the registry only ever holds a reference.
struct tool_data_t;

/// One artifact the tool produces.
class OutputWriter
{
public:
    virtual ~OutputWriter() = default;

    /// tool_data is non-const: writers filter records in place and the PC
    /// sampling feature finalizes itself.
    virtual void write(tool_data_t& tool_data) = 0;

    /// Used in log messages when a writer throws.
    virtual std::string_view name() const = 0;
};

/// Every writer in one place, so shutdown writes the tool's output with a
/// single call and no service has to own a flush path of its own.
class OutputRegistry
{
public:
    void register_writer(std::shared_ptr<OutputWriter> writer);

    /// Swaps the registered writer of that name for another. For tests, which
    /// put a mock in the place the real writer holds.
    bool replace_writer(std::string_view name, std::shared_ptr<OutputWriter> writer);

    /// A writer that throws is logged and skipped so one failure cannot lose
    /// the other writers' data.
    void generate_all(tool_data_t& tool_data);

    /// Drops the registered writers. For tests.
    void reset();

private:
    std::vector<std::shared_ptr<OutputWriter>> m_writers;
};

}  // namespace rocprofiler_compute_tool
