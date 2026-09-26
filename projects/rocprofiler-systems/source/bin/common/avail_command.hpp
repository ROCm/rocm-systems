// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "avail/catalog.hpp"

#include <iosfwd>
#include <string>
#include <string_view>

namespace rocprofsys::cli
{
/// Parsed `rocsys avail` command line. The surface is deliberately small: this
/// is a capability query, not a configuration tool.
struct avail_options
{
    // Hardware
    bool devices     = false;
    bool cpu_devices = false;
    bool nic_devices = false;

    // Traces
    bool        traces          = false;
    std::string list_operations = {};

    // Counters
    bool gpu_counters = false;
    bool cpu_counters = false;

    // Metrics
    bool cpu_metrics     = false;
    bool gpu_metrics     = false;
    bool nic_metrics     = false;
    bool storage_metrics = false;

    // Output
    std::string output_path   = {};
    bool        no_pager      = false;
    bool        help          = false;
    std::string help_topic    = {};
    bool        valid         = true;
    std::string error_message = {};

    /// True when no selector was given, which asks for the summary.
    [[nodiscard]] bool summary() const noexcept
    {
        return !devices && !cpu_devices && !nic_devices && !traces &&
               list_operations.empty() && !gpu_counters && !cpu_counters &&
               !cpu_metrics && !gpu_metrics && !nic_metrics && !storage_metrics;
    }

    /// What the summary and the selectors need from the catalog.
    [[nodiscard]] avail::query_request to_request() const noexcept
    {
        // The summary reports device counts, so it still needs the device
        // query, but never the counter or trace walks. --list-operations
        // uses the same SDK name tables as --traces.
        return avail::query_request{ devices || summary(), gpu_counters,
                                     traces || !list_operations.empty() };
    }
};

/// Parses argv for `rocsys avail`, skipping argv[0] and the `avail` token if
/// it is still present.
[[nodiscard]] avail_options
parse_avail_options(int argc, char** argv);

void
print_avail_help(std::ostream& out, std::string_view program);

/// Renders a snapshot. Diagnostics go to @p err so a degraded backend is
/// visible without corrupting piped output. Returns false if a requested
/// --list-operations domain is unknown.
[[nodiscard]] bool
print_avail_snapshot(const avail::catalog_snapshot& snapshot, const avail_options& options,
                     std::ostream& out, std::ostream& err);

/// Entry point used by the `rocsys` dispatcher. Returns a process exit code.
[[nodiscard]] int
run_avail(int argc, char** argv, std::ostream& out, std::ostream& err);
}  // namespace rocprofsys::cli
