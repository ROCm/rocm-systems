// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "avail/records.hpp"

#include <vector>

namespace rocprofsys::avail
{
/// What the caller wants to know. Nothing is queried unless asked for, so a
/// device listing never pays for the counter walk.
struct query_request
{
    bool devices      = false;
    bool gpu_counters = false;
    bool traces       = false;
};

struct catalog_snapshot
{
    std::vector<device_record>       devices          = {};
    std::vector<device_counters>     counter_groups   = {};
    std::vector<trace_domain_record> traces           = {};
    std::vector<diagnostic>          diagnostics      = {};
    bool                             devices_queried  = false;
    bool                             counters_queried = false;
    bool                             traces_queried   = false;

    [[nodiscard]] bool degraded() const noexcept { return !diagnostics.empty(); }
};

/// Runs the requested queries against the live backends.
///
/// Counters are enumerated per agent, so they imply the device query. Tracing
/// domains come from the SDK name tables and do not require GPU inventory.
[[nodiscard]] catalog_snapshot
query_catalog(const query_request& request);
}  // namespace rocprofsys::avail
