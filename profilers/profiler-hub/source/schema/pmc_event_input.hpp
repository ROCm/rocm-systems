// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace profiler_hub::schema
{

// One struct for every rocpd_pmc_event column a caller can set, across 3.0.0-3.0.5.
// SQLite fills id and guid. sample_id arrived in 3.0.3. xcc, shader_engine, and
// instance exist only on 3.0.3 and 3.0.4; 3.0.5 moved them to rocpd_info_pmc.
struct pmc_event_input_t
{
    std::optional<int64_t>     event_id;
    std::optional<int64_t>     sample_id;
    std::optional<int64_t>     pmc_id;
    std::optional<double>      value;
    std::optional<int>         xcc;
    std::optional<int>         shader_engine;
    std::optional<int>         instance;
    std::optional<std::string> extdata;
};

}  // namespace profiler_hub::schema
