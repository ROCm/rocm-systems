// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "domain_service/externals.hpp"
#include "gpu_perf_counters/externals.hpp"
#include "tracing_config/externals.hpp"

namespace rocprofsys::policies
{

/// @brief Full rocprofiler-sdk externals contract: the union of every policy
/// category's requirements. Each individual consumer (domain_service,
/// gpu_perf_counter::device, tracing_config) constrains its own template parameter
/// with only the sub-concept it actually uses.
template <typename Externals>
concept externals =
    domain_service::externals<Externals> && gpu_perf_counters::externals<Externals> &&
    tracing_config::externals<Externals>;

}  // namespace rocprofsys::policies
