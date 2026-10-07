// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "domain_service/backend.hpp"
#include "gpu_perf_counters/backend.hpp"
#include "tracing_config/backend.hpp"

namespace rocprofsys::policies
{

/// @brief Full rocprofiler-sdk backend contract: the union of every policy category's
/// requirements. Satisfied by the production backend
/// (backends::rocprofiler_sdk::backend), which implements all three surfaces; each
/// individual consumer (domain_service, gpu_perf_counter::device, tracing_config)
/// constrains its own template parameter with only the sub-concept it actually uses.
template <typename Backend>
concept backend = domain_service::backend<Backend> &&
                  gpu_perf_counters::backend<Backend> && tracing_config::backend<Backend>;

}  // namespace rocprofsys::policies
