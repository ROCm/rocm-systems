// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocprofsys::policies::gpu_perf_counters
{

/// @brief The GPU perf-counter device-counting path
/// (rocprofsys::pmc::collectors::gpu_perf_counter::device,
/// rocprofsys::pmc::device_providers::rocprofiler_sdk::provider) has no
/// external-dependency surface today, so this concept is trivially satisfied. It
/// exists so it composes with policies::externals's conjunction of every
/// rocprofiler-sdk policy category's externals contract.
template <typename Externals>
concept externals = true;

}  // namespace rocprofsys::policies::gpu_perf_counters
