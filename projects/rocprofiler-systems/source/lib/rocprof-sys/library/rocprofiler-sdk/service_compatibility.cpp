// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/service_compatibility.hpp"

#include <algorithm>
#include <cctype>
#include <string_view>

namespace rocprofsys::rocprofiler_sdk::service_compatibility
{
bool
has_spm_gpu_perf_counter_conflict(bool             spm_requested,
                                  std::string_view gpu_perf_counter_events) noexcept
{
    return spm_requested &&
           std::ranges::any_of(gpu_perf_counter_events, [](unsigned char character) {
               return std::isspace(character) == 0;
           });
}
}  // namespace rocprofsys::rocprofiler_sdk::service_compatibility
