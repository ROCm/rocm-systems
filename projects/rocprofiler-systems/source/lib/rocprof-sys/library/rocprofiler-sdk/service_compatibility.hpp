// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

namespace rocprofsys::rocprofiler_sdk::service_compatibility
{
/**
 * Reports whether SPM and GPU device-counting services were requested together.
 *
 * ROCprofiler-SDK currently configures these services on separate contexts and does
 * not report their hardware conflict. Remove this policy when the SDK provides
 * cross-context conflict detection.
 * ROCPROFSYS_ROCM_EVENTS is deliberately absent because it shares counter_ctx with
 * SPM, allowing ROCprofiler-SDK to report their conflict directly.
 *
 * @param spm_requested Whether the user requested SPM collection.
 * @param gpu_perf_counter_events Raw GPU device-counting counter setting.
 * @return True when the two incompatible services were both requested.
 */
[[nodiscard]] bool
has_spm_gpu_perf_counter_conflict(bool             spm_requested,
                                  std::string_view gpu_perf_counter_events) noexcept;
}  // namespace rocprofsys::rocprofiler_sdk::service_compatibility
