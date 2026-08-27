// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <set>
#include <string>
#include <vector>

// Split out of core/gpu.hpp so that callers correlating devices against runtime
// visibility do not have to pull in <amd_smi/amdsmi.h>.

namespace rocprofsys
{
namespace gpu
{
/**
 * @brief PCIe BDFs of the GPUs the ROCm runtime exposes.
 *
 * Reports the canonical PCIe BDF strings ("domain:bus:device.function") of the GPUs that
 * rocprofiler-sdk marks as runtime-visible, which honors ROCR_VISIBLE_DEVICES,
 * HIP_VISIBLE_DEVICES, and CUDA_VISIBLE_DEVICES.
 *
 * @return The visible BDFs, or std::nullopt when rocprofiler-sdk reports no GPU agents
 * at all.
 *
 * @note std::nullopt means visibility could not be determined, which is a different
 * signal from an empty set (agents exist, but every one is masked). Callers must not
 * treat the two as equivalent.
 */
std::optional<std::set<std::string>>
get_visible_gpu_bdfs();

/**
 * @brief Profiler GPU indices of the GPUs the HIP runtime exposes, in HIP ordinal order.
 *
 * Element @c k is the @c device_type_index of HIP device ordinal @c k. That is the space
 * hipFile uses to index @c per_gpu_stats (via @c hipGetDevice()). rocprofiler-sdk does
 * not store a HIP ordinal on the agent, only @c runtime_visibility.hip, so this is the
 * k-th GPU agent with @c hip_visible set, in agent-manager order. That matches HIP when
 * @c HIP_VISIBLE_DEVICES / @c ROCR_VISIBLE_DEVICES is an increasing subset (e.g. "4,5").
 * A permutation (e.g. "5,4") or UUID list cannot be mapped without mislabeling, so this
 * returns empty and hipFile telemetry is skipped. AMD SMI GPU sampling is unaffected
 * (@c get_visible_gpu_bdfs is order-independent).
 *
 * @return Empty when no GPU agents were found (same "unknown vs none" situation as
 *         @c get_visible_gpu_bdfs returning nullopt, collapsed to "no visible GPUs"
 *         because hipFile has nothing to index), or when the visibility mask is not
 *         an increasing integer subset.
 */
std::vector<std::size_t>
get_visible_gpu_type_indices();
}  // namespace gpu
}  // namespace rocprofsys
