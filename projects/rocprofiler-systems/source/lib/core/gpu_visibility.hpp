// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <set>
#include <string>

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
}  // namespace gpu
}  // namespace rocprofsys
