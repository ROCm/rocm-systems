/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaMPITestHelpers.hpp
 * @brief Shared sizing and device helpers for MPI tests that pin a DDA kernel path.
 *
 * Header-only and public-API only, so MPI test files can include it without the
 * internal headers that DdaFabricTestHelpers.hpp / DdaIpcTestHelpers.hpp pull in.
 */

#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>

namespace DdaMPITestHelpers
{

/// 256 KiB f32: below the default 1 MiB DDA LL one-shot threshold.
constexpr size_t kOneShotCount = 65536;

/// 1 MiB + 128 B of f32: past the default one-shot threshold, and satisfies the
/// two-shot shard alignment at typical MPI rank counts (see
/// DdaFabricEligibilityTests.cpp AllReduceLL_TwoShotClaimsPastOneShotThreshold).
constexpr size_t kTwoShotBaseCount = 262176;

/// Default RCCL_DDA_LL_ONESHOT_THRESHOLD; kOneShotCount and kTwoShotBaseCount are sized against it.
constexpr size_t kDefaultLLOneShotThresholdBytes = size_t(1) * 1024 * 1024;

/// Architecture name of the HIP device bound to the calling thread.
inline std::string currentDeviceArch()
{
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
        return {};
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, device) != hipSuccess)
        return {};
    return props.gcnArchName;
}

inline bool isGfx1250Device()
{
    return currentDeviceArch().find("gfx1250") != std::string::npos;
}

inline bool isDdaIpcArchDevice()
{
    const std::string arch = currentDeviceArch();
    return arch.find("gfx942") != std::string::npos || arch.find("gfx950") != std::string::npos;
}

inline bool ddaLLTwoShotShapeOk(size_t count, int nRanks)
{
    const size_t bytes = count * sizeof(float);
    if(bytes % static_cast<size_t>(nRanks) != 0)
        return false;
    return (bytes / static_cast<size_t>(nRanks)) % 16 == 0;
}

/// A count past the default one-shot threshold whose total bytes satisfy the two-shot
/// per-rank shard alignment for nRanks; 0 if none is found.
inline size_t twoShotCountForRanks(int nRanks)
{
    size_t count = kTwoShotBaseCount;
    if(ddaLLTwoShotShapeOk(count, nRanks))
        return count;
    count = (count + 3) & ~size_t(3);
    for(int i = 0; i < 1024; ++i, count += 4)
    {
        if(ddaLLTwoShotShapeOk(count, nRanks))
            return count;
    }
    return 0;
}

} // namespace DdaMPITestHelpers
