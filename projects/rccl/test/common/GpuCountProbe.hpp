/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once

namespace RcclUnitTesting
{

// Returned by gpuCountViaFreshImage() when no count could be obtained. Callers
// must treat it as "unknown", not as "zero GPUs".
constexpr int kGpuCountProbeFailed = -1;

/**
 * @brief Returns the number of devices HIP can open in this process.
 *
 * The query runs in a fork()+execv()'d fresh copy of the test binary, so the
 * calling process never initialises the HIP runtime. Two reasons:
 *
 *  - rocprofv3 --hip-trace (rocprofiler-sdk) cannot trace HIP across a bare
 *    fork(): a forked child that calls HIP deadlocks. See the NOTE at
 *    EnvVars.cpp:292-297, which records the same hazard removing the earlier
 *    bare-fork probes, and EnvVars::DetectGpuInfo() for the sibling use of
 *    this pattern.
 *  - Initialising HIP in a test parent opens GPU file descriptors that every
 *    subsequently forked test child inherits.
 *
 * Linking this TU is all that is required: the fresh image services the probe
 * request from a file-scope initializer, before main() and therefore before
 * any test runs.
 *
 * @return Devices HIP reports (possibly 0), or kGpuCountProbeFailed if the
 *         probe could not produce an answer.
 */
int gpuCountViaFreshImage();

} // namespace RcclUnitTesting
