// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @file features.hpp
 * @brief AMD SMI feature detection for version compatibility.
 *
 * AINIC (AI NIC) support is controlled by the CMake variable ROCPROFSYS_BUILD_AINIC,
 * which is set based on:
 * - ROCPROFSYS_USE_AINIC option being ON
 * - AMD SMI library version >= 26.3
 */

namespace rocprofsys::pmc::device_providers::amd_smi
{

/**
 * @brief Check if AMD SMI NIC support is available.
 *
 * NIC support (AINIC) was added in ROCm 7.0 (AMD SMI lib version 26.3+).
 * This is controlled by ROCPROFSYS_BUILD_AINIC defined via CMake.
 */
#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
constexpr bool has_nic_support = true;
#else
constexpr bool has_nic_support = false;
#endif

}  // namespace rocprofsys::pmc::device_providers::amd_smi
