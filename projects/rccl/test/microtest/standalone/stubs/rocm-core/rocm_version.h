/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Standalone (no-ROCm) stub for <rocm-core/rocm_version.h>.
//
// init.cc pulls in hip_rocm_version_info.h, which (for ROCM_VERSION >= 60000)
// includes <rocm-core/rocm_version.h> for the compile-time ROCm version macros.
// On a box with no /opt/rocm this header is absent, so provide the macros the
// microtest build needs. (p2p.cc did not include this chain, so this stub is
// only needed once init.cc comes under the standalone build.)

#pragma once

#ifndef ROCM_VERSION_MAJOR
#define ROCM_VERSION_MAJOR 7
#endif
#ifndef ROCM_VERSION_MINOR
#define ROCM_VERSION_MINOR 0
#endif
#ifndef ROCM_VERSION_PATCH
#define ROCM_VERSION_PATCH 2
#endif
#ifndef ROCM_BUILD_INFO
#define ROCM_BUILD_INFO "7.0.2-microtest"
#endif
