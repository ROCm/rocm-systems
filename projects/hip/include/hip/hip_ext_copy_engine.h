/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_HIP_EXT_COPY_ENGINE_H
#define HIP_INCLUDE_HIP_HIP_EXT_COPY_ENGINE_H

#include <hip/hip_runtime.h>

#if defined(__HIP_PLATFORM_AMD__) && defined(__HIPCC__)
#include <hip/amd_detail/amd_hip_ext_copy_engine.h>
#endif

#endif  // HIP_INCLUDE_HIP_HIP_EXT_COPY_ENGINE_H
