/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COMMON_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COMMON_H

#if defined(__clang__) && defined(__HIP__)
#define __HIP_CLANG_ONLY__ 1
#else
#define __HIP_CLANG_ONLY__ 0
#endif

#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COMMON_H
