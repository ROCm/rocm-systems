// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Device-side architecture capability flags (single place for HAS_* macros).
// Include only from HIP device translation units.

#pragma once

// MI250/MI200 series (CDNA2)
#if defined(__gfx90a__)
#    define ARCH_MI250          1
#    define ARCH_NAME           "MI250 (gfx90a/CDNA2)"
#    define HAS_HW_FP64_ATOMICS 1
#    define HAS_FP8             0
#    define HAS_ASYNC_LDS       0
#endif

// MI300 series (CDNA3)
#if defined(__gfx942__)
#    define ARCH_MI300          1
#    define ARCH_NAME           "MI300 (gfx942/CDNA3)"
#    define HAS_HW_FP64_ATOMICS 1
#    define HAS_FP8             1
#    define HAS_ASYNC_LDS       0
#endif

// MI350 series (CDNA4)
#if defined(__gfx950__)
#    define ARCH_MI350          1
#    define ARCH_NAME           "MI350 (gfx950/CDNA4)"
#    define HAS_HW_FP64_ATOMICS 1
#    define HAS_FP8             1
#    define HAS_FP6_FP4         1
#    define HAS_ASYNC_LDS       1
#endif

// Strix Point / Strix Halo / Krackan (RDNA 3.5 APU iGPU)
#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
#    define ARCH_RDNA35_APU     1
#    define ARCH_NAME           "RDNA 3.5 APU (gfx1150/gfx1151/gfx1152 - Strix/Krackan)"
#    define HAS_HW_FP64_ATOMICS 0
#    define HAS_FP8             0
#    define HAS_ASYNC_LDS       1
#endif

// RDNA4 consumer (gfx1200/gfx1201)
#if defined(__gfx1200__) || defined(__gfx1201__)
#    define ARCH_RDNA4_CONSUMER 1
#    define ARCH_NAME           "RDNA4 Consumer (gfx1200/gfx1201 - RX 9070 XT)"
#    define HAS_HW_FP64_ATOMICS 0
#    define HAS_FP8             1
#    define HAS_ASYNC_LDS       1
#endif

#ifndef ARCH_NAME
#    define ARCH_NAME           "Unknown"
#    define HAS_HW_FP64_ATOMICS 0
#    define HAS_FP8             0
#    define HAS_ASYNC_LDS       0
#endif
