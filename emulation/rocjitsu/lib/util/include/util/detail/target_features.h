// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DETAIL_TARGET_FEATURES_H_
#define UTIL_DETAIL_TARGET_FEATURES_H_

// Compile-time x86 conversion eligibility. Preserve the existing defined/undefined
// macros, including the F16C gate shared by BF16 and integer block conversions.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#define UTIL_ARCH_X86 1
#if defined(__AVX512F__) || defined(__F16C__)
#define UTIL_HAS_X86_F16C 1
#endif
#endif

#endif // UTIL_DETAIL_TARGET_FEATURES_H_
