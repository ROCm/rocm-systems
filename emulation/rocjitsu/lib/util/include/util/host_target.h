// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file host_target.h
/// @brief Compile-time host capabilities and SIMD backend selection.

#pragma once

#include <version>

/// @brief Architecture predicates retain the host spellings used by existing
/// execution helpers, including the x86-wide MSVC spelling. All checks use #if
/// because every predicate is defined, including when its value is zero.
#if defined(__x86_64__) || defined(_M_X64)
#define RJ_HOST_ARCH_X86_64 1
#else
#define RJ_HOST_ARCH_X86_64 0
#endif

#if RJ_HOST_ARCH_X86_64 || defined(__i386__)
#define RJ_HOST_ARCH_X86 1
#else
#define RJ_HOST_ARCH_X86 0
#endif

#if defined(__aarch64__)
#define RJ_HOST_ARCH_AARCH64 1
#else
#define RJ_HOST_ARCH_AARCH64 0
#endif

/// @brief Individual ISA features remain independent, so a partial target does
/// not acquire other capabilities merely by enabling AVX-512 or AVX2.
#if defined(__AVX2__)
#define RJ_HOST_HAS_AVX2 1
#else
#define RJ_HOST_HAS_AVX2 0
#endif

#if defined(__AVX512F__)
#define RJ_HOST_HAS_AVX512F 1
#else
#define RJ_HOST_HAS_AVX512F 0
#endif

#if defined(__F16C__)
#define RJ_HOST_HAS_F16C 1
#else
#define RJ_HOST_HAS_F16C 0
#endif

#if defined(__FMA__)
#define RJ_HOST_HAS_FMA 1
#else
#define RJ_HOST_HAS_FMA 0
#endif

/// @brief Availability of the GNU x86 pause builtin used by bounded spin loops.
/// Architecture support alone is insufficient for this compiler-specific entry point.
#if RJ_HOST_ARCH_X86 && (defined(__GNUC__) || defined(__clang__))
#define RJ_HOST_HAS_GNU_X86_PAUSE_BUILTIN 1
#else
#define RJ_HOST_HAS_GNU_X86_PAUSE_BUILTIN 0
#endif

/// @brief Preserve the existing x86 block-conversion backend selection.
/// @details This is an implementation choice shared by half, bf16 and integer
/// converters. It does not imply that every conversion instruction needs F16C.
#define RJ_HOST_USE_X86_BLOCK_CONVERSIONS                                                          \
  (RJ_HOST_ARCH_X86 && (RJ_HOST_HAS_AVX512F || RJ_HOST_HAS_F16C))

/// @brief GNU-compatible x86-64 targets with the indicated ISA enabled.
/// @details These narrower capabilities preserve the qualified target set used
/// by dense stride checks. They are distinct from width-independent ISA bits
/// used by the existing conversion and register helpers.
#if RJ_HOST_ARCH_X86_64 && (defined(__GNUC__) || defined(__clang__))
#define RJ_HOST_HAS_GNU_X86_64_AVX512F RJ_HOST_HAS_AVX512F
#define RJ_HOST_HAS_GNU_X86_64_AVX2 RJ_HOST_HAS_AVX2
#else
#define RJ_HOST_HAS_GNU_X86_64_AVX512F 0
#define RJ_HOST_HAS_GNU_X86_64_AVX2 0
#endif

/// @brief GNU-compatible x86-64 targets with either qualified stride ISA.
#define RJ_HOST_HAS_GNU_X86_64_AVX2_OR_AVX512F                                                     \
  (RJ_HOST_HAS_GNU_X86_64_AVX512F || RJ_HOST_HAS_GNU_X86_64_AVX2)

/// @brief Presence of the standard-library experimental SIMD header.
/// @details The qualified implementation is libstdc++; header presence does not
/// establish that another standard library implements every SIMD TS operation.
#if __has_include(<experimental/simd>)
#define RJ_HOST_HAS_STDX_SIMD 1
#else
#define RJ_HOST_HAS_STDX_SIMD 0
#endif

/// @brief Select existing AVX-512 matrix paths when the SIMD TS header is present.
/// The qualified-library limitation of RJ_HOST_HAS_STDX_SIMD also applies here.
#define RJ_HOST_HAS_AVX512_STDX_SIMD (RJ_HOST_HAS_AVX512F && RJ_HOST_HAS_STDX_SIMD)

/// @brief Select the fallback for unreliable native 64-bit SIMD masks on the
/// affected Clang, libstdc++ and AVX-512 combination.
#if defined(__clang__) && defined(__GLIBCXX__) && RJ_HOST_HAS_AVX512F
#define RJ_HOST_NEEDS_STDX_64BIT_MASK_FALLBACK 1
#else
#define RJ_HOST_NEEDS_STDX_64BIT_MASK_FALLBACK 0
#endif

/// @brief Preserve the existing command-line override for fixed toolchains.
/// @details Record whether the user supplied the policy, then apply the detected
/// default only when no override was supplied. All consumers see the same policy
/// regardless of which utility or standard header was included first.
/// Use -DRJ_SIMD_BROKEN_NATIVE_64BIT_MASKS=0 only with a toolchain that can
/// compile native 64-bit SIMD masks; affected toolchains require the fallback.
#ifdef RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS
#define RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS_USER_OVERRIDE 1
#else
#define RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS_USER_OVERRIDE 0
#define RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS RJ_HOST_NEEDS_STDX_64BIT_MASK_FALLBACK
#endif
