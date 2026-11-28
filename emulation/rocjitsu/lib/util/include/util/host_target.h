// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file host_target.h
/// @brief Compile-time host capabilities and SIMD backend selection.

#pragma once

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

/// @brief Small-model x86-64 ELF uses raw function addresses and can resolve signed
/// 32-bit code offsets at link time.
#if RJ_HOST_ARCH_X86_64 && defined(__ELF__) && defined(__code_model_small__) &&                    \
    defined(__LP64__) && (defined(__GNUC__) || defined(__clang__))
#define RJ_HOST_HAS_X86_64_ELF_REL32 1
#else
#define RJ_HOST_HAS_X86_64_ELF_REL32 0
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

/// @brief Availability of the GNU x86 pause builtin used by bounded spin loops.
/// Architecture support alone is insufficient for this compiler-specific entry point.
#if RJ_HOST_ARCH_X86 && (defined(__GNUC__) || defined(__clang__))
#define RJ_HOST_HAS_GNU_X86_PAUSE_BUILTIN 1
#else
#define RJ_HOST_HAS_GNU_X86_PAUSE_BUILTIN 0
#endif

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
