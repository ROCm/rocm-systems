// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_compiler.h
/// @brief Compiler-specific macros for the rocjitsu C API.

#ifndef ROCJITSU_BASE_RJ_COMPILER_H_
#define ROCJITSU_BASE_RJ_COMPILER_H_

/// @brief Marks a symbol for export from the shared library.
#if defined(_WIN32) || defined(__CYGWIN__)
#define RJ_API_EXPORT __declspec(dllexport)
#else
#define RJ_API_EXPORT __attribute__((visibility("default")))
#endif

/// @brief Suppress specific compiler warnings around third-party headers.
///
/// Usage:
/// @code
///   RJ_DIAGNOSTIC_PUSH
///   RJ_DIAGNOSTIC_IGNORE_PEDANTIC
///   #include "third_party/header.h"
///   RJ_DIAGNOSTIC_POP
/// @endcode
#if defined(__clang__)
#define RJ_DIAGNOSTIC_PUSH _Pragma("GCC diagnostic push")
#define RJ_DIAGNOSTIC_POP _Pragma("GCC diagnostic pop")
#define RJ_DIAGNOSTIC_IGNORE_PEDANTIC _Pragma("GCC diagnostic ignored \"-Wpedantic\"")
#define RJ_DIAGNOSTIC_IGNORE_NONNULL_COMPARE                                                       \
  _Pragma("GCC diagnostic ignored \"-Wpointer-bool-conversion\"")
#define RJ_DIAGNOSTIC_IGNORE_CLOBBERED // Clang does not have -Wclobbered.
#elif defined(__GNUC__)
#define RJ_DIAGNOSTIC_PUSH _Pragma("GCC diagnostic push")
#define RJ_DIAGNOSTIC_POP _Pragma("GCC diagnostic pop")
#define RJ_DIAGNOSTIC_IGNORE_PEDANTIC _Pragma("GCC diagnostic ignored \"-Wpedantic\"")
#define RJ_DIAGNOSTIC_IGNORE_NONNULL_COMPARE _Pragma("GCC diagnostic ignored \"-Wnonnull-compare\"")
#define RJ_DIAGNOSTIC_IGNORE_CLOBBERED _Pragma("GCC diagnostic ignored \"-Wclobbered\"")
#elif defined(_MSC_VER)
#define RJ_DIAGNOSTIC_PUSH __pragma(warning(push))
#define RJ_DIAGNOSTIC_POP __pragma(warning(pop))
#define RJ_DIAGNOSTIC_IGNORE_PEDANTIC __pragma(warning(disable : 4200))
#define RJ_DIAGNOSTIC_IGNORE_NONNULL_COMPARE
#define RJ_DIAGNOSTIC_IGNORE_CLOBBERED
#else
static_assert(false, "Unsupported compiler: define RJ_DIAGNOSTIC macros for your toolchain");
#endif

#endif // ROCJITSU_BASE_RJ_COMPILER_H_
