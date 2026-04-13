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

#endif // ROCJITSU_BASE_RJ_COMPILER_H_
