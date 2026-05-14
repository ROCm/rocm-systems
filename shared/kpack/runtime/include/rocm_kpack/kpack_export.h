// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCM_KPACK_EXPORT_H
#define ROCM_KPACK_EXPORT_H

// Symbol visibility control for shared/static library builds
//
// This header provides cross-platform macros for controlling symbol visibility.
// - Shared library: Only KPACK_API functions are exported
// - Static library: Everything is compiled with hidden visibility by default
//
// On POSIX (GCC/Clang):
//   - Default visibility is hidden (-fvisibility=hidden)
//   - KPACK_API marks symbols as visible
//
// On Windows (MSVC):
//   - KPACK_API uses __declspec(dllexport) when building the DLL
//   - KPACK_API uses __declspec(dllimport) when consuming the DLL
//   - For static library, KPACK_API is empty

#if defined(_WIN32) || defined(__CYGWIN__)
// Windows DLL import/export
#ifdef ROCM_KPACK_STATIC
// Static library: no decoration needed
#define KPACK_API
#else
// Shared library
#ifdef ROCM_KPACK_EXPORTS
// Building the DLL
#define KPACK_API __declspec(dllexport)
#else
// Consuming the DLL
#define KPACK_API __declspec(dllimport)
#endif
#endif
#else
// POSIX (GCC/Clang)
#if defined(__GNUC__) || defined(__clang__)
// Mark public API as visible (default visibility is hidden)
#define KPACK_API __attribute__((visibility("default")))
#else
#define KPACK_API
#endif
#endif

#endif  // ROCM_KPACK_EXPORT_H
