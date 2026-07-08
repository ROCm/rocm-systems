// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file export.h
/// @brief Visibility / DLL-export macro for libsimdojo's public ABI.

#ifndef SIMDOJO_EXPORT_H_
#define SIMDOJO_EXPORT_H_

/// @def SIMDOJO_EXPORT
/// @brief Marks a type or function as part of libsimdojo's exported ABI.
///
/// @details On Windows the annotated symbol is exported from simdojo.dll while
/// the library itself is built (CMake defines `simdojo_EXPORTS` for simdojo's
/// own translation units) and imported by every consumer that links it. On
/// other toolchains it forces default visibility so the symbol survives
/// `-fvisibility=hidden`. This mirrors the ROCJITSU_PLUGIN_EXPORT convention
/// but adds the import side that a linked-against library needs (plugins are
/// only ever dlopen()'d, so they export unconditionally).
#if defined(_WIN32)
#if defined(simdojo_EXPORTS)
#define SIMDOJO_EXPORT __declspec(dllexport)
#else
#define SIMDOJO_EXPORT __declspec(dllimport)
#endif
#else
#define SIMDOJO_EXPORT __attribute__((visibility("default")))
#endif

#endif // SIMDOJO_EXPORT_H_
