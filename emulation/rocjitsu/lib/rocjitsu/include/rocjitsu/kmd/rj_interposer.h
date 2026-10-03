// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_interposer.h
/// @brief Startup-only activation of the preloaded Linux interposer.

#ifndef ROCJITSU_KMD_RJ_INTERPOSER_H_
#define ROCJITSU_KMD_RJ_INTERPOSER_H_

#include "rocjitsu/base/rj_compiler.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup kmd
/// @{

/// @brief Query whether this process owns global activation, including inheritance.
/// @returns Nonzero after global activation or in an exec descendant that inherited
/// it; zero for configured CLI launches, pending activation, and a fork child
/// that inherited an initialized GPU context. This is not a runtime health check.
RJ_API_EXPORT int rj_interposer_is_enabled_v1(void);

/// @brief Activate local simulation in a --preload-only process.
/// @details Call once, before any GPU discovery, without concurrent GPU
/// initialization. Activation cannot be undone. Configured CLI/Session launches
/// and DBT configurations are not accepted. The source config can subsequently
/// be changed or removed; lazy initialization reads the private snapshot.
///
/// This function does not modify the environment. To enable inheritance by exec
/// children, the caller must export ROCJITSU_INVOCATION_DIR=runtime_directory
/// and ROCJITSU_PROGRAMMATIC=2 after success, preserving LD_PRELOAD. Python's
/// enable() does this automatically. Keep the directory until the process and
/// its exec children exit. Forking after GPU initialization requires exec.
/// An exec descendant can repeat activation with the same effective config;
/// EALREADY reports that its existing handoff must be preserved.
///
/// Unlike VM object APIs, this Linux interposer entry point uses errno values
/// so callers can distinguish a busy process from unsupported startup state.
/// No exceptions cross this C boundary.
/// @param[in] config_path Input simulation JSON file.
/// @param[in] runtime_directory Private, existing writable directory owned by the
/// caller; receives enabled.json and config_path on success.
/// @param[in] cpu_thread_budget Optional CLI-equivalent budget override (null: none).
/// @param[out] error Receives a NUL-terminated diagnostic, truncated to error_size;
/// may be null when error_size is zero.
/// @param[in] error_size Capacity of error, including the terminating NUL.
/// @retval 0 Activation succeeded and a new handoff was written.
/// @retval EALREADY Exec descendant already uses this effective config; no handoff was written.
/// @retval EINVAL Invalid config or snapshot I/O failure.
/// @retval EBUSY Activation/discovery already attempted, or inherited active fork context.
/// @retval ENOTSUP Not preloaded, or process launched with a native configuration.
RJ_API_EXPORT int rj_interposer_enable_v1(const char *config_path, const char *runtime_directory,
                                          const uint32_t *cpu_thread_budget, char *error,
                                          size_t error_size);

/// @}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROCJITSU_KMD_RJ_INTERPOSER_H_
