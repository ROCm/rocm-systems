// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_threads.h
/// @brief Public C API for evaluating a config's execution-thread allocation.

#ifndef ROCJITSU_CONFIG_RJ_THREADS_H_
#define ROCJITSU_CONFIG_RJ_THREADS_H_

#include <stddef.h>
#include <stdint.h>

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/base/rj_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup config
/// @{

/// @brief Report the host thread count the allocator would observe.
///
/// @details The allocation rule is a function of the host width, so a caller
/// rendering the same table for several budgets has to hold that width fixed.
/// Querying it once and passing it back to
/// @ref rj_config_resolve_execution_threads keeps every row of such a table on
/// one basis, which re-querying per row would not guarantee if the process
/// affinity mask changed midway.
///
/// @param[out] out_host_threads Receives the observed host thread count. Never
/// zero; a host whose affinity cannot be read reports one.
/// @retval ROCJITSU_STATUS_SUCCESS The count is stored.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT @p out_host_threads is NULL.
RJ_API_EXPORT rj_status_t rj_config_available_host_threads(uint32_t *out_host_threads);

/// @brief Resolve the execution threads a config would allocate.
///
/// @details Reads only the config's thread requests and topology dimensions and
/// then applies the pure table selector, so it builds no simulator components
/// and starts no worker threads. This exists so a launcher written in another
/// language can show what a config will do before running it, without linking
/// the C++ loader or restating its selection rule.
///
/// The dispatch widths are returned through a caller-provided buffer. Passing
/// NULL for @p out_dispatch stores the count the config would produce, which
/// lets a caller size its buffer in a first call and fill it in a second.
///
/// @param[in] config_path Path to the JSON config to evaluate.
/// @param[in] budget VM-wide thread budget to evaluate. Zero evaluates the
/// budget the config itself requests.
/// @param[in] host_threads Host width to resolve against. Zero queries the host,
/// matching @ref rj_config_available_host_threads.
/// @param[out] out_engines Receives the effective engine count.
/// @param[out] out_dispatch Receives the inclusive dispatch width per SoC. May
/// be NULL to query the count only.
/// @param[in,out] inout_dispatch_count On entry, the capacity of
/// @p out_dispatch in elements; on return, the number of widths the config
/// produces, whether or not they fit.
/// @retval ROCJITSU_STATUS_SUCCESS The allocation is stored.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT A required argument is NULL or
/// @p config_path is empty.
/// @retval ROCJITSU_STATUS_OUT_OF_RESOURCES @p out_dispatch is too small; the
/// required count is stored in @p inout_dispatch_count and no widths are
/// written.
/// @retval ROCJITSU_STATUS_INVALID_FILE The config could not be read or parsed,
/// or its allocation metadata is invalid.
RJ_API_EXPORT rj_status_t rj_config_resolve_execution_threads(
    const char *config_path, uint32_t budget, uint32_t host_threads, uint32_t *out_engines,
    uint32_t *out_dispatch, size_t *inout_dispatch_count);

/// @}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROCJITSU_CONFIG_RJ_THREADS_H_
