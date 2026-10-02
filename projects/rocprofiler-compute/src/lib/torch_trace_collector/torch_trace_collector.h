// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// Plain-C interface to the collector, which emits ROCTX ranges around PyTorch
// operators through a RecordFunction callback.

#pragma once

#include <stdint.h>

#define TORCH_TRACE_COLLECTOR_ABI_REVISION   1U
#define TORCH_TRACE_COLLECTOR_STATS_ABI_SIZE 96U

#if defined(__GNUC__)
#    define TORCH_TRACE_COLLECTOR_EXPORT __attribute__((visibility("default")))
#else
#    define TORCH_TRACE_COLLECTOR_EXPORT
#endif

#if defined(__cplusplus)
#    define TORCH_TRACE_COLLECTOR_NODISCARD [[nodiscard]]
#elif defined(__GNUC__)
#    define TORCH_TRACE_COLLECTOR_NODISCARD __attribute__((warn_unused_result))
#else
#    define TORCH_TRACE_COLLECTOR_NODISCARD
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/** Fixed-width snapshot of the live-stack collector counters. */
struct torch_trace_collector_stats
{
    uint32_t struct_size;
    uint32_t installed;
    uint64_t pushes;
    uint64_t pops;
    uint64_t user_scope_pushes;
    uint64_t user_scope_pops;
    uint64_t user_scope_inherits;
    uint64_t snapshots_saved;
    uint64_t snapshots_consumed;
    uint64_t snapshots_dropped;
    uint64_t snapshots_overwritten;
    uint64_t callback_errors;
    uint64_t snapshots_pending;
};

/** @return TORCH_TRACE_COLLECTOR_ABI_REVISION. */
TORCH_TRACE_COLLECTOR_NODISCARD TORCH_TRACE_COLLECTOR_EXPORT uint32_t torch_trace_collector_abi_revision(void);

/**
 * Installs the global RecordFunction callback that emits a ROCTX range around
 * each PyTorch operator. Idempotent.
 * @return Zero on success and nonzero on failure.
 */
TORCH_TRACE_COLLECTOR_NODISCARD TORCH_TRACE_COLLECTOR_EXPORT int32_t torch_trace_collector_install(void);

/**
 * Removes the registered callback and clears snapshots.
 * @return Zero on success and nonzero on failure.
 */
TORCH_TRACE_COLLECTOR_NODISCARD TORCH_TRACE_COLLECTOR_EXPORT int32_t torch_trace_collector_uninstall(void);

/** @return One when the callback is installed, otherwise zero. */
TORCH_TRACE_COLLECTOR_NODISCARD TORCH_TRACE_COLLECTOR_EXPORT int32_t torch_trace_collector_is_installed(void);

/**
 * Pushes a marker frame, emits a ROCTX range, and publishes the thread stack
 * to ThreadLocalDebugInfo so autograd workers inherit it. The collector must
 * already be installed and all strings must be non-null UTF-8.
 * @return Zero on success and nonzero on failure.
 */
TORCH_TRACE_COLLECTOR_NODISCARD TORCH_TRACE_COLLECTOR_EXPORT int32_t
    torch_trace_collector_push_user_scope(const char* marker, const char* context, const char* backend);

/**
 * Pops the most recent torch_trace_collector_push_user_scope frame on the
 * calling thread.
 * @return Zero on success and nonzero when the thread has no frame to pop.
 */
TORCH_TRACE_COLLECTOR_NODISCARD TORCH_TRACE_COLLECTOR_EXPORT int32_t torch_trace_collector_pop_user_scope(void);

/**
 * Copies the collector counters into @p stats. The caller must initialize
 * `stats->struct_size` to at least `sizeof(struct torch_trace_collector_stats)`.
 * @return Zero on success and nonzero on invalid input or failure.
 */
TORCH_TRACE_COLLECTOR_NODISCARD TORCH_TRACE_COLLECTOR_EXPORT int32_t
    torch_trace_collector_get_stats(struct torch_trace_collector_stats* stats);

#ifdef __cplusplus
}
#endif

#undef TORCH_TRACE_COLLECTOR_EXPORT
#undef TORCH_TRACE_COLLECTOR_NODISCARD
