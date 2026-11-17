// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_daemon.h
/// @brief Public C API for hosting a RocJitsu VM over the daemon RPC transport.

#ifndef ROCJITSU_DAEMON_RJ_DAEMON_H_
#define ROCJITSU_DAEMON_RJ_DAEMON_H_

#include <stdint.h>

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/base/rj_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup daemon
/// @{

/// @brief Opaque handle for a running daemon.
typedef struct rj_daemon_t rj_daemon_t;

/// @brief Observable daemon lifecycle state.
typedef int32_t rj_daemon_status_t;

/// @brief Named values for @ref rj_daemon_status_t.
enum {
  RJ_DAEMON_STATUS_STOPPED = 0,
  RJ_DAEMON_STATUS_STARTING = 1,
  RJ_DAEMON_STATUS_RUNNING = 2,
  RJ_DAEMON_STATUS_STOPPING = 3,
  RJ_DAEMON_STATUS_ERROR = 4,
};

/// @brief Optional daemon diagnostics selected before the VM starts.
/// @details fabric_trace_capacity must be zero unless
/// RJ_DAEMON_OPTION_TRACE_FABRIC_EFFECTS is set. When tracing is enabled, zero
/// selects RJ_DAEMON_FABRIC_TRACE_MAX_CAPACITY; nonzero values from 1 through
/// that limit are accepted. Larger values are rejected before VM construction.
typedef struct rj_daemon_options_t {
  uint32_t struct_size;           ///< Set to sizeof(rj_daemon_options_t).
  uint32_t flags;                 ///< Bitwise OR of RJ_DAEMON_OPTION_* values.
  uint32_t fabric_trace_capacity; ///< Zero selects RJ_DAEMON_FABRIC_TRACE_MAX_CAPACITY.
} rj_daemon_options_t;

enum { RJ_DAEMON_OPTION_TRACE_FABRIC_EFFECTS = 1u << 0 };
enum { RJ_DAEMON_FABRIC_TRACE_MAX_CAPACITY = 200000u };

/// @brief Start a daemon-mode VM and its Unix-socket RPC server.
///
/// @details Parses the supplied JSON configuration and returns only after the
/// socket is bound and listening and the engine and accept threads have
/// started. The caller owns the returned handle and must eventually pass it to
/// rj_daemon_stop. An abandoned socket is recovered, while a live socket or
/// non-socket path is never removed.
///
/// @param[in] json VM JSON configuration string.
/// @param[in] socket_path Filesystem path for the Unix domain socket.
/// @param[out] daemon Newly created daemon handle. Set to NULL on failure.
/// @retval ROCJITSU_STATUS_SUCCESS The daemon is running.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT A required argument is NULL, the
/// JSON configuration is invalid, or the socket path cannot be represented by
/// sockaddr_un.
/// @retval ROCJITSU_STATUS_OUT_OF_RESOURCES Allocation or thread creation failed.
/// @retval ROCJITSU_STATUS_ERROR Socket setup or VM initialization failed.
RJ_API_EXPORT rj_status_t rj_daemon_start(const char *json, const char *socket_path,
                                          rj_daemon_t **daemon);

/// @brief Start a daemon with explicit optional diagnostics.
/// @details NULL options have the same behavior as rj_daemon_start. Unknown
/// flags and a mismatched struct_size are rejected before VM construction.
RJ_API_EXPORT rj_status_t rj_daemon_start_with_options(const char *json, const char *socket_path,
                                                       const rj_daemon_options_t *options,
                                                       rj_daemon_t **daemon);

/// @brief Stop a daemon and release its handle.
///
/// @details Stops accepting clients, unblocks and joins every client thread,
/// stops and joins the simulation engine, destroys the VM, and removes the
/// socket if it is still the endpoint created by rj_daemon_start. Passing NULL
/// is a successful no-op. On success the handle is invalid. If final process
/// maintenance fails, the stopped daemon retains its VM and retryable close
/// state; the caller must retain the handle and call rj_daemon_stop again.
/// Dropped diagnostic trace records are logged separately and do not make a
/// completed VM teardown retryable.
/// Callers must synchronize concurrent status queries before stopping it.
///
/// @param[in] daemon Daemon handle, or NULL.
/// @retval ROCJITSU_STATUS_SUCCESS The daemon was stopped and released.
/// @retval ROCJITSU_STATUS_ERROR Final process maintenance failed. errno holds
/// the first driver error and the daemon handle remains valid for retry.
RJ_API_EXPORT rj_status_t rj_daemon_stop(rj_daemon_t *daemon);

/// @brief Return the daemon's current lifecycle state.
///
/// @details Thread-safe while the handle remains valid. A NULL handle reports
/// RJ_DAEMON_STATUS_STOPPED.
///
/// @note Do NOT call on a daemon after rj_daemon_stop succeeds, as the handle
/// is invalid and may have been freed. A failed stop retains the handle.
///
/// @param[in] daemon Daemon handle, or NULL.
RJ_API_EXPORT rj_daemon_status_t rj_daemon_status(const rj_daemon_t *daemon);

/// @}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROCJITSU_DAEMON_RJ_DAEMON_H_
