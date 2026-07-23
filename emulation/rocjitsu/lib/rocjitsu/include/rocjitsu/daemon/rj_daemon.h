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

/// @brief Stop a daemon and release its handle.
///
/// @details Stops accepting clients, unblocks and joins every client thread,
/// stops and joins the simulation engine, destroys the VM, and removes the
/// socket if it is still the endpoint created by rj_daemon_start. Passing NULL
/// is a successful no-op. The handle is invalid after this function returns;
/// callers must synchronize concurrent status queries before stopping it.
///
/// @param[in] daemon Daemon handle, or NULL.
/// @retval ROCJITSU_STATUS_SUCCESS The daemon was stopped and released.
RJ_API_EXPORT rj_status_t rj_daemon_stop(rj_daemon_t *daemon);

/// @brief Return the daemon's current lifecycle state.
///
/// @details Thread-safe while the handle remains valid. A NULL handle reports
/// RJ_DAEMON_STATUS_STOPPED.
///
/// @note Do NOT call on a daemon after rj_daemon_stop has been called,
/// as the handle is invalid and may have been freed.
///
/// @param[in] daemon Daemon handle, or NULL.
RJ_API_EXPORT rj_daemon_status_t rj_daemon_status(const rj_daemon_t *daemon);

/// @}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROCJITSU_DAEMON_RJ_DAEMON_H_
