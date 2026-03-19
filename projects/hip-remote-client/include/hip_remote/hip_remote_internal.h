/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file hip_remote_internal.h
 * @brief Internal client machinery for remote HIP execution
 *
 * This header provides the internal API for managing connections to
 * the remote HIP worker service and sending/receiving protocol messages.
 * Source files implementing HIP API functions should include this header.
 */

#ifndef HIP_REMOTE_INTERNAL_H
#define HIP_REMOTE_INTERNAL_H

#include "hip_remote/hip_remote_client.h"
#include "hip_remote/hip_remote_protocol.h"
#include "hip_remote/hip_remote_platform.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Client State
 * ============================================================================ */

/**
 * Client connection state
 */
typedef struct {
    hip_socket_t socket_fd;         /**< Socket file descriptor */
    hip_mutex_t lock;               /**< Mutex for thread safety */
    uint32_t next_request_id;       /**< Next request ID */
    bool connected;                 /**< Connection status */
    bool debug_enabled;             /**< Debug logging enabled */
    char worker_host[256];          /**< Worker hostname */
    int worker_port;                /**< Worker port */
    int connect_timeout_sec;        /**< Connection timeout (seconds) */
    int io_timeout_sec;             /**< I/O timeout (seconds) */
    hipError_t last_error;          /**< Last error code */
} HipRemoteClientState;

/**
 * Get the global client state.
 * Thread-safe after first call.
 */
HipRemoteClientState* hip_remote_get_client_state(void);

/* ============================================================================
 * Connection Management
 * ============================================================================ */

/**
 * Ensure client is connected to worker.
 * Automatically connects on first call or after disconnect.
 *
 * @return 0 on success, -1 on failure
 */
int hip_remote_ensure_connected(void);

/**
 * Disconnect from worker.
 */
void hip_remote_disconnect(void);

/**
 * Check if client is connected.
 */
bool hip_remote_is_connected(void);

/* ============================================================================
 * Message I/O
 * ============================================================================ */

/**
 * Send a request and receive a response (synchronous).
 *
 * @param op_code Operation code
 * @param request Request payload (may be NULL)
 * @param request_size Request payload size
 * @param response Response buffer
 * @param response_size Response buffer size
 * @return hipError_t from response, or error if communication failed
 */
hipError_t hip_remote_request(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    void* response,
    size_t response_size
);

/**
 * Send a request with inline data and receive response.
 *
 * @param op_code Operation code
 * @param request Request payload
 * @param request_size Request payload size
 * @param data Inline data to send
 * @param data_size Inline data size
 * @param response Response buffer
 * @param response_size Response buffer size
 * @return hipError_t from response, or error if communication failed
 */
hipError_t hip_remote_request_with_data(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    const void* data,
    size_t data_size,
    void* response,
    size_t response_size
);

/**
 * Send a request and receive response with inline data.
 *
 * @param op_code Operation code
 * @param request Request payload
 * @param request_size Request payload size
 * @param response Response buffer
 * @param response_size Response buffer size
 * @param data_out Buffer for received inline data
 * @param data_size Size of data to receive
 * @return hipError_t from response, or error if communication failed
 */
hipError_t hip_remote_request_receive_data(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    void* response,
    size_t response_size,
    void* data_out,
    size_t data_size
);

/**
 * Fire-and-forget request — sends the request but does NOT wait for a response.
 * Used for async GPU operations (kernel launches, memset, etc.) to eliminate
 * round-trip latency. Errors are deferred to the next sync operation.
 */
hipError_t hip_remote_request_fire_and_forget(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size
);

/**
 * Fire-and-forget request with inline data (e.g. H2D memcpy).
 */
hipError_t hip_remote_request_with_data_fire_and_forget(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    const void* data,
    size_t data_size
);

/**
 * Flush the client-side write buffer. Must be called before any synchronous
 * operation that reads from the GPU (D2H memcpy, event query, synchronize).
 * Automatically called by hip_remote_request() and related synchronous calls.
 */
void hip_remote_flush(void);

/**
 * Returns non-zero if a CUDA graph capture is in progress.
 * Used by hipMalloc to fall back to synchronous allocation during capture,
 * since FnF MALLOC_VADDR would execute outside the capture context.
 */
int hip_remote_is_capturing(void);

/* ============================================================================
 * Content Cache
 * ============================================================================ */

int hip_remote_content_cache_enabled(void);
int hip_remote_content_cache_has(uint64_t hash);
void hip_remote_content_cache_add(uint64_t hash);

/* ============================================================================
 * Logging
 * ============================================================================ */

/**
 * Log a debug message (only if debug enabled).
 */
void hip_remote_log_debug(const char* fmt, ...);

/**
 * Log an error message.
 */
void hip_remote_log_error(const char* fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* HIP_REMOTE_INTERNAL_H */
