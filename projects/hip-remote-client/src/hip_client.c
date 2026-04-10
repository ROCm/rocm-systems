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
 * @file hip_client.c
 * @brief Core client implementation for remote HIP execution
 * Cross-platform: Works on Windows, macOS, and Linux.
 */

#include "hip_remote/hip_remote_internal.h"
#include "hip_remote/hip_remote_protocol.h"
#include "hip_remote/hip_remote_platform.h"
#include "hip_remote/hip_remote_shm.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Global State
 * ============================================================================ */

static HipRemoteClientState g_client_state = {
    .socket_fd = HIP_INVALID_SOCKET,
    .lock = HIP_MUTEX_INIT,
    .next_request_id = 1,
    .connected = false,
    .debug_enabled = false,
    .worker_host = "localhost",
    .worker_port = HIP_REMOTE_DEFAULT_PORT,
    .connect_timeout_sec = 30,
    .io_timeout_sec = 0,  /* no timeout -- keepalive handles dead connections */
    .last_error = hipSuccess
};

static hip_once_t g_init_once = HIP_ONCE_INIT;
static int g_wsa_initialized = 0;
static int g_dll_loading_complete = 0;  /* Set to 1 after DLL loading finishes */

/* Write coalescing buffer: accumulates FnF requests and flushes them in bulk. */
#define WRITE_BUFFER_SIZE (256 * 1024)
static uint8_t g_write_buffer[WRITE_BUFFER_SIZE];
static size_t  g_write_buffer_used = 0;

/* Eager flush: send accumulated FnF data every N requests so the worker
 * starts processing sooner, reducing GPU starvation during cold start.
 * Disabled (0) for high-latency links where fewer larger sends are better. */
static int g_eager_flush_interval = 0;
static int g_fnf_since_flush = 0;

/* Shared memory IPC transport (for localhost) */
static int g_use_shm = 0;
static HipShmHandle g_shm;
static int g_shm_initialized = 0;

/* ============================================================================
 * Profiling Counters
 * ============================================================================ */

#define PROF_MAX_OPCODE 4096

static uint64_t g_prof_call_count[PROF_MAX_OPCODE];
static uint64_t g_prof_data_bytes[PROF_MAX_OPCODE];
static int g_prof_enabled = 0;

static void prof_record(uint16_t op_code, uint64_t data_bytes) {
    if (g_prof_enabled && op_code < PROF_MAX_OPCODE) {
        g_prof_call_count[op_code]++;
        g_prof_data_bytes[op_code] += data_bytes;
    }
}

static void prof_dump(void) {
    if (!g_prof_enabled) return;

    typedef struct { uint16_t op; uint64_t count; uint64_t bytes; } entry_t;
    entry_t entries[256];
    int n = 0;

    for (int i = 0; i < PROF_MAX_OPCODE && n < 256; i++) {
        if (g_prof_call_count[i] > 0) {
            entries[n].op = (uint16_t)i;
            entries[n].count = g_prof_call_count[i];
            entries[n].bytes = g_prof_data_bytes[i];
            n++;
        }
    }

    /* Sort by count descending (simple insertion sort) */
    for (int i = 1; i < n; i++) {
        entry_t key = entries[i];
        int j = i - 1;
        while (j >= 0 && entries[j].count < key.count) {
            entries[j + 1] = entries[j];
            j--;
        }
        entries[j + 1] = key;
    }

    uint64_t total_calls = 0, total_bytes = 0;
    for (int i = 0; i < n; i++) {
        total_calls += entries[i].count;
        total_bytes += entries[i].bytes;
    }

    fprintf(stderr, "\n[HIP-Remote PROFILE] === API Call Summary ===\n");
    fprintf(stderr, "  %-45s %10s %15s\n", "Operation", "Calls", "Data(bytes)");
    fprintf(stderr, "  %-45s %10s %15s\n", "---------", "-----", "-----------");
    for (int i = 0; i < n; i++) {
        fprintf(stderr, "  %-45s %10llu %15llu\n",
                hip_remote_op_name((HipRemoteOpCode)entries[i].op),
                (unsigned long long)entries[i].count,
                (unsigned long long)entries[i].bytes);
    }
    fprintf(stderr, "  %-45s %10llu %15llu\n",
            "TOTAL", (unsigned long long)total_calls, (unsigned long long)total_bytes);
    fprintf(stderr, "[HIP-Remote PROFILE] === End Summary ===\n\n");
}

/* ============================================================================
 * Content Cache Hash Set
 *
 * Tracks which content hashes have been acknowledged by the worker's GPU
 * cache.  When a hash is present the client skips the data payload, sending
 * only the hash for a fast D2D copy on the worker side.
 * ============================================================================ */

#define XXH_INLINE_ALL
#include "hip_remote/xxhash.h"

#define CONTENT_CACHE_INITIAL 4096

static uint64_t* g_content_hashes = NULL;
static int g_content_hash_count = 0;
static int g_content_hash_capacity = 0;
static int g_content_cache_enabled = 0;

int hip_remote_content_cache_enabled(void) {
    return g_content_cache_enabled;
}

int hip_remote_content_cache_has(uint64_t hash) {
    for (int i = 0; i < g_content_hash_count; i++) {
        if (g_content_hashes[i] == hash) return 1;
    }
    return 0;
}

void hip_remote_content_cache_add(uint64_t hash) {
    if (hip_remote_content_cache_has(hash)) return;
    if (g_content_hash_count >= g_content_hash_capacity) {
        int new_cap = g_content_hash_capacity == 0
            ? CONTENT_CACHE_INITIAL
            : g_content_hash_capacity * 2;
        uint64_t* new_arr = (uint64_t*)realloc(
            g_content_hashes, new_cap * sizeof(uint64_t));
        if (!new_arr) return;
        g_content_hashes = new_arr;
        g_content_hash_capacity = new_cap;
    }
    g_content_hashes[g_content_hash_count++] = hash;
}

static void content_cache_reset(void) {
    g_content_hash_count = 0;
}

/* ============================================================================
 * Logging
 * ============================================================================ */

void hip_remote_log_debug(const char* fmt, ...) {
    if (!g_client_state.debug_enabled) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[HIP-Remote] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void hip_remote_log_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[HIP-Remote ERROR] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

/**
 * Send all bytes, handling partial sends.
 */
static int send_all(hip_socket_t fd, const void* buf, size_t len) {
    const uint8_t* p = (const uint8_t*)buf;
    while (len > 0) {
        int n = hip_send(fd, p, len, HIP_MSG_NOSIGNAL);
        if (n < 0) {
            int err = hip_socket_errno();
            if (err == HIP_EINTR) continue;
            return -1;
        }
        if (n == 0) {
            return -1;
        }
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

/**
 * Scatter-gather send: sends up to 8 iovecs in a single syscall where
 * possible, falling back to sequential send_all for remainder.
 */
static int send_all_v(hip_socket_t fd, hip_iovec_t* iov, int iovcnt) {
    while (iovcnt > 0) {
        /* Skip zero-length or NULL entries */
        if (iov[0].len == 0 || iov[0].base == NULL) {
            iov++;
            iovcnt--;
            continue;
        }

        int n = hip_sendv(fd, iov, iovcnt);
        if (n < 0) {
            int err = hip_socket_errno();
            if (err == HIP_EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;

        /* Advance past fully-sent iovecs */
        size_t sent = (size_t)n;
        while (iovcnt > 0 && sent >= iov[0].len) {
            sent -= iov[0].len;
            iov++;
            iovcnt--;
        }
        /* Partial iovec: adjust base and len */
        if (iovcnt > 0 && sent > 0) {
            iov[0].base = (const char*)iov[0].base + sent;
            iov[0].len -= sent;
        }
    }
    return 0;
}

/**
 * Receive all bytes, handling partial receives.
 */
static int recv_all(hip_socket_t fd, void* buf, size_t len) {
    uint8_t* p = (uint8_t*)buf;
    while (len > 0) {
        int n = hip_recv(fd, p, len, 0);
        if (n < 0) {
            int err = hip_socket_errno();
            if (err == HIP_EINTR) continue;
            return -1;
        }
        if (n == 0) {
            return -1;
        }
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static void mark_disconnected_locked(const char* reason);

/**
 * Flush the write buffer to the socket. Must be called with lock held.
 * Returns 0 on success, -1 on error (and marks disconnected).
 */
static int flush_write_buffer_locked(void) {
    if (g_write_buffer_used == 0) return 0;
    if (!g_client_state.connected) return -1;

    int rc;
    if (g_use_shm && g_shm_initialized) {
        rc = hip_shm_fnf_write(&g_shm, g_write_buffer, g_write_buffer_used);
    } else {
        rc = send_all(g_client_state.socket_fd, g_write_buffer, g_write_buffer_used);
    }
    if (rc != 0) {
        mark_disconnected_locked("flush write buffer");
        g_write_buffer_used = 0;
        g_fnf_since_flush = 0;
        return -1;
    }
    g_write_buffer_used = 0;
    g_fnf_since_flush = 0;
    return 0;
}

/**
 * Mark connection as disconnected.
 */
static int g_permanently_disconnected = 0;
static int g_shutdown_in_progress = 0;

static void mark_disconnected_locked(const char* reason) {
    if (reason && !g_shutdown_in_progress) {
        int err = hip_socket_errno();
        hip_remote_log_error("Disconnected: %s (errno=%d: %s)",
                             reason, err, hip_socket_strerror(err));
    }
    if (g_client_state.socket_fd != HIP_INVALID_SOCKET) {
        hip_close_socket(g_client_state.socket_fd);
    }
    g_client_state.socket_fd = HIP_INVALID_SOCKET;
    g_client_state.connected = false;
    /* Mark permanently disconnected. Reconnection would create a new
     * worker child with empty vaddr/function caches, causing silent
     * corruption from stale client-side handles. */
    g_permanently_disconnected = 1;
}

/**
 * Initialize client state from environment.
 */
static void init_from_environment(void) {
    /* Initialize socket subsystem (Winsock on Windows, no-op on POSIX).
     * May fail silently during DLL loading; will be retried later. */
    if (hip_socket_init() == 0) {
        g_wsa_initialized = 1;
    }

    const char* debug = getenv("TF_DEBUG");
    if (debug && strcmp(debug, "1") == 0) {
        g_client_state.debug_enabled = true;
    }

    const char* host = getenv("TF_WORKER_HOST");
    if (host && host[0] != '\0') {
        strncpy(g_client_state.worker_host, host,
                sizeof(g_client_state.worker_host) - 1);
        g_client_state.worker_host[sizeof(g_client_state.worker_host) - 1] = '\0';
    }

    const char* port_str = getenv("TF_WORKER_PORT");
    if (port_str && port_str[0] != '\0') {
        int port = atoi(port_str);
        if (port > 0 && port < 65536) {
            g_client_state.worker_port = port;
        }
    }

    const char* connect_timeout = getenv("TF_CONNECT_TIMEOUT");
    if (connect_timeout && connect_timeout[0] != '\0') {
        int timeout = atoi(connect_timeout);
        if (timeout > 0) {
            g_client_state.connect_timeout_sec = timeout;
        }
    }

    const char* io_timeout = getenv("TF_IO_TIMEOUT");
    if (io_timeout && io_timeout[0] != '\0') {
        int timeout = atoi(io_timeout);
        if (timeout > 0) {
            g_client_state.io_timeout_sec = timeout;
        }
    }

    const char* prof = getenv("HIP_REMOTE_PROFILE");
    if (prof && strcmp(prof, "1") == 0) {
        g_prof_enabled = 1;
        memset(g_prof_call_count, 0, sizeof(g_prof_call_count));
        memset(g_prof_data_bytes, 0, sizeof(g_prof_data_bytes));
    }

    const char* cache = getenv("HIP_REMOTE_CACHE");
    if (cache && strcmp(cache, "1") == 0) {
        g_content_cache_enabled = 1;
    }

    const char* eager = getenv("HIP_REMOTE_EAGER_FLUSH");
    if (eager) {
        g_eager_flush_interval = atoi(eager);
    }

    /* Auto-detect transport: SHM for localhost, TCP for remote */
    const char* transport = getenv("HIP_REMOTE_TRANSPORT");
    {
        const char* host = getenv("TF_WORKER_HOST");
        int is_local = host && (strcmp(host, "localhost") == 0 ||
                                strcmp(host, "127.0.0.1") == 0 ||
                                strcmp(host, "::1") == 0);

        if (transport && strcmp(transport, "shm") == 0) {
            g_use_shm = 1;
        } else if (transport && strcmp(transport, "tcp") == 0) {
            g_use_shm = 0;
        } else {
            g_use_shm = is_local;
        }

        if (!eager && is_local && !g_use_shm) {
            g_eager_flush_interval = 64;
        }
    }

    hip_remote_log_debug("Client initialized: host=%s port=%d transport=%s",
                         g_client_state.worker_host, g_client_state.worker_port,
                         g_use_shm ? "shm" : "tcp");
}

/**
 * Connect to worker service.
 */
static int connect_to_worker_locked(void) {
    hip_call_once(&g_init_once, init_from_environment);

    if (g_client_state.connected) {
        return 0;
    }
    if (g_permanently_disconnected) {
        return -1;
    }

    /* Ensure socket subsystem is initialized */
    if (!g_wsa_initialized) {
        if (hip_socket_init() == 0) {
            g_wsa_initialized = 1;
        } else {
            /* WSAStartup failed — likely called during DLL loading under
             * the loader lock. Return silently; callers will retry later. */
            return -1;
        }
    }

    hip_remote_log_debug("Connecting to %s:%d...",
                         g_client_state.worker_host, g_client_state.worker_port);

    /* Resolve hostname */
    struct addrinfo hints, *result = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", g_client_state.worker_port);

    int gai_err = getaddrinfo(g_client_state.worker_host, port_str, &hints, &result);
    if (gai_err != 0 || !result) {
        hip_remote_log_error("Failed to resolve host: %s", g_client_state.worker_host);
        if (result) freeaddrinfo(result);
        return -1;
    }

    /* Create socket */
    g_client_state.socket_fd = socket(result->ai_family, result->ai_socktype,
                                      result->ai_protocol);
    if (g_client_state.socket_fd == HIP_INVALID_SOCKET) {
        hip_remote_log_error("Failed to create socket: %s",
                             hip_socket_strerror(hip_socket_errno()));
        freeaddrinfo(result);
        return -1;
    }

    /* Set socket options */
    hip_set_nodelay(g_client_state.socket_fd);
    hip_set_socket_timeout(g_client_state.socket_fd, g_client_state.io_timeout_sec);

    /* Connect */
    if (connect(g_client_state.socket_fd,
                result->ai_addr, (int)result->ai_addrlen) < 0) {
        hip_remote_log_error("Failed to connect: %s",
                             hip_socket_strerror(hip_socket_errno()));
        freeaddrinfo(result);
        mark_disconnected_locked("connect");
        return -1;
    }

    freeaddrinfo(result);

    g_client_state.connected = true;
    hip_remote_log_debug("Connected successfully");

    /* Create SHM region if using shared memory transport */
    char shm_name[HIP_SHM_NAME_MAX] = {0};
    if (g_use_shm) {
#ifdef _WIN32
        snprintf(shm_name, sizeof(shm_name), "hip_shm_%lu",
                 (unsigned long)GetCurrentProcessId());
#else
        snprintf(shm_name, sizeof(shm_name), "/hip_shm_%d",
                 (int)getpid());
#endif
        if (hip_shm_create(&g_shm, shm_name) != 0) {
            hip_remote_log_error("Failed to create SHM, falling back to TCP");
            g_use_shm = 0;
            shm_name[0] = '\0';
        } else {
            hip_remote_log_debug("SHM created: %s (%zu bytes)",
                                 shm_name, g_shm.total_size);
        }
    }

    /* Send init message (with SHM name if applicable) */
    uint32_t init_payload_size = shm_name[0] ? (uint32_t)strlen(shm_name) + 1 : 0;
    HipRemoteHeader header;
    hip_remote_init_header(&header, HIP_OP_INIT,
                           g_client_state.next_request_id++,
                           init_payload_size);

    if (send_all(g_client_state.socket_fd, &header, sizeof(header)) != 0) {
        mark_disconnected_locked("send init");
        return -1;
    }
    if (init_payload_size > 0) {
        if (send_all(g_client_state.socket_fd, shm_name, init_payload_size) != 0) {
            mark_disconnected_locked("send init shm name");
            return -1;
        }
    }

    /* Receive init response */
    HipRemoteHeader resp_header;
    if (recv_all(g_client_state.socket_fd, &resp_header, sizeof(resp_header)) != 0) {
        mark_disconnected_locked("recv init header");
        return -1;
    }

    if (hip_remote_validate_header(&resp_header) != 0) {
        hip_remote_log_error("Invalid response header from worker");
        mark_disconnected_locked("bad header");
        return -1;
    }

    if (resp_header.payload_length > 0) {
        size_t plen = resp_header.payload_length;
        uint8_t* buf = (uint8_t*)malloc(plen);
        if (!buf) {
            mark_disconnected_locked("init alloc");
            return -1;
        }
        if (recv_all(g_client_state.socket_fd, buf, plen) != 0) {
            free(buf);
            mark_disconnected_locked("recv init body");
            return -1;
        }
        HipRemoteResponseHeader* resp = (HipRemoteResponseHeader*)buf;
        if (resp->error_code != hipSuccess) {
            hip_remote_log_error("Worker init failed: %d", resp->error_code);
            free(buf);
            mark_disconnected_locked("init error");
            return -1;
        }

        if (g_content_cache_enabled &&
            plen >= sizeof(HipRemoteInitResponse)) {
            HipRemoteInitResponse* iresp = (HipRemoteInitResponse*)buf;
            uint32_t count = iresp->cache_count;
            size_t hashes_offset = sizeof(HipRemoteInitResponse);
            size_t hashes_bytes = count * sizeof(uint64_t);
            if (hashes_offset + hashes_bytes <= plen) {
                content_cache_reset();
                uint64_t* hashes = (uint64_t*)(buf + hashes_offset);
                for (uint32_t i = 0; i < count; i++) {
                    hip_remote_content_cache_add(hashes[i]);
                }
                hip_remote_log_debug("Loaded %u cached hashes from worker (epoch=%u)",
                                     count, iresp->cache_epoch);
            }
        }
        free(buf);
    }

    if (g_use_shm && shm_name[0]) {
        /* Wait for the worker to open the SHM and set worker_alive */
        int wait_ms = 0;
        while (!g_shm.header->worker_alive && wait_ms < 5000) {
#ifdef _WIN32
            Sleep(1);
#else
            usleep(1000);
#endif
            wait_ms++;
        }
        if (g_shm.header->worker_alive) {
            g_shm_initialized = 1;
            hip_remote_log_debug("SHM transport active: %s (waited %dms)", shm_name, wait_ms);
        } else {
            hip_remote_log_error("SHM: worker did not open SHM within 5s, falling back to TCP");
            hip_shm_close(&g_shm);
            g_use_shm = 0;
        }
    }

    hip_remote_log_debug("Init handshake complete (transport=%s)",
                         g_shm_initialized ? "shm" : "tcp");

    g_client_state.last_error = hipSuccess;
    return 0;
}

/* ============================================================================
 * Public API
 * ============================================================================ */

void hip_remote_mark_ready(void) {
    g_dll_loading_complete = 1;
}

HipRemoteClientState* hip_remote_get_client_state(void) {
    hip_call_once(&g_init_once, init_from_environment);
    return &g_client_state;
}

int hip_remote_ensure_connected(void) {
    hip_call_once(&g_init_once, init_from_environment);

    hip_mutex_lock(&g_client_state.lock);
    int result = connect_to_worker_locked();
    hip_mutex_unlock(&g_client_state.lock);
    return result;
}

void hip_remote_disconnect(void) {
    prof_dump();

    hip_mutex_lock(&g_client_state.lock);

    if (g_client_state.connected) {
        g_shutdown_in_progress = 1;
        flush_write_buffer_locked();
        HipRemoteHeader header;
        hip_remote_init_header(&header, HIP_OP_SHUTDOWN,
                               g_client_state.next_request_id++, 0);
        (void)send_all(g_client_state.socket_fd, &header, sizeof(header));
        if (g_client_state.socket_fd != HIP_INVALID_SOCKET) {
#ifdef _WIN32
            struct linger lg = { 1, 2 };
            setsockopt(g_client_state.socket_fd, SOL_SOCKET, SO_LINGER,
                       (const char*)&lg, sizeof(lg));
            shutdown(g_client_state.socket_fd, SD_SEND);
#else
            shutdown(g_client_state.socket_fd, SHUT_WR);
#endif
            hip_close_socket(g_client_state.socket_fd);
        }
        g_client_state.socket_fd = HIP_INVALID_SOCKET;
        g_client_state.connected = false;
        g_permanently_disconnected = 1;

        if (g_shm_initialized) {
            hip_shm_close(&g_shm);
            g_shm_initialized = 0;
        }
    }

    hip_mutex_unlock(&g_client_state.lock);
}

bool hip_remote_is_connected(void) {
    hip_mutex_lock(&g_client_state.lock);
    bool connected = g_client_state.connected;
    hip_mutex_unlock(&g_client_state.lock);
    return connected;
}

void hip_remote_flush(void) {
    hip_mutex_lock(&g_client_state.lock);
    flush_write_buffer_locked();
    hip_mutex_unlock(&g_client_state.lock);
}

hipError_t hip_remote_request(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    void* response,
    size_t response_size
) {
    prof_record((uint16_t)op_code, 0);

    hip_mutex_lock(&g_client_state.lock);

    /* Ensure connected (also runs init_from_environment on first call) */
    if (connect_to_worker_locked() != 0) {
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    /* Flush buffered FnF requests before any synchronous round-trip */
    if (flush_write_buffer_locked() != 0) {
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    /* SHM sync path: pack header + payload into sync slot, get response */
    if (g_use_shm && g_shm_initialized) {
        HipRemoteHeader header;
        hip_remote_init_header(&header, op_code,
                               g_client_state.next_request_id++,
                               (uint32_t)request_size);

        /* Build sync request: header + payload */
        size_t total_req = sizeof(header) + (request ? request_size : 0);
        uint8_t sync_buf[4096];
        uint8_t* req_buf = (total_req <= sizeof(sync_buf))
            ? sync_buf : (uint8_t*)malloc(total_req);
        memcpy(req_buf, &header, sizeof(header));
        if (request && request_size > 0)
            memcpy(req_buf + sizeof(header), request, request_size);

        /* Send sync request and wait for response via SHM */
        uint8_t resp_buf[4096];
        size_t resp_total = sizeof(HipRemoteHeader) + response_size;
        uint8_t* rbuf = (resp_total <= sizeof(resp_buf))
            ? resp_buf : (uint8_t*)malloc(resp_total);

        int rsize = hip_shm_sync_request(&g_shm, req_buf, total_req,
                                         rbuf, resp_total);
        if (req_buf != sync_buf) free(req_buf);

        if (rsize < 0) {
            if (rbuf != resp_buf) free(rbuf);
            mark_disconnected_locked("shm sync");
            hip_mutex_unlock(&g_client_state.lock);
            return hipErrorNotInitialized;
        }

        /* Parse response: skip response header, copy payload */
        hipError_t result = hipSuccess;
        if (rsize >= (int)sizeof(HipRemoteHeader) && response && response_size > 0) {
            size_t payload_size = rsize - sizeof(HipRemoteHeader);
            size_t copy = payload_size < response_size ? payload_size : response_size;
            memcpy(response, rbuf + sizeof(HipRemoteHeader), copy);
            HipRemoteResponseHeader* resp = (HipRemoteResponseHeader*)response;
            result = (hipError_t)resp->error_code;
        }

        if (rbuf != resp_buf) free(rbuf);
        if (result != 500 && result != 600 && result != 801)
            g_client_state.last_error = result;
        hip_mutex_unlock(&g_client_state.lock);
        return result;
    }

    /* TCP sync path */
    HipRemoteHeader header;
    hip_remote_init_header(&header, op_code,
                           g_client_state.next_request_id++,
                           (uint32_t)request_size);

    hip_remote_log_debug("Sending %s (id=%u, payload=%zu)",
                         hip_remote_op_name(op_code),
                         header.request_id, request_size);

    hip_iovec_t iov[2] = {
        { &header, sizeof(header) },
        { request, request_size }
    };
    int nv = (request && request_size > 0) ? 2 : 1;
    if (send_all_v(g_client_state.socket_fd, iov, nv) != 0) {
        mark_disconnected_locked("send request");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    /* Receive response header */
    HipRemoteHeader resp_header;
    if (recv_all(g_client_state.socket_fd, &resp_header, sizeof(resp_header)) != 0) {
        mark_disconnected_locked("recv header");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (hip_remote_validate_header(&resp_header) != 0) {
        mark_disconnected_locked("invalid header");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorInvalidValue;
        return hipErrorInvalidValue;
    }

    /* Receive response payload */
    hipError_t result = hipSuccess;
    if (response && response_size > 0 && resp_header.payload_length > 0) {
        size_t to_read = resp_header.payload_length < response_size ?
                         resp_header.payload_length : response_size;
        if (recv_all(g_client_state.socket_fd, response, to_read) != 0) {
            mark_disconnected_locked("recv payload");
            hip_mutex_unlock(&g_client_state.lock);
            g_client_state.last_error = hipErrorNotInitialized;
            return hipErrorNotInitialized;
        }

        /* Drain extra bytes if response larger than buffer */
        if (resp_header.payload_length > response_size) {
            size_t extra = resp_header.payload_length - response_size;
            uint8_t drain[256];
            while (extra > 0) {
                size_t chunk = extra < sizeof(drain) ? extra : sizeof(drain);
                if (recv_all(g_client_state.socket_fd, drain, chunk) != 0) {
                    mark_disconnected_locked("drain");
                    hip_mutex_unlock(&g_client_state.lock);
                    g_client_state.last_error = hipErrorNotInitialized;
                    return hipErrorNotInitialized;
                }
                extra -= chunk;
            }
        }

        /* Extract error code from response */
        HipRemoteResponseHeader* resp = (HipRemoteResponseHeader*)response;
        result = (hipError_t)resp->error_code;
    }

    hip_remote_log_debug("Received response for %s: error=%d",
                         hip_remote_op_name(op_code), result);

    /* Only store genuine errors. These are expected/non-fatal:
     * - hipErrorNotFound (500): module iteration in hipBLASLt/Tensile
     * - hipErrorNotReady (600): event query for incomplete event
     * - hipErrorNotSupported (801): unsupported query attributes */
    if (result != 500 && result != 600 && result != 801) {
        g_client_state.last_error = result;
    }
    hip_mutex_unlock(&g_client_state.lock);
    return result;
}

/* ============================================================================
 * Fire-and-Forget Request (no response expected)
 *
 * Sends the request but does NOT wait for a response. The worker processes
 * the operation but skips sending a reply. Used for async GPU operations
 * (kernel launches, memset, etc.) to eliminate round-trip latency.
 * ============================================================================ */

hipError_t hip_remote_request_fire_and_forget(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size) {

    prof_record((uint16_t)op_code, 0);
    hip_mutex_lock(&g_client_state.lock);

    if (connect_to_worker_locked() != 0) {
        hip_mutex_unlock(&g_client_state.lock);
        return hipErrorNotInitialized;
    }

    HipRemoteHeader header;
    hip_remote_init_header(&header, op_code,
                           g_client_state.next_request_id++,
                           (uint64_t)request_size);
    header.flags |= HIP_REMOTE_FLAG_NO_REPLY;

    size_t total = sizeof(header) + (request ? request_size : 0);

    /* If it fits in the buffer, append; otherwise flush and send directly */
    if (total <= WRITE_BUFFER_SIZE - g_write_buffer_used) {
        memcpy(g_write_buffer + g_write_buffer_used, &header, sizeof(header));
        g_write_buffer_used += sizeof(header);
        if (request && request_size > 0) {
            memcpy(g_write_buffer + g_write_buffer_used, request, request_size);
            g_write_buffer_used += request_size;
        }
    } else {
        if (flush_write_buffer_locked() != 0) {
            hip_mutex_unlock(&g_client_state.lock);
            return hipErrorNotInitialized;
        }
        if (g_use_shm && g_shm_initialized) {
            if (hip_shm_fnf_write(&g_shm, &header, sizeof(header)) != 0 ||
                (request && request_size > 0 &&
                 hip_shm_fnf_write(&g_shm, request, request_size) != 0)) {
                mark_disconnected_locked("send (fnf shm)");
                hip_mutex_unlock(&g_client_state.lock);
                return hipErrorNotInitialized;
            }
        } else {
            hip_iovec_t iov[2] = {
                { &header, sizeof(header) },
                { request, request_size }
            };
            int nv = (request && request_size > 0) ? 2 : 1;
            if (send_all_v(g_client_state.socket_fd, iov, nv) != 0) {
                mark_disconnected_locked("send (fnf)");
                hip_mutex_unlock(&g_client_state.lock);
                return hipErrorNotInitialized;
            }
        }
        g_fnf_since_flush = 0;
    }

    if (g_eager_flush_interval > 0 && ++g_fnf_since_flush >= g_eager_flush_interval) {
        flush_write_buffer_locked();
        g_fnf_since_flush = 0;
    }

    hip_mutex_unlock(&g_client_state.lock);
    return hipSuccess;
}

hipError_t hip_remote_request_with_data_fire_and_forget(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    const void* data,
    size_t data_size) {

    prof_record((uint16_t)op_code, data_size);
    hip_mutex_lock(&g_client_state.lock);

    if (connect_to_worker_locked() != 0) {
        hip_mutex_unlock(&g_client_state.lock);
        return hipErrorNotInitialized;
    }

    HipRemoteHeader header;
    hip_remote_init_header(&header, op_code,
                           g_client_state.next_request_id++,
                           (uint64_t)(request_size + data_size));
    header.flags |= HIP_REMOTE_FLAG_NO_REPLY | HIP_REMOTE_FLAG_HAS_INLINE_DATA;

    size_t req_bytes = request ? request_size : 0;
    size_t dat_bytes = data ? data_size : 0;
    size_t total = sizeof(header) + req_bytes + dat_bytes;

    if (total <= WRITE_BUFFER_SIZE - g_write_buffer_used) {
        memcpy(g_write_buffer + g_write_buffer_used, &header, sizeof(header));
        g_write_buffer_used += sizeof(header);
        if (req_bytes > 0) {
            memcpy(g_write_buffer + g_write_buffer_used, request, req_bytes);
            g_write_buffer_used += req_bytes;
        }
        if (dat_bytes > 0) {
            memcpy(g_write_buffer + g_write_buffer_used, data, dat_bytes);
            g_write_buffer_used += dat_bytes;
        }
    } else {
        if (flush_write_buffer_locked() != 0) {
            hip_mutex_unlock(&g_client_state.lock);
            return hipErrorNotInitialized;
        }
        if (g_use_shm && g_shm_initialized) {
            if (hip_shm_fnf_write(&g_shm, &header, sizeof(header)) != 0 ||
                (request && request_size > 0 &&
                 hip_shm_fnf_write(&g_shm, request, request_size) != 0) ||
                (data && data_size > 0 &&
                 hip_shm_fnf_write(&g_shm, data, data_size) != 0)) {
                mark_disconnected_locked("send (fnf+data shm)");
                hip_mutex_unlock(&g_client_state.lock);
                return hipErrorNotInitialized;
            }
        } else {
            hip_iovec_t iov[3] = {
                { &header, sizeof(header) },
                { request, request_size },
                { data, data_size }
            };
            int nv = 1;
            if (request && request_size > 0) nv = 2;
            if (data && data_size > 0) nv = 3;
            if (send_all_v(g_client_state.socket_fd, iov, nv) != 0) {
                mark_disconnected_locked("send (fnf+data)");
                hip_mutex_unlock(&g_client_state.lock);
                return hipErrorNotInitialized;
            }
        }
        g_fnf_since_flush = 0;
    }

    if (g_eager_flush_interval > 0 && ++g_fnf_since_flush >= g_eager_flush_interval) {
        flush_write_buffer_locked();
        g_fnf_since_flush = 0;
    }

    hip_mutex_unlock(&g_client_state.lock);
    return hipSuccess;
}

hipError_t hip_remote_request_with_data(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    const void* data,
    size_t data_size,
    void* response,
    size_t response_size
) {
    hip_call_once(&g_init_once, init_from_environment);
    prof_record((uint16_t)op_code, data_size);

    hip_mutex_lock(&g_client_state.lock);

    if (connect_to_worker_locked() != 0) {
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (flush_write_buffer_locked() != 0) {
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    /* Build request with inline data */
    HipRemoteHeader header;
    hip_remote_init_header(&header, op_code,
                           g_client_state.next_request_id++,
                           (uint32_t)(request_size + data_size));
    header.flags |= HIP_REMOTE_FLAG_HAS_INLINE_DATA;

    hip_remote_log_debug("Sending %s with data (id=%u, payload=%zu, data=%zu)",
                         hip_remote_op_name(op_code),
                         header.request_id, request_size, data_size);

    /* SHM sync-with-data path */
    if (g_use_shm && g_shm_initialized) {
        size_t total_req = sizeof(header) + request_size + data_size;
        uint8_t* req_buf = (uint8_t*)malloc(total_req);
        if (!req_buf) {
            hip_mutex_unlock(&g_client_state.lock);
            return hipErrorOutOfMemory;
        }
        size_t off = 0;
        memcpy(req_buf + off, &header, sizeof(header)); off += sizeof(header);
        if (request && request_size > 0) {
            memcpy(req_buf + off, request, request_size); off += request_size;
        }
        if (data && data_size > 0) {
            memcpy(req_buf + off, data, data_size); off += data_size;
        }

        uint8_t resp_buf[4096];
        size_t resp_total = sizeof(HipRemoteHeader) + response_size;
        uint8_t* rbuf = (resp_total <= sizeof(resp_buf))
            ? resp_buf : (uint8_t*)malloc(resp_total);

        int rsize = hip_shm_sync_request(&g_shm, req_buf, total_req,
                                         rbuf, resp_total);
        free(req_buf);

        hipError_t result = hipSuccess;
        if (rsize >= (int)sizeof(HipRemoteHeader) && response && response_size > 0) {
            size_t payload_size = rsize - sizeof(HipRemoteHeader);
            size_t copy = payload_size < response_size ? payload_size : response_size;
            memcpy(response, rbuf + sizeof(HipRemoteHeader), copy);
            HipRemoteResponseHeader* resp = (HipRemoteResponseHeader*)response;
            result = (hipError_t)resp->error_code;
        } else if (rsize < 0) {
            result = hipErrorNotInitialized;
        }

        if (rbuf != resp_buf) free(rbuf);
        if (result != 500 && result != 600 && result != 801)
            g_client_state.last_error = result;
        hip_mutex_unlock(&g_client_state.lock);
        return result;
    }

    /* TCP path */
    hip_iovec_t iov[3] = {
        { &header, sizeof(header) },
        { request, request_size },
        { data, data_size }
    };
    int nv = 1;
    if (request && request_size > 0) nv = 2;
    if (data && data_size > 0) nv = 3;
    if (send_all_v(g_client_state.socket_fd, iov, nv) != 0) {
        mark_disconnected_locked("send request+data");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    /* Receive response */
    HipRemoteHeader resp_header;
    if (recv_all(g_client_state.socket_fd, &resp_header, sizeof(resp_header)) != 0) {
        mark_disconnected_locked("recv header");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (hip_remote_validate_header(&resp_header) != 0) {
        mark_disconnected_locked("invalid header");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorInvalidValue;
        return hipErrorInvalidValue;
    }

    hipError_t result = hipSuccess;
    if (response && response_size > 0 && resp_header.payload_length > 0) {
        size_t to_read = resp_header.payload_length < response_size ?
                         resp_header.payload_length : response_size;
        if (recv_all(g_client_state.socket_fd, response, to_read) != 0) {
            mark_disconnected_locked("recv payload");
            hip_mutex_unlock(&g_client_state.lock);
            g_client_state.last_error = hipErrorNotInitialized;
            return hipErrorNotInitialized;
        }

        HipRemoteResponseHeader* resp = (HipRemoteResponseHeader*)response;
        result = (hipError_t)resp->error_code;
    }

    if (result != 500 && result != 600 && result != 801) {
        g_client_state.last_error = result;
    }
    hip_mutex_unlock(&g_client_state.lock);
    return result;
}

hipError_t hip_remote_request_receive_data(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    void* response,
    size_t response_size,
    void* data_out,
    size_t data_size
) {
    hip_call_once(&g_init_once, init_from_environment);
    prof_record((uint16_t)op_code, data_size);

    hip_mutex_lock(&g_client_state.lock);

    if (connect_to_worker_locked() != 0) {
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (flush_write_buffer_locked() != 0) {
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    /* Send request */
    HipRemoteHeader header;
    hip_remote_init_header(&header, op_code,
                           g_client_state.next_request_id++,
                           (uint32_t)request_size);

    hip_remote_log_debug("Sending %s expecting data (id=%u, payload=%zu, expect_data=%zu)",
                         hip_remote_op_name(op_code),
                         header.request_id, request_size, data_size);

    hip_iovec_t iov[2] = {
        { &header, sizeof(header) },
        { request, request_size }
    };
    int nv = (request && request_size > 0) ? 2 : 1;
    if (send_all_v(g_client_state.socket_fd, iov, nv) != 0) {
        mark_disconnected_locked("send request");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    /* Receive response header */
    HipRemoteHeader resp_header;
    if (recv_all(g_client_state.socket_fd, &resp_header, sizeof(resp_header)) != 0) {
        mark_disconnected_locked("recv header");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (hip_remote_validate_header(&resp_header) != 0) {
        mark_disconnected_locked("invalid header");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorInvalidValue;
        return hipErrorInvalidValue;
    }

    /* Receive response struct */
    hipError_t result = hipSuccess;
    if (response && response_size > 0) {
        if (recv_all(g_client_state.socket_fd, response, response_size) != 0) {
            mark_disconnected_locked("recv response");
            hip_mutex_unlock(&g_client_state.lock);
            g_client_state.last_error = hipErrorNotInitialized;
            return hipErrorNotInitialized;
        }

        HipRemoteResponseHeader* resp = (HipRemoteResponseHeader*)response;
        result = (hipError_t)resp->error_code;
    }

    /* Receive inline data */
    if (result == hipSuccess && data_out && data_size > 0) {
        size_t remaining = resp_header.payload_length - response_size;
        size_t to_read = remaining < data_size ? remaining : data_size;
        if (to_read > 0) {
            if (recv_all(g_client_state.socket_fd, data_out, to_read) != 0) {
                mark_disconnected_locked("recv data");
                hip_mutex_unlock(&g_client_state.lock);
                g_client_state.last_error = hipErrorNotInitialized;
                return hipErrorNotInitialized;
            }
        }
    }

    if (result != 500 && result != 600 && result != 801) {
        g_client_state.last_error = result;
    }
    hip_mutex_unlock(&g_client_state.lock);
    return result;
}

/* ============================================================================
 * Library Constructor/Destructor
 * ============================================================================ */

#ifdef _WIN32

static void hip_remote_atexit_cleanup(void) {
    hip_remote_disconnect();
}

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpReserved) {
    (void)hinstDLL; (void)lpReserved;
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
            atexit(hip_remote_atexit_cleanup);
            break;
        case DLL_PROCESS_DETACH:
            if (g_wsa_initialized) {
                hip_socket_cleanup();
                g_wsa_initialized = 0;
            }
            break;
    }
    return TRUE;
}

#else

__attribute__((constructor))
static void hip_remote_client_init(void) {
    init_from_environment();
    hip_remote_log_debug("Remote HIP client library loaded");
}

__attribute__((destructor))
static void hip_remote_client_cleanup(void) {
    hip_remote_disconnect();
    hip_remote_log_debug("Remote HIP client library unloaded");
}

#endif
