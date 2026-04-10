/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

/**
 * @file hip_remote_shm.h
 * @brief Cross-platform shared memory ring buffer for IPC transport.
 *
 * SPSC (single-producer, single-consumer) lock-free ring buffer using
 * shared memory. Used as an alternative to TCP for same-machine
 * client-worker communication.
 *
 * Layout of the shared memory region:
 *   [ShmHeader][FnF ring data (4MB)][Sync request slot][Sync response slot]
 */

#ifndef HIP_REMOTE_SHM_H
#define HIP_REMOTE_SHM_H

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define HIP_SHM_MAGIC       0x48495053  /* "HIPS" */
#define HIP_SHM_VERSION     1
#define HIP_SHM_FNF_SIZE    (4 * 1024 * 1024)  /* 4MB FnF ring */
#define HIP_SHM_SYNC_SIZE   (16 * 1024 * 1024)  /* 16MB sync request slot (module loads) */
#define HIP_SHM_RESP_SIZE   (256 * 1024)         /* 256KB sync response slot */
#define HIP_SHM_NAME_MAX    128

/**
 * Shared memory region header (at offset 0).
 * All fields use relaxed or acquire/release atomic access.
 */
typedef struct {
    volatile uint32_t magic;
    volatile uint32_t version;

    /* FnF ring: client writes, worker reads */
    volatile uint64_t fnf_write_pos;   /* bytes written (mod FNF_SIZE) */
    volatile uint64_t fnf_read_pos;    /* bytes read (mod FNF_SIZE) */

    /* Sync request: client writes a complete message, sets sync_ready=1 */
    volatile uint32_t sync_req_size;   /* size of sync request data */
    volatile uint32_t sync_req_ready;  /* 1 = request available */

    /* Sync response: worker writes, sets sync_resp_ready=1 */
    volatile uint32_t sync_resp_size;  /* size of sync response data */
    volatile uint32_t sync_resp_ready; /* 1 = response available */

    volatile uint32_t client_alive;    /* client sets to 1, worker checks */
    volatile uint32_t worker_alive;    /* worker sets to 1, client checks */

    uint32_t fnf_ring_size;
    uint32_t sync_slot_size;
    uint32_t resp_slot_size;
    uint32_t _pad;
} HipShmHeader;

/* Offsets into the shared memory region */
#define HIP_SHM_HEADER_SIZE   256  /* padded for cache alignment */
#define HIP_SHM_FNF_OFFSET    HIP_SHM_HEADER_SIZE
#define HIP_SHM_SYNC_OFFSET   (HIP_SHM_FNF_OFFSET + HIP_SHM_FNF_SIZE)
#define HIP_SHM_RESP_OFFSET   (HIP_SHM_SYNC_OFFSET + HIP_SHM_SYNC_SIZE)
#define HIP_SHM_TOTAL_SIZE    (HIP_SHM_RESP_OFFSET + HIP_SHM_RESP_SIZE)

/**
 * Shared memory handle (per-process).
 */
typedef struct {
    void*    base;         /* mapped base pointer */
    size_t   total_size;
    char     name[HIP_SHM_NAME_MAX];
    int      is_creator;   /* 1 if this process created the region */
#ifdef _WIN32
    HANDLE   hMapFile;
#else
    int      shm_fd;
#endif

    /* Convenience pointers into the mapped region */
    HipShmHeader* header;
    uint8_t*      fnf_ring;    /* FnF ring data */
    uint8_t*      sync_slot;   /* sync request data */
    uint8_t*      resp_slot;   /* sync response data */
} HipShmHandle;

/**
 * Create a new shared memory region. Returns 0 on success.
 * The name should be unique per session (e.g., "hip_shm_<pid>").
 */
int hip_shm_create(HipShmHandle* shm, const char* name);

/**
 * Open an existing shared memory region by name. Returns 0 on success.
 */
int hip_shm_open(HipShmHandle* shm, const char* name);

/**
 * Write data to the FnF ring buffer. Spins if the ring is full.
 * Returns 0 on success, -1 if peer is dead.
 */
int hip_shm_fnf_write(HipShmHandle* shm, const void* data, size_t len);

/**
 * Read data from the FnF ring buffer. Spins if empty.
 * Reads exactly `len` bytes into `buf`.
 * Returns 0 on success, -1 if peer is dead.
 */
int hip_shm_fnf_read(HipShmHandle* shm, void* buf, size_t len);

/**
 * Write a sync request and wait for the response.
 * `req` is written to the sync slot, then blocks until response is ready.
 * Response is copied to `resp` (up to `resp_size` bytes).
 * Returns actual response size, or -1 on error.
 */
int hip_shm_sync_request(HipShmHandle* shm,
                         const void* req, size_t req_size,
                         void* resp, size_t resp_size);

/**
 * Read a sync request from the sync slot (worker side).
 * Blocks until a request is available.
 * Returns request size, or -1 if peer is dead.
 */
int hip_shm_sync_recv(HipShmHandle* shm, void* buf, size_t buf_size);

/**
 * Write a sync response (worker side).
 */
int hip_shm_sync_respond(HipShmHandle* shm, const void* data, size_t len);

/**
 * Close and optionally unlink the shared memory region.
 */
void hip_shm_close(HipShmHandle* shm);

/**
 * Check how much space is available in the FnF ring for writing.
 */
size_t hip_shm_fnf_available(HipShmHandle* shm);

/**
 * Check how many bytes are available to read from the FnF ring.
 */
size_t hip_shm_fnf_readable(HipShmHandle* shm);

#ifdef __cplusplus
}
#endif

#endif /* HIP_REMOTE_SHM_H */
