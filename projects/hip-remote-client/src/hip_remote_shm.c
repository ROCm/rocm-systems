/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "hip_remote/hip_remote_shm.h"
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#define MEMORY_BARRIER()  MemoryBarrier()
#define ATOMIC_LOAD(p)    (*(volatile uint64_t*)(p))
#define ATOMIC_STORE(p,v) do { *(volatile uint64_t*)(p) = (v); MEMORY_BARRIER(); } while(0)
#define ATOMIC_LOAD32(p)  (*(volatile uint32_t*)(p))
#define ATOMIC_STORE32(p,v) do { *(volatile uint32_t*)(p) = (v); MEMORY_BARRIER(); } while(0)
#define YIELD()           SwitchToThread()
#else
#define MEMORY_BARRIER()  __atomic_thread_fence(__ATOMIC_SEQ_CST)
#define ATOMIC_LOAD(p)    __atomic_load_n((volatile uint64_t*)(p), __ATOMIC_ACQUIRE)
#define ATOMIC_STORE(p,v) __atomic_store_n((volatile uint64_t*)(p), (v), __ATOMIC_RELEASE)
#define ATOMIC_LOAD32(p)  __atomic_load_n((volatile uint32_t*)(p), __ATOMIC_ACQUIRE)
#define ATOMIC_STORE32(p,v) __atomic_store_n((volatile uint32_t*)(p), (v), __ATOMIC_RELEASE)
#define YIELD()           sched_yield()
#include <sched.h>
#include <errno.h>
#endif

static void shm_setup_pointers(HipShmHandle* shm) {
    uint8_t* base = (uint8_t*)shm->base;
    shm->header   = (HipShmHeader*)base;
    shm->fnf_ring = base + HIP_SHM_FNF_OFFSET;
    shm->sync_slot = base + HIP_SHM_SYNC_OFFSET;
    shm->resp_slot = base + HIP_SHM_RESP_OFFSET;
}

int hip_shm_create(HipShmHandle* shm, const char* name) {
    memset(shm, 0, sizeof(*shm));
    strncpy(shm->name, name, HIP_SHM_NAME_MAX - 1);
    shm->total_size = HIP_SHM_TOTAL_SIZE;
    shm->is_creator = 1;

#ifdef _WIN32
    shm->hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
        (DWORD)(shm->total_size >> 32),
        (DWORD)(shm->total_size & 0xFFFFFFFF),
        name);
    if (!shm->hMapFile) {
        fprintf(stderr, "[HIP-SHM] CreateFileMapping failed: %lu\n", GetLastError());
        return -1;
    }
    shm->base = MapViewOfFile(shm->hMapFile, FILE_MAP_ALL_ACCESS,
                              0, 0, shm->total_size);
    if (!shm->base) {
        fprintf(stderr, "[HIP-SHM] MapViewOfFile failed: %lu\n", GetLastError());
        CloseHandle(shm->hMapFile);
        return -1;
    }
#else
    shm->shm_fd = shm_open(name, O_CREAT | O_RDWR, 0600);
    if (shm->shm_fd < 0) {
        fprintf(stderr, "[HIP-SHM] shm_open failed: %s\n", strerror(errno));
        return -1;
    }
    if (ftruncate(shm->shm_fd, (off_t)shm->total_size) != 0) {
        fprintf(stderr, "[HIP-SHM] ftruncate failed: %s\n", strerror(errno));
        close(shm->shm_fd);
        shm_unlink(name);
        return -1;
    }
    shm->base = mmap(NULL, shm->total_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, shm->shm_fd, 0);
    if (shm->base == MAP_FAILED) {
        fprintf(stderr, "[HIP-SHM] mmap failed: %s\n", strerror(errno));
        close(shm->shm_fd);
        shm_unlink(name);
        shm->base = NULL;
        return -1;
    }
#endif

    memset(shm->base, 0, shm->total_size);
    shm_setup_pointers(shm);

    shm->header->magic = HIP_SHM_MAGIC;
    shm->header->version = HIP_SHM_VERSION;
    shm->header->fnf_ring_size = HIP_SHM_FNF_SIZE;
    shm->header->sync_slot_size = HIP_SHM_SYNC_SIZE;
    shm->header->resp_slot_size = HIP_SHM_RESP_SIZE;
    ATOMIC_STORE32(&shm->header->client_alive, 1);
    MEMORY_BARRIER();

    return 0;
}

int hip_shm_open(HipShmHandle* shm, const char* name) {
    memset(shm, 0, sizeof(*shm));
    strncpy(shm->name, name, HIP_SHM_NAME_MAX - 1);
    shm->total_size = HIP_SHM_TOTAL_SIZE;
    shm->is_creator = 0;

#ifdef _WIN32
    shm->hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name);
    if (!shm->hMapFile) {
        fprintf(stderr, "[HIP-SHM] OpenFileMapping failed: %lu\n", GetLastError());
        return -1;
    }
    shm->base = MapViewOfFile(shm->hMapFile, FILE_MAP_ALL_ACCESS,
                              0, 0, shm->total_size);
    if (!shm->base) {
        fprintf(stderr, "[HIP-SHM] MapViewOfFile failed: %lu\n", GetLastError());
        CloseHandle(shm->hMapFile);
        return -1;
    }
#else
    shm->shm_fd = shm_open(name, O_RDWR, 0600);
    if (shm->shm_fd < 0) {
        fprintf(stderr, "[HIP-SHM] shm_open (open) failed: %s\n", strerror(errno));
        return -1;
    }
    shm->base = mmap(NULL, shm->total_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, shm->shm_fd, 0);
    if (shm->base == MAP_FAILED) {
        fprintf(stderr, "[HIP-SHM] mmap failed: %s\n", strerror(errno));
        close(shm->shm_fd);
        shm->base = NULL;
        return -1;
    }
#endif

    shm_setup_pointers(shm);

    if (shm->header->magic != HIP_SHM_MAGIC) {
        fprintf(stderr, "[HIP-SHM] bad magic: 0x%08x\n", shm->header->magic);
        hip_shm_close(shm);
        return -1;
    }

    ATOMIC_STORE32(&shm->header->worker_alive, 1);
    return 0;
}

/* FnF ring: SPSC lock-free ring buffer.
 * write_pos and read_pos are monotonically increasing byte counters.
 * Actual ring index = pos % ring_size. */

size_t hip_shm_fnf_available(HipShmHandle* shm) {
    uint64_t wp = ATOMIC_LOAD(&shm->header->fnf_write_pos);
    uint64_t rp = ATOMIC_LOAD(&shm->header->fnf_read_pos);
    uint64_t used = wp - rp;
    return (size_t)(shm->header->fnf_ring_size - used);
}

size_t hip_shm_fnf_readable(HipShmHandle* shm) {
    uint64_t wp = ATOMIC_LOAD(&shm->header->fnf_write_pos);
    uint64_t rp = ATOMIC_LOAD(&shm->header->fnf_read_pos);
    return (size_t)(wp - rp);
}

int hip_shm_fnf_write(HipShmHandle* shm, const void* data, size_t len) {
    if (len == 0) return 0;
    uint32_t ring_size = shm->header->fnf_ring_size;
    const uint8_t* src = (const uint8_t*)data;
    size_t written = 0;

    while (written < len) {
        uint64_t wp = ATOMIC_LOAD(&shm->header->fnf_write_pos);
        uint64_t rp = ATOMIC_LOAD(&shm->header->fnf_read_pos);
        uint64_t used = wp - rp;
        size_t avail = ring_size - (size_t)used;

        if (avail == 0) {
            if (!ATOMIC_LOAD32(&shm->header->worker_alive)) return -1;
            YIELD();
            continue;
        }

        size_t chunk = len - written;
        if (chunk > avail) chunk = avail;

        uint32_t idx = (uint32_t)(wp % ring_size);
        uint32_t first = ring_size - idx;
        if (first > chunk) first = (uint32_t)chunk;

        memcpy(shm->fnf_ring + idx, src + written, first);
        if (chunk > first) {
            memcpy(shm->fnf_ring, src + written + first, chunk - first);
        }

        ATOMIC_STORE(&shm->header->fnf_write_pos, wp + chunk);
        written += chunk;
    }
    return 0;
}

int hip_shm_fnf_read(HipShmHandle* shm, void* buf, size_t len) {
    if (len == 0) return 0;
    uint32_t ring_size = shm->header->fnf_ring_size;
    uint8_t* dst = (uint8_t*)buf;
    size_t read_total = 0;

    while (read_total < len) {
        uint64_t wp = ATOMIC_LOAD(&shm->header->fnf_write_pos);
        uint64_t rp = ATOMIC_LOAD(&shm->header->fnf_read_pos);
        size_t avail = (size_t)(wp - rp);

        if (avail == 0) {
            if (!ATOMIC_LOAD32(&shm->header->client_alive)) return -1;
            YIELD();
            continue;
        }

        size_t chunk = len - read_total;
        if (chunk > avail) chunk = avail;

        uint32_t idx = (uint32_t)(rp % ring_size);
        uint32_t first = ring_size - idx;
        if (first > chunk) first = (uint32_t)chunk;

        memcpy(dst + read_total, shm->fnf_ring + idx, first);
        if (chunk > first) {
            memcpy(dst + read_total + first, shm->fnf_ring, chunk - first);
        }

        ATOMIC_STORE(&shm->header->fnf_read_pos, rp + chunk);
        read_total += chunk;
    }
    return 0;
}

/* Sync slot: simple request/response with busy-wait signaling. */

int hip_shm_sync_request(HipShmHandle* shm,
                         const void* req, size_t req_size,
                         void* resp, size_t resp_size) {
    if (req_size > shm->header->sync_slot_size) return -1;

    memcpy(shm->sync_slot, req, req_size);
    ATOMIC_STORE32(&shm->header->sync_req_size, (uint32_t)req_size);
    ATOMIC_STORE32(&shm->header->sync_req_ready, 1);

    while (!ATOMIC_LOAD32(&shm->header->sync_resp_ready)) {
        if (!ATOMIC_LOAD32(&shm->header->worker_alive)) return -1;
        YIELD();
    }

    uint32_t rsize = ATOMIC_LOAD32(&shm->header->sync_resp_size);
    size_t copy = rsize < resp_size ? rsize : resp_size;
    memcpy(resp, shm->resp_slot, copy);

    ATOMIC_STORE32(&shm->header->sync_resp_ready, 0);
    return (int)rsize;
}

int hip_shm_sync_recv(HipShmHandle* shm, void* buf, size_t buf_size) {
    while (!ATOMIC_LOAD32(&shm->header->sync_req_ready)) {
        if (!ATOMIC_LOAD32(&shm->header->client_alive)) return -1;
        /* Don't busy-wait on sync -- check FnF ring first (caller does this) */
        return 0;
    }

    uint32_t rsize = ATOMIC_LOAD32(&shm->header->sync_req_size);
    size_t copy = rsize < buf_size ? rsize : buf_size;
    memcpy(buf, shm->sync_slot, copy);

    ATOMIC_STORE32(&shm->header->sync_req_ready, 0);
    return (int)rsize;
}

int hip_shm_sync_respond(HipShmHandle* shm, const void* data, size_t len) {
    if (len > shm->header->resp_slot_size) return -1;

    memcpy(shm->resp_slot, data, len);
    ATOMIC_STORE32(&shm->header->sync_resp_size, (uint32_t)len);
    ATOMIC_STORE32(&shm->header->sync_resp_ready, 1);
    return 0;
}

void hip_shm_close(HipShmHandle* shm) {
    if (!shm->base) return;

    if (shm->is_creator) {
        ATOMIC_STORE32(&shm->header->client_alive, 0);
    } else {
        ATOMIC_STORE32(&shm->header->worker_alive, 0);
    }

#ifdef _WIN32
    UnmapViewOfFile(shm->base);
    CloseHandle(shm->hMapFile);
#else
    munmap(shm->base, shm->total_size);
    close(shm->shm_fd);
    if (shm->is_creator) {
        shm_unlink(shm->name);
    }
#endif
    shm->base = NULL;
}
