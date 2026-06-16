/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* various-mem-rw - Round-trip a file through hipFile using one of three
 * memory types for the transfer buffer: plain device memory, managed
 * memory, or pinned host memory.
 *
 * Usage: ./various-mem-rw INPUT OUTPUT MODE [GPUID]
 *
 *   INPUT    Existing file to read. Up to VMR_CAP (default 1 MiB) is
 *            consumed. Create with:
 *              dd if=/dev/urandom of=input.bin bs=1M count=1
 *   OUTPUT   Path to the output file. Created/truncated. Receives a copy of
 *            the bytes read from INPUT.
 *   MODE     Memory type used for the transfer buffer:
 *              1 = device memory      (hipMalloc)
 *              2 = managed memory     (hipMallocManaged - no BufRegister needed)
 *              3 = pinned host memory (hipHostMalloc)
 *   GPUID    GPU device index (optional, default 0).
 *
 * Steps:
 *   1. Select GPU
 *   2. Open + register input file
 *   3. Allocate buffer of the chosen memory type + zero it
 *   4. hipFileRead into buffer
 *   5. Open + register output file + hipFileWrite + ftruncate
 *   6. Hash verify
 */

#include "examples_common.h"

#include <hipfile.h>
#include <hip/hip_runtime_api.h>

#include <cerrno>
#include <fcntl.h>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/// @brief Hard cap on bytes consumed from INPUT. Override at compile time.
#ifndef VMR_CAP
#define VMR_CAP (1024UL * 1024UL)
#endif

typedef enum {
    MEM_DEVICE  = 1,
    MEM_MANAGED = 2,
    MEM_PINNED  = 3,
} mem_mode_t;

/// @brief Human-readable label for a memory mode.
static const char *
mode_name(mem_mode_t mode)
{
    switch (mode) {
        case MEM_DEVICE:
            return "device";
        case MEM_MANAGED:
            return "managed";
        case MEM_PINNED:
            return "pinned-host";
        default:
            return "unknown";
    }
}

/// @brief Allocate `size` bytes in the memory backing chosen by `mode`.
static int
alloc_buf(mem_mode_t mode, size_t size, void **out_buf)
{
    hipError_t hip_err;
    switch (mode) {
        case MEM_DEVICE:
            hip_err = hipMalloc(out_buf, size);
            break;
        case MEM_MANAGED:
            hip_err = hipMallocManaged(out_buf, size, hipMemAttachGlobal);
            break;
        case MEM_PINNED:
            hip_err = hipHostMalloc(out_buf, size, hipHostMallocDefault);
            break;
        default:
            fprintf(stderr, "alloc_buf: invalid mode %d\n", (int)mode);
            return 1;
    }
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not allocate %zu bytes (%s, hip err %d)\n", size, mode_name(mode), hip_err);
        return 1;
    }
    return 0;
}

/// @brief Free a buffer previously returned by alloc_buf.
static int
free_buf(mem_mode_t mode, void *buf)
{
    hipError_t hip_err;
    switch (mode) {
        case MEM_DEVICE:
        case MEM_MANAGED:
            hip_err = hipFree(buf);
            break;
        case MEM_PINNED:
            hip_err = hipHostFree(buf);
            break;
        default:
            fprintf(stderr, "free_buf: invalid mode %d\n", (int)mode);
            return 1;
    }
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not free buffer (%s, hip err %d)\n", mode_name(mode), hip_err);
        return 1;
    }
    return 0;
}

/// @brief Zero a buffer previously returned by alloc_buf.
static int
zero_buf(mem_mode_t mode, void *buf, size_t size)
{
    if (mode == MEM_PINNED) {
        memset(buf, 0, size);
        return 0;
    }
    hipError_t hip_err = hipMemset(buf, 0, size);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not zero buffer (%s, hip err %d)\n", mode_name(mode), hip_err);
        return 1;
    }
    /* hipMemset is async w.r.t. the host; block until it completes (testing). */
    hip_err = hipDeviceSynchronize();
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not synchronize after memset (%s, hip err %d)\n", mode_name(mode), hip_err);
        return 1;
    }
    return 0;
}

int
main(int argc, char *argv[])
{
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "Usage: %s INPUT OUTPUT MODE [GPUID]\n", argv[0]);
        fprintf(stderr, "  MODE: 1=device, 2=managed, 3=pinned-host\n");
        return EXIT_FAILURE;
    }

    const char *in_path  = argv[1];
    const char *out_path = argv[2];
    const int   mode_raw = atoi(argv[3]);
    const int   gpu_id   = (argc == 5) ? atoi(argv[4]) : 0;

    if (mode_raw < 1 || mode_raw > 3) {
        fprintf(stderr, "MODE must be 1 (device), 2 (managed), or 3 (pinned-host)\n");
        return EXIT_FAILURE;
    }
    const mem_mode_t mode = (mem_mode_t)mode_raw;

    /* Stat the input file to get its size and the filesystem block size. */
    size_t file_size, block_size;
    {
        struct stat statbuf;
        if (stat(in_path, &statbuf)) {
            fprintf(stderr, "Could not stat %s (%s)\n", in_path, strerror(errno));
            return EXIT_FAILURE;
        }
        file_size  = (size_t)statbuf.st_size;
        block_size = (size_t)statbuf.st_blksize;
        if (!is_power_of_two(block_size)) {
            fprintf(stderr, "Block size is not a power of two (%zu)\n", block_size);
            return EXIT_FAILURE;
        }
    }

    if (0 == file_size) {
        fprintf(stderr, "Input file %s is empty\n", in_path);
        return EXIT_FAILURE;
    }

    const size_t payload_size = (file_size < VMR_CAP) ? file_size : VMR_CAP;
    const size_t alloc_size   = align_up(payload_size, block_size);

    int             in_fd  = -1;
    int             out_fd = -1;
    hipFileHandle_t in_handle, out_handle;
    void           *buf         = NULL;
    int             exit_status = EXIT_FAILURE;
    hipError_t      hip_err;
    ssize_t         nbytes;

    /* 1. Select GPU */
    hip_err = hipSetDevice(gpu_id);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not select GPU %d (%d)\n", gpu_id, hip_err);
        return EXIT_FAILURE;
    }

    /* 2. Open + register input file */
    if (open_file(in_path, O_RDONLY | O_DIRECT, 0, &in_fd, &in_handle))
        return EXIT_FAILURE;

    /* 3. Allocate buffer of the chosen memory type + zero it */
    if (alloc_buf(mode, alloc_size, &buf))
        goto close_in;

    if (zero_buf(mode, buf, alloc_size))
        goto release_buf;

    /* 4. hipFileRead into buffer (single call) */
    nbytes = hipFileRead(in_handle, buf, alloc_size, /*file_offset=*/0, /*buf_offset=*/0);
    if (nbytes < 0) {
        fprintf(stderr, "Could not read from %s (%zd) (%s)\n", in_path, nbytes,
                IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
        goto release_buf;
    }
    if ((size_t)nbytes < payload_size) {
        fprintf(stderr, "Short read on %s: got %zd bytes, expected at least %zu\n", in_path, nbytes,
                payload_size);
        goto release_buf;
    }

    /* 5. Open + register output file + hipFileWrite + ftruncate */
    if (open_file(out_path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH,
                  &out_fd, &out_handle)) {
        goto release_buf;
    }

    nbytes = hipFileWrite(out_handle, buf, alloc_size, /*file_offset=*/0, /*buf_offset=*/0);
    if (nbytes < 0) {
        fprintf(stderr, "Could not write to %s (%zd) (%s)\n", out_path, nbytes,
                IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
        goto close_out;
    }

    if (-1 == ftruncate(out_fd, (off_t)payload_size)) {
        fprintf(stderr, "Could not truncate %s (%s)\n", out_path, strerror(errno));
        goto close_out;
    }

    if (close_file(out_path, out_fd, out_handle)) {
        out_fd = -1;
        goto release_buf;
    }
    out_fd = -1;

    /* 6. Hash verify */
    {
        uint64_t hash_in, hash_out;

        if (hash_file_range(in_path, 0, payload_size, &hash_in))
            goto release_buf;
        if (hash_file_range(out_path, 0, payload_size, &hash_out))
            goto release_buf;

        if (hash_in != hash_out) {
            fprintf(stderr, "Hash mismatch (%s mode): %s=0x%016" PRIx64 "  %s=0x%016" PRIx64 "\n",
                    mode_name(mode), in_path, hash_in, out_path, hash_out);
            goto release_buf;
        }

        printf("OK  [%s] %s -> %s  (%zu bytes, hash 0x%016" PRIx64 ")\n", mode_name(mode), in_path, out_path,
               payload_size, hash_in);
    }

    exit_status = EXIT_SUCCESS;

close_out:
    if (out_fd != -1) {
        if (close_file(out_path, out_fd, out_handle))
            exit_status = EXIT_FAILURE;
    }

release_buf:
    if (free_buf(mode, buf))
        exit_status = EXIT_FAILURE;

close_in:
    if (close_file(in_path, in_fd, in_handle))
        exit_status = EXIT_FAILURE;

    return exit_status;
}
