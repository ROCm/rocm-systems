/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* no-bufregister-write - Write a GPU buffer to a file via hipFile without
 * explicitly registering it. hipFile copies through its internal pool buffer.
 *
 * Usage: ./no-bufregister-write OUTPUT [GPUID]
 *
 *   OUTPUT   Path to the output file. Created/truncated. Receives NBW_SIZE
 *            (default 1 MiB) bytes of generated test pattern. hipFile copies
 *            through its internal pool buffer because the GPU buffer is not
 *            explicitly registered.
 *   GPUID    GPU device index (optional, default 0).
 *
 * Steps:
 *   1. Select GPU
 *   2. Build CPU pattern + copy to GPU
 *   3. open file with O_DIRECT + hipFileHandleRegister (no buf registration)
 *   4. hipFileWrite via hipFile's internal pool
 *   5. ftruncate to exact size
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

/// @brief Size of the test payload in bytes. Override at compile time.
#ifndef NBW_SIZE
#define NBW_SIZE (1024UL * 1024UL)
#endif

int
main(int argc, char *argv[])
{
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s OUTPUT [GPUID]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char  *out_path     = argv[1];
    const int    gpu_id       = (argc == 3) ? atoi(argv[2]) : 0;
    const size_t payload_size = NBW_SIZE;
    const size_t alloc_size   = align_up(payload_size, BLOCK_ALIGN);

    int             out_fd = -1;
    hipFileHandle_t out_handle;
    void           *devbuf      = NULL;
    uint8_t        *cpu_pattern = NULL;
    int             exit_status = EXIT_FAILURE;
    hipError_t      hip_err;
    ssize_t         nbytes;

    /* 1. Select GPU */
    hip_err = hipSetDevice(gpu_id);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not select GPU %d (%d)\n", gpu_id, hip_err);
        return EXIT_FAILURE;
    }

    /* 2. Build CPU pattern + copy to GPU */
    cpu_pattern = static_cast<uint8_t *>(malloc(payload_size));
    if (!cpu_pattern) {
        fprintf(stderr, "Could not allocate CPU pattern buffer\n");
        return EXIT_FAILURE;
    }
    fill_pattern(cpu_pattern, payload_size);

    hip_err = hipMalloc(&devbuf, alloc_size);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not allocate %zu bytes on GPU %d (%d)\n", alloc_size, gpu_id, hip_err);
        goto free_cpu_pattern;
    }

    hip_err = hipMemcpy(devbuf, cpu_pattern, payload_size, hipMemcpyHostToDevice);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "hipMemcpy to device failed (%d)\n", hip_err);
        goto free_devbuf;
    }

    /* 3. open file with O_DIRECT + hipFileHandleRegister (no buf registration) */
    if (open_file(out_path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH,
                  &out_fd, &out_handle)) {
        goto free_devbuf;
    }

    /* 4. hipFileWrite via hipFile's internal pool */
    nbytes = hipFileWrite(out_handle, devbuf, alloc_size, /*file_offset=*/0,
                          /*buffer_offset=*/0);
    if (nbytes < 0) {
        fprintf(stderr, "Could not write to %s (%zd) (%s)\n", out_path, nbytes,
                IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
        goto close_out;
    }

    /* 5. ftruncate to exact size */
    if (-1 == ftruncate(out_fd, static_cast<off_t>(payload_size))) {
        fprintf(stderr, "Could not truncate %s (%s)\n", out_path, strerror(errno));
        goto close_out;
    }

    if (close_file(out_path, out_fd, out_handle)) {
        out_fd = -1;
        goto free_devbuf;
    }
    out_fd = -1;

    /* 6. Hash verify */
    {
        uint64_t hash_pattern = hash_buffer(cpu_pattern, payload_size);
        uint64_t hash_written;

        if (hash_file_range(out_path, 0, payload_size, &hash_written))
            goto free_devbuf;

        if (hash_pattern != hash_written) {
            fprintf(stderr, "Hash mismatch: pattern=0x%016" PRIx64 "  %s=0x%016" PRIx64 "\n", hash_pattern,
                    out_path, hash_written);
            goto free_devbuf;
        }

        printf("OK  %s  (%zu bytes, hash 0x%016" PRIx64 ")\n", out_path, payload_size, hash_pattern);
    }

    exit_status = EXIT_SUCCESS;

close_out:
    if (out_fd != -1) {
        if (close_file(out_path, out_fd, out_handle))
            exit_status = EXIT_FAILURE;
    }

free_devbuf:
    hip_err = hipFree(devbuf);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not free device buffer (%d)\n", hip_err);
        exit_status = EXIT_FAILURE;
    }

free_cpu_pattern:
    free(cpu_pattern);
    return exit_status;
}
