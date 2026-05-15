/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* no-odirect-write - Write a registered GPU buffer to a file opened without
 * O_DIRECT. hipFile falls back to its POSIX-IO path because the fast
 * GPU-direct path requires O_DIRECT.
 *
 * Usage: ./no-odirect-write OUTPUT [GPUID]
 *
 *   OUTPUT   Path to the output file. Created/truncated. Receives NOW_SIZE
 *            (default 128 KiB) bytes of generated test pattern via the
 *            compat/POSIX fallback path because the file is opened without
 *            O_DIRECT.
 *   GPUID    GPU device index (optional, default 0).
 *
 * Steps:
 *   1. Select GPU
 *   2. Build CPU pattern + copy to GPU
 *   3. hipFileBufRegister
 *   4. open file without O_DIRECT + hipFileHandleRegister
 *   5. hipFileWrite via POSIX fallback
 *   6. ftruncate to exact size
 *   7. Hash verify
 *
 * Compat mode must be enabled via env var:
 *   HIPFILE_ALLOW_COMPAT_MODE=1 ./no-odirect-write OUTPUT
 */

#include "basics_common.h"

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
#ifndef NOW_SIZE
#define NOW_SIZE (128UL * 1024UL)
#endif

/// @brief Alignment used for the registered GPU buffer (must be a power of two).
#define BLOCK_ALIGN ((size_t)4096)
static_assert(is_power_of_two(BLOCK_ALIGN), "BLOCK_ALIGN must be a power of two");

int
main(int argc, char *argv[])
{
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s OUTPUT [GPUID]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char  *out_path     = argv[1];
    const int    gpu_id       = (argc == 3) ? atoi(argv[2]) : 0;
    const size_t payload_size = NOW_SIZE;
    const size_t alloc_size   = align_up(payload_size, BLOCK_ALIGN);

    int             out_fd = -1;
    hipFileHandle_t out_handle;
    void           *devbuf         = NULL;
    uint8_t        *cpu_pattern    = NULL;
    bool            buf_registered = false;
    int             exit_status    = EXIT_FAILURE;
    hipError_t      hip_err;
    hipFileError_t  hipfile_err;
    ssize_t         nbytes;

    /* 1. Select GPU */
    hip_err = hipSetDevice(gpu_id);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not select GPU %d (%d)\n", gpu_id, hip_err);
        return EXIT_FAILURE;
    }

    /* 2. Build CPU pattern + copy to GPU */
    cpu_pattern = (uint8_t *)malloc(payload_size);
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

    /* 3. hipFileBufRegister */
    hipfile_err = hipFileBufRegister(devbuf, alloc_size, 0);
    if (hipFileSuccess != hipfile_err.err) {
        fprintf(stderr, "Buffer register failed (%s)\n", hipFileGetOpErrorString(hipfile_err.err));
        goto free_devbuf;
    }
    buf_registered = true;

    /* 4. open file WITHOUT O_DIRECT + hipFileHandleRegister */
    if (open_file(out_path, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH, &out_fd,
                  &out_handle)) {
        goto deregister_buf;
    }

    /* 5. hipFileWrite via POSIX fallback.
     * Without O_DIRECT there is no transfer-size alignment requirement, so we
     * can write the exact payload size. */
    nbytes = hipFileWrite(out_handle, devbuf, payload_size, /*file_offset=*/0,
                          /*buffer_offset=*/0);
    if (nbytes < 0) {
        fprintf(stderr, "Could not write to %s (%zd) (%s)\n", out_path, nbytes,
                IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
        goto close_out;
    }

    /* 6. ftruncate to exact size (defensive; no-op when we wrote payload_size exactly) */
    if (-1 == ftruncate(out_fd, (off_t)payload_size)) {
        fprintf(stderr, "Could not truncate %s (%s)\n", out_path, strerror(errno));
        goto close_out;
    }

    if (close_file(out_path, out_fd, out_handle)) {
        out_fd = -1;
        goto deregister_buf;
    }
    out_fd = -1;

    /* 7. Hash verify */
    {
        uint64_t hash_pattern = hash_buffer(cpu_pattern, payload_size);
        uint64_t hash_written;

        if (hash_file_range(out_path, 0, payload_size, &hash_written))
            goto deregister_buf;

        if (hash_pattern != hash_written) {
            fprintf(stderr, "Hash mismatch: pattern=0x%016" PRIx64 "  %s=0x%016" PRIx64 "\n", hash_pattern,
                    out_path, hash_written);
            goto deregister_buf;
        }

        printf("OK  %s  (%zu bytes, hash 0x%016" PRIx64 ")\n", out_path, payload_size, hash_pattern);
    }

    exit_status = EXIT_SUCCESS;

close_out:
    if (out_fd != -1) {
        if (close_file(out_path, out_fd, out_handle))
            exit_status = EXIT_FAILURE;
    }

deregister_buf:
    if (buf_registered) {
        hipfile_err = hipFileBufDeregister(devbuf);
        if (hipFileSuccess != hipfile_err.err) {
            fprintf(stderr, "Buffer deregister failed (%s)\n", hipFileGetOpErrorString(hipfile_err.err));
            exit_status = EXIT_FAILURE;
        }
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
