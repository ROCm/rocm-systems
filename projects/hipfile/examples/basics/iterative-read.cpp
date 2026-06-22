/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* iterative-read - Read INPUT into GPU memory chunk-by-chunk (host pointer
 * advances each iteration), then write the full buffer to OUTPUT in one call.
 *
 * Usage: ./iterative-read INPUT OUTPUT [GPUID]
 *
 *   INPUT    Existing file to read. Up to IR_CAP (default 16 MiB) is
 *            consumed; larger files are silently truncated to that limit.
 *            Create with:
 *              dd if=/dev/urandom of=input.bin bs=1M count=1
 *   OUTPUT   Path to the output file. Created/truncated. Receives a copy of
 *            the bytes read from INPUT.
 *   GPUID    GPU device index (optional, default 0).
 *
 * Chunk size is IR_CHUNK_SIZE (default 64 KiB); override at compile time.
 *
 * Steps:
 *   1. Select GPU
 *   2. Open + register input file
 *   3. Allocate device buffer for full read
 *   4. Chunk-read loop (host pointer advances each iteration)
 *   5. Open + register output file + hipFileWrite in one call
 *   6. ftruncate to exact size + hash verify
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
#ifndef IR_CAP
#define IR_CAP (16UL * 1024UL * 1024UL)
#endif

/// @brief Per-iteration read size. Must be a multiple of the filesystem
/// block size for O_DIRECT. Override at compile time.
#ifndef IR_CHUNK_SIZE
#define IR_CHUNK_SIZE (64UL * 1024UL)
#endif

int
main(int argc, char *argv[])
{
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s INPUT OUTPUT [GPUID]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *in_path  = argv[1];
    const char *out_path = argv[2];
    const int   gpu_id   = (argc == 4) ? atoi(argv[3]) : 0;

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

    const size_t chunk_size = IR_CHUNK_SIZE;
    if (chunk_size == 0 || (chunk_size & (block_size - 1)) != 0) {
        fprintf(stderr, "IR_CHUNK_SIZE (%zu) must be a non-zero multiple of block size (%zu)\n", chunk_size,
                block_size);
        return EXIT_FAILURE;
    }

    /* Cap the consumed bytes; over-sized files are silently truncated to IR_CAP. */
    const size_t payload_size = (file_size < IR_CAP) ? file_size : IR_CAP;
    const size_t alloc_size   = align_up(payload_size, block_size);

    int             in_fd  = -1;
    int             out_fd = -1;
    hipFileHandle_t in_handle, out_handle;
    void           *devbuf      = NULL;
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

    /* 3. Allocate device buffer for the full read */
    hip_err = hipMalloc(&devbuf, alloc_size);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not allocate %zu bytes on GPU %d (%d)\n", alloc_size, gpu_id, hip_err);
        goto close_in;
    }

    /* Zero the buffer so any alignment padding past payload_size is deterministic
     * rather than uninitialized device memory. The padding is still written by
     * hipFileWrite (alloc_size > payload_size when payload_size is not block-
     * aligned) and only trimmed by ftruncate. */
    hip_err = hipMemset(devbuf, 0, alloc_size);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not zero device buffer (%d)\n", hip_err);
        goto free_devbuf;
    }

    /* hipMemset is async w.r.t. the host; block until it completes (testing). */
    hip_err = hipDeviceSynchronize();
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not synchronize after memset (%d)\n", hip_err);
        goto free_devbuf;
    }

    /* 4. Chunk-read loop. The host pointer passed to hipFileRead advances by
     *    `bytes_read` each iteration; the file_offset advances in lockstep. */
    {
        size_t bytes_read = 0;
        while (bytes_read < alloc_size) {
            const size_t remaining  = alloc_size - bytes_read;
            const size_t this_chunk = (chunk_size < remaining) ? chunk_size : remaining;
            void        *dst        = (char *)devbuf + bytes_read;

            nbytes = hipFileRead(in_handle, dst, this_chunk, (hoff_t)bytes_read, 0);
            if (nbytes < 0) {
                fprintf(stderr, "Could not read from %s (%zd) (%s)\n", in_path, nbytes,
                        IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
                goto free_devbuf;
            }
            if (nbytes == 0)
                break; /* EOF — file ended before alloc_size; OK */
            bytes_read += (size_t)nbytes;
        }

        if (bytes_read < payload_size) {
            fprintf(stderr, "Short read on %s: got %zu bytes, expected at least %zu\n", in_path, bytes_read,
                    payload_size);
            goto free_devbuf;
        }
    }

    /* 5. Open + register output file + hipFileWrite in one call */
    if (open_file(out_path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH,
                  &out_fd, &out_handle)) {
        goto free_devbuf;
    }

    nbytes = hipFileWrite(out_handle, devbuf, alloc_size, /*file_offset=*/0,
                          /*buffer_offset=*/0);
    if (nbytes < 0) {
        fprintf(stderr, "Could not write to %s (%zd) (%s)\n", out_path, nbytes,
                IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
        goto close_out;
    }

    /* 6. ftruncate to exact size + hash verify */
    if (-1 == ftruncate(out_fd, (off_t)payload_size)) {
        fprintf(stderr, "Could not truncate %s (%s)\n", out_path, strerror(errno));
        goto close_out;
    }

    if (close_file(out_path, out_fd, out_handle)) {
        out_fd = -1;
        goto close_in;
    }
    out_fd = -1;

    {
        uint64_t hash;
        if (verify_files_match(in_path, out_path, payload_size, &hash))
            goto close_in;
        printf("OK  %s -> %s  (%zu bytes, hash 0x%016" PRIx64 ")\n", in_path, out_path, payload_size, hash);
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

close_in:
    if (close_file(in_path, in_fd, in_handle))
        exit_status = EXIT_FAILURE;

    return exit_status;
}
