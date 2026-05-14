/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* iterative-devmem-offset-read - Read INPUT into a registered GPU buffer
 * chunk-by-chunk using hipFile's buf_offset parameter (the base device
 * pointer stays fixed), then write the full buffer to OUTPUT in one call.
 *
 * Usage: ./iterative-devmem-offset-read INPUT OUTPUT [GPUID]
 *
 *   INPUT    Existing file to read. Up to IDOR_CAP (default 1 MiB) is
 *            consumed. Create with:
 *              dd if=/dev/urandom of=input.bin bs=1M count=1
 *   OUTPUT   Path to the output file. Created/truncated. Receives a copy of
 *            the bytes read from INPUT.
 *   GPUID    GPU device index (optional, default 0).
 *
 * Chunk size is IDOR_CHUNK_SIZE (default 64 KiB); override at compile time.
 *
 * Unlike iterative-read, the GPU buffer is registered with
 * hipFileBufRegister and the buf_offset argument to hipFileRead advances
 * each iteration instead of the base host pointer.
 *
 * Steps:
 *   1. Select GPU
 *   2. Open + register input file
 *   3. Allocate + zero device buffer + hipFileBufRegister
 *   4. Chunk-read loop (base pointer fixed, buf_offset parameter advances)
 *   5. Open + register output file + hipFileWrite
 *   6. ftruncate to exact size + hash verify
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

/// @brief Hard cap on bytes consumed from INPUT. Override at compile time.
#ifndef IDOR_CAP
#define IDOR_CAP (1024UL * 1024UL)
#endif

/// @brief Per-iteration read size. Must be a multiple of the filesystem
/// block size for O_DIRECT. Override at compile time.
#ifndef IDOR_CHUNK_SIZE
#define IDOR_CHUNK_SIZE (64UL * 1024UL)
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

    const size_t chunk_size = IDOR_CHUNK_SIZE;
    if (chunk_size == 0 || (chunk_size & (block_size - 1)) != 0) {
        fprintf(stderr, "IDOR_CHUNK_SIZE (%zu) must be a non-zero multiple of block size (%zu)\n", chunk_size,
                block_size);
        return EXIT_FAILURE;
    }

    const size_t payload_size = (file_size < IDOR_CAP) ? file_size : IDOR_CAP;
    const size_t alloc_size   = align_up(payload_size, block_size);

    int             in_fd  = -1;
    int             out_fd = -1;
    hipFileHandle_t in_handle, out_handle;
    void           *devbuf         = NULL;
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

    /* 2. Open + register input file */
    if (open_file(in_path, O_RDONLY | O_DIRECT, 0, &in_fd, &in_handle))
        return EXIT_FAILURE;

    /* 3. Allocate + zero device buffer + hipFileBufRegister */
    hip_err = hipMalloc(&devbuf, alloc_size);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not allocate %zu bytes on GPU %d (%d)\n", alloc_size, gpu_id, hip_err);
        goto close_in;
    }

    hip_err = hipMemset(devbuf, 0, alloc_size);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not zero device buffer (%d)\n", hip_err);
        goto free_devbuf;
    }

    hipfile_err = hipFileBufRegister(devbuf, alloc_size, 0);
    if (hipFileSuccess != hipfile_err.err) {
        fprintf(stderr, "Buffer register failed (%s)\n", hipFileGetOpErrorString(hipfile_err.err));
        goto free_devbuf;
    }
    buf_registered = true;

    /* 4. Chunk-read loop. The base device pointer (devbuf) stays fixed; the
     *    buf_offset argument advances by `bytes_read` each iteration. */
    {
        size_t bytes_read = 0;
        while (bytes_read < alloc_size) {
            const size_t remaining  = alloc_size - bytes_read;
            const size_t this_chunk = (chunk_size < remaining) ? chunk_size : remaining;

            nbytes = hipFileRead(in_handle, devbuf, this_chunk,
                                 /*file_offset=*/(hoff_t)bytes_read,
                                 /*buf_offset=*/(hoff_t)bytes_read);
            if (nbytes < 0) {
                fprintf(stderr, "Could not read from %s (%zd) (%s)\n", in_path, nbytes,
                        IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
                goto deregister_buf;
            }
            if (nbytes == 0)
                break; /* EOF — file ended before alloc_size; OK */
            bytes_read += (size_t)nbytes;
        }

        if (bytes_read < payload_size) {
            fprintf(stderr, "Short read on %s: got %zu bytes, expected at least %zu\n", in_path, bytes_read,
                    payload_size);
            goto deregister_buf;
        }
    }

    /* 5. Open + register output file + hipFileWrite */
    if (open_file(out_path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH,
                  &out_fd, &out_handle)) {
        goto deregister_buf;
    }

    nbytes = hipFileWrite(out_handle, devbuf, alloc_size, /*file_offset=*/0,
                          /*buf_offset=*/0);
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
        goto deregister_buf;
    }
    out_fd = -1;

    {
        uint64_t hash_in, hash_out;

        if (hash_file_range(in_path, 0, payload_size, &hash_in))
            goto deregister_buf;
        if (hash_file_range(out_path, 0, payload_size, &hash_out))
            goto deregister_buf;

        if (hash_in != hash_out) {
            fprintf(stderr, "Hash mismatch: %s=0x%016" PRIx64 "  %s=0x%016" PRIx64 "\n", in_path, hash_in,
                    out_path, hash_out);
            goto deregister_buf;
        }

        printf("OK  %s -> %s  (%zu bytes, hash 0x%016" PRIx64 ")\n", in_path, out_path, payload_size,
               hash_in);
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

close_in:
    if (close_file(in_path, in_fd, in_handle))
        exit_status = EXIT_FAILURE;

    return exit_status;
}
