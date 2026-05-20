/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* subregion-write - Read INPUT into a registered GPU buffer in one call, then
 * write only the bytes at or after SW_SUB_OFFSET to OUTPUT. The write uses
 * non-zero buffer_offset to skip the leading bytes in GPU memory while
 * writing from offset 0 in the output file.
 *
 * Usage: ./subregion-write INPUT OUTPUT [GPUID]
 *
 *   INPUT    Existing file to read. Up to SW_CAP (default 1 MiB) is consumed.
 *            Must be larger than SW_SUB_OFFSET (default 8192). Create with:
 *              dd if=/dev/urandom of=input.bin bs=1M count=1
 *   OUTPUT   Path to the output file. Created/truncated. Receives the bytes
 *            of INPUT starting at SW_SUB_OFFSET (size = payload - SW_SUB_OFFSET).
 *   GPUID    GPU device index (optional, default 0).
 *
 * Steps:
 *   1. Select GPU
 *   2. Open + register input file
 *   3. Allocate device buffer + hipFileBufRegister + hipFileRead full file
 *   4. Open + register output file
 *   5. hipFileWrite with file_offset=0 and buffer_offset=SW_SUB_OFFSET
 *   6. ftruncate to logical sub-region size + hash verify
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
#ifndef SW_CAP
#define SW_CAP (1024UL * 1024UL)
#endif

/// @brief Byte offset into the in-memory file image at which writing starts.
/// Must be a multiple of the filesystem block size for O_DIRECT.
#ifndef SW_SUB_OFFSET
#define SW_SUB_OFFSET (8192UL)
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

    const size_t sub_offset = SW_SUB_OFFSET;
    if (sub_offset == 0 || (sub_offset & (block_size - 1)) != 0) {
        fprintf(stderr, "SW_SUB_OFFSET (%zu) must be a non-zero multiple of block size (%zu)\n", sub_offset,
                block_size);
        return EXIT_FAILURE;
    }

    const size_t payload_size = (file_size < SW_CAP) ? file_size : SW_CAP;
    if (sub_offset >= payload_size) {
        fprintf(stderr, "SW_SUB_OFFSET (%zu) must be smaller than the consumed payload (%zu bytes)\n",
                sub_offset, payload_size);
        return EXIT_FAILURE;
    }
    const size_t alloc_size = align_up(payload_size, block_size);
    const size_t write_size = payload_size - sub_offset;
    const size_t write_xfer = align_up(write_size, block_size);

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

    /* 3. Allocate device buffer + hipFileBufRegister + hipFileRead full file */
    hip_err = hipMalloc(&devbuf, alloc_size);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not allocate %zu bytes on GPU %d (%d)\n", alloc_size, gpu_id, hip_err);
        goto close_in;
    }

    hipfile_err = hipFileBufRegister(devbuf, alloc_size, 0);
    if (hipFileSuccess != hipfile_err.err) {
        fprintf(stderr, "Buffer register failed (%s)\n", hipFileGetOpErrorString(hipfile_err.err));
        goto free_devbuf;
    }
    buf_registered = true;

    /* Zero the buffer so any alignment padding past payload_size is deterministic
     * rather than whatever uninitialized device memory contained. The padding is
     * still written out by hipFileWrite (write_xfer >= write_size) and only
     * trimmed by ftruncate. */
    hip_err = hipMemset(devbuf, 0, alloc_size);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not zero device buffer (%d)\n", hip_err);
        goto deregister_buf;
    }

    nbytes = hipFileRead(in_handle, devbuf, alloc_size, /*file_offset=*/0, /*buf_offset=*/0);
    if (nbytes < 0) {
        fprintf(stderr, "Could not read from %s (%zd) (%s)\n", in_path, nbytes,
                IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
        goto deregister_buf;
    }
    if ((size_t)nbytes < payload_size) {
        fprintf(stderr, "Short read on %s: got %zd bytes, expected at least %zu\n", in_path, nbytes,
                payload_size);
        goto deregister_buf;
    }

    /* 4. Open + register output file */
    if (open_file(out_path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH,
                  &out_fd, &out_handle)) {
        goto deregister_buf;
    }

    /* 5. hipFileWrite with file_offset=0 and buffer_offset=sub_offset.
     * The write reads from devbuf[sub_offset .. sub_offset+write_xfer) and
     * lands at OUTPUT[0 .. write_xfer). The trailing bytes past write_size
     * (alignment padding read past payload_size) are removed by ftruncate. */
    nbytes = hipFileWrite(out_handle, devbuf, write_xfer, /*file_offset=*/0,
                          /*buf_offset=*/(hoff_t)sub_offset);
    if (nbytes < 0) {
        fprintf(stderr, "Could not write to %s (%zd) (%s)\n", out_path, nbytes,
                IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
        goto close_out;
    }

    /* 6. ftruncate to logical sub-region size + hash verify */
    if (-1 == ftruncate(out_fd, (off_t)write_size)) {
        fprintf(stderr, "Could not truncate %s (%s)\n", out_path, strerror(errno));
        goto close_out;
    }

    if (close_file(out_path, out_fd, out_handle)) {
        out_fd = -1;
        goto deregister_buf;
    }
    out_fd = -1;

    {
        uint64_t hash_in_tail, hash_out;

        /* Hash bytes [sub_offset, payload_size) of INPUT vs all of OUTPUT. */
        if (hash_file_range(in_path, (off_t)sub_offset, write_size, &hash_in_tail))
            goto deregister_buf;
        if (hash_file_range(out_path, 0, write_size, &hash_out))
            goto deregister_buf;

        if (hash_in_tail != hash_out) {
            fprintf(stderr, "Hash mismatch: %s[%zu:%zu]=0x%016" PRIx64 "  %s=0x%016" PRIx64 "\n", in_path,
                    sub_offset, payload_size, hash_in_tail, out_path, hash_out);
            goto deregister_buf;
        }

        printf("OK  %s[%zu:%zu] -> %s  (%zu bytes, hash 0x%016" PRIx64 ")\n", in_path, sub_offset,
               payload_size, out_path, write_size, hash_in_tail);
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
