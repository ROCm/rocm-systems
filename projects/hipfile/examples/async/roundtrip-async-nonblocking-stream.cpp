/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* roundtrip-async-nonblocking-stream - Same async round trip as
 *   roundtrip-async, but issued on an explicit hipStreamNonBlocking
 *   stream instead of the default stream. The non-blocking flag means this
 *   stream does not implicitly synchronize with the legacy default stream.
 *
 * Usage: ./roundtrip-async-nonblocking-stream READ_FILE WRITE_FILE [GPUID]
 *
 *   READ_FILE   Path to the input file. Created/truncated and seeded via a
 *               plain POSIX write with align_up(RV_SIZE, BLOCK_ALIGN) bytes of
 *               generated test pattern so the O_DIRECT async read of the
 *               aligned size has known content to pull. Only the first RV_SIZE
 *               (default 1 MiB) bytes are treated as the logical payload.
 *   WRITE_FILE  Path to the output file. Opened O_RDWR|O_CREAT|O_DIRECT (the
 *               file is created if missing; existing contents past the write
 *               are left in place by ftruncate down to payload size).
 *   GPUID       GPU device index (optional, default 0).
 *
 * Steps:
 *   1. Select GPU
 *   2. Seed READ_FILE with pattern (plain POSIX write, no O_DIRECT)
 *   3. hipMalloc + hipFileBufRegister
 *   4. Open READ_FILE (O_RDONLY|O_DIRECT) and WRITE_FILE (O_RDWR|O_CREAT|O_DIRECT)
 *   5. hipStreamCreateWithFlags(hipStreamNonBlocking) + hipMemsetAsync(0) on it
 *   6. hipFileReadAsync + hipFileWriteAsync on the non-blocking stream
 *   7. hipStreamSynchronize, assert bytes_read == bytes_written == alloc_size
 *   8. ftruncate WRITE_FILE down to payload size
 *   9. Hash both files and compare
 *  10. hipStreamDestroy, then unwind handles/buffer
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
#ifndef RV_SIZE
#define RV_SIZE (1UL * 1024UL * 1024UL)
#endif

int
main(int argc, char *argv[])
{
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s READ_FILE WRITE_FILE [GPUID]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *read_path  = argv[1];
    const char *write_path = argv[2];
    const int   gpu_id     = (argc == 4) ? atoi(argv[3]) : 0;

    const size_t payload_size = RV_SIZE;
    const size_t alloc_size   = align_up(payload_size, BLOCK_ALIGN);

    int             rfd = -1, wfd = -1;
    hipFileHandle_t rhandle, whandle;
    bool            rhandle_open   = false;
    bool            whandle_open   = false;
    void           *devbuf         = NULL;
    bool            buf_registered = false;
    hipStream_t     stream         = NULL;
    bool            stream_created = false;
    int             exit_status    = EXIT_FAILURE;
    hipError_t      hip_err;
    hipFileError_t  hipfile_err;

    /* 1. Select GPU */
    hip_err = hipSetDevice(gpu_id);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not select GPU %d (%d)\n", gpu_id, hip_err);
        return EXIT_FAILURE;
    }

    /* 2. Seed READ_FILE with a deterministic pattern. Seed alloc_size, not
     * payload_size, so the O_DIRECT-aligned async read below has full content
     * to pull even when RV_SIZE is overridden to a non-BLOCK_ALIGN multiple. */
    if (seed_read_file(read_path, alloc_size))
        return EXIT_FAILURE;

    /* 3. Allocate the GPU buffer and register it with hipFile */
    hip_err = hipMalloc(&devbuf, alloc_size);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not allocate %zu bytes on GPU %d (%d)\n", alloc_size, gpu_id, hip_err);
        return EXIT_FAILURE;
    }

    hipfile_err = hipFileBufRegister(devbuf, alloc_size, 0);
    if (hipFileSuccess != hipfile_err.err) {
        fprintf(stderr, "Buffer register failed (%s)\n", hipFileGetOpErrorString(hipfile_err.err));
        goto free_devbuf;
    }
    buf_registered = true;

    /* 4. Open the input and output files with O_DIRECT and register both.
     * WRITE_FILE is opened O_RDWR so future variants can read-back through
     * the same handle without reopening. */
    if (open_file(read_path, O_RDONLY | O_DIRECT, 0, &rfd, &rhandle))
        goto deregister_buf;
    rhandle_open = true;

    if (open_file(write_path, O_RDWR | O_CREAT | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH, &wfd,
                  &whandle))
        goto close_rfile;
    whandle_open = true;

    /* 5. Create a non-blocking stream and zero the buffer on it. The
     * hipStreamNonBlocking flag means this stream does not implicitly
     * synchronize with the legacy default (NULL) stream. */
    hip_err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "hipStreamCreateWithFlags failed (%d)\n", hip_err);
        goto close_wfile;
    }
    stream_created = true;

    hip_err = hipMemsetAsync(devbuf, 0, alloc_size, stream);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "hipMemsetAsync failed (%d)\n", hip_err);
        goto destroy_stream;
    }

    /* 6+7. Submit async read, then async write, on the non-blocking stream
     * and drain. Same-stream ordering means the write reads what the read
     * deposited without an intermediate sync. */
    {
        size_t  rsize         = alloc_size;
        hoff_t  rfile_offset  = 0;
        hoff_t  rbuf_offset   = 0;
        ssize_t bytes_read    = 0;
        size_t  wsize         = alloc_size;
        hoff_t  wfile_offset  = 0;
        hoff_t  wbuf_offset   = 0;
        ssize_t bytes_written = 0;

        hipfile_err =
            hipFileReadAsync(rhandle, devbuf, &rsize, &rfile_offset, &rbuf_offset, &bytes_read, stream);
        if (hipFileSuccess != hipfile_err.err) {
            fprintf(stderr, "hipFileReadAsync submit failed (%s)\n",
                    hipFileGetOpErrorString(hipfile_err.err));
            goto destroy_stream;
        }

        hipfile_err =
            hipFileWriteAsync(whandle, devbuf, &wsize, &wfile_offset, &wbuf_offset, &bytes_written, stream);
        if (hipFileSuccess != hipfile_err.err) {
            fprintf(stderr, "hipFileWriteAsync submit failed (%s)\n",
                    hipFileGetOpErrorString(hipfile_err.err));
            goto destroy_stream;
        }

        hip_err = hipStreamSynchronize(stream);
        if (hipSuccess != hip_err) {
            fprintf(stderr, "hipStreamSynchronize failed (%d)\n", hip_err);
            goto destroy_stream;
        }

        if (bytes_read != static_cast<ssize_t>(alloc_size)) {
            fprintf(stderr, "Async read short: %zd of %zu\n", bytes_read, alloc_size);
            goto destroy_stream;
        }
        if (bytes_written != static_cast<ssize_t>(alloc_size)) {
            fprintf(stderr, "Async write short: %zd of %zu\n", bytes_written, alloc_size);
            goto destroy_stream;
        }
    }

    /* 8. Trim WRITE_FILE down to the true payload size (we wrote padded). */
    if (-1 == ftruncate(wfd, static_cast<off_t>(payload_size))) {
        fprintf(stderr, "Could not truncate %s (%s)\n", write_path, strerror(errno));
        goto destroy_stream;
    }

    /* 9. Close the O_DIRECT fds before the verify path reopens the files with
     * buffered I/O. Mixing O_DIRECT writes and page-cache reads on the same
     * file has undefined coherence per open(2); closing first sidesteps that
     * and also flushes file-size metadata for the fresh open() in the hasher.
     * Then hash and compare. */
    if (close_file(write_path, wfd, whandle)) {
        wfd          = -1;
        whandle_open = false;
        goto destroy_stream;
    }
    wfd          = -1;
    whandle_open = false;

    if (close_file(read_path, rfd, rhandle)) {
        rfd          = -1;
        rhandle_open = false;
        goto destroy_stream;
    }
    rfd          = -1;
    rhandle_open = false;

    {
        uint64_t hash;
        if (verify_files_match(read_path, write_path, payload_size, &hash))
            goto destroy_stream;

        printf("OK  %s == %s  (%zu bytes, hash 0x%016" PRIx64 ")\n", read_path, write_path, payload_size,
               hash);
    }

    exit_status = EXIT_SUCCESS;

destroy_stream:
    if (stream_created) {
        hip_err = hipStreamDestroy(stream);
        if (hipSuccess != hip_err) {
            fprintf(stderr, "hipStreamDestroy failed (%d)\n", hip_err);
            exit_status = EXIT_FAILURE;
        }
    }

close_wfile:
    if (whandle_open) {
        if (close_file(write_path, wfd, whandle))
            exit_status = EXIT_FAILURE;
    }

close_rfile:
    if (rhandle_open) {
        if (close_file(read_path, rfd, rhandle))
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

    return exit_status;
}
