/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* roundtrip-async-multi-stream-registered - Same NUM_STREAMS x SLICE_SIZE
 *   concurrent round trip as roundtrip-async-multi-stream, but each
 *   stream is additionally registered with hipFileStreamRegister using the
 *   fixed-{buf,file}-offset / fixed-file-size / page-aligned-inputs hints.
 *   These hints let the driver skip per-submission validation on every
 *   subsequent hipFileReadAsync / hipFileWriteAsync that hits this stream.
 *
 * Usage: ./roundtrip-async-multi-stream-registered READ_FILE WRITE_FILE [GPUID]
 *
 *   READ_FILE   Path to the input file. Created/truncated and seeded with
 *               TOTAL_SIZE bytes of generated test pattern via a plain POSIX
 *               write so the async path has known content to read.
 *   WRITE_FILE  Path to the output file. Opened O_WRONLY|O_CREAT|O_DIRECT
 *               (no O_TRUNC). SLICE_SIZE is already block-aligned so no
 *               ftruncate is needed.
 *   GPUID       GPU device index (optional, default 0).
 *
 * Steps:
 *   1. Select GPU
 *   2. Seed READ_FILE with pattern (plain POSIX write, no O_DIRECT)
 *   3. For each of NUM_STREAMS entries:
 *        - hipMalloc + hipFileBufRegister
 *        - Assign file offset i * SLICE_SIZE
 *        - hipStreamCreateWithFlags(hipStreamNonBlocking) + hipMemsetAsync(0)
 *        - hipFileStreamRegister with FIXED_BUF_OFFSET | FIXED_FILE_OFFSET
 *          | FIXED_FILE_SIZE | PAGE_ALIGNED_INPUTS
 *   4. Open READ_FILE (O_RDONLY|O_DIRECT) and WRITE_FILE (O_WRONLY|O_CREAT|O_DIRECT)
 *   5. Loop: submit hipFileReadAsync on each stream
 *   6. Loop: submit hipFileWriteAsync on each stream
 *   7. Loop: hipStreamSynchronize per stream + assert per-entry byte counts
 *   8. Hash the full TOTAL_SIZE of both files and compare
 *   9. Cleanup unwinds each entry:
 *        hipFileStreamDeregister -> hipStreamDestroy -> hipFileBufDeregister -> hipFree
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

/// @brief Number of concurrent streams / slices.
#ifndef NUM_STREAMS
#define NUM_STREAMS 4
#endif

/// @brief Per-stream slice size in bytes. Must be a multiple of BLOCK_ALIGN.
#ifndef SLICE_SIZE
#define SLICE_SIZE (1UL * 1024UL * 1024UL)
#endif

#define TOTAL_SIZE ((size_t)NUM_STREAMS * (size_t)SLICE_SIZE)

static_assert((SLICE_SIZE % BLOCK_ALIGN) == 0, "SLICE_SIZE must be a multiple of BLOCK_ALIGN");

/// @brief Optimization hints passed to hipFileStreamRegister. Asserts to the
/// driver that buffer offset, file offset, and file size are fixed across all
/// submissions on this stream and that all three are 4 KiB aligned.
#define STREAM_REG_FLAGS                                                                                     \
    (HIPFILE_STREAM_FIXED_BUF_OFFSET | HIPFILE_STREAM_FIXED_FILE_OFFSET | HIPFILE_STREAM_FIXED_FILE_SIZE |   \
     HIPFILE_STREAM_PAGE_ALIGNED_INPUTS)

/* Per-stream state. Sizes/offsets live in the struct because the async API
 * takes them by pointer and dereferences them at completion time, so they
 * must outlive submission. */
struct slice_state {
    void       *devbuf;
    bool        buf_registered;
    hipStream_t stream;
    bool        stream_created;
    bool        stream_registered;
    size_t      io_size;
    hoff_t      file_offset;
    hoff_t      buf_offset;
    ssize_t     bytes_read;
    ssize_t     bytes_written;
};

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

    int             rfd = -1, wfd = -1;
    hipFileHandle_t rhandle, whandle;
    bool            rhandle_open        = false;
    bool            whandle_open        = false;
    slice_state     slices[NUM_STREAMS] = {};
    int             exit_status         = EXIT_FAILURE;
    hipError_t      hip_err;
    hipFileError_t  hipfile_err;

    /* 1. Select GPU */
    hip_err = hipSetDevice(gpu_id);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not select GPU %d (%d)\n", gpu_id, hip_err);
        return EXIT_FAILURE;
    }

    /* 2. Seed READ_FILE with a deterministic pattern */
    if (seed_read_file(read_path, TOTAL_SIZE))
        return EXIT_FAILURE;

    /* 3. Per-entry setup: allocate, register buffer, build stream, zero on it,
     * then register the stream with the driver hints. On any failure, jump to
     * cleanup_slices which walks slices[] and unwinds whatever each entry has
     * populated so far (per-entry bool flags). */
    for (int i = 0; i < NUM_STREAMS; ++i) {
        slice_state &s = slices[i];
        s.io_size      = SLICE_SIZE;
        s.file_offset  = (hoff_t)i * (hoff_t)SLICE_SIZE;
        s.buf_offset   = 0;

        hip_err = hipMalloc(&s.devbuf, SLICE_SIZE);
        if (hipSuccess != hip_err) {
            fprintf(stderr, "Could not allocate %zu bytes on GPU %d for slice %d (%d)\n", (size_t)SLICE_SIZE,
                    gpu_id, i, hip_err);
            goto cleanup_slices;
        }

        hipfile_err = hipFileBufRegister(s.devbuf, SLICE_SIZE, 0);
        if (hipFileSuccess != hipfile_err.err) {
            fprintf(stderr, "Buffer register failed for slice %d (%s)\n", i,
                    hipFileGetOpErrorString(hipfile_err.err));
            goto cleanup_slices;
        }
        s.buf_registered = true;

        hip_err = hipStreamCreateWithFlags(&s.stream, hipStreamNonBlocking);
        if (hipSuccess != hip_err) {
            fprintf(stderr, "hipStreamCreateWithFlags failed for slice %d (%d)\n", i, hip_err);
            goto cleanup_slices;
        }
        s.stream_created = true;

        hip_err = hipMemsetAsync(s.devbuf, 0, SLICE_SIZE, s.stream);
        if (hipSuccess != hip_err) {
            fprintf(stderr, "hipMemsetAsync failed for slice %d (%d)\n", i, hip_err);
            goto cleanup_slices;
        }

        hipfile_err = hipFileStreamRegister(s.stream, STREAM_REG_FLAGS);
        if (hipFileSuccess != hipfile_err.err) {
            fprintf(stderr, "hipFileStreamRegister failed for slice %d (%s)\n", i,
                    hipFileGetOpErrorString(hipfile_err.err));
            goto cleanup_slices;
        }
        s.stream_registered = true;
    }

    /* 4. Open the input and output files with O_DIRECT and register both.
     * The same pair of fds is shared across all NUM_STREAMS streams. */
    if (open_file(read_path, O_RDONLY | O_DIRECT, 0, &rfd, &rhandle))
        goto cleanup_slices;
    rhandle_open = true;

    if (open_file(write_path, O_WRONLY | O_CREAT | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH, &wfd,
                  &whandle))
        goto close_rfile;
    whandle_open = true;

    /* 5. Submit all reads. With the stream registered under the fixed-*
     * hints, the driver can skip per-submission validation of buf_offset,
     * file_offset, and io_size. */
    for (int i = 0; i < NUM_STREAMS; ++i) {
        slice_state &s = slices[i];
        hipfile_err    = hipFileReadAsync(rhandle, s.devbuf, &s.io_size, &s.file_offset, &s.buf_offset,
                                          &s.bytes_read, s.stream);
        if (hipFileSuccess != hipfile_err.err) {
            fprintf(stderr, "hipFileReadAsync submit failed for slice %d (%s)\n", i,
                    hipFileGetOpErrorString(hipfile_err.err));
            goto close_wfile;
        }
    }

    /* 6. Submit all writes. Each write is on the same stream as its
     * corresponding read, so it observes the read's completion. */
    for (int i = 0; i < NUM_STREAMS; ++i) {
        slice_state &s = slices[i];
        hipfile_err    = hipFileWriteAsync(whandle, s.devbuf, &s.io_size, &s.file_offset, &s.buf_offset,
                                           &s.bytes_written, s.stream);
        if (hipFileSuccess != hipfile_err.err) {
            fprintf(stderr, "hipFileWriteAsync submit failed for slice %d (%s)\n", i,
                    hipFileGetOpErrorString(hipfile_err.err));
            goto close_wfile;
        }
    }

    /* 7. Drain each stream and check completion counts. */
    for (int i = 0; i < NUM_STREAMS; ++i) {
        slice_state &s = slices[i];
        hip_err        = hipStreamSynchronize(s.stream);
        if (hipSuccess != hip_err) {
            fprintf(stderr, "hipStreamSynchronize failed for slice %d (%d)\n", i, hip_err);
            goto close_wfile;
        }

        if (s.bytes_read != (ssize_t)SLICE_SIZE) {
            fprintf(stderr, "Async read short on slice %d: %zd of %zu\n", i, s.bytes_read,
                    (size_t)SLICE_SIZE);
            goto close_wfile;
        }
        if (s.bytes_written != (ssize_t)SLICE_SIZE) {
            fprintf(stderr, "Async write short on slice %d: %zd of %zu\n", i, s.bytes_written,
                    (size_t)SLICE_SIZE);
            goto close_wfile;
        }
    }

    /* 8. Close the O_DIRECT fds before the verify path reopens the files with
     * buffered I/O. Mixing O_DIRECT writes and page-cache reads on the same
     * file has undefined coherence per open(2); closing first sidesteps that
     * and also flushes file-size metadata for the fresh open() in the hasher.
     * Then hash the full TOTAL_SIZE and compare. */
    if (close_file(write_path, wfd, whandle)) {
        wfd          = -1;
        whandle_open = false;
        goto close_rfile;
    }
    wfd          = -1;
    whandle_open = false;

    if (close_file(read_path, rfd, rhandle)) {
        rfd          = -1;
        rhandle_open = false;
        goto cleanup_slices;
    }
    rfd          = -1;
    rhandle_open = false;

    {
        uint64_t hash;
        if (verify_files_match(read_path, write_path, TOTAL_SIZE, &hash))
            goto cleanup_slices;

        printf("OK  %s == %s  (%zu bytes across %d registered streams, hash 0x%016" PRIx64 ")\n", read_path,
               write_path, (size_t)TOTAL_SIZE, NUM_STREAMS, hash);
    }

    exit_status = EXIT_SUCCESS;

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

cleanup_slices:
    /* Teardown order:
     *   hipFileStreamDeregister -> hipStreamDestroy -> hipFileBufDeregister -> hipFree
     * Per-entry bool flags handle partial setup from a failed init loop. */
    for (int i = 0; i < NUM_STREAMS; ++i) {
        slice_state &s = slices[i];

        if (s.stream_registered) {
            hipfile_err = hipFileStreamDeregister(s.stream);
            if (hipFileSuccess != hipfile_err.err) {
                fprintf(stderr, "hipFileStreamDeregister failed for slice %d (%s)\n", i,
                        hipFileGetOpErrorString(hipfile_err.err));
                exit_status = EXIT_FAILURE;
            }
        }

        if (s.stream_created) {
            hip_err = hipStreamDestroy(s.stream);
            if (hipSuccess != hip_err) {
                fprintf(stderr, "hipStreamDestroy failed for slice %d (%d)\n", i, hip_err);
                exit_status = EXIT_FAILURE;
            }
        }

        if (s.buf_registered) {
            hipfile_err = hipFileBufDeregister(s.devbuf);
            if (hipFileSuccess != hipfile_err.err) {
                fprintf(stderr, "Buffer deregister failed for slice %d (%s)\n", i,
                        hipFileGetOpErrorString(hipfile_err.err));
                exit_status = EXIT_FAILURE;
            }
        }

        if (s.devbuf) {
            hip_err = hipFree(s.devbuf);
            if (hipSuccess != hip_err) {
                fprintf(stderr, "Could not free device buffer for slice %d (%d)\n", i, hip_err);
                exit_status = EXIT_FAILURE;
            }
        }
    }

    return exit_status;
}
