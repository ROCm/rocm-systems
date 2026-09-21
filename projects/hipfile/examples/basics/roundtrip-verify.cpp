/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* roundtrip-verify - Full GPU-mediated round trip:
 *   write a deterministic pattern to CREATED via hipFile, read CREATED back
 *   into the same registered GPU buffer, write that to COPIED, then hash the
 *   two files and assert they match.
 *
 * Usage: ./roundtrip-verify CREATED COPIED [GPUID]
 *
 *   CREATED  Path to the first output file. Created/truncated. Receives RV_SIZE
 *            (default 64 KiB) bytes of generated test pattern written from a
 *            registered GPU buffer.
 *   COPIED   Path to the second output file. Created/truncated. Receives the
 *            same payload after a GPU-mediated read-back of CREATED.
 *   GPUID    GPU device index (optional, default 0).
 *
 * Steps:
 *   1. Select GPU
 *   2. Build CPU pattern + copy to GPU + hipFileBufRegister
 *   3. Phase 1: open CREATED + hipFileWrite + ftruncate + close
 *   4. Phase 2: open CREATED for read + hipFileRead back into GPU + close
 *   5. Phase 3: open COPIED + hipFileWrite + ftruncate + close
 *   6. Hash both files and compare
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
#define RV_SIZE (64UL * 1024UL)
#endif

int
main(int argc, char *argv[])
{
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s CREATED COPIED [GPUID]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *created_path = argv[1];
    const char *copied_path  = argv[2];
    const int   gpu_id       = (argc == 4) ? atoi(argv[3]) : 0;

    const size_t payload_size = RV_SIZE;
    const size_t alloc_size   = align_up(payload_size, BLOCK_ALIGN);

    int             fd = -1;
    hipFileHandle_t handle;
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

    /* 2. Build CPU pattern + copy to GPU + hipFileBufRegister */
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

    hipfile_err = hipFileBufRegister(devbuf, alloc_size, 0);
    if (hipFileSuccess != hipfile_err.err) {
        fprintf(stderr, "Buffer register failed (%s)\n", hipFileGetOpErrorString(hipfile_err.err));
        goto free_devbuf;
    }
    buf_registered = true;

    /* 3. Phase 1: write CREATED from registered GPU buffer */
    if (open_file(created_path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT,
                  S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH, &fd, &handle)) {
        goto deregister_buf;
    }

    nbytes = hipFileWrite(handle, devbuf, alloc_size, /*file_offset=*/0, /*buf_offset=*/0);
    if (nbytes < 0) {
        fprintf(stderr, "Could not write to %s (%zd) (%s)\n", created_path, nbytes,
                IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
        close_file(created_path, fd, handle);
        fd = -1;
        goto deregister_buf;
    }

    if (-1 == ftruncate(fd, static_cast<off_t>(payload_size))) {
        fprintf(stderr, "Could not truncate %s (%s)\n", created_path, strerror(errno));
        close_file(created_path, fd, handle);
        fd = -1;
        goto deregister_buf;
    }

    if (close_file(created_path, fd, handle)) {
        fd = -1;
        goto deregister_buf;
    }
    fd = -1;

    /* 4. Phase 2: read CREATED back into the SAME GPU buffer.
     * This overwrites devbuf with whatever is on disk. If the round trip is
     * lossless, devbuf now holds the same payload it did after the H2D copy
     * in step 2. */
    if (open_file(created_path, O_RDONLY | O_DIRECT, 0, &fd, &handle))
        goto deregister_buf;

    nbytes = hipFileRead(handle, devbuf, alloc_size, /*file_offset=*/0, /*buf_offset=*/0);
    if (nbytes < 0) {
        fprintf(stderr, "Could not read from %s (%zd) (%s)\n", created_path, nbytes,
                IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
        close_file(created_path, fd, handle);
        fd = -1;
        goto deregister_buf;
    }

    if (close_file(created_path, fd, handle)) {
        fd = -1;
        goto deregister_buf;
    }
    fd = -1;

    /* 5. Phase 3: write the read-back contents to COPIED */
    if (open_file(copied_path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH,
                  &fd, &handle)) {
        goto deregister_buf;
    }

    nbytes = hipFileWrite(handle, devbuf, alloc_size, /*file_offset=*/0, /*buf_offset=*/0);
    if (nbytes < 0) {
        fprintf(stderr, "Could not write to %s (%zd) (%s)\n", copied_path, nbytes,
                IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
        close_file(copied_path, fd, handle);
        fd = -1;
        goto deregister_buf;
    }

    if (-1 == ftruncate(fd, static_cast<off_t>(payload_size))) {
        fprintf(stderr, "Could not truncate %s (%s)\n", copied_path, strerror(errno));
        close_file(copied_path, fd, handle);
        fd = -1;
        goto deregister_buf;
    }

    if (close_file(copied_path, fd, handle)) {
        fd = -1;
        goto deregister_buf;
    }
    fd = -1;

    /* 6. Hash both files and compare */
    {
        uint64_t hash;
        if (verify_files_match(created_path, copied_path, payload_size, &hash))
            goto deregister_buf;
        printf("OK  %s == %s  (%zu bytes, hash 0x%016" PRIx64 ")\n", created_path, copied_path, payload_size,
               hash);
    }

    exit_status = EXIT_SUCCESS;

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
