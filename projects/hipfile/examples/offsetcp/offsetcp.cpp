/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* filetomem - Read a file into GPU memory, then write a region of it to an output file
 *
 * Usage: ./filetomem INPUT OUTPUT GPUID
 *
 * Reads the entirety of INPUT into device memory on GPUID via hipFile, then
 * writes all bytes at or after FILETOMEM_WRITE_OFFSET (default: 0) to OUTPUT.
 */

#include <hipfile.h>
#include <hip/hip_runtime_api.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/// @brief Byte offset into the in-memory file image from which to start writing OUTPUT.
/// Override at compile time with -DFILETOMEM_WRITE_OFFSET=<bytes>.
#ifndef FILETOMEM_WRITE_OFFSET
#define FILETOMEM_WRITE_OFFSET 8192
#endif

/// @brief Round value up to the next multiple of align. align _must_ be a power of 2.
static inline size_t
align_up(size_t value, size_t align)
{
    return (value + align - 1) & ~(align - 1);
}

/// @brief Determine if value is a power of two
static inline bool
is_power_of_two(size_t value)
{
    return (value > 0) && ((value & (value - 1)) == 0);
}

/// @brief Open a file and register it with hipFile
/// @param path   [in]  Path to the file
/// @param flags  [in]  Flags to pass to open(2) (O_DIRECT is added automatically)
/// @param mode   [in]  Mode to pass to open(2)
/// @param fd     [out] Resulting file descriptor
/// @param handle [out] Resulting hipFile handle
/// @return zero on success, non-zero on failure
static int
open_file(const char *path, int flags, mode_t mode, int *fd, hipFileHandle_t *handle)
{
    hipFileError_t  hipfile_err;
    hipFileDescr_t  descr;

    *fd = open(path, flags | O_DIRECT, mode);
    if (-1 == *fd) {
        fprintf(stderr, "Could not open %s (%s)\n", path, strerror(errno));
        return 1;
    }

    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = *fd;

    hipfile_err = hipFileHandleRegister(handle, &descr);
    if (hipFileSuccess != hipfile_err.err) {
        fprintf(stderr, "Could not register %s (%s)\n", path,
                hipFileGetOpErrorString(hipfile_err.err));
        close(*fd);
        return 1;
    }

    return 0;
}

/// @brief Deregister a hipFile handle and close the underlying file descriptor
/// @param path   [in] Path (used only for error messages)
/// @param fd     [in] File descriptor to close
/// @param handle [in] hipFile handle to deregister
/// @return zero on success, non-zero on failure
static int
close_file(const char *path, int fd, hipFileHandle_t handle)
{
    hipFileHandleDeregister(handle);
    if (-1 == close(fd)) {
        fprintf(stderr, "Could not close %s (%s)\n", path, strerror(errno));
        return 1;
    }
    return 0;
}

int
main(int argc, char *argv[])
{
    const char     *in_path, *out_path;
    int             in_fd, out_fd;
    hipFileHandle_t in_handle, out_handle;
    void           *devbuf         = nullptr;
    hipError_t      hip_err;
    int             exit_status    = EXIT_FAILURE;
    size_t          file_size, block_size, alloc_size;
    ssize_t         nread, nwrite, nbytes;
    hoff_t          file_offset;
    int             gpu_id;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s INPUT OUTPUT GPUID\n", argv[0]);
        return EXIT_FAILURE;
    }

    in_path  = argv[1];
    out_path = argv[2];
    gpu_id   = atoi(argv[3]);

    // Resolve file size and block size from the input file
    {
        struct stat statbuf;
        if (stat(in_path, &statbuf)) {
            fprintf(stderr, "Could not stat %s (%s)\n", in_path, strerror(errno));
            return EXIT_FAILURE;
        }
        file_size  = static_cast<size_t>(statbuf.st_size);
        block_size = static_cast<size_t>(statbuf.st_blksize);
        if (!is_power_of_two(block_size)) {
            fprintf(stderr, "Block size is not a power of two (%zu)\n", block_size);
            return EXIT_FAILURE;
        }
    }

    // Validate the write offset before doing any real work
    const size_t write_offset = FILETOMEM_WRITE_OFFSET;
    if (write_offset > file_size) {
        fprintf(stderr,
                "FILETOMEM_WRITE_OFFSET (%zu) exceeds file size (%zu)\n",
                write_offset, file_size);
        return EXIT_FAILURE;
    }

    // Select the GPU
    hip_err = hipSetDevice(gpu_id);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not select GPU %d (%d)\n", gpu_id, hip_err);
        return EXIT_FAILURE;
    }

    if (open_file(in_path, O_RDONLY, 0, &in_fd, &in_handle)) {
        return EXIT_FAILURE;
    }

    if (open_file(out_path, O_WRONLY | O_CREAT | O_TRUNC, S_IWUSR | S_IRUSR | S_IRGRP | S_IROTH,
                  &out_fd, &out_handle)) {
        goto close_in;
    }

    if (0 == file_size) {
        exit_status = EXIT_SUCCESS;
        goto close_out;
    }

    // Allocate device memory large enough to hold the entire file.
    // Round up to block_size so hipFile O_DIRECT reads always land on an
    // aligned boundary.
    alloc_size = align_up(file_size, block_size);
    hip_err    = hipMalloc(&devbuf, alloc_size);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not allocate %zu bytes on GPU %d (%d)\n",
                alloc_size, gpu_id, hip_err);
        goto close_out;
    }

    // Read the entire file into device memory in one shot
    file_offset = 0;
    nread       = hipFileRead(in_handle, devbuf, alloc_size, file_offset, 0);
    if (nread < 0) {
        fprintf(stderr, "Could not read from %s (%zd) (%s)\n", in_path, nread,
                IS_HIPFILE_ERR(nread) ? HIPFILE_ERRSTR(nread) : strerror(errno));
        goto free_devbuf;
    }

    // Write all bytes from write_offset onward to the output file.
    // The device-buffer offset and the file offset advance together so
    // hipFile always reads from the correct region of devbuf.
    {
        const size_t write_size = static_cast<size_t>(nread) - write_offset;

        nwrite = 0;
        while (nwrite < static_cast<ssize_t>(write_size)) {
            const size_t remaining    = write_size - static_cast<size_t>(nwrite);
            const size_t aligned_size = align_up(remaining, block_size);

            // buf_offset points into devbuf at write_offset + bytes already written
            const hoff_t buf_offset = static_cast<hoff_t>(write_offset)
                                      + static_cast<hoff_t>(nwrite);

            nbytes = hipFileWrite(out_handle, devbuf, aligned_size,
                                  /*file_offset=*/static_cast<hoff_t>(nwrite),
                                  /*buffer_offset=*/buf_offset);
            if (nbytes < 0) {
                fprintf(stderr, "Could not write to %s (%zd) (%s)\n", out_path, nbytes,
                        IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
                goto free_devbuf;
            }
            nwrite += nbytes;
        }

        // Trim the output file to the exact number of bytes from the payload
        if (-1 == ftruncate(out_fd, static_cast<off_t>(write_size))) {
            fprintf(stderr, "Could not truncate %s (%s)\n", out_path, strerror(errno));
            goto free_devbuf;
        }
    }

    exit_status = EXIT_SUCCESS;

free_devbuf:
    hip_err = hipFree(devbuf);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not free device buffer (%d)\n", hip_err);
        exit_status = EXIT_FAILURE;
    }

close_out:
    if (close_file(out_path, out_fd, out_handle)) {
        exit_status = EXIT_FAILURE;
    }

close_in:
    if (close_file(in_path, in_fd, in_handle)) {
        exit_status = EXIT_FAILURE;
    }

    return exit_status;
}
