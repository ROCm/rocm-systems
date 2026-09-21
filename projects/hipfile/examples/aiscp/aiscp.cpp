/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* aiscp - AMD Infinity Storage Copy
 *
 * Usage: ./aiscp SOURCE DEST
 *
 * This _very_basic_ program copies SOURCE to DEST via GPU memory.
 */

#include <hipfile.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/// @brief AISCP_CHUNK_SIZE defines the maximum size of each aiscp IO operation.
/// Currently AISCP_CHUNK_SIZE is defined to match the maximum IO size on Linux
/// (MAX_RW_COUNT).
#ifndef AISCP_CHUNK_SIZE
#define AISCP_CHUNK_SIZE 0x7ffff000LU
#endif

/// @brief Open and register a file
/// @param path [in] Path to the file
/// @param flags [in] flags: Flags to pass to open (2)
/// @param mode [in] mode: Mode to pass to open (2)
/// @param fd [out] fd: The file descriptor of the opened file
/// @param handle [out] handle: The handle to use with hipFile APIs
/// @return zero on success, non-zero on failure
static int
open_file(const char *path, int flags, mode_t mode, int *fd, hipFileHandle_t *handle)
{
    hipFileError_t hipfile_err;
    hipFileDescr_t descr;

    *fd = open(path, flags | O_DIRECT, mode);
    if (-1 == *fd) {
        fprintf(stderr, "Could not open %s (%s)\n", path, strerror(errno));
        return 1;
    }

    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = *fd;

    hipfile_err = hipFileHandleRegister(handle, &descr);
    if (hipFileSuccess != hipfile_err.err) {
        fprintf(stderr, "Could not register %s (%s)\n", path, hipFileGetOpErrorString(hipfile_err.err));
        close(*fd);
        return 1;
    }

    return 0;
}

/// @brief Unregister and close a file
/// @param path [in] Path to the file
/// @param fd [in] The file descriptor of the opened file
/// @param handle [in] The handle of the opened file
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

/// @brief Round value to the next multiple of align. Align _must_ be a power of 2.
/// @param value The value to round up.
/// @param align Value will be rounded up to a multiple of align
/// @return Value rounded up to a multiple of align.
static inline size_t
align_up(size_t value, size_t align)
{
    return (value + align - 1) & ~(align - 1);
}

/// @brief Determines if value is a power of two
/// @param value The value to inspect
/// @return True if value is a power of two, false otherwise
static inline bool
is_power_of_two(size_t value)
{
    return (value > 0) && ((value & (value - 1)) == 0);
}

int
main(int argc, char *argv[])
{
    const char     *src_path, *dst_path;
    int             src_fd, dst_fd;
    hipFileHandle_t src_handle, dst_handle;
    void           *devbuf;
    hipError_t      hip_err;
    int             exit_status = EXIT_FAILURE;
    size_t          buffer_size, file_size, block_size;
    ssize_t         nwrite{}, nread{}, nbytes{};
    hoff_t          file_offset{};

    if (argc != 3) {
        fprintf(stderr, "Usage: %s SOURCE DEST\n", argv[0]);
        exit(1);
    }

    src_path = argv[1];
    dst_path = argv[2];

    {
        struct stat statbuf;
        if (stat(src_path, &statbuf)) {
            fprintf(stderr, "Could not stat file: %s (%s)\n", src_path, strerror(errno));
            goto program_exit;
        }
        file_size  = static_cast<size_t>(statbuf.st_size);
        block_size = static_cast<size_t>(statbuf.st_blksize);
        if (!is_power_of_two(block_size)) {
            fprintf(stderr, "Blocksize is not a power of two (%zu)", block_size);
            goto program_exit;
        }
    }

    if (open_file(dst_path, O_WRONLY | O_CREAT, S_IWUSR | S_IRUSR | S_IRGRP | S_IROTH, &dst_fd,
                  &dst_handle)) {
        goto program_exit;
    }

    if (0 == file_size) {
        exit_status = EXIT_SUCCESS;
        goto close_dst;
    }

    if (open_file(src_path, O_RDONLY, 0, &src_fd, &src_handle)) {
        goto close_dst;
    }

    buffer_size = align_up(std::min(file_size, AISCP_CHUNK_SIZE), block_size);
    hip_err     = hipMalloc(&devbuf, buffer_size);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not allocate device buffer (%d)", hip_err);
        goto close_src;
    }

    // Copy the file chunk-by-chunk until we hit EOF
    do {
        nread = hipFileRead(src_handle, devbuf, buffer_size, file_offset, 0);
        if (nread < 0) {
            fprintf(stderr, "Could not read from %s (%zd) (%s)\n", src_path, nread,
                    IS_HIPFILE_ERR(nread) ? HIPFILE_ERRSTR(nread) : strerror(errno));
            goto free_devbuf;
        }

        nwrite = 0;
        while (nwrite < nread) {
            nbytes =
                hipFileWrite(dst_handle, devbuf, align_up(static_cast<size_t>(nread - nwrite), block_size),
                             file_offset + nwrite, nwrite);
            if (nbytes < 0) {
                fprintf(stderr, "Could not write to %s (%zd) (%s)\n", dst_path, nbytes,
                        IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
                goto free_devbuf;
            }
            nwrite += nbytes;
        }
        file_offset += nread;
    } while (nread > 0);

    if (-1 == ftruncate(dst_fd, static_cast<off_t>(file_size))) {
        fprintf(stderr, "Could not truncate %s (%zu) (%s)\n", dst_path, file_size, strerror(errno));
    }

    exit_status = EXIT_SUCCESS;

free_devbuf:
    hip_err = hipFree(devbuf);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could free device buffer (%d)\n", hip_err);
        exit_status = EXIT_FAILURE;
    }

close_src:
    if (close_file(src_path, src_fd, src_handle)) {
        exit_status = EXIT_FAILURE;
    }

close_dst:
    if (close_file(dst_path, dst_fd, dst_handle)) {
        exit_status = EXIT_FAILURE;
    }

program_exit:
    return exit_status;
}
