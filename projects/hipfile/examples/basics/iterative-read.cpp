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

static const uint64_t FNV1A_OFFSET = 14695981039346656037ULL;
static const uint64_t FNV1A_PRIME  = 1099511628211ULL;

/// @brief Round value up to the next multiple of align. align must be a power of 2.
static inline size_t
align_up(size_t value, size_t align)
{
    return (value + align - 1) & ~(align - 1);
}

/// @brief Determine if value is a power of two.
static inline bool
is_power_of_two(size_t value)
{
    return (value > 0) && ((value & (value - 1)) == 0);
}

/// @brief Compute FNV-1a 64-bit hash of a memory buffer.
static uint64_t
hash_buffer(const void *buf, size_t size)
{
    uint64_t       h = FNV1A_OFFSET;
    const uint8_t *p = (const uint8_t *)buf;
    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= FNV1A_PRIME;
    }
    return h;
}

/// @brief Read the first `size` bytes of `path` and return their FNV-1a hash.
static int
hash_file(const char *path, size_t size, uint64_t *out_hash)
{
    uint8_t *cpu_buf = (uint8_t *)malloc(size);
    if (!cpu_buf) {
        fprintf(stderr, "hash_file: malloc failed for %zu bytes\n", size);
        return 1;
    }

    int fd = open(path, O_RDONLY);
    if (-1 == fd) {
        fprintf(stderr, "hash_file: could not open %s (%s)\n", path, strerror(errno));
        free(cpu_buf);
        return 1;
    }

    size_t total = 0;
    while (total < size) {
        ssize_t n = read(fd, cpu_buf + total, size - total);
        if (n < 0) {
            fprintf(stderr, "hash_file: read %s failed (%s)\n", path, strerror(errno));
            close(fd);
            free(cpu_buf);
            return 1;
        }
        if (n == 0)
            break;
        total += (size_t)n;
    }
    close(fd);

    if (total != size) {
        fprintf(stderr, "hash_file: short read on %s (%zu of %zu bytes)\n", path, total, size);
        free(cpu_buf);
        return 1;
    }

    *out_hash = hash_buffer(cpu_buf, size);
    free(cpu_buf);
    return 0;
}

/// @brief Open a file and register it with hipFile.
/// @param path   Path to the file.
/// @param flags  Flags to pass to open(2); O_DIRECT is added automatically.
/// @param mode   Mode bits for open(2) (used when O_CREAT is set).
/// @param fd     [out] Resulting file descriptor.
/// @param handle [out] Resulting hipFile handle.
/// @return zero on success, non-zero on failure.
static int
open_file(const char *path, int flags, mode_t mode, int *fd, hipFileHandle_t *handle)
{
    *fd = open(path, flags | O_DIRECT, mode);
    if (-1 == *fd) {
        fprintf(stderr, "Could not open %s (%s)\n", path, strerror(errno));
        return 1;
    }

    hipFileDescr_t descr;
    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = *fd;

    hipFileError_t hipfile_err = hipFileHandleRegister(handle, &descr);
    if (hipFileSuccess != hipfile_err.err) {
        fprintf(stderr, "Could not register %s (%s)\n", path, hipFileGetOpErrorString(hipfile_err.err));
        close(*fd);
        return 1;
    }

    return 0;
}

/// @brief Deregister a hipFile handle and close the underlying file descriptor.
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
    if (open_file(in_path, O_RDONLY, 0, &in_fd, &in_handle))
        return EXIT_FAILURE;

    /* 3. Allocate device buffer for the full read */
    hip_err = hipMalloc(&devbuf, alloc_size);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not allocate %zu bytes on GPU %d (%d)\n", alloc_size, gpu_id, hip_err);
        goto close_in;
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
    if (open_file(out_path, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH, &out_fd,
                  &out_handle)) {
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
        uint64_t hash_in, hash_out;

        if (hash_file(in_path, payload_size, &hash_in))
            goto close_in;
        if (hash_file(out_path, payload_size, &hash_out))
            goto close_in;

        if (hash_in != hash_out) {
            fprintf(stderr, "Hash mismatch: %s=0x%016" PRIx64 "  %s=0x%016" PRIx64 "\n", in_path, hash_in,
                    out_path, hash_out);
            goto close_in;
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
