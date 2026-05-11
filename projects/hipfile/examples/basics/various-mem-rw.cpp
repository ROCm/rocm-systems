/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* various-mem-rw - Round-trip a file through hipFile using one of three
 * memory types for the transfer buffer: plain device memory, managed
 * memory, or pinned host memory.
 *
 * Usage: ./various-mem-rw INPUT OUTPUT MODE [GPUID]
 *
 *   INPUT    Existing file to read. Up to VMR_CAP (default 1 MiB) is
 *            consumed. Create with:
 *              dd if=/dev/urandom of=input.bin bs=1M count=1
 *   OUTPUT   Path to the output file. Created/truncated. Receives a copy of
 *            the bytes read from INPUT.
 *   MODE     Memory type used for the transfer buffer:
 *              1 = device memory      (hipMalloc)
 *              2 = managed memory     (hipMallocManaged - no BufRegister needed)
 *              3 = pinned host memory (hipHostMalloc)
 *   GPUID    GPU device index (optional, default 0).
 *
 * Steps:
 *   1. Select GPU
 *   2. Open + register input file
 *   3. Allocate buffer of the chosen memory type + zero it
 *   4. hipFileRead into buffer
 *   5. Open + register output file + hipFileWrite + ftruncate
 *   6. Hash verify
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
#ifndef VMR_CAP
#define VMR_CAP (1024UL * 1024UL)
#endif

typedef enum {
    MEM_DEVICE  = 1,
    MEM_MANAGED = 2,
    MEM_PINNED  = 3,
} mem_mode_t;

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

/// @brief Human-readable label for a memory mode.
static const char *
mode_name(mem_mode_t mode)
{
    switch (mode) {
        case MEM_DEVICE:
            return "device";
        case MEM_MANAGED:
            return "managed";
        case MEM_PINNED:
            return "pinned-host";
        default:
            return "unknown";
    }
}

/// @brief Allocate `size` bytes in the memory backing chosen by `mode`.
static int
alloc_buf(mem_mode_t mode, size_t size, void **out_buf)
{
    hipError_t hip_err;
    switch (mode) {
        case MEM_DEVICE:
            hip_err = hipMalloc(out_buf, size);
            break;
        case MEM_MANAGED:
            hip_err = hipMallocManaged(out_buf, size, hipMemAttachGlobal);
            break;
        case MEM_PINNED:
            hip_err = hipHostMalloc(out_buf, size, hipHostMallocDefault);
            break;
        default:
            fprintf(stderr, "alloc_buf: invalid mode %d\n", (int)mode);
            return 1;
    }
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not allocate %zu bytes (%s, hip err %d)\n", size, mode_name(mode), hip_err);
        return 1;
    }
    return 0;
}

/// @brief Free a buffer previously returned by alloc_buf.
static int
free_buf(mem_mode_t mode, void *buf)
{
    hipError_t hip_err;
    switch (mode) {
        case MEM_DEVICE:
        case MEM_MANAGED:
            hip_err = hipFree(buf);
            break;
        case MEM_PINNED:
            hip_err = hipHostFree(buf);
            break;
        default:
            fprintf(stderr, "free_buf: invalid mode %d\n", (int)mode);
            return 1;
    }
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not free buffer (%s, hip err %d)\n", mode_name(mode), hip_err);
        return 1;
    }
    return 0;
}

/// @brief Zero a buffer previously returned by alloc_buf.
static int
zero_buf(mem_mode_t mode, void *buf, size_t size)
{
    if (mode == MEM_PINNED) {
        memset(buf, 0, size);
        return 0;
    }
    hipError_t hip_err = hipMemset(buf, 0, size);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not zero buffer (%s, hip err %d)\n", mode_name(mode), hip_err);
        return 1;
    }
    return 0;
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
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "Usage: %s INPUT OUTPUT MODE [GPUID]\n", argv[0]);
        fprintf(stderr, "  MODE: 1=device, 2=managed, 3=pinned-host\n");
        return EXIT_FAILURE;
    }

    const char *in_path  = argv[1];
    const char *out_path = argv[2];
    const int   mode_raw = atoi(argv[3]);
    const int   gpu_id   = (argc == 5) ? atoi(argv[4]) : 0;

    if (mode_raw < 1 || mode_raw > 3) {
        fprintf(stderr, "MODE must be 1 (device), 2 (managed), or 3 (pinned-host)\n");
        return EXIT_FAILURE;
    }
    const mem_mode_t mode = (mem_mode_t)mode_raw;

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

    const size_t payload_size = (file_size < VMR_CAP) ? file_size : VMR_CAP;
    const size_t alloc_size   = align_up(payload_size, block_size);

    int             in_fd  = -1;
    int             out_fd = -1;
    hipFileHandle_t in_handle, out_handle;
    void           *buf         = NULL;
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

    /* 3. Allocate buffer of the chosen memory type + zero it */
    if (alloc_buf(mode, alloc_size, &buf))
        goto close_in;

    if (zero_buf(mode, buf, alloc_size))
        goto release_buf;

    /* 4. hipFileRead into buffer (single call) */
    nbytes = hipFileRead(in_handle, buf, alloc_size, /*file_offset=*/0, /*buf_offset=*/0);
    if (nbytes < 0) {
        fprintf(stderr, "Could not read from %s (%zd) (%s)\n", in_path, nbytes,
                IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
        goto release_buf;
    }
    if ((size_t)nbytes < payload_size) {
        fprintf(stderr, "Short read on %s: got %zd bytes, expected at least %zu\n", in_path, nbytes,
                payload_size);
        goto release_buf;
    }

    /* 5. Open + register output file + hipFileWrite + ftruncate */
    if (open_file(out_path, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH, &out_fd,
                  &out_handle)) {
        goto release_buf;
    }

    nbytes = hipFileWrite(out_handle, buf, alloc_size, /*file_offset=*/0, /*buf_offset=*/0);
    if (nbytes < 0) {
        fprintf(stderr, "Could not write to %s (%zd) (%s)\n", out_path, nbytes,
                IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
        goto close_out;
    }

    if (-1 == ftruncate(out_fd, (off_t)payload_size)) {
        fprintf(stderr, "Could not truncate %s (%s)\n", out_path, strerror(errno));
        goto close_out;
    }

    if (close_file(out_path, out_fd, out_handle)) {
        out_fd = -1;
        goto release_buf;
    }
    out_fd = -1;

    /* 6. Hash verify */
    {
        uint64_t hash_in, hash_out;

        if (hash_file(in_path, payload_size, &hash_in))
            goto release_buf;
        if (hash_file(out_path, payload_size, &hash_out))
            goto release_buf;

        if (hash_in != hash_out) {
            fprintf(stderr, "Hash mismatch (%s mode): %s=0x%016" PRIx64 "  %s=0x%016" PRIx64 "\n",
                    mode_name(mode), in_path, hash_in, out_path, hash_out);
            goto release_buf;
        }

        printf("OK  [%s] %s -> %s  (%zu bytes, hash 0x%016" PRIx64 ")\n", mode_name(mode), in_path, out_path,
               payload_size, hash_in);
    }

    exit_status = EXIT_SUCCESS;

close_out:
    if (out_fd != -1) {
        if (close_file(out_path, out_fd, out_handle))
            exit_status = EXIT_FAILURE;
    }

release_buf:
    if (free_buf(mode, buf))
        exit_status = EXIT_FAILURE;

close_in:
    if (close_file(in_path, in_fd, in_handle))
        exit_status = EXIT_FAILURE;

    return exit_status;
}
