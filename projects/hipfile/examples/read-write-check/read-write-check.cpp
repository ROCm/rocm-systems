/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* read-write-check - Write a pattern to a file via hipFile, copy it to a second
 * file, then verify both files are identical via hash comparison.
 *
 * Usage: ./read-write-check CREATED COPIED [GPUID]
 *
 * CREATED receives RWC_SIZE bytes of generated test pattern.
 * COPIED  is filled by reading CREATED back through GPU memory.
 * Both files are hashed and compared; a mismatch exits non-zero.
 */

#include <hipfile.h>
#include <hip/hip_runtime_api.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/// @brief Size of the test payload in bytes. Override at compile time.
#ifndef RWC_SIZE
#define RWC_SIZE (12UL * 1024UL)
#endif

/// @brief Alignment used for O_DIRECT transfers (must be a power of two).
static constexpr size_t BLOCK_ALIGN = 4096;

// ---------------------------------------------------------------------------
// Pattern generation
// ---------------------------------------------------------------------------

/// @brief Fill buf with a deterministic test pattern.
/// @param buf  Buffer to fill.
/// @param size Number of bytes to write.
static void
fill_pattern(void *buf, size_t size)
{
    auto *p = static_cast<uint8_t *>(buf);
    for (size_t i = 0; i < size; ++i)
        p[i] = static_cast<uint8_t>(i & 0xFFU);
}

// ---------------------------------------------------------------------------
// Hashing (FNV-1a 64-bit, no external dependencies)
// ---------------------------------------------------------------------------

static constexpr uint64_t FNV1A_OFFSET = 14695981039346656037ULL;
static constexpr uint64_t FNV1A_PRIME  = 1099511628211ULL;

/// @brief Compute FNV-1a 64-bit hash of a memory buffer.
/// @param buf  Data to hash.
/// @param size Number of bytes.
/// @return Hash value.
static uint64_t
hash_buffer(const void *buf, size_t size)
{
    uint64_t    h = FNV1A_OFFSET;
    const auto *p = static_cast<const uint8_t *>(buf);
    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= FNV1A_PRIME;
    }
    return h;
}

/// @brief Read a file into a CPU buffer and return its FNV-1a hash.
/// @param path     File to hash.
/// @param size     Expected number of bytes to read.
/// @param out_hash [out] Computed hash.
/// @return zero on success, non-zero on failure.
static int
hash_file(const char *path, size_t size, uint64_t *out_hash)
{
    auto *cpu_buf = static_cast<uint8_t *>(malloc(size));
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

    ssize_t nread = read(fd, cpu_buf, size);
    close(fd);

    if (nread != static_cast<ssize_t>(size)) {
        fprintf(stderr, "hash_file: read %zd bytes from %s, expected %zu\n", nread, path, size);
        free(cpu_buf);
        return 1;
    }

    *out_hash = hash_buffer(cpu_buf, size);
    free(cpu_buf);
    return 0;
}

// ---------------------------------------------------------------------------
// hipFile helpers
// ---------------------------------------------------------------------------

/// @brief Round value up to the next multiple of align. align must be a power of 2.
static constexpr size_t
align_up(size_t value, size_t align)
{
    return (value + align - 1) & ~(align - 1);
}

/// @brief Open a file and register it with hipFile.
/// @param path   Path to the file.
/// @param flags  Flags to pass to open(2); O_DIRECT is added automatically.
/// @param mode   Mode bits for open(2) (ignored unless O_CREAT is set).
/// @param fd     [out] Resulting file descriptor.
/// @param handle [out] Resulting hipFile handle.
/// @return zero on success, non-zero on failure.
static int
open_hipfile(const char *path, int flags, mode_t mode, int *fd, hipFileHandle_t *handle)
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
/// @param path   Path (used only for error messages).
/// @param fd     File descriptor to close.
/// @param handle hipFile handle to deregister.
/// @return zero on success, non-zero on failure.
static int
close_hipfile(const char *path, int fd, hipFileHandle_t handle)
{
    hipFileHandleDeregister(handle);
    if (-1 == close(fd)) {
        fprintf(stderr, "Could not close %s (%s)\n", path, strerror(errno));
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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

    static constexpr size_t payload_size = RWC_SIZE;
    static constexpr size_t alloc_size   = align_up(payload_size, BLOCK_ALIGN);

    int             created_fd, copied_fd;
    hipFileHandle_t created_handle, copied_handle;
    void           *devbuf      = nullptr;
    uint8_t        *cpu_pattern = nullptr;
    int             exit_status = EXIT_FAILURE;
    hipError_t      hip_err;
    ssize_t         nbytes;

    hip_err = hipSetDevice(gpu_id);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not select GPU %d (%d)\n", gpu_id, hip_err);
        return EXIT_FAILURE;
    }

    // Allocate and populate the CPU-side pattern buffer
    cpu_pattern = static_cast<uint8_t *>(malloc(payload_size));
    if (!cpu_pattern) {
        fprintf(stderr, "Could not allocate CPU pattern buffer\n");
        return EXIT_FAILURE;
    }
    fill_pattern(cpu_pattern, payload_size);

    // Allocate GPU buffer (aligned to BLOCK_ALIGN for O_DIRECT)
    hip_err = hipMalloc(&devbuf, alloc_size);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not allocate %zu bytes on GPU %d (%d)\n", alloc_size, gpu_id, hip_err);
        goto free_cpu_pattern;
    }

    // -----------------------------------------------------------------------
    // Phase 1: write pattern to CREATED via hipFile
    // -----------------------------------------------------------------------

    hip_err = hipMemcpy(devbuf, cpu_pattern, payload_size, hipMemcpyHostToDevice);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "hipMemcpy to device failed (%d)\n", hip_err);
        goto free_devbuf;
    }

    if (open_hipfile(created_path, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH,
                     &created_fd, &created_handle)) {
        goto free_devbuf;
    }

    nbytes = hipFileWrite(created_handle, devbuf, alloc_size, /*file_offset=*/0,
                          /*buffer_offset=*/0);
    if (nbytes < 0) {
        fprintf(stderr, "Could not write to %s (%zd) (%s)\n", created_path, nbytes,
                IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
        goto close_created;
    }

    if (-1 == ftruncate(created_fd, static_cast<off_t>(payload_size))) {
        fprintf(stderr, "Could not truncate %s (%s)\n", created_path, strerror(errno));
        goto close_created;
    }

    if (close_hipfile(created_path, created_fd, created_handle)) {
        goto free_devbuf;
    }
    created_fd     = -1;
    created_handle = {};

    // -----------------------------------------------------------------------
    // Phase 2: read CREATED back into GPU memory, then write to COPIED
    // -----------------------------------------------------------------------

    if (open_hipfile(created_path, O_RDONLY, 0, &created_fd, &created_handle)) {
        goto free_devbuf;
    }

    nbytes = hipFileRead(created_handle, devbuf, alloc_size, /*file_offset=*/0, 0);
    if (nbytes < 0) {
        fprintf(stderr, "Could not read from %s (%zd) (%s)\n", created_path, nbytes,
                IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
        goto close_created;
    }

    if (close_hipfile(created_path, created_fd, created_handle)) {
        goto free_devbuf;
    }
    created_fd     = -1;
    created_handle = {};

    if (open_hipfile(copied_path, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH,
                     &copied_fd, &copied_handle)) {
        goto free_devbuf;
    }

    nbytes = hipFileWrite(copied_handle, devbuf, alloc_size, /*file_offset=*/0,
                          /*buffer_offset=*/0);
    if (nbytes < 0) {
        fprintf(stderr, "Could not write to %s (%zd) (%s)\n", copied_path, nbytes,
                IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
        goto close_copied;
    }

    if (-1 == ftruncate(copied_fd, static_cast<off_t>(payload_size))) {
        fprintf(stderr, "Could not truncate %s (%s)\n", copied_path, strerror(errno));
        goto close_copied;
    }

    if (close_hipfile(copied_path, copied_fd, copied_handle)) {
        goto free_devbuf;
    }
    copied_fd     = -1;
    copied_handle = {};

    // -----------------------------------------------------------------------
    // Phase 3: hash both files and compare
    // -----------------------------------------------------------------------

    {
        uint64_t hash_created, hash_copied;

        if (hash_file(created_path, payload_size, &hash_created)) {
            goto free_devbuf;
        }
        if (hash_file(copied_path, payload_size, &hash_copied)) {
            goto free_devbuf;
        }

        if (hash_created != hash_copied) {
            fprintf(stderr, "Hash mismatch: %s=0x%016llx  %s=0x%016llx\n", created_path,
                    (unsigned long long)hash_created, copied_path, (unsigned long long)hash_copied);
            goto free_devbuf;
        }

        printf("OK  %s == %s  (hash 0x%016llx)\n", created_path, copied_path,
               (unsigned long long)hash_created);
    }

    exit_status = EXIT_SUCCESS;

    // Normal unwind — files already closed above; jump straight to memory cleanup
    goto free_devbuf;

close_copied:
    close_hipfile(copied_path, copied_fd, copied_handle);

close_created:
    if (created_fd != -1)
        close_hipfile(created_path, created_fd, created_handle);

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
