/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "examples_common.h"

#include <hipfile.h>

#include <cerrno>
#include <cinttypes>
#include <fcntl.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

static const uint64_t FNV1A_OFFSET = 14695981039346656037ULL;
static const uint64_t FNV1A_PRIME  = 1099511628211ULL;

void
fill_pattern(void *buf, size_t size)
{
    uint8_t *p = (uint8_t *)buf;
    for (size_t i = 0; i < size; ++i)
        p[i] = (uint8_t)(i & 0xFFU);
}

uint64_t
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

int
hash_file_range(const char *path, off_t offset, size_t size, uint64_t *out_hash)
{
    uint8_t *cpu_buf = (uint8_t *)malloc(size);
    if (!cpu_buf) {
        fprintf(stderr, "hash_file_range: malloc failed for %zu bytes\n", size);
        return 1;
    }

    int fd = open(path, O_RDONLY);
    if (-1 == fd) {
        fprintf(stderr, "hash_file_range: could not open %s (%s)\n", path, strerror(errno));
        free(cpu_buf);
        return 1;
    }

    if (offset != 0 && lseek(fd, offset, SEEK_SET) == (off_t)-1) {
        fprintf(stderr, "hash_file_range: lseek %s failed (%s)\n", path, strerror(errno));
        close(fd);
        free(cpu_buf);
        return 1;
    }

    size_t total = 0;
    while (total < size) {
        ssize_t n = read(fd, cpu_buf + total, size - total);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            fprintf(stderr, "hash_file_range: read %s failed (%s)\n", path, strerror(errno));
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
        fprintf(stderr, "hash_file_range: short read on %s (%zu of %zu bytes)\n", path, total, size);
        free(cpu_buf);
        return 1;
    }

    *out_hash = hash_buffer(cpu_buf, size);
    free(cpu_buf);
    return 0;
}

int
seed_read_file(const char *path, size_t size)
{
    uint8_t *buf = (uint8_t *)malloc(size);
    if (!buf) {
        fprintf(stderr, "seed_read_file: could not allocate %zu bytes\n", size);
        return 1;
    }
    fill_pattern(buf, size);

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (-1 == fd) {
        fprintf(stderr, "seed_read_file: could not open %s (%s)\n", path, strerror(errno));
        free(buf);
        return 1;
    }

    size_t written = 0;
    while (written < size) {
        ssize_t n = write(fd, buf + written, size - written);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            fprintf(stderr, "seed_read_file: write to %s failed (%s)\n", path, strerror(errno));
            close(fd);
            free(buf);
            return 1;
        }
        written += (size_t)n;
    }

    if (-1 == close(fd)) {
        fprintf(stderr, "seed_read_file: could not close %s (%s)\n", path, strerror(errno));
        free(buf);
        return 1;
    }

    free(buf);
    return 0;
}

int
verify_files_match(const char *read_path, const char *write_path, size_t size, uint64_t *out_hash)
{
    uint64_t hash_read, hash_write;

    if (hash_file_range(read_path, 0, size, &hash_read))
        return 1;
    if (hash_file_range(write_path, 0, size, &hash_write))
        return 1;

    if (hash_read != hash_write) {
        fprintf(stderr, "Hash mismatch: %s=0x%016" PRIx64 "  %s=0x%016" PRIx64 "\n", read_path, hash_read,
                write_path, hash_write);
        return 1;
    }

    *out_hash = hash_read;
    return 0;
}

int
open_file(const char *path, int flags, mode_t mode, int *fd, hipFileHandle_t *handle)
{
    *fd = open(path, flags, mode);
    if (-1 == *fd) {
        fprintf(stderr, "Could not open %s (%s)\n", path, strerror(errno));
        return 1;
    }

    hipFileDescr_t descr{};
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

int
close_file(const char *path, int fd, hipFileHandle_t handle)
{
    hipFileHandleDeregister(handle);
    if (-1 == close(fd)) {
        fprintf(stderr, "Could not close %s (%s)\n", path, strerror(errno));
        return 1;
    }
    return 0;
}
