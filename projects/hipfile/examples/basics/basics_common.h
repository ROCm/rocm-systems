/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* Shared helpers for the basics/ examples. Pulled out of the per-example
 * .cpp files to remove verbatim duplication; each example still drives the
 * hipFile API directly in its main() so the example flow stays readable
 * top-to-bottom. */

#pragma once

#include <hipfile.h>

#include <cstddef>
#include <cstdint>
#include <sys/types.h>

/// @brief Round value up to the next multiple of align. align must be a power of 2.
inline size_t
align_up(size_t value, size_t align)
{
    return (value + align - 1) & ~(align - 1);
}

/// @brief Determine if value is a power of two.
inline bool
is_power_of_two(size_t value)
{
    return (value > 0) && ((value & (value - 1)) == 0);
}

/// @brief Fill buf with a deterministic test pattern (byte i = i & 0xFF).
void fill_pattern(void *buf, size_t size);

/// @brief Compute FNV-1a 64-bit hash of a memory buffer.
uint64_t hash_buffer(const void *buf, size_t size);

/// @brief Read the first `size` bytes of `path` and return their FNV-1a hash.
/// @return zero on success, non-zero on failure.
int hash_file(const char *path, size_t size, uint64_t *out_hash);

/// @brief Read `size` bytes of `path` starting at byte `offset` and return their FNV-1a hash.
/// @return zero on success, non-zero on failure.
int hash_file_range(const char *path, off_t offset, size_t size, uint64_t *out_hash);

/// @brief Open a file with O_DIRECT and register it with hipFile.
/// @param path   Path to the file.
/// @param flags  Flags to pass to open(2); O_DIRECT is added automatically.
/// @param mode   Mode bits for open(2) (used when O_CREAT is set).
/// @param fd     [out] Resulting file descriptor.
/// @param handle [out] Resulting hipFile handle.
/// @return zero on success, non-zero on failure.
int open_file(const char *path, int flags, mode_t mode, int *fd, hipFileHandle_t *handle);

/// @brief Open a file (caller-controlled flags, no O_DIRECT added) and register it with hipFile.
/// Routes hipFile through its POSIX-IO compat path.
/// @return zero on success, non-zero on failure.
int open_file_no_odirect(const char *path, int flags, mode_t mode, int *fd, hipFileHandle_t *handle);

/// @brief Deregister a hipFile handle and close the underlying file descriptor.
/// @return zero on success, non-zero on failure.
int close_file(const char *path, int fd, hipFileHandle_t handle);
