/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* Shared helpers for the basics/ and async/ examples. Pulled out of the
 * per-example .cpp files to remove verbatim duplication; each example still
 * drives the hipFile API directly in its main() so the example flow stays
 * readable top-to-bottom. */

#pragma once

#include <cstddef>
#include <cstdint>
#include <sys/types.h>

/* Forward-declared from <hipfile.h> to keep this header free of the hipFile
 * include. Must stay in sync with the typedef in hipfile.h. */
typedef void *hipFileHandle_t;

/// @brief Determines if value is a power of two.
/// @param value [in] The value to inspect.
/// @return True if value is a power of two, false otherwise.
constexpr bool
is_power_of_two(size_t value)
{
    return (value > 0) && ((value & (value - 1)) == 0);
}

/// @brief Alignment used for O_DIRECT transfers. Must be a power of two and a
/// multiple of the underlying filesystem's logical block size; 4 KiB satisfies
/// both for the ext4/xfs setups hipFile supports today.
#define BLOCK_ALIGN ((size_t)4096)
static_assert(is_power_of_two(BLOCK_ALIGN), "BLOCK_ALIGN must be a power of two");

/// @brief Round value up to the next multiple of align. Align _must_ be a power of 2.
/// @param value [in] The value to round up.
/// @param align [in] Value will be rounded up to a multiple of align.
/// @return Value rounded up to a multiple of align.
inline size_t
align_up(size_t value, size_t align)
{
    return (value + align - 1) & ~(align - 1);
}

/// @brief Fill a buffer with a deterministic test pattern (byte i = i & 0xFF).
/// Used so writers can emit known content and verifiers can recompute the
/// expected hash without keeping the input around.
/// @param buf  [out] Buffer to fill.
/// @param size [in]  Number of bytes to write into buf.
void fill_pattern(void *buf, size_t size);

/// @brief Compute the FNV-1a 64-bit hash of a memory buffer.
/// @param buf  [in] Pointer to the bytes to hash.
/// @param size [in] Number of bytes at buf to feed into the hash.
/// @return The 64-bit FNV-1a hash of the size bytes at buf.
uint64_t hash_buffer(const void *buf, size_t size);

/// @brief Read size bytes of path starting at byte offset, then FNV-1a hash them.
/// Reads via plain POSIX I/O (no O_DIRECT), so this is the host-side reference
/// path for verifying what hipFile wrote out.
/// @param path     [in]  Path to the file to read.
/// @param offset   [in]  Byte offset within the file to start reading from.
/// @param size     [in]  Number of bytes to read and hash.
/// @param out_hash [out] Receives the FNV-1a hash on success; untouched on failure.
/// @return zero on success, non-zero on failure.
int hash_file_range(const char *path, off_t offset, size_t size, uint64_t *out_hash);

/// @brief Seed a file with size bytes of fill_pattern via plain POSIX I/O
/// (no O_DIRECT). Useful for examples that need a known input file to read
/// back through the hipFile path. The file is created/truncated.
/// @param path [in] Path to the file to create.
/// @param size [in] Number of bytes of pattern to write.
/// @return zero on success, non-zero on failure.
int seed_read_file(const char *path, size_t size);

/// @brief Hash the first size bytes of two files (FNV-1a) and compare.
/// On match, *out_hash receives the shared hash. On mismatch (or any I/O
/// error), an error has already been printed and out_hash is left untouched.
/// @param read_path  [in]  Path to the first file.
/// @param write_path [in]  Path to the second file.
/// @param size       [in]  Number of bytes from each file to hash.
/// @param out_hash   [out] Receives the matching hash on success.
/// @return zero on match, non-zero on mismatch or read failure.
int verify_files_match(const char *read_path, const char *write_path, size_t size, uint64_t *out_hash);

/// @brief Open a file and register it with hipFile.
/// The caller controls all open(2) flags. Pass O_DIRECT to take hipFile's
/// GPU-direct fast path; omit it to route through the POSIX-IO compat path.
/// @param path   [in]  Path to the file.
/// @param flags  [in]  Flags to pass to open(2). O_DIRECT is _not_ added for you.
/// @param mode   [in]  Mode to pass to open(2) (used when O_CREAT is in flags).
/// @param fd     [out] The file descriptor of the opened file.
/// @param handle [out] The handle to use with hipFile APIs.
/// @return zero on success, non-zero on failure.
int open_file(const char *path, int flags, mode_t mode, int *fd, hipFileHandle_t *handle);

/// @brief Unregister and close a file previously opened with open_file.
/// @param path   [in] Path to the file (used only for error messages).
/// @param fd     [in] The file descriptor of the opened file.
/// @param handle [in] The handle of the opened file.
/// @return zero on success, non-zero on failure.
int close_file(const char *path, int fd, hipFileHandle_t handle);
