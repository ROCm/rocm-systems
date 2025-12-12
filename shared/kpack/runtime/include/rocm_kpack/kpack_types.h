// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCM_KPACK_TYPES_H
#define ROCM_KPACK_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to an opened kpack archive
typedef struct kpack_archive* kpack_archive_t;

// Opaque handle to a kpack cache (for loader API)
typedef struct kpack_cache* kpack_cache_t;

// Error codes
typedef enum {
  KPACK_SUCCESS = 0,
  KPACK_ERROR_INVALID_ARGUMENT = 1,
  KPACK_ERROR_FILE_NOT_FOUND = 2,
  KPACK_ERROR_INVALID_FORMAT = 3,
  KPACK_ERROR_UNSUPPORTED_VERSION = 4,
  KPACK_ERROR_KERNEL_NOT_FOUND = 5,
  KPACK_ERROR_DECOMPRESSION_FAILED = 6,
  KPACK_ERROR_OUT_OF_MEMORY = 7,
  KPACK_ERROR_NOT_IMPLEMENTED = 8,
  KPACK_ERROR_IO_ERROR = 9,
  KPACK_ERROR_MSGPACK_PARSE_FAILED = 10,
  // Loader API error codes
  KPACK_ERROR_PATH_DISCOVERY_FAILED = 11,
  KPACK_ERROR_INVALID_METADATA = 12,
  KPACK_ERROR_ARCHIVE_NOT_FOUND = 13,
  KPACK_ERROR_ARCH_NOT_FOUND = 14,
} kpack_error_t;

// Kpack file format version
#define KPACK_MAGIC "KPAK"
#define KPACK_MAGIC_SIZE 4
#define KPACK_CURRENT_VERSION 1

// Compression schemes
typedef enum {
  KPACK_COMPRESSION_NOOP = 0,
  KPACK_COMPRESSION_ZSTD_PER_KERNEL = 1,
} kpack_compression_scheme_t;

#ifdef __cplusplus
}
#endif

#endif  // ROCM_KPACK_TYPES_H
