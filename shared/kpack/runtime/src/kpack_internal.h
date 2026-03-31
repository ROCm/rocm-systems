// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef KPACK_INTERNAL_H
#define KPACK_INTERNAL_H

#include <zstd.h>

#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "rocm_kpack/kpack.h"
#include "rocm_kpack/kpack_types.h"

// POC NOTE: This is proof-of-concept code with intentional limitations:
// - Uses fopen/fread instead of mmap for file access
// - Caches entire Zstd blob in memory (can be large for big archives)
// - Single kernel cache that overwrites on each get_kernel() call
// - No streaming decompression or advanced caching strategies
//
// Production implementation should use:
// - mmap for efficient file access
// - LRU or arena-based kernel caching
// - Streaming decompression for large kernels
// - Better memory management overall

namespace kpack {

// Blob metadata for NoOp compression
struct BlobInfo {
  uint64_t offset;  // Absolute file offset
  uint64_t size;    // Blob size in bytes
};

// Frame metadata for Zstd compression
struct FrameInfo {
  uint64_t offset_in_blob;   // Offset within cached blob
  uint32_t compressed_size;  // Compressed frame size
};

// Per-kernel metadata from TOC
struct KernelMetadata {
  std::string type;        // "hsaco"
  uint32_t ordinal;        // Index in blobs/frames array
  uint64_t original_size;  // Decompressed size
};

// Custom deleter for FILE*
struct FileDeleter {
  void operator()(FILE* f) const {
    if (f) fclose(f);
  }
};

// Custom deleter for ZSTD_DCtx
struct ZstdContextDeleter {
  void operator()(ZSTD_DCtx* ctx) const {
    if (ctx) ZSTD_freeDCtx(ctx);
  }
};

}  // namespace kpack

// Archive handle struct - opaque to C API
struct kpack_archive {
  // File handle
  std::unique_ptr<FILE, kpack::FileDeleter> file;
  std::string file_path;

  // Header
  uint32_t version;
  uint64_t toc_offset;

  // TOC metadata
  std::string group_name;
  std::string gfx_arch_family;
  std::vector<std::string> gfx_arches;
  std::vector<std::string> binary_names;  // Cached for enumeration
  kpack_compression_scheme_t compression_scheme;

  // Nested TOC: binary_path -> arch -> metadata
  std::map<std::string, std::map<std::string, kpack::KernelMetadata>> toc;

  // Compression state (NoOp)
  std::vector<kpack::BlobInfo> blobs;

  // Compression state (Zstd)
  // POC: Cache entire blob in memory
  uint64_t zstd_offset;
  uint64_t zstd_size;
  std::vector<uint8_t> zstd_blob;  // POC: Full blob cache
  std::vector<kpack::FrameInfo> zstd_frames;
  std::unique_ptr<ZSTD_DCtx, kpack::ZstdContextDeleter> zstd_ctx;

  // Kernel cache
  // POC: Single kernel cache - overwrites on each get_kernel()
  // Thread safety: Lock kernel_mutex before accessing kernel_cache
  std::mutex kernel_mutex;
  std::vector<uint8_t> kernel_cache;
};

// Cache handle struct - opaque to C API
// Keeps archives open and env vars resolved for fast repeated access
struct kpack_cache {
  // Environment variables - resolved once at creation (thread-safe after init)
  std::vector<std::string> env_path_override;  // ROCM_KPACK_PATH (split)
  std::vector<std::string> env_path_prefix;    // ROCM_KPACK_PATH_PREFIX (split)
  bool disabled;                               // ROCM_KPACK_DISABLE
  bool debug;                                  // ROCM_KPACK_DEBUG

  // Archive cache - keeps archives open for fast repeated access
  // Key: canonical archive path, Value: opened archive handle
  std::mutex archive_mutex;
  std::unordered_map<std::string, kpack_archive_t> archives;

  // Per-archive architecture sets (derived from archive TOC at open time)
  // Used for correct arch-first search without re-querying each archive
  std::unordered_map<std::string, std::set<std::string>> archive_archs;
};

// Internal functions (defined in other translation units)
namespace kpack {

// TOC parsing (toc_parser.cpp)
kpack_error_t parse_toc(FILE* file, uint64_t toc_offset, uint64_t file_size,
                        kpack_archive* archive);

// Compression (compression.cpp)
kpack_error_t decompress_noop(kpack_archive* archive, uint32_t ordinal,
                              uint64_t expected_size);
kpack_error_t decompress_zstd(kpack_archive* archive, uint32_t ordinal,
                              uint64_t expected_size);
kpack_error_t build_zstd_frame_index(kpack_archive* archive);

}  // namespace kpack

#endif  // KPACK_INTERNAL_H
