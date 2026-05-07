// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>

#include "kpack_internal.h"

namespace kpack {

kpack_error_t decompress_noop(kpack_archive* archive, uint32_t ordinal,
                              uint64_t expected_size) {
  if (ordinal >= archive->blobs.size()) {
    return KPACK_ERROR_KERNEL_NOT_FOUND;
  }

  const BlobInfo& blob = archive->blobs[ordinal];

  // Allocate kernel cache
  archive->kernel_cache.resize(blob.size);

  // Seek to blob offset
  if (fseek(archive->file.get(), blob.offset, SEEK_SET) != 0) {
    return KPACK_ERROR_IO_ERROR;
  }

  // Read kernel data
  size_t read =
      fread(archive->kernel_cache.data(), 1, blob.size, archive->file.get());
  if (read != blob.size) {
    return KPACK_ERROR_IO_ERROR;
  }

  return KPACK_SUCCESS;
}

kpack_error_t build_zstd_frame_index(kpack_archive* archive) {
  // Validate blob size is reasonable before allocating
  // Max 4GB per blob (arbitrary but prevents obvious attacks)
  constexpr uint64_t MAX_BLOB_SIZE = 4ULL * 1024 * 1024 * 1024;
  if (archive->zstd_size > MAX_BLOB_SIZE) {
    return KPACK_ERROR_INVALID_FORMAT;
  }

  // POC: Cache entire blob in memory
  archive->zstd_blob.resize(archive->zstd_size);

  // Seek to blob start
  if (fseek(archive->file.get(), archive->zstd_offset, SEEK_SET) != 0) {
    return KPACK_ERROR_IO_ERROR;
  }

  // Read blob
  size_t read = fread(archive->zstd_blob.data(), 1, archive->zstd_size,
                      archive->file.get());
  if (read != archive->zstd_size) {
    return KPACK_ERROR_IO_ERROR;
  }

  // Validate minimum blob size for header
  if (archive->zstd_size < sizeof(uint32_t)) {
    return KPACK_ERROR_INVALID_FORMAT;
  }

  // Parse blob header
  const uint8_t* ptr = archive->zstd_blob.data();
  const uint8_t* end = ptr + archive->zstd_size;

  uint32_t num_kernels;
  memcpy(&num_kernels, ptr, sizeof(uint32_t));
  ptr += sizeof(uint32_t);

  // Validate kernel count is reasonable
  // Each frame needs at least 4 bytes for size header
  constexpr uint32_t MAX_KERNELS = 1024 * 1024;  // 1M kernels max
  if (num_kernels > MAX_KERNELS) {
    return KPACK_ERROR_INVALID_FORMAT;
  }

  uint64_t offset = sizeof(uint32_t);  // Start after num_kernels header
  archive->zstd_frames.reserve(num_kernels);

  // Parse frame headers with bounds checking
  for (uint32_t i = 0; i < num_kernels; ++i) {
    // Check bounds for frame size header
    if (ptr + sizeof(uint32_t) > end) {
      return KPACK_ERROR_INVALID_FORMAT;
    }

    uint32_t frame_size;
    memcpy(&frame_size, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    offset += sizeof(uint32_t);

    // Check bounds for frame data
    if (ptr + frame_size > end) {
      return KPACK_ERROR_INVALID_FORMAT;
    }

    FrameInfo frame;
    frame.offset_in_blob = offset;
    frame.compressed_size = frame_size;
    archive->zstd_frames.push_back(frame);

    ptr += frame_size;
    offset += frame_size;
  }

  // Create decompression context
  archive->zstd_ctx.reset(ZSTD_createDCtx());
  if (!archive->zstd_ctx) {
    return KPACK_ERROR_OUT_OF_MEMORY;
  }

  return KPACK_SUCCESS;
}

kpack_error_t decompress_zstd(kpack_archive* archive, uint32_t ordinal,
                              uint64_t expected_size) {
  if (ordinal >= archive->zstd_frames.size()) {
    return KPACK_ERROR_KERNEL_NOT_FOUND;
  }

  const FrameInfo& frame = archive->zstd_frames[ordinal];

  // Get compressed frame from cached blob
  const void* compressed = archive->zstd_blob.data() + frame.offset_in_blob;
  size_t compressed_size = frame.compressed_size;

  // Allocate decompression buffer
  archive->kernel_cache.resize(expected_size);

  // Decompress
  size_t result =
      ZSTD_decompressDCtx(archive->zstd_ctx.get(), archive->kernel_cache.data(),
                          expected_size, compressed, compressed_size);

  if (ZSTD_isError(result)) {
    return KPACK_ERROR_DECOMPRESSION_FAILED;
  }

  if (result != expected_size) {
    return KPACK_ERROR_DECOMPRESSION_FAILED;
  }

  return KPACK_SUCCESS;
}

}  // namespace kpack
