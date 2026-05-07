// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cerrno>
#include <cstring>

#include "kpack_internal.h"

namespace kpack {

static kpack_error_t validate_header(FILE* file, uint32_t* version,
                                     uint64_t* toc_offset) {
  // Read 16-byte header
  uint8_t header[16];
  if (fread(header, 1, 16, file) != 16) {
    return KPACK_ERROR_INVALID_FORMAT;
  }

  // Validate magic
  if (memcmp(header, KPACK_MAGIC, KPACK_MAGIC_SIZE) != 0) {
    return KPACK_ERROR_INVALID_FORMAT;
  }

  // Extract version (little-endian uint32 at offset 4)
  uint32_t ver;
  memcpy(&ver, header + 4, 4);
  *version = ver;

  if (ver != KPACK_CURRENT_VERSION) {
    return KPACK_ERROR_UNSUPPORTED_VERSION;
  }

  // Extract TOC offset (little-endian uint64 at offset 8)
  uint64_t toc_off;
  memcpy(&toc_off, header + 8, 8);
  *toc_offset = toc_off;

  return KPACK_SUCCESS;
}

// Returns file size, or 0 on error (caller should validate)
static uint64_t get_file_size(FILE* file) {
  long current = ftell(file);
  if (current < 0) {
    return 0;  // ftell() failed
  }
  if (fseek(file, 0, SEEK_END) != 0) {
    return 0;  // fseek() failed
  }
  long size = ftell(file);
  if (size < 0) {
    fseek(file, current, SEEK_SET);  // Try to restore position
    return 0;                        // ftell() failed
  }
  if (fseek(file, current, SEEK_SET) != 0) {
    return 0;  // fseek() failed to restore
  }
  return static_cast<uint64_t>(size);
}

}  // namespace kpack

extern "C" {

kpack_error_t kpack_open(const char* path, kpack_archive_t* archive) {
  if (!path || !archive) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  // Open file
  FILE* file = fopen(path, "rb");
  if (!file) {
    return errno == ENOENT ? KPACK_ERROR_FILE_NOT_FOUND : KPACK_ERROR_IO_ERROR;
  }

  uint64_t file_size = kpack::get_file_size(file);
  if (file_size == 0) {
    fclose(file);
    return KPACK_ERROR_IO_ERROR;  // ftell/fseek failed or empty file
  }

  // Create archive handle with unique_ptr for automatic cleanup
  auto arch =
      std::unique_ptr<kpack_archive>(new (std::nothrow) kpack_archive());
  if (!arch) {
    fclose(file);
    return KPACK_ERROR_OUT_OF_MEMORY;
  }

  arch->file.reset(file);
  arch->file_path = path;

  // Validate header
  kpack_error_t err =
      kpack::validate_header(file, &arch->version, &arch->toc_offset);
  if (err != KPACK_SUCCESS) {
    return err;
  }

  // Parse TOC
  err = kpack::parse_toc(file, arch->toc_offset, file_size, arch.get());
  if (err != KPACK_SUCCESS) {
    return err;
  }

  // Build binary names list for enumeration
  arch->binary_names.reserve(arch->toc.size());
  for (const auto& entry : arch->toc) {
    arch->binary_names.push_back(entry.first);
  }

  // Build Zstd frame index if needed
  if (arch->compression_scheme == KPACK_COMPRESSION_ZSTD_PER_KERNEL) {
    err = kpack::build_zstd_frame_index(arch.get());
    if (err != KPACK_SUCCESS) {
      return err;
    }
  }

  *archive = arch.release();
  return KPACK_SUCCESS;
}

void kpack_close(kpack_archive_t archive) {
  // RAII handles all cleanup
  delete archive;
}

}  // extern "C"
