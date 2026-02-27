// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <cstring>

#include "kpack_internal.h"

extern "C" {

// Archive opening/closing implemented in archive.cpp

kpack_error_t kpack_get_architecture_count(kpack_archive_t archive,
                                           size_t* count) {
  if (!archive || !count) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  *count = archive->gfx_arches.size();
  return KPACK_SUCCESS;
}

kpack_error_t kpack_get_architecture(kpack_archive_t archive, size_t index,
                                     const char** arch) {
  if (!archive || !arch) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  if (index >= archive->gfx_arches.size()) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  *arch = archive->gfx_arches[index].c_str();
  return KPACK_SUCCESS;
}

kpack_error_t kpack_get_binary_count(kpack_archive_t archive, size_t* count) {
  if (!archive || !count) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  *count = archive->binary_names.size();
  return KPACK_SUCCESS;
}

kpack_error_t kpack_get_binary(kpack_archive_t archive, size_t index,
                               const char** binary) {
  if (!archive || !binary) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  if (index >= archive->binary_names.size()) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  *binary = archive->binary_names[index].c_str();
  return KPACK_SUCCESS;
}

kpack_error_t kpack_get_kernel(kpack_archive_t archive, const char* binary_name,
                               const char* arch, void** kernel_data,
                               size_t* kernel_size) {
  if (!archive || !binary_name || !arch || !kernel_data || !kernel_size) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  // Lookup kernel in TOC (read-only, no lock needed)
  auto binary_it = archive->toc.find(binary_name);
  if (binary_it == archive->toc.end()) {
    return KPACK_ERROR_KERNEL_NOT_FOUND;
  }

  auto arch_it = binary_it->second.find(arch);
  if (arch_it == binary_it->second.end()) {
    return KPACK_ERROR_KERNEL_NOT_FOUND;
  }

  const kpack::KernelMetadata& km = arch_it->second;

  // Lock for kernel_cache access (decompression modifies shared buffer)
  std::lock_guard<std::mutex> lock(archive->kernel_mutex);

  // Decompress based on scheme
  kpack_error_t err;
  if (archive->compression_scheme == KPACK_COMPRESSION_NOOP) {
    err = kpack::decompress_noop(archive, km.ordinal, km.original_size);
  } else if (archive->compression_scheme == KPACK_COMPRESSION_ZSTD_PER_KERNEL) {
    err = kpack::decompress_zstd(archive, km.ordinal, km.original_size);
  } else {
    return KPACK_ERROR_NOT_IMPLEMENTED;
  }

  if (err != KPACK_SUCCESS) {
    return err;
  }

  // Allocate copy for caller (thread-safe: data copied while lock held)
  size_t size = archive->kernel_cache.size();
  void* copy = std::malloc(size);
  if (!copy) {
    return KPACK_ERROR_OUT_OF_MEMORY;
  }

  std::memcpy(copy, archive->kernel_cache.data(), size);

  *kernel_data = copy;
  *kernel_size = size;
  return KPACK_SUCCESS;
}

void kpack_free_kernel(void* kernel_data) { std::free(kernel_data); }

}  // extern "C"
