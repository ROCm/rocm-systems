// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/hsa_code_object_reader_rewriter.h"

#include <unistd.h>

#include <limits>
#include <new>

namespace rocjitsu {

namespace {

uint64_t reader_handle(hsa_code_object_reader_t reader) { return reader.handle; }

} // namespace

void HsaCodeObjectReaderRewriter::set_api(CreateFromMemoryFn create_from_memory,
                                          DestroyFn destroy) {
  create_from_memory_ = create_from_memory;
  destroy_ = destroy;
}

hsa_status_t
HsaCodeObjectReaderRewriter::create_from_memory(const void *code_object, size_t size,
                                                hsa_code_object_reader_t *code_object_reader,
                                                RewriteFn rewrite, void *rewrite_data) {
  if (create_from_memory_ == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (code_object == nullptr || size == 0 || code_object_reader == nullptr || rewrite == nullptr)
    return create_from_memory_(code_object, size, code_object_reader);

  try {
    const std::span<const uint8_t> image(static_cast<const uint8_t *>(code_object), size);
    auto rewritten = rewrite(image, rewrite_data);
    if (rewritten.has_value()) {
      const hsa_status_t status =
          create_reader_from_rewritten_memory(std::move(*rewritten), code_object_reader);
      if (status == HSA_STATUS_SUCCESS)
        return status;
    }
  } catch (const std::bad_alloc &) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  return create_from_memory_(code_object, size, code_object_reader);
}

hsa_status_t HsaCodeObjectReaderRewriter::create_from_file(
    hsa_file_t file, hsa_code_object_reader_t *code_object_reader,
    CreateFromFileFn original_create_from_file, RewriteFn rewrite, void *rewrite_data) {
  if (original_create_from_file == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (code_object_reader == nullptr || create_from_memory_ == nullptr || rewrite == nullptr)
    return original_create_from_file(file, code_object_reader);

  try {
    std::vector<uint8_t> image;
    if (read_hsa_file(file, &image)) {
      auto rewritten = rewrite(image, rewrite_data);
      if (rewritten.has_value()) {
        const hsa_status_t status =
            create_reader_from_rewritten_memory(std::move(*rewritten), code_object_reader);
        if (status == HSA_STATUS_SUCCESS)
          return status;
      }
    }
  } catch (const std::bad_alloc &) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  return original_create_from_file(file, code_object_reader);
}

hsa_status_t HsaCodeObjectReaderRewriter::create_from_file_with_offset_size(
    hsa_file_t file, size_t offset, size_t size, hsa_code_object_reader_t *code_object_reader,
    CreateFromFileWithOffsetSizeFn original_create_from_file_with_offset_size, RewriteFn rewrite,
    void *rewrite_data) {
  if (original_create_from_file_with_offset_size == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (code_object_reader == nullptr || create_from_memory_ == nullptr || rewrite == nullptr) {
    return original_create_from_file_with_offset_size(file, offset, size, code_object_reader);
  }

  try {
    std::vector<uint8_t> image;
    if (read_hsa_file_region(file, offset, size, &image)) {
      auto rewritten = rewrite(image, rewrite_data);
      if (rewritten.has_value()) {
        const hsa_status_t status =
            create_reader_from_rewritten_memory(std::move(*rewritten), code_object_reader);
        if (status == HSA_STATUS_SUCCESS)
          return status;
      }
    }
  } catch (const std::bad_alloc &) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  return original_create_from_file_with_offset_size(file, offset, size, code_object_reader);
}

hsa_status_t HsaCodeObjectReaderRewriter::destroy(hsa_code_object_reader_t code_object_reader) {
  if (destroy_ == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;

  const hsa_status_t status = destroy_(code_object_reader);
  if (status == HSA_STATUS_SUCCESS) {
    std::lock_guard<std::mutex> lock(mutex_);
    rewritten_readers_.erase(reader_handle(code_object_reader));
  }
  return status;
}

hsa_status_t HsaCodeObjectReaderRewriter::create_reader_from_rewritten_memory(
    std::vector<uint8_t> rewritten, hsa_code_object_reader_t *code_object_reader) {
  hsa_code_object_reader_t reader{};
  const hsa_status_t status = create_from_memory_(rewritten.data(), rewritten.size(), &reader);
  if (status != HSA_STATUS_SUCCESS)
    return status;

  try {
    std::lock_guard<std::mutex> lock(mutex_);
    rewritten_readers_[reader_handle(reader)] = std::move(rewritten);
  } catch (const std::bad_alloc &) {
    (void)destroy_(reader);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  *code_object_reader = reader;
  return HSA_STATUS_SUCCESS;
}

bool read_hsa_file_region(hsa_file_t file, size_t offset, size_t size,
                          std::vector<uint8_t> *bytes) {
  if (bytes == nullptr || size == 0)
    return false;

  bytes->assign(size, 0);
  size_t done = 0;
  while (done < size) {
    if (offset > static_cast<size_t>(std::numeric_limits<off_t>::max()) - done)
      return false;
    const ssize_t nread =
        pread(file, bytes->data() + done, size - done, static_cast<off_t>(offset + done));
    if (nread <= 0)
      return false;
    done += static_cast<size_t>(nread);
  }
  return true;
}

bool read_hsa_file(hsa_file_t file, std::vector<uint8_t> *bytes) {
  const off_t saved = lseek(file, 0, SEEK_CUR);
  const off_t end = lseek(file, 0, SEEK_END);
  if (saved >= 0)
    (void)lseek(file, saved, SEEK_SET);
  if (end <= 0)
    return false;
  return read_hsa_file_region(file, 0, static_cast<size_t>(end), bytes);
}

} // namespace rocjitsu
