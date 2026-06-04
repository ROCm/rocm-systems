// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_CODE_HSA_CODE_OBJECT_READER_REWRITER_H_
#define ROCJITSU_CODE_HSA_CODE_OBJECT_READER_REWRITER_H_

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#endif
#include <hsa/hsa.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

/// @brief Owns replacement HSA code-object reader memory while readers exist.
///
/// HSA reader APIs do not take ownership of memory passed to
/// hsa_code_object_reader_create_from_memory(). This helper centralizes the
/// common interposer pattern: read code-object bytes, optionally rewrite them,
/// create a replacement memory reader, and retain those replacement bytes until
/// the real reader destroy succeeds.
class HsaCodeObjectReaderRewriter {
public:
  using CreateFromMemoryFn = hsa_status_t (*)(const void *, size_t, hsa_code_object_reader_t *);
  using CreateFromFileFn = hsa_status_t (*)(hsa_file_t, hsa_code_object_reader_t *);
  using CreateFromFileWithOffsetSizeFn = hsa_status_t (*)(hsa_file_t, size_t, size_t,
                                                          hsa_code_object_reader_t *);
  using DestroyFn = hsa_status_t (*)(hsa_code_object_reader_t);
  using RewriteFn = std::optional<std::vector<uint8_t>> (*)(std::span<const uint8_t>, void *);

  HsaCodeObjectReaderRewriter() = default;
  HsaCodeObjectReaderRewriter(const HsaCodeObjectReaderRewriter &) = delete;
  HsaCodeObjectReaderRewriter &operator=(const HsaCodeObjectReaderRewriter &) = delete;

  void set_api(CreateFromMemoryFn create_from_memory, DestroyFn destroy);

  hsa_status_t create_from_memory(const void *code_object, size_t size,
                                  hsa_code_object_reader_t *code_object_reader, RewriteFn rewrite,
                                  void *rewrite_data);

  hsa_status_t create_from_file(hsa_file_t file, hsa_code_object_reader_t *code_object_reader,
                                CreateFromFileFn original_create_from_file, RewriteFn rewrite,
                                void *rewrite_data);

  hsa_status_t create_from_file_with_offset_size(
      hsa_file_t file, size_t offset, size_t size, hsa_code_object_reader_t *code_object_reader,
      CreateFromFileWithOffsetSizeFn original_create_from_file_with_offset_size, RewriteFn rewrite,
      void *rewrite_data);

  hsa_status_t destroy(hsa_code_object_reader_t code_object_reader);

private:
  hsa_status_t create_reader_from_rewritten_memory(std::vector<uint8_t> rewritten,
                                                   hsa_code_object_reader_t *code_object_reader);

  CreateFromMemoryFn create_from_memory_ = nullptr;
  DestroyFn destroy_ = nullptr;
  std::mutex mutex_;
  std::unordered_map<uint64_t, std::vector<uint8_t>> rewritten_readers_;
};

bool read_hsa_file_region(hsa_file_t file, size_t offset, size_t size, std::vector<uint8_t> *bytes);
bool read_hsa_file(hsa_file_t file, std::vector<uint8_t> *bytes);

} // namespace rocjitsu

#endif // ROCJITSU_CODE_HSA_CODE_OBJECT_READER_REWRITER_H_
