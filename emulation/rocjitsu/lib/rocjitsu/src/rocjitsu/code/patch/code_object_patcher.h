// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;
struct KdTranslation;

class CodeObjectPatcher {
public:
  explicit CodeObjectPatcher(const AmdGpuCodeObject &obj);

  std::span<uint8_t> text_bytes();
  std::span<const uint8_t> text_bytes() const;

  std::span<const uint8_t> image_bytes() const { return {image_.data(), image_.size()}; }
  uint64_t text_offset() const { return text_offset_; }
  uint64_t text_size() const { return text_size_; }

  /// @brief Replace the original .text payload with DBT-relocated code.
  ///
  /// @details Updates the .text section size, shifts later file contents, grows
  /// the executable LOAD segment that contains .text, preserves LOAD alignment,
  /// updates moved symbols and relocation places, and keeps descriptor-relative
  /// entries coherent with explicit descriptor patches applied by DBT.
  ///
  /// @param max_file_growth Maximum number of bytes that this caller permits
  /// the ELF image to grow. This includes both the new text and any padding
  /// required to preserve section and segment alignment. Keeping the budget at
  /// the call site lets each transformation choose its own resource policy.
  /// @returns false, without changing this patcher, when the replacement is
  /// malformed, exceeds @p max_file_growth, or cannot be allocated.
  [[nodiscard]] bool replace_text(std::span<const uint8_t> new_text,
                                  size_t max_file_growth) noexcept;

  void update_elf_flags(uint32_t new_flags);

  [[nodiscard]] bool patch_kernel_descriptor(uint64_t file_offset,
                                             std::span<const uint8_t> descriptor);

  /// @brief Apply a descriptor translation plan to the in-memory ELF image.
  ///
  /// KernelDescriptorTranslator owns the resource/ABI decision. BinaryTranslator
  /// owns text relocation and any local prologue layout. The patcher only
  /// mutates descriptor bytes and redirects the descriptor to the already-known
  /// relocated entry offset.
  [[nodiscard]] bool apply_kernel_descriptor_translation(const KdTranslation &translation,
                                                         rj_code_arch_t target_arch);

  /// @brief Redirect one kernel descriptor from @p old_entry_text_offset to @p
  /// new_entry_text_offset.
  ///
  /// AMDHSA stores the entry as a signed KD-relative byte offset. The
  /// text-offset delta is the same delta in KD-relative coordinates, so the
  /// patcher does not need symbol virtual addresses here.
  [[nodiscard]] bool redirect_kernel_entry(uint64_t descriptor_file_offset,
                                           uint64_t old_entry_text_offset,
                                           uint64_t new_entry_text_offset);

  std::vector<uint8_t> emit() const;

private:
  [[nodiscard]] bool replace_text_impl(std::span<const uint8_t> new_text, size_t max_file_growth);

  std::vector<uint8_t> image_;
  uint64_t text_offset_;
  uint64_t text_size_;
};

} // namespace rocjitsu
