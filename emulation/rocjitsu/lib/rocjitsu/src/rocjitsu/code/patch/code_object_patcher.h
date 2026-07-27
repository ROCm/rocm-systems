// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;
struct KdTranslation;

/// @brief Location of a sidecar kernel descriptor appended into a loaded ELF segment.
struct AppendedSidecarDescriptor {
  uint64_t file_offset = 0;
  uint64_t vaddr = 0;
};

/// Resource-policy details for one transactional .text replacement attempt.
///
/// `required_file_growth` is populated after ELF alignment requirements have
/// been resolved, even when the caller's limit rejects the transaction. It is
/// therefore the exact inserted byte count, including alignment padding.
struct TextReplacementInfo {
  std::optional<size_t> required_file_growth;
  bool file_growth_limit_exceeded = false;
};

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
  /// @param info Optional exact resource-policy details. This is reset before
  /// the attempt and identifies a file-growth-limit rejection separately from
  /// malformed input or allocation failure.
  /// @returns false, without changing this patcher, when the replacement is
  /// malformed, exceeds @p max_file_growth, or cannot be allocated.
  [[nodiscard]] bool replace_text(std::span<const uint8_t> new_text, size_t max_file_growth,
                                  TextReplacementInfo *info = nullptr) noexcept;

  /// @brief True if any relocation's place (r_offset) falls inside .text.
  ///
  /// @details DBT compacts/expands/moves instructions within .text but does not
  /// remap relocation places that land in .text — replace_text() only shifts
  /// relocation offsets for whole sections moved *after* .text. An in-.text
  /// relocation would therefore be applied to the wrong translated bytes.
  /// BinaryTranslator uses this to fail closed instead of miscompiling. Real
  /// AMDHSA kernel code objects carry no such relocations, so this rejects only
  /// genuinely unsupported inputs.
  [[nodiscard]] bool has_relocations_within_text() const;

  /// @brief True if any relocation resolves against a location inside .text.
  ///
  /// @details DBT moves (and can duplicate) .text blocks, but the symbol values
  /// (st_value) of anything defined in .text are not remapped. A relocation
  /// elsewhere (e.g. a function-pointer table in .data) that resolves against
  /// such a location would therefore point at its stale pre-move PC. This covers
  /// every symbol whose st_shndx is the text section regardless of type —
  /// STT_FUNC helpers, STT_NOTYPE labels, and an STT_SECTION symbol for .text
  /// (whose addend selects an in-.text offset) all alias moved code. Kernel entry
  /// points are dispatched through the descriptor's kernel_code_entry_byte_offset
  /// (which DBT does update) and are not the target of in-object relocations, so
  /// this rejects only genuinely address-taken text locations that cannot be
  /// relocated safely yet. BinaryTranslator uses it to fail closed instead of
  /// resolving to a wrong address.
  [[nodiscard]] bool has_relocation_to_text_symbol() const;

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

  /// @brief Append non-symbolized sidecar descriptors in loaded memory after .text.
  ///
  /// @details Sidecar descriptors must be visible to the GPU loader, but they
  /// do not need to be discoverable through the AMDHSA symbol table. The patcher
  /// therefore places descriptor bytes in the executable LOAD tail immediately
  /// after the translated .text payload, without growing the .text section
  /// itself. Runtime metadata records the returned virtual addresses.
  [[nodiscard]] std::optional<std::vector<AppendedSidecarDescriptor>>
  append_sidecar_descriptor_translations(std::span<const KdTranslation> translations,
                                         rj_code_arch_t target_arch, uint64_t alignment = 64);

  /// @brief Append a named, non-allocated ELF section without moving loadable bytes.
  ///
  /// @details DBT runtime metadata must not perturb code-object load addresses:
  /// ROCR and the kernel descriptor ABI have already consumed those addresses.
  /// This helper appends the payload, a copied section-string table, and a new
  /// section-header table at EOF. Program headers and allocated sections are
  /// left untouched, so the new section is available to tools and rocjitsu's
  /// own loader-side metadata parser but is not mapped into GPU code memory.
  [[nodiscard]] bool append_nonalloc_section(std::string_view name,
                                             std::span<const uint8_t> contents,
                                             uint64_t alignment = 1);

  /// @brief Redirect one kernel descriptor from @p old_entry_text_offset to @p
  /// new_entry_text_offset.
  ///
  /// AMDHSA stores the entry as a signed KD-relative byte offset. The
  /// text-offset delta is the same delta in KD-relative coordinates, so the
  /// patcher does not need symbol virtual addresses here.
  [[nodiscard]] bool redirect_kernel_entry(uint64_t descriptor_file_offset,
                                           uint64_t old_entry_text_offset,
                                           uint64_t new_entry_text_offset);

  std::vector<uint8_t> emit() const &;
  std::vector<uint8_t> emit() &&;

private:
  [[nodiscard]] bool replace_text_impl(std::span<const uint8_t> new_text, size_t max_file_growth,
                                       TextReplacementInfo *info);

  std::vector<uint8_t> image_;
  uint64_t text_offset_;
  uint64_t text_size_;
  uint64_t text_vaddr_;
  uint64_t text_tail_size_;
};

} // namespace rocjitsu
