// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;
struct KdTranslation;

/// @brief One exact source-to-target offset mapping inside `.text`.
///
/// @details DBT supplies instruction starts and block ends after final kernel
/// placement. The ELF patcher uses these mappings to keep local labels and
/// function symbols attached to relocated code without interpreting the ISA.
struct TextOffsetRelocation {
  uint64_t source_offset = 0;
  uint64_t target_offset = 0;
};

/// @brief One relocated literal64 PC builder whose target is outside `.text`.
struct PcRelativeDataRelocation {
  uint64_t target_getpc_offset = 0;
  uint64_t target_literal_offset = 0;
  uint64_t source_target_vaddr = 0;
};

/// @brief Location of a sidecar kernel descriptor appended into a loaded ELF segment.
struct AppendedSidecarDescriptor {
  uint64_t file_offset = 0;
  uint64_t vaddr = 0;
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
  [[nodiscard]] bool replace_text(std::span<const uint8_t> new_text,
                                  std::span<const TextOffsetRelocation> text_relocations = {},
                                  std::span<const PcRelativeDataRelocation> data_relocations = {});

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

  /// @brief True if any relocation references .text in a form DBT cannot remap.
  ///
  /// @details DBT can remap zero-addend RELA references to ordinary symbols
  /// defined in .text by updating the symbol value, and symbol-less
  /// R_AMDGPU_RELATIVE64 references by updating their explicit addend. Section
  /// symbols, REL records with implicit addends, and named-symbol references
  /// with nonzero addends require relocation-specific address reconstruction
  /// that is not implemented. BinaryTranslator uses this predicate to reject
  /// only those unsupported forms while allowing relocation-backed function
  /// tables that the text offset map can update safely.
  [[nodiscard]] bool has_unsupported_relocation_to_text() const;

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
  std::vector<uint8_t> image_;
  uint64_t text_offset_;
  uint64_t text_size_;
  uint64_t text_vaddr_;
  uint64_t text_tail_size_;
};

} // namespace rocjitsu
