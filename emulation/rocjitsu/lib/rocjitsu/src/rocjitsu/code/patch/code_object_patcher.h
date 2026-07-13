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

  void overwrite_text(std::span<const uint8_t> new_text);

  /// @brief Replace the original .text payload, allowing it to grow.
  ///
  /// @details The new payload is written as the same .text section. If it grows,
  /// later file contents and allocated virtual addresses are shifted while
  /// preserving LOAD alignment and keeping moved relocations, symbols, dynamic
  /// pointers, and kernel descriptor entry offsets coherent.
  [[nodiscard]] bool replace_text(std::span<const uint8_t> new_text);

  void update_elf_flags(uint32_t new_mach);

  /// @brief Replace same-length AMDHSA metadata target ISA strings in-place.
  ///
  /// PT_NOTE metadata cannot be resized here, so @p old_isa and @p new_isa must
  /// have the same byte length. The helper patches all occurrences because the
  /// target ISA appears in multiple msgpack/string-table contexts depending on
  /// the code-object producer.
  [[nodiscard]] bool patch_metadata_target_isa(std::string_view old_isa, std::string_view new_isa);

  /// @brief Raise AMDGPU metadata `.vgpr_count` entries in-place when possible.
  ///
  /// AMDHSA loaders may use the PT_NOTE metadata resource counts in addition to
  /// the kernel descriptor fields. This helper patches compact msgpack integer
  /// values without resizing the note payload.
  [[nodiscard]] bool patch_metadata_vgpr_count(uint32_t vgpr_count);

  /// @brief Raise AMDGPU metadata `.sgpr_count` entries in-place when possible.
  [[nodiscard]] bool patch_metadata_sgpr_count(uint32_t sgpr_count);

  /// @brief Patch AMDGPU metadata `.private_segment_fixed_size` entries.
  ///
  /// Values are patched by matching each metadata `.symbol` entry to its kernel
  /// descriptor symbol. Compact msgpack integers are resized when needed, which
  /// can shift later allocated sections in both file and virtual address space.
  [[nodiscard]] bool
  patch_metadata_private_segment_fixed_sizes(std::span<const KdTranslation> translations);

  [[nodiscard]] bool patch_kernel_descriptor(uint64_t file_offset,
                                             std::span<const uint8_t> descriptor);

  /// @brief Apply a descriptor translation plan to the in-memory ELF image.
  ///
  /// KernelDescriptorTranslator owns the resource/ABI decision. The patcher
  /// owns the byte-level descriptor update, cave placement, and entry redirect.
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
  [[nodiscard]] bool refresh_text_section_cache();

  std::vector<uint8_t> image_;
  uint64_t text_offset_;
  uint64_t text_size_;
  uint64_t text_vaddr_;
  uint64_t text_tail_size_;
};

} // namespace rocjitsu
