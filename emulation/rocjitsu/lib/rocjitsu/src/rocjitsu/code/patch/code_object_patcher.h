// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

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

  /// @brief Append descriptor-derived instructions and return their .text offset.
  ///
  /// The descriptor translator owns the ABI policy and emits @p prologue_words.
  /// The patcher owns the byte-level placement: it puts those words in the code
  /// cave at a 256-byte-aligned launch address and appends a branch into the
  /// original kernel entry. The caller then redirects the kernel descriptor to
  /// the returned offset.
  [[nodiscard]] std::optional<uint64_t>
  append_kernel_entry_prologue(uint64_t entry_text_offset, std::span<const uint32_t> prologue_words,
                               rj_code_arch_t arch,
                               std::optional<uint16_t> long_branch_sgpr_pair = std::nullopt);

  /// @brief Redirect one kernel descriptor from @p old_entry_text_offset to @p
  /// new_entry_text_offset.
  ///
  /// AMDHSA stores the entry as a signed KD-relative byte offset. The
  /// text-offset delta is the same delta in KD-relative coordinates, so the
  /// patcher does not need symbol virtual addresses here.
  [[nodiscard]] bool redirect_kernel_entry(uint64_t descriptor_file_offset,
                                           uint64_t old_entry_text_offset,
                                           uint64_t new_entry_text_offset);

  void append_cave_body(std::span<const uint32_t> words);

  void overwrite_cave_body(uint64_t offset, std::span<const uint32_t> words);

  void truncate_cave_body(uint64_t size);

  uint64_t cave_body_size() const { return cave_body_.size(); }

  /// @brief Materialize the accumulated cave body as an executable ELF section.
  ///
  /// The section data is inserted immediately after the original .text bytes so
  /// all cave offsets are still expressible as .text-relative byte offsets:
  /// original .text occupies [0, text_size_) and this section starts at
  /// text_size_. The caller must therefore set cave_start() to text_size()
  /// before emitting branch stubs.
  ///
  /// @returns true if there was no cave body to emit or the section was
  ///          materialized successfully; false if the original .text section
  ///          header could not be found.
  [[nodiscard]] bool append_cave_section(std::string_view section_name = ".rj_translations");

  /// @brief Set the .text-relative byte offset where the cave body will be placed.
  /// This must be called before any apply_semantic() calls so branch offsets
  /// are computed correctly.
  void set_cave_start(uint64_t offset) { cave_start_ = offset; }

  /// @brief Get the .text-relative cave start offset.
  uint64_t cave_start() const { return cave_start_; }

  std::span<const uint8_t> cave_body() const { return cave_body_; }

  std::vector<uint8_t> emit() const;

private:
  [[nodiscard]] bool refresh_text_section_cache();

  std::vector<uint8_t> image_;
  uint64_t text_offset_;
  uint64_t text_size_;
  std::vector<uint8_t> cave_body_;
  uint64_t cave_start_ = 0;
};

} // namespace rocjitsu
