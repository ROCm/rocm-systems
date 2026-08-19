// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/rj_code.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
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

/// @brief Section-relative location of an address in allocated non-executable storage.
struct AllocatedDataSectionAddress {
  size_t section_index = 0;
  uint64_t section_offset = 0;
};

/// @brief Resolve a data address, including a nonempty section's one-past-end value.
///
/// @details Returns the first allocated, non-executable, nonempty section that contains @p vaddr or
/// ends there. Empty sections do not own an endpoint. If two sections meet at @p vaddr, callers
/// must not rely on which one is returned.
[[nodiscard]] std::optional<AllocatedDataSectionAddress>
resolve_allocated_data_section_address(std::span<const Elf64_Shdr> sections, uint64_t vaddr);

/// @brief Resolve a PC-relative data target without reclassifying an address inside source text.
[[nodiscard]] std::optional<AllocatedDataSectionAddress>
resolve_pc_relative_data_section_address(std::span<const Elf64_Shdr> sections, uint64_t vaddr,
                                         uint64_t text_vaddr, uint64_t text_size);

/// @brief One relocated literal64 PC builder whose target is inside `.text`.
///
/// @details Separate from PcRelativeDataRelocation because the target is named by a final `.text`
/// offset rather than a virtual address: a code target moves with the body that holds it, so it
/// cannot be resolved by locating the section that contains a fixed address. The literal is
/// recomputed as @c target_text_offset - (target_getpc_offset + 4), the distance an `s_get_pc_i64`
/// leaves to be made up, using only offsets within the emitted text.
struct PcRelativeTextRelocation {
  uint64_t target_getpc_offset = 0;
  uint64_t target_literal_offset = 0;
  uint64_t target_text_offset = 0;
};

/// @brief Location of a sidecar kernel descriptor appended into a loaded ELF segment.
struct AppendedSidecarDescriptor {
  uint64_t file_offset = 0;
  uint64_t vaddr = 0;
};

enum class CodeObjectPatchOutcome : uint8_t {
  Success,
  MalformedInput,
  AllocationFailure,
};

[[nodiscard]] constexpr std::string_view
code_object_patch_outcome_name(CodeObjectPatchOutcome outcome) {
  switch (outcome) {
  case CodeObjectPatchOutcome::Success:
    return "success";
  case CodeObjectPatchOutcome::MalformedInput:
    return "malformed input";
  case CodeObjectPatchOutcome::AllocationFailure:
    return "allocation failure";
  }
  return "unknown";
}

static_assert(code_object_patch_outcome_name(CodeObjectPatchOutcome::Success) == "success");
static_assert(code_object_patch_outcome_name(CodeObjectPatchOutcome::MalformedInput) ==
              "malformed input");
static_assert(code_object_patch_outcome_name(CodeObjectPatchOutcome::AllocationFailure) ==
              "allocation failure");
static_assert(code_object_patch_outcome_name(static_cast<CodeObjectPatchOutcome>(0xff)) ==
              "unknown");

/// Typed result of a transactional patch operation without a return payload.
class CodeObjectPatchResult {
public:
  [[nodiscard]] static constexpr CodeObjectPatchResult success() {
    return CodeObjectPatchOutcome::Success;
  }
  [[nodiscard]] static constexpr CodeObjectPatchResult malformed_input() {
    return CodeObjectPatchOutcome::MalformedInput;
  }
  [[nodiscard]] static constexpr CodeObjectPatchResult allocation_failure() {
    return CodeObjectPatchOutcome::AllocationFailure;
  }

  [[nodiscard]] constexpr CodeObjectPatchOutcome outcome() const { return outcome_; }
  [[nodiscard]] constexpr bool succeeded() const {
    return outcome_ == CodeObjectPatchOutcome::Success;
  }
  [[nodiscard]] constexpr explicit operator bool() const { return succeeded(); }

private:
  constexpr CodeObjectPatchResult(CodeObjectPatchOutcome outcome) : outcome_(outcome) {}

  CodeObjectPatchOutcome outcome_;
};

/// Typed result of appending one batch of loaded sidecar descriptors.
class SidecarDescriptorAppendResult {
public:
  [[nodiscard]] static SidecarDescriptorAppendResult
  success(std::vector<AppendedSidecarDescriptor> descriptors) {
    return {CodeObjectPatchOutcome::Success, std::move(descriptors)};
  }
  [[nodiscard]] static SidecarDescriptorAppendResult malformed_input() {
    return {CodeObjectPatchOutcome::MalformedInput, {}};
  }
  [[nodiscard]] static SidecarDescriptorAppendResult allocation_failure() {
    return {CodeObjectPatchOutcome::AllocationFailure, {}};
  }

  [[nodiscard]] CodeObjectPatchOutcome outcome() const { return outcome_; }
  [[nodiscard]] bool succeeded() const { return outcome_ == CodeObjectPatchOutcome::Success; }
  [[nodiscard]] explicit operator bool() const { return succeeded(); }
  [[nodiscard]] const std::vector<AppendedSidecarDescriptor> &descriptors() const {
    return descriptors_;
  }
  [[nodiscard]] std::vector<AppendedSidecarDescriptor> take_descriptors() && {
    return std::move(descriptors_);
  }

private:
  SidecarDescriptorAppendResult(CodeObjectPatchOutcome outcome,
                                std::vector<AppendedSidecarDescriptor> descriptors)
      : outcome_(outcome), descriptors_(std::move(descriptors)) {}

  CodeObjectPatchOutcome outcome_;
  std::vector<AppendedSidecarDescriptor> descriptors_;
};

enum class TextReplacementOutcome : uint8_t {
  Success,
  MalformedInput,
  AllocationFailure,
  FileGrowthLimitExceeded,
};

[[nodiscard]] constexpr std::string_view
text_replacement_outcome_name(TextReplacementOutcome outcome) {
  switch (outcome) {
  case TextReplacementOutcome::Success:
    return "success";
  case TextReplacementOutcome::MalformedInput:
    return "malformed input";
  case TextReplacementOutcome::AllocationFailure:
    return "allocation failure";
  case TextReplacementOutcome::FileGrowthLimitExceeded:
    return "file growth limit exceeded";
  }
  return "unknown";
}

/// Typed result of one transactional .text replacement attempt.
///
/// A successful replacement and a file-growth-limit rejection always carry the
/// exact inserted byte count, including alignment padding. Malformed-input and
/// allocation failures carry that count only when alignment had already been
/// resolved before the failure. An engaged zero is reserved for a transaction
/// that was proven not to grow the file; unresolved growth is disengaged.
class TextReplacementResult {
public:
  [[nodiscard]] static TextReplacementResult success(size_t required_file_growth) {
    return {TextReplacementOutcome::Success, required_file_growth};
  }

  [[nodiscard]] static TextReplacementResult
  malformed_input(std::optional<size_t> required_file_growth = std::nullopt) {
    return {TextReplacementOutcome::MalformedInput, required_file_growth};
  }

  [[nodiscard]] static TextReplacementResult
  allocation_failure(std::optional<size_t> required_file_growth = std::nullopt) {
    return {TextReplacementOutcome::AllocationFailure, required_file_growth};
  }

  [[nodiscard]] static TextReplacementResult
  file_growth_limit_exceeded(size_t required_file_growth) {
    return {TextReplacementOutcome::FileGrowthLimitExceeded, required_file_growth};
  }

  [[nodiscard]] TextReplacementOutcome outcome() const { return outcome_; }
  [[nodiscard]] bool succeeded() const { return outcome_ == TextReplacementOutcome::Success; }
  [[nodiscard]] explicit operator bool() const { return succeeded(); }
  /// Exact aligned file growth when it was resolved before this outcome.
  [[nodiscard]] const std::optional<size_t> &required_file_growth() const {
    return required_file_growth_;
  }

private:
  TextReplacementResult(TextReplacementOutcome outcome, std::optional<size_t> required_file_growth)
      : outcome_(outcome), required_file_growth_(required_file_growth) {}

  TextReplacementOutcome outcome_;
  std::optional<size_t> required_file_growth_;
};

class CodeObjectPatcher {
public:
  explicit CodeObjectPatcher(const AmdGpuCodeObject &obj);
  CodeObjectPatcher(const CodeObjectPatcher &) = delete;
  CodeObjectPatcher &operator=(const CodeObjectPatcher &) = delete;
  CodeObjectPatcher(CodeObjectPatcher &&other) noexcept;
  CodeObjectPatcher &operator=(CodeObjectPatcher &&) = delete;
  ~CodeObjectPatcher();

  std::span<uint8_t> text_bytes();
  std::span<const uint8_t> text_bytes() const;

  std::span<const uint8_t> image_bytes() const { return {image_.data(), image_.size()}; }
  /// @brief Validated raw section headers, including SHT_NOBITS entries.
  [[nodiscard]] std::vector<Elf64_Shdr> section_headers() const;
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
  /// @returns A typed outcome that distinguishes success, malformed input,
  /// allocation failure, and an exact @p max_file_growth rejection. Every
  /// failure leaves this patcher unchanged.
  [[nodiscard]] TextReplacementResult replace_text(std::span<const uint8_t> new_text,
                                                   size_t max_file_growth) noexcept;
  /// @brief Replace text while applying explicit text and data relocation maps.
  [[nodiscard]] bool replace_text(std::span<const uint8_t> new_text,
                                  std::span<const TextOffsetRelocation> text_relocations = {},
                                  std::span<const PcRelativeDataRelocation> data_relocations = {},
                                  std::span<const PcRelativeTextRelocation> code_relocations = {},
                                  bool require_every_text_symbol_mapped = false,
                                  const std::unordered_map<uint64_t, uint64_t>
                                      *canonical_code_pointer_placement = nullptr);

  /// @brief True if any non-inert relocation's place (r_offset) falls inside .text.
  ///
  /// @details DBT compacts/expands/moves instructions within .text but does not
  /// remap relocation places that land in .text — replace_text() only shifts
  /// relocation offsets for whole sections moved *after* .text. An in-.text
  /// relocation would therefore be applied to the wrong translated bytes.
  /// R_AMDGPU_NONE performs no write at its relocation place. Runtime acceptance
  /// of that record is checked separately by has_rocr_rejected_none_relocation().
  /// BinaryTranslator uses this to fail closed instead of miscompiling. Real
  /// AMDHSA kernel code objects carry no such relocations, so this rejects only
  /// genuinely unsupported inputs.
  [[nodiscard]] bool has_relocations_within_text() const;

  /// @brief True if ROCr cannot resolve an ET_DYN SHT_RELA section target.
  ///
  /// @details ROCr accepts section zero as the target-less dynamic form and a
  /// valid non-null section as an explicit target. A nonzero SHT_NULL target or
  /// an out-of-range sh_info fails before relocation records are dispatched.
  [[nodiscard]] bool has_malformed_rocr_relocation_section() const;

  /// @brief True if ROCr rejects an R_AMDGPU_NONE record before DBT can rewrite the image.
  ///
  /// @details Target-less SHT_RELA sections in ET_DYN code objects are processed as
  /// dynamic relocations, where ROCr has no R_AMDGPU_NONE case. Detect those records by
  /// section mode and type alone, before consulting their place or symbol metadata.
  /// Malformed section metadata is owned by has_malformed_rocr_relocation_section().
  [[nodiscard]] bool has_rocr_rejected_none_relocation() const;

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

  /// @brief Overwrite one kernel descriptor's `private_segment_fixed_size`.
  ///
  /// @details Used by the DBI spill path to grow a kernel's per-lane scratch to
  /// cover the appended spill zone. Reads the descriptor at @p
  /// descriptor_file_offset, sets the scratch-size field to @p bytes, and writes
  /// it back. Returns false if the descriptor does not fit in the image.
  ///
  /// Deliberately narrower than DBT's apply_kernel_descriptor_translation: a
  /// same-arch instrument-only pass must re-encode no other descriptor field
  /// (VGPR/SGPR granules, mode bits, USER_SGPR_COUNT, kernarg size).
  [[nodiscard]] bool set_private_segment_fixed_size(uint64_t descriptor_file_offset,
                                                    uint32_t bytes);

  /// @brief Apply a descriptor translation plan to the in-memory ELF image.
  ///
  /// KernelDescriptorTranslator owns the resource/ABI decision. BinaryTranslator
  /// owns text relocation and any local prologue layout. The patcher only
  /// mutates descriptor bytes and redirects the descriptor to the already-known
  /// relocated entry offset. Failure leaves this patcher unchanged.
  [[nodiscard]] bool apply_kernel_descriptor_translation(const KdTranslation &translation,
                                                         rj_code_arch_t target_arch);

  /// @brief Append non-symbolized sidecar descriptors in loaded memory after .text.
  ///
  /// @details Sidecar descriptors must be visible to the GPU loader, but they
  /// do not need to be discoverable through the AMDHSA symbol table. The patcher
  /// therefore places descriptor bytes in the executable LOAD tail immediately
  /// after the translated .text payload, without growing the .text section
  /// itself. Runtime metadata records the returned virtual addresses.
  /// @returns A typed transactional result. Every failure leaves this patcher
  /// unchanged.
  [[nodiscard]] SidecarDescriptorAppendResult
  append_sidecar_descriptor_translations(std::span<const KdTranslation> translations,
                                         rj_code_arch_t target_arch,
                                         uint64_t alignment = 64) noexcept;

  /// @brief Append a named, non-allocated ELF section without moving loadable bytes.
  ///
  /// @details DBT runtime metadata must not perturb code-object load addresses:
  /// ROCR and the kernel descriptor ABI have already consumed those addresses.
  /// This helper appends the payload, a copied section-string table, and a new
  /// section-header table at EOF. Program headers and allocated sections are
  /// left untouched, so the new section is available to tools and rocjitsu's
  /// own loader-side metadata parser but is not mapped into GPU code memory.
  /// @returns A typed transactional result. Every failure leaves this patcher
  /// unchanged.
  [[nodiscard]] CodeObjectPatchResult append_nonalloc_section(std::string_view name,
                                                              std::span<const uint8_t> contents,
                                                              uint64_t alignment = 1) noexcept;

  /// @brief Redirect one kernel descriptor from @p old_entry_text_offset to @p
  /// new_entry_text_offset.
  ///
  /// AMDHSA stores the entry as a signed KD-relative byte offset. The
  /// text-offset delta is the same delta in KD-relative coordinates, so the
  /// patcher does not need symbol virtual addresses here. Failure leaves this
  /// patcher unchanged.
  [[nodiscard]] bool redirect_kernel_entry(uint64_t descriptor_file_offset,
                                           uint64_t old_entry_text_offset,
                                           uint64_t new_entry_text_offset);

  std::vector<uint8_t> emit() const &;
  std::vector<uint8_t> emit() &&;

private:
  [[nodiscard]] TextReplacementResult replace_text_impl(std::span<const uint8_t> new_text,
                                                        size_t max_file_growth);
  [[nodiscard]] SidecarDescriptorAppendResult
  append_sidecar_descriptor_translations_impl(std::span<const KdTranslation> translations,
                                              rj_code_arch_t target_arch, uint64_t alignment);
  [[nodiscard]] CodeObjectPatchResult
  append_nonalloc_section_impl(std::string_view name, std::span<const uint8_t> contents,
                               uint64_t alignment);

  std::vector<uint8_t> image_;
  std::optional<size_t> text_section_index_;
  uint64_t text_offset_;
  uint64_t text_size_;
  uint64_t text_vaddr_;
  uint64_t text_tail_size_;
};

} // namespace rocjitsu
