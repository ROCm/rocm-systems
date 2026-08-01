// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
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
  uint64_t text_offset_;
  uint64_t text_size_;
  uint64_t text_vaddr_;
  uint64_t text_tail_size_;
};

} // namespace rocjitsu
