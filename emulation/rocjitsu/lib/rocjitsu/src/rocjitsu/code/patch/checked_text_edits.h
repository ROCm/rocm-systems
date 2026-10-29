// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/result.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;
class InstructionBoundaries;

enum class TextEditErrorCode {
  None,
  InvalidImage,
  UnsupportedTarget,
  UnsupportedReferences,
  InvalidRange,
  Overlap,
  StaleBytes,
  InvalidInstruction,
  Clause,
};

struct TextEditError {
  TextEditErrorCode code = TextEditErrorCode::None;
  uint64_t offset = 0;
  std::string message;
};

/// @brief Source section facts; offsets are relative to the supplied code object.
struct TextSectionIdentity {
  uint64_t index = 0;
  uint64_t file_offset = 0;
  uint64_t vaddr = 0;
  uint64_t size = 0;
};

/// @brief Replace one complete original instruction with equally many bytes.
class CheckedTextEdit {
public:
  uint64_t offset = 0;
  std::vector<uint8_t> expected;
  std::vector<uint8_t> replacement;
  bool operator==(const CheckedTextEdit &) const = default;
};

/// @brief An equal-length range map, including unchanged source bytes.
struct TextEditMapping {
  uint64_t source_offset = 0;
  uint64_t candidate_offset = 0;
  uint64_t size = 0;
};

struct CheckedTextEditResult {
  std::vector<uint8_t> image;
  TextSectionIdentity section;
  /// Concrete public binding when present; legacy ELF MACH identities remain
  /// in image unchanged even when no public target enum has been allocated.
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID;
  std::vector<CheckedTextEdit> edits;
  std::vector<TextEditMapping> mapping;
};

/// @brief Reusable full-text structural facts tied to one immutable source object.
/// @details Borrows the source, which must outlive this context and remain unchanged.
/// It fixes concrete decoder features, the entire text range and linear clause/reference
/// policy. It does not contain kernel roots or prove execution ownership. No caller can
/// supply replacement source facts; Instrumentor alone can reuse its private full-text index.
class CheckedTextEditContext {
public:
  static FailureOr<std::unique_ptr<CheckedTextEditContext>> create(const AmdGpuCodeObject &source,
                                                                   TextEditError &error);
  ~CheckedTextEditContext();
  const TextSectionIdentity &section() const;
  std::span<const Elf64_Shdr> section_headers() const;
  /// @brief Authoritative instruction, clause and reference facts for the complete source text.
  /// @details The returned reference is borrowed and must not outlive this context.
  const InstructionBoundaries &instruction_boundaries() const;
  CheckedTextEditContext(const CheckedTextEditContext &) = delete;
  CheckedTextEditContext &operator=(const CheckedTextEditContext &) = delete;

private:
  class State;
  explicit CheckedTextEditContext(const AmdGpuCodeObject &source);
  Result initialize(const InstructionBoundaries *boundaries, TextEditError &error);
  /// @brief Reuse Instrumentor's trusted full-text index without decoding it again.
  /// @details The index must describe this exact source using its source-derived decoder
  /// features, including concrete target identity. It must cover the complete text, not just
  /// reachable instructions. Initialization checks coverage but trusts decoded instruction,
  /// clause and reference facts. Both source and index are borrowed; they must remain immutable
  /// and outlive the returned context.
  static FailureOr<std::unique_ptr<CheckedTextEditContext>>
  from_instructions(const AmdGpuCodeObject &source, const InstructionBoundaries &boundaries,
                    TextEditError &error);
  std::unique_ptr<State> state_;
  friend class Instrumentor;
  friend FailureOr<CheckedTextEditResult> apply_checked_text_edits(const CheckedTextEditContext &,
                                                                   std::span<const CheckedTextEdit>,
                                                                   TextEditError &);
};

/// @brief Preflight every request, then apply to a private CodeObjectPatcher image.
/// @details Supports one executable .text and complete equal-length 4/8/12-byte edits.
/// Checks source format, aliases, instruction extents, clauses, references, stale bytes
/// and replacement encoding before writing. Other sections may be executable. Concrete
/// bindings select concrete decoders; known legacy non-variant ELF architectures also work.
/// Equal-length writes leave ELF layout unchanged under the patcher's existing contract.
/// No post-emission expected-delta or recipe analysis runs here; emitter tests establish
/// output semantics. PT_LOAD, descriptors and ownership remain consumer preconditions.
[[nodiscard]] FailureOr<CheckedTextEditResult>
apply_checked_text_edits(const CheckedTextEditContext &context,
                         std::span<const CheckedTextEdit> edits, TextEditError &error);

/// @brief One-shot convenience path; repeated edits should retain a checked context.
[[nodiscard]] FailureOr<CheckedTextEditResult>
apply_checked_text_edits(const AmdGpuCodeObject &source, std::span<const CheckedTextEdit> edits,
                         TextEditError &error);

} // namespace rocjitsu
