// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/text_edit_error.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/result.h"

#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;
class CodeObjectPatcher;
class Decoder;
class InstructionBoundaries;
class Instrumentor;

enum class CheckedTextEditKind { Instruction, Cdna5TerminalExit };

// Private records translated from Instrumentor requests or resolved anchors.
struct CheckedTextEdit {
  uint64_t offset = 0;
  std::vector<uint8_t> expected;
  std::vector<uint8_t> replacement;
  CheckedTextEditKind kind = CheckedTextEditKind::Instruction;
};

// Immutable reference storage, independent of any decoder or private writer.
struct TextEditReferences {
  std::vector<uint64_t> branch_addresses;
  std::vector<uint64_t> symbol_endpoints;
  std::vector<uint64_t> relative_text_references;
  std::vector<uint64_t> kernel_entries;
};

/// @brief Local preflight and writer for one Instrumentor edit batch.
/// @details Borrows the source, selected decoder, complete boundaries and collected
/// references for one synchronous call. Observation collects references locally;
/// replacement queues share immutable source facts. No decoder or writer is shared.
class CheckedTextEdits {
private:
  CheckedTextEdits(const AmdGpuCodeObject &source, Decoder &decoder,
                   const InstructionBoundaries &boundaries, const TextEditReferences &references,
                   rocjitsu::IsaFamily architecture);
  static Result validate_source_extent(const AmdGpuCodeObject &source, TextEditError &error);
  static FailureOr<TextEditReferences>
  collect_references(const AmdGpuCodeObject &source, rocjitsu::IsaFamily architecture,
                     const InstructionBoundaries &boundaries, const CodeObjectPatcher &patcher,
                     std::span<const uint64_t> kernel_entries, TextEditError &error);
  Result validate_source_edits(std::span<const CheckedTextEdit> edits, TextEditError &error) const;
  Result validate(std::span<const CheckedTextEdit> edits, TextEditError &error) const;
  Result apply(std::span<const CheckedTextEdit> edits, std::span<uint8_t> destination,
               TextEditError &error) const;

  const AmdGpuCodeObject &source_;
  rocjitsu::IsaFamily architecture_;
  Decoder &decoder_;
  const InstructionBoundaries &boundaries_;
  const TextEditReferences &references_;
  friend class Instrumentor;
};

} // namespace rocjitsu
