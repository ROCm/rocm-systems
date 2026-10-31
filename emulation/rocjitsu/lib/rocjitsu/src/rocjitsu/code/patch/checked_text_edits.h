// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/result.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;
class CodeObjectPatcher;
class InstructionBoundaries;
class Instrumentor;

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

/// @brief Structural edit diagnostic shared by validation and patching.
struct TextEditError {
  TextEditErrorCode code = TextEditErrorCode::None;
  uint64_t offset = 0; // .text-relative byte location when known; zero is also the default.
  std::string message;
};

// Internal transaction record. Requests enter through InstrumentationPoint;
// these records never carry independent authority to mutate a source object.
struct CheckedTextEdit {
  uint64_t offset = 0;
  std::vector<uint8_t> expected;
  std::vector<uint8_t> replacement;
};

/// @brief Instrumentor's private source-bound preflight and text writer.
/// @details This is shared implementation, not a second editing API. Only Instrumentor
/// creates or uses it. Facts borrow the immutable source but own their instruction
/// index and reference storage, so a fresh queue can share them safely. No ELF is emitted
/// here: after complete preflight, writes target the orchestrator's private text buffer.
class CheckedTextEdits {
public:
  ~CheckedTextEdits();

private:
  class State;
  explicit CheckedTextEdits(const AmdGpuCodeObject &source);
  // Optional boundaries must describe this unchanged source under the same
  // decoder profile. Initialization checks complete coverage but trusts the
  // clause/reference facts. It copies the index before returning; the caller
  // must keep the supplied index unchanged and alive during this call.
  static FailureOr<std::shared_ptr<const CheckedTextEdits>>
  create(const AmdGpuCodeObject &source, rj_code_arch_t arch,
         const InstructionBoundaries *boundaries, const CodeObjectPatcher &patcher,
         TextEditError &error);
  Result initialize(rj_code_arch_t arch, const InstructionBoundaries *boundaries,
                    const CodeObjectPatcher &patcher, TextEditError &error);
  const InstructionBoundaries &instruction_boundaries() const;
  Result validate(std::span<const CheckedTextEdit> edits, TextEditError &error) const;
  Result apply(std::span<const CheckedTextEdit> edits, std::span<uint8_t> destination,
               TextEditError &error) const;

  std::unique_ptr<State> state_;
  friend class Instrumentor;
};

} // namespace rocjitsu
