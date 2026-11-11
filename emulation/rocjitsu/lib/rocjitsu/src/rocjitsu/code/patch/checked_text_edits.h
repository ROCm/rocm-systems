// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/text_edit_error.h"
#include "rocjitsu/code/rj_code.h"
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

// Private original-text anchor replacement, constructed from a resolved site
// and its built trampoline. This is not an independent mutation request API.
struct CheckedTextEdit {
  uint64_t offset = 0;
  std::vector<uint8_t> expected;
  std::vector<uint8_t> replacement;
};

/// @brief Local preflight and writer for Instrumentor's observation anchor batch.
/// @details Borrows the source, selected decoder, full-section boundaries and patcher
/// for one synchronous call. Kernel entries are original .text-relative offsets from
/// observation resolution. It emits no ELF and retains no facts after the call.
class CheckedTextEdits {
private:
  CheckedTextEdits(const AmdGpuCodeObject &source, rj_code_arch_t architecture, Decoder &decoder,
                   const InstructionBoundaries &boundaries, const CodeObjectPatcher &patcher,
                   std::span<const uint64_t> kernel_entries);
  Result apply(std::span<const CheckedTextEdit> edits, std::span<uint8_t> destination,
               TextEditError &error) const;

  const AmdGpuCodeObject &source_;
  rj_code_arch_t architecture_;
  Decoder &decoder_;
  const InstructionBoundaries &boundaries_;
  const CodeObjectPatcher &patcher_;
  std::span<const uint64_t> kernel_entries_;
  friend class Instrumentor;
};

} // namespace rocjitsu
