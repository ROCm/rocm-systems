// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_versioned_publication.h
/// @brief Shared lane-state algebra for bounded MOI publication transactions.

#pragma once

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <cstdint>
#include <vector>

namespace rocjitsu::consan_moi_impl {

/// Target-normalized EXEC state used by versioned slot transactions.
///
/// Callers assign semantic lifetimes to mask registers and provide comparison
/// predicates. This type is the sole owner of save/restore/narrow mechanics,
/// allowing one publication protocol to operate on one slot or a bounded set
/// without encoding the evidence domain.
class MoiPublicationExec {
public:
  MoiPublicationExec(std::vector<uint32_t> &words, uint16_t narrow_save, rj_code_arch_t arch)
      : sequence_(words), narrow_save_(narrow_save), arch_(arch) {}

  [[nodiscard]] bool restore(uint16_t source) {
    return sequence_.emit(instrumentation::build_s_mov_b64(kAmdGpuExecLo, source, arch_));
  }

  [[nodiscard]] bool save(uint16_t destination) {
    return sequence_.emit(instrumentation::build_s_mov_b64(destination, kAmdGpuExecLo, arch_));
  }

  [[nodiscard]] bool narrow_vcc() {
    return sequence_.emit(
        instrumentation::build_s_and_saveexec_b64(narrow_save_, kAmdGpuVccLo, arch_));
  }

  template <typename Predicate> [[nodiscard]] bool narrow(const Predicate &predicate) {
    return sequence_.emit_all(
        predicate, instrumentation::build_s_and_saveexec_b64(narrow_save_, kAmdGpuVccLo, arch_));
  }

  [[nodiscard]] bool require_literal(uint16_t value, uint32_t literal, bool equal) {
    const auto compare = equal ? instrumentation::build_v_cmp_eq_u32_vcc(
                                     scalar_positive_inline_u32(literal), value, arch_)
                               : instrumentation::build_v_cmp_ne_u32_vcc(
                                     scalar_positive_inline_u32(literal), value, arch_);
    return narrow(compare);
  }

private:
  InstructionSequence sequence_;
  uint16_t narrow_save_ = 0;
  rj_code_arch_t arch_{};
};

} // namespace rocjitsu::consan_moi_impl
