// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_versioned_publication.h
/// @brief Shared lane-state algebra for bounded MOI publication transactions.

#pragma once

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
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

/// Emit one odd/even version transition through a device-scope compare-swap.
///
/// `desired_vgpr` and `expected_vgpr` are the adjacent data pair required by
/// the native B32 compare-swap encoding. The returned old value overwrites
/// `desired_vgpr`; successful lanes remain in EXEC. A delta of zero is an
/// explicit restoration of `base_version_vgpr`, allowing the same operation
/// to claim, commit, or roll back a slot without bespoke CAS sequences.
[[nodiscard]] inline bool append_moi_version_transition(
    std::vector<uint32_t> &words, InstructionSequence &sequence, MoiPublicationExec &exec,
    uint16_t slot_address_vgpr, uint16_t base_version_vgpr, uint16_t desired_vgpr,
    uint16_t expected_vgpr, uint32_t desired_delta, uint32_t expected_delta,
    rj_code_arch_t arch) {
  if (expected_vgpr != static_cast<uint16_t>(desired_vgpr + 1u))
    return false;
  const auto materialize = [&](uint16_t destination, uint32_t delta) {
    return delta == 0u
               ? std::optional<std::vector<uint32_t>>{{build_v_mov_b32_e32(
                     destination, vector_source_vgpr(base_version_vgpr), arch)}}
               : instrumentation::build_v_add_u32(destination,
                                                  scalar_positive_inline_u32(delta),
                                                  base_version_vgpr, arch);
  };
  const auto desired = materialize(desired_vgpr, desired_delta);
  const auto expected = materialize(expected_vgpr, expected_delta);
  if (!desired || !expected ||
      !sequence.emit_all(*desired, *expected,
                         instrumentation::build_flat_atomic_cmpswap_b32(
                             slot_address_vgpr, desired_vgpr, desired_vgpr,
                             /*return_old_value=*/true, kAmdGpuScopeDevice, arch)) ||
      !append_moi_global_atomic_wait(words, arch)) {
    return false;
  }
  return exec.narrow(instrumentation::build_v_cmp_eq_u32_vcc(
      vector_source_vgpr(expected_vgpr), desired_vgpr, arch));
}

} // namespace rocjitsu::consan_moi_impl
