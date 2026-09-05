// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_versioned_publication.h
/// @brief Shared lane-state algebra for bounded MOI publication transactions.

#pragma once

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_moi_exact_shadow_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <cstdint>
#include <initializer_list>
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
  MoiPublicationExec(InstructionSequence &sequence, uint16_t narrow_save, rj_code_arch_t arch)
      : sequence_(sequence), narrow_save_(narrow_save), arch_(arch) {}

  [[nodiscard]] bool restore(uint16_t source) {
    sequence_.append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, source, arch_));
    return static_cast<bool>(sequence_);
  }

  [[nodiscard]] bool save(uint16_t destination) {
    sequence_.append(instrumentation::build_s_mov_b64(destination, kAmdGpuExecLo, arch_));
    return static_cast<bool>(sequence_);
  }

  [[nodiscard]] bool narrow_vcc() {
    sequence_.append(instrumentation::build_s_and_saveexec_b64(narrow_save_, kAmdGpuVccLo, arch_));
    return static_cast<bool>(sequence_);
  }

  template <typename Predicate> [[nodiscard]] bool narrow(const Predicate &predicate) {
    sequence_.append(predicate,
                     instrumentation::build_s_and_saveexec_b64(narrow_save_, kAmdGpuVccLo, arch_));
    return static_cast<bool>(sequence_);
  }

  [[nodiscard]] bool require_literal(uint16_t value, uint32_t literal, bool equal) {
    const auto compare = equal ? instrumentation::build_v_cmp_eq_u32_vcc(
                                     scalar_positive_inline_u32(literal), value, arch_)
                               : instrumentation::build_v_cmp_ne_u32_vcc(
                                     scalar_positive_inline_u32(literal), value, arch_);
    return narrow(compare);
  }

private:
  InstructionSequence &sequence_;
  uint16_t narrow_save_ = 0;
  rj_code_arch_t arch_{};
};

/// Register and policy description for one direct-mapped version claim.
///
/// The protocol deliberately uses the same bounded loop for a cheap metadata
/// record and for a larger payload snapshot. `append_candidate` below owns
/// only the domain's stable-snapshot and replacement predicates; this object
/// owns retry, reservation, and winner selection.
struct MoiVersionClaim {
  uint16_t slot_address_vgpr = 0;
  uint16_t eligible_exec_sgpr = 0;
  uint16_t claimed_exec_sgpr = 0;
  uint16_t retry_exec_sgpr = 0;
  uint16_t retry_count_sgpr = 0;
  uint16_t base_version_vgpr = 0;
  uint16_t desired_vgpr = 0;
  uint16_t expected_vgpr = 0;
  uint32_t version_offset = 0;
  uint32_t retry_limit = 0;
  uint32_t sleep_delay = 0;
};

/// One typed word in a publication payload. Slot layout stays domain-owned;
/// the shared protocol owns the ordered memory operation over that layout.
struct MoiPublicationField {
  uint32_t offset = 0;
  uint16_t vgpr = 0;
};

[[nodiscard]] inline bool
append_moi_publication_loads(std::vector<uint32_t> &words, uint16_t slot_address_vgpr,
                             std::initializer_list<MoiPublicationField> fields,
                             rj_code_arch_t arch) {
  for (const MoiPublicationField field : fields) {
    if (!consan_moi_detail::append_load_u32_vgpr_at_offset(words, slot_address_vgpr, field.offset,
                                                           field.vgpr, arch)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool
append_moi_publication_stores(std::vector<uint32_t> &words, uint16_t slot_address_vgpr,
                              std::initializer_list<MoiPublicationField> fields,
                              rj_code_arch_t arch) {
  for (const MoiPublicationField field : fields) {
    if (!consan_moi_detail::append_store_u32_vgpr_at_offset(words, slot_address_vgpr, field.offset,
                                                            field.vgpr, arch)) {
      return false;
    }
  }
  return true;
}

/// Emit one odd/even version transition through a device-scope compare-swap.
///
/// `desired_vgpr` and `expected_vgpr` are the adjacent data pair required by
/// the native B32 compare-swap encoding. The returned old value overwrites
/// `desired_vgpr`; successful lanes remain in EXEC. A delta of zero is an
/// explicit restoration of `base_version_vgpr`, allowing the same operation
/// to claim, commit, or roll back a slot without bespoke CAS sequences.
[[nodiscard]] inline bool
append_moi_version_transition(std::vector<uint32_t> &words, InstructionSequence &sequence,
                              MoiPublicationExec &exec, uint16_t slot_address_vgpr,
                              uint32_t version_offset, uint16_t base_version_vgpr,
                              uint16_t desired_vgpr, uint16_t expected_vgpr, uint32_t desired_delta,
                              uint32_t expected_delta, rj_code_arch_t arch) {
  if (expected_vgpr != static_cast<uint16_t>(desired_vgpr + 1u))
    return false;
  const auto materialize = [&](uint16_t destination, uint32_t delta) {
    return delta == 0u
               ? std::optional<std::vector<uint32_t>>{{build_v_mov_b32_e32(
                     destination, vector_source_vgpr(base_version_vgpr), arch)}}
               : instrumentation::build_v_add_u32(destination, scalar_positive_inline_u32(delta),
                                                  base_version_vgpr, arch);
  };
  const auto desired = materialize(desired_vgpr, desired_delta);
  const auto expected = materialize(expected_vgpr, expected_delta);
  if (!desired || !expected ||
      !append_add_literal_field(words, slot_address_vgpr, version_offset, desired_vgpr, arch) ||
      !sequence.emit_all(*desired, *expected,
                         instrumentation::build_flat_atomic_cmpswap_b32(
                             slot_address_vgpr, desired_vgpr, desired_vgpr,
                             /*return_old_value=*/true, kAmdGpuScopeDevice, arch)) ||
      !append_moi_global_atomic_wait(words, arch)) {
    return false;
  }
  return exec.narrow(instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected_vgpr),
                                                             desired_vgpr, arch));
}

/// Claim one stable versioned slot through a bounded common transaction.
///
/// `append_candidate` runs with the eligible lanes in EXEC. It must load the
/// authoritative even version into `base_version_vgpr` and narrow EXEC to the
/// lanes permitted to replace the slot. A zero retry limit requests exactly
/// one attempt. On return EXEC and `claimed_exec_sgpr` contain the winners;
/// exhausted or ineligible lanes are absent, and `slot_address_vgpr` once
/// again names the start of the slot rather than its version field.
template <typename AppendCandidate>
[[nodiscard]] bool
append_moi_bounded_version_claim(std::vector<uint32_t> &words, InstructionSequence &sequence,
                                 MoiPublicationExec &exec, const MoiVersionClaim &claim,
                                 AppendCandidate &&append_candidate, rj_code_arch_t arch) {
  if (claim.expected_vgpr != static_cast<uint16_t>(claim.desired_vgpr + 1u))
    return false;

  if (claim.retry_limit != 0u) {
    constexpr uint16_t kScalarLiteralSource = 255u;
    words.push_back(build_s_mov_b32(claim.retry_count_sgpr, kScalarLiteralSource, arch));
    words.push_back(claim.retry_limit);
  }

  const InstructionSequence::Label retry = sequence.mark_label();
  if (!exec.restore(claim.eligible_exec_sgpr) || !append_candidate() ||
      !exec.save(claim.retry_exec_sgpr) ||
      !append_moi_version_transition(words, sequence, exec, claim.slot_address_vgpr,
                                     claim.version_offset, claim.base_version_vgpr,
                                     claim.desired_vgpr, claim.expected_vgpr, /*desired_delta=*/1u,
                                     /*expected_delta=*/0u, arch) ||
      !exec.save(claim.claimed_exec_sgpr) || !exec.restore(claim.retry_exec_sgpr) ||
      !append_add_literal_field(words, claim.slot_address_vgpr, 0u - claim.version_offset,
                                claim.expected_vgpr, arch) ||
      !exec.restore(claim.claimed_exec_sgpr)) {
    return false;
  }

  if (claim.retry_limit == 0u)
    return true;

  if (!sequence.emit(instrumentation::build_s_andn2_b64(
          claim.retry_exec_sgpr, claim.eligible_exec_sgpr, claim.claimed_exec_sgpr, arch))) {
    return false;
  }
  const InstructionSequence::Label claimed = sequence.make_label();
  if (!sequence.emit_branch(claimed, InstructionSequence::BranchKind::ExecNonzero) ||
      !exec.restore(claim.retry_exec_sgpr)) {
    return false;
  }
  words.push_back(build_s_sleep(claim.sleep_delay, arch));
  if (!sequence.emit_all(instrumentation::build_s_sub_u32(claim.retry_count_sgpr,
                                                          claim.retry_count_sgpr,
                                                          scalar_positive_inline_u32(1), arch),
                         instrumentation::build_s_cmp_lg_u32(claim.retry_count_sgpr,
                                                             scalar_positive_inline_u32(0), arch),
                         instrumentation::build_s_cbranch_scc0(/*simm16=*/1, arch)) ||
      !sequence.emit_branch(retry, InstructionSequence::BranchKind::ExecNonzero) ||
      !sequence.bind(claimed)) {
    return false;
  }
  return exec.restore(claim.claimed_exec_sgpr);
}

} // namespace rocjitsu::consan_moi_impl
