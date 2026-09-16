// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file legalization/gfx1250_b0_to_a0.cpp
/// @brief Handwritten gfx1250 B0-to-A0 legalization classification.

#include "rocjitsu/code/dbt/legalization/gfx1250_b0_to_a0.h"

#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/analysis/gfx1250_vgpr_msb.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/dbt/semantic/gfx1250_flat_scratch_base.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/instruction.h"

#include <array>
#include <cstring>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocjitsu {
namespace {

/// @brief Exact instruction names handled by the B0-to-A0 expansion profile.
///
/// @details Keep this list aligned with the implemented B0-to-A0 semantic
/// rules. Prefix-classified WMMA/SWMMAC and cluster-load instructions are
/// handled separately by family-level translation rules.
///
/// A rule keyed on an exact (encoding id, opcode) that only inserts spacing and
/// leaves the opcode alone needs no entry here at all: with no classification
/// the instruction reaches the semantic rule table on its own and, when the rule
/// declines, takes the verbatim copy path. The MODE-write separation rule is
/// registered that way, and the diff report is what shows it, as "expand
/// semantic" with no legalization action.
///
/// An entry here does more than label the report when it is a mnemonic prefix
/// that covers more instructions than the semantic table implements: the
/// unimplemented siblings then fail closed instead of passing through silently.
/// That is why the integer WMMA prefixes below keep their entry.
///
/// Separately, non-opcode-keyed rewrites are classified by SemanticTranslator's
/// profile registry, and the barrier-state and sleep/monitor families are
/// DEFERRED with a pass-through report rather than fail-closed (see
/// gfx1250_b0_to_a0_is_deferred_family).
inline constexpr std::array<std::string_view, 17> kExactB0ToA0TranslationMnemonics = {
    "ds_load_2addr_b32",
    "ds_load_2addr_b64",
    "ds_load_2addr_stride64_b32",
    "ds_load_2addr_stride64_b64",
    "ds_store_2addr_b32",
    "ds_store_2addr_b64",
    "ds_store_2addr_stride64_b32",
    "ds_store_2addr_stride64_b64",
    "ds_storexchg_2addr_rtn_b32",
    "ds_storexchg_2addr_rtn_b64",
    "ds_storexchg_2addr_stride64_rtn_b32",
    "ds_storexchg_2addr_stride64_rtn_b64",
    "ds_load_addtid_b32",
    "ds_store_addtid_b32",
    "v_cvt_pk_fp8_f32",
    "v_cvt_sr_fp8_f32",
    "tensor_load_to_lds",
};

[[nodiscard]] bool requires_b0_to_a0_expansion(std::string_view mnemonic) {
  // This is deliberately more conservative than the reference patch
  // patterns. Rocjitsu relocates and expands instructions, so it cannot retain
  // a source clause without revalidating the translated membership and
  // placement constraints.
  if (mnemonic == "s_clause")
    return true;

  for (std::string_view exact : kExactB0ToA0TranslationMnemonics) {
    if (mnemonic == exact)
      return true;
  }

  // Every cluster-load form is kept as a cluster load and wrapped to run with
  // M0 forced to zero (save M0, set M0 = 0, load, restore M0). The semantic rule
  // performs the rewrite.
  if (mnemonic.starts_with("cluster_load_"))
    return true;

  // The reference patch accepts every encoding suffix in this conversion
  // family. The semantic rule further restricts the expansion to the
  // operand/modifier combinations selected by the B0-to-A0 profile.
  if (mnemonic.starts_with("v_cvt_f32_fp8"))
    return true;

  // The eight K=128 FP8/BF8 forms and the standalone 32x16 FP4 WMMA exist on B0
  // but not A0, so they require semantic expansion. The common f32 K=128 forms
  // use one neutral regular-Scale mixed-format operation. Source fields with no
  // meaning for these opcodes are discarded while constructing the target.
  // The standalone 32x16 FP4 form splits into two scaled M=16 halves. Packed-f16
  // K=128 forms lower through an f32 accumulator and pack the final result.
  const bool is_k128_fp8_bf8 = (mnemonic.starts_with("v_wmma_f16_16x16x128_") ||
                                mnemonic.starts_with("v_wmma_f32_16x16x128_")) &&
                               (mnemonic.ends_with("_fp8_fp8") || mnemonic.ends_with("_fp8_bf8") ||
                                mnemonic.ends_with("_bf8_fp8") || mnemonic.ends_with("_bf8_bf8"));
  if (is_k128_fp8_bf8 || mnemonic == "v_wmma_f32_32x16x128_f4")
    return true;

  // A0 trap/CWSR recovery requires every low-precision F8F6F4 WMMA to carry its
  // load-scale prefix, even when the requested scale is 1.0. Standalone input
  // is therefore wrapped with inline-zero neutral E8M0 scales. Native M=16
  // Scale16 is retained with its unused prefix field normalized. B0-only M=32
  // scaled forms split into two native M=16 operations.
  if (mnemonic == "v_wmma_f32_16x16x128_f8f6f4" || mnemonic.starts_with("v_wmma_scale"))
    return true;

  // K=64 FP8/BF8 WMMA is present on A0 and retains its architectural encoding.
  // It stays on the ordinary copy path. It is distinct from the scale-capable
  // K=128 F8F6F4 matrix body: only that body consumes an immediately preceding
  // LD_SCALE. The A0 profile therefore wraps bare F8F6F4 input in the scaled
  // four-DWORD form, while native K=64 FP8/BF8 remains unscaled.
  //
  // FP8/BF8 SWMMAC is present on both A0 and B0. Unlike dense K=128 WMMA,
  // the gfx1250 A0-to-B0 change table does not classify these sparse forms as
  // B0 additions, and their opcodes remain inside the A0 seven-bit VOP3P
  // opcode field. They therefore stay on the same-stepping byte-copy path.

  // The A0 co-execution distance exceeds B0 only for integer IU8/IU4 WMMA or
  // SWMMAC. FP16/BF16 need four spacing slots on both revisions, while floating
  // FP8 forms need no additional A0 padding. The implemented long-K IU8 forms
  // use conservative fixed padding; other integer forms fail closed pending a
  // CFG-aware spacing pass that can inspect following instructions.
  const bool is_wmma_like = mnemonic.starts_with("v_wmma_") || mnemonic.starts_with("v_swmmac_");
  return is_wmma_like && (mnemonic.find("_iu8") != std::string_view::npos ||
                          mnemonic.find("_iu4") != std::string_view::npos);
}

/// @brief True when a B0 FP8 conversion selects the B0-only E5M3 mode.
///
/// @details The affected VOP3 conversions reuse CLAMP as the E5M3 selector on
/// B0. A0 implements the same CLAMP=0 E4M3 operation, so those instructions
/// must remain on the ordinary byte-copy path. CLAMP lives in the eight-byte
/// VOP3 base encoding; a trailing literal increases the decoded size without
/// moving that field.
[[nodiscard]] bool requires_fp8_clamp_emulation(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  const bool affected = mnemonic == "v_cvt_pk_fp8_f32" || mnemonic == "v_cvt_sr_fp8_f32" ||
                        mnemonic.starts_with("v_cvt_f32_fp8");
  const bool is_vop3 = inst.encoding_id() >= cdna5::encoding::kVop3 &&
                       inst.encoding_id() <= cdna5::encoding::kVop3OpHi6;
  if (!affected || !is_vop3 || inst.size() < static_cast<int>(sizeof(cdna5::Vop3MachineInst)) ||
      inst.raw_encoding() == nullptr)
    return false;

  cdna5::Vop3MachineInst encoding{};
  std::memcpy(&encoding, inst.raw_encoding(), sizeof(encoding));
  return encoding.clamp != 0;
}

[[nodiscard]] bool is_wmma_completion_wait(const Instruction &inst) {
  if (inst.size() != static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr ||
      inst.mnemonic() != "s_wait_alu")
    return false;

  // S_WAIT_ALU SIMM16[15:12] is VA_VDST. Zero waits for all outstanding
  // destination writes, regardless of any additional counters the source
  // instruction also drains.
  constexpr uint32_t kVaVdstMask = 0xf000u;
  return (inst.raw_encoding()[0] & kVaVdstMask) == 0;
}

[[nodiscard]] bool has_wmma_completion_wait_ahead(
    const Instruction &inst,
    const std::unordered_map<uint64_t, const Instruction *> &source_instruction_by_offset,
    const Gfx1250VgprMsbAnalysis &vgpr_msb) {
  RegisterSet pending_defs =
      InstDefUse(inst, &vgpr_msb, UnknownVgprDefPolicy::ExpandAll).defs.ordinary_only();
  uint64_t next_offset = inst.src_loc() + inst.size();
  while (true) {
    const auto next_it = source_instruction_by_offset.find(next_offset);
    if (next_it == source_instruction_by_offset.end())
      return false;

    const Instruction &next = *next_it->second;
    constexpr uint64_t kControlTransferOrTerminator =
        BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR;
    if ((next.flags() & kControlTransferOrTerminator) != 0)
      return false;
    if (is_wmma_completion_wait(next))
      return true;

    // Bound the scan to affected WMMA and the exact scalar instructions that
    // generated split forms may place before their trailing completion wait.
    // The def/use check below independently verifies VGPR transparency for
    // both generated and hand-written source streams.
    const std::string_view mnemonic = next.mnemonic();
    const bool canonical_split_scaffolding =
        mnemonic == "s_wait_xcnt" || mnemonic == "s_set_vgpr_msb";
    if (!canonical_split_scaffolding && !gfx1250_b0_to_a0_requires_wmma_completion_wait(next))
      return false;

    const InstDefUse access(next, &vgpr_msb, UnknownVgprDefPolicy::ExpandAll);
    if (access.uses.intersects(pending_defs))
      return false;
    pending_defs |= access.defs.ordinary_only();

    if (next.size() <= 0)
      return false;
    next_offset += static_cast<uint64_t>(next.size());
  }
}

} // namespace

bool gfx1250_b0_to_a0_requires_wmma_completion_wait(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  if (!mnemonic.starts_with("v_wmma_"))
    return false;

  return mnemonic.find("_fp8") != std::string_view::npos ||
         mnemonic.find("_bf8") != std::string_view::npos ||
         mnemonic.find("_f8f6f4") != std::string_view::npos || mnemonic.ends_with("_f4");
}

void gfx1250_b0_to_a0_append_wmma_completion_wait_if_needed(
    const Instruction &inst,
    const std::unordered_map<uint64_t, const Instruction *> &source_instruction_by_offset,
    const Gfx1250VgprMsbAnalysis &vgpr_msb, std::vector<uint32_t> &words) {
  if (!gfx1250_b0_to_a0_requires_wmma_completion_wait(inst) ||
      has_wmma_completion_wait_ahead(inst, source_instruction_by_offset, vgpr_msb)) {
    return;
  }

  // TODO: Replace this conservative producer-side drain with block-local
  // S_WAIT_ALU analysis over the final translated instruction stream. It should
  // track VA_VDST/VM_VSRC dependencies and wait only before a dependent use.
  // B0 code may schedule scaled WMMA consumers under SCHED_MODE 2 using B0
  // completion timing. A0 has a lower FP8/FP4 WMMA issue rate, and mode 2 makes
  // software responsible for VA_VDST dependencies. Wait for pending VALU
  // destinations without draining unrelated ALU dependency counters.
  // The no-wait default is 0xff9f; clearing only VA_VDST[15:12] gives 0x0f9f.
  constexpr uint16_t kWaitVaVdstZero = 0x0f9f;
  words.push_back(cdna5::build_sopp(cdna5::kSWaitAluSopp, {.simm16 = kWaitVaVdstZero})[0]);
}

bool gfx1250_b0_to_a0_is_deferred_family(std::string_view mnemonic) {
  // s_sleep and s_sleep_var are deliberately absent. They behave identically on
  // A0 and B0. Only s_monitor_sleep('forever') with MWAIT=0 requires an A0
  // translation. Copying a plain sleep through is the correct translation, not
  // an unimplemented one, so reporting
  // it said nothing and buried the reports that do name a real gap -- one RCCL
  // all_reduce run emitted 104,831 of them.
  return mnemonic == "s_get_barrier_state" || mnemonic == "s_monitor_sleep";
}

const InstructionLegalization *gfx1250_b0_to_a0_legalization(const Instruction &inst) {
  // CLAMP=0 is the common E4M3 operation on both steppings. CLAMP=1 selects
  // the B0-only E5M3 behavior and therefore requires a semantic expansion.
  const std::string_view mnemonic = inst.mnemonic();
  const bool fp8_clamp_family = mnemonic == "v_cvt_pk_fp8_f32" || mnemonic == "v_cvt_sr_fp8_f32" ||
                                mnemonic.starts_with("v_cvt_f32_fp8");
  if (fp8_clamp_family && !requires_fp8_clamp_emulation(inst))
    return nullptr;

  if (!requires_b0_to_a0_expansion(inst.mnemonic())) {
    return nullptr;
  }

  // The runtime uses only the action and target opcode for this revision-specific
  // classification. Source keys remain zero because matching is performed on
  // the fully decoded mnemonic, which is necessary for contextual gfx1250
  // variants that share structural opcode fields.
  static constexpr InstructionLegalization kExpand{
      .src_opcode = 0,
      .src_encoding_id = 0,
      .action = Action::Expand,
      .target_opcode = 0,
  };
  return &kExpand;
}

} // namespace rocjitsu
