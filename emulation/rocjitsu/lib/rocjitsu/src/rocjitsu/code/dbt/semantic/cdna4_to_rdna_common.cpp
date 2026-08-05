// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/cdna4_to_rdna_common.cpp
/// @brief Lowerings shared by CDNA4-to-RDNA semantic rule tables.

#include "rocjitsu/code/dbt/semantic/cdna4_to_rdna_common.h"

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/dbt/translation_rule.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/opcodes.h"
#include "rocjitsu/isa/instruction.h"

#include <array>
#include <cstring>
#include <string>
#include <utility>

namespace rocjitsu {
namespace {

/// @brief Build the explicit-SDST VOP3 form used by the carry chain.
/// @details RDNA3 and RDNA4 currently share the layout, but selecting the
/// generated builder by host ISA prevents a future XML change from silently
/// encoding one target with the other target's format.
[[nodiscard]] std::array<uint32_t, 2> build_rdna_vop3_sdst(rj_code_arch_t host_arch, uint16_t op,
                                                           uint8_t vdst, uint8_t sdst,
                                                           uint16_t src0, uint16_t src1 = 0,
                                                           uint16_t src2 = 0) {
  if (host_arch == ROCJITSU_CODE_ARCH_RDNA3) {
    return rdna3::build_vop3_sdst_enc(
        op, {.vdst = vdst, .sdst = sdst, .src0 = src0, .src1 = src1, .src2 = src2});
  }
  return rdna4::build_vop3_sdst_enc(
      op, {.vdst = vdst, .sdst = sdst, .src0 = src0, .src1 = src1, .src2 = src2});
}

/// @brief Build a plain VOP3 instruction, selecting the builder by host ISA.
[[nodiscard]] std::array<uint32_t, 2> build_rdna_vop3(rj_code_arch_t host_arch, uint16_t op,
                                                      uint8_t vdst, uint16_t src0,
                                                      uint16_t src1 = 0, uint16_t src2 = 0) {
  if (host_arch == ROCJITSU_CODE_ARCH_RDNA3)
    return rdna3::build_vop3(op, {.vdst = vdst, .src0 = src0, .src1 = src1, .src2 = src2});
  return rdna4::build_vop3(op, {.vdst = vdst, .src0 = src0, .src1 = src1, .src2 = src2});
}

ExpandResult lower_v_lshl_add_u64(const Instruction &inst, rj_code_arch_t host_arch,
                                  const LivenessAnalysis &liveness, TranslationContext &context) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 8)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " cannot lower v_lshl_add_u64 without a complete 64-bit raw "
                                "encoding");

  cdna4::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const uint16_t vdst = src.vdst;
  const uint16_t src0 = src.src0;
  const uint16_t src1 = src.src1;
  const uint16_t src2 = src.src2;

  if (host_arch != ROCJITSU_CODE_ARCH_RDNA3 && host_arch != ROCJITSU_CODE_ARCH_RDNA4)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " v_lshl_add_u64 lowering only supports RDNA3/RDNA4 hosts");
  if (vdst == 255)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " v_lshl_add_u64 lowering needs two destination VGPRs but vdst "
                                "is v255");

  // A two-word literal shift count would require carrying the literal dword
  // through the emitted v_lshlrev_b64; that operand form is not decoded here.
  if (src1 == 254 || src1 == 255)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " v_lshl_add_u64 lowering does not support a literal shift count");

  // The 64-bit addend S2 is added as two 32-bit halves read from src2 and
  // src2 + 1. That +1 high-half derivation is only valid when src2 names a real
  // 64-bit register pair: a VGPR pair (256..510, so the high half src2+1 is still
  // a VGPR <= 511 -- v255/511 is rejected because 512 does not encode a VGPR), an
  // even SGPR pair (0..104), or VCC (106/107). For an inline constant (128..208,
  // 240..247) the encodings are consecutive single values, so src2 + 1 would read
  // the NEXT constant (e.g. addend 0 encoded as 128 -> high half reads 129 ==
  // constant 1), and a literal (254/255) has no second encoded dword. Reject
  // those non-register addends rather than silently miscompiling; S0 is fed whole
  // to v_lshlrev_b64 so it may be any 64-bit source form and needs no such guard.
  constexpr uint16_t kVccLo = 106;
  const bool src2_is_vgpr_pair = src2 >= 256 && src2 <= 510;
  const bool src2_is_sgpr_pair = src2 <= 104 && (src2 % 2) == 0;
  const bool src2_is_vcc = src2 == kVccLo;
  if (!src2_is_vgpr_pair && !src2_is_sgpr_pair && !src2_is_vcc)
    return ExpandResult::failed(
        std::string(inst.mnemonic()) +
        " v_lshl_add_u64 lowering does not support a non-register 64-bit "
        "addend (inline constant, literal, or v255 whose high half cannot "
        "encode a VGPR); the high-half derivation requires a register pair");

  // v_lshl_add_u64 computes D = (S0 << S1[5:0]) + S2, lowered as a shift into a
  // 64-bit temporary followed by a 64-bit carry-add of S2. The shift result must
  // not clobber the addend S2 before the add reads it. Choosing the shift
  // destination:
  //   - Normally shift directly into vdst. vdst overlapping S0 is safe because
  //     v_lshlrev_b64 reads all of S0 before writing.
  //   - If vdst overlaps a VGPR S2, shifting into vdst would destroy S2 first, so
  //     shift into a dead scratch pair instead and add scratch + S2 into vdst.
  const bool src2_is_vgpr = src2_is_vgpr_pair;
  const uint16_t src2_vgpr = src2_is_vgpr ? static_cast<uint16_t>(src2 - 256) : 0;
  const bool vdst_aliases_src2 = src2_is_vgpr && vdst < static_cast<uint16_t>(src2_vgpr + 2) &&
                                 src2_vgpr < static_cast<uint16_t>(vdst + 2);

  uint16_t shift_dst = vdst;
  if (vdst_aliases_src2) {
    // Need a dead, even-aligned VGPR pair that overlaps neither S0/S2 nor vdst.
    const auto scratch = liveness.find_free_run(&inst, /*count=*/2, /*search_start=*/0,
                                                /*base_alignment=*/2);
    if (!scratch)
      return ExpandResult::failed(
          std::string(inst.mnemonic()) +
          " v_lshl_add_u64 lowering needs a scratch VGPR pair when the destination overlaps the "
          "addend, but no dead pair is available");
    shift_dst = *scratch;
    // find_free_run can return a dead pair above the source descriptor's VGPR
    // allocation. Report the requirement so the target descriptor grows to cover
    // the scratch pair; otherwise the emitted code names unallocated VGPRs.
    context.require_vgprs(static_cast<uint32_t>(shift_dst) + 2u);
  }

  // The carry chain needs a scalar carry destination. v_lshl_add_u64 defines no
  // carry output, and VCC is NOT liveness-tracked here, so hardcoding VCC would
  // silently clobber a live VCC that the surrounding code still depends on.
  // Allocate a dead ordinary SGPR pair instead and grow the target descriptor to
  // cover it, matching the EXEC-save precedent in the MFMA lowering.
  const auto carry_sgpr_opt = liveness.find_free_sgpr_pair(&inst);
  if (!carry_sgpr_opt)
    return ExpandResult::failed(
        std::string(inst.mnemonic()) +
        " v_lshl_add_u64 lowering could not find a free SGPR pair for the add carry");
  const uint8_t carry_sgpr = static_cast<uint8_t>(*carry_sgpr_opt);
  context.require_sgprs(static_cast<uint32_t>(carry_sgpr) + 2u);

  // These generated opcodes currently match on RDNA3 and RDNA4. Keep the
  // architecture-specific selection explicit so an ISA XML change cannot
  // silently make this shared lowering emit the other target's opcode.
  const uint16_t lshlrev_b64_op =
      host_arch == ROCJITSU_CODE_ARCH_RDNA3 ? rdna3::kVLshlrevB64Vop3 : rdna4::kVLshlrevB64Vop3;
  const uint16_t add_co_u32_op = host_arch == ROCJITSU_CODE_ARCH_RDNA3
                                     ? rdna3::kVAddCoU32Vop3SdstEnc
                                     : rdna4::kVAddCoU32Vop3SdstEnc;
  const uint16_t add_co_ci_u32_op = host_arch == ROCJITSU_CODE_ARCH_RDNA3
                                        ? rdna3::kVAddCoCiU32Vop3SdstEnc
                                        : rdna4::kVAddCoCiU32Vop3SdstEnc;

  std::vector<uint32_t> words;

  // shift_dst = S0 << S1[5:0]. v_lshlrev_b64 takes the shift count as src0 and
  // the 64-bit value as src1.
  {
    auto [w0, w1] =
        build_rdna_vop3(host_arch, lshlrev_b64_op, static_cast<uint8_t>(shift_dst), src1, src0);
    words.push_back(w0);
    words.push_back(w1);
  }

  // GFX12 requires an explicit VALU dependency wait before the shifted value is
  // consumed by the following add. GFX11 relies on hardware scoreboarding.
  if (host_arch == ROCJITSU_CODE_ARCH_RDNA4)
    words.push_back(rdna4::build_sopp(rdna4::kSWaitAlu, {.simm16 = 0xFFFD})[0]);

  // vdst = shift_dst + S2. v_add_co_u32 writes the carry into the dead SGPR pair,
  // then v_add_co_ci_u32 consumes it as carry-in. The low add reads the freshly
  // shifted low half.
  {
    auto [w0, w1] = build_rdna_vop3_sdst(host_arch, add_co_u32_op, static_cast<uint8_t>(vdst),
                                         carry_sgpr, static_cast<uint16_t>(256 + shift_dst), src2);
    words.push_back(w0);
    words.push_back(w1);
  }

  // GFX12 requires an explicit VALU dependency wait before the carry read.
  // GFX11 handles this pair through hardware scoreboarding, and SOPP opcode 8
  // is not s_wait_alu there, so only emit it for RDNA4.
  if (host_arch == ROCJITSU_CODE_ARCH_RDNA4)
    words.push_back(rdna4::build_sopp(rdna4::kSWaitAlu, {.simm16 = 0xFFFD})[0]);

  {
    auto [w0, w1] = build_rdna_vop3_sdst(
        host_arch, add_co_ci_u32_op, static_cast<uint8_t>(vdst + 1), carry_sgpr,
        static_cast<uint16_t>(256 + shift_dst + 1), static_cast<uint16_t>(src2 + 1), carry_sgpr);
    words.push_back(w0);
    words.push_back(w1);
  }

  return ExpandResult::success(std::move(words));
}

} // namespace

ExpandResult expand_cdna4_v_lshl_add_u64_for_rdna(const Instruction &inst, uint32_t host_arch,
                                                  uint64_t, std::span<const uint8_t>,
                                                  const LivenessAnalysis &liveness,
                                                  TranslationContext &context, const LaneLayout *,
                                                  const LaneLayout *) {
  return lower_v_lshl_add_u64(inst, static_cast<rj_code_arch_t>(host_arch), liveness, context);
}

} // namespace rocjitsu
