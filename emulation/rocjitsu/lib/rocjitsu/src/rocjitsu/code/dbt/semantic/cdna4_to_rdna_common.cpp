// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/cdna4_to_rdna_common.cpp
/// @brief Lowerings shared by CDNA4-to-RDNA semantic rule tables.

#include "rocjitsu/code/dbt/semantic/cdna4_to_rdna_common.h"
#include "rocjitsu/code/dbt/semantic/rules.h"

#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/dbt/translation_rule.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/isa/instruction.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

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

// These CDNA4 selectors name a different register, or have no source selector, on RDNA3/4.
bool has_no_rdna_count_selector(uint16_t selector) {
  using CdnaSrc = cdna4::OpSelSrcSimple;
  switch (selector) {
  case CdnaSrc::OPR_SRC_SIMPLE_FLAT_SCRATCH_LO:
  case CdnaSrc::OPR_SRC_SIMPLE_FLAT_SCRATCH_HI:
  case CdnaSrc::OPR_SRC_SIMPLE_XNACK_MASK_LO:
  case CdnaSrc::OPR_SRC_SIMPLE_XNACK_MASK_HI:
  case CdnaSrc::OPR_SRC_SIMPLE_SRC_POPS_EXITING_WAVE_ID:
  case CdnaSrc::OPR_SRC_SIMPLE_SRC_VCCZ:
    return true;
  default:
    return false;
  }
}

bool cdna4_reverse_shift_needs_rewrite(const Instruction &inst) {
  // CDNA4 VOP2 encoding IDs incorporate the opcode and the high VDST bits.
  if (inst.encoding_id() >= cdna4::encoding::kVopc || !inst.raw_encoding())
    return false;
  switch (inst.opcode()) {
  case cdna4::kVLshlrevB16Vop2:
  case cdna4::kVLshrrevB16Vop2:
  case cdna4::kVAshrrevI16Vop2:
    return true;
  case cdna4::kVLshlrevB32Vop2:
  case cdna4::kVLshrrevB32Vop2:
  case cdna4::kVAshrrevI32Vop2:
    return (*inst.raw_encoding() & 0x1ffu) == amdgpu::SRC_DPP;
  default:
    return false;
  }
}

template <rj_code_arch_t Host>
ExpandResult lower_cdna4_reverse_shift(const Instruction &inst, uint64_t, std::span<const uint8_t>,
                                       const LivenessAnalysis &, TranslationContext &) {
  cdna4::Vop2MachineInst src{};
  std::memcpy(&src, inst.raw_encoding(), sizeof(src));
  if (src.src0 == amdgpu::SRC_DPP)
    return ExpandResult::failed(
        std::string(inst.mnemonic()) +
        " DPP permutes the shifted value on CDNA4 and the count on RDNA3/4");
  if (src.src0 == amdgpu::SRC_SDWA)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " SDWA halfword shift has no equivalent RDNA3/4 encoding");
  if (has_no_rdna_count_selector(src.src0))
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " shift count has no equivalent RDNA source selector");

  // VOP2 selects the low 16 bits and produces a zero-extended DWORD on CDNA4.
  // RDNA's VOP3 shift preserves the upper half, so explicitly clear it afterwards.
  constexpr uint16_t inline_zero = 128;
  uint16_t count = src.src0;
  if (const auto bits = inst.src_operand(0)->const_value())
    count = inline_zero + (*bits & 15u);
  else if (count == cdna4::OpSelSrcSimple::OPR_SRC_SIMPLE_SRC_EXECZ)
    count = inline_zero;
  else if (count == amdgpu::kLegacyM0Selector)
    count = amdgpu::kModernM0Selector;

  uint16_t opcode;
  switch (inst.opcode()) {
  case cdna4::kVLshlrevB16Vop2:
    opcode = Host == ROCJITSU_CODE_ARCH_RDNA3 ? rdna3::kVLshlrevB16Vop3 : rdna4::kVLshlrevB16Vop3;
    break;
  case cdna4::kVLshrrevB16Vop2:
    opcode = Host == ROCJITSU_CODE_ARCH_RDNA3 ? rdna3::kVLshrrevB16Vop3 : rdna4::kVLshrrevB16Vop3;
    break;
  default:
    opcode = Host == ROCJITSU_CODE_ARCH_RDNA3 ? rdna3::kVAshrrevI16Vop3 : rdna4::kVAshrrevI16Vop3;
    break;
  }
  const auto shift = build_rdna_vop3(Host, opcode, src.vdst, count, 256 + src.vsrc1);
  constexpr uint16_t extract =
      Host == ROCJITSU_CODE_ARCH_RDNA3 ? rdna3::kVBfeU32Vop3 : rdna4::kVBfeU32Vop3;
  const auto extend =
      build_rdna_vop3(Host, extract, src.vdst, 256 + src.vdst, inline_zero, inline_zero + 16);
  return ExpandResult::success({shift[0], shift[1], extend[0], extend[1]});
}

template <rj_code_arch_t Host>
constexpr RegisteredInstructionRewrite kReverseShiftRules[] = {{"CDNA4 VOP2 reverse shifts",
                                                                cdna4_reverse_shift_needs_rewrite,
                                                                lower_cdna4_reverse_shift<Host>,
                                                                false,
                                                                {}}};

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

  using CdnaSrc = cdna4::OpSelSrcSimple;
  if (has_no_rdna_count_selector(src1))
    return ExpandResult::failed(
        "v_lshl_add_u64 shift count has no equivalent RDNA source selector");

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
  constexpr uint16_t kVgprBase = 256;
  const bool src2_is_vgpr_pair = src2 >= kVgprBase && src2 <= 510;
  const bool src2_is_sgpr_pair = src2 <= 104 && (src2 % 2) == 0;
  const bool src2_is_vcc = src2 == kVccLo;
  if (!src2_is_vgpr_pair && !src2_is_sgpr_pair && !src2_is_vcc)
    return ExpandResult::failed(
        std::string(inst.mnemonic()) +
        " v_lshl_add_u64 lowering does not support a non-register 64-bit "
        "addend (inline constant, literal, or v255 whose high half cannot "
        "encode a VGPR); the high-half derivation requires a register pair");

  // CDNA4 ISA section 3.6.4 requires even VGPR bases for 64-bit operands. S1 is a 32-bit count.
  if (vdst % 2 != 0 || (src0 >= kVgprBase && src0 % 2 != 0) || (src2_is_vgpr_pair && src2 % 2 != 0))
    return ExpandResult::failed(
        std::string(inst.mnemonic()) +
        " requires even VGPR bases for the 64-bit destination, value and addend");

  // Lower the CDNA shift into a 64-bit temporary followed by a carry-add of S2.
  // Preserve the addend until both halves are consumed. Choosing the shift
  // destination:
  //   - Normally shift directly into vdst. vdst overlapping S0 is safe because
  //     v_lshlrev_b64 reads all of S0 before writing.
  //   - If vdst overlaps a VGPR S2, shifting into vdst would destroy S2 first, so
  //     shift into a dead scratch pair instead and add scratch + S2 into vdst.
  const bool vdst_aliases_src2 = src2 == kVgprBase + vdst;

  // A site-dead register can still receive an earlier memory result. Prefer scratch
  // unused by the whole kernel. Drain outstanding operations before borrowing a
  // site-dead register when all candidate registers occur somewhere in the kernel.
  bool wait_for_scratch = false;
  auto find_vgpr_scratch = [&](uint16_t count,
                               uint16_t alignment) -> FailureOr<std::optional<uint16_t>> {
    FailureOr<std::optional<uint16_t>> unused =
        liveness.find_globally_unused_vgpr_run(&inst, count, 0, alignment);
    if (unused.failed() || unused.value())
      return unused;
    FailureOr<bool> complete = liveness.global_vgpr_usage_is_complete();
    if (complete.failed())
      return Result::failure();
    // Relative operands can hide a live value anywhere in the source allocation.
    const uint16_t start = complete.value() ? 0 : static_cast<uint16_t>(context.num_vgprs);
    wait_for_scratch = true;
    return liveness.find_free_run(&inst, count, start, alignment);
  };

  uint16_t shift_dst = vdst;
  if (vdst_aliases_src2) {
    // Need a dead, even-aligned VGPR pair that overlaps neither S0/S2 nor vdst.
    FailureOr<std::optional<uint16_t>> maybe_scratch =
        find_vgpr_scratch(/*count=*/2, /*alignment=*/2);
    if (maybe_scratch.failed())
      return ExpandResult::liveness_unavailable();
    const std::optional<uint16_t> scratch = maybe_scratch.value();
    if (!scratch)
      return ExpandResult::failed(
          std::string(inst.mnemonic()) +
          " v_lshl_add_u64 lowering needs a scratch VGPR pair when the destination overlaps the "
          "addend, but no dead pair is available");
    shift_dst = *scratch;
    // Scratch above the source allocation must be included in the target descriptor.
    context.require_vgprs(static_cast<uint32_t>(shift_dst) + 2u);
  }

  // The carry chain needs a scalar carry destination. v_lshl_add_u64 defines no
  // carry output, and VCC is NOT liveness-tracked here, so hardcoding VCC would
  // silently clobber a live VCC that the surrounding code still depends on.
  // Allocate a dead ordinary SGPR pair instead and grow the target descriptor to
  // cover it, matching the EXEC-save precedent in the MFMA lowering.
  FailureOr<std::optional<uint16_t>> maybe_carry_sgpr_opt =
      liveness.find_globally_unused_sgpr_pair(&inst);
  if (maybe_carry_sgpr_opt.succeeded() && !maybe_carry_sgpr_opt.value()) {
    FailureOr<bool> complete = liveness.global_sgpr_usage_is_complete();
    if (complete.failed())
      return ExpandResult::liveness_unavailable();
    const uint16_t start = complete.value() ? 0 : static_cast<uint16_t>(context.num_sgprs);
    maybe_carry_sgpr_opt = liveness.find_free_sgpr_pair(&inst, start);
    wait_for_scratch = true;
  }
  if (maybe_carry_sgpr_opt.failed())
    return ExpandResult::liveness_unavailable();
  const std::optional<uint16_t> carry_sgpr_opt = maybe_carry_sgpr_opt.value();
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

  constexpr uint16_t kInlineZero = 128;
  uint16_t host_count;
  if (src1 == CdnaSrc::OPR_SRC_SIMPLE_SRC_EXECZ) {
    // An executing VALU lane necessarily has nonzero EXEC.
    host_count = kInlineZero;
  } else if (const auto value = inst.src_operand(1)->const_value()) {
    const uint16_t shift = *value & 7u;
    host_count = kInlineZero + (shift <= 4 ? shift : 0);
  } else {
    // Use the low half of the shift destination while the addend is protected.
    // If it holds part of S0, the unshifted input must survive until the B64 shift.
    uint16_t count_dst = shift_dst;
    if (src0 >= kVgprBase && kVgprBase + count_dst >= src0 && kVgprBase + count_dst <= src0 + 1) {
      FailureOr<std::optional<uint16_t>> scratch = find_vgpr_scratch(/*count=*/1, /*alignment=*/1);
      if (scratch.failed())
        return ExpandResult::liveness_unavailable();
      if (!scratch.value())
        return ExpandResult::failed(
            "v_lshl_add_u64 needs a dead VGPR to normalize the shift count");
      count_dst = *scratch.value();
      context.require_vgprs(count_dst + 1u);
    }
    host_count = kVgprBase + count_dst;
    // Each nibble in 0x00043210 is the shift for one value of S1[2:0]:
    // 0, 1, 2, 3, 4, 0, 0, 0. V_BFE_U32 uses the low five bits of its
    // bit offset, so multiplying S1 by four selects the corresponding nibble.
    const uint16_t shl =
        host_arch == ROCJITSU_CODE_ARCH_RDNA3 ? rdna3::kVLshlrevB32Vop3 : rdna4::kVLshlrevB32Vop3;
    const uint16_t bfe =
        host_arch == ROCJITSU_CODE_ARCH_RDNA3 ? rdna3::kVBfeU32Vop3 : rdna4::kVBfeU32Vop3;
    const uint16_t count_src =
        src1 == CdnaSrc::OPR_SRC_SIMPLE_M0 ? amdgpu::kModernM0Selector : src1;
    const auto offset = build_rdna_vop3(host_arch, shl, count_dst, kInlineZero + 2, count_src);
    words.insert(words.end(), offset.begin(), offset.end());
    const auto extract =
        build_rdna_vop3(host_arch, bfe, count_dst, /*literal=*/255, host_count, kInlineZero + 4);
    words.insert(words.end(), extract.begin(), extract.end());
    words.push_back(0x00043210);
  }

  {
    auto [w0, w1] = build_rdna_vop3(host_arch, lshlrev_b64_op, static_cast<uint8_t>(shift_dst),
                                    host_count, src0);
    words.push_back(w0);
    words.push_back(w1);
  }

  // If S2 is VCC, wait for a previous VALU write before reading the addend.
  if (host_arch == ROCJITSU_CODE_ARCH_RDNA4 && src2_is_vcc)
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

  // VA_SDST=0 waits for the low add's scalar write before the high add reads the carry.
  // On GFX11 wave64, an earlier high add may have read the same pair as a lane mask.
  if (host_arch == ROCJITSU_CODE_ARCH_RDNA3)
    words.push_back(rdna3::build_sopp(rdna3::kSWaitcntDepctrSopp, {.simm16 = 0xF19F})[0]);
  else
    words.push_back(rdna4::build_sopp(rdna4::kSWaitAlu, {.simm16 = 0xF1FF})[0]);

  {
    auto [w0, w1] = build_rdna_vop3_sdst(
        host_arch, add_co_ci_u32_op, static_cast<uint8_t>(vdst + 1), carry_sgpr,
        static_cast<uint16_t>(256 + shift_dst + 1), static_cast<uint16_t>(src2 + 1), carry_sgpr);
    words.push_back(w0);
    words.push_back(w1);
  }

  if (wait_for_scratch) {
    const auto wait = host_arch == ROCJITSU_CODE_ARCH_RDNA3
                          ? rdna3::build_sopp(rdna3::kSWaitIdleSopp)
                          : rdna4::build_sopp(rdna4::kSWaitIdleSopp);
    words.insert(words.begin(), wait[0]);
  }
  return ExpandResult::success(std::move(words));
}

} // namespace

std::span<const RegisteredInstructionRewrite>
cdna4_reverse_shift_rules_for_rdna(rj_code_arch_t host_arch) {
  if (host_arch == ROCJITSU_CODE_ARCH_RDNA3)
    return kReverseShiftRules<ROCJITSU_CODE_ARCH_RDNA3>;
  return kReverseShiftRules<ROCJITSU_CODE_ARCH_RDNA4>;
}

ExpandResult expand_cdna4_v_lshl_add_u64_for_rdna(const Instruction &inst, uint32_t host_arch,
                                                  uint64_t, std::span<const uint8_t>,
                                                  const LivenessAnalysis &liveness,
                                                  TranslationContext &context, const LaneLayout *,
                                                  const LaneLayout *) {
  return lower_v_lshl_add_u64(inst, static_cast<rj_code_arch_t>(host_arch), liveness, context);
}

} // namespace rocjitsu
