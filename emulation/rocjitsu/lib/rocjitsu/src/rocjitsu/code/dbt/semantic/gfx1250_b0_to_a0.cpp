// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/gfx1250_b0_to_a0.cpp
/// @brief Handwritten semantic expansions for gfx1250 B0-to-A0 errata.

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/dbt/semantic/rules.h"
#include "rocjitsu/code/dbt/semantic_scratch.h"
#include "rocjitsu/code/dbt/translation_rule.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/builders.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/opcodes.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu {

namespace {

/// @brief gfx1250 special-scalar operand encodings.
/// @details CRITICAL: on gfx1250 these are the INVERSE of CDNA — M0 = 125 and
/// NULL = 124, whereas CDNA encodes M0 = 124. Every hand-written encoding below
/// must use kGfx1250M0 where the machine reads/writes M0 and kGfx1250Null only
/// where a discarded/zero NULL operand is genuinely intended. See
/// gfx1250/operand_types.h (OPR_SRC_NULL = 124, OPR_SRC_M0 = 125).
constexpr uint8_t kGfx1250Null = 124;
constexpr uint8_t kGfx1250M0 = 125;

/// @brief VOP3P opcode of the WMMA-scale prefix half of a scaled-WMMA pair.
/// @details The scale prefix that fuses with a following WMMA is not a standalone
/// named VOP3P op in the generated opcode table (it is a structural VOP3PX2/PX3
/// prefix), so its opcode is named here rather than pulled from gfx1250 opcodes.
/// kWmmaScaleSrc2PrefixOp is the VOP3PX2 scale-src2 prefix; kWmmaScale16PrefixOp
/// is the VOP3PX3 Scale16 prefix.
constexpr uint16_t kWmmaScaleSrc2PrefixOp = 0x35;
constexpr uint16_t kWmmaScale16PrefixOp = 0x3a;

/// @brief Append a generated instruction's words to one replacement sequence.
template <size_t N>
void append_words(std::vector<uint32_t> &output, const std::array<uint32_t, N> &words) {
  output.insert(output.end(), words.begin(), words.end());
}

/// @brief Change the gfx1250 VGPR-bank mode while preserving trap recovery state.
///
/// @details SIMM16[7:0] selects the new SRC0/SRC1/SRC2/DST banks. The gfx1250
/// trap convention stores the immediately preceding mode in SIMM16[15:8]. If
/// this is the first instruction in a generated sequence, conservatively place
/// an S_NOP in front of it: the source-stream predecessor is outside the
/// expansion and may be an S_SETREG* write to MODE, which must not immediately
/// precede S_SET_VGPR_MSB. Once an expansion has emitted any instruction, that
/// instruction already provides the required separation.
///
/// An S_WAIT_XCNT 0 is emitted immediately before each S_SET_VGPR_MSB: changing
/// the VGPR-bank selection while cross-lane/memory work (XCNT) is still
/// outstanding could let instruction replay observe a different VGPR mapping.
/// Draining XCNT first makes the bank change observable to a consistent register
/// view. The S_WAIT_XCNT precedes the S_SET_VGPR_MSB and does not affect the
/// S_NOP separation from a preceding MODE write.
void append_gfx1250_vgpr_msb_transition(std::vector<uint32_t> &words, uint8_t &current_mode,
                                        uint8_t new_mode) {
  if (current_mode == new_mode)
    return;

  if (words.empty())
    append_words(words, gfx1250::build_sopp(gfx1250::kSNopSopp, {.simm16 = 0}));

  append_words(words, gfx1250::build_sopp(gfx1250::kSWaitXcntSopp, {.simm16 = 0}));
  const uint16_t immediate =
      static_cast<uint16_t>(new_mode) | (static_cast<uint16_t>(current_mode) << 8);
  append_words(words, gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = immediate}));
  current_mode = new_mode;
}

/// @brief Conservatively remove one hard-clause scheduling directive.
///
/// @details A legal S_CLAUSE has no architectural data result; it only groups
/// following instructions for issue. DBT transformations can change clause
/// membership and placement, and rocjitsu does not currently revalidate those
/// constraints. Replacing every clause with a same-size S_NOP is functionally
/// conservative. A future performance pass may retain clauses after proving
/// they remain valid in the translated control flow.
ExpandResult expand_gfx1250_s_clause(const Instruction &inst, uint32_t, uint64_t,
                                     std::span<const uint8_t>, const LivenessAnalysis &,
                                     TranslationContext &, const LaneLayout *, const LaneLayout *) {
  if (inst.mnemonic() != "s_clause" || inst.size() != static_cast<int>(sizeof(uint32_t)))
    return ExpandResult::failed("gfx1250 S_CLAUSE rule received an unsupported instruction");

  const auto nop = gfx1250::build_sopp(gfx1250::kSNopSopp, {.simm16 = 0});
  return ExpandResult::success(std::vector<uint32_t>(nop.begin(), nop.end()));
}

struct Gfx1250Ds2Shape {
  uint16_t replacement_opcode = 0;
  uint8_t element_dwords = 0;
  bool stride64 = false;
  enum class Kind : uint8_t { Load, Store, StoreExchange } kind = Kind::Load;
};

/// @brief Describe one B0 DS2 opcode and its A0 single-address replacement.
[[nodiscard]] Gfx1250Ds2Shape gfx1250_ds2_shape(uint16_t opcode) {
  using Kind = Gfx1250Ds2Shape::Kind;
  switch (opcode) {
  case gfx1250::kDsLoad2addrB32Vds:
    return {gfx1250::kDsLoadB32Vds, 1, false, Kind::Load};
  case gfx1250::kDsLoad2addrStride64B32Vds:
    return {gfx1250::kDsLoadB32Vds, 1, true, Kind::Load};
  case gfx1250::kDsStore2addrB32Vds:
    return {gfx1250::kDsStoreB32Vds, 1, false, Kind::Store};
  case gfx1250::kDsStore2addrStride64B32Vds:
    return {gfx1250::kDsStoreB32Vds, 1, true, Kind::Store};
  case gfx1250::kDsStorexchg2addrRtnB32Vds:
    return {gfx1250::kDsStorexchgRtnB32Vds, 1, false, Kind::StoreExchange};
  case gfx1250::kDsStorexchg2addrStride64RtnB32Vds:
    return {gfx1250::kDsStorexchgRtnB32Vds, 1, true, Kind::StoreExchange};
  case gfx1250::kDsLoad2addrB64Vds:
    return {gfx1250::kDsLoadB64Vds, 2, false, Kind::Load};
  case gfx1250::kDsLoad2addrStride64B64Vds:
    return {gfx1250::kDsLoadB64Vds, 2, true, Kind::Load};
  case gfx1250::kDsStore2addrB64Vds:
    return {gfx1250::kDsStoreB64Vds, 2, false, Kind::Store};
  case gfx1250::kDsStore2addrStride64B64Vds:
    return {gfx1250::kDsStoreB64Vds, 2, true, Kind::Store};
  case gfx1250::kDsStorexchg2addrRtnB64Vds:
    return {gfx1250::kDsStorexchgRtnB64Vds, 2, false, Kind::StoreExchange};
  case gfx1250::kDsStorexchg2addrStride64RtnB64Vds:
    return {gfx1250::kDsStorexchgRtnB64Vds, 2, true, Kind::StoreExchange};
  default:
    return {};
  }
}

/// @brief Build one single-address DS instruction from a DS2 operand half.
[[nodiscard]] std::array<uint32_t, 2> build_gfx1250_ds2_half(const gfx1250::VdsMachineInst &source,
                                                             const Gfx1250Ds2Shape &shape,
                                                             uint16_t byte_offset,
                                                             bool second_half) {
  const uint8_t tuple_delta = second_half ? shape.element_dwords : 0;
  // Plain DS stores have no destination operand, and their reserved VDST field
  // must remain zero. Loads and returning exchanges use consecutive VDST
  // tuples for the two halves.
  const uint8_t vdst = shape.kind == Gfx1250Ds2Shape::Kind::Store
                           ? 0
                           : static_cast<uint8_t>(source.vdst + tuple_delta);
  return gfx1250::build_vds(
      shape.replacement_opcode,
      {.offset0 = static_cast<uint8_t>(byte_offset),
       .offset1 = static_cast<uint8_t>(byte_offset >> 8),
       .addr = static_cast<uint8_t>(source.addr),
       // A single-address store/exchange consumes DATA0. The second DS2 data
       // operand therefore moves from the source DATA1 field into DATA0.
       .data0 = static_cast<uint8_t>(second_half ? source.data1 : source.data0),
       .data1 = 0,
       .vdst = vdst});
}

/// @brief Expand a gfx1250 B0 two-address DS operation for A0.
///
/// @details A0 requires DS2 offsets to satisfy alignment restrictions which B0
/// relaxed. Two ordinary DS operations accept byte offsets and avoid that
/// erratum. A local DSCNT drain preserves the completion semantics of the one
/// original DS instruction without having to rewrite downstream wait counts.
ExpandResult expand_gfx1250_ds2(const Instruction &inst, uint32_t, uint64_t,
                                std::span<const uint8_t>, const LivenessAnalysis &liveness,
                                TranslationContext &, const LaneLayout *, const LaneLayout *) {
  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || static_cast<size_t>(inst.size()) < sizeof(gfx1250::VdsMachineInst)) {
    return ExpandResult::failed("gfx1250 DS2 instruction has no complete VDS encoding",
                                {"Decode the complete eight-byte VDS instruction."});
  }

  gfx1250::VdsMachineInst source{};
  std::memcpy(&source, raw, sizeof(source));
  const Gfx1250Ds2Shape shape = gfx1250_ds2_shape(inst.opcode());
  if (shape.element_dwords == 0) {
    return ExpandResult::failed("gfx1250 DS2 semantic rule received an unsupported opcode");
  }

  // DS2 immediates are element indices. Stride64 forms add another factor of
  // 64; ordinary single-address DS instructions instead encode a 16-bit byte
  // offset directly.
  const uint32_t byte_scale =
      static_cast<uint32_t>(shape.element_dwords) * sizeof(uint32_t) * (shape.stride64 ? 64u : 1u);
  const uint32_t offset0 = static_cast<uint32_t>(source.offset0) * byte_scale;
  const uint32_t offset1 = static_cast<uint32_t>(source.offset1) * byte_scale;
  constexpr uint32_t kSingleAddressOffsetMax = 0xffff;
  if (offset0 > kSingleAddressOffsetMax || offset1 > kSingleAddressOffsetMax) {
    return ExpandResult::failed(
        "gfx1250 DS2 scaled offset exceeds the single-address 16-bit field",
        {"Use a scratch-address lowering for DS2 offsets larger than 65535 bytes."});
  }

  const auto first = build_gfx1250_ds2_half(source, shape, static_cast<uint16_t>(offset0), false);
  const auto second = build_gfx1250_ds2_half(source, shape, static_cast<uint16_t>(offset1), true);
  const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
  const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  if (!src0_bank || !src1_bank || !src2_bank || !dst_bank) {
    return ExpandResult::failed(
        "gfx1250 DS2 lowering cannot prove the VGPR-MSB mode",
        {"Make the VGPR-MSB fields known on every CFG path reaching this instruction."});
  }
  const uint8_t original_mode =
      static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) | (*src2_bank << 4) | (*dst_bank << 6));

  const auto physical = [](uint8_t selector, uint8_t bank) {
    return static_cast<uint16_t>(static_cast<uint16_t>(bank) * 256u + selector);
  };
  const auto physical_overlap = [](uint16_t lhs, uint8_t lhs_width, uint16_t rhs,
                                   uint8_t rhs_width) {
    return lhs < static_cast<uint32_t>(rhs) + rhs_width &&
           rhs < static_cast<uint32_t>(lhs) + lhs_width;
  };

  const uint16_t first_dst = physical(static_cast<uint8_t>(source.vdst), *dst_bank);
  const uint32_t second_dst_wide = static_cast<uint32_t>(first_dst) + shape.element_dwords;
  if ((shape.kind == Gfx1250Ds2Shape::Kind::Load ||
       shape.kind == Gfx1250Ds2Shape::Kind::StoreExchange) &&
      second_dst_wide + shape.element_dwords > 1024u) {
    return ExpandResult::failed("gfx1250 DS2 destination tuple exceeds the VGPR address space");
  }
  const uint16_t second_dst = static_cast<uint16_t>(second_dst_wide);
  const uint8_t second_dst_bank = static_cast<uint8_t>(second_dst / 256u);

  // DATA1 is a SRC2 operand in DS2 but becomes DATA0/SRC1 in the second
  // single-address instruction. Select its original bank for the new role.
  const uint8_t second_src1_bank =
      shape.kind == Gfx1250Ds2Shape::Kind::Load ? *src1_bank : *src2_bank;
  const uint8_t second_mode = static_cast<uint8_t>(*src0_bank | (second_src1_bank << 2) |
                                                   (*src2_bank << 4) | (second_dst_bank << 6));
  bool second_first = false;

  if (shape.kind == Gfx1250Ds2Shape::Kind::Load) {
    // The compound load captures ADDR before writing either destination half.
    // If the first half aliases ADDR, issue the independent second load first.
    second_first = physical_overlap(first_dst, shape.element_dwords,
                                    physical(static_cast<uint8_t>(source.addr), *src0_bank), 1);
  } else if (shape.kind == Gfx1250Ds2Shape::Kind::StoreExchange) {
    // Each exchange writes one destination half while the other still needs
    // ADDR and its input data. Pick a safe direction. A dependency in both
    // directions needs scratch storage and must fail closed for now.
    const bool first_clobbers_second =
        physical_overlap(first_dst, shape.element_dwords,
                         physical(static_cast<uint8_t>(source.addr), *src0_bank), 1) ||
        physical_overlap(first_dst, shape.element_dwords,
                         physical(static_cast<uint8_t>(source.data1), *src2_bank),
                         shape.element_dwords);
    const bool second_clobbers_first =
        physical_overlap(second_dst, shape.element_dwords,
                         physical(static_cast<uint8_t>(source.addr), *src0_bank), 1) ||
        physical_overlap(second_dst, shape.element_dwords,
                         physical(static_cast<uint8_t>(source.data0), *src1_bank),
                         shape.element_dwords);
    if (first_clobbers_second && second_clobbers_first) {
      return ExpandResult::failed("gfx1250 DS2 exchange has cyclic destination/source overlap",
                                  {"Add a scratch-VGPR DS2 exchange lowering for cyclic overlap."});
    }
    second_first = first_clobbers_second;
  }

  std::vector<uint32_t> words;
  words.reserve(9);
  uint8_t current_mode = original_mode;
  const auto set_mode = [&](uint8_t mode) {
    append_gfx1250_vgpr_msb_transition(words, current_mode, mode);
  };
  if (second_first) {
    if (second_mode != original_mode)
      set_mode(second_mode);
    append_words(words, second);
    if (second_mode != original_mode)
      set_mode(original_mode);
    append_words(words, first);
  } else {
    append_words(words, first);
    if (second_mode != original_mode)
      set_mode(second_mode);
    append_words(words, second);
    if (second_mode != original_mode)
      set_mode(original_mode);
  }
  append_words(words, gfx1250::build_sopp(gfx1250::kSWaitDscntSopp, {.simm16 = 0}));
  return ExpandResult::success(std::move(words));
}

/// @brief Disable Tensor-DMA multicast for one A0 tensor load.
///
/// @details TENSOR_LOAD_TO_LDS does not encode multicast in the instruction.
/// Descriptor group 1 bits [15:0], held in the first SGPR named by VADDR1,
/// select the workgroups which receive a multicast load.  On A0 those bits
/// must therefore be cleared for every tensor load; inspecting only the
/// instruction cannot prove that the runtime descriptor mask is zero.
/// Preserve the guest descriptor value around the load because later tensor
/// instructions commonly reuse and update the same descriptor.
ExpandResult expand_gfx1250_tensor_load_to_lds(const Instruction &inst, uint32_t, uint64_t,
                                               std::span<const uint8_t>,
                                               const LivenessAnalysis &liveness,
                                               TranslationContext &, const LaneLayout *,
                                               const LaneLayout *) {
  if (inst.mnemonic() != "tensor_load_to_lds" ||
      inst.size() != static_cast<int>(sizeof(gfx1250::VimageMachineInst)) ||
      inst.raw_encoding() == nullptr) {
    return ExpandResult::failed(
        "gfx1250 tensor-load mask rule received an unsupported instruction");
  }

  gfx1250::VimageMachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  constexpr uint8_t kLastOrdinarySgpr = 105;
  const uint8_t descriptor_base = static_cast<uint8_t>(source.vaddr1);
  if (descriptor_base == kGfx1250Null || descriptor_base > kLastOrdinarySgpr - 7u) {
    return ExpandResult::failed(
        "gfx1250 tensor-load group-1 descriptor is not a valid eight-SGPR tuple",
        {"Provide TENSOR_LOAD_TO_LDS VADDR1 as an ordinary eight-SGPR descriptor."});
  }

  const std::optional<uint16_t> scratch = liveness.find_free_sgpr(&inst);
  if (!scratch || *scratch > kLastOrdinarySgpr) {
    return ExpandResult::failed(
        "gfx1250 tensor-load mask rule could not allocate a dead scratch SGPR",
        {"Provide one dead ordinary SGPR to preserve the descriptor mask word."});
  }

  std::vector<uint32_t> words;
  words.reserve(6);
  append_words(
      words, gfx1250::build_sop1(gfx1250::kSMovB32Sop1, {.ssrc0 = descriptor_base,
                                                         .sdst = static_cast<uint8_t>(*scratch)}));
  // PACK_HH forms {SRC1[31:16], SRC0[31:16]}. Inline zero as SRC0 clears
  // D1[15:0] while preserving all descriptor fields in D1[31:16].
  append_words(words, gfx1250::build_sop2(
                          gfx1250::kSPackHhB32B16Sop2,
                          {.ssrc0 = 128, .ssrc1 = descriptor_base, .sdst = descriptor_base}));
  words.insert(words.end(), inst.raw_encoding(),
               inst.raw_encoding() + sizeof(gfx1250::VimageMachineInst) / sizeof(uint32_t));
  append_words(words,
               gfx1250::build_sop1(gfx1250::kSMovB32Sop1, {.ssrc0 = static_cast<uint8_t>(*scratch),
                                                           .sdst = descriptor_base}));
  return ExpandResult::success(std::move(words));
}

/// @brief Return the K=64 replacement opcode for one B0-only K=128 WMMA.
[[nodiscard]] uint16_t gfx1250_k128_wmma_replacement(uint16_t opcode) {
  switch (opcode) {
  case gfx1250::kVWmmaF3216x16x128Fp8Fp8Vop3p:
    return gfx1250::kVWmmaF3216x16x64Fp8Fp8Vop3p;
  case gfx1250::kVWmmaF3216x16x128Fp8Bf8Vop3p:
    return gfx1250::kVWmmaF3216x16x64Fp8Bf8Vop3p;
  case gfx1250::kVWmmaF3216x16x128Bf8Fp8Vop3p:
    return gfx1250::kVWmmaF3216x16x64Bf8Fp8Vop3p;
  case gfx1250::kVWmmaF3216x16x128Bf8Bf8Vop3p:
    return gfx1250::kVWmmaF3216x16x64Bf8Bf8Vop3p;
  case gfx1250::kVWmmaF1616x16x128Fp8Fp8Vop3p:
    return gfx1250::kVWmmaF1616x16x64Fp8Fp8Vop3p;
  case gfx1250::kVWmmaF1616x16x128Fp8Bf8Vop3p:
    return gfx1250::kVWmmaF1616x16x64Fp8Bf8Vop3p;
  case gfx1250::kVWmmaF1616x16x128Bf8Fp8Vop3p:
    return gfx1250::kVWmmaF1616x16x64Bf8Fp8Vop3p;
  case gfx1250::kVWmmaF1616x16x128Bf8Bf8Vop3p:
    return gfx1250::kVWmmaF1616x16x64Bf8Bf8Vop3p;
  default:
    return 0;
  }
}

/// @brief Replace one bit field in a 32-bit instruction word.
void set_word_field(uint32_t &word, uint32_t value, uint32_t shift, uint32_t width) {
  const uint32_t mask = ((uint32_t{1} << width) - 1) << shift;
  word = (word & ~mask) | ((value << shift) & mask);
}

/// @brief Remove the false scalar dependency from regular scaled WMMA.
///
/// @details VOP3PX2 bits [58:50] are an unused `scale_src2` encoding which SQ
/// nevertheless decodes as a source register. Encoding VGPR0 (0x100) prevents
/// the zero-filled B0 encoding from creating a false SGPR dependency. This is
/// an encoding erratum only; every architectural operand remains unchanged.
ExpandResult expand_gfx1250_wmma_scale_src2(const Instruction &inst, uint32_t, uint64_t,
                                            std::span<const uint8_t>, const LivenessAnalysis &,
                                            TranslationContext &, const LaneLayout *,
                                            const LaneLayout *) {
  if (!inst.mnemonic().starts_with("v_wmma_scale_f32_") || inst.size() != 4 * sizeof(uint32_t) ||
      inst.raw_encoding() == nullptr) {
    return ExpandResult::failed(
        "gfx1250 scaled-WMMA SRC2 rule received an unsupported VOP3PX2 instruction");
  }

  gfx1250::Vop3pMachineInst matrix{};
  std::memcpy(&matrix, inst.raw_encoding() + 2, sizeof(matrix));
  if (matrix.op == gfx1250::kVWmmaF3232x16x128F4Vop3p) {
    return ExpandResult::failed(
        "gfx1250 regular-Scale 32x16 FP4 WMMA needs a prefix-aware M split");
  }

  std::vector<uint32_t> words(inst.raw_encoding(), inst.raw_encoding() + 4);
  // Instruction bits [58:50] occupy word 1 bits [26:18].
  set_word_field(words[1], 0x100, 18, 9);
  return ExpandResult::success(std::move(words));
}

/// @brief Split the standalone B0 32x16 FP4 WMMA for A0.
/// @details Fails closed. The split would emit bare low-precision WMMA halves,
/// which are exactly the forms the legalizer rejects on input (their standalone
/// two-dword base encoding is not safe to emit for A0). The safe replacement is
/// the regular-Scale-prefixed encoding with neutral inline scales, but that
/// lowering is not yet implemented, so the whole instruction fails closed rather
/// than emitting a form that would itself need re-legalization.
ExpandResult expand_gfx1250_wmma_32x16_f4(const Instruction &inst, uint32_t, uint64_t,
                                          std::span<const uint8_t>, const LivenessAnalysis &,
                                          TranslationContext &, const LaneLayout *,
                                          const LaneLayout *) {
  if (inst.mnemonic() != "v_wmma_f32_32x16x128_f4" ||
      inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr) {
    return ExpandResult::failed("gfx1250 32x16 FP4 WMMA rule received an unsupported instruction");
  }
  return ExpandResult::failed("gfx1250 32x16 FP4 WMMA A0 lowering is not yet implemented",
                              {"Provide the regular-scale WMMA lowering for this form."});
}

/// @brief Encode an inline non-negative integer accepted by a VALU source.
[[nodiscard]] constexpr uint16_t gfx1250_inline_u32(uint16_t value) {
  return static_cast<uint16_t>(128u + value);
}

/// @brief Encode one low-bank VGPR as a generic VALU source operand.
[[nodiscard]] constexpr uint16_t gfx1250_vgpr_src(uint16_t vgpr) {
  return static_cast<uint16_t>(256u + vgpr);
}

/// @brief Gather either the even or odd bytes of one B64 Scale16 operand.
///
/// @details Scale16 stores eight block-16 exponent bytes in two VGPRs. A0
/// regular Scale consumes four block-32 bytes. An exact lowering therefore
/// executes two WMMAs: the low-K pass gathers bytes 0,2,4,6 and the high-K
/// pass gathers bytes 1,3,5,7. Combining adjacent bytes (for example with
/// max) is not equivalent because each scale applies to different matrix data.
void append_gfx1250_scale16_gather(std::vector<uint32_t> &words, uint16_t src_lo, uint16_t dst,
                                   uint16_t temp, bool odd) {
  const auto vgpr = gfx1250_vgpr_src;
  const auto bfe = [&](uint16_t out, uint16_t src, uint16_t bit) {
    append_words(words,
                 gfx1250::build_vop3(gfx1250::kVBfeU32Vop3, {.vdst = static_cast<uint8_t>(out),
                                                             .src0 = vgpr(src),
                                                             .src1 = gfx1250_inline_u32(bit),
                                                             .src2 = gfx1250_inline_u32(8)}));
  };
  const auto insert = [&](uint16_t value, uint16_t shift) {
    append_words(words,
                 gfx1250::build_vop3(gfx1250::kVLshlOrB32Vop3, {.vdst = static_cast<uint8_t>(dst),
                                                                .src0 = vgpr(value),
                                                                .src1 = gfx1250_inline_u32(shift),
                                                                .src2 = vgpr(dst)}));
  };

  const uint16_t first_bit = odd ? 8 : 0;
  bfe(dst, src_lo, first_bit);
  bfe(temp, src_lo, static_cast<uint16_t>(first_bit + 16));
  insert(temp, 8);
  bfe(temp, static_cast<uint16_t>(src_lo + 1u), first_bit);
  insert(temp, 16);
  bfe(temp, static_cast<uint16_t>(src_lo + 1u), static_cast<uint16_t>(first_bit + 16));
  insert(temp, 24);
}

/// @brief Convert B0 Scale16 WMMA to an exact pair of A0 regular-Scale WMMAs.
///
/// @details The two passes consume the even and odd Scale16 bytes and mutually
/// exclusive K=16 portions of matrix A. Their results accumulate through D.
/// The B0-only 32x16 form also needs an independent M split and is rejected
/// until a combined four-pass lowering exists.
ExpandResult expand_gfx1250_wmma_scale16(const Instruction &inst, uint32_t, uint64_t,
                                         std::span<const uint8_t>, const LivenessAnalysis &liveness,
                                         TranslationContext &context, const LaneLayout *,
                                         const LaneLayout *) {
  // The prefix opcode shares its structural lookup key with ordinary VOP3
  // instructions. Decline those collisions so their own legalization can
  // report an unimplemented expansion instead of a misleading Scale16 error.
  if (!inst.mnemonic().starts_with("v_wmma_scale16_f32_"))
    return ExpandResult::not_handled();
  if (inst.size() != 4 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr) {
    return ExpandResult::failed(
        "gfx1250 Scale16 WMMA rule received an unsupported VOP3PX3 instruction");
  }

  gfx1250::Vop3pMachineInst scale{};
  gfx1250::Vop3pMachineInst matrix{};
  std::memcpy(&scale, inst.raw_encoding(), sizeof(scale));
  std::memcpy(&matrix, inst.raw_encoding() + 2, sizeof(matrix));
  if (scale.op != kWmmaScale16PrefixOp || (matrix.op != gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p &&
                                           matrix.op != gfx1250::kVWmmaF3232x16x128F4Vop3p)) {
    return ExpandResult::failed("gfx1250 Scale16 WMMA rule received an unsupported base opcode");
  }

  constexpr uint16_t kVgprEncoding = 256;
  if (scale.src0 < kVgprEncoding || scale.src1 < kVgprEncoding) {
    return ExpandResult::failed("gfx1250 Scale16 WMMA scale operands are not VGPR pairs");
  }
  const uint16_t scale_a = static_cast<uint16_t>(scale.src0 - kVgprEncoding);
  const uint16_t scale_b = static_cast<uint16_t>(scale.src1 - kVgprEncoding);
  if (scale_a >= 255 || scale_b >= 255) {
    return ExpandResult::failed("gfx1250 Scale16 WMMA scale pair crosses the low VGPR bank");
  }

  if (matrix.op == gfx1250::kVWmmaF3232x16x128F4Vop3p) {
    return ExpandResult::failed(
        "gfx1250 32x16 Scale16 WMMA needs a combined M/K split",
        {"Use ordinary Scale for the 32x16 FP4 form or provide a four-pass lowering."});
  }

  if (matrix.src0 < kVgprEncoding || matrix.src1 < kVgprEncoding) {
    return ExpandResult::failed("gfx1250 Scale16 WMMA matrix inputs are not VGPR ranges");
  }
  const uint16_t matrix_a = static_cast<uint16_t>(matrix.src0 - kVgprEncoding);
  const uint16_t matrix_b = static_cast<uint16_t>(matrix.src1 - kVgprEncoding);

  const uint8_t matrix_a_fmt = static_cast<uint8_t>(matrix.opsel);
  const uint8_t matrix_b_fmt =
      static_cast<uint8_t>((matrix.pad_14 != 0 ? 4u : 0u) | matrix.opsel_hi);
  const auto matrix_width = [](uint8_t fmt) -> uint16_t {
    if (fmt <= 1)
      return 16; // FP8/BF8: K subblocks are selected by lane.
    if (fmt <= 3)
      return 12; // FP6/BF6: three VGPRs per K=16 subblock.
    if (fmt == 4)
      return 8; // FP4: two VGPRs per K=16 subblock.
    return 0;
  };
  const uint16_t matrix_a_width = matrix_width(matrix_a_fmt);
  const uint16_t matrix_b_width = matrix_width(matrix_b_fmt);
  if (matrix_a_width == 0 || matrix_b_width == 0) {
    return ExpandResult::failed("gfx1250 Scale16 WMMA has an unknown matrix format");
  }
  if (static_cast<uint32_t>(matrix_a) + matrix_a_width > 256u ||
      static_cast<uint32_t>(matrix_b) + matrix_b_width > 256u) {
    return ExpandResult::failed("gfx1250 Scale16 WMMA matrix range crosses a VGPR bank");
  }

  SemanticScratchAllocator allocator(
      inst, liveness, context,
      SemanticScratchPolicy{.max_vgprs = 256, .max_spill_dword_offset = 0});
  SemanticScratchRequest request;
  request.count = static_cast<uint16_t>(5u + matrix_a_width);
  request.allow_spill = false;
  const SemanticScratchResult scratch = allocator.acquire_vgprs(request);
  if (!scratch) {
    return ExpandResult::failed(
        "gfx1250 Scale16 WMMA could not allocate scratch VGPRs for an exact K split",
        {"Provide dead low-bank VGPRs for four scales, one temporary, and matrix A."});
  }
  const uint16_t scale_a_lo = scratch.lease->base;
  const uint16_t scale_b_lo = static_cast<uint16_t>(scale_a_lo + 1u);
  const uint16_t scale_a_hi = static_cast<uint16_t>(scale_a_lo + 2u);
  const uint16_t scale_b_hi = static_cast<uint16_t>(scale_a_lo + 3u);
  const uint16_t temp = static_cast<uint16_t>(scale_a_lo + 4u);
  const uint16_t masked_a = static_cast<uint16_t>(scale_a_lo + 5u);

  const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
  const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  if (!src0_bank || !src1_bank || !src2_bank || !dst_bank) {
    return ExpandResult::failed("gfx1250 Scale16 WMMA cannot prove the VGPR-MSB mode");
  }
  if (*src0_bank != 0 || *src1_bank != 0 || *src2_bank != 0 || *dst_bank != 0) {
    return ExpandResult::failed(
        "gfx1250 Scale16 exact split does not yet support nonzero VGPR-MSB banks");
  }

  constexpr uint16_t kDestinationDwords = 8;
  const auto overlaps = [](uint16_t lhs, uint16_t lhs_count, uint16_t rhs, uint16_t rhs_count) {
    return lhs < static_cast<uint32_t>(rhs) + rhs_count &&
           rhs < static_cast<uint32_t>(lhs) + lhs_count;
  };
  if (overlaps(matrix.vdst, kDestinationDwords, matrix_a, matrix_a_width) ||
      overlaps(matrix.vdst, kDestinationDwords, matrix_b, matrix_b_width)) {
    return ExpandResult::failed("gfx1250 Scale16 destination destructively overlaps matrix A or B");
  }

  std::optional<uint16_t> mask_sgpr;
  if (matrix_a_fmt <= 1) {
    mask_sgpr = liveness.find_free_sgpr(&inst);
    if (!mask_sgpr || *mask_sgpr > 105) {
      return ExpandResult::failed(
          "gfx1250 Scale16 lane-mask split could not allocate a dead scratch SGPR");
    }
  }

  std::vector<uint32_t> words;
  words.reserve(128);
  append_gfx1250_scale16_gather(words, scale_a, scale_a_lo, temp, false);
  append_gfx1250_scale16_gather(words, scale_b, scale_b_lo, temp, false);
  append_gfx1250_scale16_gather(words, scale_a, scale_a_hi, temp, true);
  append_gfx1250_scale16_gather(words, scale_b, scale_b_hi, temp, true);

  const auto append_masked_a = [&](bool high) {
    if (matrix_a_fmt <= 1) {
      append_words(words,
                   gfx1250::build_sop1(gfx1250::kSMovB32Sop1,
                                       {.ssrc0 = 0xff, .sdst = static_cast<uint8_t>(*mask_sgpr)}));
      words.push_back(high ? 0xffff0000u : 0x0000ffffu);
      for (uint16_t i = 0; i < matrix_a_width; ++i) {
        append_words(words, gfx1250::build_vop3(gfx1250::kVCndmaskB32Vop3,
                                                {.vdst = static_cast<uint8_t>(masked_a + i),
                                                 .src0 = gfx1250_inline_u32(0),
                                                 .src1 = gfx1250_vgpr_src(matrix_a + i),
                                                 .src2 = *mask_sgpr}));
      }
      return;
    }

    const uint16_t subblock_width = matrix_a_fmt <= 3 ? 3 : 2;
    for (uint16_t i = 0; i < matrix_a_width; ++i) {
      const bool is_high_subblock = ((i / subblock_width) & 1u) != 0;
      const uint16_t source =
          is_high_subblock == high ? gfx1250_vgpr_src(matrix_a + i) : gfx1250_inline_u32(0);
      append_words(
          words, gfx1250::build_vop3(gfx1250::kVMovB32Vop3,
                                     {.vdst = static_cast<uint8_t>(masked_a + i), .src0 = source}));
    }
  };

  const auto build_pass = [&](uint16_t pass_scale_a, uint16_t pass_scale_b, bool accumulate_d) {
    std::array<uint32_t, 4> pass = {inst.raw_encoding()[0], inst.raw_encoding()[1],
                                    inst.raw_encoding()[2], inst.raw_encoding()[3]};
    set_word_field(pass[0], kWmmaScaleSrc2PrefixOp, 16, 8);
    set_word_field(pass[1], gfx1250_vgpr_src(pass_scale_a), 0, 9);
    set_word_field(pass[1], gfx1250_vgpr_src(pass_scale_b), 9, 9);
    set_word_field(pass[1], gfx1250_vgpr_src(0), 18, 9);
    set_word_field(pass[3], gfx1250_vgpr_src(masked_a), 0, 9);
    if (accumulate_d) {
      set_word_field(pass[3], gfx1250_vgpr_src(matrix.vdst), 18, 9);
      pass[2] &= ~(uint32_t{1} << 10);
      pass[3] &= ~(uint32_t{1} << 31);
    }
    words.insert(words.end(), pass.begin(), pass.end());
  };

  append_masked_a(false);
  build_pass(scale_a_lo, scale_b_lo, false);
  append_words(words, gfx1250::build_vop1(gfx1250::kVNopVop1));
  append_masked_a(true);
  build_pass(scale_a_hi, scale_b_hi, true);
  return ExpandResult::success(std::move(words));
}

/// @brief Conservatively separate B0 integer WMMA from its A0 successor.
///
/// @details A0 requires eight safe co-execution slots after integer IU8/IU4
/// WMMA, whereas B0 requires four. This temporary local lowering does not yet
/// inspect the following VALU or control-flow successors, so it appends eight
/// V_NOPs unconditionally and is intentionally safe rather than optimal.
ExpandResult expand_gfx1250_wmma_iu8_spacing(const Instruction &inst, uint32_t, uint64_t,
                                             std::span<const uint8_t>, const LivenessAnalysis &,
                                             TranslationContext &, const LaneLayout *,
                                             const LaneLayout *) {
  if (inst.mnemonic() != "v_wmma_i32_16x16x64_iu8" &&
      inst.mnemonic() != "v_swmmac_i32_16x16x128_iu8")
    return ExpandResult::not_handled();
  if (inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr)
    return ExpandResult::failed("gfx1250 IU8 WMMA rule received an unsupported VOP3P instruction");

  std::vector<uint32_t> words(inst.raw_encoding(),
                              inst.raw_encoding() + inst.size() / sizeof(uint32_t));
  // TODO: Reduce this to four V_NOPs once we establish that B0 compiler output
  // always supplies its required four safe co-execution slots.
  // TODO: Replace fixed padding with a whole-kernel lookahead that counts
  // existing V_NOPs/independent VALU and inserts exactly the A0-required eight slots.
  for (int slot = 0; slot < 8; ++slot)
    append_words(words, gfx1250::build_vop1(gfx1250::kVNopVop1));
  return ExpandResult::success(std::move(words));
}

/// @brief True if @p opcode is a gfx1250 cluster-load form this rule covers
/// (both the plain and async-to-LDS families, all widths).
[[nodiscard]] bool is_gfx1250_cluster_load(uint16_t opcode) {
  switch (opcode) {
  case gfx1250::kClusterLoadB32Vglobal:
  case gfx1250::kClusterLoadB64Vglobal:
  case gfx1250::kClusterLoadB128Vglobal:
  case gfx1250::kClusterLoadAsyncToLdsB8Vglobal:
  case gfx1250::kClusterLoadAsyncToLdsB32Vglobal:
  case gfx1250::kClusterLoadAsyncToLdsB64Vglobal:
  case gfx1250::kClusterLoadAsyncToLdsB128Vglobal:
    return true;
  default:
    return false;
  }
}

/// @brief Rewrite a gfx1250 cluster load to run with M0 = 0.
///
/// @details Every cluster-load form (both SADDR and off/NULL-saddr, all widths)
/// is left as a cluster load and wrapped so it executes with M0 forced to zero:
/// save M0 to a dead SGPR, set M0 = 0, run the load, then restore M0. The opcode
/// is not changed.
ExpandResult expand_gfx1250_cluster_load(const Instruction &inst, uint32_t, uint64_t,
                                         std::span<const uint8_t>, const LivenessAnalysis &liveness,
                                         TranslationContext &, const LaneLayout *,
                                         const LaneLayout *) {
  if (!is_gfx1250_cluster_load(inst.opcode()) ||
      inst.size() != 3 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr) {
    return ExpandResult::failed("gfx1250 cluster-load rule received an unsupported instruction");
  }

  const std::optional<uint16_t> scratch = liveness.find_free_sgpr(&inst);
  if (!scratch || *scratch > 105) {
    return ExpandResult::failed(
        "gfx1250 cluster load could not allocate a dead SGPR for M0 preservation");
  }

  // Save M0 to scratch, set M0 = 0, run the load, then restore M0. A binary
  // translator cannot prove M0 is dead after the load, so it saves and restores
  // the original value around it.
  //
  // Inline constant 0 encodes as 128 in a scalar source. Every M0 reference here
  // MUST use kGfx1250M0 (125): on gfx1250 M0 encodes as 125 and NULL as 124 (the
  // inverse of CDNA), so a write to 124 would be a discarded NULL write.
  constexpr uint8_t kInlineZero = 128;
  std::vector<uint32_t> words;
  words.reserve(6);
  append_words(words,
               gfx1250::build_sop1(gfx1250::kSMovB32Sop1,
                                   {.ssrc0 = kGfx1250M0, .sdst = static_cast<uint8_t>(*scratch)}));
  append_words(words, gfx1250::build_sop1(gfx1250::kSMovB32Sop1,
                                          {.ssrc0 = kInlineZero, .sdst = kGfx1250M0}));
  words.insert(words.end(), inst.raw_encoding(), inst.raw_encoding() + 3);
  append_words(words,
               gfx1250::build_sop1(gfx1250::kSMovB32Sop1,
                                   {.ssrc0 = static_cast<uint8_t>(*scratch), .sdst = kGfx1250M0}));
  return ExpandResult::success(std::move(words));
}

/// @brief Materialize the B0 DS ADDTID address and issue an ordinary A0 DS op.
///
/// @details ds_*_addtid_b32 computes its LDS byte address on-chip as
/// `(M0 + tid*4) & 0xfffff`, where `tid` is the wave-local thread id and M0
/// carries the LDS base. A0 does not honor that addressing for these opcodes, so
/// the address is materialized explicitly and an ordinary ds_load_b32/ds_store_b32
/// is issued against it. The emitted sequence reproduces the formula term by term:
///   1. v_mbcnt_lo/hi_u32_b32 with mask -1 -> tid (population count of lanes below).
///   2. v_lshlrev_b32 by 2 -> tid*4.
///   3. v_add_nc_u32 with SRC0 = M0 (gfx1250 M0 encodes as 125) -> M0 + tid*4.
///   4. v_bfe_u32 offset 0 width 20 -> (M0 + tid*4) & 0xfffff.
/// This is the A0-stepping contract, not the revision-neutral simulator's
/// documented ADDTID model; the test pins the emitted operands rather than
/// executing against that model, which would test the B0 contract instead.
ExpandResult expand_gfx1250_ds_addtid(const Instruction &inst, uint32_t, uint64_t,
                                      std::span<const uint8_t>, const LivenessAnalysis &liveness,
                                      TranslationContext &context, const LaneLayout *,
                                      const LaneLayout *) {
  const bool is_load = inst.opcode() == gfx1250::kDsLoadAddtidB32Vds;
  const bool is_store = inst.opcode() == gfx1250::kDsStoreAddtidB32Vds;
  if ((!is_load && !is_store) || inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) ||
      inst.raw_encoding() == nullptr) {
    return ExpandResult::failed("gfx1250 ADDTID rule received an unsupported instruction");
  }

  gfx1250::VdsMachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  uint16_t temp = source.vdst;
  std::optional<SemanticScratchResult> store_scratch;
  if (is_store) {
    SemanticScratchAllocator allocator(
        inst, liveness, context,
        SemanticScratchPolicy{.max_vgprs = 256, .max_spill_dword_offset = 0});
    SemanticScratchRequest request;
    request.count = 1;
    request.allow_spill = false;
    store_scratch = allocator.acquire_vgprs(request);
    if (!*store_scratch) {
      return ExpandResult::failed(
          "gfx1250 DS store ADDTID could not allocate a dead low-bank scratch VGPR");
    }
    temp = (*store_scratch).lease->base;
  }

  const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
  const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  if (!src0_bank || !src1_bank || !src2_bank || !dst_bank) {
    return ExpandResult::failed("gfx1250 ADDTID lowering cannot prove the VGPR-MSB mode");
  }
  const uint8_t original_mode =
      static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) | (*src2_bank << 4) | (*dst_bank << 6));
  const uint8_t compute_bank = is_load ? *dst_bank : 0;
  const uint8_t compute_mode = static_cast<uint8_t>(compute_bank | (compute_bank << 2) |
                                                    (compute_bank << 4) | (compute_bank << 6));

  std::vector<uint32_t> words;
  words.reserve(18);
  uint8_t current_mode = original_mode;
  append_gfx1250_vgpr_msb_transition(words, current_mode, compute_mode);
  append_words(
      words, gfx1250::build_vop3(gfx1250::kVMbcntLoU32B32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                                .src0 = 193, // inline -1
                                                                .src1 = gfx1250_inline_u32(0)}));
  append_words(
      words, gfx1250::build_vop3(gfx1250::kVMbcntHiU32B32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                                .src0 = 193,
                                                                .src1 = gfx1250_vgpr_src(temp)}));
  append_words(words,
               gfx1250::build_vop3(gfx1250::kVLshlrevB32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                               .src0 = gfx1250_inline_u32(2),
                                                               .src1 = gfx1250_vgpr_src(temp)}));
  append_words(words,
               gfx1250::build_vop3(gfx1250::kVAddNcU32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                             .src0 = kGfx1250M0,
                                                             .src1 = gfx1250_vgpr_src(temp)}));
  append_words(words,
               gfx1250::build_vop3(gfx1250::kVBfeU32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                           .src0 = gfx1250_vgpr_src(temp),
                                                           .src1 = gfx1250_inline_u32(0),
                                                           .src2 = gfx1250_inline_u32(20)}));

  if (is_store) {
    const uint8_t ds_mode = static_cast<uint8_t>(*src0_bank << 2);
    append_gfx1250_vgpr_msb_transition(words, current_mode, ds_mode);
    append_words(words, gfx1250::build_vds(gfx1250::kDsStoreB32Vds,
                                           {.offset0 = static_cast<uint8_t>(source.offset0),
                                            .offset1 = static_cast<uint8_t>(source.offset1),
                                            .addr = static_cast<uint8_t>(temp),
                                            .data0 = static_cast<uint8_t>(source.data0)}));
    append_gfx1250_vgpr_msb_transition(words, current_mode, original_mode);
  } else {
    append_words(words, gfx1250::build_vds(gfx1250::kDsLoadB32Vds,
                                           {.offset0 = static_cast<uint8_t>(source.offset0),
                                            .offset1 = static_cast<uint8_t>(source.offset1),
                                            .addr = static_cast<uint8_t>(temp),
                                            .vdst = static_cast<uint8_t>(temp)}));
    append_gfx1250_vgpr_msb_transition(words, current_mode, original_mode);
  }
  return ExpandResult::success(std::move(words));
}

/// @brief Emulate B0 CLAMP=1 UE5M3 unpack on A0.
ExpandResult expand_gfx1250_cvt_f32_fp8_e5m3(const Instruction &inst, uint32_t, uint64_t,
                                             std::span<const uint8_t>,
                                             const LivenessAnalysis &liveness,
                                             TranslationContext &context, const LaneLayout *,
                                             const LaneLayout *) {
  if (!inst.mnemonic().starts_with("v_cvt_f32_fp8") ||
      inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr) {
    return ExpandResult::failed("gfx1250 E5M3 unpack rule received an unsupported instruction");
  }
  gfx1250::Vop3MachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  constexpr uint16_t kVgprEncoding = 256;
  if (source.clamp == 0)
    return ExpandResult::not_handled();
  if (source.src0 < kVgprEncoding) {
    return ExpandResult::failed("gfx1250 E5M3 unpack source is not a VGPR");
  }

  SemanticScratchAllocator allocator(
      inst, liveness, context,
      SemanticScratchPolicy{.max_vgprs = 256, .max_spill_dword_offset = 0});
  SemanticScratchRequest request;
  request.count = 2;
  request.allow_spill = false;
  const SemanticScratchResult scratch = allocator.acquire_vgprs(request);
  if (!scratch) {
    return ExpandResult::failed("gfx1250 E5M3 unpack could not allocate two scratch VGPRs");
  }
  const uint16_t out = scratch.lease->base;
  const uint16_t temp = static_cast<uint16_t>(out + 1u);

  const std::optional<uint16_t> nan_mask = liveness.find_free_sgpr(&inst);
  const std::optional<uint16_t> exp31_mask =
      nan_mask ? liveness.find_free_sgpr(&inst, static_cast<uint16_t>(*nan_mask + 1u))
               : std::nullopt;
  if (!nan_mask || !exp31_mask || *exp31_mask > 105) {
    return ExpandResult::failed("gfx1250 E5M3 unpack could not allocate two dead SGPR masks");
  }

  const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
  const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  if (!src0_bank || !src1_bank || !src2_bank || !dst_bank) {
    return ExpandResult::failed("gfx1250 E5M3 unpack cannot prove the VGPR-MSB mode");
  }
  const uint8_t original_mode =
      static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) | (*src2_bank << 4) | (*dst_bank << 6));
  const uint8_t extract_mode = *src0_bank;

  const auto append_vop3_literal = [](std::vector<uint32_t> &words, uint16_t opcode,
                                      gfx1250::Vop3BuilderFields fields, uint32_t literal) {
    append_words(words, gfx1250::build_vop3(opcode, fields));
    words.push_back(literal);
  };
  const auto append_compare_literal = [](std::vector<uint32_t> &words, uint16_t opcode,
                                         uint8_t sdst, uint16_t src1, uint32_t literal) {
    append_words(words,
                 gfx1250::build_vop3_sdst_enc(opcode, {.sdst = sdst, .src0 = 255, .src1 = src1}));
    words.push_back(literal);
  };

  // The VOP3 encoding swaps the two byte-select bits for this opcode.
  const uint8_t byte_sel =
      static_cast<uint8_t>(((source.opsel & 1u) << 1u) | ((source.opsel & 2u) >> 1u));
  std::vector<uint32_t> words;
  words.reserve(40);
  uint8_t current_mode = original_mode;
  append_gfx1250_vgpr_msb_transition(words, current_mode, extract_mode);
  append_words(
      words, gfx1250::build_vop3(gfx1250::kVBfeU32Vop3,
                                 {.vdst = static_cast<uint8_t>(out),
                                  .src0 = static_cast<uint16_t>(source.src0),
                                  .src1 = gfx1250_inline_u32(static_cast<uint16_t>(byte_sel * 8u)),
                                  .src2 = gfx1250_inline_u32(8)}));
  append_gfx1250_vgpr_msb_transition(words, current_mode, 0);

  append_compare_literal(words, gfx1250::kVCmpEqU32Vop3, static_cast<uint8_t>(*nan_mask),
                         gfx1250_vgpr_src(out), 0xffu);
  append_compare_literal(words, gfx1250::kVCmpLtU32Vop3, static_cast<uint8_t>(*exp31_mask),
                         gfx1250_vgpr_src(out), 0xf7u);
  append_words(words,
               gfx1250::build_vop3(gfx1250::kVAndB32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                           .src0 = gfx1250_inline_u32(7),
                                                           .src1 = gfx1250_vgpr_src(out)}));
  append_words(words,
               gfx1250::build_vop3(gfx1250::kVLshlrevB32Vop3, {.vdst = static_cast<uint8_t>(temp),
                                                               .src0 = gfx1250_inline_u32(20),
                                                               .src1 = gfx1250_vgpr_src(temp)}));
  append_vop3_literal(
      words, gfx1250::kVOrB32Vop3,
      {.vdst = static_cast<uint8_t>(temp), .src0 = 255, .src1 = gfx1250_vgpr_src(temp)},
      0x47800000u);
  append_words(words,
               gfx1250::build_vop3(gfx1250::kVLshlrevB32Vop3, {.vdst = static_cast<uint8_t>(out),
                                                               .src0 = gfx1250_inline_u32(7),
                                                               .src1 = gfx1250_vgpr_src(out)}));
  append_words(words,
               gfx1250::build_vop3(gfx1250::kVCvtF32F16Vop3, {.vdst = static_cast<uint8_t>(out),
                                                              .src0 = gfx1250_vgpr_src(out)}));
  append_words(words,
               gfx1250::build_vop3(gfx1250::kVCndmaskB32Vop3, {.vdst = static_cast<uint8_t>(out),
                                                               .src0 = gfx1250_vgpr_src(out),
                                                               .src1 = gfx1250_vgpr_src(temp),
                                                               .src2 = *exp31_mask}));
  append_vop3_literal(words, gfx1250::kVMovB32Vop3,
                      {.vdst = static_cast<uint8_t>(temp), .src0 = 255}, 0x7fa3d000u);

  const uint8_t final_mode = static_cast<uint8_t>(*dst_bank << 6);
  append_gfx1250_vgpr_msb_transition(words, current_mode, final_mode);
  append_words(words, gfx1250::build_vop3(gfx1250::kVCndmaskB32Vop3,
                                          {.vdst = static_cast<uint8_t>(source.vdst),
                                           .src0 = gfx1250_vgpr_src(out),
                                           .src1 = gfx1250_vgpr_src(temp),
                                           .src2 = *nan_mask}));
  append_gfx1250_vgpr_msb_transition(words, current_mode, original_mode);
  return ExpandResult::success(std::move(words));
}

/// @brief Lower a B0-only K=128 FP8/BF8 WMMA for A0.
///
/// @details Fails closed. Splitting a K=128 WMMA into two A0 K=64 halves would
/// emit bare low-precision K=64 WMMA, which are exactly the forms the legalizer
/// rejects on input (their standalone two-dword base encoding is not safe to emit
/// for A0). The safe replacement is the regular-Scale-prefixed encoding with
/// neutral inline scales, but that lowering is not yet implemented, so the whole
/// instruction fails closed rather than emitting a form that would itself need
/// re-legalization.
ExpandResult expand_gfx1250_k128_wmma(const Instruction &inst, uint32_t, uint64_t,
                                      std::span<const uint8_t>, const LivenessAnalysis &,
                                      TranslationContext &, const LaneLayout *,
                                      const LaneLayout *) {
  // Validate the opcode is one of the covered K=128 forms, then fail closed (see
  // the function's doxygen for why no split is emitted).
  if (gfx1250_k128_wmma_replacement(inst.opcode()) == 0)
    return ExpandResult::failed("gfx1250 K=128 WMMA rule received an unsupported opcode");
  return ExpandResult::failed("gfx1250 K=128 WMMA A0 lowering is not yet implemented",
                              {"Provide the regular-scale K=64 WMMA lowering for this form."});
}

// The semantic translator binary-searches this table, so entries must stay
// sorted by the full encoding ID and then opcode. VDS encoding IDs include the
// high opcode bits, hence the four consecutive kVdsOpHi* groups below.
inline constexpr std::array<TranslationRule, 37> kGfx1250B0ToA0ExpandRules = {{
    {gfx1250::encoding::kSopp, gfx1250::kSClauseSopp, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_s_clause, nullptr, nullptr},
    {gfx1250::encoding::kVop3p, kWmmaScaleSrc2PrefixOp, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_wmma_scale_src2, nullptr, nullptr},
    {gfx1250::encoding::kVop3p, kWmmaScale16PrefixOp, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_wmma_scale16, nullptr, nullptr},
    {gfx1250::encoding::kVop3p, gfx1250::kVWmmaI3216x16x64Iu8Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_wmma_iu8_spacing, nullptr, nullptr},
    {gfx1250::encoding::kVop3p, gfx1250::kVSwmmacI3216x16x128Iu8Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_wmma_iu8_spacing, nullptr, nullptr},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF3216x16x128Fp8Fp8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF3216x16x128Fp8Bf8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF3216x16x128Bf8Fp8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF3216x16x128Bf8Bf8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF1616x16x128Fp8Fp8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF1616x16x128Fp8Bf8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF1616x16x128Bf8Fp8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF1616x16x128Bf8Bf8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF3232x16x128F4Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_wmma_32x16_f4, nullptr, nullptr},
    {gfx1250::encoding::kVimage, gfx1250::kTensorLoadToLdsVimage, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_tensor_load_to_lds, nullptr, nullptr},
    {gfx1250::encoding::kVop3OpHi3, gfx1250::kVCvtF32Fp8Vop3, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_cvt_f32_fp8_e5m3, nullptr, nullptr},
    {gfx1250::encoding::kVds, gfx1250::kDsStore2addrB32Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVds, gfx1250::kDsStore2addrStride64B32Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi1, gfx1250::kDsStorexchg2addrRtnB32Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi1, gfx1250::kDsStorexchg2addrStride64RtnB32Vds, RuleAction::Expand,
     0, 0, nullptr, expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi1, gfx1250::kDsLoad2addrB32Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi1, gfx1250::kDsLoad2addrStride64B32Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi2, gfx1250::kDsStore2addrB64Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi2, gfx1250::kDsStore2addrStride64B64Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi3, gfx1250::kDsStorexchg2addrRtnB64Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi3, gfx1250::kDsStorexchg2addrStride64RtnB64Vds, RuleAction::Expand,
     0, 0, nullptr, expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi3, gfx1250::kDsLoad2addrB64Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi3, gfx1250::kDsLoad2addrStride64B64Vds, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_ds2, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi5, gfx1250::kDsStoreAddtidB32Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds_addtid, nullptr, nullptr},
    {gfx1250::encoding::kVdsOpHi5, gfx1250::kDsLoadAddtidB32Vds, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_ds_addtid, nullptr, nullptr},
    {gfx1250::encoding::kVglobal, gfx1250::kClusterLoadB32Vglobal, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_cluster_load, nullptr, nullptr},
    {gfx1250::encoding::kVglobal, gfx1250::kClusterLoadB64Vglobal, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_cluster_load, nullptr, nullptr},
    {gfx1250::encoding::kVglobal, gfx1250::kClusterLoadB128Vglobal, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_cluster_load, nullptr, nullptr},
    {gfx1250::encoding::kVglobal, gfx1250::kClusterLoadAsyncToLdsB8Vglobal, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_cluster_load, nullptr, nullptr},
    {gfx1250::encoding::kVglobal, gfx1250::kClusterLoadAsyncToLdsB32Vglobal, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_cluster_load, nullptr, nullptr},
    {gfx1250::encoding::kVglobal, gfx1250::kClusterLoadAsyncToLdsB64Vglobal, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_cluster_load, nullptr, nullptr},
    {gfx1250::encoding::kVglobal, gfx1250::kClusterLoadAsyncToLdsB128Vglobal, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_cluster_load, nullptr, nullptr},
}};

} // namespace

std::span<const TranslationRule> semantic_expand_rules_gfx1250_b0_to_a0() {
  return kGfx1250B0ToA0ExpandRules;
}

} // namespace rocjitsu
