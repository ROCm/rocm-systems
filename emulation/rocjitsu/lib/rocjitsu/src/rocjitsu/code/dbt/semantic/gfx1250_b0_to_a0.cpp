// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/gfx1250_b0_to_a0.cpp
/// @brief Handwritten semantic expansions for gfx1250 B0-to-A0 translation.

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
constexpr uint16_t kGfx1250InlineZero = 128;
constexpr uint32_t kGfx1250ScratchMaxDwordOffset = 0x7ffffcu;

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

/// @brief Save or restore one spill-backed low-bank VGPR lease through scratch ST mode.
///
/// @details VSCRATCH ST mode uses NULL SADDR, SVE=0, and a signed 24-bit
/// immediate. Semantic spill storage is non-negative and dword aligned, so the
/// largest usable dword offset is 0x7ffffc. The caller selects VGPR-MSB mode
/// zero before using this helper.
[[nodiscard]] bool append_gfx1250_scratch_preservation(std::vector<uint32_t> &words,
                                                       const SemanticScratchLease &lease,
                                                       bool restore) {
  if (!lease.spilled)
    return true;
  if (lease.reg_class != RegClass::VGPR || lease.count == 0 ||
      static_cast<uint32_t>(lease.base) + lease.count > 256u ||
      lease.spill_offset + (static_cast<uint32_t>(lease.count) - 1u) * sizeof(uint32_t) >
          kGfx1250ScratchMaxDwordOffset) {
    return false;
  }

  for (uint16_t i = 0; i < lease.count; ++i) {
    const uint8_t vgpr = static_cast<uint8_t>(lease.base + i);
    const uint32_t byte_offset = lease.spill_offset + static_cast<uint32_t>(i) * sizeof(uint32_t);
    append_words(words, gfx1250::build_vscratch(restore ? gfx1250::kScratchLoadB32Vscratch
                                                        : gfx1250::kScratchStoreB32Vscratch,
                                                {.saddr = kGfx1250Null,
                                                 .vdst = restore ? vgpr : uint8_t{0},
                                                 .vsrc = restore ? uint8_t{0} : vgpr,
                                                 .ioffset = byte_offset}));
  }
  append_words(
      words, gfx1250::build_sopp(restore ? gfx1250::kSWaitLoadcntSopp : gfx1250::kSWaitStorecntSopp,
                                 {.simm16 = 0}));
  return true;
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
/// @details A0 and B0 use different DS2 offset alignment rules. The B0-to-A0
/// profile translates the operation into two ordinary DS operations with byte
/// offsets. A local DSCNT drain preserves the completion semantics of the
/// original instruction without rewriting downstream wait counts.
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

/// @brief Replace one bit field in a 32-bit instruction word.
void set_word_field(uint32_t &word, uint32_t value, uint32_t shift, uint32_t width) {
  const uint32_t mask = ((uint32_t{1} << width) - 1) << shift;
  word = (word & ~mask) | ((value << shift) & mask);
}

/// @brief Apply the B0-to-A0 regular-scale translation, including the M=32 split.
///
/// @details The translation profile encodes VGPR0 (0x100) in the VOP3PX2
/// `scale_src2` field for the M=16 form.
///
/// gfx1250 B0 additionally introduces the M=32 FP4 form. A0 has the M=16
/// F8F6F4 operation, so the profile translates M=32 into two independent
/// halves. The scale layout assigns M=0..15 and M=16..31 to lanes 0..15 and
/// 16..31 of the same
/// A-scale VGPR; SCL_OPSEL selects the corresponding lane half. Matrix B and
/// its scale are shared. D, A, and a VGPR C are sliced by eight dwords. Both
/// replacement matrix-format fields are forced to FP4, and reuse promises are
/// cleared because each half names a different A/D register range.
ExpandResult expand_gfx1250_wmma_scale_src2(const Instruction &inst, uint32_t, uint64_t,
                                            std::span<const uint8_t>,
                                            const LivenessAnalysis &liveness, TranslationContext &,
                                            const LaneLayout *, const LaneLayout *) {
  if (!inst.mnemonic().starts_with("v_wmma_scale_f32_") || inst.size() != 4 * sizeof(uint32_t) ||
      inst.raw_encoding() == nullptr) {
    return ExpandResult::failed(
        "gfx1250 scaled-WMMA SRC2 rule received an unsupported VOP3PX2 instruction");
  }

  gfx1250::Vop3pMachineInst scale{};
  gfx1250::Vop3pMachineInst matrix{};
  std::memcpy(&scale, inst.raw_encoding(), sizeof(scale));
  std::memcpy(&matrix, inst.raw_encoding() + 2, sizeof(matrix));
  if (matrix.op != gfx1250::kVWmmaF3232x16x128F4Vop3p) {
    std::vector<uint32_t> words(inst.raw_encoding(), inst.raw_encoding() + 4);
    // Instruction bits [58:50] occupy word 1 bits [26:18].
    set_word_field(words[1], 0x100, 18, 9);
    return ExpandResult::success(std::move(words));
  }

  constexpr uint16_t kVgprEncoding = 256;
  constexpr uint16_t kHalfDwords = 8;
  if (scale.src0 < kVgprEncoding || scale.src1 < kVgprEncoding || matrix.src0 < kVgprEncoding ||
      matrix.src1 < kVgprEncoding) {
    return ExpandResult::failed("gfx1250 regular-Scale 32x16 FP4 operands are not VGPR ranges");
  }
  const bool src2_is_vgpr = matrix.src2 >= kVgprEncoding;
  const uint16_t scale_src0 = static_cast<uint16_t>(scale.src0 - kVgprEncoding);
  const uint16_t scale_src1 = static_cast<uint16_t>(scale.src1 - kVgprEncoding);
  const uint16_t src0 = static_cast<uint16_t>(matrix.src0 - kVgprEncoding);
  const uint16_t src1 = static_cast<uint16_t>(matrix.src1 - kVgprEncoding);
  const uint16_t src2 = src2_is_vgpr ? static_cast<uint16_t>(matrix.src2 - kVgprEncoding) : 0;

  const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
  const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  if (!src0_bank || !src1_bank || !src2_bank || !dst_bank) {
    return ExpandResult::failed(
        "gfx1250 regular-Scale 32x16 FP4 split cannot prove the VGPR-MSB mode");
  }

  const auto physical = [](uint8_t bank, uint16_t reg) {
    return static_cast<uint16_t>(bank * 256u + reg);
  };
  const auto overlaps = [](uint16_t lhs, uint16_t lhs_count, uint16_t rhs, uint16_t rhs_count) {
    return lhs < static_cast<uint32_t>(rhs) + rhs_count &&
           rhs < static_cast<uint32_t>(lhs) + lhs_count;
  };
  const uint16_t first_dst = physical(*dst_bank, matrix.vdst);
  const uint16_t upper_a = physical(*src0_bank, static_cast<uint16_t>(src0 + kHalfDwords));
  const uint16_t shared_b = physical(*src1_bank, src1);
  if (overlaps(first_dst, kHalfDwords, upper_a, kHalfDwords) ||
      overlaps(first_dst, kHalfDwords, shared_b, kHalfDwords) ||
      overlaps(first_dst, kHalfDwords, physical(0, scale_src0), 1) ||
      overlaps(first_dst, kHalfDwords, physical(0, scale_src1), 1) ||
      (src2_is_vgpr &&
       overlaps(first_dst, kHalfDwords,
                physical(*src2_bank, static_cast<uint16_t>(src2 + kHalfDwords)), kHalfDwords))) {
    return ExpandResult::failed(
        "gfx1250 regular-Scale 32x16 lower destination overlaps an input needed by the upper half");
  }

  const bool dst_crosses = static_cast<uint32_t>(matrix.vdst) + kHalfDwords > 0xffu;
  const bool src0_crosses = static_cast<uint32_t>(src0) + kHalfDwords > 0xffu;
  const bool src2_crosses = src2_is_vgpr && static_cast<uint32_t>(src2) + kHalfDwords > 0xffu;
  if ((src0_crosses && *src0_bank == 3) || (dst_crosses && *dst_bank == 3) ||
      (src2_crosses && *src2_bank == 3)) {
    return ExpandResult::failed(
        "gfx1250 regular-Scale 32x16 FP4 split exceeds the VGPR address space");
  }

  const uint8_t original_mode =
      static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) | (*src2_bank << 4) | (*dst_bank << 6));
  std::vector<uint32_t> words;
  words.reserve(12);
  uint8_t current_mode = original_mode;
  for (uint16_t half = 0; half < 2; ++half) {
    if (half != 0 && (src0_crosses || dst_crosses || src2_crosses)) {
      const uint8_t upper_mode =
          static_cast<uint8_t>((*src0_bank + (src0_crosses ? 1u : 0u)) | (*src1_bank << 2) |
                               ((*src2_bank + (src2_crosses ? 1u : 0u)) << 4) |
                               ((*dst_bank + (dst_crosses ? 1u : 0u)) << 6));
      append_gfx1250_vgpr_msb_transition(words, current_mode, upper_mode);
    }

    std::array<uint32_t, 2> prefix = {inst.raw_encoding()[0], inst.raw_encoding()[1]};
    // The B0 M=32 prefix uses bits 13/14 for matrix reuse. The split changes A
    // and D register ranges, so neither promise is retained.
    prefix[0] &= ~((uint32_t{1} << 13) | (uint32_t{1} << 14));
    set_word_field(prefix[0], half, 11, 1);          // SCL_OPSEL: select this M half.
    set_word_field(prefix[1], kVgprEncoding, 18, 9); // unused scale_src2 = v0.
    append_words(words, prefix);

    const uint16_t delta = static_cast<uint16_t>(half * kHalfDwords);
    auto replacement = gfx1250::build_vop3p(
        gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p,
        {.vdst = static_cast<uint8_t>(matrix.vdst + delta),
         .neg_hi = static_cast<uint8_t>(matrix.neg_hi),
         .opsel = 4, // matrix A: FP4
         .clamp = static_cast<uint8_t>(matrix.clamp),
         .src0 = static_cast<uint16_t>(kVgprEncoding + ((src0 + delta) & 0xffu)),
         .src1 = static_cast<uint16_t>(kVgprEncoding + src1),
         .src2 = src2_is_vgpr ? static_cast<uint16_t>(kVgprEncoding + ((src2 + delta) & 0xffu))
                              : static_cast<uint16_t>(matrix.src2),
         .opsel_hi = 0,
         .neg = static_cast<uint8_t>(matrix.neg)});
    replacement[0] |= uint32_t{1} << 14; // matrix B: FP4
    append_words(words, replacement);
  }
  if (src0_crosses || dst_crosses || src2_crosses)
    append_gfx1250_vgpr_msb_transition(words, current_mode, original_mode);
  return ExpandResult::success(std::move(words));
}

/// @brief Translate the standalone B0 32x16 FP4 WMMA for A0.
/// @details The profile represents this operation as regular-Scale-prefixed
/// halves with neutral inline scales. That lowering is not yet implemented, so
/// this rule fails closed.
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

/// @brief Convert B0 Scale16 WMMA to exact A0 regular-Scale WMMAs.
///
/// @details The two passes consume the even and odd Scale16 bytes and mutually
/// exclusive K=16 portions of matrix A. Their results accumulate through D.
/// The B0-only 32x16 FP4 form additionally splits M into two 16-row halves,
/// producing four passes. On gfx1250, lanes 0..15 and
/// 16..31 of the same A-scale pair correspond to those two M halves; the
/// replacement regular-Scale prefix selects the half with SCL_OPSEL. Each M
/// half slices eight VGPRs from A, C, and D while sharing B.
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
  if ((scale_a & 1u) != 0 || (scale_b & 1u) != 0) {
    return ExpandResult::failed(
        "gfx1250 Scale16 WMMA scale pair is not even-aligned as required by the ISA");
  }

  if (matrix.src0 < kVgprEncoding || matrix.src1 < kVgprEncoding) {
    return ExpandResult::failed("gfx1250 Scale16 WMMA matrix inputs are not VGPR ranges");
  }
  const uint16_t matrix_a = static_cast<uint16_t>(matrix.src0 - kVgprEncoding);
  const uint16_t matrix_b = static_cast<uint16_t>(matrix.src1 - kVgprEncoding);
  const bool is_m32 = matrix.op == gfx1250::kVWmmaF3232x16x128F4Vop3p;

  const uint8_t matrix_a_fmt = is_m32 ? 4u : static_cast<uint8_t>(matrix.opsel);
  const uint8_t matrix_b_fmt =
      is_m32 ? 4u : static_cast<uint8_t>((matrix.pad_14 != 0 ? 4u : 0u) | matrix.opsel_hi);
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
  const uint16_t matrix_a_total_width = static_cast<uint16_t>(matrix_a_width * (is_m32 ? 2u : 1u));
  const uint16_t matrix_b_width = matrix_width(matrix_b_fmt);
  if (matrix_a_width == 0 || matrix_b_width == 0) {
    return ExpandResult::failed("gfx1250 Scale16 WMMA has an unknown matrix format");
  }
  // gfx1250 permits one VALU co-execution slot for an F8F6F4 WMMA only when
  // both matrix inputs are FP4. Every other F8F6F4 combination permits three.
  // Generated VALU that overwrites masked-A must cover the complete window.
  const int wmma_valu_hazard_slots = matrix_a_fmt == 4u && matrix_b_fmt == 4u ? 1 : 3;

  const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
  const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  if (!src0_bank || !src1_bank || !src2_bank || !dst_bank) {
    return ExpandResult::failed("gfx1250 Scale16 WMMA cannot prove the VGPR-MSB mode");
  }
  const uint8_t original_mode =
      static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) | (*src2_bank << 4) | (*dst_bank << 6));
  const uint32_t matrix_a_absolute = matrix_a + static_cast<uint32_t>(*src0_bank) * 256u;
  const uint32_t matrix_b_absolute = matrix_b + static_cast<uint32_t>(*src1_bank) * 256u;
  if (matrix_a_absolute + matrix_a_total_width > REGISTER_SET_MAX_VGPRS ||
      matrix_b_absolute + matrix_b_width > REGISTER_SET_MAX_VGPRS) {
    return ExpandResult::failed("gfx1250 Scale16 WMMA matrix range exceeds the VGPR file");
  }

  const uint16_t destination_dwords = static_cast<uint16_t>(is_m32 ? 16u : 8u);
  const bool src2_is_vgpr = matrix.src2 >= kVgprEncoding;
  const uint16_t matrix_c = src2_is_vgpr ? static_cast<uint16_t>(matrix.src2 - kVgprEncoding) : 0u;
  const uint32_t matrix_c_absolute = matrix_c + static_cast<uint32_t>(*src2_bank) * 256u;
  const uint32_t destination_absolute = matrix.vdst + static_cast<uint32_t>(*dst_bank) * 256u;

  RegisterSet architectural_vgprs;
  architectural_vgprs.expand(RegisterRef{RegClass::VGPR, scale_a, static_cast<uint8_t>(2)});
  architectural_vgprs.expand(RegisterRef{RegClass::VGPR, scale_b, static_cast<uint8_t>(2)});
  architectural_vgprs.expand(RegisterRef{RegClass::VGPR, static_cast<uint16_t>(matrix_a_absolute),
                                         static_cast<uint8_t>(matrix_a_total_width)});
  architectural_vgprs.expand(RegisterRef{RegClass::VGPR, static_cast<uint16_t>(matrix_b_absolute),
                                         static_cast<uint8_t>(matrix_b_width)});
  architectural_vgprs.expand(RegisterRef{RegClass::VGPR,
                                         static_cast<uint16_t>(destination_absolute),
                                         static_cast<uint8_t>(destination_dwords)});
  if (src2_is_vgpr) {
    architectural_vgprs.expand(RegisterRef{RegClass::VGPR, static_cast<uint16_t>(matrix_c_absolute),
                                           static_cast<uint8_t>(destination_dwords)});
  }

  SemanticScratchAllocator allocator(
      inst, liveness, context,
      SemanticScratchPolicy{.max_vgprs = 256,
                            .max_spill_dword_offset = kGfx1250ScratchMaxDwordOffset});
  SemanticScratchRequest matrix_request;
  matrix_request.count = static_cast<uint16_t>(matrix_a_width + 5u);
  matrix_request.alignment = 2;
  matrix_request.forbidden = architectural_vgprs;
  matrix_request.allow_spill = true;
  const SemanticScratchResult matrix_scratch = allocator.acquire_vgprs(matrix_request);
  if (!matrix_scratch) {
    return ExpandResult::failed(
        "gfx1250 Scale16 WMMA could not allocate its contiguous scratch range");
  }
  const uint16_t masked_a = matrix_scratch.lease->base;

  std::array<uint16_t, 5> scalar_scratch{};
  for (uint16_t i = 0; i < scalar_scratch.size(); ++i) {
    scalar_scratch[i] = static_cast<uint16_t>(masked_a + matrix_a_width + i);
  }
  const uint16_t scale_a_lo = scalar_scratch[0];
  const uint16_t scale_b_lo = scalar_scratch[1];
  const uint16_t scale_a_hi = scalar_scratch[2];
  const uint16_t scale_b_hi = scalar_scratch[3];
  const uint16_t temp = scalar_scratch[4];

  const auto overlaps = [](uint32_t lhs, uint16_t lhs_count, uint32_t rhs, uint16_t rhs_count) {
    return lhs < rhs + rhs_count && rhs < lhs + lhs_count;
  };
  constexpr uint16_t kMHalfDwords = 8;
  const bool dst_crosses = is_m32 && static_cast<uint32_t>(matrix.vdst) + kMHalfDwords > 0xffu;
  const bool src2_crosses =
      is_m32 && src2_is_vgpr && static_cast<uint32_t>(matrix_c) + kMHalfDwords > 0xffu;
  if ((dst_crosses && *dst_bank == 3) || (src2_crosses && *src2_bank == 3)) {
    return ExpandResult::failed("gfx1250 Scale16 M=32 C/D split exceeds the VGPR address space");
  }
  if (destination_absolute + destination_dwords > REGISTER_SET_MAX_VGPRS ||
      (src2_is_vgpr && matrix_c_absolute + destination_dwords > REGISTER_SET_MAX_VGPRS)) {
    return ExpandResult::failed("gfx1250 Scale16 WMMA C/D range exceeds the VGPR file");
  }

  if (overlaps(destination_absolute, destination_dwords, matrix_a_absolute, matrix_a_total_width) ||
      overlaps(destination_absolute, destination_dwords, matrix_b_absolute, matrix_b_width)) {
    return ExpandResult::failed("gfx1250 Scale16 destination destructively overlaps matrix A or B");
  }
  if (is_m32 && src2_is_vgpr && overlaps(destination_absolute, 8, matrix_c_absolute + 8u, 8)) {
    return ExpandResult::failed("gfx1250 Scale16 lower destination overlaps the upper C half");
  }

  const auto first_wmma_mode = [&](uint16_t m_half) {
    const uint8_t pass_src2_bank =
        static_cast<uint8_t>(*src2_bank + (m_half != 0 && src2_crosses ? 1u : 0u));
    const uint8_t pass_dst_bank =
        static_cast<uint8_t>(*dst_bank + (m_half != 0 && dst_crosses ? 1u : 0u));
    return static_cast<uint8_t>((*src1_bank << 2) | (pass_src2_bank << 4) | (pass_dst_bank << 6));
  };
  const auto accumulate_wmma_mode = [&](uint16_t m_half) {
    const uint8_t pass_dst_bank =
        static_cast<uint8_t>(*dst_bank + (m_half != 0 && dst_crosses ? 1u : 0u));
    return static_cast<uint8_t>((*src1_bank << 2) | (pass_dst_bank << 4) | (pass_dst_bank << 6));
  };

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
  uint8_t current_mode = original_mode;
  // Scale operands architecturally ignore VGPR-MSB. The gather instructions
  // are ordinary VALU, so explicitly select low-bank scratch/source registers.
  append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
  if (!append_gfx1250_scratch_preservation(words, *matrix_scratch.lease, false)) {
    return ExpandResult::failed(
        "gfx1250 Scale16 WMMA could not preserve its borrowed masked-A range");
  }
  append_gfx1250_scale16_gather(words, scale_a, scale_a_lo, temp, false);
  append_gfx1250_scale16_gather(words, scale_b, scale_b_lo, temp, false);
  append_gfx1250_scale16_gather(words, scale_a, scale_a_hi, temp, true);
  append_gfx1250_scale16_gather(words, scale_b, scale_b_hi, temp, true);

  const auto append_masked_a = [&](uint16_t m_half, bool high) {
    const uint32_t matrix_a_half =
        matrix_a_absolute + static_cast<uint32_t>(m_half) * matrix_a_width;
    if (matrix_a_fmt <= 1) {
      append_words(words,
                   gfx1250::build_sop1(gfx1250::kSMovB32Sop1,
                                       {.ssrc0 = 0xff, .sdst = static_cast<uint8_t>(*mask_sgpr)}));
      words.push_back(high ? 0xffff0000u : 0x0000ffffu);
      for (uint16_t i = 0; i < matrix_a_width; ++i) {
        const uint32_t source = matrix_a_half + i;
        const uint8_t source_bank = static_cast<uint8_t>(source / 256u);
        append_gfx1250_vgpr_msb_transition(words, current_mode,
                                           static_cast<uint8_t>(source_bank << 2));
        append_words(words, gfx1250::build_vop3(
                                gfx1250::kVCndmaskB32Vop3,
                                {.vdst = static_cast<uint8_t>(masked_a + i),
                                 .src0 = gfx1250_inline_u32(0),
                                 .src1 = gfx1250_vgpr_src(static_cast<uint16_t>(source & 0xffu)),
                                 .src2 = *mask_sgpr}));
      }
      return;
    }

    const uint16_t subblock_width = matrix_a_fmt <= 3 ? 3 : 2;
    for (uint16_t i = 0; i < matrix_a_width; ++i) {
      const bool is_high_subblock = ((i / subblock_width) & 1u) != 0;
      const uint32_t source_absolute = matrix_a_half + i;
      const uint8_t source_bank = static_cast<uint8_t>(source_absolute / 256u);
      append_gfx1250_vgpr_msb_transition(words, current_mode, source_bank);
      const uint16_t source = is_high_subblock == high
                                  ? gfx1250_vgpr_src(static_cast<uint16_t>(source_absolute & 0xffu))
                                  : gfx1250_inline_u32(0);
      append_words(
          words, gfx1250::build_vop3(gfx1250::kVMovB32Vop3,
                                     {.vdst = static_cast<uint8_t>(masked_a + i), .src0 = source}));
    }
  };

  const auto build_pass = [&](uint16_t m_half, uint16_t pass_scale_a, uint16_t pass_scale_b,
                              bool accumulate_d) {
    std::array<uint32_t, 4> pass = {inst.raw_encoding()[0], inst.raw_encoding()[1],
                                    inst.raw_encoding()[2], inst.raw_encoding()[3]};
    set_word_field(pass[0], kWmmaScaleSrc2PrefixOp, 16, 8);
    // Splitting changes the A/D ranges, so the original reuse promises no
    // longer apply. M=32 assigns its two halves to scale lanes 0..15 and
    // 16..31. M=16 already carries an architectural SCL_OPSEL selection which
    // must survive the Scale16-to-regular-Scale conversion.
    pass[0] &= ~((uint32_t{1} << 13) | (uint32_t{1} << 14));
    set_word_field(pass[0], is_m32 ? m_half : static_cast<uint16_t>(scale.opsel & 1u), 11, 1);
    set_word_field(pass[1], gfx1250_vgpr_src(pass_scale_a), 0, 9);
    set_word_field(pass[1], gfx1250_vgpr_src(pass_scale_b), 9, 9);
    set_word_field(pass[1], gfx1250_vgpr_src(0), 18, 9);
    set_word_field(pass[3], gfx1250_vgpr_src(masked_a), 0, 9);
    if (is_m32) {
      const uint16_t m_delta = static_cast<uint16_t>(m_half * kMHalfDwords);
      set_word_field(pass[2], static_cast<uint32_t>((matrix.vdst + m_delta) & 0xffu), 0, 8);
      set_word_field(pass[2], 4, 11, 3); // matrix A format: FP4
      pass[2] |= uint32_t{1} << 14;      // matrix B format: FP4
      set_word_field(pass[2], gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p, 16, 8);
      set_word_field(pass[3], 0, 27, 2); // remaining B format bits for FP4
      if (!accumulate_d) {
        set_word_field(pass[3],
                       src2_is_vgpr
                           ? gfx1250_vgpr_src(static_cast<uint16_t>((matrix_c + m_delta) & 0xffu))
                           : static_cast<uint16_t>(matrix.src2),
                       18, 9);
      }
    }
    if (accumulate_d) {
      const uint16_t m_delta = static_cast<uint16_t>(is_m32 ? m_half * kMHalfDwords : 0u);
      set_word_field(
          pass[3], gfx1250_vgpr_src(static_cast<uint16_t>((matrix.vdst + m_delta) & 0xffu)), 18, 9);
      pass[2] &= ~(uint32_t{1} << 10);
      pass[3] &= ~(uint32_t{1} << 31);
    }
    words.insert(words.end(), pass.begin(), pass.end());
  };

  const uint16_t m_halves = static_cast<uint16_t>(is_m32 ? 2u : 1u);
  for (uint16_t m_half = 0; m_half < m_halves; ++m_half) {
    append_masked_a(m_half, false);
    append_gfx1250_vgpr_msb_transition(words, current_mode, first_wmma_mode(m_half));
    build_pass(m_half, scale_a_lo, scale_b_lo, false);
    // The next VALU sequence overwrites masked-A while the preceding WMMA may
    // still read it. Cover every format-dependent co-execution slot.
    for (int slot = 0; slot < wmma_valu_hazard_slots; ++slot)
      append_words(words, gfx1250::build_vop1(gfx1250::kVNopVop1));
    append_masked_a(m_half, true);
    append_gfx1250_vgpr_msb_transition(words, current_mode, accumulate_wmma_mode(m_half));
    build_pass(m_half, scale_a_hi, scale_b_hi, true);
    if (m_half + 1u != m_halves) {
      for (int slot = 0; slot < wmma_valu_hazard_slots; ++slot)
        append_words(words, gfx1250::build_vop1(gfx1250::kVNopVop1));
    }
  }
  // The allocator proves only that an unspilled range is dead before the
  // source WMMA. A following source VALU may therefore define masked-A even
  // when no spill restore is needed. Keep every subsequent VALU, including
  // restoration loads, outside the final WMMA input-read window.
  for (int slot = 0; slot < wmma_valu_hazard_slots; ++slot)
    append_words(words, gfx1250::build_vop1(gfx1250::kVNopVop1));
  append_gfx1250_vgpr_msb_transition(words, current_mode, 0);
  if (!append_gfx1250_scratch_preservation(words, *matrix_scratch.lease, true)) {
    return ExpandResult::failed(
        "gfx1250 Scale16 WMMA could not restore its borrowed masked-A range");
  }
  append_gfx1250_vgpr_msb_transition(words, current_mode, original_mode);
  return ExpandResult::success(std::move(words));
}

/// @brief Conservatively separate B0 integer WMMA from its A0 successor.
///
/// @details gfx1250 requires nine separating V_NOPs when dense IU8 WMMA feeds a
/// following XDL matrix input, and five for sparse IU8 SWMMAC. These bounds also
/// cover their shorter WMMA-to-VALU hazard windows. This temporary local
/// lowering does not inspect following control-flow successors, so it appends
/// the full instruction-family bound unconditionally.
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
  const int required_slots = inst.mnemonic() == "v_wmma_i32_16x16x64_iu8" ? 9 : 5;
  // TODO: Replace fixed padding with a whole-kernel lookahead that counts
  // existing V_NOPs/independent VALU and inserts exactly the required slots.
  for (int slot = 0; slot < required_slots; ++slot)
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
    // The emitted ds_store_b32 keeps the original store-data VGPR in data0, and
    // data0 is a Src1-role operand in both ds_store_addtid_b32 and ds_store_b32,
    // so its high bank is src1_bank. The address VGPR is a fresh low-bank
    // scratch, so only the Src1 field needs the original store-data bank.
    const uint8_t ds_mode = static_cast<uint8_t>(*src1_bank << 2);
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
  // The allocator searches only REGISTER_SET_ALLOCATABLE_SGPRS, so successful
  // results are already ordinary encodable SGPRs.
  if (!nan_mask || !exp31_mask) {
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
    // gfx1250 VOP3 compares encode their scalar mask destination in the ordinary
    // VOP3 vdst field. Vop3SdstEnc is a different format whose sdst bits overlap
    // modifiers here; using it leaves vdst=0 and corrupts live s0.
    append_words(words, gfx1250::build_vop3(opcode, {.vdst = sdst, .src0 = 255, .src1 = src1}));
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

/// @brief Wrap a standalone low-precision WMMA in an A0-safe neutral scale prefix.
///
/// @details gfx1250 A0 cannot safely expose the bare F8F6F4 matrix instruction to
/// trap/CWSR recovery. The documented workaround is the regular-Scale four-DWORD
/// form. In the scale-source context, inline integer zero selects the neutral
/// E8M0 scale. Keep the original two-DWORD matrix body byte-for-byte so its
/// formats, accumulator, modifiers, and register operands retain their source
/// semantics. The prefix's otherwise-unused SRC2 must encode VGPR0 to avoid the
/// documented false scalar dependency.
ExpandResult expand_gfx1250_bare_f8f6f4_wmma(const Instruction &inst, uint32_t, uint64_t,
                                             std::span<const uint8_t>, const LivenessAnalysis &,
                                             TranslationContext &, const LaneLayout *,
                                             const LaneLayout *) {
  if (inst.mnemonic() != "v_wmma_f32_16x16x128_f8f6f4" ||
      inst.opcode() != gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p ||
      inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr) {
    return ExpandResult::failed(
        "gfx1250 bare F8F6F4 WMMA rule received an unsupported instruction");
  }

  std::vector<uint32_t> words;
  words.reserve(4);
  constexpr uint16_t kVgprEncoding = 256;
  append_words(words, gfx1250::build_vop3p(kWmmaScaleSrc2PrefixOp, {.src0 = kGfx1250InlineZero,
                                                                    .src1 = kGfx1250InlineZero,
                                                                    .src2 = kVgprEncoding}));
  words.insert(words.end(), inst.raw_encoding(), inst.raw_encoding() + 2);
  return ExpandResult::success(std::move(words));
}

/// @brief Lower a B0-only K=128 FP8/BF8 WMMA for A0.
///
/// @details A0's regular-Scale mixed-format WMMA implements the same f32 K=128
/// operation when both scale sources use the context-sensitive inline-zero
/// encoding for neutral E8M0 scaling. This avoids a temporary SGPR and a
/// scalar-to-vector ALU dependency. Select the FP8/BF8 matrix formats in the
/// mixed base instruction and encode VGPR0 in the regular-Scale prefix's
/// architecturally unused SRC2 field.
///
/// The regular-Scale instruction only has an f32 output. The f16 forms remain
/// fail-closed because chaining two K=64 instructions would round the
/// intermediate accumulator to f16 instead of preserving one final f16 round.
ExpandResult expand_gfx1250_k128_wmma(const Instruction &inst, uint32_t, uint64_t,
                                      std::span<const uint8_t>, const LivenessAnalysis &,
                                      TranslationContext &, const LaneLayout *,
                                      const LaneLayout *) {
  uint8_t matrix_a_fmt = 0;
  uint8_t matrix_b_fmt = 0;
  switch (inst.opcode()) {
  case gfx1250::kVWmmaF3216x16x128Fp8Fp8Vop3p:
    break;
  case gfx1250::kVWmmaF3216x16x128Fp8Bf8Vop3p:
    matrix_b_fmt = 1;
    break;
  case gfx1250::kVWmmaF3216x16x128Bf8Fp8Vop3p:
    matrix_a_fmt = 1;
    break;
  case gfx1250::kVWmmaF3216x16x128Bf8Bf8Vop3p:
    matrix_a_fmt = 1;
    matrix_b_fmt = 1;
    break;
  case gfx1250::kVWmmaF1616x16x128Fp8Fp8Vop3p:
  case gfx1250::kVWmmaF1616x16x128Fp8Bf8Vop3p:
  case gfx1250::kVWmmaF1616x16x128Bf8Fp8Vop3p:
  case gfx1250::kVWmmaF1616x16x128Bf8Bf8Vop3p:
    return ExpandResult::failed(
        "gfx1250 f16 K=128 WMMA A0 lowering is not yet implemented",
        {"Provide an exact packed-f16 accumulator and single-rounding lowering."});
  default:
    return ExpandResult::failed("gfx1250 K=128 WMMA rule received an unsupported opcode");
  }

  if (inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr)
    return ExpandResult::failed("gfx1250 f32 K=128 WMMA rule received a non-base encoding");

  gfx1250::Vop3pMachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  constexpr uint16_t kVgprEncoding = 256;
  if (source.src0 < kVgprEncoding || source.src1 < kVgprEncoding) {
    return ExpandResult::failed("gfx1250 K=128 WMMA matrix operands are not ordinary VGPR ranges");
  }

  std::vector<uint32_t> words;
  words.reserve(4);
  append_words(words, gfx1250::build_vop3p(kWmmaScaleSrc2PrefixOp, {.src0 = kGfx1250InlineZero,
                                                                    .src1 = kGfx1250InlineZero,
                                                                    .src2 = kVgprEncoding}));
  append_words(words, gfx1250::build_vop3p(gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p,
                                           {.vdst = static_cast<uint8_t>(source.vdst),
                                            .neg_hi = static_cast<uint8_t>(source.neg_hi),
                                            .opsel = matrix_a_fmt,
                                            .clamp = static_cast<uint8_t>(source.clamp),
                                            .src0 = static_cast<uint16_t>(source.src0),
                                            .src1 = static_cast<uint16_t>(source.src1),
                                            .src2 = static_cast<uint16_t>(source.src2),
                                            .opsel_hi = matrix_b_fmt,
                                            .neg = static_cast<uint8_t>(source.neg)}));
  return ExpandResult::success(std::move(words));
}

// The semantic translator binary-searches this table, so entries must stay
// sorted by the full encoding ID and then opcode. VDS encoding IDs include the
// high opcode bits, hence the four consecutive kVdsOpHi* groups below.
inline constexpr std::array<TranslationRule, 38> kGfx1250B0ToA0ExpandRules = {{
    {gfx1250::encoding::kSopp, gfx1250::kSClauseSopp, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_s_clause, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3p, gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_bare_f8f6f4_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3p, kWmmaScaleSrc2PrefixOp, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_wmma_scale_src2, nullptr, nullptr},
    {gfx1250::encoding::kVop3p, kWmmaScale16PrefixOp, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_wmma_scale16, nullptr, nullptr},
    {gfx1250::encoding::kVop3p, gfx1250::kVWmmaI3216x16x64Iu8Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_wmma_iu8_spacing, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3p, gfx1250::kVSwmmacI3216x16x128Iu8Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_wmma_iu8_spacing, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF3216x16x128Fp8Fp8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF3216x16x128Fp8Bf8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF3216x16x128Bf8Fp8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF3216x16x128Bf8Bf8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF1616x16x128Fp8Fp8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF1616x16x128Fp8Bf8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF1616x16x128Bf8Fp8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF1616x16x128Bf8Bf8Vop3p, RuleAction::Expand, 0,
     0, nullptr, expand_gfx1250_k128_wmma, nullptr, nullptr, false},
    {gfx1250::encoding::kVop3pOpHi1, gfx1250::kVWmmaF3232x16x128F4Vop3p, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_wmma_32x16_f4, nullptr, nullptr, false},
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
