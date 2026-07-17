// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/gfx1250_b0_to_a0.h"

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
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

/// @brief Exact instruction names whose A0 workaround needs an expansion.
///
/// @details Keep this list aligned with the implemented B0-to-A0 reference
/// patches. Prefix-classified WMMA/SWMMAC and cluster-load instructions are
/// handled separately because their contextual workarounds apply to families.
inline constexpr std::array<std::string_view, 18> kExactErrataMnemonics = {
    "s_barrier_signal_isfirst",
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
    "tensor_load_to_lds",
    "v_cvt_pk_fp8_f32",
    "v_cvt_sr_fp8_f32",
};

[[nodiscard]] bool requires_errata_expansion(std::string_view mnemonic) {
  for (std::string_view exact : kExactErrataMnemonics) {
    if (mnemonic == exact)
      return true;
  }

  // Every cluster-load form needs either demotion to a global load or an M0
  // cluster-mask sequence. Operand inspection will choose the precise rule.
  if (mnemonic.starts_with("cluster_load_"))
    return true;

  // The reference patch accepts every encoding suffix in this conversion
  // family. Its eventual semantic rule will further restrict the failure to
  // the operand/modifier combinations that actually need the A0 workaround.
  if (mnemonic.starts_with("v_cvt_f32_fp8"))
    return true;

  // These eight K=128 FP8/BF8 forms exist on B0 but must be split into K=64
  // operations for A0. Match the closed family precisely: ordinary K=128
  // F8F6F4 is the A0 replacement for another workaround and is not a split
  // candidate itself.
  const bool is_k128_fp8_bf8 = (mnemonic.starts_with("v_wmma_f16_16x16x128_") ||
                                mnemonic.starts_with("v_wmma_f32_16x16x128_")) &&
                               (mnemonic.ends_with("_fp8_fp8") || mnemonic.ends_with("_fp8_bf8") ||
                                mnemonic.ends_with("_bf8_fp8") || mnemonic.ends_with("_bf8_bf8"));
  if (is_k128_fp8_bf8 || mnemonic == "v_wmma_f32_32x16x128_f4")
    return true;

  // Scale16 and regular Scale have separate mandatory encoding/scale-source
  // workarounds. Keep them fail-closed until their semantic rules land.
  if (mnemonic.starts_with("v_wmma_scale"))
    return true;

  // The A0 co-execution distance exceeds B0 only for integer IU8/IU4 WMMA or
  // SWMMAC. FP16/BF16 need four safe slots on both steppings, while floating
  // FP8 forms need no additional A0 padding. The integer forms remain
  // fail-closed until a CFG-aware spacing pass can inspect following VALU.
  const bool is_wmma_like = mnemonic.starts_with("v_wmma_") || mnemonic.starts_with("v_swmmac_");
  return is_wmma_like && (mnemonic.find("_iu8") != std::string_view::npos ||
                          mnemonic.find("_iu4") != std::string_view::npos);
}

/// @brief Append a generated instruction's words to one replacement sequence.
template <size_t N>
void append_words(std::vector<uint32_t> &output, const std::array<uint32_t, N> &words) {
  output.insert(output.end(), words.begin(), words.end());
}

/// @brief True when two raw gfx1250 VGPR slices overlap.
///
/// @details DS2 register fields are eight-bit selectors. The source instruction
/// is already required to encode valid contiguous tuples, so comparing the
/// selected slices is sufficient to choose an order which preserves all source
/// operands until their corresponding replacement has issued.
[[nodiscard]] bool register_slices_overlap(uint8_t lhs, uint8_t lhs_width, uint8_t rhs,
                                           uint8_t rhs_width) {
  const uint16_t lhs_end = static_cast<uint16_t>(lhs) + lhs_width;
  const uint16_t rhs_end = static_cast<uint16_t>(rhs) + rhs_width;
  return lhs < rhs_end && rhs < lhs_end;
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
                                std::span<const uint8_t>, const LivenessAnalysis &,
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
  bool second_first = false;

  if (shape.kind == Gfx1250Ds2Shape::Kind::Load) {
    // The compound load captures ADDR before writing either destination half.
    // If the first half aliases ADDR, issue the independent second load first.
    second_first = register_slices_overlap(static_cast<uint8_t>(source.vdst), shape.element_dwords,
                                           static_cast<uint8_t>(source.addr), 1);
  } else if (shape.kind == Gfx1250Ds2Shape::Kind::StoreExchange) {
    // Each exchange writes one destination half while the other still needs
    // ADDR and its input data. Pick a safe direction. A dependency in both
    // directions needs scratch storage and must fail closed for now.
    const uint8_t first_dst = static_cast<uint8_t>(source.vdst);
    const uint8_t second_dst = static_cast<uint8_t>(source.vdst + shape.element_dwords);
    const bool first_clobbers_second =
        register_slices_overlap(first_dst, shape.element_dwords, static_cast<uint8_t>(source.addr),
                                1) ||
        register_slices_overlap(first_dst, shape.element_dwords, static_cast<uint8_t>(source.data1),
                                shape.element_dwords);
    const bool second_clobbers_first =
        register_slices_overlap(second_dst, shape.element_dwords, static_cast<uint8_t>(source.addr),
                                1) ||
        register_slices_overlap(second_dst, shape.element_dwords,
                                static_cast<uint8_t>(source.data0), shape.element_dwords);
    if (first_clobbers_second && second_clobbers_first) {
      return ExpandResult::failed("gfx1250 DS2 exchange has cyclic destination/source overlap",
                                  {"Add a scratch-VGPR DS2 exchange lowering for cyclic overlap."});
    }
    second_first = first_clobbers_second;
  }

  std::vector<uint32_t> words;
  words.reserve(5);
  if (second_first) {
    append_words(words, second);
    append_words(words, first);
  } else {
    append_words(words, first);
    append_words(words, second);
  }
  append_words(words, gfx1250::build_sopp(gfx1250::kSWaitDscntSopp, {.simm16 = 0}));
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

  std::vector<uint32_t> words(inst.raw_encoding(), inst.raw_encoding() + 4);
  // Instruction bits [58:50] occupy word 1 bits [26:18].
  set_word_field(words[1], 0x100, 18, 9);
  return ExpandResult::success(std::move(words));
}

/// @brief Split a B0-only K=128 FP8/BF8 WMMA into two A0 K=64 WMMAs.
///
/// @details WMMA computes D=A*B+C. The first instruction consumes the low
/// eight-VGPR halves of A and B with the original C. The second consumes the
/// high halves and feeds the first D back as C. Matrix-reuse flags are cleared
/// because each half names a different A/B range, and C negation bits are
/// cleared on the second instruction because its C is the intermediate D.
ExpandResult expand_gfx1250_k128_wmma(const Instruction &inst, uint32_t, uint64_t offset,
                                      std::span<const uint8_t> source_text,
                                      const LivenessAnalysis &liveness, TranslationContext &,
                                      const LaneLayout *, const LaneLayout *) {
  const uint16_t replacement_opcode = gfx1250_k128_wmma_replacement(inst.opcode());
  if (replacement_opcode == 0)
    return ExpandResult::failed("gfx1250 K=128 WMMA rule received an unsupported opcode");
  // The first K64 instruction writes the intermediate through VDST. The
  // second reads that same encoded register through SRC2, so its physical bank
  // is preserved only when the two role fields agree at this program point.
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  if (!dst_bank || !src2_bank || *dst_bank != *src2_bank) {
    return ExpandResult::failed(
        "gfx1250 K=128 WMMA split cannot prove compatible VDST and SRC2 VGPR-MSB banks",
        {"Make the VGPR-MSB fields agree on every CFG path reaching this instruction."});
  }
  if (offset > source_text.size() ||
      static_cast<size_t>(inst.size()) > source_text.size() - offset ||
      inst.size() < static_cast<int>(sizeof(gfx1250::Vop3pMachineInst))) {
    return ExpandResult::failed("gfx1250 K=128 WMMA has no complete source encoding");
  }

  const size_t source_word_count = static_cast<size_t>(inst.size()) / sizeof(uint32_t);
  std::vector<uint32_t> first(source_word_count);
  std::memcpy(first.data(), source_text.data() + offset, static_cast<size_t>(inst.size()));
  std::array<uint32_t, 2> second = {first[0], first[1]};

  // Both K64 instructions use the replacement opcode and must discard matrix
  // reuse. gfx1250 encodes matrix_a_reuse in bit 13 and matrix_b_reuse in bit
  // 14; the generated generic VOP3P struct intentionally names those bits as
  // format padding, so manipulate the audited encoding bits directly here.
  set_word_field(first[0], replacement_opcode, 16, 8);
  first[0] &= ~((uint32_t{1} << 13) | (uint32_t{1} << 14));
  set_word_field(second[0], replacement_opcode, 16, 8);
  second[0] &= ~((uint32_t{1} << 13) | (uint32_t{1} << 14));

  const uint16_t src0 = static_cast<uint16_t>(first[1] & 0x1ff);
  const uint16_t src1 = static_cast<uint16_t>((first[1] >> 9) & 0x1ff);
  constexpr uint16_t kVgprOperandEncoding = 256;
  constexpr uint16_t kHalfInputDwords = 8;
  if (src0 < kVgprOperandEncoding || src1 < kVgprOperandEncoding) {
    return ExpandResult::failed("gfx1250 K=128 WMMA A/B operands are not splittable VGPR ranges");
  }

  const uint16_t src0_low = static_cast<uint16_t>(src0 - kVgprOperandEncoding);
  const uint16_t src1_low = static_cast<uint16_t>(src1 - kVgprOperandEncoding);
  const bool src0_crosses_bank = src0_low + kHalfInputDwords > 0xff;
  const bool src1_crosses_bank = src1_low + kHalfInputDwords > 0xff;

  // A gfx1250 tuple may naturally cross v255/v256, so a K=128 input starting
  // at (for example) v250 is legal. Its second K64 half starts at physical
  // v258, which must be encoded as low byte 2 under the next SRC bank. Build a
  // temporary mode only when one of the two input bases wraps.
  std::optional<uint8_t> original_mode;
  std::optional<uint8_t> second_mode;
  if (src0_crosses_bank || src1_crosses_bank) {
    const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
    const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
    if (!src0_bank || !src1_bank || !dst_bank || !src2_bank) {
      return ExpandResult::failed(
          "gfx1250 K=128 WMMA split cannot prove VGPR-MSB state for a bank-crossing input");
    }
    if ((src0_crosses_bank && *src0_bank == 3) || (src1_crosses_bank && *src1_bank == 3)) {
      return ExpandResult::failed(
          "gfx1250 K=128 WMMA bank-crossing input exceeds the addressable VGPR range");
    }

    original_mode =
        static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) | (*src2_bank << 4) | (*dst_bank << 6));
    second_mode = static_cast<uint8_t>((*src0_bank + (src0_crosses_bank ? 1u : 0u)) |
                                       ((*src1_bank + (src1_crosses_bank ? 1u : 0u)) << 2) |
                                       (*src2_bank << 4) | (*dst_bank << 6));
  }

  set_word_field(second[1], kVgprOperandEncoding + ((src0_low + kHalfInputDwords) & 0xffu), 0, 9);
  set_word_field(second[1], kVgprOperandEncoding + ((src1_low + kHalfInputDwords) & 0xffu), 9, 9);
  const uint16_t vdst_as_src2 = static_cast<uint16_t>(kVgprOperandEncoding + (first[0] & 0xff));
  set_word_field(second[1], vdst_as_src2, 18, 9);
  second[0] &= ~(uint32_t{1} << 10); // neg_hi[2] applies to the original C.
  second[1] &= ~(uint32_t{1} << 31); // neg_lo[2] applies to the original C.

  std::vector<uint32_t> words;
  words.reserve(first.size() + second.size() + (second_mode ? 2 : 0));
  words.insert(words.end(), first.begin(), first.end());
  if (second_mode) {
    append_words(words, gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = *second_mode}));
  }
  append_words(words, second);
  if (original_mode) {
    append_words(words, gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = *original_mode}));
  }
  return ExpandResult::success(std::move(words));
}

/// @brief Mask the tensor D# group-1 workgroup mask for A0.
///
/// @details `tensor_load_to_lds` names the eight-SGPR D# group 1 through
/// VIMAGE.VADDR1. Its first SGPR bits [15:0] are `workgroup_mask`, which routes
/// multicast loads through cluster operations on B0. A0 requires that field to
/// be zero. Preserve the guest value through a dead scratch SGPR because the D#
/// group is a source operand and may be reused after the tensor load.
ExpandResult expand_gfx1250_tensor_load_to_lds(const Instruction &inst, uint32_t, uint64_t,
                                               std::span<const uint8_t>,
                                               const LivenessAnalysis &liveness,
                                               TranslationContext &, const LaneLayout *,
                                               const LaneLayout *) {
  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || static_cast<size_t>(inst.size()) < sizeof(gfx1250::VimageMachineInst)) {
    return ExpandResult::failed("gfx1250 tensor_load_to_lds has no complete VIMAGE encoding");
  }

  gfx1250::VimageMachineInst source{};
  std::memcpy(&source, raw, sizeof(source));
  constexpr uint16_t kDescriptorSgprs = 8;
  constexpr uint16_t kMaxGfx1250Sgprs = 106;
  const uint16_t descriptor_base = source.vaddr1;
  if (static_cast<uint32_t>(descriptor_base) + kDescriptorSgprs > kMaxGfx1250Sgprs) {
    return ExpandResult::failed(
        "gfx1250 tensor_load_to_lds group descriptor is not an encodable SGPR tuple");
  }

  // gfx1250 allocates all 106 normal SGPRs to every wave; there is no
  // descriptor-declared unused tail that a lowering may claim. Reuse only a
  // register proven dead by kernel CFG liveness. This target-specific scan
  // includes s102-s105, which the generic cross-ISA allocator intentionally
  // excludes because CDNA exposes only 102 ordinary SGPRs. VCC in s106:s107,
  // trap temporaries, and the special/hidden s124:s127 locations are never
  // candidates.
  const RegisterSet &live = liveness.live_before(inst);
  std::optional<uint16_t> scratch;
  for (uint16_t candidate = 0; candidate < kMaxGfx1250Sgprs; ++candidate) {
    if (!live.contains({RegClass::SGPR, candidate, 1})) {
      scratch = candidate;
      break;
    }
  }
  if (!scratch) {
    return ExpandResult::failed(
        "gfx1250 tensor_load_to_lds could not find a dead SGPR for descriptor preservation",
        {"Add scalar spilling for gfx1250 semantic lowerings or reduce SGPR pressure."});
  }

  std::vector<uint32_t> words;
  words.reserve(3 + static_cast<size_t>(inst.size()) / sizeof(uint32_t));
  append_words(words, gfx1250::build_sop1(gfx1250::kSMovB32Sop1,
                                          {.ssrc0 = static_cast<uint8_t>(descriptor_base),
                                           .sdst = static_cast<uint8_t>(*scratch)}));
  // S_PACK_HH_B32_B16 D, S0, S1 computes D={S1[31:16],S0[31:16]}.
  // Inline constant 0 is selector 128, so this preserves bits [31:16] and
  // clears D#.workgroup_mask in bits [15:0].
  append_words(words, gfx1250::build_sop2(gfx1250::kSPackHhB32B16Sop2,
                                          {.ssrc0 = 128,
                                           .ssrc1 = static_cast<uint8_t>(descriptor_base),
                                           .sdst = static_cast<uint8_t>(descriptor_base)}));
  words.insert(words.end(), raw, raw + inst.size() / sizeof(uint32_t));
  append_words(words, gfx1250::build_sop1(gfx1250::kSMovB32Sop1,
                                          {.ssrc0 = static_cast<uint8_t>(*scratch),
                                           .sdst = static_cast<uint8_t>(descriptor_base)}));

  return ExpandResult::success(std::move(words));
}

// The semantic translator binary-searches this table, so entries must stay
// sorted by the full encoding ID and then opcode. VDS encoding IDs include the
// high opcode bits, hence the four consecutive kVdsOpHi* groups below.
inline constexpr std::array<TranslationRule, 22> kGfx1250B0ToA0ExpandRules = {{
    {gfx1250::encoding::kVop3p, 0x35, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_wmma_scale_src2, nullptr, nullptr},
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
    {gfx1250::encoding::kVimage, gfx1250::kTensorLoadToLdsVimage, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_tensor_load_to_lds, nullptr, nullptr},
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
}};

} // namespace

const InstructionLegalization *gfx1250_b0_to_a0_legalization(const Instruction &inst) {
  if (!requires_errata_expansion(inst.mnemonic()))
    return nullptr;

  // The runtime uses only the action and target opcode for this initial
  // semantic failure. Source keys remain zero because matching is performed on
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

std::span<const TranslationRule> semantic_expand_rules_gfx1250_b0_to_a0() {
  return kGfx1250B0ToA0ExpandRules;
}

} // namespace rocjitsu
