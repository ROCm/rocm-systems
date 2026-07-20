// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/gfx1250_b0_to_a0.h"

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
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

/// @brief Exact instruction names whose A0 workaround needs an expansion.
///
/// @details Keep this list aligned with the implemented B0-to-A0 reference
/// patches. Prefix-classified WMMA/SWMMAC and cluster-load instructions are
/// handled separately because their contextual workarounds apply to families.
inline constexpr std::array<std::string_view, 19> kExactErrataMnemonics = {
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
    "v_cvt_pk_fp8_f32",
    "v_cvt_sr_fp8_f32",
    "tensor_load_to_lds",
};

[[nodiscard]] bool requires_errata_expansion(std::string_view mnemonic) {
  // This is deliberately more conservative than the reference patch
  // patterns. Rocjitsu relocates and expands instructions, so it cannot retain
  // a source clause without revalidating the translated membership and
  // placement constraints.
  if (mnemonic == "s_clause")
    return true;

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

/// @brief True when a B0 FP8 conversion selects the B0-only E5M3 mode.
///
/// @details The affected VOP3 conversions reuse CLAMP as the E5M3 selector on
/// B0. A0 implements the same CLAMP=0 E4M3 operation, so those instructions
/// must remain on the ordinary byte-copy path. Only the eight-byte VOP3 form
/// carries this selector.
[[nodiscard]] bool requires_fp8_clamp_emulation(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  const bool affected = mnemonic == "v_cvt_pk_fp8_f32" || mnemonic == "v_cvt_sr_fp8_f32" ||
                        mnemonic.starts_with("v_cvt_f32_fp8");
  if (!affected || inst.size() != static_cast<int>(sizeof(gfx1250::Vop3MachineInst)) ||
      inst.raw_encoding() == nullptr)
    return false;

  gfx1250::Vop3MachineInst encoding{};
  std::memcpy(&encoding, inst.raw_encoding(), sizeof(encoding));
  return encoding.clamp != 0;
}

/// @brief Append a generated instruction's words to one replacement sequence.
template <size_t N>
void append_words(std::vector<uint32_t> &output, const std::array<uint32_t, N> &words) {
  output.insert(output.end(), words.begin(), words.end());
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

/// @brief Disable Tensor-DMA multicast for one A0 tensor load.
///
/// @details TENSOR_LOAD_TO_LDS does not encode multicast in the instruction.
/// Descriptor group 1 bits [15:0], held in the first SGPR named by VADDR1,
/// select the workgroups which receive a multicast load.  On A0 those bits
/// must therefore be cleared for every tensor load; inspecting only the
/// instruction cannot prove that the runtime descriptor mask is zero.
/// Preserve the guest descriptor value around the load because later tensor
/// instructions commonly reuse and update the same descriptor.
ExpandResult expand_gfx1250_tensor_load_to_lds(
    const Instruction &inst, uint32_t, uint64_t, std::span<const uint8_t>,
    const LivenessAnalysis &liveness, TranslationContext &, const LaneLayout *,
    const LaneLayout *) {
  if (inst.mnemonic() != "tensor_load_to_lds" ||
      inst.size() != static_cast<int>(sizeof(gfx1250::VimageMachineInst)) ||
      inst.raw_encoding() == nullptr) {
    return ExpandResult::failed(
        "gfx1250 tensor-load mask rule received an unsupported instruction");
  }

  gfx1250::VimageMachineInst source{};
  std::memcpy(&source, inst.raw_encoding(), sizeof(source));
  constexpr uint8_t kSgprNull = 124;
  constexpr uint8_t kLastOrdinarySgpr = 105;
  const uint8_t descriptor_base = static_cast<uint8_t>(source.vaddr1);
  if (descriptor_base == kSgprNull || descriptor_base > kLastOrdinarySgpr - 7u) {
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
  append_words(words,
               gfx1250::build_sop1(gfx1250::kSMovB32Sop1,
                                   {.ssrc0 = descriptor_base,
                                    .sdst = static_cast<uint8_t>(*scratch)}));
  // PACK_HH forms {SRC1[31:16], SRC0[31:16]}. Inline zero as SRC0 clears
  // D1[15:0] while preserving all descriptor fields in D1[31:16].
  append_words(words,
               gfx1250::build_sop2(gfx1250::kSPackHhB32B16Sop2,
                                   {.ssrc0 = 128,
                                    .ssrc1 = descriptor_base,
                                    .sdst = descriptor_base}));
  words.insert(words.end(), inst.raw_encoding(),
               inst.raw_encoding() + sizeof(gfx1250::VimageMachineInst) / sizeof(uint32_t));
  append_words(words,
               gfx1250::build_sop1(gfx1250::kSMovB32Sop1,
                                   {.ssrc0 = static_cast<uint8_t>(*scratch),
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

  std::vector<uint32_t> words(inst.raw_encoding(), inst.raw_encoding() + 4);
  // Instruction bits [58:50] occupy word 1 bits [26:18].
  set_word_field(words[1], 0x100, 18, 9);
  return ExpandResult::success(std::move(words));
}

/// @brief Encode an inline non-negative integer accepted by a VALU source.
[[nodiscard]] constexpr uint16_t gfx1250_inline_u32(uint16_t value) {
  return static_cast<uint16_t>(128u + value);
}

/// @brief Encode one low-bank VGPR as a generic VALU source operand.
[[nodiscard]] constexpr uint16_t gfx1250_vgpr_src(uint16_t vgpr) {
  return static_cast<uint16_t>(256u + vgpr);
}

/// @brief Emit one block-16 to block-32 scale reduction.
///
/// @details A Scale16 operand is a pair of VGPRs containing eight byte scales.
/// Adjacent byte pairs describe the two block-16 halves of one A0 block-32
/// group. Taking the unsigned maximum of each pair produces the conservative
/// four-byte A0 scale operand recommended for exponent scales. All emitted
/// instructions are VOP3 forms so the temporary zero VGPR-MSB mode applies to
/// every register role uniformly.
void append_gfx1250_scale16_reduction(std::vector<uint32_t> &words, uint16_t src_lo,
                                      uint16_t dst, uint16_t temp1, uint16_t temp2) {
  const auto vgpr = gfx1250_vgpr_src;
  const auto bfe = [&](uint16_t out, uint16_t src, uint16_t bit) {
    append_words(words, gfx1250::build_vop3(
                            gfx1250::kVBfeU32Vop3,
                            {.vdst = static_cast<uint8_t>(out),
                             .src0 = vgpr(src),
                             .src1 = gfx1250_inline_u32(bit),
                             .src2 = gfx1250_inline_u32(8)}));
  };
  const auto maximum = [&](uint16_t out, uint16_t lhs, uint16_t rhs) {
    append_words(words, gfx1250::build_vop3(
                            gfx1250::kVMaxU32Vop3,
                            {.vdst = static_cast<uint8_t>(out),
                             .src0 = vgpr(lhs),
                             .src1 = vgpr(rhs)}));
  };
  const auto insert = [&](uint16_t shift) {
    append_words(words, gfx1250::build_vop3(
                            gfx1250::kVLshlOrB32Vop3,
                            {.vdst = static_cast<uint8_t>(dst),
                             .src0 = vgpr(temp1),
                             .src1 = gfx1250_inline_u32(shift),
                             .src2 = vgpr(dst)}));
  };

  bfe(temp1, src_lo, 0);
  bfe(temp2, src_lo, 8);
  maximum(dst, temp1, temp2);

  bfe(temp1, src_lo, 16);
  bfe(temp2, src_lo, 24);
  maximum(temp1, temp1, temp2);
  insert(8);

  bfe(temp1, static_cast<uint16_t>(src_lo + 1u), 0);
  bfe(temp2, static_cast<uint16_t>(src_lo + 1u), 8);
  maximum(temp1, temp1, temp2);
  insert(16);

  bfe(temp1, static_cast<uint16_t>(src_lo + 1u), 16);
  bfe(temp2, static_cast<uint16_t>(src_lo + 1u), 24);
  maximum(temp1, temp1, temp2);
  insert(24);
}

/// @brief Convert B0 Scale16 WMMA to A0 regular Scale WMMA.
///
/// @details The B0 prefix consumes two-VGPR block-16 scale operands. A0's
/// regular Scale prefix consumes one reduced VGPR per matrix. The B0-only
/// 32x16 FP4 form is additionally split along M into two A0 16x16 F8F6F4
/// instructions with both matrix formats forced to FP4. The halves share B
/// and reduced scales while D, A, and a VGPR C operand are sliced by eight
/// dwords. Inline C is copied to both halves.
ExpandResult expand_gfx1250_wmma_scale16(const Instruction &inst, uint32_t, uint64_t,
                                         std::span<const uint8_t>,
                                         const LivenessAnalysis &liveness,
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
  if (scale.op != 0x3a ||
      (matrix.op != gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p &&
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

  SemanticScratchAllocator allocator(
      inst, liveness, context,
      SemanticScratchPolicy{.max_vgprs = 256, .max_spill_dword_offset = 0});
  SemanticScratchRequest request;
  request.count = 4;
  request.allow_spill = false;
  const SemanticScratchResult scratch = allocator.acquire_vgprs(request);
  if (!scratch) {
    return ExpandResult::failed(
        "gfx1250 Scale16 WMMA could not allocate four dead low-bank scratch VGPRs",
        {"Provide four dead VGPRs below v256 for scale reduction."});
  }
  const uint16_t reduced_a = scratch.lease->base;
  const uint16_t reduced_b = static_cast<uint16_t>(reduced_a + 1u);
  const uint16_t temp1 = static_cast<uint16_t>(reduced_a + 2u);
  const uint16_t temp2 = static_cast<uint16_t>(reduced_a + 3u);

  const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
  const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  if (!src0_bank || !src1_bank || !src2_bank || !dst_bank) {
    return ExpandResult::failed("gfx1250 Scale16 WMMA cannot prove the VGPR-MSB mode");
  }
  const uint8_t original_mode = static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) |
                                                      (*src2_bank << 4) | (*dst_bank << 6));

  std::vector<uint32_t> words;
  words.reserve(64);
  if (original_mode != 0)
    append_words(words, gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0}));
  append_gfx1250_scale16_reduction(words, scale_a, reduced_a, temp1, temp2);
  append_gfx1250_scale16_reduction(words, scale_b, reduced_b, temp1, temp2);
  if (original_mode != 0)
    append_words(words,
                 gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = original_mode}));

  const auto build_scale_prefix = [&](uint8_t a_row) {
    // Matrix reuse bits describe the original B64 operands and must not carry
    // into the reduced B32 instruction. Preserve only scale-row and scale-format fields.
    return gfx1250::build_vop3p(
        0x35,
        {.neg_hi = static_cast<uint8_t>(scale.neg_hi & 0x3u),
         .opsel = static_cast<uint8_t>(a_row & 0x1u),
         .src0 = gfx1250_vgpr_src(reduced_a),
         .src1 = gfx1250_vgpr_src(reduced_b),
         .src2 = gfx1250_vgpr_src(0),
         .opsel_hi = static_cast<uint8_t>(scale.opsel_hi & 0x1u),
         .neg = static_cast<uint8_t>(scale.neg & 0x3u)});
  };

  if (matrix.op == gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p) {
    append_words(words, build_scale_prefix(static_cast<uint8_t>(scale.opsel & 0x1u)));
    words.insert(words.end(), inst.raw_encoding() + 2, inst.raw_encoding() + 4);
    return ExpandResult::success(std::move(words));
  }

  if (matrix.src0 < kVgprEncoding || matrix.src1 < kVgprEncoding) {
    return ExpandResult::failed("gfx1250 32x16 Scale16 matrix inputs are not VGPR ranges");
  }
  const bool src2_is_vgpr = matrix.src2 >= kVgprEncoding;
  const uint16_t src0_low = static_cast<uint16_t>(matrix.src0 - kVgprEncoding);
  const uint16_t src1_low = static_cast<uint16_t>(matrix.src1 - kVgprEncoding);
  const uint16_t src2_low =
      src2_is_vgpr ? static_cast<uint16_t>(matrix.src2 - kVgprEncoding) : 0;
  constexpr uint16_t kHalfDwords = 8;
  const bool dst_crosses = static_cast<uint32_t>(matrix.vdst) + kHalfDwords > 0xffu;
  const bool src0_crosses = static_cast<uint32_t>(src0_low) + kHalfDwords > 0xffu;
  const bool src2_crosses =
      src2_is_vgpr && static_cast<uint32_t>(src2_low) + kHalfDwords > 0xffu;
  if ((dst_crosses && *dst_bank == 3) || (src0_crosses && *src0_bank == 3) ||
      (src2_crosses && *src2_bank == 3)) {
    return ExpandResult::failed("gfx1250 32x16 Scale16 split exceeds the VGPR address space");
  }

  for (uint16_t half = 0; half < 2; ++half) {
    std::optional<uint8_t> half_mode;
    if (half == 1 && (dst_crosses || src0_crosses || src2_crosses)) {
      half_mode = static_cast<uint8_t>((*src0_bank + (src0_crosses ? 1u : 0u)) |
                                       (*src1_bank << 2) |
                                       ((*src2_bank + (src2_crosses ? 1u : 0u)) << 4) |
                                       ((*dst_bank + (dst_crosses ? 1u : 0u)) << 6));
      append_words(words,
                   gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = *half_mode}));
    }

    append_words(words, build_scale_prefix(static_cast<uint8_t>(half)));
    const uint16_t delta = static_cast<uint16_t>(half * kHalfDwords);
    const uint16_t half_src2 =
        src2_is_vgpr
            ? static_cast<uint16_t>(kVgprEncoding + ((src2_low + delta) & 0xffu))
            : static_cast<uint16_t>(matrix.src2);
    auto wmma = gfx1250::build_vop3p(
        gfx1250::kVWmmaF3216x16x128F8f6f4Vop3p,
        {.vdst = static_cast<uint8_t>(matrix.vdst + delta),
         .neg_hi = static_cast<uint8_t>(matrix.neg_hi),
         .opsel = 4,
         .clamp = static_cast<uint8_t>(matrix.clamp),
         .src0 = static_cast<uint16_t>(kVgprEncoding + ((src0_low + delta) & 0xffu)),
         .src1 = static_cast<uint16_t>(kVgprEncoding + src1_low),
         .src2 = half_src2,
         .opsel_hi = 0,
         .neg = static_cast<uint8_t>(matrix.neg)});
    // MATRIX_FMT_FP4 for B is {pad_14,opsel_hi}=4.
    wmma[0] |= uint32_t{1} << 14;
    append_words(words, wmma);

    if (half_mode)
      append_words(words,
                   gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = original_mode}));
  }
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
  if (inst.mnemonic() != "v_wmma_i32_16x16x64_iu8")
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
  // second reads that same encoded register through SRC2. When those roles use
  // different banks in the B0 instruction, temporarily select VDST's bank for
  // SRC2 around only the second K64 instruction.
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Dst);
  const auto src2_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src2);
  if (!dst_bank || !src2_bank) {
    return ExpandResult::failed(
        "gfx1250 K=128 WMMA split cannot prove VDST and SRC2 VGPR-MSB banks",
        {"Make the VGPR-MSB fields known on every CFG path reaching this instruction."});
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
  // temporary mode when an input base wraps or when the intermediate must be
  // read through a different SRC2 bank than the original accumulator.
  std::optional<uint8_t> original_mode;
  std::optional<uint8_t> second_mode;
  if (src0_crosses_bank || src1_crosses_bank || *src2_bank != *dst_bank) {
    const auto src0_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src0);
    const auto src1_bank = liveness.vgpr_msb_bank_before(inst, amdgpu::VgprMsbRole::Src1);
    if (!src0_bank || !src1_bank || !dst_bank || !src2_bank) {
      return ExpandResult::failed(
          "gfx1250 K=128 WMMA split cannot prove VGPR-MSB state for its temporary mode");
    }
    if ((src0_crosses_bank && *src0_bank == 3) || (src1_crosses_bank && *src1_bank == 3)) {
      return ExpandResult::failed(
          "gfx1250 K=128 WMMA bank-crossing input exceeds the addressable VGPR range");
    }

    original_mode =
        static_cast<uint8_t>(*src0_bank | (*src1_bank << 2) | (*src2_bank << 4) | (*dst_bank << 6));
    second_mode = static_cast<uint8_t>((*src0_bank + (src0_crosses_bank ? 1u : 0u)) |
                                       ((*src1_bank + (src1_crosses_bank ? 1u : 0u)) << 2) |
                                       (*dst_bank << 4) | (*dst_bank << 6));
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

// The semantic translator binary-searches this table, so entries must stay
// sorted by the full encoding ID and then opcode. VDS encoding IDs include the
// high opcode bits, hence the four consecutive kVdsOpHi* groups below.
inline constexpr std::array<TranslationRule, 25> kGfx1250B0ToA0ExpandRules = {{
    {gfx1250::encoding::kSopp, gfx1250::kSClauseSopp, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_s_clause, nullptr, nullptr},
    {gfx1250::encoding::kVop3p, 0x35, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_wmma_scale_src2, nullptr, nullptr},
    {gfx1250::encoding::kVop3p, 0x3a, RuleAction::Expand, 0, 0, nullptr,
     expand_gfx1250_wmma_scale16, nullptr, nullptr},
    {gfx1250::encoding::kVop3p, gfx1250::kVWmmaI3216x16x64Iu8Vop3p, RuleAction::Expand, 0, 0,
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
    {gfx1250::encoding::kVimage, gfx1250::kTensorLoadToLdsVimage, RuleAction::Expand, 0, 0,
     nullptr, expand_gfx1250_tensor_load_to_lds, nullptr, nullptr},
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
  // CLAMP=0 is the common E4M3 operation on both steppings. Keep CLAMP=1
  // fail-closed until the E5M3 software lowering is implemented.
  const std::string_view mnemonic = inst.mnemonic();
  const bool fp8_clamp_family = mnemonic == "v_cvt_pk_fp8_f32" || mnemonic == "v_cvt_sr_fp8_f32" ||
                                mnemonic.starts_with("v_cvt_f32_fp8");
  if (fp8_clamp_family && !requires_fp8_clamp_emulation(inst))
    return nullptr;

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
