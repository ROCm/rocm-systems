// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/binary_translator.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/code_placement.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/encoding_gfx1250_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_gfx1250_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/dbt/lds_virtualization.h"
#include "rocjitsu/code/dbt/semantic_translator.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/sidecar_metadata.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/vgpr_msb.h"

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

constexpr uint32_t kConservativeLoweringMinimumVgprs = 128;
constexpr uint32_t kGfx1250Rdna4SemanticMinimumVgprs = 128;
// The 16-bit K32 WMMA splits search for scratch at v208 so the worst
// non-accumulating form fits in v208..v224. RDNA4 wave32 descriptors granulate
// VGPRs by 8, so those scopes need a 232-VGPR floor to make v224 addressable.
constexpr uint32_t kGfx1250K32WmmaScratchMinimumVgprs = 232;
constexpr uint32_t kMaxInPlaceMetadataVgprs = 127;
constexpr uint16_t kGfx1250VopdEncodingId = 0x032u;
constexpr uint16_t kGfx1250Vopd3EncodingId = 0x0CFu;
constexpr uint16_t kGfx1250Vop3pEncodingId = 0x198u;
constexpr uint16_t kGfx1250Vop3p1EncodingId = 0x199u;
constexpr uint16_t kGfx1250Vop2AddNcU64EncodingId0 = 0x0A0u;
constexpr uint16_t kGfx1250Vop2AddNcU64EncodingId3 = 0x0A3u;
constexpr uint16_t kGfx1250Vop2SubNcU64EncodingId0 = 0x0A4u;
constexpr uint16_t kGfx1250Vop2SubNcU64EncodingId3 = 0x0A7u;
constexpr uint16_t kGfx1250Vop2MulU64EncodingId0 = 0x0A8u;
constexpr uint16_t kGfx1250Vop2MulU64EncodingId3 = 0x0ABu;
constexpr uint16_t kGfx1250VAddNcU64E32Opcode = 40u;
constexpr uint16_t kGfx1250VSubNcU64E32Opcode = 41u;
constexpr uint16_t kGfx1250VMulU64E32Opcode = 42u;
constexpr uint8_t kGfx1250VAddF16E32Opcode = 50u;
constexpr uint32_t kVop3Encoding = 0x35u;
constexpr uint16_t kGfx1250VPkFmaF32Vop3pOpcode = 31u;
constexpr uint16_t kGfx1250VPkFmaBf16Vop3pOpcode = 17u;
constexpr uint16_t kGfx1250VPkMulF32Vop3pOpcode = 40u;
constexpr uint16_t kGfx1250VPkAddF32Vop3pOpcode = 41u;
constexpr uint16_t kGfx1250VFmaMixloBf16Vop3pOpcode = 0x3Eu;
constexpr uint16_t kGfx1250VCvtPkF16F32Vop3Opcode = 879u;
constexpr uint16_t kGfx1250VAddCoU32Vop3SdstOpcode = 768u;
constexpr uint16_t kGfx1250WmmaF32F8f6f4K128Opcode = 0x33u;
constexpr uint16_t kGfx1250WmmaScaleF32F8f6f4K128Opcode = 0x35u;
constexpr uint16_t kGfx1250WmmaScale16F32F8f6f4K128Opcode = 0x3Au;
constexpr uint16_t kGfx1250WmmaF32F16K32Opcode = 0x60u;
constexpr uint16_t kGfx1250WmmaF16F16K32Opcode = 0x61u;
constexpr uint16_t kGfx1250WmmaF32Bf16K32Opcode = 0x62u;
constexpr uint16_t kGfx1250WmmaBf16Bf16K32Opcode = 0x63u;
constexpr uint16_t kGfx1250WmmaBf16F32Bf16K32Opcode = 0x64u;
constexpr uint16_t kGfx1250SwmmacF32F16K64Opcode = 0x65u;
constexpr uint16_t kGfx1250SwmmacBf16f32K64LastOpcode = 0x69u;
constexpr uint16_t kGfx1250WmmaI32Iu8K64Opcode = 0x72u;
constexpr uint16_t kGfx1250SwmmacF32Fp8K128Opcode = 0x73u;
constexpr uint16_t kGfx1250SwmmacF16Bf8K128LastOpcode = 0x7Au;
constexpr uint16_t kGfx1250SwmmacI32Iu8K128Opcode = 0x7Bu;
constexpr uint16_t kGfx1250WmmaF32Fp8Fp8K128Opcode = 0x80u;
constexpr uint16_t kGfx1250WmmaF32F4M32K128Opcode = 0x88u;
constexpr uint32_t kGfx1250HighBankBaseVgpr = 256u;
constexpr uint32_t kGfx1250VMulU64HighBankScratchCount = 2u;
constexpr uint32_t kGfx1250SemanticPrivateBorrowedVgprCount = 22u;
constexpr uint32_t kGfx1250RedirectPrivateBorrowedVgprCount = 16u;
constexpr uint32_t kGfx1250PrivateBorrowedVgprCount =
    kGfx1250SemanticPrivateBorrowedVgprCount + kGfx1250RedirectPrivateBorrowedVgprCount;
constexpr uint32_t kGfx1250RedirectPrivateBorrowOffset =
    kGfx1250SemanticPrivateBorrowedVgprCount * sizeof(uint32_t);
constexpr uint32_t kGfx1250PrivateBorrowScratchBytes =
    kGfx1250PrivateBorrowedVgprCount * sizeof(uint32_t);
// A remapped high-bank sparse WMMA can touch dst[8], src0[8], src1[16], and
// src2[1] in one semantic expansion. If the shadow window aliases live low
// VGPRs, all touched physical lanes need a private save/restore slot.
constexpr uint32_t kGfx1250HighBankShadowLowSaveVgprCount = 33u;
constexpr uint32_t kGfx1250HighBankShadowLowSaveBytes =
    kGfx1250HighBankShadowLowSaveVgprCount * sizeof(uint32_t);
constexpr uint32_t kRdna4MaxVgprsPerWave = 512u;
constexpr uint32_t kRdna4MaxSgprsPerWave = 106u;
constexpr uint8_t kRdna4NullSgpr = 124u;
constexpr uint32_t kRdna4RawBufferConfigWord = 0x31016000u;
constexpr uint32_t kRdna4RawBufferUnboundedRange = 0xFFFFFFFFu;
constexpr uint8_t kGfx1250SOpAndB32 = 22u;
constexpr uint8_t kGfx1250SOpOrB32 = 24u;
constexpr uint8_t kGfx1250SOpOrB64 = 25u;
constexpr uint16_t kGfx1250ScalarOperandTtmp7 = 115u;
constexpr uint16_t kGfx1250ScalarOperandTtmp9 = 117u;
constexpr uint8_t kSoppWaitLoadcnt = 64u;
constexpr uint8_t kSoppWaitStorecnt = 65u;
constexpr uint8_t kSoppWaitDscnt = 70u;
constexpr uint8_t kSoppWaitKmcnt = 71u;
constexpr uint8_t kGfx1250SoppWaitXcnt = 69u;
constexpr uint64_t kSoppBranchMaxForwardBytes =
    static_cast<uint64_t>(std::numeric_limits<int16_t>::max()) * sizeof(uint32_t);
constexpr uint64_t kKernargPreloadSkipBytes = 256;
// Tensile StreamK kernels can initialize an SRD thousands of instructions
// before the first buffer access. Keep the scan bounded, but large enough to
// see those long descriptor setup regions.
constexpr size_t kGfx1250DescriptorUseScanLimit = 4096;

uint32_t metadata_vgpr_count_for_in_place_patch(std::span<const KdTranslation> translations) {
  uint32_t max_vgprs = 0;
  for (const KdTranslation &translation : translations)
    max_vgprs = std::max(max_vgprs, translation.target_vgpr_allocation_count);
  return std::min(max_vgprs, kMaxInPlaceMetadataVgprs);
}

uint32_t metadata_sgpr_count_for_patch(std::span<const KdTranslation> translations) {
  uint32_t max_sgprs = 0;
  for (const KdTranslation &translation : translations)
    max_sgprs = std::max(max_sgprs, translation.target_sgpr_count);
  return max_sgprs;
}

uint32_t conservative_lowering_minimum_vgprs(rj_code_arch_t guest_arch, rj_code_arch_t host_arch) {
  if (guest_arch == ROCJITSU_CODE_ARCH_GFX1250 && host_arch == ROCJITSU_CODE_ARCH_RDNA4)
    return kGfx1250Rdna4SemanticMinimumVgprs;
  return kConservativeLoweringMinimumVgprs;
}

bool needs_metadata_private_segment_patch(std::span<const KdTranslation> translations) {
  return std::ranges::any_of(translations, [](const KdTranslation &translation) {
    return translation.private_spill_zone_bytes != 0;
  });
}

EncodingTranslateFn select_encoding_translator(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4;
  if (guest == ROCJITSU_CODE_ARCH_GFX1250 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return gfx1250_to_rdna4::translate_encoding_gfx1250_to_rdna4;
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3)
    return cdna4_to_cdna3::translate_encoding_cdna4_to_cdna3;
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3)
    return cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3;
  return nullptr;
}

LegalizationLookupFn select_legalization(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_rdna4, enc_id, opcode);
    };
  }
  if (guest == ROCJITSU_CODE_ARCH_GFX1250 && host == ROCJITSU_CODE_ARCH_RDNA4) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_gfx1250_to_rdna4, enc_id, opcode);
    };
  }
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_cdna3, enc_id, opcode);
    };
  }
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_rdna3, enc_id, opcode);
    };
  }
  return nullptr;
}

[[nodiscard]] bool is_gfx1250_k128_fp8_wmma(const Instruction &inst) {
  return inst.encoding_id() == kGfx1250Vop3p1EncodingId &&
         inst.opcode() == kGfx1250WmmaF32Fp8Fp8K128Opcode;
}

[[nodiscard]] bool is_gfx1250_k32_16bit_wmma(const Instruction &inst) {
  return inst.encoding_id() == kGfx1250Vop3pEncodingId &&
         inst.opcode() >= kGfx1250WmmaF32F16K32Opcode &&
         inst.opcode() <= kGfx1250WmmaBf16F32Bf16K32Opcode;
}

[[nodiscard]] uint32_t
gfx1250_wmma_f8f6f4_matrix_tmp_vgpr_count(const gfx1250::Vop3pMachineInst &src) {
  const uint32_t matrix_a_fmt = src.opsel;
  const uint32_t matrix_b_fmt = (src.pad_14 << 2u) | src.opsel_hi;
  if (matrix_a_fmt > 4u || matrix_b_fmt > 4u)
    return 0;
  if (matrix_a_fmt > 1u || matrix_b_fmt > 1u)
    return 10;
  if (matrix_a_fmt == 0u && matrix_b_fmt == 0u)
    return 4;
  return 0;
}

[[nodiscard]] uint32_t gfx1250_wmma_f8f6f4_tmp_vgpr_count(const Instruction &inst) {
  const auto *raw = inst.raw_encoding();
  if (!raw)
    return 0;

  if (inst.encoding_id() == kGfx1250Vop3pEncodingId &&
      inst.opcode() == kGfx1250WmmaF32F8f6f4K128Opcode &&
      inst.size() >= static_cast<int>(sizeof(gfx1250::Vop3pMachineInst))) {
    gfx1250::Vop3pMachineInst src{};
    std::memcpy(&src, raw, sizeof(src));
    return gfx1250_wmma_f8f6f4_matrix_tmp_vgpr_count(src);
  }

  if (inst.encoding_id() == kGfx1250Vop3p1EncodingId &&
      inst.opcode() == kGfx1250WmmaF32F4M32K128Opcode)
    return 10;

  if (inst.encoding_id() == kGfx1250Vop3pEncodingId &&
      (inst.opcode() == kGfx1250WmmaScaleF32F8f6f4K128Opcode ||
       inst.opcode() == kGfx1250WmmaScale16F32F8f6f4K128Opcode) &&
      inst.size() >= static_cast<int>(2 * sizeof(gfx1250::Vop3pMachineInst))) {
    gfx1250::Vop3pMachineInst src{};
    std::memcpy(&src, raw + 2, sizeof(src));
    if (src.op == kGfx1250WmmaF32F4M32K128Opcode)
      return 10;
    if (src.op != kGfx1250WmmaF32F8f6f4K128Opcode)
      return 0;
    return gfx1250_wmma_f8f6f4_matrix_tmp_vgpr_count(src);
  }

  return 0;
}

[[nodiscard]] bool is_gfx1250_pk_f32_vop3p(const Instruction &inst) {
  if (inst.encoding_id() != kGfx1250Vop3pEncodingId)
    return false;
  const uint16_t op = inst.opcode();
  return op == kGfx1250VPkFmaF32Vop3pOpcode || op == kGfx1250VPkMulF32Vop3pOpcode ||
         op == kGfx1250VPkAddF32Vop3pOpcode;
}

[[nodiscard]] BasicBlock *block_for_offset(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                           uint64_t offset) {
  for (const auto &block : blocks) {
    if (block && block->start_offset() <= offset && offset < block->end_offset())
      return block.get();
  }
  return nullptr;
}

[[nodiscard]] std::optional<uint16_t>
find_long_return_scc_sgpr(const LivenessAnalysis &liveness, const Instruction &inst,
                          std::optional<uint16_t> long_return_sgpr_pair) {
  uint16_t search_start = 0;
  for (;;) {
    auto sgpr = liveness.find_free_sgpr(&inst, search_start);
    if (!sgpr)
      return std::nullopt;
    if (!long_return_sgpr_pair ||
        (*sgpr != *long_return_sgpr_pair && *sgpr != *long_return_sgpr_pair + 1u)) {
      return sgpr;
    }
    if (*sgpr == std::numeric_limits<uint16_t>::max())
      return std::nullopt;
    search_start = static_cast<uint16_t>(*sgpr + 1u);
  }
}

[[nodiscard]] std::vector<uint32_t> raw_words_for_inst(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return {};
  return {raw, raw + inst.size() / sizeof(uint32_t)};
}

[[nodiscard]] bool words_changed(std::span<const uint32_t> before,
                                 std::span<const uint32_t> after) {
  if (before.size() != after.size())
    return true;
  return !std::ranges::equal(before, after);
}

void append_diagnostic(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticSeverity severity,
                       DiagnosticKind kind, std::string message,
                       std::optional<uint64_t> guest_offset = std::nullopt,
                       std::string mnemonic = {}, std::vector<std::string> required_work = {}) {
  diagnostics.push_back({.severity = severity,
                         .kind = kind,
                         .guest_offset = guest_offset,
                         .mnemonic = std::move(mnemonic),
                         .message = std::move(message),
                         .required_work = std::move(required_work)});
}

void append_error(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticKind kind,
                  std::string message, std::optional<uint64_t> guest_offset = std::nullopt,
                  std::string mnemonic = {}, std::vector<std::string> required_work = {}) {
  append_diagnostic(diagnostics, DiagnosticSeverity::Error, kind, std::move(message), guest_offset,
                    std::move(mnemonic), std::move(required_work));
}

void append_diagnostics(std::vector<TranslationDiagnostic> &dst,
                        const std::vector<TranslationDiagnostic> &src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

[[nodiscard]] bool patch_translated_metadata_target_isa(CodeObjectPatcher &patcher,
                                                        rj_code_arch_t guest_arch,
                                                        rj_code_arch_t host_arch) {
  if (guest_arch != ROCJITSU_CODE_ARCH_GFX1250 || host_arch != ROCJITSU_CODE_ARCH_RDNA4)
    return true;

  return patcher.patch_metadata_target_isa("amdgcn-amd-amdhsa--gfx1250",
                                           "amdgcn-amd-amdhsa--gfx1201");
}

[[nodiscard]] uint32_t read_u32(std::span<const uint8_t> bytes, uint64_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

struct StaticPcRelativeSetpcEdge {
  uint64_t source_offset = 0;
  uint64_t getpc_offset = 0;
  uint64_t add_tmp_offset = 0;
  uint64_t target_offset = 0;
};

struct StaticPcRelativeAddress {
  uint16_t sgpr_pair = 0;
  uint64_t getpc_offset = 0;
  uint64_t add_tmp_offset = 0;
  uint64_t target_offset = 0;
  std::optional<int64_t> original_target_offset;
  ExpandedTextPcRelativeFixup::Form form = ExpandedTextPcRelativeFixup::Form::MaterializedSAddCoI32;
};

struct StaticPcRelativeCallEdge {
  uint16_t return_sgpr_pair = 0;
  uint16_t target_sgpr_pair = 0;
  uint64_t call_offset = 0;
  uint64_t return_offset = 0;
};

[[nodiscard]] std::optional<uint64_t> aligned_text_target(int64_t target, uint64_t text_size) {
  if (target < 0 || target % static_cast<int64_t>(sizeof(uint32_t)) != 0)
    return std::nullopt;
  const auto offset = static_cast<uint64_t>(target);
  if (offset >= text_size)
    return std::nullopt;
  return offset;
}

[[nodiscard]] std::vector<StaticPcRelativeSetpcEdge>
static_pc_relative_setpc_edges(std::span<const uint8_t> text) {
  std::vector<StaticPcRelativeSetpcEdge> edges;
  constexpr uint32_t kOpSAddCoU32 = 0;
  constexpr uint32_t kOpSSubCoU32 = 1;
  constexpr uint32_t kOpSAddCoI32 = 2;
  constexpr uint32_t kOpSAddCoCiU32 = 4;
  constexpr uint32_t kOpSSubCoCiU32 = 5;
  constexpr uint32_t kOpSAbsI32 = 21;
  constexpr uint32_t kOpSGetPcB64 = 71;
  constexpr uint32_t kOpSSetPcB64 = 72;

  for (uint64_t offset = 0; offset + 6 * sizeof(uint32_t) <= text.size();
       offset += sizeof(uint32_t)) {
    const uint32_t getpc = read_u32(text, offset);
    const uint16_t sgpr_pair = static_cast<uint16_t>((getpc >> 16) & 0x7Fu);
    if (sgpr_pair >= 127 || getpc != pack_sop1(kOpSGetPcB64, sgpr_pair, 0))
      continue;

    const uint32_t add_tmp_word = read_u32(text, offset + sizeof(uint32_t));
    const uint16_t tmp = static_cast<uint16_t>((add_tmp_word >> 16) & 0x7Fu);
    if (tmp >= 128 ||
        add_tmp_word != pack_sop2(kOpSAddCoI32, tmp, 255, scalar_positive_inline_u32(4)))
      continue;

    const auto literal = static_cast<int32_t>(read_u32(text, offset + 2 * sizeof(uint32_t)));
    const int64_t signed_delta = static_cast<int64_t>(literal) + 4;

    const uint64_t add_lo_offset = offset + 3 * sizeof(uint32_t);
    if (read_u32(text, add_lo_offset) == pack_sop2(kOpSAddCoU32, sgpr_pair, sgpr_pair, tmp) &&
        read_u32(text, offset + 4 * sizeof(uint32_t)) ==
            pack_sop2(kOpSAddCoCiU32, static_cast<uint16_t>(sgpr_pair + 1u),
                      static_cast<uint16_t>(sgpr_pair + 1u), scalar_positive_inline_u32(0)) &&
        read_u32(text, offset + 5 * sizeof(uint32_t)) == pack_sop1(kOpSSetPcB64, 0, sgpr_pair)) {
      const int64_t target = static_cast<int64_t>(offset + sizeof(uint32_t)) + signed_delta;
      if (auto aligned = aligned_text_target(target, text.size()))
        edges.push_back(
            {offset + 5 * sizeof(uint32_t), offset, offset + sizeof(uint32_t), *aligned});
      continue;
    }

    if (offset + 13 * sizeof(uint32_t) <= text.size() &&
        read_u32(text, add_lo_offset) ==
            build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_GFX1250) &&
        read_u32(text, offset + 4 * sizeof(uint32_t)) ==
            pack_sopc(3, tmp, scalar_positive_inline_u32(0)) &&
        read_u32(text, offset + 5 * sizeof(uint32_t)) == 0xBFA20004u &&
        read_u32(text, offset + 6 * sizeof(uint32_t)) == pack_sop1(kOpSAbsI32, tmp, tmp) &&
        read_u32(text, offset + 7 * sizeof(uint32_t)) ==
            pack_sop2(kOpSSubCoU32, sgpr_pair, sgpr_pair, tmp) &&
        read_u32(text, offset + 8 * sizeof(uint32_t)) ==
            pack_sop2(kOpSSubCoCiU32, static_cast<uint16_t>(sgpr_pair + 1u),
                      static_cast<uint16_t>(sgpr_pair + 1u), scalar_positive_inline_u32(0)) &&
        read_u32(text, offset + 9 * sizeof(uint32_t)) == pack_sop1(kOpSSetPcB64, 0, sgpr_pair) &&
        read_u32(text, offset + 10 * sizeof(uint32_t)) ==
            pack_sop2(kOpSAddCoU32, sgpr_pair, sgpr_pair, tmp) &&
        read_u32(text, offset + 11 * sizeof(uint32_t)) ==
            pack_sop2(kOpSAddCoCiU32, static_cast<uint16_t>(sgpr_pair + 1u),
                      static_cast<uint16_t>(sgpr_pair + 1u), scalar_positive_inline_u32(0)) &&
        read_u32(text, offset + 12 * sizeof(uint32_t)) == pack_sop1(kOpSSetPcB64, 0, sgpr_pair)) {
      const int64_t target = static_cast<int64_t>(offset + sizeof(uint32_t)) + signed_delta;
      if (auto aligned = aligned_text_target(target, text.size())) {
        edges.push_back(
            {offset + 9 * sizeof(uint32_t), offset, offset + sizeof(uint32_t), *aligned});
        edges.push_back(
            {offset + 12 * sizeof(uint32_t), offset, offset + sizeof(uint32_t), *aligned});
      }
      continue;
    }

    if (offset + 8 * sizeof(uint32_t) > text.size())
      continue;
    if (read_u32(text, add_lo_offset) !=
            build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_GFX1250) ||
        read_u32(text, offset + 4 * sizeof(uint32_t)) != pack_sop1(kOpSAbsI32, tmp, tmp) ||
        read_u32(text, offset + 5 * sizeof(uint32_t)) !=
            pack_sop2(kOpSSubCoU32, sgpr_pair, sgpr_pair, tmp) ||
        read_u32(text, offset + 6 * sizeof(uint32_t)) !=
            pack_sop2(kOpSSubCoCiU32, static_cast<uint16_t>(sgpr_pair + 1u),
                      static_cast<uint16_t>(sgpr_pair + 1u), scalar_positive_inline_u32(0)) ||
        read_u32(text, offset + 7 * sizeof(uint32_t)) != pack_sop1(kOpSSetPcB64, 0, sgpr_pair))
      continue;

    const int64_t magnitude = signed_delta < 0 ? -signed_delta : signed_delta;
    const int64_t target = static_cast<int64_t>(offset + sizeof(uint32_t)) - magnitude;
    if (auto aligned = aligned_text_target(target, text.size()))
      edges.push_back({offset + 7 * sizeof(uint32_t), offset, offset + sizeof(uint32_t), *aligned});
  }
  return edges;
}

[[nodiscard]] std::vector<StaticPcRelativeAddress>
static_pc_relative_address_edges(std::span<const uint8_t> text) {
  std::vector<StaticPcRelativeAddress> edges;
  constexpr uint32_t kOpSAddCoU32 = 0;
  constexpr uint32_t kOpSAddCoI32 = 2;
  constexpr uint32_t kOpSAddCoCiU32 = 4;
  constexpr uint32_t kOpSAddNcU64 = 83;
  constexpr uint32_t kOpSGetPcB64 = 71;
  constexpr uint32_t kOpSSetPcB64 = 72;

  for (uint64_t offset = 0; offset + 6 * sizeof(uint32_t) <= text.size();
       offset += sizeof(uint32_t)) {
    const uint32_t getpc = read_u32(text, offset);
    const uint16_t sgpr_pair = static_cast<uint16_t>((getpc >> 16) & 0x7Fu);
    if (sgpr_pair >= 127 || getpc != pack_sop1(kOpSGetPcB64, sgpr_pair, 0))
      continue;

    const uint64_t add_tmp_offset = offset + sizeof(uint32_t);
    const uint32_t add_tmp_word = read_u32(text, add_tmp_offset);
    if (add_tmp_word == pack_sop2(kOpSAddNcU64, sgpr_pair, sgpr_pair, 254)) {
      const uint64_t literal_bits =
          static_cast<uint64_t>(read_u32(text, offset + 2 * sizeof(uint32_t))) |
          (static_cast<uint64_t>(read_u32(text, offset + 3 * sizeof(uint32_t))) << 32u);
      const int64_t literal = std::bit_cast<int64_t>(literal_bits);
      const auto literal_lo = static_cast<int32_t>(literal_bits & 0xFFFF'FFFFu);
      if (static_cast<int64_t>(literal_lo) == literal) {
        const int64_t target = static_cast<int64_t>(offset + sizeof(uint32_t)) + literal;
        if (auto aligned = aligned_text_target(target, text.size())) {
          edges.push_back({sgpr_pair, offset, add_tmp_offset, *aligned, std::nullopt,
                           ExpandedTextPcRelativeFixup::Form::DirectSAddNcU64Literal64});
        } else {
          edges.push_back({sgpr_pair, offset, add_tmp_offset, 0, target,
                           ExpandedTextPcRelativeFixup::Form::DirectSAddNcU64Literal64});
        }
      }
      continue;
    }

    const uint16_t tmp = static_cast<uint16_t>((add_tmp_word >> 16) & 0x7Fu);
    if (tmp >= 128 ||
        add_tmp_word != pack_sop2(kOpSAddCoI32, tmp, 255, scalar_positive_inline_u32(4)))
      continue;

    const auto literal = static_cast<int32_t>(read_u32(text, offset + 2 * sizeof(uint32_t)));
    uint64_t add_lo_offset = offset + 3 * sizeof(uint32_t);
    if (read_u32(text, add_lo_offset) ==
        build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_GFX1250)) {
      add_lo_offset += sizeof(uint32_t);
    }
    if (add_lo_offset + 3 * sizeof(uint32_t) > text.size())
      continue;
    if (read_u32(text, add_lo_offset) != pack_sop2(kOpSAddCoU32, sgpr_pair, sgpr_pair, tmp) ||
        read_u32(text, add_lo_offset + sizeof(uint32_t)) !=
            pack_sop2(kOpSAddCoCiU32, static_cast<uint16_t>(sgpr_pair + 1u),
                      static_cast<uint16_t>(sgpr_pair + 1u), scalar_positive_inline_u32(0))) {
      continue;
    }
    if (read_u32(text, add_lo_offset + 2 * sizeof(uint32_t)) ==
        pack_sop1(kOpSSetPcB64, 0, sgpr_pair)) {
      continue;
    }

    const int64_t target =
        static_cast<int64_t>(offset + 2 * sizeof(uint32_t)) + static_cast<int64_t>(literal);
    if (auto aligned = aligned_text_target(target, text.size()))
      edges.push_back({sgpr_pair, offset, add_tmp_offset, *aligned, std::nullopt});
  }
  return edges;
}

[[nodiscard]] std::vector<StaticPcRelativeCallEdge>
static_pc_relative_call_edges(std::span<const uint8_t> text) {
  std::vector<StaticPcRelativeCallEdge> edges;
  constexpr uint32_t kOpSSwapPcI64 = 73;

  for (uint64_t offset = 0; offset + sizeof(uint32_t) <= text.size(); offset += sizeof(uint32_t)) {
    const uint32_t word = read_u32(text, offset);
    const uint16_t return_pair = static_cast<uint16_t>((word >> 16) & 0x7Fu);
    const uint16_t target_pair = static_cast<uint16_t>(word & 0xFFu);
    if (return_pair >= 127 || target_pair >= 127 ||
        word != pack_sop1(kOpSSwapPcI64, return_pair, target_pair))
      continue;

    const uint64_t return_offset = offset + sizeof(uint32_t);
    if (return_offset >= text.size())
      continue;

    edges.push_back({return_pair, target_pair, offset, return_offset});
  }

  return edges;
}

[[nodiscard]] std::optional<uint32_t> read_trailing_literal_u32(std::span<const uint8_t> text,
                                                                uint64_t offset,
                                                                uint32_t literal_byte_offset) {
  if (offset + literal_byte_offset + sizeof(uint32_t) > text.size())
    return std::nullopt;
  return read_u32(text, offset + literal_byte_offset);
}

[[nodiscard]] bool has_unimplemented_expand_gap(const std::vector<std::string> *warnings) {
  if (warnings == nullptr)
    return false;
  return std::ranges::any_of(*warnings, [](const std::string &warning) {
    return warning.rfind("EXPAND not yet implemented for ", 0) == 0;
  });
}

[[nodiscard]] bool has_hardware_pending_semantic_lowering(std::string_view mnemonic) {
  (void)mnemonic;
  return false;
}

[[nodiscard]] constexpr std::pair<uint32_t, uint32_t>
build_rdna4_vop3(uint16_t op, uint8_t vdst, uint16_t src0, uint16_t src1, uint16_t src2 = 0) {
  const uint32_t w0 = (vdst & 0xFFu) | ((op & 0x3FFu) << 16) | (kVop3Encoding << 26);
  const uint32_t w1 = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18);
  return {w0, w1};
}

[[nodiscard]] constexpr std::pair<uint32_t, uint32_t>
build_rdna4_vop3_sdst(uint16_t op, uint8_t vdst, uint8_t sdst, uint16_t src0, uint16_t src1,
                      uint16_t src2 = 0) {
  const uint32_t w0 =
      (vdst & 0xFFu) | ((sdst & 0x7Fu) << 8) | ((op & 0x3FFu) << 16) | (kVop3Encoding << 26);
  const uint32_t w1 = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18);
  return {w0, w1};
}

[[nodiscard]] constexpr uint32_t build_rdna4_vop1(uint8_t op, uint8_t vdst, uint16_t src0) {
  return (src0 & 0x1FFu) | ((op & 0x7Fu) << 9) | ((vdst & 0xFFu) << 17) | (0x3Fu << 25);
}

[[nodiscard]] constexpr uint32_t build_rdna4_vop2(uint8_t op, uint8_t vdst, uint16_t src0,
                                                  uint8_t vsrc1) {
  return (src0 & 0x1FFu) | ((vsrc1 & 0xFFu) << 9) | ((vdst & 0xFFu) << 17) | ((op & 0x3Fu) << 25);
}

[[nodiscard]] constexpr uint32_t build_rdna4_vopc(uint8_t op, uint16_t src0, uint8_t vsrc1) {
  return (src0 & 0x1FFu) | ((vsrc1 & 0xFFu) << 9) | ((op & 0xFFu) << 17) | (0x3Eu << 25);
}

struct Gfx1250Sop1MovB32 {
  uint8_t sdst = 0;
  uint16_t ssrc0 = 0;
  std::optional<uint32_t> literal32;
  std::optional<uint64_t> literal64;
};

struct Gfx1250Sop1MovB64 {
  uint8_t sdst = 0;
  uint16_t ssrc0 = 0;
  std::optional<uint32_t> literal32;
  std::optional<uint64_t> literal64;
};

struct Gfx1250SopkMovkI32 {
  uint8_t sdst = 0;
  uint16_t simm16 = 0;
};

struct Gfx1250Sop2Literal32 {
  uint8_t sdst = 0;
  uint16_t non_literal_src = 0;
  uint32_t literal = 0;
};

struct Gfx1250Sop2Literal64 {
  uint8_t sdst = 0;
  uint16_t non_literal_src = 0;
  uint64_t literal = 0;
};

struct Gfx1250Sop2 {
  uint8_t sdst = 0;
  uint16_t ssrc0 = 0;
  uint16_t ssrc1 = 0;
};

[[nodiscard]] std::optional<Gfx1250Sop1MovB32> decode_gfx1250_s_mov_b32(const Instruction &inst) {
  if (inst.encoding_id() != kEnc_SOP1 || inst.opcode() != 0)
    return std::nullopt;
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return std::nullopt;

  const auto src = std::bit_cast<gfx1250::Sop1MachineInst>(raw[0]);
  if (src.op != 0)
    return std::nullopt;
  Gfx1250Sop1MovB32 decoded{};
  decoded.sdst = static_cast<uint8_t>(src.sdst);
  decoded.ssrc0 = src.ssrc0;
  if (src.ssrc0 == 255u) {
    if (inst.size() != static_cast<int>(2 * sizeof(uint32_t)))
      return std::nullopt;
    const Operand *operand = inst.src_operand(0);
    if (!operand)
      return std::nullopt;
    decoded.literal32 = static_cast<uint32_t>(operand->encoding_value());
  } else if (src.ssrc0 == 254u) {
    if (inst.size() != static_cast<int>(3 * sizeof(uint32_t)))
      return std::nullopt;
    const Operand *operand = inst.src_operand(0);
    if (!operand)
      return std::nullopt;
    decoded.literal64 = operand->literal64_value();
    if (!decoded.literal64)
      return std::nullopt;
  } else if (inst.size() != static_cast<int>(sizeof(uint32_t))) {
    return std::nullopt;
  }
  return decoded;
}

[[nodiscard]] std::optional<Gfx1250Sop1MovB64> decode_gfx1250_s_mov_b64(const Instruction &inst) {
  if (inst.encoding_id() != kEnc_SOP1 || inst.opcode() != 1)
    return std::nullopt;
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return std::nullopt;

  const auto src = std::bit_cast<gfx1250::Sop1MachineInst>(raw[0]);
  if (src.op != 1)
    return std::nullopt;
  Gfx1250Sop1MovB64 decoded{};
  decoded.sdst = static_cast<uint8_t>(src.sdst);
  decoded.ssrc0 = src.ssrc0;
  if (src.ssrc0 == 255u) {
    if (inst.size() != static_cast<int>(2 * sizeof(uint32_t)))
      return std::nullopt;
    const Operand *operand = inst.src_operand(0);
    if (!operand)
      return std::nullopt;
    decoded.literal32 = static_cast<uint32_t>(operand->encoding_value());
  } else if (src.ssrc0 == 254u) {
    if (inst.size() != static_cast<int>(3 * sizeof(uint32_t)))
      return std::nullopt;
    const Operand *operand = inst.src_operand(0);
    if (!operand)
      return std::nullopt;
    decoded.literal64 = operand->literal64_value();
    if (!decoded.literal64)
      return std::nullopt;
  } else if (inst.size() != static_cast<int>(sizeof(uint32_t))) {
    return std::nullopt;
  }
  return decoded;
}

[[nodiscard]] std::optional<Gfx1250SopkMovkI32> decode_gfx1250_s_movk_i32(const Instruction &inst) {
  if ((inst.encoding_id() & 0x1E0u) != kEnc_SOPK || inst.opcode() != 0 ||
      inst.size() != static_cast<int>(sizeof(uint32_t)))
    return std::nullopt;
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return std::nullopt;

  const auto src = std::bit_cast<gfx1250::SopkMachineInst>(raw[0]);
  if (src.op != 0)
    return std::nullopt;
  return Gfx1250SopkMovkI32{static_cast<uint8_t>(src.sdst), static_cast<uint16_t>(src.simm16)};
}

[[nodiscard]] bool is_sop2_encoding(const Instruction &inst) {
  return inst.encoding_id() >= kEnc_SOP2 && inst.encoding_id() < kEnc_SOPK;
}

bool remap_gfx1250_ttmp_grid_reads_to_sgpr(const Instruction &inst, int16_t rdna4_grid_x_sgpr,
                                           int16_t rdna4_grid_yz_sgpr,
                                           std::vector<uint32_t> &words) {
  if (words.empty())
    return false;

  auto replacement_sgpr = [&](uint16_t source_operand) -> std::optional<uint16_t> {
    if (source_operand == kGfx1250ScalarOperandTtmp9 && rdna4_grid_x_sgpr >= 0)
      return static_cast<uint16_t>(rdna4_grid_x_sgpr);
    if (source_operand == kGfx1250ScalarOperandTtmp7 && rdna4_grid_yz_sgpr >= 0)
      return static_cast<uint16_t>(rdna4_grid_yz_sgpr);
    return std::nullopt;
  };

  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return false;

  bool changed = false;
  auto replace_bits = [&](uint32_t mask, uint32_t shift, uint16_t sgpr) {
    words[0] = (words[0] & ~mask) | ((static_cast<uint32_t>(sgpr) << shift) & mask);
    changed = true;
  };

  if (inst.encoding_id() == kEnc_SOP1) {
    if (((words[0] >> 23) & 0x1FFu) != kEnc_SOP1)
      return false;
    const auto src = std::bit_cast<gfx1250::Sop1MachineInst>(raw[0]);
    if (const auto sgpr = replacement_sgpr(src.ssrc0))
      replace_bits(0xFFu, 0, *sgpr);
    return changed;
  }

  if (inst.encoding_id() == kEnc_SOPC) {
    if (((words[0] >> 23) & 0x1FFu) != kEnc_SOPC)
      return false;
    const auto src = std::bit_cast<gfx1250::SopcMachineInst>(raw[0]);
    if (const auto sgpr = replacement_sgpr(src.ssrc0))
      replace_bits(0xFFu, 0, *sgpr);
    if (const auto sgpr = replacement_sgpr(src.ssrc1))
      replace_bits(0xFF00u, 8, *sgpr);
    return changed;
  }

  if (is_sop2_encoding(inst)) {
    if (((words[0] >> 30) & 0x3u) != kSop2EncodingPrefix)
      return false;
    const auto src = std::bit_cast<gfx1250::Sop2MachineInst>(raw[0]);
    if (const auto sgpr = replacement_sgpr(src.ssrc0))
      replace_bits(0xFFu, 0, *sgpr);
    if (const auto sgpr = replacement_sgpr(src.ssrc1))
      replace_bits(0xFF00u, 8, *sgpr);
  }
  return changed;
}

[[nodiscard]] std::optional<Gfx1250Sop2Literal32>
decode_gfx1250_sop2_literal32(const Instruction &inst, uint8_t op) {
  if (!is_sop2_encoding(inst) || inst.opcode() != op ||
      inst.size() != static_cast<int>(2 * sizeof(uint32_t)))
    return std::nullopt;
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return std::nullopt;

  const auto src = std::bit_cast<gfx1250::Sop2MachineInst>(raw[0]);
  uint16_t non_literal_src = 0;
  uint8_t literal_operand = 0;
  if (src.ssrc0 == 255u && src.ssrc1 != 255u) {
    non_literal_src = src.ssrc1;
    literal_operand = 0;
  } else if (src.ssrc1 == 255u && src.ssrc0 != 255u) {
    non_literal_src = src.ssrc0;
    literal_operand = 1;
  } else {
    return std::nullopt;
  }

  const Operand *operand = inst.src_operand(literal_operand);
  if (!operand)
    return std::nullopt;
  return Gfx1250Sop2Literal32{static_cast<uint8_t>(src.sdst), non_literal_src,
                              static_cast<uint32_t>(operand->encoding_value())};
}

[[nodiscard]] std::optional<Gfx1250Sop2Literal64>
decode_gfx1250_sop2_literal64(const Instruction &inst, uint8_t op) {
  if (!is_sop2_encoding(inst) || inst.opcode() != op ||
      inst.size() != static_cast<int>(3 * sizeof(uint32_t)))
    return std::nullopt;
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return std::nullopt;

  const auto src = std::bit_cast<gfx1250::Sop2MachineInst>(raw[0]);
  uint16_t non_literal_src = 0;
  uint8_t literal_operand = 0;
  if (src.ssrc0 == 254u && src.ssrc1 != 254u) {
    non_literal_src = src.ssrc1;
    literal_operand = 0;
  } else if (src.ssrc1 == 254u && src.ssrc0 != 254u) {
    non_literal_src = src.ssrc0;
    literal_operand = 1;
  } else {
    return std::nullopt;
  }

  const Operand *operand = inst.src_operand(literal_operand);
  if (!operand)
    return std::nullopt;
  const auto literal = operand->literal64_value();
  if (!literal)
    return std::nullopt;
  return Gfx1250Sop2Literal64{static_cast<uint8_t>(src.sdst), non_literal_src, *literal};
}

[[nodiscard]] std::optional<Gfx1250Sop2> decode_gfx1250_sop2(const Instruction &inst, uint8_t op) {
  if (!is_sop2_encoding(inst) || inst.opcode() != op ||
      inst.size() != static_cast<int>(sizeof(uint32_t)))
    return std::nullopt;
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return std::nullopt;

  const auto src = std::bit_cast<gfx1250::Sop2MachineInst>(raw[0]);
  return Gfx1250Sop2{static_cast<uint8_t>(src.sdst), static_cast<uint16_t>(src.ssrc0),
                     static_cast<uint16_t>(src.ssrc1)};
}

[[nodiscard]] std::optional<uint8_t> raw_buffer_resource_base_for_descriptor_word2(uint8_t sgpr) {
  if (sgpr >= 2 && (sgpr % 4u) == 2u)
    return static_cast<uint8_t>(sgpr - 2u);
  return std::nullopt;
}

[[nodiscard]] std::optional<uint8_t> raw_buffer_resource_base_for_descriptor_config(uint8_t sgpr) {
  if (sgpr >= 3 && (sgpr % 4u) == 3u)
    return static_cast<uint8_t>(sgpr - 3u);
  return std::nullopt;
}

[[nodiscard]] std::optional<uint32_t>
rdna4_range_from_gfx1250_descriptor_word2_units(uint64_t units) {
  if (units == 0)
    return kRdna4RawBufferUnboundedRange;
  if (units >= std::numeric_limits<uint32_t>::max() / 128u)
    return kRdna4RawBufferUnboundedRange;
  const uint64_t inclusive_last_block_bytes = (units + 1u) * 128u;
  if (units <= 16u)
    return static_cast<uint32_t>(std::max(units * 256u, inclusive_last_block_bytes));
  return static_cast<uint32_t>(inclusive_last_block_bytes);
}

[[nodiscard]] std::optional<uint32_t> rdna4_range_from_gfx1250_descriptor_word2_src(uint16_t src) {
  if (src < scalar_positive_inline_u32(0) || src > scalar_positive_inline_u32(64))
    return std::nullopt;

  return rdna4_range_from_gfx1250_descriptor_word2_units(src - scalar_positive_inline_u32(0));
}

[[nodiscard]] std::optional<uint32_t>
rdna4_range_from_gfx1250_descriptor_word2_mov(uint16_t src, std::optional<uint32_t> literal32,
                                              std::optional<uint64_t> literal64) {
  if (literal64)
    return rdna4_range_from_gfx1250_descriptor_word2_units(*literal64);
  if (literal32)
    return rdna4_range_from_gfx1250_descriptor_word2_units(*literal32);
  return rdna4_range_from_gfx1250_descriptor_word2_src(src);
}

[[nodiscard]] bool
gfx1250_descriptor_word2_mov_is_statically_nonzero(uint16_t src, std::optional<uint32_t> literal32,
                                                   std::optional<uint64_t> literal64) {
  if (literal64)
    return *literal64 != 0;
  if (literal32)
    return *literal32 != 0;
  if (src >= scalar_positive_inline_u32(0) && src <= scalar_positive_inline_u32(64))
    return src != scalar_positive_inline_u32(0);
  return false;
}

[[nodiscard]] bool is_raw_buffer_descriptor_word2_sgpr_src(uint16_t src) {
  if (src >= 128u)
    return false;
  return raw_buffer_resource_base_for_descriptor_word2(static_cast<uint8_t>(src)).has_value();
}

[[nodiscard]] bool defines_sgpr(const Instruction &inst, uint8_t sgpr) {
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const Operand *operand = inst.dst_operand(i);
    if (!operand)
      continue;
    const auto reg = operand->to_register_ref();
    if (!reg || reg->cls != RegClass::SGPR)
      continue;
    if (reg->index <= sgpr && sgpr < reg->index + reg->width)
      return true;
  }

  RegisterSet implicit_defs;
  inst.implicit_defs(implicit_defs);
  return implicit_defs.contains(RegisterRef{RegClass::SGPR, sgpr, 1});
}

[[nodiscard]] bool uses_sgpr(const Instruction &inst, uint8_t sgpr) {
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const Operand *operand = inst.src_operand(i);
    if (!operand)
      continue;
    const auto reg = operand->to_register_ref();
    if (!reg || reg->cls != RegClass::SGPR)
      continue;
    if (reg->index <= sgpr && sgpr < reg->index + reg->width)
      return true;
  }

  RegisterSet implicit_uses;
  inst.implicit_uses(implicit_uses);
  if (implicit_uses.contains(RegisterRef{RegClass::SGPR, sgpr, 1}))
    return true;

  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return false;
  if (inst.encoding_id() == kEnc_SOP1) {
    const auto src = std::bit_cast<gfx1250::Sop1MachineInst>(raw[0]);
    if (src.ssrc0 >= 128u)
      return false;
    const Operand *operand = inst.src_operand(0);
    const auto width = static_cast<uint8_t>(std::max(1, operand ? operand->size_bits() / 32 : 1));
    return src.ssrc0 <= sgpr && sgpr < src.ssrc0 + width;
  }
  if (is_sop2_encoding(inst)) {
    const auto src = std::bit_cast<gfx1250::Sop2MachineInst>(raw[0]);
    const auto overlaps_scalar_src = [&inst, sgpr](uint16_t ssrc, uint8_t operand_index) {
      if (ssrc >= 128u)
        return false;
      const Operand *operand = inst.src_operand(operand_index);
      const auto width = static_cast<uint8_t>(std::max(1, operand ? operand->size_bits() / 32 : 1));
      return ssrc <= sgpr && sgpr < ssrc + width;
    };
    return overlaps_scalar_src(src.ssrc0, 0) || overlaps_scalar_src(src.ssrc1, 1);
  }
  return false;
}

[[nodiscard]] std::optional<uint32_t> vbuffer_access_size_bytes(uint32_t op) {
  switch (op) {
  case 16: // buffer_load_u8
  case 17: // buffer_load_i8
    return 1;
  case 18: // buffer_load_u16
  case 19: // buffer_load_i16
  case 25: // buffer_store_b16
    return 2;
  case 20: // buffer_load_b32
  case 26: // buffer_store_b32
    return 4;
  case 21: // buffer_load_b64
  case 27: // buffer_store_b64
    return 8;
  case 22: // buffer_load_b96
  case 28: // buffer_store_b96
    return 12;
  case 23: // buffer_load_b128
  case 29: // buffer_store_b128
    return 16;
  case 24: // buffer_store_b8
    return 1;
  case 30: // buffer_load_d16_u8
  case 31: // buffer_load_d16_i8
  case 33: // buffer_load_d16_hi_u8
  case 34: // buffer_load_d16_hi_i8
  case 36: // buffer_store_d16_hi_b8
    return 1;
  case 32: // buffer_load_d16_b16
  case 35: // buffer_load_d16_hi_b16
  case 37: // buffer_store_d16_hi_b16
    return 2;
  default:
    return std::nullopt;
  }
}

[[nodiscard]] std::optional<uint32_t> gfx1250_vbuffer_access_size_bytes(const Instruction &inst,
                                                                        uint8_t resource_base) {
  if (!inst.mnemonic().starts_with("buffer_"))
    return std::nullopt;
  if ((inst.encoding_id() & 0x1FCu) != kEnc_VBUFFER ||
      inst.size() < static_cast<int>(3 * sizeof(uint32_t)))
    return std::nullopt;
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return std::nullopt;
  if (((raw[0] >> 24u) & 0xFFu) == 0xEEu)
    return std::nullopt;

  const auto fields = gfx1250_to_rdna4::decode_vbuffer_gfx1250(raw[0], raw[1], raw[2]);
  if (fields.rsrc != resource_base)
    return std::nullopt;
  return vbuffer_access_size_bytes(fields.op);
}

enum class DescriptorUseScanResult : uint8_t {
  FoundUse,
  UnsafeUse,
  Defined,
  ReachedEnd,
  Terminated,
};

[[nodiscard]] bool defines_any_sgpr(const Instruction &inst, std::span<const uint8_t> sgprs) {
  for (const uint8_t sgpr : sgprs) {
    if (defines_sgpr(inst, sgpr))
      return true;
  }
  return false;
}

[[nodiscard]] bool uses_any_sgpr(const Instruction &inst, std::span<const uint8_t> sgprs) {
  for (const uint8_t sgpr : sgprs) {
    if (uses_sgpr(inst, sgpr))
      return true;
  }
  return false;
}

[[nodiscard]] bool gfx1250_mov_src_is_zero(uint16_t src, std::optional<uint32_t> literal32,
                                           std::optional<uint64_t> literal64) {
  if (literal64)
    return *literal64 == 0;
  if (literal32)
    return *literal32 == 0;
  return src == scalar_positive_inline_u32(0);
}

[[nodiscard]] bool nearest_gfx1250_sgpr_def_is_zero_mov(InstructionList::Iterator block_begin,
                                                        InstructionList::Iterator inst_it,
                                                        uint8_t sgpr) {
  for (auto scan_it = inst_it; scan_it != block_begin;) {
    --scan_it;
    const Instruction &prev = *scan_it;
    if (!defines_sgpr(prev, sgpr))
      continue;

    if (const auto mov = decode_gfx1250_s_mov_b32(prev)) {
      return mov->sdst == sgpr &&
             gfx1250_mov_src_is_zero(mov->ssrc0, mov->literal32, mov->literal64);
    }
    if (const auto mov = decode_gfx1250_s_mov_b64(prev)) {
      return mov->sdst <= sgpr && sgpr < mov->sdst + 2u &&
             gfx1250_mov_src_is_zero(mov->ssrc0, mov->literal32, mov->literal64);
    }
    return false;
  }
  return false;
}

[[nodiscard]] bool
zero_sgpr_was_used_as_descriptor_word2_after_def(InstructionList::Iterator block_begin,
                                                 InstructionList::Iterator inst_it, uint8_t sgpr) {
  const auto resource_base = raw_buffer_resource_base_for_descriptor_word2(sgpr);
  if (!resource_base)
    return false;

  for (auto scan_it = inst_it; scan_it != block_begin;) {
    --scan_it;
    const Instruction &prev = *scan_it;
    if (!defines_sgpr(prev, sgpr))
      continue;

    bool zero_def = false;
    if (const auto mov = decode_gfx1250_s_mov_b32(prev)) {
      zero_def =
          mov->sdst == sgpr && gfx1250_mov_src_is_zero(mov->ssrc0, mov->literal32, mov->literal64);
    } else if (const auto mov = decode_gfx1250_s_mov_b64(prev)) {
      zero_def = mov->sdst <= sgpr && sgpr < mov->sdst + 2u &&
                 gfx1250_mov_src_is_zero(mov->ssrc0, mov->literal32, mov->literal64);
    }
    if (!zero_def)
      return false;

    for (auto use_it = scan_it;;) {
      ++use_it;
      if (use_it == inst_it)
        return false;
      if (defines_sgpr(*use_it, sgpr))
        return false;
      if (gfx1250_vbuffer_access_size_bytes(*use_it, *resource_base))
        return true;
    }
  }
  return false;
}

[[nodiscard]] bool
zero_sgpr_was_used_as_descriptor_word2_in_scope(std::span<BasicBlock *const> scope_blocks,
                                                uint64_t inst_offset, uint8_t sgpr) {
  const auto resource_base = raw_buffer_resource_base_for_descriptor_word2(sgpr);
  if (!resource_base)
    return false;

  std::optional<uint64_t> def_offset;
  bool zero_def = false;
  for (BasicBlock *block : scope_blocks) {
    if (!block)
      continue;
    uint64_t offset = block->start_offset();
    for (const Instruction &candidate : block->instructions()) {
      if (offset >= inst_offset)
        break;
      if (defines_sgpr(candidate, sgpr) && (!def_offset || offset > *def_offset)) {
        def_offset = offset;
        zero_def = false;
        if (const auto mov = decode_gfx1250_s_mov_b32(candidate)) {
          zero_def = mov->sdst == sgpr &&
                     gfx1250_mov_src_is_zero(mov->ssrc0, mov->literal32, mov->literal64);
        } else if (const auto mov = decode_gfx1250_s_mov_b64(candidate)) {
          zero_def = mov->sdst <= sgpr && sgpr < mov->sdst + 2u &&
                     gfx1250_mov_src_is_zero(mov->ssrc0, mov->literal32, mov->literal64);
        }
      }
      offset += candidate.size();
    }
  }
  if (!def_offset || !zero_def)
    return false;

  for (BasicBlock *block : scope_blocks) {
    if (!block)
      continue;
    uint64_t offset = block->start_offset();
    for (const Instruction &candidate : block->instructions()) {
      if (offset > *def_offset && offset < inst_offset &&
          gfx1250_vbuffer_access_size_bytes(candidate, *resource_base)) {
        return true;
      }
      offset += candidate.size();
    }
  }
  return false;
}

void rewrite_gfx1250_zero_sgpr_v_mov_sources(std::vector<uint32_t> &words, const Instruction &inst,
                                             InstructionList::Iterator block_begin,
                                             InstructionList::Iterator inst_it) {
  constexpr uint8_t kRdna4VMovB32Op = 1;
  for (uint32_t &word : words) {
    auto vop1 = std::bit_cast<rdna4::Vop1MachineInst>(word);
    if (vop1.encoding != 0x3Fu || vop1.op != kRdna4VMovB32Op || vop1.src0 >= 128u)
      continue;

    const auto sgpr = static_cast<uint8_t>(vop1.src0);
    if (!uses_sgpr(inst, sgpr))
      continue;
    if (!nearest_gfx1250_sgpr_def_is_zero_mov(block_begin, inst_it, sgpr))
      continue;

    vop1.src0 = scalar_positive_inline_u32(0);
    word = std::bit_cast<uint32_t>(vop1);
  }
}

[[nodiscard]] bool
is_descriptor_setup_copy_from_tracked_sgpr(const Instruction &inst,
                                           std::span<const uint8_t> tracked_sgprs);

void rewrite_gfx1250_zero_sgpr_scalar_sources(std::vector<uint32_t> &words, const Instruction &inst,
                                              InstructionList::Iterator block_begin,
                                              InstructionList::Iterator inst_it,
                                              uint64_t inst_offset,
                                              std::span<BasicBlock *const> scope_blocks) {
  if (words.empty())
    return;

  const auto rewrite_descriptor_zero_src = [&](uint32_t src) -> uint32_t {
    if (src >= 128u)
      return src;
    if (!is_raw_buffer_descriptor_word2_sgpr_src(src))
      return src;

    const auto sgpr = static_cast<uint8_t>(src);
    if (!uses_sgpr(inst, sgpr))
      return src;
    if (is_descriptor_setup_copy_from_tracked_sgpr(inst, std::span<const uint8_t>(&sgpr, 1)))
      return src;
    const bool descriptor_zero_promoted =
        zero_sgpr_was_used_as_descriptor_word2_after_def(block_begin, inst_it, sgpr) ||
        zero_sgpr_was_used_as_descriptor_word2_in_scope(scope_blocks, inst_offset, sgpr);
    if (!descriptor_zero_promoted)
      return src;

    return scalar_positive_inline_u32(0);
  };

  if (is_sop2_encoding(inst)) {
    auto sop2 = std::bit_cast<rdna4::Sop2MachineInst>(words[0]);
    sop2.ssrc0 = rewrite_descriptor_zero_src(sop2.ssrc0);
    sop2.ssrc1 = rewrite_descriptor_zero_src(sop2.ssrc1);
    words[0] = std::bit_cast<uint32_t>(sop2);
    return;
  }

  if (inst.encoding_id() == kEnc_SOPC) {
    auto sopc = std::bit_cast<rdna4::SopcMachineInst>(words[0]);
    sopc.ssrc0 = rewrite_descriptor_zero_src(sopc.ssrc0);
    sopc.ssrc1 = rewrite_descriptor_zero_src(sopc.ssrc1);
    words[0] = std::bit_cast<uint32_t>(sopc);
  }
}

[[nodiscard]] bool
is_descriptor_setup_copy_from_tracked_sgpr(const Instruction &inst,
                                           std::span<const uint8_t> tracked_sgprs) {
  const auto mov = decode_gfx1250_s_mov_b32(inst);
  if (!mov || mov->ssrc0 >= 128u)
    return false;

  const auto src = static_cast<uint8_t>(mov->ssrc0);
  const auto src_overlaps_tracked = [src](uint8_t sgpr) { return src <= sgpr && sgpr <= src + 1u; };
  if (!std::ranges::any_of(tracked_sgprs, src_overlaps_tracked))
    return false;

  return raw_buffer_resource_base_for_descriptor_word2(mov->sdst).has_value() ||
         raw_buffer_resource_base_for_descriptor_config(mov->sdst).has_value();
}

[[nodiscard]] bool is_zero_scalar_src(const Instruction &inst, uint16_t src,
                                      uint8_t operand_index) {
  if (src == scalar_positive_inline_u32(0))
    return true;

  const Operand *operand = inst.src_operand(operand_index);
  if (!operand)
    return false;
  if (src == 255u)
    return static_cast<uint32_t>(operand->encoding_value()) == 0;
  if (src == 254u) {
    const auto literal64 = operand->literal64_value();
    return literal64 && *literal64 == 0;
  }
  return false;
}

[[nodiscard]] bool
is_allowed_nonzero_descriptor_zero_compare(const Instruction &inst,
                                           std::span<const uint8_t> zero_compare_safe_sgprs) {
  constexpr uint8_t kSCmpEqU32 = 6;
  constexpr uint8_t kSCmpLgU32 = 7;
  if (inst.encoding_id() != kEnc_SOPC ||
      (inst.opcode() != kSCmpEqU32 && inst.opcode() != kSCmpLgU32))
    return false;

  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return false;
  const auto sopc = std::bit_cast<gfx1250::SopcMachineInst>(raw[0]);
  const auto is_safe_descriptor_word2 = [zero_compare_safe_sgprs](uint16_t src) {
    if (src >= 128u)
      return false;
    const auto sgpr = static_cast<uint8_t>(src);
    if (!raw_buffer_resource_base_for_descriptor_word2(sgpr))
      return false;
    return std::ranges::find(zero_compare_safe_sgprs, sgpr) != zero_compare_safe_sgprs.end();
  };

  return (is_safe_descriptor_word2(sopc.ssrc0) && is_zero_scalar_src(inst, sopc.ssrc1, 1)) ||
         (is_safe_descriptor_word2(sopc.ssrc1) && is_zero_scalar_src(inst, sopc.ssrc0, 0));
}

[[nodiscard]] bool has_no_fallthrough(const Instruction &inst) {
  return (inst.flags() & (BRANCH | INDIRECT_BRANCH | PROGRAM_TERMINATOR)) &&
         !(inst.flags() & COND_BRANCH);
}

[[nodiscard]] bool is_direct_conditional_branch(const Instruction &inst) {
  return (inst.flags() & COND_BRANCH) && inst.branch_offset_bytes().has_value();
}

[[nodiscard]] DescriptorUseScanResult
scan_vbuffer_use_before_any_def(InstructionList::Iterator scan_it, InstructionList::Iterator end,
                                uint8_t resource_base, std::span<const uint8_t> tracked_sgprs,
                                std::span<const uint8_t> zero_compare_safe_sgprs, size_t &scanned) {
  for (; scan_it != end && scanned < kGfx1250DescriptorUseScanLimit; ++scan_it, ++scanned) {
    const Instruction &future = *scan_it;
    if (auto access_size = gfx1250_vbuffer_access_size_bytes(future, resource_base))
      return DescriptorUseScanResult::FoundUse;
    if (uses_any_sgpr(future, tracked_sgprs) && !is_direct_conditional_branch(future) &&
        !is_descriptor_setup_copy_from_tracked_sgpr(future, tracked_sgprs) &&
        !is_allowed_nonzero_descriptor_zero_compare(future, zero_compare_safe_sgprs))
      return DescriptorUseScanResult::UnsafeUse;
    if (defines_any_sgpr(future, tracked_sgprs))
      return DescriptorUseScanResult::Defined;
    if (has_no_fallthrough(future))
      return DescriptorUseScanResult::Terminated;
  }
  return scanned < kGfx1250DescriptorUseScanLimit ? DescriptorUseScanResult::ReachedEnd
                                                  : DescriptorUseScanResult::UnsafeUse;
}

[[nodiscard]] bool successor_vbuffer_uses_resource_before_any_def(
    const BasicBlock *block, uint8_t resource_base, std::span<const uint8_t> tracked_sgprs,
    std::span<const uint8_t> zero_compare_safe_sgprs, std::span<BasicBlock *const> scope_blocks,
    size_t scanned_before_successors) {
  if (!block)
    return false;

  std::vector<std::pair<BasicBlock *, size_t>> worklist;
  worklist.reserve(block->successors().size() + 1);
  const auto enqueue = [&](BasicBlock *successor) {
    if (!successor)
      return;
    if (std::ranges::none_of(worklist,
                             [successor](const auto &entry) { return entry.first == successor; })) {
      worklist.emplace_back(successor, scanned_before_successors);
    }
  };
  for (BasicBlock *successor : block->successors())
    enqueue(successor);

  const Instruction *terminator = block->terminator();
  if ((!terminator || !has_no_fallthrough(*terminator)) && !scope_blocks.empty()) {
    const uint64_t fallthrough_offset = block->end_offset();
    const auto fallthrough = std::ranges::find_if(scope_blocks, [&](const BasicBlock *candidate) {
      return candidate && candidate->start_offset() == fallthrough_offset;
    });
    if (fallthrough != scope_blocks.end())
      enqueue(*fallthrough);
  }

  std::unordered_set<const BasicBlock *> visited;
  bool found_use = false;
  while (!worklist.empty()) {
    const auto [successor, scanned_so_far] = worklist.back();
    worklist.pop_back();
    if (!successor || !visited.insert(successor).second)
      continue;

    size_t scanned = scanned_so_far;
    auto result = scan_vbuffer_use_before_any_def(successor->instructions().begin(),
                                                  successor->instructions().end(), resource_base,
                                                  tracked_sgprs, zero_compare_safe_sgprs, scanned);
    if (result == DescriptorUseScanResult::FoundUse) {
      found_use = true;
      continue;
    }
    if (result == DescriptorUseScanResult::UnsafeUse)
      return false;
    if (result == DescriptorUseScanResult::Defined || result == DescriptorUseScanResult::Terminated)
      continue;

    for (BasicBlock *next : successor->successors())
      worklist.emplace_back(next, scanned);
  }
  return found_use;
}

[[nodiscard]] bool future_vbuffer_uses_resource_before_any_def(
    InstructionList::Iterator inst_it, InstructionList::Iterator end, uint8_t resource_base,
    std::span<const uint8_t> tracked_sgprs, std::span<const uint8_t> zero_compare_safe_sgprs = {},
    const BasicBlock *block = nullptr, std::span<BasicBlock *const> scope_blocks = {}) {
  auto scan_it = inst_it;
  ++scan_it;
  size_t scanned = 0;
  auto result = scan_vbuffer_use_before_any_def(scan_it, end, resource_base, tracked_sgprs,
                                                zero_compare_safe_sgprs, scanned);
  if (result == DescriptorUseScanResult::FoundUse)
    return true;
  if (result == DescriptorUseScanResult::UnsafeUse || result == DescriptorUseScanResult::Defined)
    return false;
  return successor_vbuffer_uses_resource_before_any_def(
      block, resource_base, tracked_sgprs, zero_compare_safe_sgprs, scope_blocks, scanned);
}

[[nodiscard]] bool future_vbuffer_uses_resource_before_def(
    InstructionList::Iterator inst_it, InstructionList::Iterator end, uint8_t resource_base,
    uint8_t tracked_sgpr, const BasicBlock *block = nullptr,
    std::span<BasicBlock *const> scope_blocks = {}, bool allow_zero_compare = false) {
  const std::span<const uint8_t> zero_compare_safe_sgprs =
      allow_zero_compare ? std::span<const uint8_t>(&tracked_sgpr, 1) : std::span<const uint8_t>();
  return future_vbuffer_uses_resource_before_any_def(inst_it, end, resource_base,
                                                     std::span<const uint8_t>(&tracked_sgpr, 1),
                                                     zero_compare_safe_sgprs, block, scope_blocks);
}

[[nodiscard]] bool future_vbuffer_uses_resource_before_scan_limit(
    InstructionList::Iterator inst_it, InstructionList::Iterator end, uint8_t resource_base,
    const BasicBlock *block = nullptr, std::span<BasicBlock *const> scope_blocks = {}) {
  return future_vbuffer_uses_resource_before_any_def(inst_it, end, resource_base, {}, {}, block,
                                                     scope_blocks);
}

[[nodiscard]] bool
scope_order_vbuffer_uses_resource_before_sgpr_reuse(std::span<BasicBlock *const> scope_blocks,
                                                    uint64_t inst_offset, uint8_t resource_base,
                                                    uint8_t tracked_sgpr) {
  std::vector<BasicBlock *> ordered_blocks;
  ordered_blocks.reserve(scope_blocks.size());
  for (BasicBlock *block : scope_blocks) {
    if (block != nullptr && block->end_offset() > inst_offset)
      ordered_blocks.push_back(block);
  }
  std::ranges::sort(ordered_blocks, [](const BasicBlock *lhs, const BasicBlock *rhs) {
    return lhs->start_offset() < rhs->start_offset();
  });

  const std::array<uint8_t, 1> tracked_sgprs = {tracked_sgpr};
  size_t scanned = 0;
  for (BasicBlock *block : ordered_blocks) {
    uint64_t offset = block->start_offset();
    for (const Instruction &candidate : block->instructions()) {
      const uint32_t inst_size = candidate.size();
      if (offset <= inst_offset) {
        offset += inst_size;
        continue;
      }
      if (scanned++ >= kGfx1250DescriptorUseScanLimit)
        return false;

      if (gfx1250_vbuffer_access_size_bytes(candidate, resource_base))
        return true;
      if (uses_sgpr(candidate, tracked_sgpr) &&
          !is_descriptor_setup_copy_from_tracked_sgpr(candidate, tracked_sgprs))
        return false;
      if (defines_sgpr(candidate, tracked_sgpr))
        return false;

      offset += inst_size;
    }
  }

  return false;
}

[[nodiscard]] bool scope_order_vbuffer_uses_resource(std::span<BasicBlock *const> scope_blocks,
                                                     uint64_t inst_offset, uint8_t resource_base) {
  std::vector<BasicBlock *> ordered_blocks;
  ordered_blocks.reserve(scope_blocks.size());
  for (BasicBlock *block : scope_blocks) {
    if (block != nullptr && block->end_offset() > inst_offset)
      ordered_blocks.push_back(block);
  }
  std::ranges::sort(ordered_blocks, [](const BasicBlock *lhs, const BasicBlock *rhs) {
    return lhs->start_offset() < rhs->start_offset();
  });

  size_t scanned = 0;
  for (BasicBlock *block : ordered_blocks) {
    uint64_t offset = block->start_offset();
    for (const Instruction &candidate : block->instructions()) {
      const uint32_t inst_size = candidate.size();
      if (offset <= inst_offset) {
        offset += inst_size;
        continue;
      }
      if (scanned++ >= kGfx1250DescriptorUseScanLimit)
        return false;

      if (gfx1250_vbuffer_access_size_bytes(candidate, resource_base))
        return true;

      offset += inst_size;
    }
  }

  return false;
}

[[nodiscard]] std::optional<uint8_t> raw_buffer_resource_base_for_descriptor_base(uint8_t sgpr) {
  if ((sgpr % 4u) == 0u)
    return sgpr;
  return std::nullopt;
}

[[nodiscard]] std::optional<uint8_t>
raw_buffer_resource_base_for_descriptor_base_high(uint8_t sgpr) {
  if (sgpr >= 1 && (sgpr % 4u) == 1u)
    return static_cast<uint8_t>(sgpr - 1u);
  return std::nullopt;
}

[[nodiscard]] bool is_gfx1250_flat_address_alignment_low_mask(uint32_t mask) {
  return mask == 0xFFFF'E000u || mask == 0xFFFF'C000u;
}

[[nodiscard]] bool is_gfx1250_flat_address_alignment_high_mask(uint32_t mask) {
  return mask == 0x7FFF'FFFFu || mask == 0x1FFF'FFFFu;
}

[[nodiscard]] std::vector<uint32_t>
lower_gfx1250_contextual_s_and_b32_address_mask_high(InstructionList::Iterator block_begin,
                                                     InstructionList::Iterator inst_it,
                                                     rj_code_arch_t host_arch) {
  const auto high = decode_gfx1250_sop2_literal32(*inst_it, kGfx1250SOpAndB32);
  if (!high || !is_gfx1250_flat_address_alignment_high_mask(high->literal) ||
      inst_it == block_begin)
    return {};

  auto prev_it = inst_it;
  --prev_it;
  const auto low = decode_gfx1250_sop2_literal32(*prev_it, kGfx1250SOpAndB32);
  if (!low || !is_gfx1250_flat_address_alignment_low_mask(low->literal) ||
      low->sdst + 1u != high->sdst || low->non_literal_src + 1u != high->non_literal_src)
    return {};

  return {build_s_mov_b32(high->sdst, scalar_positive_inline_u32(0), host_arch),
          build_s_nop(0, host_arch)};
}

[[nodiscard]] bool next_gfx1250_raw_buffer_descriptor_high_pack(
    InstructionList::Iterator block_begin, InstructionList::Iterator inst_it,
    InstructionList::Iterator end, uint8_t high_sgpr, uint8_t word2_sgpr);

[[nodiscard]] std::vector<uint32_t> lower_gfx1250_contextual_raw_buffer_descriptor_high_mask(
    InstructionList::Iterator block_begin, InstructionList::Iterator inst_it,
    InstructionList::Iterator end, uint64_t inst_offset, rj_code_arch_t host_arch,
    const BasicBlock *block = nullptr, std::span<BasicBlock *const> scope_blocks = {}) {
  constexpr uint32_t kGfx1250RawBufferBaseHighMask = 0x01FF'FFFFu;
  const auto high = decode_gfx1250_sop2_literal32(*inst_it, kGfx1250SOpAndB32);
  if (!high || high->literal != kGfx1250RawBufferBaseHighMask)
    return {};

  const auto resource_base = raw_buffer_resource_base_for_descriptor_base_high(high->sdst);
  const uint8_t word2_sgpr = resource_base ? static_cast<uint8_t>(*resource_base + 2u) : 0u;
  if (!resource_base ||
      !(future_vbuffer_uses_resource_before_scan_limit(inst_it, end, *resource_base, block,
                                                       scope_blocks) ||
        next_gfx1250_raw_buffer_descriptor_high_pack(block_begin, inst_it, end, high->sdst,
                                                     word2_sgpr) ||
        scope_order_vbuffer_uses_resource(scope_blocks, inst_offset, *resource_base)))
    return {};

  return {build_s_nop(0, host_arch), build_s_nop(0, host_arch)};
}

[[nodiscard]] bool previous_gfx1250_raw_buffer_descriptor_high_mask(
    InstructionList::Iterator block_begin, InstructionList::Iterator inst_it, uint8_t high_sgpr) {
  constexpr uint32_t kGfx1250RawBufferBaseHighMask = 0x01FF'FFFFu;
  constexpr size_t kScanLimit = 8;
  size_t scanned = 0;
  for (auto scan_it = inst_it; scan_it != block_begin && scanned < kScanLimit;) {
    --scan_it;
    ++scanned;
    if (const auto high = decode_gfx1250_sop2_literal32(*scan_it, kGfx1250SOpAndB32))
      return high->sdst == high_sgpr && high->literal == kGfx1250RawBufferBaseHighMask;
    if (defines_sgpr(*scan_it, high_sgpr))
      return false;
  }
  return false;
}

[[nodiscard]] bool
previous_gfx1250_raw_buffer_descriptor_split_range_temp(InstructionList::Iterator block_begin,
                                                        InstructionList::Iterator inst_it,
                                                        uint8_t word2_sgpr, uint8_t tmp_sgpr) {
  constexpr uint32_t kGfx1250RawBufferSplitRangeLowMask = 0x7Fu;
  constexpr size_t kScanLimit = 12;

  auto lshl_it = inst_it;
  bool found_lshl = false;
  size_t scanned = 0;
  for (; lshl_it != block_begin && scanned < kScanLimit; ++scanned) {
    --lshl_it;
    if (!defines_sgpr(*lshl_it, tmp_sgpr))
      continue;

    const auto lshl = decode_gfx1250_sop2(*lshl_it, sop2_op_lshl_b32(ROCJITSU_CODE_ARCH_GFX1250));
    if (!lshl || lshl->sdst != tmp_sgpr || lshl->ssrc0 != tmp_sgpr ||
        lshl->ssrc1 != scalar_positive_inline_u32(25))
      return false;
    found_lshl = true;
    break;
  }
  if (!found_lshl)
    return false;

  scanned = 0;
  for (auto scan_it = lshl_it; scan_it != block_begin && scanned < kScanLimit; ++scanned) {
    --scan_it;
    if (!defines_sgpr(*scan_it, tmp_sgpr))
      continue;

    const auto and32 = decode_gfx1250_sop2_literal32(*scan_it, kGfx1250SOpAndB32);
    return and32 && and32->sdst == tmp_sgpr && and32->non_literal_src == word2_sgpr &&
           and32->literal == kGfx1250RawBufferSplitRangeLowMask;
  }

  return false;
}

[[nodiscard]] bool is_gfx1250_raw_buffer_descriptor_high_pack(InstructionList::Iterator block_begin,
                                                              InstructionList::Iterator inst_it,
                                                              uint8_t high_sgpr,
                                                              uint8_t word2_sgpr) {
  const auto or32 = decode_gfx1250_sop2(*inst_it, kGfx1250SOpOrB32);
  if (!or32 || or32->sdst != high_sgpr)
    return false;
  if (or32->ssrc0 != high_sgpr && or32->ssrc1 != high_sgpr)
    return false;
  if (!previous_gfx1250_raw_buffer_descriptor_high_mask(block_begin, inst_it, high_sgpr))
    return false;

  const uint16_t tmp_src = or32->ssrc0 == high_sgpr ? or32->ssrc1 : or32->ssrc0;
  if (tmp_src >= 128u)
    return false;
  return previous_gfx1250_raw_buffer_descriptor_split_range_temp(block_begin, inst_it, word2_sgpr,
                                                                 static_cast<uint8_t>(tmp_src));
}

[[nodiscard]] bool next_gfx1250_raw_buffer_descriptor_high_pack(
    InstructionList::Iterator block_begin, InstructionList::Iterator inst_it,
    InstructionList::Iterator end, uint8_t high_sgpr, uint8_t word2_sgpr) {
  constexpr size_t kScanLimit = 8;
  size_t scanned = 0;
  for (auto scan_it = inst_it; scan_it != end && scanned < kScanLimit; ++scanned) {
    ++scan_it;
    if (scan_it == end)
      break;
    if (is_gfx1250_raw_buffer_descriptor_high_pack(block_begin, scan_it, high_sgpr, word2_sgpr))
      return true;
    if (defines_sgpr(*scan_it, high_sgpr))
      return false;
  }
  return false;
}

[[nodiscard]] bool future_gfx1250_raw_buffer_descriptor_high_pack_after_base_copy(
    InstructionList::Iterator block_begin, InstructionList::Iterator inst_it,
    InstructionList::Iterator end, uint8_t resource_base) {
  constexpr uint32_t kGfx1250RawBufferBaseHighMask = 0x01FF'FFFFu;
  constexpr size_t kScanLimit = 16;
  const uint8_t high_sgpr = static_cast<uint8_t>(resource_base + 1u);
  const uint8_t word2_sgpr = static_cast<uint8_t>(resource_base + 2u);

  size_t scanned = 0;
  for (auto scan_it = inst_it; scan_it != end && scanned < kScanLimit; ++scanned) {
    ++scan_it;
    if (scan_it == end)
      break;

    if (is_gfx1250_raw_buffer_descriptor_high_pack(block_begin, scan_it, high_sgpr, word2_sgpr))
      return true;

    if (const auto high_mask = decode_gfx1250_sop2_literal32(*scan_it, kGfx1250SOpAndB32);
        high_mask && high_mask->sdst == high_sgpr &&
        high_mask->literal == kGfx1250RawBufferBaseHighMask) {
      continue;
    }

    if (defines_sgpr(*scan_it, resource_base) || defines_sgpr(*scan_it, high_sgpr))
      return false;
  }
  return false;
}

[[nodiscard]] bool scope_order_gfx1250_raw_buffer_descriptor_high_pack_after_base_copy(
    std::span<BasicBlock *const> scope_blocks, uint64_t inst_offset, uint8_t resource_base) {
  constexpr uint32_t kGfx1250RawBufferBaseHighMask = 0x01FF'FFFFu;
  const uint8_t high_sgpr = static_cast<uint8_t>(resource_base + 1u);
  const uint8_t word2_sgpr = static_cast<uint8_t>(resource_base + 2u);

  std::vector<BasicBlock *> ordered_blocks;
  ordered_blocks.reserve(scope_blocks.size());
  for (BasicBlock *block : scope_blocks) {
    if (block != nullptr && block->end_offset() > inst_offset)
      ordered_blocks.push_back(block);
  }
  std::ranges::sort(ordered_blocks, [](const BasicBlock *lhs, const BasicBlock *rhs) {
    return lhs->start_offset() < rhs->start_offset();
  });

  size_t scanned = 0;
  for (BasicBlock *block : ordered_blocks) {
    uint64_t offset = block->start_offset();
    for (auto it = block->instructions().begin(); it != block->instructions().end(); ++it) {
      const Instruction &candidate = *it;
      const uint32_t inst_size = candidate.size();
      if (offset <= inst_offset) {
        offset += inst_size;
        continue;
      }
      if (scanned++ >= kGfx1250DescriptorUseScanLimit)
        return false;

      if (is_gfx1250_raw_buffer_descriptor_high_pack(block->instructions().begin(), it, high_sgpr,
                                                     word2_sgpr))
        return true;

      if (const auto high_mask = decode_gfx1250_sop2_literal32(candidate, kGfx1250SOpAndB32);
          high_mask && high_mask->sdst == high_sgpr &&
          high_mask->literal == kGfx1250RawBufferBaseHighMask) {
        offset += inst_size;
        continue;
      }

      if (gfx1250_vbuffer_access_size_bytes(candidate, resource_base))
        return false;
      if (defines_sgpr(candidate, resource_base) || defines_sgpr(candidate, high_sgpr))
        return false;

      offset += inst_size;
    }
  }
  return false;
}

[[nodiscard]] bool
raw_buffer_base_copy_source_tail_has_ordinary_use(InstructionList::Iterator inst_it,
                                                  InstructionList::Iterator end,
                                                  uint8_t resource_base, uint8_t source_base) {
  const std::array<uint8_t, 2> source_tail = {static_cast<uint8_t>(source_base + 2u),
                                              static_cast<uint8_t>(source_base + 3u)};
  std::array<bool, 2> active = {source_tail[0] < 128u, source_tail[1] < 128u};

  const auto active_tail_is_used = [&](const Instruction &candidate) {
    for (size_t i = 0; i < source_tail.size(); ++i) {
      if (active[i] && uses_sgpr(candidate, source_tail[i]))
        return true;
    }
    return false;
  };
  const auto deactivate_defs = [&](const Instruction &candidate) {
    for (size_t i = 0; i < source_tail.size(); ++i) {
      if (active[i] && defines_sgpr(candidate, source_tail[i]))
        active[i] = false;
    }
  };
  const auto any_active = [&] { return active[0] || active[1]; };
  const auto source_tail_span = std::span<const uint8_t>(
      source_tail.data(), source_tail[1] < 128u ? source_tail.size() : size_t{1});

  size_t scanned = 0;
  for (auto scan_it = inst_it; scan_it != end && scanned < kGfx1250DescriptorUseScanLimit;
       ++scanned) {
    ++scan_it;
    if (scan_it == end)
      break;

    const Instruction &candidate = *scan_it;
    if (gfx1250_vbuffer_access_size_bytes(candidate, resource_base))
      return false;

    if (active_tail_is_used(candidate) &&
        !is_descriptor_setup_copy_from_tracked_sgpr(candidate, source_tail_span))
      return true;

    deactivate_defs(candidate);
    if (!any_active())
      return false;

    if (defines_sgpr(candidate, resource_base))
      return false;
    if (has_no_fallthrough(candidate))
      return false;
  }

  return false;
}

[[nodiscard]] bool scope_order_raw_buffer_base_copy_source_tail_has_ordinary_use(
    std::span<BasicBlock *const> scope_blocks, uint64_t inst_offset, uint8_t resource_base,
    uint8_t source_base) {
  if (scope_blocks.empty())
    return false;

  const std::array<uint8_t, 2> source_tail = {static_cast<uint8_t>(source_base + 2u),
                                              static_cast<uint8_t>(source_base + 3u)};
  std::array<bool, 2> active = {source_tail[0] < 128u, source_tail[1] < 128u};
  const auto source_tail_span = std::span<const uint8_t>(
      source_tail.data(), source_tail[1] < 128u ? source_tail.size() : size_t{1});

  const auto active_tail_is_used = [&](const Instruction &candidate) {
    for (size_t i = 0; i < source_tail.size(); ++i) {
      if (active[i] && uses_sgpr(candidate, source_tail[i]))
        return true;
    }
    return false;
  };
  const auto deactivate_defs = [&](const Instruction &candidate) {
    for (size_t i = 0; i < source_tail.size(); ++i) {
      if (active[i] && defines_sgpr(candidate, source_tail[i]))
        active[i] = false;
    }
  };
  const auto any_active = [&] { return active[0] || active[1]; };

  std::vector<BasicBlock *> ordered_blocks;
  ordered_blocks.reserve(scope_blocks.size());
  for (BasicBlock *block : scope_blocks) {
    if (block != nullptr && block->end_offset() > inst_offset)
      ordered_blocks.push_back(block);
  }
  std::ranges::sort(ordered_blocks, [](const BasicBlock *lhs, const BasicBlock *rhs) {
    return lhs->start_offset() < rhs->start_offset();
  });

  size_t scanned = 0;
  for (BasicBlock *block : ordered_blocks) {
    uint64_t offset = block->start_offset();
    for (const Instruction &candidate : block->instructions()) {
      const uint32_t inst_size = candidate.size();
      if (offset <= inst_offset) {
        offset += inst_size;
        continue;
      }
      if (scanned++ >= kGfx1250DescriptorUseScanLimit)
        return false;

      if (gfx1250_vbuffer_access_size_bytes(candidate, resource_base))
        return false;

      if (active_tail_is_used(candidate) &&
          !is_descriptor_setup_copy_from_tracked_sgpr(candidate, source_tail_span))
        return true;

      deactivate_defs(candidate);
      if (!any_active())
        return false;

      if (defines_sgpr(candidate, resource_base))
        return false;

      offset += inst_size;
    }
  }

  return false;
}

[[nodiscard]] bool
previous_gfx1250_raw_buffer_descriptor_high_pack(InstructionList::Iterator block_begin,
                                                 InstructionList::Iterator inst_it,
                                                 uint8_t high_sgpr, uint8_t word2_sgpr) {
  constexpr size_t kScanLimit = 8;
  size_t scanned = 0;
  for (auto scan_it = inst_it; scan_it != block_begin && scanned < kScanLimit; ++scanned) {
    --scan_it;
    if (is_gfx1250_raw_buffer_descriptor_high_pack(block_begin, scan_it, high_sgpr, word2_sgpr))
      return true;
    if (defines_sgpr(*scan_it, high_sgpr))
      return false;
  }
  return false;
}

[[nodiscard]] std::vector<uint32_t> lower_gfx1250_contextual_raw_buffer_descriptor_high_pack(
    InstructionList::Iterator block_begin, InstructionList::Iterator inst_it,
    InstructionList::Iterator end, uint64_t inst_offset, rj_code_arch_t host_arch,
    const BasicBlock *block = nullptr, std::span<BasicBlock *const> scope_blocks = {}) {
  const auto or32 = decode_gfx1250_sop2(*inst_it, kGfx1250SOpOrB32);
  if (!or32)
    return {};

  const auto resource_base = raw_buffer_resource_base_for_descriptor_base_high(or32->sdst);
  if (!resource_base || (or32->ssrc0 != or32->sdst && or32->ssrc1 != or32->sdst) ||
      !previous_gfx1250_raw_buffer_descriptor_high_mask(block_begin, inst_it, or32->sdst) ||
      !(future_vbuffer_uses_resource_before_scan_limit(inst_it, end, *resource_base, block,
                                                       scope_blocks) ||
        is_gfx1250_raw_buffer_descriptor_high_pack(block_begin, inst_it, or32->sdst,
                                                   static_cast<uint8_t>(*resource_base + 2u)) ||
        scope_order_vbuffer_uses_resource(scope_blocks, inst_offset, *resource_base)))
    return {};

  return {build_s_nop(0, host_arch)};
}

[[nodiscard]] std::optional<uint32_t>
descriptor_range_from_gfx1250_high_literal(uint32_t literal_hi) {
  if ((literal_hi & 0xFFFF0000u) == 0)
    return std::nullopt;
  return literal_hi >> 25u;
}

[[nodiscard]] std::vector<uint32_t> lower_gfx1250_contextual_raw_buffer_descriptor_base(
    InstructionList::Iterator block_begin, InstructionList::Iterator inst_it,
    InstructionList::Iterator end, uint64_t inst_offset, rj_code_arch_t host_arch,
    const BasicBlock *block = nullptr, std::span<BasicBlock *const> scope_blocks = {}) {
  if (const auto mov64 = decode_gfx1250_s_mov_b64(*inst_it)) {
    const auto resource_base = raw_buffer_resource_base_for_descriptor_base(mov64->sdst);
    if (!resource_base || mov64->ssrc0 >= 126u)
      return {};

    const uint8_t high_sgpr = static_cast<uint8_t>(*resource_base + 1u);
    const auto src = static_cast<uint8_t>(mov64->ssrc0);
    if (!(future_gfx1250_raw_buffer_descriptor_high_pack_after_base_copy(block_begin, inst_it, end,
                                                                         *resource_base) ||
          scope_order_gfx1250_raw_buffer_descriptor_high_pack_after_base_copy(
              scope_blocks, inst_offset, *resource_base)) ||
        raw_buffer_base_copy_source_tail_has_ordinary_use(inst_it, end, *resource_base, src) ||
        scope_order_raw_buffer_base_copy_source_tail_has_ordinary_use(scope_blocks, inst_offset,
                                                                      *resource_base, src))
      return {};

    return {build_s_mov_b32(*resource_base, static_cast<uint16_t>(src + 1u), host_arch),
            build_s_mov_b32(high_sgpr, static_cast<uint16_t>(src + 2u), host_arch)};
  }

  if (const auto or32 = decode_gfx1250_sop2_literal32(*inst_it, kGfx1250SOpOrB32)) {
    if (or32->sdst == 0)
      return {};
    const auto range = descriptor_range_from_gfx1250_high_literal(or32->literal);
    const auto resource_base =
        raw_buffer_resource_base_for_descriptor_base(static_cast<uint8_t>(or32->sdst - 1u));
    if (!range || !resource_base ||
        !future_vbuffer_uses_resource_before_def(inst_it, end, *resource_base, or32->sdst, block,
                                                 scope_blocks))
      return {};
    const uint8_t config_sgpr = static_cast<uint8_t>(*resource_base + 3u);
    if (previous_gfx1250_raw_buffer_descriptor_high_mask(block_begin, inst_it, or32->sdst) &&
        future_vbuffer_uses_resource_before_def(inst_it, end, *resource_base, config_sgpr, block,
                                                scope_blocks)) {
      return {build_s_mov_b32(config_sgpr, 255, host_arch), kRdna4RawBufferConfigWord};
    }
    return {pack_sop2(kGfx1250SOpAndB32, or32->sdst, or32->non_literal_src, 255), 0x01FF'FFFFu};
  }

  if (const auto or64 = decode_gfx1250_sop2_literal64(*inst_it, kGfx1250SOpOrB64)) {
    const uint32_t literal_lo = static_cast<uint32_t>(or64->literal);
    const uint32_t literal_hi = static_cast<uint32_t>(or64->literal >> 32u);
    const auto range = descriptor_range_from_gfx1250_high_literal(literal_hi);
    const auto resource_base = raw_buffer_resource_base_for_descriptor_base(or64->sdst);
    if (literal_lo != 0 || !range || !resource_base || or64->non_literal_src >= 123u ||
        !future_vbuffer_uses_resource_before_def(inst_it, end, *resource_base,
                                                 static_cast<uint8_t>(or64->sdst + 1u), block,
                                                 scope_blocks))
      return {};
    return {build_s_mov_b32(or64->sdst, or64->non_literal_src, host_arch),
            pack_sop2(kGfx1250SOpAndB32, static_cast<uint8_t>(or64->sdst + 1u),
                      static_cast<uint16_t>(or64->non_literal_src + 1u), 255),
            0x01FF'FFFFu};
  }

  return {};
}

[[nodiscard]] std::vector<uint32_t> lower_gfx1250_contextual_raw_buffer_descriptor_lshr(
    InstructionList::Iterator block_begin, InstructionList::Iterator inst_it,
    InstructionList::Iterator end, uint64_t inst_offset, rj_code_arch_t host_arch,
    const BasicBlock *block = nullptr, std::span<BasicBlock *const> scope_blocks = {}) {
  const auto lshr = decode_gfx1250_sop2(*inst_it, sop2_op_lshr_b32(ROCJITSU_CODE_ARCH_GFX1250));
  if (!lshr || lshr->sdst != lshr->ssrc0 || lshr->ssrc1 != scalar_positive_inline_u32(7))
    return {};

  const auto resource_base = raw_buffer_resource_base_for_descriptor_word2(lshr->sdst);
  if (!resource_base ||
      !(future_vbuffer_uses_resource_before_def(inst_it, end, *resource_base, lshr->sdst, block,
                                                scope_blocks) ||
        previous_gfx1250_raw_buffer_descriptor_high_pack(
            block_begin, inst_it, static_cast<uint8_t>(*resource_base + 1u), lshr->sdst) ||
        scope_order_vbuffer_uses_resource_before_sgpr_reuse(scope_blocks, inst_offset,
                                                            *resource_base, lshr->sdst)))
    return {};

  return {build_s_nop(0, host_arch)};
}

[[nodiscard]] std::vector<uint32_t> lower_gfx1250_contextual_raw_buffer_descriptor_mov(
    InstructionList::Iterator inst_it, InstructionList::Iterator end, uint64_t inst_offset,
    rj_code_arch_t host_arch, const BasicBlock *block = nullptr,
    std::span<BasicBlock *const> scope_blocks = {}) {
  if (const auto mov64 = decode_gfx1250_s_mov_b64(*inst_it)) {
    if (const auto resource_base = raw_buffer_resource_base_for_descriptor_word2(mov64->sdst)) {
      const std::array<uint8_t, 2> tracked_sgprs = {mov64->sdst,
                                                    static_cast<uint8_t>(mov64->sdst + 1u)};
      const std::span<const uint8_t> zero_compare_safe_sgprs =
          gfx1250_descriptor_word2_mov_is_statically_nonzero(mov64->ssrc0, mov64->literal32,
                                                             mov64->literal64)
              ? std::span<const uint8_t>(tracked_sgprs.data(), 1)
              : std::span<const uint8_t>();
      if (future_vbuffer_uses_resource_before_any_def(inst_it, end, *resource_base, tracked_sgprs,
                                                      zero_compare_safe_sgprs, block,
                                                      scope_blocks)) {
        const uint32_t range = rdna4_range_from_gfx1250_descriptor_word2_mov(
                                   mov64->ssrc0, mov64->literal32, mov64->literal64)
                                   .value_or(kRdna4RawBufferUnboundedRange);
        return {build_s_mov_b32(mov64->sdst, 255, host_arch), range,
                build_s_mov_b32(static_cast<uint8_t>(mov64->sdst + 1u), 255, host_arch),
                kRdna4RawBufferConfigWord};
      }
    }
  }

  if (const auto movk = decode_gfx1250_s_movk_i32(*inst_it)) {
    if (const auto resource_base = raw_buffer_resource_base_for_descriptor_word2(movk->sdst);
        resource_base &&
        future_vbuffer_uses_resource_before_def(inst_it, end, *resource_base, movk->sdst, block,
                                                scope_blocks, movk->simm16 != 0)) {
      const uint32_t range = rdna4_range_from_gfx1250_descriptor_word2_units(movk->simm16)
                                 .value_or(kRdna4RawBufferUnboundedRange);
      return {build_s_mov_b32(movk->sdst, 255, host_arch), range};
    }
  }

  const auto mov = decode_gfx1250_s_mov_b32(*inst_it);
  if (!mov)
    return {};

  if (const auto resource_base = raw_buffer_resource_base_for_descriptor_config(mov->sdst);
      resource_base && (future_vbuffer_uses_resource_before_def(inst_it, end, *resource_base,
                                                                mov->sdst, block, scope_blocks) ||
                        (gfx1250_mov_src_is_zero(mov->ssrc0, mov->literal32, mov->literal64) &&
                         scope_order_vbuffer_uses_resource_before_sgpr_reuse(
                             scope_blocks, inst_offset, *resource_base, mov->sdst)))) {
    return {build_s_mov_b32(mov->sdst, 255, host_arch), kRdna4RawBufferConfigWord};
  }

  if (const auto resource_base = raw_buffer_resource_base_for_descriptor_word2(mov->sdst);
      resource_base && future_vbuffer_uses_resource_before_def(
                           inst_it, end, *resource_base, mov->sdst, block, scope_blocks,
                           gfx1250_descriptor_word2_mov_is_statically_nonzero(
                               mov->ssrc0, mov->literal32, mov->literal64))) {
    if (is_raw_buffer_descriptor_word2_sgpr_src(mov->ssrc0))
      return {};
    const uint32_t range =
        rdna4_range_from_gfx1250_descriptor_word2_mov(mov->ssrc0, mov->literal32, mov->literal64)
            .value_or(kRdna4RawBufferUnboundedRange);
    return {build_s_mov_b32(mov->sdst, 255, host_arch), range};
  }

  const bool zero_or_canonical_zero_copy =
      mov->ssrc0 == scalar_positive_inline_u32(0) || mov->ssrc0 == 2;
  if (!zero_or_canonical_zero_copy)
    return {};

  return {};
}

[[nodiscard]] std::vector<uint32_t> lower_gfx1250_smem_nv_to_rdna4(const Instruction &inst,
                                                                   rj_code_arch_t) {
  if (inst.size() != static_cast<int>(2 * sizeof(uint32_t)))
    return {};
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return {};
  if ((raw[0] >> 26u) != 0x3Du)
    return {};

  gfx1250::SmemMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  // Scaled SMEM needs a semantic expansion because RDNA4 has no equivalent
  // scale_offset bit.  Let that lowering clear nv as part of the replacement.
  if (src.nv == 0 || src.scale_offset != 0)
    return {};
  src.nv = 0;

  std::array<uint32_t, 2> words{};
  std::memcpy(words.data(), &src, sizeof(src));
  return {words[0], words[1]};
}

[[nodiscard]] std::vector<uint32_t> lower_gfx1250_s_wait_xcnt_to_rdna4(const Instruction &inst,
                                                                       rj_code_arch_t host_arch) {
  if (host_arch != ROCJITSU_CODE_ARCH_RDNA4 || inst.encoding_id() != kEnc_SOPP ||
      inst.opcode() != kGfx1250SoppWaitXcnt)
    return {};

  // GFX1250 XCNT tracks memory progress across VMEM, LDS/scratch, and SMEM
  // groups. RDNA4 has no single equivalent counter, so conservatively drain the
  // split counters.
  // For nonzero source waits this is stricter than the original instruction but
  // preserves ordering.
  return {pack_sopp(kSoppWaitLoadcnt, 0), pack_sopp(kSoppWaitStorecnt, 0),
          pack_sopp(kSoppWaitDscnt, 0), pack_sopp(kSoppWaitKmcnt, 0)};
}

[[nodiscard]] constexpr uint16_t build_hwreg(uint8_t reg_id, uint8_t offset, uint8_t size) {
  return static_cast<uint16_t>((reg_id & 0x3Fu) | ((offset & 0x1Fu) << 6) |
                               (((size - 1u) & 0x1Fu) << 11));
}

[[nodiscard]] constexpr uint32_t build_sopk(uint8_t op, uint16_t simm16, uint8_t sdst = 0) {
  return 0xB0000000u | (simm16 & 0xFFFFu) | ((sdst & 0x7Fu) << 16) | ((op & 0x1Fu) << 23);
}

[[nodiscard]] constexpr std::array<uint32_t, 3> build_scratch_store_b32(uint8_t vdata,
                                                                        uint32_t offset) {
  return {0xED06807Cu, static_cast<uint32_t>(vdata) << 23, (offset & 0xFFFFFFu) << 8};
}

[[nodiscard]] constexpr std::array<uint32_t, 3> build_scratch_load_b32(uint8_t vdst,
                                                                       uint32_t offset) {
  return {0xED05007Cu, static_cast<uint32_t>(vdst), (offset & 0xFFFFFFu) << 8};
}

void append_scratch_store_b32(std::vector<uint32_t> &words, uint8_t vdata, uint32_t offset) {
  const auto encoded = build_scratch_store_b32(vdata, offset);
  words.insert(words.end(), encoded.begin(), encoded.end());
}

void append_scratch_load_b32(std::vector<uint32_t> &words, uint8_t vdst, uint32_t offset) {
  const auto encoded = build_scratch_load_b32(vdst, offset);
  words.insert(words.end(), encoded.begin(), encoded.end());
}

[[nodiscard]] constexpr uint16_t vgpr_msb_mode_hwreg() {
  return build_hwreg(1, amdgpu::VGPR_MSB_MODE_SHIFT, 8);
}

void append_raw_s_get_vgpr_msb_mode(std::vector<uint32_t> &words, uint8_t sdst) {
  constexpr uint8_t kOpSGetregB32 = 17;
  words.push_back(build_sopk(kOpSGetregB32, vgpr_msb_mode_hwreg(), sdst));
}

void append_raw_s_set_vgpr_msb_mode_from_sgpr(std::vector<uint32_t> &words, uint8_t ssrc) {
  constexpr uint8_t kOpSSetregB32 = 18;
  words.push_back(build_sopk(kOpSSetregB32, vgpr_msb_mode_hwreg(), ssrc));
}

void append_raw_s_set_vgpr_msb_mode(std::vector<uint32_t> &words, uint8_t mode) {
  constexpr uint8_t kOpSSetregImm32B32 = 19;
  const uint32_t mode_literal = amdgpu::set_vgpr_msb_to_mode_layout(mode);
  words.push_back(build_sopk(kOpSSetregImm32B32, vgpr_msb_mode_hwreg()));
  words.push_back(mode_literal);
}

void grow_required_vgpr_count_for_src(uint32_t &minimum_vgprs, uint16_t src) {
  if (src >= 256u && src < 512u)
    minimum_vgprs = std::max(minimum_vgprs, static_cast<uint32_t>(src - 256u + 1u));
}

[[nodiscard]] constexpr uint16_t scalar_negative_inline_i32(int16_t value) {
  return static_cast<uint16_t>(192 - value);
}

[[nodiscard]] constexpr std::optional<uint16_t> scalar_inline_i32(int32_t value) {
  if (value >= 0 && value <= 64)
    return scalar_positive_inline_u32(static_cast<uint16_t>(value));
  if (value >= -16 && value <= -1)
    return scalar_negative_inline_i32(static_cast<int16_t>(value));
  return std::nullopt;
}

[[nodiscard]] constexpr std::optional<uint8_t> raw_vgpr_index(uint16_t src) {
  if (src < 256u || src >= 512u)
    return std::nullopt;
  return static_cast<uint8_t>(src - 256u);
}

[[nodiscard]] constexpr bool raw_overlaps_vdst_pair(uint8_t vgpr, uint8_t vdst) {
  return vgpr == vdst || vgpr == static_cast<uint8_t>(vdst + 1u);
}

[[nodiscard]] bool raw_overlaps_vgpr_run(uint16_t run_base, uint16_t count, uint8_t vgpr) {
  return vgpr >= run_base && vgpr < run_base + count;
}

void add_raw_avoid_vgpr(std::vector<uint8_t> &avoid, uint8_t vgpr) {
  if (std::find(avoid.begin(), avoid.end(), vgpr) == avoid.end())
    avoid.push_back(vgpr);
}

void add_raw_avoid_vgpr_run(std::vector<uint8_t> &avoid, uint8_t base, uint8_t count) {
  for (uint8_t i = 0; i < count; ++i)
    add_raw_avoid_vgpr(avoid, static_cast<uint8_t>(base + i));
}

void add_raw_avoid_src_vgpr(std::vector<uint8_t> &avoid, uint16_t src) {
  if (auto vgpr = raw_vgpr_index(src))
    add_raw_avoid_vgpr(avoid, *vgpr);
}

[[nodiscard]] std::optional<uint16_t>
raw_pair_hi_src_with_literal(uint16_t src, std::optional<uint32_t> literal) {
  if (src == 255) {
    if (!literal)
      return std::nullopt;
    return scalar_inline_i32(static_cast<int32_t>(*literal) < 0 ? -1 : 0);
  }
  if (src == 254)
    return std::nullopt;
  if (src < 128u || src >= 256u)
    return static_cast<uint16_t>(src + 1u);
  if (src >= scalar_positive_inline_u32(0) && src <= scalar_positive_inline_u32(64))
    return scalar_positive_inline_u32(0);
  if (src >= scalar_negative_inline_i32(-1) && src <= scalar_negative_inline_i32(-16))
    return scalar_negative_inline_i32(-1);
  return std::nullopt;
}

[[nodiscard]] bool raw_source_pair_reads_vdst_pair(uint8_t vdst, uint16_t src_lo, uint16_t src_hi) {
  const auto lo = raw_vgpr_index(src_lo);
  const auto hi = raw_vgpr_index(src_hi);
  return (lo && raw_overlaps_vdst_pair(*lo, vdst)) || (hi && raw_overlaps_vdst_pair(*hi, vdst));
}

void add_raw_avoid_source_pair_vgprs(std::vector<uint8_t> &avoid, uint16_t src_lo,
                                     uint16_t src_hi) {
  add_raw_avoid_src_vgpr(avoid, src_lo);
  add_raw_avoid_src_vgpr(avoid, src_hi);
}

std::optional<uint16_t> find_raw_free_vgpr_run_avoiding(const Instruction &inst,
                                                        const LivenessAnalysis &liveness,
                                                        uint16_t count,
                                                        const std::vector<uint8_t> &avoid) {
  uint16_t search_start = 0;
  while (true) {
    auto tmp_base = liveness.find_free_run(&inst, count, search_start);
    if (!tmp_base || *tmp_base + count - 1u > 255u)
      return std::nullopt;
    bool overlaps = false;
    for (uint8_t vgpr : avoid)
      overlaps |= raw_overlaps_vgpr_run(*tmp_base, count, vgpr);
    if (!overlaps)
      return tmp_base;
    search_start = static_cast<uint16_t>(*tmp_base + 1u);
  }
}

std::optional<uint8_t> find_raw_borrowable_low_vgpr_run(uint8_t count, uint8_t alignment,
                                                        const std::vector<uint8_t> &avoid) {
  for (uint16_t base = 0; base + count <= 128u; ++base) {
    if ((base % alignment) != 0)
      continue;
    bool overlaps = false;
    for (uint8_t vgpr : avoid)
      overlaps |= raw_overlaps_vgpr_run(base, count, vgpr);
    if (!overlaps)
      return static_cast<uint8_t>(base);
  }
  return std::nullopt;
}

std::optional<uint16_t> find_raw_v_mul_u64_low_scratch(uint8_t vdst, uint16_t src0,
                                                       uint16_t src0_hi, uint16_t src1,
                                                       uint16_t src1_hi, const Instruction &inst,
                                                       const LivenessAnalysis &liveness) {
  std::vector<uint8_t> avoid;
  add_raw_avoid_vgpr_run(avoid, vdst, 2);
  add_raw_avoid_source_pair_vgprs(avoid, src0, src0_hi);
  add_raw_avoid_source_pair_vgprs(avoid, src1, src1_hi);
  return find_raw_free_vgpr_run_avoiding(inst, liveness, 2, avoid);
}

void append_raw_vop3(std::vector<uint32_t> &words, uint16_t op, uint8_t vdst, uint16_t src0,
                     uint16_t src1, uint16_t src2 = 0,
                     std::optional<uint32_t> literal = std::nullopt) {
  auto [w0, w1] = build_rdna4_vop3(op, vdst, src0, src1, src2);
  words.push_back(w0);
  words.push_back(w1);
  if (literal && (src0 == 255 || src1 == 255 || src2 == 255))
    words.push_back(*literal);
}

void append_raw_vop3_sdst(std::vector<uint32_t> &words, uint16_t op, uint8_t vdst, uint8_t sdst,
                          uint16_t src0, uint16_t src1, uint16_t src2 = 0,
                          std::optional<uint32_t> literal = std::nullopt) {
  auto [w0, w1] = build_rdna4_vop3_sdst(op, vdst, sdst, src0, src1, src2);
  words.push_back(w0);
  words.push_back(w1);
  if (literal && (src0 == 255 || src1 == 255 || src2 == 255))
    words.push_back(*literal);
}

void append_raw_vop1(std::vector<uint32_t> &words, uint8_t op, uint8_t vdst, uint16_t src0) {
  words.push_back(build_rdna4_vop1(op, vdst, src0));
}

void append_raw_vop2(std::vector<uint32_t> &words, uint8_t op, uint8_t vdst, uint16_t src0,
                     uint8_t vsrc1, std::optional<uint32_t> literal = std::nullopt) {
  words.push_back(build_rdna4_vop2(op, vdst, src0, vsrc1));
  if (literal && src0 == 255)
    words.push_back(*literal);
}

[[nodiscard]] bool starts_with(std::string_view value, std::string_view prefix);
[[nodiscard]] std::vector<uint32_t> nop_words(uint32_t size, rj_code_arch_t host_arch);
void append_raw_v_mul_u64_low64(std::vector<uint32_t> &words, uint8_t out_lo, uint8_t out_hi,
                                uint16_t src0_lo, uint16_t src0_hi, uint16_t src1_lo,
                                uint16_t src1_hi, std::optional<uint32_t> literal);

enum class HighBankRole : uint8_t {
  Src0,
  Src1,
  Src2,
  Dst,
};

struct HighBankShadowPlan {
  uint8_t base = 0;
  uint16_t count = 0;
  uint16_t private_slot_count = 0;
  bool spill_to_private = false;
};

struct HighBankShadowSlot {
  uint8_t selector = 0;
  uint8_t logical = 0;
};

[[nodiscard]] bool operator==(const HighBankShadowSlot &lhs, const HighBankShadowSlot &rhs) {
  return lhs.selector == rhs.selector && lhs.logical == rhs.logical;
}

struct HighBankShadowLowSave {
  uint8_t physical = 0;
  uint8_t slot = 0;
};

struct HighBankShadowState {
  uint8_t mode = 0;
};

enum class HighBankShadowLoweringKind {
  NotApplicable,
  Lowered,
  RemappedGuest,
  Unsupported,
};

struct HighBankShadowLowering {
  HighBankShadowLoweringKind kind = HighBankShadowLoweringKind::NotApplicable;
  std::vector<uint32_t> words;
  std::string message;
  std::vector<HighBankShadowSlot> private_loads;
  std::vector<HighBankShadowSlot> private_stores;
  std::vector<uint32_t> prefix_words;
  std::vector<uint32_t> suffix_words;

  HighBankShadowLowering() = default;
  HighBankShadowLowering(HighBankShadowLoweringKind kind, std::vector<uint32_t> words,
                         std::string message = {},
                         std::vector<HighBankShadowSlot> private_loads = {},
                         std::vector<HighBankShadowSlot> private_stores = {},
                         std::vector<uint32_t> prefix_words = {},
                         std::vector<uint32_t> suffix_words = {})
      : kind(kind), words(std::move(words)), message(std::move(message)),
        private_loads(std::move(private_loads)), private_stores(std::move(private_stores)),
        prefix_words(std::move(prefix_words)), suffix_words(std::move(suffix_words)) {}
};

[[nodiscard]] constexpr uint8_t high_bank_role_shift(HighBankRole role) {
  switch (role) {
  case HighBankRole::Src0:
    return 0;
  case HighBankRole::Src1:
    return 2;
  case HighBankRole::Src2:
    return 4;
  case HighBankRole::Dst:
    return 6;
  }
  return 0;
}

[[nodiscard]] constexpr uint8_t high_bank_selector(uint8_t mode, HighBankRole role) {
  return static_cast<uint8_t>((mode >> high_bank_role_shift(role)) & 0x3u);
}

[[nodiscard]] bool is_gfx1250_s_set_vgpr_msb(uint32_t word, uint16_t &simm16) {
  const auto sopp = std::bit_cast<gfx1250::SoppMachineInst>(word);
  constexpr uint8_t kOpSSetVgprMsb = 6;
  if (sopp.encoding != 0x17Fu || sopp.op != kOpSSetVgprMsb)
    return false;
  simm16 = static_cast<uint16_t>(sopp.simm16);
  return true;
}

[[nodiscard]] std::optional<uint8_t> shadow_vgpr(uint8_t logical,
                                                 const std::optional<HighBankShadowPlan> &plan) {
  if (!plan || logical >= plan->count)
    return std::nullopt;
  const uint16_t physical = static_cast<uint16_t>(plan->base) + logical;
  if (physical > 255u)
    return std::nullopt;
  return static_cast<uint8_t>(physical);
}

[[nodiscard]] uint32_t high_bank_shadow_private_scratch_bytes(const HighBankShadowPlan &plan) {
  if (!plan.spill_to_private)
    return 0;
  return kGfx1250PrivateBorrowScratchBytes + plan.private_slot_count * sizeof(uint32_t) +
         kGfx1250HighBankShadowLowSaveBytes;
}

[[nodiscard]] std::optional<uint32_t>
high_bank_shadow_private_base(const LivenessAnalysis &liveness,
                              const std::optional<HighBankShadowPlan> &plan) {
  if (!plan || !plan->spill_to_private)
    return std::nullopt;
  const auto private_base = liveness.private_spill_base();
  if (!private_base)
    return std::nullopt;
  const uint32_t required_bytes = high_bank_shadow_private_scratch_bytes(*plan);
  if (liveness.private_spill_bytes() < required_bytes)
    return std::nullopt;
  return *private_base + kGfx1250PrivateBorrowScratchBytes;
}

[[nodiscard]] std::optional<uint32_t>
high_bank_shadow_low_save_base(const LivenessAnalysis &liveness,
                               const std::optional<HighBankShadowPlan> &plan) {
  const auto private_base = high_bank_shadow_private_base(liveness, plan);
  if (!private_base || !plan)
    return std::nullopt;
  return *private_base + plan->private_slot_count * sizeof(uint32_t);
}

[[nodiscard]] std::optional<uint16_t>
high_bank_shadow_private_slot(const HighBankShadowSlot &slot,
                              const std::optional<HighBankShadowPlan> &plan) {
  if (!plan || !plan->spill_to_private || slot.selector == 0 || slot.selector > 3)
    return std::nullopt;
  const uint16_t index = static_cast<uint16_t>((slot.selector - 1u) * 256u + slot.logical);
  if (index >= plan->private_slot_count)
    return std::nullopt;
  return index;
}

void add_unique_shadow_slot(std::vector<HighBankShadowSlot> &regs, HighBankShadowSlot slot) {
  if (std::find(regs.begin(), regs.end(), slot) == regs.end())
    regs.push_back(slot);
}

void add_unique_physical_vgpr(std::vector<uint8_t> &regs, uint8_t physical) {
  if (std::find(regs.begin(), regs.end(), physical) == regs.end())
    regs.push_back(physical);
}

[[nodiscard]] bool collect_high_bank_vgpr_spill(std::vector<HighBankShadowSlot> &regs,
                                                uint8_t logical, uint16_t width, uint8_t mode,
                                                HighBankRole role,
                                                const std::optional<HighBankShadowPlan> &plan) {
  if (!plan || !plan->spill_to_private)
    return true;
  const uint8_t selector = high_bank_selector(mode, role);
  if (selector > 3 || width == 0)
    return false;

  uint16_t remaining = width;
  uint16_t cursor = logical;
  uint8_t current_selector = selector;
  while (remaining != 0) {
    if (cursor >= 256u) {
      current_selector = static_cast<uint8_t>(current_selector + cursor / 256u);
      cursor %= 256u;
    }
    if (current_selector > 3u)
      return false;

    const uint16_t chunk = std::min<uint16_t>(remaining, static_cast<uint16_t>(256u - cursor));
    if (current_selector != 0) {
      if (cursor + chunk > plan->count)
        return false;
      for (uint16_t i = 0; i < chunk; ++i) {
        const HighBankShadowSlot slot{current_selector, static_cast<uint8_t>(cursor + i)};
        if (!high_bank_shadow_private_slot(slot, plan))
          return false;
        add_unique_shadow_slot(regs, slot);
      }
    }
    remaining = static_cast<uint16_t>(remaining - chunk);
    cursor = 0;
    ++current_selector;
  }
  return true;
}

[[nodiscard]] bool collect_high_bank_src_spill(std::vector<HighBankShadowSlot> &regs, uint16_t src,
                                               uint16_t width, uint8_t mode, HighBankRole role,
                                               const std::optional<HighBankShadowPlan> &plan) {
  if (!plan || !plan->spill_to_private)
    return true;
  const uint8_t selector = high_bank_selector(mode, role);
  if (selector == 0)
    return true;
  if (selector > 3)
    return false;
  const auto logical = raw_vgpr_index(src);
  if (!logical)
    return true;
  return collect_high_bank_vgpr_spill(regs, *logical, width, mode, role, plan);
}

[[nodiscard]] std::optional<std::vector<uint32_t>> wrap_high_bank_shadow_private_spills(
    const Instruction &inst, std::vector<uint32_t> words,
    const std::vector<HighBankShadowSlot> &loads, const std::vector<HighBankShadowSlot> &stores,
    const LivenessAnalysis &liveness, const std::optional<HighBankShadowPlan> &plan) {
  (void)inst;
  if (!plan || !plan->spill_to_private || (loads.empty() && stores.empty()))
    return words;

  const auto private_base = high_bank_shadow_private_base(liveness, plan);
  if (!private_base)
    return std::nullopt;

  const auto physical_for_slot = [&](const HighBankShadowSlot &slot) {
    return shadow_vgpr(slot.logical, plan);
  };

  for (size_t i = 0; i < loads.size(); ++i) {
    const auto lhs = physical_for_slot(loads[i]);
    if (!lhs)
      return std::nullopt;
    for (size_t j = i + 1; j < loads.size(); ++j) {
      const auto rhs = physical_for_slot(loads[j]);
      if (!rhs || (*lhs == *rhs && loads[i] != loads[j]))
        return std::nullopt;
    }
  }
  for (size_t i = 0; i < stores.size(); ++i) {
    const auto lhs = physical_for_slot(stores[i]);
    if (!lhs)
      return std::nullopt;
    for (size_t j = i + 1; j < stores.size(); ++j) {
      const auto rhs = physical_for_slot(stores[j]);
      if (!rhs || (*lhs == *rhs && stores[i] != stores[j]))
        return std::nullopt;
    }
  }

  std::vector<uint8_t> touched_physicals;
  touched_physicals.reserve(loads.size() + stores.size());
  for (const HighBankShadowSlot &slot : loads) {
    const auto physical = physical_for_slot(slot);
    if (!physical)
      return std::nullopt;
    add_unique_physical_vgpr(touched_physicals, *physical);
  }
  for (const HighBankShadowSlot &slot : stores) {
    const auto physical = physical_for_slot(slot);
    if (!physical)
      return std::nullopt;
    add_unique_physical_vgpr(touched_physicals, *physical);
  }

  const RegisterSet &live = liveness.live_before(inst);
  std::vector<HighBankShadowLowSave> low_saves;
  low_saves.reserve(touched_physicals.size());
  for (const uint8_t physical : touched_physicals) {
    if (!live.contains({RegClass::VGPR, physical, 1}))
      continue;
    if (low_saves.size() >= kGfx1250HighBankShadowLowSaveVgprCount)
      return std::nullopt;
    low_saves.push_back(HighBankShadowLowSave{physical, static_cast<uint8_t>(low_saves.size())});
  }

  const auto low_save_base = low_saves.empty() ? std::optional<uint32_t>{}
                                               : high_bank_shadow_low_save_base(liveness, plan);
  if (!low_saves.empty() && !low_save_base)
    return std::nullopt;

  std::vector<uint32_t> wrapped;
  wrapped.reserve(low_saves.size() * 6 + loads.size() * 3 + words.size() + stores.size() * 3 + 16);
  if (!low_saves.empty()) {
    wrapped.push_back(build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));
    wrapped.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
    wrapped.push_back(pack_sopp(kSoppWaitDscnt, 0));
    for (const HighBankShadowLowSave &save : low_saves) {
      append_scratch_store_b32(wrapped, save.physical,
                               *low_save_base + save.slot * sizeof(uint32_t));
    }
    wrapped.push_back(pack_sopp(kSoppWaitStorecnt, 0));
  }

  for (const HighBankShadowSlot &slot : loads) {
    const auto physical = physical_for_slot(slot);
    const auto private_slot = high_bank_shadow_private_slot(slot, plan);
    if (!physical || !private_slot)
      return std::nullopt;
    append_scratch_load_b32(wrapped, *physical, *private_base + *private_slot * sizeof(uint32_t));
  }
  if (!loads.empty())
    wrapped.push_back(pack_sopp(kSoppWaitLoadcnt, 0));

  wrapped.insert(wrapped.end(), words.begin(), words.end());

  if (!stores.empty()) {
    wrapped.push_back(build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));
    wrapped.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
    wrapped.push_back(pack_sopp(kSoppWaitDscnt, 0));
    for (const HighBankShadowSlot &slot : stores) {
      const auto physical = physical_for_slot(slot);
      const auto private_slot = high_bank_shadow_private_slot(slot, plan);
      if (!physical || !private_slot)
        return std::nullopt;
      append_scratch_store_b32(wrapped, *physical,
                               *private_base + *private_slot * sizeof(uint32_t));
    }
    wrapped.push_back(pack_sopp(kSoppWaitStorecnt, 0));
  }

  if (!low_saves.empty()) {
    for (const HighBankShadowLowSave &save : low_saves) {
      append_scratch_load_b32(wrapped, save.physical,
                              *low_save_base + save.slot * sizeof(uint32_t));
    }
    wrapped.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
  }

  return wrapped;
}

[[nodiscard]] std::optional<uint8_t>
remap_high_bank_vgpr(uint8_t logical, uint8_t mode, HighBankRole role,
                     const std::optional<HighBankShadowPlan> &plan, bool &changed) {
  const uint8_t selector = high_bank_selector(mode, role);
  if (selector == 0)
    return logical;
  if (selector > 3)
    return std::nullopt;
  auto mapped = shadow_vgpr(logical, plan);
  if (!mapped)
    return std::nullopt;
  changed = true;
  return *mapped;
}

[[nodiscard]] std::optional<uint8_t>
remap_high_bank_vgpr_run(uint8_t logical, uint16_t width, uint8_t mode, HighBankRole role,
                         const std::optional<HighBankShadowPlan> &plan, bool &changed) {
  if (width == 0 || logical + width - 1u > 255u)
    return std::nullopt;
  const uint8_t selector = high_bank_selector(mode, role);
  if (selector == 0)
    return logical;
  if (selector > 3)
    return std::nullopt;
  auto first = shadow_vgpr(logical, plan);
  auto last = shadow_vgpr(static_cast<uint8_t>(logical + width - 1u), plan);
  if (!first || !last || static_cast<uint16_t>(*first) + width - 1u > 255u)
    return std::nullopt;
  changed = true;
  return *first;
}

[[nodiscard]] std::optional<uint16_t>
remap_high_bank_src(uint16_t src, uint8_t mode, HighBankRole role,
                    const std::optional<HighBankShadowPlan> &plan, bool &changed) {
  const uint8_t selector = high_bank_selector(mode, role);
  if (selector == 0)
    return src;
  if (selector > 3)
    return std::nullopt;
  auto vgpr = raw_vgpr_index(src);
  if (!vgpr)
    return src;
  auto mapped = shadow_vgpr(*vgpr, plan);
  if (!mapped)
    return std::nullopt;
  changed = true;
  return static_cast<uint16_t>(256u + *mapped);
}

[[nodiscard]] std::optional<uint16_t>
remap_high_bank_src_run(uint16_t src, uint16_t width, uint8_t mode, HighBankRole role,
                        const std::optional<HighBankShadowPlan> &plan, bool &changed) {
  const uint8_t selector = high_bank_selector(mode, role);
  if (selector == 0)
    return src;
  if (selector > 3)
    return std::nullopt;
  auto logical = raw_vgpr_index(src);
  if (!logical)
    return src;
  auto mapped = remap_high_bank_vgpr_run(*logical, width, mode, role, plan, changed);
  if (!mapped)
    return std::nullopt;
  return static_cast<uint16_t>(256u + *mapped);
}

[[nodiscard]] bool source_run_fits_low_vgprs(uint16_t src, uint16_t width) {
  const auto base = raw_vgpr_index(src);
  return base && static_cast<uint16_t>(*base) + width <= 256u;
}

struct RemappedSrcPair {
  uint16_t lo = 0;
  uint16_t hi = 0;
};

[[nodiscard]] std::optional<RemappedSrcPair>
remap_high_bank_src_pair(uint16_t src_lo, std::optional<uint32_t> literal, uint8_t mode,
                         HighBankRole role, const std::optional<HighBankShadowPlan> &plan,
                         bool &changed) {
  auto src_hi = raw_pair_hi_src_with_literal(src_lo, literal);
  if (!src_hi)
    return std::nullopt;
  auto lo = remap_high_bank_src(src_lo, mode, role, plan, changed);
  auto hi = remap_high_bank_src(*src_hi, mode, role, plan, changed);
  if (!lo || !hi)
    return std::nullopt;
  return RemappedSrcPair{*lo, *hi};
}

[[nodiscard]] bool source_pair_overlaps_any_vgpr(const RemappedSrcPair &pair,
                                                 const std::vector<uint8_t> &vgprs) {
  const auto lo = raw_vgpr_index(pair.lo);
  const auto hi = raw_vgpr_index(pair.hi);
  return (lo && std::find(vgprs.begin(), vgprs.end(), *lo) != vgprs.end()) ||
         (hi && std::find(vgprs.begin(), vgprs.end(), *hi) != vgprs.end());
}

[[nodiscard]] std::optional<std::pair<HighBankShadowSlot, HighBankShadowSlot>>
private_shadow_source_pair_slots(uint16_t src_lo, std::optional<uint32_t> literal, uint8_t mode,
                                 HighBankRole role, const std::optional<HighBankShadowPlan> &plan) {
  const uint8_t selector = high_bank_selector(mode, role);
  if (!plan || !plan->spill_to_private || selector == 0 || selector > 3)
    return std::nullopt;
  const auto src_hi = raw_pair_hi_src_with_literal(src_lo, literal);
  if (!src_hi)
    return std::nullopt;
  const auto lo = raw_vgpr_index(src_lo);
  const auto hi = raw_vgpr_index(*src_hi);
  if (!lo || !hi)
    return std::nullopt;
  return std::pair<HighBankShadowSlot, HighBankShadowSlot>{HighBankShadowSlot{selector, *lo},
                                                           HighBankShadowSlot{selector, *hi}};
}

[[nodiscard]] bool redirect_high_bank_shadow_source_run(
    uint16_t original_src, uint16_t width, uint8_t mode, HighBankRole role, const Instruction &inst,
    const LivenessAnalysis &liveness, const std::optional<HighBankShadowPlan> &plan,
    std::vector<HighBankShadowSlot> &loads, std::vector<uint8_t> avoid,
    std::vector<uint32_t> &prefix_words, std::vector<uint32_t> &suffix_words,
    uint16_t &mapped_src) {
  if (width == 0 || width > kGfx1250RedirectPrivateBorrowedVgprCount)
    return false;
  const auto original_base = raw_vgpr_index(original_src);
  if (!original_base)
    return false;

  const auto private_spill_base = liveness.private_spill_base();
  if (!private_spill_base || liveness.private_spill_bytes() <
                                 kGfx1250RedirectPrivateBorrowOffset + width * sizeof(uint32_t))
    return false;

  const uint8_t base_selector = high_bank_selector(mode, role);
  if (base_selector > 3u)
    return false;

  for (uint16_t i = 0; i < width; ++i) {
    const uint16_t absolute = static_cast<uint16_t>(*original_base + i);
    const uint8_t selector = static_cast<uint8_t>(base_selector + absolute / 256u);
    const uint8_t logical = static_cast<uint8_t>(absolute % 256u);
    if (selector == 0) {
      add_raw_avoid_vgpr(avoid, logical);
      continue;
    }
    if (selector > 3u || !plan)
      return false;
    if (plan->spill_to_private) {
      const HighBankShadowSlot slot{selector, logical};
      if (!high_bank_shadow_private_slot(slot, plan))
        return false;
      continue;
    }
    const auto physical = shadow_vgpr(logical, plan);
    if (!physical)
      return false;
    add_raw_avoid_vgpr(avoid, *physical);
  }

  const auto borrowed = find_raw_borrowable_low_vgpr_run(static_cast<uint8_t>(width), 1, avoid);
  if (!borrowed || static_cast<uint16_t>(*borrowed) + width > 256u)
    return false;

  const uint32_t borrow_offset = *private_spill_base + kGfx1250RedirectPrivateBorrowOffset;
  prefix_words.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
  prefix_words.push_back(pack_sopp(kSoppWaitDscnt, 0));
  for (uint16_t i = 0; i < width; ++i) {
    append_scratch_store_b32(prefix_words, static_cast<uint8_t>(*borrowed + i),
                             borrow_offset + i * sizeof(uint32_t));
  }
  prefix_words.push_back(pack_sopp(kSoppWaitStorecnt, 0));

  bool loaded_private = false;
  constexpr uint8_t kOpMovB32 = 1;
  const auto remove_load = [&](HighBankShadowSlot slot) {
    loads.erase(std::remove(loads.begin(), loads.end(), slot), loads.end());
  };
  for (uint16_t i = 0; i < width; ++i) {
    const uint16_t absolute = static_cast<uint16_t>(*original_base + i);
    const uint8_t selector = static_cast<uint8_t>(base_selector + absolute / 256u);
    const uint8_t logical = static_cast<uint8_t>(absolute % 256u);
    const uint8_t dst = static_cast<uint8_t>(*borrowed + i);
    if (selector == 0) {
      append_raw_vop1(prefix_words, kOpMovB32, dst, static_cast<uint16_t>(256u + logical));
      continue;
    }
    if (!plan)
      return false;
    if (plan->spill_to_private) {
      const HighBankShadowSlot slot{selector, logical};
      const auto private_slot = high_bank_shadow_private_slot(slot, plan);
      if (!private_slot)
        return false;
      const auto private_base = high_bank_shadow_private_base(liveness, plan);
      if (!private_base)
        return false;
      append_scratch_load_b32(prefix_words, dst, *private_base + *private_slot * sizeof(uint32_t));
      remove_load(slot);
      loaded_private = true;
      continue;
    }
    const auto physical = shadow_vgpr(logical, plan);
    if (!physical)
      return false;
    append_raw_vop1(prefix_words, kOpMovB32, dst, static_cast<uint16_t>(256u + *physical));
  }
  if (loaded_private)
    prefix_words.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
  prefix_words.push_back(build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));

  suffix_words.push_back(build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));
  for (uint16_t i = 0; i < width; ++i) {
    append_scratch_load_b32(suffix_words, static_cast<uint8_t>(*borrowed + i),
                            borrow_offset + i * sizeof(uint32_t));
  }
  suffix_words.push_back(pack_sopp(kSoppWaitLoadcnt, 0));

  mapped_src = static_cast<uint16_t>(256u + *borrowed);
  (void)inst;
  return true;
}

struct ShadowFootprint {
  bool needed = false;
  bool unsupported = false;
  std::string unsupported_reason;
  bool requires_private_spill = false;
  uint8_t max_logical_vgpr = 0;
  uint16_t max_private_slot = 0;
  std::bitset<256> used_low_vgprs;
  std::vector<const Instruction *> high_mode_insts;
};

struct HighBankShadowAnalysis {
  std::optional<HighBankShadowPlan> plan;
  bool unsupported = false;
  std::string unsupported_reason;
};

void mark_high_bank_unsupported(ShadowFootprint &footprint, const Instruction &inst, uint8_t mode,
                                std::string_view reason) {
  footprint.unsupported = true;
  if (!footprint.unsupported_reason.empty())
    return;
  std::ostringstream os;
  os << reason << " for " << inst.mnemonic() << " in VGPR-MSB mode 0x" << std::hex
     << static_cast<uint32_t>(mode);
  footprint.unsupported_reason = os.str();
}

void record_high_bank_logical_vgpr(uint8_t selector, uint16_t logical, uint16_t width,
                                   ShadowFootprint &footprint) {
  if (selector == 0)
    return;
  if (selector > 3 || width == 0 || logical + width - 1u > 255u) {
    footprint.unsupported = true;
    return;
  }
  footprint.needed = true;
  if (selector != 1)
    footprint.requires_private_spill = true;
  footprint.max_logical_vgpr =
      std::max<uint8_t>(footprint.max_logical_vgpr, static_cast<uint8_t>(logical + width - 1u));
  const uint16_t max_slot = static_cast<uint16_t>((selector - 1u) * 256u + logical + width - 1u);
  footprint.max_private_slot = std::max(footprint.max_private_slot, max_slot);
}

void record_high_bank_run(uint8_t selector, uint16_t logical, uint16_t width,
                          ShadowFootprint &footprint) {
  if (width == 0) {
    footprint.unsupported = true;
    return;
  }

  uint16_t remaining = width;
  uint16_t cursor = logical;
  uint8_t current_selector = selector;
  while (remaining != 0) {
    if (cursor >= 256u) {
      current_selector = static_cast<uint8_t>(current_selector + cursor / 256u);
      cursor %= 256u;
    }
    if (current_selector > 3u) {
      footprint.unsupported = true;
      return;
    }

    const uint16_t chunk = std::min<uint16_t>(remaining, static_cast<uint16_t>(256u - cursor));
    record_high_bank_logical_vgpr(current_selector, cursor, chunk, footprint);
    remaining = static_cast<uint16_t>(remaining - chunk);
    cursor = 0;
    ++current_selector;
  }
}

void record_high_bank_src(uint16_t src, uint16_t width, uint8_t mode, HighBankRole role,
                          ShadowFootprint &footprint) {
  const uint8_t selector = high_bank_selector(mode, role);
  if (selector > 3) {
    footprint.unsupported = true;
    return;
  }
  if (auto vgpr = raw_vgpr_index(src))
    record_high_bank_run(selector, *vgpr, width, footprint);
}

void record_high_bank_vgpr(uint8_t logical, uint16_t width, uint8_t mode, HighBankRole role,
                           ShadowFootprint &footprint) {
  const uint8_t selector = high_bank_selector(mode, role);
  if (selector > 3) {
    footprint.unsupported = true;
    return;
  }
  record_high_bank_run(selector, logical, width, footprint);
}

struct DsHighBankOperands {
  bool recognized = false;
  bool uses_addr = false;
  uint16_t vdst_width = 0;
  uint16_t data0_width = 0;
};

struct VbufferHighBankOperands {
  bool recognized = false;
  bool is_load = false;
  bool is_store = false;
  uint16_t vaddr_width = 0;
  uint16_t vdata_width = 0;
};

[[nodiscard]] bool is_gfx1250_vbuffer_instruction(const Instruction &inst) {
  return inst.mnemonic().starts_with("buffer_") && (inst.encoding_id() & 0x1F8u) == kEnc_VBUFFER &&
         inst.size() >= static_cast<int>(sizeof(gfx1250::VbufferMachineInst));
}

[[nodiscard]] std::optional<uint16_t> vgpr_operand_width(const Operand *operand) {
  if (operand == nullptr || !operand->is_vgpr() || operand->size_bits() <= 0)
    return std::nullopt;
  return static_cast<uint16_t>((operand->size_bits() + 31) / 32);
}

[[nodiscard]] VbufferHighBankOperands
describe_high_bank_vbuffer_operands(const Instruction &inst, const VbufferFields &fields) {
  VbufferHighBankOperands ops;
  const std::string_view mnemonic = inst.mnemonic();

  ops.vaddr_width = static_cast<uint16_t>((fields.idxen ? 1u : 0u) + (fields.offen ? 1u : 0u));

  if (mnemonic.starts_with("buffer_load_")) {
    const auto vdata_width =
        inst.num_dst_operands() > 0 ? vgpr_operand_width(inst.dst_operand(0)) : std::nullopt;
    if (!vdata_width)
      return ops;
    ops.vdata_width = *vdata_width;
    if (fields.tfe)
      ops.vdata_width = static_cast<uint16_t>(ops.vdata_width + 1u);
    ops.recognized = true;
    ops.is_load = true;
    return ops;
  }
  if (mnemonic.starts_with("buffer_store_")) {
    const auto vdata_width =
        inst.num_src_operands() > 0 ? vgpr_operand_width(inst.src_operand(0)) : std::nullopt;
    if (!vdata_width)
      return ops;
    ops.vdata_width = *vdata_width;
    ops.recognized = true;
    ops.is_store = true;
    return ops;
  }
  return ops;
}

[[nodiscard]] uint16_t ds_vgpr_width_from_mnemonic(std::string_view mnemonic) {
  if (mnemonic.find("_b128") != std::string_view::npos)
    return 4;
  if (mnemonic.find("_b96") != std::string_view::npos)
    return 3;
  if (mnemonic.find("_b64") != std::string_view::npos)
    return 2;
  return 1;
}

[[nodiscard]] DsHighBankOperands describe_high_bank_ds_operands(std::string_view mnemonic) {
  DsHighBankOperands ops;
  if (starts_with(mnemonic, "ds_load_b") || starts_with(mnemonic, "ds_load_i") ||
      starts_with(mnemonic, "ds_load_u")) {
    ops.recognized = true;
    ops.uses_addr = true;
    ops.vdst_width = ds_vgpr_width_from_mnemonic(mnemonic);
    return ops;
  }
  if (starts_with(mnemonic, "ds_store_b")) {
    ops.recognized = true;
    ops.uses_addr = true;
    ops.data0_width = ds_vgpr_width_from_mnemonic(mnemonic);
    return ops;
  }
  if (mnemonic == "ds_permute_b32" || mnemonic == "ds_bpermute_b32") {
    ops.recognized = true;
    ops.uses_addr = true;
    ops.vdst_width = 1;
    ops.data0_width = 1;
    return ops;
  }
  return ops;
}

[[nodiscard]] bool is_single_32bit_vgpr_dst(const Instruction &inst) {
  if (inst.num_dst_operands() != 1)
    return false;
  const Operand *dst = inst.dst_operand(0);
  return dst != nullptr && dst->is_vgpr() && dst->size_bits() == 32;
}

[[nodiscard]] bool is_32bit_src_operand(const Instruction &inst, int index) {
  const Operand *src = inst.src_operand(index);
  return src != nullptr && src->size_bits() == 32;
}

[[nodiscard]] bool is_gfx1250_remappable_single_src_vop1(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
    return false;
  const auto op = std::bit_cast<gfx1250::Vop1MachineInst>(raw[0]);
  return op.encoding == 0x3Fu && is_single_32bit_vgpr_dst(inst) && inst.num_src_operands() == 1 &&
         is_32bit_src_operand(inst, 0);
}

[[nodiscard]] bool is_gfx1250_remappable_two_src_vop2(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
    return false;
  const auto op = std::bit_cast<gfx1250::Vop2MachineInst>(raw[0]);
  return op.encoding == 0 && is_single_32bit_vgpr_dst(inst) && inst.num_src_operands() == 2 &&
         is_32bit_src_operand(inst, 0) && is_32bit_src_operand(inst, 1);
}

[[nodiscard]] bool is_gfx1250_remappable_two_src_vop3(const Instruction &inst, uint16_t op) {
  if (!gfx1250_to_rdna4::rdna4_vop3_has_unused_src2(op))
    return false;
  return is_single_32bit_vgpr_dst(inst) && inst.num_src_operands() == 2 &&
         is_32bit_src_operand(inst, 0) && is_32bit_src_operand(inst, 1);
}

[[nodiscard]] bool is_gfx1250_remappable_two_src_vop3_scalar_dst(const Instruction &inst,
                                                                 uint16_t op) {
  if (!gfx1250_to_rdna4::rdna4_vop3_has_unused_src2(op) || inst.num_dst_operands() != 1)
    return false;
  const Operand *dst = inst.dst_operand(0);
  return dst != nullptr && !dst->is_vgpr() && inst.num_src_operands() == 2 &&
         is_32bit_src_operand(inst, 0) && is_32bit_src_operand(inst, 1);
}

[[nodiscard]] bool is_gfx1250_remappable_vop3_with_scalar_src2(const Instruction &inst) {
  if (!is_single_32bit_vgpr_dst(inst) || inst.num_src_operands() != 3 ||
      !is_32bit_src_operand(inst, 0) || !is_32bit_src_operand(inst, 1)) {
    return false;
  }
  const Operand *src2 = inst.src_operand(2);
  return src2 != nullptr && !src2->is_vgpr();
}

[[nodiscard]] bool is_gfx1250_remappable_three_src_vop3(const Instruction &inst) {
  return is_single_32bit_vgpr_dst(inst) && inst.num_src_operands() == 3 &&
         is_32bit_src_operand(inst, 0) && is_32bit_src_operand(inst, 1) &&
         is_32bit_src_operand(inst, 2);
}

[[nodiscard]] bool is_gfx1250_remappable_three_src_vop3p(const Instruction &inst) {
  if (inst.encoding_id() != kGfx1250Vop3pEncodingId)
    return false;
  return is_single_32bit_vgpr_dst(inst) && inst.num_src_operands() == 3 &&
         is_32bit_src_operand(inst, 0) && is_32bit_src_operand(inst, 1) &&
         is_32bit_src_operand(inst, 2);
}

void record_high_bank_vop_footprint(const Instruction &inst, uint8_t mode,
                                    ShadowFootprint &footprint) {
  if (mode == 0)
    return;
  const std::string_view mnemonic = inst.mnemonic();
  if (starts_with(mnemonic, "v_") && !starts_with(mnemonic, "v_nop"))
    footprint.high_mode_insts.push_back(&inst);
  if (starts_with(mnemonic, "ds_"))
    footprint.high_mode_insts.push_back(&inst);
  if (starts_with(mnemonic, "buffer_"))
    footprint.high_mode_insts.push_back(&inst);
  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
    return;

  const uint32_t w0 = raw[0];
  if (((w0 >> 25) & 0x3Fu) == kGfx1250VAddF16E32Opcode && inst.size() == sizeof(uint32_t)) {
    footprint.high_mode_insts.push_back(&inst);
    const auto op = std::bit_cast<gfx1250::Vop2MachineInst>(w0);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vdst), 1, mode, HighBankRole::Dst, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 1, mode, HighBankRole::Src0, footprint);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vsrc1), 1, mode, HighBankRole::Src1, footprint);
    return;
  }
  if (starts_with(mnemonic, "buffer_")) {
    if (!is_gfx1250_vbuffer_instruction(inst)) {
      mark_high_bank_unsupported(footprint, inst, mode, "unrecognized VBUFFER encoding");
      return;
    }
    const auto fields = gfx1250_to_rdna4::decode_vbuffer_gfx1250(raw[0], raw[1], raw[2]);
    const VbufferHighBankOperands ops = describe_high_bank_vbuffer_operands(inst, fields);
    if (!ops.recognized) {
      mark_high_bank_unsupported(footprint, inst, mode, "unrecognized VBUFFER operands");
      return;
    }
    if (ops.vaddr_width != 0)
      record_high_bank_vgpr(static_cast<uint8_t>(fields.vaddr), ops.vaddr_width, mode,
                            HighBankRole::Src0, footprint);
    if (ops.is_load && ops.vdata_width != 0)
      record_high_bank_vgpr(static_cast<uint8_t>(fields.vdata), ops.vdata_width, mode,
                            HighBankRole::Dst, footprint);
    if (ops.is_store && ops.vdata_width != 0)
      record_high_bank_vgpr(static_cast<uint8_t>(fields.vdata), ops.vdata_width, mode,
                            HighBankRole::Src1, footprint);
    return;
  }
  if (starts_with(mnemonic, "ds_")) {
    const DsHighBankOperands ops = describe_high_bank_ds_operands(mnemonic);
    if (!ops.recognized || inst.size() < static_cast<int>(sizeof(gfx1250::VdsMachineInst))) {
      mark_high_bank_unsupported(footprint, inst, mode, "unrecognized DS operands");
      return;
    }
    gfx1250::VdsMachineInst op{};
    std::memcpy(&op, raw, sizeof(op));
    if (ops.uses_addr)
      record_high_bank_vgpr(static_cast<uint8_t>(op.addr), 1, mode, HighBankRole::Src0, footprint);
    if (ops.vdst_width != 0)
      record_high_bank_vgpr(static_cast<uint8_t>(op.vdst), ops.vdst_width, mode, HighBankRole::Dst,
                            footprint);
    if (ops.data0_width != 0)
      record_high_bank_vgpr(static_cast<uint8_t>(op.data0), ops.data0_width, mode,
                            HighBankRole::Src1, footprint);
    return;
  }
  if (is_gfx1250_remappable_single_src_vop1(inst)) {
    const auto op = std::bit_cast<gfx1250::Vop1MachineInst>(w0);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vdst), 1, mode, HighBankRole::Dst, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 1, mode, HighBankRole::Src0, footprint);
    return;
  }
  if (starts_with(mnemonic, "v_readfirstlane_b32")) {
    const auto op = std::bit_cast<gfx1250::Vop1MachineInst>(w0);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 1, mode, HighBankRole::Src0, footprint);
    return;
  }
  if (mnemonic == "v_mov_b64_e32") {
    const auto op = std::bit_cast<gfx1250::Vop1MachineInst>(w0);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vdst), 2, mode, HighBankRole::Dst, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 2, mode, HighBankRole::Src0, footprint);
    return;
  }
  if (mnemonic == "v_add_nc_u64_e32" || mnemonic == "v_sub_nc_u64_e32" ||
      mnemonic == "v_mul_u64_e32" || mnemonic == "v_lshlrev_b64_e32") {
    const auto op = std::bit_cast<gfx1250::Vop2MachineInst>(w0);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vdst), 2, mode, HighBankRole::Dst, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 2, mode, HighBankRole::Src0, footprint);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vsrc1), 2, mode, HighBankRole::Src1, footprint);
    return;
  }
  if (is_gfx1250_remappable_two_src_vop2(inst) || mnemonic == "v_add_f16_e32" ||
      (((w0 >> 25) & 0x3Fu) == kGfx1250VAddF16E32Opcode && inst.size() == sizeof(uint32_t))) {
    const auto op = std::bit_cast<gfx1250::Vop2MachineInst>(w0);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vdst), 1, mode, HighBankRole::Dst, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 1, mode, HighBankRole::Src0, footprint);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vsrc1), 1, mode, HighBankRole::Src1, footprint);
    return;
  }
  if (starts_with(mnemonic, "v_cmp") && (mnemonic.find("_u64_e32") != std::string_view::npos ||
                                         mnemonic.find("_i64_e32") != std::string_view::npos)) {
    const auto op = std::bit_cast<gfx1250::VopcMachineInst>(w0);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 2, mode, HighBankRole::Src0, footprint);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vsrc1), 2, mode, HighBankRole::Src1, footprint);
    return;
  }
  if (inst.encoding_id() == kGfx1250Vop3pEncodingId &&
      inst.opcode() == kGfx1250SwmmacF32F16K64Opcode &&
      inst.size() >= static_cast<int>(sizeof(gfx1250::Vop3pMachineInst))) {
    gfx1250::Vop3pMachineInst op{};
    std::memcpy(&op, raw, sizeof(op));
    record_high_bank_vgpr(static_cast<uint8_t>(op.vdst), 8, mode, HighBankRole::Dst, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 8, mode, HighBankRole::Src0, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src1), 16, mode, HighBankRole::Src1, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src2), 1, mode, HighBankRole::Src2, footprint);
    return;
  }
  if (is_gfx1250_pk_f32_vop3p(inst) &&
      inst.size() >= static_cast<int>(sizeof(gfx1250::Vop3pMachineInst))) {
    gfx1250::Vop3pMachineInst op{};
    std::memcpy(&op, raw, sizeof(op));
    record_high_bank_vgpr(static_cast<uint8_t>(op.vdst), 2, mode, HighBankRole::Dst, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 2, mode, HighBankRole::Src0, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src1), 2, mode, HighBankRole::Src1, footprint);
    if (op.op == kGfx1250VPkFmaF32Vop3pOpcode)
      record_high_bank_src(static_cast<uint16_t>(op.src2), 2, mode, HighBankRole::Src2, footprint);
    return;
  }
  if (is_gfx1250_remappable_three_src_vop3p(inst) &&
      inst.size() >= static_cast<int>(sizeof(gfx1250::Vop3pMachineInst))) {
    gfx1250::Vop3pMachineInst op{};
    std::memcpy(&op, raw, sizeof(op));
    record_high_bank_vgpr(static_cast<uint8_t>(op.vdst), 1, mode, HighBankRole::Dst, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 1, mode, HighBankRole::Src0, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src1), 1, mode, HighBankRole::Src1, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src2), 1, mode, HighBankRole::Src2, footprint);
    return;
  }
  if (inst.size() >= static_cast<int>(2 * sizeof(uint32_t)) && (w0 >> 26) == kVop3Encoding) {
    const uint16_t op = static_cast<uint16_t>((w0 >> 16) & 0x3FFu);
    const uint8_t vdst = static_cast<uint8_t>(w0 & 0xFFu);
    const uint32_t w1 = raw[1];
    const uint16_t src0 = static_cast<uint16_t>(w1 & 0x1FFu);
    const uint16_t src1 = static_cast<uint16_t>((w1 >> 9) & 0x1FFu);
    const uint16_t src2 = static_cast<uint16_t>((w1 >> 18) & 0x1FFu);
    if (op == kGfx1250VAddCoU32Vop3SdstOpcode) {
      record_high_bank_vgpr(vdst, 1, mode, HighBankRole::Dst, footprint);
      record_high_bank_src(src0, 1, mode, HighBankRole::Src0, footprint);
      record_high_bank_src(src1, 1, mode, HighBankRole::Src1, footprint);
      return;
    }
    if (op == 594u) {
      record_high_bank_vgpr(vdst, 2, mode, HighBankRole::Dst, footprint);
      record_high_bank_src(src0, 2, mode, HighBankRole::Src0, footprint);
      record_high_bank_src(src1, 1, mode, HighBankRole::Src1, footprint);
      record_high_bank_src(src2, 2, mode, HighBankRole::Src2, footprint);
      return;
    }
    if (op == 762u) {
      record_high_bank_vgpr(vdst, 2, mode, HighBankRole::Dst, footprint);
      record_high_bank_src(src0, 1, mode, HighBankRole::Src0, footprint);
      record_high_bank_src(src1, 1, mode, HighBankRole::Src1, footprint);
      record_high_bank_src(src2, 2, mode, HighBankRole::Src2, footprint);
      return;
    }
    if (op == 582u || op == 597u || op == 598u) {
      record_high_bank_vgpr(vdst, 1, mode, HighBankRole::Dst, footprint);
      record_high_bank_src(src0, 1, mode, HighBankRole::Src0, footprint);
      record_high_bank_src(src1, 1, mode, HighBankRole::Src1, footprint);
      record_high_bank_src(src2, 1, mode, HighBankRole::Src2, footprint);
      return;
    }
    if (op == 599u || op == 600u) {
      record_high_bank_vgpr(vdst, 1, mode, HighBankRole::Dst, footprint);
      record_high_bank_src(src0, 1, mode, HighBankRole::Src0, footprint);
      record_high_bank_src(src1, 1, mode, HighBankRole::Src1, footprint);
      record_high_bank_src(src2, 1, mode, HighBankRole::Src2, footprint);
      return;
    }
    if (op == 812u || op == 813u) {
      record_high_bank_vgpr(vdst, 1, mode, HighBankRole::Dst, footprint);
      record_high_bank_src(src0, 1, mode, HighBankRole::Src0, footprint);
      record_high_bank_src(src1, 1, mode, HighBankRole::Src1, footprint);
      return;
    }
    if (op == kGfx1250VCvtPkF16F32Vop3Opcode) {
      record_high_bank_vgpr(vdst, 1, mode, HighBankRole::Dst, footprint);
      record_high_bank_src(src0, 1, mode, HighBankRole::Src0, footprint);
      record_high_bank_src(src1, 1, mode, HighBankRole::Src1, footprint);
      return;
    }
    if (is_gfx1250_remappable_vop3_with_scalar_src2(inst)) {
      record_high_bank_vgpr(vdst, 1, mode, HighBankRole::Dst, footprint);
      record_high_bank_src(src0, 1, mode, HighBankRole::Src0, footprint);
      record_high_bank_src(src1, 1, mode, HighBankRole::Src1, footprint);
      return;
    }
    if (is_gfx1250_remappable_three_src_vop3(inst)) {
      record_high_bank_vgpr(vdst, 1, mode, HighBankRole::Dst, footprint);
      record_high_bank_src(src0, 1, mode, HighBankRole::Src0, footprint);
      record_high_bank_src(src1, 1, mode, HighBankRole::Src1, footprint);
      record_high_bank_src(src2, 1, mode, HighBankRole::Src2, footprint);
      return;
    }
    if (is_gfx1250_remappable_two_src_vop3(inst, op)) {
      record_high_bank_vgpr(vdst, 1, mode, HighBankRole::Dst, footprint);
      record_high_bank_src(src0, 1, mode, HighBankRole::Src0, footprint);
      record_high_bank_src(src1, 1, mode, HighBankRole::Src1, footprint);
      return;
    }
    if (is_gfx1250_remappable_two_src_vop3_scalar_dst(inst, op)) {
      record_high_bank_src(src0, 1, mode, HighBankRole::Src0, footprint);
      record_high_bank_src(src1, 1, mode, HighBankRole::Src1, footprint);
      return;
    }
  }

  if (starts_with(mnemonic, "v_") && !starts_with(mnemonic, "v_nop"))
    mark_high_bank_unsupported(footprint, inst, mode, "unrecognized VALU operands");
}

void record_low_vgpr_uses(const Instruction &inst, ShadowFootprint &footprint) {
  InstDefUse du(inst);
  const auto record = [&](RegisterRef ref) {
    if (ref.cls != RegClass::VGPR || ref.index >= 256u)
      return;
    footprint.used_low_vgprs.set(ref.index);
  };
  du.defs.for_each(record);
  du.uses.for_each(record);
}

[[nodiscard]] bool shadow_window_is_live(const ShadowFootprint &footprint,
                                         const LivenessAnalysis &liveness, uint16_t base,
                                         uint16_t count) {
  for (const Instruction *inst : footprint.high_mode_insts) {
    if (inst == nullptr)
      continue;
    const RegisterSet &live = liveness.live_before(*inst);
    for (uint16_t i = 0; i < count; ++i) {
      if (live.contains({RegClass::VGPR, static_cast<uint16_t>(base + i), 1}))
        return true;
    }
  }
  return false;
}

[[nodiscard]] bool shadow_window_overlaps_low_uses(const ShadowFootprint &footprint, uint16_t base,
                                                   uint16_t count) {
  for (uint16_t i = 0; i < count; ++i) {
    if (footprint.used_low_vgprs.test(base + i))
      return true;
  }
  return false;
}

[[nodiscard]] std::optional<uint8_t> find_shadow_base(const ShadowFootprint &footprint,
                                                      const LivenessAnalysis &liveness,
                                                      uint16_t count) {
  if (count == 0 || count > 256u)
    return std::nullopt;
  const auto find_in_range = [&](uint16_t begin, uint16_t end,
                                 bool require_globally_unused) -> std::optional<uint8_t> {
    for (uint16_t base = begin; base + count <= end; ++base) {
      bool overlaps = false;
      for (uint16_t i = 0; i < count; ++i) {
        if (require_globally_unused && footprint.used_low_vgprs.test(base + i)) {
          overlaps = true;
          break;
        }
      }
      if (!overlaps && shadow_window_is_live(footprint, liveness, base, count))
        overlaps = true;
      if (!overlaps)
        return static_cast<uint8_t>(base);
    }
    return std::nullopt;
  };

  if (auto base = find_in_range(64, 128, true))
    return base;
  if (auto base = find_in_range(128, 256, true))
    return base;
  if (auto base = find_in_range(0, 256, true))
    return base;
  if (auto base = find_in_range(64, 128, false))
    return base;
  if (auto base = find_in_range(128, 256, false))
    return base;
  if (auto base = find_in_range(0, 256, false))
    return base;

  // Dense IREE kernels can mention most low VGPRs somewhere in the scope even
  // though the high-bank regions have short local live ranges. Keep the shadow
  // window inside the capped RDNA4 Wave32 allocation; hardware validation
  // exercises whether the fallback aliases a real low-bank live range.
  if (64u + count <= 256u)
    return 64u;
  if (128u + count <= 256u)
    return 128u;
  return std::nullopt;
}

HighBankShadowLowering
lower_gfx1250_high_bank_shadow_instruction(std::span<const uint8_t> text, const Instruction &inst,
                                           uint64_t offset, const LivenessAnalysis &liveness,
                                           const std::optional<HighBankShadowPlan> &plan,
                                           HighBankShadowState &state) {
  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
    return {};

  uint16_t simm16 = 0;
  if (is_gfx1250_s_set_vgpr_msb(raw[0], simm16)) {
    state.mode = amdgpu::s_set_vgpr_msb_new_mode(simm16);
    return {HighBankShadowLoweringKind::Lowered,
            nop_words(static_cast<uint32_t>(inst.size()), ROCJITSU_CODE_ARCH_RDNA4),
            {}};
  }

  if (state.mode == 0)
    return {};

  const auto unsupported = [&]() {
    std::ostringstream os;
    os << "expanded text copy cannot virtualize gfx1250 high-bank operands for " << inst.mnemonic()
       << " at .text+0x" << std::hex << offset;
    if (plan)
      os << " mode=0x" << static_cast<uint32_t>(state.mode) << " shadow_base=0x"
         << static_cast<uint32_t>(plan->base) << " shadow_count=0x"
         << static_cast<uint32_t>(plan->count);
    else
      os << " (no shadow window)";
    return HighBankShadowLowering{HighBankShadowLoweringKind::Unsupported, {}, os.str()};
  };

  bool changed = false;
  const std::string_view mnemonic = inst.mnemonic();
  const uint32_t w0 = raw[0];
  if (starts_with(mnemonic, "v_nop"))
    return {};

  if (starts_with(mnemonic, "ds_")) {
    const DsHighBankOperands ops = describe_high_bank_ds_operands(mnemonic);
    if (!ops.recognized || inst.size() < static_cast<int>(sizeof(gfx1250::VdsMachineInst)))
      return unsupported();

    gfx1250::VdsMachineInst src{};
    std::memcpy(&src, raw, sizeof(src));
    gfx1250::VdsMachineInst dst = src;

    std::vector<HighBankShadowSlot> loads;
    std::vector<HighBankShadowSlot> stores;
    if (ops.uses_addr) {
      if (!collect_high_bank_vgpr_spill(loads, static_cast<uint8_t>(src.addr), 1, state.mode,
                                        HighBankRole::Src0, plan))
        return unsupported();
      auto mapped_addr = remap_high_bank_vgpr(static_cast<uint8_t>(src.addr), state.mode,
                                              HighBankRole::Src0, plan, changed);
      if (!mapped_addr)
        return unsupported();
      dst.addr = *mapped_addr;
    }
    if (ops.vdst_width != 0) {
      if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), ops.vdst_width,
                                        state.mode, HighBankRole::Dst, plan))
        return unsupported();
      auto mapped_vdst = remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst), state.mode,
                                              HighBankRole::Dst, plan, changed);
      if (!mapped_vdst)
        return unsupported();
      dst.vdst = *mapped_vdst;
    }
    if (ops.data0_width != 0) {
      if (!collect_high_bank_vgpr_spill(loads, static_cast<uint8_t>(src.data0), ops.data0_width,
                                        state.mode, HighBankRole::Src1, plan))
        return unsupported();
      auto mapped_data0 = remap_high_bank_vgpr(static_cast<uint8_t>(src.data0), state.mode,
                                               HighBankRole::Src1, plan, changed);
      if (!mapped_data0)
        return unsupported();
      dst.data0 = *mapped_data0;
    }
    if (!changed)
      return {};

    std::array<uint32_t, 2> words{};
    std::memcpy(words.data(), &dst, sizeof(dst));
    return {HighBankShadowLoweringKind::RemappedGuest,
            {words[0], words[1]},
            {},
            std::move(loads),
            std::move(stores)};
  }

  if (starts_with(mnemonic, "buffer_")) {
    if (!is_gfx1250_vbuffer_instruction(inst))
      return unsupported();

    VbufferFields fields = gfx1250_to_rdna4::decode_vbuffer_gfx1250(raw[0], raw[1], raw[2]);
    const VbufferHighBankOperands ops = describe_high_bank_vbuffer_operands(inst, fields);
    if (!ops.recognized)
      return unsupported();

    std::vector<HighBankShadowSlot> loads;
    std::vector<HighBankShadowSlot> stores;
    if (ops.vaddr_width != 0) {
      if (!collect_high_bank_vgpr_spill(loads, static_cast<uint8_t>(fields.vaddr), ops.vaddr_width,
                                        state.mode, HighBankRole::Src0, plan))
        return unsupported();
      auto mapped_vaddr =
          remap_high_bank_vgpr_run(static_cast<uint8_t>(fields.vaddr), ops.vaddr_width, state.mode,
                                   HighBankRole::Src0, plan, changed);
      if (!mapped_vaddr)
        return unsupported();
      fields.vaddr = *mapped_vaddr;
    }
    if (ops.is_load && ops.vdata_width != 0) {
      if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(fields.vdata), ops.vdata_width,
                                        state.mode, HighBankRole::Dst, plan))
        return unsupported();
      auto mapped_vdata =
          remap_high_bank_vgpr_run(static_cast<uint8_t>(fields.vdata), ops.vdata_width, state.mode,
                                   HighBankRole::Dst, plan, changed);
      if (!mapped_vdata)
        return unsupported();
      fields.vdata = *mapped_vdata;
    }
    if (ops.is_store && ops.vdata_width != 0) {
      if (!collect_high_bank_vgpr_spill(loads, static_cast<uint8_t>(fields.vdata), ops.vdata_width,
                                        state.mode, HighBankRole::Src1, plan))
        return unsupported();
      auto mapped_vdata =
          remap_high_bank_vgpr_run(static_cast<uint8_t>(fields.vdata), ops.vdata_width, state.mode,
                                   HighBankRole::Src1, plan, changed);
      if (!mapped_vdata)
        return unsupported();
      fields.vdata = *mapped_vdata;
    }
    if (!changed)
      return {};

    const TranslationResult translated =
        gfx1250_to_rdna4::encode_vbuffer_rdna4(fields, static_cast<uint16_t>(fields.op));
    if (translated.word_count == 0)
      return unsupported();
    std::vector<uint32_t> words(translated.words, translated.words + translated.word_count);
    auto wrapped =
        wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores, liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (is_gfx1250_remappable_single_src_vop1(inst)) {
    const auto src = std::bit_cast<gfx1250::Vop1MachineInst>(w0);
    const std::optional<uint32_t> literal =
        src.src0 == 255u && inst.size() >= static_cast<int>(2 * sizeof(uint32_t))
            ? read_trailing_literal_u32(text, offset, sizeof(uint32_t))
            : std::nullopt;
    if (src.src0 == 255u && !literal)
      return unsupported();
    std::vector<HighBankShadowSlot> loads;
    std::vector<HighBankShadowSlot> stores;
    if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), 1, state.mode,
                                      HighBankRole::Dst, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 1, state.mode,
                                     HighBankRole::Src0, plan))
      return unsupported();
    auto vdst = remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst), state.mode, HighBankRole::Dst,
                                     plan, changed);
    auto src0 = remap_high_bank_src(static_cast<uint16_t>(src.src0), state.mode, HighBankRole::Src0,
                                    plan, changed);
    if (!vdst || !src0)
      return unsupported();
    if (!changed)
      return {};
    std::vector<uint32_t> words{build_rdna4_vop1(static_cast<uint8_t>(src.op), *vdst, *src0)};
    if (literal)
      words.push_back(*literal);
    auto wrapped =
        wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores, liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (starts_with(mnemonic, "v_readfirstlane_b32")) {
    const auto src = std::bit_cast<gfx1250::Vop1MachineInst>(w0);
    std::vector<HighBankShadowSlot> loads;
    if (!collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 1, state.mode,
                                     HighBankRole::Src0, plan))
      return unsupported();
    auto src0 = remap_high_bank_src(static_cast<uint16_t>(src.src0), state.mode, HighBankRole::Src0,
                                    plan, changed);
    if (!src0)
      return unsupported();
    if (!changed)
      return {};
    auto wrapped = wrap_high_bank_shadow_private_spills(
        inst,
        {build_rdna4_vop1(static_cast<uint8_t>(src.op), static_cast<uint8_t>(src.vdst), *src0)},
        loads, {}, liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (mnemonic == "v_mov_b64_e32") {
    const auto src = std::bit_cast<gfx1250::Vop1MachineInst>(w0);
    std::vector<HighBankShadowSlot> loads;
    std::vector<HighBankShadowSlot> stores;
    if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), 2, state.mode,
                                      HighBankRole::Dst, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 2, state.mode,
                                     HighBankRole::Src0, plan))
      return unsupported();
    auto dst_lo = remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst), state.mode,
                                       HighBankRole::Dst, plan, changed);
    auto dst_hi = src.vdst < 255u
                      ? remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst + 1u), state.mode,
                                             HighBankRole::Dst, plan, changed)
                      : std::optional<uint8_t>{};
    auto src_pair = remap_high_bank_src_pair(static_cast<uint16_t>(src.src0), std::nullopt,
                                             state.mode, HighBankRole::Src0, plan, changed);
    if (!dst_lo || !dst_hi || !src_pair)
      return unsupported();
    if (!changed)
      return {};
    std::vector<uint32_t> words;
    append_raw_vop1(words, 1, *dst_lo, src_pair->lo);
    append_raw_vop1(words, 1, *dst_hi, src_pair->hi);
    auto wrapped =
        wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores, liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (mnemonic == "v_add_nc_u64_e32" || mnemonic == "v_sub_nc_u64_e32") {
    const auto src = std::bit_cast<gfx1250::Vop2MachineInst>(w0);
    const std::optional<uint32_t> literal =
        src.src0 == 255u && inst.size() >= static_cast<int>(2 * sizeof(uint32_t))
            ? read_trailing_literal_u32(text, offset, sizeof(uint32_t))
            : std::nullopt;
    if (src.src0 == 255u && !literal)
      return unsupported();
    std::vector<HighBankShadowSlot> loads;
    std::vector<HighBankShadowSlot> stores;
    if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), 2, state.mode,
                                      HighBankRole::Dst, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 2, state.mode,
                                     HighBankRole::Src0, plan) ||
        !collect_high_bank_vgpr_spill(loads, static_cast<uint8_t>(src.vsrc1), 2, state.mode,
                                      HighBankRole::Src1, plan))
      return unsupported();
    auto dst_lo = remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst), state.mode,
                                       HighBankRole::Dst, plan, changed);
    auto dst_hi = src.vdst < 255u
                      ? remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst + 1u), state.mode,
                                             HighBankRole::Dst, plan, changed)
                      : std::optional<uint8_t>{};
    auto src0_pair = remap_high_bank_src_pair(static_cast<uint16_t>(src.src0), literal, state.mode,
                                              HighBankRole::Src0, plan, changed);
    auto src1_pair = remap_high_bank_src_pair(static_cast<uint16_t>(256u + src.vsrc1), std::nullopt,
                                              state.mode, HighBankRole::Src1, plan, changed);
    if (!dst_lo || !dst_hi || !src0_pair || !src1_pair)
      return unsupported();
    if (!changed)
      return {};

    std::vector<uint8_t> protected_low_sources;
    if (high_bank_selector(state.mode, HighBankRole::Src0) == 0)
      add_raw_avoid_source_pair_vgprs(protected_low_sources, src0_pair->lo, src0_pair->hi);
    if (high_bank_selector(state.mode, HighBankRole::Src1) == 0)
      add_raw_avoid_source_pair_vgprs(protected_low_sources, src1_pair->lo, src1_pair->hi);

    std::vector<uint8_t> temp_avoid = protected_low_sources;
    add_raw_avoid_vgpr_run(temp_avoid, *dst_lo, 2);
    add_raw_avoid_source_pair_vgprs(temp_avoid, src0_pair->lo, src0_pair->hi);
    add_raw_avoid_source_pair_vgprs(temp_avoid, src1_pair->lo, src1_pair->hi);

    std::vector<uint32_t> prefix_words;
    std::vector<uint32_t> suffix_words;
    uint32_t borrowed_private_slots = 0;
    const auto redirect_private_shadow_pair = [&](RemappedSrcPair &pair, uint16_t original_src_lo,
                                                  std::optional<uint32_t> literal,
                                                  HighBankRole role) -> bool {
      if (!source_pair_overlaps_any_vgpr(pair, protected_low_sources))
        return true;
      const auto slots =
          private_shadow_source_pair_slots(original_src_lo, literal, state.mode, role, plan);
      if (!slots)
        return true;
      const auto private_base = high_bank_shadow_private_base(liveness, plan);
      if (!private_base)
        return false;
      const auto tmp_base = find_raw_free_vgpr_run_avoiding(inst, liveness, 2, temp_avoid);
      std::optional<uint16_t> redirected_base = tmp_base;
      if (!redirected_base || *redirected_base > 254u) {
        const auto private_spill_base = liveness.private_spill_base();
        if (!private_spill_base ||
            liveness.private_spill_bytes() < kGfx1250PrivateBorrowScratchBytes ||
            borrowed_private_slots + 2u > kGfx1250PrivateBorrowedVgprCount)
          return false;
        const auto borrowed_base = find_raw_borrowable_low_vgpr_run(2, 2, temp_avoid);
        if (!borrowed_base || *borrowed_base > 254u)
          return false;

        redirected_base = *borrowed_base;
        const uint32_t borrow_offset =
            *private_spill_base + borrowed_private_slots * sizeof(uint32_t);
        prefix_words.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
        prefix_words.push_back(pack_sopp(kSoppWaitDscnt, 0));
        append_scratch_store_b32(prefix_words, static_cast<uint8_t>(*redirected_base),
                                 borrow_offset);
        append_scratch_store_b32(prefix_words, static_cast<uint8_t>(*redirected_base + 1u),
                                 borrow_offset + sizeof(uint32_t));
        prefix_words.push_back(pack_sopp(kSoppWaitStorecnt, 0));

        suffix_words.push_back(build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));
        append_scratch_load_b32(suffix_words, static_cast<uint8_t>(*redirected_base),
                                borrow_offset);
        append_scratch_load_b32(suffix_words, static_cast<uint8_t>(*redirected_base + 1u),
                                borrow_offset + sizeof(uint32_t));
        suffix_words.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
        borrowed_private_slots += 2u;
      }

      const auto remove_load = [&](HighBankShadowSlot slot) {
        loads.erase(std::remove(loads.begin(), loads.end(), slot), loads.end());
      };
      const auto lo_private_slot = high_bank_shadow_private_slot(slots->first, plan);
      const auto hi_private_slot = high_bank_shadow_private_slot(slots->second, plan);
      if (!lo_private_slot || !hi_private_slot)
        return false;
      append_scratch_load_b32(prefix_words, static_cast<uint8_t>(*redirected_base),
                              *private_base + *lo_private_slot * sizeof(uint32_t));
      append_scratch_load_b32(prefix_words, static_cast<uint8_t>(*redirected_base + 1u),
                              *private_base + *hi_private_slot * sizeof(uint32_t));
      prefix_words.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
      remove_load(slots->first);
      remove_load(slots->second);
      pair.lo = static_cast<uint16_t>(256u + *redirected_base);
      pair.hi = static_cast<uint16_t>(256u + *redirected_base + 1u);
      add_raw_avoid_vgpr_run(temp_avoid, static_cast<uint8_t>(*redirected_base), 2);
      return true;
    };

    if (!redirect_private_shadow_pair(*src0_pair, static_cast<uint16_t>(src.src0), literal,
                                      HighBankRole::Src0) ||
        !redirect_private_shadow_pair(*src1_pair, static_cast<uint16_t>(256u + src.vsrc1),
                                      std::nullopt, HighBankRole::Src1))
      return unsupported();

    const auto carry_opt = liveness.find_free_sgpr_pair(&inst);
    if (!carry_opt || *carry_opt > 105u)
      return unsupported();
    const uint8_t carry = static_cast<uint8_t>(*carry_opt);
    const bool is_sub = mnemonic == "v_sub_nc_u64_e32";
    std::vector<uint32_t> words = std::move(prefix_words);
    append_raw_vop3_sdst(words, is_sub ? 769 : 768, *dst_lo, carry, src0_pair->lo, src1_pair->lo, 0,
                         literal);
    words.push_back(build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));
    append_raw_vop3_sdst(words, is_sub ? 289 : 288, *dst_hi, kRdna4NullSgpr, src0_pair->hi,
                         src1_pair->hi, carry);
    words.insert(words.end(), suffix_words.begin(), suffix_words.end());
    auto wrapped =
        wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores, liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (mnemonic == "v_mul_u64_e32") {
    const auto src = std::bit_cast<gfx1250::Vop2MachineInst>(w0);
    const std::optional<uint32_t> literal =
        src.src0 == 255u && inst.size() >= static_cast<int>(2 * sizeof(uint32_t))
            ? read_trailing_literal_u32(text, offset, sizeof(uint32_t))
            : std::nullopt;
    if (src.src0 == 255u && !literal)
      return unsupported();
    std::vector<HighBankShadowSlot> loads;
    std::vector<HighBankShadowSlot> stores;
    if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), 2, state.mode,
                                      HighBankRole::Dst, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 2, state.mode,
                                     HighBankRole::Src0, plan) ||
        !collect_high_bank_vgpr_spill(loads, static_cast<uint8_t>(src.vsrc1), 2, state.mode,
                                      HighBankRole::Src1, plan))
      return unsupported();
    auto dst_lo = remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst), state.mode,
                                       HighBankRole::Dst, plan, changed);
    auto dst_hi = src.vdst < 255u
                      ? remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst + 1u), state.mode,
                                             HighBankRole::Dst, plan, changed)
                      : std::optional<uint8_t>{};
    auto src0_pair = remap_high_bank_src_pair(static_cast<uint16_t>(src.src0), literal, state.mode,
                                              HighBankRole::Src0, plan, changed);
    auto src1_pair = remap_high_bank_src_pair(static_cast<uint16_t>(256u + src.vsrc1), std::nullopt,
                                              state.mode, HighBankRole::Src1, plan, changed);
    if (!dst_lo || !dst_hi || !src0_pair || !src1_pair)
      return unsupported();
    if (!changed)
      return {};
    std::vector<uint32_t> words;
    append_raw_v_mul_u64_low64(words, *dst_lo, *dst_hi, src0_pair->lo, src0_pair->hi, src1_pair->lo,
                               src1_pair->hi, literal);
    auto wrapped =
        wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores, liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (mnemonic == "v_lshlrev_b64_e32") {
    const auto src = std::bit_cast<gfx1250::Vop2MachineInst>(w0);
    std::vector<HighBankShadowSlot> loads;
    std::vector<HighBankShadowSlot> stores;
    if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), 2, state.mode,
                                      HighBankRole::Dst, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 1, state.mode,
                                     HighBankRole::Src0, plan) ||
        !collect_high_bank_vgpr_spill(loads, static_cast<uint8_t>(src.vsrc1), 2, state.mode,
                                      HighBankRole::Src1, plan))
      return unsupported();
    auto dst_lo = remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst), state.mode,
                                       HighBankRole::Dst, plan, changed);
    auto dst_hi = src.vdst < 255u
                      ? remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst + 1u), state.mode,
                                             HighBankRole::Dst, plan, changed)
                      : std::optional<uint8_t>{};
    auto mapped_src0 = remap_high_bank_src(static_cast<uint16_t>(src.src0), state.mode,
                                           HighBankRole::Src0, plan, changed);
    auto src1_pair = remap_high_bank_src_pair(static_cast<uint16_t>(256u + src.vsrc1), std::nullopt,
                                              state.mode, HighBankRole::Src1, plan, changed);
    if (!dst_lo || !dst_hi || !mapped_src0 || !src1_pair)
      return unsupported();
    if (!changed)
      return {};
    std::vector<uint32_t> words;
    append_raw_vop2(words, static_cast<uint8_t>(src.op), *dst_lo, *mapped_src0,
                    static_cast<uint8_t>(src1_pair->lo - 256u));
    auto wrapped =
        wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores, liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (is_gfx1250_remappable_two_src_vop2(inst) || mnemonic == "v_add_f16_e32" ||
      (((w0 >> 25) & 0x3Fu) == kGfx1250VAddF16E32Opcode && inst.size() == sizeof(uint32_t))) {
    const auto src = std::bit_cast<gfx1250::Vop2MachineInst>(w0);
    const std::optional<uint32_t> literal =
        src.src0 == 255u && inst.size() >= static_cast<int>(2 * sizeof(uint32_t))
            ? read_trailing_literal_u32(text, offset, sizeof(uint32_t))
            : std::nullopt;
    if (src.src0 == 255u && !literal)
      return unsupported();
    std::vector<HighBankShadowSlot> loads;
    std::vector<HighBankShadowSlot> stores;
    if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), 1, state.mode,
                                      HighBankRole::Dst, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 1, state.mode,
                                     HighBankRole::Src0, plan) ||
        !collect_high_bank_vgpr_spill(loads, static_cast<uint8_t>(src.vsrc1), 1, state.mode,
                                      HighBankRole::Src1, plan))
      return unsupported();
    auto mapped_vdst = remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst), state.mode,
                                            HighBankRole::Dst, plan, changed);
    auto mapped_src0 = remap_high_bank_src(static_cast<uint16_t>(src.src0), state.mode,
                                           HighBankRole::Src0, plan, changed);
    auto mapped_vsrc1 = remap_high_bank_vgpr(static_cast<uint8_t>(src.vsrc1), state.mode,
                                             HighBankRole::Src1, plan, changed);
    if (!mapped_vdst || !mapped_src0 || !mapped_vsrc1)
      return unsupported();
    if (!changed)
      return {};
    std::vector<uint32_t> words;
    append_raw_vop2(words, static_cast<uint8_t>(src.op), *mapped_vdst, *mapped_src0, *mapped_vsrc1,
                    literal);
    auto wrapped =
        wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores, liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (starts_with(mnemonic, "v_cmp") && (mnemonic.find("_u64_e32") != std::string_view::npos ||
                                         mnemonic.find("_i64_e32") != std::string_view::npos)) {
    const auto src = std::bit_cast<gfx1250::VopcMachineInst>(w0);
    std::vector<HighBankShadowSlot> loads;
    if (!collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 2, state.mode,
                                     HighBankRole::Src0, plan) ||
        !collect_high_bank_vgpr_spill(loads, static_cast<uint8_t>(src.vsrc1), 2, state.mode,
                                      HighBankRole::Src1, plan))
      return unsupported();
    auto src0_pair = remap_high_bank_src_pair(static_cast<uint16_t>(src.src0), std::nullopt,
                                              state.mode, HighBankRole::Src0, plan, changed);
    auto src1_lo = remap_high_bank_vgpr(static_cast<uint8_t>(src.vsrc1), state.mode,
                                        HighBankRole::Src1, plan, changed);
    auto src1_hi = src.vsrc1 < 255u
                       ? remap_high_bank_vgpr(static_cast<uint8_t>(src.vsrc1 + 1u), state.mode,
                                              HighBankRole::Src1, plan, changed)
                       : std::optional<uint8_t>{};
    if (!src0_pair || !src1_lo || !src1_hi)
      return unsupported();
    if (!changed)
      return {};
    auto wrapped = wrap_high_bank_shadow_private_spills(
        inst, {build_rdna4_vopc(static_cast<uint8_t>(src.op), src0_pair->lo, *src1_lo)}, loads, {},
        liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (is_gfx1250_pk_f32_vop3p(inst)) {
    if (inst.size() != static_cast<int>(sizeof(gfx1250::Vop3pMachineInst)))
      return unsupported();

    gfx1250::Vop3pMachineInst src{};
    std::memcpy(&src, raw, sizeof(src));
    std::vector<HighBankShadowSlot> loads;
    std::vector<HighBankShadowSlot> stores;
    if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), 2, state.mode,
                                      HighBankRole::Dst, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 2, state.mode,
                                     HighBankRole::Src0, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src1), 2, state.mode,
                                     HighBankRole::Src1, plan))
      return unsupported();
    if (src.op == kGfx1250VPkFmaF32Vop3pOpcode &&
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src2), 2, state.mode,
                                     HighBankRole::Src2, plan))
      return unsupported();

    auto mapped_vdst = remap_high_bank_vgpr_run(static_cast<uint8_t>(src.vdst), 2, state.mode,
                                                HighBankRole::Dst, plan, changed);
    auto mapped_src0 = remap_high_bank_src_run(static_cast<uint16_t>(src.src0), 2, state.mode,
                                               HighBankRole::Src0, plan, changed);
    auto mapped_src1 = remap_high_bank_src_run(static_cast<uint16_t>(src.src1), 2, state.mode,
                                               HighBankRole::Src1, plan, changed);
    std::optional<uint16_t> mapped_src2 = static_cast<uint16_t>(src.src2);
    if (src.op == kGfx1250VPkFmaF32Vop3pOpcode) {
      mapped_src2 = remap_high_bank_src_run(static_cast<uint16_t>(src.src2), 2, state.mode,
                                            HighBankRole::Src2, plan, changed);
    }
    if (!mapped_vdst || !mapped_src0 || !mapped_src1 || !mapped_src2)
      return unsupported();
    if (!changed)
      return {};

    gfx1250::Vop3pMachineInst dst = src;
    dst.vdst = *mapped_vdst;
    dst.src0 = *mapped_src0;
    dst.src1 = *mapped_src1;
    dst.src2 = *mapped_src2;

    std::array<uint32_t, 2> words{};
    std::memcpy(words.data(), &dst, sizeof(dst));
    return {HighBankShadowLoweringKind::RemappedGuest,
            {words[0], words[1]},
            {},
            std::move(loads),
            std::move(stores)};
  }

  if (is_gfx1250_remappable_three_src_vop3p(inst)) {
    if (inst.size() < static_cast<int>(sizeof(gfx1250::Vop3pMachineInst)))
      return unsupported();

    gfx1250::Vop3pMachineInst src{};
    std::memcpy(&src, raw, sizeof(src));
    if (src.src0 == 254u || src.src1 == 254u || src.src2 == 254u)
      return unsupported();
    const bool uses_literal = src.src0 == 255u || src.src1 == 255u || src.src2 == 255u;
    if (uses_literal && inst.size() < static_cast<int>(3 * sizeof(uint32_t)))
      return unsupported();
    const std::optional<uint32_t> literal =
        uses_literal ? read_trailing_literal_u32(text, offset, 2 * sizeof(uint32_t)) : std::nullopt;
    if (uses_literal && !literal)
      return unsupported();

    std::vector<HighBankShadowSlot> loads;
    std::vector<HighBankShadowSlot> stores;
    if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), 1, state.mode,
                                      HighBankRole::Dst, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 1, state.mode,
                                     HighBankRole::Src0, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src1), 1, state.mode,
                                     HighBankRole::Src1, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src2), 1, state.mode,
                                     HighBankRole::Src2, plan))
      return unsupported();

    auto mapped_vdst = remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst), state.mode,
                                            HighBankRole::Dst, plan, changed);
    auto mapped_src0 = remap_high_bank_src(static_cast<uint16_t>(src.src0), state.mode,
                                           HighBankRole::Src0, plan, changed);
    auto mapped_src1 = remap_high_bank_src(static_cast<uint16_t>(src.src1), state.mode,
                                           HighBankRole::Src1, plan, changed);
    auto mapped_src2 = remap_high_bank_src(static_cast<uint16_t>(src.src2), state.mode,
                                           HighBankRole::Src2, plan, changed);
    if (!mapped_vdst || !mapped_src0 || !mapped_src1 || !mapped_src2)
      return unsupported();
    if (!changed)
      return {};

    gfx1250::Vop3pMachineInst dst = src;
    dst.vdst = *mapped_vdst;
    dst.src0 = *mapped_src0;
    dst.src1 = *mapped_src1;
    dst.src2 = *mapped_src2;

    std::array<uint32_t, 2> words{};
    std::memcpy(words.data(), &dst, sizeof(dst));
    std::vector<uint32_t> guest_words{words[0], words[1]};
    if (literal)
      guest_words.push_back(*literal);
    return {HighBankShadowLoweringKind::RemappedGuest,
            std::move(guest_words),
            {},
            std::move(loads),
            std::move(stores)};
  }

  if (inst.encoding_id() == kGfx1250Vop3pEncodingId &&
      inst.opcode() == kGfx1250SwmmacF32F16K64Opcode) {
    if (inst.size() < static_cast<int>(sizeof(gfx1250::Vop3pMachineInst)))
      return unsupported();

    gfx1250::Vop3pMachineInst src{};
    std::memcpy(&src, raw, sizeof(src));
    std::vector<HighBankShadowSlot> loads;
    std::vector<HighBankShadowSlot> stores;
    if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), 8, state.mode,
                                      HighBankRole::Dst, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 8, state.mode,
                                     HighBankRole::Src0, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src1), 16, state.mode,
                                     HighBankRole::Src1, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src2), 1, state.mode,
                                     HighBankRole::Src2, plan))
      return unsupported();

    auto mapped_vdst = remap_high_bank_vgpr_run(static_cast<uint8_t>(src.vdst), 8, state.mode,
                                                HighBankRole::Dst, plan, changed);
    auto mapped_src0 = remap_high_bank_src_run(static_cast<uint16_t>(src.src0), 8, state.mode,
                                               HighBankRole::Src0, plan, changed);
    auto mapped_src1 = remap_high_bank_src_run(static_cast<uint16_t>(src.src1), 16, state.mode,
                                               HighBankRole::Src1, plan, changed);
    auto mapped_src2 = remap_high_bank_src_run(static_cast<uint16_t>(src.src2), 1, state.mode,
                                               HighBankRole::Src2, plan, changed);
    if (!mapped_vdst)
      return unsupported();

    std::vector<uint8_t> redirect_avoid;
    add_raw_avoid_vgpr_run(redirect_avoid, *mapped_vdst, 8);
    const auto add_src_avoid_run = [&](std::optional<uint16_t> mapped, uint16_t width) {
      if (!mapped)
        return;
      if (const auto base = raw_vgpr_index(*mapped))
        add_raw_avoid_vgpr_run(redirect_avoid, *base, static_cast<uint8_t>(width));
    };
    add_src_avoid_run(mapped_src0, 8);
    add_src_avoid_run(mapped_src1, 16);
    add_src_avoid_run(mapped_src2, 1);

    std::vector<uint32_t> prefix_words;
    std::vector<uint32_t> suffix_words;
    const auto ensure_contiguous_src_run = [&](std::optional<uint16_t> &mapped, uint16_t original,
                                               uint16_t width, HighBankRole role) -> bool {
      if (mapped && source_run_fits_low_vgprs(*mapped, width))
        return true;
      uint16_t redirected = 0;
      if (!redirect_high_bank_shadow_source_run(original, width, state.mode, role, inst, liveness,
                                                plan, loads, redirect_avoid, prefix_words,
                                                suffix_words, redirected))
        return false;
      mapped = redirected;
      changed = true;
      if (const auto base = raw_vgpr_index(redirected))
        add_raw_avoid_vgpr_run(redirect_avoid, *base, static_cast<uint8_t>(width));
      return true;
    };
    if (!ensure_contiguous_src_run(mapped_src0, static_cast<uint16_t>(src.src0), 8,
                                   HighBankRole::Src0) ||
        !ensure_contiguous_src_run(mapped_src1, static_cast<uint16_t>(src.src1), 16,
                                   HighBankRole::Src1) ||
        !ensure_contiguous_src_run(mapped_src2, static_cast<uint16_t>(src.src2), 1,
                                   HighBankRole::Src2))
      return unsupported();
    if (!changed)
      return {};

    gfx1250::Vop3pMachineInst dst = src;
    dst.vdst = *mapped_vdst;
    dst.src0 = *mapped_src0;
    dst.src1 = *mapped_src1;
    dst.src2 = *mapped_src2;

    std::array<uint32_t, 2> words{};
    std::memcpy(words.data(), &dst, sizeof(dst));
    return {HighBankShadowLoweringKind::RemappedGuest,
            {words[0], words[1]},
            {},
            std::move(loads),
            std::move(stores),
            std::move(prefix_words),
            std::move(suffix_words)};
  }

  if (inst.size() >= static_cast<int>(2 * sizeof(uint32_t)) && (w0 >> 26) == kVop3Encoding) {
    const uint16_t op = static_cast<uint16_t>((w0 >> 16) & 0x3FFu);
    const uint8_t vdst = static_cast<uint8_t>(w0 & 0xFFu);
    const uint32_t w1 = raw[1];
    const uint16_t src0 = static_cast<uint16_t>(w1 & 0x1FFu);
    const uint16_t src1 = static_cast<uint16_t>((w1 >> 9) & 0x1FFu);
    const uint16_t src2 = static_cast<uint16_t>((w1 >> 18) & 0x1FFu);

    if (op == kGfx1250VAddCoU32Vop3SdstOpcode) {
      if (src0 == 254u || src1 == 254u)
        return unsupported();
      const bool uses_literal = src0 == 255u || src1 == 255u;
      if (uses_literal && inst.size() < static_cast<int>(3 * sizeof(uint32_t)))
        return unsupported();
      const std::optional<uint32_t> literal =
          uses_literal ? read_trailing_literal_u32(text, offset, 2 * sizeof(uint32_t))
                       : std::nullopt;
      if (uses_literal && !literal)
        return unsupported();

      std::vector<HighBankShadowSlot> loads;
      std::vector<HighBankShadowSlot> stores;
      if (!collect_high_bank_vgpr_spill(stores, vdst, 1, state.mode, HighBankRole::Dst, plan) ||
          !collect_high_bank_src_spill(loads, src0, 1, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan))
        return unsupported();
      auto mapped_vdst = remap_high_bank_vgpr(vdst, state.mode, HighBankRole::Dst, plan, changed);
      auto mapped_src0 = remap_high_bank_src(src0, state.mode, HighBankRole::Src0, plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      if (!mapped_vdst || !mapped_src0 || !mapped_src1)
        return unsupported();
      if (!changed)
        return {};

      gfx1250::Vop3SdstEncMachineInst dst{};
      std::memcpy(&dst, raw, sizeof(dst));
      dst.vdst = *mapped_vdst;
      dst.src0 = *mapped_src0;
      dst.src1 = *mapped_src1;
      dst.src2 = 0;
      dst.neg &= 0x3u;

      std::array<uint32_t, 2> words{};
      std::memcpy(words.data(), &dst, sizeof(dst));
      std::vector<uint32_t> guest_words{words[0], words[1]};
      if (literal)
        guest_words.push_back(*literal);
      return {HighBankShadowLoweringKind::RemappedGuest,
              std::move(guest_words),
              {},
              std::move(loads),
              std::move(stores)};
    }

    if (op == kGfx1250VCvtPkF16F32Vop3Opcode) {
      if (src0 == 254u || src1 == 254u || src0 == 255u || src1 == 255u)
        return unsupported();
      std::vector<HighBankShadowSlot> loads;
      std::vector<HighBankShadowSlot> stores;
      if (!collect_high_bank_vgpr_spill(stores, vdst, 1, state.mode, HighBankRole::Dst, plan) ||
          !collect_high_bank_src_spill(loads, src0, 1, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan))
        return unsupported();
      auto mapped_vdst = remap_high_bank_vgpr(vdst, state.mode, HighBankRole::Dst, plan, changed);
      auto mapped_src0 = remap_high_bank_src(src0, state.mode, HighBankRole::Src0, plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      if (!mapped_vdst || !mapped_src0 || !mapped_src1)
        return unsupported();
      if (!changed)
        return {};

      gfx1250::Vop3MachineInst dst{};
      std::memcpy(&dst, raw, sizeof(dst));
      dst.vdst = *mapped_vdst;
      dst.src0 = *mapped_src0;
      dst.src1 = *mapped_src1;

      std::array<uint32_t, 2> words{};
      std::memcpy(words.data(), &dst, sizeof(dst));
      return {HighBankShadowLoweringKind::RemappedGuest,
              {words[0], words[1]},
              {},
              std::move(loads),
              std::move(stores)};
    }

    if (op == 594u) {
      std::vector<HighBankShadowSlot> loads;
      std::vector<HighBankShadowSlot> stores;
      if (!collect_high_bank_vgpr_spill(stores, vdst, 2, state.mode, HighBankRole::Dst, plan) ||
          !collect_high_bank_src_spill(loads, src0, 2, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan) ||
          !collect_high_bank_src_spill(loads, src2, 2, state.mode, HighBankRole::Src2, plan))
        return unsupported();
      auto dst_lo = remap_high_bank_vgpr(vdst, state.mode, HighBankRole::Dst, plan, changed);
      auto dst_hi = vdst < 255u ? remap_high_bank_vgpr(static_cast<uint8_t>(vdst + 1u), state.mode,
                                                       HighBankRole::Dst, plan, changed)
                                : std::optional<uint8_t>{};
      auto src0_pair = remap_high_bank_src_pair(src0, std::nullopt, state.mode, HighBankRole::Src0,
                                                plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      auto src2_pair = remap_high_bank_src_pair(src2, std::nullopt, state.mode, HighBankRole::Src2,
                                                plan, changed);
      if (!dst_lo || !dst_hi || !src0_pair || !mapped_src1 || !src2_pair)
        return unsupported();
      if (!changed)
        return {};
      if (*mapped_src1 != scalar_positive_inline_u32(1) &&
          *mapped_src1 != scalar_positive_inline_u32(2))
        return unsupported();
      const uint16_t shift = *mapped_src1 == scalar_positive_inline_u32(1) ? 1u : 2u;

      const auto src0_lo_vgpr = raw_vgpr_index(src0_pair->lo);
      const auto src2_lo_vgpr = raw_vgpr_index(src2_pair->lo);
      const auto src2_hi_vgpr = raw_vgpr_index(src2_pair->hi);
      const bool can_shift_into_dst =
          !(src0_lo_vgpr && *src0_lo_vgpr == *dst_hi) &&
          !(src2_lo_vgpr && raw_overlaps_vdst_pair(*src2_lo_vgpr, *dst_lo)) &&
          !(src2_hi_vgpr && raw_overlaps_vdst_pair(*src2_hi_vgpr, *dst_lo));

      uint16_t shifted_lo = static_cast<uint16_t>(256u + *dst_lo);
      uint16_t shifted_hi = static_cast<uint16_t>(256u + *dst_hi);
      if (!can_shift_into_dst) {
        std::vector<uint8_t> avoid;
        add_raw_avoid_vgpr_run(avoid, *dst_lo, 2);
        add_raw_avoid_source_pair_vgprs(avoid, src0_pair->lo, src0_pair->hi);
        add_raw_avoid_source_pair_vgprs(avoid, src2_pair->lo, src2_pair->hi);
        auto tmp_base = find_raw_free_vgpr_run_avoiding(inst, liveness, 2, avoid);
        if (!tmp_base || *tmp_base > 254u)
          return unsupported();
        shifted_lo = static_cast<uint16_t>(256u + *tmp_base);
        shifted_hi = static_cast<uint16_t>(256u + *tmp_base + 1u);
      }

      const auto carry_opt = liveness.find_free_sgpr_pair(&inst);
      if (!carry_opt || *carry_opt > 105u)
        return unsupported();
      const uint8_t carry = static_cast<uint8_t>(*carry_opt);

      constexpr uint16_t kOpAlignbitB32 = 534;
      constexpr uint16_t kOpLshlrevB32 = 280;
      constexpr uint16_t kOpAddCoCiU32 = 288;
      constexpr uint16_t kOpAddCoU32 = 768;
      std::vector<uint32_t> words;
      words.reserve(9);
      append_raw_vop3(words, kOpAlignbitB32, static_cast<uint8_t>(shifted_hi - 256u), src0_pair->hi,
                      src0_pair->lo,
                      scalar_positive_inline_u32(static_cast<uint16_t>(32u - shift)));
      append_raw_vop3(words, kOpLshlrevB32, static_cast<uint8_t>(shifted_lo - 256u),
                      scalar_positive_inline_u32(shift), src0_pair->lo);
      append_raw_vop3_sdst(words, kOpAddCoU32, *dst_lo, carry, shifted_lo, src2_pair->lo);
      words.push_back(build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));
      append_raw_vop3_sdst(words, kOpAddCoCiU32, *dst_hi, kRdna4NullSgpr, shifted_hi, src2_pair->hi,
                           carry);
      auto wrapped = wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores,
                                                          liveness, plan);
      if (!wrapped)
        return unsupported();
      return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
    }

    if (op == 762u) {
      if (src0 == 255u || src1 == 255u || src2 == 255u)
        return unsupported();
      std::vector<HighBankShadowSlot> loads;
      std::vector<HighBankShadowSlot> stores;
      if (!collect_high_bank_vgpr_spill(stores, vdst, 2, state.mode, HighBankRole::Dst, plan) ||
          !collect_high_bank_src_spill(loads, src0, 1, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan) ||
          !collect_high_bank_src_spill(loads, src2, 2, state.mode, HighBankRole::Src2, plan))
        return unsupported();
      auto dst_lo = remap_high_bank_vgpr(vdst, state.mode, HighBankRole::Dst, plan, changed);
      auto dst_hi = vdst < 255u ? remap_high_bank_vgpr(static_cast<uint8_t>(vdst + 1u), state.mode,
                                                       HighBankRole::Dst, plan, changed)
                                : std::optional<uint8_t>{};
      auto mapped_src0 = remap_high_bank_src(src0, state.mode, HighBankRole::Src0, plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      auto src2_pair = remap_high_bank_src_pair(src2, std::nullopt, state.mode, HighBankRole::Src2,
                                                plan, changed);
      if (!dst_lo || !dst_hi || !mapped_src0 || !mapped_src1 || !src2_pair)
        return unsupported();
      if (!changed)
        return {};
      std::vector<uint32_t> words;
      append_raw_vop3_sdst(words, 766, *dst_lo, kRdna4NullSgpr, *mapped_src0, *mapped_src1,
                           src2_pair->lo);
      auto wrapped = wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores,
                                                          liveness, plan);
      if (!wrapped)
        return unsupported();
      return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
    }

    if (op == 582u || op == 598u) {
      if (src0 == 254u || src1 == 254u || src2 == 254u)
        return unsupported();
      const bool uses_literal = src0 == 255u || src1 == 255u || src2 == 255u;
      if (uses_literal && inst.size() < static_cast<int>(3 * sizeof(uint32_t)))
        return unsupported();
      const std::optional<uint32_t> literal =
          uses_literal ? read_trailing_literal_u32(text, offset, 2 * sizeof(uint32_t))
                       : std::nullopt;
      if (uses_literal && !literal)
        return unsupported();

      std::vector<HighBankShadowSlot> loads;
      std::vector<HighBankShadowSlot> stores;
      if (!collect_high_bank_vgpr_spill(stores, vdst, 1, state.mode, HighBankRole::Dst, plan) ||
          !collect_high_bank_src_spill(loads, src0, 1, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan) ||
          !collect_high_bank_src_spill(loads, src2, 1, state.mode, HighBankRole::Src2, plan))
        return unsupported();
      auto mapped_vdst = remap_high_bank_vgpr(vdst, state.mode, HighBankRole::Dst, plan, changed);
      auto mapped_src0 = remap_high_bank_src(src0, state.mode, HighBankRole::Src0, plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      auto mapped_src2 = remap_high_bank_src(src2, state.mode, HighBankRole::Src2, plan, changed);
      if (!mapped_vdst || !mapped_src0 || !mapped_src1 || !mapped_src2)
        return unsupported();
      if (!changed)
        return {};

      std::vector<uint32_t> prefix_words;
      std::vector<uint32_t> suffix_words;
      uint8_t shift_dst = *mapped_vdst;
      if (const auto src2_vgpr = raw_vgpr_index(*mapped_src2);
          src2_vgpr && *src2_vgpr == *mapped_vdst) {
        std::vector<uint8_t> avoid{*mapped_vdst};
        add_raw_avoid_src_vgpr(avoid, *mapped_src0);
        add_raw_avoid_src_vgpr(avoid, *mapped_src1);
        add_raw_avoid_src_vgpr(avoid, *mapped_src2);
        const auto tmp_opt = find_raw_free_vgpr_run_avoiding(inst, liveness, 1, avoid);
        if (tmp_opt && *tmp_opt <= 255u) {
          shift_dst = static_cast<uint8_t>(*tmp_opt);
        } else {
          const auto private_spill_base = liveness.private_spill_base();
          if (!private_spill_base ||
              liveness.private_spill_bytes() < kGfx1250PrivateBorrowScratchBytes)
            return unsupported();
          const auto borrowed = find_raw_borrowable_low_vgpr_run(1, 1, avoid);
          if (!borrowed)
            return unsupported();
          shift_dst = *borrowed;
          prefix_words.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
          prefix_words.push_back(pack_sopp(kSoppWaitDscnt, 0));
          append_scratch_store_b32(prefix_words, shift_dst, *private_spill_base);
          prefix_words.push_back(pack_sopp(kSoppWaitStorecnt, 0));

          suffix_words.push_back(build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));
          append_scratch_load_b32(suffix_words, shift_dst, *private_spill_base);
          suffix_words.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
        }
      }

      constexpr uint16_t kOpLshlrevB32 = 280;
      constexpr uint16_t kOpAddNcU32 = 293;
      constexpr uint8_t kOpOrB32 = 28;
      std::vector<uint32_t> words = std::move(prefix_words);
      words.reserve((uses_literal ? 1u : 0u) + 5u);
      append_raw_vop3(words, kOpLshlrevB32, shift_dst, *mapped_src1, *mapped_src0, 0, literal);
      words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
      if (op == 582u) {
        append_raw_vop3(words, kOpAddNcU32, *mapped_vdst, *mapped_src2,
                        static_cast<uint16_t>(256u + shift_dst), 0, literal);
      } else {
        append_raw_vop2(words, kOpOrB32, *mapped_vdst, *mapped_src2, shift_dst, literal);
      }
      words.insert(words.end(), suffix_words.begin(), suffix_words.end());
      auto wrapped = wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores,
                                                          liveness, plan);
      if (!wrapped)
        return unsupported();
      return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
    }

    if (op == 597u) {
      if (src0 == 255u || src1 == 255u || src2 == 255u)
        return unsupported();
      std::vector<HighBankShadowSlot> loads;
      std::vector<HighBankShadowSlot> stores;
      if (!collect_high_bank_vgpr_spill(stores, vdst, 1, state.mode, HighBankRole::Dst, plan) ||
          !collect_high_bank_src_spill(loads, src0, 1, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan) ||
          !collect_high_bank_src_spill(loads, src2, 1, state.mode, HighBankRole::Src2, plan))
        return unsupported();
      auto mapped_vdst = remap_high_bank_vgpr(vdst, state.mode, HighBankRole::Dst, plan, changed);
      auto mapped_src0 = remap_high_bank_src(src0, state.mode, HighBankRole::Src0, plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      auto mapped_src2 = remap_high_bank_src(src2, state.mode, HighBankRole::Src2, plan, changed);
      if (!mapped_vdst || !mapped_src0 || !mapped_src1 || !mapped_src2)
        return unsupported();
      if (!changed)
        return {};
      std::vector<uint32_t> words;
      append_raw_vop3(words, 597, *mapped_vdst, *mapped_src0, *mapped_src1, *mapped_src2);
      auto wrapped = wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores,
                                                          liveness, plan);
      if (!wrapped)
        return unsupported();
      return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
    }

    if (op == 599u || op == 600u) {
      const bool uses_literal = src0 == 255u || src1 == 255u || src2 == 255u;
      if (uses_literal && inst.size() < static_cast<int>(3 * sizeof(uint32_t)))
        return unsupported();
      const std::optional<uint32_t> literal =
          uses_literal ? read_trailing_literal_u32(text, offset, 2 * sizeof(uint32_t))
                       : std::nullopt;
      if (uses_literal && !literal)
        return unsupported();
      std::vector<HighBankShadowSlot> loads;
      std::vector<HighBankShadowSlot> stores;
      if (!collect_high_bank_vgpr_spill(stores, vdst, 1, state.mode, HighBankRole::Dst, plan) ||
          !collect_high_bank_src_spill(loads, src0, 1, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan) ||
          !collect_high_bank_src_spill(loads, src2, 1, state.mode, HighBankRole::Src2, plan))
        return unsupported();
      auto mapped_vdst = remap_high_bank_vgpr(vdst, state.mode, HighBankRole::Dst, plan, changed);
      auto mapped_src0 = remap_high_bank_src(src0, state.mode, HighBankRole::Src0, plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      auto mapped_src2 = remap_high_bank_src(src2, state.mode, HighBankRole::Src2, plan, changed);
      if (!mapped_vdst || !mapped_src0 || !mapped_src1 || !mapped_src2)
        return unsupported();
      if (!changed)
        return {};
      std::vector<uint32_t> words;
      append_raw_vop3(words, op, *mapped_vdst, *mapped_src0, *mapped_src1, *mapped_src2, literal);
      auto wrapped = wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores,
                                                          liveness, plan);
      if (!wrapped)
        return unsupported();
      return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
    }

    if (op == 812u || op == 813u) {
      const bool uses_literal = src0 == 255u || src1 == 255u;
      if (uses_literal && inst.size() < static_cast<int>(3 * sizeof(uint32_t)))
        return unsupported();
      const std::optional<uint32_t> literal =
          uses_literal ? read_trailing_literal_u32(text, offset, 2 * sizeof(uint32_t))
                       : std::nullopt;
      if (uses_literal && !literal)
        return unsupported();
      std::vector<HighBankShadowSlot> loads;
      std::vector<HighBankShadowSlot> stores;
      if (!collect_high_bank_vgpr_spill(stores, vdst, 1, state.mode, HighBankRole::Dst, plan) ||
          !collect_high_bank_src_spill(loads, src0, 1, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan))
        return unsupported();
      auto mapped_vdst = remap_high_bank_vgpr(vdst, state.mode, HighBankRole::Dst, plan, changed);
      auto mapped_src0 = remap_high_bank_src(src0, state.mode, HighBankRole::Src0, plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      if (!mapped_vdst || !mapped_src0 || !mapped_src1)
        return unsupported();
      if (!changed)
        return {};
      std::vector<uint32_t> words;
      append_raw_vop3(words, op, *mapped_vdst, *mapped_src0, *mapped_src1, 0, literal);
      auto wrapped = wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores,
                                                          liveness, plan);
      if (!wrapped)
        return unsupported();
      return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
    }

    if (is_gfx1250_remappable_vop3_with_scalar_src2(inst)) {
      if (src0 == 254u || src1 == 254u || src2 == 254u)
        return unsupported();
      const bool uses_literal = src0 == 255u || src1 == 255u || src2 == 255u;
      if (uses_literal && inst.size() < static_cast<int>(3 * sizeof(uint32_t)))
        return unsupported();
      const std::optional<uint32_t> literal =
          uses_literal ? read_trailing_literal_u32(text, offset, 2 * sizeof(uint32_t))
                       : std::nullopt;
      if (uses_literal && !literal)
        return unsupported();

      std::vector<HighBankShadowSlot> loads;
      std::vector<HighBankShadowSlot> stores;
      if (!collect_high_bank_vgpr_spill(stores, vdst, 1, state.mode, HighBankRole::Dst, plan) ||
          !collect_high_bank_src_spill(loads, src0, 1, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan))
        return unsupported();
      auto mapped_vdst = remap_high_bank_vgpr(vdst, state.mode, HighBankRole::Dst, plan, changed);
      auto mapped_src0 = remap_high_bank_src(src0, state.mode, HighBankRole::Src0, plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      if (!mapped_vdst || !mapped_src0 || !mapped_src1)
        return unsupported();
      if (!changed)
        return {};

      gfx1250::Vop3MachineInst dst{};
      std::memcpy(&dst, raw, sizeof(dst));
      dst.vdst = *mapped_vdst;
      dst.src0 = *mapped_src0;
      dst.src1 = *mapped_src1;

      std::array<uint32_t, 2> words{};
      std::memcpy(words.data(), &dst, sizeof(dst));
      std::vector<uint32_t> guest_words{words[0], words[1]};
      if (literal)
        guest_words.push_back(*literal);
      return {HighBankShadowLoweringKind::RemappedGuest,
              std::move(guest_words),
              {},
              std::move(loads),
              std::move(stores)};
    }

    if (is_gfx1250_remappable_three_src_vop3(inst)) {
      if (src0 == 254u || src1 == 254u || src2 == 254u)
        return unsupported();
      const bool uses_literal = src0 == 255u || src1 == 255u || src2 == 255u;
      if (uses_literal && inst.size() < static_cast<int>(3 * sizeof(uint32_t)))
        return unsupported();
      const std::optional<uint32_t> literal =
          uses_literal ? read_trailing_literal_u32(text, offset, 2 * sizeof(uint32_t))
                       : std::nullopt;
      if (uses_literal && !literal)
        return unsupported();

      std::vector<HighBankShadowSlot> loads;
      std::vector<HighBankShadowSlot> stores;
      if (!collect_high_bank_vgpr_spill(stores, vdst, 1, state.mode, HighBankRole::Dst, plan) ||
          !collect_high_bank_src_spill(loads, src0, 1, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan) ||
          !collect_high_bank_src_spill(loads, src2, 1, state.mode, HighBankRole::Src2, plan))
        return unsupported();
      auto mapped_vdst = remap_high_bank_vgpr(vdst, state.mode, HighBankRole::Dst, plan, changed);
      auto mapped_src0 = remap_high_bank_src(src0, state.mode, HighBankRole::Src0, plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      auto mapped_src2 = remap_high_bank_src(src2, state.mode, HighBankRole::Src2, plan, changed);
      if (!mapped_vdst || !mapped_src0 || !mapped_src1 || !mapped_src2)
        return unsupported();
      if (!changed)
        return {};

      gfx1250::Vop3MachineInst dst{};
      std::memcpy(&dst, raw, sizeof(dst));
      dst.vdst = *mapped_vdst;
      dst.src0 = *mapped_src0;
      dst.src1 = *mapped_src1;
      dst.src2 = *mapped_src2;

      std::array<uint32_t, 2> words{};
      std::memcpy(words.data(), &dst, sizeof(dst));
      std::vector<uint32_t> guest_words{words[0], words[1]};
      if (literal)
        guest_words.push_back(*literal);
      return {HighBankShadowLoweringKind::RemappedGuest,
              std::move(guest_words),
              {},
              std::move(loads),
              std::move(stores)};
    }

    if (is_gfx1250_remappable_two_src_vop3(inst, op)) {
      if (src0 == 254u || src1 == 254u)
        return unsupported();
      const bool uses_literal = src0 == 255u || src1 == 255u;
      if (uses_literal && inst.size() < static_cast<int>(3 * sizeof(uint32_t)))
        return unsupported();
      const std::optional<uint32_t> literal =
          uses_literal ? read_trailing_literal_u32(text, offset, 2 * sizeof(uint32_t))
                       : std::nullopt;
      if (uses_literal && !literal)
        return unsupported();

      std::vector<HighBankShadowSlot> loads;
      std::vector<HighBankShadowSlot> stores;
      if (!collect_high_bank_vgpr_spill(stores, vdst, 1, state.mode, HighBankRole::Dst, plan) ||
          !collect_high_bank_src_spill(loads, src0, 1, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan))
        return unsupported();
      auto mapped_vdst = remap_high_bank_vgpr(vdst, state.mode, HighBankRole::Dst, plan, changed);
      auto mapped_src0 = remap_high_bank_src(src0, state.mode, HighBankRole::Src0, plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      if (!mapped_vdst || !mapped_src0 || !mapped_src1)
        return unsupported();
      if (!changed)
        return {};

      gfx1250::Vop3MachineInst dst{};
      std::memcpy(&dst, raw, sizeof(dst));
      dst.vdst = *mapped_vdst;
      dst.src0 = *mapped_src0;
      dst.src1 = *mapped_src1;
      dst.src2 = 0;
      dst.abs &= 0x3u;
      dst.opsel &= 0xBu;
      dst.neg &= 0x3u;

      std::array<uint32_t, 2> words{};
      std::memcpy(words.data(), &dst, sizeof(dst));
      std::vector<uint32_t> guest_words{words[0], words[1]};
      if (literal)
        guest_words.push_back(*literal);
      return {HighBankShadowLoweringKind::RemappedGuest,
              std::move(guest_words),
              {},
              std::move(loads),
              std::move(stores)};
    }

    if (is_gfx1250_remappable_two_src_vop3_scalar_dst(inst, op)) {
      if (src0 == 254u || src1 == 254u)
        return unsupported();
      const bool uses_literal = src0 == 255u || src1 == 255u;
      if (uses_literal && inst.size() < static_cast<int>(3 * sizeof(uint32_t)))
        return unsupported();
      const std::optional<uint32_t> literal =
          uses_literal ? read_trailing_literal_u32(text, offset, 2 * sizeof(uint32_t))
                       : std::nullopt;
      if (uses_literal && !literal)
        return unsupported();

      std::vector<HighBankShadowSlot> loads;
      if (!collect_high_bank_src_spill(loads, src0, 1, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan))
        return unsupported();
      auto mapped_src0 = remap_high_bank_src(src0, state.mode, HighBankRole::Src0, plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      if (!mapped_src0 || !mapped_src1)
        return unsupported();
      if (!changed)
        return {};

      gfx1250::Vop3MachineInst dst{};
      std::memcpy(&dst, raw, sizeof(dst));
      dst.src0 = *mapped_src0;
      dst.src1 = *mapped_src1;
      dst.src2 = 0;
      dst.abs &= 0x3u;
      dst.opsel &= 0xBu;
      dst.neg &= 0x3u;

      std::array<uint32_t, 2> words{};
      std::memcpy(words.data(), &dst, sizeof(dst));
      std::vector<uint32_t> guest_words{words[0], words[1]};
      if (literal)
        guest_words.push_back(*literal);
      return {HighBankShadowLoweringKind::RemappedGuest,
              std::move(guest_words),
              {},
              std::move(loads),
              {}};
    }
  }

  if (starts_with(mnemonic, "v_"))
    return unsupported();
  return {};
}

void append_raw_v_mul_u64_low64(std::vector<uint32_t> &words, uint8_t out_lo, uint8_t out_hi,
                                uint16_t src0_lo, uint16_t src0_hi, uint16_t src1_lo,
                                uint16_t src1_hi, std::optional<uint32_t> literal) {
  constexpr uint16_t kOpMulLoU32 = 812;
  constexpr uint16_t kOpMulHiU32 = 813;
  constexpr uint16_t kOpAddNcU32 = 293;
  constexpr uint16_t kVgprSrcBase = 256;

  append_raw_vop3(words, kOpMulHiU32, out_hi, src0_lo, src1_lo, 0, literal);
  append_raw_vop3(words, kOpMulLoU32, out_lo, src0_hi, src1_lo, 0, literal);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_raw_vop3(words, kOpAddNcU32, out_hi, static_cast<uint16_t>(kVgprSrcBase + out_hi),
                  static_cast<uint16_t>(kVgprSrcBase + out_lo));
  append_raw_vop3(words, kOpMulLoU32, out_lo, src0_lo, src1_hi, 0, literal);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_raw_vop3(words, kOpAddNcU32, out_hi, static_cast<uint16_t>(kVgprSrcBase + out_hi),
                  static_cast<uint16_t>(kVgprSrcBase + out_lo));
  append_raw_vop3(words, kOpMulLoU32, out_lo, src0_lo, src1_lo, 0, literal);
}

[[nodiscard]] bool append_raw_v_mul_u64_low_scratch_replacement(
    std::vector<uint32_t> &words, uint8_t vdst, uint16_t src0, uint16_t src1,
    std::optional<uint32_t> literal, const Instruction &inst, const LivenessAnalysis &liveness) {
  const auto src0_hi = raw_pair_hi_src_with_literal(src0, literal);
  const auto src1_hi = raw_pair_hi_src_with_literal(src1, literal);
  if (!src0_hi || !src1_hi)
    return false;
  if (src0 > 511u || src1 > 511u || *src0_hi > 511u || *src1_hi > 511u)
    return false;
  if (!raw_source_pair_reads_vdst_pair(vdst, src0, *src0_hi) &&
      !raw_source_pair_reads_vdst_pair(vdst, src1, *src1_hi))
    return false;

  const auto tmp_opt =
      find_raw_v_mul_u64_low_scratch(vdst, src0, *src0_hi, src1, *src1_hi, inst, liveness);
  if (!tmp_opt)
    return false;

  constexpr uint16_t kVgprSrcBase = 256;
  constexpr uint8_t kOpMovB32 = 1;
  const uint8_t tmp = static_cast<uint8_t>(*tmp_opt);
  words.reserve(literal ? 19 : 15);
  append_raw_v_mul_u64_low64(words, tmp, static_cast<uint8_t>(tmp + 1u), src0, *src0_hi, src1,
                             *src1_hi, literal);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_raw_vop1(words, kOpMovB32, vdst, static_cast<uint16_t>(kVgprSrcBase + tmp));
  append_raw_vop1(words, kOpMovB32, static_cast<uint8_t>(vdst + 1u),
                  static_cast<uint16_t>(kVgprSrcBase + tmp + 1u));
  return true;
}

[[nodiscard]] bool append_raw_v_mul_u64_high_scratch_replacement(
    std::vector<uint32_t> &words, uint8_t vdst, uint16_t src0, uint16_t src1,
    std::optional<uint32_t> literal, const Instruction &inst, const LivenessAnalysis &liveness) {
  const auto src0_hi = raw_pair_hi_src_with_literal(src0, literal);
  const auto src1_hi = raw_pair_hi_src_with_literal(src1, literal);
  if (!src0_hi || !src1_hi)
    return false;
  if (src0 > 511u || src1 > 511u || *src0_hi > 511u || *src1_hi > 511u)
    return false;
  if (!raw_source_pair_reads_vdst_pair(vdst, src0, *src0_hi) &&
      !raw_source_pair_reads_vdst_pair(vdst, src1, *src1_hi))
    return false;

  const auto scratch_base_opt = liveness.high_vgpr_scratch_base();
  if (!scratch_base_opt || *scratch_base_opt > 254)
    return false;
  const auto mode_save_opt = liveness.find_free_sgpr(&inst);
  if (!mode_save_opt || *mode_save_opt > 105)
    return false;

  constexpr uint16_t kOpMulLoU32 = 812;
  constexpr uint16_t kOpMulHiU32 = 813;
  constexpr uint16_t kOpAddNcU32 = 293;
  constexpr uint16_t kVgprSrcBase = 256;
  constexpr uint8_t kModeSrc0High = 0x01;
  constexpr uint8_t kModeScratchAdd = 0x45;
  constexpr uint8_t kModeDstHigh = 0x40;
  constexpr uint8_t kOpMovB32 = 1;

  const uint8_t tmp_lo = static_cast<uint8_t>(*scratch_base_opt);
  const uint8_t tmp_hi = static_cast<uint8_t>(tmp_lo + 1u);
  const uint8_t mode_save = static_cast<uint8_t>(*mode_save_opt);

  words.reserve(literal ? 32 : 28);
  append_raw_s_get_vgpr_msb_mode(words, mode_save);

  append_raw_s_set_vgpr_msb_mode(words, kModeDstHigh);
  append_raw_vop3(words, kOpMulHiU32, tmp_hi, src0, src1, 0, literal);
  append_raw_vop3(words, kOpMulLoU32, tmp_lo, *src0_hi, src1, 0, literal);

  append_raw_s_set_vgpr_msb_mode(words, kModeScratchAdd);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_raw_vop3(words, kOpAddNcU32, tmp_hi, static_cast<uint16_t>(kVgprSrcBase + tmp_hi),
                  static_cast<uint16_t>(kVgprSrcBase + tmp_lo));

  append_raw_s_set_vgpr_msb_mode(words, kModeDstHigh);
  append_raw_vop3(words, kOpMulLoU32, tmp_lo, src0, *src1_hi, 0, literal);

  append_raw_s_set_vgpr_msb_mode(words, kModeScratchAdd);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_raw_vop3(words, kOpAddNcU32, tmp_hi, static_cast<uint16_t>(kVgprSrcBase + tmp_hi),
                  static_cast<uint16_t>(kVgprSrcBase + tmp_lo));

  append_raw_s_set_vgpr_msb_mode(words, kModeDstHigh);
  append_raw_vop3(words, kOpMulLoU32, tmp_lo, src0, src1, 0, literal);

  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_raw_s_set_vgpr_msb_mode(words, kModeSrc0High);
  append_raw_vop1(words, kOpMovB32, vdst, static_cast<uint16_t>(kVgprSrcBase + tmp_lo));
  append_raw_vop1(words, kOpMovB32, static_cast<uint8_t>(vdst + 1u),
                  static_cast<uint16_t>(kVgprSrcBase + tmp_hi));
  append_raw_s_set_vgpr_msb_mode_from_sgpr(words, mode_save);
  return true;
}

[[nodiscard]] bool append_raw_v_mul_u64_replacement(std::vector<uint32_t> &words, uint8_t vdst,
                                                    uint16_t src0, uint16_t src1,
                                                    std::optional<uint32_t> literal) {
  const auto src0_hi = raw_pair_hi_src_with_literal(src0, literal);
  const auto src1_hi = raw_pair_hi_src_with_literal(src1, literal);
  if (!src0_hi || !src1_hi)
    return false;
  if (src0 > 511u || src1 > 511u || *src0_hi > 511u || *src1_hi > 511u)
    return false;
  if (raw_source_pair_reads_vdst_pair(vdst, src0, *src0_hi) ||
      raw_source_pair_reads_vdst_pair(vdst, src1, *src1_hi))
    return false;

  words.reserve(literal ? 15 : 11);
  append_raw_v_mul_u64_low64(words, vdst, static_cast<uint8_t>(vdst + 1u), src0, *src0_hi, src1,
                             *src1_hi, literal);
  return true;
}

[[nodiscard]] std::vector<uint32_t>
lower_raw_gfx1250_v_mul_u64_vop3(const uint32_t *raw, uint32_t inst_size,
                                 uint32_t *source_size = nullptr) {
  if (!raw || inst_size < 2 * sizeof(uint32_t))
    return {};

  const uint32_t w0 = raw[0];
  if ((w0 >> 26) != kVop3Encoding || ((w0 >> 16) & 0x3FFu) != 0)
    return {};

  const uint8_t vdst = static_cast<uint8_t>(w0 & 0xFFu);
  if (vdst > 254)
    return {};

  const uint32_t w1 = raw[1];
  const uint16_t src0 = static_cast<uint16_t>(w1 & 0x1FFu);
  const uint16_t src1 = static_cast<uint16_t>((w1 >> 9) & 0x1FFu);
  if (src0 == 254 || src1 == 254)
    return {};
  const bool uses_literal = src0 == 255 || src1 == 255;
  if (uses_literal && inst_size < 3 * sizeof(uint32_t))
    return {};
  if (source_size)
    *source_size = uses_literal ? 3 * sizeof(uint32_t) : 2 * sizeof(uint32_t);

  std::vector<uint32_t> words;
  if (!append_raw_v_mul_u64_replacement(
          words, vdst, src0, src1, uses_literal ? std::optional<uint32_t>(raw[2]) : std::nullopt))
    return {};
  return words;
}

[[nodiscard]] std::vector<uint32_t>
lower_raw_gfx1250_v_mul_u64_vop3(const uint32_t *raw, uint32_t inst_size, const Instruction &inst,
                                 const LivenessAnalysis &liveness,
                                 uint32_t *source_size = nullptr) {
  if (!raw || inst_size < 2 * sizeof(uint32_t))
    return {};

  const uint32_t w0 = raw[0];
  if ((w0 >> 26) != kVop3Encoding || ((w0 >> 16) & 0x3FFu) != 0)
    return {};

  const uint8_t vdst = static_cast<uint8_t>(w0 & 0xFFu);
  if (vdst > 254)
    return {};

  const uint32_t w1 = raw[1];
  const uint16_t src0 = static_cast<uint16_t>(w1 & 0x1FFu);
  const uint16_t src1 = static_cast<uint16_t>((w1 >> 9) & 0x1FFu);
  if (src0 == 254 || src1 == 254)
    return {};
  const bool uses_literal = src0 == 255 || src1 == 255;
  if (uses_literal && inst_size < 3 * sizeof(uint32_t))
    return {};
  if (source_size)
    *source_size = uses_literal ? 3 * sizeof(uint32_t) : 2 * sizeof(uint32_t);

  const std::optional<uint32_t> literal =
      uses_literal ? std::optional<uint32_t>(raw[2]) : std::nullopt;
  std::vector<uint32_t> words;
  if (append_raw_v_mul_u64_replacement(words, vdst, src0, src1, literal))
    return words;
  words.clear();
  if (append_raw_v_mul_u64_low_scratch_replacement(words, vdst, src0, src1, literal, inst,
                                                   liveness))
    return words;
  words.clear();
  if (append_raw_v_mul_u64_high_scratch_replacement(words, vdst, src0, src1, literal, inst,
                                                    liveness))
    return words;
  return {};
}

[[nodiscard]] std::vector<uint32_t>
lower_raw_gfx1250_v_mul_u64_vop3(std::span<const uint8_t> text, uint64_t offset,
                                 const Instruction &inst, const LivenessAnalysis &liveness,
                                 uint32_t &source_size) {
  if (offset + 2 * sizeof(uint32_t) > text.size())
    return {};

  uint32_t raw[3] = {};
  std::memcpy(raw, text.data() + offset, 2 * sizeof(uint32_t));
  const uint16_t src0 = static_cast<uint16_t>(raw[1] & 0x1FFu);
  const uint16_t src1 = static_cast<uint16_t>((raw[1] >> 9) & 0x1FFu);
  const uint32_t available_size =
      (src0 == 255 || src1 == 255) && offset + 3 * sizeof(uint32_t) <= text.size()
          ? 3 * sizeof(uint32_t)
          : 2 * sizeof(uint32_t);
  if (available_size == 3 * sizeof(uint32_t))
    std::memcpy(raw + 2, text.data() + offset + 2 * sizeof(uint32_t), sizeof(uint32_t));

  uint32_t consumed_size = 0;
  auto words =
      lower_raw_gfx1250_v_mul_u64_vop3(raw, available_size, inst, liveness, &consumed_size);
  if (!words.empty())
    source_size = consumed_size;
  return words;
}

[[nodiscard]] std::vector<uint32_t>
lower_raw_gfx1250_v_mul_u64_e32(const uint32_t *raw, uint32_t inst_size, const Instruction &inst,
                                const LivenessAnalysis &liveness, uint32_t *source_size = nullptr) {
  if (!raw || inst_size < sizeof(uint32_t))
    return {};

  const uint32_t w0 = raw[0];
  if ((w0 >> 26) == kVop3Encoding)
    return {};
  if (((w0 >> 25) & 0x3Fu) != kGfx1250VMulU64E32Opcode)
    return {};

  const uint16_t src0 = static_cast<uint16_t>(w0 & 0x1FFu);
  if (src0 == 254)
    return {};
  const bool uses_literal = src0 == 255;
  if (uses_literal && inst_size < 2 * sizeof(uint32_t))
    return {};
  if (source_size)
    *source_size = uses_literal ? 2 * sizeof(uint32_t) : sizeof(uint32_t);

  const uint8_t vsrc1 = static_cast<uint8_t>((w0 >> 9) & 0xFFu);
  const uint8_t vdst = static_cast<uint8_t>((w0 >> 17) & 0xFFu);
  if (vdst > 254)
    return {};
  const uint16_t src1 = static_cast<uint16_t>(256u + vsrc1);
  const std::optional<uint32_t> literal =
      uses_literal ? std::optional<uint32_t>(raw[1]) : std::nullopt;

  std::vector<uint32_t> words;
  if (append_raw_v_mul_u64_replacement(words, vdst, src0, src1, literal))
    return words;
  words.clear();
  if (append_raw_v_mul_u64_low_scratch_replacement(words, vdst, src0, src1, literal, inst,
                                                   liveness))
    return words;
  words.clear();
  if (append_raw_v_mul_u64_high_scratch_replacement(words, vdst, src0, src1, literal, inst,
                                                    liveness))
    return words;
  return {};
}

[[nodiscard]] std::vector<uint32_t>
lower_raw_gfx1250_v_mul_u64_e32(std::span<const uint8_t> text, uint64_t offset,
                                const Instruction &inst, const LivenessAnalysis &liveness,
                                uint32_t &source_size) {
  if (offset + sizeof(uint32_t) > text.size())
    return {};

  uint32_t raw[2] = {};
  std::memcpy(raw, text.data() + offset, sizeof(uint32_t));
  const uint16_t src0 = static_cast<uint16_t>(raw[0] & 0x1FFu);
  const uint32_t available_size = src0 == 255 && offset + 2 * sizeof(uint32_t) <= text.size()
                                      ? 2 * sizeof(uint32_t)
                                      : sizeof(uint32_t);
  if (available_size == 2 * sizeof(uint32_t))
    std::memcpy(raw + 1, text.data() + offset + sizeof(uint32_t), sizeof(uint32_t));

  uint32_t consumed_size = 0;
  auto words = lower_raw_gfx1250_v_mul_u64_e32(raw, available_size, inst, liveness, &consumed_size);
  if (!words.empty())
    source_size = consumed_size;
  return words;
}

void grow_required_vgpr_count_for_raw_gfx1250_v_mul_u64(const Instruction &inst,
                                                        uint32_t &minimum_vgprs) {
  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
    return;

  const uint32_t w0 = raw[0];
  const uint8_t vdst = static_cast<uint8_t>(w0 & 0xFFu);
  if ((w0 >> 26) == kVop3Encoding && ((w0 >> 16) & 0x3FFu) == 0u &&
      inst.size() >= static_cast<int>(2 * sizeof(uint32_t))) {
    if (vdst <= 254u)
      minimum_vgprs = std::max(minimum_vgprs, static_cast<uint32_t>(vdst + 2u));
    const uint32_t w1 = raw[1];
    const uint16_t src0 = static_cast<uint16_t>(w1 & 0x1FFu);
    const uint16_t src1 = static_cast<uint16_t>((w1 >> 9) & 0x1FFu);
    const bool uses_literal = src0 == 255 || src1 == 255;
    const std::optional<uint32_t> literal =
        uses_literal && inst.size() >= static_cast<int>(3 * sizeof(uint32_t))
            ? std::optional<uint32_t>(raw[2])
            : std::nullopt;
    grow_required_vgpr_count_for_src(minimum_vgprs, src0);
    grow_required_vgpr_count_for_src(minimum_vgprs, src1);
    if (auto src0_hi = raw_pair_hi_src_with_literal(src0, literal))
      grow_required_vgpr_count_for_src(minimum_vgprs, *src0_hi);
    if (auto src1_hi = raw_pair_hi_src_with_literal(src1, literal))
      grow_required_vgpr_count_for_src(minimum_vgprs, *src1_hi);
    return;
  }

  if ((w0 >> 26) == kVop3Encoding)
    return;
  if (((w0 >> 25) & 0x3Fu) != kGfx1250VMulU64E32Opcode)
    return;

  const uint8_t e32_vdst = static_cast<uint8_t>((w0 >> 17) & 0xFFu);
  if (e32_vdst <= 254u)
    minimum_vgprs = std::max(minimum_vgprs, static_cast<uint32_t>(e32_vdst + 2u));
  const uint16_t src0 = static_cast<uint16_t>(w0 & 0x1FFu);
  const uint16_t src1 = static_cast<uint16_t>(256u + ((w0 >> 9) & 0xFFu));
  const std::optional<uint32_t> literal =
      src0 == 255 && inst.size() >= static_cast<int>(2 * sizeof(uint32_t))
          ? std::optional<uint32_t>(raw[1])
          : std::nullopt;
  grow_required_vgpr_count_for_src(minimum_vgprs, src0);
  grow_required_vgpr_count_for_src(minimum_vgprs, src1);
  if (auto src0_hi = raw_pair_hi_src_with_literal(src0, literal))
    grow_required_vgpr_count_for_src(minimum_vgprs, *src0_hi);
  if (auto src1_hi = raw_pair_hi_src_with_literal(src1, literal))
    grow_required_vgpr_count_for_src(minimum_vgprs, *src1_hi);
}

[[nodiscard]] bool
gfx1250_v_mul_u64_needs_high_bank_scratch(const Instruction &inst,
                                          const LivenessAnalysis *liveness = nullptr) {
  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
    return false;

  const uint32_t w0 = raw[0];
  if ((w0 >> 26) == kVop3Encoding) {
    if (((w0 >> 16) & 0x3FFu) != 0u || inst.size() < static_cast<int>(2 * sizeof(uint32_t)))
      return false;

    const uint8_t vdst = static_cast<uint8_t>(w0 & 0xFFu);
    if (vdst > 254u)
      return false;

    const uint32_t w1 = raw[1];
    const uint16_t src0 = static_cast<uint16_t>(w1 & 0x1FFu);
    const uint16_t src1 = static_cast<uint16_t>((w1 >> 9) & 0x1FFu);
    const bool uses_literal = src0 == 255 || src1 == 255;
    const std::optional<uint32_t> literal =
        uses_literal && inst.size() >= static_cast<int>(3 * sizeof(uint32_t))
            ? std::optional<uint32_t>(raw[2])
            : std::nullopt;
    const auto src0_hi = raw_pair_hi_src_with_literal(src0, literal);
    const auto src1_hi = raw_pair_hi_src_with_literal(src1, literal);
    if (!src0_hi || !src1_hi)
      return false;
    const bool overlaps = raw_source_pair_reads_vdst_pair(vdst, src0, *src0_hi) ||
                          raw_source_pair_reads_vdst_pair(vdst, src1, *src1_hi);
    if (!overlaps)
      return false;
    return liveness == nullptr ||
           !find_raw_v_mul_u64_low_scratch(vdst, src0, *src0_hi, src1, *src1_hi, inst, *liveness);
  }

  if (((w0 >> 25) & 0x3Fu) != kGfx1250VMulU64E32Opcode)
    return false;

  const uint8_t vdst = static_cast<uint8_t>((w0 >> 17) & 0xFFu);
  if (vdst > 254u)
    return false;

  const uint16_t src0 = static_cast<uint16_t>(w0 & 0x1FFu);
  const uint16_t src1 = static_cast<uint16_t>(256u + ((w0 >> 9) & 0xFFu));
  const std::optional<uint32_t> literal =
      src0 == 255 && inst.size() >= static_cast<int>(2 * sizeof(uint32_t))
          ? std::optional<uint32_t>(raw[1])
          : std::nullopt;
  const auto src0_hi = raw_pair_hi_src_with_literal(src0, literal);
  const auto src1_hi = raw_pair_hi_src_with_literal(src1, literal);
  if (!src0_hi || !src1_hi)
    return false;
  const bool overlaps = raw_source_pair_reads_vdst_pair(vdst, src0, *src0_hi) ||
                        raw_source_pair_reads_vdst_pair(vdst, src1, *src1_hi);
  if (!overlaps)
    return false;
  return liveness == nullptr ||
         !find_raw_v_mul_u64_low_scratch(vdst, src0, *src0_hi, src1, *src1_hi, inst, *liveness);
}

void append_hardware_pending_warning(std::vector<std::string> *warnings,
                                     std::string_view mnemonic) {
  if (!warnings || !has_hardware_pending_semantic_lowering(mnemonic))
    return;
  const std::string warning = "hardware validation pending for " + std::string(mnemonic);
  if (std::find(warnings->begin(), warnings->end(), warning) == warnings->end())
    warnings->push_back(warning);
}

enum class PlacementTier {
  InPlace,
  LocalPaddingCave,
  LocalUnreachableTextCave,
  ChainedAppendedCave,
  LocalPaddingBranchIsland,
  LocalUnreachableBranchIsland,
  AppendedCaveLongBranchIsland,
  AppendedCaveBranchChain,
  AppendedCave,
};

[[nodiscard]] std::string_view placement_tier_name(PlacementTier tier) {
  switch (tier) {
  case PlacementTier::InPlace:
    return "in-place";
  case PlacementTier::LocalPaddingCave:
    return "local-padding-cave";
  case PlacementTier::LocalUnreachableTextCave:
    return "local-unreachable-text-cave";
  case PlacementTier::ChainedAppendedCave:
    return "chained-appended-cave";
  case PlacementTier::LocalPaddingBranchIsland:
    return "local-padding-branch-island";
  case PlacementTier::LocalUnreachableBranchIsland:
    return "local-unreachable-branch-island";
  case PlacementTier::AppendedCaveLongBranchIsland:
    return "appended-cave-long-branch-island";
  case PlacementTier::AppendedCaveBranchChain:
    return "appended-cave-branch-chain";
  case PlacementTier::AppendedCave:
    return "appended-cave";
  }
  return "unknown";
}

[[nodiscard]] bool cave_diagnostics_enabled() {
  const char *enabled = std::getenv("ROCJITSU_DBT_CAVE_DIAGNOSTICS");
  return enabled != nullptr && enabled[0] != '\0';
}

void append_placement_diagnostic(std::vector<std::string> *warnings, std::string_view action,
                                 PlacementTier tier, const SemanticReplacement &repl,
                                 std::string_view detail = {}) {
  if (!warnings || !cave_diagnostics_enabled())
    return;

  std::ostringstream os;
  os << "code placement " << action << " tier=" << placement_tier_name(tier);
  if (!repl.source_mnemonic.empty())
    os << " source_mnemonic=" << repl.source_mnemonic;
  os << " source_offset=0x" << std::hex << repl.start_offset << " source_size=0x"
     << (repl.end_offset - repl.start_offset) << " target_size=0x"
     << (repl.target_words.size() * sizeof(uint32_t));
  if (!detail.empty())
    os << " " << detail;
  warnings->push_back(os.str());
}

[[nodiscard]] std::string cave_range_diagnostic(const char *kind, const SemanticReplacement &repl,
                                                uint64_t cave_byte_offset, uint32_t target_size) {
  std::ostringstream os;
  os << kind;
  if (!repl.source_mnemonic.empty())
    os << " source_mnemonic=" << repl.source_mnemonic;
  os << " source_offset=0x" << std::hex << repl.start_offset << " source_size=0x"
     << (repl.end_offset - repl.start_offset) << " target_size=0x" << target_size
     << " cave_offset=0x" << cave_byte_offset;
  return os.str();
}

[[nodiscard]] bool requires_semantic_expansion(rj_code_arch_t guest, const Instruction &inst) {
  if (guest != ROCJITSU_CODE_ARCH_GFX1250)
    return false;
  if (inst.encoding_id() == kGfx1250VopdEncodingId || inst.encoding_id() == kGfx1250Vopd3EncodingId)
    return true;
  if (inst.opcode() == kGfx1250VAddNcU64E32Opcode &&
      inst.encoding_id() >= kGfx1250Vop2AddNcU64EncodingId0 &&
      inst.encoding_id() <= kGfx1250Vop2AddNcU64EncodingId3)
    return true;
  if (inst.opcode() == kGfx1250VSubNcU64E32Opcode &&
      inst.encoding_id() >= kGfx1250Vop2SubNcU64EncodingId0 &&
      inst.encoding_id() <= kGfx1250Vop2SubNcU64EncodingId3)
    return true;
  if (inst.opcode() == kGfx1250VMulU64E32Opcode &&
      inst.encoding_id() >= kGfx1250Vop2MulU64EncodingId0 &&
      inst.encoding_id() <= kGfx1250Vop2MulU64EncodingId3)
    return true;
  return false;
}

[[nodiscard]] bool starts_with(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

[[nodiscard]] std::optional<uint16_t> raw_vop3_opcode(uint32_t w0) {
  if ((w0 >> 26) != kVop3Encoding)
    return std::nullopt;
  return static_cast<uint16_t>((w0 >> 16) & 0x3FFu);
}

[[nodiscard]] bool is_gfx1250_vop3_compare_opcode(uint16_t op) {
  return (op >= 1 && op <= 14) || (op >= 17 && op <= 30) || (op >= 33 && op <= 46) ||
         (op >= 49 && op <= 54) || (op >= 57 && op <= 62) || (op >= 65 && op <= 70) ||
         (op >= 73 && op <= 78) || (op >= 81 && op <= 86) || (op >= 89 && op <= 94) ||
         (op >= 129 && op <= 142) || (op >= 145 && op <= 158) || (op >= 161 && op <= 174) ||
         (op >= 177 && op <= 182) || (op >= 185 && op <= 190) || (op >= 193 && op <= 198) ||
         (op >= 201 && op <= 206) || (op >= 209 && op <= 214) || (op >= 217 && op <= 222);
}

[[nodiscard]] bool is_scalar_alu_encoding(uint16_t encoding_id) {
  if (encoding_id == kEnc_SOP1 || encoding_id == kEnc_SOPC)
    return true;
  if ((encoding_id & 0x180u) == kEnc_SOP2)
    return true;
  if ((encoding_id & 0x1E0u) == kEnc_SOPK)
    return true;
  return false;
}

[[nodiscard]] bool is_rdna4_scc_defining_scalar_alu(const Instruction &inst) {
  if (inst.encoding_id() == kEnc_SOPC)
    return true;

  if (is_sop2_encoding(inst)) {
    const uint16_t op = inst.opcode();
    // GFX12 SOP2 SCC producers: carry-out arithmetic, absdiff, shifts,
    // scaled-add, min/max, bitwise ops, and BFE. Carry-in ops are included
    // because the hardware form carries out through SCC as well.
    return op <= 6u || (op >= 8u && op <= 41u);
  }

  const std::string_view mnemonic = inst.mnemonic();
  if ((inst.encoding_id() & 0x1E0u) == kEnc_SOPK)
    return starts_with(mnemonic, "s_cmpk_") || starts_with(mnemonic, "s_addk_");

  if (inst.encoding_id() != kEnc_SOP1)
    return false;

  return starts_with(mnemonic, "s_not_") || starts_with(mnemonic, "s_wqm_") ||
         starts_with(mnemonic, "s_bcnt0_") || starts_with(mnemonic, "s_bcnt1_") ||
         starts_with(mnemonic, "s_quadmask_") || mnemonic == "s_abs_i32" ||
         starts_with(mnemonic, "s_and_saveexec_") || starts_with(mnemonic, "s_or_saveexec_") ||
         starts_with(mnemonic, "s_xor_saveexec_") || starts_with(mnemonic, "s_andn2_saveexec_") ||
         starts_with(mnemonic, "s_orn2_saveexec_") ||
         starts_with(mnemonic, "s_and_not1_saveexec_") ||
         starts_with(mnemonic, "s_or_not1_saveexec_") ||
         starts_with(mnemonic, "s_nand_saveexec_") || starts_with(mnemonic, "s_nor_saveexec_") ||
         starts_with(mnemonic, "s_xnor_saveexec_") ||
         starts_with(mnemonic, "s_barrier_signal_isfirst");
}

[[nodiscard]] bool has_explicit_sgpr_destination(const Instruction &inst) {
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const Operand *operand = inst.dst_operand(i);
    if (!operand)
      continue;
    const auto reg = operand->to_register_ref();
    if (reg && reg->cls == RegClass::SGPR)
      return true;
  }
  return false;
}

void append_rdna4_scalar_dependency_barriers_after(std::vector<uint32_t> &words,
                                                   const Instruction &inst,
                                                   rj_code_arch_t host_arch) {
  if (inst.is_branch() || inst.is_barrier() || inst.is_waitcnt() || inst.encoding_id() == kEnc_SOPP)
    return;
  if (is_rdna4_scc_defining_scalar_alu(inst)) {
    words.push_back(build_s_delay_alu(kDelayAluSaluDep1, host_arch));
    if (has_explicit_sgpr_destination(inst))
      words.push_back(build_s_wait_alu(kWaitAluDepctrSaSdst0, host_arch));
    return;
  }
  if (is_scalar_alu_encoding(inst.encoding_id())) {
    words.push_back(build_s_wait_alu(kWaitAluDepctrSaSdst0, host_arch));
    return;
  }

  const std::string_view mnemonic = inst.mnemonic();
  if (starts_with(mnemonic, "v_cmpx")) {
    words.push_back(build_s_wait_alu(kWaitAluDepctrVaSdst0, host_arch));
    return;
  }
  if (starts_with(mnemonic, "v_cmp")) {
    if (has_explicit_sgpr_destination(inst))
      words.push_back(build_s_wait_alu(kWaitAluDepctrVaSdst0, host_arch));
    else
      words.push_back(build_s_wait_alu(kWaitAluDepctrVaVcc0, host_arch));
    return;
  }
  if (starts_with(mnemonic, "v_readfirstlane_b32") || starts_with(mnemonic, "v_readlane_b32")) {
    words.push_back(build_s_wait_alu(kWaitAluDepctrVaSdst0, host_arch));
    return;
  }
  if (starts_with(mnemonic, "v_") && has_explicit_sgpr_destination(inst))
    words.push_back(build_s_wait_alu(kWaitAluDepctrVaSdst0, host_arch));
}

[[nodiscard]] bool rdna4_memory_dependency_wait_before(const Instruction &inst) {
  const uint16_t enc = inst.encoding_id();
  if ((enc & 0x1FEu) == kEnc_VFLAT || (enc & 0x1FEu) == kEnc_VSCRATCH ||
      (enc & 0x1FEu) == kEnc_VGLOBAL)
    return true;
  if ((enc & 0x1F8u) == kEnc_VBUFFER || (enc & 0x1F8u) == kEnc_VDS)
    return true;
  return false;
}

[[nodiscard]] bool is_s_wait_alu(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  if (raw != nullptr && inst.size() >= static_cast<int>(sizeof(uint32_t))) {
    constexpr uint8_t kSoppWaitAlu = 8;
    return (raw[0] >> 23) == kSoppEncodingPrefix && ((raw[0] >> 16) & 0x7Fu) == kSoppWaitAlu;
  }
  return inst.mnemonic() == "s_wait_alu";
}

[[nodiscard]] bool is_s_delay_alu(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  if (raw != nullptr && inst.size() >= static_cast<int>(sizeof(uint32_t))) {
    constexpr uint8_t kSoppDelayAlu = 7;
    return (raw[0] >> 23) == kSoppEncodingPrefix && ((raw[0] >> 16) & 0x7Fu) == kSoppDelayAlu;
  }
  return inst.mnemonic() == "s_delay_alu";
}

[[nodiscard]] bool previous_vgpr_def_is_used_by(const Instruction &prev, const Instruction &inst) {
  InstDefUse prev_du(prev);
  InstDefUse inst_du(inst);
  RegisterSet prev_defs = prev_du.defs;
  RegisterSet inst_uses = inst_du.uses;
  prev_defs.clear_class(RegClass::SGPR);
  prev_defs.clear_class(RegClass::ACC_VGPR);
  inst_uses.clear_class(RegClass::SGPR);
  inst_uses.clear_class(RegClass::ACC_VGPR);
  return prev_defs.intersects(inst_uses);
}

[[nodiscard]] bool is_vector_alu_dependency_participant(const Instruction &inst) {
  if (inst.is_branch() || inst.is_barrier() || inst.is_waitcnt() || inst.is_memory_op() ||
      is_s_wait_alu(inst) || is_s_delay_alu(inst))
    return false;
  return starts_with(inst.mnemonic(), "v_");
}

[[nodiscard]] bool
rdna4_adjacent_valu_dependency_delay_needed_before(InstructionList::Iterator block_begin,
                                                   InstructionList::Iterator inst_it) {
  if (inst_it == block_begin || !is_vector_alu_dependency_participant(*inst_it))
    return false;

  auto prev_it = inst_it;
  --prev_it;
  const Instruction &prev = *prev_it;
  if (!is_vector_alu_dependency_participant(prev))
    return false;
  return previous_vgpr_def_is_used_by(prev, *inst_it);
}

[[nodiscard]] bool rdna4_memory_dependency_wait_needed_before(InstructionList::Iterator block_begin,
                                                              InstructionList::Iterator inst_it) {
  if (!rdna4_memory_dependency_wait_before(*inst_it))
    return false;
  if (inst_it == block_begin)
    return true;

  constexpr size_t kBackscanLimit = 32;
  size_t scanned = 0;
  for (auto prev_it = inst_it; prev_it != block_begin;) {
    --prev_it;
    const Instruction &prev = *prev_it;
    if (is_s_wait_alu(prev))
      return false;
    if (previous_vgpr_def_is_used_by(prev, *inst_it))
      return true;
    if (prev.is_memory_op())
      return false;
    if (++scanned >= kBackscanLimit)
      return false;
  }
  return false;
}

void fill_nops(std::vector<uint8_t> &text, uint64_t offset, uint32_t size,
               rj_code_arch_t host_arch) {
  const uint32_t nop = build_s_nop(0, host_arch);
  for (uint32_t i = 0; i < size; i += sizeof(nop))
    std::memcpy(text.data() + offset + i, &nop, sizeof(nop));
}

[[nodiscard]] std::vector<uint32_t> nop_words(uint32_t size, rj_code_arch_t host_arch) {
  assert(size % sizeof(uint32_t) == 0 && "instruction size must be word aligned");
  return std::vector<uint32_t>(size / sizeof(uint32_t), build_s_nop(0, host_arch));
}

[[nodiscard]] std::vector<uint32_t> copy_instruction_words(std::span<const uint8_t> text,
                                                           uint64_t offset, uint32_t size) {
  assert(size % sizeof(uint32_t) == 0 && "instruction size must be word aligned");
  assert(offset + size <= text.size() && "instruction exceeds text bounds");
  std::vector<uint32_t> words(size / sizeof(uint32_t));
  std::memcpy(words.data(), text.data() + offset, size);
  return words;
}

void nop_untranslated_gfx1250_vgpr_msb_zero_resets(std::vector<uint8_t> &text,
                                                   rj_code_arch_t host_arch) {
  assert(text.size() % sizeof(uint32_t) == 0 && "text size must be word aligned");
  const uint32_t nop = build_s_nop(0, host_arch);
  for (size_t offset = 0; offset + sizeof(uint32_t) <= text.size(); offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, text.data() + offset, sizeof(word));
    uint16_t simm16 = 0;
    if (!is_gfx1250_s_set_vgpr_msb(word, simm16))
      continue;
    if (amdgpu::s_set_vgpr_msb_new_mode(simm16) != 0)
      continue;
    std::memcpy(text.data() + offset, &nop, sizeof(nop));
  }
}

// Large kernels can need thousands of independent branch islands before later
// source offsets get a direct branch to the appended cave. Keep a sizeable
// front slab so early source offsets have enough reachable long-branch slots.
constexpr uint32_t kCaveBranchIslandBlockWords = 16384;

void append_cave_branch_island_block(CodeObjectPatcher &patcher, CaveBranchIslandState &state,
                                     rj_code_arch_t host_arch) {
  const uint64_t block_body_offset = patcher.cave_body_size();
  std::vector<uint32_t> island_nops(kCaveBranchIslandBlockWords, build_s_nop(0, host_arch));
  patcher.append_cave_body(island_nops);

  for (uint32_t index = 0; index < kCaveBranchIslandBlockWords; ++index) {
    state.body_offsets.push_back(block_body_offset + index * sizeof(uint32_t));
    state.used.push_back(false);
  }
  state.initialized = true;
}

void ensure_cave_branch_islands(CodeObjectPatcher &patcher, CaveBranchIslandState &state,
                                rj_code_arch_t host_arch) {
  if (!state.initialized)
    append_cave_branch_island_block(patcher, state, host_arch);
}

[[nodiscard]] bool allocate_cave_branch_chain(CodeObjectPatcher &patcher,
                                              CaveBranchIslandState &state,
                                              const SemanticReplacement &repl,
                                              uint64_t island_target, rj_code_arch_t host_arch,
                                              int16_t &entry_branch_dwords) {
  ensure_cave_branch_islands(patcher, state, host_arch);

  const uint64_t cave_start = patcher.cave_start();
  const auto island_pc = [&](size_t index) { return cave_start + state.body_offsets[index]; };
  const auto upper_index_for_pc = [&](uint64_t pc) {
    return static_cast<size_t>(
        std::ranges::upper_bound(state.body_offsets, pc, {},
                                 [&](uint64_t offset) { return cave_start + offset; }) -
        state.body_offsets.begin());
  };

  // Return the furthest unused island that advances the chain and is reachable
  // by one SOPP branch. body_offsets is sorted, so this normally examines only
  // one slot instead of rescanning the entire island slab for every hop.
  const auto furthest_reachable_island = [&](uint64_t current_pc) -> std::optional<size_t> {
    constexpr uint64_t kBranchPcBiasBytes = sizeof(uint32_t);
    const uint64_t max_delta = kSoppBranchMaxForwardBytes + kBranchPcBiasBytes;
    const uint64_t max_reachable = current_pc > std::numeric_limits<uint64_t>::max() - max_delta
                                       ? std::numeric_limits<uint64_t>::max()
                                       : current_pc + max_delta;
    const uint64_t last_candidate_pc =
        island_target == 0 ? 0 : std::min(max_reachable, island_target - 1);

    const size_t first = upper_index_for_pc(current_pc);
    size_t end = upper_index_for_pc(last_candidate_pc);
    while (end > first) {
      const size_t index = --end;
      if (state.used[index])
        continue;
      int16_t ignored = 0;
      if (compute_sopp_branch_offset(current_pc, island_pc(index), ignored))
        return index;
    }
    return std::nullopt;
  };

  uint64_t current_pc = repl.start_offset;
  std::vector<size_t> chain;

  for (;;) {
    int16_t direct_dwords = 0;
    if (compute_sopp_branch_offset(current_pc, island_target, direct_dwords)) {
      break;
    }

    const auto best_index = furthest_reachable_island(current_pc);
    if (!best_index)
      return false;
    chain.push_back(*best_index);
    current_pc = island_pc(*best_index);
  }

  if (chain.empty())
    return false;

  if (!compute_sopp_branch_offset(repl.start_offset, island_pc(chain.front()),
                                  entry_branch_dwords)) {
    return false;
  }
  for (size_t position = 0; position < chain.size(); ++position) {
    const size_t index = chain[position];
    const uint64_t target =
        position + 1 < chain.size() ? island_pc(chain[position + 1]) : island_target;
    int16_t branch_dwords = 0;
    if (!compute_sopp_branch_offset(island_pc(index), target, branch_dwords))
      return false;
    const std::array<uint32_t, 1> island_words{build_s_branch(branch_dwords, host_arch)};
    patcher.overwrite_cave_body(state.body_offsets[index], island_words);
    state.used[index] = true;
  }
  return true;
}

[[nodiscard]] bool allocate_cave_long_branch_island(CodeObjectPatcher &patcher,
                                                    CaveBranchIslandState &state,
                                                    const SemanticReplacement &repl,
                                                    uint64_t island_target,
                                                    rj_code_arch_t host_arch, uint16_t sgpr_pair,
                                                    int16_t &entry_branch_dwords) {
  ensure_cave_branch_islands(patcher, state, host_arch);

  const uint64_t cave_start = patcher.cave_start();
  constexpr uint64_t kBranchPcBiasBytes = sizeof(uint32_t);
  const uint64_t max_delta = kSoppBranchMaxForwardBytes + kBranchPcBiasBytes;
  const uint64_t max_reachable =
      repl.start_offset > std::numeric_limits<uint64_t>::max() - max_delta
          ? std::numeric_limits<uint64_t>::max()
          : repl.start_offset + max_delta;
  const auto project_pc = [&](uint64_t offset) { return cave_start + offset; };
  const size_t reachable_begin = static_cast<size_t>(
      std::ranges::upper_bound(state.body_offsets, repl.start_offset, {}, project_pc) -
      state.body_offsets.begin());
  const size_t reachable_end = static_cast<size_t>(
      std::ranges::upper_bound(state.body_offsets, max_reachable, {}, project_pc) -
      state.body_offsets.begin());
  if (reachable_begin == reachable_end)
    return false;

  const auto try_range = [&](size_t begin, size_t end) -> std::optional<size_t> {
    for (size_t index = begin; index < end; ++index) {
      if (state.used[index])
        continue;

      const uint64_t island_pc = cave_start + state.body_offsets[index];
      int16_t entry_dwords = 0;
      if (!compute_sopp_branch_offset(repl.start_offset, island_pc, entry_dwords))
        continue;

      auto long_branch = build_s_setpc_long_branch(island_pc, island_target, sgpr_pair);
      if (long_branch.empty())
        continue;
      if (index + long_branch.size() > state.body_offsets.size())
        continue;

      bool available = true;
      for (size_t word = 0; word < long_branch.size(); ++word) {
        if (state.used[index + word] || state.body_offsets[index + word] !=
                                            state.body_offsets[index] + word * sizeof(uint32_t)) {
          available = false;
          break;
        }
      }
      if (!available)
        continue;

      patcher.overwrite_cave_body(state.body_offsets[index], long_branch);
      for (size_t word = 0; word < long_branch.size(); ++word)
        state.used[index + word] = true;
      state.next_long_search_index = index + long_branch.size();
      entry_branch_dwords = entry_dwords;
      return index;
    }
    return std::nullopt;
  };

  const size_t search_start =
      std::clamp(state.next_long_search_index, reachable_begin, reachable_end);
  if (try_range(search_start, reachable_end))
    return true;
  if (search_start == reachable_begin)
    return false;
  return try_range(reachable_begin, search_start).has_value();
}

[[nodiscard]] std::vector<uint64_t> kernel_entry_offsets(std::span<const KdTranslation> kernels) {
  std::vector<uint64_t> offsets;
  offsets.reserve(kernels.size());
  for (const KdTranslation &kernel : kernels)
    offsets.push_back(kernel.entry_text_offset);

  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

struct KernelTranslationScope {
  KdTranslation *translation = nullptr;
  BasicBlock *entry = nullptr;
  std::vector<BasicBlock *> blocks;
};

[[nodiscard]] bool s_setpc_from_sreg(const Instruction &inst, uint32_t word, uint16_t ssrc0) {
  if (inst.size() != sizeof(uint32_t) || inst.mnemonic() != "s_setpc_b64")
    return false;
  return static_cast<uint16_t>(word & 0xffu) == ssrc0;
}

[[nodiscard]] std::vector<BasicBlock *>
function_return_blocks(BasicBlock &callee, uint16_t return_sreg, std::span<const uint8_t> text,
                       const std::unordered_set<BasicBlock *> &allowed_blocks) {
  std::vector<BasicBlock *> returns;
  std::vector<BasicBlock *> stack{&callee};
  std::unordered_set<BasicBlock *> visited;

  while (!stack.empty()) {
    BasicBlock *block = stack.back();
    stack.pop_back();
    if (block == nullptr || !allowed_blocks.contains(block) || !visited.insert(block).second)
      continue;

    const Instruction *term = block->terminator();
    if (term != nullptr && s_setpc_from_sreg(*term, read_u32(text, term->src_loc()), return_sreg)) {
      returns.push_back(block);
      continue;
    }

    for (BasicBlock *successor : block->successors())
      stack.push_back(successor);
  }
  return returns;
}

[[nodiscard]] std::unordered_set<uint64_t>
scoped_call_return_offsets(std::span<BasicBlock *const> blocks, std::span<const uint8_t> text) {
  std::unordered_set<BasicBlock *> allowed_blocks(blocks.begin(), blocks.end());
  std::unordered_set<uint64_t> returns;
  for (BasicBlock *block : blocks) {
    if (block == nullptr)
      continue;
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      if (call.callee == nullptr || call.continuation == nullptr ||
          !allowed_blocks.contains(call.callee) || !allowed_blocks.contains(call.continuation)) {
        continue;
      }
      for (BasicBlock *return_block :
           function_return_blocks(*call.callee, call.return_sreg, text, allowed_blocks)) {
        if (const Instruction *term = return_block->terminator())
          returns.insert(term->src_loc());
      }
    }
  }
  return returns;
}

using HighBankBlockModeMap = std::unordered_map<BasicBlock *, uint8_t>;
using HighBankBlockModeSetMap = std::unordered_map<BasicBlock *, std::bitset<256>>;
using HighBankBlockSet = std::unordered_set<BasicBlock *>;

[[nodiscard]] std::unordered_map<const BasicBlock *, uint32_t>
scope_reach_count_by_block(std::span<const KernelTranslationScope> scopes) {
  std::unordered_map<const BasicBlock *, uint32_t> counts;
  for (const KernelTranslationScope &scope : scopes) {
    for (const BasicBlock *block : scope.blocks) {
      if (block != nullptr)
        ++counts[block];
    }
  }
  return counts;
}

[[nodiscard]] bool block_is_shared(const BasicBlock *block,
                                   const std::unordered_map<const BasicBlock *, uint32_t> &counts) {
  const auto it = counts.find(block);
  return it != counts.end() && it->second > 1;
}

[[nodiscard]] bool
scope_has_shared_blocks(const KernelTranslationScope &scope,
                        const std::unordered_map<const BasicBlock *, uint32_t> &counts) {
  return std::ranges::any_of(
      scope.blocks, [&](const BasicBlock *block) { return block_is_shared(block, counts); });
}

[[nodiscard]] std::vector<BasicBlock *>
unique_blocks_for_scopes(std::span<const KernelTranslationScope> scopes, bool shared_only,
                         const std::unordered_map<const BasicBlock *, uint32_t> &counts) {
  std::vector<BasicBlock *> blocks;
  std::unordered_set<const BasicBlock *> seen;
  for (const KernelTranslationScope &scope : scopes) {
    for (BasicBlock *block : scope.blocks) {
      if (block == nullptr || (shared_only && !block_is_shared(block, counts)) ||
          !seen.insert(block).second) {
        continue;
      }
      blocks.push_back(block);
    }
  }
  return blocks;
}

[[nodiscard]] TranslationContext
merged_translation_context_for_scopes(std::span<const KernelTranslationScope> scopes) {
  TranslationContext context;
  for (const KernelTranslationScope &scope : scopes) {
    if (scope.translation == nullptr)
      continue;
    context.num_vgprs = std::max(context.num_vgprs, scope.translation->target_vgpr_count);
    context.num_agprs = std::max(context.num_agprs, scope.translation->target_agpr_count);
    context.accum_offset = std::max(context.accum_offset, scope.translation->target_accvgpr_base);
    context.num_sgprs = std::max(context.num_sgprs, scope.translation->target_sgpr_count);
  }
  return context;
}

void merge_translation_context_requirements(TranslationContext &dst,
                                            const TranslationContext &src) {
  dst.require_vgprs(src.required_vgpr_count);
  dst.require_sgprs(src.required_sgpr_count);
}

[[nodiscard]] uint8_t first_high_bank_mode(const std::bitset<256> &modes) {
  for (uint16_t mode = 0; mode < 256; ++mode) {
    if (modes.test(mode))
      return static_cast<uint8_t>(mode);
  }
  return 0;
}

[[nodiscard]] constexpr uint8_t high_bank_role_mask(HighBankRole role) {
  return static_cast<uint8_t>(1u << static_cast<uint8_t>(role));
}

[[nodiscard]] uint8_t gfx1250_high_bank_role_mask(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  if (starts_with(mnemonic, "v_nop"))
    return 0;
  if (starts_with(mnemonic, "v_"))
    return high_bank_role_mask(HighBankRole::Src0) | high_bank_role_mask(HighBankRole::Src1) |
           high_bank_role_mask(HighBankRole::Src2) | high_bank_role_mask(HighBankRole::Dst);
  if (starts_with(mnemonic, "ds_")) {
    const DsHighBankOperands ops = describe_high_bank_ds_operands(mnemonic);
    if (!ops.recognized)
      return high_bank_role_mask(HighBankRole::Src0) | high_bank_role_mask(HighBankRole::Src1) |
             high_bank_role_mask(HighBankRole::Dst);
    uint8_t mask = 0;
    if (ops.uses_addr)
      mask |= high_bank_role_mask(HighBankRole::Src0);
    if (ops.data0_width != 0)
      mask |= high_bank_role_mask(HighBankRole::Src1);
    if (ops.vdst_width != 0)
      mask |= high_bank_role_mask(HighBankRole::Dst);
    return mask;
  }
  if (starts_with(mnemonic, "buffer_")) {
    const uint32_t *raw = inst.raw_encoding();
    if (!raw || !is_gfx1250_vbuffer_instruction(inst))
      return high_bank_role_mask(HighBankRole::Src0) | high_bank_role_mask(HighBankRole::Src1) |
             high_bank_role_mask(HighBankRole::Dst);
    const auto fields = gfx1250_to_rdna4::decode_vbuffer_gfx1250(raw[0], raw[1], raw[2]);
    const VbufferHighBankOperands ops = describe_high_bank_vbuffer_operands(inst, fields);
    if (!ops.recognized)
      return high_bank_role_mask(HighBankRole::Src0) | high_bank_role_mask(HighBankRole::Src1) |
             high_bank_role_mask(HighBankRole::Dst);
    uint8_t mask = 0;
    if (ops.vaddr_width != 0)
      mask |= high_bank_role_mask(HighBankRole::Src0);
    if (ops.is_store && ops.vdata_width != 0)
      mask |= high_bank_role_mask(HighBankRole::Src1);
    if (ops.is_load && ops.vdata_width != 0)
      mask |= high_bank_role_mask(HighBankRole::Dst);
    return mask;
  }
  return 0;
}

[[nodiscard]] bool high_bank_modes_equivalent_for_roles(const std::bitset<256> &modes,
                                                        uint8_t role_mask) {
  std::optional<uint8_t> reference;
  for (uint16_t mode = 0; mode < 256; ++mode) {
    if (!modes.test(mode))
      continue;
    if (!reference) {
      reference = static_cast<uint8_t>(mode);
      continue;
    }
    for (const HighBankRole role :
         {HighBankRole::Src0, HighBankRole::Src1, HighBankRole::Src2, HighBankRole::Dst}) {
      if ((role_mask & high_bank_role_mask(role)) == 0)
        continue;
      if (high_bank_selector(*reference, role) !=
          high_bank_selector(static_cast<uint8_t>(mode), role))
        return false;
    }
  }
  return true;
}

[[nodiscard]] std::optional<std::bitset<256>>
gfx1250_high_bank_exit_modes_for_block(BasicBlock &block, std::bitset<256> modes) {
  for (const Instruction &inst : block.instructions()) {
    const uint32_t *raw = inst.raw_encoding();
    if (raw != nullptr && inst.size() >= static_cast<int>(sizeof(uint32_t))) {
      uint16_t simm16 = 0;
      if (is_gfx1250_s_set_vgpr_msb(raw[0], simm16)) {
        modes.reset();
        modes.set(amdgpu::s_set_vgpr_msb_new_mode(simm16));
        continue;
      }
    }

    if (modes.count() != 1) {
      const uint8_t role_mask = gfx1250_high_bank_role_mask(inst);
      if (role_mask != 0 && !high_bank_modes_equivalent_for_roles(modes, role_mask))
        return std::nullopt;
    }
  }
  return modes;
}

[[nodiscard]] bool gfx1250_block_ends_with_sop1(BasicBlock &block, uint8_t op) {
  const Instruction *term = block.terminator();
  if (term == nullptr || term->size() != static_cast<int>(sizeof(uint32_t)))
    return false;
  const uint32_t *raw = term->raw_encoding();
  if (raw == nullptr)
    return false;
  const auto inst = std::bit_cast<gfx1250::Sop1MachineInst>(raw[0]);
  return inst.encoding == 0x17Du && inst.op == op;
}

[[nodiscard]] bool gfx1250_block_ends_with_s_setpc(BasicBlock &block) {
  constexpr uint8_t kOpSSetPcB64 = 72;
  return gfx1250_block_ends_with_sop1(block, kOpSSetPcB64);
}

[[nodiscard]] bool gfx1250_block_ends_with_s_swappc(BasicBlock &block) {
  constexpr uint8_t kOpSSwapPcB64 = 73;
  return gfx1250_block_ends_with_sop1(block, kOpSSwapPcB64);
}

[[nodiscard]] bool gfx1250_has_call_fallthrough_predecessor(BasicBlock &block) {
  return std::ranges::any_of(block.predecessors(), [&](BasicBlock *pred) {
    return pred != nullptr && pred->end_offset() == block.start_offset() &&
           gfx1250_block_ends_with_s_swappc(*pred);
  });
}

[[nodiscard]] bool gfx1250_is_skipped_call_return_edge(BasicBlock &pred, BasicBlock &succ) {
  return gfx1250_block_ends_with_s_setpc(pred) && gfx1250_has_call_fallthrough_predecessor(succ);
}

[[nodiscard]] HighBankBlockSet
gfx1250_skipped_call_return_continuation_closure(const KernelTranslationScope &scope,
                                                 const HighBankBlockSet &scope_blocks,
                                                 const HighBankBlockSet &reached_blocks) {
  HighBankBlockSet skipped;
  bool changed = true;
  while (changed) {
    changed = false;
    for (BasicBlock *block : scope.blocks) {
      if (block == nullptr || reached_blocks.contains(block) || skipped.contains(block))
        continue;

      bool has_in_scope_predecessor = false;
      bool has_skipped_reason = false;
      bool all_predecessors_skipped = true;
      for (BasicBlock *pred : block->predecessors()) {
        if (pred == nullptr || !scope_blocks.contains(pred))
          continue;
        has_in_scope_predecessor = true;
        if (skipped.contains(pred) || gfx1250_is_skipped_call_return_edge(*pred, *block)) {
          has_skipped_reason = true;
          continue;
        }
        all_predecessors_skipped = false;
        break;
      }

      if (has_in_scope_predecessor && has_skipped_reason && all_predecessors_skipped) {
        skipped.insert(block);
        changed = true;
      }
    }
  }
  return skipped;
}

[[nodiscard]] std::optional<HighBankBlockModeMap>
gfx1250_high_bank_entry_modes_for_scope(const KernelTranslationScope &scope) {
  if (scope.entry == nullptr)
    return std::nullopt;

  std::unordered_set<BasicBlock *> scope_blocks;
  scope_blocks.reserve(scope.blocks.size());
  for (BasicBlock *block : scope.blocks) {
    if (block != nullptr)
      scope_blocks.insert(block);
  }

  HighBankBlockModeSetMap entry_mode_sets;
  std::vector<BasicBlock *> worklist;
  entry_mode_sets[scope.entry].set(0);
  worklist.push_back(scope.entry);

  while (!worklist.empty()) {
    BasicBlock *block = worklist.back();
    worklist.pop_back();
    if (block == nullptr)
      continue;

    const auto mode_it = entry_mode_sets.find(block);
    if (mode_it == entry_mode_sets.end())
      continue;
    const auto exit_modes = gfx1250_high_bank_exit_modes_for_block(*block, mode_it->second);
    if (!exit_modes)
      return std::nullopt;

    for (BasicBlock *succ : block->successors()) {
      if (succ == nullptr || !scope_blocks.contains(succ))
        continue;
      auto &succ_modes = entry_mode_sets[succ];
      const auto old_modes = succ_modes;
      succ_modes |= *exit_modes;
      if (succ_modes != old_modes) {
        worklist.push_back(succ);
      }
    }
  }

  HighBankBlockSet reached_blocks;
  reached_blocks.reserve(entry_mode_sets.size());
  for (const auto &[block, modes] : entry_mode_sets) {
    (void)modes;
    if (block != nullptr)
      reached_blocks.insert(block);
  }
  const HighBankBlockSet skipped_blocks =
      gfx1250_skipped_call_return_continuation_closure(scope, scope_blocks, reached_blocks);

  HighBankBlockModeMap entry_modes;
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    const auto mode_it = entry_mode_sets.find(block);
    if (mode_it == entry_mode_sets.end()) {
      if (skipped_blocks.contains(block))
        continue;
      return std::nullopt;
    }
    entry_modes.emplace(block, first_high_bank_mode(mode_it->second));
  }

  return entry_modes;
}

[[nodiscard]] HighBankShadowAnalysis
gfx1250_high_bank_shadow_analysis_for_scope(const KernelTranslationScope &scope,
                                            const LivenessAnalysis &liveness,
                                            const HighBankBlockModeMap &entry_modes) {
  HighBankShadowAnalysis analysis;
  ShadowFootprint footprint;
  std::unordered_set<BasicBlock *> scope_blocks;
  scope_blocks.reserve(scope.blocks.size());
  for (BasicBlock *block : scope.blocks) {
    if (block != nullptr)
      scope_blocks.insert(block);
  }
  HighBankBlockSet reached_blocks;
  reached_blocks.reserve(entry_modes.size());
  for (const auto &[block, mode] : entry_modes) {
    (void)mode;
    if (block != nullptr)
      reached_blocks.insert(block);
  }
  const HighBankBlockSet skipped_blocks =
      gfx1250_skipped_call_return_continuation_closure(scope, scope_blocks, reached_blocks);

  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    const auto mode_it = entry_modes.find(block);
    if (mode_it == entry_modes.end()) {
      if (skipped_blocks.contains(block))
        continue;
      analysis.unsupported = true;
      return analysis;
    }
    uint8_t mode = mode_it->second;
    for (const Instruction &inst : block->instructions()) {
      record_low_vgpr_uses(inst, footprint);
      const uint32_t *raw = inst.raw_encoding();
      if (raw == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
        continue;
      uint16_t simm16 = 0;
      if (is_gfx1250_s_set_vgpr_msb(raw[0], simm16)) {
        mode = amdgpu::s_set_vgpr_msb_new_mode(simm16);
        continue;
      }
      record_high_bank_vop_footprint(inst, mode, footprint);
    }
  }

  if (footprint.unsupported) {
    analysis.unsupported = true;
    analysis.unsupported_reason = std::move(footprint.unsupported_reason);
    return analysis;
  }
  if (!footprint.needed)
    return analysis;
  const uint16_t count = static_cast<uint16_t>(footprint.max_logical_vgpr + 1u);
  auto base = find_shadow_base(footprint, liveness, count);
  if (!base && footprint.requires_private_spill && count <= 256u)
    base = 0u;
  if (!base) {
    analysis.unsupported = true;
    return analysis;
  }
  const bool spill_to_private =
      footprint.requires_private_spill || shadow_window_overlaps_low_uses(footprint, *base, count);
  const uint16_t private_slot_count =
      spill_to_private ? static_cast<uint16_t>(footprint.max_private_slot + 1u) : 0u;
  analysis.plan = HighBankShadowPlan{*base, count, private_slot_count, spill_to_private};
  return analysis;
}

struct HighBankScratchPlan {
  uint16_t encoded_base = 0;
  uint32_t minimum_vgprs = 0;
};

[[nodiscard]] bool gfx1250_semantic_lowering_may_borrow_private_vgprs(const Instruction &inst) {
  if (is_gfx1250_k128_fp8_wmma(inst))
    return true;

  if (inst.encoding_id() == kGfx1250Vop3pEncodingId) {
    if (inst.opcode() >= kGfx1250SwmmacF32F16K64Opcode &&
        inst.opcode() <= kGfx1250SwmmacBf16f32K64LastOpcode)
      return true;
    if (inst.opcode() >= kGfx1250SwmmacF32Fp8K128Opcode &&
        inst.opcode() <= kGfx1250SwmmacF16Bf8K128LastOpcode)
      return true;
    return false;
  }

  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < static_cast<int>(2 * sizeof(uint32_t)))
    return false;
  if ((raw[0] >> 26) != kVop3Encoding)
    return false;
  const uint16_t op = static_cast<uint16_t>((raw[0] >> 16) & 0x3FFu);
  return op == kGfx1250VCvtPkF16F32Vop3Opcode;
}

[[nodiscard]] bool
scope_contains_gfx1250_private_borrow_semantic(const KernelTranslationScope &scope) {
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (gfx1250_semantic_lowering_may_borrow_private_vgprs(inst))
        return true;
    }
  }
  return false;
}

[[nodiscard]] bool scope_contains_gfx1250_k32_16bit_wmma(const KernelTranslationScope &scope) {
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (is_gfx1250_k32_16bit_wmma(inst))
        return true;
    }
  }
  return false;
}

[[nodiscard]] bool scope_contains_relative_vgpr_access(const KernelTranslationScope &scope) {
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (starts_with(inst.mnemonic(), "v_movrel"))
        return true;
    }
  }
  return false;
}

void reserve_source_vgprs_for_relative_access(const KernelTranslationScope &scope,
                                              LivenessAnalysis &liveness) {
  if (scope.translation == nullptr || !scope_contains_relative_vgpr_access(scope))
    return;

  // M0 makes the actual v_movrel source or destination register dynamic. The
  // decoded operand exposes only the encoded base, so ordinary liveness cannot
  // safely lend any source-level VGPR to a semantic expansion in this scope.
  // Reserve the full descriptor-backed source file and let descriptor growth
  // provide scratch registers above it.
  const uint32_t count =
      std::min<uint32_t>(scope.translation->guest_vgpr_count, REGISTER_SET_MAX_VGPRS);
  for (uint32_t base = 0; base < count; base += 32u) {
    const auto width = static_cast<uint8_t>(std::min<uint32_t>(32u, count - base));
    liveness.reserve_scratch_registers({RegClass::VGPR, static_cast<uint16_t>(base), width});
  }
}

[[nodiscard]] bool
scope_contains_gfx1250_v_mul_u64_needing_high_scratch(const KernelTranslationScope &scope,
                                                      const LivenessAnalysis &liveness) {
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (gfx1250_v_mul_u64_needs_high_bank_scratch(inst, &liveness))
        return true;
    }
  }
  return false;
}

[[nodiscard]] uint32_t
gfx1250_high_bank_scratch_count_for_scope(const KernelTranslationScope &scope,
                                          const LivenessAnalysis &liveness) {
  uint32_t scratch_count = 0;
  if (scope_contains_gfx1250_v_mul_u64_needing_high_scratch(scope, liveness))
    scratch_count = std::max(scratch_count, kGfx1250VMulU64HighBankScratchCount);
  return scratch_count;
}

[[nodiscard]] uint32_t gfx1250_private_scratch_bytes_for_scope(
    const KernelTranslationScope &scope,
    const std::optional<HighBankShadowPlan> &high_bank_shadow_plan = std::nullopt) {
  uint32_t bytes =
      scope_contains_gfx1250_private_borrow_semantic(scope) ? kGfx1250PrivateBorrowScratchBytes : 0;
  if (high_bank_shadow_plan)
    bytes = std::max(bytes, high_bank_shadow_private_scratch_bytes(*high_bank_shadow_plan));
  return bytes;
}

[[nodiscard]] std::optional<HighBankScratchPlan>
gfx1250_high_bank_scratch_plan(const KdTranslation &translation, uint32_t scratch_count) {
  if (scratch_count == 0)
    return std::nullopt;
  const uint32_t physical_base = std::max(translation.guest_vgpr_count, kGfx1250HighBankBaseVgpr);
  const uint32_t minimum_vgprs = physical_base + scratch_count;
  if (physical_base < kGfx1250HighBankBaseVgpr || minimum_vgprs > kRdna4MaxVgprsPerWave)
    return std::nullopt;
  return HighBankScratchPlan{static_cast<uint16_t>(physical_base - kGfx1250HighBankBaseVgpr),
                             minimum_vgprs};
}

void grow_required_vgpr_count_for_register_set(const RegisterSet &registers,
                                               uint32_t &minimum_vgprs) {
  registers.for_each([&](RegisterRef ref) {
    if (ref.cls != RegClass::VGPR)
      return;
    minimum_vgprs = std::max(minimum_vgprs, static_cast<uint32_t>(ref.index + 1u));
  });
}

void grow_required_sgpr_count_for_register_set(const RegisterSet &registers,
                                               uint32_t &minimum_sgprs) {
  registers.for_each([&](RegisterRef ref) {
    if (ref.cls != RegClass::SGPR)
      return;
    minimum_sgprs = std::max(minimum_sgprs, static_cast<uint32_t>(ref.index + 1u));
  });
}

[[nodiscard]] uint32_t gfx1250_rdna4_semantic_tmp_vgpr_count(const Instruction &inst) {
  if (inst.encoding_id() != kGfx1250Vop3pEncodingId)
    return 0;

  switch (inst.opcode()) {
  case kGfx1250VPkFmaBf16Vop3pOpcode:
    return 7;
  case kGfx1250WmmaF32F8f6f4K128Opcode:
  case kGfx1250WmmaScaleF32F8f6f4K128Opcode:
  case kGfx1250WmmaScale16F32F8f6f4K128Opcode:
    return gfx1250_wmma_f8f6f4_tmp_vgpr_count(inst);
  case kGfx1250WmmaF32F16K32Opcode:
  case kGfx1250WmmaF32Bf16K32Opcode:
    return 8;
  case kGfx1250WmmaF16F16K32Opcode:
  case kGfx1250WmmaBf16Bf16K32Opcode:
    return 4;
  case kGfx1250WmmaBf16F32Bf16K32Opcode:
    return 10;
  case kGfx1250VFmaMixloBf16Vop3pOpcode:
    return 5;
  case kGfx1250WmmaI32Iu8K64Opcode:
    // Worst-case K64 i8 lowering uses an aligned scratch accumulator plus A/B
    // relayout and lane-xor address temporaries.
    return 13;
  case kGfx1250SwmmacF32Fp8K128Opcode:
  case kGfx1250SwmmacF32Fp8K128Opcode + 1u:
  case kGfx1250SwmmacF32Fp8K128Opcode + 2u:
  case kGfx1250SwmmacF32Fp8K128Opcode + 3u:
  case kGfx1250SwmmacF32Fp8K128Opcode + 4u:
  case kGfx1250SwmmacF32Fp8K128Opcode + 5u:
  case kGfx1250SwmmacF32Fp8K128Opcode + 6u:
  case kGfx1250SwmmacF16Bf8K128LastOpcode:
    return 17;
  case kGfx1250SwmmacI32Iu8K128Opcode:
    return 15;
  case kGfx1250SwmmacF32F16K64Opcode:
  case kGfx1250SwmmacF32F16K64Opcode + 1u:
  case kGfx1250SwmmacF32F16K64Opcode + 2u:
  case kGfx1250SwmmacF32F16K64Opcode + 3u:
  case kGfx1250SwmmacBf16f32K64LastOpcode:
    return 22;
  default:
    return 0;
  }
}

void grow_gfx1250_rdna4_semantic_tmp_vgpr_count(const Instruction &inst, uint32_t &max_tmp_count) {
  const uint32_t tmp_count = gfx1250_rdna4_semantic_tmp_vgpr_count(inst);
  if (tmp_count == 0)
    return;
  max_tmp_count = std::max(max_tmp_count, tmp_count);
}

[[nodiscard]] constexpr uint32_t align_up_vgpr_count(uint32_t count, uint32_t alignment) {
  return ((count + alignment - 1u) / alignment) * alignment;
}

[[nodiscard]] uint32_t
required_vgpr_count_for_gfx1250_rdna4_scope(const KernelTranslationScope &scope) {
  uint32_t minimum_vgprs = 0;
  uint32_t semantic_tmp_vgprs = 0;
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      InstDefUse du(inst);
      grow_required_vgpr_count_for_register_set(du.defs, minimum_vgprs);
      grow_required_vgpr_count_for_register_set(du.uses, minimum_vgprs);
      grow_required_vgpr_count_for_raw_gfx1250_v_mul_u64(inst, minimum_vgprs);
      grow_gfx1250_rdna4_semantic_tmp_vgpr_count(inst, semantic_tmp_vgprs);
    }
  }
  if (semantic_tmp_vgprs != 0) {
    // Semantic expansion allocates temporary runs after liveness has reserved
    // source-level live registers. Dense IREE kernels can leave only a high run
    // available, so the launch descriptor must cover that emitted run too.
    minimum_vgprs =
        std::max(minimum_vgprs, align_up_vgpr_count(minimum_vgprs, 8) + semantic_tmp_vgprs);
  }
  return minimum_vgprs;
}

[[nodiscard]] uint32_t
required_sgpr_count_for_gfx1250_rdna4_scope(const KernelTranslationScope &scope) {
  uint32_t minimum_sgprs = 0;
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      InstDefUse du(inst);
      grow_required_sgpr_count_for_register_set(du.defs, minimum_sgprs);
      grow_required_sgpr_count_for_register_set(du.uses, minimum_sgprs);
    }
  }
  return minimum_sgprs;
}

void configure_liveness_scratch(rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                                const KernelTranslationScope &scope, LivenessAnalysis &liveness) {
  if (guest_arch != ROCJITSU_CODE_ARCH_GFX1250 || host_arch != ROCJITSU_CODE_ARCH_RDNA4 ||
      scope.translation == nullptr)
    return;
  liveness.set_allocatable_sgpr_limit(static_cast<uint16_t>(kRdna4MaxSgprsPerWave));
  const uint32_t reserved_source_sgprs = std::max(scope.translation->target_abi_sgpr_count,
                                                  scope.translation->target_source_sgpr_count);
  if (reserved_source_sgprs != 0) {
    const auto reserved_count = static_cast<uint16_t>(std::min<uint32_t>(
        reserved_source_sgprs, static_cast<uint32_t>(REGISTER_SET_ALLOCATABLE_SGPRS)));
    liveness.reserve_scratch_registers({RegClass::SGPR, 0, static_cast<uint8_t>(reserved_count)});
  }
  if (scope.translation->rdna4_grid_x_sgpr >= 0) {
    liveness.reserve_scratch_registers(
        {RegClass::SGPR, static_cast<uint16_t>(scope.translation->rdna4_grid_x_sgpr), 1});
  }
  if (scope.translation->rdna4_grid_yz_sgpr >= 0) {
    liveness.reserve_scratch_registers(
        {RegClass::SGPR, static_cast<uint16_t>(scope.translation->rdna4_grid_yz_sgpr), 1});
  }
  reserve_source_vgprs_for_relative_access(scope, liveness);
  const uint32_t scratch_count = gfx1250_high_bank_scratch_count_for_scope(scope, liveness);
  if (auto plan = gfx1250_high_bank_scratch_plan(*scope.translation, scratch_count)) {
    liveness.set_high_vgpr_scratch_base(plan->encoded_base);
  }
  if (scope.translation->private_spill_zone_bytes != 0) {
    liveness.set_private_spill_zone(scope.translation->private_spill_zone_base,
                                    scope.translation->private_spill_zone_bytes);
  }
}

void configure_shared_liveness_scratch(rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                                       std::span<const KernelTranslationScope> scopes,
                                       LivenessAnalysis &liveness) {
  if (guest_arch != ROCJITSU_CODE_ARCH_GFX1250 || host_arch != ROCJITSU_CODE_ARCH_RDNA4)
    return;

  liveness.set_allocatable_sgpr_limit(static_cast<uint16_t>(kRdna4MaxSgprsPerWave));

  uint32_t reserved_source_sgprs = 0;
  std::optional<uint16_t> high_vgpr_scratch_base;
  bool high_vgpr_scratch_conflict = false;
  std::optional<uint32_t> private_spill_base;
  uint32_t private_spill_bytes = 0;
  bool private_spill_conflict = false;

  for (const KernelTranslationScope &scope : scopes) {
    if (scope.translation == nullptr)
      continue;

    reserved_source_sgprs =
        std::max({reserved_source_sgprs, scope.translation->target_abi_sgpr_count,
                  scope.translation->target_source_sgpr_count});
    reserve_source_vgprs_for_relative_access(scope, liveness);

    if (scope.translation->rdna4_grid_x_sgpr >= 0) {
      liveness.reserve_scratch_registers(
          {RegClass::SGPR, static_cast<uint16_t>(scope.translation->rdna4_grid_x_sgpr), 1});
    }
    if (scope.translation->rdna4_grid_yz_sgpr >= 0) {
      liveness.reserve_scratch_registers(
          {RegClass::SGPR, static_cast<uint16_t>(scope.translation->rdna4_grid_yz_sgpr), 1});
    }

    const uint32_t scratch_count = gfx1250_high_bank_scratch_count_for_scope(scope, liveness);
    if (auto plan = gfx1250_high_bank_scratch_plan(*scope.translation, scratch_count)) {
      if (!high_vgpr_scratch_base) {
        high_vgpr_scratch_base = plan->encoded_base;
      } else if (*high_vgpr_scratch_base != plan->encoded_base) {
        high_vgpr_scratch_conflict = true;
      }
    }

    if (scope.translation->private_spill_zone_bytes != 0) {
      if (!private_spill_base) {
        private_spill_base = scope.translation->private_spill_zone_base;
        private_spill_bytes = scope.translation->private_spill_zone_bytes;
      } else if (*private_spill_base != scope.translation->private_spill_zone_base ||
                 private_spill_bytes != scope.translation->private_spill_zone_bytes) {
        private_spill_conflict = true;
      }
    }
  }

  if (reserved_source_sgprs != 0) {
    const auto reserved_count = static_cast<uint16_t>(std::min<uint32_t>(
        reserved_source_sgprs, static_cast<uint32_t>(REGISTER_SET_ALLOCATABLE_SGPRS)));
    liveness.reserve_scratch_registers({RegClass::SGPR, 0, static_cast<uint8_t>(reserved_count)});
  }

  if (high_vgpr_scratch_base && !high_vgpr_scratch_conflict)
    liveness.set_high_vgpr_scratch_base(*high_vgpr_scratch_base);
  if (private_spill_base && !private_spill_conflict)
    liveness.set_private_spill_zone(private_spill_base, private_spill_bytes);
}

[[nodiscard]] std::optional<HighBankBlockModeMap> merged_shared_high_bank_entry_modes(
    std::span<const KernelTranslationScope> scopes,
    const std::unordered_map<const BasicBlock *, uint32_t> &block_counts) {
  HighBankBlockModeMap merged;
  for (const KernelTranslationScope &scope : scopes) {
    if (!scope_has_shared_blocks(scope, block_counts))
      continue;
    const auto scope_modes = gfx1250_high_bank_entry_modes_for_scope(scope);
    if (!scope_modes)
      return std::nullopt;

    for (const auto &[block, mode] : *scope_modes) {
      if (!block_is_shared(block, block_counts))
        continue;
      auto [it, inserted] = merged.emplace(block, mode);
      if (!inserted && it->second != mode)
        return std::nullopt;
    }
  }
  return merged;
}

[[nodiscard]] std::vector<KernelDescriptorResourceOverride>
descriptor_resource_overrides_for_scopes(rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                                         std::span<const KernelTranslationScope> scopes) {
  std::vector<KernelDescriptorResourceOverride> overrides;
  if (guest_arch != ROCJITSU_CODE_ARCH_GFX1250 || host_arch != ROCJITSU_CODE_ARCH_RDNA4)
    return overrides;

  const uint32_t lowering_minimum_vgprs =
      conservative_lowering_minimum_vgprs(guest_arch, host_arch);
  for (const KernelTranslationScope &scope : scopes) {
    if (scope.translation == nullptr)
      continue;
    LivenessAnalysis liveness(KernelBlockScope(scope.blocks));
    uint32_t minimum_vgprs = required_vgpr_count_for_gfx1250_rdna4_scope(scope);
    if (scope_contains_gfx1250_k32_16bit_wmma(scope))
      minimum_vgprs = std::max(minimum_vgprs, kGfx1250K32WmmaScratchMinimumVgprs);
    const uint32_t minimum_sgprs = required_sgpr_count_for_gfx1250_rdna4_scope(scope);
    const uint32_t scratch_count = gfx1250_high_bank_scratch_count_for_scope(scope, liveness);
    if (const auto plan = gfx1250_high_bank_scratch_plan(*scope.translation, scratch_count)) {
      // Dense IREE kernels frequently leave no contiguous low-VGPR scratch
      // window. Grow only the kernels that need a high-bank scratch fallback.
      minimum_vgprs = std::max(minimum_vgprs, plan->minimum_vgprs);
    }
    std::optional<HighBankShadowPlan> high_bank_shadow_plan;
    if (const auto entry_modes = gfx1250_high_bank_entry_modes_for_scope(scope)) {
      const auto high_bank_shadow_analysis =
          gfx1250_high_bank_shadow_analysis_for_scope(scope, liveness, *entry_modes);
      high_bank_shadow_plan = high_bank_shadow_analysis.plan;
    }
    uint32_t target_vgpr_count_override = 0;
    if (high_bank_shadow_plan && scope.translation->guest_vgpr_count > kRdna4MaxVgprsPerWave) {
      const uint32_t shadow_window_end =
          static_cast<uint32_t>(high_bank_shadow_plan->base) + high_bank_shadow_plan->count;
      target_vgpr_count_override = std::max(minimum_vgprs, shadow_window_end);
      minimum_vgprs = std::max(minimum_vgprs, shadow_window_end);
    }
    const uint32_t private_scratch_bytes =
        gfx1250_private_scratch_bytes_for_scope(scope, high_bank_shadow_plan);
    if (minimum_vgprs > lowering_minimum_vgprs || minimum_sgprs != 0 ||
        private_scratch_bytes != 0 || target_vgpr_count_override != 0) {
      overrides.push_back({scope.translation->entry_text_offset, minimum_vgprs,
                           target_vgpr_count_override, minimum_sgprs, 0, private_scratch_bytes});
    }
  }
  return overrides;
}

[[nodiscard]] std::vector<std::pair<uint64_t, uint64_t>>
reachable_code_ranges(std::span<const KernelTranslationScope> scopes) {
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  for (const KernelTranslationScope &scope : scopes) {
    for (const BasicBlock *block : scope.blocks) {
      if (block == nullptr)
        continue;
      ranges.emplace_back(block->start_offset(), block->end_offset());
    }
  }

  std::ranges::sort(ranges);
  ranges.erase(std::ranges::unique(ranges).begin(), ranges.end());

  std::vector<std::pair<uint64_t, uint64_t>> merged;
  merged.reserve(ranges.size());
  for (const auto &[start, end] : ranges) {
    if (end <= start)
      continue;
    if (merged.empty() || start > merged.back().second) {
      merged.emplace_back(start, end);
      continue;
    }
    merged.back().second = std::max(merged.back().second, end);
  }
  return merged;
}

[[nodiscard]] uint64_t covered_range_bytes(std::span<const std::pair<uint64_t, uint64_t>> ranges) {
  uint64_t covered = 0;
  uint64_t cursor = 0;
  for (const auto &[start, end] : ranges) {
    if (end <= start)
      continue;
    const uint64_t merged_start = std::max(start, cursor);
    if (end > merged_start)
      covered += end - merged_start;
    cursor = std::max(cursor, end);
  }
  return covered;
}

[[nodiscard]] bool
has_unresolved_reachable_indirect_control_flow(std::span<const KernelTranslationScope> scopes) {
  for (const KernelTranslationScope &scope : scopes) {
    for (BasicBlock *block : scope.blocks) {
      if (block == nullptr)
        continue;
      for (const Instruction &inst : block->instructions()) {
        if ((inst.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0) {
          if ((inst.flags() & INDIRECT_BRANCH) != 0 && !block->successors().empty())
            continue;
          if ((inst.flags() & INDIRECT_CALL) != 0 && block->successors().size() > 1)
            continue;
          return true;
        }
      }
    }
  }
  return false;
}

[[nodiscard]] bool
supports_expanded_text_copy(rj_code_arch_t guest_arch, rj_code_arch_t host_arch, uint64_t text_size,
                            uint64_t protected_text_bytes,
                            std::span<const KdTranslation> descriptor_translations,
                            std::span<const KernelTranslationScope> scopes) {
  if (guest_arch == ROCJITSU_CODE_ARCH_CDNA4 && host_arch == ROCJITSU_CODE_ARCH_CDNA3)
    return true;
  if (guest_arch != ROCJITSU_CODE_ARCH_GFX1250 || host_arch != ROCJITSU_CODE_ARCH_RDNA4)
    return false;
  if (text_size <= kSoppBranchMaxForwardBytes)
    return false;
  if (has_unresolved_reachable_indirect_control_flow(scopes))
    return false;
  for (const KdTranslation &translation : descriptor_translations) {
    if (translation.prologue_words.empty())
      continue;
    int16_t branch_dwords = 0;
    if (!compute_sopp_branch_offset(translation.entry_text_offset, text_size, branch_dwords))
      return true;
  }
  if (protected_text_bytes * 2 <= text_size)
    return false;
  return true;
}

[[nodiscard]] bool has_unprotected_text(std::span<const std::pair<uint64_t, uint64_t>> ranges,
                                        uint64_t text_size) {
  uint64_t cursor = 0;
  for (const auto &[start, end] : ranges) {
    if (start > cursor)
      return true;
    cursor = std::max(cursor, end);
  }
  return cursor < text_size;
}

[[nodiscard]] std::vector<BasicBlock *>
reachable_kernel_blocks(const std::vector<std::unique_ptr<BasicBlock>> &blocks, BasicBlock &entry,
                        const std::unordered_set<uint64_t> &kernel_entries) {
  std::unordered_set<const BasicBlock *> reachable;
  std::vector<BasicBlock *> stack{&entry};

  while (!stack.empty()) {
    BasicBlock *block = stack.back();
    stack.pop_back();
    if (block == nullptr || !reachable.insert(block).second)
      continue;

    for (BasicBlock *succ : block->successors()) {
      if (succ == nullptr)
        continue;
      if (succ->start_offset() != entry.start_offset() &&
          kernel_entries.contains(succ->start_offset()))
        continue;
      stack.push_back(succ);
    }
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      BasicBlock *callee = call.callee;
      if (callee == nullptr)
        continue;
      if (callee->start_offset() != entry.start_offset() &&
          kernel_entries.contains(callee->start_offset())) {
        continue;
      }
      stack.push_back(callee);
    }
  }

  std::vector<BasicBlock *> ordered;
  ordered.reserve(reachable.size());
  for (const auto &block : blocks) {
    if (block && reachable.contains(block.get()))
      ordered.push_back(block.get());
  }
  return ordered;
}

void add_static_pc_relative_setpc_successors(
    std::vector<std::unique_ptr<BasicBlock>> &blocks,
    std::span<const StaticPcRelativeSetpcEdge> setpc_edges) {
  if (setpc_edges.empty())
    return;

  std::unordered_map<uint64_t, BasicBlock *> block_by_offset;
  block_by_offset.reserve(blocks.size());
  for (const auto &block : blocks) {
    if (block)
      block_by_offset.emplace(block->start_offset(), block.get());
  }

  const auto block_containing = [&](uint64_t offset) -> BasicBlock * {
    for (const auto &block : blocks) {
      if (block && block->start_offset() <= offset && offset < block->end_offset())
        return block.get();
    }
    return nullptr;
  };

  for (const StaticPcRelativeSetpcEdge &edge : setpc_edges) {
    BasicBlock *source = block_containing(edge.source_offset);
    const auto target_it = block_by_offset.find(edge.target_offset);
    if (source == nullptr || target_it == block_by_offset.end())
      continue;
    source->add_cfg_successor(*target_it->second);
  }
}

void add_static_pc_relative_address_successors(
    std::vector<std::unique_ptr<BasicBlock>> &blocks,
    std::span<const StaticPcRelativeAddress> pc_relative_addresses) {
  if (pc_relative_addresses.empty())
    return;

  std::unordered_map<uint64_t, BasicBlock *> block_by_offset;
  block_by_offset.reserve(blocks.size());
  for (const auto &block : blocks) {
    if (block)
      block_by_offset.emplace(block->start_offset(), block.get());
  }

  for (const StaticPcRelativeAddress &address : pc_relative_addresses) {
    BasicBlock *source = block_for_offset(blocks, address.getpc_offset);
    const auto target_it = block_by_offset.find(address.add_tmp_offset);
    if (source == nullptr || target_it == block_by_offset.end())
      continue;
    source->add_cfg_successor(*target_it->second);
  }
}

using StaticAddressTargetSet = std::set<uint64_t>;
using StaticAddressState = std::unordered_map<uint16_t, StaticAddressTargetSet>;

[[nodiscard]] bool merge_static_address_state(StaticAddressState &dst,
                                              const StaticAddressState &src) {
  bool changed = false;
  for (const auto &[pair, targets] : src) {
    if (targets.empty())
      continue;
    auto &dst_targets = dst[pair];
    const size_t old_size = dst_targets.size();
    dst_targets.insert(targets.begin(), targets.end());
    changed |= dst_targets.size() != old_size;
  }
  return changed;
}

[[nodiscard]] std::unordered_map<uint64_t, StaticAddressTargetSet>
static_call_targets_by_offset(std::vector<std::unique_ptr<BasicBlock>> &blocks,
                              std::span<const StaticPcRelativeAddress> pc_relative_addresses,
                              std::span<const StaticPcRelativeCallEdge> call_edges) {
  std::unordered_map<uint64_t, std::vector<const StaticPcRelativeAddress *>> addresses_by_offset;
  for (const StaticPcRelativeAddress &address : pc_relative_addresses) {
    if (!address.original_target_offset)
      addresses_by_offset[address.add_tmp_offset].push_back(&address);
  }

  std::unordered_map<uint64_t, std::vector<const StaticPcRelativeCallEdge *>> calls_by_offset;
  for (const StaticPcRelativeCallEdge &edge : call_edges)
    calls_by_offset[edge.call_offset].push_back(&edge);

  std::unordered_map<BasicBlock *, StaticAddressState> entry_states;
  std::vector<BasicBlock *> worklist;
  worklist.reserve(blocks.size());
  for (const auto &block : blocks) {
    if (block)
      worklist.push_back(block.get());
  }

  std::unordered_map<uint64_t, StaticAddressTargetSet> call_targets;
  while (!worklist.empty()) {
    BasicBlock *block = worklist.back();
    worklist.pop_back();
    if (block == nullptr)
      continue;

    StaticAddressState state = entry_states[block];
    uint64_t offset = block->start_offset();
    for (const Instruction &inst : block->instructions()) {
      if (const auto defs_it = addresses_by_offset.find(offset);
          defs_it != addresses_by_offset.end()) {
        for (const StaticPcRelativeAddress *address : defs_it->second) {
          if (address == nullptr)
            continue;
          auto &targets = state[address->sgpr_pair];
          targets.clear();
          targets.insert(address->target_offset);
        }
      }

      if (const auto calls_it = calls_by_offset.find(offset); calls_it != calls_by_offset.end()) {
        for (const StaticPcRelativeCallEdge *edge : calls_it->second) {
          if (edge == nullptr)
            continue;
          const auto targets_it = state.find(edge->target_sgpr_pair);
          if (targets_it == state.end())
            continue;
          auto &targets = call_targets[edge->call_offset];
          targets.insert(targets_it->second.begin(), targets_it->second.end());
        }
      }

      offset += static_cast<uint64_t>(inst.size());
    }

    for (BasicBlock *succ : block->successors()) {
      if (succ != nullptr && merge_static_address_state(entry_states[succ], state))
        worklist.push_back(succ);
    }
  }

  return call_targets;
}

void add_static_pc_relative_call_successors(
    std::vector<std::unique_ptr<BasicBlock>> &blocks,
    std::span<const StaticPcRelativeAddress> pc_relative_addresses,
    std::span<const StaticPcRelativeCallEdge> call_edges) {
  if (call_edges.empty())
    return;

  constexpr uint32_t kOpSSetPcI64 = 72;

  std::unordered_map<uint64_t, BasicBlock *> block_by_offset;
  block_by_offset.reserve(blocks.size());
  for (const auto &block : blocks) {
    if (block)
      block_by_offset.emplace(block->start_offset(), block.get());
  }

  const auto call_targets =
      static_call_targets_by_offset(blocks, pc_relative_addresses, call_edges);
  std::map<std::pair<uint64_t, uint16_t>, std::vector<BasicBlock *>> return_blocks_cache;

  auto return_blocks_for_target = [&](uint64_t target_offset,
                                      uint16_t return_pair) -> std::vector<BasicBlock *> & {
    const auto key = std::make_pair(target_offset, return_pair);
    if (auto it = return_blocks_cache.find(key); it != return_blocks_cache.end())
      return it->second;

    std::vector<BasicBlock *> return_blocks;
    const auto target_it = block_by_offset.find(target_offset);
    if (target_it != block_by_offset.end()) {
      std::vector<BasicBlock *> stack{target_it->second};
      std::unordered_set<BasicBlock *> visited;
      while (!stack.empty()) {
        BasicBlock *block = stack.back();
        stack.pop_back();
        if (block == nullptr || !visited.insert(block).second)
          continue;

        const Instruction *term = block->terminator();
        if (term != nullptr && term->size() == static_cast<int>(sizeof(uint32_t))) {
          const auto *raw = term->raw_encoding();
          if (raw != nullptr && raw[0] == pack_sop1(kOpSSetPcI64, 0, return_pair)) {
            return_blocks.push_back(block);
            continue;
          }
        }

        for (BasicBlock *succ : block->successors()) {
          if (succ != nullptr)
            stack.push_back(succ);
        }
      }
    }

    auto [it, inserted] = return_blocks_cache.emplace(key, std::move(return_blocks));
    (void)inserted;
    return it->second;
  };

  for (const StaticPcRelativeCallEdge &edge : call_edges) {
    BasicBlock *call = block_for_offset(blocks, edge.call_offset);
    const auto return_it = block_by_offset.find(edge.return_offset);
    if (return_it == block_by_offset.end())
      continue;

    const auto targets_it = call_targets.find(edge.call_offset);
    if (targets_it == call_targets.end())
      continue;

    for (uint64_t target_offset : targets_it->second) {
      const auto target_it = block_by_offset.find(target_offset);
      if (call != nullptr && target_it != block_by_offset.end())
        call->add_cfg_successor(*target_it->second);

      for (BasicBlock *return_block :
           return_blocks_for_target(target_offset, edge.return_sgpr_pair)) {
        if (return_block != nullptr)
          return_block->add_cfg_successor(*return_it->second);
      }
    }
  }
}

[[nodiscard]] std::vector<KernelTranslationScope>
kernel_translation_scopes(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                          std::span<KdTranslation> kernels) {
  std::vector<KernelTranslationScope> scopes;
  const auto entries = kernel_entry_offsets(kernels);
  if (entries.empty())
    return scopes;

  std::unordered_set<uint64_t> entry_set(entries.begin(), entries.end());
  std::vector<KdTranslation *> ordered_kernels;
  ordered_kernels.reserve(kernels.size());
  std::unordered_set<uint64_t> seen_entries;
  for (KdTranslation &kernel : kernels) {
    if (seen_entries.insert(kernel.entry_text_offset).second)
      ordered_kernels.push_back(&kernel);
  }

  std::ranges::sort(ordered_kernels, [](const auto *lhs, const auto *rhs) {
    return lhs->entry_text_offset < rhs->entry_text_offset;
  });

  scopes.reserve(ordered_kernels.size());
  for (KdTranslation *kernel : ordered_kernels) {
    BasicBlock *entry = block_for_offset(blocks, kernel->entry_text_offset);
    if (entry == nullptr)
      continue;

    scopes.push_back({kernel, entry, reachable_kernel_blocks(blocks, *entry, entry_set)});
  }
  return scopes;
}

[[nodiscard]] bool append_virtual_lds_metadata(CodeObjectPatcher &patcher,
                                               std::span<const KdTranslation> translations,
                                               std::vector<TranslationDiagnostic> &diagnostics) {
  if (!std::ranges::any_of(translations, [](const KdTranslation &translation) {
        return translation.needs_virtual_lds_buffer;
      }))
    return true;

  const auto patched_image = patcher.image_bytes();
  AmdGpuCodeObject patched_object(patched_image.data(), patched_image.size());
  if (!patched_object.is_valid()) {
    append_error(diagnostics, DiagnosticKind::ResourceLimit,
                 "translated ELF cannot be reparsed for virtual LDS metadata");
    return false;
  }
  std::vector<VirtualLdsKernelMetadata> metadata;
  for (const KdTranslation &translation : translations) {
    if (!translation.needs_virtual_lds_buffer)
      continue;
    std::string kernel_name = translation.symbol_name;
    if (kernel_name.ends_with(".kd"))
      kernel_name.resize(kernel_name.size() - 3u);
    const uint64_t descriptor_vaddr = patched_object.kernel_descriptor_offset(kernel_name);
    if (descriptor_vaddr == 0) {
      append_error(diagnostics, DiagnosticKind::KernelDescriptor,
                   "virtual LDS metadata cannot resolve kernel descriptor " + kernel_name);
      return false;
    }
    VirtualLdsKernelMetadata record{};
    record.kernel_name = std::move(kernel_name);
    record.normal_descriptor_vaddr = descriptor_vaddr;
    record.virtual_descriptor_vaddr = descriptor_vaddr;
    record.static_lds_bytes = translation.virtual_lds_size;
    record.kernarg_size = translation.kernarg_size;
    record.backing_pointer_kernarg_offset = translation.virtual_lds_kernarg_pointer_offset;
    record.virtual_lds_base_sgpr = translation.virtual_lds_lowering.base_sgpr;
    record.flags = kVirtualLdsFlagRuntimeStateBlock;
    if (translation.workgroup_id_sgpr_x >= 0)
      record.flags |= kVirtualLdsFlagWorkgroupIdX;
    if (translation.workgroup_id_sgpr_y >= 0)
      record.flags |= kVirtualLdsFlagWorkgroupIdY;
    if (translation.workgroup_id_sgpr_z >= 0)
      record.flags |= kVirtualLdsFlagWorkgroupIdZ;
    metadata.push_back(std::move(record));
  }
  const auto bytes = serialize_virtual_lds_metadata(metadata);
  if (bytes.empty() || !patcher.append_nonalloc_section(kVirtualLdsMetadataSectionName, bytes, 8)) {
    append_error(diagnostics, DiagnosticKind::ResourceLimit,
                 "virtual LDS runtime metadata could not be appended");
    return false;
  }
  return true;
}

} // namespace

BinaryTranslator::~BinaryTranslator() = default;

BinaryTranslator::BinaryTranslator(rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                                   uint32_t target_mach, BinaryTranslatorOptions options)
    : guest_arch_(guest_arch), host_arch_(host_arch),
      target_mach_(target_mach ? target_mach : elf_mach_for_arch(host_arch)), options_(options),
      encoding_translate_(select_encoding_translator(guest_arch, host_arch)),
      legalization_lookup_(select_legalization(guest_arch, host_arch)),
      semantic_translator_(std::make_unique<SemanticTranslator>(guest_arch, host_arch)) {}

void BinaryTranslator::set_trace_callback(TranslationTraceCallback callback) {
  trace_callback_ = std::move(callback);
}

std::span<const std::pair<uint64_t, uint64_t>>
BinaryTranslator::PlacementState::reserved_local_text() const {
  return local_caves;
}

bool BinaryTranslator::PlacementState::overlaps_reserved_local_text(uint64_t start,
                                                                    uint64_t end) const {
  return overlaps_any_range(start, end, reserved_local_text());
}

void BinaryTranslator::PlacementState::reserve_local_text(uint64_t start, uint64_t end) {
  const auto it = std::ranges::lower_bound(local_caves, start, {},
                                           [](const auto &range) { return range.first; });
  local_caves.insert(it, {start, end});
}

TranslatedCodeObject BinaryTranslator::translate(const AmdGpuCodeObject &obj) {
  TranslatedCodeObject result;
  result.host_arch = host_arch_;
  warnings_ = &result.warnings;
  diagnostics_ = &result.diagnostics;

  CodeObjectPatcher patcher(obj);
  auto leave_unchanged = [&]() {
    warnings_ = nullptr;
    diagnostics_ = nullptr;
    const auto *image = reinterpret_cast<const uint8_t *>(obj.image_data());
    result.elf_bytes.assign(image, image + obj.image_size());
    return result;
  };
  auto text = patcher.text_bytes();
  if (text.empty()) {
    return leave_unchanged();
  }

  auto decoder = Decoder::create(guest_arch_);
  if (!decoder) {
    append_error(result.diagnostics, DiagnosticKind::UnsupportedGuestArch,
                 "unsupported guest_arch: no decoder available");
    return leave_unchanged();
  }
  KernelDescriptorTranslator descriptor_translator(guest_arch_, host_arch_);
  KernelDescriptorTranslationOptions descriptor_options;
  // Semantic lowerings allocate temporary VGPRs from liveness. Descriptor
  // translation runs before those choices are known, so keep per-lowering
  // headroom for now.
  // TODO: Have lowerings report their actual highest temporary VGPR demand and
  // use that instead of this conservative floor.
  descriptor_options.minimum_vgprs = conservative_lowering_minimum_vgprs(guest_arch_, host_arch_);
  const bool can_virtualize_lds = supports_virtual_lds_sidecars(guest_arch_, host_arch_);
  descriptor_options.allow_oversized_lds = can_virtualize_lds;
  auto descriptor_translations = descriptor_translator.translate_image(
      patcher.image_bytes(), patcher.text_offset(), patcher.text_size(), descriptor_options);
  if (descriptor_translations.empty()) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptors are required for kernel-level translation");
    return leave_unchanged();
  }

  auto entry_offsets = kernel_entry_offsets(descriptor_translations);
  const auto static_setpc_edges = guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250
                                      ? static_pc_relative_setpc_edges(text)
                                      : std::vector<StaticPcRelativeSetpcEdge>{};
  const auto static_pc_relative_addresses = guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250
                                                ? static_pc_relative_address_edges(text)
                                                : std::vector<StaticPcRelativeAddress>{};
  const auto static_pc_relative_calls = guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250
                                            ? static_pc_relative_call_edges(text)
                                            : std::vector<StaticPcRelativeCallEdge>{};
  std::vector<ExpandedTextPcRelativeFixup> static_pc_relative_fixups;
  static_pc_relative_fixups.reserve(static_setpc_edges.size() +
                                    static_pc_relative_addresses.size());
  for (const StaticPcRelativeSetpcEdge &edge : static_setpc_edges) {
    static_pc_relative_fixups.push_back({.getpc_offset = edge.getpc_offset,
                                         .add_tmp_offset = edge.add_tmp_offset,
                                         .target_offset = edge.target_offset,
                                         .original_target_offset = std::nullopt,
                                         .kind = "setpc"});
  }
  for (const StaticPcRelativeAddress &address : static_pc_relative_addresses) {
    static_pc_relative_fixups.push_back({.getpc_offset = address.getpc_offset,
                                         .add_tmp_offset = address.add_tmp_offset,
                                         .target_offset = address.target_offset,
                                         .original_target_offset = address.original_target_offset,
                                         .kind = "address",
                                         .form = address.form,
                                         .sgpr_pair = address.sgpr_pair});
  }
  std::vector<uint64_t> block_leaders(entry_offsets.begin(), entry_offsets.end());
  block_leaders.reserve(block_leaders.size() + static_setpc_edges.size() +
                        2 * static_pc_relative_addresses.size() +
                        2 * static_pc_relative_calls.size());
  for (const StaticPcRelativeSetpcEdge &edge : static_setpc_edges)
    block_leaders.push_back(edge.target_offset);
  for (const StaticPcRelativeAddress &address : static_pc_relative_addresses) {
    block_leaders.push_back(address.add_tmp_offset);
    if (!address.original_target_offset)
      block_leaders.push_back(address.target_offset);
  }
  for (const StaticPcRelativeCallEdge &edge : static_pc_relative_calls)
    block_leaders.push_back(edge.return_offset);
  auto blocks = BasicBlock::build(obj, *decoder, guest_arch_, block_leaders);
  add_static_pc_relative_setpc_successors(blocks, static_setpc_edges);
  add_static_pc_relative_address_successors(blocks, static_pc_relative_addresses);
  add_static_pc_relative_call_successors(blocks, static_pc_relative_addresses,
                                         static_pc_relative_calls);
  auto scopes = kernel_translation_scopes(blocks, descriptor_translations);

  if (scopes.size() != entry_offsets.size()) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptor entry offsets are required to map to decoded text blocks");
    return leave_unchanged();
  }

  const auto descriptor_overrides =
      descriptor_resource_overrides_for_scopes(guest_arch_, host_arch_, scopes);
  if (!descriptor_overrides.empty()) {
    descriptor_options.kernel_overrides = std::span<const KernelDescriptorResourceOverride>(
        descriptor_overrides.data(), descriptor_overrides.size());
    descriptor_translations = descriptor_translator.translate_image(
        patcher.image_bytes(), patcher.text_offset(), patcher.text_size(), descriptor_options);
    entry_offsets = kernel_entry_offsets(descriptor_translations);
    scopes = kernel_translation_scopes(blocks, descriptor_translations);
    if (scopes.size() != entry_offsets.size()) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "kernel descriptor entry offsets are required to map to decoded text blocks");
      return leave_unchanged();
    }
  }

  // Static oversized LDS takes one path: the complete allocation is moved to
  // a global backing buffer. No prefix remains in hardware LDS.
  if (can_virtualize_lds) {
    const uint32_t host_lds_limit = arch_lds_bytes(host_arch_);
    for (KdTranslation &translation : descriptor_translations) {
      if (host_lds_limit == 0 || translation.target_lds_size <= host_lds_limit)
        continue;

      KernelDescriptorTranslationOptions virtual_options = descriptor_options;
      virtual_options.kernel_overrides = {};
      virtual_options.virtualize_lds = true;
      virtual_options.allow_oversized_lds = false;
      for (const KernelDescriptorResourceOverride &override : descriptor_overrides) {
        if (override.entry_text_offset != translation.entry_text_offset)
          continue;
        virtual_options.minimum_vgprs =
            std::max(virtual_options.minimum_vgprs, override.minimum_vgprs);
        virtual_options.target_vgpr_count_override = std::max(
            virtual_options.target_vgpr_count_override, override.target_vgpr_count_override);
        virtual_options.minimum_sgprs =
            std::max(virtual_options.minimum_sgprs, override.minimum_sgprs);
        virtual_options.group_segment_fixed_size_addend =
            std::max(virtual_options.group_segment_fixed_size_addend,
                     override.group_segment_fixed_size_addend);
        virtual_options.private_segment_fixed_size_addend =
            std::max(virtual_options.private_segment_fixed_size_addend,
                     override.private_segment_fixed_size_addend);
      }
      auto virtualized = descriptor_translator.translate_descriptor(
          patcher.image_bytes(), translation.descriptor_file_offset, translation.entry_text_offset,
          virtual_options, translation.symbol_name);
      if (!virtualized) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "full LDS virtualization descriptor could not be computed");
        return leave_unchanged();
      }
      translation = std::move(*virtualized);
    }
    scopes = kernel_translation_scopes(blocks, descriptor_translations);
  }

  bool descriptors_supported = true;
  for (const auto &translation : descriptor_translations) {
    append_diagnostics(result.diagnostics, translation.diagnostics);
    descriptors_supported &= translation.supported;
  }
  if (!descriptors_supported) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptor translation requires unsupported resource or ABI "
                 "virtualization; leaving code object unchanged");
    return leave_unchanged();
  }

  std::vector<uint8_t> translated_text(text.begin(), text.end());
  const bool continue_after_failure = options_.debug_continue_after_failure;

  auto copy_original_instruction = [&](const Instruction &inst, uint64_t offset) {
    const uint32_t inst_size = inst.size();
    std::memcpy(translated_text.data() + offset, text.data() + offset, inst_size);
    if (!trace_callback_)
      return;

    // Continued-failure mode is diagnostic-only. Emit an explicit copy event so
    // diff reports make it clear which failed source instruction was preserved.
    const auto source_words = raw_words_for_inst(inst);
    trace_callback_({.source_offset = offset,
                     .source_size = inst_size,
                     .source_words = source_words,
                     .legalization = nullptr,
                     .copied_original = true,
                     .semantic_lowering = false,
                     .changed = false,
                     .emitted_in_cave = false,
                     .target_offset = offset,
                     .target_words = source_words});
  };

  auto continue_after_instruction_error = [&](const Instruction &inst, uint64_t offset) {
    if (!continue_after_failure)
      return false;
    copy_original_instruction(inst, offset);
    return true;
  };

  const auto protected_ranges = reachable_code_ranges(scopes);
  const bool allow_unreachable_text_caves = has_unprotected_text(protected_ranges, text.size()) &&
                                            !has_unresolved_reachable_indirect_control_flow(scopes);
  CaveBranchIslandState cave_branch_island_storage;
  CaveBranchIslandState *cave_branch_islands =
      guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4
          ? &cave_branch_island_storage
          : nullptr;
  PlacementState placement;
  placement.protected_ranges = protected_ranges;
  placement.allow_unreachable_text_caves = allow_unreachable_text_caves;
  placement.cave_branch_islands = cave_branch_islands;

  // Code caves live in a separate executable section that is placed immediately
  // after the original .text bytes. Treating that section as a .text-relative
  // continuation keeps existing instruction addresses stable while avoiding any
  // dependency on compiler-emitted NOP padding after s_endpgm.
  patcher.set_cave_start(text.size());

  // Placement policy:
  // 1. Use an expanded local .text tail when the whole copied CFG can be
  //    relocated and descriptors can enter that copy.
  // 2. Otherwise preserve source addresses and let apply_semantic() choose
  //    among local padding caves, unreachable-text caves, branch islands, and
  //    appended .rj_translations fallback caves for each size-growing rewrite.
  const bool has_virtual_lds =
      std::ranges::any_of(descriptor_translations, [](const KdTranslation &translation) {
        return translation.needs_virtual_lds_buffer;
      });
  const bool cdna4_to_cdna3_relocation =
      guest_arch_ == ROCJITSU_CODE_ARCH_CDNA4 && host_arch_ == ROCJITSU_CODE_ARCH_CDNA3;
  // The CDNA compatibility path predates appended translation caves and its
  // public contract is a compact replacement .text.  GFX1250 translations, on
  // the other hand, deliberately preserve the original addresses and append a
  // relocated copy.  Keep those placement policies explicit here.
  const uint64_t expanded_text_base_bytes = cdna4_to_cdna3_relocation ? 0 : text.size();
  if (supports_expanded_text_copy(guest_arch_, host_arch_, text.size(),
                                  covered_range_bytes(protected_ranges), descriptor_translations,
                                  scopes)) {
    std::vector<uint32_t> expanded_words;
    std::unordered_map<uint64_t, uint64_t> copied_entry_offsets;
    bool expanded_copy_ok = true;

    auto fail_expanded_copy = [&](std::string message) {
      if (expanded_copy_ok && warnings_)
        warnings_->push_back(std::move(message));
      expanded_copy_ok = false;
    };

    for (const KernelTranslationScope &scope : scopes) {
      if (!expanded_copy_ok)
        break;
      if (scope.blocks.empty() || scope.translation == nullptr)
        continue;

      LivenessAnalysis liveness(KernelBlockScope(scope.blocks));
      configure_liveness_scratch(guest_arch_, host_arch_, scope, liveness);
      TranslationContext expanded_context(
          scope.translation->target_vgpr_count, scope.translation->target_agpr_count,
          scope.translation->target_accvgpr_base, scope.translation->target_sgpr_count,
          scope.translation->target_private_size);
      if (scope.translation->needs_virtual_lds_buffer) {
        auto reservation = reserve_virtual_lds_base_sgpr_pair(
            expanded_context, KernelBlockScope(scope.blocks), *scope.translation, host_arch_);
        if (!reservation) {
          fail_expanded_copy("full LDS virtualization cannot reserve its backing-buffer SGPRs");
          break;
        }
        // The backing pointer is live for the entire guest body.  Resource
        // accounting alone does not keep per-instruction semantic scratch
        // allocation from borrowing it, so make the pair unavailable to the
        // liveness allocator as well.  The prologue-only temp pair may be
        // reused after entry.
        liveness.reserve_scratch_registers({RegClass::SGPR, reservation->base, 2});
        expanded_context.virtualize_lds = true;
        expanded_context.virtual_lds_base_sgpr = reservation->base;
        expanded_context.virtual_lds_base_sgpr_spill_per_use = reservation->spill_per_use;
        expanded_context.virtual_lds_kernarg_segment_ptr_sgpr =
            scope.translation->kernarg_segment_ptr_sgpr;
        expanded_context.virtual_lds_kernarg_pointer_offset =
            scope.translation->virtual_lds_kernarg_pointer_offset;
        if (!scope.translation->virtual_lds_lowering.configured) {
          scope.translation->virtual_lds_lowering.configured = true;
          scope.translation->virtual_lds_lowering.base_sgpr = reservation->base;
          scope.translation->virtual_lds_lowering.prologue_temp_sgpr = reservation->prologue_temp;
          scope.translation->virtual_lds_lowering.base_sgpr_spill_per_use =
              reservation->spill_per_use;
          if (!append_virtual_lds_entry_prologue(*scope.translation, host_arch_)) {
            fail_expanded_copy("full LDS virtualization cannot materialize its entry prologue");
            break;
          }
        }
      }
      std::optional<HighBankBlockModeMap> high_bank_entry_modes;
      if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
        high_bank_entry_modes = gfx1250_high_bank_entry_modes_for_scope(scope);
        if (!high_bank_entry_modes) {
          fail_expanded_copy(
              "expanded text copy cannot determine consistent gfx1250 VGPR MSB mode across CFG");
          break;
        }
      }
      std::optional<HighBankShadowPlan> high_bank_shadow_plan;
      if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
        const auto high_bank_shadow_analysis =
            gfx1250_high_bank_shadow_analysis_for_scope(scope, liveness, *high_bank_entry_modes);
        if (high_bank_shadow_analysis.unsupported) {
          fail_expanded_copy("expanded text copy cannot virtualize unsupported gfx1250 VGPR MSB "
                             "mode" +
                             (high_bank_shadow_analysis.unsupported_reason.empty()
                                  ? std::string{}
                                  : ": " + high_bank_shadow_analysis.unsupported_reason));
          break;
        }
        high_bank_shadow_plan = high_bank_shadow_analysis.plan;
      }
      HighBankShadowState high_bank_shadow_state;
      std::vector<uint32_t> scope_words;
      std::vector<uint32_t> scope_word_group_sizes;
      std::vector<ExpandedTextBranchFixup> direct_branches;
      std::unordered_map<uint64_t, uint64_t> scope_offsets;
      const auto valid_call_return_offsets =
          cdna4_to_cdna3_relocation
              ? scoped_call_return_offsets(KernelBlockScope(scope.blocks), text)
              : std::unordered_set<uint64_t>{};
      std::vector<IndirectCallFixup> recovered_indirect_fixups;
      std::unordered_set<uint64_t> recovered_indirect_call_offsets;
      if (cdna4_to_cdna3_relocation) {
        for (BasicBlock *block : scope.blocks) {
          if (block == nullptr)
            continue;
          for (const IndirectCallFixup &fixup : block->static_indirect_call_fixups()) {
            recovered_indirect_fixups.push_back(fixup);
            recovered_indirect_call_offsets.insert(fixup.source_call_offset);
          }
        }
      }

      uint64_t consumed_until = 0;
      for (BasicBlock *block : scope.blocks) {
        if (!expanded_copy_ok)
          break;
        if (block == nullptr)
          continue;
        if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
          const auto mode_it = high_bank_entry_modes->find(block);
          if (mode_it == high_bank_entry_modes->end()) {
            std::ostringstream message;
            message << "expanded text copy cannot determine gfx1250 VGPR MSB mode for copied "
                       "block at offset 0x"
                    << std::hex << block->start_offset() << " in kernel scope 0x"
                    << scope.translation->entry_text_offset;
            fail_expanded_copy(message.str());
            break;
          }
          high_bank_shadow_state.mode = mode_it->second;
        } else {
          high_bank_shadow_state.mode = 0;
        }

        InstructionList &instructions = block->instructions();
        uint64_t offset = block->start_offset();
        for (auto inst_it = instructions.begin(); inst_it != instructions.end(); ++inst_it) {
          const Instruction &inst = *inst_it;
          const uint32_t inst_size = inst.size();
          if (offset < consumed_until) {
            offset += inst_size;
            continue;
          }

          const uint64_t translated_offset = scope_words.size() * sizeof(uint32_t);
          scope_offsets.emplace(offset, translated_offset);

          const auto direct_branch_delta = inst.branch_offset_bytes();
          if (cdna4_to_cdna3_relocation &&
              (inst.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0 &&
              !recovered_indirect_call_offsets.contains(offset) &&
              !valid_call_return_offsets.contains(offset) && !direct_branch_delta) {
            append_error(result.diagnostics, DiagnosticKind::Legalization,
                         "indirect branch or call target recovery is not implemented for relocated "
                         "kernel text",
                         offset, std::string(inst.mnemonic()));
            return leave_unchanged();
          }

          uint32_t source_size = inst_size;
          std::vector<uint32_t> words;
          if (words.empty() && guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 &&
              host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
            auto shadow_lowering = lower_gfx1250_high_bank_shadow_instruction(
                text, inst, offset, liveness, high_bank_shadow_plan, high_bank_shadow_state);
            if (shadow_lowering.kind == HighBankShadowLoweringKind::Unsupported) {
              fail_expanded_copy(std::move(shadow_lowering.message));
              break;
            }
            if (shadow_lowering.kind == HighBankShadowLoweringKind::RemappedGuest) {
              words = translate_remapped_guest_instruction_words(
                  inst, liveness, expanded_context, shadow_lowering.words,
                  scope.translation->rdna4_grid_x_sgpr, scope.translation->rdna4_grid_yz_sgpr);
              if (words.empty()) {
                fail_expanded_copy(
                    "expanded text copy cannot translate remapped gfx1250 high-bank instruction");
                break;
              }
              if (!shadow_lowering.prefix_words.empty() || !shadow_lowering.suffix_words.empty()) {
                std::vector<uint32_t> wrapped_words;
                wrapped_words.reserve(shadow_lowering.prefix_words.size() + words.size() +
                                      shadow_lowering.suffix_words.size());
                wrapped_words.insert(wrapped_words.end(), shadow_lowering.prefix_words.begin(),
                                     shadow_lowering.prefix_words.end());
                wrapped_words.insert(wrapped_words.end(), words.begin(), words.end());
                wrapped_words.insert(wrapped_words.end(), shadow_lowering.suffix_words.begin(),
                                     shadow_lowering.suffix_words.end());
                words = std::move(wrapped_words);
              }
              auto wrapped = wrap_high_bank_shadow_private_spills(
                  inst, std::move(words), shadow_lowering.private_loads,
                  shadow_lowering.private_stores, liveness, high_bank_shadow_plan);
              if (!wrapped) {
                fail_expanded_copy(
                    "expanded text copy cannot wrap remapped gfx1250 high-bank semantic lowering");
                break;
              }
              words = std::move(*wrapped);
            } else if (shadow_lowering.kind == HighBankShadowLoweringKind::Lowered) {
              words = std::move(shadow_lowering.words);
            }
            if (words.empty()) {
              if (auto raw_opcode = raw_vop3_opcode(read_u32(text, offset));
                  raw_opcode && is_gfx1250_vop3_compare_opcode(*raw_opcode)) {
                words = copy_instruction_words(text, offset, inst_size);
              }
            }
            if (words.empty())
              words = lower_gfx1250_s_wait_xcnt_to_rdna4(inst, host_arch_);
            if (words.empty())
              words = lower_gfx1250_smem_nv_to_rdna4(inst, host_arch_);
            if (words.empty())
              words = lower_gfx1250_contextual_raw_buffer_descriptor_mov(
                  inst_it, instructions.end(), offset, host_arch_, block, scope.blocks);
            if (words.empty())
              words = lower_gfx1250_contextual_raw_buffer_descriptor_base(
                  instructions.begin(), inst_it, instructions.end(), offset, host_arch_, block,
                  scope.blocks);
            if (words.empty())
              words = lower_gfx1250_contextual_raw_buffer_descriptor_high_mask(
                  instructions.begin(), inst_it, instructions.end(), offset, host_arch_, block,
                  scope.blocks);
            if (words.empty())
              words = lower_gfx1250_contextual_raw_buffer_descriptor_high_pack(
                  instructions.begin(), inst_it, instructions.end(), offset, host_arch_, block,
                  scope.blocks);
            if (words.empty())
              words = lower_gfx1250_contextual_raw_buffer_descriptor_lshr(
                  instructions.begin(), inst_it, instructions.end(), offset, host_arch_, block,
                  scope.blocks);
            if (words.empty())
              words = lower_gfx1250_contextual_s_and_b32_address_mask_high(instructions.begin(),
                                                                           inst_it, host_arch_);
            if (words.empty())
              words = lower_raw_gfx1250_v_mul_u64_vop3(text, offset, inst, liveness, source_size);
            if (words.empty())
              words = lower_raw_gfx1250_v_mul_u64_e32(text, offset, inst, liveness, source_size);
          }
          if (words.empty()) {
            auto virtual_lds_expansion = lower_virtual_lds_instruction(
                inst, liveness, expanded_context, guest_arch_, host_arch_);
            if (virtual_lds_expansion.status == ExpandStatus::Failed) {
              fail_expanded_copy(virtual_lds_expansion.message.empty()
                                     ? "full LDS virtualization failed"
                                     : std::move(virtual_lds_expansion.message));
              break;
            }
            if (virtual_lds_expansion.status == ExpandStatus::Success)
              words = std::move(virtual_lds_expansion.words);
          }
          if (words.empty())
            words = translate_instruction_words(inst, offset, liveness, expanded_context, text,
                                                scope.translation->rdna4_grid_x_sgpr,
                                                scope.translation->rdna4_grid_yz_sgpr);
          if (cdna4_to_cdna3_relocation && has_error_diagnostic(result.diagnostics)) {
            if (!continue_after_failure)
              return leave_unchanged();
            words = copy_instruction_words(text, offset, inst_size);
          }
          if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
            rewrite_gfx1250_zero_sgpr_v_mov_sources(words, inst, instructions.begin(), inst_it);
            rewrite_gfx1250_zero_sgpr_scalar_sources(words, inst, instructions.begin(), inst_it,
                                                     offset, scope.blocks);
            if (rdna4_adjacent_valu_dependency_delay_needed_before(instructions.begin(), inst_it)) {
              words.insert(words.begin(), build_s_delay_alu(1, host_arch_));
            }
            if (words.size() == inst_size / sizeof(uint32_t) &&
                rdna4_memory_dependency_wait_needed_before(instructions.begin(), inst_it)) {
              words.insert(words.begin(),
                           build_s_wait_alu(kWaitAluDepctrVaVdstVmVsrc0, host_arch_));
            }
            append_rdna4_scalar_dependency_barriers_after(words, inst, host_arch_);
          }

          if (source_size == inst_size) {
            if (direct_branch_delta) {
              if (words.size() != 1 || inst_size != sizeof(uint32_t)) {
                fail_expanded_copy("expanded text copy cannot relocate non-SOPP direct branch " +
                                   std::string(inst.mnemonic()));
                break;
              }
              const int64_t target = static_cast<int64_t>(offset + inst_size) +
                                     static_cast<int64_t>(*direct_branch_delta);
              if (target < 0) {
                if (cdna4_to_cdna3_relocation) {
                  append_error(result.diagnostics, DiagnosticKind::Legalization,
                               "direct branch target is outside the source .text range", offset,
                               std::string(inst.mnemonic()));
                  return leave_unchanged();
                }
                fail_expanded_copy("expanded text copy branch target is before .text for " +
                                   std::string(inst.mnemonic()));
                break;
              }
              direct_branches.push_back({scope_words.size(), static_cast<uint64_t>(target),
                                         std::string(inst.mnemonic()),
                                         liveness.find_free_sgpr_pair(&inst)});
            }
          }

          if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
            const size_t group_start = scope_words.size();
            scope_word_group_sizes.resize(group_start + words.size(), 0);
            if (!words.empty())
              scope_word_group_sizes[group_start] = static_cast<uint32_t>(words.size());
          }
          scope_words.insert(scope_words.end(), words.begin(), words.end());
          consumed_until = offset + source_size;
          offset += inst_size;
        }
      }

      if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
        const size_t old_word_count = scope_words.size();
        if (scope_word_group_sizes.size() != old_word_count)
          scope_word_group_sizes.assign(old_word_count, 1);
        std::vector<size_t> old_to_new(old_word_count + 1);
        std::vector<uint32_t> rewritten_words;
        rewritten_words.reserve(scope_words.size());

        for (size_t word_index = 0; word_index < old_word_count;) {
          old_to_new[word_index] = rewritten_words.size();
          const size_t group_words =
              scope_word_group_sizes[word_index] == 0 ? 1 : scope_word_group_sizes[word_index];
          if (group_words >= 2 && word_index + group_words <= old_word_count) {
            uint32_t consumed_size = 0;
            const uint32_t available_size = static_cast<uint32_t>(group_words * sizeof(uint32_t));
            auto expansion = lower_raw_gfx1250_v_mul_u64_vop3(scope_words.data() + word_index,
                                                              available_size, &consumed_size);
            const size_t consumed_words = consumed_size / sizeof(uint32_t);
            if (!expansion.empty() && consumed_words != 0 && consumed_words <= group_words) {
              rewritten_words.insert(rewritten_words.end(), expansion.begin(), expansion.end());
              for (size_t consumed = 1; consumed < consumed_words; ++consumed)
                old_to_new[word_index + consumed] = rewritten_words.size();
              for (size_t remaining = consumed_words; remaining < group_words; ++remaining) {
                old_to_new[word_index + remaining] = rewritten_words.size();
                rewritten_words.push_back(scope_words[word_index + remaining]);
              }
              word_index += group_words;
              continue;
            }
          }

          for (size_t group_offset = 0;
               group_offset < group_words && word_index + group_offset < old_word_count;
               ++group_offset) {
            old_to_new[word_index + group_offset] = rewritten_words.size();
            rewritten_words.push_back(scope_words[word_index + group_offset]);
          }
          word_index += group_words;
        }
        old_to_new[old_word_count] = rewritten_words.size();

        if (rewritten_words.size() != scope_words.size()) {
          for (auto &[_, translated_offset] : scope_offsets) {
            const size_t old_word_index = translated_offset / sizeof(uint32_t);
            translated_offset = old_to_new[old_word_index] * sizeof(uint32_t);
          }
          for (ExpandedTextBranchFixup &branch : direct_branches)
            branch.word_index = old_to_new[branch.word_index];
          scope_words = std::move(rewritten_words);
        }
      }

      auto relocate_direct_branches = [&]() {
        if (direct_branches.empty())
          return true;

        if (cdna4_to_cdna3_relocation) {
          for (const ExpandedTextBranchFixup &branch : direct_branches) {
            if (!scope_offsets.contains(branch.target_offset)) {
              append_error(result.diagnostics, DiagnosticKind::Legalization,
                           "direct branch target is not present in the kernel-local relocated "
                           "body",
                           std::nullopt, branch.mnemonic);
              return false;
            }
          }
        }

        std::vector<std::pair<uint64_t, uint64_t>> offset_map;
        offset_map.reserve(scope_offsets.size());
        for (const auto &[source_offset, translated_offset] : scope_offsets)
          offset_map.emplace_back(source_offset, translated_offset);

        auto relocation = relocate_expanded_text_branches(
            {.words = scope_words, .branches = direct_branches, .offset_map = offset_map});
        if (!relocation.success) {
          fail_expanded_copy(std::move(relocation.message));
          return false;
        }
        scope_words = std::move(relocation.words);
        scope_offsets.clear();
        for (const auto &[source_offset, translated_offset] : relocation.offset_map)
          scope_offsets.emplace(source_offset, translated_offset);
        return true;
      };

      if (!relocate_direct_branches()) {
        if (cdna4_to_cdna3_relocation && has_error_diagnostic(result.diagnostics))
          return leave_unchanged();
        break;
      }
      if (!expanded_copy_ok)
        break;

      if (cdna4_to_cdna3_relocation) {
        std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> rewritten_regions;
        for (const IndirectCallFixup &fixup : recovered_indirect_fixups) {
          const auto getpc_it = scope_offsets.find(fixup.source_getpc_offset);
          const auto recovery_begin_it = scope_offsets.find(fixup.source_recovery_begin_offset);
          const auto recovery_end_it = scope_offsets.find(fixup.source_recovery_end_offset);
          const auto target_it = scope_offsets.find(fixup.source_target_offset);
          if (getpc_it == scope_offsets.end() || recovery_begin_it == scope_offsets.end() ||
              recovery_end_it == scope_offsets.end() || target_it == scope_offsets.end()) {
            append_error(result.diagnostics, DiagnosticKind::Legalization,
                         "recovered indirect call target is not present in the kernel-local "
                         "relocated body",
                         fixup.source_call_offset, "indirect branch");
            return leave_unchanged();
          }

          const auto rewrite_key =
              std::pair{recovery_end_it->second, static_cast<uint64_t>(target_it->second)};
          auto [rewrite_it, inserted] =
              rewritten_regions.emplace(recovery_begin_it->second, rewrite_key);
          if (!inserted) {
            if (rewrite_it->second != rewrite_key) {
              append_error(result.diagnostics, DiagnosticKind::Legalization,
                           "recovered indirect branch builder is reused for incompatible targets",
                           fixup.source_call_offset, "indirect branch");
              return leave_unchanged();
            }
            continue;
          }

          const int64_t delta = static_cast<int64_t>(target_it->second) -
                                static_cast<int64_t>(getpc_it->second + sizeof(uint32_t));
          std::vector<uint32_t> replacement_words;
          if (!append_pc_delta_builder(replacement_words, host_arch_, fixup.source_call_sreg,
                                       delta)) {
            append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                         "target ISA cannot encode canonical recovered indirect branch builder",
                         fixup.source_call_offset, "indirect branch");
            return leave_unchanged();
          }

          const uint64_t recovery_size = recovery_end_it->second - recovery_begin_it->second;
          const uint64_t replacement_size = replacement_words.size() * sizeof(uint32_t);
          if (replacement_size > recovery_size) {
            append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                         "recovered indirect branch builder does not fit in its source range",
                         fixup.source_call_offset, "indirect branch");
            return leave_unchanged();
          }

          const size_t begin_word = recovery_begin_it->second / sizeof(uint32_t);
          std::copy(replacement_words.begin(), replacement_words.end(),
                    scope_words.begin() + begin_word);
          for (uint64_t byte = replacement_size; byte < recovery_size; byte += sizeof(uint32_t))
            scope_words[begin_word + byte / sizeof(uint32_t)] = build_s_nop(0, host_arch_);
        }
      }

      const uint32_t required_vgprs =
          std::max(scope.translation->target_vgpr_count, expanded_context.required_vgpr_count);
      const uint32_t required_sgprs =
          std::max(scope.translation->target_sgpr_count, expanded_context.required_sgpr_count);
      if (required_vgprs != scope.translation->target_vgpr_count ||
          required_sgprs != scope.translation->target_sgpr_count) {
        KernelDescriptorTranslationOptions updated_options;
        updated_options.minimum_vgprs =
            std::max(conservative_lowering_minimum_vgprs(guest_arch_, host_arch_), required_vgprs);
        updated_options.minimum_sgprs = required_sgprs;
        updated_options.virtualize_lds = scope.translation->needs_virtual_lds_buffer;
        if (const auto override_it = std::ranges::find_if(
                descriptor_overrides,
                [&](const KernelDescriptorResourceOverride &override) {
                  return override.entry_text_offset == scope.translation->entry_text_offset;
                });
            override_it != descriptor_overrides.end()) {
          updated_options.minimum_vgprs =
              std::max(updated_options.minimum_vgprs, override_it->minimum_vgprs);
          updated_options.target_vgpr_count_override = override_it->target_vgpr_count_override;
          updated_options.minimum_sgprs =
              std::max(updated_options.minimum_sgprs, override_it->minimum_sgprs);
          updated_options.group_segment_fixed_size_addend =
              override_it->group_segment_fixed_size_addend;
          updated_options.private_segment_fixed_size_addend =
              override_it->private_segment_fixed_size_addend;
        }

        auto updated = descriptor_translator.translate_descriptor(
            patcher.image_bytes(), scope.translation->descriptor_file_offset,
            scope.translation->entry_text_offset, updated_options, scope.translation->symbol_name);
        if (!updated) {
          fail_expanded_copy(
              "expanded text copy could not recompute descriptor scratch requirements");
          break;
        }
        append_diagnostics(result.diagnostics, updated->diagnostics);
        if (!updated->supported) {
          fail_expanded_copy(
              "expanded text copy requires unsupported descriptor scratch resources");
          break;
        }
        if (scope.translation->needs_virtual_lds_buffer) {
          updated->virtual_lds_lowering = scope.translation->virtual_lds_lowering;
          if (!append_virtual_lds_entry_prologue(*updated, host_arch_)) {
            fail_expanded_copy(
                "expanded text copy could not recompute its full-LDS virtualization prologue");
            break;
          }
        }
        *scope.translation = std::move(*updated);
      }

      const auto entry_it = scope_offsets.find(scope.translation->entry_text_offset);
      if (entry_it == scope_offsets.end()) {
        fail_expanded_copy("expanded text copy is missing a translated kernel entry");
        break;
      }

      auto relocate_static_pc_relative_fixups = [&](uint64_t scope_base) {
        if (static_pc_relative_fixups.empty())
          return true;

        std::vector<std::pair<uint64_t, uint64_t>> offset_map;
        offset_map.reserve(scope_offsets.size());
        for (const auto &[source_offset, translated_offset] : scope_offsets)
          offset_map.emplace_back(source_offset, translated_offset);

        auto relocation =
            relocate_expanded_text_pc_relative_fixups({.words = scope_words,
                                                       .offset_map = offset_map,
                                                       .fixups = static_pc_relative_fixups,
                                                       .original_text_size_bytes = text.size(),
                                                       .scope_base_bytes = scope_base});
        if (!relocation.success) {
          fail_expanded_copy(std::move(relocation.message));
          return false;
        }
        return true;
      };

      if (cdna4_to_cdna3_relocation && scope.translation->has_kernarg_preload) {
        const uint64_t launch_stub_bytes =
            (scope.translation->prologue_words.size() + 1) * sizeof(uint32_t);
        if (launch_stub_bytes > kKernargPreloadSkipBytes) {
          append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                       "kernel descriptor prologue does not fit in the 256-byte kernarg preload "
                       "compatibility window; leaving code object unchanged",
                       scope.translation->entry_text_offset);
          return leave_unchanged();
        }

        const auto preload_it =
            scope_offsets.find(scope.translation->kernarg_preload_entry_text_offset);
        if (preload_it == scope_offsets.end()) {
          append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                       "kernarg preload firmware entry offset is not present in the relocated "
                       "body",
                       scope.translation->kernarg_preload_entry_text_offset);
          return leave_unchanged();
        }

        const uint64_t current_bytes = expanded_words.size() * sizeof(uint32_t);
        const uint64_t entry_residue = scope.translation->entry_text_offset % 256;
        const uint64_t launch_padding = (entry_residue + 256 - current_bytes % 256) % 256;
        expanded_words.insert(expanded_words.end(), launch_padding / sizeof(uint32_t),
                              build_s_nop(0, host_arch_));
        const uint64_t launch_offset = expanded_words.size() * sizeof(uint32_t);
        const uint64_t body_offset = launch_offset + kKernargPreloadSkipBytes + launch_stub_bytes;
        expanded_words.resize(body_offset / sizeof(uint32_t), build_s_nop(0, host_arch_));
        if (!relocate_static_pc_relative_fixups(body_offset))
          break;
        expanded_words.insert(expanded_words.end(), scope_words.begin(), scope_words.end());

        auto write_launch_stub = [&](uint64_t stub_offset, uint64_t target_offset) {
          const size_t stub_word = stub_offset / sizeof(uint32_t);
          std::copy(scope.translation->prologue_words.begin(),
                    scope.translation->prologue_words.end(), expanded_words.begin() + stub_word);
          const uint64_t branch_pc =
              stub_offset + scope.translation->prologue_words.size() * sizeof(uint32_t);
          int16_t branch_dwords = 0;
          if (!compute_sopp_branch_offset(branch_pc, target_offset, branch_dwords))
            return false;
          expanded_words[branch_pc / sizeof(uint32_t)] = build_s_branch(branch_dwords, host_arch_);
          return true;
        };

        const uint64_t body_entry = body_offset + entry_it->second;
        const uint64_t preload_body_entry = body_offset + preload_it->second;
        if (!write_launch_stub(launch_offset, body_entry) ||
            !write_launch_stub(launch_offset + kKernargPreloadSkipBytes, preload_body_entry)) {
          append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                       "kernel descriptor prologue branch range exceeds s_branch simm16; leaving "
                       "code object unchanged",
                       scope.translation->entry_text_offset);
          return leave_unchanged();
        }

        copied_entry_offsets.emplace(scope.translation->entry_text_offset, launch_offset);
        scope.translation->target_entry_text_offset = launch_offset;
        scope.translation->target_body_entry_text_offset = body_entry;
        continue;
      }

      std::optional<uint16_t> prologue_long_branch_pair;
      std::optional<uint16_t> prologue_long_branch_scc;
      if (!scope.translation->prologue_words.empty()) {
        uint32_t scratch_floor = scope.translation->target_source_sgpr_count;
        if (scope.translation->rdna4_grid_x_sgpr >= 0)
          scratch_floor = std::max(scratch_floor,
                                   static_cast<uint32_t>(scope.translation->rdna4_grid_x_sgpr + 1));
        if (scope.translation->rdna4_grid_yz_sgpr >= 0)
          scratch_floor = std::max(
              scratch_floor, static_cast<uint32_t>(scope.translation->rdna4_grid_yz_sgpr + 1));
        const uint32_t scratch_pair = align_up_vgpr_count(scratch_floor, 2);
        if (scratch_pair + 2u < scope.translation->target_sgpr_count && scratch_pair + 2u <= 105u) {
          prologue_long_branch_pair = static_cast<uint16_t>(scratch_pair);
          prologue_long_branch_scc = static_cast<uint16_t>(scratch_pair + 2u);
        }
      }

      const ExpandedTextScopePlacementRequest placement_request{
          .original_text_size_bytes = expanded_text_base_bytes,
          .current_tail_size_bytes = expanded_words.size() * sizeof(uint32_t),
          .original_entry_offset = scope.translation->entry_text_offset,
          .translated_entry_offset = entry_it->second,
          .prologue_size_bytes = scope.translation->prologue_words.size() * sizeof(uint32_t),
          .long_branch_sgpr_pair = prologue_long_branch_pair,
          .long_branch_scc_sgpr = prologue_long_branch_scc,
      };
      const auto scope_placement = plan_expanded_text_scope_placement(placement_request);
      if (!scope_placement) {
        fail_expanded_copy("expanded text copy could not place copied kernel entry");
        break;
      }

      if (scope.translation->prologue_words.empty()) {
        expanded_words.insert(expanded_words.end(),
                              scope_placement->padding_bytes / sizeof(uint32_t),
                              build_s_nop(0, host_arch_));

        const uint64_t scope_base = scope_placement->body_offset;
        assert(expanded_words.size() * sizeof(uint32_t) == scope_base &&
               "expanded scope body placement mismatch");
        if (!relocate_static_pc_relative_fixups(scope_base))
          break;
        expanded_words.insert(expanded_words.end(), scope_words.begin(), scope_words.end());
        copied_entry_offsets.emplace(scope.translation->entry_text_offset,
                                     scope_placement->descriptor_entry_offset);
        scope.translation->target_entry_text_offset = scope_placement->descriptor_entry_offset;
        scope.translation->target_body_entry_text_offset = scope_base + entry_it->second;
      } else {
        expanded_words.insert(expanded_words.end(),
                              scope_placement->padding_bytes / sizeof(uint32_t),
                              build_s_nop(0, host_arch_));

        assert(expanded_words.size() * sizeof(uint32_t) == *scope_placement->launch_stub_offset &&
               "expanded launch stub placement mismatch");
        std::vector<uint32_t> launch_stub(scope.translation->prologue_words.begin(),
                                          scope.translation->prologue_words.end());
        launch_stub.insert(launch_stub.end(), scope_placement->prologue_branch_words.begin(),
                           scope_placement->prologue_branch_words.end());

        expanded_words.insert(expanded_words.end(), launch_stub.begin(), launch_stub.end());
        const uint64_t scope_base = scope_placement->body_offset;
        assert(expanded_words.size() * sizeof(uint32_t) == scope_base &&
               "expanded launch stub size mismatch");
        if (!relocate_static_pc_relative_fixups(scope_base))
          break;
        expanded_words.insert(expanded_words.end(), scope_words.begin(), scope_words.end());
        copied_entry_offsets.emplace(scope.translation->entry_text_offset,
                                     scope_placement->descriptor_entry_offset);
        scope.translation->target_entry_text_offset = scope_placement->descriptor_entry_offset;
        scope.translation->target_body_entry_text_offset = scope_base + entry_it->second;
      }
    }

    if (continue_after_failure && has_error_diagnostic(result.diagnostics))
      return leave_unchanged();

    if (expanded_copy_ok) {
      if (cdna4_to_cdna3_relocation) {
        while (expanded_words.size() * sizeof(uint32_t) < text.size())
          expanded_words.push_back(build_s_nop(0, host_arch_));
        translated_text.clear();
      }
      if (!expanded_words.empty()) {
        const auto *expanded_bytes = reinterpret_cast<const uint8_t *>(expanded_words.data());
        translated_text.insert(translated_text.end(), expanded_bytes,
                               expanded_bytes + expanded_words.size() * sizeof(uint32_t));
      }

      std::vector<ExpandedTextDescriptorEntry> descriptor_entries;
      descriptor_entries.reserve(descriptor_translations.size());
      std::unordered_map<uint64_t, const KdTranslation *> descriptor_by_file_offset;
      for (const KdTranslation &translation : descriptor_translations) {
        descriptor_entries.push_back({.descriptor_file_offset = translation.descriptor_file_offset,
                                      .entry_text_offset = translation.entry_text_offset});
        descriptor_by_file_offset.emplace(translation.descriptor_file_offset, &translation);
      }
      std::vector<std::pair<uint64_t, uint64_t>> copied_entry_map;
      copied_entry_map.reserve(copied_entry_offsets.size());
      for (const auto &[entry_text_offset, copied_entry_offset] : copied_entry_offsets)
        copied_entry_map.emplace_back(entry_text_offset, copied_entry_offset);

      const auto descriptor_plan = plan_expanded_text_descriptor_redirections(
          descriptor_entries, copied_entry_map, expanded_text_base_bytes);
      if (!descriptor_plan.success) {
        result.warnings.push_back(descriptor_plan.message);
        return leave_unchanged();
      }
      for (const ExpandedTextDescriptorRedirection &redirection : descriptor_plan.redirections) {
        const auto descriptor_it =
            descriptor_by_file_offset.find(redirection.descriptor_file_offset);
        if (descriptor_it == descriptor_by_file_offset.end() || descriptor_it->second == nullptr) {
          result.warnings.push_back("expanded text copy could not map a kernel descriptor entry; "
                                    "leaving code object unchanged");
          return leave_unchanged();
        }
        const KdTranslation &translation = *descriptor_it->second;
        KdTranslation descriptor_patch = translation;
        descriptor_patch.prologue_words.clear();
        if (!patcher.apply_kernel_descriptor_translation(descriptor_patch, host_arch_) ||
            !patcher.redirect_kernel_entry(redirection.descriptor_file_offset,
                                           redirection.original_entry_offset,
                                           redirection.redirected_entry_offset)) {
          result.warnings.push_back("kernel descriptor translation could not be applied safely; "
                                    "leaving code object unchanged");
          return leave_unchanged();
        }
      }

      if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4)
        nop_untranslated_gfx1250_vgpr_msb_zero_resets(translated_text, host_arch_);

      if (!patcher.replace_text(translated_text)) {
        result.warnings.push_back(
            "expanded text copy could not be materialized safely; leaving code object unchanged");
        return leave_unchanged();
      }

      if (const uint32_t metadata_vgprs =
              metadata_vgpr_count_for_in_place_patch(descriptor_translations);
          metadata_vgprs != 0 && !patcher.patch_metadata_vgpr_count(metadata_vgprs)) {
        result.warnings.push_back("AMDGPU metadata VGPR count could not be patched safely; leaving "
                                  "code object unchanged");
        return leave_unchanged();
      }
      if (const uint32_t metadata_sgprs = metadata_sgpr_count_for_patch(descriptor_translations);
          metadata_sgprs != 0 && !patcher.patch_metadata_sgpr_count(metadata_sgprs)) {
        result.warnings.push_back("AMDGPU metadata SGPR count could not be patched safely; leaving "
                                  "code object unchanged");
        return leave_unchanged();
      }
      if (needs_metadata_private_segment_patch(descriptor_translations) &&
          !patcher.patch_metadata_private_segment_fixed_sizes(descriptor_translations)) {
        result.warnings.push_back(
            "AMDGPU metadata private segment size could not be patched safely; leaving code object "
            "unchanged");
        return leave_unchanged();
      }

      if (!patch_translated_metadata_target_isa(patcher, guest_arch_, host_arch_)) {
        result.warnings.push_back("AMDGPU metadata target ISA could not be patched safely; leaving "
                                  "code object unchanged");
        return leave_unchanged();
      }
      if (target_mach_)
        patcher.update_elf_flags(target_mach_);
      if (!append_virtual_lds_metadata(patcher, descriptor_translations, result.diagnostics))
        return leave_unchanged();

      result.elf_bytes = patcher.emit();
      warnings_ = nullptr;
      diagnostics_ = nullptr;
      return result;
    }
  }

  // The CDNA4-to-RDNA4 in-place path materializes descriptor ABI prologues in
  // the appended translation section and returns to the original kernel entry
  // with a SOPP branch.  Reject an image whose shortest possible return already
  // exceeds simm16 instead of emitting a descriptor that cannot launch safely.
  if (guest_arch_ == ROCJITSU_CODE_ARCH_CDNA4 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
    for (const KdTranslation &translation : descriptor_translations) {
      if (translation.prologue_words.empty())
        continue;
      const uint64_t branch_pc = text.size() + translation.prologue_words.size() * sizeof(uint32_t);
      int16_t branch_dwords = 0;
      if (!compute_sopp_branch_offset(branch_pc, translation.entry_text_offset, branch_dwords)) {
        append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                     "kernel descriptor prologue branch range exceeds s_branch simm16; leaving "
                     "code object unchanged",
                     translation.entry_text_offset);
        return leave_unchanged();
      }
    }
  }

  auto make_liveness_options = [&]() {
    LivenessAnalysisOptions liveness_options;
    if (options_.debug_min_free_vgpr)
      liveness_options.min_free_vgpr = *options_.debug_min_free_vgpr;
    return liveness_options;
  };

  const auto block_reach_counts = scope_reach_count_by_block(scopes);
  const auto shared_blocks = unique_blocks_for_scopes(scopes, true, block_reach_counts);
  const bool has_shared_blocks = !shared_blocks.empty();
  const auto all_scope_blocks = has_shared_blocks
                                    ? unique_blocks_for_scopes(scopes, false, block_reach_counts)
                                    : std::vector<BasicBlock *>{};
  std::unique_ptr<LivenessAnalysis> shared_liveness;
  std::optional<HighBankBlockModeMap> shared_high_bank_entry_modes;
  std::optional<HighBankShadowPlan> shared_high_bank_shadow_plan;
  TranslationContext shared_context;
  if (has_shared_blocks) {
    shared_liveness = std::make_unique<LivenessAnalysis>(KernelBlockScope(all_scope_blocks),
                                                         make_liveness_options());
    configure_shared_liveness_scratch(guest_arch_, host_arch_, scopes, *shared_liveness);
    shared_context = merged_translation_context_for_scopes(scopes);
    if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
      shared_high_bank_entry_modes =
          merged_shared_high_bank_entry_modes(scopes, block_reach_counts);
      if (shared_high_bank_entry_modes) {
        KernelTranslationScope shared_scope{nullptr, nullptr, shared_blocks};
        const auto high_bank_shadow_analysis = gfx1250_high_bank_shadow_analysis_for_scope(
            shared_scope, *shared_liveness, *shared_high_bank_entry_modes);
        if (high_bank_shadow_analysis.unsupported) {
          result.warnings.push_back("in-place translation cannot virtualize unsupported gfx1250 "
                                    "VGPR MSB mode" +
                                    (high_bank_shadow_analysis.unsupported_reason.empty()
                                         ? std::string{}
                                         : ": " + high_bank_shadow_analysis.unsupported_reason));
          return leave_unchanged();
        }
        shared_high_bank_shadow_plan = high_bank_shadow_analysis.plan;
      }
    }
  }

  std::unordered_set<const BasicBlock *> translated_blocks;
  for (const KernelTranslationScope &scope : scopes) {
    if (scope.blocks.empty())
      continue;

    TranslationContext kernel_context(
        scope.translation->target_vgpr_count, scope.translation->target_agpr_count,
        scope.translation->target_accvgpr_base, scope.translation->target_sgpr_count,
        scope.translation->target_private_size);
    LivenessAnalysis liveness(KernelBlockScope(scope.blocks), make_liveness_options());
    configure_liveness_scratch(guest_arch_, host_arch_, scope, liveness);
    if (scope.translation->needs_virtual_lds_buffer) {
      if (scope_has_shared_blocks(scope, block_reach_counts)) {
        append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                     "full LDS virtualization does not support a CFG block shared with another "
                     "kernel entry",
                     scope.translation->entry_text_offset);
        return leave_unchanged();
      }
      auto reservation = reserve_virtual_lds_base_sgpr_pair(
          kernel_context, KernelBlockScope(scope.blocks), *scope.translation, host_arch_);
      if (!reservation) {
        append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                     "full LDS virtualization cannot reserve its backing-buffer SGPRs",
                     scope.translation->entry_text_offset);
        return leave_unchanged();
      }
      // The backing pointer remains live until s_endpgm.  Reserve it in the
      // scratch allocator as well as in descriptor resource accounting.
      liveness.reserve_scratch_registers({RegClass::SGPR, reservation->base, 2});
      kernel_context.virtualize_lds = true;
      kernel_context.virtual_lds_base_sgpr = reservation->base;
      kernel_context.virtual_lds_base_sgpr_spill_per_use = reservation->spill_per_use;
      kernel_context.virtual_lds_kernarg_segment_ptr_sgpr =
          scope.translation->kernarg_segment_ptr_sgpr;
      kernel_context.virtual_lds_kernarg_pointer_offset =
          scope.translation->virtual_lds_kernarg_pointer_offset;
      if (!scope.translation->virtual_lds_lowering.configured) {
        scope.translation->virtual_lds_lowering.configured = true;
        scope.translation->virtual_lds_lowering.base_sgpr = reservation->base;
        scope.translation->virtual_lds_lowering.prologue_temp_sgpr = reservation->prologue_temp;
        scope.translation->virtual_lds_lowering.base_sgpr_spill_per_use =
            reservation->spill_per_use;
        if (!append_virtual_lds_entry_prologue(*scope.translation, host_arch_)) {
          append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                       "full LDS virtualization cannot materialize its entry prologue",
                       scope.translation->entry_text_offset);
          return leave_unchanged();
        }
      }
    }
    std::optional<HighBankBlockModeMap> high_bank_entry_modes;
    std::optional<HighBankShadowPlan> high_bank_shadow_plan;
    if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
      high_bank_entry_modes = gfx1250_high_bank_entry_modes_for_scope(scope);
      if (high_bank_entry_modes) {
        const auto high_bank_shadow_analysis =
            gfx1250_high_bank_shadow_analysis_for_scope(scope, liveness, *high_bank_entry_modes);
        if (high_bank_shadow_analysis.unsupported) {
          result.warnings.push_back("in-place translation cannot virtualize unsupported gfx1250 "
                                    "VGPR MSB mode" +
                                    (high_bank_shadow_analysis.unsupported_reason.empty()
                                         ? std::string{}
                                         : ": " + high_bank_shadow_analysis.unsupported_reason));
          return leave_unchanged();
        }
        high_bank_shadow_plan = high_bank_shadow_analysis.plan;
      }
    }

    for (BasicBlock *block : scope.blocks) {
      if (block == nullptr)
        continue;
      const bool shared_block = block_is_shared(block, block_reach_counts);
      if (!translated_blocks.insert(block).second)
        continue;

      LivenessAnalysis &active_liveness =
          shared_block && shared_liveness ? *shared_liveness : liveness;
      TranslationContext &active_context =
          shared_block && shared_liveness ? shared_context : kernel_context;
      const std::span<BasicBlock *const> active_scope_blocks =
          shared_block && shared_liveness
              ? std::span<BasicBlock *const>(all_scope_blocks.data(), all_scope_blocks.size())
              : std::span<BasicBlock *const>(scope.blocks.data(), scope.blocks.size());
      const std::optional<HighBankBlockModeMap> &active_high_bank_entry_modes =
          shared_block && shared_liveness ? shared_high_bank_entry_modes : high_bank_entry_modes;
      const std::optional<HighBankShadowPlan> &active_high_bank_shadow_plan =
          shared_block && shared_liveness ? shared_high_bank_shadow_plan : high_bank_shadow_plan;

      HighBankShadowState high_bank_shadow_state;
      if (active_high_bank_entry_modes) {
        const auto mode_it = active_high_bank_entry_modes->find(block);
        if (mode_it == active_high_bank_entry_modes->end()) {
          result.warnings.push_back(
              "in-place translation cannot determine gfx1250 VGPR MSB mode for block");
          return leave_unchanged();
        }
        high_bank_shadow_state.mode = mode_it->second;
      }

      InstructionList &instructions = block->instructions();
      CaveChainState cave_chain;
      uint64_t offset = block->start_offset();
      for (auto it = instructions.begin(); it != instructions.end(); ++it) {
        const auto &inst = *it;
        const uint32_t inst_size = inst.size();
        if (placement.overlaps_reserved_local_text(offset, offset + inst_size)) {
          offset += inst_size;
          continue;
        }

        const uint32_t *raw = inst.raw_encoding();
        if (!raw) {
          std::memcpy(translated_text.data() + offset, text.data() + offset, inst_size);
          if (trace_callback_) {
            trace_callback_({.source_offset = offset,
                             .source_size = inst_size,
                             .source_words = {},
                             .legalization = nullptr,
                             .copied_original = true,
                             .semantic_lowering = false,
                             .changed = false,
                             .emitted_in_cave = false,
                             .target_offset = offset,
                             .target_words = {}});
          }
          offset += inst_size;
          continue;
        }

        auto require_long_return_sgprs = [&](std::optional<uint16_t> sgpr_pair,
                                             std::optional<uint16_t> scc_sgpr) {
          if (sgpr_pair)
            active_context.require_sgprs(static_cast<uint32_t>(*sgpr_pair) + 2u);
          if (scc_sgpr)
            active_context.require_sgprs(static_cast<uint32_t>(*scc_sgpr) + 1u);
        };

        auto apply_semantic_for_inst = [&](const SemanticReplacement &repl) {
          const auto long_return_sgpr_pair = active_liveness.find_free_sgpr_pair(&inst);
          const auto long_return_scc_sgpr =
              find_long_return_scc_sgpr(active_liveness, inst, long_return_sgpr_pair);
          require_long_return_sgprs(long_return_sgpr_pair, long_return_scc_sgpr);
          placement.long_return_sgpr_pair = long_return_sgpr_pair;
          placement.long_return_scc_sgpr = long_return_scc_sgpr;
          placement.cave_chain = &cave_chain;
          return apply_semantic(repl, translated_text, patcher, placement);
        };

        if (active_high_bank_shadow_plan) {
          auto shadow_lowering = lower_gfx1250_high_bank_shadow_instruction(
              text, inst, offset, active_liveness, active_high_bank_shadow_plan,
              high_bank_shadow_state);
          if (shadow_lowering.kind == HighBankShadowLoweringKind::Unsupported) {
            result.warnings.push_back(std::move(shadow_lowering.message));
            return leave_unchanged();
          }
          if (shadow_lowering.kind == HighBankShadowLoweringKind::RemappedGuest) {
            auto words = translate_remapped_guest_instruction_words(
                inst, active_liveness, active_context, shadow_lowering.words,
                scope.translation->rdna4_grid_x_sgpr, scope.translation->rdna4_grid_yz_sgpr);
            if (words.empty()) {
              result.warnings.push_back(
                  "in-place translation cannot translate remapped gfx1250 high-bank instruction");
              return leave_unchanged();
            }
            if (!shadow_lowering.prefix_words.empty() || !shadow_lowering.suffix_words.empty()) {
              std::vector<uint32_t> wrapped_words;
              wrapped_words.reserve(shadow_lowering.prefix_words.size() + words.size() +
                                    shadow_lowering.suffix_words.size());
              wrapped_words.insert(wrapped_words.end(), shadow_lowering.prefix_words.begin(),
                                   shadow_lowering.prefix_words.end());
              wrapped_words.insert(wrapped_words.end(), words.begin(), words.end());
              wrapped_words.insert(wrapped_words.end(), shadow_lowering.suffix_words.begin(),
                                   shadow_lowering.suffix_words.end());
              words = std::move(wrapped_words);
            }
            auto wrapped = wrap_high_bank_shadow_private_spills(
                inst, std::move(words), shadow_lowering.private_loads,
                shadow_lowering.private_stores, active_liveness, active_high_bank_shadow_plan);
            if (!wrapped) {
              result.warnings.push_back(
                  "in-place translation cannot wrap remapped gfx1250 high-bank semantic lowering");
              return leave_unchanged();
            }
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(*wrapped)};
            if (!apply_semantic_for_inst(repl))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
          if (shadow_lowering.kind == HighBankShadowLoweringKind::Lowered) {
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(shadow_lowering.words)};
            if (!apply_semantic_for_inst(repl))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
        }

        if (!active_high_bank_shadow_plan && active_high_bank_entry_modes &&
            guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4 &&
            inst_size == sizeof(uint32_t)) {
          uint16_t simm16 = 0;
          if (is_gfx1250_s_set_vgpr_msb(raw[0], simm16)) {
            high_bank_shadow_state.mode = amdgpu::s_set_vgpr_msb_new_mode(simm16);
            fill_nops(translated_text, offset, inst_size, host_arch_);
            offset += inst_size;
            continue;
          }
        }

        if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
          auto contextual = lower_gfx1250_s_wait_xcnt_to_rdna4(inst, host_arch_);
          if (!contextual.empty()) {
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(contextual)};
            if (!apply_semantic_for_inst(repl))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
          contextual = lower_gfx1250_smem_nv_to_rdna4(inst, host_arch_);
          if (!contextual.empty()) {
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(contextual)};
            if (!apply_semantic_for_inst(repl))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
          contextual = lower_gfx1250_contextual_raw_buffer_descriptor_mov(
              it, instructions.end(), offset, host_arch_, block, active_scope_blocks);
          if (!contextual.empty()) {
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(contextual)};
            if (!apply_semantic_for_inst(repl))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
          contextual = lower_gfx1250_contextual_raw_buffer_descriptor_base(
              instructions.begin(), it, instructions.end(), offset, host_arch_, block,
              active_scope_blocks);
          if (!contextual.empty()) {
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(contextual)};
            if (!apply_semantic_for_inst(repl))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
          contextual = lower_gfx1250_contextual_raw_buffer_descriptor_high_mask(
              instructions.begin(), it, instructions.end(), offset, host_arch_, block,
              active_scope_blocks);
          if (!contextual.empty()) {
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(contextual)};
            if (!apply_semantic_for_inst(repl))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
          contextual = lower_gfx1250_contextual_raw_buffer_descriptor_high_pack(
              instructions.begin(), it, instructions.end(), offset, host_arch_, block,
              active_scope_blocks);
          if (!contextual.empty()) {
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(contextual)};
            if (!apply_semantic_for_inst(repl))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
          contextual = lower_gfx1250_contextual_raw_buffer_descriptor_lshr(
              instructions.begin(), it, instructions.end(), offset, host_arch_, block,
              active_scope_blocks);
          if (!contextual.empty()) {
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(contextual)};
            if (!apply_semantic_for_inst(repl))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
          contextual = lower_gfx1250_contextual_s_and_b32_address_mask_high(instructions.begin(),
                                                                            it, host_arch_);
          if (!contextual.empty()) {
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(contextual)};
            if (!apply_semantic_for_inst(repl))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
        }

        const InstructionLegalization *leg = nullptr;
        if (legalization_lookup_)
          leg = legalization_lookup_(inst.encoding_id(), inst.opcode());

        const uint16_t dst_opcode = leg ? leg->target_opcode : inst.opcode();

        {
          auto expansion = lower_virtual_lds_instruction(inst, active_liveness, active_context,
                                                         guest_arch_, host_arch_);
          if (expansion.status == ExpandStatus::Failed) {
            append_error(result.diagnostics, DiagnosticKind::ExpandFailed,
                         expansion.message.empty() ? "full LDS virtualization failed"
                                                   : expansion.message,
                         offset, std::string(inst.mnemonic()), std::move(expansion.required_work));
            return leave_unchanged();
          }
          if (expansion.status == ExpandStatus::Success) {
            const bool emitted_in_cave = expansion.words.size() * sizeof(uint32_t) > inst_size;
            const uint64_t target_offset =
                emitted_in_cave ? patcher.cave_start() + patcher.cave_body_size() : offset;
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(expansion.words)};
            if (!apply_semantic_for_inst(repl))
              return leave_unchanged();
            if (trace_callback_) {
              trace_callback_({.source_offset = offset,
                               .source_size = inst_size,
                               .source_words = raw_words_for_inst(inst),
                               .legalization = leg,
                               .copied_original = false,
                               .semantic_lowering = true,
                               .changed = true,
                               .emitted_in_cave = emitted_in_cave,
                               .target_offset = target_offset,
                               .target_words = repl.target_words});
            }
            offset += inst_size;
            continue;
          }
        }

        // Try semantic lowering before raw encoding translation. A matched
        // semantic rule that cannot safely emit code is a translation error:
        // falling through would silently preserve guest semantics on the wrong
        // host ISA.
        {
          auto expansion = semantic_translator_->try_lower_expand(inst, offset, text,
                                                                  active_liveness, active_context);
          if (expansion.status == ExpandStatus::Failed) {
            append_error(result.diagnostics, DiagnosticKind::ExpandFailed,
                         expansion.message.empty()
                             ? "semantic EXPAND rule matched, but could not safely lower"
                             : expansion.message,
                         offset, std::string(inst.mnemonic()), std::move(expansion.required_work));
            if (continue_after_instruction_error(inst, offset)) {
              offset += inst_size;
              continue;
            }
            return leave_unchanged();
          }

          if (expansion.status == ExpandStatus::Success) {
            if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 &&
                host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
              rewrite_gfx1250_zero_sgpr_v_mov_sources(expansion.words, inst, instructions.begin(),
                                                      it);
              rewrite_gfx1250_zero_sgpr_scalar_sources(expansion.words, inst, instructions.begin(),
                                                       it, offset, active_scope_blocks);
            }
            append_hardware_pending_warning(&result.warnings, inst.mnemonic());
            const bool emitted_in_cave = expansion.words.size() * sizeof(uint32_t) > inst_size;
            const uint64_t target_offset =
                emitted_in_cave ? patcher.cave_start() + patcher.cave_body_size() : offset;
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(expansion.words)};
            if (!apply_semantic_for_inst(repl)) {
              if (continue_after_instruction_error(inst, offset)) {
                offset += inst_size;
                continue;
              }
              return leave_unchanged();
            }
            if (trace_callback_) {
              const auto source_words = raw_words_for_inst(inst);
              trace_callback_({.source_offset = offset,
                               .source_size = inst_size,
                               .source_words = source_words,
                               .legalization = leg,
                               .copied_original = false,
                               .semantic_lowering = true,
                               .changed = true,
                               .emitted_in_cave = emitted_in_cave,
                               .target_offset = target_offset,
                               .target_words = repl.target_words});
            }
            offset += inst_size;
            continue;
          }
        }

        if (leg && leg->action == Action::Expand) {
          if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
            result.warnings.push_back("EXPAND not yet implemented for " +
                                      std::string(inst.mnemonic()));
            fill_nops(translated_text, offset, inst_size, host_arch_);
            offset += inst_size;
            continue;
          }

          append_error(result.diagnostics, DiagnosticKind::ExpandMissing,
                       "legalization requires EXPAND, but no expansion rule is implemented", offset,
                       std::string(inst.mnemonic()),
                       {"Add a semantic expansion rule for this mnemonic."});
          if (continue_after_instruction_error(inst, offset)) {
            offset += inst_size;
            continue;
          }
          return leave_unchanged();
        }

        if (requires_semantic_expansion(guest_arch_, inst)) {
          result.warnings.push_back("EXPAND not yet implemented for " +
                                    std::string(inst.mnemonic()));
          fill_nops(translated_text, offset, inst_size, host_arch_);
          offset += inst_size;
          continue;
        }

        const auto long_return_sgpr_pair = active_liveness.find_free_sgpr_pair(&inst);
        const auto long_return_scc_sgpr =
            find_long_return_scc_sgpr(active_liveness, inst, long_return_sgpr_pair);
        require_long_return_sgprs(long_return_sgpr_pair, long_return_scc_sgpr);
        placement.long_return_sgpr_pair = long_return_sgpr_pair;
        placement.long_return_scc_sgpr = long_return_scc_sgpr;
        placement.cave_chain = &cave_chain;
        if (!handle_encoding(inst, offset, translated_text, dst_opcode, patcher, text, placement,
                             scope.translation ? scope.translation->rdna4_grid_x_sgpr : -1,
                             scope.translation ? scope.translation->rdna4_grid_yz_sgpr : -1,
                             instructions.begin(), it, active_scope_blocks)) {
          if (continue_after_instruction_error(inst, offset)) {
            offset += inst_size;
            continue;
          }
          return leave_unchanged();
        }
        offset += inst_size;
      }
    }

    if (continue_after_failure && has_error_diagnostic(result.diagnostics))
      continue;

    if (scope_has_shared_blocks(scope, block_reach_counts))
      merge_translation_context_requirements(kernel_context, shared_context);

    if (kernel_context.required_vgpr_count > kernel_context.num_vgprs)
      scope.translation->target_vgpr_count = kernel_context.required_vgpr_count;
    if (kernel_context.required_sgpr_count > kernel_context.num_sgprs)
      scope.translation->target_sgpr_count = kernel_context.required_sgpr_count;

    if (scope.translation->target_vgpr_count != kernel_context.num_vgprs ||
        scope.translation->target_sgpr_count != kernel_context.num_sgprs) {
      // Semantic rules may allocate descriptor-backed scratch registers beyond
      // the kernel's original SGPR/VGPR counts. Recompute the descriptor with
      // those larger minimums before patching it into the output image.
      KernelDescriptorTranslationOptions descriptor_options;
      descriptor_options.minimum_vgprs = scope.translation->target_vgpr_count;
      descriptor_options.minimum_sgprs = scope.translation->target_sgpr_count;
      descriptor_options.virtualize_lds = scope.translation->needs_virtual_lds_buffer;

      // Descriptor growth is intentionally done after instruction lowering so
      // each kernel is translated once. Only descriptors that enter this code
      // scope need the larger floor; rescanning the whole image would also
      // recompute unrelated kernels and risks mixing diagnostics across scopes.
      bool recomputed_descriptor = false;
      for (KdTranslation &translation : descriptor_translations) {
        if (translation.entry_text_offset != scope.translation->entry_text_offset)
          continue;

        auto updated = descriptor_translator.translate_descriptor(
            patcher.image_bytes(), translation.descriptor_file_offset,
            translation.entry_text_offset, descriptor_options, translation.symbol_name);
        if (!updated) {
          append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                       "kernel descriptor translation could not be recomputed; leaving code object "
                       "unchanged");
          return leave_unchanged();
        }

        if (translation.needs_virtual_lds_buffer) {
          updated->virtual_lds_lowering = translation.virtual_lds_lowering;
          if (!append_virtual_lds_entry_prologue(*updated, host_arch_)) {
            append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                         "full LDS virtualization entry prologue could not be recomputed");
            return leave_unchanged();
          }
        }

        append_diagnostics(result.diagnostics, updated->diagnostics);
        if (!updated->supported) {
          append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                       "kernel descriptor translation requires unsupported resource or ABI "
                       "virtualization; leaving code object unchanged");
          return leave_unchanged();
        }

        translation = std::move(*updated);
        recomputed_descriptor = true;
      }

      if (!recomputed_descriptor) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "kernel descriptor translation could not be recomputed; leaving code object "
                     "unchanged");
        return leave_unchanged();
      }
    }
  }

  if (continue_after_failure && has_error_diagnostic(result.diagnostics))
    return leave_unchanged();

  std::unordered_set<uint64_t> applied_descriptors;
  for (const KdTranslation &translation : descriptor_translations) {
    if (applied_descriptors.insert(translation.descriptor_file_offset).second) {
      if (!patcher.apply_kernel_descriptor_translation(translation, host_arch_)) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "kernel descriptor translation could not be applied safely; leaving code "
                     "object unchanged");
        return leave_unchanged();
      }
    }
  }

  if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4)
    nop_untranslated_gfx1250_vgpr_msb_zero_resets(translated_text, host_arch_);
  patcher.overwrite_text(translated_text);
  if (!patcher.append_cave_section()) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 "code cave section could not be materialized safely; leaving code object "
                 "unchanged");
    return leave_unchanged();
  }

  if (const uint32_t metadata_vgprs =
          metadata_vgpr_count_for_in_place_patch(descriptor_translations);
      metadata_vgprs != 0 && !patcher.patch_metadata_vgpr_count(metadata_vgprs)) {
    result.warnings.push_back(
        "AMDGPU metadata VGPR count could not be patched safely; leaving code object unchanged");
    return leave_unchanged();
  }
  if (const uint32_t metadata_sgprs = metadata_sgpr_count_for_patch(descriptor_translations);
      metadata_sgprs != 0 && !patcher.patch_metadata_sgpr_count(metadata_sgprs)) {
    result.warnings.push_back(
        "AMDGPU metadata SGPR count could not be patched safely; leaving code object unchanged");
    return leave_unchanged();
  }
  if (needs_metadata_private_segment_patch(descriptor_translations) &&
      !patcher.patch_metadata_private_segment_fixed_sizes(descriptor_translations)) {
    result.warnings.push_back("AMDGPU metadata private segment size could not be patched safely; "
                              "leaving code object unchanged");
    return leave_unchanged();
  }

  if (!patch_translated_metadata_target_isa(patcher, guest_arch_, host_arch_)) {
    result.warnings.push_back("AMDGPU metadata target ISA could not be patched safely; leaving "
                              "code object unchanged");
    return leave_unchanged();
  }
  if (target_mach_)
    patcher.update_elf_flags(target_mach_);

  if (has_virtual_lds &&
      !append_virtual_lds_metadata(patcher, descriptor_translations, result.diagnostics))
    return leave_unchanged();

  warnings_ = nullptr;
  diagnostics_ = nullptr;
  result.elf_bytes = patcher.emit();
  return result;
}

std::vector<uint32_t> BinaryTranslator::translate_instruction_words(
    const Instruction &inst, uint64_t offset, const LivenessAnalysis &liveness,
    TranslationContext &context, std::span<const uint8_t> orig_text, int16_t rdna4_grid_x_sgpr,
    int16_t rdna4_grid_yz_sgpr) {
  const uint32_t inst_size = inst.size();
  const uint32_t *raw = inst.raw_encoding();
  auto finish_words = [&](std::vector<uint32_t> words) {
    if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4)
      remap_gfx1250_ttmp_grid_reads_to_sgpr(inst, rdna4_grid_x_sgpr, rdna4_grid_yz_sgpr, words);
    return words;
  };
  if (!raw)
    return finish_words(copy_instruction_words(orig_text, offset, inst_size));

  const uint32_t w0 = read_u32(orig_text, offset);
  const uint32_t w1 = inst_size > 4 ? read_u32(orig_text, offset + sizeof(uint32_t)) : 0;
  const uint32_t w2 = inst_size > 8 ? read_u32(orig_text, offset + 2 * sizeof(uint32_t)) : 0;

  if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4 &&
      encoding_translate_) {
    if (auto raw_opcode = raw_vop3_opcode(w0);
        raw_opcode && is_gfx1250_vop3_compare_opcode(*raw_opcode)) {
      std::vector<uint32_t> words{w0, w1};
      if (inst_size > 2 * sizeof(uint32_t))
        words.push_back(w2);
      return finish_words(std::move(words));
    }
  }

  if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
    uint32_t source_size = inst_size;
    auto raw_expansion =
        lower_raw_gfx1250_v_mul_u64_vop3(orig_text, offset, inst, liveness, source_size);
    if (raw_expansion.empty())
      raw_expansion =
          lower_raw_gfx1250_v_mul_u64_e32(orig_text, offset, inst, liveness, source_size);
    if (!raw_expansion.empty())
      return finish_words(std::move(raw_expansion));
  }

  uint16_t source_opcode = inst.opcode();
  if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
    if (auto raw_opcode = raw_vop3_opcode(w0))
      source_opcode = *raw_opcode;
  }

  const InstructionLegalization *leg = nullptr;
  if (legalization_lookup_)
    leg = legalization_lookup_(inst.encoding_id(), source_opcode);

  const uint16_t dst_opcode = leg ? leg->target_opcode : source_opcode;

  auto expansion = semantic_translator_->try_lower_expand_with_opcode(
      inst, offset, orig_text, liveness, context, source_opcode);
  if (expansion.status == ExpandStatus::Failed) {
    if (diagnostics_)
      append_error(*diagnostics_, DiagnosticKind::ExpandFailed,
                   expansion.message.empty()
                       ? "semantic EXPAND rule matched, but could not safely lower"
                       : expansion.message,
                   offset, std::string(inst.mnemonic()), std::move(expansion.required_work));
    if (options_.debug_continue_after_failure)
      return finish_words(copy_instruction_words(orig_text, offset, inst_size));
    return {};
  }
  if (expansion.status == ExpandStatus::Success) {
    append_hardware_pending_warning(warnings_, inst.mnemonic());
    return finish_words(std::move(expansion.words));
  }

  if (leg && leg->action == Action::Expand) {
    if (diagnostics_)
      append_error(*diagnostics_, DiagnosticKind::ExpandMissing,
                   "legalization requires EXPAND, but no expansion rule is implemented", offset,
                   std::string(inst.mnemonic()),
                   {"Add a semantic expansion rule for this mnemonic."});
    if (options_.debug_continue_after_failure)
      return finish_words(copy_instruction_words(orig_text, offset, inst_size));
    return {};
  }

  if (requires_semantic_expansion(guest_arch_, inst)) {
    if (diagnostics_)
      append_error(*diagnostics_, DiagnosticKind::ExpandMissing,
                   "legalization requires EXPAND, but no expansion rule is implemented", offset,
                   std::string(inst.mnemonic()),
                   {"Add a semantic expansion rule for this mnemonic."});
    if (options_.debug_continue_after_failure)
      return finish_words(copy_instruction_words(orig_text, offset, inst_size));
    return {};
  }

  if (!encoding_translate_)
    return finish_words(copy_instruction_words(orig_text, offset, inst_size));

  auto tr = encoding_translate_(inst.encoding_id(), w0, w1, w2, dst_opcode);
  if (tr.word_count == 0)
    return finish_words(copy_instruction_words(orig_text, offset, inst_size));

  const uint32_t translated_bytes = tr.word_count * sizeof(uint32_t);
  if (inst_size >= translated_bytes && inst_size - translated_bytes == sizeof(uint32_t) &&
      tr.word_count < 3) {
    uint32_t lit_word = 0;
    std::memcpy(&lit_word, orig_text.data() + offset + translated_bytes, sizeof(lit_word));
    tr.words[tr.word_count++] = lit_word;
  }

  return finish_words({tr.words, tr.words + tr.word_count});
}

std::vector<uint32_t> BinaryTranslator::translate_remapped_guest_instruction_words(
    const Instruction &inst, LivenessAnalysis &liveness, TranslationContext &context,
    std::span<const uint32_t> guest_words, int16_t rdna4_grid_x_sgpr, int16_t rdna4_grid_yz_sgpr) {
  if (guest_words.empty())
    return {};

  auto decoder = Decoder::create(guest_arch_);
  if (!decoder)
    return {};

  std::unique_ptr<Instruction> remapped(
      decoder->decode(reinterpret_cast<const rj_code_binary_inst_t *>(guest_words.data())));
  if (!remapped || remapped->size() != static_cast<int>(guest_words.size() * sizeof(uint32_t)))
    return {};
  if (!liveness.alias_live_before(*remapped, inst))
    return {};

  const auto *bytes = reinterpret_cast<const uint8_t *>(guest_words.data());
  const std::span<const uint8_t> remapped_text(bytes, guest_words.size() * sizeof(uint32_t));
  auto virtual_lds =
      lower_virtual_lds_instruction(*remapped, liveness, context, guest_arch_, host_arch_);
  if (virtual_lds.status == ExpandStatus::Success)
    return std::move(virtual_lds.words);
  if (virtual_lds.status == ExpandStatus::Failed)
    return {};
  return translate_instruction_words(*remapped, 0, liveness, context, remapped_text,
                                     rdna4_grid_x_sgpr, rdna4_grid_yz_sgpr);
}

bool BinaryTranslator::apply_semantic(const SemanticReplacement &repl, std::vector<uint8_t> &text,
                                      CodeObjectPatcher &patcher, PlacementState &placement) {
  assert(repl.matched() && "apply_semantic called with unmatched replacement");
  assert(repl.start_offset < repl.end_offset && "invalid replacement range");
  assert(repl.end_offset <= text.size() && "replacement exceeds text bounds");

  const uint32_t source_size = repl.end_offset - repl.start_offset;
  const uint32_t target_size = repl.target_words.size() * 4;

  if (target_size <= source_size) {
    std::memcpy(text.data() + repl.start_offset, repl.target_words.data(), target_size);
    if (target_size < source_size)
      fill_nops(text, repl.start_offset + target_size, source_size - target_size, host_arch_);
    append_placement_diagnostic(warnings_, "selected", PlacementTier::InPlace, repl);
    return true;
  }

  const uint64_t stub_next = repl.start_offset + source_size;
  const uint64_t branch_pc = repl.start_offset;

  auto patch_source_branch = [&](int16_t fwd_dwords) {
    const uint32_t stub = build_s_branch(fwd_dwords, host_arch_);
    std::memcpy(text.data() + repl.start_offset, &stub, sizeof(stub));
    const uint32_t nop = build_s_nop(0, host_arch_);
    for (uint64_t off = repl.start_offset + sizeof(uint32_t); off < repl.end_offset;
         off += sizeof(uint32_t))
      std::memcpy(text.data() + off, &nop, sizeof(nop));
  };

  const uint64_t local_cave_size = target_size + sizeof(uint32_t);
  const LocalTextCaveRequest local_cave_request{
      .source = {.start = repl.start_offset, .end = repl.end_offset},
      .body_size_bytes = target_size,
      .cave_size_bytes = local_cave_size,
  };

  auto apply_local_cave = [&](const BranchableTextPlacement &local_cave, PlacementTier tier) {
    auto cave_words = repl.target_words;
    cave_words.push_back(build_s_branch(local_cave.exit_branch_dwords, host_arch_));

    // An active appended-cave chain may later absorb the source bytes between
    // its return target and another remote rewrite.  A local detour mutates
    // those bytes into a branch stub whose return lands back in the original
    // text; copying that stub into the appended chain would leave the chain and
    // resume in source bytes that the extension has replaced with NOPs.
    // Terminate the chain at the local detour so a later remote rewrite starts
    // a fresh chain instead of bridging across non-linear control flow.
    if (placement.cave_chain != nullptr)
      *placement.cave_chain = {};

    patch_source_branch(local_cave.entry_branch_dwords);
    std::memcpy(text.data() + local_cave.offset, cave_words.data(),
                cave_words.size() * sizeof(uint32_t));
    placement.reserve_local_text(local_cave.offset,
                                 local_cave.offset + cave_words.size() * sizeof(uint32_t));
    std::ostringstream detail;
    detail << "local_cave_offset=0x" << std::hex << local_cave.offset;
    append_placement_diagnostic(warnings_, "selected", tier, repl, detail.str());
  };

  if (auto local_cave_offset =
          find_local_text_cave(text, local_cave_request, placement.reserved_local_text(),
                               placement.protected_ranges, false)) {
    apply_local_cave(*local_cave_offset, PlacementTier::LocalPaddingCave);
    return true;
  }
  if (placement.allow_unreachable_text_caves) {
    if (auto local_cave_offset =
            find_local_text_cave(text, local_cave_request, placement.reserved_local_text(),
                                 placement.protected_ranges, true)) {
      apply_local_cave(*local_cave_offset, PlacementTier::LocalUnreachableTextCave);
      return true;
    }
  }

  if (placement.cave_branch_islands != nullptr)
    ensure_cave_branch_islands(patcher, *placement.cave_branch_islands, host_arch_);

  const uint64_t cave_byte_offset = patcher.cave_start() + patcher.cave_body_size();

  auto make_return_trailer = [&](uint64_t return_pc, uint64_t target) -> std::vector<uint32_t> {
    int16_t short_ret_dwords = 0;
    if (compute_sopp_branch_offset(return_pc, target, short_ret_dwords))
      return {build_s_branch(short_ret_dwords, host_arch_)};
    if (!placement.long_return_sgpr_pair)
      return {};
    if (!placement.long_return_scc_sgpr)
      return {};
    return build_s_setpc_long_branch_preserving_scc(
        return_pc, target, *placement.long_return_sgpr_pair, *placement.long_return_scc_sgpr);
  };

  auto record_cave_chain = [&](uint64_t return_target, uint64_t trailer_body_offset) {
    if (placement.cave_chain == nullptr)
      return;
    placement.cave_chain->active = true;
    placement.cave_chain->return_target = return_target;
    placement.cave_chain->trailer_body_offset = trailer_body_offset;
  };

  // s_branch simm16 targets (PC + 4 + simm16*4).
  int16_t fwd_dwords = 0;
  if (!compute_sopp_branch_offset(branch_pc, cave_byte_offset, fwd_dwords)) {
    if (placement.cave_chain != nullptr && placement.cave_chain->active &&
        placement.cave_chain->return_target <= repl.start_offset) {
      const uint64_t bridge_size = repl.start_offset - placement.cave_chain->return_target;
      const uint64_t chained_body_size = bridge_size + target_size;
      const uint64_t trailer_body_offset =
          placement.cave_chain->trailer_body_offset + chained_body_size;
      const uint64_t return_branch_pc = patcher.cave_start() + trailer_body_offset;
      auto return_trailer = make_return_trailer(return_branch_pc, stub_next);
      if (!return_trailer.empty()) {
        std::vector<uint32_t> bridge_words;
        if (bridge_size != 0) {
          assert(bridge_size % sizeof(uint32_t) == 0 && "bridge must be word-aligned");
          bridge_words.resize(bridge_size / sizeof(uint32_t));
          std::memcpy(bridge_words.data(), text.data() + placement.cave_chain->return_target,
                      bridge_size);
        }

        patcher.truncate_cave_body(placement.cave_chain->trailer_body_offset);
        fill_nops(text, placement.cave_chain->return_target,
                  static_cast<uint32_t>(bridge_size + source_size), host_arch_);
        auto cave_words = repl.target_words;
        cave_words.insert(cave_words.begin(), bridge_words.begin(), bridge_words.end());
        cave_words.insert(cave_words.end(), return_trailer.begin(), return_trailer.end());
        patcher.append_cave_body(cave_words);
        record_cave_chain(stub_next, trailer_body_offset);
        append_placement_diagnostic(warnings_, "selected", PlacementTier::ChainedAppendedCave,
                                    repl);
        return true;
      }
    }

    auto apply_forward_branch_island = [&](bool allow_unreachable_text,
                                           PlacementTier tier) -> bool {
      const uint64_t return_branch_pc =
          cave_byte_offset + repl.target_words.size() * sizeof(uint32_t);
      auto return_trailer = make_return_trailer(return_branch_pc, stub_next);
      if (return_trailer.empty())
        return false;

      const LocalBranchIslandRequest branch_island_request{
          .source = {.start = repl.start_offset, .end = repl.end_offset},
          .island_target = cave_byte_offset,
      };
      auto island_offset =
          find_local_branch_island(text, branch_island_request, placement.reserved_local_text(),
                                   placement.protected_ranges, allow_unreachable_text);
      if (!island_offset)
        return false;

      patch_source_branch(island_offset->entry_branch_dwords);
      const uint32_t island_branch = build_s_branch(island_offset->exit_branch_dwords, host_arch_);
      std::memcpy(text.data() + island_offset->offset, &island_branch, sizeof(island_branch));
      placement.reserve_local_text(island_offset->offset,
                                   island_offset->offset + sizeof(island_branch));

      auto cave_words = repl.target_words;
      cave_words.insert(cave_words.end(), return_trailer.begin(), return_trailer.end());
      patcher.append_cave_body(cave_words);
      record_cave_chain(stub_next,
                        patcher.cave_body_size() - return_trailer.size() * sizeof(uint32_t));
      std::ostringstream detail;
      detail << "island_offset=0x" << std::hex << island_offset->offset;
      append_placement_diagnostic(warnings_, "selected", tier, repl, detail.str());
      return true;
    };

    if (apply_forward_branch_island(false, PlacementTier::LocalPaddingBranchIsland))
      return true;
    if (placement.allow_unreachable_text_caves &&
        apply_forward_branch_island(true, PlacementTier::LocalUnreachableBranchIsland))
      return true;

    auto apply_cave_branch_chain = [&]() -> bool {
      if (placement.cave_branch_islands == nullptr)
        return false;

      const uint64_t branch_target = patcher.cave_start() + patcher.cave_body_size();
      const uint64_t return_branch_pc = branch_target + repl.target_words.size() * sizeof(uint32_t);
      auto return_trailer = make_return_trailer(return_branch_pc, stub_next);
      int16_t island_entry_dwords = 0;
      if (!return_trailer.empty() && placement.long_return_sgpr_pair &&
          allocate_cave_long_branch_island(patcher, *placement.cave_branch_islands, repl,
                                           branch_target, host_arch_,
                                           *placement.long_return_sgpr_pair, island_entry_dwords)) {
        patch_source_branch(island_entry_dwords);

        auto cave_words = repl.target_words;
        cave_words.insert(cave_words.end(), return_trailer.begin(), return_trailer.end());
        patcher.append_cave_body(cave_words);
        record_cave_chain(stub_next,
                          patcher.cave_body_size() - return_trailer.size() * sizeof(uint32_t));
        append_placement_diagnostic(warnings_, "selected",
                                    PlacementTier::AppendedCaveLongBranchIsland, repl);
        return true;
      }

      if (return_trailer.empty() ||
          !allocate_cave_branch_chain(patcher, *placement.cave_branch_islands, repl, branch_target,
                                      host_arch_, island_entry_dwords)) {
        // A newly appended island block would begin at branch_target. If the
        // existing island graph cannot reach that address, adding slots at or
        // after it cannot make the source branch reachable and only grows the
        // code object. Fail immediately so the caller can select another
        // placement strategy.
        return false;
      }

      patch_source_branch(island_entry_dwords);
      auto cave_words = repl.target_words;
      cave_words.insert(cave_words.end(), return_trailer.begin(), return_trailer.end());
      patcher.append_cave_body(cave_words);
      record_cave_chain(stub_next,
                        patcher.cave_body_size() - return_trailer.size() * sizeof(uint32_t));
      append_placement_diagnostic(warnings_, "selected", PlacementTier::AppendedCaveBranchChain,
                                  repl);
      return true;
    };

    if (apply_cave_branch_chain())
      return true;

    const bool can_defer = has_unimplemented_expand_gap(warnings_);
    if (warnings_) {
      warnings_->push_back(
          can_defer ? "code cave branch range exceeds s_branch simm16; leaving source instruction "
                      "unchanged"
                    : "code cave branch range exceeds s_branch simm16; leaving code object "
                      "unchanged");
      if (cave_diagnostics_enabled())
        warnings_->push_back(cave_range_diagnostic("code cave branch range diagnostic", repl,
                                                   cave_byte_offset, target_size));
      append_placement_diagnostic(warnings_, "failed", PlacementTier::AppendedCave, repl,
                                  "reason=source-branch-out-of-range");
    }
    if (can_defer)
      return true;
    if (diagnostics_)
      append_error(*diagnostics_, DiagnosticKind::ResourceLimit,
                   "code cave branch range exceeds s_branch simm16; leaving code object unchanged",
                   repl.start_offset);
    return false;
  }

  const uint64_t return_branch_pc = cave_byte_offset + repl.target_words.size() * sizeof(uint32_t);
  auto return_trailer = make_return_trailer(return_branch_pc, stub_next);
  if (return_trailer.empty()) {
    const bool can_defer = has_unimplemented_expand_gap(warnings_);
    if (warnings_) {
      warnings_->push_back(
          can_defer ? "code cave return branch range exceeds s_branch simm16; leaving source "
                      "instruction unchanged"
                    : "code cave return branch range exceeds s_branch simm16; leaving code object "
                      "unchanged");
      if (cave_diagnostics_enabled())
        warnings_->push_back(cave_range_diagnostic("code cave return range diagnostic", repl,
                                                   cave_byte_offset, target_size));
      append_placement_diagnostic(warnings_, "failed", PlacementTier::AppendedCave, repl,
                                  "reason=return-branch-out-of-range");
    }
    if (can_defer)
      return true;
    if (diagnostics_)
      append_error(*diagnostics_, DiagnosticKind::ResourceLimit,
                   "code cave return branch range exceeds s_branch simm16; leaving code object "
                   "unchanged",
                   repl.start_offset);
    return false;
  }

  patch_source_branch(fwd_dwords);
  auto cave_words = repl.target_words;
  cave_words.insert(cave_words.end(), return_trailer.begin(), return_trailer.end());

  patcher.append_cave_body(cave_words);
  record_cave_chain(stub_next, patcher.cave_body_size() - return_trailer.size() * sizeof(uint32_t));
  append_placement_diagnostic(warnings_, "selected", PlacementTier::AppendedCave, repl);
  return true;
}

bool BinaryTranslator::handle_encoding(
    const Instruction &inst, uint64_t offset, std::vector<uint8_t> &text, uint16_t dst_opcode,
    CodeObjectPatcher &patcher, std::span<const uint8_t> orig_text, PlacementState &placement,
    int16_t rdna4_grid_x_sgpr, int16_t rdna4_grid_yz_sgpr, InstructionList::Iterator block_begin,
    InstructionList::Iterator inst_it, std::span<BasicBlock *const> scope_blocks) {
  const uint32_t *raw = inst.raw_encoding();
  assert(raw && "handle_encoding called without raw encoding");
  const bool tracing = static_cast<bool>(trace_callback_);
  const auto source_words = tracing ? raw_words_for_inst(inst) : std::vector<uint32_t>{};

  auto emit_trace = [&](bool copied_original, bool changed, bool emitted_in_cave,
                        uint64_t target_offset, std::span<const uint32_t> target_words) {
    if (!trace_callback_)
      return;
    trace_callback_({.source_offset = offset,
                     .source_size = static_cast<uint32_t>(inst.size()),
                     .source_words = source_words,
                     .legalization = nullptr,
                     .copied_original = copied_original,
                     .semantic_lowering = false,
                     .changed = changed,
                     .emitted_in_cave = emitted_in_cave,
                     .target_offset = target_offset,
                     .target_words = target_words});
  };

  const uint32_t w0 = raw[0];
  const uint32_t w1 = inst.size() > 4 ? raw[1] : 0;
  const uint32_t w2 = inst.size() > 8 ? raw[2] : 0;

  std::vector<uint32_t> replacement_words;
  if (!encoding_translate_) {
    replacement_words = copy_instruction_words(orig_text, offset, inst.size());
  } else {
    auto tr = encoding_translate_(inst.encoding_id(), w0, w1, w2, dst_opcode);

    if (tr.word_count == 0) {
      replacement_words = copy_instruction_words(orig_text, offset, inst.size());
    } else {
      // Append trailing literal constant when the source instruction is larger
      // than the translated encoding. This handles single-word formats (SOP1,
      // SOP2, VOP1, VOP2, etc.) with a 32-bit literal appended when a source
      // operand is 0xFF. The encoding translator returns the format's native
      // word count; the literal is always one extra word beyond that.
      // Guard: only append if the gap is exactly one word (the literal). Larger
      // gaps would indicate a format mismatch, not a trailing literal.
      const uint32_t translated_bytes = tr.word_count * 4u;
      const uint32_t orig_bytes = inst.size();
      if (orig_bytes >= translated_bytes && orig_bytes - translated_bytes == 4 &&
          tr.word_count < 3) {
        uint32_t lit_word;
        std::memcpy(&lit_word, orig_text.data() + offset + translated_bytes, 4);
        tr.words[tr.word_count++] = lit_word;
      }

      replacement_words.assign(tr.words, tr.words + tr.word_count);
    }
  }

  const uint32_t orig_bytes = inst.size();
  if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4 &&
      !replacement_words.empty()) {
    remap_gfx1250_ttmp_grid_reads_to_sgpr(inst, rdna4_grid_x_sgpr, rdna4_grid_yz_sgpr,
                                          replacement_words);
    rewrite_gfx1250_zero_sgpr_v_mov_sources(replacement_words, inst, block_begin, inst_it);
    rewrite_gfx1250_zero_sgpr_scalar_sources(replacement_words, inst, block_begin, inst_it, offset,
                                             scope_blocks);
    if (rdna4_adjacent_valu_dependency_delay_needed_before(block_begin, inst_it)) {
      replacement_words.insert(replacement_words.begin(), build_s_delay_alu(1, host_arch_));
    }
    if (rdna4_memory_dependency_wait_needed_before(block_begin, inst_it)) {
      replacement_words.insert(replacement_words.begin(),
                               build_s_wait_alu(kWaitAluDepctrVaVdstVmVsrc0, host_arch_));
    }
    append_rdna4_scalar_dependency_barriers_after(replacement_words, inst, host_arch_);
  }

  const uint32_t target_size = replacement_words.size() * sizeof(uint32_t);
  const bool emitted_in_cave = target_size > orig_bytes;
  const uint64_t target_offset =
      emitted_in_cave ? patcher.cave_start() + patcher.cave_body_size() : offset;
  const bool copied_original =
      !encoding_translate_ || words_changed(source_words, replacement_words) == false;
  const bool changed = tracing && words_changed(source_words, replacement_words);
  const std::vector<uint32_t> target_words_for_trace =
      tracing ? replacement_words : std::vector<uint32_t>{};

  if (target_size <= orig_bytes) {
    std::memcpy(text.data() + offset, replacement_words.data(), target_size);
    if (target_size < orig_bytes)
      fill_nops(text, offset + target_size, orig_bytes - target_size, host_arch_);
  } else {
    SemanticReplacement repl{offset, offset + inst.size(), std::string(inst.mnemonic()),
                             std::move(replacement_words)};
    if (!apply_semantic(repl, text, patcher, placement))
      return false;
  }
  emit_trace(copied_original, changed, emitted_in_cave, target_offset, target_words_for_trace);
  return true;
}

} // namespace rocjitsu
