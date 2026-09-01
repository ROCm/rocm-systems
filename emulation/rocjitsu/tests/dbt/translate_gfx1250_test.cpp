// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file translate_gfx1250_test.cpp
/// @brief CPU-only tests for gfx1250 revision translation.

#include "decode_test_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/dbt/semantic/rules.h"
#include "rocjitsu/code/dbt/semantic_translator.h"
#include "rocjitsu/code/dbt/virtual_lds.h"
#include "rocjitsu/code/dbt/waitcnt_translator.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/kernarg_extension.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/code/patch/sidecar_metadata.h"
#include "rocjitsu/code/relocation_function_table.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/code/rj_code_internal.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "support/elf_test_support.h"
#include "support/translate_test_support.h"
#include "util/data_types.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
#include "hsa/hsa.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

template <size_t N>
std::vector<uint8_t>
make_gfx1250_image_with_live_sgprs(const std::array<uint32_t, N> &instruction_words,
                                   uint16_t live_sgprs) {
  if (live_sgprs > REGISTER_SET_MAX_SGPRS) {
    ADD_FAILURE() << "liveness wall exceeds the gfx1250 ordinary SGPR range";
    return {};
  }
  constexpr uint16_t kGfx1250M0Operand = 125;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  std::vector<uint32_t> words(instruction_words.begin(), instruction_words.end());
  for (uint16_t sgpr = 0; sgpr < live_sgprs; ++sgpr) {
    words.push_back(cdna5::build_sop1(
        cdna5::kSMovB32Sop1, {.ssrc0 = static_cast<uint8_t>(sgpr), .sdst = kGfx1250M0Operand})[0]);
  }
  words.push_back(kGfx1250SEndpgm);
  return test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
}

void enable_kernarg_segment_ptr_sgpr(std::vector<uint8_t> &image, uint32_t kernarg_size = 16) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;

  AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(KD));

  KD kd{};
  std::memcpy(&kd, image.data() + rodata->sectionOffset(), sizeof(kd));
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
  AMDHSA_BITS_SET(kd.kernel_code_properties,
                  rocr::llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
  kd.kernarg_size = kernarg_size;
  std::memcpy(image.data() + rodata->sectionOffset(), &kd, sizeof(kd));
}

} // namespace
} // namespace rocjitsu

namespace cdna3 = rocjitsu::cdna3;
namespace cdna4 = rocjitsu::cdna4;
namespace cdna5 = rocjitsu::cdna5;
namespace rdna4 = rocjitsu::rdna4;

constexpr uint16_t kGfx1250WmmaCompletionWaitImmediate = 0x0f9f;
constexpr auto kGfx1250WmmaCompletionWait =
    cdna5::build_sopp(cdna5::kSWaitAluSopp, {.simm16 = kGfx1250WmmaCompletionWaitImmediate});

/// @brief Build explicit gfx1250 revision options without partial aggregate
/// initializers, which are rejected by the test suite's warning policy.
rocjitsu::BinaryTranslatorOptions
gfx1250_revision_options(rocjitsu::ProcessorRevision input_revision,
                         rocjitsu::ProcessorRevision output_revision) {
  rocjitsu::BinaryTranslatorOptions options;
  options.input_revision = input_revision;
  options.output_revision = output_revision;
  options.verify_rewrite_discharge = input_revision == rocjitsu::ProcessorRevision::Gfx1250B0 &&
                                     output_revision == rocjitsu::ProcessorRevision::Gfx1250A0;
  return options;
}

constexpr uint16_t kAnyExpectedField = 0xffff;

enum class ExpectedCdna3Kind {
  Vop3,
  Vop3p,
  Vop2,
  Vop1,
  Sop2,
  Sop1,
  Vop3pMfma,
  Ds,
  Mubuf,
  Sopp,
};

struct ExpectedCdna3Inst {
  ExpectedCdna3Kind kind = ExpectedCdna3Kind::Vop3;
  uint16_t op = 0;
  uint16_t vdst = kAnyExpectedField;
  uint16_t acc = kAnyExpectedField;
  uint16_t acc_cd = kAnyExpectedField;
  uint16_t src0 = kAnyExpectedField;
  uint16_t src1 = kAnyExpectedField;
  uint16_t src2 = kAnyExpectedField;
  uint16_t data0 = kAnyExpectedField;
  uint16_t vdata = kAnyExpectedField;
};

struct Cdna4ToCdna3SemanticRuleCase {
  const char *name = "";
  uint16_t encoding_id = 0;
  uint16_t opcode = 0;
  std::array<uint32_t, 2> words{};
  std::vector<ExpectedCdna3Inst> expected{};
  size_t word_count = 2;
};

ExpectedCdna3Inst expect_vop3(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Vop3;
  inst.op = op;
  return inst;
}

ExpectedCdna3Inst expect_vop3(uint16_t op, uint16_t vdst, uint16_t src0, uint16_t src1,
                              uint16_t src2 = kAnyExpectedField) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Vop3;
  inst.op = op;
  inst.vdst = vdst;
  inst.src0 = src0;
  inst.src1 = src1;
  inst.src2 = src2;
  return inst;
}

ExpectedCdna3Inst expect_vop3p(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Vop3p;
  inst.op = op;
  return inst;
}

ExpectedCdna3Inst expect_vop2(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Vop2;
  inst.op = op;
  return inst;
}

ExpectedCdna3Inst expect_vop1(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Vop1;
  inst.op = op;
  return inst;
}

ExpectedCdna3Inst expect_sop2(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Sop2;
  inst.op = op;
  return inst;
}

ExpectedCdna3Inst expect_sop1(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Sop1;
  inst.op = op;
  return inst;
}

ExpectedCdna3Inst expect_mfma(uint16_t op, uint16_t vdst, uint16_t acc_cd, uint16_t src0,
                              uint16_t src1, uint16_t src2, uint16_t acc = 0) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Vop3pMfma;
  inst.op = op;
  inst.vdst = vdst;
  inst.acc = acc;
  inst.acc_cd = acc_cd;
  inst.src0 = src0;
  inst.src1 = src1;
  inst.src2 = src2;
  return inst;
}

ExpectedCdna3Inst expect_ds(uint16_t op, uint16_t data0 = kAnyExpectedField) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Ds;
  inst.op = op;
  inst.data0 = data0;
  return inst;
}

ExpectedCdna3Inst expect_mubuf(uint16_t op, uint16_t vdata = kAnyExpectedField) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Mubuf;
  inst.op = op;
  inst.vdata = vdata;
  return inst;
}

ExpectedCdna3Inst expect_sopp(uint16_t op) {
  ExpectedCdna3Inst inst{};
  inst.kind = ExpectedCdna3Kind::Sopp;
  inst.op = op;
  return inst;
}

std::vector<ExpectedCdna3Inst> expected_cdna3_bitop3_sequence(bool b16) {
  // Truth table 0xde lowers to S2 ^ S1 ^ (S1 & S2) ^ S0 ^ (S0 & S1).
  std::vector<ExpectedCdna3Inst> expected = {
      expect_vop3(cdna3::kVMovB32Vop3), // v_mov_b32
      expect_vop3(cdna3::kVXorB32Vop3), // v_xor_b32
      expect_vop3(cdna3::kVAndB32Vop3), // v_and_b32
      expect_vop3(cdna3::kVXorB32Vop3), // v_xor_b32
      expect_vop3(cdna3::kVXorB32Vop3), // v_xor_b32
      expect_vop3(cdna3::kVAndB32Vop3), // v_and_b32
      expect_vop3(cdna3::kVXorB32Vop3), // v_xor_b32
  };
  if (b16) {
    expected.push_back(expect_vop3(cdna3::kVLshlrevB32Vop3)); // v_lshlrev_b32
    expected.push_back(expect_vop3(cdna3::kVLshrrevB32Vop3)); // v_lshrrev_b32
  }
  expected.push_back(
      expect_vop3(cdna3::kVMovB32Vop3)); // v_mov_b32 copy scratch accumulator to vdst.
  return expected;
}

std::vector<ExpectedCdna3Inst> expected_cdna3_bitop3_fast_sequence(uint16_t op, bool b16) {
  std::vector<ExpectedCdna3Inst> expected = {expect_vop3(op)};
  if (b16) {
    // Compact V_BITOP3_B16 fast paths still need the same high-half cleanup as
    // the generic ANF expansion; otherwise the CDNA3 B32 op leaves stale bits
    // above the architectural 16-bit result.
    expected.push_back(expect_vop3(cdna3::kVLshlrevB32Vop3)); // v_lshlrev_b32
    expected.push_back(expect_vop3(cdna3::kVLshrrevB32Vop3)); // v_lshrrev_b32
  }
  return expected;
}

std::vector<ExpectedCdna3Inst> expected_cdna3_mfma_sequence(uint16_t narrow_op, uint16_t src2 = 128,
                                                            uint16_t source_acc = 0) {
  // Wide-K MFMA lowering is intentionally high-half first, then low-half.  The
  // order is part of the layout contract verified against MI300X Qwen q_proj.
  return {
      expect_mfma(narrow_op, 0, 1, 258, 262, src2, source_acc),
      expect_mfma(narrow_op, 0, 1, 256, 260, 256, source_acc),
  };
}

std::vector<ExpectedCdna3Inst>
expected_cdna3_buffer_load_lds_sequence(uint16_t mubuf_op, uint16_t ds_op,
                                        uint16_t scratch_data = kAnyExpectedField) {
  return {
      expect_sop1(cdna3::kSMovB64),            // s_mov_b64 save EXEC.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_lo, -1.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_hi, -1.
      expect_vop3(cdna3::kVMbcntLoU32B32Vop3), // v_mbcnt_lo_u32_b32
      expect_vop3(cdna3::kVMbcntHiU32B32Vop3), // v_mbcnt_hi_u32_b32
      expect_vop3(cdna3::kVLshlrevB32Vop3),    // v_lshlrev_b32 lane_id, 4
      expect_vop3(cdna3::kVAddU32Vop3),        // v_add_u32 m0, lane_offset
      expect_sop1(cdna3::kSMovB64),            // s_mov_b64 restore EXEC.
      expect_mubuf(mubuf_op, scratch_data),    // buffer_load_dword{x3,x4} into scratch VGPRs.
      expect_sopp(cdna3::kSWaitcntSopp),       // s_waitcnt 0 before consuming VMEM data.
      expect_ds(ds_op, scratch_data),          // ds_write_b32/b96/b128
      expect_sopp(cdna3::kSWaitcntSopp),       // s_waitcnt lgkmcnt(0) for the explicit DS write.
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_permlane32_swap_sequence() {
  return {
      expect_sop1(cdna3::kSMovB64),            // s_mov_b64 save EXEC.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_lo, -1.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_hi, -1.
      expect_vop3(cdna3::kVMbcntLoU32B32Vop3), // v_mbcnt_lo_u32_b32
      expect_vop3(cdna3::kVMbcntHiU32B32Vop3), // v_mbcnt_hi_u32_b32
      expect_vop3(cdna3::kVXorB32Vop3),        // v_xor_b32 lane, 32.
      expect_vop3(cdna3::kVLshlrevB32Vop3),    // v_lshlrev_b32 byte address.
      expect_ds(cdna3::kDsBpermuteB32Ds),      // ds_bpermute_b32 from old vdst high half.
      expect_ds(cdna3::kDsBpermuteB32Ds),      // ds_bpermute_b32 from old src low half.
      expect_sopp(cdna3::kSWaitcntSopp),       // s_waitcnt lgkmcnt(0).
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_lo, low-half mask.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_hi, low-half mask.
      expect_vop3(cdna3::kVMovB32Vop3),        // v_mov_b32 src <- old vdst high.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_lo, high-half mask.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_hi, high-half mask.
      expect_vop3(cdna3::kVMovB32Vop3),        // v_mov_b32 vdst <- old src low.
      expect_sop1(cdna3::kSMovB64),            // s_mov_b64 restore EXEC.
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_raw_b16_pack_sequence() {
  return {
      expect_vop3(cdna3::kVMovB32Vop3),     // v_mov_b32 -1
      expect_vop3(cdna3::kVLshrrevB32Vop3), // v_lshrrev_b32 16, mask
      expect_vop3(cdna3::kVAndB32Vop3),     // v_and_b32 low half
      expect_vop3(cdna3::kVAndB32Vop3),     // v_and_b32 high half
      expect_vop3(cdna3::kVLshlrevB32Vop3), // v_lshlrev_b32 16, high half
      expect_vop3(cdna3::kVOrB32Vop3),      // v_or_b32
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_cvt_pk_f16_f32_sequence() {
  return {
      expect_vop3(cdna3::kVCvtF16F32Vop3),  // v_cvt_f16_f32 low half into scratch.
      expect_vop3(cdna3::kVCvtF16F32Vop3),  // v_cvt_f16_f32 high half into scratch.
      expect_vop3(cdna3::kVLshlrevB32Vop3), // v_lshlrev_b32 16, high half.
      expect_vop3(cdna3::kVOrB32Vop3),      // v_or_b32 pack low/high halves into vdst.
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_cvt_pk_bf16_f32_sequence() {
  // Per-half NaN-safe RNE lowering (emit_cdna3_f32_to_bf16_rne): compute the
  // rounded value and the NaN-preserving value in parallel, then blend with an
  // exponent-all-ones mask via v_bfi_b32. See semantic/cdna4_to_cdna3.cpp.
  auto half = []() {
    return std::vector<ExpectedCdna3Inst>{
        expect_vop3(cdna3::kVMovB32Vop3),     // f = source bits into scratch.
        expect_vop3(cdna3::kVLshrrevB32Vop3), // (f >> 16)
        expect_vop3(cdna3::kVAndB32Vop3),     // & 1  -> rounding lsb
        expect_vop2(cdna3::kVAddU32Vop2),     // + 0x7fff  (rounding bias)
        expect_vop3(cdna3::kVAddU32Vop3),     // f + bias  -> rounded (t0)
        expect_vop3(cdna3::kVBfeU32Vop3),     // f & 0xffff -> low bits
        expect_vop3(cdna3::kVMinU32Vop3),     // min(1, low) -> nonzero flag
        expect_vop3(cdna3::kVLshlrevB32Vop3), // flag << 16
        expect_vop3(cdna3::kVOrB32Vop3),      // f | (flag<<16) -> NaN-preserving (t1)
        expect_vop2(cdna3::kVAndB32Vop2),     // f & 0x7f800000  (exponent field)
        expect_vop2(cdna3::kVXorB32Vop2),     // ^ 0x7f800000    -> 0 iff exp all-ones
        expect_vop3(cdna3::kVSubU32Vop3),     // - 1
        expect_vop3(cdna3::kVLshrrevB32Vop3), // >> 31
        expect_vop3(cdna3::kVSubU32Vop3),     // 0 - x  -> all-ones mask (t2)
        expect_vop3(cdna3::kVBfiB32Vop3),     // bfi(mask, t1, t0)  -> select
        expect_vop3(cdna3::kVLshrrevB32Vop3), // >> 16  -> BF16 half
    };
  };
  std::vector<ExpectedCdna3Inst> expected = half();
  const std::vector<ExpectedCdna3Inst> high = half();
  expected.insert(expected.end(), high.begin(), high.end());
  expected.push_back(expect_vop3(cdna3::kVLshlrevB32Vop3)); // high half into position.
  expected.push_back(expect_vop3(cdna3::kVOrB32Vop3));      // pack low/high into vdst.
  return expected;
}

std::vector<ExpectedCdna3Inst> expected_cdna3_cvt_f32_bf16_sequence() {
  return {
      expect_vop3(
          cdna3::kVLshlrevB32Vop3), // v_lshlrev_b32 16, src; BF16 bits become FP32 high half.
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_cvt_f32_bf16_sdwa_word0_sequence() {
  return {
      expect_vop3(cdna3::kVLshlrevB32Vop3, 4, rocjitsu::scalar_positive_inline_u32(16), 256 + 4),
  };
}

std::vector<ExpectedCdna3Inst> expected_cdna3_cvt_f32_bf16_sdwa_word1_sequence() {
  return {
      expect_vop3(cdna3::kVLshrrevB32Vop3, 4, rocjitsu::scalar_positive_inline_u32(16), 256 + 76),
      expect_vop3(cdna3::kVLshlrevB32Vop3, 4, rocjitsu::scalar_positive_inline_u32(16), 256 + 4),
  };
}

std::vector<ExpectedCdna3Inst>
expected_cdna3_dot2_f32_bf16_sequence(uint8_t vdst, uint16_t src0, uint16_t src1, uint16_t src2,
                                      uint8_t scratch_base, uint8_t op_sel = 0,
                                      uint8_t op_sel_hi = 0, uint8_t neg = 0, uint8_t neg_hi = 0,
                                      uint8_t op_sel_hi_2 = 0, bool clamp = false) {
  const uint8_t a_lo = scratch_base;
  const uint8_t b_lo = static_cast<uint8_t>(scratch_base + 1);
  const uint8_t a_hi = static_cast<uint8_t>(scratch_base + 2);
  const uint8_t b_hi = static_cast<uint8_t>(scratch_base + 3);
  std::vector<ExpectedCdna3Inst> expected;
  auto expect_widen = [&](uint8_t dst, uint16_t src, bool high_half, bool negate) {
    if (high_half) {
      expected.push_back(
          expect_vop3(cdna3::kVLshrrevB32Vop3, dst, rocjitsu::scalar_positive_inline_u32(16), src));
      expected.push_back(expect_vop3(cdna3::kVLshlrevB32Vop3, dst,
                                     rocjitsu::scalar_positive_inline_u32(16), 256 + dst));
    } else {
      expected.push_back(
          expect_vop3(cdna3::kVLshlrevB32Vop3, dst, rocjitsu::scalar_positive_inline_u32(16), src));
    }
    if (negate)
      expected.push_back(expect_vop2(cdna3::kVXorB32Vop2)); // v_xor_b32 literal flips FP32 sign.
  };
  expect_widen(a_lo, src0, (op_sel & 0x1u) != 0, (neg & 0x1u) != 0);
  expect_widen(b_lo, src1, (op_sel & 0x2u) != 0, (neg & 0x2u) != 0);
  expect_widen(a_hi, src0, op_sel_hi_2 != 0, (neg_hi & 0x1u) != 0);
  expect_widen(b_hi, src1, (op_sel_hi & 0x2u) != 0, (neg_hi & 0x2u) != 0);
  expected.push_back(expect_vop3(cdna3::kVMulF32Vop3, a_lo, 256 + a_lo, 256 + b_lo));
  expected.push_back(expect_vop3(cdna3::kVMulF32Vop3, a_hi, 256 + a_hi, 256 + b_hi));
  expected.push_back(expect_vop3(cdna3::kVAddF32Vop3, a_lo, 256 + a_lo, 256 + a_hi));
  expected.push_back(expect_vop3(cdna3::kVAddF32Vop3, vdst, 256 + a_lo, src2));
  if (clamp)
    expected.push_back(
        expect_vop3(cdna3::kVMed3F32Vop3, vdst, 256 + vdst, 128, 242)); // v_med3_f32 clamps [0,1].
  return expected;
}

std::vector<ExpectedCdna3Inst> expected_cdna3_ds_read_b64_tr_b16_sequence(bool acc_dst = false) {
  std::vector<ExpectedCdna3Inst> expected = {
      expect_sop1(cdna3::kSMovB64),            // s_mov_b64 save EXEC.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_lo, all lanes.
      expect_sop1(cdna3::kSMovB32),            // s_mov_b32 exec_hi, all lanes.
      expect_ds(cdna3::kDsReadB64Ds),          // ds_read_b64
      expect_sopp(cdna3::kSWaitcntSopp),       // s_waitcnt lgkmcnt(0)
      expect_vop3(cdna3::kVMbcntLoU32B32Vop3), // v_mbcnt_lo_u32_b32
      expect_vop3(cdna3::kVMbcntHiU32B32Vop3), // v_mbcnt_hi_u32_b32
      expect_vop3(cdna3::kVAndB32Vop3),        // v_and_b32
      expect_vop3(cdna3::kVLshlrevB32Vop3),    // v_lshlrev_b32
      expect_vop3(cdna3::kVAndB32Vop3),        // v_and_b32
      expect_vop3(cdna3::kVLshlrevB32Vop3),    // v_lshlrev_b32
      expect_vop3(cdna3::kVAndB32Vop3),        // v_and_b32
      expect_vop3(cdna3::kVOrB32Vop3),         // v_or_b32
      expect_vop3(cdna3::kVAddU32Vop3),        // v_add_u32
      expect_vop3(cdna3::kVLshlrevB32Vop3),    // v_lshlrev_b32
      expect_vop3(cdna3::kVOrB32Vop3),         // v_or_b32
      expect_ds(cdna3::kDsBpermuteB32Ds),      // ds_bpermute_b32
      expect_ds(cdna3::kDsBpermuteB32Ds),      // ds_bpermute_b32
      expect_sopp(cdna3::kSWaitcntSopp),       // s_waitcnt lgkmcnt(0)
      expect_vop3(cdna3::kVPermB32Vop3),       // v_perm_b32
      expect_vop3(cdna3::kVAddU32Vop3),        // v_add_u32
      expect_ds(cdna3::kDsBpermuteB32Ds),
      expect_ds(cdna3::kDsBpermuteB32Ds),
      expect_sopp(cdna3::kSWaitcntSopp),
      expect_vop3(cdna3::kVPermB32Vop3),
      expect_vop3(cdna3::kVAddU32Vop3),
      expect_ds(cdna3::kDsBpermuteB32Ds),
      expect_ds(cdna3::kDsBpermuteB32Ds),
      expect_sopp(cdna3::kSWaitcntSopp),
      expect_vop3(cdna3::kVPermB32Vop3),
      expect_vop3(cdna3::kVAddU32Vop3),
      expect_ds(cdna3::kDsBpermuteB32Ds),
      expect_ds(cdna3::kDsBpermuteB32Ds),
      expect_sopp(cdna3::kSWaitcntSopp),
      expect_vop3(cdna3::kVPermB32Vop3),
      expect_sop1(cdna3::kSMovB64), // s_mov_b64 restore EXEC.
  };
  auto first_pack = expected_cdna3_raw_b16_pack_sequence();
  expected.insert(expected.end(), first_pack.begin(), first_pack.end());
  auto second_pack = expected_cdna3_raw_b16_pack_sequence();
  expected.insert(expected.end(), second_pack.begin(), second_pack.end());
  if (acc_dst) {
    expected.push_back(expect_vop3p(cdna3::kVAccvgprWriteVop3p)); // v_accvgpr_write_b32 low dword.
    expected.push_back(expect_vop3p(cdna3::kVAccvgprWriteVop3p)); // v_accvgpr_write_b32 high dword.
  }
  return expected;
}

template <typename MachineInst>
std::array<uint32_t, 2> encode_two_word_inst(const MachineInst &inst) {
  std::array<uint32_t, 2> words{};
  std::memcpy(words.data(), &inst, sizeof(inst));
  return words;
}

std::array<uint32_t, 2> make_cdna4_bitop3_words(uint16_t opcode, uint8_t vdst,
                                                uint8_t truth_table = 0xde) {
  rocjitsu::cdna4::Vop3MachineInst inst{};
  inst.encoding = 0x34;
  inst.op = opcode;
  inst.vdst = vdst;
  inst.src0 = static_cast<uint16_t>(256 + vdst + 1);
  inst.src1 = static_cast<uint16_t>(256 + vdst + 2);
  inst.src2 = static_cast<uint16_t>(256 + vdst + 3);

  inst.omod = truth_table >> 6;
  inst.abs = (truth_table >> 3) & 0x7;
  inst.neg = truth_table & 0x7;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_v_lshl_add_u64_words(uint8_t vdst, uint16_t src0, uint16_t src1,
                                                        uint16_t src2) {
  rocjitsu::cdna4::Vop3MachineInst inst{};
  inst.encoding = 0x34;
  inst.op = cdna4::kVLshlAddU64Vop3;
  inst.vdst = vdst;
  inst.src0 = src0;
  inst.src1 = src1;
  inst.src2 = src2;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_bitop3_b16_unsupported_op_sel_words() {
  rocjitsu::cdna4::Vop3MachineInst inst{};
  inst.encoding = 0x34;
  inst.op = cdna4::kVBitop3B16Vop3;
  inst.vdst = 8;
  inst.src0 = 256 + 9;
  inst.src1 = 256 + 10;
  inst.src2 = 256 + 11;
  inst.op_sel = 1;
  inst.omod = 1;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_cvt_pk_f16_f32_words() {
  rocjitsu::cdna4::Vop3MachineInst inst{};
  inst.encoding = 0x34;
  inst.op = cdna4::kVCvtPkF16F32Vop3;
  inst.vdst = 0;
  inst.src0 = 256 + 1;
  inst.src1 = 256 + 2;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_cvt_pk_bf16_f32_words(uint8_t vdst = 0, uint16_t src0 = 256 + 1,
                                                         uint16_t src1 = 256 + 2) {
  rocjitsu::cdna4::Vop3MachineInst inst{};
  inst.encoding = 0x34;
  inst.op = cdna4::kVCvtPkBf16F32Vop3;
  inst.vdst = vdst;
  inst.src0 = src0;
  inst.src1 = src1;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_cvt_f32_bf16_words(uint16_t encoding_id) {
  rocjitsu::cdna4::Vop1MachineInst inst{};
  // As with the VOP1 permlane cases below, semantic-rule lookup sees generated
  // primary-decode ids 0xfc..0xff while the raw instruction keeps the hardware
  // VOP1 selector in bits 31:25. Vary VDST high bits to exercise each id.
  inst.encoding = 0x3f;
  inst.op = cdna4::kVCvtF32Bf16Vop1;
  inst.vdst = static_cast<uint8_t>((encoding_id - 0xFCu) << 6);
  inst.src0 = 256 + 1;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_dot2_f32_bf16_words(uint8_t vdst, uint16_t src0, uint16_t src1,
                                                       uint16_t src2, uint8_t op_sel = 0,
                                                       uint8_t op_sel_hi = 0, uint8_t neg = 0,
                                                       uint8_t neg_hi = 0, uint8_t op_sel_hi_2 = 0,
                                                       bool clamp = false) {
  rocjitsu::cdna4::Vop3pMachineInst inst{};
  inst.encoding = cdna4::encoding::kVop3p;
  inst.op = cdna4::kVDot2F32Bf16Vop3p;
  inst.vdst = vdst;
  inst.op_sel = op_sel;
  inst.op_sel_hi = op_sel_hi;
  inst.op_sel_hi_2 = op_sel_hi_2;
  inst.clamp = clamp ? 1 : 0;
  inst.neg = neg;
  inst.neg_hi = neg_hi;
  inst.src0 = src0;
  inst.src1 = src1;
  inst.src2 = src2;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_cvt_f32_bf16_sdwa_word0_words() {
  rocjitsu::cdna4::Vop1VopSdwaMachineInst inst{};
  inst.encoding = 0x3f;
  inst.op = cdna4::kVCvtF32Bf16Vop1;
  inst.vdst = 4;
  inst.src0 = 249;
  inst.vsrc0 = 4;
  inst.dst_sel = 6;
  inst.dst_unused = 2;
  inst.src0_sel = 4;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_cvt_f32_bf16_sdwa_word1_words() {
  rocjitsu::cdna4::Vop1VopSdwaMachineInst inst{};
  inst.encoding = 0x3f;
  inst.op = cdna4::kVCvtF32Bf16Vop1;
  inst.vdst = 4;
  inst.src0 = 249;
  inst.vsrc0 = 76;
  inst.dst_sel = 6;
  inst.dst_unused = 2;
  inst.src0_sel = 5;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_permlane_swap_b32_words(uint16_t encoding_id, uint16_t opcode) {
  rocjitsu::cdna4::Vop1MachineInst inst{};
  // The legalization table's VOP1 encoding ids (0xfc..0xff) are the generated
  // primary-decode ids, not the raw 7-bit VOP1 selector.  Primary decode looks
  // at bits 31:23, so VOP1 contributes its fixed selector in bits 31:25 and
  // VDST[7:6] in bits 24:23.  Keep the real VOP1 selector at 0x3f and vary
  // VDST's high bits to exercise each generated semantic rule.
  inst.encoding = 0x3f;
  inst.op = opcode & 0xFF;
  inst.vdst = static_cast<uint8_t>((encoding_id - cdna4::encoding::kVop1) << 6);
  inst.src0 = 256 + 1;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_permlane16_swap_b32_words(uint16_t encoding_id) {
  return make_cdna4_permlane_swap_b32_words(encoding_id, cdna4::kVPermlane16SwapB32Vop1);
}

std::array<uint32_t, 2> make_cdna4_permlane32_swap_b32_words(uint16_t encoding_id) {
  return make_cdna4_permlane_swap_b32_words(encoding_id, cdna4::kVPermlane32SwapB32Vop1);
}

std::array<uint32_t, 2> make_cdna4_mfma_words(uint16_t opcode, uint8_t vdst, uint16_t src0,
                                              uint16_t src1, uint16_t src2 = 128, uint8_t acc = 0) {
  rocjitsu::cdna4::Vop3pMfmaMachineInst inst{};
  inst.encoding = cdna4::encoding::kVop3p;
  inst.op = opcode & 0x7F;
  inst.vdst = vdst;
  inst.acc_cd = 1;
  inst.src0 = src0;
  inst.src1 = src1;
  inst.src2 = src2;
  inst.acc = acc;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_mfma_vgpr_dst_alias_words() {
  rocjitsu::cdna4::Vop3pMfmaMachineInst inst{};
  inst.encoding = cdna4::encoding::kVop3pMfma;
  inst.op = cdna4::kVMfmaF3216x16x32F16Vop3pMfma;
  inst.vdst = 0;
  inst.acc_cd = 0;
  // Ordinary-VGPR destination v[0:3] overlaps the first wide source window.
  // The lowering must therefore place the first narrow MFMA's partial result in
  // scratch and report that scratch through TranslationContext::require_vgprs().
  inst.src0 = 256;
  inst.src1 = 260;
  inst.src2 = 128;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_dot2c_unimplemented_expand_words() {
  // v_dot2c_f32_bf16 is present in CDNA4 and not CDNA3. The generated
  // legalization table marks raw encoding-id 88/opcode 22 as EXPAND, but no
  // handwritten semantic rule exists yet.
  return {0x2C000000U, 0x00000000U};
}

std::array<uint32_t, 2> make_cdna4_ds_read_b64_tr_b16_words(uint16_t byte_offset = 0,
                                                            uint8_t addr = 2, bool acc = false) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsReadB64TrB16Ds;
  inst.offset0 = byte_offset & 0xFFu;
  inst.offset1 = byte_offset >> 8;
  inst.addr = addr;
  inst.vdst = 0;
  inst.acc = acc ? 1 : 0;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read_b32_words(uint16_t byte_offset = 0x0134,
                                                     uint8_t addr = 4, uint8_t vdst = 7) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsReadB32Ds;
  inst.offset0 = byte_offset & 0xFFu;
  inst.offset1 = byte_offset >> 8;
  inst.addr = addr;
  inst.vdst = vdst;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read2_b32_words(uint8_t offset0 = 1, uint8_t offset1 = 2,
                                                      uint8_t addr = 12, uint8_t vdst = 20) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsRead2B32Ds;
  inst.offset0 = offset0;
  inst.offset1 = offset1;
  inst.addr = addr;
  inst.vdst = vdst;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read2st64_b32_words(uint8_t offset0 = 2,
                                                          uint8_t offset1 = 3) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsRead2st64B32Ds;
  inst.offset0 = offset0;
  inst.offset1 = offset1;
  inst.addr = 12;
  inst.vdst = 20;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read2_b64_words(uint8_t offset0 = 3, uint8_t offset1 = 68,
                                                      uint8_t addr = 58, uint8_t vdst = 66) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsRead2B64Ds;
  inst.offset0 = offset0;
  inst.offset1 = offset1;
  inst.addr = addr;
  inst.vdst = vdst;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read2st64_b64_words(uint8_t offset0 = 2, uint8_t offset1 = 3,
                                                          uint8_t addr = 98, uint8_t vdst = 112) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsRead2st64B64Ds;
  inst.offset0 = offset0;
  inst.offset1 = offset1;
  inst.addr = addr;
  inst.vdst = vdst;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read_b64_words(uint8_t addr = 4, uint8_t vdst = 8,
                                                     uint16_t byte_offset = 0) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsReadB64Ds;
  inst.offset0 = byte_offset & 0xFFu;
  inst.offset1 = byte_offset >> 8;
  inst.addr = addr;
  inst.vdst = vdst;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read_b128_words(uint8_t addr = 4, uint8_t vdst = 8,
                                                      uint16_t byte_offset = 0, bool acc = false) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsReadB128Ds;
  inst.offset0 = byte_offset & 0xFFu;
  inst.offset1 = byte_offset >> 8;
  inst.addr = addr;
  inst.vdst = vdst;
  inst.acc = acc ? 1 : 0;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read_u16_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsReadU16Ds;
  inst.offset0 = 0x20;
  inst.addr = 4;
  inst.vdst = 7;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read_u8_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsReadU8Ds;
  inst.offset0 = 0x20;
  inst.addr = 8;
  inst.vdst = 12;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_read_i8_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsReadI8Ds;
  inst.offset0 = 0x20;
  inst.addr = 8;
  inst.vdst = 12;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write_b8_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWriteB8Ds;
  inst.offset0 = 0x10;
  inst.addr = 4;
  inst.data0 = 7;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write_b8_d16_hi_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWriteB8D16HiDs;
  inst.offset0 = 0x10;
  inst.addr = 4;
  inst.data0 = 7;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write_b32_words(uint8_t addr = 4, uint8_t data = 7,
                                                      uint16_t byte_offset = 0) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWriteB32Ds;
  inst.offset0 = byte_offset & 0xFFu;
  inst.offset1 = byte_offset >> 8;
  inst.addr = addr;
  inst.data0 = data;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write_b96_words(uint8_t addr = 4, uint8_t data = 8,
                                                      uint16_t byte_offset = 0) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWriteB96Ds;
  inst.offset0 = byte_offset & 0xFFu;
  inst.offset1 = byte_offset >> 8;
  inst.addr = addr;
  inst.data0 = data;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write_b64_words(uint8_t addr = 4, uint8_t data = 8,
                                                      uint16_t byte_offset = 0) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWriteB64Ds;
  inst.offset0 = byte_offset & 0xFFu;
  inst.offset1 = byte_offset >> 8;
  inst.addr = addr;
  inst.data0 = data;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write_b128_words(uint8_t addr = 4, uint8_t data = 8,
                                                       uint16_t byte_offset = 0, bool acc = false) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWriteB128Ds;
  inst.offset0 = byte_offset & 0xFFu;
  inst.offset1 = byte_offset >> 8;
  inst.addr = addr;
  inst.data0 = data;
  inst.acc = acc ? 1 : 0;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write2_b32_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWrite2B32Ds;
  inst.offset0 = 1;
  inst.offset1 = 2;
  inst.addr = 4;
  inst.data0 = 7;
  inst.data1 = 9;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write2_b64_words(uint8_t addr = 4, uint8_t data0 = 8,
                                                       uint8_t data1 = 12) {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWrite2B64Ds;
  inst.offset0 = 1;
  inst.offset1 = 2;
  inst.addr = addr;
  inst.data0 = data0;
  inst.data1 = data1;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_write2st64_b64_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsWrite2st64B64Ds;
  inst.offset0 = 1;
  inst.offset1 = 2;
  inst.addr = 4;
  inst.data0 = 8;
  inst.data1 = 12;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_ds_add_u32_words() {
  rocjitsu::cdna4::DsMachineInst inst{};
  inst.encoding = 0x36;
  inst.op = cdna4::kDsAddU32Ds;
  inst.addr = 3;
  inst.data0 = 7;
  return encode_two_word_inst(inst);
}

std::array<uint32_t, 2> make_cdna4_buffer_load_lds_words(uint16_t op, uint8_t vaddr = 2) {
  rocjitsu::cdna4::MubufMachineInst inst{};
  inst.encoding = 0x38;
  inst.op = op & 0x7F;
  inst.lds = 1;
  inst.offen = 1;
  inst.vaddr = vaddr;
  inst.vdata = 0;
  inst.srsrc = 4;
  inst.soffset = 0;
  return encode_two_word_inst(inst);
}

std::vector<Cdna4ToCdna3SemanticRuleCase> cdna4_to_cdna3_semantic_rule_cases() {
  return {
      {"VBitop3B16", cdna4::encoding::kVop3OpHi4, cdna4::kVBitop3B16Vop3,
       make_cdna4_bitop3_words(cdna4::kVBitop3B16Vop3, 8), expected_cdna3_bitop3_sequence(true)},
      {"VBitop3B16FastEc", cdna4::encoding::kVop3OpHi4, cdna4::kVBitop3B16Vop3,
       make_cdna4_bitop3_words(cdna4::kVBitop3B16Vop3, 8, 0xec),
       expected_cdna3_bitop3_fast_sequence(cdna3::kVAndOrB32Vop3, true)},
      {"VBitop3B16FastF8", cdna4::encoding::kVop3OpHi4, cdna4::kVBitop3B16Vop3,
       make_cdna4_bitop3_words(cdna4::kVBitop3B16Vop3, 8, 0xf8),
       expected_cdna3_bitop3_fast_sequence(cdna3::kVAndOrB32Vop3, true)},
      {"VBitop3B16FastFe", cdna4::encoding::kVop3OpHi4, cdna4::kVBitop3B16Vop3,
       make_cdna4_bitop3_words(cdna4::kVBitop3B16Vop3, 8, 0xfe),
       expected_cdna3_bitop3_fast_sequence(cdna3::kVOr3B32Vop3, true)},
      {"VBitop3B32", cdna4::encoding::kVop3OpHi4, cdna4::kVBitop3B32Vop3,
       make_cdna4_bitop3_words(cdna4::kVBitop3B32Vop3, 16), expected_cdna3_bitop3_sequence(false)},
      {"VCvtPkF16F32", cdna4::encoding::kVop3OpHi4, cdna4::kVCvtPkF16F32Vop3,
       make_cdna4_cvt_pk_f16_f32_words(), expected_cdna3_cvt_pk_f16_f32_sequence()},
      {"VCvtPkBf16F32", cdna4::encoding::kVop3OpHi4, cdna4::kVCvtPkBf16F32Vop3,
       make_cdna4_cvt_pk_bf16_f32_words(), expected_cdna3_cvt_pk_bf16_f32_sequence()},
      {"VCvtF32Bf16E32", cdna4::encoding::kVop1, cdna4::kVCvtF32Bf16Vop1,
       make_cdna4_cvt_f32_bf16_words(cdna4::encoding::kVop1),
       expected_cdna3_cvt_f32_bf16_sequence(), 1},
      {"VCvtF32Bf16E32Hi1", cdna4::encoding::kVop1Hi1, cdna4::kVCvtF32Bf16Vop1,
       make_cdna4_cvt_f32_bf16_words(cdna4::encoding::kVop1Hi1),
       expected_cdna3_cvt_f32_bf16_sequence(), 1},
      {"VCvtF32Bf16E32Hi2", cdna4::encoding::kVop1Hi2, cdna4::kVCvtF32Bf16Vop1,
       make_cdna4_cvt_f32_bf16_words(cdna4::encoding::kVop1Hi2),
       expected_cdna3_cvt_f32_bf16_sequence(), 1},
      {"VCvtF32Bf16E32Hi3", cdna4::encoding::kVop1Hi3, cdna4::kVCvtF32Bf16Vop1,
       make_cdna4_cvt_f32_bf16_words(cdna4::encoding::kVop1Hi3),
       expected_cdna3_cvt_f32_bf16_sequence(), 1},
      {"VCvtF32Bf16SdwaWord0", cdna4::encoding::kVop1, cdna4::kVCvtF32Bf16Vop1,
       make_cdna4_cvt_f32_bf16_sdwa_word0_words(),
       expected_cdna3_cvt_f32_bf16_sdwa_word0_sequence(), 2},
      {"VCvtF32Bf16SdwaWord1", cdna4::encoding::kVop1, cdna4::kVCvtF32Bf16Vop1,
       make_cdna4_cvt_f32_bf16_sdwa_word1_words(),
       expected_cdna3_cvt_f32_bf16_sdwa_word1_sequence(), 2},
      {"VDot2F32Bf16AccumAlias", cdna4::encoding::kVop3p, cdna4::kVDot2F32Bf16Vop3p,
       make_cdna4_dot2_f32_bf16_words(/*vdst=*/0, /*src0=*/256 + 1, /*src1=*/256 + 2,
                                      /*src2=*/256),
       expected_cdna3_dot2_f32_bf16_sequence(/*vdst=*/0, /*src0=*/256 + 1,
                                             /*src1=*/256 + 2, /*src2=*/256,
                                             /*scratch_base=*/3)},
      {"VDot2F32Bf16OpSelNeg", cdna4::encoding::kVop3p, cdna4::kVDot2F32Bf16Vop3p,
       make_cdna4_dot2_f32_bf16_words(/*vdst=*/0, /*src0=*/256 + 1, /*src1=*/256 + 2,
                                      /*src2=*/256, /*op_sel=*/1, /*op_sel_hi=*/2,
                                      /*neg=*/1, /*neg_hi=*/2),
       expected_cdna3_dot2_f32_bf16_sequence(/*vdst=*/0, /*src0=*/256 + 1,
                                             /*src1=*/256 + 2, /*src2=*/256,
                                             /*scratch_base=*/3, /*op_sel=*/1,
                                             /*op_sel_hi=*/2, /*neg=*/1, /*neg_hi=*/2)},
      {"VDot2F32Bf16OpSelHi2Clamp", cdna4::encoding::kVop3p, cdna4::kVDot2F32Bf16Vop3p,
       make_cdna4_dot2_f32_bf16_words(/*vdst=*/0, /*src0=*/256 + 1, /*src1=*/256 + 2,
                                      /*src2=*/256, /*op_sel=*/0, /*op_sel_hi=*/0,
                                      /*neg=*/0, /*neg_hi=*/0, /*op_sel_hi_2=*/1,
                                      /*clamp=*/true),
       expected_cdna3_dot2_f32_bf16_sequence(/*vdst=*/0, /*src0=*/256 + 1,
                                             /*src1=*/256 + 2, /*src2=*/256,
                                             /*scratch_base=*/3, /*op_sel=*/0,
                                             /*op_sel_hi=*/0, /*neg=*/0, /*neg_hi=*/0,
                                             /*op_sel_hi_2=*/1, /*clamp=*/true)},
      {"VPermlane16SwapB32E32", cdna4::encoding::kVop1, cdna4::kVPermlane16SwapB32Vop1,
       make_cdna4_permlane16_swap_b32_words(cdna4::encoding::kVop1),
       expected_cdna3_permlane32_swap_sequence(), 1},
      {"VPermlane16SwapB32E32Hi1", cdna4::encoding::kVop1Hi1, cdna4::kVPermlane16SwapB32Vop1,
       make_cdna4_permlane16_swap_b32_words(cdna4::encoding::kVop1Hi1),
       expected_cdna3_permlane32_swap_sequence(), 1},
      {"VPermlane16SwapB32E32Hi2", cdna4::encoding::kVop1Hi2, cdna4::kVPermlane16SwapB32Vop1,
       make_cdna4_permlane16_swap_b32_words(cdna4::encoding::kVop1Hi2),
       expected_cdna3_permlane32_swap_sequence(), 1},
      {"VPermlane16SwapB32E32Hi3", cdna4::encoding::kVop1Hi3, cdna4::kVPermlane16SwapB32Vop1,
       make_cdna4_permlane16_swap_b32_words(cdna4::encoding::kVop1Hi3),
       expected_cdna3_permlane32_swap_sequence(), 1},
      {"VPermlane32SwapB32E32", cdna4::encoding::kVop1, cdna4::kVPermlane32SwapB32Vop1,
       make_cdna4_permlane32_swap_b32_words(cdna4::encoding::kVop1),
       expected_cdna3_permlane32_swap_sequence(), 1},
      {"VPermlane32SwapB32E32Hi1", cdna4::encoding::kVop1Hi1, cdna4::kVPermlane32SwapB32Vop1,
       make_cdna4_permlane32_swap_b32_words(cdna4::encoding::kVop1Hi1),
       expected_cdna3_permlane32_swap_sequence(), 1},
      {"VPermlane32SwapB32E32Hi2", cdna4::encoding::kVop1Hi2, cdna4::kVPermlane32SwapB32Vop1,
       make_cdna4_permlane32_swap_b32_words(cdna4::encoding::kVop1Hi2),
       expected_cdna3_permlane32_swap_sequence(), 1},
      {"VPermlane32SwapB32E32Hi3", cdna4::encoding::kVop1Hi3, cdna4::kVPermlane32SwapB32Vop1,
       make_cdna4_permlane32_swap_b32_words(cdna4::encoding::kVop1Hi3),
       expected_cdna3_permlane32_swap_sequence(), 1},
      {"MfmaF32_16x16x32Bf16", cdna4::encoding::kVop3p, cdna4::kVMfmaF3216x16x32Bf16Vop3pMfma,
       make_cdna4_mfma_words(cdna4::kVMfmaF3216x16x32Bf16Vop3pMfma, 0, 256, 260),
       expected_cdna3_mfma_sequence(cdna3::kVMfmaF3216x16x16Bf16Vop3pMfma)},
      {"MfmaF32_32x32x16Bf16", cdna4::encoding::kVop3p, cdna4::kVMfmaF3232x32x16Bf16Vop3pMfma,
       make_cdna4_mfma_words(cdna4::kVMfmaF3232x32x16Bf16Vop3pMfma, 0, 256, 260),
       expected_cdna3_mfma_sequence(cdna3::kVMfmaF3232x32x8Bf16Vop3pMfma)},
      {"MfmaF32_32x32x16Bf16Source1Acc", cdna4::encoding::kVop3p,
       cdna4::kVMfmaF3232x32x16Bf16Vop3pMfma,
       make_cdna4_mfma_words(cdna4::kVMfmaF3232x32x16Bf16Vop3pMfma, 0, 256, 260, 272, 2),
       expected_cdna3_mfma_sequence(cdna3::kVMfmaF3232x32x8Bf16Vop3pMfma, 272, 2)},
      {"MfmaF32_16x16x32F16", cdna4::encoding::kVop3p, cdna4::kVMfmaF3216x16x32F16Vop3pMfma,
       make_cdna4_mfma_words(cdna4::kVMfmaF3216x16x32F16Vop3pMfma, 0, 256, 260),
       expected_cdna3_mfma_sequence(cdna3::kVMfmaF3216x16x16F16Vop3pMfma)},
      {"MfmaF32_32x32x16F16", cdna4::encoding::kVop3p, cdna4::kVMfmaF3232x32x16F16Vop3pMfma,
       make_cdna4_mfma_words(cdna4::kVMfmaF3232x32x16F16Vop3pMfma, 0, 256, 260),
       expected_cdna3_mfma_sequence(cdna3::kVMfmaF3232x32x8F16Vop3pMfma)},
      {"MfmaF32_16x16x32F16AccumVgpr", cdna4::encoding::kVop3p,
       cdna4::kVMfmaF3216x16x32F16Vop3pMfma,
       make_cdna4_mfma_words(cdna4::kVMfmaF3216x16x32F16Vop3pMfma, 0, 256, 260, 272),
       expected_cdna3_mfma_sequence(cdna3::kVMfmaF3216x16x16F16Vop3pMfma, 272)},
      {"MfmaF32_32x32x16F16AccumVgpr", cdna4::encoding::kVop3p,
       cdna4::kVMfmaF3232x32x16F16Vop3pMfma,
       make_cdna4_mfma_words(cdna4::kVMfmaF3232x32x16F16Vop3pMfma, 0, 256, 260, 272),
       expected_cdna3_mfma_sequence(cdna3::kVMfmaF3232x32x8F16Vop3pMfma, 272)},
      {"DsReadB64TrB16", cdna4::encoding::kDsHi3, cdna4::kDsReadB64TrB16Ds,
       make_cdna4_ds_read_b64_tr_b16_words(), expected_cdna3_ds_read_b64_tr_b16_sequence()},
      {"DsReadB64TrB16Acc", cdna4::encoding::kDsHi7, cdna4::kDsReadB64TrB16Ds,
       make_cdna4_ds_read_b64_tr_b16_words(/*byte_offset=*/0, /*addr=*/2, /*acc=*/true),
       expected_cdna3_ds_read_b64_tr_b16_sequence(/*acc_dst=*/true)},
      {"BufferLoadDwordLds", cdna4::encoding::kMubuf, cdna4::kBufferLoadDwordMubuf,
       make_cdna4_buffer_load_lds_words(cdna4::kBufferLoadDwordMubuf),
       expected_cdna3_buffer_load_lds_sequence(cdna3::kBufferLoadDwordMubuf, cdna3::kDsWriteB32Ds)},
      {"BufferLoadDwordx3Lds", cdna4::encoding::kMubuf, cdna4::kBufferLoadDwordx3Mubuf,
       make_cdna4_buffer_load_lds_words(cdna4::kBufferLoadDwordx3Mubuf),
       expected_cdna3_buffer_load_lds_sequence(cdna3::kBufferLoadDwordx3Mubuf,
                                               cdna3::kDsWriteB96Ds)},
      {"BufferLoadDwordx3LdsEvenScratch", cdna4::encoding::kMubuf, cdna4::kBufferLoadDwordx3Mubuf,
       make_cdna4_buffer_load_lds_words(cdna4::kBufferLoadDwordx3Mubuf, /*vaddr=*/0),
       expected_cdna3_buffer_load_lds_sequence(cdna3::kBufferLoadDwordx3Mubuf, cdna3::kDsWriteB96Ds,
                                               /*scratch_data=*/2)},
      {"BufferLoadDwordx4Lds", cdna4::encoding::kMubuf, cdna4::kBufferLoadDwordx4Mubuf,
       make_cdna4_buffer_load_lds_words(cdna4::kBufferLoadDwordx4Mubuf),
       expected_cdna3_buffer_load_lds_sequence(cdna3::kBufferLoadDwordx4Mubuf,
                                               cdna3::kDsWriteB128Ds)},
      {"BufferLoadDwordx4LdsEvenScratch", cdna4::encoding::kMubuf, cdna4::kBufferLoadDwordx4Mubuf,
       make_cdna4_buffer_load_lds_words(cdna4::kBufferLoadDwordx4Mubuf, /*vaddr=*/0),
       expected_cdna3_buffer_load_lds_sequence(cdna3::kBufferLoadDwordx4Mubuf,
                                               cdna3::kDsWriteB128Ds, /*scratch_data=*/2)},
  };
}

bool has_cdna4_to_cdna3_semantic_rule(uint16_t encoding_id, uint16_t opcode) {
  for (const auto &rule : rocjitsu::semantic_expand_rules_cdna4_to_cdna3()) {
    if (rule.src_encoding_id == encoding_id && rule.src_opcode == opcode)
      return true;
  }
  return false;
}

bool has_cdna4_to_cdna3_semantic_rule_case(uint16_t encoding_id, uint16_t opcode) {
  for (const auto &test_case : cdna4_to_cdna3_semantic_rule_cases()) {
    if (test_case.encoding_id == encoding_id && test_case.opcode == opcode)
      return true;
  }
  return false;
}

void expect_field_matches(uint16_t expected, uint16_t actual, std::string_view field_name) {
  if (expected != kAnyExpectedField) {
    EXPECT_EQ(actual, expected) << field_name;
  }
}

void expect_cdna3_instruction_matches(const rocjitsu::Instruction &inst,
                                      const ExpectedCdna3Inst &expected) {
  const uint32_t *raw = inst.raw_encoding();
  ASSERT_NE(raw, nullptr);

  switch (expected.kind) {
  case ExpectedCdna3Kind::Vop3: {
    rocjitsu::cdna3::Vop3MachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, 0x34u);
    EXPECT_EQ(actual.op, expected.op);
    expect_field_matches(expected.vdst, static_cast<uint16_t>(actual.vdst), "vdst");
    expect_field_matches(expected.src0, static_cast<uint16_t>(actual.src0), "src0");
    expect_field_matches(expected.src1, static_cast<uint16_t>(actual.src1), "src1");
    expect_field_matches(expected.src2, static_cast<uint16_t>(actual.src2), "src2");
    break;
  }
  case ExpectedCdna3Kind::Vop3p: {
    rocjitsu::cdna3::Vop3pMachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, cdna3::encoding::kVop3p);
    EXPECT_EQ(actual.op, expected.op);
    expect_field_matches(expected.vdst, static_cast<uint16_t>(actual.vdst), "vdst");
    expect_field_matches(expected.src0, static_cast<uint16_t>(actual.src0), "src0");
    expect_field_matches(expected.src1, static_cast<uint16_t>(actual.src1), "src1");
    expect_field_matches(expected.src2, static_cast<uint16_t>(actual.src2), "src2");
    break;
  }
  case ExpectedCdna3Kind::Vop2: {
    rocjitsu::cdna3::Vop2MachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.op, expected.op);
    break;
  }
  case ExpectedCdna3Kind::Vop1: {
    rocjitsu::cdna3::Vop1MachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.op, expected.op);
    break;
  }
  case ExpectedCdna3Kind::Sop2: {
    rocjitsu::cdna3::Sop2MachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, 0x2u);
    EXPECT_EQ(actual.op, expected.op);
    break;
  }
  case ExpectedCdna3Kind::Sop1: {
    rocjitsu::cdna3::Sop1MachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, cdna3::encoding::kSop1);
    EXPECT_EQ(actual.op, expected.op);
    break;
  }
  case ExpectedCdna3Kind::Vop3pMfma: {
    rocjitsu::cdna3::Vop3pMfmaMachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, cdna3::encoding::kVop3pMfma);
    EXPECT_EQ(actual.op, expected.op);
    expect_field_matches(expected.vdst, static_cast<uint16_t>(actual.vdst), "vdst");
    expect_field_matches(expected.acc, static_cast<uint16_t>(actual.acc), "acc");
    expect_field_matches(expected.acc_cd, static_cast<uint16_t>(actual.acc_cd), "acc_cd");
    expect_field_matches(expected.src0, static_cast<uint16_t>(actual.src0), "src0");
    expect_field_matches(expected.src1, static_cast<uint16_t>(actual.src1), "src1");
    expect_field_matches(expected.src2, static_cast<uint16_t>(actual.src2), "src2");
    break;
  }
  case ExpectedCdna3Kind::Ds: {
    rocjitsu::cdna3::DsMachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, 0x36u);
    EXPECT_EQ(actual.op, expected.op);
    expect_field_matches(expected.vdst, static_cast<uint16_t>(actual.vdst), "vdst");
    expect_field_matches(expected.data0, static_cast<uint16_t>(actual.data0), "data0");
    break;
  }
  case ExpectedCdna3Kind::Mubuf: {
    rocjitsu::cdna3::MubufMachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, 0x38u);
    EXPECT_EQ(actual.op, expected.op);
    EXPECT_EQ(actual.lds, 0u);
    expect_field_matches(expected.vdata, static_cast<uint16_t>(actual.vdata), "vdata");
    break;
  }
  case ExpectedCdna3Kind::Sopp: {
    rocjitsu::cdna3::SoppMachineInst actual{};
    std::memcpy(&actual, raw, sizeof(actual));
    EXPECT_EQ(actual.encoding, cdna3::encoding::kSopp);
    EXPECT_EQ(actual.op, expected.op);
    break;
  }
  }
}

void expect_cdna3_text_matches(const rocjitsu::Section &text,
                               const std::vector<ExpectedCdna3Inst> &expected,
                               bool allow_non_nop_tail = false) {
  ASSERT_EQ(text.size() % sizeof(uint32_t), 0u);

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);

  const auto *words = reinterpret_cast<const uint32_t *>(text.data());
  const size_t word_count = text.size() / sizeof(uint32_t);
  std::vector<std::unique_ptr<rocjitsu::Instruction>> actual;
  for (size_t pc = 0; pc < word_count;) {
    SCOPED_TRACE(pc);
    auto inst = std::unique_ptr<rocjitsu::Instruction>(decode_valid(*decoder, &words[pc]));
    ASSERT_NE(inst, nullptr);
    ASSERT_GT(inst->size(), 0u);
    ASSERT_EQ(inst->size() % sizeof(uint32_t), 0u);
    ASSERT_LE(pc + inst->size() / sizeof(uint32_t), word_count);
    pc += inst->size() / sizeof(uint32_t);
    actual.push_back(std::move(inst));
  }

  ASSERT_GE(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    SCOPED_TRACE(i);
    expect_cdna3_instruction_matches(*actual[i], expected[i]);
  }
  if (allow_non_nop_tail)
    return;
  for (size_t i = expected.size(); i < actual.size(); ++i) {
    SCOPED_TRACE(i);
    const uint32_t *raw = actual[i]->raw_encoding();
    ASSERT_NE(raw, nullptr);
    EXPECT_EQ(*raw, rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA3));
  }
}

void expect_cdna3_translated_descriptor_vgprs_at_least(const std::vector<uint8_t> &image,
                                                       uint32_t expected_minimum) {
  rocjitsu::AmdGpuCodeObject translated(image.data(), image.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  rocjitsu::KernelDescriptorTranslator parser(ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA3);
  const auto infos = parser.translate_image(image, translated.text_sections()[0]->sectionOffset(),
                                            translated.text_sections()[0]->size(),
                                            rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(infos.size(), 1u);
  EXPECT_GE(infos[0].target_vgpr_count, expected_minimum);
}

std::optional<uint8_t> find_vop2_literal_add(const uint32_t *words, size_t word_count,
                                             uint8_t vsrc1, uint32_t literal) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::Vop2InstLiteralMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0u && actual.op == 52u && actual.src0 == 0xFFu &&
        actual.vsrc1 == vsrc1 && actual.simm32 == literal)
      return static_cast<uint8_t>(actual.vdst);
  }
  return std::nullopt;
}

size_t count_cdna3_s_mov_b32_literal(const uint32_t *words, size_t word_count, uint8_t sdst,
                                     uint32_t literal) {
  size_t count = 0;
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::Sop1MachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x17Du && actual.op == 0u && actual.sdst == sdst &&
        actual.ssrc0 == 0xFFu && words[i + 1] == literal) {
      ++count;
    }
  }
  return count;
}

bool contains_vop3_mov_b32(const uint32_t *words, size_t word_count, uint8_t vdst, uint16_t src0) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::Vop3MachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x34u && actual.op == 321u && actual.vdst == vdst && actual.src0 == src0)
      return true;
  }
  return false;
}

bool contains_flat_global_load(const uint32_t *words, size_t word_count, uint8_t op, uint8_t vdst,
                               uint8_t addr, uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.vdst == vdst &&
        actual.addr == addr && actual.offset == (offset & 0x0FFFu) &&
        actual.pad_12 == ((offset >> 12) & 0x1u) && actual.sc0 == 1u && actual.sc1 == 0u &&
        actual.saddr != 0x7Fu)
      return true;
  }
  return false;
}

bool contains_flat_global_load_acc(const uint32_t *words, size_t word_count, uint8_t op,
                                   uint8_t vdst, uint8_t addr, uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.vdst == vdst &&
        actual.addr == addr && actual.offset == (offset & 0x0FFFu) &&
        actual.pad_12 == ((offset >> 12) & 0x1u) && actual.sc0 == 1u && actual.sc1 == 0u &&
        actual.acc == 1u && actual.saddr != 0x7Fu)
      return true;
  }
  return false;
}

std::optional<uint8_t> find_flat_global_load_saddr(const uint32_t *words, size_t word_count,
                                                   uint8_t op, uint8_t vdst, uint8_t addr,
                                                   uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.vdst == vdst &&
        actual.addr == addr && actual.offset == (offset & 0x0FFFu) &&
        actual.pad_12 == ((offset >> 12) & 0x1u) && actual.sc0 == 1u && actual.sc1 == 0u &&
        actual.saddr != 0x7Fu)
      return static_cast<uint8_t>(actual.saddr);
  }
  return std::nullopt;
}

bool contains_flat_global_load_addr(const uint32_t *words, size_t word_count, uint8_t op,
                                    uint8_t addr, uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.addr == addr &&
        actual.offset == (offset & 0x0FFFu) && actual.pad_12 == ((offset >> 12) & 0x1u) &&
        actual.sc0 == 1u && actual.sc1 == 0u && actual.saddr != 0x7Fu)
      return true;
  }
  return false;
}

std::optional<uint8_t> find_flat_global_load_vdst(const uint32_t *words, size_t word_count,
                                                  uint8_t op, uint8_t addr, uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.addr == addr &&
        actual.offset == (offset & 0x0FFFu) && actual.pad_12 == ((offset >> 12) & 0x1u) &&
        actual.sc0 == 1u && actual.sc1 == 0u && actual.saddr != 0x7Fu)
      return static_cast<uint8_t>(actual.vdst);
  }
  return std::nullopt;
}

std::optional<uint8_t> find_flat_global_load_addr_for_vdst(const uint32_t *words, size_t word_count,
                                                           uint8_t op, uint8_t vdst,
                                                           uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.vdst == vdst &&
        actual.offset == (offset & 0x0FFFu) && actual.pad_12 == ((offset >> 12) & 0x1u) &&
        actual.sc0 == 1u && actual.sc1 == 0u && actual.saddr != 0x7Fu)
      return static_cast<uint8_t>(actual.addr);
  }
  return std::nullopt;
}

bool contains_flat_global_store(const uint32_t *words, size_t word_count, uint8_t op, uint8_t data,
                                uint8_t addr, uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.data == data &&
        actual.addr == addr && actual.offset == (offset & 0x0FFFu) &&
        actual.pad_12 == ((offset >> 12) & 0x1u) && actual.sc0 == 1u && actual.sc1 == 0u &&
        actual.saddr != 0x7Fu)
      return true;
  }
  return false;
}

bool contains_flat_global_store_acc(const uint32_t *words, size_t word_count, uint8_t op,
                                    uint8_t data, uint8_t addr, uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.data == data &&
        actual.addr == addr && actual.offset == (offset & 0x0FFFu) &&
        actual.pad_12 == ((offset >> 12) & 0x1u) && actual.sc0 == 1u && actual.sc1 == 0u &&
        actual.acc == 1u && actual.saddr != 0x7Fu)
      return true;
  }
  return false;
}

std::optional<uint8_t> find_flat_global_store_data(const uint32_t *words, size_t word_count,
                                                   uint8_t op, uint8_t addr, uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.addr == addr &&
        actual.offset == (offset & 0x0FFFu) && actual.pad_12 == ((offset >> 12) & 0x1u) &&
        actual.sc0 == 1u && actual.sc1 == 0u && actual.saddr != 0x7Fu)
      return static_cast<uint8_t>(actual.data);
  }
  return std::nullopt;
}

std::optional<uint8_t> find_flat_global_store_addr_for_data(const uint32_t *words,
                                                            size_t word_count, uint8_t op,
                                                            uint8_t data, uint16_t offset) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.data == data &&
        actual.offset == (offset & 0x0FFFu) && actual.pad_12 == ((offset >> 12) & 0x1u) &&
        actual.sc0 == 1u && actual.sc1 == 0u && actual.saddr != 0x7Fu)
      return static_cast<uint8_t>(actual.addr);
  }
  return std::nullopt;
}

bool contains_flat_global_store_op(const uint32_t *words, size_t word_count, uint8_t op) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.lds == 0u &&
        actual.saddr != 0x7Fu)
      return true;
  }
  return false;
}

bool contains_flat_global_op_with_null_saddr(const uint32_t *words, size_t word_count, uint8_t op) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 2u && actual.lds == 0u &&
        actual.saddr == 0x7Fu)
      return true;
  }
  return false;
}

bool contains_mubuf_lds_op(const uint32_t *words, size_t word_count, uint8_t op) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::MubufMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x38u && actual.op == op && actual.lds != 0u)
      return true;
  }
  return false;
}

bool contains_flat_scratch_dword(const uint32_t *words, size_t word_count, uint8_t op, uint8_t vgpr,
                                 uint16_t offset, bool is_load) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatScratchMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    const uint8_t encoded_vgpr =
        is_load ? static_cast<uint8_t>(actual.vdst) : static_cast<uint8_t>(actual.data);
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 1u && actual.saddr == 0x7Fu &&
        encoded_vgpr == vgpr && actual.offset == offset)
      return true;
  }
  return false;
}

bool contains_flat_scratch_dword_offset(const uint32_t *words, size_t word_count, uint8_t op,
                                        uint16_t offset, bool is_load) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::FlatScratchMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == op && actual.seg == 1u && actual.saddr == 0x7Fu &&
        actual.offset == offset) {
      const bool decoded_load = actual.op == 20u;
      if (decoded_load == is_load)
        return true;
    }
  }
  return false;
}

bool contains_cdna3_vop1(const uint32_t *words, size_t word_count, uint8_t op, uint8_t vdst,
                         uint16_t src0) {
  for (size_t i = 0; i < word_count; ++i) {
    rocjitsu::cdna3::Vop1MachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x3Fu && actual.op == op && actual.vdst == vdst && actual.src0 == src0)
      return true;
  }
  return false;
}

size_t count_cdna3_vop1_writes(const uint32_t *words, size_t word_count, uint8_t op, uint8_t vdst) {
  size_t count = 0;
  for (size_t i = 0; i < word_count; ++i) {
    rocjitsu::cdna3::Vop1MachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x3Fu && actual.op == op && actual.vdst == vdst)
      ++count;
  }
  return count;
}

bool contains_cdna3_vop3p(const uint32_t *words, size_t word_count, uint8_t op, uint8_t vdst,
                          uint16_t src0) {
  for (size_t i = 0; i + 1 < word_count; ++i) {
    rocjitsu::cdna3::Vop3pMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x1A7u && actual.op == op && actual.vdst == vdst && actual.src0 == src0)
      return true;
  }
  return false;
}

std::optional<uint8_t> find_cdna3_vop1_vdst(const uint32_t *words, size_t word_count, uint8_t op,
                                            uint16_t src0) {
  for (size_t i = 0; i < word_count; ++i) {
    rocjitsu::cdna3::Vop1MachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x3Fu && actual.op == op && actual.src0 == src0)
      return static_cast<uint8_t>(actual.vdst);
  }
  return std::nullopt;
}

bool contains_sopp(const uint32_t *words, size_t word_count, uint8_t op) {
  for (size_t i = 0; i < word_count; ++i) {
    rocjitsu::cdna3::SoppMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x17Fu && actual.op == op)
      return true;
  }
  return false;
}

bool contains_smem_load_dwordx2_with_wait(const uint32_t *words, size_t word_count, uint8_t sdata,
                                          uint8_t sbase_sgpr, uint32_t offset) {
  for (size_t i = 0; i + 2 < word_count; ++i) {
    rocjitsu::cdna3::SmemMachineInst actual{};
    std::memcpy(&actual, words + i, sizeof(actual));
    if (actual.encoding == 0x30u && actual.op == 1u && actual.sdata == sdata &&
        actual.sbase == (sbase_sgpr / 2u) && actual.imm == 1u && actual.offset == offset &&
        words[i + 2] == rocjitsu::pack_sopp(cdna3::kSWaitcntSopp, 0))
      return true;
  }
  return false;
}

void expect_cdna3_translated_descriptor_sgprs_eq(const std::vector<uint8_t> &image,
                                                 uint32_t expected) {
  rocjitsu::AmdGpuCodeObject translated(image.data(), image.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  rocjitsu::KernelDescriptorTranslator parser(ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA3);
  const auto infos = parser.translate_image(image, translated.text_sections()[0]->sectionOffset(),
                                            translated.text_sections()[0]->size(),
                                            rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(infos.size(), 1u);
  EXPECT_EQ(infos[0].target_sgpr_count, expected);
}

struct InstructionWordsView {
  const uint32_t *words = nullptr;
  size_t word_count = 0;
};

uint32_t cdna3_descriptor_vgpr_allocation_count(
    const rocjitsu::test_support::TestKernelDescriptor &descriptor) {
  const uint32_t granulated =
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                      rocr::llvm::amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  return (granulated + 1u) * 8u;
}

uint32_t
cdna3_descriptor_sgpr_count(const rocjitsu::test_support::TestKernelDescriptor &descriptor) {
  const uint32_t granulated =
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                      rocr::llvm::amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  return (granulated + 1u) * 8u;
}

std::optional<rocjitsu::VirtualLdsKernelMetadata>
find_virtual_lds_metadata_record_for_test(const std::vector<uint8_t> &image,
                                          std::string_view kernel_name = "kernel") {
  rocjitsu::AmdGpuCodeObject translated(image.data(), image.size());
  if (!translated.is_valid()) {
    ADD_FAILURE() << "translated code object is invalid";
    return std::nullopt;
  }

  const auto *metadata_section =
      rocjitsu::test_support::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  if (metadata_section == nullptr) {
    ADD_FAILURE() << "missing " << rocjitsu::kVirtualLdsMetadataSectionName << " section";
    return std::nullopt;
  }

  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  if (!parsed.has_value()) {
    ADD_FAILURE() << "could not parse virtual-LDS metadata";
    return std::nullopt;
  }

  const auto it =
      std::ranges::find_if(*parsed, [&](const rocjitsu::VirtualLdsKernelMetadata &record) {
        return record.kernel_name == kernel_name;
      });
  if (it == parsed->end()) {
    ADD_FAILURE() << "missing virtual-LDS metadata for " << kernel_name;
    return std::nullopt;
  }
  return *it;
}

std::optional<rocjitsu::SidecarVariantMetadata>
find_sidecar_metadata_record_for_test(const std::vector<uint8_t> &image,
                                      std::string_view kernel_name = "kernel") {
  rocjitsu::AmdGpuCodeObject translated(image.data(), image.size());
  const auto *section =
      rocjitsu::test_support::find_section(translated, rocjitsu::kSidecarMetadataSectionName);
  if (section == nullptr)
    return std::nullopt;
  const auto parsed = rocjitsu::parse_sidecar_metadata(
      {reinterpret_cast<const uint8_t *>(section->data()), section->size()});
  if (!parsed)
    return std::nullopt;
  const auto record = std::ranges::find_if(*parsed, [&](const auto &candidate) {
    return candidate.kernel_name == kernel_name &&
           candidate.variant_name == rocjitsu::kVirtualLdsSidecarVariantName;
  });
  return record == parsed->end() ? std::nullopt : std::optional{*record};
}

std::optional<rocjitsu::KernargExtensionMetadata>
find_kernarg_extension_record_for_test(const std::vector<uint8_t> &image,
                                       std::string_view kernel_name = "kernel") {
  rocjitsu::AmdGpuCodeObject translated(image.data(), image.size());
  const auto *section = rocjitsu::test_support::find_section(
      translated, rocjitsu::kKernargExtensionMetadataSectionName);
  if (section == nullptr)
    return std::nullopt;
  const auto parsed = rocjitsu::parse_kernarg_extension_metadata(
      {reinterpret_cast<const uint8_t *>(section->data()), section->size()});
  if (!parsed)
    return std::nullopt;
  const auto record = std::ranges::find_if(*parsed, [&](const auto &candidate) {
    return candidate.kernel_name == kernel_name &&
           candidate.variant_name == rocjitsu::kVirtualLdsSidecarVariantName;
  });
  return record == parsed->end() ? std::nullopt : std::optional{*record};
}

std::optional<rocjitsu::KernargExtensionLayout>
make_kernarg_extension_layout_for_test(const rocjitsu::KernargExtensionMetadata &metadata) {
  std::vector<rocjitsu::KernargExtensionPayloadLayout> payloads;
  for (const auto &payload : metadata.payloads)
    payloads.push_back({.size = payload.size, .alignment = payload.alignment});
  return rocjitsu::make_kernarg_extension_layout(metadata.original_kernarg_size, payloads);
}

std::optional<rocjitsu::test_support::TestKernelDescriptor>
read_descriptor_at_loaded_vaddr_for_test(const std::vector<uint8_t> &image, uint64_t vaddr) {
  const auto descriptor_file_offset =
      rocjitsu::test_support::loaded_vaddr_to_file_offset(image, vaddr);
  if (!descriptor_file_offset.has_value()) {
    ADD_FAILURE() << "could not map descriptor vaddr 0x" << std::hex << vaddr << std::dec;
    return std::nullopt;
  }
  if (*descriptor_file_offset > image.size() ||
      sizeof(rocjitsu::test_support::TestKernelDescriptor) >
          image.size() - *descriptor_file_offset) {
    ADD_FAILURE() << "descriptor vaddr 0x" << std::hex << vaddr << std::dec
                  << " maps outside the ELF image";
    return std::nullopt;
  }
  return rocjitsu::test_support::read_elf_struct_for_test<
      rocjitsu::test_support::TestKernelDescriptor>(image, *descriptor_file_offset);
}

std::optional<rocjitsu::test_support::TestKernelDescriptor>
read_virtual_lds_sidecar_descriptor_for_test(const std::vector<uint8_t> &image,
                                             std::string_view kernel_name = "kernel") {
  const auto record = find_sidecar_metadata_record_for_test(image, kernel_name);
  if (!record.has_value())
    return std::nullopt;

  return read_descriptor_at_loaded_vaddr_for_test(image, record->variant_descriptor_vaddr);
}

std::optional<InstructionWordsView>
virtual_lds_sidecar_entry_words_for_test(const std::vector<uint8_t> &image,
                                         std::string_view kernel_name = "kernel") {
  rocjitsu::AmdGpuCodeObject translated(image.data(), image.size());
  if (!translated.is_valid()) {
    ADD_FAILURE() << "translated code object is invalid";
    return std::nullopt;
  }

  const auto record = find_sidecar_metadata_record_for_test(image, kernel_name);
  const auto descriptor = read_virtual_lds_sidecar_descriptor_for_test(image, kernel_name);
  if (!record.has_value() || !descriptor.has_value())
    return std::nullopt;

  const int64_t entry_delta =
      rocjitsu::test_support::read_kernel_descriptor_entry_offset(&*descriptor);
  const int64_t entry_vaddr_signed =
      static_cast<int64_t>(record->variant_descriptor_vaddr) + entry_delta;
  if (entry_vaddr_signed < 0) {
    ADD_FAILURE() << "sidecar entry resolves before address zero";
    return std::nullopt;
  }

  const uint64_t entry_vaddr = static_cast<uint64_t>(entry_vaddr_signed);
  const auto entry_file_offset =
      rocjitsu::test_support::loaded_vaddr_to_file_offset(image, entry_vaddr);
  if (!entry_file_offset.has_value()) {
    ADD_FAILURE() << "could not map sidecar entry vaddr 0x" << std::hex << entry_vaddr << std::dec;
    return std::nullopt;
  }

  for (const auto *text : translated.text_sections()) {
    const uint64_t text_begin = text->sectionOffset();
    const uint64_t text_end = text_begin + text->size();
    if (*entry_file_offset < text_begin || *entry_file_offset >= text_end)
      continue;

    const uint64_t byte_count = text_end - *entry_file_offset;
    return InstructionWordsView{
        reinterpret_cast<const uint32_t *>(image.data() + *entry_file_offset),
        static_cast<size_t>(byte_count / sizeof(uint32_t)),
    };
  }

  ADD_FAILURE() << "sidecar entry does not point inside a text section";
  return std::nullopt;
}

void expect_cdna3_sidecar_descriptor_sgprs_eq(const std::vector<uint8_t> &image,
                                              uint32_t expected) {
  const auto sidecar_kd = read_virtual_lds_sidecar_descriptor_for_test(image);
  ASSERT_TRUE(sidecar_kd.has_value());
  EXPECT_EQ(cdna3_descriptor_sgpr_count(*sidecar_kd), expected);
}

void expect_cdna3_sidecar_descriptor_vgprs_at_least(const std::vector<uint8_t> &image,
                                                    uint32_t expected_minimum) {
  const auto sidecar_kd = read_virtual_lds_sidecar_descriptor_for_test(image);
  ASSERT_TRUE(sidecar_kd.has_value());
  EXPECT_GE(cdna3_descriptor_vgpr_allocation_count(*sidecar_kd), expected_minimum);
}

uint32_t build_s_getpc_b64(uint16_t sdst, rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rocjitsu::build_sop1_encoding(arch, 0x47, sdst, 0);
  default:
    return rocjitsu::build_sop1_encoding(arch, 0x1c, sdst, 0);
  }
}

uint32_t build_s_setpc_b64(uint16_t ssrc0, rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rocjitsu::build_sop1_encoding(arch, 0x48, 0, ssrc0);
  default:
    return rocjitsu::build_sop1_encoding(arch, 0x1d, 0, ssrc0);
  }
}

uint32_t build_s_swappc_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rocjitsu::build_sop1_encoding(arch, 0x49, sdst, ssrc0);
  default:
    return rocjitsu::build_sop1_encoding(arch, 0x1e, sdst, ssrc0);
  }
}

uint32_t build_s_call_b64(uint16_t sdst, int16_t simm16) {
  return cdna4::build_sopk(cdna4::kSCallB64Sopk, {.simm16 = static_cast<uint16_t>(simm16),
                                                  .sdst = static_cast<uint8_t>(sdst)})[0];
}

uint32_t build_s_trap(uint16_t simm16) {
  // CDNA1-4 encode S_TRAP at SOPP opcode 0x12. This helper is intentionally
  // local to the CDNA4->CDNA3 tests below; RDNA3+ uses a different SOPP opcode.
  return cdna4::build_sopp(cdna4::kSTrapSopp, {.simm16 = simm16})[0];
}

uint32_t build_s_add_u32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  return cdna4::build_sop2(cdna4::kSAddU32Sop2, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                                 .ssrc1 = static_cast<uint8_t>(ssrc1),
                                                 .sdst = static_cast<uint8_t>(sdst)})[0];
}

uint32_t build_s_addc_u32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  return cdna4::build_sop2(cdna4::kSAddcU32Sop2, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                                  .ssrc1 = static_cast<uint8_t>(ssrc1),
                                                  .sdst = static_cast<uint8_t>(sdst)})[0];
}

std::array<uint32_t, 2> build_cdna4_smem_load(uint8_t op, uint8_t sdata, uint8_t sbase,
                                              uint32_t byte_offset) {
  rocjitsu::cdna4::SmemMachineInst inst{};
  inst.encoding = 0x30;
  inst.op = op;
  inst.sbase = (sbase / 2) & 0x3f;
  inst.sdata = sdata & 0x7f;
  inst.imm = 1;
  inst.offset = byte_offset & 0x1fffff;
  std::array<uint32_t, 2> words{};
  std::memcpy(words.data(), &inst, sizeof(inst));
  return words;
}

std::vector<std::unique_ptr<rocjitsu::Instruction>>
decode_text_instructions(const rocjitsu::Section &text, rj_code_arch_t arch) {
  std::vector<std::unique_ptr<rocjitsu::Instruction>> decoded;
  auto decoder = rocjitsu::Decoder::create(arch);
  if (!decoder)
    return decoded;

  const auto *words = reinterpret_cast<const rj_code_binary_inst_t *>(text.data());
  const size_t word_count = text.size() / sizeof(rj_code_binary_inst_t);
  size_t word_offset = 0;
  while (word_offset < word_count) {
    std::unique_ptr<rocjitsu::Instruction> inst(
        decode_valid(*decoder, words + word_offset, word_offset * sizeof(rj_code_binary_inst_t)));
    if (!inst)
      break;
    word_offset += static_cast<size_t>(inst->size()) / sizeof(rj_code_binary_inst_t);
    decoded.push_back(std::move(inst));
  }
  return decoded;
}

// --- Synthetic BinaryTranslator integration tests ---
TEST(BinaryTranslatorE2E, TranslatesMultiKernelCodeObject) {
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_two_kernel_descriptors();
  rocjitsu::AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());
  const auto *text = co.text_sections()[0];
  const auto *rodata = rocjitsu::test_support::find_section(co, ".rodata");
  ASSERT_NE(rodata, nullptr);

  rocjitsu::KernelDescriptorTranslator original_parser(ROCJITSU_CODE_ARCH_CDNA4,
                                                       ROCJITSU_CODE_ARCH_RDNA4);
  const auto original_infos = original_parser.translate_image(
      image, text->sectionOffset(), text->size(), rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(original_infos.size(), 2u);

  std::vector<uint64_t> original_entries;
  std::vector<uint64_t> original_descriptor_offsets;
  for (const auto &info : original_infos) {
    original_entries.push_back(info.entry_text_offset);
    original_descriptor_offsets.push_back(info.descriptor_file_offset);
  }
  std::ranges::sort(original_entries);
  std::ranges::sort(original_descriptor_offsets);
  EXPECT_EQ(original_entries, (std::vector<uint64_t>{0, sizeof(uint32_t)}));
  EXPECT_EQ(original_descriptor_offsets,
            (std::vector<uint64_t>{rodata->sectionOffset(),
                                   rodata->sectionOffset() +
                                       rocjitsu::test_support::kKernelDescriptorSize}));

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(co);
  ASSERT_FALSE(result.elf_bytes.empty());
  EXPECT_TRUE(result.ok());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  EXPECT_EQ(translated.text_sections()[0]->size(), text->size());
  EXPECT_EQ(rocjitsu::test_support::find_section(translated, ".rj_translations"), nullptr)
      << "this fixture should exercise multi-kernel descriptor handling without code caves";

  const auto *translated_header =
      reinterpret_cast<const rocjitsu::Elf64_Ehdr *>(result.elf_bytes.data());
  EXPECT_EQ(translated_header->e_flags & rocjitsu::EF_AMDGPU_MACH,
            rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1200);

  rocjitsu::KernelDescriptorTranslator translated_parser(ROCJITSU_CODE_ARCH_RDNA4,
                                                         ROCJITSU_CODE_ARCH_RDNA4);
  const auto translated_infos = translated_parser.translate_image(
      result.elf_bytes, translated.text_sections()[0]->sectionOffset(),
      translated.text_sections()[0]->size(), rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(translated_infos.size(), 2u);

  std::vector<uint64_t> translated_entries;
  std::vector<uint64_t> translated_descriptor_offsets;
  for (const auto &info : translated_infos) {
    translated_entries.push_back(info.entry_text_offset);
    translated_descriptor_offsets.push_back(info.descriptor_file_offset);
  }
  std::ranges::sort(translated_entries);
  std::ranges::sort(translated_descriptor_offsets);
  EXPECT_EQ(translated_entries, (std::vector<uint64_t>{0, sizeof(uint32_t)}));
  EXPECT_EQ(translated_descriptor_offsets, original_descriptor_offsets);
}

TEST(BinaryTranslatorE2E, SkipFailedKernelKeepsIndependentKernelTranslating) {
  constexpr uint16_t kReturnSreg = 0;
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA4),
      kCdna4SEndpgm,
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_two_kernel_descriptors(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.skip_failed_kernels = true;
  options.debug_continue_after_failure = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  // A skipped kernel is only a warning, so ok() stays true, but the artifact is
  // NOT dispatchable: its s_endpgm stub would silently complete if dispatched. Code
  // object output paths (CLI, API consumers) must gate on dispatchable().
  EXPECT_FALSE(result.dispatchable());
  ASSERT_FALSE(result.elf_bytes.empty());
  const auto skipped = std::ranges::find_if(result.diagnostics, [](const auto &diagnostic) {
    return diagnostic.kind == rocjitsu::DiagnosticKind::KernelSkipped;
  });
  ASSERT_NE(skipped, result.diagnostics.end());
  EXPECT_EQ(skipped->severity, rocjitsu::DiagnosticSeverity::Warning);
  EXPECT_EQ(skipped->guest_offset, std::optional<uint64_t>(0));
  EXPECT_NE(skipped->message.find("kernel0"), std::string::npos);
  EXPECT_NE(skipped->message.find("indirect branch"), std::string::npos);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *translated_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(translated_words[0], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3))
      << "the failed kernel body exits safely if it is dispatched";
  EXPECT_NE(skipped->message.find("S_ENDPGM"), std::string::npos);
  EXPECT_NE(skipped->message.find("INVALID OUTPUTS"), std::string::npos);

  rocjitsu::KernelDescriptorTranslator parser(ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA3);
  const auto infos = parser.translate_image(
      result.elf_bytes, translated.text_sections()[0]->sectionOffset(),
      translated.text_sections()[0]->size(), rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(infos.size(), 2u);
  std::vector<uint64_t> entries;
  for (const auto &info : infos)
    entries.push_back(info.entry_text_offset);
  std::ranges::sort(entries);
  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0], 0u);
  EXPECT_EQ(entries[1], sizeof(uint32_t))
      << "the independent kernel follows the one-instruction skipped body";
  ASSERT_EQ(entries[1] % sizeof(uint32_t), 0u);
  ASSERT_LT(entries[1], translated.text_sections()[0]->size());
  EXPECT_EQ(translated_words[entries[1] / sizeof(uint32_t)],
            rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3))
      << "the independent kernel still translates instead of using the skipped-kernel trap";
}

TEST(BinaryTranslatorE2E, OversizedTargetLdsDescriptorEmitsVirtualVariantInsteadOfSkipping) {
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_two_kernel_descriptors();
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), 2 * rocjitsu::test_support::kKernelDescriptorSize);

  // gfx950/CDNA4 can advertise 160 KiB LDS kernels, but gfx942/CDNA3 can only
  // dispatch 64 KiB per workgroup in the checked-in topology. Keep the normal
  // descriptor for launches that fit, and emit a virtual-LDS sidecar descriptor
  // for runtime packet rewriting when static plus dynamic LDS exceeds the host.
  rocjitsu::test_support::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() +
          offsetof(rocjitsu::test_support::TestKernelDescriptor, group_segment_fixed_size),
      105600u);
  rocjitsu::test_support::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() +
          offsetof(rocjitsu::test_support::TestKernelDescriptor, kernarg_size),
      16u);
  uint32_t source_rsrc2 = 0;
  AMDHSA_BITS_SET(source_rsrc2, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                  1);
  AMDHSA_BITS_SET(source_rsrc2, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                  1);
  AMDHSA_BITS_SET(source_rsrc2, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
  // COMPUTE_PGM_RSRC2.LDS_SIZE is programmed by CP from the dispatch packet, not
  // by the code-object descriptor. Seed a stale guest value here so the test
  // proves translated normal and virtual descriptors do not preserve it.
  AMDHSA_BITS_SET(source_rsrc2, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE, 22);
  rocjitsu::test_support::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() +
          offsetof(rocjitsu::test_support::TestKernelDescriptor, compute_pgm_rsrc2),
      source_rsrc2);
  uint16_t source_properties = 0;
  AMDHSA_BITS_SET(source_properties,
                  rocr::llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
  rocjitsu::test_support::write_value_for_test<uint16_t>(
      image,
      rodata->sectionOffset() +
          offsetof(rocjitsu::test_support::TestKernelDescriptor, kernel_code_properties),
      source_properties);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.skip_failed_kernels = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *translated_rodata = rocjitsu::test_support::find_section(translated, ".rodata");
  ASSERT_NE(translated_rodata, nullptr);
  const auto normal_kd = rocjitsu::test_support::read_elf_struct_for_test<
      rocjitsu::test_support::TestKernelDescriptor>(result.elf_bytes,
                                                    translated_rodata->sectionOffset());
  EXPECT_EQ(normal_kd.group_segment_fixed_size, 105600u);
  EXPECT_EQ(normal_kd.kernarg_size, 16u);
  EXPECT_EQ(AMDHSA_BITS_GET(normal_kd.compute_pgm_rsrc2,
                            rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE),
            0u);

  const auto *metadata_section =
      rocjitsu::test_support::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  const auto &record = parsed->front();
  const auto sidecar = find_sidecar_metadata_record_for_test(result.elf_bytes, "kernel0");
  const auto extension = find_kernarg_extension_record_for_test(result.elf_bytes, "kernel0");
  ASSERT_TRUE(sidecar.has_value());
  ASSERT_TRUE(extension.has_value());
  const auto wrapper_layout = make_kernarg_extension_layout_for_test(*extension);
  ASSERT_TRUE(wrapper_layout.has_value());
  EXPECT_EQ(record.kernel_name, "kernel0");
  EXPECT_EQ(sidecar->normal_descriptor_vaddr, translated.kernel_descriptor_offset("kernel0"));
  EXPECT_NE(sidecar->variant_descriptor_vaddr, sidecar->normal_descriptor_vaddr);
  EXPECT_EQ(record.static_lds_bytes, 105600u);
  EXPECT_EQ(extension->original_kernarg_size, 16u);
  ASSERT_EQ(wrapper_layout->payload_offsets.size(), 1u);
  EXPECT_EQ(wrapper_layout->payload_offsets.front(), 24u);
  EXPECT_NE(record.virtual_lds_base_sgpr, 0u);
  EXPECT_NE(record.flags & rocjitsu::kVirtualLdsFlagRuntimeStateBlock, 0u);
  EXPECT_NE(record.flags & rocjitsu::kVirtualLdsFlagWorkgroupIdX, 0u);
  EXPECT_NE(record.flags & rocjitsu::kVirtualLdsFlagWorkgroupIdY, 0u);
  EXPECT_EQ(record.flags & rocjitsu::kVirtualLdsFlagWorkgroupIdZ, 0u);

  const auto virtual_descriptor_offset = rocjitsu::test_support::loaded_vaddr_to_file_offset(
      result.elf_bytes, sidecar->variant_descriptor_vaddr);
  ASSERT_TRUE(virtual_descriptor_offset.has_value());
  const auto virtual_kd = rocjitsu::test_support::read_elf_struct_for_test<
      rocjitsu::test_support::TestKernelDescriptor>(result.elf_bytes, *virtual_descriptor_offset);
  EXPECT_EQ(virtual_kd.group_segment_fixed_size, 0u);
  EXPECT_EQ(virtual_kd.kernarg_size, 48u);
  EXPECT_EQ(AMDHSA_BITS_GET(virtual_kd.compute_pgm_rsrc2,
                            rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE),
            0u);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarPreservesReservedBytesAfterTextGrowth) {
  // Regression for the sidecar being copied from a stale pre-.text-growth file
  // offset. The DS-read fixture lowers to a flat-global load, which GROWS .text
  // and thus shifts the descriptor section that follows it. The sidecar template
  // must come from a snapshot of the source descriptor, not from re-reading the
  // now-shifted offset (which would read relocated instruction bytes). Seed the
  // reserved regions with recognizable patterns and assert the emitted sidecar
  // carries them unchanged.
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);

  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  // Seed reserved1[20] with a distinctive byte pattern. reserved1 is a field DBT
  // never rewrites, so it is only correct in the sidecar if the template was
  // copied from the (pre-growth) source descriptor snapshot.
  const uint64_t reserved1_off =
      rodata->sectionOffset() + offsetof(rocjitsu::test_support::TestKernelDescriptor, reserved1);
  std::array<uint8_t, 20> reserved1_pattern{};
  for (size_t i = 0; i < reserved1_pattern.size(); ++i)
    reserved1_pattern[i] = static_cast<uint8_t>(0xA0 + i);
  for (size_t i = 0; i < reserved1_pattern.size(); ++i)
    image[reserved1_off + i] = reserved1_pattern[i];

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        rocjitsu::BinaryTranslatorOptions{});
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  // Confirm .text actually grew (otherwise the regression could not occur).
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  EXPECT_GT(translated.text_sections()[0]->size(), 3u * sizeof(uint32_t));

  const auto sidecar_kd = read_virtual_lds_sidecar_descriptor_for_test(result.elf_bytes, "kernel");
  ASSERT_TRUE(sidecar_kd.has_value());
  EXPECT_TRUE(std::equal(std::begin(sidecar_kd->reserved1), std::end(sidecar_kd->reserved1),
                         reserved1_pattern.begin()))
      << "sidecar reserved1 was not copied from the source descriptor snapshot";
}

TEST(BinaryTranslatorE2E, VirtualLdsWrapperIgnoresInlineKernargLoadsBeyondDescriptorSize) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  constexpr uint16_t kKernargSgpr = 0;
  constexpr uint16_t kInlineInt0 = 128;
  constexpr uint16_t kInlineInt16 = 144;
  constexpr uint32_t kDescriptorKernargSize = 48;
  constexpr uint32_t kExpectedStateOffset = 56;
  constexpr uint32_t kExpectedWrapperBytes = 80;
  const auto load_header = build_cdna4_smem_load(/*op=*/0, /*sdata=*/16, kKernargSgpr,
                                                 /*byte_offset=*/0);
  const auto load_userargs = build_cdna4_smem_load(/*op=*/4, /*sdata=*/20, kKernargSgpr,
                                                   /*byte_offset=*/0x40);

  std::vector<uint32_t> words = {
      load_header[0],
      load_header[1],
      build_s_add_u32(kKernargSgpr, kKernargSgpr, kInlineInt16),
      build_s_addc_u32(kKernargSgpr + 1, kKernargSgpr + 1, kInlineInt0),
      load_userargs[0],
      load_userargs[1],
      kCdna4SEndpgm,
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image, kDescriptorKernargSize);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));
  rocjitsu::test_support::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() +
          offsetof(rocjitsu::test_support::TestKernelDescriptor, group_segment_fixed_size),
      6144u);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslatorOptions options;
  options.skip_failed_kernels = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *translated_rodata = rocjitsu::test_support::find_section(translated, ".rodata");
  ASSERT_NE(translated_rodata, nullptr);
  const auto normal_kd = rocjitsu::test_support::read_elf_struct_for_test<
      rocjitsu::test_support::TestKernelDescriptor>(result.elf_bytes,
                                                    translated_rodata->sectionOffset());
  EXPECT_EQ(normal_kd.kernarg_size, kDescriptorKernargSize);

  const auto *metadata_section =
      rocjitsu::test_support::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  const auto sidecar = find_sidecar_metadata_record_for_test(result.elf_bytes);
  const auto extension = find_kernarg_extension_record_for_test(result.elf_bytes);
  ASSERT_TRUE(sidecar.has_value());
  ASSERT_TRUE(extension.has_value());
  const auto wrapper_layout = make_kernarg_extension_layout_for_test(*extension);
  ASSERT_TRUE(wrapper_layout.has_value());
  EXPECT_EQ(extension->original_kernarg_size, kDescriptorKernargSize);
  ASSERT_EQ(wrapper_layout->payload_offsets.size(), 1u);
  EXPECT_EQ(wrapper_layout->payload_offsets.front(), kExpectedStateOffset);

  const auto virtual_descriptor_offset = rocjitsu::test_support::loaded_vaddr_to_file_offset(
      result.elf_bytes, sidecar->variant_descriptor_vaddr);
  ASSERT_TRUE(virtual_descriptor_offset.has_value());
  const auto virtual_kd = rocjitsu::test_support::read_elf_struct_for_test<
      rocjitsu::test_support::TestKernelDescriptor>(result.elf_bytes, *virtual_descriptor_offset);
  EXPECT_EQ(virtual_kd.kernarg_size, kExpectedWrapperBytes);
}

TEST(BinaryTranslatorE2E, VirtualLdsWrapperIgnoresDirectKernargLoadsInSharedHelperBlocks) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  constexpr uint16_t kKernargSgpr = 0;
  constexpr uint32_t kDescriptorKernargSize = 0x90;
  constexpr uint32_t kExpectedStateOffset = 0x98;
  constexpr uint32_t kExpectedWrapperBytes = 0xB0;
  const auto load_helper_tail =
      build_cdna4_smem_load(/*op=*/0, /*sdata=*/16, kKernargSgpr, /*byte_offset=*/0xA0);

  // Model Tensile-style helper code that is present in the code object but not
  // reached by the local descriptor CFG. The virtual-LDS dispatcher still must
  // not size its wrapper prefix from a decoded direct kernarg load range from
  // the same ABI kernarg SGPR pair.
  std::vector<uint32_t> words = {
      rocjitsu::pack_sopp(/*op=*/5, /*simm16=*/2),
      load_helper_tail[0],
      load_helper_tail[1],
      kCdna4SEndpgm,
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image, kDescriptorKernargSize);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));
  rocjitsu::test_support::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() +
          offsetof(rocjitsu::test_support::TestKernelDescriptor, group_segment_fixed_size),
      6144u);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslatorOptions options;
  options.skip_failed_kernels = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *translated_rodata = rocjitsu::test_support::find_section(translated, ".rodata");
  ASSERT_NE(translated_rodata, nullptr);
  const auto normal_kd = rocjitsu::test_support::read_elf_struct_for_test<
      rocjitsu::test_support::TestKernelDescriptor>(result.elf_bytes,
                                                    translated_rodata->sectionOffset());
  EXPECT_EQ(normal_kd.kernarg_size, kDescriptorKernargSize);

  const auto *metadata_section =
      rocjitsu::test_support::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  const auto sidecar = find_sidecar_metadata_record_for_test(result.elf_bytes);
  const auto extension = find_kernarg_extension_record_for_test(result.elf_bytes);
  ASSERT_TRUE(sidecar.has_value());
  ASSERT_TRUE(extension.has_value());
  const auto wrapper_layout = make_kernarg_extension_layout_for_test(*extension);
  ASSERT_TRUE(wrapper_layout.has_value());
  EXPECT_EQ(extension->original_kernarg_size, kDescriptorKernargSize);
  ASSERT_EQ(wrapper_layout->payload_offsets.size(), 1u);
  EXPECT_EQ(wrapper_layout->payload_offsets.front(), kExpectedStateOffset);

  const auto virtual_descriptor_offset = rocjitsu::test_support::loaded_vaddr_to_file_offset(
      result.elf_bytes, sidecar->variant_descriptor_vaddr);
  ASSERT_TRUE(virtual_descriptor_offset.has_value());
  const auto virtual_kd = rocjitsu::test_support::read_elf_struct_for_test<
      rocjitsu::test_support::TestKernelDescriptor>(result.elf_bytes, *virtual_descriptor_offset);
  EXPECT_EQ(virtual_kd.kernarg_size, kExpectedWrapperBytes);
}

TEST(BinaryTranslatorE2E, Gfx1250LongDirectBranchGrowthIsIdempotent) {
  // Each selected DS2 becomes two two-word operations plus a wait. Choose the
  // count so the first target starts one dword below SOPP's forward limit.
  // Growing the already-out-of-range second branch then shifts that near target
  // and forces a second monotonic layout round; both branches must end up long.
  constexpr size_t kSoppMaxForwardDwords = 0x7fff;
  constexpr size_t kExpandedDs2Words = 5;
  constexpr size_t kNearExpansionCount = (kSoppMaxForwardDwords - 2) / kExpandedDs2Words;
  static_assert(1 + kNearExpansionCount * kExpandedDs2Words == kSoppMaxForwardDwords - 1);
  constexpr size_t kExpansionCount = kNearExpansionCount + 1;
  constexpr size_t kNearTargetWord = 2 + kNearExpansionCount * 2;
  constexpr size_t kFarTargetWord = 2 + kExpansionCount * 2;
  constexpr auto ds2 =
      cdna5::build_vds(cdna5::kDsStore2addrB32Vds,
                       {.offset0 = 3, .offset1 = 5, .addr = 20, .data0 = 30, .data1 = 40});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  std::vector<uint32_t> words;
  words.reserve(kFarTargetWord + 1);
  words.push_back(cdna5::build_sopp(cdna5::kSCbranchScc0Sopp,
                                    {.simm16 = static_cast<uint16_t>(kNearTargetWord - 1)})[0]);
  words.push_back(cdna5::build_sopp(cdna5::kSCbranchScc0Sopp,
                                    {.simm16 = static_cast<uint16_t>(kFarTargetWord - 2)})[0]);
  for (size_t i = 0; i < kExpansionCount; ++i) {
    words.push_back(ds2[0]);
    words.push_back(ds2[1]);
  }
  words.push_back(kGfx1250SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const uint32_t marker = rocjitsu::build_s_nop(rocjitsu::kLongDirectBranchMarkerNopImmediate,
                                                ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ((target_words[0] >> 16) & 0x7fu, cdna5::kSCbranchScc1Sopp);
  EXPECT_EQ(target_words[1], marker);
  const auto translated_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_EQ(std::count(target_words, target_words + translated_word_count, marker), 2);

  rocjitsu::BinaryTranslator verifier(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto second = verifier.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250LongBranchReportsSgprExhaustionAfterSemanticExpansion) {
  using namespace rocr::llvm::amdhsa;

  constexpr size_t kTargetWord = 0x8000;
  constexpr uint16_t kLiveSgprs = 104;
  constexpr auto conversion =
      cdna5::build_vop3(cdna5::kVCvtF32Fp8Vop3, {.vdst = 30, .clamp = 1, .src0 = 256 + 22});
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  std::vector<uint32_t> words;
  words.reserve(kTargetWord + 1);
  words.push_back(cdna5::build_sopp(cdna5::kSCbranchScc0Sopp,
                                    {.simm16 = static_cast<uint16_t>(kTargetWord - 1u)})[0]);
  words.insert(words.end(), conversion.begin(), conversion.end());
  for (uint16_t sgpr = 0; sgpr < kLiveSgprs; ++sgpr) {
    words.push_back(cdna5::build_sop1(cdna5::kSMovB32Sop1,
                                      {.ssrc0 = static_cast<uint8_t>(sgpr), .sdst = 125})[0]);
  }
  words.resize(kTargetWord, kGfx1250SNop);
  words.push_back(kGfx1250SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  auto descriptor = rocjitsu::test_support::read_kernel_descriptor_for_test(rodata->data());
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  12);
  rocjitsu::test_support::write_kernel_descriptor_for_test(image.data() + rodata->sectionOffset(),
                                                           descriptor);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ResourceLimit,
      "long direct branch requires an additional descriptor-backed SGPR pair after semantic "
      "expansion"));
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &item) {
    return item.kind == rocjitsu::DiagnosticKind::ResourceLimit;
  });
  ASSERT_NE(diagnostic, result.diagnostics.end());
  EXPECT_EQ(diagnostic->guest_offset, std::optional<uint64_t>(0))
      << "the resource diagnostic must identify the branch that could not be expanded";
}

TEST(BinaryTranslatorE2E, Gfx1250CompactConditionalLayoutIsIdempotent) {
  constexpr size_t kTargetWord = 20000;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  std::vector<uint32_t> words;
  words.reserve(kTargetWord + 1);
  words.push_back(
      rocjitsu::pack_sopp(cdna5::kSCbranchExecnzSopp, static_cast<uint16_t>(kTargetWord - 1)));
  for (size_t i = 1; i < kTargetWord; ++i)
    words.push_back(rocjitsu::pack_sopp(cdna5::kSCbranchScc0Sopp, 0));
  words.push_back(kGfx1250SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], rocjitsu::pack_sopp(cdna5::kSCbranchExecnzSopp,
                                                 static_cast<uint16_t>(kTargetWord - 1)));

  rocjitsu::BinaryTranslator verifier(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto second = verifier.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

// A sized FUNC symbol over a compiler long jump, translated twice, asserting the WHOLE ELF image
// rather than just `.text` -- metadata drift is invisible to a text-only comparison.
//
// Scope warning, stated because the name would otherwise overclaim: this does NOT reproduce the
// st_size overshoot fixed in binary_translator.cpp (block ends inside a preserved marked window
// mapping to the running output size). That needs the symbol's extent to end exactly on the
// window's consumer offset in the SECOND pass's view, which depends on where the first pass placed
// the regenerated window -- not something this fixture can pin without hard-coding a layout. It
// was verified against the fixture with the fix reverted and still passed. The overshoot is
// currently covered only by --verify-idempotence on a large device image; a targeted case for it
// is still missing.
// An address-taken body is emitted once, as an adopted root, so a scope that does not own it must
// not clone it -- cloning is what made re-translation diverge. That reasoning covers the INDIRECT
// edges only. A direct s_call_b64 leaves a BranchFixup in the calling scope, and
// patch_direct_branch_fixups() resolves it against that scope's own layout, so dropping the body
// there makes the call unresolvable. Kernel 1 calls the helper directly; the pointer slot adopts it
// into kernel 0's scope.
TEST(BinaryTranslatorE2E, Gfx1250KeepsDirectlyCalledHelperThatIsAlsoAddressTaken) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  constexpr uint16_t kReturnSreg = 30;
  const std::vector<uint32_t> words = {
      kGfx1250SEndpgm, // word 0: kernel 0 entry
      kGfx1250SNop,    // word 1: padding
      rocjitsu::build_s_call_b64(kReturnSreg, 1, ROCJITSU_CODE_ARCH_CDNA5), // word 2: kernel 1
      kGfx1250SEndpgm,                                                      // word 3
      kGfx1250SNop,                                                         // word 4: helper
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),   // word 5: return
  };

  auto image =
      rocjitsu::test_support::make_minimal_amdgpu_elf_with_two_kernels_and_function_pointers(
          words, /*kernel1_entry_word=*/2, {{.offset_word = 4, .words = 2}});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  rocjitsu::BinaryTranslator verifier(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto second = verifier.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

// block_for_offset() answers with the block CONTAINING an offset, so a pointer naming the literal
// word of a two-word instruction would otherwise be adopted as a function entry and seeded with the
// ABI's entry VGPR_MSB state. Refusing is the only safe answer: there is no instruction boundary
// there to relocate a body from.
//
// This pins the required behaviour, not the mechanism. The refusal here survives disabling
// adopt()'s exact-start check, because a mid-instruction offset has no placement either way and
// relocate_relative_text_addends() then fails. Isolating that check needs a unit test on adopt();
// what this guards is that the object is never quietly accepted.
// The kernarg-provenance admission and its guards. A target this object never computed and never
// stored still has a provenance when it arrives through kernarg: the host wrote it, and the only
// `.text` address a host can hold is one it read from a symbol that relocate_text_symbols()
// rewrites. These fixtures carry no pointer table and no getpc, so object_produces_code_addresses
// is false and the kernarg set is the ONLY route to acceptance -- each case is non-vacuous by
// construction.
namespace {

constexpr uint16_t kKernargTargetSreg = 80;
constexpr uint16_t kKernargReturnSreg = 30;

[[nodiscard]] uint32_t kernarg_load_word(size_t index) {
  // sbase is encoded halved, so s[0:1] is 0. The analysis matches `s_load_b`, which is the gfx1250
  // spelling; CDNA's `s_load_dwordx2` would not be recognized.
  return cdna5::build_smem(cdna5::kSLoadB64Smem, {.sbase = 0, .sdata = kKernargTargetSreg})[index];
}

[[nodiscard]] rocjitsu::TranslatedCodeObject
translate_kernarg_fixture(const std::vector<uint32_t> &words, bool enable_kernarg) {
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  if (enable_kernarg)
    rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  EXPECT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  return translator.translate(source);
}

[[nodiscard]] bool refused_unrecovered_transfer(const rocjitsu::TranslatedCodeObject &result) {
  return !result.ok() && rocjitsu::test_support::has_error_containing(
                             result, rocjitsu::DiagnosticKind::Legalization,
                             "indirect branch or call target recovery is not implemented");
}

} // namespace

TEST(BinaryTranslatorE2E, Gfx1250AdmitsKernargLoadedIndirectCallTarget) {
  const std::vector<uint32_t> words = {
      kernarg_load_word(0),
      kernarg_load_word(1),
      rocjitsu::build_s_swappc_b64(kKernargReturnSreg, kKernargTargetSreg,
                                   ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
  };
  EXPECT_TRUE(translate_kernarg_fixture(words, /*enable_kernarg=*/true).ok());
}

// The admission in an object that DOES define a device function. The fixtures above are
// kernel-only, so object_defines_only_kernels() already holds and
// object_admits_kernarg_supplied_transfer is never consulted -- they cannot tell whether that path
// works. Here an exported body sits past the kernel and no edge reaches it, which is the shape
// rocPRIM's trampoline_kernel objects have: the admission turns on the strict symbol requirement,
// so the body has to be adopted or replace_text() refuses the object for a symbol naming nothing.
// Both halves are asserted, because adopting the body and keeping the partition a fixed point are
// separate obligations and only the second is visible on a re-translation.
TEST(BinaryTranslatorE2E, Gfx1250AdmitsKernargTargetInObjectDefiningAnExportedDeviceFunction) {
  constexpr size_t kFunctionOffsetWords = 4;
  constexpr size_t kFunctionWords = 2;
  const std::vector<uint32_t> words = {
      kernarg_load_word(0),
      kernarg_load_word(1),
      rocjitsu::build_s_swappc_b64(kKernargReturnSreg, kKernargTargetSreg,
                                   ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
      // Unreachable exported body: nothing branches or calls here, so only adoption can emit it.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      words, kFunctionWords, kFunctionOffsetWords, /*function_pointer_table_target_words=*/
      std::nullopt, /*name_function_pointer_table_with_symbol=*/true,
      /*export_text_function=*/true);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  // Non-vacuity, asserted rather than assumed: if this object still looked kernel-only the older
  // fence would admit the transfer and the case below would prove nothing about the new path.
  ASSERT_FALSE(rocjitsu::object_defines_only_kernels(source));
  ASSERT_FALSE(rocjitsu::discover_externally_resolvable_text_function_offsets(source).empty());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto first = translator.translate(source);
  ASSERT_TRUE(first.ok()) << (first.diagnostics.empty() ? "" : first.diagnostics.front().message);

  // The exported symbol must still name a body rather than have been undefined or dropped.
  rocjitsu::AmdGpuCodeObject translated(first.elf_bytes.data(), first.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto exported = rocjitsu::discover_externally_resolvable_text_function_offsets(translated);
  EXPECT_FALSE(exported.empty());

  // Adoption keyed off anything the first pass changes would move `.text` here.
  rocjitsu::BinaryTranslator second_translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto second = second_translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(first.elf_bytes, second.elf_bytes);
}

// The same words with no kernarg segment pointer in the descriptor. Nothing then supplies the
// pair, so the transfer has no provenance and must be refused -- this is what separates "the host
// wrote it" from "this object simply never computed it", which an undefined live-in also satisfies.
TEST(BinaryTranslatorE2E, Gfx1250RefusesIndirectCallWhenNoKernargSegmentExists) {
  const std::vector<uint32_t> words = {
      kernarg_load_word(0),
      kernarg_load_word(1),
      rocjitsu::build_s_swappc_b64(kKernargReturnSreg, kKernargTargetSreg,
                                   ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
  };
  EXPECT_TRUE(refused_unrecovered_transfer(translate_kernarg_fixture(words,
                                                                     /*enable_kernarg=*/false)));
}

// A write to s82 must not disturb s[80:81]: they share no register. Killing the pair below the one
// written is the bug this pins.
TEST(BinaryTranslatorE2E, Gfx1250KernargFactSurvivesWriteToTheRegisterAboveThePair) {
  const std::vector<uint32_t> words = {
      kernarg_load_word(0),
      kernarg_load_word(1),
      cdna5::build_sop1(cdna5::kSMovB32Sop1,
                        {.ssrc0 = 128, .sdst = static_cast<uint8_t>(kKernargTargetSreg + 2)})[0],
      rocjitsu::build_s_swappc_b64(kKernargReturnSreg, kKernargTargetSreg,
                                   ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
  };
  EXPECT_TRUE(translate_kernarg_fixture(words, /*enable_kernarg=*/true).ok());
}

// Clobbered on ONE of the two paths into the consumer. A meet that is not an intersection would
// carry the surviving path's fact through and admit a transfer whose target is unknown on the
// other path.
TEST(BinaryTranslatorE2E, Gfx1250RefusesKernargTargetClobberedOnOnePath) {
  const std::vector<uint32_t> words = {
      kernarg_load_word(0),
      kernarg_load_word(1),
      cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 1})[0],
      cdna5::build_sop1(cdna5::kSMovB64Sop1,
                        {.ssrc0 = 128, .sdst = static_cast<uint8_t>(kKernargTargetSreg)})[0],
      rocjitsu::build_s_swappc_b64(kKernargReturnSreg, kKernargTargetSreg,
                                   ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
  };
  EXPECT_TRUE(refused_unrecovered_transfer(translate_kernarg_fixture(words,
                                                                     /*enable_kernarg=*/true)));
}

// The control for the case above: same shape, nothing clobbered, so both paths agree and the
// transfer is admitted. Without it the refusal above could be passing for the wrong reason.
TEST(BinaryTranslatorE2E, Gfx1250AdmitsKernargTargetLiveOnBothPaths) {
  const std::vector<uint32_t> words = {
      kernarg_load_word(0),
      kernarg_load_word(1),
      cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 1})[0],
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_swappc_b64(kKernargReturnSreg, kKernargTargetSreg,
                                   ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
  };
  EXPECT_TRUE(translate_kernarg_fixture(words, /*enable_kernarg=*/true).ok());
}

// A latch edge back into the consumer. A must-analysis that starts non-entry blocks at the empty
// set instead of top can never grow the fact back across that meet, and would refuse this.
TEST(BinaryTranslatorE2E, Gfx1250AdmitsKernargTargetReachedThroughALoopLatch) {
  const std::vector<uint32_t> words = {
      kernarg_load_word(0),
      kernarg_load_word(1),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_swappc_b64(kKernargReturnSreg, kKernargTargetSreg,
                                   ROCJITSU_CODE_ARCH_CDNA5),
      cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = static_cast<uint16_t>(-3)})[0],
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
  };
  EXPECT_TRUE(translate_kernarg_fixture(words, /*enable_kernarg=*/true).ok());
}

// adopted_root_return_offsets() walks forward from each adopted root and stops at the next one.
// AMDGPU pads between functions rather than terminating them, so without that wall root A's walk
// falls through into root B and classifies a setpc under A's entry state. The wall depends on
// adopt()'s exact-start guarantee: "the next root starts here" has to be a fact, not a guess.
//
// Two plain adjacent bodies cannot show this -- a root's own walk always accepts its own setpc --
// so the setpc here is SHARED, reached from A by a branch and from B by fallthrough. The bare
// getpc at word 0 is load-bearing too: it is an unresolved PC producer the lattice never sees, so
// the whole-object permission is withheld and the misclassification is not masked by it.
TEST(BinaryTranslatorE2E, Gfx1250StopsAdoptedRootReturnWalkAtTheNextRoot) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint16_t kSharedReturnSreg = 30;
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_getpc_b64(10, ROCJITSU_CODE_ARCH_CDNA5), // word 0: kernel 0, poisons the
                                                                 // code-address accounting
      kGfx1250SEndpgm,                                           // word 1: kernel 1 entry
      cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 3})[0], // word 2: root A -> word 6
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),            // word 3: A falls into B
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),            // word 4: root B
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),            // word 5
      rocjitsu::build_s_setpc_b64(kSharedReturnSreg,
                                  ROCJITSU_CODE_ARCH_CDNA5), // word 6: reached from both
      kGfx1250SEndpgm,                                       // word 7
  };

  auto image =
      rocjitsu::test_support::make_minimal_amdgpu_elf_with_two_kernels_and_function_pointers(
          words, /*kernel1_entry_word=*/1,
          {{.offset_word = 2, .words = 1}, {.offset_word = 4, .words = 3}});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  // Both walks reach an out-of-body predecessor at the shared setpc and fail closed. Removing the
  // boundary lets A's walk pull B in, the setpc classifies as A's return, and the object is
  // accepted with a return proven from the wrong entry state.
  EXPECT_FALSE(result.ok())
      << "a setpc shared by two adopted roots must not be classified from one root's entry state";
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::Legalization,
      "indirect branch or call target recovery is not implemented"));
}

TEST(BinaryTranslatorE2E, Gfx1250RefusesFunctionPointerIntoTheMiddleOfAnInstruction) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint16_t kPcSreg = 4;
  constexpr uint16_t kGfx1250SAddNcU64Opcode = 83;
  constexpr uint16_t kLiteralOperand = 255;
  const std::vector<uint32_t> words = {
      kGfx1250SEndpgm, // word 0: kernel entry
      rocjitsu::build_sop2_encoding(ROCJITSU_CODE_ARCH_CDNA5, kGfx1250SAddNcU64Opcode, kPcSreg,
                                    kPcSreg, kLiteralOperand), // word 1: two-word instruction
      0x00000010u,                                             // word 2: its literal
      kGfx1250SEndpgm,                                         // word 3
  };

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      words, /*text_function_words=*/1, /*text_function_offset_words=*/2,
      /*function_pointer_table_target_words=*/2);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  EXPECT_FALSE(result.ok())
      << "a pointer into the middle of an instruction must not be adopted as a function entry";
}

TEST(BinaryTranslatorE2E, Gfx1250SizedFunctionSymbolEndingInsideMarkedWindowKeepsItsSize) {
  constexpr uint16_t kPcSreg = 4;
  constexpr uint16_t kGfx1250SAddNcU64Opcode = 83;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint32_t kTargetOffset = 140000;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  // The leading nop must be the long-transfer marker, not a plain nop: only a marked window is
  // preserved across a pass, and only a preserved window has interior offsets that emission
  // skips. A plain nop makes the consumer compact instead, which never exercises the mapping.
  std::vector<uint32_t> words = {
      rocjitsu::build_s_nop(rocjitsu::kLongDirectBranchMarkerNopImmediate,
                            ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_sop2_encoding(ROCJITSU_CODE_ARCH_CDNA5, kGfx1250SAddNcU64Opcode, kPcSreg,
                                    kPcSreg, kLiteralOperand),
      kTargetOffset - 2 * sizeof(uint32_t),
      rocjitsu::build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5),
  };
  words.resize(kTargetOffset / sizeof(uint32_t),
               rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5));
  words.push_back(kGfx1250SEndpgm);

  // Four words: the marker, the getpc, and its two-word delta add. The extent therefore ends
  // exactly on the setpc -- strictly inside the window, which spans marker through consumer.
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      words, /*text_function_words=*/4, /*text_function_offset_words=*/0);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto first = translator.translate(source);
  ASSERT_TRUE(first.ok()) << (first.diagnostics.empty() ? "" : first.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(first.elf_bytes.data(), first.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  rocjitsu::BinaryTranslator verifier(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto second = verifier.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);

  // The window is regenerated at its reserved size, so the symbol keeps its four words. Mapping
  // the extent to kernel_text.size() instead would report the whole window and round st_size up.
  const auto first_symbol =
      rocjitsu::test_support::find_elf_symbol_for_test(first.elf_bytes, "kernel");
  ASSERT_TRUE(first_symbol.has_value());
  EXPECT_EQ(first_symbol->st_size, 4 * sizeof(uint32_t));
  const auto second_symbol =
      rocjitsu::test_support::find_elf_symbol_for_test(second.elf_bytes, "kernel");
  ASSERT_TRUE(second_symbol.has_value());
  EXPECT_EQ(second_symbol->st_size, first_symbol->st_size);

  EXPECT_EQ(second.elf_bytes, first.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250CompilerLongJumpCompactsIdempotentlyAfterPaddingNop) {
  constexpr uint16_t kPcSreg = 4;
  constexpr uint16_t kGfx1250SAddNcU64Opcode = 83;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint32_t kTargetOffset = 140000;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  std::vector<uint32_t> words = {
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_sop2_encoding(ROCJITSU_CODE_ARCH_CDNA5, kGfx1250SAddNcU64Opcode, kPcSreg,
                                    kPcSreg, kLiteralOperand),
      kTargetOffset - 2 * sizeof(uint32_t),
      rocjitsu::build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5),
  };
  words.resize(kTargetOffset / sizeof(uint32_t),
               rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5));
  words.push_back(kGfx1250SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5));
  EXPECT_EQ((target_words[4] >> 16) & 0x7fu, cdna5::kSBranchSopp);

  rocjitsu::BinaryTranslator verifier(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto second = verifier.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250MarkedLongJumpStaysLongWhenTargetIsDirectlyReachable) {
  constexpr uint16_t kPcSreg = 4;
  constexpr uint16_t kGfx1250SAddNcU64Opcode = 83;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kTargetOffset = 5 * sizeof(uint32_t);
  const uint32_t marker = rocjitsu::build_s_nop(rocjitsu::kLongDirectBranchMarkerNopImmediate,
                                                ROCJITSU_CODE_ARCH_CDNA5);
  const std::vector<uint32_t> words = {
      marker,
      rocjitsu::build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_sop2_encoding(ROCJITSU_CODE_ARCH_CDNA5, kGfx1250SAddNcU64Opcode, kPcSreg,
                                    kPcSreg, kLiteralOperand),
      kTargetOffset - 2 * sizeof(uint32_t),
      rocjitsu::build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5),
      kGfx1250SEndpgm,
  };

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], marker);
  EXPECT_EQ(target_words[4], rocjitsu::build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5));

  rocjitsu::BinaryTranslator verifier(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto second = verifier.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250DoesNotTreatLiteralAsLongJumpMarker) {
  constexpr uint16_t kPcSreg = 4;
  constexpr uint16_t kUnrelatedSgpr = 20;
  constexpr uint16_t kGfx1250SAddNcU64Opcode = 83;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kTargetOffset = 7 * sizeof(uint32_t);
  const uint32_t marker = rocjitsu::build_s_nop(rocjitsu::kLongDirectBranchMarkerNopImmediate,
                                                ROCJITSU_CODE_ARCH_CDNA5);
  const std::vector<uint32_t> words = {
      rocjitsu::build_sop2_encoding(ROCJITSU_CODE_ARCH_CDNA5, 0, kUnrelatedSgpr, kUnrelatedSgpr,
                                    kLiteralOperand),
      marker,
      rocjitsu::build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_sop2_encoding(ROCJITSU_CODE_ARCH_CDNA5, kGfx1250SAddNcU64Opcode, kPcSreg,
                                    kPcSreg, kLiteralOperand),
      kTargetOffset - 3 * sizeof(uint32_t),
      rocjitsu::build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),
      kGfx1250SEndpgm,
  };

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "s_setpc_b64"; }),
            0);

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250DoesNotPreserveLongJumpMarkerAcrossBlockBoundary) {
  constexpr uint16_t kPcSreg = 4;
  constexpr uint16_t kGfx1250SAddNcU64Opcode = 83;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kTargetOffset = 7 * sizeof(uint32_t);
  const uint32_t marker = rocjitsu::build_s_nop(rocjitsu::kLongDirectBranchMarkerNopImmediate,
                                                ROCJITSU_CODE_ARCH_CDNA5);
  const std::vector<uint32_t> words = {
      cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 1})[0],
      marker,
      rocjitsu::build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_sop2_encoding(ROCJITSU_CODE_ARCH_CDNA5, kGfx1250SAddNcU64Opcode, kPcSreg,
                                    kPcSreg, kLiteralOperand),
      kTargetOffset - 3 * sizeof(uint32_t),
      rocjitsu::build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),
      kGfx1250SEndpgm,
  };

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "s_setpc_b64"; }),
            0);

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250DoesNotPreserveLongJumpMarkerAcrossInteriorBlockEntry) {
  constexpr uint16_t kPcSreg = 4;
  constexpr uint16_t kGfx1250SAddNcU64Opcode = 83;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kTargetOffset = 8 * sizeof(uint32_t);
  const uint32_t marker = rocjitsu::build_s_nop(rocjitsu::kLongDirectBranchMarkerNopImmediate,
                                                ROCJITSU_CODE_ARCH_CDNA5);
  const std::vector<uint32_t> words = {
      marker,
      rocjitsu::build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // The branch retains the getpc result while making the add an interior
      // block entry in the recovered window.
      rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_sop2_encoding(ROCJITSU_CODE_ARCH_CDNA5, kGfx1250SAddNcU64Opcode, kPcSreg,
                                    kPcSreg, kLiteralOperand),
      kTargetOffset - 2 * sizeof(uint32_t),
      rocjitsu::build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),
      kGfx1250SEndpgm,
  };

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::Legalization,
      "indirect branch or call target recovery is not implemented"));
}

TEST(BinaryTranslatorE2E, Gfx1250DoesNotPreserveLongJumpMarkerAtConsumerBlockEntry) {
  constexpr uint16_t kPcSreg = 4;
  constexpr uint16_t kGfx1250SAddNcU64Opcode = 83;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kTargetOffset = 5 * sizeof(uint32_t);
  const uint32_t marker = rocjitsu::build_s_nop(rocjitsu::kLongDirectBranchMarkerNopImmediate,
                                                ROCJITSU_CODE_ARCH_CDNA5);
  const std::vector<uint32_t> words = {
      marker,
      rocjitsu::build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_sop2_encoding(ROCJITSU_CODE_ARCH_CDNA5, kGfx1250SAddNcU64Opcode, kPcSreg,
                                    kPcSreg, kLiteralOperand),
      kTargetOffset - 2 * sizeof(uint32_t),
      rocjitsu::build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // The recovered target branches back to the consumer, making the
      // consumer itself a second block entry into the marked window.
      rocjitsu::build_s_branch(-2, ROCJITSU_CODE_ARCH_CDNA5),
      kGfx1250SEndpgm,
  };

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "s_setpc_b64"; }),
            0);

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, LongDirectBranchInvertsEveryGfx1250ConditionalPair) {
  struct TestCase {
    uint16_t source_opcode;
    uint16_t inverse_opcode;
  };
  constexpr std::array test_cases = {
      TestCase{cdna5::kSCbranchScc0Sopp, cdna5::kSCbranchScc1Sopp},
      TestCase{cdna5::kSCbranchScc1Sopp, cdna5::kSCbranchScc0Sopp},
      TestCase{cdna5::kSCbranchVcczSopp, cdna5::kSCbranchVccnzSopp},
      TestCase{cdna5::kSCbranchVccnzSopp, cdna5::kSCbranchVcczSopp},
      TestCase{cdna5::kSCbranchExeczSopp, cdna5::kSCbranchExecnzSopp},
      TestCase{cdna5::kSCbranchExecnzSopp, cdna5::kSCbranchExeczSopp},
  };
  constexpr uint64_t kTargetOffset = 140000;

  for (const TestCase &test_case : test_cases) {
    const uint32_t branch_word = rocjitsu::pack_sopp(test_case.source_opcode, 0);
    const auto branch = rocjitsu::test_support::decode_one(branch_word, ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(branch, nullptr);

    std::vector<uint8_t> text(kTargetOffset + sizeof(uint32_t), 0);
    rocjitsu::KernelTextLayout layout;
    layout.body_end = text.size();
    layout.long_branch_sgpr = 8;
    layout.blocks.push_back(
        {.source_start = 0,
         .source_end = sizeof(uint32_t),
         .target_start = 0,
         .target_end = rocjitsu::kMaxDirectBranchTransferWords * sizeof(uint32_t)});
    layout.blocks.push_back({.source_start = sizeof(uint32_t),
                             .source_end = 2 * sizeof(uint32_t),
                             .target_start = kTargetOffset,
                             .target_end = kTargetOffset + sizeof(uint32_t)});
    layout.branch_fixups.push_back(
        {.inst = branch.get(),
         .source_inst_offset = 0,
         .source_target_offset = sizeof(uint32_t),
         .target_inst_offset = 0,
         .target_window_bytes = rocjitsu::kMaxDirectBranchTransferWords * sizeof(uint32_t),
         .translated_words = {branch_word}});

    const auto result =
        rocjitsu::patch_direct_branch_fixups(text, layout, ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_TRUE(result.ok) << result.message;
    uint32_t inverted = 0;
    std::memcpy(&inverted, text.data(), sizeof(inverted));
    EXPECT_EQ((inverted >> 16) & 0x7fu, test_case.inverse_opcode);
  }
}

TEST(BinaryTranslatorE2E, RebaseTextOffsetShiftsOffsetsAtInsertionBoundaries) {
  constexpr uint64_t kWord = sizeof(uint32_t);
  const std::array insertions = {
      rocjitsu::TextLayoutInsertion{.offset = kWord, .size = kWord},
      rocjitsu::TextLayoutInsertion{.offset = 2 * kWord, .size = 2 * kWord},
  };

  uint64_t before = kWord - 1;
  uint64_t at_first = kWord;
  uint64_t between = 2 * kWord - 1;
  uint64_t at_second = 2 * kWord;
  rocjitsu::rebase_text_offset(before, insertions);
  rocjitsu::rebase_text_offset(at_first, insertions);
  rocjitsu::rebase_text_offset(between, insertions);
  rocjitsu::rebase_text_offset(at_second, insertions);

  EXPECT_EQ(before, kWord - 1);
  EXPECT_EQ(at_first, 2 * kWord);
  EXPECT_EQ(between, 3 * kWord - 1);
  EXPECT_EQ(at_second, 5 * kWord);
}

TEST(BinaryTranslatorE2E, GrowDirectBranchWindowsRebasesEveryLaterLayoutOffset) {
  const uint32_t branch_word = rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA5);
  const auto branch = rocjitsu::test_support::decode_one(branch_word, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(branch, nullptr);

  constexpr uint64_t kWord = sizeof(uint32_t);
  std::vector<uint8_t> text(8 * kWord, 0);
  rocjitsu::KernelTextLayout layout;
  layout.body_end = text.size();
  layout.blocks.push_back(
      {.source_start = 0, .source_end = kWord, .target_start = 0, .target_end = kWord});
  layout.blocks.push_back({.source_start = kWord,
                           .source_end = 8 * kWord,
                           .target_start = kWord,
                           .target_end = 8 * kWord});
  layout.branch_fixups.push_back({.inst = branch.get(),
                                  .source_inst_offset = 0,
                                  .source_target_offset = kWord,
                                  .target_inst_offset = 0,
                                  .target_window_bytes = kWord,
                                  .translated_words = {branch_word}});
  layout.branch_fixups.push_back({.inst = branch.get(),
                                  .source_inst_offset = 2 * kWord,
                                  .source_target_offset = 0,
                                  .target_inst_offset = 2 * kWord,
                                  .target_window_bytes = kWord,
                                  .translated_words = {branch_word}});
  layout.recovered_indirect_fixups.push_back({.source_call_offset = 3 * kWord,
                                              .target_window_offset = 3 * kWord,
                                              .target_window_bytes = 3 * kWord});
  layout.recovered_builder_fixups.push_back({.target_getpc_offset = 4 * kWord,
                                             .target_recovery_begin_offset = 5 * kWord,
                                             .target_recovery_end_offset = 6 * kWord});
  layout.branch_island_slots.push_back(7 * kWord);

  const std::array requirements = {
      rocjitsu::ControlFlowWindowRequirement{.source_inst_offset = 0,
                                             .required_window_bytes = 3 * kWord},
      rocjitsu::ControlFlowWindowRequirement{.kind =
                                                 rocjitsu::ControlFlowWindowKind::RecoveredIndirect,
                                             .source_inst_offset = 3 * kWord,
                                             .required_window_bytes = 5 * kWord},
  };
  const auto insertions =
      rocjitsu::grow_control_flow_windows(text, layout, requirements, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(insertions.has_value());
  ASSERT_EQ(insertions->size(), 2u);
  EXPECT_EQ(insertions->at(0).offset, kWord);
  EXPECT_EQ(insertions->at(0).size, 2 * kWord);
  EXPECT_EQ(insertions->at(1).offset, 6 * kWord);
  EXPECT_EQ(insertions->at(1).size, 2 * kWord);
  EXPECT_EQ(text.size(), 12 * kWord);
  EXPECT_EQ(layout.body_end, text.size());
  EXPECT_EQ(layout.blocks[0].target_end, 3 * kWord);
  EXPECT_EQ(layout.blocks[1].target_start, 3 * kWord);
  EXPECT_EQ(layout.blocks[1].target_end, 12 * kWord);
  EXPECT_EQ(layout.branch_fixups[0].target_inst_offset, 0u);
  EXPECT_EQ(layout.branch_fixups[0].target_window_bytes, 3 * kWord);
  EXPECT_EQ(layout.branch_fixups[1].target_inst_offset, 4 * kWord);
  EXPECT_EQ(layout.recovered_indirect_fixups[0].target_window_offset, 5 * kWord);
  EXPECT_EQ(layout.recovered_indirect_fixups[0].target_window_bytes, 5 * kWord);
  EXPECT_EQ(layout.recovered_builder_fixups[0].target_getpc_offset, 6 * kWord);
  EXPECT_EQ(layout.recovered_builder_fixups[0].target_recovery_begin_offset, 7 * kWord);
  EXPECT_EQ(layout.recovered_builder_fixups[0].target_recovery_end_offset, 10 * kWord);
  EXPECT_EQ(layout.branch_island_slots[0], 11 * kWord);

  const uint32_t expected_nop = rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5);
  uint32_t inserted_word = 0;
  std::memcpy(&inserted_word, text.data() + kWord, sizeof(inserted_word));
  EXPECT_EQ(inserted_word, expected_nop);
  std::memcpy(&inserted_word, text.data() + 2 * kWord, sizeof(inserted_word));
  EXPECT_EQ(inserted_word, expected_nop);
  std::memcpy(&inserted_word, text.data() + 8 * kWord, sizeof(inserted_word));
  EXPECT_EQ(inserted_word, expected_nop);
  std::memcpy(&inserted_word, text.data() + 9 * kWord, sizeof(inserted_word));
  EXPECT_EQ(inserted_word, expected_nop);
}

TEST(BinaryTranslatorE2E, FixedDirectBranchWindowRejectsGrowthWithoutMutation) {
  constexpr uint64_t kWord = sizeof(uint32_t);
  constexpr uint64_t kTargetOffset = 140000;
  const uint32_t branch_word = rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA5);
  const auto branch = rocjitsu::test_support::decode_one(branch_word, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(branch, nullptr);

  std::vector<uint8_t> text(kTargetOffset + kWord, 0);
  rocjitsu::KernelTextLayout layout;
  layout.body_end = text.size();
  layout.long_branch_sgpr = 8;
  layout.blocks.push_back(
      {.source_start = 0, .source_end = kWord, .target_start = 0, .target_end = kWord});
  layout.blocks.push_back({.source_start = kWord,
                           .source_end = 2 * kWord,
                           .target_start = kTargetOffset,
                           .target_end = kTargetOffset + kWord});
  layout.branch_fixups.push_back({.inst = branch.get(),
                                  .source_inst_offset = 0,
                                  .source_target_offset = kWord,
                                  .target_inst_offset = 0,
                                  .target_window_bytes = kWord,
                                  .allow_window_growth = false,
                                  .translated_words = {branch_word}});

  const auto original_text = text;
  const auto result = rocjitsu::patch_direct_branch_fixups(text, layout, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.reason, rocjitsu::TextLayoutFailureReason::BranchOutOfRange);
  EXPECT_TRUE(result.required_windows.empty());
  EXPECT_NE(result.message.find("fixed-size"), std::string::npos);
  EXPECT_EQ(text, original_text);
}

TEST(BinaryTranslatorE2E, DirectBranchWindowsRejectInvalidGrowthWithoutMutation) {
  const uint32_t branch_word = rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA5);
  const auto branch = rocjitsu::test_support::decode_one(branch_word, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(branch, nullptr);

  auto rejects_without_mutation =
      [&](std::vector<rocjitsu::ControlFlowWindowRequirement> requirements) {
        std::vector<uint8_t> text(sizeof(uint32_t), 0);
        rocjitsu::KernelTextLayout layout;
        layout.body_end = text.size();
        layout.branch_fixups.push_back({.inst = branch.get(),
                                        .source_inst_offset = 0,
                                        .source_target_offset = sizeof(uint32_t),
                                        .target_inst_offset = 0,
                                        .target_window_bytes = sizeof(uint32_t),
                                        .translated_words = {branch_word}});

        const auto original_text = text;
        const auto result = rocjitsu::grow_control_flow_windows(text, layout, requirements,
                                                                ROCJITSU_CODE_ARCH_CDNA5);
        EXPECT_FALSE(result.has_value());
        EXPECT_EQ(text, original_text);
        EXPECT_EQ(layout.body_end, sizeof(uint32_t));
        EXPECT_EQ(layout.branch_fixups.front().target_inst_offset, 0u);
        EXPECT_EQ(layout.branch_fixups.front().target_window_bytes, sizeof(uint32_t));
      };

  rejects_without_mutation(
      {{.source_inst_offset = 0, .required_window_bytes = sizeof(uint32_t) + 1}});
  rejects_without_mutation(
      {{.source_inst_offset = 0x1000, .required_window_bytes = 2 * sizeof(uint32_t)}});
  rejects_without_mutation({{.source_inst_offset = 0, .required_window_bytes = sizeof(uint32_t)}});
  rejects_without_mutation(
      {{.source_inst_offset = 0, .required_window_bytes = 2 * sizeof(uint32_t)},
       {.source_inst_offset = 0, .required_window_bytes = 3 * sizeof(uint32_t)}});

  {
    std::vector<uint8_t> text(sizeof(uint32_t), 0);
    rocjitsu::KernelTextLayout layout;
    layout.body_end = text.size();
    layout.recovered_indirect_fixups.push_back({.source_call_offset = 0,
                                                .target_window_offset = 0,
                                                .target_window_bytes = sizeof(uint32_t)});
    const std::array requirements = {rocjitsu::ControlFlowWindowRequirement{
        .kind = rocjitsu::ControlFlowWindowKind::RecoveredIndirect,
        .source_inst_offset = 0x1000,
        .required_window_bytes = 2 * sizeof(uint32_t),
    }};

    const auto original_text = text;
    const auto result =
        rocjitsu::grow_control_flow_windows(text, layout, requirements, ROCJITSU_CODE_ARCH_CDNA5);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(text, original_text);
    EXPECT_EQ(layout.body_end, sizeof(uint32_t));
    EXPECT_EQ(layout.recovered_indirect_fixups.front().target_window_bytes, sizeof(uint32_t));
  }
}

TEST(BinaryTranslatorE2E, PatchesOutOfRangeConditionalThroughIslandSlotWithoutSgpr) {
  constexpr uint64_t kWord = sizeof(uint32_t);
  constexpr uint64_t kTargetOffset = 140000;
  constexpr uint64_t kIslandOffset = 70000;
  const uint32_t branch_word =
      rocjitsu::pack_sopp(cdna5::kSCbranchScc1Sopp, static_cast<uint16_t>(0));
  const auto branch = rocjitsu::test_support::decode_one(branch_word, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(branch, nullptr);

  std::vector<uint8_t> text(kTargetOffset + kWord, 0);
  rocjitsu::KernelTextLayout layout;
  layout.body_end = text.size();
  layout.blocks.push_back(
      {.source_start = 0, .source_end = kWord, .target_start = 0, .target_end = kWord});
  layout.blocks.push_back({.source_start = kWord,
                           .source_end = 2 * kWord,
                           .target_start = kTargetOffset,
                           .target_end = kTargetOffset + kWord});
  layout.branch_fixups.push_back({.inst = branch.get(),
                                  .source_inst_offset = 0,
                                  .source_target_offset = kWord,
                                  .target_inst_offset = 0,
                                  .target_window_bytes = kWord,
                                  .translated_words = {branch_word}});
  layout.branch_island_slots.push_back(kIslandOffset);

  const auto before_growth =
      rocjitsu::patch_direct_branch_fixups(text, layout, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_FALSE(before_growth.ok);
  ASSERT_EQ(before_growth.required_windows.size(), 1u);
  EXPECT_EQ(before_growth.required_windows.front().required_window_bytes, 2 * kWord);

  const auto insertions = rocjitsu::grow_control_flow_windows(
      text, layout, before_growth.required_windows, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(insertions.has_value());
  ASSERT_TRUE(rocjitsu::patch_direct_branch_fixups(text, layout, ROCJITSU_CODE_ARCH_CDNA5).ok);

  const auto *target_words = reinterpret_cast<const uint32_t *>(text.data());
  EXPECT_EQ(target_words[0], rocjitsu::pack_sopp(cdna5::kSCbranchScc0Sopp, 1));
  const uint64_t rebased_island_offset = kIslandOffset + kWord;
  const auto source_to_island = rocjitsu::compute_sopp_branch_simm16(kWord, rebased_island_offset);
  ASSERT_TRUE(source_to_island.has_value());
  EXPECT_EQ(target_words[1], rocjitsu::build_s_branch(*source_to_island, ROCJITSU_CODE_ARCH_CDNA5));

  uint32_t island_word = 0;
  std::memcpy(&island_word, text.data() + rebased_island_offset, sizeof(island_word));
  const auto island_to_target =
      rocjitsu::compute_sopp_branch_simm16(rebased_island_offset, kTargetOffset + kWord);
  ASSERT_TRUE(island_to_target.has_value());
  EXPECT_EQ(island_word, rocjitsu::build_s_branch(*island_to_target, ROCJITSU_CODE_ARCH_CDNA5));
}

TEST(BinaryTranslatorE2E, FullSgprConditionalPreservesPoolAfterUnconditionalBranch) {
  using namespace rocr::llvm::amdhsa;

  constexpr size_t kExpansionCount = 16000;
  const auto cvt = make_cdna4_cvt_f32_bf16_words(cdna4::encoding::kVop1);
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;

  std::vector<uint32_t> words;
  words.reserve(2 * kExpansionCount + 2);
  for (size_t i = 0; i < kExpansionCount; ++i) {
    words.push_back(cvt[0]);
    // The first pass inserts each pool after one of these source blocks. Once
    // relocated, the direct branch skips that pool's header while the
    // out-of-range branch below still reaches one of its private slots.
    words.push_back(rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4));
  }
  const int16_t back_to_entry = static_cast<int16_t>(-static_cast<int64_t>(words.size() + 1));
  words.push_back(cdna4::build_sopp(cdna4::kSCbranchScc1Sopp,
                                    {.simm16 = static_cast<uint16_t>(back_to_entry)})[0]);
  words.push_back(kCdna4SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));
  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  13);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t translated_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const uint32_t marker = rocjitsu::build_s_nop(rocjitsu::kBranchIslandPoolMarkerNopImmediate,
                                                ROCJITSU_CODE_ARCH_CDNA3);
  const uint32_t skip_pool = rocjitsu::build_s_branch(16, ROCJITSU_CODE_ARCH_CDNA3);
  const uint32_t skip_over_pool =
      rocjitsu::build_s_branch(static_cast<int16_t>(rocjitsu::kGeneratedIslandPoolHeaderWords +
                                                    rocjitsu::kDirectBranchIslandPoolSlots),
                               ROCJITSU_CODE_ARCH_CDNA3);
  const uint32_t unused_slot = rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3);
  bool found_live_skipped_pool = false;
  for (size_t i = 1;
       i + rocjitsu::kGeneratedIslandPoolHeaderWords + rocjitsu::kDirectBranchIslandPoolSlots <=
       translated_word_count;
       ++i) {
    if (target_words[i] != marker || target_words[i - 1] != skip_over_pool ||
        target_words[i + 1] != skip_pool) {
      continue;
    }
    const size_t first_slot = i + rocjitsu::kGeneratedIslandPoolHeaderWords;
    const size_t past_slots = first_slot + rocjitsu::kDirectBranchIslandPoolSlots;
    if (std::any_of(target_words + first_slot, target_words + past_slots,
                    [&](uint32_t word) { return word != unused_slot; })) {
      found_live_skipped_pool = true;
      break;
    }
  }
  EXPECT_TRUE(found_live_skipped_pool);
  bool found_island_source = false;
  const uint32_t inverted = cdna3::build_sopp(cdna3::kSCbranchScc0Sopp, {.simm16 = 1})[0];
  for (size_t i = 0; i + 1 < translated_word_count; ++i) {
    if (target_words[i] == inverted && ((target_words[i + 1] >> 23) & 0x1ffu) == 0x17fu) {
      found_island_source = true;
      break;
    }
  }
  EXPECT_TRUE(found_island_source);
  expect_cdna3_translated_descriptor_sgprs_eq(result.elf_bytes, 112);

  rocjitsu::BinaryTranslator verifier(ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA3);
  const auto second = verifier.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250RelocatesReachableGeneratedIslandPoolSlotsIdempotently) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  std::vector<uint32_t> words = {
      // The conditional reaches the first private slot while fallthrough
      // reaches the pool header and skip.
      cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 2})[0],
      rocjitsu::build_s_nop(rocjitsu::kBranchIslandPoolMarkerNopImmediate,
                            ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_branch(static_cast<int16_t>(rocjitsu::kDirectBranchIslandPoolSlots),
                               ROCJITSU_CODE_ARCH_CDNA5),
  };
  words.insert(words.end(), rocjitsu::kDirectBranchIslandPoolSlots,
               rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA5));
  words.push_back(kGfx1250SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto first = translator.translate(source);
  ASSERT_TRUE(first.ok()) << (first.diagnostics.empty() ? "" : first.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(first.elf_bytes.data(), first.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, first.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250RejectsGeneratedIslandPoolSlotTargetingOutsideText) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  std::vector<uint32_t> words = {
      // Reach the first private slot so it is live in this kernel scope.
      cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 2})[0],
      rocjitsu::build_s_nop(rocjitsu::kBranchIslandPoolMarkerNopImmediate,
                            ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_branch(static_cast<int16_t>(rocjitsu::kDirectBranchIslandPoolSlots),
                               ROCJITSU_CODE_ARCH_CDNA5),
  };
  words.insert(words.end(), rocjitsu::kDirectBranchIslandPoolSlots,
               rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA5));
  words[rocjitsu::kGeneratedIslandPoolHeaderWords + 1] =
      rocjitsu::build_s_branch(30000, ROCJITSU_CODE_ARCH_CDNA5);
  words.push_back(kGfx1250SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::Legalization,
      "generated direct branch island pool targets outside source .text"));
}

TEST(BinaryTranslatorE2E, Gfx1250InvalidPoolCandidateFallsBackToNormalDecodeDiagnostic) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  std::vector<uint32_t> words = {
      rocjitsu::build_s_nop(rocjitsu::kBranchIslandPoolMarkerNopImmediate,
                            ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_branch(static_cast<int16_t>(rocjitsu::kDirectBranchIslandPoolSlots),
                               ROCJITSU_CODE_ARCH_CDNA5),
  };
  words.insert(words.end(), rocjitsu::kDirectBranchIslandPoolSlots,
               rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA5));
  words[rocjitsu::kGeneratedIslandPoolHeaderWords] =
      cdna5::build_vop1(cdna5::kVReadfirstlaneB32Vop1, {.src0 = 255})[0];
  words.push_back(kGfx1250SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::Legalization, "does not support 32-bit literals"));
}

TEST(BinaryTranslatorE2E, Gfx1250FourWordMalformedPoolCandidateFailsSafely) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  std::vector<uint32_t> words = {
      rocjitsu::build_s_nop(rocjitsu::kBranchIslandPoolMarkerNopImmediate,
                            ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_branch(static_cast<int16_t>(rocjitsu::kDirectBranchIslandPoolSlots),
                               ROCJITSU_CODE_ARCH_CDNA5),
  };
  words.insert(words.end(), rocjitsu::kDirectBranchIslandPoolSlots,
               rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA5));
  // Force speculative pool recognition down the four-word VOP3PX2 decode path.
  words[rocjitsu::kGeneratedIslandPoolHeaderWords] = 0xCC350000u;
  words.push_back(kGfx1250SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
}

TEST(BinaryTranslatorE2E, Gfx1250RejectsNearMissGeneratedIslandPool) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  std::vector<uint32_t> words = {
      rocjitsu::build_s_nop(rocjitsu::kBranchIslandPoolMarkerNopImmediate,
                            ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_branch(static_cast<int16_t>(rocjitsu::kDirectBranchIslandPoolSlots),
                               ROCJITSU_CODE_ARCH_CDNA5),
  };
  words.insert(words.end(), rocjitsu::kDirectBranchIslandPoolSlots,
               rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA5));
  words[2 + rocjitsu::kDirectBranchIslandPoolSlots / 2] =
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5);
  words.push_back(kGfx1250SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto first = translator.translate(source);
  ASSERT_TRUE(first.ok()) << (first.diagnostics.empty() ? "" : first.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(first.elf_bytes.data(), first.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(decoded,
                                  [](const auto &inst) { return inst->mnemonic() == "s_branch"; }),
            1);

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, first.elf_bytes);
}

TEST(BinaryTranslatorE2E, DuplicatesSharedReachableBlocksPerKernel) {
  constexpr uint32_t kCdna4SEndpgm = cdna4::build_sopp(cdna4::kSEndpgmSopp)[0];
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4), // kernel0: 0x00 -> helper 0x08.
      rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4), // kernel1: 0x04 -> helper 0x08.
      kCdna4SEndpgm,                                         // Shared source helper.
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_two_kernel_descriptors(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *text = translated.text_sections()[0];

  rocjitsu::KernelDescriptorTranslator parser(ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA3);
  const auto infos = parser.translate_image(result.elf_bytes, text->sectionOffset(), text->size(),
                                            rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(infos.size(), 2u);
  std::vector<uint64_t> translated_entries;
  for (const auto &info : infos)
    translated_entries.push_back(info.entry_text_offset);
  std::ranges::sort(translated_entries);
  ASSERT_EQ(translated_entries[0], 0u);
  ASSERT_GT(translated_entries[1], 2 * sizeof(uint32_t));
  ASSERT_LE(translated_entries[1] + 2 * sizeof(uint32_t), text->size());

  const auto *target_words = reinterpret_cast<const uint32_t *>(text->data());
  const uint64_t second_entry_word = translated_entries[1] / sizeof(uint32_t);
  EXPECT_EQ(target_words[0], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], kCdna4SEndpgm);
  EXPECT_EQ(target_words[second_entry_word], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[second_entry_word + 1], kCdna4SEndpgm);
}

TEST(BinaryTranslatorE2E, RejectsAmbiguousRuntimeEntryClonedAcrossKernelScopes) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA5), // kernel0 -> helper 0x08.
      rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA5), // kernel1 -> helper 0x08.
      kGfx1250SEndpgm,
  };
  for (const auto relocation : {rocjitsu::test_support::TestRuntimeTextRelocation::Abs64,
                                rocjitsu::test_support::TestRuntimeTextRelocation::Relative64}) {
    SCOPED_TRACE(relocation == rocjitsu::test_support::TestRuntimeTextRelocation::Abs64
                     ? "ABS64"
                     : "RELATIVE64");
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_two_kernel_descriptors(
        words, rocjitsu::test_support::TestRuntimeTextReference{
                   .relocation = relocation,
                   .target_text_offset = 2 * sizeof(uint32_t),
               });
    rocjitsu::test_support::write_value_for_test<uint32_t>(
        image, offsetof(rocjitsu::Elf64_Ehdr, e_flags), rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250);
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());

    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
        result, rocjitsu::DiagnosticKind::ResourceLimit,
        "relocated .text could not be materialized safely"));
  }
}

TEST(BinaryTranslatorE2E, PreservesUniqueRelocationBackedTextEntries) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint64_t kReferencedEntry = sizeof(uint32_t);
  struct RelocationCase {
    rocjitsu::test_support::TestRuntimeTextRelocation fixture;
    uint32_t relocation_type;
  };
  constexpr std::array relocation_cases = {
      RelocationCase{rocjitsu::test_support::TestRuntimeTextRelocation::Abs64,
                     rocjitsu::R_AMDGPU_ABS32_LO},
      RelocationCase{rocjitsu::test_support::TestRuntimeTextRelocation::Abs64,
                     rocjitsu::R_AMDGPU_ABS32_HI},
      RelocationCase{rocjitsu::test_support::TestRuntimeTextRelocation::Abs64,
                     rocjitsu::R_AMDGPU_ABS64},
      RelocationCase{rocjitsu::test_support::TestRuntimeTextRelocation::Abs64,
                     rocjitsu::R_AMDGPU_ABS32},
      RelocationCase{rocjitsu::test_support::TestRuntimeTextRelocation::Relative64,
                     rocjitsu::R_AMDGPU_RELATIVE64},
  };

  for (const RelocationCase &test_case : relocation_cases) {
    SCOPED_TRACE(test_case.relocation_type);
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_two_kernel_descriptors(
        {kGfx1250SEndpgm, kGfx1250SEndpgm}, rocjitsu::test_support::TestRuntimeTextReference{
                                                .relocation = test_case.fixture,
                                                .relocation_type = test_case.relocation_type,
                                                .target_text_offset = kReferencedEntry,
                                            });
    rocjitsu::test_support::write_value_for_test<uint32_t>(
        image, offsetof(rocjitsu::Elf64_Ehdr, e_flags), rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250);
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());

    auto options = gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                            rocjitsu::ProcessorRevision::Gfx1250A0);
    options.verify_rewrite_discharge = true;
    rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
                                          options);
    const auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);
    EXPECT_TRUE(result.rewrite_discharge_checked);
    EXPECT_TRUE(result.rewrite_discharge_verified);

    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(translated.is_valid());
    ASSERT_EQ(translated.text_sections().size(), 1u);
    const auto *text = translated.text_sections().front();
    rocjitsu::KernelDescriptorTranslator parser(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5);
    const auto infos = parser.translate_image(result.elf_bytes, text->sectionOffset(), text->size(),
                                              rocjitsu::KernelDescriptorTranslationOptions{});
    ASSERT_EQ(infos.size(), 2u);
    const auto referenced_descriptor =
        std::ranges::find_if(infos, [](const auto &info) { return info.entry_text_offset != 0; });
    ASSERT_NE(referenced_descriptor, infos.end());
    const uint64_t expected_vaddr = text->vaddr() + referenced_descriptor->entry_text_offset;

    const auto output_ehdr =
        rocjitsu::test_support::read_elf_struct_for_test<rocjitsu::Elf64_Ehdr>(result.elf_bytes, 0);
    const auto output_shdrs = rocjitsu::test_support::read_elf_array_for_test<rocjitsu::Elf64_Shdr>(
        result.elf_bytes, output_ehdr.e_shoff, output_ehdr.e_shnum);
    const auto rela = std::ranges::find_if(output_shdrs, [](const rocjitsu::Elf64_Shdr &section) {
      return section.sh_type == rocjitsu::SHT_RELA;
    });
    ASSERT_NE(rela, output_shdrs.end());
    const auto relocation_record =
        rocjitsu::test_support::read_elf_struct_for_test<rocjitsu::Elf64_Rela>(result.elf_bytes,
                                                                               rela->sh_offset);
    EXPECT_EQ(rocjitsu::elf_reloc_type(relocation_record.r_info), test_case.relocation_type);
    if (test_case.fixture == rocjitsu::test_support::TestRuntimeTextRelocation::Relative64) {
      EXPECT_EQ(relocation_record.r_addend, static_cast<int64_t>(expected_vaddr));
      continue;
    }

    ASSERT_LT(rela->sh_link, output_shdrs.size());
    const auto &symtab = output_shdrs[rela->sh_link];
    const auto symbols = rocjitsu::test_support::read_elf_array_for_test<rocjitsu::Elf64_Sym>(
        result.elf_bytes, symtab.sh_offset, symtab.sh_size / sizeof(rocjitsu::Elf64_Sym));
    const uint32_t symbol_index = rocjitsu::elf_reloc_sym(relocation_record.r_info);
    ASSERT_LT(symbol_index, symbols.size());
    EXPECT_EQ(symbols[symbol_index].st_value, expected_vaddr);
  }
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3SemanticExpandRulesHaveTranslationFixtures) {
  const auto test_cases = cdna4_to_cdna3_semantic_rule_cases();
  const auto rules = rocjitsu::semantic_expand_rules_cdna4_to_cdna3();

  for (const auto &rule : rules) {
    EXPECT_TRUE(has_cdna4_to_cdna3_semantic_rule_case(rule.src_encoding_id, rule.src_opcode))
        << "missing fixture for CDNA4->CDNA3 semantic rule encoding=0x" << std::hex
        << rule.src_encoding_id << " opcode=" << rule.src_opcode << std::dec;
  }
  for (const auto &test_case : test_cases) {
    EXPECT_TRUE(has_cdna4_to_cdna3_semantic_rule(test_case.encoding_id, test_case.opcode))
        << "test fixture has no CDNA4->CDNA3 semantic rule: " << test_case.name;
  }
}

class Cdna4ToCdna3SemanticRuleTranslationTest
    : public ::testing::TestWithParam<Cdna4ToCdna3SemanticRuleCase> {};

TEST_P(Cdna4ToCdna3SemanticRuleTranslationTest, TranslatesSingleInstruction) {
  const auto &test_case = GetParam();
  SCOPED_TRACE(test_case.name);

  ASSERT_LE(test_case.word_count, test_case.words.size());
  const std::vector<uint32_t> source_words(test_case.words.begin(),
                                           test_case.words.begin() + test_case.word_count);
  auto image =
      rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(source_words);
  rocjitsu::AmdGpuCodeObject source_layout(image.data(), image.size());
  ASSERT_TRUE(source_layout.is_valid());
  ASSERT_FALSE(source_layout.text_sections().empty());

  const auto *source_text = source_layout.text_sections()[0];
  ASSERT_EQ(source_text->size(), source_words.size() * sizeof(uint32_t));

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  EXPECT_EQ(rocjitsu::test_support::find_section(translated, ".rj_translations"), nullptr);
  const bool has_sidecar = rocjitsu::test_support::find_section(
                               translated, rocjitsu::kVirtualLdsMetadataSectionName) != nullptr;
  expect_cdna3_text_matches(*translated.text_sections()[0], test_case.expected, has_sidecar);
}

INSTANTIATE_TEST_SUITE_P(ImplementedRules, Cdna4ToCdna3SemanticRuleTranslationTest,
                         ::testing::ValuesIn(cdna4_to_cdna3_semantic_rule_cases()),
                         [](const ::testing::TestParamInfo<Cdna4ToCdna3SemanticRuleCase> &info) {
                           return std::string(info.param.name);
                         });

TEST(BinaryTranslatorE2E, LshlAddU64ForRdnaEmitsShiftAndCarryAdd) {
  // v[4:5] = (v[0:1] << 2) + v[2:3]. vdst does not alias the addend, so the shift
  // goes straight into vdst. Verify a v_lshlrev_b64 (the shift the old lowering
  // dropped) precedes the carry-add pair.
  const auto words = make_cdna4_v_lshl_add_u64_words(/*vdst=*/4, /*src0=*/256 + 0,
                                                     /*src1=*/2, /*src2=*/256 + 2);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {words[0], words[1], 0xBF810000u});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_RDNA4);
  const bool has_shift = std::any_of(decoded.begin(), decoded.end(), [](const auto &inst) {
    return inst->mnemonic() == "v_lshlrev_b64";
  });
  EXPECT_TRUE(has_shift) << "lowering must materialize the shift, not drop it";
  const bool has_add = std::any_of(decoded.begin(), decoded.end(), [](const auto &inst) {
    return inst->mnemonic() == "v_add_co_u32";
  });
  EXPECT_TRUE(has_add) << "lowering must emit the 64-bit carry add";
}

TEST(BinaryTranslatorE2E, LshlAddU64ForRdnaHandlesDestinationAliasingAddend) {
  // v[2:3] = (v[0:1] << 2) + v[2:3]: the destination aliases the VGPR addend, the
  // common address-computation pattern emitted by real kernels (e.g. vector_add).
  // The lowering must shift into a dead scratch pair rather than clobbering the
  // addend, and must still succeed (the earlier fail-closed guard wrongly rejected
  // this).
  const auto words = make_cdna4_v_lshl_add_u64_words(/*vdst=*/2, /*src0=*/256 + 0,
                                                     /*src1=*/2, /*src2=*/256 + 2);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {words[0], words[1], 0xBF810000u});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_RDNA4);

  const auto shift = std::find_if(decoded.begin(), decoded.end(), [](const auto &inst) {
    return inst->mnemonic() == "v_lshlrev_b64";
  });
  ASSERT_NE(shift, decoded.end()) << "lowering must materialize the shift";
  // The shift destination must not be v2/v3 (the aliased addend); it must land in
  // a separate scratch pair so the following add can still read the addend.
  ASSERT_GE((*shift)->num_dst_operands(), 1);
  const auto shift_dst = (*shift)->dst_operand(0)->to_register_ref();
  ASSERT_TRUE(shift_dst.has_value());
  EXPECT_NE(shift_dst->index, 2u)
      << "shift result must not overwrite the aliased addend v[2:3] before the add reads it";
}

TEST(BinaryTranslatorE2E, LshlAddU64ForRdnaRejectsInlineConstantAddend) {
  // v[4:5] = (v[0:1] << 2) + <inline constant 0>. The 64-bit addend is encoded as
  // a single inline-constant operand (128), so the high half cannot be derived as
  // src2 + 1 (129 == constant 1). The lowering must fail closed rather than
  // silently add 1 to the high word.
  constexpr uint16_t kInlineConstZero = 128;
  const auto words = make_cdna4_v_lshl_add_u64_words(/*vdst=*/4, /*src0=*/256 + 0,
                                                     /*src1=*/2, /*src2=*/kInlineConstZero);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {words[0], words[1], 0xBF810000u});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(source);

  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed, "non-register 64-bit addend"));
}

TEST(BinaryTranslatorE2E, LshlAddU64ForRdnaRejectsV255Addend) {
  // v[4:5] = (v[0:1] << 2) + v[255:256]. src2=v255 (selector 511) has no valid
  // high half: the derived src2+1 selector is 512, which does not encode a VGPR.
  // The lowering must fail closed rather than emit an add reading an invalid
  // operand.
  constexpr uint16_t kV255 = 256 + 255;
  const auto words = make_cdna4_v_lshl_add_u64_words(/*vdst=*/4, /*src0=*/256 + 0,
                                                     /*src1=*/2, /*src2=*/kV255);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {words[0], words[1], 0xBF810000u});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(source);

  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed, "non-register 64-bit addend"));
}

TEST(BinaryTranslatorE2E, LshlAddU64ForRdnaCarryUsesScalarSgprNotVcc) {
  // v_lshl_add_u64 defines no carry output and VCC is not liveness-tracked, so
  // the carry chain must target a dead ordinary SGPR pair, never VCC (which could
  // be live across the instruction). Verify the emitted v_add_co_u32 /
  // v_add_co_ci_u32 write a scalar SDST that is not VCC.
  const auto words = make_cdna4_v_lshl_add_u64_words(/*vdst=*/4, /*src0=*/256 + 0,
                                                     /*src1=*/2, /*src2=*/256 + 2);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {words[0], words[1], 0xBF810000u});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_RDNA4);

  // VCC_LO is scalar selector 106. The carry SDST must be an ordinary SGPR.
  constexpr uint32_t kVccLo = 106;
  bool saw_add = false;
  for (const auto &inst : decoded) {
    const auto mnemonic = inst->mnemonic();
    if (mnemonic != "v_add_co_u32" && mnemonic != "v_add_co_ci_u32")
      continue;
    saw_add = true;
    ASSERT_GE(inst->num_dst_operands(), 2)
        << mnemonic << " must expose an explicit scalar carry destination";
    const auto sdst = inst->dst_operand(1)->to_register_ref();
    ASSERT_TRUE(sdst.has_value()) << mnemonic << " carry SDST must be a register";
    EXPECT_NE(sdst->index, kVccLo) << mnemonic << " must not clobber VCC as its carry destination";
  }
  EXPECT_TRUE(saw_add) << "lowering must emit the carry-add pair";
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsReadB32ToFlatGlobalLoad) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto saddr =
      find_flat_global_load_saddr(target_words, target_word_count, /*op=*/20, /*vdst=*/7,
                                  /*addr=*/4, /*offset=*/0x134);
  ASSERT_TRUE(saddr.has_value());
  EXPECT_TRUE(contains_smem_load_dwordx2_with_wait(target_words, target_word_count, *saddr,
                                                   /*sbase_sgpr=*/0, /*offset=*/24));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersMubufDwordLdsToFlatGlobalStore) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto mubuf = make_cdna4_buffer_load_lds_words(/*op=*/20);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {mubuf[0], mubuf[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  // A virtual-LDS sidecar cannot leave the MUBUF LDS bit set: the side effect
  // would target zero-sized hardware LDS instead of rocjitsu's backing buffer.
  EXPECT_FALSE(contains_mubuf_lds_op(target_words, target_word_count, /*op=*/20));
  EXPECT_TRUE(contains_flat_global_store_op(target_words, target_word_count, /*op=*/28));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarReservesFreshSgprsBelowCdnaSpecialTail) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_mov_b32(/*sdst=*/47, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4),
      ds[0],
      ds[1],
      kCdna4SEndpgm,
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  // A HipKittens-style descriptor can allocate more SGPRs than the ordinary
  // guest body names because VCC/flat-scratch/XNACK and granularity padding are
  // included in COMPUTE_PGM_RSRC1. Virtual LDS must not treat that tail as
  // ordinary scratch space.
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  6);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *metadata_section =
      rocjitsu::test_support::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  EXPECT_EQ(parsed->front().virtual_lds_base_sgpr, 48u);
  expect_cdna3_sidecar_descriptor_sgprs_eq(result.elf_bytes, 64);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarKeepsDescriptorSpecialTailFree) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  13);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  expect_cdna3_translated_descriptor_sgprs_eq(result.elf_bytes, 112);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *metadata_section =
      rocjitsu::test_support::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  EXPECT_EQ(parsed->front().virtual_lds_base_sgpr, 2u);

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  // A large SGPR allocation does not imply all of those numbers are ordinary
  // scratchable registers. VCC/flat-scratch/XNACK live in the descriptor tail,
  // so a guest body that only names low SGPRs should place virtual-LDS scratch
  // below that tail instead of borrowing a high pair with spill-per-use.
  EXPECT_FALSE(contains_sopp(target_words, target_word_count, /*op=*/8));
  EXPECT_TRUE(contains_smem_load_dwordx2_with_wait(target_words, target_word_count, /*sdata=*/2,
                                                   /*sbase_sgpr=*/0, /*offset=*/24));
  rocjitsu::cdna3::FlatMachineInst actual{};
  bool found_load = false;
  for (size_t i = 0; i + 1 < target_word_count; ++i) {
    std::memcpy(&actual, target_words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == 20u && actual.seg == 2u && actual.addr == 4u &&
        actual.vdst == 7u && actual.sc0 == 1u && actual.sc1 == 0u && actual.saddr == 2u) {
      found_load = true;
      break;
    }
  }
  EXPECT_TRUE(found_load);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarSpillsTouchedHighSgprPairWhenDescriptorSgprsAreFull) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  std::vector<uint32_t> words;
  words.reserve(102 + 3);
  for (uint16_t sgpr = 0; sgpr < 102; ++sgpr)
    words.push_back(rocjitsu::build_s_mov_b32(sgpr, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4));
  words.push_back(ds[0]);
  words.push_back(ds[1]);
  words.push_back(kCdna4SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  13);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  expect_cdna3_translated_descriptor_sgprs_eq(result.elf_bytes, 112);

  const auto entry = virtual_lds_sidecar_entry_words_for_test(result.elf_bytes);
  ASSERT_TRUE(entry.has_value());
  const auto *target_words = entry->words;
  const size_t target_word_count = entry->word_count;
  EXPECT_TRUE(contains_sopp(target_words, target_word_count, /*op=*/8));
  const auto saved_lo =
      find_cdna3_vop1_vdst(target_words, target_word_count, /*op=*/1, /*src0=*/100);
  const auto saved_hi =
      find_cdna3_vop1_vdst(target_words, target_word_count, /*op=*/1, /*src0=*/101);
  ASSERT_TRUE(saved_lo.has_value());
  ASSERT_TRUE(saved_hi.has_value());
  EXPECT_NE(*saved_lo, *saved_hi);
  EXPECT_TRUE(contains_smem_load_dwordx2_with_wait(target_words, target_word_count, /*sdata=*/100,
                                                   /*sbase_sgpr=*/0, /*offset=*/24));
  // Spill-per-use lowering first installs the runtime backing pointer in the
  // borrowed SGPR pair, then restores the guest scalar values after the lowered
  // memory access. The two operations can use different VGPR temps because the
  // entry prologue saves the backing pointer in persistent scratch.
  EXPECT_GE(count_cdna3_vop1_writes(target_words, target_word_count, /*op=*/2, /*vdst=*/100), 2u);
  EXPECT_GE(count_cdna3_vop1_writes(target_words, target_word_count, /*op=*/2, /*vdst=*/101), 2u);
}

TEST(BinaryTranslatorE2E, VirtualLdsMubufSemanticRuleUsesSpillPerUseAccessEmitter) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto mubuf = make_cdna4_buffer_load_lds_words(/*op=*/20);
  std::vector<uint32_t> words;
  words.reserve(102 + 3);
  for (uint16_t sgpr = 0; sgpr < 102; ++sgpr)
    words.push_back(rocjitsu::build_s_mov_b32(sgpr, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4));
  words.push_back(mubuf[0]);
  words.push_back(mubuf[1]);
  words.push_back(kCdna4SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  13);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  const auto entry = virtual_lds_sidecar_entry_words_for_test(result.elf_bytes);
  ASSERT_TRUE(entry.has_value());
  EXPECT_TRUE(contains_flat_global_op_with_null_saddr(entry->words, entry->word_count, /*op=*/28));
  EXPECT_TRUE(contains_sopp(entry->words, entry->word_count, /*op=*/8));
  EXPECT_GE(count_cdna3_vop1_writes(entry->words, entry->word_count, /*op=*/2, /*vdst=*/100), 2u);
  EXPECT_GE(count_cdna3_vop1_writes(entry->words, entry->word_count, /*op=*/2, /*vdst=*/101), 2u);
}

TEST(BinaryTranslatorE2E, VirtualLdsTransposeSemanticRuleUsesSpillPerUseAccessEmitter) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b64_tr_b16_words();
  std::vector<uint32_t> words;
  words.reserve(102 + 3);
  for (uint16_t sgpr = 0; sgpr < 102; ++sgpr)
    words.push_back(rocjitsu::build_s_mov_b32(sgpr, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4));
  words.push_back(ds[0]);
  words.push_back(ds[1]);
  words.push_back(kCdna4SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  13);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  const auto entry = virtual_lds_sidecar_entry_words_for_test(result.elf_bytes);
  ASSERT_TRUE(entry.has_value());
  EXPECT_TRUE(contains_flat_global_op_with_null_saddr(entry->words, entry->word_count, /*op=*/21));
  EXPECT_TRUE(contains_sopp(entry->words, entry->word_count, /*op=*/8));
  EXPECT_GE(count_cdna3_vop1_writes(entry->words, entry->word_count, /*op=*/2, /*vdst=*/100), 2u);
  EXPECT_GE(count_cdna3_vop1_writes(entry->words, entry->word_count, /*op=*/2, /*vdst=*/101), 2u);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarSpillPerUseGrowsDescriptorForBorrowedHighSgprs) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_mov_b32(/*sdst=*/100, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4),
      ds[0],
      ds[1],
      kCdna4SEndpgm,
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  2);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  expect_cdna3_sidecar_descriptor_sgprs_eq(result.elf_bytes, 104);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *metadata_section =
      rocjitsu::test_support::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  EXPECT_EQ(parsed->front().virtual_lds_base_sgpr, 98u);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarSpillPerUseCanGrowTinyNoAgprKernelsPastAccumOffset) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write_b32_words(/*addr=*/0, /*data=*/1);
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_mov_b32(/*sdst=*/100, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4),
      ds[0],
      ds[1],
      kCdna4SEndpgm,
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  2);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  bool found_full_vaddr_store = false;
  for (size_t i = 0; i + 1 < target_word_count; ++i) {
    rocjitsu::cdna3::FlatMachineInst actual{};
    std::memcpy(&actual, target_words + i, sizeof(actual));
    if (actual.encoding == 0x37u && actual.op == 28u && actual.seg == 2u && actual.addr == 0u &&
        actual.saddr == 0x7Fu) {
      found_full_vaddr_store = true;
      break;
    }
  }
  EXPECT_TRUE(found_full_vaddr_store);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarRaisesDescriptorForExplicitHighDsVgprs) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds_write = make_cdna4_ds_write_b32_words(/*addr=*/62, /*data=*/0, /*byte_offset=*/256);
  const auto ds_read = make_cdna4_ds_read_b32_words(/*byte_offset=*/256, /*addr=*/62, /*vdst=*/116);
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_mov_b32(/*sdst=*/100, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4),
      ds_write[0],
      ds_write[1],
      ds_read[0],
      ds_read[1],
      kCdna4SEndpgm,
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  2);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  // The source descriptor intentionally under-declares the inline-assembly
  // VGPRs. Virtual-LDS lowering must grow the target descriptor for both the
  // synthetic GLOBAL address high half v63 and the explicit read destination
  // v116 before the runtime dispatches the translated body.
  expect_cdna3_sidecar_descriptor_vgprs_at_least(result.elf_bytes, 117);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarSpillPerUseCapturesWrapperBeforeGuestClobber) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  const uint32_t touch_s100 =
      rocjitsu::build_s_mov_b32(/*sdst=*/100, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4);
  const uint32_t touch_s101 =
      rocjitsu::build_s_mov_b32(/*sdst=*/101, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4);
  const uint32_t clobber_dispatch_ptr =
      rocjitsu::build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {touch_s100, touch_s101, clobber_dispatch_ptr, ds[0], ds[1], kCdna4SEndpgm});

  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  source_kd->group_segment_fixed_size = 6144;
  source_kd->kernarg_size = 0;
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  13);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
  AMDHSA_BITS_SET(source_kd->kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR,
                  1);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  const auto entry = virtual_lds_sidecar_entry_words_for_test(result.elf_bytes);
  ASSERT_TRUE(entry.has_value());
  const auto *entry_words = entry->words;
  const size_t entry_word_count = std::min(entry->word_count, size_t{48});
  // The descriptor entry must consume the wrapper kernarg pointer before the
  // guest body executes `s_mov_b32 s0, 0`. Spill-per-use virtual LDS then
  // reloads the backing pointer from private scratch, not from a guest-owned
  // ABI pointer SGPR.
  EXPECT_TRUE(contains_smem_load_dwordx2_with_wait(entry_words, entry_word_count,
                                                   /*sdata=*/100, /*sbase_sgpr=*/2,
                                                   /*offset=*/8));

  EXPECT_TRUE(contains_flat_scratch_dword_offset(entry_words, entry_word_count, /*op=*/28,
                                                 /*offset=*/0, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword_offset(entry_words, entry_word_count, /*op=*/28,
                                                 /*offset=*/4, /*is_load=*/false));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarSpillPerUseKeepsEntryPrologueStableAfterVgprGrowth) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  const auto grow_vgprs =
      make_cdna4_permlane32_swap_b32_words(/*encoding_id=*/cdna4::encoding::kVop1Hi3);
  const uint32_t touch_s100 =
      rocjitsu::build_s_mov_b32(/*sdst=*/100, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4);
  const uint32_t touch_s101 =
      rocjitsu::build_s_mov_b32(/*sdst=*/101, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {touch_s100, touch_s101, ds[0], ds[1], grow_vgprs[0], grow_vgprs[1], kCdna4SEndpgm});

  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  source_kd->group_segment_fixed_size = 6144;
  source_kd->kernarg_size = 0;
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  13);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
  AMDHSA_BITS_SET(source_kd->kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR,
                  1);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_min_free_vgpr = 193;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  expect_cdna3_sidecar_descriptor_vgprs_at_least(result.elf_bytes, 196);

  const auto entry = virtual_lds_sidecar_entry_words_for_test(result.elf_bytes);
  ASSERT_TRUE(entry.has_value());
  const auto *entry_words = entry->words;
  const size_t entry_word_count = std::min(entry->word_count, size_t{48});
  // The test-only liveness floor makes the permlane lowering grow the final
  // target VGPR count to at least v195 after the entry prologue has already
  // been emitted. Spill-per-use virtual LDS must still reuse the original low
  // entry temps during descriptor recomputation instead of regenerating the
  // prologue with v194/v195.
  const auto saved_lo = find_cdna3_vop1_vdst(entry_words, entry_word_count, /*op=*/1, /*src0=*/100);
  const auto saved_hi = find_cdna3_vop1_vdst(entry_words, entry_word_count, /*op=*/1, /*src0=*/101);
  ASSERT_TRUE(saved_lo.has_value());
  ASSERT_TRUE(saved_hi.has_value());
  EXPECT_LT(*saved_lo, 194u);
  EXPECT_LT(*saved_hi, 194u);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarEmitsRuntimeMetadataSection) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  source_kd->group_segment_fixed_size = 6144;

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *metadata_section =
      rocjitsu::test_support::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);

  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  const auto &record = parsed->front();
  const auto sidecar = find_sidecar_metadata_record_for_test(result.elf_bytes);
  const auto extension = find_kernarg_extension_record_for_test(result.elf_bytes);
  ASSERT_TRUE(sidecar.has_value());
  ASSERT_TRUE(extension.has_value());
  const auto wrapper_layout = make_kernarg_extension_layout_for_test(*extension);
  ASSERT_TRUE(wrapper_layout.has_value());
  EXPECT_EQ(record.kernel_name, "kernel");
  EXPECT_EQ(sidecar->normal_descriptor_vaddr, translated.kernel_descriptor_offset("kernel"));
  EXPECT_NE(sidecar->variant_descriptor_vaddr, sidecar->normal_descriptor_vaddr);
  EXPECT_EQ(record.static_lds_bytes, 6144u);
  EXPECT_EQ(extension->original_kernarg_size, 16u);
  ASSERT_EQ(wrapper_layout->payload_offsets.size(), 1u);
  EXPECT_EQ(wrapper_layout->payload_offsets.front(), 24u);
  EXPECT_NE(record.virtual_lds_base_sgpr, 0u);
  EXPECT_NE(record.flags & rocjitsu::kVirtualLdsFlagRuntimeStateBlock, 0u);

  const auto normal_kd =
      read_descriptor_at_loaded_vaddr_for_test(result.elf_bytes, sidecar->normal_descriptor_vaddr);
  ASSERT_TRUE(normal_kd.has_value());
  EXPECT_EQ(normal_kd->kernarg_size, 16u);

  const auto sidecar_kd =
      read_descriptor_at_loaded_vaddr_for_test(result.elf_bytes, sidecar->variant_descriptor_vaddr);
  ASSERT_TRUE(sidecar_kd.has_value());
  EXPECT_EQ(sidecar_kd->kernarg_size, 48u);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarZeroKernargUsesWrapperKernarg) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  source_kd->group_segment_fixed_size = 6144;
  source_kd->kernarg_size = 0;
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
  AMDHSA_BITS_SET(source_kd->kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR,
                  1);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  const auto saddr =
      find_flat_global_load_saddr(target_words, target_word_count, /*op=*/20, /*vdst=*/7,
                                  /*addr=*/4, /*offset=*/0x134);
  ASSERT_TRUE(saddr.has_value());
  EXPECT_TRUE(contains_smem_load_dwordx2_with_wait(target_words, target_word_count, *saddr,
                                                   /*sbase_sgpr=*/2, /*offset=*/8));

  const auto *metadata_section =
      rocjitsu::test_support::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  const auto &record = parsed->front();
  const auto sidecar = find_sidecar_metadata_record_for_test(result.elf_bytes);
  const auto extension = find_kernarg_extension_record_for_test(result.elf_bytes);
  ASSERT_TRUE(sidecar.has_value());
  ASSERT_TRUE(extension.has_value());
  const auto wrapper_layout = make_kernarg_extension_layout_for_test(*extension);
  ASSERT_TRUE(wrapper_layout.has_value());
  EXPECT_EQ(extension->original_kernarg_size, 0u);
  ASSERT_EQ(wrapper_layout->payload_offsets.size(), 1u);
  EXPECT_EQ(wrapper_layout->payload_offsets.front(), 8u);
  EXPECT_NE(record.flags & rocjitsu::kVirtualLdsFlagRuntimeStateBlock, 0u);

  const auto normal_kd =
      read_descriptor_at_loaded_vaddr_for_test(result.elf_bytes, sidecar->normal_descriptor_vaddr);
  const auto sidecar_kd =
      read_descriptor_at_loaded_vaddr_for_test(result.elf_bytes, sidecar->variant_descriptor_vaddr);
  ASSERT_TRUE(normal_kd.has_value());
  ASSERT_TRUE(sidecar_kd.has_value());
  EXPECT_EQ(record.normal_private_segment_size, normal_kd->private_segment_fixed_size);
  EXPECT_EQ(record.virtual_private_segment_size, sidecar_kd->private_segment_fixed_size);
  EXPECT_EQ(sidecar_kd->kernarg_size, 32u);
  EXPECT_EQ(AMDHSA_BITS_GET(sidecar_kd->kernel_code_properties,
                            KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR),
            1u)
      << "kernel_code_properties=0x" << std::hex << sidecar_kd->kernel_code_properties << std::dec;
  EXPECT_EQ(AMDHSA_BITS_GET(sidecar_kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT), 4u);
}

TEST(BinaryTranslatorE2E, LdsKernelEmitsNormalAndVirtualDescriptorVariants) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b32_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  source_kd->group_segment_fixed_size = 6144;

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *metadata_section =
      rocjitsu::test_support::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);

  const auto &record = parsed->front();
  const auto sidecar = find_sidecar_metadata_record_for_test(result.elf_bytes);
  const auto extension = find_kernarg_extension_record_for_test(result.elf_bytes);
  ASSERT_TRUE(sidecar.has_value());
  ASSERT_TRUE(extension.has_value());
  const auto wrapper_layout = make_kernarg_extension_layout_for_test(*extension);
  ASSERT_TRUE(wrapper_layout.has_value());
  EXPECT_EQ(record.kernel_name, "kernel");
  EXPECT_EQ(sidecar->normal_descriptor_vaddr, translated.kernel_descriptor_offset("kernel"));
  EXPECT_NE(sidecar->variant_descriptor_vaddr, sidecar->normal_descriptor_vaddr);
  EXPECT_EQ(record.static_lds_bytes, 6144u);
  EXPECT_EQ(extension->original_kernarg_size, 16u);
  ASSERT_EQ(wrapper_layout->payload_offsets.size(), 1u);
  EXPECT_EQ(wrapper_layout->payload_offsets.front(), 24u);
  EXPECT_NE(record.virtual_lds_base_sgpr, 0u);
  EXPECT_NE(record.flags & rocjitsu::kVirtualLdsFlagRuntimeStateBlock, 0u);

  const auto *translated_rodata = rocjitsu::test_support::find_section(translated, ".rodata");
  ASSERT_NE(translated_rodata, nullptr);
  const auto normal_kd = rocjitsu::test_support::read_elf_struct_for_test<
      rocjitsu::test_support::TestKernelDescriptor>(result.elf_bytes,
                                                    translated_rodata->sectionOffset());
  EXPECT_EQ(normal_kd.group_segment_fixed_size, 6144u);
  EXPECT_EQ(normal_kd.kernarg_size, 16u);

  const auto virtual_descriptor_offset = rocjitsu::test_support::loaded_vaddr_to_file_offset(
      result.elf_bytes, sidecar->variant_descriptor_vaddr);
  ASSERT_TRUE(virtual_descriptor_offset.has_value());
  const auto virtual_kd = rocjitsu::test_support::read_elf_struct_for_test<
      rocjitsu::test_support::TestKernelDescriptor>(result.elf_bytes, *virtual_descriptor_offset);
  EXPECT_EQ(virtual_kd.group_segment_fixed_size, 0u);
  EXPECT_EQ(virtual_kd.kernarg_size, 48u);
}

TEST(BinaryTranslatorE2E, MubufLdsKernelEmitsNormalAndVirtualDescriptorVariants) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto mubuf = make_cdna4_buffer_load_lds_words(/*op=*/23);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {mubuf[0], mubuf[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  source_kd->group_segment_fixed_size = 6144;

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  // MUBUF instructions encode the LDS side effect in a modifier bit, not in the
  // mnemonic. The sidecar detector must inspect the decoded machine fields, or
  // MUBUF-only LDS kernels would keep only the normal hardware-LDS descriptor.
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *metadata_section =
      rocjitsu::test_support::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName);
  ASSERT_NE(metadata_section, nullptr);
  const auto parsed = rocjitsu::parse_virtual_lds_metadata(
      {reinterpret_cast<const uint8_t *>(metadata_section->data()), metadata_section->size()});
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);

  const auto &record = parsed->front();
  const auto sidecar = find_sidecar_metadata_record_for_test(result.elf_bytes);
  const auto extension = find_kernarg_extension_record_for_test(result.elf_bytes);
  ASSERT_TRUE(sidecar.has_value());
  ASSERT_TRUE(extension.has_value());
  const auto wrapper_layout = make_kernarg_extension_layout_for_test(*extension);
  ASSERT_TRUE(wrapper_layout.has_value());
  EXPECT_EQ(record.kernel_name, "kernel");
  EXPECT_EQ(sidecar->normal_descriptor_vaddr, translated.kernel_descriptor_offset("kernel"));
  EXPECT_NE(sidecar->variant_descriptor_vaddr, sidecar->normal_descriptor_vaddr);
  EXPECT_EQ(record.static_lds_bytes, 6144u);
  EXPECT_EQ(extension->original_kernarg_size, 16u);
  ASSERT_EQ(wrapper_layout->payload_offsets.size(), 1u);
  EXPECT_EQ(wrapper_layout->payload_offsets.front(), 24u);
  EXPECT_NE(record.virtual_lds_base_sgpr, 0u);
  EXPECT_NE(record.flags & rocjitsu::kVirtualLdsFlagRuntimeStateBlock, 0u);

  const auto *translated_rodata = rocjitsu::test_support::find_section(translated, ".rodata");
  ASSERT_NE(translated_rodata, nullptr);
  const auto normal_kd = rocjitsu::test_support::read_elf_struct_for_test<
      rocjitsu::test_support::TestKernelDescriptor>(result.elf_bytes,
                                                    translated_rodata->sectionOffset());
  EXPECT_EQ(normal_kd.group_segment_fixed_size, 6144u);
  EXPECT_EQ(normal_kd.kernarg_size, 16u);

  const auto virtual_descriptor_offset = rocjitsu::test_support::loaded_vaddr_to_file_offset(
      result.elf_bytes, sidecar->variant_descriptor_vaddr);
  ASSERT_TRUE(virtual_descriptor_offset.has_value());
  const auto virtual_kd = rocjitsu::test_support::read_elf_struct_for_test<
      rocjitsu::test_support::TestKernelDescriptor>(result.elf_bytes, *virtual_descriptor_offset);
  EXPECT_EQ(virtual_kd.group_segment_fixed_size, 0u);
  EXPECT_EQ(virtual_kd.kernarg_size, 48u);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarMaterializesLargeDsReadB32Offset) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  constexpr uint16_t kLargeDsOffset = 0x1234;
  const auto ds = make_cdna4_ds_read_b32_words(kLargeDsOffset);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/20, /*vdst=*/7,
                                        /*addr=*/4, /*offset=*/0));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsReadU16ToFlatGlobalLoadUshort) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_u16_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/18, /*vdst=*/7,
                                        /*addr=*/4, /*offset=*/0x20));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsReadU8ToFlatGlobalLoadUbyte) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_u8_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/16, /*vdst=*/12,
                                        /*addr=*/8, /*offset=*/0x20));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsReadI8ToFlatGlobalLoadSbyte) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_i8_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/17, /*vdst=*/12,
                                        /*addr=*/8, /*offset=*/0x20));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsWriteB8ToFlatGlobalStoreByte) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write_b8_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/24, /*data=*/7,
                                         /*addr=*/4, /*offset=*/0x10));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsWriteB8D16HiToFlatGlobalStoreByteD16Hi) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write_b8_d16_hi_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/25, /*data=*/7,
                                         /*addr=*/4, /*offset=*/0x10));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesDsWriteB96DataOverlappingAddressPair) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write_b96_words(/*addr=*/4, /*data=*/4);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_data = find_flat_global_store_data(target_words, target_word_count, /*op=*/30,
                                                       /*addr=*/4, /*offset=*/0);
  ASSERT_TRUE(staged_data.has_value());
  EXPECT_NE(*staged_data, 4u);
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, *staged_data,
                                  /*src0=*/256 + 4));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1,
                                  static_cast<uint8_t>(*staged_data + 1), /*src0=*/256 + 5));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1,
                                  static_cast<uint8_t>(*staged_data + 2), /*src0=*/256 + 6));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesOddDsAddressIntoEvenGlobalAddressPair) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write_b128_words(/*addr=*/7, /*data=*/58);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_addr =
      find_flat_global_store_addr_for_data(target_words, target_word_count, /*op=*/31,
                                           /*data=*/58, /*offset=*/0);
  ASSERT_TRUE(staged_addr.has_value());
  EXPECT_EQ(*staged_addr % 2, 0u);
  EXPECT_NE(*staged_addr, 7u);
  EXPECT_EQ(find_vop2_literal_add(target_words, target_word_count, /*vsrc1=*/7, /*literal=*/0),
            staged_addr);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersAccDsWriteB128ToAccFlatGlobalStore) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write_b128_words(/*addr=*/28, /*data=*/30, /*byte_offset=*/0x40,
                                                 /*acc=*/true);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_store_acc(target_words, target_word_count, /*op=*/31,
                                             /*data=*/30, /*addr=*/28, /*offset=*/0x40));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesOddWideStoreDataIntoEvenTuple) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write_b64_words(/*addr=*/4, /*data=*/7);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_data = find_flat_global_store_data(target_words, target_word_count, /*op=*/29,
                                                       /*addr=*/4, /*offset=*/0);
  ASSERT_TRUE(staged_data.has_value());
  EXPECT_EQ(*staged_data % 2, 0u);
  EXPECT_NE(*staged_data, 7u);
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, *staged_data,
                                  /*src0=*/256 + 7));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1,
                                  static_cast<uint8_t>(*staged_data + 1), /*src0=*/256 + 8));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesOddWideLoadResultIntoEvenTuple) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b64_words(/*addr=*/4, /*vdst=*/7);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_vdst = find_flat_global_load_vdst(target_words, target_word_count, /*op=*/21,
                                                      /*addr=*/4, /*offset=*/0);
  ASSERT_TRUE(staged_vdst.has_value());
  EXPECT_EQ(*staged_vdst % 2, 0u);
  EXPECT_NE(*staged_vdst, 7u);
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/7,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst)));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/8,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst + 1)));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsReadB128ToFlatGlobalLoad) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b128_words(/*addr=*/28, /*vdst=*/30, /*byte_offset=*/0x40);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/23, /*vdst=*/30,
                                        /*addr=*/28, /*offset=*/0x40));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersAccDsReadB128ToAccFlatGlobalLoad) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b128_words(/*addr=*/28, /*vdst=*/30, /*byte_offset=*/0x40,
                                                /*acc=*/true);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load_acc(target_words, target_word_count, /*op=*/23,
                                            /*vdst=*/30, /*addr=*/28, /*offset=*/0x40));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesOddDsReadB128AddressIntoEvenGlobalPair) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b128_words(/*addr=*/27, /*vdst=*/30);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_addr =
      find_flat_global_load_addr_for_vdst(target_words, target_word_count, /*op=*/23,
                                          /*vdst=*/30, /*offset=*/0);
  ASSERT_TRUE(staged_addr.has_value());
  EXPECT_EQ(*staged_addr % 2, 0u);
  EXPECT_NE(*staged_addr, 27u);
  EXPECT_EQ(find_vop2_literal_add(target_words, target_word_count, /*vsrc1=*/27, /*literal=*/0),
            staged_addr);
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesOddDsReadB128ResultIntoEvenTuple) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b128_words(/*addr=*/28, /*vdst=*/31);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_vdst = find_flat_global_load_vdst(target_words, target_word_count, /*op=*/23,
                                                      /*addr=*/28, /*offset=*/0);
  ASSERT_TRUE(staged_vdst.has_value());
  EXPECT_EQ(*staged_vdst % 2, 0u);
  EXPECT_NE(*staged_vdst, 31u);
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/31,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst)));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/32,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst + 1)));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/33,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst + 2)));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/34,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst + 3)));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsWrite2B32ToFlatGlobalStores) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write2_b32_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/28, /*data=*/7,
                                         /*addr=*/4, /*offset=*/4));
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/28, /*data=*/9,
                                         /*addr=*/4, /*offset=*/8));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsWrite2B64ToFlatGlobalStores) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write2_b64_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/29, /*data=*/8,
                                         /*addr=*/4, /*offset=*/8));
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/29, /*data=*/12,
                                         /*addr=*/4, /*offset=*/16));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesDsWrite2B64DataOverlappingAddressPair) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write2_b64_words(/*addr=*/8, /*data0=*/8, /*data1=*/12);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_data = find_flat_global_store_data(target_words, target_word_count, /*op=*/29,
                                                       /*addr=*/8, /*offset=*/8);
  ASSERT_TRUE(staged_data.has_value());
  EXPECT_NE(*staged_data, 8u);
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, *staged_data,
                                  /*src0=*/256 + 8));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1,
                                  /*vdst=*/static_cast<uint8_t>(*staged_data + 1),
                                  /*src0=*/256 + 9));
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/29, /*data=*/12,
                                         /*addr=*/8, /*offset=*/16));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsWrite2st64B64ToFlatGlobalStores) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_write2st64_b64_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/29, /*data=*/8,
                                         /*addr=*/4, /*offset=*/512));
  EXPECT_TRUE(contains_flat_global_store(target_words, target_word_count, /*op=*/29, /*data=*/12,
                                         /*addr=*/4, /*offset=*/1024));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsRead2B32ToFlatGlobalLoads) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read2_b32_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/20, /*vdst=*/20,
                                        /*addr=*/12, /*offset=*/4));
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/20, /*vdst=*/21,
                                        /*addr=*/12, /*offset=*/8));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesDsRead2B32ResultsOverlappingAddressPair) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read2_b32_words(/*offset0=*/1, /*offset1=*/2, /*addr=*/20,
                                                /*vdst=*/20);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto temp0 = find_flat_global_load_vdst(target_words, target_word_count, /*op=*/20,
                                                /*addr=*/20, /*offset=*/4);
  const auto temp1 = find_flat_global_load_vdst(target_words, target_word_count, /*op=*/20,
                                                /*addr=*/20, /*offset=*/8);
  ASSERT_TRUE(temp0.has_value());
  ASSERT_TRUE(temp1.has_value());
  EXPECT_NE(*temp0, 20u);
  EXPECT_NE(*temp1, 21u);
  EXPECT_EQ(*temp1, static_cast<uint8_t>(*temp0 + 1));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/20,
                                  /*src0=*/static_cast<uint16_t>(256u + *temp0)));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/21,
                                  /*src0=*/static_cast<uint16_t>(256u + *temp1)));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsRead2st64B32ToFlatGlobalLoads) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read2st64_b32_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/20, /*vdst=*/20,
                                        /*addr=*/12, /*offset=*/512));
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/20, /*vdst=*/21,
                                        /*addr=*/12, /*offset=*/768));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsRead2B64ToFlatGlobalLoads) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read2_b64_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/21, /*vdst=*/66,
                                        /*addr=*/58, /*offset=*/24));
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/21, /*vdst=*/68,
                                        /*addr=*/58, /*offset=*/544));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesOddWideRead2ResultsIntoEvenTuple) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read2_b64_words(/*offset0=*/3, /*offset1=*/68, /*addr=*/58,
                                                /*vdst=*/67);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_vdst = find_flat_global_load_vdst(target_words, target_word_count, /*op=*/21,
                                                      /*addr=*/58, /*offset=*/24);
  ASSERT_TRUE(staged_vdst.has_value());
  EXPECT_EQ(*staged_vdst % 2, 0u);
  EXPECT_NE(*staged_vdst, 67u);
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/67,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst)));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/68,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst + 1)));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/69,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst + 2)));
  EXPECT_TRUE(contains_cdna3_vop1(target_words, target_word_count, /*op=*/1, /*vdst=*/70,
                                  /*src0=*/static_cast<uint16_t>(256u + *staged_vdst + 3)));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsRead2st64B64ToFlatGlobalLoads) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read2st64_b64_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/21, /*vdst=*/112,
                                        /*addr=*/98, /*offset=*/1024));
  EXPECT_TRUE(contains_flat_global_load(target_words, target_word_count, /*op=*/21, /*vdst=*/114,
                                        /*addr=*/98, /*offset=*/1536));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarUsesDistinctSpillSlotsForTwoAddressTemps) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  // vdst aliases addr, so the replacement must preserve the original address
  // VGPR. Offsets 9 and 10 exceed the flat/global immediate range after B64 ST64
  // scaling. The descriptor is already at the ordinary VGPR ceiling and the body
  // touches s[100:101], so virtual LDS must use the spill-per-use path: the
  // preserved address temp and two backing-pointer VGPR temps are live together
  // and must receive distinct private scratch slots.
  const auto ds = make_cdna4_ds_read2st64_b64_words(/*offset0=*/9, /*offset1=*/10,
                                                    /*addr=*/4, /*vdst=*/4);
  const uint32_t touch_s100 =
      rocjitsu::build_s_mov_b32(/*sdst=*/100, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4);
  const uint32_t touch_s101 =
      rocjitsu::build_s_mov_b32(/*sdst=*/101, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_CDNA4);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {touch_s100, touch_s101, ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  63);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  13);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 63);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  EXPECT_TRUE(contains_flat_scratch_dword_offset(target_words, target_word_count, /*op=*/28,
                                                 /*offset=*/16, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword_offset(target_words, target_word_count, /*op=*/28,
                                                 /*offset=*/20, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword_offset(target_words, target_word_count, /*op=*/28,
                                                 /*offset=*/24, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword_offset(target_words, target_word_count, /*op=*/20,
                                                 /*offset=*/16, /*is_load=*/true));
  EXPECT_TRUE(contains_flat_scratch_dword_offset(target_words, target_word_count, /*op=*/20,
                                                 /*offset=*/20, /*is_load=*/true));
  EXPECT_TRUE(contains_flat_scratch_dword_offset(target_words, target_word_count, /*op=*/20,
                                                 /*offset=*/24, /*is_load=*/true));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarLowersDsReadB64TrB16LoadToFlatGlobal) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b64_tr_b16_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_addr =
      find_vop2_literal_add(target_words, target_word_count, /*vsrc1=*/2, /*literal=*/0);
  ASSERT_TRUE(staged_addr.has_value());
  EXPECT_EQ(*staged_addr % 2, 0u);
  EXPECT_TRUE(contains_vop3_mov_b32(target_words, target_word_count,
                                    static_cast<uint8_t>(*staged_addr + 1),
                                    rocjitsu::scalar_positive_inline_u32(0)));
  EXPECT_TRUE(contains_flat_global_load_addr(target_words, target_word_count, /*op=*/21,
                                             *staged_addr, /*offset=*/0));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarStagesOddDsReadB64TrB16AddressIntoEvenGlobalPair) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_read_b64_tr_b16_words(/*byte_offset=*/0, /*addr=*/7);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto staged_addr =
      find_vop2_literal_add(target_words, target_word_count, /*vsrc1=*/7, /*literal=*/0);
  ASSERT_TRUE(staged_addr.has_value());
  EXPECT_EQ(*staged_addr % 2, 0u);
  EXPECT_NE(*staged_addr, 7u);
  EXPECT_TRUE(contains_vop3_mov_b32(target_words, target_word_count,
                                    static_cast<uint8_t>(*staged_addr + 1),
                                    rocjitsu::scalar_positive_inline_u32(0)));
  EXPECT_TRUE(contains_flat_global_load_addr(target_words, target_word_count, /*op=*/21,
                                             *staged_addr, /*offset=*/0));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarMaterializesLargeDsReadB64TrB16Offset) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  constexpr uint16_t kLargeDsOffset = 0x1234;
  const auto ds = make_cdna4_ds_read_b64_tr_b16_words(kLargeDsOffset);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  const auto temp =
      find_vop2_literal_add(target_words, target_word_count, /*vsrc1=*/2, kLargeDsOffset);
  ASSERT_TRUE(temp.has_value());
  EXPECT_EQ(*temp % 2, 0u);
  EXPECT_TRUE(contains_vop3_mov_b32(target_words, target_word_count,
                                    static_cast<uint8_t>(*temp + 1),
                                    rocjitsu::scalar_positive_inline_u32(0)));
  EXPECT_TRUE(contains_flat_global_load_addr(target_words, target_word_count, /*op=*/21, *temp,
                                             /*offset=*/0));
}

TEST(BinaryTranslatorE2E, VirtualLdsSidecarRejectsUnsupportedDsMemoryOpcode) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_add_u32_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));
  rocjitsu::test_support::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() +
          offsetof(rocjitsu::test_support::TestKernelDescriptor, group_segment_fixed_size),
      105600u);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &d) {
    return d.message.find("virtual LDS lowering does not support this DS opcode") !=
           std::string::npos;
  });
  EXPECT_NE(diagnostic, result.diagnostics.end());
}

TEST(BinaryTranslatorE2E, SkipFailedVirtualLdsSidecarKeepsNormalDescriptor) {
  // Static LDS that fits the host limit: the sidecar is only needed for
  // *dynamic* LDS overflow, so the normal hardware-LDS descriptor is a valid
  // launch target on its own. A sidecar lowering failure must therefore leave
  // the normal descriptor intact.
  constexpr uint32_t kBelowHostLdsBytes = 1024u;
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_add_u32_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));
  rocjitsu::test_support::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() +
          offsetof(rocjitsu::test_support::TestKernelDescriptor, group_segment_fixed_size),
      kBelowHostLdsBytes);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.skip_failed_kernels = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());
  const auto skipped = std::ranges::find_if(result.diagnostics, [](const auto &diagnostic) {
    return diagnostic.kind == rocjitsu::DiagnosticKind::KernelSkipped &&
           diagnostic.message.find("virtual LDS lowering does not support this DS opcode") !=
               std::string::npos;
  });
  ASSERT_NE(skipped, result.diagnostics.end());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *translated_text = translated.text_sections()[0];
  const auto *translated_rodata = rocjitsu::test_support::find_section(translated, ".rodata");
  ASSERT_NE(translated_rodata, nullptr);
  ASSERT_GE(translated_rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));
  const auto normal_kd = rocjitsu::test_support::read_elf_struct_for_test<
      rocjitsu::test_support::TestKernelDescriptor>(result.elf_bytes,
                                                    translated_rodata->sectionOffset());
  const int64_t normal_entry_vaddr =
      static_cast<int64_t>(translated_rodata->vaddr()) +
      rocjitsu::test_support::read_kernel_descriptor_entry_offset(&normal_kd);
  ASSERT_GE(normal_entry_vaddr, 0);
  const auto normal_entry_file_offset = rocjitsu::test_support::loaded_vaddr_to_file_offset(
      result.elf_bytes, static_cast<uint64_t>(normal_entry_vaddr));
  ASSERT_TRUE(normal_entry_file_offset.has_value());

  // The optional virtual-LDS sidecar failed, but below-threshold launches still
  // use the normal descriptor. Its entry must therefore continue to point at the
  // translated hardware-LDS body, not at the skipped-kernel stub appended for
  // the sidecar variant.
  EXPECT_EQ(*normal_entry_file_offset, translated_text->sectionOffset());
  EXPECT_EQ(
      rocjitsu::test_support::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName),
      nullptr);
}

TEST(BinaryTranslatorE2E, SkipFailedVirtualLdsSidecarStubsOversizedNormalDescriptor) {
  // Static LDS that exceeds the host limit: the normal descriptor advertises
  // more hardware LDS than the host has, so it is only launchable through its
  // virtual sidecar. When the sidecar lowering fails and is skipped, the normal
  // descriptor must be stubbed too — leaving it dispatchable would fault the
  // host at launch. This is the regression guard for the sidecar-dependent case.
  constexpr uint32_t kOverHostLdsBytes = 105600u;
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto ds = make_cdna4_ds_add_u32_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {ds[0], ds[1], kCdna4SEndpgm});
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));
  rocjitsu::test_support::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() +
          offsetof(rocjitsu::test_support::TestKernelDescriptor, group_segment_fixed_size),
      kOverHostLdsBytes);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.skip_failed_kernels = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());
  const auto skipped = std::ranges::find_if(result.diagnostics, [](const auto &diagnostic) {
    return diagnostic.kind == rocjitsu::DiagnosticKind::KernelSkipped &&
           diagnostic.message.find("virtual LDS lowering does not support this DS opcode") !=
               std::string::npos;
  });
  ASSERT_NE(skipped, result.diagnostics.end());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *translated_rodata = rocjitsu::test_support::find_section(translated, ".rodata");
  ASSERT_NE(translated_rodata, nullptr);
  ASSERT_GE(translated_rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));
  const auto normal_kd = rocjitsu::test_support::read_elf_struct_for_test<
      rocjitsu::test_support::TestKernelDescriptor>(result.elf_bytes,
                                                    translated_rodata->sectionOffset());
  const int64_t normal_entry_vaddr =
      static_cast<int64_t>(translated_rodata->vaddr()) +
      rocjitsu::test_support::read_kernel_descriptor_entry_offset(&normal_kd);
  ASSERT_GE(normal_entry_vaddr, 0);
  const auto normal_entry_file_offset = rocjitsu::test_support::loaded_vaddr_to_file_offset(
      result.elf_bytes, static_cast<uint64_t>(normal_entry_vaddr));
  ASSERT_TRUE(normal_entry_file_offset.has_value());

  // The normal descriptor's advertised static LDS (105600) exceeds the CDNA3
  // host limit, so it could only ever launch via the failed sidecar. Its entry
  // must now point at the skipped-kernel endpgm stub, and the descriptor must
  // advertise no fixed LDS so a launch attempt cannot fault the host.
  const auto *entry_words =
      reinterpret_cast<const uint32_t *>(result.elf_bytes.data() + *normal_entry_file_offset);
  EXPECT_EQ(entry_words[0], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(normal_kd.group_segment_fixed_size, 0u);
  EXPECT_EQ(
      rocjitsu::test_support::find_section(translated, rocjitsu::kVirtualLdsMetadataSectionName),
      nullptr);
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3KernargPreloadFirmwareEntryKeepsMaskedDefLive) {
  using namespace rocr::llvm::amdhsa;
  constexpr uint16_t kScratchFloor = 120;

  auto image = rocjitsu::test_support::make_large_amdgpu_elf_with_waitcnt_entry();
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_FALSE(layout.text_sections().empty());

  auto source_kd = rocjitsu::test_support::read_kernel_descriptor_for_test(image.data() +
                                                                           rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd.kernarg_preload, KERNARG_PRELOAD_SPEC_LENGTH, 1);
  // Declare a small VGPR file so the globally-unused search is exhausted at the
  // scratch floor, forcing the allocator onto the per-point find_free_run path
  // (the only path the firmware EXEC pin can influence).
  AMDHSA_BITS_SET(source_kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  15);
  rocjitsu::test_support::write_kernel_descriptor_for_test(image.data() + rodata->sectionOffset(),
                                                           source_kd);

  auto *words =
      reinterpret_cast<uint32_t *>(image.data() + layout.text_sections()[0]->sectionOffset());
  words[0] = 0xBEFE01C1u; // s_mov_b64 exec, -1  (ordinary path establishes Full).
  const auto cvt = make_cdna4_cvt_pk_f16_f32_words(); // Needs one scratch temp.
  words[64] = cvt[0];
  words[65] = cvt[1];
  // Masked vector def of the floor VGPR, read afterwards. Under the firmware
  // entry's pinned-Unknown EXEC this def is not a whole-register kill, so v120 is
  // live across the cvt and find_free_run must not reuse it.
  words[66] = cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = 0, .vdst = kScratchFloor})[0];
  words[67] = cdna4::build_vop1(
      cdna4::kVMovB32Vop1, {.src0 = static_cast<uint16_t>(256 + kScratchFloor), .vdst = 20})[0];
  words[68] = cdna4::build_sopp(cdna4::kSEndpgmSopp)[0];

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_min_free_vgpr = kScratchFloor;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  // The packed-F16 lowering converts the high half into a scratch temp picked by
  // find_free_run (globally-unused VGPRs are exhausted at the floor). With the
  // firmware entry pinned Unknown, v120's masked def is not a kill, so v120 is
  // live across the cvt and the temp must skip it (v121). If the firmware entry
  // wrongly inherited Full, v120 would look dead and be reused as the temp (v120),
  // clobbering the value the kernel reads after the preload window.
  const auto writes_cvt_f16_to = [&](uint16_t vdst) {
    for (size_t i = 0; i + 1 < target_word_count; ++i) {
      rocjitsu::cdna3::Vop3MachineInst v{};
      std::memcpy(&v, target_words + i, sizeof(v));
      if (v.encoding == 0x34u && v.op == cdna3::kVCvtF16F32Vop3 && v.vdst == vdst)
        return true;
    }
    return false;
  };
  EXPECT_TRUE(writes_cvt_f16_to(kScratchFloor + 1))
      << "temp must skip the live firmware-entry def register v" << kScratchFloor;
  EXPECT_FALSE(writes_cvt_f16_to(kScratchFloor))
      << "temp reused v" << kScratchFloor << ", clobbering the live firmware-entry def";
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3Bitop3ScratchGrowsDescriptor) {
  constexpr uint16_t kScratchFloor = 120;
  const auto words = make_cdna4_bitop3_words(cdna4::kVBitop3B32Vop3, 16);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {words[0], words[1]});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_min_free_vgpr = kScratchFloor;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  ASSERT_FALSE(result.elf_bytes.empty());
  // This fixture's LUT needs a two-VGPR scratch run. The conservative liveness
  // floor forces that run above the descriptor's original allocation, so missing
  // require_vgprs() feedback would leave the patched descriptor too small.
  expect_cdna3_translated_descriptor_vgprs_at_least(result.elf_bytes, kScratchFloor + 2);
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3Bitop3UsesSpillBackedScratchWhenVgprsAreFull) {
  using namespace rocr::llvm::amdhsa;

  const auto words = make_cdna4_bitop3_words(cdna4::kVBitop3B32Vop3, 16);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {words[0], words[1]});
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  63);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 63);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_min_free_vgpr = 256;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  // With find_free_run() forced above the physical VGPR namespace, the bitop3
  // lowering must borrow descriptor-backed v0/v1, preserve them in the reusable
  // per-lane semantic spill window, and restore them after writing VDST. The
  // forbidden set excludes v16..v19 so the borrowed window cannot clobber the
  // destination or any source while the LUT expression is being synthesized.
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/28,
                                          /*vgpr=*/0, /*offset=*/0, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/28,
                                          /*vgpr=*/1, /*offset=*/4, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/20,
                                          /*vgpr=*/0, /*offset=*/0, /*is_load=*/true));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/20,
                                          /*vgpr=*/1, /*offset=*/4, /*is_load=*/true));
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3CvtPkBf16SpillScratchAvoidsVgprSources) {
  using namespace rocr::llvm::amdhsa;

  const auto words = make_cdna4_cvt_pk_bf16_f32_words(/*vdst=*/0, /*src0=*/256 + 4,
                                                      /*src1=*/256 + 1);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {words[0], words[1]});
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  1);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_min_free_vgpr = 256;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  // Force liveness to report no dead physical VGPR window. The BF16 lowering
  // must then spill a borrowed three-VGPR window, but that window cannot include
  // VDST or the still-needed high source. The old allocator borrowed v[1:3]
  // here and clobbered SRC1 before converting the high packed half.
  EXPECT_FALSE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/28,
                                           /*vgpr=*/1, /*offset=*/0, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/28,
                                          /*vgpr=*/2, /*offset=*/0, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/28,
                                          /*vgpr=*/3, /*offset=*/4, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/28,
                                          /*vgpr=*/4, /*offset=*/8, /*is_load=*/false));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/20,
                                          /*vgpr=*/2, /*offset=*/0, /*is_load=*/true));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/20,
                                          /*vgpr=*/3, /*offset=*/4, /*is_load=*/true));
  EXPECT_TRUE(contains_flat_scratch_dword(target_words, target_word_count, /*op=*/20,
                                          /*vgpr=*/4, /*offset=*/8, /*is_load=*/true));
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3MfmaPartialScratchGrowsDescriptor) {
  constexpr uint16_t kScratchFloor = 121;
  constexpr uint16_t kAlignedScratch = 122;
  const auto words = make_cdna4_mfma_vgpr_dst_alias_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {words[0], words[1]});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_min_free_vgpr = kScratchFloor;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  ASSERT_FALSE(result.elf_bytes.empty());
  // The 16x16x32 F16 lowering uses a four-VGPR partial accumulator when an
  // ordinary destination overlaps the still-needed wide A/B source window.
  expect_cdna3_translated_descriptor_vgprs_at_least(result.elf_bytes, kAlignedScratch + 4);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  std::optional<rocjitsu::cdna3::Vop3pMfmaMachineInst> partial_mfma;
  for (size_t i = 0; i + 1 < target_word_count; ++i) {
    rocjitsu::cdna3::Vop3pMfmaMachineInst actual{};
    std::memcpy(&actual, target_words + i, sizeof(actual));
    if (actual.encoding == 0x1A7u && actual.op == 77u && actual.acc_cd == 0u) {
      partial_mfma = actual;
      break;
    }
  }
  ASSERT_TRUE(partial_mfma.has_value());
  EXPECT_EQ(partial_mfma->vdst, kAlignedScratch);
  EXPECT_EQ(partial_mfma->vdst % 2u, 0u);
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3Dot2ScratchGrowsDescriptor) {
  constexpr uint16_t kScratchFloor = 121;
  const auto words = make_cdna4_dot2_f32_bf16_words(/*vdst=*/0, /*src0=*/256 + 1,
                                                    /*src1=*/256 + 2, /*src2=*/256);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {words[0], words[1]});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_min_free_vgpr = kScratchFloor;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;
  ASSERT_FALSE(result.elf_bytes.empty());
  // The BF16 dot2 fallback widens four packed halves into ordinary VGPR
  // temporaries before issuing scalar FP32 arithmetic. If liveness selects a
  // scratch run above the guest allocation, the descriptor must grow to cover
  // all four generated VGPRs.
  expect_cdna3_translated_descriptor_vgprs_at_least(result.elf_bytes, kScratchFloor + 4);
}

TEST(BinaryTranslatorE2E, Rdna4ScratchAllocationDoesNotWrapPastV255) {
  const auto words = make_cdna4_mfma_words(cdna4::kVMfmaF3216x16x16F16Vop3pMfma, 0, 256, 260);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {words[0], words[1]});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_min_free_vgpr = 256;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4, 0,
                                        options);
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "MFMA lowering could not find a free VGPR for ds_bpermute addresses"));
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3PermlaneRejectsDescriptorFullExecSave) {
  using namespace rocr::llvm::amdhsa;

  const auto permlane =
      make_cdna4_permlane32_swap_b32_words(/*encoding_id=*/cdna4::encoding::kVop1);
  std::vector<uint32_t> words = {permlane[0], permlane[1]};
  for (uint16_t sgpr = 0; sgpr < 102; ++sgpr) {
    // Keep every ordinary SGPR live after the permlane replacement point. With
    // the descriptor already full, the EXEC-save helper must then fail instead
    // of borrowing and clobbering one of these guest scalar values.
    words.push_back(rocjitsu::build_s_mov_b32(sgpr, sgpr, ROCJITSU_CODE_ARCH_CDNA4));
  }
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);

  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  12);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &d) {
    return d.kind == rocjitsu::DiagnosticKind::ExpandFailed &&
           d.message.find("No free descriptor-backed SGPR pair for EXEC save/restore") !=
               std::string::npos;
  });
  EXPECT_NE(diagnostic, result.diagnostics.end());
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3PermlaneExecSaveReservesSpecialSgprTail) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto permlane =
      make_cdna4_permlane32_swap_b32_words(/*encoding_id=*/cdna4::encoding::kVop1);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {permlane[0], permlane[1], kCdna4SEndpgm});

  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  // This mirrors the Qwen SDPA `attn_fwd` shape: the source descriptor allocates
  // 64 SGPRs, while the semantic permlane lowering introduces an EXEC-save pair
  // starting at s64. Growing only through s65 leaves no room for the
  // architecture-owned VCC/flat-scratch/XNACK tail, so DBT must reserve that
  // tail explicitly when materializing the generated ordinary SGPR pair.
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  7);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());
  expect_cdna3_translated_descriptor_sgprs_eq(result.elf_bytes, 80);
}

TEST(BinaryTranslatorE2E, Cdna4ToCdna3Permlane16SwapWritesBothRowPairs) {
  constexpr uint32_t kCdna4SEndpgm = 0xBF810000u;
  const auto permlane =
      make_cdna4_permlane16_swap_b32_words(/*encoding_id=*/cdna4::encoding::kVop1);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {permlane[0], permlane[1], kCdna4SEndpgm});

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  // V_PERMLANE16_SWAP_B32 swaps both 16-lane row pairs: lanes 0..15 with
  // 16..31, and lanes 32..47 with 48..63. The lowering therefore emits the
  // same low/high row masks into EXEC_LO and EXEC_HI.
  constexpr uint8_t kExecLoSgpr = 126;
  constexpr uint8_t kExecHiSgpr = 127;
  EXPECT_EQ(
      count_cdna3_s_mov_b32_literal(target_words, target_word_count, kExecLoSgpr, 0x0000ffffu), 1u);
  EXPECT_EQ(
      count_cdna3_s_mov_b32_literal(target_words, target_word_count, kExecLoSgpr, 0xffff0000u), 1u);
  EXPECT_EQ(
      count_cdna3_s_mov_b32_literal(target_words, target_word_count, kExecHiSgpr, 0x0000ffffu), 1u);
  EXPECT_EQ(
      count_cdna3_s_mov_b32_literal(target_words, target_word_count, kExecHiSgpr, 0xffff0000u), 1u);
}

TEST(BinaryTranslatorE2E, RelocatedKernelCompactsReachableBodyAndPatchesBranches) {
  constexpr uint32_t kCdna4SEndpgm = cdna4::build_sopp(cdna4::kSEndpgmSopp)[0];
  constexpr uint32_t kCdna4SCbranchScc1ToSourceTarget =
      cdna4::build_sopp(cdna4::kSCbranchScc1Sopp, {.simm16 = 4})[0];
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(2, ROCJITSU_CODE_ARCH_CDNA4), // 0x00 -> source 0x0c.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),    // 0x04 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),    // 0x08 unreachable.
      kCdna4SCbranchScc1ToSourceTarget,                      // 0x0c -> 0x20, else 0x10.
      rocjitsu::build_s_branch(4, ROCJITSU_CODE_ARCH_CDNA4), // 0x10 -> source 0x24.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),    // 0x14 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),    // 0x18 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),    // 0x1c unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),    // 0x20 conditional target.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x24 fallthrough-branch target.
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  // This kernel does not use the kernarg-preload compatibility entry path, so
  // relocation emits only the reachable CFG body. Source gaps disappear and
  // branch immediates are patched against the compact target layout.
  const std::vector<uint32_t> expected = {
      rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3),
      cdna3::build_sopp(cdna3::kSCbranchScc1Sopp, {.simm16 = 1})[0],
      rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA3),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA3),
      kCdna4SEndpgm,
  };
  for (size_t i = 0; i < expected.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(target_words[i], expected[i]);
  }
}

TEST(BinaryTranslatorE2E, RelocatedKernelCompactsReachableBlocksAfterEntry) {
  std::vector<uint32_t> words(74, rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  words[0] = rocjitsu::build_s_branch(63, ROCJITSU_CODE_ARCH_CDNA4); // 0x00 -> 0x100.
  words[64] = rocjitsu::build_s_branch(7, ROCJITSU_CODE_ARCH_CDNA4); // 0x100 -> 0x120.
  words[72] = rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);    // Reachable target.

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source_layout(image.data(), image.size());
  ASSERT_TRUE(source_layout.is_valid());
  const auto *source_rodata = rocjitsu::test_support::find_section(source_layout, ".rodata");
  ASSERT_NE(source_rodata, nullptr);
  ASSERT_GE(source_rodata->size(), sizeof(rocr::llvm::amdhsa::kernel_descriptor_t));
  auto source_kd = rocjitsu::test_support::read_kernel_descriptor_for_test(
      image.data() + source_rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd.kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 1);
  rocjitsu::test_support::write_kernel_descriptor_for_test(
      image.data() + source_rodata->sectionOffset(), source_kd);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *text = translated.text_sections()[0];
  ASSERT_GE(text->size(), words.size() * sizeof(uint32_t));

  const auto *target_words = reinterpret_cast<const uint32_t *>(text->data());
  // The synthesized preload launch window occupies words 0 and 64. The compact
  // relocated body starts after that protected window, and the source 0x120
  // target lands immediately after the source 0x100 branch instead of remaining
  // at the original word 72.
  EXPECT_EQ(target_words[0], rocjitsu::build_s_branch(64, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[64], rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[65], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[66], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[67], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[72], rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, PatchesRecoveredSetpcTargetAfterRelocation) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalGetpcDelta = 20;
  const std::vector<uint32_t> words = {
      build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),    // 0x00.
      build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand),      // 0x04.
      kOriginalGetpcDelta,                                     // 0x08.
      build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x0c.
      build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),    // 0x10.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),      // 0x14 unreachable.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),      // 0x18 target.
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  ASSERT_GE(translated.text_sections()[0]->size(), 6 * sizeof(uint32_t));
  // Recovered indirect jumps now patch the consumer site rather than rewriting
  // the source-side getpc/add builder. Its consumer starts as one compact word
  // and grows only if final placement requires the canonical long form.
  EXPECT_EQ(target_words[0], build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand));
  // The consumer became a branch window, leaving this builder dead -- and the compaction that
  // drops the unreachable word has to move its delta with it. Pinning the SOURCE delta, as this
  // test once did, asserted a builder measured against the pre-relocation layout: unreachable at
  // runtime, but on a re-translation indistinguishable from a live producer of an unrelocated code
  // address.
  constexpr uint32_t kRelocatedGetpcDelta =
      5 * sizeof(uint32_t) - /*the getpc leaves the address of the next instruction=*/
      sizeof(uint32_t);
  static_assert(kRelocatedGetpcDelta != kOriginalGetpcDelta,
                "compaction must actually move the target, or this assertion proves nothing");
  EXPECT_EQ(target_words[2], kRelocatedGetpcDelta);
  EXPECT_EQ(target_words[3], build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0));
  EXPECT_EQ(target_words[4], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[5], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, PatchesRecoveredSwappcTargetAfterRelocation) {
  constexpr uint16_t kPcSreg = 10;
  constexpr uint16_t kReturnSreg = 20;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalGetpcDelta = 24;
  const std::vector<uint32_t> words = {
      build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),               // 0x00.
      build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand),                 // 0x04.
      kOriginalGetpcDelta,                                                // 0x08.
      build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0),            // 0x0c.
      build_s_swappc_b64(kReturnSreg, kPcSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x10.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x14 unreachable continuation.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4), // 0x18 unreachable.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x1c target.
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  ASSERT_GE(translated.text_sections()[0]->size(), 6 * sizeof(uint32_t));
  EXPECT_EQ(target_words[0], build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand));
  // The consumer became a call window, leaving this builder dead -- and the same compaction that
  // moves the target has to move its delta with it. Pinning the SOURCE delta here, as this test
  // once did, asserted a builder still measured against the pre-relocation layout: unreachable at
  // runtime, but on a re-translation indistinguishable from a live producer of an unrelocated code
  // address, which is a code object whose translation is not a fixed point.
  constexpr uint32_t kRelocatedGetpcDelta =
      5 * sizeof(uint32_t) - /*the getpc leaves the address of the next instruction=*/
      sizeof(uint32_t);
  static_assert(kRelocatedGetpcDelta != kOriginalGetpcDelta,
                "compaction must actually move the target, or this assertion proves nothing");
  EXPECT_EQ(target_words[2], kRelocatedGetpcDelta);
  EXPECT_EQ(target_words[3], build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0));
  // The target is non-returning, so relocation drops the dead continuation
  // and compacts the target immediately after the call.
  EXPECT_EQ(target_words[4], build_s_call_b64(kReturnSreg, 0));
  EXPECT_EQ(target_words[5], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, TranslatesDirectSCallWithSetpcReturn) {
  constexpr uint16_t kReturnSreg = 30;
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(2, ROCJITSU_CODE_ARCH_CDNA4),    // 0x00 -> call block at 0x0c.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x04 unreachable gap.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x08 unreachable gap.
      build_s_call_b64(kReturnSreg, 2),                         // 0x0c -> callee at 0x18.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),       // 0x10 call continuation.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x14 unreachable gap.
      build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x18 callee return.
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], build_s_call_b64(kReturnSreg, 1))
      << "the direct call target must be recomputed after unreachable source gaps are compacted";
  EXPECT_EQ(target_words[2], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[3], build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA3));

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_GE(decoded.size(), 4u);
  ASSERT_EQ(decoded[1]->mnemonic(), "s_call_b64");
  ASSERT_TRUE(decoded[1]->branch_offset_bytes().has_value());
  EXPECT_EQ(*decoded[1]->branch_offset_bytes(), 4)
      << "translated call should branch from word 1 to the relocated return block at word 3";
}

TEST(BinaryTranslatorE2E, TranslatesDirectSCallWhenCalleeBranchesToSetpcReturn) {
  constexpr uint16_t kReturnSreg = 30;
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(2, ROCJITSU_CODE_ARCH_CDNA4),    // 0x00 -> call block at 0x0c.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x04 unreachable gap.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x08 unreachable gap.
      build_s_call_b64(kReturnSreg, 2),                         // 0x0c -> callee at 0x18.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),       // 0x10 call continuation.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x14 unreachable gap.
      rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4),    // 0x18 callee -> return.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x1c unreachable gap.
      build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x20 callee return.
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], build_s_call_b64(kReturnSreg, 1));
  EXPECT_EQ(target_words[2], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[3], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[4], build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, TranslatesNestedCallReturningThroughOuterPair) {
  constexpr uint16_t kOuterReturnSreg = 30;
  constexpr uint16_t kInnerReturnSreg = 28;
  const std::vector<uint32_t> words = {
      build_s_call_b64(kOuterReturnSreg, 1),              // 0x00 -> outer callee at 0x08.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x04 outer continuation.
      build_s_call_b64(kInnerReturnSreg, 1),              // 0x08 -> inner callee at 0x10.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x0c inner continuation.
      rocjitsu::pack_sopp(5, 1),                          // 0x10 -> outer return at 0x18.
      build_s_setpc_b64(kInnerReturnSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x14 inner return.
      build_s_setpc_b64(kOuterReturnSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x18 outer return.
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_GE(decoded.size(), words.size());
  const auto count_mnemonic = [&](std::string_view mnemonic) {
    return std::ranges::count_if(
        decoded, [&](const auto &inst) { return inst != nullptr && inst->mnemonic() == mnemonic; });
  };
  EXPECT_EQ(count_mnemonic("s_call_b64"), 2);
  EXPECT_EQ(count_mnemonic("s_setpc_b64"), 2);
}

TEST(BinaryTranslatorE2E, TranslatesSwappcCallWhenCalleeSetpcBranchesToReturn) {
  constexpr uint16_t kCallTargetSreg = 10;
  constexpr uint16_t kReturnTargetSreg = 12;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalCallTargetDelta = 28;
  constexpr uint32_t kOriginalReturnTargetDelta = 20;
  const std::vector<uint32_t> words = {
      build_s_getpc_b64(kCallTargetSreg, ROCJITSU_CODE_ARCH_CDNA4),               // 0x00.
      build_s_add_u32(kCallTargetSreg, kCallTargetSreg, kLiteralOperand),         // 0x04.
      kOriginalCallTargetDelta,                                                   // 0x08.
      build_s_addc_u32(kCallTargetSreg + 1, kCallTargetSreg + 1, kInlineInt0),    // 0x0c.
      build_s_swappc_b64(kReturnSreg, kCallTargetSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x10.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),                     // 0x14 continuation.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),                     // 0x18 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),                     // 0x1c unreachable.
      build_s_getpc_b64(kReturnTargetSreg, ROCJITSU_CODE_ARCH_CDNA4),         // 0x20 callee.
      build_s_add_u32(kReturnTargetSreg, kReturnTargetSreg, kLiteralOperand), // 0x24.
      kOriginalReturnTargetDelta,                                             // 0x28.
      build_s_addc_u32(kReturnTargetSreg + 1, kReturnTargetSreg + 1, kInlineInt0), // 0x2c.
      build_s_setpc_b64(kReturnTargetSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x30 -> return.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),             // 0x34 unreachable.
      build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA4),       // 0x38 return.
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  ASSERT_GE(translated.text_sections()[0]->size(), 12 * sizeof(uint32_t));
  EXPECT_EQ(target_words[0], build_s_getpc_b64(kCallTargetSreg, ROCJITSU_CODE_ARCH_CDNA3));
  // Both consumers became transfer windows, so both builders are dead -- but each must still name
  // the instruction it originally named, at that instruction's relocated home. The source builders
  // point at the callee entry (0x20) and at the return setpc (0x38); compaction drops the
  // unreachable words between them, landing those at target words 6 and 11. Asserting the SOURCE
  // deltas here, as this test once did, pinned builders measured against the pre-relocation layout
  // -- unreachable at runtime, yet on a re-translation indistinguishable from live producers of
  // unrelocated code addresses, which is a code object whose translation is not a fixed point.
  constexpr uint32_t kWordBytes = sizeof(uint32_t);
  constexpr uint32_t kRelocatedCallTargetDelta = (6 - 1) * kWordBytes;
  constexpr uint32_t kRelocatedReturnTargetDelta = (11 - 7) * kWordBytes;
  static_assert(kRelocatedCallTargetDelta != kOriginalCallTargetDelta &&
                    kRelocatedReturnTargetDelta != kOriginalReturnTargetDelta,
                "compaction must actually move both targets, or these assertions prove nothing");
  EXPECT_EQ(target_words[2], kRelocatedCallTargetDelta);
  EXPECT_EQ(target_words[4], build_s_call_b64(kReturnSreg, 1))
      << "the swappc call window should patch directly to the compact callee body";
  EXPECT_EQ(target_words[5], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[6], build_s_getpc_b64(kReturnTargetSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[8], kRelocatedReturnTargetDelta);
  EXPECT_EQ(target_words[10], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3))
      << "the callee's recovered setpc window is what reaches the return block";
  EXPECT_EQ(target_words[11], build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, PatchesOneRecoveredBuilderUsedByTwoSetpcConsumers) {
  constexpr uint16_t kPcSreg = 12;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalGetpcDelta = 28;
  const std::vector<uint32_t> words = {
      build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),          // 0x00.
      build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand),            // 0x04.
      kOriginalGetpcDelta,                                           // 0x08.
      build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0),       // 0x0c.
      cdna4::build_sopp(cdna4::kSCbranchScc1Sopp, {.simm16 = 1})[0], // 0x10 -> second consumer.
      build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),          // 0x14 first consumer.
      build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),          // 0x18 carried consumer.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x1c unreachable gap.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x20 shared target.
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  ASSERT_GE(translated.text_sections()[0]->size(), 8 * sizeof(uint32_t));
  EXPECT_EQ(target_words[0], build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand));
  // Both consumers became direct-transfer windows, which leaves this builder dead -- but dead is
  // not the same as free to be wrong. The unreachable gap at source 0x1c is dropped by compaction,
  // so the shared target moves from source 0x20 to target word 7 and the delta from the
  // instruction after the getpc must follow it. Asserting the SOURCE delta here, as this test once
  // did, pinned a builder still measured against the pre-relocation layout: harmless to run, since
  // the windows supersede it, but indistinguishable on a re-translation from a live producer of an
  // unrelocated code address, and so a code object whose translation is not a fixed point.
  constexpr uint32_t kRelocatedGetpcDelta =
      7 * sizeof(uint32_t) - /*the getpc leaves the address of the next instruction=*/
      sizeof(uint32_t);
  static_assert(kRelocatedGetpcDelta != kOriginalGetpcDelta,
                "compaction must actually move the target, or this assertion proves nothing");
  EXPECT_EQ(target_words[2], kRelocatedGetpcDelta);
  EXPECT_EQ(target_words[3], build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0));
  EXPECT_EQ(target_words[4], cdna3::build_sopp(cdna3::kSCbranchScc1Sopp, {.simm16 = 1})[0]);
  EXPECT_EQ(target_words[5], rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[6], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[7], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, RewritesDistinctBuildersForOneMultiTargetSetpcConsumer) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalTargetADelta = 44;
  constexpr uint32_t kOriginalTargetBDelta = 28;
  constexpr uint32_t kRelocatedTargetADelta = 40;
  constexpr uint32_t kRelocatedTargetBDelta = 24;
  const std::vector<uint32_t> words = {
      cdna4::build_sopp(cdna4::kSCbranchScc1Sopp, {.simm16 = 5})[0], // 0x00 -> builder B.
      build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),          // 0x04 builder A.
      build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand),            // 0x08.
      kOriginalTargetADelta,                                         // 0x0c -> target A.
      build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0),       // 0x10.
      rocjitsu::build_s_branch(5, ROCJITSU_CODE_ARCH_CDNA4),         // 0x14 -> consumer.
      build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),          // 0x18 builder B.
      build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand),            // 0x1c.
      kOriginalTargetBDelta,                                         // 0x20 -> target B.
      build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0),       // 0x24.
      rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4),         // 0x28 -> consumer.
      build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),          // 0x2c multi-target consumer.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x30 unreachable.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x34 target A.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x38 target B.
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  ASSERT_GE(translated.text_sections()[0]->size(), 15 * sizeof(uint32_t));
  EXPECT_EQ(target_words[3], kRelocatedTargetADelta)
      << "builder A should be rewritten once for its relocated target";
  EXPECT_EQ(target_words[8], kRelocatedTargetBDelta)
      << "builder B should be rewritten once for its relocated target";
  EXPECT_EQ(target_words[11], build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA3))
      << "one consumer with multiple possible targets must stay indirect";
  EXPECT_EQ(target_words[12], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[13], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));
}

TEST(BinaryTranslatorE2E, RelocatesDirectCallReturnAcrossShiftedOffsets) {
  constexpr uint16_t kReturnSreg = 28;
  std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(4, ROCJITSU_CODE_ARCH_CDNA4),    // 0x00 -> call block at 0x14.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x04 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x08 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x0c unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x10 unreachable.
      build_s_call_b64(kReturnSreg, 6),                         // 0x14 -> callee at 0x30.
      rocjitsu::build_s_branch(6, ROCJITSU_CODE_ARCH_CDNA4),    // 0x18 continuation -> 0x34.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x1c unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x20 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x24 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x28 unreachable.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),       // 0x2c unreachable.
      build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x30 callee return.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),       // 0x34 final continuation.
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[1], build_s_call_b64(kReturnSreg, 1))
      << "source call target 0x30 should relocate to compact word 3";
  EXPECT_EQ(target_words[2], rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA3))
      << "the call continuation branch should relocate to compact word 4";
  EXPECT_EQ(target_words[3], build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(target_words[4], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3));

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_GE(decoded.size(), 5u);
  ASSERT_TRUE(decoded[1]->branch_offset_bytes().has_value());
  EXPECT_EQ(*decoded[1]->branch_offset_bytes(), 4);
}

TEST(BinaryTranslatorE2E, EndpgmAfterTrapTerminatesCfgBeforeFollowingFunction) {
  // A resumable S_TRAP is not a CFG terminator: on hardware it transfers to the
  // trap handler and may return to the next instruction. A real terminator
  // (S_ENDPGM) must follow it to end the block. The S_ENDPGM terminates the CFG
  // so the following ELF function bytes stay unreachable and the unrecovered
  // S_SETPC_B64 below is never decoded into the CFG.
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4), // 0x00 -> trap block.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x04 unreachable gap.
      build_s_trap(3),                                       // 0x08 falls through to endpgm.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x0c terminates the block.
      build_s_setpc_b64(30, ROCJITSU_CODE_ARCH_CDNA4),       // 0x10 next function body.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x14 unreachable.
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_GE(decoded.size(), 3u);
  EXPECT_EQ(decoded[0]->mnemonic(), "s_branch");
  EXPECT_EQ(decoded[1]->mnemonic(), "s_trap");
  EXPECT_EQ(decoded[2]->mnemonic(), "s_endpgm");
  // The trailing S_ENDPGM (a real terminator) prevents a bogus fallthrough into
  // the following ELF function bytes, so the unrecovered S_SETPC_B64 is never
  // reached and translation does not emit it.
  EXPECT_TRUE(std::none_of(decoded.begin(), decoded.end(),
                           [](const auto &inst) { return inst->mnemonic() == "s_setpc_b64"; }));
}

TEST(BinaryTranslatorE2E, ResumableTrapFallsThroughAndDoesNotHideFollowingCode) {
  // Trap IDs other than ROCr's assertion/abort trap remain resumable, so code
  // after them is reachable. Here the fallthrough reaches an unrecovered
  // S_SETPC_B64, which the translator must surface as an error.
  const std::vector<uint32_t> words = {
      build_s_trap(3),                                 // 0x00 falls through.
      build_s_setpc_b64(30, ROCJITSU_CODE_ARCH_CDNA4), // 0x04 reachable, unrecovered.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);

  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::Legalization,
      "indirect branch or call target recovery is not implemented"));
}

TEST(BinaryTranslatorE2E, RocrAbortTrapTerminatesCfgBeforeFollowingCode) {
  // ROCr reserves trap ID 2 for assertion/abort handling. That path does not
  // return, so bytes after S_TRAP 2 must not be pulled into the current CFG.
  const std::vector<uint32_t> words = {
      build_s_trap(2),                                 // 0x00 terminates the program path.
      build_s_setpc_b64(30, ROCJITSU_CODE_ARCH_CDNA4), // 0x04 unrelated, unreachable code.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_FALSE(decoded.empty());
  EXPECT_EQ(decoded[0]->mnemonic(), "s_trap");
  EXPECT_TRUE(std::none_of(decoded.begin(), decoded.end(),
                           [](const auto &inst) { return inst->mnemonic() == "s_setpc_b64"; }));
}

TEST(BinaryTranslatorE2E, RocrAbortDeadEdgeDoesNotPoisonRecoveredCall) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // Exercise BinaryTranslator's ExplicitOnly entry policy rather than calling
  // BasicBlock::build with the policy directly:
  //
  //   builder --conditional--------------------------> call
  //                 |
  //                 +--> s_trap 2 -X-> dead branch --^
  //
  // Treating the dead post-trap block as an inferred entry would add an
  // unconstrained path to s[8:9] and make translation fail closed.
  const std::vector<uint32_t> words = {
      build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA4),    // 0x00.
      build_s_add_u32(kPcSreg, kPcSreg, kLiteralOperand),      // 0x04.
      40,                                                      // 0x08: 0x04 + 40 = helper 0x2c.
      build_s_addc_u32(kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x0c.
      cdna4::build_sopp(cdna4::kSCbranchScc0Sopp, {.simm16 = 3})[0],      // 0x10 -> 0x20.
      build_s_trap(2),                                                    // 0x14.
      rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4),              // 0x18 -> 0x20.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),                 // 0x1c.
      build_s_swappc_b64(kReturnSreg, kPcSreg, ROCJITSU_CODE_ARCH_CDNA4), // 0x20.
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),                 // 0x24.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),                 // 0x28.
      build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA4),           // 0x2c.
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << result.diagnostics.front().message;

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA3);
  EXPECT_TRUE(std::ranges::any_of(
      decoded, [](const auto &inst) { return inst->mnemonic() == "s_call_b64"; }));
}

TEST(BinaryTranslatorE2E, RejectsUnrecoveredIndirectBranchInstructions) {
  struct Case {
    const char *name;
    std::vector<uint32_t> words;
    const char *mnemonic;
  };

  const std::array<Case, 3> cases = {{
      {"SetpcS0", {0xBE801D00u, 0x00000000u}, "s_setpc_b64"},
      {"SetpcS30WithoutCall",
       {build_s_setpc_b64(30, ROCJITSU_CODE_ARCH_CDNA4),
        rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4)},
       "s_setpc_b64"},
      {"Swappc", {0xBE801E00u, 0x00000000u}, "s_swappc_b64"},
  }};

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    auto image =
        rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(test_case.words);
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());

    rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
    auto result = translator.translate(source);

    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
        result, rocjitsu::DiagnosticKind::Legalization,
        "indirect branch or call target recovery is not implemented"));
    const auto diagnostic =
        std::find_if(result.diagnostics.begin(), result.diagnostics.end(),
                     [&](const auto &d) { return d.mnemonic == test_case.mnemonic; });
    EXPECT_NE(diagnostic, result.diagnostics.end());
  }
}

TEST(BinaryTranslatorE2E, RejectsDirectBranchTargetBeforeText) {
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(-2, ROCJITSU_CODE_ARCH_CDNA4), // 0x00 -> -0x04.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);

  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::Legalization,
      "direct branch target is outside the source .text range"));
}

TEST(BinaryTranslatorE2E, RejectsDirectBranchTargetAbsentFromRelocatedBody) {
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4), // 0x00 -> .text end.
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);

  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::Legalization,
      "direct branch target is not present in the kernel-local relocated body"));
}

TEST(BinaryTranslatorE2E, PlacesDescriptorPrologueBeforeLargeRelocatedBody) {
  constexpr size_t kBodyWordsPastBranchRange = 32769;
  std::vector<uint32_t> words(kBodyWordsPastBranchRange, 0xBF800000u);
  words.push_back(0xBF810000u);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::test_support::enable_workgroup_id_x_sgpr(image);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(source);

  EXPECT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  EXPECT_FALSE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ResourceLimit,
      "kernel descriptor prologue branch range exceeds s_branch simm16"));
}

TEST(BinaryTranslatorE2E, ExpandLegalizationWithoutSemanticRuleFails) {
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject source_layout(image.data(), image.size());
  ASSERT_TRUE(source_layout.is_valid());
  ASSERT_FALSE(source_layout.text_sections().empty());

  const auto words = make_cdna4_dot2c_unimplemented_expand_words();
  const auto *source_text = source_layout.text_sections()[0];
  ASSERT_EQ(source_text->size(), words.size() * sizeof(uint32_t));
  std::memcpy(image.data() + source_text->sectionOffset(), words.data(),
              words.size() * sizeof(uint32_t));

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  ASSERT_FALSE(result.diagnostics.empty());
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &d) {
    return d.kind == rocjitsu::DiagnosticKind::ExpandMissing;
  });
  ASSERT_NE(diagnostic, result.diagnostics.end());
  EXPECT_EQ(diagnostic->severity, rocjitsu::DiagnosticSeverity::Error);
  EXPECT_EQ(diagnostic->guest_offset, std::optional<uint64_t>(0));
  EXPECT_FALSE(diagnostic->required_work.empty());
}

TEST(BinaryTranslatorE2E, DebugContinueAfterFailureCollectsMultipleExpandDiagnostics) {
  const auto first = make_cdna4_dot2c_unimplemented_expand_words();
  const auto second = make_cdna4_dot2c_unimplemented_expand_words();
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {first[0], first[1], second[0], second[1]});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslatorOptions options;
  options.debug_continue_after_failure = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3, 0,
                                        options);
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image)
      << "continued-failure diagnostics must not emit partially translated code";

  std::vector<uint64_t> expand_offsets;
  for (const auto &diagnostic : result.diagnostics) {
    if (diagnostic.kind == rocjitsu::DiagnosticKind::ExpandMissing &&
        diagnostic.guest_offset.has_value())
      expand_offsets.push_back(*diagnostic.guest_offset);
  }
  EXPECT_EQ(expand_offsets, (std::vector<uint64_t>{0, 8}));
}

TEST(BinaryTranslatorE2E, Gfx1250CopiesUnaffectedInstructionsForB0ToA0) {
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {kGfx1250SNop, kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  EXPECT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  EXPECT_EQ(result.host_arch, ROCJITSU_CODE_ARCH_CDNA5);
}

TEST(BinaryTranslatorE2E, Gfx1250ReplacesSClauseWithNopForB0ToA0) {
  constexpr auto source_clause = cdna5::build_sopp(cdna5::kSClauseSopp, {.simm16 = 4});
  constexpr auto source_end = cdna5::build_sopp(cdna5::kSEndpgmSopp, {.simm16 = 0});
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_clause[0], source_end[0]});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], cdna5::build_sopp(cdna5::kSNopSopp, {.simm16 = 0})[0]);
  EXPECT_EQ(target_words[1], source_end[0]);
}

TEST(BinaryTranslatorE2E, Gfx1250PreservesSClauseOutsideB0ToA0) {
  constexpr auto source_clause = cdna5::build_sopp(cdna5::kSClauseSopp, {.simm16 = 4});
  constexpr auto source_end = cdna5::build_sopp(cdna5::kSEndpgmSopp, {.simm16 = 0});

  for (const rocjitsu::ProcessorRevision revision : {
           rocjitsu::ProcessorRevision::Gfx1250A0,
           rocjitsu::ProcessorRevision::Gfx1250B0,
       }) {
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {source_clause[0], source_end[0]});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());

    rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
                                          gfx1250_revision_options(revision, revision));
    auto result = translator.translate(source);

    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);
    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_FALSE(translated.text_sections().empty());
    const auto *target_words =
        reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    EXPECT_EQ(target_words[0], source_clause[0]);
    EXPECT_EQ(target_words[1], source_end[0]);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250RelocatesDirectCallForB0ToA0) {
  constexpr uint16_t kReturnSreg = 30;
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_branch(2, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_call_b64(kReturnSreg, 2, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  auto options = gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                          rocjitsu::ProcessorRevision::Gfx1250A0);
  options.verify_rewrite_discharge = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
                                        options);
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  EXPECT_TRUE(result.rewrite_discharge_checked);
  EXPECT_TRUE(result.rewrite_discharge_verified);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA5));
  EXPECT_EQ(target_words[1], rocjitsu::build_s_call_b64(kReturnSreg, 1, ROCJITSU_CODE_ARCH_CDNA5));
  EXPECT_EQ(target_words[2], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));
  EXPECT_EQ(target_words[3], rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5));

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_GE(decoded.size(), 4u);
  EXPECT_EQ(decoded[1]->mnemonic(), "s_call_i64");
  ASSERT_TRUE(decoded[1]->branch_offset_bytes().has_value());
  EXPECT_EQ(*decoded[1]->branch_offset_bytes(), 4);
}

TEST(BinaryTranslatorE2E, Gfx1250DirectTailTransferCompactionIsIdempotent) {
  constexpr uint16_t kReturnSreg = 30;
  const std::vector<uint32_t> words = {
      rocjitsu::build_s_call_b64(kReturnSreg, 1, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], rocjitsu::build_s_call_b64(kReturnSreg, 0, ROCJITSU_CODE_ARCH_CDNA5));
  EXPECT_EQ(target_words[1], rocjitsu::build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));

  rocjitsu::BinaryTranslator verifier(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto second = verifier.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250KernargPreloadUsesSingleDescriptorEntry) {
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {kGfx1250SNop, kGfx1250SEndpgm});

  rocjitsu::AmdGpuCodeObject source_layout(image.data(), image.size());
  ASSERT_TRUE(source_layout.is_valid());
  const auto *source_rodata = rocjitsu::test_support::find_section(source_layout, ".rodata");
  ASSERT_NE(source_rodata, nullptr);
  auto source_kd = rocjitsu::test_support::read_kernel_descriptor_for_test(
      image.data() + source_rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd.kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 1);
  AMDHSA_BITS_SET(source_kd.kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_OFFSET, 2);
  rocjitsu::test_support::write_kernel_descriptor_for_test(
      image.data() + source_rodata->sectionOffset(), source_kd);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto *target_rodata = rocjitsu::test_support::find_section(translated, ".rodata");
  ASSERT_NE(target_rodata, nullptr);
  const auto target_kd = rocjitsu::test_support::read_kernel_descriptor_for_test(
      translated.image_data() + target_rodata->sectionOffset());
  EXPECT_EQ(target_kd.kernarg_preload, source_kd.kernarg_preload)
      << "gfx1250 preload fields remain active while DBT uses the descriptor's single entry";
}

TEST(BinaryTranslatorE2E, Gfx1250CopiesUnaffectedLiteralOperandsForB0ToA0) {
  constexpr uint32_t kVMovB32Literal = 0x7E0202FFu;
  constexpr uint32_t kLiteral = 0x11223344u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {kVMovB32Literal, kLiteral, kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], kVMovB32Literal);
  EXPECT_EQ(target_words[1], kLiteral);
  EXPECT_EQ(target_words[2], kGfx1250SEndpgm);
}

TEST(BinaryTranslatorE2E, Gfx1250CopiesFp8ConversionsWhenClampIsClear) {
  // CLAMP selects the E5M3 encoding these rules exist for. With it clear the
  // instruction already runs on A0, so both rules must decline rather than
  // rewrite it.
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  for (const uint16_t opcode : {cdna5::kVCvtPkFp8F32Vop3, cdna5::kVCvtSrFp8F32Vop3}) {
    const auto source_cvt =
        cdna5::build_vop3(opcode, {.vdst = 30, .clamp = 0, .src0 = 256 + 22, .src1 = 256 + 2});
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {source_cvt[0], source_cvt[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    auto result = translator.translate(source);

    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);
    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_FALSE(translated.text_sections().empty());
    const auto *target_words =
        reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    EXPECT_EQ(target_words[0], source_cvt[0]);
    EXPECT_EQ(target_words[1], source_cvt[1]);
    EXPECT_EQ(target_words[2], kGfx1250SEndpgm);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250EmulatesCvtPkFp8F32ClampSetForA0) {
  constexpr auto source_cvt = cdna5::build_vop3(
      cdna5::kVCvtPkFp8F32Vop3, {.vdst = 30, .clamp = 1, .src0 = 256 + 22, .src1 = 256 + 2});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_cvt[0], source_cvt[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &item) { return item->mnemonic() == "v_cvt_pk_fp8_f32"; }),
            0);
  EXPECT_GE(std::ranges::count_if(
                decoded, [](const auto &item) { return item->mnemonic() == "v_cndmask_b32"; }),
            6);
}

TEST(BinaryTranslatorE2E, Gfx1250EmulatesLiteralCvtPkFp8F32ClampSetForA0) {
  constexpr auto source_cvt = cdna5::build_vop3(
      cdna5::kVCvtPkFp8F32Vop3, {.vdst = 30, .clamp = 1, .src0 = 255, .src1 = 256 + 2});
  constexpr uint32_t kLiteral = 0x3F800000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_cvt[0], source_cvt[1], kLiteral, kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &item) { return item->mnemonic() == "v_cvt_pk_fp8_f32"; }),
            0);
}

TEST(BinaryTranslatorE2E, Gfx1250MaterializedE5m3LiteralIgnoresTheGuestSourceBank) {
  // Src0 is a literal and its guest bank is 1; src1, src2 and the destination
  // stay in bank 0. The literal is materialized into a low-bank scratch VGPR,
  // so no helper may run under a src-bank-1 mode (0x04) -- doing so would read
  // a different physical register than the one the v_mov just wrote.
  constexpr auto set_vgpr_msb = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x01});
  constexpr auto source_cvt = cdna5::build_vop3(
      cdna5::kVCvtPkFp8F32Vop3, {.vdst = 30, .clamp = 1, .src0 = 255, .src1 = 256 + 2});
  constexpr uint32_t kLiteral = 0x3F800000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {set_vgpr_msb[0], source_cvt[0], source_cvt[1], kLiteral, kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "v_cvt_pk_fp8_f32"; }),
            0);
  std::vector<uint8_t> modes;
  for (const auto &inst : decoded) {
    if (inst->mnemonic() == "s_set_vgpr_msb") {
      ASSERT_NE(inst->raw_encoding(), nullptr);
      modes.push_back(static_cast<uint8_t>(inst->raw_encoding()[0] & 0xffu));
    }
  }
  ASSERT_FALSE(modes.empty());
  EXPECT_EQ(std::ranges::find(modes, 0x04u), modes.end())
      << "materialized literal must be read from the low bank";
  EXPECT_EQ(modes.back(), 0x01u) << "expansion must restore the incoming VGPR-MSB mode";
}

TEST(BinaryTranslatorE2E, Gfx1250EmulatesCvtSrFp8F32ClampSetForA0) {
  constexpr auto source_cvt =
      cdna5::build_vop3(cdna5::kVCvtSrFp8F32Vop3,
                        {.vdst = 30, .opsel = 12, .clamp = 1, .src0 = 256 + 22, .src1 = 256 + 2});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_cvt[0], source_cvt[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &item) { return item->mnemonic() == "v_cvt_sr_fp8_f32"; }),
            0);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &item) { return item->mnemonic() == "v_cvt_f16_f32"; }),
            0);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &item) { return item->mnemonic() == "s_getreg_b32"; }),
            1);
}

/// @brief What executing one translated E5M3 replacement left behind.
struct Gfx1250E5m3Execution {
  uint32_t vdst = 0;          ///< Destination VGPR of the operand's own bank.
  uint8_t final_msb_mode = 0; ///< VGPR-MSB mode the replacement left in effect.
};

/// @brief One source of a conversion under test, and where its value lives.
///
/// @details The expansions materialize a literal into scratch and read a scalar
/// through a different operand path than a VGPR, so each kind reaches different
/// code. VOP3 carries at most one literal, so at most one source may be one.
struct Gfx1250E5m3Operand {
  enum class Kind : uint8_t { Vgpr, Sgpr, Literal };
  Kind kind = Kind::Vgpr;
  uint32_t value = 0;
};

constexpr Gfx1250E5m3Operand e5m3_vgpr(uint32_t value) {
  return {Gfx1250E5m3Operand::Kind::Vgpr, value};
}
constexpr Gfx1250E5m3Operand e5m3_sgpr(uint32_t value) {
  return {Gfx1250E5m3Operand::Kind::Sgpr, value};
}
constexpr Gfx1250E5m3Operand e5m3_literal(uint32_t value) {
  return {Gfx1250E5m3Operand::Kind::Literal, value};
}

/// @brief Run one translated gfx1250 E5M3 replacement on the emulated CU.
///
/// @details The B0 conversion is translated on its own and every instruction of
/// the A0 replacement then executes in order on a single-lane wavefront. That
/// makes the emitted sequence comparable against the execution model's own
/// conversion helpers, which mnemonic-counting tests cannot check.
///
/// @param vgpr_msb_mode Incoming S_SET_VGPR_MSB layout. Each operand is read and
///        written through its own role's bank, so a lowering that forwards the
///        wrong role's bank addresses a register the test never wrote.
std::optional<Gfx1250E5m3Execution>
run_gfx1250_e5m3_replacement(uint16_t opcode, uint8_t opsel, Gfx1250E5m3Operand src0,
                             Gfx1250E5m3Operand src1, uint32_t vdst_initial, bool fp16_ovfl,
                             uint8_t vgpr_msb_mode = 0, uint16_t live_sgprs = 0) {
  constexpr uint16_t kSrc0Vgpr = 22;
  constexpr uint16_t kSrc1Vgpr = 2;
  constexpr uint16_t kDstVgpr = 30;
  constexpr uint16_t kSrc0Sgpr = 8;
  constexpr uint16_t kSrc1Sgpr = 9;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint16_t kM0Operand = 125;
  using Kind = Gfx1250E5m3Operand::Kind;
  if (src0.kind == Kind::Literal && src1.kind == Kind::Literal) {
    ADD_FAILURE() << "VOP3 carries at most one literal";
    return std::nullopt;
  }
  const uint32_t src0_bank = vgpr_msb_mode & 0x3u;
  const uint32_t src1_bank = (vgpr_msb_mode >> 2) & 0x3u;
  const uint32_t dst_bank = (vgpr_msb_mode >> 6) & 0x3u;
  const auto selector = [](Gfx1250E5m3Operand operand, uint16_t vgpr, uint16_t sgpr) -> uint16_t {
    switch (operand.kind) {
    case Kind::Vgpr:
      return static_cast<uint16_t>(256 + vgpr);
    case Kind::Sgpr:
      return sgpr;
    case Kind::Literal:
      break;
    }
    return 255;
  };

  std::vector<uint32_t> source_words;
  if (vgpr_msb_mode != 0) {
    source_words.push_back(
        cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = vgpr_msb_mode})[0]);
  }
  const auto conversion = cdna5::build_vop3(opcode, {.vdst = kDstVgpr,
                                                     .opsel = opsel,
                                                     .clamp = 1,
                                                     .src0 = selector(src0, kSrc0Vgpr, kSrc0Sgpr),
                                                     .src1 = selector(src1, kSrc1Vgpr, kSrc1Sgpr)});
  source_words.push_back(conversion[0]);
  source_words.push_back(conversion[1]);
  if (src0.kind == Kind::Literal)
    source_words.push_back(src0.value);
  else if (src1.kind == Kind::Literal)
    source_words.push_back(src1.value);
  // Readers that keep every ordinary SGPR live, so the expansion has to borrow
  // VGPR carriers and guard them instead of taking scratch SGPRs outright.
  for (uint16_t sgpr = 0; sgpr < live_sgprs; ++sgpr) {
    source_words.push_back(cdna5::build_sop1(
        cdna5::kSMovB32Sop1, {.ssrc0 = static_cast<uint8_t>(sgpr), .sdst = kM0Operand})[0]);
  }
  source_words.push_back(kGfx1250SEndpgm);

  auto image =
      rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(source_words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  if (!result.ok()) {
    ADD_FAILURE() << (result.diagnostics.empty() ? "" : result.diagnostics.front().message);
    return std::nullopt;
  }
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  if (translated.text_sections().empty()) {
    ADD_FAILURE() << "translated code object has no .text";
    return std::nullopt;
  }
  const auto *text = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  rocjitsu::amdgpu::GpuMemory gpu_mem("gfx1250_e5m3_mem");
  rocjitsu::amdgpu::L2Cache l2("gfx1250_e5m3_l2");
  rocjitsu::amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  // Every bank must be addressable: a role's two-bit selector resolves an
  // operand to reg + 256 * selector, so the file has to span all four.
  cfg.vgprs_per_wf = 1024;
  cfg.lds_size_kb = 64;
  auto cu = rocjitsu::amdgpu::ComputeUnitCore::create("gfx1250_e5m3", cfg, &gpu_mem, &l2);
  if (cu == nullptr) {
    ADD_FAILURE() << "could not create a gfx1250 compute unit";
    return std::nullopt;
  }
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  if (wf == nullptr) {
    ADD_FAILURE() << "could not dispatch a wavefront";
    return std::nullopt;
  }
  wf->set_exec(0x1u);
  wf->set_mode_raw(fp16_ovfl ? rocjitsu::amdgpu::Wavefront::FP16_OVFL_BIT : 0u);
  const uint32_t vb = wf->vgpr_alloc().base;
  const uint32_t sb = wf->sgpr_alloc().base;
  const auto place = [&](Gfx1250E5m3Operand operand, uint32_t bank, uint16_t vgpr, uint16_t sgpr) {
    if (operand.kind == Kind::Vgpr)
      cu->write_vgpr(vb + (bank << 8) + vgpr, 0, operand.value);
    else if (operand.kind == Kind::Sgpr)
      cu->write_sgpr(sb + sgpr, operand.value);
  };
  place(src0, src0_bank, kSrc0Vgpr, kSrc0Sgpr);
  place(src1, src1_bank, kSrc1Vgpr, kSrc1Sgpr);
  cu->write_vgpr(vb + (dst_bank << 8) + kDstVgpr, 0, vdst_initial);

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  if (!decoder) {
    ADD_FAILURE() << "no gfx1250 decoder";
    return std::nullopt;
  }
  const std::string_view source_mnemonic =
      opcode == cdna5::kVCvtPkFp8F32Vop3 ? "v_cvt_pk_fp8_f32" : "v_cvt_sr_fp8_f32";
  bool borrowed_a_carrier = false;
  for (size_t index = 0; index < word_count;) {
    std::unique_ptr<rocjitsu::Instruction> inst(decode_valid(*decoder, text + index));
    if (inst == nullptr) {
      ADD_FAILURE() << "undecodable replacement word at " << index;
      return std::nullopt;
    }
    if (std::string_view(inst->mnemonic()) == "s_endpgm")
      break;
    // Executing the untranslated conversion would compare the model against
    // itself and pass no matter what the expansion emits.
    if (std::string_view(inst->mnemonic()) == source_mnemonic) {
      ADD_FAILURE() << "the source conversion survived translation";
      return std::nullopt;
    }
    borrowed_a_carrier =
        borrowed_a_carrier || std::string_view(inst->mnemonic()) == "v_readfirstlane_b32_e32";
    cu->execute_instruction(inst.get(), *wf);
    index += static_cast<size_t>(inst->size()) / sizeof(uint32_t);
  }
  // A caller that walled off the SGPRs asked for the carrier path. Without this
  // the shape could quietly degenerate into the ordinary one and still compare
  // equal, testing nothing it was written for.
  if (live_sgprs != 0 && !borrowed_a_carrier) {
    ADD_FAILURE() << "the SGPR-pressure shape did not borrow a VGPR carrier";
    return std::nullopt;
  }
  return Gfx1250E5m3Execution{cu->read_vgpr(vb + (dst_bank << 8) + kDstVgpr, 0),
                              wf->vgpr_msb_mode()};
}

TEST(BinaryTranslatorE2E, Gfx1250E5m3PackReplacementMatchesReferenceConversion) {
  // Deliberately spans the encoding's discontinuities: signed zeros, the tiny
  // and subnormal quanta, the normal boundary, a rounding tie, max finite, the
  // rounded and exponent overflow cases, infinity and NaN.
  static constexpr std::array<uint32_t, 19> kValues = {
      0x00000000u, 0x80000000u, 0x00000001u, 0x36800000u, 0x37000000u, 0x37800000u, 0x38800000u,
      0x3F800000u, 0x3FC00000u, 0x3F900000u, 0x47C00000u, 0x47D00000u, 0x47E00000u, 0x47E80000u,
      0x47F00000u, 0x47F80000u, 0x7F000000u, 0x7F800000u, 0x7FC00000u};
  constexpr uint32_t kDstInitial = 0xa5a5a5a5u;

  for (const bool fp16_ovfl : {false, true}) {
    for (const bool write_high : {false, true}) {
      for (size_t index = 0; index < kValues.size(); ++index) {
        const uint32_t src0 = kValues[index];
        const uint32_t src1 = kValues[(index + 5) % kValues.size()];
        const auto produced = run_gfx1250_e5m3_replacement(
            cdna5::kVCvtPkFp8F32Vop3, static_cast<uint8_t>(write_high ? 8 : 0), e5m3_vgpr(src0),
            e5m3_vgpr(src1), kDstInitial, fp16_ovfl);
        ASSERT_TRUE(produced.has_value());
        const uint32_t packed =
            util::f32_to_fp8_e5m3_rne_mode(std::bit_cast<float>(src0), fp16_ovfl) |
            (static_cast<uint32_t>(
                 util::f32_to_fp8_e5m3_rne_mode(std::bit_cast<float>(src1), fp16_ovfl))
             << 8);
        const uint32_t expected = write_high ? ((kDstInitial & 0x0000ffffu) | (packed << 16))
                                             : ((kDstInitial & 0xffff0000u) | packed);
        EXPECT_EQ(produced->vdst, expected)
            << "src0=" << std::hex << src0 << " src1=" << src1 << " fp16_ovfl=" << std::dec
            << fp16_ovfl << " write_high=" << write_high;
      }
    }
  }
}

TEST(BinaryTranslatorE2E, Gfx1250E5m3StochasticReplacementMatchesReferenceConversion) {
  static constexpr std::array<uint32_t, 14> kValues = {
      0x00000000u, 0x80000000u, 0x00000001u, 0x36800000u, 0x37000000u, 0x38800000u, 0x3F800000u,
      0x3FC00000u, 0x47D00000u, 0x47E00000u, 0x47F00000u, 0x47F80000u, 0x7F800000u, 0x7FC00000u};
  // Both seed extremes plus one interior value: stochastic rounding must round
  // down at seed 0 and up at UINT32_MAX for every representable magnitude.
  static constexpr std::array<uint32_t, 3> kSeeds = {0u, 0x0f0f0f0fu, 0xffffffffu};
  constexpr uint32_t kDstInitial = 0xa5a5a5a5u;

  for (const bool fp16_ovfl : {false, true}) {
    for (uint8_t dst_byte = 0; dst_byte < 4; ++dst_byte) {
      for (const uint32_t seed : kSeeds) {
        for (const uint32_t value : kValues) {
          const auto produced = run_gfx1250_e5m3_replacement(
              cdna5::kVCvtSrFp8F32Vop3, static_cast<uint8_t>(dst_byte << 2), e5m3_vgpr(value),
              e5m3_vgpr(seed), kDstInitial, fp16_ovfl);
          ASSERT_TRUE(produced.has_value());
          const uint32_t byte =
              util::f32_to_fp8_e5m3_sr_mode(std::bit_cast<float>(value), seed, fp16_ovfl);
          const uint32_t shift = static_cast<uint32_t>(dst_byte) * 8u;
          const uint32_t expected = (kDstInitial & ~(0xffu << shift)) | (byte << shift);
          EXPECT_EQ(produced->vdst, expected)
              << "value=" << std::hex << value << " seed=" << seed << std::dec
              << " fp16_ovfl=" << fp16_ovfl << " dst_byte=" << static_cast<unsigned>(dst_byte);
        }
      }
    }
  }
}

TEST(BinaryTranslatorE2E, Gfx1250E5m3ReplacementsHonorPerRoleVgprBanks) {
  // Every mode gives src0, src1 and the destination three different banks, so a
  // lowering that forwards one role's selector in place of another's resolves to
  // a register nothing wrote rather than one that happens to hold the answer.
  // Rotating through them puts each role on each of selectors 1, 2 and 3, which
  // is what catches a selector whose high bit is dropped.
  static constexpr std::array<uint8_t, 3> modes = {
      0xC9u, // src0 bank 1, src1 bank 2, destination bank 3.
      0x87u, // src0 bank 3, src1 bank 1, destination bank 2.
      0x4Eu, // src0 bank 2, src1 bank 3, destination bank 1.
  };
  static constexpr std::array<uint32_t, 6> kValues = {0x00000000u, 0x37000000u, 0x3F800000u,
                                                      0x47E00000u, 0x47F80000u, 0x7FC00000u};
  constexpr uint32_t kDstInitial = 0x5a5a5a5au;

  for (const uint8_t mode : modes) {
    for (const bool fp16_ovfl : {false, true}) {
      for (size_t index = 0; index < kValues.size(); ++index) {
        const uint32_t src0 = kValues[index];
        const uint32_t src1 = kValues[(index + 2) % kValues.size()];

        const auto packed_run =
            run_gfx1250_e5m3_replacement(cdna5::kVCvtPkFp8F32Vop3, 0, e5m3_vgpr(src0),
                                         e5m3_vgpr(src1), kDstInitial, fp16_ovfl, mode);
        ASSERT_TRUE(packed_run.has_value());
        const uint32_t packed =
            util::f32_to_fp8_e5m3_rne_mode(std::bit_cast<float>(src0), fp16_ovfl) |
            (static_cast<uint32_t>(
                 util::f32_to_fp8_e5m3_rne_mode(std::bit_cast<float>(src1), fp16_ovfl))
             << 8);
        EXPECT_EQ(packed_run->vdst, (kDstInitial & 0xffff0000u) | packed)
            << "src0=" << std::hex << src0 << " src1=" << src1 << std::dec
            << " fp16_ovfl=" << fp16_ovfl << " mode=" << static_cast<unsigned>(mode);
        EXPECT_EQ(packed_run->final_msb_mode, mode) << "the incoming mode must be restored";

        const auto stochastic_run =
            run_gfx1250_e5m3_replacement(cdna5::kVCvtSrFp8F32Vop3, 2u << 2, e5m3_vgpr(src0),
                                         e5m3_vgpr(src1), kDstInitial, fp16_ovfl, mode);
        ASSERT_TRUE(stochastic_run.has_value());
        const uint32_t byte =
            util::f32_to_fp8_e5m3_sr_mode(std::bit_cast<float>(src0), src1, fp16_ovfl);
        EXPECT_EQ(stochastic_run->vdst, (kDstInitial & ~(0xffu << 16)) | (byte << 16))
            << "value=" << std::hex << src0 << " seed=" << src1 << std::dec
            << " fp16_ovfl=" << fp16_ovfl << " mode=" << static_cast<unsigned>(mode);
        EXPECT_EQ(stochastic_run->final_msb_mode, mode) << "the incoming mode must be restored";
      }
    }
  }
}

TEST(BinaryTranslatorE2E, Gfx1250E5m3ReplacementsMatchForNonVgprSources) {
  // Each source kind reaches different code: a literal is materialized into
  // scratch first, a scalar is read through the operand's scalar path, and the
  // SGPR-pressure shape forces VGPR carriers behind an execz guard. All three
  // must still produce what the reference conversion does.
  static constexpr std::array<uint32_t, 5> kValues = {0x00000000u, 0x37000000u, 0x3F800000u,
                                                      0x47E00000u, 0x7FC00000u};
  constexpr uint32_t kDstInitial = 0xa5a5a5a5u;
  constexpr uint32_t kSeed = 0x0f0f0f0fu;

  const auto expect_packed = [&](const std::optional<Gfx1250E5m3Execution> &run, uint32_t src0,
                                 uint32_t src1, bool fp16_ovfl, const char *shape) {
    ASSERT_TRUE(run.has_value()) << shape;
    const uint32_t packed = util::f32_to_fp8_e5m3_rne_mode(std::bit_cast<float>(src0), fp16_ovfl) |
                            (static_cast<uint32_t>(util::f32_to_fp8_e5m3_rne_mode(
                                 std::bit_cast<float>(src1), fp16_ovfl))
                             << 8);
    EXPECT_EQ(run->vdst, (kDstInitial & 0xffff0000u) | packed)
        << shape << " src0=" << std::hex << src0 << " src1=" << src1 << std::dec
        << " fp16_ovfl=" << fp16_ovfl;
  };

  for (const bool fp16_ovfl : {false, true}) {
    for (size_t index = 0; index < kValues.size(); ++index) {
      const uint32_t src0 = kValues[index];
      const uint32_t src1 = kValues[(index + 3) % kValues.size()];

      expect_packed(run_gfx1250_e5m3_replacement(cdna5::kVCvtPkFp8F32Vop3, 0, e5m3_literal(src0),
                                                 e5m3_vgpr(src1), kDstInitial, fp16_ovfl),
                    src0, src1, fp16_ovfl, "literal src0");
      expect_packed(run_gfx1250_e5m3_replacement(cdna5::kVCvtPkFp8F32Vop3, 0, e5m3_vgpr(src0),
                                                 e5m3_literal(src1), kDstInitial, fp16_ovfl),
                    src0, src1, fp16_ovfl, "literal src1");
      expect_packed(run_gfx1250_e5m3_replacement(cdna5::kVCvtPkFp8F32Vop3, 0, e5m3_sgpr(src0),
                                                 e5m3_vgpr(src1), kDstInitial, fp16_ovfl),
                    src0, src1, fp16_ovfl, "scalar src0");
      expect_packed(run_gfx1250_e5m3_replacement(cdna5::kVCvtPkFp8F32Vop3, 0, e5m3_vgpr(src0),
                                                 e5m3_sgpr(src1), kDstInitial, fp16_ovfl),
                    src0, src1, fp16_ovfl, "scalar src1");
      expect_packed(run_gfx1250_e5m3_replacement(cdna5::kVCvtPkFp8F32Vop3, 0, e5m3_vgpr(src0),
                                                 e5m3_vgpr(src1), kDstInitial, fp16_ovfl, 0,
                                                 rocjitsu::REGISTER_SET_MAX_SGPRS),
                    src0, src1, fp16_ovfl, "sgpr pressure");

      // The stochastic lowering reads the seed through the same operand paths.
      for (const auto &[seed, shape] :
           std::initializer_list<std::pair<Gfx1250E5m3Operand, const char *>>{
               {e5m3_literal(kSeed), "literal seed"},
               {e5m3_sgpr(kSeed), "scalar seed"},
               {e5m3_vgpr(kSeed), "sgpr pressure seed"}}) {
        const uint16_t live_sgprs =
            std::string_view(shape) == "sgpr pressure seed" ? rocjitsu::REGISTER_SET_MAX_SGPRS : 0;
        const auto run = run_gfx1250_e5m3_replacement(cdna5::kVCvtSrFp8F32Vop3, 0, e5m3_vgpr(src0),
                                                      seed, kDstInitial, fp16_ovfl, 0, live_sgprs);
        ASSERT_TRUE(run.has_value()) << shape;
        const uint32_t byte =
            util::f32_to_fp8_e5m3_sr_mode(std::bit_cast<float>(src0), kSeed, fp16_ovfl);
        EXPECT_EQ(run->vdst, (kDstInitial & 0xffffff00u) | byte)
            << shape << " value=" << std::hex << src0 << std::dec << " fp16_ovfl=" << fp16_ovfl;
      }
    }
  }
}

TEST(BinaryTranslatorE2E, Gfx1250E5m3PackRejectsDpp) {
  constexpr uint32_t kEndpgm = 0xBFB00000u;
  for (const uint16_t opcode : {cdna5::kVCvtPkFp8F32Vop3, cdna5::kVCvtSrFp8F32Vop3}) {
    cdna5::Vop3VopDpp16MachineInst dpp{};
    dpp.vdst = 30;
    dpp.clamp = 1;
    dpp.op = opcode;
    dpp.encoding = 0x35;
    dpp.src0 = 250;
    dpp.src1 = 256 + 2;
    dpp.vsrc0 = 22;
    dpp.fi = 1;
    dpp.bank_mask = 0xf;
    dpp.row_mask = 0xf;
    std::array<uint32_t, 3> words{};
    std::memcpy(words.data(), &dpp, sizeof(dpp));
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {words[0], words[1], words[2], kEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);
    ASSERT_FALSE(result.ok());
    ASSERT_FALSE(result.diagnostics.empty());
    EXPECT_NE(result.diagnostics.front().message.find("does not support DPP"), std::string::npos);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250E5m3PackRejectsLiteral64) {
  constexpr uint32_t kEndpgm = 0xBFB00000u;
  constexpr uint32_t kLiteralLow = 0x3F800000u;
  // The gfx1250 VOP3 decoder reports only the first literal word, so the second
  // one is decoded on its own. Keep it an s_nop rather than undecodable bytes,
  // which would fail the kernel before the expansion rule is consulted.
  constexpr uint32_t kLiteralHigh = 0xBF800000u;
  for (const uint16_t opcode : {cdna5::kVCvtPkFp8F32Vop3, cdna5::kVCvtSrFp8F32Vop3}) {
    for (const bool literal_in_src0 : {true, false}) {
      const auto conversion = cdna5::build_vop3(
          opcode, {.vdst = 30,
                   .clamp = 1,
                   .src0 = static_cast<uint16_t>(literal_in_src0 ? 254 : 256 + 22),
                   .src1 = static_cast<uint16_t>(literal_in_src0 ? 256 + 2 : 254)});
      auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
          {conversion[0], conversion[1], kLiteralLow, kLiteralHigh, kEndpgm});
      rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
      rocjitsu::BinaryTranslator translator(
          ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
          gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                   rocjitsu::ProcessorRevision::Gfx1250A0));
      const auto result = translator.translate(source);
      ASSERT_FALSE(result.ok());
      EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
          result, rocjitsu::DiagnosticKind::Legalization, "does not support 64-bit literals"));
    }
  }
}

TEST(BinaryTranslatorE2E, Gfx1250E5m3PackUsesCarriersUnderSgprPressure) {
  for (const uint16_t opcode : {cdna5::kVCvtPkFp8F32Vop3, cdna5::kVCvtSrFp8F32Vop3}) {
    const auto conversion =
        cdna5::build_vop3(opcode, {.vdst = 30, .clamp = 1, .src0 = 256 + 22, .src1 = 256 + 23});
    auto image =
        rocjitsu::make_gfx1250_image_with_live_sgprs(conversion, rocjitsu::REGISTER_SET_MAX_SGPRS);
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);
    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    const auto decoded =
        decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
    EXPECT_EQ(std::ranges::count_if(
                  decoded, [](const auto &inst) { return inst->mnemonic() == "s_cbranch_execz"; }),
              1);
    EXPECT_GE(std::ranges::count_if(
                  decoded,
                  [](const auto &inst) { return inst->mnemonic() == "v_readfirstlane_b32_e32"; }),
              4);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250CopiesLiteralCvtF32Fp8E32WithoutClampEmulation) {
  constexpr auto conversion = cdna5::build_vop1(cdna5::kVCvtF32Fp8Vop1, {.src0 = 255, .vdst = 30});
  constexpr uint32_t kLiteral = 0x3F800000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {conversion[0], kLiteral, kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], conversion[0]);
  EXPECT_EQ(target_words[1], kLiteral);
  EXPECT_EQ(target_words[2], kGfx1250SEndpgm);
}

TEST(BinaryTranslatorE2E, Gfx1250EmulatesCvtF32Fp8E5m3ForA0) {
  constexpr auto conversion = cdna5::build_vop3(
      cdna5::kVCvtF32Fp8Vop3, {.vdst = 30, .opsel = 2, .clamp = 1, .src0 = 256 + 22});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {conversion[0], conversion[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "v_cvt_f32_fp8"; }),
            0);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "v_cndmask_b32"; }),
            2);

  // Assert the value-defining literal constants of the emitted sequence, not just
  // mnemonic counts: an off-by-one in the NaN threshold, exponent clamp, or the
  // F16/F32 reconstruction constants would otherwise pass. These constants are
  // distinctive, so scan the raw translated text words for each. This checks the
  // emitted bytes directly without coupling the test to decoded instruction objects.
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto contains_word = [&](uint32_t value) {
    return std::any_of(target_words, target_words + target_word_count,
                       [&](uint32_t w) { return w == value; });
  };
  EXPECT_TRUE(contains_word(0xffu)) << "missing E5M3 NaN threshold literal 0xff";
  EXPECT_TRUE(contains_word(0xf7u)) << "missing E5M3 exponent-clamp literal 0xf7";
  EXPECT_TRUE(contains_word(0x47800000u)) << "missing F16 reconstruction constant";
  EXPECT_TRUE(contains_word(0x7fa3d000u)) << "missing F32 max-finite constant";

  // Pin the byte-select. The source opsel=2 must map through the VOP3
  // byte-select bit swap to byte 1, so the single v_bfe_u32 that isolates the
  // packed byte reads bit offset 1*8=8 (inline-encoded as 128+8=136). Without
  // the swap it would read byte 2 (offset 16, encoded 144) and silently unpack
  // the wrong lane's fp8 datum. Inspect the decoded operand so a byte-select
  // regression cannot pass.
  int bfe_count = 0;
  for (const auto &inst : decoded) {
    if (inst->mnemonic() != "v_bfe_u32")
      continue;
    ++bfe_count;
    const rocjitsu::Operand *offset = inst->src_operand(1);
    ASSERT_NE(offset, nullptr);
    EXPECT_EQ(offset->encoding_value(), 136) << "E5M3 byte-select must be byte 1 (offset 8)";
  }
  EXPECT_EQ(bfe_count, 1) << "E5M3 unpack must isolate the byte with exactly one v_bfe_u32";
}

TEST(BinaryTranslatorE2E, Gfx1250E5m3CompareMasksTargetDeadSgprs) {
  constexpr auto conversion = cdna5::build_vop3(
      cdna5::kVCvtF32Fp8Vop3, {.vdst = 30, .opsel = 2, .clamp = 1, .src0 = 256 + 22});
  // s103 is the first dead SGPR but cannot be borrowed safely because it backs
  // src_flat_scratch_base. The two independent Wave32 masks use s104:s105.
  constexpr uint16_t kLiveSgprs = 103;
  constexpr uint16_t kExpectedMaskBase = 104;
  auto image = rocjitsu::make_gfx1250_image_with_live_sgprs(conversion, kLiveSgprs);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  std::optional<uint16_t> nan_mask;
  std::optional<uint16_t> exp31_mask;
  std::vector<uint16_t> cndmask_predicates;
  for (const auto &inst : decoded) {
    if (inst->mnemonic() == "v_cmp_eq_u32" || inst->mnemonic() == "v_cmp_lt_u32") {
      ASSERT_NE(inst->raw_encoding(), nullptr);
      cdna5::Vop3MachineInst encoded{};
      std::memcpy(&encoded, inst->raw_encoding(), sizeof(encoded));
      if (inst->mnemonic() == "v_cmp_eq_u32")
        nan_mask = static_cast<uint16_t>(encoded.vdst);
      else
        exp31_mask = static_cast<uint16_t>(encoded.vdst);
    } else if (inst->mnemonic() == "v_cndmask_b32") {
      const rocjitsu::Operand *predicate = inst->src_operand(2);
      ASSERT_NE(predicate, nullptr);
      cndmask_predicates.push_back(predicate->encoding_value());
    }
  }
  ASSERT_TRUE(nan_mask.has_value());
  ASSERT_TRUE(exp31_mask.has_value());
  ASSERT_EQ(cndmask_predicates.size(), 2u);
  EXPECT_EQ(*nan_mask, kExpectedMaskBase);
  EXPECT_EQ(*exp31_mask, kExpectedMaskBase + 1u);
  EXPECT_EQ(cndmask_predicates[0], *exp31_mask);
  EXPECT_EQ(cndmask_predicates[1], *nan_mask);
}

TEST(BinaryTranslatorE2E, Gfx1250E5m3UnpackUsesVgprCarriersUnderSgprPressure) {
  constexpr auto conversion = cdna5::build_vop3(
      cdna5::kVCvtF32Fp8Vop3, {.vdst = 30, .opsel = 2, .clamp = 1, .src0 = 256 + 22});
  for (const uint16_t live_sgprs : {static_cast<uint16_t>(rocjitsu::REGISTER_SET_MAX_SGPRS - 1u),
                                    static_cast<uint16_t>(rocjitsu::REGISTER_SET_MAX_SGPRS)}) {
    SCOPED_TRACE(live_sgprs);
    auto image = rocjitsu::make_gfx1250_image_with_live_sgprs(conversion, live_sgprs);
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);

    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);
    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    const auto decoded =
        decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
    EXPECT_EQ(std::ranges::count_if(
                  decoded, [](const auto &inst) { return inst->mnemonic() == "s_cbranch_execz"; }),
              1);
    EXPECT_EQ(std::ranges::count_if(
                  decoded,
                  [](const auto &inst) { return inst->mnemonic() == "v_readfirstlane_b32_e32"; }),
              2);

    const auto guard = std::ranges::find_if(
        decoded, [](const auto &inst) { return inst->mnemonic() == "s_cbranch_execz"; });
    ASSERT_NE(guard, decoded.end());
    ASSERT_TRUE((*guard)->branch_offset_bytes().has_value());
    const uint64_t target =
        (*guard)->src_loc() + (*guard)->size() + *(*guard)->branch_offset_bytes();
    const auto following =
        std::ranges::find_if(decoded, [&](const auto &inst) { return inst->src_loc() == target; });
    ASSERT_NE(following, decoded.end());
    EXPECT_EQ((*following)->mnemonic(), "s_mov_b32")
        << "the EXEC-zero guard must land on the first following guest instruction";
  }
}

TEST(BinaryTranslatorE2E, Gfx1250E5m3SeparatesVectorAndScalarSpillSlots) {
  constexpr auto conversion = cdna5::build_vop3(
      cdna5::kVCvtF32Fp8Vop3, {.vdst = 30, .opsel = 2, .clamp = 1, .src0 = 256 + 22});
  auto image =
      rocjitsu::make_gfx1250_image_with_live_sgprs(conversion, rocjitsu::REGISTER_SET_MAX_SGPRS);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  auto options = gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                          rocjitsu::ProcessorRevision::Gfx1250A0);
  options.debug_min_free_vgpr = 256;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
                                        options);
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  std::vector<uint32_t> store_offsets;
  std::vector<uint32_t> load_offsets;
  for (const auto &inst : decoded) {
    if (inst->mnemonic() != "scratch_store_b32" && inst->mnemonic() != "scratch_load_b32")
      continue;
    ASSERT_NE(inst->raw_encoding(), nullptr);
    cdna5::VscratchMachineInst scratch{};
    std::memcpy(&scratch, inst->raw_encoding(), sizeof(scratch));
    (inst->mnemonic() == "scratch_load_b32" ? load_offsets : store_offsets)
        .push_back(scratch.ioffset);
  }
  EXPECT_EQ(store_offsets, (std::vector<uint32_t>{0, 4, 8, 12}));
  EXPECT_EQ(load_offsets, (std::vector<uint32_t>{8, 12, 0, 4}));

  const auto *rodata = rocjitsu::test_support::find_section(translated, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto descriptor = rocjitsu::test_support::read_kernel_descriptor_for_test(rodata->data());
  EXPECT_GE(descriptor.private_segment_fixed_size, 16u);

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250CopiesWmmaOutsideB0ToA0Profile) {
  constexpr auto source_wmma = cdna5::build_vop3p(
      cdna5::kVWmmaF3216x16x32Bf16Vop3p, {.vdst = 8, .src0 = 256, .src1 = 264, .src2 = 272});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], source_wmma[0]);
  EXPECT_EQ(target_words[1], source_wmma[1]);
  EXPECT_EQ(target_words[2], kGfx1250SEndpgm);
}

TEST(BinaryTranslatorE2E, Gfx1250EqualRevisionsUseIdentityProfile) {
  constexpr auto source_ds2 = cdna5::build_vds(cdna5::kDsLoad2addrB32Vds,
                                               {.offset0 = 1, .offset1 = 3, .addr = 7, .vdst = 9});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_ds2[0], source_ds2[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  for (const rocjitsu::ProcessorRevision revision :
       {rocjitsu::ProcessorRevision::Gfx1250A0, rocjitsu::ProcessorRevision::Gfx1250B0}) {
    SCOPED_TRACE(revision == rocjitsu::ProcessorRevision::Gfx1250A0 ? "A0-to-A0" : "B0-to-B0");
    rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
                                          gfx1250_revision_options(revision, revision));
    auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);

    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_FALSE(translated.text_sections().empty());
    const auto *target_words =
        reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    EXPECT_EQ(target_words[0], source_ds2[0]);
    EXPECT_EQ(target_words[1], source_ds2[1]);
    EXPECT_EQ(target_words[2], kGfx1250SEndpgm);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250RejectsA0ToB0Translation) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image =
      rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text({kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250A0,
                               rocjitsu::ProcessorRevision::Gfx1250B0));
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::Legalization, "A0-to-B0 translation is not supported"));
}

TEST(BinaryTranslatorE2E, Gfx1250ExpandsEveryDs2VariantForA0) {
  struct Ds2Case {
    const char *name;
    uint16_t source_opcode;
    uint16_t target_opcode;
    uint8_t element_dwords;
    bool stride64;
  };
  constexpr std::array cases = {
      Ds2Case{"store_b32", cdna5::kDsStore2addrB32Vds, cdna5::kDsStoreB32Vds, 1, false},
      Ds2Case{"store_stride64_b32", cdna5::kDsStore2addrStride64B32Vds, cdna5::kDsStoreB32Vds, 1,
              true},
      Ds2Case{"storexchg_b32", cdna5::kDsStorexchg2addrRtnB32Vds, cdna5::kDsStorexchgRtnB32Vds, 1,
              false},
      Ds2Case{"storexchg_stride64_b32", cdna5::kDsStorexchg2addrStride64RtnB32Vds,
              cdna5::kDsStorexchgRtnB32Vds, 1, true},
      Ds2Case{"load_b32", cdna5::kDsLoad2addrB32Vds, cdna5::kDsLoadB32Vds, 1, false},
      Ds2Case{"load_stride64_b32", cdna5::kDsLoad2addrStride64B32Vds, cdna5::kDsLoadB32Vds, 1,
              true},
      Ds2Case{"store_b64", cdna5::kDsStore2addrB64Vds, cdna5::kDsStoreB64Vds, 2, false},
      Ds2Case{"store_stride64_b64", cdna5::kDsStore2addrStride64B64Vds, cdna5::kDsStoreB64Vds, 2,
              true},
      Ds2Case{"storexchg_b64", cdna5::kDsStorexchg2addrRtnB64Vds, cdna5::kDsStorexchgRtnB64Vds, 2,
              false},
      Ds2Case{"storexchg_stride64_b64", cdna5::kDsStorexchg2addrStride64RtnB64Vds,
              cdna5::kDsStorexchgRtnB64Vds, 2, true},
      Ds2Case{"load_b64", cdna5::kDsLoad2addrB64Vds, cdna5::kDsLoadB64Vds, 2, false},
      Ds2Case{"load_stride64_b64", cdna5::kDsLoad2addrStride64B64Vds, cdna5::kDsLoadB64Vds, 2,
              true},
  };
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr cdna5::VdsBuilderFields source_fields{
      .offset0 = 3, .offset1 = 5, .addr = 20, .data0 = 30, .data1 = 40, .vdst = 50};

  for (const Ds2Case &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto source_ds2 = cdna5::build_vds(test_case.source_opcode, source_fields);
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {source_ds2[0], source_ds2[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());

    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);

    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(translated.is_valid());
    ASSERT_FALSE(translated.text_sections().empty());
    const auto *target_words =
        reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    ASSERT_GE(translated.text_sections()[0]->size(), 6 * sizeof(uint32_t));

    const uint16_t scale = static_cast<uint16_t>(test_case.element_dwords * sizeof(uint32_t) *
                                                 (test_case.stride64 ? 64 : 1));
    const bool plain_store = test_case.target_opcode == cdna5::kDsStoreB32Vds ||
                             test_case.target_opcode == cdna5::kDsStoreB64Vds;
    const auto first =
        cdna5::build_vds(test_case.target_opcode,
                         {.offset0 = static_cast<uint8_t>((source_fields.offset0 * scale) & 0xff),
                          .offset1 = static_cast<uint8_t>((source_fields.offset0 * scale) >> 8),
                          .addr = source_fields.addr,
                          .data0 = source_fields.data0,
                          .vdst = static_cast<uint8_t>(plain_store ? 0 : source_fields.vdst)});
    const auto second =
        cdna5::build_vds(test_case.target_opcode,
                         {.offset0 = static_cast<uint8_t>((source_fields.offset1 * scale) & 0xff),
                          .offset1 = static_cast<uint8_t>((source_fields.offset1 * scale) >> 8),
                          .addr = source_fields.addr,
                          .data0 = source_fields.data1,
                          .vdst = static_cast<uint8_t>(
                              plain_store ? 0 : source_fields.vdst + test_case.element_dwords)});
    EXPECT_EQ(target_words[0], first[0]);
    EXPECT_EQ(target_words[1], first[1]);
    EXPECT_EQ(target_words[2], second[0]);
    EXPECT_EQ(target_words[3], second[1]);
    EXPECT_EQ(target_words[4], cdna5::build_sopp(cdna5::kSWaitDscntSopp, {.simm16 = 0})[0]);
    EXPECT_EQ(target_words[5], kGfx1250SEndpgm);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250Ds2LoadPreservesAliasedAddressUntilBothLoadsIssue) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  // ADDR=v7 aliases the first destination half v7. The second load must issue
  // first, exactly matching the executable far-DS2 alias oracle.
  constexpr cdna5::VdsBuilderFields fields{.offset0 = 1, .offset1 = 3, .addr = 7, .vdst = 7};
  constexpr auto source_ds2 = cdna5::build_vds(cdna5::kDsLoad2addrB32Vds, fields);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_ds2[0], source_ds2[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());

  constexpr auto second =
      cdna5::build_vds(cdna5::kDsLoadB32Vds, {.offset0 = 12, .addr = fields.addr, .vdst = 8});
  constexpr auto first =
      cdna5::build_vds(cdna5::kDsLoadB32Vds, {.offset0 = 4, .addr = fields.addr, .vdst = 7});
  EXPECT_EQ(target_words[0], second[0]);
  EXPECT_EQ(target_words[1], second[1]);
  EXPECT_EQ(target_words[2], first[0]);
  EXPECT_EQ(target_words[3], first[1]);
}

TEST(BinaryTranslatorE2E, Gfx1250Ds2ExchangeCyclicOverlapFailsClosed) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  // First destination v10 aliases ADDR; second destination v11 aliases DATA0.
  // Neither issue order preserves every compound source operand.
  constexpr auto source_ds2 = cdna5::build_vds(cdna5::kDsStorexchg2addrRtnB32Vds,
                                               {.addr = 10, .data0 = 11, .data1 = 20, .vdst = 10});
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_ds2[0], source_ds2[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  EXPECT_FALSE(result.ok());
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &item) {
    return item.kind == rocjitsu::DiagnosticKind::ExpandFailed;
  });
  ASSERT_NE(diagnostic, result.diagnostics.end());
  EXPECT_NE(diagnostic->message.find("cyclic"), std::string::npos);
}

TEST(BinaryTranslatorE2E, Gfx1250Ds2OversizedStride64OffsetFailsClosed) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr auto source_ds2 =
      cdna5::build_vds(cdna5::kDsLoad2addrStride64B64Vds, {.offset1 = 255, .addr = 4, .vdst = 8});
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_ds2[0], source_ds2[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  EXPECT_FALSE(result.ok());
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &item) {
    return item.kind == rocjitsu::DiagnosticKind::ExpandFailed;
  });
  ASSERT_NE(diagnostic, result.diagnostics.end());
  EXPECT_NE(diagnostic->message.find("16-bit"), std::string::npos);
}

TEST(BinaryTranslatorE2E, Gfx1250Ds2MovesData1VgprBankToSecondData0Role) {
  constexpr auto set_vgpr_msb = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x10});
  constexpr auto source_ds2 = cdna5::build_vds(
      cdna5::kDsStore2addrB32Vds, {.offset0 = 1, .offset1 = 2, .addr = 4, .data0 = 8, .data1 = 12});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {set_vgpr_msb[0], source_ds2[0], source_ds2[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  // The generated bank transitions carry these immediates, and each generated
  // s_set_vgpr_msb must be preceded by an s_wait_xcnt drain. Collect the generated
  // transitions dynamically so the assertion does not depend on absolute offsets.
  std::vector<uint16_t> generated_modes;
  for (size_t index = 0; index < decoded.size(); ++index) {
    if (decoded[index]->mnemonic() != "s_set_vgpr_msb")
      continue;
    ASSERT_NE(decoded[index]->raw_encoding(), nullptr);
    const uint16_t imm = static_cast<uint16_t>(decoded[index]->raw_encoding()[0] & 0xffffu);
    // The 0x0010 transition is the copied source s_set_vgpr_msb; the two generated
    // transitions (0x1014, 0x1410) must each be guarded by s_wait_xcnt.
    if (imm != 0x0010) {
      ASSERT_GT(index, 0u);
      EXPECT_EQ(decoded[index - 1]->mnemonic(), "s_wait_xcnt");
    }
    generated_modes.push_back(imm);
  }
  EXPECT_EQ(generated_modes, (std::vector<uint16_t>{0x0010, 0x1014, 0x1410}));
}

TEST(BinaryTranslatorE2E, Gfx1250Ds2AdjustsDestinationBankForSecondLoad) {
  constexpr auto source_ds2 = cdna5::build_vds(
      cdna5::kDsLoad2addrB64Vds, {.offset0 = 1, .offset1 = 2, .addr = 4, .vdst = 254});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_ds2[0], source_ds2[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  // Two generated bank transitions (0x0040, 0x4000), each guarded by s_wait_xcnt.
  std::vector<uint16_t> generated_modes;
  for (size_t index = 0; index < decoded.size(); ++index) {
    if (decoded[index]->mnemonic() != "s_set_vgpr_msb")
      continue;
    ASSERT_NE(decoded[index]->raw_encoding(), nullptr);
    ASSERT_GT(index, 0u);
    EXPECT_EQ(decoded[index - 1]->mnemonic(), "s_wait_xcnt");
    generated_modes.push_back(static_cast<uint16_t>(decoded[index]->raw_encoding()[0] & 0xffffu));
  }
  EXPECT_EQ(generated_modes, (std::vector<uint16_t>{0x0040, 0x4000}));
}

TEST(SemanticTranslator, Gfx1250ClassifiesLivenessFreeExpandRules) {
  std::vector<uint32_t> liveness_free_rules;
  for (const rocjitsu::TranslationRule &rule : rocjitsu::semantic_expand_rules_gfx1250_b0_to_a0()) {
    if (!rule.requires_liveness) {
      liveness_free_rules.push_back((static_cast<uint32_t>(rule.src_encoding_id) << 16) |
                                    rule.src_opcode);
    }
  }

  const std::vector<uint32_t> expected = {
      // A SOPK encoding id is the SOPK base plus the opcode, so the MODE-write
      // separation rules sort ahead of everything else in the table.
      (static_cast<uint32_t>(cdna5::encoding::kSopk + cdna5::kSSetregB32Sopk) << 16) |
          cdna5::kSSetregB32Sopk,
      (static_cast<uint32_t>(cdna5::encoding::kSopk + cdna5::kSSetregImm32B32Sopk) << 16) |
          cdna5::kSSetregImm32B32Sopk,
      (static_cast<uint32_t>(cdna5::encoding::kSop1) << 16) | cdna5::kSBarrierSignalIsfirstSop1,
      (static_cast<uint32_t>(cdna5::encoding::kSopp) << 16) | cdna5::kSClauseSopp,
      (static_cast<uint32_t>(cdna5::encoding::kVop3p) << 16) | cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
      (static_cast<uint32_t>(cdna5::encoding::kVop3p) << 16) | cdna5::kVWmmaI3216x16x64Iu8Vop3p,
      (static_cast<uint32_t>(cdna5::encoding::kVop3p) << 16) | cdna5::kVSwmmacI3216x16x128Iu8Vop3p,
      (static_cast<uint32_t>(cdna5::encoding::kVop3pOpHi1) << 16) |
          cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p,
      (static_cast<uint32_t>(cdna5::encoding::kVop3pOpHi1) << 16) |
          cdna5::kVWmmaF3216x16x128Fp8Bf8Vop3p,
      (static_cast<uint32_t>(cdna5::encoding::kVop3pOpHi1) << 16) |
          cdna5::kVWmmaF3216x16x128Bf8Fp8Vop3p,
      (static_cast<uint32_t>(cdna5::encoding::kVop3pOpHi1) << 16) |
          cdna5::kVWmmaF3216x16x128Bf8Bf8Vop3p,
  };
  // The standalone 32x16 FP4 and packed-f16 K=128 rules are deliberately
  // absent because they query operand-bank liveness.
  EXPECT_EQ(liveness_free_rules, expected);

  rocjitsu::SemanticTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5,
                                          rocjitsu::ProcessorRevision::Gfx1250B0,
                                          rocjitsu::ProcessorRevision::Gfx1250A0);
  constexpr auto tensor_mask_clear =
      cdna5::build_sop2(cdna5::kSPackHhB32B16Sop2, {.ssrc0 = 128, .ssrc1 = 0, .sdst = 0});
  constexpr auto cluster_m0_clear =
      cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 125});
  constexpr auto v_nop = cdna5::build_vop1(cdna5::kVNopVop1);
  const auto tensor_clear_inst =
      rocjitsu::test_support::decode_one(tensor_mask_clear[0], ROCJITSU_CODE_ARCH_CDNA5);
  const auto cluster_clear_inst =
      rocjitsu::test_support::decode_one(cluster_m0_clear[0], ROCJITSU_CODE_ARCH_CDNA5);
  const auto v_nop_inst = rocjitsu::test_support::decode_one(v_nop[0], ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(tensor_clear_inst, nullptr);
  ASSERT_NE(cluster_clear_inst, nullptr);
  ASSERT_NE(v_nop_inst, nullptr);
  EXPECT_FALSE(translator.has_expand_rule(*tensor_clear_inst));
  EXPECT_FALSE(translator.has_expand_rule(*cluster_clear_inst));
  EXPECT_FALSE(translator.has_expand_rule(*v_nop_inst));
}

TEST(SemanticTranslator, Gfx1250RegistryHasCompleteDischargeContracts) {
  const rocjitsu::RewriteRegistry registry = rocjitsu::rewrite_registry_gfx1250_b0_to_a0();
  EXPECT_TRUE(registry.has_complete_discharge());
  ASSERT_EQ(registry.opcode_rules.size(), 43u);
  ASSERT_EQ(registry.instruction_rules.size(), 1u);

  size_t checked_rules = 0;
  size_t no_success_rules = 0;
  std::vector<uint32_t> no_success_rule_keys;
  std::vector<uint32_t> block_context_rule_keys;
  for (const rocjitsu::TranslationRule &rule : registry.opcode_rules) {
    EXPECT_TRUE(rule.discharge.valid());
    if (rule.discharge.disposition == rocjitsu::RewriteDischargeDisposition::Checked) {
      ++checked_rules;
      if (rule.discharge.context == rocjitsu::RewriteDischargeContext::BasicBlock) {
        block_context_rule_keys.push_back((static_cast<uint32_t>(rule.src_encoding_id) << 16) |
                                          rule.src_opcode);
      }
    }
    if (rule.discharge.disposition ==
        rocjitsu::RewriteDischargeDisposition::NoSuccessfulExpansion) {
      ++no_success_rules;
      no_success_rule_keys.push_back((static_cast<uint32_t>(rule.src_encoding_id) << 16) |
                                     rule.src_opcode);
      EXPECT_FALSE(rule.discharge.allows(rocjitsu::ExpandStatus::Success));
      EXPECT_TRUE(rule.discharge.allows(rocjitsu::ExpandStatus::Failed));
      EXPECT_TRUE(rule.discharge.allows(rocjitsu::ExpandStatus::NotHandled));
    }
  }
  EXPECT_EQ(checked_rules, 42u);
  EXPECT_EQ(no_success_rules, 1u);
  const std::vector<uint32_t> expected_no_success_rule_keys = {
      (static_cast<uint32_t>(cdna5::encoding::kSop1) << 16) | cdna5::kSBarrierSignalIsfirstSop1,
  };
  EXPECT_EQ(no_success_rule_keys, expected_no_success_rule_keys);
  const std::vector<uint32_t> expected_block_context_rule_keys = {
      (static_cast<uint32_t>(cdna5::encoding::kSopk + cdna5::kSSetregB32Sopk) << 16) |
          cdna5::kSSetregB32Sopk,
      (static_cast<uint32_t>(cdna5::encoding::kSopk + cdna5::kSSetregImm32B32Sopk) << 16) |
          cdna5::kSSetregImm32B32Sopk,
      (static_cast<uint32_t>(cdna5::encoding::kVop3p) << 16) | cdna5::kVWmmaI3216x16x64Iu8Vop3p,
      (static_cast<uint32_t>(cdna5::encoding::kVop3p) << 16) | cdna5::kVSwmmacI3216x16x128Iu8Vop3p,
      (static_cast<uint32_t>(cdna5::encoding::kVimage) << 16) | cdna5::kTensorLoadToLdsVimage,
      (static_cast<uint32_t>(cdna5::encoding::kVglobal) << 16) | cdna5::kClusterLoadB32Vglobal,
      (static_cast<uint32_t>(cdna5::encoding::kVglobal) << 16) | cdna5::kClusterLoadB64Vglobal,
      (static_cast<uint32_t>(cdna5::encoding::kVglobal) << 16) | cdna5::kClusterLoadB128Vglobal,
      (static_cast<uint32_t>(cdna5::encoding::kVglobal) << 16) |
          cdna5::kClusterLoadAsyncToLdsB8Vglobal,
      (static_cast<uint32_t>(cdna5::encoding::kVglobal) << 16) |
          cdna5::kClusterLoadAsyncToLdsB32Vglobal,
      (static_cast<uint32_t>(cdna5::encoding::kVglobal) << 16) |
          cdna5::kClusterLoadAsyncToLdsB64Vglobal,
      (static_cast<uint32_t>(cdna5::encoding::kVglobal) << 16) |
          cdna5::kClusterLoadAsyncToLdsB128Vglobal,
  };
  EXPECT_EQ(block_context_rule_keys, expected_block_context_rule_keys);

  const std::array out_of_order_rules = {registry.opcode_rules[1], registry.opcode_rules[0]};
  const rocjitsu::RewriteRegistry out_of_order_registry{out_of_order_rules, {}};
  EXPECT_FALSE(out_of_order_registry.has_complete_discharge());
  const std::array duplicate_rules = {registry.opcode_rules[0], registry.opcode_rules[0]};
  const rocjitsu::RewriteRegistry duplicate_registry{duplicate_rules, {}};
  EXPECT_FALSE(duplicate_registry.has_complete_discharge());

  const rocjitsu::RegisteredInstructionRewrite divergent_selector_rule{
      "divergent-selector",
      +[](const rocjitsu::Instruction &) { return false; },
      +[](const rocjitsu::Instruction &, uint64_t, std::span<const uint8_t>,
          const rocjitsu::LivenessAnalysis &,
          rocjitsu::TranslationContext &) { return rocjitsu::ExpandResult::not_handled(); },
      false,
      rocjitsu::RewriteDischarge::checked(
          +[](const rocjitsu::Instruction &) { return true; },
          rocjitsu::RewriteDischargeContext::BasicBlock),
  };
  const std::array divergent_selector_rules = {divergent_selector_rule};
  const rocjitsu::RewriteRegistry divergent_selector_registry{{}, divergent_selector_rules};
  EXPECT_TRUE(divergent_selector_registry.has_complete_discharge());
  EXPECT_TRUE(divergent_selector_registry.instruction_rewrites_require_basic_block())
      << "CFG fallback must not depend on the lowering selector";

  const rocjitsu::RegisteredInstructionRewrite &flat_scratch = registry.instruction_rules.front();
  EXPECT_STREQ(flat_scratch.name, "flat-scratch-base-64bit-source");
  EXPECT_TRUE(flat_scratch.requires_liveness);
  EXPECT_TRUE(flat_scratch.valid());
  EXPECT_EQ(flat_scratch.discharge.disposition, rocjitsu::RewriteDischargeDisposition::Checked);
  EXPECT_EQ(flat_scratch.discharge.context, rocjitsu::RewriteDischargeContext::Instruction);

  rocjitsu::SemanticTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5,
                                          rocjitsu::ProcessorRevision::Gfx1250B0,
                                          rocjitsu::ProcessorRevision::Gfx1250A0);
  EXPECT_TRUE(translator.supports_rewrite_discharge());
  rocjitsu::SemanticTranslator legacy_translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                                                 rocjitsu::ProcessorRevision::Unspecified,
                                                 rocjitsu::ProcessorRevision::Unspecified);
  EXPECT_FALSE(legacy_translator.supports_rewrite_discharge());
  constexpr auto selector_230 = cdna5::build_sop1(cdna5::kSMovB64Sop1, {.ssrc0 = 230, .sdst = 10});
  constexpr auto selector_231 = cdna5::build_sop1(cdna5::kSMovB64Sop1, {.ssrc0 = 231, .sdst = 10});
  const auto low = rocjitsu::test_support::decode_one(selector_230[0], ROCJITSU_CODE_ARCH_CDNA5);
  const auto high = rocjitsu::test_support::decode_one(selector_231[0], ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(low, nullptr);
  ASSERT_NE(high, nullptr);
  EXPECT_TRUE(translator.has_instruction_rewrite(*low));
  EXPECT_TRUE(translator.has_instruction_rewrite(*high));
  EXPECT_TRUE(translator.rewrite_requires_liveness(*low));
  EXPECT_TRUE(translator.rewrite_requires_liveness(*high));
  EXPECT_FALSE(translator.residual_rewrite_applies(*low));
  EXPECT_TRUE(translator.residual_rewrite_applies(*high));
  EXPECT_FALSE(translator.residual_rewrite_needs_basic_block(*high));
  EXPECT_TRUE(translator.instruction_local_residual_rewrite_applies(*high));
}

TEST(SemanticTranslator, Gfx1250ResidualChecksRecognizeEveryActionableSourceRule) {
  std::vector<std::vector<uint32_t>> samples;
  const auto add_sample = [&](const auto &words) {
    samples.emplace_back(words.begin(), words.end());
  };
  const auto add_compound_sample = [&](const auto &prefix, const auto &body) {
    std::vector<uint32_t> words(prefix.begin(), prefix.end());
    words.insert(words.end(), body.begin(), body.end());
    samples.push_back(std::move(words));
  };

  constexpr uint16_t kModeHwreg = 1u;
  add_sample(cdna5::build_sopk(cdna5::kSSetregB32Sopk, {.simm16 = kModeHwreg, .sdst = 8}));
  constexpr std::array<uint32_t, 1> setreg_literal = {0};
  add_compound_sample(cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = kModeHwreg}),
                      setreg_literal);
  add_sample(cdna5::build_sopp(cdna5::kSClauseSopp, {.simm16 = 4}));
  add_sample(cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
                                {.vdst = 8, .src0 = 256, .src1 = 264, .src2 = 272}));

  constexpr auto regular_scale =
      cdna5::build_vop3p(0x35, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 256});
  constexpr auto scale16 =
      cdna5::build_vop3p(0x3a, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 256});
  constexpr auto scale_body =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                         {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48});
  add_compound_sample(regular_scale, scale_body);
  add_compound_sample(scale16, scale_body);

  add_sample(cdna5::build_vop3p(cdna5::kVWmmaI3216x16x64Iu8Vop3p,
                                {.vdst = 8, .src0 = 256, .src1 = 264, .src2 = 272}));
  add_sample(cdna5::build_vop3p(cdna5::kVSwmmacI3216x16x128Iu8Vop3p,
                                {.vdst = 8, .src0 = 256, .src1 = 264, .src2 = 272}));

  for (const uint16_t opcode : {
           cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p,
           cdna5::kVWmmaF3216x16x128Fp8Bf8Vop3p,
           cdna5::kVWmmaF3216x16x128Bf8Fp8Vop3p,
           cdna5::kVWmmaF3216x16x128Bf8Bf8Vop3p,
       }) {
    add_sample(cdna5::build_vop3p(opcode, {.vdst = 8, .src0 = 256, .src1 = 264, .src2 = 272}));
  }
  for (const uint16_t opcode : {
           cdna5::kVWmmaF1616x16x128Fp8Fp8Vop3p,
           cdna5::kVWmmaF1616x16x128Fp8Bf8Vop3p,
           cdna5::kVWmmaF1616x16x128Bf8Fp8Vop3p,
           cdna5::kVWmmaF1616x16x128Bf8Bf8Vop3p,
       }) {
    add_sample(cdna5::build_vop3p(opcode, {.vdst = 8, .src0 = 256, .src1 = 264, .src2 = 272}));
  }
  add_sample(
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                         {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 40, .src2 = 256 + 64}));

  add_sample(
      cdna5::build_vimage(cdna5::kTensorLoadToLdsVimage,
                          {.vaddr4 = 124, .vaddr0 = 8, .vaddr1 = 0, .vaddr2 = 124, .vaddr3 = 124}));
  add_sample(cdna5::build_vop3(cdna5::kVCvtF32Fp8Vop3, {.vdst = 30, .clamp = 1, .src0 = 256 + 22}));
  add_sample(cdna5::build_vop3(cdna5::kVCvtPkFp8F32Vop3,
                               {.vdst = 30, .clamp = 1, .src0 = 256 + 22, .src1 = 256 + 2}));
  add_sample(cdna5::build_vop3(cdna5::kVCvtSrFp8F32Vop3,
                               {.vdst = 30, .clamp = 1, .src0 = 256 + 22, .src1 = 256 + 2}));

  constexpr cdna5::VdsBuilderFields ds2_fields{
      .offset0 = 3, .offset1 = 5, .addr = 20, .data0 = 30, .data1 = 40, .vdst = 50};
  for (const uint16_t opcode : {
           cdna5::kDsStore2addrB32Vds,
           cdna5::kDsStore2addrStride64B32Vds,
           cdna5::kDsStorexchg2addrRtnB32Vds,
           cdna5::kDsStorexchg2addrStride64RtnB32Vds,
           cdna5::kDsLoad2addrB32Vds,
           cdna5::kDsLoad2addrStride64B32Vds,
           cdna5::kDsStore2addrB64Vds,
           cdna5::kDsStore2addrStride64B64Vds,
           cdna5::kDsStorexchg2addrRtnB64Vds,
           cdna5::kDsStorexchg2addrStride64RtnB64Vds,
           cdna5::kDsLoad2addrB64Vds,
           cdna5::kDsLoad2addrStride64B64Vds,
       }) {
    add_sample(cdna5::build_vds(opcode, ds2_fields));
  }
  add_sample(cdna5::build_vds(cdna5::kDsStoreAddtidB32Vds,
                              {.offset0 = 0x34, .offset1 = 0x12, .data0 = 16}));
  add_sample(
      cdna5::build_vds(cdna5::kDsLoadAddtidB32Vds, {.offset0 = 0x34, .offset1 = 0x12, .vdst = 8}));

  for (const uint16_t opcode : {
           cdna5::kClusterLoadB32Vglobal,
           cdna5::kClusterLoadB64Vglobal,
           cdna5::kClusterLoadB128Vglobal,
           cdna5::kClusterLoadAsyncToLdsB8Vglobal,
           cdna5::kClusterLoadAsyncToLdsB32Vglobal,
           cdna5::kClusterLoadAsyncToLdsB64Vglobal,
           cdna5::kClusterLoadAsyncToLdsB128Vglobal,
       }) {
    add_sample(cdna5::build_vglobal(opcode, {.saddr = 4, .vdst = 8, .vaddr = 12}));
  }

  std::vector<const rocjitsu::TranslationRule *> residual_rules;
  for (const rocjitsu::TranslationRule &rule : rocjitsu::semantic_expand_rules_gfx1250_b0_to_a0()) {
    if (rule.discharge.disposition == rocjitsu::RewriteDischargeDisposition::Checked)
      residual_rules.push_back(&rule);
  }
  ASSERT_EQ(samples.size(), residual_rules.size());

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  rocjitsu::SemanticTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5,
                                          rocjitsu::ProcessorRevision::Gfx1250B0,
                                          rocjitsu::ProcessorRevision::Gfx1250A0);
  for (size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
    SCOPED_TRACE(sample_index);
    rocjitsu::DecodeResult decoded = decoder->decode(samples[sample_index].data(), 0);
    ASSERT_TRUE(decoded.succeeded());
    std::unique_ptr<rocjitsu::Instruction> inst = std::move(decoded).value();
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->encoding_id(), residual_rules[sample_index]->src_encoding_id);
    EXPECT_EQ(inst->opcode(), residual_rules[sample_index]->src_opcode);
    EXPECT_TRUE(translator.residual_rewrite_applies(*inst));
    const bool needs_basic_block = residual_rules[sample_index]->discharge.context ==
                                   rocjitsu::RewriteDischargeContext::BasicBlock;
    EXPECT_EQ(translator.residual_rewrite_needs_basic_block(*inst), needs_basic_block);
    EXPECT_EQ(translator.instruction_local_residual_rewrite_applies(*inst), !needs_basic_block);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250UsesNeutralScaledK128Fp8Bf8Wmma) {
  struct WmmaCase {
    const char *name;
    uint16_t source_opcode;
    uint8_t matrix_a_fmt;
    uint8_t matrix_b_fmt;
  };
  constexpr std::array cases = {
      WmmaCase{"fp8_fp8", cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p, 0, 0},
      WmmaCase{"fp8_bf8", cdna5::kVWmmaF3216x16x128Fp8Bf8Vop3p, 0, 1},
      WmmaCase{"bf8_fp8", cdna5::kVWmmaF3216x16x128Bf8Fp8Vop3p, 1, 0},
      WmmaCase{"bf8_bf8", cdna5::kVWmmaF3216x16x128Bf8Bf8Vop3p, 1, 1},
  };
  constexpr cdna5::Vop3pBuilderFields fields{
      .vdst = 54,
      .neg_hi = 7,
      .opsel = 7,
      .src0 = 256 + 16,
      .src1 = 256 + 32,
      .src2 = 256 + 48,
      .opsel_hi = 3,
      .neg = 7,
  };
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr auto completion_wait = kGfx1250WmmaCompletionWait;

  EXPECT_EQ(rocjitsu::semantic_expand_rules_gfx1250_b0_to_a0().size(), 43u);
  for (const WmmaCase &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    auto source_wmma = cdna5::build_vop3p(test_case.source_opcode, fields);
    source_wmma[0] |= uint32_t{1} << 14;
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);
    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_FALSE(translated.text_sections().empty());
    ASSERT_GE(translated.text_sections()[0]->size(), 6 * sizeof(uint32_t));
    const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    cdna5::Vop3pMachineInst prefix{};
    cdna5::Vop3pMachineInst matrix{};
    std::memcpy(&prefix, words, sizeof(prefix));
    std::memcpy(&matrix, words + 2, sizeof(matrix));

    EXPECT_EQ(prefix.op, 0x35u);
    EXPECT_EQ(prefix.src0, 128u);
    EXPECT_EQ(prefix.src1, 128u);
    EXPECT_EQ(prefix.src2, 256u);
    EXPECT_EQ(prefix.opsel, 0u) << "clear source matrix-A reuse in the new encoding";
    EXPECT_EQ(prefix.opsel_hi_2, 0u) << "clear source matrix-B reuse in the new encoding";
    EXPECT_EQ(prefix.opsel_hi, 0u);

    EXPECT_EQ(matrix.op, cdna5::kVWmmaF3216x16x128F8f6f4Vop3p);
    EXPECT_EQ(matrix.vdst, fields.vdst);
    EXPECT_EQ(matrix.src0, fields.src0);
    EXPECT_EQ(matrix.src1, fields.src1);
    EXPECT_EQ(matrix.src2, fields.src2);
    EXPECT_EQ(matrix.opsel, test_case.matrix_a_fmt);
    EXPECT_EQ((matrix.opsel_hi_2 << 2) | matrix.opsel_hi, test_case.matrix_b_fmt);
    EXPECT_EQ(matrix.neg_hi, 4u) << "preserve only C absolute";
    EXPECT_EQ(matrix.neg, 4u) << "preserve only C negate";
    EXPECT_EQ(words[4], completion_wait[0]);
    EXPECT_EQ(words[5], kGfx1250SEndpgm);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250K128WmmaTranslatesCapturedEncodingWithoutK64Fallback) {
  // Captured compiler encoding. OPSEL_HI is three, but these source fields do
  // not select matrix formats for this opcode and must not gate translation.
  constexpr std::array<uint32_t, 2> source_wmma = {0xCC800088u, 0x1A02E542u};
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  EXPECT_EQ(translated.text_sections()[0]->size(), 6 * sizeof(uint32_t));
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(decoded,
                                  [](const auto &candidate) {
                                    return candidate->mnemonic() ==
                                           "v_wmma_scale_f32_16x16x128_f8f6f4";
                                  }),
            1);
  EXPECT_EQ(std::ranges::count_if(decoded,
                                  [](const auto &candidate) {
                                    return candidate->mnemonic().find("16x16x64_fp8") !=
                                               std::string_view::npos ||
                                           candidate->mnemonic().find("16x16x64_bf8") !=
                                               std::string_view::npos;
                                  }),
            0);
}

TEST(BinaryTranslatorE2E, Gfx1250F32K128WmmaRejectsNonVgprMatrixInputs) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  for (const cdna5::Vop3pBuilderFields fields : {
           cdna5::Vop3pBuilderFields{
               .vdst = 54, .src0 = 128, .src1 = 256 + 32, .src2 = 256 + 48, .opsel_hi = 3},
           cdna5::Vop3pBuilderFields{
               .vdst = 54, .src0 = 256 + 16, .src1 = 128, .src2 = 256 + 48, .opsel_hi = 3},
       }) {
    const auto source_wmma = cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p, fields);
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
        result, rocjitsu::DiagnosticKind::Legalization,
        "invalid VGPR source selector at .text byte offset 0"));
  }
}

TEST(BinaryTranslatorE2E, Gfx1250F32K128WmmaRejectsClamp) {
  constexpr auto source_wmma =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p, {.vdst = 54,
                                                                .clamp = 1,
                                                                .src0 = 256 + 16,
                                                                .src1 = 256 + 32,
                                                                .src2 = 256 + 48,
                                                                .opsel_hi = 3});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "Input is malformed, CLAMP \"must be set to zero\" for WMMA/SWMMAC producing "
      "floating-point results"));
}

// The f32 lowering re-encodes VDST, SRC0, SRC1, and SRC2 unchanged. Preserve
// operand encodings that the packed-f16 lowering cannot address when the
// scaled f32 form can still represent them.
TEST(BinaryTranslatorE2E, Gfx1250F32K128WmmaAcceptsOperandsThePackedF16PathRejects) {
  struct OperandCase {
    const char *name;
    cdna5::Vop3pBuilderFields fields;
  };
  const std::array cases = {
      OperandCase{"odd_matrix_and_destination",
                  {.vdst = 55, .src0 = 256 + 17, .src1 = 256 + 33, .src2 = 256 + 49}},
      OperandCase{"nonzero_inline_accumulator",
                  {.vdst = 54, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 129}},
  };
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  for (const OperandCase &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto source_wmma =
        cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p, test_case.fields);
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);

    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_FALSE(translated.text_sections().empty());
    const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    cdna5::Vop3pMachineInst matrix{};
    std::memcpy(&matrix, words + 2, sizeof(matrix));
    EXPECT_EQ(matrix.op, cdna5::kVWmmaF3216x16x128F8f6f4Vop3p);
    EXPECT_EQ(matrix.vdst, test_case.fields.vdst);
    EXPECT_EQ(matrix.src0, test_case.fields.src0);
    EXPECT_EQ(matrix.src1, test_case.fields.src1);
    EXPECT_EQ(matrix.src2, test_case.fields.src2);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250F32K128WmmaRejectsScalarAccumulator) {
  const auto source_wmma =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p,
                         {.vdst = 54, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 8});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "gfx1250 K=128 f32 WMMA scalar accumulator cannot be represented by the scaled form"));
}

TEST(BinaryTranslatorE2E, Gfx1250LowersF16K128Fp8Bf8WmmaThroughF32Scratch) {
  constexpr std::array source_opcodes = {
      cdna5::kVWmmaF1616x16x128Fp8Fp8Vop3p,
      cdna5::kVWmmaF1616x16x128Fp8Bf8Vop3p,
      cdna5::kVWmmaF1616x16x128Bf8Fp8Vop3p,
      cdna5::kVWmmaF1616x16x128Bf8Bf8Vop3p,
  };
  constexpr cdna5::Vop3pBuilderFields fields{
      .vdst = 54,
      .neg_hi = 4,
      .src0 = 256 + 16,
      .src1 = 256 + 32,
      .src2 = 256 + 48,
      .neg = 4,
  };
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  for (const uint16_t source_opcode : source_opcodes) {
    const auto source_wmma = cdna5::build_vop3p(source_opcode, fields);
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);

    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_FALSE(translated.text_sections().empty());
    const auto decoded =
        decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
    const auto matrix = std::ranges::find_if(decoded, [](const auto &candidate) {
      return candidate->mnemonic() == "v_wmma_scale_f32_16x16x128_f8f6f4";
    });
    ASSERT_NE(matrix, decoded.end());
    ASSERT_NE(matrix, decoded.begin());
    EXPECT_EQ((*std::prev(matrix))->mnemonic(), "s_wait_alu");
    ASSERT_NE(std::next(matrix), decoded.end());
    EXPECT_EQ((*std::next(matrix))->mnemonic(), "s_wait_alu");
    ASSERT_NE((*matrix)->raw_encoding(), nullptr);
    cdna5::Vop3pMachineInst lowered_matrix{};
    std::memcpy(&lowered_matrix, (*matrix)->raw_encoding() + 2, sizeof(lowered_matrix));
    EXPECT_EQ(lowered_matrix.vdst % 2u, 0u) << "WMMA scratch tuples must start at an even VGPR";
    EXPECT_EQ(lowered_matrix.opsel, source_opcode == cdna5::kVWmmaF1616x16x128Bf8Fp8Vop3p ||
                                            source_opcode == cdna5::kVWmmaF1616x16x128Bf8Bf8Vop3p
                                        ? 1u
                                        : 0u);
    EXPECT_EQ(lowered_matrix.opsel_hi, source_opcode == cdna5::kVWmmaF1616x16x128Fp8Bf8Vop3p ||
                                               source_opcode == cdna5::kVWmmaF1616x16x128Bf8Bf8Vop3p
                                           ? 1u
                                           : 0u);
    EXPECT_EQ(lowered_matrix.neg_hi, 4u);
    EXPECT_EQ(lowered_matrix.neg, 4u);
    auto after_wait = std::next(matrix, 2);
    ASSERT_LE(std::distance(decoded.begin(), after_wait) + 16,
              std::distance(decoded.begin(), decoded.end()));
    for (uint16_t slot = 0; slot < 16; ++slot, ++after_wait)
      EXPECT_EQ((*after_wait)->mnemonic(), "v_nop_e32");
    EXPECT_EQ(std::ranges::count_if(decoded,
                                    [](const auto &candidate) {
                                      return candidate->mnemonic() ==
                                             "v_wmma_scale_f32_16x16x128_f8f6f4";
                                    }),
              1);
    EXPECT_EQ(std::ranges::count_if(
                  decoded,
                  [](const auto &candidate) { return candidate->mnemonic() == "v_cvt_f32_f16"; }),
              8);
    EXPECT_EQ(std::ranges::count_if(decoded,
                                    [](const auto &candidate) {
                                      return candidate->mnemonic() == "v_cvt_pk_f16_f32";
                                    }),
              4);
    EXPECT_EQ(
        std::ranges::count_if(
            decoded, [](const auto &candidate) { return candidate->mnemonic() == "s_getreg_b32"; }),
        1);
    EXPECT_EQ(std::ranges::count_if(
                  decoded,
                  [](const auto &candidate) { return candidate->mnemonic() == "v_maximum_f32"; }),
              8);
    EXPECT_EQ(std::ranges::count_if(
                  decoded,
                  [](const auto &candidate) { return candidate->mnemonic() == "v_minimum_f32"; }),
              8);
    EXPECT_EQ(std::ranges::count_if(decoded,
                                    [](const auto &candidate) {
                                      return candidate->mnemonic().find("16x16x64") !=
                                             std::string_view::npos;
                                    }),
              0);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250F16K128WmmaAcceptsInlineZeroAccumulator) {
  constexpr auto source_wmma =
      cdna5::build_vop3p(cdna5::kVWmmaF1616x16x128Fp8Fp8Vop3p,
                         {.vdst = 54, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 128});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(
      std::ranges::count_if(
          decoded, [](const auto &candidate) { return candidate->mnemonic() == "v_cvt_f32_f16"; }),
      0);
  const auto matrix = std::ranges::find_if(decoded, [](const auto &candidate) {
    return candidate->mnemonic() == "v_wmma_scale_f32_16x16x128_f8f6f4";
  });
  ASSERT_NE(matrix, decoded.end());
  cdna5::Vop3pMachineInst lowered_matrix{};
  std::memcpy(&lowered_matrix, (*matrix)->raw_encoding() + 2, sizeof(lowered_matrix));
  EXPECT_EQ(lowered_matrix.src2, 128u);
}

TEST(BinaryTranslatorE2E, Gfx1250F16K128WmmaRejectsNonVgprNonzeroAccumulator) {
  for (const uint16_t accumulator : {uint16_t{0}, uint16_t{129}}) {
    SCOPED_TRACE(accumulator);
    const auto source_wmma =
        cdna5::build_vop3p(cdna5::kVWmmaF1616x16x128Fp8Fp8Vop3p,
                           {.vdst = 54, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = accumulator});
    constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
        result, rocjitsu::DiagnosticKind::ExpandFailed,
        "K=128 WMMA accumulator is not a VGPR range or inline zero"))
        << (result.diagnostics.empty() ? "" : result.diagnostics.front().message);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250F16K128WmmaUsesAccumulatorBankAsGeneratedSrc0) {
  constexpr auto set_vgpr_msb = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x10});
  constexpr auto source_wmma =
      cdna5::build_vop3p(cdna5::kVWmmaF1616x16x128Fp8Fp8Vop3p,
                         {.vdst = 54, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {set_vgpr_msb[0], source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  std::vector<uint8_t> modes;
  for (const auto &candidate : decoded) {
    if (candidate->mnemonic() == "s_set_vgpr_msb")
      modes.push_back(static_cast<uint8_t>(candidate->raw_encoding()[0] & 0xffu));
  }
  ASSERT_GE(modes.size(), 2u);
  EXPECT_EQ(modes[0], 0x10u);
  EXPECT_EQ(modes[1], 0x01u) << "the generated conversion reads the accumulator through SRC0";
}

TEST(BinaryTranslatorE2E, Gfx1250F16K128WmmaAllowsSequentialTuplesToCrossVgprBanks) {
  constexpr auto source_wmma =
      cdna5::build_vop3p(cdna5::kVWmmaF1616x16x128Fp8Fp8Vop3p,
                         {.vdst = 254, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 254});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  std::vector<uint16_t> unpack_sources;
  std::vector<uint8_t> packed_destinations;
  std::vector<uint8_t> modes;
  for (const auto &candidate : decoded) {
    if (candidate->mnemonic() == "s_set_vgpr_msb") {
      modes.push_back(static_cast<uint8_t>(candidate->raw_encoding()[0] & 0xffu));
    } else if (candidate->mnemonic() == "v_cvt_f32_f16") {
      cdna5::Vop3MachineInst conversion{};
      std::memcpy(&conversion, candidate->raw_encoding(), sizeof(conversion));
      unpack_sources.push_back(conversion.src0);
    } else if (candidate->mnemonic() == "v_cvt_pk_f16_f32") {
      cdna5::Vop3MachineInst conversion{};
      std::memcpy(&conversion, candidate->raw_encoding(), sizeof(conversion));
      packed_destinations.push_back(conversion.vdst);
    }
  }
  EXPECT_EQ(unpack_sources, (std::vector<uint16_t>{510, 510, 511, 511, 256, 256, 257, 257}));
  EXPECT_EQ(packed_destinations, (std::vector<uint8_t>{254, 255, 0, 1}));
  EXPECT_NE(std::ranges::find(modes, 0x01u), modes.end())
      << "the high accumulator dwords must select SRC0 bank 1";
  EXPECT_NE(std::ranges::find(modes, 0x40u), modes.end())
      << "the high destination dwords must select DST bank 1";
}

TEST(BinaryTranslatorE2E, Gfx1250F16K128WmmaRejectsOddVgprTuples) {
  constexpr auto source_wmma =
      cdna5::build_vop3p(cdna5::kVWmmaF1616x16x128Fp8Fp8Vop3p,
                         {.vdst = 54, .src0 = 256 + 17, .src1 = 256 + 32, .src2 = 256 + 48});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "K=128 WMMA matrix operands and destination are not even VGPR ranges"));
}

TEST(BinaryTranslatorE2E, Gfx1250F16K128WmmaUsesCarrierUnderSgprPressure) {
  constexpr auto source_wmma =
      cdna5::build_vop3p(cdna5::kVWmmaF1616x16x128Fp8Fp8Vop3p,
                         {.vdst = 54, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48});
  auto image =
      rocjitsu::make_gfx1250_image_with_live_sgprs(source_wmma, rocjitsu::REGISTER_SET_MAX_SGPRS);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(
      std::ranges::count_if(
          decoded, [](const auto &inst) { return inst->mnemonic() == "v_readfirstlane_b32_e32"; }),
      1);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "s_cbranch_execz"; }),
            1);
}

TEST(BinaryTranslatorE2E, Gfx1250F16K128WmmaUsesSpillBackedScratchWhenVgprsAreFull) {
  using namespace rocr::llvm::amdhsa;

  constexpr auto source_wmma =
      cdna5::build_vop3p(cdna5::kVWmmaF1616x16x128Fp8Fp8Vop3p,
                         {.vdst = 54, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_wmma[0], source_wmma[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  auto descriptor = rocjitsu::test_support::read_kernel_descriptor_for_test(rodata->data());
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  63);
  rocjitsu::test_support::write_kernel_descriptor_for_test(image.data() + rodata->sectionOffset(),
                                                           descriptor);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  auto options = gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                          rocjitsu::ProcessorRevision::Gfx1250A0);
  options.debug_min_free_vgpr = 256;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
                                        options);
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "scratch_store_b32"; }),
            9);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "scratch_load_b32"; }),
            9);
  EXPECT_EQ(std::ranges::count_if(decoded,
                                  [](const auto &inst) {
                                    return inst->mnemonic() == "v_wmma_scale_f32_16x16x128_f8f6f4";
                                  }),
            1);
}

// Numerical evidence that the packed-f16 lowering preserves values, including
// the FP16 overflow contract. MI400 Shader Programming 4.6.12 makes a WMMA
// result of 16 bits or fewer saturate to +/-MAX when MODE.FP16_OVFL is set,
// infinities included, while V_CVT_PK_F16_F32 preserves infinities in the same
// mode. The lowering compensates with an explicit clamp, so both mode settings
// have to be exercised, and the accumulator has to carry both infinity signs
// plus values that overflow FP16 only after the matrix product.
TEST(BinaryTranslatorE2E, Gfx1250F16K128WmmaLoweringMatchesUnloweredExecution) {
  constexpr uint16_t kMatrixA = 16; // A occupies 16 dwords of packed FP8.
  constexpr uint16_t kMatrixB = 32; // B occupies 16 dwords of packed FP8.
  constexpr uint16_t kMatrixC = 48; // C occupies 4 dwords of packed FP16.
  constexpr uint8_t kDestination = 52;
  constexpr uint32_t kMatrixARegs = 16;
  constexpr uint32_t kMatrixBRegs = 16;
  constexpr uint32_t kMatrixCRegs = 4;
  constexpr uint32_t kDestinationRegs = 4;

  constexpr auto matrix =
      cdna5::build_vop3p(cdna5::kVWmmaF1616x16x128Fp8Fp8Vop3p, {.vdst = kDestination,
                                                                .src0 = 256 + kMatrixA,
                                                                .src1 = 256 + kMatrixB,
                                                                .src2 = 256 + kMatrixC});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t text_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  std::vector<uint32_t> body_words;
  for (size_t offset = 0; offset < text_word_count;) {
    std::unique_ptr<rocjitsu::Instruction> inst(decode_valid(*decoder, text_words + offset));
    ASSERT_NE(inst, nullptr) << "translated word " << offset << " failed to decode";
    const std::string_view mnemonic(inst->mnemonic());
    if (mnemonic == "s_endpgm")
      break;
    EXPECT_NE(mnemonic, "s_set_vgpr_msb") << "bank-0 operands must not need a mode transition";
    EXPECT_NE(mnemonic, "scratch_store_b32") << "the scratch lease must not spill here";
    const size_t words = static_cast<size_t>(inst->size()) / sizeof(uint32_t);
    ASSERT_GT(words, 0u);
    body_words.insert(body_words.end(), text_words + offset, text_words + offset + words);
    offset += words;
  }

  rocjitsu::amdgpu::GpuMemory gpu_mem("gfx1250_f16_k128_diff_mem");
  rocjitsu::amdgpu::L2Cache l2("gfx1250_f16_k128_diff_l2");
  rocjitsu::amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu = rocjitsu::amdgpu::ComputeUnitCore::create("gfx1250", cfg, &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec((1ULL << wf->wf_size()) - 1ULL);
  const uint32_t vb = wf->vgpr_alloc().base;
  const uint32_t lanes = wf->wf_size();

  // FP8 E4M3 magnitudes reach 448, and K=128 terms accumulate, so ordinary
  // pseudo-random matrices already overflow FP16 in most output elements. The
  // accumulator adds both infinities, which only the clamp handles. NaN is
  // excluded here and covered separately below.
  auto fill_inputs = [&](uint32_t accumulator_override) {
    uint32_t state = 0x0f16ac21u;
    auto next = [&state] {
      state = state * 1664525u + 1013904223u;
      return state;
    };
    auto without_fp8_nan = [](uint32_t word) {
      uint32_t out = 0;
      for (uint32_t byte = 0; byte < 4; ++byte) {
        uint32_t value = (word >> (byte * 8)) & 0xffu;
        if ((value & 0x7fu) == 0x7fu)
          value ^= 1u;
        out |= value << (byte * 8);
      }
      return out;
    };
    auto without_f16_nan = [](uint32_t word) {
      uint32_t out = 0;
      for (uint32_t half = 0; half < 2; ++half) {
        uint32_t value = (word >> (half * 16)) & 0xffffu;
        if ((value & 0x7c00u) == 0x7c00u && (value & 0x03ffu) != 0)
          value &= 0xfbffu;
        out |= value << (half * 16);
      }
      return out;
    };
    for (uint32_t reg = 0; reg < kMatrixARegs; ++reg)
      for (uint32_t lane = 0; lane < lanes; ++lane)
        cu->write_vgpr(vb + kMatrixA + reg, lane, without_fp8_nan(next()));
    for (uint32_t reg = 0; reg < kMatrixBRegs; ++reg)
      for (uint32_t lane = 0; lane < lanes; ++lane)
        cu->write_vgpr(vb + kMatrixB + reg, lane, without_fp8_nan(next()));
    for (uint32_t reg = 0; reg < kMatrixCRegs; ++reg)
      for (uint32_t lane = 0; lane < lanes; ++lane)
        cu->write_vgpr(vb + kMatrixC + reg, lane,
                       accumulator_override != 0 ? accumulator_override : without_f16_nan(next()));
    if (accumulator_override != 0)
      return;
    // Packed FP16 pairs: +Inf/-Inf, +MAX/-MAX, +0/-0, denormal/one.
    static constexpr std::array<uint32_t, 4> kSpecials = {0xFC007C00u, 0xFBFF7BFFu, 0x80000000u,
                                                          0x3C000001u};
    for (uint32_t reg = 0; reg < kMatrixCRegs; ++reg)
      for (uint32_t i = 0; i < kSpecials.size(); ++i)
        cu->write_vgpr(vb + kMatrixC + reg, i * 7u, kSpecials[i]);
  };

  auto zero_destination = [&] {
    for (uint32_t reg = 0; reg < kDestinationRegs; ++reg)
      for (uint32_t lane = 0; lane < lanes; ++lane)
        cu->write_vgpr(vb + kDestination + reg, lane, 0u);
  };

  auto snapshot_destination = [&] {
    std::vector<uint32_t> out;
    out.reserve(kDestinationRegs * lanes);
    for (uint32_t reg = 0; reg < kDestinationRegs; ++reg)
      for (uint32_t lane = 0; lane < lanes; ++lane)
        out.push_back(cu->read_vgpr(vb + kDestination + reg, lane));
    return out;
  };

  auto run_source = [&] {
    const std::array<uint32_t, 2> words = {matrix[0], matrix[1]};
    std::unique_ptr<rocjitsu::Instruction> inst(decode_valid(*decoder, words.data()));
    EXPECT_NE(inst, nullptr);
    EXPECT_EQ(std::string_view(inst->mnemonic()), "v_wmma_f16_16x16x128_fp8_fp8");
    cu->execute_instruction(inst.get(), *wf);
  };

  auto run_lowered = [&] {
    for (size_t offset = 0; offset < body_words.size();) {
      std::unique_ptr<rocjitsu::Instruction> inst(
          decode_valid(*decoder, body_words.data() + offset));
      ASSERT_NE(inst, nullptr);
      cu->execute_instruction(inst.get(), *wf);
      offset += static_cast<size_t>(inst->size()) / sizeof(uint32_t);
    }
  };

  std::array<std::vector<uint32_t>, 2> source_by_mode;
  for (const bool fp16_ovfl : {false, true}) {
    SCOPED_TRACE(fp16_ovfl ? "FP16_OVFL=1" : "FP16_OVFL=0");
    wf->set_mode_raw(fp16_ovfl ? rocjitsu::amdgpu::Wavefront::FP16_OVFL_BIT : 0u);

    fill_inputs(0);
    zero_destination();
    run_source();
    const std::vector<uint32_t> run_a = snapshot_destination();

    fill_inputs(0);
    zero_destination();
    run_lowered();
    const std::vector<uint32_t> run_b = snapshot_destination();

    EXPECT_EQ(run_a, run_b);
    for (size_t i = 0; i < run_a.size() && run_a != run_b; ++i) {
      if (run_a[i] == run_b[i])
        continue;
      ADD_FAILURE() << "first divergence at D reg " << (i / lanes) << " lane " << (i % lanes)
                    << ": source=0x" << std::hex << run_a[i] << " lowered=0x" << run_b[i]
                    << std::dec;
      break;
    }
    source_by_mode[fp16_ovfl ? 1 : 0] = run_a;
  }

  // The comparison above only proves the two paths agree. This proves they
  // agree on something: every element that reaches infinity with the mode clear
  // must saturate to the same-signed MAX with it set, and nothing else moves.
  ASSERT_EQ(source_by_mode[0].size(), source_by_mode[1].size());
  size_t saturated_positive = 0;
  size_t saturated_negative = 0;
  for (size_t i = 0; i < source_by_mode[0].size(); ++i) {
    for (uint32_t half = 0; half < 2; ++half) {
      const auto clear = static_cast<uint16_t>(source_by_mode[0][i] >> (half * 16));
      const auto set = static_cast<uint16_t>(source_by_mode[1][i] >> (half * 16));
      if (clear == 0x7C00u) {
        EXPECT_EQ(set, 0x7BFFu) << "+infinity must saturate to +MAX";
        ++saturated_positive;
      } else if (clear == 0xFC00u) {
        EXPECT_EQ(set, 0xFBFFu) << "-infinity must saturate to -MAX";
        ++saturated_negative;
      } else {
        EXPECT_EQ(set, clear) << "FP16_OVFL must not disturb representable results";
      }
    }
  }
  EXPECT_GT(saturated_positive, 0u) << "no output element reached +infinity";
  EXPECT_GT(saturated_negative, 0u) << "no output element reached -infinity";

  // A NaN accumulator produces a NaN result on both paths, but not the same
  // encoding: V_MAXIMUM_F32 and V_MINIMUM_F32 return the canonical quiet NaN,
  // so the clamp drops the sign and payload the source instruction carries
  // through. IEEE 754 leaves both uninterpreted and the ISA promises only that
  // a NaN input yields a NaN output, so compare NaN-ness instead of bits.
  wf->set_mode_raw(0u);
  fill_inputs(0xFE00FE00u);
  zero_destination();
  run_source();
  const std::vector<uint32_t> nan_source = snapshot_destination();
  fill_inputs(0xFE00FE00u);
  zero_destination();
  run_lowered();
  const std::vector<uint32_t> nan_lowered = snapshot_destination();

  auto all_halves_are_nan = [](const std::vector<uint32_t> &values) {
    return std::ranges::all_of(values, [](uint32_t word) {
      for (uint32_t half = 0; half < 2; ++half) {
        const auto value = static_cast<uint16_t>(word >> (half * 16));
        if ((value & 0x7C00u) != 0x7C00u || (value & 0x03FFu) == 0)
          return false;
      }
      return true;
    });
  };
  EXPECT_TRUE(all_halves_are_nan(nan_source));
  EXPECT_TRUE(all_halves_are_nan(nan_lowered));

  if (!wf->is_halted())
    wf->halt();
}

TEST(BinaryTranslatorE2E, Gfx1250MasksM0AroundOffFormClusterLoadForA0) {
  // An off-form (NULL-saddr) cluster load is NOT demoted to a global load. Like
  // every cluster-load form, it stays a cluster load and is wrapped so it runs
  // with M0 = 0; the opcode is unchanged.
  constexpr auto cluster =
      cdna5::build_vglobal(cdna5::kClusterLoadB32Vglobal, {.saddr = 124, .vdst = 8, .vaddr = 12});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {cluster[0], cluster[1], cluster[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  // The load stays a cluster load (no demotion to global_load).
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "cluster_load_b32"; }),
            1);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "global_load_b32"; }),
            0);
  // And it is framed by an M0=0 save/clear/restore. M0 encodes as 125 on gfx1250,
  // inline 0 as 128; SOP1 s_mov is SSRC0[7:0], SDST[22:16].
  constexpr uint16_t kGfx1250M0Operand = 125;
  constexpr uint16_t kInlineZeroOperand = 128;
  bool cleared_m0 = false;
  for (const auto &inst : decoded) {
    if (inst->mnemonic() != "s_mov_b32" || inst->raw_encoding() == nullptr)
      continue;
    const uint32_t word = inst->raw_encoding()[0];
    if (((word >> 16) & 0x7fu) == kGfx1250M0Operand && (word & 0xffu) == kInlineZeroOperand)
      cleared_m0 = true;
  }
  EXPECT_TRUE(cleared_m0) << "off-form cluster load must still set M0 (125) = inline 0 (128)";

  rocjitsu::BinaryTranslator verifier(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto second_result = verifier.translate(translated);
  ASSERT_TRUE(second_result.ok()) << (second_result.diagnostics.empty()
                                          ? ""
                                          : second_result.diagnostics.front().message);
  EXPECT_EQ(second_result.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250MasksM0AroundSaddrClusterLoadForA0) {
  constexpr auto cluster =
      cdna5::build_vglobal(cdna5::kClusterLoadB64Vglobal, {.saddr = 4, .vdst = 8, .vaddr = 12});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {cluster[0], cluster[1], cluster[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "cluster_load_b64"; }),
            1);
  // The load is wrapped to run with M0 = 0, saving/restoring the original M0
  // through a dead SGPR. Assert operands, not just mnemonics: M0 encodes as 125 on
  // gfx1250 and NULL as 124, so a write to 124 would be a discarded NULL write.
  // SOP1 (s_mov): SSRC0[7:0], SDST[22:16]; inline 0 encodes as 128. The sequence
  // is: s_mov scratch, M0 (save) / s_mov M0, 0 (clear) / cluster_load /
  // s_mov M0, scratch (restore).
  constexpr uint16_t kGfx1250M0Operand = 125;
  constexpr uint16_t kInlineZeroOperand = 128;
  bool saved_m0 = false;    // some s_mov reads M0 as source
  bool cleared_m0 = false;  // some s_mov writes M0 from inline 0
  bool restored_m0 = false; // some s_mov writes M0 from a non-inline (scratch) source
  for (const auto &inst : decoded) {
    if (inst->mnemonic() != "s_mov_b32" || inst->raw_encoding() == nullptr)
      continue;
    const uint32_t word = inst->raw_encoding()[0];
    const uint16_t ssrc0 = word & 0xffu;
    const uint16_t sdst = (word >> 16) & 0x7fu;
    if (ssrc0 == kGfx1250M0Operand)
      saved_m0 = true;
    if (sdst == kGfx1250M0Operand && ssrc0 == kInlineZeroOperand)
      cleared_m0 = true;
    if (sdst == kGfx1250M0Operand && ssrc0 != kInlineZeroOperand)
      restored_m0 = true;
  }
  EXPECT_TRUE(saved_m0) << "no s_mov_b32 reads M0 (125) to save it";
  EXPECT_TRUE(cleared_m0) << "no s_mov_b32 sets M0 (125) = inline 0 (128)";
  EXPECT_TRUE(restored_m0) << "no s_mov_b32 restores M0 (125) from the saved scratch";

  rocjitsu::BinaryTranslator verifier(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto second_result = verifier.translate(translated);
  ASSERT_TRUE(second_result.ok()) << (second_result.diagnostics.empty()
                                          ? ""
                                          : second_result.diagnostics.front().message);
  EXPECT_EQ(second_result.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250ClusterLoadsUseVgprCarrierUnderSgprPressure) {
  for (const uint16_t opcode :
       {cdna5::kClusterLoadB32Vglobal, cdna5::kClusterLoadAsyncToLdsB32Vglobal}) {
    SCOPED_TRACE(opcode == cdna5::kClusterLoadB32Vglobal ? "cluster_load_b32"
                                                         : "cluster_load_async_to_lds_b32");
    const auto cluster = cdna5::build_vglobal(opcode, {.saddr = 124, .vdst = 8, .vaddr = 12});
    auto image =
        rocjitsu::make_gfx1250_image_with_live_sgprs(cluster, rocjitsu::REGISTER_SET_MAX_SGPRS);
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);

    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    const auto decoded =
        decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
    EXPECT_EQ(std::ranges::count_if(
                  decoded, [](const auto &inst) { return inst->mnemonic() == "s_cbranch_execz"; }),
              0);
    EXPECT_EQ(std::ranges::count_if(
                  decoded,
                  [](const auto &inst) { return inst->mnemonic() == "v_readfirstlane_b32_e32"; }),
              0);
    EXPECT_EQ(std::ranges::count_if(
                  decoded, [](const auto &inst) { return inst->mnemonic() == "v_writelane_b32"; }),
              1);
    EXPECT_EQ(std::ranges::count_if(
                  decoded, [](const auto &inst) { return inst->mnemonic() == "v_readlane_b32"; }),
              1);
    EXPECT_EQ(
        std::ranges::count_if(decoded, [&](const auto &inst) { return inst->opcode() == opcode; }),
        1);
    ASSERT_GE(decoded.size(), 8u);
    ASSERT_EQ(decoded[0]->mnemonic(), "s_wait_idle");
    ASSERT_EQ(decoded[1]->mnemonic(), "v_writelane_b32");
    ASSERT_EQ(decoded[6]->mnemonic(), "v_readlane_b32");

    ASSERT_NE(decoded[1]->raw_encoding(), nullptr);
    ASSERT_NE(decoded[6]->raw_encoding(), nullptr);
    cdna5::Vop3MachineInst save{};
    cdna5::Vop3MachineInst restore{};
    std::memcpy(&save, decoded[1]->raw_encoding(), sizeof(save));
    std::memcpy(&restore, decoded[6]->raw_encoding(), sizeof(restore));
    EXPECT_EQ(save.src0, restore.vdst) << "the carrier must restore the borrowed SGPR";
    EXPECT_EQ(static_cast<uint16_t>(256u + save.vdst), restore.src0)
        << "the carrier save and restore must use the same VGPR";
    constexpr uint16_t kInlineZero = 128;
    EXPECT_EQ(save.src1, kInlineZero) << "the EXEC-independent carrier must save through lane zero";
    EXPECT_EQ(restore.src1, kInlineZero)
        << "the EXEC-independent carrier must restore through lane zero";

    const auto second = translator.translate(translated);
    ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                            : second.diagnostics.front().message);
    EXPECT_EQ(second.elf_bytes, result.elf_bytes);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250ClusterLoadFailsClosedWithoutExecIndependentCarrier) {
  constexpr auto cluster =
      cdna5::build_vglobal(cdna5::kClusterLoadB32Vglobal, {.saddr = 124, .vdst = 8, .vaddr = 12});
  auto image =
      rocjitsu::make_gfx1250_image_with_live_sgprs(cluster, rocjitsu::REGISTER_SET_MAX_SGPRS);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  auto options = gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                          rocjitsu::ProcessorRevision::Gfx1250A0);
  options.debug_min_free_vgpr = 256;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
                                        options);
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "gfx1250 cluster load could not allocate scalar scratch for M0 preservation"));
}

TEST(BinaryTranslatorE2E, Gfx1250DischargesEveryClusterLoadVariant) {
  struct Case {
    std::array<uint32_t, 3> words;
    std::string_view mnemonic;
  };
  constexpr std::array cases = {
      Case{{0xEE19C000u, 0x00000001u, 0x00000000u}, "cluster_load_b32"},
      Case{{0xEE1A0000u, 0x00000000u, 0x00000002u}, "cluster_load_b64"},
      Case{{0xEE1A4000u, 0x00010002u, 0x00000000u}, "cluster_load_b128"},
      Case{{0xEE1A8000u, 0x00000000u, 0x00000000u}, "cluster_load_async_to_lds_b8"},
      Case{{0xEE1AC000u, 0x00000000u, 0x00000800u}, "cluster_load_async_to_lds_b32"},
      Case{{0xEE1B0000u, 0x00000004u, 0x00000004u}, "cluster_load_async_to_lds_b64"},
      Case{{0xEE1B4000u, 0x00010001u, 0x00000000u}, "cluster_load_async_to_lds_b128"},
  };
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  for (const Case &test_case : cases) {
    SCOPED_TRACE(test_case.mnemonic);
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {test_case.words[0], test_case.words[1], test_case.words[2], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());

    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);

    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);
    EXPECT_TRUE(result.rewrite_discharge_checked);
    EXPECT_TRUE(result.rewrite_discharge_verified);
    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_FALSE(translated.text_sections().empty());
    const auto decoded =
        decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
    EXPECT_EQ(
        std::ranges::count_if(
            decoded, [&](const auto &inst) { return inst->mnemonic() == test_case.mnemonic; }),
        1);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250ClusterLoadPreservesGuestM0ClearWithoutRestore) {
  constexpr auto clear_m0 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 125});
  constexpr auto cluster =
      cdna5::build_vglobal(cdna5::kClusterLoadB32Vglobal, {.saddr = 124, .vdst = 8, .vaddr = 12});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {clear_m0[0], cluster[0], cluster[1], cluster[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  ASSERT_EQ(translated.text_sections()[0]->size(), 5 * sizeof(uint32_t));
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], clear_m0[0]);
  EXPECT_EQ(target_words[1], cluster[0]);
  EXPECT_EQ(target_words[2], cluster[1]);
  EXPECT_EQ(target_words[3], cluster[2]);
  EXPECT_EQ(target_words[4], kGfx1250SEndpgm);

  rocjitsu::BinaryTranslator verifier(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto second_result = verifier.translate(translated);
  ASSERT_TRUE(second_result.ok()) << (second_result.diagnostics.empty()
                                          ? ""
                                          : second_result.diagnostics.front().message);
  EXPECT_EQ(second_result.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250ClusterLoadDoesNotReuseNonzeroM0Write) {
  constexpr auto nonzero_m0 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 1, .sdst = 125});
  constexpr auto clear_m0 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 125});
  constexpr auto cluster =
      cdna5::build_vglobal(cdna5::kClusterLoadB32Vglobal, {.saddr = 124, .vdst = 8, .vaddr = 12});
  constexpr auto generated_save = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 125, .sdst = 0});
  constexpr auto generated_restore =
      cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 0, .sdst = 125});
  constexpr auto dependency_barrier = cdna5::build_sopp(cdna5::kSWaitIdleSopp);
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {nonzero_m0[0], cluster[0], cluster[1], cluster[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  ASSERT_EQ(translated.text_sections()[0]->size(), 9 * sizeof(uint32_t));
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], nonzero_m0[0]);
  EXPECT_EQ(target_words[1], dependency_barrier[0]);
  EXPECT_EQ(target_words[2], generated_save[0]);
  EXPECT_EQ(target_words[3], clear_m0[0]);
  EXPECT_EQ(target_words[4], cluster[0]);
  EXPECT_EQ(target_words[5], cluster[1]);
  EXPECT_EQ(target_words[6], cluster[2]);
  EXPECT_EQ(target_words[7], generated_restore[0]);
  EXPECT_EQ(target_words[8], kGfx1250SEndpgm);

  rocjitsu::BinaryTranslator verifier(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto second_result = verifier.translate(translated);
  ASSERT_TRUE(second_result.ok()) << (second_result.diagnostics.empty()
                                          ? ""
                                          : second_result.diagnostics.front().message);
  EXPECT_EQ(second_result.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250ClusterLoadDoesNotReuseZeroWriteToOtherSgpr) {
  constexpr auto clear_s7 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 7});
  constexpr auto clear_m0 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 125});
  constexpr auto cluster =
      cdna5::build_vglobal(cdna5::kClusterLoadB32Vglobal, {.saddr = 124, .vdst = 8, .vaddr = 12});
  constexpr auto dependency_barrier = cdna5::build_sopp(cdna5::kSWaitIdleSopp);
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {clear_s7[0], cluster[0], cluster[1], cluster[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  ASSERT_EQ(translated.text_sections()[0]->size(), 9 * sizeof(uint32_t));
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], clear_s7[0]);
  EXPECT_EQ(target_words[1], dependency_barrier[0]);
  EXPECT_EQ(target_words[3], clear_m0[0]);

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250ClusterLoadDoesNotReuseM0ClearBypassedByBranch) {
  constexpr auto branch_to_cluster = cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 2});
  constexpr auto source_save = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 125, .sdst = 12});
  constexpr auto clear_m0 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 125});
  constexpr auto cluster =
      cdna5::build_vglobal(cdna5::kClusterLoadB32Vglobal, {.saddr = 124, .vdst = 8, .vaddr = 12});
  constexpr auto source_restore =
      cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 12, .sdst = 125});
  constexpr auto dependency_barrier = cdna5::build_sopp(cdna5::kSWaitIdleSopp);
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {branch_to_cluster[0], source_save[0], clear_m0[0], cluster[0], cluster[1], cluster[2],
       source_restore[0], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  ASSERT_EQ(translated.text_sections()[0]->size(), 12 * sizeof(uint32_t));
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  constexpr auto generated_save = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 125, .sdst = 0});
  constexpr auto generated_restore =
      cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 0, .sdst = 125});
  EXPECT_EQ(target_words[0], branch_to_cluster[0]);
  EXPECT_EQ(target_words[1], source_save[0]);
  EXPECT_EQ(target_words[2], clear_m0[0]);
  EXPECT_EQ(target_words[3], dependency_barrier[0]);
  EXPECT_EQ(target_words[4], generated_save[0]);
  EXPECT_EQ(target_words[5], clear_m0[0]);
  EXPECT_EQ(target_words[6], cluster[0]);
  EXPECT_EQ(target_words[7], cluster[1]);
  EXPECT_EQ(target_words[8], cluster[2]);
  EXPECT_EQ(target_words[9], generated_restore[0]);
  EXPECT_EQ(target_words[10], source_restore[0]);
  EXPECT_EQ(target_words[11], kGfx1250SEndpgm);

  rocjitsu::BinaryTranslator verifier(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto second_result = verifier.translate(translated);
  ASSERT_TRUE(second_result.ok()) << (second_result.diagnostics.empty()
                                          ? ""
                                          : second_result.diagnostics.front().message);
  EXPECT_EQ(second_result.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250MaterializesDsAddtidAddressForA0) {
  constexpr auto addtid =
      cdna5::build_vds(cdna5::kDsLoadAddtidB32Vds, {.offset0 = 0x34, .offset1 = 0x12, .vdst = 8});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {addtid[0], addtid[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "ds_load_addtid_b32"; }),
            0);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "ds_load_b32"; }),
            1);
  EXPECT_EQ(std::ranges::count_if(decoded,
                                  [](const auto &inst) {
                                    return inst->mnemonic() == "v_mbcnt_lo_u32_b32" ||
                                           inst->mnemonic() == "v_mbcnt_hi_u32_b32";
                                  }),
            2);
  // Pin the A0 ADDTID address formula (M0 + tid*4) & 0xfffff, not just mnemonics:
  // this is the stepping-specific contract, so a miscompiled shift/mask/base would
  // silently target the wrong LDS offset. The emitted sequence is
  //   tid   = v_mbcnt_lo/hi           (checked by count above)
  //   tid*4 = v_lshlrev_b32 by 2
  //   +M0   = v_add_nc_u32 SRC0 = M0  (gfx1250 M0 = 125; NULL is 124 — the inverse
  //                                    of CDNA. Reading NULL would use base 0.)
  //   &mask = v_bfe_u32 offset 0 width 20  (0xfffff = 20-bit LDS byte address)
  // Inline integer N encodes as 128+N in a VOP3 source; VOP3 SRC0 is the low 9
  // bits of word[1], SRC1 the next 9 (bits [17:9]), SRC2 the next 9 (word[1]
  // bits... read from the raw words the builders emit).
  constexpr uint16_t kGfx1250M0Operand = 125;
  constexpr uint16_t kInline = 128; // inline integer 0 encodes as 128.

  auto v_lshl = std::ranges::find_if(
      decoded, [](const auto &inst) { return inst->mnemonic() == "v_lshlrev_b32"; });
  ASSERT_NE(v_lshl, decoded.end());
  ASSERT_NE((*v_lshl)->raw_encoding(), nullptr);
  // SRC0 is the shift amount: inline 2 -> encoding 130 (shift left by 2 == *4).
  EXPECT_EQ((*v_lshl)->raw_encoding()[1] & 0x1ffu, static_cast<uint16_t>(kInline + 2));

  auto v_add = std::ranges::find_if(
      decoded, [](const auto &inst) { return inst->mnemonic() == "v_add_nc_u32"; });
  ASSERT_NE(v_add, decoded.end());
  ASSERT_NE((*v_add)->raw_encoding(), nullptr);
  EXPECT_EQ((*v_add)->raw_encoding()[1] & 0x1ffu, kGfx1250M0Operand);

  auto v_bfe = std::ranges::find_if(
      decoded, [](const auto &inst) { return inst->mnemonic() == "v_bfe_u32"; });
  ASSERT_NE(v_bfe, decoded.end());
  ASSERT_NE((*v_bfe)->raw_encoding(), nullptr);
  // SRC1 (bits [17:9]) is the field offset (inline 0 -> 128); SRC2 (bits [26:18])
  // is the field width (inline 20 -> 148). Together: (value >> 0) & ((1<<20)-1).
  EXPECT_EQ(((*v_bfe)->raw_encoding()[1] >> 9) & 0x1ffu, kInline);
  EXPECT_EQ(((*v_bfe)->raw_encoding()[1] >> 18) & 0x1ffu, static_cast<uint16_t>(kInline + 20));
}

// A getpc-plus-literal names its target by distance from the getpc, so relocating the body that
// contains it silently retargets it. A callee packed up against its caller moves whenever the
// source left padding between them, and its instructions are copied verbatim -- so nothing else in
// the pipeline notices that the pair now reaches a different address. The literal must therefore be
// recomputed from the getpc's final placement.
// Read back the single `R_AMDGPU_RELATIVE64` addend and resolve it to `.text` words, so a test can
// assert where a function pointer ends up pointing rather than only that translation succeeded.
[[nodiscard]] std::vector<uint32_t> text_words_at_relative64_addend(std::span<const uint8_t> image,
                                                                    size_t count) {
  rocjitsu::Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  std::vector<rocjitsu::Elf64_Shdr> shdrs(ehdr.e_shnum);
  std::memcpy(shdrs.data(), image.data() + ehdr.e_shoff,
              ehdr.e_shnum * sizeof(rocjitsu::Elf64_Shdr));

  std::optional<uint64_t> addend;
  for (const rocjitsu::Elf64_Shdr &section : shdrs) {
    if (section.sh_type != rocjitsu::SHT_RELA)
      continue;
    for (size_t i = 0; i < section.sh_size / sizeof(rocjitsu::Elf64_Rela); ++i) {
      rocjitsu::Elf64_Rela rela{};
      std::memcpy(&rela, image.data() + section.sh_offset + i * sizeof(rocjitsu::Elf64_Rela),
                  sizeof(rela));
      if ((rela.r_info & 0xffffffffu) == rocjitsu::R_AMDGPU_RELATIVE64)
        addend = static_cast<uint64_t>(rela.r_addend);
    }
  }
  if (!addend)
    return {};

  for (const rocjitsu::Elf64_Shdr &section : shdrs) {
    if ((section.sh_flags & rocjitsu::SHF_EXECINSTR) == 0 || *addend < section.sh_addr ||
        *addend - section.sh_addr >= section.sh_size) {
      continue;
    }
    const uint64_t offset = section.sh_offset + (*addend - section.sh_addr);
    std::vector<uint32_t> words(count);
    std::memcpy(words.data(), image.data() + offset, count * sizeof(uint32_t));
    return words;
  }
  return {};
}

// Translated text replaces .text wholesale, so a device function body that reaches no kernel scope
// would not be preserved: its address range is reoccupied by relocated code. A body a function
// pointer names is therefore adopted as a translation root instead, which both carries it into the
// relocated text and gives the addend a placement to be rewritten to. Kernel bodies need no such
// adoption because their descriptors already make them roots.
TEST(BinaryTranslatorE2E, Gfx1250AdoptsDeviceFunctionBodyReachedByNoScope) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  // The kernel ends at word 0, so nothing below it lands in a scope. A function-pointer table names
  // the body at word 3, so an address the decoded graph never accounts for can still arrive there.
  // That is what separates this from a dead local a linker merely retained: the identical object
  // without the table is accepted by the test below, so the table is the only thing being pinned.
  const std::vector<uint32_t> words = {
      kGfx1250SEndpgm, kGfx1250SNop, kGfx1250SNop, kGfx1250SNop, kGfx1250SEndpgm,
  };

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      words, /*text_function_words=*/2, /*text_function_offset_words=*/3,
      /*function_pointer_table_target_words=*/3);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_TRUE(result.dispatchable());
  EXPECT_NE(result.elf_bytes, image);
  // The addend must land on the adopted body, not on whatever now occupies its original range.
  EXPECT_EQ(text_words_at_relative64_addend(result.elf_bytes, 2),
            (std::vector<uint32_t>{kGfx1250SNop, kGfx1250SEndpgm}));
}
// The same shape with the pointer array's own symbol stripped, which is what a compiler-anonymous
// pointer array looks like. Adoption asks the raw `R_AMDGPU_RELATIVE64` relocations which bodies
// are address-taken, not just the function tables discovery can name, so the slot still names an
// adoptable body even though no `STT_OBJECT` wraps it. The table assertion below is what keeps
// this distinct from the test above: it proves the adoption came from the relocation rather than
// from table discovery. A separately compiled device library reaches most of its functions exactly
// this way, so refusing here would refuse every such object.
TEST(BinaryTranslatorE2E, Gfx1250AdoptsDeviceFunctionBodyHeldByAnUnnamedPointerSlot) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  const std::vector<uint32_t> words = {
      kGfx1250SEndpgm, kGfx1250SNop, kGfx1250SNop, kGfx1250SNop, kGfx1250SEndpgm,
  };

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      words, /*text_function_words=*/2, /*text_function_offset_words=*/3,
      /*function_pointer_table_target_words=*/3,
      /*name_function_pointer_table_with_symbol=*/false);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_TRUE(rocjitsu::discover_relocation_function_tables(source).empty())
      << "the slot must not qualify as a table, or this covers the same path as the test above";

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_TRUE(result.dispatchable());
  EXPECT_NE(result.elf_bytes, image);
  // Accepting the object is not the property worth pinning -- landing the pointer on the right
  // bytes is. The addend must name the adopted body, not whatever now occupies its original range.
  EXPECT_EQ(text_words_at_relative64_addend(result.elf_bytes, 2),
            (std::vector<uint32_t>{kGfx1250SNop, kGfx1250SEndpgm}));
}

// An unprovable call through a pointer, in an object whose only code-address producer is an
// anonymous RELATIVE64 slot. This reaches the producer gate guarding the relocated-by-construction
// permission: with no discovered table and no getpc builder the object used to look producerless,
// so the unresolved s_swap_pc_i64 took the rejection path even though the addend naming its callee
// is discovered, adopted and relocated here. A separately compiled device library is this shape.
TEST(BinaryTranslatorE2E, Gfx1250AllowsUnprovableCallWhenOnlyProducerIsAnAnonymousPointerSlot) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  // s_swap_pc_i64 s[30:31], s[0:1]. Nothing writes s[0:1], so recovery cannot name a target and
  // the consumer stays unresolved, which is the only way to reach the gate.
  constexpr uint32_t kGfx1250SwapPcS30FromS0 = 0xBE9E4900u;
  const std::vector<uint32_t> words = {
      kGfx1250SwapPcS30FromS0, kGfx1250SEndpgm, kGfx1250SNop, kGfx1250SNop, kGfx1250SEndpgm,
  };

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      words, /*text_function_words=*/2, /*text_function_offset_words=*/3,
      /*function_pointer_table_target_words=*/3,
      /*name_function_pointer_table_with_symbol=*/false);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_TRUE(rocjitsu::discover_relocation_function_tables(source).empty())
      << "the slot must stay anonymous, or the producer comes from table discovery instead";

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  ASSERT_TRUE(result.dispatchable());
  EXPECT_EQ(text_words_at_relative64_addend(result.elf_bytes, 2),
            (std::vector<uint32_t>{kGfx1250SNop, kGfx1250SEndpgm}));
}

TEST(BinaryTranslatorE2E, Gfx1250RepointsPcRelativeDataRangeBoundsAfterRelocation) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  constexpr uint32_t kGfx1250GetPcS0 = 0xBE804700u;    // s_get_pc_i64 s[0:1]
  constexpr uint32_t kGfx1250AddNcU64S0 = 0xA980FE00u; // s_add_nc_u64 s[0:1], s[0:1], lit64
  constexpr uint32_t kGfx1250SetPcS30 = 0xBE80481Eu;   // s_set_pc_i64 s[30:31]
  // s_call_i64 s[30:31], simm16 jumps to (this instruction + 4) + simm16 * 4. The callee starts at
  // word 8, so simm16 is 7; words 2..7 are the padding that disappears when it is packed against
  // the caller, which is what moves the getpc.
  constexpr uint32_t kGfx1250CallToWord8 = 0xBA1E0007u;
  constexpr size_t kCalleeWord = 8;

  std::vector<uint32_t> words = {
      kGfx1250CallToWord8, kGfx1250SEndpgm, kGfx1250SNop,     kGfx1250SNop,    kGfx1250SNop,
      kGfx1250SNop,        kGfx1250SNop,    kGfx1250SNop,     kGfx1250GetPcS0, kGfx1250AddNcU64S0,
      0u /*lit lo*/,       0u /*lit hi*/,   kGfx1250SetPcS30,
  };
  const auto sections = [](std::span<const uint8_t> img) {
    rocjitsu::Elf64_Ehdr ehdr{};
    std::memcpy(&ehdr, img.data(), sizeof(ehdr));
    std::vector<rocjitsu::Elf64_Shdr> shdrs(ehdr.e_shnum);
    std::memcpy(shdrs.data(), img.data() + ehdr.e_shoff,
                ehdr.e_shnum * sizeof(rocjitsu::Elf64_Shdr));
    return shdrs;
  };

  const auto verify_target = [&](bool points_one_past_end) {
    SCOPED_TRACE(points_one_past_end ? "one-past-end pointer" : "section-start pointer");
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);

    // Locate .text and a real non-executable allocated target, then point the literal at either
    // bound of that section. Both bounds are valid values in a compiler-generated [begin, end)
    // pair, and both must follow the section when text growth shifts it.
    auto shdrs = sections(image);
    const auto text = std::ranges::find_if(shdrs, [](const rocjitsu::Elf64_Shdr &s) {
      return (s.sh_flags & rocjitsu::SHF_EXECINSTR) != 0;
    });
    const auto data = std::ranges::find_if(shdrs, [](const rocjitsu::Elf64_Shdr &s) {
      return (s.sh_flags & rocjitsu::SHF_ALLOC) != 0 &&
             (s.sh_flags & rocjitsu::SHF_EXECINSTR) == 0 && s.sh_size != 0;
    });
    ASSERT_NE(text, shdrs.end());
    ASSERT_NE(data, shdrs.end());

    const uint64_t source_getpc_result =
        text->sh_addr + kCalleeWord * sizeof(uint32_t) + sizeof(uint32_t);
    const uint64_t target_section_offset = points_one_past_end ? data->sh_size : 0;
    const uint64_t target_vaddr = data->sh_addr + target_section_offset;
    const uint64_t source_literal = target_vaddr - source_getpc_result;
    std::memcpy(image.data() + text->sh_offset + (kCalleeWord + 2) * sizeof(uint32_t),
                &source_literal, sizeof(source_literal));

    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);

    // Find the relocated getpc by its encoding; the callee moved, so its offset is not known up
    // front.
    auto out_shdrs = sections(result.elf_bytes);
    const auto out_text = std::ranges::find_if(out_shdrs, [](const rocjitsu::Elf64_Shdr &s) {
      return (s.sh_flags & rocjitsu::SHF_EXECINSTR) != 0;
    });
    const auto out_data = std::ranges::find_if(out_shdrs, [](const rocjitsu::Elf64_Shdr &s) {
      return (s.sh_flags & rocjitsu::SHF_ALLOC) != 0 &&
             (s.sh_flags & rocjitsu::SHF_EXECINSTR) == 0 && s.sh_size != 0;
    });
    ASSERT_NE(out_text, out_shdrs.end());
    ASSERT_NE(out_data, out_shdrs.end());

    std::optional<size_t> getpc_word;
    for (size_t i = 0; i + 3 < out_text->sh_size / sizeof(uint32_t); ++i) {
      uint32_t word = 0;
      std::memcpy(&word, result.elf_bytes.data() + out_text->sh_offset + i * sizeof(uint32_t),
                  sizeof(word));
      if (word == kGfx1250GetPcS0) {
        getpc_word = i;
        break;
      }
    }
    ASSERT_TRUE(getpc_word.has_value()) << "the callee body must survive into translated text";
    ASSERT_NE(*getpc_word, kCalleeWord) << "the callee must have moved, or this proves nothing";

    uint64_t relocated_literal = 0;
    std::memcpy(&relocated_literal,
                result.elf_bytes.data() + out_text->sh_offset +
                    (*getpc_word + 2) * sizeof(uint32_t),
                sizeof(relocated_literal));
    const uint64_t relocated_getpc_result =
        out_text->sh_addr + *getpc_word * sizeof(uint32_t) + sizeof(uint32_t);
    EXPECT_EQ(relocated_getpc_result + relocated_literal, out_data->sh_addr + target_section_offset)
        << "the relocated body must still reach the same data address";
    EXPECT_NE(relocated_literal, source_literal) << "the body moved, so the literal had to change";

    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(translated.is_valid());
    const auto second = translator.translate(translated);
    ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                            : second.diagnostics.front().message);
    EXPECT_EQ(second.elf_bytes, result.elf_bytes);
  };

  ASSERT_NO_FATAL_FAILURE(verify_target(false));
  ASSERT_NO_FATAL_FAILURE(verify_target(true));
}

TEST(BinaryTranslatorE2E, Gfx1250FailsClosedOnExcludedBarrierSignalIsfirst) {
  // Inline constants encode -1 at 193, so the excluded id is 195.
  constexpr uint8_t kExcludedBarrierIdInline = 195;
  constexpr auto barrier =
      cdna5::build_sop1(cdna5::kSBarrierSignalIsfirstSop1, {.ssrc0 = kExcludedBarrierIdInline});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {barrier[0], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image) << "a fail-closed translation must leave the object unchanged";
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "s_barrier_signal_isfirst cannot name barrier id -3 (inline selector 195)"));
}

TEST(BinaryTranslatorE2E, Gfx1250RejectsLiteralSpellingOfExcludedBarrierId) {
  constexpr auto barrier = cdna5::build_sop1(cdna5::kSBarrierSignalIsfirstSop1, {.ssrc0 = 255});
  constexpr uint32_t kLiteralMinusThree = 0xfffffffdu;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {barrier[0], kLiteralMinusThree, kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.dispatchable());
  EXPECT_EQ(result.elf_bytes, image) << "a fail-closed translation must leave the object unchanged";
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::Legalization,
      "does not support 32-bit literals at .text byte offset 0"))
      << (result.diagnostics.empty() ? "" : result.diagnostics.front().message);
}

TEST(BinaryTranslatorE2E, Gfx1250FailsClosedOnReservedBarrierSignalSelector) {
  constexpr uint8_t kReservedSelector = 0;
  constexpr auto barrier =
      cdna5::build_sop1(cdna5::kSBarrierSignalIsfirstSop1, {.ssrc0 = kReservedSelector});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {barrier[0], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.dispatchable());
  EXPECT_EQ(result.elf_bytes, image) << "a fail-closed translation must leave the object unchanged";
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::Legalization,
      "invalid scalar source selector at .text byte offset 0"));
}

// Every other barrier id must survive translation unchanged. This operand
// admits inline constants and M0; the neighbouring inline ids and an M0 source
// therefore cover the supported spellings.
//
// M0 is the one case here that copies a run-time value. It is copied rather
// than refused because that form draws the id from a zero-extended low field
// and so cannot reach the excluded negative id. If that ever stopped holding,
// this case would belong in the fail-closed test above instead.
TEST(BinaryTranslatorE2E, Gfx1250CopiesRemainingBarrierSignalIsfirstFormsForA0) {
  constexpr uint8_t kInlineMinusOne = 193;
  constexpr uint8_t kInlineMinusTwo = 194;
  constexpr uint8_t kInlineMinusFour = 196;
  constexpr uint8_t kInlinePlusOne = 129;
  constexpr uint8_t kM0Selector = 125;
  for (const uint8_t barrier_id :
       {kInlineMinusOne, kInlineMinusTwo, kInlineMinusFour, kInlinePlusOne, kM0Selector}) {
    const auto barrier =
        cdna5::build_sop1(cdna5::kSBarrierSignalIsfirstSop1, {.ssrc0 = barrier_id});
    constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {barrier[0], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << "barrier id " << static_cast<int>(barrier_id) << ": "
                             << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);

    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_FALSE(translated.text_sections().empty());
    const auto *target_words =
        reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    EXPECT_EQ(target_words[0], barrier[0])
        << "barrier id " << static_cast<int>(barrier_id) << " must be copied verbatim";
  }
}

TEST(BinaryTranslatorE2E, Gfx1250WrapsBareF8f6f4WmmaInNeutralScaleForA0) {
  constexpr cdna5::Vop3pBuilderFields fields{
      .vdst = 32,
      .neg_hi = 4,
      .opsel = 4,
      .src0 = 256 + 64,
      .src1 = 256 + 128,
      .src2 = 256 + 192,
      .opsel_hi = 1,
      .neg = 4,
  };
  constexpr auto wmma = cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128F8f6f4Vop3p, fields);
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr auto completion_wait = kGfx1250WmmaCompletionWait;
  constexpr size_t kPrefixAndMatrixWords = 4;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {wmma[0], wmma[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_GE(decoded.size(), 3u);
  EXPECT_EQ(decoded[0]->mnemonic(), "v_wmma_scale_f32_16x16x128_f8f6f4");
  EXPECT_EQ(decoded[0]->size(), 4 * static_cast<int>(sizeof(uint32_t)));
  EXPECT_EQ(decoded[1]->mnemonic(), "s_wait_alu");
  EXPECT_EQ(decoded[2]->mnemonic(), "s_endpgm");

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  ASSERT_GE(translated.text_sections()[0]->size() / sizeof(uint32_t), kPrefixAndMatrixWords + 2u);
  cdna5::Vop3pMachineInst prefix{};
  std::memcpy(&prefix, target_words, sizeof(prefix));
  EXPECT_EQ(prefix.op, 0x35u);
  EXPECT_EQ(prefix.src0, 128u);
  EXPECT_EQ(prefix.src1, 128u);
  EXPECT_EQ(prefix.src2, 256u);
  EXPECT_EQ(prefix.neg, 0u);
  EXPECT_EQ(prefix.neg_hi, 0u);
  EXPECT_EQ(prefix.opsel, 0u);
  EXPECT_EQ(prefix.opsel_hi, 0u);
  EXPECT_EQ(prefix.opsel_hi_2, 0u);
  EXPECT_EQ(target_words[2], wmma[0]);
  EXPECT_EQ(target_words[3], wmma[1]);
  EXPECT_EQ(target_words[kPrefixAndMatrixWords], completion_wait[0]);
  EXPECT_EQ(target_words[kPrefixAndMatrixWords + 1], kGfx1250SEndpgm);
}

TEST(BinaryTranslatorE2E, Gfx1250PreservesSupportedK64LowPrecisionWmmaAndAppendsWaitForA0) {
  struct Case {
    const char *name;
    uint16_t opcode;
  };
  const std::array<Case, 8> cases = {{
      {"v_wmma_f32_16x16x64_fp8_fp8", cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p},
      {"v_wmma_f32_16x16x64_fp8_bf8", cdna5::kVWmmaF3216x16x64Fp8Bf8Vop3p},
      {"v_wmma_f32_16x16x64_bf8_fp8", cdna5::kVWmmaF3216x16x64Bf8Fp8Vop3p},
      {"v_wmma_f32_16x16x64_bf8_bf8", cdna5::kVWmmaF3216x16x64Bf8Bf8Vop3p},
      {"v_wmma_f16_16x16x64_fp8_fp8", cdna5::kVWmmaF1616x16x64Fp8Fp8Vop3p},
      {"v_wmma_f16_16x16x64_fp8_bf8", cdna5::kVWmmaF1616x16x64Fp8Bf8Vop3p},
      {"v_wmma_f16_16x16x64_bf8_fp8", cdna5::kVWmmaF1616x16x64Bf8Fp8Vop3p},
      {"v_wmma_f16_16x16x64_bf8_bf8", cdna5::kVWmmaF1616x16x64Bf8Bf8Vop3p},
  }};
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr auto completion_wait = kGfx1250WmmaCompletionWait;
  for (const auto &c : cases) {
    const auto wmma = cdna5::build_vop3p(
        c.opcode, {.vdst = 32, .src0 = 256 + 64, .src1 = 256 + 128, .src2 = 256 + 192});
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {wmma[0], wmma[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid()) << c.name;
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    struct TraceClassification {
      bool copied_original;
      bool semantic_lowering;
      bool changed;
      size_t target_word_count;
    };
    std::optional<TraceClassification> wmma_trace;
    translator.set_trace_callback([&](const rocjitsu::TranslationTraceEvent &event) {
      if (event.source_words.size() != wmma.size() || event.source_words[0] != wmma[0] ||
          event.source_words[1] != wmma[1]) {
        return;
      }
      wmma_trace = {.copied_original = event.copied_original,
                    .semantic_lowering = event.semantic_lowering,
                    .changed = event.changed,
                    .target_word_count = event.target_words.size()};
    });
    auto result = translator.translate(source);

    ASSERT_TRUE(result.ok()) << c.name << ": "
                             << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);
    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_FALSE(translated.text_sections().empty()) << c.name;
    ASSERT_GE(translated.text_sections()[0]->size(), 4 * sizeof(uint32_t)) << c.name;
    const auto *target_words =
        reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    EXPECT_EQ(target_words[0], wmma[0]) << c.name;
    EXPECT_EQ(target_words[1], wmma[1]) << c.name;
    EXPECT_EQ(target_words[2], completion_wait[0]) << c.name;
    EXPECT_EQ(target_words[3], kGfx1250SEndpgm) << c.name;
    ASSERT_TRUE(wmma_trace.has_value()) << c.name;
    EXPECT_TRUE(wmma_trace->copied_original) << c.name;
    EXPECT_FALSE(wmma_trace->semantic_lowering) << c.name;
    EXPECT_TRUE(wmma_trace->changed) << c.name;
    EXPECT_EQ(wmma_trace->target_word_count, 3u) << c.name;

    const auto second = translator.translate(translated);
    ASSERT_TRUE(second.ok()) << c.name << ": "
                             << (second.diagnostics.empty() ? ""
                                                            : second.diagnostics.front().message);
    EXPECT_EQ(second.elf_bytes, result.elf_bytes) << c.name;
  }
}

TEST(BinaryTranslatorE2E, Gfx1250DoesNotCreditTrailingWaitAcrossDependentWmma) {
  constexpr auto producer =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                         {.vdst = 32, .src0 = 256 + 64, .src1 = 256 + 128, .src2 = 256 + 192});
  constexpr auto consumer =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                         {.vdst = 96, .src0 = 256 + 8, .src1 = 256 + 16, .src2 = 256 + 32});
  constexpr auto completion_wait = kGfx1250WmmaCompletionWait;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {producer[0], producer[1], consumer[0], consumer[1], completion_wait[0], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));

  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  ASSERT_GE(translated.text_sections()[0]->size(), 7 * sizeof(uint32_t));
  EXPECT_EQ(target_words[0], producer[0]);
  EXPECT_EQ(target_words[1], producer[1]);
  EXPECT_EQ(target_words[2], completion_wait[0]);
  EXPECT_EQ(target_words[3], consumer[0]);
  EXPECT_EQ(target_words[4], consumer[1]);
  EXPECT_EQ(target_words[5], completion_wait[0]);
  EXPECT_EQ(target_words[6], kGfx1250SEndpgm);

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250CompletionWaitCreditChecksOnlyVaVdst) {
  struct Case {
    const char *name;
    uint16_t simm16;
    bool expect_inserted_wait;
  };
  const std::array<Case, 2> cases = {{
      {"nonzero VA_VDST", 0x1f9f, true},
      {"zero VA_VDST with other waits", 0x0000, false},
  }};
  constexpr auto wmma =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                         {.vdst = 32, .src0 = 256 + 64, .src1 = 256 + 128, .src2 = 256 + 192});
  constexpr auto completion_wait = kGfx1250WmmaCompletionWait;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  for (const auto &c : cases) {
    const auto source_wait = cdna5::build_sopp(cdna5::kSWaitAluSopp, {.simm16 = c.simm16});
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {wmma[0], wmma[1], source_wait[0], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));

    const auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << c.name << ": "
                             << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);
    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_FALSE(translated.text_sections().empty()) << c.name;
    ASSERT_GE(translated.text_sections()[0]->size(),
              (c.expect_inserted_wait ? 5u : 4u) * sizeof(uint32_t))
        << c.name;
    const auto *target_words =
        reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    const size_t source_wait_index = c.expect_inserted_wait ? 3u : 2u;
    if (c.expect_inserted_wait) {
      EXPECT_EQ(target_words[2], completion_wait[0]) << c.name;
    }
    EXPECT_EQ(target_words[source_wait_index], source_wait[0]) << c.name;
    EXPECT_EQ(target_words[source_wait_index + 1], kGfx1250SEndpgm) << c.name;

    const auto second = translator.translate(translated);
    ASSERT_TRUE(second.ok()) << c.name << ": "
                             << (second.diagnostics.empty() ? ""
                                                            : second.diagnostics.front().message);
    EXPECT_EQ(second.elf_bytes, result.elf_bytes) << c.name;
  }
}

TEST(BinaryTranslatorE2E, Gfx1250CreditsWaitAcrossCanonicalBankTransitionScaffolding) {
  constexpr auto lower_bank_wmma =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                         {.vdst = 32, .src0 = 256 + 64, .src1 = 256 + 128, .src2 = 256 + 192});
  constexpr auto wait_xcnt = cdna5::build_sopp(cdna5::kSWaitXcntSopp, {.simm16 = 0});
  constexpr auto select_bank_one = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x0055});
  constexpr auto upper_bank_wmma =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                         {.vdst = 96, .src0 = 256 + 8, .src1 = 256 + 16, .src2 = 256 + 24});
  constexpr auto completion_wait = kGfx1250WmmaCompletionWait;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {lower_bank_wmma[0], lower_bank_wmma[1], wait_xcnt[0], select_bank_one[0], upper_bank_wmma[0],
       upper_bank_wmma[1], completion_wait[0], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));

  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "s_wait_alu"; }),
            1);

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250CreditsPhysicallyAdjacentWaitAcrossBasicBlockBoundary) {
  // The conditional branch makes the wait a block leader while the physically
  // preceding WMMA remains on the fallthrough path.
  constexpr auto branch_to_wait = cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 2});
  constexpr auto wmma =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                         {.vdst = 32, .src0 = 256 + 64, .src1 = 256 + 128, .src2 = 256 + 192});
  constexpr auto completion_wait = kGfx1250WmmaCompletionWait;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {branch_to_wait[0], wmma[0], wmma[1], completion_wait[0], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));

  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "s_wait_alu"; }),
            1);

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250CopiesA0CompatibleLowPrecisionSwmmac) {
  // FP8/BF8 sparse K=128 instructions exist on both steppings. They are not in
  // the gfx1250 B0-additions table and their opcodes fit the A0 VOP3P opcode
  // field, so B0-to-A0 translation must preserve their authoritative bytes.
  constexpr auto swmmac =
      cdna5::build_vop3p(cdna5::kVSwmmacF3216x16x128Fp8Fp8Vop3p,
                         {.vdst = 32, .src0 = 256 + 64, .src1 = 256 + 128, .src2 = 256 + 192});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {swmmac[0], swmmac[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto text = translated.text_sections();
  ASSERT_EQ(text.size(), 1u);
  ASSERT_GE(text[0]->size(), 3 * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(text[0]->data(), swmmac.data(), 2 * sizeof(uint32_t)), 0);
  const auto *target_words = reinterpret_cast<const uint32_t *>(text[0]->data());
  EXPECT_EQ(target_words[2], kGfx1250SEndpgm);
}

TEST(BinaryTranslatorE2E, Gfx1250Allows32BitFlatScratchBaseHiSource) {
  // The same special value in a 32-bit source position is unaffected; s_mov_b32
  // must translate cleanly (identity, same-arch stepping copy).
  constexpr uint16_t kFlatScratchBaseHi = 231;
  constexpr auto mov32 =
      cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = kFlatScratchBaseHi, .sdst = 0});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {mov32[0], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  EXPECT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
}

namespace {
// Populate a C-API code object handle owning a copy of @p image, for
// rj_code_translate. rj_code_object_t is non-copyable (RefCounted), so fill a
// caller-provided handle in place.
void init_c_code_object(rj_code_object_t &handle, const std::vector<uint8_t> &image) {
  handle.owned_co = std::make_unique<rocjitsu::AmdGpuCodeObject>(image.data(), image.size());
  handle.co = handle.owned_co.get();
}
} // namespace

TEST(RjCodeTranslateCApi, RejectsSameArchGfx1250) {
  // The C ABI carries no silicon revision, and gfx1250 A0 and B0 share an ELF
  // machine ID, so a same-architecture translation is direction-ambiguous.
  // Revision-aware B0-to-A0 translation uses the C++ BinaryTranslator instead.
  constexpr auto mov = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 0});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {mov[0], kGfx1250SEndpgm});
  rj_code_object_t source{};
  init_c_code_object(source, image);
  ASSERT_TRUE(source.co->is_valid());

  rj_code_dbt_options_t options{};
  options.guest_arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.host_arch = ROCJITSU_CODE_ARCH_CDNA5;
  rj_code_object_t *translated = nullptr;
  EXPECT_NE(rj_code_translate(&source, &options, &translated), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(translated, nullptr);
}

TEST(BinaryTranslatorEnforcesGfx1250Revisions, SameArchUnspecifiedRevisionFailsClosed) {
  // Direct BinaryTranslator use with same-arch gfx1250 and no revisions fails
  // closed because no translation profile was selected.
  constexpr auto cluster =
      cdna5::build_vglobal(cdna5::kClusterLoadB64Vglobal, {.saddr = 4, .vdst = 8, .vaddr = 12});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {cluster[0], cluster[1], cluster[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5);
  auto result = translator.translate(source);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image) << "fail-closed must leave the object unchanged";
}

TEST(BinaryTranslatorEnforcesGfx1250Revisions, TranslatesGfx1250B0ToA0WithRevisions) {
  // With both revisions provided, B0-to-A0 translation selects its profile. The
  // cluster load remains a cluster load framed by an M0=0 sequence.
  constexpr auto cluster =
      cdna5::build_vglobal(cdna5::kClusterLoadB64Vglobal, {.saddr = 4, .vdst = 8, .vaddr = 12});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {cluster[0], cluster[1], cluster[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "cluster_load_b64"; }),
            1);
}

TEST(BinaryTranslatorE2E, Gfx1250GeneratedVgprMsbTransitionsCarryPreviousState) {
  constexpr uint16_t kAllVgprMsbFieldsHwreg = 1u | (12u << 6) | (7u << 11);
  constexpr auto original_mode =
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = kAllVgprMsbFieldsHwreg});
  constexpr auto addtid = cdna5::build_vds(cdna5::kDsStoreAddtidB32Vds, {.offset0 = 4, .data0 = 8});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {original_mode[0], 1u << 2, addtid[0], addtid[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  // The guest MODE write opens the block, so its separation rule puts two V_NOPs
  // ahead of it before anything this test is about.
  ASSERT_GE(decoded.size(), 7u);
  EXPECT_EQ(decoded[0]->mnemonic(), "v_nop_e32");
  EXPECT_EQ(decoded[1]->mnemonic(), "v_nop_e32");
  EXPECT_EQ(decoded[2]->mnemonic(), "s_setreg_imm32_b32");
  // The dependency barrier also separates the guest MODE write from the
  // generated transition. The transition retains its dedicated memory drain.
  EXPECT_EQ(decoded[3]->mnemonic(), "s_wait_idle");
  EXPECT_EQ(decoded[4]->mnemonic(), "s_wait_xcnt");
  EXPECT_EQ(decoded[5]->mnemonic(), "s_set_vgpr_msb");
  ASSERT_NE(decoded[5]->raw_encoding(), nullptr);
  EXPECT_EQ(decoded[5]->raw_encoding()[0] & 0xffffu, 0x0100u);

  // Every generated s_set_vgpr_msb must be immediately preceded by an s_wait_xcnt.
  std::vector<uint16_t> generated_modes;
  for (size_t index = 1; index < decoded.size(); ++index) {
    if (decoded[index]->mnemonic() == "s_set_vgpr_msb") {
      ASSERT_NE(decoded[index]->raw_encoding(), nullptr);
      ASSERT_GT(index, 0u);
      EXPECT_EQ(decoded[index - 1]->mnemonic(), "s_wait_xcnt")
          << "s_set_vgpr_msb at index " << index << " is not guarded by s_wait_xcnt";
      generated_modes.push_back(static_cast<uint16_t>(decoded[index]->raw_encoding()[0] & 0xffffu));
    }
  }
  // Input mode has SRC0 bank 1, SRC1 bank 0. The ds_store_addtid_b32 store-data
  // operand is a SRC1-role VGPR, so the store mode carries src1_bank (0) in the
  // SRC1 field, not src0_bank. With src1_bank 0 the store transition is a no-op
  // and is elided, leaving only the compute transition and the final restore.
  EXPECT_EQ(generated_modes, (std::vector<uint16_t>{0x0100, 0x0001}));
}

TEST(BinaryTranslatorE2E, Gfx1250AddtidStoreUsesSpillBackedScratchWhenVgprsAreFull) {
  using namespace rocr::llvm::amdhsa;

  constexpr auto addtid = cdna5::build_vds(cdna5::kDsStoreAddtidB32Vds, {.offset0 = 4, .data0 = 8});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {addtid[0], addtid[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  auto descriptor = rocjitsu::test_support::read_kernel_descriptor_for_test(rodata->data());
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  63);
  rocjitsu::test_support::write_kernel_descriptor_for_test(image.data() + rodata->sectionOffset(),
                                                           descriptor);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  auto options = gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                          rocjitsu::ProcessorRevision::Gfx1250A0);
  options.debug_min_free_vgpr = 256;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
                                        options);
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "scratch_store_b32"; }),
            1);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "scratch_load_b32"; }),
            1);
  EXPECT_EQ(
      std::ranges::count_if(
          decoded, [](const auto &inst) { return inst->mnemonic() == "ds_store_addtid_b32"; }),
      0);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "ds_store_b32"; }),
            1);

  std::optional<size_t> ds_store_index;
  std::optional<size_t> ds_wait_index;
  std::optional<size_t> scratch_load_index;
  std::optional<cdna5::VscratchMachineInst> scratch_store;
  std::optional<cdna5::VscratchMachineInst> scratch_load;
  std::optional<cdna5::VdsMachineInst> ds_store;
  for (size_t index = 0; index < decoded.size(); ++index) {
    if (decoded[index]->mnemonic() == "scratch_store_b32") {
      scratch_store.emplace();
      std::memcpy(&*scratch_store, decoded[index]->raw_encoding(), sizeof(*scratch_store));
    } else if (decoded[index]->mnemonic() == "ds_store_b32") {
      ds_store_index = index;
      ds_store.emplace();
      std::memcpy(&*ds_store, decoded[index]->raw_encoding(), sizeof(*ds_store));
    } else if (decoded[index]->mnemonic() == "s_wait_dscnt") {
      ds_wait_index = index;
    } else if (decoded[index]->mnemonic() == "scratch_load_b32") {
      scratch_load_index = index;
      scratch_load.emplace();
      std::memcpy(&*scratch_load, decoded[index]->raw_encoding(), sizeof(*scratch_load));
    }
  }
  ASSERT_TRUE(ds_store_index.has_value());
  ASSERT_TRUE(ds_wait_index.has_value());
  ASSERT_TRUE(scratch_load_index.has_value());
  EXPECT_EQ(*ds_wait_index, *ds_store_index + 1u)
      << "the DS store must complete before its address VGPR is restored";
  EXPECT_LT(*ds_wait_index, *scratch_load_index);
  ASSERT_TRUE(scratch_store.has_value());
  ASSERT_TRUE(scratch_load.has_value());
  ASSERT_TRUE(ds_store.has_value());
  EXPECT_EQ(scratch_store->ioffset, scratch_load->ioffset);
  EXPECT_EQ(scratch_store->vsrc, scratch_load->vdst);
  EXPECT_EQ(ds_store->addr, scratch_load->vdst);

  const auto *translated_rodata = rocjitsu::test_support::find_section(translated, ".rodata");
  ASSERT_NE(translated_rodata, nullptr);
  const auto translated_descriptor =
      rocjitsu::test_support::read_kernel_descriptor_for_test(translated_rodata->data());
  EXPECT_GE(translated_descriptor.private_segment_fixed_size, sizeof(uint32_t));

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250SpillBackedRulesFailClosedForDynamicStackKernel) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const auto expect_failure = [&](std::vector<uint32_t> words, std::string_view diagnostic) {
    SCOPED_TRACE(diagnostic);
    words.push_back(kGfx1250SEndpgm);
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
    rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
    ASSERT_TRUE(layout.is_valid());
    const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
    ASSERT_NE(rodata, nullptr);
    auto descriptor = rocjitsu::test_support::read_kernel_descriptor_for_test(rodata->data());
    descriptor.private_segment_fixed_size = 32;
    AMDHSA_BITS_SET(descriptor.kernel_code_properties, KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK, 1);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                    63);
    rocjitsu::test_support::write_kernel_descriptor_for_test(image.data() + rodata->sectionOffset(),
                                                             descriptor);

    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    auto options = gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                            rocjitsu::ProcessorRevision::Gfx1250A0);
    options.debug_min_free_vgpr = 256;
    rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
                                          options);
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
        result, rocjitsu::DiagnosticKind::ExpandFailed, diagnostic));
  };

  constexpr auto addtid = cdna5::build_vds(cdna5::kDsStoreAddtidB32Vds, {.offset0 = 4, .data0 = 8});
  expect_failure(
      {addtid[0], addtid[1]},
      "gfx1250 DS store ADDTID cannot use private-memory spills in a dynamic-stack kernel");

  constexpr auto e5m3 = cdna5::build_vop3(cdna5::kVCvtF32Fp8Vop3,
                                          {.vdst = 30, .opsel = 2, .clamp = 1, .src0 = 256 + 22});
  expect_failure({e5m3[0], e5m3[1]},
                 "gfx1250 E5M3 unpack cannot use private-memory spills in a dynamic-stack kernel");

  constexpr auto pack = cdna5::build_vop3(
      cdna5::kVCvtPkFp8F32Vop3, {.vdst = 30, .clamp = 1, .src0 = 256 + 22, .src1 = 256 + 2});
  expect_failure({pack[0], pack[1]},
                 "gfx1250 E5M3 pack cannot use private-memory spills in a dynamic-stack kernel");

  constexpr auto stochastic = cdna5::build_vop3(
      cdna5::kVCvtSrFp8F32Vop3, {.vdst = 30, .clamp = 1, .src0 = 256 + 22, .src1 = 256 + 2});
  expect_failure(
      {stochastic[0], stochastic[1]},
      "gfx1250 stochastic E5M3 cannot use private-memory spills in a dynamic-stack kernel");
}

TEST(BinaryTranslatorE2E, Gfx1250SpillWaitsForOutstandingVgprProducer) {
  using namespace rocr::llvm::amdhsa;

  constexpr auto pending_load = cdna5::build_vds(cdna5::kDsLoadB32Vds, {.addr = 4, .vdst = 0});
  constexpr auto addtid = cdna5::build_vds(cdna5::kDsStoreAddtidB32Vds, {.offset0 = 4, .data0 = 8});
  constexpr auto consume_v0 = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 256, .vdst = 9});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {pending_load[0], pending_load[1], addtid[0], addtid[1], consume_v0[0], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  auto descriptor = rocjitsu::test_support::read_kernel_descriptor_for_test(rodata->data());
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  63);
  rocjitsu::test_support::write_kernel_descriptor_for_test(image.data() + rodata->sectionOffset(),
                                                           descriptor);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  auto options = gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                          rocjitsu::ProcessorRevision::Gfx1250A0);
  options.debug_min_free_vgpr = 256;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
                                        options);
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  std::optional<size_t> pending_load_index;
  std::optional<size_t> wait_idle_index;
  std::optional<size_t> scratch_store_index;
  for (size_t index = 0; index < decoded.size(); ++index) {
    if (decoded[index]->mnemonic() == "ds_load_b32")
      pending_load_index = index;
    else if (decoded[index]->mnemonic() == "s_wait_idle")
      wait_idle_index = index;
    else if (decoded[index]->mnemonic() == "scratch_store_b32") {
      scratch_store_index = index;
      ASSERT_NE(decoded[index]->raw_encoding(), nullptr);
      cdna5::VscratchMachineInst scratch{};
      std::memcpy(&scratch, decoded[index]->raw_encoding(), sizeof(scratch));
      EXPECT_EQ(scratch.vsrc, 0u);
    }
  }
  ASSERT_TRUE(pending_load_index.has_value());
  ASSERT_TRUE(wait_idle_index.has_value());
  ASSERT_TRUE(scratch_store_index.has_value());
  EXPECT_LT(*pending_load_index, *wait_idle_index);
  EXPECT_EQ(*scratch_store_index, *wait_idle_index + 1u)
      << "the borrowed live VGPR must not be read before its producer completes";

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250DeadScratchWaitsForOutstandingVgprProducer) {
  constexpr auto pending_load = cdna5::build_vds(cdna5::kDsLoadB32Vds, {.addr = 4, .vdst = 0});
  constexpr auto addtid = cdna5::build_vds(cdna5::kDsStoreAddtidB32Vds, {.offset0 = 4, .data0 = 8});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {pending_load[0], pending_load[1], addtid[0], addtid[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  std::optional<size_t> pending_load_index;
  std::optional<size_t> wait_idle_index;
  std::optional<size_t> scratch_write_index;
  for (size_t index = 0; index < decoded.size(); ++index) {
    if (decoded[index]->mnemonic() == "ds_load_b32")
      pending_load_index = index;
    else if (decoded[index]->mnemonic() == "s_wait_idle")
      wait_idle_index = index;
    else if (decoded[index]->mnemonic() == "v_mbcnt_lo_u32_b32") {
      scratch_write_index = index;
      const rocjitsu::Operand *destination = decoded[index]->dst_operand(0);
      ASSERT_NE(destination, nullptr);
      EXPECT_EQ(destination->encoding_value(), 1u)
          << "globally unused v1 is preferred over site-dead v0";
    }
  }
  ASSERT_TRUE(pending_load_index.has_value());
  ASSERT_TRUE(wait_idle_index.has_value());
  ASSERT_TRUE(scratch_write_index.has_value());
  EXPECT_LT(*pending_load_index, *wait_idle_index);
  EXPECT_LT(*wait_idle_index, *scratch_write_index)
      << "a dead scratch VGPR may still have an outstanding producer";
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "scratch_store_b32"; }),
            0);

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250AddtidLoadWaitsForOutstandingVgprProducer) {
  constexpr auto pending_load = cdna5::build_vds(cdna5::kDsLoadB32Vds, {.addr = 4, .vdst = 0});
  constexpr auto addtid = cdna5::build_vds(cdna5::kDsLoadAddtidB32Vds, {.offset0 = 4, .vdst = 0});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {pending_load[0], pending_load[1], addtid[0], addtid[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  std::optional<size_t> wait_idle_index;
  std::optional<size_t> scratch_write_index;
  for (size_t index = 0; index < decoded.size(); ++index) {
    if (decoded[index]->mnemonic() == "s_wait_idle")
      wait_idle_index = index;
    else if (decoded[index]->mnemonic() == "v_mbcnt_lo_u32_b32")
      scratch_write_index = index;
  }
  ASSERT_TRUE(wait_idle_index.has_value());
  ASSERT_TRUE(scratch_write_index.has_value());
  EXPECT_LT(*wait_idle_index, *scratch_write_index)
      << "the ADDTID load destination may still have an outstanding producer";

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250AddtidStoreModeUsesSrc1BankNotSrc0) {
  // Regression for the ADDTID store-data bank: the emitted ds_store_b32 keeps
  // the original store-data VGPR in data0, a SRC1-role operand, so its high bank
  // must come from src1_bank. Choose differing nonzero banks (SRC0 bank 1, SRC1
  // bank 2) so a src0_bank/src1_bank swap is observable: the store mode must be
  // 0x08 (bank 2 in the SRC1 field), not 0x04 (bank 1).
  constexpr uint16_t kAllVgprMsbFieldsHwreg = 1u | (12u << 6) | (7u << 11);
  constexpr auto original_mode =
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = kAllVgprMsbFieldsHwreg});
  // The literal supplies the low eight bits written into MODE[19:12], whose
  // layout is {SRC2,SRC1,SRC0,DST}.
  constexpr uint32_t kModeLiteral = (1u << 2) | (2u << 4); // SRC0 bank 1, SRC1 bank 2.
  constexpr auto addtid = cdna5::build_vds(cdna5::kDsStoreAddtidB32Vds, {.offset0 = 4, .data0 = 8});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {original_mode[0], kModeLiteral, addtid[0], addtid[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);

  // Assert the full generated s_set_vgpr_msb sequence, not just the store mode's
  // low byte, so this also catches a regression that drops the previous-mode
  // carry (SIMM16[15:8]) or restores the wrong original mode. The set_vgpr_msb
  // byte is {DST[7:6], SRC2[5:4], SRC1[3:2], SRC0[1:0]}; the high byte carries
  // the immediately-preceding mode. The live mode is SRC0 bank 1, SRC1 bank 2 =
  // 0x09. Transitions: compute (new 0x00, prev 0x09) = 0x0900; store (new
  // src1_bank 2 in the SRC1 field 0x08, prev 0x00) = 0x0008; restore (new 0x09,
  // prev 0x08) = 0x0809. A src0_bank/src1_bank swap would make the store mode
  // 0x04 instead of 0x08.
  std::vector<uint16_t> generated_modes;
  for (size_t index = 0; index < decoded.size(); ++index) {
    if (decoded[index]->mnemonic() == "s_set_vgpr_msb") {
      ASSERT_NE(decoded[index]->raw_encoding(), nullptr);
      generated_modes.push_back(static_cast<uint16_t>(decoded[index]->raw_encoding()[0] & 0xffffu));
    }
  }
  EXPECT_EQ(generated_modes, (std::vector<uint16_t>{0x0900, 0x0008, 0x0809}))
      << "ADDTID store mode must carry src1_bank (2) in the SRC1 field, and every transition "
         "must record the previous mode in SIMM16[15:8]";
}

TEST(BinaryTranslatorE2E, Gfx1250Iu8WmmaSpacingCreditsOnlySameBlockCanonicalVNops) {
  enum class Consumer {
    EndProgram,
    ValuThenVNops,
    DenseWmma,
    ScalarNopThenDenseWmma,
  };
  struct Case {
    const char *name;
    uint16_t opcode;
    size_t required_slots;
    size_t existing_slots;
    Consumer consumer;
    size_t trailing_slots;
  };
  constexpr std::array cases = {
      Case{.name = "dense_no_existing_padding",
           .opcode = cdna5::kVWmmaI3216x16x64Iu8Vop3p,
           .required_slots = 9,
           .existing_slots = 0,
           .consumer = Consumer::EndProgram,
           .trailing_slots = 0},
      Case{.name = "sparse_no_existing_padding",
           .opcode = cdna5::kVSwmmacI3216x16x128Iu8Vop3p,
           .required_slots = 5,
           .existing_slots = 0,
           .consumer = Consumer::EndProgram,
           .trailing_slots = 0},
      Case{.name = "dense_partial_credit",
           .opcode = cdna5::kVWmmaI3216x16x64Iu8Vop3p,
           .required_slots = 9,
           .existing_slots = 4,
           .consumer = Consumer::EndProgram,
           .trailing_slots = 0},
      Case{.name = "sparse_partial_credit",
           .opcode = cdna5::kVSwmmacI3216x16x128Iu8Vop3p,
           .required_slots = 5,
           .existing_slots = 3,
           .consumer = Consumer::EndProgram,
           .trailing_slots = 0},
      Case{.name = "sparse_exact_credit",
           .opcode = cdna5::kVSwmmacI3216x16x128Iu8Vop3p,
           .required_slots = 5,
           .existing_slots = 5,
           .consumer = Consumer::EndProgram,
           .trailing_slots = 0},
      Case{.name = "dense_over_padded",
           .opcode = cdna5::kVWmmaI3216x16x64Iu8Vop3p,
           .required_slots = 9,
           .existing_slots = 12,
           .consumer = Consumer::EndProgram,
           .trailing_slots = 0},
      Case{.name = "dense_credit_stops_at_valu",
           .opcode = cdna5::kVWmmaI3216x16x64Iu8Vop3p,
           .required_slots = 9,
           .existing_slots = 0,
           .consumer = Consumer::ValuThenVNops,
           .trailing_slots = 8},
      Case{.name = "back_to_back_dense",
           .opcode = cdna5::kVWmmaI3216x16x64Iu8Vop3p,
           .required_slots = 9,
           .existing_slots = 0,
           .consumer = Consumer::DenseWmma,
           .trailing_slots = 0},
      Case{.name = "dense_partial_credit_before_matrix_consumer",
           .opcode = cdna5::kVWmmaI3216x16x64Iu8Vop3p,
           .required_slots = 9,
           .existing_slots = 4,
           .consumer = Consumer::DenseWmma,
           .trailing_slots = 0},
      Case{.name = "scalar_nop_receives_no_credit",
           .opcode = cdna5::kVWmmaI3216x16x64Iu8Vop3p,
           .required_slots = 9,
           .existing_slots = 0,
           .consumer = Consumer::ScalarNopThenDenseWmma,
           .trailing_slots = 0},
  };
  const uint32_t v_nop = cdna5::build_vop1(cdna5::kVNopVop1)[0];
  const uint32_t v_mov = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 0})[0];
  const uint32_t s_nop = cdna5::build_sopp(cdna5::kSNopSopp, {.simm16 = 8})[0];
  const auto consumer_wmma =
      cdna5::build_vop3p(cdna5::kVWmmaI3216x16x64Iu8Vop3p,
                         {.vdst = 64, .src0 = 256 + 72, .src1 = 256 + 136, .src2 = 256 + 200});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  for (const Case &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto source_wmma = cdna5::build_vop3p(
        test_case.opcode, {.vdst = 32, .src0 = 256 + 64, .src1 = 256 + 128, .src2 = 256 + 192});
    std::vector<uint32_t> source_words = {source_wmma[0], source_wmma[1]};
    source_words.insert(source_words.end(), test_case.existing_slots, v_nop);
    if (test_case.consumer == Consumer::ValuThenVNops) {
      source_words.push_back(v_mov);
      source_words.insert(source_words.end(), test_case.trailing_slots, v_nop);
    } else if (test_case.consumer == Consumer::DenseWmma) {
      source_words.insert(source_words.end(), consumer_wmma.begin(), consumer_wmma.end());
    } else if (test_case.consumer == Consumer::ScalarNopThenDenseWmma) {
      source_words.push_back(s_nop);
      source_words.insert(source_words.end(), consumer_wmma.begin(), consumer_wmma.end());
    }
    source_words.push_back(kGfx1250SEndpgm);
    auto image =
        rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(source_words);
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);

    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_FALSE(translated.text_sections().empty());
    const size_t expected_prefix_slots =
        std::max(test_case.required_slots, test_case.existing_slots);
    std::vector<uint32_t> expected_words = {source_wmma[0], source_wmma[1]};
    expected_words.insert(expected_words.end(), expected_prefix_slots, v_nop);
    if (test_case.consumer == Consumer::ValuThenVNops) {
      expected_words.push_back(v_mov);
      expected_words.insert(expected_words.end(), test_case.trailing_slots, v_nop);
    } else if (test_case.consumer == Consumer::DenseWmma) {
      expected_words.insert(expected_words.end(), consumer_wmma.begin(), consumer_wmma.end());
      expected_words.insert(expected_words.end(), 9, v_nop);
    } else if (test_case.consumer == Consumer::ScalarNopThenDenseWmma) {
      expected_words.push_back(s_nop);
      expected_words.insert(expected_words.end(), consumer_wmma.begin(), consumer_wmma.end());
      expected_words.insert(expected_words.end(), 9, v_nop);
    }
    expected_words.push_back(kGfx1250SEndpgm);
    ASSERT_EQ(translated.text_sections()[0]->size(), expected_words.size() * sizeof(uint32_t));
    const auto *target_words =
        reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    EXPECT_EQ(std::vector<uint32_t>(target_words, target_words + expected_words.size()),
              expected_words);

    rocjitsu::BinaryTranslator verifier(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    auto second_result = verifier.translate(translated);
    ASSERT_TRUE(second_result.ok())
        << (second_result.diagnostics.empty() ? "" : second_result.diagnostics.front().message);
    EXPECT_EQ(second_result.elf_bytes, result.elf_bytes);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250Iu8WmmaDoesNotCreditVNopsAcrossBlockBoundary) {
  constexpr auto branch_to_padding = cdna5::build_sopp(cdna5::kSCbranchScc1Sopp, {.simm16 = 2});
  constexpr auto source_wmma =
      cdna5::build_vop3p(cdna5::kVWmmaI3216x16x64Iu8Vop3p,
                         {.vdst = 32, .src0 = 256 + 64, .src1 = 256 + 128, .src2 = 256 + 192});
  const uint32_t v_nop = cdna5::build_vop1(cdna5::kVNopVop1)[0];
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  std::vector<uint32_t> source_words = {branch_to_padding[0], source_wmma[0], source_wmma[1]};
  source_words.insert(source_words.end(), 9, v_nop);
  source_words.push_back(kGfx1250SEndpgm);
  auto image =
      rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(source_words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  ASSERT_EQ(translated.text_sections()[0]->size(), 22 * sizeof(uint32_t));
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[1], source_wmma[0]);
  EXPECT_EQ(target_words[2], source_wmma[1]);
  for (size_t slot = 0; slot < 18; ++slot)
    EXPECT_EQ(target_words[3 + slot], v_nop);
  EXPECT_EQ(target_words[21], kGfx1250SEndpgm);

  rocjitsu::BinaryTranslator verifier(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto second_result = verifier.translate(translated);
  ASSERT_TRUE(second_result.ok()) << (second_result.diagnostics.empty()
                                          ? ""
                                          : second_result.diagnostics.front().message);
  EXPECT_EQ(second_result.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250TensorLoadAlwaysClearsDynamicMulticastMask) {
  // Multicast is selected dynamically by D# group-1 bits [15:0], not by an
  // instruction bit. A0 must therefore clear the runtime mask for every tensor
  // load and restore the descriptor word after the instruction.
  constexpr auto source_tensor =
      cdna5::build_vimage(cdna5::kTensorLoadToLdsVimage,
                          {.vaddr4 = 124, .vaddr0 = 8, .vaddr1 = 0, .vaddr2 = 124, .vaddr3 = 124});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_tensor[0], source_tensor[1], source_tensor[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  ASSERT_GE(translated.text_sections()[0]->size(), 8 * sizeof(uint32_t));
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  // D0 occupies s[8:11] and D1 occupies s[0:7], so the first dead ordinary
  // scalar register is s12.
  constexpr auto save_d1 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 0, .sdst = 12});
  constexpr auto clear_mask =
      cdna5::build_sop2(cdna5::kSPackHhB32B16Sop2, {.ssrc0 = 128, .ssrc1 = 0, .sdst = 0});
  constexpr auto restore_d1 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 12, .sdst = 0});
  constexpr auto dependency_barrier = cdna5::build_sopp(cdna5::kSWaitIdleSopp);
  EXPECT_EQ(target_words[0], dependency_barrier[0]);
  EXPECT_EQ(target_words[1], save_d1[0]);
  EXPECT_EQ(target_words[2], clear_mask[0]);
  EXPECT_EQ(target_words[3], source_tensor[0]);
  EXPECT_EQ(target_words[4], source_tensor[1]);
  EXPECT_EQ(target_words[5], source_tensor[2]);
  EXPECT_EQ(target_words[6], restore_d1[0]);
  EXPECT_EQ(target_words[7], kGfx1250SEndpgm);

  auto second_result = translator.translate(translated);
  ASSERT_TRUE(second_result.ok()) << (second_result.diagnostics.empty()
                                          ? ""
                                          : second_result.diagnostics.front().message);
  EXPECT_EQ(second_result.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250TensorLoadUsesHighDeadSgprUnderCompilerPressure) {
  constexpr auto source_tensor =
      cdna5::build_vimage(cdna5::kTensorLoadToLdsVimage,
                          {.vaddr4 = 124, .vaddr0 = 8, .vaddr1 = 0, .vaddr2 = 124, .vaddr3 = 124});
  auto image = rocjitsu::make_gfx1250_image_with_live_sgprs(
      source_tensor, rocjitsu::REGISTER_SET_ALLOCATABLE_SGPRS);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "s_cbranch_execz"; }),
            0);
  EXPECT_EQ(
      std::ranges::count_if(
          decoded, [](const auto &inst) { return inst->mnemonic() == "v_readfirstlane_b32_e32"; }),
      0);
  EXPECT_EQ(std::ranges::count_if(
                decoded, [](const auto &inst) { return inst->mnemonic() == "tensor_load_to_lds"; }),
            1);

  ASSERT_GE(translated.text_sections()[0]->size(), 7 * sizeof(uint32_t));
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  // The generic 102-SGPR ceiling leaves s102/s103 dead, but gfx1250 scratch
  // allocation skips those scratch-base registers and uses s104.
  constexpr uint8_t kScratch = 104;
  constexpr auto save_d1 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 0, .sdst = kScratch});
  constexpr auto clear_mask =
      cdna5::build_sop2(cdna5::kSPackHhB32B16Sop2, {.ssrc0 = 128, .ssrc1 = 0, .sdst = 0});
  constexpr auto restore_d1 =
      cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = kScratch, .sdst = 0});
  constexpr auto dependency_barrier = cdna5::build_sopp(cdna5::kSWaitIdleSopp);
  EXPECT_EQ(target_words[0], dependency_barrier[0]);
  EXPECT_EQ(target_words[1], save_d1[0]);
  EXPECT_EQ(target_words[2], clear_mask[0]);
  EXPECT_EQ(target_words[3], source_tensor[0]);
  EXPECT_EQ(target_words[4], source_tensor[1]);
  EXPECT_EQ(target_words[5], source_tensor[2]);
  EXPECT_EQ(target_words[6], restore_d1[0]);

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250TensorScratchWaitsForOutstandingSgprProducer) {
  constexpr auto pending_load =
      cdna5::build_smem(cdna5::kSLoadB32Smem, {.sbase = 8, .sdata = 12, .soffset = 128});
  constexpr auto source_tensor =
      cdna5::build_vimage(cdna5::kTensorLoadToLdsVimage,
                          {.vaddr4 = 124, .vaddr0 = 8, .vaddr1 = 0, .vaddr2 = 124, .vaddr3 = 124});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {pending_load[0], pending_load[1], source_tensor[0], source_tensor[1], source_tensor[2],
       kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  std::optional<size_t> pending_load_index;
  std::optional<size_t> wait_idle_index;
  std::optional<size_t> scratch_write_index;
  for (size_t index = 0; index < decoded.size(); ++index) {
    if (decoded[index]->mnemonic() == "s_load_b32")
      pending_load_index = index;
    else if (decoded[index]->mnemonic() == "s_wait_idle")
      wait_idle_index = index;
    else if (decoded[index]->mnemonic() == "s_mov_b32") {
      ASSERT_NE(decoded[index]->raw_encoding(), nullptr);
      cdna5::Sop1MachineInst move{};
      std::memcpy(&move, decoded[index]->raw_encoding(), sizeof(move));
      if (move.ssrc0 == 0 && move.sdst == 12)
        scratch_write_index = index;
    }
  }
  ASSERT_TRUE(pending_load_index.has_value());
  ASSERT_TRUE(wait_idle_index.has_value());
  ASSERT_TRUE(scratch_write_index.has_value());
  EXPECT_LT(*pending_load_index, *wait_idle_index);
  EXPECT_EQ(*scratch_write_index, *wait_idle_index + 1u)
      << "the dead SGPR scratch may still have an outstanding producer";

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250TensorLoadFailsClosedWhenEverySgprIsLive) {
  constexpr auto source_tensor =
      cdna5::build_vimage(cdna5::kTensorLoadToLdsVimage,
                          {.vaddr4 = 124, .vaddr0 = 8, .vaddr1 = 0, .vaddr2 = 124, .vaddr3 = 124});
  auto image =
      rocjitsu::make_gfx1250_image_with_live_sgprs(source_tensor, rocjitsu::REGISTER_SET_MAX_SGPRS);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "gfx1250 tensor-load mask rule could not allocate scalar scratch"));
}

TEST(BinaryTranslatorE2E, Gfx1250TensorLoadPreservesGuestMaskClearWithoutRestore) {
  // An immediately preceding clear in the same block already guarantees the
  // A0 mask invariant. Preserve the guest's choice not to restore the mask.
  constexpr auto source_clear =
      cdna5::build_sop2(cdna5::kSPackHhB32B16Sop2, {.ssrc0 = 128, .ssrc1 = 0, .sdst = 0});
  constexpr auto source_tensor =
      cdna5::build_vimage(cdna5::kTensorLoadToLdsVimage,
                          {.vaddr4 = 124, .vaddr0 = 8, .vaddr1 = 0, .vaddr2 = 124, .vaddr3 = 124});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_clear[0], source_tensor[0], source_tensor[1], source_tensor[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  auto options = gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                          rocjitsu::ProcessorRevision::Gfx1250A0);
  options.verify_rewrite_discharge = true;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
                                        options);
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  EXPECT_TRUE(result.rewrite_discharge_checked);
  EXPECT_TRUE(result.rewrite_discharge_verified);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  ASSERT_EQ(translated.text_sections()[0]->size(), 5 * sizeof(uint32_t));
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], source_clear[0]);
  EXPECT_EQ(target_words[1], source_tensor[0]);
  EXPECT_EQ(target_words[2], source_tensor[1]);
  EXPECT_EQ(target_words[3], source_tensor[2]);
  EXPECT_EQ(target_words[4], kGfx1250SEndpgm);

  auto second_result = translator.translate(translated);
  ASSERT_TRUE(second_result.ok()) << (second_result.diagnostics.empty()
                                          ? ""
                                          : second_result.diagnostics.front().message);
  EXPECT_EQ(second_result.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250TensorLoadDoesNotReuseNonzeroMaskWrite) {
  constexpr auto source_nonzero =
      cdna5::build_sop2(cdna5::kSPackHhB32B16Sop2, {.ssrc0 = 1, .ssrc1 = 0, .sdst = 0});
  constexpr auto source_tensor =
      cdna5::build_vimage(cdna5::kTensorLoadToLdsVimage,
                          {.vaddr4 = 124, .vaddr0 = 8, .vaddr1 = 0, .vaddr2 = 124, .vaddr3 = 124});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_nonzero[0], source_tensor[0], source_tensor[1], source_tensor[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto first = translator.translate(source);
  ASSERT_TRUE(first.ok()) << (first.diagnostics.empty() ? "" : first.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(first.elf_bytes.data(), first.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  constexpr auto save_d1 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 0, .sdst = 12});
  constexpr auto clear_mask =
      cdna5::build_sop2(cdna5::kSPackHhB32B16Sop2, {.ssrc0 = 128, .ssrc1 = 0, .sdst = 0});
  constexpr auto restore_d1 = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 12, .sdst = 0});
  constexpr auto dependency_barrier = cdna5::build_sopp(cdna5::kSWaitIdleSopp);
  EXPECT_EQ(target_words[0], source_nonzero[0]);
  EXPECT_EQ(target_words[1], dependency_barrier[0]);
  EXPECT_EQ(target_words[2], save_d1[0]);
  EXPECT_EQ(target_words[3], clear_mask[0]);
  EXPECT_EQ(target_words[4], source_tensor[0]);
  EXPECT_EQ(target_words[5], source_tensor[1]);
  EXPECT_EQ(target_words[6], source_tensor[2]);
  EXPECT_EQ(target_words[7], restore_d1[0]);
  EXPECT_EQ(target_words[8], kGfx1250SEndpgm);

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, first.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250TensorLoadDoesNotReuseClearOfDifferentDescriptor) {
  constexpr auto clear_s4 =
      cdna5::build_sop2(cdna5::kSPackHhB32B16Sop2, {.ssrc0 = 128, .ssrc1 = 4, .sdst = 4});
  constexpr auto clear_s0 =
      cdna5::build_sop2(cdna5::kSPackHhB32B16Sop2, {.ssrc0 = 128, .ssrc1 = 0, .sdst = 0});
  constexpr auto source_tensor =
      cdna5::build_vimage(cdna5::kTensorLoadToLdsVimage,
                          {.vaddr4 = 124, .vaddr0 = 8, .vaddr1 = 0, .vaddr2 = 124, .vaddr3 = 124});
  constexpr auto dependency_barrier = cdna5::build_sopp(cdna5::kSWaitIdleSopp);
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {clear_s4[0], source_tensor[0], source_tensor[1], source_tensor[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  ASSERT_EQ(translated.text_sections()[0]->size(), 9 * sizeof(uint32_t));
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], clear_s4[0]);
  EXPECT_EQ(target_words[1], dependency_barrier[0]);
  EXPECT_EQ(target_words[3], clear_s0[0]);

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250TensorLoadDoesNotReuseMaskPrefixBypassedByBranch) {
  // The branch makes the tensor load a block leader, so it can bypass the
  // guest-authored clear and must receive a new wrapper. The guest save keeps
  // s12 live across the load, making s13 the first free scratch register.
  constexpr auto branch_to_tensor = cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 2});
  constexpr auto source_save = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 0, .sdst = 12});
  constexpr auto source_clear =
      cdna5::build_sop2(cdna5::kSPackHhB32B16Sop2, {.ssrc0 = 128, .ssrc1 = 0, .sdst = 0});
  constexpr auto source_tensor =
      cdna5::build_vimage(cdna5::kTensorLoadToLdsVimage,
                          {.vaddr4 = 124, .vaddr0 = 8, .vaddr1 = 0, .vaddr2 = 124, .vaddr3 = 124});
  constexpr auto source_restore = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 12, .sdst = 0});
  constexpr auto dependency_barrier = cdna5::build_sopp(cdna5::kSWaitIdleSopp);
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {branch_to_tensor[0], source_save[0], source_clear[0], source_tensor[0], source_tensor[1],
       source_tensor[2], source_restore[0], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  ASSERT_GE(translated.text_sections()[0]->size(), 10 * sizeof(uint32_t));
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  constexpr auto relocated_branch = cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 0});
  constexpr auto generated_save = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 0, .sdst = 13});
  constexpr auto generated_restore =
      cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 13, .sdst = 0});
  EXPECT_EQ(target_words[0], relocated_branch[0]);
  EXPECT_EQ(target_words[1], dependency_barrier[0]);
  EXPECT_EQ(target_words[2], generated_save[0]);
  EXPECT_EQ(target_words[3], source_clear[0]);
  EXPECT_EQ(target_words[4], source_tensor[0]);
  EXPECT_EQ(target_words[5], source_tensor[1]);
  EXPECT_EQ(target_words[6], source_tensor[2]);
  EXPECT_EQ(target_words[7], generated_restore[0]);
  EXPECT_EQ(target_words[8], source_restore[0]);
  EXPECT_EQ(target_words[9], kGfx1250SEndpgm);

  auto second_result = translator.translate(translated);
  ASSERT_TRUE(second_result.ok()) << (second_result.diagnostics.empty()
                                          ? ""
                                          : second_result.diagnostics.front().message);
  EXPECT_EQ(second_result.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250RegularWmmaScaleEncodesSrc2AndWaitsForCompletion) {
  // Real VOP3PX2 encoding from the mxscale offline oracle. Bits [58:50] are
  // the inline-zero selector in the B0 object even though SQ decodes that unused
  // field as a scalar dependency. The A0 output must encode VGPR0 without changing
  // other bits.
  constexpr std::array<uint32_t, 4> base_scale = {0xCC350000u, 0x0202954Eu, 0xCC332042u,
                                                  0x050A01CAu};
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kScaleSrc2Mask = 0x1ffu << 18;
  constexpr auto completion_wait = kGfx1250WmmaCompletionWait;
  for (const uint32_t source_src2 : {0x080u, 0x100u}) {
    SCOPED_TRACE(::testing::Message() << "source_src2=" << source_src2);
    auto source_scale = base_scale;
    source_scale[1] = (source_scale[1] & ~kScaleSrc2Mask) | (source_src2 << 18);
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {source_scale[0], source_scale[1], source_scale[2], source_scale[3], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    auto result = translator.translate(source);
    ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                            : result.diagnostics.front().message);

    rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_FALSE(translated.text_sections().empty());
    const auto *target_words =
        reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
    const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
    ASSERT_EQ(target_word_count, 6u);
    EXPECT_EQ(target_words[0], source_scale[0]);
    EXPECT_EQ(target_words[1], (source_scale[1] & ~kScaleSrc2Mask) | (0x100u << 18));
    EXPECT_EQ(target_words[2], source_scale[2]);
    EXPECT_EQ(target_words[3], source_scale[3]);
    EXPECT_EQ(target_words[4], completion_wait[0]);
    EXPECT_EQ(target_words[5], kGfx1250SEndpgm);
  }
}

TEST(BinaryTranslatorE2E, Gfx1250FloatingScaledWmmaRejectsNonzeroCmFields) {
  struct CmCase {
    const char *name;
    uint16_t prefix_opcode;
    uint8_t prefix_cm;
    uint8_t matrix_cm;
  };
  constexpr std::array cases = {
      CmCase{"regular_prefix", 0x35, 1, 0},
      CmCase{"regular_matrix", 0x35, 0, 1},
      CmCase{"scale16_prefix", 0x3a, 1, 0},
      CmCase{"scale16_matrix", 0x3a, 0, 1},
  };
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  for (const CmCase &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto prefix = cdna5::build_vop3p(
        test_case.prefix_opcode, {.clamp = test_case.prefix_cm, .src0 = 4, .src1 = 6, .src2 = 0});
    const auto matrix =
        cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128F8f6f4Vop3p, {.vdst = 96,
                                                                  .clamp = test_case.matrix_cm,
                                                                  .src0 = 256 + 16,
                                                                  .src1 = 256 + 32,
                                                                  .src2 = 128});
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {prefix[0], prefix[1], matrix[0], matrix[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
        result, rocjitsu::DiagnosticKind::Legalization, "Invalid instruction opcode"));
  }
}

TEST(BinaryTranslatorE2E, Gfx1250BareFloatingWmmaRejectsClamp) {
  constexpr auto matrix =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
                         {.vdst = 96, .clamp = 1, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 128});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "CLAMP \"must be set to zero\" for WMMA/SWMMAC producing floating-point results"));
}

TEST(BinaryTranslatorE2E, Gfx1250PreservesVop3DppExtensionWord) {
  // Real rocSPARSE encoding: v_fma_f32_e64_dpp is a 64-bit VOP3 followed by
  // one DPP control DWORD. The decoder must consume and preserve all 12 bytes,
  // rather than treating the control word as a new instruction.
  constexpr std::array<uint32_t, 3> source_vop3_dpp = {0xD6130000u, 0x020200FAu, 0xFF08E403u};
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {source_vop3_dpp[0], source_vop3_dpp[1], source_vop3_dpp[2], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  EXPECT_EQ(target_words[0], source_vop3_dpp[0]);
  EXPECT_EQ(target_words[1], source_vop3_dpp[1]);
  EXPECT_EQ(target_words[2], source_vop3_dpp[2]);
  EXPECT_EQ(target_words[3], kGfx1250SEndpgm);
}

TEST(BinaryTranslatorE2E, Gfx1250SplitsRegularScaleFp4AcrossMForA0) {
  constexpr auto scale =
      cdna5::build_vop3p(0x35, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 0, .opsel_hi = 1});
  constexpr auto matrix = cdna5::build_vop3p(
      cdna5::kVWmmaF3232x16x128F4Vop3p,
      {.vdst = 96, .neg_hi = 4, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48, .neg = 4});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {scale[0], scale[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  std::vector<size_t> pass_offsets;
  for (size_t i = 0; i < word_count; ++i) {
    if (((words[i] >> 16) & 0xffu) == 0x35u)
      pass_offsets.push_back(i);
  }
  ASSERT_EQ(pass_offsets.size(), 2u);
  for (uint16_t half = 0; half < 2; ++half) {
    const size_t offset = pass_offsets[half];
    ASSERT_LE(offset + 4, word_count);
    EXPECT_EQ((words[offset] >> 11) & 0x1u, half);
    EXPECT_EQ(words[offset] & ((1u << 13) | (1u << 14)), 0u);
    EXPECT_EQ((words[offset + 1] >> 18) & 0x1ffu, 0x100u);
    EXPECT_EQ((words[offset + 1] >> 27) & 0x1u, 1u) << "B scale selection uses SCL_OPSEL_HI[0]";

    cdna5::Vop3pMachineInst replacement{};
    std::memcpy(&replacement, words + offset + 2, sizeof(replacement));
    EXPECT_EQ(replacement.op, cdna5::kVWmmaF3216x16x128F8f6f4Vop3p);
    EXPECT_EQ(replacement.vdst, 96u + half * 8u);
    EXPECT_EQ(replacement.src0, 256u + 16u + half * 8u);
    EXPECT_EQ(replacement.src1, 256u + 32u);
    EXPECT_EQ(replacement.src2, 256u + 48u + half * 8u);
    EXPECT_EQ(replacement.opsel, 4u);
    EXPECT_EQ(replacement.opsel_hi_2, 1u);
    EXPECT_EQ(replacement.opsel_hi, 0u);
    EXPECT_EQ(replacement.neg_hi, 4u);
    EXPECT_EQ(replacement.clamp, 0u);
    EXPECT_EQ(replacement.neg, 4u);
  }

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250RegularScaleFp4RejectsEveryDestructiveUpperInputOverlap) {
  struct OverlapCase {
    const char *name;
    cdna5::Vop3pBuilderFields matrix;
  };
  constexpr std::array cases = {
      OverlapCase{"upper_a", {.vdst = 96, .src0 = 256 + 88, .src1 = 256 + 32, .src2 = 256 + 48}},
      OverlapCase{"shared_b", {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 96, .src2 = 256 + 48}},
      OverlapCase{"upper_c", {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 88}},
  };
  constexpr auto scale = cdna5::build_vop3p(0x35, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 0});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  for (const OverlapCase &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto matrix = cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p, test_case.matrix);
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {scale[0], scale[1], matrix[0], matrix[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
        result, rocjitsu::DiagnosticKind::ExpandFailed,
        "regular-Scale 32x16 lower destination overlaps an input needed by the upper half"));
  }
}

TEST(BinaryTranslatorE2E, Gfx1250RegularScaleFp4RejectsScaleSourceOverlap) {
  struct ScaleCase {
    const char *name;
    uint16_t scale_src0;
    uint16_t scale_src1;
  };
  constexpr std::array cases = {
      ScaleCase{"scale_src0", 256 + 96, 256 + 66},
      ScaleCase{"scale_src1", 256 + 64, 256 + 96},
  };
  // Scale operands ignore VGPR-MSB and always name bank-zero registers. Keep
  // matrix A/B in nonzero banks while D remains in bank zero so a lowering that
  // incorrectly applies the matrix SRC0/SRC1 bank to the scale operands misses
  // this destructive overlap.
  constexpr auto set_vgpr_msb = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x09});
  constexpr auto matrix =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                         {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  for (const ScaleCase &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto scale = cdna5::build_vop3p(
        0x35, {.src0 = test_case.scale_src0, .src1 = test_case.scale_src1, .src2 = 0});
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {set_vgpr_msb[0], scale[0], scale[1], matrix[0], matrix[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
        result, rocjitsu::DiagnosticKind::ExpandFailed,
        "regular-Scale 32x16 lower destination overlaps an input needed by the upper half"));
  }
}

TEST(BinaryTranslatorE2E, Gfx1250RegularScaleFp4AcceptsScalarAndInlineScaleSources) {
  constexpr auto scale = cdna5::build_vop3p(0x35, {.src0 = 4, .src1 = 128, .src2 = 0});
  constexpr auto matrix =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                         {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {scale[0], scale[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  size_t prefix_count = 0;
  for (size_t i = 0; i + 1 < word_count; ++i) {
    if (((words[i] >> 16) & 0xffu) != 0x35u)
      continue;
    cdna5::Vop3pMachineInst prefix{};
    std::memcpy(&prefix, words + i, sizeof(prefix));
    EXPECT_EQ(prefix.src0, 4u);
    EXPECT_EQ(prefix.src1, 128u);
    EXPECT_EQ(prefix.src2, 256u);
    ++prefix_count;
  }
  EXPECT_EQ(prefix_count, 2u);
}

TEST(BinaryTranslatorE2E, Gfx1250RegularScaleFp4AdjustsUpperCAndDestinationBanks) {
  constexpr auto set_vgpr_msb = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x90});
  constexpr auto scale = cdna5::build_vop3p(0x35, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 0});
  constexpr auto matrix =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                         {.vdst = 252, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 252});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {set_vgpr_msb[0], scale[0], scale[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  std::vector<size_t> pass_offsets;
  for (size_t i = 0; i < word_count; ++i) {
    if (((words[i] >> 16) & 0xffu) == 0x35u)
      pass_offsets.push_back(i);
  }
  ASSERT_EQ(pass_offsets.size(), 2u);
  cdna5::Vop3pMachineInst upper{};
  std::memcpy(&upper, words + pass_offsets[1] + 2, sizeof(upper));
  EXPECT_EQ(upper.vdst, 4u);
  EXPECT_EQ(upper.src0, 256u + 24u);
  EXPECT_EQ(upper.src2, 256u + 4u);

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  std::vector<uint8_t> modes;
  for (const auto &inst : decoded) {
    if (inst->mnemonic() == "s_set_vgpr_msb") {
      ASSERT_NE(inst->raw_encoding(), nullptr);
      modes.push_back(static_cast<uint8_t>(inst->raw_encoding()[0] & 0xffu));
    }
  }
  EXPECT_NE(std::ranges::find(modes, 0xe0u), modes.end());
  EXPECT_EQ(modes.back(), 0x90u);
}

TEST(BinaryTranslatorE2E, Gfx1250RegularScaleFp4AdjustsUpperABank) {
  constexpr auto scale = cdna5::build_vop3p(0x35, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 0});
  constexpr auto matrix =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                         {.vdst = 96, .src0 = 256 + 252, .src1 = 256 + 32, .src2 = 256 + 48});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {scale[0], scale[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  std::vector<size_t> pass_offsets;
  for (size_t i = 0; i < word_count; ++i) {
    if (((words[i] >> 16) & 0xffu) == 0x35u)
      pass_offsets.push_back(i);
  }
  ASSERT_EQ(pass_offsets.size(), 2u);
  cdna5::Vop3pMachineInst upper{};
  std::memcpy(&upper, words + pass_offsets[1] + 2, sizeof(upper));
  EXPECT_EQ(upper.src0, 256u + 4u);

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  std::vector<uint8_t> modes;
  for (const auto &inst : decoded) {
    if (inst->mnemonic() == "s_set_vgpr_msb") {
      ASSERT_NE(inst->raw_encoding(), nullptr);
      modes.push_back(static_cast<uint8_t>(inst->raw_encoding()[0] & 0xffu));
    }
  }
  EXPECT_NE(std::ranges::find(modes, 0x01u), modes.end());
  EXPECT_EQ(modes.back(), 0u);
}

TEST(BinaryTranslatorE2E, Gfx1250RegularScaleFp4RejectsBank3UpperHalfOverflow) {
  constexpr auto set_vgpr_msb = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0xf0});
  constexpr auto scale = cdna5::build_vop3p(0x35, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 0});
  constexpr auto matrix =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                         {.vdst = 252, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 252});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {set_vgpr_msb[0], scale[0], scale[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "regular-Scale 32x16 FP4 split exceeds the VGPR address space"));
}

TEST(BinaryTranslatorE2E, Gfx1250SplitsScale16Fp4AcrossMOnlyForA0) {
  auto scale16 = cdna5::build_vop3p(0x3a, {.neg_hi = 2,
                                           .opsel = 4,
                                           .src0 = 256 + 64,
                                           .src1 = 256 + 66,
                                           .src2 = 0,
                                           .opsel_hi = 1,
                                           .neg = 2});
  scale16[0] |= uint32_t{1} << 14;
  constexpr auto matrix = cdna5::build_vop3p(
      cdna5::kVWmmaF3232x16x128F4Vop3p,
      {.vdst = 96, .neg_hi = 4, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48, .neg = 4});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr auto completion_wait = kGfx1250WmmaCompletionWait;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {scale16[0], scale16[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  std::vector<size_t> pass_offsets;
  for (size_t i = 0; i < word_count; ++i) {
    if (((words[i] >> 16) & 0xffu) == 0x3au)
      pass_offsets.push_back(i);
  }
  ASSERT_EQ(pass_offsets.size(), 2u);

  for (uint16_t half = 0; half < 2; ++half) {
    const size_t offset = pass_offsets[half];
    ASSERT_LE(offset + 4, word_count);
    cdna5::Vop3pMachineInst prefix{};
    std::memcpy(&prefix, words + offset, sizeof(prefix));
    EXPECT_EQ(prefix.op, 0x3au);
    EXPECT_EQ(prefix.src0, 256u + 64u);
    EXPECT_EQ(prefix.src1, 256u + 66u);
    EXPECT_EQ(prefix.src2, 256u);
    EXPECT_EQ(prefix.opsel, half);
    EXPECT_EQ(prefix.opsel_hi, 1u);
    EXPECT_EQ(prefix.opsel_hi_2, 0u);
    EXPECT_EQ(prefix.neg_hi, 2u);
    EXPECT_EQ(prefix.neg, 2u);
    EXPECT_EQ(words[offset] & ((1u << 13) | (1u << 14)), 0u);

    cdna5::Vop3pMachineInst replacement{};
    std::memcpy(&replacement, words + offset + 2, sizeof(replacement));
    const uint16_t destination = static_cast<uint16_t>(96u + half * 8u);
    EXPECT_EQ(replacement.op, cdna5::kVWmmaF3216x16x128F8f6f4Vop3p);
    EXPECT_EQ(replacement.vdst, destination);
    EXPECT_EQ(replacement.src0, 256u + 16u + half * 8u);
    EXPECT_EQ(replacement.src1, 256u + 32u);
    EXPECT_EQ(replacement.src2, 256u + 48u + half * 8u);
    EXPECT_EQ(replacement.opsel, 4u);
    EXPECT_EQ(replacement.opsel_hi_2, 1u);
    EXPECT_EQ(replacement.opsel_hi, 0u);
    EXPECT_EQ(replacement.neg_hi, 4u);
    EXPECT_EQ(replacement.neg, 4u);
  }
  EXPECT_EQ(word_count, 10u);
  EXPECT_EQ(words[8], completion_wait[0]);
  EXPECT_EQ(words[9], kGfx1250SEndpgm);

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250Scale16Fp4AcceptsScalarAndInlineScaleSources) {
  constexpr auto scale16 = cdna5::build_vop3p(0x3a, {.src0 = 4, .src1 = 128, .src2 = 0});
  constexpr auto matrix =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                         {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {scale16[0], scale16[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  size_t prefix_count = 0;
  for (size_t i = 0; i + 1 < word_count; ++i) {
    if (((words[i] >> 16) & 0xffu) != 0x3au)
      continue;
    cdna5::Vop3pMachineInst prefix{};
    std::memcpy(&prefix, words + i, sizeof(prefix));
    EXPECT_EQ(prefix.src0, 4u);
    EXPECT_EQ(prefix.src1, 128u);
    EXPECT_EQ(prefix.src2, 256u);
    ++prefix_count;
  }
  EXPECT_EQ(prefix_count, 2u);
}

TEST(BinaryTranslatorE2E, Gfx1250Scale16RejectsMisalignedOrOutOfRangeVgprPairs) {
  struct MatrixCase {
    const char *name;
    uint16_t opcode;
  };
  constexpr std::array matrix_cases = {
      MatrixCase{"m16", cdna5::kVWmmaF3216x16x128F8f6f4Vop3p},
      MatrixCase{"m32", cdna5::kVWmmaF3232x16x128F4Vop3p},
  };
  struct ScaleCase {
    const char *name;
    bool source1;
    uint16_t base;
  };
  constexpr std::array scale_cases = {
      ScaleCase{"src0_odd", false, 65},
      ScaleCase{"src1_odd", true, 67},
      ScaleCase{"src0_v255", false, 255},
      ScaleCase{"src1_v255", true, 255},
  };
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  for (const MatrixCase &matrix_case : matrix_cases) {
    for (const ScaleCase &scale_case : scale_cases) {
      SCOPED_TRACE(std::string(matrix_case.name) + "_" + scale_case.name);
      const auto scale16 = cdna5::build_vop3p(
          0x3a, {.src0 = static_cast<uint16_t>(256 + (scale_case.source1 ? 64 : scale_case.base)),
                 .src1 = static_cast<uint16_t>(256 + (scale_case.source1 ? scale_case.base : 66)),
                 .src2 = 0});
      const auto matrix = cdna5::build_vop3p(
          matrix_case.opcode, {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48});
      auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
          {scale16[0], scale16[1], matrix[0], matrix[1], kGfx1250SEndpgm});
      rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
      rocjitsu::BinaryTranslator translator(
          ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
          gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                   rocjitsu::ProcessorRevision::Gfx1250A0));
      const auto result = translator.translate(source);

      EXPECT_FALSE(result.ok());
      EXPECT_EQ(result.elf_bytes, image);
      EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
          result, rocjitsu::DiagnosticKind::Legalization, "Invalid instruction opcode"));
    }
  }
}

TEST(BinaryTranslatorE2E, Gfx1250Scale16Fp4RejectsEveryDestructiveUpperInputOverlap) {
  struct OverlapCase {
    const char *name;
    cdna5::Vop3pBuilderFields scale;
    cdna5::Vop3pBuilderFields matrix;
  };
  constexpr std::array cases = {
      OverlapCase{"scale_src0_second_register",
                  {.src0 = 256 + 96, .src1 = 256 + 64, .src2 = 0},
                  {.vdst = 97, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48}},
      OverlapCase{"scale_src1_second_register",
                  {.src0 = 256 + 64, .src1 = 256 + 96, .src2 = 0},
                  {.vdst = 97, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48}},
      OverlapCase{"upper_a",
                  {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 0},
                  {.vdst = 96, .src0 = 256 + 88, .src1 = 256 + 32, .src2 = 256 + 48}},
      OverlapCase{"shared_b",
                  {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 0},
                  {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 96, .src2 = 256 + 48}},
      OverlapCase{"upper_c",
                  {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 0},
                  {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 88}},
  };
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  for (const OverlapCase &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto scale16 = cdna5::build_vop3p(0x3a, test_case.scale);
    const auto matrix = cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p, test_case.matrix);
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {scale16[0], scale16[1], matrix[0], matrix[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image);
    EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
        result, rocjitsu::DiagnosticKind::ExpandFailed,
        "Scale16 32x16 lower destination overlaps an input needed by the upper half"));
  }
}

TEST(BinaryTranslatorE2E, Gfx1250Scale16Fp4AdjustsUpperCAndDestinationBanks) {
  constexpr auto set_vgpr_msb = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x90});
  constexpr auto scale16 =
      cdna5::build_vop3p(0x3a, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 256});
  constexpr auto matrix =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                         {.vdst = 252, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 252});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {set_vgpr_msb[0], scale16[0], scale16[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  std::vector<size_t> pass_offsets;
  for (size_t i = 0; i < word_count; ++i) {
    if (((words[i] >> 16) & 0xffu) == 0x3au)
      pass_offsets.push_back(i);
  }
  ASSERT_EQ(pass_offsets.size(), 2u);
  cdna5::Vop3pMachineInst upper{};
  std::memcpy(&upper, words + pass_offsets[1] + 2, sizeof(upper));
  EXPECT_EQ(upper.vdst, 4u);
  EXPECT_EQ(upper.src0, 256u + 24u);
  EXPECT_EQ(upper.src2, 256u + 4u);

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  std::vector<uint8_t> modes;
  for (const auto &inst : decoded) {
    if (inst->mnemonic() == "s_set_vgpr_msb") {
      ASSERT_NE(inst->raw_encoding(), nullptr);
      modes.push_back(static_cast<uint8_t>(inst->raw_encoding()[0] & 0xffu));
    }
  }
  EXPECT_NE(std::ranges::find(modes, 0xe0u), modes.end());
  EXPECT_EQ(modes.back(), 0x90u);
}

TEST(BinaryTranslatorE2E, Gfx1250Scale16Fp4RejectsBank3UpperHalfOverflow) {
  constexpr auto set_vgpr_msb = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0xf0});
  constexpr auto scale16 =
      cdna5::build_vop3p(0x3a, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 256});
  constexpr auto matrix =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                         {.vdst = 252, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 252});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {set_vgpr_msb[0], scale16[0], scale16[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "Scale16 32x16 FP4 split exceeds the VGPR address space"));
}

TEST(BinaryTranslatorE2E, Gfx1250PreservesNativeM16Scale16ForA0) {
  constexpr auto scale16 = cdna5::build_vop3p(
      0x3a, {.neg_hi = 2, .opsel = 1, .src0 = 4, .src1 = 6, .src2 = 0, .opsel_hi = 1});
  auto matrix = cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128F8f6f4Vop3p, {.vdst = 96,
                                                                          .neg_hi = 4,
                                                                          .opsel = 3,
                                                                          .src0 = 256 + 16,
                                                                          .src1 = 256 + 32,
                                                                          .src2 = 128,
                                                                          .neg = 4});
  matrix[0] |= uint32_t{1} << 14;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr auto completion_wait = kGfx1250WmmaCompletionWait;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {scale16[0], scale16[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  ASSERT_EQ(target_word_count, 6u);
  constexpr uint32_t kUnusedScaleSrc2Mask = 0x1ffu << 18;
  EXPECT_EQ(target_words[0], scale16[0]);
  EXPECT_EQ(target_words[1],
            (scale16[1] & ~kUnusedScaleSrc2Mask) | (static_cast<uint32_t>(256u) << 18));
  EXPECT_EQ(target_words[2], matrix[0]);
  EXPECT_EQ(target_words[3], matrix[1]);
  EXPECT_EQ(target_words[4], completion_wait[0]);
  EXPECT_EQ(target_words[5], kGfx1250SEndpgm);

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(std::ranges::count_if(decoded,
                                  [](const auto &candidate) {
                                    return candidate->mnemonic() ==
                                           "v_wmma_scale16_f32_16x16x128_f8f6f4";
                                  }),
            1);
}

TEST(BinaryTranslatorE2E, Gfx1250Scale16M32DoesNotNeedScratchWhenVgprsAreFull) {
  using namespace rocr::llvm::amdhsa;

  constexpr auto scale16 =
      cdna5::build_vop3p(0x3a, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 256});
  constexpr auto matrix =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                         {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {scale16[0], scale16[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  auto descriptor = rocjitsu::test_support::read_kernel_descriptor_for_test(rodata->data());
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  63);
  rocjitsu::test_support::write_kernel_descriptor_for_test(image.data() + rodata->sectionOffset(),
                                                           descriptor);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  auto options = gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                          rocjitsu::ProcessorRevision::Gfx1250A0);
  options.debug_min_free_vgpr = 256;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
                                        options);
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);

  struct ScratchAccess {
    size_t word_offset;
    uint8_t vgpr;
    uint32_t byte_offset;
  };
  std::vector<ScratchAccess> stores;
  std::vector<ScratchAccess> loads;
  size_t wait_store_count = 0;
  size_t wait_load_count = 0;
  for (const auto &inst : decoded) {
    if (inst->mnemonic() == "s_wait_storecnt") {
      ++wait_store_count;
      continue;
    }
    if (inst->mnemonic() == "s_wait_loadcnt") {
      ++wait_load_count;
      continue;
    }
    if (inst->mnemonic() != "scratch_store_b32" && inst->mnemonic() != "scratch_load_b32")
      continue;
    ASSERT_NE(inst->raw_encoding(), nullptr);
    cdna5::VscratchMachineInst scratch{};
    std::memcpy(&scratch, inst->raw_encoding(), sizeof(scratch));
    const bool is_load = inst->mnemonic() == "scratch_load_b32";
    (is_load ? loads : stores)
        .push_back(
            ScratchAccess{.word_offset = inst->src_loc() / sizeof(uint32_t),
                          .vgpr = static_cast<uint8_t>(is_load ? scratch.vdst : scratch.vsrc),
                          .byte_offset = scratch.ioffset});
  }

  EXPECT_TRUE(stores.empty());
  EXPECT_TRUE(loads.empty());
  EXPECT_EQ(wait_store_count, 0u);
  EXPECT_EQ(wait_load_count, 0u);
  EXPECT_EQ(std::ranges::count_if(decoded,
                                  [](const auto &candidate) {
                                    return candidate->mnemonic() ==
                                           "v_wmma_scale16_f32_16x16x128_f8f6f4";
                                  }),
            2);
  ASSERT_GT(target_word_count, 0u);
  EXPECT_EQ(target_words[target_word_count - 1], kGfx1250SEndpgm);
}

TEST(BinaryTranslatorE2E, Gfx1250Scale16M32IgnoresUnavailableScratchSpillSpace) {
  using namespace rocr::llvm::amdhsa;

  constexpr auto scale16 =
      cdna5::build_vop3p(0x3a, {.src0 = 256 + 64, .src1 = 256 + 66, .src2 = 256});
  constexpr auto matrix =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                         {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = 256 + 48});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {scale16[0], scale16[1], matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  auto descriptor = rocjitsu::test_support::read_kernel_descriptor_for_test(rodata->data());
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  63);
  descriptor.private_segment_fixed_size = 0x7ffffffdu;
  rocjitsu::test_support::write_kernel_descriptor_for_test(image.data() + rodata->sectionOffset(),
                                                           descriptor);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  auto options = gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                          rocjitsu::ProcessorRevision::Gfx1250A0);
  options.debug_min_free_vgpr = 256;
  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
                                        options);
  const auto result = translator.translate(source);

  EXPECT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
}

// The standalone form produces floating-point results, so it carries the same
// control requirement as the scaled entry points. A malformed encoding must be
// refused rather than split.
TEST(BinaryTranslatorE2E, Gfx1250Standalone32x16Fp4RejectsNonZeroClampForA0) {
  constexpr auto matrix = cdna5::build_vop3p(
      cdna5::kVWmmaF3232x16x128F4Vop3p,
      {.vdst = 96, .clamp = 1, .src0 = 256 + 16, .src1 = 256 + 40, .src2 = 256 + 64});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.elf_bytes, image) << "fail-closed must leave the object unchanged";
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "Input is malformed, CLAMP \"must be set to zero\" for WMMA/SWMMAC producing "
      "floating-point results"));
}

TEST(BinaryTranslatorE2E, Gfx1250SplitsStandalone32x16Fp4IntoScaledHalvesForA0) {
  // A0 has no M=32 FP4 matrix opcode, and it cannot resume an unprefixed
  // low-precision matrix body after a trap. The lowering is therefore two M=16
  // halves, each carrying a neutral inline-zero scale prefix.
  constexpr uint16_t kMatrixA = 16;
  constexpr uint16_t kMatrixB = 40;
  constexpr uint16_t kMatrixC = 64;
  constexpr uint8_t kDestination = 96;
  constexpr uint16_t kHalfDwords = 8;
  constexpr auto matrix =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p, {.vdst = kDestination,
                                                            .src0 = 256 + kMatrixA,
                                                            .src1 = 256 + kMatrixB,
                                                            .src2 = 256 + kMatrixC});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  // The B0-only M=32 opcode must be gone, replaced by two scaled M=16 forms.
  EXPECT_EQ(
      std::ranges::count_if(
          decoded, [](const auto &inst) { return inst->mnemonic() == "v_wmma_f32_32x16x128_f4"; }),
      0);
  EXPECT_EQ(std::ranges::count_if(decoded,
                                  [](const auto &inst) {
                                    return inst->mnemonic() == "v_wmma_scale_f32_16x16x128_f8f6f4";
                                  }),
            2);

  const auto *target_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t target_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  constexpr uint32_t kScalePrefixOp = 0x35;
  std::vector<size_t> pass_offsets;
  for (size_t i = 0; i < target_word_count; ++i) {
    if (((target_words[i] >> 16) & 0xffu) == kScalePrefixOp)
      pass_offsets.push_back(i);
  }
  ASSERT_EQ(pass_offsets.size(), 2u) << "each half must carry its own scale prefix";

  constexpr uint16_t kInlineZero = 128;
  constexpr uint16_t kVgpr0 = 256;
  for (size_t half = 0; half < pass_offsets.size(); ++half) {
    const size_t prefix = pass_offsets[half];
    ASSERT_LT(prefix + 3, target_word_count);
    // Inline integer zero in a scale source reads as the E8M0 value for 1.0.
    EXPECT_EQ(target_words[prefix + 1] & 0x1ffu, kInlineZero) << "matrix A scale must be neutral";
    EXPECT_EQ((target_words[prefix + 1] >> 9) & 0x1ffu, kInlineZero)
        << "matrix B scale must be neutral";
    EXPECT_EQ((target_words[prefix + 1] >> 18) & 0x1ffu, kVgpr0)
        << "the unused scale SRC2 must encode VGPR0";

    // The matrix body is the A0 M=16 operation with both formats forced to FP4
    // and A, C, and D sliced by eight dwords while B is shared. Read it back as
    // the encoding struct so the whole matrix-B format is checked: the high bit
    // alone leaves opsel_hi free to select formats 5 through 7.
    cdna5::Vop3pMachineInst body{};
    std::memcpy(&body, &target_words[prefix + 2], sizeof(body));
    EXPECT_EQ(body.op, cdna5::kVWmmaF3216x16x128F8f6f4Vop3p);
    EXPECT_EQ(body.opsel, 4u) << "matrix A format must be FP4";
    EXPECT_EQ(body.opsel_hi_2, 1u) << "matrix B format must be FP4";
    EXPECT_EQ(body.opsel_hi, 0u) << "matrix B format must be exactly FP4";

    const uint16_t delta = static_cast<uint16_t>(half * kHalfDwords);
    EXPECT_EQ(body.vdst, kDestination + delta);
    EXPECT_EQ(body.src0, 256u + kMatrixA + delta);
    EXPECT_EQ(body.src1, 256u + kMatrixB) << "matrix B is shared by both halves";
    EXPECT_EQ(body.src2, 256u + kMatrixC + delta);
  }

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

// The upper half slices eight dwords off matrix A, so a base near the top of a
// bank makes the split cross into the next one. That must produce a bank
// transition and restore the entry mode afterwards.
TEST(BinaryTranslatorE2E, Gfx1250Standalone32x16Fp4AdjustsUpperABank) {
  constexpr auto matrix =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                         {.vdst = 96, .src0 = 256 + 252, .src1 = 256 + 32, .src2 = 256 + 48});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  std::vector<size_t> pass_offsets;
  for (size_t i = 0; i < word_count; ++i) {
    if (((words[i] >> 16) & 0xffu) == 0x35u)
      pass_offsets.push_back(i);
  }
  ASSERT_EQ(pass_offsets.size(), 2u);
  // 252 + 8 wraps to 4 in the next bank.
  cdna5::Vop3pMachineInst upper{};
  std::memcpy(&upper, words + pass_offsets[1] + 2, sizeof(upper));
  EXPECT_EQ(upper.src0, 256u + 4u);
  EXPECT_EQ(upper.src1, 256u + 32u) << "matrix B is not sliced and keeps its bank";

  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  std::vector<uint8_t> modes;
  for (size_t index = 0; index < decoded.size(); ++index) {
    if (decoded[index]->mnemonic() != "s_set_vgpr_msb")
      continue;
    ASSERT_NE(decoded[index]->raw_encoding(), nullptr);
    modes.push_back(static_cast<uint8_t>(decoded[index]->raw_encoding()[0] & 0xffu));
    ASSERT_GT(index, 0u);
    EXPECT_EQ(decoded[index - 1]->mnemonic(), "s_wait_xcnt")
        << "each bank change must be preceded by a drain";
  }
  ASSERT_GE(modes.size(), 2u) << "the crossing must emit a transition and a restore";
  EXPECT_EQ(modes.front(), 0x01u) << "the upper half selects the next Src0 bank";
  EXPECT_EQ(modes.back(), 0x00u) << "the entry mode must be restored";

  const auto second = translator.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

// Matrix A and B must be register ranges; the standalone form has no lowering
// for anything else and must leave the object untouched.
TEST(BinaryTranslatorE2E, Gfx1250FailsClosedOnStandalone32x16Fp4NonVgprMatrixForA0) {
  constexpr uint16_t kInlineZero = 128;
  const std::vector<std::pair<const char *, std::array<uint32_t, 2>>> cases = {
      {"matrix A",
       cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                          {.vdst = 96, .src0 = kInlineZero, .src1 = 256 + 32, .src2 = 256 + 48})},
      {"matrix B",
       cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                          {.vdst = 96, .src0 = 256 + 16, .src1 = kInlineZero, .src2 = 256 + 48})},
  };
  for (const auto &[name, matrix] : cases) {
    SCOPED_TRACE(name);
    constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
    auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
        {matrix[0], matrix[1], kGfx1250SEndpgm});
    rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
    rocjitsu::BinaryTranslator translator(
        ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
        gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                                 rocjitsu::ProcessorRevision::Gfx1250A0));
    const auto result = translator.translate(source);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.elf_bytes, image)
        << "a fail-closed translation must leave the object unchanged";
    EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
        result, rocjitsu::DiagnosticKind::Legalization,
        "invalid VGPR source selector at .text byte offset 0"));
  }
}

// An inline C operand names no register range, so neither half may slice it.
TEST(BinaryTranslatorE2E, Gfx1250Standalone32x16Fp4PreservesInlineMatrixCForA0) {
  constexpr uint16_t kInlineZero = 128;
  constexpr auto matrix =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                         {.vdst = 96, .src0 = 256 + 16, .src1 = 256 + 32, .src2 = kInlineZero});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  const auto *words = reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);
  std::vector<size_t> pass_offsets;
  for (size_t i = 0; i < word_count; ++i) {
    if (((words[i] >> 16) & 0xffu) == 0x35u)
      pass_offsets.push_back(i);
  }
  ASSERT_EQ(pass_offsets.size(), 2u);
  for (const size_t prefix : pass_offsets) {
    cdna5::Vop3pMachineInst body{};
    std::memcpy(&body, words + prefix + 2, sizeof(body));
    EXPECT_EQ(body.src2, kInlineZero) << "an inline C operand must be copied to both halves";
  }
}

// Numerical evidence that the M=32 FP4 split preserves values. The encoding
// tests above only prove the right bits are emitted; this one runs both the
// original B0-only M=32 instruction and the translated pair of scaled M=16
// halves on the instruction simulator over identical inputs. The scaled path
// rounds each block dot before applying its neutral E8M0 scales, as required by
// the ISA, while the unscaled path accumulates continuously. The results must
// therefore agree numerically rather than bit-for-bit.
TEST(BinaryTranslatorE2E, Gfx1250Standalone32x16Fp4SplitMatchesUnsplitExecution) {
  // Every operand range stays inside VGPR bank 0 so the lowering needs no
  // s_set_vgpr_msb transitions and the translated body is exactly two pairs.
  constexpr uint16_t kMatrixA = 16; // A occupies 16 dwords, halves at +0/+8.
  constexpr uint16_t kMatrixB = 32; // B occupies 8 dwords and is shared.
  constexpr uint16_t kMatrixC = 48; // C occupies 16 dwords, halves at +0/+8.
  constexpr uint8_t kDestination = 64;
  constexpr uint32_t kMatrixARegs = 16;
  constexpr uint32_t kMatrixBRegs = 8;
  constexpr uint32_t kMatrixCRegs = 16;
  constexpr uint32_t kDestinationRegs = 16;

  constexpr auto matrix =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p, {.vdst = kDestination,
                                                            .src0 = 256 + kMatrixA,
                                                            .src1 = 256 + kMatrixB,
                                                            .src2 = 256 + kMatrixC});
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr auto completion_wait = kGfx1250WmmaCompletionWait;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {matrix[0], matrix[1], kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *text_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t text_word_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  // Collect the translated body, stopping at the terminator.
  std::vector<uint32_t> body_words;
  for (size_t offset = 0; offset < text_word_count;) {
    std::unique_ptr<rocjitsu::Instruction> inst(decode_valid(*decoder, text_words + offset));
    ASSERT_NE(inst, nullptr) << "translated word " << offset << " failed to decode";
    if (std::string_view(inst->mnemonic()) == "s_endpgm")
      break;
    EXPECT_NE(std::string_view(inst->mnemonic()), "s_set_vgpr_msb")
        << "bank-0 operands must not need a mode transition";
    const size_t words = static_cast<size_t>(inst->size()) / sizeof(uint32_t);
    ASSERT_GT(words, 0u);
    body_words.insert(body_words.end(), text_words + offset, text_words + offset + words);
    offset += words;
  }
  ASSERT_EQ(body_words.size(), 9u)
      << "expected two four-word VOP3PX2 pairs and one completion wait";
  EXPECT_EQ(body_words.back(), completion_wait[0]);
  body_words.pop_back();

  rocjitsu::amdgpu::GpuMemory gpu_mem("gfx1250_fp4_split_diff_mem");
  rocjitsu::amdgpu::L2Cache l2("gfx1250_fp4_split_diff_l2");
  rocjitsu::amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu = rocjitsu::amdgpu::ComputeUnitCore::create("gfx1250", cfg, &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec((1ULL << wf->wf_size()) - 1ULL);
  const uint32_t vb = wf->vgpr_alloc().base;
  const uint32_t lanes = wf->wf_size();

  // Deterministic inputs. A and B are packed FP4, so raw pseudo-random dwords
  // are all valid. C holds FP32 accumulators, so the exponent is confined to a
  // wide but finite range and the interesting encodings are injected by hand.
  auto fill_inputs = [&] {
    uint32_t state = 0x1234abcdu;
    auto next = [&state] {
      state = state * 1664525u + 1013904223u;
      return state;
    };
    for (uint32_t reg = 0; reg < kMatrixARegs; ++reg)
      for (uint32_t lane = 0; lane < lanes; ++lane)
        cu->write_vgpr(vb + kMatrixA + reg, lane, next());
    for (uint32_t reg = 0; reg < kMatrixBRegs; ++reg)
      for (uint32_t lane = 0; lane < lanes; ++lane)
        cu->write_vgpr(vb + kMatrixB + reg, lane, next());
    for (uint32_t reg = 0; reg < kMatrixCRegs; ++reg) {
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        const uint32_t bits = next();
        const uint32_t exponent = ((bits >> 23u) % 60u) + 100u;
        cu->write_vgpr(vb + kMatrixC + reg, lane, (bits & 0x807fffffu) | (exponent << 23u));
      }
    }
    // qNaN, -Inf, +Inf, smallest denormal, -0.0, +0.0.
    static constexpr std::array<uint32_t, 6> kSpecials = {0x7fc00000u, 0xff800000u, 0x7f800000u,
                                                          0x00000001u, 0x80000000u, 0x00000000u};
    // Placed in both halves of C. Each half is accumulated by a different
    // instruction in the translated form, so confining these to one half would
    // leave the other never seeing them.
    for (uint32_t i = 0; i < kSpecials.size(); ++i) {
      cu->write_vgpr(vb + kMatrixC + i, i * 5u, kSpecials[i]);
      cu->write_vgpr(vb + kMatrixC + 8u + i, i * 5u, kSpecials[i]);
    }
  };

  auto zero_destination = [&] {
    for (uint32_t reg = 0; reg < kDestinationRegs; ++reg)
      for (uint32_t lane = 0; lane < lanes; ++lane)
        cu->write_vgpr(vb + kDestination + reg, lane, 0u);
  };

  auto snapshot_destination = [&] {
    std::vector<uint32_t> out;
    out.reserve(kDestinationRegs * lanes);
    for (uint32_t reg = 0; reg < kDestinationRegs; ++reg)
      for (uint32_t lane = 0; lane < lanes; ++lane)
        out.push_back(cu->read_vgpr(vb + kDestination + reg, lane));
    return out;
  };

  // Run A: the untranslated B0 M=32 instruction.
  fill_inputs();
  zero_destination();
  {
    const std::array<uint32_t, 2> words = {matrix[0], matrix[1]};
    std::unique_ptr<rocjitsu::Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(std::string_view(inst->mnemonic()), "v_wmma_f32_32x16x128_f4");
    cu->execute_instruction(inst.get(), *wf);
  }
  const std::vector<uint32_t> run_a = snapshot_destination();

  // Both halves of the destination must have been written, or a split that
  // silently dropped one of them would still compare equal.
  const size_t half_size = 8u * lanes;
  ASSERT_TRUE(std::ranges::any_of(run_a.begin(), run_a.begin() + half_size, [](uint32_t v) {
    return v != 0u;
  })) << "the unsplit run left the lower destination half empty";
  ASSERT_TRUE(std::ranges::any_of(run_a.begin() + half_size, run_a.end(), [](uint32_t v) {
    return v != 0u;
  })) << "the unsplit run left the upper destination half empty";

  // Run B: the translated pair of scaled M=16 halves over the same inputs.
  fill_inputs();
  zero_destination();
  size_t executed = 0;
  for (size_t offset = 0; offset < body_words.size();) {
    std::unique_ptr<rocjitsu::Instruction> inst(decode_valid(*decoder, body_words.data() + offset));
    ASSERT_NE(inst, nullptr);
    cu->execute_instruction(inst.get(), *wf);
    offset += static_cast<size_t>(inst->size()) / sizeof(uint32_t);
    ++executed;
  }
  EXPECT_EQ(executed, 2u);
  const std::vector<uint32_t> run_b = snapshot_destination();

  auto ordered_float_bits = [](uint32_t bits) {
    return (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
  };
  for (size_t i = 0; i < run_a.size(); ++i) {
    if (run_a[i] == run_b[i])
      continue;
    const float unsplit = std::bit_cast<float>(run_a[i]);
    const float split = std::bit_cast<float>(run_b[i]);
    if (std::isnan(unsplit) && std::isnan(split))
      continue;
    if (unsplit == split)
      continue;
    ASSERT_TRUE(std::isfinite(unsplit) && std::isfinite(split))
        << "D reg " << (i / lanes) << " lane " << (i % lanes) << ": unsplit=0x" << std::hex
        << run_a[i] << " split=0x" << run_b[i] << std::dec;
    const uint32_t unsplit_ordered = ordered_float_bits(run_a[i]);
    const uint32_t split_ordered = ordered_float_bits(run_b[i]);
    const uint32_t ulps = unsplit_ordered > split_ordered ? unsplit_ordered - split_ordered
                                                          : split_ordered - unsplit_ordered;
    // Reassociating a 128-term dot at the ISA's block boundaries can move a
    // finite result by several dozen ULPs, especially after cancellation.
    EXPECT_LE(ulps, 64u) << "D reg " << (i / lanes) << " lane " << (i % lanes) << ": unsplit=0x"
                         << std::hex << run_a[i] << " split=0x" << run_b[i] << std::dec;
  }

  if (!wf->is_halted())
    wf->halt();
}

TEST(BinaryTranslatorE2E, Gfx1250RecoversAddNcU64PcBuilder) {
  constexpr uint16_t kPcSreg = 4;
  constexpr uint16_t kGfx1250SAddNcU64Opcode = 83;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint32_t kTargetOffset = 20;
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  const std::vector<uint32_t> words = {
      rocjitsu::build_s_getpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5),
      rocjitsu::build_sop2_encoding(ROCJITSU_CODE_ARCH_CDNA5, kGfx1250SAddNcU64Opcode, kPcSreg,
                                    kPcSreg, kLiteralOperand),
      kTargetOffset - sizeof(uint32_t),
      rocjitsu::build_s_setpc_b64(kPcSreg, ROCJITSU_CODE_ARCH_CDNA5),
      kGfx1250SNop,
      kGfx1250SEndpgm,
  };
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  EXPECT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
}

TEST(BinaryTranslatorE2E, Gfx1250RecoveredBuilderRewriteKeepsTheXcntDrain) {
  // A recovered gfx1250 signed-delta range holds the s_wait_xcnt the compiler
  // put ahead of the pair writes. The canonical builder overwrites the whole
  // range and writes that same pair, so the drain has to be re-emitted in front
  // of it instead of being NOP-filled away with the rest of the range.
  constexpr uint64_t kWord = sizeof(uint32_t);
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kLiteralOperand = 255;
  std::vector<uint8_t> text(8 * kWord, 0);
  rocjitsu::KernelTextLayout layout;
  layout.body_end = text.size();
  layout.blocks.push_back(
      {.source_start = 0, .source_end = 4 * kWord, .target_start = 0, .target_end = 4 * kWord});
  layout.blocks.push_back({.source_start = 4 * kWord,
                           .source_end = 8 * kWord,
                           .target_start = 4 * kWord,
                           .target_end = 8 * kWord});
  layout.recovered_builder_fixups.push_back({.source_target_offset = 4 * kWord,
                                             .source_call_sreg = kPcSreg,
                                             .source_builder_sreg = kPcSreg,
                                             .source_requires_xcnt_drain = true,
                                             .target_getpc_offset = 0,
                                             .target_recovery_begin_offset = kWord,
                                             .target_recovery_end_offset = 4 * kWord});

  const auto patched =
      rocjitsu::patch_recovered_builder_fixups(text, layout, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(patched.ok) << patched.message;

  std::array<uint32_t, 3> replacement{};
  std::memcpy(replacement.data(), text.data() + kWord, replacement.size() * sizeof(uint32_t));
  EXPECT_EQ(replacement[0], rocjitsu::build_s_wait_xcnt(ROCJITSU_CODE_ARCH_CDNA5).value());
  EXPECT_EQ(replacement[1],
            rocjitsu::build_sop2_encoding(ROCJITSU_CODE_ARCH_CDNA5, cdna5::kSAddNcU64Sop2, kPcSreg,
                                          kPcSreg, kLiteralOperand));
  // s_get_pc_i64 leaves the address of the next instruction, and the target is
  // the block at 4 * kWord.
  EXPECT_EQ(replacement[2], 3 * kWord);
}

TEST(BinaryTranslatorE2E, Gfx1250RecoveredBuilderRewriteOmitsAnUnneededXcntDrain) {
  constexpr uint64_t kWord = sizeof(uint32_t);
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kLiteral64Operand = 254;
  std::vector<uint8_t> text(8 * kWord, 0);
  rocjitsu::KernelTextLayout layout;
  layout.body_end = text.size();
  layout.blocks.push_back(
      {.source_start = 0, .source_end = 4 * kWord, .target_start = 0, .target_end = 4 * kWord});
  layout.blocks.push_back({.source_start = 4 * kWord,
                           .source_end = 8 * kWord,
                           .target_start = 4 * kWord,
                           .target_end = 8 * kWord});
  layout.recovered_builder_fixups.push_back({.source_target_offset = 4 * kWord,
                                             .source_call_sreg = kPcSreg,
                                             .source_builder_sreg = kPcSreg,
                                             .target_getpc_offset = 0,
                                             .target_recovery_begin_offset = kWord,
                                             .target_recovery_end_offset = 4 * kWord});

  const auto patched =
      rocjitsu::patch_recovered_builder_fixups(text, layout, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(patched.ok) << patched.message;

  // The point of this case is that no XCNT drain precedes the add -- the first word of the range
  // is the add itself. The literal width is incidental to that, but it is asserted exactly rather
  // than loosely: the rewrite deliberately emits the literal64 form so the relocation lattice,
  // which models only that encoding, can still see this builder on a later translation pass.
  uint32_t first = 0;
  std::memcpy(&first, text.data() + kWord, sizeof(first));
  EXPECT_EQ(first, rocjitsu::build_sop2_encoding(ROCJITSU_CODE_ARCH_CDNA5, cdna5::kSAddNcU64Sop2,
                                                 kPcSreg, kPcSreg, kLiteral64Operand));
}

TEST(BinaryTranslatorE2E, Gfx1250RecoveredBuilderRewriteNamesTheBuilderPairNotTheConsumerPair) {
  // A lane-banked dispatcher builds the callee address in a scratch pair, stashes it through
  // v_writelane_b32, and restores it into a different pair before the transfer. The rewrite
  // regenerates only the delta half and leaves the original s_get_pc_i64 in place, so it has to
  // add into the pair that getpc wrote. Naming the consumer's pair instead would add the delta to
  // whatever that register happened to hold, leave the stash carrying a bare PC, and pair the
  // fresh add with an unrelated earlier getpc that the next pass reads as a code address.
  constexpr uint64_t kWord = sizeof(uint32_t);
  constexpr uint16_t kBuilderSreg = 4;
  constexpr uint16_t kConsumerSreg = 56;
  constexpr uint16_t kLiteral64Operand = 254;
  std::vector<uint8_t> text(8 * kWord, 0);
  rocjitsu::KernelTextLayout layout;
  layout.body_end = text.size();
  layout.blocks.push_back(
      {.source_start = 0, .source_end = 4 * kWord, .target_start = 0, .target_end = 4 * kWord});
  layout.blocks.push_back({.source_start = 4 * kWord,
                           .source_end = 8 * kWord,
                           .target_start = 4 * kWord,
                           .target_end = 8 * kWord});
  layout.recovered_builder_fixups.push_back({.source_target_offset = 4 * kWord,
                                             .source_call_sreg = kConsumerSreg,
                                             .source_builder_sreg = kBuilderSreg,
                                             .target_getpc_offset = 0,
                                             .target_recovery_begin_offset = kWord,
                                             .target_recovery_end_offset = 4 * kWord});

  const auto patched =
      rocjitsu::patch_recovered_builder_fixups(text, layout, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(patched.ok) << patched.message;

  uint32_t first = 0;
  std::memcpy(&first, text.data() + kWord, sizeof(first));
  EXPECT_EQ(first, rocjitsu::build_sop2_encoding(ROCJITSU_CODE_ARCH_CDNA5, cdna5::kSAddNcU64Sop2,
                                                 kBuilderSreg, kBuilderSreg, kLiteral64Operand));
  EXPECT_NE(first, rocjitsu::build_sop2_encoding(ROCJITSU_CODE_ARCH_CDNA5, cdna5::kSAddNcU64Sop2,
                                                 kConsumerSreg, kConsumerSreg, kLiteral64Operand));
}

TEST(BinaryTranslatorE2E, RecoveredBuilderReuseRejectsDisagreeingBuilderPair) {
  // The replacement is an add written against the builder's own SGPR pair, so two consumers that
  // share a range while naming different pairs need different words. Reuse compares a key rather
  // than the words themselves, so the pair has to be part of that key: without it the second
  // consumer silently inherits an add against the first consumer's register. A lane-banked
  // dispatcher produces exactly this shape, because its producer and consumer pairs differ.
  constexpr uint64_t kWord = sizeof(uint32_t);
  constexpr uint16_t kConsumerSreg = 8;
  constexpr uint16_t kBuilderSregA = 4;
  constexpr uint16_t kBuilderSregB = 56;
  std::vector<uint8_t> text(8 * kWord, 0);
  rocjitsu::KernelTextLayout layout;
  layout.body_end = text.size();
  layout.blocks.push_back(
      {.source_start = 0, .source_end = 4 * kWord, .target_start = 0, .target_end = 4 * kWord});
  layout.blocks.push_back({.source_start = 4 * kWord,
                           .source_end = 8 * kWord,
                           .target_start = 4 * kWord,
                           .target_end = 8 * kWord});
  layout.recovered_builder_fixups.push_back({.source_target_offset = 4 * kWord,
                                             .source_call_sreg = kConsumerSreg,
                                             .source_builder_sreg = kBuilderSregA,
                                             .target_getpc_offset = 0,
                                             .target_recovery_begin_offset = kWord,
                                             .target_recovery_end_offset = 4 * kWord});
  layout.recovered_builder_fixups.push_back({.source_target_offset = 4 * kWord,
                                             .source_call_sreg = kConsumerSreg,
                                             .source_builder_sreg = kBuilderSregB,
                                             .target_getpc_offset = 0,
                                             .target_recovery_begin_offset = kWord,
                                             .target_recovery_end_offset = 4 * kWord});

  const auto patched =
      rocjitsu::patch_recovered_builder_fixups(text, layout, ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_FALSE(patched.ok);
  EXPECT_NE(patched.message.find("incompatible replacements"), std::string::npos)
      << patched.message;
}

TEST(BinaryTranslatorE2E, RecoveredBuilderReuseRejectsDisagreeingXcntDrain) {
  // Only the first consumer of a shared range writes the replacement, so a
  // second consumer that wants different words must not be silently dropped.
  constexpr uint64_t kWord = sizeof(uint32_t);
  constexpr uint16_t kPcSreg = 8;
  std::vector<uint8_t> text(8 * kWord, 0);
  rocjitsu::KernelTextLayout layout;
  layout.body_end = text.size();
  layout.blocks.push_back(
      {.source_start = 0, .source_end = 4 * kWord, .target_start = 0, .target_end = 4 * kWord});
  layout.blocks.push_back({.source_start = 4 * kWord,
                           .source_end = 8 * kWord,
                           .target_start = 4 * kWord,
                           .target_end = 8 * kWord});
  layout.recovered_builder_fixups.push_back({.source_target_offset = 4 * kWord,
                                             .source_call_sreg = kPcSreg,
                                             .source_builder_sreg = kPcSreg,
                                             .target_getpc_offset = 0,
                                             .target_recovery_begin_offset = kWord,
                                             .target_recovery_end_offset = 4 * kWord});
  layout.recovered_builder_fixups.push_back({.source_target_offset = 4 * kWord,
                                             .source_call_sreg = kPcSreg,
                                             .source_builder_sreg = kPcSreg,
                                             .source_requires_xcnt_drain = true,
                                             .target_getpc_offset = 0,
                                             .target_recovery_begin_offset = kWord,
                                             .target_recovery_end_offset = 4 * kWord});

  const auto patched =
      rocjitsu::patch_recovered_builder_fixups(text, layout, ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_FALSE(patched.ok);
  EXPECT_NE(patched.message.find("incompatible replacements"), std::string::npos)
      << patched.message;
}

TEST(BinaryTranslatorE2E, Gfx1250IgnoresUndecodablePaddingAfterTerminator) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {kGfx1250SEndpgm, 0, kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  EXPECT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
}

TEST(BinaryTranslatorE2E, Gfx1250TerminatesFallthroughIntoZeroPadding) {
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {kGfx1250SNop, 0, kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  // Only the first two instructions are pinned: the relocated body may be followed by target-side
  // alignment padding, which decodes as additional instructions.
  ASSERT_GE(decoded.size(), 2u);
  EXPECT_EQ(decoded[0]->mnemonic(), "s_nop");
  EXPECT_EQ(decoded[1]->mnemonic(), "s_endpgm");
  // The synthesized boundary must stay attributable: pin the offset as well as the message so a
  // silent early exit cannot lose its diagnostic while the output bytes still match.
  EXPECT_TRUE(rocjitsu::test_support::has_warning_at(result, rocjitsu::DiagnosticKind::Legalization,
                                                     "synthesized s_endpgm boundary",
                                                     /*guest_offset=*/4));
}

// On this target an s_setreg* naming MODE requires two V_NOPs immediately ahead of it to be ordered
// against the instructions that precede it, and the compiler emits none. The requirement is
// specific to MODE, so a write naming any other hardware register must stay untouched. Both
// spellings carry the selector the same way, so both are covered: the immediate form (opcode 19,
// which carries a literal) and the register-source form (opcode 18, which does not).
TEST(BinaryTranslatorE2E, Gfx1250SeparatesModeWriteWithLeadingVNops) {
  // s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 0, 32), 0 -- register id 1.
  constexpr uint32_t kSetregWaveMode = 0xb980f801u;
  // The same write naming HW_REG_WAVE_STATUS -- register id 2.
  constexpr uint32_t kSetregWaveStatus = 0xb980f802u;
  // s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 0, 32), s0 -- register id 1, no literal.
  constexpr uint32_t kSetregB32WaveMode = 0xb900f801u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {kSetregWaveMode, 0u, kSetregWaveStatus, 0u, kSetregB32WaveMode, kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_EQ(decoded.size(), 8u);
  EXPECT_EQ(decoded[0]->mnemonic(), "v_nop_e32");
  EXPECT_EQ(decoded[1]->mnemonic(), "v_nop_e32");
  ASSERT_NE(decoded[2]->raw_encoding(), nullptr);
  EXPECT_EQ(decoded[2]->raw_encoding()[0], kSetregWaveMode)
      << "the MODE write itself must be preserved, only separated";
  // The status write follows the MODE write with no V_NOPs of its own, which is what proves the
  // rule discriminates on the hardware-register selector rather than the mnemonic.
  ASSERT_NE(decoded[3]->raw_encoding(), nullptr);
  EXPECT_EQ(decoded[3]->raw_encoding()[0], kSetregWaveStatus);
  // The register-source form is separated on the same terms. Its predecessor is the status write
  // rather than a V_NOP, so it receives a fresh pair rather than counting anything.
  EXPECT_EQ(decoded[4]->mnemonic(), "v_nop_e32");
  EXPECT_EQ(decoded[5]->mnemonic(), "v_nop_e32");
  ASSERT_NE(decoded[6]->raw_encoding(), nullptr);
  EXPECT_EQ(decoded[6]->mnemonic(), "s_setreg_b32");
  EXPECT_EQ(decoded[6]->raw_encoding()[0], kSetregB32WaveMode);
  EXPECT_EQ(decoded[7]->mnemonic(), "s_endpgm");

  // Separation already present must be counted, or a repeated translation would keep prepending.
  rocjitsu::BinaryTranslator verifier(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto second = verifier.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

// Counting V_NOPs that are already present is only sound inside one basic block: a V_NOP that a
// branch can jump past does not run before the write on every path that reaches it. The write here
// is a branch target, so its textual predecessors are in another block and must not be counted,
// even though they are canonical V_NOPs sitting immediately in front of it.
TEST(BinaryTranslatorE2E, Gfx1250IgnoresLeadingVNopsAcrossABlockBoundary) {
  constexpr uint32_t kSetregWaveMode = 0xb980f801u;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const uint32_t v_nop = cdna5::build_vop1(cdna5::kVNopVop1)[0];

  // s_cbranch_scc0 +2 lands on the MODE write, so the two V_NOPs it skips open no path to it.
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 2})[0], v_nop, v_nop, kSetregWaveMode,
       0u, kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_EQ(decoded.size(), 7u)
      << "the write must receive a fresh pair, not count the skipped ones";
  EXPECT_EQ(decoded[0]->mnemonic(), "s_cbranch_scc0");
  for (size_t i = 1; i < 5; ++i)
    EXPECT_EQ(decoded[i]->mnemonic(), "v_nop_e32") << "at index " << i;
  ASSERT_NE(decoded[5]->raw_encoding(), nullptr);
  EXPECT_EQ(decoded[5]->raw_encoding()[0], kSetregWaveMode);
  EXPECT_EQ(decoded[6]->mnemonic(), "s_endpgm");

  // The branch must land on the separation rather than on the bare write, or the path through it
  // would reach the write with no V_NOPs in front at all -- the case this rule exists to prevent.
  ASSERT_NE(decoded[0]->raw_encoding(), nullptr);
  const size_t landing_word =
      1 + static_cast<size_t>(static_cast<int16_t>(decoded[0]->raw_encoding()[0] & 0xffffu));
  size_t word = 0;
  size_t landing_index = 0;
  while (landing_index < decoded.size() && word < landing_word) {
    word += static_cast<size_t>(decoded[landing_index]->size()) / sizeof(uint32_t);
    ++landing_index;
  }
  ASSERT_EQ(word, landing_word) << "the branch must land on an instruction boundary";
  ASSERT_LT(landing_index + 2, decoded.size());
  EXPECT_EQ(decoded[landing_index]->mnemonic(), "v_nop_e32");
  EXPECT_EQ(decoded[landing_index + 1]->mnemonic(), "v_nop_e32");
  ASSERT_NE(decoded[landing_index + 2]->raw_encoding(), nullptr);
  EXPECT_EQ(decoded[landing_index + 2]->raw_encoding()[0], kSetregWaveMode)
      << "exactly two V_NOPs must separate the branch target from the write";

  // Retargeting moved the write, so this is the case where a second pass could disagree with the
  // first about which V_NOPs belong to the write's own block.
  rocjitsu::BinaryTranslator verifier(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto second = verifier.translate(translated);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, result.elf_bytes);
}

TEST(BinaryTranslatorE2E, Gfx1250TerminatesSetupFallthroughIntoZeroPadding) {
  // clang emits this one-instruction body for rocPRIM target-specialized
  // trampolines whose source ends in __builtin_unreachable(). The following
  // zero is alignment padding, not a second instruction.
  constexpr uint32_t kSetReplayMode = 0xb9800641u;
  constexpr uint32_t kLiteralOne = 1u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {kSetReplayMode, kLiteralOne, 0}, 2);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  // The MODE write opens its block, so its separation rule contributes both V_NOPs here.
  ASSERT_EQ(decoded.size(), 4u)
      << "the synthesized endpgm must be the only instruction after the stub body";
  EXPECT_EQ(decoded[0]->mnemonic(), "v_nop_e32");
  EXPECT_EQ(decoded[1]->mnemonic(), "v_nop_e32");
  ASSERT_NE(decoded[2]->raw_encoding(), nullptr);
  EXPECT_EQ(decoded[2]->mnemonic(), "s_setreg_imm32_b32");
  EXPECT_EQ(decoded[2]->raw_encoding()[0], kSetReplayMode);
  EXPECT_EQ(decoded[2]->raw_encoding()[1], kLiteralOne);
  EXPECT_EQ(decoded[3]->mnemonic(), "s_endpgm");

  const auto kernel_symbol =
      rocjitsu::test_support::find_elf_symbol_for_test(result.elf_bytes, "kernel");
  ASSERT_TRUE(kernel_symbol.has_value());
  EXPECT_EQ(kernel_symbol->st_size, 5 * sizeof(uint32_t))
      << "the translated function extent must include its synthesized terminator";
}

TEST(BinaryTranslatorE2E, Gfx1250TerminatesPrefetchSetupFallthroughIntoZeroPadding) {
  // Newer clang adds the exact gfx1250 prefetch prologue in front of the same
  // unreachable rocPRIM target-specialization body.
  constexpr uint32_t kGlobalPrefetchB8[] = {0xee174000u, 0x00040000u, 0u};
  constexpr uint32_t kVNop = 0x7e000000u;
  constexpr uint32_t kSetReplayMode[] = {0xb9800641u, 1u};
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {kGlobalPrefetchB8[0], kGlobalPrefetchB8[1], kGlobalPrefetchB8[2], kVNop, kSetReplayMode[0],
       kSetReplayMode[1], 0});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto decoded =
      decode_text_instructions(*translated.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  // The prologue's own V_NOP counts toward the MODE write's separation, so it needs
  // only one more rather than a fresh pair.
  ASSERT_EQ(decoded.size(), 5u)
      << "the synthesized endpgm must be the only instruction after the stub body";
  EXPECT_EQ(decoded[0]->mnemonic(), "global_prefetch_b8");
  EXPECT_EQ(decoded[1]->mnemonic(), "v_nop_e32");
  EXPECT_EQ(decoded[2]->mnemonic(), "v_nop_e32");
  EXPECT_EQ(decoded[3]->mnemonic(), "s_setreg_imm32_b32");
  ASSERT_NE(decoded[3]->raw_encoding(), nullptr);
  EXPECT_EQ(decoded[3]->raw_encoding()[0], kSetReplayMode[0]);
  EXPECT_EQ(decoded[3]->raw_encoding()[1], kSetReplayMode[1]);
  EXPECT_EQ(decoded[4]->mnemonic(), "s_endpgm");
}

// A stub whose last instruction ends exactly at the end of `.text` has no padding after it, but it
// is no more terminated than one that does. Inferring the boundary only when the linker happened to
// emit alignment bytes would make the translated tail depend on section layout, so section end is
// treated as the same implicit boundary and the stub still receives its `s_endpgm`.
TEST(BinaryTranslatorE2E, Gfx1250MaterializesSectionFinalClangUnreachableKernelStub) {
  constexpr uint32_t kSetReplayMode = 0xb9800641u;
  constexpr uint32_t kLiteralOne = 1u;
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
      {kSetReplayMode, kLiteralOne});
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto first = translator.translate(source);
  ASSERT_TRUE(first.ok()) << (first.diagnostics.empty() ? "" : first.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject first_output(first.elf_bytes.data(), first.elf_bytes.size());
  const auto decoded =
      decode_text_instructions(*first_output.text_sections()[0], ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(decoded.size(), 4u);
  EXPECT_EQ(decoded[0]->mnemonic(), "v_nop_e32");
  EXPECT_EQ(decoded[1]->mnemonic(), "v_nop_e32");
  EXPECT_EQ(decoded[2]->mnemonic(), "s_setreg_imm32_b32");
  EXPECT_EQ(decoded[3]->mnemonic(), "s_endpgm");
  // Section end is the same inferred boundary as padding, so it must be reported the same way.
  EXPECT_TRUE(rocjitsu::test_support::has_warning_at(first, rocjitsu::DiagnosticKind::Legalization,
                                                     "synthesized s_endpgm boundary",
                                                     /*guest_offset=*/8));

  auto second = translator.translate(first_output);
  ASSERT_TRUE(second.ok()) << (second.diagnostics.empty() ? ""
                                                          : second.diagnostics.front().message);
  EXPECT_EQ(second.elf_bytes, first.elf_bytes);
  // The first pass supplied the terminator, so the second must not infer another boundary.
  EXPECT_FALSE(rocjitsu::test_support::has_warning_at(
      second, rocjitsu::DiagnosticKind::Legalization, "synthesized s_endpgm boundary",
      /*guest_offset=*/8));
}

TEST(BinaryTranslatorE2E, MatchedSemanticExpandRuleFailureIsDiagnostic) {
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject source_layout(image.data(), image.size());
  ASSERT_TRUE(source_layout.is_valid());
  ASSERT_FALSE(source_layout.text_sections().empty());

  const auto words = make_cdna4_bitop3_b16_unsupported_op_sel_words();
  const auto *source_text = source_layout.text_sections()[0];
  ASSERT_EQ(source_text->size(), words.size() * sizeof(uint32_t));
  std::memcpy(image.data() + source_text->sectionOffset(), words.data(),
              words.size() * sizeof(uint32_t));

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  ASSERT_FALSE(result.diagnostics.empty());
  const auto diagnostic = std::ranges::find_if(result.diagnostics, [](const auto &d) {
    return d.kind == rocjitsu::DiagnosticKind::ExpandFailed;
  });
  ASSERT_NE(diagnostic, result.diagnostics.end());
  EXPECT_EQ(diagnostic->severity, rocjitsu::DiagnosticSeverity::Error);
  EXPECT_EQ(diagnostic->guest_offset, std::optional<uint64_t>(0));
  EXPECT_FALSE(diagnostic->message.empty());
}

// Regression: the gfx1250 B0-to-A0 patcher used to zero COMPUTE_PGM_RSRC3 and then write the
// small architectural default into INST_PREF_SIZE, discarding the value the producing compiler
// chose to cover its own kernel body. Shrinking the instruction preload that way turned
// TDM-pipelined MXFP GEMM kernels into intermittent MEMORY_APERTURE_VIOLATION aborts on real
// gfx1250 parts, so a nonzero source request must survive translation untouched.
TEST(BinaryTranslatorE2E, Gfx1250PreservesSourceInstPrefSize) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  // 107 exceeds the 6-bit GFX11 field: GFX12+ widened INST_PREF_SIZE to 8 bits, and reading it
  // through the narrow definition truncates 107 to 43. 25% of the kernels in the gfx1250 Gluon
  // suites request more than 63 units, so this width matters in practice.
  constexpr uint32_t kSourceInstPrefSize = 107u;
  auto image =
      rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text({kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);

  auto descriptor = rocjitsu::test_support::read_kernel_descriptor_for_test(rodata->data());
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE,
                  kSourceInstPrefSize);
  // NAMED_BAR_CNT is a GFX125-only control with no GFX10/GFX11 equivalent. Rebuilding RSRC3
  // used to drop it, which silently removes a kernel's named-barrier allocation.
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT, 5);
  rocjitsu::test_support::write_kernel_descriptor_for_test(image.data() + rodata->sectionOffset(),
                                                           descriptor);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *target_rodata = rocjitsu::test_support::find_section(translated, ".rodata");
  ASSERT_NE(target_rodata, nullptr);
  const auto target =
      rocjitsu::test_support::read_kernel_descriptor_for_test(target_rodata->data());
  EXPECT_EQ(AMDHSA_BITS_GET(target.compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE),
            kSourceInstPrefSize);
  EXPECT_EQ(AMDHSA_BITS_GET(target.compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT), 5u);
}

// Zero is a legitimate request: INST_PREF_SIZE == 0 disables the instruction preload for the
// wave. A same-architecture translation must carry that through rather than substitute the
// architectural default, which would silently re-enable prefetch the producer turned off. The
// default is reserved for sources whose RSRC3 is not in the GFX10+ layout at all (GFX90A-family,
// where those bits are ACCUM_OFFSET) and therefore has to be rebuilt.
TEST(BinaryTranslatorE2E, Gfx1250PreservesDisabledInstPrefSizeFromGfx10PlusSource) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  auto image =
      rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text({kGfx1250SEndpgm});
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);

  auto descriptor = rocjitsu::test_support::read_kernel_descriptor_for_test(rodata->data());
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX10_PLUS_INST_PREF_SIZE, 0);
  rocjitsu::test_support::write_kernel_descriptor_for_test(image.data() + rodata->sectionOffset(),
                                                           descriptor);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *target_rodata = rocjitsu::test_support::find_section(translated, ".rodata");
  ASSERT_NE(target_rodata, nullptr);
  const auto target =
      rocjitsu::test_support::read_kernel_descriptor_for_test(target_rodata->data());
  EXPECT_EQ(AMDHSA_BITS_GET(target.compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX10_PLUS_INST_PREF_SIZE),
            0u);
}

TEST(KernelDescriptorTranslator, Gfx1250UsesWave32SixteenVgprGranularity) {
  using namespace rocr::llvm::amdhsa;

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  const auto *text = rocjitsu::test_support::find_section(layout, ".text");
  ASSERT_NE(rodata, nullptr);
  ASSERT_NE(text, nullptr);

  auto descriptor = rocjitsu::test_support::read_kernel_descriptor_for_test(rodata->data());
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  63);
  // Leave ENABLE_WAVEFRONT_SIZE32 clear to verify that the Wave32-only gfx1250
  // architecture does not misinterpret a legacy producer's missing bit.
  rocjitsu::test_support::write_kernel_descriptor_for_test(image.data() + rodata->sectionOffset(),
                                                           descriptor);

  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA5,
                                                  ROCJITSU_CODE_ARCH_CDNA5);
  const auto translations = translator.translate_image(
      image, text->sectionOffset(), text->size(), rocjitsu::KernelDescriptorTranslationOptions{});

  ASSERT_EQ(translations.size(), 1u);
  EXPECT_TRUE(translations[0].supported);
  EXPECT_EQ(translations[0].guest_wavefront_size, 32);
  EXPECT_EQ(translations[0].target_wave_size, 32);
  EXPECT_EQ(translations[0].guest_vgpr_count, 1024u);
  EXPECT_EQ(translations[0].target_vgpr_count, 1024u);
  EXPECT_EQ(translations[0].target_vgpr_granulated, 63u);
}

TEST(BinaryTranslatorE2E, Gfx1250PreservesSixBitUserSgprCount) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  constexpr uint32_t kUserSgprCount = 32;
  auto image =
      rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text({kGfx1250SEndpgm});
  rocjitsu::test_support::write_value_for_test<uint32_t>(
      image, offsetof(rocjitsu::Elf64_Ehdr, e_flags), rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250);

  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);

  auto descriptor = rocjitsu::test_support::read_kernel_descriptor_for_test(rodata->data());
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2,
                  rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_GFX125_USER_SGPR_COUNT, kUserSgprCount);
  rocjitsu::test_support::write_kernel_descriptor_for_test(image.data() + rodata->sectionOffset(),
                                                           descriptor);

  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto *target_rodata = rocjitsu::test_support::find_section(translated, ".rodata");
  ASSERT_NE(target_rodata, nullptr);
  const auto target =
      rocjitsu::test_support::read_kernel_descriptor_for_test(target_rodata->data());
  EXPECT_EQ(AMDHSA_BITS_GET(target.compute_pgm_rsrc2,
                            rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_GFX125_USER_SGPR_COUNT),
            kUserSgprCount);
}

TEST(KernelDescriptorTranslator, Gfx1250AcceptsLdsAboveTcpPartitionSize) {
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  const auto *text = rocjitsu::test_support::find_section(layout, ".text");
  ASSERT_NE(rodata, nullptr);
  ASSERT_NE(text, nullptr);

  // gfx1250's descriptor limit is 320 KiB per workgroup. A TCP may be
  // configured with a smaller LDS/vector-cache partition, but that partition
  // size is not the architectural descriptor validation limit.
  constexpr uint32_t kStaticLdsBytes = 176128u;
  rocjitsu::test_support::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() +
          offsetof(rocjitsu::test_support::TestKernelDescriptor, group_segment_fixed_size),
      kStaticLdsBytes);

  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA5,
                                                  ROCJITSU_CODE_ARCH_CDNA5);
  const auto translations = translator.translate_image(
      image, text->sectionOffset(), text->size(), rocjitsu::KernelDescriptorTranslationOptions{});

  ASSERT_EQ(translations.size(), 1u);
  EXPECT_TRUE(translations[0].supported);
  EXPECT_EQ(translations[0].target_lds_size, kStaticLdsBytes);
}

TEST(KernelDescriptorTranslator, CdnaToCdnaRejectsOversizedLdsWithoutVirtualization) {
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto *text = rocjitsu::test_support::find_section(layout, ".text");
  ASSERT_NE(text, nullptr);

  rocjitsu::test_support::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() +
          offsetof(rocjitsu::test_support::TestKernelDescriptor, group_segment_fixed_size),
      105600u);

  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_CDNA3);
  const auto translations = translator.translate_image(
      image, text->sectionOffset(), text->size(), rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_EQ(translations.size(), 1u);
  EXPECT_FALSE(translations[0].supported);
  EXPECT_EQ(translations[0].target_lds_size, 105600u);
  EXPECT_FALSE(translations[0].needs_lds_overflow_buf);
  EXPECT_EQ(translations[0].lds_overflow_size, 0u);

  const bool reported_lds_limit =
      std::ranges::any_of(translations[0].diagnostics, [](const auto &diagnostic) {
        return diagnostic.message.find("target LDS size exceeds host per-workgroup limit") !=
               std::string::npos;
      });
  EXPECT_TRUE(reported_lds_limit);
}

TEST(KernelDescriptorTranslator, CdnaToCdnaVirtualizesOversizedStaticLdsDescriptor) {
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto *text = rocjitsu::test_support::find_section(layout, ".text");
  ASSERT_NE(text, nullptr);

  rocjitsu::test_support::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() +
          offsetof(rocjitsu::test_support::TestKernelDescriptor, group_segment_fixed_size),
      105600u);
  auto source_descriptor = rocjitsu::test_support::read_elf_struct_for_test<
      rocjitsu::test_support::TestKernelDescriptor>(image, rodata->sectionOffset());
  uint32_t source_rsrc2 = source_descriptor.compute_pgm_rsrc2;
  AMDHSA_BITS_SET(source_rsrc2, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE, 22);
  rocjitsu::test_support::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() +
          offsetof(rocjitsu::test_support::TestKernelDescriptor, compute_pgm_rsrc2),
      source_rsrc2);

  rocjitsu::KernelDescriptorTranslationOptions options;
  options.virtualize_lds = true;
  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_CDNA3);
  const auto translations =
      translator.translate_image(image, text->sectionOffset(), text->size(), options);
  ASSERT_EQ(translations.size(), 1u);
  EXPECT_TRUE(translations[0].supported);
  EXPECT_EQ(translations[0].target_lds_size, 0u);
  EXPECT_TRUE(translations[0].needs_lds_overflow_buf);
  EXPECT_EQ(translations[0].lds_overflow_size, 105600u);
  EXPECT_EQ(translations[0].kernarg_size, 16u);
  EXPECT_EQ(translations[0].kernarg_wrapper_original_pointer_offset, 16u);
  EXPECT_EQ(translations[0].lds_overflow_kernarg_pointer_offset, 24u);
  EXPECT_EQ(translations[0].target_kernarg_size, 48u);

  rocjitsu::AmdGpuCodeObject mutated(image.data(), image.size());
  ASSERT_TRUE(mutated.is_valid());
  rocjitsu::CodeObjectPatcher patcher(mutated);
  auto patch_plan = translations[0];
  patch_plan.target_entry_text_offset = patch_plan.entry_text_offset;
  patch_plan.target_body_entry_text_offset = patch_plan.entry_text_offset;
  ASSERT_TRUE(patcher.apply_kernel_descriptor_translation(patch_plan, ROCJITSU_CODE_ARCH_CDNA3));

  const auto patched_image = patcher.emit();
  const auto patched_kd = rocjitsu::test_support::read_elf_struct_for_test<
      rocjitsu::test_support::TestKernelDescriptor>(patched_image, rodata->sectionOffset());
  EXPECT_EQ(patched_kd.group_segment_fixed_size, 0u);
  EXPECT_EQ(patched_kd.kernarg_size, 48u);
  EXPECT_EQ(AMDHSA_BITS_GET(patched_kd.compute_pgm_rsrc2,
                            rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE),
            0u);
}

// SHARED_VGPR_COUNT is carried rather than recomputed, so the registers it reserves have to come
// out of the target budget before the allocation is chosen. Planning that ignores them picks an
// allocation the shared blocks no longer fit alongside, and the only way to encode that is to
// take blocks away from a body that still uses them.
TEST(KernelDescriptorTranslator, SharedVgprBlocksAreReservedFromTheTargetVgprBudget) {
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto *text = rocjitsu::test_support::find_section(layout, ".text");
  ASSERT_NE(text, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  // Wave64 is what gives shared VGPR blocks any meaning.
  AMDHSA_BITS_SET(source_kd->kernel_code_properties,
                  rocr::llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 0);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc3,
                  rocr::llvm::amdhsa::COMPUTE_PGM_RSRC3_GFX10_PLUS_SHARED_VGPR_COUNT, 9);

  auto plan = [&](uint32_t minimum_vgprs) {
    rocjitsu::KernelDescriptorTranslationOptions options;
    options.minimum_vgprs = minimum_vgprs;
    rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_RDNA3,
                                                    ROCJITSU_CODE_ARCH_RDNA3);
    return translator.translate_image(image, text->sectionOffset(), text->size(), options);
  };

  // Nine blocks reserve 72 of the 256 registers, leaving 184 for the translated allocation.
  const auto fits = plan(/*minimum_vgprs=*/180);
  ASSERT_EQ(fits.size(), 1u);
  EXPECT_TRUE(fits[0].supported);

  const auto overcommits = plan(/*minimum_vgprs=*/200);
  ASSERT_EQ(overcommits.size(), 1u);
  EXPECT_FALSE(overcommits[0].supported)
      << "the body still needs the shared blocks its descriptor declared";
}

TEST(KernelDescriptorTranslator, VirtualLdsPreservesKernargPreloadRangeWhenSizeIsZero) {
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image, /*kernarg_size=*/0);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto *text = rocjitsu::test_support::find_section(layout, ".text");
  ASSERT_NE(text, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_OFFSET, 2);
  AMDHSA_BITS_SET(source_kd->kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 6);

  rocjitsu::KernelDescriptorTranslationOptions options;
  options.virtualize_lds = true;
  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_CDNA3);
  const auto translations =
      translator.translate_image(image, text->sectionOffset(), text->size(), options);
  ASSERT_EQ(translations.size(), 1u);
  EXPECT_TRUE(translations[0].supported);
  EXPECT_EQ(translations[0].kernarg_size, 32u);
  EXPECT_EQ(translations[0].kernarg_wrapper_original_pointer_offset, 32u);
  EXPECT_EQ(translations[0].lds_overflow_kernarg_pointer_offset, 40u);
  EXPECT_EQ(translations[0].target_kernarg_size, 64u);
}

TEST(KernelDescriptorTranslator, VirtualLdsKeepsOddKernargPreloadCopyExtentExact) {
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image, /*kernarg_size=*/0);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto *text = rocjitsu::test_support::find_section(layout, ".text");
  ASSERT_NE(text, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  AMDHSA_BITS_SET(source_kd->kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_OFFSET, 0);
  AMDHSA_BITS_SET(source_kd->kernarg_preload, rocr::llvm::amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 11);

  rocjitsu::KernelDescriptorTranslationOptions options;
  options.virtualize_lds = true;
  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_CDNA3);
  const auto translations =
      translator.translate_image(image, text->sectionOffset(), text->size(), options);
  ASSERT_EQ(translations.size(), 1u);
  EXPECT_TRUE(translations[0].supported);
  EXPECT_EQ(translations[0].kernarg_size, 44u);
  // Wrapper fields remain aligned independently of the exact source-copy size.
  EXPECT_EQ(translations[0].kernarg_wrapper_original_pointer_offset, 48u);
  EXPECT_EQ(translations[0].lds_overflow_kernarg_pointer_offset, 56u);
  EXPECT_EQ(translations[0].target_kernarg_size, 80u);
}

TEST(KernelDescriptorTranslator, VirtualLdsAcceptsZeroKernargSizeWithWrapper) {
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::enable_kernarg_segment_ptr_sgpr(image, /*kernarg_size=*/0);
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto *text = rocjitsu::test_support::find_section(layout, ".text");
  ASSERT_NE(text, nullptr);

  rocjitsu::test_support::write_value_for_test<uint32_t>(
      image,
      rodata->sectionOffset() +
          offsetof(rocjitsu::test_support::TestKernelDescriptor, group_segment_fixed_size),
      105600u);

  rocjitsu::KernelDescriptorTranslationOptions options;
  options.virtualize_lds = true;
  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_CDNA3);
  const auto translations =
      translator.translate_image(image, text->sectionOffset(), text->size(), options);
  ASSERT_EQ(translations.size(), 1u);
  EXPECT_TRUE(translations[0].supported);
  EXPECT_TRUE(translations[0].needs_lds_overflow_buf);
  EXPECT_EQ(translations[0].kernarg_size, 0u);
  EXPECT_EQ(translations[0].kernarg_wrapper_original_pointer_offset, 0u);
  EXPECT_EQ(translations[0].lds_overflow_kernarg_pointer_offset, 8u);
  EXPECT_EQ(translations[0].target_kernarg_size, 32u);
}

TEST(KernelDescriptorTranslator, VirtualLdsAddsKernargSegmentPointerWhenMissing) {
  using namespace rocr::llvm::amdhsa;

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto *text = rocjitsu::test_support::find_section(layout, ".text");
  ASSERT_NE(text, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  source_kd->group_segment_fixed_size = 105600u;
  source_kd->kernarg_size = 0;
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1);
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 1);
  AMDHSA_BITS_SET(source_kd->kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR,
                  1);

  rocjitsu::KernelDescriptorTranslationOptions options;
  options.virtualize_lds = true;
  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_CDNA3);
  const auto translations =
      translator.translate_image(image, text->sectionOffset(), text->size(), options);
  ASSERT_EQ(translations.size(), 1u);
  EXPECT_TRUE(translations[0].supported);
  EXPECT_TRUE(translations[0].needs_lds_overflow_buf);
  EXPECT_EQ(translations[0].kernarg_size, 0u);
  EXPECT_EQ(translations[0].kernarg_wrapper_original_pointer_offset, 0u);
  EXPECT_EQ(translations[0].lds_overflow_kernarg_pointer_offset, 8u);
  EXPECT_EQ(translations[0].target_kernarg_size, 32u);
  EXPECT_FALSE(translations[0].source_has_kernarg_segment_ptr);
  EXPECT_TRUE(translations[0].has_kernarg_segment_ptr);
  EXPECT_EQ(translations[0].kernarg_segment_ptr_sgpr, 2u);
  EXPECT_EQ(translations[0].source_user_sgpr_count, 2u);
  EXPECT_EQ(translations[0].target_user_sgpr_count, 4u);
  EXPECT_EQ(translations[0].workgroup_id_sgpr_x, 2);
  EXPECT_EQ(translations[0].lds_overflow_workgroup_id_sgpr_x, 4);
  EXPECT_EQ(translations[0].user_sgpr_repair_start, 2u);
  // gfx950 initializes architected FLAT_SCRATCH rather than an additional
  // ordinary SGPR, so only the enabled workgroup-id SGPR needs repair.
  EXPECT_EQ(translations[0].user_sgpr_repair_count, 1u);
  EXPECT_TRUE(translations[0].has_dispatch_ptr);
  EXPECT_EQ(translations[0].dispatch_ptr_sgpr, 0u);
}

TEST(KernelDescriptorTranslator, VirtualLdsRejectsMissingKernargSegmentPointerWithFullUserSgprs) {
  using namespace rocr::llvm::amdhsa;

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = rocjitsu::test_support::find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  const auto *text = rocjitsu::test_support::find_section(layout, ".text");
  ASSERT_NE(text, nullptr);
  ASSERT_GE(rodata->size(), sizeof(rocjitsu::test_support::TestKernelDescriptor));

  auto *source_kd = reinterpret_cast<rocjitsu::test_support::TestKernelDescriptor *>(
      image.data() + rodata->sectionOffset());
  source_kd->group_segment_fixed_size = 105600u;
  source_kd->kernarg_size = 0;
  AMDHSA_BITS_SET(source_kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 15);

  rocjitsu::KernelDescriptorTranslationOptions options;
  options.virtualize_lds = true;
  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_CDNA3);
  const auto translations =
      translator.translate_image(image, text->sectionOffset(), text->size(), options);
  ASSERT_EQ(translations.size(), 1u);
  EXPECT_FALSE(translations[0].supported);
  EXPECT_TRUE(translations[0].needs_lds_overflow_buf);
  const bool reported_user_sgpr_limit =
      std::ranges::any_of(translations[0].diagnostics, [](const auto &diagnostic) {
        return diagnostic.message.find("USER_SGPR_COUNT") != std::string::npos &&
               diagnostic.message.find("16 SGPR") != std::string::npos;
      });
  EXPECT_TRUE(reported_user_sgpr_limit);
}

TEST(KernelDescriptorTranslator, IgnoresNonAllocExecutableSectionsForEntryRange) {
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text();
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  const uint64_t descriptor_vaddr = source.kernel_descriptor_offset("kernel");
  ASSERT_NE(descriptor_vaddr, 0u);
  const auto ehdr =
      rocjitsu::test_support::read_elf_struct_for_test<rocjitsu::Elf64_Ehdr>(image, 0);
  auto shdrs = rocjitsu::test_support::read_elf_array_for_test<rocjitsu::Elf64_Shdr>(
      image, ehdr.e_shoff, ehdr.e_shnum);

  const auto descriptor_section = std::ranges::find_if(shdrs, [&](const auto &section) {
    return (section.sh_flags & rocjitsu::SHF_ALLOC) != 0 &&
           (section.sh_flags & rocjitsu::SHF_EXECINSTR) == 0 &&
           descriptor_vaddr >= section.sh_addr &&
           descriptor_vaddr - section.sh_addr < section.sh_size;
  });
  ASSERT_NE(descriptor_section, shdrs.end());
  const uint64_t descriptor_file_offset =
      descriptor_section->sh_offset + (descriptor_vaddr - descriptor_section->sh_addr);

  constexpr uint64_t fake_exec_vaddr = 0x9000;
  shdrs[5].sh_flags = rocjitsu::SHF_EXECINSTR;
  shdrs[5].sh_addr = fake_exec_vaddr;
  shdrs[5].sh_size = sizeof(uint32_t);
  for (size_t i = 0; i < shdrs.size(); ++i)
    rocjitsu::test_support::write_elf_struct_for_test(
        image, ehdr.e_shoff + i * sizeof(rocjitsu::Elf64_Shdr), shdrs[i]);

  rocjitsu::test_support::write_kernel_descriptor_entry_offset(
      image.data() + descriptor_file_offset,
      static_cast<int64_t>(fake_exec_vaddr) - static_cast<int64_t>(descriptor_vaddr));

  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_RDNA4);
  const auto translations = translator.translate_image(
      image, shdrs[1].sh_offset, shdrs[1].sh_size, rocjitsu::KernelDescriptorTranslationOptions{});
  EXPECT_TRUE(translations.empty())
      << "non-loadable executable sections must not extend valid kernel entry range";
}

namespace {

/// @brief Scalar selector naming the low half of the flat-scratch base.
constexpr uint16_t kFlatScratchBaseLoSelector = 230;
/// @brief Scalar selector naming the high half of the flat-scratch base.
constexpr uint16_t kFlatScratchBaseHiSelector = 231;
/// @brief Dependency wait required before a generated flat-scratch selector read.
constexpr auto kFlatScratchSelectorWait = cdna5::build_sopp(cdna5::kSWaitAluSopp, {.simm16 = 0});

/// @brief Translate a single-instruction gfx1250 kernel with the B0-to-A0 profile.
[[nodiscard]] std::vector<uint32_t>
translate_gfx1250_b0_to_a0_words(std::vector<uint32_t> words,
                                 rocjitsu::TranslationTraceCallback trace_callback = {}) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  words.push_back(kGfx1250SEndpgm);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());

  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  if (trace_callback)
    translator.set_trace_callback(std::move(trace_callback));
  auto result = translator.translate(source);
  EXPECT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  if (!result.ok())
    return {};

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  EXPECT_FALSE(translated.text_sections().empty());
  if (translated.text_sections().empty())
    return {};
  const auto *section = translated.text_sections()[0];
  const auto *target_words = reinterpret_cast<const uint32_t *>(section->data());
  return std::vector<uint32_t>(target_words, target_words + section->size() / sizeof(uint32_t));
}

} // namespace

// A scalar 64-bit read reaches the whole flat-scratch base through either
// selector. The A0 profile spells it with the low selector, so the high
// selector is rewritten in place and the instruction keeps its size.
TEST(BinaryTranslatorE2E, Gfx1250RewritesScalarFlatScratchBase64BitSourceForA0) {
  const auto source = cdna5::build_sop1(
      cdna5::kSMovB64Sop1, {.ssrc0 = static_cast<uint8_t>(kFlatScratchBaseHiSelector), .sdst = 10});
  struct TraceClassification {
    const rocjitsu::InstructionLegalization *legalization;
    bool copied_original;
    bool semantic_lowering;
    bool changed;
  };
  std::optional<TraceClassification> rewrite_trace;
  const auto out = translate_gfx1250_b0_to_a0_words(
      {source[0]}, [&](const rocjitsu::TranslationTraceEvent &event) {
        if (event.source_words.size() != source.size() || event.source_words[0] != source[0])
          return;
        rewrite_trace = {.legalization = event.legalization,
                         .copied_original = event.copied_original,
                         .semantic_lowering = event.semantic_lowering,
                         .changed = event.changed};
      });
  ASSERT_GE(out.size(), 2u);

  EXPECT_EQ(out[0] & 0xffu, kFlatScratchBaseLoSelector)
      << "the scalar read must use the low selector on A0";
  // Only the source selector changes: destination, opcode, and encoding stay.
  EXPECT_EQ(out[0] & ~0xffu, source[0] & ~0xffu);
  EXPECT_EQ(out.size(), 2u) << "a scalar rewrite must not add instructions";
  ASSERT_TRUE(rewrite_trace.has_value());
  EXPECT_EQ(rewrite_trace->legalization, nullptr)
      << "registered instruction rewrites are not opcode legalizations";
  EXPECT_FALSE(rewrite_trace->copied_original);
  EXPECT_TRUE(rewrite_trace->semantic_lowering);
  EXPECT_TRUE(rewrite_trace->changed);
}

// A 64-bit read in the first scalar source position of a two-source encoding.
TEST(BinaryTranslatorE2E, Gfx1250RewritesScalarFlatScratchBaseInFirstSourceOfSop2ForA0) {
  const auto source = cdna5::build_sop2(
      cdna5::kSLshlB64Sop2,
      {.ssrc0 = static_cast<uint8_t>(kFlatScratchBaseHiSelector), .ssrc1 = 130, .sdst = 12});
  const auto out = translate_gfx1250_b0_to_a0_words({source[0]});
  ASSERT_GE(out.size(), 2u);

  EXPECT_EQ(out[0] & 0xffu, kFlatScratchBaseLoSelector);
  EXPECT_EQ((out[0] >> 8) & 0xffu, 130u) << "the 32-bit shift source is unaffected";
  EXPECT_EQ(out.size(), 2u);
}

// The second scalar source position is a separate encoding field, so it needs
// its own coverage: a rewrite that only ever touched the first would pass every
// test above.
TEST(BinaryTranslatorE2E, Gfx1250RewritesScalarFlatScratchBaseInSecondSourceOfSop2ForA0) {
  constexpr uint8_t kOrdinaryPair = 20;
  const auto source = cdna5::build_sop2(cdna5::kSAndB64Sop2,
                                        {.ssrc0 = kOrdinaryPair,
                                         .ssrc1 = static_cast<uint8_t>(kFlatScratchBaseHiSelector),
                                         .sdst = 12});
  const auto out = translate_gfx1250_b0_to_a0_words({source[0]});
  ASSERT_GE(out.size(), 2u);

  EXPECT_EQ(out[0] & 0xffu, kOrdinaryPair) << "the first source is untouched";
  EXPECT_EQ((out[0] >> 8) & 0xffu, kFlatScratchBaseLoSelector)
      << "the second source must use the low selector on A0";
  EXPECT_EQ(out.size(), 2u) << "a scalar rewrite must not add instructions";
}

// The two-source scalar compare is the remaining modelled scalar encoding, and
// each of its fields is separate from the ones covered above.
TEST(BinaryTranslatorE2E, Gfx1250RewritesScalarFlatScratchBaseInBothSourcesOfSopcForA0) {
  constexpr uint8_t kOrdinaryPair = 20;
  struct SopcCase {
    const char *name;
    uint8_t ssrc0;
    uint8_t ssrc1;
    uint32_t expected_ssrc0;
    uint32_t expected_ssrc1;
  };
  const SopcCase cases[] = {
      {"first source", static_cast<uint8_t>(kFlatScratchBaseHiSelector), kOrdinaryPair,
       kFlatScratchBaseLoSelector, kOrdinaryPair},
      {"second source", kOrdinaryPair, static_cast<uint8_t>(kFlatScratchBaseHiSelector),
       kOrdinaryPair, kFlatScratchBaseLoSelector},
      {"both sources", static_cast<uint8_t>(kFlatScratchBaseHiSelector),
       static_cast<uint8_t>(kFlatScratchBaseHiSelector), kFlatScratchBaseLoSelector,
       kFlatScratchBaseLoSelector},
  };
  for (const SopcCase &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto source = cdna5::build_sopc(cdna5::kSCmpEqU64Sopc,
                                          {.ssrc0 = test_case.ssrc0, .ssrc1 = test_case.ssrc1});
    const auto out = translate_gfx1250_b0_to_a0_words({source[0]});
    ASSERT_GE(out.size(), 2u);

    EXPECT_EQ(out[0] & 0xffu, test_case.expected_ssrc0);
    EXPECT_EQ((out[0] >> 8) & 0xffu, test_case.expected_ssrc1);
    EXPECT_EQ(out.size(), 2u) << "a scalar rewrite must not add instructions";
  }
}

// The single-source vector encoding is the last modelled layout without
// coverage. Its source is nine bits wide, so it reaches the selector and takes
// the staged pair like the other vector forms.
TEST(BinaryTranslatorE2E, Gfx1250MovesFlatScratchBaseInSingleSourceVectorFormToSgprPairForA0) {
  const auto source =
      cdna5::build_vop1(cdna5::kVMovB64Vop1, {.src0 = kFlatScratchBaseHiSelector, .vdst = 4});
  const auto out = translate_gfx1250_b0_to_a0_words({source[0]});
  ASSERT_GE(out.size(), 4u) << "the rewrite prepends a wait and one scalar move";

  EXPECT_EQ(out[0], kFlatScratchSelectorWait[0]);
  ASSERT_EQ((out[1] >> 23) & 0x1ffu, 381u) << "prologue must be a SOP1 instruction";
  EXPECT_EQ(out[1] & 0xffu, kFlatScratchBaseLoSelector);
  const uint32_t pair_base = (out[1] >> 16) & 0x7fu;
  EXPECT_EQ(pair_base % 2u, 0u) << "a 64-bit scalar operand must be even-aligned";
  EXPECT_EQ(out[2] & 0x1ffu, pair_base) << "the vector source must name the borrowed pair";
}

// Two affected positions in one instruction share a single borrowed pair. This
// is the only direct check that the prologue is emitted once rather than once
// per rewritten source.
TEST(BinaryTranslatorE2E, Gfx1250SharesOneBorrowedPairAcrossTwoVectorSourcesForA0) {
  const auto source = cdna5::build_vop3(
      cdna5::kVAddNcU64Vop3,
      {.vdst = 0, .src0 = kFlatScratchBaseHiSelector, .src1 = kFlatScratchBaseHiSelector});
  const auto out = translate_gfx1250_b0_to_a0_words({source[0], source[1]});
  ASSERT_GE(out.size(), 5u);

  EXPECT_EQ(out[0], kFlatScratchSelectorWait[0]);
  ASSERT_EQ((out[1] >> 23) & 0x1ffu, 381u) << "prologue must be a SOP1 instruction";
  EXPECT_EQ(out[1] & 0xffu, kFlatScratchBaseLoSelector);
  const uint32_t pair_base = (out[1] >> 16) & 0x7fu;

  EXPECT_EQ(out[3] & 0x1ffu, pair_base) << "src0 must name the borrowed pair";
  EXPECT_EQ((out[3] >> 9) & 0x1ffu, pair_base) << "src1 must name the same pair";
  EXPECT_EQ(out.size(), 5u) << "one move serves both positions, so only one is emitted";
}

// The third source is the last modelled field and the only one no other case
// reaches, so a rewrite that walked just the first two would pass every test
// above.
TEST(BinaryTranslatorE2E, Gfx1250MovesFlatScratchBaseInThirdVectorSourceToSgprPairForA0) {
  const auto source = cdna5::build_vop3(
      cdna5::kVFmaF64Vop3,
      {.vdst = 0, .src0 = 256 + 2, .src1 = 256 + 4, .src2 = kFlatScratchBaseHiSelector});
  const auto out = translate_gfx1250_b0_to_a0_words({source[0], source[1]});
  ASSERT_GE(out.size(), 5u) << "the rewrite prepends a wait and one scalar move";

  EXPECT_EQ(out[0], kFlatScratchSelectorWait[0]);
  ASSERT_EQ((out[1] >> 23) & 0x1ffu, 381u) << "prologue must be a SOP1 instruction";
  EXPECT_EQ(out[1] & 0xffu, kFlatScratchBaseLoSelector);
  const uint32_t pair_base = (out[1] >> 16) & 0x7fu;
  EXPECT_EQ(pair_base % 2u, 0u) << "a 64-bit scalar operand must be even-aligned";

  EXPECT_EQ(out[3] & 0x1ffu, 256u + 2u) << "src0 is unchanged";
  EXPECT_EQ((out[3] >> 9) & 0x1ffu, 256u + 4u) << "src1 is unchanged";
  EXPECT_EQ((out[3] >> 18) & 0x1ffu, pair_base) << "src2 must name the borrowed pair";
}

// VOP3P carries packed 64-bit sources through nine-bit fields, so it reaches
// the selector exactly as VOP3 does. Leaving it unmodelled would send it to the
// conservative scan, which skips vector-typed operands and would copy the
// instruction through unchanged.
TEST(BinaryTranslatorE2E, Gfx1250MovesVop3pFlatScratchBaseSourceToSgprPairForA0) {
  const auto source = cdna5::build_vop3p(
      cdna5::kVPkMulF32Vop3p, {.vdst = 0, .src0 = kFlatScratchBaseHiSelector, .src1 = 258});
  const auto out = translate_gfx1250_b0_to_a0_words({source[0], source[1]});
  ASSERT_GE(out.size(), 5u) << "the rewrite prepends a wait and one scalar move";

  EXPECT_EQ(out[0], kFlatScratchSelectorWait[0]);
  ASSERT_EQ((out[1] >> 23) & 0x1ffu, 381u) << "prologue must be a SOP1 instruction";
  EXPECT_EQ(out[1] & 0xffu, kFlatScratchBaseLoSelector);
  const uint32_t pair_base = (out[1] >> 16) & 0x7fu;
  EXPECT_EQ(pair_base % 2u, 0u) << "a 64-bit scalar operand must be even-aligned";

  EXPECT_EQ(out[2], source[0]) << "the first word is unchanged";
  EXPECT_EQ(out[3] & 0x1ffu, pair_base) << "src0 must name the borrowed pair";
  EXPECT_EQ((out[3] >> 9) & 0x1ffu, 258u) << "src1 is unchanged";
}

// A decoder may report more sources than the encoding has source fields: an
// accumulate form repeats its destination and a literal takes a position of its
// own. Neither can carry the selector, so their presence must not by itself
// refuse the rewrite.
TEST(BinaryTranslatorE2E, Gfx1250RewritesCompactFormsWithExtraDecodedOperandsForA0) {
  struct ExtraOperandCase {
    const char *name;
    std::vector<uint32_t> words;
  };
  const std::vector<ExtraOperandCase> cases = {
      {"destination alias",
       {cdna5::build_vop2(cdna5::kVFmacF64Vop2,
                          {.src0 = kFlatScratchBaseHiSelector, .vsrc1 = 2})[0]}},
  };
  for (const ExtraOperandCase &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto out = translate_gfx1250_b0_to_a0_words(test_case.words);
    ASSERT_GE(out.size(), 4u) << "the rewrite prepends a wait and one scalar move";

    EXPECT_EQ(out[0], kFlatScratchSelectorWait[0]);
    ASSERT_EQ((out[1] >> 23) & 0x1ffu, 381u) << "prologue must be a SOP1 instruction";
    EXPECT_EQ(out[1] & 0xffu, kFlatScratchBaseLoSelector);
    const uint32_t pair_base = (out[1] >> 16) & 0x7fu;
    EXPECT_EQ(out[2] & 0x1ffu, pair_base) << "src0 must name the borrowed pair";
    EXPECT_EQ((out[2] >> 9) & 0xffu, 2u) << "the VGPR source is unchanged";
  }
}

// With every ordinary scalar register live there is no pair to borrow. A
// non-aliasing two-register destination is overwritten by the guest
// instruction anyway, so stage the two 32-bit halves there instead.
TEST(BinaryTranslatorE2E, Gfx1250StagesFlatScratchBaseInDestinationWhenSgprsAreLive) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const auto scratch_alias_producer =
      cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 0, .sdst = 102});
  const auto vector_read = cdna5::build_vop3(
      cdna5::kVAddNcU64Vop3, {.vdst = 0, .src0 = kFlatScratchBaseHiSelector, .src1 = 256 + 2});
  std::vector<uint32_t> words = {scratch_alias_producer[0], vector_read[0], vector_read[1]};
  // Reading every ordinary scalar register after the rewrite keeps them all
  // live across it, so the allocator has nothing to take.
  for (uint16_t base = 0; base + 1 <= 101; base += 2) {
    words.push_back(cdna5::build_sop2(cdna5::kSAndB32Sop2, {.ssrc0 = static_cast<uint8_t>(base),
                                                            .ssrc1 = static_cast<uint8_t>(base + 1),
                                                            .sdst = 102})[0]);
  }
  words.push_back(kGfx1250SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *section = translated.text_sections()[0];
  const auto *out = reinterpret_cast<const uint32_t *>(section->data());
  const size_t out_words = section->size() / sizeof(uint32_t);
  constexpr auto low =
      cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = kFlatScratchBaseLoSelector, .vdst = 0});
  constexpr auto high =
      cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = kFlatScratchBaseHiSelector, .vdst = 1});
  const auto low_move = std::ranges::find(out, out + out_words, low[0]);
  ASSERT_NE(low_move, out + out_words);
  ASSERT_NE(low_move, out);
  EXPECT_EQ(*std::prev(low_move), kFlatScratchSelectorWait[0])
      << "the selector read must wait for the preceding s102 producer";
  EXPECT_NE(std::ranges::find(low_move + 1, out + out_words, high[0]), out + out_words);

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  bool found_rewritten_add = false;
  for (size_t offset = 0; offset < out_words;) {
    std::unique_ptr<rocjitsu::Instruction> inst(decode_valid(*decoder, out + offset));
    ASSERT_NE(inst, nullptr) << "translated word " << offset << " failed to decode";
    if (std::string_view(inst->mnemonic()) == "v_add_nc_u64") {
      ASSERT_NE(inst->raw_encoding(), nullptr);
      EXPECT_EQ(inst->raw_encoding()[1] & 0x1ffu, 256u)
          << "the wide source must read staged v[0:1]";
      found_rewritten_add = true;
    }
    offset += static_cast<size_t>(inst->size()) / sizeof(uint32_t);
  }
  EXPECT_TRUE(found_rewritten_add);
}

// The source and destination roles both select bank 1. Executing the translated
// sequence proves that the two staging moves and the rewritten source all name
// that physical bank, rather than merely agreeing in the encoded low byte.
TEST(BinaryTranslatorE2E, Gfx1250DestinationStagingExecutesInMatchingNonzeroBank) {
  constexpr uint8_t kMatchingBankOneMode = 0x41;
  constexpr uint64_t kScratchBase = 0x1122334455667788ull;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const auto set_vgpr_msb =
      cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = kMatchingBankOneMode});
  const auto vector_read = cdna5::build_vop3(
      cdna5::kVAddNcU64Vop3, {.vdst = 0, .src0 = kFlatScratchBaseHiSelector, .src1 = 256 + 2});
  std::vector<uint32_t> words = {set_vgpr_msb[0], vector_read[0], vector_read[1]};
  for (uint16_t base = 0; base + 1 <= 101; base += 2) {
    words.push_back(cdna5::build_sop2(cdna5::kSAndB32Sop2, {.ssrc0 = static_cast<uint8_t>(base),
                                                            .ssrc1 = static_cast<uint8_t>(base + 1),
                                                            .sdst = 102})[0]);
  }
  words.push_back(kGfx1250SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);
  ASSERT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_FALSE(translated.text_sections().empty());
  const auto *translated_words =
      reinterpret_cast<const uint32_t *>(translated.text_sections()[0]->data());
  const size_t translated_count = translated.text_sections()[0]->size() / sizeof(uint32_t);

  rocjitsu::amdgpu::GpuMemory gpu_mem("gfx1250_flat_scratch_stage_mem");
  rocjitsu::amdgpu::L2Cache l2("gfx1250_flat_scratch_stage_l2");
  rocjitsu::amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 1024;
  cfg.lds_size_kb = 64;
  auto cu =
      rocjitsu::amdgpu::ComputeUnitCore::create("gfx1250_flat_scratch_stage", cfg, &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);
  wf->set_scratch_base(kScratchBase);
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + 2, 0, 0u);
  cu->write_vgpr(vb + 3, 0, 0u);

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  for (size_t offset = 0; offset < translated_count;) {
    std::unique_ptr<rocjitsu::Instruction> inst(decode_valid(*decoder, translated_words + offset));
    ASSERT_NE(inst, nullptr) << "translated word " << offset << " failed to decode";
    if (std::string_view(inst->mnemonic()) == "s_endpgm")
      break;
    cu->execute_instruction(inst.get(), *wf);
    offset += static_cast<size_t>(inst->size()) / sizeof(uint32_t);
  }

  EXPECT_EQ(cu->read_vgpr(vb + 256, 0), static_cast<uint32_t>(kScratchBase));
  EXPECT_EQ(cu->read_vgpr(vb + 257, 0), static_cast<uint32_t>(kScratchBase >> 32));
  EXPECT_EQ(wf->vgpr_msb_mode(), kMatchingBankOneMode);
}

TEST(BinaryTranslatorE2E, Gfx1250DestinationStagingRejectsMismatchedBanks) {
  constexpr uint8_t kSourceBankOneDestinationBankTwo = 0x81;
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const auto set_vgpr_msb =
      cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = kSourceBankOneDestinationBankTwo});
  const auto vector_read = cdna5::build_vop3(
      cdna5::kVAddNcU64Vop3, {.vdst = 0, .src0 = kFlatScratchBaseHiSelector, .src1 = 256 + 2});
  std::vector<uint32_t> words = {set_vgpr_msb[0], vector_read[0], vector_read[1]};
  for (uint16_t base = 0; base + 1 <= 101; base += 2) {
    words.push_back(cdna5::build_sop2(cdna5::kSAndB32Sop2, {.ssrc0 = static_cast<uint8_t>(base),
                                                            .ssrc1 = static_cast<uint8_t>(base + 1),
                                                            .sdst = 102})[0]);
  }
  words.push_back(kGfx1250SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.dispatchable());
  EXPECT_EQ(result.elf_bytes, image);
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "flat-scratch-base rewrite could not allocate safe temporary storage"));
}

// Destination staging is not safe when another source reads that pair: the
// moves would overwrite the guest input before the original instruction. With
// all SGPRs live as well, the rewrite must fail closed.
TEST(BinaryTranslatorE2E, Gfx1250FailsClosedWhenFlatScratchTemporariesAlias) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  const auto vector_read = cdna5::build_vop3(
      cdna5::kVAddNcU64Vop3, {.vdst = 0, .src0 = kFlatScratchBaseHiSelector, .src1 = 256});
  std::vector<uint32_t> words = {vector_read[0], vector_read[1]};
  for (uint16_t base = 0; base + 1 <= 101; base += 2) {
    words.push_back(cdna5::build_sop2(cdna5::kSAndB32Sop2, {.ssrc0 = static_cast<uint8_t>(base),
                                                            .ssrc1 = static_cast<uint8_t>(base + 1),
                                                            .sdst = 102})[0]);
  }
  words.push_back(kGfx1250SEndpgm);

  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  rocjitsu::BinaryTranslator translator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
  const auto result = translator.translate(source);

  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.dispatchable());
  EXPECT_EQ(result.elf_bytes, image) << "fail-closed must leave the object unchanged";
  EXPECT_TRUE(rocjitsu::test_support::has_error_containing(
      result, rocjitsu::DiagnosticKind::ExpandFailed,
      "flat-scratch-base rewrite could not allocate safe temporary storage"));
}

// An unmodelled encoding carrying a wide vector register whose number equals
// the selector must still be copied; only genuine scalar reads are refused.
TEST(BinaryTranslatorE2E, Gfx1250LeavesWideVgprMatchingSelectorNumberUnchangedForA0) {
  constexpr uint8_t kVgprMatchingSelector = 231;
  // A NULL scalar base makes the address a 64-bit vector pair, which is a
  // source operand of exactly the affected width.
  constexpr uint8_t kNullScalarBase = 124;
  const auto load =
      cdna5::build_vglobal(cdna5::kGlobalLoadB64Vglobal,
                           {.saddr = kNullScalarBase, .vdst = 0, .vaddr = kVgprMatchingSelector});
  const auto out = translate_gfx1250_b0_to_a0_words({load[0], load[1], load[2]});
  ASSERT_GE(out.size(), 4u);
  EXPECT_EQ(out[0], load[0]);
  EXPECT_EQ(out[1], load[1]);
  EXPECT_EQ(out[2], load[2]);
}

// With the low registers live the rewrite must borrow from above them. The
// scalar file is fixed on this target rather than descriptor-allocated, so the
// requirement is that the borrowed pair stays inside the architectural file.
TEST(BinaryTranslatorE2E, Gfx1250BorrowsFlatScratchBasePairAboveLiveRegistersForA0) {
  constexpr uint16_t kHighestOrdinarySgpr = 101;
  const auto vector_read = cdna5::build_vop3(
      cdna5::kVAddNcU64Vop3, {.vdst = 0, .src0 = kFlatScratchBaseHiSelector, .src1 = 256 + 2});
  std::vector<uint32_t> words = {vector_read[0], vector_read[1]};
  // Reading s0..s7 after the rewrite keeps them live across it, so the
  // allocator cannot choose any of them.
  for (uint8_t base = 0; base < 8; base += 2) {
    words.push_back(
        cdna5::build_sop2(cdna5::kSAndB32Sop2,
                          {.ssrc0 = base, .ssrc1 = static_cast<uint8_t>(base + 1), .sdst = 20})[0]);
  }
  const auto out = translate_gfx1250_b0_to_a0_words(words);
  ASSERT_GE(out.size(), 4u);

  EXPECT_EQ(out[0], kFlatScratchSelectorWait[0]);
  ASSERT_EQ((out[1] >> 23) & 0x1ffu, 381u) << "prologue must be a SOP1 instruction";
  EXPECT_EQ(out[1] & 0xffu, kFlatScratchBaseLoSelector);
  const uint32_t pair_base = (out[1] >> 16) & 0x7fu;
  EXPECT_GE(pair_base, 8u) << "s0..s7 are live, so the pair must sit above them";
  EXPECT_EQ(pair_base % 2u, 0u) << "a 64-bit scalar operand must be even-aligned";
  EXPECT_LE(pair_base + 1u, kHighestOrdinarySgpr)
      << "the borrowed pair must stay inside the ordinary scalar file";
  EXPECT_EQ(out[3] & 0x1ffu, pair_base) << "the vector read must name the borrowed pair";
}

// The compact vector formats reach the same rewrite as their VOP3 forms.
TEST(BinaryTranslatorE2E, Gfx1250MovesCompactVectorFlatScratchBaseSourceToSgprPairForA0) {
  struct CompactCase {
    const char *name;
    std::vector<uint32_t> words;
  };
  const std::vector<CompactCase> cases = {
      {"vop2",
       {cdna5::build_vop2(cdna5::kVAddNcU64Vop2,
                          {.src0 = kFlatScratchBaseHiSelector, .vsrc1 = 2})[0]}},
      {"vopc",
       {cdna5::build_vopc(cdna5::kVCmpEqU64Vopc,
                          {.src0 = kFlatScratchBaseHiSelector, .vsrc1 = 2})[0]}},
  };
  for (const CompactCase &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto out = translate_gfx1250_b0_to_a0_words(test_case.words);
    ASSERT_GE(out.size(), 4u) << "the rewrite prepends a wait and one scalar move";

    EXPECT_EQ(out[0], kFlatScratchSelectorWait[0]);
    EXPECT_EQ((out[1] >> 23) & 0x1ffu, 381u) << "prologue must be a SOP1 instruction";
    EXPECT_EQ(out[1] & 0xffu, kFlatScratchBaseLoSelector);
    const uint32_t pair_base = (out[1] >> 16) & 0x7fu;
    EXPECT_EQ(pair_base % 2u, 0u);

    EXPECT_EQ(out[2] & 0x1ffu, pair_base) << "src0 must name the borrowed pair";
    EXPECT_EQ((out[2] >> 9) & 0xffu, 2u) << "the VGPR source is unchanged";
  }
}

// The compact formats decode their second source as a plain VGPR index, so a
// VGPR whose number matches the selector must not be mistaken for it.
TEST(BinaryTranslatorE2E, Gfx1250LeavesCompactVgprMatchingSelectorNumberUnchangedForA0) {
  constexpr uint8_t kVgprMatchingSelector = 231;
  const std::vector<uint32_t> sources = {
      cdna5::build_vop2(cdna5::kVAddNcU64Vop2,
                        {.src0 = 256 + 4, .vsrc1 = kVgprMatchingSelector})[0],
      cdna5::build_vopc(cdna5::kVCmpEqU64Vopc,
                        {.src0 = 256 + 4, .vsrc1 = kVgprMatchingSelector})[0],
  };
  for (const uint32_t source_word : sources) {
    SCOPED_TRACE(source_word);
    const auto out = translate_gfx1250_b0_to_a0_words({source_word});
    ASSERT_GE(out.size(), 2u);
    EXPECT_EQ(out[0], source_word) << "an ordinary VGPR must be copied verbatim";
  }
}

// A literal source reports the literal's own value as its encoding value, so
// the constants 230 and 231 collide numerically with the two selectors. Reading
// the operand instead of the encoding field mistakes such a constant for the
// base, and rewriting the field then erases the marker that says a literal
// follows -- stranding its dword as a standalone illegal instruction.
//
// The first VOP2 case is the RCCL all_reduce_perf kernel that first exposed
// this: `v_mul_u64_e32 v[10:11], 0xe7, v[0:1]`, words 0x541400ff 0x000000e7.
TEST(BinaryTranslatorE2E, Gfx1250LeavesLiteralMatchingSelectorNumberUnchangedForA0) {
  struct LiteralCase {
    const char *name;
    std::vector<uint32_t> words;
  };
  constexpr uint32_t kLiteralSelector = 255;
  constexpr uint32_t kLiteral64Selector = 254;
  const std::vector<LiteralCase> cases = {
      // The reported reproducer, and its twin on the other colliding constant.
      {"vop2 literal 0xe7",
       {cdna5::build_vop2(cdna5::kVMulU64Vop2,
                          {.src0 = kLiteralSelector, .vsrc1 = 0, .vdst = 10})[0],
        kFlatScratchBaseHiSelector}},
      {"vop2 literal 0xe6",
       {cdna5::build_vop2(cdna5::kVMulU64Vop2,
                          {.src0 = kLiteralSelector, .vsrc1 = 0, .vdst = 10})[0],
        kFlatScratchBaseLoSelector}},
      // The scalar path rewrites the selector in place, so a stale literal there
      // keeps the instruction's size and changes only what it reads: a silent
      // wrong answer alongside the orphaned dword.
      {"sop1 literal 0xe7",
       {cdna5::build_sop1(cdna5::kSMovB64Sop1,
                          {.ssrc0 = static_cast<uint8_t>(kLiteralSelector), .sdst = 0})[0],
        kFlatScratchBaseHiSelector}},
      // A 64-bit literal reports only its low dword as the encoding value, so it
      // collides the same way while stranding two dwords rather than one.
      {"vop2 literal64 low dword 0xe7",
       {cdna5::build_vop2(cdna5::kVMulU64Vop2,
                          {.src0 = kLiteral64Selector, .vsrc1 = 0, .vdst = 10})[0],
        kFlatScratchBaseHiSelector, 1u}},
  };
  for (const LiteralCase &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto out = translate_gfx1250_b0_to_a0_words(test_case.words);
    ASSERT_GE(out.size(), test_case.words.size() + 1u);

    for (size_t i = 0; i < test_case.words.size(); ++i) {
      EXPECT_EQ(out[i], test_case.words[i])
          << "word " << i << " of a literal-source instruction must be copied verbatim";
    }
    EXPECT_EQ(out.size(), test_case.words.size() + 1u)
        << "no prologue may be prepended for a literal that is not the selector";
  }
}

// The cases above all put their source fields in word 0. VOP3 puts them in word
// 1 and its literal in word 2, so it is the only layout where the field being
// rewritten and the literal that must survive live in different words.
//
// Both sources here read 231: src0 genuinely names the selector, while src1 is
// an ordinary literal that merely collides with its number. The two must be
// treated differently in a single instruction -- src0 repointed at the borrowed
// pair, src1 and its trailing dword untouched.
TEST(BinaryTranslatorE2E, Gfx1250RewritesVop3SelectorButKeepsCollidingLiteralForA0) {
  constexpr uint32_t kLiteralSelector = 255;
  const auto source =
      cdna5::build_vop3(cdna5::kVMulU64Vop3,
                        {.vdst = 10, .src0 = kFlatScratchBaseHiSelector, .src1 = kLiteralSelector});
  const auto out =
      translate_gfx1250_b0_to_a0_words({source[0], source[1], kFlatScratchBaseHiSelector});
  ASSERT_GE(out.size(), 6u) << "the rewrite prepends a wait and scalar move and keeps four words";

  EXPECT_EQ(out[0], kFlatScratchSelectorWait[0]);
  ASSERT_EQ((out[1] >> 23) & 0x1ffu, 381u) << "prologue must be a SOP1 instruction";
  EXPECT_EQ(out[1] & 0xffu, kFlatScratchBaseLoSelector);
  const uint32_t pair_base = (out[1] >> 16) & 0x7fu;
  EXPECT_EQ(pair_base % 2u, 0u) << "a 64-bit scalar operand must be even-aligned";

  EXPECT_EQ(out[2], source[0]) << "vdst, opcode, and modifiers are unchanged";
  EXPECT_EQ(out[3] & 0x1ffu, pair_base) << "the real selector must name the borrowed pair";
  EXPECT_EQ((out[3] >> 9) & 0x1ffu, kLiteralSelector)
      << "the literal marker must survive, or its dword becomes an instruction";
  EXPECT_EQ(out[4], kFlatScratchBaseHiSelector) << "the literal dword itself is unchanged";
  EXPECT_EQ(out.size(), 6u) << "only the wait and move prologue are added";
}

namespace {

/// @brief Mnemonics reported once per translation for a deferred family, in the
/// order the translation reported them.
///
/// The report is an ordinary translation diagnostic, so it reaches the library
/// callback and the hotswap renderer alongside every other one. Reading it back
/// off the result needs no stream redirection.
[[nodiscard]] std::vector<std::string>
deferred_family_reports(const rocjitsu::TranslatedCodeObject &result) {
  std::vector<std::string> reported;
  for (const auto &diagnostic : result.diagnostics) {
    if (diagnostic.severity == rocjitsu::DiagnosticSeverity::Warning &&
        diagnostic.kind == rocjitsu::DiagnosticKind::Legalization &&
        diagnostic.message.find("target-specific handling is not yet implemented") !=
            std::string::npos) {
      reported.push_back(diagnostic.mnemonic);
    }
  }
  return reported;
}

/// @brief Translate a single-kernel gfx1250 object with the B0-to-A0 profile.
[[nodiscard]] rocjitsu::TranslatedCodeObject
translate_gfx1250_b0_to_a0_result(rocjitsu::BinaryTranslator &translator,
                                  std::vector<uint32_t> words) {
  constexpr uint32_t kGfx1250SEndpgm = 0xBFB00000u;
  words.push_back(kGfx1250SEndpgm);
  auto image = rocjitsu::test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(words);
  rocjitsu::AmdGpuCodeObject source(image.data(), image.size());
  auto result = translator.translate(source);
  EXPECT_TRUE(result.ok()) << (result.diagnostics.empty() ? ""
                                                          : result.diagnostics.front().message);
  return result;
}

[[nodiscard]] rocjitsu::BinaryTranslator make_gfx1250_b0_to_a0_translator() {
  return rocjitsu::BinaryTranslator(
      ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
      gfx1250_revision_options(rocjitsu::ProcessorRevision::Gfx1250B0,
                               rocjitsu::ProcessorRevision::Gfx1250A0));
}

/// @brief An s_monitor_sleep whose A0 handling is deferred.
[[nodiscard]] uint32_t gfx1250_deferred_word() {
  return cdna5::build_sopp(cdna5::kSMonitorSleepSopp, {.simm16 = 1})[0];
}

} // namespace

// A deferred family passes through unchanged and is reported so the gap stays
// visible. The report describes the mnemonic, not the site, so repeating it per
// instruction adds nothing: one RCCL all_reduce run emitted 104,831 copies.
TEST(BinaryTranslatorE2E, Gfx1250ReportsOncePerDeferredMnemonicWithinOneTranslation) {
  const uint32_t deferred = gfx1250_deferred_word();
  auto translator = make_gfx1250_b0_to_a0_translator();
  const auto result = translate_gfx1250_b0_to_a0_result(
      translator, {deferred, deferred, deferred, deferred, deferred});

  EXPECT_EQ(deferred_family_reports(result), (std::vector<std::string>{"s_monitor_sleep"}))
      << "five deferred instructions must produce one diagnostic, not five";
}

// Distinct deferred mnemonics are distinct gaps, so suppressing one must not
// suppress another.
TEST(BinaryTranslatorE2E, Gfx1250ReportsSeparatelyForEachDeferredMnemonic) {
  const uint32_t deferred = gfx1250_deferred_word();
  const uint32_t barrier_state =
      cdna5::build_sop1(cdna5::kSGetBarrierStateSop1, {.ssrc0 = 128, .sdst = 0})[0];
  auto translator = make_gfx1250_b0_to_a0_translator();
  const auto result = translate_gfx1250_b0_to_a0_result(
      translator, {deferred, barrier_state, deferred, barrier_state});

  auto reported = deferred_family_reports(result);
  std::ranges::sort(reported);
  EXPECT_EQ(reported, (std::vector<std::string>{"s_get_barrier_state", "s_monitor_sleep"}));
}

// The suppression is scoped to one translation. Process-wide state would let
// the first code object that uses a deferred mnemonic hide the same gap in
// every object loaded afterwards, which is exactly how HotSwap runs: many code
// objects translated in one process.
TEST(BinaryTranslatorE2E, Gfx1250ReportsAgainForEachSeparateTranslation) {
  const uint32_t deferred = gfx1250_deferred_word();
  auto first_translator = make_gfx1250_b0_to_a0_translator();
  auto second_translator = make_gfx1250_b0_to_a0_translator();
  const auto first = translate_gfx1250_b0_to_a0_result(first_translator, {deferred, deferred});
  const auto second = translate_gfx1250_b0_to_a0_result(second_translator, {deferred, deferred});

  EXPECT_EQ(deferred_family_reports(first), (std::vector<std::string>{"s_monitor_sleep"}));
  EXPECT_EQ(deferred_family_reports(second), (std::vector<std::string>{"s_monitor_sleep"}))
      << "a second code object must report the gap independently";
}

// A translator instance reused across code objects must behave the same way,
// since the state lives on the translator rather than in a local.
TEST(BinaryTranslatorE2E, Gfx1250ReportsAgainWhenOneTranslatorIsReused) {
  const uint32_t deferred = gfx1250_deferred_word();
  auto translator = make_gfx1250_b0_to_a0_translator();
  const auto first = translate_gfx1250_b0_to_a0_result(translator, {deferred, deferred});
  const auto second = translate_gfx1250_b0_to_a0_result(translator, {deferred, deferred});

  EXPECT_EQ(deferred_family_reports(first), (std::vector<std::string>{"s_monitor_sleep"}));
  EXPECT_EQ(deferred_family_reports(second), (std::vector<std::string>{"s_monitor_sleep"}))
      << "translate() must reset the per-translation suppression state";
}

// The report points at the first instruction of the family so a reader can find
// one, and carries the mnemonic so callback consumers can group by gap.
TEST(BinaryTranslatorE2E, Gfx1250DeferredFamilyReportCarriesOffsetAndMnemonic) {
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  const uint32_t deferred = gfx1250_deferred_word();
  auto translator = make_gfx1250_b0_to_a0_translator();
  const auto result =
      translate_gfx1250_b0_to_a0_result(translator, {kGfx1250SNop, deferred, deferred});

  const auto reported = std::ranges::find_if(result.diagnostics, [](const auto &diagnostic) {
    return diagnostic.mnemonic == "s_monitor_sleep";
  });
  ASSERT_NE(reported, result.diagnostics.end());
  EXPECT_EQ(reported->severity, rocjitsu::DiagnosticSeverity::Warning);
  EXPECT_EQ(reported->guest_offset, std::optional<uint64_t>(sizeof(uint32_t)))
      << "the offset must name the first deferred instruction, not the s_nop before it";
}

// s_sleep and s_sleep_var behave identically on A0 and B0. The only sleep-family
// A0 translation requirement is specific to s_monitor_sleep('forever') with
// MWAIT=0. Copying a plain sleep through is therefore the correct translation,
// not an unimplemented one, and reporting it drowned the reports that do name a
// real gap: one RCCL all_reduce run emitted 104,831 of them.
TEST(BinaryTranslatorE2E, Gfx1250CopiesPlainSleepWithoutReportingAGap) {
  const std::vector<uint32_t> sleeps = {
      cdna5::build_sopp(cdna5::kSSleepSopp, {.simm16 = 1})[0],
      cdna5::build_sop1(cdna5::kSSleepVarSop1, {.ssrc0 = 0})[0],
  };
  auto translator = make_gfx1250_b0_to_a0_translator();
  const auto result = translate_gfx1250_b0_to_a0_result(translator, sleeps);

  EXPECT_TRUE(deferred_family_reports(result).empty())
      << "a plain sleep translates correctly and must not be reported as a gap";
  for (const auto &diagnostic : result.diagnostics) {
    EXPECT_EQ(diagnostic.severity, rocjitsu::DiagnosticSeverity::Warning) << diagnostic.message;
  }
}

// A vector 64-bit read cannot name the selector on A0, so the base is moved
// into a dead SGPR pair and the source position is repointed at that pair.
TEST(BinaryTranslatorE2E, Gfx1250MovesVectorFlatScratchBase64BitSourceToSgprPairForA0) {
  const auto source = cdna5::build_vop3(
      cdna5::kVAddNcU64Vop3, {.vdst = 0, .src0 = kFlatScratchBaseHiSelector, .src1 = 256 + 2});
  const auto out = translate_gfx1250_b0_to_a0_words({source[0], source[1]});
  ASSERT_GE(out.size(), 5u) << "the rewrite prepends a wait and one scalar move";

  // The prologue is a 64-bit scalar read of the base through the low selector.
  const auto expected_prologue_op = static_cast<uint32_t>(cdna5::kSMovB64Sop1);
  EXPECT_EQ(out[0], kFlatScratchSelectorWait[0]);
  EXPECT_EQ((out[1] >> 23) & 0x1ffu, 381u) << "prologue must be a SOP1 instruction";
  EXPECT_EQ((out[1] >> 8) & 0xffu, expected_prologue_op);
  EXPECT_EQ(out[1] & 0xffu, kFlatScratchBaseLoSelector);

  const uint32_t pair_base = (out[1] >> 16) & 0x7fu;
  EXPECT_EQ(pair_base % 2u, 0u) << "a 64-bit scalar operand must be even-aligned";
  EXPECT_LE(pair_base + 1u, 105u) << "the borrowed pair must be an ordinary SGPR pair";

  // The vector instruction now reads that pair; its other operands are intact.
  EXPECT_EQ(out[2] & 0xffffu, source[0] & 0xffffu) << "vdst and modifiers are unchanged";
  EXPECT_EQ(out[3] & 0x1ffu, pair_base) << "src0 must name the borrowed pair";
  EXPECT_EQ((out[3] >> 9) & 0x1ffu, 256u + 2u) << "src1 is unchanged";
}
