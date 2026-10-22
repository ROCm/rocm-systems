// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/async_scoreboard.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <format>
#include <memory>
#include <span>
#include <vector>

namespace {
using namespace rocjitsu;
using namespace rocjitsu::amdgpu;

class IsaDiagnosticsTest : public ::testing::TestWithParam<rj_code_arch_t> {
protected:
  void SetUp() override { create(); }

  void create(uint32_t wave_size = 0, unsigned helpers = 0) {
    ComputeUnitCore::Config config{};
    config.arch = GetParam();
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 128;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    config.async_resources = std::make_shared<matrix_coexecution::ExecutionResources>(helpers);
    cu = ComputeUnitCore::create("isa_test_cu", config, &memory, &l2);
    decoder = Decoder::create(GetParam());
    ASSERT_NE(cu, nullptr);
    ASSERT_NE(decoder, nullptr);
    wf = cu->dispatch_wf(7, 0x100000, 106, 256, wave_size);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(~uint64_t{0});
    wf->debug_write_sgpr(0, 0x4000);
    wf->debug_write_sgpr(1, 0);
    wf->debug_write_sgpr(2, 0x1000);
    wf->debug_write_sgpr(3, 0);
  }

  bool gfx12() const {
    return GetParam() == ROCJITSU_CODE_ARCH_RDNA4 || GetParam() == ROCJITSU_CODE_ARCH_CDNA5;
  }

  bool vopd() const {
    return gfx12() || GetParam() == ROCJITSU_CODE_ARCH_RDNA3 ||
           GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5;
  }

  std::array<uint32_t, 2> smem(int32_t immediate, bool buffer = false, uint16_t load_op = 0) {
    uint32_t bits = static_cast<uint32_t>(immediate);
    if (GetParam() == ROCJITSU_CODE_ARCH_CDNA5)
      return cdna5::build_smem(buffer ? 16 : load_op,
                               {.sdata = 16, .ioffset = bits & 0xffffff, .soffset = 4});
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
      return rdna4::build_smem(buffer ? 16 : load_op,
                               {.sdata = 16, .ioffset = bits & 0xffffff, .soffset = 4});
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3)
      return rdna3::build_smem(buffer ? 8 : load_op,
                               {.sdata = 16, .offset = bits & 0x1fffff, .soffset = 4});
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
      return rdna3_5::build_smem(buffer ? 8 : load_op,
                                 {.sdata = 16, .offset = bits & 0x1fffff, .soffset = 4});
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA1)
      return rdna1::build_smem(buffer ? 8 : load_op,
                               {.sdata = 16, .offset = bits & 0x1fffff, .soffset = 4});
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA2)
      return rdna2::build_smem(buffer ? 8 : load_op,
                               {.sdata = 16, .offset = bits & 0x1fffff, .soffset = 4});
    if (GetParam() == ROCJITSU_CODE_ARCH_CDNA1)
      return cdna1::build_smem(
          buffer ? 8 : load_op,
          {.sdata = 16, .soffset_en = 1, .imm = 1, .offset = bits & 0x1fffff, .soffset = 4});
    if (GetParam() == ROCJITSU_CODE_ARCH_CDNA2)
      return cdna2::build_smem(
          buffer ? 8 : load_op,
          {.sdata = 16, .soffset_en = 1, .imm = 1, .offset = bits & 0x1fffff, .soffset = 4});
    return cdna4::build_smem(
        buffer ? 8 : load_op,
        {.sdata = 16, .soffset_en = 1, .imm = 1, .offset = bits & 0x1fffff, .soffset = 4});
  }

  std::array<uint32_t, 2> packed(uint16_t op, uint8_t neg, uint8_t neg_hi) {
    if (GetParam() == ROCJITSU_CODE_ARCH_CDNA5)
      return cdna5::build_vop3p(
          op, {.vdst = 64, .neg_hi = neg_hi, .src0 = 256, .src1 = 288, .src2 = 320, .neg = neg});
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
      return rdna4::build_vop3p(
          op, {.vdst = 64, .neg_hi = neg_hi, .src0 = 256, .src1 = 288, .src2 = 320, .neg = neg});
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
      return rdna3_5::build_vop3p(
          op, {.vdst = 64, .neg_hi = neg_hi, .src0 = 256, .src1 = 288, .src2 = 320, .neg = neg});
    return rdna3::build_vop3p(
        op, {.vdst = 64, .neg_hi = neg_hi, .src0 = 256, .src1 = 288, .src2 = 320, .neg = neg});
  }

  void execute(std::span<const uint32_t> words, unsigned warnings, const char *mnemonic_prefix) {
    std::array<uint32_t, 4> padded{};
    std::ranges::copy(words, padded.begin());
    SCOPED_TRACE(std::format("encoding: {:#x} {:#x} {:#x}", padded[0], padded[1], padded[2]));
    auto decoded = decoder->decode(padded.data());
    ASSERT_TRUE(decoded.succeeded());
    std::unique_ptr<Instruction> inst = std::move(decoded.value());
    ASSERT_TRUE(inst->mnemonic().starts_with(mnemonic_prefix)) << inst->mnemonic();
    const uint64_t before = cu->isa_diagnostic_count();
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
    EXPECT_EQ(cu->isa_diagnostic_count() - before, warnings);
  }

  GpuMemory memory{"isa_test_memory"};
  L2Cache l2{"isa_test_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  Wavefront *wf = nullptr;
};

TEST_P(IsaDiagnosticsTest, ScalarLoadOffsetBoundaryAndCdna5Exception) {
  wf->debug_write_sgpr(4, 0);
  execute(smem(-4), GetParam() == ROCJITSU_CODE_ARCH_CDNA5 ? 0 : 1, "s_load_");
  execute(smem(0), 0, "s_load_");
  wf->debug_write_sgpr(4, 4);
  execute(smem(-4), 0, "s_load_");
  // Register offsets are unsigned; bit 31 is not a negative offset.
  wf->debug_write_sgpr(4, 0x80000000u);
  execute(smem(-4), 0, "s_load_");
}

TEST_P(IsaDiagnosticsTest, BufferImmediateRuleDependsOnArchitecture) {
  wf->debug_write_sgpr(4, 8);
  execute(smem(-4, true), gfx12() ? 1 : 0, "s_buffer_load_");
  execute(smem(0, true), 0, "s_buffer_load_");
  wf->debug_write_sgpr(4, 0);
  // RDNA3 specifies a MEMVIOL; CDNA3/4 buffer offsets are unqualified.
  execute(smem(-4, true), gfx12(), "s_buffer_load_");
  // A positive offset beyond the descriptor is a defined zero-fill load.
  execute(smem(0x2000, true), 0, "s_buffer_load_");
}

TEST_P(IsaDiagnosticsTest, ScalarLoadUnalignedOffsetBoundary) {
  for (auto [immediate, offset] : {std::pair{-1, 3u}, std::pair{-1, 1u}, std::pair{-3, 2u},
                                   std::pair{-3, 4u}, std::pair{-4, 3u}}) {
    wf->debug_write_sgpr(4, offset);
    execute(smem(immediate),
            GetParam() != ROCJITSU_CODE_ARCH_CDNA5 && immediate + static_cast<int64_t>(offset) < 0,
            "s_load_");
  }
}

TEST_P(IsaDiagnosticsTest, CdnaScalarStoreRuleExcludesBufferStores) {
  if (GetParam() != ROCJITSU_CODE_ARCH_CDNA1 && GetParam() != ROCJITSU_CODE_ARCH_CDNA2 &&
      GetParam() != ROCJITSU_CODE_ARCH_CDNA3 && GetParam() != ROCJITSU_CODE_ARCH_CDNA4)
    GTEST_SKIP();
  for (uint16_t op : std::array<uint16_t, 6>{16, 17, 18, 24, 25, 26}) {
    const bool buffer = op >= 24;
    wf->debug_write_sgpr(4, 0);
    execute(smem(-4, false, op), !buffer, buffer ? "s_buffer_store_" : "s_store_");
    wf->debug_write_sgpr(4, 4);
    execute(smem(-4, false, op), 0, buffer ? "s_buffer_store_" : "s_store_");
  }
}

TEST_P(IsaDiagnosticsTest, ScalarByteAndHalfwordOffsets) {
  if (!gfx12())
    GTEST_SKIP();
  for (auto [op, size] : {std::pair{9u, 1u}, std::pair{11u, 2u}}) {
    wf->debug_write_sgpr(4, 0);
    execute(smem(-static_cast<int32_t>(size), false, op),
            GetParam() == ROCJITSU_CODE_ARCH_CDNA5 ? 0 : 1, "s_load_");
    wf->debug_write_sgpr(4, size);
    execute(smem(-static_cast<int32_t>(size), false, op), 0, "s_load_");
  }
}

TEST_P(IsaDiagnosticsTest, CountsSuppressedReports) {
  for (unsigned i = 0; i < 20; ++i)
    wf->report_undefined_behavior("test restriction");
  EXPECT_EQ(cu->isa_diagnostic_count(), 20u);
}

TEST_P(IsaDiagnosticsTest, MfmaBroadcastBlockBoundary) {
  if (GetParam() != ROCJITSU_CODE_ARCH_CDNA3 && GetParam() != ROCJITSU_CODE_ARCH_CDNA4)
    GTEST_SKIP();
  // The 16x16x1 instruction processes four blocks; 16x16x4 processes one.
  for (auto [opcode, max_cbsz] : {std::pair{cdna4::kVMfmaF3216x16x14bF32Vop3pMfma, 2u},
                                  std::pair{cdna4::kVMfmaF3216x16x4F32Vop3pMfma, 0u},
                                  std::pair{cdna4::kVMfmaI3216x16x44bI8Vop3pMfma, 2u}}) {
    for (unsigned cbsz = 0; cbsz <= max_cbsz + 1; ++cbsz) {
      auto words = cdna4::build_vop3p_mfma(
          opcode,
          {.vdst = 32, .cbsz = static_cast<uint8_t>(cbsz), .src0 = 256, .src1 = 257, .src2 = 128});
      execute(words, cbsz > max_cbsz, "v_mfma_");
    }
  }
}

TEST_P(IsaDiagnosticsTest, MfmaF64IgnoresBroadcastFields) {
  if (GetParam() != ROCJITSU_CODE_ARCH_CDNA3 && GetParam() != ROCJITSU_CODE_ARCH_CDNA4)
    GTEST_SKIP();
  execute(cdna4::build_vop3p_mfma(cdna4::kVMfmaF6416x16x4F64Vop3pMfma,
                                  {.vdst = 32, .cbsz = 7, .src0 = 256, .src1 = 258, .src2 = 128}),
          0, "v_mfma_");
}

TEST_P(IsaDiagnosticsTest, MfmaFormatSelectorsAreNotBroadcastGroups) {
  if (GetParam() != ROCJITSU_CODE_ARCH_CDNA4)
    GTEST_SKIP();
  auto words = cdna4::build_vop3p_mfma(
      cdna4::kVMfmaF3216x16x128F8f6f4Vop3pMfma,
      {.vdst = 64, .cbsz = 4, .src0 = 256, .src1 = 288, .src2 = 128, .blgp = 4});
  execute(words, 0, "v_mfma_");
}

TEST_P(IsaDiagnosticsTest, TransposeLoadsRequireFullOrEmptyEffectiveExec) {
  if (!gfx12())
    GTEST_SKIP();
  for (uint32_t wave : {32u, 64u}) {
    if (wave == 64 && GetParam() == ROCJITSU_CODE_ARCH_CDNA5)
      continue;
    create(wave);
    ASSERT_NE(wf, nullptr);
    const uint64_t full = ~uint64_t{0} >> (64 - wave);
    std::vector<uint16_t> ops;
    if (GetParam() == ROCJITSU_CODE_ARCH_CDNA5)
      ops = {cdna5::kGlobalLoadTr16B128Vglobal, cdna5::kGlobalLoadTr8B64Vglobal,
             cdna5::kGlobalLoadTr4B64Vglobal, cdna5::kGlobalLoadTr6B96Vglobal};
    else
      ops = {rdna4::kGlobalLoadTrB128Vglobal, rdna4::kGlobalLoadTrB64Vglobal};
    for (uint16_t op : ops) {
      const auto words = GetParam() == ROCJITSU_CODE_ARCH_CDNA5
                             ? cdna5::build_vglobal(op, {.saddr = 0, .vdst = 16, .vaddr = 0})
                             : rdna4::build_vglobal(op, {.saddr = 0, .vdst = 16, .vaddr = 0});
      for (uint64_t exec : {uint64_t{0}, full, ~uint64_t{0}, uint64_t{1}, full >> 1}) {
        wf->set_exec(exec);
        execute(words, (exec & full) != 0 && (exec & full) != full, "global_load_tr");
      }
    }
  }
}

TEST_P(IsaDiagnosticsTest, AsyncMfmaReportsItsIssuingPc) {
  if (GetParam() != ROCJITSU_CODE_ARCH_CDNA4)
    GTEST_SKIP();
  create(64, 2);
  const uint64_t pc = wf->pc;
  // Independent, helper-eligible MFMAs followed by a branch that drains work.
  for (unsigned i = 0; i < 2; ++i) {
    const auto words = cdna4::build_vop3p_mfma(cdna4::kVMfmaF3216x16x32F16Vop3pMfma,
                                               {.vdst = static_cast<uint8_t>(64 + i * 16),
                                                .cbsz = 1,
                                                .src0 = 256,
                                                .src1 = 288,
                                                .src2 = 128});
    for (unsigned word = 0; word < 2; ++word)
      memory.write32(pc + i * 8 + word * 4, words[word]);
  }
  memory.write32(pc + 16, cdna4::build_sopp(cdna4::kSBranchSopp, {.simm16 = 0xffff})[0]);
  const auto submitted = async_execution::stats.mma;
  testing::internal::CaptureStderr();
  for (unsigned i = 0; i < 4 && wf->pc < pc + 16; ++i)
    cu->step();
  cu->step(); // Drain at the branch before inspecting the result.
  const std::string output = testing::internal::GetCapturedStderr();
  EXPECT_GT(async_execution::stats.mma, submitted);
  EXPECT_EQ(cu->isa_diagnostic_count(), 2u);
  EXPECT_NE(output.find("pc=0x100000:"), std::string::npos) << output;
  EXPECT_NE(output.find("pc=0x100008:"), std::string::npos) << output;
  EXPECT_EQ(output.find("pc=0x100010:"), std::string::npos) << output;
}

TEST_P(IsaDiagnosticsTest, IuModifiersPreserveSignednessAndFloatModifiers) {
  if (!vopd())
    GTEST_SKIP();
  create(32);
  const bool cdna5_target = GetParam() == ROCJITSU_CODE_ARCH_CDNA5;
  const uint16_t wmma =
      cdna5_target ? cdna5::kVWmmaI3216x16x64Iu8Vop3p : rdna4::kVWmmaI3216x16x16Iu8Vop3p;
  const char *wmma_name = cdna5_target ? "v_wmma_i32_16x16x64_iu8" : "v_wmma_i32_16x16x16_iu8";
  for (const auto &[op, name] :
       {std::pair{rdna4::kVDot4I32Iu8Vop3p, "v_dot4_i32_iu8"},
        std::pair{rdna4::kVDot8I32Iu4Vop3p, "v_dot8_i32_iu4"}, std::pair{wmma, wmma_name}}) {
    // CDNA5's explicit undefined-modifier rule names DOT4 and WMMA, not DOT8.
    const bool qualified = !cdna5_target || op != cdna5::kVDot8I32Iu4Vop3p;
    for (uint8_t neg : {0, 1, 2, 3})
      execute(packed(op, neg, 0), 0, name);
    execute(packed(op, 4, 0), qualified, name);
    for (uint8_t neg_hi : {1, 2, 4})
      execute(packed(op, 3, neg_hi), qualified, name);
  }
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
    execute(packed(rdna4::kVWmmaI3216x16x32Iu4Vop3p, 4, 4), 1, "v_wmma_i32_16x16x32_iu4");
    execute(packed(rdna4::kVSwmmacI3216x16x32Iu8Vop3p, 4, 4), 0, "v_swmmac_i32_16x16x32_iu8");
  }
  if (cdna5_target)
    execute(packed(cdna5::kVSwmmacI3216x16x128Iu8Vop3p, 4, 4), 0, "v_swmmac_i32_16x16x128_iu8");
  // Floating packed-math operands retain their defined sign modifiers.
  if (cdna5_target)
    execute(packed(cdna5::kVPkAddF16Vop3p, 4, 4), 0, "v_pk_add_");
  else
    execute(packed(rdna4::kVDot2F32F16Vop3p, 4, 4), 0, "v_dot2_");
  if (!cdna5_target)
    execute(packed(rdna4::kVWmmaF3216x16x16F16Vop3p, 4, 4), 0, "v_wmma_");
  wf->set_exec(0);
  execute(packed(wmma, 4, 0), 0, "v_wmma_");
}

TEST_P(IsaDiagnosticsTest, ScalarDataAlignmentHasNarrowTargetAndOperandScope) {
  if (!vopd())
    GTEST_SKIP();
  const auto mov = [&](uint8_t dst, uint8_t src) {
    if (GetParam() == ROCJITSU_CODE_ARCH_CDNA5)
      return cdna5::build_sop1(cdna5::kSMovB64Sop1, {.ssrc0 = src, .sdst = dst});
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
      return rdna4::build_sop1(rdna4::kSMovB64Sop1, {.ssrc0 = src, .sdst = dst});
    return rdna3::build_sop1(rdna3::kSMovB64Sop1, {.ssrc0 = src, .sdst = dst});
  };
  execute(mov(16, 2), 0, "s_mov_b64");
  execute(mov(17, 2), gfx12(), "s_mov_b64");
  execute(mov(16, 3), gfx12(), "s_mov_b64");
  execute(mov(16, 129), 0, "s_mov_b64"); // Odd inline constant is not a register.
  execute(mov(16, 126), 0, "s_mov_b64"); // EXEC has its own selector semantics.
}

TEST_P(IsaDiagnosticsTest, ScalarLoadTupleAndBufferDescriptorAlignment) {
  if (!gfx12())
    GTEST_SKIP();
  const auto load = [&](uint16_t op, uint8_t dst, uint8_t base) {
    return GetParam() == ROCJITSU_CODE_ARCH_CDNA5
               ? cdna5::build_smem(op, {.sbase = base, .sdata = dst, .soffset = 4})
               : rdna4::build_smem(op, {.sbase = base, .sdata = dst, .soffset = 4});
  };
  // B32, B64, B128, B256, B512, B96 have exact widths in both ISA families.
  for (uint16_t op = 0; op <= 5; ++op) {
    execute(load(op, 16, 0), 0, "s_load_");
    execute(load(op, 17, 0), op != 0, "s_load_");
    execute(load(op, 18, 0), op >= 2, "s_load_");
    execute(load(op, 124, 0), 0, "s_load_"); // NULL discards at any width.
    execute(load(op + 16, 16, 0), 0, "s_buffer_load_");
    execute(load(op + 16, 16, 1), 1, "s_buffer_load_");
  }
}

TEST_P(IsaDiagnosticsTest, PermlaneGroupWidthUsesScalarValueOnce) {
  if (GetParam() != ROCJITSU_CODE_ARCH_CDNA5)
    GTEST_SKIP();
  for (uint16_t op : {cdna5::kVPermlaneBcastB32Vop3, cdna5::kVPermlaneUpB32Vop3,
                      cdna5::kVPermlaneDownB32Vop3, cdna5::kVPermlaneXorB32Vop3}) {
    const auto words = cdna5::build_vop3(op, {.vdst = 16, .src0 = 256, .src1 = 128, .src2 = 4});
    for (uint32_t width : {1u, 2u, 16u, 32u, 64u, 0x80000000u}) {
      wf->debug_write_sgpr(4, width);
      execute(words, 0, "v_permlane_");
    }
    for (uint32_t width : {0u, 3u, 24u}) {
      wf->debug_write_sgpr(4, width);
      execute(words, 1, "v_permlane_");
    }
    wf->set_exec(0);
    execute(words, 0, "v_permlane_");
    wf->set_exec(~uint64_t{0});
  }
}

std::array<uint32_t, 3> vopd_pair(uint8_t x, uint8_t y, uint16_t x0, uint8_t x1, uint16_t y0,
                                  uint8_t y1) {
  // Use v64/v65 destinations, leaving source banks independent of destinations.
  return {(0x32u << 26) | (uint32_t{x} << 22) | (uint32_t{y} << 17) | (uint32_t{x1} << 9) | x0,
          (64u << 24) | (32u << 17) | (uint32_t{y1} << 9) | y0, 0x3f800000u};
}

TEST_P(IsaDiagnosticsTest, VopdBanksHonorSharedSourcesAndIgnoredPorts) {
  if (!vopd())
    GTEST_SKIP();
  create(32);
  constexpr uint8_t mul = 3, mov = 8;
  execute(vopd_pair(mul, mul, 256, 1, 258, 2), 0, "v_dual_");
  execute(vopd_pair(mul, mul, 0, 1, 4, 2), 0, "v_dual_");
  execute(vopd_pair(mul, mul, 129, 1, 133, 2), 0, "v_dual_");
  execute(vopd_pair(mul, mul, 0, 1, 256, 2), 0, "v_dual_");
  execute(vopd_pair(mul, mul, 256, 1, 260, 2), 1, "v_dual_");
  execute(vopd_pair(mul, mul, 256, 1, 256, 2), !gfx12(), "v_dual_");
  execute(vopd_pair(mul, mul, 256, 1, 258, 5), 1, "v_dual_");
  execute(vopd_pair(mul, mul, 256, 1, 258, 1), !gfx12(), "v_dual_");
  // MOV ignores SRC1, and only GFX12 routes the second MOV through SRC2.
  execute(vopd_pair(mov, mov, 256, 1, 260, 5), !gfx12(), "v_dual_");
  execute(vopd_pair(mov, mul, 256, 1, 258, 5), 0, "v_dual_");
  wf->set_exec(0);
  execute(vopd_pair(mul, mul, 256, 1, 260, 2), 0, "v_dual_");
}

TEST_P(IsaDiagnosticsTest, VopdSrc2ParityAndScalarBudget) {
  if (!vopd())
    GTEST_SKIP();
  create(32);
  constexpr uint8_t fmac = 0, fmaak = 1, fmamk = 2, mul = 3, mov = 8;
  // FMAMK's second VGPR uses SRC2, not SRC1.
  execute(vopd_pair(fmamk, mul, 256, 1, 258, 5), 0, "v_dual_");
  execute(vopd_pair(fmamk, fmamk, 256, 1, 258, 3), 1, "v_dual_");
  execute(vopd_pair(fmamk, fmamk, 256, 1, 258, 2), 0, "v_dual_");
  execute(vopd_pair(fmamk, fmamk, 256, 1, 258, 1), !gfx12(), "v_dual_");
  // CDNA5 does not establish the older implicit-accumulator SRC2 rule.
  execute(vopd_pair(fmamk, fmac, 256, 1, 258, 2), GetParam() != ROCJITSU_CODE_ARCH_CDNA5,
          "v_dual_");
  execute(vopd_pair(fmaak, mul, 0, 1, 2, 2), 1, "v_dual_");
  execute(vopd_pair(fmaak, mul, 0, 1, 0, 2), 0, "v_dual_");
  execute(vopd_pair(mov, mov, 0, 1, 2, 2), 0, "v_dual_");
  execute(vopd_pair(fmaak, mul, 129, 1, 2, 2), 0, "v_dual_");
}

TEST_P(IsaDiagnosticsTest, VopdScalarBudgetQualifiesSpecialAndImplicitReads) {
  if (!vopd())
    GTEST_SKIP();
  create(32);
  constexpr uint8_t fmaak = 1, mov = 8, cndmask = 9;
  for (uint16_t selector : {126, 127, 253})
    execute(vopd_pair(fmaak, mov, 0, 1, selector, 2), gfx12(), "v_dual_");
  const bool implicit_vcc = GetParam() != ROCJITSU_CODE_ARCH_CDNA5;
  execute(vopd_pair(cndmask, cndmask, 0, 1, 2, 2), implicit_vcc, "v_dual_");
  execute(vopd_pair(cndmask, cndmask, 0, 1, 0, 2), 0, "v_dual_");
  execute(vopd_pair(fmaak, cndmask, 106, 1, 106, 2), 0, "v_dual_");
  execute(vopd_pair(fmaak, cndmask, 107, 1, 107, 2), implicit_vcc, "v_dual_");
}

TEST_P(IsaDiagnosticsTest, VopdDependenciesRemainUnqualified) {
  if (!vopd())
    GTEST_SKIP();
  create(32);
  // RDNA4's broad restriction conflicts with LLVM's single-cycle pairing rule.
  execute(vopd_pair(3, 3, 258, 1, 320, 2), 0, "v_dual_");
  // The reverse dependency reads the old value on every qualified target.
  execute(vopd_pair(3, 3, 321, 1, 258, 2), 0, "v_dual_");
}

TEST_P(IsaDiagnosticsTest, VopdRdna4CompilerAntidependencyRemainsSilent) {
  if (GetParam() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP();
  create(32);
  // The FP16 smoke emits MOV v10,v25 :: ADD v95,0,v10.
  const std::array<uint32_t, 2> words{0xca200119, 0x0a5e1480};
  execute(words, 0, "v_dual_mov_b32");
}

TEST_P(IsaDiagnosticsTest, Vopd3BanksRespectWidthAndExplicitScalarSelect) {
  if (GetParam() != ROCJITSU_CODE_ARCH_CDNA5)
    GTEST_SKIP();
  using test::cdna5::make_vopd3_pair;
  using test::cdna5::VopdOp;
  test::cdna5::VopdSlot x{VopdOp::FmaF32, 256, 1, 2, 64};
  test::cdna5::VopdSlot y{VopdOp::FmaF32, 256, 1, 2, 68};
  execute(make_vopd3_pair(x, y), 0, "v_dual_");
  y.src2 = 6;
  execute(make_vopd3_pair(x, y), 1, "v_dual_");
  y.src2 = 3;
  execute(make_vopd3_pair(x, y), 0, "v_dual_");
  x.op = VopdOp::FmaF64;
  execute(make_vopd3_pair(x, y), 1, "v_dual_"); // Same register, different widths.
  x = {VopdOp::CndmaskB32, 0, 1, 2, 64};
  y = {VopdOp::CndmaskB32, 3, 2, 2, 68};
  execute(make_vopd3_pair(x, y), 1, "v_dual_"); // Three explicit SGPR selectors.
  y.src0 = 0;
  execute(make_vopd3_pair(x, y), 0, "v_dual_");
}

TEST_P(IsaDiagnosticsTest, Vopd3F64SourcesCheckBothBanksOnEachPort) {
  if (GetParam() != ROCJITSU_CODE_ARCH_CDNA5)
    GTEST_SKIP();
  using test::cdna5::make_vopd3_pair;
  using test::cdna5::VopdOp;
  test::cdna5::VopdSlot x{VopdOp::FmaF64, 256, 4, 8, 64};
  const test::cdna5::VopdSlot legal_y{VopdOp::FmaF32, 258, 6, 10, 68};
  execute(make_vopd3_pair(x, legal_y), 0, "v_dual_");
  // Each F64 source occupies banks 0 and 1. Check one port at a time,
  // including overlapping registers and distinct registers in the same banks.
  for (uint8_t reg : {0, 1, 4, 5}) {
    auto y = legal_y;
    y.src0 = 256 + reg;
    execute(make_vopd3_pair(x, y), 1, "v_dual_");
    y = legal_y;
    y.src1 = 4 + reg;
    execute(make_vopd3_pair(x, y), 1, "v_dual_");
    y = legal_y;
    y.src2 = 8 + reg;
    execute(make_vopd3_pair(x, y), 1, "v_dual_");
  }
  // Both remaining banks are legal, including registers beyond the first four.
  for (uint8_t reg : {2, 3, 6, 7}) {
    const test::cdna5::VopdSlot y{VopdOp::FmaF32, static_cast<uint16_t>(256 + reg), reg, reg, 68};
    execute(make_vopd3_pair(x, y), 0, "v_dual_");
  }
  auto y = legal_y;
  y.src0 = 257;
  // SGPRs and inline constants do not occupy VGPR banks.
  for (uint16_t selector : {0, 128}) {
    x.src0 = selector;
    execute(make_vopd3_pair(x, y), 0, "v_dual_");
  }
}

TEST_P(IsaDiagnosticsTest, Vopd3MovesUseSeparatePortsWithSameBankSources) {
  if (GetParam() != ROCJITSU_CODE_ARCH_CDNA5)
    GTEST_SKIP();
  // The small FP8 smoke emits MOV v32,v4 :: MOV v18,v24 in VOPD3 form.
  // Both sources use bank 0; Y still uses SRC2, as it does in VOPD.
  const std::array<uint32_t, 3> words{0xcf208104, 0x00000118, 0x12000020};
  execute(words, 0, "v_dual_mov_b32");
}

INSTANTIATE_TEST_SUITE_P(Architectures, IsaDiagnosticsTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                           ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4,
                                           ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                           ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                           ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5));
} // namespace
