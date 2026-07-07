// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file decode_smoke_test.cpp
/// @brief Parameterized decode smoke test for all AMDGPU ISAs.
///
/// For each ISA we encode a known-good instruction word, call the generated
/// decoder, and assert:
///   1. Decoder::create() returns a non-null decoder.
///   2. decode() returns a non-null Instruction.
///   3. The decoded mnemonic matches the expected value.
///   4. The decoded instruction size in bytes is correct.
///
/// SOPP encoding:
///   bits[31:23] = 0x17F  (SOPP opcode base)
///   bits[22:16] = op     (per-instruction opcode field, 7 bits)
///   bits[15:0]  = simm16 (16-bit signed immediate, 0 here)
///
///   0xBF800000 = s_nop    (op = 0) — all AMDGPU ISAs
///   0xBF810000 = s_endpgm (op = 1) — CDNA1/2/3/4, RDNA1/2
///                                     (GFX9/10: op=1 is s_endpgm)
///   0xBFB00000 = s_endpgm (op = 48) — RDNA3/3.5/4
///                                     (GFX11/12: op=1 is s_setkill; s_endpgm moved to op=48)

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/vopd.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/vopd.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/vopd.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/except.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

namespace {

using namespace rocjitsu;

// SOPP encodings.
// s_nop is op=0 on all AMDGPU ISAs.
// s_endpgm is op=1 on CDNA1-4 and RDNA1/2 (GFX9/GFX10).
// s_endpgm is op=48 on RDNA3/3.5/4 (GFX11/GFX12) — op=1 is s_setkill there.
constexpr uint32_t S_NOP = 0xBF800000u;          ///< s_nop    (SOPP op=0,  simm16=0)
constexpr uint32_t S_ENDPGM_GFX9 = 0xBF810000u;  ///< s_endpgm (SOPP op=1,  simm16=0): CDNA/RDNA1/2
constexpr uint32_t S_ENDPGM_GFX11 = 0xBFB00000u; ///< s_endpgm (SOPP op=48, simm16=0): RDNA3/3.5/4

constexpr uint32_t make_sopp(uint32_t op, uint32_t simm16) {
  return (0x17Fu << 23) | ((op & 0x7Fu) << 16) | (simm16 & 0xFFFFu);
}

TEST(CodeArchApiTest, PreservesExistingPublicEnumValues) {
  EXPECT_EQ(static_cast<int>(ROCJITSU_CODE_ARCH_RDNA4), 8);
  EXPECT_EQ(static_cast<int>(ROCJITSU_CODE_ARCH_RV32I), 9);
  EXPECT_EQ(static_cast<int>(ROCJITSU_CODE_ARCH_RV64I), 10);
  EXPECT_EQ(static_cast<int>(ROCJITSU_CODE_ARCH_GFX1250), 11);
  EXPECT_EQ(static_cast<int>(ROCJITSU_CODE_ARCH_NUM_ARCHS), 12);
}

struct DecodeCase {
  rj_code_arch_t arch;
  const char *arch_name;
  uint32_t word;
  const char *expected_mnemonic;
  int expected_size_bytes;
};

class DecoderSmokeTest : public ::testing::TestWithParam<DecodeCase> {};

TEST_P(DecoderSmokeTest, DecodesCorrectly) {
  const DecodeCase &tc = GetParam();

  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr) << "Decoder::create() returned nullptr for arch=" << tc.arch_name;

  std::unique_ptr<Instruction> inst(decoder->decode(&tc.word));
  ASSERT_NE(inst, nullptr) << "decode() returned nullptr for arch=" << tc.arch_name << " word=0x"
                           << std::hex << tc.word;

  EXPECT_EQ(inst->mnemonic(), tc.expected_mnemonic) << "Wrong mnemonic for arch=" << tc.arch_name;
  EXPECT_EQ(inst->size(), tc.expected_size_bytes) << "Wrong size for arch=" << tc.arch_name;
}

// AMDGPU ISAs × 2 instructions.
// CDNA1/2/3/4 and RDNA1/2 share the GFX9/GFX10 s_endpgm encoding (op=1).
// RDNA3/3.5/4 use the GFX11/GFX12 encoding where s_endpgm moved to op=48.
INSTANTIATE_TEST_SUITE_P(
    AllIsas, DecoderSmokeTest,
    ::testing::Values(
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA1, "cdna1", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA1, "cdna1", S_ENDPGM_GFX9, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA2, "cdna2", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA2, "cdna2", S_ENDPGM_GFX9, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA3, "cdna3", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA3, "cdna3", S_ENDPGM_GFX9, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA4, "cdna4", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA4, "cdna4", S_ENDPGM_GFX9, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA1, "rdna1", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA1, "rdna1", S_ENDPGM_GFX9, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA2, "rdna2", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA2, "rdna2", S_ENDPGM_GFX9, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3", S_ENDPGM_GFX11, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", S_ENDPGM_GFX11, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", S_ENDPGM_GFX11, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_GFX1250, "gfx1250", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_GFX1250, "gfx1250", S_ENDPGM_GFX11, "s_endpgm", 4}),
    [](const ::testing::TestParamInfo<DecodeCase> &info) {
      std::string name = info.param.arch_name;
      name += "_";
      name += info.param.expected_mnemonic;
      return name;
    });

TEST(Rdna4WaitcntDecodeSmokeTest, FormatsCompatWaitcntWithGfx11Layout) {
  constexpr uint32_t s_waitcnt_vmcnt1 = make_sopp(/*op=*/9, /*simm16=*/1u << 10);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);

  std::unique_ptr<Instruction> inst(decoder->decode(&s_waitcnt_vmcnt1));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "s_waitcnt");
  EXPECT_EQ(inst->disassemble(), "s_waitcnt vmcnt(1) expcnt(0) lgkmcnt(0)");
}

struct VopdDecodeCase {
  rj_code_arch_t arch;
  const char *arch_name;
  const char *case_name;
  std::array<uint32_t, 3> words;
  int expected_size_bytes;
  const char *expected_mnemonic;
  const char *expected_disasm_substring;
};

constexpr uint16_t vopd_src0_vgpr(uint16_t reg) { return 256 + reg; }

constexpr std::array<uint32_t, 3>
make_vopdxy_pair(uint8_t opx, uint8_t opy, uint16_t srcx0 = vopd_src0_vgpr(1), uint8_t vsrcx1 = 2,
                 uint16_t srcy0 = vopd_src0_vgpr(3), uint8_t vsrcy1 = 4, uint8_t vdstx = 0,
                 uint8_t vdsty = 1, uint32_t literal = 0) {
  return {
      (0x32u << 26) | ((static_cast<uint32_t>(opx) & 0xFu) << 22) |
          ((static_cast<uint32_t>(opy) & 0x1Fu) << 17) | (static_cast<uint32_t>(vsrcx1) << 9) |
          (srcx0 & 0x1FFu),
      (static_cast<uint32_t>(vdstx) << 24) | (static_cast<uint32_t>(vdsty >> 1) << 17) |
          (static_cast<uint32_t>(vsrcy1) << 9) | (srcy0 & 0x1FFu),
      literal,
  };
}

class RdnaVopdDecodeSmokeTest : public ::testing::TestWithParam<VopdDecodeCase> {};

TEST_P(RdnaVopdDecodeSmokeTest, DecodesDualSlotForms) {
  const auto &tc = GetParam();
  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr) << tc.arch_name;

  std::unique_ptr<Instruction> inst(decoder->decode(tc.words.data()));
  ASSERT_NE(inst, nullptr) << tc.arch_name << " " << tc.case_name;
  EXPECT_EQ(inst->mnemonic(), tc.expected_mnemonic);
  EXPECT_EQ(inst->size(), tc.expected_size_bytes);

  std::string disasm = inst->disassemble();
  EXPECT_NE(disasm.find(tc.expected_disasm_substring), std::string::npos) << disasm;
}

INSTANTIATE_TEST_SUITE_P(
    RdnaVopd, RdnaVopdDecodeSmokeTest,
    ::testing::Values(
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3", "vopdxy", make_vopdxy_pair(9, 8), 8,
                       "v_dual_cndmask_b32 :: v_dual_mov_b32", "v_dual_cndmask_b32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3,
                       "rdna3",
                       "literal",
                       {0xC8D006FFu, 0x04020080u, 0x4F7FFFFEu},
                       12,
                       "v_dual_mul_f32 :: v_dual_mov_b32",
                       "0x4f7ffffe"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3", "max_min", make_vopdxy_pair(10, 11), 8,
                       "v_dual_max_f32 :: v_dual_min_f32", "v_dual_min_f32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3", "dot_and", make_vopdxy_pair(12, 18), 8,
                       "v_dual_dot2acc_f32_f16 :: v_dual_and_b32", "v_dual_and_b32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3", "dot_bf16_add", make_vopdxy_pair(13, 16),
                       8, "v_dual_dot2acc_f32_bf16 :: v_dual_add_nc_u32",
                       "v_dual_dot2acc_f32_bf16"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", "vopdxy", make_vopdxy_pair(9, 8), 8,
                       "v_dual_cndmask_b32 :: v_dual_mov_b32", "v_dual_cndmask_b32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3_5,
                       "rdna3_5",
                       "literal",
                       {0xC8D006FFu, 0x04020080u, 0x4F7FFFFEu},
                       12,
                       "v_dual_mul_f32 :: v_dual_mov_b32",
                       "0x4f7ffffe"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", "max_min", make_vopdxy_pair(10, 11),
                       8, "v_dual_max_f32 :: v_dual_min_f32", "v_dual_min_f32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", "dot_and", make_vopdxy_pair(12, 18),
                       8, "v_dual_dot2acc_f32_f16 :: v_dual_and_b32", "v_dual_and_b32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", "dot_bf16_add",
                       make_vopdxy_pair(13, 16), 8, "v_dual_dot2acc_f32_bf16 :: v_dual_add_nc_u32",
                       "v_dual_dot2acc_f32_bf16"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", "vopdxy", make_vopdxy_pair(9, 8), 8,
                       "v_dual_cndmask_b32 :: v_dual_mov_b32", "v_dual_cndmask_b32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA4,
                       "rdna4",
                       "literal",
                       {0xC8D006FFu, 0x04020080u, 0x4F7FFFFEu},
                       12,
                       "v_dual_mul_f32 :: v_dual_mov_b32",
                       "0x4f7ffffe"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", "max_min", make_vopdxy_pair(10, 11), 8,
                       "v_dual_max_num_f32 :: v_dual_min_num_f32", "v_dual_min_num_f32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", "dot_and", make_vopdxy_pair(12, 18), 8,
                       "v_dual_dot2acc_f32_f16 :: v_dual_and_b32", "v_dual_and_b32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", "dot_bf16_add", make_vopdxy_pair(13, 16),
                       8, "v_dual_dot2acc_f32_bf16 :: v_dual_add_nc_u32",
                       "v_dual_dot2acc_f32_bf16"}),
    [](const ::testing::TestParamInfo<VopdDecodeCase> &info) {
      std::string name = info.param.arch_name;
      name += "_";
      name += info.param.case_name;
      return name;
    });

struct InvalidVopdDecodeCase {
  rj_code_arch_t arch;
  const char *arch_name;
  std::array<uint32_t, 3> words;
};

class RdnaInvalidVopdDecodeSmokeTest : public ::testing::TestWithParam<InvalidVopdDecodeCase> {};

TEST_P(RdnaInvalidVopdDecodeSmokeTest, DoesNotClaimVopd3Encoding) {
  const auto &tc = GetParam();
  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr) << tc.arch_name;

  switch (tc.arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
    EXPECT_FALSE(
        rdna3::Vopd::is_vopd(reinterpret_cast<const rdna3::MachineInst *>(tc.words.data())));
    break;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    EXPECT_FALSE(
        rdna3_5::Vopd::is_vopd(reinterpret_cast<const rdna3_5::MachineInst *>(tc.words.data())));
    break;
  case ROCJITSU_CODE_ARCH_RDNA4:
    EXPECT_FALSE(
        rdna4::Vopd::is_vopd(reinterpret_cast<const rdna4::MachineInst *>(tc.words.data())));
    break;
  default:
    FAIL() << "unexpected test arch";
  }

  EXPECT_THROW(static_cast<void>(decoder->decode(tc.words.data())), util::InvalidInst)
      << tc.arch_name << " should reserve the 0xCF VOPD3 prefix";
}

INSTANTIATE_TEST_SUITE_P(
    RdnaVopd, RdnaInvalidVopdDecodeSmokeTest,
    ::testing::Values(
        InvalidVopdDecodeCase{
            ROCJITSU_CODE_ARCH_RDNA3, "rdna3", {0xCF455083u, 0x00000086u, 0x0A000001u}},
        InvalidVopdDecodeCase{
            ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", {0xCF455083u, 0x00000086u, 0x0A000001u}},
        InvalidVopdDecodeCase{
            ROCJITSU_CODE_ARCH_RDNA4, "rdna4", {0xCF455083u, 0x00000086u, 0x0A000001u}}),
    [](const ::testing::TestParamInfo<InvalidVopdDecodeCase> &info) {
      return std::string(info.param.arch_name);
    });

struct RdnaVopdExecutionCase {
  rj_code_arch_t arch;
  const char *arch_name;
};

class RdnaVopdExecutionSmokeTest : public ::testing::TestWithParam<RdnaVopdExecutionCase> {};

TEST_P(RdnaVopdExecutionSmokeTest, PreservesFpRoundingAndDx9ZeroSemantics) {
  const auto &tc = GetParam();
  constexpr uint32_t kSrc0 = 0x3F800001u;
  constexpr uint32_t kSrc1 = 0x3F7FFFFFu;
  constexpr uint32_t kLiteralAddend = 0xBF800000u;
  constexpr uint32_t kQuietNan = 0x7FC00000u;
  constexpr uint32_t kPositiveZero = 0x00000000u;
  constexpr uint32_t kNegativeZero = 0x80000000u;
  constexpr uint32_t kFmaakOp = 1;
  constexpr uint32_t kMulDx9ZeroOp = 7;
  constexpr uint32_t kFmaDst = 4;
  constexpr uint32_t kDx9Dst = 5;
  constexpr uint64_t kExecMask = 0xFFFF'FFFFULL;

  const auto words = make_vopdxy_pair(kFmaakOp, kMulDx9ZeroOp, vopd_src0_vgpr(0), 1,
                                      vopd_src0_vgpr(2), 3, kFmaDst, kDx9Dst, kLiteralAddend);
  const uint32_t expected_fma =
      std::bit_cast<uint32_t>(std::fma(std::bit_cast<float>(kSrc0), std::bit_cast<float>(kSrc1),
                                       std::bit_cast<float>(kLiteralAddend)));
  ASSERT_EQ(expected_fma, 0x337FFFFEu);

  amdgpu::GpuMemory gpu_mem(std::string(tc.arch_name) + "_vopd_exec_mem");
  amdgpu::L2Cache l2(std::string(tc.arch_name) + "_vopd_exec_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = tc.arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu =
      amdgpu::ComputeUnitCore::create(std::string(tc.arch_name) + "_vopd_exec", cfg, &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(kExecMask);

  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words.data()));
  ASSERT_NE(inst, nullptr);

  const uint32_t vb = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu->write_vgpr(vb + 0, lane, kSrc0);
    cu->write_vgpr(vb + 1, lane, kSrc1);
    cu->write_vgpr(vb + 2, lane, kQuietNan);
    cu->write_vgpr(vb + 3, lane, (lane & 1u) ? kNegativeZero : kPositiveZero);
    cu->write_vgpr(vb + kFmaDst, lane, 0xDEADBEEFu);
    cu->write_vgpr(vb + kDx9Dst, lane, 0xDEADBEEFu);
  }

  cu->execute_instruction(inst.get(), *wf);

  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    EXPECT_EQ(cu->read_vgpr(vb + kFmaDst, lane), expected_fma) << tc.arch_name << " lane " << lane;
    EXPECT_EQ(cu->read_vgpr(vb + kDx9Dst, lane), kPositiveZero) << tc.arch_name << " lane " << lane;
  }
}

TEST_P(RdnaVopdExecutionSmokeTest, DualCndmaskConsumesVccLo) {
  const auto &tc = GetParam();
  constexpr std::array<uint32_t, 2> kWords = {
      0xCA521307u, // v_dual_cndmask_b32 v7, v7, v9 :: v_dual_cndmask_b32 v6, v6, v8
      0x07061106u,
  };
  constexpr uint32_t kXDst = 7;
  constexpr uint32_t kYDst = 6;
  constexpr uint32_t kXFalse = 0x10100000u;
  constexpr uint32_t kXTrue = 0x20200000u;
  constexpr uint32_t kYFalse = 0x30300000u;
  constexpr uint32_t kYTrue = 0x40400000u;
  constexpr uint64_t kVcc = 0xAAAA'AAAAu;
  constexpr uint64_t kExecMask = 0xFFFF'FFFFULL;

  amdgpu::GpuMemory gpu_mem(std::string(tc.arch_name) + "_vopd_cndmask_mem");
  amdgpu::L2Cache l2(std::string(tc.arch_name) + "_vopd_cndmask_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = tc.arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(std::string(tc.arch_name) + "_vopd_cndmask", cfg,
                                            &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(kExecMask);
  wf->set_vcc(kVcc);

  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(kWords.data()));
  ASSERT_NE(inst, nullptr);

  const uint32_t vb = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu->write_vgpr(vb + 7, lane, kXFalse | lane);
    cu->write_vgpr(vb + 9, lane, kXTrue | lane);
    cu->write_vgpr(vb + 6, lane, kYFalse | lane);
    cu->write_vgpr(vb + 8, lane, kYTrue | lane);
  }

  cu->execute_instruction(inst.get(), *wf);

  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    const bool select_true = ((kVcc >> lane) & 1u) != 0;
    EXPECT_EQ(cu->read_vgpr(vb + kXDst, lane), (select_true ? kXTrue : kXFalse) | lane)
        << tc.arch_name << " lane " << lane;
    EXPECT_EQ(cu->read_vgpr(vb + kYDst, lane), (select_true ? kYTrue : kYFalse) | lane)
        << tc.arch_name << " lane " << lane;
  }
}

TEST_P(RdnaVopdExecutionSmokeTest, DualCndmaskAfterScalarVccMerge) {
  const auto &tc = GetParam();
  constexpr uint32_t kModulusMinusOne = 0xFF65CF7Au;
  constexpr uint32_t kLaneCount = 9;
  constexpr uint64_t kExecMask = (1ULL << kLaneCount) - 1;
  constexpr uint64_t kCorrectionMask = 0x1D8u;
  constexpr std::array<uint64_t, kLaneCount> kA = {
      0,
      1,
      2,
      4284862330ULL,
      4284862329ULL,
      1071215583ULL,
      2142431166ULL,
      3213646750ULL,
      4284862327ULL,
  };
  constexpr std::array<uint64_t, kLaneCount> kB = {
      0, 1, 2, 4284862330ULL, 4284862329ULL, 1071215588ULL, 2142431165ULL, 3213646748ULL, 4,
  };
  constexpr std::array<uint64_t, kLaneCount> kExpected = {
      0, 2, 4, 4284862329ULL, 4284862327ULL, 2142431171ULL, 0, 2142431167ULL, 0,
  };
  const std::array<std::array<uint32_t, 3>, 10> words = {{
      {0xBE8001FFu, kModulusMinusOne, 0u},     // s_mov_b64 s[0:1], 0xff65cf7a
      {0xD7006A06u, 0x02020D08u, 0u},          // v_add_co_u32 v6, vcc_lo, v8, v6
      {0x400E0F09u, 0u, 0u},                   // v_add_co_ci_u32_e32 v7, vcc_lo, v9, v7, vcc_lo
      {0xD4590000u, 0x02020C00u, 0u},          // v_cmp_lt_u64_e64 s0, s[0:1], v[6:7]
      {0xD7000108u, 0x02020CFFu, 0x009A3085u}, // v_add_co_u32 v8, s1, 0x9a3085, v6
      {0xD5207C09u, 0x00060EC1u, 0u},          // v_add_co_ci_u32_e64 v9, null, -1, v7, s1
      {0xD7000100u, 0x02020002u, 0u},          // v_add_co_u32 v0, s1, s2, v0
      {0x8C6A006Au, 0u, 0u},                   // s_or_b32 vcc_lo, vcc_lo, s0
      {0xD5207C01u, 0x00060203u, 0u},          // v_add_co_ci_u32_e64 v1, null, s3, v1, s1
      {0xCA521307u, 0x07061106u, 0u},          // v_dual_cndmask_b32 v7/v6
  }};

  amdgpu::GpuMemory gpu_mem(std::string(tc.arch_name) + "_vopd_vcc_merge_mem");
  amdgpu::L2Cache l2(std::string(tc.arch_name) + "_vopd_vcc_merge_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = tc.arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(std::string(tc.arch_name) + "_vopd_vcc_merge", cfg,
                                            &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(kExecMask);

  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr);
  const auto execute = [&](const std::array<uint32_t, 3> &inst_words) {
    std::unique_ptr<Instruction> inst(decoder->decode(inst_words.data()));
    ASSERT_NE(inst, nullptr);
    cu->execute_instruction(inst.get(), *wf);
  };

  const uint32_t sb = wf->sgpr_alloc().base;
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_sgpr(sb + 0, 0xAAAAAAAAu);
  cu->write_sgpr(sb + 1, 0xBBBBBBBBu);
  cu->write_sgpr(sb + 2, 0u);
  cu->write_sgpr(sb + 3, 0u);
  for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
    cu->write_vgpr(vb + 0, lane, 0u);
    cu->write_vgpr(vb + 1, lane, 0u);
    cu->write_vgpr(vb + 6, lane, static_cast<uint32_t>(kA[lane]));
    cu->write_vgpr(vb + 7, lane, static_cast<uint32_t>(kA[lane] >> 32));
    cu->write_vgpr(vb + 8, lane, static_cast<uint32_t>(kB[lane]));
    cu->write_vgpr(vb + 9, lane, static_cast<uint32_t>(kB[lane] >> 32));
  }

  execute(words[0]);
  EXPECT_EQ(cu->read_sgpr(sb + 0), kModulusMinusOne);
  EXPECT_EQ(cu->read_sgpr(sb + 1), 0u);

  execute(words[1]);
  execute(words[2]);
  EXPECT_EQ(wf->vcc() & kExecMask, 0u);

  execute(words[3]);
  EXPECT_EQ(cu->read_sgpr(sb + 0) & kExecMask, kCorrectionMask);

  execute(words[4]);
  execute(words[5]);
  execute(words[6]);
  execute(words[7]);
  EXPECT_EQ(wf->vcc() & kExecMask, kCorrectionMask);
  execute(words[8]);
  EXPECT_EQ(wf->vcc() & kExecMask, kCorrectionMask);
  execute(words[9]);

  for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
    uint64_t actual = static_cast<uint64_t>(cu->read_vgpr(vb + 6, lane)) |
                      (static_cast<uint64_t>(cu->read_vgpr(vb + 7, lane)) << 32);
    EXPECT_EQ(actual, kExpected[lane]) << tc.arch_name << " lane " << lane;
  }
}

TEST_P(RdnaVopdExecutionSmokeTest, DualCndmaskAfterScalarVccMergeViaCuStep) {
  const auto &tc = GetParam();
  constexpr uint32_t kModulusMinusOne = 0xFF65CF7Au;
  constexpr uint32_t kLaneCount = 9;
  constexpr uint64_t kExecMask = (1ULL << kLaneCount) - 1;
  constexpr uint64_t kCorrectionMask = 0x1D8u;
  constexpr std::array<uint64_t, kLaneCount> kA = {
      0,
      1,
      2,
      4284862330ULL,
      4284862329ULL,
      1071215583ULL,
      2142431166ULL,
      3213646750ULL,
      4284862327ULL,
  };
  constexpr std::array<uint64_t, kLaneCount> kB = {
      0, 1, 2, 4284862330ULL, 4284862329ULL, 1071215588ULL, 2142431165ULL, 3213646748ULL, 4,
  };
  constexpr std::array<uint64_t, kLaneCount> kExpected = {
      0, 2, 4, 4284862329ULL, 4284862327ULL, 2142431171ULL, 0, 2142431167ULL, 0,
  };
  constexpr std::array<uint32_t, 19> kWords = {
      0xBE8001FFu, kModulusMinusOne,              // s_mov_b64 s[0:1], 0xff65cf7a
      0xD7006A06u, 0x02020D08u,                   // v_add_co_u32 v6, vcc_lo, v8, v6
      0x400E0F09u,                                // v_add_co_ci_u32_e32 v7, vcc_lo, v9, v7, vcc_lo
      0xD4590000u, 0x02020C00u,                   // v_cmp_lt_u64_e64 s0, s[0:1], v[6:7]
      0xD7000108u, 0x02020CFFu,      0x009A3085u, // v_add_co_u32 v8, s1, 0x9a3085, v6
      0xD5207C09u, 0x00060EC1u,                   // v_add_co_ci_u32_e64 v9, null, -1, v7, s1
      0xD7000100u, 0x02020002u,                   // v_add_co_u32 v0, s1, s2, v0
      0x8C6A006Au,                                // s_or_b32 vcc_lo, vcc_lo, s0
      0xD5207C01u, 0x00060203u,                   // v_add_co_ci_u32_e64 v1, null, s3, v1, s1
      0xCA521307u, 0x07061106u,                   // v_dual_cndmask_b32 v7/v6
  };
  constexpr std::array<uint64_t, 10> kExpectedPc = {8, 16, 20, 28, 40, 48, 56, 60, 68, 76};

  amdgpu::GpuMemory gpu_mem(std::string(tc.arch_name) + "_vopd_vcc_merge_step_mem");
  amdgpu::L2Cache l2(std::string(tc.arch_name) + "_vopd_vcc_merge_step_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = tc.arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(std::string(tc.arch_name) + "_vopd_vcc_merge_step", cfg,
                                            &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(kExecMask);

  for (uint32_t i = 0; i < kWords.size(); ++i)
    gpu_mem.write32(i * sizeof(uint32_t), kWords[i]);

  const uint32_t sb = wf->sgpr_alloc().base;
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_sgpr(sb + 0, 0xAAAAAAAAu);
  cu->write_sgpr(sb + 1, 0xBBBBBBBBu);
  cu->write_sgpr(sb + 2, 0u);
  cu->write_sgpr(sb + 3, 0u);
  for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
    cu->write_vgpr(vb + 0, lane, 0u);
    cu->write_vgpr(vb + 1, lane, 0u);
    cu->write_vgpr(vb + 6, lane, static_cast<uint32_t>(kA[lane]));
    cu->write_vgpr(vb + 7, lane, static_cast<uint32_t>(kA[lane] >> 32));
    cu->write_vgpr(vb + 8, lane, static_cast<uint32_t>(kB[lane]));
    cu->write_vgpr(vb + 9, lane, static_cast<uint32_t>(kB[lane] >> 32));
  }

  for (uint32_t i = 0; i < kExpectedPc.size(); ++i) {
    cu->step();
    EXPECT_EQ(wf->pc, kExpectedPc[i]) << tc.arch_name << " step " << i;
    if (i == 0) {
      EXPECT_EQ(cu->read_sgpr(sb + 0), kModulusMinusOne);
      EXPECT_EQ(cu->read_sgpr(sb + 1), 0u);
    }
  }

  EXPECT_EQ(wf->vcc() & kExecMask, kCorrectionMask);
  for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
    uint64_t actual = static_cast<uint64_t>(cu->read_vgpr(vb + 6, lane)) |
                      (static_cast<uint64_t>(cu->read_vgpr(vb + 7, lane)) << 32);
    EXPECT_EQ(actual, kExpected[lane]) << tc.arch_name << " lane " << lane;
  }
}

INSTANTIATE_TEST_SUITE_P(
    RdnaVopd, RdnaVopdExecutionSmokeTest,
    ::testing::Values(RdnaVopdExecutionCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3"},
                      RdnaVopdExecutionCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5"},
                      RdnaVopdExecutionCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4"}),
    [](const ::testing::TestParamInfo<RdnaVopdExecutionCase> &info) {
      return std::string(info.param.arch_name);
    });

// ---------------------------------------------------------------------------
// MUBUF lds modifier test: verify that buffer_load_dword with the lds bit set
// (bit 16 of dword 0) produces a disassembly string containing " lds".
//
// MUBUF encoding (CDNA3/4):
//   dword 0: [31:26]=enc  [24:18]=op  [17]=nt  [16]=lds  [15]=sc1
//            [14]=sc0  [13]=idxen  [12]=offen  [11:0]=offset
//   dword 1: [31:24]=vdata  [20:16]=vaddr  [15:11]=srsrc  [10:8]=soffset
//
// buffer_load_dword without lds: {0xE0500000, 0x00000000}
// buffer_load_dword with    lds: {0xE0510000, 0x00000000}  (bit 16 set)
// ---------------------------------------------------------------------------

struct MubufLdsCase {
  rj_code_arch_t arch;
  const char *arch_name;
  uint32_t words[2];
  bool expect_lds;
};

class MubufLdsModifierTest : public ::testing::TestWithParam<MubufLdsCase> {};

TEST_P(MubufLdsModifierTest, LdsModifierInDisassembly) {
  const auto &tc = GetParam();
  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr);

  std::unique_ptr<Instruction> inst(decoder->decode(tc.words));
  ASSERT_NE(inst, nullptr) << "decode() returned nullptr for " << tc.arch_name;
  EXPECT_EQ(inst->mnemonic(), "buffer_load_dword");

  std::string disasm = inst->disassemble();
  if (tc.expect_lds) {
    EXPECT_NE(disasm.find(" lds"), std::string::npos)
        << "Expected ' lds' in disassembly: " << disasm;
  } else {
    EXPECT_EQ(disasm.find(" lds"), std::string::npos)
        << "Unexpected ' lds' in disassembly: " << disasm;
  }
}

INSTANTIATE_TEST_SUITE_P(
    MubufLds, MubufLdsModifierTest,
    ::testing::Values(
        // CDNA4: buffer_load_dword without lds
        MubufLdsCase{ROCJITSU_CODE_ARCH_CDNA4, "cdna4", {0xE0500000u, 0x00000000u}, false},
        // CDNA4: buffer_load_dword with lds (bit 16 set)
        MubufLdsCase{ROCJITSU_CODE_ARCH_CDNA4, "cdna4", {0xE0510000u, 0x00000000u}, true},
        // CDNA1: buffer_load_dword without lds (GFX9 MUBUF enc=0x38)
        MubufLdsCase{ROCJITSU_CODE_ARCH_CDNA1, "cdna1", {0xE0500000u, 0x00000000u}, false},
        // CDNA1: buffer_load_dword with lds (bit 16 set)
        MubufLdsCase{ROCJITSU_CODE_ARCH_CDNA1, "cdna1", {0xE0510000u, 0x00000000u}, true}),
    [](const ::testing::TestParamInfo<MubufLdsCase> &info) {
      std::string name = info.param.arch_name;
      name += info.param.expect_lds ? "_with_lds" : "_without_lds";
      return name;
    });

} // namespace
