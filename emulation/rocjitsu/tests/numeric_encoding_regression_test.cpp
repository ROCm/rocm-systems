// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <tuple>

namespace rocjitsu {
namespace {

class NumericEncodingRegression : public testing::TestWithParam<std::tuple<rj_code_arch_t, bool>> {
protected:
  void SetUp() override {
    saved_scalar_ = util::force_scalar();
    util::set_force_scalar_for_testing(std::get<1>(GetParam()));
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = std::get<0>(GetParam());
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    ASSERT_NO_FATAL_FAILURE(rocjitsu::test::initialize_compute_unit(cu_, "numeric_encoding", config,
                                                                    &memory_, &cache_));
    decoder_ = Decoder::create(config.arch);
    wave_ = cu_->dispatch_wf(0, 0, 106, 256);
    ASSERT_NE(wave_, nullptr);
    wave_->set_exec(~uint64_t{0});
  }

  void TearDown() override {
    if (wave_)
      wave_->halt();
    util::set_force_scalar_for_testing(saved_scalar_);
  }

  void put(unsigned reg, uint32_t value) {
    for (unsigned lane = 0; lane != wave_->wf_size(); ++lane)
      cu_->write_vgpr(wave_->vgpr_alloc().base + reg, lane, value);
  }

  void check(std::array<uint32_t, 2> words, uint32_t expected, uint32_t initial = 0xdeadbeefu) {
    SCOPED_TRACE(testing::Message() << "encoded words=" << std::hex << words[0] << ":" << words[1]);
    put(4, initial);
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder_, words.data()));
    ASSERT_NE(instruction, nullptr);
    SCOPED_TRACE(testing::Message() << instruction->mnemonic() << " words=" << std::hex << words[0]
                                    << ":" << words[1]);
    ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
    for (unsigned lane = 0; lane != wave_->wf_size(); ++lane)
      EXPECT_EQ(cu_->read_vgpr_storage(wave_->vgpr_alloc().base + 4, lane),
                wave_->exec() & (uint64_t{1} << lane) ? expected : initial)
          << "lane=" << lane;
  }

  static std::array<uint32_t, 2> vop1(uint32_t opcode) {
    return {0x7e000100u | (4u << 17) | (opcode << 9), 0};
  }

  static std::array<uint32_t, 2> vop3(uint32_t opcode, uint32_t opsel = 0, uint32_t omod = 0) {
    return {0xd0000004u | (opcode << 16) | (opsel << 11),
            256u | (257u << 9) | (258u << 18) | (omod << 27)};
  }

  bool saved_scalar_ = false;
  amdgpu::GpuMemory memory_{"numeric_memory"};
  amdgpu::L2Cache cache_{"numeric_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  std::unique_ptr<Decoder> decoder_;
  amdgpu::Wavefront *wave_ = nullptr;
};

class LiteralMadGenerationTest : public NumericEncodingRegression {};

INSTANTIATE_TEST_SUITE_P(Gfx90aAndGfx942, LiteralMadGenerationTest,
                         testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_CDNA2,
                                                          ROCJITSU_CODE_ARCH_CDNA3),
                                          testing::Bool()));

TEST_P(LiteralMadGenerationTest, SameOpcodeRoundsDifferentlyAfterCancellation) {
  // (1 + 2^-23) * (1 - 2^-23) - 1 is zero with a rounded product, or -2^-46 fused.
  const uint32_t expected = std::get<0>(GetParam()) == ROCJITSU_CODE_ARCH_CDNA2 ? 0 : 0xa8800000;
  wave_->set_mode_raw(0xf0);
  wave_->set_exec(0x8000000000000001ULL);
  put(0, 0x3f800001);
  put(1, 0x3f7ffffe);
  check({0x30080300, 0xbf800000}, expected); // gfx90a MADAK / gfx942 FMAAK
  put(1, 0xbf800000);
  check({0x2e080300, 0x3f7ffffe}, expected); // gfx90a MADMK / gfx942 FMAMK
}

class CdnaVectorRegression : public NumericEncodingRegression {
protected:
  void execute(std::array<uint32_t, 2> words) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder_, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wave_).succeeded());
  }
};

INSTANTIATE_TEST_SUITE_P(Gfx90aGfx942AndGfx950, CdnaVectorRegression,
                         testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_CDNA2,
                                                          ROCJITSU_CODE_ARCH_CDNA3,
                                                          ROCJITSU_CODE_ARCH_CDNA4),
                                          testing::Bool()));

class Cdna2VectorRegression : public CdnaVectorRegression {};

INSTANTIATE_TEST_SUITE_P(Gfx90a, Cdna2VectorRegression,
                         testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_CDNA2),
                                          testing::Bool()));

TEST_P(Cdna2VectorRegression, IndexedDestinationsCannotCrossIntoAccumulators) {
  const auto read = [&](unsigned reg) {
    return cu_->read_vgpr_storage(wave_->vgpr_alloc().base + reg, 0);
  };
  constexpr uint32_t marker = 0xdeadbeef;
  constexpr uint64_t old_vcc = 0x123456789abcdef0ULL;
  put(0, 0xffffffff);
  put(256, marker); // a0 follows v255 in physical storage.
  cu_->write_sgpr(wave_->sgpr_alloc().base, 255);
  ASSERT_NO_FATAL_FAILURE(execute({0xbf110800, 0})); // s_set_gpr_idx_on s0, gpr_idx(DST)
  ASSERT_NO_FATAL_FAILURE(execute({0x7e020300, 0})); // v_mov_b32 v1, v0
  EXPECT_EQ(read(256), marker);
  wave_->set_vcc_raw(old_vcc);
  ASSERT_NO_FATAL_FAILURE(execute({0x32020100, 0})); // v_add_co_u32 v1, vcc, v0, v0
  EXPECT_EQ(read(256), marker);
  EXPECT_EQ(wave_->vcc(), old_vcc);
  put(2, marker);
  ASSERT_NO_FATAL_FAILURE(execute({0xbf110100, 0})); // The second V_SWAP output uses SRC0 indexing.
  ASSERT_NO_FATAL_FAILURE(execute({0x7e04a301, 0})); // v_swap_b32 v2, v1
  EXPECT_EQ(read(2), marker);
  EXPECT_EQ(read(0), 0xffffffffu);

  cu_->write_sgpr(wave_->sgpr_alloc().base, 2);
  ASSERT_NO_FATAL_FAILURE(execute({0xbf110900, 0})); // gpr_idx(SRC0,DST)
  put(257, marker);
  put(258, 0xfacefeed);
  ASSERT_NO_FATAL_FAILURE(execute({0x7e00a501, 0})); // v_accvgpr_mov_b32 a0, a1
  EXPECT_EQ(read(256), marker);
  EXPECT_EQ(read(258), 0xfacefeedu);

  put(0, 1);
  put(253, marker);
  put(254, marker);
  put(255, marker);
  cu_->write_sgpr(wave_->sgpr_alloc().base, 1);
  ASSERT_NO_FATAL_FAILURE(execute({0xbf110800, 0}));
  ASSERT_NO_FATAL_FAILURE(execute({0x7ffc0900, 0})); // v_cvt_f64_i32 v[254:255], v0
  EXPECT_EQ(read(255), marker);
  EXPECT_EQ(read(256), marker);
  ASSERT_NO_FATAL_FAILURE(execute({0xd3d500fc, 0x02020500})); // MFMA v[252:255], v0, v2, 0
  for (unsigned reg = 253; reg <= 256; ++reg)
    EXPECT_EQ(read(reg), marker);
  // The same encoded MFMA tuple is in range in the separate accumulator file.
  put(508, marker);
  put(2, 0);
  ASSERT_NO_FATAL_FAILURE(execute({0xd3d580fc, 0x02020500}));
  EXPECT_EQ(read(508), 0u);

  // At the highest valid VGPR pair the conversion must still write both words.
  cu_->write_sgpr(wave_->sgpr_alloc().base, 0);
  ASSERT_NO_FATAL_FAILURE(execute({0xbf110800, 0}));
  ASSERT_NO_FATAL_FAILURE(execute({0x7ffc0900, 0}));
  EXPECT_EQ(read(254), 0u);
  EXPECT_EQ(read(255), 0x3ff00000u);
}

TEST_P(CdnaVectorRegression, ValuHonorsBothRegisterQuotasAndAllDestinations) {
  wave_->halt();
  wave_ = cu_->dispatch_wf(0, 0, 16, 32);
  ASSERT_NE(wave_, nullptr);
  wave_->set_accum_offset(24); // v0-v23 plus a0-a7.
  constexpr uint32_t marker = 0xdeadbeef;
  const auto read = [&](unsigned reg) {
    return cu_->read_vgpr_storage(wave_->vgpr_alloc().base + reg, 0);
  };
  put(0, 1);
  put(1, marker);
  ASSERT_NO_FATAL_FAILURE(execute({0xd1191e01, 0x00020100})); // v_add_co_u32 v1, s[30:31], v0, v0
  EXPECT_EQ(read(1), marker);
  ASSERT_NO_FATAL_FAILURE(execute({0xd1190001, 0x00020100})); // s[0:1] is allocated.
  EXPECT_EQ(read(1), 2u);
  put(24, marker);
  ASSERT_NO_FATAL_FAILURE(execute({0x7e300300, 0})); // v_mov_b32 v24, v0
  EXPECT_EQ(read(24), marker);
  ASSERT_NO_FATAL_FAILURE(execute({0x7e020318, 0})); // v_mov_b32 v1, v24 reads v0.
  EXPECT_EQ(read(1), 1u);
  put(256 + 1, marker);
  put(256 + 12, marker);
  ASSERT_NO_FATAL_FAILURE(execute({0xd3d94001, 0x18000100})); // v_accvgpr_write_b32 a1, v0
  ASSERT_NO_FATAL_FAILURE(execute({0xd3d9400c, 0x18000100})); // v_accvgpr_write_b32 a12, v0
  EXPECT_EQ(read(256 + 1), 1u);
  EXPECT_EQ(read(256 + 12), marker);
  put(256 + 4, marker);
  put(256 + 8, marker);
  put(2, 0);
  // The four outputs at a[4:7] fit, those at a[8:11] do not.
  ASSERT_NO_FATAL_FAILURE(execute({0xd3c58004, 0x02020500}));
  ASSERT_NO_FATAL_FAILURE(execute({0xd3c58008, 0x02020500}));
  EXPECT_EQ(read(256 + 4), 0u);
  EXPECT_EQ(read(256 + 8), marker);

  put(0, 0x3f800000); // v0 and v1 are 1.0, stale v24 holds 2.0.
  put(1, 0x3f800000);
  put(24, 0x40000000);
  // v_mfma_f32_16x16x4f32 v[4:7], va, vb, 0. Both out-of-range inputs select v0.
  for (const auto source : {0x02020300u, 0x02020318u, 0x02023100u}) {
    put(4, marker);
    ASSERT_NO_FATAL_FAILURE(execute({0xd3c50004, source}));
    EXPECT_EQ(read(4), 0x40800000u);
  }
}

TEST_P(Cdna2VectorRegression, LegacyIntegerMadAndPackedSaturation) {
  put(0, 0xffff);
  put(1, 2);
  put(2, 0);
  wave_->set_exec(0x8000000000000001ULL);
  auto legacy = vop3(491);
  legacy[0] |= 1u << 15;
  check(legacy, 0x0000ffff);
  put(0, 0x7fff);
  legacy = vop3(492);
  legacy[0] |= 1u << 15;
  check(legacy, 0x00007fff);
  // V_PK_MUL_LO_U16 clamp and V_PK_SUB_U16 clamp use per-half saturation.
  put(0, 0x7fffffff);
  put(1, 0x08070605);
  check({0xd381c004, 0x18020300}, 0xffffffff);
  check({0xd38bc004, 0x18020300}, 0x77f8f9fa);
}

TEST_P(Cdna2VectorRegression, MadAndFmaUseDifferentDenormalAndZeroRules) {
  wave_->set_mode_raw(0xf0);
  put(0, 1); // Smallest positive F32 subnormal.
  put(1, 0x3f800000);
  put(2, 0);
  check(vop3(449), 0);
  check(vop3(459), 1);
  put(0, 0);
  put(1, 0x7f800000);
  put(2, 0x40400000);
  check(vop3(448), 0x40400000);
  // MI200's VOP3 OMOD enable still follows MODE even though MAD flushes its arithmetic.
  put(0, 0x3f800000);
  put(1, 0x40000000);
  put(2, 0x3f800000);
  check(vop3(449, 0, 1), 0x40400000);
  wave_->set_mode_raw(0);
  check(vop3(449, 0, 1), 0x40c00000);
  wave_->set_mode_raw(0xf0);
  // 1-2^-46 rounds to one before the subtraction; a fused operation keeps -2^-46.
  constexpr uint32_t x = 0x3f800001u, y = 0x3f7ffffeu, minus_one = 0xbf800000u;
  put(0, x);
  put(1, y);
  put(2, minus_one);
  check(vop3(449), 0);
  check(vop3(459), 0xa8800000);
  check({0x30080300u, minus_one}, 0); // MADAK v4,v0,v1,-1.
  put(1, minus_one);
  check({0x2e080300u, y}, 0); // MADMK v4,v0,y,v1.
  put(1, y);
  for (const auto &encoding : {std::array{0x2c080300u, 0u}, std::array{0xd1160004u, 0x00020300u}}) {
    put(4, minus_one);
    ASSERT_NO_FATAL_FAILURE(execute(encoding));
    EXPECT_EQ(cu_->read_vgpr_storage(wave_->vgpr_alloc().base + 4, 0), 0u);
  }
}

TEST_P(Cdna2VectorRegression, HalfMadRoundsTheProductBeforeTheAdd) {
  // (256 + 2^-2) * (1 - 2^-10) - 256 = -2^-12 if fused, zero as separate half ops.
  wave_->set_mode_raw(0xf0);
  put(0, 0x5c01);
  put(1, 0x3bfe);
  put(2, 0xdc00);
  check(vop3(515), 0, 0);          // MAD_F16
  check(vop3(490), 0);             // MAD_LEGACY_F16 clears it.
  check(vop3(518), 0x8c00, 0);     // FMA_F16
  check({0x4a080300u, 0xdc00}, 0); // MADAK_F16
  put(1, 0xdc00);
  check({0x48080300u, 0x3bfe}, 0); // MADMK_F16
  put(1, 0x3bfe);
  for (const auto &encoding : {std::array{0x46080300u, 0u}, std::array{0xd1230004u, 0x00020300u}}) {
    put(4, 0xdc00);
    ASSERT_NO_FATAL_FAILURE(execute(encoding));
    EXPECT_EQ(cu_->read_vgpr_storage(wave_->vgpr_alloc().base + 4, 0), 0u);
  }
}

TEST_P(Cdna2VectorRegression, ClassCompareUsesExplicitDestinationAndExecuteMask) {
  constexpr uint64_t mask = 0x123456789abcdef0ULL;
  constexpr uint64_t old_vcc = 0xaabbccdd11223344ULL;
  put(0, 0x3f800000);
  put(1, 1u << 8); // Positive normal.
  wave_->set_exec(mask);
  wave_->set_vcc_raw(old_vcc);
  ASSERT_NO_FATAL_FAILURE(execute({0xd0110006u, 0x00020300u}));
  EXPECT_EQ(wave_->exec(), mask);
  EXPECT_EQ(wave_->vcc(), old_vcc);
  EXPECT_EQ(cu_->read_sgpr(wave_->sgpr_alloc().base + 6), uint32_t(mask));
  EXPECT_EQ(cu_->read_sgpr(wave_->sgpr_alloc().base + 7), uint32_t(mask >> 32));
}

TEST_P(Cdna2VectorRegression, MatrixBroadcastUsesAdjacentBlocks) {
  // AMD's matrix instruction calculator (2ef91896) selects adjacent A blocks on gfx90a.
  // 4x4x1 F32, 16 blocks. In each pair of adjacent blocks, select the
  // first A block. All B entries are one, C is inline zero.
  for (unsigned lane = 0; lane < 64; ++lane)
    cu_->write_vgpr(wave_->vgpr_alloc().base, lane, std::bit_cast<uint32_t>(float(lane / 4)));
  put(1, 0x3f800000);
  ASSERT_NO_FATAL_FAILURE(execute({0xd3c20104, 0x02020300}));
  for (unsigned lane = 0; lane < 64; ++lane)
    EXPECT_EQ(cu_->read_vgpr_storage(wave_->vgpr_alloc().base + 4, lane),
              std::bit_cast<uint32_t>(float((lane / 4) & ~1u)));
}

TEST_P(Cdna2VectorRegression, MatrixInlineMinusOneAndDenormals) {
  put(0, 0);
  put(1, 0);
  // V_MFMA_I32_16X16X16I8 with an integer -1 accumulator.
  ASSERT_NO_FATAL_FAILURE(execute({0xd3d50004, 0x03060300}));
  EXPECT_EQ(cu_->read_vgpr_storage(wave_->vgpr_alloc().base + 4, 0), UINT32_MAX);

  // 16x16x4 F32, all A entries smallest subnormal, B = 1 and C = 0.
  put(0, 1);
  put(1, 0x3f800000);
  for (uint32_t mode : {0u, 0x30u}) {
    wave_->set_mode_raw(mode);
    ASSERT_NO_FATAL_FAILURE(execute({0xd3c50004, 0x02020300}));
    EXPECT_EQ(cu_->read_vgpr_storage(wave_->vgpr_alloc().base + 4, 0), mode ? 4u : 0u);
  }
  // 16x16x4 F16, all packed A entries subnormal, both halves of B are 1.
  put(0, 0x00010001);
  put(1, 0x00010001);
  put(2, 0x3c003c00);
  put(3, 0x3c003c00);
  ASSERT_NO_FATAL_FAILURE(execute({0xd3c90004, 0x02020500}));
  EXPECT_EQ(cu_->read_vgpr_storage(wave_->vgpr_alloc().base + 4, 0), 0u);
}

class LegacyMathRegression : public NumericEncodingRegression {};

TEST_P(LegacyMathRegression, LegacyMathDecodesAndExecutesBothForms) {
  put(0, 0x3f800000);
  check(vop1(cdna4::kVExpLegacyF32Vop1), 0x40000000); // exp2(1) = 2.
  check(vop3(cdna4::kVExpLegacyF32Vop3), 0x40000000);
  put(0, 0x40000000);
  check(vop1(cdna4::kVLogLegacyF32Vop1), 0x3f800000); // log2(2) = 1.
  check(vop3(cdna4::kVLogLegacyF32Vop3), 0x3f800000);
  wave_->set_mode_raw(3u << 4);
  put(0, 0xc3020000); // exp2(-130) flushes the output under either guest mode.
  check(vop1(cdna4::kVExpLegacyF32Vop1), 0);
  check(vop3(cdna4::kVExpLegacyF32Vop3), 0);
  put(0, 0xff800000); // exp2(-infinity).
  check(vop1(cdna4::kVExpLegacyF32Vop1), 0);
  for (uint32_t zero : {0u, 0x80000000u}) {
    put(0, zero);
    check(vop1(cdna4::kVLogLegacyF32Vop1), 0xff800000);
    check(vop3(cdna4::kVLogLegacyF32Vop3), 0xff800000);
  }
}

// The same truth table uses a full-word destination on gfx950, but a half write on gfx1250.
TEST_P(NumericEncodingRegression, Bitop3TruthTableAndHalfPublication) {
  const bool is_cdna5 = std::get<0>(GetParam()) == ROCJITSU_CODE_ARCH_CDNA5;
  constexpr uint32_t a = 0x1234abcdU, b = 0xff00aa55U, c = 0x55558888U;
  put(0, a);
  put(1, b);
  put(2, c);
  wave_->set_exec(0x3333333333333333ULL);
  for (bool half : {false, true}) {
    for (uint32_t table : {0x96U, 0xcaU}) {
      uint32_t result = table == 0x96 ? a ^ b ^ c : (a & b) | (~a & c);
      if (half) {
        result = (result & 0xffffU) | (is_cdna5 ? 0xdead0000U : 0U);
      }
      // Populate only encoding fields: ABS/NEG/OMOD encode the table, not source modifiers.
      auto encode = [&](auto inst, uint16_t op, uint32_t encoding) {
        inst.encoding = encoding;
        inst.op = op;
        inst.vdst = 4;
        inst.src0 = 256;
        inst.src1 = 257;
        inst.src2 = 258;
        inst.omod = table >> 6;
        inst.abs = (table >> 3) & 7;
        inst.neg = table & 7;
        inst.clamp = 1;
        return std::bit_cast<std::array<uint32_t, 2>>(inst);
      };
      if (is_cdna5) {
        check(encode(cdna5::Vop3MachineInst{},
                     half ? cdna5::kVBitop3B16Vop3 : cdna5::kVBitop3B32Vop3, 0x35),
              result);
      } else {
        check(encode(cdna4::Vop3MachineInst{},
                     half ? cdna4::kVBitop3B16Vop3 : cdna4::kVBitop3B32Vop3, 0x34),
              result);
      }
    }
  }
}

TEST_P(NumericEncodingRegression, PrngUsesFeedbackAndExecMask) {
  const bool is_cdna5 = std::get<0>(GetParam()) == ROCJITSU_CODE_ARCH_CDNA5;
  constexpr std::array<std::array<uint32_t, 2>, 4> cases{{
      {0, 0},
      {1, 2},
      {0x80000000, 197},
      {0xffffffff, 0xffffff3b},
  }};
  for (uint64_t mask : {~uint64_t{0}, uint64_t{0x3333333333333333}}) {
    wave_->set_exec(mask);
    for (const auto &[input, expected] : cases) {
      put(0, input);
      check(vop1(is_cdna5 ? cdna5::kVPrngB32Vop1 : cdna4::kVPrngB32Vop1), expected);
      check(is_cdna5 ? cdna5::build_vop3(cdna5::kVPrngB32Vop3, {.vdst = 4, .src0 = 256})
                     : vop3(cdna4::kVPrngB32Vop3),
            expected);
    }
    if (!is_cdna5)
      continue;

    // Distinct per-lane inputs distinguish source permutation from ignored DPP.
    // The partial EXEC mask keeps each selected source/destination pair active.
    auto dpp = std::bit_cast<cdna5::Vop1VopDpp16MachineInst>(std::array{
        cdna5::build_vop1(cdna5::kVPrngB32Vop1, {.src0 = amdgpu::SRC_DPP, .vdst = 4})[0], 0u});
    dpp.dpp_ctrl = 0xb1; // quad_perm:[1,0,3,2].
    dpp.bank_mask = dpp.row_mask = 15;
    auto dpp8 = std::bit_cast<cdna5::Vop1VopDpp8MachineInst>(std::array{
        cdna5::build_vop1(cdna5::kVPrngB32Vop1, {.src0 = amdgpu::SRC_DPP8_FI_0, .vdst = 4})[0],
        0u});
    dpp8.lane_sel_0 = 1;
    dpp8.lane_sel_1 = 0;
    dpp8.lane_sel_2 = 3;
    dpp8.lane_sel_3 = 2;
    dpp8.lane_sel_4 = 5;
    dpp8.lane_sel_5 = 4;
    dpp8.lane_sel_6 = 7;
    dpp8.lane_sel_7 = 6;
    for (const auto &words : {std::bit_cast<std::array<uint32_t, 2>>(dpp),
                              std::bit_cast<std::array<uint32_t, 2>>(dpp8)}) {
      put(4, 0xdeadbeef);
      for (unsigned lane = 0; lane < wave_->wf_size(); ++lane)
        cu_->write_vgpr(wave_->vgpr_alloc().base, lane, cases[lane % cases.size()][0]);
      std::unique_ptr<Instruction> instruction(decode_valid(*decoder_, words.data()));
      ASSERT_NE(instruction, nullptr);
      ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
      for (unsigned lane = 0; lane < wave_->wf_size(); ++lane)
        EXPECT_EQ(cu_->read_vgpr_storage(wave_->vgpr_alloc().base + 4, lane),
                  mask & (uint64_t{1} << lane) ? cases[(lane ^ 1) % cases.size()][1] : 0xdeadbeefu)
            << "lane=" << lane << " selector=" << (words[0] & 0x1ff);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Legacy, LegacyMathRegression,
    testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                     ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4),
                     testing::Bool()));

INSTANTIATE_TEST_SUITE_P(Cdna, NumericEncodingRegression,
                         testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_CDNA4,
                                                          ROCJITSU_CODE_ARCH_CDNA5),
                                          testing::Bool()));

class Rdna2NumericEncoding : public NumericEncodingRegression {};

INSTANTIATE_TEST_SUITE_P(Rdna2, Rdna2NumericEncoding,
                         testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_RDNA2),
                                          testing::Bool()));

TEST_P(LegacyMathRegression, UnsignedHalfwordProductClampFollowsTheVop3AndSdwaPolicy) {
  const bool saturates = std::get<0>(GetParam()) == ROCJITSU_CODE_ARCH_CDNA3;
  put(0, 0xffff);
  put(1, 32);
  for (const uint32_t clamp : {0u, 1u}) {
    const uint32_t expected = clamp && saturates ? 0xffffu : 0xffe0u;
    check({0xd1290004u | (clamp << 15), 0x00020300}, expected, 0); // VOP3
    check({0x520802f9, 0x04040400u | (clamp << 13)}, expected, 0); // SDWA, WORD_0, PAD
  }
}

TEST_P(Rdna2NumericEncoding, ClampsUnsignedHalfwordProductsAndDifferences) {
  put(0, 0xffff);
  put(1, 32);
  check({0xd7050004, 0x02020300}, 0xdeadffe0); // v_mul_lo_u16
  check({0xd7058004, 0x02020300}, 0xdeadffff); // v_mul_lo_u16 clamp
  put(0, 0x7fffffff);
  put(1, 0x08070605);
  check({0xcc0bc004, 0x1a020300}, 0x77f8f9fa); // v_pk_sub_u16 clamp
}

TEST_P(Rdna2NumericEncoding, LegacyFmaAndSdwaRounding) {
  put(0, 0);
  put(1, 0x7f800000);
  put(2, 0x40400000);
  check({0xd5400004, 0x040a0300}, 0x40400000); // v_fma_legacy_f32 0*inf+3
  wave_->set_mode_raw(0);
  put(0, 0x7bff);
  put(1, 0x7bff);
  // v_add_f16_sdwa div:2 dst_sel:WORD_0 dst_unused:UNUSED_PAD
  check({0x640802f9, 0x0404c400}, 0x7c00);
  wave_->set_mode_raw(1u << 9); // IEEE, FP exceptions disabled for trapping.
  wave_->set_trapsts(0);
  check({0x64080300, 0}, 0xdead7c00); // v_add_f16 v4,v0,v1
  EXPECT_NE(wave_->trapsts() & (1u << 3), 0u);
  put(0, 0x7fc00000);
  put(1, 0x3f800000);
  // v_cmp_lt_f32_e64 vcc_lo,v0,v1 clamp signals invalid on a quiet NaN.
  std::array<uint32_t, 2> compare_words{0xd401806a, 0x02020300};
  std::unique_ptr<Instruction> compare(decode_valid(*decoder_, compare_words.data()));
  ASSERT_NE(compare, nullptr);
  ASSERT_TRUE(cu_->execute_instruction(compare.get(), *wave_).succeeded());
  EXPECT_EQ(wave_->trapsts() & 9u, 9u); // Newly raised INVALID preserves sticky overflow.
}

TEST_P(Rdna2NumericEncoding, HalfOverflowFollowsGuestRoundingBeforeSaturation) {
  // Add each magnitude to 65504, the largest finite half. Each mask names the
  // modes that overflow (nearest-even, up, down, zero) for either result sign.
  struct Case {
    uint16_t addend;
    uint8_t positive_overflow;
    uint8_t negative_overflow;
  };
  constexpr Case cases[] = {
      {0x0000, 0, 0},     // exact maximum
      {0x3c00, 0x2, 0x4}, // +1: below halfway
      {0x4c00, 0x3, 0x5}, // +16: nearest-even carries
      {0x4fc0, 0x3, 0x5}, // +31: inward rounding still fits
      {0x5000, 0xf, 0xf}, // +32: even truncation needs exponent 16
  };
  for (uint32_t width : {32u, 64u}) {
    wave_->halt();
    wave_ = cu_->dispatch_wf(0, 0, 106, 256, width);
    ASSERT_NE(wave_, nullptr);
    wave_->set_exec(uint64_t{1} << (width - 1));
    for (uint32_t mode = 0; mode < 4; ++mode) {
      for (bool negative : {false, true}) {
        const uint16_t sign = negative ? 0x8000 : 0;
        put(0, sign | 0x7bff);
        for (const Case &c : cases) {
          put(1, sign | c.addend);
          const bool overflow =
              (negative ? c.negative_overflow : c.positive_overflow) & (1u << mode);
          const uint32_t flags = (overflow ? 1u << 3 : 0u) | (c.addend ? 1u << 5 : 0u);
          const bool infinity = overflow && (mode == 0 || mode == (negative ? 2u : 1u));
          for (bool saturate : {false, true}) {
            wave_->set_mode_raw((mode << 2) | (saturate ? amdgpu::Wavefront::FP16_OVFL_BIT : 0u));
            const uint16_t raw = sign | (infinity && !saturate ? 0x7c00 : 0x7bff);
            SCOPED_TRACE(testing::Message()
                         << "width=" << width << " round=" << mode << " negative=" << negative
                         << " addend=" << c.addend << " saturate=" << saturate);
            wave_->set_trapsts(0);
            check({0x64080300, 0}, 0xdead0000u | raw); // v_add_f16_e32 v4,v0,v1
            EXPECT_EQ(wave_->trapsts(), flags);
            wave_->set_trapsts(0);
            check({0xd5328004, 0x02020300}, 0xdead0000u | (negative ? 0u : 0x3c00u)); // e64 clamp
            EXPECT_EQ(wave_->trapsts(), flags);
          }
        }
      }
    }
  }
}

TEST_P(Rdna2NumericEncoding, HalfUnderflowUsesTininessAfterGuestRounding) {
  struct Case {
    uint16_t lhs, rhs;
    uint8_t positive_underflow, negative_underflow;
    bool exact;
  };
  constexpr Case cases[] = {
      {0x0400, 0x3800, 0, 0, true},      // An exact subnormal product.
      {0x0575, 0x39dd, 0xc, 0xa, false}, // Above the unbounded-exponent rounding midpoint.
      {0x04c0, 0x3abc, 0xd, 0xb, false}, // Below it.
  };
  for (uint32_t mode = 0; mode < 4; ++mode) {
    wave_->set_mode_raw((3u << 6) | (mode << 2)); // Preserve input and output denormals.
    for (const Case &c : cases) {
      for (bool negative : {false, true}) {
        SCOPED_TRACE(testing::Message() << "round=" << mode << " negative=" << negative
                                        << " lhs=" << c.lhs << " rhs=" << c.rhs);
        const uint16_t sign = negative ? 0x8000 : 0;
        put(0, sign | c.lhs);
        put(1, c.rhs);
        const bool rounds_up = mode == 0 || mode == (negative ? 2u : 1u);
        const uint16_t magnitude = c.exact ? 0x0200 : rounds_up ? 0x0400 : 0x03ff;
        const bool tiny = (negative ? c.negative_underflow : c.positive_underflow) & (1u << mode);
        wave_->set_trapsts(0);
        check({0x6a080300, 0}, 0xdead0000u | sign | magnitude); // v_mul_f16_e32 v4,v0,v1
        EXPECT_EQ(wave_->trapsts(), (tiny ? 1u << 4 : 0u) | (c.exact ? 0u : 1u << 5));
      }
    }
  }
  // An exact minimum subnormal times one separates MODE input and output flushing.
  put(0, 1);
  put(1, 0x3c00);
  for (uint32_t denorm = 0; denorm < 4; ++denorm) {
    wave_->set_mode_raw(denorm << 6);
    wave_->set_trapsts(0);
    check({0x6a080300, 0}, 0xdead0000u | (denorm == 3 ? 1u : 0u));
    EXPECT_EQ(wave_->trapsts() & (1u << 1), denorm & 1u ? 1u << 1 : 0u);
  }
}

TEST_P(Rdna2NumericEncoding, DirectLdsBroadcastUsesM0TypeAndWaveAllocation) {
  wave_->set_lds_base(256);
  wave_->set_lds_size(256);
  wave_->lds().write32(256 + 64, 0x7654ff80);
  for (const auto &[type, expected] : std::array<std::pair<uint32_t, uint32_t>, 5>{
           {{0, 0x80}, {1, 0xff80}, {2, 0x7654ff80}, {4, 0xffffff80}, {5, 0xffffff80}}}) {
    wave_->set_m0((type << 16) | 64);
    check({0x7e0802fe, 0}, expected); // v_mov_b32 v4,src_lds_direct
    const std::array<uint32_t, 2> words = {0xd7600000, 0x02010afe};
    std::unique_ptr<Instruction> lane_read(decode_valid(*decoder_, words.data()));
    ASSERT_NE(lane_read, nullptr);
    ASSERT_TRUE(cu_->execute_instruction(lane_read.get(), *wave_).succeeded());
    EXPECT_EQ(cu_->read_sgpr(wave_->sgpr_alloc().base), expected);
  }
  wave_->lds().write32(512, 0x12345678);
  wave_->set_m0((2u << 16) | 256);
  check({0x7e0802fe, 0}, 0);
}

} // namespace
} // namespace rocjitsu
