// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"

namespace {

using namespace rocjitsu;
using namespace rocjitsu::test::cdna5;

class ForceScalarGuard {
public:
  ForceScalarGuard() : original_(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(original_); }

private:
  bool original_;
};

TEST(Gfx1250ExecutionTest, TargetProvidesImmutableExecutionBackend) {
  const IsaTargetDescriptor *target = default_isa_target_registry().find("gfx1250");
  ASSERT_NE(target, nullptr);
  EXPECT_TRUE(target->supports_execution);
  EXPECT_TRUE(cdna5::Operand::full_execution_backend_complete());
}

TEST(Gfx1250ExecutionTest, DivScaleWritesExplicitSdstMask) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  constexpr uint32_t kOne = 0x3f800000u;
  constexpr uint32_t kTwoTo8 = 0x43800000u;
  constexpr uint32_t kTwoTo100 = 0x71800000u;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  auto write_vgpr = [&](uint32_t reg, uint32_t value) {
    cu->write_vgpr(vgpr_base + reg, kLane, value);
  };
  auto read_vgpr = [&](uint32_t reg) { return cu->read_vgpr(vgpr_base + reg, kLane); };
  auto write_sgpr = [&](uint32_t reg, uint32_t value) {
    cu->write_sgpr(wf->sgpr_alloc().base + reg, value);
  };

  write_vgpr(1, kOne);
  write_vgpr(2, kTwoTo100);
  wf->set_vcc(0x5a5a5a5au);
  const std::array<uint32_t, 2> null_sdst_words = {
      0xd6fc7c00u, 0x040a0301u}; // v_div_scale_f32 v0, null, v1, v1, v2
  cdna5::VDivScaleF32Vop3SdstEnc null_sdst(null_sdst_words.data());
  null_sdst.execute_impl(*wf);
  EXPECT_EQ(wf->vcc(), 0x5a5a5a5au);
  EXPECT_EQ(read_vgpr(0), 0x5f800000u); // 2^64

  write_sgpr(7, kTwoTo8);
  write_vgpr(3, kOne);
  wf->set_vcc(0xa5a5a5a5u);
  const std::array<uint32_t, 2> normal_null_sdst_words = {
      0xd6fc7c09u, 0x040c0e07u}; // v_div_scale_f32 v9, null, s7, s7, v3
  cdna5::VDivScaleF32Vop3SdstEnc normal_null_sdst(normal_null_sdst_words.data());
  normal_null_sdst.execute_impl(*wf);
  EXPECT_EQ(wf->vcc(), 0xa5a5a5a5u);
  EXPECT_EQ(read_vgpr(9), kTwoTo8);

  write_vgpr(4, kOne);
  write_vgpr(5, kTwoTo100);
  wf->set_vcc(0);
  const std::array<uint32_t, 2> vcc_sdst_words = {
      0xd6fc6a03u, 0x04160904u}; // v_div_scale_f32 v3, vcc_lo, v4, v4, v5
  cdna5::VDivScaleF32Vop3SdstEnc vcc_sdst(vcc_sdst_words.data());
  vcc_sdst.execute_impl(*wf);
  EXPECT_EQ(wf->vcc(), 1u);
  EXPECT_EQ(read_vgpr(3), 0x5f800000u);

  write_vgpr(7, kOne);
  write_vgpr(8, kTwoTo100);
  write_sgpr(3, 0xfefefefeu);
  wf->set_vcc(0x12345678u);
  const std::array<uint32_t, 2> sgpr_sdst_words = {
      0xd6fc0206u, 0x04220f07u}; // v_div_scale_f32 v6, s2, v7, v7, v8
  cdna5::VDivScaleF32Vop3SdstEnc sgpr_sdst(sgpr_sdst_words.data());
  sgpr_sdst.execute_impl(*wf);
  EXPECT_EQ(wf->vcc(), 0x12345678u);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 2), 0x12345679u);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 3), 0xfefefefeu);
  EXPECT_EQ(read_vgpr(6), 0x5f800000u);
}

TEST(Gfx1250ExecutionTest, VMovB16HighVdstMergesIntoLowPhysicalVgpr) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 1, kLane, 0xAAAA5555u);
  cu->write_vgpr(vgpr_base + 129, kLane, 0xDEADBEEFu);

  const std::array<uint32_t, 1> words = {0x7F023880u}; // v_mov_b16_e32 v1.h, 0
  cdna5::VMovB16Vop1 high_half_mov(words.data());
  high_half_mov.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, kLane), 0x00005555u);
  EXPECT_EQ(cu->read_vgpr(vgpr_base + 129, kLane), 0xDEADBEEFu);
}

TEST(Gfx1250ExecutionTest, VNotB16HighVdstMergesIntoLowPhysicalVgpr) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 0, kLane, 0x000000FFu);
  cu->write_vgpr(vgpr_base + 1, kLane, 0xAAAA5555u);
  cu->write_vgpr(vgpr_base + 129, kLane, 0xDEADBEEFu);

  const std::array<uint32_t, 1> words = {0x7F02D300u}; // v_not_b16_e32 v1.h, v0.l
  cdna5::VNotB16Vop1 high_half_not(words.data());
  high_half_not.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, kLane), 0xFF005555u);
  EXPECT_EQ(cu->read_vgpr(vgpr_base + 129, kLane), 0xDEADBEEFu);
}

TEST(Gfx1250ExecutionTest, VAddF16HighVdstMergesIntoLowPhysicalVgpr) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 0, kLane, 0x00003C00u);
  cu->write_vgpr(vgpr_base + 1, kLane, 0xAAAA5555u);
  cu->write_vgpr(vgpr_base + 2, kLane, 0x00003C00u);
  cu->write_vgpr(vgpr_base + 129, kLane, 0xDEADBEEFu);

  const std::array<uint32_t, 1> words = {0x65020500u}; // v_add_f16_e32 v1.h, v0.l, v2.l
  cdna5::VAddF16Vop2 high_half_add(words.data());
  high_half_add.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, kLane), 0x40005555u);
  EXPECT_EQ(cu->read_vgpr(vgpr_base + 129, kLane), 0xDEADBEEFu);
}

TEST(Gfx1250ExecutionTest, IreeF16ReductionTailKeepsLane31Sum) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec(0xffffffffu);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);

  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  const uint32_t packed_1_2 = 0x40003c00u;
  const uint32_t packed_3_4 = 0x44004200u;
  const uint32_t packed_5_6 = 0x46004500u;
  const uint32_t packed_7_8 = 0x48004700u;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu->write_vgpr(vgpr_base + 10, lane, packed_7_8);
    cu->write_vgpr(vgpr_base + 11, lane, packed_1_2);
    cu->write_vgpr(vgpr_base + 16, lane, packed_5_6);
    cu->write_vgpr(vgpr_base + 17, lane, packed_3_4);
  }

  const std::array<std::array<uint32_t, 3>, 20> words = {{
      {0x64021680u, 0, 0},                     // v_add_f16_e32 v1, 0, v11
      {0x32041690u, 0, 0},                     // v_lshrrev_b32_e32 v2, 16, v11
      {0x64020501u, 0, 0},                     // v_add_f16_e32 v1, v1, v2
      {0x32042290u, 0, 0},                     // v_lshrrev_b32_e32 v2, 16, v17
      {0x64022301u, 0, 0},                     // v_add_f16_e32 v1, v1, v17
      {0x64020501u, 0, 0},                     // v_add_f16_e32 v1, v1, v2
      {0x32042090u, 0, 0},                     // v_lshrrev_b32_e32 v2, 16, v16
      {0x64022101u, 0, 0},                     // v_add_f16_e32 v1, v1, v16
      {0x64020501u, 0, 0},                     // v_add_f16_e32 v1, v1, v2
      {0x32041490u, 0, 0},                     // v_lshrrev_b32_e32 v2, 16, v10
      {0x64021501u, 0, 0},                     // v_add_f16_e32 v1, v1, v10
      {0x64020501u, 0, 0},                     // v_add_f16_e32 v1, v1, v2
      {0xd5320001u, 0x000202fau, 0xff08b101u}, // quad_perm:[1,0,3,2]
      {0xd5320001u, 0x000202fau, 0xff084e01u}, // quad_perm:[2,3,0,1]
      {0xd5320001u, 0x000202fau, 0xff094101u}, // row_half_mirror
      {0xd5320001u, 0x000202fau, 0xff094001u}, // row_mirror
      {0xd65c0802u, 0x03058301u, 0},           // v_permlanex16_b32 v2, v1, -1, -1
      {0x64020302u, 0, 0},                     // v_add_f16_e32 v1, v2, v1
      {0xd7600000u, 0x02013f01u, 0},           // v_readlane_b32 s0, v1, 31
      {0xa4808000u, 0, 0},                     // s_add_f16 s0, s0, 0
  }};

  for (const auto &inst_words : words) {
    std::unique_ptr<Instruction> inst(decoder->decode(inst_words.data()));
    ASSERT_NE(inst, nullptr);
    cu->execute_instruction(inst.get(), *wf);
  }

  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 0) & 0xffffu, 0x6480u);
}

TEST(Gfx1250ExecutionTest, VFmacF16Vop3HighVdstUsesHighHalfAddend) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 0, kLane, 0x00003C00u);
  cu->write_vgpr(vgpr_base + 1, kLane, 0x40003C00u);
  cu->write_vgpr(vgpr_base + 2, kLane, 0x00003C00u);

  const std::array<uint32_t, 2> words = {
      0xD5364001u, // v_fmac_f16 v1.h, v0.l, v2.l
      0x02020500u,
  };
  cdna5::VFmacF16Vop3 high_half_fmac(words.data());
  high_half_fmac.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, kLane), 0x42003C00u);
}

TEST(Gfx1250ExecutionTest, VFmacF16Vop2HighVdstUsesHighHalfAddend) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 0, kLane, 0x00003C00u);
  cu->write_vgpr(vgpr_base + 1, kLane, 0x40003C00u);
  cu->write_vgpr(vgpr_base + 2, kLane, 0x00003C00u);
  cu->write_vgpr(vgpr_base + 129, kLane, 0x3C003C00u);

  const std::array<uint32_t, 1> words = {0x6D020500u}; // v_fmac_f16_e32 v1.h, v0.l, v2.l
  cdna5::VFmacF16Vop2 high_half_fmac(words.data());
  high_half_fmac.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, kLane), 0x42003C00u);
  EXPECT_EQ(cu->read_vgpr(vgpr_base + 129, kLane), 0x3C003C00u);
}

TEST(Gfx1250ExecutionTest, VMadU32LiteralTimesScalarAddsVector) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  write_wave_sgpr(*cu, *wf, 3, 1);
  cu->write_vgpr(vgpr_base + 4, kLane, 0x24u);

  const std::array<uint32_t, 3> words = {
      0xD6350004u, // v_mad_u32 v4, 0x48, s3, v4
      0x041006FFu,
      0x00000048u,
  };
  cdna5::VMadU32Vop3 mad(words.data());
  mad.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 4, kLane), 0x6Cu);
}

TEST(Gfx1250LiteralOperandTest, SplitBackendPreservesSignedAndEncodingSemantics) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  ASSERT_TRUE(cdna5::Operand::full_execution_backend_complete());
  amdgpu::RegisterAccess regs(*wf);

  struct LiteralCase {
    uint32_t encoded;
    uint64_t signed_value;
  };
  constexpr std::array cases{
      LiteralCase{0x7fffffffu, 0x000000007fffffffULL},
      LiteralCase{0x80000000u, 0xffffffff80000000ULL},
      LiteralCase{0xffffffffu, 0xffffffffffffffffULL},
  };

  for (const auto &[literal, signed_value] : cases) {
    SCOPED_TRACE(::testing::Message() << "literal=" << literal);

    const auto signed_mad_base = cdna5::build_vop3(
        cdna5::kVMadNcI64I32Vop3, {.vdst = 4, .src0 = 129, .src1 = 129, .src2 = 255});
    const std::array signed_mad_words{signed_mad_base[0], signed_mad_base[1], literal};
    std::unique_ptr<Instruction> signed_mad_decoded(decoder->decode(signed_mad_words.data()));
    ASSERT_NE(signed_mad_decoded, nullptr);
    EXPECT_EQ(signed_mad_decoded->mnemonic(), "v_mad_nc_i64_i32");
    EXPECT_EQ(signed_mad_decoded->size(), 12);
    auto *signed_mad = dynamic_cast<cdna5::VMadNcI64I32Vop3 *>(signed_mad_decoded.get());
    ASSERT_NE(signed_mad, nullptr);

    const Operand *signed_addend = signed_mad->src_operand(2);
    ASSERT_NE(signed_addend, nullptr);
    EXPECT_EQ(signed_addend->name(), std::format("0x{:x}", literal));
    EXPECT_EQ(static_cast<uint32_t>(signed_addend->encoding_value()), literal);
    EXPECT_FALSE(signed_addend->literal64_value().has_value());
    EXPECT_EQ(regs.read_lane64(*signed_addend, 0), signed_value);
    signed_mad->execute_impl(*wf);
    EXPECT_EQ(regs.read_lane64(*signed_mad->dst_operand(0), 0), signed_value + 1u);

    const auto unsigned_mad_base = cdna5::build_vop3(
        cdna5::kVMadNcU64U32Vop3, {.vdst = 6, .src0 = 129, .src1 = 129, .src2 = 255});
    const std::array unsigned_mad_words{unsigned_mad_base[0], unsigned_mad_base[1], literal};
    std::unique_ptr<Instruction> unsigned_mad_decoded(decoder->decode(unsigned_mad_words.data()));
    ASSERT_NE(unsigned_mad_decoded, nullptr);
    auto *unsigned_mad = dynamic_cast<cdna5::VMadNcU64U32Vop3 *>(unsigned_mad_decoded.get());
    ASSERT_NE(unsigned_mad, nullptr);

    const Operand *unsigned_addend = unsigned_mad->src_operand(2);
    ASSERT_NE(unsigned_addend, nullptr);
    EXPECT_FALSE(unsigned_addend->literal64_value().has_value());
    EXPECT_EQ(regs.read_lane64(*unsigned_addend, 0), static_cast<uint64_t>(literal));
    unsigned_mad->execute_impl(*wf);
    EXPECT_EQ(regs.read_lane64(*unsigned_mad->dst_operand(0), 0),
              static_cast<uint64_t>(literal) + 1u);

    const auto scalar_base =
        cdna5::build_sop2(cdna5::kSAshrI64Sop2, {.ssrc0 = 255, .ssrc1 = 128, .sdst = 0});
    const std::array scalar_words{scalar_base[0], literal};
    std::unique_ptr<Instruction> scalar_decoded(decoder->decode(scalar_words.data()));
    ASSERT_NE(scalar_decoded, nullptr);
    EXPECT_EQ(scalar_decoded->mnemonic(), "s_ashr_i64");
    EXPECT_EQ(scalar_decoded->size(), 8);
    auto *scalar = dynamic_cast<cdna5::SAshrI64Sop2 *>(scalar_decoded.get());
    ASSERT_NE(scalar, nullptr);

    const Operand *scalar_value = scalar->src_operand(0);
    ASSERT_NE(scalar_value, nullptr);
    EXPECT_FALSE(scalar_value->literal64_value().has_value());
    EXPECT_EQ(regs.read_scalar64(*scalar_value), signed_value);
    scalar->execute_impl(*wf);
    EXPECT_EQ(regs.read_scalar64(*scalar->dst_operand(0)), signed_value);

    const auto b64_base =
        cdna5::build_sop2(cdna5::kSAndB64Sop2, {.ssrc0 = 255, .ssrc1 = 193, .sdst = 2});
    const std::array b64_words{b64_base[0], literal};
    std::unique_ptr<Instruction> b64_decoded(decoder->decode(b64_words.data()));
    ASSERT_NE(b64_decoded, nullptr);
    auto *b64 = dynamic_cast<cdna5::SAndB64Sop2 *>(b64_decoded.get());
    ASSERT_NE(b64, nullptr);

    const Operand *b64_value = b64->src_operand(0);
    ASSERT_NE(b64_value, nullptr);
    EXPECT_FALSE(b64_value->literal64_value().has_value());
    EXPECT_EQ(regs.read_scalar64(*b64_value), static_cast<uint64_t>(literal));
    b64->execute_impl(*wf);
    EXPECT_EQ(regs.read_scalar64(*b64->dst_operand(0)), static_cast<uint64_t>(literal));
  }
}

TEST(Gfx1250LiteralOperandTest, NegativeI64CompareCoversScalarAndAvailableSimdPath) {
  ForceScalarGuard force_scalar_guard;
  const auto run_case = [](bool force_scalar) {
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");
    util::set_force_scalar_for_testing(force_scalar);

    Gfx1250Sim sim;
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(0x3u);

    const uint32_t vgpr_base = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < 2; ++lane) {
      cu->write_vgpr(vgpr_base, lane, 0u);
      cu->write_vgpr(vgpr_base + 1, lane, 0u);
    }

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
    ASSERT_NE(decoder, nullptr);
    const auto compare_base =
        cdna5::build_vop3(cdna5::kVCmpLtI64Vop3, {.vdst = 0, .src0 = 255, .src1 = 256});
    const std::array compare_words{compare_base[0], compare_base[1], 0xffffffffu};
    std::unique_ptr<Instruction> compare(decoder->decode(compare_words.data()));
    ASSERT_NE(compare, nullptr);
    EXPECT_EQ(compare->mnemonic(), "v_cmp_lt_i64");
    EXPECT_EQ(compare->size(), 12);

    auto *typed_compare = dynamic_cast<cdna5::VCmpLtI64Vop3 *>(compare.get());
    ASSERT_NE(typed_compare, nullptr);
    if (!force_scalar) {
      EXPECT_TRUE(amdgpu::try_execute_vopc64_vop3_int_simd<int64_t>(
          *typed_compare, *wf, [](auto a, auto b) { return a < b; }));
      EXPECT_EQ(read_wave_sgpr(*cu, *wf, 0), 0x3u);
      write_wave_sgpr(*cu, *wf, 0, 0u);
      write_wave_sgpr(*cu, *wf, 1, 0u);
    }

    cu->execute_instruction(compare.get(), *wf);
    EXPECT_EQ(read_wave_sgpr(*cu, *wf, 0), 0x3u);
  };

  run_case(true);
  if constexpr (util::has_stdx_simd && !UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS)
    run_case(false);
}

TEST(Gfx1250LiteralOperandTest, ScalarMaskOperandsRejectLiteralMarkers) {
  struct TestCase {
    const char *name;
    std::array<uint32_t, 2> encoding;
  };
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  for (const uint16_t marker : {uint16_t{254}, uint16_t{255}}) {
    const std::array test_cases{
        TestCase{"v_cndmask_b32", cdna5::build_vop3(cdna5::kVCndmaskB32Vop3,
                                                    {.src0 = 128, .src1 = 128, .src2 = marker})},
        TestCase{"v_cndmask_b16", cdna5::build_vop3(cdna5::kVCndmaskB16Vop3,
                                                    {.src0 = 128, .src1 = 128, .src2 = marker})},
        TestCase{"v_add_co_ci_u32",
                 cdna5::build_vop3_sdst_enc(cdna5::kVAddCoCiU32Vop3SdstEnc,
                                            {.src0 = 128, .src1 = 128, .src2 = marker})},
        TestCase{"v_sub_co_ci_u32",
                 cdna5::build_vop3_sdst_enc(cdna5::kVSubCoCiU32Vop3SdstEnc,
                                            {.src0 = 128, .src1 = 128, .src2 = marker})},
        TestCase{"v_subrev_co_ci_u32",
                 cdna5::build_vop3_sdst_enc(cdna5::kVSubrevCoCiU32Vop3SdstEnc,
                                            {.src0 = 128, .src1 = 128, .src2 = marker})},
    };
    for (const TestCase &test_case : test_cases) {
      SCOPED_TRACE(test_case.name);
      SCOPED_TRACE(marker);
      const std::array words{test_case.encoding[0], test_case.encoding[1], 0xffffffffu, 0u};
      EXPECT_THROW(std::unique_ptr<Instruction>(decoder->decode(words.data())), util::InvalidInst);
    }
  }
}

TEST(Gfx1250LiteralOperandTest, PkF32LiteralReplicatesAndUsesAvailableSimdPath) {
  ForceScalarGuard force_scalar_guard;
  const auto run_case = [](bool force_scalar) {
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");
    util::set_force_scalar_for_testing(force_scalar);

    constexpr uint32_t literal = 0x3f800000u;
    constexpr uint64_t replicated =
        (static_cast<uint64_t>(literal) << 32) | static_cast<uint64_t>(literal);
    const auto add_base = cdna5::build_vop3p(cdna5::kVPkAddF32Vop3p,
                                             {.vdst = 0, .src0 = 255, .src1 = 128, .opsel_hi = 3});
    const std::array add_words{add_base[0], add_base[1], literal};

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
    ASSERT_NE(decoder, nullptr);
    std::unique_ptr<Instruction> add(decoder->decode(add_words.data()));
    ASSERT_NE(add, nullptr);
    auto *typed_add = dynamic_cast<cdna5::VPkAddF32Vop3p *>(add.get());
    ASSERT_NE(typed_add, nullptr);

    Gfx1250Sim sim;
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(0x3u);

    const Operand *literal_operand = typed_add->src_operand(0);
    ASSERT_NE(literal_operand, nullptr);
    EXPECT_EQ(static_cast<uint32_t>(literal_operand->encoding_value()), literal);
    EXPECT_FALSE(literal_operand->literal64_value().has_value());
    EXPECT_EQ(amdgpu::RegisterAccess(*wf).read_lane64(*literal_operand, 0), replicated);

    const uint32_t vgpr_base = wf->vgpr_alloc().base;
    if (!force_scalar) {
      EXPECT_TRUE(amdgpu::try_execute_vop3p_pk_binary_f32_simd(
          *typed_add, *wf, 0u, 3u, [](auto a, auto b) { return a + b; }));
      for (uint32_t lane = 0; lane < 2; ++lane) {
        EXPECT_EQ(cu->read_vgpr(vgpr_base, lane), literal);
        EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, lane), literal);
        cu->write_vgpr(vgpr_base, lane, 0u);
        cu->write_vgpr(vgpr_base + 1, lane, 0u);
      }
    }

    cu->execute_instruction(add.get(), *wf);
    for (uint32_t lane = 0; lane < 2; ++lane) {
      EXPECT_EQ(cu->read_vgpr(vgpr_base, lane), literal);
      EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, lane), literal);
    }
  };

  run_case(true);
  if constexpr (util::has_stdx_simd)
    run_case(false);
}

TEST(Gfx1250LiteralOperandTest, PkF32MixedLiteralVgprSourcesUseAvailableSimdPath) {
  if constexpr (!util::has_stdx_simd)
    GTEST_SKIP() << "requires stdx SIMD";

  ForceScalarGuard force_scalar_guard;
  util::set_force_scalar_for_testing(false);
  enum class Operation { Add, Mul, Fma };
  struct TestCase {
    Operation operation;
    uint16_t opcode;
    uint32_t literal_source;
    float expected_lo;
    float expected_hi;
  };
  constexpr std::array test_cases{
      TestCase{Operation::Add, cdna5::kVPkAddF32Vop3p, 0, 7.0f, 8.0f},
      TestCase{Operation::Add, cdna5::kVPkAddF32Vop3p, 1, 5.0f, 6.0f},
      TestCase{Operation::Mul, cdna5::kVPkMulF32Vop3p, 0, 10.0f, 12.0f},
      TestCase{Operation::Mul, cdna5::kVPkMulF32Vop3p, 1, 6.0f, 8.0f},
      TestCase{Operation::Fma, cdna5::kVPkFmaF32Vop3p, 0, 17.0f, 20.0f},
      TestCase{Operation::Fma, cdna5::kVPkFmaF32Vop3p, 1, 13.0f, 16.0f},
      TestCase{Operation::Fma, cdna5::kVPkFmaF32Vop3p, 2, 17.0f, 26.0f},
  };
  constexpr uint32_t kLiteral = 0x40000000u; // 2.0f
  constexpr uint64_t kReplicatedLiteral = (static_cast<uint64_t>(kLiteral) << 32) | kLiteral;

  for (const TestCase &test_case : test_cases) {
    SCOPED_TRACE(static_cast<uint32_t>(test_case.operation));
    SCOPED_TRACE(test_case.literal_source);
    cdna5::Vop3pBuilderFields fields;
    fields.vdst = 6;
    fields.src0 = 256;
    fields.src1 = 258;
    fields.src2 = 260;
    fields.opsel_hi = 3;
    if (test_case.literal_source == 0)
      fields.src0 = 255;
    else if (test_case.literal_source == 1)
      fields.src1 = 255;
    else
      fields.src2 = 255;

    auto base = cdna5::build_vop3p(test_case.opcode, fields);
    if (test_case.operation == Operation::Fma)
      base[0] |= uint32_t{1} << 14; // pad_14 is the src2 high-half selector.
    const std::array words{base[0], base[1], kLiteral};
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
    ASSERT_NE(decoder, nullptr);
    std::unique_ptr<Instruction> instruction(decoder->decode(words.data()));
    ASSERT_NE(instruction, nullptr);

    const Operand *literal_operand = instruction->src_operand(test_case.literal_source);
    ASSERT_NE(literal_operand, nullptr);

    Gfx1250Sim sim;
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(1u);
    EXPECT_EQ(amdgpu::RegisterAccess(*wf).read_lane64(*literal_operand, 0), kReplicatedLiteral);
    const uint32_t vgpr_base = wf->vgpr_alloc().base;
    cu->write_vgpr(vgpr_base, 0, std::bit_cast<uint32_t>(3.0f));
    cu->write_vgpr(vgpr_base + 1, 0, std::bit_cast<uint32_t>(4.0f));
    cu->write_vgpr(vgpr_base + 2, 0, std::bit_cast<uint32_t>(5.0f));
    cu->write_vgpr(vgpr_base + 3, 0, std::bit_cast<uint32_t>(6.0f));
    cu->write_vgpr(vgpr_base + 4, 0, std::bit_cast<uint32_t>(7.0f));
    cu->write_vgpr(vgpr_base + 5, 0, std::bit_cast<uint32_t>(8.0f));

    bool accepted = false;
    if (test_case.operation == Operation::Add) {
      auto *typed = dynamic_cast<cdna5::VPkAddF32Vop3p *>(instruction.get());
      ASSERT_NE(typed, nullptr);
      accepted = amdgpu::try_execute_vop3p_pk_binary_f32_simd(*typed, *wf, 0u, 3u,
                                                              [](auto a, auto b) { return a + b; });
    } else if (test_case.operation == Operation::Mul) {
      auto *typed = dynamic_cast<cdna5::VPkMulF32Vop3p *>(instruction.get());
      ASSERT_NE(typed, nullptr);
      accepted = amdgpu::try_execute_vop3p_pk_binary_f32_simd(*typed, *wf, 0u, 3u,
                                                              [](auto a, auto b) { return a * b; });
    } else {
      auto *typed = dynamic_cast<cdna5::VPkFmaF32Vop3p *>(instruction.get());
      ASSERT_NE(typed, nullptr);
      accepted = amdgpu::try_execute_vop3p_pk_ternary_f32_simd(
          *typed, *wf, 0u, 3u, 1u, [](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); });
    }
    EXPECT_TRUE(accepted);
    EXPECT_EQ(cu->read_vgpr(vgpr_base + 6, 0), std::bit_cast<uint32_t>(test_case.expected_lo));
    EXPECT_EQ(cu->read_vgpr(vgpr_base + 7, 0), std::bit_cast<uint32_t>(test_case.expected_hi));

    cu->write_vgpr(vgpr_base + 6, 0, 0u);
    cu->write_vgpr(vgpr_base + 7, 0, 0u);
    cu->execute_instruction(instruction.get(), *wf);
    EXPECT_EQ(cu->read_vgpr(vgpr_base + 6, 0), std::bit_cast<uint32_t>(test_case.expected_lo));
    EXPECT_EQ(cu->read_vgpr(vgpr_base + 7, 0), std::bit_cast<uint32_t>(test_case.expected_hi));
  }
}

TEST(Gfx1250LiteralOperandTest, PkF32MixedLiteralSourceSpecificSelectorFallsBackToScalar) {
  ForceScalarGuard force_scalar_guard;
  util::set_force_scalar_for_testing(false);
  constexpr uint32_t kLiteral = 0x40000000u; // 2.0f
  const auto base = cdna5::build_vop3p(cdna5::kVPkAddF32Vop3p,
                                       {.vdst = 4, .src0 = 255, .src1 = 258, .opsel_hi = 2});
  const std::array words{base[0], base[1], kLiteral};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> instruction(decoder->decode(words.data()));
  ASSERT_NE(instruction, nullptr);
  auto *typed = dynamic_cast<cdna5::VPkAddF32Vop3p *>(instruction.get());
  ASSERT_NE(typed, nullptr);

  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 2, 0, std::bit_cast<uint32_t>(5.0f));
  cu->write_vgpr(vgpr_base + 3, 0, std::bit_cast<uint32_t>(6.0f));

  EXPECT_FALSE(amdgpu::try_execute_vop3p_pk_binary_f32_simd(*typed, *wf, 0u, 2u,
                                                            [](auto a, auto b) { return a + b; }));
  cu->execute_instruction(instruction.get(), *wf);
  EXPECT_EQ(cu->read_vgpr(vgpr_base + 4, 0), std::bit_cast<uint32_t>(7.0f));
  EXPECT_EQ(cu->read_vgpr(vgpr_base + 5, 0), std::bit_cast<uint32_t>(8.0f));
}

TEST(Gfx1250ExecutionTest, PkF32AddMulSimdMatchesScalarWithPartialExec) {
  ForceScalarGuard force_scalar_guard;
  struct TestCase {
    uint16_t opcode;
    const char *name;
  };
  constexpr std::array test_cases{
      TestCase{cdna5::kVPkAddF32Vop3p, "add"},
      TestCase{cdna5::kVPkMulF32Vop3p, "mul"},
  };
  constexpr uint32_t kExec = 0xa5a5a5a5u;
  constexpr uint32_t kDstLoSeed = 0xdeadbeefu;
  constexpr uint32_t kDstHiSeed = 0xbaadf00du;

  for (const TestCase &test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    std::array<uint32_t, 64> scalar_result{};
    std::array<uint32_t, 64> simd_result{};
    const auto run_case = [&](bool force_scalar, std::array<uint32_t, 64> &result) {
      SCOPED_TRACE(force_scalar ? "scalar" : "simd");
      util::set_force_scalar_for_testing(force_scalar);

      const auto words = cdna5::build_vop3p(
          test_case.opcode,
          {.vdst = 4, .neg_hi = 2, .src0 = 256, .src1 = 258, .opsel_hi = 3, .neg = 1});
      auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
      ASSERT_NE(decoder, nullptr);
      std::unique_ptr<Instruction> instruction(decoder->decode(words.data()));
      ASSERT_NE(instruction, nullptr);

      Gfx1250Sim sim;
      auto *cu = sim.cu();
      auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
      ASSERT_NE(wf, nullptr);
      wf->set_exec(kExec);
      const uint32_t vgpr_base = wf->vgpr_alloc().base;
      for (uint32_t lane = 0; lane < 32; ++lane) {
        const float lane_value = static_cast<float>(lane + 1);
        cu->write_vgpr(vgpr_base, lane, std::bit_cast<uint32_t>(lane_value * 0.25f));
        cu->write_vgpr(vgpr_base + 1, lane, std::bit_cast<uint32_t>(lane_value * -0.5f));
        cu->write_vgpr(vgpr_base + 2, lane, std::bit_cast<uint32_t>(lane_value + 0.75f));
        cu->write_vgpr(vgpr_base + 3, lane, std::bit_cast<uint32_t>(lane_value * 1.5f));
        cu->write_vgpr(vgpr_base + 4, lane, kDstLoSeed);
        cu->write_vgpr(vgpr_base + 5, lane, kDstHiSeed);
      }

      if (!force_scalar) {
        bool accepted = false;
        if (test_case.opcode == cdna5::kVPkAddF32Vop3p) {
          auto *typed = dynamic_cast<cdna5::VPkAddF32Vop3p *>(instruction.get());
          ASSERT_NE(typed, nullptr);
          accepted = amdgpu::try_execute_vop3p_pk_binary_f32_simd(
              *typed, *wf, 0u, 3u, [](auto a, auto b) { return a + b; });
        } else {
          auto *typed = dynamic_cast<cdna5::VPkMulF32Vop3p *>(instruction.get());
          ASSERT_NE(typed, nullptr);
          accepted = amdgpu::try_execute_vop3p_pk_binary_f32_simd(
              *typed, *wf, 0u, 3u, [](auto a, auto b) { return a * b; });
        }
        EXPECT_TRUE(accepted);
        for (uint32_t lane = 0; lane < 32; ++lane) {
          cu->write_vgpr(vgpr_base + 4, lane, kDstLoSeed);
          cu->write_vgpr(vgpr_base + 5, lane, kDstHiSeed);
        }
      }

      cu->execute_instruction(instruction.get(), *wf);
      for (uint32_t lane = 0; lane < 32; ++lane) {
        result[lane * 2] = cu->read_vgpr(vgpr_base + 4, lane);
        result[lane * 2 + 1] = cu->read_vgpr(vgpr_base + 5, lane);
        if ((kExec & (1u << lane)) == 0u) {
          EXPECT_EQ(result[lane * 2], kDstLoSeed);
          EXPECT_EQ(result[lane * 2 + 1], kDstHiSeed);
        } else {
          const float lane_value = static_cast<float>(lane + 1);
          const float a_lo = lane_value * 0.25f;
          const float a_hi = lane_value * -0.5f;
          const float b_lo = lane_value + 0.75f;
          const float b_hi = lane_value * 1.5f;
          const float expected_lo =
              test_case.opcode == cdna5::kVPkAddF32Vop3p ? -a_lo + b_lo : -a_lo * b_lo;
          const float expected_hi =
              test_case.opcode == cdna5::kVPkAddF32Vop3p ? a_hi - b_hi : a_hi * -b_hi;
          EXPECT_EQ(result[lane * 2], std::bit_cast<uint32_t>(expected_lo));
          EXPECT_EQ(result[lane * 2 + 1], std::bit_cast<uint32_t>(expected_hi));
        }
      }
    };

    run_case(true, scalar_result);
    if constexpr (util::has_stdx_simd) {
      run_case(false, simd_result);
      EXPECT_EQ(simd_result, scalar_result);
    }
  }
}

TEST(Gfx1250ExecutionTest, PkF32EveryNondefaultSelectorGateFallsBackToScalar) {
  ForceScalarGuard force_scalar_guard;
  struct TestCase {
    const char *name;
    bool ternary;
    uint32_t op_sel;
    uint32_t op_sel_hi;
    uint32_t op_sel_hi_2;
  };
  constexpr std::array test_cases{
      TestCase{"binary-opsel", false, 1u, 3u, 0u},
      TestCase{"binary-opsel-hi", false, 0u, 2u, 0u},
      TestCase{"ternary-opsel", true, 1u, 3u, 1u},
      TestCase{"ternary-opsel-hi", true, 0u, 2u, 1u},
      TestCase{"ternary-opsel-hi-2", true, 0u, 3u, 0u},
  };
  constexpr uint32_t kExec = 0x5a5a5a5au;
  constexpr uint32_t kDstLoSeed = 0xdeadbeefu;
  constexpr uint32_t kDstHiSeed = 0xbaadf00du;

  for (const TestCase &test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    std::array<uint32_t, 64> scalar_result{};
    std::array<uint32_t, 64> fallback_result{};
    const auto run_case = [&](bool force_scalar, std::array<uint32_t, 64> &result) {
      util::set_force_scalar_for_testing(force_scalar);
      cdna5::Vop3pBuilderFields fields;
      fields.vdst = 6;
      fields.opsel = static_cast<uint8_t>(test_case.op_sel);
      fields.src0 = 256;
      fields.src1 = 258;
      fields.src2 = 260;
      fields.opsel_hi = static_cast<uint8_t>(test_case.op_sel_hi);
      auto words = cdna5::build_vop3p(
          test_case.ternary ? cdna5::kVPkFmaF32Vop3p : cdna5::kVPkAddF32Vop3p, fields);
      if (test_case.op_sel_hi_2 != 0u)
        words[0] |= uint32_t{1} << 14;
      auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
      ASSERT_NE(decoder, nullptr);
      std::unique_ptr<Instruction> instruction(decoder->decode(words.data()));
      ASSERT_NE(instruction, nullptr);

      Gfx1250Sim sim;
      auto *cu = sim.cu();
      auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
      ASSERT_NE(wf, nullptr);
      wf->set_exec(kExec);
      const uint32_t vgpr_base = wf->vgpr_alloc().base;
      for (uint32_t lane = 0; lane < 32; ++lane) {
        const float value = static_cast<float>(lane + 1);
        cu->write_vgpr(vgpr_base, lane, std::bit_cast<uint32_t>(value));
        cu->write_vgpr(vgpr_base + 1, lane, std::bit_cast<uint32_t>(value + 0.5f));
        cu->write_vgpr(vgpr_base + 2, lane, std::bit_cast<uint32_t>(value * 2.0f));
        cu->write_vgpr(vgpr_base + 3, lane, std::bit_cast<uint32_t>(value * -3.0f));
        cu->write_vgpr(vgpr_base + 4, lane, std::bit_cast<uint32_t>(value + 4.0f));
        cu->write_vgpr(vgpr_base + 5, lane, std::bit_cast<uint32_t>(value * 0.25f));
        cu->write_vgpr(vgpr_base + 6, lane, kDstLoSeed);
        cu->write_vgpr(vgpr_base + 7, lane, kDstHiSeed);
      }

      if (!force_scalar) {
        if (test_case.ternary) {
          auto *typed = dynamic_cast<cdna5::VPkFmaF32Vop3p *>(instruction.get());
          ASSERT_NE(typed, nullptr);
          EXPECT_FALSE(amdgpu::try_execute_vop3p_pk_ternary_f32_simd(
              *typed, *wf, test_case.op_sel, test_case.op_sel_hi, test_case.op_sel_hi_2,
              [](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }));
        } else {
          auto *typed = dynamic_cast<cdna5::VPkAddF32Vop3p *>(instruction.get());
          ASSERT_NE(typed, nullptr);
          EXPECT_FALSE(amdgpu::try_execute_vop3p_pk_binary_f32_simd(
              *typed, *wf, test_case.op_sel, test_case.op_sel_hi,
              [](auto a, auto b) { return a + b; }));
        }
      }
      cu->execute_instruction(instruction.get(), *wf);
      for (uint32_t lane = 0; lane < 32; ++lane) {
        result[lane * 2] = cu->read_vgpr(vgpr_base + 6, lane);
        result[lane * 2 + 1] = cu->read_vgpr(vgpr_base + 7, lane);
        if ((kExec & (1u << lane)) == 0u) {
          EXPECT_EQ(result[lane * 2], kDstLoSeed);
          EXPECT_EQ(result[lane * 2 + 1], kDstHiSeed);
        }
      }
    };

    run_case(true, scalar_result);
    if constexpr (util::has_stdx_simd) {
      run_case(false, fallback_result);
      EXPECT_EQ(fallback_result, scalar_result);
    }
  }
}

TEST(Gfx1250ExecutionTest, PkFmaF32SimdMatchesScalarWithPartialExec) {
  ForceScalarGuard force_scalar_guard;
  constexpr uint32_t kExec = 0xc3c3c3c3u;
  constexpr uint32_t kDstLoSeed = 0xdeadbeefu;
  constexpr uint32_t kDstHiSeed = 0xbaadf00du;
  std::array<uint32_t, 64> scalar_result{};
  std::array<uint32_t, 64> simd_result{};

  const auto run_case = [&](bool force_scalar, std::array<uint32_t, 64> &result) {
    util::set_force_scalar_for_testing(force_scalar);
    auto words = cdna5::build_vop3p(
        cdna5::kVPkFmaF32Vop3p,
        {.vdst = 6, .neg_hi = 4, .src0 = 256, .src1 = 258, .src2 = 260, .opsel_hi = 3, .neg = 2});
    words[0] |= uint32_t{1} << 14; // pad_14 is the src2 high-half selector.
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
    ASSERT_NE(decoder, nullptr);
    std::unique_ptr<Instruction> instruction(decoder->decode(words.data()));
    ASSERT_NE(instruction, nullptr);
    auto *typed = dynamic_cast<cdna5::VPkFmaF32Vop3p *>(instruction.get());
    ASSERT_NE(typed, nullptr);

    Gfx1250Sim sim;
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(kExec);
    const uint32_t vgpr_base = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < 32; ++lane) {
      const float value = static_cast<float>(lane + 1);
      cu->write_vgpr(vgpr_base, lane, std::bit_cast<uint32_t>(value * 0.25f));
      cu->write_vgpr(vgpr_base + 1, lane, std::bit_cast<uint32_t>(value * -0.5f));
      cu->write_vgpr(vgpr_base + 2, lane, std::bit_cast<uint32_t>(value + 0.75f));
      cu->write_vgpr(vgpr_base + 3, lane, std::bit_cast<uint32_t>(value * 1.5f));
      cu->write_vgpr(vgpr_base + 4, lane, std::bit_cast<uint32_t>(value * -2.0f));
      cu->write_vgpr(vgpr_base + 5, lane, std::bit_cast<uint32_t>(value + 3.0f));
      cu->write_vgpr(vgpr_base + 6, lane, kDstLoSeed);
      cu->write_vgpr(vgpr_base + 7, lane, kDstHiSeed);
    }

    if (!force_scalar) {
      EXPECT_TRUE(amdgpu::try_execute_vop3p_pk_ternary_f32_simd(
          *typed, *wf, 0u, 3u, 1u,
          [](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }));
      for (uint32_t lane = 0; lane < 32; ++lane) {
        cu->write_vgpr(vgpr_base + 6, lane, kDstLoSeed);
        cu->write_vgpr(vgpr_base + 7, lane, kDstHiSeed);
      }
    }

    cu->execute_instruction(instruction.get(), *wf);
    for (uint32_t lane = 0; lane < 32; ++lane) {
      result[lane * 2] = cu->read_vgpr(vgpr_base + 6, lane);
      result[lane * 2 + 1] = cu->read_vgpr(vgpr_base + 7, lane);
      if ((kExec & (1u << lane)) == 0u) {
        EXPECT_EQ(result[lane * 2], kDstLoSeed);
        EXPECT_EQ(result[lane * 2 + 1], kDstHiSeed);
      } else {
        const float value = static_cast<float>(lane + 1);
        const float expected_lo = std::fma(value * 0.25f, -(value + 0.75f), value * -2.0f);
        const float expected_hi = std::fma(value * -0.5f, value * 1.5f, -(value + 3.0f));
        EXPECT_EQ(result[lane * 2], std::bit_cast<uint32_t>(expected_lo));
        EXPECT_EQ(result[lane * 2 + 1], std::bit_cast<uint32_t>(expected_hi));
      }
    }
  };

  run_case(true, scalar_result);
  if constexpr (util::has_stdx_simd) {
    run_case(false, simd_result);
    EXPECT_EQ(simd_result, scalar_result);
  }
}

TEST(Gfx1250DecodeTest, Vop3pRejectsLiteral64SelectorInEverySourcePosition) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);

  for (const cdna5::Vop3pBuilderFields fields : {
           cdna5::Vop3pBuilderFields{.src0 = 254},
           cdna5::Vop3pBuilderFields{.src1 = 254},
           cdna5::Vop3pBuilderFields{.src2 = 254},
       }) {
    const auto words = cdna5::build_vop3p(cdna5::kVPkFmaF32Vop3p, fields);
    EXPECT_THROW(static_cast<void>(decoder->decode(words.data())), util::InvalidInst);
  }
}

TEST(Gfx1250DecodeTest, BinaryVop3pIgnoresLiteral64SelectorInUnusedSrc2) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);

  for (const uint32_t opcode : {cdna5::kVPkAddF32Vop3p, cdna5::kVPkMulF32Vop3p}) {
    const auto words = cdna5::build_vop3p(opcode, {.src0 = 128, .src1 = 129, .src2 = 254});
    std::unique_ptr<Instruction> instruction(decoder->decode(words.data()));
    ASSERT_NE(instruction, nullptr);
    EXPECT_EQ(instruction->size(), 8);
    EXPECT_EQ(instruction->num_src_operands(), 2);
  }
}

TEST(Gfx1250ExecutionTest, VCmpGtU32Wave32ExplicitSdstPreservesHighSgpr) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x3u);

  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 4, 0, 3u);
  cu->write_vgpr(vgpr_base + 4, 1, 5u);
  write_wave_sgpr(*cu, *wf, 2, 0xaaaaaaaau);
  write_wave_sgpr(*cu, *wf, 3, 0xfefefefeu);
  wf->set_vcc(0x12345678u);

  const std::array<uint32_t, 2> words = {
      0xD44C0002u, // v_cmp_gt_u32_e64 s2, 4, v4
      0x02020884u,
  };
  cdna5::VCmpGtU32Vop3 cmp(words.data());
  cmp.execute_impl(*wf);

  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 2), 0x1u);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 3), 0xfefefefeu);
  EXPECT_EQ(wf->vcc(), 0x12345678u);
}

TEST(Gfx1250ExecutionTest, Wave32ScalarVccHiWritePreservesUpperHalf) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec(0xffff0000u);
  wf->set_vcc(0);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);

  const uint32_t words[] = {0x8c6b7e6bu, 0}; // s_or_b32 vcc_hi, vcc_hi, exec_lo
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "s_or_b32");
  cu->execute_instruction(inst.get(), *wf);

  EXPECT_EQ(wf->vcc(), 0xffff0000'00000000ULL);
}

} // namespace
