// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file simd_coverage_test.cpp
/// @brief Exact SIMD coverage for packed, true16, compare and dual-slot paths.

#include "decode_test_util.h"
#include "util/simd_test_hooks.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <memory>
#include <random>
#include <string_view>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;
using O = cdna5::Operand;
using K = cdna5::OperandType;
using U = util::native<uint32_t>;

class SimdCoverage : public ::testing::Test {
protected:
  const IsaExecutionBackend backend{.operand_backend = O::full_execution_backend()};
  ScopedIsaExecutionBackend backend_scope{&backend};
  GpuMemory memory{"simd_coverage_memory"};
  L2Cache l2{"simd_coverage_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  Wavefront *wf = nullptr;
  bool old_force_scalar = util::force_scalar();
  void SetUp() override {
    util::set_force_scalar_for_testing(false);
    ComputeUnitCore::Config config{};
    config.arch = ROCJITSU_CODE_ARCH_CDNA5;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    cu = ComputeUnitCore::create("simd_coverage_cu", config, &memory, &l2);
    wf = cu->dispatch_wf(0, 0, 106, 256, 32);
    wf->set_mode_raw(3u << 6);
    wf->set_exec(0xffffffffu);
  }
  void TearDown() override { util::set_force_scalar_for_testing(old_force_scalar); }
  void put(uint32_t reg, uint32_t lane, uint32_t value) {
    cu->write_vgpr(wf->vgpr_alloc().base + reg, lane, value);
  }
  uint32_t get(uint32_t reg, uint32_t lane) {
    return cu->read_vgpr(wf->vgpr_alloc().base + reg, lane);
  }
};

struct WordInst {
  struct Encoding {
    uint32_t src0 = 256, src1 = 257, src2 = 258, vdst = 3;
    uint32_t opsel = 0, opsel_hi = 3, opsel_hi_2 = 1;
    uint32_t neg = 0, neg_hi = 0, abs = 0, clamp = 0, omod = 0;
  } inst_;
  O src0{32, K::OPR_VGPR, 0}, src1{32, K::OPR_VGPR, 1}, src2{32, K::OPR_VGPR, 2};
  O vdst{32, K::OPR_VGPR, 3};
};

TEST_F(SimdCoverage, AllBooleanTruthTables) {
  std::mt19937 rng(731);
  for (unsigned table = 0; table < 256; ++table) {
    U a([&](auto) { return uint32_t(rng()); }), b([&](auto) { return uint32_t(rng()); }),
        c([&](auto) { return uint32_t(rng()); });
    U actual = bitop3_words(a, b, c, table);
    for (unsigned lane = 0; lane < U::size(); ++lane) {
      uint32_t expected = 0;
      for (unsigned bit = 0; bit < 32; ++bit) {
        unsigned index = (((a[lane] >> bit) & 1u) << 2) | (((b[lane] >> bit) & 1u) << 1) |
                         ((c[lane] >> bit) & 1u);
        expected |= ((table >> index) & 1u) << bit;
      }
      ASSERT_EQ(actual[lane], expected) << table;
      ASSERT_EQ(bitop3_words(uint32_t(a[lane]), uint32_t(b[lane]), uint32_t(c[lane]), table),
                expected);
    }
  }
}

template <typename T> void check_div_scale_vectors() {
  using Bits = typename DivisionFormat<T>::Bits;
  using V = util::native<T>;
  using B = util::native<Bits>;
  std::mt19937_64 rng(381);
  for (uint32_t mode = 0; mode < 16; ++mode) {
    for (unsigned iteration = 0; iteration < 4000; ++iteration) {
      B db([&](auto) { return Bits(rng()); }), nb([&](auto) { return Bits(rng()); });
      if (iteration < 16) {
        constexpr Bits corners[] = {0,
                                    DivisionFormat<T>::sign,
                                    1,
                                    DivisionFormat<T>::infinity,
                                    DivisionFormat<T>::infinity | 1,
                                    DivisionFormat<T>::sign | DivisionFormat<T>::infinity,
                                    DivisionFormat<T>::fraction_mask};
        db = B(corners[iteration % 7]);
        nb = B(corners[(iteration / 2) % 7]);
      }
      V d = std::bit_cast<V>(db), n = std::bit_cast<V>(nb), value = iteration & 1 ? d : n;
      auto [actual, mask] = div_scale_simd<T>(value, d, n, mode & 3, mode >> 2);
      for (unsigned i = 0; i < V::size(); ++i) {
        auto expected = div_scale<T>(value[i], d[i], n[i], mode & 3, mode >> 2);
        ASSERT_EQ(std::bit_cast<Bits>(T(actual[i])), std::bit_cast<Bits>(expected.value))
            << mode << ":" << iteration;
        ASSERT_EQ((mask >> i) & 1u, expected.post_scale) << mode << ":" << iteration;
      }
    }
  }
}
TEST_F(SimdCoverage, DivisionScaleF32EveryMode) { check_div_scale_vectors<float>(); }
TEST_F(SimdCoverage, DivisionScaleF64EveryMode) { check_div_scale_vectors<double>(); }

TEST_F(SimdCoverage, WordPathsPreserveHalvesMasksAndAliasing) {
  WordInst inst;
  inst.inst_.opsel = 1 | 8;
  inst.vdst = O(32, K::OPR_VGPR, 0);
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    put(0, lane, 0x12345678u + lane);
    put(1, lane, 0xfff0u);
  }
  wf->set_exec(0x55555555u);
  ASSERT_TRUE(
      (try_execute_words_simd<2, false, true, 3>(inst, *wf, [](auto a, auto b) { return a & b; })));
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
    ASSERT_EQ(get(0, lane), lane % 2 ? 0x12345678u + lane : 0x12305678u + lane);
  inst.src0 = O(16, K::OPR_VGPR, 128, true);
  inst.vdst = O(16, K::OPR_VGPR, 129, false, true);
  inst.inst_.vdst = 129;
  ASSERT_TRUE((try_execute_words_simd<1, true, true, 1>(inst, *wf, [](auto a) { return a; })));
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
    ASSERT_EQ(get(1, lane), lane % 2 ? 0xfff0u : 0x1230fff0u);
  wf->set_exec(0);
  ASSERT_TRUE((try_execute_words_simd<1, true, true, 1>(inst, *wf, [](auto a) { return ~a; })));
  ASSERT_EQ(get(1, 0), 0x1230fff0u);
}

TEST_F(SimdCoverage, DecodedTrue16MovePreservesScalarAndVectorSelectors) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  constexpr uint64_t exec = 0x55555555u;
  wf->set_exec(exec);
  cu->write_sgpr(wf->sgpr_alloc().base + 9, 0xc0003c00u);
  // Scalar register, integer/floating inline constants, literal and both VGPR halves.
  for (uint32_t source : {9u, 129u, 242u, 255u, 256u, 384u}) {
    for (bool high : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "source=" << source << " high=" << high);
      const uint32_t words[] = {0x7e003800u | ((7u | (high ? 128u : 0u)) << 17) | source,
                                0x76544321u};
      std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words));
      ASSERT_NE(instruction, nullptr);
      std::array<uint32_t, 32> expected;
      for (bool scalar : {true, false}) {
        util::set_force_scalar_for_testing(scalar);
        for (uint32_t lane = 0; lane < 32; ++lane) {
          put(0, lane, 0xc0003c00u + lane);
          put(7, lane, 0xabcd1234u);
          put(9, lane, 0xdeadbeefu); // Detect accidental scalar-to-vector reinterpretation.
        }
        ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wf).succeeded());
        for (uint32_t lane = 0; lane < 32; ++lane) {
          if (scalar)
            expected[lane] = get(7, lane);
          else
            EXPECT_EQ(get(7, lane), expected[lane]) << "lane=" << lane;
          if (!(exec & (uint64_t{1} << lane))) {
            EXPECT_EQ(get(7, lane), 0xabcd1234u);
          } else if (source == 9) {
            EXPECT_EQ(get(7, lane), high ? 0x3c001234u : 0xabcd3c00u);
          }
        }
      }
    }
  }
}

template <PackedFloatOp Op, bool Bf16> void check_packed_half(Wavefront &wf) {
  SCOPED_TRACE(int(Op));
  SCOPED_TRACE(Bf16);
  using V = util::narrow32<uint32_t>;
  std::mt19937 rng(871);
  constexpr uint16_t corners[] = {0,      0x8000, 1,      0x83ff, 0x3c00, 0xbc00, 0x7bff, 0x7c00,
                                  0xfc00, 0x7e01, 0x7c01, 0x7f80, 0xff80, 0x3f80, 0xbf80, 0x7fc1};
  for (uint32_t mode = 0; mode < 32; ++mode) {
    wf.set_mode_raw(((mode & 3) << 2) | (((mode >> 2) & 3) << 6) |
                    ((mode & 16) ? Wavefront::FP16_OVFL_BIT : 0));
    for (unsigned iteration = 0; iteration < 400; ++iteration) {
      auto generate = [&]() {
        return iteration < 64 ? uint32_t(corners[rng() % std::size(corners)])
                              : (uint32_t(rng()) & 0xffffu);
      };
      V a([&](auto) { return generate(); }), b([&](auto) { return generate(); }),
          c([&](auto) { return generate(); });
      fp_mode::ScopedEnvironment environment(0);
      V actual = packed_float_half_simd<Op, Bf16>(a, b, c, wf, iteration & 1);
      for (unsigned i = 0; i < V::size(); ++i) {
        uint16_t expected;
        if constexpr (Bf16) {
          float af = util::bf16_to_f32(a[i]), bf = util::bf16_to_f32(b[i]),
                cf = util::bf16_to_f32(c[i]);
          if constexpr (Op == PackedFloatOp::ADD)
            expected = fp_mode::packed_add_bf16(af, bf, wf.fp16_ovfl());
          else if constexpr (Op == PackedFloatOp::MUL)
            expected = fp_mode::packed_mul_bf16(af, bf, wf.fp16_ovfl());
          else if constexpr (Op == PackedFloatOp::FMA)
            expected = fp_mode::packed_fma_bf16(af, bf, cf, wf.fp16_ovfl());
          else
            expected = fp_mode::packed_select_bf16(af, bf, Op == PackedFloatOp::MIN);
          expected = fp_mode::clamp_bf16(expected, iteration & 1, floating_clamp_nan_to_zero(wf));
        } else if constexpr (Op == PackedFloatOp::FMA) {
          expected =
              fp_mode::fma_f16(a[i], b[i], c[i], false, false, false, false, false, false,
                               wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64(), 0,
                               iteration & 1, wf.fp16_ovfl(), floating_clamp_nan_to_zero(wf));
        } else {
          constexpr auto operation = Op == PackedFloatOp::ADD   ? fp_mode::PackedBinaryOp::ADD
                                     : Op == PackedFloatOp::MUL ? fp_mode::PackedBinaryOp::MUL
                                     : Op == PackedFloatOp::MIN ? fp_mode::PackedBinaryOp::MIN
                                     : Op == PackedFloatOp::MAX ? fp_mode::PackedBinaryOp::MAX
                                     : Op == PackedFloatOp::MINIMUM
                                         ? fp_mode::PackedBinaryOp::MINIMUM
                                         : fp_mode::PackedBinaryOp::MAXIMUM;
          expected = fp_mode::packed_binary_f16(operation, a[i], b[i], wf.fp_round_mode_f16_f64(),
                                                wf.fp_denorm_mode_f16_f64(), iteration & 1,
                                                wf.fp16_ovfl(), floating_clamp_nan_to_zero(wf));
        }
        ASSERT_EQ(actual[i], expected)
            << mode << ":" << iteration << " inputs=" << a[i] << "," << b[i] << "," << c[i];
      }
    }
  }
}
TEST_F(SimdCoverage, PackedF16Numerics) {
  check_packed_half<PackedFloatOp::ADD, false>(*wf);
  check_packed_half<PackedFloatOp::MUL, false>(*wf);
  check_packed_half<PackedFloatOp::FMA, false>(*wf);
  check_packed_half<PackedFloatOp::MIN, false>(*wf);
  check_packed_half<PackedFloatOp::MAX, false>(*wf);
  check_packed_half<PackedFloatOp::MINIMUM, false>(*wf);
  check_packed_half<PackedFloatOp::MAXIMUM, false>(*wf);
}
TEST_F(SimdCoverage, PackedBf16Numerics) {
  check_packed_half<PackedFloatOp::ADD, true>(*wf);
  check_packed_half<PackedFloatOp::MUL, true>(*wf);
  check_packed_half<PackedFloatOp::FMA, true>(*wf);
  check_packed_half<PackedFloatOp::MIN, true>(*wf);
  check_packed_half<PackedFloatOp::MAX, true>(*wf);
}

TEST_F(SimdCoverage, PackedBroadcastAndInlineConstantEnterSimd) {
  WordInst inst;
  inst.inst_.opsel_hi = 1;
  inst.src1 = O(32, K::OPR_SRC, 242); // inline 1.0
  inst.inst_.src1 = 242;
  wf->set_mode_raw(3u << 6);
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
    put(0, lane, 0x40003c00u);
  ASSERT_TRUE((try_execute_packed_float_simd<PackedFloatOp::ADD, false>(inst, *wf)));
  ASSERT_EQ(get(3, 0), 0x42004000u); // {1+1, 2+1}
  util::set_force_scalar_for_testing(true);
  ASSERT_FALSE((try_execute_packed_float_simd<PackedFloatOp::ADD, false>(inst, *wf)));
}

struct Slot {
  uint16_t op;
  O *dst, *src0, *src1, *src2;
  uint32_t src2_imm = 0;
  uint8_t neg = 0;
  bool has_src2_operand = false, uses_vcc = false;
};
TEST_F(SimdCoverage, VopdStagesBothSlotsBeforeAliasingWrites) {
  O a(32, K::OPR_VGPR, 0), b(32, K::OPR_VGPR, 1), c(32, K::OPR_VGPR, 2);
  Slot x{16, &a, &b, &c, nullptr}, y{16, &b, &a, &c, nullptr};
  wf->set_exec(0x55555555u);
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    put(0, lane, 3);
    put(1, lane, 7);
    put(2, lane, 11);
  }
  ASSERT_TRUE((try_execute_vopd_simd<true>(x, y, *wf)));
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    ASSERT_EQ(get(0, lane), lane % 2 ? 3u : 18u);
    ASSERT_EQ(get(1, lane), lane % 2 ? 7u : 14u);
  }
}

class VopdReadRecorder final : public ExecutionPlugin {
public:
  VopdReadRecorder() : ExecutionPlugin("vopd_read_recorder") {}

  void onAmdgpuReadVgprLanes(const Wavefront *wave, uint32_t physical_reg, uint64_t lane_mask,
                             uint8_t byte_mask) override {
    const uint32_t reg = physical_reg - wave->vgpr_alloc().base;
    ASSERT_LT(reg, lanes.size());
    lanes[reg] |= lane_mask;
    bytes[reg] |= byte_mask;
  }

  std::array<uint64_t, 8> lanes{};
  std::array<uint8_t, 8> bytes{};
};

TEST_F(SimdCoverage, DecodedVopdMovAndFloatPairReadsOnlyUsedRegisters) {
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto plugin = std::make_unique<VopdReadRecorder>();
  auto *recorder = plugin.get();
  group->add(std::move(plugin));
  cu->set_plugin_group(group);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  for (uint32_t mov_source : {0u, 4u}) {
    // MOV v6,v0 has unused VSRCX1=5; MOV v6,v4 has unused VSRCX1=0.
    // Both pair with MUL v7,v2,v3, selecting the floating-point VOPD helper.
    const uint32_t words[] = {mov_source == 0 ? 0xca060b00u : 0xca060104u, 0x06060702u};
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words));
    ASSERT_NE(instruction, nullptr);
    for (uint64_t exec : {uint64_t{0}, uint64_t{1}, uint64_t{0x55555555}}) {
      for (bool force_scalar : {true, false}) {
        SCOPED_TRACE(::testing::Message()
                     << "source=" << mov_source << " exec=" << exec << " scalar=" << force_scalar);
        for (uint32_t reg = 0; reg < 8; ++reg)
          for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
            put(reg, lane, 0x3f800000u);
        wf->set_exec(exec);
        recorder->lanes = {};
        recorder->bytes = {};
        util::set_force_scalar_for_testing(force_scalar);
        ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wf).succeeded());
        for (uint32_t reg = 0; reg < 8; ++reg) {
          const bool used = reg == mov_source || reg == 2 || reg == 3;
          EXPECT_EQ(recorder->lanes[reg], used ? exec : 0) << "v" << reg;
          EXPECT_EQ(recorder->bytes[reg], used && exec != 0 ? 0xf : 0) << "v" << reg;
        }
      }
    }
  }
}

TEST_F(SimdCoverage, DecodedPackedFmacPreservesDirectedZeroSigns) {
  struct ZeroCase {
    uint32_t multiplicand, multiplier, addend;
    bool matching_negative_zero, cancellation;
  };
  constexpr ZeroCase cases[] = {
      {0x3c003c00u, 0x3c003c00u, 0xbc00bc00u, false, true},
      {0xbc00bc00u, 0x3c003c00u, 0x3c003c00u, false, true},
      {0x80008000u, 0x3c003c00u, 0x80008000u, true, false},
      {0x00000000u, 0x3c003c00u, 0x00000000u, false, false},
      {0x00000000u, 0x3c003c00u, 0x80008000u, false, true},
      {0x80008000u, 0x3c003c00u, 0x00000000u, false, true},
  };
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA4}) {
    auto decoder = Decoder::create(arch);
    const uint32_t words[] = {0x780c0300u}; // v_pk_fmac_f16 v6, v0, v1
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words));
    ASSERT_NE(instruction, nullptr);
    ComputeUnitCore::Config config{};
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    auto target = ComputeUnitCore::create("packed_fmac_zero", config, &memory, &l2);
    auto *wave = target->dispatch_wf(0, 0, 106, 256, 32);
    ASSERT_NE(wave, nullptr);
    for (uint32_t rounding = 0; rounding < 4; ++rounding) {
      wave->set_mode_raw(rounding << 2);
      for (const auto &test : cases) {
        for (bool force_scalar : {true, false}) {
          SCOPED_TRACE(::testing::Message()
                       << "arch=" << arch << " rounding=" << rounding << " scalar=" << force_scalar
                       << " addend=" << test.addend);
          for (uint32_t lane = 0; lane < wave->wf_size(); ++lane) {
            target->write_vgpr(wave->vgpr_alloc().base, lane, test.multiplicand);
            target->write_vgpr(wave->vgpr_alloc().base + 1, lane, test.multiplier);
            target->write_vgpr(wave->vgpr_alloc().base + 6, lane, test.addend);
          }
          wave->set_exec(0x55555555u);
          util::set_force_scalar_for_testing(force_scalar);
          ASSERT_TRUE(target->execute_instruction(instruction.get(), *wave).succeeded());
          const bool negative = test.matching_negative_zero || (test.cancellation && rounding == 2);
          for (uint32_t lane = 0; lane < wave->wf_size(); ++lane)
            EXPECT_EQ(target->read_vgpr(wave->vgpr_alloc().base + 6, lane), lane % 2   ? test.addend
                                                                            : negative ? 0x80008000u
                                                                                       : 0u);
        }
      }
    }
  }
}

struct DecodedCase {
  rj_code_arch_t arch;
  const char *assembly;
  uint32_t words[4];
};
// Encodings assembled with the local gfx1250-capable LLVM assembler.
const DecodedCase kDecodedCases[] = {
    {ROCJITSU_CODE_ARCH_CDNA4, "v_cmpx_lt_f32_e64 s[8:9], v0, v1", {0xd0510008u, 0x00020300u}},
    {ROCJITSU_CODE_ARCH_RDNA4, "v_div_scale_f32 v6, s8, v0, v1, v2", {0xd6fc0806u, 0x040a0300u}},
    {ROCJITSU_CODE_ARCH_CDNA5,
     "v_pk_add_f16 v6, v0, 1.0 op_sel_hi:[1,0]",
     {0xcc0f4006u, 0x0a01e500u}},
    {ROCJITSU_CODE_ARCH_CDNA5,
     "v_pk_mul_f16 v6, 0x3000, v1 op_sel_hi:[0,1]",
     {0xcc104006u, 0x120202ffu, 0x00003000u}},
    {ROCJITSU_CODE_ARCH_CDNA5, "v_pk_fma_f16 v6, v0, v1, v2", {0xcc0e4006u, 0x1c0a0300u}},
    {ROCJITSU_CODE_ARCH_CDNA5, "v_pk_min_f16 v6, v0, v1", {0xcc1b4006u, 0x1a020300u}},
    {ROCJITSU_CODE_ARCH_CDNA5, "v_cvt_pk_f16_f32 v6, v0, v1", {0xd76f0006u, 0x02020300u}},
    {ROCJITSU_CODE_ARCH_CDNA5, "v_cmpx_lt_f32 v0, v1", {0x7d220300u}},
    {ROCJITSU_CODE_ARCH_CDNA5, "v_cmpx_lt_f32_e64 v0, v1", {0xd491007eu, 0x02020300u}},
    {ROCJITSU_CODE_ARCH_CDNA5, "v_mov_b16 v6.h, v0.h", {0x7f0c3980u}},
    {ROCJITSU_CODE_ARCH_CDNA5,
     "v_mov_b16 v6.h, v0.h quad_perm:[1,0,3,2]",
     {0x7f0c38fau, 0xff00b180u}},
    {ROCJITSU_CODE_ARCH_CDNA5, "v_add_f16 v6.h, v0.h, v1.l", {0x650c0380u}},
    {ROCJITSU_CODE_ARCH_CDNA5, "v_and_b16 v6.h, v0.h, v1.l", {0xd7624806u, 0x02020300u}},
    {ROCJITSU_CODE_ARCH_CDNA5,
     "v_dual_add_f32 v6, v0, v1 :: v_dual_mul_f32 v7, v2, v3",
     {0xc9060300u, 0x06060702u}},
    {ROCJITSU_CODE_ARCH_CDNA5, "v_pk_fmac_f16 v6, v0, v1", {0x780c0300u}},
    {ROCJITSU_CODE_ARCH_CDNA5, "v_cvt_f16_f32 v6.h, v0", {0x7f0c1500u}},
    {ROCJITSU_CODE_ARCH_CDNA5, "v_cmpx_lt_f16_e64 v0.h, v1.l", {0xd481087eu, 0x02020300u}},
    {ROCJITSU_CODE_ARCH_CDNA5, "v_cvt_pk_bf16_f32 v6, v0, v1", {0xd76d0006u, 0x02020300u}},
    {ROCJITSU_CODE_ARCH_CDNA5,
     "v_bitop3_b32 v6, v0, v1, v2 bitop3:0x96",
     {0xd6340206u, 0xd40a0300u}},
    {ROCJITSU_CODE_ARCH_CDNA5, "v_pk_add_bf16 v6, v0, v1", {0xcc234006u, 0x1a020300u}},
    {ROCJITSU_CODE_ARCH_CDNA5, "v_pk_fma_bf16 v6, v0, v1, v2", {0xcc114006u, 0x1c0a0300u}},
    {ROCJITSU_CODE_ARCH_CDNA5, "v_cvt_f32_bf16 v6, v0.h", {0x7e0ce580u}},
    {ROCJITSU_CODE_ARCH_CDNA5, "v_div_scale_f32 v6, s8, v0, v1, v2", {0xd6fc0806u, 0x040a0300u}},
    {ROCJITSU_CODE_ARCH_CDNA4,
     "v_pk_add_f16 v6, v0, 1.0 op_sel_hi:[1,0]",
     {0xd38f4006u, 0x0801e500u}},
    {ROCJITSU_CODE_ARCH_CDNA4, "v_pk_fma_f16 v6, v0, v1, v2", {0xd38e4006u, 0x1c0a0300u}},
    {ROCJITSU_CODE_ARCH_CDNA4, "v_pk_min_f16 v6, v0, v1", {0xd3914006u, 0x18020300u}},
    {ROCJITSU_CODE_ARCH_CDNA4, "v_cvt_pk_f16_f32 v6, v0, v1", {0xd2670006u, 0x00020300u}},
    {ROCJITSU_CODE_ARCH_CDNA4, "v_cmpx_lt_f32 v0, v1", {0x7ca20300u}},
    {ROCJITSU_CODE_ARCH_CDNA4, "v_cvt_pk_bf16_f32 v6, v0, v1", {0xd2680006u, 0x00020300u}},
    {ROCJITSU_CODE_ARCH_CDNA4,
     "v_bitop3_b32 v6, v0, v1, v2 bitop3:0x96",
     {0xd2340206u, 0xd40a0300u}},
    {ROCJITSU_CODE_ARCH_CDNA4,
     "v_div_scale_f32 v6, s[8:9], v0, v1, v2",
     {0xd1e00806u, 0x040a0300u}},
    {ROCJITSU_CODE_ARCH_RDNA4,
     "v_pk_add_f16 v6, v0, 1.0 op_sel_hi:[1,0]",
     {0xcc0f4006u, 0x0a01e500u}},
    {ROCJITSU_CODE_ARCH_RDNA4,
     "v_pk_mul_f16 v6, 0x3000, v1 op_sel_hi:[0,1]",
     {0xcc104006u, 0x120202ffu, 0x00003000u}},
    {ROCJITSU_CODE_ARCH_RDNA4, "v_pk_fma_f16 v6, v0, v1, v2", {0xcc0e4006u, 0x1c0a0300u}},
    {ROCJITSU_CODE_ARCH_RDNA4, "v_pk_min_f16 v6, v0, v1", {0xcc1b4006u, 0x1a020300u}},
    {ROCJITSU_CODE_ARCH_RDNA4, "v_cmpx_lt_f32 v0, v1", {0x7d220300u}},
    {ROCJITSU_CODE_ARCH_RDNA4, "v_cmpx_lt_f32_e64 v0, v1", {0xd491007eu, 0x02020300u}},
    {ROCJITSU_CODE_ARCH_RDNA4, "v_mov_b16 v6.h, v0.h", {0x7f0c3980u}},
    {ROCJITSU_CODE_ARCH_RDNA4,
     "v_mov_b16 v6.h, v0.h quad_perm:[1,0,3,2]",
     {0x7f0c38fau, 0xff00b180u}},
    {ROCJITSU_CODE_ARCH_RDNA4, "v_add_f16 v6.h, v0.h, v1.l", {0x650c0380u}},
    {ROCJITSU_CODE_ARCH_RDNA4, "v_and_b16 v6.h, v0.h, v1.l", {0xd7624806u, 0x02020300u}},
    {ROCJITSU_CODE_ARCH_RDNA4,
     "v_dual_add_f32 v6, v0, v1 :: v_dual_mul_f32 v7, v2, v3",
     {0xc9060300u, 0x06060702u}},
    {ROCJITSU_CODE_ARCH_RDNA4, "v_pk_fmac_f16 v6, v0, v1", {0x780c0300u}},
    {ROCJITSU_CODE_ARCH_RDNA4, "v_cvt_f16_f32 v6.h, v0", {0x7f0c1500u}},
    {ROCJITSU_CODE_ARCH_RDNA4, "v_cmpx_lt_f16_e64 v0.h, v1.l", {0xd481087eu, 0x02020300u}},
};

TEST_F(SimdCoverage, DecodedTargetPathsMatchScalar) {
  constexpr uint32_t input[] = {0x3c004000u, 0xbc004200u, 0x80000000u, 0x00008000u, 0x3f800000u,
                                0x40000000u, 0xc0400000u, 0x04000400u, 0x7bfffbffu, 0x3f80bf80u};
  for (const auto &test : kDecodedCases) {
    SCOPED_TRACE(test.assembly);
    SCOPED_TRACE(test.arch);
    auto decoder = Decoder::create(test.arch);
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder, test.words));
    ASSERT_NE(instruction, nullptr);
    for (uint64_t exec : {uint64_t{0}, uint64_t{0x55555555}, uint64_t{0xffffffff}}) {
      std::vector<uint32_t> reference;
      for (bool force_scalar : {true, false}) {
        ComputeUnitCore::Config config{};
        config.arch = test.arch;
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 106;
        config.vgprs_per_wf = 256;
        config.lds_size_kb = 64;
        auto target = ComputeUnitCore::create("simd_decoded_target", config, &memory, &l2);
        const uint32_t wave_size = test.arch == ROCJITSU_CODE_ARCH_CDNA4 ? 64 : 32;
        auto *wave = target->dispatch_wf(0, 0, 106, 256, wave_size);
        ASSERT_NE(wave, nullptr);
        wave->set_mode_raw(3u << 6);
        wave->set_exec(exec);
        wave->set_vcc_raw(0x1234567887654321ULL);
        for (uint32_t reg = 0; reg < 10; ++reg)
          for (uint32_t lane = 0; lane < wave_size; ++lane)
            target->write_vgpr(wave->vgpr_alloc().base + reg, lane,
                               input[(reg + lane) % std::size(input)]);
        util::set_force_scalar_for_testing(force_scalar);
        ASSERT_TRUE(target->execute_instruction(instruction.get(), *wave).succeeded());
        std::vector<uint32_t> output;
        for (uint32_t reg = 0; reg < 10; ++reg)
          for (uint32_t lane = 0; lane < wave_size; ++lane)
            output.push_back(target->read_vgpr(wave->vgpr_alloc().base + reg, lane));
        output.push_back(uint32_t(wave->exec()));
        output.push_back(uint32_t(wave->exec() >> 32));
        output.push_back(uint32_t(wave->vcc()));
        output.push_back(uint32_t(wave->vcc() >> 32));
        output.push_back(target->read_sgpr(wave->sgpr_alloc().base + 8));
        output.push_back(target->read_sgpr(wave->sgpr_alloc().base + 9));
        if (force_scalar)
          reference = output;
        else {
          ASSERT_EQ(output.size(), reference.size());
          for (size_t i = 0; i < output.size(); ++i)
            ASSERT_EQ(output[i], reference[i]) << "output index=" << i << " exec=" << exec;
        }
      }
    }
  }
}

TEST_F(SimdCoverage, CompareWriterUsesOldExecAndPreservesVcc) {
  struct CompareInst {
    WordInst::Encoding inst_;
    O src0{32, K::OPR_VGPR, 0}, vsrc1{32, K::OPR_VGPR, 1};
  } inst;
  for (uint32_t lane = 0; lane < 32; ++lane) {
    put(0, lane, lane);
    put(1, lane, 16);
  }
  wf->set_exec(0x55555555u);
  wf->set_vcc_raw(0x1234567887654321ULL);
  ASSERT_TRUE((try_execute_vopc_simd<uint32_t>(
      inst, *wf, [](auto a, auto b) { return a < b; },
      [&](uint64_t result) { wf->set_exec(result); })));
  EXPECT_EQ(wf->exec(), 0x5555u);
  EXPECT_EQ(wf->vcc(), 0x1234567887654321ULL);
}

TEST_F(SimdCoverage, DivisionScalePathReturnsAndPreservesInactiveMaskBits) {
  WordInst inst;
  for (uint32_t lane = 0; lane < 32; ++lane) {
    put(0, lane, 0x3f800000u);
    put(1, lane, 0x3f800000u);
    put(2, lane, 0x40000000u);
    put(3, lane, 0xdeadbeefu);
  }
  wf->set_exec(0x55555555u);
  wf->set_vcc_raw(0xabcdef01ffffffffULL);
  uint64_t result = 0;
  ASSERT_TRUE((try_execute_div_scale_simd<float>(inst, *wf, [&](uint64_t v) { result = v; })));
  EXPECT_EQ(result, 0xabcdef01aaaaaaaaULL);
  for (uint32_t lane = 0; lane < 32; ++lane)
    EXPECT_EQ(get(3, lane), lane % 2 ? 0xdeadbeefu : 0x3f800000u);
}

TEST_F(SimdCoverage, MixedFmaUsesProfileSemanticsAndCompilerGuard) {
  WordInst inst;
  inst.inst_.opsel_hi = 0;
  inst.inst_.opsel_hi_2 = 0;
  const float a = std::bit_cast<float>(0x3f800001u);
  const float b = std::bit_cast<float>(0x3f7fffffu);
  for (uint32_t lane = 0; lane < 32; ++lane) {
    put(0, lane, 0x3f800001u);
    put(1, lane, 0x3f7fffffu);
    put(2, lane, 0xbf800000u);
  }
#if defined(__clang__) && defined(__FMA__)
  ASSERT_FALSE((try_execute_vop3p_fma_mix_simd<FmaMixDst::F32, false>(inst, *wf)));
#else
  ASSERT_TRUE((try_execute_vop3p_fma_mix_simd<FmaMixDst::F32, false>(inst, *wf)));
  EXPECT_EQ(get(3, 0), std::bit_cast<uint32_t>(a * b - 1.0f));
#endif
  ASSERT_TRUE((try_execute_vop3p_fma_mix_simd<FmaMixDst::F32, true>(inst, *wf)));
  EXPECT_EQ(get(3, 0), std::bit_cast<uint32_t>(std::fma(a, b, -1.0f)));
}

// Opt-in simulator host timing. Alternate scalar/SIMD order with matched inputs
// and warmups. These timings describe execution overhead, not GPU hardware.
TEST_F(SimdCoverage, DISABLED_InstructionTimings) {
  constexpr unsigned iterations = 5000;
  for (const auto &test : kDecodedCases) {
    if (std::string_view(test.assembly).find("fmac") != std::string_view::npos)
      continue;
    auto decoder = Decoder::create(test.arch);
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder, test.words));
    ASSERT_NE(instruction, nullptr);
    ComputeUnitCore::Config config{};
    config.arch = test.arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    auto target = ComputeUnitCore::create("simd_benchmark_target", config, &memory, &l2);
    uint32_t wave_size = test.arch == ROCJITSU_CODE_ARCH_CDNA4 ? 64 : 32;
    auto *wave = target->dispatch_wf(0, 0, 106, 256, wave_size);
    ASSERT_NE(wave, nullptr);
    wave->set_mode_raw(3u << 6);
    for (unsigned sample = 0; sample < 12; ++sample) {
      for (unsigned order = 0; order < 2; ++order) {
        bool scalar = (order ^ (sample & 1)) != 0;
        util::set_force_scalar_for_testing(scalar);
        const std::string_view name(test.assembly);
        const bool packed = name.find("pk_") != std::string_view::npos;
        const uint32_t input =
            packed ? (name.find("bf16") != std::string_view::npos ? 0x3f803f80u : 0x3c003c00u)
                   : 0x3f800000u;
        for (uint32_t reg = 0; reg < 10; ++reg)
          for (uint32_t lane = 0; lane < wave_size; ++lane)
            target->write_vgpr(wave->vgpr_alloc().base + reg, lane, input);
        const auto begin = std::chrono::steady_clock::now();
        for (unsigned i = 0; i < iterations; ++i) {
          wave->set_exec(wave_size == 64 ? ~uint64_t{0} : 0xffffffffu);
          auto status = target->execute_instruction(instruction.get(), *wave);
          if (!status.succeeded()) {
            FAIL() << test.assembly;
          }
        }
        const auto elapsed =
            std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - begin)
                .count();
        if (sample >= 3)
          std::printf("SIMD_TIMING\t%d\t%s\t%u\t%d\t%.3f\n", int(test.arch), test.assembly,
                      sample - 3, scalar, elapsed / iterations);
      }
    }
  }
}

} // namespace
