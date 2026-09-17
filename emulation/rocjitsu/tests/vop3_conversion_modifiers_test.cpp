// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

namespace {

using namespace rocjitsu;

struct ForceScalarGuard {
  explicit ForceScalarGuard(bool force_scalar) : old_force_scalar(util::force_scalar()) {
    util::set_force_scalar_for_testing(force_scalar);
  }
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(old_force_scalar); }

  bool old_force_scalar;
};

TEST(RdnaVop3ConversionTest, IntegerToFloatAppliesOutputModifiers) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA4}) {
    for (bool force_scalar : {false, true}) {
      ForceScalarGuard guard(force_scalar);
      amdgpu::GpuMemory memory("conversion_memory");
      amdgpu::L2Cache l2("conversion_l2");
      amdgpu::ComputeUnitCore::Config cfg{};
      cfg.arch = arch;
      cfg.num_wf_slots = 1;
      cfg.sgprs_per_wf = 106;
      cfg.vgprs_per_wf = 256;
      cfg.lds_size_kb = 64;
      auto cu = amdgpu::ComputeUnitCore::create("conversion_cu", cfg, &memory, &l2);
      auto decoder = Decoder::create(arch);
      auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
      ASSERT_NE(wf, nullptr);
      wf->set_exec(1);
      wf->set_mode_raw(0);
      const uint32_t vb = wf->vgpr_alloc().base;
      for (uint32_t opcode : {389u, 390u, 401u, 402u, 403u, 404u}) {
        for (uint32_t input : {1u, 0xffffffffu}) {
          cu->write_vgpr(vb, 0, input);
          for (uint32_t omod = 0; omod < 4; ++omod) {
            for (uint32_t clamp = 0; clamp < 2; ++clamp) {
              const uint32_t words[] = {0xd4000001u | (opcode << 16) | (clamp << 15),
                                        256u | (omod << 27)};
              std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
              ASSERT_NE(inst, nullptr);
              cu->write_vgpr(vb + 1, 1, 0xdeadbeefu);
              ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
              float expected = opcode == 389 ? static_cast<float>(static_cast<int32_t>(input))
                               : opcode == 390
                                   ? static_cast<float>(input)
                                   : static_cast<float>((input >> ((opcode - 401) * 8)) & 255);
              expected *= std::array{1.0f, 2.0f, 4.0f, 0.5f}[omod];
              if (clamp)
                expected = std::clamp(expected, 0.0f, 1.0f);
              EXPECT_EQ(cu->read_vgpr(vb + 1, 0), std::bit_cast<uint32_t>(expected))
                  << "arch=" << arch << " opcode=" << opcode << " omod=" << omod
                  << " clamp=" << clamp << " scalar=" << force_scalar;
              EXPECT_EQ(cu->read_vgpr(vb + 1, 1), 0xdeadbeefu);
            }
          }
        }
      }
      wf->halt();
    }
  }
}

TEST(RdnaVop3ConversionTest, FloatToHalfPreservesOverflowMode) {
  struct Case {
    float input;
    uint32_t normal;
    uint32_t saturated;
  };
  constexpr float inf = std::numeric_limits<float>::infinity();
  constexpr Case cases[] = {
      {70000.0f, 0x7c00, 0x7bff},  {-70000.0f, 0xfc00, 0xfbff}, {65504.0f, 0x7bff, 0x7bff},
      {-65504.0f, 0xfbff, 0xfbff}, {1.0f, 0x3c00, 0x3c00},      {0.0f, 0, 0},
      {inf, 0x7c00, 0x7c00},       {-inf, 0xfc00, 0xfc00},
  };
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    for (bool force_scalar : {false, true}) {
      ForceScalarGuard guard(force_scalar);
      amdgpu::GpuMemory memory("half_conversion_memory");
      amdgpu::L2Cache l2("half_conversion_l2");
      amdgpu::ComputeUnitCore::Config cfg{};
      cfg.arch = arch;
      cfg.num_wf_slots = 1;
      cfg.sgprs_per_wf = 106;
      cfg.vgprs_per_wf = 16;
      auto cu = amdgpu::ComputeUnitCore::create("half_conversion_cu", cfg, &memory, &l2);
      auto decoder = Decoder::create(arch);
      auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
      ASSERT_NE(wf, nullptr);
      const uint32_t vb = wf->vgpr_alloc().base;
      // v_cvt_f16_f32 v2, v0, with no source or output modifiers.
      const uint32_t words[] = {0xd58a0002, 0x00000100};
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
      ASSERT_NE(inst, nullptr);
      for (bool fp16_ovfl : {false, true}) {
        wf->set_mode_raw(fp16_ovfl ? amdgpu::Wavefront::FP16_OVFL_BIT : 0u);
        for (uint64_t exec : {wf->wf_size() == 64 ? ~0ull : 0xffffffffull, 0x55ull}) {
          wf->set_exec(exec);
          for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
            cu->write_vgpr(vb, lane, std::bit_cast<uint32_t>(cases[lane % std::size(cases)].input));
            cu->write_vgpr(vb + 2, lane, 0xdeadbeef);
          }
          ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
          for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
            const auto &c = cases[lane % std::size(cases)];
            const uint32_t expected =
                exec & (1ull << lane) ? (fp16_ovfl ? c.saturated : c.normal) : 0xdeadbeefu;
            EXPECT_EQ(cu->read_vgpr(vb + 2, lane), expected)
                << "arch=" << arch << " input=" << c.input << " fp16_ovfl=" << fp16_ovfl
                << " scalar=" << force_scalar << " exec=" << exec << " lane=" << lane;
          }
        }
      }
      wf->halt();
    }
  }
}

class Vop3ConversionModifierTest : public ::testing::TestWithParam<rj_code_arch_t> {
protected:
  bool gfx9() const {
    const auto arch = GetParam();
    return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
           arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
  }
};

INSTANTIATE_TEST_SUITE_P(AllArchitectures, Vop3ConversionModifierTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                           ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                           ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA1,
                                           ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                                           ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5));

TEST_P(Vop3ConversionModifierTest, FloatToIntegerConversionAppliesSourceModifiers) {
  for (auto arch : {GetParam()}) {
    for (bool force_scalar : {false, true}) {
      ForceScalarGuard guard(force_scalar);
      amdgpu::GpuMemory mem("cvt_modifier_mem");
      amdgpu::L2Cache l2("cvt_modifier_l2");
      amdgpu::ComputeUnitCore::Config cfg{};
      cfg.arch = arch;
      cfg.num_wf_slots = 1;
      cfg.sgprs_per_wf = gfx9() ? 102 : 106;
      cfg.vgprs_per_wf = 16;
      cfg.lds_size_kb = 64;
      auto cu = amdgpu::ComputeUnitCore::create("cvt_modifier", cfg, &mem, &l2);
      auto decoder = Decoder::create(arch);
      auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, 16);
      ASSERT_NE(wf, nullptr);
      const auto vb = wf->vgpr_alloc().base;
      for (uint32_t op : {gfx9() ? 327u : 391u, gfx9() ? 328u : 392u, gfx9() ? 332u : 396u,
                          gfx9() ? 333u : 397u}) {
        for (uint32_t modifiers = 0; modifiers < 4; ++modifiers) {
          for (uint64_t exec : {wf->wf_size() == 64 ? ~0ull : 0xffffffffull, 0x55ull}) {
            wf->set_exec(exec);
            for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
              cu->write_vgpr(vb, lane, std::bit_cast<uint32_t>(-7.75f));
              cu->write_vgpr(vb + 2, lane, 0xdeadbeef);
            }
            // Truncating, nearest and floor conversions with independent ABS and NEG.
            std::array<uint32_t, 2> words{(gfx9() ? 0xd0000002u : 0xd4000002u) | (op << 16) |
                                              ((modifiers & 1) << 8),
                                          256u | ((modifiers >> 1) << 29)};
            std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
            ASSERT_NE(inst, nullptr);
            ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
            const bool positive = modifiers == 1 || modifiers == 2;
            const uint32_t expected = op == (gfx9() ? 332u : 396u) ? (positive ? 8u : uint32_t(-8))
                                      : op == (gfx9() ? 333u : 397u)
                                          ? (positive ? 7u : uint32_t(-8))
                                      : positive                 ? 7u
                                      : (op == 391 || op == 327) ? 0u
                                                                 : uint32_t(-7);
            for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
              EXPECT_EQ(cu->read_vgpr(vb + 2, lane),
                        exec & (1ull << lane) ? expected : 0xdeadbeefu);
          }
        }
      }
      wf->halt();
    }
  }
}

TEST_P(Vop3ConversionModifierTest, NearestRoundsTiesUpWithoutRoundingAdjacentInputs) {
  struct Case {
    uint32_t bits;
    int32_t expected;
  };
  // Ties and adjacent representable values match physical gfx1100/gfx1201.
  // The same rounding rule is specified for RPI on earlier RDNA and CDNA.
  constexpr Case cases[] = {
      {0xc0200001, -3},          {0xc0200000, -2},        {0xc01fffff, -2},
      {0xbfc00001, -2},          {0xbfc00000, -1},        {0xbfbfffff, -1},
      {0xbf000001, -1},          {0xbf000000, 0},         {0xbeffffff, 0},
      {0x3effffff, 0},           {0x3f000000, 1},         {0x3f000001, 1},
      {0x3fbfffff, 1},           {0x3fc00000, 2},         {0x3fc00001, 2},
      {0x401fffff, 2},           {0x40200000, 3},         {0x40200001, 3},
      {0x00000000, 0},           {0x80000000, 0},         {0x00000001, 0},
      {0x80000001, 0},           {0x4b000001, 8388609},   {0xcb000001, -8388609},
      {0x4b800000, 16777216},    {0xcb800000, -16777216}, {0x4effffff, 2147483520},
      {0xceffffff, -2147483520}, {0x4f000000, INT32_MAX}, {0xcf000000, INT32_MIN},
      {0x4f000001, INT32_MAX},   {0xcf000001, INT32_MIN}, {0x7f800000, INT32_MAX},
      {0xff800000, INT32_MIN},   {0x7fc00000, 0},         {0xffc00000, 0},
      {0x7f800001, 0},
  };
  for (bool force_scalar : {false, true}) {
    ForceScalarGuard guard(force_scalar);
    amdgpu::GpuMemory memory("nearest_memory");
    amdgpu::L2Cache l2("nearest_l2");
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = GetParam();
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = gfx9() ? 102 : 106;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("nearest", cfg, &memory, &l2);
    auto decoder = Decoder::create(GetParam());
    auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, 16);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(wf->wf_size() == 64 ? ~0ull : 0xffffffffull);
    const auto vb = wf->vgpr_alloc().base;
    for (bool vop3 : {false, true}) {
      const std::array<uint32_t, 2> words =
          vop3 ? std::array<uint32_t, 2>{(gfx9() ? 0xd0000002u : 0xd4000002u) |
                                             ((gfx9() ? 332u : 396u) << 16),
                                         256u}
               : std::array<uint32_t, 2>{0x7e041900u, 0u};
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
      ASSERT_NE(inst, nullptr);
      for (size_t first = 0; first < std::size(cases); first += wf->wf_size()) {
        for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
          const auto &c = cases[(first + lane) % std::size(cases)];
          cu->write_vgpr(vb, lane, c.bits);
        }
        ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
        for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
          const auto &c = cases[(first + lane) % std::size(cases)];
          EXPECT_EQ(cu->read_vgpr(vb + 2, lane), static_cast<uint32_t>(c.expected))
              << "arch=" << GetParam() << " input=" << c.bits << " vop3=" << vop3
              << " scalar=" << force_scalar;
        }
      }
    }
    wf->halt();
  }
}

} // namespace
