// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rdna_swmmac_i32_test.cpp
/// @brief Decoded sparse integer WMMA regressions from physical gfx1201 captures.
/// @details Saved local outputs constrain a best-effort simulator model. These
/// tests do not establish hardware internals or guarantee untested inputs,
/// flags, modes or devices. See fixtures/float_dot/README.md for provenance.

#include "fixtures/float_dot/rdna_swmmac_iu4_cases.h"
#include "fixtures/float_dot/rdna_swmmac_iu4_k64_cases.h"
#include "fixtures/float_dot/rdna_swmmac_iu8_cases.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace rocjitsu;

struct SwmmacI32Shape {
  uint16_t opcode;
  std::string_view mnemonic;
  uint32_t input_bits;
  uint32_t k_size;
  uint32_t packed_b;
  std::array<uint32_t, 2> key_counts;
  std::array<int32_t, 4> signedness;
  uint32_t wrapped;
  std::span<const RdnaSwmmacIntegerCase> cases;
};

constexpr std::array<SwmmacI32Shape, 3> kSwmmacI32Shapes{{
    {rdna4::kVSwmmacI3216x16x32Iu8Vop3p,
     "v_swmmac_i32_16x16x32_iu8",
     8,
     32,
     0x80808080u,
     {2, 4},
     {522247, -2041, -522233, 2055},
     0x8007f7ffu,
     kRdnaSwmmacIu8Saturated},
    {rdna4::kVSwmmacI3216x16x32Iu4Vop3p,
     "v_swmmac_i32_16x16x32_iu4",
     4,
     32,
     0x88888888u,
     {2, 2},
     {1927, -121, -1913, 135},
     0x8000077fu,
     kRdnaSwmmacIu4Saturated},
    {rdna4::kVSwmmacI3216x16x64Iu4Vop3p,
     "v_swmmac_i32_16x16x64_iu4",
     4,
     64,
     0x88888888u,
     {1, 2},
     {3847, -249, -3833, 263},
     0x80000effu,
     kRdnaSwmmacIu4K64Saturated},
}};

struct SwmmacI32Case {
  uint32_t width;
  bool scalar;
  SwmmacI32Shape shape;
};

class RdnaSwmmacI32 : public testing::TestWithParam<SwmmacI32Case> {
protected:
  void SetUp() override {
    const auto &[width, scalar, shape] = GetParam();
    old_scalar = util::force_scalar();
    util::set_force_scalar_for_testing(scalar);
    cache.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 256;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("swmmac_i32_test", cfg, &memory, &cache);
    decoder = Decoder::create(cfg.arch);
    wave = cu->dispatch_wf(0, 0, 106, 256, width);
    wave->set_exec(width == 64 ? ~uint64_t{0} : 0xffffffffu);
    wave->set_mode_raw(0xf0);
    base = wave->vgpr_alloc().base;
  }
  void TearDown() override {
    if (wave)
      wave->halt();
    util::set_force_scalar_for_testing(old_scalar);
  }
  void execute(uint32_t neg, bool clamp, uint32_t key, uint32_t dst = 24) {
    const auto words = rdna4::build_vop3p(GetParam().shape.opcode, {.vdst = uint8_t(dst),
                                                                    .opsel = uint8_t(key),
                                                                    .opsel_hi_2 = 1,
                                                                    .clamp = uint8_t(clamp),
                                                                    .src0 = 256,
                                                                    .src1 = 264,
                                                                    .src2 = 288,
                                                                    .opsel_hi = 3,
                                                                    .neg = uint8_t(neg)});
    const std::array<uint32_t, 4> code{words[0], words[1], 0, 0};
    auto decoded = decoder->decode(code.data());
    ASSERT_TRUE(decoded.succeeded());
    auto inst = std::move(decoded).value();
    ASSERT_EQ(inst->mnemonic(), GetParam().shape.mnemonic);
    for (uint32_t reg = 0; reg < 33; ++reg)
      for (uint32_t lane = 0; lane < wave->wf_size(); ++lane)
        cu->write_vgpr(base + reg, lane, input[reg * wave->wf_size() + lane]);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
  }
  void seeded_input() {
    uint32_t rng = 0x12056101u;
    constexpr std::array<uint32_t, 8> c{0,           1,           0xffffffffu, 0x7fffffffu,
                                        0x80000000u, 0x7ffff000u, 0x80001000u, 0x12345678u};
    constexpr std::array<uint32_t, 6> pairs{4, 8, 12, 9, 13, 14};
    for (uint32_t reg = 0; reg < 33; ++reg)
      for (uint32_t lane = 0; lane < wave->wf_size(); ++lane) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        uint32_t value = reg >= 16 && reg < 32 ? c[(rng >> 24) & 7] : rng;
        if (reg == 32) {
          value = 0;
          for (uint32_t i = 0; i < 8; ++i)
            value |= pairs[((rng >> (4 * i)) & 15) % 6] << (4 * i);
        }
        input[reg * wave->wf_size() + lane] = value;
      }
  }
  void uniform_input(uint32_t c) {
    for (uint32_t reg = 0; reg < 33; ++reg)
      for (uint32_t lane = 0; lane < wave->wf_size(); ++lane)
        input[reg * wave->wf_size() + lane] = reg < 8    ? 0xffffffffu
                                              : reg < 16 ? GetParam().shape.packed_b
                                              : reg < 24 ? 0
                                              : reg < 32 ? c
                                                         : 0xeeddc988u;
  }
  void expect_registers() {
    for (uint32_t reg = 0; reg < 33; ++reg)
      for (uint32_t lane = 0; lane < wave->wf_size(); ++lane)
        EXPECT_EQ(cu->read_vgpr(base + reg, lane), expected[reg * wave->wf_size() + lane])
            << "reg=" << reg << " lane=" << lane;
  }
  amdgpu::GpuMemory memory{"swmmac_memory"};
  amdgpu::L2Cache cache{"swmmac_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wave = nullptr;
  uint32_t base = 0;
  bool old_scalar = false;
  std::array<uint32_t, 33 * 64> input{}, expected{};
};

TEST_P(RdnaSwmmacI32, HardwareSaturationAndOverlap) {
  for (const auto &fixture : GetParam().shape.cases) {
    if (fixture.width != wave->wf_size())
      continue;
    SCOPED_TRACE(testing::Message() << "key=" << fixture.key << " dst=" << fixture.dst);
    seeded_input();
    expected = input;
    std::copy(fixture.expected.begin(), fixture.expected.end(),
              expected.begin() + fixture.dst * fixture.width);
    const uint32_t count = GetParam().shape.key_counts[wave->wf_size() == 64];
    for (uint32_t key = fixture.key; key < 4; key += count) {
      SCOPED_TRACE(testing::Message() << "encoded key=" << key);
      ASSERT_NO_FATAL_FAILURE(execute(3, true, key, fixture.dst));
      expect_registers();
    }
  }
}

TEST_P(RdnaSwmmacI32, SignednessAndKeys) {
  for (uint32_t neg = 0; neg < 4; ++neg)
    for (bool clamp : {false, true})
      for (uint32_t key = 0; key < 4; ++key) {
        SCOPED_TRACE(testing::Message() << "neg=" << neg << " clamp=" << clamp << " key=" << key);
        uniform_input(7);
        expected = input;
        std::fill_n(expected.begin() + 24 * wave->wf_size(), 256,
                    static_cast<uint32_t>(GetParam().shape.signedness[neg]));
        ASSERT_NO_FATAL_FAILURE(execute(neg, clamp, key));
        expect_registers();
      }
}

TEST_P(RdnaSwmmacI32, OverflowWrapsUnlessClamped) {
  for (bool clamp : {false, true}) {
    uniform_input(0x7fffffffu);
    expected = input;
    std::fill_n(expected.begin() + 24 * wave->wf_size(), 256,
                clamp ? 0x7fffffffu : GetParam().shape.wrapped);
    ASSERT_NO_FATAL_FAILURE(execute(0, clamp, 0));
    expect_registers();
  }
}

std::vector<SwmmacI32Case> rdna_swmmac_i32_cases() {
  std::vector<SwmmacI32Case> cases;
  for (uint32_t width : {32u, 64u})
    for (bool scalar : {false, true})
      for (const auto &shape : kSwmmacI32Shapes)
        cases.push_back({width, scalar, shape});
  return cases;
}

std::string rdna_swmmac_i32_case_name(const testing::TestParamInfo<SwmmacI32Case> &info) {
  const auto &[width, scalar, shape] = info.param;
  return "Iu" + std::to_string(shape.input_bits) + "K" + std::to_string(shape.k_size) + "Wave" +
         std::to_string(width) + (scalar ? "Scalar" : "Default");
}

INSTANTIATE_TEST_SUITE_P(HardwareWidthsAndPaths, RdnaSwmmacI32,
                         testing::ValuesIn(rdna_swmmac_i32_cases()), rdna_swmmac_i32_case_name);
} // namespace
