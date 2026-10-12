// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file packed_wmma_test.cpp
/// @brief Packed WMMA raw hardware fixtures across architectures and wave sizes.

#include "fixtures/float_dot/packed_operand_cases.h"
#include "fixtures/float_dot/packed_wmma_cases.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"

#include <array>
#include <cfenv>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>

namespace {
using namespace rocjitsu;

struct HostState {
  fenv_t environment;
  bool scalar = util::force_scalar();
  HostState() { std::fegetenv(&environment); }
  ~HostState() {
    std::fesetenv(&environment);
    util::set_force_scalar_for_testing(scalar);
  }
};
class DotMachine {
public:
  explicit DotMachine(rj_code_arch_t arch, uint32_t width = 32)
      : memory("dot_memory"), cache("dot_cache") {
    cache.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 256;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("dot_test", cfg, &memory, &cache);
    decoder = Decoder::create(arch);
    wave = cu->dispatch_wf(0, 0, 106, 256, width);
    wave->set_exec(width == 64 ? ~uint64_t{0} : 0xffffffff);
    base = wave->vgpr_alloc().base;
  }
  ~DotMachine() { wave->halt(); }
  void set(uint32_t reg, uint32_t lane, uint32_t bits) { cu->write_vgpr(base + reg, lane, bits); }
  uint32_t get(uint32_t reg, uint32_t lane) { return cu->read_vgpr(base + reg, lane); }
  void execute(std::array<uint32_t, 2> encoding) {
    std::array<uint32_t, 4> words{encoding[0], encoding[1], 0, 0};
    auto result = decoder->decode(words.data());
    ASSERT_TRUE(result.succeeded());
    auto instruction = std::move(result).value();
    ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
  }
  amdgpu::GpuMemory memory;
  amdgpu::L2Cache cache;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wave;
  uint32_t base;
};

std::array<uint32_t, 2> packed_encoding(bool gfx12, bool bf16, uint32_t dst, uint32_t high,
                                        uint16_t src2 = 272, uint8_t neg = 0, uint8_t neg_hi = 0) {
  if (gfx12)
    return rdna4::build_vop3p(bf16 ? rdna4::kVWmmaBf1616x16x16Bf16Vop3p
                                   : rdna4::kVWmmaF1616x16x16F16Vop3p,
                              {.vdst = uint8_t(dst),
                               .neg_hi = neg_hi,
                               .src0 = 256,
                               .src1 = 264,
                               .src2 = src2,
                               .opsel_hi = 3,
                               .neg = neg});
  return rdna3::build_vop3p(bf16 ? rdna3::kVWmmaBf1616x16x16Bf16Vop3p
                                 : rdna3::kVWmmaF1616x16x16F16Vop3p,
                            {.vdst = uint8_t(dst),
                             .neg_hi = neg_hi,
                             .op_sel = uint8_t(high << 2),
                             .src0 = 256,
                             .src1 = 264,
                             .src2 = src2,
                             .op_sel_hi = 3,
                             .neg = neg});
}

TEST(PackedWmma, HardwareMatricesBothWavesAndOperandOverlap) {
  HostState restore;
  for (auto arch :
       {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4}) {
    const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
    for (uint32_t width : {32u, 64u}) {
      DotMachine machine(arch, width);
      for (bool bf16 : {false, true})
        for (uint32_t high = 0; high < (gfx12 ? 1u : 2u); ++high)
          for (uint32_t dst : {16u, 24u, 0u, 8u})
            for (uint32_t fixture = 0; fixture < 7; ++fixture)
              for (bool scalar : {false, true}) {
                util::set_force_scalar_for_testing(scalar);
                std::fesetround(scalar ? FE_UPWARD : FE_DOWNWARD);
                std::feclearexcept(FE_ALL_EXCEPT);
                std::feraiseexcept(FE_INEXACT);
                const auto &input = kPackedWmmaInputs[bf16][fixture];
                for (uint32_t lane = 0; lane < width; ++lane) {
                  for (uint32_t reg = 0; reg < 32; ++reg)
                    machine.set(reg, lane, 0xa55a1234u ^ (lane << 16) ^ reg);
                  for (uint32_t reg = 0; reg < (gfx12 ? 128 / width : 8); ++reg) {
                    const uint32_t k = gfx12 ? (lane / 16) * (256 / width) + 2 * reg : 2 * reg;
                    const uint32_t idx = lane % 16;
                    machine.set(reg, lane,
                                input.a[idx * 16 + k] |
                                    (uint32_t(input.a[idx * 16 + k + 1]) << 16));
                    machine.set(8 + reg, lane,
                                input.b[k * 16 + idx] |
                                    (uint32_t(input.b[(k + 1) * 16 + idx]) << 16));
                  }
                  for (uint32_t reg = 0; reg < 256 / width / (gfx12 ? 2 : 1); ++reg) {
                    const uint32_t row = !gfx12 ? (width / 16) * reg + lane / 16
                                         : width == 32
                                             ? (lane / 16) * 8 + 2 * reg
                                             : ((lane >> 4) & 1) * 8 + (lane >> 5) * 4 + 2 * reg;
                    uint32_t value = input.c[row * 16 + lane % 16];
                    if (gfx12)
                      value = (value & 65535) | (input.c[(row + 1) * 16 + lane % 16] << 16);
                    else if (high)
                      value = (value << 16) | (value >> 16);
                    machine.set(16 + reg, lane, value);
                  }
                }
                std::array<uint32_t, 256> old{};
                for (uint32_t reg = 0; reg < 256 / width / (gfx12 ? 2 : 1); ++reg)
                  for (uint32_t lane = 0; lane < width; ++lane)
                    old[reg * width + lane] = machine.get(dst + reg, lane);
                machine.execute(packed_encoding(gfx12, bf16, dst, high));
                for (uint32_t row = 0; row < 16; ++row)
                  for (uint32_t col = 0; col < 16; ++col) {
                    const uint32_t lane = !gfx12 ? col + 16 * (row % (width / 16))
                                          : width == 32
                                              ? col + 16 * (row / 8)
                                              : col + 16 * ((row >> 3) & 1) + 32 * ((row >> 2) & 1);
                    const uint32_t reg = !gfx12 ? row / (width / 16) : (row % (256 / width)) / 2;
                    const uint32_t half = gfx12 ? row % 2 : high;
                    const uint32_t expected =
                        kPackedWmmaExpected[gfx12][width == 64][bf16][fixture][row * 16 + col];
                    const uint32_t actual = machine.get(dst + reg, lane);
                    EXPECT_EQ((actual >> (16 * half)) & 65535, expected)
                        << "arch=" << arch << " width=" << width << " bf16=" << bf16
                        << " fixture=" << fixture << " row=" << row << " col=" << col
                        << " dst=" << dst;
                    if (!gfx12) {
                      const uint32_t mask = high ? 65535 : 0xffff0000u;
                      EXPECT_EQ(actual & mask, old[reg * width + lane] & mask);
                    }
                  }
                EXPECT_EQ(std::fegetround(), scalar ? FE_UPWARD : FE_DOWNWARD);
                EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INEXACT);
              }
    }
  }
}

TEST(PackedWmma, HardwareModifiersSelectorsAndInlineConstants) {
  for (auto arch :
       {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4}) {
    const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
    for (uint32_t width : {32u, 64u}) {
      DotMachine machine(arch, width);
      for (const auto &fixture : kPackedOperandCases) {
        for (uint32_t lane = 0; lane < width; ++lane)
          for (uint32_t reg = 0; reg < 8; ++reg) {
            machine.set(reg, lane, fixture.a);
            machine.set(8 + reg, lane, fixture.b);
            machine.set(16 + reg, lane, fixture.c);
            machine.set(24 + reg, lane, 0x56781234);
          }
        machine.execute({fixture.encoding[0], fixture.encoding[1]});
        for (uint32_t lane = 0; lane < width; ++lane)
          for (uint32_t reg = 0; reg < 8; ++reg) {
            const uint32_t expected =
                reg < (gfx12 ? 128u : 256u) / width ? fixture.expected[gfx12] : 0x56781234;
            EXPECT_EQ(machine.get(24 + reg, lane), expected)
                << "arch=" << arch << " width=" << width << " reg=" << reg
                << " encoding=" << std::hex << fixture.encoding[0] << ':' << fixture.encoding[1];
          }
      }
    }
  }
}

TEST(PackedWmma, HardwareOverflowModeAtIntermediateSteps) {
  for (auto arch :
       {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4}) {
    const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
    for (uint32_t width : {32u, 64u}) {
      DotMachine machine(arch, width);
      for (bool bf16 : {false, true})
        for (bool saturate : {false, true})
          for (uint32_t pattern = 0; pattern < 4; ++pattern)
            for (uint32_t mode : {0u, 0xf0u, 0xffu, 0x300u}) {
              machine.wave->set_mode_raw(mode | (uint32_t(saturate) << 23));
              const uint16_t maximum = bf16 ? 0x7f7f : 0x7bff;
              const uint16_t infinity = bf16 ? 0x7f80 : 0x7c00;
              for (uint32_t lane = 0; lane < width; ++lane) {
                for (uint32_t reg = 0; reg < (gfx12 ? 128 / width : 8); ++reg) {
                  uint32_t a = 0, b = 0;
                  for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t k = !gfx12 ? 2 * reg + half
                                       : width == 32
                                           ? (reg / 2) * 8 + (lane / 16) * 4 + (reg % 2) * 2 + half
                                           : (lane / 16) * 4 + 2 * reg + half;
                    uint16_t av = 0, bv = 0;
                    if (k < 2) {
                      av = pattern == 3 ? infinity : maximum;
                      if (pattern == 2)
                        av |= 0x8000;
                      bv = 0x4000;
                    }
                    if (pattern && k == (gfx12 ? 4u : 2u)) {
                      av = maximum | (pattern == 2 ? 0 : 0x8000);
                      bv = bf16 ? 0x3f80 : 0x3c00;
                    }
                    a |= uint32_t(av) << (16 * half);
                    b |= uint32_t(bv) << (16 * half);
                  }
                  machine.set(reg, lane, a);
                  machine.set(8 + reg, lane, b);
                }
                for (uint32_t reg = 0; reg < 8; ++reg) {
                  machine.set(16 + reg, lane, 0);
                  machine.set(24 + reg, lane, 0x56781234);
                }
              }
              machine.execute(packed_encoding(gfx12, bf16, 24, 0));
              uint16_t expected = infinity | (pattern == 2 ? 0x8000 : 0);
              if (!bf16 && saturate && pattern != 3)
                expected = pattern == 0 ? 0x7bff : gfx12 ? 0 : pattern == 1 ? 0x9800 : 0x1c00;
              const uint32_t word =
                  gfx12 ? expected | (uint32_t(expected) << 16) : 0x56780000 | expected;
              for (uint32_t lane = 0; lane < width; ++lane)
                for (uint32_t reg = 0; reg < (gfx12 ? 128u : 256u) / width; ++reg)
                  EXPECT_EQ(machine.get(24 + reg, lane), word)
                      << "arch=" << arch << " width=" << width << " bf16=" << bf16
                      << " saturate=" << saturate << " pattern=" << pattern;
            }
    }
  }
}
} // namespace
