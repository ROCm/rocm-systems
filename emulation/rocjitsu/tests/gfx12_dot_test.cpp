// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx12_dot_test.cpp
/// @brief Hardware-derived GFX12 DOT2 arithmetic and operand regression tests.

#include "fixtures/float_dot/gfx1201_cases.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx12_dot.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/simd_test_hooks.h"

#include <array>
#include <cfenv>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <span>

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

TEST(Gfx12Dot, HardwareBitsAreIndependentOfHostFpState) {
  HostState restore;
  for (int round : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    std::fesetround(round);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INEXACT);
    for (bool bf16 : {false, true}) {
      const auto cases = bf16 ? std::span<const Gfx12DotCase>(kGfx12DotBF16Cases)
                              : std::span<const Gfx12DotCase>(kGfx12DotF16Cases);
      for (const auto &c : cases) {
        const uint32_t result =
            bf16 ? amdgpu::gfx12_dot2_f32<true>(c.a, c.b, c.a >> 16, c.b >> 16, c.c)
                 : amdgpu::gfx12_dot2_f32<false>(c.a, c.b, c.a >> 16, c.b >> 16, c.c);
        EXPECT_EQ(result, c.expected) << std::hex << c.a << ' ' << c.b << ' ' << c.c;
      }
    }
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INEXACT);
    EXPECT_EQ(std::fegetround(), round);
  }
}

TEST(Gfx12Dot, DecodedHardwareCasesAndInactiveLanes) {
  HostState restore;
  constexpr auto arch = ROCJITSU_CODE_ARCH_RDNA4;
  DotMachine machine(arch);
  for (bool force_scalar : {false, true}) {
    util::set_force_scalar_for_testing(force_scalar);
    for (bool bf16 : {false, true}) {
      const auto cases = bf16 ? std::span<const Gfx12DotCase>(kGfx12DotBF16Cases)
                              : std::span<const Gfx12DotCase>(kGfx12DotF16Cases);
      for (uint32_t exec : {0xffffffffu, 0xa55a5aa5u}) {
        machine.wave->set_exec(exec);
        for (uint32_t modifiers = 0; modifiers < 32; ++modifiers) {
          const uint8_t neg = modifiers & 7, neg_hi = (modifiers >> 3) & 3;
          for (size_t offset = 0; offset < cases.size(); offset += 32) {
            for (uint32_t lane = 0; lane < 32; ++lane) {
              const auto &c = cases[(offset + lane) % cases.size()];
              // Fold the inverse source modifiers into the data. Raw-bit
              // negation preserves signaling NaNs, unlike host arithmetic.
              machine.set(0, lane, c.a ^ ((neg & 1) << 15) ^ ((neg_hi & 1) << 31));
              machine.set(1, lane, c.b ^ ((neg & 2) << 14) ^ ((neg_hi & 2) << 30));
              machine.set(2, lane, c.c ^ ((neg & 4) << 29));
              machine.set(16, lane, 0xfeedface);
            }
            machine.execute(rdna4::build_vop3p(
                bf16 ? rdna4::kVDot2F32Bf16Vop3p : rdna4::kVDot2F32F16Vop3p, {.vdst = 16,
                                                                              .neg_hi = neg_hi,
                                                                              .src0 = 256,
                                                                              .src1 = 257,
                                                                              .src2 = 258,
                                                                              .opsel_hi = 3,
                                                                              .neg = neg}));
            for (uint32_t lane = 0; lane < 32; ++lane)
              EXPECT_EQ(machine.get(16, lane), (exec & (1u << lane))
                                                   ? cases[(offset + lane) % cases.size()].expected
                                                   : 0xfeedfaceu)
                  << "arch=" << arch << " bf16=" << bf16 << " modifiers=" << modifiers;
          }
        }
      }
    }
  }
}

TEST(Gfx12Dot, InlineSourcesMatchHardware) {
  // R9700 captures. F16 narrows floats into the low half; BF16 broadcasts
  // upper16(FP32) without rounding, including selector 248 (1 / (2*pi)).
  constexpr auto arch = ROCJITSU_CODE_ARCH_RDNA4;
  DotMachine machine(arch);
  for (bool bf16 : {false, true}) {
    const uint32_t packed = bf16 ? 0x40003f80 : 0x40003c00; // (1, 2)
    for (uint16_t selector : {uint16_t(129), uint16_t(242), uint16_t(248)}) {
      const std::array<uint32_t, 4> expected =
          selector == 129
              ? (bf16 ? std::array<uint32_t, 4>{0x00030000, 0x00030000, 0x00030000, 0x00030000}
                      : std::array<uint32_t, 4>{0x34400000, 0x33800000, 0x34000000, 0})
          : bf16
              ? (selector == 242
                     ? std::array<uint32_t, 4>{0x40400000, 0x40400000, 0x40400000, 0x40400000}
                     : std::array<uint32_t, 4>{0x3ef30000, 0x3ef30000, 0x3ef30000, 0x3ef30000})
              : (selector == 242 ? std::array<uint32_t, 4>{0x40400000, 0x3f800000, 0x40000000, 0}
                                 : std::array<uint32_t, 4>{0x3ef48000, 0x3e230000, 0x3ea30000, 0});
      for (uint32_t position = 0; position < 2; ++position) {
        for (uint32_t select = 0; select < 4; ++select) {
          machine.set(0, 0, packed);
          machine.execute(rdna4::build_vop3p(
              bf16 ? rdna4::kVDot2F32Bf16Vop3p : rdna4::kVDot2F32F16Vop3p,
              {.vdst = 16,
               .opsel = uint8_t((select >> 1) << position),
               .src0 = uint16_t(position ? 256 : selector),
               .src1 = uint16_t(position ? selector : 256),
               .src2 = 128,
               .opsel_hi = uint8_t((3 ^ (1 << position)) | ((select & 1) << position))}));
          EXPECT_EQ(machine.get(16, 0), expected[select])
              << "bf16=" << bf16 << " selector=" << selector << " position=" << position
              << " select=" << select;
        }
      }
    }
  }
}

TEST(Gfx12Dot, WmmaHardwareRoundingOrderAndOverlappingSources) {
  for (uint32_t width : {32u, 64u}) {
    DotMachine machine(ROCJITSU_CODE_ARCH_RDNA4, width);
    for (bool bf16 : {false, true}) {
      const auto cases = bf16 ? std::span<const Gfx12WmmaCase>(kGfx12WmmaBF16Cases)
                              : std::span<const Gfx12WmmaCase>(kGfx12WmmaF16Cases);
      for (const auto &fixture : cases) {
        // All rows/columns carry the captured vectors, so every output must
        // reproduce the captured value. Cover D/C, D/A, and D/B overlap.
        for (uint8_t dst : {uint8_t(16), uint8_t(0), uint8_t(8)}) {
          for (uint32_t lane = 0; lane < width; ++lane) {
            for (uint32_t reg = 0; reg < 128 / width; ++reg) {
              const uint32_t k = (lane / 16) * (256 / width) + 2 * reg;
              machine.set(reg, lane, fixture.a[k] | (uint32_t(fixture.a[k + 1]) << 16));
              machine.set(8 + reg, lane, fixture.b[k] | (uint32_t(fixture.b[k + 1]) << 16));
            }
            for (uint32_t reg = 0; reg < 256 / width; ++reg)
              machine.set(16 + reg, lane, fixture.c);
          }
          machine.wave->set_exec(0x55aa55aa); // WMMA uses all lanes and preserves EXEC.
          machine.execute(rdna4::build_vop3p(
              bf16 ? rdna4::kVWmmaF3216x16x16Bf16Vop3p : rdna4::kVWmmaF3216x16x16F16Vop3p,
              {.vdst = dst, .src0 = 256, .src1 = 264, .src2 = 272, .opsel_hi = 3}));
          EXPECT_EQ(machine.wave->exec(), 0x55aa55aau);
          for (uint32_t lane = 0; lane < width; ++lane)
            for (uint32_t reg = 0; reg < 256 / width; ++reg)
              EXPECT_EQ(machine.get(dst + reg, lane),
                        width == 32 ? fixture.expected32 : fixture.expected64)
                  << "width=" << width << " bf16=" << bf16 << " dst=" << unsigned(dst);
        }
      }
    }
  }
}

TEST(Gfx12Dot, WmmaSignModifiersAndWaveSizes) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_RDNA4;
  for (uint32_t width : {32u, 64u}) {
    DotMachine machine(arch, width);
    for (bool bf16 : {false, true}) {
      for (uint32_t modifiers = 0; modifiers < 64; ++modifiers) {
        const uint8_t neg = modifiers & 7, neg_hi = modifiers >> 3;
        std::array<uint32_t, 256> expected{};
        for (bool explicit_modifiers : {true, false}) {
          for (uint32_t lane = 0; lane < width; ++lane) {
            for (uint32_t reg = 0; reg < 128 / width; ++reg) {
              const uint32_t index = lane * (128 / width) + reg;
              // Integers, signed zeros, infinities and arbitrary NaN bits.
              uint32_t src0 = index * 0x1f123bb5u;
              uint32_t src1 = index * 0x9e3779b9u;
              if (explicit_modifiers) {
                src0 ^= ((neg & 1) << 15) | ((neg_hi & 1) << 31);
                src1 ^= ((neg & 2) << 14) | ((neg_hi & 2) << 30);
              }
              machine.set(reg, lane, src0);
              machine.set(8 + reg, lane, src1);
            }
            for (uint32_t reg = 0; reg < 256 / width; ++reg) {
              uint32_t acc = (lane + reg * width) * 0x6c078965u;
              if (explicit_modifiers) {
                if (neg_hi & 4)
                  acc &= 0x7fffffffu;
                if (neg & 4)
                  acc ^= 0x80000000u;
              }
              machine.set(16 + reg, lane, acc);
            }
          }
          machine.execute(rdna4::build_vop3p(bf16 ? rdna4::kVWmmaF3216x16x16Bf16Vop3p
                                                  : rdna4::kVWmmaF3216x16x16F16Vop3p,
                                             {.vdst = 16,
                                              .neg_hi = uint8_t(explicit_modifiers ? 0 : neg_hi),
                                              .src0 = 256,
                                              .src1 = 264,
                                              .src2 = 272,
                                              .opsel_hi = 3,
                                              .neg = uint8_t(explicit_modifiers ? 0 : neg)}));
          for (uint32_t lane = 0; lane < width; ++lane) {
            for (uint32_t reg = 0; reg < 256 / width; ++reg) {
              if (explicit_modifiers)
                expected[reg * width + lane] = machine.get(16 + reg, lane);
              else
                EXPECT_EQ(machine.get(16 + reg, lane), expected[reg * width + lane])
                    << "arch=" << arch << " width=" << width << " bf16=" << bf16
                    << " modifiers=" << modifiers;
            }
          }
        }
      }
    }
  }
}

TEST(Gfx12Dot, WmmaInlineAccumulatorPreservesAllOnesNan) {
  for (uint32_t width : {32u, 64u}) {
    DotMachine machine(ROCJITSU_CODE_ARCH_RDNA4, width);
    for (bool bf16 : {false, true}) {
      for (uint32_t modifiers = 0; modifiers < 4; ++modifiers) {
        for (uint32_t lane = 0; lane < width; ++lane) {
          for (uint32_t reg = 0; reg < 128 / width; ++reg) {
            machine.set(reg, lane, 0);
            machine.set(8 + reg, lane, 0);
          }
          for (uint32_t reg = 0; reg < 256 / width; ++reg)
            machine.set(16 + reg, lane, 0x12345678);
        }
        machine.execute(rdna4::build_vop3p(bf16 ? rdna4::kVWmmaF3216x16x16Bf16Vop3p
                                                : rdna4::kVWmmaF3216x16x16F16Vop3p,
                                           {.vdst = 16,
                                            .neg_hi = uint8_t((modifiers & 2) << 1),
                                            .src0 = 256,
                                            .src1 = 264,
                                            .src2 = 193,
                                            .opsel_hi = 3,
                                            .neg = uint8_t((modifiers & 1) << 2)}));
        uint32_t expected = modifiers & 2 ? 0x7fffffffu : 0xffffffffu;
        if (modifiers & 1)
          expected ^= 0x80000000u;
        for (uint32_t lane = 0; lane < width; ++lane)
          for (uint32_t reg = 0; reg < 256 / width; ++reg)
            EXPECT_EQ(machine.get(16 + reg, lane), expected);
      }
    }
  }
}

TEST(Gfx12Dot, AccumulateEncodingsAndInlineSources) {
  HostState restore;
  constexpr auto arch = ROCJITSU_CODE_ARCH_RDNA4;
  DotMachine machine(arch);
  for (bool scalar : {false, true}) {
    util::set_force_scalar_for_testing(scalar);
    for (bool bf16 : {false, true}) {
      const auto cases = bf16 ? std::span<const Gfx12DotCase>(kGfx12DotBF16Cases)
                              : std::span<const Gfx12DotCase>(kGfx12DotF16Cases);
      for (uint32_t form = 2; form <= 3; ++form) {
        const uint32_t dst = form == 3 ? 17 : 16;
        auto encode = [&](uint16_t selector) -> std::array<uint32_t, 2> {
          const uint32_t dot_op = bf16 ? 13 : 12;
          const uint32_t opx = form == 2 ? dot_op : 8, opy = form == 3 ? dot_op : 8;
          return {0xc8000000u | (opx << 22) | (opy << 17) | (1 << 9) |
                      (form == 2 ? selector : 128u),
                  (16u << 24) | (8 << 17) | (1 << 9) | (form == 3 ? selector : 128u)};
        };
        for (const auto &fixture : cases) {
          machine.set(0, 0, fixture.a);
          machine.set(1, 0, fixture.b);
          machine.set(dst, 0, fixture.c);
          machine.execute(encode(256));
          EXPECT_EQ(machine.get(dst, 0), fixture.expected)
              << "arch=" << arch << " form=" << form << " bf16=" << bf16;
        }
        for (uint16_t selector : {uint16_t(129), uint16_t(242), uint16_t(248)}) {
          machine.set(1, 0, bf16 ? 0x40003f80 : 0x40003c00);
          machine.set(dst, 0, 0);
          machine.execute(encode(selector));
          const uint32_t expected = selector == 129   ? (bf16 ? 0x00030000 : 0x34400000)
                                    : selector == 242 ? 0x40400000
                                    : bf16            ? 0x3ef30000
                                                      : 0x3ef48000;
          EXPECT_EQ(machine.get(dst, 0), expected)
              << "form=" << form << " bf16=" << bf16 << " selector=" << selector;
        }
      }
    }
  }
}
} // namespace
