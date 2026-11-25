// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Older FP64 native math and range-reduction table: guest controls, aliases and masks.
#include "decode_test_util.h"
#include "fp_reference.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/execution_backend.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/isa/target_registry.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"

#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <random>

namespace rocjitsu::amdgpu {
namespace {

// Restore test-only dispatch overrides even when an assertion exits early.
struct ScalarGuard {
  bool saved = util::force_scalar();
  ~ScalarGuard() { util::set_force_scalar_for_testing(saved); }
};

// The inherited model uses host double math, with source NaNs and positive invalid root NaNs.
double reference(double value, int operation, bool ieee) {
  if (std::isnan(value)) {
    return std::bit_cast<double>(std::bit_cast<uint64_t>(value) |
                                 (ieee ? 0x0008000000000000ULL : 0));
  }
  if (operation && value < 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return operation == 0 ? 1.0 / value : operation == 1 ? 1.0 / std::sqrt(value) : std::sqrt(value);
}

// Check every newly routed native F64 form across CPU paths, masks, guest controls and aliases.
TEST(LegacyNativeF64, Arithmetic) {
  ScalarGuard guard;
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    std::array<uint16_t, 3> vop1, vop3;
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1,
       &cdna1::execution_backend(),
       {cdna1::kVRcpF64Vop1, cdna1::kVRsqF64Vop1, cdna1::kVSqrtF64Vop1},
       {cdna1::kVRcpF64Vop3, cdna1::kVRsqF64Vop3, cdna1::kVSqrtF64Vop3}},
      {rocjitsu::IsaFamily::Cdna2,
       &cdna2::execution_backend(),
       {cdna2::kVRcpF64Vop1, cdna2::kVRsqF64Vop1, cdna2::kVSqrtF64Vop1},
       {cdna2::kVRcpF64Vop3, cdna2::kVRsqF64Vop3, cdna2::kVSqrtF64Vop3}},
      {rocjitsu::IsaFamily::Rdna1,
       &rdna1::execution_backend(),
       {rdna1::kVRcpF64Vop1, rdna1::kVRsqF64Vop1, rdna1::kVSqrtF64Vop1},
       {rdna1::kVRcpF64Vop3, rdna1::kVRsqF64Vop3, rdna1::kVSqrtF64Vop3}},
      {rocjitsu::IsaFamily::Rdna2,
       &rdna2::execution_backend(),
       {rdna2::kVRcpF64Vop1, rdna2::kVRsqF64Vop1, rdna2::kVSqrtF64Vop1},
       {rdna2::kVRcpF64Vop3, rdna2::kVRsqF64Vop3, rdna2::kVSqrtF64Vop3}},
      {rocjitsu::IsaFamily::Rdna3_5,
       &rdna3_5::execution_backend(),
       {rdna3_5::kVRcpF64Vop1, rdna3_5::kVRsqF64Vop1, rdna3_5::kVSqrtF64Vop1},
       {rdna3_5::kVRcpF64Vop3, rdna3_5::kVRsqF64Vop3, rdna3_5::kVSqrtF64Vop3}},
  };
  struct Scenario {
    bool vop3;
    uint32_t mode;
    unsigned omod = 0;
    bool negate = false, absolute = false, clamp = false, scalar = false, suppress = false;
  };
  const Scenario scenarios[] = {
      {false, 0},
      {false, 0xc0},
      {false, 0x200},
      {true, 0},
      {true, 0xc0},
      {true, 0xff},
      {true, 0, 2},
      {true, 0x80, 2},
      {true, 0xc0, 2},
      {true, 0x2c0, 2},
      {true, 0x80, 3, true},
      {true, 0x80, 1, false, true, true},
      {true, 0x80, 1, false, false, true},
      {true, 0x180, 1, false, false, true},
      {true, 0, 0, false, false, false, true},
      {false, 0, 0, false, false, false, false, true},
  };
  constexpr uint64_t special[] = {0,
                                  0x8000000000000000ULL,
                                  1,
                                  0x8000000000000001ULL,
                                  0x000fffffffffffffULL,
                                  0x0010000000000000ULL,
                                  0x7fefffffffffffffULL,
                                  0x7ff0000000000000ULL,
                                  0xfff0000000000000ULL,
                                  0x7ff0123456789abcULL,
                                  0xfff8abcdef012345ULL,
                                  0x3ff8000000000000ULL,
                                  0xc010000000000000ULL,
                                  0xbff8000000000000ULL};
  std::minstd_rand random;
  std::array<uint64_t, 64> input;
  for (auto &bits : input) {
    bits = 0;
    for (int half = 0; half < 4; ++half) {
      bits = (bits << 16) | uint64_t(random() >> 15);
    }
  }
  for (int lane = 0; lane < int(std::size(special)); ++lane) {
    input[lane] = special[lane];
  }
  for (const auto &family : families) {
    ScopedIsaExecutionBackend backend(family.backend);
    const bool cdna =
        family.arch == rocjitsu::IsaFamily::Cdna1 || family.arch == rocjitsu::IsaFamily::Cdna2;
    for (int lanes : {32, 64}) {
      if (cdna && lanes == 32) {
        continue;
      }
      for (bool scalar : {true, false}) {
        util::set_force_scalar_for_testing(scalar);
        for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
          ComputeUnitCore::Config config{};
          config.arch = family.arch;
          config.arithmetic = {semantics};
          config.num_wf_slots = 1;
          config.sgprs_per_wf = 104;
          config.vgprs_per_wf = 8;
          config.lds_size_kb = 64;
          GpuMemory memory("native_memory");
          L2Cache cache("native_cache");
          auto cu = ComputeUnitCore::create("native", config, &memory, &cache);
          auto *wave = cu->dispatch_wf(0, 0, 104, 8, lanes);
          ASSERT_NE(wave, nullptr);
          auto decoder = Decoder::create(family.arch);
          const auto base = wave->vgpr_alloc().base;
          for (int operation = 0; operation < 3; ++operation) {
            for (const auto &sample : scenarios) {
              for (int aliasing_offset : {0, 1, 3}) {
                SCOPED_TRACE(::testing::Message()
                             << family.arch << ':' << lanes << ':' << scalar << ':'
                             << int(semantics) << ':' << operation << ':' << sample.vop3 << ':'
                             << sample.mode << ':' << sample.omod << ':' << aliasing_offset);
                const uint64_t exec_mask = lanes == 64 ? 0x800000038000ffffULL : 0x8000ffffULL;
                wave->set_exec(exec_mask);
                wave->set_vgpr_write_mask(sample.suppress ? 0 : ~0ULL);
                wave->set_mode_raw(sample.mode);
                wave->set_trapsts(0x100);
                cu->write_sgpr(wave->sgpr_alloc().base, 0);
                cu->write_sgpr(wave->sgpr_alloc().base + 1, 0x3ff80000);
                for (int lane = 0; lane < lanes; ++lane) {
                  for (int reg = 0; reg < 8; ++reg) {
                    cu->write_vgpr(base + reg, lane, 0xdeadbeef);
                  }
                  cu->write_vgpr(base, lane, uint32_t(input[lane]));
                  cu->write_vgpr(base + 1, lane, uint32_t(input[lane] >> 32));
                }
                const uint32_t source = sample.scalar ? 0 : 256;
                const std::array<uint32_t, 2> words =
                    sample.vop3
                        ? std::array<uint32_t, 2>{(cdna ? 0xd0000000U : 0xd4000000U) |
                                                      (uint32_t(family.vop3[operation]) << 16) |
                                                      uint32_t(aliasing_offset) |
                                                      (sample.absolute ? 1U << 8 : 0) |
                                                      (sample.clamp ? 1U << 15 : 0),
                                                  source | (sample.omod << 27) |
                                                      (sample.negate ? 1U << 29 : 0)}
                        : std::array<uint32_t, 2>{0x7e000000U | (uint32_t(aliasing_offset) << 17) |
                                                      (uint32_t(family.vop1[operation]) << 9) |
                                                      source,
                                                  0};
                std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
                ASSERT_NE(instruction, nullptr);
                ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
                EXPECT_EQ(wave->trapsts(), 0x100U);
                for (int lane = 0; lane < lanes; ++lane) {
                  uint64_t expected = 0;
                  for (int word = 0; word < 2; ++word) {
                    const int reg = aliasing_offset + word;
                    const uint32_t old = reg < 2 ? uint32_t(input[lane] >> (reg * 32)) : 0xdeadbeef;
                    expected |= uint64_t(old) << (word * 32);
                  }
                  if ((exec_mask >> lane & 1) && !sample.suppress) {
                    uint64_t raw = sample.scalar ? 0x3ff8000000000000ULL : input[lane];
                    if (sample.absolute) {
                      raw &= 0x7fffffffffffffffULL;
                    }
                    if (sample.negate) {
                      raw ^= 0x8000000000000000ULL;
                    }
                    double value =
                        reference(std::bit_cast<double>(raw), operation, wave->ieee_mode());
                    // Older targets gate OMOD on IEEE and output-denormal preservation.
                    const unsigned omod =
                        !wave->ieee_mode() && !(wave->fp_denorm_mode_f16_f64() & 2) ? sample.omod
                                                                                    : 0;
                    if (omod) {
                      value *= omod == 3 ? 0.5 : double(1U << omod);
                    }
                    // Native F64 scales first, then active OMOD flushes tiny values to +0.
                    if (omod && std::abs(value) < std::numeric_limits<double>::min()) {
                      value = 0.0;
                    }
                    if (sample.clamp) {
                      value = rocjitsu::test::clamp_floating(value, wave->dx10_clamp());
                    }
                    expected = std::bit_cast<uint64_t>(value);
                  }
                  const uint64_t actual =
                      uint64_t(cu->read_vgpr(base + aliasing_offset, lane)) |
                      (uint64_t(cu->read_vgpr(base + aliasing_offset + 1, lane)) << 32);
                  EXPECT_EQ(actual, expected) << "lane=" << lane;
                }
              }
            }
          }
          wave->halt();
        }
      }
    }
  }
}

// The older profiles share the inherited 1201-bit table and MODE-gated output scaling.

// Check inherited table pieces, output scaling and overlapping registers.
TEST(LegacyNativeF64, TrigPreop) {
  ScalarGuard guard;
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    uint16_t opcode;
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1, &cdna1::execution_backend(), cdna1::kVTrigPreopF64Vop3},
      {rocjitsu::IsaFamily::Cdna2, &cdna2::execution_backend(), cdna2::kVTrigPreopF64Vop3},
      {rocjitsu::IsaFamily::Rdna1, &rdna1::execution_backend(), rdna1::kVTrigPreopF64Vop3},
      {rocjitsu::IsaFamily::Rdna2, &rdna2::execution_backend(), rdna2::kVTrigPreopF64Vop3},
      {rocjitsu::IsaFamily::Rdna3_5, &rdna3_5::execution_backend(), rdna3_5::kVTrigPreopF64Vop3},
  };
  struct Sample {
    unsigned exponent, selector, mode, omod;
    bool clamp;
    uint64_t output;
  };
  // Selected legacy outputs distinguish table tails and scale-before-flush from gfx11 policy.
  constexpr Sample samples[] = {
      {1023, 0, 0, 0, false, 0x3fe45f306dc9c882ULL},
      {1023, 0, 0, 1, false, 0x3ff45f306dc9c882ULL},
      {1023, 0, 0x200, 1, false, 0x3fe45f306dc9c882ULL},
      {1023, 0, 0x80, 1, false, 0x3fe45f306dc9c882ULL},
      {1091, 19, 0, 1, false, 0x0019e839cfbc5294ULL},
      {1091, 19, 4, 3, false, 0},
      {1997, 4, 0, 0, false, 0x012aba10ac06608dULL},
      {1023, 0, 0, 1, true, 0x3ff0000000000000ULL},
      {2047, 31, 0, 0, false, 0},
      {1091, 19, 0x200, 1, false, 0x000cf41ce7de294aULL},
  };
  for (const auto &family : families) {
    ScopedIsaExecutionBackend backend(family.backend);
    const bool cdna =
        family.arch == rocjitsu::IsaFamily::Cdna1 || family.arch == rocjitsu::IsaFamily::Cdna2;
    for (int lanes : {32, 64}) {
      if (cdna && lanes == 32) {
        continue;
      }
      for (bool scalar : {true, false}) {
        util::set_force_scalar_for_testing(scalar);
        for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
          ComputeUnitCore::Config config{};
          config.arch = family.arch;
          config.arithmetic = {semantics};
          config.num_wf_slots = 1;
          config.sgprs_per_wf = 104;
          config.vgprs_per_wf = 8;
          config.lds_size_kb = 64;
          GpuMemory memory("preop_memory");
          L2Cache cache("preop_cache");
          auto cu = ComputeUnitCore::create("preop", config, &memory, &cache);
          auto *wave = cu->dispatch_wf(0, 0, 104, 8, lanes);
          ASSERT_NE(wave, nullptr);
          auto decoder = Decoder::create(family.arch);
          const auto base = wave->vgpr_alloc().base;
          for (const auto &sample : samples) {
            for (int aliasing_offset : {0, 1, 2, 4}) {
              SCOPED_TRACE(::testing::Message() << family.arch << ':' << lanes << ':' << scalar
                                                << ':' << int(semantics) << ':' << sample.exponent
                                                << ':' << sample.omod << ':' << aliasing_offset);
              const uint64_t exec_mask = lanes == 64 ? 0x8000000180000003ULL : 0x80000003ULL;
              wave->set_exec(exec_mask);
              wave->set_mode_raw(sample.mode);
              wave->set_trapsts(0x40);
              for (int lane = 0; lane < lanes; ++lane) {
                for (int reg = 0; reg < 8; ++reg) {
                  cu->write_vgpr(base + reg, lane, 0xdeadbeef);
                }
                cu->write_vgpr(base, lane, uint32_t(lane));
                cu->write_vgpr(base + 1, lane,
                               (sample.exponent << 20) | (uint32_t(lane & 1) << 31));
                cu->write_vgpr(base + 2, lane, sample.selector | 0xffffffe0U);
              }
              std::array<uint64_t, 64> before;
              for (int lane = 0; lane < lanes; ++lane) {
                before[lane] = uint64_t(cu->read_vgpr(base + aliasing_offset, lane)) |
                               (uint64_t(cu->read_vgpr(base + aliasing_offset + 1, lane)) << 32);
              }
              const uint32_t words[] = {
                  (cdna ? 0xd0000000U : 0xd4000000U) | (uint32_t(family.opcode) << 16) |
                      uint32_t(aliasing_offset) | (sample.clamp ? 0x8000U : 0) | 0x100U,
                  256U | (258U << 9) | (sample.omod << 27) | (1U << 29)};
              auto instruction = decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              ASSERT_TRUE(cu->execute_instruction(instruction.value().get(), *wave).succeeded());
              EXPECT_EQ(wave->trapsts(), 0x40U);
              for (int lane = 0; lane < lanes; ++lane) {
                const uint64_t actual =
                    uint64_t(cu->read_vgpr(base + aliasing_offset, lane)) |
                    (uint64_t(cu->read_vgpr(base + aliasing_offset + 1, lane)) << 32);
                EXPECT_EQ(actual, (exec_mask & (1ULL << lane)) ? sample.output : before[lane])
                    << "lane=" << lane;
              }
            }
          }
          wave->halt();
        }
      }
    }
  }
}

} // namespace
} // namespace rocjitsu::amdgpu
