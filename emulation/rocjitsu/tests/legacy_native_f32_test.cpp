// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded older FP32 reciprocal/root bindings: output controls, aliases and masks.
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
#include "unary_binding.h"
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

// Binding reference only: numerical models have independent public API goldens.
float reference(float value, int operation, bool quiet) {
  const rocjitsu::test::Unary32 calls[] = {rj_arith_v_rcp_f32_w32, rj_arith_v_rsq_f32_w32,
                                           rj_arith_v_sqrt_f32_w32, rj_arith_v_rcp_f32_w32};
  const uint64_t flags = RJ_ARITH_SEMANTICS_COMPATIBILITY | (quiet ? RJ_ARITH_FP_IEEE_MODE : 0);
  return std::bit_cast<float>(uint32_t(rocjitsu::test::evaluate_unary32(
      calls[operation], flags, 0, std::bit_cast<uint32_t>(value))));
}

// Check every newly routed native F32 form across CPU paths, masks, guest controls and aliases.
TEST(LegacyNativeF32, Arithmetic) {
  ScalarGuard guard;
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    std::array<uint16_t, 4> vop1, vop3;
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1,
       &cdna1::execution_backend(),
       {cdna1::kVRcpF32Vop1, cdna1::kVRsqF32Vop1, cdna1::kVSqrtF32Vop1, cdna1::kVRcpIflagF32Vop1},
       {cdna1::kVRcpF32Vop3, cdna1::kVRsqF32Vop3, cdna1::kVSqrtF32Vop3, cdna1::kVRcpIflagF32Vop3}},
      {rocjitsu::IsaFamily::Cdna2,
       &cdna2::execution_backend(),
       {cdna2::kVRcpF32Vop1, cdna2::kVRsqF32Vop1, cdna2::kVSqrtF32Vop1, cdna2::kVRcpIflagF32Vop1},
       {cdna2::kVRcpF32Vop3, cdna2::kVRsqF32Vop3, cdna2::kVSqrtF32Vop3, cdna2::kVRcpIflagF32Vop3}},
      {rocjitsu::IsaFamily::Rdna1,
       &rdna1::execution_backend(),
       {rdna1::kVRcpF32Vop1, rdna1::kVRsqF32Vop1, rdna1::kVSqrtF32Vop1, rdna1::kVRcpIflagF32Vop1},
       {rdna1::kVRcpF32Vop3, rdna1::kVRsqF32Vop3, rdna1::kVSqrtF32Vop3, rdna1::kVRcpIflagF32Vop3}},
      {rocjitsu::IsaFamily::Rdna2,
       &rdna2::execution_backend(),
       {rdna2::kVRcpF32Vop1, rdna2::kVRsqF32Vop1, rdna2::kVSqrtF32Vop1, rdna2::kVRcpIflagF32Vop1},
       {rdna2::kVRcpF32Vop3, rdna2::kVRsqF32Vop3, rdna2::kVSqrtF32Vop3, rdna2::kVRcpIflagF32Vop3}},
      {rocjitsu::IsaFamily::Rdna3_5,
       &rdna3_5::execution_backend(),
       {rdna3_5::kVRcpF32Vop1, rdna3_5::kVRsqF32Vop1, rdna3_5::kVSqrtF32Vop1,
        rdna3_5::kVRcpIflagF32Vop1},
       {rdna3_5::kVRcpF32Vop3, rdna3_5::kVRsqF32Vop3, rdna3_5::kVSqrtF32Vop3,
        rdna3_5::kVRcpIflagF32Vop3}},
  };
  struct Scenario {
    bool vop3;
    uint32_t mode;
    unsigned omod = 0;
    bool negate = false, absolute = false, clamp = false, scalar = false, suppress = false;
    int extension = 0; // 1: SDWA high-half store; 2: DPP identity with row-zero writes suppressed.
  };
  const Scenario scenarios[] = {
      {false, 0},
      {false, 0x30},
      {false, 0x200},
      {true, 0},
      {true, 0x30},
      {true, 0xff},
      {true, 0, 2},
      {true, 0x20, 2},
      {true, 0x30, 2},
      {true, 0x230, 2},
      {true, 0x20, 3, true},
      {true, 0x20, 1, false, true, true},
      {true, 0x20, 1, false, false, true},
      {true, 0x120, 1, false, false, true},
      {true, 0, 0, false, false, false, true},
      {false, 0, 0, false, false, false, false, true},
      {false, 0x20, 0, false, false, false, false, false, 1},
      {false, 0x20, 0, false, false, false, false, false, 2},
  };
  constexpr uint32_t special[] = {0,           0x80000000U, 1,           0x80000001U,
                                  0x007fffffU, 0x00800000U, 0x7f7fffffU, 0x7f800000U,
                                  0xff800000U, 0x7f812345U, 0xffc12345U, 0x3fc00000U,
                                  0xc0800000U, 0xbfc00000U, 0x40000000U, 0x40800000U};
  std::minstd_rand random;
  std::array<uint32_t, 64> input;
  for (auto &bits : input) {
    const uint32_t high = random() >> 15;
    bits = (high << 16) | uint32_t(random() >> 15);
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
          for (int operation = 0; operation < 4; ++operation) {
            for (const auto &sample : scenarios) {
              if ((sample.extension == 1 && semantics == ArithmeticSemantics::Loose) ||
                  (sample.extension && family.arch == rocjitsu::IsaFamily::Rdna3_5)) {
                continue;
              }
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
                cu->write_sgpr(wave->sgpr_alloc().base, 0x40800000U);
                for (int lane = 0; lane < lanes; ++lane) {
                  for (int reg = 0; reg < 8; ++reg) {
                    cu->write_vgpr(base + reg, lane, 0xdeadbeef);
                  }
                  cu->write_vgpr(base, lane, input[lane]);
                }
                const uint32_t source = sample.extension == 1   ? SRC_SDWA
                                        : sample.extension == 2 ? SRC_DPP
                                        : sample.scalar         ? 0
                                                                : 256;
                std::array<uint32_t, 2> words =
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
                if (sample.extension == 1) {
                  words[1] = (sdwa::WORD_1 << 8) | (sdwa::UNUSED_PRESERVE << 11) | (1U << 14) |
                             (sdwa::DWORD << 16);
                } else if (sample.extension == 2) {
                  words[1] = (0xe4U << 8) | (0xfU << 24) | (2U << 28);
                }
                const uint64_t write_mask = sample.suppress         ? 0
                                            : sample.extension == 2 ? exec_mask & 0xffff0000ULL
                                                                    : exec_mask;
                std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
                ASSERT_NE(instruction, nullptr);
                ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
                uint32_t causes = 0;
                for (int lane = 0; lane < lanes; ++lane) {
                  uint32_t expected = aliasing_offset == 0 ? input[lane] : 0xdeadbeef;
                  uint32_t raw = sample.scalar ? 0x40800000U : input[lane];
                  const uint32_t unmodified = raw;
                  if (sample.absolute) {
                    raw &= 0x7fffffffU;
                  }
                  if (sample.negate) {
                    raw ^= 0x80000000U;
                  }
                  if ((exec_mask >> lane & 1) && semantics == ArithmeticSemantics::Compatibility) {
                    if (operation == 2 && (raw >> 31) && (raw & 0x7fffffffU) != 0 &&
                        (raw & 0x7fffffffU) <= 0x7f800000U) {
                      causes |= 1;
                    }
                    // IFLAG classifies the original source before ABS/NEG and destination
                    // suppression.
                    if (operation == 3 && !(unmodified & 0x7fffffffU)) {
                      causes |= 64;
                    }
                  }
                  if (write_mask >> lane & 1) {
                    float value = reference(std::bit_cast<float>(raw), operation,
                                            operation == 3 || wave->ieee_mode());
                    // Older targets disable OMOD under IEEE or preserved output denormals.
                    const bool scale =
                        !wave->ieee_mode() && (operation != 3 || !(wave->fp_denorm_mode_f32() & 2));
                    const unsigned omod = scale ? (sample.extension == 1 ? 1 : sample.omod) : 0;
                    value = rocjitsu::test::finish_floating(value, {omod, 0, false, false});
                    if (sample.clamp) {
                      value = rocjitsu::test::clamp_floating(value, wave->dx10_clamp());
                    }
                    expected = std::bit_cast<uint32_t>(value);
                    if (sample.extension == 1) {
                      const uint32_t old = aliasing_offset == 0 ? input[lane] : 0xdeadbeef;
                      expected = (expected << 16) | (old & 0xffffU);
                    }
                  }
                  const uint32_t actual = cu->read_vgpr(base + aliasing_offset, lane);
                  if (semantics == ArithmeticSemantics::Loose && (write_mask >> lane & 1) &&
                      sample.extension != 1) {
                    const float a = std::bit_cast<float>(actual),
                                e = std::bit_cast<float>(expected);
                    if (std::isnan(e)) {
                      EXPECT_TRUE(std::isnan(a));
                    } else if (std::isinf(e)) {
                      EXPECT_EQ(a, e);
                    } else {
                      EXPECT_NEAR(a, e, 1e-6f + 1e-5f * std::abs(e));
                    }
                  } else {
                    EXPECT_EQ(actual, expected) << "lane=" << lane;
                  }
                }
                EXPECT_EQ(wave->trapsts(), 0x100U | causes);
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
