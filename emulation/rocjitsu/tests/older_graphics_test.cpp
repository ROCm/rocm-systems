// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Older MULLIT decoding retains the inherited fixup, guest controls, and aliases.
// Cube's existing cross-architecture witnesses cover the other migrated operations.
#include "compute_unit_test_util.h"
#include "decode_test_util.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>

namespace {

using namespace rocjitsu;

// Restore the test override even when an assertion leaves the current scope.
struct ScalarGuard {
  bool previous = util::force_scalar();
  explicit ScalarGuard(bool scalar) { util::set_force_scalar_for_testing(scalar); }
  ~ScalarGuard() { util::set_force_scalar_for_testing(previous); }
};

// Sparse lanes and overlapping destinations share the same compact golden inputs.
TEST(OlderGraphics, Mullit) {
  struct Case {
    std::array<uint32_t, 3> input;
    uint32_t output;
  };
  constexpr Case cases[] = {
      {{0x40000000, 0x40400000, 0x3f800000}, 0x40c00000}, // 2 * 3
      {{0x40000000, 0x40400000, 0xbf800000}, 0xff7fffff}, // Nonpositive predicate.
      {{0, 0x7fc00123, 0x3f800000}, 0xff7fffff},          // Fixup precedes zero product.
      {{0x80000000, 0x40400000, 0x3f800000}, 0},          // Legacy +0 product.
      {{0x7f800123, 0x3f800000, 0x3f800000}, 0x7fc00123},
      {{0x00000001, 0x4b000000, 0x3f800000}, 0x00800000}, // Preserved tiny source.
  };
  for (auto arch :
       {rocjitsu::IsaFamily::Rdna1, rocjitsu::IsaFamily::Rdna2, rocjitsu::IsaFamily::Rdna3_5}) {
    for (bool scalar : {false, true}) {
      ScalarGuard guard(scalar);
      for (auto semantics :
           {amdgpu::ArithmeticSemantics::Compatibility, amdgpu::ArithmeticSemantics::Loose}) {
        amdgpu::GpuMemory memory("mullit_memory");
        amdgpu::L2Cache l2("mullit_l2");
        amdgpu::ComputeUnitCore::Config cfg{};
        cfg.arch = arch;
        cfg.num_wf_slots = 1;
        cfg.sgprs_per_wf = 106;
        cfg.vgprs_per_wf = 16;
        cfg.lds_size_kb = 64;
        cfg.arithmetic.semantics = semantics;
        std::unique_ptr<amdgpu::ComputeUnitCore> cu;
        ASSERT_NO_FATAL_FAILURE(test::initialize_compute_unit(cu, "mullit_cu", cfg, &memory, &l2));
        auto decoder = Decoder::create(arch);
        auto *wave = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
        ASSERT_NE(wave, nullptr);
        const uint32_t base = wave->vgpr_alloc().base;
        // CLAMP preserves the NaN unless DX10_CLAMP requests conversion to zero.
        if (semantics == amdgpu::ArithmeticSemantics::Compatibility) {
          const uint32_t opcode = arch == rocjitsu::IsaFamily::Rdna3_5 ? 536 : 336;
          const uint32_t words[] = {0xd4008000U | (opcode << 16) | 3U,
                                    256U | (257U << 9) | (258U << 18)};
          std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words));
          ASSERT_NE(instruction, nullptr);
          for (uint32_t mode : {0x0f0U, 0x1f0U}) {
            wave->set_mode_raw(mode);
            wave->set_exec(1);
            cu->write_vgpr(base, 0, 0x7fc00000U);
            cu->write_vgpr(base + 1, 0, 0x3f800000U);
            cu->write_vgpr(base + 2, 0, 0x3f800000U);
            ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
            EXPECT_EQ(cu->read_vgpr(base + 3, 0), mode == 0x0f0U ? 0x7fc00000U : 0U);
          }
        }
        for (int aliasing_offset : {0, 1, 2, 3}) {
          const uint32_t opcode = arch == rocjitsu::IsaFamily::Rdna3_5 ? 536 : 336;
          const uint32_t words[] = {0xd4000000U | (opcode << 16) | uint32_t(aliasing_offset),
                                    256U | (257U << 9) | (258U << 18)};
          std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words));
          ASSERT_NE(instruction, nullptr);
          for (uint64_t exec_mask : {0ULL, 0x80000001ULL, 0xffffffffULL}) {
            wave->set_exec(exec_mask);
            wave->set_mode_raw(0x2f0); // IEEE and both denormal directions preserved.
            for (int lane = 0; lane < int(wave->wf_size()); ++lane) {
              for (int source = 0; source < 4; ++source) {
                cu->write_vgpr(base + source, lane,
                               source < 3 ? cases[lane % std::size(cases)].input[source]
                                          : 0x12345678);
              }
            }
            ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
            for (int lane = 0; lane < int(wave->wf_size()); ++lane) {
              const auto &test = cases[lane % std::size(cases)];
              for (int reg = 0; reg < 4; ++reg) {
                uint32_t expected = reg < 3 ? test.input[reg] : 0x12345678;
                if (reg == aliasing_offset && (exec_mask & (1ULL << lane))) {
                  expected = test.output;
                }
                EXPECT_EQ(cu->read_vgpr(base + reg, lane), expected)
                    << "arch=" << arch << " lane=" << lane << " reg=" << reg;
              }
            }
          }
        }
        wave->halt();
      }
    }
  }
}

} // namespace
