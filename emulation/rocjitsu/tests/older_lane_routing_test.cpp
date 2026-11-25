// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded older Radeon permutations: cross-half routing, inactive fetches and aliased output.
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/execution_backend.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <array>
#include <cstdint>
#include <gtest/gtest.h>

namespace rocjitsu::amdgpu {
namespace {
// Identity nibble selectors fetch from the opposite 16-lane half; FI overrides BOUND_CTRL.
TEST(OlderLaneRouting, CrossHalfAliasingAndMasks) {
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    std::array<uint32_t, 2> (*encode)(uint8_t);
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Rdna1, &rdna1::execution_backend(),
       [](uint8_t controls) {
         return rdna1::build_vop3(
             rdna1::kVPermlanex16B32Vop3,
             {.vdst = 0, .op_sel = controls, .src0 = 256, .src1 = 0, .src2 = 1});
       }},
      {rocjitsu::IsaFamily::Rdna2, &rdna2::execution_backend(),
       [](uint8_t controls) {
         return rdna2::build_vop3(
             rdna2::kVPermlanex16B32Vop3,
             {.vdst = 0, .op_sel = controls, .src0 = 256, .src1 = 0, .src2 = 1});
       }},
      {rocjitsu::IsaFamily::Rdna3_5, &rdna3_5::execution_backend(),
       [](uint8_t controls) {
         return rdna3_5::build_vop3(
             rdna3_5::kVPermlanex16B32Vop3,
             {.vdst = 0, .op_sel = controls, .src0 = 256, .src1 = 0, .src2 = 1});
       }},
  };
  for (const auto &family : families) {
    ScopedIsaExecutionBackend backend(family.backend);
    for (unsigned lanes : {32U, 64U}) {
      for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility}) {
        ComputeUnitCore::Config config{};
        config.arch = family.arch;
        config.arithmetic = {semantics};
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 104;
        config.vgprs_per_wf = 16;
        config.lds_size_kb = 64;
        GpuMemory memory("older_lane_memory");
        L2Cache l2("older_lane_l2");
        auto cu = ComputeUnitCore::create("older_lane", config, &memory, &l2);
        auto *wave = cu->dispatch_wf(0, 0, 104, 16, lanes);
        ASSERT_NE(wave, nullptr);
        auto decoder = Decoder::create(family.arch);
        const auto vb = wave->vgpr_alloc().base, sb = wave->sgpr_alloc().base;
        cu->write_sgpr(sb, 0x76543210U);
        cu->write_sgpr(sb + 1, 0xfedcba98U);
        for (uint64_t exec_mask : {~0ULL, 0x0000ffff0000ffffULL, 0ULL}) {
          exec_mask &= lanes == 64 ? ~0ULL : 0xffffffffULL;
          for (uint8_t controls : {0, 1, 2, 3}) {
            SCOPED_TRACE(family.arch);
            SCOPED_TRACE(lanes);
            SCOPED_TRACE(controls);
            constexpr uint64_t old_vcc = 0x55aac0de12345678ULL;
            wave->set_exec(exec_mask);
            wave->set_vcc_raw(old_vcc);
            wave->write_scc(false);
            for (unsigned lane = 0; lane < lanes; ++lane) {
              cu->write_vgpr(vb, lane, 0x3f000001U + lane);
            }
            auto decoded = decoder->decode_window(family.encode(controls));
            ASSERT_TRUE(decoded.succeeded());
            ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
            for (unsigned lane = 0; lane < lanes; ++lane) {
              const unsigned source = lane ^ 16;
              uint32_t expected = 0x3f000001U + lane;
              if ((exec_mask >> lane) & 1) {
                if (((exec_mask >> source) & 1) || (controls & 1)) {
                  expected = 0x3f000001U + source;
                } else if (controls & 2) {
                  expected = 0;
                }
              }
              EXPECT_EQ(cu->read_vgpr(vb, lane), expected) << lane;
            }
            EXPECT_EQ(wave->exec(), exec_mask);
            EXPECT_EQ(wave->vcc(), old_vcc);
            EXPECT_FALSE(wave->read_scc());
          }
        }
      }
    }
  }
}
// The packed-storage route and logical fallback preserve the same two halfword results.
TEST(OlderLaneRouting, PackedHalfSwapsPreserveAliasesAndMasks) {
  ScopedIsaExecutionBackend backend(&rdna3_5::execution_backend());
  for (unsigned lanes : {32U, 64U}) {
    ComputeUnitCore::Config config{};
    config.arch = rocjitsu::IsaFamily::Rdna3_5;
    config.arithmetic = {ArithmeticSemantics::Loose};
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 8;
    config.lds_size_kb = 64;
    GpuMemory memory("packed_swap_memory");
    L2Cache l2("packed_swap_l2");
    auto cu = ComputeUnitCore::create("packed_swap", config, &memory, &l2);
    auto *wave = cu->dispatch_wf(0, 0, 104, 8, lanes);
    ASSERT_NE(wave, nullptr);
    auto decoder = Decoder::create(config.arch);
    const auto base = wave->vgpr_alloc().base;
    const auto value = [](unsigned reg, unsigned lane) {
      return 0x1234abcdU + reg * 0x11110000U + lane;
    };
    for (bool high : {false, true}) {
      for (bool alias : {false, true}) {
        for (bool suppressed : {false, true}) {
          const unsigned source = alias ? 0 : 2, ds = high ? 16 : 0, as = high ? 0 : 16;
          const uint64_t exec_mask = 0x5555555555555555ULL & (lanes == 32 ? 0xffffffffULL : ~0ULL);
          const uint64_t write_mask = exec_mask & (suppressed ? ~4ULL : ~0ULL);
          wave->set_exec(exec_mask);
          wave->set_vgpr_write_mask(suppressed ? ~4ULL : ~0ULL);
          for (unsigned reg : {0U, 2U}) {
            for (unsigned lane = 0; lane < lanes; ++lane) {
              cu->write_vgpr(base + reg, lane, value(reg, lane));
            }
          }
          auto words = rdna3_5::build_vop1(
              rdna3_5::kVSwapB16Vop1,
              {.src0 = uint16_t(256 + source + 128 * !high), .vdst = uint8_t(128 * high)});
          auto decoded = decoder->decode_window(words);
          ASSERT_TRUE(decoded.succeeded());
          const auto diagnostics = cu->isa_diagnostic_count();
          ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
          EXPECT_EQ(cu->isa_diagnostic_count(), diagnostics);
          for (unsigned reg : {0U, 2U}) {
            for (unsigned lane = 0; lane < lanes; ++lane) {
              uint32_t expected = value(reg, lane);
              if ((write_mask >> lane) & 1) {
                if (reg == 0) {
                  expected = (expected & ~(0xffffU << ds)) |
                             (((value(source, lane) >> as) & 0xffffU) << ds);
                }
                if (reg == source) {
                  expected =
                      (expected & ~(0xffffU << as)) | (((value(0, lane) >> ds) & 0xffffU) << as);
                }
              }
              EXPECT_EQ(cu->read_vgpr(base + reg, lane), expected) << reg << ',' << lane;
            }
          }
        }
      }
    }
  }
}
} // namespace
} // namespace rocjitsu::amdgpu
