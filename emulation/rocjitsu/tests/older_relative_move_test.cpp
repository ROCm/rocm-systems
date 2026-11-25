// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Older Radeon M0 moves retain effective-register bounds, masks and overlapping operands.
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
#include <gtest/gtest.h>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {
// Each operation resolves M0 before copying; invalid reads use v0, invalid writes do nothing.
TEST(OlderRelativeMove, AddressingMasksAndAliasing) {
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    std::vector<uint32_t> (*encode)(int, uint8_t, uint16_t);
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Rdna1, &rdna1::execution_backend(),
       [](int operation, uint8_t dst, uint16_t src) {
         const uint16_t opcodes[] = {rdna1::kVMovrelsB32Vop1, rdna1::kVMovreldB32Vop1,
                                     rdna1::kVMovrelsdB32Vop1, rdna1::kVMovrelsd2B32Vop1,
                                     rdna1::kVSwaprelB32Vop1};
         // VOP3 MOVRELD retains this profile's legacy raw-bit treatment of sign controls.
         if (operation == 5) {
           auto words =
               rdna1::build_vop3(rdna1::kVMovreldB32Vop3, {.vdst = dst, .abs = 1, .src0 = src});
           return std::vector<uint32_t>(words.begin(), words.end());
         }
         auto words = rdna1::build_vop1(opcodes[operation], {.src0 = src, .vdst = dst});
         return std::vector<uint32_t>(words.begin(), words.end());
       }},
      {rocjitsu::IsaFamily::Rdna2, &rdna2::execution_backend(),
       [](int operation, uint8_t dst, uint16_t src) {
         const uint16_t opcodes[] = {rdna2::kVMovrelsB32Vop1, rdna2::kVMovreldB32Vop1,
                                     rdna2::kVMovrelsdB32Vop1, rdna2::kVMovrelsd2B32Vop1,
                                     rdna2::kVSwaprelB32Vop1};
         // VOP3 MOVRELD retains this profile's legacy raw-bit treatment of sign controls.
         if (operation == 5) {
           auto words =
               rdna2::build_vop3(rdna2::kVMovreldB32Vop3, {.vdst = dst, .abs = 1, .src0 = src});
           return std::vector<uint32_t>(words.begin(), words.end());
         }
         auto words = rdna2::build_vop1(opcodes[operation], {.src0 = src, .vdst = dst});
         return std::vector<uint32_t>(words.begin(), words.end());
       }},
      {rocjitsu::IsaFamily::Rdna3_5, &rdna3_5::execution_backend(),
       [](int operation, uint8_t dst, uint16_t src) {
         const uint16_t opcodes[] = {rdna3_5::kVMovrelsB32Vop1, rdna3_5::kVMovreldB32Vop1,
                                     rdna3_5::kVMovrelsdB32Vop1, rdna3_5::kVMovrelsd2B32Vop1,
                                     rdna3_5::kVSwaprelB32Vop1};
         // VOP3 MOVRELD retains this profile's legacy raw-bit treatment of sign controls.
         if (operation == 5) {
           auto words =
               rdna3_5::build_vop3(rdna3_5::kVMovreldB32Vop3, {.vdst = dst, .abs = 1, .src0 = src});
           return std::vector<uint32_t>(words.begin(), words.end());
         }
         auto words = rdna3_5::build_vop1(opcodes[operation], {.src0 = src, .vdst = dst});
         return std::vector<uint32_t>(words.begin(), words.end());
       }},
  };
  const auto value = [](unsigned reg, unsigned lane) { return 0x80010000U + reg * 0x100U + lane; };
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
        GpuMemory memory("relative_move_memory");
        L2Cache l2("relative_move_l2");
        auto cu = ComputeUnitCore::create("relative_move", config, &memory, &l2);
        auto *wave = cu->dispatch_wf(0, 0, 104, 16, lanes);
        ASSERT_NE(wave, nullptr);
        auto decoder = Decoder::create(family.arch);
        const auto vb = wave->vgpr_alloc().base;
        for (int operation = 0; operation < 6; ++operation) {
          for (uint32_t m0 : {0U, 3U, 0x40003U, 1023U}) {
            for (bool alias : {false, true}) {
              for (bool inactive : {false, true}) {
                SCOPED_TRACE(family.arch);
                SCOPED_TRACE(operation);
                SCOPED_TRACE(m0);
                const unsigned src_base = 1, dst_base = alias ? 1 : 8;
                const bool destination_only = operation == 1 || operation == 5;
                const unsigned src_offset = destination_only ? 0 : operation >= 3 ? m0 & 1023 : m0;
                const unsigned dst_offset = operation == 0                     ? 0
                                            : operation == 3 || operation == 4 ? (m0 >> 16) & 1023
                                                                               : m0;
                const bool src_valid = uint64_t(src_base) + src_offset < 16;
                const bool dst_valid = uint64_t(dst_base) + dst_offset < 16;
                const unsigned src = src_valid ? src_base + src_offset : 0,
                               dst = dst_valid ? dst_base + dst_offset : 0;
                const uint64_t exec_mask = (inactive ? 0ULL : 0xa55af00f0f0f00f3ULL) &
                                           (lanes == 32 ? 0xffffffffULL : ~0ULL);
                wave->set_exec(exec_mask);
                wave->set_vgpr_write_mask(~2ULL);
                wave->set_m0(m0);
                wave->set_vcc_raw(0x123456789abcdef0ULL);
                wave->write_scc(true);
                for (unsigned reg = 0; reg < 16; ++reg) {
                  for (unsigned lane = 0; lane < lanes; ++lane) {
                    cu->write_vgpr(vb + reg, lane, value(reg, lane));
                  }
                }
                auto decoded =
                    decoder->decode_window(family.encode(operation, dst_base, 256 + src_base));
                ASSERT_TRUE(decoded.succeeded());
                const auto diagnostics = cu->isa_diagnostic_count();
                ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
                EXPECT_EQ(cu->isa_diagnostic_count(), diagnostics);
                const bool writes = dst_valid && (operation != 4 || src_valid);
                for (unsigned reg = 0; reg < 16; ++reg) {
                  for (unsigned lane = 0; lane < lanes; ++lane) {
                    uint32_t expected = value(reg, lane);
                    if (writes && (((exec_mask & ~2ULL) >> lane) & 1)) {
                      if (reg == dst) {
                        expected = value(src, lane);
                      }
                      if (operation == 4 && reg == src) {
                        expected = value(dst, lane);
                      }
                    }
                    EXPECT_EQ(cu->read_vgpr(vb + reg, lane), expected) << reg << ',' << lane;
                  }
                }
                EXPECT_EQ(wave->exec(), exec_mask);
                EXPECT_EQ(wave->vcc(), 0x123456789abcdef0ULL);
                EXPECT_TRUE(wave->read_scc());
              }
            }
          }
        }
      }
    }
  }
}
} // namespace
} // namespace rocjitsu::amdgpu
