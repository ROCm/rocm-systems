// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Gfx1150 floating comparisons retain DX10_CLAMP-gated INVALID, including
// constant predicates, selected FP16 halves, sparse EXEC and CMPX's EXEC result.
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <array>
#include <gtest/gtest.h>
#include <stdint.h>

namespace rocjitsu {

// Literal masks and causes distinguish sNaN from qNaN, and status gating from CLAMP.
TEST(Gfx1150ComparisonTest, MasksAndSignaling) {
  struct Format {
    int bits;
    uint16_t equal, never, always, cmpx;
    uint64_t one, quiet, signaling;
  };
  const Format formats[] = {
      {16, rdna3_5::kVCmpEqF16Vop3, rdna3_5::kVCmpFF16Vop3, rdna3_5::kVCmpTF16Vop3,
       rdna3_5::kVCmpxEqF16Vop3, 0x3c003c00, 0x7e013c00, 0x7c013c00},
      {32, rdna3_5::kVCmpEqF32Vop3, rdna3_5::kVCmpFF32Vop3, rdna3_5::kVCmpTF32Vop3,
       rdna3_5::kVCmpxEqF32Vop3, 0x3f800000, 0x7fc00001, 0x7f800001},
      {64, rdna3_5::kVCmpEqF64Vop3, rdna3_5::kVCmpFF64Vop3, rdna3_5::kVCmpTF64Vop3,
       rdna3_5::kVCmpxEqF64Vop3, 0x3ff0000000000000ULL, 0x7ff8000000000001ULL,
       0x7ff0000000000001ULL},
  };
  for (int lanes : {32, 64}) {
    for (auto semantics :
         {amdgpu::ArithmeticSemantics::Compatibility, amdgpu::ArithmeticSemantics::Loose}) {
      amdgpu::GpuMemory memory("gfx1150_compare");
      amdgpu::L2Cache l2("gfx1150_compare");
      amdgpu::ComputeUnitCore::Config config{};
      config.arch = rocjitsu::IsaFamily::Rdna3_5;
      config.arithmetic.semantics = semantics;
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 104;
      config.vgprs_per_wf = 8;
      auto cu = amdgpu::ComputeUnitCore::create("gfx1150_compare", config, &memory, &l2);
      auto decoder = Decoder::create(config.arch);
      ASSERT_NE(cu, nullptr);
      ASSERT_NE(decoder, nullptr);
      auto *wave = cu->dispatch_wf(0, 0, 104, 8, lanes);
      ASSERT_NE(wave, nullptr);
      const auto base = wave->vgpr_alloc().base, scalar = wave->sgpr_alloc().base;
      const uint64_t full_exec_mask = lanes == 32 ? 0xffffffffULL : ~0ULL;
      for (const auto &f : formats) {
        SCOPED_TRACE(::testing::Message() << lanes << '/' << f.bits);
        for (int lane = 0; lane < lanes; ++lane) {
          const uint64_t value = lane == 0 ? f.signaling : lane == 1 ? f.quiet : f.one;
          cu->write_vgpr(base, lane, uint32_t(value));
          cu->write_vgpr(base + 1, lane, uint32_t(value >> 32));
          cu->write_vgpr(base + 2, lane, uint32_t(f.one));
          cu->write_vgpr(base + 3, lane, uint32_t(f.one >> 32));
        }
        for (bool dx10 : {false, true}) {
          for (bool clamp : {false, true}) {
            for (uint16_t opcode : {f.equal, f.never, f.always, f.cmpx}) {
              for (uint64_t exec_mask :
                   std::array<uint64_t, 5>{full_exec_mask, 1, 2, 1ULL << (lanes - 1), 0}) {
                wave->set_mode_raw(0xf0 | (dx10 ? 0x100 : 0));
                wave->set_exec(exec_mask);
                wave->set_trapsts(0x100);
                wave->set_vcc_raw(full_exec_mask);
                wave->write_scc(true);
                cu->write_sgpr(scalar, 0xdeadbeef);
                cu->write_sgpr(scalar + 1, 0xa5a5a5a5);
                auto words =
                    rdna3_5::build_vop3(opcode, {.vdst = uint8_t(opcode == f.cmpx ? 126 : 0),
                                                 .op_sel = uint8_t(f.bits == 16 ? 3 : 0),
                                                 .clamp = uint8_t(clamp),
                                                 .src0 = 256,
                                                 .src1 = 258});
                auto instruction = decoder->decode_window(words);
                ASSERT_TRUE(instruction.succeeded()) << opcode;
                ASSERT_TRUE(cu->execute_instruction(instruction.value().get(), *wave).succeeded());
                const uint64_t expected = opcode == f.never    ? 0
                                          : opcode == f.always ? exec_mask
                                                               : exec_mask & ~3ULL;
                EXPECT_EQ(cu->read_sgpr(scalar),
                          opcode == f.cmpx ? 0xdeadbeefU : uint32_t(expected));
                EXPECT_EQ(cu->read_sgpr(scalar + 1),
                          opcode == f.cmpx || lanes == 32 ? 0xa5a5a5a5U : uint32_t(expected >> 32));
                EXPECT_EQ(wave->exec(), opcode == f.cmpx ? expected : exec_mask);
                EXPECT_EQ(wave->vcc(), full_exec_mask);
                EXPECT_TRUE(wave->read_scc());
                const bool invalid = semantics == amdgpu::ArithmeticSemantics::Compatibility &&
                                     !dx10 && ((exec_mask & 1) || (clamp && (exec_mask & 2)));
                EXPECT_EQ(wave->trapsts(), invalid ? 0x101U : 0x100U);
              }
            }
          }
        }
      }
      wave->halt();
    }
  }
}

} // namespace rocjitsu
