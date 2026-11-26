// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Gfx1010/gfx1030 native-precision masks, signaling constants and scalar aliases.
// CMPX updates EXEC without introducing the GFX9 scalar-mask side effect.

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
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

// Literal mask/status expectations distinguish signaling NaNs, F/TRU reads,
// Wave32's untouched neighboring SGPR, and CMPX's EXEC-only result.
TEST(Gfx10ComparisonTest, NativeMasksAndSignaling) {
  struct Format {
    int bits;
    uint16_t equal, never, always, cmpx;
    uint64_t one, quiet, signaling;
  };
  for (auto arch : {rocjitsu::IsaFamily::Rdna1, rocjitsu::IsaFamily::Rdna2}) {
    const bool gfx1030 = arch == rocjitsu::IsaFamily::Rdna2;
    const Format formats[] = {
        {32, gfx1030 ? rdna2::kVCmpEqF32Vop3 : rdna1::kVCmpEqF32Vop3,
         gfx1030 ? rdna2::kVCmpFF32Vop3 : rdna1::kVCmpFF32Vop3,
         gfx1030 ? rdna2::kVCmpTruF32Vop3 : rdna1::kVCmpTruF32Vop3,
         gfx1030 ? rdna2::kVCmpxEqF32Vop3 : rdna1::kVCmpxEqF32Vop3, 0x3f800000, 0x7fc00001,
         0x7f800001},
        {64, gfx1030 ? rdna2::kVCmpEqF64Vop3 : rdna1::kVCmpEqF64Vop3,
         gfx1030 ? rdna2::kVCmpFF64Vop3 : rdna1::kVCmpFF64Vop3,
         gfx1030 ? rdna2::kVCmpTruF64Vop3 : rdna1::kVCmpTruF64Vop3,
         gfx1030 ? rdna2::kVCmpxEqF64Vop3 : rdna1::kVCmpxEqF64Vop3, 0x3ff0000000000000ULL,
         0x7ff8000000000001ULL, 0x7ff0000000000001ULL},
    };
    for (int lanes : {32, 64}) {
      for (auto semantics :
           {amdgpu::ArithmeticSemantics::Compatibility, amdgpu::ArithmeticSemantics::Loose}) {
        amdgpu::GpuMemory memory("gfx10_compare");
        amdgpu::L2Cache l2("gfx10_compare");
        amdgpu::ComputeUnitCore::Config config{};
        config.arch = arch;
        config.arithmetic.semantics = semantics;
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 104;
        config.vgprs_per_wf = 8;
        auto cu = amdgpu::ComputeUnitCore::create("gfx10_compare", config, &memory, &l2);
        auto decoder = Decoder::create(arch);
        ASSERT_NE(cu, nullptr);
        ASSERT_NE(decoder, nullptr);
        auto *wave = cu->dispatch_wf(0, 0, 104, 8, lanes);
        ASSERT_NE(wave, nullptr);
        const auto base = wave->vgpr_alloc().base;
        const auto scalar = wave->sgpr_alloc().base;
        // The fields have the same meaning but remain bound to each real decoder.
        auto words = [&](uint16_t opcode, bool clamp, uint16_t source_b = 258,
                         uint8_t destination = 0) {
          return gfx1030 ? rdna2::build_vop3(opcode, {.vdst = destination,
                                                      .clamp = uint8_t(clamp),
                                                      .src0 = 256,
                                                      .src1 = source_b})
                         : rdna1::build_vop3(opcode, {.vdst = destination,
                                                      .clamp = uint8_t(clamp),
                                                      .src0 = 256,
                                                      .src1 = source_b});
        };
        const uint64_t full_exec_mask = lanes == 32 ? 0xffffffffULL : ~0ULL;
        const uint64_t nan_lanes = 2ULL | (1ULL << (lanes - 1));
        for (const auto &f : formats) {
          SCOPED_TRACE(::testing::Message() << arch << '/' << lanes << '/' << f.bits);
          for (int lane = 0; lane < lanes; ++lane) {
            const uint64_t a = lane == 1 ? f.signaling : lane == lanes - 1 ? f.quiet : f.one;
            cu->write_vgpr(base, lane, uint32_t(a));
            cu->write_vgpr(base + 1, lane, uint32_t(a >> 32));
            cu->write_vgpr(base + 2, lane, uint32_t(f.one));
            cu->write_vgpr(base + 3, lane, uint32_t(f.one >> 32));
          }
          for (uint16_t opcode : {f.equal, f.never, f.always, f.cmpx}) {
            for (bool clamp : {false, true}) {
              for (uint64_t exec_mask :
                   std::array<uint64_t, 5>{full_exec_mask, 1, 2, 1ULL << (lanes - 1), 0}) {
                wave->set_mode_raw(0xf0);
                wave->set_exec(exec_mask);
                wave->set_trapsts(0x100);
                wave->set_vcc_raw(full_exec_mask);
                wave->write_scc(true);
                cu->write_sgpr(scalar, 0xdeadbeef);
                cu->write_sgpr(scalar + 1, 0xa5a5a5a5);
                // CMPX's destination field must encode EXEC rather than an SGPR.
                auto instruction =
                    decoder->decode_window(words(opcode, clamp, 258, opcode == f.cmpx ? 126 : 0));
                ASSERT_TRUE(instruction.succeeded()) << opcode << "/clamp=" << clamp;
                ASSERT_TRUE(cu->execute_instruction(instruction.value().get(), *wave).succeeded());
                const uint64_t expected = opcode == f.never    ? 0
                                          : opcode == f.always ? exec_mask
                                                               : exec_mask & ~nan_lanes;
                EXPECT_EQ(cu->read_sgpr(scalar),
                          opcode == f.cmpx ? 0xdeadbeefU : uint32_t(expected));
                EXPECT_EQ(cu->read_sgpr(scalar + 1),
                          opcode == f.cmpx || lanes == 32 ? 0xa5a5a5a5U : uint32_t(expected >> 32));
                EXPECT_EQ(wave->exec(), opcode == f.cmpx ? expected : exec_mask);
                EXPECT_EQ(wave->vcc(), full_exec_mask);
                EXPECT_TRUE(wave->read_scc());
                const bool invalid = clamp &&
                                     semantics == amdgpu::ArithmeticSemantics::Compatibility &&
                                     (exec_mask & nan_lanes);
                EXPECT_EQ(wave->trapsts(), invalid ? 0x101U : 0x100U);
              }
            }
          }
          // Snapshot the uniform source before replacing its SGPR pair with D.
          wave->set_exec(1ULL | (1ULL << (lanes - 1)));
          cu->write_sgpr(scalar, uint32_t(f.one));
          cu->write_sgpr(scalar + 1, uint32_t(f.one >> 32));
          auto alias = decoder->decode_window(words(f.equal, true, 0));
          ASSERT_TRUE(alias.succeeded());
          ASSERT_TRUE(cu->execute_instruction(alias.value().get(), *wave).succeeded());
          EXPECT_EQ(cu->read_sgpr(scalar), 1U);
          EXPECT_EQ(cu->read_sgpr(scalar + 1), lanes == 32 ? uint32_t(f.one >> 32) : 0U);
        }
        wave->halt();
      }
    }
  }
}

} // namespace rocjitsu
