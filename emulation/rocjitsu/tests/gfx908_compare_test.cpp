// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded gfx908/gfx942 comparisons: selected source words, signaling constants,
// architectural mask writes, and rejection of unsupported wave sizes.

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace rocjitsu {

// The selected NaNs affect only signaling Compatibility status, including F/TRU.
// Inactive lanes and unselected halfwords cannot contribute INVALID.
TEST(CdnaComparisonTest, Gfx9SignalingConstantsAndMasks) {
  struct Format {
    int bits;
    uint16_t equal, never, always, cmpx;
    uint64_t one, quiet, signaling;
  };
  constexpr Format gfx908_formats[] = {
      {16, cdna1::kVCmpEqF16Vop3, cdna1::kVCmpFF16Vop3, cdna1::kVCmpTruF16Vop3,
       cdna1::kVCmpxEqF16Vop3, 0x3c00, 0x7e01, 0x7c01},
      {32, cdna1::kVCmpEqF32Vop3, cdna1::kVCmpFF32Vop3, cdna1::kVCmpTruF32Vop3,
       cdna1::kVCmpxEqF32Vop3, 0x3f800000, 0x7fc00001, 0x7f800001},
      {64, cdna1::kVCmpEqF64Vop3, cdna1::kVCmpFF64Vop3, cdna1::kVCmpTruF64Vop3,
       cdna1::kVCmpxEqF64Vop3, 0x3ff0000000000000ULL, 0x7ff8000000000001ULL, 0x7ff0000000000001ULL},
  };
  constexpr Format gfx942_formats[] = {
      {16, cdna3::kVCmpEqF16Vop3, cdna3::kVCmpFF16Vop3, cdna3::kVCmpTruF16Vop3,
       cdna3::kVCmpxEqF16Vop3, 0x3c00, 0x7e01, 0x7c01},
      {32, cdna3::kVCmpEqF32Vop3, cdna3::kVCmpFF32Vop3, cdna3::kVCmpTruF32Vop3,
       cdna3::kVCmpxEqF32Vop3, 0x3f800000, 0x7fc00001, 0x7f800001},
      {64, cdna3::kVCmpEqF64Vop3, cdna3::kVCmpFF64Vop3, cdna3::kVCmpTruF64Vop3,
       cdna3::kVCmpxEqF64Vop3, 0x3ff0000000000000ULL, 0x7ff8000000000001ULL, 0x7ff0000000000001ULL},
  };
  for (auto arch : {rocjitsu::IsaFamily::Cdna1, rocjitsu::IsaFamily::Cdna3}) {
    const auto &formats = arch == rocjitsu::IsaFamily::Cdna1 ? gfx908_formats : gfx942_formats;
    const auto encode = [&](uint16_t opcode, uint8_t half, bool clamp) {
      if (arch == rocjitsu::IsaFamily::Cdna1) {
        return cdna1::build_vop3(
            opcode, {.vdst = 0, .op_sel = half, .clamp = uint8_t(clamp), .src0 = 256, .src1 = 258});
      }
      return cdna3::build_vop3(
          opcode, {.vdst = 0, .op_sel = half, .clamp = uint8_t(clamp), .src0 = 256, .src1 = 258});
    };
    for (auto semantics :
         {amdgpu::ArithmeticSemantics::Compatibility, amdgpu::ArithmeticSemantics::Loose}) {
      amdgpu::GpuMemory memory("gfx908_compare");
      amdgpu::L2Cache l2("gfx908_compare");
      amdgpu::ComputeUnitCore::Config config{};
      config.arch = arch;
      config.arithmetic.semantics = semantics;
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 104;
      config.vgprs_per_wf = 8;
      auto cu = amdgpu::ComputeUnitCore::create("gfx908_compare", config, &memory, &l2);
      auto decoder = Decoder::create(config.arch);
      ASSERT_NE(cu, nullptr);
      ASSERT_NE(decoder, nullptr);
      auto *wave = cu->dispatch_wf(0, 0, 104, 8);
      ASSERT_NE(wave, nullptr);
      const auto base = wave->vgpr_alloc().base;
      const auto scalar = wave->sgpr_alloc().base;
      for (const auto &f : formats) {
        SCOPED_TRACE(f.bits);
        for (int lane = 0; lane < 64; ++lane) {
          const uint64_t a = lane == 1 ? f.signaling : lane == 63 ? f.quiet : f.one;
          cu->write_vgpr(base, lane, f.bits == 16 ? uint32_t(a << 16) | 0x7e01 : uint32_t(a));
          cu->write_vgpr(base + 1, lane, uint32_t(a >> 32));
          cu->write_vgpr(base + 2, lane, f.bits == 16 ? uint32_t(f.one << 16) : uint32_t(f.one));
          cu->write_vgpr(base + 3, lane, uint32_t(f.one >> 32));
        }
        for (uint16_t opcode : {f.equal, f.never, f.always, f.cmpx}) {
          for (bool clamp : {false, true}) {
            for (uint64_t exec_mask : {~0ULL, 1ULL, 2ULL, 1ULL << 63, 0ULL}) {
              wave->set_mode_raw(0xf0);
              wave->set_exec(exec_mask);
              wave->set_trapsts(0x100);
              wave->set_vcc_raw(~0ULL);
              cu->write_sgpr(scalar, 0xdeadbeef);
              cu->write_sgpr(scalar + 1, 0xa5a5a5a5);
              const auto words = encode(opcode, uint8_t(f.bits == 16 ? 3 : 0), clamp);
              auto instruction = decoder->decode_window(words);
              ASSERT_TRUE(instruction.succeeded());
              ASSERT_TRUE(cu->execute_instruction(instruction.value().get(), *wave).succeeded());
              const uint64_t expected = opcode == f.never    ? 0
                                        : opcode == f.always ? exec_mask
                                                             : exec_mask & ~(2ULL | (1ULL << 63));
              EXPECT_EQ(uint64_t(cu->read_sgpr(scalar)) |
                            (uint64_t(cu->read_sgpr(scalar + 1)) << 32),
                        expected);
              EXPECT_EQ(wave->exec(), opcode == f.cmpx ? expected : exec_mask);
              EXPECT_EQ(wave->vcc(), ~0ULL);
              const bool invalid = clamp &&
                                   semantics == amdgpu::ArithmeticSemantics::Compatibility &&
                                   (exec_mask & (2ULL | (1ULL << 63)));
              EXPECT_EQ(wave->trapsts(), invalid ? 0x101U : 0x100U);
            }
          }
        }
      }
      wave->halt();
      // Unsupported Wave32 test slots fail before the omitted callback is used.
      wave = cu->dispatch_wf(0, 0, 104, 8, 32);
      ASSERT_NE(wave, nullptr);
      wave->set_exec(1);
      wave->set_trapsts(0x100);
      cu->write_sgpr(wave->sgpr_alloc().base, 0xdeadbeef);
      const auto words = encode(formats[1].equal, 0, true);
      auto instruction = decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      EXPECT_TRUE(cu->execute_instruction(instruction.value().get(), *wave).failed());
      EXPECT_EQ(wave->exec(), 1U);
      EXPECT_EQ(wave->trapsts(), 0x100U);
      EXPECT_EQ(cu->read_sgpr(wave->sgpr_alloc().base), 0xdeadbeefU);
      wave->halt();
    }
  }
}

} // namespace rocjitsu
