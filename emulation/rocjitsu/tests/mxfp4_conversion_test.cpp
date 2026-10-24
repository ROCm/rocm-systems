// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>

#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace {

using namespace rocjitsu;

// Keep the former double-scale/divide/F32-rounding algorithm as the oracle.
RJ_NOINLINE uint8_t reference_conversion(uint16_t input, uint8_t scale_exp) {
  const double scale = std::ldexp(1.0, static_cast<int>(scale_exp) - 127);
  const float scaled = static_cast<float>(static_cast<double>(util::bf16_to_f32(input)) / scale);
  return util::f32_to_fp4_e2m1_rne(scaled);
}

TEST(Mxfp4Conversion, AllBf16ValuesAndValidScaleExponentsMatchReference) {
  for (uint32_t scale_exp = 0; scale_exp < 255; ++scale_exp)
    for (uint32_t input = 0; input < 65536; ++input) {
      const auto expected = reference_conversion(input, scale_exp);
      const auto actual = util::bf16_to_fp4_e2m1_scaled_rne(input, scale_exp);
      ASSERT_EQ(actual, expected) << "input=" << input << " scale=" << scale_exp;
    }
}

TEST(Mxfp4Conversion, HostRoundingAndDenormalModesMatchLegacyReference) {
#if defined(__SSE2__)
  struct HostMxcsrGuard {
    uint32_t saved = _mm_getcsr();
    ~HostMxcsrGuard() { _mm_setcsr(saved); }
  } guard;
  // BF16 midpoint neighbors, input underflow boundaries, signed zero, NaN,
  // infinity, and extreme normal inputs. Every valid scale is exercised.
  constexpr std::array<uint16_t, 36> magnitudes = {
      0x0000, 0x0001, 0x0040, 0x007f, 0x0080, 0x0081, 0x3e7f, 0x3e80, 0x3e81,
      0x3f3f, 0x3f40, 0x3f41, 0x3f9f, 0x3fa0, 0x3fa1, 0x3fdf, 0x3fe0, 0x3fe1,
      0x401f, 0x4020, 0x4021, 0x405f, 0x4060, 0x4061, 0x409f, 0x40a0, 0x40a1,
      0x7f00, 0x7f7e, 0x7f7f, 0x7f80, 0x7f81, 0x7f9a, 0x7fc0, 0x7ffe, 0x7fff};
  for (uint32_t mode = 0; mode < 16; ++mode) {
    const uint32_t rounding = (mode & 3u) << 13;
    const uint32_t ftz = (mode & 4u) ? 0x8000u : 0u;
    const uint32_t daz = (mode & 8u) ? 0x40u : 0u;
    _mm_setcsr((guard.saved & ~0xe040u) | rounding | ftz | daz);
    for (uint32_t sign : {0u, 0x8000u})
      for (uint16_t magnitude : magnitudes)
        for (uint32_t scale_exp = 0; scale_exp < 255; ++scale_exp) {
          const uint16_t input = magnitude | sign;
          const auto expected = reference_conversion(input, scale_exp);
          const auto actual = util::bf16_to_fp4_e2m1_scaled_rne(input, scale_exp);
          ASSERT_EQ(actual, expected)
              << "mode=" << mode << " input=" << input << " scale=" << scale_exp;
        }
  }
#else
  GTEST_SKIP() << "host MXCSR modes require x86 SSE2";
#endif
}

TEST(Mxfp4Conversion, PackedInstructionPreservesBytesExecAndAliasedSources) {
  amdgpu::GpuMemory gpu_mem("mxfp4_conversion_mem");
  amdgpu::L2Cache l2("mxfp4_conversion_l2");
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_CDNA4;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 256;
  config.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("mxfp4_conversion", config, &gpu_mem, &l2);
  auto *wf = cu->dispatch_wf(0, 0, config.sgprs_per_wf, config.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  const uint32_t base = wf->vgpr_alloc().base;
  auto decoder = Decoder::create(config.arch);
  constexpr std::array<uint16_t, 16> inputs = {0x0000, 0x8000, 0x0040, 0x807f, 0x0080, 0x8080,
                                               0x3e80, 0xbf40, 0x3fa0, 0xbfe0, 0x4020, 0xc060,
                                               0x40a0, 0xff80, 0x7f81, 0xffc0};
  constexpr std::array<uint8_t, 8> exponents = {0, 1, 126, 127, 128, 253, 254, 255};
  for (uint32_t destination : {0u, 1u, 2u})
    for (uint32_t byte = 0; byte < 4; ++byte)
      for (uint64_t exec : {~uint64_t{0}, uint64_t{0xa55aa55aa55aa55a}, uint64_t{0}}) {
        SCOPED_TRACE(testing::Message()
                     << "destination=" << destination << " byte=" << byte << " exec=" << exec);
        wf->set_exec(exec);
        std::array<uint32_t, 64> expected{};
        for (uint32_t lane = 0; lane < 64; ++lane) {
          const uint32_t data = inputs[lane % inputs.size()] |
                                (uint32_t(inputs[(lane * 7 + 1) % inputs.size()]) << 16);
          // Sign and fraction bits in the scale operand do not affect E8M0.
          const uint32_t scale = (uint32_t(exponents[(lane / 2) % exponents.size()]) << 23) |
                                 ((lane & 1u) << 31) | (lane * 7919u);
          cu->write_vgpr(base, lane, data);
          cu->write_vgpr(base + 1, lane, scale);
          cu->write_vgpr(base + 2, lane, 0xa1b2c3d4u ^ lane);
          const uint32_t old = cu->read_vgpr(base + destination, lane);
          const uint32_t exponent = (scale >> 23) & 255u;
          expected[lane] = old;
          if ((exec & (uint64_t{1} << lane)) && exponent != 255) {
            const uint32_t packed = reference_conversion(data, exponent) |
                                    (uint32_t(reference_conversion(data >> 16, exponent)) << 4);
            expected[lane] = (old & ~(255u << (byte * 8))) | (packed << (byte * 8));
          }
        }
        const auto words = cdna4::build_vop3(cdna4::kVCvtScalef32PkFp4Bf16Vop3,
                                             {.vdst = static_cast<uint8_t>(destination),
                                              .op_sel = static_cast<uint8_t>(byte << 2),
                                              .src0 = 256,
                                              .src1 = 257});
        std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
        ASSERT_NE(inst, nullptr);
        ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
        EXPECT_EQ(wf->exec(), exec);
        for (uint32_t lane = 0; lane < 64; ++lane)
          EXPECT_EQ(cu->read_vgpr(base + destination, lane), expected[lane]) << "lane=" << lane;
      }
}

} // namespace
