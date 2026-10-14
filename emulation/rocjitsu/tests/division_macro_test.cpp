// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"
#include "decode_test_util.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include <array>
#include <bit>
#include <cstdint>
#include <span>

namespace {
using namespace rocjitsu;
using namespace rocjitsu::test::cdna5;

class ForceScalarGuard {
public:
  ForceScalarGuard() : original_(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(original_); }

private:
  bool original_;
};

// Arithmetic instructions from hipcc -O2 -fno-gpu-flush-denormals-to-zero
// --offload-arch=gfx1250 for a / b. Inputs and outputs retain compiler registers.
constexpr std::array<uint32_t, 2> kMacroF32[] = {
    {0xd6fc7c03u, 0x04060502u}, // v_div_scale_f32 v3, null, v2, v2, v1
    {0x7e085503u, 0x00000000u}, // v_rcp_f32_e32 v4, v3
    {0xd6130005u, 0x23ca0903u}, // v_fma_f32 v5, -v3, v4, 1.0
    {0x56080905u, 0x00000000u}, // v_fmac_f32_e32 v4, v5, v4
    {0xd6fc6a05u, 0x04060501u}, // v_div_scale_f32 v5, vcc_lo, v1, v2, v1
    {0x100c0905u, 0x00000000u}, // v_mul_f32_e32 v6, v5, v4
    {0xd6130007u, 0x24160d03u}, // v_fma_f32 v7, -v3, v6, v5
    {0x560c0907u, 0x00000000u}, // v_fmac_f32_e32 v6, v7, v4
    {0xd6130003u, 0x24160d03u}, // v_fma_f32 v3, -v3, v6, v5
    {0xd6370003u, 0x041a0903u}, // v_div_fmas_f32 v3, v3, v4, v6
    {0xd6270001u, 0x04060503u}, // v_div_fixup_f32 v1, v3, v2, v1
};

constexpr std::array<uint32_t, 2> kMacroF64[] = {
    {0xd6fd7c06u, 0x040a0904u}, // v_div_scale_f64 v[6:7], null, v[4:5], v[4:5], v[2:3]
    {0x7e105f06u, 0x00000000u}, // v_rcp_f64_e32 v[8:9], v[6:7]
    {0xd614000au, 0x23ca1106u}, // v_fma_f64 v[10:11], -v[6:7], v[8:9], 1.0
    {0x2e101508u, 0x00000000u}, // v_fmac_f64_e32 v[8:9], v[8:9], v[10:11]
    {0xd614000au, 0x23ca1106u}, // v_fma_f64 v[10:11], -v[6:7], v[8:9], 1.0
    {0x2e101508u, 0x00000000u}, // v_fmac_f64_e32 v[8:9], v[8:9], v[10:11]
    {0xd6fd6a0au, 0x040a0902u}, // v_div_scale_f64 v[10:11], vcc_lo, v[2:3], v[4:5], v[2:3]
    {0x0c18110au, 0x00000000u}, // v_mul_f64_e32 v[12:13], v[10:11], v[8:9]
    {0xd6140006u, 0x242a1906u}, // v_fma_f64 v[6:7], -v[6:7], v[12:13], v[10:11]
    {0xd6380006u, 0x04321106u}, // v_div_fmas_f64 v[6:7], v[6:7], v[8:9], v[12:13]
    {0xd6280002u, 0x040a0906u}, // v_div_fixup_f64 v[2:3], v[6:7], v[4:5], v[2:3]
};

TEST(Gfx1250ExecutionTest, DivisionMacrosCoverScalingAndExceptionalQuotients) {
  ForceScalarGuard scalar_guard;
  struct Case {
    double n, d;
  };
  constexpr Case cases32[] = {
      {0x1p-140, 1.},    {0x1p-10, 0x1p127}, {0x1p100, 0x1p-20},  {1., 0x1p-140},
      {0x1p-30, 0x1p70}, {0x1p-24, 0x1p65},  {0x1p100, 0x1p-130}, {0x1p-105, 0x1p70},
      {3., 7.},          {-3., 7.},          {-0., 3.},           {3., -0.},
  };
  constexpr Case cases64[] = {
      {0x1p-1040, 1.},     {0x1p-10, 0x1p1023}, {0x1p1000, 0x1p-20},
      {0x1p-30, 0x1p1000}, {0x1p-960, 0x1p600}, {3., 7.},
      {-3., 7.},           {-0., 3.},
  };
  for (bool scalar : {false, true}) {
    util::set_force_scalar_for_testing(scalar);
    Gfx1250Sim sim;
    amdgpu::ComputeUnitCore *cu = sim.cu();
    amdgpu::Wavefront *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_mode_raw(0xf0); // RNE, preserve input/output denormals in both formats.
    wf->set_exec(0x55555555u);
    const uint32_t base = wf->vgpr_alloc().base;
    std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    for (bool wide : {false, true}) {
      const std::span<const Case> cases =
          wide ? std::span<const Case>(cases64) : std::span<const Case>(cases32);
      for (uint32_t lane = 0; lane < 32; ++lane) {
        const Case &test = cases[(lane / 2) % cases.size()];
        if (wide) {
          const uint64_t n = std::bit_cast<uint64_t>(test.n), d = std::bit_cast<uint64_t>(test.d);
          cu->write_vgpr(base + 2, lane, static_cast<uint32_t>(n));
          cu->write_vgpr(base + 3, lane, static_cast<uint32_t>(n >> 32));
          cu->write_vgpr(base + 4, lane, static_cast<uint32_t>(d));
          cu->write_vgpr(base + 5, lane, static_cast<uint32_t>(d >> 32));
        } else {
          cu->write_vgpr(base + 1, lane, std::bit_cast<uint32_t>(static_cast<float>(test.n)));
          cu->write_vgpr(base + 2, lane, std::bit_cast<uint32_t>(static_cast<float>(test.d)));
        }
      }
      const std::span<const std::array<uint32_t, 2>> instructions =
          wide ? std::span(kMacroF64) : std::span(kMacroF32);
      for (const std::array<uint32_t, 2> &words : instructions) {
        std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
        ASSERT_NE(inst, nullptr);
        inst->execute(*inst, wf);
      }
      for (uint32_t lane = 0; lane < 32; ++lane) {
        const Case &test = cases[(lane / 2) % cases.size()];
        if (wide) {
          const uint64_t actual =
              cu->read_vgpr(base + 2, lane) | (uint64_t{cu->read_vgpr(base + 3, lane)} << 32);
          const double expected = lane % 2 ? test.n : test.n / test.d;
          EXPECT_EQ(actual, std::bit_cast<uint64_t>(expected))
              << "f64 lane=" << lane << " scalar=" << scalar;
        } else {
          const float n = test.n, d = test.d;
          const float expected = lane % 2 ? n : n / d;
          EXPECT_EQ(cu->read_vgpr(base + 1, lane), std::bit_cast<uint32_t>(expected))
              << "f32 lane=" << lane << " scalar=" << scalar;
        }
      }
    }
  }
}

// Use different MODE fields for each format so a generator argument mix-up is
// observable. These cases exercise the decoded scalar and SIMD entry points.
TEST(DivisionTest, GuestModeAndFixupCausesMatchGfx1201) {
  ForceScalarGuard scalar_guard;
  for (bool scalar : {false, true}) {
    util::set_force_scalar_for_testing(scalar);
    // RDNA4 is measured hardware; CDNA5 checks the shared model with the same
    // expectations. Physical gfx1250 has not been independently qualified.
    for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
      SCOPED_TRACE(::testing::Message() << "arch=" << static_cast<int>(arch));
      amdgpu::GpuMemory memory("division_memory");
      amdgpu::L2Cache cache("division_cache");
      cache.set_backing_memory(&memory);
      amdgpu::ComputeUnitCore::Config config{};
      config.arch = arch;
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 128;
      config.vgprs_per_wf = 32;
      config.lds_size_kb = 64;
      std::unique_ptr<amdgpu::ComputeUnitCore> cu =
          amdgpu::ComputeUnitCore::create("division", config, &memory, &cache);
      amdgpu::Wavefront *wf = cu->dispatch_wf(0, 0, 128, 32);
      ASSERT_NE(wf, nullptr);
      wf->set_exec(1);
      const uint32_t base = wf->vgpr_alloc().base;
      std::unique_ptr<Decoder> decoder = Decoder::create(arch);
      ASSERT_NE(decoder, nullptr);
      for (bool wide : {false, true}) {
        const uint32_t stride = wide ? 2 : 1;
        const uint64_t one = wide ? 0x3ff0000000000000ULL : 0x3f800000u;
        const uint64_t inf = wide ? 0x7ff0000000000000ULL : 0x7f800000u;
        const uint64_t midpoint =
            wide ? std::bit_cast<uint64_t>(0x1p-947) : std::bit_cast<uint32_t>(0x1p-86f);
        const auto write = [&](uint32_t reg, uint64_t bits) {
          cu->write_vgpr(base + reg, 0, static_cast<uint32_t>(bits));
          if (wide)
            cu->write_vgpr(base + reg + 1, 0, static_cast<uint32_t>(bits >> 32));
        };
        const auto result = [&]() -> uint64_t {
          return cu->read_vgpr(base, 0) | (wide ? uint64_t{cu->read_vgpr(base + 1, 0)} << 32 : 0);
        };
        const std::array<uint32_t, 2> fmas_words =
            wide ? std::array<uint32_t, 2>{0xd6380000u, 0x041a0902u}
                 : std::array<uint32_t, 2>{0xd6370000u, 0x040e0501u};
        const std::array<uint32_t, 2> fixup_words =
            wide ? std::array<uint32_t, 2>{0xd6280000u, 0x041a0902u}
                 : std::array<uint32_t, 2>{0xd6270000u, 0x040e0501u};
        std::unique_ptr<Instruction> fmas(decode_valid(*decoder, fmas_words.data()));
        std::unique_ptr<Instruction> fixup(decode_valid(*decoder, fixup_words.data()));
        ASSERT_NE(fmas, nullptr);
        ASSERT_NE(fixup, nullptr);
        for (uint32_t rounding = 0; rounding < 4; ++rounding) {
          for (uint32_t denorm = 0; denorm < 4; ++denorm) {
            SCOPED_TRACE(::testing::Message() << "wide=" << wide << " scalar=" << scalar
                                              << " rounding=" << rounding << " denorm=" << denorm);
            wf->set_mode_raw(
                wide ? (rounding << 2) | (denorm << 6) | (rounding ^ 3u) | ((denorm ^ 3u) << 4)
                     : rounding | (denorm << 4) | ((rounding ^ 3u) << 2) | ((denorm ^ 3u) << 6));
            write(stride, 0);
            write(2 * stride, 0);
            write(3 * stride, midpoint);
            wf->set_vcc_mask(1);
            cu->write_vgpr(base, 1, 0x12345678u);
            fmas->execute(*fmas, wf);
            EXPECT_EQ(result(), rounding == 1 && (denorm & 2u) ? 1u : 0u);
            EXPECT_EQ(cu->read_vgpr(base, 1), 0x12345678u);

            // A flushed subnormal denominator is zero for the quotient and its
            // divide-by-zero cause. Only preserved inputs raise input-denormal.
            write(stride, one);
            write(2 * stride, 1);
            write(3 * stride, one);
            wf->set_trapsts(1u << 5); // An unrelated sticky cause must survive.
            wf->clear_pending_alu_causes();
            fixup->execute(*fixup, wf);
            const uint32_t causes = (denorm & 1u) ? 1u << 1 : 1u << 2;
            EXPECT_EQ(result(), (denorm & 1u) ? one : inf);
            EXPECT_EQ(wf->pending_alu_causes(), causes);
            EXPECT_EQ(wf->trapsts(), (1u << 5) | causes);
            EXPECT_EQ(cu->read_vgpr(base, 1), 0x12345678u);
          }
        }
        for (bool clamp : {false, true}) {
          wf->set_mode_raw(0);
          write(stride, inf - 1); // Finite quotient before OMOD.
          write(2 * stride, one);
          write(3 * stride, inf - 1);
          std::array<uint32_t, 2> words = fixup_words;
          words[0] |= clamp ? 0x8000u : 0;
          words[1] |= 1u << 27; // OMOD x2.
          std::unique_ptr<Instruction> modified(decode_valid(*decoder, words.data()));
          ASSERT_NE(modified, nullptr);
          wf->set_trapsts(1u << 5);
          wf->clear_pending_alu_causes();
          modified->execute(*modified, wf);
          EXPECT_EQ(result(), clamp ? one : inf);
          EXPECT_EQ(wf->pending_alu_causes(), 1u << 3);
          EXPECT_EQ(wf->trapsts(), (1u << 3) | (1u << 5));

          // Directed FIXUP recovery can produce max-finite, which OMOD then
          // overflows. ABS/NEG on either original input determines that recovery.
          const uint64_t sign = wide ? 0x8000000000000000ULL : 0x80000000u;
          const uint64_t small =
              wide ? std::bit_cast<uint64_t>(0x1p-20) : std::bit_cast<uint32_t>(0x1p-20f);
          for (uint32_t source : {1u, 2u}) {
            for (bool absolute : {false, true}) {
              SCOPED_TRACE(::testing::Message()
                           << "wide=" << wide << " scalar=" << scalar << " clamp=" << clamp
                           << " source=" << source << " absolute=" << absolute);
              const uint32_t rounding = absolute ? 2 : 1;
              wf->set_mode_raw(wide ? rounding << 2 : rounding);
              write(stride, inf);
              write(2 * stride, small | (absolute && source == 1 ? sign : 0));
              write(3 * stride, (inf - 1) | (absolute && source == 2 ? sign : 0));
              words = fixup_words;
              words[0] |= (clamp ? 0x8000u : 0) | (absolute ? 1u << (8 + source) : 0);
              words[1] |= (1u << 27) | (absolute ? 0 : 1u << (29 + source));
              modified.reset(decode_valid(*decoder, words.data()));
              ASSERT_NE(modified, nullptr);
              wf->set_trapsts(1u << 5);
              wf->clear_pending_alu_causes();
              modified->execute(*modified, wf);
              const uint64_t expected =
                  absolute ? (clamp ? one : inf - 1) : (clamp ? 0 : (inf - 1) | sign);
              EXPECT_EQ(result(), expected);
              EXPECT_EQ(wf->pending_alu_causes(), 1u << 3);
              EXPECT_EQ(wf->trapsts(), (1u << 3) | (1u << 5));
              EXPECT_EQ(cu->read_vgpr(base, 1), 0x12345678u);
            }
          }
        }

        // Values and exception masks measured on physical gfx1201, in all four
        // rounding modes. FIXUP recovery reports overflow/underflow and inexact;
        // OMOD suppresses underflow and inexact, but retains overflow. Source
        // modifiers and equivalent premodified operands must agree.
        const uint64_t sign = wide ? 0x8000000000000000ULL : 0x80000000u;
        const uint64_t nan = inf | sign | (wide ? 0x0008000000000000ULL : 0x00400000u);
        const uint64_t tiny =
            wide ? std::bit_cast<uint64_t>(0x1p-1030) : std::bit_cast<uint32_t>(0x1p-130f);
        const uint64_t huge =
            wide ? std::bit_cast<uint64_t>(0x1p1000) : std::bit_cast<uint32_t>(0x1p100f);
        const uint64_t minimum = wide ? 1ULL << 52 : 1ULL << 23;
        const uint64_t max = inf - 1;
        const uint64_t half_max = max - (wide ? 1ULL << 52 : 1ULL << 23);
        const std::array<uint64_t, 4> negative_overflow = {inf | sign, max | sign, inf | sign,
                                                           max | sign};
        const std::array<uint64_t, 4> positive_overflow = {inf, inf, max, max};
        struct FixupCase {
          std::array<uint64_t, 3> raw, effective;
          uint32_t abs, neg, mode, omod, causes;
          std::array<uint64_t, 4> expected;
        };
        const FixupCase fixup_cases[] = {
            {{inf, one, one}, {inf, one, one | sign}, 0, 4, 0, 1, 0x8, negative_overflow},
            {{nan, tiny | sign, huge}, {nan, tiny, huge}, 2, 0, 0xf0, 1, 0xa, positive_overflow},
            {{inf, one, one}, {inf, one, one | sign}, 0, 4, 0xf0, 0, 0x28, negative_overflow},
            {{nan, tiny | sign, huge}, {nan, tiny, huge}, 2, 0, 0xf0, 0, 0x2a, positive_overflow},
            {{inf, one, one}, {inf, one, one | sign}, 0, 4, 0xf0, 2, 0x8, negative_overflow},
            {{inf, one, one},
             {inf, one, one | sign},
             0,
             4,
             0xf0,
             3,
             0x8,
             {inf | sign, half_max | sign, inf | sign, half_max | sign}},
            {{nan, tiny | sign, huge},
             {nan, tiny, huge},
             2,
             0,
             0xf0,
             3,
             0xa,
             {inf, inf, half_max, half_max}},
            {{}, {nan, huge, tiny}, 0, 0, 0xf0, 0, 0x32, {0, 1, 0, 0}},
            {{}, {nan, huge, tiny}, 0, 0, 0xf0, 1, 0x2, {0, 0, 0, 0}},
            {{}, {inf, one, inf}, 0, 0, 0xf0, 1, 0, {inf, inf, inf, inf}},
            {{}, {inf, 0, one}, 0, 0, 0xf0, 1, 0x4, {inf, inf, inf, inf}},
            {{}, {tiny, one, one}, 0, 0, 0xf0, 0, 0, {tiny, tiny, tiny, tiny}},
            {{}, {max, one, max}, 0, 0, 0xf0, 1, 0x8, positive_overflow},
            {{}, {max, one, max}, 0, 0, 0xf0, 3, 0, {half_max, half_max, half_max, half_max}},
            {{}, {one, tiny, one}, 0, 0, 0, 0, 0x4, {inf, inf, inf, inf}},
            {{}, {one, one, tiny}, 0, 0, 0, 0, 0, {0, 0, 0, 0}},
            {{}, {one, tiny, tiny}, 0, 0, 0, 0, 0x1, {nan, nan, nan, nan}},
            // OMOD flushes before rounding at the normal/subnormal boundary.
            {{}, {2 * minimum - 1, one, 2 * minimum - 1}, 0, 0, 0xf0, 3, 0, {0, 0, 0, 0}},
            {{}, {minimum - 1, one, minimum - 1}, 0, 0, 0xf0, 1, 0x2, {0, 0, 0, 0}},
            // A zero generated by OMOD underflow retains its negative sign.
            {{}, {minimum, one, minimum | sign}, 0, 0, 0xf0, 3, 0, {sign, sign, sign, sign}},
            // An existing negative zero or subnormal becomes positive zero.
            {{}, {0, one, one | sign}, 0, 0, 0xf0, 1, 0, {0, 0, 0, 0}},
            {{}, {1, one, one | sign}, 0, 0, 0xf0, 1, 0, {0, 0, 0, 0}},
        };
        for (const FixupCase &test : fixup_cases) {
          for (uint32_t rounding = 0; rounding < 4; ++rounding) {
            for (bool use_modifiers : {false, true}) {
              if (use_modifiers && !(test.abs | test.neg))
                continue;
              SCOPED_TRACE(::testing::Message()
                           << "wide=" << wide << " scalar=" << scalar << " abs=" << test.abs
                           << " neg=" << test.neg << " use_modifiers=" << use_modifiers
                           << " rounding=" << rounding << " omod=" << test.omod
                           << " causes=" << test.causes);
              wf->set_mode_raw(test.mode | (wide ? rounding << 2 : rounding));
              const std::array<uint64_t, 3> &inputs = use_modifiers ? test.raw : test.effective;
              write(stride, inputs[0]);
              write(2 * stride, inputs[1]);
              write(3 * stride, inputs[2]);
              std::array<uint32_t, 2> words = fixup_words;
              words[0] |= use_modifiers ? test.abs << 8 : 0;
              words[1] |= (test.omod << 27) | (use_modifiers ? test.neg << 29 : 0);
              std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
              ASSERT_NE(instruction, nullptr);
              wf->set_trapsts(1u << 6); // An unrelated sticky cause must survive.
              wf->clear_pending_alu_causes();
              instruction->execute(*instruction, wf);
              EXPECT_EQ(result(), test.expected[rounding]);
              EXPECT_EQ(wf->pending_alu_causes(), test.causes);
              EXPECT_EQ(wf->trapsts(), (1u << 6) | test.causes);
              EXPECT_EQ(cu->read_vgpr(base, 1), 0x12345678u);
            }
          }
        }
      }
      wf->halt();
    }
  }
}
} // namespace
