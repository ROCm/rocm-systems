// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP16/FP32 interpolation stages and quad-lane operand routing.

#include "graphics/helpers.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include "rj_arith/detail/rj_arith_graphics.h"

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p10_f32_w32, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Interp32, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p2_f32_w32, 1,
                                            rocjitsu::arith::test::GraphicsFamily::Interp32, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10F16F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p10_f16_f32_w32, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2F16F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p2_f16_f32_w32, 1,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10RtzF16F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p10_rtz_f16_f32_w32, 2,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2RtzF16F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p2_rtz_f16_f32_w32, 3,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p10_f32_w64, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Interp32, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p2_f32_w64, 1,
                                            rocjitsu::arith::test::GraphicsFamily::Interp32, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10F16F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p10_f16_f32_w64, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2F16F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p2_f16_f32_w64, 1,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10RtzF16F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p10_rtz_f16_f32_w64, 2,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2RtzF16F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p2_rtz_f16_f32_w64, 3,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

namespace {

// Hardware results at MODE 0xf0 (denormals kept, nearest even), where no round/denormal flag
// selects the guest path. Operands are the ones lane 0 reads: A[1] (P10) or A[2] (P2), B[0] and
// C[0]. An invalid product returns 0xffc00000 even when C is a NaN; otherwise the first NaN of
// A, B, C is returned after NEG, and an FP16 NaN keeps its payload when widened. gfx1201 and
// gfx1100 with MODE.IEEE set quiet that NaN; gfx1100 with IEEE clear returns it unquieted.
struct NanCase {
  int op;
  bool ieee;
  uint32_t modifiers, a, b, c, expected;
};

const NanCase nan_cases[] = {
    {0, true, 0, 0x7f800000, 0, 0x7fc00000, 0xffc00000},
    {0, true, 0, 0x7f800000, 0, 0x7f800001, 0xffc00000},
    {0, true, RJ_ARITH_ALU_NEG_A, 0x7fc00000, 0x7fc00000, 0, 0xffc00000},
    {0, true, RJ_ARITH_ALU_NEG_B, 0x7fc00000, 0x7fc00000, 0, 0x7fc00000},
    {1, true, 0, 0x7f800001, 0x7fc00000, 0, 0x7fc00001},
    {1, true, RJ_ARITH_ALU_NEG_A, 0x7f800001, 0x7fc00000, 0, 0xffc00001},
    {1, true, RJ_ARITH_ALU_NEG_C, 0xff800000, 0, 0x7f800001, 0xffc00000},
    {2, true, 0, 0x83ff7c00, 0, 0xb8007c01, 0xffc00000},
    {2, true, RJ_ARITH_ALU_HIGH_A, 0x7e553e00, 0x7fc00000, 0x80000000, 0x7fcaa000},
    {3, true, RJ_ARITH_ALU_HIGH_A, 0x7e553e00, 0x7fc00000, 0x80000000, 0x7fcaa000},
    {3, true, RJ_ARITH_ALU_NEG_C, 0xcae0fde7, 0x7faac328, 0x82a84d0c, 0xfffce000},
    {0, false, RJ_ARITH_ALU_NEG_A, 0x7f88174e, 0x80ee2802, 0x43900000, 0xff88174e},
    {1, false, RJ_ARITH_ALU_NEG_A, 0x7fab7504, 0x42900000, 0x45e510ef, 0xffab7504},
    {2, false, RJ_ARITH_ALU_NEG_C, 0xbc008000, 0, 0xb8007c01, 0xff802000},
};

} // namespace

// Check NaN results against the hardware without round/denormal controls, in Compatibility and
// every bit-exact semantics, so a dispatch change in any policy shows up as a wrong NaN.
TEST(GraphicsInterpW32, ExactNanMatchesHardware) {
  decltype(&rj_arith_v_interp_p10_f32_w32) calls[] = {
      rj_arith_v_interp_p10_f32_w32, rj_arith_v_interp_p2_f32_w32,
      rj_arith_v_interp_p10_f16_f32_w32, rj_arith_v_interp_p10_rtz_f16_f32_w32};
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
          RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
          RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
      for (const NanCase &test : nan_cases) {
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << (&test - nan_cases));
        const bool p2 = test.op == 1;
        rocjitsu::arith::test::Registers<4, 32> r{};
        r.v[0][p2 ? 2 : 1] = test.a;
        r.v[1][0] = test.b;
        r.v[2][0] = test.c;
        const uint64_t flags = cpu | semantics | (test.ieee ? RJ_ARITH_FP_IEEE_MODE : 0);
        ASSERT_EQ(calls[test.op](flags, 1, test.modifiers, r.v + 3, r.v, r.v + 1, r.v + 2),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[3][0], test.expected);
      }
    }
  }
}

namespace {

// Operands spanning the ranges that separate the specialized kernels from the guest model:
// normals, both zeroes, the denormal boundary, infinities, and quiet and signaling NaNs.
const uint32_t dispatch_pool[] = {
    0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x40490fdb, 0x7f7fffff, 0xff7fffff,
    0x00800000, 0x80800000, 0x00000001, 0x007fffff, 0x7f800000, 0xff800000, 0x7fc00000,
    0x7f800001, 0xffc00000, 0x33000000, 0x4b000000, 0x3f7fffff, 0x5f000000,
};

} // namespace

// Compare quad routing and numerical results with the inherited guest model.
// Compatibility/bit-exact modes also require identical NaN payloads; Loose permits
// native NaN handling and rounding. All CPU levels exercise their selected backend.
TEST(GraphicsInterpW32, MatchesTheGuestModelWithoutGuestControls) {
  constexpr int pool = sizeof(dispatch_pool) / sizeof(dispatch_pool[0]);
  const uint32_t modifiers[] = {0, RJ_ARITH_ALU_NEG_A, RJ_ARITH_ALU_NEG_B, RJ_ARITH_ALU_NEG_C,
                                RJ_ARITH_ALU_CLAMP};
  // P10 and P2 read different quad lanes for A and C, and the FP16 form widens A and C.
  struct Form {
    const char *name;
    decltype(&rj_arith_v_interp_p10_f32_w32) call;
    bool p2, half;
  };
  const Form forms[] = {
      {"p10_f32", rj_arith_v_interp_p10_f32_w32, false, false},
      {"p2_f32", rj_arith_v_interp_p2_f32_w32, true, false},
      {"p10_f16", rj_arith_v_interp_p10_f16_f32_w32, false, true},
  };
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (bool ieee : {false, true}) {
      for (uint32_t modifier : modifiers) {
        for (const Form &form : forms) {
          for (int seed = 0; seed < pool; ++seed) {
            SCOPED_TRACE(::testing::Message() << cpu << '/' << ieee << '/' << modifier << '/'
                                              << form.name << '/' << seed);
            // Unmodified P2 selects AVX-512 only with an aligned streaming source.
            // B/C retain their guarded, unaligned views below.
            alignas(64) uint32_t a[32];
            uint32_t b[32], c[32];
            // Walk the pool across lanes so each quad mixes finite and non-finite operands.
            for (int lane = 0; lane < 32; ++lane) {
              a[lane] = dispatch_pool[(seed + lane) % pool];
              b[lane] = dispatch_pool[(seed + lane * 3 + 1) % pool];
              c[lane] = dispatch_pool[(seed + lane * 7 + 2) % pool];
            }

            // The guest model: broadcast the quad coefficients, then one exact mixed FMA.
            uint32_t av[32], cv[32], expected[32];
            for (int lane = 0; lane < 32; ++lane) {
              av[lane] = a[(lane & ~3) + (form.p2 ? 2 : 1)];
              cv[lane] = c[form.p2 ? lane : (lane & ~3)];
              expected[lane] = 0xdeadbe00u + uint32_t(lane);
            }
            uint32_t *ep = expected;
            const uint32_t *avp = av, *bp = b, *cvp = cv;
            const uint64_t guest_flags = cpu | (ieee ? RJ_ARITH_FP_IEEE_MODE : 0) |
                                         RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
            const uint64_t guest_mode =
                modifier | (form.half ? RJ_ARITH_MIX_F16_A | RJ_ARITH_MIX_F16_C : 0);
            ASSERT_EQ(rj_arith_v_fma_mix_f32_w32(guest_flags, ~0u, guest_mode, &ep, &avp, &bp, &cvp,
                                                 nullptr),
                      RJ_ARITH_SUCCESS);

            for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY,
                                       RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                                       RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
              rocjitsu::arith::test::Registers<4, 32> r{};
              for (int lane = 0; lane < 32; ++lane) {
                r.v[1][lane] = b[lane];
                r.v[2][lane] = c[lane];
                r.v[3][lane] = 0xdeadbe00u + uint32_t(lane);
              }
              const uint64_t flags = cpu | semantics | (ieee ? RJ_ARITH_FP_IEEE_MODE : 0);
              const uint32_t *source_a = a;
              ASSERT_EQ(form.call(flags, ~0u, modifier, r.v + 3, &source_a, r.v + 1, r.v + 2),
                        RJ_ARITH_SUCCESS)
                  << semantics;
              for (int lane = 0; lane < 32; ++lane) {
                if (semantics == RJ_ARITH_SEMANTICS_LOOSE) {
                  rocjitsu::arith::test::expect_f32(r.v[3][lane],
                                                    std::bit_cast<float>(expected[lane]), semantics,
                                                    /*atol=*/1e-44f, /*rtol=*/2e-7f);
                } else {
                  EXPECT_EQ(r.v[3][lane], expected[lane]) << semantics << " lane " << lane;
                }
              }
            }
          }
        }
      }
    }
  }
}
