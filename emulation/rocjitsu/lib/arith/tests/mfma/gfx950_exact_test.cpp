// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// MI350X-calibrated gfx950 dense MFMA and sparse SMFMAC exact arithmetic, per-pass FP16_OVFL
// saturation, 13-bit OCP FP8/BF8 chunk alignment, controls gating, and whole-VGPR aliasing.
#include "common/cpu.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include <array>
#include <bit>
#include <cstdint>
#include <gtest/gtest.h>

namespace {
using namespace rocjitsu::arith::test;
using DenseEntry = decltype(&rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx950);
using SparseEntry = decltype(&rj_arith_v_smfmac_f32_16x16x32_f16_w64_gfx950);

struct DenseForm {
  const char *name;
  int m, k, batches;
  int format_a, format_b; // 0=F16, 1=BF16, 2=OCP_FP8, 3=OCP_BF8
  DenseEntry entry;
};

const DenseForm kDenseForms[] = {
    {"f16_32x32x4_2b", 32, 4, 2, 0, 0, rj_arith_v_mfma_f32_32x32x4_2b_f16_w64_gfx950},
    {"f16_16x16x4_4b", 16, 4, 4, 0, 0, rj_arith_v_mfma_f32_16x16x4_4b_f16_w64_gfx950},
    {"f16_4x4x4_16b", 4, 4, 16, 0, 0, rj_arith_v_mfma_f32_4x4x4_16b_f16_w64_gfx950},
    {"f16_32x32x8", 32, 8, 1, 0, 0, rj_arith_v_mfma_f32_32x32x8_f16_w64_gfx950},
    {"f16_16x16x16", 16, 16, 1, 0, 0, rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx950},
    {"f16_16x16x32", 16, 32, 1, 0, 0, rj_arith_v_mfma_f32_16x16x32_f16_w64},
    {"f16_32x32x16", 32, 16, 1, 0, 0, rj_arith_v_mfma_f32_32x32x16_f16_w64},
    {"bf16_32x32x4_2b", 32, 4, 2, 1, 1, rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64_gfx950},
    {"bf16_16x16x4_4b", 16, 4, 4, 1, 1, rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64_gfx950},
    {"bf16_4x4x4_16b", 4, 4, 16, 1, 1, rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64_gfx950},
    {"bf16_32x32x8", 32, 8, 1, 1, 1, rj_arith_v_mfma_f32_32x32x8_bf16_w64_gfx950},
    {"bf16_16x16x16", 16, 16, 1, 1, 1, rj_arith_v_mfma_f32_16x16x16_bf16_w64_gfx950},
    {"bf16_16x16x32", 16, 32, 1, 1, 1, rj_arith_v_mfma_f32_16x16x32_bf16_w64},
    {"bf16_32x32x16", 32, 16, 1, 1, 1, rj_arith_v_mfma_f32_32x32x16_bf16_w64},
    {"fp8_fp8_16x16x32", 16, 32, 1, 2, 2, rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64_gfx950},
    {"fp8_bf8_16x16x32", 16, 32, 1, 2, 3, rj_arith_v_mfma_f32_16x16x32_fp8_bf8_w64_gfx950},
    {"bf8_fp8_16x16x32", 16, 32, 1, 3, 2, rj_arith_v_mfma_f32_16x16x32_bf8_fp8_w64_gfx950},
    {"bf8_bf8_16x16x32", 16, 32, 1, 3, 3, rj_arith_v_mfma_f32_16x16x32_bf8_bf8_w64_gfx950},
    {"fp8_fp8_32x32x16", 32, 16, 1, 2, 2, rj_arith_v_mfma_f32_32x32x16_fp8_fp8_w64_gfx950},
    {"fp8_bf8_32x32x16", 32, 16, 1, 2, 3, rj_arith_v_mfma_f32_32x32x16_fp8_bf8_w64_gfx950},
    {"bf8_fp8_32x32x16", 32, 16, 1, 3, 2, rj_arith_v_mfma_f32_32x32x16_bf8_fp8_w64_gfx950},
    {"bf8_bf8_32x32x16", 32, 16, 1, 3, 3, rj_arith_v_mfma_f32_32x32x16_bf8_bf8_w64_gfx950},
};

struct SparseForm {
  const char *name;
  int m, k;
  int format_a, format_b; // 0=F16, 1=BF16, 2=OCP_FP8, 3=OCP_BF8
  SparseEntry entry;
};

const SparseForm kSparseForms[] = {
    {"smfmac_f16_16x16x32", 16, 32, 0, 0, rj_arith_v_smfmac_f32_16x16x32_f16_w64_gfx950},
    {"smfmac_bf16_16x16x32", 16, 32, 1, 1, rj_arith_v_smfmac_f32_16x16x32_bf16_w64_gfx950},
    {"smfmac_f16_32x32x16", 32, 16, 0, 0, rj_arith_v_smfmac_f32_32x32x16_f16_w64_gfx950},
    {"smfmac_bf16_32x32x16", 32, 16, 1, 1, rj_arith_v_smfmac_f32_32x32x16_bf16_w64_gfx950},
    {"smfmac_f16_16x16x64", 16, 64, 0, 0, rj_arith_v_smfmac_f32_16x16x64_f16_w64},
    {"smfmac_bf16_16x16x64", 16, 64, 1, 1, rj_arith_v_smfmac_f32_16x16x64_bf16_w64},
    {"smfmac_f16_32x32x32", 32, 32, 0, 0, rj_arith_v_smfmac_f32_32x32x32_f16_w64},
    {"smfmac_bf16_32x32x32", 32, 32, 1, 1, rj_arith_v_smfmac_f32_32x32x32_bf16_w64},
    {"smfmac_fp8_fp8_16x16x64", 16, 64, 2, 2, rj_arith_v_smfmac_f32_16x16x64_fp8_fp8_w64_gfx950},
    {"smfmac_fp8_bf8_16x16x64", 16, 64, 2, 3, rj_arith_v_smfmac_f32_16x16x64_fp8_bf8_w64_gfx950},
    {"smfmac_bf8_fp8_16x16x64", 16, 64, 3, 2, rj_arith_v_smfmac_f32_16x16x64_bf8_fp8_w64_gfx950},
    {"smfmac_bf8_bf8_16x16x64", 16, 64, 3, 3, rj_arith_v_smfmac_f32_16x16x64_bf8_bf8_w64_gfx950},
    {"smfmac_fp8_fp8_32x32x32", 32, 32, 2, 2, rj_arith_v_smfmac_f32_32x32x32_fp8_fp8_w64_gfx950},
    {"smfmac_fp8_bf8_32x32x32", 32, 32, 2, 3, rj_arith_v_smfmac_f32_32x32x32_fp8_bf8_w64_gfx950},
    {"smfmac_bf8_fp8_32x32x32", 32, 32, 3, 2, rj_arith_v_smfmac_f32_32x32x32_bf8_fp8_w64_gfx950},
    {"smfmac_bf8_bf8_32x32x32", 32, 32, 3, 3, rj_arith_v_smfmac_f32_32x32x32_bf8_bf8_w64_gfx950},
    {"smfmac_fp8_fp8_16x16x128", 16, 128, 2, 2, rj_arith_v_smfmac_f32_16x16x128_fp8_fp8_w64},
    {"smfmac_fp8_bf8_16x16x128", 16, 128, 2, 3, rj_arith_v_smfmac_f32_16x16x128_fp8_bf8_w64},
    {"smfmac_bf8_fp8_16x16x128", 16, 128, 3, 2, rj_arith_v_smfmac_f32_16x16x128_bf8_fp8_w64},
    {"smfmac_bf8_bf8_16x16x128", 16, 128, 3, 3, rj_arith_v_smfmac_f32_16x16x128_bf8_bf8_w64},
    {"smfmac_fp8_fp8_32x32x64", 32, 64, 2, 2, rj_arith_v_smfmac_f32_32x32x64_fp8_fp8_w64},
    {"smfmac_fp8_bf8_32x32x64", 32, 64, 2, 3, rj_arith_v_smfmac_f32_32x32x64_fp8_bf8_w64},
    {"smfmac_bf8_fp8_32x32x64", 32, 64, 3, 2, rj_arith_v_smfmac_f32_32x32x64_bf8_fp8_w64},
    {"smfmac_bf8_bf8_32x32x64", 32, 64, 3, 3, rj_arith_v_smfmac_f32_32x32x64_bf8_bf8_w64},
};

uint32_t code_one(int format, bool neg = false) {
  const uint32_t sign = neg ? (format < 2 ? 0x8000u : 0x80u) : 0u;
  return sign | (format == 0 ? 0x3c00u : format == 1 ? 0x3f80u : format == 2 ? 0x38u : 0x3cu);
}

uint32_t code_nan(int format) {
  return format == 0 ? 0x7e01u : format == 1 ? 0x7fc1u : format == 2 ? 0x7fu : 0x7du;
}

TEST(MfmaGfx950Exact, All22DenseFormsSupportExactModesOvflAndAliasing) {
  for (const auto &form : kDenseForms) {
    SCOPED_TRACE(form.name);
    const int per_word = form.format_a < 2 ? 2 : 4;
    const int inputs = form.m * form.k * form.batches / (64 * per_word);
    const int outputs = form.m * form.m * form.batches / 64;
    const int per_lane = form.k * form.m * form.batches / 64;

    // Place a NaN in depth 0 of batch 0, row 0, col 0, and 1.0 * 1.0 = 1.0 in depth (k - 1).
    // On single-pass shapes (k <= pps), ovfl=0 produces canonical 0xffc00000 and ovfl=1 flushes
    // NaN to +0.0. On multi-pass shapes (k > pps), pass 0 produces NaN -> flushed to +0.0 under
    // ovfl=1, and the final pass adds 1.0 * 1.0 = 1.0 (0x3f800000)!
    std::array<std::array<uint32_t, 64>, 72> input{};
    auto put_factor = [&](bool b_side, int depth, uint32_t code) {
      const int lane = depth / per_lane * form.m * form.batches;
      const int reg = depth % per_lane / per_word;
      const int shift = (32 / per_word) * (depth % per_word);
      input[(b_side ? inputs : 0) + reg][lane] |= code << shift;
    };
    put_factor(false, 0, code_nan(form.format_a));
    put_factor(true, 0, code_one(form.format_b));
    put_factor(false, form.k - 1, code_one(form.format_a));
    put_factor(true, form.k - 1, code_one(form.format_b));

    const int pps = form.format_a < 2 ? 8 : 16;
    const uint32_t expected_ovfl1 = form.k > pps ? 0x3f800000u : 0x00000000u;

    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t sem : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                           RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                           RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
        for (bool ovfl : {false, true}) {
          for (int alias : {0, inputs, 2 * inputs, 2 * inputs + outputs}) {
            auto storage = input;
            const uint32_t *a[4], *b[4], *c[32];
            uint32_t *d[32];
            for (int r = 0; r < inputs; ++r) {
              a[r] = storage[r].data();
              b[r] = storage[inputs + r].data();
            }
            for (int r = 0; r < outputs; ++r) {
              c[r] = storage[2 * inputs + r].data();
              d[r] = storage[alias + r].data();
            }
            const uint64_t flags = cpu | sem | (ovfl ? RJ_ARITH_FP16_OVFL : 0);
            ASSERT_EQ(form.entry(flags, 0, d, a, b, c), RJ_ARITH_SUCCESS);
            EXPECT_EQ(storage[alias][0], ovfl ? expected_ovfl1 : 0xffc00000u);
          }
        }
      }
    }

    // Non-zero CBSZ/ABID/BLGP controls remain gated in strict modes.
    for (uint64_t sem : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                         RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
      EXPECT_EQ(form.entry(sem, 1, nullptr, nullptr, nullptr, nullptr),
                RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
    }
  }
}

TEST(MfmaGfx950Exact, OcpFp8Uses13BitAligned8ProductChunksAndDropsFarC) {
  // Verify MI350X's 13-bit chunk truncation on OCP FP8 16x16x32:
  // In chunk 0 (k=0..7): k=0 has +256 * 1 = +256 (exp 8), k=1 has 2^-6 * 1 = 2^-6 (rsh = 14 > 13
  // -> truncated to 0 in chunk 0!), and k=2 has -256 * 1 = -256, so chunk 0 cancels to 0!
  // Meanwhile if k=8 (chunk 1) has 2^-6 * 1 = 2^-6 (with no 256 in chunk 1), chunk 1 retains
  // 2^-6 = 0x3c800000 / 2^6 = 0x3c800000? 2^-6 = 0x3c800000 is 2^-6 = 0.015625 = 0x3c800000!
  std::array<std::array<uint32_t, 64>, 8> st{};
  // OCP E4M3FN (bias 7):
  // +256 = 1.0 * 2^8 -> exp_field = 15, frac = 0 -> 0x78
  // -256 = -1.0 * 2^8 -> 0xf8
  // 2^-6 = 1.0 * 2^-6 -> exp_field = 1, frac = 0 -> 0x08
  // 1.0  = 1.0 * 2^0  -> exp_field = 7, frac = 0 -> 0x38
  // For 16x16x32 (per_lane = 8, 2 VGPRs of 4 bytes per lane):
  // lane 0 holds k=0..3 in reg 0, k=4..7 in reg 1; lane 16 holds k=8..11 in reg 0.
  st[0][0] = 0x00f80878u;  // k=0:+256, k=1:2^-6, k=2:-256, k=3:0 (chunk 0)
  st[0][16] = 0x00000008u; // k=8:2^-6 (chunk 1)
  st[2][0] = 0x00383838u;  // b at k=0,1,2 = 1.0
  st[2][16] = 0x00000038u; // b at k=8 = 1.0
  const uint32_t *a[] = {st[0].data(), st[1].data()};
  const uint32_t *b[] = {st[2].data(), st[3].data()};
  const uint32_t *c[] = {st[4].data(), st[5].data(), st[6].data(), st[7].data()};
  uint32_t out[4][64]{};
  uint32_t *d[] = {out[0], out[1], out[2], out[3]};

  ASSERT_EQ(rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64_gfx950(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL,
                                                            0, d, a, b, c),
            RJ_ARITH_SUCCESS);
  // k=1's 2^-6 in chunk 0 was truncated to 0 alongside +-256, leaving only chunk 1's 2^-6!
  // (A 24-bit-aligned model like gfx942 would keep BOTH 2^-6 terms and return 2^-5 = 0x3d000000.)
  EXPECT_EQ(out[0][0], 0x3c800000u);

  // Pin both sides of the ce < te - 25 accumulator cutoff:
  // 1) Single 32 * 128 = 4096 (te = 12) product: C = -2^-14 (ce = -14 < -13) is dropped ->
  //    0x45800000, whereas C = -2^-13 (ce = -13, at the cutoff) borrows -> 0x457fffff.
  st[0][0] = 0x00000060u; // +32.0 (E4M3FN exp_field = 12)
  st[0][16] = 0u;
  st[2][0] = 0x00000070u; // +128.0 (E4M3FN exp_field = 14)
  st[2][16] = 0u;
  st[4][0] = 0xb8800000u; // -2^-14 (below cutoff)
  ASSERT_EQ(rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64_gfx950(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL,
                                                            0, d, a, b, c),
            RJ_ARITH_SUCCESS);
  EXPECT_EQ(out[0][0], 0x45800000u);

  st[4][0] = 0xb9000000u; // -2^-13 (at cutoff)
  ASSERT_EQ(rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64_gfx950(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL,
                                                            0, d, a, b, c),
            RJ_ARITH_SUCCESS);
  EXPECT_EQ(out[0][0], 0x457fffffu);

  st[4][0] = 0xff800000u; // -Inf survives!
  ASSERT_EQ(rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64_gfx950(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL,
                                                            0, d, a, b, c),
            RJ_ARITH_SUCCESS);
  EXPECT_EQ(out[0][0], 0xff800000u);

  // Under RJ_ARITH_FP16_OVFL, -Inf saturates to -0x7f7fffff (0xff7fffff).
  ASSERT_EQ(rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64_gfx950(
                RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | RJ_ARITH_FP16_OVFL, 0, d, a, b, c),
            RJ_ARITH_SUCCESS);
  EXPECT_EQ(out[0][0], 0xff7fffffu);

  // 2) MI350X-captured OCP BF8 16x16x32 rows (gfx950_mfma.bin) pinning both sides of ce < te - 25:
  auto run_bf8_16x16x32 = [](const uint8_t a_row[32], const uint8_t b_col[32], uint32_t c0) {
    std::array<std::array<uint32_t, 64>, 8> buf{};
    for (int k = 0; k < 32; ++k) {
      const int lane = (k / 8) * 16;
      const int reg = (k % 8) / 4;
      const int shift = 8 * (k % 4);
      buf[reg][lane] |= uint32_t(a_row[k]) << shift;
      buf[2 + reg][lane] |= uint32_t(b_col[k]) << shift;
    }
    buf[4][0] = c0;
    const uint32_t *pa[] = {buf[0].data(), buf[1].data()};
    const uint32_t *pb[] = {buf[2].data(), buf[3].data()};
    const uint32_t *pc[] = {buf[4].data(), buf[5].data(), buf[6].data(), buf[7].data()};
    uint32_t res[4][64]{};
    uint32_t *pd[] = {res[0], res[1], res[2], res[3]};
    EXPECT_EQ(rj_arith_v_mfma_f32_16x16x32_bf8_bf8_w64_gfx950(
                  RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0, pd, pa, pb, pc),
              RJ_ARITH_SUCCESS);
    return res[0][0];
  };

  // At cutoff (ce == te - 25, w=20 r=4 c=3, C=-1.0f): MI350X returns 0x4c1f7ab7 (tightening cutoff
  // to ce < te - 24 would return 0x4c1f7ab8).
  const uint8_t keep25_a[32] = {0x5e, 0x4d, 0x43, 0xc2, 0x5b, 0x4a, 0x82, 0xc5, 0x69, 0xc7, 0x99,
                                0xc2, 0xda, 0x3a, 0xc1, 0xc0, 0xc2, 0x45, 0xb3, 0xc1, 0xad, 0xb1,
                                0xab, 0x42, 0x59, 0xb1, 0xf2, 0xc3, 0xe6, 0xc1, 0x34, 0xb6};
  const uint8_t keep25_b[32] = {0x23, 0xbf, 0x2f, 0xb2, 0xc6, 0xbb, 0x84, 0xbb, 0x74, 0xbf, 0x9f,
                                0x3d, 0x24, 0xc5, 0x30, 0x32, 0x16, 0xc0, 0x3e, 0xce, 0x84, 0xce,
                                0x04, 0xb6, 0xe3, 0x34, 0x92, 0xc3, 0x44, 0x3b, 0x7b, 0xb4};
  EXPECT_EQ(run_bf8_16x16x32(keep25_a, keep25_b, 0xbf800000u), 0x4c1f7ab7u);

  // Below cutoff (ce == te - 26, w=21 r=10 c=1, C=-1.0f): MI350X drops C and returns 0x4a84500f
  // (loosening cutoff to ce < te - 26 or disabling it would return 0x4a845007).
  const uint8_t drop26_a[32] = {0x9b, 0xbc, 0x7a, 0xcb, 0x3d, 0xb5, 0xe9, 0xc7, 0x3c, 0x4b, 0x91,
                                0x37, 0x22, 0x4a, 0x3a, 0x4b, 0x17, 0xc6, 0xcd, 0xb1, 0x93, 0x33,
                                0x2d, 0xb3, 0x19, 0x34, 0x4e, 0x48, 0x82, 0x40, 0x2e, 0xcd};
  const uint8_t drop26_b[32] = {0x5c, 0xca, 0xe8, 0x4b, 0x98, 0x3a, 0xf9, 0x35, 0xe6, 0x4b, 0x9e,
                                0x3e, 0x77, 0x33, 0x5a, 0x42, 0x50, 0xcd, 0xef, 0xbf, 0x31, 0x47,
                                0xc1, 0x37, 0x8e, 0x4e, 0xd3, 0x38, 0x39, 0x3c, 0xa4, 0x39};
  EXPECT_EQ(run_bf8_16x16x32(drop26_a, drop26_b, 0xbf800000u), 0x4a84500fu);
}

TEST(MfmaGfx950Exact, Mi350xCapturedHalfAlignmentAndPerPassFp16Ovfl) {
  auto run_f16_32x32x8 = [](const uint16_t a_row[8], const uint16_t b_col[8], uint32_t c0) {
    std::array<std::array<uint32_t, 64>, 20> st{};
    for (int k = 0; k < 8; ++k) {
      const int lane = (k / 4) * 32;
      const int reg = (k % 4) / 2;
      const int shift = 16 * (k % 2);
      st[reg][lane] |= uint32_t(a_row[k]) << shift;
      st[2 + reg][lane] |= uint32_t(b_col[k]) << shift;
    }
    st[4][0] = c0;
    const uint32_t *a[] = {st[0].data(), st[1].data()};
    const uint32_t *b[] = {st[2].data(), st[3].data()};
    const uint32_t *c[16];
    uint32_t *d[16];
    std::array<std::array<uint32_t, 64>, 16> out{};
    for (int r = 0; r < 16; ++r) {
      c[r] = st[4 + r].data();
      d[r] = out[r].data();
    }
    EXPECT_EQ(rj_arith_v_mfma_f32_32x32x8_f16_w64_gfx950(
                  RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0, d, a, b, c),
              RJ_ARITH_SUCCESS);
    return out[0][0];
  };

  // 0) Finite F16 24-bit alignment independent of BF16:
  //    - Single-pass cancellation 1*1 + 2^-14*1 - 1*1 -> 2^-14 (0x38800000 vs 0 at 13-bit align).
  //    - MI350X capture row (gfx950_mfma.bin, v_mfma_f32_32x32x8_f16 w=5 row=1 col=4): MI350X
  //      returns 0x477fe47e (vs 0x477fe000 at 13-bit align and 0x477fe480 under sequential FMA).
  const uint16_t cancel_f16_a[8] = {0x3c00, 0x0400, 0xbc00, 0, 0, 0, 0, 0};
  const uint16_t cancel_f16_b[8] = {0x3c00, 0x3c00, 0x3c00, 0, 0, 0, 0, 0};
  EXPECT_EQ(run_f16_32x32x8(cancel_f16_a, cancel_f16_b, 0u), 0x38800000u);

  const uint16_t f16_align_a[8] = {0x3800, 0x03ff, 0x0001, 0x8000, 0xbc00, 0x8000, 0x7bff, 0x0000};
  const uint16_t f16_align_b[8] = {0x3c01, 0x7bff, 0xbc00, 0x3c01, 0xfbff, 0x3c01, 0x0001, 0x0400};
  EXPECT_EQ(run_f16_32x32x8(f16_align_a, f16_align_b, 0x80000010u), 0x477fe47eu);
  auto run_bf16_16x16x16 = [](const uint16_t a_row[16], const uint16_t b_col[16], uint32_t c0,
                              uint64_t flags) {
    std::array<std::array<uint32_t, 64>, 8> st{};
    for (int k = 0; k < 16; ++k) {
      const int lane = (k / 4) * 16;
      const int reg = (k % 4) / 2;
      const int shift = 16 * (k % 2);
      st[reg][lane] |= uint32_t(a_row[k]) << shift;
      st[2 + reg][lane] |= uint32_t(b_col[k]) << shift;
    }
    st[4][0] = c0;
    const uint32_t *a[] = {st[0].data(), st[1].data()};
    const uint32_t *b[] = {st[2].data(), st[3].data()};
    const uint32_t *c[] = {st[4].data(), st[5].data(), st[6].data(), st[7].data()};
    uint32_t out[4][64]{};
    uint32_t *d[] = {out[0], out[1], out[2], out[3]};
    EXPECT_EQ(rj_arith_v_mfma_f32_16x16x16_bf16_w64_gfx950(flags, 0, d, a, b, c), RJ_ARITH_SUCCESS);
    return out[0][0];
  };

  // 1) Finite MI350X capture row (gfx950_mfma.bin, v_mfma_f32_16x16x16_bf16 w=4 row=1 col=10)
  // distinguishing 24-bit half/BF16 chunk alignment (0x7f7fc27f) from 13-bit alignment
  // (0x7f7fc27d).
  const uint16_t align_a[16] = {0x3c00, 0x0001, 0x0400, 0xfc09, 0x8000, 0x7bff, 0x3c00, 0x7bff,
                                0x03ff, 0xbc00, 0x7e05, 0x7c01, 0x3800, 0x0001, 0x0000, 0x0001};
  const uint16_t align_b[16] = {0x7bff, 0x7c01, 0xfc09, 0x03ff, 0x7c00, 0x0400, 0xfe03, 0x3800,
                                0xfbff, 0x3800, 0x0400, 0x8001, 0x3800, 0x0400, 0x3c00, 0xfbff};
  EXPECT_EQ(run_bf16_16x16x16(align_a, align_b, 0x7f7fffffu,
                              RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS),
            0x7f7fc27fu);

  // 2) Per-pass NaN flush under FP16_OVFL (gfx950_mfma.bin, v_mfma_f32_16x16x16_bf16 w=4 row=0
  // col=2): pass 0 produces Inf - Inf -> NaN (0xffc00000 when ovfl=0); under ovfl=1 pass 0 flushes
  // to +0.0f so pass 1's finite BF16 products accumulate to 0x00000202.
  const uint16_t nan_a[16] = {0xfc00, 0xfc00, 0x8000, 0x3c00, 0xfc09, 0x7e05, 0x0001, 0x8001,
                              0x0001, 0x0400, 0x0000, 0x3c00, 0x0000, 0xbc00, 0x03ff, 0x03ff};
  const uint16_t nan_b[16] = {0x0000, 0x0400, 0x7c01, 0x8000, 0xfbff, 0xfe03, 0xfbff, 0x7e05,
                              0x3800, 0x0000, 0x0001, 0x0001, 0x7c00, 0x0000, 0x03ff, 0x8001};
  EXPECT_EQ(run_bf16_16x16x16(nan_a, nan_b, 0x7f7fffffu,
                              RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS),
            0xffc00000u);
  EXPECT_EQ(
      run_bf16_16x16x16(nan_a, nan_b, 0x7f7fffffu,
                        RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | RJ_ARITH_FP16_OVFL),
      0x00000202u);

  // 3) Finite-overflow saturation under FP16_OVFL (gfx950_mfma.bin, v_mfma_f32_16x16x16_bf16 w=0
  // row=0 col=0): 0x7f800000 when ovfl=0 vs 0x7f7fffff when ovfl=1.
  const uint16_t ovfl_ab[16] = {0x7bff, 0x7bff, 0x7bff, 0x7bff, 0x7bff, 0x7bff, 0x7bff, 0x7bff,
                                0x7bff, 0x7bff, 0x7bff, 0x7bff, 0x7bff, 0x7bff, 0x7bff, 0x7bff};
  EXPECT_EQ(run_bf16_16x16x16(ovfl_ab, ovfl_ab, 0x7f7fffffu,
                              RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS),
            0x7f800000u);
  EXPECT_EQ(
      run_bf16_16x16x16(ovfl_ab, ovfl_ab, 0x7f7fffffu,
                        RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | RJ_ARITH_FP16_OVFL),
      0x7f7fffffu);

  // 4) Source +Inf clamping under FP16_OVFL in exact modes vs Loose/Compatibility preservation
  // (gfx950_mfma.bin, v_mfma_f32_16x16x4_4b_bf16 w=6 batch=0 row=4 col=0): A[2]=0x7f80 (+Inf) *
  // B[2]=0x7f7f + 1.0f -> 0x7f800000 (ovfl=0 and Loose/Compatibility ovfl=1) vs 0x7f7fffff
  // (exact ovfl=1).
  std::array<std::array<uint32_t, 64>, 20> st4{};
  st4[0][0] = 0x00808001u;
  st4[1][0] = 0x80007f80u;
  st4[2][0] = 0x43000000u;
  st4[3][0] = 0x00007f7fu;
  st4[4][0] = 0x3f800000u;
  const uint32_t *a4[] = {st4[0].data(), st4[1].data()};
  const uint32_t *b4[] = {st4[2].data(), st4[3].data()};
  const uint32_t *c4[16];
  uint32_t *d4[16];
  std::array<std::array<uint32_t, 64>, 16> out4{};
  for (int r = 0; r < 16; ++r) {
    c4[r] = st4[4 + r].data();
    d4[r] = out4[r].data();
  }
  ASSERT_EQ(rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64_gfx950(
                RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0, d4, a4, b4, c4),
            RJ_ARITH_SUCCESS);
  EXPECT_EQ(out4[0][0], 0x7f800000u);
  ASSERT_EQ(rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64_gfx950(
                RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | RJ_ARITH_FP16_OVFL, 0, d4, a4,
                b4, c4),
            RJ_ARITH_SUCCESS);
  EXPECT_EQ(out4[0][0], 0x7f7fffffu);
  ASSERT_EQ(rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64_gfx950(
                RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP16_OVFL, 0, d4, a4, b4, c4),
            RJ_ARITH_SUCCESS);
  EXPECT_EQ(out4[0][0], 0x7f800000u);
}

TEST(MfmaGfx950Exact, All24SparseFormsSupportExactModesControlsAndOvfl) {
  for (const auto &form : kSparseForms) {
    SCOPED_TRACE(form.name);
    const bool half = form.format_a < 2;
    const int a_regs = form.m * form.k * (half ? 16 : 8) / 4096;
    const int b_regs = 2 * a_regs;
    const int d_regs = form.m * form.m / 64;
    const int set_bits = form.m * form.k / 64;

    std::array<std::array<uint32_t, 64>, 29> input{};
    // Fill A with NaN in lane 0 word 0 and 1.0 elsewhere, B with 1.0 everywhere, metadata with
    // valid selector nibbles 0x4 (selects indices 0 and 1 in every 4-group).
    const uint32_t one_a = code_one(form.format_a) * (half ? 0x10001u : 0x01010101u);
    const uint32_t one_b = code_one(form.format_b) * (half ? 0x10001u : 0x01010101u);
    for (int r = 0; r < a_regs; ++r) {
      input[r].fill(one_a);
    }
    input[0][0] = (one_a & (half ? 0xffff0000u : 0xffffff00u)) | code_nan(form.format_a);
    for (int r = 0; r < b_regs; ++r) {
      input[a_regs + r].fill(one_b);
    }

    for (uint64_t controls : {0ULL, 8ULL, 16ULL, 9ULL}) {
      const int shift = (controls & 3) == 0 && set_bits < 32
                            ? set_bits * int((controls >> 3) & (32 / set_bits - 1))
                            : 0;
      const uint32_t mask = set_bits == 32 ? ~0u : ((1u << set_bits) - 1u);
      input[a_regs + b_regs].fill(((0x44444444u & mask) << shift) |
                                  (~(mask << shift) & 0xeeeeeeeeu));

      for (uint64_t cpu : supported_cpu_levels()) {
        for (uint64_t sem : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                             RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                             RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
          for (bool ovfl : {false, true}) {
            auto st = input;
            const uint32_t *a[4], *b[8], *idx[] = {st[a_regs + b_regs].data()};
            uint32_t *d[16];
            for (int r = 0; r < a_regs; ++r) {
              a[r] = st[r].data();
            }
            for (int r = 0; r < b_regs; ++r) {
              b[r] = st[a_regs + r].data();
            }
            for (int r = 0; r < d_regs; ++r) {
              d[r] = st[a_regs + b_regs + 1 + r].data();
            }
            ASSERT_EQ(
                form.entry(cpu | sem | (ovfl ? RJ_ARITH_FP16_OVFL : 0), controls, d, a, b, idx),
                RJ_ARITH_SUCCESS);
            if (!ovfl) {
              EXPECT_EQ(d[0][0], 0xffc00000u);
            } else {
              // Pass 0's NaN flushes to +0.0; subsequent 1.0*1.0 passes accumulate cleanly!
              const int pps = half ? 8 : 16;
              const int rem_products = (form.k / 2) - pps;
              float expected = float(rem_products > 0 ? rem_products : 0);
              EXPECT_EQ(d[0][0], std::bit_cast<uint32_t>(expected));
            }
          }
        }
      }
    }
  }
}

} // namespace
