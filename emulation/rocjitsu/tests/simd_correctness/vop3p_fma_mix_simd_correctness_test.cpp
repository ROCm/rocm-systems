// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3p_fma_mix_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the six VOP3P
/// fma_mix / mad_mix mixed-precision ops. Each case runs TWICE in the same
/// process -- once forcing the scalar body, once the SIMD fast path, with
/// identical inputs/EXEC -- and the results are asserted equal per active,
/// non-skipped lane (util::set_force_scalar_for_testing flips the gate
/// in-process). NaN-input lanes carry an accepted NaN-payload divergence and are
/// excluded from the comparison — the skip condition is computed from the
/// inputs, identical in both runs, so both skip the same lanes. In-process
/// inactive lanes must keep the dst seed. The six ops:
///   - RDNA3+ : v_fma_mix_f32 / v_fma_mixlo_f16 / v_fma_mixhi_f16
///   - CDNA1-4: v_mad_mix_f32 / v_mad_mixlo_f16 / v_mad_mixhi_f16
///
/// All six implement the same scalar formula (`a * b + c` plain, no std::fma,
/// then optional clamp to [0, 1]) and differ only in (a) the per-source f16
/// widening shape selected by `op_sel_hi` + `op_sel_hi_2` (widen f16 half to
/// f32 vs read raw f32) and the f16-half pick selected by `op_sel`, and
/// (b) the destination narrowing shape (f32 / f16-lo / f16-hi). The field
/// named `neg_hi` in the shared VOP3P layout carries per-source abs for this
/// mix-family encoding, `neg` applies per-source sign xor, and `clamp`
/// saturates the result to [0, 1].
///
/// The test covers every per-source neg combination, several op_sel /
/// op_sel_hi patterns (so all four widen-modes per source are exercised),
/// clamp on/off, full and partial EXEC masks. Input lanes carry an f32 edge
/// list packed both as raw f32 (when read in the no-widen mode) and as two
/// f16 halves (when the corresponding op_sel_hi bit is set), so every
/// per-source data fetch path is hit. NaN-input lanes are skipped because
/// the scalar `a * b + c` and the SIMD chain can quiet sNaN to qNaN with
/// different payloads (accepted divergence, mirroring the existing VOP3
/// ternary tests).

#include "util/simd_test_hooks.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/execute_shared.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "util/data_types.h"
#include "util/simd.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>

namespace {

using namespace rocjitsu;

constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t kDstVgpr = 6;
constexpr uint32_t DST_SENTINEL = 0xCDCDCDCDu;

// Restores the process force-scalar gate on scope exit so flipping it for an
// in-process A/B comparison cannot leak into later tests in the same process.
struct ForceScalarGuard {
  bool orig;
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }
};

// VOP3P encoding word0 layout (shared between CDNA4 and RDNA3 except for the
// width of the encoding marker): bits 0-7 vdst, 8-10 neg_hi, 11-13 op_sel,
// 14 op_sel_hi_2, 15 clamp, 16-22 op (7 bits), 23 pad (RDNA3) / encoding
// extension (CDNA4), 24-31 encoding (RDNA3 = 0xCC) / 23-31 encoding (CDNA4 =
// 0x1A7 = primary_decode_table slot that dispatches to subDecodeVop3p).
// word1 layout (identical): bits 0-8 src0, 9-17 src1, 18-26 src2,
// 27-28 op_sel_hi (src0/src1), 29-31 neg.
constexpr void vop3p_encode_rdna3(uint32_t op, uint32_t vdst, uint32_t neg_hi, uint32_t op_sel,
                                  uint32_t op_sel_hi_2, uint32_t clamp, uint32_t src0,
                                  uint32_t src1, uint32_t src2, uint32_t op_sel_hi, uint32_t neg,
                                  uint32_t words[2]) {
  words[0] = (vdst & 0xFFu) | ((neg_hi & 0x7u) << 8) | ((op_sel & 0x7u) << 11) |
             ((op_sel_hi_2 & 0x1u) << 14) | ((clamp & 0x1u) << 15) | ((op & 0x7Fu) << 16) |
             (0xCCu << 24);
  words[1] = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18) |
             ((op_sel_hi & 0x3u) << 27) | ((neg & 0x7u) << 29);
}

constexpr void vop3p_encode_cdna4(uint32_t op, uint32_t vdst, uint32_t neg_hi, uint32_t op_sel,
                                  uint32_t op_sel_hi_2, uint32_t clamp, uint32_t src0,
                                  uint32_t src1, uint32_t src2, uint32_t op_sel_hi, uint32_t neg,
                                  uint32_t words[2]) {
  words[0] = (vdst & 0xFFu) | ((neg_hi & 0x7u) << 8) | ((op_sel & 0x7u) << 11) |
             ((op_sel_hi_2 & 0x1u) << 14) | ((clamp & 0x1u) << 15) | ((op & 0x7Fu) << 16) |
             (0x1A7u << 23);
  words[1] = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18) |
             ((op_sel_hi & 0x3u) << 27) | ((neg & 0x7u) << 29);
}

// Edge f32 bit patterns + an "everyday" gradient. Every lane carries one of
// these in src0/src1/src2 (rotated per source). Includes ±0, ±denormal,
// ±finite, ±Inf — NaN inputs are excluded so the lane-wise comparator does
// not need a NaN-payload carve-out.
const std::array<uint32_t, 16> kSrcA = {{
    0x3F800000u, // 1.0
    0xBF800000u, // -1.0
    0x40400000u, // 3.0
    0xC0400000u, // -3.0
    0x00000000u, // +0.0
    0x80000000u, // -0.0
    0x7F800000u, // +Inf
    0xFF800000u, // -Inf
    0x40490FDBu, // pi
    0xC0490FDBu, // -pi
    0x3DCCCCCDu, // 0.1
    0xBDCCCCCDu, // -0.1
    0x00400000u, // small denormal-ish
    0x80400000u, // negative ditto
    0x42280000u, // 42.0
    0xC2280000u, // -42.0
}};
const std::array<uint32_t, 16> kSrcB = {{
    0x3FC00000u, // 1.5
    0xBFC00000u, // -1.5
    0x3F000000u, // 0.5
    0xBF000000u, // -0.5
    0x40000000u, // 2.0
    0xC0000000u, // -2.0
    0x3F800000u, // 1.0
    0xBF800000u, // -1.0
    0x3F4CCCCDu, // 0.8
    0xBF4CCCCDu, // -0.8
    0x44800000u, // 1024
    0xC4800000u, // -1024
    0x3E800000u, // 0.25
    0xBE800000u, // -0.25
    0x40A00000u, // 5.0
    0xC0A00000u, // -5.0
}};
const std::array<uint32_t, 16> kSrcC = {{
    0x00000000u,
    0x3F800000u,
    0xBF800000u,
    0x3DCCCCCDu,
    0xBDCCCCCDu,
    0x40000000u,
    0xC0000000u,
    0x40A00000u,
    0xC0A00000u,
    0x3F4CCCCDu,
    0xBF4CCCCDu,
    0x42280000u,
    0xC2280000u,
    0x80000000u,
    0x3F000000u,
    0xBF000000u,
}};

// Per-source f16 packing: low 16 of `kSrcA[i]` is one f16 sample, high 16 is
// the next sample's first half. When op_sel_hi=1 + op_sel selects low vs high
// the scalar widens via util::f16_to_f32 — these patterns make sure both
// halves carry distinct finite f16 values across every lane.
constexpr uint32_t pack_two_f16(uint32_t lo16, uint32_t hi16) {
  return (lo16 & 0xFFFFu) | ((hi16 & 0xFFFFu) << 16);
}

const std::array<uint32_t, 16> kSrcA_f16 = {{
    pack_two_f16(0x3C00u, 0xBC00u), // 1.0, -1.0
    pack_two_f16(0x4200u, 0xC200u), // 3.0, -3.0
    pack_two_f16(0x0000u, 0x8000u), // +0, -0
    pack_two_f16(0x7C00u, 0xFC00u), // +Inf, -Inf  (Inf only, no NaN)
    pack_two_f16(0x4900u, 0xC900u), // ~pi, ~-pi
    pack_two_f16(0x2E66u, 0xAE66u), // ~0.1, ~-0.1
    pack_two_f16(0x0001u, 0x8001u), // smallest denormal both signs
    pack_two_f16(0x5140u, 0xD140u), // 42.0, -42.0
    pack_two_f16(0x3800u, 0xB800u), // 0.5, -0.5
    pack_two_f16(0x4000u, 0xC000u), // 2.0, -2.0
    pack_two_f16(0x33CCu, 0xB3CCu), // 0.25-ish, neg
    pack_two_f16(0x6400u, 0xE400u), // 1024-ish, neg
    pack_two_f16(0x3C00u, 0x4200u), // 1.0, 3.0
    pack_two_f16(0x0001u, 0x3C00u), // denormal then 1.0
    pack_two_f16(0x2E66u, 0xBC00u), // 0.1, -1.0
    pack_two_f16(0x3800u, 0x4200u), // 0.5, 3.0
}};

// Same f16 pool but rotated so every lane sees a different src0/src1/src2
// combination across rotations.
const std::array<uint32_t, 16> kSrcB_f16 = kSrcA_f16;
const std::array<uint32_t, 16> kSrcC_f16 = kSrcA_f16;

// Modifier combinations swept per op. (neg, op_sel, op_sel_hi, op_sel_hi_2,
// clamp, abs). op_sel only matters when the matching op_sel_hi[i] bit is set.
// Eight neg patterns × handful of op_sel / widen-mode combos × clamp on/off,
// plus abs-only and abs+neg cases.
struct ModCombo {
  uint32_t neg;         // bit0=src0, bit1=src1, bit2=src2
  uint32_t op_sel;      // bit0=src0 lo/hi, bit1=src1 lo/hi, bit2=src2 lo/hi
  uint32_t op_sel_hi;   // bit0=src0 widen, bit1=src1 widen
  uint32_t op_sel_hi_2; // 0/1: src2 widen
  uint32_t clamp;       // 0/1
  uint32_t abs = 0;     // bit0=src0, bit1=src1, bit2=src2
};

constexpr std::array<ModCombo, 36> kModCombos = {{
    // All-defaults: read every src as raw f32, no neg, no clamp.
    {0b000, 0b000, 0b00, 0, 0},
    // Per-source neg.
    {0b001, 0b000, 0b00, 0, 0},
    {0b010, 0b000, 0b00, 0, 0},
    {0b100, 0b000, 0b00, 0, 0},
    {0b111, 0b000, 0b00, 0, 0},
    // Clamp on/off with neg combos.
    {0b000, 0b000, 0b00, 0, 1},
    {0b101, 0b000, 0b00, 0, 1},
    {0b011, 0b000, 0b00, 0, 1},
    // Widen src0 from f16 (lo half).
    {0b000, 0b000, 0b01, 0, 0},
    {0b000, 0b001, 0b01, 0, 0}, // widen src0 hi half
    // Widen src1 from f16.
    {0b000, 0b000, 0b10, 0, 0},
    {0b000, 0b010, 0b10, 0, 0},
    // Widen src2 from f16.
    {0b000, 0b000, 0b00, 1, 0},
    {0b000, 0b100, 0b00, 1, 0},
    // Widen all three.
    {0b000, 0b000, 0b11, 1, 0},
    {0b000, 0b111, 0b11, 1, 0},
    // Widen all three + neg all + clamp.
    {0b111, 0b000, 0b11, 1, 1},
    {0b111, 0b111, 0b11, 1, 1},
    // Mixed: widen src0+src2, neg src1.
    {0b010, 0b000, 0b01, 1, 0},
    {0b010, 0b101, 0b01, 1, 0},
    // Widen src1+src2 only.
    {0b000, 0b000, 0b10, 1, 0},
    {0b101, 0b110, 0b10, 1, 0},
    // Cover op_sel without widen (op_sel bit ignored by scalar/SIMD on that lane).
    {0b000, 0b111, 0b00, 0, 0},
    {0b111, 0b111, 0b00, 0, 0},
    // Clamp + widen combos for f16 result paths.
    {0b001, 0b001, 0b01, 0, 1},
    {0b010, 0b010, 0b10, 0, 1},
    {0b100, 0b100, 0b00, 1, 1},
    // Edge: neg + widen src0 hi only.
    {0b001, 0b001, 0b01, 0, 0},
    {0b001, 0b001, 0b11, 1, 0},
    // Edge: clamp + widen mixed.
    {0b011, 0b010, 0b11, 1, 1},
    {0b110, 0b110, 0b10, 1, 1},
    {0b101, 0b101, 0b01, 1, 1},
    // Mix-family abs modifiers are encoded in the shared field named neg_hi.
    {0b000, 0b000, 0b00, 0, 0, 0b001},
    {0b000, 0b000, 0b00, 0, 0, 0b110},
    {0b101, 0b000, 0b00, 0, 0, 0b101},
    {0b011, 0b111, 0b11, 1, 1, 0b111},
}};

template <uint32_t WF_SIZE, int ArchTag> struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3p_fma_mix_mem"), l2("vop3p_fma_mix_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = (ArchTag == 0) ? ROCJITSU_CODE_ARCH_RDNA3 : ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3p_fma_mix", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(cfg.arch);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_vgprs(uint32_t rot, bool widen0, bool widen1, bool widen2, uint64_t exec,
                  uint32_t dst_seed) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint32_t i0 = (lane + rot) % kSrcA.size();
      uint32_t i1 = (lane + 2 * rot + 3) % kSrcA.size();
      uint32_t i2 = (lane + 3 * rot + 7) % kSrcA.size();
      cu->write_vgpr(vb + 0, lane, widen0 ? kSrcA_f16[i0] : kSrcA[i0]);
      cu->write_vgpr(vb + 1, lane, widen1 ? kSrcB_f16[i1] : kSrcB[i1]);
      cu->write_vgpr(vb + 2, lane, widen2 ? kSrcC_f16[i2] : kSrcC[i2]);
      cu->write_vgpr(vb + kDstVgpr, lane, dst_seed);
    }
    wf->set_exec(exec);
  }

  std::array<uint32_t, WF_SIZE> run(Instruction *inst, uint32_t rot, bool widen0, bool widen1,
                                    bool widen2, uint64_t exec, uint32_t dst_seed) {
    seed_vgprs(rot, widen0, widen1, widen2, exec, dst_seed);
    cu->execute_instruction(inst, *wf);
    std::array<uint32_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vb + kDstVgpr, lane);
    return out;
  }
};

// Skip lanes whose three (widened-if-applicable) input operands include NaN.
// We do NOT carry a NaN-payload guarantee through the f32 intermediate (the
// scalar `a*b+c` and the SIMD chain can pick different operand to quiet).
bool input_lane_is_nan(uint32_t raw0, uint32_t raw1, uint32_t raw2, bool widen0, bool widen1,
                       bool widen2, uint32_t op_sel) {
  auto half_is_nan = [](uint32_t bits16) {
    uint32_t exp = (bits16 >> 10) & 0x1Fu;
    uint32_t mant = bits16 & 0x3FFu;
    return exp == 0x1Fu && mant != 0u;
  };
  auto f32_is_nan = [](uint32_t bits) {
    uint32_t exp = (bits >> 23) & 0xFFu;
    uint32_t mant = bits & 0x7FFFFFu;
    return exp == 0xFFu && mant != 0u;
  };
  auto src_is_nan = [&](uint32_t raw, bool widen, uint32_t sel_bit) {
    if (widen) {
      uint32_t h = sel_bit ? (raw >> 16) : (raw & 0xFFFFu);
      return half_is_nan(h);
    }
    return f32_is_nan(raw);
  };
  return src_is_nan(raw0, widen0, op_sel & 1u) || src_is_nan(raw1, widen1, (op_sel >> 1) & 1u) ||
         src_is_nan(raw2, widen2, (op_sel >> 2) & 1u);
}

template <uint32_t WF_SIZE, int ArchTag>
void check_combo(uint32_t opcode, const ModCombo &mc, uint64_t exec, uint32_t dst_seed,
                 const char *label) {
  const bool widen0 = (mc.op_sel_hi & 1u) != 0;
  const bool widen1 = ((mc.op_sel_hi >> 1) & 1u) != 0;
  const bool widen2 = mc.op_sel_hi_2 != 0;

  auto run_mode = [&](bool force_scalar, uint32_t rot) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture<WF_SIZE, ArchTag> fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    if constexpr (ArchTag == 0) {
      vop3p_encode_rdna3(opcode, kDstVgpr, /*neg_hi=*/mc.abs, mc.op_sel, mc.op_sel_hi_2, mc.clamp,
                         /*src0=*/256, /*src1=*/257, /*src2=*/258, mc.op_sel_hi, mc.neg, words);
    } else {
      vop3p_encode_cdna4(opcode, kDstVgpr, /*neg_hi=*/mc.abs, mc.op_sel, mc.op_sel_hi_2, mc.clamp,
                         /*src0=*/256, /*src1=*/257, /*src2=*/258, mc.op_sel_hi, mc.neg, words);
    }
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << label << " decode failed";
    auto out = fx.run(inst, rot, widen0, widen1, widen2, exec, dst_seed);
    delete inst;
    return out;
  };

  for (uint32_t rot = 0; rot < 4; ++rot) {
    const auto scalar_out = run_mode(/*force_scalar=*/true, rot);
    const auto simd_out = run_mode(/*force_scalar=*/false, rot);

    // Core A/B equivalence per active, non-skipped lane. NaN-input lanes carry an
    // accepted NaN-payload divergence and are excluded identically in both runs
    // (the skip condition is input-derived).
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (active) {
        uint32_t i0 = (lane + rot) % kSrcA.size();
        uint32_t i1 = (lane + 2 * rot + 3) % kSrcA.size();
        uint32_t i2 = (lane + 3 * rot + 7) % kSrcA.size();
        uint32_t raw0 = widen0 ? kSrcA_f16[i0] : kSrcA[i0];
        uint32_t raw1 = widen1 ? kSrcB_f16[i1] : kSrcB[i1];
        uint32_t raw2 = widen2 ? kSrcC_f16[i2] : kSrcC[i2];
        if (input_lane_is_nan(raw0, raw1, raw2, widen0, widen1, widen2, mc.op_sel))
          continue;
        EXPECT_EQ(scalar_out[lane], simd_out[lane])
            << label << " rot=" << rot << " lane=" << lane << ": SIMD path diverged from scalar";
      } else {
        EXPECT_EQ(simd_out[lane], dst_seed)
            << label << " rot=" << rot << " lane=" << lane << ": clobbered inactive lane";
        EXPECT_EQ(scalar_out[lane], dst_seed)
            << label << " rot=" << rot << " lane=" << lane << ": clobbered inactive lane";
      }
    }
  }
}

template <uint32_t WF_SIZE, int ArchTag>
void check_op(uint32_t opcode, uint64_t exec, uint32_t dst_seed, const char *label) {
  ForceScalarGuard gate_guard;
  for (const auto &mc : kModCombos)
    check_combo<WF_SIZE, ArchTag>(opcode, mc, exec, dst_seed, label);
}

constexpr uint32_t kVop3pOpFmaMixF32 = 32;
constexpr uint32_t kVop3pOpFmaMixLoF16 = 33;
constexpr uint32_t kVop3pOpFmaMixHiF16 = 34;
constexpr uint32_t kVop3pOpMadMixF32 = 32;
constexpr uint32_t kVop3pOpMadMixLoF16 = 33;
constexpr uint32_t kVop3pOpMadMixHiF16 = 34;

constexpr uint64_t kFullExecRdna = 0xFFFFFFFFULL;
constexpr uint64_t kPartialExecRdna = 0xA5A5F0F0ULL;
constexpr uint64_t kFullExecCdna = 0xFFFFFFFFFFFFFFFFULL;
constexpr uint64_t kPartialExecCdna = 0xA5A5F0F012348001ULL;

// --- RDNA3: v_fma_mix_* ---

TEST(Vop3pFmaMixSimdCorrectness, RdnaFmaMixF32_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_op<32, 0>(kVop3pOpFmaMixF32, kFullExecRdna, DST_SENTINEL, "v_fma_mix_f32");
}
TEST(Vop3pFmaMixSimdCorrectness, RdnaFmaMixF32_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_op<32, 0>(kVop3pOpFmaMixF32, kPartialExecRdna, DST_SENTINEL, "v_fma_mix_f32");
}
TEST(Vop3pFmaMixSimdCorrectness, RdnaFmaMixLoF16_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_op<32, 0>(kVop3pOpFmaMixLoF16, kFullExecRdna, DST_SENTINEL, "v_fma_mixlo_f16");
}
TEST(Vop3pFmaMixSimdCorrectness, RdnaFmaMixLoF16_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_op<32, 0>(kVop3pOpFmaMixLoF16, kPartialExecRdna, DST_SENTINEL, "v_fma_mixlo_f16");
}
TEST(Vop3pFmaMixSimdCorrectness, RdnaFmaMixHiF16_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_op<32, 0>(kVop3pOpFmaMixHiF16, kFullExecRdna, DST_SENTINEL, "v_fma_mixhi_f16");
}
TEST(Vop3pFmaMixSimdCorrectness, RdnaFmaMixHiF16_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_op<32, 0>(kVop3pOpFmaMixHiF16, kPartialExecRdna, DST_SENTINEL, "v_fma_mixhi_f16");
}

// --- CDNA4: v_mad_mix_* ---

TEST(Vop3pFmaMixSimdCorrectness, CdnaMadMixF32_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_op<64, 1>(kVop3pOpMadMixF32, kFullExecCdna, DST_SENTINEL, "v_mad_mix_f32");
}
TEST(Vop3pFmaMixSimdCorrectness, CdnaMadMixF32_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_op<64, 1>(kVop3pOpMadMixF32, kPartialExecCdna, DST_SENTINEL, "v_mad_mix_f32");
}
TEST(Vop3pFmaMixSimdCorrectness, CdnaMadMixLoF16_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_op<64, 1>(kVop3pOpMadMixLoF16, kFullExecCdna, DST_SENTINEL, "v_mad_mixlo_f16");
}
TEST(Vop3pFmaMixSimdCorrectness, CdnaMadMixLoF16_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_op<64, 1>(kVop3pOpMadMixLoF16, kPartialExecCdna, DST_SENTINEL, "v_mad_mixlo_f16");
}
TEST(Vop3pFmaMixSimdCorrectness, CdnaMadMixHiF16_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_op<64, 1>(kVop3pOpMadMixHiF16, kFullExecCdna, DST_SENTINEL, "v_mad_mixhi_f16");
}
TEST(Vop3pFmaMixSimdCorrectness, CdnaMadMixHiF16_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_op<64, 1>(kVop3pOpMadMixHiF16, kPartialExecCdna, DST_SENTINEL, "v_mad_mixhi_f16");
}

} // namespace
