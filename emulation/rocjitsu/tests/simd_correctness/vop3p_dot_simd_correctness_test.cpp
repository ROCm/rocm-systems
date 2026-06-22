// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3p_dot_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the VOP3P
/// dot-product family on CDNA4. Each case runs TWICE in the same process -- once
/// forcing the scalar body, once the SIMD fast path, with identical inputs/EXEC
/// -- and the results are asserted equal per active, non-skipped lane
/// (util::set_force_scalar_for_testing flips the gate in-process). The f16 form's
/// NaN-input lanes carry a toolchain-dependent NaN-payload divergence and are
/// excluded from the comparison (the skip condition is computed from the inputs,
/// so both runs skip the same lanes). In-process inactive lanes must keep the
/// sentinel. Ops covered:
///   integer: v_dot4_i32_i8 / v_dot4_u32_u8 / v_dot8_i32_i4 / v_dot8_u32_u4 /
///            v_dot2_i32_i16 / v_dot2_u32_u16
///   float:   v_dot2_f32_f16
///
/// Unlike the packed-16 family the destination is a single 32-bit lane (NOT
/// packed) and src2 is a per-lane accumulator; the dot reduction happens
/// within each lane and the fast path vectorizes across lanes. The signed
/// integer forms lower-clamp to 0 when the clamp bit is set, so the integer
/// test sweeps clamp ∈ {0,1}; the f16 form half-selects via op_sel, sign-
/// flips via neg/neg_hi, and clamps to [0,1], so it sweeps every neg/neg_hi
/// combination × clamp with NaN-input lanes skipped (toolchain-dependent NaN
/// payload divergence, same carve-out as the pk_fma slices). Default packing
/// (op_sel = 0, op_sel_hi = 3) only — non-default modes bail to scalar.

#include "util/simd_test_hooks.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/execute_shared.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "util/simd.h"

#include <array>
#include <bit>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
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

// CDNA4 VOP3P dot encoding (all dot ops are 3-source). Default packing:
// op_sel = 0, op_sel_hi = 3, op_sel_hi_2 = 1. clamp -> word0 bit 15; neg ->
// word1 bits 29:31; neg_hi -> word0 bits 8:10.
constexpr void vop3p_encode_dot(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1,
                                uint32_t src2, uint32_t neg, uint32_t neg_hi, uint32_t clamp,
                                uint32_t words[2]) {
  words[0] = (vdst & 0xFFu) | ((neg_hi & 0x7u) << 8) | (1u << 14) | ((clamp & 0x1u) << 15) |
             ((op & 0x7Fu) << 16) | (0x1A7u << 23);
  words[1] = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18) | (0x3u << 27) |
             ((neg & 0x7u) << 29);
}

struct IntCase {
  const char *name;
  uint32_t opcode;
};

const std::array<IntCase, 6> kIntCases = {{
    {"v_dot4_i32_i8_vop3p", 40},
    {"v_dot4_u32_u8_vop3p", 41},
    {"v_dot8_i32_i4_vop3p", 42},
    {"v_dot8_u32_u4_vop3p", 43},
    {"v_dot2_i32_i16_vop3p", 38},
    {"v_dot2_u32_u16_vop3p", 39},
}};

// Byte/nibble/half-varied 32-bit words; src2 (the accumulator) is seeded from
// the same pool, so signed sums land both positive and negative to exercise
// the lower clamp.
const std::array<uint32_t, 14> kVals = {{
    0x00000000u,
    0x01020304u,
    0x0000FFFFu,
    0xFFFF0000u,
    0xFFFFFFFFu, // all-ones: -1 per signed sub-word
    0x80008000u, // INT16_MIN halves / INT_MIN as acc
    0x7FFF7FFFu, // INT16_MAX halves
    0x12345678u,
    0xDEADBEEFu,
    0xCAFEBABEu,
    0xA5A5A5A5u,
    0x7F7F7F7Fu,
    0x88888888u, // -8 per nibble (signed i4)
    0x0F0F0F0Fu,
}};

bool is_f16_nan(uint16_t b) { return ((b >> 10) & 0x1Fu) == 0x1Fu && (b & 0x3FFu) != 0u; }

// ---- integer dot products -------------------------------------------------

void check_int_case(const IntCase &c, uint64_t exec, uint32_t clamp) {
  ForceScalarGuard gate_guard;

  auto run_mode = [&](bool force_scalar, uint32_t rot) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    amdgpu::GpuMemory gpu_mem("vop3p_dot_int_mem");
    amdgpu::L2Cache l2("vop3p_dot_int_l2");
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("cu_vop3p_dot_int", cfg, &gpu_mem, &l2);
    EXPECT_NE(cu, nullptr);
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    amdgpu::Wavefront *wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
    EXPECT_NE(wf, nullptr);

    uint32_t words[2] = {0u, 0u};
    vop3p_encode_dot(c.opcode, kDstVgpr, /*src0=*/256, /*src1=*/257, /*src2=*/258, /*neg=*/0,
                     /*neg_hi=*/0, clamp, words);
    Instruction *inst = decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.name << " decode failed";

    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      cu->write_vgpr(vb + 0, lane, kVals[lane % kVals.size()]);
      cu->write_vgpr(vb + 1, lane, kVals[(lane + rot) % kVals.size()]);
      cu->write_vgpr(vb + 2, lane, kVals[(lane + 2 * rot + 5) % kVals.size()]);
      cu->write_vgpr(vb + kDstVgpr, lane, DST_SENTINEL);
    }
    wf->set_exec(exec);
    cu->execute_instruction(inst, *wf);
    std::array<uint32_t, WF_SIZE> out{};
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vb + kDstVgpr, lane);
    delete inst;
    return out;
  };

  for (uint32_t rot = 0; rot < kVals.size(); ++rot) {
    const auto scalar_out = run_mode(/*force_scalar=*/true, rot);
    const auto simd_out = run_mode(/*force_scalar=*/false, rot);

    EXPECT_EQ(scalar_out, simd_out) << c.name << " clamp=" << clamp << " rot=" << rot
                                    << ": SIMD path diverged from scalar body";

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (!active) {
        EXPECT_EQ(simd_out[lane], DST_SENTINEL) << c.name << " clamp=" << clamp << " rot=" << rot
                                                << ": clobbered inactive lane " << lane;
        EXPECT_EQ(scalar_out[lane], DST_SENTINEL) << c.name << " clamp=" << clamp << " rot=" << rot
                                                  << ": clobbered inactive lane " << lane;
      }
    }
  }
}

TEST(Vop3pDotIntSimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kIntCases)
    for (uint32_t clamp = 0; clamp < 2; ++clamp)
      check_int_case(c, /*exec=*/~0ULL, clamp);
}

TEST(Vop3pDotIntSimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kIntCases)
    for (uint32_t clamp = 0; clamp < 2; ++clamp)
      check_int_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL, clamp);
}

// ---- v_dot2_f32_f16 -------------------------------------------------------

const std::array<uint16_t, 14> kF16Bits = {{
    0x0000u, // +0
    0x8000u, // -0
    0x3C00u, // +1
    0xBC00u, // -1
    0x4200u, // +3
    0xC900u, // -10
    0x7BFFu, // +max-normal
    0xFBFFu, // -max-normal
    0x0400u, // +smallest-normal
    0x83FFu, // largest -denormal
    0x4900u, // +10
    0x4400u, // +4
    0x4500u, // +5
    0x3800u, // +0.5
}};

// Finite f32 accumulators (bit patterns), no NaN/Inf.
const std::array<float, 7> kF32Acc = {{0.0f, -0.0f, 1.0f, -2.5f, 100.0f, -1024.0f, 0.25f}};

void check_f16_case(uint64_t exec, uint32_t neg, uint32_t neg_hi, uint32_t clamp) {
  ForceScalarGuard gate_guard;

  auto half_bits = [&](uint32_t lane, uint32_t rot) {
    uint16_t a_lo = kF16Bits[lane % kF16Bits.size()];
    uint16_t a_hi = kF16Bits[(lane + 3) % kF16Bits.size()];
    uint16_t b_lo = kF16Bits[(lane + rot) % kF16Bits.size()];
    uint16_t b_hi = kF16Bits[(lane + rot + 5) % kF16Bits.size()];
    return std::array<uint16_t, 4>{a_lo, a_hi, b_lo, b_hi};
  };

  auto run_mode = [&](bool force_scalar, uint32_t rot) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    amdgpu::GpuMemory gpu_mem("vop3p_dot_f16_mem");
    amdgpu::L2Cache l2("vop3p_dot_f16_l2");
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("cu_vop3p_dot_f16", cfg, &gpu_mem, &l2);
    EXPECT_NE(cu, nullptr);
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    amdgpu::Wavefront *wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
    EXPECT_NE(wf, nullptr);

    uint32_t words[2] = {0u, 0u};
    vop3p_encode_dot(/*op=*/35, kDstVgpr, /*src0=*/256, /*src1=*/257, /*src2=*/258, neg, neg_hi,
                     clamp, words);
    Instruction *inst = decoder->decode(words);
    EXPECT_NE(inst, nullptr) << "v_dot2_f32_f16 decode failed";

    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      auto h = half_bits(lane, rot);
      uint32_t s0 = static_cast<uint32_t>(h[0]) | (static_cast<uint32_t>(h[1]) << 16);
      uint32_t s1 = static_cast<uint32_t>(h[2]) | (static_cast<uint32_t>(h[3]) << 16);
      uint32_t s2 = std::bit_cast<uint32_t>(kF32Acc[(lane + rot) % kF32Acc.size()]);
      cu->write_vgpr(vb + 0, lane, s0);
      cu->write_vgpr(vb + 1, lane, s1);
      cu->write_vgpr(vb + 2, lane, s2);
      cu->write_vgpr(vb + kDstVgpr, lane, DST_SENTINEL);
    }
    wf->set_exec(exec);
    cu->execute_instruction(inst, *wf);
    std::array<uint32_t, WF_SIZE> out{};
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vb + kDstVgpr, lane);
    delete inst;
    return out;
  };

  for (uint32_t rot = 0; rot < kF16Bits.size(); ++rot) {
    const auto scalar_out = run_mode(/*force_scalar=*/true, rot);
    const auto simd_out = run_mode(/*force_scalar=*/false, rot);

    // Core A/B equivalence per active, non-skipped lane. NaN-input lanes carry a
    // toolchain-dependent NaN-payload divergence and are excluded identically in
    // both runs (the skip condition is input-derived).
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (active) {
        auto h = half_bits(lane, rot);
        bool skip = is_f16_nan(h[0]) || is_f16_nan(h[1]) || is_f16_nan(h[2]) || is_f16_nan(h[3]);
        if (skip)
          continue;
        EXPECT_EQ(scalar_out[lane], simd_out[lane])
            << "v_dot2_f32_f16 neg=" << neg << " neg_hi=" << neg_hi << " clamp=" << clamp
            << " rot=" << rot << " lane " << lane << ": SIMD path diverged from scalar body";
      } else {
        EXPECT_EQ(simd_out[lane], DST_SENTINEL)
            << "v_dot2_f32_f16 neg=" << neg << " neg_hi=" << neg_hi << " clamp=" << clamp
            << " rot=" << rot << ": clobbered inactive lane " << lane;
        EXPECT_EQ(scalar_out[lane], DST_SENTINEL)
            << "v_dot2_f32_f16 neg=" << neg << " neg_hi=" << neg_hi << " clamp=" << clamp
            << " rot=" << rot << ": clobbered inactive lane " << lane;
      }
    }
  }
}

TEST(Vop3pDotF16SimdCorrectness, FullExecAllModifiers) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (uint32_t neg = 0; neg < 8; ++neg)
    for (uint32_t neg_hi = 0; neg_hi < 4; ++neg_hi)
      for (uint32_t clamp = 0; clamp < 2; ++clamp)
        check_f16_case(/*exec=*/~0ULL, neg, neg_hi, clamp);
}

TEST(Vop3pDotF16SimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check_f16_case(/*exec=*/0xA5A5'F0F0'1234'8001ULL, /*neg=*/0, /*neg_hi=*/0, /*clamp=*/0);
}

} // namespace
