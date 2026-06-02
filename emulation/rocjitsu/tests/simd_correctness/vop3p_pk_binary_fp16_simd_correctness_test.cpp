// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3p_pk_binary_fp16_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the VOP3P
/// packed-16 floating-point binary family on CDNA4:
///   v_pk_add_f16 / v_pk_mul_f16 / v_pk_max_f16 / v_pk_min_f16.
///
/// Each 32-bit lane holds two f16 values. The SIMD path widens each half to
/// f32, applies per-half sign-flip from neg/neg_hi, runs the functor in
/// f32, narrows back to f16 and packs. Default packing (op_sel = 0,
/// op_sel_hi = 3) only — non-default modes bail to scalar. The process runs one
/// fixed execute mode (RJ_FORCE_SCALAR, immutable); each result is recorded and
/// the scalar-vs-SIMD equivalence is asserted by diffing the two runs (see
/// simd_ab.h / the simd_ab_diff CTest entry). NaN-input lanes carry a
/// toolchain-dependent NaN-payload divergence and are overwritten with a fixed
/// sentinel before recording — the skip condition is computed from the inputs,
/// identical in both runs, so both mask the same lanes. In-process inactive
/// lanes must keep the sentinel.

#include "simd_ab.h"

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
// Fixed marker written over accepted-divergence (NaN-input) lanes before
// recording, so the cross-run diff ignores them identically in both runs.
constexpr uint32_t SKIP_SENTINEL = 0xA11D1FFFu;

// CDNA4 VOP3P encoding. neg goes into word1 bits 29:31; neg_hi goes into
// word0 bits 8:10; op_sel/op_sel_hi packed for default lo/hi pick.
constexpr void vop3p_encode_binary(uint32_t op, uint32_t vdst, uint32_t neg_hi, uint32_t src0,
                                   uint32_t src1, uint32_t neg, uint32_t words[2]) {
  // op_sel = 0, op_sel_hi = 3 (default packing).
  words[0] = (vdst & 0xFFu) | ((neg_hi & 0x7u) << 8) | ((op & 0x7Fu) << 16) | (0x1A7u << 23);
  words[1] = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | (0x3u << 27) | ((neg & 0x7u) << 29);
}
constexpr void vop3p_encode_ternary(uint32_t op, uint32_t vdst, uint32_t neg_hi, uint32_t src0,
                                    uint32_t src1, uint32_t src2, uint32_t neg, uint32_t words[2]) {
  // op_sel = 0, op_sel_hi = 3, op_sel_hi_2 = 1.
  words[0] =
      (vdst & 0xFFu) | ((neg_hi & 0x7u) << 8) | (1u << 14) | ((op & 0x7Fu) << 16) | (0x1A7u << 23);
  words[1] = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18) | (0x3u << 27) |
             ((neg & 0x7u) << 29);
}

struct Case {
  const char *name;
  uint32_t opcode;
  bool ternary;
};

const std::array<Case, 5> kCases = {{
    {"v_pk_add_f16_vop3p", 15, false},
    {"v_pk_mul_f16_vop3p", 16, false},
    {"v_pk_max_f16_vop3p", 18, false},
    {"v_pk_min_f16_vop3p", 17, false},
    {"v_pk_fma_f16_vop3p", 14, true},
}};

// f16 bit patterns covering normals + corners. Each 32-bit lane carries
// two halves picked from this set, so the two halves see distinct values
// (avoids accidentally hiding a per-half divergence).
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

bool is_f16_nan(uint16_t b) { return ((b >> 10) & 0x1Fu) == 0x1Fu && (b & 0x3FFu) != 0u; }

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3p_pk_bin_fp16_mem"), l2("vop3p_pk_bin_fp16_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3p_pk_bin_fp16", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_inputs(uint32_t rot, uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint16_t a_lo = kF16Bits[lane % kF16Bits.size()];
      uint16_t a_hi = kF16Bits[(lane + 3) % kF16Bits.size()];
      uint16_t b_lo = kF16Bits[(lane + rot) % kF16Bits.size()];
      uint16_t b_hi = kF16Bits[(lane + rot + 5) % kF16Bits.size()];
      uint16_t c_lo = kF16Bits[(lane + 2 * rot + 1) % kF16Bits.size()];
      uint16_t c_hi = kF16Bits[(lane + 2 * rot + 7) % kF16Bits.size()];
      uint32_t s0 = static_cast<uint32_t>(a_lo) | (static_cast<uint32_t>(a_hi) << 16);
      uint32_t s1 = static_cast<uint32_t>(b_lo) | (static_cast<uint32_t>(b_hi) << 16);
      uint32_t s2 = static_cast<uint32_t>(c_lo) | (static_cast<uint32_t>(c_hi) << 16);
      cu->write_vgpr(vb + 0, lane, s0);
      cu->write_vgpr(vb + 1, lane, s1);
      cu->write_vgpr(vb + 2, lane, s2);
      cu->write_vgpr(vb + kDstVgpr, lane, DST_SENTINEL);
    }
    wf->set_exec(exec);
  }

  std::array<uint32_t, WF_SIZE> run(Instruction *inst, uint32_t rot, uint64_t exec) {
    seed_inputs(rot, exec);
    cu->execute_instruction(inst, *wf);
    std::array<uint32_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vb + kDstVgpr, lane);
    return out;
  }
};

void check_case(const Case &c, uint64_t exec, uint32_t neg, uint32_t neg_hi) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  uint32_t words[2] = {0u, 0u};
  if (c.ternary)
    vop3p_encode_ternary(c.opcode, kDstVgpr, neg_hi, /*src0=*/256, /*src1=*/257, /*src2=*/258, neg,
                         words);
  else
    vop3p_encode_binary(c.opcode, kDstVgpr, neg_hi, /*src0=*/256, /*src1=*/257, neg, words);
  Instruction *inst = fx.decoder->decode(words);
  ASSERT_NE(inst, nullptr) << c.name << " decode failed";
  for (uint32_t rot = 0; rot < kF16Bits.size(); ++rot) {
    auto out = fx.run(inst, rot, exec);

    // Mask NaN-input lanes (accepted f16 NaN-payload divergence) before recording.
    // The skip condition is computed from the inputs (identical in both runs), so
    // both the scalar and SIMD runs mask the same lanes.
    uint32_t words_out[WF_SIZE];
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      bool skip = false;
      if (active) {
        uint16_t a_lo = kF16Bits[lane % kF16Bits.size()];
        uint16_t a_hi = kF16Bits[(lane + 3) % kF16Bits.size()];
        uint16_t b_lo = kF16Bits[(lane + rot) % kF16Bits.size()];
        uint16_t b_hi = kF16Bits[(lane + rot + 5) % kF16Bits.size()];
        skip = is_f16_nan(a_lo) || is_f16_nan(a_hi) || is_f16_nan(b_lo) || is_f16_nan(b_hi);
        if (!skip && c.ternary) {
          uint16_t c_lo = kF16Bits[(lane + 2 * rot + 1) % kF16Bits.size()];
          uint16_t c_hi = kF16Bits[(lane + 2 * rot + 7) % kF16Bits.size()];
          skip = is_f16_nan(c_lo) || is_f16_nan(c_hi);
        }
      }
      words_out[lane] = (active && skip) ? SKIP_SENTINEL : out[lane];
    }
    const std::string sub = std::string(c.name) + ":n" + std::to_string(neg) + "h" +
                            std::to_string(neg_hi) + "r" + std::to_string(rot);
    simd_ab::record(sub, exec, words_out, WF_SIZE);

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (!active) {
        EXPECT_EQ(out[lane], DST_SENTINEL)
            << c.name << " neg=" << neg << " neg_hi=" << neg_hi << " rot=" << rot
            << ": clobbered inactive lane " << lane;
      }
    }
  }
  delete inst;
}

TEST(Vop3pPkBinaryFp16SimdCorrectness, FullExecAllModifiers) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    for (uint32_t neg = 0; neg < 4; ++neg)
      for (uint32_t neg_hi = 0; neg_hi < 4; ++neg_hi)
        check_case(c, /*exec=*/~0ULL, neg, neg_hi);
}

TEST(Vop3pPkBinaryFp16SimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL, /*neg=*/0, /*neg_hi=*/0);
}

} // namespace
