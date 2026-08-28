// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop2_fma_f16_rdna_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the two f16
/// FMA-K forms that exist only on RDNA (RDNA3+): v_fmaak_f16 and v_fmamk_f16.
/// The CDNA fixture (vop2_fma_simd_correctness_test.cpp) cannot cover these --
/// CDNA4 has no v_fmaak_f16/v_fmamk_f16 decode path -- so this RDNA4-keyed
/// fixture exercises the changed literal read (`inst.simm32.encoding_value_` in
/// SIMD_VOP2_TERNARY) for both f16 paths:
///   v_fmaak_f16 (56) -> fma(s0, s1, K)
///   v_fmamk_f16 (55) -> fma(s0, K, s1)
/// Each op runs TWICE in the same process -- once forcing the scalar body, once
/// the SIMD fast path, with identical seed/inputs/EXEC -- and the two result
/// arrays are asserted equal per active, non-skipped lane. fma is bit-exact for
/// all finite/Inf inputs; a NaN *input* may propagate a different NaN payload
/// through the packed vs scalar FMA (accepted), so lanes with a NaN input are
/// excluded identically in both runs (the skip is input-derived).

#include "util/simd_test_hooks.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/execute_shared.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <string>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 32; // RDNA wave32
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;

// RDNA VOP2: enc[31]=0 | op[30:25] | vdst[24:17] | vsrc1[16:9] | src0[8:0].
constexpr uint32_t vop2_encode(uint32_t opcode, uint32_t vdst, uint32_t vsrc1, uint32_t src0) {
  return ((opcode & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF);
}

// f32 edge bit patterns; the low 16 bits of each drive the f16 edge lanes
// (NaN/Inf/denorm/zero) the op reads. No NaN kept intentionally -- NaN-input
// lanes are skipped in the comparison anyway (accepted payload divergence).
const std::array<uint32_t, 12> kF32Edges = {{
    0x00000000u, // +0
    0x80000000u, // -0
    0x7F800000u, // +Inf
    0xFF800000u, // -Inf
    0x00000001u, // smallest denormal
    0x807FFFFFu, // -largest denormal
    0x007FFFFFu, // largest denormal
    0x3F800000u, // 1.0
    0xBF800000u, // -1.0
    0x4F000000u, // 2^31
    0x7F7FFFFFu, // FLT_MAX
    0x00800000u, // smallest normal
}};

bool is_f16_nan(uint32_t u) { return ((u >> 10) & 0x1Fu) == 0x1Fu && (u & 0x3FFu) != 0u; }

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop2_fma_f16_rdna_mem"), l2("vop2_fma_f16_rdna_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop2_fma_f16_rdna", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  // Seed v0 (src0), v1 (vsrc1), v2 (vdst sentinel). Low lanes get f16 edge
  // patterns (packed in both halves); the rest are raw random. Records which
  // lanes carry a NaN f16 input so the comparison can skip them identically.
  void seed_inputs(uint64_t seed, uint64_t exec, std::array<bool, WF_SIZE> *nan_lane) {
    std::mt19937_64 rng(seed);
    uint32_t vbase = wf->vgpr_alloc().base;
    auto edge = [&](uint32_t i) -> uint32_t {
      uint32_t lo = (kF32Edges[i % kF32Edges.size()] >> 16) & 0xFFFFu;
      return lo | (lo << 16);
    };
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint32_t r0, r1;
      if (lane < kF32Edges.size()) {
        r0 = edge(lane);
        r1 = edge(lane + 3);
      } else {
        r0 = static_cast<uint32_t>(rng());
        r1 = static_cast<uint32_t>(rng());
      }
      cu->write_vgpr(vbase + 0, lane, r0);
      cu->write_vgpr(vbase + 1, lane, r1);
      cu->write_vgpr(vbase + 2, lane, DST_SENTINEL);
      if (nan_lane)
        (*nan_lane)[lane] = is_f16_nan(r0 & 0xFFFFu) || is_f16_nan(r1 & 0xFFFFu);
    }
    wf->set_exec(exec);
  }

  std::array<uint32_t, WF_SIZE> snapshot_dst() const {
    std::array<uint32_t, WF_SIZE> out{};
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vbase + 2, lane);
    return out;
  }

  static constexpr uint32_t DST_SENTINEL = 0xCAFEF00Du;
};

struct FmaCase {
  const char *label;
  uint32_t opcode;
};

// The two f16 FMA-K forms that only decode on RDNA. Both carry the K literal in
// words[1] and read it via inst.simm32.encoding_value_ in the shared body.
const FmaCase kCases[] = {
    {"v_fmaak_f16", 56}, // dst = fma(s0, s1, K)
    {"v_fmamk_f16", 55}, // dst = fma(s0, K, s1)
};

// Restores the process force-scalar gate on scope exit so flipping it for an
// in-process A/B comparison cannot leak into later tests in the same process.
struct ForceScalarGuard {
  bool orig;
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }
};

void check_case(const FmaCase &c, uint64_t exec) {
  ForceScalarGuard gate_guard;

  constexpr uint64_t SEED = 0xF16A'1234'5678'9ABCULL;
  // K = 1.5h in the low 16 bits (high bits ignored by the op).
  constexpr uint32_t kLiteral = 0x00003E00u;

  std::array<bool, WF_SIZE> nan_lane{};

  auto run_mode = [&](bool force_scalar) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t enc = vop2_encode(c.opcode, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256);
    uint32_t words[4] = {enc, kLiteral, 0u, 0u};
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.label << ": decode failed";
    fx.seed_inputs(SEED, exec, &nan_lane);
    fx.cu->execute_instruction(inst, *fx.wf);
    auto out = fx.snapshot_dst();
    delete inst;
    return out;
  };

  const auto scalar_out = run_mode(/*force_scalar=*/true);
  const auto simd_out = run_mode(/*force_scalar=*/false);

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (exec >> lane) & 1ULL;
    if (active) {
      if (nan_lane[lane])
        continue;
      EXPECT_EQ(scalar_out[lane], simd_out[lane])
          << c.label << " lane " << lane << ": SIMD path diverged from scalar body";
    } else {
      EXPECT_EQ(simd_out[lane], Fixture::DST_SENTINEL)
          << c.label << ": clobbered inactive lane " << lane;
      EXPECT_EQ(scalar_out[lane], Fixture::DST_SENTINEL)
          << c.label << ": clobbered inactive lane " << lane;
    }
  }
}

TEST(Vop2FmaF16RdnaSimdCorrectness, FullExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xFFFFFFFFULL);
}

TEST(Vop2FmaF16RdnaSimdCorrectness, PartialExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  // Alternating + sparse 32-bit pattern: exercises masked_store blend and
  // inactive-lane preservation across SIMD chunk boundaries.
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5F0F0ULL);
}

} // namespace
