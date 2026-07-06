// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_fp16_unary_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the f16 VOP3
/// unary rounding ops (ceil/floor/trunc/rndne) plus sqrt on CDNA4. The scalar
/// body widens f16 -> f32, applies src0 abs/neg, runs the op, applies
/// omod/clamp, narrows back via f32_to_f16. The new SIMD glue
/// (try_execute_unary_vop3_fp16_simd) does the same chain in-vector. The process
/// runs one fixed execute mode (RJ_FORCE_SCALAR, immutable); each (case, mods)
/// runs TWICE in the same process -- once forcing the scalar body, once the SIMD
/// fast path, with identical inputs/EXEC -- and the results are asserted equal
/// per active, non-skipped lane (util::set_force_scalar_for_testing flips the
/// gate in-process). NaN-result lanes carry an accepted payload divergence and
/// are excluded from the comparison — NaN-ness is deterministic from the inputs,
/// so both runs skip the same lanes. In-process inactive lanes must keep the
/// sentinel.

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
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t kDstVgpr = 2;
constexpr uint32_t DST_SENTINEL = 0xCDCDCDCDu;

constexpr void vop3_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t abs, uint32_t neg,
                           uint32_t omod, uint32_t clamp, uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((abs & 0x7) << 8) | ((clamp & 0x1) << 15) | ((op & 0x3FF) << 16) |
             (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((omod & 0x3) << 27) | ((neg & 0x7) << 29);
}

bool is_f16_nan(uint32_t bits) {
  uint16_t h = static_cast<uint16_t>(bits);
  return ((h >> 10) & 0x1Fu) == 0x1Fu && (h & 0x3FFu) != 0;
}

struct Case {
  const char *name;
  uint32_t opcode;
};

const std::array<Case, 6> kCases = {{
    {"v_ceil_f16_vop3", 389},
    {"v_floor_f16_vop3", 388},
    {"v_trunc_f16_vop3", 390},
    {"v_rndne_f16_vop3", 391},
    {"v_sqrt_f16_vop3", 382},
    {"v_fract_f16_vop3", 392},
}};

// f16 inputs across every IEEE class + values that exercise clamp / omod
// overflow (0.5 -> omod *2 = 1, 0.7 -> omod *2 > 1 saturated by clamp, etc.).
const std::array<uint32_t, 16> kF16 = {{
    0xDEAD0000u, // +0
    0xDEAD8000u, // -0
    0xDEAD3C00u, // +1
    0xDEADBC00u, // -1
    0xDEAD4248u, // +pi
    0xDEADC248u, // -pi
    0xDEAD7C00u, // +Inf
    0xDEADFC00u, // -Inf
    0xDEAD7E00u, // +qNaN
    0xDEADFD00u, // -sNaN
    0xDEAD0001u, // +denormal
    0xDEAD8001u, // -denormal
    0xDEAD7BFFu, // +max
    0xDEAD3800u, // +0.5
    0xDEAD3666u, // +0.4
    0xDEAD4000u, // +2
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3_fp16_una_mem"), l2("vop3_fp16_una_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_fp16_una", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_inputs(uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      cu->write_vgpr(vb + 0, lane, kF16[lane % kF16.size()]);
      cu->write_vgpr(vb + kDstVgpr, lane, DST_SENTINEL);
    }
    wf->set_exec(exec);
  }

  std::array<uint32_t, WF_SIZE> run(Instruction *inst, uint64_t exec) {
    seed_inputs(exec);
    cu->execute_instruction(inst, *wf);
    std::array<uint32_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vb + kDstVgpr, lane);
    return out;
  }
};

// Restores the process force-scalar gate on scope exit so flipping it for an
// in-process A/B comparison cannot leak into later tests in the same process.
struct ForceScalarGuard {
  bool orig;
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }
};

void check_case(const Case &c, uint32_t abs, uint32_t neg, uint32_t omod, uint32_t clamp,
                uint64_t exec) {
  ForceScalarGuard gate_guard;

  auto run_mode = [&](bool force_scalar) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_encode(c.opcode, /*vdst=*/kDstVgpr, /*src0=*/256, abs, neg, omod, clamp, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.name << " decode failed";
    auto out = fx.run(inst, exec);
    delete inst;
    return out;
  };

  const auto scalar_out = run_mode(/*force_scalar=*/true);
  const auto simd_out = run_mode(/*force_scalar=*/false);

  // Core A/B equivalence per active, non-skipped lane. NaN-result lanes carry an
  // accepted f16 NaN-payload divergence and are excluded identically in both runs.
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (exec >> lane) & 1ULL;
    if (active) {
      if (is_f16_nan(scalar_out[lane]) || is_f16_nan(simd_out[lane]))
        continue;
      EXPECT_EQ(scalar_out[lane], simd_out[lane])
          << c.name << " abs=" << abs << " neg=" << neg << " omod=" << omod << " clamp=" << clamp
          << " lane " << lane << ": SIMD path diverged from scalar body";
    } else {
      EXPECT_EQ(simd_out[lane], DST_SENTINEL)
          << c.name << " abs=" << abs << " neg=" << neg << ": clobbered inactive lane " << lane;
      EXPECT_EQ(scalar_out[lane], DST_SENTINEL)
          << c.name << " abs=" << abs << " neg=" << neg << ": clobbered inactive lane " << lane;
    }
  }
}

void check_all_mods(const Case &c, uint64_t exec) {
  for (uint32_t abs = 0; abs < 2; ++abs)
    for (uint32_t neg = 0; neg < 2; ++neg)
      for (uint32_t omod = 0; omod < 4; ++omod)
        for (uint32_t clamp = 0; clamp < 2; ++clamp)
          check_case(c, abs, neg, omod, clamp, exec);
}

TEST(Vop3Fp16UnarySimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_all_mods(c, /*exec=*/~0ULL);
}

TEST(Vop3Fp16UnarySimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*abs=*/0, /*neg=*/0, /*omod=*/0, /*clamp=*/0, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
