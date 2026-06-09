// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop1_f64_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the 64-bit-lane
/// VOP1 unary ops on CDNA4: the f64 math family (ceil/floor/trunc/rndne/fract/
/// rcp/rsq/sqrt) and the pure 64-bit move v_mov_b64. Each case runs TWICE in the
/// same process -- once forcing the scalar body, once the SIMD fast path, with
/// identical inputs/EXEC -- and the destination f64 results are asserted equal
/// per active, non-skipped lane (util::set_force_scalar_for_testing flips the
/// gate in-process). In-process inactive lanes must stay preserved under full
/// and partial EXEC.
///
/// The f64 math ops map to correctly-rounded IEEE operations (vroundpd /
/// vsqrtpd / vdivpd), bit-exact to std::ceil/std::sqrt/... for finite and
/// infinite inputs; a NaN *result* may carry a different NaN payload between the
/// packed and scalar paths (accepted divergence), so those lanes are excluded
/// from the comparison — NaN-ness of the result is deterministic from the
/// inputs, so both runs skip the same lanes. v_mov_b64 is a pure bit copy and is
/// compared exactly (no NaN carve-out).

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
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;

// CDNA4 VOP1: enc[31:25] = 0b0111111, vdst[24:17], op[16:9], src0[8:0].
constexpr uint32_t vop1_encode(uint32_t op, uint32_t vdst, uint32_t src0) {
  return (0x3Fu << 25) | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF);
}

// f64 inputs covering ±0, ±1, ±Inf, denormal, max, and ordinary normals. The
// negatives feed sqrt/rsq a NaN result (skipped on compare); rcp(±0) -> ±Inf and
// ceil/floor/trunc/rndne of the specials are well defined and bit-exact.
const std::array<double, 18> kEdge = {{
    +0.0, -0.0, 1.0, -1.0, 2.5, 0.5, std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::denorm_min(),
    std::numeric_limits<double>::max(), 3.141592653589793, -2.718281828459045,
    // sNaN / qNaN (frexp quiets the former; math ops skip NaN-result lanes).
    std::bit_cast<double>(static_cast<uint64_t>(0x7FF0000000000001ull)),
    std::bit_cast<double>(static_cast<uint64_t>(0xFFF8000000000001ull)),
    // f64 denormals with the highest mantissa bit at varied positions, to
    // exercise frexp's denormal renormalization across the shift range.
    std::bit_cast<double>(static_cast<uint64_t>(0x000FFFFFFFFFFFFFull)), // p = 51
    std::bit_cast<double>(static_cast<uint64_t>(0x8000000100000000ull)), // p = 32, negative
    std::bit_cast<double>(static_cast<uint64_t>(0x0000000000010000ull)), // p = 16
}};

bool is_f64_nan(uint64_t bits) { return std::isnan(std::bit_cast<double>(bits)); }

struct F64UnaryCase {
  const char *name;
  uint32_t opcode;
  bool is_float; // false for v_mov_b64 (exact bit copy, no NaN carve-out)
};

const std::array<F64UnaryCase, 10> kCases = {{
    {"v_ceil_f64", 24, true},
    {"v_floor_f64", 26, true},
    {"v_trunc_f64", 23, true},
    {"v_rndne_f64", 25, true},
    {"v_fract_f64", 50, true},
    {"v_rcp_f64", 37, true},
    {"v_rsq_f64", 38, true},
    {"v_sqrt_f64", 40, true},
    {"v_mov_b64", 56, false},
    // frexp mantissa: deterministic bit-exact (incl. sNaN quieting), so exact
    // compare with no NaN-result skip (is_float=false).
    {"v_frexp_mant_f64", 49, false},
}};

// Per-lane vdst sentinel so an inactive-lane clobber is detectable.
uint64_t dst_sentinel(uint32_t lane) {
  return (static_cast<uint64_t>(0xCAFE0000u | lane) << 32) | (0xBEEF0000u + lane);
}

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop1_f64_simd_mem"), l2("vop1_f64_simd_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop1_f64_simd", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void write64(uint32_t reg, uint32_t lane, uint64_t v) {
    cu->write_vgpr(reg, lane, static_cast<uint32_t>(v));
    cu->write_vgpr(reg + 1, lane, static_cast<uint32_t>(v >> 32));
  }
  uint64_t read64(uint32_t reg, uint32_t lane) {
    return static_cast<uint64_t>(cu->read_vgpr(reg + 1, lane)) << 32 | cu->read_vgpr(reg, lane);
  }

  // src0 = v0:v1, vdst = v2:v3 (relative to the alloc base). vdst is stamped with
  // a per-lane sentinel so inactive-lane preservation is checkable.
  void seed_inputs(uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      double a = kEdge[lane % kEdge.size()];
      write64(vb + 0, lane, std::bit_cast<uint64_t>(a));
      write64(vb + 2, lane, dst_sentinel(lane));
    }
    wf->set_exec(exec);
  }

  std::array<uint64_t, WF_SIZE> run(Instruction *inst, uint64_t exec) {
    seed_inputs(exec);
    cu->execute_instruction(inst, *wf);
    std::array<uint64_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = read64(vb + 2, lane);
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

void check_case(const F64UnaryCase &c, uint64_t exec) {
  ForceScalarGuard gate_guard;

  auto run_mode = [&](bool force_scalar) -> std::array<uint64_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    // src0 = v0:v1 (enc 256), vdst = v2:v3.
    uint32_t enc = vop1_encode(c.opcode, /*vdst=*/2, /*src0=*/256);
    uint32_t words[4] = {enc, 0u, 0u, 0u};
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.name << " decode failed";
    auto out = fx.run(inst, exec);
    delete inst;
    return out;
  };

  const auto scalar_out = run_mode(/*force_scalar=*/true);
  const auto simd_out = run_mode(/*force_scalar=*/false);

  // Core A/B equivalence per active, non-skipped lane. For float ops a NaN
  // result carries a possibly-different NaN payload and is excluded identically
  // in both runs; v_mov_b64 (is_float=false) has no carve-out.
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (exec >> lane) & 1ULL;
    if (active) {
      if (c.is_float && (is_f64_nan(scalar_out[lane]) || is_f64_nan(simd_out[lane])))
        continue;
      EXPECT_EQ(scalar_out[lane], simd_out[lane])
          << c.name << " lane " << lane << ": SIMD path diverged from scalar body";
    } else {
      // Inactive lane: must leave the seeded vdst sentinel intact.
      EXPECT_EQ(simd_out[lane], dst_sentinel(lane))
          << c.name << " clobbered inactive lane " << lane;
      EXPECT_EQ(scalar_out[lane], dst_sentinel(lane))
          << c.name << " clobbered inactive lane " << lane;
    }
  }
}

TEST(Vop1F64SimdCorrectness, FullExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/~0ULL);
}

TEST(Vop1F64SimdCorrectness, PartialExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  // Pattern crossing the 8-wide f64 chunk boundaries on a wave64.
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
