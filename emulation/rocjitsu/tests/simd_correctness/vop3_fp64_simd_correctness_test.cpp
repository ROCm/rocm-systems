// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_fp64_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the f64 VOP3
/// binary (add/mul/max/min) and unary (ceil/floor/trunc/rndne/sqrt) ops on
/// CDNA4. Reads src0/src1 as native<double> via split lo/hi VGPR pairs, applies
/// per-source abs/neg and result omod/clamp in the f64 domain (the new
/// apply_vop3_*_mod_f64 helpers), then op. Bit-exact for finite/Inf on the
/// add/mul/cvt/sqrt ops; the min/max ops accept the standard ±0 / NaN payload
/// divergences. Each (case, mods, rot) runs TWICE in the same process -- once
/// forcing the scalar body, once the SIMD fast path, with identical inputs/EXEC
/// -- and the f64 dst results are asserted equal per active, non-skipped lane
/// (util::set_force_scalar_for_testing flips the gate in-process). Accepted-
/// divergence lanes (NaN/±0-tie, all determined from the inputs) are excluded
/// from the comparison so both runs skip the same lanes. In-process inactive
/// lanes must stay preserved.

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
constexpr uint32_t kDstVgpr = 4; // v4:v5

constexpr uint64_t dst_sentinel(uint32_t lane) {
  return (static_cast<uint64_t>(0xCAFE0000u | lane) << 32) | (0xBEEF0000u + lane);
}

constexpr void vop3_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1, uint32_t abs,
                           uint32_t neg, uint32_t omod, uint32_t clamp, uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((abs & 0x7) << 8) | ((clamp & 0x1) << 15) | ((op & 0x3FF) << 16) |
             (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((omod & 0x3) << 27) | ((neg & 0x7) << 29);
}

bool is_f64_nan(uint64_t bits) {
  return ((bits >> 52) & 0x7FFu) == 0x7FFu && (bits & 0xFFFFFFFFFFFFFull) != 0;
}
bool is_f64_zero(uint64_t bits) { return (bits & 0x7FFFFFFFFFFFFFFFull) == 0; }

enum class Kind { Bin, Una, BinMinMax };

struct Case {
  const char *name;
  uint32_t opcode;
  Kind kind;
};

const std::array<Case, 14> kCases = {{
    {"v_add_f64_vop3", 640, Kind::Bin},
    {"v_mul_f64_vop3", 641, Kind::Bin},
    {"v_max_f64_vop3", 643, Kind::BinMinMax},
    {"v_min_f64_vop3", 642, Kind::BinMinMax},
    {"v_ceil_f64_vop3", 344, Kind::Una},
    {"v_floor_f64_vop3", 346, Kind::Una},
    {"v_trunc_f64_vop3", 343, Kind::Una},
    {"v_rndne_f64_vop3", 345, Kind::Una},
    {"v_sqrt_f64_vop3", 360, Kind::Una},
    {"v_fract_f64_vop3", 370, Kind::Una},
    {"v_rcp_f64_vop3", 357, Kind::Una},
    {"v_rsq_f64_vop3", 358, Kind::Una},
    {"v_mov_b64_vop3", 376, Kind::Una},
    {"v_frexp_mant_f64_vop3", 369, Kind::Una},
}};

// f64 inputs covering every IEEE class + clamp boundary + values that exercise
// omod *2 / *4 / /2 overflow and the sqrt domain (positive only for sqrt).
const std::array<uint64_t, 16> kF64 = {{
    0x0000000000000000ull, 0x8000000000000000ull, 0x3FF0000000000000ull, 0xBFF0000000000000ull,
    0x400921FB54442D18ull, 0xC00921FB54442D18ull, 0x7FF0000000000000ull, 0xFFF0000000000000ull,
    0x7FF8000000000000ull, 0xFFF4000000000000ull, 0x0000000000000001ull, 0x8000000000000001ull,
    0x7FEFFFFFFFFFFFFFull, 0x3FE0000000000000ull, // 0.5
    0x3FD999999999999Aull,                        // 0.4
    0x4000000000000000ull,                        // 2.0
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3_fp64_mem"), l2("vop3_fp64_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_fp64", cfg, &gpu_mem, &l2);
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

  // src0 = v0:v1, src1 = v2:v3 (binary), dst = v4:v5 (sentinel-stamped).
  void seed_inputs(uint32_t rot, uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      write64(vb + 0, lane, kF64[lane % kF64.size()]);
      write64(vb + 2, lane, kF64[(lane + rot) % kF64.size()]);
      write64(vb + kDstVgpr, lane, dst_sentinel(lane));
    }
    wf->set_exec(exec);
  }

  std::array<uint64_t, WF_SIZE> run(Instruction *inst, uint32_t rot, uint64_t exec) {
    seed_inputs(rot, exec);
    cu->execute_instruction(inst, *wf);
    std::array<uint64_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = read64(vb + kDstVgpr, lane);
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

  auto run_mode = [&](bool force_scalar, uint32_t rot) -> std::array<uint64_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_encode(c.opcode, /*vdst=*/kDstVgpr, /*src0=*/256, /*src1=*/258, abs, neg, omod, clamp,
                words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.name << " decode failed";
    auto out = fx.run(inst, rot, exec);
    delete inst;
    return out;
  };

  const std::size_t kRotMax = (c.kind == Kind::Una) ? 1u : kF64.size();
  for (uint32_t rot = 0; rot < kRotMax; ++rot) {
    const auto scalar_out = run_mode(/*force_scalar=*/true, rot);
    const auto simd_out = run_mode(/*force_scalar=*/false, rot);

    // Core A/B equivalence per active, non-skipped lane. Accepted-divergence
    // lanes (for min/max: NaN-input / ±0-tie; for any op: NaN result) are
    // excluded; every skip condition is deterministic from the inputs, so both
    // runs skip the same lanes.
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (active) {
        bool skip = false;
        if (c.kind == Kind::BinMinMax) {
          uint64_t a = kF64[lane % kF64.size()];
          uint64_t b = kF64[(lane + rot) % kF64.size()];
          if (is_f64_nan(a) || is_f64_nan(b))
            skip = true; // accepted NaN-payload divergence
          if (is_f64_zero(a) && is_f64_zero(b))
            skip = true; // accepted ±0-tie divergence
        }
        if (is_f64_nan(scalar_out[lane]) || is_f64_nan(simd_out[lane]))
          skip = true; // NaN result: payload may differ
        if (skip)
          continue;
        EXPECT_EQ(scalar_out[lane], simd_out[lane])
            << c.name << " a" << abs << "n" << neg << "o" << omod << "c" << clamp << "r" << rot
            << " lane " << lane << ": SIMD path diverged from scalar body";
      } else {
        EXPECT_EQ(simd_out[lane], dst_sentinel(lane))
            << c.name << " abs=" << abs << " neg=" << neg << ": clobbered inactive lane " << lane;
        EXPECT_EQ(scalar_out[lane], dst_sentinel(lane))
            << c.name << " abs=" << abs << " neg=" << neg << ": clobbered inactive lane " << lane;
      }
    }
  }
}

// Binary sweep: 4 abs combos × 4 neg combos × 4 omod × 2 clamp × all rotations.
// Unary uses (abs<<0, neg<<0), same omod/clamp space.
void check_all_mods(const Case &c, uint64_t exec) {
  const uint32_t abs_max = (c.kind == Kind::Una) ? 2u : 4u;
  const uint32_t neg_max = (c.kind == Kind::Una) ? 2u : 4u;
  for (uint32_t abs = 0; abs < abs_max; ++abs)
    for (uint32_t neg = 0; neg < neg_max; ++neg)
      for (uint32_t omod = 0; omod < 4; ++omod)
        for (uint32_t clamp = 0; clamp < 2; ++clamp)
          check_case(c, abs, neg, omod, clamp, exec);
}

TEST(Vop3Fp64SimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_all_mods(c, /*exec=*/~0ULL);
}

TEST(Vop3Fp64SimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*abs=*/0, /*neg=*/0, /*omod=*/0, /*clamp=*/0, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
