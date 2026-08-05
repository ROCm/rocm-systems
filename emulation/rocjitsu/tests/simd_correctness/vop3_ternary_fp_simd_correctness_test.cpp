// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_ternary_fp_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the f32 / f16 /
/// f64 ternary VOP3 ops on CDNA4 (FMA + MAD family, non-accumulate). The new
/// ternary fp glue applies per-source abs/neg and result omod/clamp in the f32 /
/// f64 domain; f16 widens each src then operates in f32 then narrows. NaN-result
/// lanes are an accepted divergence (gcc-13 packed FMA quiets a different NaN
/// operand). Each (case, mods, rot) runs TWICE in the same process -- once
/// forcing the scalar body, once the SIMD fast path, with identical inputs/EXEC
/// -- and the dst results are asserted equal per active, non-skipped lane
/// (util::set_force_scalar_for_testing flips the gate in-process). NaN-result
/// lanes are excluded from the comparison — NaN-ness is deterministic from the
/// inputs, so both runs skip the same lanes.
///
/// The dst-accumulate variants (v_fmac_f32 / v_mac_f32 / v_fmac_f16 / v_mac_f16
/// / v_fmac_f64) are NOT exercised here: their per-isa codegen classes only
/// initialize src0+src1+vdst (the third FMA arg comes from vdst, no src2
/// member), so they need a separate accumulate-form glue path.

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
constexpr uint32_t kDstVgpr32 = 6;
constexpr uint32_t kDstVgpr64 = 6; // v6:v7
constexpr uint32_t DST_SENTINEL32 = 0xCDCDCDCDu;

constexpr void vop3_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1, uint32_t src2,
                           uint32_t abs, uint32_t neg, uint32_t omod, uint32_t clamp,
                           uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((abs & 0x7) << 8) | ((clamp & 0x1) << 15) | ((op & 0x3FF) << 16) |
             (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((src2 & 0x1FF) << 18) |
             ((omod & 0x3) << 27) | ((neg & 0x7) << 29);
}

bool is_f32_nan(uint32_t bits) {
  return ((bits >> 23) & 0xFFu) == 0xFFu && (bits & 0x7FFFFFu) != 0;
}
bool is_f16_nan(uint32_t bits) {
  uint16_t h = static_cast<uint16_t>(bits);
  return ((h >> 10) & 0x1Fu) == 0x1Fu && (h & 0x3FFu) != 0;
}
bool is_f64_nan(uint64_t bits) {
  return ((bits >> 52) & 0x7FFu) == 0x7FFu && (bits & 0xFFFFFFFFFFFFFull) != 0;
}

enum class Kind { F32, F16, F64 };

struct Case {
  const char *name;
  uint32_t opcode;
  Kind kind;
};

const std::array<Case, 10> kCases = {{
    {"v_fma_f32_vop3", 459, Kind::F32},
    {"v_fma_f16_vop3", 518, Kind::F16},
    {"v_mad_f16_vop3", 515, Kind::F16},
    {"v_fma_f64_vop3", 460, Kind::F64},
    // min3/max3/med3: fmax/fmin compositions. Inputs are finite non-zero
    // normals, so the fmax/fmin NaN-payload / signed-zero-tie carve-out never
    // triggers — bit-exact vs scalar on every lane.
    {"v_max3_f32_vop3", 467, Kind::F32},
    {"v_min3_f32_vop3", 464, Kind::F32},
    {"v_med3_f32_vop3", 470, Kind::F32},
    {"v_max3_f16_vop3", 503, Kind::F16},
    {"v_min3_f16_vop3", 500, Kind::F16},
    {"v_med3_f16_vop3", 506, Kind::F16},
}};

// Finite-normal f32 sanitized inputs (avoid NaN/Inf for cleaner FMA bit-equality
// checks; NaN lanes would be skipped anyway).
const std::array<uint32_t, 16> kF32 = {{
    0x3F800000u,
    0xBF800000u,
    0x40000000u,
    0xC0000000u,
    0x3FC00000u,
    0xBFC00000u,
    0x40400000u,
    0x40800000u,
    0x3F000000u,
    0xBF000000u,
    0x3E800000u,
    0xC0900000u,
    0x40000000u,
    0x40C00000u,
    0x41000000u,
    0xC1000000u,
}};
// f16 sanitized finite normals in low 16 (high 16 = sentinel).
const std::array<uint32_t, 16> kF16 = {{
    0xDEAD3C00u,
    0xDEADBC00u,
    0xDEAD4000u,
    0xDEADC000u,
    0xDEAD4200u,
    0xDEADC200u,
    0xDEAD4400u,
    0xDEAD4500u,
    0xDEAD3800u,
    0xDEADB800u,
    0xDEAD3400u,
    0xDEADC480u,
    0xDEAD4000u,
    0xDEAD4600u,
    0xDEAD4800u,
    0xDEADC800u,
}};
// f64 sanitized finite normals.
const std::array<uint64_t, 16> kF64 = {{
    0x3FF0000000000000ull,
    0xBFF0000000000000ull,
    0x4000000000000000ull,
    0xC000000000000000ull,
    0x3FF8000000000000ull,
    0xBFF8000000000000ull,
    0x4008000000000000ull,
    0x4010000000000000ull,
    0x3FE0000000000000ull,
    0xBFE0000000000000ull,
    0x3FD0000000000000ull,
    0xC012000000000000ull,
    0x4000000000000000ull,
    0x4018000000000000ull,
    0x4020000000000000ull,
    0xC020000000000000ull,
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3_tern_fp_mem"), l2("vop3_tern_fp_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_tern_fp", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void write64(uint32_t reg, uint32_t lane, uint64_t v) {
    cu->write_vgpr(reg, lane, static_cast<uint32_t>(v));
    cu->write_vgpr(reg + 1, lane, static_cast<uint32_t>(v >> 32));
  }

  // f32/f16: src0=v0, src1=v1, src2=v2, dst=v6. f64: src0=v0:v1, src1=v2:v3,
  // src2=v4:v5, dst=v6:v7.
  void seed_inputs(Kind k, uint32_t rot, uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      if (k == Kind::F64) {
        write64(vb + 0, lane, kF64[lane % kF64.size()]);
        write64(vb + 2, lane, kF64[(lane + rot) % kF64.size()]);
        write64(vb + 4, lane, kF64[(lane + 2 * rot) % kF64.size()]);
        write64(vb + kDstVgpr64, lane, 0xCDCDCDCDCDCDCDCDull);
      } else {
        const auto &v = (k == Kind::F32) ? kF32 : kF16;
        cu->write_vgpr(vb + 0, lane, v[lane % v.size()]);
        cu->write_vgpr(vb + 1, lane, v[(lane + rot) % v.size()]);
        cu->write_vgpr(vb + 2, lane, v[(lane + 2 * rot) % v.size()]);
        cu->write_vgpr(vb + kDstVgpr32, lane, DST_SENTINEL32);
      }
    }
    wf->set_exec(exec);
  }

  std::array<uint64_t, WF_SIZE> run(Instruction *inst, Kind k, uint32_t rot, uint64_t exec) {
    seed_inputs(k, rot, exec);
    cu->execute_instruction(inst, *wf);
    std::array<uint64_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      if (k == Kind::F64)
        out[lane] = static_cast<uint64_t>(cu->read_vgpr(vb + kDstVgpr64 + 1, lane)) << 32 |
                    cu->read_vgpr(vb + kDstVgpr64, lane);
      else
        out[lane] = cu->read_vgpr(vb + kDstVgpr32, lane);
    }
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

bool result_is_nan(Kind k, uint64_t v) {
  if (k == Kind::F32)
    return is_f32_nan(static_cast<uint32_t>(v));
  if (k == Kind::F16)
    return is_f16_nan(static_cast<uint32_t>(v));
  return is_f64_nan(v);
}

void check_case(const Case &c, uint32_t abs, uint32_t neg, uint32_t omod, uint32_t clamp,
                uint64_t exec) {
  ForceScalarGuard gate_guard;

  auto run_mode = [&](bool force_scalar, uint32_t rot) -> std::array<uint64_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    const uint32_t vdst = (c.kind == Kind::F64) ? kDstVgpr64 : kDstVgpr32;
    const uint32_t src1_v = (c.kind == Kind::F64) ? 2u : 1u;
    const uint32_t src2_v = (c.kind == Kind::F64) ? 4u : 2u;
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_encode(c.opcode, vdst, /*src0=*/256, /*src1=*/256 + src1_v, /*src2=*/256 + src2_v, abs,
                neg, omod, clamp, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.name << " decode failed";
    auto out = fx.run(inst, c.kind, rot, exec);
    delete inst;
    return out;
  };

  const std::size_t rot_max = (c.kind == Kind::F64) ? kF64.size() : kF32.size();
  for (uint32_t rot = 0; rot < rot_max; ++rot) {
    const auto scalar_out = run_mode(/*force_scalar=*/true, rot);
    const auto simd_out = run_mode(/*force_scalar=*/false, rot);

    // Core A/B equivalence per active, non-skipped lane. NaN-result lanes carry
    // an accepted payload divergence and are excluded identically in both runs.
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (active &&
          (result_is_nan(c.kind, scalar_out[lane]) || result_is_nan(c.kind, simd_out[lane])))
        continue;
      EXPECT_EQ(scalar_out[lane], simd_out[lane])
          << c.name << " a" << abs << "n" << neg << "o" << omod << "c" << clamp << "r" << rot
          << " lane " << lane << ": SIMD path diverged from scalar body";
    }
  }
}

// A representative subset of the 8x8x4x2 = 512 modifier grid. The per-source
// abs/neg masks select independent branches (`if (abs & (1u << SrcIdx))`), so
// bit 0 vs bit 1 vs bit 2 exercise the same code path -- a handful of masks
// covers every distinct path. These 15 combos cover: no modifiers, each
// single-source abs, each single-source neg, all-sources abs, all-sources neg,
// abs+neg combined, each omod value, clamp, and a mixed case. ~1.3s vs the full
// grid's 163,840 fixtures (timed out at 15s).
void check_representative_mods(const Case &c, uint64_t exec) {
  struct ModCombo {
    uint32_t abs, neg, omod, clamp;
  };
  static constexpr ModCombo kCombos[] = {
      {0, 0, 0, 0}, {1, 0, 0, 0}, {2, 0, 0, 0}, {4, 0, 0, 0}, {7, 0, 0, 0},
      {0, 1, 0, 0}, {0, 2, 0, 0}, {0, 4, 0, 0}, {0, 7, 0, 0}, {7, 7, 0, 0},
      {0, 0, 1, 0}, {0, 0, 2, 0}, {0, 0, 3, 0}, {0, 0, 0, 1}, {3, 5, 1, 1},
  };
  for (const auto &m : kCombos)
    check_case(c, m.abs, m.neg, m.omod, m.clamp, exec);
}

TEST(Vop3TernaryFpSimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_representative_mods(c, /*exec=*/~0ULL);
}

TEST(Vop3TernaryFpSimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*abs=*/0, /*neg=*/0, /*omod=*/0, /*clamp=*/0, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
