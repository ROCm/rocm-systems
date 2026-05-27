// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop1_f64_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs forced-scalar body) for the
/// 64-bit-lane VOP1 unary ops on CDNA4: the f64 math family (ceil/floor/trunc/
/// rndne/fract/rcp/rsq/sqrt) and the pure 64-bit move v_mov_b64. Each runs the
/// instruction in both modes in-process via amdgpu::simd_force_scalar() and
/// asserts the destination VGPR pair agrees on active lanes and that inactive
/// lanes are preserved, under full and partial EXEC.
///
/// The f64 math ops map to correctly-rounded IEEE operations (vroundpd /
/// vsqrtpd / vdivpd), bit-exact to std::ceil/std::sqrt/... for finite and
/// infinite inputs; a NaN *result* may carry a different NaN payload between the
/// packed and scalar paths (accepted divergence), so those lanes are skipped.
/// v_mov_b64 is a pure bit copy and is compared exactly (no NaN carve-out).

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
const std::array<double, 12> kEdge = {{
    +0.0,
    -0.0,
    1.0,
    -1.0,
    2.5,
    0.5,
    std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::denorm_min(),
    std::numeric_limits<double>::max(),
    3.141592653589793,
    -2.718281828459045,
}};

bool is_f64_nan(uint64_t bits) { return std::isnan(std::bit_cast<double>(bits)); }

struct F64UnaryCase {
  const char *name;
  uint32_t opcode;
  bool is_float; // false for v_mov_b64 (exact bit copy, no NaN carve-out)
};

const std::array<F64UnaryCase, 9> kCases = {{
    {"v_ceil_f64", 24, true},
    {"v_floor_f64", 26, true},
    {"v_trunc_f64", 23, true},
    {"v_rndne_f64", 25, true},
    {"v_fract_f64", 50, true},
    {"v_rcp_f64", 37, true},
    {"v_rsq_f64", 38, true},
    {"v_sqrt_f64", 40, true},
    {"v_mov_b64", 56, false},
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

  std::array<uint64_t, WF_SIZE> run(Instruction *inst, bool force_scalar, uint64_t exec) {
    amdgpu::simd_force_scalar() = force_scalar;
    seed_inputs(exec);
    cu->execute_instruction(inst, *wf);
    amdgpu::simd_force_scalar() = false;
    std::array<uint64_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = read64(vb + 2, lane);
    return out;
  }
};

void check_case(const F64UnaryCase &c, uint64_t exec) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  // src0 = v0:v1 (enc 256), vdst = v2:v3.
  uint32_t enc = vop1_encode(c.opcode, /*vdst=*/2, /*src0=*/256);
  uint32_t words[4] = {enc, 0u, 0u, 0u};
  Instruction *inst = fx.decoder->decode(words);
  ASSERT_NE(inst, nullptr) << c.name << " decode failed";

  auto scalar = fx.run(inst, /*force_scalar=*/true, exec);
  auto simd = fx.run(inst, /*force_scalar=*/false, exec);

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    if (!(exec & (1ULL << lane))) {
      // Inactive lane: both modes must leave the seeded vdst sentinel intact.
      EXPECT_EQ(simd[lane], dst_sentinel(lane))
          << c.name << " clobbered inactive lane " << lane << " (simd)";
      EXPECT_EQ(scalar[lane], dst_sentinel(lane))
          << c.name << " clobbered inactive lane " << lane << " (scalar)";
      continue;
    }
    // Active lane: skip NaN-result lanes for the float ops (NaN payload may
    // diverge between packed and scalar — accepted).
    if (c.is_float && (is_f64_nan(scalar[lane]) || is_f64_nan(simd[lane])))
      continue;
    EXPECT_EQ(scalar[lane], simd[lane]) << c.name << " dst divergence at lane " << lane << std::hex
                                        << " scalar=0x" << scalar[lane] << " simd=0x" << simd[lane];
  }
  delete inst;
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
