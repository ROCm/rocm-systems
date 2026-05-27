// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop2_minmax_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs forced-scalar body) for the
/// float min/max VOP2 ops: v_max_f32 (11), v_min_f32 (10), v_max_f16 (45),
/// v_min_f16 (46). These use std::fmax/std::fmin in the scalar body; the SIMD
/// path uses util::stdx::fmax/fmin, which is bit-exact for finite/Inf and the
/// signed-zero tie cases. NaN-input lanes can differ in NaN payload (accepted),
/// so they are skipped in the comparison. The inputs deliberately pair up ±0
/// and NaN in both operand orders on the low lanes — the corner cases the
/// earlier (reverted) f16 min/max shipped without and silently diverged on.

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

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t DST_SENTINEL = 0xCAFEF00Du;

constexpr uint32_t vop2_encode(uint32_t opcode, uint32_t vdst, uint32_t vsrc1, uint32_t src0) {
  return ((opcode & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF);
}

bool is_f32_nan(uint32_t u) { return ((u >> 23) & 0xFFu) == 0xFFu && (u & 0x7FFFFFu) != 0u; }
bool is_f16_nan(uint32_t u) { return ((u >> 10) & 0x1Fu) == 0x1Fu && (u & 0x3FFu) != 0u; }
bool is_f32_zero(uint32_t u) { return (u & 0x7FFFFFFFu) == 0u; }
bool is_f16_zero(uint32_t u) { return (u & 0x7FFFu) == 0u; }

// Edge operand pairs (src0, vsrc1) for the low lanes, as f32 bit patterns. The
// pairing exercises ±0 ties in both orders, NaN in either slot, and ordinary
// ordering. For the f16 cases the low 16 bits of each are reinterpreted.
struct Pair {
  uint32_t a, b;
};
const std::array<Pair, 14> kEdges = {{
    {0x80000000u, 0x00000000u}, // -0, +0
    {0x00000000u, 0x80000000u}, // +0, -0
    {0x80000000u, 0x80000000u}, // -0, -0
    {0x3F800000u, 0xBF800000u}, // 1.0, -1.0
    {0x7F800000u, 0x3F800000u}, // +Inf, 1.0
    {0xFF800000u, 0x3F800000u}, // -Inf, 1.0
    {0x7FC00000u, 0x3F800000u}, // QNaN, 1.0  (skipped)
    {0x3F800000u, 0x7FC00000u}, // 1.0, QNaN  (skipped)
    {0x7F800001u, 0x40000000u}, // SNaN, 2.0  (skipped)
    {0x00800000u, 0x00000001u}, // smallest normal, smallest denormal
    {0x007FFFFFu, 0x80000000u}, // largest denormal, -0
    {0x7F7FFFFFu, 0xFF7FFFFFu}, // FLT_MAX, -FLT_MAX
    {0xC0490FDBu, 0x40490FDBu}, // -pi, +pi
    {0x42280000u, 0x42280000u}, // 42.0, 42.0
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop2_minmax_mem"), l2("vop2_minmax_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop2_minmax", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  std::array<bool, WF_SIZE> seed_inputs(uint64_t seed, bool is_f16, uint64_t exec) {
    std::mt19937_64 rng(seed);
    std::array<bool, WF_SIZE> nan_lane{};
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint32_t r0, r1;
      if (lane < kEdges.size()) {
        r0 = kEdges[lane].a;
        r1 = kEdges[lane].b;
      } else {
        r0 = static_cast<uint32_t>(rng());
        r1 = static_cast<uint32_t>(rng());
        // Bias toward ±0 ties on some random lanes too.
        if ((rng() & 7u) == 0u)
          r0 &= 0x80000000u;
        if ((rng() & 7u) == 0u)
          r1 &= 0x80000000u;
      }
      cu->write_vgpr(vbase + 0, lane, r0);
      cu->write_vgpr(vbase + 1, lane, r1);
      cu->write_vgpr(vbase + 2, lane, DST_SENTINEL);
      auto nan = [&](uint32_t v) { return is_f16 ? is_f16_nan(v & 0xFFFFu) : is_f32_nan(v); };
      auto zero = [&](uint32_t v) { return is_f16 ? is_f16_zero(v & 0xFFFFu) : is_f32_zero(v); };
      // Skip a lane when its result may legitimately differ: a NaN input (NaN
      // payload) or a signed-zero tie (-0 vs +0). Both are accepted divergences.
      nan_lane[lane] = nan(r0) || nan(r1) || (zero(r0) && zero(r1));
    }
    wf->set_exec(exec);
    return nan_lane;
  }

  std::array<uint32_t, WF_SIZE> snapshot_dst() const {
    std::array<uint32_t, WF_SIZE> out{};
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vbase + 2, lane);
    return out;
  }
};

struct MinMaxCase {
  const char *label;
  uint32_t opcode;
  bool is_f16;
};

const MinMaxCase kCases[] = {
    {"v_max_f32", 11, false},
    {"v_min_f32", 10, false},
    {"v_max_f16", 45, true},
    {"v_min_f16", 46, true},
};

void check_case(const MinMaxCase &c, uint64_t exec) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  uint32_t enc = vop2_encode(c.opcode, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256);
  uint32_t words[4] = {enc, 0u, 0u, 0u};
  Instruction *inst = fx.decoder->decode(words);
  ASSERT_NE(inst, nullptr) << c.label << ": decode failed";

  constexpr uint64_t SEED = 0x3E12'7777'1234ULL;

  amdgpu::simd_force_scalar() = true;
  fx.seed_inputs(SEED, c.is_f16, exec);
  fx.cu->execute_instruction(inst, *fx.wf);
  auto scalar = fx.snapshot_dst();

  amdgpu::simd_force_scalar() = false;
  auto nan_lane = fx.seed_inputs(SEED, c.is_f16, exec);
  fx.cu->execute_instruction(inst, *fx.wf);
  auto simd = fx.snapshot_dst();

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (exec >> lane) & 1ULL;
    if (nan_lane[lane])
      continue; // accepted divergence: NaN payload or signed-zero tie
    EXPECT_EQ(scalar[lane], simd[lane]) << c.label << ": divergence at lane " << lane << std::hex
                                        << " scalar=0x" << scalar[lane] << " simd=0x" << simd[lane];
    if (!active) {
      EXPECT_EQ(simd[lane], DST_SENTINEL) << c.label << ": SIMD clobbered inactive lane " << lane;
    }
  }
  delete inst;
}

TEST(Vop2MinMaxSimdCorrectness, FullExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/~0ULL);
}

TEST(Vop2MinMaxSimdCorrectness, PartialExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
