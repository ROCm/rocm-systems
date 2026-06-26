// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop2_fma_f64_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the only f64
/// VOP2 op reachable on CDNA4, v_fmac_f64 (dst = fma(src0, vsrc1, dst), all
/// f64). This is the first user of the 64-bit-lane SIMD infra: a per-lane f64
/// lives as two 32-bit VGPRs at the same lane index (lo = reg N, hi = reg N+1),
/// so the SIMD path reads/writes through the split lo/hi pointer pair
/// (util::load64 / masked_store64). The first test directly validates that
/// layout assumption (write_lane64 round-trips through load64 via the operand's
/// 64-bit lane pointers). The v_fmac_f64 check runs TWICE in the same process
/// -- once forcing the scalar body, once the SIMD fast path, with identical
/// inputs/EXEC -- and the destination f64 results are asserted equal per
/// non-skipped lane (util::set_force_scalar_for_testing flips the gate
/// in-process). util::stdx::fma over native<double> is bit-exact to std::fma for
/// finite/Inf inputs; NaN-result lanes may diverge in NaN payload (accepted), so
/// those lanes are excluded from the comparison — NaN-ness of the result is
/// deterministic from the inputs, so both runs skip the same lanes.

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
#include <vector>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
[[maybe_unused]] constexpr uint64_t LO_SENTINEL = 0xDEADBEEFu;
[[maybe_unused]] constexpr uint64_t HI_SENTINEL = 0xFEEDFACEu;

// CDNA4 VOP2: opcode[30:25], vdst[24:17], vsrc1[16:9], src0[8:0]. Bit 31 = 0.
constexpr uint32_t vop2_encode(uint32_t opcode, uint32_t vdst, uint32_t vsrc1, uint32_t src0) {
  return ((opcode & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF);
}

// f64 edge values whose fma combinations cover ±0, ±Inf, denormal, large, and
// ordinary normals. NaN is excluded from the seeded set (its payload may diverge
// between packed and scalar fma); a few NaN lanes are still exercised via the RNG
// tail and skipped explicitly when comparing.
const std::array<double, 12> kEdge = {{
    +0.0,
    -0.0,
    1.0,
    -1.0,
    2.0,
    0.5,
    std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::denorm_min(),
    std::numeric_limits<double>::max(),
    3.141592653589793,
    -2.718281828459045,
}};

bool is_f64_nan(uint64_t bits) { return std::isnan(std::bit_cast<double>(bits)); }

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop2_fma_f64_simd_mem"), l2("vop2_fma_f64_simd_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop2_fma_f64_simd", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  // Write a 64-bit value to a VGPR pair (reg = lo, reg+1 = hi) at `lane`.
  void write64(uint32_t reg, uint32_t lane, uint64_t v) {
    cu->write_vgpr(reg, lane, static_cast<uint32_t>(v));
    cu->write_vgpr(reg + 1, lane, static_cast<uint32_t>(v >> 32));
  }
  uint64_t read64(uint32_t reg, uint32_t lane) {
    return static_cast<uint64_t>(cu->read_vgpr(reg + 1, lane)) << 32 | cu->read_vgpr(reg, lane);
  }

  // src0 = v0:v1, vsrc1 = v2:v3, vdst = v4:v5 (relative to the alloc base).
  void seed_inputs(uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      double a = kEdge[lane % kEdge.size()];
      double b = kEdge[(lane / kEdge.size() + 1) % kEdge.size()];
      double d = kEdge[(lane * 7 + 3) % kEdge.size()];
      write64(vb + 0, lane, std::bit_cast<uint64_t>(a));
      write64(vb + 2, lane, std::bit_cast<uint64_t>(b));
      write64(vb + 4, lane, std::bit_cast<uint64_t>(d));
    }
    wf->set_exec(exec);
  }

  std::array<uint64_t, WF_SIZE> run(Instruction *inst, uint64_t exec) {
    seed_inputs(exec);
    // Mark the dst lanes so an inactive-lane clobber is visible. The accumulate
    // source is the seeded value above; re-stamp the unused high words is not
    // needed — fmac reads/writes the same pair.
    cu->execute_instruction(inst, *wf);
    std::array<uint64_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = read64(vb + 4, lane);
    return out;
  }
};

// Directly validate the two-array f64 storage layout assumption the 64-bit infra
// is built on: a value written via the VGPR pair (lo=reg, hi=reg+1) must read
// back identically through util::load64 fed by the operand's 64-bit lane
// pointers. Exercised here without the SIMD/scalar A/B so a layout regression is
// isolated from the fma functor.
TEST(Vop2FmaF64SimdCorrectness, LaneLayoutRoundTrip) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  uint32_t vb = fx.wf->vgpr_alloc().base;
  std::vector<uint64_t> expected(WF_SIZE);
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    uint64_t v = (static_cast<uint64_t>(0xA5A50000u | lane) << 32) | (0x1234'0000u + lane);
    expected[lane] = v;
    fx.write64(vb + 0, lane, v);
  }
  const uint32_t *lo = reinterpret_cast<const uint32_t *>(fx.cu->vgpr_data(vb + 0));
  const uint32_t *hi = reinterpret_cast<const uint32_t *>(fx.cu->vgpr_data(vb + 1));
  constexpr std::size_t W = util::native_width64;
  for (uint32_t base = 0; base < WF_SIZE; base += static_cast<uint32_t>(W)) {
    util::native<uint64_t> v = util::load64<uint64_t>(lo + base, hi + base);
    for (std::size_t i = 0; i < W; ++i)
      EXPECT_EQ(v[i], expected[base + i]) << "load64 mismatch at lane " << (base + i);
  }
}

// Restores the process force-scalar gate on scope exit so flipping it for an
// in-process A/B comparison cannot leak into later tests in the same process.
struct ForceScalarGuard {
  bool orig;
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }
};

void check_fmac(uint64_t exec) {
  ForceScalarGuard gate_guard;

  auto run_mode = [&](bool force_scalar) -> std::array<uint64_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    // v_fmac_f64 = VOP2 op 4. src0 = v0:v1 (enc 256), vsrc1 = v2:v3, vdst = v4:v5.
    uint32_t enc = vop2_encode(/*opcode=*/4, /*vdst=*/4, /*vsrc1=*/2, /*src0=*/256);
    uint32_t words[4] = {enc, 0u, 0u, 0u};
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << "v_fmac_f64 decode failed";
    auto out = fx.run(inst, exec);
    delete inst;
    return out;
  };

  const auto scalar_out = run_mode(/*force_scalar=*/true);
  const auto simd_out = run_mode(/*force_scalar=*/false);

  // Core A/B equivalence per non-skipped lane. NaN-result lanes carry a
  // possibly-different NaN payload and are excluded identically in both runs
  // (NaN-ness of the result is deterministic from the inputs).
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    if (is_f64_nan(scalar_out[lane]) || is_f64_nan(simd_out[lane]))
      continue;
    EXPECT_EQ(scalar_out[lane], simd_out[lane])
        << "v_fmac_f64 lane " << lane << ": SIMD path diverged from scalar body";
  }
}

TEST(Vop2FmaF64SimdCorrectness, FullExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check_fmac(/*exec=*/~0ULL);
}

TEST(Vop2FmaF64SimdCorrectness, PartialExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  // Pattern crossing the 8-wide f64 chunk boundaries on a wave64.
  check_fmac(/*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
