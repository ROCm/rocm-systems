// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file v_add_simd_benchmark.cpp
/// @brief A/B microbenchmark: SIMD vs forced-scalar execution of v_add_f32
/// and v_add_u32 (CDNA4, VOP2 form). Both modes run in the same process; the
/// `amdgpu::simd_force_scalar()` thread-local flag flips between them.
///
/// Reports ns/instruction and speedup, and asserts SIMD-vs-scalar results
/// are bit-identical for u32 and within 0 ULP for fp32 add (IEEE-754
/// binary32 single-rounded; matches the host SIMD adder).

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
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace rocjitsu;
using Clock = std::chrono::steady_clock;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr int ITERATIONS = 200'000;

// CDNA4 VOP2 encoding helper: opcode in [30:25], vdst in [24:17],
// vsrc1 in [16:9], src0 in [8:0]. Bit 31 = 0 for VOP2.
constexpr uint32_t vop2_encode(uint32_t opcode, uint32_t vdst, uint32_t vsrc1, uint32_t src0) {
  return ((opcode & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF);
}

struct BenchFixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  BenchFixture() : gpu_mem("vadd_simd_bench_mem"), l2("vadd_simd_bench_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vadd_simd", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  // Fill v0 and v1 with deterministic random uint32 values across all lanes.
  // If `sanitize_finite` is set, force each lane to a finite normal IEEE-754
  // binary32 value (no NaN, no Inf, no denormal): clear the sign bit (sign
  // stays zero — we don't need negatives for the host-SIMD equivalence
  // check), then remap the exponent field into a finite-normal range.
  // Mantissa untouched. v_add_f32 over these inputs produces a bit-identical
  // result on host SIMD vs the scalar generated body.
  void seed_inputs(uint64_t seed, bool sanitize_finite = false) {
    std::mt19937_64 rng(seed);
    uint32_t vbase = wf->vgpr_alloc().base;
    auto sanitize = [](uint32_t raw) -> uint32_t {
      uint32_t mantissa = raw & 0x007FFFFFu;
      uint32_t raw_exp = (raw >> 23) & 0xFFu;
      uint32_t exp = 0x40u | (raw_exp & 0x7Eu);
      return (exp << 23) | mantissa;
    };
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint32_t r0 = static_cast<uint32_t>(rng());
      uint32_t r1 = static_cast<uint32_t>(rng());
      if (sanitize_finite) {
        r0 = sanitize(r0);
        r1 = sanitize(r1);
      }
      cu->write_vgpr(vbase + 0, lane, r0);
      cu->write_vgpr(vbase + 1, lane, r1);
      cu->write_vgpr(vbase + 2, lane, 0u); // dst
    }
    wf->set_exec(~0ULL); // All lanes active.
  }

  std::array<uint32_t, WF_SIZE> snapshot_v2() const {
    std::array<uint32_t, WF_SIZE> out{};
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vbase + 2, lane);
    return out;
  }
};

struct ModeStats {
  double ns_per_inst = 0;
  double mips = 0;
  uint64_t total_ns = 0;
};

ModeStats time_mode(BenchFixture &fx, Instruction *inst, bool force_scalar, uint64_t seed,
                    bool sanitize_finite) {
  amdgpu::simd_force_scalar() = force_scalar;
  fx.seed_inputs(seed, sanitize_finite);
  // Warmup.
  for (int i = 0; i < 100; ++i)
    fx.cu->execute_instruction(inst, *fx.wf);
  fx.seed_inputs(seed, sanitize_finite);
  auto t0 = Clock::now();
  for (int i = 0; i < ITERATIONS; ++i)
    fx.cu->execute_instruction(inst, *fx.wf);
  auto t1 = Clock::now();
  amdgpu::simd_force_scalar() = false;
  ModeStats s;
  s.total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  s.ns_per_inst = static_cast<double>(s.total_ns) / ITERATIONS;
  s.mips = (1e9 / s.ns_per_inst) / 1e6;
  return s;
}

void run_one(const char *label, uint32_t encoding_word, bool sanitize_finite) {
  BenchFixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.decoder, nullptr);
  ASSERT_NE(fx.wf, nullptr);

  uint32_t words[4] = {encoding_word, 0u, 0u, 0u};
  Instruction *inst = fx.decoder->decode(words);
  ASSERT_NE(inst, nullptr) << label << ": decode failed";

  constexpr uint64_t SEED = 0xC0FFEE'1234'5678ULL;

  // Correctness pre-check: snapshot both modes' results lane-by-lane.
  amdgpu::simd_force_scalar() = true;
  fx.seed_inputs(SEED, sanitize_finite);
  fx.cu->execute_instruction(inst, *fx.wf);
  auto result_scalar = fx.snapshot_v2();

  amdgpu::simd_force_scalar() = false;
  fx.seed_inputs(SEED, sanitize_finite);
  fx.cu->execute_instruction(inst, *fx.wf);
  auto result_simd = fx.snapshot_v2();
  amdgpu::simd_force_scalar() = false;

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    EXPECT_EQ(result_scalar[lane], result_simd[lane])
        << label << ": divergence at lane " << lane << " scalar=0x" << std::hex
        << result_scalar[lane] << " simd=0x" << result_simd[lane];
  }

  ModeStats sc = time_mode(fx, inst, /*force_scalar=*/true, SEED, sanitize_finite);
  ModeStats sd = time_mode(fx, inst, /*force_scalar=*/false, SEED, sanitize_finite);

  double speedup = (sd.ns_per_inst > 0) ? (sc.ns_per_inst / sd.ns_per_inst) : 0.0;
  std::printf("\n  === %s (CDNA4 VOP2, wave64) ===\n"
              "  iterations: %d  enc: 0x%08x\n"
              "  scalar : %7.1f ns/inst  (%6.2f MIPS)  wall %.1f ms\n"
              "  simd   : %7.1f ns/inst  (%6.2f MIPS)  wall %.1f ms\n"
              "  speedup: %5.2fx\n",
              label, ITERATIONS, encoding_word, sc.ns_per_inst, sc.mips,
              static_cast<double>(sc.total_ns) / 1e6, sd.ns_per_inst, sd.mips,
              static_cast<double>(sd.total_ns) / 1e6, speedup);

  EXPECT_GT(sc.mips, 0.01) << label << ": scalar baseline too slow";
  EXPECT_GT(sd.mips, 0.01) << label << ": simd path too slow";

  delete inst;
}

} // namespace

// v_add_f32 v2, v0, v1  (CDNA4 VOP2 opcode 1)
TEST(VAddSimdBenchmark, Cdna4_VAddF32_Vop2) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  // src0 = 256 (= v0 in VOP2 SRC encoding), vsrc1 = 1 (= v1), vdst = 2 (= v2).
  uint32_t enc = vop2_encode(/*opcode=*/1, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256);
  run_one("v_add_f32 v2, v0, v1", enc, /*sanitize_finite=*/true);
}

// v_add_u32 v2, v0, v1  (CDNA4 VOP2 opcode 52, no carry)
TEST(VAddSimdBenchmark, Cdna4_VAddU32_Vop2) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t enc = vop2_encode(/*opcode=*/52, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256);
  run_one("v_add_u32 v2, v0, v1", enc, /*sanitize_finite=*/false);
}

// Diagnostic: report whether the SIMD fast path is compiled in.
TEST(VAddSimdBenchmark, SimdCompileTimeReport) {
#if __has_include(<experimental/simd>)
  using simd_u32 = std::experimental::native_simd<uint32_t>;
  std::printf("\n  util::has_stdx_simd=true, native_simd<uint32_t>::size() = %zu\n",
              simd_u32::size());
  SUCCEED();
#else
  // Scalar fallback is documented as correct; missing <experimental/simd> is
  // a host capability, not a defect.
  GTEST_SKIP() << "util::has_stdx_simd=false — SIMD path disabled at compile time";
#endif
}
