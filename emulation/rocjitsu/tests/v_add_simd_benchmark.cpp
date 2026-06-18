// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file v_add_simd_benchmark.cpp
/// @brief Microbenchmark of v_add_f32 / v_add_u32 (CDNA4, VOP2 form). The
/// execute mode is fixed per process by `RJ_FORCE_SCALAR` (force_scalar() is
/// immutable), so each run times one mode and reports ns/instruction. Compare
/// SIMD vs scalar throughput by running twice: `RJ_FORCE_SCALAR=1` (scalar)
/// then unset (SIMD). Scalar-vs-SIMD bit-equivalence is verified separately by
/// the *SimdCorrectness* suites (in-process double-run + EXPECT_EQ), not here.

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

// CDNA4 VOP1 encoding: [31:25]=0b0111111, vdst[24:17], op[16:9], src0[8:0].
constexpr uint32_t vop1_encode(uint32_t opcode, uint32_t vdst, uint32_t src0) {
  return (0x3Fu << 25) | ((vdst & 0xFF) << 17) | ((opcode & 0xFF) << 9) | (src0 & 0x1FF);
}

// CDNA4 VOP3 binary encoding ([31:26]=0x34): word0 = vdst[7:0] | abs[10:8] |
// clamp[15] | op[25:16]; word1 = src0[8:0] | src1[17:9] | omod[28:27] | neg[31:29].
constexpr void vop3_bin_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1,
                               uint32_t abs, uint32_t neg, uint32_t omod, uint32_t clamp,
                               uint32_t &w0, uint32_t &w1) {
  w0 = (vdst & 0xFF) | ((abs & 0x7) << 8) | ((clamp & 0x1) << 15) | ((op & 0x3FF) << 16) |
       (0x34u << 26);
  w1 = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((omod & 0x3) << 27) | ((neg & 0x7) << 29);
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

ModeStats time_mode(BenchFixture &fx, Instruction *inst, uint64_t seed, bool sanitize_finite) {
  fx.seed_inputs(seed, sanitize_finite);
  // Warmup.
  for (int i = 0; i < 100; ++i)
    fx.cu->execute_instruction(inst, *fx.wf);
  fx.seed_inputs(seed, sanitize_finite);
  auto t0 = Clock::now();
  for (int i = 0; i < ITERATIONS; ++i)
    fx.cu->execute_instruction(inst, *fx.wf);
  auto t1 = Clock::now();
  ModeStats s;
  s.total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  s.ns_per_inst = static_cast<double>(s.total_ns) / ITERATIONS;
  s.mips = (1e9 / s.ns_per_inst) / 1e6;
  return s;
}

void run_words(const char *label, uint32_t w0, uint32_t w1, bool sanitize_finite) {
  BenchFixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.decoder, nullptr);
  ASSERT_NE(fx.wf, nullptr);

  uint32_t words[4] = {w0, w1, 0u, 0u};
  Instruction *inst = fx.decoder->decode(words);
  ASSERT_NE(inst, nullptr) << label << ": decode failed";

  constexpr uint64_t SEED = 0xC0FFEE'1234'5678ULL;

  // This benchmark times whichever execute mode the process starts in (seeded
  // from RJ_FORCE_SCALAR). Scalar-vs-SIMD bit-equivalence is checked separately
  // by the *SimdCorrectness* suites (in-process double-run + EXPECT_EQ). To
  // compare throughput, run the benchmark twice: `RJ_FORCE_SCALAR=1 ...`
  // (scalar) and unset (SIMD).
  const char *mode = amdgpu::simd_force_scalar() ? "scalar" : "simd";
  ModeStats s = time_mode(fx, inst, SEED, sanitize_finite);

  std::printf("\n  === %s (CDNA4, wave64) [%s] ===\n"
              "  iterations: %d  enc: 0x%08x 0x%08x\n"
              "  %-6s : %7.1f ns/inst  (%6.2f MIPS)  wall %.1f ms\n",
              label, mode, ITERATIONS, w0, w1, mode, s.ns_per_inst, s.mips,
              static_cast<double>(s.total_ns) / 1e6);

  EXPECT_GT(s.mips, 0.01) << label << ": " << mode << " path too slow";

  delete inst;
}

// Single-word (VOP1/VOP2/VOPC) wrapper.
void run_one(const char *label, uint32_t encoding_word, bool sanitize_finite) {
  run_words(label, encoding_word, 0u, sanitize_finite);
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

// v_add_u16 v2, v0, v1  (CDNA4 VOP2 opcode 38) — 16-bit integer, low-16 path.
TEST(VAddSimdBenchmark, Cdna4_VAddU16_Vop2) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t enc = vop2_encode(/*opcode=*/38, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256);
  run_one("v_add_u16 v2, v0, v1", enc, /*sanitize_finite=*/false);
}

// v_add_f16 v2, v0, v1  (CDNA4 VOP2 opcode 31) — f16 op via f32 intermediate.
TEST(VAddSimdBenchmark, Cdna4_VAddF16_Vop2) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t enc = vop2_encode(/*opcode=*/31, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256);
  run_one("v_add_f16 v2, v0, v1", enc, /*sanitize_finite=*/false);
}

// v_add_co_u32 v2, v0, v1  (CDNA4 VOP2 opcode 25) — carry-out into VCC. The
// SIMD path pays a per-chunk VCC pack/merge (and a scalar carry-in expand)
// on top of the binary add, so this shows the carry-glue overhead vs v_add_u32.
TEST(VAddSimdBenchmark, Cdna4_VAddCoU32_Vop2) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t enc = vop2_encode(/*opcode=*/25, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256);
  run_one("v_add_co_u32 v2, v0, v1", enc, /*sanitize_finite=*/false);
}

// v_fmaak_f32 v2, v0, v1, K  (CDNA4 VOP2 opcode 24) — dst = fma(v0, v1, K). The
// inline literal K is words[1] (0.0f here), so this exercises the literal-form
// ternary glue. Inputs sanitized to finite for bit-exact comparison.
//
// NOTE: the dst-accumulate forms (v_fmac/v_mac, dst = fma(s0, s1, dst)) are not
// benchmarked here. Looping the *same* instruction re-creates a loop-carried
// RAW dependency on the accumulator register, which serializes both the scalar
// and SIMD runs on store->load latency rather than compute throughput, so the
// measured "speedup" (~1x) reflects the microbenchmark shape, not the fast path
// (which does engage — see Vop2FmaSimdCorrectness). The literal form below has
// no such loop-carried dependency and shows the representative speedup.
TEST(VAddSimdBenchmark, Cdna4_VFmaakF32_Vop2) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t enc = vop2_encode(/*opcode=*/24, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256);
  run_one("v_fmaak_f32 v2, v0, v1, 0", enc, /*sanitize_finite=*/true);
}

// v_cndmask_b32 v2, v0, v1  (CDNA4 VOP2 opcode 0) — VCC-driven per-lane select.
// VCC defaults to 0 here (all lanes pick src0); still drives the SIMD blend.
TEST(VAddSimdBenchmark, Cdna4_VCndmaskB32_Vop2) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t enc = vop2_encode(/*opcode=*/0, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256);
  run_one("v_cndmask_b32 v2, v0, v1", enc, /*sanitize_finite=*/false);
}

// v_mul_hi_u32_u24 v2, v0, v1  (CDNA4 VOP2 opcode 9) — 48-bit product, high 32
// bits via the 64-bit-lane widening path. Shows the widening-mul overhead.
TEST(VAddSimdBenchmark, Cdna4_VMulHiU32U24_Vop2) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t enc = vop2_encode(/*opcode=*/9, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256);
  run_one("v_mul_hi_u32_u24 v2, v0, v1", enc, /*sanitize_finite=*/false);
}

// v_max_f32 v2, v0, v1  (CDNA4 VOP2 opcode 11) — std::fmax; inputs sanitized to
// finite normals for the bit-exact pre-check.
TEST(VAddSimdBenchmark, Cdna4_VMaxF32_Vop2) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t enc = vop2_encode(/*opcode=*/11, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256);
  run_one("v_max_f32 v2, v0, v1", enc, /*sanitize_finite=*/true);
}

// v_ldexp_f16 v2, v0, v1  (CDNA4 VOP2 opcode 51) — f16 ldexp; vsrc1 is the
// signed exponent. Raw inputs (ldexp NaN handling is bit-exact).
TEST(VAddSimdBenchmark, Cdna4_VLdexpF16_Vop2) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t enc = vop2_encode(/*opcode=*/51, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256);
  run_one("v_ldexp_f16 v2, v0, v1", enc, /*sanitize_finite=*/false);
}

// v_ceil_f16 v2, v0  (CDNA4 VOP1 opcode 69) — f16 op via f32 intermediate.
TEST(VAddSimdBenchmark, Cdna4_VCeilF16_Vop1) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t enc = vop1_encode(/*opcode=*/69, /*vdst=*/2, /*src0=*/256);
  run_one("v_ceil_f16 v2, v0", enc, /*sanitize_finite=*/false);
}

// v_cmp_lt_f64 (VOPC64 opcode 97): a real (non-accumulate) 64-bit-lane SIMD op
// exercising the split lo/hi VGPR-pair read path. src0 = v0:v1, vsrc1 = v2:v3;
// the compare result lands in VCC (run_one's v2 correctness pre-check is a no-op
// here, but the timing is what matters). f64 fmac is NOT benched: its dst-
// accumulate loop-carried RAW serializes both modes into a ~1x artifact.
TEST(VAddSimdBenchmark, Cdna4_VCmpLtF64_Vopc) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  // VOPC: [31:25]=0x3E, op[24:17], vsrc1[16:9], src0[8:0].
  uint32_t enc = (0x3Eu << 25) | (97u << 17) | (2u << 9) | 256u;
  run_one("v_cmp_lt_f64 v0:v1, v2:v3", enc, /*sanitize_finite=*/false);
}

// v_cvt_i32_f64 v2, v0:v1  (CDNA4 VOP1 opcode 3) — f64 src -> 32-bit i32 dst via
// the mixed-width cvt glue (8-wide f64 chunk, scalar lo/hi gather, clamp/NaN in
// the double domain). Exact for any input (NaN->0, saturating clamp), so the
// strict bit-exact pre-check holds on raw random doubles.
TEST(VAddSimdBenchmark, Cdna4_VCvtI32F64_Vop1) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t enc = vop1_encode(/*opcode=*/3, /*vdst=*/2, /*src0=*/256);
  run_one("v_cvt_i32_f64 v2, v0:v1", enc, /*sanitize_finite=*/false);
}

// v_cvt_f64_i32 v2:v3, v0  (CDNA4 VOP1 opcode 4) — 32-bit i32 src -> f64 dst
// (exact widening) via the b32->f64 cvt glue (narrow32 load, write_simd64).
TEST(VAddSimdBenchmark, Cdna4_VCvtF64I32_Vop1) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t enc = vop1_encode(/*opcode=*/4, /*vdst=*/2, /*src0=*/256);
  run_one("v_cvt_f64_i32 v2:v3, v0", enc, /*sanitize_finite=*/false);
}

// v_cmp_class_f32 (VOPC opcode 16): src0 = v0, vsrc1 = v1 (10-bit class mask);
// result -> VCC. The class-decode functor over the 32-bit VOPC glue. v2 is
// untouched, so run_one's correctness pre-check is a no-op; timing is the point.
TEST(VAddSimdBenchmark, Cdna4_VCmpClassF32_Vopc) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t enc = (0x3Eu << 25) | (16u << 17) | (1u << 9) | 256u;
  run_one("v_cmp_class_f32 v0, v1", enc, /*sanitize_finite=*/false);
}

// v_cmp_class_f64 (VOPC opcode 18): src0 = v0:v1 (f64), vsrc1 = v2 (32-bit mask);
// result -> VCC. The mixed-width class glue (read_simd64 value + narrow32 mask).
TEST(VAddSimdBenchmark, Cdna4_VCmpClassF64_Vopc) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t enc = (0x3Eu << 25) | (18u << 17) | (2u << 9) | 256u;
  run_one("v_cmp_class_f64 v0:v1, v2", enc, /*sanitize_finite=*/false);
}

// v_cmp_class_f32_e64 (VOP3 opcode 16): the VOP3 form of v_cmp_class — abs/neg
// modifiers (0 here), mask from src1=v1, result into the SGPR-pair dst (VCC).
// Shows the VOP3 class glue cost vs the VOPC form. word0[31:26]=0x34.
TEST(VAddSimdBenchmark, Cdna4_VCmpClassF32_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0 = (106u & 0xFF) | ((16u & 0x3FF) << 16) | (0x34u << 26); // vdst=VCC, op=16
  uint32_t w1 = 256u | (1u << 9);                                      // src0=v0, src1=v1
  run_words("v_cmp_class_f32_e64 v0, v1 -> vcc", w0, w1, /*sanitize_finite=*/false);
}

// === VOP3-encoded binary twins (src0/src1, abs/neg/omod/clamp modifiers) ===

// v_add_f32_e64 v2, v0, v1  (CDNA4 VOP3 opcode 257) — no modifiers. Baseline
// VOP3 f32 binary; should match the VOP2 v_add_f32 speedup (same functor, the
// fp glue's modifier reads are zero-cost when all modifier fields are 0).
TEST(VAddSimdBenchmark, Cdna4_VAddF32_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(257, /*vdst=*/2, /*src0=*/256, /*src1=*/257, 0, 0, 0, 0, w0, w1);
  run_words("v_add_f32_e64 v2, v0, v1", w0, w1, /*sanitize_finite=*/true);
}

// v_add_f32_e64 v2, |v0|, -v1 *2 clamp  (CDNA4 VOP3 opcode 257) — all four
// modifier kinds active. Shows the in-vector abs/neg/omod/clamp cost vs the
// no-modifier form above. abs=src0, neg=src1, omod=1 (*2), clamp=1.
TEST(VAddSimdBenchmark, Cdna4_VAddF32_Vop3_Modifiers) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(257, /*vdst=*/2, /*src0=*/256, /*src1=*/257, /*abs=*/0x1, /*neg=*/0x2, /*omod=*/1,
                  /*clamp=*/1, w0, w1);
  run_words("v_add_f32_e64 |v0|,-v1 *2 clamp", w0, w1, /*sanitize_finite=*/true);
}

// v_mul_f32_e64 v2, v0, v1  (CDNA4 VOP3 opcode 261) — no modifiers.
TEST(VAddSimdBenchmark, Cdna4_VMulF32_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(261, /*vdst=*/2, /*src0=*/256, /*src1=*/257, 0, 0, 0, 0, w0, w1);
  run_words("v_mul_f32_e64 v2, v0, v1", w0, w1, /*sanitize_finite=*/true);
}

// v_and_b32_e64 v2, v0, v1  (CDNA4 VOP3 opcode 275) — integer/bitwise, plain.
TEST(VAddSimdBenchmark, Cdna4_VAndB32_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(275, /*vdst=*/2, /*src0=*/256, /*src1=*/257, 0, 0, 0, 0, w0, w1);
  run_words("v_and_b32_e64 v2, v0, v1", w0, w1, /*sanitize_finite=*/false);
}

// v_add_u32_e64 v2, v0, v1  (CDNA4 VOP3 opcode 308) — integer, plain.
TEST(VAddSimdBenchmark, Cdna4_VAddU32_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(308, /*vdst=*/2, /*src0=*/256, /*src1=*/257, 0, 0, 0, 0, w0, w1);
  run_words("v_add_u32_e64 v2, v0, v1", w0, w1, /*sanitize_finite=*/false);
}

// === VOP3-encoded unary twins ===

// v_floor_f32_e64 v2, v0  (CDNA4 VOP3 opcode 351) — f32 unary, no modifiers.
TEST(VAddSimdBenchmark, Cdna4_VFloorF32_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(351, /*vdst=*/2, /*src0=*/256, /*src1=*/0, 0, 0, 0, 0, w0, w1);
  run_words("v_floor_f32_e64 v2, v0", w0, w1, /*sanitize_finite=*/true);
}

// v_floor_f32_e64 v2, -|v0| *2 clamp  (opcode 351) — full modifier set, shows
// the in-vector abs/neg/omod/clamp cost on the unary fp path.
TEST(VAddSimdBenchmark, Cdna4_VFloorF32_Vop3_Modifiers) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(351, /*vdst=*/2, /*src0=*/256, /*src1=*/0, /*abs=*/0x1, /*neg=*/0x1, /*omod=*/1,
                  /*clamp=*/1, w0, w1);
  run_words("v_floor_f32_e64 -|v0| *2 clamp", w0, w1, /*sanitize_finite=*/true);
}

// v_cvt_f32_i32_e64 v2, v0  (CDNA4 VOP3 opcode 325) — plain int->float cvt
// (reuses the VOP1 unary path, no modifiers).
TEST(VAddSimdBenchmark, Cdna4_VCvtF32I32_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(325, /*vdst=*/2, /*src0=*/256, /*src1=*/0, 0, 0, 0, 0, w0, w1);
  run_words("v_cvt_f32_i32_e64 v2, v0", w0, w1, /*sanitize_finite=*/false);
}

// === VOP3 form of the VOPC integer compares ===
// dst is the SGPR-pair (we point at VCC=106) instead of the fixed VCC. v2 stays
// untouched, so the correctness pre-check is a no-op (the compare result lands in
// VCC); timing is the point.

// v_cmp_lt_i32_e64 vcc, v0, v1  (CDNA4 VOP3 opcode 193) — 32-bit-lane integer.
TEST(VAddSimdBenchmark, Cdna4_VCmpLtI32_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(193, /*vdst=*/106, /*src0=*/256, /*src1=*/257, 0, 0, 0, 0, w0, w1);
  run_words("v_cmp_lt_i32_e64 vcc, v0, v1", w0, w1, /*sanitize_finite=*/false);
}

// v_cmp_lt_i64_e64 vcc, v0:v1, v2:v3  (CDNA4 VOP3 opcode 225) — 64-bit-lane
// integer; exercises the split lo/hi VGPR-pair gather/scatter path (~half the
// 32-bit speedup because 8-wide chunks + scalar lo/hi reads).
TEST(VAddSimdBenchmark, Cdna4_VCmpLtI64_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(225, /*vdst=*/106, /*src0=*/256, /*src1=*/258, 0, 0, 0, 0, w0, w1);
  run_words("v_cmp_lt_i64_e64 vcc, v0:v1, v2:v3", w0, w1, /*sanitize_finite=*/false);
}

// v_cmp_eq_f32_e64 vcc, v0, v1  (CDNA4 VOP3 opcode 66) — f32 with no modifiers.
TEST(VAddSimdBenchmark, Cdna4_VCmpEqF32_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(66, /*vdst=*/106, /*src0=*/256, /*src1=*/257, 0, 0, 0, 0, w0, w1);
  run_words("v_cmp_eq_f32_e64 vcc, v0, v1", w0, w1, /*sanitize_finite=*/true);
}

// v_cmp_eq_f32_e64 vcc, |v0|, -v1  (opcode 66) — both src abs/neg modifiers
// set; shows the in-vector apply_vop3_src_mod_f32 cost vs the no-modifier form.
TEST(VAddSimdBenchmark, Cdna4_VCmpEqF32_Vop3_Modifiers) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(66, /*vdst=*/106, /*src0=*/256, /*src1=*/257, /*abs=*/0x1, /*neg=*/0x2, 0, 0, w0,
                  w1);
  run_words("v_cmp_eq_f32_e64 vcc, |v0|, -v1", w0, w1, /*sanitize_finite=*/true);
}

// v_cmp_eq_f16_e64 vcc, v0, v1  (CDNA4 VOP3 opcode 34) — f16 widen-then-compare.
// Pays the f16_to_f32_simd widen tax on both operands; no modifiers.
TEST(VAddSimdBenchmark, Cdna4_VCmpEqF16_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(34, /*vdst=*/106, /*src0=*/256, /*src1=*/257, 0, 0, 0, 0, w0, w1);
  run_words("v_cmp_eq_f16_e64 vcc, v0, v1", w0, w1, /*sanitize_finite=*/false);
}

// v_cmp_eq_f64_e64 vcc, v0:v1, v2:v3  (CDNA4 VOP3 opcode 98) — 64-bit-lane
// f64, split lo/hi gather/scatter, no modifiers.
TEST(VAddSimdBenchmark, Cdna4_VCmpEqF64_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(98, /*vdst=*/106, /*src0=*/256, /*src1=*/258, 0, 0, 0, 0, w0, w1);
  run_words("v_cmp_eq_f64_e64 vcc, v0:v1, v2:v3", w0, w1, /*sanitize_finite=*/false);
}

// === VOP3 integer ternary (src0/src1/src2, no modifiers) ===
//
// v_add3_u32 v2, v0, v1, v3  (CDNA4 VOP3 opcode 511) — plain element-wise
// a + b + c on 32-bit lanes. src2 is wired through the ternary VOP3 glue.
TEST(VAddSimdBenchmark, Cdna4_VAdd3U32_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  // VOP3: word0 = vdst|op<<16|0x34<<26; word1 = src0|src1<<9|src2<<18.
  uint32_t w0 = (2u) | ((511u & 0x3FF) << 16) | (0x34u << 26);
  uint32_t w1 = 256u | (1u << 9) | (3u << 18); // src0=v0 src1=v1 src2=v3
  run_words("v_add3_u32_e64 v2, v0, v1, v3", w0, w1, /*sanitize_finite=*/false);
}

// v_cvt_i32_f64_e64 v2, v0:v1  (CDNA4 VOP3 opcode 323) — VOP3 form of the f64
// cvt. Routed through the existing SIMD_CVT_F64_TO_B32 glue via a _vop3->_vop1
// fallback (the scalar body drops the abs/neg/omod/clamp modifier reads).
TEST(VAddSimdBenchmark, Cdna4_VCvtI32F64_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(323, /*vdst=*/2, /*src0=*/256, /*src1=*/0, 0, 0, 0, 0, w0, w1);
  run_words("v_cvt_i32_f64_e64 v2, v0:v1", w0, w1, /*sanitize_finite=*/false);
}

// v_add_f64_e64 v2:v3, v0:v1, v4:v5  (CDNA4 VOP3 opcode 640) — 64-bit-lane f64
// binary via the new try_execute_binary_vop3_fp64_simd glue.
TEST(VAddSimdBenchmark, Cdna4_VAddF64_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(640, /*vdst=*/2, /*src0=*/256, /*src1=*/260, 0, 0, 0, 0, w0, w1);
  run_words("v_add_f64_e64 v2:v3, v0:v1, v4:v5", w0, w1, /*sanitize_finite=*/false);
}

// v_ceil_f64_e64 v2:v3, v0:v1  (CDNA4 VOP3 opcode 344) — f64 unary; uses the
// new try_execute_unary_vop3_fp64_simd glue (stdx::ceil on native<double>).
TEST(VAddSimdBenchmark, Cdna4_VCeilF64_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(344, /*vdst=*/2, /*src0=*/256, /*src1=*/0, 0, 0, 0, 0, w0, w1);
  run_words("v_ceil_f64_e64 v2:v3, v0:v1", w0, w1, /*sanitize_finite=*/false);
}

// v_ceil_f16_e64 v2, v0  (CDNA4 VOP3 opcode 389) — f16 unary; widen-then-narrow
// VOP3 glue. Pays the f16<->f32 round-trip on top of the rounding op.
TEST(VAddSimdBenchmark, Cdna4_VCeilF16_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(389, /*vdst=*/2, /*src0=*/256, /*src1=*/0, 0, 0, 0, 0, w0, w1);
  run_words("v_ceil_f16_e64 v2, v0", w0, w1, /*sanitize_finite=*/false);
}

// v_div_fmas_f32_e64 v6, v0, v1, v2  (CDNA4 VOP3 opcode 482) — fma(s0,s1,s2)
// followed by a VCC-bit-gated ldexp(result, 32); no omod/clamp. Tests the
// new try_execute_div_fmas_f32_simd glue and its VCC input-mask path.
TEST(VAddSimdBenchmark, Cdna4_VDivFmasF32_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0 = (6u) | ((482u & 0x3FF) << 16) | (0x34u << 26);
  uint32_t w1 = 256u | (1u << 9) | (2u << 18); // src0=v0 src1=v1 src2=v2
  run_words("v_div_fmas_f32_e64 v6, v0, v1, v2", w0, w1, /*sanitize_finite=*/true);
}

// v_cndmask_b32_e64 v2, v0, v1, vcc  (CDNA4 VOP3 opcode 256) — VOP3 form of
// the VCC-driven select; selector is an SGPR-pair operand (src2=VCC=106), so
// the path is bit-identical to the VOP2 form. No modifiers applied by scalar.
TEST(VAddSimdBenchmark, Cdna4_VCndmaskB32_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  // VOP3 ternary encoding: word0 = vdst|op<<16|0x34<<26; word1 = src0|src1<<9|
  // src2<<18. src2=VCC=106. Selector pattern is set by seed_inputs (vcc init).
  uint32_t w0 = (2u) | ((256u & 0x3FF) << 16) | (0x34u << 26);
  uint32_t w1 = 256u | (1u << 9) | (106u << 18);
  run_words("v_cndmask_b32_e64 v2, v0, v1, vcc", w0, w1, /*sanitize_finite=*/false);
}

// v_rcp_f16_e64 v2, v0  (CDNA4 VOP3 opcode 381) — f16 transcendental; widen
// f16->f32, FTZ-flush input, IEEE 1/x, FTZ-flush output, narrow f32->f16. Pays
// the f16<->f32 round-trip on top of the rcp helper. Same NaN-passthrough
// (input NaN bits preserved) in both scalar and SIMD paths, so the bit-exact
// pre-check holds on raw random inputs (no `sanitize_finite` needed).
TEST(VAddSimdBenchmark, Cdna4_VRcpF16_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  uint32_t w0, w1;
  vop3_bin_encode(381, /*vdst=*/2, /*src0=*/256, /*src1=*/0, 0, 0, 0, 0, w0, w1);
  run_words("v_rcp_f16_e64 v2, v0", w0, w1, /*sanitize_finite=*/false);
}

// v_fma_f32_e64 v6, v0, v1, v2  (CDNA4 VOP3 opcode 459) — f32 ternary FMA via
// the new try_execute_ternary_vop3_fp_simd glue. Encoded with src2=v2 (NOT
// dst-accumulate; the v_fmac form is deferred to a separate slice).
TEST(VAddSimdBenchmark, Cdna4_VFmaF32_Vop3) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  // VOP3 ternary: word0 vdst|op<<16|0x34<<26; word1 src0|src1<<9|src2<<18.
  uint32_t w0 = (6u) | ((459u & 0x3FF) << 16) | (0x34u << 26);
  uint32_t w1 = 256u | (1u << 9) | (2u << 18); // src0=v0 src1=v1 src2=v2
  run_words("v_fma_f32_e64 v6, v0, v1, v2", w0, w1, /*sanitize_finite=*/true);
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
