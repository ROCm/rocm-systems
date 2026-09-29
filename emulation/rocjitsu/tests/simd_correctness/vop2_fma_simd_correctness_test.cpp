// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop2_fma_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the ternary
/// VOP2 fused multiply-add family wired into SIMD_VOP2_TERNARY. Covers the
/// three operand shapes available on CDNA4:
///   dst-accumulate : v_fmac_f32 (59), v_mac_f16 (35)   -> fma(s0, s1, dst)
///   literal addend : v_fmaak_f32 (24), v_madak_f16 (37) -> fma(s0, s1, K)
///   literal mult   : v_fmamk_f32 (23), v_madmk_f16 (36) -> fma(s0, K, s1)
/// The dst-accumulate forms read vdst as the third operand, so v2 is seeded
/// with live values (not a sentinel) and the accumulate is exercised. The
/// literal forms carry an inline-constant dword (words[1]). Inputs are raw
/// random bit patterns plus explicit fp edge lanes (0, ±0, ±Inf, denormal,
/// large). fma is bit-exact for all finite/Inf inputs (incl. Inf*0->NaN); a NaN
/// *input* may propagate a different NaN payload through the packed vs scalar
/// FMA, an accepted difference, so lanes with a NaN input are excluded from the
/// per-lane comparison (the skip condition is computed from the inputs,
/// identical in both runs, so both runs skip the same lanes). Each op runs TWICE
/// in the same process -- once forcing the scalar body, once the SIMD fast path,
/// with identical seed/inputs/EXEC -- and the two result arrays are asserted
/// equal per active, non-skipped lane (util::set_force_scalar_for_testing flips
/// the gate in-process).

#include "decode_test_util.h"
#include "util/simd_test_hooks.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/execute_shared.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <cstdint>
#include <memory>
#include <random>
#include <string>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;

// CDNA4 VOP2: opcode[30:25], vdst[24:17], vsrc1[16:9], src0[8:0]. Bit 31 = 0.
constexpr uint32_t vop2_encode(uint32_t opcode, uint32_t vdst, uint32_t vsrc1, uint32_t src0) {
  return ((opcode & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF);
}

// f32 edge bit patterns placed on the low lanes to stress fma rounding/±0/Inf.
// No NaN: NaN-input lanes have an accepted payload divergence and are skipped
// in the comparison anyway (see is_f32_nan / is_f16_nan below).
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

bool is_f32_nan(uint32_t u) { return ((u >> 23) & 0xFFu) == 0xFFu && (u & 0x7FFFFFu) != 0u; }
bool is_f16_nan(uint32_t u) { return ((u >> 10) & 0x1Fu) == 0x1Fu && (u & 0x3FFu) != 0u; }

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop2_fma_simd_mem"), l2("vop2_fma_simd_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop2_fma_simd", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  // Seed v0 (src0), v1 (vsrc1), v2 (vdst/accumulator). The low lanes get the
  // fp edge patterns (in both halves for f16); the rest are raw random. v2 is
  // live for the dst-accumulate forms. Records the seeded v2 for the inactive-
  // lane preservation check.
  std::array<uint32_t, WF_SIZE> seed_inputs(uint64_t seed, bool is_f16, uint64_t exec,
                                            std::array<bool, WF_SIZE> *nan_lane = nullptr) {
    std::mt19937_64 rng(seed);
    std::array<uint32_t, WF_SIZE> seeded_dst{};
    uint32_t vbase = wf->vgpr_alloc().base;
    auto edge = [&](uint32_t i) -> uint32_t {
      uint32_t e = kF32Edges[i % kF32Edges.size()];
      if (is_f16) {
        // Pack two f16 edge halves: low = e>>16, high = e>>5 — covers f16
        // NaN/Inf/denorm/zero in the low 16 bits the op reads.
        uint32_t lo = (e >> 16) & 0xFFFFu;
        return lo | (lo << 16);
      }
      return e;
    };
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint32_t r0, r1, r2;
      if (lane < kF32Edges.size()) {
        r0 = edge(lane);
        r1 = edge(lane + 3);
        r2 = edge(lane + 6);
      } else {
        r0 = static_cast<uint32_t>(rng());
        r1 = static_cast<uint32_t>(rng());
        r2 = static_cast<uint32_t>(rng());
      }
      cu->write_vgpr(vbase + 0, lane, r0);
      cu->write_vgpr(vbase + 1, lane, r1);
      cu->write_vgpr(vbase + 2, lane, r2);
      seeded_dst[lane] = r2;
      if (nan_lane) {
        auto nan = [&](uint32_t v) { return is_f16 ? is_f16_nan(v & 0xFFFFu) : is_f32_nan(v); };
        (*nan_lane)[lane] = nan(r0) || nan(r1) || nan(r2);
      }
    }
    wf->set_exec(exec);
    return seeded_dst;
  }

  std::array<uint32_t, WF_SIZE> snapshot_dst() const {
    std::array<uint32_t, WF_SIZE> out{};
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vbase + 2, lane);
    return out;
  }
};

struct FmaCase {
  const char *label;
  uint32_t opcode;
  bool is_f16;
  bool has_literal;
};

const FmaCase kCases[] = {
    {"v_fmac_f32", 59, false, false}, // dst = fma(s0, s1, dst)
    {"v_fmaak_f32", 24, false, true}, // dst = fma(s0, s1, K)
    {"v_fmamk_f32", 23, false, true}, // dst = fma(s0, K, s1)
    {"v_mac_f16", 35, true, false},   // f16 dst-accumulate
    {"v_madak_f16", 37, true, true},  // f16 literal addend
    {"v_madmk_f16", 36, true, true},  // f16 literal multiplier
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

  constexpr uint64_t SEED = 0xF1A'1234'5678'9ABCULL;

  std::array<bool, WF_SIZE> nan_lane{};
  std::array<uint32_t, WF_SIZE> seeded{};

  auto run_mode = [&](bool force_scalar) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t enc = vop2_encode(c.opcode, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256);
    // Inline literal: 0x3FC00000 = 1.5f for f32; for f16 the low 16 bits are the
    // f16 constant 0x3E00 = 1.5h (high bits ignored by the op).
    const uint32_t literal = c.is_f16 ? 0x00003E00u : 0x3FC00000u;
    uint32_t words[4] = {enc, c.has_literal ? literal : 0u, 0u, 0u};
    Instruction *inst = decode_valid(*fx.decoder, words);
    EXPECT_NE(inst, nullptr) << c.label << ": decode failed";
    seeded = fx.seed_inputs(SEED, c.is_f16, exec, &nan_lane);
    EXPECT_TRUE(fx.cu->execute_instruction(inst, *fx.wf).succeeded());
    auto out = fx.snapshot_dst();
    delete inst;
    return out;
  };

  const auto scalar_out = run_mode(/*force_scalar=*/true);
  const auto simd_out = run_mode(/*force_scalar=*/false);

  // F32 compares every active lane, including the selected NaN payload.
  // Legacy F16 forms retain their separate host-arithmetic behavior.
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (exec >> lane) & 1ULL;
    if (active) {
      if (c.is_f16 && nan_lane[lane])
        continue;
      EXPECT_EQ(scalar_out[lane], simd_out[lane])
          << c.label << " lane " << lane << ": SIMD path diverged from scalar body";
    } else {
      // Inactive-lane preservation: each inactive lane must keep its seeded v2
      // value (deterministic, identical runs).
      EXPECT_EQ(simd_out[lane], seeded[lane]) << c.label << ": clobbered inactive lane " << lane;
      EXPECT_EQ(scalar_out[lane], seeded[lane]) << c.label << ": clobbered inactive lane " << lane;
    }
  }
}

TEST(Vop2FmaSimdCorrectness, FullExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/~0ULL);
}

TEST(Vop2FmaSimdCorrectness, PartialExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

template <typename Float, typename Bits, std::size_t N>
void check_host_fma(const std::array<std::array<Bits, 3>, N> &inputs) {
  using V = util::native<Float>;
  using U = util::native<Bits>;
  for (uint32_t round = 0; round < 4; ++round) {
    for (std::size_t base = 0; base < inputs.size(); base += V::size()) {
      const auto operand = [&](std::size_t index) {
        return V([&](auto lane) {
          return std::bit_cast<Float>(inputs[(base + lane) % inputs.size()][index]);
        });
      };
      const V a = operand(0), b = operand(1), c = operand(2);
      struct Result {
        U bits;
        int error;
      };
      const auto run = [&](bool hardware) {
        std::fenv_t saved;
        const int saved_error = errno;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
        const uint32_t saved_mxcsr = _mm_getcsr();
#endif
        EXPECT_EQ(std::feholdexcept(&saved), 0);
        EXPECT_EQ(std::fesetround(FE_UPWARD), 0);
        EXPECT_EQ(std::feraiseexcept(FE_DIVBYZERO | FE_INEXACT), 0);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
        const uint32_t host_mxcsr = _mm_getcsr() | (1u << 6) | (1u << 15);
        _mm_setcsr(host_mxcsr);
#endif
        const int host_flags = std::fetestexcept(FE_ALL_EXCEPT);
        errno = E2BIG;
        V value;
        {
          amdgpu::fp_mode::ScopedEnvironment environment(round);
          value = hardware ? amdgpu::host_fma::fused(a, b, c) : util::stdx::fma(a, b, c);
        }
        const int error = errno;
        const int actual_round = std::fegetround(), flags = std::fetestexcept(FE_ALL_EXCEPT);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
        const uint32_t mxcsr = _mm_getcsr();
#endif
        const int restored = std::fesetenv(&saved);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
        _mm_setcsr(saved_mxcsr);
#endif
        errno = saved_error;
        EXPECT_EQ(restored, 0);
        EXPECT_EQ(actual_round, FE_UPWARD);
        EXPECT_EQ(flags, host_flags);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
        EXPECT_EQ(mxcsr, host_mxcsr);
#endif
        return Result{std::bit_cast<U>(value), error};
      };
      const Result reference = run(false), actual = run(true);
      EXPECT_EQ(actual.error, reference.error) << "round=" << round << " base=" << base;
      for (std::size_t lane = 0; lane < V::size(); ++lane) {
        EXPECT_EQ(actual.bits[lane], reference.bits[lane])
            << "round=" << round << " case=" << (base + lane) % inputs.size();
      }
    }
  }
}

TEST(Vop2FmaSimdCorrectness, HostF32FmaMatchesLibraryRangeAndRounding) {
  // Finite inputs exercise the raw backend. Architectural NaN selection stays
  // in fma_f32_simd and is covered by the instruction comparisons above.
  check_host_fma<float>(std::array<std::array<uint32_t, 3>, 12>{{
      {0x3f800001u, 0x3f7fffffu, 0xbf800000u},
      {0x3f800001u, 0x3f800001u, 0xbf800002u},
      {0x7f7fffffu, 0x40000000u, 0u},
      {0xff7fffffu, 0x40000000u, 0u},
      {0x00800000u, 0x3f7fffffu, 0u},
      {0x80800000u, 0x3f7fffffu, 0x80000000u},
      {0x00800000u, 0x33000000u, 0u},
      {0x80800000u, 0x33000000u, 0x80000000u},
      {1u, 1u, 0x00800000u},
      {0x80000001u, 1u, 0x00800000u},
      {0x80000000u, 0x3f800000u, 0x80000000u},
      {0u, 0x3f800000u, 0x80000000u},
  }});
}

TEST(Vop2FmaSimdCorrectness, HostF64FmaMatchesLibraryRangeAndRounding) {
  check_host_fma<double>(std::array<std::array<uint64_t, 3>, 12>{{
      {0x3ff0000000000001ULL, 0x3fefffffffffffffULL, 0xbff0000000000000ULL},
      {0x3ff0000000000001ULL, 0x3ff0000000000001ULL, 0xbff0000000000002ULL},
      {0x7fefffffffffffffULL, 0x4000000000000000ULL, 0ULL},
      {0xffefffffffffffffULL, 0x4000000000000000ULL, 0ULL},
      {0x0010000000000000ULL, 0x3fefffffffffffffULL, 0ULL},
      {0x8010000000000000ULL, 0x3fefffffffffffffULL, 0x8000000000000000ULL},
      {0x0010000000000000ULL, 0x3c90000000000000ULL, 0ULL},
      {0x8010000000000000ULL, 0x3c90000000000000ULL, 0x8000000000000000ULL},
      {1ULL, 1ULL, 0x0010000000000000ULL},
      {0x8000000000000001ULL, 1ULL, 0x0010000000000000ULL},
      {0x8000000000000000ULL, 0x3ff0000000000000ULL, 0x8000000000000000ULL},
      {0ULL, 0x3ff0000000000000ULL, 0x8000000000000000ULL},
  }});
}

} // namespace
