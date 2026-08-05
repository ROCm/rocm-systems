// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop1_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the Tier-2
/// unary VOP1 instructions wired into SIMD_VOP1_UNARY. Each op runs TWICE in the
/// same process -- once forcing the scalar body, once the SIMD fast path, with
/// identical seed/inputs/EXEC -- and the two 64-lane result arrays are asserted
/// equal with EXPECT_EQ (util::set_force_scalar_for_testing flips the gate
/// in-process). In-process the test also checks inactive-lane preservation under
/// full and partial EXEC masks.
///
/// All wired ops (move/bitwise, exact int<->float casts, and correctly-rounded
/// IEEE floor/ceil/trunc/rndne/fract/rcp/rsq/sqrt) are bit-identical to their
/// scalar bodies for every input — including NaN/Inf/denormal — so inputs are
/// raw random with explicit edge lanes injected (0, ±0, ±Inf, NaN, denormal,
/// INT32 extremes) rather than sanitized to finite normals.

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
#include <cstdint>
#include <memory>
#include <random>
#include <string>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t DST_SENTINEL = 0xCAFEF00Du;

// CDNA4 VOP1: [31:25]=0b0111111, vdst[24:17], op[16:9], src0[8:0]. v<N> as a
// source operand encodes to 256+N; vdst holds the VGPR number directly.
constexpr uint32_t vop1_encode(uint32_t op, uint32_t vdst, uint32_t src0) {
  return (0x3Fu << 25) | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF);
}

// Bit patterns that stress IEEE edge handling. Placed in the first lanes; the
// rest are filled with deterministic random words.
constexpr uint32_t kEdgeBits[] = {
    0x00000000u, // +0.0 / int 0
    0x80000000u, // -0.0 / INT32_MIN
    0x7F800000u, // +Inf
    0xFF800000u, // -Inf
    0x7FC00000u, // qNaN
    0x7F800001u, // sNaN
    0x00000001u, // smallest denormal / int 1
    0x7FFFFFFFu, // largest finite-ish / INT32_MAX
    0x3F800000u, // 1.0f
    0xBF800000u, // -1.0f
    0x4B000000u, // 2^23 (boundary for float<->int exactness)
    0xCB000000u, // -2^23
    0x4F000000u, // 2^31         (cvt_i32 clamp boundary -> INT32_MAX)
    0x4EFFFFFFu, // just below 2^31 (largest exact int32-range float)
    0x4F400000u, // 3*2^30       (in [2^31,2^32): cvt_u32 mid, cvt_i32 clamps)
    0x4F800000u, // 2^32         (cvt_u32 clamp boundary -> UINT32_MAX)
    0xCF000000u, // -2^31        (cvt_i32 -> INT32_MIN, not yet clamped)
    0xCF000001u, // below -2^31  (cvt_i32 clamp -> INT32_MIN)
    // f16 bit patterns in the low 16 bits (high half irrelevant for f16 ops).
    0x00000001u, // f16 smallest denormal
    0x000003FFu, // f16 largest denormal
    0x00007BFFu, // f16 max normal (65504)
    0x00007C00u, // f16 +Inf
    0x0000FC00u, // f16 -Inf
    0x00007E00u, // f16 qNaN
    0x00007C01u, // f16 sNaN
    0x00003C00u, // f16 1.0
    0x0000C000u, // f16 -2.0
    0x00008000u, // f16 -0.0
    // f32 denormals with the highest mantissa bit at varied positions, to
    // exercise frexp's denormal renormalization across the full shift range.
    0x00400000u, // p = 22 (largest denormal exponent)
    0x80200001u, // p = 21, negative
    0x00080040u, // p = 19
    0x00000200u, // p = 9
    0x0000007Fu, // p = 6
};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop1_simd_mem"), l2("vop1_simd_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop1_simd", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_inputs(uint64_t seed, uint64_t exec) {
    std::mt19937_64 rng(seed);
    uint32_t vbase = wf->vgpr_alloc().base;
    constexpr uint32_t n_edge = std::size(kEdgeBits);
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint32_t s0 = lane < n_edge ? kEdgeBits[lane] : static_cast<uint32_t>(rng());
      cu->write_vgpr(vbase + 0, lane, s0);
      cu->write_vgpr(vbase + 2, lane, DST_SENTINEL);
    }
    wf->set_exec(exec);
  }

  std::array<uint32_t, WF_SIZE> snapshot_dst() const {
    std::array<uint32_t, WF_SIZE> out{};
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vbase + 2, lane);
    return out;
  }

  std::array<uint32_t, WF_SIZE> run(Instruction *inst, uint64_t seed, uint64_t exec) {
    seed_inputs(seed, exec);
    cu->execute_instruction(inst, *wf);
    return snapshot_dst();
  }
};

struct Vop1Case {
  const char *label;
  uint32_t opcode;
};

// Tier-2 ops wired into SIMD_VOP1_UNARY (CDNA4 VOP1 opcodes).
const Vop1Case kCases[] = {
    {"v_mov_b32", 1},
    {"v_not_b32", 43},
    {"v_bfrev_b32", 44},
    {"v_cvt_f32_ubyte0", 17},
    {"v_cvt_f32_ubyte1", 18},
    {"v_cvt_f32_ubyte2", 19},
    {"v_cvt_f32_ubyte3", 20},
    {"v_cvt_f32_i32", 5},
    {"v_cvt_f32_u32", 6},
    {"v_floor_f32", 31},
    {"v_ceil_f32", 29},
    {"v_trunc_f32", 28},
    {"v_rndne_f32", 30},
    {"v_fract_f32", 27},
    {"v_rcp_f32", 34},
    {"v_rcp_iflag_f32", 35},
    {"v_rsq_f32", 36},
    {"v_sqrt_f32", 39},
    {"v_exp_f32", 32},
    {"v_log_f32", 33},
    // G3: float -> int with NaN->0 + saturating clamp.
    {"v_cvt_i32_f32", 8},
    {"v_cvt_u32_f32", 7},
    {"v_cvt_flr_i32_f32", 13},
    {"v_cvt_rpi_i32_f32", 12},
    // f16 ops (route through f32 intermediate).
    {"v_floor_f16", 68},
    {"v_ceil_f16", 69},
    {"v_trunc_f16", 70},
    {"v_rndne_f16", 71},
    {"v_fract_f16", 72},
    {"v_rcp_f16", 61},
    {"v_rsq_f16", 63},
    {"v_sqrt_f16", 62},
    {"v_exp_f16", 65},
    {"v_log_f16", 64},
    {"v_cvt_f32_f16", 11},
    {"v_cvt_f16_f32", 10},
    {"v_cvt_f16_i16", 58},
    {"v_cvt_f16_u16", 57},
    {"v_cvt_i16_f16", 60},
    {"v_cvt_u16_f16", 59},
    // Bit-scan (SWAR popcount/clz/ctz functors). 0-input lanes (kEdgeBits[0])
    // exercise the 0xFFFFFFFF sentinel; INT32_MIN / -1 / 1 cover sign cases.
    {"v_ffbh_u32", 45},
    {"v_ffbl_b32", 46},
    {"v_ffbh_i32", 47},
    // frexp f32 split: exponent (raw int32) and mantissa (float). Edge lanes
    // cover ±0 / denormal / Inf / NaN / normal, where frexp's special cases live.
    {"v_frexp_exp_i32_f32", 51},
    {"v_frexp_mant_f32", 52},
    // frexp f16 (f16<->f32 round trip): mantissa / exponent narrowed to f16.
    {"v_frexp_mant_f16", 66},
    {"v_frexp_exp_i16_f16", 67},
};

// Restores the process force-scalar gate on scope exit so flipping it for an
// in-process A/B comparison cannot leak into later tests in the same process.
struct ForceScalarGuard {
  bool orig;
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }
};

void check_case(const Vop1Case &c, uint64_t exec) {
  ForceScalarGuard gate_guard;

  // Runs the op for one seed in the requested execute mode (fresh Fixture +
  // decode per run isolates VGPR state). All wired ops are bit-exact for every
  // input, so the full 64-lane arrays must match between the two modes.
  auto run_mode = [&](bool force_scalar, uint64_t seed) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t enc = vop1_encode(c.opcode, /*vdst=*/2, /*src0=*/256);
    uint32_t words[4] = {enc, 0u, 0u, 0u};
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.label << ": decode failed";
    auto out = fx.run(inst, seed, exec);
    delete inst;
    return out;
  };

  // Sweep several seeds so the random (non-edge) lanes cover more of the
  // input space — in particular the [2^31, 2^32) cvt range and clamp regions.
  for (uint64_t seed = 0; seed < 16; ++seed) {
    const auto scalar_out = run_mode(/*force_scalar=*/true, seed);
    const auto simd_out = run_mode(/*force_scalar=*/false, seed);

    // Core A/B equivalence: SIMD fast path must be bit-identical to the scalar
    // body across all 64 lanes (active and inactive).
    EXPECT_EQ(scalar_out, simd_out)
        << c.label << " (exec=0x" << std::hex << exec << ", seed=" << std::dec << seed
        << "): SIMD path diverged from scalar body";

    // Inactive lanes must keep the destination sentinel in either mode.
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (!active) {
        EXPECT_EQ(simd_out[lane], DST_SENTINEL)
            << c.label << ": clobbered inactive lane " << lane << " at seed " << seed;
        EXPECT_EQ(scalar_out[lane], DST_SENTINEL)
            << c.label << ": clobbered inactive lane " << lane << " at seed " << seed;
      }
    }
  }
}

TEST(Vop1SimdCorrectness, FullExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/~0ULL);
}

TEST(Vop1SimdCorrectness, PartialExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
