// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop2_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the Tier-1
/// VOP2 binary instructions wired into SIMD_VOP2_BINARY. Each op runs TWICE in
/// the same process -- once forcing the scalar body, once the SIMD fast path,
/// with identical seed/inputs/EXEC -- and the two 64-lane result arrays are
/// asserted equal with EXPECT_EQ (util::set_force_scalar_for_testing flips the
/// gate in-process; the prior two-process file-dump diff is no longer needed
/// here). In-process the test also checks inactive-lane preservation under full
/// and partial EXEC masks.

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

// CDNA4 VOP2: opcode[30:25], vdst[24:17], vsrc1[16:9], src0[8:0]. Bit 31 = 0.
constexpr uint32_t vop2_encode(uint32_t opcode, uint32_t vdst, uint32_t vsrc1, uint32_t src0) {
  return ((opcode & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF);
}

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop2_simd_mem"), l2("vop2_simd_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop2_simd", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  // Map a raw 32-bit value to a finite-normal IEEE-754 binary32 (no NaN/Inf/
  // denormal), keeping fp arithmetic single-rounded and host-SIMD-identical.
  static uint32_t finite_normal(uint32_t raw) {
    uint32_t mantissa = raw & 0x007FFFFFu;
    uint32_t exp = 0x40u | ((raw >> 23) & 0x7Eu);
    uint32_t sign = raw & 0x80000000u;
    return sign | (exp << 23) | mantissa;
  }

  void seed_inputs(uint64_t seed, bool is_float, uint64_t exec) {
    std::mt19937_64 rng(seed);
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint32_t r0 = static_cast<uint32_t>(rng());
      uint32_t r1 = static_cast<uint32_t>(rng());
      if (is_float) {
        r0 = finite_normal(r0);
        r1 = finite_normal(r1);
      }
      cu->write_vgpr(vbase + 0, lane, r0);
      cu->write_vgpr(vbase + 1, lane, r1);
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

  // Runs the instruction in the process's current execute mode (SIMD or
  // scalar, selected by util::force_scalar()). The caller flips the gate
  // between two calls with identical seed/inputs to compare the paths.
  std::array<uint32_t, WF_SIZE> run(Instruction *inst, uint64_t seed, bool is_float,
                                    uint64_t exec) {
    seed_inputs(seed, is_float, exec);
    cu->execute_instruction(inst, *wf);
    return snapshot_dst();
  }
};

struct Vop2Case {
  const char *label;
  uint32_t opcode;
  bool is_float;
};

// Tier-1 ops wired into SIMD_VOP2_BINARY (CDNA4 VOP2 opcodes).
const Vop2Case kCases[] = {
    {"v_add_f32", 1, true},
    {"v_sub_f32", 2, true},
    {"v_subrev_f32", 3, true},
    {"v_mul_f32", 5, true},
    {"v_add_u32", 52, false},
    {"v_sub_u32", 53, false},
    {"v_subrev_u32", 54, false},
    {"v_and_b32", 19, false},
    {"v_or_b32", 20, false},
    {"v_xor_b32", 21, false},
    {"v_xnor_b32", 61, false},
    {"v_lshlrev_b32", 18, false},
    {"v_lshrrev_b32", 16, false},
    {"v_mul_u32_u24", 8, false},
    {"v_mul_i32_i24", 6, false},
    {"v_mul_hi_i32_i24", 7, false},
    {"v_mul_hi_u32_u24", 9, false},
    {"v_max_u32", 15, false},
    {"v_min_u32", 14, false},
    {"v_ashrrev_i32", 17, false},
    {"v_max_i32", 13, false},
    {"v_min_i32", 12, false},
    // 16-bit integer (low 16 bits, zero-extended result).
    {"v_add_u16", 38, false},
    {"v_sub_u16", 39, false},
    {"v_subrev_u16", 40, false},
    {"v_mul_lo_u16", 41, false},
    {"v_lshlrev_b16", 42, false},
    {"v_lshrrev_b16", 43, false},
    {"v_ashrrev_i16", 44, false},
    {"v_max_u16", 47, false},
    {"v_max_i16", 48, false},
    {"v_min_u16", 49, false},
    {"v_min_i16", 50, false},
    // f16 binary (low 16 bits f16, zero-extended result; raw inputs cover
    // f16 NaN/Inf/denormal in the low half).
    {"v_add_f16", 31, false},
    {"v_sub_f16", 32, false},
    {"v_subrev_f16", 33, false},
    {"v_mul_f16", 34, false},
    // f16 ldexp: src0 = f16, vsrc1 = signed-16 exponent. ldexp NaN handling is
    // bit-exact (no operand ambiguity), so raw inputs need no NaN skip.
    {"v_ldexp_f16", 51, false},
    // v_max_f16 / v_min_f16 deliberately omitted: std::fmax/fmin reference is
    // not SIMD-bit-reproducible on signed zero / NaN (see simd_codegen.py).
};

// Restores the process force-scalar gate on scope exit so flipping it for an
// in-process A/B comparison cannot leak into later tests in the same process.
struct ForceScalarGuard {
  bool orig;
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }
};

void check_case(const Vop2Case &c, uint64_t exec) {
  ForceScalarGuard gate_guard;

  constexpr uint64_t SEED = 0xC0FFEE'1234'5678ULL;

  // Run the same instruction with identical seed/inputs/EXEC in both execute
  // modes. A fresh Fixture per run isolates VGPR state; finite_normal inputs
  // (f32) plus the bit-exact f16/ldexp ops and the deliberate omission of
  // v_max_f16/v_min_f16 mean every active lane is bit-reproducible, so no
  // per-lane NaN skip is needed -- the full 64-lane arrays must match.
  auto run_mode = [&](bool force_scalar) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t enc = vop2_encode(c.opcode, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256);
    uint32_t words[4] = {enc, 0u, 0u, 0u};
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.label << ": decode failed";
    auto out = fx.run(inst, SEED, c.is_float, exec);
    delete inst;
    return out;
  };

  const auto scalar_out = run_mode(/*force_scalar=*/true);
  const auto simd_out = run_mode(/*force_scalar=*/false);

  // Core A/B equivalence: the SIMD fast path must be bit-identical to the
  // scalar generated body across all 64 lanes (active and inactive).
  EXPECT_EQ(scalar_out, simd_out) << c.label << " (exec=0x" << std::hex << exec << "): SIMD path "
                                  << "diverged from scalar body";

  // In-process invariant that holds in either mode: inactive lanes must keep
  // the destination sentinel (no out-of-bounds writes / masked-store leaks).
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (exec >> lane) & 1ULL;
    if (!active) {
      EXPECT_EQ(simd_out[lane], DST_SENTINEL) << c.label << ": clobbered inactive lane " << lane
                                              << std::hex << " value=0x" << simd_out[lane];
      EXPECT_EQ(scalar_out[lane], DST_SENTINEL) << c.label << ": clobbered inactive lane " << lane
                                                << std::hex << " value=0x" << scalar_out[lane];
    }
  }
}

TEST(Vop2SimdCorrectness, FullExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/~0ULL);
}

TEST(Vop2SimdCorrectness, PartialExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  // Alternating + sparse pattern: exercises masked_store blend and inactive-lane
  // preservation across SIMD chunk boundaries.
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
