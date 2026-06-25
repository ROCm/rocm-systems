// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_binary_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the
/// VOP3-encoded twins of the SIMD VOP2 binary ops on CDNA4. The VOP3 form reads
/// src0/src1 and carries per-source abs/neg and per-instruction omod/clamp
/// modifiers. f32 ops apply the modifiers in-vector (so the fast path fires even
/// when modifiers are set — every abs/neg/omod/clamp combination is swept);
/// integer/bitwise ops apply none. Each (case, modifier combo) runs TWICE in the
/// same process -- once forcing the scalar body, once the SIMD fast path, with
/// identical inputs/EXEC -- and the two 64-lane result arrays are asserted equal
/// with EXPECT_EQ (util::set_force_scalar_for_testing flips the gate in-process).
/// In-process inactive lanes must stay preserved under full and partial EXEC.

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
constexpr uint32_t DST_SENTINEL = 0xCDCDCDCDu;
constexpr uint32_t kDstVgpr = 4;

// VOP3 (encoding[31:26]=0x34): word0 = vdst[7:0] | abs[10:8] | op_sel[14:11] |
// clamp[15] | op[25:16] | enc[31:26]; word1 = src0[8:0] | src1[17:9] | src2[26:18]
// | omod[28:27] | neg[31:29]. VGPR operands are encoded as 256 + reg.
void vop3_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1, uint32_t abs,
                 uint32_t neg, uint32_t omod, uint32_t clamp, uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((abs & 0x7) << 8) | ((clamp & 0x1) << 15) | ((op & 0x3FF) << 16) |
             (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((omod & 0x3) << 27) | ((neg & 0x7) << 29);
}

// Mixed f32 inputs: normals, ±0, ±Inf, NaN, denormal, in-[0,1] (for clamp), and
// values whose products/sums exercise omod overflow to Inf.
const std::array<uint32_t, 16> kF32A = {{
    0x3F000000u,
    0x40490FDBu,
    0xC0490FDBu,
    0x00000000u,
    0x80000000u,
    0x7F800000u,
    0xFF800000u,
    0x7FC00000u,
    0x00000001u,
    0x3DCCCCCDu,
    0x42F60000u,
    0xC2F60000u,
    0x3F800000u,
    0xBF800000u,
    0x4B000000u,
    0x7F7FFFFFu,
}};
const std::array<uint32_t, 16> kF32B = {{
    0x40000000u,
    0x3F800000u,
    0x40490FDBu,
    0x3F000000u,
    0x7F800000u,
    0x40000000u,
    0x3F800000u,
    0x40490FDBu,
    0x40A00000u,
    0x3F000000u,
    0x40000000u,
    0x40000000u,
    0xBF800000u,
    0x3F800000u,
    0x4B000000u,
    0x40000000u,
}};
const std::array<uint32_t, 16> kU32A = {{
    0x00000000u,
    0xFFFFFFFFu,
    0x12345678u,
    0x80000000u,
    0x0000FFFFu,
    0xDEADBEEFu,
    0x00000001u,
    0xCAFEBABEu,
    0xA5A5A5A5u,
    0x0F0F0F0Fu,
    0x7FFFFFFFu,
    0x00FF00FFu,
    0x33333333u,
    0xCCCCCCCCu,
    0x10000000u,
    0xFEDCBA98u,
}};
const std::array<uint32_t, 16> kU32B = {{
    0xFFFFFFFFu,
    0x00000001u,
    0x0000000Fu,
    0x00000003u,
    0xFFFF0000u,
    0x0000001Fu,
    0x80000000u,
    0x00000007u,
    0x5A5A5A5Au,
    0xF0F0F0F0u,
    0x00000002u,
    0xFF00FF00u,
    0xCCCCCCCCu,
    0x33333333u,
    0x00000004u,
    0x01234567u,
}};

// f16 corner set (normals, ±0, ±Inf, NaN, denormals, ±max-normal) — packed two
// per 32-bit lane (see f16_src0/f16_src1) so both halves see distinct values.
const std::array<uint16_t, 16> kF16 = {{
    0x3C00u,
    0xBC00u,
    0x4000u,
    0xC000u,
    0x0000u,
    0x8000u,
    0x7C00u,
    0xFC00u,
    0x7E00u,
    0x4900u,
    0xC900u,
    0x3800u,
    0x0400u,
    0x8400u,
    0x7BFFu,
    0xFBFFu,
}};
bool is_f16_nan(uint16_t b) { return ((b >> 10) & 0x1Fu) == 0x1Fu && (b & 0x3FFu) != 0u; }
bool is_f16_zero(uint16_t b) { return (b & 0x7FFFu) == 0u; }
uint32_t f16_src0(uint32_t lane) {
  return kF16[lane % kF16.size()] | (static_cast<uint32_t>(kF16[(lane + 3) % kF16.size()]) << 16);
}
uint32_t f16_src1(uint32_t lane) {
  return kF16[(lane + 5) % kF16.size()] |
         (static_cast<uint32_t>(kF16[(lane + 9) % kF16.size()]) << 16);
}

enum class Kind { F32, INT, F16 };

struct Case {
  const char *name;
  uint32_t opcode;
  Kind kind;
  bool minmax = false;
};

const std::array<Case, 16> kCases = {{
    {"v_add_f32", 257, Kind::F32},
    {"v_sub_f32", 258, Kind::F32},
    {"v_mul_f32", 261, Kind::F32},
    {"v_min_f32", 266, Kind::F32},
    {"v_max_f32", 267, Kind::F32},
    {"v_and_b32", 275, Kind::INT},
    {"v_or_b32", 276, Kind::INT},
    {"v_xor_b32", 277, Kind::INT},
    {"v_add_u32", 308, Kind::INT},
    {"v_mul_legacy_f32", 673, Kind::F32},
    // f16 VOP3 binaries: the scalar body applies abs/neg/omod/clamp around the
    // f16<->f32 round trip; the packed SIMD path bails to scalar when any
    // modifier is set, otherwise runs the unmodified packed op.
    {"v_add_f16", 287, Kind::F16},
    {"v_sub_f16", 288, Kind::F16},
    {"v_subrev_f16", 289, Kind::F16},
    {"v_mul_f16", 290, Kind::F16},
    {"v_max_f16", 301, Kind::F16, true},
    {"v_min_f16", 302, Kind::F16, true},
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3_bin_simd_mem"), l2("vop3_bin_simd_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_bin_simd", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_inputs(Kind k, uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      if (k == Kind::F32) {
        cu->write_vgpr(vb + 0, lane, kF32A[lane % kF32A.size()]);
        cu->write_vgpr(vb + 1, lane, kF32B[lane % kF32B.size()]);
      } else if (k == Kind::F16) {
        cu->write_vgpr(vb + 0, lane, f16_src0(lane));
        cu->write_vgpr(vb + 1, lane, f16_src1(lane));
      } else {
        cu->write_vgpr(vb + 0, lane, kU32A[lane % kU32A.size()]);
        cu->write_vgpr(vb + 1, lane, kU32B[lane % kU32B.size()]);
      }
      cu->write_vgpr(vb + kDstVgpr, lane, DST_SENTINEL);
    }
    wf->set_exec(exec);
  }

  std::array<uint32_t, WF_SIZE> run(Instruction *inst, Kind k, uint64_t exec) {
    seed_inputs(k, exec);
    cu->execute_instruction(inst, *wf);
    uint32_t vb = wf->vgpr_alloc().base;
    std::array<uint32_t, WF_SIZE> out{};
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vb + kDstVgpr, lane);
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

void check(const Case &c, uint32_t abs, uint32_t neg, uint32_t omod, uint32_t clamp,
           uint64_t exec) {
  ForceScalarGuard gate_guard;

  // Run the same (case, modifier combo) in both execute modes with identical
  // deterministic inputs (fresh Fixture + decode per run isolates VGPR state).
  auto run_mode = [&](bool force_scalar) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_encode(c.opcode, /*vdst=*/kDstVgpr, /*src0=*/256, /*src1=*/257, abs, neg, omod, clamp,
                words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.name << " decode failed";
    auto out = fx.run(inst, c.kind, exec);
    delete inst;
    return out;
  };

  const auto scalar_out = run_mode(/*force_scalar=*/true);
  const auto simd_out = run_mode(/*force_scalar=*/false);

  // Core A/B equivalence: SIMD fast path must be bit-identical to the scalar
  // body across all 64 lanes (active and inactive).
  EXPECT_EQ(scalar_out, simd_out) << c.name << " abs=" << abs << " neg=" << neg << " omod=" << omod
                                  << " clamp=" << clamp << " (exec=0x" << std::hex << exec
                                  << "): SIMD path diverged from scalar body";

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (exec >> lane) & 1ULL;
    if (!active) {
      EXPECT_EQ(simd_out[lane], DST_SENTINEL)
          << c.name << " abs=" << abs << " neg=" << neg << " omod=" << omod << " clamp=" << clamp
          << ": clobbered inactive lane " << lane;
      EXPECT_EQ(scalar_out[lane], DST_SENTINEL)
          << c.name << " abs=" << abs << " neg=" << neg << " omod=" << omod << " clamp=" << clamp
          << ": clobbered inactive lane " << lane;
    }
  }
}

// f32 ops: sweep every abs/neg/omod/clamp combination.
void check_f32_all_mods(const Case &c, uint64_t exec) {
  for (uint32_t abs = 0; abs < 4; ++abs)   // bits 0,1 = src0,src1
    for (uint32_t neg = 0; neg < 4; ++neg) // bits 0,1 = src0,src1
      for (uint32_t omod = 0; omod < 4; ++omod)
        for (uint32_t clamp = 0; clamp < 2; ++clamp)
          check(c, abs, neg, omod, clamp, exec);
}

TEST(Vop3BinarySimdCorrectness, F32_AllModifiers_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    if (c.kind == Kind::F32)
      check_f32_all_mods(c, /*exec=*/~0ULL);
}

TEST(Vop3BinarySimdCorrectness, F32_AllModifiers_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    if (c.kind == Kind::F32)
      check_f32_all_mods(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

TEST(Vop3BinarySimdCorrectness, Int_FullAndPartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases) {
    if (c.kind != Kind::INT)
      continue;
    check(c, 0, 0, 0, 0, /*exec=*/~0ULL);
    check(c, 0, 0, 0, 0, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
  }
}

// f16 ops: sweep every modifier combination. The packed SIMD path applies no
// modifiers and bails to the scalar body whenever any is set; the unmodified
// case runs the packed op. Per-lane compare with the usual input-derived skips
// (NaN payload may diverge; ±0 min/max ties are an accepted carve-out). Without
// the modifier bail, the abs/neg/omod/clamp cases would return the unmodified
// result and diverge from the scalar body — this is the regression guard.
void check_f16(const Case &c, uint32_t abs, uint32_t neg, uint32_t omod, uint32_t clamp,
               uint64_t exec) {
  ForceScalarGuard gate_guard;
  auto run_mode = [&](bool force_scalar) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_encode(c.opcode, /*vdst=*/kDstVgpr, /*src0=*/256, /*src1=*/257, abs, neg, omod, clamp,
                words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.name << " decode failed";
    auto out = fx.run(inst, c.kind, exec);
    delete inst;
    return out;
  };
  const auto scalar_out = run_mode(/*force_scalar=*/true);
  const auto simd_out = run_mode(/*force_scalar=*/false);

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (exec >> lane) & 1ULL;
    if (!active) {
      EXPECT_EQ(simd_out[lane], DST_SENTINEL) << c.name << ": clobbered inactive lane " << lane;
      EXPECT_EQ(scalar_out[lane], DST_SENTINEL) << c.name << ": clobbered inactive lane " << lane;
      continue;
    }
    const uint32_t s0 = f16_src0(lane), s1 = f16_src1(lane);
    const uint16_t a_lo = s0 & 0xFFFFu, a_hi = static_cast<uint16_t>(s0 >> 16);
    const uint16_t b_lo = s1 & 0xFFFFu, b_hi = static_cast<uint16_t>(s1 >> 16);
    bool skip = is_f16_nan(a_lo) || is_f16_nan(a_hi) || is_f16_nan(b_lo) || is_f16_nan(b_hi);
    if (!skip && c.minmax)
      skip = (is_f16_zero(a_lo) && is_f16_zero(b_lo)) || (is_f16_zero(a_hi) && is_f16_zero(b_hi));
    if (skip)
      continue;
    EXPECT_EQ(scalar_out[lane], simd_out[lane])
        << c.name << " abs=" << abs << " neg=" << neg << " omod=" << omod << " clamp=" << clamp
        << " (exec=0x" << std::hex << exec << "): SIMD path diverged from scalar body, lane "
        << std::dec << lane;
  }
}

void check_f16_all_mods(const Case &c, uint64_t exec) {
  for (uint32_t abs = 0; abs < 4; ++abs)
    for (uint32_t neg = 0; neg < 4; ++neg)
      for (uint32_t omod = 0; omod < 4; ++omod)
        for (uint32_t clamp = 0; clamp < 2; ++clamp)
          check_f16(c, abs, neg, omod, clamp, exec);
}

TEST(Vop3BinarySimdCorrectness, F16_AllModifiers_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    if (c.kind == Kind::F16)
      check_f16_all_mods(c, /*exec=*/~0ULL);
}

TEST(Vop3BinarySimdCorrectness, F16_AllModifiers_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    if (c.kind == Kind::F16)
      check_f16_all_mods(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
