// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_unary_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the
/// VOP3-encoded twins of the SIMD VOP1 unary ops on CDNA4. The f32 forms carry
/// per-source abs/neg and result omod/clamp modifiers, applied in-vector — every
/// abs/neg/omod/clamp combination is swept.
/// The integer/cvt forms apply no modifiers and reuse the VOP1 unary path. Each
/// (case, modifier combo) runs TWICE in the same process -- once forcing the
/// scalar body, once the SIMD fast path, with identical inputs/EXEC -- and the
/// two 64-lane result arrays are asserted equal with EXPECT_EQ
/// (util::set_force_scalar_for_testing flips the gate in-process). In-process
/// inactive lanes must stay preserved under full/partial EXEC.

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

// VOP3 ([31:26]=0x34): word0 = vdst[7:0] | abs[10:8] | clamp[15] | op[25:16];
// word1 = src0[8:0] | omod[28:27] | neg[31:29]. Unary: src1 unused.
void vop3_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t abs, uint32_t neg,
                 uint32_t omod, uint32_t clamp, uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((abs & 0x7) << 8) | ((clamp & 0x1) << 15) | ((op & 0x3FF) << 16) |
             (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((omod & 0x3) << 27) | ((neg & 0x7) << 29);
}

// f32 source patterns (for the fp/mov ops): normals, ±0, ±Inf, NaN, denormal,
// in-[0,1], and large values that omod overflows to Inf.
const std::array<uint32_t, 16> kF32 = {{
    0x3F000000u,
    0x40490FDBu,
    0xC0490FDBu,
    0x00000000u,
    0x80000000u,
    0x7F800000u,
    0xFF800000u,
    0x7FC00000u,
    0x00000001u,
    0x80000001u,
    0x3DCCCCCDu,
    0x42F60000u,
    0xBF800000u,
    0x3F800000u,
    0x4B000000u,
    0x7F7FFFFFu,
}};
// Mixed int/float bit patterns for the cvt/not/bfrev plain ops.
const std::array<uint32_t, 16> kBits = {{
    0x00000000u,
    0xFFFFFFFFu,
    0x12345678u,
    0x80000000u,
    0x0000FFFFu,
    0xDEADBEEFu,
    0x00000001u,
    0xCAFEBABEu,
    0x3F800000u,
    0xC2F60000u,
    0x7FFFFFFFu,
    0x00FF00FFu,
    0x40490FDBu,
    0xCCCCCCCCu,
    0x000000FFu,
    0xFEDCBA98u,
}};

enum class Kind { FP, PLAIN };

struct Case {
  const char *name;
  uint32_t opcode;
  Kind kind;
};

const std::array<Case, 15> kCases = {{
    {"v_mov_b32", 321, Kind::FP},
    {"v_floor_f32", 351, Kind::FP},
    {"v_ceil_f32", 349, Kind::FP},
    {"v_trunc_f32", 348, Kind::FP},
    {"v_rcp_f32", 354, Kind::FP},
    {"v_cvt_f32_i32", 325, Kind::PLAIN},
    {"v_cvt_f32_u32", 326, Kind::PLAIN},
    {"v_cvt_i32_f32", 328, Kind::PLAIN},
    {"v_not_b32", 363, Kind::PLAIN},
    {"v_bfrev_b32", 364, Kind::PLAIN},
    // Bit-scan VOP3 twins (auto-routed through the VOP1 unary glue; modifier-free
    // integer bodies) + the VOP3-only v_bcnt_u32_b32 (UNARY_INT_EXTRA popcount).
    {"v_ffbh_u32", 365, Kind::PLAIN},
    {"v_ffbl_b32", 366, Kind::PLAIN},
    {"v_ffbh_i32", 367, Kind::PLAIN},
    {"v_bcnt_u32_b32", 651, Kind::PLAIN},
    // frexp mantissa VOP3: routed through the f32 unary FP glue (abs/neg on the
    // source, omod/clamp on the mantissa); every modifier combination is swept.
    {"v_frexp_mant_f32", 372, Kind::FP},
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3_un_simd_mem"), l2("vop3_un_simd_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_un_simd", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_inputs(Kind k, uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const auto &vals = (k == Kind::FP) ? kF32 : kBits;
      cu->write_vgpr(vb + 0, lane, vals[lane % vals.size()]);
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

  auto run_mode = [&](bool force_scalar) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_encode(c.opcode, /*vdst=*/kDstVgpr, /*src0=*/256, abs, neg, omod, clamp, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.name << " decode failed";
    auto out = fx.run(inst, c.kind, exec);
    delete inst;
    return out;
  };

  const auto scalar_out = run_mode(/*force_scalar=*/true);
  const auto simd_out = run_mode(/*force_scalar=*/false);

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

void check_fp_all_mods(const Case &c, uint64_t exec) {
  for (uint32_t abs = 0; abs < 2; ++abs)   // bit 0 = src0
    for (uint32_t neg = 0; neg < 2; ++neg) // bit 0 = src0
      for (uint32_t omod = 0; omod < 4; ++omod)
        for (uint32_t clamp = 0; clamp < 2; ++clamp)
          check(c, abs, neg, omod, clamp, exec);
}

TEST(Vop3UnarySimdCorrectness, Fp_AllModifiers_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    if (c.kind == Kind::FP)
      check_fp_all_mods(c, /*exec=*/~0ULL);
}

TEST(Vop3UnarySimdCorrectness, Fp_AllModifiers_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    if (c.kind == Kind::FP)
      check_fp_all_mods(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

TEST(Vop3UnarySimdCorrectness, Plain_FullAndPartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases) {
    if (c.kind != Kind::PLAIN)
      continue;
    check(c, 0, 0, 0, 0, /*exec=*/~0ULL);
    check(c, 0, 0, 0, 0, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
  }
}

// Absolute behavior regression for the v_mov_b32 VOP3 codegen fix: with no
// modifiers it must MOVE the bits (not truncate float->int). Before the fix the
// generated body returned the f32 value to a uint32 write_lane without a
// bit_cast, so 0.5 (0x3F000000) wrote 0; neg must flip the sign bit. The
// expected bit patterns are mode-independent, so this pins the result in
// whichever execute mode (RJ_FORCE_SCALAR) the process runs.
TEST(Vop3UnarySimdCorrectness, MovB32_PreservesBits) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  struct Sub {
    uint32_t abs, neg, omod, clamp, in, expect;
  };
  // v_mov_b32 is an integer bit-move (D.u = S0.u): its generated scalar body is
  // a raw copy that ignores the VOP3 float modifiers (abs/neg are float input
  // modifiers, omod/clamp float output modifiers — none apply to a .u move). So
  // every modifier combination must preserve the source bits exactly; the SIMD
  // fast path (plain VOP1 unary copy) must match.
  const std::array<Sub, 4> subs = {{
      {0, 0, 0, 0, 0x3F000000u, 0x3F000000u}, // 0.5 -> 0.5 (bits preserved)
      {0, 0, 0, 0, 0x12345678u, 0x12345678u}, // arbitrary bits preserved
      {0, 1, 0, 0, 0x3F000000u, 0x3F000000u}, // neg modifier ignored -> bits preserved
      {1, 0, 0, 0, 0xBF000000u, 0xBF000000u}, // abs modifier ignored -> bits preserved
  }};
  for (const auto &s : subs) {
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_encode(/*op=v_mov_b32=*/321, /*vdst=*/kDstVgpr, /*src0=*/256, s.abs, s.neg, s.omod,
                s.clamp, words);
    Instruction *inst = fx.decoder->decode(words);
    ASSERT_NE(inst, nullptr);
    uint32_t vb = fx.wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      fx.cu->write_vgpr(vb + 0, lane, s.in);
    fx.wf->set_exec(~0ULL);
    fx.cu->execute_instruction(inst, *fx.wf);
    EXPECT_EQ(fx.cu->read_vgpr(vb + kDstVgpr, 0), s.expect)
        << "v_mov_b32 abs=" << s.abs << " neg=" << s.neg << " in=0x" << std::hex << s.in;
    delete inst;
  }
}

} // namespace
