// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_int_widening_ternary_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the 32x32->64
/// widening multiplies (v_mul_lo_u32, v_mul_hi_u32, v_mul_hi_i32) and three more
/// integer ternary VOP3 ops (v_xad_u32 = (a^b)+c, v_and_or_b32 = (a&b)|c,
/// v_lshl_or_b32 = (a<<(b&31))|c). The mul_hi forms use the
/// fixed_size_simd<uint64_t/int64_t> widening pattern already proven for the
/// 24-bit muls; mul_lo uses the plain `a * b` (uint32 wrap is identical
/// signed/unsigned for the low half). The lshl_or shift count is masked to
/// the low 5 bits to match x86 `shl` semantics. Each (case, rot) runs TWICE in
/// the same process -- once forcing the scalar body, once the SIMD fast path,
/// with identical inputs/EXEC -- and the two 64-lane result arrays are asserted
/// equal with EXPECT_EQ (util::set_force_scalar_for_testing flips the gate
/// in-process). In-process inactive lanes must keep the sentinel.

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
constexpr uint32_t kDstVgpr = 6;
constexpr uint32_t DST_SENTINEL = 0xCDCDCDCDu;

constexpr void vop3_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1, uint32_t src2,
                           uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((op & 0x3FF) << 16) | (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((src2 & 0x1FF) << 18);
}

struct Case {
  const char *name;
  uint32_t opcode;
  bool ternary;
};

const std::array<Case, 17> kCases = {{
    {"v_mul_lo_u32_vop3", 645, false},
    {"v_mul_hi_u32_vop3", 646, false},
    {"v_mul_hi_i32_vop3", 647, false},
    {"v_xad_u32_vop3", 499, true},
    {"v_and_or_b32_vop3", 513, true},
    {"v_lshl_or_b32_vop3", 512, true},
    {"v_bfm_b32_vop3", 659, false},
    {"v_mad_i32_i24_vop3", 450, true},
    {"v_mad_u32_u24_vop3", 451, true},
    {"v_alignbit_b32_vop3", 462, true},
    {"v_alignbyte_b32_vop3", 463, true},
    // 16-bit multiply-add family: mixed-width (i32_i16/u32_u16) and 16-bit-wrap
    // (i16/u16, incl. _legacy twins).
    {"v_mad_i32_i16_vop3", 498, true},
    {"v_mad_u32_u16_vop3", 497, true},
    {"v_mad_i16_vop3", 517, true},
    {"v_mad_u16_vop3", 516, true},
    {"v_mad_legacy_i16_vop3", 492, true},
    {"v_mad_legacy_u16_vop3", 491, true},
}};

const std::array<uint32_t, 14> kVals = {{
    0x00000000u,
    0x00000001u,
    0x00000005u,
    0x0000001Fu,
    0x00000010u,
    0x80000000u,
    0x7FFFFFFFu,
    0xFFFFFFFFu,
    0x12345678u,
    0xDEADBEEFu,
    0xCAFEBABEu,
    0xA5A5A5A5u,
    0x5A5A5A5Au,
    0xF0F0F0F0u,
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3_wide_tern_mem"), l2("vop3_wide_tern_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_wide_tern", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_inputs(uint32_t rot, uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      cu->write_vgpr(vb + 0, lane, kVals[lane % kVals.size()]);
      cu->write_vgpr(vb + 1, lane, kVals[(lane + rot) % kVals.size()]);
      cu->write_vgpr(vb + 2, lane, kVals[(lane + 2 * rot) % kVals.size()]);
      cu->write_vgpr(vb + kDstVgpr, lane, DST_SENTINEL);
    }
    wf->set_exec(exec);
  }

  std::array<uint32_t, WF_SIZE> run(Instruction *inst, uint32_t rot, uint64_t exec) {
    seed_inputs(rot, exec);
    cu->execute_instruction(inst, *wf);
    std::array<uint32_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
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

void check_case(const Case &c, uint64_t exec) {
  ForceScalarGuard gate_guard;

  // Runs one (case, rot) in the requested execute mode (fresh Fixture + decode
  // per run isolates VGPR state).
  auto run_mode = [&](bool force_scalar, uint32_t rot) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_encode(c.opcode, /*vdst=*/kDstVgpr, /*src0=*/256, /*src1=*/257,
                /*src2=*/(c.ternary ? 258u : 0u), words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.name << " decode failed";
    auto out = fx.run(inst, rot, exec);
    delete inst;
    return out;
  };

  for (uint32_t rot = 0; rot < kVals.size(); ++rot) {
    const auto scalar_out = run_mode(/*force_scalar=*/true, rot);
    const auto simd_out = run_mode(/*force_scalar=*/false, rot);

    EXPECT_EQ(scalar_out, simd_out) << c.name << " rot=" << rot << " (exec=0x" << std::hex << exec
                                    << "): SIMD path diverged from scalar body";

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (!active) {
        EXPECT_EQ(simd_out[lane], DST_SENTINEL)
            << c.name << " rot=" << rot << ": clobbered inactive lane " << lane;
        EXPECT_EQ(scalar_out[lane], DST_SENTINEL)
            << c.name << " rot=" << rot << ": clobbered inactive lane " << lane;
      }
    }
  }
}

TEST(Vop3IntWideningTernarySimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/~0ULL);
}

TEST(Vop3IntWideningTernarySimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
