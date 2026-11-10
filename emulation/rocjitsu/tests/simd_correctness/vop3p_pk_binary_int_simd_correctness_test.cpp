// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3p_pk_binary_int_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the VOP3P
/// packed-16 integer binary family wired into SIMD_VOP3P_PK_BINARY_INT on CDNA4.
/// Each (case, rot) runs TWICE in the same process -- once forcing the scalar
/// body, once the SIMD fast path, with identical inputs/EXEC -- and the two
/// 64-lane result arrays are asserted equal with EXPECT_EQ
/// (util::set_force_scalar_for_testing flips the gate in-process).
/// In-process inactive lanes must keep the sentinel. Ops covered:
///   v_pk_add_u16 / v_pk_add_i16 / v_pk_sub_u16 / v_pk_sub_i16
///   v_pk_mul_lo_u16
///   v_pk_lshlrev_b16 / v_pk_lshrrev_b16 / v_pk_ashrrev_i16
///   v_pk_min_u16 / v_pk_max_u16 / v_pk_min_i16 / v_pk_max_i16
///
/// Each 32-bit lane holds two 16-bit values. The CDNA4 sweep uses default
/// half selectors and 14 rotations under full and partial EXEC. The sign-bit
/// cases below exercise non-default half selectors on newer architectures.

#include "compute_unit_test_util.h"
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

// CDNA4 VOP3P encoding (op_sel=0, op_sel_hi=3 = default packing for srcs
// 0/1; for ternary pk_mad the third source's high-half pick op_sel_hi_2
// must be 1 to mirror op_sel_hi=3's behaviour on the binary form).
constexpr void vop3p_encode_default_binary(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1,
                                           uint32_t words[2]) {
  words[0] = (vdst & 0xFFu) | ((op & 0x7Fu) << 16) | (0x1A7u << 23);
  words[1] = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | (0x3u << 27);
}
constexpr void vop3p_encode_default_ternary(uint32_t op, uint32_t vdst, uint32_t src0,
                                            uint32_t src1, uint32_t src2, uint32_t words[2]) {
  // op_sel_hi_2 = bit 14 of word0 (must be 1 for default high-half pick on src2).
  words[0] = (vdst & 0xFFu) | ((1u & 0x1u) << 14) | ((op & 0x7Fu) << 16) | (0x1A7u << 23);
  words[1] = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18) | (0x3u << 27);
}

struct Case {
  const char *name;
  uint32_t opcode;
  bool ternary;
};

const std::array<Case, 14> kCases = {{
    {"v_pk_add_u16_vop3p", 10, false},
    {"v_pk_add_i16_vop3p", 2, false},
    {"v_pk_sub_u16_vop3p", 11, false},
    {"v_pk_sub_i16_vop3p", 3, false},
    {"v_pk_mul_lo_u16_vop3p", 1, false},
    {"v_pk_lshlrev_b16_vop3p", 4, false},
    {"v_pk_lshrrev_b16_vop3p", 5, false},
    {"v_pk_ashrrev_i16_vop3p", 6, false},
    {"v_pk_max_i16_vop3p", 7, false},
    {"v_pk_min_i16_vop3p", 8, false},
    {"v_pk_max_u16_vop3p", 12, false},
    {"v_pk_min_u16_vop3p", 13, false},
    {"v_pk_mad_i16_vop3p", 0, true},
    {"v_pk_mad_u16_vop3p", 9, true},
}};

// Inputs hand-picked to exercise low/high overflow independently, sign
// boundary on both halves, and shift-count-15 saturation.
const std::array<uint32_t, 14> kVals = {{
    0x00000000u,
    0x00010001u,
    0x0000FFFFu, // low all-ones, high zero
    0xFFFF0000u, // high all-ones, low zero
    0xFFFFFFFFu,
    0x80007FFFu, // signed extreme low INT16_MAX, high INT16_MIN
    0x7FFF8000u,
    0x12345678u,
    0xDEADBEEFu,
    0xCAFEBABEu,
    0xA5A5A5A5u,
    0x00050003u, // small shift counts
    0x000F000Eu,
    0x00080001u,
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  explicit Fixture(rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA4)
      : gpu_mem("vop3p_pk_bin_int_mem"), l2("vop3p_pk_bin_int_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    rocjitsu::test::initialize_compute_unit(cu, "cu_vop3p_pk_bin_int", cfg, &gpu_mem, &l2);
    if (!cu)
      return;
    decoder = Decoder::create(arch);
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
    EXPECT_TRUE(cu->execute_instruction(inst, *wf).succeeded());
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
    if (::testing::Test::HasFatalFailure())
      return {};
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[2] = {0u, 0u};
    if (c.ternary)
      vop3p_encode_default_ternary(c.opcode, kDstVgpr, /*src0=*/256, /*src1=*/257, /*src2=*/258,
                                   words);
    else
      vop3p_encode_default_binary(c.opcode, kDstVgpr, /*src0=*/256, /*src1=*/257, words);
    Instruction *inst = decode_valid(*fx.decoder, words);
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

TEST(Vop3pPkBinaryIntSimdCorrectness, AppliesSignBitsAfterSelectingSourceHalves) {
  struct NegCase {
    uint32_t words[2];
    uint32_t expected;
  };
  // LLVM encodings for v6, v0, v1[, v2] on gfx1100, gfx1151, gfx1201 and gfx1250.
  constexpr NegCase cases[] = {
      {{0xcc00c206, 0x3c0a0300}, 0x7ffe8000}, // MAD_I16: negate low A and high B, clamp.
      {{0xcc004406, 0x9c0a0300}, 0x7ffe800f}, // MAD_I16: negate both halves of C.
      {{0xcc0a4a06, 0x32020300}, 0x80068003}, // ADD_U16: swap A's halves, negate low A and high B.
      {{0xcc0a4106, 0x12020300}, 0x80060007}, // ADD_U16: select low A twice and negate only high A.
      {{0xcc006006, 0x9c0a0300}, 0xfffe800d}, // MAD_I16: select high C twice and negate only low C.
  };
  ForceScalarGuard guard;
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4,
                    ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(arch);
    Fixture fx(arch);
    ASSERT_NE(fx.wf, nullptr);
    const uint32_t base = fx.wf->vgpr_alloc().base;
    fx.wf->set_exec(1);
    fx.cu->write_vgpr(base, 0, 0xffff0003);
    fx.cu->write_vgpr(base + 1, 0, 0x00030004);
    fx.cu->write_vgpr(base + 2, 0, 0x00010003);
    for (const NegCase &test_case : cases) {
      std::unique_ptr<Instruction> inst(decode_valid(*fx.decoder, test_case.words));
      ASSERT_NE(inst, nullptr);
      SCOPED_TRACE(inst->mnemonic());
      for (bool force_scalar : {false, true}) {
        SCOPED_TRACE(force_scalar);
        util::set_force_scalar_for_testing(force_scalar);
        fx.cu->write_vgpr(base + kDstVgpr, 0, DST_SENTINEL);
        fx.cu->write_vgpr(base + kDstVgpr, 1, DST_SENTINEL);
        ASSERT_TRUE(fx.cu->execute_instruction(inst.get(), *fx.wf).succeeded());
        EXPECT_EQ(fx.cu->read_vgpr(base + kDstVgpr, 0), test_case.expected);
        EXPECT_EQ(fx.cu->read_vgpr(base + kDstVgpr, 1), DST_SENTINEL);
      }
    }
  }
}

TEST(Vop3pPkBinaryIntSimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/~0ULL);
}

TEST(Vop3pPkBinaryIntSimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
