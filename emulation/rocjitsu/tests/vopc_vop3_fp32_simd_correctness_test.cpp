// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vopc_vop3_fp32_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs forced-scalar body) for the
/// VOP3 form of the f32 VOPC relational compares on CDNA4. The VOP3 form
/// reads src0/src1 with per-source abs/neg modifiers (std::fabs then unary
/// minus, applied to the f32 value before comparing) and writes the per-lane
/// compare result into an arbitrary SGPR-pair dst via inst.vdst.read/write_scalar64
/// instead of the fixed VCC. We point the SGPR-pair at VCC (106) so the same
/// harness checks the result via wf.vcc(). All four abs/neg combinations are
/// swept (4 × 16 rels × 16 rotations) under full + partial EXEC, and inactive
/// SGPR-pair bits must be preserved.

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

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t kVccSdst = 106;

// VOP3 ([31:26]=0x34): word0 = vdst[7:0] | abs[10:8] | clamp[15] | op[25:16] |
// enc[31:26]; word1 = src0[8:0] | src1[17:9] | omod[28:27] | neg[31:29]. The
// compare result is a single bit, so omod/clamp are irrelevant and stay zero.
constexpr void vop3_cmp_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1,
                               uint32_t abs, uint32_t neg, uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((abs & 0x7) << 8) | ((op & 0x3FF) << 16) | (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((neg & 0x7) << 29);
}

struct Case {
  const char *name;
  uint32_t opcode;
};

// 16 cdna4-decodable f32 VOP3 cmp relations (the 17th, 'v_cmp_t_f32_vop3',
// has no cdna4 decoder entry — RDNA-only — so it's omitted here even though
// the SIMD probe is still injected for cross-ISA shared correctness).
const std::array<Case, 16> kCases = {{
    {"v_cmp_f_f32_vop3", 64},
    {"v_cmp_lt_f32_vop3", 65},
    {"v_cmp_eq_f32_vop3", 66},
    {"v_cmp_le_f32_vop3", 67},
    {"v_cmp_gt_f32_vop3", 68},
    {"v_cmp_lg_f32_vop3", 69},
    {"v_cmp_ge_f32_vop3", 70},
    {"v_cmp_o_f32_vop3", 71},
    {"v_cmp_u_f32_vop3", 72},
    {"v_cmp_nge_f32_vop3", 73},
    {"v_cmp_nlg_f32_vop3", 74},
    {"v_cmp_ngt_f32_vop3", 75},
    {"v_cmp_nle_f32_vop3", 76},
    {"v_cmp_neq_f32_vop3", 77},
    {"v_cmp_nlt_f32_vop3", 78},
    {"v_cmp_tru_f32_vop3", 79},
}};

// Mixed f32 inputs covering every IEEE class plus equality / ordering /
// signed-zero edge pairs. f32 compares produce a single bit, so NaN payloads
// don't propagate — pairing a value with itself, with its negative, and with
// neighbouring normals fires every relation. The abs/neg modifiers act on the
// sign bit, so include both polarities of every value.
const std::array<uint32_t, 14> kF32 = {{
    0x00000000u, // +0
    0x80000000u, // -0
    0x3F800000u, // +1
    0xBF800000u, // -1
    0x40490FDBu, // +pi
    0xC0490FDBu, // -pi
    0x7F800000u, // +Inf
    0xFF800000u, // -Inf
    0x7FC00000u, // +qNaN
    0xFFA00000u, // -sNaN
    0x00000001u, // +denormal
    0x80000001u, // -denormal
    0x7F7FFFFFu, // +max
    0xFF7FFFFFu, // -max
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vopc_vop3_fp32_mem"), l2("vopc_vop3_fp32_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vopc_vop3_fp32", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_inputs(uint32_t rot, uint64_t exec, uint64_t vcc_in) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      cu->write_vgpr(vb + 0, lane, kF32[lane % kF32.size()]);
      cu->write_vgpr(vb + 1, lane, kF32[(lane + rot) % kF32.size()]);
    }
    wf->set_exec(exec);
    wf->set_vcc(vcc_in);
  }

  uint64_t run(Instruction *inst, bool force_scalar, uint32_t rot, uint64_t exec, uint64_t vcc_in) {
    amdgpu::simd_force_scalar() = force_scalar;
    seed_inputs(rot, exec, vcc_in);
    cu->execute_instruction(inst, *wf);
    amdgpu::simd_force_scalar() = false;
    return wf->vcc();
  }
};

void check_case(const Case &c, uint64_t exec) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  const uint64_t kVcc[] = {0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0xA5A5A5A5A5A5A5A5ULL};
  for (uint32_t abs = 0; abs < 4; ++abs) { // bits 0,1 = src0,src1 abs
    for (uint32_t neg = 0; neg < 4; ++neg) {
      uint32_t words[4] = {0u, 0u, 0u, 0u};
      vop3_cmp_encode(c.opcode, /*vdst=*/kVccSdst, /*src0=*/256, /*src1=*/257, abs, neg, words);
      Instruction *inst = fx.decoder->decode(words);
      ASSERT_NE(inst, nullptr) << c.name << " decode failed (abs=" << abs << " neg=" << neg << ")";
      for (uint32_t rot = 0; rot < kF32.size(); ++rot) {
        for (uint64_t vcc_in : kVcc) {
          uint64_t scalar = fx.run(inst, /*force_scalar=*/true, rot, exec, vcc_in);
          uint64_t simd = fx.run(inst, /*force_scalar=*/false, rot, exec, vcc_in);
          EXPECT_EQ(scalar, simd) << c.name << " abs=" << abs << " neg=" << neg << " rot=" << rot
                                  << " vcc_in=0x" << std::hex << vcc_in
                                  << ": dst divergence scalar=0x" << scalar << " simd=0x" << simd;
          const uint64_t inactive = ~exec;
          EXPECT_EQ(simd & inactive, vcc_in & inactive)
              << c.name << " abs=" << abs << " neg=" << neg << " rot=" << rot
              << ": SIMD altered inactive-lane dst bit";
        }
      }
      delete inst;
    }
  }
}

TEST(VopcVop3Fp32SimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/~0ULL);
}

TEST(VopcVop3Fp32SimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
