// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_ternary_int_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs forced-scalar body) for the
/// integer 3-source VOP3 ops on CDNA4. Plain element-wise functions of
/// (src0, src1, src2) with no modifiers — routed through
/// try_execute_ternary_vop3_simd<uint32_t>. CDNA4 decodes 5 of the 6 enabled
/// ops (v_xor3_b32 is RDNA-only).

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
constexpr uint32_t kDstVgpr = 6;
constexpr uint32_t DST_SENTINEL = 0xCDCDCDCDu;

// VOP3 ternary ([31:26]=0x34): word0 = vdst[7:0] | op[25:16] | enc[31:26];
// word1 = src0[8:0] | src1[17:9] | src2[26:18]. abs/neg/omod/clamp stay 0.
constexpr void vop3_tern_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1,
                                uint32_t src2, uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((op & 0x3FF) << 16) | (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((src2 & 0x1FF) << 18);
}

struct Case {
  const char *name;
  uint32_t opcode;
};

// CDNA4-decodable VOP3 integer 3-source ops (the original Slice T set plus
// the Slice AE int min3/max3/med3 family added 2026-05-29). v_xor3_b32 is
// RDNA-only so omitted here even though the SIMD probe ships for cross-ISA
// correctness.
const std::array<Case, 19> kCases = {{
    {"v_add3_u32_vop3", 511},
    {"v_or3_b32_vop3", 514},
    {"v_lshl_add_u32_vop3", 509},
    {"v_add_lshl_u32_vop3", 510},
    {"v_bfi_b32_vop3", 458},
    {"v_bfe_u32_vop3", 456},
    {"v_bfe_i32_vop3", 457},
    // int min3/max3/med3 family
    {"v_min3_u32_vop3", 466},
    {"v_max3_u32_vop3", 469},
    {"v_med3_u32_vop3", 472},
    {"v_min3_i32_vop3", 465},
    {"v_max3_i32_vop3", 468},
    {"v_med3_i32_vop3", 471},
    {"v_min3_u16_vop3", 502},
    {"v_max3_u16_vop3", 505},
    {"v_med3_u16_vop3", 508},
    {"v_min3_i16_vop3", 501},
    {"v_max3_i16_vop3", 504},
    {"v_med3_i16_vop3", 507},
}};

// 14 mixed uint32 values; shift-count cases include small (<32) and full-low
// 5-bit values, so the lshl ops exercise normal and high-end shifts.
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

  Fixture() : gpu_mem("vop3_tern_int_mem"), l2("vop3_tern_int_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_tern_int", cfg, &gpu_mem, &l2);
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

  std::array<uint32_t, WF_SIZE> run(Instruction *inst, bool force_scalar, uint32_t rot,
                                    uint64_t exec) {
    amdgpu::simd_force_scalar() = force_scalar;
    seed_inputs(rot, exec);
    cu->execute_instruction(inst, *wf);
    amdgpu::simd_force_scalar() = false;
    std::array<uint32_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vb + kDstVgpr, lane);
    return out;
  }
};

void check_case(const Case &c, uint64_t exec) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  uint32_t words[4] = {0u, 0u, 0u, 0u};
  vop3_tern_encode(c.opcode, /*vdst=*/kDstVgpr, /*src0=*/256, /*src1=*/257, /*src2=*/258, words);
  Instruction *inst = fx.decoder->decode(words);
  ASSERT_NE(inst, nullptr) << c.name << " decode failed";
  for (uint32_t rot = 0; rot < kVals.size(); ++rot) {
    auto scalar = fx.run(inst, /*force_scalar=*/true, rot, exec);
    auto simd = fx.run(inst, /*force_scalar=*/false, rot, exec);
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (!active) {
        EXPECT_EQ(simd[lane], DST_SENTINEL)
            << c.name << " rot=" << rot << ": SIMD clobbered inactive lane " << lane;
        continue;
      }
      EXPECT_EQ(scalar[lane], simd[lane])
          << c.name << " rot=" << rot << " lane=" << lane << ": divergence scalar=0x" << std::hex
          << scalar[lane] << " simd=0x" << simd[lane];
    }
  }
  delete inst;
}

TEST(Vop3TernaryIntSimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/~0ULL);
}

TEST(Vop3TernaryIntSimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
