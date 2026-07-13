// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_cndmask_pack_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for two VOP3 ops
/// that don't fit the binary/unary tables and reach SIMD through dedicated
/// paths. Each case runs TWICE in the same process -- once forcing the scalar
/// body, once the SIMD fast path, with identical inputs/EXEC -- and the two
/// 64-lane result arrays are asserted equal with EXPECT_EQ
/// (util::set_force_scalar_for_testing flips the gate in-process).
/// In-process inactive lanes must keep the sentinel.
///   - v_cndmask_b32_vop3: per-lane select from a 64-bit selector read out of
///     the SGPR-pair `src2` (instead of fixed VCC); routes through the new
///     try_execute_cndmask_vop3_simd glue.
///   - v_pack_b32_f16_vop3: pack the OPSEL-selected 16-bit halves of src0/src1
///     into a b32 dst.

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
constexpr uint32_t kDstVgpr = 4;
constexpr uint32_t DST_SENTINEL = 0xCDCDCDCDu;

// VOP3 ternary encoding (also fits cndmask which has 3 srcs). word0 =
// vdst[7:0] | op[25:16] | enc[31:26] (0x34); word1 = src0[8:0] | src1[17:9] |
// src2[26:18]. abs/neg/omod/clamp stay 0.
constexpr void vop3_tern_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1,
                                uint32_t src2, uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((op & 0x3FF) << 16) | (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((src2 & 0x1FF) << 18);
}

// VOP3 binary encoding (cndmask uses ternary; pack uses this). word0 = vdst |
// op<<16 | 0x34<<26; word1 = src0 | src1<<9.
constexpr uint16_t low16(uint32_t value) { return static_cast<uint16_t>(value); }
constexpr uint16_t high16(uint32_t value) { return static_cast<uint16_t>(value >> 16); }

constexpr void vop3_bin_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1,
                               uint32_t words[2], uint32_t op_sel = 0) {
  words[0] = (vdst & 0xFF) | ((op_sel & 0xFu) << 11) | ((op & 0x3FF) << 16) | (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9);
}

// 16 mixed 32-bit values; cndmask treats them as raw b32, pack uses low 16 bits.
const std::array<uint32_t, 16> kSrcA = {{
    0x00000000u,
    0xFFFFFFFFu,
    0x12345678u,
    0xDEADBEEFu,
    0xA5A5A5A5u,
    0x5A5A5A5Au,
    0x80000000u,
    0x7FFFFFFFu,
    0x0F0F0F0Fu,
    0xF0F0F0F0u,
    0x00FF00FFu,
    0xFF00FF00u,
    0xCAFEBABEu,
    0x0123FEDCu,
    0x76543210u,
    0xABCDEF01u,
}};

const std::array<uint32_t, 16> kSrcB = {{
    0xC0DEC0DEu,
    0x13579BDFu,
    0x2468ACE0u,
    0xBADF00D0u,
    0x55555555u,
    0xAAAAAAAAu,
    0x33333333u,
    0xCCCCCCCCu,
    0x00010203u,
    0xFCFDFEFFu,
    0xDEADC0DEu,
    0x8BADF00Du,
    0xFEEDFACEu,
    0xF00DBABEu,
    0x12345678u,
    0x87654321u,
}};

// Selector patterns spanning fully-on, fully-off, alternating, and biased.
const std::array<uint64_t, 6> kSels = {{
    0x0000000000000000ULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xAAAAAAAAAAAAAAAAULL,
    0x5555555555555555ULL,
    0x0123456789ABCDEFULL,
    0xFEDCBA9876543210ULL,
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3_cmkpack_mem"), l2("vop3_cmkpack_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_cmkpack", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_vgprs(uint32_t rot, uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      cu->write_vgpr(vb + 0, lane, kSrcA[lane % kSrcA.size()]);
      cu->write_vgpr(vb + 1, lane, kSrcB[(lane + rot) % kSrcB.size()]);
      cu->write_vgpr(vb + kDstVgpr, lane, DST_SENTINEL);
    }
    wf->set_exec(exec);
  }

  // Write selector into the wavefront's first allocated SGPR pair.
  void seed_sgpr_pair(uint64_t sel) {
    uint32_t sb = wf->sgpr_alloc().base;
    cu->write_sgpr(sb + 0, static_cast<uint32_t>(sel));
    cu->write_sgpr(sb + 1, static_cast<uint32_t>(sel >> 32));
  }

  std::array<uint32_t, WF_SIZE> run(Instruction *inst, uint32_t rot, uint64_t exec) {
    seed_vgprs(rot, exec);
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

void check_cndmask_one(uint64_t exec, uint64_t sel) {
  ForceScalarGuard gate_guard;

  auto run_mode = [&](bool force_scalar, uint32_t rot) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    fx.seed_sgpr_pair(sel);
    uint32_t sb = fx.wf->sgpr_alloc().base;
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_tern_encode(/*op=*/256, /*vdst=*/kDstVgpr, /*src0=*/256, /*src1=*/257,
                     /*src2=*/sb, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << "v_cndmask_b32_vop3 decode failed";
    auto out = fx.run(inst, rot, exec);
    delete inst;
    return out;
  };

  for (uint32_t rot = 0; rot < kSrcA.size(); ++rot) {
    const auto scalar_out = run_mode(/*force_scalar=*/true, rot);
    const auto simd_out = run_mode(/*force_scalar=*/false, rot);

    EXPECT_EQ(scalar_out, simd_out)
        << "v_cndmask_b32_vop3 sel=0x" << std::hex << sel << " rot=" << std::dec << rot
        << ": SIMD path diverged from scalar body";

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (!active) {
        EXPECT_EQ(simd_out[lane], DST_SENTINEL)
            << "v_cndmask_b32_vop3 sel=0x" << std::hex << sel << " rot=" << std::dec << rot
            << ": clobbered inactive lane " << lane;
        EXPECT_EQ(scalar_out[lane], DST_SENTINEL)
            << "v_cndmask_b32_vop3 sel=0x" << std::hex << sel << " rot=" << std::dec << rot
            << ": clobbered inactive lane " << lane;
      }
    }
  }
}

void check_pack_one(uint64_t exec) {
  ForceScalarGuard gate_guard;

  auto run_mode = [&](bool force_scalar, uint32_t rot) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_bin_encode(/*op=*/672, /*vdst=*/kDstVgpr, /*src0=*/256, /*src1=*/257, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << "v_pack_b32_f16_vop3 decode failed";
    auto out = fx.run(inst, rot, exec);
    delete inst;
    return out;
  };

  for (uint32_t rot = 0; rot < kSrcA.size(); ++rot) {
    const auto scalar_out = run_mode(/*force_scalar=*/true, rot);
    const auto simd_out = run_mode(/*force_scalar=*/false, rot);

    EXPECT_EQ(scalar_out, simd_out)
        << "v_pack_b32_f16_vop3 rot=" << rot << ": SIMD path diverged from scalar body";

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (!active) {
        EXPECT_EQ(simd_out[lane], DST_SENTINEL)
            << "v_pack_b32_f16_vop3 rot=" << rot << ": clobbered inactive lane " << lane;
        EXPECT_EQ(scalar_out[lane], DST_SENTINEL)
            << "v_pack_b32_f16_vop3 rot=" << rot << ": clobbered inactive lane " << lane;
      }
    }
  }
}

void check_pack_opsel_high_halves(uint64_t exec) {
  ForceScalarGuard gate_guard;
  constexpr uint32_t kOpSelSrc0Src1High = 0x3u;
  constexpr uint32_t kRot = 5;

  auto run_mode = [&](bool force_scalar) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_bin_encode(/*op=*/672, /*vdst=*/kDstVgpr, /*src0=*/256, /*src1=*/257, words,
                    kOpSelSrc0Src1High);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << "v_pack_b32_f16_vop3 decode failed";
    auto out = fx.run(inst, kRot, exec);
    delete inst;
    return out;
  };

  const auto scalar_out = run_mode(/*force_scalar=*/true);
  const auto simd_out = run_mode(/*force_scalar=*/false);
  bool saw_distinguishing_lane = false;
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (exec >> lane) & 1ULL;
    if (!active) {
      EXPECT_EQ(scalar_out[lane], DST_SENTINEL) << "scalar clobbered inactive lane " << lane;
      EXPECT_EQ(simd_out[lane], DST_SENTINEL) << "simd clobbered inactive lane " << lane;
      continue;
    }
    const uint16_t lo = high16(kSrcA[lane % kSrcA.size()]);
    const uint16_t hi = high16(kSrcB[(lane + kRot) % kSrcB.size()]);
    const uint32_t want = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    EXPECT_EQ(scalar_out[lane], want) << "scalar lane " << lane;
    EXPECT_EQ(simd_out[lane], want) << "simd lane " << lane;
    const uint32_t low_half_result =
        static_cast<uint32_t>(low16(kSrcA[lane % kSrcA.size()])) |
        (static_cast<uint32_t>(low16(kSrcB[(lane + kRot) % kSrcB.size()])) << 16);
    saw_distinguishing_lane |= want != low_half_result;
  }
  EXPECT_TRUE(saw_distinguishing_lane) << "test input must distinguish low and high source halves";
}

TEST(Vop3CndmaskPackSimdCorrectness, Cndmask_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (uint64_t sel : kSels)
    check_cndmask_one(/*exec=*/~0ULL, sel);
}

TEST(Vop3CndmaskPackSimdCorrectness, Cndmask_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (uint64_t sel : kSels)
    check_cndmask_one(/*exec=*/0xA5A5'F0F0'1234'8001ULL, sel);
}

TEST(Vop3CndmaskPackSimdCorrectness, Pack_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check_pack_one(/*exec=*/~0ULL);
}

TEST(Vop3CndmaskPackSimdCorrectness, Pack_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check_pack_one(/*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

TEST(Vop3CndmaskPackSimdCorrectness, Pack_OpSelHighHalves) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check_pack_opsel_high_halves(/*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
