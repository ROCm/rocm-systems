// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_b16_rdna_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for five
/// 16-bit-lane VOP3 ops that are RDNA3+ only (no CDNA4 decode path). Each case
/// runs TWICE in the same process -- once forcing the scalar body, once the SIMD
/// fast path, with identical inputs/EXEC -- and the two result arrays are
/// asserted equal with EXPECT_EQ (util::set_force_scalar_for_testing flips the
/// gate in-process). In-process inactive lanes must keep the sentinel.
///   - v_and_b16, v_or_b16, v_xor_b16: low-16 bitwise binary, routed
///     through the int VOP3 binary glue with a `& 0xFFFFu` mask.
///   - v_not_b16: low-16 bitwise unary, routed through the VOP1 unary glue
///     with a `& 0xFFFFu` mask.
///   - v_cndmask_b16: low-16 VCC/SGPR-pair select, routed through the new
///     try_execute_cndmask_b16_vop3_simd glue.
///
/// All five share the scalar-body pattern `uint32_t(uint16_t(... low16 ...))`
/// — the high 16 bits of the destination VGPR are zeroed. The CU and decoder
/// are built for RDNA3; encoding marker is 0x35<<26.

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

constexpr uint32_t WF_SIZE = 32; // RDNA3 wave32
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t kDstVgpr = 4;
constexpr uint32_t DST_SENTINEL = 0xCDCDCDCDu;

// RDNA3 VOP3 (encoding[31:26]=0x35). Layout matches cdna4 except the marker.
// word0 = vdst[7:0] | abs[10:8] | op_sel[14:11] | clamp[15] | op[25:16] | enc[31:26];
// word1 = src0[8:0] | src1[17:9] | src2[26:18] | omod[28:27] | neg[31:29].
constexpr void rdna3_vop3_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1,
                                 uint32_t src2, uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((op & 0x3FF) << 16) | (0x35u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((src2 & 0x1FF) << 18);
}

// Same as rdna3_vop3_encode but also sets the per-instruction clamp and omod
// modifier fields. Used by v_mov_b16, which is the only op in this file that
// honours omod / clamp (the bitwise b16 ops ignore both).
constexpr void rdna3_vop3_encode_mod(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1,
                                     uint32_t src2, uint32_t clamp, uint32_t omod,
                                     uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((clamp & 0x1u) << 15) | ((op & 0x3FF) << 16) | (0x35u << 26);
  words[1] =
      (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((src2 & 0x1FF) << 18) | ((omod & 0x3u) << 27);
}

const std::array<uint32_t, 16> kSrcA = {{
    0x00000000u,
    0xFFFFFFFFu,
    0x12345678u,
    0xDEADBEEFu,
    0xA5A5A5A5u,
    0x5A5A5A5Au,
    0x80008000u,
    0x7FFF7FFFu,
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

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3_b16_mem"), l2("vop3_b16_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_RDNA3;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_b16", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3);
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

struct BinCase {
  const char *name;
  uint32_t opcode;
};

const std::array<BinCase, 3> kBinCases = {{
    {"v_and_b16_vop3", 866},
    {"v_or_b16_vop3", 867},
    {"v_xor_b16_vop3", 868},
}};

// Restores the process force-scalar gate on scope exit so flipping it for an
// in-process A/B comparison cannot leak into later tests in the same process.
struct ForceScalarGuard {
  bool orig;
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }
};

void check_binary(const BinCase &c, uint64_t exec) {
  ForceScalarGuard gate_guard;

  auto run_mode = [&](bool force_scalar, uint32_t rot) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    rdna3_vop3_encode(c.opcode, kDstVgpr, /*src0=*/256, /*src1=*/257, /*src2=*/0, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.name << " decode failed";
    auto out = fx.run(inst, rot, exec);
    delete inst;
    return out;
  };

  for (uint32_t rot = 0; rot < kSrcB.size(); ++rot) {
    const auto scalar_out = run_mode(/*force_scalar=*/true, rot);
    const auto simd_out = run_mode(/*force_scalar=*/false, rot);

    EXPECT_EQ(scalar_out, simd_out)
        << c.name << " rot=" << rot << ": SIMD path diverged from scalar body";

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

void check_not_b16(uint64_t exec) {
  ForceScalarGuard gate_guard;

  auto run_mode = [&](bool force_scalar) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    rdna3_vop3_encode(/*op=*/489, /*vdst=*/kDstVgpr, /*src0=*/256, /*src1=*/0, /*src2=*/0, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << "v_not_b16_vop3 decode failed";
    auto out = fx.run(inst, 0, exec);
    delete inst;
    return out;
  };

  const auto scalar_out = run_mode(/*force_scalar=*/true);
  const auto simd_out = run_mode(/*force_scalar=*/false);

  EXPECT_EQ(scalar_out, simd_out) << "v_not_b16_vop3: SIMD path diverged from scalar body";

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (exec >> lane) & 1ULL;
    if (!active) {
      EXPECT_EQ(simd_out[lane], DST_SENTINEL) << "v_not_b16_vop3 clobbered inactive lane " << lane;
      EXPECT_EQ(scalar_out[lane], DST_SENTINEL)
          << "v_not_b16_vop3 clobbered inactive lane " << lane;
    }
  }
}

void check_cndmask_b16(uint64_t exec, uint64_t sel) {
  ForceScalarGuard gate_guard;

  auto run_mode = [&](bool force_scalar, uint32_t rot) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t sb = fx.wf->sgpr_alloc().base;
    fx.cu->write_sgpr(sb + 0, static_cast<uint32_t>(sel));
    fx.cu->write_sgpr(sb + 1, static_cast<uint32_t>(sel >> 32));
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    rdna3_vop3_encode(/*op=*/605, /*vdst=*/kDstVgpr, /*src0=*/256, /*src1=*/257, /*src2=*/sb,
                      words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << "v_cndmask_b16_vop3 decode failed";
    auto out = fx.run(inst, rot, exec);
    delete inst;
    return out;
  };

  for (uint32_t rot = 0; rot < kSrcB.size(); ++rot) {
    const auto scalar_out = run_mode(/*force_scalar=*/true, rot);
    const auto simd_out = run_mode(/*force_scalar=*/false, rot);

    EXPECT_EQ(scalar_out, simd_out)
        << "v_cndmask_b16_vop3 sel=0x" << std::hex << sel << " rot=" << std::dec << rot
        << ": SIMD path diverged from scalar body";

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (!active) {
        EXPECT_EQ(simd_out[lane], DST_SENTINEL) << "v_cndmask_b16_vop3 sel=0x" << std::hex << sel
                                                << ": clobbered inactive lane " << lane;
        EXPECT_EQ(scalar_out[lane], DST_SENTINEL) << "v_cndmask_b16_vop3 sel=0x" << std::hex << sel
                                                  << ": clobbered inactive lane " << lane;
      }
    }
  }
}

TEST(Vop3B16RdnaSimdCorrectness, BitwiseBinary_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  for (const auto &c : kBinCases)
    check_binary(c, /*exec=*/0xFFFFFFFFULL);
}
TEST(Vop3B16RdnaSimdCorrectness, BitwiseBinary_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  for (const auto &c : kBinCases)
    check_binary(c, /*exec=*/0xA5A58001ULL);
}
TEST(Vop3B16RdnaSimdCorrectness, NotB16_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_not_b16(/*exec=*/0xFFFFFFFFULL);
}
TEST(Vop3B16RdnaSimdCorrectness, NotB16_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_not_b16(/*exec=*/0xA5A58001ULL);
}
TEST(Vop3B16RdnaSimdCorrectness, CndmaskB16_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  for (uint64_t sel :
       {uint64_t{0}, uint64_t{~0ULL}, uint64_t{0xAAAAAAAAULL}, uint64_t{0x12345678ULL}})
    check_cndmask_b16(/*exec=*/0xFFFFFFFFULL, sel);
}
TEST(Vop3B16RdnaSimdCorrectness, CndmaskB16_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  for (uint64_t sel :
       {uint64_t{0}, uint64_t{~0ULL}, uint64_t{0xAAAAAAAAULL}, uint64_t{0x12345678ULL}})
    check_cndmask_b16(/*exec=*/0xA5A58001ULL, sel);
}

// --- v_mov_b16: u16 src0 -> f32 -> omod / clamp -> u16 dst (zero-extended) ---

void check_mov_b16(uint64_t exec, uint32_t omod, uint32_t clamp) {
  ForceScalarGuard gate_guard;

  auto run_mode = [&](bool force_scalar) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    rdna3_vop3_encode_mod(/*op=*/412, /*vdst=*/kDstVgpr, /*src0=*/256, /*src1=*/0, /*src2=*/0,
                          clamp, omod, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << "v_mov_b16_vop3 decode failed";
    auto out = fx.run(inst, 0, exec);
    delete inst;
    return out;
  };

  const auto scalar_out = run_mode(/*force_scalar=*/true);
  const auto simd_out = run_mode(/*force_scalar=*/false);

  EXPECT_EQ(scalar_out, simd_out) << "v_mov_b16 omod=" << omod << " clamp=" << clamp
                                  << ": SIMD path diverged from scalar body";

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (exec >> lane) & 1ULL;
    if (!active) {
      EXPECT_EQ(simd_out[lane], DST_SENTINEL)
          << "v_mov_b16 omod=" << omod << " clamp=" << clamp << " clobbered inactive lane " << lane;
      EXPECT_EQ(scalar_out[lane], DST_SENTINEL)
          << "v_mov_b16 omod=" << omod << " clamp=" << clamp << " clobbered inactive lane " << lane;
    }
  }
}

TEST(Vop3B16RdnaSimdCorrectness, MovB16_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  for (uint32_t omod : {0u, 1u, 2u, 3u})
    for (uint32_t clamp : {0u, 1u})
      check_mov_b16(/*exec=*/0xFFFFFFFFULL, omod, clamp);
}
TEST(Vop3B16RdnaSimdCorrectness, MovB16_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  for (uint32_t omod : {0u, 1u, 2u, 3u})
    for (uint32_t clamp : {0u, 1u})
      check_mov_b16(/*exec=*/0xA5A58001ULL, omod, clamp);
}

} // namespace
