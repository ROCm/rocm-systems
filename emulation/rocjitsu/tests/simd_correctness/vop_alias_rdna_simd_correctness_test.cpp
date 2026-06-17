// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop_alias_rdna_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the RDNA-only
/// VOP1 / VOP2 / VOP3 aliases routed into existing SIMD glue tables in the
/// 2026-05-29 sweep slice. RDNA3 fixture (wave32, encoding markers VOP3=0x35).
/// Each case runs TWICE in the same process -- once forcing the scalar body,
/// once the SIMD fast path, with identical inputs/EXEC/VCC-in -- and the
/// scalar-vs-SIMD equivalence on the destination VGPR (and VCC for the carry
/// forms) is asserted with EXPECT_EQ (util::set_force_scalar_for_testing flips
/// the gate in-process). In-process inactive lanes must keep the sentinel.
///
/// Ops covered:
///   VOP2 (6, RDNA-only naming):
///     v_add_nc_u32 / v_sub_nc_u32 / v_subrev_nc_u32  (plain int add/sub)
///     v_add_co_ci_u32 / v_sub_co_ci_u32 / v_subrev_co_ci_u32  (cin from VCC,
///       co to VCC; same scalar shape as CDNA's addc/subb/subbrev forms).
///   VOP1 (6, RDNA3+):
///     v_mov_b16 / v_not_b16   (low-16 forms)
///     v_cvt_i32_i16 / v_cvt_u32_u16  (sign/zero-extend low 16)
///     v_cvt_floor_i32_f32 / v_cvt_nearest_i32_f32  (aliases for flr / rpi)
///   VOP3 (8):
///     same 6 VOP1 ops in their VOP3 form (all route via the `_vop3` ->
///     SIMD_VOP1_UNARY twin fallback since the scalar body of each ignores
///     modifiers — verified inline in the regen for this slice) + 2 ints
///     missing from CDNA4 (v_minmax / v_maxmin u32/i32 are RDNA3+ only).

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
#include <random>
#include <string>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 32; // RDNA3 wave32
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t kDstVgpr = 6;
constexpr uint32_t DST_SENTINEL = 0xDEFEC8EDu;
constexpr uint32_t SRC0_VGPR = 256u;
constexpr uint32_t SRC1_VGPR = 257u;
constexpr uint32_t SRC2_VGPR = 258u;

// VOP1 encoding: enc[31:25]=0x3F | vdst[24:17] | op[16:9] | src0[8:0].
constexpr uint32_t vop1_encode(uint32_t op, uint32_t vdst, uint32_t src0) {
  return (0x3Fu << 25) | ((vdst & 0xFFu) << 17) | ((op & 0xFFu) << 9) | (src0 & 0x1FFu);
}

// VOP2 encoding: enc[31]=0 | op[30:25] | vdst[24:17] | vsrc1[16:9] | src0[8:0].
constexpr uint32_t vop2_encode(uint32_t op, uint32_t vdst, uint32_t vsrc1, uint32_t src0) {
  return ((op & 0x3Fu) << 25) | ((vdst & 0xFFu) << 17) | ((vsrc1 & 0xFFu) << 9) | (src0 & 0x1FFu);
}

// RDNA3 VOP3 encoding (Vop3MachineInst, marker 0x35 << 26):
//   word0: vdst[7:0] | abs[10:8] | opsel[14:11] | clamp[15] | op[25:16] | enc[31:26]
//   word1: src0[8:0] | src1[17:9] | src2[26:18] | omod[28:27] | neg[31:29]
constexpr void vop3_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1, uint32_t src2,
                           uint32_t words[2]) {
  words[0] = (vdst & 0xFFu) | ((op & 0x3FFu) << 16) | (0x35u << 26);
  words[1] = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18);
}

// Mixed input set for VGPR seeding. Hits the carry/borrow boundary for the
// VOP2 carry forms and the int32/uint32 / int16/uint16 extremes for cvts /
// min/max ternaries.
const std::array<uint32_t, 14> kVals = {{
    0x00000000u,
    0x00000001u,
    0x0000FFFFu,
    0x00008000u,
    0xFFFF0000u,
    0x80000000u,
    0x7FFFFFFFu,
    0xFFFFFFFFu,
    0xDEADBEEFu,
    0xCAFEBABEu,
    0xA5A5A5A5u,
    0x5A5A5A5Au,
    0x00010001u,
    0xFFFE7FFFu,
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop_alias_rdna_mem"), l2("vop_alias_rdna_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_RDNA3;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop_alias_rdna", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_inputs(uint32_t rot, uint64_t exec, uint64_t vcc_in) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      cu->write_vgpr(vb + 0, lane, kVals[lane % kVals.size()]);
      cu->write_vgpr(vb + 1, lane, kVals[(lane + rot) % kVals.size()]);
      cu->write_vgpr(vb + 2, lane, kVals[(lane + 2 * rot) % kVals.size()]);
      cu->write_vgpr(vb + kDstVgpr, lane, DST_SENTINEL);
    }
    wf->set_exec(exec);
    wf->set_vcc(vcc_in);
  }

  struct Result {
    std::array<uint32_t, WF_SIZE> dst{};
    uint64_t vcc = 0;
  };

  Result run(Instruction *inst, uint32_t rot, uint64_t exec, uint64_t vcc_in) {
    seed_inputs(rot, exec, vcc_in);
    cu->execute_instruction(inst, *wf);
    Result res;
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      res.dst[lane] = cu->read_vgpr(vb + kDstVgpr, lane);
    res.vcc = wf->vcc();
    return res;
  }
};

struct VopCase {
  const char *name;
  enum class Kind { VOP1, VOP2, VOP2_CARRY, VOP3_UNARY, VOP3_TERNARY } kind;
  uint32_t opcode;
};

const VopCase kCases[] = {
    // VOP2 nc / co_ci
    {"v_add_nc_u32_vop2", VopCase::Kind::VOP2, 37},
    {"v_sub_nc_u32_vop2", VopCase::Kind::VOP2, 38},
    {"v_subrev_nc_u32_vop2", VopCase::Kind::VOP2, 39},
    {"v_add_co_ci_u32_vop2", VopCase::Kind::VOP2_CARRY, 32},
    {"v_sub_co_ci_u32_vop2", VopCase::Kind::VOP2_CARRY, 33},
    {"v_subrev_co_ci_u32_vop2", VopCase::Kind::VOP2_CARRY, 34},
    // VOP1 b16 + cvts
    {"v_mov_b16_vop1", VopCase::Kind::VOP1, 28},
    {"v_not_b16_vop1", VopCase::Kind::VOP1, 105},
    {"v_cvt_i32_i16_vop1", VopCase::Kind::VOP1, 106},
    {"v_cvt_u32_u16_vop1", VopCase::Kind::VOP1, 107},
    {"v_cvt_nearest_i32_f32_vop1", VopCase::Kind::VOP1, 12},
    {"v_cvt_floor_i32_f32_vop1", VopCase::Kind::VOP1, 13},
    // VOP3 forms of the same (same opcode index in sub_decode_vop3 differs;
    // values resolved via the regen lookup).
    {"v_mov_b16_vop3", VopCase::Kind::VOP3_UNARY, 412},
    {"v_not_b16_vop3", VopCase::Kind::VOP3_UNARY, 489},
    {"v_cvt_i32_i16_vop3", VopCase::Kind::VOP3_UNARY, 490},
    {"v_cvt_u32_u16_vop3", VopCase::Kind::VOP3_UNARY, 491},
    {"v_cvt_nearest_i32_f32_vop3", VopCase::Kind::VOP3_UNARY, 396},
    {"v_cvt_floor_i32_f32_vop3", VopCase::Kind::VOP3_UNARY, 397},
    // VOP3 ternary int minmax / maxmin remain scalar-only: their scalar bodies
    // use std::fmin/fmax on ints and write back through a double->uint32 cast
    // that saturates negatives to 0xFFFFFFFF (x86 overflow), which the integer
    // SIMD min/max cannot match. The fp minmax/maxmin f32/f16 forms ARE wired
    // (SIMD_VOP3_TERNARY_FP32/_FP16, same fmax/fmin as the verified min3/max3),
    // but are not exercised here: this fixture's values alias to f32 NaNs and
    // the comparison has no NaN skip.
};

const uint64_t kVccPatterns[] = {
    0x0000000000000000ULL, 0x00000000FFFFFFFFULL, 0x00000000AAAAAAAAULL,
    0x0000000055555555ULL, 0x000000000123456FULL,
};

// Restores the process force-scalar gate on scope exit so flipping it for an
// in-process A/B comparison cannot leak into later tests in the same process.
struct ForceScalarGuard {
  bool orig;
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }
};

void check_case(const VopCase &c, uint64_t exec) {
  ForceScalarGuard gate_guard;

  const bool has_vcc_in = (c.kind == VopCase::Kind::VOP2_CARRY);

  // Runs one (case, vcc_in, rot) in the requested execute mode (fresh Fixture +
  // decode per run isolates VGPR/VCC state).
  auto run_mode = [&](bool force_scalar, uint64_t vcc_in, uint32_t rot) -> Fixture::Result {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    switch (c.kind) {
    case VopCase::Kind::VOP1:
      words[0] = vop1_encode(c.opcode, /*vdst=*/kDstVgpr, /*src0=*/SRC0_VGPR);
      break;
    case VopCase::Kind::VOP2:
    case VopCase::Kind::VOP2_CARRY:
      words[0] = vop2_encode(c.opcode, /*vdst=*/kDstVgpr, /*vsrc1=*/(SRC1_VGPR & 0xFFu),
                             /*src0=*/SRC0_VGPR);
      break;
    case VopCase::Kind::VOP3_UNARY:
      vop3_encode(c.opcode, kDstVgpr, SRC0_VGPR, /*src1=*/0, /*src2=*/0, words);
      break;
    case VopCase::Kind::VOP3_TERNARY:
      vop3_encode(c.opcode, kDstVgpr, SRC0_VGPR, SRC1_VGPR, SRC2_VGPR, words);
      break;
    }
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.name << ": decode failed";
    auto out = fx.run(inst, rot, exec, vcc_in);
    delete inst;
    return out;
  };

  const size_t vcc_n = has_vcc_in ? std::size(kVccPatterns) : 1u;
  for (size_t v = 0; v < vcc_n; ++v) {
    uint64_t vcc_in = kVccPatterns[v];
    for (uint32_t rot = 0; rot < kVals.size(); ++rot) {
      const auto scalar_out = run_mode(/*force_scalar=*/true, vcc_in, rot);
      const auto simd_out = run_mode(/*force_scalar=*/false, vcc_in, rot);

      // Core A/B equivalence on the dst; for carry forms also the full 64-bit VCC.
      EXPECT_EQ(scalar_out.dst, simd_out.dst)
          << c.name << " vcc=0x" << std::hex << vcc_in << " rot=" << std::dec << rot
          << ": SIMD dst diverged from scalar body";
      if (has_vcc_in) {
        EXPECT_EQ(scalar_out.vcc, simd_out.vcc)
            << c.name << " vcc=0x" << std::hex << vcc_in << " rot=" << std::dec << rot
            << ": SIMD VCC diverged from scalar body";
      }

      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        const bool active = (exec >> lane) & 1ULL;
        if (!active) {
          EXPECT_EQ(simd_out.dst[lane], DST_SENTINEL)
              << c.name << " rot=" << rot << ": clobbered inactive lane " << lane;
          EXPECT_EQ(scalar_out.dst[lane], DST_SENTINEL)
              << c.name << " rot=" << rot << ": clobbered inactive lane " << lane;
        }
      }
    }
  }
}

TEST(VopAliasRdnaSimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0x00000000FFFFFFFFULL);
}

TEST(VopAliasRdnaSimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0x00000000A5A5F0F1ULL);
}

} // namespace
