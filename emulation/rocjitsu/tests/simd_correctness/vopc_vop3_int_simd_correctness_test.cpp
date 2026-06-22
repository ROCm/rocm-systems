// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vopc_vop3_int_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the VOP3 form
/// of the integer VOPC relational compares on CDNA4. The VOP3 form reads
/// src0/src1 (not src0/vsrc1) and writes the per-lane compare result into an
/// arbitrary SGPR-pair dst via inst.vdst.read/write_scalar64 instead of the
/// fixed VCC. We point the SGPR-pair at VCC (106) so the same harness reads the
/// result via wf.vcc(). Each (case, rot, vcc_in) runs TWICE in the same process
/// -- once forcing the scalar body, once the SIMD fast path, with identical
/// inputs/EXEC/VCC-in -- and the 64-bit compare results are asserted equal with
/// EXPECT_EQ (util::set_force_scalar_for_testing flips the gate in-process).
/// In-process inactive SGPR-pair bits must be zeroed.
///
/// Coverage: every (rel, suffix) pair routed through try_execute_vopc_vop3_*_int_simd
/// — 8 rels × {i16,u16,i32,u32,i64,u64} = 48 ops, full + partial EXEC. Each lane
/// gets a distinct (a, b) pair; a rotation sweep pairs each value with every other
/// over the test loop so eq/lt/gt/le/ge/ne all fire on real boundary cases (incl.
/// signed-negative vs unsigned wrap).

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
#include <vector>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t kVccSdst = 106;

// VOP3 ([31:26]=0x34): word0 = vdst[7:0] | abs[10:8] | clamp[15] | op[25:16] |
// enc[31:26]; word1 = src0[8:0] | src1[17:9] | omod[28:27] | neg[31:29]. The
// integer compares apply no modifiers, so abs/neg/omod/clamp stay zero.
constexpr void vop3_cmp_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1,
                               uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((op & 0x3FF) << 16) | (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9);
}

enum class Width { B32, B64 };
enum class Sign { S, U };
enum class Rel { EQ, GE, GT, LE, LT, NE, F, T };

struct Case {
  const char *name;
  uint32_t opcode;
  Width width;
  Sign sign;
  Rel rel;
};

// 48 cases — every {rel} × {i16, u16, i32, u32, i64, u64} VOP3 form. Opcodes
// come from cdna4 sub_decode_vop3 (see test setup notes in commit message).
const std::array<Case, 48> kCases = {{
    {"v_cmp_eq_i16_vop3", 162, Width::B32, Sign::S, Rel::EQ},
    {"v_cmp_eq_u16_vop3", 170, Width::B32, Sign::U, Rel::EQ},
    {"v_cmp_eq_i32_vop3", 194, Width::B32, Sign::S, Rel::EQ},
    {"v_cmp_eq_u32_vop3", 202, Width::B32, Sign::U, Rel::EQ},
    {"v_cmp_eq_i64_vop3", 226, Width::B64, Sign::S, Rel::EQ},
    {"v_cmp_eq_u64_vop3", 234, Width::B64, Sign::U, Rel::EQ},
    {"v_cmp_ge_i16_vop3", 166, Width::B32, Sign::S, Rel::GE},
    {"v_cmp_ge_u16_vop3", 174, Width::B32, Sign::U, Rel::GE},
    {"v_cmp_ge_i32_vop3", 198, Width::B32, Sign::S, Rel::GE},
    {"v_cmp_ge_u32_vop3", 206, Width::B32, Sign::U, Rel::GE},
    {"v_cmp_ge_i64_vop3", 230, Width::B64, Sign::S, Rel::GE},
    {"v_cmp_ge_u64_vop3", 238, Width::B64, Sign::U, Rel::GE},
    {"v_cmp_gt_i16_vop3", 164, Width::B32, Sign::S, Rel::GT},
    {"v_cmp_gt_u16_vop3", 172, Width::B32, Sign::U, Rel::GT},
    {"v_cmp_gt_i32_vop3", 196, Width::B32, Sign::S, Rel::GT},
    {"v_cmp_gt_u32_vop3", 204, Width::B32, Sign::U, Rel::GT},
    {"v_cmp_gt_i64_vop3", 228, Width::B64, Sign::S, Rel::GT},
    {"v_cmp_gt_u64_vop3", 236, Width::B64, Sign::U, Rel::GT},
    {"v_cmp_le_i16_vop3", 163, Width::B32, Sign::S, Rel::LE},
    {"v_cmp_le_u16_vop3", 171, Width::B32, Sign::U, Rel::LE},
    {"v_cmp_le_i32_vop3", 195, Width::B32, Sign::S, Rel::LE},
    {"v_cmp_le_u32_vop3", 203, Width::B32, Sign::U, Rel::LE},
    {"v_cmp_le_i64_vop3", 227, Width::B64, Sign::S, Rel::LE},
    {"v_cmp_le_u64_vop3", 235, Width::B64, Sign::U, Rel::LE},
    {"v_cmp_lt_i16_vop3", 161, Width::B32, Sign::S, Rel::LT},
    {"v_cmp_lt_u16_vop3", 169, Width::B32, Sign::U, Rel::LT},
    {"v_cmp_lt_i32_vop3", 193, Width::B32, Sign::S, Rel::LT},
    {"v_cmp_lt_u32_vop3", 201, Width::B32, Sign::U, Rel::LT},
    {"v_cmp_lt_i64_vop3", 225, Width::B64, Sign::S, Rel::LT},
    {"v_cmp_lt_u64_vop3", 233, Width::B64, Sign::U, Rel::LT},
    {"v_cmp_ne_i16_vop3", 165, Width::B32, Sign::S, Rel::NE},
    {"v_cmp_ne_u16_vop3", 173, Width::B32, Sign::U, Rel::NE},
    {"v_cmp_ne_i32_vop3", 197, Width::B32, Sign::S, Rel::NE},
    {"v_cmp_ne_u32_vop3", 205, Width::B32, Sign::U, Rel::NE},
    {"v_cmp_ne_i64_vop3", 229, Width::B64, Sign::S, Rel::NE},
    {"v_cmp_ne_u64_vop3", 237, Width::B64, Sign::U, Rel::NE},
    {"v_cmp_f_i16_vop3", 160, Width::B32, Sign::S, Rel::F},
    {"v_cmp_f_u16_vop3", 168, Width::B32, Sign::U, Rel::F},
    {"v_cmp_f_i32_vop3", 192, Width::B32, Sign::S, Rel::F},
    {"v_cmp_f_u32_vop3", 200, Width::B32, Sign::U, Rel::F},
    {"v_cmp_f_i64_vop3", 224, Width::B64, Sign::S, Rel::F},
    {"v_cmp_f_u64_vop3", 232, Width::B64, Sign::U, Rel::F},
    {"v_cmp_t_i16_vop3", 167, Width::B32, Sign::S, Rel::T},
    {"v_cmp_t_u16_vop3", 175, Width::B32, Sign::U, Rel::T},
    {"v_cmp_t_i32_vop3", 199, Width::B32, Sign::S, Rel::T},
    {"v_cmp_t_u32_vop3", 207, Width::B32, Sign::U, Rel::T},
    {"v_cmp_t_i64_vop3", 231, Width::B64, Sign::S, Rel::T},
    {"v_cmp_t_u64_vop3", 239, Width::B64, Sign::U, Rel::T},
}};

// Boundary values that stress every relation including signed-vs-unsigned wrap.
// Keep low 16 bits varied so the i16/u16 paths (which narrow to low 16) get
// equally good coverage; high 16 bits are non-zero so the high half is exercised
// for i32/u32. 14 values × 14 rotations gives 196 distinct (a,b) pairings.
const std::array<uint32_t, 14> kVals32 = {{
    0x00000000u,
    0x00000001u,
    0x0000FFFFu,
    0x00008000u,
    0x00007FFFu,
    0x80000000u,
    0x80000001u,
    0x7FFFFFFFu,
    0xFFFFFFFFu,
    0x12345678u,
    0xDEADBEEFu,
    0xCAFEBABEu,
    0xA5A5A5A5u,
    0x5A5A5A5Au,
}};
const std::array<uint64_t, 14> kVals64 = {{
    0x0000000000000000ull,
    0x0000000000000001ull,
    0x000000000000FFFFull,
    0x0000000080000000ull,
    0x000000007FFFFFFFull,
    0x8000000000000000ull,
    0x8000000000000001ull,
    0x7FFFFFFFFFFFFFFFull,
    0xFFFFFFFFFFFFFFFFull,
    0x123456789ABCDEF0ull,
    0xDEADBEEFCAFEBABEull,
    0xA5A5A5A55A5A5A5Aull,
    0x0000000100000000ull,
    0xFFFFFFFE00000001ull,
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vopc_vop3_int_mem"), l2("vopc_vop3_int_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vopc_vop3_int", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void write64(uint32_t reg, uint32_t lane, uint64_t v) {
    cu->write_vgpr(reg, lane, static_cast<uint32_t>(v));
    cu->write_vgpr(reg + 1, lane, static_cast<uint32_t>(v >> 32));
  }

  // src0 in v0 (or v0:v1 for 64-bit), src1 in v2 (or v2:v3 for 64-bit). +rot
  // sweeps the (a,b) pairing across the kVals tables.
  void seed_inputs(Width w, uint32_t rot, uint64_t exec, uint64_t vcc_in) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      if (w == Width::B64) {
        uint64_t a = kVals64[lane % kVals64.size()];
        uint64_t b = kVals64[(lane + rot) % kVals64.size()];
        write64(vb + 0, lane, a);
        write64(vb + 2, lane, b);
      } else {
        uint32_t a = kVals32[lane % kVals32.size()];
        uint32_t b = kVals32[(lane + rot) % kVals32.size()];
        cu->write_vgpr(vb + 0, lane, a);
        cu->write_vgpr(vb + 2, lane, b);
      }
    }
    wf->set_exec(exec);
    wf->set_vcc(vcc_in);
  }

  uint64_t run(Instruction *inst, Width w, uint32_t rot, uint64_t exec, uint64_t vcc_in) {
    seed_inputs(w, rot, exec, vcc_in);
    cu->execute_instruction(inst, *wf);
    return wf->vcc();
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

  // Runs one (case, rot, vcc_in) in the requested execute mode (fresh Fixture +
  // decode per run isolates VGPR/VCC state).
  auto run_mode = [&](bool force_scalar, uint32_t rot, uint64_t vcc_in) -> uint64_t {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    const uint32_t src1_vgpr = (c.width == Width::B64) ? 2u : 2u; // v2:v3 / v2
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_cmp_encode(c.opcode, /*vdst=*/kVccSdst, /*src0=*/256u, /*src1=*/256u + src1_vgpr, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.name << " decode failed";
    uint64_t vcc = fx.run(inst, c.width, rot, exec, vcc_in);
    delete inst;
    return vcc;
  };

  const uint64_t kVcc[] = {0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0xA5A5A5A5A5A5A5A5ULL};
  const std::size_t kRotMax = (c.width == Width::B64) ? kVals64.size() : kVals32.size();
  for (uint32_t rot = 0; rot < kRotMax; ++rot) {
    for (uint64_t vcc_in : kVcc) {
      const uint64_t scalar_vcc = run_mode(/*force_scalar=*/true, rot, vcc_in);
      const uint64_t simd_vcc = run_mode(/*force_scalar=*/false, rot, vcc_in);

      EXPECT_EQ(scalar_vcc, simd_vcc) << c.name << " rot=" << rot << " vcc_in=0x" << std::hex
                                      << vcc_in << ": SIMD result diverged from scalar body";

      const uint64_t inactive = ~exec;
      EXPECT_EQ(simd_vcc & inactive, 0ULL)
          << c.name << " rot=" << rot << ": inactive-lane dst bit not zeroed";
      EXPECT_EQ(scalar_vcc & inactive, 0ULL)
          << c.name << " rot=" << rot << ": inactive-lane dst bit not zeroed";
    }
  }
}

TEST(VopcVop3IntSimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/~0ULL);
}

TEST(VopcVop3IntSimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
