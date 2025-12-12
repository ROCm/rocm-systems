// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_int_binary_extra_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the VOP3-only
/// plain integer add/sub forms that have no VOP2 twin on CDNA4 (v_add_i32,
/// v_sub_i32, v_add_i16, v_sub_i16). All apply no modifiers and reuse
/// try_execute_binary_vop3_simd<uint32_t> via the SIMD_VOP3_BINARY_INT_EXTRA
/// table — the i16 forms mask the low 16 bits in the functor to match the scalar
/// `uint32(uint16(int16(...)))` zero-extension. Each (case, rot) runs TWICE in
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
constexpr uint32_t kDstVgpr = 4;
constexpr uint32_t DST_SENTINEL = 0xCDCDCDCDu;

constexpr void vop3_bin_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1,
                               uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((op & 0x3FF) << 16) | (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9);
}

struct Case {
  const char *name;
  uint32_t opcode;
  bool i16;
};

const std::array<Case, 6> kCases = {{
    {"v_add_i32_vop3", 668, false},
    {"v_sub_i32_vop3", 669, false},
    {"v_add_i16_vop3", 670, true},
    {"v_sub_i16_vop3", 671, true},
    // Pack two clamped 32-bit ints into the dst's hi/lo 16-bit halves. kVals
    // includes the ±32768 / 0xFFFF / INT extremes that exercise both clamps.
    {"v_cvt_pk_u16_u32_vop3", 663, false},
    {"v_cvt_pk_i16_i32_vop3", 664, false},
}};

// Mixed uint32 values; for the i16 ops, only the low 16 bits matter to the
// result (high 16 must be ignored by the functor's & 0xFFFFu mask, the scalar
// equivalent is the `static_cast<int16_t>(read_lane)` truncation).
const std::array<uint32_t, 14> kVals = {{
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

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3_int_binx_mem"), l2("vop3_int_binx_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_int_binx", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_inputs(uint32_t rot, uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      cu->write_vgpr(vb + 0, lane, kVals[lane % kVals.size()]);
      cu->write_vgpr(vb + 1, lane, kVals[(lane + rot) % kVals.size()]);
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
    vop3_bin_encode(c.opcode, /*vdst=*/kDstVgpr, /*src0=*/256, /*src1=*/257, words);
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

TEST(Vop3IntBinaryExtraSimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/~0ULL);
}

TEST(Vop3IntBinaryExtraSimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
