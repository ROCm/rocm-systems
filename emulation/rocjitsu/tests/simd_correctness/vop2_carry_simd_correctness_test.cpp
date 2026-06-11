// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop2_carry_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the
/// carry-bearing VOP2 instructions wired into SIMD_VOP2_CARRY:
///   v_add_co_u32, v_sub_co_u32, v_subrev_co_u32,
///   v_addc_co_u32, v_subb_co_u32, v_subbrev_co_u32.
/// Unlike the plain binary VOP2 path these write per-lane carry/borrow into
/// VCC, and the addc/subb/subbrev forms read VCC as a carry-in. Each op runs
/// TWICE in the same process -- once forcing the scalar body, once the SIMD fast
/// path, with identical seed/inputs/EXEC/VCC-in -- and the scalar-vs-SIMD
/// equivalence on BOTH the 64-lane destination AND the full 64-bit VCC is
/// asserted with EXPECT_EQ (util::set_force_scalar_for_testing flips the gate
/// in-process). Inputs deliberately seed the
/// 32-bit carry/borrow boundary (0xFFFFFFFF+1, a<b, a==b+cin, ...) on the low
/// lanes — random-only inputs hide these corners. Inactive-lane carry-out VCC
/// bits are zeroed under full and partial EXEC masks.

#include "util/simd_test_hooks.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/execute_shared.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <string>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t DST_SENTINEL = 0xCAFEF00Du;

// CDNA4 VOP2: opcode[30:25], vdst[24:17], vsrc1[16:9], src0[8:0]. Bit 31 = 0.
constexpr uint32_t vop2_encode(uint32_t opcode, uint32_t vdst, uint32_t vsrc1, uint32_t src0) {
  return ((opcode & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF);
}

// Per-lane (src0, vsrc1) pairs that straddle the carry/borrow boundary. Placed
// on the low lanes; remaining lanes are filled with RNG. These exercise the
// add carry-out (sum > 0xFFFFFFFF), the subtract borrow (a < b), and the
// carry-in tie cases (a == b, a == b + 1) that the addc/subb chain depends on.
const std::array<std::pair<uint32_t, uint32_t>, 14> kEdgePairs = {{
    {0xFFFFFFFFu, 0x00000001u}, // add: carry; sub: borrow
    {0xFFFFFFFFu, 0x00000000u}, // add: no carry; sub: no borrow
    {0x00000000u, 0x00000000u}, // zero; with cin=1 addc carries on 0xFFFFFFFF? no
    {0x00000001u, 0x00000000u}, // a>b
    {0x00000000u, 0x00000001u}, // a<b borrow; addc 0+1
    {0x80000000u, 0x80000000u}, // sum = 0x1_00000000 carry; a==b no borrow
    {0x7FFFFFFFu, 0x00000001u},
    {0xFFFFFFFFu, 0xFFFFFFFFu}, // sum carries; a==b
    {0x00000000u, 0xFFFFFFFFu}, // borrow; addc 0+0xFFFFFFFF+1 = carry
    {0xFFFFFFFEu, 0x00000001u}, // sum = 0xFFFFFFFF, +cin=1 -> carry
    {0x12345678u, 0x12345678u}, // a==b: subb borrow iff cin
    {0x12345679u, 0x12345678u}, // a==b+1: subb borrow iff cin
    {0xAAAAAAAAu, 0x55555555u},
    {0x00000002u, 0x00000003u},
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop2_carry_simd_mem"), l2("vop2_carry_simd_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop2_carry_simd", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_inputs(uint64_t seed, uint64_t exec, uint64_t vcc_in) {
    std::mt19937_64 rng(seed);
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint32_t r0, r1;
      if (lane < kEdgePairs.size()) {
        r0 = kEdgePairs[lane].first;
        r1 = kEdgePairs[lane].second;
      } else {
        r0 = static_cast<uint32_t>(rng());
        r1 = static_cast<uint32_t>(rng());
      }
      cu->write_vgpr(vbase + 0, lane, r0);
      cu->write_vgpr(vbase + 1, lane, r1);
      cu->write_vgpr(vbase + 2, lane, DST_SENTINEL);
    }
    wf->set_exec(exec);
    wf->set_vcc(vcc_in);
  }

  struct Result {
    std::array<uint32_t, WF_SIZE> dst{};
    uint64_t vcc = 0;
  };

  Result run(Instruction *inst, uint64_t seed, uint64_t exec, uint64_t vcc_in) {
    seed_inputs(seed, exec, vcc_in);
    cu->execute_instruction(inst, *wf);
    Result res;
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      res.dst[lane] = cu->read_vgpr(vbase + 2, lane);
    res.vcc = wf->vcc();
    return res;
  }
};

struct CarryCase {
  const char *label;
  uint32_t opcode;
};

const CarryCase kCases[] = {
    {"v_add_co_u32", 25},  {"v_sub_co_u32", 26},  {"v_subrev_co_u32", 27},
    {"v_addc_co_u32", 28}, {"v_subb_co_u32", 29}, {"v_subbrev_co_u32", 30},
};

// Carry-in VCC patterns to seed before execution. addc/subb/subbrev read these;
// the non-carry-in forms ignore them. Inactive carry-out bits are zeroed.
const uint64_t kVccPatterns[] = {
    0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0xAAAAAAAAAAAAAAAAULL,
    0x5555555555555555ULL, 0x0123456789ABCDEFULL,
};

// Restores the process force-scalar gate on scope exit so flipping it for an
// in-process A/B comparison cannot leak into later tests in the same process.
struct ForceScalarGuard {
  bool orig;
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }
};

void check_case(const CarryCase &c, uint64_t exec) {
  ForceScalarGuard gate_guard;

  constexpr uint64_t SEED = 0xC0FFEE'1234'5678ULL;

  // Runs one (case, vcc_in) in the requested execute mode (fresh Fixture +
  // decode per run isolates VGPR/VCC state).
  auto run_mode = [&](bool force_scalar, uint64_t vcc_in) -> Fixture::Result {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t enc = vop2_encode(c.opcode, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256);
    uint32_t words[4] = {enc, 0u, 0u, 0u};
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.label << ": decode failed";
    auto out = fx.run(inst, SEED, exec, vcc_in);
    delete inst;
    return out;
  };

  for (uint64_t vcc_in : kVccPatterns) {
    const auto scalar_out = run_mode(/*force_scalar=*/true, vcc_in);
    const auto simd_out = run_mode(/*force_scalar=*/false, vcc_in);

    // Core A/B equivalence on BOTH the destination words and the full 64-bit
    // VCC carry/borrow output.
    EXPECT_EQ(scalar_out.dst, simd_out.dst)
        << c.label << " vcc_in=0x" << std::hex << vcc_in << " (exec=0x" << exec
        << "): SIMD dst diverged from scalar body";
    EXPECT_EQ(scalar_out.vcc, simd_out.vcc)
        << c.label << " vcc_in=0x" << std::hex << vcc_in << " (exec=0x" << exec
        << "): SIMD VCC diverged from scalar body";

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (!active) {
        EXPECT_EQ(simd_out.dst[lane], DST_SENTINEL)
            << c.label << ": clobbered inactive dst lane " << lane;
        EXPECT_EQ(scalar_out.dst[lane], DST_SENTINEL)
            << c.label << ": clobbered inactive dst lane " << lane;
      }
    }
    // Inactive EXEC lanes must be zeroed in the carry-out.
    const uint64_t inactive = ~exec;
    EXPECT_EQ(simd_out.vcc & inactive, 0ULL)
        << c.label << " vcc_in=0x" << std::hex << vcc_in << ": inactive-lane VCC bit not zeroed";
    EXPECT_EQ(scalar_out.vcc & inactive, 0ULL)
        << c.label << " vcc_in=0x" << std::hex << vcc_in << ": inactive-lane VCC bit not zeroed";
  }
}

TEST(Vop2CarrySimdCorrectness, FullExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/~0ULL);
}

TEST(Vop2CarrySimdCorrectness, PartialExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  // Sparse/alternating pattern crossing SIMD chunk boundaries: exercises the
  // masked result store and per-chunk VCC merge (active bits updated, inactive
  // zeroed) on both halves of a wave64.
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
