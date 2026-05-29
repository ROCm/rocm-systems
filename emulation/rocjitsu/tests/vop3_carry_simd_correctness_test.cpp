// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_carry_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs forced-scalar body) for the
/// no-carry-in VOP3 sdst-enc carry-bearing ops on CDNA4:
///   v_add_co_u32, v_sub_co_u32, v_subrev_co_u32.
/// These write the per-lane carry/borrow into an arbitrary SGPR pair
/// (sdst) rather than VCC. The carry-in forms (addc_co / subb_co /
/// subbrev_co on CDNA4, plus add_co_ci / sub_co_ci / subrev_co_ci on
/// RDNA3+) read carry-in from a different source (src2 SGPR-pair vs VCC)
/// and need a separate glue shape; they are deferred to a follow-up.
/// The test runs both modes in-process via simd_force_scalar() and
/// asserts that the destination VGPR AND the full 64-bit SGPR-pair
/// carry result agree, under full and partial EXEC masks. Inputs
/// deliberately seed the 32-bit carry/borrow boundary
/// (0xFFFFFFFF+1, a<b, …) on the low lanes — random-only inputs hide
/// these corners.

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

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t DST_SENTINEL = 0xCAFEF00Du;

// CDNA4 Vop3SdstEnc layout (shared/machine_insts_cdna.h):
//   word0: vdst[7:0] | sdst[14:8] | clamp[15] | op[25:16] | encoding[31:26]
//   word1: src0[8:0] | src1[17:9] | src2[26:18] | omod[28:27] | neg[31:29]
// CDNA4 VOP3 encoding marker = 0x34 << 26.
constexpr void vop3_sdstenc_encode(uint32_t op, uint32_t vdst, uint32_t sdst, uint32_t src0,
                                   uint32_t src1, uint32_t words[2]) {
  words[0] = (vdst & 0xFFu) | ((sdst & 0x7Fu) << 8) | ((op & 0x3FFu) << 16) | (0x34u << 26);
  words[1] = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9);
}

const std::array<std::pair<uint32_t, uint32_t>, 14> kEdgePairs = {{
    {0xFFFFFFFFu, 0x00000001u},
    {0xFFFFFFFFu, 0x00000000u},
    {0x00000000u, 0x00000000u},
    {0x00000001u, 0x00000000u},
    {0x00000000u, 0x00000001u},
    {0x80000000u, 0x80000000u},
    {0x7FFFFFFFu, 0x00000001u},
    {0xFFFFFFFFu, 0xFFFFFFFFu},
    {0x00000000u, 0xFFFFFFFFu},
    {0xFFFFFFFEu, 0x00000001u},
    {0x12345678u, 0x12345678u},
    {0x12345679u, 0x12345678u},
    {0xAAAAAAAAu, 0x55555555u},
    {0x00000002u, 0x00000003u},
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3_carry_simd_mem"), l2("vop3_carry_simd_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_carry_simd", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_inputs(uint64_t seed, uint64_t exec, uint64_t vcc_in, uint64_t sdst_in,
                   uint32_t sdst_sgpr) {
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
    cu->write_sgpr(sdst_sgpr + 0, static_cast<uint32_t>(sdst_in));
    cu->write_sgpr(sdst_sgpr + 1, static_cast<uint32_t>(sdst_in >> 32));
  }

  struct Result {
    std::array<uint32_t, WF_SIZE> dst{};
    uint64_t sdst = 0;
  };

  Result run(Instruction *inst, bool force_scalar, uint64_t seed, uint64_t exec, uint64_t vcc_in,
             uint64_t sdst_in, uint32_t sdst_sgpr) {
    amdgpu::simd_force_scalar() = force_scalar;
    seed_inputs(seed, exec, vcc_in, sdst_in, sdst_sgpr);
    cu->execute_instruction(inst, *wf);
    amdgpu::simd_force_scalar() = false;
    Result res;
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      res.dst[lane] = cu->read_vgpr(vbase + 2, lane);
    uint64_t lo = cu->read_sgpr(sdst_sgpr + 0);
    uint64_t hi = cu->read_sgpr(sdst_sgpr + 1);
    res.sdst = (hi << 32) | lo;
    return res;
  }
};

struct CarryCase {
  const char *label;
  uint32_t opcode;
};

const CarryCase kCases[] = {
    {"v_add_co_u32_vop3", 281},
    {"v_sub_co_u32_vop3", 282},
    {"v_subrev_co_u32_vop3", 283},
};

const uint64_t kVccPatterns[] = {
    0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0xAAAAAAAAAAAAAAAAULL,
    0x5555555555555555ULL, 0x0123456789ABCDEFULL,
};

void check_case(const CarryCase &c, uint64_t exec) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  uint32_t sb = fx.wf->sgpr_alloc().base;
  // SGPR-pair carry-out target; must be an even-aligned SREG index.
  ASSERT_EQ(sb % 2u, 0u) << c.label << ": sgpr_alloc base not pair-aligned";
  uint32_t words[4] = {0u, 0u, 0u, 0u};
  vop3_sdstenc_encode(c.opcode, /*vdst=*/2, /*sdst=*/sb,
                      /*src0=*/256, /*src1=*/257, words);
  Instruction *inst = fx.decoder->decode(words);
  ASSERT_NE(inst, nullptr) << c.label << ": decode failed";

  constexpr uint64_t SEED = 0xC0FFEE'1234'5678ULL;
  // Seed the SGPR-pair with a recognisable pattern so any "didn't write"
  // bug surfaces as a divergent (incoming vs scalar) result.
  constexpr uint64_t SDST_SEED = 0xDEADBEEFCAFEF00DULL;
  for (uint64_t vcc_in : kVccPatterns) {
    auto scalar = fx.run(inst, /*force_scalar=*/true, SEED, exec, vcc_in, SDST_SEED, sb);
    auto simd = fx.run(inst, /*force_scalar=*/false, SEED, exec, vcc_in, SDST_SEED, sb);
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      EXPECT_EQ(scalar.dst[lane], simd.dst[lane])
          << c.label << " vcc_in=0x" << std::hex << vcc_in << ": dst divergence at lane "
          << std::dec << lane << std::hex << " scalar=0x" << scalar.dst[lane] << " simd=0x"
          << simd.dst[lane];
      if (!active) {
        EXPECT_EQ(simd.dst[lane], DST_SENTINEL)
            << c.label << ": SIMD clobbered inactive dst lane " << lane;
      }
    }
    EXPECT_EQ(scalar.sdst, simd.sdst)
        << c.label << " vcc_in=0x" << std::hex << vcc_in << ": sdst divergence scalar=0x"
        << scalar.sdst << " simd=0x" << simd.sdst;
  }
  delete inst;
}

TEST(Vop3CarrySimdCorrectness, FullExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/~0ULL);
}

TEST(Vop3CarrySimdCorrectness, PartialExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
