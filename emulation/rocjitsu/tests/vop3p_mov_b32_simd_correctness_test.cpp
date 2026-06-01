// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3p_mov_b32_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for
/// v_pk_mov_b32_vop3p on CDNA4. Default packing only (op_sel=0,
/// op_sel_hi=3): the 64-bit result is (src0_lo, src1_hi), where each src
/// pair lives in consecutive VGPRs {base, base+1}. The SIMD path uses
/// read_simd64/write_simd64 to fetch and store the 64-bit pair. The process runs
/// one fixed execute mode (RJ_FORCE_SCALAR, immutable); the 64-bit dst (lo,hi
/// per lane) is recorded and the scalar-vs-SIMD equivalence is asserted by
/// diffing the two runs (see simd_ab.h / the simd_ab_diff CTest entry).
/// In-process inactive lanes must keep the sentinel.

#include "simd_ab.h"

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
constexpr uint32_t kDstVgpr = 8; // pair occupies kDstVgpr..kDstVgpr+1
constexpr uint32_t DST_SENTINEL = 0xCDCDCDCDu;

constexpr void vop3p_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1,
                            uint32_t words[2]) {
  // op_sel = 0, op_sel_hi = 3 (default packing). neg / neg_hi / clamp = 0.
  words[0] = (vdst & 0xFFu) | ((op & 0x7Fu) << 16) | (0x1A7u << 23);
  words[1] = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | (0x3u << 27);
}

const std::array<uint32_t, 14> kVals = {{
    0x00000000u,
    0xFFFFFFFFu,
    0x12345678u,
    0xDEADBEEFu,
    0xCAFEBABEu,
    0xA5A5A5A5u,
    0x5A5A5A5Au,
    0x80000000u,
    0x7FFFFFFFu,
    0xAAAAAAAAu,
    0x55555555u,
    0x00010001u,
    0xFEDCBA98u,
    0x13579BDFu,
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3p_mov_b32_mem"), l2("vop3p_mov_b32_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3p_mov_b32", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_inputs(uint32_t rot, uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      // src0 pair at v0:v1, src1 pair at v2:v3.
      cu->write_vgpr(vb + 0, lane, kVals[lane % kVals.size()]);
      cu->write_vgpr(vb + 1, lane, kVals[(lane + 1) % kVals.size()]);
      cu->write_vgpr(vb + 2, lane, kVals[(lane + rot) % kVals.size()]);
      cu->write_vgpr(vb + 3, lane, kVals[(lane + rot + 1) % kVals.size()]);
      cu->write_vgpr(vb + kDstVgpr + 0, lane, DST_SENTINEL);
      cu->write_vgpr(vb + kDstVgpr + 1, lane, DST_SENTINEL);
    }
    wf->set_exec(exec);
  }

  std::array<uint64_t, WF_SIZE> run(Instruction *inst, uint32_t rot, uint64_t exec) {
    seed_inputs(rot, exec);
    cu->execute_instruction(inst, *wf);
    std::array<uint64_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint64_t lo = cu->read_vgpr(vb + kDstVgpr + 0, lane);
      uint64_t hi = cu->read_vgpr(vb + kDstVgpr + 1, lane);
      out[lane] = lo | (hi << 32);
    }
    return out;
  }
};

void check(uint64_t exec) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  uint32_t words[2] = {0u, 0u};
  // src0 = VGPR 256 (pair v0:v1), src1 = VGPR 258 (pair v2:v3).
  vop3p_encode(/*op=*/51, kDstVgpr, /*src0=*/256, /*src1=*/258, words);
  Instruction *inst = fx.decoder->decode(words);
  ASSERT_NE(inst, nullptr) << "v_pk_mov_b32_vop3p decode failed";
  for (uint32_t rot = 0; rot < kVals.size(); ++rot) {
    auto out = fx.run(inst, rot, exec);

    uint32_t words_out[2 * WF_SIZE];
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      words_out[2 * lane] = static_cast<uint32_t>(out[lane]);
      words_out[2 * lane + 1] = static_cast<uint32_t>(out[lane] >> 32);
    }
    simd_ab::record("v_pk_mov_b32_vop3p:r" + std::to_string(rot), exec, words_out, 2 * WF_SIZE);

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (!active) {
        EXPECT_EQ(out[lane], (uint64_t{DST_SENTINEL} | (uint64_t{DST_SENTINEL} << 32)))
            << "rot=" << rot << ": clobbered inactive lane " << lane;
      }
    }
  }
  delete inst;
}

TEST(Vop3pMovB32SimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check(/*exec=*/~0ULL);
}

TEST(Vop3pMovB32SimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check(/*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
