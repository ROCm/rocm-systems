// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop1_cvt_f8_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs forced-scalar body) for the f8 ->
/// f32 expand ops on CDNA4: v_cvt_f32_fp8 (E4M3) and v_cvt_f32_bf8 (E5M2), in
/// both their VOP1 and VOP3 encodings. The 8-bit source has only 256 distinct
/// values, so every byte 0..255 is swept exhaustively (4 wavefront fills of 64
/// lanes) under both a full and a partial EXEC mask. Each op runs both modes via
/// the amdgpu::simd_force_scalar() thread-local; all active lanes must agree
/// bit-for-bit (covering ±0, denormals, max-normal/Inf, NaN) and inactive lanes
/// must keep the DST sentinel.

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
constexpr uint32_t DST_SENTINEL = 0xCAFEF00Du;
constexpr uint32_t kDstVgpr = 2;

// CDNA4 VOP1: [31:25]=0b0111111, vdst[24:17], op[16:9], src0[8:0].
constexpr uint32_t vop1_encode(uint32_t op, uint32_t vdst, uint32_t src0) {
  return (0x3Fu << 25) | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF);
}

// CDNA4 VOP3 ([31:26]=0x34): word0 = vdst[7:0] | op[25:16] | (0x34<<26);
// word1 = src0[8:0]. Unary, no modifiers exercised (the f8 cvt bodies read none).
void vop3_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((op & 0x3FF) << 16) | (0x34u << 26);
  words[1] = (src0 & 0x1FF);
}

struct Case {
  const char *name;
  uint32_t vop1_op;
  uint32_t vop3_op;
};

const std::array<Case, 2> kCases = {{
    {"v_cvt_f32_fp8", 84, 404},
    {"v_cvt_f32_bf8", 85, 405},
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop1_f8_simd_mem"), l2("vop1_f8_simd_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop1_f8_simd", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  // Lane k holds the byte (block * 64 + k) in the low 8 bits, plus high-bit
  // noise so the body's low-byte masking is exercised too.
  void seed_inputs(uint32_t block, uint64_t exec) {
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint32_t byte = (block * WF_SIZE + lane) & 0xFFu;
      uint32_t s0 = byte | 0xABCD'00u; // upper bytes must be ignored by the op
      cu->write_vgpr(vbase + 0, lane, s0);
      cu->write_vgpr(vbase + kDstVgpr, lane, DST_SENTINEL);
    }
    wf->set_exec(exec);
  }

  std::array<uint32_t, WF_SIZE> run(Instruction *inst, bool force_scalar, uint32_t block,
                                    uint64_t exec) {
    amdgpu::simd_force_scalar() = force_scalar;
    seed_inputs(block, exec);
    cu->execute_instruction(inst, *wf);
    amdgpu::simd_force_scalar() = false;
    std::array<uint32_t, WF_SIZE> out{};
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vbase + kDstVgpr, lane);
    return out;
  }
};

// Decode `words` and exhaustively A/B all 256 byte values against the scalar.
void check_words(const char *label, const uint32_t words[4], uint64_t exec) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  Instruction *inst = fx.decoder->decode(words);
  ASSERT_NE(inst, nullptr) << label << " decode failed";
  for (uint32_t block = 0; block < 4; ++block) { // 4 * 64 = all 256 byte values
    auto scalar = fx.run(inst, /*force_scalar=*/true, block, exec);
    auto simd = fx.run(inst, /*force_scalar=*/false, block, exec);
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      uint32_t byte = (block * WF_SIZE + lane) & 0xFFu;
      EXPECT_EQ(scalar[lane], simd[lane])
          << label << ": divergence on byte 0x" << std::hex << byte << " lane " << std::dec << lane
          << std::hex << " scalar=0x" << scalar[lane] << " simd=0x" << simd[lane];
      if (!active) {
        EXPECT_EQ(simd[lane], DST_SENTINEL) << label << ": SIMD clobbered inactive lane " << lane;
      }
    }
  }
  delete inst;
}

void check_both_encodings(const Case &c, uint64_t exec) {
  uint32_t v1[4] = {vop1_encode(c.vop1_op, /*vdst=*/kDstVgpr, /*src0=*/256), 0u, 0u, 0u};
  check_words(c.name, v1, exec);
  uint32_t v3[4] = {0u, 0u, 0u, 0u};
  vop3_encode(c.vop3_op, /*vdst=*/kDstVgpr, /*src0=*/256, v3);
  check_words(c.name, v3, exec);
}

TEST(Vop1CvtF8SimdCorrectness, AllBytes_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_both_encodings(c, /*exec=*/~0ULL);
}

TEST(Vop1CvtF8SimdCorrectness, AllBytes_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_both_encodings(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
