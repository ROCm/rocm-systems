// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop1_cvt_f8_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the f8 -> f32
/// expand ops on CDNA4: v_cvt_f32_fp8 (E4M3) and v_cvt_f32_bf8 (E5M2), in both
/// their VOP1 and VOP3 encodings. The 8-bit source has only 256 distinct
/// values, so every byte 0..255 is swept exhaustively (4 wavefront fills of 64
/// lanes) under both a full and a partial EXEC mask. The process runs one fixed
/// execute mode (RJ_FORCE_SCALAR, immutable); each (encoding, block) runs TWICE
/// in the same process -- once forcing the scalar body, once the SIMD fast path,
/// with identical inputs/EXEC -- and the two 64-lane result arrays are asserted
/// equal with EXPECT_EQ (util::set_force_scalar_for_testing flips the gate
/// in-process). In-process inactive lanes must keep the DST sentinel.

#include "compute_unit_test_util.h"
#include "decode_test_util.h"
#include "util/simd_test_hooks.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/execute_shared.h"
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
    rocjitsu::test::initialize_compute_unit(cu, "cu_vop1_f8_simd", cfg, &gpu_mem, &l2);
    if (!cu)
      return;
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

  std::array<uint32_t, WF_SIZE> run(Instruction *inst, uint32_t block, uint64_t exec) {
    seed_inputs(block, exec);
    EXPECT_TRUE(cu->execute_instruction(inst, *wf).succeeded());
    std::array<uint32_t, WF_SIZE> out{};
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vbase + kDstVgpr, lane);
    return out;
  }
};

TEST(Cdna4CvtScaleTest, PackingWithUnitScaleMatchesRegisterScale) {
  Fixture fx;
  ASSERT_TRUE(fx.wf);
  const auto base = fx.wf->vgpr_alloc().base;
  const std::array<uint32_t, 16> inputs{0,          0x80000000, 1,          0x807fffff,
                                        0x00800000, 0x3f880000, 0xbf980000, 0x3a800000,
                                        0x43e00000, 0x47700000, 0x7f7fffff, 0x7f800000,
                                        0xff800000, 0x7f800001, 0xffc12345, 0x3b400000};
  for (const auto op : {cdna4::kVCvtScalef32PkFp8F32Vop3, cdna4::kVCvtScalef32PkBf8F32Vop3}) {
    for (unsigned high : {0u, 1u}) {
      uint32_t words[4]{};
      vop3_encode(op, 0, 256, words);
      words[0] |= high << 14;
      words[1] |= (257u << 9) | (242u << 18);
      std::unique_ptr<Instruction> fast(decode_valid(*fx.decoder, words));
      words[0] |= 4u;
      words[1] = 256u | (257u << 9) | (258u << 18);
      std::unique_ptr<Instruction> reference(decode_valid(*fx.decoder, words));
      ASSERT_TRUE(fast && reference);
      for (unsigned mode = 0; mode < 8; ++mode) {
        fx.wf->set_mode_raw(((mode & 3u) << 4) | ((mode & 4u) ? fx.wf->FP16_OVFL_BIT : 0u));
        fx.wf->set_exec(0x5555555555555555ULL);
        for (unsigned lane = 0; lane < WF_SIZE; ++lane) {
          const auto x = inputs[(lane / 2) % inputs.size()];
          fx.cu->write_vgpr(base, lane, x);
          fx.cu->write_vgpr(base + 4, lane, x);
          fx.cu->write_vgpr(base + 1, lane, inputs[(lane / 2 + 7) % inputs.size()]);
          fx.cu->write_vgpr(base + 2, lane, 0x3f800000u);
        }
        ASSERT_TRUE(fx.cu->execute_instruction(reference.get(), *fx.wf).succeeded());
        ASSERT_TRUE(fx.cu->execute_instruction(fast.get(), *fx.wf).succeeded());
        for (unsigned lane = 0; lane < WF_SIZE; ++lane)
          EXPECT_EQ(fx.cu->read_vgpr(base, lane), fx.cu->read_vgpr(base + 4, lane)) << lane;
      }
    }
  }
}

TEST(Cdna4CvtScaleTest, PackedInlineUnitScaleMatchesRegisterScaleForEveryByte) {
  Fixture fx;
  ASSERT_TRUE(fx.wf);
  const auto base = fx.wf->vgpr_alloc().base;
  for (const auto op : {cdna4::kVCvtScalef32PkF32Fp8Vop3, cdna4::kVCvtScalef32PkF32Bf8Vop3}) {
    for (unsigned high : {0u, 1u}) {
      uint32_t words[4]{};
      vop3_encode(op, 2, 256, words);
      words[0] |= high << 11;
      words[1] |= 242u << 9; // Inline 1.0.
      std::unique_ptr<Instruction> fast(decode_valid(*fx.decoder, words));
      words[0] = (words[0] & ~0xFFu) | 4u;
      words[1] = 256u | (257u << 9); // v1 holds the same E8M0 scale.
      std::unique_ptr<Instruction> reference(decode_valid(*fx.decoder, words));
      ASSERT_TRUE(fast && reference);
      for (uint32_t block = 0; block < 4; ++block) {
        for (auto mask : {UINT64_MAX, uint64_t{0x5a5a5a5a5a5a5a5a}}) {
          fx.wf->set_exec(mask);
          for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
            const uint32_t byte = block * WF_SIZE + lane;
            const auto packed = (byte * 0x01010101u) ^ 0xAA55FF00u;
            fx.cu->write_vgpr(base, lane, packed);
            fx.cu->write_vgpr(base + 1, lane, 0x3f800000u);
            for (unsigned reg = 2; reg <= 5; ++reg)
              fx.cu->write_vgpr(base + reg, lane, DST_SENTINEL);
          }
          ASSERT_TRUE(fx.cu->execute_instruction(fast.get(), *fx.wf).succeeded());
          ASSERT_TRUE(fx.cu->execute_instruction(reference.get(), *fx.wf).succeeded());
          for (unsigned word = 0; word < 2; ++word)
            for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
              EXPECT_EQ(fx.cu->read_vgpr(base + 2 + word, lane),
                        (mask >> lane & 1) ? fx.cu->read_vgpr(base + 4 + word, lane)
                                           : DST_SENTINEL);
        }
      }
    }
  }
}

// Restores the process force-scalar gate on scope exit so flipping it for an
// in-process A/B comparison cannot leak into later tests in the same process.
struct ForceScalarGuard {
  bool orig;
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }
};

// Decode `words` and exhaustively sweep all 256 byte values; compare scalar vs
// SIMD per block. `label` must be unique per encoding.
void check_words(const std::string &label, const uint32_t words[4], uint64_t exec) {
  ForceScalarGuard gate_guard;

  auto run_mode = [&](bool force_scalar, uint32_t block) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    if (::testing::Test::HasFatalFailure())
      return {};
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    Instruction *inst = decode_valid(*fx.decoder, words);
    EXPECT_NE(inst, nullptr) << label << " decode failed";
    auto out = fx.run(inst, block, exec);
    delete inst;
    return out;
  };

  for (uint32_t block = 0; block < 4; ++block) { // 4 * 64 = all 256 byte values
    const auto scalar_out = run_mode(/*force_scalar=*/true, block);
    const auto simd_out = run_mode(/*force_scalar=*/false, block);

    EXPECT_EQ(scalar_out, simd_out)
        << label << " block=" << block << ": SIMD path diverged from scalar body";

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (!active) {
        EXPECT_EQ(simd_out[lane], DST_SENTINEL) << label << ": clobbered inactive lane " << lane;
        EXPECT_EQ(scalar_out[lane], DST_SENTINEL) << label << ": clobbered inactive lane " << lane;
      }
    }
  }
}

void check_both_encodings(const Case &c, uint64_t exec) {
  uint32_t v1[4] = {vop1_encode(c.vop1_op, /*vdst=*/kDstVgpr, /*src0=*/256), 0u, 0u, 0u};
  check_words(std::string(c.name) + ":vop1", v1, exec);
  uint32_t v3[4] = {0u, 0u, 0u, 0u};
  vop3_encode(c.vop3_op, /*vdst=*/kDstVgpr, /*src0=*/256, v3);
  check_words(std::string(c.name) + ":vop3", v3, exec);
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
