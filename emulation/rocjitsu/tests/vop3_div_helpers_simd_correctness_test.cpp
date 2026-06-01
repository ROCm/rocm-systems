// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_div_helpers_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs forced-scalar body) for four
/// VOP3 division helpers on CDNA4:
///   - v_div_fixup_f32 / v_div_fixup_f64: NaN/Inf/zero `else if` cascade
///     ((p, b, c) -> selected float per AMD spec), routed through the
///     existing fp ternary VOP3 glue with a `div_fixup_*_simd` functor that
///     reproduces the cascade via lowest-priority-first `where` blends.
///   - v_div_fmas_f32 / v_div_fmas_f64: `fma(s0, s1, s2)` then a VCC-bit-
///     gated `ldexp(result, 32)` (f32) or `ldexp(result, 64)` (f64); no
///     omod/clamp; routed through a dedicated glue that reads VCC as an
///     input side-channel (similar to v_cndmask_b32 VCC select).
/// NaN-input lanes are skipped per-lane in the comparison (the gcc-13 packed
/// FMA quiets a different NaN operand vs scalar std::fma — accepted
/// divergence shared with the rest of the ternary fp suite).

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
constexpr uint32_t DST_SENTINEL_32 = 0xCDCDCDCDu;
constexpr uint64_t DST_SENTINEL_64 = 0xCDCDCDCDCDCDCDCDULL;
constexpr uint32_t kDstVgpr32 = 6;
constexpr uint32_t kDstVgpr64 = 8;

constexpr void vop3_tern_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1,
                                uint32_t src2, uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((op & 0x3FF) << 16) | (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((src2 & 0x1FF) << 18);
}

bool is_f32_nan(uint32_t bits) { return ((bits & 0x7FFFFFFFu) > 0x7F800000u); }
bool is_f64_nan(uint64_t bits) { return ((bits & 0x7FFFFFFFFFFFFFFFULL) > 0x7FF0000000000000ULL); }

// 16 mixed f32 values covering every IEEE class + values that exercise the
// div_fixup cascade boundaries (±0, ±Inf, NaN, normal).
const std::array<uint32_t, 16> kF32 = {{
    0x00000000u, // +0
    0x80000000u, // -0
    0x3F800000u, // +1
    0xBF800000u, // -1
    0x7F800000u, // +Inf
    0xFF800000u, // -Inf
    0x7FC00000u, // qNaN
    0xFFC00000u, // -qNaN
    0x40490FDBu, // pi
    0xC0490FDBu, // -pi
    0x42F60000u, // 123
    0xC2F60000u, // -123
    0x3DCCCCCDu, // 0.1
    0xBDCCCCCDu, // -0.1
    0x7F7FFFFFu, // +MAX
    0xFF7FFFFFu, // -MAX
}};

const std::array<uint64_t, 16> kF64 = {{
    0x0000000000000000ULL, // +0
    0x8000000000000000ULL, // -0
    0x3FF0000000000000ULL, // +1
    0xBFF0000000000000ULL, // -1
    0x7FF0000000000000ULL, // +Inf
    0xFFF0000000000000ULL, // -Inf
    0x7FF8000000000000ULL, // qNaN
    0xFFF8000000000000ULL, // -qNaN
    0x400921FB54442D18ULL, // pi
    0xC00921FB54442D18ULL, // -pi
    0x405EC00000000000ULL, // 123
    0xC05EC00000000000ULL, // -123
    0x3FB999999999999AULL, // 0.1
    0xBFB999999999999AULL, // -0.1
    0x7FEFFFFFFFFFFFFFULL, // +MAX
    0xFFEFFFFFFFFFFFFFULL, // -MAX
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3_div_mem"), l2("vop3_div_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_div", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_vgprs_f32(uint32_t rot0, uint32_t rot1, uint32_t rot2, uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      cu->write_vgpr(vb + 0, lane, kF32[(lane + rot0) % kF32.size()]);
      cu->write_vgpr(vb + 1, lane, kF32[(lane + rot1) % kF32.size()]);
      cu->write_vgpr(vb + 2, lane, kF32[(lane + rot2) % kF32.size()]);
      cu->write_vgpr(vb + kDstVgpr32, lane, DST_SENTINEL_32);
    }
    wf->set_exec(exec);
  }

  void seed_vgprs_f64(uint32_t rot0, uint32_t rot1, uint32_t rot2, uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint64_t s0 = kF64[(lane + rot0) % kF64.size()];
      uint64_t s1 = kF64[(lane + rot1) % kF64.size()];
      uint64_t s2 = kF64[(lane + rot2) % kF64.size()];
      cu->write_vgpr(vb + 0, lane, static_cast<uint32_t>(s0));
      cu->write_vgpr(vb + 1, lane, static_cast<uint32_t>(s0 >> 32));
      cu->write_vgpr(vb + 2, lane, static_cast<uint32_t>(s1));
      cu->write_vgpr(vb + 3, lane, static_cast<uint32_t>(s1 >> 32));
      cu->write_vgpr(vb + 4, lane, static_cast<uint32_t>(s2));
      cu->write_vgpr(vb + 5, lane, static_cast<uint32_t>(s2 >> 32));
      cu->write_vgpr(vb + kDstVgpr64 + 0, lane, static_cast<uint32_t>(DST_SENTINEL_64));
      cu->write_vgpr(vb + kDstVgpr64 + 1, lane, static_cast<uint32_t>(DST_SENTINEL_64 >> 32));
    }
    wf->set_exec(exec);
  }

  std::array<uint32_t, WF_SIZE> run32(Instruction *inst, bool force_scalar, uint32_t rot0,
                                      uint32_t rot1, uint32_t rot2, uint64_t exec, uint64_t vcc) {
    amdgpu::simd_force_scalar() = force_scalar;
    seed_vgprs_f32(rot0, rot1, rot2, exec);
    wf->set_vcc(vcc);
    cu->execute_instruction(inst, *wf);
    amdgpu::simd_force_scalar() = false;
    std::array<uint32_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vb + kDstVgpr32, lane);
    return out;
  }

  std::array<uint64_t, WF_SIZE> run64(Instruction *inst, bool force_scalar, uint32_t rot0,
                                      uint32_t rot1, uint32_t rot2, uint64_t exec, uint64_t vcc) {
    amdgpu::simd_force_scalar() = force_scalar;
    seed_vgprs_f64(rot0, rot1, rot2, exec);
    wf->set_vcc(vcc);
    cu->execute_instruction(inst, *wf);
    amdgpu::simd_force_scalar() = false;
    std::array<uint64_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint64_t lo = cu->read_vgpr(vb + kDstVgpr64 + 0, lane);
      uint64_t hi = cu->read_vgpr(vb + kDstVgpr64 + 1, lane);
      out[lane] = lo | (hi << 32);
    }
    return out;
  }
};

void check_div_fixup_f32(uint64_t exec) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  // v_div_fixup_f32 (478) plus the _f16 / _legacy_f16 twins (519 / 495): the
  // generated CDNA4 bodies for the f16 forms read/write raw f32 (bit_cast,
  // not f16_to_f32), so all three share the f32 div_fixup cascade and are
  // verified with the same f32 special-value sweep.
  struct Op {
    uint32_t opcode;
    const char *name;
  };
  for (const Op &o : {Op{478, "v_div_fixup_f32_vop3"}, Op{519, "v_div_fixup_f16_vop3"},
                      Op{495, "v_div_fixup_legacy_f16_vop3"}}) {
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_tern_encode(o.opcode, /*vdst=*/kDstVgpr32, /*src0=*/256, /*src1=*/257,
                     /*src2=*/258, words);
    Instruction *inst = fx.decoder->decode(words);
    ASSERT_NE(inst, nullptr) << o.name << " decode failed";
    // Sweep rotations so every (b,c) class pairing hits the cascade.
    for (uint32_t r1 = 0; r1 < kF32.size(); ++r1)
      for (uint32_t r2 = 0; r2 < kF32.size(); r2 += 3) {
        auto sc = fx.run32(inst, true, 0, r1, r2, exec, /*vcc=*/0);
        auto sd = fx.run32(inst, false, 0, r1, r2, exec, /*vcc=*/0);
        for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
          const bool active = (exec >> lane) & 1ULL;
          if (!active) {
            EXPECT_EQ(sd[lane], DST_SENTINEL_32) << o.name << " r1=" << r1 << " r2=" << r2
                                                 << ": SIMD clobbered inactive lane " << lane;
            continue;
          }
          if (is_f32_nan(sc[lane]) || is_f32_nan(sd[lane]))
            continue; // NaN payload may differ
          EXPECT_EQ(sc[lane], sd[lane])
              << o.name << " r1=" << r1 << " r2=" << r2 << " lane=" << lane << ": sc=0x" << std::hex
              << sc[lane] << " sd=0x" << sd[lane];
        }
      }
    delete inst;
  }
}

void check_div_fixup_f64(uint64_t exec) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  uint32_t words[4] = {0u, 0u, 0u, 0u};
  // f64 reads two consecutive VGPRs per operand: src0=v0:v1, src1=v2:v3,
  // src2=v4:v5; vdst = kDstVgpr64:kDstVgpr64+1.
  vop3_tern_encode(/*op=*/479, /*vdst=*/kDstVgpr64, /*src0=*/256, /*src1=*/258,
                   /*src2=*/260, words);
  Instruction *inst = fx.decoder->decode(words);
  ASSERT_NE(inst, nullptr) << "v_div_fixup_f64_vop3 decode failed";
  for (uint32_t r1 = 0; r1 < kF64.size(); ++r1)
    for (uint32_t r2 = 0; r2 < kF64.size(); r2 += 3) {
      auto sc = fx.run64(inst, true, 0, r1, r2, exec, /*vcc=*/0);
      auto sd = fx.run64(inst, false, 0, r1, r2, exec, /*vcc=*/0);
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        const bool active = (exec >> lane) & 1ULL;
        if (!active) {
          EXPECT_EQ(sd[lane], DST_SENTINEL_64) << "v_div_fixup_f64 r1=" << r1 << " r2=" << r2
                                               << ": SIMD clobbered inactive lane " << lane;
          continue;
        }
        if (is_f64_nan(sc[lane]) || is_f64_nan(sd[lane]))
          continue;
        EXPECT_EQ(sc[lane], sd[lane])
            << "v_div_fixup_f64 r1=" << r1 << " r2=" << r2 << " lane=" << lane << ": sc=0x"
            << std::hex << sc[lane] << " sd=0x" << sd[lane];
      }
    }
  delete inst;
}

void check_div_fmas_f32(uint64_t exec) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  uint32_t words[4] = {0u, 0u, 0u, 0u};
  vop3_tern_encode(/*op=*/482, /*vdst=*/kDstVgpr32, /*src0=*/256, /*src1=*/257,
                   /*src2=*/258, words);
  Instruction *inst = fx.decoder->decode(words);
  ASSERT_NE(inst, nullptr) << "v_div_fmas_f32_vop3 decode failed";
  for (uint64_t vcc : {uint64_t{0}, uint64_t{~0ULL}, uint64_t{0xAAAAAAAAAAAAAAAAULL},
                       uint64_t{0x5555555555555555ULL}}) {
    for (uint32_t r1 = 0; r1 < kF32.size(); r1 += 3) {
      auto sc = fx.run32(inst, true, 0, r1, 2 * r1, exec, vcc);
      auto sd = fx.run32(inst, false, 0, r1, 2 * r1, exec, vcc);
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        const bool active = (exec >> lane) & 1ULL;
        if (!active) {
          EXPECT_EQ(sd[lane], DST_SENTINEL_32)
              << "v_div_fmas_f32 vcc=0x" << std::hex << vcc << " r1=" << std::dec << r1
              << ": SIMD clobbered inactive lane " << lane;
          continue;
        }
        if (is_f32_nan(sc[lane]) || is_f32_nan(sd[lane]))
          continue;
        EXPECT_EQ(sc[lane], sd[lane])
            << "v_div_fmas_f32 vcc=0x" << std::hex << vcc << " r1=" << std::dec << r1
            << " lane=" << lane << ": sc=0x" << std::hex << sc[lane] << " sd=0x" << sd[lane];
      }
    }
  }
  delete inst;
}

void check_div_fmas_f64(uint64_t exec) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  uint32_t words[4] = {0u, 0u, 0u, 0u};
  vop3_tern_encode(/*op=*/483, /*vdst=*/kDstVgpr64, /*src0=*/256, /*src1=*/258,
                   /*src2=*/260, words);
  Instruction *inst = fx.decoder->decode(words);
  ASSERT_NE(inst, nullptr) << "v_div_fmas_f64_vop3 decode failed";
  for (uint64_t vcc : {uint64_t{0}, uint64_t{~0ULL}, uint64_t{0xAAAAAAAAAAAAAAAAULL},
                       uint64_t{0x5555555555555555ULL}}) {
    for (uint32_t r1 = 0; r1 < kF64.size(); r1 += 3) {
      auto sc = fx.run64(inst, true, 0, r1, 2 * r1, exec, vcc);
      auto sd = fx.run64(inst, false, 0, r1, 2 * r1, exec, vcc);
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        const bool active = (exec >> lane) & 1ULL;
        if (!active) {
          EXPECT_EQ(sd[lane], DST_SENTINEL_64)
              << "v_div_fmas_f64 vcc=0x" << std::hex << vcc << " r1=" << std::dec << r1
              << ": SIMD clobbered inactive lane " << lane;
          continue;
        }
        if (is_f64_nan(sc[lane]) || is_f64_nan(sd[lane]))
          continue;
        EXPECT_EQ(sc[lane], sd[lane])
            << "v_div_fmas_f64 vcc=0x" << std::hex << vcc << " r1=" << std::dec << r1
            << " lane=" << lane << ": sc=0x" << std::hex << sc[lane] << " sd=0x" << sd[lane];
      }
    }
  }
  delete inst;
}

TEST(Vop3DivHelpersSimdCorrectness, DivFixupF32_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_div_fixup_f32(/*exec=*/~0ULL);
}
TEST(Vop3DivHelpersSimdCorrectness, DivFixupF32_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_div_fixup_f32(/*exec=*/0xA5A5'F0F0'1234'8001ULL);
}
TEST(Vop3DivHelpersSimdCorrectness, DivFixupF64_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_div_fixup_f64(/*exec=*/~0ULL);
}
TEST(Vop3DivHelpersSimdCorrectness, DivFixupF64_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_div_fixup_f64(/*exec=*/0xA5A5'F0F0'1234'8001ULL);
}
TEST(Vop3DivHelpersSimdCorrectness, DivFmasF32_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_div_fmas_f32(/*exec=*/~0ULL);
}
TEST(Vop3DivHelpersSimdCorrectness, DivFmasF32_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_div_fmas_f32(/*exec=*/0xA5A5'F0F0'1234'8001ULL);
}
TEST(Vop3DivHelpersSimdCorrectness, DivFmasF64_FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_div_fmas_f64(/*exec=*/~0ULL);
}
TEST(Vop3DivHelpersSimdCorrectness, DivFmasF64_PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  check_div_fmas_f64(/*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
