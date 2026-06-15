// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_cvt_pk_f32_rdna_correctness_test.cpp
/// @brief Value-correctness check for v_cvt_pk_i16_f32 / v_cvt_pk_u16_f32, two
/// FLOAT-input packed converts that are RDNA3+ only (no CDNA4 decode path).
///
/// Regression guard for the SIMD fast-path bug fixed alongside this file: these
/// ops were briefly routed through the integer VOP3 binary glue
/// (try_execute_binary_vop3_simd<uint32_t>), which feeds the functor the raw
/// 32-bit lane *bits*. The functor then clamped the float's bit pattern as an
/// int32 to [-32768, 32767] — a wrong result for essentially every finite input
/// (e.g. 40000.0f packed to 32767 instead of 40000), and the u16 form silently
/// used the signed i16 range. They are now (correctly) left on the scalar path:
///   v_cvt_pk_i16_f32: lane = (uint16_t)(int16_t)clamp(f, -32768, 32767)
///   v_cvt_pk_u16_f32: lane = (uint16_t)clamp(f, 0, 65535)
/// with src0 -> low half, src1 -> high half of the dst.
///
/// This file also pins the deliberate codegen QUIRK in the *normalized* packs:
/// RDNA spells them v_cvt_pk_norm_{i16,u16}_f32 (underscore), and the underscore
/// i16 form is generated with the UNSIGNED u16 lambda — clamp(f * 65535, 0,
/// 65535) — despite "i16" in its name (see simd_codegen.py "QUIRK"). The
/// no-underscore signed spelling (v_cvt_pknorm_i16_f32) is CDNA-only and not
/// decodable here. The norm cases below lock the underscore form to u16 so a
/// future "the name says i16" change is caught on both the scalar and SIMD path.
///
/// Each case is executed twice in the same process (force-scalar and force-SIMD
/// gate) and BOTH runs are asserted equal to an independent golden reference, so
/// any future re-introduction of a mismatched SIMD functor is caught. The CU and
/// decoder are built for RDNA3; VOP3 encoding marker is 0x35<<26.

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

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 32; // RDNA3 wave32
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t kDstVgpr = 4;
constexpr uint32_t DST_SENTINEL = 0xCDCDCDCDu;

// RDNA3 VOP3 opcodes (from rdna3/test_encodings.h: 0xD7060000 / 0xD7070000).
constexpr uint32_t kOpCvtPkI16F32 = 774;
constexpr uint32_t kOpCvtPkU16F32 = 775;
// Normalized i16 pack (rdna3/test_encodings.h: 0xD7210000). Note the underscore
// spelling — RDNA exposes v_cvt_pk_norm_i16_f32, which despite its name is
// generated with the u16-normalized lambda (the QUIRK we pin below).
constexpr uint32_t kOpCvtPkNormI16F32 = 801; // 0xD7210000 >> 16 & 0x3FF

// RDNA3 VOP3 (encoding[31:26]=0x35). word0 = vdst[7:0] | op[25:16] | enc[31:26];
// word1 = src0[8:0] | src1[17:9] | src2[26:18]. No modifiers used here.
constexpr void rdna3_vop3_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1,
                                 uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((op & 0x3FF) << 16) | (0x35u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9);
}

// Finite inputs covering: in-range, fractional (truncation), the signed/unsigned
// boundaries, over/underflow, and negatives. Deliberately includes 40000.0f —
// the value that exposed the original bug (clamps to 40000 for u16, but the
// broken int-domain SIMD path produced 32767).
const std::array<float, 16> kInputs = {{
    0.0f,
    1.0f,
    100.5f,
    40000.0f,
    70000.0f,
    -5.0f,
    -40000.0f,
    32767.0f,
    32768.0f,
    65535.0f,
    65536.0f,
    12345.75f,
    255.5f,
    1000.0f,
    -1.0f,
    60000.0f,
}};

uint16_t cvt_i16(float f) {
  return static_cast<uint16_t>(static_cast<int16_t>(std::clamp(f, -32768.0f, 32767.0f)));
}
uint16_t cvt_u16(float f) { return static_cast<uint16_t>(std::clamp(f, 0.0f, 65535.0f)); }

// Golden: src0 (f0) -> low 16, src1 (f1) -> high 16.
uint32_t golden(bool is_u16, float f0, float f1) {
  uint16_t lo = is_u16 ? cvt_u16(f0) : cvt_i16(f0);
  uint16_t hi = is_u16 ? cvt_u16(f1) : cvt_i16(f1);
  return (static_cast<uint32_t>(hi) << 16) | static_cast<uint32_t>(lo);
}

// --- Normalized pack-convert (the underscore-spelling u16 quirk) ------------
// Both arch-generated lambdas scale by K, clamp, map NaN->0, then truncate
// toward zero (mirrors execute_shared.h / util::cvt_pknorm_*_f32_simd). The
// QUIRK: v_cvt_pk_norm_i16_f32 (underscore) uses the UNSIGNED form below.
uint16_t cvt_norm_u16(float f) {
  if (std::isnan(f))
    return 0;
  return static_cast<uint16_t>(std::clamp(f * 65535.0f, 0.0f, 65535.0f));
}
// What a (wrong) signed-i16 reading would produce — used only to show the two
// spellings diverge, never as the expected result for the underscore form.
uint16_t cvt_norm_i16_wrong(float f) {
  if (std::isnan(f))
    return 0;
  return static_cast<uint16_t>(static_cast<int16_t>(std::clamp(f * 32767.0f, -32768.0f, 32767.0f)));
}
uint32_t golden_norm_u16(float f0, float f1) {
  return (static_cast<uint32_t>(cvt_norm_u16(f1)) << 16) | static_cast<uint32_t>(cvt_norm_u16(f0));
}
// The value the underscore form must NOT produce (signed-i16 reading).
uint32_t golden_norm_i16_wrong(float f0, float f1) {
  return (static_cast<uint32_t>(cvt_norm_i16_wrong(f1)) << 16) |
         static_cast<uint32_t>(cvt_norm_i16_wrong(f0));
}

// Normalized-domain inputs: in-range, fractional/truncation, the i16/u16
// divergence points (±1.0), out-of-range (clamped), negatives (-> 0 under u16),
// and NaN (-> 0).
const std::array<float, 8> kNormInputs = {{
    0.0f,
    1.0f,
    0.5f,
    -1.0f,
    -0.5f,
    2.0f,
    0.25f,
    std::numeric_limits<float>::quiet_NaN(),
}};
float norm_f0_for(uint32_t lane) { return kNormInputs[lane % kNormInputs.size()]; }
float norm_f1_for(uint32_t lane) { return kNormInputs[(lane + 3) % kNormInputs.size()]; }

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("cvt_pk_f32_mem"), l2("cvt_pk_f32_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_RDNA3;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_cvt_pk_f32", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  float f0_for(uint32_t lane) const { return kInputs[lane % kInputs.size()]; }
  float f1_for(uint32_t lane) const { return kInputs[(lane + 7) % kInputs.size()]; }

  void seed(uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      cu->write_vgpr(vb + 0, lane, std::bit_cast<uint32_t>(f0_for(lane)));
      cu->write_vgpr(vb + 1, lane, std::bit_cast<uint32_t>(f1_for(lane)));
      cu->write_vgpr(vb + kDstVgpr, lane, DST_SENTINEL);
    }
    wf->set_exec(exec);
  }

  std::array<uint32_t, WF_SIZE> run(Instruction *inst, uint64_t exec) {
    seed(exec);
    cu->execute_instruction(inst, *wf);
    std::array<uint32_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vb + kDstVgpr, lane);
    return out;
  }
};

// Restores the process force-scalar gate on scope exit.
struct ForceScalarGuard {
  bool orig;
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }
};

void check_op(uint32_t opcode, bool is_u16, uint64_t exec) {
  ForceScalarGuard gate_guard;
  const char *name = is_u16 ? "v_cvt_pk_u16_f32" : "v_cvt_pk_i16_f32";

  auto run_mode = [&](bool force_scalar) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    rdna3_vop3_encode(opcode, kDstVgpr, /*src0=*/256, /*src1=*/257, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << name << " decode failed";
    auto out = fx.run(inst, exec);
    delete inst;
    return out;
  };

  const auto scalar_out = run_mode(/*force_scalar=*/true);
  const auto simd_out = run_mode(/*force_scalar=*/false);

  Fixture ref; // only used for the per-lane input mapping
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (exec >> lane) & 1ULL;
    if (!active) {
      EXPECT_EQ(scalar_out[lane], DST_SENTINEL) << name << ": clobbered inactive lane " << lane;
      EXPECT_EQ(simd_out[lane], DST_SENTINEL) << name << ": clobbered inactive lane " << lane;
      continue;
    }
    const uint32_t want = golden(is_u16, ref.f0_for(lane), ref.f1_for(lane));
    EXPECT_EQ(scalar_out[lane], want) << name << " (scalar) lane " << lane
                                      << " f0=" << ref.f0_for(lane) << " f1=" << ref.f1_for(lane);
    EXPECT_EQ(simd_out[lane], want) << name << " (simd) lane " << lane << " f0=" << ref.f0_for(lane)
                                    << " f1=" << ref.f1_for(lane);
  }
}

TEST(Vop3CvtPkF32RdnaCorrectness, I16_FullExec) {
  check_op(kOpCvtPkI16F32, /*is_u16=*/false, /*exec=*/0xFFFFFFFFULL);
}
TEST(Vop3CvtPkF32RdnaCorrectness, I16_PartialExec) {
  check_op(kOpCvtPkI16F32, /*is_u16=*/false, /*exec=*/0xA5A58001ULL);
}
TEST(Vop3CvtPkF32RdnaCorrectness, U16_FullExec) {
  check_op(kOpCvtPkU16F32, /*is_u16=*/true, /*exec=*/0xFFFFFFFFULL);
}
TEST(Vop3CvtPkF32RdnaCorrectness, U16_PartialExec) {
  check_op(kOpCvtPkU16F32, /*is_u16=*/true, /*exec=*/0xA5A58001ULL);
}

// Pin the exact value that exposed the bug: 40000.0f converts to 40000 for u16
// (clamp [0, 65535]) but to 32767 for i16 (clamp [-32768, 32767]). The broken
// SIMD path produced 32767 for BOTH; assert they now differ as expected.
TEST(Vop3CvtPkF32RdnaCorrectness, BugMarker_40000) {
  ForceScalarGuard gate_guard;
  for (bool force_scalar : {true, false}) {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    ASSERT_NE(fx.cu, nullptr);
    uint32_t vb = fx.wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      fx.cu->write_vgpr(vb + 0, lane, std::bit_cast<uint32_t>(40000.0f));
      fx.cu->write_vgpr(vb + 1, lane, std::bit_cast<uint32_t>(40000.0f));
      fx.cu->write_vgpr(vb + kDstVgpr, lane, DST_SENTINEL);
    }
    fx.wf->set_exec(0xFFFFFFFFULL);

    auto exec_op = [&](uint32_t opcode) -> uint32_t {
      uint32_t words[4] = {0u, 0u, 0u, 0u};
      rdna3_vop3_encode(opcode, kDstVgpr, 256, 257, words);
      Instruction *inst = fx.decoder->decode(words);
      EXPECT_NE(inst, nullptr);
      fx.cu->execute_instruction(inst, *fx.wf);
      uint32_t r = fx.cu->read_vgpr(vb + kDstVgpr, 0);
      delete inst;
      return r;
    };

    // u16: 40000 in both halves -> 0x9C40_9C40.
    EXPECT_EQ(exec_op(kOpCvtPkU16F32), 0x9C409C40u)
        << "force_scalar=" << force_scalar << ": u16 must clamp to [0, 65535]";
    // i16: 40000 saturates to 32767 (0x7FFF) in both halves -> 0x7FFF_7FFF.
    EXPECT_EQ(exec_op(kOpCvtPkI16F32), 0x7FFF7FFFu)
        << "force_scalar=" << force_scalar << ": i16 must clamp to [-32768, 32767]";
  }
}

// Drives v_cvt_pk_norm_i16_f32 in both gate modes against the UNSIGNED golden.
// Seeds the norm input set directly (Fixture::seed feeds the non-norm kInputs).
void check_cvt_pk_norm_i16(uint64_t exec) {
  ForceScalarGuard gate_guard;
  const char *name = "v_cvt_pk_norm_i16_f32";

  auto run_mode = [&](bool force_scalar) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t vb = fx.wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      fx.cu->write_vgpr(vb + 0, lane, std::bit_cast<uint32_t>(norm_f0_for(lane)));
      fx.cu->write_vgpr(vb + 1, lane, std::bit_cast<uint32_t>(norm_f1_for(lane)));
      fx.cu->write_vgpr(vb + kDstVgpr, lane, DST_SENTINEL);
    }
    fx.wf->set_exec(exec);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    rdna3_vop3_encode(kOpCvtPkNormI16F32, kDstVgpr, /*src0=*/256, /*src1=*/257, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << name << " decode failed";
    fx.cu->execute_instruction(inst, *fx.wf);
    std::array<uint32_t, WF_SIZE> out{};
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = fx.cu->read_vgpr(vb + kDstVgpr, lane);
    delete inst;
    return out;
  };

  const auto scalar_out = run_mode(/*force_scalar=*/true);
  const auto simd_out = run_mode(/*force_scalar=*/false);

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (exec >> lane) & 1ULL;
    if (!active) {
      EXPECT_EQ(scalar_out[lane], DST_SENTINEL) << name << ": clobbered inactive lane " << lane;
      EXPECT_EQ(simd_out[lane], DST_SENTINEL) << name << ": clobbered inactive lane " << lane;
      continue;
    }
    const uint32_t want = golden_norm_u16(norm_f0_for(lane), norm_f1_for(lane));
    EXPECT_EQ(scalar_out[lane], want) << name << " (scalar) lane " << lane;
    EXPECT_EQ(simd_out[lane], want) << name << " (simd) lane " << lane;
  }
}

TEST(Vop3CvtPkF32RdnaCorrectness, NormI16_UsesUnsignedU16_FullExec) {
  check_cvt_pk_norm_i16(/*exec=*/0xFFFFFFFFULL);
}
TEST(Vop3CvtPkF32RdnaCorrectness, NormI16_UsesUnsignedU16_PartialExec) {
  check_cvt_pk_norm_i16(/*exec=*/0xA5A58001ULL);
}

// Pin the quirk at the values where signed-i16 and unsigned-u16 normalization
// diverge: +1.0 -> 0xFFFF under u16 (vs 0x7FFF under i16); -1.0 -> 0x0000 under
// u16 (vs 0x8001 under i16). The underscore spelling MUST produce the u16
// result. If someone "corrects" the codegen to signed i16, these flip and fail.
TEST(Vop3CvtPkF32RdnaCorrectness, NormI16_QuirkMarker) {
  ForceScalarGuard gate_guard;
  for (bool force_scalar : {true, false}) {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    ASSERT_NE(fx.cu, nullptr);
    uint32_t vb = fx.wf->vgpr_alloc().base;
    // lane 0: (+1.0, +1.0) -> both halves 0xFFFF; lane 1: (-1.0, -1.0) -> 0.
    fx.cu->write_vgpr(vb + 0, 0, std::bit_cast<uint32_t>(1.0f));
    fx.cu->write_vgpr(vb + 1, 0, std::bit_cast<uint32_t>(1.0f));
    fx.cu->write_vgpr(vb + 0, 1, std::bit_cast<uint32_t>(-1.0f));
    fx.cu->write_vgpr(vb + 1, 1, std::bit_cast<uint32_t>(-1.0f));
    fx.cu->write_vgpr(vb + kDstVgpr, 0, DST_SENTINEL);
    fx.cu->write_vgpr(vb + kDstVgpr, 1, DST_SENTINEL);
    fx.wf->set_exec(0x3ULL);

    uint32_t words[4] = {0u, 0u, 0u, 0u};
    rdna3_vop3_encode(kOpCvtPkNormI16F32, kDstVgpr, 256, 257, words);
    Instruction *inst = fx.decoder->decode(words);
    ASSERT_NE(inst, nullptr);
    fx.cu->execute_instruction(inst, *fx.wf);
    const uint32_t lane0 = fx.cu->read_vgpr(vb + kDstVgpr, 0);
    const uint32_t lane1 = fx.cu->read_vgpr(vb + kDstVgpr, 1);
    delete inst;

    EXPECT_EQ(lane0, 0xFFFFFFFFu)
        << "force_scalar=" << force_scalar
        << ": v_cvt_pk_norm_i16_f32 must use UNSIGNED u16 (+1.0 -> 0xFFFF, not 0x7FFF)";
    EXPECT_EQ(lane1, 0x00000000u) << "force_scalar=" << force_scalar
                                  << ": negatives clamp to 0 under u16 normalization";
    // Sanity: the signed-i16 reading we are guarding against differs at both lanes.
    EXPECT_NE(lane0, golden_norm_i16_wrong(1.0f, 1.0f))
        << "i16 reading leaked into the underscore spelling";
    EXPECT_NE(lane1, golden_norm_i16_wrong(-1.0f, -1.0f))
        << "i16 reading leaked into the underscore spelling";
  }
}

} // namespace
