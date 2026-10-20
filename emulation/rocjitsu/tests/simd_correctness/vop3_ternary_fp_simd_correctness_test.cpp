// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_ternary_fp_simd_correctness_test.cpp
/// @brief Compare ternary VOP3 results with SIMD enabled and forced scalar.
/// @details General F16/F32/F64 cases use CDNA4. F16 output-order boundaries
/// also cover RDNA2/3, including signed underflow and MODE-dependent overflow.
/// These are execution-path regressions, not hardware captures.
/// Each pass uses identical inputs/EXEC; inactive lanes retain the sentinel.
/// General cases skip NaN results because older paths may select different payloads.
/// SIMD-enabled execution may fall back to scalar when a fast-path gate rejects it.
/// Accumulate forms have separate FMAC and GFX1250 execution-policy coverage.

#include "decode_test_util.h"
#include "util/simd_test_hooks.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/execute_shared.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_path_test_hooks.h"
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
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t kDstVgpr32 = 6;
constexpr uint32_t kDstVgpr64 = 6; // v6:v7
constexpr uint32_t DST_SENTINEL32 = 0xCDCDCDCDu;

constexpr void vop3_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1, uint32_t src2,
                           uint32_t abs, uint32_t neg, uint32_t omod, uint32_t clamp,
                           uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((abs & 0x7) << 8) | ((clamp & 0x1) << 15) | ((op & 0x3FF) << 16) |
             (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((src2 & 0x1FF) << 18) |
             ((omod & 0x3) << 27) | ((neg & 0x7) << 29);
}

bool is_f32_nan(uint32_t bits) {
  return ((bits >> 23) & 0xFFu) == 0xFFu && (bits & 0x7FFFFFu) != 0;
}
bool is_f16_nan(uint32_t bits) {
  uint16_t h = static_cast<uint16_t>(bits);
  return ((h >> 10) & 0x1Fu) == 0x1Fu && (h & 0x3FFu) != 0;
}
bool is_f64_nan(uint64_t bits) {
  return ((bits >> 52) & 0x7FFu) == 0x7FFu && (bits & 0xFFFFFFFFFFFFFull) != 0;
}

enum class Kind { F32, F16, F64 };

struct Case {
  const char *name;
  uint32_t opcode;
  Kind kind;
  std::optional<amdgpu::SimdFastPath> expected_simd_path = std::nullopt;
};

const std::array<Case, 10> kCases = {{
    {"v_fma_f32_vop3", 459, Kind::F32},
    {"v_fma_f16_vop3", 518, Kind::F16, amdgpu::SimdFastPath::VOP3_FMA_FP16},
    {"v_fma_f64_vop3", 460, Kind::F64},
    {"v_mad_f16_vop3", 515, Kind::F16, amdgpu::SimdFastPath::VOP3_TERNARY_FP16},
    // min3/max3/med3: fmax/fmin compositions. Inputs are finite non-zero
    // normals, so the fmax/fmin NaN-payload / signed-zero-tie carve-out never
    // triggers — bit-exact vs scalar on every lane.
    {"v_max3_f32_vop3", 467, Kind::F32},
    {"v_min3_f32_vop3", 464, Kind::F32},
    {"v_med3_f32_vop3", 470, Kind::F32},
    {"v_max3_f16_vop3", 503, Kind::F16, amdgpu::SimdFastPath::VOP3_TERNARY_FP16},
    {"v_min3_f16_vop3", 500, Kind::F16, amdgpu::SimdFastPath::VOP3_TERNARY_FP16},
    {"v_med3_f16_vop3", 506, Kind::F16, amdgpu::SimdFastPath::VOP3_TERNARY_FP16},
}};

// Finite-normal f32 sanitized inputs (avoid NaN/Inf for cleaner FMA bit-equality
// checks; NaN lanes would be skipped anyway).
const std::array<uint32_t, 16> kF32 = {{
    0x3F800000u,
    0xBF800000u,
    0x40000000u,
    0xC0000000u,
    0x3FC00000u,
    0xBFC00000u,
    0x40400000u,
    0x40800000u,
    0x3F000000u,
    0xBF000000u,
    0x3E800000u,
    0xC0900000u,
    0x40000000u,
    0x40C00000u,
    0x41000000u,
    0xC1000000u,
}};
// f16 sanitized finite normals in low 16 (high 16 = sentinel).
const std::array<uint32_t, 16> kF16 = {{
    0xDEAD3C00u,
    0xDEADBC00u,
    0xDEAD4000u,
    0xDEADC000u,
    0xDEAD4200u,
    0xDEADC200u,
    0xDEAD4400u,
    0xDEAD4500u,
    0xDEAD3800u,
    0xDEADB800u,
    0xDEAD3400u,
    0xDEADC480u,
    0xDEAD4000u,
    0xDEAD4600u,
    0xDEAD4800u,
    0xDEADC800u,
}};
// f64 sanitized finite normals.
const std::array<uint64_t, 16> kF64 = {{
    0x3FF0000000000000ull,
    0xBFF0000000000000ull,
    0x4000000000000000ull,
    0xC000000000000000ull,
    0x3FF8000000000000ull,
    0xBFF8000000000000ull,
    0x4008000000000000ull,
    0x4010000000000000ull,
    0x3FE0000000000000ull,
    0xBFE0000000000000ull,
    0x3FD0000000000000ull,
    0xC012000000000000ull,
    0x4000000000000000ull,
    0x4018000000000000ull,
    0x4020000000000000ull,
    0xC020000000000000ull,
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  explicit Fixture(rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA4)
      : gpu_mem("vop3_tern_fp_mem"), l2("vop3_tern_fp_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_tern_fp", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(arch);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void write64(uint32_t reg, uint32_t lane, uint64_t v) {
    cu->write_vgpr(reg, lane, static_cast<uint32_t>(v));
    cu->write_vgpr(reg + 1, lane, static_cast<uint32_t>(v >> 32));
  }

  // f32/f16: src0=v0, src1=v1, src2=v2, dst=v6. f64: src0=v0:v1, src1=v2:v3,
  // src2=v4:v5, dst=v6:v7.
  void seed_inputs(Kind k, uint32_t rot, uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      if (k == Kind::F64) {
        write64(vb + 0, lane, kF64[lane % kF64.size()]);
        write64(vb + 2, lane, kF64[(lane + rot) % kF64.size()]);
        write64(vb + 4, lane, kF64[(lane + 2 * rot) % kF64.size()]);
        write64(vb + kDstVgpr64, lane, 0xCDCDCDCDCDCDCDCDull);
      } else {
        const auto &v = (k == Kind::F32) ? kF32 : kF16;
        cu->write_vgpr(vb + 0, lane, v[lane % v.size()]);
        cu->write_vgpr(vb + 1, lane, v[(lane + rot) % v.size()]);
        cu->write_vgpr(vb + 2, lane, v[(lane + 2 * rot) % v.size()]);
        cu->write_vgpr(vb + kDstVgpr32, lane, DST_SENTINEL32);
      }
    }
    wf->set_exec(exec);
  }

  std::array<uint64_t, WF_SIZE> run(Instruction *inst, Kind k, uint32_t rot, uint64_t exec) {
    seed_inputs(k, rot, exec);
    EXPECT_TRUE(cu->execute_instruction(inst, *wf).succeeded());
    std::array<uint64_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      if (k == Kind::F64)
        out[lane] = static_cast<uint64_t>(cu->read_vgpr(vb + kDstVgpr64 + 1, lane)) << 32 |
                    cu->read_vgpr(vb + kDstVgpr64, lane);
      else
        out[lane] = cu->read_vgpr(vb + kDstVgpr32, lane);
    }
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

bool result_is_nan(Kind k, uint64_t v) {
  if (k == Kind::F32)
    return is_f32_nan(static_cast<uint32_t>(v));
  if (k == Kind::F16)
    return is_f16_nan(static_cast<uint32_t>(v));
  return is_f64_nan(v);
}

void check_case(const Case &c, uint32_t abs, uint32_t neg, uint32_t omod, uint32_t clamp,
                uint64_t exec) {
  ForceScalarGuard gate_guard;

  auto run_mode = [&](bool force_scalar, uint32_t rot) -> std::array<uint64_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    const uint32_t vdst = (c.kind == Kind::F64) ? kDstVgpr64 : kDstVgpr32;
    const uint32_t src1_v = (c.kind == Kind::F64) ? 2u : 1u;
    const uint32_t src2_v = (c.kind == Kind::F64) ? 4u : 2u;
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_encode(c.opcode, vdst, /*src0=*/256, /*src1=*/256 + src1_v, /*src2=*/256 + src2_v, abs,
                neg, omod, clamp, words);
    Instruction *inst = decode_valid(*fx.decoder, words);
    EXPECT_NE(inst, nullptr) << c.name << " decode failed";
    amdgpu::ScopedSimdFastPathTracker tracker;
    auto out = fx.run(inst, c.kind, rot, exec);
    if (c.expected_simd_path.has_value()) {
      if (force_scalar)
        EXPECT_TRUE(tracker.no_tracked_path_executed())
            << c.name << ": forced-scalar execution used SIMD";
      else
        EXPECT_TRUE(tracker.only_tracked_path_executed(*c.expected_simd_path))
            << c.name << ": eligible execution did not use its expected SIMD path";
    }
    delete inst;
    return out;
  };

  const std::size_t rot_max = (c.kind == Kind::F64) ? kF64.size() : kF32.size();
  for (uint32_t rot = 0; rot < rot_max; ++rot) {
    const auto scalar_out = run_mode(/*force_scalar=*/true, rot);
    const auto simd_out = run_mode(/*force_scalar=*/false, rot);

    // Core A/B equivalence per active, non-skipped lane. NaN-result lanes carry
    // an accepted payload divergence and are excluded identically in both runs.
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (active &&
          (result_is_nan(c.kind, scalar_out[lane]) || result_is_nan(c.kind, simd_out[lane])))
        continue;
      EXPECT_EQ(scalar_out[lane], simd_out[lane])
          << c.name << " a" << abs << "n" << neg << "o" << omod << "c" << clamp << "r" << rot
          << " lane " << lane << ": SIMD path diverged from scalar body";
    }
  }
}

// A representative subset of the 8x8x4x2 = 512 modifier grid. The per-source
// abs/neg masks select independent branches (`if (abs & (1u << SrcIdx))`), so
// bit 0 vs bit 1 vs bit 2 exercise the same code path -- a handful of masks
// covers every distinct path. These 15 combos cover: no modifiers, each
// single-source abs, each single-source neg, all-sources abs, all-sources neg,
// abs+neg combined, each omod value, clamp, and a mixed case. ~1.3s vs the full
// grid's 163,840 fixtures (timed out at 15s).
void check_representative_mods(const Case &c, uint64_t exec) {
  struct ModCombo {
    uint32_t abs, neg, omod, clamp;
  };
  static constexpr ModCombo kCombos[] = {
      {0, 0, 0, 0}, {1, 0, 0, 0}, {2, 0, 0, 0}, {4, 0, 0, 0}, {7, 0, 0, 0},
      {0, 1, 0, 0}, {0, 2, 0, 0}, {0, 4, 0, 0}, {0, 7, 0, 0}, {7, 7, 0, 0},
      {0, 0, 1, 0}, {0, 0, 2, 0}, {0, 0, 3, 0}, {0, 0, 0, 1}, {3, 5, 1, 1},
  };
  for (const auto &m : kCombos)
    check_case(c, m.abs, m.neg, m.omod, m.clamp, exec);
}

TEST(Vop3TernaryFpSimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_representative_mods(c, /*exec=*/~0ULL);
}

TEST(Vop3TernaryFpSimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*abs=*/0, /*neg=*/0, /*omod=*/0, /*clamp=*/0, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

struct F16OutputCase {
  std::string name;
  rj_code_arch_t arch;
  uint16_t opcode;
  std::array<uint16_t, 3> sources;
  uint32_t mode;
  uint8_t omod;
  uint16_t expected;
  uint8_t clamp = 0;
};

// These cases separate destination-format rounding from OMOD/CLAMP ordering.
// Unless a case comment says otherwise, they pin the emulator contract rather
// than a hardware capture.
std::vector<F16OutputCase> f16_output_cases() {
  std::vector<F16OutputCase> cases;
  for (const auto &[name, opcode] :
       {std::pair{"Mad", cdna4::kVMadF16Vop3}, std::pair{"MadLegacy", cdna4::kVMadLegacyF16Vop3}}) {
    cases.push_back({std::string("Gfx950") + name + "Div2SignedUnderflow",
                     ROCJITSU_CODE_ARCH_CDNA4,
                     opcode,
                     {0x8400, 0x3c00, 0},
                     0,
                     3,
                     0x8000});
    cases.push_back({std::string("Gfx950") + name + "Mul2RtzOverflow",
                     ROCJITSU_CODE_ARCH_CDNA4,
                     opcode,
                     {0x7bff, 0x3c00, 0},
                     0x0c,
                     1,
                     0x7bff});
    cases.push_back({std::string("Gfx950") + name + "RoundsBeforeDiv2",
                     ROCJITSU_CODE_ARCH_CDNA4,
                     opcode,
                     {0x7bff, 0x4000, 0},
                     0,
                     3,
                     0x7c00});
    // MAD_F16 currently ignores MODE input/output denormal controls and
    // non-nearest rounding. These cases expose that separate semantic gap
    // while requiring the scalar and SIMD paths to retain the same contract.
    cases.push_back({std::string("Gfx950") + name + "CurrentFlushInputModeKeepsSubnormal",
                     ROCJITSU_CODE_ARCH_CDNA4,
                     opcode,
                     {0x0001, 0x3c00, 0},
                     0x00,
                     0,
                     0x0001});
    cases.push_back({std::string("Gfx950") + name + "KeepsSubnormalInput",
                     ROCJITSU_CODE_ARCH_CDNA4,
                     opcode,
                     {0x0001, 0x3c00, 0},
                     0xc0,
                     0,
                     0x0001});
    cases.push_back({std::string("Gfx950") + name + "CurrentFlushOutputModeKeepsTinyResult",
                     ROCJITSU_CODE_ARCH_CDNA4,
                     opcode,
                     {0x0400, 0x3800, 0},
                     0x40,
                     0,
                     0x0200});
    cases.push_back({std::string("Gfx950") + name + "KeepsTinyOutput",
                     ROCJITSU_CODE_ARCH_CDNA4,
                     opcode,
                     {0x0400, 0x3800, 0},
                     0xc0,
                     0,
                     0x0200});
    cases.push_back({std::string("Gfx950") + name + "CurrentRoundUpContract",
                     ROCJITSU_CODE_ARCH_CDNA4,
                     opcode,
                     {0x3c00, 0x3c00, 0x1000},
                     0xc4,
                     0,
                     0x3c00});
  }
  const auto add_selections = [&](const char *target, rj_code_arch_t arch, auto operations) {
    for (const auto &[name, opcode] : operations) {
      const auto prefix = std::string(target) + name;
      cases.push_back(
          {prefix + "Div2SignedUnderflow", arch, opcode, {0x8400, 0x8400, 0x8400}, 0, 3, 0x8000});
      cases.push_back(
          {prefix + "Mul2RtzOverflow", arch, opcode, {0x7bff, 0x7bff, 0x7bff}, 0x0c, 1, 0x7bff});
      cases.push_back(
          {prefix + "Mul4RneOverflow", arch, opcode, {0x7bff, 0x7bff, 0x7bff}, 0, 2, 0x7c00});
      cases.push_back(
          {prefix + "Mul2Saturates", arch, opcode, {0x7bff, 0x7bff, 0x7bff}, 1u << 23, 1, 0x7bff});
      cases.push_back(
          {prefix + "ClampAfterOmod", arch, opcode, {0x7bff, 0x7bff, 0x7bff}, 0x0c, 1, 0x3c00, 1});
      cases.push_back(
          {prefix + "IeeeIgnoresOmod", arch, opcode, {0x8400, 0x8400, 0x8400}, 1u << 9, 3, 0x8400});
      cases.push_back({prefix + "CurrentPolicyKeepOutputsIgnoresOmod",
                       arch,
                       opcode,
                       {0x8400, 0x8400, 0x8400},
                       0x80,
                       3,
                       0x8400});
    }
  };
  // MODE 0x80 cases exercise the emulator's old-target output-denormal gate.
  // The available gfx1030/gfx1100 captures keep output denormals disabled, so
  // they do not measure this policy.
  add_selections("Gfx950", ROCJITSU_CODE_ARCH_CDNA4,
                 std::array{std::pair{"Min3", cdna4::kVMin3F16Vop3},
                            std::pair{"Max3", cdna4::kVMax3F16Vop3},
                            std::pair{"Med3", cdna4::kVMed3F16Vop3}});
  add_selections("Gfx1030", ROCJITSU_CODE_ARCH_RDNA2,
                 std::array{std::pair{"Min3", rdna2::kVMin3F16Vop3},
                            std::pair{"Max3", rdna2::kVMax3F16Vop3},
                            std::pair{"Med3", rdna2::kVMed3F16Vop3}});
  add_selections("Gfx1100", ROCJITSU_CODE_ARCH_RDNA3,
                 std::array{std::pair{"Min3", rdna3::kVMin3F16Vop3},
                            std::pair{"Max3", rdna3::kVMax3F16Vop3},
                            std::pair{"Med3", rdna3::kVMed3F16Vop3},
                            std::pair{"Minmax", rdna3::kVMinmaxF16Vop3},
                            std::pair{"Maxmin", rdna3::kVMaxminF16Vop3}});
  // DIV_FIXUP still uses the pre-migration scalar ordering on CDNA4. gfx950 is
  // unmeasured; gfx1201 captures on nearby boundary vectors instead round to
  // F16 before applying OMOD. These cases pin the current emulator behavior,
  // not an architectural result, until the measured DIV_FIXUP fix is stacked.
  for (const auto &[name, opcode] : {std::pair{"DivFixup", cdna4::kVDivFixupF16Vop3},
                                     std::pair{"DivFixupLegacy", cdna4::kVDivFixupLegacyF16Vop3}}) {
    cases.push_back({std::string("Gfx950") + name + "PinsCurrentDiv2UnderflowOrdering",
                     ROCJITSU_CODE_ARCH_CDNA4,
                     opcode,
                     {0x8400, 0x3c00, 0xbc00},
                     0,
                     3,
                     0});
    cases.push_back({std::string("Gfx950") + name + "PinsCurrentMul2RtzOrdering",
                     ROCJITSU_CODE_ARCH_CDNA4,
                     opcode,
                     {0x7bff, 0x3c00, 0x3c00},
                     0x0c,
                     1,
                     0x7c00});
  }
  return cases;
}

std::array<uint32_t, 2> f16_output_words(const F16OutputCase &test, uint8_t opsel) {
  const auto build = [&](auto builder) {
    return builder(test.opcode, {.vdst = kDstVgpr32,
                                 .op_sel = opsel,
                                 .clamp = test.clamp,
                                 .src0 = 256,
                                 .src1 = 257,
                                 .src2 = 258,
                                 .omod = test.omod});
  };
  switch (test.arch) {
  case ROCJITSU_CODE_ARCH_CDNA4:
    return build(cdna4::build_vop3);
  case ROCJITSU_CODE_ARCH_RDNA2:
    return build(rdna2::build_vop3);
  case ROCJITSU_CODE_ARCH_RDNA3:
    return build(rdna3::build_vop3);
  default:
    ADD_FAILURE() << "unexpected ternary boundary target";
    return {};
  }
}

class Vop3F16TernaryOutputOrderTest : public testing::TestWithParam<F16OutputCase> {};

TEST_P(Vop3F16TernaryOutputOrderTest, MatchesScalarOutputContractWithSimdEnabledAndForcedScalar) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  ForceScalarGuard guard;
  amdgpu::fp_mode::ScopedEnvironment environment(0);
  const auto &test = GetParam();
  // RDNA2 uses the generic low-half executor; CDNA4/RDNA3 use true16 selection.
  const uint8_t selection_count = test.arch == ROCJITSU_CODE_ARCH_RDNA2 ? 1 : 2;
  for (uint8_t selection = 0; selection < selection_count; ++selection)
    for (const uint64_t exec : {~0ULL, 0xA5A5'F0F0'1234'8001ULL}) {
      const uint8_t opsel = selection ? 15 : 0;
      std::array<uint32_t, WF_SIZE> scalar_out{};
      for (const bool force_scalar : {true, false}) {
        SCOPED_TRACE(test.name + " opsel=" + std::to_string(opsel) +
                     (force_scalar ? " forced scalar" : " SIMD enabled"));
        util::set_force_scalar_for_testing(force_scalar);
        Fixture fx(test.arch);
        ASSERT_NE(fx.cu, nullptr);
        ASSERT_NE(fx.wf, nullptr);
        auto words = f16_output_words(test, opsel);
        std::unique_ptr<Instruction> inst(decode_valid(*fx.decoder, words.data()));
        ASSERT_NE(inst, nullptr);
        const uint32_t vb = fx.wf->vgpr_alloc().base;
        const bool high = (opsel & 8u) != 0;
        for (uint32_t lane = 0; lane < fx.wf->wf_size(); ++lane) {
          for (uint32_t src = 0; src < test.sources.size(); ++src) {
            const uint32_t half = test.sources[src];
            fx.cu->write_vgpr(vb + src, lane, high ? (half << 16) | 0x3555u : 0x35550000u | half);
          }
          fx.cu->write_vgpr(vb + kDstVgpr32, lane, DST_SENTINEL32);
        }
        fx.wf->set_exec(exec);
        fx.wf->set_mode_raw(test.mode);
        amdgpu::ScopedSimdFastPathTracker tracker;
        ASSERT_TRUE(fx.cu->execute_instruction(inst.get(), *fx.wf).succeeded());
        if (force_scalar)
          EXPECT_TRUE(tracker.no_tracked_path_executed())
              << test.name << ": forced-scalar execution used SIMD";
        else
          EXPECT_TRUE(tracker.only_tracked_path_executed(amdgpu::SimdFastPath::VOP3_TERNARY_FP16))
              << test.name << ": eligible execution did not use the ternary F16 SIMD path";
        for (uint32_t lane = 0; lane < fx.wf->wf_size(); ++lane) {
          const uint32_t actual = fx.cu->read_vgpr(vb + kDstVgpr32, lane);
          if (!(exec & (1ULL << lane))) {
            EXPECT_EQ(actual, DST_SENTINEL32) << "inactive lane " << lane;
          } else {
            EXPECT_EQ(static_cast<uint16_t>(actual >> (high ? 16 : 0)), test.expected)
                << "active lane " << lane;
            if (high) {
              EXPECT_EQ(static_cast<uint16_t>(actual), static_cast<uint16_t>(DST_SENTINEL32));
            } else if (test.arch == ROCJITSU_CODE_ARCH_RDNA3) {
              EXPECT_EQ(actual >> 16, DST_SENTINEL32 >> 16);
            }
          }
          if (force_scalar)
            scalar_out[lane] = actual;
          else
            EXPECT_EQ(actual, scalar_out[lane]) << "path disagreement in lane " << lane;
        }
      }
    }
}

INSTANTIATE_TEST_SUITE_P(OutputOrderBoundaries, Vop3F16TernaryOutputOrderTest,
                         testing::ValuesIn(f16_output_cases()),
                         [](const testing::TestParamInfo<F16OutputCase> &info) {
                           return info.param.name;
                         });

} // namespace
