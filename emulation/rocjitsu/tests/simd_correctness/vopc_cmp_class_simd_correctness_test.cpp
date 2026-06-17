// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vopc_cmp_class_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the
/// v_cmp_class VOPC ops on CDNA4. v_cmp_class tests src0's IEEE-754 float class
/// against a 10-bit class mask in vsrc1 and writes one VCC bit per active lane;
/// it is not a relational compare, so it uses a class-decode functor over the
/// existing VOPC glue. Each case runs TWICE in the same process -- once forcing
/// the scalar body, once the SIMD fast path, with identical inputs/EXEC/VCC-in
/// -- and the full 64-bit VCC results are asserted equal with EXPECT_EQ
/// (util::set_force_scalar_for_testing flips the gate in-process). In-process
/// inactive-lane VCC bits must be zeroed under full and partial EXEC.
///
/// f16 and f32 read src0 as 32-bit raw bits; f64 reads src0 as a 64-bit VGPR pair
/// while vsrc1 stays a 32-bit mask (the mixed-width class glue). The
/// classification is pure bit decode (matching the scalar isnan/isnormal/signbit
/// outcomes for every input incl. NaN payload), so the compare is bit-exact with
/// no carve-out.

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
#include <string>
#include <vector>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;

// VOPC: encoding[31:25]=0x3E, op[24:17], vsrc1[16:9], src0[8:0].
constexpr uint32_t vopc_encode(uint32_t op, uint32_t src0, uint32_t vsrc1) {
  return (0x3Eu << 25) | ((op & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF);
}

// VOP3 (encoding[31:26]=0x34): word0 = vdst[7:0] | abs[10:8] | op_sel[14:11] |
// clamp[15] | op[25:16] | enc[31:26]; word1 = src0[8:0] | src1[17:9] | src2[26:18]
// | omod[28:27] | neg[31:29]. abs/neg bit 0 select the src0 modifier. The class
// result is written to the SGPR-pair dst — we point it at VCC (106) so the same
// harness (set_vcc / read wf.vcc()) checks it.
constexpr uint32_t kVccSdst = 106;
void vop3_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1, uint32_t abs,
                 uint32_t neg, uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((abs & 0x7) << 8) | ((op & 0x3FF) << 16) | (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((neg & 0x7) << 29);
}

// One representative of each IEEE class (and a few extra normals). The negatives
// of NaN confirm the sign bit does not affect the sNaN/qNaN classification.
const std::array<uint32_t, 14> kF32 = {{
    0x7FA00000u, // sNaN (+)
    0x7FC00000u, // qNaN (+)
    0xFF800000u, // -Inf
    0xBF800000u, // -normal (-1)
    0x80000001u, // -denormal
    0x80000000u, // -0
    0x00000000u, // +0
    0x00000001u, // +denormal
    0x3F800000u, // +normal (1)
    0x7F800000u, // +Inf
    0x40490FDBu, // +normal (pi)
    0x7F7FFFFFu, // +normal (max)
    0xFFA00000u, // sNaN (-)
    0xFFC00000u, // qNaN (-)
}};
// Same classes as f16 in the low 16 bits, with garbage in the high half to
// confirm both modes ignore it.
const std::array<uint32_t, 14> kF16 = {{
    0xDEAD7D00u, // sNaN (+)
    0xDEAD7E00u, // qNaN (+)
    0xDEADFC00u, // -Inf
    0xDEADBC00u, // -normal (-1)
    0xDEAD8001u, // -denormal
    0xDEAD8000u, // -0
    0xDEAD0000u, // +0
    0xDEAD0001u, // +denormal
    0xDEAD3C00u, // +normal (1)
    0xDEAD7C00u, // +Inf
    0xDEAD4248u, // +normal
    0xDEAD7BFFu, // +normal (max)
    0xDEADFD00u, // sNaN (-)
    0xDEADFE00u, // qNaN (-)
}};

// One representative of each f64 class (and a few extra normals), same layout.
const std::array<uint64_t, 14> kF64 = {{
    0x7FF4000000000000ull, // sNaN (+)
    0x7FF8000000000000ull, // qNaN (+)
    0xFFF0000000000000ull, // -Inf
    0xBFF0000000000000ull, // -normal (-1)
    0x8000000000000001ull, // -denormal
    0x8000000000000000ull, // -0
    0x0000000000000000ull, // +0
    0x0000000000000001ull, // +denormal
    0x3FF0000000000000ull, // +normal (1)
    0x7FF0000000000000ull, // +Inf
    0x400921FB54442D18ull, // +normal (pi)
    0x7FEFFFFFFFFFFFFFull, // +normal (max)
    0xFFF4000000000000ull, // sNaN (-)
    0xFFF8000000000000ull, // qNaN (-)
}};

// Class masks: each single class bit, all bits, none, and a few mixes.
const std::array<uint32_t, 16> kMasks = {{
    0x001u,
    0x002u,
    0x004u,
    0x008u,
    0x010u,
    0x020u,
    0x040u,
    0x080u,
    0x100u,
    0x200u,
    0x3FFu,
    0x000u,
    0x155u,
    0x2AAu,
    0x0C0u,
    0x006u,
}};

enum class Kind { F16, F32, F64 };

struct ClassCase {
  const char *name;
  uint32_t opcode;
  Kind kind;
};

const std::array<ClassCase, 3> kCases = {{
    {"v_cmp_class_f16", 20, Kind::F16},
    {"v_cmp_class_f32", 16, Kind::F32},
    {"v_cmp_class_f64", 18, Kind::F64},
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vopc_class_simd_mem"), l2("vopc_class_simd_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vopc_class_simd", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void write64(uint32_t reg, uint32_t lane, uint64_t v) {
    cu->write_vgpr(reg, lane, static_cast<uint32_t>(v));
    cu->write_vgpr(reg + 1, lane, static_cast<uint32_t>(v >> 32));
  }

  // src0 = value bits (v0, or v0:v1 for f64), vsrc1 = 32-bit class mask (v1, or v2
  // for f64). Each lane gets a distinct (value, mask) pairing; the +rot shift
  // sweeps the pairing across runs so every value meets every mask over the loop.
  void seed_inputs(Kind k, uint32_t rot, uint64_t exec, uint64_t vcc_in) {
    uint32_t vb = wf->vgpr_alloc().base;
    const uint32_t mask_reg = (k == Kind::F64) ? 2u : 1u;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      if (k == Kind::F64) {
        write64(vb + 0, lane, kF64[lane % kF64.size()]);
      } else {
        const auto &vals = (k == Kind::F16) ? kF16 : kF32;
        cu->write_vgpr(vb + 0, lane, vals[lane % vals.size()]);
      }
      cu->write_vgpr(vb + mask_reg, lane, kMasks[(lane + rot) % kMasks.size()]);
    }
    wf->set_exec(exec);
    wf->set_vcc(vcc_in);
  }

  uint64_t run(Instruction *inst, Kind k, uint32_t rot, uint64_t exec, uint64_t vcc_in) {
    seed_inputs(k, rot, exec, vcc_in);
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

void check_all(uint64_t exec) {
  ForceScalarGuard gate_guard;
  const uint64_t kVcc[] = {0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0xA5A5A5A5A5A5A5A5ULL};
  for (const auto &c : kCases) {
    auto run_mode = [&](bool force_scalar, uint32_t rot, uint64_t vcc_in) -> uint64_t {
      util::set_force_scalar_for_testing(force_scalar);
      Fixture fx;
      EXPECT_NE(fx.cu, nullptr);
      EXPECT_NE(fx.wf, nullptr);
      uint32_t vsrc1 = (c.kind == Kind::F64) ? 2u : 1u; // f64 src0 spans v0:v1
      uint32_t enc = vopc_encode(c.opcode, /*src0=*/256, vsrc1);
      uint32_t words[4] = {enc, 0u, 0u, 0u};
      Instruction *inst = fx.decoder->decode(words);
      EXPECT_NE(inst, nullptr) << c.name << " decode failed";
      uint64_t vcc = fx.run(inst, c.kind, rot, exec, vcc_in);
      delete inst;
      return vcc;
    };
    for (uint32_t rot = 0; rot < kMasks.size(); ++rot) {
      for (uint64_t vcc_in : kVcc) {
        const uint64_t scalar_vcc = run_mode(/*force_scalar=*/true, rot, vcc_in);
        const uint64_t simd_vcc = run_mode(/*force_scalar=*/false, rot, vcc_in);

        EXPECT_EQ(scalar_vcc, simd_vcc) << c.name << " rot=" << rot << " vcc_in=0x" << std::hex
                                        << vcc_in << ": SIMD VCC diverged from scalar body";

        const uint64_t inactive = ~exec;
        EXPECT_EQ(simd_vcc & inactive, 0ULL)
            << c.name << " rot=" << rot << ": inactive-lane VCC bit not zeroed";
        EXPECT_EQ(scalar_vcc & inactive, 0ULL)
            << c.name << " rot=" << rot << ": inactive-lane VCC bit not zeroed";
      }
    }
  }
}

// VOP3 forms: same classification, plus the abs/neg src0 modifiers, mask read
// from src1, and an SGPR-pair (VCC here) dst. Swept over all four abs/neg combos.
struct Vop3ClassCase {
  const char *name;
  uint32_t opcode;
  Kind kind;
};

const std::array<Vop3ClassCase, 3> kVop3Cases = {{
    {"v_cmp_class_f16_e64", 20, Kind::F16},
    {"v_cmp_class_f32_e64", 16, Kind::F32},
    {"v_cmp_class_f64_e64", 18, Kind::F64},
}};

void check_all_vop3(uint64_t exec) {
  ForceScalarGuard gate_guard;
  const uint64_t kVcc[] = {0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0xA5A5A5A5A5A5A5A5ULL};
  for (const auto &c : kVop3Cases) {
    const uint32_t src1 = 256u + ((c.kind == Kind::F64) ? 2u : 1u); // mask vgpr
    for (uint32_t abs = 0; abs <= 1; ++abs) {
      for (uint32_t neg = 0; neg <= 1; ++neg) {
        auto run_mode = [&](bool force_scalar, uint32_t rot, uint64_t vcc_in) -> uint64_t {
          util::set_force_scalar_for_testing(force_scalar);
          Fixture fx;
          EXPECT_NE(fx.cu, nullptr);
          EXPECT_NE(fx.wf, nullptr);
          uint32_t words[4] = {0u, 0u, 0u, 0u};
          vop3_encode(c.opcode, /*vdst=*/kVccSdst, /*src0=*/256, src1, abs, neg, words);
          Instruction *inst = fx.decoder->decode(words);
          EXPECT_NE(inst, nullptr) << c.name << " decode failed";
          uint64_t vcc = fx.run(inst, c.kind, rot, exec, vcc_in);
          delete inst;
          return vcc;
        };
        for (uint32_t rot = 0; rot < kMasks.size(); ++rot) {
          for (uint64_t vcc_in : kVcc) {
            const uint64_t scalar_vcc = run_mode(/*force_scalar=*/true, rot, vcc_in);
            const uint64_t simd_vcc = run_mode(/*force_scalar=*/false, rot, vcc_in);

            EXPECT_EQ(scalar_vcc, simd_vcc)
                << c.name << " abs=" << abs << " neg=" << neg << " rot=" << rot << " vcc_in=0x"
                << std::hex << vcc_in << ": SIMD VCC diverged from scalar body";

            const uint64_t inactive = ~exec;
            EXPECT_EQ(simd_vcc & inactive, 0ULL)
                << c.name << " abs=" << abs << " neg=" << neg << ": inactive VCC bit not zeroed";
            EXPECT_EQ(scalar_vcc & inactive, 0ULL)
                << c.name << " abs=" << abs << " neg=" << neg << ": inactive VCC bit not zeroed";
          }
        }
      }
    }
  }
}

TEST(VopcCmpClassSimdCorrectness, FullExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check_all(/*exec=*/~0ULL);
  check_all_vop3(/*exec=*/~0ULL);
}

TEST(VopcCmpClassSimdCorrectness, PartialExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check_all(/*exec=*/0xA5A5'F0F0'1234'8001ULL);
  check_all_vop3(/*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
