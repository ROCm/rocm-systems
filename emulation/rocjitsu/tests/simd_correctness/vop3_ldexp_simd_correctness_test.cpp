// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_ldexp_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the
/// mixed-width VOP3 ldexp ops on CDNA4: v_ldexp_f32 (f32 src0 + int32 src1
/// exp) and v_ldexp_f64 (f64 src0 + int32 src1 exp). stdx::ldexp is bit-exact
/// to std::ldexp for every input incl. NaN (proven in the VOP2 v_ldexp_f16
/// path), so no carve-out needed. Each (case, mods, rot) runs TWICE in the same
/// process -- once forcing the scalar body, once the SIMD fast path, with
/// identical inputs/EXEC -- and the result arrays are asserted equal with
/// EXPECT_EQ (util::set_force_scalar_for_testing flips the gate in-process).

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

constexpr void vop3_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1, uint32_t abs,
                           uint32_t neg, uint32_t omod, uint32_t clamp, uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((abs & 0x7) << 8) | ((clamp & 0x1) << 15) | ((op & 0x3FF) << 16) |
             (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((omod & 0x3) << 27) | ((neg & 0x7) << 29);
}

struct Case {
  const char *name;
  uint32_t opcode;
  bool f64;
};

const std::array<Case, 2> kCases = {{
    {"v_ldexp_f32_vop3", 648, false},
    {"v_ldexp_f64_vop3", 644, true},
}};

const std::array<uint32_t, 12> kF32 = {{
    0x3F800000u,
    0xBF800000u,
    0x40000000u,
    0xC0490FDBu,
    0x3F000000u,
    0x7F7FFFFFu,
    0x00800000u,
    0x80000001u,
    0x7F800000u,
    0xFF800000u,
    0x7FC00000u,
    0x00000000u,
}};
const std::array<uint64_t, 12> kF64 = {{
    0x3FF0000000000000ull,
    0xBFF0000000000000ull,
    0x4000000000000000ull,
    0xC00921FB54442D18ull,
    0x3FE0000000000000ull,
    0x7FEFFFFFFFFFFFFFull,
    0x0010000000000000ull,
    0x8000000000000001ull,
    0x7FF0000000000000ull,
    0xFFF0000000000000ull,
    0x7FF8000000000000ull,
    0x0000000000000000ull,
}};
const std::array<int32_t, 12> kExp = {
    {-50, -10, -1, 0, 1, 5, 10, 31, 50, 127, -127, 0x7FFFFFFF},
};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3_ldexp_mem"), l2("vop3_ldexp_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_ldexp", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void write64(uint32_t reg, uint32_t lane, uint64_t v) {
    cu->write_vgpr(reg, lane, static_cast<uint32_t>(v));
    cu->write_vgpr(reg + 1, lane, static_cast<uint32_t>(v >> 32));
  }

  void seed_inputs(bool f64, uint32_t rot, uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      if (f64) {
        write64(vb + 0, lane, kF64[lane % kF64.size()]);
        cu->write_vgpr(vb + 2, lane, static_cast<uint32_t>(kExp[(lane + rot) % kExp.size()]));
        write64(vb + 4, lane, 0xCDCDCDCDCDCDCDCDull);
      } else {
        cu->write_vgpr(vb + 0, lane, kF32[lane % kF32.size()]);
        cu->write_vgpr(vb + 1, lane, static_cast<uint32_t>(kExp[(lane + rot) % kExp.size()]));
        cu->write_vgpr(vb + 4, lane, 0xCDCDCDCDu);
      }
    }
    wf->set_exec(exec);
  }

  std::array<uint64_t, WF_SIZE> run(Instruction *inst, bool f64, uint32_t rot, uint64_t exec) {
    seed_inputs(f64, rot, exec);
    cu->execute_instruction(inst, *wf);
    std::array<uint64_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      if (f64)
        out[lane] =
            static_cast<uint64_t>(cu->read_vgpr(vb + 5, lane)) << 32 | cu->read_vgpr(vb + 4, lane);
      else
        out[lane] = cu->read_vgpr(vb + 4, lane);
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

void check_case(const Case &c, uint32_t abs, uint32_t neg, uint32_t omod, uint32_t clamp,
                uint64_t exec) {
  ForceScalarGuard gate_guard;

  auto run_mode = [&](bool force_scalar, uint32_t rot) -> std::array<uint64_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    const uint32_t src1_v = c.f64 ? 2u : 1u;
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_encode(c.opcode, /*vdst=*/4, /*src0=*/256, /*src1=*/256 + src1_v, abs, neg, omod, clamp,
                words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.name << " decode failed";
    auto out = fx.run(inst, c.f64, rot, exec);
    delete inst;
    return out;
  };

  // ldexp is bit-exact in both modes (no NaN carve-out), so the full result
  // arrays must match.
  for (uint32_t rot = 0; rot < kExp.size(); ++rot) {
    const auto scalar_out = run_mode(/*force_scalar=*/true, rot);
    const auto simd_out = run_mode(/*force_scalar=*/false, rot);
    EXPECT_EQ(scalar_out, simd_out)
        << c.name << " a" << abs << "n" << neg << "o" << omod << "c" << clamp << "r" << rot
        << ": SIMD path diverged from scalar body";
  }
}

void check_all_mods(const Case &c, uint64_t exec) {
  // Only src0 has abs/neg in ldexp (src1 is integer exp); sweep src0 bits only.
  for (uint32_t abs = 0; abs < 2; ++abs)
    for (uint32_t neg = 0; neg < 2; ++neg)
      for (uint32_t omod = 0; omod < 4; ++omod)
        for (uint32_t clamp = 0; clamp < 2; ++clamp)
          check_case(c, abs, neg, omod, clamp, exec);
}

TEST(Vop3LdexpSimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_all_mods(c, /*exec=*/~0ULL);
}

TEST(Vop3LdexpSimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*abs=*/0, /*neg=*/0, /*omod=*/0, /*clamp=*/0, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
