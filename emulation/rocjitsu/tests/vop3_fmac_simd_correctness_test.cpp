// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_fmac_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs forced-scalar body) for the
/// dst-accumulate FMA / MAC VOP3 forms on CDNA4 (v_fmac_f32, v_mac_f16,
/// v_fmac_f64). Per-isa codegen classes for these ops only initialize
/// src0+src1+vdst — the third FMA operand IS vdst. The new accumulate-form
/// glue (try_execute_fmac_vop3_fp*_simd) reads inst.vdst as the third operand
/// and applies abs/neg only to src0/src1, matching the scalar body. NaN-input
/// lanes skipped (accepted FMA carve-out).
///
/// These ops are NOT benched: looping the same instruction creates a loop-
/// carried RAW dep on the accumulator that serializes both modes to ~1x.

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
constexpr uint32_t kAccVgpr = 4; // accumulator (vdst)

constexpr void vop3_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1, uint32_t abs,
                           uint32_t neg, uint32_t omod, uint32_t clamp, uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((abs & 0x7) << 8) | ((clamp & 0x1) << 15) | ((op & 0x3FF) << 16) |
             (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((omod & 0x3) << 27) | ((neg & 0x7) << 29);
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
};

const std::array<Case, 3> kCases = {{
    {"v_fmac_f32_vop3", 315, Kind::F32},
    {"v_mac_f16_vop3", 291, Kind::F16},
    {"v_fmac_f64_vop3", 260, Kind::F64},
}};

// Finite-normal sanitized inputs (NaN lanes skipped; payload divergence is
// accepted for the FMA path).
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

  Fixture() : gpu_mem("vop3_fmac_mem"), l2("vop3_fmac_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_fmac", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void write64(uint32_t reg, uint32_t lane, uint64_t v) {
    cu->write_vgpr(reg, lane, static_cast<uint32_t>(v));
    cu->write_vgpr(reg + 1, lane, static_cast<uint32_t>(v >> 32));
  }

  void seed_inputs(Kind k, uint32_t rot, uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      if (k == Kind::F64) {
        write64(vb + 0, lane, kF64[lane % kF64.size()]);
        write64(vb + 2, lane, kF64[(lane + rot) % kF64.size()]);
        // Accumulator: distinct value (used as 3rd FMA arg + overwritten).
        write64(vb + kAccVgpr, lane, kF64[(lane + 2 * rot) % kF64.size()]);
      } else {
        const auto &v = (k == Kind::F32) ? kF32 : kF16;
        cu->write_vgpr(vb + 0, lane, v[lane % v.size()]);
        cu->write_vgpr(vb + 1, lane, v[(lane + rot) % v.size()]);
        cu->write_vgpr(vb + kAccVgpr, lane, v[(lane + 2 * rot) % v.size()]);
      }
    }
    wf->set_exec(exec);
  }

  std::array<uint64_t, WF_SIZE> run(Instruction *inst, Kind k, bool force_scalar, uint32_t rot,
                                    uint64_t exec) {
    amdgpu::simd_force_scalar() = force_scalar;
    seed_inputs(k, rot, exec);
    cu->execute_instruction(inst, *wf);
    amdgpu::simd_force_scalar() = false;
    std::array<uint64_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      if (k == Kind::F64)
        out[lane] = static_cast<uint64_t>(cu->read_vgpr(vb + kAccVgpr + 1, lane)) << 32 |
                    cu->read_vgpr(vb + kAccVgpr, lane);
      else
        out[lane] = cu->read_vgpr(vb + kAccVgpr, lane);
    }
    return out;
  }
};

void check_case(const Case &c, uint32_t abs, uint32_t neg, uint32_t omod, uint32_t clamp,
                uint64_t exec) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  const uint32_t src1_v = (c.kind == Kind::F64) ? 2u : 1u;
  uint32_t words[4] = {0u, 0u, 0u, 0u};
  vop3_encode(c.opcode, /*vdst=*/kAccVgpr, /*src0=*/256, /*src1=*/256 + src1_v, abs, neg, omod,
              clamp, words);
  Instruction *inst = fx.decoder->decode(words);
  ASSERT_NE(inst, nullptr) << c.name << " decode failed";
  const std::size_t rot_max = (c.kind == Kind::F64) ? kF64.size() : kF32.size();
  for (uint32_t rot = 0; rot < rot_max; ++rot) {
    auto scalar = fx.run(inst, c.kind, /*force_scalar=*/true, rot, exec);
    auto simd = fx.run(inst, c.kind, /*force_scalar=*/false, rot, exec);
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      if (!((exec >> lane) & 1ULL))
        continue;
      bool nan_either = false;
      if (c.kind == Kind::F32)
        nan_either = is_f32_nan(static_cast<uint32_t>(scalar[lane])) ||
                     is_f32_nan(static_cast<uint32_t>(simd[lane]));
      else if (c.kind == Kind::F16)
        nan_either = is_f16_nan(static_cast<uint32_t>(scalar[lane])) ||
                     is_f16_nan(static_cast<uint32_t>(simd[lane]));
      else
        nan_either = is_f64_nan(scalar[lane]) || is_f64_nan(simd[lane]);
      if (nan_either)
        continue;
      EXPECT_EQ(scalar[lane], simd[lane])
          << c.name << " abs=" << abs << " neg=" << neg << " omod=" << omod << " clamp=" << clamp
          << " rot=" << rot << " lane=" << lane << ": divergence scalar=0x" << std::hex
          << scalar[lane] << " simd=0x" << simd[lane];
    }
  }
  delete inst;
}

void check_all_mods(const Case &c, uint64_t exec) {
  // Only src0/src1 carry abs/neg (vdst is accumulator); skip src2 bit.
  for (uint32_t abs = 0; abs < 4; ++abs)
    for (uint32_t neg = 0; neg < 4; ++neg)
      for (uint32_t omod = 0; omod < 4; ++omod)
        for (uint32_t clamp = 0; clamp < 2; ++clamp)
          check_case(c, abs, neg, omod, clamp, exec);
}

TEST(Vop3FmacSimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_all_mods(c, /*exec=*/~0ULL);
}

TEST(Vop3FmacSimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*abs=*/0, /*neg=*/0, /*omod=*/0, /*clamp=*/0, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
