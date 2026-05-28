// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vopc_vop3_fp64_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs forced-scalar body) for the
/// VOP3 form of the f64 VOPC relational compares on CDNA4. 64-bit-lane
/// counterpart of the f32 path: read src0/src1 as native<double> via the
/// split lo/hi VGPR-pair path, apply abs/neg in the f64 domain, compare.
/// Result merges into an SGPR-pair dst (pointed at VCC=106 here so the same
/// harness checks via wf.vcc()). All four abs/neg combos swept; 16 of 17
/// rels are cdna4-decodable.

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
constexpr uint32_t kVccSdst = 106;

constexpr void vop3_cmp_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1,
                               uint32_t abs, uint32_t neg, uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((abs & 0x7) << 8) | ((op & 0x3FF) << 16) | (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((neg & 0x7) << 29);
}

struct Case {
  const char *name;
  uint32_t opcode;
};

const std::array<Case, 16> kCases = {{
    {"v_cmp_f_f64_vop3", 96},
    {"v_cmp_lt_f64_vop3", 97},
    {"v_cmp_eq_f64_vop3", 98},
    {"v_cmp_le_f64_vop3", 99},
    {"v_cmp_gt_f64_vop3", 100},
    {"v_cmp_lg_f64_vop3", 101},
    {"v_cmp_ge_f64_vop3", 102},
    {"v_cmp_o_f64_vop3", 103},
    {"v_cmp_u_f64_vop3", 104},
    {"v_cmp_nge_f64_vop3", 105},
    {"v_cmp_nlg_f64_vop3", 106},
    {"v_cmp_ngt_f64_vop3", 107},
    {"v_cmp_nle_f64_vop3", 108},
    {"v_cmp_neq_f64_vop3", 109},
    {"v_cmp_nlt_f64_vop3", 110},
    {"v_cmp_tru_f64_vop3", 111},
}};

// f64 IEEE classes + ±0 / ±1 / ±max / ±denormal so abs/neg sign flips exercise
// real boundary pairs. NaN payloads don't propagate through compares.
const std::array<uint64_t, 14> kF64 = {{
    0x0000000000000000ull, // +0
    0x8000000000000000ull, // -0
    0x3FF0000000000000ull, // +1
    0xBFF0000000000000ull, // -1
    0x400921FB54442D18ull, // +pi
    0xC00921FB54442D18ull, // -pi
    0x7FF0000000000000ull, // +Inf
    0xFFF0000000000000ull, // -Inf
    0x7FF8000000000000ull, // +qNaN
    0xFFF4000000000000ull, // -sNaN
    0x0000000000000001ull, // +denormal
    0x8000000000000001ull, // -denormal
    0x7FEFFFFFFFFFFFFFull, // +max
    0xFFEFFFFFFFFFFFFFull, // -max
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vopc_vop3_fp64_mem"), l2("vopc_vop3_fp64_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vopc_vop3_fp64", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void write64(uint32_t reg, uint32_t lane, uint64_t v) {
    cu->write_vgpr(reg, lane, static_cast<uint32_t>(v));
    cu->write_vgpr(reg + 1, lane, static_cast<uint32_t>(v >> 32));
  }

  // src0 = v0:v1, src1 = v2:v3. +rot sweeps pairing across kF64.
  void seed_inputs(uint32_t rot, uint64_t exec, uint64_t vcc_in) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      write64(vb + 0, lane, kF64[lane % kF64.size()]);
      write64(vb + 2, lane, kF64[(lane + rot) % kF64.size()]);
    }
    wf->set_exec(exec);
    wf->set_vcc(vcc_in);
  }

  uint64_t run(Instruction *inst, bool force_scalar, uint32_t rot, uint64_t exec, uint64_t vcc_in) {
    amdgpu::simd_force_scalar() = force_scalar;
    seed_inputs(rot, exec, vcc_in);
    cu->execute_instruction(inst, *wf);
    amdgpu::simd_force_scalar() = false;
    return wf->vcc();
  }
};

void check_case(const Case &c, uint64_t exec) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  const uint64_t kVcc[] = {0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0xA5A5A5A5A5A5A5A5ULL};
  for (uint32_t abs = 0; abs < 4; ++abs) {
    for (uint32_t neg = 0; neg < 4; ++neg) {
      uint32_t words[4] = {0u, 0u, 0u, 0u};
      vop3_cmp_encode(c.opcode, /*vdst=*/kVccSdst, /*src0=*/256, /*src1=*/258, abs, neg, words);
      Instruction *inst = fx.decoder->decode(words);
      ASSERT_NE(inst, nullptr) << c.name << " decode failed (abs=" << abs << " neg=" << neg << ")";
      for (uint32_t rot = 0; rot < kF64.size(); ++rot) {
        for (uint64_t vcc_in : kVcc) {
          uint64_t scalar = fx.run(inst, /*force_scalar=*/true, rot, exec, vcc_in);
          uint64_t simd = fx.run(inst, /*force_scalar=*/false, rot, exec, vcc_in);
          EXPECT_EQ(scalar, simd) << c.name << " abs=" << abs << " neg=" << neg << " rot=" << rot
                                  << " vcc_in=0x" << std::hex << vcc_in
                                  << ": dst divergence scalar=0x" << scalar << " simd=0x" << simd;
          const uint64_t inactive = ~exec;
          EXPECT_EQ(simd & inactive, vcc_in & inactive)
              << c.name << " abs=" << abs << " neg=" << neg << " rot=" << rot
              << ": SIMD altered inactive-lane dst bit";
        }
      }
      delete inst;
    }
  }
}

TEST(VopcVop3Fp64SimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/~0ULL);
}

TEST(VopcVop3Fp64SimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
