// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_cvt_f64_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the VOP3 form
/// of the six mixed-width f64<->32-bit conversions on CDNA4. Their scalar
/// generated bodies drop the abs/neg/omod/clamp modifier reads (verified
/// per-op), so the VOP3 form is bit-identical to the VOP1 form and reuses the
/// existing try_execute_cvt_{f64_to_b32, b32_to_f64}_simd glue via a `_vop3 ->
/// _vop1` fallback in simd_probe_line. The process runs one fixed execute mode
/// (RJ_FORCE_SCALAR, immutable); the destination word pair is recorded and the
/// scalar-vs-SIMD equivalence is asserted by diffing the two runs (see simd_ab.h
/// / the simd_ab_diff CTest entry). In-process inactive lanes must stay
/// preserved under full and partial EXEC.
///
/// The float-result conversions (f32_f64 / f64_f32) are correctly rounded
/// (vcvtpd2ps / vcvtps2pd), bit-exact for finite/Inf inputs; a NaN result may
/// differ in payload between packed and scalar (accepted), so those lanes are
/// overwritten with a fixed sentinel before recording — NaN-ness is
/// deterministic from the inputs, so both runs mask the same lanes. The int
/// conversions are exact (NaN -> 0, saturating clamp), recorded with no carve-out.

#include "simd_ab.h"

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
#include <bit>
#include <cstdint>
#include <memory>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
// Fixed marker written over accepted-divergence (NaN-result) lanes before
// recording, so the cross-run diff ignores them identically in both runs.
constexpr uint32_t SKIP_SENTINEL = 0xA11D1FFFu;

constexpr void vop3_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((op & 0x3FF) << 16) | (0x34u << 26);
  words[1] = (src0 & 0x1FF);
}

bool is_f32_nan(uint32_t bits) {
  return ((bits >> 23) & 0xFFu) == 0xFFu && (bits & 0x7FFFFFu) != 0;
}
bool is_f64_nan(uint64_t bits) {
  return ((bits >> 52) & 0x7FFu) == 0x7FFu && (bits & 0xFFFFFFFFFFFFFull) != 0;
}

struct CvtCase {
  const char *name;
  uint32_t opcode;
  bool src64;
  bool dst64;
  bool skip_nan;
};

const std::array<CvtCase, 6> kCases = {{
    {"v_cvt_f32_f64_vop3", 335, true, false, true},
    {"v_cvt_i32_f64_vop3", 323, true, false, false},
    {"v_cvt_u32_f64_vop3", 341, true, false, false},
    {"v_cvt_f64_f32_vop3", 336, false, true, true},
    {"v_cvt_f64_i32_vop3", 324, false, true, false},
    {"v_cvt_f64_u32_vop3", 342, false, true, false},
}};

// Mixed f64 inputs (bit pattern == every IEEE class + nonzero high words so the
// scalar-fix is exercised) and 32-bit raw inputs covering signed/unsigned wrap.
const std::array<uint64_t, 16> kF64In = {{
    0x0000000000000000ull, 0x8000000000000000ull, 0x3FF0000000000000ull, 0xBFF0000000000000ull,
    0x400921FB54442D18ull, 0xC00921FB54442D18ull, 0x7FF0000000000000ull, 0xFFF0000000000000ull,
    0x7FF8000000000000ull, 0xFFF4000000000000ull, 0x0000000000000001ull, 0x8000000000000001ull,
    0x7FEFFFFFFFFFFFFFull, 0x41E0000000000000ull, // 2^31
    0x41F0000000000000ull,                        // 2^32
    0xC1E0000000000001ull,                        // -2^31 - eps
}};
const std::array<uint32_t, 16> kB32In = {{
    0x00000000u,
    0x00000001u,
    0x80000000u,
    0x7FFFFFFFu,
    0xFFFFFFFFu,
    0x12345678u,
    0xDEADBEEFu,
    0xCAFEBABEu,
    0x3F800000u,
    0x40490FDBu,
    0x7F800000u,
    0xFF800000u,
    0x7FC00000u,
    0x00000001u,
    0x80000001u,
    0x7F7FFFFFu,
}};

uint64_t dst_sentinel(uint32_t lane) {
  return (static_cast<uint64_t>(0xCAFE0000u | lane) << 32) | (0xBEEF0000u + lane);
}

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3_cvt_f64_mem"), l2("vop3_cvt_f64_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_cvt_f64", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void write64(uint32_t reg, uint32_t lane, uint64_t v) {
    cu->write_vgpr(reg, lane, static_cast<uint32_t>(v));
    cu->write_vgpr(reg + 1, lane, static_cast<uint32_t>(v >> 32));
  }
  uint64_t read64(uint32_t reg, uint32_t lane) {
    return static_cast<uint64_t>(cu->read_vgpr(reg + 1, lane)) << 32 | cu->read_vgpr(reg, lane);
  }

  void seed_inputs(const CvtCase &c, uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      if (c.src64)
        write64(vb + 0, lane, kF64In[lane % kF64In.size()]);
      else
        cu->write_vgpr(vb + 0, lane, kB32In[lane % kB32In.size()]);
      write64(vb + 2, lane, dst_sentinel(lane));
    }
    wf->set_exec(exec);
  }

  std::array<uint64_t, WF_SIZE> run(Instruction *inst, const CvtCase &c, uint64_t exec) {
    seed_inputs(c, exec);
    cu->execute_instruction(inst, *wf);
    std::array<uint64_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = read64(vb + 2, lane);
    return out;
  }
};

void check_case(const CvtCase &c, uint64_t exec) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  uint32_t words[4] = {0u, 0u, 0u, 0u};
  vop3_encode(c.opcode, /*vdst=*/2, /*src0=*/256, words);
  Instruction *inst = fx.decoder->decode(words);
  ASSERT_NE(inst, nullptr) << c.name << " decode failed";
  auto out = fx.run(inst, c, exec);

  // Record the dst word pair (lo, hi). For float-result conversions a NaN-result
  // lane carries a possibly-different NaN payload (accepted divergence); the
  // NaN-ness is deterministic from the inputs, so both runs mask the same lanes
  // with a fixed sentinel. For the 32-bit-dst forms the high half is the
  // (deterministic) sentinel, recorded as-is so the high-half no-write invariant
  // is also covered by the cross-run diff.
  uint32_t words_out[2 * WF_SIZE];
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (exec >> lane) & 1ULL;
    uint64_t v = out[lane];
    uint32_t lo = static_cast<uint32_t>(v);
    uint32_t hi = static_cast<uint32_t>(v >> 32);
    if (active && c.skip_nan) {
      if (c.dst64) {
        if (is_f64_nan(v)) {
          lo = SKIP_SENTINEL;
          hi = SKIP_SENTINEL;
        }
      } else if (is_f32_nan(lo)) {
        lo = SKIP_SENTINEL;
      }
    }
    words_out[2 * lane] = lo;
    words_out[2 * lane + 1] = hi;
  }
  simd_ab::record(c.name, exec, words_out, 2 * WF_SIZE);

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    if (!(exec & (1ULL << lane))) {
      EXPECT_EQ(out[lane], dst_sentinel(lane)) << c.name << " clobbered inactive lane " << lane;
    }
  }
  delete inst;
}

TEST(Vop3CvtF64SimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/~0ULL);
}

TEST(Vop3CvtF64SimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
