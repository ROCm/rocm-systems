// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop1_cvt_f64_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs forced-scalar body) for the six
/// mixed-width f64<->32-bit VOP1 conversions on CDNA4: the f64-source forms
/// (v_cvt_f32_f64, v_cvt_i32_f64, v_cvt_u32_f64) and the f64-dst forms
/// (v_cvt_f64_f32, v_cvt_f64_i32, v_cvt_f64_u32). These bridge an 8-wide
/// (native_width64) f64 chunk and the same number of 32-bit lanes, so they use
/// dedicated cvt glue (read_simd64 / narrow32) rather than the equal-width unary
/// path. Each op runs in both modes in-process via amdgpu::simd_force_scalar()
/// and the destination VGPR(s) must agree on active lanes, with inactive lanes
/// preserved, under full and partial EXEC.
///
/// The float-result conversions (v_cvt_f32_f64, v_cvt_f64_f32) are correctly
/// rounded (vcvtpd2ps / vcvtps2pd), bit-exact for finite/Inf inputs; a NaN
/// *result* may carry a different payload between packed and scalar (accepted),
/// so those lanes are skipped. The int conversions are exact (NaN->0 and the
/// saturating clamp are deterministic), compared with no carve-out.

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
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;

// CDNA4 VOP1: enc[31:25] = 0b0111111, vdst[24:17], op[16:9], src0[8:0].
constexpr uint32_t vop1_encode(uint32_t op, uint32_t vdst, uint32_t src0) {
  return (0x3Fu << 25) | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF);
}

// f64-source inputs: ±0, ±1, fractional, the clamp boundaries (2^31, 2^32) and
// values straddling them, ±huge, NaN, ±Inf, denorm, pi, -e. Exercises every
// branch of the int clamp/NaN logic and the correctly-rounded narrowing cast.
const std::array<double, 20> kF64In = {{
    +0.0,
    -0.0,
    1.0,
    -1.0,
    2.5,
    -2.5,
    0.5,
    3.0e9,                                       // > 2^31, < 2^32
    5.0e9,                                       // > 2^32
    -5.0e9,                                      // < -2^31
    1.0e300,                                     // overflow both
    std::numeric_limits<double>::quiet_NaN(),    //
    std::numeric_limits<double>::infinity(),     //
    -std::numeric_limits<double>::infinity(),    //
    std::numeric_limits<double>::denorm_min(),   //
    2147483647.0,                                // exactly INT32_MAX
    2147483648.0,                                // exactly 2^31
    4294967295.0,                                // exactly UINT32_MAX
    3.141592653589793,                           //
    -2.718281828459045,                          //
}};

// 32-bit-source inputs, reinterpreted per op as f32 / i32 / u32 bits. Covers ±0,
// ±1, ±Inf, NaN, denorm, pi, max (as f32) and signed/extreme boundaries (as int).
const std::array<uint32_t, 14> kB32In = {{
    0x00000000u, 0x80000000u, 0x3F800000u, 0xBF800000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u,
    0x00000001u, 0x40490FDBu, 0x7F7FFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu, 0x12345678u, 0xAAAA5555u,
}};

bool is_f32_nan(uint32_t bits) { return std::isnan(std::bit_cast<float>(bits)); }
bool is_f64_nan(uint64_t bits) { return std::isnan(std::bit_cast<double>(bits)); }

struct CvtCase {
  const char *name;
  uint32_t opcode;
  bool src64;    // source operand is a 64-bit f64 (VGPR pair)
  bool dst64;    // destination is a 64-bit f64 (VGPR pair)
  bool skip_nan; // result is a float whose NaN payload may diverge (accepted)
};

const std::array<CvtCase, 6> kCases = {{
    {"v_cvt_f32_f64", 15, true, false, true},
    {"v_cvt_i32_f64", 3, true, false, false},
    {"v_cvt_u32_f64", 21, true, false, false},
    {"v_cvt_f64_f32", 16, false, true, true},
    {"v_cvt_f64_i32", 4, false, true, false},
    {"v_cvt_f64_u32", 22, false, true, false},
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

  Fixture() : gpu_mem("vop1_cvt_f64_simd_mem"), l2("vop1_cvt_f64_simd_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop1_cvt_f64_simd", cfg, &gpu_mem, &l2);
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

  // src0 = v0(:v1), vdst = v2(:v3). vdst is stamped with a per-lane sentinel so
  // inactive-lane preservation is checkable in both width configurations.
  void seed_inputs(const CvtCase &c, uint64_t exec) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      if (c.src64)
        write64(vb + 0, lane, std::bit_cast<uint64_t>(kF64In[lane % kF64In.size()]));
      else
        cu->write_vgpr(vb + 0, lane, kB32In[lane % kB32In.size()]);
      write64(vb + 2, lane, dst_sentinel(lane));
    }
    wf->set_exec(exec);
  }

  std::array<uint64_t, WF_SIZE> run(Instruction *inst, const CvtCase &c, bool force_scalar,
                                    uint64_t exec) {
    amdgpu::simd_force_scalar() = force_scalar;
    seed_inputs(c, exec);
    cu->execute_instruction(inst, *wf);
    amdgpu::simd_force_scalar() = false;
    std::array<uint64_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = read64(vb + 2, lane); // reads both halves; 32-bit dst leaves v3 = sentinel hi
    return out;
  }
};

void check_case(const CvtCase &c, uint64_t exec) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  uint32_t enc = vop1_encode(c.opcode, /*vdst=*/2, /*src0=*/256);
  uint32_t words[4] = {enc, 0u, 0u, 0u};
  Instruction *inst = fx.decoder->decode(words);
  ASSERT_NE(inst, nullptr) << c.name << " decode failed";

  auto scalar = fx.run(inst, c, /*force_scalar=*/true, exec);
  auto simd = fx.run(inst, c, /*force_scalar=*/false, exec);

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    if (!(exec & (1ULL << lane))) {
      EXPECT_EQ(simd[lane], dst_sentinel(lane))
          << c.name << " clobbered inactive lane " << lane << " (simd)";
      EXPECT_EQ(scalar[lane], dst_sentinel(lane))
          << c.name << " clobbered inactive lane " << lane << " (scalar)";
      continue;
    }
    if (c.dst64) {
      // Full 64-bit f64 result.
      if (c.skip_nan && (is_f64_nan(scalar[lane]) || is_f64_nan(simd[lane])))
        continue;
      EXPECT_EQ(scalar[lane], simd[lane])
          << c.name << " dst divergence at lane " << lane << std::hex << " scalar=0x"
          << scalar[lane] << " simd=0x" << simd[lane];
    } else {
      // 32-bit result lives in the low half; the high half must keep the sentinel.
      uint32_t sc = static_cast<uint32_t>(scalar[lane]);
      uint32_t sd = static_cast<uint32_t>(simd[lane]);
      EXPECT_EQ(simd[lane] >> 32, dst_sentinel(lane) >> 32)
          << c.name << " wrote into the high VGPR at lane " << lane;
      if (c.skip_nan && (is_f32_nan(sc) || is_f32_nan(sd)))
        continue;
      EXPECT_EQ(sc, sd) << c.name << " dst divergence at lane " << lane << std::hex << " scalar=0x"
                        << sc << " simd=0x" << sd;
    }
  }
  delete inst;
}

TEST(Vop1CvtF64SimdCorrectness, FullExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  for (const auto &c : kCases)
    check_case(c, /*exec=*/~0ULL);
}

TEST(Vop1CvtF64SimdCorrectness, PartialExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  // Pattern crossing the 8-wide chunk boundaries on a wave64.
  for (const auto &c : kCases)
    check_case(c, /*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
