// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop1_cvt_f64_scalar_correctness_test.cpp
/// @brief Regression test for the mixed-width f64<->32-bit VOP1 conversions.
/// These six ops have a 64-bit operand on one side and a 32-bit operand on the
/// other; the scalar (functional-sim) bodies must read/write the f64 side
/// through the full 64-bit lane accessors (read_lane64/write_lane64), not the
/// 32-bit ones. A prior codegen bug bound every operand to a single
/// instruction-level width, so the f64 side only read/wrote its low 32 bits and
/// any double with a nonzero high word (e.g. 3.7) decoded to garbage.
///
/// Every input here is deliberately chosen to have a NONZERO high 32 bits on
/// the f64 side, so a low-word-only read or a high-word-dropping write produces
/// a visibly wrong result. No SIMD is involved (these ops are not on the SIMD
/// fast path); this exercises the plain decode+execute scalar path.

#include "rocjitsu/code/rj_code.h"
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
#include <memory>

namespace {

using namespace rocjitsu;

[[maybe_unused]] constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;

constexpr uint32_t vop1_encode(uint32_t op, uint32_t vdst, uint32_t src0) {
  return (0x3Fu << 25) | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF);
}

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop1_cvt_f64_scalar_mem"), l2("vop1_cvt_f64_scalar_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop1_cvt_f64_scalar", cfg, &gpu_mem, &l2);
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

  // Execute `op` once on lane 0 with src0 (v0[:v1]) seeded to `src_lo`/`src_hi`.
  // Returns the destination VGPR pair as a 64-bit value (32-bit dst uses the
  // low word). vdst = v2[:v3]; the dst pair is pre-stamped so a dropped high
  // word from a 32-bit write is visible.
  uint64_t run(uint32_t op, uint32_t src_lo, uint32_t src_hi) {
    uint32_t vb = wf->vgpr_alloc().base;
    wf->set_exec(~0ULL);
    write64(vb + 0, 0, (static_cast<uint64_t>(src_hi) << 32) | src_lo);
    write64(vb + 2, 0, 0xA5A5A5A5A5A5A5A5ull); // sentinel in dst pair
    uint32_t enc = vop1_encode(op, /*vdst=*/2, /*src0=*/256);
    uint32_t words[4] = {enc, 0u, 0u, 0u};
    Instruction *inst = decoder->decode(words);
    EXPECT_NE(inst, nullptr);
    cu->execute_instruction(inst, *wf);
    delete inst;
    return read64(vb + 2, 0);
  }
};

// --- f64 source -> 32-bit dest ------------------------------------------------

TEST(Vop1CvtF64Scalar, F32FromF64_FullDoubleRead) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  // Doubles with nonzero high words; a low-word-only read would corrupt them.
  for (double d : {3.7, -3.7, 123456.789, -9.87654321e8, 1e-10, 2.5}) {
    uint64_t bits = std::bit_cast<uint64_t>(d);
    uint64_t got = fx.run(/*v_cvt_f32_f64=*/15, static_cast<uint32_t>(bits),
                          static_cast<uint32_t>(bits >> 32));
    uint32_t want = std::bit_cast<uint32_t>(static_cast<float>(d));
    EXPECT_EQ(static_cast<uint32_t>(got), want) << "v_cvt_f32_f64 of " << d;
  }
}

TEST(Vop1CvtF64Scalar, I32FromF64_FullDoubleRead) {
  Fixture fx;
  for (double d : {3.7, -3.7, 100.9, -100.9, 1e12, -1e12}) {
    uint64_t bits = std::bit_cast<uint64_t>(d);
    uint64_t got =
        fx.run(/*v_cvt_i32_f64=*/3, static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32));
    int32_t ref = std::isnan(d)       ? 0
                  : d >= 2147483648.0 ? INT32_MAX
                  : d < -2147483648.0 ? INT32_MIN
                                      : static_cast<int32_t>(d);
    EXPECT_EQ(static_cast<uint32_t>(got), static_cast<uint32_t>(ref)) << "v_cvt_i32_f64 of " << d;
  }
}

TEST(Vop1CvtF64Scalar, U32FromF64_FullDoubleRead) {
  Fixture fx;
  for (double d : {3.7, 100.9, 4.0e9, 1e12, -1.0}) {
    uint64_t bits = std::bit_cast<uint64_t>(d);
    uint64_t got = fx.run(/*v_cvt_u32_f64=*/21, static_cast<uint32_t>(bits),
                          static_cast<uint32_t>(bits >> 32));
    uint32_t ref = (std::isnan(d) || d < 0.0) ? 0u
                   : d >= 4294967296.0        ? UINT32_MAX
                                              : static_cast<uint32_t>(d);
    EXPECT_EQ(static_cast<uint32_t>(got), ref) << "v_cvt_u32_f64 of " << d;
  }
}

// --- 32-bit source -> f64 dest (high word must be written) --------------------

TEST(Vop1CvtF64Scalar, F64FromF32_HighWordWritten) {
  Fixture fx;
  for (float f : {3.7f, -3.7f, 123456.78f, -9.876e8f, 1e-10f, 2.5f}) {
    uint64_t got = fx.run(/*v_cvt_f64_f32=*/16, std::bit_cast<uint32_t>(f), 0u);
    uint64_t want = std::bit_cast<uint64_t>(static_cast<double>(f));
    EXPECT_EQ(got, want) << "v_cvt_f64_f32 of " << f;
  }
}

TEST(Vop1CvtF64Scalar, F64FromI32_HighWordWritten) {
  Fixture fx;
  for (int32_t i : {3, -3, 123456789, -987654321, INT32_MAX, INT32_MIN}) {
    uint64_t got = fx.run(/*v_cvt_f64_i32=*/4, static_cast<uint32_t>(i), 0u);
    uint64_t want = std::bit_cast<uint64_t>(static_cast<double>(i));
    EXPECT_EQ(got, want) << "v_cvt_f64_i32 of " << i;
  }
}

TEST(Vop1CvtF64Scalar, F64FromU32_HighWordWritten) {
  Fixture fx;
  for (uint32_t u : {3u, 123456789u, 0xFFFFFFFFu, 0x80000000u, 1u}) {
    uint64_t got = fx.run(/*v_cvt_f64_u32=*/22, u, 0u);
    uint64_t want = std::bit_cast<uint64_t>(static_cast<double>(u));
    EXPECT_EQ(got, want) << "v_cvt_f64_u32 of " << u;
  }
}

} // namespace
