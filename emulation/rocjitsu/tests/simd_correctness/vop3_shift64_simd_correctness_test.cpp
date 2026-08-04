// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_shift64_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the 64-bit-lane
/// VOP3 shifts on CDNA4: the reverse shifts v_lshlrev_b64 / v_lshrrev_b64 /
/// v_ashrrev_i64 (shift the 64-bit src1 by the 32-bit src0, masked to [0,63])
/// and v_lshl_add_u64 ((src0 << (src1 & 63)) + src2). The mad64 coverage also
/// checks v_mad_u64_u32's explicit SDST carry-out mask. These use the new
/// mixed-width 64-bit shift glue (32-bit count + 64-bit value); shifts produce
/// no NaN so every value is compared exactly. Each case runs TWICE in the same
/// process -- once forcing the scalar body, once the SIMD fast path, with
/// identical inputs/EXEC -- and the f64 dst results are asserted equal with
/// EXPECT_EQ (util::set_force_scalar_for_testing flips the gate in-process).
/// The wide MAD cases also compare the explicit SGPR-pair carry/overflow mask.
/// In-process inactive lanes must keep the sentinel.

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
#include <limits>
#include <memory>
#include <string>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t kDstVgpr = 6; // v6:v7
constexpr uint64_t SDST_SENTINEL = 0xDEADBEEFCAFEF00DULL;

constexpr uint64_t dst_sentinel(uint32_t lane) {
  return (static_cast<uint64_t>(0xCAFE0000u | lane) << 32) | (0xBEEF0000u + lane);
}

// VOP3 ([31:26]=0x34): word0 = vdst | op<<16 | enc; word1 = src0 | src1<<9 |
// src2<<18. No modifiers (integer shifts ignore abs/neg/omod/clamp).
constexpr void vop3_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1, uint32_t src2,
                           uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((op & 0x3FF) << 16) | (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((src2 & 0x1FF) << 18);
}

// VOP3_SDST_ENC: word0 also carries sdst[14:8].
constexpr void vop3_sdstenc_encode(uint32_t op, uint32_t vdst, uint32_t sdst, uint32_t src0,
                                   uint32_t src1, uint32_t src2, uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((sdst & 0x7F) << 8) | ((op & 0x3FF) << 16) | (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((src2 & 0x1FF) << 18);
}

// 64-bit values: sign bit set/clear, all-ones, alternating, powers of two.
const std::array<uint64_t, 12> kVals64 = {{
    0x0000000000000000ull,
    0x0000000000000001ull,
    0xFFFFFFFFFFFFFFFFull,
    0x8000000000000000ull,
    0x7FFFFFFFFFFFFFFFull,
    0x123456789ABCDEF0ull,
    0xDEADBEEFCAFEBABEull,
    0xA5A5A5A5A5A5A5A5ull,
    0x0000000100000000ull,
    0xFFFFFFFF00000000ull,
    0x00000000FFFFFFFFull,
    0x5A5A5A5A5A5A5A5Aull,
}};
// Shift counts: include <64, ==0, and >=64 (exercises the &63 mask vs scalar).
const std::array<uint32_t, 10> kShifts = {{0u, 1u, 7u, 31u, 32u, 33u, 63u, 64u, 100u, 0xFFFFFFFFu}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3_shift64_mem"), l2("vop3_shift64_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_shift64", cfg, &gpu_mem, &l2);
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
  void write_sgpr64(uint32_t reg, uint64_t v) {
    cu->write_sgpr(reg, static_cast<uint32_t>(v));
    cu->write_sgpr(reg + 1, static_cast<uint32_t>(v >> 32));
  }
  uint64_t read_sgpr64(uint32_t reg) {
    return static_cast<uint64_t>(cu->read_sgpr(reg + 1)) << 32 | cu->read_sgpr(reg);
  }
};

// Restores the process force-scalar gate on scope exit so flipping it for an
// in-process A/B comparison cannot leak into later tests in the same process.
struct ForceScalarGuard {
  bool orig;
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }
};

void compare_and_check_inactive(const char *name, uint64_t exec,
                                const std::array<uint64_t, WF_SIZE> &scalar_out,
                                const std::array<uint64_t, WF_SIZE> &simd_out,
                                const std::string &sub) {
  EXPECT_EQ(scalar_out, simd_out) << name << " " << sub << ": SIMD path diverged from scalar body";
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (exec >> lane) & 1ULL;
    if (!active) {
      EXPECT_EQ(simd_out[lane], dst_sentinel(lane)) << name << ": clobbered inactive lane " << lane;
      EXPECT_EQ(scalar_out[lane], dst_sentinel(lane))
          << name << ": clobbered inactive lane " << lane;
    }
  }
}

// Reverse shifts: src0 = v0 (32-bit count), src1 = v2:v3 (64-bit value),
// dst = v6:v7.
void check_revshift(const char *name, uint32_t op, uint64_t exec) {
  ForceScalarGuard gate_guard;
  auto run = [&](bool force_scalar, uint32_t srot, uint32_t vrot) -> std::array<uint64_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[2] = {0u, 0u};
    vop3_encode(op, kDstVgpr, /*src0=*/256, /*src1=*/258, /*src2=*/0, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << name << " decode failed";
    uint32_t vb = fx.wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      fx.cu->write_vgpr(vb + 0, lane, kShifts[(lane + srot) % kShifts.size()]);
      fx.write64(vb + 2, lane, kVals64[(lane + vrot) % kVals64.size()]);
      fx.write64(vb + kDstVgpr, lane, dst_sentinel(lane));
    }
    fx.wf->set_exec(exec);
    fx.cu->execute_instruction(inst, *fx.wf);
    std::array<uint64_t, WF_SIZE> out{};
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = fx.read64(vb + kDstVgpr, lane);
    delete inst;
    return out;
  };
  for (uint32_t srot = 0; srot < kShifts.size(); ++srot)
    for (uint32_t vrot = 0; vrot < kVals64.size(); vrot += 3) {
      const auto scalar_out = run(/*force_scalar=*/true, srot, vrot);
      const auto simd_out = run(/*force_scalar=*/false, srot, vrot);
      compare_and_check_inactive(name, exec, scalar_out, simd_out,
                                 "s" + std::to_string(srot) + ":v" + std::to_string(vrot));
    }
}

// v_lshl_add_u64: src0 = v0:v1 (64-bit value), src1 = v2 (32-bit shift),
// src2 = v4:v5 (64-bit addend), dst = v6:v7.
void check_lshl_add(uint64_t exec) {
  ForceScalarGuard gate_guard;
  auto run = [&](bool force_scalar, uint32_t srot, uint32_t crot) -> std::array<uint64_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[2] = {0u, 0u};
    vop3_encode(/*op=*/520, kDstVgpr, /*src0=*/256, /*src1=*/258, /*src2=*/260, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << "v_lshl_add_u64 decode failed";
    uint32_t vb = fx.wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      fx.write64(vb + 0, lane, kVals64[lane % kVals64.size()]);
      fx.cu->write_vgpr(vb + 2, lane, kShifts[(lane + srot) % kShifts.size()]);
      fx.write64(vb + 4, lane, kVals64[(lane + crot) % kVals64.size()]);
      fx.write64(vb + kDstVgpr, lane, dst_sentinel(lane));
    }
    fx.wf->set_exec(exec);
    fx.cu->execute_instruction(inst, *fx.wf);
    std::array<uint64_t, WF_SIZE> out{};
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = fx.read64(vb + kDstVgpr, lane);
    delete inst;
    return out;
  };
  for (uint32_t srot = 0; srot < kShifts.size(); ++srot)
    for (uint32_t crot = 0; crot < kVals64.size(); crot += 3) {
      const auto scalar_out = run(/*force_scalar=*/true, srot, crot);
      const auto simd_out = run(/*force_scalar=*/false, srot, crot);
      compare_and_check_inactive("v_lshl_add_u64", exec, scalar_out, simd_out,
                                 "s" + std::to_string(srot) + ":c" + std::to_string(crot));
    }
}

struct Mad64Result {
  uint64_t value;
  bool carry;
};

Mad64Result expected_mad_u64_u32(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
  uint64_t value = product + c;
  return {value, value < product};
}

Mad64Result expected_mad_i64_i32(uint32_t a, uint32_t b, uint64_t c) {
  int64_t lhs = static_cast<int64_t>(static_cast<int32_t>(a));
  int64_t rhs = static_cast<int64_t>(static_cast<int32_t>(b));
  int64_t product_signed = lhs * rhs;
  uint64_t product = static_cast<uint64_t>(product_signed);
  uint64_t value = product + c;
  int64_t addend = static_cast<int64_t>(c);
  bool overflow = (addend > 0 && product_signed > std::numeric_limits<int64_t>::max() - addend) ||
                  (addend < 0 && product_signed < std::numeric_limits<int64_t>::min() - addend);
  return {value, overflow};
}

uint64_t expected_mad64_sdst(bool is_signed, uint64_t exec, uint32_t arot, uint32_t crot) {
  uint64_t sdst = 0;
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    if (((exec >> lane) & 1ULL) == 0)
      continue;
    uint32_t a = kShifts[lane % kShifts.size()];
    uint32_t b = kShifts[(lane + arot) % kShifts.size()];
    uint64_t c = kVals64[(lane + crot) % kVals64.size()];
    const auto result = is_signed ? expected_mad_i64_i32(a, b, c) : expected_mad_u64_u32(a, b, c);
    if (result.carry)
      sdst |= (1ULL << lane);
  }
  return sdst;
}

struct Mad64Output {
  std::array<uint64_t, WF_SIZE> dst{};
  uint64_t sdst = 0;
};

// v_mad_u64_u32 / v_mad_i64_i32: src0 = v0 (32-bit), src1 = v2 (32-bit),
// src2 = v4:v5 (64-bit addend), dst = v6:v7.
void check_mad64(const char *name, uint32_t op, bool is_signed, uint64_t exec) {
  ForceScalarGuard gate_guard;
  auto run = [&](bool force_scalar, uint32_t arot, uint32_t crot) -> Mad64Output {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t sb = fx.wf->sgpr_alloc().base;
    EXPECT_EQ(sb % 2u, 0u) << name << ": sgpr_alloc base not pair-aligned";
    uint32_t words[2] = {0u, 0u};
    vop3_sdstenc_encode(op, kDstVgpr, sb, /*src0=*/256, /*src1=*/258, /*src2=*/260, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << name << " decode failed";
    uint32_t vb = fx.wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      fx.cu->write_vgpr(vb + 0, lane, kShifts[lane % kShifts.size()]);
      fx.cu->write_vgpr(vb + 2, lane, kShifts[(lane + arot) % kShifts.size()]);
      fx.write64(vb + 4, lane, kVals64[(lane + crot) % kVals64.size()]);
      fx.write64(vb + kDstVgpr, lane, dst_sentinel(lane));
    }
    fx.write_sgpr64(sb, SDST_SENTINEL);
    fx.wf->set_exec(exec);
    fx.cu->execute_instruction(inst, *fx.wf);
    Mad64Output out{};
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out.dst[lane] = fx.read64(vb + kDstVgpr, lane);
    out.sdst = fx.read_sgpr64(sb);
    delete inst;
    return out;
  };
  for (uint32_t arot = 0; arot < kShifts.size(); ++arot)
    for (uint32_t crot = 0; crot < kVals64.size(); crot += 3) {
      const auto scalar_out = run(/*force_scalar=*/true, arot, crot);
      const auto simd_out = run(/*force_scalar=*/false, arot, crot);
      const uint64_t expected_sdst = expected_mad64_sdst(is_signed, exec, arot, crot);
      EXPECT_EQ(scalar_out.sdst, expected_sdst)
          << name << ": scalar sdst mismatch for a" << arot << ":c" << crot;
      EXPECT_EQ(simd_out.sdst, expected_sdst)
          << name << ": SIMD sdst mismatch for a" << arot << ":c" << crot;
      compare_and_check_inactive(name, exec, scalar_out.dst, simd_out.dst,
                                 "a" + std::to_string(arot) + ":c" + std::to_string(crot));
      EXPECT_EQ(scalar_out.sdst, simd_out.sdst)
          << name << " a" << arot << ":c" << crot << ": SIMD SDST diverged from scalar body";
    }
}

TEST(Vop3Shift64SimdCorrectness, MadWide64) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check_mad64("v_mad_u64_u32_vop3", 488, /*is_signed=*/false, ~0ULL);
  check_mad64("v_mad_i64_i32_vop3", 489, /*is_signed=*/true, ~0ULL);
  check_mad64("v_mad_u64_u32_vop3", 488, /*is_signed=*/false, 0xA5A5'F0F0'1234'8001ULL);
  check_mad64("v_mad_i64_i32_vop3", 489, /*is_signed=*/true, 0xA5A5'F0F0'1234'8001ULL);
}

TEST(Vop3Shift64SimdCorrectness, RevShiftsFullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check_revshift("v_lshlrev_b64_vop3", 655, ~0ULL);
  check_revshift("v_lshrrev_b64_vop3", 656, ~0ULL);
  check_revshift("v_ashrrev_i64_vop3", 657, ~0ULL);
}

TEST(Vop3Shift64SimdCorrectness, RevShiftsPartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  const uint64_t exec = 0xA5A5'F0F0'1234'8001ULL;
  check_revshift("v_lshlrev_b64_vop3", 655, exec);
  check_revshift("v_lshrrev_b64_vop3", 656, exec);
  check_revshift("v_ashrrev_i64_vop3", 657, exec);
}

TEST(Vop3Shift64SimdCorrectness, LshlAddFullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check_lshl_add(~0ULL);
}

TEST(Vop3Shift64SimdCorrectness, LshlAddPartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check_lshl_add(0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
