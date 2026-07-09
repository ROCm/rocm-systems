// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cdna3_predicate_mask_test.cpp
/// @brief CDNA3 predicate-mask semantics from hip-fpsan algebraic kernels.

#include "util/simd_test_hooks.h"

#include "rocjitsu/code/rj_code.h"
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

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 64;
constexpr uint32_t DST_SENTINEL = 0x13579BDFu;

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("cdna3_predicate_mask_mem"), l2("cdna3_predicate_mask_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA3;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_cdna3_predicate_mask", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void write_vgpr(uint32_t reg, uint32_t lane, uint32_t val) {
    cu->write_vgpr(wf->vgpr_alloc().base + reg, lane, val);
  }

  uint32_t read_vgpr(uint32_t reg, uint32_t lane) const {
    return cu->read_vgpr(wf->vgpr_alloc().base + reg, lane);
  }

  uint64_t read_vgpr64(uint32_t reg, uint32_t lane) const {
    const uint32_t lo = read_vgpr(reg, lane);
    const uint32_t hi = read_vgpr(reg + 1, lane);
    return (static_cast<uint64_t>(hi) << 32) | lo;
  }

  void write_vgpr64(uint32_t reg, uint32_t lane, uint64_t val) {
    write_vgpr(reg, lane, static_cast<uint32_t>(val));
    write_vgpr(reg + 1, lane, static_cast<uint32_t>(val >> 32));
  }

  void write_sgpr_pair(uint32_t reg, uint64_t val) {
    const uint32_t sb = wf->sgpr_alloc().base;
    cu->write_sgpr(sb + reg, static_cast<uint32_t>(val));
    cu->write_sgpr(sb + reg + 1, static_cast<uint32_t>(val >> 32));
  }

  uint64_t read_sgpr_pair(uint32_t reg) const {
    const uint32_t sb = wf->sgpr_alloc().base;
    const uint32_t lo = cu->read_sgpr(sb + reg);
    const uint32_t hi = cu->read_sgpr(sb + reg + 1);
    return (static_cast<uint64_t>(hi) << 32) | lo;
  }

  void execute(const std::array<uint32_t, 2> &words) {
    std::unique_ptr<Instruction> inst(decoder->decode(words.data()));
    ASSERT_NE(inst, nullptr);
    cu->execute_instruction(inst.get(), *wf);
  }
};

struct ForceScalarGuard {
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }

  bool orig;
};

uint64_t expected_ge_mask(uint64_t exec, const Fixture &fx, uint32_t lhs_reg, uint32_t rhs_reg) {
  uint64_t mask = 0;
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    if (((exec >> lane) & 1ULL) == 0)
      continue;
    if (fx.read_vgpr(lhs_reg, lane) >= fx.read_vgpr(rhs_reg, lane))
      mask |= 1ULL << lane;
  }
  return mask;
}

TEST(Cdna3PredicateMaskTest, Vop3CompareWritesArbitrarySgprPair) {
  // v_cmp_ge_u32_e64 s[2:3], v5, v23
  constexpr std::array<uint32_t, 2> kCmpGeU32S2 = {0xD0CE0002u, 0x00022F05u};
  constexpr uint64_t kExec = 0xF0F0'0F0F'A5A5'8001ULL;
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);

  const std::array<uint32_t, 8> lhs = {{
      0x00000000u,
      0x00000001u,
      0x7fffffffu,
      0x80000000u,
      0xff68043au,
      0xff68043bu,
      0xff68043cu,
      0xffffffffu,
  }};
  const std::array<uint32_t, 8> rhs = {{
      0x00000001u,
      0x00000001u,
      0x80000000u,
      0x7fffffffu,
      0xff68043bu,
      0xff68043bu,
      0xff68043bu,
      0xffffffffu,
  }};

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    fx.write_vgpr(5, lane, lhs[lane % lhs.size()]);
    fx.write_vgpr(23, lane, rhs[lane % rhs.size()]);
  }
  fx.wf->set_exec(kExec);
  fx.write_sgpr_pair(2, 0xDEAD'BEEF'DEAD'BEEFULL);

  const uint64_t expected = expected_ge_mask(kExec, fx, 5, 23);
  fx.execute(kCmpGeU32S2);

  EXPECT_EQ(fx.read_sgpr_pair(2), expected);
}

TEST(Cdna3PredicateMaskTest, Vop3CndmaskReadsArbitrarySgprPair) {
  // v_cndmask_b32_e64 v33, 0, -1, s[2:3]
  constexpr std::array<uint32_t, 2> kCndmaskS2 = {0xD1000021u, 0x00098280u};
  constexpr uint64_t kExec = 0xA5A5'F0F0'1234'8001ULL;
  constexpr uint64_t kSel = 0x0123'4567'89AB'CDEFULL;
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
    fx.write_vgpr(33, lane, DST_SENTINEL);
  fx.wf->set_exec(kExec);
  fx.write_sgpr_pair(2, kSel);

  fx.execute(kCndmaskS2);

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = ((kExec >> lane) & 1ULL) != 0;
    const bool selected = ((kSel >> lane) & 1ULL) != 0;
    const uint32_t expected = active ? (selected ? 0xFFFFFFFFu : 0u) : DST_SENTINEL;
    EXPECT_EQ(fx.read_vgpr(33, lane), expected) << "lane " << lane;
  }
}

TEST(Cdna3PredicateMaskTest, Vop3CompareThenCndmaskMatchesMask) {
  // Sequence from the hip-fpsan parity probe:
  //   v_cmp_ge_u32_e64 s[2:3], v5, v23
  //   v_cndmask_b32_e64 v33, 0, -1, s[2:3]
  constexpr std::array<uint32_t, 2> kCmpGeU32S2 = {0xD0CE0002u, 0x00022F05u};
  constexpr std::array<uint32_t, 2> kCndmaskS2 = {0xD1000021u, 0x00098280u};
  constexpr uint64_t kExec = 0xFFFF'FFFF'FFFF'FFFFULL;
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const uint32_t lhs = (lane % 4 == 0)   ? 0xff68043au
                         : (lane % 4 == 1) ? 0xff68043bu
                         : (lane % 4 == 2) ? 0xff68043cu
                                           : 0x00000000u;
    const uint32_t rhs = (lane % 4 == 3) ? 0xffffffffu : 0xff68043bu;
    fx.write_vgpr(5, lane, lhs);
    fx.write_vgpr(23, lane, rhs);
    fx.write_vgpr(33, lane, DST_SENTINEL);
  }
  fx.wf->set_exec(kExec);
  fx.write_sgpr_pair(2, 0);

  const uint64_t expected_mask = expected_ge_mask(kExec, fx, 5, 23);
  fx.execute(kCmpGeU32S2);
  EXPECT_EQ(fx.read_sgpr_pair(2), expected_mask);

  fx.execute(kCndmaskS2);
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const uint32_t expected = ((expected_mask >> lane) & 1ULL) ? 0xFFFFFFFFu : 0u;
    EXPECT_EQ(fx.read_vgpr(33, lane), expected) << "lane " << lane;
  }
}

TEST(Cdna3PredicateMaskTest, Vop3CompareU64WithScalarSource) {
  // v_cmp_eq_u64_e64 s[0:1], s[2:3], v[6:7]
  constexpr std::array<uint32_t, 2> kCmpEqU64S0 = {0xD0EA0000u, 0x00020C02u};
  constexpr uint64_t kExec = 0x0F0F'F0F0'AAAA'5555ULL;
  constexpr uint64_t kModulus = 0x00000000FF68043BULL;
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const uint64_t val = (lane % 4 == 0)   ? kModulus
                         : (lane % 4 == 1) ? kModulus - 1
                         : (lane % 4 == 2) ? kModulus + 1
                                           : 0;
    fx.write_vgpr64(6, lane, val);
  }
  fx.wf->set_exec(kExec);
  fx.write_sgpr_pair(0, 0xDEAD'BEEF'DEAD'BEEFULL);
  fx.write_sgpr_pair(2, kModulus);

  uint64_t expected = 0;
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    if (((kExec >> lane) & 1ULL) && (lane % 4 == 0))
      expected |= 1ULL << lane;
  }

  fx.execute(kCmpEqU64S0);

  EXPECT_EQ(fx.read_sgpr_pair(0), expected);
  EXPECT_EQ(fx.read_sgpr_pair(2), kModulus);
}

TEST(Cdna3PredicateMaskTest, VopcGtU64E32ScalarSourceWritesExpectedVcc) {
  // v_cmp_gt_u64_e32 vcc, s[2:3], v[8:9]
  constexpr std::array<uint32_t, 2> kCmpGtU64Vcc = {0x7DD81002u, 0u};
  constexpr uint64_t kExec = 0xF0F0'0F0F'A5A5'8001ULL;
  constexpr uint64_t kModulus = 0x00000000FF65CF7BULL;
  ForceScalarGuard guard;

  for (bool force_scalar : {true, false}) {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const uint64_t val = (lane % 8 == 0)   ? kModulus - 1
                           : (lane % 8 == 1) ? kModulus
                           : (lane % 8 == 2) ? kModulus + 1
                           : (lane % 8 == 3) ? 0
                           : (lane % 8 == 4) ? 1
                           : (lane % 8 == 5) ? 0xFFFF'FFFFULL
                           : (lane % 8 == 6) ? 0x1'0000'0000ULL
                                             : 0xFFFF'FFFF'FFFF'FFFFULL;
      fx.write_vgpr64(8, lane, val);
    }
    fx.wf->set_exec(kExec);
    fx.wf->set_vcc(0xDEAD'BEEF'DEAD'BEEFULL);
    fx.write_sgpr_pair(2, kModulus);

    uint64_t expected = 0;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      if (((kExec >> lane) & 1ULL) && kModulus > fx.read_vgpr64(8, lane))
        expected |= 1ULL << lane;
    }

    fx.execute(kCmpGtU64Vcc);

    EXPECT_EQ(fx.wf->vcc(), expected) << "force_scalar=" << force_scalar;
    EXPECT_EQ(fx.read_sgpr_pair(2), kModulus) << "force_scalar=" << force_scalar;
  }
}

TEST(Cdna3PredicateMaskTest, AddmodCompareCndmaskLshlAddSequence) {
  // Sequence emitted for alg_addmod's `(n > s) ? s : s - n` selection:
  //   v_cmp_gt_u64_e32 vcc, s[2:3], v[8:9]
  //   v_cndmask_b32_e64 v5, -1, 0, vcc
  //   v_cndmask_b32_e64 v4, v1, 0, vcc
  //   v_lshl_add_u64 v[4:5], v[4:5], 0, v[8:9]
  constexpr std::array<uint32_t, 2> kCmpGtU64Vcc = {0x7DD81002u, 0u};
  constexpr std::array<uint32_t, 2> kCndHigh = {0xD1000005u, 0x01A900C1u};
  constexpr std::array<uint32_t, 2> kCndLow = {0xD1000004u, 0x01A90101u};
  constexpr std::array<uint32_t, 2> kLshlAdd = {0xD2080004u, 0x04210104u};
  constexpr uint64_t kExec = 0xF0F0'0F0F'A5A5'8001ULL;
  constexpr uint64_t kModulus = 0x00000000FF65CF7BULL;
  constexpr uint64_t kNegMod = 0xFFFF'FFFF'009A3085ULL;
  constexpr uint64_t kSentinel = 0x13579BDF2468ACE0ULL;
  ForceScalarGuard guard;

  for (bool force_scalar : {true, false}) {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const uint64_t s = (lane % 8 == 0)   ? kModulus - 1
                         : (lane % 8 == 1) ? kModulus
                         : (lane % 8 == 2) ? kModulus + 1
                         : (lane % 8 == 3) ? 0
                         : (lane % 8 == 4) ? 1
                         : (lane % 8 == 5) ? (2 * kModulus) - 1
                         : (lane % 8 == 6) ? 0x1'0000'0000ULL
                                           : 0xFFFF'FFFF'FFFF'FFFFULL;
      fx.write_vgpr64(8, lane, s);
      fx.write_vgpr64(4, lane, kSentinel);
    }
    fx.wf->set_exec(kExec);
    fx.wf->set_vcc(0);
    fx.write_sgpr_pair(2, kModulus);
    fx.write_vgpr(1, 0, static_cast<uint32_t>(kNegMod));
    for (uint32_t lane = 1; lane < WF_SIZE; ++lane)
      fx.write_vgpr(1, lane, static_cast<uint32_t>(kNegMod));

    fx.execute(kCmpGtU64Vcc);
    fx.execute(kCndHigh);
    fx.execute(kCndLow);
    fx.execute(kLshlAdd);

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const uint64_t s = fx.read_vgpr64(8, lane);
      const uint64_t expected =
          ((kExec >> lane) & 1ULL) == 0 ? kSentinel : (kModulus > s ? s : s + kNegMod);
      EXPECT_EQ(fx.read_vgpr64(4, lane), expected)
          << "force_scalar=" << force_scalar << " lane " << lane;
    }
  }
}

TEST(Cdna3PredicateMaskTest, AddmodFullLshlAddReductionSequence) {
  // Full inline alg_addmod arithmetic from the probe:
  //   v_lshl_add_u64 v[8:9], v[2:3], 0, v[28:29]  ; s = a + b
  //   v_cmp_gt_u64_e32 vcc, s[2:3], v[8:9]
  //   v_cndmask_b32_e64 v5, -1, 0, vcc
  //   v_cndmask_b32_e64 v4, v1, 0, vcc
  //   v_lshl_add_u64 v[4:5], v[4:5], 0, v[8:9]
  constexpr std::array<uint32_t, 2> kAddInputs = {0xD2080008u, 0x04710102u};
  constexpr std::array<uint32_t, 2> kCmpGtU64Vcc = {0x7DD81002u, 0u};
  constexpr std::array<uint32_t, 2> kCndHigh = {0xD1000005u, 0x01A900C1u};
  constexpr std::array<uint32_t, 2> kCndLow = {0xD1000004u, 0x01A90101u};
  constexpr std::array<uint32_t, 2> kReduce = {0xD2080004u, 0x04210104u};
  constexpr uint64_t kExec = 0xF0F0'0F0F'A5A5'8001ULL;
  constexpr uint64_t kModulus = 0x00000000FF65CF7BULL;
  constexpr uint64_t kNegMod = 0xFFFF'FFFF'009A3085ULL;
  constexpr uint64_t kSentinel = 0x13579BDF2468ACE0ULL;
  ForceScalarGuard guard;

  for (bool force_scalar : {true, false}) {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const uint64_t a = (lane % 8 == 0)   ? 0
                         : (lane % 8 == 1) ? 1
                         : (lane % 8 == 2) ? 2
                         : (lane % 8 == 3) ? kModulus - 1
                         : (lane % 8 == 4) ? kModulus - 2
                         : (lane % 8 == 5) ? 0x1'0000'0000ULL
                         : (lane % 8 == 6) ? 0x7FFF'FFFFULL
                                           : 0xFFFF'FFFFULL;
      const uint64_t b = (lane % 8 == 0)   ? kModulus - 1
                         : (lane % 8 == 1) ? kModulus - 1
                         : (lane % 8 == 2) ? kModulus - 1
                         : (lane % 8 == 3) ? 1
                         : (lane % 8 == 4) ? 3
                         : (lane % 8 == 5) ? 1
                         : (lane % 8 == 6) ? 0x8000'0000ULL
                                           : 2;
      fx.write_vgpr64(2, lane, a);
      fx.write_vgpr64(28, lane, b);
      fx.write_vgpr64(4, lane, kSentinel);
      fx.write_vgpr64(8, lane, kSentinel);
    }
    fx.wf->set_exec(kExec);
    fx.wf->set_vcc(0);
    fx.write_sgpr_pair(2, kModulus);
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      fx.write_vgpr(1, lane, static_cast<uint32_t>(kNegMod));

    fx.execute(kAddInputs);
    fx.execute(kCmpGtU64Vcc);
    fx.execute(kCndHigh);
    fx.execute(kCndLow);
    fx.execute(kReduce);

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = ((kExec >> lane) & 1ULL) != 0;
      const uint64_t a = fx.read_vgpr64(2, lane);
      const uint64_t b = fx.read_vgpr64(28, lane);
      const uint64_t s = a + b;
      const uint64_t expected_sum = active ? s : kSentinel;
      const uint64_t expected_reduced = active ? (kModulus > s ? s : s + kNegMod) : kSentinel;
      EXPECT_EQ(fx.read_vgpr64(8, lane), expected_sum)
          << "force_scalar=" << force_scalar << " lane " << lane;
      EXPECT_EQ(fx.read_vgpr64(4, lane), expected_reduced)
          << "force_scalar=" << force_scalar << " lane " << lane;
    }
  }
}

} // namespace
