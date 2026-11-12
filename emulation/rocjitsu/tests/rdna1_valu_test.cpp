// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/lds.h"

#include <array>
#include <bit>
#include <gtest/gtest.h>
#include <memory>

namespace {
using namespace rocjitsu;

class Rdna1ValuTest : public ::testing::Test {
protected:
  void SetUp() override {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_RDNA1;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 256;
    cfg.lds_size_kb = 64;
    ASSERT_NO_FATAL_FAILURE(test::initialize_compute_unit(cu, "gfx1010", cfg, &memory, &cache));
    decoder = Decoder::create(cfg.arch);
    wf = cu->dispatch_wf(0, 0, 106, 8, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(1);
    wf->set_mode_raw(0xf0);
    cu->allocate_lds(256);
    wf->set_lds_base(cu->allocate_lds(256));
    wf->set_lds_size(256);
  }
  void TearDown() override {
    if (wf)
      wf->halt();
  }
  void execute(uint32_t first, uint32_t second = 0) {
    const std::array<uint32_t, 3> words{first, second, 0};
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
  }
  void set(unsigned reg, uint32_t value, unsigned lane = 0) {
    cu->write_vgpr(wf->vgpr_alloc().base + reg, lane, value);
  }
  uint32_t get(unsigned reg, unsigned lane = 0) {
    return cu->read_vgpr(wf->vgpr_alloc().base + reg, lane);
  }
  amdgpu::GpuMemory memory{"memory"};
  amdgpu::L2Cache cache{"cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;
};

TEST_F(Rdna1ValuTest, MadRoundsProductBeforeAddAndFlushesSubnormals) {
  constexpr uint32_t hi = 0x3f800001, lo = 0x3f7ffffe, minus_one = 0xbf800000;
  struct Case {
    uint32_t first, second;
    std::array<uint32_t, 4> inputs;
    uint32_t expected;
  };
  // (1 + 2^-23) * (1 - 2^-23) - 1 rounds to zero with an FP32 intermediate.
  const Case cases[] = {
      {0x3e000501, 0, {minus_one, hi, lo, 0}, 0},          // MAC
      {0x40000501, lo, {0, hi, minus_one, 0}, 0},          // MADMK
      {0x42000501, minus_one, {0, hi, lo, 0}, 0},          // MADAK
      {0xd5410000, 0x040e0501, {0, hi, lo, minus_one}, 0}, // MAD
      {0x56000501, 0, {minus_one, hi, lo, 0}, 0xa8800000}, // FMAC
      {0xd5410000, 0x040e0501, {0, 0x00800000, 0x3f000000, 0}, 0},
      {0x0c000501, 0, {0x3f800000, 0, 0x7f800000, 0}, 0x3f800000},          // MAC_LEGACY
      {0xd5400000, 0x040e0501, {0, 0, 0x7f800000, 0x3f800000}, 0x3f800000}, // MAD_LEGACY
  };
  for (const auto &c : cases) {
    SCOPED_TRACE(c.first);
    for (unsigned i = 0; i < c.inputs.size(); ++i)
      set(i, c.inputs[i]);
    ASSERT_NO_FATAL_FAILURE(execute(c.first, c.second));
    EXPECT_EQ(get(0), c.expected);
  }
}

TEST_F(Rdna1ValuTest, RowXmaskPermutesLanes) {
  set(1, 0x11223344);
  wf->set_exec(3);
  set(1, 0x99, 1);
  ASSERT_NO_FATAL_FAILURE(execute(0x7e0002fa, 0xff016101));
  EXPECT_EQ(get(0), 0x99u);
  EXPECT_EQ(get(0, 1), 0x11223344u);
  for (uint32_t ctrl : {0x130, 0x134, 0x138, 0x13c, 0x142, 0x143}) {
    std::array<uint32_t, 2> words{0x7e0002fa, 0xff000001 | (ctrl << 8)};
    EXPECT_TRUE(decode_fails(*decoder, words.data()));
  }
}

TEST_F(Rdna1ValuTest, StickyExceptionsFromHalfAddAndSignalingCompare) {
  set(1, 0x7c00);
  set(2, 0xfc00);
  wf->set_trapsts(0x100);
  ASSERT_NO_FATAL_FAILURE(execute(0x64000501)); // V_ADD_F16
  EXPECT_EQ(wf->trapsts(), 0x101u);
  ASSERT_NO_FATAL_FAILURE(execute(0x7e008200)); // V_CLREXCP
  EXPECT_EQ(wf->trapsts(), 0x100u);
  set(1, 0x7fc00000, 1);
  set(2, 0x3f800000, 1);
  ASSERT_NO_FATAL_FAILURE(execute(0xd402806a, 0x02020501)); // inactive QNaN
  EXPECT_EQ(wf->trapsts(), 0x100u);
  wf->set_exec(2);
  ASSERT_NO_FATAL_FAILURE(execute(0xd402006a, 0x02020501)); // quiet CMP
  EXPECT_EQ(wf->trapsts(), 0x100u);
  ASSERT_NO_FATAL_FAILURE(execute(0xd402806a, 0x02020501));
  EXPECT_EQ(wf->trapsts(), 0x101u);
}

TEST_F(Rdna1ValuTest, OutOfRangeVgprSourceReadsV0AndDestinationSuppressesCarry) {
  set(0, 0xcafef00d);
  ASSERT_NO_FATAL_FAILURE(execute(0x7e0203ff)); // V_MOV_B32 v1, v255
  EXPECT_EQ(get(1), 0xcafef00du);
  ASSERT_NO_FATAL_FAILURE(execute(0x7e0005ff)); // V_READFIRSTLANE_B32 s0, v255
  EXPECT_EQ(cu->read_sgpr(wf->sgpr_alloc().base), 0xcafef00du);
  // Only V0..V7 belong to this wave, despite the 256-register backing slot.
  set(255, 0x12345678);
  ASSERT_NO_FATAL_FAILURE(execute(0x7e0202fa, 0xff00e4ff)); // V_MOV_B32_DPP v1, v255
  EXPECT_EQ(get(1), 0xcafef00du);
  set(7, 0xaabbccdd);
  ASSERT_NO_FATAL_FAILURE(execute(0xd6ff0002, 0x02020e80)); // V_LSHLREV_B64 v[2:3], 0, v[7:8]
  EXPECT_EQ(get(2), 0xaabbccddu);
  EXPECT_EQ(get(3), 0xcafef00du);
  set(1, 0xffffffff);
  set(2, 1);
  wf->set_vcc(0);
  ASSERT_NO_FATAL_FAILURE(execute(0xd70f6aff, 0x02020501));
  EXPECT_EQ(wf->vcc(), 0u);
  set(1, 0x7c00);
  set(2, 0xfc00);
  wf->set_trapsts(0);
  ASSERT_NO_FATAL_FAILURE(execute(0x65fe0501)); // V_ADD_F16 v255, +Inf, -Inf
  EXPECT_EQ(wf->trapsts(), 0u);
  EXPECT_EQ(get(255), 0x12345678u);
}

TEST_F(Rdna1ValuTest, LdsDirectBroadcastIncludesSignedSubwordsAndReadfirstlane) {
  cu->lds().write32(wf->lds_base() + 16, 0x1234ff80);
  wf->set_m0((4u << 16) | 16);
  ASSERT_NO_FATAL_FAILURE(execute(0x7e0002fe)); // V_MOV_B32 v0, LDS_DIRECT
  EXPECT_EQ(get(0), 0xffffff80u);
  wf->set_exec(8);
  wf->set_m0((2u << 16) | 16);
  ASSERT_NO_FATAL_FAILURE(execute(0x7e0004fe));
  EXPECT_EQ(cu->read_sgpr(wf->sgpr_alloc().base), 0x1234ff80u);
}

TEST_F(Rdna1ValuTest, InterpolationSelectsPrimitiveChannelAndAccumulates) {
  constexpr uint32_t offset = 16;
  wf->set_m0((1u << 16) | offset); // quad 1 starts the second primitive
  wf->set_exec(0x11);
  for (unsigned prim = 0; prim < 2; ++prim) {
    const auto write = [&](unsigned slot, float value) {
      // Attribute 1 follows both primitive blocks of attribute 0.
      cu->lds().write32(wf->lds_base() + offset + 48 * (2 + prim) + slot * 4,
                        std::bit_cast<uint32_t>(value));
    };
    write(2, 2.0f + prim); // P0.y
    write(3, 4.0f + prim); // P10.y
    write(9, 6.0f + prim); // P20.y
    set(2, std::bit_cast<uint32_t>(2.f), 4 * prim);
  }
  set(5, 0xcafe, 1);
  auto p1 = rdna1::build_vintrp(0, {.vsrc = 2, .attrchan = 1, .attr = 1, .vdst = 5});
  ASSERT_NO_FATAL_FAILURE(execute(p1[0]));
  EXPECT_EQ(std::bit_cast<float>(get(5)), 10.f);
  EXPECT_EQ(std::bit_cast<float>(get(5, 4)), 13.f);
  ASSERT_NO_FATAL_FAILURE(execute(0xd6010005, 0x02020441)); // P2.e64, attr1.y
  EXPECT_EQ(std::bit_cast<float>(get(5)), 22.f);
  EXPECT_EQ(std::bit_cast<float>(get(5, 4)), 27.f);
  EXPECT_EQ(get(5, 1), 0xcafeu);
  auto mov = rdna1::build_vintrp(2, {.vsrc = 2, .attrchan = 1, .attr = 1, .vdst = 5});
  ASSERT_NO_FATAL_FAILURE(execute(mov[0])); // p0
  EXPECT_EQ(std::bit_cast<float>(get(5, 4)), 3.f);
}

TEST_F(Rdna1ValuTest, HalfInterpolationSelectsHighCoefficientsAndPreservesDestinationHalf) {
  wf->set_m0(0);
  cu->lds().write32(wf->lds_base(), 0x42003c00);      // P0.x = {1,3}
  cu->lds().write32(wf->lds_base() + 4, 0x44003c00);  // P10.x = {1,4}
  cu->lds().write32(wf->lds_base() + 32, 0x40003c00); // P20.x = {1,2}
  set(2, std::bit_cast<uint32_t>(2.f));
  ASSERT_NO_FATAL_FAILURE(execute(0xd7420005, 0x02020500)); // P1LL high
  EXPECT_EQ(std::bit_cast<float>(get(5)), 11.f);
  set(3, 0x45003c00);                                       // S2.high = 5
  ASSERT_NO_FATAL_FAILURE(execute(0xd7430005, 0x040e0500)); // P1LV high
  EXPECT_EQ(std::bit_cast<float>(get(5)), 13.f);
  set(3, std::bit_cast<uint32_t>(8.f));
  set(5, 0xdeadbeef);
  ASSERT_NO_FATAL_FAILURE(execute(0xd75a4005, 0x440e0500)); // P2 high, -v2, dst.high
  EXPECT_EQ(get(5), 0x4400beefu);                           // (2 * -2 + 8) = 4
}

} // namespace
