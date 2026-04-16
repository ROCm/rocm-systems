// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file shared_infra_test.cpp
/// @brief Phase B unit tests: addr_calc, mfma_exec, wavefront context, CU factory.

#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mfma_exec.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <memory>

namespace {

using namespace rocjitsu;

// ---------------------------------------------------------------------------
// Concept and trait verification (compile-time)
// ---------------------------------------------------------------------------

static_assert(GpuIsa<cdna3::Isa>);
static_assert(GpuIsa<rdna4::Isa>);
static_assert(HasAccVgpr<cdna3::Isa>);
static_assert(!HasAccVgpr<rdna4::Isa>);
static_assert(HasMonolithicWaitcnt<cdna3::Isa>);
static_assert(!HasMonolithicWaitcnt<rdna4::Isa>);

// RDNA3/3.5 retain monolithic S_WAITCNT (GFX11 layout).
static_assert(HasMonolithicWaitcnt<rdna3::Isa>);

// RDNA2 supports Wave64 (WF_SIZE_MAX inherited as 64).
static_assert(rdna2::Isa::WF_SIZE_MAX == 64);

// CDNA1 has no AccVGPRs; CDNA2/3/4 have 256.
static_assert(cdna1::Isa::MAX_ACC_VGPRS_PER_WF == 0);
static_assert(cdna2::Isa::MAX_ACC_VGPRS_PER_WF == 256);
static_assert(cdna3::Isa::MAX_ACC_VGPRS_PER_WF == 256);

// ---------------------------------------------------------------------------
// MFMA register layout tests
// ---------------------------------------------------------------------------

TEST(MfmaExecTest, InputLocF32_32x32) {
  // v_mfma_f32_32x32x1f32: M=32, K=1, B=1, f32 inputs.
  // lanes_per_block = 64 / (32 * 1) = 2, elems_per_group = 1 / 2 = 0 -> special case.
  // Actually for M=32,K=2,B=1: lanes_per_block = 64/(32*1) = 2, elems = 2/2 = 1.
  // Use M=4, K=4, B=4 which is v_mfma_f32_4x4x4f16 (valid shape).
  // lanes_per_block = 64 / (4 * 4) = 4, elems_per_group = 4/4 = 1.
  auto loc = amdgpu::mfma::input_loc(4, 4, 4, /*i=*/2, /*k=*/0, /*b=*/0, 32);
  EXPECT_EQ(loc.vgpr_offset, 0u);
  EXPECT_EQ(loc.lane, 2u); // b*dim + ... = 0*4 + (0/1)*4*4 + 2 = 2
  EXPECT_EQ(loc.sub_element, 0u);
}

TEST(MfmaExecTest, InputLocF16_16x16) {
  // 16x16x16 with 1 block, f16 inputs: each lane holds 16 * 2B = 32B = 8 dwords.
  // lanes_per_block = 64 / (16 * 1) = 4
  // elems_per_group = 16 / 4 = 4
  // For i=0, k=0, b=0: local=0%4=0, lane=0*16+0*16*1+0=0, per_dword=2
  // vgpr_offset = 0/2 = 0, sub_element = 0%2 = 0
  auto loc = amdgpu::mfma::input_loc(16, 16, 1, 0, 0, 0, 16);
  EXPECT_EQ(loc.vgpr_offset, 0u);
  EXPECT_EQ(loc.lane, 0u);
  EXPECT_EQ(loc.sub_element, 0u);

  // k=1: local=1, vgpr_offset = 1/2 = 0, sub_element = 1
  auto loc1 = amdgpu::mfma::input_loc(16, 16, 1, 0, 1, 0, 16);
  EXPECT_EQ(loc1.vgpr_offset, 0u);
  EXPECT_EQ(loc1.sub_element, 1u);
}

TEST(MfmaExecTest, OutputLoc32_4x4) {
  // 4x4 matrix, block 0: reg = column index, lane = row index.
  auto loc = amdgpu::mfma::output_loc_32(4, 4, /*col=*/2, /*row=*/1, /*b=*/0);
  EXPECT_EQ(loc.reg, 2u);
  EXPECT_EQ(loc.lane, 1u);
}

TEST(MfmaExecTest, ResolveAccConstant) {
  // Encoding value 0-255 = inline constant. The callback should be invoked.
  uint32_t const_acc = 0;
  uint32_t result = amdgpu::mfma::resolve_acc<amdgpu::mfma::AccMode::Unified>(
      /*vb=*/100, /*dst=*/200, /*src2_ev=*/128, const_acc, [&]() -> uint32_t { return 42u; });
  EXPECT_EQ(const_acc, 42u);
  EXPECT_EQ(result, 200u); // Returns dst when constant.
}

TEST(MfmaExecTest, ResolveAccVgpr) {
  // Encoding value 256-511 = VGPR.
  uint32_t const_acc = 0;
  uint32_t result = amdgpu::mfma::resolve_acc<amdgpu::mfma::AccMode::Unified>(
      /*vb=*/100, /*dst=*/200, /*src2_ev=*/260, const_acc, [&]() -> uint32_t { return 99u; });
  EXPECT_EQ(const_acc, amdgpu::mfma::ACC_FROM_VGPR);
  EXPECT_EQ(result, 100u + 4u); // vb + (260 - 256)
}

TEST(MfmaExecTest, ResolveAccAccVgpr) {
  // Encoding value 768-1023 = AccVGPR (unified alias).
  uint32_t const_acc = 0;
  uint32_t result = amdgpu::mfma::resolve_acc<amdgpu::mfma::AccMode::Unified>(
      /*vb=*/100, /*dst=*/200, /*src2_ev=*/770, const_acc, [&]() -> uint32_t { return 99u; });
  EXPECT_EQ(const_acc, amdgpu::mfma::ACC_FROM_VGPR);
  EXPECT_EQ(result, 100u + 256u + 2u); // vb + ACC_VGPR_OFFSET + (770 - 768)
}

// ---------------------------------------------------------------------------
// CU factory tests — verify all 9 ISAs can be instantiated
// ---------------------------------------------------------------------------

class CuFactoryTest : public ::testing::TestWithParam<rj_code_arch_t> {};

TEST_P(CuFactoryTest, CreatesSuccessfully) {
  auto arch = GetParam();
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = arch;
  cfg.num_wf_slots = 2;
  cfg.sgprs_per_wf = 102;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create("test_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);
  EXPECT_EQ(cu->arch(), arch);
}

INSTANTIATE_TEST_SUITE_P(AllIsas, CuFactoryTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                           ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4,
                                           ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                           ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                           ROCJITSU_CODE_ARCH_RDNA4));

// ---------------------------------------------------------------------------
// DPP permutation tests
// ---------------------------------------------------------------------------

TEST(DppPermuteTest, QuadPerm) {
  using namespace amdgpu::dpp;
  // quad_perm(1,0,3,2) = swap pairs within each quad
  // Encoding: lane0->1, lane1->0, lane2->3, lane3->2
  // = (1 << 0) | (0 << 2) | (3 << 4) | (2 << 6) = 0xB1
  bool oob = false;
  EXPECT_EQ(dpp_permute(0xB1, 0, 64, oob), 1);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(0xB1, 1, 64, oob), 0);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(0xB1, 2, 64, oob), 3);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(0xB1, 3, 64, oob), 2);
  EXPECT_FALSE(oob);
  // Quad boundary: lane 4 starts a new quad, same permutation.
  EXPECT_EQ(dpp_permute(0xB1, 4, 64, oob), 5);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(0xB1, 5, 64, oob), 4);
  EXPECT_FALSE(oob);
}

TEST(DppPermuteTest, RowShr1) {
  using namespace amdgpu::dpp;
  bool oob = false;
  // row_shr 1 = 0x111: data shifts right, so lane K reads from lane K-1.
  EXPECT_EQ(dpp_permute(0x111, 1, 64, oob), 0);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(0x111, 15, 64, oob), 14);
  EXPECT_FALSE(oob);
  // Lane 0 (first in row) goes OOB (no lane -1).
  oob = false;
  dpp_permute(0x111, 0, 64, oob);
  EXPECT_TRUE(oob);
}

TEST(DppPermuteTest, RowShl1) {
  using namespace amdgpu::dpp;
  bool oob = false;
  // row_shl 1 = 0x101: data shifts left, so lane K reads from lane K+1.
  EXPECT_EQ(dpp_permute(0x101, 0, 64, oob), 1);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(0x101, 14, 64, oob), 15);
  EXPECT_FALSE(oob);
  // Lane 15 (last in row) goes OOB (no lane 16 in this row).
  oob = false;
  dpp_permute(0x101, 15, 64, oob);
  EXPECT_TRUE(oob);
}

TEST(DppPermuteTest, RowMirror) {
  using namespace amdgpu::dpp;
  bool oob = false;
  // row_mirror = 0x140: reverse lane order within a row.
  EXPECT_EQ(dpp_permute(0x140, 0, 64, oob), 15);
  EXPECT_EQ(dpp_permute(0x140, 15, 64, oob), 0);
  EXPECT_EQ(dpp_permute(0x140, 7, 64, oob), 8);
  // Second row.
  EXPECT_EQ(dpp_permute(0x140, 16, 64, oob), 31);
}

TEST(DppPermuteTest, RowXmask) {
  using namespace amdgpu::dpp;
  bool oob = false;
  // row_xmask with mask=1 = 0x151: XOR lane offset with 1 (swap adjacent pairs).
  EXPECT_EQ(dpp_permute(0x151, 0, 64, oob), 1);
  EXPECT_EQ(dpp_permute(0x151, 1, 64, oob), 0);
  EXPECT_EQ(dpp_permute(0x151, 2, 64, oob), 3);
  EXPECT_EQ(dpp_permute(0x151, 3, 64, oob), 2);
}

TEST(DppPermuteTest, DppRead) {
  using namespace amdgpu::dpp;
  // Set up 64 source values: src[i] = i * 10.
  uint32_t src[64];
  for (int i = 0; i < 64; ++i)
    src[i] = i * 10;

  // row_shr 1: lane 1 reads from lane 0.
  uint32_t val = dpp_read(src, 1, 64, 0x111, 0xF, 0xF, 1, 999);
  EXPECT_EQ(val, 0u); // src[0] = 0

  // Lane 5 reads from lane 4 (src[4] = 40).
  val = dpp_read(src, 5, 64, 0x111, 0xF, 0xF, 1, 999);
  EXPECT_EQ(val, 40u);

  // Lane 0 goes OOB, bound_ctrl=1 -> returns 0.
  val = dpp_read(src, 0, 64, 0x111, 0xF, 0xF, 1, 999);
  EXPECT_EQ(val, 0u);

  // Lane 0 goes OOB, bound_ctrl=0 -> returns old_val.
  val = dpp_read(src, 0, 64, 0x111, 0xF, 0xF, 0, 999);
  EXPECT_EQ(val, 999u);

  // Row mask disables row 0 (bits [3:0], row0 = lanes 0-15).
  val = dpp_read(src, 5, 64, 0x111, 0xE, 0xF, 1, 999);
  EXPECT_EQ(val, 999u); // row 0 masked -> old_val

  // Bank mask disables bank 1 (lanes 4-7 within each row).
  val = dpp_read(src, 5, 64, 0x111, 0xF, 0xD, 1, 999);
  EXPECT_EQ(val, 999u); // bank 1 disabled -> old_val

  // Unmasked lane in row 1: lane 17 reads from lane 16.
  val = dpp_read(src, 17, 64, 0x111, 0xF, 0xF, 1, 999);
  EXPECT_EQ(val, 160u); // src[16] = 160
}

// ---------------------------------------------------------------------------
// SDWA tests
// ---------------------------------------------------------------------------

TEST(SdwaTest, SrcSelect) {
  using namespace amdgpu::sdwa;
  uint32_t val = 0xDEADBEEF;

  EXPECT_EQ(sdwa_src_select(val, BYTE_0, false), 0xEFu);
  EXPECT_EQ(sdwa_src_select(val, BYTE_1, false), 0xBEu);
  EXPECT_EQ(sdwa_src_select(val, BYTE_2, false), 0xADu);
  EXPECT_EQ(sdwa_src_select(val, BYTE_3, false), 0xDEu);
  EXPECT_EQ(sdwa_src_select(val, WORD_0, false), 0xBEEFu);
  EXPECT_EQ(sdwa_src_select(val, WORD_1, false), 0xDEADu);
  EXPECT_EQ(sdwa_src_select(val, DWORD, false), val);

  // Sign extension.
  EXPECT_EQ(sdwa_src_select(0x00000080, BYTE_0, true), 0xFFFFFF80u);
  EXPECT_EQ(sdwa_src_select(0x00000080, BYTE_0, false), 0x80u);
  EXPECT_EQ(sdwa_src_select(0x00008000, WORD_0, true), 0xFFFF8000u);
}

TEST(SdwaTest, DstMerge) {
  using namespace amdgpu::sdwa;
  // Write result byte 0x42 into BYTE_1, zero-pad rest.
  uint32_t merged = sdwa_dst_merge(0x42, 0xAAAAAAAA, BYTE_1, UNUSED_PAD);
  EXPECT_EQ(merged, 0x00004200u);

  // Preserve unused bytes.
  merged = sdwa_dst_merge(0x42, 0xAABBCCDD, BYTE_1, UNUSED_PRESERVE);
  EXPECT_EQ(merged, 0xAABB42DDu);

  // Full dword: just return result.
  merged = sdwa_dst_merge(0x12345678, 0xAAAAAAAA, DWORD, UNUSED_PAD);
  EXPECT_EQ(merged, 0x12345678u);
}

// ---------------------------------------------------------------------------
// Scratch address calculation tests
// ---------------------------------------------------------------------------

TEST(ScratchAddrCalcTest, FlatScratchUsesWavefrontBase) {
  // Verify that FLAT with seg==1 (SCRATCH) computes:
  //   address = scratch_base + VGPR[lane] + offset
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 104;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("scratch_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 104, 16);
  ASSERT_NE(wf, nullptr);

  // Set scratch base to a known address.
  constexpr uint64_t SCRATCH_BASE = 0x1'0000'0000ULL;
  wf->set_scratch_base(SCRATCH_BASE);

  // Write a 32-bit offset into VGPR[0] lane 0.
  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase, 0, 0x100); // lane 0: offset 0x100

  // Set EXEC so only lane 0 is active.
  wf->set_exec(1ULL);

  // Build a FlatMachineInst with seg=1 (SCRATCH), saddr=0x7F (no SADDR),
  // offset=0x10.
  cdna4::FlatMachineInst inst{};
  inst.seg = 1;       // SCRATCH
  inst.saddr = 0x7F;  // No SADDR
  inst.addr = 0;      // VGPR index 0
  inst.offset = 0x10; // 12-bit immediate offset
  inst.pad_12 = 0;

  std::array<uint64_t, 64> addrs{};
  uint64_t lane_mask = 0;
  amdgpu::addr_calc::flat_calculate_addresses(inst, *wf, addrs, lane_mask);

  EXPECT_EQ(lane_mask, 1ULL);
  // scratch_base (0x1_0000_0000) + VGPR (0x100) + offset (0x10) = 0x1_0000_0110
  EXPECT_EQ(addrs[0], SCRATCH_BASE + 0x100 + 0x10);
}

TEST(ScratchAddrCalcTest, FlatGlobalDoesNotUseScratchBase) {
  // Verify that FLAT with seg==2 (GLOBAL) does NOT add scratch_base.
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 104;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("global_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 104, 16);
  ASSERT_NE(wf, nullptr);

  // Set scratch base — should be ignored for GLOBAL.
  wf->set_scratch_base(0xDEAD'0000ULL);

  // Write a 64-bit address into VGPR[0:1] lane 0.
  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase, 0, 0x2000);     // low 32
  cu->write_vgpr(vbase + 1, 0, 0x0001); // high 32 → addr = 0x1_0000_2000
  wf->set_exec(1ULL);

  cdna4::FlatMachineInst inst{};
  inst.seg = 2;      // GLOBAL
  inst.saddr = 0x7F; // No SADDR → use 64-bit VGPR pair
  inst.addr = 0;
  inst.offset = 0;
  inst.pad_12 = 0;

  std::array<uint64_t, 64> addrs{};
  uint64_t lane_mask = 0;
  amdgpu::addr_calc::flat_calculate_addresses(inst, *wf, addrs, lane_mask);

  EXPECT_EQ(addrs[0], 0x1'0000'2000ULL); // No scratch_base added.
}

} // namespace
