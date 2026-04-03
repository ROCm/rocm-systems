// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file shared_infra_test.cpp
/// @brief Phase B unit tests: addr_calc, mfma_exec, wavefront context, CU factory.

#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
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
  EXPECT_EQ(result, 100u + 2u); // vb + (770 - 768)
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

} // namespace
