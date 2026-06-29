// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file shared_infra_test.cpp
/// @brief Phase B unit tests: addr_calc, MMA execution, wavefront context, CU factory.

#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna1/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna1/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/cdna1/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/isa.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/operand.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/sopk.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include "simdojo/sim/simulation.h"
#include "util/bit.h"
#include "util/data_types.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace {

using namespace rocjitsu;

// ---------------------------------------------------------------------------
// Concept and trait verification (compile-time)
// ---------------------------------------------------------------------------

static_assert(GpuIsa<cdna3::Isa>);
static_assert(GpuIsa<gfx1250::Isa>);
static_assert(GpuIsa<rdna4::Isa>);
static_assert(HasAccVgpr<cdna2::Isa>);
static_assert(HasAccVgpr<cdna3::Isa>);
static_assert(HasAccVgpr<cdna4::Isa>);
static_assert(!HasAccVgpr<gfx1250::Isa>);
static_assert(!HasAccVgpr<rdna4::Isa>);
static_assert(HasMonolithicWaitcnt<cdna3::Isa>);
static_assert(!HasMonolithicWaitcnt<gfx1250::Isa>);
static_assert(!HasMonolithicWaitcnt<rdna4::Isa>);

// RDNA3/3.5 retain monolithic S_WAITCNT (GFX11 layout).
static_assert(HasMonolithicWaitcnt<rdna3::Isa>);

// RDNA2 supports Wave64 (WF_SIZE_MAX inherited as 64).
static_assert(rdna2::Isa::WF_SIZE_MAX == 64);
static_assert(gfx1250::Isa::WF_SIZE_MAX == 32);

// CDNA1 and GFX1250 have no AccVGPRs; CDNA2/3/4 have 256.
constexpr uint32_t kCdnaAccVgprsPerWf = cdna2::Isa::MAX_ACC_VGPRS_PER_WF;
static_assert(cdna1::Isa::MAX_ACC_VGPRS_PER_WF == 0);
static_assert(kCdnaAccVgprsPerWf == 256);
static_assert(cdna3::Isa::MAX_ACC_VGPRS_PER_WF == kCdnaAccVgprsPerWf);
static_assert(cdna4::Isa::MAX_ACC_VGPRS_PER_WF == kCdnaAccVgprsPerWf);
static_assert(gfx1250::Isa::MAX_ACC_VGPRS_PER_WF == 0);

class Rdna3MemoryTestCu
    : public amdgpu::IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, rdna3::Isa> {
public:
  using Base = amdgpu::IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, rdna3::Isa>;

  Rdna3MemoryTestCu(std::string name, const amdgpu::ComputeUnitCore::Config &config,
                    amdgpu::GpuMemory *memory, amdgpu::L2Cache *l2)
      : Base(std::move(name), config, memory, l2) {
    if (l2)
      l2->set_backing_memory(memory);
    set_memory(memory);
    set_l2(l2);
  }

  void execute_and_route(Instruction *inst, amdgpu::Wavefront &wf) {
    execute_instruction(inst, wf);
    if (inst->is_memory_op())
      route_memory_inst(inst, wf);
    else
      delete inst;
  }
};

TEST(RdnaWaitcntTest, Rdna3NamedSopkWaitcntsSetFineGrainedTargets) {
  amdgpu::GpuMemory mem("rdna3_waitcnt_mem");
  amdgpu::L2Cache l2("rdna3_waitcnt_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA3;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create("rdna3_waitcnt_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto dispatch = [&]() -> amdgpu::Wavefront * {
    cu->reset_all_wf();
    auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
    EXPECT_NE(wf, nullptr);
    return wf;
  };

  {
    auto *wf = dispatch();
    ASSERT_NE(wf, nullptr);
    wf->wait_counters().increment(amdgpu::WaitCounterType::STORECNT);
    const std::array<rdna3::MachineInst, 2> words = {0xBC000000u, 0x00000000u};
    rdna3::SWaitcntVscntSopk inst(words.data());

    EXPECT_NO_THROW(inst.execute_impl(*wf));
    EXPECT_EQ(wf->wait_target().vscnt, 0);
    EXPECT_EQ(wf->state(), amdgpu::WfState::WAITCNT);
  }

  {
    auto *wf = dispatch();
    ASSERT_NE(wf, nullptr);
    wf->wait_counters().increment(amdgpu::WaitCounterType::LOADCNT);
    const std::array<rdna3::MachineInst, 2> words = {0xBC800000u, 0x00000000u};
    rdna3::SWaitcntVmcntSopk inst(words.data());

    EXPECT_NO_THROW(inst.execute_impl(*wf));
    EXPECT_EQ(wf->wait_target().vmcnt, 0);
    EXPECT_EQ(wf->state(), amdgpu::WfState::WAITCNT);
  }

  {
    auto *wf = dispatch();
    ASSERT_NE(wf, nullptr);
    wf->wait_counters().increment(amdgpu::WaitCounterType::EXPCNT);
    const std::array<rdna3::MachineInst, 2> words = {0xBD000000u, 0x00000000u};
    rdna3::SWaitcntExpcntSopk inst(words.data());

    EXPECT_NO_THROW(inst.execute_impl(*wf));
    EXPECT_EQ(wf->wait_target().expcnt, 0);
    EXPECT_EQ(wf->state(), amdgpu::WfState::WAITCNT);
  }

  {
    auto *wf = dispatch();
    ASSERT_NE(wf, nullptr);
    wf->wait_counters().increment(amdgpu::WaitCounterType::DSCNT);
    const std::array<rdna3::MachineInst, 2> words = {0xBD800000u, 0x00000000u};
    rdna3::SWaitcntLgkmcntSopk inst(words.data());

    EXPECT_NO_THROW(inst.execute_impl(*wf));
    EXPECT_EQ(wf->wait_target().lgkmcnt, 0);
    EXPECT_EQ(wf->state(), amdgpu::WfState::WAITCNT);
  }
}

TEST(UtilBitTest, IsAlignedChecksPowerOfTwoAlignment) {
  EXPECT_TRUE(util::is_aligned<uint64_t>(0x1000u, 4u));
  EXPECT_TRUE(util::is_aligned<uint32_t>(0u, 8u));
  EXPECT_FALSE(util::is_aligned<uint64_t>(0x1003u, 4u));
}

// ---------------------------------------------------------------------------
// MFMA register layout tests
// ---------------------------------------------------------------------------

TEST(MfmaExecTest, InputLocF16_4x4x4) {
  // v_mfma_f32_4x4x4f16:
  // lanes_per_block = 64 / (4 * 4) = 4, elems_per_group = 4/4 = 1.
  auto loc = amdgpu::input_loc(4, 4, 4, /*i=*/2, /*k=*/0, /*b=*/0, 32);
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
  auto loc = amdgpu::input_loc(16, 16, 1, 0, 0, 0, 16);
  EXPECT_EQ(loc.vgpr_offset, 0u);
  EXPECT_EQ(loc.lane, 0u);
  EXPECT_EQ(loc.sub_element, 0u);

  // k=1: local=1, vgpr_offset = 1/2 = 0, sub_element = 1
  auto loc1 = amdgpu::input_loc(16, 16, 1, 0, 1, 0, 16);
  EXPECT_EQ(loc1.vgpr_offset, 0u);
  EXPECT_EQ(loc1.sub_element, 1u);
}

TEST(MfmaExecTest, Gfx11WmmaIu8InputLocReplicatesKAcrossHalfwaves) {
  auto g0k0 = amdgpu::gfx11_wmma_input_loc(16, 16, /*i=*/3, /*k=*/0, 8,
                                           /*lane_group=*/0);
  EXPECT_EQ(g0k0.vgpr_offset, 0u);
  EXPECT_EQ(g0k0.lane, 3u);
  EXPECT_EQ(g0k0.sub_element, 0u);

  auto g0k15 = amdgpu::gfx11_wmma_input_loc(16, 16, /*i=*/3, /*k=*/15, 8,
                                            /*lane_group=*/0);
  EXPECT_EQ(g0k15.vgpr_offset, 3u);
  EXPECT_EQ(g0k15.lane, 3u);
  EXPECT_EQ(g0k15.sub_element, 3u);

  auto g1k15 = amdgpu::gfx11_wmma_input_loc(16, 16, /*i=*/3, /*k=*/15, 8,
                                            /*lane_group=*/1);
  EXPECT_EQ(g1k15.vgpr_offset, 3u);
  EXPECT_EQ(g1k15.lane, 19u);
  EXPECT_EQ(g1k15.sub_element, 3u);
}

TEST(MfmaExecTest, Gfx11WmmaOutputLoc32PairsRowsAcrossHalfwaves) {
  auto r0 = amdgpu::gfx11_wmma_output_loc_32(amdgpu::WMMA_WAVE32, 16, 16, /*row=*/0, /*col=*/5);
  EXPECT_EQ(r0.reg, 0u);
  EXPECT_EQ(r0.lane, 5u);

  auto r1 = amdgpu::gfx11_wmma_output_loc_32(amdgpu::WMMA_WAVE32, 16, 16, /*row=*/1, /*col=*/5);
  EXPECT_EQ(r1.reg, 0u);
  EXPECT_EQ(r1.lane, 21u);

  auto r2 = amdgpu::gfx11_wmma_output_loc_32(amdgpu::WMMA_WAVE32, 16, 16, /*row=*/2, /*col=*/5);
  EXPECT_EQ(r2.reg, 1u);
  EXPECT_EQ(r2.lane, 5u);

  auto r15 = amdgpu::gfx11_wmma_output_loc_32(amdgpu::WMMA_WAVE32, 16, 16, /*row=*/15, /*col=*/5);
  EXPECT_EQ(r15.reg, 7u);
  EXPECT_EQ(r15.lane, 21u);
}

TEST(MfmaExecTest, Gfx11WmmaOutputLoc32UsesFourLaneGroupsForWave64) {
  auto r0 = amdgpu::gfx11_wmma_output_loc_32(amdgpu::WMMA_WAVE64, 16, 16, /*row=*/0, /*col=*/5);
  EXPECT_EQ(r0.reg, 0u);
  EXPECT_EQ(r0.lane, 5u);

  auto r1 = amdgpu::gfx11_wmma_output_loc_32(amdgpu::WMMA_WAVE64, 16, 16, /*row=*/1, /*col=*/5);
  EXPECT_EQ(r1.reg, 0u);
  EXPECT_EQ(r1.lane, 21u);

  auto r2 = amdgpu::gfx11_wmma_output_loc_32(amdgpu::WMMA_WAVE64, 16, 16, /*row=*/2, /*col=*/5);
  EXPECT_EQ(r2.reg, 0u);
  EXPECT_EQ(r2.lane, 37u);

  auto r3 = amdgpu::gfx11_wmma_output_loc_32(amdgpu::WMMA_WAVE64, 16, 16, /*row=*/3, /*col=*/5);
  EXPECT_EQ(r3.reg, 0u);
  EXPECT_EQ(r3.lane, 53u);

  auto r15 = amdgpu::gfx11_wmma_output_loc_32(amdgpu::WMMA_WAVE64, 16, 16, /*row=*/15, /*col=*/5);
  EXPECT_EQ(r15.reg, 3u);
  EXPECT_EQ(r15.lane, 53u);
}

TEST(MfmaExecTest, SwmmacK32InputLocUsesSparseHardwareLayout) {
  // RDNA4 K=32 SWMMAC is not the dense WMMA K=16/K=32 register layout.
  // These positions match the gfx12 builtin layout used by the silicon tests.
  auto a_h_g2s0 = amdgpu::swmmac_a_input_loc(16, 32, /*row=*/3, /*compressed_k=*/4, 16);
  EXPECT_EQ(a_h_g2s0.lane, 19u);
  EXPECT_EQ(a_h_g2s0.vgpr_offset, 0u);
  EXPECT_EQ(a_h_g2s0.sub_element, 0u);

  auto a_h_g4s1 = amdgpu::swmmac_a_input_loc(16, 32, /*row=*/3, /*compressed_k=*/9, 16);
  EXPECT_EQ(a_h_g4s1.lane, 3u);
  EXPECT_EQ(a_h_g4s1.vgpr_offset, 2u);
  EXPECT_EQ(a_h_g4s1.sub_element, 1u);

  auto b_h_k10 = amdgpu::swmmac_b_input_loc(16, 32, /*col=*/5, /*dense_k=*/10, 16);
  EXPECT_EQ(b_h_k10.lane, 21u);
  EXPECT_EQ(b_h_k10.vgpr_offset, 1u);
  EXPECT_EQ(b_h_k10.sub_element, 0u);

  auto a_fp8_g4s1 = amdgpu::swmmac_a_input_loc(16, 32, /*row=*/3, /*compressed_k=*/9, 8);
  EXPECT_EQ(a_fp8_g4s1.lane, 19u);
  EXPECT_EQ(a_fp8_g4s1.vgpr_offset, 0u);
  EXPECT_EQ(a_fp8_g4s1.sub_element, 1u);

  auto b_fp8_k10 = amdgpu::swmmac_b_input_loc(16, 32, /*col=*/5, /*dense_k=*/10, 8);
  EXPECT_EQ(b_fp8_k10.lane, 5u);
  EXPECT_EQ(b_fp8_k10.vgpr_offset, 2u);
  EXPECT_EQ(b_fp8_k10.sub_element, 2u);

  auto h_idx_g2s0 = amdgpu::swmmac_index_loc(16, 32, 16, /*row=*/3, /*compressed_k=*/4, 16);
  EXPECT_EQ(h_idx_g2s0.lane, 19u);
  EXPECT_EQ(h_idx_g2s0.local_compressed_k, 0u);

  auto fp8_idx_g4s0 = amdgpu::swmmac_index_loc(16, 32, 8, /*row=*/3, /*compressed_k=*/8, 16);
  EXPECT_EQ(fp8_idx_g4s0.lane, 19u);
  EXPECT_EQ(fp8_idx_g4s0.local_compressed_k, 0u);
}

TEST(MfmaExecTest, Gfx12Wave64WmmaLocSplitsKAcrossFourLaneGroups) {
  auto a_k8 = amdgpu::gfx12_wmma_input_loc(amdgpu::WMMA_WAVE64, 16, 16, /*i=*/3, /*k=*/8, 16);
  EXPECT_EQ(a_k8.lane, 35u);
  EXPECT_EQ(a_k8.vgpr_offset, 0u);
  EXPECT_EQ(a_k8.sub_element, 0u);

  auto a_k15 = amdgpu::gfx12_wmma_input_loc(amdgpu::WMMA_WAVE64, 16, 16, /*i=*/3, /*k=*/15, 8);
  EXPECT_EQ(a_k15.lane, 51u);
  EXPECT_EQ(a_k15.vgpr_offset, 0u);
  EXPECT_EQ(a_k15.sub_element, 3u);

  auto out32 = amdgpu::gfx12_wmma_output_loc_32(amdgpu::WMMA_WAVE64, 16, 16, /*row=*/13, /*col=*/5);
  EXPECT_EQ(out32.lane, 53u);
  EXPECT_EQ(out32.reg, 1u);

  auto out16 = amdgpu::gfx12_wmma_output_loc_16(amdgpu::WMMA_WAVE64, 16, 16, /*row=*/13, /*col=*/5);
  EXPECT_EQ(out16.lane, 53u);
  EXPECT_EQ(out16.reg, 0u);
  EXPECT_EQ(out16.sub_element, 1u);
}

TEST(MfmaExecTest, Gfx12Wave64SwmmacK32LocUsesSparseHardwareLayout) {
  auto a_h_g4s1 =
      amdgpu::swmmac_a_input_loc(amdgpu::WMMA_WAVE64, 16, 32, /*row=*/3, /*compressed_k=*/9, 16);
  EXPECT_EQ(a_h_g4s1.lane, 35u);
  EXPECT_EQ(a_h_g4s1.vgpr_offset, 0u);
  EXPECT_EQ(a_h_g4s1.sub_element, 1u);

  auto b_h_k20 =
      amdgpu::swmmac_b_input_loc(amdgpu::WMMA_WAVE64, 16, 32, /*col=*/5, /*dense_k=*/20, 16);
  EXPECT_EQ(b_h_k20.lane, 37u);
  EXPECT_EQ(b_h_k20.vgpr_offset, 2u);
  EXPECT_EQ(b_h_k20.sub_element, 0u);

  auto a_fp8_g2s0 =
      amdgpu::swmmac_a_input_loc(amdgpu::WMMA_WAVE64, 16, 32, /*row=*/3, /*compressed_k=*/4, 8);
  EXPECT_EQ(a_fp8_g2s0.lane, 35u);
  EXPECT_EQ(a_fp8_g2s0.vgpr_offset, 0u);
  EXPECT_EQ(a_fp8_g2s0.sub_element, 0u);

  auto b_fp8_k10 =
      amdgpu::swmmac_b_input_loc(amdgpu::WMMA_WAVE64, 16, 32, /*col=*/5, /*dense_k=*/10, 8);
  EXPECT_EQ(b_fp8_k10.lane, 37u);
  EXPECT_EQ(b_fp8_k10.vgpr_offset, 0u);
  EXPECT_EQ(b_fp8_k10.sub_element, 2u);

  auto h_idx_g4s1 = amdgpu::swmmac_index_loc(amdgpu::WMMA_WAVE64, 16, 32, 16, /*row=*/3,
                                             /*compressed_k=*/9, 16);
  EXPECT_EQ(h_idx_g4s1.lane, 35u);
  EXPECT_EQ(h_idx_g4s1.local_compressed_k, 1u);

  auto fp8_idx_g2s0 = amdgpu::swmmac_index_loc(amdgpu::WMMA_WAVE64, 16, 32, 8, /*row=*/3,
                                               /*compressed_k=*/4, 16);
  EXPECT_EQ(fp8_idx_g2s0.lane, 35u);
  EXPECT_EQ(fp8_idx_g2s0.local_compressed_k, 0u);
}

TEST(MfmaExecTest, Cdna3CvtFp8Bf8UsesSameFnuzDecodeAsMfma) {
  amdgpu::GpuMemory mem("cdna3_cvt_fnuz_mem");
  amdgpu::L2Cache l2("cdna3_cvt_fnuz_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA3;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 104;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create("cdna3_cvt_fnuz_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);
  wf->set_exec(~0ULL);

  constexpr uint32_t kSrc = 4;
  constexpr uint32_t kDstFp8 = 8;
  constexpr uint32_t kDstBf8 = 9;
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    uint8_t byte = (lane & 1u) ? 0xC0u : 0x40u;
    cu->write_vgpr(vbase + kSrc, lane, byte);
  }

  cdna3::Vop1MachineInst raw_fp8{};
  raw_fp8.src0 = 256 + kSrc;
  raw_fp8.vdst = kDstFp8;
  cdna3::VCvtF32Fp8Vop1 cvt_fp8(reinterpret_cast<const cdna3::MachineInst *>(&raw_fp8));
  cvt_fp8.execute_impl(*wf);

  cdna3::Vop1MachineInst raw_bf8{};
  raw_bf8.src0 = 256 + kSrc;
  raw_bf8.vdst = kDstBf8;
  cdna3::VCvtF32Bf8Vop1 cvt_bf8(reinterpret_cast<const cdna3::MachineInst *>(&raw_bf8));
  cvt_bf8.execute_impl(*wf);

  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    amdgpu::InputLoc loc{/*vgpr_offset=*/0, lane, /*sub_element=*/0};
    float mfma_fp8 = amdgpu::extract_fp8_fnuz(*cu, vbase + kSrc, loc);
    float cvt_fp8_value = std::bit_cast<float>(cu->read_vgpr(vbase + kDstFp8, lane));
    EXPECT_EQ(cvt_fp8_value, mfma_fp8) << "fp8 lane=" << lane;

    float mfma_bf8 = amdgpu::extract_bf8_fnuz(*cu, vbase + kSrc, loc);
    float cvt_bf8_value = std::bit_cast<float>(cu->read_vgpr(vbase + kDstBf8, lane));
    EXPECT_EQ(cvt_bf8_value, mfma_bf8) << "bf8 lane=" << lane;
  }
}

void write_packed_byte(amdgpu::ComputeUnitCore &cu, uint32_t reg, uint32_t lane, uint32_t byte,
                       uint8_t value) {
  const uint32_t shift = 8u * byte;
  const uint32_t old = cu.read_vgpr(reg, lane);
  cu.write_vgpr(reg, lane, (old & ~(0xFFu << shift)) | (static_cast<uint32_t>(value) << shift));
}

void fill_vgprs(amdgpu::ComputeUnitCore &cu, uint32_t base, uint32_t regs, uint32_t lanes,
                uint32_t value) {
  for (uint32_t reg = 0; reg < regs; ++reg)
    for (uint32_t lane = 0; lane < lanes; ++lane)
      cu.write_vgpr(base + reg, lane, value);
}

TEST(MfmaExecTest, SwmmacF32K32Fp8MatchesSparseReference) {
  amdgpu::GpuMemory gpu_mem("rdna4_swmmac_fp8_exec_mem");
  amdgpu::L2Cache l2("rdna4_swmmac_fp8_exec_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create("rdna4_swmmac_fp8_exec", cfg, &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec((1ULL << wf->wf_size()) - 1ULL);

  const uint32_t vb = wf->vgpr_alloc().base;
  const uint32_t a_base = vb + 0;
  const uint32_t b_base = vb + 4;
  const uint32_t c_base = vb + 12;
  const uint32_t d_base = vb + 20;
  const uint32_t index_base = vb + 32;

  fill_vgprs(*cu, a_base, 40, wf->wf_size(), 0);

  constexpr uint32_t M = 16;
  constexpr uint32_t N = 16;
  constexpr uint32_t K = 32;
  constexpr uint32_t Groups = K / 4;
  const std::array<std::array<uint32_t, 2>, 4> pairs = {{{0u, 2u}, {1u, 3u}, {0u, 1u}, {2u, 3u}}};

  auto a_value = [](uint32_t row, uint32_t group, uint32_t slot) {
    return static_cast<float>(static_cast<int>((row + 2u * group + slot) % 5u) - 2);
  };
  auto b_value = [](uint32_t k, uint32_t col) {
    return static_cast<float>(static_cast<int>((col + 3u * k) % 5u) - 2);
  };
  auto c_value = [](uint32_t row, uint32_t col) {
    return static_cast<float>(static_cast<int>((2u * row + col) % 7u) - 3);
  };

  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t group = 0; group < Groups; ++group) {
      const auto pair = pairs[(row + group) % pairs.size()];
      const uint32_t index_lane = row + 16u * (group / 4u);
      const uint32_t field = pair[0] | (pair[1] << 2u);
      const uint32_t index_shift = 4u * (group & 3u);
      const uint32_t old_index = cu->read_vgpr(index_base, index_lane);
      cu->write_vgpr(index_base, index_lane, old_index | (field << index_shift));
      for (uint32_t slot = 0; slot < 2; ++slot) {
        const uint32_t compressed_k = 2u * group + slot;
        const auto loc = amdgpu::swmmac_a_input_loc(M, K, row, compressed_k, 8);
        write_packed_byte(*cu, a_base + loc.vgpr_offset, loc.lane, loc.sub_element,
                          util::f32_to_fp8_e4m3_rne(a_value(row, group, slot)));
      }
    }
  }

  for (uint32_t k = 0; k < K; ++k) {
    for (uint32_t col = 0; col < N; ++col) {
      const auto loc = amdgpu::swmmac_b_input_loc(N, K, col, k, 8);
      write_packed_byte(*cu, b_base + loc.vgpr_offset, loc.lane, loc.sub_element,
                        util::f32_to_fp8_e4m3_rne(b_value(k, col)));
    }
  }

  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      const auto out = amdgpu::wmma_output_loc_32(M, N, row, col);
      cu->write_vgpr(c_base + out.reg, out.lane, std::bit_cast<uint32_t>(c_value(row, col)));
    }
  }

  amdgpu::exec_swmmac_f32(*cu, M, N, K, 8, d_base, a_base, b_base, c_base, index_base, 16,
                          /*index_key=*/0, amdgpu::extract_fp8, amdgpu::extract_fp8);

  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      float ref = c_value(row, col);
      for (uint32_t group = 0; group < Groups; ++group) {
        const auto pair = pairs[(row + group) % pairs.size()];
        for (uint32_t slot = 0; slot < 2; ++slot) {
          const uint32_t k = 4u * group + pair[slot];
          const float av =
              util::fp8_e4m3_to_f32(util::f32_to_fp8_e4m3_rne(a_value(row, group, slot)));
          const float bv = util::fp8_e4m3_to_f32(util::f32_to_fp8_e4m3_rne(b_value(k, col)));
          ref += av * bv;
        }
      }
      const auto out = amdgpu::wmma_output_loc_32(M, N, row, col);
      EXPECT_EQ(cu->read_vgpr(d_base + out.reg, out.lane), std::bit_cast<uint32_t>(ref))
          << "row=" << row << " col=" << col;
    }
  }

  cu->reset_all_wf();
}

TEST(MfmaExecTest, WmmaF8f6f4K128InputLocUsesPairAwareSubbyteLayouts) {
  auto ab6 = amdgpu::wmma_f8f6f4_input_loc(16, 128, /*i=*/3, /*k=*/4, 6,
                                           /*mixed_subbyte=*/false);
  EXPECT_EQ(ab6.lane, 19u);
  EXPECT_EQ(ab6.vgpr_offset, 0u);
  EXPECT_EQ(ab6.bit_offset, 0u);
  EXPECT_EQ(ab6.data_bits, 6u);

  auto ab4 = amdgpu::wmma_f8f6f4_input_loc(16, 128, /*i=*/3, /*k=*/4, 4,
                                           /*mixed_subbyte=*/false);
  EXPECT_EQ(ab4.lane, 19u);
  EXPECT_EQ(ab4.vgpr_offset, 0u);
  EXPECT_EQ(ab4.bit_offset, 0u);
  EXPECT_EQ(ab4.data_bits, 4u);

  auto mixed6 = amdgpu::wmma_f8f6f4_input_loc(16, 128, /*i=*/3, /*k=*/4, 6,
                                              /*mixed_subbyte=*/true);
  EXPECT_EQ(mixed6.lane, 3u);
  EXPECT_EQ(mixed6.vgpr_offset, 3u);
  EXPECT_EQ(mixed6.bit_offset, 0u);

  auto mixed4 = amdgpu::wmma_f8f6f4_input_loc(16, 128, /*i=*/3, /*k=*/31, 4,
                                              /*mixed_subbyte=*/true);
  EXPECT_EQ(mixed4.lane, 3u);
  EXPECT_EQ(mixed4.vgpr_offset, 3u);
  EXPECT_EQ(mixed4.bit_offset, 28u);

  auto ab8 = amdgpu::wmma_f8f6f4_input_loc(16, 128, /*i=*/3, /*k=*/4, 8,
                                           /*mixed_subbyte=*/false);
  EXPECT_EQ(ab8.lane, 19u);
  EXPECT_EQ(ab8.vgpr_offset, 0u);
  EXPECT_EQ(ab8.sub_element, 0u);

  EXPECT_EQ(amdgpu::wmma_f8f6f4_scale_byte(/*k=*/4, /*data_bits=*/6,
                                           /*mixed_pair=*/false, /*scale16=*/false),
            1u);
  EXPECT_EQ(amdgpu::wmma_f8f6f4_scale_byte(/*k=*/32, /*data_bits=*/6,
                                           /*mixed_pair=*/false, /*scale16=*/true),
            1u);
  EXPECT_EQ(amdgpu::wmma_f8f6f4_scale_byte(/*k=*/64, /*data_bits=*/4,
                                           /*mixed_pair=*/false, /*scale16=*/true),
            4u);
  EXPECT_EQ(amdgpu::wmma_f8f6f4_scale_byte(/*k=*/32, /*data_bits=*/4,
                                           /*mixed_pair=*/true, /*scale16=*/false),
            1u);
  EXPECT_EQ(amdgpu::wmma_f8f6f4_scale_byte(/*k=*/36, /*data_bits=*/4,
                                           /*mixed_pair=*/true, /*scale16=*/true),
            3u);
}

TEST(MfmaExecTest, WmmaF4_32x16x128UsesSiliconGroundedALayoutAndScaleLane) {
  auto row0 = amdgpu::wmma_a_input_loc(32, 128, /*row=*/0, /*k=*/0, 4, 4);
  EXPECT_EQ(row0.lane, 0u);
  EXPECT_EQ(row0.vgpr_offset, 0u);
  EXPECT_EQ(row0.bit_offset, 0u);

  auto row8 = amdgpu::wmma_a_input_loc(32, 128, /*row=*/8, /*k=*/0, 4, 4);
  EXPECT_EQ(row8.lane, 0u);
  EXPECT_EQ(row8.vgpr_offset, 8u);
  EXPECT_EQ(row8.bit_offset, 0u);

  auto row16 = amdgpu::wmma_a_input_loc(32, 128, /*row=*/16, /*k=*/4, 4, 4);
  EXPECT_EQ(row16.lane, 24u);
  EXPECT_EQ(row16.vgpr_offset, 0u);
  EXPECT_EQ(row16.bit_offset, 0u);

  auto row24 = amdgpu::wmma_a_input_loc(32, 128, /*row=*/24, /*k=*/7, 4, 4);
  EXPECT_EQ(row24.lane, 24u);
  EXPECT_EQ(row24.vgpr_offset, 8u);
  EXPECT_EQ(row24.bit_offset, 12u);

  EXPECT_EQ(amdgpu::wmma_a_scale_lane(32, 128, /*row=*/0, 0, 4, 4), 0u);
  EXPECT_EQ(amdgpu::wmma_a_scale_lane(32, 128, /*row=*/8, 0, 4, 4), 16u);
  EXPECT_EQ(amdgpu::wmma_a_scale_lane(32, 128, /*row=*/16, 0, 4, 4), 8u);
  EXPECT_EQ(amdgpu::wmma_a_scale_lane(32, 128, /*row=*/24, 0, 4, 4), 24u);
}

TEST(MfmaExecTest, OutputLoc32_4x4) {
  // 4x4 matrix, block 0: reg = column index, lane = row index.
  auto loc = amdgpu::output_loc_32(4, 4, /*col=*/2, /*row=*/1, /*b=*/0);
  EXPECT_EQ(loc.reg, 2u);
  EXPECT_EQ(loc.lane, 1u);
}

TEST(MfmaExecTest, ResolveAccConstant) {
  // Encoding value 0-255 = inline constant. The callback should be invoked.
  uint32_t const_acc = 0;
  uint32_t result = amdgpu::resolve_acc<amdgpu::AccMode::Unified>(
      /*vb=*/100, /*dst=*/200, /*src2_ev=*/128, const_acc, [&]() -> uint32_t { return 42u; });
  EXPECT_EQ(const_acc, 42u);
  EXPECT_EQ(result, 200u); // Returns dst when constant.
}

TEST(MfmaExecTest, ResolveAccVgpr) {
  // Encoding value 256-511 = VGPR.
  uint32_t const_acc = 0;
  uint32_t result = amdgpu::resolve_acc<amdgpu::AccMode::Unified>(
      /*vb=*/100, /*dst=*/200, /*src2_ev=*/260, const_acc, [&]() -> uint32_t { return 99u; });
  EXPECT_EQ(const_acc, amdgpu::ACC_FROM_VGPR);
  EXPECT_EQ(result, 100u + 4u); // vb + (260 - 256)
}

TEST(MfmaExecTest, ResolveAccAccVgpr) {
  // Encoding value 768-1023 = AccVGPR (unified alias).
  uint32_t const_acc = 0;
  uint32_t result = amdgpu::resolve_acc<amdgpu::AccMode::Unified>(
      /*vb=*/100, /*dst=*/200, /*src2_ev=*/770, const_acc, [&]() -> uint32_t { return 99u; });
  EXPECT_EQ(const_acc, amdgpu::ACC_FROM_VGPR);
  EXPECT_EQ(result, 100u + amdgpu::ACC_VGPR_OFFSET + 2u);
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

TEST(CuFactoryTest, CdnaAccVgprsDoNotAliasNextWaveSlot) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(::testing::Message() << "arch=" << static_cast<int>(arch));
    amdgpu::GpuMemory mem("test_mem");
    amdgpu::L2Cache l2("test_l2");

    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 2;
    cfg.sgprs_per_wf = 104;
    cfg.vgprs_per_wf = 256;
    cfg.lds_size_kb = 64;

    auto cu = amdgpu::ComputeUnitCore::create("test_cu", cfg, &mem, &l2);
    ASSERT_NE(cu, nullptr);

    auto *wf0 = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
    ASSERT_NE(wf0, nullptr);
    auto *wf1 = cu->dispatch_wf(1, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
    ASSERT_NE(wf1, nullptr);

    const uint32_t wf0_acc0 = wf0->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET;
    const uint32_t wf0_acc_last = wf0_acc0 + kCdnaAccVgprsPerWf - 1;
    const uint32_t wf1_v0 = wf1->vgpr_alloc().base;
    EXPECT_NE(wf0_acc0, wf1_v0);
    EXPECT_LT(wf0_acc_last, wf1_v0);

    cu->write_vgpr(wf0_acc0, 0, 0xA55A0001u);
    cu->write_vgpr(wf0_acc_last, 0, 0xDEADBEEFu);
    cu->write_vgpr(wf1_v0, 0, 0x5AA50002u);

    EXPECT_EQ(cu->read_vgpr(wf0_acc0, 0), 0xA55A0001u);
    EXPECT_EQ(cu->read_vgpr(wf0_acc_last, 0), 0xDEADBEEFu);
    EXPECT_EQ(cu->read_vgpr(wf1_v0, 0), 0x5AA50002u);
  }
}

TEST(CuFactoryTest, CdnaAccVgprsAreClearedOnRedispatch) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(::testing::Message() << "arch=" << static_cast<int>(arch));
    amdgpu::GpuMemory mem("test_mem");
    amdgpu::L2Cache l2("test_l2");

    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 104;
    cfg.vgprs_per_wf = 256;
    cfg.lds_size_kb = 64;

    auto cu = amdgpu::ComputeUnitCore::create("test_cu", cfg, &mem, &l2);
    ASSERT_NE(cu, nullptr);

    auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
    ASSERT_NE(wf, nullptr);
    const uint32_t acc0 = wf->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET;
    const uint32_t acc_last = acc0 + kCdnaAccVgprsPerWf - 1;
    cu->write_vgpr(acc0, 0, 0xFFFFFFFFu);
    cu->write_vgpr(acc_last, 0, 0xDEADBEEFu);

    cu->reset_all_wf();
    wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
    ASSERT_NE(wf, nullptr);

    EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET, 0), 0u);
    EXPECT_EQ(
        cu->read_vgpr(wf->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET + kCdnaAccVgprsPerWf - 1, 0),
        0u);
  }
}

INSTANTIATE_TEST_SUITE_P(AllIsas, CuFactoryTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                           ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4,
                                           ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_RDNA1,
                                           ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3,
                                           ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4));

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

TEST(DppPermuteTest, RowRor1) {
  using namespace amdgpu::dpp;
  bool oob = false;
  // row_ror 1 = 0x121: lane K reads from lane K-1, wrapping within the row.
  EXPECT_EQ(dpp_permute(0x121, 0, 64, oob), 15);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(0x121, 1, 64, oob), 0);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(0x121, 16, 64, oob), 31);
  EXPECT_FALSE(oob);
}

TEST(DppPermuteTest, WaveShiftAndRotate) {
  using namespace amdgpu::dpp;
  bool oob = false;

  EXPECT_EQ(dpp_permute(WF_SHL1, 0, 64, oob), 1);
  EXPECT_FALSE(oob);
  oob = false;
  dpp_permute(WF_SHL1, 63, 64, oob);
  EXPECT_TRUE(oob);

  oob = false;
  EXPECT_EQ(dpp_permute(WF_ROL1, 0, 64, oob), 1);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(WF_ROL1, 63, 64, oob), 0);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(WF_ROL1, 31, 32, oob), 0);
  EXPECT_FALSE(oob);

  EXPECT_EQ(dpp_permute(WF_SRL1, 1, 64, oob), 0);
  EXPECT_FALSE(oob);
  oob = false;
  dpp_permute(WF_SRL1, 0, 64, oob);
  EXPECT_TRUE(oob);

  oob = false;
  EXPECT_EQ(dpp_permute(WF_ROR1, 0, 64, oob), 63);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(WF_ROR1, 1, 64, oob), 0);
  EXPECT_FALSE(oob);
}

TEST(DppPermuteTest, RowMirrors) {
  using namespace amdgpu::dpp;
  bool oob = false;
  // row_mirror = 0x140: reverse lane order within a row.
  EXPECT_EQ(dpp_permute(0x140, 0, 64, oob), 15);
  EXPECT_EQ(dpp_permute(0x140, 15, 64, oob), 0);
  EXPECT_EQ(dpp_permute(0x140, 7, 64, oob), 8);
  // Second row.
  EXPECT_EQ(dpp_permute(0x140, 16, 64, oob), 31);

  // row_half_mirror = 0x141: reverse lane order within each 8-lane half-row.
  EXPECT_EQ(dpp_permute(0x141, 0, 64, oob), 7);
  EXPECT_EQ(dpp_permute(0x141, 7, 64, oob), 0);
  EXPECT_EQ(dpp_permute(0x141, 8, 64, oob), 15);
  EXPECT_EQ(dpp_permute(0x141, 16, 64, oob), 23);
}

TEST(DppPermuteTest, RowShareAndXmask) {
  using namespace amdgpu::dpp;
  bool oob = false;
  // row_share with lane_sel=1 = 0x151: every lane in a row reads row lane 1.
  EXPECT_EQ(dpp_permute(0x151, 0, 64, oob), 1);
  EXPECT_EQ(dpp_permute(0x151, 15, 64, oob), 1);
  EXPECT_EQ(dpp_permute(0x151, 16, 64, oob), 17);
  EXPECT_EQ(dpp_permute(0x151, 31, 64, oob), 17);

  // row_xmask with mask=1 = 0x161: XOR lane offset with 1 (swap adjacent pairs).
  EXPECT_EQ(dpp_permute(0x161, 0, 64, oob), 1);
  EXPECT_EQ(dpp_permute(0x161, 1, 64, oob), 0);
  EXPECT_EQ(dpp_permute(0x161, 2, 64, oob), 3);
  EXPECT_EQ(dpp_permute(0x161, 3, 64, oob), 2);
  EXPECT_EQ(dpp_permute(0x161, 16, 64, oob), 17);
  EXPECT_EQ(dpp_permute(0x161, 17, 64, oob), 16);
}

TEST(DppPermuteTest, RowBroadcasts) {
  using namespace amdgpu::dpp;
  bool oob = false;

  EXPECT_EQ(dpp_permute(ROW_BCAST15, 16, 64, oob), 15);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(ROW_BCAST15, 31, 64, oob), 15);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(ROW_BCAST15, 32, 64, oob), 31);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(ROW_BCAST15, 47, 64, oob), 31);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(ROW_BCAST15, 48, 64, oob), 47);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(ROW_BCAST15, 63, 64, oob), 47);
  EXPECT_FALSE(oob);

  oob = false;
  dpp_permute(ROW_BCAST15, 15, 64, oob);
  EXPECT_TRUE(oob);

  oob = false;
  EXPECT_EQ(dpp_permute(ROW_BCAST31, 32, 64, oob), 31);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(ROW_BCAST31, 63, 64, oob), 31);
  EXPECT_FALSE(oob);

  oob = false;
  dpp_permute(ROW_BCAST31, 31, 64, oob);
  EXPECT_TRUE(oob);
}

TEST(DppPermuteTest, Dpp8SelectsWithinGroupsOfEight) {
  using namespace amdgpu::dpp;
  const uint32_t lane_sel = (7u << 0u) | (0u << 3u) | (3u << 6u) | (2u << 9u) | (5u << 12u) |
                            (4u << 15u) | (1u << 18u) | (6u << 21u);

  EXPECT_EQ(dpp8_src_lane(0, lane_sel), 7u);
  EXPECT_EQ(dpp8_src_lane(1, lane_sel), 0u);
  EXPECT_EQ(dpp8_src_lane(6, lane_sel), 1u);
  EXPECT_EQ(dpp8_src_lane(7, lane_sel), 6u);
  EXPECT_EQ(dpp8_src_lane(8, lane_sel), 15u);
  EXPECT_EQ(dpp8_src_lane(15, lane_sel), 14u);
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

TEST(DppPermuteTest, WriteMaskHonorsBoundCtrlAndBroadcastValidity) {
  using namespace amdgpu::dpp;

  EXPECT_FALSE(dpp_lane_write_enabled(0, 64, ROW_SHR1, 0xF, 0xF, 0));
  EXPECT_TRUE(dpp_lane_write_enabled(0, 64, ROW_SHR1, 0xF, 0xF, 1));
  EXPECT_TRUE(dpp_lane_write_enabled(1, 64, ROW_SHR1, 0xF, 0xF, 0));

  EXPECT_FALSE(dpp_lane_write_enabled(15, 64, ROW_BCAST15, 0xF, 0xF, 0));
  EXPECT_TRUE(dpp_lane_write_enabled(16, 64, ROW_BCAST15, 0xF, 0xF, 0));
  EXPECT_TRUE(dpp_lane_write_enabled(32, 64, ROW_BCAST15, 0xF, 0xF, 0));
  EXPECT_FALSE(dpp_lane_write_enabled(31, 64, ROW_BCAST31, 0xF, 0xF, 0));
  EXPECT_TRUE(dpp_lane_write_enabled(32, 64, ROW_BCAST31, 0xF, 0xF, 0));

  EXPECT_FALSE(dpp_lane_write_enabled(16, 64, ROW_BCAST15, 0xD, 0xF, 0));
  EXPECT_FALSE(dpp_lane_write_enabled(20, 64, ROW_BCAST15, 0xF, 0xD, 0));

  uint64_t mask = dpp_write_mask(64, ROW_BCAST15, 0xF, 0xF, 0);
  EXPECT_EQ(mask, 0xFFFFFFFFFFFF0000ULL);

  mask = dpp_write_mask(64, ROW_BCAST31, 0xF, 0xF, 0);
  EXPECT_EQ(mask, 0xFFFFFFFF00000000ULL);
}

struct Cdna1DppTraits {
  static constexpr const char *name = "cdna1";
  static constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA1;
  using MachineInst = cdna1::MachineInst;
  using Vop1VopDppMachineInst = cdna1::Vop1VopDppMachineInst;
  using VMovB32Vop1 = cdna1::VMovB32Vop1;
  using VCmpEqU32Vopc = cdna1::VCmpEqU32Vopc;
  using VCmpxEqU32Vopc = cdna1::VCmpxEqU32Vopc;
};

struct Cdna2DppTraits {
  static constexpr const char *name = "cdna2";
  static constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA2;
  using MachineInst = cdna2::MachineInst;
  using Vop1VopDppMachineInst = cdna2::Vop1VopDppMachineInst;
  using VMovB32Vop1 = cdna2::VMovB32Vop1;
  using VCmpEqU32Vopc = cdna2::VCmpEqU32Vopc;
  using VCmpxEqU32Vopc = cdna2::VCmpxEqU32Vopc;
};

struct Cdna3DppTraits {
  static constexpr const char *name = "cdna3";
  static constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA3;
  using MachineInst = cdna3::MachineInst;
  using Vop1VopDppMachineInst = cdna3::Vop1VopDppMachineInst;
  using VMovB32Vop1 = cdna3::VMovB32Vop1;
  using VCmpEqU32Vopc = cdna3::VCmpEqU32Vopc;
  using VCmpxEqU32Vopc = cdna3::VCmpxEqU32Vopc;
};

struct Cdna4DppTraits {
  static constexpr const char *name = "cdna4";
  static constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA4;
  using MachineInst = cdna4::MachineInst;
  using Vop1VopDppMachineInst = cdna4::Vop1VopDppMachineInst;
  using VMovB32Vop1 = cdna4::VMovB32Vop1;
  using VCmpEqU32Vopc = cdna4::VCmpEqU32Vopc;
  using VCmpxEqU32Vopc = cdna4::VCmpxEqU32Vopc;
};

template <typename Traits> void cdna_generated_vop1_uses_shared_row_broadcast() {
  SCOPED_TRACE(Traits::name);
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp_broadcast_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp_broadcast_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 104;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_dpp_broadcast_cu", cfg,
                                            &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);
  wf->set_exec(~0ULL);

  constexpr uint32_t kSrc = 4;
  constexpr uint32_t kDst = 8;
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu->write_vgpr(vbase + kSrc, lane, 0x1000u + lane);
    cu->write_vgpr(vbase + kDst, lane, 0xDEAD0000u + lane);
  }

  typename Traits::Vop1VopDppMachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = amdgpu::dpp::ROW_BCAST15;
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;

  typename Traits::VMovB32Vop1 inst(reinterpret_cast<const typename Traits::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 0), 0u);
  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 15), 0u);
  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 16), 0x100Fu);
  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 31), 0x100Fu);
  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 32), 0x101Fu);
  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 47), 0x101Fu);
  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 48), 0x102Fu);
  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 63), 0x102Fu);
}

template <typename Traits> void cdna_generated_vop1_dpp_write_mask_honors_bound_ctrl() {
  SCOPED_TRACE(Traits::name);
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp_write_mask_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp_write_mask_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 104;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_dpp_write_mask_cu", cfg,
                                            &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);
  wf->set_exec(~0ULL);

  constexpr uint32_t kSrc = 4;
  constexpr uint32_t kDst = 8;
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu->write_vgpr(vbase + kSrc, lane, 0x1000u + lane);
    cu->write_vgpr(vbase + kDst, lane, 0xDEAD0000u + lane);
  }

  typename Traits::Vop1VopDppMachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = amdgpu::dpp::ROW_BCAST15;
  raw.bound_ctrl = 0;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;

  typename Traits::VMovB32Vop1 inst(reinterpret_cast<const typename Traits::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 0), 0xDEAD0000u);
  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 15), 0xDEAD000Fu);
  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 16), 0x100Fu);
  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 31), 0x100Fu);
  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 32), 0x101Fu);
  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 47), 0x101Fu);
  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 48), 0x102Fu);
  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 63), 0x102Fu);
}

template <typename Traits> void cdna_generated_vopc_dpp_write_mask_honors_bound_ctrl() {
  SCOPED_TRACE(Traits::name);
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp_vopc_write_mask_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp_vopc_write_mask_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 104;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_dpp_vopc_write_mask_cu",
                                            cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);
  wf->set_exec(~0ULL);
  wf->set_vcc(0);

  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    uint32_t src = 0x1000u + lane;
    uint32_t cmp = src;
    if (lane >= 16 && lane < 32)
      cmp = 0x100Fu;
    else if (lane >= 32 && lane < 48)
      cmp = 0x101Fu;
    else if (lane >= 48)
      cmp = 0x102Fu;
    cu->write_vgpr(vbase + kSrc0, lane, src);
    cu->write_vgpr(vbase + kSrc1, lane, cmp);
  }

  typename Traits::Vop1VopDppMachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc0;
  // VOPC reads bits [16:9] as vsrc1; those overlap the VOP_DPP op field.
  raw.op = kSrc1;
  raw.dpp_ctrl = amdgpu::dpp::ROW_BCAST15;
  raw.bound_ctrl = 0;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;

  typename Traits::VCmpEqU32Vopc inst(reinterpret_cast<const typename Traits::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  EXPECT_EQ(wf->vcc(), 0xFFFFFFFFFFFF0000ULL);
}

template <typename Traits> void cdna_generated_vcmpx_dpp_write_mask_preserves_exec() {
  SCOPED_TRACE(Traits::name);
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp_vcmpx_exec_mask_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp_vcmpx_exec_mask_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 104;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_dpp_vcmpx_exec_mask_cu",
                                            cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);

  constexpr uint64_t kOldExec = 0xFFFFFFFFFFFF005AULL;
  constexpr uint64_t kOldVcc = 0x00000000000000A5ULL;
  wf->set_exec(kOldExec);
  wf->set_vcc(kOldVcc);

  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    uint32_t src = 0x1000u + lane;
    uint32_t cmp = 0xDEAD0000u + lane;
    if (lane >= 16 && lane < 32)
      cmp = 0x100Fu;
    else if (lane >= 32 && lane < 48)
      cmp = 0x101Fu;
    else if (lane >= 48)
      cmp = 0x102Fu;
    if (lane == 20)
      cmp = 0xDEAD0020u;
    cu->write_vgpr(vbase + kSrc0, lane, src);
    cu->write_vgpr(vbase + kSrc1, lane, cmp);
  }

  typename Traits::Vop1VopDppMachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc0;
  raw.op = kSrc1;
  raw.dpp_ctrl = amdgpu::dpp::ROW_BCAST15;
  raw.bound_ctrl = 0;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;

  typename Traits::VCmpxEqU32Vopc inst(
      reinterpret_cast<const typename Traits::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  EXPECT_EQ(wf->vcc(), 0xFFFFFFFFFFEF00A5ULL);
  EXPECT_EQ(wf->exec(), 0xFFFFFFFFFFEF005AULL);
}

TEST(DppPermuteTest, CdnaGeneratedVop1UsesSharedRowBroadcast) {
  cdna_generated_vop1_uses_shared_row_broadcast<Cdna1DppTraits>();
  cdna_generated_vop1_uses_shared_row_broadcast<Cdna2DppTraits>();
  cdna_generated_vop1_uses_shared_row_broadcast<Cdna3DppTraits>();
  cdna_generated_vop1_uses_shared_row_broadcast<Cdna4DppTraits>();
}

TEST(DppPermuteTest, CdnaGeneratedVop1DppWriteMaskHonorsBoundCtrl) {
  cdna_generated_vop1_dpp_write_mask_honors_bound_ctrl<Cdna1DppTraits>();
  cdna_generated_vop1_dpp_write_mask_honors_bound_ctrl<Cdna2DppTraits>();
  cdna_generated_vop1_dpp_write_mask_honors_bound_ctrl<Cdna3DppTraits>();
  cdna_generated_vop1_dpp_write_mask_honors_bound_ctrl<Cdna4DppTraits>();
}

TEST(DppPermuteTest, CdnaGeneratedVopcDppWriteMaskHonorsBoundCtrl) {
  cdna_generated_vopc_dpp_write_mask_honors_bound_ctrl<Cdna1DppTraits>();
  cdna_generated_vopc_dpp_write_mask_honors_bound_ctrl<Cdna2DppTraits>();
  cdna_generated_vopc_dpp_write_mask_honors_bound_ctrl<Cdna3DppTraits>();
  cdna_generated_vopc_dpp_write_mask_honors_bound_ctrl<Cdna4DppTraits>();
}

TEST(DppPermuteTest, CdnaGeneratedVcmpxDppWriteMaskPreservesExec) {
  cdna_generated_vcmpx_dpp_write_mask_preserves_exec<Cdna1DppTraits>();
  cdna_generated_vcmpx_dpp_write_mask_preserves_exec<Cdna2DppTraits>();
  cdna_generated_vcmpx_dpp_write_mask_preserves_exec<Cdna3DppTraits>();
  cdna_generated_vcmpx_dpp_write_mask_preserves_exec<Cdna4DppTraits>();
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

  // Set scratch base via the wavefront's dedicated FLAT_SCRATCH register.
  // On CDNA4 this is an architected HW register, not SGPRs s[102:103].
  constexpr uint64_t SCRATCH_BASE = 0x1'0000'0000ULL;
  wf->set_scratch_base(SCRATCH_BASE);

  // Write a 32-bit offset into VGPR[0] lane 0.
  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase, 0, 0x100); // lane 0: offset 0x100

  // Set EXEC so only lane 0 is active.
  wf->set_exec(1ULL);

  // Build a FlatScratchMachineInst with seg=1 (SCRATCH), sve=1 (VADDR enabled),
  // saddr=0x7F (no SADDR), offset=0x10.
  cdna4::FlatScratchMachineInst inst{};
  inst.seg = 1;       // SCRATCH
  inst.sve = 1;       // VADDR enabled
  inst.saddr = 0x7F;  // No SADDR
  inst.addr = 0;      // VGPR index 0
  inst.offset = 0x10; // 13-bit signed offset

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  amdgpu::addr_calc::flat_calculate_addresses(inst, *wf, d);

  EXPECT_EQ(d.lane_mask, 1ULL);
  // scratch_base (0x1_0000_0000) + VGPR (0x100) + offset (0x10) = 0x1_0000_0110
  EXPECT_EQ(d.per_lane_addr[0], SCRATCH_BASE + 0x100 + 0x10);
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

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  amdgpu::addr_calc::flat_calculate_addresses(inst, *wf, d);

  EXPECT_EQ(d.per_lane_addr[0], 0x1'0000'2000ULL); // No scratch_base added.
}

TEST(RdnaAddrCalcTest, Rdna3Saddr7cUsesVgprPair) {
  amdgpu::GpuMemory mem("rdna3_addr_mem");
  amdgpu::L2Cache l2("rdna3_addr_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA3;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("rdna3_addr_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1ULL);

  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase, 0, 0x2000);
  cu->write_vgpr(vbase + 1, 0, 0x0001);
  cu->write_sgpr(wf->sgpr_alloc().base + 0x7C, 0xDEAD0000);
  cu->write_sgpr(wf->sgpr_alloc().base + 0x7D, 0xDEAD0001);

  rdna3::FlatMachineInst inst{};
  inst.saddr = 0x7C;
  inst.addr = 0;
  inst.offset = 0x20;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  rdna3::flat_calculate_addresses(inst, *wf, d);
  EXPECT_EQ(d.per_lane_addr[0], 0x1'0000'2020ULL);

  cu->write_sgpr(wf->sgpr_alloc().base + 4, 0x3000);
  cu->write_sgpr(wf->sgpr_alloc().base + 5, 0x0002);
  cu->write_vgpr(vbase, 0, 0x40);
  cu->write_vgpr(vbase + 1, 0, 0xDEAD);

  inst.saddr = 4;
  inst.offset = 0x10;
  rdna3::flat_calculate_addresses(inst, *wf, d);
  EXPECT_EQ(d.per_lane_addr[0], 0x2'0000'3050ULL);
}

TEST(RdnaAddrCalcTest, Rdna3ScratchOffUsesScratchBaseAndLaneStride) {
  amdgpu::GpuMemory mem("rdna3_scratch_addr_mem");
  amdgpu::L2Cache l2("rdna3_scratch_addr_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA3;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("rdna3_scratch_addr_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x3ULL);
  wf->set_scratch_base(0x6'0000'0000ULL);
  wf->set_scratch_lane_size(0x100);

  uint32_t sbase = wf->sgpr_alloc().base;
  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase, 0, 0x20);
  cu->write_vgpr(vbase, 1, 0x30);
  cu->write_vgpr(vbase + 1, 0, 0xBAD0);
  cu->write_vgpr(vbase + 1, 1, 0xBAD1);
  cu->write_sgpr(sbase + 0x7C, 0xDEAD0000);
  cu->write_sgpr(sbase + 0x7D, 0xDEAD0001);

  rdna3::FlatMachineInst inst{};
  inst.seg = 1;
  inst.saddr = 0x7C;
  inst.sve = 1;
  inst.addr = 0;
  inst.offset = 0x10;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  rdna3::flat_calculate_addresses(inst, *wf, d);
  EXPECT_EQ(d.per_lane_addr[0], 0x6'0000'0030ULL);
  EXPECT_EQ(d.per_lane_addr[1], 0x6'0000'0140ULL);

  cu->write_sgpr(sbase + 4, 0x70);
  cu->write_vgpr(vbase, 0, 0x200);
  cu->write_vgpr(vbase, 1, 0x300);
  inst.saddr = 4;
  inst.sve = 0;
  rdna3::flat_calculate_addresses(inst, *wf, d);
  EXPECT_EQ(d.per_lane_addr[0], 0x6'0000'0080ULL);
  EXPECT_EQ(d.per_lane_addr[1], 0x6'0000'0180ULL);
}

TEST(RdnaScratchExecutionTest, Rdna3B128StoreFeedsB32LoadsWithVectorOffsets) {
  amdgpu::GpuMemory mem("rdna3_scratch_exec_mem");
  amdgpu::L2Cache l2("rdna3_scratch_exec_l2");
  l2.set_backing_memory(&mem);

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA3;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;
  auto cu = std::make_unique<Rdna3MemoryTestCu>("rdna3_scratch_exec_cu", cfg, &mem, &l2);

  auto *wf = cu->dispatch_wf(0, 0, 128, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x1FULL);
  wf->set_scratch_base(0x6'0000'0000ULL);
  wf->set_scratch_lane_size(0x100);

  const uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < 5; ++lane) {
    cu->write_vgpr(vbase + 10, lane, 0x1000u + lane);
    cu->write_vgpr(vbase + 11, lane, 0x1100u + lane);
    cu->write_vgpr(vbase + 12, lane, 0x1200u + lane);
    cu->write_vgpr(vbase + 13, lane, 0x1300u + lane);
    cu->write_vgpr(vbase + 2, lane, lane * 4);
  }

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_NE(decoder, nullptr);
  auto execute_mem = [&](std::array<uint32_t, 2> words) {
    std::unique_ptr<Instruction> inst(decoder->decode(words.data()));
    ASSERT_NE(inst, nullptr);
    cu->execute_and_route(inst.release(), *wf);
  };

  // scratch_store_b128 off, v[10:13], off
  execute_mem({0xDC750000u, 0x007C0A00u});
  // scratch_load_b32 v1, v2, off
  execute_mem({0xDC510000u, 0x01FC0002u});

  EXPECT_EQ(cu->read_vgpr(vbase + 1, 0), 0x1000u);
  EXPECT_EQ(cu->read_vgpr(vbase + 1, 1), 0x1101u);
  EXPECT_EQ(cu->read_vgpr(vbase + 1, 2), 0x1202u);
  EXPECT_EQ(cu->read_vgpr(vbase + 1, 3), 0x1303u);
  EXPECT_EQ(cu->read_vgpr(vbase + 1, 4), 0u);
}

TEST(RdnaAddrCalcTest, Rdna3SmemSoffsetHandlesNullM0AndSgprSelectors) {
  amdgpu::GpuMemory mem("rdna3_smem_addr_mem");
  amdgpu::L2Cache l2("rdna3_smem_addr_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA3;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("rdna3_smem_addr_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kBase = 0x1'0000'1000ULL;
  uint32_t sbase = wf->sgpr_alloc().base;
  cu->write_sgpr(sbase, static_cast<uint32_t>(kBase));
  cu->write_sgpr(sbase + 1, static_cast<uint32_t>(kBase >> 32));
  cu->write_sgpr(sbase + rdna3::OPR_SMEM_OFFSET_NULL, 0x100);
  cu->write_sgpr(sbase + rdna3::OPR_SMEM_OFFSET_M0, 0x200);
  cu->write_sgpr(sbase + 8, 0x80);
  wf->set_m0(0x40);

  rdna3::SmemMachineInst inst{};
  inst.sbase = 0;
  inst.offset = 0x20;

  inst.soffset = rdna3::OPR_SMEM_OFFSET_NULL;
  EXPECT_EQ(rdna3::smem_calculate_address(inst, *wf), kBase + 0x20);

  inst.soffset = rdna3::OPR_SMEM_OFFSET_M0;
  EXPECT_EQ(rdna3::smem_calculate_address(inst, *wf), kBase + 0x20 + 0x40);

  inst.soffset = 8;
  EXPECT_EQ(rdna3::smem_calculate_address(inst, *wf), kBase + 0x20 + 0x80);
}

TEST(RdnaAddrCalcTest, Rdna3MubufWrapsOffsetPartBeforeBoundsCheck) {
  amdgpu::GpuMemory mem("rdna3_mubuf_wrap_mem");
  amdgpu::L2Cache l2("rdna3_mubuf_wrap_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA3;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("rdna3_mubuf_wrap_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1ULL);

  constexpr uint64_t kBase = 0x2'0000'1000ULL;
  uint32_t sbase = wf->sgpr_alloc().base;
  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_sgpr(sbase, static_cast<uint32_t>(kBase));
  cu->write_sgpr(sbase + 1, static_cast<uint32_t>(kBase >> 32));
  cu->write_sgpr(sbase + 2, 0x1000);
  cu->write_sgpr(sbase + 3, 0);
  cu->write_vgpr(vbase + 4, 0, 0xFFFF'FFF0u);

  rdna3::MubufMachineInst inst{};
  inst.srsrc = 0;
  inst.soffset = 0x80;
  inst.offen = 1;
  inst.idxen = 0;
  inst.vaddr = 4;
  inst.offset = 0x10;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  rdna3::mubuf_calculate_addresses(inst, *wf, d);
  EXPECT_EQ(d.lane_mask, 1ULL);
  EXPECT_EQ(d.per_lane_addr[0], kBase);
}

TEST(RdnaAddrCalcTest, Rdna4Saddr7cCoversGlobalFlatAndScratch) {
  amdgpu::GpuMemory mem("rdna4_addr_mem");
  amdgpu::L2Cache l2("rdna4_addr_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("rdna4_addr_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1ULL);

  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase, 0, 0x4000);
  cu->write_vgpr(vbase + 1, 0, 0x0003);
  cu->write_sgpr(wf->sgpr_alloc().base + 0x7C, 0xDEAD0000);
  cu->write_sgpr(wf->sgpr_alloc().base + 0x7D, 0xDEAD0001);

  rdna4::VglobalMachineInst global_inst{};
  global_inst.saddr = 0x7C;
  global_inst.vaddr = 0;
  global_inst.ioffset = 0x20;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  rdna4::flat_calculate_addresses(global_inst, *wf, d);
  EXPECT_EQ(d.per_lane_addr[0], 0x3'0000'4020ULL);

  cu->write_sgpr(wf->sgpr_alloc().base + 4, 0x5000);
  cu->write_sgpr(wf->sgpr_alloc().base + 5, 0x0004);
  cu->write_vgpr(vbase, 0, 0x60);
  cu->write_vgpr(vbase + 1, 0, 0xDEAD);

  global_inst.saddr = 4;
  global_inst.ioffset = 0x10;
  rdna4::flat_calculate_addresses(global_inst, *wf, d);
  EXPECT_EQ(d.per_lane_addr[0], 0x4'0000'5070ULL);

  cu->write_vgpr(vbase, 0, 0x8000);
  cu->write_vgpr(vbase + 1, 0, 0x0005);
  cu->write_sgpr(wf->sgpr_alloc().base + 0x7C, 0xBAD00000);
  cu->write_sgpr(wf->sgpr_alloc().base + 0x7D, 0xBAD00001);

  rdna4::VflatMachineInst flat_inst{};
  flat_inst.saddr = 0x7C;
  flat_inst.vaddr = 0;
  flat_inst.ioffset = 0x30;
  rdna4::flat_calculate_addresses(flat_inst, *wf, d);
  EXPECT_EQ(d.per_lane_addr[0], 0x5'0000'8030ULL);

  wf->set_exec(0x3ULL);
  wf->set_scratch_base(0x6'0000'0000ULL);
  wf->set_scratch_lane_size(0x100);
  cu->write_vgpr(vbase, 0, 0x80);
  cu->write_vgpr(vbase, 1, 0x90);
  cu->write_vgpr(vbase + 1, 0, 0xBAD);
  cu->write_sgpr(wf->sgpr_alloc().base + 0x7C, 0xBAD00000);

  rdna4::VscratchMachineInst scratch_inst{};
  scratch_inst.saddr = 0x7C;
  scratch_inst.sve = 1;
  scratch_inst.vaddr = 0;
  scratch_inst.ioffset = 0x24;
  rdna4::flat_calculate_addresses(scratch_inst, *wf, d);
  EXPECT_EQ(d.per_lane_addr[0], 0x6'0000'00A4ULL);
  EXPECT_EQ(d.per_lane_addr[1], 0x6'0000'01B4ULL);

  cu->write_sgpr(wf->sgpr_alloc().base + 4, 0x7000);
  cu->write_vgpr(vbase, 0, 0x60);
  cu->write_vgpr(vbase, 1, 0x70);
  scratch_inst.saddr = 4;
  scratch_inst.ioffset = 0x10;
  rdna4::flat_calculate_addresses(scratch_inst, *wf, d);
  EXPECT_EQ(d.per_lane_addr[0], 0x6'0000'7070ULL);
  EXPECT_EQ(d.per_lane_addr[1], 0x6'0000'7180ULL);

  scratch_inst.sve = 0;
  rdna4::flat_calculate_addresses(scratch_inst, *wf, d);
  EXPECT_EQ(d.per_lane_addr[0], 0x6'0000'7010ULL);
  EXPECT_EQ(d.per_lane_addr[1], 0x6'0000'7110ULL);
}

TEST(Gfx1250AddrCalcTest, FlatPrivateScratchDecodesLaneBits) {
  amdgpu::GpuMemory mem("gfx1250_flat_private_mem");
  amdgpu::L2Cache l2("gfx1250_flat_private_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_GFX1250;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_flat_private_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 32);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec(0x7ULL);

  constexpr uint64_t kScratchBase = 0x0002'0000'0000'0000ULL;
  constexpr uint64_t kPrivateBase = 0x0007'0000'0000'0000ULL;
  constexpr uint32_t kPrivateSegmentSize = 0x80;
  wf->set_scratch_base(kScratchBase);
  wf->set_scratch_lane_size(kPrivateSegmentSize);
  wf->set_apertures(0x0001'0000'0000'0000ULL, 0x0001'0000'ffff'ffffULL, kPrivateBase,
                    kPrivateBase + 0xffff'ffffULL);

  gfx1250::Operand flat_scratch_base(
      64, gfx1250::OperandType::OPR_SRC,
      static_cast<int>(gfx1250::OpSelSrc::OPR_SRC_SRC_FLAT_SCRATCH_BASE_LO));
  ASSERT_EQ(flat_scratch_base.read_scalar64(*wf), kScratchBase);

  const uint64_t private_offsets[] = {0x10, 0x14, kPrivateSegmentSize + 0x20};
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < 3; ++lane) {
    uint64_t private_offset = private_offsets[lane];
    uint64_t flat_private_addr =
        kScratchBase + (static_cast<uint64_t>(lane) << 52) + private_offset;
    cu->write_vgpr(vbase, lane, static_cast<uint32_t>(flat_private_addr));
    cu->write_vgpr(vbase + 1, lane, static_cast<uint32_t>(flat_private_addr >> 32));
  }

  gfx1250::VflatMachineInst inst{};
  inst.saddr = gfx1250::OPR_SREG_NULL;
  inst.vaddr = 0;
  inst.ioffset = 4;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  gfx1250::flat_calculate_addresses(inst, *wf, d);

  for (uint32_t lane = 0; lane < 3; ++lane) {
    uint64_t private_offset = private_offsets[lane] + inst.ioffset;
    uint64_t expected =
        kScratchBase + static_cast<uint64_t>(lane) * kPrivateSegmentSize + private_offset;
    EXPECT_EQ(d.per_lane_addr[lane], expected) << "lane " << lane;
  }
}

TEST(RdnaAddrCalcTest, Rdna4SmemSoffsetHandlesNullM0AndSgprSelectors) {
  amdgpu::GpuMemory mem("rdna4_smem_addr_mem");
  amdgpu::L2Cache l2("rdna4_smem_addr_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("rdna4_smem_addr_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kBase = 0x2'0000'1000ULL;
  uint32_t sbase = wf->sgpr_alloc().base;
  cu->write_sgpr(sbase, static_cast<uint32_t>(kBase));
  cu->write_sgpr(sbase + 1, static_cast<uint32_t>(kBase >> 32));
  cu->write_sgpr(sbase + rdna4::OPR_SMEM_OFFSET_NULL, 0x100);
  cu->write_sgpr(sbase + rdna4::OPR_SMEM_OFFSET_M0, 0x200);
  cu->write_sgpr(sbase + 8, 0x80);
  wf->set_m0(0x40);

  rdna4::SmemMachineInst inst{};
  inst.sbase = 0;
  inst.ioffset = 0x20;

  inst.soffset = rdna4::OPR_SMEM_OFFSET_NULL;
  EXPECT_EQ(rdna4::smem_calculate_address(inst, *wf), kBase + 0x20);

  inst.soffset = rdna4::OPR_SMEM_OFFSET_M0;
  EXPECT_EQ(rdna4::smem_calculate_address(inst, *wf), kBase + 0x20 + 0x40);

  inst.soffset = 8;
  EXPECT_EQ(rdna4::smem_calculate_address(inst, *wf), kBase + 0x20 + 0x80);
}

TEST(RdnaAddrCalcTest, Rdna4VbufferUsesDecodedRsrcAndOptionalSoffset) {
  amdgpu::GpuMemory mem("rdna4_vbuffer_addr_mem");
  amdgpu::L2Cache l2("rdna4_vbuffer_addr_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("rdna4_vbuffer_addr_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x3ULL);

  uint32_t sbase = wf->sgpr_alloc().base;
  uint32_t vbase = wf->vgpr_alloc().base;
  constexpr uint64_t kBase = 0x2'0000'1000ULL;
  cu->write_sgpr(sbase + 4, 0x1000);
  cu->write_sgpr(sbase + 5, 0x0002);
  cu->write_sgpr(sbase + 6, 0xDEAD0006);
  cu->write_sgpr(sbase + 7, 0xDEAD0007);

  // A previous implementation incorrectly treated rsrc as a descriptor index and
  // read s[rsrc * 4:rsrc * 4 + 3]. Keep that range distinct so the test fails
  // if the decoded first-SGPR selector is scaled again.
  cu->write_sgpr(sbase + 16, 0xBAD00010);
  cu->write_sgpr(sbase + 17, 0xBAD00011);

  cu->write_vgpr(vbase + 4, 0, 0x20);
  cu->write_vgpr(vbase + 4, 1, 0x30);

  rdna4::VbufferMachineInst inst{};
  inst.rsrc = 4;
  inst.soffset = rdna4::OPR_SREG_M0_NULL;
  inst.offen = 1;
  inst.idxen = 0;
  inst.vaddr = 4;
  inst.ioffset = 0x10;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  rdna4::mubuf_calculate_addresses(inst, *wf, d);
  EXPECT_EQ(d.lane_mask, 0x3ULL);
  EXPECT_EQ(d.per_lane_addr[0], kBase + 0x20 + 0x10);
  EXPECT_EQ(d.per_lane_addr[1], kBase + 0x30 + 0x10);

  cu->write_sgpr(sbase + 8, 0x80);
  inst.soffset = 8;
  rdna4::mubuf_calculate_addresses(inst, *wf, d);
  EXPECT_EQ(d.per_lane_addr[0], kBase + 0x20 + 0x10 + 0x80);

  wf->set_m0(0x40);
  inst.soffset = rdna4::OPR_SREG_M0_M0;
  rdna4::mubuf_calculate_addresses(inst, *wf, d);
  EXPECT_EQ(d.per_lane_addr[0], kBase + 0x20 + 0x10 + 0x40);
}

TEST(RdnaAddrCalcTest, Rdna4VbufferWrapsOffsetPartBeforeBaseAddition) {
  amdgpu::GpuMemory mem("rdna4_vbuffer_wrap_mem");
  amdgpu::L2Cache l2("rdna4_vbuffer_wrap_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("rdna4_vbuffer_wrap_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1ULL);

  constexpr uint64_t kBase = 0x2'0000'1000ULL;
  uint32_t sbase = wf->sgpr_alloc().base;
  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_sgpr(sbase + 4, static_cast<uint32_t>(kBase));
  cu->write_sgpr(sbase + 5, static_cast<uint32_t>(kBase >> 32));
  cu->write_sgpr(sbase + 6, 0x1000);
  cu->write_sgpr(sbase + 7, 0);
  cu->write_vgpr(vbase + 4, 0, 0xFFFF'8200u);

  rdna4::VbufferMachineInst inst{};
  inst.rsrc = 4;
  inst.soffset = rdna4::OPR_SREG_M0_NULL;
  inst.offen = 1;
  inst.idxen = 0;
  inst.vaddr = 4;
  inst.ioffset = 0x7E00;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  rdna4::mubuf_calculate_addresses(inst, *wf, d);
  EXPECT_EQ(d.lane_mask, 1ULL);
  EXPECT_EQ(d.per_lane_addr[0], kBase);
}

TEST(RdnaAddrCalcTest, Gfx1250VbufferWrapsOffsetPartBeforeBoundsCheck) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_wrap_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_wrap_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_GFX1250;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_wrap_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1ULL);

  constexpr uint64_t kBase = 0x2'0000'1000ULL;
  uint32_t sbase = wf->sgpr_alloc().base;
  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_sgpr(sbase + 4, static_cast<uint32_t>(kBase));
  cu->write_sgpr(sbase + 5, static_cast<uint32_t>(kBase >> 32));
  cu->write_sgpr(sbase + 6, 0x100);
  cu->write_sgpr(sbase + 7, 0);
  cu->write_vgpr(vbase + 4, 0, 0xFFFF'8200u);

  gfx1250::VbufferMachineInst inst{};
  inst.rsrc = 4;
  inst.soffset = gfx1250::OPR_SREG_NULL;
  inst.offen = 1;
  inst.idxen = 0;
  inst.vaddr = 4;
  inst.ioffset = 0x7E00;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  gfx1250::mubuf_calculate_addresses(inst, *wf, d);
  EXPECT_EQ(d.lane_mask, 1ULL);
  EXPECT_EQ(d.per_lane_addr[0], kBase);
}

void expect_vector_lane_reads_use_own_wave_vgprs(rj_code_arch_t arch) {
  amdgpu::GpuMemory mem("rdna_lane_read_mem");
  amdgpu::L2Cache l2("rdna_lane_read_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = arch;
  cfg.num_wf_slots = 4;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create("rdna_lane_read_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  std::array<amdgpu::Wavefront *, 4> wfs{};
  for (uint32_t i = 0; i < wfs.size(); ++i) {
    wfs[i] = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
    ASSERT_NE(wfs[i], nullptr);
    const uint32_t vbase = wfs[i]->vgpr_alloc().base;
    cu->write_vgpr(vbase + 1, 0, 0x100u + i);
    cu->write_vgpr(vbase + 4, 0, 0xdead0000u + i);
    cu->write_vgpr(vbase + 4, 31, 0x300u + i);
    cu->write_vgpr(vbase + 141, 0, 0xbeef0000u + i);
    cu->write_vgpr(vbase + 141, 2, 0x200u + i);
    cu->write_vgpr(vbase + 141, 31, 0x400u + i);
  }

  auto decoder = Decoder::create(arch);
  ASSERT_NE(decoder, nullptr);

  constexpr std::array<uint32_t, 2> kReadfirstlaneS24V1 = {
      0x7e300501u, // v_readfirstlane_b32_e32 s24, v1
      0u,
  };
  constexpr std::array<uint32_t, 2> kReadlaneS4V141Lane2 = {
      0xd7600004u, // v_readlane_b32 s4, v141, 2
      0x0201058du,
  };
  constexpr std::array<uint32_t, 2> kReadlaneS4V4Lane31 = {
      0xd7600004u, // v_readlane_b32 s4, v4, 31
      0x02013f04u,
  };
  constexpr std::array<uint32_t, 2> kReadlaneS4V141S2 = {
      0xd7600004u, // v_readlane_b32 s4, v141, s2
      0x0200058du,
  };
  constexpr std::array<uint32_t, 2> kWritelaneV141S4S2 = {
      0xd761008du, // v_writelane_b32 v141, s4, s2
      0x02000404u,
  };

  for (uint32_t i = 0; i < wfs.size(); ++i) {
    const uint32_t sbase = wfs[i]->sgpr_alloc().base;
    std::unique_ptr<Instruction> inst(decoder->decode(kReadfirstlaneS24V1.data()));
    ASSERT_NE(inst, nullptr);
    cu->write_sgpr(sbase + 24, 0);
    cu->execute_instruction(inst.get(), *wfs[i]);
    EXPECT_EQ(cu->read_sgpr(sbase + 24), 0x100u + i);
  }

  for (uint32_t i = 0; i < wfs.size(); ++i) {
    const uint32_t sbase = wfs[i]->sgpr_alloc().base;
    std::unique_ptr<Instruction> inst(decoder->decode(kReadlaneS4V141Lane2.data()));
    ASSERT_NE(inst, nullptr);
    cu->write_sgpr(sbase + 2, 0);
    cu->write_sgpr(sbase + 4, 0);
    cu->execute_instruction(inst.get(), *wfs[i]);
    EXPECT_EQ(cu->read_sgpr(sbase + 4), 0x200u + i);
  }

  for (uint32_t i = 0; i < wfs.size(); ++i) {
    const uint32_t sbase = wfs[i]->sgpr_alloc().base;
    std::unique_ptr<Instruction> inst(decoder->decode(kReadlaneS4V4Lane31.data()));
    ASSERT_NE(inst, nullptr);
    cu->write_sgpr(sbase + 4, 0);
    cu->write_sgpr(sbase + 31, 0);
    cu->execute_instruction(inst.get(), *wfs[i]);
    EXPECT_EQ(cu->read_sgpr(sbase + 4), 0x300u + i);
  }

  for (uint32_t i = 0; i < wfs.size(); ++i) {
    const uint32_t sbase = wfs[i]->sgpr_alloc().base;
    std::unique_ptr<Instruction> inst(decoder->decode(kReadlaneS4V141S2.data()));
    ASSERT_NE(inst, nullptr);
    cu->write_sgpr(sbase + 2, 31);
    cu->write_sgpr(sbase + 4, 0);
    cu->execute_instruction(inst.get(), *wfs[i]);
    EXPECT_EQ(cu->read_sgpr(sbase + 4), 0x400u + i);
  }

  for (uint32_t i = 0; i < wfs.size(); ++i) {
    const uint32_t sbase = wfs[i]->sgpr_alloc().base;
    const uint32_t vbase = wfs[i]->vgpr_alloc().base;
    std::unique_ptr<Instruction> inst(decoder->decode(kWritelaneV141S4S2.data()));
    ASSERT_NE(inst, nullptr);
    cu->write_sgpr(sbase + 2, 31);
    cu->write_sgpr(sbase + 4, 0x500u + i);
    cu->execute_instruction(inst.get(), *wfs[i]);
    EXPECT_EQ(cu->read_vgpr(vbase + 141, 2), 0x200u + i);
    EXPECT_EQ(cu->read_vgpr(vbase + 141, 31), 0x500u + i);
  }
}

TEST(RdnaVectorLaneReadTest, ReadlaneFamilyUsesDecodedSourceVgprPerWave) {
  {
    SCOPED_TRACE("rdna3");
    expect_vector_lane_reads_use_own_wave_vgprs(ROCJITSU_CODE_ARCH_RDNA3);
  }
  {
    SCOPED_TRACE("rdna4");
    expect_vector_lane_reads_use_own_wave_vgprs(ROCJITSU_CODE_ARCH_RDNA4);
  }
}

} // namespace
