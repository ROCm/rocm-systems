// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file shared_infra_test.cpp
/// @brief Phase B unit tests: addr_calc, MMA execution, wavefront context, CU factory.

#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna5/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/cdna5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/vop2.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vop2.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vop3p.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/vop2.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/sopk.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/vop3p.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/vop3p.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/smem.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vop2.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vop3p.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vopc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/isa/arch/amdgpu/shared/alu_exceptions.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/ds_transpose.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include "simdojo/sim/simulation.h"
#include "util/bit.h"
#include "util/data_types.h"
#include "util/except.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include "rocjitsu/vm/amdgpu/register_access.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using namespace rocjitsu;

struct ForceScalarGuard {
  explicit ForceScalarGuard(bool force_scalar) : old_force_scalar(util::force_scalar()) {
    util::set_force_scalar_for_testing(force_scalar);
  }
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(old_force_scalar); }

  bool old_force_scalar;
};

// ---------------------------------------------------------------------------
// Concept and trait verification (compile-time)
// ---------------------------------------------------------------------------

/*
 * \NPI new ISA family: add GpuIsa<<isa>::Isa> plus the relevant trait \
 * static_asserts (HasAccVgpr, HasMonolithicWaitcnt, ...) for it here.
 */
static_assert(GpuIsa<cdna3::Isa>);
static_assert(GpuIsa<cdna5::Isa>);
static_assert(GpuIsa<rdna4::Isa>);
static_assert(HasAccVgpr<cdna2::Isa>);
static_assert(HasAccVgpr<cdna3::Isa>);
static_assert(HasAccVgpr<cdna4::Isa>);
static_assert(!HasAccVgpr<cdna5::Isa>);
static_assert(!HasAccVgpr<rdna4::Isa>);
static_assert(HasMonolithicWaitcnt<cdna3::Isa>);
static_assert(!HasMonolithicWaitcnt<cdna5::Isa>);
static_assert(!HasMonolithicWaitcnt<rdna4::Isa>);
static_assert(!isa_properties(ROCJITSU_CODE_ARCH_CDNA3).supports_wgp_mode);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_RDNA4).supports_wgp_mode);
static_assert(!isa_properties(ROCJITSU_CODE_ARCH_CDNA5).supports_wgp_mode);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA1).mode_has_gpr_idx_en);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA2).mode_has_gpr_idx_en);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA3).mode_has_gpr_idx_en);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA4).mode_has_gpr_idx_en);
static_assert(!isa_properties(ROCJITSU_CODE_ARCH_RDNA1).mode_has_gpr_idx_en);
static_assert(!isa_properties(ROCJITSU_CODE_ARCH_RDNA2).mode_has_gpr_idx_en);
static_assert(!isa_properties(ROCJITSU_CODE_ARCH_RDNA3).mode_has_gpr_idx_en);
static_assert(!isa_properties(ROCJITSU_CODE_ARCH_RDNA3_5).mode_has_gpr_idx_en);
static_assert(!isa_properties(ROCJITSU_CODE_ARCH_RDNA4).mode_has_gpr_idx_en);
static_assert(!isa_properties(ROCJITSU_CODE_ARCH_CDNA5).mode_has_gpr_idx_en);
static_assert(!isa_properties(ROCJITSU_CODE_ARCH_CDNA3).uses_ttmp_workgroup_ids);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_RDNA4).uses_ttmp_workgroup_ids);
static_assert(!isa_properties(ROCJITSU_CODE_ARCH_RDNA4).uses_cluster_ttmp_workgroup_ids);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA5).uses_ttmp_workgroup_ids);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA5).uses_cluster_ttmp_workgroup_ids);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA3).wave_state_layout ==
              WaveStateLayout::Legacy);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_RDNA4).wave_state_layout == WaveStateLayout::Gfx12);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA5).wave_state_layout ==
              WaveStateLayout::Gfx12_5);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA3).compute_tmpring_wavesize_granule == 1024);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_RDNA3).compute_tmpring_wavesize_granule == 256);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_RDNA3).compute_tmpring_wavesize_bits == 15);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_RDNA4).compute_tmpring_wavesize_granule == 256);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_RDNA4).compute_tmpring_wavesize_bits == 18);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA5).compute_tmpring_wavesize_granule == 256);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA5).compute_tmpring_wavesize_bits == 18);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA2).descriptor_vgpr_count_granule_wave32 == 0);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA2).descriptor_vgpr_count_granule_wave64 == 8);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA3).descriptor_vgpr_count_granule_wave32 == 0);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA3).descriptor_vgpr_count_granule_wave64 == 8);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_RDNA4).descriptor_vgpr_count_granule_wave32 == 8);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_RDNA4).descriptor_vgpr_count_granule_wave64 == 4);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA5).descriptor_vgpr_count_granule_wave32 == 16);
static_assert(isa_properties(ROCJITSU_CODE_ARCH_CDNA5).descriptor_vgpr_count_granule_wave64 == 0);

// RDNA3/3.5 retain monolithic S_WAITCNT (GFX11 layout).
static_assert(HasMonolithicWaitcnt<rdna3::Isa>);

// RDNA2 supports Wave64 (WF_SIZE_MAX inherited as 64).
static_assert(rdna2::Isa::WF_SIZE_MAX == 64);
static_assert(cdna5::Isa::WF_SIZE_MAX == 32);

// CDNA1 and GFX1250 have no AccVGPRs; CDNA2/3/4 have 256.
constexpr uint32_t kCdnaAccVgprsPerWf = cdna2::Isa::MAX_ACC_VGPRS_PER_WF;
static_assert(cdna1::Isa::MAX_ACC_VGPRS_PER_WF == 0);
static_assert(kCdnaAccVgprsPerWf == 256);
static_assert(cdna3::Isa::MAX_ACC_VGPRS_PER_WF == kCdnaAccVgprsPerWf);
static_assert(cdna4::Isa::MAX_ACC_VGPRS_PER_WF == kCdnaAccVgprsPerWf);
static_assert(cdna5::Isa::MAX_ACC_VGPRS_PER_WF == 0);

TEST(IsaPropertiesTest, DescriptorVgprGranuleSupportsEveryAmdgpuWavefrontMode) {
  struct ExpectedGranule {
    rj_code_arch_t arch;
    uint32_t wavefront_size;
    uint32_t granule;
  };
  constexpr std::array cases = {
      ExpectedGranule{ROCJITSU_CODE_ARCH_CDNA1, 64, 4},
      ExpectedGranule{ROCJITSU_CODE_ARCH_CDNA2, 64, 8},
      ExpectedGranule{ROCJITSU_CODE_ARCH_CDNA3, 64, 8},
      ExpectedGranule{ROCJITSU_CODE_ARCH_CDNA4, 64, 8},
      ExpectedGranule{ROCJITSU_CODE_ARCH_RDNA1, 32, 8},
      ExpectedGranule{ROCJITSU_CODE_ARCH_RDNA1, 64, 4},
      ExpectedGranule{ROCJITSU_CODE_ARCH_RDNA2, 32, 8},
      ExpectedGranule{ROCJITSU_CODE_ARCH_RDNA2, 64, 4},
      ExpectedGranule{ROCJITSU_CODE_ARCH_RDNA3, 32, 8},
      ExpectedGranule{ROCJITSU_CODE_ARCH_RDNA3, 64, 4},
      ExpectedGranule{ROCJITSU_CODE_ARCH_RDNA3_5, 32, 8},
      ExpectedGranule{ROCJITSU_CODE_ARCH_RDNA3_5, 64, 4},
      ExpectedGranule{ROCJITSU_CODE_ARCH_RDNA4, 32, 8},
      ExpectedGranule{ROCJITSU_CODE_ARCH_RDNA4, 64, 4},
      ExpectedGranule{ROCJITSU_CODE_ARCH_CDNA5, 32, 16},
  };

  for (const auto &[arch, wavefront_size, granule] : cases) {
    const auto actual = descriptor_vgpr_count_granule_for_wavefront(arch, wavefront_size);
    ASSERT_TRUE(actual.has_value());
    EXPECT_EQ(*actual, granule);
  }
}

TEST(IsaPropertiesTest, DescriptorVgprGranuleRejectsUnsupportedInputs) {
  EXPECT_FALSE(
      descriptor_vgpr_count_granule_for_wavefront(ROCJITSU_CODE_ARCH_RV32I, 32).has_value());
  EXPECT_FALSE(
      descriptor_vgpr_count_granule_for_wavefront(ROCJITSU_CODE_ARCH_RV64I, 64).has_value());
  EXPECT_FALSE(
      descriptor_vgpr_count_granule_for_wavefront(ROCJITSU_CODE_ARCH_CDNA2, 32).has_value());
  EXPECT_FALSE(
      descriptor_vgpr_count_granule_for_wavefront(ROCJITSU_CODE_ARCH_CDNA5, 64).has_value());
}

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
    // Recycle the single scratch slot: terminate a resident wave (freeing it) so
    // the next dispatch has a free slot, mirroring s_endpgm on real hardware.
    if (cu->wf(0) && !cu->wf(0)->is_halted())
      cu->wf(0)->halt();
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

TEST(MfmaExecTest, Gfx1250WmmaIu8K64PreservesSixteenElementKBlocks) {
  struct Anchor {
    uint32_t k;
    uint32_t lane;
    uint32_t slot;
  };
  constexpr std::array anchors{
      Anchor{0, 3, 0},   Anchor{15, 3, 15}, Anchor{16, 19, 0},  Anchor{31, 19, 15},
      Anchor{32, 3, 16}, Anchor{47, 3, 31}, Anchor{48, 19, 16}, Anchor{63, 19, 31},
  };
  for (const auto &anchor : anchors) {
    const auto loc = amdgpu::wmma_input_loc(16, 64, /*i=*/3, anchor.k, 8);
    EXPECT_EQ(loc.lane, anchor.lane) << anchor.k;
    EXPECT_EQ(loc.vgpr_offset, anchor.slot / 4) << anchor.k;
    EXPECT_EQ(loc.sub_element, anchor.slot % 4) << anchor.k;
  }
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

TEST(MfmaExecTest, Gfx1250SwmmacK128Iu8LocationsMatchHardwareReferenceKernels) {
  // These anchors record the layout required by hardware-reference Tensile
  // kernels. They intentionally do not claim agreement with the public CDNA5
  // sparse-layout text, which describes different B and selector routing.
  for (uint32_t ck = 0; ck < 64; ++ck) {
    const uint32_t expected_a_lane = 5u + 16u * ((ck >> 4) & 1u);
    const uint32_t expected_a_slot = (ck & 15u) + 16u * (ck >> 5);
    const auto a = amdgpu::swmmac_a_input_loc(16, 128, /*row=*/5, ck, 8);
    EXPECT_EQ(a.lane, expected_a_lane) << ck;
    EXPECT_EQ(a.vgpr_offset, expected_a_slot / 4) << ck;
    EXPECT_EQ(a.sub_element, expected_a_slot % 4) << ck;

    const auto index = amdgpu::swmmac_index_loc(16, 128, 8, /*row=*/5, ck, /*index_entries=*/32);
    EXPECT_EQ(index.lane, 5u + 16u * (ck / 32u)) << ck;
    EXPECT_EQ(index.local_compressed_k, ck % 32u) << ck;
  }

  for (uint32_t k = 0; k < 128; ++k) {
    const uint32_t expected_lane = 7u + 16u * ((k >> 5) & 1u);
    const uint32_t expected_slot = (k & 31u) + 32u * (k >> 6);
    const auto b = amdgpu::swmmac_b_input_loc(16, 128, /*col=*/7, k, 8);
    EXPECT_EQ(b.lane, expected_lane) << k;
    EXPECT_EQ(b.vgpr_offset, expected_slot / 4) << k;
    EXPECT_EQ(b.sub_element, expected_slot % 4) << k;
  }
}

TEST(TransposeLoadTest, WmmaTrB8Wave32MatchesMatrixLayout) {
  amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
  state.wf_size = 32;
  state.num_elems = 2;
  state.transpose = static_cast<uint8_t>(amdgpu::TransposeKind::WMMA_TR_B8);
  state.response_data.resize(32 * 8);
  for (uint32_t source_lane = 0; source_lane < 32; ++source_lane)
    for (uint32_t byte = 0; byte < 8; ++byte)
      state.response_data[source_lane * 8 + byte] = static_cast<uint8_t>(source_lane * 8 + byte);

  amdgpu::transpose_response(state);

  ASSERT_EQ(state.num_elems, 2u);
  ASSERT_EQ(state.response_data.size(), 32u * 8u);
  for (uint32_t source_lane = 0; source_lane < 32; ++source_lane) {
    const uint32_t k = (source_lane & 7u) + 8u * (source_lane >> 4);
    for (uint32_t byte = 0; byte < 8; ++byte) {
      const uint32_t row = byte + 8u * ((source_lane >> 3) & 1u);
      const uint32_t lane = row + 16u * ((k >> 3) & 1u);
      const uint32_t slot = (k & 3u) + 4u * ((k >> 2) & 1u);
      EXPECT_EQ(state.response_data[lane * 8 + slot], static_cast<uint8_t>(source_lane * 8 + byte));
    }
  }
}

TEST(TransposeLoadTest, WmmaTrB8Wave64UsesFirst32AddressesAndOneVgpr) {
  amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
  state.wf_size = 64;
  state.num_elems = 2;
  state.lane_mask = ~0ULL;
  state.transpose = static_cast<uint8_t>(amdgpu::TransposeKind::WMMA_TR_B8);
  state.response_data.assign(64 * 8, 0xEE);
  for (uint32_t source_lane = 0; source_lane < 32; ++source_lane)
    for (uint32_t byte = 0; byte < 8; ++byte)
      state.response_data[source_lane * 8 + byte] = static_cast<uint8_t>(source_lane * 8 + byte);

  EXPECT_EQ(amdgpu::transpose_request_lane_mask(state), 0xFFFF'FFFFULL);
  amdgpu::transpose_response(state);

  ASSERT_EQ(state.num_elems, 1u);
  ASSERT_EQ(state.response_data.size(), 64u * 4u);
  for (uint32_t source_lane = 0; source_lane < 32; ++source_lane) {
    const uint32_t k = (source_lane & 7u) + 8u * (source_lane >> 4);
    for (uint32_t byte = 0; byte < 8; ++byte) {
      const uint32_t row = byte + 8u * ((source_lane >> 3) & 1u);
      const uint32_t lane = row + 16u * ((k >> 3) & 1u) + 32u * ((k >> 2) & 1u);
      EXPECT_EQ(state.response_data[lane * 4 + (k & 3u)],
                static_cast<uint8_t>(source_lane * 8 + byte));
    }
  }
}

TEST(TransposeLoadTest, Cdna4B64TrB8MatchesMfmaLaneLayout) {
  amdgpu::VectorMemState state(amdgpu::LOCAL_MEM);
  state.wf_size = 64;
  state.num_elems = 2;
  state.lane_mask = ~0ULL;
  state.transpose = static_cast<uint8_t>(amdgpu::TransposeKind::B64_TR_B8);
  state.response_data.resize(64 * 8);
  for (uint32_t source_lane = 0; source_lane < 64; ++source_lane)
    for (uint32_t byte = 0; byte < 8; ++byte)
      state.response_data[source_lane * 8 + byte] = static_cast<uint8_t>(source_lane * 8 + byte);

  const auto input = state.response_data;
  amdgpu::transpose_response(state);

  ASSERT_EQ(state.num_elems, 2u);
  for (uint32_t dest_lane = 0; dest_lane < 64; ++dest_lane) {
    const uint32_t source_byte = dest_lane & 7u;
    const uint32_t source_base = (dest_lane & ~0xfu) | ((dest_lane >> 3) & 1u);
    for (uint32_t dest_byte = 0; dest_byte < 8; ++dest_byte)
      EXPECT_EQ(state.response_data[dest_lane * 8 + dest_byte],
                input[(source_base + 2 * dest_byte) * 8 + source_byte]);
  }
}

TEST(TransposeLoadTest, Cdna5DsTrB8MatchesMatrixLayout) {
  amdgpu::VectorMemState state(amdgpu::LOCAL_MEM);
  state.wf_size = 32;
  state.num_elems = 2;
  state.lane_mask = 0xFFFF'FFFFULL;
  state.transpose = static_cast<uint8_t>(amdgpu::TransposeKind::CDNA5_DS_TR_B8);
  state.response_data.resize(32 * 8);
  for (uint32_t source_lane = 0; source_lane < 32; ++source_lane) {
    const uint32_t row =
        8u * (source_lane >> 4) + (source_lane & 3u) + 4u * ((source_lane >> 3) & 1u);
    const uint32_t col_base = 8u * ((source_lane >> 2) & 1u);
    for (uint32_t byte = 0; byte < 8; ++byte)
      state.response_data[source_lane * 8 + byte] =
          static_cast<uint8_t>(row * 16 + col_base + byte);
  }

  amdgpu::transpose_response(state);

  ASSERT_EQ(state.num_elems, 2u);
  for (uint32_t dest_lane = 0; dest_lane < 32; ++dest_lane) {
    const uint32_t row_base = 8u * (dest_lane >> 4);
    const uint32_t col = dest_lane & 15u;
    for (uint32_t byte = 0; byte < 8; ++byte)
      EXPECT_EQ(state.response_data[dest_lane * 8 + byte],
                static_cast<uint8_t>((row_base + byte) * 16 + col));
  }
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
  ScopedIsaExecutionBackend execution_backend_scope{&cdna3::execution_backend()};
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

  wf->halt();
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
  EXPECT_EQ(ab8.lane, 3u);
  EXPECT_EQ(ab8.vgpr_offset, 1u);
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

TEST(MfmaExecTest, WmmaF8f6f4K128InputLocMatchesManualLayoutsExhaustively) {
  for (uint32_t data_bits : {4u, 6u, 8u}) {
    for (uint32_t index = 0; index < 16; ++index) {
      for (uint32_t k = 0; k < 128; ++k) {
        SCOPED_TRACE(::testing::Message()
                     << "data_bits=" << data_bits << " index=" << index << " k=" << k);
        uint32_t expected_lane;
        uint32_t expected_slot;
        if (data_bits == 8) {
          expected_lane = index + 16u * ((k >> 4u) & 1u);
          expected_slot = (k & 15u) + 16u * (k >> 5u);
        } else {
          expected_lane = index + 16u * ((k >> 2u) & 1u);
          const uint32_t expected_reg = ((k >> 1u) & 1u) + 2u * ((k >> 3u) & 1u) +
                                        4u * ((k >> 4u) & 1u) + 8u * ((k >> 5u) & 1u) +
                                        16u * ((k >> 6u) & 1u);
          expected_slot = 2u * expected_reg + (k & 1u);
        }
        const uint32_t expected_bit = expected_slot * data_bits;
        const auto actual =
            amdgpu::wmma_f8f6f4_input_loc(16, 128, index, k, data_bits, /*mixed_subbyte=*/false);
        EXPECT_EQ(actual.lane, expected_lane);
        EXPECT_EQ(actual.vgpr_offset, expected_bit / 32u);
        EXPECT_EQ(actual.bit_offset, expected_bit % 32u);
        EXPECT_EQ(actual.data_bits, data_bits);
      }
    }
  }

  for (uint32_t data_bits : {4u, 6u}) {
    for (uint32_t index = 0; index < 16; ++index) {
      for (uint32_t k = 0; k < 128; ++k) {
        SCOPED_TRACE(::testing::Message()
                     << "mixed data_bits=" << data_bits << " index=" << index << " k=" << k);
        const uint32_t expected_lane = index + 16u * ((k >> 5u) & 1u);
        const uint32_t expected_slot = 32u * ((k >> 6u) & 1u) + 16u * ((k >> 2u) & 1u) +
                                       8u * ((k >> 4u) & 1u) + 4u * ((k >> 3u) & 1u) +
                                       2u * ((k >> 1u) & 1u) + (k & 1u);
        const uint32_t expected_bit = expected_slot * data_bits;
        const auto actual =
            amdgpu::wmma_f8f6f4_input_loc(16, 128, index, k, data_bits, /*mixed_subbyte=*/true);
        EXPECT_EQ(actual.lane, expected_lane);
        EXPECT_EQ(actual.vgpr_offset, expected_bit / 32u);
        EXPECT_EQ(actual.bit_offset, expected_bit % 32u);
        EXPECT_EQ(actual.data_bits, data_bits);
      }
    }
  }
}

TEST(MfmaExecTest, Cdna5BlockScaledWmmaUsesContiguousKBlocks) {
  for (uint32_t data_bits : {4u, 6u, 8u}) {
    const uint32_t block_elems = data_bits == 8 ? 16u : 32u;
    for (uint32_t index = 0; index < 16; ++index) {
      for (uint32_t k = 0; k < 128; ++k) {
        SCOPED_TRACE(::testing::Message()
                     << "data_bits=" << data_bits << " index=" << index << " k=" << k);
        const uint32_t expected_lane = index + 16u * ((k / block_elems) & 1u);
        const uint32_t expected_slot = (k / (2u * block_elems)) * block_elems + (k % block_elems);
        const uint32_t expected_bit = expected_slot * data_bits;
        const auto actual = amdgpu::wmma_block_scaled_input_loc(16, 128, index, k, data_bits);
        EXPECT_EQ(actual.lane, expected_lane);
        EXPECT_EQ(actual.vgpr_offset, expected_bit / 32u);
        EXPECT_EQ(actual.bit_offset, expected_bit % 32u);
      }
    }
  }

  for (uint32_t k = 0; k < 128; ++k) {
    EXPECT_EQ(amdgpu::wmma_block_scale_byte(k, /*scale16=*/false), k / 32u);
    EXPECT_EQ(amdgpu::wmma_block_scale_byte(k, /*scale16=*/true), k / 16u);
  }
}

TEST(MfmaExecTest, Cdna4ScaledMfmaInputLocMatchesDenseF8F6F4Layout) {
  constexpr std::array<uint32_t, 3> widths = {8, 6, 4};
  for (const auto &[dim, k_size] :
       std::array<std::pair<uint32_t, uint32_t>, 2>{{{16, 128}, {32, 64}}}) {
    for (uint32_t data_bits : widths) {
      for (uint32_t index = 0; index < dim; ++index) {
        for (uint32_t k = 0; k < k_size; ++k) {
          SCOPED_TRACE(::testing::Message()
                       << "dim=" << dim << " k_size=" << k_size << " data_bits=" << data_bits
                       << " index=" << index << " k=" << k);
          const auto actual = amdgpu::mfma_scale_f8f6f4_input_loc(dim, k_size, index, k, data_bits);
          const auto expected =
              amdgpu::input_loc(dim, k_size, /*B=*/1, index, k, /*b=*/0, data_bits);
          EXPECT_EQ(actual.vgpr_offset, expected.vgpr_offset);
          EXPECT_EQ(actual.lane, expected.lane);
          EXPECT_EQ(actual.sub_element, expected.sub_element);
          EXPECT_EQ(actual.bit_offset, expected.bit_offset);
          EXPECT_EQ(actual.data_bits, expected.data_bits);
        }
      }
    }
  }
}

TEST(MfmaExecTest, WmmaF4_32x16x128UsesConsecutiveM16ALayoutAndScaleLane) {
  auto row0 = amdgpu::wmma_a_input_loc(32, 128, /*row=*/0, /*k=*/0, 4, 4);
  EXPECT_EQ(row0.lane, 0u);
  EXPECT_EQ(row0.vgpr_offset, 0u);
  EXPECT_EQ(row0.bit_offset, 0u);

  auto row8 = amdgpu::wmma_a_input_loc(32, 128, /*row=*/8, /*k=*/0, 4, 4);
  EXPECT_EQ(row8.lane, 8u);
  EXPECT_EQ(row8.vgpr_offset, 0u);
  EXPECT_EQ(row8.bit_offset, 0u);

  auto row16 = amdgpu::wmma_a_input_loc(32, 128, /*row=*/16, /*k=*/4, 4, 4);
  EXPECT_EQ(row16.lane, 16u);
  EXPECT_EQ(row16.vgpr_offset, 8u);
  EXPECT_EQ(row16.bit_offset, 0u);

  auto row24 = amdgpu::wmma_a_input_loc(32, 128, /*row=*/24, /*k=*/7, 4, 4);
  EXPECT_EQ(row24.lane, 24u);
  EXPECT_EQ(row24.vgpr_offset, 8u);
  EXPECT_EQ(row24.bit_offset, 12u);

  EXPECT_EQ(amdgpu::wmma_a_scale_lane(32, 128, /*row=*/0, 0, 4, 4), 0u);
  EXPECT_EQ(amdgpu::wmma_a_scale_lane(32, 128, /*row=*/8, 0, 4, 4), 8u);
  EXPECT_EQ(amdgpu::wmma_a_scale_lane(32, 128, /*row=*/16, 0, 4, 4), 16u);
  EXPECT_EQ(amdgpu::wmma_a_scale_lane(32, 128, /*row=*/24, 0, 4, 4), 24u);
}

// The M=32 A layout is represented by two consecutive M=16 slices.
TEST(MfmaExecTest, WmmaF4_32x16x128ALayoutMatchesConsecutiveM16Slices) {
  const auto split_a_loc = [](uint32_t row, uint32_t k) {
    auto loc = amdgpu::wmma_f8f6f4_input_loc(16, 128, row % 16, k, 4,
                                             /*mixed_subbyte=*/false);
    loc.vgpr_offset += 8 * (row / 16);
    return loc;
  };

  for (uint32_t row = 0; row < 32; ++row) {
    for (uint32_t k = 0; k < 128; ++k) {
      SCOPED_TRACE(::testing::Message() << "row=" << row << " k=" << k);
      const auto actual = amdgpu::wmma_a_input_loc(32, 128, row, k, 4, 4);
      const auto expected = split_a_loc(row, k);
      ASSERT_EQ(actual.lane, expected.lane);
      ASSERT_EQ(actual.vgpr_offset, expected.vgpr_offset);
      ASSERT_EQ(actual.bit_offset, expected.bit_offset);
      ASSERT_EQ(actual.data_bits, expected.data_bits);
    }
  }
}

// C and D use the same consecutive row slices as A.
TEST(MfmaExecTest, WmmaF4_32x16x128CDLayoutMatchesConsecutiveM16Slices) {
  const auto split_output_loc = [](uint32_t row, uint32_t col) {
    auto loc = amdgpu::wmma_output_loc_32(16, 16, row % 16, col);
    loc.reg += 8 * (row / 16);
    return loc;
  };

  for (uint32_t row = 0; row < 32; ++row) {
    for (uint32_t col = 0; col < 16; ++col) {
      SCOPED_TRACE(::testing::Message() << "row=" << row << " col=" << col);
      const auto actual = amdgpu::wmma_output_loc_32(32, 16, row, col);
      const auto expected = split_output_loc(row, col);
      ASSERT_EQ(actual.reg, expected.reg);
      ASSERT_EQ(actual.lane, expected.lane);
    }
  }
}

// A-scale lane selection follows the row index directly.
TEST(MfmaExecTest, WmmaF4_32x16x128AScaleLaneMatchesRow) {
  for (uint32_t row = 0; row < 32; ++row) {
    SCOPED_TRACE(::testing::Message() << "row=" << row);
    const uint32_t actual = amdgpu::wmma_a_scale_lane(32, 128, row, /*scale_select=*/0, 4, 4);
    ASSERT_EQ(actual, row);
  }
}

TEST(MfmaExecTest, OutputLoc32_4x4) {
  // 4x4 matrix, block 0: reg = column index, lane = row index.
  auto loc = amdgpu::output_loc_32(4, 4, /*col=*/2, /*row=*/1, /*b=*/0);
  EXPECT_EQ(loc.reg, 2u);
  EXPECT_EQ(loc.lane, 1u);
}

TEST(MfmaExecTest, DstBaseMapsCdna1OprAccvgprRange) {
  // CDNA1 types MFMA/accvgpr destinations as OPR_ACCVGPR, canonicalized into
  // [768, 1023] (OPR_ACCVGPR_ACC_MIN = 768). acc0 -> vb + ACC_VGPR_OFFSET, not
  // vb + ACC_VGPR_OFFSET + 256 (the pre-fix bug from the shared -512 fold).
  EXPECT_EQ(amdgpu::dst_base(/*vb=*/100, /*ev=*/768, /*acc_cd=*/1),
            100u + amdgpu::ACC_VGPR_OFFSET + 0u);
  EXPECT_EQ(amdgpu::dst_base(/*vb=*/100, /*ev=*/770, /*acc_cd=*/1),
            100u + amdgpu::ACC_VGPR_OFFSET + 2u);
}

TEST(MfmaExecTest, DstBaseMapsCdna2To4AccAndVgprRanges) {
  // CDNA2-4: acc destination via OpSel sits at [512, 767] (acc0 = 512); an
  // ordinary VGPR destination (acc_cd=0, ev 0-255) stays in the arch bank.
  EXPECT_EQ(amdgpu::dst_base(/*vb=*/100, /*ev=*/512, /*acc_cd=*/1),
            100u + amdgpu::ACC_VGPR_OFFSET + 0u);
  EXPECT_EQ(amdgpu::dst_base(/*vb=*/100, /*ev=*/5, /*acc_cd=*/1),
            100u + amdgpu::ACC_VGPR_OFFSET + 5u);
  EXPECT_EQ(amdgpu::dst_base(/*vb=*/100, /*ev=*/5, /*acc_cd=*/0), 100u + 5u);
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

TEST(MfmaExecTest, ScalarF32ReadsEachMatrixOperandOnce) {
  constexpr uint32_t wf_size = 64;
  constexpr uint32_t M = 16, N = 16, K = 4, B = 2;
  constexpr uint32_t sgprs_per_wf = 106, vgprs_per_wf = 256;
  constexpr uint32_t source_a = 0, source_b = 16, accumulator = 32, destination = 48;
  constexpr uint32_t source_a_regs = (M * K * B + wf_size - 1) / wf_size;
  constexpr uint32_t source_b_regs = (N * K * B + wf_size - 1) / wf_size;
  constexpr uint32_t destination_regs = (M * N * B + wf_size - 1) / wf_size;

  amdgpu::GpuMemory gpu_mem("mfma_scalar_snapshot_mem");
  amdgpu::L2Cache l2("mfma_scalar_snapshot_l2");
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_CDNA4;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = sgprs_per_wf;
  config.vgprs_per_wf = vgprs_per_wf;
  config.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("mfma_scalar_snapshot_cu", config, &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, sgprs_per_wf, vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), wf_size);
  const uint32_t vb = wf->vgpr_alloc().base;

  for (uint32_t reg = 0; reg < source_a_regs; ++reg)
    for (uint32_t lane = 0; lane < wf_size; ++lane)
      cu->write_vgpr(vb + source_a + reg, lane, std::bit_cast<uint32_t>(1.0f));
  for (uint32_t reg = 0; reg < source_b_regs; ++reg)
    for (uint32_t lane = 0; lane < wf_size; ++lane)
      cu->write_vgpr(vb + source_b + reg, lane, std::bit_cast<uint32_t>(2.0f));
  for (uint32_t reg = 0; reg < destination_regs; ++reg)
    for (uint32_t lane = 0; lane < wf_size; ++lane)
      cu->write_vgpr(vb + accumulator + reg, lane, std::bit_cast<uint32_t>(3.0f));

  size_t source_a_reads = 0;
  size_t source_b_reads = 0;
  auto extract_a = [&](auto &source_cu, uint32_t base, const amdgpu::InputLoc &loc) {
    ++source_a_reads;
    return amdgpu::extract_f32(source_cu, base, loc);
  };
  auto extract_b = [&](auto &source_cu, uint32_t base, const amdgpu::InputLoc &loc) {
    ++source_b_reads;
    return amdgpu::extract_f32(source_cu, base, loc);
  };

  ForceScalarGuard force_scalar(/*force_scalar=*/true);
  amdgpu::exec_f32(*cu, M, N, K, B, /*in_bits=*/32, vb + destination, vb + source_a, vb + source_b,
                   vb + accumulator, extract_a, extract_b, amdgpu::ACC_FROM_VGPR,
                   /*cbsz=*/1, /*abid=*/0, /*blgp=*/1);

  EXPECT_EQ(source_a_reads, static_cast<size_t>(M) * K * B);
  EXPECT_EQ(source_b_reads, static_cast<size_t>(N) * K * B);
  for (uint32_t reg = 0; reg < destination_regs; ++reg)
    for (uint32_t lane = 0; lane < wf_size; ++lane)
      EXPECT_FLOAT_EQ(std::bit_cast<float>(cu->read_vgpr(vb + destination + reg, lane)), 11.0f);
}

// ---------------------------------------------------------------------------
// L2 cache tests
// ---------------------------------------------------------------------------

TEST(L2CacheTest, UcStoreInvalidatesResidentLineBeforeAtomicRmw) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0x2000;
  mem.write32(kAddr, 1);

  uint32_t first_old = 0;
  l2.atomic_rmw(kAddr, sizeof(uint32_t), [&](uint8_t *line, uint32_t offset) {
    std::memcpy(&first_old, line + offset, sizeof(first_old));
    std::memcpy(line + offset, &first_old, sizeof(first_old));
  });
  ASSERT_EQ(first_old, 1u);

  const uint32_t unlocked = 0;
  l2.write(kAddr, reinterpret_cast<const uint8_t *>(&unlocked), sizeof(unlocked),
           amdgpu::Mtype::UC);

  uint32_t second_old = 1;
  l2.atomic_rmw(kAddr, sizeof(uint32_t), [&](uint8_t *line, uint32_t offset) {
    std::memcpy(&second_old, line + offset, sizeof(second_old));
    std::memcpy(line + offset, &second_old, sizeof(second_old));
  });
  EXPECT_EQ(second_old, 0u);
}

TEST(L2CacheTest, AtomicRmwRefetchesLinesCachedByAnotherXcd) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache first_l2("first_l2");
  amdgpu::L2Cache second_l2("second_l2");
  first_l2.set_backing_memory(&mem);
  second_l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0x2040;
  mem.write32(kAddr, 1);

  auto atomic_add = [&](amdgpu::L2Cache &l2, uint32_t increment) {
    uint32_t old_value = 0;
    l2.atomic_rmw(kAddr, sizeof(uint32_t), [&](uint8_t *line, uint32_t offset) {
      std::memcpy(&old_value, line + offset, sizeof(old_value));
      const uint32_t new_value = old_value + increment;
      std::memcpy(line + offset, &new_value, sizeof(new_value));
    });
    return old_value;
  };

  EXPECT_EQ(atomic_add(first_l2, 0), 1u);
  EXPECT_EQ(atomic_add(second_l2, 0), 1u);
  EXPECT_EQ(atomic_add(first_l2, 1), 1u);
  EXPECT_EQ(atomic_add(second_l2, 1), 2u);
  EXPECT_EQ(mem.read32(kAddr), 3u);
}

TEST(L2CacheTest, UcReadFlushesDirtyResidentLine) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0x2080;
  mem.write32(kAddr, 7);

  uint8_t line[amdgpu::L2Cache::LINE_SIZE] = {};
  uint32_t dirty_value = 9;
  std::memcpy(line, &dirty_value, sizeof(dirty_value));
  l2.writeback_line(kAddr, line, amdgpu::Mtype::RW);

  uint32_t read_value = 0;
  l2.read(kAddr, reinterpret_cast<uint8_t *>(&read_value), sizeof(read_value), amdgpu::Mtype::UC);
  EXPECT_EQ(read_value, 9u);
}

TEST(L2CacheTest, UcReadCrossingLineBoundaryFlushesBothDirtyResidentLines) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  l2.set_backing_memory(&mem);

  constexpr uint64_t kBase = 0x3000;
  constexpr uint64_t kAddr = kBase + amdgpu::L2Cache::LINE_SIZE - 4;
  static_assert((kBase & (amdgpu::L2Cache::LINE_SIZE - 1)) == 0);

  std::array<uint8_t, amdgpu::L2Cache::LINE_SIZE> first_line{};
  std::array<uint8_t, amdgpu::L2Cache::LINE_SIZE> second_line{};
  for (uint32_t i = 0; i < amdgpu::L2Cache::LINE_SIZE; ++i) {
    first_line[i] = static_cast<uint8_t>(0x10u + i);
    second_line[i] = static_cast<uint8_t>(0x80u + i);
    mem.write8(kBase + i, 0xA0u);
    mem.write8(kBase + amdgpu::L2Cache::LINE_SIZE + i, 0xB0u);
  }

  l2.writeback_line(kBase, first_line.data(), amdgpu::Mtype::RW);
  l2.writeback_line(kBase + amdgpu::L2Cache::LINE_SIZE, second_line.data(), amdgpu::Mtype::RW);

  std::array<uint8_t, 8> read_value{};
  l2.read(kAddr, read_value.data(), read_value.size(), amdgpu::Mtype::UC);

  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(read_value[i], first_line[amdgpu::L2Cache::LINE_SIZE - 4 + i])
        << "first line byte " << i;
    EXPECT_EQ(read_value[4 + i], second_line[i]) << "second line byte " << i;
  }
}

TEST(L2CacheTest, UcWriteCrossingLineBoundaryFlushesBothDirtyResidentLines) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  l2.set_backing_memory(&mem);

  constexpr uint64_t kBase = 0x4000;
  constexpr uint64_t kAddr = kBase + amdgpu::L2Cache::LINE_SIZE - 4;
  static_assert((kBase & (amdgpu::L2Cache::LINE_SIZE - 1)) == 0);

  std::array<uint8_t, amdgpu::L2Cache::LINE_SIZE> first_line{};
  std::array<uint8_t, amdgpu::L2Cache::LINE_SIZE> second_line{};
  for (uint32_t i = 0; i < amdgpu::L2Cache::LINE_SIZE; ++i) {
    first_line[i] = static_cast<uint8_t>(0x20u + i);
    second_line[i] = static_cast<uint8_t>(0x90u + i);
    mem.write8(kBase + i, 0xC0u);
    mem.write8(kBase + amdgpu::L2Cache::LINE_SIZE + i, 0xD0u);
  }

  l2.writeback_line(kBase, first_line.data(), amdgpu::Mtype::RW);
  l2.writeback_line(kBase + amdgpu::L2Cache::LINE_SIZE, second_line.data(), amdgpu::Mtype::RW);

  const std::array<uint8_t, 8> store_value = {0x01u, 0x02u, 0x03u, 0x04u,
                                              0x05u, 0x06u, 0x07u, 0x08u};
  l2.write(kAddr, store_value.data(), store_value.size(), amdgpu::Mtype::UC);

  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(mem.read8(kAddr + i), store_value[i]) << "first line store byte " << i;
    EXPECT_EQ(mem.read8(kAddr + 4 + i), store_value[4 + i]) << "second line store byte " << i;
  }
  EXPECT_EQ(mem.read8(kBase), first_line[0])
      << "dirty byte outside the first-line UC store should be preserved";
  EXPECT_EQ(mem.read8(kBase + amdgpu::L2Cache::LINE_SIZE + 4), second_line[4])
      << "dirty byte outside the second-line UC store should be preserved";
}

TEST(L1ScalarCacheTest, UcReadInvalidatesResidentWriteThroughLine) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1ScalarCache l1(&l2);
  l2.set_backing_memory(&mem);
  l1.set_memory(&mem);

  constexpr uint32_t kVmid = 1;
  constexpr uint64_t kAddr = 0x5000;
  constexpr uint32_t kBackingValue = 0x11111111;
  constexpr uint32_t kStoredValue = 0x22222222;
  constexpr uint32_t kReloadValue = 0x33333333;

  std::array<uint8_t, KfdProcess::kPageSize> backing{};
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  page_table[kAddr >> KfdProcess::kPageShift] = {backing.data(), amdgpu::Mtype::RW};
  mem.register_process(kVmid, &page_table, &page_table_mutex);

  mem.write32(kAddr, kBackingValue, kVmid);
  l1.store(kAddr, /*num_dwords=*/1, &kStoredValue, kVmid);

  {
    std::unique_lock lock(page_table_mutex);
    page_table[kAddr >> KfdProcess::kPageShift].mtype = amdgpu::Mtype::UC;
  }
  uint32_t read_value = 0;
  l1.load(kAddr, /*num_dwords=*/1, &read_value, kVmid);

  EXPECT_EQ(read_value, kStoredValue);
  EXPECT_EQ(mem.read32(kAddr, kVmid), kStoredValue);

  l2.write(kAddr, reinterpret_cast<const uint8_t *>(&kReloadValue), sizeof(kReloadValue),
           amdgpu::Mtype::RW, kVmid);
  {
    std::unique_lock lock(page_table_mutex);
    page_table[kAddr >> KfdProcess::kPageShift].mtype = amdgpu::Mtype::RW;
  }

  read_value = 0;
  l1.load(kAddr, /*num_dwords=*/1, &read_value, kVmid);
  EXPECT_EQ(read_value, kReloadValue);
}

TEST(L1ScalarCacheTest, UcLoadBytesInvalidatesResidentWriteThroughLine) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1ScalarCache l1(&l2);
  l2.set_backing_memory(&mem);
  l1.set_memory(&mem);

  constexpr uint32_t kVmid = 4;
  constexpr uint64_t kAddr = 0x5400;
  constexpr uint32_t kBackingValue = 0x11111111;
  constexpr uint32_t kStoredValue = 0x44332211;
  constexpr uint32_t kReloadValue = 0x88776655;

  std::array<uint8_t, KfdProcess::kPageSize> backing{};
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  page_table[kAddr >> KfdProcess::kPageShift] = {backing.data(), amdgpu::Mtype::RW};
  mem.register_process(kVmid, &page_table, &page_table_mutex);

  mem.write32(kAddr, kBackingValue, kVmid);
  l1.store(kAddr, /*num_dwords=*/1, &kStoredValue, kVmid);

  {
    std::unique_lock lock(page_table_mutex);
    page_table[kAddr >> KfdProcess::kPageShift].mtype = amdgpu::Mtype::UC;
  }

  std::array<uint8_t, 2> read_bytes{};
  l1.load_bytes(kAddr + 1, read_bytes.size(), read_bytes.data(), kVmid);

  EXPECT_EQ(read_bytes[0], 0x22);
  EXPECT_EQ(read_bytes[1], 0x33);
  EXPECT_EQ(mem.read32(kAddr, kVmid), kStoredValue);

  l2.write(kAddr, reinterpret_cast<const uint8_t *>(&kReloadValue), sizeof(kReloadValue),
           amdgpu::Mtype::RW, kVmid);
  {
    std::unique_lock lock(page_table_mutex);
    page_table[kAddr >> KfdProcess::kPageShift].mtype = amdgpu::Mtype::RW;
  }

  read_bytes.fill(0);
  l1.load_bytes(kAddr + 1, read_bytes.size(), read_bytes.data(), kVmid);
  EXPECT_EQ(read_bytes[0], 0x66);
  EXPECT_EQ(read_bytes[1], 0x77);
}

TEST(L1ScalarCacheTest, CcReadInvalidatesResidentWriteThroughLine) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1ScalarCache l1(&l2);
  l2.set_backing_memory(&mem);
  l1.set_memory(&mem);

  constexpr uint32_t kVmid = 5;
  constexpr uint64_t kAddr = 0x5800;
  constexpr uint32_t kBackingValue = 0x11111111;
  constexpr uint32_t kStoredValue = 0x22222222;
  constexpr uint32_t kReloadValue = 0x33333333;

  std::array<uint8_t, KfdProcess::kPageSize> backing{};
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  page_table[kAddr >> KfdProcess::kPageShift] = {backing.data(), amdgpu::Mtype::RW};
  mem.register_process(kVmid, &page_table, &page_table_mutex);

  mem.write32(kAddr, kBackingValue, kVmid);
  l1.store(kAddr, /*num_dwords=*/1, &kStoredValue, kVmid);

  {
    std::unique_lock lock(page_table_mutex);
    page_table[kAddr >> KfdProcess::kPageShift].mtype = amdgpu::Mtype::CC;
  }
  uint32_t read_value = 0;
  l1.load(kAddr, /*num_dwords=*/1, &read_value, kVmid);

  EXPECT_EQ(read_value, kStoredValue);
  EXPECT_EQ(mem.read32(kAddr, kVmid), kStoredValue);

  l2.write(kAddr, reinterpret_cast<const uint8_t *>(&kReloadValue), sizeof(kReloadValue),
           amdgpu::Mtype::RW, kVmid);
  {
    std::unique_lock lock(page_table_mutex);
    page_table[kAddr >> KfdProcess::kPageShift].mtype = amdgpu::Mtype::RW;
  }

  read_value = 0;
  l1.load(kAddr, /*num_dwords=*/1, &read_value, kVmid);
  EXPECT_EQ(read_value, kReloadValue);
}

TEST(L1ScalarCacheTest, CcLoadBytesInvalidatesResidentWriteThroughLine) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1ScalarCache l1(&l2);
  l2.set_backing_memory(&mem);
  l1.set_memory(&mem);

  constexpr uint32_t kVmid = 6;
  constexpr uint64_t kAddr = 0x5C00;
  constexpr uint32_t kBackingValue = 0x11111111;
  constexpr uint32_t kStoredValue = 0x44332211;
  constexpr uint32_t kReloadValue = 0x88776655;

  std::array<uint8_t, KfdProcess::kPageSize> backing{};
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  page_table[kAddr >> KfdProcess::kPageShift] = {backing.data(), amdgpu::Mtype::RW};
  mem.register_process(kVmid, &page_table, &page_table_mutex);

  mem.write32(kAddr, kBackingValue, kVmid);
  l1.store(kAddr, /*num_dwords=*/1, &kStoredValue, kVmid);

  {
    std::unique_lock lock(page_table_mutex);
    page_table[kAddr >> KfdProcess::kPageShift].mtype = amdgpu::Mtype::CC;
  }

  std::array<uint8_t, 2> read_bytes{};
  l1.load_bytes(kAddr + 1, read_bytes.size(), read_bytes.data(), kVmid);

  EXPECT_EQ(read_bytes[0], 0x22);
  EXPECT_EQ(read_bytes[1], 0x33);
  EXPECT_EQ(mem.read32(kAddr, kVmid), kStoredValue);

  l2.write(kAddr, reinterpret_cast<const uint8_t *>(&kReloadValue), sizeof(kReloadValue),
           amdgpu::Mtype::RW, kVmid);
  {
    std::unique_lock lock(page_table_mutex);
    page_table[kAddr >> KfdProcess::kPageShift].mtype = amdgpu::Mtype::RW;
  }

  read_bytes.fill(0);
  l1.load_bytes(kAddr + 1, read_bytes.size(), read_bytes.data(), kVmid);
  EXPECT_EQ(read_bytes[0], 0x66);
  EXPECT_EQ(read_bytes[1], 0x77);
}

TEST(L1ScalarCacheTest, UcWriteInvalidatesResidentLineBeforeBypassStore) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1ScalarCache l1(&l2);
  l2.set_backing_memory(&mem);
  l1.set_memory(&mem);

  constexpr uint32_t kVmid = 2;
  constexpr uint64_t kBase = 0x6000;
  constexpr uint64_t kStoreAddr = kBase + 4;
  constexpr uint32_t kStoredOutsideValue = 0x11111111;
  constexpr uint32_t kStoredTargetValue = 0x22222222;
  constexpr uint32_t kUcStoreValue = 0x33333333;

  std::array<uint8_t, KfdProcess::kPageSize> backing{};
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  page_table[kBase >> KfdProcess::kPageShift] = {backing.data(), amdgpu::Mtype::RW};
  mem.register_process(kVmid, &page_table, &page_table_mutex);

  const uint32_t stored_values[] = {kStoredOutsideValue, kStoredTargetValue};
  l1.store(kBase, /*num_dwords=*/2, stored_values, kVmid);

  {
    std::unique_lock lock(page_table_mutex);
    page_table[kBase >> KfdProcess::kPageShift].mtype = amdgpu::Mtype::UC;
  }
  l1.store(kStoreAddr, /*num_dwords=*/1, &kUcStoreValue, kVmid);
  l1.writeback_all(kVmid);

  EXPECT_EQ(mem.read32(kBase, kVmid), kStoredOutsideValue);
  EXPECT_EQ(mem.read32(kStoreAddr, kVmid), kUcStoreValue);
}

TEST(L1ScalarCacheTest, CcWriteInvalidatesResidentLineBeforeBypassStore) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1ScalarCache l1(&l2);
  l2.set_backing_memory(&mem);
  l1.set_memory(&mem);

  constexpr uint32_t kVmid = 7;
  constexpr uint64_t kBase = 0x6400;
  constexpr uint64_t kStoreAddr = kBase + 4;
  constexpr uint32_t kBackingOutsideValue = 0x01010101;
  constexpr uint32_t kBackingTargetValue = 0x02020202;
  constexpr uint32_t kStoredOutsideValue = 0x11111111;
  constexpr uint32_t kStoredTargetValue = 0x22222222;
  constexpr uint32_t kCcStoreValue = 0x33333333;

  std::array<uint8_t, KfdProcess::kPageSize> backing{};
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  page_table[kBase >> KfdProcess::kPageShift] = {backing.data(), amdgpu::Mtype::RW};
  mem.register_process(kVmid, &page_table, &page_table_mutex);

  mem.write32(kBase, kBackingOutsideValue, kVmid);
  mem.write32(kStoreAddr, kBackingTargetValue, kVmid);
  const uint32_t stored_values[] = {kStoredOutsideValue, kStoredTargetValue};
  l1.store(kBase, /*num_dwords=*/2, stored_values, kVmid);

  {
    std::unique_lock lock(page_table_mutex);
    page_table[kBase >> KfdProcess::kPageShift].mtype = amdgpu::Mtype::CC;
  }
  l1.store(kStoreAddr, /*num_dwords=*/1, &kCcStoreValue, kVmid);

  EXPECT_EQ(mem.read32(kBase, kVmid), kStoredOutsideValue);
  EXPECT_EQ(mem.read32(kStoreAddr, kVmid), kCcStoreValue);
}

TEST(L1ScalarCacheTest, WriteThroughStoresAndCleanEvictionReachBacking) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1ScalarCache l1(&l2);
  l2.set_backing_memory(&mem);

  constexpr uint64_t kBase = 0x9000;
  constexpr uint64_t kSetStride = uint64_t{1}
                                  << (amdgpu::L1ScalarCache::LINE_SIZE_BITS +
                                      std::bit_width(amdgpu::L1ScalarCache::NUM_SETS - 1));
  static_assert(amdgpu::L1ScalarCache::ASSOCIATIVITY == 4);

  std::array<uint64_t, 5> addrs{};
  std::array<uint32_t, 5> values{};
  for (uint32_t i = 0; i < addrs.size(); ++i) {
    addrs[i] = kBase + i * kSetStride;
    values[i] = 0x11110000u + i;
  }

  for (uint32_t i = 0; i < 4; ++i) {
    l1.store(addrs[i], /*num_dwords=*/1, &values[i]);
    EXPECT_EQ(mem.read32(addrs[i]), values[i]) << "line " << i;
  }

  l1.store(addrs[4], /*num_dwords=*/1, &values[4]);
  EXPECT_EQ(mem.read32(addrs[4]), values[4]);

  l1.writeback_all();
  for (uint32_t i = 0; i < addrs.size(); ++i)
    EXPECT_EQ(mem.read32(addrs[i]), values[i]) << "line " << i;
}

TEST(L1ScalarCacheTest, CacheableStoresWriteThroughBeforeWriteback) {
  constexpr uint32_t kVmid = 9;
  constexpr uint64_t kAddr = 0x6800;

  for (const auto mtype : {amdgpu::Mtype::RW, amdgpu::Mtype::WB, amdgpu::Mtype::NT}) {
    SCOPED_TRACE(static_cast<int>(mtype));
    amdgpu::GpuMemory mem("test_mem");
    amdgpu::L2Cache l2("test_l2");
    amdgpu::L1ScalarCache l1(&l2);
    l2.set_backing_memory(&mem);
    l1.set_memory(&mem);

    std::array<uint8_t, KfdProcess::kPageSize> backing{};
    KfdProcess::PageTable page_table;
    std::shared_mutex page_table_mutex;
    page_table[kAddr >> KfdProcess::kPageShift] = {backing.data(), mtype};
    mem.register_process(kVmid, &page_table, &page_table_mutex);

    const uint32_t value = 0xCAFE0000u + static_cast<uint32_t>(mtype);
    l1.store(kAddr, /*num_dwords=*/1, &value, kVmid);
    EXPECT_EQ(mem.read32(kAddr, kVmid), value);

    l1.writeback_all(kVmid);
    EXPECT_EQ(mem.read32(kAddr, kVmid), value);
    mem.unregister_process(kVmid);
  }
}

TEST(L1ScalarCacheTest, UnalignedStoreCrossingLineWritesThroughExactBytes) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1ScalarCache l1(&l2);
  l2.set_backing_memory(&mem);
  l1.set_memory(&mem);

  constexpr uint32_t kVmid = 10;
  constexpr uint64_t kPageBase = 0x7000;
  constexpr uint64_t kAddr = kPageBase + (uint64_t{1} << amdgpu::L1ScalarCache::LINE_SIZE_BITS) - 2;
  constexpr uint32_t kValue = 0x44332211;

  std::array<uint8_t, KfdProcess::kPageSize> backing{};
  backing.fill(0xA5);
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  page_table[kPageBase >> KfdProcess::kPageShift] = {backing.data(), amdgpu::Mtype::RW};
  mem.register_process(kVmid, &page_table, &page_table_mutex);

  l1.store(kAddr, /*num_dwords=*/1, &kValue, kVmid);

  std::array<uint8_t, sizeof(kValue)> expected{};
  std::memcpy(expected.data(), &kValue, sizeof(kValue));
  const size_t backing_offset = kAddr - kPageBase;
  EXPECT_EQ(backing[backing_offset - 1], 0xA5);
  for (size_t i = 0; i < expected.size(); ++i)
    EXPECT_EQ(backing[backing_offset + i], expected[i]) << "byte " << i;
  EXPECT_EQ(backing[backing_offset + expected.size()], 0xA5);

  l1.writeback_all(kVmid);
  for (size_t i = 0; i < expected.size(); ++i)
    EXPECT_EQ(backing[backing_offset + i], expected[i]) << "byte after writeback " << i;
  mem.unregister_process(kVmid);
}

TEST(L1ScalarCacheTest, ScalarWritebackDoesNotClobberAtomicAtDisjointAddress) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2a("l2a");
  amdgpu::L2Cache l2b("l2b");
  l2a.set_backing_memory(&mem);
  l2b.set_backing_memory(&mem);
  amdgpu::L1ScalarCache l1(&l2a);
  l1.set_memory(&mem);

  constexpr uint32_t kVmid = 17;
  constexpr uint64_t kVa = 0x500000;
  constexpr uint32_t kScalarValue = 0x5A5A5A5A;
  std::array<uint8_t, KfdProcess::kPageSize> backing{};
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  page_table[kVa >> KfdProcess::kPageShift] = {backing.data(), amdgpu::Mtype::RW};
  mem.register_process(kVmid, &page_table, &page_table_mutex);

  l1.store(kVa + sizeof(uint32_t), /*num_dwords=*/1, &kScalarValue, kVmid);
  l2b.atomic_rmw(
      kVa, sizeof(uint32_t),
      [](uint8_t *storage, uint32_t offset) {
        uint32_t value = 0;
        std::memcpy(&value, storage + offset, sizeof(value));
        ++value;
        std::memcpy(storage + offset, &value, sizeof(value));
      },
      kVmid);

  l1.writeback_all(kVmid);
  EXPECT_EQ(mem.read32(kVa, kVmid), 1u);
  EXPECT_EQ(mem.read32(kVa + sizeof(uint32_t), kVmid), kScalarValue);
  mem.unregister_process(kVmid);
}

TEST(L1ScalarCacheTest, CleanEvictionDoesNotClobberAtomicAtDisjointAddress) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2a("l2a");
  amdgpu::L2Cache l2b("l2b");
  l2a.set_backing_memory(&mem);
  l2b.set_backing_memory(&mem);
  amdgpu::L1ScalarCache l1(&l2a);

  constexpr uint64_t kBase = 0x600000;
  constexpr uint64_t kSetStride = uint64_t{1}
                                  << (amdgpu::L1ScalarCache::LINE_SIZE_BITS +
                                      std::bit_width(amdgpu::L1ScalarCache::NUM_SETS - 1));
  constexpr uint32_t kScalarValue = 0x6B6B6B6B;
  l1.store(kBase + sizeof(uint32_t), /*num_dwords=*/1, &kScalarValue);
  l2b.atomic_rmw(kBase, sizeof(uint32_t), [](uint8_t *storage, uint32_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, storage + offset, sizeof(value));
    ++value;
    std::memcpy(storage + offset, &value, sizeof(value));
  });

  for (uint32_t i = 1; i <= amdgpu::L1ScalarCache::ASSOCIATIVITY; ++i) {
    uint32_t ignored = 0;
    l1.load(kBase + i * kSetStride, /*num_dwords=*/1, &ignored);
  }

  EXPECT_EQ(mem.read32(kBase), 1u);
  EXPECT_EQ(mem.read32(kBase + sizeof(uint32_t)), kScalarValue);
}

TEST(L1ScalarCacheTest, UcAndCcFlushDoNotClobberAtomicAtDisjointAddress) {
  constexpr uint32_t kVmid = 18;
  constexpr uint64_t kVa = 0x700000;
  constexpr uint32_t kScalarValue = 0x7C7C7C7C;

  for (const auto mtype : {amdgpu::Mtype::UC, amdgpu::Mtype::CC}) {
    SCOPED_TRACE(static_cast<int>(mtype));
    amdgpu::GpuMemory mem("test_mem");
    amdgpu::L2Cache l2a("l2a");
    amdgpu::L2Cache l2b("l2b");
    l2a.set_backing_memory(&mem);
    l2b.set_backing_memory(&mem);
    amdgpu::L1ScalarCache l1(&l2a);
    l1.set_memory(&mem);

    std::array<uint8_t, KfdProcess::kPageSize> backing{};
    KfdProcess::PageTable page_table;
    std::shared_mutex page_table_mutex;
    page_table[kVa >> KfdProcess::kPageShift] = {backing.data(), amdgpu::Mtype::RW};
    mem.register_process(kVmid, &page_table, &page_table_mutex);

    l1.store(kVa + sizeof(uint32_t), /*num_dwords=*/1, &kScalarValue, kVmid);
    l2b.atomic_rmw(
        kVa, sizeof(uint32_t),
        [](uint8_t *storage, uint32_t offset) {
          uint32_t value = 0;
          std::memcpy(&value, storage + offset, sizeof(value));
          ++value;
          std::memcpy(storage + offset, &value, sizeof(value));
        },
        kVmid);

    {
      std::unique_lock lock(page_table_mutex);
      page_table[kVa >> KfdProcess::kPageShift].mtype = mtype;
    }
    uint32_t ignored = 0;
    l1.load(kVa + 2 * sizeof(uint32_t), /*num_dwords=*/1, &ignored, kVmid);

    EXPECT_EQ(mem.read32(kVa, kVmid), 1u);
    EXPECT_EQ(mem.read32(kVa + sizeof(uint32_t), kVmid), kScalarValue);
    mem.unregister_process(kVmid);
  }
}

TEST(DeviceCacheCoherenceTest, ScalarWriteThroughCannotClobberRemoteAtomic) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache scalar_l2("scalar_l2");
  amdgpu::L2Cache atomic_l2("atomic_l2");
  amdgpu::L1ScalarCache scalar_l1(&scalar_l2);
  scalar_l2.set_backing_memory(&mem);
  atomic_l2.set_backing_memory(&mem);

  constexpr uint64_t kLine = 0xA000;
  constexpr uint64_t kAtomicAddr = kLine;
  constexpr uint64_t kScalarAddr = kLine + sizeof(uint32_t);
  constexpr uint32_t kScalarValue = 0xA5A5A5A5;
  mem.write32(kAtomicAddr, 0);
  mem.write32(kScalarAddr, 0);

  // The scalar store read-allocates the atomic dword's old value into the same
  // K$ line. The store must update only its target bytes, and the atomic must
  // invalidate the stale clean snapshot.
  scalar_l1.store(kScalarAddr, /*num_dwords=*/1, &kScalarValue);
  atomic_l2.atomic_rmw(kAtomicAddr, sizeof(uint32_t), [](uint8_t *line, uint32_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, line + offset, sizeof(value));
    ++value;
    std::memcpy(line + offset, &value, sizeof(value));
  });
  scalar_l1.writeback_all();

  EXPECT_EQ(mem.read32(kAtomicAddr), 1u);
  EXPECT_EQ(mem.read32(kScalarAddr), kScalarValue);
}

TEST(DeviceCacheCoherenceTest, DisjointScalarWriteThroughStoresSurviveRemoteAtomic) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache first_scalar_l2("first_scalar_l2");
  amdgpu::L2Cache second_scalar_l2("second_scalar_l2");
  amdgpu::L2Cache atomic_l2("atomic_l2");
  amdgpu::L1ScalarCache first_scalar_l1(&first_scalar_l2);
  amdgpu::L1ScalarCache second_scalar_l1(&second_scalar_l2);
  first_scalar_l2.set_backing_memory(&mem);
  second_scalar_l2.set_backing_memory(&mem);
  atomic_l2.set_backing_memory(&mem);

  constexpr uint64_t kLine = 0xA080;
  constexpr uint64_t kAtomicAddr = kLine;
  constexpr uint64_t kFirstScalarAddr = kLine + sizeof(uint32_t);
  constexpr uint64_t kSecondScalarAddr = kLine + 2 * sizeof(uint32_t);
  constexpr uint32_t kFirstScalarValue = 0x11112222;
  constexpr uint32_t kSecondScalarValue = 0x33334444;
  mem.write32(kAtomicAddr, 0);
  mem.write32(kFirstScalarAddr, 0);
  mem.write32(kSecondScalarAddr, 0);

  // Both K$ instances fill the same old line. Their disjoint write-through
  // stores must merge without either cached snapshot replacing the other.
  first_scalar_l1.store(kFirstScalarAddr, /*num_dwords=*/1, &kFirstScalarValue);
  second_scalar_l1.store(kSecondScalarAddr, /*num_dwords=*/1, &kSecondScalarValue);
  atomic_l2.atomic_rmw(kAtomicAddr, sizeof(uint32_t), [](uint8_t *line, uint32_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, line + offset, sizeof(value));
    ++value;
    std::memcpy(line + offset, &value, sizeof(value));
  });
  first_scalar_l1.writeback_all();
  second_scalar_l1.writeback_all();

  EXPECT_EQ(mem.read32(kAtomicAddr), 1u);
  EXPECT_EQ(mem.read32(kFirstScalarAddr), kFirstScalarValue);
  EXPECT_EQ(mem.read32(kSecondScalarAddr), kSecondScalarValue);
}

TEST(DeviceCacheCoherenceTest, RemoteAtomicInvalidatesScalarCachedRead) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache scalar_l2("scalar_l2");
  amdgpu::L2Cache atomic_l2("atomic_l2");
  amdgpu::L1ScalarCache scalar_l1(&scalar_l2);
  scalar_l2.set_backing_memory(&mem);
  atomic_l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0xA100;
  mem.write32(kAddr, 10);
  uint32_t value = 0;
  scalar_l1.load(kAddr, /*num_dwords=*/1, &value);
  ASSERT_EQ(value, 10u);

  atomic_l2.atomic_rmw(kAddr, sizeof(uint32_t), [](uint8_t *line, uint32_t offset) {
    uint32_t current = 0;
    std::memcpy(&current, line + offset, sizeof(current));
    ++current;
    std::memcpy(line + offset, &current, sizeof(current));
  });

  value = 0;
  scalar_l1.load(kAddr, /*num_dwords=*/1, &value);
  EXPECT_EQ(value, 11u);
}

TEST(DeviceCacheCoherenceTest, RemoteAtomicInvalidatesVectorCachedRead) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache vector_l2("vector_l2");
  amdgpu::L2Cache atomic_l2("atomic_l2");
  amdgpu::L1VectorCache vector_l1(&vector_l2);
  vector_l2.set_backing_memory(&mem);
  atomic_l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0xA200;
  mem.write32(kAddr, 10);
  uint64_t addrs[cdna3::Isa::WF_SIZE] = {};
  addrs[0] = kAddr;
  std::array<uint8_t, cdna3::Isa::WF_SIZE * sizeof(uint32_t)> bytes{};
  auto load_value = [&] {
    bytes.fill(0);
    vector_l1.load(addrs, /*lane_mask=*/1, /*elem_size=*/sizeof(uint32_t),
                   /*num_elems=*/1, bytes.data(), amdgpu::Mtype::RW,
                   /*non_temporal=*/false, /*request_l1_bypass=*/false, cdna3::Isa::WF_SIZE);
    uint32_t value = 0;
    std::memcpy(&value, bytes.data(), sizeof(value));
    return value;
  };
  ASSERT_EQ(load_value(), 10u);

  atomic_l2.atomic_rmw(kAddr, sizeof(uint32_t), [](uint8_t *line, uint32_t offset) {
    uint32_t current = 0;
    std::memcpy(&current, line + offset, sizeof(current));
    ++current;
    std::memcpy(line + offset, &current, sizeof(current));
  });

  EXPECT_EQ(load_value(), 11u);
}

TEST(DeviceCacheCoherenceTest, AtomicConsumesScalarWriteThroughTarget) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache scalar_l2("scalar_l2");
  amdgpu::L2Cache atomic_l2("atomic_l2");
  amdgpu::L1ScalarCache scalar_l1(&scalar_l2);
  scalar_l2.set_backing_memory(&mem);
  atomic_l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0xA300;
  constexpr uint32_t kStoredValue = 40;
  mem.write32(kAddr, 0);
  scalar_l1.store(kAddr, /*num_dwords=*/1, &kStoredValue);

  atomic_l2.atomic_rmw(kAddr, sizeof(uint32_t), [](uint8_t *line, uint32_t offset) {
    uint32_t current = 0;
    std::memcpy(&current, line + offset, sizeof(current));
    ++current;
    std::memcpy(line + offset, &current, sizeof(current));
  });

  EXPECT_EQ(mem.read32(kAddr), 41u);
  uint32_t reloaded = 0;
  scalar_l1.load(kAddr, /*num_dwords=*/1, &reloaded);
  EXPECT_EQ(reloaded, 41u);
}

TEST(DeviceCacheCoherenceTest, DestroyedCachesAreRemovedFromRegistry) {
  amdgpu::GpuMemory mem("test_mem");
  constexpr uint64_t kAddr = 0xA400;

  struct alignas(amdgpu::L2Cache) L2Storage {
    std::byte data[sizeof(amdgpu::L2Cache)];
  };
  struct alignas(amdgpu::L1ScalarCache) ScalarStorage {
    std::byte data[sizeof(amdgpu::L1ScalarCache)];
  };
  struct alignas(amdgpu::L1VectorCache) VectorStorage {
    std::byte data[sizeof(amdgpu::L1VectorCache)];
  };

  auto l2_storage = std::make_unique<L2Storage>();
  auto scalar_storage = std::make_unique<ScalarStorage>();
  auto vector_storage = std::make_unique<VectorStorage>();
  auto *transient_l2 =
      std::construct_at(reinterpret_cast<amdgpu::L2Cache *>(l2_storage->data), "transient_l2");
  auto *transient_scalar = std::construct_at(
      reinterpret_cast<amdgpu::L1ScalarCache *>(scalar_storage->data), transient_l2);
  auto *transient_vector = std::construct_at(
      reinterpret_cast<amdgpu::L1VectorCache *>(vector_storage->data), transient_l2);
  transient_l2->set_backing_memory(&mem);

  constexpr uint32_t kDirty = 7;
  transient_scalar->store(kAddr + sizeof(uint32_t), /*num_dwords=*/1, &kDirty);
  uint64_t addrs[cdna3::Isa::WF_SIZE] = {};
  addrs[0] = kAddr;
  std::array<uint8_t, cdna3::Isa::WF_SIZE * sizeof(uint32_t)> bytes{};
  transient_vector->load(addrs, /*lane_mask=*/1, /*elem_size=*/sizeof(uint32_t),
                         /*num_elems=*/1, bytes.data(), amdgpu::Mtype::RW,
                         /*non_temporal=*/false, /*request_l1_bypass=*/false, cdna3::Isa::WF_SIZE);

  std::destroy_at(transient_vector);
  std::destroy_at(transient_scalar);
  std::destroy_at(transient_l2);
  std::memset(vector_storage->data, 0xA5, sizeof(vector_storage->data));
  std::memset(scalar_storage->data, 0xA5, sizeof(scalar_storage->data));
  std::memset(l2_storage->data, 0xA5, sizeof(l2_storage->data));

  amdgpu::L2Cache survivor("survivor");
  EXPECT_NE(static_cast<const void *>(&survivor), static_cast<const void *>(transient_l2));
  survivor.set_backing_memory(&mem);
  mem.write32(kAddr, 0);
  survivor.atomic_rmw(kAddr, sizeof(uint32_t), [](uint8_t *line, uint32_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, line + offset, sizeof(value));
    ++value;
    std::memcpy(line + offset, &value, sizeof(value));
  });
  EXPECT_EQ(mem.read32(kAddr), 1u);
}

TEST(L1VectorCacheTest, UcReadInvalidatesResidentLine) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1VectorCache l1(&l2);
  l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0x7000;
  constexpr uint32_t kOldValue = 0x11111111;
  constexpr uint32_t kNewValue = 0x22222222;

  uint64_t addrs[64] = {};
  addrs[0] = kAddr;
  std::array<uint8_t, 64 * sizeof(uint32_t)> bytes{};

  mem.write32(kAddr, kOldValue);
  l1.load(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, bytes.data(),
          amdgpu::Mtype::RW, /*non_temporal=*/false, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE);
  uint32_t value = 0;
  std::memcpy(&value, bytes.data(), sizeof(value));
  ASSERT_EQ(value, kOldValue);

  mem.write32(kAddr, kNewValue);
  bytes.fill(0);
  l1.load(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, bytes.data(),
          amdgpu::Mtype::UC, /*non_temporal=*/false, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE);
  std::memcpy(&value, bytes.data(), sizeof(value));
  ASSERT_EQ(value, kNewValue);

  bytes.fill(0);
  l1.load(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, bytes.data(),
          amdgpu::Mtype::RW, /*non_temporal=*/false, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE);
  std::memcpy(&value, bytes.data(), sizeof(value));
  EXPECT_EQ(value, kNewValue);
}

TEST(GpuMemoryTest, BlockAccessSpansSparseFallbackPages) {
  amdgpu::GpuMemory mem("test_mem");
  constexpr uint64_t kAddr = amdgpu::GpuMemory::PAGE_SIZE - 8;
  constexpr std::array<uint8_t, 16> kData = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                                             0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};

  mem.write_block(kAddr, std::span<const uint8_t>(kData));
  std::array<uint8_t, kData.size()> result{};
  mem.read_block(kAddr, std::span<uint8_t>(result));

  EXPECT_EQ(result, kData);
}

TEST(GpuMemoryTest, BlockAccessSpansMappedPages) {
  amdgpu::GpuMemory mem("test_mem");
  constexpr uint32_t kVmid = 8;
  constexpr uint64_t kAddr = KfdProcess::kPageSize - 8;
  constexpr std::array<uint8_t, 16> kData = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                                             0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};

  std::array<uint8_t, KfdProcess::kPageSize> first_page{};
  std::array<uint8_t, KfdProcess::kPageSize> second_page{};
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  page_table[0] = {first_page.data(), amdgpu::Mtype::RW};
  page_table[1] = {second_page.data(), amdgpu::Mtype::RW};
  mem.register_process(kVmid, &page_table, &page_table_mutex);

  mem.write_block(kAddr, std::span<const uint8_t>(kData), kVmid);
  std::array<uint8_t, kData.size()> result{};
  mem.read_block(kAddr, std::span<uint8_t>(result), kVmid);

  EXPECT_EQ(result, kData);
}

TEST(GpuMemoryTest, BlockAccessRechecksTranslationAfterSparseFallbackPage) {
  amdgpu::GpuMemory mem("test_mem");
  constexpr uint32_t kVmid = 9;
  constexpr uint64_t kAddr = KfdProcess::kPageSize - 8;
  constexpr std::array<uint8_t, 16> kReadData = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                                                 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27};
  constexpr std::array<uint8_t, 16> kWriteData = {0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
                                                  0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47};

  std::array<uint8_t, KfdProcess::kPageSize> second_page{};
  std::copy(kReadData.begin() + 8, kReadData.end(), second_page.begin());
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  page_table[1] = {second_page.data(), amdgpu::Mtype::RW};
  mem.register_process(kVmid, &page_table, &page_table_mutex);
  for (size_t i = 0; i < 8; ++i)
    mem.write8(kAddr + i, kReadData[i], kVmid);

  std::array<uint8_t, kReadData.size()> result{};
  mem.read_block(kAddr, std::span<uint8_t>(result), kVmid);
  EXPECT_EQ(result, kReadData);

  mem.write_block(kAddr, std::span<const uint8_t>(kWriteData), kVmid);
  for (size_t i = 0; i < 8; ++i)
    EXPECT_EQ(mem.read8(kAddr + i, kVmid), kWriteData[i]);
  EXPECT_TRUE(std::equal(kWriteData.begin() + 8, kWriteData.end(), second_page.begin()));
}

TEST(GpuMemoryTest, CopyBlockTransfersPageableClientMemoryAcrossPageBoundaries) {
  amdgpu::GpuMemory mem("test_mem");
  constexpr uint32_t kVmid = 10;
  constexpr uint64_t kGpuAddr = 0x100000 + KfdProcess::kPageSize - 11;
  constexpr size_t kSize = KfdProcess::kPageSize + 37;

  std::array<uint8_t, KfdProcess::kPageSize> first_page{};
  std::array<uint8_t, KfdProcess::kPageSize> second_page{};
  std::array<uint8_t, KfdProcess::kPageSize> third_page{};
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  const uint64_t first_page_number = kGpuAddr >> amdgpu::GpuMemory::PAGE_SHIFT;
  page_table[first_page_number] = {first_page.data(), amdgpu::Mtype::RW};
  page_table[first_page_number + 1] = {second_page.data(), amdgpu::Mtype::RW};
  page_table[first_page_number + 2] = {third_page.data(), amdgpu::Mtype::RW};
  mem.register_process(kVmid, &page_table, &page_table_mutex);
  mem.set_process_client_pid(kVmid, getpid());

  std::vector<uint8_t> host_source(kSize + 19);
  auto source = std::span<uint8_t>(host_source).subspan(7, kSize);
  for (size_t i = 0; i < source.size(); ++i)
    source[i] = static_cast<uint8_t>((i * 37 + 11) & 0xff);

  ASSERT_EQ(
      mem.copy_block(kGpuAddr, reinterpret_cast<uint64_t>(source.data()), source.size(), kVmid),
      amdgpu::CopyOutcome::Complete);
  std::vector<uint8_t> gpu_result(kSize);
  mem.read_block(kGpuAddr, std::span<uint8_t>(gpu_result), kVmid);
  EXPECT_TRUE(std::equal(source.begin(), source.end(), gpu_result.begin()));

  std::vector<uint8_t> host_destination(kSize + 23, 0);
  auto destination = std::span<uint8_t>(host_destination).subspan(13, kSize);
  ASSERT_EQ(mem.copy_block(reinterpret_cast<uint64_t>(destination.data()), kGpuAddr,
                           destination.size(), kVmid),
            amdgpu::CopyOutcome::Complete);
  EXPECT_TRUE(std::equal(source.begin(), source.end(), destination.begin()));
}

TEST(GpuMemoryTest, AuthorizedProcMemAccessesAnonymousTargetMemory) {
  amdgpu::GpuMemory mem("test_mem");
  constexpr uint32_t kVmid = 11;
  KfdProcess::PageTable page_table;
  std::shared_mutex page_table_mutex;
  mem.register_process(kVmid, &page_table, &page_table_mutex);

  const int target_mem_fd = ::open("/proc/self/mem", O_RDWR | O_CLOEXEC);
  ASSERT_GE(target_mem_fd, 0);
  mem.set_process_mem_fd(kVmid, target_mem_fd);
  ::close(target_mem_fd);

  std::array<uint8_t, 16> target = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                                    0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
  const uint64_t address = reinterpret_cast<uint64_t>(target.data());
  EXPECT_TRUE(mem.is_fetchable(address, kVmid));

  std::array<uint8_t, target.size()> readback{};
  mem.read_block(address, std::span<uint8_t>(readback), kVmid);
  EXPECT_EQ(readback, target);

  std::array<uint8_t, target.size()> replacement{};
  std::ranges::fill(replacement, 0xA5);
  mem.write_block(address, std::span<const uint8_t>(replacement), kVmid);
  EXPECT_EQ(target, replacement);
}

TEST(L1VectorCacheTest, UcDwordx4RoundTripPreservesVectorTransaction) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1VectorCache l1(&l2);
  l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0x7800;
  constexpr std::array<uint32_t, 4> kLane0 = {0x01234567u, 0x89ABCDEFu, 0x76543210u, 0xFEDCBA98u};
  constexpr std::array<uint32_t, 4> kLane1 = {0x11112222u, 0x33334444u, 0x55556666u, 0x77778888u};
  uint64_t addrs[64] = {};
  addrs[0] = kAddr;
  addrs[1] = kAddr + 0x100;
  std::array<uint8_t, 64 * sizeof(kLane0)> store_bytes{};
  std::memcpy(store_bytes.data(), kLane0.data(), sizeof(kLane0));
  std::memcpy(store_bytes.data() + sizeof(kLane0), kLane1.data(), sizeof(kLane1));

  l1.store(addrs, /*lane_mask=*/0x3, /*elem_size=*/4, /*num_elems=*/4, store_bytes.data(),
           amdgpu::Mtype::UC, /*non_temporal=*/false, cdna3::Isa::WF_SIZE);
  EXPECT_EQ(l1.store_l2_writes(), 2u);
  EXPECT_EQ(l2.backing_write_transactions(), 2u);

  std::array<uint8_t, 64 * sizeof(kLane0)> load_bytes{};
  l1.load(addrs, /*lane_mask=*/0x3, /*elem_size=*/4, /*num_elems=*/4, load_bytes.data(),
          amdgpu::Mtype::UC, /*non_temporal=*/false, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE);
  EXPECT_EQ(l2.backing_read_transactions(), 2u);
  EXPECT_EQ(std::memcmp(load_bytes.data(), kLane0.data(), sizeof(kLane0)), 0);
  EXPECT_EQ(std::memcmp(load_bytes.data() + sizeof(kLane0), kLane1.data(), sizeof(kLane1)), 0);
}

TEST(L1VectorCacheTest, ScratchDwordCrossesInterleaveBoundary) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1VectorCache l1(&l2);
  l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0x7A01;
  constexpr uint32_t kScratchStride = 64 * sizeof(uint32_t);
  constexpr uint32_t kInitialValue = 0x55443322;
  constexpr uint32_t kStoredValue = 0x87654321;
  mem.write8(kAddr, 0x22);
  mem.write8(kAddr + 1, 0x33);
  mem.write8(kAddr + 2, 0x44);
  mem.write8((kAddr & ~uint64_t{3}) + kScratchStride, 0x55);

  uint64_t addrs[64] = {};
  addrs[0] = kAddr;
  std::array<uint8_t, 64 * sizeof(uint32_t)> bytes{};
  l1.load(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, bytes.data(),
          amdgpu::Mtype::UC, /*non_temporal=*/false, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE, /*vmid=*/0, kScratchStride);
  uint32_t value = 0;
  std::memcpy(&value, bytes.data(), sizeof(value));
  EXPECT_EQ(value, kInitialValue);

  std::memcpy(bytes.data(), &kStoredValue, sizeof(kStoredValue));
  l1.store(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, bytes.data(),
           amdgpu::Mtype::UC, /*non_temporal=*/false, cdna3::Isa::WF_SIZE, /*vmid=*/0,
           kScratchStride);
  EXPECT_EQ(mem.read8(kAddr), 0x21);
  EXPECT_EQ(mem.read8(kAddr + 1), 0x43);
  EXPECT_EQ(mem.read8(kAddr + 2), 0x65);
  EXPECT_EQ(mem.read8((kAddr & ~uint64_t{3}) + kScratchStride), 0x87);
}

TEST(L1VectorCacheTest, PartialElementMasksSkipMaskedLoads) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1VectorCache l1(&l2);
  l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0x7880;
  constexpr std::array<uint32_t, 4> kLane0 = {0x101u, 0x102u, 0x103u, 0x104u};
  constexpr std::array<uint32_t, 4> kLane1 = {0x201u, 0x202u, 0x203u, 0x204u};
  for (uint32_t elem = 0; elem < 4; ++elem) {
    mem.write32(kAddr + elem * sizeof(uint32_t), kLane0[elem]);
    mem.write32(kAddr + 0x40 + elem * sizeof(uint32_t), kLane1[elem]);
  }

  uint64_t addrs[64] = {};
  addrs[0] = kAddr;
  addrs[1] = kAddr + 0x40;
  constexpr std::array<uint64_t, 4> kElementMasks = {0x3, 0x1, 0x1, 0x0};
  std::array<uint8_t, 64 * 4 * sizeof(uint32_t)> bytes{};
  l1.load(addrs, /*lane_mask=*/0x3, /*elem_size=*/4, /*num_elems=*/4, bytes.data(),
          amdgpu::Mtype::UC, /*non_temporal=*/false, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE, /*vmid=*/0, /*addr_stride=*/0, kElementMasks);

  std::array<uint32_t, 4> lane0{};
  std::array<uint32_t, 4> lane1{};
  std::memcpy(lane0.data(), bytes.data(), sizeof(lane0));
  std::memcpy(lane1.data(), bytes.data() + sizeof(lane0), sizeof(lane1));
  EXPECT_EQ(lane0, (std::array<uint32_t, 4>{0x101u, 0x102u, 0x103u, 0u}));
  EXPECT_EQ(lane1, (std::array<uint32_t, 4>{0x201u, 0u, 0u, 0u}));
}

TEST(L1VectorCacheTest, PartialElementMasksDropSkippedStores) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1VectorCache l1(&l2);
  l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0x78c0;
  constexpr std::array<uint32_t, 4> kInitial = {0xA0u, 0xA1u, 0xA2u, 0xA3u};
  constexpr std::array<uint32_t, 4> kStored = {0xB0u, 0xB1u, 0xB2u, 0xB3u};
  for (uint32_t elem = 0; elem < 4; ++elem)
    mem.write32(kAddr + elem * sizeof(uint32_t), kInitial[elem]);

  uint64_t addrs[64] = {};
  addrs[0] = kAddr;
  std::array<uint8_t, 64 * sizeof(kStored)> bytes{};
  std::memcpy(bytes.data(), kStored.data(), sizeof(kStored));
  constexpr std::array<uint64_t, 4> kElementMasks = {0x1, 0x1, 0x0, 0x0};
  l1.store(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/4, bytes.data(),
           amdgpu::Mtype::UC, /*non_temporal=*/false, cdna3::Isa::WF_SIZE, /*vmid=*/0,
           /*addr_stride=*/0, kElementMasks);

  EXPECT_EQ(mem.read32(kAddr), kStored[0]);
  EXPECT_EQ(mem.read32(kAddr + 4), kStored[1]);
  EXPECT_EQ(mem.read32(kAddr + 8), kInitial[2]);
  EXPECT_EQ(mem.read32(kAddr + 12), kInitial[3]);
}

TEST(L1VectorCacheTest, PartialElementMasksCoalesceFullyValidLanes) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1VectorCache l1(&l2);
  l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0x78e0;
  constexpr std::array<uint32_t, 4> kLane0 = {0x101u, 0x102u, 0x103u, 0x104u};
  constexpr std::array<uint32_t, 4> kLane1 = {0x201u, 0x202u, 0x203u, 0x204u};
  constexpr std::array<uint32_t, 4> kLane2 = {0x301u, 0x302u, 0x303u, 0x304u};
  uint64_t addrs[64] = {};
  addrs[0] = kAddr;
  addrs[1] = kAddr + sizeof(kLane0);
  addrs[2] = kAddr + 2 * sizeof(kLane0);
  std::array<uint8_t, 64 * sizeof(kLane0)> store_bytes{};
  std::memcpy(store_bytes.data(), kLane0.data(), sizeof(kLane0));
  std::memcpy(store_bytes.data() + sizeof(kLane0), kLane1.data(), sizeof(kLane1));
  std::memcpy(store_bytes.data() + 2 * sizeof(kLane0), kLane2.data(), sizeof(kLane2));
  constexpr std::array<uint64_t, 4> kElementMasks = {0x7, 0x7, 0x3, 0x3};

  l1.store(addrs, /*lane_mask=*/0x7, /*elem_size=*/4, /*num_elems=*/4, store_bytes.data(),
           amdgpu::Mtype::UC, /*non_temporal=*/false, cdna3::Isa::WF_SIZE, /*vmid=*/0,
           /*addr_stride=*/0, kElementMasks);
  EXPECT_EQ(l1.store_l2_writes(), 3u);
  EXPECT_EQ(l2.backing_write_transactions(), 3u);
  EXPECT_EQ(mem.read32(kAddr + 0), kLane0[0]);
  EXPECT_EQ(mem.read32(kAddr + 4), kLane0[1]);
  EXPECT_EQ(mem.read32(kAddr + 8), kLane0[2]);
  EXPECT_EQ(mem.read32(kAddr + 12), kLane0[3]);
  EXPECT_EQ(mem.read32(kAddr + 16), kLane1[0]);
  EXPECT_EQ(mem.read32(kAddr + 20), kLane1[1]);
  EXPECT_EQ(mem.read32(kAddr + 24), kLane1[2]);
  EXPECT_EQ(mem.read32(kAddr + 28), kLane1[3]);
  EXPECT_EQ(mem.read32(kAddr + 32), kLane2[0]);
  EXPECT_EQ(mem.read32(kAddr + 36), kLane2[1]);
  EXPECT_EQ(mem.read32(kAddr + 40), 0u);
  EXPECT_EQ(mem.read32(kAddr + 44), 0u);

  std::array<uint8_t, 64 * sizeof(kLane0)> load_bytes{};
  l1.load(addrs, /*lane_mask=*/0x7, /*elem_size=*/4, /*num_elems=*/4, load_bytes.data(),
          amdgpu::Mtype::UC, /*non_temporal=*/false, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE, /*vmid=*/0, /*addr_stride=*/0, kElementMasks);
  EXPECT_EQ(l2.backing_read_transactions(), 3u);
  EXPECT_EQ(std::memcmp(load_bytes.data(), kLane0.data(), sizeof(kLane0)), 0);
  EXPECT_EQ(std::memcmp(load_bytes.data() + sizeof(kLane0), kLane1.data(), sizeof(kLane1)), 0);
  std::array<uint32_t, 4> loaded_lane2{};
  std::memcpy(loaded_lane2.data(), load_bytes.data() + 2 * sizeof(kLane0), sizeof(loaded_lane2));
  EXPECT_EQ(loaded_lane2, (std::array<uint32_t, 4>{kLane2[0], kLane2[1], 0u, 0u}));
}

TEST(L1VectorCacheTest, UcDwordx4CoalescesAdjacentLanes) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1VectorCache l1(&l2);
  l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0x7900;
  constexpr std::array<uint32_t, 4> kLane0 = {0x01234567u, 0x89ABCDEFu, 0x76543210u, 0xFEDCBA98u};
  constexpr std::array<uint32_t, 4> kLane1 = {0x11112222u, 0x33334444u, 0x55556666u, 0x77778888u};
  uint64_t addrs[64] = {};
  addrs[0] = kAddr;
  addrs[1] = kAddr + sizeof(kLane0);
  std::array<uint8_t, 64 * sizeof(kLane0)> store_bytes{};
  std::memcpy(store_bytes.data(), kLane0.data(), sizeof(kLane0));
  std::memcpy(store_bytes.data() + sizeof(kLane0), kLane1.data(), sizeof(kLane1));

  l1.store(addrs, /*lane_mask=*/0x3, /*elem_size=*/4, /*num_elems=*/4, store_bytes.data(),
           amdgpu::Mtype::UC, /*non_temporal=*/false, cdna3::Isa::WF_SIZE);
  EXPECT_EQ(l1.store_l2_writes(), 1u);
  EXPECT_EQ(l2.backing_write_transactions(), 1u);

  std::array<uint8_t, 64 * sizeof(kLane0)> load_bytes{};
  l1.load(addrs, /*lane_mask=*/0x3, /*elem_size=*/4, /*num_elems=*/4, load_bytes.data(),
          amdgpu::Mtype::UC, /*non_temporal=*/false, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE);
  EXPECT_EQ(l2.backing_read_transactions(), 1u);
  EXPECT_EQ(std::memcmp(load_bytes.data(), kLane0.data(), sizeof(kLane0)), 0);
  EXPECT_EQ(std::memcmp(load_bytes.data() + sizeof(kLane0), kLane1.data(), sizeof(kLane1)), 0);
}

TEST(L1VectorCacheTest, UcWriteInvalidatesResidentLine) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1VectorCache l1(&l2);
  l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0x8000;
  constexpr uint32_t kOldValue = 0x11111111;
  constexpr uint32_t kNewValue = 0x22222222;

  uint64_t addrs[64] = {};
  addrs[0] = kAddr;
  std::array<uint8_t, 64 * sizeof(uint32_t)> bytes{};

  mem.write32(kAddr, kOldValue);
  l1.load(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, bytes.data(),
          amdgpu::Mtype::RW, /*non_temporal=*/false, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE);
  uint32_t value = 0;
  std::memcpy(&value, bytes.data(), sizeof(value));
  ASSERT_EQ(value, kOldValue);

  std::array<uint8_t, 64 * sizeof(uint32_t)> store_bytes{};
  std::memcpy(store_bytes.data(), &kNewValue, sizeof(kNewValue));
  l1.store(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, store_bytes.data(),
           amdgpu::Mtype::UC, /*non_temporal=*/false, cdna3::Isa::WF_SIZE);

  bytes.fill(0);
  l1.load(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, bytes.data(),
          amdgpu::Mtype::RW, /*non_temporal=*/false, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE);
  std::memcpy(&value, bytes.data(), sizeof(value));
  EXPECT_EQ(value, kNewValue);
}

TEST(L1VectorCacheTest, NonTemporalReadInvalidatesResidentLine) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1VectorCache l1(&l2);
  l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0x8100;
  constexpr uint32_t kOldValue = 0x11111111;
  constexpr uint32_t kNewValue = 0x22222222;

  uint64_t addrs[64] = {};
  addrs[0] = kAddr;
  std::array<uint8_t, 64 * sizeof(uint32_t)> bytes{};

  mem.write32(kAddr, kOldValue);
  l1.load(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, bytes.data(),
          amdgpu::Mtype::RW, /*non_temporal=*/false, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE);
  uint32_t value = 0;
  std::memcpy(&value, bytes.data(), sizeof(value));
  ASSERT_EQ(value, kOldValue);

  l2.write(kAddr, reinterpret_cast<const uint8_t *>(&kNewValue), sizeof(kNewValue),
           amdgpu::Mtype::RW);

  bytes.fill(0);
  l1.load(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, bytes.data(),
          amdgpu::Mtype::RW, /*non_temporal=*/true, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE);
  std::memcpy(&value, bytes.data(), sizeof(value));
  ASSERT_EQ(value, kNewValue);

  bytes.fill(0);
  l1.load(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, bytes.data(),
          amdgpu::Mtype::RW, /*non_temporal=*/false, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE);
  std::memcpy(&value, bytes.data(), sizeof(value));
  EXPECT_EQ(value, kNewValue);
}

TEST(L1VectorCacheTest, L1BypassReadInvalidatesResidentLine) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1VectorCache l1(&l2);
  l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0x8200;
  constexpr uint32_t kOldValue = 0x11111111;
  constexpr uint32_t kNewValue = 0x22222222;

  uint64_t addrs[64] = {};
  addrs[0] = kAddr;
  std::array<uint8_t, 64 * sizeof(uint32_t)> bytes{};

  mem.write32(kAddr, kOldValue);
  l1.load(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, bytes.data(),
          amdgpu::Mtype::RW, /*non_temporal=*/false, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE);
  uint32_t value = 0;
  std::memcpy(&value, bytes.data(), sizeof(value));
  ASSERT_EQ(value, kOldValue);

  l2.write(kAddr, reinterpret_cast<const uint8_t *>(&kNewValue), sizeof(kNewValue),
           amdgpu::Mtype::RW);

  bytes.fill(0);
  l1.load(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, bytes.data(),
          amdgpu::Mtype::RW, /*non_temporal=*/false, /*request_l1_bypass=*/true,
          cdna3::Isa::WF_SIZE);
  std::memcpy(&value, bytes.data(), sizeof(value));
  ASSERT_EQ(value, kNewValue);

  bytes.fill(0);
  l1.load(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, bytes.data(),
          amdgpu::Mtype::RW, /*non_temporal=*/false, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE);
  std::memcpy(&value, bytes.data(), sizeof(value));
  EXPECT_EQ(value, kNewValue);
}

TEST(L1VectorCacheTest, NonTemporalWriteInvalidatesResidentLine) {
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::L1VectorCache l1(&l2);
  l2.set_backing_memory(&mem);

  constexpr uint64_t kAddr = 0x8300;
  constexpr uint32_t kOldValue = 0x11111111;
  constexpr uint32_t kNewValue = 0x22222222;

  uint64_t addrs[64] = {};
  addrs[0] = kAddr;
  std::array<uint8_t, 64 * sizeof(uint32_t)> bytes{};

  mem.write32(kAddr, kOldValue);
  l1.load(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, bytes.data(),
          amdgpu::Mtype::RW, /*non_temporal=*/false, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE);
  uint32_t value = 0;
  std::memcpy(&value, bytes.data(), sizeof(value));
  ASSERT_EQ(value, kOldValue);

  std::array<uint8_t, 64 * sizeof(uint32_t)> store_bytes{};
  std::memcpy(store_bytes.data(), &kNewValue, sizeof(kNewValue));
  l1.store(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, store_bytes.data(),
           amdgpu::Mtype::RW, /*non_temporal=*/true, cdna3::Isa::WF_SIZE);

  bytes.fill(0);
  l1.load(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, bytes.data(),
          amdgpu::Mtype::RW, /*non_temporal=*/false, /*request_l1_bypass=*/false,
          cdna3::Isa::WF_SIZE);
  std::memcpy(&value, bytes.data(), sizeof(value));
  EXPECT_EQ(value, kNewValue);
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
    wf->set_status_raw(0xFFFFFFFFu);

    wf->halt();
    wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
    ASSERT_NE(wf, nullptr);
    EXPECT_EQ(wf->status_raw(), 0u);

    EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET, 0), 0u);
    EXPECT_EQ(
        cu->read_vgpr(wf->vgpr_alloc().base + amdgpu::ACC_VGPR_OFFSET + kCdnaAccVgprsPerWf - 1, 0),
        0u);
  }
}

INSTANTIATE_TEST_SUITE_P(AllIsas, CuFactoryTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                           ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4,
                                           ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA1,
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

TEST(DppPermuteTest, Dpp8RejectsSourcesWiderThanOneDword) {
  amdgpu::GpuMemory mem("dpp8_width_guard_mem");
  amdgpu::L2Cache l2("dpp8_width_guard_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("dpp8_width_guard_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);

  Operand wide_src(64, 0);
  std::optional<StagedOperand> storage;
  EXPECT_THROW(amdgpu::dpp::apply_dpp8(wide_src, 0, 0, storage, *wf), util::InvalidInst);
}

TEST(DppPermuteTest, ScopedOperandDelegateRestoresPreviousDelegate) {
  Operand source(32, 1);
  uint32_t first_data[1] = {0x11111111u};
  uint32_t second_data[1] = {0x22222222u};
  DppOperand first(source, first_data, 1);
  DppOperand second(source, second_data, 1);

  {
    ScopedOperandDelegate first_binding(source, &first);
    EXPECT_EQ(source.delegate(), &first);
    {
      ScopedOperandDelegate second_binding(source, &second);
      EXPECT_EQ(source.delegate(), &second);
    }
    EXPECT_EQ(source.delegate(), &first);
  }
  EXPECT_EQ(source.delegate(), nullptr);
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

TEST(DppPermuteTest, True16SourceByteMaskFollowsOpSel) {
  using namespace amdgpu::dpp;

  EXPECT_EQ(true16_source_byte_mask(/*opsel=*/0b0000, /*source_index=*/0),
            ExecutionPlugin::kLowHalfByteMask);
  EXPECT_EQ(true16_source_byte_mask(/*opsel=*/0b0001, /*source_index=*/0),
            ExecutionPlugin::kHighHalfByteMask);
  EXPECT_EQ(true16_source_byte_mask(/*opsel=*/0b0000, /*source_index=*/1),
            ExecutionPlugin::kLowHalfByteMask);
  EXPECT_EQ(true16_source_byte_mask(/*opsel=*/0b0010, /*source_index=*/1),
            ExecutionPlugin::kHighHalfByteMask);
}

TEST(DppPermuteTest, AccessPlanTracksSelectedSources) {
  using namespace amdgpu::dpp;
  constexpr uint64_t kAllLanesActive = ~0ULL;
  const auto plan = make_dpp_plan(64, ROW_SHR1, 0xE, 0xD, /*bound_ctrl=*/1, /*fi=*/1,
                                  kAllLanesActive, /*inactive_uses_bound_ctrl=*/false);

  EXPECT_EQ(plan.source_lanes[1], 0);
  EXPECT_EQ(plan.source_lanes[5], 4);
  EXPECT_EQ(plan.source_lanes[17], 16);
  EXPECT_NE(plan.zero_source_mask & 1u, 0u);
  EXPECT_NE(plan.physical_read_dest_mask & (uint64_t{1} << 5), 0u);
  EXPECT_EQ(plan.row_bank_mask & (uint64_t{1} << 5), 0u);
  EXPECT_EQ(dpp_physical_source_mask(plan, kAllLanesActive, 64) & (uint64_t{1} << 4),
            uint64_t{1} << 4);
}

TEST(DppPermuteTest, FetchInactiveControlsInactiveSourceReads) {
  using namespace amdgpu::dpp;
  constexpr uint64_t kLane0Inactive = ~1ULL;

  const auto dpp_fi0 = make_dpp_plan(64, ROW_SHR1, 0xF, 0xF, /*bound_ctrl=*/1, /*fi=*/0,
                                     kLane0Inactive, /*inactive_uses_bound_ctrl=*/false);
  EXPECT_NE(dpp_fi0.zero_source_mask & (uint64_t{1} << 1), 0u);
  EXPECT_EQ(dpp_fi0.physical_read_dest_mask & (uint64_t{1} << 1), 0u);

  const auto dpp_fi1 = make_dpp_plan(64, ROW_SHR1, 0xF, 0xF, /*bound_ctrl=*/1, /*fi=*/1,
                                     kLane0Inactive, /*inactive_uses_bound_ctrl=*/false);
  EXPECT_EQ(dpp_fi1.source_lanes[1], 0);
  EXPECT_NE(dpp_fi1.physical_read_dest_mask & (uint64_t{1} << 1), 0u);

  constexpr uint32_t kAllLanesSelectLane0 = 0;
  const auto dpp8_fi0 = make_dpp8_plan(32, kAllLanesSelectLane0, /*fi=*/0, kLane0Inactive);
  EXPECT_NE(dpp8_fi0.zero_source_mask & (uint64_t{1} << 1), 0u);

  const auto dpp8_fi1 = make_dpp8_plan(32, kAllLanesSelectLane0, /*fi=*/1, kLane0Inactive);
  EXPECT_EQ(dpp8_fi1.source_lanes[1], 0);
  EXPECT_NE(dpp8_fi1.physical_read_dest_mask & (uint64_t{1} << 1), 0u);

  EXPECT_EQ(src_dpp8_fi(amdgpu::SRC_DPP8_FI_0), 0u);
  EXPECT_EQ(src_dpp8_fi(amdgpu::SRC_DPP8_FI_1), 1u);
}

TEST(DppPermuteTest, RejectsReservedDpp16Controls) {
  EXPECT_TRUE(amdgpu::dpp::dpp_ctrl_is_valid(0x00u, false, false, false));
  EXPECT_TRUE(amdgpu::dpp::dpp_ctrl_is_valid(amdgpu::dpp::ROW_SHR1, false, false, false));
  EXPECT_TRUE(amdgpu::dpp::dpp_ctrl_is_valid(amdgpu::dpp::ROW_SELECT_MAX, false, false, false));
  EXPECT_FALSE(amdgpu::dpp::dpp_ctrl_is_valid(0x100u, true, true, true));
  EXPECT_FALSE(amdgpu::dpp::dpp_ctrl_is_valid(0x110u, true, true, true));
  EXPECT_FALSE(amdgpu::dpp::dpp_ctrl_is_valid(0x131u, true, true, true));
  EXPECT_FALSE(amdgpu::dpp::dpp_ctrl_is_valid(0x144u, true, true, true));
  EXPECT_FALSE(amdgpu::dpp::dpp_ctrl_is_valid(amdgpu::dpp::WF_ROR1, false, true, false));
  EXPECT_TRUE(amdgpu::dpp::dpp_ctrl_is_valid(amdgpu::dpp::WF_ROR1, true, false, false));
  EXPECT_FALSE(amdgpu::dpp::dpp_ctrl_is_valid(amdgpu::dpp::ROW_BCAST15, true, false, false));
  EXPECT_TRUE(amdgpu::dpp::dpp_ctrl_is_valid(amdgpu::dpp::ROW_BCAST15, false, true, false));
  EXPECT_FALSE(amdgpu::dpp::dpp_ctrl_is_valid(amdgpu::dpp::ROW_XMASK_BASE, false, false, false));
  EXPECT_TRUE(amdgpu::dpp::dpp_ctrl_is_valid(amdgpu::dpp::ROW_XMASK_BASE, false, false, true));
}

TEST(DppPermuteTest, WriteMaskHonorsBoundCtrlAndBroadcastValidity) {
  using namespace amdgpu::dpp;

  // The public DPP16 field descriptions make out-of-range selection a
  // BOUND_CTRL decision for either FI value. FI only changes whether an
  // in-range inactive source lane may be fetched.
  for (uint32_t fi : {0u, 1u}) {
    uint64_t oob_mask = dpp_source_write_mask(64, ROW_SHR1, 0, fi, ~0ULL, true);
    EXPECT_EQ(oob_mask & 1u, 0u) << "FI=" << fi;
    oob_mask = dpp_source_write_mask(64, ROW_SHR1, 1, fi, ~0ULL, true);
    EXPECT_NE(oob_mask & 1u, 0u) << "FI=" << fi;
  }

  uint64_t mask = dpp_source_write_mask(64, ROW_SHR1, 0, 1, ~0ULL, false);
  EXPECT_EQ(mask & 0x3, 0x2u);
  mask = dpp_source_write_mask(64, ROW_SHR1, 1, 1, ~0ULL, false);
  EXPECT_EQ(mask, ~0ULL);

  mask = dpp_source_write_mask(64, ROW_BCAST15, 0, 1, ~0ULL, false);
  EXPECT_EQ(mask, 0xFFFFFFFFFFFF0000ULL);
  mask = dpp_source_write_mask(64, ROW_BCAST31, 0, 1, ~0ULL, false);
  EXPECT_EQ(mask, 0xFFFFFFFF00000000ULL);
  mask = dpp_source_write_mask(32, ROW_BCAST15, 0, 1, ~0ULL, false);
  EXPECT_EQ(mask, 0xFFFF0000ULL);

  EXPECT_EQ(dpp_row_bank_mask(64, 0xD, 0xF), 0xFFFFFFFF0000FFFFULL);
  EXPECT_EQ(dpp_row_bank_mask(64, 0xF, 0xD), 0xFF0FFF0FFF0FFF0FULL);
}

TEST(DppPermuteTest, SourceWriteMaskAppliesBoundCtrlToInactiveSourcesWhenRequested) {
  using namespace amdgpu::dpp;

  constexpr uint64_t kLane0Inactive = ~1ULL;
  uint64_t legacy = dpp_source_write_mask(32, ROW_SHR1, 0, 0, kLane0Inactive, false);
  uint64_t modern = dpp_source_write_mask(32, ROW_SHR1, 0, 0, kLane0Inactive, true);
  EXPECT_NE(legacy & (1ULL << 1), 0u);
  EXPECT_EQ(modern & (1ULL << 1), 0u);

  modern = dpp_source_write_mask(32, ROW_SHR1, 1, 0, kLane0Inactive, true);
  EXPECT_EQ(modern, 0xFFFFFFFFULL);
}

TEST(DppPermuteTest, CompareResultZerosMaskedInactiveAndInvalidSourceLanes) {
  using namespace amdgpu::dpp;

  constexpr uint64_t kOldExec = 0xFFFFFDFFULL;
  constexpr uint64_t kRowBank = 0x0000FF0FULL;
  constexpr uint64_t kSourceWrites = ~uint64_t{0x9};
  constexpr uint64_t kNewResult = (1ULL << 1) | (1ULL << 9);
  // Lane 1 is newly true. Lane 9 is also supplied as true, but is inactive and
  // must be cleared by the helper itself. Lanes 0 and 3 are also zero because
  // their source writes are suppressed; compare results never preserve them.
  EXPECT_EQ(dpp_compare_result(kNewResult, kOldExec, kRowBank, kSourceWrites), 0x2u);
}

struct Cdna1DppTraits {
  static constexpr const char *name = "cdna1";
  static constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA1;
  static constexpr uint16_t vopc_opcode = cdna1::kVCmpEqU32Vopc;
  static const IsaExecutionBackend &backend() { return cdna1::execution_backend(); }
  using MachineInst = cdna1::MachineInst;
  using Vop1VopDppMachineInst = cdna1::Vop1VopDppMachineInst;
  using Vop1DppTestMachineInst = cdna1::Vop1VopDppMachineInst;
  using VMovB32Vop1 = cdna1::VMovB32Vop1;
  using VCmpEqU32Vopc = cdna1::VCmpEqU32Vopc;
  using VCmpxEqU32Vopc = cdna1::VCmpxEqU32Vopc;
};

struct Cdna2DppTraits {
  static constexpr const char *name = "cdna2";
  static constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA2;
  static constexpr uint16_t vopc_opcode = cdna2::kVCmpEqU32Vopc;
  static const IsaExecutionBackend &backend() { return cdna2::execution_backend(); }
  using MachineInst = cdna2::MachineInst;
  using Vop1VopDppMachineInst = cdna2::Vop1VopDppMachineInst;
  using Vop1DppTestMachineInst = cdna2::Vop1VopDppMachineInst;
  using VMovB32Vop1 = cdna2::VMovB32Vop1;
  using VCvtF64I32Vop1 = cdna2::VCvtF64I32Vop1;
  using VCmpEqU32Vopc = cdna2::VCmpEqU32Vopc;
  using VCmpxEqU32Vopc = cdna2::VCmpxEqU32Vopc;
};

struct Cdna3DppTraits {
  static constexpr const char *name = "cdna3";
  static constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA3;
  static constexpr uint16_t vopc_opcode = cdna3::kVCmpEqU32Vopc;
  static const IsaExecutionBackend &backend() { return cdna3::execution_backend(); }
  using MachineInst = cdna3::MachineInst;
  using Vop1VopDppMachineInst = cdna3::Vop1VopDppMachineInst;
  using Vop1DppTestMachineInst = cdna3::Vop1VopDppMachineInst;
  using VMovB32Vop1 = cdna3::VMovB32Vop1;
  using VCvtF64I32Vop1 = cdna3::VCvtF64I32Vop1;
  using VCmpEqU32Vopc = cdna3::VCmpEqU32Vopc;
  using VCmpxEqU32Vopc = cdna3::VCmpxEqU32Vopc;
};

struct Cdna4DppTraits {
  static constexpr const char *name = "cdna4";
  static constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA4;
  static constexpr uint16_t vopc_opcode = cdna4::kVCmpEqU32Vopc;
  static const IsaExecutionBackend &backend() { return cdna4::execution_backend(); }
  static constexpr uint32_t wf_size = 64;
  using MachineInst = cdna4::MachineInst;
  using Vop1VopDppMachineInst = cdna4::Vop1VopDppMachineInst;
  using Vop2DppCarryMachineInst = cdna4::Vop2VopDppMachineInst;
  using Vop1DppTestMachineInst = cdna4::Vop1VopDppMachineInst;
  using Vop1Dpp64MachineInst = cdna4::Vop1VopDppMachineInst;
  using VMovB32Vop1 = cdna4::VMovB32Vop1;
  using VMovB64Vop1 = cdna4::VMovB64Vop1;
  using Vop2DppCarryInst = cdna4::VAddCoU32Vop2;
  using VCmpEqU32Vopc = cdna4::VCmpEqU32Vopc;
  using VCmpxEqU32Vopc = cdna4::VCmpxEqU32Vopc;
  static constexpr bool dpp_carry_has_fi = false;
  static constexpr bool dpp_carry_has_carry_in = false;
};

struct Rdna1DppTraits {
  static constexpr const char *name = "rdna1";
  static constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA1;
  static constexpr uint16_t vopc_opcode = rdna1::kVCmpEqU32Vopc;
  static const IsaExecutionBackend &backend() { return rdna1::execution_backend(); }
  static constexpr bool inactive_uses_bound_ctrl = false;
  using MachineInst = rdna1::MachineInst;
  using VopcMachineInst = rdna1::VopcMachineInst;
  using VopcVopSdwaSdstEncMachineInst = rdna1::VopcVopSdwaSdstEncMachineInst;
  using Vop1VopDpp16MachineInst = rdna1::Vop1VopDpp16MachineInst;
  using Vop1DppTestMachineInst = rdna1::Vop1VopDpp16MachineInst;
  using Vop1VopDpp8MachineInst = rdna1::Vop1VopDpp8MachineInst;
  using VMovB32Vop1 = rdna1::VMovB32Vop1;
  using VCmpEqU32Vopc = rdna1::VCmpEqU32Vopc;
  using VCmpxEqU32Vopc = rdna1::VCmpxEqU32Vopc;
  static constexpr uint32_t vcmp_eq_u32_op = rdna1::kVCmpEqU32Vopc;
  static constexpr uint32_t vopc_encoding = rdna1::encoding::kVopc >> 2;
};

struct Rdna2DppTraits {
  static constexpr const char *name = "rdna2";
  static constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA2;
  static constexpr uint16_t vopc_opcode = rdna2::kVCmpEqU32Vopc;
  static const IsaExecutionBackend &backend() { return rdna2::execution_backend(); }
  static constexpr bool inactive_uses_bound_ctrl = false;
  using MachineInst = rdna2::MachineInst;
  using VopcMachineInst = rdna2::VopcMachineInst;
  using VopcVopSdwaSdstEncMachineInst = rdna2::VopcVopSdwaSdstEncMachineInst;
  using Vop1VopDpp16MachineInst = rdna2::Vop1VopDpp16MachineInst;
  using Vop1DppTestMachineInst = rdna2::Vop1VopDpp16MachineInst;
  using Vop1VopDpp8MachineInst = rdna2::Vop1VopDpp8MachineInst;
  using VMovB32Vop1 = rdna2::VMovB32Vop1;
  using VCmpEqU32Vopc = rdna2::VCmpEqU32Vopc;
  using VCmpxEqU32Vopc = rdna2::VCmpxEqU32Vopc;
  static constexpr uint32_t vcmp_eq_u32_op = rdna2::kVCmpEqU32Vopc;
  static constexpr uint32_t vopc_encoding = rdna2::encoding::kVopc >> 2;
};

struct Rdna4DppTraits {
  static constexpr const char *name = "rdna4";
  static constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4;
  static const IsaExecutionBackend &backend() { return rdna4::execution_backend(); }
  static constexpr uint32_t wf_size = 32;
  static constexpr bool inactive_uses_bound_ctrl = true;
  using MachineInst = rdna4::MachineInst;
  using VopcMachineInst = rdna4::VopcMachineInst;
  using Vop1VopDpp16MachineInst = rdna4::Vop1VopDpp16MachineInst;
  using Vop2DppCarryMachineInst = rdna4::Vop2VopDpp16MachineInst;
  using Vop1DppTestMachineInst = rdna4::Vop1VopDpp16MachineInst;
  using Vop1VopDpp8MachineInst = rdna4::Vop1VopDpp8MachineInst;
  using VopcVopDpp16MachineInst = rdna4::VopcVopDpp16MachineInst;
  using Vop3VopDpp16MachineInst = rdna4::Vop3VopDpp16MachineInst;
  using Vop3pVopDpp16MachineInst = rdna4::Vop3pVopDpp16MachineInst;
  using Vop3SdstEncVopDpp16MachineInst = rdna4::Vop3SdstEncVopDpp16MachineInst;
  using VMovB32Vop1 = rdna4::VMovB32Vop1;
  using Vop2DppCarryInst = rdna4::VAddCoCiU32Vop2;
  using VCmpEqU32Vopc = rdna4::VCmpEqU32Vopc;
  using VCmpxEqU32Vopc = rdna4::VCmpxEqU32Vopc;
  using VCmpEqU32Vop3 = rdna4::VCmpEqU32Vop3;
  using VCmpxEqU32Vop3 = rdna4::VCmpxEqU32Vop3;
  using VFmaMixF32Vop3p = rdna4::VFmaMixF32Vop3p;
  using VAddCoCiU32Vop3SdstEnc = rdna4::VAddCoCiU32Vop3SdstEnc;
  static constexpr bool dpp_carry_has_fi = true;
  static constexpr bool dpp_carry_has_carry_in = true;
  static void set_aligned_vop3p_opsel(Vop3pVopDpp16MachineInst &raw) {
    raw.opsel = 0;
    raw.opsel_hi = 0x3;
    raw.opsel_hi_2 = 1;
  }
};

struct Rdna3DppTraits {
  static constexpr const char *name = "rdna3";
  static constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA3;
  static const IsaExecutionBackend &backend() { return rdna3::execution_backend(); }
  static constexpr bool inactive_uses_bound_ctrl = true;
  using MachineInst = rdna3::MachineInst;
  using VopcMachineInst = rdna3::VopcMachineInst;
  using Vop1VopDpp16MachineInst = rdna3::Vop1VopDpp16MachineInst;
  using Vop1DppTestMachineInst = rdna3::Vop1VopDpp16MachineInst;
  using Vop1VopDpp8MachineInst = rdna3::Vop1VopDpp8MachineInst;
  using VopcVopDpp16MachineInst = rdna3::VopcVopDpp16MachineInst;
  using Vop3VopDpp16MachineInst = rdna3::Vop3VopDpp16MachineInst;
  using Vop3pVopDpp16MachineInst = rdna3::Vop3pVopDpp16MachineInst;
  using Vop3SdstEncVopDpp16MachineInst = rdna3::Vop3SdstEncVopDpp16MachineInst;
  using VMovB32Vop1 = rdna3::VMovB32Vop1;
  using VCmpEqU32Vopc = rdna3::VCmpEqU32Vopc;
  using VCmpxEqU32Vopc = rdna3::VCmpxEqU32Vopc;
  using VCmpEqU32Vop3 = rdna3::VCmpEqU32Vop3;
  using VCmpxEqU32Vop3 = rdna3::VCmpxEqU32Vop3;
  using VFmaMixF32Vop3p = rdna3::VFmaMixF32Vop3p;
  using VAddCoCiU32Vop3SdstEnc = rdna3::VAddCoCiU32Vop3SdstEnc;
  static void set_aligned_vop3p_opsel(Vop3pVopDpp16MachineInst &raw) {
    raw.op_sel = 0;
    raw.op_sel_hi = 0x3;
    raw.op_sel_hi_2 = 1;
  }
};

struct Rdna3_5DppTraits {
  static constexpr const char *name = "rdna3_5";
  static constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA3_5;
  static const IsaExecutionBackend &backend() { return rdna3_5::execution_backend(); }
  static constexpr bool inactive_uses_bound_ctrl = true;
  using MachineInst = rdna3_5::MachineInst;
  using VopcMachineInst = rdna3_5::VopcMachineInst;
  using Vop1VopDpp16MachineInst = rdna3_5::Vop1VopDpp16MachineInst;
  using Vop1DppTestMachineInst = rdna3_5::Vop1VopDpp16MachineInst;
  using Vop1VopDpp8MachineInst = rdna3_5::Vop1VopDpp8MachineInst;
  using VopcVopDpp16MachineInst = rdna3_5::VopcVopDpp16MachineInst;
  using Vop3VopDpp16MachineInst = rdna3_5::Vop3VopDpp16MachineInst;
  using Vop3pVopDpp16MachineInst = rdna3_5::Vop3pVopDpp16MachineInst;
  using Vop3SdstEncVopDpp16MachineInst = rdna3_5::Vop3SdstEncVopDpp16MachineInst;
  using VMovB32Vop1 = rdna3_5::VMovB32Vop1;
  using VCmpEqU32Vopc = rdna3_5::VCmpEqU32Vopc;
  using VCmpxEqU32Vopc = rdna3_5::VCmpxEqU32Vopc;
  using VCmpEqU32Vop3 = rdna3_5::VCmpEqU32Vop3;
  using VCmpxEqU32Vop3 = rdna3_5::VCmpxEqU32Vop3;
  using VFmaMixF32Vop3p = rdna3_5::VFmaMixF32Vop3p;
  using VAddCoCiU32Vop3SdstEnc = rdna3_5::VAddCoCiU32Vop3SdstEnc;
  static void set_aligned_vop3p_opsel(Vop3pVopDpp16MachineInst &raw) {
    raw.op_sel = 0;
    raw.op_sel_hi = 0x3;
    raw.op_sel_hi_2 = 1;
  }
};

struct Gfx1250DppTraits {
  static constexpr const char *name = "gfx1250";
  static constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA5;
  static const IsaExecutionBackend &backend() { return cdna5::execution_backend(); }
  static constexpr uint32_t wf_size = 32;
  static constexpr bool inactive_uses_bound_ctrl = true;
  using MachineInst = cdna5::MachineInst;
  using VopcMachineInst = cdna5::VopcMachineInst;
  using Vop1VopDpp16MachineInst = cdna5::Vop1VopDpp16MachineInst;
  using Vop2DppCarryMachineInst = cdna5::Vop2VopDpp16MachineInst;
  using Vop1DppTestMachineInst = cdna5::Vop1VopDpp16MachineInst;
  using Vop1VopDpp8MachineInst = cdna5::Vop1VopDpp8MachineInst;
  using Vop1Dpp64MachineInst = cdna5::Vop1VopDpp16MachineInst;
  using VopcVopDpp16MachineInst = cdna5::VopcVopDpp16MachineInst;
  using Vop3VopDpp16MachineInst = cdna5::Vop3VopDpp16MachineInst;
  using Vop3pVopDpp16MachineInst = cdna5::Vop3pVopDpp16MachineInst;
  using Vop3SdstEncVopDpp16MachineInst = cdna5::Vop3SdstEncVopDpp16MachineInst;
  using VMovB32Vop1 = cdna5::VMovB32Vop1;
  using VMovB64Vop1 = cdna5::VMovB64Vop1;
  using Vop2DppCarryInst = cdna5::VAddCoCiU32Vop2;
  using VCmpEqU32Vopc = cdna5::VCmpEqU32Vopc;
  using VCmpxEqU32Vopc = cdna5::VCmpxEqU32Vopc;
  using VCmpEqU32Vop3 = cdna5::VCmpEqU32Vop3;
  using VCmpxEqU32Vop3 = cdna5::VCmpxEqU32Vop3;
  using VFmaMixF32Vop3p = cdna5::VFmaMixF32Vop3p;
  using VAddCoCiU32Vop3SdstEnc = cdna5::VAddCoCiU32Vop3SdstEnc;
  static constexpr bool dpp_carry_has_fi = true;
  static constexpr bool dpp_carry_has_carry_in = true;
  static void set_aligned_vop3p_opsel(Vop3pVopDpp16MachineInst &raw) {
    raw.opsel = 0;
    raw.opsel_hi = 0x3;
    raw.opsel_hi_2 = 1;
  }
};

template <typename Traits> void generated_dpp_instruction_reuse_reads_current_source() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp_reuse_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp_reuse_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;
  auto cu =
      amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_dpp_reuse_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(wf->wf_size() == 64 ? ~0ULL : 0xFFFFFFFFULL);

  constexpr uint32_t kSrc = 4;
  constexpr uint32_t kDst = 8;
  uint32_t vbase = wf->vgpr_alloc().base;

  typename Traits::Vop1DppTestMachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = 0xB1; // quad_perm:[1,0,3,2]
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;

  for (bool force_scalar : {false, true}) {
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");
    ForceScalarGuard force_scalar_guard(force_scalar);
    typename Traits::VMovB32Vop1 inst(reinterpret_cast<const typename Traits::MachineInst *>(&raw));
    const Operand *architectural_src0 = inst.src_operand(0);
    ASSERT_NE(architectural_src0, nullptr);

    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      cu->write_vgpr(vbase + kSrc, lane, 0x10000000u + lane);
    inst.execute_impl(*wf);
    EXPECT_EQ(inst.src_operand(0), architectural_src0);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 0), 0x10000001u);

    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      cu->write_vgpr(vbase + kSrc, lane, 0x20000000u + lane);
    inst.execute_impl(*wf);
    EXPECT_EQ(inst.src_operand(0), architectural_src0);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 0), 0x20000001u);
  }
}

template <typename Traits> void wave32_sdwa_explicit_compare_writes_only_destination_low() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_sdwa_sdst_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_sdwa_sdst_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;
  auto cu =
      amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_sdwa_sdst_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);

  constexpr uint32_t kSdst = 4;
  constexpr uint32_t kSource0 = 0;
  constexpr uint32_t kSource1 = 1;
  constexpr uint32_t kAdjacentSentinel = 0xA5A55A5Au;
  constexpr uint64_t kOldVcc = 0xC3C33C3C00000000ULL;
  const uint32_t sb = wf->sgpr_alloc().base;
  const uint32_t vb = wf->vgpr_alloc().base;

  typename Traits::VopcVopSdwaSdstEncMachineInst raw{};
  raw.src0 = amdgpu::SRC_SDWA;
  raw.vsrc1 = kSource1;
  raw.op = Traits::vcmp_eq_u32_op;
  raw.encoding = Traits::vopc_encoding;
  raw.vsrc0 = kSource0;
  raw.sdst = kSdst;
  raw.sd = 1;
  raw.src0_sel = amdgpu::sdwa::DWORD;
  raw.src1_sel = amdgpu::sdwa::DWORD;
  typename Traits::VCmpEqU32Vopc inst(reinterpret_cast<const typename Traits::MachineInst *>(&raw));

  for (bool force_scalar : {false, true}) {
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");
    ForceScalarGuard force_scalar_guard(force_scalar);
    wf->set_exec(0xFFFFFFFFULL);
    wf->set_vcc_raw(kOldVcc);
    cu->write_sgpr(sb + kSdst, 0);
    cu->write_sgpr(sb + kSdst + 1, kAdjacentSentinel);
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vb + kSource0, lane, lane + 1);
      cu->write_vgpr(vb + kSource1, lane, lane + 1);
    }

    inst.execute_impl(*wf);

    EXPECT_EQ(cu->read_sgpr(sb + kSdst), 0xFFFFFFFFu);
    EXPECT_EQ(cu->read_sgpr(sb + kSdst + 1), kAdjacentSentinel);
    EXPECT_EQ(wf->vcc(), kOldVcc);
  }
}

template <typename Traits> void cdna_generated_vop1_uses_shared_row_broadcast() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
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
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
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

template <typename Traits> void wave32_generated_vop1_dpp_write_mask_honors_bound_ctrl() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp_vop1_wave32_write_mask_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp_vop1_wave32_write_mask_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(
      std::string(Traits::name) + "_dpp_vop1_wave32_write_mask_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec(0xFFFFFFFFULL);

  constexpr uint32_t kSrc = 4;
  constexpr uint32_t kDst = 8;
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu->write_vgpr(vbase + kSrc, lane, 0x1000u + lane);
    cu->write_vgpr(vbase + kDst, lane, 0xDEAD0000u + lane);
  }

  typename Traits::Vop1VopDpp16MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
  raw.bound_ctrl = 0;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;

  typename Traits::VMovB32Vop1 inst(reinterpret_cast<const typename Traits::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  for (uint32_t lane = 0; lane < 32; ++lane) {
    uint32_t expected = lane % 16 == 0 ? 0xDEAD0000u + lane : 0x1000u + lane - 1;
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, lane), expected);
  }
}

template <typename Traits> void generated_vop1_dpp64_preserves_masked_destination() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp_vop1_b64_mask_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp_vop1_b64_mask_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_dpp_vop1_b64_mask_cu",
                                            cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), Traits::wf_size);
  wf->set_exec(Traits::wf_size == 64 ? ~0ULL : 0xFFFFFFFFULL);

  constexpr uint32_t kSrc = 4;
  constexpr uint32_t kDst = 8;
  constexpr uint64_t kSrcBase = 0x1122334400005000ULL;
  constexpr uint64_t kOldDstBase = 0xA5A500005A5A0000ULL;
  uint32_t vbase = wf->vgpr_alloc().base;
  uint64_t source_values[64]{};
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    source_values[lane] = kSrcBase + (static_cast<uint64_t>(lane) << 32) + lane;
    const uint64_t old_dst = kOldDstBase + lane;
    cu->write_vgpr(vbase + kSrc, lane, static_cast<uint32_t>(source_values[lane]));
    cu->write_vgpr(vbase + kSrc + 1, lane, static_cast<uint32_t>(source_values[lane] >> 32));
    cu->write_vgpr(vbase + kDst, lane, static_cast<uint32_t>(old_dst));
    cu->write_vgpr(vbase + kDst + 1, lane, static_cast<uint32_t>(old_dst >> 32));
  }

  typename Traits::Vop1Dpp64MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = amdgpu::dpp::ROW_SELECT_BASE;
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0x1;

  typename Traits::VMovB64Vop1 inst(reinterpret_cast<const typename Traits::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    const uint64_t actual = static_cast<uint64_t>(cu->read_vgpr(vbase + kDst, lane)) |
                            (static_cast<uint64_t>(cu->read_vgpr(vbase + kDst + 1, lane)) << 32);
    const uint64_t expected = lane < 16 ? source_values[0] : kOldDstBase + lane;
    EXPECT_EQ(actual, expected) << "lane " << lane;
  }
}

void rdna4_generated_vop1_64_preserves_inactive_destination() {
  ScopedIsaExecutionBackend execution_backend_scope{&rdna4::execution_backend()};
  amdgpu::GpuMemory mem("rdna4_vop1_f64_exec_mask_mem");
  amdgpu::L2Cache l2("rdna4_vop1_f64_exec_mask_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create("rdna4_vop1_f64_exec_mask_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  constexpr uint64_t kExec = 0x55555555ULL;
  wf->set_exec(kExec);

  constexpr uint32_t kSrc = 4;
  constexpr uint32_t kDst = 8;
  constexpr uint64_t kOldDstBase = 0xA5A500005A5A0000ULL;
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    const uint64_t old_dst = kOldDstBase + lane;
    cu->write_vgpr(vbase + kSrc, lane, 100u + lane);
    cu->write_vgpr(vbase + kDst, lane, static_cast<uint32_t>(old_dst));
    cu->write_vgpr(vbase + kDst + 1, lane, static_cast<uint32_t>(old_dst >> 32));
  }

  rdna4::Vop1MachineInst raw{};
  raw.src0 = 256 + kSrc;
  raw.vdst = kDst;

  rdna4::VCvtF64I32Vop1 inst(reinterpret_cast<const rdna4::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    const uint64_t expected = (kExec & (1ULL << lane))
                                  ? std::bit_cast<uint64_t>(static_cast<double>(100u + lane))
                                  : kOldDstBase + lane;
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, lane), static_cast<uint32_t>(expected))
        << "low dword, lane " << lane;
    EXPECT_EQ(cu->read_vgpr(vbase + kDst + 1, lane), static_cast<uint32_t>(expected >> 32))
        << "high dword, lane " << lane;
  }
}

template <typename Traits> void generated_vop1_dpp64_input_permutes_both_words() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp_vop1_b64_input_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp_vop1_b64_input_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_dpp_vop1_b64_input_cu",
                                            cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), Traits::wf_size);

  constexpr uint32_t kSrc = 4;
  constexpr uint32_t kDst = 8;
  uint32_t vbase = wf->vgpr_alloc().base;
  uint64_t source_values[64]{};
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    source_values[lane] = 0x2468ACE000001000ULL + (static_cast<uint64_t>(lane) << 32) + lane;
    cu->write_vgpr(vbase + kSrc, lane, static_cast<uint32_t>(source_values[lane]));
    cu->write_vgpr(vbase + kSrc + 1, lane, static_cast<uint32_t>(source_values[lane] >> 32));
  }
  // Every selected source contains meaningful data in both physical VGPRs;
  // losing or independently permuting either word changes the moved value.
  for (uint32_t row = 0; row < wf->wf_size() / 16; ++row) {
    uint64_t value = source_values[row * 16 + 3];
    ASSERT_NE(static_cast<uint32_t>(value), 0u);
    ASSERT_NE(static_cast<uint32_t>(value >> 32), 0u);
  }

  typename Traits::Vop1Dpp64MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = amdgpu::dpp::ROW_SELECT_BASE + 3;
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;

  typename Traits::VMovB64Vop1 inst(reinterpret_cast<const typename Traits::MachineInst *>(&raw));
  for (bool force_scalar : {false, true}) {
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");
    ForceScalarGuard force_scalar_guard(force_scalar);
    wf->set_exec(Traits::wf_size == 64 ? ~0ULL : 0xFFFFFFFFULL);
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vbase + kDst, lane, 0xDEADBEEFu);
      cu->write_vgpr(vbase + kDst + 1, lane, 0xBAD0C0DEu);
    }

    inst.execute_impl(*wf);

    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      uint32_t source_lane = (lane & ~15u) + 3u;
      uint64_t actual = static_cast<uint64_t>(cu->read_vgpr(vbase + kDst, lane)) |
                        (static_cast<uint64_t>(cu->read_vgpr(vbase + kDst + 1, lane)) << 32);
      EXPECT_EQ(actual, source_values[source_lane]) << "lane " << lane;
    }
  }
}

template <typename Traits> void wave32_generated_vop1_dpp16_fetch_inactive_uses_fi() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp16_fi_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp16_fi_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu =
      amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_dpp16_fi_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);

  constexpr uint32_t kSrc = 4;
  constexpr uint32_t kDst = 8;
  constexpr uint32_t kSrcLane0Value = 0xA5A50000u;
  uint32_t vbase = wf->vgpr_alloc().base;

  auto run = [&](uint32_t fi) {
    wf->set_exec(0xFFFFFFFEULL);
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vbase + kSrc, lane, kSrcLane0Value + lane);
      cu->write_vgpr(vbase + kDst, lane, 0xDEAD0000u + lane);
    }

    typename Traits::Vop1VopDpp16MachineInst raw{};
    raw.src0 = amdgpu::SRC_DPP;
    raw.vsrc0 = kSrc;
    raw.vdst = kDst;
    raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
    raw.fi = fi;
    raw.bound_ctrl = 1;
    raw.bank_mask = 0xF;
    raw.row_mask = 0xF;

    typename Traits::VMovB32Vop1 inst(reinterpret_cast<const typename Traits::MachineInst *>(&raw));
    inst.execute_impl(*wf);
    return cu->read_vgpr(vbase + kDst, 1);
  };

  EXPECT_EQ(run(0), 0u);
  EXPECT_EQ(run(1), kSrcLane0Value);
}

template <typename Traits> void wave32_generated_vop1_dpp16_fi_zero_obeys_arch_bound_ctrl() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp16_fi_bound_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp16_fi_bound_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_dpp16_fi_bound_cu", cfg,
                                            &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec(0xFFFFFFFEULL);

  constexpr uint32_t kSrc = 4;
  constexpr uint32_t kDst = 8;
  constexpr uint32_t kSrcLane0Value = 0xA5A50000u;
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu->write_vgpr(vbase + kSrc, lane, kSrcLane0Value + lane);
    cu->write_vgpr(vbase + kDst, lane, 0xDEAD0000u + lane);
  }

  typename Traits::Vop1VopDpp16MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
  raw.fi = 0;
  raw.bound_ctrl = 0;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;

  typename Traits::VMovB32Vop1 inst(reinterpret_cast<const typename Traits::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 0), 0xDEAD0000u);
  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 1), Traits::inactive_uses_bound_ctrl ? 0xDEAD0001u : 0u);
}

void rdna4_wave64_generated_vop1_dpp16_fetch_inactive_uses_upper_exec_bit() {
  ScopedIsaExecutionBackend execution_backend_scope{&rdna4::execution_backend()};
  amdgpu::GpuMemory mem("rdna4_wave64_dpp16_fi_mem");
  amdgpu::L2Cache l2("rdna4_wave64_dpp16_fi_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = rdna4::Isa::MAX_SGPRS_PER_WF;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create("rdna4_wave64_dpp16_fi_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf, 64);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);

  constexpr uint32_t kSrc = 4;
  constexpr uint32_t kDst = 8;
  constexpr uint32_t kSrcLane32Value = 0x5A5A0020u;
  uint32_t vbase = wf->vgpr_alloc().base;

  auto run = [&](uint32_t fi) {
    wf->set_exec(~(1ULL << 32u));
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vbase + kSrc, lane, 0x5A5A0000u + lane);
      cu->write_vgpr(vbase + kDst, lane, 0xDEAD0000u + lane);
    }

    rdna4::Vop1VopDpp16MachineInst raw{};
    raw.src0 = amdgpu::SRC_DPP;
    raw.vsrc0 = kSrc;
    raw.vdst = kDst;
    raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
    raw.fi = fi;
    raw.bound_ctrl = 0;
    raw.bank_mask = 0xF;
    raw.row_mask = 0xF;

    rdna4::VMovB32Vop1 inst(reinterpret_cast<const rdna4::MachineInst *>(&raw));
    inst.execute_impl(*wf);
    return cu->read_vgpr(vbase + kDst, 33);
  };

  EXPECT_EQ(run(0), 0xDEAD0021u);
  EXPECT_EQ(run(1), kSrcLane32Value);
}

void rdna4_wave64_generated_vop1_dpp8_fetch_inactive_uses_upper_exec_bit() {
  ScopedIsaExecutionBackend execution_backend_scope{&rdna4::execution_backend()};
  amdgpu::GpuMemory mem("rdna4_wave64_dpp8_fi_mem");
  amdgpu::L2Cache l2("rdna4_wave64_dpp8_fi_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = rdna4::Isa::MAX_SGPRS_PER_WF;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create("rdna4_wave64_dpp8_fi_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf, 64);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);

  constexpr uint32_t kSrc = 4;
  constexpr uint32_t kDst = 8;
  uint32_t vbase = wf->vgpr_alloc().base;

  struct Dpp8Case {
    uint32_t dst_lane;
    uint32_t src_lane;
  };
  constexpr std::array<Dpp8Case, 2> cases{{
      {33, 32},
      {35, 37},
  }};

  auto set_lane_sel = [](rdna4::Vop1VopDpp8MachineInst &raw, uint32_t dst_lane, uint32_t src_lane) {
    uint32_t selector = src_lane & 7u;
    switch (dst_lane & 7u) {
    case 0:
      raw.lane_sel_0 = selector;
      break;
    case 1:
      raw.lane_sel_1 = selector;
      break;
    case 2:
      raw.lane_sel_2 = selector;
      break;
    case 3:
      raw.lane_sel_3 = selector;
      break;
    case 4:
      raw.lane_sel_4 = selector;
      break;
    case 5:
      raw.lane_sel_5 = selector;
      break;
    case 6:
      raw.lane_sel_6 = selector;
      break;
    case 7:
      raw.lane_sel_7 = selector;
      break;
    }
  };

  auto run = [&](const Dpp8Case &test_case, uint32_t src0_marker) {
    wf->set_exec(~(1ULL << test_case.src_lane));
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vbase + kSrc, lane, 0x6B6B0000u + lane);
      cu->write_vgpr(vbase + kDst, lane, 0xDEAD0000u + lane);
    }

    rdna4::Vop1VopDpp8MachineInst raw{};
    raw.src0 = src0_marker;
    raw.vsrc0 = kSrc;
    raw.vdst = kDst;
    set_lane_sel(raw, test_case.dst_lane, test_case.src_lane);

    rdna4::VMovB32Vop1 inst(reinterpret_cast<const rdna4::MachineInst *>(&raw));
    inst.execute_impl(*wf);
    return cu->read_vgpr(vbase + kDst, test_case.dst_lane);
  };

  for (const auto &test_case : cases) {
    SCOPED_TRACE("dst lane " + std::to_string(test_case.dst_lane) + " src lane " +
                 std::to_string(test_case.src_lane));
    EXPECT_EQ(run(test_case, amdgpu::SRC_DPP8_FI_0), 0u);
    EXPECT_EQ(run(test_case, amdgpu::SRC_DPP8_FI_1), 0x6B6B0000u + test_case.src_lane);
  }
}

void rdna4_wave64_generated_dpp_compares_mask_upper_results() {
  ScopedIsaExecutionBackend execution_backend_scope{&rdna4::execution_backend()};
  amdgpu::GpuMemory mem("rdna4_wave64_dpp_compare_mem");
  amdgpu::L2Cache l2("rdna4_wave64_dpp_compare_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = rdna4::Isa::MAX_SGPRS_PER_WF;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create("rdna4_wave64_dpp_compare_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf, 64);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);

  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  constexpr uint32_t kScalarDst = 12;
  constexpr uint64_t kOldExec = ~((1ULL << 50) | (1ULL << 57));
  // Lane 49 is the only active, row/bank-enabled lane whose comparison is
  // true. Invalid-source and row/bank-masked compare results are zero rather
  // than restored from the old VCC, SGPR, or EXEC value.
  constexpr uint64_t kExpected = 1ULL << 49;
  uint32_t vbase = wf->vgpr_alloc().base;
  uint32_t sbase = wf->sgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu->write_vgpr(vbase + kSrc0, lane, 0x1000u + lane);
    cu->write_vgpr(vbase + kSrc1, lane, 0xDEAD0000u + lane);
  }
  // Lane 49 is a valid, enabled true result (it selects source lane 48).
  cu->write_vgpr(vbase + kSrc1, 49, 0x1030u);

  auto make_vopc = [] {
    rdna4::VopcVopDpp16MachineInst raw{};
    raw.src0 = amdgpu::SRC_DPP;
    raw.vsrc0 = kSrc0;
    raw.vsrc1 = kSrc1;
    raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
    raw.fi = 0;
    raw.bound_ctrl = 0;
    raw.bank_mask = 0xD;
    raw.row_mask = 0x8;
    return raw;
  };
  auto make_vop3 = [] {
    rdna4::Vop3VopDpp16MachineInst raw{};
    raw.vdst = kScalarDst;
    raw.src0 = amdgpu::SRC_DPP;
    raw.src1 = 256 + kSrc1;
    raw.vsrc0 = kSrc0;
    raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
    raw.fi = 0;
    raw.bound_ctrl = 0;
    raw.bank_mask = 0xD;
    raw.row_mask = 0x8;
    return raw;
  };

  wf->set_exec(kOldExec);
  wf->set_vcc(~0ULL);
  auto vopc_raw = make_vopc();
  rdna4::VCmpEqU32Vopc vopc(reinterpret_cast<const rdna4::MachineInst *>(&vopc_raw));
  vopc.execute_impl(*wf);
  EXPECT_EQ(wf->vcc(), kExpected);

  wf->set_exec(kOldExec);
  auto vopcx_raw = make_vopc();
  rdna4::VCmpxEqU32Vopc vopcx(reinterpret_cast<const rdna4::MachineInst *>(&vopcx_raw));
  vopcx.execute_impl(*wf);
  EXPECT_EQ(wf->exec(), kExpected);

  wf->set_exec(kOldExec);
  cu->write_sgpr(sbase + kScalarDst, 0xFFFFFFFFu);
  cu->write_sgpr(sbase + kScalarDst + 1, 0xFFFFFFFFu);
  auto vop3_raw = make_vop3();
  rdna4::VCmpEqU32Vop3 vop3(reinterpret_cast<const rdna4::MachineInst *>(&vop3_raw));
  vop3.execute_impl(*wf);
  EXPECT_EQ(cu->read_sgpr(sbase + kScalarDst), 0u);
  EXPECT_EQ(cu->read_sgpr(sbase + kScalarDst + 1), static_cast<uint32_t>(kExpected >> 32));

  wf->set_exec(kOldExec);
  auto vop3x_raw = make_vop3();
  rdna4::VCmpxEqU32Vop3 vop3x(reinterpret_cast<const rdna4::MachineInst *>(&vop3x_raw));
  vop3x.execute_impl(*wf);
  EXPECT_EQ(wf->exec(), kExpected);
}

void rdna4_generated_vop2_dpp_inactive_source_preserves_destination() {
  ScopedIsaExecutionBackend execution_backend_scope{&rdna4::execution_backend()};
  amdgpu::GpuMemory mem("rdna4_vop2_dpp_inactive_mem");
  amdgpu::L2Cache l2("rdna4_vop2_dpp_inactive_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = rdna4::Isa::MAX_SGPRS_PER_WF;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("rdna4_vop2_dpp_inactive_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xFFFFFFFEULL);

  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  constexpr uint32_t kDst = 12;
  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase + kSrc0, 0, std::bit_cast<uint32_t>(1.0f));
  cu->write_vgpr(vbase + kSrc1, 1, std::bit_cast<uint32_t>(2.0f));
  cu->write_vgpr(vbase + kDst, 1, std::bit_cast<uint32_t>(42.0f));

  rdna4::Vop2VopDpp16MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc0;
  raw.vsrc1 = kSrc1;
  raw.vdst = kDst;
  raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
  raw.fi = 0;
  raw.bound_ctrl = 0;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;
  rdna4::VAddF32Vop2 inst(reinterpret_cast<const rdna4::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vbase + kDst, 1), std::bit_cast<uint32_t>(42.0f));
}

template <typename Traits> void generated_vop2_dpp_carry_uses_source_write_mask_only() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_vop2_dpp_carry_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_vop2_dpp_carry_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_vop2_dpp_carry_cu", cfg,
                                            &mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), Traits::wf_size);

  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  constexpr uint32_t kDst = 12;
  constexpr uint32_t kSentinel = 0xCAFE1234u;
  uint32_t vbase = wf->vgpr_alloc().base;

  auto make_raw = [] {
    typename Traits::Vop2DppCarryMachineInst raw{};
    raw.src0 = amdgpu::SRC_DPP;
    raw.vsrc0 = kSrc0;
    raw.vsrc1 = kSrc1;
    raw.vdst = kDst;
    raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
    if constexpr (Traits::dpp_carry_has_fi)
      raw.fi = 1;
    raw.bound_ctrl = 0;
    raw.bank_mask = 0xF;
    raw.row_mask = 0xF;
    return raw;
  };

  auto execute = [&](const typename Traits::Vop2DppCarryMachineInst &raw) {
    typename Traits::Vop2DppCarryInst inst(
        reinterpret_cast<const typename Traits::MachineInst *>(&raw));
    inst.execute_impl(*wf);
  };

  auto fill_vgprs = [&](uint32_t src0, uint32_t src1) {
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vbase + kSrc0, lane, src0);
      cu->write_vgpr(vbase + kSrc1, lane, src1);
      cu->write_vgpr(vbase + kDst, lane, kSentinel);
    }
  };

  const uint64_t full_exec = wf->wf_size() == 64 ? ~0ULL : 0xFFFFFFFFULL;
  const uint64_t vcc_hi_sentinel = wf->wf_size() == 32 ? 0xA5A5A5A500000000ULL : 0;

  for (bool force_scalar : {false, true}) {
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");
    ForceScalarGuard force_scalar_guard(force_scalar);

    fill_vgprs(0, 0);

    // ROW_SHR1 has no source for lane 0. BOUND_CTRL=0 suppresses both the
    // vector destination and the VCC side result for that active lane.
    wf->set_exec(full_exec);
    wf->set_vcc_raw(vcc_hi_sentinel | 1);
    auto bc0_raw = make_raw();
    execute(bc0_raw);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 0), kSentinel);
    EXPECT_EQ(wf->vcc(), vcc_hi_sentinel | 1);

    // BOUND_CTRL=1 supplies zero instead, so the lane executes normally and
    // consumes its incoming carry bit.
    wf->set_exec(full_exec);
    wf->set_vcc_raw(vcc_hi_sentinel | 1);
    cu->write_vgpr(vbase + kDst, 0, kSentinel);
    auto bc1_raw = make_raw();
    bc1_raw.bound_ctrl = 1;
    execute(bc1_raw);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 0), Traits::dpp_carry_has_carry_in ? 1u : 0u);
    EXPECT_EQ(wf->vcc(), vcc_hi_sentinel);

    // Row/bank masks select only the VGPR destination. A bank-masked lane's
    // carry bit is still fully written.
    fill_vgprs(0xFFFFFFFFu, 1);
    wf->set_exec(full_exec);
    wf->set_vcc_raw(0);
    auto row_bank_raw = make_raw();
    row_bank_raw.dpp_ctrl = amdgpu::dpp::ROW_SELECT_BASE;
    row_bank_raw.bank_mask = 0xE;
    execute(row_bank_raw);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 0), kSentinel);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 4), 0u);
    EXPECT_EQ(wf->vcc(), full_exec);

    // Row masking has the same VGPR-only scope. Exercise a distinct row so
    // this cannot pass solely through the bank-mask implementation.
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      cu->write_vgpr(vbase + kDst, lane, kSentinel);
    wf->set_exec(full_exec);
    wf->set_vcc_raw(0);
    auto row_mask_raw = make_raw();
    row_mask_raw.dpp_ctrl = amdgpu::dpp::ROW_SELECT_BASE;
    row_mask_raw.row_mask = 0xD;
    execute(row_mask_raw);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 16), kSentinel);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, wf->wf_size() == 64 ? 32 : 0), 0u);
    EXPECT_EQ(wf->vcc(), full_exec);

    // On modern DPP16, FI=0 makes lane 1's ROW_SHR1 source invalid when lane
    // 0 is inactive. BOUND_CTRL=0 therefore preserves lane 1's old carry,
    // while the inactive destination lane itself is cleared in VCC.
    if constexpr (Traits::dpp_carry_has_fi) {
      fill_vgprs(0, 0);
      wf->set_exec(full_exec & ~1ULL);
      wf->set_vcc_raw(3);
      auto inactive_source_raw = make_raw();
      inactive_source_raw.fi = 0;
      execute(inactive_source_raw);
      EXPECT_EQ(cu->read_vgpr(vbase + kDst, 1), kSentinel);
      EXPECT_EQ(wf->vcc(), 2u);
    } else {
      // CDNA4 is Wave64-only. Verify that source suppression and legacy
      // inactive-source reads work in upper rows with sparse EXEC/VCC masks.
      constexpr uint32_t kUpperRowFirstLane = 48;
      fill_vgprs(0, 0);
      wf->set_exec(1ULL << kUpperRowFirstLane);
      wf->set_vcc(1ULL << kUpperRowFirstLane);
      auto upper_invalid_raw = make_raw();
      execute(upper_invalid_raw);
      EXPECT_EQ(cu->read_vgpr(vbase + kDst, kUpperRowFirstLane), kSentinel);
      EXPECT_EQ(wf->vcc(), 1ULL << kUpperRowFirstLane);

      fill_vgprs(0, 0);
      cu->write_vgpr(vbase + kSrc0, kUpperRowFirstLane, 0xFFFFFFFFu);
      cu->write_vgpr(vbase + kSrc1, kUpperRowFirstLane + 1, 1);
      wf->set_exec(1ULL << (kUpperRowFirstLane + 1));
      wf->set_vcc(0);
      auto upper_valid_raw = make_raw();
      execute(upper_valid_raw);
      EXPECT_EQ(cu->read_vgpr(vbase + kDst, kUpperRowFirstLane + 1), 0u);
      EXPECT_EQ(wf->vcc(), 1ULL << (kUpperRowFirstLane + 1));
    }
  }
}

template <typename Traits> void wave32_generated_vop1_dpp8_fetch_inactive_uses_fi() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp8_fi_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp8_fi_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu =
      amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_dpp8_fi_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);

  constexpr uint32_t kSrc = 4;
  constexpr uint32_t kDst = 8;
  constexpr uint32_t kSrcLane0Value = 0x5A5A0000u;
  uint32_t vbase = wf->vgpr_alloc().base;

  auto run = [&](uint32_t src0_marker) {
    wf->set_exec(0xFFFFFFFEULL);
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vbase + kSrc, lane, kSrcLane0Value + lane);
      cu->write_vgpr(vbase + kDst, lane, 0xDEAD0000u + lane);
    }

    typename Traits::Vop1VopDpp8MachineInst raw{};
    raw.src0 = src0_marker;
    raw.vsrc0 = kSrc;
    raw.vdst = kDst;

    typename Traits::VMovB32Vop1 inst(reinterpret_cast<const typename Traits::MachineInst *>(&raw));
    inst.execute_impl(*wf);
    return cu->read_vgpr(vbase + kDst, 1);
  };

  EXPECT_EQ(run(amdgpu::SRC_DPP8_FI_0), 0u);
  EXPECT_EQ(run(amdgpu::SRC_DPP8_FI_1), kSrcLane0Value);
}

template <typename Traits> void wave32_generated_vopc_dpp_invalid_source_bc0_zeros_result() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp_vopc_wave32_write_mask_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp_vopc_wave32_write_mask_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(
      std::string(Traits::name) + "_dpp_vopc_wave32_write_mask_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec(0xFFFFFFFFULL);

  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    uint32_t src = 0x1000u + lane;
    uint32_t cmp = lane == 0 ? 0u : 0x1000u + lane - 1;
    if (lane == 20)
      cmp = 0xDEAD0020u;
    cu->write_vgpr(vbase + kSrc0, lane, src);
    cu->write_vgpr(vbase + kSrc1, lane, cmp);
  }

  typename Traits::VopcVopDpp16MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc1 = kSrc1;
  raw.vsrc0 = kSrc0;
  raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
  raw.fi = 1;
  raw.bound_ctrl = 0;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;

  for (bool force_scalar : {false, true}) {
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");
    ForceScalarGuard force_scalar_guard(force_scalar);
    for (uint64_t old_vcc : {0xA5A5A5A500000000ULL, 0x5A5A5A5AFFFFFFFFULL}) {
      wf->set_exec(0xFFFFFFFFULL);
      wf->set_vcc_raw(old_vcc);
      typename Traits::VCmpEqU32Vopc inst(
          reinterpret_cast<const typename Traits::MachineInst *>(&raw));
      inst.execute_impl(*wf);
      EXPECT_EQ(wf->vcc(), (old_vcc & 0xFFFFFFFF00000000ULL) | 0xFFEEFFFEULL);
    }
  }
}

template <typename Traits> void wave32_generated_vopc_dpp_zeros_masked_compare_bits() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp_vopc_masked_result_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp_vopc_masked_result_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(
      std::string(Traits::name) + "_dpp_vopc_masked_result_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);

  constexpr uint64_t kOldExec = 0xFFFFFDFBULL; // source lane 2 and destination lane 9 inactive
  wf->set_exec(kOldExec);

  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu->write_vgpr(vbase + kSrc0, lane, 0x1000u + lane);
    cu->write_vgpr(vbase + kSrc1, lane, 0xDEAD0000u + lane);
  }
  cu->write_vgpr(vbase + kSrc1, 1, 0x1000u);
  cu->write_vgpr(vbase + kSrc1, 0, 0u);
  cu->write_vgpr(vbase + kSrc1, 3, 0u);
  cu->write_vgpr(vbase + kSrc1, 10, 0u);

  typename Traits::VopcVopDpp16MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc1 = kSrc1;
  raw.vsrc0 = kSrc0;
  raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
  raw.fi = 0;
  raw.bound_ctrl = 0;
  raw.bank_mask = 0xD;
  raw.row_mask = 0x1;

  for (bool force_scalar : {false, true}) {
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");
    ForceScalarGuard force_scalar_guard(force_scalar);
    for (uint64_t old_vcc : {0xA5A5A5A500000000ULL, 0x5A5A5A5AFFFFFFFFULL}) {
      wf->set_exec(kOldExec);
      wf->set_vcc_raw(old_vcc);
      typename Traits::VCmpEqU32Vopc inst(
          reinterpret_cast<const typename Traits::MachineInst *>(&raw));
      inst.execute_impl(*wf);

      // Lane 1 compares true. Lane 0 has an OOB source, while lanes 3 and 10
      // select inactive sources with FI=0; all three invalid-source result bits
      // are forced to zero independently of their old VCC values. Row/bank-
      // masked lanes and inactive destination lanes are also zero.
      EXPECT_EQ(wf->vcc(), (old_vcc & 0xFFFFFFFF00000000ULL) | 0x2u);
    }

    raw.bound_ctrl = 1;
    for (uint64_t old_vcc : {0xA5A5A5A500000000ULL, 0x5A5A5A5AFFFFFFFFULL}) {
      wf->set_exec(kOldExec);
      wf->set_vcc_raw(old_vcc);
      typename Traits::VCmpEqU32Vopc zero_fill(
          reinterpret_cast<const typename Traits::MachineInst *>(&raw));
      zero_fill.execute_impl(*wf);

      // BOUND_CTRL=1 substitutes zero, so the deliberately-zero comparison
      // operands make the same invalid-source lanes true rather than forcing
      // their result bits to zero.
      EXPECT_EQ(wf->vcc(), (old_vcc & 0xFFFFFFFF00000000ULL) | 0x40Bu);
    }
    raw.bound_ctrl = 0;
  }
}

template <typename Traits> void wave32_generated_vop3_dpp_masks_actual_compare_destination() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp_vop3_cmp_result_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp_vop3_cmp_result_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_dpp_vop3_cmp_result_cu",
                                            cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);

  constexpr uint64_t kOldExec = 0xFFFFFDFBULL;
  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  constexpr uint32_t kScalarDst = 12;
  uint32_t vbase = wf->vgpr_alloc().base;
  uint32_t sbase = wf->sgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu->write_vgpr(vbase + kSrc0, lane, 0x1000u + lane);
    cu->write_vgpr(vbase + kSrc1, lane, 0xDEAD0000u + lane);
  }
  cu->write_vgpr(vbase + kSrc1, 1, 0x1000u);

  auto make_raw = [] {
    typename Traits::Vop3VopDpp16MachineInst raw{};
    raw.vdst = kScalarDst;
    raw.src0 = amdgpu::SRC_DPP;
    raw.src1 = 256 + kSrc1;
    raw.vsrc0 = kSrc0;
    raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
    raw.fi = 0;
    raw.bound_ctrl = 0;
    raw.bank_mask = 0xD;
    raw.row_mask = 0x1;
    return raw;
  };

  for (bool force_scalar : {false, true}) {
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");
    ForceScalarGuard force_scalar_guard(force_scalar);

    for (uint32_t old_result : {0u, 0xFFFFFFFFu}) {
      wf->set_exec(kOldExec);
      cu->write_sgpr(sbase + kScalarDst, old_result);
      auto cmp_raw = make_raw();
      typename Traits::VCmpEqU32Vop3 cmp(
          reinterpret_cast<const typename Traits::MachineInst *>(&cmp_raw));
      cmp.execute_impl(*wf);
      EXPECT_EQ(cu->read_sgpr(sbase + kScalarDst), 0x2u);
      EXPECT_EQ(wf->exec(), kOldExec);
    }

    wf->set_exec(kOldExec);
    cu->write_sgpr(sbase + kScalarDst, 0xFFFFFFFFu);
    auto cmp_raw = make_raw();
    cmp_raw.bound_ctrl = 1;
    typename Traits::VCmpEqU32Vop3 zero_fill_cmp(
        reinterpret_cast<const typename Traits::MachineInst *>(&cmp_raw));
    zero_fill_cmp.execute_impl(*wf);
    EXPECT_EQ(cu->read_sgpr(sbase + kScalarDst), 0x2u);
    EXPECT_EQ(wf->exec(), kOldExec);

    wf->set_exec(kOldExec);
    auto cmpx_raw = make_raw();
    typename Traits::VCmpxEqU32Vop3 cmpx(
        reinterpret_cast<const typename Traits::MachineInst *>(&cmpx_raw));
    cmpx.execute_impl(*wf);
    EXPECT_EQ(wf->exec(), 0x2u);
  }
}

template <typename Traits> void wave32_generated_legal_vop3p_dpp_masks_vgpr_destination() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp_vop3p_result_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp_vop3p_result_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_dpp_vop3p_result_cu", cfg,
                                            &mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);

  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  constexpr uint32_t kSrc2 = 9;
  constexpr uint32_t kDst = 12;
  constexpr uint32_t kPackedF16Two = 0x40004000u;
  constexpr uint32_t kPackedF16Three = 0x42004200u;
  constexpr uint32_t kPackedF16Four = 0x44004400u;
  uint32_t vbase = wf->vgpr_alloc().base;

  auto make_raw = [] {
    typename Traits::Vop3pVopDpp16MachineInst raw{};
    raw.vdst = kDst;
    raw.src0 = amdgpu::SRC_DPP;
    raw.src1 = 256 + kSrc1;
    raw.src2 = 256 + kSrc2;
    raw.vsrc0 = kSrc0;
    Traits::set_aligned_vop3p_opsel(raw);
    raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
    raw.fi = 1;
    raw.bound_ctrl = 0;
    raw.bank_mask = 0xF;
    raw.row_mask = 0xF;
    return raw;
  };

  for (bool force_scalar : {false, true}) {
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");
    ForceScalarGuard force_scalar_guard(force_scalar);
    wf->set_exec(0xFFFFFFFFULL);
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vbase + kSrc0, lane, kPackedF16Two);
      cu->write_vgpr(vbase + kSrc1, lane, kPackedF16Three);
      cu->write_vgpr(vbase + kSrc2, lane, kPackedF16Four);
      cu->write_vgpr(vbase + kDst, lane, std::bit_cast<uint32_t>(99.0f + lane));
    }

    auto preserve_raw = make_raw();
    typename Traits::VFmaMixF32Vop3p preserve(
        reinterpret_cast<const typename Traits::MachineInst *>(&preserve_raw));
    preserve.execute_impl(*wf);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 0), std::bit_cast<uint32_t>(99.0f));
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 1), std::bit_cast<uint32_t>(10.0f));

    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      cu->write_vgpr(vbase + kDst, lane, std::bit_cast<uint32_t>(99.0f + lane));
    auto zero_raw = make_raw();
    zero_raw.bound_ctrl = 1;
    typename Traits::VFmaMixF32Vop3p zero_fill(
        reinterpret_cast<const typename Traits::MachineInst *>(&zero_raw));
    zero_fill.execute_impl(*wf);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 0), std::bit_cast<uint32_t>(4.0f));
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 1), std::bit_cast<uint32_t>(10.0f));

    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      cu->write_vgpr(vbase + kDst, lane, std::bit_cast<uint32_t>(99.0f + lane));
    auto masked_raw = make_raw();
    masked_raw.dpp_ctrl = amdgpu::dpp::ROW_SELECT_BASE;
    masked_raw.bank_mask = 0xE;
    typename Traits::VFmaMixF32Vop3p masked(
        reinterpret_cast<const typename Traits::MachineInst *>(&masked_raw));
    masked.execute_impl(*wf);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 0), std::bit_cast<uint32_t>(99.0f));
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 3), std::bit_cast<uint32_t>(102.0f));
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 4), std::bit_cast<uint32_t>(10.0f));
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 16), std::bit_cast<uint32_t>(115.0f));
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 20), std::bit_cast<uint32_t>(10.0f));
  }
}

template <typename Traits>
void wave32_generated_vop3_sdst_dpp_masks_vector_and_scalar_destinations() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp_vop3_sdst_result_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp_vop3_sdst_result_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_dpp_vop3_sdst_result_cu",
                                            cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);

  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  constexpr uint32_t kDst = 12;
  constexpr uint32_t kCarryIn = 16;
  constexpr uint32_t kCarryOut = 20;
  uint32_t vbase = wf->vgpr_alloc().base;
  uint32_t sbase = wf->sgpr_alloc().base;

  auto make_raw = [] {
    typename Traits::Vop3SdstEncVopDpp16MachineInst raw{};
    raw.vdst = kDst;
    raw.sdst = kCarryOut;
    raw.src0 = amdgpu::SRC_DPP;
    raw.src1 = 256 + kSrc1;
    raw.src2 = kCarryIn;
    raw.vsrc0 = kSrc0;
    raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
    raw.fi = 1;
    raw.bound_ctrl = 0;
    raw.bank_mask = 0xF;
    raw.row_mask = 0xF;
    return raw;
  };

  auto initialize = [&] {
    wf->set_exec(0xFFFFFFFFULL);
    cu->write_sgpr(sbase + kCarryIn, 0u);
    cu->write_sgpr(sbase + kCarryIn + 1, 0u);
    cu->write_sgpr(sbase + kCarryOut, 1u);
    cu->write_sgpr(sbase + kCarryOut + 1, 0xA5A5A5A5u);
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vbase + kSrc0, lane, lane == 0 ? 0u : 0xFFFFFFFFu);
      cu->write_vgpr(vbase + kSrc1, lane, 1u);
      cu->write_vgpr(vbase + kDst, lane, 0xABCD0000u + lane);
    }
  };

  for (bool force_scalar : {false, true}) {
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");
    ForceScalarGuard force_scalar_guard(force_scalar);

    initialize();
    auto preserve_raw = make_raw();
    typename Traits::VAddCoCiU32Vop3SdstEnc preserve(
        reinterpret_cast<const typename Traits::MachineInst *>(&preserve_raw));
    preserve.execute_impl(*wf);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 0), 0xABCD0000u);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 1), 1u);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 2), 0u);
    EXPECT_EQ(cu->read_sgpr(sbase + kCarryOut), 0xFFFEFFFDu);
    EXPECT_EQ(cu->read_sgpr(sbase + kCarryOut + 1), 0xA5A5A5A5u);

    initialize();
    auto zero_raw = make_raw();
    zero_raw.bound_ctrl = 1;
    typename Traits::VAddCoCiU32Vop3SdstEnc zero_fill(
        reinterpret_cast<const typename Traits::MachineInst *>(&zero_raw));
    zero_fill.execute_impl(*wf);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 0), 1u);
    EXPECT_EQ(cu->read_sgpr(sbase + kCarryOut), 0xFFFEFFFCu);

    initialize();
    // ROW_SELECT_BASE broadcasts lane 0 within each row. Keep every other
    // lane distinct so the scalar carry mask proves that even row/bank-masked
    // destination lanes consume the permuted source rather than their
    // unpermuted src0 value.
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      cu->write_vgpr(vbase + kSrc0, lane, lane % 16 == 0 ? 0xFFFFFFFFu : 0u);
    cu->write_sgpr(sbase + kCarryOut, 0u);
    auto masked_raw = make_raw();
    masked_raw.dpp_ctrl = amdgpu::dpp::ROW_SELECT_BASE;
    masked_raw.bank_mask = 0xE;
    typename Traits::VAddCoCiU32Vop3SdstEnc masked(
        reinterpret_cast<const typename Traits::MachineInst *>(&masked_raw));
    masked.execute_impl(*wf);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 0), 0xABCD0000u);
    EXPECT_EQ(cu->read_vgpr(vbase + kDst, 4), 0u);
    EXPECT_EQ(cu->read_sgpr(sbase + kCarryOut), 0xFFFFFFFFu);
  }
}

template <typename Traits> void wave32_generated_bc1_compare_families_clear_stale_bits() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp_bc1_compare_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp_bc1_compare_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_dpp_bc1_compare_cu", cfg,
                                            &mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);

  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  constexpr uint32_t kScalarDst = 12;
  constexpr uint64_t kActiveExec = 0xFFFFFFDFULL;
  constexpr uint64_t kExpected = 0xFFFEFFDFULL;
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu->write_vgpr(vbase + kSrc0, lane, 0x1000u + lane);
    cu->write_vgpr(vbase + kSrc1, lane, lane == 0 ? 0u : 0x1000u + lane - 1);
  }

  typename Traits::VopcVopDpp16MachineInst vopc_raw{};
  vopc_raw.src0 = amdgpu::SRC_DPP;
  vopc_raw.vsrc1 = kSrc1;
  vopc_raw.vsrc0 = kSrc0;
  vopc_raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
  vopc_raw.fi = 1;
  vopc_raw.bound_ctrl = 1;
  vopc_raw.bank_mask = 0xF;
  vopc_raw.row_mask = 0xF;

  wf->set_exec(kActiveExec);
  constexpr uint64_t kVccHiSentinel = 0xA5A5A5A500000000ULL;
  wf->set_vcc_raw(kVccHiSentinel | 0xFFFFFFFFULL);
  typename Traits::VCmpEqU32Vopc vopc(
      reinterpret_cast<const typename Traits::MachineInst *>(&vopc_raw));
  vopc.execute_impl(*wf);
  EXPECT_EQ(wf->vcc(), kVccHiSentinel | kExpected);

  constexpr uint64_t kOldVcc = 0xA5A5A5A5ULL;
  wf->set_exec(kActiveExec);
  wf->set_vcc_raw(kOldVcc);
  typename Traits::VCmpxEqU32Vopc vopcx(
      reinterpret_cast<const typename Traits::MachineInst *>(&vopc_raw));
  vopcx.execute_impl(*wf);
  EXPECT_EQ(wf->exec(), kExpected);
  EXPECT_EQ(wf->vcc(), kOldVcc);

  typename Traits::Vop3VopDpp16MachineInst vop3_raw{};
  vop3_raw.vdst = kScalarDst;
  vop3_raw.src0 = amdgpu::SRC_DPP;
  vop3_raw.src1 = 256 + kSrc1;
  vop3_raw.vsrc0 = kSrc0;
  vop3_raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
  vop3_raw.fi = 1;
  vop3_raw.bound_ctrl = 1;
  vop3_raw.bank_mask = 0xF;
  vop3_raw.row_mask = 0xF;
  wf->set_exec(kActiveExec);
  typename Traits::VCmpxEqU32Vop3 vop3_cmpx(
      reinterpret_cast<const typename Traits::MachineInst *>(&vop3_raw));
  vop3_cmpx.execute_impl(*wf);
  EXPECT_EQ(wf->exec(), kExpected);
}

template <typename Traits>
void wave32_generated_vcmpx_dpp_zeros_invalid_sources_preserves_exec_hi() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_dpp_vcmpx_wave32_exec_mask_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_dpp_vcmpx_wave32_exec_mask_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(
      std::string(Traits::name) + "_dpp_vcmpx_wave32_exec_mask_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);

  constexpr uint64_t kOldExec = 0xA5A55A5AFFFF005BULL;
  constexpr uint64_t kOldVcc = 0x00000000000000A5ULL;
  wf->set_exec_raw(kOldExec);
  wf->set_vcc(kOldVcc);

  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    uint32_t src = 0x1000u + lane;
    uint32_t cmp = lane == 0 ? 0u : 0x1000u + lane - 1;
    cu->write_vgpr(vbase + kSrc0, lane, src);
    cu->write_vgpr(vbase + kSrc1, lane, cmp);
  }

  typename Traits::VopcVopDpp16MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc1 = kSrc1;
  raw.vsrc0 = kSrc0;
  raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
  raw.fi = 1;
  raw.bound_ctrl = 0;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;

  typename Traits::VCmpxEqU32Vopc inst(
      reinterpret_cast<const typename Traits::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  EXPECT_EQ(wf->vcc(), kOldVcc);
  EXPECT_EQ(wf->exec(), 0x00000000FFFE005AULL);
  EXPECT_EQ(wf->exec_raw(), 0xA5A55A5AFFFE005AULL);
}

template <typename Traits> void wave32_generated_vcmpx_preserves_exec_hi() {
  SCOPED_TRACE(Traits::name);
  ScopedIsaExecutionBackend execution_backend_scope{&Traits::backend()};
  amdgpu::GpuMemory mem(std::string(Traits::name) + "_vcmpx_wave32_exec_hi_mem");
  amdgpu::L2Cache l2(std::string(Traits::name) + "_vcmpx_wave32_exec_hi_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = Traits::arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(std::string(Traits::name) + "_vcmpx_wave32_exec_hi_cu",
                                            cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec_raw(0xA5A55A5AFFFFFFFFULL);

  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu->write_vgpr(vbase + kSrc0, lane, lane);
    cu->write_vgpr(vbase + kSrc1, lane, (lane & 4u) == 0 ? lane : lane + 1);
  }

  constexpr uint32_t kVgprSrcEncodingBase = 256;
  typename Traits::VopcMachineInst raw{};
  raw.src0 = kVgprSrcEncodingBase + kSrc0;
  raw.vsrc1 = kSrc1;

  typename Traits::VCmpxEqU32Vopc inst(
      reinterpret_cast<const typename Traits::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  EXPECT_EQ(wf->exec(), 0x0F0F0F0FULL);
  EXPECT_EQ(wf->exec_raw(), 0xA5A55A5A0F0F0F0FULL);
}

template <typename Traits> void unsupported_rdna_vopc_dpp_rejects() {
  SCOPED_TRACE(Traits::name);

  auto decoder = Decoder::create(Traits::arch);
  ASSERT_NE(decoder, nullptr);
  auto expect_rejection = [&](uint32_t src0) {
    typename Traits::Vop1VopDpp16MachineInst raw{};
    raw.src0 = src0;
    raw.op = Traits::vopc_opcode;
    raw.encoding = 62;
    EXPECT_TRUE(decode_fails(*decoder, reinterpret_cast<const uint32_t *>(&raw)));
  };

  expect_rejection(amdgpu::SRC_DPP);
  expect_rejection(amdgpu::SRC_DPP8_FI_0);
  expect_rejection(amdgpu::SRC_DPP8_FI_1);
}

template <typename Traits> void unsupported_cdna_vopc_dpp_rejects() {
  SCOPED_TRACE(Traits::name);

  typename Traits::Vop1VopDppMachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.op = Traits::vopc_opcode;
  raw.encoding = 62;
  auto decoder = Decoder::create(Traits::arch);
  ASSERT_NE(decoder, nullptr);
  EXPECT_TRUE(decode_fails(*decoder, reinterpret_cast<const uint32_t *>(&raw)));
}

void cdna4_vop1_sdwa_availability_is_instruction_specific() {
  cdna4::Vop1VopSdwaMachineInst raw{};
  raw.src0 = amdgpu::SRC_SDWA;
  raw.vsrc0 = 4;
  raw.vdst = 8;
  raw.src0_sel = amdgpu::sdwa::DWORD;
  raw.dst_sel = amdgpu::sdwa::DWORD;
  raw.dst_unused = amdgpu::sdwa::UNUSED_PAD;

  raw.encoding = cdna4::encoding::kVop1 >> 2;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  raw.op = cdna4::kVMovB32Vop1;
  EXPECT_FALSE(decode_fails(*decoder, reinterpret_cast<const uint32_t *>(&raw)));
  raw.op = cdna4::kVCvtF64I32Vop1;
  EXPECT_TRUE(decode_fails(*decoder, reinterpret_cast<const uint32_t *>(&raw)));
}

TEST(SdwaTest, Cdna4DisassemblesDestinationAndSourceAttributes) {
  cdna4::Vop1VopSdwaMachineInst raw{};
  raw.src0 = amdgpu::SRC_SDWA;
  raw.op = cdna4::kVFloorF32Vop1;
  raw.vdst = 0;
  raw.encoding = cdna4::encoding::kVop1 >> 2;
  raw.vsrc0 = 92;
  raw.dst_sel = amdgpu::sdwa::BYTE_0;
  raw.dst_unused = amdgpu::sdwa::UNUSED_PAD;
  raw.clamp = 1;
  raw.omod = 3;
  raw.src0_sel = amdgpu::sdwa::BYTE_0;

  const auto words = std::bit_cast<std::array<uint32_t, 2>>(raw);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(), "v_floor_f32_sdwa v0, v92 clamp div:2 dst_sel:BYTE_0 "
                                 "dst_unused:UNUSED_PAD src0_sel:BYTE_0");
}

TEST(SdwaTest, Cdna4DisassemblesCompareDestinationAndSourceModifiers) {
  cdna4::VopcVopSdwaSdstEncMachineInst raw{};
  raw.src0 = amdgpu::SRC_SDWA;
  raw.vsrc1 = 1;
  raw.op = cdna4::kVCmpEqF32Vopc;
  raw.encoding = cdna4::encoding::kVopc >> 2;
  raw.vsrc0 = 0;
  raw.sdst = 4;
  raw.sd = 1;
  raw.src0_sel = amdgpu::sdwa::BYTE_0;
  raw.src0_neg = 1;
  raw.src0_abs = 1;
  raw.src1_sel = amdgpu::sdwa::DWORD;
  raw.src1_neg = 1;

  const auto words = std::bit_cast<std::array<uint32_t, 2>>(raw);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(),
            "v_cmp_eq_f32_sdwa s[4:5], -|v0|, -v1 src0_sel:BYTE_0 src1_sel:DWORD");
  ASSERT_EQ(inst->num_dst_operands(), 1);
  EXPECT_EQ(inst->dst_operand(0)->to_register_ref(), (RegisterRef{RegClass::SGPR, 4, 2}));
}

TEST(SdwaTest, Cdna4DisassemblesImplicitCompareDestination) {
  cdna4::VopcVopSdwaSdstEncMachineInst raw{};
  raw.src0 = amdgpu::SRC_SDWA;
  raw.vsrc1 = 1;
  raw.op = cdna4::kVCmpEqF32Vopc;
  raw.encoding = cdna4::encoding::kVopc >> 2;
  raw.vsrc0 = 0;
  raw.sd = 0;
  raw.src0_sel = amdgpu::sdwa::DWORD;
  raw.src1_sel = amdgpu::sdwa::DWORD;

  const auto words = std::bit_cast<std::array<uint32_t, 2>>(raw);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(), "v_cmp_eq_f32_sdwa vcc, v0, v1 src0_sel:DWORD src1_sel:DWORD");
}

TEST(SdwaTest, Cdna4DisassemblesBothVop2Sources) {
  cdna4::Vop2VopSdwaMachineInst raw{};
  raw.src0 = amdgpu::SRC_SDWA;
  raw.vsrc1 = 3;
  raw.vdst = 5;
  raw.op = cdna4::kVAddF32Vop2;
  raw.encoding = cdna4::encoding::kVop2;
  raw.vsrc0 = 2;
  raw.dst_sel = amdgpu::sdwa::DWORD;
  raw.dst_unused = amdgpu::sdwa::UNUSED_PAD;
  raw.src0_sel = amdgpu::sdwa::DWORD;
  raw.src0_neg = 1;
  raw.src0_abs = 1;
  raw.src1_sel = amdgpu::sdwa::BYTE_2;
  raw.src1_neg = 1;

  const auto words = std::bit_cast<std::array<uint32_t, 2>>(raw);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(),
            "v_add_f32_sdwa v5, -|v2|, -v3 dst_sel:DWORD dst_unused:UNUSED_PAD "
            "src0_sel:DWORD src1_sel:BYTE_2");
}

TEST(SdwaTest, InvalidAttributeValuesAreExplicit) {
  EXPECT_EQ(amdgpu::sdwa::selection_name(7), "invalid(7)");
  EXPECT_EQ(amdgpu::sdwa::destination_unused_name(3), "invalid(3)");
}

TEST(DecodeSizeTest, Cdna4UsesOpcodeOperandCapabilitiesForLiteralSelectors) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);

  cdna4::SopcMachineInst set_gpr_idx{};
  set_gpr_idx.ssrc0 = 0;
  set_gpr_idx.ssrc1 = 255;
  set_gpr_idx.op = cdna4::kSSetGprIdxOnSopc;
  set_gpr_idx.encoding = cdna4::encoding::kSopc;
  const auto set_gpr_idx_words = std::bit_cast<std::array<uint32_t, 1>>(set_gpr_idx);
  std::unique_ptr<Instruction> set_gpr_idx_inst(decode_valid(*decoder, set_gpr_idx_words.data()));
  ASSERT_NE(set_gpr_idx_inst, nullptr);
  EXPECT_EQ(set_gpr_idx_inst->size(), sizeof(uint32_t));

  cdna4::Sop1MachineInst get_pc{};
  get_pc.ssrc0 = 255;
  get_pc.op = cdna4::kSGetPcB64Sop1;
  get_pc.sdst = 0;
  get_pc.encoding = cdna4::encoding::kSop1;
  const auto get_pc_words = std::bit_cast<std::array<uint32_t, 1>>(get_pc);
  std::unique_ptr<Instruction> get_pc_inst(decode_valid(*decoder, get_pc_words.data()));
  ASSERT_NE(get_pc_inst, nullptr);
  EXPECT_EQ(get_pc_inst->size(), sizeof(uint32_t));

  cdna4::Sop2MachineInst fork{};
  fork.ssrc0 = 255;
  fork.ssrc1 = 255;
  fork.op = cdna4::kSCbranchGForkSop2;
  fork.encoding = cdna4::encoding::kSop2 >> 7;
  const auto fork_words = std::bit_cast<std::array<uint32_t, 1>>(fork);
  EXPECT_TRUE(decode_fails(*decoder, fork_words.data()));

  cdna4::Vop1MachineInst v_nop{};
  v_nop.src0 = 255;
  v_nop.op = cdna4::kVNopVop1;
  v_nop.encoding = cdna4::encoding::kVop1 >> 2;
  const auto v_nop_words = std::bit_cast<std::array<uint32_t, 1>>(v_nop);
  std::unique_ptr<Instruction> v_nop_inst(decode_valid(*decoder, v_nop_words.data()));
  ASSERT_NE(v_nop_inst, nullptr);
  EXPECT_EQ(v_nop_inst->size(), sizeof(uint32_t));

  cdna4::Sop1MachineInst scalar_literal{};
  scalar_literal.ssrc0 = 255;
  scalar_literal.op = cdna4::kSNotB32Sop1;
  scalar_literal.sdst = 0;
  scalar_literal.encoding = cdna4::encoding::kSop1;
  const std::array<uint32_t, 2> scalar_literal_words = {std::bit_cast<uint32_t>(scalar_literal),
                                                        0x12345678u};
  std::unique_ptr<Instruction> scalar_literal_inst(
      decode_valid(*decoder, scalar_literal_words.data()));
  ASSERT_NE(scalar_literal_inst, nullptr);
  EXPECT_EQ(scalar_literal_inst->size(), sizeof(scalar_literal_words));
}

void cdna4_unsupported_sdwa_decode_halts_wave() {
  cdna4::Vop1VopSdwaMachineInst raw{};
  raw.src0 = amdgpu::SRC_SDWA;
  raw.op = cdna4::kVCvtF64I32Vop1;
  raw.vsrc0 = 4;
  raw.vdst = 8;
  raw.encoding = cdna4::encoding::kVop1 >> 2;
  raw.src0_sel = amdgpu::sdwa::DWORD;
  raw.dst_sel = amdgpu::sdwa::DWORD;
  raw.dst_unused = amdgpu::sdwa::UNUSED_PAD;

  static_assert(sizeof(raw) == 2 * sizeof(uint32_t));
  const auto encoded = std::bit_cast<std::array<uint32_t, 2>>(raw);
  const std::array<uint32_t, 4> words{encoded[0], encoded[1], 0, 0};

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  EXPECT_TRUE(decode_fails(*decoder, words.data()));

  amdgpu::GpuMemory mem("cdna4_unsupported_sdwa_mem");
  amdgpu::L2Cache l2("cdna4_unsupported_sdwa_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create("cdna4_unsupported_sdwa_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  for (uint32_t index = 0; index < words.size(); ++index)
    mem.write32(index * sizeof(uint32_t), words[index]);

  EXPECT_NO_THROW(static_cast<void>(cu->step()));
  EXPECT_TRUE(wf->is_halted());
}

template <typename Raw> void rdna4_vop3_dpp_marker_is_instruction_specific(uint32_t marker) {
  Raw raw{};
  raw.src0 = marker;
  raw.op = rdna4::kVAddF32Vop3;
  raw.encoding = rdna4::encoding::kVop3 >> 3;
  raw.vsrc0 = 4;
  raw.src1 = 5;
  raw.vdst = 8;

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  EXPECT_FALSE(decode_fails(*decoder, reinterpret_cast<const uint32_t *>(&raw)));
  raw.op = rdna4::kVAddF64Vop3;
  EXPECT_TRUE(decode_fails(*decoder, reinterpret_cast<const uint32_t *>(&raw)));
}

template <typename Raw> void rdna4_vop3p_dpp_marker_is_unsupported(uint32_t marker) {
  Raw raw{};
  raw.src0 = marker;
  raw.op = rdna4::kVPkAddF16Vop3p;
  raw.encoding = rdna4::encoding::kVop3p >> 1;
  raw.vsrc0 = 4;
  raw.src1 = 5;
  raw.vdst = 8;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  EXPECT_TRUE(decode_fails(*decoder, reinterpret_cast<const uint32_t *>(&raw)));
}

void rdna4_vop3_dpp_availability_is_instruction_specific() {
  rdna4_vop3_dpp_marker_is_instruction_specific<rdna4::Vop3VopDpp16MachineInst>(amdgpu::SRC_DPP);
  rdna4_vop3_dpp_marker_is_instruction_specific<rdna4::Vop3VopDpp8MachineInst>(
      amdgpu::SRC_DPP8_FI_0);
  rdna4_vop3p_dpp_marker_is_unsupported<rdna4::Vop3pVopDpp16MachineInst>(amdgpu::SRC_DPP);
  rdna4_vop3p_dpp_marker_is_unsupported<rdna4::Vop3pVopDpp8MachineInst>(amdgpu::SRC_DPP8_FI_0);
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

TEST(DppPermuteTest, CdnaVopcDppRejectsUnsupported) {
  unsupported_cdna_vopc_dpp_rejects<Cdna1DppTraits>();
  unsupported_cdna_vopc_dpp_rejects<Cdna2DppTraits>();
  unsupported_cdna_vopc_dpp_rejects<Cdna3DppTraits>();
  unsupported_cdna_vopc_dpp_rejects<Cdna4DppTraits>();
}

TEST(DppPermuteTest, Cdna4Vop1SdwaAvailabilityIsInstructionSpecific) {
  cdna4_vop1_sdwa_availability_is_instruction_specific();
}

TEST(DppPermuteTest, Cdna4UnsupportedSdwaDecodeHaltsWave) {
  cdna4_unsupported_sdwa_decode_halts_wave();
}

TEST(DppPermuteTest, Rdna4Vop3DppAvailabilityIsInstructionSpecific) {
  rdna4_vop3_dpp_availability_is_instruction_specific();
}

TEST(DppPermuteTest, RdnaGeneratedVop1DppWriteMaskHonorsBoundCtrl) {
  wave32_generated_vop1_dpp_write_mask_honors_bound_ctrl<Rdna1DppTraits>();
  wave32_generated_vop1_dpp_write_mask_honors_bound_ctrl<Rdna2DppTraits>();
  wave32_generated_vop1_dpp_write_mask_honors_bound_ctrl<Rdna3DppTraits>();
  wave32_generated_vop1_dpp_write_mask_honors_bound_ctrl<Rdna3_5DppTraits>();
  wave32_generated_vop1_dpp_write_mask_honors_bound_ctrl<Rdna4DppTraits>();
}

TEST(DppPermuteTest, Rdna4GeneratedVop1Dpp64RejectsDpp) {
  rdna4::Vop1VopDpp16MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.op = rdna4::kVCvtF64I32Vop1;
  raw.encoding = 63;
  raw.dpp_ctrl = amdgpu::dpp::ROW_SELECT_BASE;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  EXPECT_TRUE(decoder->decode(reinterpret_cast<const uint32_t *>(&raw)).failed());
}

TEST(DppPermuteTest, Rdna4GeneratedDppOwnershipFollowsLogicalSource) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);

  rdna4::Vop1VopDpp16MachineInst vop1{};
  vop1.src0 = amdgpu::SRC_DPP;
  vop1.op = rdna4::kVNopVop1;
  vop1.encoding = 63;
  EXPECT_TRUE(decoder->decode(reinterpret_cast<const uint32_t *>(&vop1)).succeeded());

  rdna4::Vop2VopDpp16MachineInst vop2{};
  vop2.src0 = amdgpu::SRC_DPP;
  vop2.op = rdna4::kVFmamkF32Vop2;
  EXPECT_TRUE(decoder->decode(reinterpret_cast<const uint32_t *>(&vop2)).failed());

  rdna4::Vop3VopDpp16MachineInst vop3{};
  vop3.src0 = amdgpu::SRC_DPP;
  vop3.op = rdna4::kVMulLoU32Vop3;
  vop3.encoding = 53;
  EXPECT_TRUE(decoder->decode(reinterpret_cast<const uint32_t *>(&vop3)).failed());

  rdna4::VopcVopDpp16MachineInst vopc{};
  vopc.src0 = amdgpu::SRC_DPP;
  vopc.op = rdna4::kVCmpEqF64Vopc;
  vopc.encoding = 62;
  EXPECT_TRUE(decoder->decode(reinterpret_cast<const uint32_t *>(&vopc)).failed());

  rdna4::Vop3pVopDpp16MachineInst vop3p{};
  vop3p.src0 = amdgpu::SRC_DPP;
  vop3p.op = rdna4::kVPkAddU16Vop3p;
  vop3p.encoding = 204;
  EXPECT_TRUE(decoder->decode(reinterpret_cast<const uint32_t *>(&vop3p)).failed());
}

TEST(DppPermuteTest, Rdna4GeneratedAllowsIndependentDppOpsel) {
  auto make_vop3 = [](uint32_t opsel) {
    rdna4::Vop3VopDpp16MachineInst raw{};
    raw.src0 = amdgpu::SRC_DPP;
    raw.op = rdna4::kVAddF16Vop3;
    raw.encoding = 53;
    raw.dpp_ctrl = amdgpu::dpp::ROW_SELECT_BASE;
    raw.opsel = opsel;
    return raw;
  };
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  auto vop3_low = make_vop3(0x0);
  auto vop3_high = make_vop3(0xB); // dst/src0/src1 all select high halves.
  auto vop3_mixed = make_vop3(0x9);
  EXPECT_TRUE(decoder->decode(reinterpret_cast<const uint32_t *>(&vop3_low)).succeeded());
  EXPECT_TRUE(decoder->decode(reinterpret_cast<const uint32_t *>(&vop3_high)).succeeded());
  EXPECT_TRUE(decoder->decode(reinterpret_cast<const uint32_t *>(&vop3_mixed)).succeeded());

  auto make_vop3p = [](uint32_t opsel, uint32_t opsel_hi) {
    rdna4::Vop3pVopDpp16MachineInst raw{};
    raw.src0 = amdgpu::SRC_DPP;
    raw.op = rdna4::kVFmaMixF32Vop3p;
    raw.encoding = 204;
    raw.dpp_ctrl = amdgpu::dpp::ROW_SELECT_BASE;
    raw.opsel = opsel;
    raw.opsel_hi = opsel_hi & 0x3;
    raw.opsel_hi_2 = opsel_hi >> 2;
    return raw;
  };
  auto vop3p_aligned = make_vop3p(0x0, 0x7);
  auto vop3p_mixed_low = make_vop3p(0x1, 0x7);
  auto vop3p_mixed_high = make_vop3p(0x0, 0x3);
  EXPECT_TRUE(decoder->decode(reinterpret_cast<const uint32_t *>(&vop3p_aligned)).succeeded());
  EXPECT_TRUE(decoder->decode(reinterpret_cast<const uint32_t *>(&vop3p_mixed_low)).succeeded());
  EXPECT_TRUE(decoder->decode(reinterpret_cast<const uint32_t *>(&vop3p_mixed_high)).succeeded());
}

TEST(DppPermuteTest, Rdna4GeneratedVop1F64PreservesInactiveDestination) {
  rdna4_generated_vop1_64_preserves_inactive_destination();
}

TEST(DppPermuteTest, Cdna4GeneratedVop1Dpp64InputPermutesBothWords) {
  generated_vop1_dpp64_input_permutes_both_words<Cdna4DppTraits>();
}

TEST(DppPermuteTest, Cdna3And4GeneratedVop1Dpp64AcceptOnlyRowSelectControls) {
  auto check_cdna3 = [](uint32_t dpp_ctrl, bool legal) {
    cdna3::Vop1VopDppMachineInst raw{};
    raw.src0 = amdgpu::SRC_DPP;
    raw.op = cdna3::kVMovB64Vop1;
    raw.encoding = 63;
    raw.dpp_ctrl = dpp_ctrl;
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
    ASSERT_NE(decoder, nullptr);
    EXPECT_EQ(decoder->decode(reinterpret_cast<const uint32_t *>(&raw)).succeeded(), legal);
  };
  auto check_cdna4 = [](uint32_t dpp_ctrl, bool legal) {
    cdna4::Vop1VopDppMachineInst raw{};
    raw.src0 = amdgpu::SRC_DPP;
    raw.op = cdna4::kVMovB64Vop1;
    raw.encoding = 63;
    raw.dpp_ctrl = dpp_ctrl;
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_NE(decoder, nullptr);
    EXPECT_EQ(decoder->decode(reinterpret_cast<const uint32_t *>(&raw)).succeeded(), legal);
  };

  check_cdna3(amdgpu::dpp::ROW_SELECT_BASE + 7, true);
  check_cdna3(0x1Bu, false); // DPP_QUAD_PERM
  check_cdna3(amdgpu::dpp::WF_SHL1, false);
  check_cdna4(amdgpu::dpp::ROW_SELECT_BASE + 7, true);
  check_cdna4(0x1Bu, false); // DPP_QUAD_PERM
  check_cdna4(amdgpu::dpp::WF_SHL1, false);
}

template <typename RawInst>
void expect_legacy_dpp_opcode_result(rj_code_arch_t arch, uint32_t opcode, uint32_t encoding,
                                     bool rejected = true) {
  std::array<uint32_t, 3> words{};
  auto *raw = reinterpret_cast<RawInst *>(words.data());
  raw->src0 = amdgpu::SRC_DPP;
  raw->op = opcode;
  raw->encoding = encoding;
  auto decoder = Decoder::create(arch);
  ASSERT_NE(decoder, nullptr);
  EXPECT_EQ(decoder->decode(words.data()).failed(), rejected);
}

TEST(DppPermuteTest, LegacyGeneratedDppOwnershipFollowsLogicalSource) {
  expect_legacy_dpp_opcode_result<cdna1::Vop1VopDppMachineInst>(ROCJITSU_CODE_ARCH_CDNA1,
                                                                cdna1::kVCvtI32F64Vop1, 63);
  expect_legacy_dpp_opcode_result<cdna2::Vop1VopDppMachineInst>(ROCJITSU_CODE_ARCH_CDNA2,
                                                                cdna2::kVReadfirstlaneB32Vop1, 63);
  expect_legacy_dpp_opcode_result<cdna3::Vop1VopDppMachineInst>(ROCJITSU_CODE_ARCH_CDNA3,
                                                                cdna3::kVSwapB32Vop1, 63);
  expect_legacy_dpp_opcode_result<cdna3::Vop1VopDppMachineInst>(ROCJITSU_CODE_ARCH_CDNA3,
                                                                cdna3::kVCvtI32F64Vop1, 63);
  expect_legacy_dpp_opcode_result<cdna4::Vop1VopDppMachineInst>(ROCJITSU_CODE_ARCH_CDNA4,
                                                                cdna4::kVClrexcpVop1, 63, false);
  expect_legacy_dpp_opcode_result<cdna4::Vop1VopDppMachineInst>(ROCJITSU_CODE_ARCH_CDNA4,
                                                                cdna4::kVCvtI32F64Vop1, 63);
  expect_legacy_dpp_opcode_result<rdna1::Vop2VopDpp16MachineInst>(ROCJITSU_CODE_ARCH_RDNA1,
                                                                  rdna1::kVMadmkF32Vop2, 0);
  expect_legacy_dpp_opcode_result<rdna2::VopcMachineInst>(ROCJITSU_CODE_ARCH_RDNA2,
                                                          rdna2::kVCmpEqF64Vopc, 62);
}

TEST(DppPermuteTest, Gfx1250GeneratedVop1DppWriteMaskHonorsBoundCtrl) {
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  wave32_generated_vop1_dpp_write_mask_honors_bound_ctrl<Gfx1250DppTraits>();
}

TEST(DppPermuteTest, Gfx1250GeneratedVop1Dpp64PreservesMaskedDestination) {
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  generated_vop1_dpp64_preserves_masked_destination<Gfx1250DppTraits>();
}

TEST(DppPermuteTest, Gfx1250GeneratedVop1Dpp64InputPermutesBothWords) {
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  generated_vop1_dpp64_input_permutes_both_words<Gfx1250DppTraits>();
}
TEST(DppPermuteTest, RdnaGeneratedVop1Dpp16FetchInactiveUsesFi) {
  wave32_generated_vop1_dpp16_fetch_inactive_uses_fi<Rdna1DppTraits>();
  wave32_generated_vop1_dpp16_fetch_inactive_uses_fi<Rdna2DppTraits>();
  wave32_generated_vop1_dpp16_fetch_inactive_uses_fi<Rdna3DppTraits>();
  wave32_generated_vop1_dpp16_fetch_inactive_uses_fi<Rdna3_5DppTraits>();
  wave32_generated_vop1_dpp16_fetch_inactive_uses_fi<Rdna4DppTraits>();
}

TEST(DppPermuteTest, Gfx1250GeneratedVop1Dpp16FetchInactiveUsesFi) {
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  wave32_generated_vop1_dpp16_fetch_inactive_uses_fi<Gfx1250DppTraits>();
}

TEST(DppPermuteTest, RdnaGeneratedVop1Dpp16InactiveSourceUsesArchitecturePolicy) {
  wave32_generated_vop1_dpp16_fi_zero_obeys_arch_bound_ctrl<Rdna1DppTraits>();
  wave32_generated_vop1_dpp16_fi_zero_obeys_arch_bound_ctrl<Rdna2DppTraits>();
  wave32_generated_vop1_dpp16_fi_zero_obeys_arch_bound_ctrl<Rdna3DppTraits>();
  wave32_generated_vop1_dpp16_fi_zero_obeys_arch_bound_ctrl<Rdna3_5DppTraits>();
  wave32_generated_vop1_dpp16_fi_zero_obeys_arch_bound_ctrl<Rdna4DppTraits>();
}

TEST(DppPermuteTest, Gfx1250GeneratedVop1Dpp16InactiveSourceUsesArchitecturePolicy) {
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  wave32_generated_vop1_dpp16_fi_zero_obeys_arch_bound_ctrl<Gfx1250DppTraits>();
}

TEST(DppPermuteTest, Rdna4GeneratedVop1Dpp16Wave64FetchInactiveUsesUpperExecBit) {
  rdna4_wave64_generated_vop1_dpp16_fetch_inactive_uses_upper_exec_bit();
}

TEST(DppPermuteTest, Rdna4GeneratedVop1Dpp8Wave64FetchInactiveUsesUpperExecBit) {
  rdna4_wave64_generated_vop1_dpp8_fetch_inactive_uses_upper_exec_bit();
}

TEST(DppPermuteTest, Rdna4GeneratedDppComparesMaskWave64UpperResults) {
  rdna4_wave64_generated_dpp_compares_mask_upper_results();
}

TEST(DppPermuteTest, Rdna4GeneratedVop2DppInactiveSourcePreservesDestination) {
  rdna4_generated_vop2_dpp_inactive_source_preserves_destination();
}

TEST(DppPermuteTest, Rdna4GeneratedVop2DppCarryUsesSourceWriteMaskOnly) {
  generated_vop2_dpp_carry_uses_source_write_mask_only<Rdna4DppTraits>();
}

TEST(DppPermuteTest, Gfx1250GeneratedVop2DppCarryUsesSourceWriteMaskOnly) {
  generated_vop2_dpp_carry_uses_source_write_mask_only<Gfx1250DppTraits>();
}

TEST(DppPermuteTest, Cdna4GeneratedVop2DppCarryUsesSourceWriteMaskOnly) {
  generated_vop2_dpp_carry_uses_source_write_mask_only<Cdna4DppTraits>();
}

TEST(DppPermuteTest, RdnaGeneratedVop1Dpp8FetchInactiveUsesFi) {
  wave32_generated_vop1_dpp8_fetch_inactive_uses_fi<Rdna1DppTraits>();
  wave32_generated_vop1_dpp8_fetch_inactive_uses_fi<Rdna2DppTraits>();
  wave32_generated_vop1_dpp8_fetch_inactive_uses_fi<Rdna3DppTraits>();
  wave32_generated_vop1_dpp8_fetch_inactive_uses_fi<Rdna3_5DppTraits>();
  wave32_generated_vop1_dpp8_fetch_inactive_uses_fi<Rdna4DppTraits>();
}

TEST(DppPermuteTest, Gfx1250GeneratedVop1Dpp8FetchInactiveUsesFi) {
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  wave32_generated_vop1_dpp8_fetch_inactive_uses_fi<Gfx1250DppTraits>();
}

TEST(DppPermuteTest, RdnaGeneratedVopcDppInvalidSourceBc0ZerosResult) {
  wave32_generated_vopc_dpp_invalid_source_bc0_zeros_result<Rdna3DppTraits>();
  wave32_generated_vopc_dpp_invalid_source_bc0_zeros_result<Rdna3_5DppTraits>();
  wave32_generated_vopc_dpp_invalid_source_bc0_zeros_result<Rdna4DppTraits>();
}

TEST(DppPermuteTest, Gfx1250GeneratedVopcDppInvalidSourceBc0ZerosResult) {
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  wave32_generated_vopc_dpp_invalid_source_bc0_zeros_result<Gfx1250DppTraits>();
}

TEST(DppPermuteTest, RdnaGeneratedVopcDppZerosMaskedCompareBits) {
  wave32_generated_vopc_dpp_zeros_masked_compare_bits<Rdna3DppTraits>();
  wave32_generated_vopc_dpp_zeros_masked_compare_bits<Rdna3_5DppTraits>();
  wave32_generated_vopc_dpp_zeros_masked_compare_bits<Rdna4DppTraits>();
}

TEST(DppPermuteTest, Gfx1250GeneratedVopcDppZerosMaskedCompareBits) {
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  wave32_generated_vopc_dpp_zeros_masked_compare_bits<Gfx1250DppTraits>();
}

TEST(DppPermuteTest, RdnaGeneratedVop3DppMasksActualCompareDestination) {
  wave32_generated_vop3_dpp_masks_actual_compare_destination<Rdna3DppTraits>();
  wave32_generated_vop3_dpp_masks_actual_compare_destination<Rdna3_5DppTraits>();
  wave32_generated_vop3_dpp_masks_actual_compare_destination<Rdna4DppTraits>();
}

TEST(DppPermuteTest, Gfx1250GeneratedVop3DppMasksActualCompareDestination) {
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  wave32_generated_vop3_dpp_masks_actual_compare_destination<Gfx1250DppTraits>();
}

TEST(DppPermuteTest, RdnaGeneratedLegalVop3pDppMasksVgprDestination) {
  wave32_generated_legal_vop3p_dpp_masks_vgpr_destination<Rdna3DppTraits>();
  wave32_generated_legal_vop3p_dpp_masks_vgpr_destination<Rdna3_5DppTraits>();
  wave32_generated_legal_vop3p_dpp_masks_vgpr_destination<Rdna4DppTraits>();
}

TEST(DppPermuteTest, Gfx1250GeneratedLegalVop3pDppMasksVgprDestination) {
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  wave32_generated_legal_vop3p_dpp_masks_vgpr_destination<Gfx1250DppTraits>();
}

TEST(DppPermuteTest, RdnaGeneratedVop3SdstDppMasksVectorAndScalarDestinations) {
  wave32_generated_vop3_sdst_dpp_masks_vector_and_scalar_destinations<Rdna3DppTraits>();
  wave32_generated_vop3_sdst_dpp_masks_vector_and_scalar_destinations<Rdna3_5DppTraits>();
  wave32_generated_vop3_sdst_dpp_masks_vector_and_scalar_destinations<Rdna4DppTraits>();
}

TEST(DppPermuteTest, Gfx1250GeneratedVop3SdstDppMasksVectorAndScalarDestinations) {
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  wave32_generated_vop3_sdst_dpp_masks_vector_and_scalar_destinations<Gfx1250DppTraits>();
}

TEST(DppPermuteTest, GeneratedDppInstructionReuseReadsCurrentSource) {
  generated_dpp_instruction_reuse_reads_current_source<Cdna1DppTraits>();
  generated_dpp_instruction_reuse_reads_current_source<Cdna2DppTraits>();
  generated_dpp_instruction_reuse_reads_current_source<Cdna3DppTraits>();
  generated_dpp_instruction_reuse_reads_current_source<Cdna4DppTraits>();
  generated_dpp_instruction_reuse_reads_current_source<Rdna1DppTraits>();
  generated_dpp_instruction_reuse_reads_current_source<Rdna2DppTraits>();
  generated_dpp_instruction_reuse_reads_current_source<Rdna3DppTraits>();
  generated_dpp_instruction_reuse_reads_current_source<Rdna3_5DppTraits>();
  generated_dpp_instruction_reuse_reads_current_source<Rdna4DppTraits>();
  {
    ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
    generated_dpp_instruction_reuse_reads_current_source<Gfx1250DppTraits>();
  }
}

TEST(DppPermuteTest, RdnaGeneratedBc1CompareFamiliesClearStaleBits) {
  wave32_generated_bc1_compare_families_clear_stale_bits<Rdna3DppTraits>();
  wave32_generated_bc1_compare_families_clear_stale_bits<Rdna3_5DppTraits>();
  wave32_generated_bc1_compare_families_clear_stale_bits<Rdna4DppTraits>();
}

TEST(DppPermuteTest, Gfx1250GeneratedBc1CompareFamiliesClearStaleBits) {
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  wave32_generated_bc1_compare_families_clear_stale_bits<Gfx1250DppTraits>();
}

TEST(DppPermuteTest, RdnaGeneratedVcmpxDppZerosInvalidSourcesPreservesExecHi) {
  wave32_generated_vcmpx_dpp_zeros_invalid_sources_preserves_exec_hi<Rdna3DppTraits>();
  wave32_generated_vcmpx_dpp_zeros_invalid_sources_preserves_exec_hi<Rdna3_5DppTraits>();
  wave32_generated_vcmpx_dpp_zeros_invalid_sources_preserves_exec_hi<Rdna4DppTraits>();
}

TEST(DppPermuteTest, Gfx1250GeneratedVcmpxDppZerosInvalidSourcesPreservesExecHi) {
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  wave32_generated_vcmpx_dpp_zeros_invalid_sources_preserves_exec_hi<Gfx1250DppTraits>();
}

TEST(ExecMaskTest, RdnaGeneratedVcmpxWave32PreservesExecHi) {
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  wave32_generated_vcmpx_preserves_exec_hi<Rdna1DppTraits>();
  wave32_generated_vcmpx_preserves_exec_hi<Rdna2DppTraits>();
  wave32_generated_vcmpx_preserves_exec_hi<Rdna3DppTraits>();
  wave32_generated_vcmpx_preserves_exec_hi<Rdna3_5DppTraits>();
  wave32_generated_vcmpx_preserves_exec_hi<Rdna4DppTraits>();
  wave32_generated_vcmpx_preserves_exec_hi<Gfx1250DppTraits>();
}

TEST(DppPermuteTest, Rdna1VopcDppRejectsUnsupported) {
  unsupported_rdna_vopc_dpp_rejects<Rdna1DppTraits>();
}

TEST(DppPermuteTest, Rdna2VopcDppRejectsUnsupported) {
  unsupported_rdna_vopc_dpp_rejects<Rdna2DppTraits>();
}

TEST(SdwaTest, RdnaWave32ExplicitCompareDoesNotClobberAdjacentSgpr) {
  wave32_sdwa_explicit_compare_writes_only_destination_low<Rdna1DppTraits>();
  wave32_sdwa_explicit_compare_writes_only_destination_low<Rdna2DppTraits>();
}

// ---------------------------------------------------------------------------
// SDWA tests
// ---------------------------------------------------------------------------

// These helper tests pin byte placement independently of instruction decode.
// End-to-end callback tests can then distinguish a selector/merge bug here
// from a generated wrapper passing the wrong read or write mask.
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

  // Sign-extension fills bytes above the selected byte/word and zeroes bytes
  // below it.
  merged = sdwa_dst_merge(0x80, 0xAABBCCDD, BYTE_1, UNUSED_SEXT);
  EXPECT_EQ(merged, 0xFFFF8000u);
  merged = sdwa_dst_merge(0x7F, 0xAABBCCDD, BYTE_3, UNUSED_SEXT);
  EXPECT_EQ(merged, 0x7F000000u);

  // Word destinations exercise both preservation and sign-extension.
  merged = sdwa_dst_merge(0x12345678, 0xAABBCCDD, WORD_0, UNUSED_PRESERVE);
  EXPECT_EQ(merged, 0xAABB5678u);
  merged = sdwa_dst_merge(0x00008000, 0xAABBCCDD, WORD_1, UNUSED_SEXT);
  EXPECT_EQ(merged, 0x80000000u);

  // The source selector and preserve destination mask use the same byte
  // windows; pad/sext destinations are full-dword writes.
  EXPECT_EQ(sdwa_src_byte_mask(BYTE_0), 0b0001);
  EXPECT_EQ(sdwa_src_byte_mask(BYTE_3), 0b1000);
  EXPECT_EQ(sdwa_src_byte_mask(WORD_0), 0b0011);
  EXPECT_EQ(sdwa_src_byte_mask(WORD_1), 0b1100);
  EXPECT_EQ(sdwa_src_byte_mask(DWORD), 0b1111);
  EXPECT_EQ(sdwa_dst_byte_mask(BYTE_1, UNUSED_PRESERVE), 0b0010);
  EXPECT_EQ(sdwa_dst_byte_mask(WORD_1, UNUSED_PRESERVE), 0b1100);
  EXPECT_EQ(sdwa_dst_byte_mask(BYTE_1, UNUSED_PAD), 0b1111);
  EXPECT_EQ(sdwa_dst_byte_mask(BYTE_1, UNUSED_SEXT), 0b1111);
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

  // Write the same per-lane private byte offset into VGPR[0] for lanes 0 and 1.
  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase, 0, 0x100); // lane 0: offset 0x100
  cu->write_vgpr(vbase, 1, 0x100); // lane 1: offset 0x100

  // Set EXEC so lanes 0 and 1 are active.
  wf->set_exec(0x3ULL);

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

  EXPECT_EQ(d.lane_mask, 0x3ULL);
  // Scratch is stored in the hardware dword-interleaved ("swizzled") layout that
  // rocm-dbgapi reads (rocdbgapi memory.cpp private_swizzled):
  //   addr = scratch_base + (off/4)*lane_count*4 + lane*4 + off%4
  // with off = VGPR + offset = 0x100 + 0x10 = 0x110 and lane_count = 64:
  //   (0x110/4)*64*4 = 0x44 * 256 = 0x4400; lane 0 -> +0, lane 1 -> +4.
  constexpr uint64_t kSwizzledBase = SCRATCH_BASE + 0x4400;
  EXPECT_EQ(d.per_lane_addr[0], kSwizzledBase + 0);
  EXPECT_EQ(d.per_lane_addr[1], kSwizzledBase + 4);
  EXPECT_TRUE(d.scratch_swizzle);
  EXPECT_EQ(d.scratch_addr_stride, 64u * sizeof(uint32_t));
}

TEST(ScratchAddrCalcTest, FlatScratchSignExtendsScalarStackOffset) {
  amdgpu::GpuMemory mem("signed_scratch_mem");
  amdgpu::L2Cache l2("signed_scratch_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 104;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("signed_scratch_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 104, 16);
  ASSERT_NE(wf, nullptr);
  constexpr uint64_t kScratchBase = 0x1'0000'0000ULL;
  wf->set_scratch_base(kScratchBase);
  wf->set_exec(1);

  const uint32_t vbase = wf->vgpr_alloc().base;
  const uint32_t sbase = wf->sgpr_alloc().base;
  cu->write_vgpr(vbase, 0, 0x40);
  cu->write_sgpr(sbase + 32, static_cast<uint32_t>(-0x20));

  cdna4::FlatScratchMachineInst inst{};
  inst.seg = 1;
  inst.sve = 1;
  inst.saddr = 32;
  inst.addr = 0;
  inst.offset = 0;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  amdgpu::addr_calc::flat_calculate_addresses(inst, *wf, d);

  // Private offset is 0x40 + (-0x20) = 0x20. In the wave64 dword-swizzled
  // layout lane 0 therefore starts 8 rows (8 * 64 * 4 bytes) above the base.
  EXPECT_EQ(d.per_lane_addr[0], kScratchBase + 0x800);
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
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
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

// SMEM SBASE, SOFFSET (SOE=1) and OFFSET[6:0] (IMM=0) name FLAT_SCRATCH / VCC / M0 through the
// scalar-source selector space, never the raw SGPR slice; byte components drop their two LSBs;
// s_scratch_* scales the register offset by 64.
TEST(CdnaAddrCalcTest, SmemOffsetSelectorsAlignmentAndScratchScale) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    amdgpu::GpuMemory mem("cdna_smem_addr_mem");
    amdgpu::L2Cache l2("cdna_smem_addr_l2");
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 128;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("cdna_smem_addr_cu", cfg, &mem, &l2);
    ASSERT_NE(cu, nullptr);
    auto *wf = cu->dispatch_wf(0, 0, 128, 16);
    ASSERT_NE(wf, nullptr);

    constexpr uint64_t kBase = 0x10'0001'0000ULL;
    constexpr uint64_t kVcc = 0x20'0000'1234ULL;
    constexpr uint64_t kScratchBase = 0x30'0003'0000ULL;
    const uint32_t sbase = wf->sgpr_alloc().base;
    cu->write_sgpr(sbase, static_cast<uint32_t>(kBase));
    cu->write_sgpr(sbase + 1, static_cast<uint32_t>(kBase >> 32));
    cu->write_sgpr(sbase + 8, 63);
    for (uint32_t sel :
         {cdna3::OPR_SMEM_OFFSET_FLAT_SCRATCH_LO, cdna3::OPR_SMEM_OFFSET_FLAT_SCRATCH_HI,
          cdna3::OPR_SMEM_OFFSET_VCC_LO, cdna3::OPR_SMEM_OFFSET_VCC_HI,
          cdna3::OPR_SMEM_OFFSET_M0}) // decoys in the aliasing SGPR slots
      cu->write_sgpr(sbase + sel, 0x777);
    wf->set_m0(0x40);
    wf->set_vcc_raw(kVcc);
    wf->set_scratch_base(kScratchBase);

    amdgpu::SmemMachineInst inst{}; // sbase = s[0:1]
    inst.op = cdna3::kSLoadDwordSmem;
    auto smem_addr = [&] {
      return arch == ROCJITSU_CODE_ARCH_CDNA3 ? cdna3::smem_calculate_address(inst, *wf)
                                              : cdna4::smem_calculate_address(inst, *wf);
    };
    inst.offset = cdna3::OPR_SMEM_OFFSET_M0; // IMM=0
    EXPECT_EQ(smem_addr(), kBase + 0x40) << arch;
    inst.soffset_en = 1;
    inst.offset = 0;
    inst.soffset = cdna3::OPR_SMEM_OFFSET_M0; // SOE=1
    EXPECT_EQ(smem_addr(), kBase + 0x40) << arch;
    inst.soffset = cdna3::OPR_SMEM_OFFSET_VCC_LO;
    EXPECT_EQ(smem_addr(), kBase + 0x1234) << arch;
    inst.soffset = 8; // SOE=1: 63 -> 60
    EXPECT_EQ(smem_addr(), kBase + 60) << arch;
    inst.imm = 1;
    inst.offset = 0x9; // IMM=1 SOE=1: 9 -> 8
    EXPECT_EQ(smem_addr(), kBase + 8 + 60) << arch;
    inst.sbase = cdna3::OPR_SMEM_OFFSET_VCC_LO / 2; // VCC pair, not s[106:107]
    EXPECT_EQ(smem_addr(), kVcc + 8 + 60) << arch;

    inst = {};
    inst.op = cdna3::kSScratchLoadDwordSmem;
    inst.offset = 8; // IMM=0: s8 * 64
    EXPECT_EQ(smem_addr(), kBase + 63 * 64) << arch;
    inst.soffset_en = 1;
    inst.soffset = 8;
    inst.imm = 1;
    inst.offset = 0x10; // IMM=1 SOE=1: 0x10 + s8 * 64
    EXPECT_EQ(smem_addr(), kBase + 0x10 + 63 * 64) << arch;
    inst.sbase = cdna3::OPR_SMEM_OFFSET_FLAT_SCRATCH_LO / 2; // FLAT_SCRATCH pair, not s[102:103]
    EXPECT_EQ(smem_addr(), kScratchBase + 0x10 + 63 * 64) << arch;
  }
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
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  amdgpu::GpuMemory mem("gfx1250_flat_private_mem");
  amdgpu::L2Cache l2("gfx1250_flat_private_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
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

  cdna5::Operand flat_scratch_base(
      64, cdna5::OperandType::OPR_SRC,
      static_cast<int>(cdna5::OpSelSrc::OPR_SRC_SRC_FLAT_SCRATCH_BASE_LO));
  ASSERT_EQ(amdgpu::RegisterAccess(*wf).read_scalar64(flat_scratch_base), kScratchBase);

  const uint64_t private_offsets[] = {0x10, 0x14, kPrivateSegmentSize + 0x20};
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < 3; ++lane) {
    uint64_t private_offset = private_offsets[lane];
    uint64_t flat_private_addr =
        kScratchBase + (static_cast<uint64_t>(lane) << 52) + private_offset;
    cu->write_vgpr(vbase, lane, static_cast<uint32_t>(flat_private_addr));
    cu->write_vgpr(vbase + 1, lane, static_cast<uint32_t>(flat_private_addr >> 32));
  }

  cdna5::VflatMachineInst inst{};
  inst.saddr = cdna5::OPR_SREG_NULL;
  inst.vaddr = 0;
  inst.ioffset = 4;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  cdna5::flat_calculate_addresses(inst, *wf, d);

  for (uint32_t lane = 0; lane < 3; ++lane) {
    uint64_t private_offset = private_offsets[lane] + inst.ioffset;
    uint64_t expected = kScratchBase + (private_offset / sizeof(uint32_t)) * 32 * sizeof(uint32_t) +
                        static_cast<uint64_t>(lane) * sizeof(uint32_t) +
                        private_offset % sizeof(uint32_t);
    EXPECT_EQ(d.per_lane_addr[lane], expected) << "lane " << lane;
  }
  EXPECT_TRUE(d.scratch_swizzle);
  EXPECT_EQ(d.scratch_lane_mask, 0x7u);
  EXPECT_EQ(d.scratch_addr_stride, 32u * sizeof(uint32_t));
}

TEST(Gfx1250AddrCalcTest, ScratchUsesDwordInterleavedWave32Layout) {
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  amdgpu::GpuMemory mem("gfx1250_scratch_mem");
  amdgpu::L2Cache l2("gfx1250_scratch_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_scratch_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x3);
  constexpr uint64_t kScratchBase = 0x0002'0000'0000'0000ULL;
  wf->set_scratch_base(kScratchBase);
  wf->set_scratch_lane_size(0x100);

  const uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase, 0, 0x20);
  cu->write_vgpr(vbase, 1, 0x30);

  cdna5::VscratchMachineInst inst{};
  inst.saddr = cdna5::OPR_SREG_NULL;
  inst.sve = 1;
  inst.vaddr = 0;
  inst.ioffset = 4;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  cdna5::flat_calculate_addresses(inst, *wf, d);

  // private offsets 0x24 and 0x34 are dword-swizzled across 32 lanes.
  EXPECT_EQ(d.per_lane_addr[0], kScratchBase + 0x480);
  EXPECT_EQ(d.per_lane_addr[1], kScratchBase + 0x684);
  EXPECT_TRUE(d.scratch_swizzle);
  EXPECT_EQ(d.scratch_lane_mask, 0x3u);
  EXPECT_EQ(d.scratch_addr_stride, 32u * sizeof(uint32_t));
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

std::array<uint32_t, 4> encode_gfx1250_buffer_resource(uint64_t base, uint64_t num_records,
                                                       uint32_t raw_stride = 0,
                                                       uint32_t stride_scale_encoding = 0,
                                                       bool swizzle = false,
                                                       bool oob_select = false, uint32_t type = 0) {
  return {
      static_cast<uint32_t>(base),
      static_cast<uint32_t>((base >> 32) & 0x01FF'FFFFu) |
          static_cast<uint32_t>((num_records & 0x7Fu) << 25),
      static_cast<uint32_t>(num_records >> 7),
      static_cast<uint32_t>((num_records >> 39) & 0x3Fu) | ((raw_stride & 0x3FFFu) << 12) |
          ((stride_scale_encoding & 0x3u) << 26) | (static_cast<uint32_t>(swizzle) << 28) |
          (static_cast<uint32_t>(oob_select) << 29) | ((type & 0x3u) << 30),
  };
}

void write_gfx1250_buffer_resource(amdgpu::ComputeUnitCore &cu, amdgpu::Wavefront &wf,
                                   uint32_t first_sgpr, const std::array<uint32_t, 4> &resource) {
  for (uint32_t i = 0; i < resource.size(); ++i)
    cu.write_sgpr(wf.sgpr_alloc().base + first_sgpr + i, resource[i]);
}

void expect_element_lane_masks(const amdgpu::ElementLaneMasks &masks,
                               std::initializer_list<uint64_t> expected) {
  const auto actual = masks.view();
  ASSERT_EQ(actual.size(), expected.size());
  EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected.begin(), expected.end()));
}

TEST(AmdgpuElementLaneMasksTest, UsesInlineWidthAndPreservesLargerFallback) {
  amdgpu::ElementLaneMasks masks;
  EXPECT_TRUE(masks.empty());
  EXPECT_EQ(amdgpu::ElementLaneMasks::kInlineCapacity, 4u);

  masks.assign(amdgpu::ElementLaneMasks::kInlineCapacity, 0x3u);
  masks[2] = 0x1u;
  expect_element_lane_masks(masks, {0x3u, 0x3u, 0x1u, 0x3u});

  masks.assign(32, 0x5u);
  EXPECT_EQ(masks.size(), 32u);
  masks[31] = 0x7u;
  const auto overflow_view = masks.view();
  ASSERT_EQ(overflow_view.size(), 32u);
  EXPECT_TRUE(std::all_of(overflow_view.begin(), overflow_view.end() - 1,
                          [](uint64_t mask) { return mask == 0x5u; }));
  EXPECT_EQ(overflow_view.back(), 0x7u);

  masks.assign(amdgpu::ElementLaneMasks::kInlineCapacity, 0x9u);
  expect_element_lane_masks(masks, {0x9u, 0x9u, 0x9u, 0x9u});

  masks.clear();
  EXPECT_TRUE(masks.empty());
}

TEST(Gfx1250AddrCalcTest, DecodesBufferResourceFields) {
  constexpr uint64_t kBase = 0x0101'2345'6789'ABCDULL;
  constexpr uint64_t kNumRecords = 0x0123'4567'89ABULL;
  constexpr uint32_t kRawStride = 0x1234;
  auto words = encode_gfx1250_buffer_resource(kBase, kNumRecords, kRawStride,
                                              /*stride_scale_encoding=*/3, /*swizzle=*/true,
                                              /*oob_select=*/true, /*type=*/2);

  cdna5::BufferResource resource =
      cdna5::decode_buffer_resource(words[0], words[1], words[2], words[3]);
  EXPECT_EQ(resource.base_address, kBase);
  EXPECT_EQ(resource.num_records, kNumRecords);
  EXPECT_EQ(resource.raw_stride, kRawStride);
  EXPECT_EQ(resource.stride_scale_encoding, 3);
  EXPECT_EQ(resource.stride, kRawStride * 32);
  EXPECT_TRUE(resource.swizzle_enabled);
  EXPECT_TRUE(resource.oob_select);
  EXPECT_EQ(resource.type, 2);
}

TEST(Gfx1250AddrCalcTest, NonBufferResourceTypesSuppressAccess) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_type_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_type_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_type_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x3u);

  cdna5::VbufferMachineInst inst{};
  inst.rsrc = 40;
  inst.soffset = cdna5::OPR_SREG_NULL;
  for (uint32_t type = 1; type <= 3; ++type) {
    write_gfx1250_buffer_resource(
        *cu, *wf, 40,
        encode_gfx1250_buffer_resource(/*base=*/0x4000, /*num_records=*/64,
                                       /*raw_stride=*/16, /*stride_scale_encoding=*/0,
                                       /*swizzle=*/false, /*oob_select=*/false, type));
    amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
    state.elem_size = 4;
    state.num_elems = 1;
    cdna5::mubuf_calculate_addresses(inst, *wf, state);
    EXPECT_EQ(state.exec_mask, 0u);
    EXPECT_EQ(state.lane_mask, 0u);
    EXPECT_TRUE(state.element_lane_masks.empty());
  }
}

TEST(Gfx1250AddrCalcTest, VbufferUses45BitBufferOffsetAnd57BitBase) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_wrap_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_wrap_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_wrap_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1ULL);

  constexpr uint64_t kBase = 0x0100'0002'0000'1000ULL;
  constexpr uint64_t kOffset = uint64_t{1} << 32;
  uint32_t vbase = wf->vgpr_alloc().base;
  write_gfx1250_buffer_resource(*cu, *wf, 4, encode_gfx1250_buffer_resource(kBase, kOffset + 4));
  cu->write_vgpr(vbase + 4, 0, 0xFFFF'8200u);

  cdna5::VbufferMachineInst inst{};
  inst.rsrc = 4;
  inst.soffset = cdna5::OPR_SREG_NULL;
  inst.offen = 1;
  inst.idxen = 0;
  inst.vaddr = 4;
  inst.ioffset = 0x7E00;

  amdgpu::VectorMemState address_state(amdgpu::GLOBAL_MEM);
  address_state.elem_size = 4;
  address_state.num_elems = 1;
  cdna5::mubuf_calculate_addresses(inst, *wf, address_state);
  EXPECT_EQ(address_state.lane_mask, 1ULL);
  EXPECT_EQ(address_state.per_lane_addr[0], kBase + kOffset);
  expect_element_lane_masks(address_state.element_lane_masks, {1ULL});
}

TEST(Gfx1250AddrCalcTest, VbufferRangeCheckDoesNotWrapAt45Bits) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_range_wrap_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_range_wrap_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_range_wrap_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x1FULL);

  constexpr uint64_t kBase = 0x2'0000'1000ULL;
  constexpr uint64_t kMaxNumRecords = (uint64_t{1} << 45) - 1;
  constexpr uint32_t kRawStride = 8192;
  write_gfx1250_buffer_resource(*cu, *wf, 40,
                                encode_gfx1250_buffer_resource(kBase, kMaxNumRecords, kRawStride,
                                                               /*stride_scale_encoding=*/1));
  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase + 4, 0, 0);
  cu->write_vgpr(vbase + 5, 0, 0);
  cu->write_vgpr(vbase + 4, 1, uint32_t{1} << 30);
  cu->write_vgpr(vbase + 5, 1, 0);
  cu->write_vgpr(vbase + 4, 2, (uint32_t{1} << 30) - 1);
  cu->write_vgpr(vbase + 5, 2, 32763);
  cu->write_vgpr(vbase + 4, 3, (uint32_t{1} << 30) - 1);
  cu->write_vgpr(vbase + 5, 3, 32764);
  cu->write_vgpr(vbase + 4, 4, uint32_t{1} << 30);
  cu->write_vgpr(vbase + 5, 4, 1);

  cdna5::VbufferMachineInst inst{};
  inst.rsrc = 40;
  inst.soffset = cdna5::OPR_SREG_NULL;
  inst.idxen = 1;
  inst.offen = 1;
  inst.vaddr = 4;

  amdgpu::VectorMemState range_state(amdgpu::GLOBAL_MEM);
  range_state.elem_size = 4;
  range_state.num_elems = 1;
  cdna5::mubuf_calculate_addresses(inst, *wf, range_state);
  EXPECT_EQ(range_state.lane_mask, 0x5ULL);
  expect_element_lane_masks(range_state.element_lane_masks, {0x5ULL});
  EXPECT_EQ(range_state.per_lane_addr[0], kBase);
  EXPECT_EQ(range_state.per_lane_addr[1], 0ULL);
  EXPECT_EQ(range_state.per_lane_addr[2], kBase + kMaxNumRecords - 4);
  EXPECT_EQ(range_state.per_lane_addr[3], 0ULL);
  EXPECT_EQ(range_state.per_lane_addr[4], 0ULL);
}

TEST(Gfx1250AddrCalcTest, VbufferRangeCheckAcceptsExactEndOnly) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_exact_end_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_exact_end_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_exact_end_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x3ULL);

  constexpr uint64_t kBase = 0x2'0000'1800ULL;
  write_gfx1250_buffer_resource(*cu, *wf, 40,
                                encode_gfx1250_buffer_resource(kBase, /*num_records=*/8));
  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase + 4, 0, 4);
  cu->write_vgpr(vbase + 4, 1, 5);

  cdna5::VbufferMachineInst inst{};
  inst.rsrc = 40;
  inst.soffset = cdna5::OPR_SREG_NULL;
  inst.offen = 1;
  inst.vaddr = 4;

  amdgpu::VectorMemState boundary_state(amdgpu::GLOBAL_MEM);
  boundary_state.elem_size = 4;
  boundary_state.num_elems = 1;
  cdna5::mubuf_calculate_addresses(inst, *wf, boundary_state);
  EXPECT_EQ(boundary_state.lane_mask, 0x1ULL);
  expect_element_lane_masks(boundary_state.element_lane_masks, {0x1ULL});
  EXPECT_EQ(boundary_state.per_lane_addr[0], kBase + 4);
  EXPECT_EQ(boundary_state.per_lane_addr[1], 0ULL);
}

TEST(Gfx1250AddrCalcTest, VbufferZeroNumRecordsMasksAllLanes) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_zero_records_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_zero_records_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_zero_records_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x3ULL);

  // The rocjitsu runtime represents a null optional pointer with an all-zero
  // descriptor. This is a runtime convention, not a distinguished ISA
  // descriptor encoding. Its zero-sized range makes every access out of bounds.
  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase + 4, 0, 0);
  cu->write_vgpr(vbase + 4, 1, 0x1000);

  cdna5::VbufferMachineInst inst{};
  inst.rsrc = 40;
  inst.soffset = cdna5::OPR_SREG_NULL;
  inst.offen = 1;
  inst.idxen = 0;
  inst.vaddr = 4;

  amdgpu::VectorMemState null_range_state(amdgpu::GLOBAL_MEM);
  null_range_state.elem_size = 4;
  null_range_state.num_elems = 1;
  cdna5::mubuf_calculate_addresses(inst, *wf, null_range_state);
  EXPECT_EQ(null_range_state.exec_mask, 0x3ULL);
  EXPECT_EQ(null_range_state.lane_mask, 0ULL);
  EXPECT_EQ(null_range_state.per_lane_addr[0], 0ULL);
  EXPECT_EQ(null_range_state.per_lane_addr[1], 0ULL);
}

TEST(Gfx1250AddrCalcTest, DescriptorCannotMixAdjacentWaveSgprs) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_cross_wave_sgpr_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_cross_wave_sgpr_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 2;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_cross_wave_sgpr_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *executing_wave = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  auto *adjacent_wave = cu->dispatch_wf(1, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(executing_wave, nullptr);
  ASSERT_NE(adjacent_wave, nullptr);
  executing_wave->set_exec(1ULL);
  ASSERT_EQ(adjacent_wave->sgpr_alloc().base, executing_wave->sgpr_alloc().base + cfg.sgprs_per_wf);

  constexpr uint64_t kBase = 0x2'0000'1000ULL;
  const uint32_t descriptor = executing_wave->sgpr_alloc().base + cfg.sgprs_per_wf - 2;
  cu->write_sgpr(descriptor, static_cast<uint32_t>(kBase));
  // Keep the base address unchanged while making the in-range descriptor
  // words describe one record. The complete descriptor still must be denied.
  cu->write_sgpr(descriptor + 1, static_cast<uint32_t>(kBase >> 32) | (1u << 25));
  cu->write_sgpr(adjacent_wave->sgpr_alloc().base, 1u);
  cu->write_sgpr(adjacent_wave->sgpr_alloc().base + 1, 0u);

  cdna5::VbufferMachineInst inst{};
  inst.rsrc = cfg.sgprs_per_wf - 2;
  inst.soffset = cdna5::OPR_SREG_NULL;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  d.elem_size = 4;
  d.num_elems = 1;
  cdna5::mubuf_calculate_addresses(inst, *executing_wave, d);
  EXPECT_EQ(d.lane_mask, 0ULL);
  EXPECT_EQ(d.per_lane_addr[0], 0ULL);
}

TEST(RdnaAddrCalcTest, SmemBasePairCannotCrossWaveSgprBoundary) {
  amdgpu::GpuMemory mem("rdna4_smem_cross_wave_sgpr_mem");
  amdgpu::L2Cache l2("rdna4_smem_cross_wave_sgpr_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 2;
  cfg.sgprs_per_wf = 105;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("rdna4_smem_cross_wave_sgpr_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *executing_wave = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  auto *adjacent_wave = cu->dispatch_wf(1, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(executing_wave, nullptr);
  ASSERT_NE(adjacent_wave, nullptr);
  ASSERT_EQ(adjacent_wave->sgpr_alloc().base, executing_wave->sgpr_alloc().base + cfg.sgprs_per_wf);

  const uint32_t base = adjacent_wave->sgpr_alloc().base - 1;
  cu->write_sgpr(base, 0x12345678u);
  cu->write_sgpr(adjacent_wave->sgpr_alloc().base, 0x00000002u);

  rdna4::SmemMachineInst inst{};
  inst.sbase = 52; // logical s[104:105], which straddles this 105-register block.
  inst.sdata = 4;
  inst.soffset = rdna4::OPR_SMEM_OFFSET_NULL;

  EXPECT_FALSE(rdna4::smem_calculate_address(inst, *executing_wave).has_value());

  rdna4::SLoadB32Smem load(reinterpret_cast<const rdna4::MachineInst *>(&inst));
  load.execute_impl(*executing_wave);
  EXPECT_EQ(load.data(), nullptr);
}

TEST(Gfx1250AddrCalcTest, VbufferStructuredStrideChecksIndexBounds) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_structured_stride_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_structured_stride_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_structured_stride_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x3ULL);

  constexpr uint64_t kBase = 0x2'0000'1000ULL;
  constexpr uint32_t kNumBytes = 32;
  constexpr uint32_t kStride = 16;
  uint32_t vbase = wf->vgpr_alloc().base;
  write_gfx1250_buffer_resource(*cu, *wf, 40,
                                encode_gfx1250_buffer_resource(kBase, kNumBytes, kStride));
  cu->write_vgpr(vbase + 4, 0, 1);
  cu->write_vgpr(vbase + 4, 1, 2);

  cdna5::VbufferMachineInst inst{};
  inst.rsrc = 40;
  inst.soffset = cdna5::OPR_SREG_NULL;
  inst.offen = 0;
  inst.idxen = 1;
  inst.vaddr = 4;

  amdgpu::VectorMemState structured_state(amdgpu::GLOBAL_MEM);
  structured_state.elem_size = 4;
  structured_state.num_elems = 1;
  cdna5::mubuf_calculate_addresses(inst, *wf, structured_state);
  EXPECT_EQ(structured_state.exec_mask, 0x3ULL);
  EXPECT_EQ(structured_state.lane_mask, 0x1ULL);
  EXPECT_EQ(structured_state.per_lane_addr[0], kBase + kStride);
  EXPECT_EQ(structured_state.per_lane_addr[1], 0ULL);
}

TEST(Gfx1250AddrCalcTest, VbufferChecksB64B96AndB128DwordsIndependently) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_partial_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_partial_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_partial_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1ULL);

  constexpr uint64_t kBase = 0x2'0000'2000ULL;
  cdna5::VbufferMachineInst inst{};
  inst.rsrc = 40;
  inst.soffset = cdna5::OPR_SREG_NULL;

  write_gfx1250_buffer_resource(*cu, *wf, 40,
                                encode_gfx1250_buffer_resource(kBase, /*num_records=*/6));
  amdgpu::VectorMemState b64(amdgpu::GLOBAL_MEM);
  b64.elem_size = 4;
  b64.num_elems = 2;
  cdna5::mubuf_calculate_addresses(inst, *wf, b64);
  EXPECT_EQ(b64.lane_mask, 1ULL);
  expect_element_lane_masks(b64.element_lane_masks, {1ULL, 0ULL});

  write_gfx1250_buffer_resource(*cu, *wf, 40,
                                encode_gfx1250_buffer_resource(kBase, /*num_records=*/10));
  amdgpu::VectorMemState b96(amdgpu::GLOBAL_MEM);
  b96.elem_size = 4;
  b96.num_elems = 3;
  cdna5::mubuf_calculate_addresses(inst, *wf, b96);
  EXPECT_EQ(b96.lane_mask, 1ULL);
  expect_element_lane_masks(b96.element_lane_masks, {1ULL, 1ULL, 0ULL});

  amdgpu::VectorMemState b128(amdgpu::GLOBAL_MEM);
  b128.elem_size = 4;
  b128.num_elems = 4;
  cdna5::mubuf_calculate_addresses(inst, *wf, b128);
  EXPECT_EQ(b128.lane_mask, 1ULL);
  expect_element_lane_masks(b128.element_lane_masks, {1ULL, 1ULL, 0ULL, 0ULL});
}

TEST(Gfx1250AddrCalcTest, VbufferNegativeIoffsetCannotReachL1) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_negative_ioffset_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_negative_ioffset_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_negative_ioffset_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1ULL);

  constexpr uint64_t kBase = 0x2'0000'2400ULL;
  write_gfx1250_buffer_resource(*cu, *wf, 40,
                                encode_gfx1250_buffer_resource(kBase, /*num_records=*/4));
  cdna5::VbufferMachineInst inst{};
  inst.rsrc = 40;
  inst.soffset = cdna5::OPR_SREG_NULL;
  inst.ioffset = 0x00FF'FFFCu; // -4 as a signed 24-bit VBUFFER IOFFSET.

  amdgpu::VectorMemState negative_ioffset_state(amdgpu::GLOBAL_MEM);
  negative_ioffset_state.elem_size = 4;
  negative_ioffset_state.num_elems = 2;
  cdna5::mubuf_calculate_addresses(inst, *wf, negative_ioffset_state);

  EXPECT_EQ(negative_ioffset_state.exec_mask, 1ULL);
  EXPECT_EQ(negative_ioffset_state.lane_mask, 0ULL);
  expect_element_lane_masks(negative_ioffset_state.element_lane_masks, {0ULL, 0ULL});
  EXPECT_EQ(negative_ioffset_state.per_lane_addr[0], 0ULL);
}

TEST(Gfx1250AddrCalcTest, VbufferSoffsetParticipatesInNumRecordsAndStrideBounds) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_soffset_bounds_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_soffset_bounds_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_soffset_bounds_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1ULL);

  constexpr uint64_t kBase = 0x2'0000'2800ULL;
  write_gfx1250_buffer_resource(
      *cu, *wf, 40,
      encode_gfx1250_buffer_resource(kBase, /*num_records=*/16, /*raw_stride=*/16,
                                     /*stride_scale_encoding=*/0, /*swizzle=*/false,
                                     /*oob_select=*/true));

  cdna5::VbufferMachineInst inst{};
  inst.rsrc = 40;
  inst.soffset = 8;

  uint32_t sbase = wf->sgpr_alloc().base;
  cu->write_sgpr(sbase + 8, 8);
  amdgpu::VectorMemState exact_end(amdgpu::GLOBAL_MEM);
  exact_end.elem_size = 4;
  exact_end.num_elems = 2;
  cdna5::mubuf_calculate_addresses(inst, *wf, exact_end);
  EXPECT_EQ(exact_end.lane_mask, 1ULL);
  expect_element_lane_masks(exact_end.element_lane_masks, {1ULL, 1ULL});
  EXPECT_EQ(exact_end.per_lane_addr[0], kBase + 8);

  cu->write_sgpr(sbase + 8, 9);
  amdgpu::VectorMemState one_byte_over(amdgpu::GLOBAL_MEM);
  one_byte_over.elem_size = 4;
  one_byte_over.num_elems = 2;
  cdna5::mubuf_calculate_addresses(inst, *wf, one_byte_over);
  EXPECT_EQ(one_byte_over.lane_mask, 1ULL);
  expect_element_lane_masks(one_byte_over.element_lane_masks, {1ULL, 0ULL});
  EXPECT_EQ(one_byte_over.per_lane_addr[0], kBase + 9);
}

TEST(Gfx1250AddrCalcTest, VbufferOobSelectAlsoChecksRecordStride) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_oob_select_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_oob_select_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_oob_select_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1ULL);

  constexpr uint64_t kBase = 0x2'0000'3000ULL;
  write_gfx1250_buffer_resource(
      *cu, *wf, 40,
      encode_gfx1250_buffer_resource(kBase, /*num_records=*/64, /*raw_stride=*/8,
                                     /*stride_scale_encoding=*/0, /*swizzle=*/false,
                                     /*oob_select=*/true));
  cu->write_vgpr(wf->vgpr_alloc().base + 4, 0, 4);

  cdna5::VbufferMachineInst inst{};
  inst.rsrc = 40;
  inst.soffset = cdna5::OPR_SREG_NULL;
  inst.offen = 1;
  inst.vaddr = 4;

  amdgpu::VectorMemState oob_select_state(amdgpu::GLOBAL_MEM);
  oob_select_state.elem_size = 4;
  oob_select_state.num_elems = 2;
  cdna5::mubuf_calculate_addresses(inst, *wf, oob_select_state);
  EXPECT_EQ(oob_select_state.lane_mask, 1ULL);
  expect_element_lane_masks(oob_select_state.element_lane_masks, {1ULL, 0ULL});
}

TEST(Gfx1250AddrCalcTest, VbufferSwizzleDefersBoundsWithNoElementMasks) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_swizzle_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_swizzle_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_swizzle_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1ULL);
  write_gfx1250_buffer_resource(
      *cu, *wf, 40,
      encode_gfx1250_buffer_resource(/*base=*/0x4000, /*num_records=*/0, /*raw_stride=*/16,
                                     /*stride_scale_encoding=*/0, /*swizzle=*/true));

  cdna5::VbufferMachineInst inst{};
  inst.rsrc = 40;
  inst.soffset = cdna5::OPR_SREG_NULL;

  amdgpu::VectorMemState swizzle_state(amdgpu::GLOBAL_MEM);
  swizzle_state.elem_size = 4;
  swizzle_state.num_elems = 4;
  cdna5::mubuf_calculate_addresses(inst, *wf, swizzle_state);
  EXPECT_EQ(swizzle_state.lane_mask, 1ULL);
  EXPECT_TRUE(swizzle_state.element_lane_masks.empty());
}

TEST(Gfx1250AddrCalcTest, VbufferSwizzleRearrangesIndexAndOffsetGroups) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_swizzle_groups_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_swizzle_groups_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_swizzle_groups_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xFULL);

  constexpr uint64_t kBase = 0x2'0000'1000ULL;
  constexpr uint32_t kStride = 64;
  write_gfx1250_buffer_resource(*cu, *wf, 40,
                                encode_gfx1250_buffer_resource(kBase, /*num_records=*/0, kStride,
                                                               /*stride_scale_encoding=*/0,
                                                               /*swizzle=*/true));

  constexpr std::array<uint32_t, 4> kIndexes = {31, 32, 1, 33};
  constexpr std::array<uint32_t, 4> kOffsets = {15, 15, 16, 31};
  constexpr std::array<uint64_t, 4> kExpectedOffsets = {511, 2063, 528, 2591};
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < kIndexes.size(); ++lane) {
    cu->write_vgpr(vbase + 4, lane, kIndexes[lane]);
    cu->write_vgpr(vbase + 5, lane, kOffsets[lane]);
  }

  cdna5::VbufferMachineInst inst{};
  inst.rsrc = 40;
  inst.soffset = cdna5::OPR_SREG_NULL;
  inst.idxen = 1;
  inst.offen = 1;
  inst.vaddr = 4;

  amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
  state.elem_size = 4;
  state.num_elems = 1;
  cdna5::mubuf_calculate_addresses(inst, *wf, state);
  EXPECT_EQ(state.exec_mask, 0xFULL);
  EXPECT_EQ(state.lane_mask, 0xFULL);
  EXPECT_TRUE(state.element_lane_masks.empty());
  for (uint32_t lane = 0; lane < kExpectedOffsets.size(); ++lane)
    EXPECT_EQ(state.per_lane_addr[lane], kBase + kExpectedOffsets[lane]) << "lane " << lane;
}

TEST(Gfx1250AddrCalcTest, VbufferSwizzlePreserves57BitBaseAndUsesLegalOffsets) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_swizzle_offsets_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_swizzle_offsets_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_swizzle_offsets_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1ULL);

  constexpr uint64_t kBase = (uint64_t{1} << 56) | 0x1000;
  constexpr uint32_t kStride = 64;
  constexpr uint32_t kSoffset = 5;
  constexpr uint32_t kIoffset = 11;
  constexpr uint32_t kIndex = 1;
  constexpr uint32_t kVoffset = 20;
  constexpr uint64_t kExpectedIndexedOffset = 528;
  constexpr uint64_t kExpectedOffsetOnlyOffset = 1028;
  write_gfx1250_buffer_resource(*cu, *wf, 40,
                                encode_gfx1250_buffer_resource(kBase, /*num_records=*/0, kStride,
                                                               /*stride_scale_encoding=*/0,
                                                               /*swizzle=*/true));
  uint32_t sbase = wf->sgpr_alloc().base;
  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_sgpr(sbase + 8, kSoffset);

  cdna5::VbufferMachineInst inst{};
  inst.rsrc = 40;
  inst.soffset = 8;
  inst.idxen = 1;
  inst.vaddr = 4;
  inst.ioffset = kIoffset;
  cu->write_vgpr(vbase + 4, 0, kIndex);

  amdgpu::VectorMemState indexed_state(amdgpu::GLOBAL_MEM);
  indexed_state.elem_size = 4;
  indexed_state.num_elems = 1;
  cdna5::mubuf_calculate_addresses(inst, *wf, indexed_state);
  EXPECT_EQ(indexed_state.per_lane_addr[0], kBase + kExpectedIndexedOffset);

  inst.idxen = 0;
  inst.offen = 1;
  cu->write_vgpr(vbase + 4, 0, kVoffset);
  amdgpu::VectorMemState offset_state(amdgpu::GLOBAL_MEM);
  offset_state.elem_size = 4;
  offset_state.num_elems = 1;
  cdna5::mubuf_calculate_addresses(inst, *wf, offset_state);
  EXPECT_EQ(offset_state.per_lane_addr[0], kBase + kExpectedOffsetOnlyOffset);

  inst.offen = 0;
  inst.soffset = cdna5::OPR_SREG_NULL;
  inst.ioffset = 0x7F'FFFFu;
  amdgpu::VectorMemState immediate_state(amdgpu::GLOBAL_MEM);
  immediate_state.elem_size = 4;
  immediate_state.num_elems = 1;
  cdna5::mubuf_calculate_addresses(inst, *wf, immediate_state);
  constexpr uint64_t kExpectedImmediateOffset = 268'434'959;
  EXPECT_EQ(immediate_state.per_lane_addr[0], kBase + kExpectedImmediateOffset);
}

TEST(Gfx1250AddrCalcTest, VbufferStrideScaleAppliesToSwizzledAndLinearAddresses) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_stride_scale_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_stride_scale_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_stride_scale_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1ULL);

  constexpr uint64_t kBase = 0x2'0000'1000ULL;
  constexpr uint64_t kNumRecords = uint64_t{1} << 20;
  constexpr uint32_t kRawStride = 16;
  constexpr uint32_t kIndex = 33;
  constexpr uint32_t kVoffset = 16;
  constexpr uint32_t kIndexStride = 32;
  constexpr uint32_t kElementSize = 16;
  constexpr std::array<uint32_t, 4> kStrideScales = {1, 4, 8, 32};
  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase + 4, 0, kIndex);
  cu->write_vgpr(vbase + 5, 0, kVoffset);

  cdna5::VbufferMachineInst inst{};
  inst.rsrc = 40;
  inst.soffset = cdna5::OPR_SREG_NULL;
  inst.idxen = 1;
  inst.offen = 1;
  inst.vaddr = 4;

  for (uint32_t scale_encoding = 0; scale_encoding < kStrideScales.size(); ++scale_encoding) {
    uint32_t stride = kRawStride * kStrideScales[scale_encoding];

    write_gfx1250_buffer_resource(*cu, *wf, 40,
                                  encode_gfx1250_buffer_resource(kBase, kNumRecords, kRawStride,
                                                                 scale_encoding,
                                                                 /*swizzle=*/true));
    amdgpu::VectorMemState swizzled(amdgpu::GLOBAL_MEM);
    swizzled.elem_size = 4;
    swizzled.num_elems = 1;
    cdna5::mubuf_calculate_addresses(inst, *wf, swizzled);
    EXPECT_EQ(swizzled.per_lane_addr[0],
              kBase + (stride + kElementSize) * kIndexStride + kElementSize)
        << "scale encoding " << scale_encoding;

    write_gfx1250_buffer_resource(
        *cu, *wf, 40,
        encode_gfx1250_buffer_resource(kBase, kNumRecords, kRawStride, scale_encoding));
    amdgpu::VectorMemState linear(amdgpu::GLOBAL_MEM);
    linear.elem_size = 4;
    linear.num_elems = 1;
    cdna5::mubuf_calculate_addresses(inst, *wf, linear);
    EXPECT_EQ(linear.per_lane_addr[0], kBase + kIndex * stride + kVoffset)
        << "scale encoding " << scale_encoding;
  }
}

TEST(Gfx1250AddrCalcTest, VbufferAtomicChecksWholePayload) {
  amdgpu::GpuMemory mem("gfx1250_vbuffer_atomic_oob_mem");
  amdgpu::L2Cache l2("gfx1250_vbuffer_atomic_oob_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("gfx1250_vbuffer_atomic_oob_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 16);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1ULL);
  write_gfx1250_buffer_resource(*cu, *wf, 40,
                                encode_gfx1250_buffer_resource(/*base=*/0x5000, /*num_records=*/6));

  cdna5::VbufferMachineInst inst{};
  inst.rsrc = 40;
  inst.soffset = cdna5::OPR_SREG_NULL;

  amdgpu::VectorMemState atomic_state(amdgpu::GLOBAL_MEM);
  atomic_state.elem_size = 8;
  atomic_state.num_elems = 1;
  atomic_state.atomic_op = amdgpu::AtomicOp::ADD;
  cdna5::mubuf_calculate_addresses(inst, *wf, atomic_state);
  EXPECT_EQ(atomic_state.lane_mask, 0ULL);
  expect_element_lane_masks(atomic_state.element_lane_masks, {0ULL});
  EXPECT_EQ(atomic_state.per_lane_addr[0], 0ULL);
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
  EXPECT_EQ(cu->vgpr_storage_lane_count(), arch == ROCJITSU_CODE_ARCH_CDNA5 ? 32u : 64u);

  std::array<amdgpu::Wavefront *, 4> wfs{};
  for (uint32_t i = 0; i < wfs.size(); ++i) {
    wfs[i] = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
    ASSERT_NE(wfs[i], nullptr);
    const uint32_t vbase = wfs[i]->vgpr_alloc().base;
    cu->write_vgpr(vbase + 1, 0, 0x100u + i);
    cu->write_vgpr(vbase + 4, 0, 0xdead0000u + i);
    cu->write_vgpr(vbase + 4, 31, 0x300u + i);
    const uint32_t registers_per_lazy_chunk =
        4096u / (cu->vgpr_storage_lane_count() * sizeof(uint32_t));
    ASSERT_EQ((vbase + 159) % registers_per_lazy_chunk, registers_per_lazy_chunk - 1);
    cu->write_vgpr(vbase + 159, 0, 0xbeef0000u + i);
    cu->write_vgpr(vbase + 159, 2, 0x200u + i);
    cu->write_vgpr(vbase + 159, 11, 0x600u + i);
    cu->write_vgpr(vbase + 159, 31, 0x400u + i);
    cu->write_vgpr(vbase + 160, 11, 0xdead6000u + i);
  }

  auto decoder = Decoder::create(arch);
  ASSERT_NE(decoder, nullptr);

  constexpr std::array<uint32_t, 2> kReadfirstlaneS24V1 = {
      0x7e300501u, // v_readfirstlane_b32_e32 s24, v1
      0u,
  };
  constexpr std::array<uint32_t, 2> kReadlaneS4V159Lane2 = {
      0xd7600004u, // v_readlane_b32 s4, v159, 2
      0x0201059fu,
  };
  constexpr std::array<uint32_t, 2> kReadlaneS4V4Lane31 = {
      0xd7600004u, // v_readlane_b32 s4, v4, 31
      0x02013f04u,
  };
  constexpr std::array<uint32_t, 2> kReadlaneS4V159S2 = {
      0xd7600004u, // v_readlane_b32 s4, v159, s2
      0x0200059fu,
  };
  constexpr std::array<uint32_t, 2> kWritelaneV159S4S2 = {
      0xd761009fu, // v_writelane_b32 v159, s4, s2
      0x02000404u,
  };

  for (uint32_t i = 0; i < wfs.size(); ++i) {
    const uint32_t sbase = wfs[i]->sgpr_alloc().base;
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, kReadfirstlaneS24V1.data()));
    ASSERT_NE(inst, nullptr);
    cu->write_sgpr(sbase + 24, 0);
    cu->execute_instruction(inst.get(), *wfs[i]);
    EXPECT_EQ(cu->read_sgpr(sbase + 24), 0x100u + i);
  }

  for (uint32_t i = 0; i < wfs.size(); ++i) {
    const uint32_t sbase = wfs[i]->sgpr_alloc().base;
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, kReadlaneS4V159Lane2.data()));
    ASSERT_NE(inst, nullptr);
    cu->write_sgpr(sbase + 2, 0);
    cu->write_sgpr(sbase + 4, 0);
    cu->execute_instruction(inst.get(), *wfs[i]);
    EXPECT_EQ(cu->read_sgpr(sbase + 4), 0x200u + i);
  }

  for (uint32_t i = 0; i < wfs.size(); ++i) {
    const uint32_t sbase = wfs[i]->sgpr_alloc().base;
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, kReadlaneS4V4Lane31.data()));
    ASSERT_NE(inst, nullptr);
    cu->write_sgpr(sbase + 4, 0);
    cu->write_sgpr(sbase + 31, 0);
    cu->execute_instruction(inst.get(), *wfs[i]);
    EXPECT_EQ(cu->read_sgpr(sbase + 4), 0x300u + i);
  }

  for (uint32_t i = 0; i < wfs.size(); ++i) {
    const uint32_t sbase = wfs[i]->sgpr_alloc().base;
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, kReadlaneS4V159S2.data()));
    ASSERT_NE(inst, nullptr);
    cu->write_sgpr(sbase + 2, 31);
    cu->write_sgpr(sbase + 4, 0);
    cu->execute_instruction(inst.get(), *wfs[i]);
    EXPECT_EQ(cu->read_sgpr(sbase + 4), 0x400u + i);
  }

  for (uint32_t i = 0; i < wfs.size(); ++i) {
    const uint32_t sbase = wfs[i]->sgpr_alloc().base;
    const uint32_t vbase = wfs[i]->vgpr_alloc().base;
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, kWritelaneV159S4S2.data()));
    ASSERT_NE(inst, nullptr);
    cu->write_sgpr(sbase + 2, 31);
    cu->write_sgpr(sbase + 4, 0x500u + i);
    cu->execute_instruction(inst.get(), *wfs[i]);
    EXPECT_EQ(cu->read_vgpr(vbase + 159, 2), 0x200u + i);
    EXPECT_EQ(cu->read_vgpr(vbase + 159, 31), 0x500u + i);
  }

  // In Wave32, only selector bits [4:0] are used. Put the source VGPR at the
  // end of a lazy-storage chunk so an unnormalized selector would access the
  // next register instead of the selected lane in this register.
  for (uint32_t i = 0; i < wfs.size(); ++i) {
    ASSERT_EQ(wfs[i]->wf_size(), 32u);
    const uint32_t sbase = wfs[i]->sgpr_alloc().base;
    const uint32_t vbase = wfs[i]->vgpr_alloc().base;
    std::unique_ptr<Instruction> write_inst(decode_valid(*decoder, kWritelaneV159S4S2.data()));
    std::unique_ptr<Instruction> read_inst(decode_valid(*decoder, kReadlaneS4V159S2.data()));
    ASSERT_NE(write_inst, nullptr);
    ASSERT_NE(read_inst, nullptr);
    cu->write_sgpr(sbase + 2, 43);
    cu->write_sgpr(sbase + 4, 0x700u + i);
    cu->execute_instruction(write_inst.get(), *wfs[i]);
    EXPECT_EQ(cu->read_vgpr(vbase + 159, 11), 0x700u + i);
    EXPECT_EQ(cu->read_vgpr(vbase + 159, 31), 0x500u + i);
    EXPECT_EQ(cu->read_vgpr(vbase + 160, 11), 0xdead6000u + i);

    cu->write_sgpr(sbase + 4, 0);
    cu->execute_instruction(read_inst.get(), *wfs[i]);
    EXPECT_EQ(cu->read_sgpr(sbase + 4), 0x700u + i);
  }
}

TEST(RdnaVectorLaneReadTest, Wave64SelectorsShareOneArchitecturalRegisterContext) {
  amdgpu::GpuMemory mem("rdna_wave64_lane_read_mem");
  amdgpu::L2Cache l2("rdna_wave64_lane_read_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("rdna_wave64_lane_read_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf, 64);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);
  EXPECT_EQ(cu->num_wfs(), 1u);

  constexpr std::array<uint32_t, 2> kReadlaneS4V159S2 = {
      0xd7600004u, // v_readlane_b32 s4, v159, s2
      0x0200059fu,
  };
  constexpr std::array<uint32_t, 2> kWritelaneV159S4S2 = {
      0xd761009fu, // v_writelane_b32 v159, s4, s2
      0x02000404u,
  };
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> read_inst(decode_valid(*decoder, kReadlaneS4V159S2.data()));
  std::unique_ptr<Instruction> write_inst(decode_valid(*decoder, kWritelaneV159S4S2.data()));
  ASSERT_NE(read_inst, nullptr);
  ASSERT_NE(write_inst, nullptr);

  const uint32_t sbase = wf->sgpr_alloc().base;
  const uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase + 159, 11, 0x11111111u);
  cu->write_vgpr(vbase + 159, 43, 0x43434343u);
  cu->write_vgpr(vbase + 160, 11, 0xdeadbeefu);
  cu->write_sgpr(sbase + 2, 43);
  cu->execute_instruction(read_inst.get(), *wf);
  EXPECT_EQ(cu->read_sgpr(sbase + 4), 0x43434343u);

  cu->write_sgpr(sbase + 4, 0x84848484u);
  cu->execute_instruction(write_inst.get(), *wf);
  EXPECT_EQ(cu->read_vgpr(vbase + 159, 43), 0x84848484u);
  EXPECT_EQ(cu->read_vgpr(vbase + 159, 11), 0x11111111u);
  EXPECT_EQ(cu->read_vgpr(vbase + 160, 11), 0xdeadbeefu);
}

TEST(CdnaVectorLaneReadTest, Wave64ScalarLaneSelectorsReachUpperHalf) {
  ScopedIsaExecutionBackend execution_backend_scope{&cdna4::execution_backend()};
  amdgpu::GpuMemory mem("cdna_wave64_lane_read_mem");
  amdgpu::L2Cache l2("cdna_wave64_lane_read_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("cdna_wave64_lane_read_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);

  constexpr uint32_t kVgpr = 159;
  constexpr uint32_t kLane = 43;
  constexpr uint32_t kValue = 0x43434343u;
  const uint32_t sbase = wf->sgpr_alloc().base;
  const uint32_t vbase = wf->vgpr_alloc().base;

  cdna4::Vop3MachineInst read_raw{};
  read_raw.vdst = 4;
  read_raw.src0 = 256u + kVgpr;
  read_raw.src1 = 2;
  cdna4::VReadlaneB32Vop3 read_inst(reinterpret_cast<const cdna4::MachineInst *>(&read_raw));

  cdna4::Vop3MachineInst write_raw{};
  write_raw.vdst = kVgpr;
  write_raw.src0 = 4;
  write_raw.src1 = 2;
  cdna4::VWritelaneB32Vop3 write_inst(reinterpret_cast<const cdna4::MachineInst *>(&write_raw));

  cu->write_vgpr(vbase + kVgpr, 11, 0x11111111u);
  cu->write_vgpr(vbase + kVgpr, kLane, kValue);
  cu->write_sgpr(sbase + 2, kLane);
  read_inst.execute_impl(*wf);
  EXPECT_EQ(cu->read_sgpr(sbase + 4), kValue);

  cu->write_sgpr(sbase + 4, 0x84848484u);
  write_inst.execute_impl(*wf);
  EXPECT_EQ(cu->read_vgpr(vbase + kVgpr, kLane), 0x84848484u);
  EXPECT_EQ(cu->read_vgpr(vbase + kVgpr, 11), 0x11111111u);
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
  {
    SCOPED_TRACE("gfx1250");
    ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
    expect_vector_lane_reads_use_own_wave_vgprs(ROCJITSU_CODE_ARCH_CDNA5);
  }
}

TEST(Rdna4GlobalLoadTransposeTest, Wave64B128ReadsLowHalfAndWritesTwoVgprs) {
  amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
  state.wf_size = 64;
  state.lane_mask = ~0ULL;
  state.elem_size = 4;
  state.num_elems = 4;
  state.transpose = static_cast<uint8_t>(amdgpu::TransposeKind::TR16_B128);
  state.response_data.resize(64 * 16, 0xEE);
  for (uint32_t lane = 0; lane < 32; ++lane) {
    for (uint32_t halfword = 0; halfword < 8; ++halfword) {
      const uint16_t value = static_cast<uint16_t>((lane << 8) | halfword);
      std::memcpy(&state.response_data[lane * 16 + halfword * 2], &value, sizeof(value));
    }
  }

  EXPECT_EQ(amdgpu::transpose_request_lane_mask(state), 0xFFFFFFFFULL);
  amdgpu::transpose_response(state);

  ASSERT_EQ(state.num_elems, 2u);
  ASSERT_EQ(state.response_data.size(), 64u * sizeof(uint64_t));
  auto result = [&](uint32_t lane) {
    uint64_t value = 0;
    std::memcpy(&value, &state.response_data[lane * sizeof(value)], sizeof(value));
    return value;
  };
  EXPECT_EQ(result(0), 0x0300020001000000ULL);
  EXPECT_EQ(result(32), 0x0700060005000400ULL);
  EXPECT_EQ(result(63), 0x1F071E071D071C07ULL);
}

// A VOP3 output modifier scales the result. INEXACT must be derived from the
// scaled exact value, not by comparing the scaled result against the unscaled
// product -- that comparison is unequal for every nonzero product as soon as
// any modifier is present, and a spurious INEXACT with MODE.excp_en set stops
// the wave for the debugger on arithmetic that was exact.
TEST(AluExceptionTest, OutputModifierDoesNotFabricateInexact) {
  constexpr uint32_t kInexact = 1u << 5;

  // 2.0 * 3.0 == 6.0 exactly, and stays exact under every OMOD scale.
  EXPECT_EQ(amdgpu::classify_mul_f32(2.0f, 3.0f) & kInexact, 0u);
  for (float scale : {2.0f, 4.0f, 0.5f})
    EXPECT_EQ(amdgpu::classify_mul_f32(2.0f, 3.0f, scale) & kInexact, 0u) << "omod scale " << scale;

  // A genuinely inexact product stays reported, with and without a modifier.
  EXPECT_EQ(amdgpu::classify_mul_f32(1.1f, 1.1f) & kInexact, kInexact);
  EXPECT_EQ(amdgpu::classify_mul_f32(1.1f, 1.1f, 2.0f) & kInexact, kInexact);

  // NaN compares unequal to itself, so an exactness test that does not exclude
  // non-finite values reports INEXACT for it even with no modifier at all.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(amdgpu::classify_mul_f32(nan, 3.0f) & kInexact, 0u);
  EXPECT_EQ(amdgpu::classify_mul_f32(nan, 3.0f, 2.0f) & kInexact, 0u);
}

} // namespace
