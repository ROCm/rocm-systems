// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/shared/memory_issue.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>

namespace {

using namespace rocjitsu;

// GWS (Global Wave Sync) DS opcodes. These values are shared by every GFX9
// (CDNA1-3) ISA that defines the instructions; CDNA4 and the GFX11/12 ISAs do
// not carry them in the machine-readable ISA.
struct GwsVariant {
  uint16_t opcode;
  std::string_view mnemonic;
  bool has_addr;
};

constexpr std::array<GwsVariant, 6> kGwsVariants{{
    {152, "ds_gws_sema_release_all", false},
    {153, "ds_gws_init", true},
    {154, "ds_gws_sema_v", false},
    {155, "ds_gws_sema_br", true},
    {156, "ds_gws_sema_p", false},
    {157, "ds_gws_barrier", true},
}};

std::array<uint32_t, 2> build_gws(rj_code_arch_t arch, uint16_t op, uint8_t gds, uint8_t addr) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::build_ds(op, {.gds = gds, .addr = addr});
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::build_ds(op, {.gds = gds, .addr = addr});
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_ds(op, {.gds = gds, .addr = addr});
  default:
    ADD_FAILURE() << "unsupported arch for GWS build: " << unsigned(arch);
    return {};
  }
}

class DsGwsTest : public ::testing::TestWithParam<rj_code_arch_t> {};

INSTANTIATE_TEST_SUITE_P(Gfx9Architectures, DsGwsTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                           ROCJITSU_CODE_ARCH_CDNA3));

// Each GWS operation must decode, execute without raising
// UnimplementedInstruction, and publish a zero-payload LOCAL_MEM state that
// participates in lgkmcnt/GDS wait accounting.
TEST_P(DsGwsTest, StructuralModelExecutesAndAccounts) {
  const auto arch = GetParam();
  amdgpu::GpuMemory mem("ds_gws_mem");
  amdgpu::L2Cache l2("ds_gws_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 102;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("ds_gws", cfg, &mem, &l2);
  auto decoder = Decoder::create(arch);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, 16);
  ASSERT_NE(wf, nullptr);
  // GWS executes independently of EXEC; use an empty mask to exercise that.
  wf->set_exec(0);
  wf->set_m0(0);

  for (const auto &variant : kGwsVariants) {
    const auto words = build_gws(arch, variant.opcode, /*gds=*/1, variant.has_addr ? 4 : 0);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr) << variant.mnemonic << " failed to decode on arch " << unsigned(arch);
    EXPECT_EQ(inst->mnemonic(), variant.mnemonic);

    // Previously these threw UnimplementedInstruction; they must now succeed.
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded())
        << variant.mnemonic << " execution failed on arch " << unsigned(arch);

    // A zero-payload LOCAL_MEM store is published so the memory pipeline takes
    // the ordinary completion path.
    ASSERT_NE(inst->data(), nullptr) << variant.mnemonic;
    EXPECT_EQ(inst->data()->tag(), amdgpu::LOCAL_MEM);
    const auto *state = inst->data_as<amdgpu::VectorMemState>();
    EXPECT_EQ(state->num_elems, 0u) << variant.mnemonic;
    EXPECT_FALSE(state->is_load) << variant.mnemonic;
    EXPECT_EQ(state->wait_counter_type, amdgpu::WaitCounterType::LGKMCNT) << variant.mnemonic;

    // Wait-accounting metadata: MEMORY_OP with an EXEC-independent LGKMCNT/GDS
    // obligation (GWS legitimately targets GDS).
    EXPECT_TRUE(inst->is_memory_op()) << variant.mnemonic;
    const auto *info = inst->amdgpu_memory_issue_info();
    ASSERT_NE(info, nullptr) << variant.mnemonic;
    EXPECT_FALSE(info->exec_masked) << variant.mnemonic;
    bool has_gds_lgkmcnt = false;
    for (const auto &ob : info->counter_obligations()) {
      if (ob.wait_counter_type() == amdgpu::WaitCounterType::LGKMCNT &&
          ob.completion_class() == amdgpu::MemoryCompletionClass::GDS)
        has_gds_lgkmcnt = true;
    }
    EXPECT_TRUE(has_gds_lgkmcnt) << variant.mnemonic;

    // The op completes cleanly through the local-memory pipeline with no
    // register effect and no cross-workgroup blocking.
    amdgpu::LocalMemPipeline pipeline;
    pipeline.issue(inst.release(), *wf);
  }

  wf->halt();
}

// Without the GDS bit the model still executes (architecturally GWS always
// uses GDS, but the structural path must not throw): it decodes and completes
// as an LDS-classified op.
TEST_P(DsGwsTest, DecodesWithoutGdsBit) {
  const auto arch = GetParam();
  auto decoder = Decoder::create(arch);
  for (const auto &variant : kGwsVariants) {
    const auto words = build_gws(arch, variant.opcode, /*gds=*/0, variant.has_addr ? 4 : 0);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr) << variant.mnemonic << " failed to decode on arch " << unsigned(arch);
    EXPECT_EQ(inst->mnemonic(), variant.mnemonic);
    EXPECT_TRUE(inst->is_memory_op()) << variant.mnemonic;
  }
}

} // namespace
