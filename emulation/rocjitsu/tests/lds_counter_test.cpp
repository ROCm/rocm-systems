// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <optional>

namespace {

using namespace rocjitsu;

class LdsCounterTest : public ::testing::TestWithParam<rj_code_arch_t> {
protected:
  bool gfx9() const {
    const auto arch = GetParam();
    return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
           arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
  }
};

INSTANTIATE_TEST_SUITE_P(AllArchitectures, LdsCounterTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                           ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                           ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA1,
                                           ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                                           ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5));

TEST_P(LdsCounterTest, AppendAndConsumeBroadcastOldCounterToSparseActiveLanes) {
  for (auto arch : {GetParam()}) {
    amdgpu::GpuMemory mem("lds_counter_mem");
    amdgpu::L2Cache l2("lds_counter_l2");
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = gfx9() ? 102 : 106;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("lds_counter", cfg, &mem, &l2);
    auto decoder = Decoder::create(arch);
    auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, 16);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(0x25);
    wf->set_m0(0);
    for (uint16_t op : {uint16_t{61}, uint16_t{62}}) {
      wf->lds().write32(0, 99);
      for (uint32_t lane = 0; lane < 6; ++lane)
        cu->write_vgpr(wf->vgpr_alloc().base + 6, lane, 0xdeadbeef);
      // LLVM encoding: GFX9 uses op 189/190; RDNA and CDNA5 use 61/62.
      const std::array<uint32_t, 2> words{
          0xd8000000u | ((op + (gfx9() ? 128u : 0u)) << (gfx9() ? 17 : 18)), 6u << 24};
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
      ASSERT_NE(inst, nullptr);
      ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
      ASSERT_NE(inst->data(), nullptr);
      amdgpu::LocalMemPipeline pipeline;
      pipeline.issue(inst.release(), *wf);
      EXPECT_EQ(wf->lds().read32(0), op == 61 ? 96u : 102u);
      for (uint32_t lane = 0; lane < 6; ++lane)
        EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + 6, lane),
                  (0x25 & (1u << lane)) ? 99u : 0xdeadbeefu);
    }
    wf->halt();
  }
}

} // namespace
