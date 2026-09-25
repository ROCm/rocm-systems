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

class ReadlaneSelectorTest : public ::testing::TestWithParam<rj_code_arch_t> {
protected:
  bool gfx9() const {
    const auto arch = GetParam();
    return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
           arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
  }
};

INSTANTIATE_TEST_SUITE_P(AllArchitectures, ReadlaneSelectorTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                           ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                           ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA1,
                                           ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                                           ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5));

TEST_P(ReadlaneSelectorTest, ReadlaneMasksInline64AndLiteralIndices) {
  for (auto arch : {GetParam()}) {
    amdgpu::GpuMemory mem("readlane_mem");
    amdgpu::L2Cache l2("readlane_l2");
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = gfx9() ? 102 : 106;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("readlane", cfg, &mem, &l2);
    auto decoder = Decoder::create(arch);
    auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, 16);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(0); // READLANE ignores EXEC, even for inactive source lanes.
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      cu->write_vgpr(wf->vgpr_alloc().base, lane, 1000 + lane);
    for (uint32_t selector : {192u, 255u}) {
      if (selector == 255 && arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA4)
        continue; // Literal lane indices are only qualified on these two targets.
      // v_readlane_b32 s4, v0, 64 / literal 95, followed by s_endpgm.
      std::array<uint32_t, 4> words{gfx9() ? 0xd2890004u : 0xd7600004u, 256u | (selector << 9), 95,
                                    0xbfb00000};
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
      ASSERT_NE(inst, nullptr);
      EXPECT_EQ(inst->size(), selector == 255 ? 12u : 8u);
      EXPECT_EQ(inst->disassemble(),
                selector == 255 ? "v_readlane_b32 s4, v0, 0x5f" : "v_readlane_b32 s4, v0, 64");
      ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
      EXPECT_EQ(cu->read_sgpr(wf->sgpr_alloc().base + 4),
                1000u + ((selector == 255 ? 95u : 64u) & (wf->wf_size() - 1)));
    }
    wf->halt();
  }
}

} // namespace
