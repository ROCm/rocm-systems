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

constexpr uint32_t pack16(uint16_t lo, uint16_t hi) {
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

class PackedFmacConstantTest : public ::testing::TestWithParam<rj_code_arch_t> {
protected:
  bool gfx9() const {
    const auto arch = GetParam();
    return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
           arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
  }
};

INSTANTIATE_TEST_SUITE_P(AllArchitectures, PackedFmacConstantTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                           ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                           ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA1,
                                           ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                                           ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5));

TEST_P(PackedFmacConstantTest, PackedFmacDistinguishesInlineAndLiteralPairs) {
  for (auto arch : {GetParam()}) {
    amdgpu::GpuMemory mem("packed_inline_mem");
    amdgpu::L2Cache l2("packed_inline_l2");
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = gfx9() ? 102 : 106;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("packed_inline", cfg, &mem, &l2);
    auto decoder = Decoder::create(arch);
    auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, 16);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(5);
    const auto vb = wf->vgpr_alloc().base;
    for (bool vop3 : {false, true}) {
      if (vop3 && !gfx9())
        continue;
      for (uint32_t selector : {244u, 255u}) {
        if (vop3 && selector == 255)
          continue; // GFX9 VOP3 has no appended literal encoding.
        for (uint32_t lane = 0; lane < 3; ++lane) {
          cu->write_vgpr(vb + 1, lane, pack16(0x3c00, 0x3c00));
          cu->write_vgpr(vb + 2, lane, lane == 1 ? 0xdeadbeef : 0);
        }
        // v_pk_fmac_f16 v2, 2.0 / literal (2,1), v1.
        std::array<uint32_t, 2> words =
            vop3 ? std::array<uint32_t, 2>{0xd13c0002u, selector | (257u << 9)}
                 : std::array<uint32_t, 2>{(60u << 25) | (2u << 17) | (1u << 9) | selector,
                                           pack16(0x4000, 0x3c00)};
        std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
        ASSERT_NE(inst, nullptr);
        ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
        const auto expected = selector == 244 ? pack16(0x4000, 0x4000) : pack16(0x4000, 0x3c00);
        EXPECT_EQ(cu->read_vgpr(vb + 2, 0), expected);
        EXPECT_EQ(cu->read_vgpr(vb + 2, 2), expected);
        EXPECT_EQ(cu->read_vgpr(vb + 2, 1), 0xdeadbeefu);
      }
    }
    wf->halt();
  }
}

} // namespace
