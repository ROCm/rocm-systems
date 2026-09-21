// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_resource.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include <gtest/gtest.h>

namespace {
using namespace rocjitsu;
template <typename Fields> Fields query_fields() {
  Fields f{};
  f.dmask = 10;
  f.a16 = 1;
  f.vaddr = 8;
  f.vdata = 8;
  f.srsrc = 1;
  if constexpr (requires { f.dim; })
    f.dim = 5;
  if constexpr (requires { f.tfe; })
    f.tfe = 1;
  return f;
}
class ImageResourceTest : public testing::TestWithParam<rj_code_arch_t> {
protected:
  std::array<uint32_t, 8> descriptor(uint32_t type, bool view, bool uav = false) {
    std::array<uint32_t, 8> r{};
    const auto set = [&](uint32_t bit, uint32_t width, uint32_t value) {
      for (uint32_t i = 0; i < width; ++i)
        r[(bit + i) / 32] |= ((value >> i) & 1u) << ((bit + i) % 32);
    };
    const bool gfx9 = arch_is_cdna_4_or_lower(GetParam());
    const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
    set(gfx9 ? 64 : 62, gfx12 ? 16 : 14, 127);
    set(78, gfx12 ? 16 : 14, 63);
    set(gfx12 ? 57 : 108, gfx12 ? 5 : 4, view ? 2 : 0);
    set(gfx12 ? 111 : 112, gfx12 ? 5 : 4, view ? 4 : 6);
    set(124, 4, type);
    set(128, 13, 17);
    set(gfx9 ? 160 : 144, 13, view ? 3 : 0);
    if (uav)
      set(gfx12 ? 164 : 160, 1, 1);
    return r;
  }
};

TEST_P(ImageResourceTest, DimensionsLayersMipViewsAndOutOfRangeLod) {
  // Expected values match descriptor-only HIP probes on gfx1100 and gfx1201.
  for (uint32_t type = 8; type < 16; ++type) {
    for (bool view : {false, true}) {
      const auto r = descriptor(type, view);
      const bool msaa = type >= 14;
      const uint32_t base = msaa ? 0 : view ? 2 : 0;
      const uint32_t levels = msaa ? 1 : view ? 3 : 7;
      for (uint32_t lod : {0u, 1u, 2u, 6u, 7u, 31u, 0xffffffffu}) {
        SCOPED_TRACE(testing::Message() << "type=" << type << " view=" << view << " lod=" << lod);
        std::array<uint32_t, 4> expected{0, 0, 0, levels};
        if (lod < levels) {
          expected[0] = std::max(1u, 128u >> (base + lod));
          if (type != 8 && type != 12)
            expected[1] = std::max(1u, 64u >> (base + lod));
          if (type == 10)
            expected[2] = std::max(1u, 18u >> (base + lod));
          if (type == 11 || type == 13 || type == 15)
            expected[2] = view ? 15 : 18;
          if (type == 12)
            expected[1] = view ? 15 : 18;
        }
        EXPECT_EQ(amdgpu::image_resource_info(r, GetParam(), lod), expected);
      }
    }
  }
  EXPECT_EQ(amdgpu::image_resource_info({}, GetParam(), 0), (std::array<uint32_t, 4>{}));
  if (!arch_is_cdna_4_or_lower(GetParam()))
    EXPECT_EQ(amdgpu::image_resource_info(descriptor(10, true, true), GetParam(), 1),
              (std::array<uint32_t, 4>{16, 8, 15, 3}));
}

TEST_P(ImageResourceTest, DecodedQueryCompactsMaskAndPreservesInactiveLanesWithAliasedLod) {
  amdgpu::GpuMemory memory("query_memory");
  amdgpu::L2Cache l2("query_l2");
  l2.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = GetParam();
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 32;
  cfg.vgprs_per_wf = 32;
  auto cu = amdgpu::ComputeUnitCore::create("query_cu", cfg, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0, 32, 32);
  ASSERT_NE(wf, nullptr);
  const uint32_t last = wf->wf_size() - 1;
  wf->set_exec(1 | (uint64_t{1} << last));
  auto r = descriptor(13, true);
  for (uint32_t i = 0; i < r.size(); ++i)
    cu->write_sgpr(wf->sgpr_alloc().base + 4 + i, r[i]);
  const auto vb = wf->vgpr_alloc().base;
  for (uint32_t lane : {0u, 1u, last}) {
    cu->write_vgpr(vb + 8, lane, 0x10001); // A16 must ignore the high half.
    cu->write_vgpr(vb + 9, lane, 0xdeadbeef);
    cu->write_vgpr(vb + 10, lane, 0xdeadbeef);
  }
  std::array<uint32_t, 3> words{};
#define LEGACY(ARCH, ENUM)                                                                         \
  case ENUM: {                                                                                     \
    const auto w =                                                                                 \
        ARCH::build_mimg(ARCH::kImageGetResinfoMimg, query_fields<ARCH::MimgBuilderFields>());     \
    std::copy(w.begin(), w.end(), words.begin());                                                  \
    break;                                                                                         \
  }
  switch (GetParam()) {
    LEGACY(cdna1, ROCJITSU_CODE_ARCH_CDNA1)
    LEGACY(cdna2, ROCJITSU_CODE_ARCH_CDNA2)
    LEGACY(rdna1, ROCJITSU_CODE_ARCH_RDNA1)
    LEGACY(rdna2, ROCJITSU_CODE_ARCH_RDNA2)
    LEGACY(rdna3, ROCJITSU_CODE_ARCH_RDNA3)
    LEGACY(rdna3_5, ROCJITSU_CODE_ARCH_RDNA3_5)
  case ROCJITSU_CODE_ARCH_RDNA4:
    words = rdna4::build_vimage(
        rdna4::kImageGetResinfoVimage,
        {.dim = 5, .a16 = 1, .dmask = 10, .vdata = 8, .rsrc = 4, .tfe = 1, .vaddr0 = 8});
    break;
  default:
    FAIL();
  }
#undef LEGACY
  auto decoder = Decoder::create(GetParam());
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
  for (uint32_t lane : {0u, last}) {
    EXPECT_EQ(cu->read_vgpr(vb + 8, lane), 8);
    EXPECT_EQ(cu->read_vgpr(vb + 9, lane), 3);
    EXPECT_EQ(cu->read_vgpr(vb + 10, lane), 0xdeadbeef);
  }
  EXPECT_EQ(cu->read_vgpr(vb + 8, 1), 0x10001);
  EXPECT_EQ(cu->read_vgpr(vb + 9, 1), 0xdeadbeef);
  wf->halt();
}
TEST(Cdna2ImageResourceTest, AccumulatorDestinationKeepsLodInVgpr) {
  amdgpu::GpuMemory memory("query_acc_memory");
  amdgpu::L2Cache l2("query_acc_l2");
  l2.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA2;
  cfg.num_wf_slots = 2;
  cfg.sgprs_per_wf = 32;
  cfg.vgprs_per_wf = 512;
  auto cu = amdgpu::ComputeUnitCore::create("query_acc_cu", cfg, &memory, &l2);
  auto *reserved = cu->dispatch_wf(0, 0, 32, 512);
  auto *wf = cu->dispatch_wf(0, 0, 32, 512);
  ASSERT_NE(reserved, nullptr);
  ASSERT_NE(wf, nullptr);
  const uint32_t last = wf->wf_size() - 1;
  const uint32_t vb = wf->vgpr_alloc().base;
  ASSERT_NE(vb, 0u);
  wf->set_exec(1 | (uint64_t{1} << last));
  // GFX9 2D array: 128x64, 18 layers, seven mip levels.
  const std::array<uint32_t, 8> descriptor{0, 0, 127 | (63 << 14), (13u << 28) | (6 << 16), 17, 0,
                                           0, 0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    cu->write_sgpr(wf->sgpr_alloc().base + 4 + i, descriptor[i]);
  for (uint32_t lane : {0u, 1u, last}) {
    cu->write_vgpr(vb + 8, lane, 0x10001);
    cu->write_vgpr(vb + 9, lane, 0xdeadbeef);
    for (uint32_t i = 0; i < 3; ++i)
      cu->write_vgpr(vb + 256 + 8 + i, lane, 0xdeadbeef);
  }
  auto fields = query_fields<cdna2::MimgBuilderFields>();
  fields.acc = 1;
  const auto words = cdna2::build_mimg(cdna2::kImageGetResinfoMimg, fields);
  auto decoder = Decoder::create(cfg.arch);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
  for (uint32_t lane : {0u, 1u, last}) {
    EXPECT_EQ(cu->read_vgpr(vb + 8, lane), 0x10001);
    EXPECT_EQ(cu->read_vgpr(vb + 9, lane), 0xdeadbeef);
    EXPECT_EQ(cu->read_vgpr(vb + 256 + 8, lane), lane == 1 ? 0xdeadbeefu : 32u);
    EXPECT_EQ(cu->read_vgpr(vb + 256 + 9, lane), lane == 1 ? 0xdeadbeefu : 7u);
    EXPECT_EQ(cu->read_vgpr(vb + 256 + 10, lane), 0xdeadbeef);
  }
  wf->halt();
  reserved->halt();
}

INSTANTIATE_TEST_SUITE_P(AllImageTargets, ImageResourceTest,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));
} // namespace
