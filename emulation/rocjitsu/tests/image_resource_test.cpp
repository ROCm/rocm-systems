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
  if (!arch_is_cdna_4_or_lower(GetParam())) {
    EXPECT_EQ(amdgpu::image_resource_info(descriptor(10, true, true), GetParam(), 1),
              (std::array<uint32_t, 4>{16, 8, 15, 3}));
  }
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

class ImageLodTest : public testing::TestWithParam<rj_code_arch_t> {};

TEST_P(ImageLodTest, PhysicalLodsWithAliasedCoordinatesMasksAndInactiveQuadLanes) {
  // Captured on gfx1100 and gfx1201. The zero-gradient row uses the ISA result;
  // RADV separately replaces its raw zero with -FLT_MAX for textureQueryLod.
  constexpr std::array<std::array<std::array<uint32_t, 2>, 4>, 4> coordinates{{
      {{{0x3f000000, 0x3f000000},
        {0x3f000000, 0x3f000000},
        {0x3f000000, 0x3f000000},
        {0x3f000000, 0x3f000000}}},
      {{{0x3dce0000, 0x3f369000},
        {0x3ef38000, 0x3e270000},
        {0x3db10000, 0x3e9e6000},
        {0x3e5ec000, 0x3e4b4000}}},
      {{{0x3eb2b000, 0x3d978000},
        {0x3ea20000, 0x3e1a2000},
        {0x3e49a000, 0x3eae6000},
        {0x3eb32000, 0x3e9ff000}}},
      {{{0x3e2d2000, 0x3cf18000},
        {0x3dbf6000, 0x3d7d4000},
        {0x3d922000, 0x3e738000},
        {0x3e699000, 0x3e089000}}},
  }};
  struct Case {
    uint32_t quad;
    std::array<uint32_t, 2> expected;
    uint32_t mip_filter = 1, min_lod = 0, max_lod = 768;
    int32_t bias = 0;
    uint32_t base_level = 0, aniso_ratio = 0;
  };
  constexpr Case cases[]{
      {0, {0, 0}},
      {0, {0x3f800000, 0}, 1, 320, 768, 576},
      {1, {0x40400000, 0x40258000}},
      {1, {0x40258000, 0x40258000}, 2},
      {2, {0x3f800000, 0x3fac0000}, 1, 320},
      {3, {0x3fa00000, 0x3f6f0000}, 2, 320},
      {1, {0x40000000, 0x40258000}, 1, 0, 448},
      {1, {0x3fe00000, 0x40258000}, 2, 0, 448},
      {1, {0x40400000, 0x40358000}, 1, 0, 768, 64},
      {2, {0x3f800000, 0x3f180000}, 1, 0, 768, -192},
      {1, {0x3f800000, 0x3f420000}, 1, 0, 768, 0, 0, 2},
      {1, {0x3f800000, 0x3f420000}, 2, 0, 768, 0, 0, 4},
      {1, {0x40000000, 0x3fcb0000}, 1, 0, 768, 0, 1},
      {1, {0x3f160000, 0x3f160000}, 2, 0, 768, 0, 2},
  };
  amdgpu::GpuMemory memory("lod_memory");
  amdgpu::L2Cache l2("lod_l2");
  l2.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = GetParam();
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 32;
  cfg.vgprs_per_wf = 32;
  auto cu = amdgpu::ComputeUnitCore::create("lod_cu", cfg, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0, 32, 32);
  ASSERT_NE(wf, nullptr);
  const auto vb = wf->vgpr_alloc().base, sb = wf->sgpr_alloc().base;
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  auto decoder = Decoder::create(GetParam());
  for (const auto &c : cases) {
    std::array<uint32_t, 8> resource{};
    resource[1] = (3u << 30) | (42u << (gfx12 ? 17 : 20)) | (3u << (gfx12 ? 12 : 16));
    resource[2] = 1u | (7u << 14);
    resource[3] = (9u << 28) | (3u << (gfx12 ? 15 : 16));
    resource[gfx12 ? 1 : 3] |= c.base_level << (gfx12 ? 25 : 12);
    resource[5] = 4u << 20;
    const uint32_t perf_mip = c.aniso_ratio ? c.aniso_ratio + 6 : 0;
    const uint32_t filter = c.aniso_ratio ? 3 : 1;
    std::array<uint32_t, 4> sampler{};
    sampler[0] = 2u | (2u << 3) | (2u << 6) | (c.aniso_ratio << 9);
    sampler[1] = c.min_lod | (c.max_lod << (gfx12 ? 13 : 12));
    sampler[2] =
        (uint32_t(c.bias) & 0x3fff) | (filter << 20) | (filter << 22) | (c.mip_filter << 26);
    if (gfx12) {
      sampler[2] |= (perf_mip & 3) << 30;
      sampler[3] |= perf_mip >> 2;
    } else {
      sampler[1] |= perf_mip << 24;
    }
    for (uint32_t i = 0; i < resource.size(); ++i)
      cu->write_sgpr(sb + i, resource[i]);
    for (uint32_t i = 0; i < sampler.size(); ++i)
      cu->write_sgpr(sb + 8 + i, sampler[i]);
    for (uint8_t mask : {1, 2, 3}) {
      std::array<uint32_t, 3> words{};
      if (gfx12) {
        words = rdna4::build_vsample(
            rdna4::kImageGetLodVsample,
            {.dim = 1, .dmask = mask, .vdata = 8, .samp = 8, .vaddr0 = 8, .vaddr1 = 9});
      } else {
        const auto encoded = rdna3::build_mimg(
            rdna3::kImageGetLodMimg, {.dim = 1, .dmask = mask, .vaddr = 8, .vdata = 8, .ssamp = 2});
        std::copy(encoded.begin(), encoded.end(), words.begin());
      }
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
      ASSERT_NE(inst, nullptr);
      for (uint64_t exec : {0xfull, 0xcull}) {
        wf->set_exec(exec);
        for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
          cu->write_vgpr(vb + 8, lane, coordinates[c.quad][lane % 4][0]);
          cu->write_vgpr(vb + 9, lane, coordinates[c.quad][lane % 4][1]);
          cu->write_vgpr(vb + 10, lane, 0xdeadbeef);
        }
        ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
        for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
          auto expected = coordinates[c.quad][lane % 4];
          if (exec & (uint64_t{1} << lane)) {
            uint32_t component = 0;
            for (uint32_t j = 0; j < 2; ++j)
              if (mask & (1u << j))
                expected[component++] = c.expected[j];
          }
          EXPECT_EQ(cu->read_vgpr(vb + 8, lane), expected[0]);
          EXPECT_EQ(cu->read_vgpr(vb + 9, lane), expected[1]);
          EXPECT_EQ(cu->read_vgpr(vb + 10, lane), 0xdeadbeef);
        }
      }
    }
  }
  wf->halt();
}

INSTANTIATE_TEST_SUITE_P(GraphicsTargets, ImageLodTest,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));

INSTANTIATE_TEST_SUITE_P(AllImageTargets, ImageResourceTest,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));
} // namespace
