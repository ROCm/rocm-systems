// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/mimg.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx9_image_access.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <memory>

namespace {
using namespace rocjitsu;

constexpr uint32_t kIdentity = 4 | (5 << 3) | (6 << 6) | (7 << 9);
constexpr uint32_t kXOne = 4 | (1 << 9); // X, 0, 0, 1 (an R-only format).
constexpr uint64_t kBase = 0x10000;
constexpr uint32_t kFmt32 = 4, kFmt8888 = 10, kUint = 4, kUnorm = 0;

struct Image {
  uint32_t data_format = kFmt32;
  uint32_t num_format = kUint;
  uint32_t width = 1, height = 1, depth = 1, pitch = 64;
  uint32_t type = amdgpu::kGfx9Image2d;
  uint32_t selectors = kXOne;
  uint32_t swizzle_mode = 0;
};

std::array<uint32_t, 8> descriptor(const Image &image) {
  std::array<uint32_t, 8> r{};
  const auto set = [&](uint32_t bit, uint32_t width, uint64_t value) {
    for (uint32_t i = 0; i < width; ++i)
      r[(bit + i) / 32] |= static_cast<uint32_t>((value >> i) & 1u) << ((bit + i) % 32);
  };
  set(0, 40, kBase >> 8);
  set(52, 6, image.data_format);
  set(58, 4, image.num_format);
  set(64, 14, image.width - 1);
  set(78, 14, image.height - 1);
  set(96, 12, image.selectors);
  set(116, 5, image.swizzle_mode);
  set(124, 4, image.type);
  set(128, 13, image.depth - 1);
  set(141, 16, image.pitch - 1);
  return r;
}

class Cdna2ImageAccessTest : public testing::Test {
protected:
  amdgpu::GpuMemory memory{"image_memory"};
  amdgpu::L2Cache l2{"image_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wf = nullptr;
  uint32_t sb = 0, vb = 0;

  void SetUp() override {
    l2.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA2;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 32;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("image_cu", cfg, &memory, &l2);
    wf = cu->dispatch_wf(0, 0, 106, 32);
    ASSERT_NE(wf, nullptr);
    sb = wf->sgpr_alloc().base;
    vb = wf->vgpr_alloc().base;
    wf->set_exec(5); // Lane 1 must not read, store or receive a load result.
  }
  void TearDown() override {
    if (wf)
      wf->halt();
  }

  void bind(const Image &image) {
    const auto words = descriptor(image);
    for (uint32_t i = 0; i < words.size(); ++i)
      cu->write_sgpr(sb + i, words[i]);
  }
  void coordinates(uint32_t lane, std::array<uint32_t, 3> c) {
    for (uint32_t i = 0; i < c.size(); ++i)
      cu->write_vgpr(vb + i, lane, c[i]);
  }
  /// Executes image_load (VDATA v8) or image_store (VDATA v8) on s[0:7], VADDR v0.
  /// @returns Whether the instruction reached the memory pipeline.
  bool issue(bool load, uint32_t dmask = 15) {
    cdna2::MimgMachineInst m{};
    m.dmask = dmask;
    m.unorm = 1;
    m.da = 1;
    m.vaddr = 0;
    m.vdata = 8;
    m.srsrc = 0;
    const auto *raw = reinterpret_cast<const cdna2::MachineInst *>(&m);
    return load ? run<cdna2::ImageLoadMimg>(raw) : run<cdna2::ImageStoreMimg>(raw);
  }
  template <typename Inst> bool run(const cdna2::MachineInst *raw) {
    auto inst = std::make_unique<Inst>(raw);
    EXPECT_TRUE(inst->is_memory_op());
    inst->execute_impl(*wf);
    if (!inst->data())
      return false;
    amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
    EXPECT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
    cu->l1_vector().flush_all();
    l2.flush_all();
    return true;
  }
  void fill_destination(uint32_t value) {
    for (uint32_t lane = 0; lane < 3; ++lane)
      for (uint32_t reg = 0; reg < 4; ++reg)
        cu->write_vgpr(vb + 8 + reg, lane, value);
  }
};

TEST_F(Cdna2ImageAccessTest, StoreThenLoadRoundTripsActiveLanesOfAnR32Image) {
  Image image;
  image.width = 64;
  bind(image);
  for (uint32_t lane = 0; lane < 3; ++lane) {
    coordinates(lane, {lane * 16, 0, 0});
    cu->write_vgpr(vb + 8, lane, 0x55000000u + lane);
  }
  ASSERT_TRUE(issue(false));
  EXPECT_EQ(memory.read32(kBase), 0x55000000u);
  EXPECT_EQ(memory.read32(kBase + 16 * 4), 0u);
  EXPECT_EQ(memory.read32(kBase + 32 * 4), 0x55000002u);

  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true));
  for (uint32_t lane : {0u, 2u}) {
    EXPECT_EQ(cu->read_vgpr(vb + 8, lane), 0x55000000u + lane);
    EXPECT_EQ(cu->read_vgpr(vb + 9, lane), 0u);
    EXPECT_EQ(cu->read_vgpr(vb + 10, lane), 0u);
    EXPECT_EQ(cu->read_vgpr(vb + 11, lane), 1u);
  }
  EXPECT_EQ(cu->read_vgpr(vb + 8, 1), 0xdeadbeef);
}

TEST_F(Cdna2ImageAccessTest, ArrayLayersFollowRowsOfPitchElements) {
  Image image;
  image.type = amdgpu::kGfx9Image2dArray;
  image.width = 3;
  image.height = 2;
  image.depth = 2;
  bind(image);
  wf->set_exec(1);
  coordinates(0, {2, 1, 1});
  cu->write_vgpr(vb + 8, 0, 0x12345678u);
  ASSERT_TRUE(issue(false));
  EXPECT_EQ(memory.read32(kBase + ((1 * 2 + 1) * 64 + 2) * 4), 0x12345678u);
}

TEST_F(Cdna2ImageAccessTest, OneDimensionalArraysTakeTheLayerFromTheSecondCoordinate) {
  Image image;
  image.type = amdgpu::kGfx9Image1dArray;
  image.width = 4;
  image.depth = 3;
  bind(image);
  wf->set_exec(1);
  coordinates(0, {1, 2, 0});
  cu->write_vgpr(vb + 8, 0, 0xabcdu);
  ASSERT_TRUE(issue(false));
  EXPECT_EQ(memory.read32(kBase + (2 * 64 + 1) * 4), 0xabcdu);
}

TEST_F(Cdna2ImageAccessTest, DmaskPacksTheEnabledComponentsAfterTheChannelSelects) {
  Image image;
  image.data_format = kFmt8888;
  image.num_format = kUint;
  image.width = 4;
  image.selectors = kIdentity;
  bind(image);
  wf->set_exec(1);
  coordinates(0, {0, 0, 0});
  memory.write32(kBase, 0x44332211u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true, 0b1010));
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x22u);
  EXPECT_EQ(cu->read_vgpr(vb + 9, 0), 0x44u);
  EXPECT_EQ(cu->read_vgpr(vb + 10, 0), 0xdeadbeef);
}

TEST_F(Cdna2ImageAccessTest, UnormStoresConvertAndLoadsReturnFloats) {
  Image image;
  image.data_format = kFmt8888;
  image.num_format = kUnorm;
  image.width = 4;
  image.selectors = kIdentity;
  bind(image);
  wf->set_exec(1);
  coordinates(0, {1, 0, 0});
  for (uint32_t i = 0; i < 4; ++i)
    cu->write_vgpr(vb + 8 + i, 0, std::bit_cast<uint32_t>(i == 3 ? 1.0f : 0.0f));
  ASSERT_TRUE(issue(false));
  EXPECT_EQ(memory.read32(kBase + 4), 0xff000000u);
  fill_destination(0);
  ASSERT_TRUE(issue(true));
  EXPECT_EQ(cu->read_vgpr(vb + 11, 0), std::bit_cast<uint32_t>(1.0f));
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0u);
}

TEST_F(Cdna2ImageAccessTest, OutOfBoundsLanesLoadZeroAndDropTheirStores) {
  Image image;
  image.width = 2;
  bind(image);
  wf->set_exec(1);
  coordinates(0, {2, 0, 0});
  memory.write32(kBase + 2 * 4, 0x77777777u);
  cu->write_vgpr(vb + 8, 0, 0x11111111u);
  ASSERT_TRUE(issue(false));
  EXPECT_EQ(memory.read32(kBase + 2 * 4), 0x77777777u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true));
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0u);
}

TEST_F(Cdna2ImageAccessTest, NullDescriptorLoadsZero) {
  for (uint32_t i = 0; i < 8; ++i)
    cu->write_sgpr(sb + i, 0);
  wf->set_exec(1);
  coordinates(0, {0, 0, 0});
  memory.write32(kBase, 0x99999999u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true));
  for (uint32_t reg = 0; reg < 4; ++reg)
    EXPECT_EQ(cu->read_vgpr(vb + 8 + reg, 0), 0u);
}

TEST_F(Cdna2ImageAccessTest, TiledSwizzleModesStopAsUnimplemented) {
  Image image;
  image.width = 64;
  image.swizzle_mode = 9; // SW_64KB_S.
  bind(image);
  wf->set_exec(1);
  coordinates(0, {0, 0, 0});
  EXPECT_FALSE(issue(true));
  EXPECT_EQ(wf->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
}

} // namespace
