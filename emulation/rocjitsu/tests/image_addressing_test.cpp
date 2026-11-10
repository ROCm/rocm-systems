// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/image_addressing.h"
#include "rocjitsu/vm/graphics/image_address.h"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <optional>

namespace {
using namespace rocjitsu;

constexpr uint64_t kBase = 0x10000;

class AllImageAddressingTest : public testing::TestWithParam<rj_code_arch_t> {
protected:
  rj_code_arch_t arch() const { return GetParam(); }
};

TEST_P(AllImageAddressingTest, MisalignedLinearPitchNeedsQualifiedRounding) {
  const amdgpu::ImageAddressConfig config = amdgpu::default_image_address_config(arch());
  const amdgpu::ImageLayout layout{.width = 30, .height = 5, .pitch = 35, .element_bytes = 4};
  const std::optional<amdgpu::ImageAddressing> addressing =
      amdgpu::ImageAddressing::prepare(config, layout);
  const bool qualified = arch() == ROCJITSU_CODE_ARCH_RDNA3 || arch() == ROCJITSU_CODE_ARCH_RDNA4;
  ASSERT_EQ(addressing.has_value(), qualified);
  if (addressing) {
    EXPECT_EQ(addressing->byte_offset(0, 1, 0), 256u);
  }
}

TEST_P(AllImageAddressingTest, NativeAddressingMatchesIndependentOffsets) {
  // Offline oracle: ROCr AddrLib at a10dca087065a82f15ea89dc7d6fdd3c6bb2c246,
  // projects/rocr-runtime/runtime/hsa-runtime/image/addrlib.
  // AddrCreate uses the Arctic Islands engine, useTileIndex=1, and these
  // (chipFamily, chipRevision, gbAddrConfig) values:
  //   GFX9     (0x8d, 0,    0)
  //   RDNA1    (0x8f, 1,    0x44)
  //   RDNA2    (0x8f, 0x28, 0x444)  Navi21 RB+
  //   RDNA3    (0x91, 1,    0x545)
  //   RDNA3.5  (0x96, 0,    0)
  //   RDNA4    (0x98, 0,    0x8200545)
  // Read addr from Addr2ComputeSurfaceAddrFromCoord (Addr3 on RDNA4).
  // Set texture=1, needEquation=1 for Addr2, and view3dAs2dArray for 3D.
  // Use TEX_1D/2D/3D from dimensions, bpp=8*element_bytes, unaligned
  // width/height/depth from layout, one sample and fragment, automatic
  // pitch, zero XOR, and the specified mip count, mip ID and (x,y,slice).
  enum class LayoutClass { Linear, Tiled, LargeBlock, Volume };
  struct Offsets {
    uint64_t gfx9, rdna1, rdna3, rdna3_5, rdna4, rdna2;
  };
  struct Case {
    LayoutClass layout_class;
    amdgpu::ImageLayout layout;
    uint32_t mip, x, y, z;
    Offsets expected;
  };
  constexpr Case cases[] = {
      {.layout_class = LayoutClass::Tiled,
       .layout = {.width = 130,
                  .height = 65,
                  .depth = 3,
                  .element_bytes = 1,
                  .dimensions = 2,
                  .mip_levels = 1},
       .mip = 0,
       .x = 129,
       .y = 64,
       .z = 2,
       .expected = {.gfx9 = 69633ull,
                    .rdna1 = 72961ull,
                    .rdna3 = 72961ull,
                    .rdna3_5 = 69633ull,
                    .rdna4 = 69633ull,
                    .rdna2 = 72961ull}},
      {.layout_class = LayoutClass::LargeBlock,
       .layout = {.width = 257,
                  .height = 129,
                  .depth = 3,
                  .element_bytes = 1,
                  .dimensions = 2,
                  .mip_levels = 1},
       .mip = 0,
       .x = 256,
       .y = 128,
       .z = 2,
       .expected = {.gfx9 = 344064ull,
                    .rdna1 = 345088ull,
                    .rdna3 = 706048ull,
                    .rdna3_5 = 628736ull,
                    .rdna4 = 671744ull,
                    .rdna2 = 327936ull}},
      {.layout_class = LayoutClass::Volume,
       .layout = {.width = 33,
                  .height = 17,
                  .depth = 5,
                  .element_bytes = 1,
                  .dimensions = 3,
                  .mip_levels = 1},
       .mip = 0,
       .x = 32,
       .y = 16,
       .z = 4,
       .expected = {.gfx9 = 18688ull,
                    .rdna1 = 264320ull,
                    .rdna3 = 264960ull,
                    .rdna3_5 = 263680ull,
                    .rdna4 = 18688ull,
                    .rdna2 = 265984ull}},
      {.layout_class = LayoutClass::Tiled,
       .layout = {.width = 130,
                  .height = 65,
                  .depth = 3,
                  .element_bytes = 2,
                  .dimensions = 2,
                  .mip_levels = 1},
       .mip = 0,
       .x = 129,
       .y = 64,
       .z = 2,
       .expected = {.gfx9 = 106498ull,
                    .rdna1 = 108290ull,
                    .rdna3 = 108290ull,
                    .rdna3_5 = 106498ull,
                    .rdna4 = 106498ull,
                    .rdna2 = 108290ull}},
      {.layout_class = LayoutClass::LargeBlock,
       .layout = {.width = 257,
                  .height = 129,
                  .depth = 3,
                  .element_bytes = 2,
                  .dimensions = 2,
                  .mip_levels = 1},
       .mip = 0,
       .x = 256,
       .y = 128,
       .z = 2,
       .expected = {.gfx9 = 720896ull,
                    .rdna1 = 721920ull,
                    .rdna3 = 755200ull,
                    .rdna3_5 = 723968ull,
                    .rdna4 = 720896ull,
                    .rdna2 = 737536ull}},
      {.layout_class = LayoutClass::Volume,
       .layout = {.width = 33,
                  .height = 17,
                  .depth = 5,
                  .element_bytes = 2,
                  .dimensions = 3,
                  .mip_levels = 1},
       .mip = 0,
       .x = 32,
       .y = 16,
       .z = 4,
       .expected = {.gfx9 = 19456ull,
                    .rdna1 = 268288ull,
                    .rdna3 = 264960ull,
                    .rdna3_5 = 265216ull,
                    .rdna4 = 19456ull,
                    .rdna2 = 265984ull}},
      {.layout_class = LayoutClass::Tiled,
       .layout = {.width = 130,
                  .height = 65,
                  .depth = 3,
                  .element_bytes = 4,
                  .dimensions = 2,
                  .mip_levels = 1},
       .mip = 0,
       .x = 129,
       .y = 64,
       .z = 2,
       .expected = {.gfx9 = 180228ull,
                    .rdna1 = 181764ull,
                    .rdna3 = 181764ull,
                    .rdna3_5 = 180228ull,
                    .rdna4 = 180228ull,
                    .rdna2 = 181764ull}},
      {.layout_class = LayoutClass::LargeBlock,
       .layout = {.width = 257,
                  .height = 129,
                  .depth = 3,
                  .element_bytes = 4,
                  .dimensions = 2,
                  .mip_levels = 1},
       .mip = 0,
       .x = 256,
       .y = 128,
       .z = 2,
       .expected = {.gfx9 = 1114112ull,
                    .rdna1 = 1115136ull,
                    .rdna3 = 1443328ull,
                    .rdna3_5 = 1442816ull,
                    .rdna4 = 1376256ull,
                    .rdna2 = 1130752ull}},
      {.layout_class = LayoutClass::Volume,
       .layout = {.width = 33,
                  .height = 17,
                  .depth = 5,
                  .element_bytes = 4,
                  .dimensions = 3,
                  .mip_levels = 1},
       .mip = 0,
       .x = 32,
       .y = 16,
       .z = 4,
       .expected = {.gfx9 = 37888ull,
                    .rdna1 = 264192ull,
                    .rdna3 = 264960ull,
                    .rdna3_5 = 268288ull,
                    .rdna4 = 37888ull,
                    .rdna2 = 265984ull}},
      {.layout_class = LayoutClass::Tiled,
       .layout = {.width = 130,
                  .height = 65,
                  .depth = 3,
                  .element_bytes = 8,
                  .dimensions = 2,
                  .mip_levels = 1},
       .mip = 0,
       .x = 129,
       .y = 64,
       .z = 2,
       .expected = {.gfx9 = 303112ull,
                    .rdna1 = 304136ull,
                    .rdna3 = 304136ull,
                    .rdna3_5 = 303112ull,
                    .rdna4 = 303112ull,
                    .rdna2 = 304136ull}},
      {.layout_class = LayoutClass::LargeBlock,
       .layout = {.width = 257,
                  .height = 129,
                  .depth = 3,
                  .element_bytes = 8,
                  .dimensions = 2,
                  .mip_levels = 1},
       .mip = 0,
       .x = 256,
       .y = 128,
       .z = 2,
       .expected = {.gfx9 = 1703936ull,
                    .rdna1 = 1704960ull,
                    .rdna3 = 3118080ull,
                    .rdna3_5 = 2885632ull,
                    .rdna4 = 2883584ull,
                    .rdna2 = 1704192ull}},
      {.layout_class = LayoutClass::Volume,
       .layout = {.width = 33,
                  .height = 17,
                  .depth = 5,
                  .element_bytes = 8,
                  .dimensions = 3,
                  .mip_levels = 1},
       .mip = 0,
       .x = 32,
       .y = 16,
       .z = 4,
       .expected = {.gfx9 = 77824ull,
                    .rdna1 = 280576ull,
                    .rdna3 = 265984ull,
                    .rdna3_5 = 274432ull,
                    .rdna4 = 77824ull,
                    .rdna2 = 265984ull}},
      {.layout_class = LayoutClass::Tiled,
       .layout = {.width = 130,
                  .height = 65,
                  .depth = 3,
                  .element_bytes = 16,
                  .dimensions = 2,
                  .mip_levels = 1},
       .mip = 0,
       .x = 129,
       .y = 64,
       .z = 2,
       .expected = {.gfx9 = 548880ull,
                    .rdna1 = 549904ull,
                    .rdna3 = 549904ull,
                    .rdna3_5 = 548880ull,
                    .rdna4 = 548880ull,
                    .rdna2 = 549904ull}},
      {.layout_class = LayoutClass::LargeBlock,
       .layout = {.width = 257,
                  .height = 129,
                  .depth = 3,
                  .element_bytes = 16,
                  .dimensions = 2,
                  .mip_levels = 1},
       .mip = 0,
       .x = 256,
       .y = 128,
       .z = 2,
       .expected = {.gfx9 = 2883584ull,
                    .rdna1 = 2884608ull,
                    .rdna3 = 4658176ull,
                    .rdna3_5 = 4457472ull,
                    .rdna4 = 4456448ull,
                    .rdna2 = 2886912ull}},
      {.layout_class = LayoutClass::Volume,
       .layout = {.width = 33,
                  .height = 17,
                  .depth = 5,
                  .element_bytes = 16,
                  .dimensions = 3,
                  .mip_levels = 1},
       .mip = 0,
       .x = 32,
       .y = 16,
       .z = 4,
       .expected = {.gfx9 = 118784ull,
                    .rdna1 = 264192ull,
                    .rdna3 = 264448ull,
                    .rdna3_5 = 278528ull,
                    .rdna4 = 118784ull,
                    .rdna2 = 264448ull}},
      {.layout_class = LayoutClass::Tiled,
       .layout = {.width = 37,
                  .height = 19,
                  .depth = 3,
                  .element_bytes = 4,
                  .dimensions = 2,
                  .mip_levels = 6},
       .mip = 0,
       .x = 36,
       .y = 18,
       .z = 2,
       .expected = {.gfx9 = 37984ull,
                    .rdna1 = 46240ull,
                    .rdna3 = 46176ull,
                    .rdna3_5 = 46176ull,
                    .rdna4 = 46176ull,
                    .rdna2 = 46176ull}},
      {.layout_class = LayoutClass::Tiled,
       .layout = {.width = 37,
                  .height = 19,
                  .depth = 3,
                  .element_bytes = 4,
                  .dimensions = 2,
                  .mip_levels = 6},
       .mip = 1,
       .x = 17,
       .y = 8,
       .z = 2,
       .expected = {.gfx9 = 43268ull,
                    .rdna1 = 40196ull,
                    .rdna3 = 40196ull,
                    .rdna3_5 = 39172ull,
                    .rdna4 = 39172ull,
                    .rdna2 = 40196ull}},
      {.layout_class = LayoutClass::Tiled,
       .layout = {.width = 37,
                  .height = 19,
                  .depth = 3,
                  .element_bytes = 4,
                  .dimensions = 2,
                  .mip_levels = 6},
       .mip = 2,
       .x = 8,
       .y = 3,
       .z = 2,
       .expected = {.gfx9 = 47696ull,
                    .rdna1 = 36400ull,
                    .rdna3 = 36392ull,
                    .rdna3_5 = 35368ull,
                    .rdna4 = 35368ull,
                    .rdna2 = 36400ull}},
      {.layout_class = LayoutClass::Tiled,
       .layout = {.width = 37,
                  .height = 19,
                  .depth = 3,
                  .element_bytes = 4,
                  .dimensions = 2,
                  .mip_levels = 6},
       .mip = 3,
       .x = 3,
       .y = 1,
       .z = 2,
       .expected = {.gfx9 = 46620ull,
                    .rdna1 = 33308ull,
                    .rdna3 = 33308ull,
                    .rdna3_5 = 34332ull,
                    .rdna4 = 34332ull,
                    .rdna2 = 33308ull}},
      {.layout_class = LayoutClass::Tiled,
       .layout = {.width = 37,
                  .height = 19,
                  .depth = 3,
                  .element_bytes = 4,
                  .dimensions = 2,
                  .mip_levels = 6},
       .mip = 4,
       .x = 1,
       .y = 0,
       .z = 2,
       .expected = {.gfx9 = 46340ull,
                    .rdna1 = 33028ull,
                    .rdna3 = 33028ull,
                    .rdna3_5 = 34052ull,
                    .rdna4 = 34052ull,
                    .rdna2 = 33028ull}},
      {.layout_class = LayoutClass::Tiled,
       .layout = {.width = 37,
                  .height = 19,
                  .depth = 3,
                  .element_bytes = 4,
                  .dimensions = 2,
                  .mip_levels = 6},
       .mip = 5,
       .x = 0,
       .y = 0,
       .z = 2,
       .expected = {.gfx9 = 46080ull,
                    .rdna1 = 32768ull,
                    .rdna3 = 32768ull,
                    .rdna3_5 = 33792ull,
                    .rdna4 = 33792ull,
                    .rdna2 = 32768ull}},
      {.layout_class = LayoutClass::Linear,
       .layout = {.width = 257,
                  .height = 1,
                  .depth = 3,
                  .element_bytes = 4,
                  .dimensions = 1,
                  .mip_levels = 6},
       .mip = 0,
       .x = 256,
       .y = 0,
       .z = 2,
       .expected = {.gfx9 = 16384ull,
                    .rdna1 = 9728ull,
                    .rdna3 = 9728ull,
                    .rdna3_5 = 9728ull,
                    .rdna4 = 9728ull,
                    .rdna2 = 9728ull}},
      {.layout_class = LayoutClass::Linear,
       .layout = {.width = 257,
                  .height = 1,
                  .depth = 3,
                  .element_bytes = 4,
                  .dimensions = 1,
                  .mip_levels = 6},
       .mip = 2,
       .x = 63,
       .y = 0,
       .z = 2,
       .expected = {.gfx9 = 18172ull,
                    .rdna1 = 7676ull,
                    .rdna3 = 7676ull,
                    .rdna3_5 = 7676ull,
                    .rdna4 = 7676ull,
                    .rdna2 = 7676ull}},
      {.layout_class = LayoutClass::Linear,
       .layout = {.width = 257,
                  .height = 1,
                  .depth = 3,
                  .element_bytes = 4,
                  .dimensions = 1,
                  .mip_levels = 6},
       .mip = 5,
       .x = 7,
       .y = 0,
       .z = 2,
       .expected = {.gfx9 = 21788ull,
                    .rdna1 = 6684ull,
                    .rdna3 = 6684ull,
                    .rdna3_5 = 6684ull,
                    .rdna4 = 6684ull,
                    .rdna2 = 6684ull}},
      {.layout_class = LayoutClass::Volume,
       .layout = {.width = 130,
                  .height = 129,
                  .depth = 17,
                  .element_bytes = 2,
                  .dimensions = 3,
                  .mip_levels = 6},
       .mip = 0,
       .x = 129,
       .y = 128,
       .z = 16,
       .expected = {.gfx9 = 1728514ull,
                    .rdna1 = 3309570ull,
                    .rdna3 = 4444162ull,
                    .rdna3_5 = 4429826ull,
                    .rdna4 = 1667074ull,
                    .rdna2 = 3279874ull}},
      {.layout_class = LayoutClass::Volume,
       .layout = {.width = 130,
                  .height = 129,
                  .depth = 17,
                  .element_bytes = 2,
                  .dimensions = 3,
                  .mip_levels = 6},
       .mip = 2,
       .x = 31,
       .y = 31,
       .z = 3,
       .expected = {.gfx9 = 382974ull,
                    .rdna1 = 611582ull,
                    .rdna3 = 817150ull,
                    .rdna3_5 = 825342ull,
                    .rdna4 = 301054ull,
                    .rdna2 = 596478ull}},
      {.layout_class = LayoutClass::Volume,
       .layout = {.width = 130,
                  .height = 129,
                  .depth = 17,
                  .element_bytes = 2,
                  .dimensions = 3,
                  .mip_levels = 6},
       .mip = 4,
       .x = 7,
       .y = 7,
       .z = 0,
       .expected = {.gfx9 = 79486ull,
                    .rdna1 = 1150ull,
                    .rdna3 = 32894ull,
                    .rdna3_5 = 8318ull,
                    .rdna4 = 1662ull,
                    .rdna2 = 1150ull}},
  };

  // AddrLib swizzle enum values, kept local to this offline reference.
  constexpr uint32_t kLinear = 0;
  constexpr uint32_t k4kbDisplayX = 22;
  constexpr uint32_t k64kbZX = 24;
  constexpr uint32_t k64kbDisplayX = 26;
  constexpr uint32_t k64kbRenderX = 27;
  constexpr uint32_t k256kbRenderX = 31;
  constexpr uint32_t kGfx12_4kb2d = 2;
  constexpr uint32_t kGfx12_256kb2d = 4;
  struct Column {
    uint64_t Offsets::*offset;
    // GFX11 Z swizzles reject 16-byte texels, so those volumes use a render swizzle.
    uint32_t tiled, large_block, volume, volume_16_byte;
  };
  Column column;
  switch (arch()) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
    column = {&Offsets::gfx9, k4kbDisplayX, k64kbDisplayX, k4kbDisplayX, k4kbDisplayX};
    break;
  case ROCJITSU_CODE_ARCH_RDNA1:
    column = {&Offsets::rdna1, k4kbDisplayX, k64kbRenderX, k64kbRenderX, k64kbRenderX};
    break;
  case ROCJITSU_CODE_ARCH_RDNA2:
    column = {&Offsets::rdna2, k4kbDisplayX, k64kbRenderX, k64kbRenderX, k64kbRenderX};
    break;
  case ROCJITSU_CODE_ARCH_RDNA3:
    column = {&Offsets::rdna3, k4kbDisplayX, k256kbRenderX, k64kbZX, k64kbRenderX};
    break;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    column = {&Offsets::rdna3_5, k4kbDisplayX, k256kbRenderX, k64kbZX, k64kbRenderX};
    break;
  case ROCJITSU_CODE_ARCH_RDNA4:
    column = {&Offsets::rdna4, kGfx12_4kb2d, kGfx12_256kb2d, kGfx12_4kb2d, kGfx12_4kb2d};
    break;
  default:
    FAIL() << "Unexpected image architecture";
  }
  const amdgpu::ImageAddressConfig config = amdgpu::default_image_address_config(arch());
  for (const Case &test : cases) {
    amdgpu::ImageLayout layout = test.layout;
    switch (test.layout_class) {
    case LayoutClass::Linear:
      layout.swizzle_mode = kLinear;
      break;
    case LayoutClass::Tiled:
      layout.swizzle_mode = column.tiled;
      break;
    case LayoutClass::LargeBlock:
      layout.swizzle_mode = column.large_block;
      break;
    case LayoutClass::Volume:
      layout.swizzle_mode = layout.element_bytes < 16 ? column.volume : column.volume_16_byte;
      break;
    }
    SCOPED_TRACE(testing::Message()
                 << layout.swizzle_mode << ':' << layout.element_bytes << ':' << test.mip);
    const std::optional<amdgpu::ImageAddressing> addressing =
        amdgpu::ImageAddressing::prepare(config, layout, test.mip);
    ASSERT_TRUE(addressing.has_value());
    ASSERT_TRUE(addressing->contains(test.x, test.y, test.z));
    EXPECT_EQ(addressing->byte_offset(test.x, test.y, test.z), test.expected.*column.offset);
  }
}

TEST_P(AllImageAddressingTest, NativeAddressingRejectsUnsupportedLayouts) {
  const amdgpu::ImageAddressConfig config = amdgpu::default_image_address_config(arch());
  amdgpu::ImageLayout layout{.width = 37, .height = 19, .depth = 3, .element_bytes = 4};
  EXPECT_FALSE(amdgpu::ImageAddressing::prepare(config, layout, 1));
  layout.element_bytes = 3;
  EXPECT_FALSE(amdgpu::ImageAddressing::prepare(config, layout));
  layout.element_bytes = 4;
  layout.mip_levels = 0;
  EXPECT_FALSE(amdgpu::ImageAddressing::prepare(config, layout));
  layout.mip_levels = 18;
  EXPECT_FALSE(amdgpu::ImageAddressing::prepare(config, layout));
  layout.mip_levels = 1;
  layout.swizzle_mode = 8;
  EXPECT_FALSE(amdgpu::ImageAddressing::prepare(config, layout));
}

TEST_P(AllImageAddressingTest, NativeAddressingSeparatesViewBoundsFromLayoutSupport) {
  const amdgpu::ImageAddressConfig config = amdgpu::default_image_address_config(arch());
  const amdgpu::ImageLayout layout{.width = 37, .height = 19, .depth = 3, .element_bytes = 4};
  const std::optional<amdgpu::ImageAddressing> addressing =
      amdgpu::ImageAddressing::prepare(config, layout);
  ASSERT_TRUE(addressing);
  EXPECT_TRUE(addressing->contains(36, 18, 2));
  EXPECT_FALSE(addressing->contains(37, 0, 0));
  EXPECT_FALSE(addressing->contains(0, 19, 0));
  EXPECT_FALSE(addressing->contains(0, 0, 3));
}

TEST_P(AllImageAddressingTest, BaseAlignmentDependsOnTheStorageBlock) {
  const amdgpu::ImageAddressConfig config = amdgpu::default_image_address_config(arch());
  amdgpu::ImageLayout layout{.element_bytes = 4};
  std::optional<amdgpu::ImageAddressing> addressing =
      amdgpu::ImageAddressing::prepare(config, layout);
  ASSERT_TRUE(addressing);
  EXPECT_TRUE(addressing->has_supported_base(kBase + 256));
  layout.swizzle_mode = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 2 : 22;
  addressing = amdgpu::ImageAddressing::prepare(config, layout);
  ASSERT_TRUE(addressing);
  EXPECT_TRUE(addressing->has_supported_base(kBase));
  EXPECT_FALSE(addressing->has_supported_base(kBase + 256));
}

TEST(ImageAddressingTest, Gfx11ZModeMipTailUsesThe32BitDepthThresholdOnBothAxes) {
  const amdgpu::ImageAddressConfig config{ROCJITSU_CODE_ARCH_RDNA3, 0x545};
  for (const auto swizzle : {24u, 28u}) {
    // Scaled 64 KiB and 256 KiB layouts have the same mip-tail transition.
    const uint32_t scale = swizzle == 24 ? 1 : 2;
    const uint32_t width = 64 * scale, height = 256 * scale;
    const amdgpu::ImageLayout layout{.width = width,
                                     .height = height,
                                     .element_bytes = 2,
                                     .swizzle_mode = swizzle,
                                     .mip_levels = 9};
    for (unsigned level : {0u, 1u, 2u, 3u}) {
      const auto mip = amdgpu::image_mip_layout(false, swizzle, 2, width, height, 9, level);
      const auto addressing = amdgpu::ImageAddressing::prepare(config, layout, level);
      ASSERT_TRUE(mip);
      ASSERT_TRUE(addressing);
      EXPECT_EQ(mip->first_tail, 2u);
      const auto address = amdgpu::gfx11_image_address(mip->offset, mip->tail_x, mip->tail_y,
                                                       mip->pitch, 2, swizzle);
      ASSERT_TRUE(address);
      EXPECT_EQ(*address, addressing->byte_offset(0, 0, 0));
      if (swizzle == 24) {
        constexpr std::array offsets{0x20000u, 0x10000u, 0x5000u, 0x1000u};
        EXPECT_EQ(*address, offsets[level]);
      }
    }
  }
}

TEST(ImageAddressingTest, Navi10ZModeMipTailMatchesIndependentOffsets) {
  // Same Addr2 recipe as NativeAddressingMatchesIndependentOffsets, using
  // its RDNA1 (Navi10) column and ADDR_SW_64KB_Z_X.
  const amdgpu::ImageAddressConfig config{ROCJITSU_CODE_ARCH_RDNA1, 0x44};
  for (const auto &[bytes, mip, offset] :
       {std::array{1u, 0u, 32768u}, std::array{2u, 1u, 18432u}, std::array{2u, 3u, 1024u}}) {
    const amdgpu::ImageLayout layout{
        .width = 67, .height = 45, .element_bytes = bytes, .swizzle_mode = 24, .mip_levels = 7};
    const std::optional<amdgpu::ImageAddressing> addressing =
        amdgpu::ImageAddressing::prepare(config, layout, mip);
    ASSERT_TRUE(addressing);
    EXPECT_EQ(addressing->byte_offset(0, 0, 0), offset);
  }
}

TEST(ImageAddressingTest, UnqualifiedCdnaTilingRegistersAreRejected) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2}) {
    const amdgpu::ImageAddressConfig config{arch, 0x82004};
    const amdgpu::ImageLayout layout{.element_bytes = 4, .swizzle_mode = 22};
    EXPECT_FALSE(amdgpu::ImageAddressing::prepare(config, layout));
  }
}

TEST(ImageAddressingTest, RejectsArchitecturesWithoutImageMemoryInstructions) {
  const amdgpu::ImageLayout layout{.element_bytes = 4};
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_INVALID, ROCJITSU_CODE_ARCH_RV32I, ROCJITSU_CODE_ARCH_RV64I,
        ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    const amdgpu::ImageAddressConfig config = amdgpu::default_image_address_config(arch);
    EXPECT_FALSE(amdgpu::ImageAddressing::prepare(config, layout));
  }
}

INSTANTIATE_TEST_SUITE_P(Architectures, AllImageAddressingTest,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));

} // namespace
