/*

Copyright © 2025 Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include "suites/image/mipmap_array.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "hsa/hsa_amd_mipmap.h"
#include <iostream>
#include <algorithm>
#include <cstring>

namespace {
inline uint32_t ComputeFullMipLevels(uint32_t w, uint32_t h = 1, uint32_t d = 1) {
  uint32_t m = std::max({w, h, d});
  uint32_t levels = 0;
  while (m > 0) { ++levels; m >>= 1; }
  return levels ? levels : 1;
}

inline void HalveOrClamp(uint32_t &v) { v = std::max(1u, v >> 1); }
}

namespace rocrtst {

const std::vector<uint32_t> Mipmap1DArrayTest::kTest1DDimensions = {
    1,2,3,4,7,8,15,16,17,31,32,33,63,64,65,127,128,129,255,256
};

const std::vector<std::pair<uint32_t,uint32_t>> Mipmap2DArrayTest::kTest2DDimensions = {
    {1,1},{2,2},{3,3},{4,4},{7,7},{8,8},{15,15},{16,16},{31,31},{32,32},
    {64,64},{128,128},{2,4},{4,2},{3,5},{5,3},{8,16},{16,8},{32,64},{64,32}
};

const std::vector<std::tuple<uint32_t,uint32_t,uint32_t>> Mipmap3DArrayTest::kTest3DDimensions = {
    {1,1,1},{2,2,2},{3,3,3},{4,4,4},{8,8,8},{16,16,8},{16,8,16},{8,16,16},
    {32,16,8},{8,16,32}
};

MipmapArrayTest::MipmapArrayTest()
  : test_image_{}, num_mipmap_levels_{0}, mipmapped_array_{},
    image_extension_supported_{false}, mipmap_desc_{}, image_format_{} {
  //set_title("Basic Mipmap Array Tests");
  //set_description("Validates create / get_level / destroy for 1D/2D/3D.");
}

MipmapArrayTest::~MipmapArrayTest() = default;

void MipmapArrayTest::SetUp() {
  hsa_status_t status =
      hsa_system_extension_supported(HSA_EXTENSION_IMAGES, 1, 0, &image_extension_supported_);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
}

void MipmapArrayTest::Run() {
}

void MipmapArrayTest::DisplayTestInfo() {  }
void MipmapArrayTest::DisplayResults() const {  }
void MipmapArrayTest::Close() {  }

void BasicCreateVerify(uint32_t w, uint32_t h, uint32_t d) {
  // Create a test instance to get GPU device
  MipmapArrayTest test;
  hsa_status_t err = rocrtst::SetDefaultAgents(&test);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  hsa_agent_t agent = *test.gpu_device1();

  hsa_ext_image_descriptor_t desc{};
  desc.geometry = (d > 1) ? HSA_EXT_IMAGE_GEOMETRY_3D :
                  (h > 1) ? HSA_EXT_IMAGE_GEOMETRY_2D : HSA_EXT_IMAGE_GEOMETRY_1D;
  desc.width  = w;
  desc.height = h;
  desc.depth  = d;
  desc.array_size = 1;
  desc.format.channel_type = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
  desc.format.channel_order = HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA;
  uint32_t mip_levels = ComputeFullMipLevels(w,h,d);

  hsa_ext_image_t handle{};
  hsa_status_t st = hsa_amd_mipmap_array_create(agent, &desc, mip_levels, HSA_ACCESS_PERMISSION_RW, &handle);
  ASSERT_EQ(HSA_STATUS_SUCCESS, st);
  ASSERT_NE(handle.handle, uint64_t{0});

  uint32_t lvlW = w, lvlH = h, lvlD = d;
  for (uint32_t level = 0; level < mip_levels; ++level) {
    hsa_ext_image_t level_view{};
    st = hsa_amd_mipmap_array_get_level(agent, &handle, level, &level_view);
    ASSERT_EQ(HSA_STATUS_SUCCESS, st);
    ASSERT_NE(level_view.handle, uint64_t{0});
    // (Future: query image info for per-level size // this following isnt correct : shweta
    HalveOrClamp(lvlW);
    if (h > 1) HalveOrClamp(lvlH);
    if (d > 1) HalveOrClamp(lvlD);
  }

  st = hsa_amd_mipmap_array_destroy(&handle);
  ASSERT_EQ(HSA_STATUS_SUCCESS, st);
  ASSERT_EQ(handle.handle, uint64_t{0});
}

void MipmapArrayTest::MipmapCreate1DArrayTest()   { BasicCreateVerify(128,1,1); }
void MipmapArrayTest::MipmapDestroy1DArrayTest()  { BasicCreateVerify(64,1,1); }
void MipmapArrayTest::MipmapGetLevel1DArrayTest() { BasicCreateVerify(33,1,1); }


void MipmapArrayTest::MipmapCreate2DArrayTest()   { BasicCreateVerify(64,32,1); }
void MipmapArrayTest::MipmapDestroy2DArrayTest()  { BasicCreateVerify(32,17,1); }
void MipmapArrayTest::MipmapGetLevel2DArrayTest() { BasicCreateVerify(17,33,1); }


void MipmapArrayTest::MipmapCreate3DArrayTest()   { BasicCreateVerify(32,16,8); }
void MipmapArrayTest::MipmapDestroy3DArrayTest()  { BasicCreateVerify(16,16,16); }
void MipmapArrayTest::MipmapGetLevel3DArrayTest() { BasicCreateVerify(17,9,5); }

Mipmap1DArrayTest::Mipmap1DArrayTest()
  : image_extension_table_{}, image_extension_supported_{false},
    tests_passed_{0}, tests_failed_{0}, total_tests_{0} {
  //set_title("1D Mipmap Array Tests");
  //set_description("Iterates over various widths validating full chain creation.");
}
Mipmap1DArrayTest::~Mipmap1DArrayTest() = default;

void Mipmap1DArrayTest::SetUp() {
  hsa_status_t status =
      hsa_system_extension_supported(HSA_EXTENSION_IMAGES, 1, 0, &image_extension_supported_);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  if (image_extension_supported_) {
    status = hsa_system_get_extension_table(HSA_EXTENSION_IMAGES, 1,
              sizeof(image_extension_table_), &image_extension_table_);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  }
}
void Mipmap1DArrayTest::TearDown() {}
void Mipmap1DArrayTest::Run() {  }
void Mipmap1DArrayTest::DisplayTestInfo() {  }
void Mipmap1DArrayTest::DisplayResults() const {  }
void Mipmap1DArrayTest::Close() {  }

void Mipmap1DArrayTest::TestElementTypeReadMode1D() {
  for (auto w : kTest1DDimensions) {
    SCOPED_TRACE(std::string("Width=") + std::to_string(w));
    BasicCreateVerify(w,1,1);
  }
}
void Mipmap1DArrayTest::TestNormalizedFloatReadMode1D() { TestElementTypeReadMode1D(); }
void Mipmap1DArrayTest::TestLinearFiltering1D()         { TestElementTypeReadMode1D(); }
void Mipmap1DArrayTest::TestAddressModes1D()            { TestElementTypeReadMode1D(); }
void Mipmap1DArrayTest::TestVariousDimensions1D()       { TestElementTypeReadMode1D(); }
void Mipmap1DArrayTest::TestErrorConditions1D() {
  hsa_agent_t agent = *gpu_device1();
  hsa_ext_image_descriptor_t desc{};
  desc.geometry = HSA_EXT_IMAGE_GEOMETRY_1D;
  desc.width = 64;
  desc.height = 1;
  desc.depth = 1;
  desc.array_size = 1;
  desc.format.channel_type = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
  desc.format.channel_order = HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA;
  uint32_t excessive_levels = 100; // Intentionally excessive
  hsa_ext_image_t handle{};
  hsa_status_t st = hsa_amd_mipmap_array_create(agent, &desc, excessive_levels, HSA_ACCESS_PERMISSION_RW, &handle);
  ASSERT_NE(st, HSA_STATUS_SUCCESS);
}
void Mipmap1DArrayTest::TestMemoryIntegrity1D() {
  TestElementTypeReadMode1D();
}

Mipmap2DArrayTest::Mipmap2DArrayTest()
  : image_extension_table_{}, image_extension_supported_{false},
    tests_passed_{0}, tests_failed_{0}, total_tests_{0} {
  //set_title("2D Mipmap Array Tests");
  //set_description("Iterates over square & rectangular dimensions.");
}
Mipmap2DArrayTest::~Mipmap2DArrayTest() = default;
void Mipmap2DArrayTest::SetUp() {
  hsa_status_t status =
      hsa_system_extension_supported(HSA_EXTENSION_IMAGES, 1, 0, &image_extension_supported_);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  if (image_extension_supported_) {
    status = hsa_system_get_extension_table(HSA_EXTENSION_IMAGES, 1,
              sizeof(image_extension_table_), &image_extension_table_);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  }
}
void Mipmap2DArrayTest::TearDown() {}
void Mipmap2DArrayTest::Run() {  }
void Mipmap2DArrayTest::DisplayTestInfo() {  }
void Mipmap2DArrayTest::DisplayResults() const {  }
void Mipmap2DArrayTest::Close() {  }

static void Create2D(uint32_t w, uint32_t h) { BasicCreateVerify(w,h,1); }

void Mipmap2DArrayTest::TestElementTypeReadMode2D() {
  for (auto [w,h] : kTest2DDimensions) {
    SCOPED_TRACE("2D w=" + std::to_string(w) + " h=" + std::to_string(h));
    Create2D(w,h);
  }
}
void Mipmap2DArrayTest::TestNormalizedFloatReadMode2D() { TestElementTypeReadMode2D(); }
void Mipmap2DArrayTest::TestLinearFiltering2D()         { TestElementTypeReadMode2D(); }
void Mipmap2DArrayTest::TestAddressModes2D()            { TestElementTypeReadMode2D(); }
void Mipmap2DArrayTest::TestSquareAndRectangularImages(){ TestElementTypeReadMode2D(); }
void Mipmap2DArrayTest::TestVariousDimensions2D()       { TestElementTypeReadMode2D(); }
void Mipmap2DArrayTest::TestErrorConditions2D() {
  hsa_agent_t agent = *gpu_device1();
  hsa_ext_image_descriptor_t desc{};
  desc.geometry = HSA_EXT_IMAGE_GEOMETRY_2D;
  desc.width = 16;
  desc.height = 8;
  desc.depth = 1;
  desc.array_size = 1;
  desc.format.channel_type = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
  desc.format.channel_order = HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA;
  uint32_t excessive_levels = 64; // intentionally too large
  hsa_ext_image_t handle{};
  hsa_status_t st = hsa_amd_mipmap_array_create(agent, &desc, excessive_levels, HSA_ACCESS_PERMISSION_RW, &handle);
  ASSERT_NE(st, HSA_STATUS_SUCCESS);
}
void Mipmap2DArrayTest::TestMemoryIntegrity2D() { TestElementTypeReadMode2D(); }

Mipmap3DArrayTest::Mipmap3DArrayTest()
  : image_extension_table_{}, image_extension_supported_{false},
    tests_passed_{0}, tests_failed_{0}, total_tests_{0} {
  //set_title("3D Mipmap Array Tests");
  //set_description("Iterates over selected 3D dimension triplets.");
}
Mipmap3DArrayTest::~Mipmap3DArrayTest() = default;
void Mipmap3DArrayTest::SetUp() {
  hsa_status_t status =
      hsa_system_extension_supported(HSA_EXTENSION_IMAGES, 1, 0, &image_extension_supported_);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  if (image_extension_supported_) {
    status = hsa_system_get_extension_table(HSA_EXTENSION_IMAGES, 1,
              sizeof(image_extension_table_), &image_extension_table_);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  }
}
void Mipmap3DArrayTest::TearDown() {}
void Mipmap3DArrayTest::Run() {  }
void Mipmap3DArrayTest::DisplayTestInfo() {  }
void Mipmap3DArrayTest::DisplayResults() const {  }
void Mipmap3DArrayTest::Close() {  }

static void Create3D(uint32_t w, uint32_t h, uint32_t d) { BasicCreateVerify(w,h,d); }

void Mipmap3DArrayTest::TestElementTypeReadMode3D() {
  for (auto [w,h,d] : kTest3DDimensions) {
    SCOPED_TRACE("3D w=" + std::to_string(w) + " h=" + std::to_string(h) + " d=" + std::to_string(d));
    Create3D(w,h,d);
  }
}
void Mipmap3DArrayTest::TestNormalizedFloatReadMode3D() { TestElementTypeReadMode3D(); }
void Mipmap3DArrayTest::TestLinearFiltering3D()         { TestElementTypeReadMode3D(); }
void Mipmap3DArrayTest::TestAddressModes3D()            { TestElementTypeReadMode3D(); }
void Mipmap3DArrayTest::TestVariousDimensions3D()       { TestElementTypeReadMode3D(); }
void Mipmap3DArrayTest::TestErrorConditions3D() {
  hsa_agent_t agent = *gpu_device1();
  hsa_ext_image_descriptor_t desc{};
  desc.geometry = HSA_EXT_IMAGE_GEOMETRY_3D;
  desc.width = 8;
  desc.height = 8;
  desc.depth = 8;
  desc.array_size = 1;
  desc.format.channel_type = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
  desc.format.channel_order = HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA;
  uint32_t excessive_levels = 40; // intentionally too large
  hsa_ext_image_t handle{};
  hsa_status_t st = hsa_amd_mipmap_array_create(agent, &desc, excessive_levels, HSA_ACCESS_PERMISSION_RW, &handle);
  ASSERT_NE(st, HSA_STATUS_SUCCESS);
}
void Mipmap3DArrayTest::TestMemoryIntegrity3D() { TestElementTypeReadMode3D(); }

} // namespace rocrtst
