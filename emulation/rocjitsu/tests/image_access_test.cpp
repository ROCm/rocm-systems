// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/mimg.h"
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
#include "rocjitsu/vm/amdgpu/image_address.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace {
using namespace rocjitsu;

constexpr uint32_t kIdentity = 4 | (5 << 3) | (6 << 6) | (7 << 9);
constexpr uint32_t kXOne = 4 | (1 << 9); // X, 0, 0, 1 (an R-only format).
constexpr uint64_t kBase = 0x10000;
constexpr uint32_t kDfmt32 = 4;
constexpr uint32_t kDfmt8888 = 10;
constexpr uint32_t kNfmtUint = 4;
constexpr uint32_t kNfmtUnorm = 0;

struct Image {
  uint32_t data_format = kDfmt32;
  uint32_t num_format = kNfmtUint;
  uint32_t width = 1, height = 1, depth = 1, pitch = 64;
  uint32_t type = amdgpu::kGfx9Image2d;
  uint32_t selectors = kXOne;
  uint32_t swizzle_mode = 0;
  uint32_t mip_levels = 1;
  uint32_t level = 0;
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
  set(108, 4, image.level);
  set(112, 4, image.level);
  set(188, 4, image.mip_levels - 1);
  set(116, 5, image.swizzle_mode);
  set(124, 4, image.type);
  set(128, 13, image.depth - 1);
  set(141, 16, image.pitch - 1);
  return r;
}

std::array<uint32_t, 8> rdna_descriptor(const Image &image, rj_code_arch_t arch) {
  std::array<uint32_t, 8> words{};
  const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
  const auto set = [&](uint32_t bit, uint32_t count, uint64_t value) {
    for (uint32_t index = 0; index < count; ++index)
      words[(bit + index) / 32] |= ((value >> index) & 1u) << ((bit + index) % 32);
  };
  set(0, 40, kBase >> 8);
  set(gfx12 ? 49 : 52, 8, 20); // FMT_32_UINT in each RDNA format table.
  set(62, 14, image.width - 1);
  set(78, 14, image.height - 1);
  set(96, 12, kXOne);
  set(124, 4, image.type);
  set(gfx12 ? 57 : 108, gfx12 ? 5 : 4, image.level);
  set(gfx12 ? 111 : 112, gfx12 ? 5 : 4, image.level);
  const bool dimensions_in_resource =
      arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
  set(gfx12 ? 44 : dimensions_in_resource ? 164 : 48, gfx12 ? 5 : 4, image.mip_levels - 1);
  set(116, 5, image.swizzle_mode);
  if (image.type == amdgpu::kGfx9Image3d || image.type == amdgpu::kGfx9Image1dArray ||
      image.type == amdgpu::kGfx9Image2dArray)
    set(128, 13, image.depth - 1);
  else if (arch != ROCJITSU_CODE_ARCH_RDNA1)
    set(128, 14, 63); // A 256-byte row pitch.
  return words;
}

class Cdna2ImageAccessTest : public testing::Test {
protected:
  /// @brief MIMG flags used by the access fixture.
  struct Encoding {
    bool a16 = false;
    bool d16 = false;
    bool da = true;
    bool unorm = true;
  };

  amdgpu::GpuMemory memory{"image_memory"};
  amdgpu::L2Cache l2{"image_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wf = nullptr;
  uint32_t sb = 0, vb = 0;

  virtual rj_code_arch_t arch() const { return ROCJITSU_CODE_ARCH_CDNA2; }
  void SetUp() override {
    l2.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch();
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
  void write_coordinates(uint32_t lane, std::array<uint32_t, 3> coords) {
    for (uint32_t i = 0; i < coords.size(); ++i)
      cu->write_vgpr(vb + i, lane, coords[i]);
  }
  bool issue(bool load, uint32_t dmask = 15) { return issue(load, dmask, Encoding{}); }
  /// @brief Execute IMAGE_LOAD/STORE from s[0:7], VADDR v0, VDATA v8.
  /// @returns Whether the instruction reached the memory pipeline.
  bool issue(bool load, uint32_t dmask, Encoding encoding) {
    cdna2::MimgMachineInst m{};
    m.dmask = dmask;
    m.unorm = encoding.unorm;
    m.da = encoding.da;
    m.a16 = encoding.a16;
    m.d16 = encoding.d16;
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
    write_coordinates(lane, {lane * 16, 0, 0});
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
  write_coordinates(0, {2, 1, 1});
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
  write_coordinates(0, {1, 2, 0});
  cu->write_vgpr(vb + 8, 0, 0xabcdu);
  ASSERT_TRUE(issue(false));
  EXPECT_EQ(memory.read32(kBase + (2 * 64 + 1) * 4), 0xabcdu);
}

TEST_F(Cdna2ImageAccessTest, ArrayWithoutDaUsesLayerZeroAndIgnoresTheNextRegister) {
  wf->set_exec(1);
  for (uint32_t type : {amdgpu::kGfx9Image1dArray, amdgpu::kGfx9Image2dArray}) {
    SCOPED_TRACE(type);
    Image image;
    image.type = type;
    image.width = 3;
    image.height = type == amdgpu::kGfx9Image1dArray ? 1 : 2;
    image.depth = 2;
    bind(image);
    const uint32_t y = image.height - 1;
    const uint64_t address = kBase + (y * image.pitch + 2) * 4;
    const uint64_t layer_one = address + image.height * image.pitch * 4;
    for (uint32_t unrelated : {1u, 0xffffffffu}) {
      SCOPED_TRACE(unrelated);
      write_coordinates(0, type == amdgpu::kGfx9Image1dArray
                               ? std::array<uint32_t, 3>{2, unrelated, unrelated}
                               : std::array<uint32_t, 3>{2, y, unrelated});
      memory.write32(address, 0x12345678);
      memory.write32(layer_one, 0x87654321);
      ASSERT_TRUE(issue(true, 1, {.da = false}));
      EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x12345678u);
      cu->write_vgpr(vb + 8, 0, 0x10203040);
      ASSERT_TRUE(issue(false, 1, {.da = false}));
      EXPECT_EQ(memory.read32(address), 0x10203040u);
      EXPECT_EQ(memory.read32(layer_one), 0x87654321u);
    }
  }
}

TEST_F(Cdna2ImageAccessTest, A16PacksThreeCoordinatesAndIgnoresTheFinalHighHalf) {
  Image image;
  image.type = amdgpu::kGfx9Image2dArray;
  image.width = 3;
  image.height = 2;
  image.depth = 2;
  bind(image);
  wf->set_exec(1);
  write_coordinates(0, {2 | (1u << 16), 1 | (0xffffu << 16), 0xffffffffu});
  cu->write_vgpr(vb + 8, 0, 0x12345678);
  ASSERT_TRUE(issue(false, 1, {.a16 = true}));
  const uint64_t address = kBase + ((1 * 2 + 1) * image.pitch + 2) * 4;
  EXPECT_EQ(memory.read32(address), 0x12345678u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true, 1, {.a16 = true}));
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x12345678u);
  EXPECT_EQ(cu->read_vgpr(vb + 9, 0), 0xdeadbeefu);
  EXPECT_EQ(cu->read_vgpr(vb + 8, 1), 0xdeadbeefu);
}

TEST_F(Cdna2ImageAccessTest, D16RoundTripsThreeComponentsAndPreservesTheUnusedHighHalf) {
  Image image;
  image.data_format = kDfmt8888;
  image.selectors = kIdentity;
  bind(image);
  wf->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  cu->write_vgpr(vb + 8, 0, 0x00220011);
  cu->write_vgpr(vb + 9, 0, 0xdead0033);
  ASSERT_TRUE(issue(false, 7, {.d16 = true}));
  EXPECT_EQ(memory.read32(kBase), 0x00332211u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true, 7, {.d16 = true}));
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x00220011u);
  EXPECT_EQ(cu->read_vgpr(vb + 9, 0), cu->sram_ecc() ? 0x0033u : 0xdead0033u);
  EXPECT_EQ(cu->read_vgpr(vb + 10, 0), 0xdeadbeefu);
  EXPECT_EQ(cu->read_vgpr(vb + 8, 1), 0xdeadbeefu);
}

TEST_F(Cdna2ImageAccessTest, D16StoresAndLoadsSelectComponentsFromPackedRegisters) {
  Image image;
  image.data_format = kDfmt8888;
  image.selectors = kIdentity;
  bind(image);
  wf->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  // Preserve sparse component selection while consuming consecutive halves.
  // This follows the explicit RDNA2 Table 41 example; gfx90a is not hardware-tested.
  for (const auto &[mask, expected] :
       {std::pair{0b0011u, 0x00002211u}, std::pair{0b1001u, 0x22000011u},
        std::pair{0b1100u, 0x22110000u}}) {
    SCOPED_TRACE(mask);
    memory.write32(kBase, 0xffffffff);
    cu->write_vgpr(vb + 8, 0, 0x00220011);
    ASSERT_TRUE(issue(false, mask, {.d16 = true}));
    EXPECT_EQ(memory.read32(kBase), expected);
  }
  memory.write32(kBase, 0x44332211);
  ASSERT_TRUE(issue(true, 0b1001, {.d16 = true}));
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x00440011u);
}

TEST_F(Cdna2ImageAccessTest, LoadWithoutUnormStillUsesIntegerCoordinates) {
  Image image;
  image.width = 4;
  bind(image);
  wf->set_exec(1);
  write_coordinates(0, {2, 0, 0});
  memory.write32(kBase + 8, 0x12345678);
  ASSERT_TRUE(issue(true, 1, {.unorm = false}));
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x12345678u);
}

TEST_F(Cdna2ImageAccessTest, StoreWithoutRequiredUnormIsRejected) {
  bind(Image{});
  wf->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  cu->write_vgpr(vb + 8, 0, 0x12345678);
  EXPECT_FALSE(issue(false, 1, {.unorm = false}));
  EXPECT_EQ(wf->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnsupportedOperandValue);
  EXPECT_EQ(memory.read32(kBase), 0u);
}

TEST_F(Cdna2ImageAccessTest, DmaskPacksTheEnabledComponentsAfterTheChannelSelects) {
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = kNfmtUint;
  image.width = 4;
  image.selectors = kIdentity;
  bind(image);
  wf->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  memory.write32(kBase, 0x44332211u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true, 0b1010));
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x22u);
  EXPECT_EQ(cu->read_vgpr(vb + 9, 0), 0x44u);
  EXPECT_EQ(cu->read_vgpr(vb + 10, 0), 0xdeadbeef);
}

TEST_F(Cdna2ImageAccessTest, UnormStoresConvertAndLoadsReturnFloats) {
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = kNfmtUnorm;
  image.width = 4;
  image.selectors = kIdentity;
  bind(image);
  wf->set_exec(1);
  write_coordinates(0, {1, 0, 0});
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
  write_coordinates(0, {2, 0, 0});
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
  write_coordinates(0, {0, 0, 0});
  memory.write32(kBase, 0x99999999u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true));
  for (uint32_t reg = 0; reg < 4; ++reg)
    EXPECT_EQ(cu->read_vgpr(vb + 8 + reg, 0), 0u);
}

TEST_F(Cdna2ImageAccessTest, TiledSwizzleModeRoundTrips) {
  Image image;
  image.width = 64;
  image.swizzle_mode = 9; // SW_64KB_S.
  image.pitch = 128;
  bind(image);
  wf->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  cu->write_vgpr(vb + 8, 0, 0x12345678u);
  ASSERT_TRUE(issue(false, 1));
  EXPECT_EQ(memory.read32(kBase), 0x12345678u);
  fill_destination(0);
  ASSERT_TRUE(issue(true, 1));
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x12345678u);
}

TEST_F(Cdna2ImageAccessTest, ThreeDimensionalSparseStoreUsesPaddedSlicePitch) {
  Image image;
  image.type = amdgpu::kGfx9Image3d;
  image.width = 3;
  image.height = 2;
  image.depth = 2;
  image.data_format = kDfmt8888;
  image.selectors = kIdentity;
  bind(image);
  wf->set_exec(1);
  write_coordinates(0, {2, 1, 1});
  const uint64_t target = kBase + ((image.height + 1) * image.pitch + 2) * 4;
  const uint64_t other_slice = kBase + (image.pitch + 2) * 4;
  memory.write32(target, 0xffffffff);
  memory.write32(other_slice, 0xabcdef01);
  cu->write_vgpr(vb + 8, 0, 0x11);
  cu->write_vgpr(vb + 9, 0, 0x22);
  ASSERT_TRUE(issue(false, 0b1001, {.da = false}));
  EXPECT_EQ(memory.read32(target), 0x22000011u);
  EXPECT_EQ(memory.read32(other_slice), 0xabcdef01u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true, 0b1001, {.da = false}));
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x11u);
  EXPECT_EQ(cu->read_vgpr(vb + 9, 0), 0x22u);
}

TEST_F(Cdna2ImageAccessTest, NullFormatIgnoresConstantOneSelector) {
  Image image;
  image.data_format = 0;
  image.num_format = kNfmtUnorm;
  bind(image);
  wf->set_exec(1);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true));
  for (uint32_t reg = 0; reg < 4; ++reg)
    EXPECT_EQ(cu->read_vgpr(vb + 8 + reg, 0), 0u);
}

TEST_F(Cdna2ImageAccessTest, DecodedLoadTracksOnlyItsDestinationRegisters) {
  wf->set_exec(1);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
  auto &state = wf->ensure_memory_wait_scoreboard();
  unsigned reports = 0;
  for (bool d16 : {false, true}) {
    for (uint32_t mask : {0u, 1u, 7u, 9u, 15u}) {
      SCOPED_TRACE(testing::Message() << "d16=" << d16 << " mask=" << mask);
      state.clear();
      reports = 0;
      const auto encoding = cdna2::build_mimg(
          cdna2::kImageLoadMimg,
          {.dmask = static_cast<uint8_t>(mask), .unorm = 1, .vdata = 8, .d16 = d16});
      auto decoded = decoder->decode(encoding.data());
      ASSERT_TRUE(decoded.succeeded());
      cu->track_memory_wait(*decoded.value(), *wf);
      state.bind(0x200, &reports,
                 [](void *context, const auto &) { ++*static_cast<unsigned *>(context); });
      const uint16_t words = (std::max(1, std::popcount(mask)) + d16) >> d16;
      state.access({RegClass::VGPR, static_cast<uint16_t>(8 + words), 1}, 1, 0xf, false);
      EXPECT_EQ(reports, 0u);
      state.access({RegClass::VGPR, static_cast<uint16_t>(7 + words), 1}, 1, 0xf, false);
      EXPECT_EQ(reports, 1u);
    }
  }
}

TEST_F(Cdna2ImageAccessTest, DecodedImageChecksTheScaledResourceRegisterRange) {
  wf->set_exec(1);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
  auto &state = wf->ensure_memory_wait_scoreboard();
  unsigned reports = 0;
  state.bind(0x200, &reports,
             [](void *context, const auto &) { ++*static_cast<unsigned *>(context); });
  for (auto opcode : {cdna2::kImageLoadMimg, cdna2::kImageStoreMimg, cdna2::kImageGetResinfoMimg}) {
    const auto encoding =
        cdna2::build_mimg(opcode, {.dmask = 1, .unorm = 1, .vdata = 8, .srsrc = 4});
    auto decoded = decoder->decode(encoding.data());
    ASSERT_TRUE(decoded.succeeded());
    for (uint16_t reg : {uint16_t{4}, uint16_t{16}}) {
      state.clear();
      reports = 0;
      const auto counter = amdgpu::WaitCounterKind::Ds;
      state.add({state.issue(counter), 0x100, 1, {RegClass::SGPR, reg, 8}, counter, 0xf});
      state.check_instruction(*decoded.value(), *wf);
      EXPECT_EQ(reports, reg == 16 ? 1u : 0u);
    }
  }
}

TEST_F(Cdna2ImageAccessTest, ImageOnly565FormatStopsAsUnimplemented) {
  Image image;
  image.data_format = 0x10; // GFX9 FMT_5_6_5, outside the buffer DFMT table.
  image.num_format = kNfmtUnorm;
  bind(image);
  EXPECT_FALSE(issue(true));
  EXPECT_EQ(wf->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
}

TEST_F(Cdna2ImageAccessTest, ImageOnlySrgbFormatStopsAsUnimplemented) {
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = 9; // GFX9 TYPE_SRGB, outside the buffer NFMT table.
  bind(image);
  EXPECT_FALSE(issue(true));
  EXPECT_EQ(wf->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
}

class AllImageAccessTest : public Cdna2ImageAccessTest,
                           public testing::WithParamInterface<rj_code_arch_t> {
protected:
  bool scattered = false;
  bool packed_coordinates = false;
  bool unnormalized = true;
  std::optional<uint32_t> dimension_override;
  rj_code_arch_t arch() const override { return GetParam(); }

  void bind_image(uint32_t dimensions, uint32_t width = 3, uint32_t height = 2,
                  uint32_t swizzle = 0) {
    Image image;
    image.width = width;
    image.height = dimensions >= 2 ? height : 1;
    image.swizzle_mode = swizzle;
    image.depth = dimensions == 3 ? 2 : 1;
    image.type = dimensions == 1   ? amdgpu::kGfx9Image1d
                 : dimensions == 2 ? amdgpu::kGfx9Image2d
                                   : amdgpu::kGfx9Image3d;
    bind_image(image);
  }
  void bind_image(const Image &image) {
    const auto words =
        arch_is_cdna_4_or_lower(arch()) ? descriptor(image) : rdna_descriptor(image, arch());
    for (uint32_t index = 0; index < words.size(); ++index)
      cu->write_sgpr(sb + index, words[index]);
  }

  template <typename Fields> Fields fields(uint32_t dimensions) {
    Fields encoding{};
    encoding.dmask = 1;
    encoding.a16 = packed_coordinates;
    encoding.vdata = 8;
    if constexpr (requires { encoding.vaddr; })
      encoding.vaddr = 0;
    else {
      encoding.vaddr0 = 0;
      encoding.vaddr1 = scattered ? 10 : 1;
      encoding.vaddr2 = scattered ? 12 : 2;
    }
    if constexpr (requires { encoding.nsa; })
      encoding.nsa = scattered;
    if constexpr (requires { encoding.unorm; })
      encoding.unorm = unnormalized;
    if constexpr (requires { encoding.da; })
      encoding.da = 1;
    if constexpr (requires { encoding.dim; })
      encoding.dim = dimension_override.value_or(dimensions - 1);
    return encoding;
  }
  std::unique_ptr<Instruction> decoded_instruction(bool load, uint32_t dimensions,
                                                   bool explicit_mip = false) {
    std::array<uint32_t, 3> words{};
#define MIMG(ARCH, ENUM)                                                                           \
  case ENUM: {                                                                                     \
    const auto encoding =                                                                          \
        ARCH::build_mimg(load ? (explicit_mip ? ARCH::kImageLoadMipMimg : ARCH::kImageLoadMimg)    \
                              : (explicit_mip ? ARCH::kImageStoreMipMimg : ARCH::kImageStoreMimg), \
                         fields<ARCH::MimgBuilderFields>(dimensions));                             \
    std::copy(encoding.begin(), encoding.end(), words.begin());                                    \
    break;                                                                                         \
  }
    switch (arch()) {
      MIMG(cdna1, ROCJITSU_CODE_ARCH_CDNA1)
      MIMG(cdna2, ROCJITSU_CODE_ARCH_CDNA2)
      MIMG(rdna1, ROCJITSU_CODE_ARCH_RDNA1)
      MIMG(rdna2, ROCJITSU_CODE_ARCH_RDNA2)
      MIMG(rdna3, ROCJITSU_CODE_ARCH_RDNA3)
      MIMG(rdna3_5, ROCJITSU_CODE_ARCH_RDNA3_5)
    case ROCJITSU_CODE_ARCH_RDNA4: {
      const auto encoding = rdna4::build_vimage(
          load ? (explicit_mip ? rdna4::kImageLoadMipVimage : rdna4::kImageLoadVimage)
               : (explicit_mip ? rdna4::kImageStoreMipVimage : rdna4::kImageStoreVimage),
          fields<rdna4::VimageBuilderFields>(dimensions));
      std::copy(encoding.begin(), encoding.end(), words.begin());
      break;
    }
    default:
      return {};
    }
#undef MIMG
    if (scattered && arch() != ROCJITSU_CODE_ARCH_RDNA4)
      words[2] = 10 | (12 << 8);
    auto decoder = Decoder::create(arch());
    auto decoded = decoder->decode(words.data());
    EXPECT_TRUE(decoded.succeeded());
    if (decoded.failed())
      return {};
    return std::move(decoded).value();
  }
  bool decoded_issue(bool load, uint32_t dimensions) {
    auto inst = decoded_instruction(load, dimensions);
    if (!inst)
      return false;
    EXPECT_TRUE(inst->is_memory_op());
    inst->execute(*inst, wf);
    if (!inst->data())
      return false;
    amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
    EXPECT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
    cu->l1_vector().flush_all();
    l2.flush_all();
    return true;
  }
};

TEST_P(AllImageAccessTest, DecodedLoadAndStoreRoundTripDimensionsAndExec) {
  for (uint32_t dimensions = 1; dimensions <= 3; ++dimensions) {
    SCOPED_TRACE(dimensions);
    bind_image(dimensions);
    for (uint32_t lane = 0; lane < 3; ++lane) {
      write_coordinates(lane, {lane, dimensions >= 2 ? 1u : 0u, dimensions == 3 ? 1u : 0u});
      cu->write_vgpr(vb + 8, lane, 0x12345678u + lane);
    }
    ASSERT_TRUE(decoded_issue(false, dimensions));
    fill_destination(0xdeadbeef);
    ASSERT_TRUE(decoded_issue(true, dimensions));
    EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x12345678u);
    EXPECT_EQ(cu->read_vgpr(vb + 8, 1), 0xdeadbeef);
    EXPECT_EQ(cu->read_vgpr(vb + 8, 2), 0x1234567au);
  }
}

TEST_P(AllImageAccessTest, TiledAddressesMatchIndependentBackingOffsets) {
  // Direct ROCr AddrLib reference calls establish these R32 offsets in a
  // 64x64 image. The latter coordinates cross 256-byte and 4-KiB tile edges.
  const auto architecture = arch();
  const bool gfx9 = arch_is_cdna_4_or_lower(architecture);
  const bool gfx10 =
      architecture == ROCJITSU_CODE_ARCH_RDNA1 || architecture == ROCJITSU_CODE_ARCH_RDNA2;
  const uint32_t swizzle = gfx9 ? 22 : architecture == ROCJITSU_CODE_ARCH_RDNA4 ? 1 : 2;
  bind_image(2, 64, 64, swizzle);
  wf->set_exec(1);
  const std::array<std::array<uint32_t, 3>, 3> cases{{{1, 1, gfx9 || gfx10 ? 20u : 12u},
                                                      {9, 9,
                                                       gfx9    ? 788u
                                                       : gfx10 ? 2324u
                                                               : 2316u},
                                                      {33, 17, gfx9 || gfx10 ? 5140u : 5132u}}};
  for (const auto &[x, y, offset] : cases) {
    SCOPED_TRACE(testing::Message() << x << ',' << y);
    const uint32_t stored = 0x12340000u + offset;
    const uint32_t loaded = 0xabcd0000u + offset;
    write_coordinates(0, {x, y, 0});
    cu->write_vgpr(vb + 8, 0, stored);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory.read32(kBase + offset), stored);
    EXPECT_NE(memory.read32(kBase + (y * 64 + x) * 4), stored);
    // Seed backing memory independently before loading, so a mutually wrong
    // store and load cannot satisfy the oracle by undoing each other.
    memory.write32(kBase + offset, loaded);
    cu->l1_vector().invalidate_all();
    l2.invalidate_all();
    ASSERT_TRUE(decoded_issue(true, 2));
    EXPECT_EQ(cu->read_vgpr(vb + 8, 0), loaded);
  }
}

TEST_P(AllImageAccessTest, TiledStoresMatchPhysicalGpuBacking) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Physical backing captures are available for gfx1100 and gfx1201";
  // ROCr opaque R32 37x29 captures on W7900 and R9700: these texel values
  // occur at exactly one DWORD in each raw backing dump. The last coordinate
  // crosses a 4-KiB tile boundary. No emulator-generated address is the oracle.
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  bind_image(2, 37, 29, gfx12 ? 2 : 22);
  wf->set_exec(1);
  const std::array<std::array<uint32_t, 4>, 3> cases{{{1, 1, 12, 0xb5f93d83},
                                                      {21, 11, 2412, 0x2167abef},
                                                      {33, 27, gfx12 ? 5420u : 4396u, 0x85c90d53}}};
  for (const auto &[x, y, offset, value] : cases) {
    SCOPED_TRACE(testing::Message() << x << ',' << y);
    write_coordinates(0, {x, y, 0});
    cu->write_vgpr(vb + 8, 0, value);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory.read32(kBase + offset), value);
  }
}

TEST_P(AllImageAccessTest, MipViewsUseTheWholeAllocationAndTheirSelectedLevel) {
  const bool gfx9 = arch_is_cdna_4_or_lower(arch());
  const bool gfx10 = arch() == ROCJITSU_CODE_ARCH_RDNA1 || arch() == ROCJITSU_CODE_ARCH_RDNA2;
  Image image;
  image.width = 37;
  image.height = 19;
  image.mip_levels = 4;
  image.swizzle_mode = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 2 : 22;
  wf->set_exec(1);
  // Independent reference constants; the GFX11/12 layouts also match physical
  // full-allocation captures after each mip-level import.
  const std::array<uint32_t, 2> offsets = gfx9    ? std::array<uint32_t, 2>{116, 8308}
                                          : gfx10 ? std::array<uint32_t, 2>{8372, 4276}
                                                  : std::array<uint32_t, 2>{8300, 4204};
  for (uint32_t level = 0; level < offsets.size(); ++level) {
    SCOPED_TRACE(level);
    image.level = level;
    bind_image(image);
    write_coordinates(0, {5, 3, 0});
    cu->write_vgpr(vb + 8, 0, 0x12340000u + level);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory.read32(kBase + offsets[level]), 0x12340000u + level);
    memory.write32(kBase + offsets[level], 0xabcd0000u + level);
    cu->l1_vector().invalidate_all();
    l2.invalidate_all();
    ASSERT_TRUE(decoded_issue(true, 2));
    EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0xabcd0000u + level);
  }
  write_coordinates(0, {image.width >> 1, 0, 0});
  cu->write_vgpr(vb + 8, 0, 0xdeadbeefu);
  ASSERT_TRUE(decoded_issue(true, 2));
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0u);
}

TEST_P(AllImageAccessTest, LinearRowsMatchPhysicalGpuBackingAtExplicitPitches) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Physical linear captures are available for gfx1100 and gfx1201";
  Image image;
  image.width = 30;
  image.height = 5;
  wf->set_exec(1);
  // Valid 256/512-byte R32 allocations and independently patched descriptors
  // establish row rounding: GFX11 uses 256-byte quanta, GFX12 uses 128.
  // A 140-byte pitch rounds to 256 on both; 128 stays 128 only on GFX12.
  for (uint32_t pitch : {32u, 35u, 64u, 128u}) {
    bind_image(image);
    cu->write_sgpr(sb + 4, pitch - 1);
    for (uint32_t row : {1u, 3u}) {
      write_coordinates(0, {5, row, 0});
      const uint32_t captured_pitch = pitch == 128                                        ? 128
                                      : pitch == 32 && arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 32
                                                                                          : 64;
      const uint32_t offset = (row * captured_pitch + 5) * 4;
      cu->write_vgpr(vb + 8, 0, 0x12340000u + offset);
      ASSERT_TRUE(decoded_issue(false, 2));
      EXPECT_EQ(memory.read32(kBase + offset), 0x12340000u + offset);
      memory.write32(kBase + offset, 0xabcd0000u + offset);
      cu->l1_vector().invalidate_all();
      l2.invalidate_all();
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0xabcd0000u + offset);
    }
  }
}

TEST_P(AllImageAccessTest, LinearArrayAndVolumeRowsMatchPhysicalGpuBacking) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Physical linear captures are available for gfx1100 and gfx1201";
  Image image;
  image.width = 30;
  image.height = 5;
  image.depth = 2;
  wf->set_exec(1);
  // Independent raw captures: GFX12 rows advance by 128 bytes, while slices
  // advance by 5 * 256 bytes. GFX11 rows and slice allocation use 256 bytes.
  const uint32_t row_pitch = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
  for (uint32_t type : {amdgpu::kGfx9Image2dArray, amdgpu::kGfx9Image3d}) {
    image.type = type;
    dimension_override = type - amdgpu::kGfx9Image1d;
    bind_image(image);
    for (const std::array<uint32_t, 3> coordinates :
         {std::array<uint32_t, 3>{5, 1, 0}, std::array<uint32_t, 3>{5, 3, 1}}) {
      const uint32_t offset =
          coordinates[2] * 1280 + coordinates[1] * row_pitch + coordinates[0] * 4;
      write_coordinates(0, coordinates);
      cu->write_vgpr(vb + 8, 0, 0x12340000u + offset);
      ASSERT_TRUE(decoded_issue(false, 3));
      EXPECT_EQ(memory.read32(kBase + offset), 0x12340000u + offset);
      memory.write32(kBase + offset, 0xabcd0000u + offset);
      cu->l1_vector().invalidate_all();
      l2.invalidate_all();
      ASSERT_TRUE(decoded_issue(true, 3));
      EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0xabcd0000u + offset);
    }
  }
}

TEST_P(AllImageAccessTest, MisalignedLinearPitchNeedsQualifiedRounding) {
  const amdgpu::ImageAddressConfig config = amdgpu::default_image_address_config(arch());
  const amdgpu::ImageLayout layout{.width = 30, .height = 5, .pitch = 35, .element_bytes = 4};
  const std::optional<amdgpu::ImageAddressing> addressing =
      amdgpu::ImageAddressing::prepare(config, layout);
  const bool qualified = arch() == ROCJITSU_CODE_ARCH_RDNA3 || arch() == ROCJITSU_CODE_ARCH_RDNA4;
  ASSERT_EQ(addressing.has_value(), qualified);
  if (addressing)
    EXPECT_EQ(addressing->byte_offset(0, 1, 0), 256u);
}

TEST_P(AllImageAccessTest, CompressedDescriptorsStopBeforeAccessingOrdinaryBacking) {
  bind_image(2, 64, 64, arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 2 : 22);
  const uint32_t word6 = cu->read_sgpr(sb + 6);
  for (uint32_t flag : {21u, 20u}) {
    if (flag == 20 && arch_is_cdna_4_or_lower(arch()))
      continue;
    cu->write_sgpr(sb + 6, word6 | (1u << flag));
    for (bool load : {false, true}) {
      wf->clear_instruction_execution_error();
      EXPECT_FALSE(decoded_issue(load, 2));
      EXPECT_EQ(wf->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
    }
  }
}

TEST_P(AllImageAccessTest, UnsupportedLayoutIsDiagnosedEvenWithoutInBoundsLanes) {
  bind_image(2, 3, 2, 8);
  for (uint64_t lanes : {uint64_t{0}, uint64_t{1}}) {
    wf->clear_instruction_execution_error();
    wf->set_exec(lanes);
    write_coordinates(0, {100, 100, 100});
    EXPECT_FALSE(decoded_issue(true, 2));
    EXPECT_EQ(wf->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
  }
}

TEST_P(AllImageAccessTest, StoreRequiresUnnormalizedCoordinates) {
  if (arch() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "VIMAGE has no UNORM control";
  bind_image(2);
  unnormalized = false;
  EXPECT_FALSE(decoded_issue(false, 2));
  EXPECT_EQ(wf->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnsupportedOperandValue);
}

TEST_P(AllImageAccessTest, LegalCubeAndMsaaDimensionsStopAsUnimplemented) {
  if (arch_is_cdna_4_or_lower(arch()))
    GTEST_SKIP() << "GFX9 has no DIM field";
  bind_image(2);
  for (uint32_t dimension : {3u, 6u, 7u}) {
    SCOPED_TRACE(dimension);
    wf->clear_instruction_execution_error();
    dimension_override = dimension;
    EXPECT_FALSE(decoded_issue(true, 2));
    EXPECT_EQ(wf->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
  }
}

TEST_P(AllImageAccessTest, ScatteredAndPackedCoordinatesReachTheSameTexel) {
  if (arch_is_cdna_4_or_lower(arch()))
    GTEST_SKIP() << "GFX9 image coordinates are sequential";
  wf->set_exec(1);
  bind_image(3);
  scattered = true;
  for (bool packed : {false, true}) {
    packed_coordinates = packed;
    cu->write_vgpr(vb, 0, packed ? 2 | (1u << 16) : 2);
    cu->write_vgpr(vb + 10, 0, 1);
    cu->write_vgpr(vb + 12, 0, 1);
    cu->write_vgpr(vb + 8, 0, 0x10203040u + packed);
    ASSERT_TRUE(decoded_issue(false, 3));
    cu->write_vgpr(vb + 8, 0, 0);
    ASSERT_TRUE(decoded_issue(true, 3));
    EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x10203040u + packed);
  }
}

TEST_P(AllImageAccessTest, WaitChecksConsumeTheEncodedCoordinateRegisters) {
  wf->set_exec(1);
  auto &state = wf->ensure_memory_wait_scoreboard();
  scattered = !arch_is_cdna_4_or_lower(arch());
  for (uint16_t pending : {0, 1, 2, 3, 10, 12}) {
    SCOPED_TRACE(pending);
    state.clear();
    const auto sequence = state.issue(amdgpu::WaitCounterKind::Load);
    state.add(
        {sequence, 0x100, 1, {RegClass::VGPR, pending, 1}, amdgpu::WaitCounterKind::Load, 0xf});
    unsigned reports = 0;
    state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
    auto instruction = decoded_instruction(false, 3);
    ASSERT_NE(instruction, nullptr);
    state.check_instruction(*instruction, *wf);
    EXPECT_EQ(reports,
              (scattered ? pending == 0 || pending == 10 || pending == 12 : pending < 3) ? 1u : 0u);
  }
}

INSTANTIATE_TEST_SUITE_P(Architectures, AllImageAccessTest,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));

TEST_P(AllImageAccessTest, NativeAddressingMatchesIndependentOffsets) {
  // Captured from an independent address oracle before removing the runtime
  // dependency. The expected offsets are constants, not calculated by RJ.
  // Columns cover GFX9, GFX10, GFX11 with GB_ADDR_CONFIG=0x545 and zero,
  // and GFX12. Separate physical raw-backing tests qualify GFX11/12 layouts.
  struct Case {
    uint32_t kind, dimensions, bytes, width, height, depth, levels, mip, x, y, z;
    std::array<uint64_t, 5> expected;
  };
  constexpr Case cases[] = {
      {1, 2, 1, 130, 65, 3, 1, 0, 129, 64, 2, {69633ull, 72961ull, 72961ull, 69633ull, 69633ull}},
      {2,
       2,
       1,
       257,
       129,
       3,
       1,
       0,
       256,
       128,
       2,
       {344064ull, 345088ull, 706048ull, 628736ull, 671744ull}},
      {3, 3, 1, 33, 17, 5, 1, 0, 32, 16, 4, {18688ull, 264320ull, 264960ull, 263680ull, 18688ull}},
      {1,
       2,
       2,
       130,
       65,
       3,
       1,
       0,
       129,
       64,
       2,
       {106498ull, 108290ull, 108290ull, 106498ull, 106498ull}},
      {2,
       2,
       2,
       257,
       129,
       3,
       1,
       0,
       256,
       128,
       2,
       {720896ull, 721920ull, 755200ull, 723968ull, 720896ull}},
      {3, 3, 2, 33, 17, 5, 1, 0, 32, 16, 4, {19456ull, 268288ull, 264960ull, 265216ull, 19456ull}},
      {1,
       2,
       4,
       130,
       65,
       3,
       1,
       0,
       129,
       64,
       2,
       {180228ull, 181764ull, 181764ull, 180228ull, 180228ull}},
      {2,
       2,
       4,
       257,
       129,
       3,
       1,
       0,
       256,
       128,
       2,
       {1114112ull, 1115136ull, 1443328ull, 1442816ull, 1376256ull}},
      {3, 3, 4, 33, 17, 5, 1, 0, 32, 16, 4, {37888ull, 264192ull, 264960ull, 268288ull, 37888ull}},
      {1,
       2,
       8,
       130,
       65,
       3,
       1,
       0,
       129,
       64,
       2,
       {303112ull, 304136ull, 304136ull, 303112ull, 303112ull}},
      {2,
       2,
       8,
       257,
       129,
       3,
       1,
       0,
       256,
       128,
       2,
       {1703936ull, 1704960ull, 3118080ull, 2885632ull, 2883584ull}},
      {3, 3, 8, 33, 17, 5, 1, 0, 32, 16, 4, {77824ull, 280576ull, 265984ull, 274432ull, 77824ull}},
      {1,
       2,
       16,
       130,
       65,
       3,
       1,
       0,
       129,
       64,
       2,
       {548880ull, 549904ull, 549904ull, 548880ull, 548880ull}},
      {2,
       2,
       16,
       257,
       129,
       3,
       1,
       0,
       256,
       128,
       2,
       {2883584ull, 2884608ull, 4658176ull, 4457472ull, 4456448ull}},
      {3,
       3,
       16,
       33,
       17,
       5,
       1,
       0,
       32,
       16,
       4,
       {118784ull, 264192ull, 264448ull, 278528ull, 118784ull}},
      {1, 2, 4, 37, 19, 3, 6, 0, 36, 18, 2, {37984ull, 46240ull, 46176ull, 46176ull, 46176ull}},
      {1, 2, 4, 37, 19, 3, 6, 1, 17, 8, 2, {43268ull, 40196ull, 40196ull, 39172ull, 39172ull}},
      {1, 2, 4, 37, 19, 3, 6, 2, 8, 3, 2, {47696ull, 36400ull, 36392ull, 35368ull, 35368ull}},
      {1, 2, 4, 37, 19, 3, 6, 3, 3, 1, 2, {46620ull, 33308ull, 33308ull, 34332ull, 34332ull}},
      {1, 2, 4, 37, 19, 3, 6, 4, 1, 0, 2, {46340ull, 33028ull, 33028ull, 34052ull, 34052ull}},
      {1, 2, 4, 37, 19, 3, 6, 5, 0, 0, 2, {46080ull, 32768ull, 32768ull, 33792ull, 33792ull}},
      {0, 1, 4, 257, 1, 3, 6, 0, 256, 0, 2, {16384ull, 9728ull, 9728ull, 9728ull, 9728ull}},
      {0, 1, 4, 257, 1, 3, 6, 2, 63, 0, 2, {18172ull, 7676ull, 7676ull, 7676ull, 7676ull}},
      {0, 1, 4, 257, 1, 3, 6, 5, 7, 0, 2, {21788ull, 6684ull, 6684ull, 6684ull, 6684ull}},
      {3,
       3,
       2,
       130,
       129,
       17,
       6,
       0,
       129,
       128,
       16,
       {1728514ull, 3309570ull, 4444162ull, 4429826ull, 1667074ull}},
      {3,
       3,
       2,
       130,
       129,
       17,
       6,
       2,
       31,
       31,
       3,
       {382974ull, 611582ull, 817150ull, 825342ull, 301054ull}},
      {3, 3, 2, 130, 129, 17, 6, 4, 7, 7, 0, {79486ull, 1150ull, 32894ull, 8318ull, 1662ull}},
  };
  const amdgpu::ImageAddressConfig config = amdgpu::default_image_address_config(arch());
  const uint32_t group = arch_is_cdna_4_or_lower(arch()) ? 0
                         : arch() == ROCJITSU_CODE_ARCH_RDNA1 || arch() == ROCJITSU_CODE_ARCH_RDNA2
                             ? 1
                         : arch() == ROCJITSU_CODE_ARCH_RDNA3   ? 2
                         : arch() == ROCJITSU_CODE_ARCH_RDNA3_5 ? 3
                                                                : 4;
  for (const auto &c : cases) {
    SCOPED_TRACE(testing::Message() << c.kind << ':' << c.bytes << ':' << c.mip);
    const uint32_t mode = c.kind == 0    ? 0
                          : c.kind == 1  ? (group == 4 ? 2 : 22)
                          : c.kind == 2  ? (group == 0   ? 26
                                            : group == 1 ? 27
                                            : group == 4 ? 4
                                                         : 31)
                          : group == 0   ? 22
                          : group == 1   ? 27
                          : group == 4   ? 2
                          : c.bytes < 16 ? 24
                                         : 27;
    const amdgpu::ImageLayout layout{.width = c.width,
                                     .height = c.height,
                                     .depth = c.depth,
                                     .element_bytes = c.bytes,
                                     .swizzle_mode = mode,
                                     .dimensions = c.dimensions,
                                     .mip_levels = c.levels};
    const std::optional<amdgpu::ImageAddressing> addressing =
        amdgpu::ImageAddressing::prepare(config, layout, c.mip);
    ASSERT_TRUE(addressing.has_value());
    ASSERT_TRUE(addressing->contains(c.x, c.y, c.z));
    EXPECT_EQ(addressing->byte_offset(c.x, c.y, c.z), c.expected[group]);
  }
}

TEST_P(AllImageAccessTest, NativeAddressingRejectsUnsupportedLayouts) {
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

TEST_P(AllImageAccessTest, NativeAddressingSeparatesViewBoundsFromLayoutSupport) {
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

TEST_P(AllImageAccessTest, UnsupportedExplicitMipInstructionsReportAnExecutionError) {
  bind_image(2);
  for (bool load : {false, true}) {
    wf->clear_instruction_execution_error();
    auto instruction = decoded_instruction(load, 2, true);
    ASSERT_TRUE(instruction);
    instruction->execute(*instruction, wf);
    EXPECT_EQ(wf->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
  }
}

TEST_P(AllImageAccessTest, ThickDisplayVolumesStopBeforeImageMemoryAccess) {
  if (arch_is_cdna_4_or_lower(arch()) || arch() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "GFX9 display volumes are thin; GFX12 has different swizzle encodings";
  bind_image(3, 64, 64, 26);
  for (bool load : {false, true}) {
    wf->clear_instruction_execution_error();
    EXPECT_FALSE(decoded_issue(load, 3));
    EXPECT_EQ(wf->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
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

} // namespace
