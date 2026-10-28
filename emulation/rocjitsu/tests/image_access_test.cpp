// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/mimg.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_access.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_resource.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "gfx11_mimg_test_common.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <utility>

namespace {
using namespace rocjitsu;

constexpr uint32_t kIdentity = 4 | (5 << 3) | (6 << 6) | (7 << 9);
constexpr uint32_t kXyzOne = 4 | (5 << 3) | (6 << 6) | (1 << 9);
constexpr uint32_t kZyxOne = 6 | (5 << 3) | (4 << 6) | (1 << 9);
constexpr uint32_t kXOne = 4 | (1 << 9); // X, 0, 0, 1 (an R-only format).
constexpr uint64_t kBase = 0x10000;
constexpr uint32_t kDfmt32 = 4;
constexpr uint32_t kDfmt8888 = 10;
constexpr uint32_t kNfmtUint = 4;
constexpr uint32_t kNfmtUnorm = 0;
constexpr uint32_t kRgba32Uint = 61;

struct Image {
  uint32_t data_format = kDfmt32;
  uint32_t num_format = kNfmtUint;
  uint32_t width = 1, height = 1, depth = 1, pitch = 64;
  uint32_t type = amdgpu::kImage2d;
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

struct RdnaImageDescriptor {
  uint32_t type = amdgpu::kImage2d;
  uint32_t format = 20;
  uint32_t selectors = kXOne;
  uint32_t width = 3;
  uint32_t height = 2;
  uint32_t pitch = 64;
  uint32_t depth = 4;
  bool gfx12_bit_157 = false;
};

std::array<uint32_t, 8> rdna_image_descriptor(rj_code_arch_t arch,
                                              const RdnaImageDescriptor &image = {}) {
  std::array<uint32_t, 8> words{};
  const auto set = [&](uint32_t bit, uint32_t width, uint64_t value) {
    for (uint32_t i = 0; i < width; ++i)
      words[(bit + i) / 32] |= static_cast<uint32_t>((value >> i) & 1u) << ((bit + i) % 32);
  };
  const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
  set(0, 40, kBase >> 8);
  set(gfx12 ? 49 : 52, 8, image.format);
  set(62, gfx12 ? 16 : 14, image.width - 1);
  set(78, gfx12 ? 16 : 14, image.height - 1);
  set(96, 12, image.selectors);
  set(124, 4, image.type);
  if (amdgpu::image_resource_encodes_pitch(image.type))
    set(128, gfx12 ? 16 : 14, image.pitch - 1);
  else
    set(128, gfx12 ? 14 : 13, image.depth - 1);
  if (gfx12 && image.gfx12_bit_157)
    set(157, 1, 1);
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
  image.type = amdgpu::kImage2dArray;
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
  image.type = amdgpu::kImage1dArray;
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
  for (uint32_t type : {amdgpu::kImage1dArray, amdgpu::kImage2dArray}) {
    SCOPED_TRACE(type);
    Image image;
    image.type = type;
    image.width = 3;
    image.height = type == amdgpu::kImage1dArray ? 1 : 2;
    image.depth = 2;
    bind(image);
    const uint32_t y = image.height - 1;
    const uint64_t address = kBase + (y * image.pitch + 2) * 4;
    const uint64_t layer_one = address + image.height * image.pitch * 4;
    for (uint32_t unrelated : {1u, 0xffffffffu}) {
      SCOPED_TRACE(unrelated);
      write_coordinates(0, type == amdgpu::kImage1dArray
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
  image.type = amdgpu::kImage2dArray;
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

TEST_F(Cdna2ImageAccessTest, XyzOneStoresRetainCompactedSelectorBehavior) {
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = kNfmtUint;
  image.selectors = kXyzOne;
  bind(image);
  wf->set_exec(1);
  write_coordinates(0, {0, 0, 0});

  struct Case {
    uint8_t dmask;
    std::array<uint32_t, 4> packed;
    uint32_t expected;
  };
  const Case cases[] = {
      {15, {0x11, 0x22, 0x33, 0x44}, 0x01332211},
      {7, {0x11, 0x22, 0x33}, 0x00332211},
      {9, {0x11, 0x44}, 0x00000011},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(unsigned{test.dmask});
    memory.write32(kBase, 0xdeadbeef);
    for (uint32_t component = 0; component < test.packed.size(); ++component)
      cu->write_vgpr(vb + 8 + component, 0, test.packed[component]);
    ASSERT_TRUE(issue(false, test.dmask));
    EXPECT_EQ(memory.read32(kBase), test.expected);
  }
}

TEST_F(Cdna2ImageAccessTest, ArgbStoreRetainsCompactedSelectorBehavior) {
  constexpr uint32_t kYzwx = 5 | (6 << 3) | (7 << 6) | (4 << 9);
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = kNfmtUint;
  image.selectors = kYzwx;
  bind(image);
  wf->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  for (uint32_t component = 0; component < 4; ++component)
    cu->write_vgpr(vb + 8 + component, 0, 0x11 * (component + 1));

  ASSERT_TRUE(issue(false, 15));
  EXPECT_EQ(memory.read32(kBase), 0x33221144u);
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

TEST_F(Cdna2ImageAccessTest, TiledSwizzleModesStopAsUnimplemented) {
  Image image;
  image.width = 64;
  image.swizzle_mode = 9; // SW_64KB_S.
  bind(image);
  wf->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  EXPECT_FALSE(issue(true));
  EXPECT_EQ(wf->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
}

TEST_F(Cdna2ImageAccessTest, ThreeDimensionalSparseStoreUsesPaddedSlicePitch) {
  Image image;
  image.type = amdgpu::kImage3d;
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

TEST(ImageAccessTest, InstructionDimCoordinatesShareA16PackingContract) {
  struct Case {
    uint8_t dim;
    uint32_t coordinates;
  };
  for (const Case test : {
           Case{amdgpu::kImageDim1d, 1},
           Case{amdgpu::kImageDim2d, 2},
           Case{amdgpu::kImageDim3d, 3},
           Case{amdgpu::kImageDimCube, 3},
           Case{amdgpu::kImageDim1dArray, 2},
           Case{amdgpu::kImageDim2dArray, 3},
           Case{amdgpu::kImageDim2dMsaa, 3},
           Case{amdgpu::kImageDim2dMsaaArray, 4},
       }) {
    EXPECT_EQ(amdgpu::image_coordinate_count(test.dim), test.coordinates);
    EXPECT_EQ(amdgpu::image_coordinate_register_count(test.dim, false), test.coordinates);
    EXPECT_EQ(amdgpu::image_coordinate_register_count(test.dim, true), (test.coordinates + 1) / 2);
  }
}

TEST(ImageAccessTest, RdnaLinearR32DescriptorsPrepareTheExpectedMemoryAccess) {
  for (auto arch :
       {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4}) {
    SCOPED_TRACE(arch);
    amdgpu::GpuMemory memory{"rdna_image_memory"};
    amdgpu::L2Cache l2{"rdna_image_l2"};
    l2.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 32;
    cfg.vgprs_per_wf = 32;
    auto cu = amdgpu::ComputeUnitCore::create("rdna_image_cu", cfg, &memory, &l2);
    auto *wf = cu->dispatch_wf(0, 0, 32, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(1);

    const auto words =
        rdna_image_descriptor(arch, {.width = 64, .height = 1, .gfx12_bit_157 = true});
    for (uint32_t i = 0; i < words.size(); ++i)
      cu->write_sgpr(wf->sgpr_alloc().base + i, words[i]);
    cu->write_vgpr(wf->vgpr_alloc().base, 0, 0);
    cu->write_vgpr(wf->vgpr_alloc().base + 1, 0, 0);

    amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
    state.is_load = true;
    ASSERT_TRUE(amdgpu::prepare_image_access(*wf, state, 0,
                                             amdgpu::contiguous_image_address_registers(0),
                                             {.dmask = 15,
                                              .dim = amdgpu::kImageDim2d,
                                              .a16 = false,
                                              .d16 = false,
                                              .lwe = false,
                                              .da = true,
                                              .unorm = true})
                    .succeeded());
    EXPECT_EQ(state.lane_mask, 1u);
    EXPECT_EQ(state.per_lane_addr[0], kBase);
    EXPECT_EQ(state.elem_size, 4u);
    EXPECT_EQ(state.buffer_format, 20u);
    wf->halt();
  }
}

class RdnaDecodedImageAccessTest : public testing::TestWithParam<rj_code_arch_t> {
protected:
  struct Encoding {
    bool r128 = false;
    uint8_t dim = amdgpu::kImageDim2d;
    bool a16 = false;
    bool nsa = true;
    uint8_t dmask = 1;
    bool d16 = false;
    bool tfe = false;
  };

  static constexpr uint32_t kResourceBase = 8;
  amdgpu::GpuMemory memory{"rdna_image_memory"};
  amdgpu::L2Cache l2{"rdna_image_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  void SetUp() override {
    l2.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = GetParam();
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 32;
    cfg.vgprs_per_wf = 32;
    cu = amdgpu::ComputeUnitCore::create("rdna_image_cu", cfg, &memory, &l2);
    wf = cu->dispatch_wf(0, 0, 32, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(5); // Lane 1 must not store or receive a load result.
    decoder = Decoder::create(GetParam());
    ASSERT_NE(decoder, nullptr);

    bind_resource();
  }

  void bind_resource(const RdnaImageDescriptor &descriptor) {
    const auto resource = rdna_image_descriptor(GetParam(), descriptor);
    for (uint32_t i = 0; i < resource.size(); ++i)
      cu->write_sgpr(wf->sgpr_alloc().base + kResourceBase + i, resource[i]);
  }

  void bind_resource(uint32_t format = 20, uint32_t selectors = kXOne) {
    bind_resource(RdnaImageDescriptor{.format = format, .selectors = selectors});
  }

  void TearDown() override {
    if (wf)
      wf->halt();
  }

  DecodeResult decode(bool load) { return decode(load, Encoding{}); }

  DecodeResult decode(bool load, Encoding encoding) {
    const test::Gfx11MimgEncoding mimg{.nsa = encoding.nsa,
                                       .dim = encoding.dim,
                                       .unorm = true,
                                       .dmask = encoding.dmask,
                                       .r128 = encoding.r128,
                                       .a16 = encoding.a16,
                                       .d16 = encoding.d16,
                                       .vaddr = 2,
                                       .vdata = 8,
                                       .srsrc = 2,
                                       .tfe = encoding.tfe,
                                       .nsa_word = 6};
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3) {
      return test::decode_gfx11_mimg<rdna3::MimgBuilderFields>(
          *decoder, rdna3::build_mimg, load ? rdna3::kImageLoadMimg : rdna3::kImageStoreMimg, mimg);
    }
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5) {
      return test::decode_gfx11_mimg<rdna3_5::MimgBuilderFields>(
          *decoder, rdna3_5::build_mimg, load ? rdna3_5::kImageLoadMimg : rdna3_5::kImageStoreMimg,
          mimg);
    }
    const auto words = rdna4::build_vimage(
        load ? rdna4::kImageLoadVimage : rdna4::kImageStoreVimage, {.dim = encoding.dim,
                                                                    .r128 = encoding.r128,
                                                                    .d16 = encoding.d16,
                                                                    .a16 = encoding.a16,
                                                                    .dmask = encoding.dmask,
                                                                    .vdata = 8,
                                                                    .rsrc = kResourceBase,
                                                                    .tfe = encoding.tfe,
                                                                    .vaddr0 = 2,
                                                                    .vaddr1 = 6});
    return decoder->decode(words.data());
  }

  void issue(bool load) { issue(load, Encoding{}); }

  void issue(bool load, Encoding encoding) {
    auto decoded = decode(load, encoding);
    ASSERT_TRUE(decoded.succeeded());
    std::unique_ptr<Instruction> inst = std::move(decoded).value();
    ASSERT_NE(inst, nullptr);

    const InstDefUse def_use(*inst);
    EXPECT_TRUE(def_use.uses.contains(RegisterRef{RegClass::VGPR, 2, 1}));
    EXPECT_TRUE(def_use.uses.contains(RegisterRef{RegClass::VGPR, 6, 1}));
    EXPECT_FALSE(def_use.uses.contains(RegisterRef{RegClass::VGPR, 3, 1}));
    EXPECT_TRUE(def_use.uses.contains(RegisterRef{RegClass::SGPR, kResourceBase, 8}));
    EXPECT_FALSE(def_use.uses.contains(RegisterRef{RegClass::SGPR, 2, 1}));

    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
    cu->l1_vector().flush_all();
    l2.flush_all();
  }
};

TEST_P(RdnaDecodedImageAccessTest,
       StoreThenLoadUsesPaddedRowsNonconsecutiveCoordinatesAndExecMask) {
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + 2, 0, 2);
  cu->write_vgpr(vb + 6, 0, 1);
  cu->write_vgpr(vb + 2, 1, 0);
  cu->write_vgpr(vb + 6, 1, 0);
  cu->write_vgpr(vb + 2, 2, 1);
  cu->write_vgpr(vb + 6, 2, 0);
  for (uint32_t lane = 0; lane < 3; ++lane)
    cu->write_vgpr(vb + 3, lane, 0); // Consecutive-register bug reads this as Y.

  cu->write_vgpr(vb + 8, 0, 0x11111111);
  cu->write_vgpr(vb + 8, 1, 0x22222222);
  cu->write_vgpr(vb + 8, 2, 0x33333333);
  issue(false);
  EXPECT_EQ(memory.read32(kBase + 264), 0x11111111u);
  EXPECT_EQ(memory.read32(kBase), 0u);
  EXPECT_EQ(memory.read32(kBase + 4), 0x33333333u);

  for (uint32_t lane = 0; lane < 3; ++lane)
    cu->write_vgpr(vb + 8, lane, 0xdeadbeef);
  issue(true);
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x11111111u);
  EXPECT_EQ(cu->read_vgpr(vb + 8, 1), 0xdeadbeefu);
  EXPECT_EQ(cu->read_vgpr(vb + 8, 2), 0x33333333u);
}

TEST_P(RdnaDecodedImageAccessTest, NarrowRowPitchIsRejectedAndUnitWidthRemainsValid) {
  const uint32_t vb = wf->vgpr_alloc().base;
  wf->set_exec(1);
  cu->write_vgpr(vb + 2, 0, 0);
  cu->write_vgpr(vb + 6, 0, 1);
  memory.write32(kBase, 0xaaaaaaaa);
  memory.write32(kBase + 4, 0xbbbbbbbb);

  bind_resource(RdnaImageDescriptor{.width = 64, .height = 2, .pitch = 1});
  for (bool load : {false, true}) {
    SCOPED_TRACE(testing::Message() << "load=" << load);
    cu->write_vgpr(vb + 8, 0, 0xdeadbeef);
    auto decoded = decode(load);
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_TRUE(cu->execute_instruction(decoded.value().get(), *wf).failed());
    EXPECT_EQ(wf->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(decoded.value()->data(), nullptr);
    EXPECT_EQ(memory.read32(kBase + 4), 0xbbbbbbbbu);
    EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0xdeadbeefu);
  }

  bind_resource(RdnaImageDescriptor{.width = 64, .height = 2, .pitch = 64});
  cu->write_vgpr(vb + 8, 0, 0x12345678);
  issue(false);
  EXPECT_EQ(memory.read32(kBase + 64 * 4), 0x12345678u);
  cu->write_vgpr(vb + 8, 0, 0);
  issue(true);
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x12345678u);

  bind_resource(RdnaImageDescriptor{.width = 1, .height = 2, .pitch = 1});
  cu->write_vgpr(vb + 8, 0, 0x87654321);
  issue(false);
  EXPECT_EQ(memory.read32(kBase + 4), 0x87654321u);
}

TEST_P(RdnaDecodedImageAccessTest, StoreDmaskExpandsPackedInputsAndAppliesQualifiedSelectors) {
  const uint32_t vb = wf->vgpr_alloc().base;
  wf->set_exec(1);
  cu->write_vgpr(vb + 2, 0, 0);
  cu->write_vgpr(vb + 6, 0, 0);
  constexpr uint32_t kBgra = 6 | (5 << 3) | (4 << 6) | (7 << 9);
  struct Case {
    uint32_t selectors;
    uint8_t dmask;
    bool d16;
    std::array<uint32_t, 4> packed;
    std::array<uint32_t, 4> gfx11;
    std::array<uint32_t, 4> gfx12;
  };
  const Case cases[] = {
      {kIdentity, 1, false, {0x11}, {0x11, 0, 0, 0}, {0x11, 0x11, 0x11, 0x11}},
      {kIdentity, 3, false, {0x11, 0x22}, {0x11, 0x22, 0, 0}, {0x11, 0x22, 0x11, 0x11}},
      // GFX12 replicates X only when DMASK supplies X.
      {kIdentity, 2, false, {0x22}, {0, 0x22, 0, 0}, {0, 0x22, 0, 0}},
      // Sparse D16 inputs remain packed in VDATA but route through X/Z.
      {kIdentity, 5, true, {0x00330011}, {0x11, 0, 0x33, 0}, {0x11, 0x11, 0x33, 0x11}},
      {kBgra, 1, false, {0x11}, {0, 0, 0x11, 0}, {0x11, 0x11, 0x11, 0x11}},
      // XYZ1/ZYX1 write constant alpha independently of DMASK.
      {kXyzOne, 15, false, {0x11, 0x22, 0x33, 0x44}, {0x11, 0x22, 0x33, 1}, {0x11, 0x22, 0x33, 1}},
      {kXyzOne, 7, false, {0x11, 0x22, 0x33}, {0x11, 0x22, 0x33, 1}, {0x11, 0x22, 0x33, 1}},
      {kXyzOne, 1, false, {0x11}, {0x11, 0, 0, 1}, {0x11, 0x11, 0x11, 1}},
      {kZyxOne, 15, false, {0x11, 0x22, 0x33, 0x44}, {0x33, 0x22, 0x11, 1}, {0x33, 0x22, 0x11, 1}},
      {kZyxOne, 7, false, {0x11, 0x22, 0x33}, {0x33, 0x22, 0x11, 1}, {0x33, 0x22, 0x11, 1}},
      {kZyxOne, 1, false, {0x11}, {0, 0, 0x11, 1}, {0x11, 0x11, 0x11, 1}},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(testing::Message() << "selectors=" << test.selectors
                                    << " dmask=" << unsigned{test.dmask} << " d16=" << test.d16);
    bind_resource(kRgba32Uint, test.selectors);
    for (uint32_t offset = 0; offset < 4; ++offset) {
      memory.write32(kBase + offset * 4, 0xdeadbeef);
      cu->write_vgpr(vb + 8 + offset, 0, test.packed[offset]);
    }
    issue(false, {.dmask = test.dmask, .d16 = test.d16});
    const auto &expected = GetParam() == ROCJITSU_CODE_ARCH_RDNA4 ? test.gfx12 : test.gfx11;
    for (uint32_t offset = 0; offset < expected.size(); ++offset)
      EXPECT_EQ(memory.read32(kBase + offset * 4), expected[offset]) << "offset=" << offset;
  }
}

TEST_P(RdnaDecodedImageAccessTest, PackedRgbStoresWriteConstantAlpha) {
  const uint32_t vb = wf->vgpr_alloc().base;
  wf->set_exec(1);
  cu->write_vgpr(vb + 2, 0, 0);
  cu->write_vgpr(vb + 6, 0, 0);
  constexpr uint32_t kRgba2101010Uint = 40;
  for (const auto &[selectors, expected] :
       {std::pair{kXyzOne, 0x11u | (0x22u << 10) | (0x33u << 20) | (1u << 30)},
        std::pair{kZyxOne, 0x33u | (0x22u << 10) | (0x11u << 20) | (1u << 30)}}) {
    SCOPED_TRACE(selectors);
    bind_resource(kRgba2101010Uint, selectors);
    for (uint32_t component = 0; component < 4; ++component)
      cu->write_vgpr(vb + 8 + component, 0, 0x11 * (component + 1));
    issue(false, {.dmask = 15});
    EXPECT_EQ(memory.read32(kBase), expected);
  }
}

TEST_P(RdnaDecodedImageAccessTest, UnmodeledStoreSelectorsReportUnimplemented) {
  const uint32_t vb = wf->vgpr_alloc().base;
  wf->set_exec(1);
  cu->write_vgpr(vb + 2, 0, 0);
  cu->write_vgpr(vb + 6, 0, 0);
  constexpr uint32_t kXyzz = 4 | (5 << 3) | (6 << 6) | (6 << 9);
  constexpr uint32_t kX000 = 4;
  for (const uint32_t selectors : {kXyzz, kX000}) {
    SCOPED_TRACE(selectors);
    bind_resource(kRgba32Uint, selectors);
    auto decoded = decode(false, {.dmask = 15});
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_TRUE(cu->execute_instruction(decoded.value().get(), *wf).failed());
    EXPECT_EQ(wf->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
  }

  // The single-channel X001 layout remains supported.
  bind_resource(20, kXOne);
  auto decoded = decode(false);
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_TRUE(cu->execute_instruction(decoded.value().get(), *wf).succeeded());
}

TEST_P(RdnaDecodedImageAccessTest, D16LoadPreservesAndReadsOnlyThePartialDestination) {
  const uint32_t vb = wf->vgpr_alloc().base;
  wf->set_exec(1);
  cu->write_vgpr(vb + 2, 0, 0);
  cu->write_vgpr(vb + 6, 0, 0);
  memory.write32(kBase, 0x1234);

  for (const bool d16 : {false, true}) {
    for (const uint8_t dmask : {1, 3, 7, 15}) {
      SCOPED_TRACE(testing::Message() << "d16=" << d16 << " dmask=" << unsigned{dmask});
      for (uint32_t reg = 8; reg < 12; ++reg)
        cu->write_vgpr(vb + reg, 0, 0xabcdffff);

      auto decoded = decode(true, {.dmask = dmask, .d16 = d16});
      ASSERT_TRUE(decoded.succeeded());
      std::unique_ptr<Instruction> inst = std::move(decoded).value();
      ASSERT_NE(inst, nullptr);

      const uint32_t components = std::popcount(dmask);
      const uint32_t registers = d16 ? (components + 1) / 2 : components;
      const bool preserves_last = d16 && (components & 1u);
      const InstDefUse def_use(*inst);
      EXPECT_TRUE(def_use.defs.contains({RegClass::VGPR, 8, static_cast<uint8_t>(registers)}));
      for (uint32_t offset = 0; offset < registers; ++offset) {
        const bool expected_use = preserves_last && offset == components / 2;
        EXPECT_EQ(def_use.uses.contains({RegClass::VGPR, static_cast<uint16_t>(8 + offset), 1}),
                  expected_use)
            << "offset=" << offset;
      }

      ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
      amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
      ASSERT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
      cu->l1_vector().flush_all();
      l2.flush_all();

      for (uint32_t offset = 0; offset < registers; ++offset) {
        uint32_t expected = offset == 0 ? 0x1234 : 0;
        if (!d16 && dmask == 15 && offset == 3)
          expected = 1; // Descriptor W selector is constant one.
        if (d16 && dmask == 15 && offset == 1)
          expected = 1u << 16;
        if (preserves_last && offset == components / 2)
          expected |= 0xabcd0000;
        EXPECT_EQ(cu->read_vgpr(vb + 8 + offset, 0), expected) << "offset=" << offset;
      }
    }
  }
}

TEST_P(RdnaDecodedImageAccessTest, TfeLoadReportsUnimplementedBeforeProducingData) {
  const uint32_t vb = wf->vgpr_alloc().base;
  wf->set_exec(1);
  cu->write_vgpr(vb + 2, 0, 0);
  cu->write_vgpr(vb + 6, 0, 0);
  cu->write_vgpr(vb + 8, 0, 0xaaaaaaaa);
  cu->write_vgpr(vb + 9, 0, 0xdeadbeef);
  memory.write32(kBase, 0x12345678);

  auto decoded = decode(true, {.tfe = true});
  ASSERT_TRUE(decoded.succeeded());
  std::unique_ptr<Instruction> inst = std::move(decoded).value();
  ASSERT_NE(inst, nullptr);
  const InstDefUse def_use(*inst);
  EXPECT_TRUE(def_use.defs.contains({RegClass::VGPR, 8, 2}));

  EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).failed());
  EXPECT_EQ(wf->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
  EXPECT_EQ(inst->data(), nullptr);
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0xaaaaaaaau);
  EXPECT_EQ(cu->read_vgpr(vb + 9, 0), 0xdeadbeefu);
}

TEST_P(RdnaDecodedImageAccessTest, OneDimensionalDimRejectsTwoDimensionalResource) {
  for (const uint8_t dim : {amdgpu::kImageDim1d, amdgpu::kImageDim1dArray}) {
    for (const bool load : {false, true}) {
      for (const bool a16 : {false, true}) {
        for (const bool nsa : {false, true}) {
          if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4 && nsa)
            continue;
          SCOPED_TRACE(testing::Message() << "dim=" << unsigned{dim} << " load=" << load
                                          << " a16=" << a16 << " nsa=" << nsa);
          auto decoded = decode(load, {.dim = dim, .a16 = a16, .nsa = nsa});
          ASSERT_TRUE(decoded.succeeded());
          std::unique_ptr<Instruction> inst = std::move(decoded).value();
          ASSERT_NE(inst, nullptr);
          EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).failed());
          EXPECT_EQ(wf->instruction_execution_error(),
                    amdgpu::InstructionExecutionError::UnimplementedInstruction);
          EXPECT_EQ(inst->data(), nullptr);
        }
      }
    }
  }
}

TEST_P(RdnaDecodedImageAccessTest, TwoDimensionalArrayDimUsesTwoDimensionalResourcePrefix) {
  const uint32_t vb = wf->vgpr_alloc().base;
  wf->set_exec(1);
  for (const bool a16 : {false, true}) {
    for (const bool nsa : {false, true}) {
      if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4 && nsa)
        continue;
      SCOPED_TRACE(testing::Message() << "a16=" << a16 << " nsa=" << nsa);
      const bool noncontiguous = GetParam() == ROCJITSU_CODE_ARCH_RDNA4 || nsa;
      cu->write_vgpr(vb + 2, 0, a16 ? 2u | (1u << 16) : 2u);
      cu->write_vgpr(vb + 3, 0, noncontiguous ? 0xdeadbeef : 1u);
      cu->write_vgpr(vb + 6, 0, noncontiguous ? 1u : 0xdeadbeef);
      const uint32_t value = 0x12340000u | (uint32_t{a16} << 1) | uint32_t{nsa};
      cu->write_vgpr(vb + 8, 0, value);

      for (const bool load : {false, true}) {
        auto decoded = decode(load, {.dim = amdgpu::kImageDim2dArray, .a16 = a16, .nsa = nsa});
        ASSERT_TRUE(decoded.succeeded());
        std::unique_ptr<Instruction> inst = std::move(decoded).value();
        ASSERT_NE(inst, nullptr);
        const InstDefUse def_use(*inst);
        EXPECT_TRUE(def_use.uses.contains({RegClass::VGPR, 2, 1}));
        if (!a16) {
          EXPECT_TRUE(def_use.uses.contains(
              {RegClass::VGPR, static_cast<uint16_t>(noncontiguous ? 6 : 3), 1}));
        }

        ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
        amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
        ASSERT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
        cu->l1_vector().flush_all();
        l2.flush_all();
        if (!load) {
          EXPECT_EQ(memory.read32(kBase + 264), value);
          cu->write_vgpr(vb + 8, 0, 0xdeadbeef);
        } else {
          EXPECT_EQ(cu->read_vgpr(vb + 8, 0), value);
        }
      }
    }
  }
}

TEST_P(RdnaDecodedImageAccessTest, NonNsaAddressWidthFollowsDimAndA16) {
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "RDNA4 VIMAGE encodes address registers individually";

  struct Case {
    uint8_t dim;
    bool a16;
    uint32_t expected_bits;
  };
  for (const Case test : {Case{.dim = amdgpu::kImageDim1d, .a16 = false, .expected_bits = 32},
                          Case{.dim = amdgpu::kImageDim1d, .a16 = true, .expected_bits = 32},
                          Case{.dim = amdgpu::kImageDim2d, .a16 = false, .expected_bits = 64},
                          Case{.dim = amdgpu::kImageDim2d, .a16 = true, .expected_bits = 32}}) {
    for (bool load : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << "dim=" << unsigned{test.dim} << " a16=" << test.a16 << " load=" << load);
      auto decoded = decode(load, {.dim = test.dim, .a16 = test.a16, .nsa = false});
      ASSERT_TRUE(decoded.succeeded());
      const std::unique_ptr<Instruction> inst = std::move(decoded).value();
      ASSERT_NE(inst, nullptr);
      const Operand *vaddr = inst->src_operand(load ? 0 : 1);
      ASSERT_NE(vaddr, nullptr);
      EXPECT_EQ(vaddr->size_bits(), test.expected_bits);
    }
  }
}

TEST_P(RdnaDecodedImageAccessTest, R128LoadAndStoreReportUnimplemented) {
  const uint32_t sb = wf->sgpr_alloc().base;
  const uint32_t vb = wf->vgpr_alloc().base;
  for (bool poison_trailing_words : {false, true}) {
    SCOPED_TRACE(testing::Message() << "poison_trailing_words=" << poison_trailing_words);
    bind_resource();
    if (poison_trailing_words)
      for (uint32_t reg = kResourceBase + 4; reg < kResourceBase + 8; ++reg)
        cu->write_sgpr(sb + reg, 0xffffffffu);

    for (bool load : {false, true}) {
      SCOPED_TRACE(testing::Message() << "load=" << load);
      memory.write32(kBase, 0x12345678u);
      cu->write_vgpr(vb + 8, 0, 0xdeadbeefu);
      auto decoded = decode(load, {.r128 = true});
      ASSERT_TRUE(decoded.succeeded());
      std::unique_ptr<Instruction> inst = std::move(decoded).value();
      ASSERT_NE(inst, nullptr);
      EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).failed());
      EXPECT_EQ(wf->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
      EXPECT_EQ(inst->data(), nullptr);
      EXPECT_EQ(memory.read32(kBase), 0x12345678u);
      EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0xdeadbeefu);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(RdnaArchitectures, RdnaDecodedImageAccessTest,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));

TEST(ImageAccessTest, RdnaLayeredAnd3dLinearImagesRejectUnknownPitch) {
  for (auto arch :
       {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4}) {
    for (uint32_t type : {amdgpu::kImage2dArray, amdgpu::kImage3d}) {
      SCOPED_TRACE(testing::Message() << "arch=" << arch << " type=" << type);
      amdgpu::GpuMemory memory{"rdna_layered_image_memory"};
      amdgpu::L2Cache l2{"rdna_layered_image_l2"};
      l2.set_backing_memory(&memory);
      amdgpu::ComputeUnitCore::Config cfg{};
      cfg.arch = arch;
      cfg.num_wf_slots = 1;
      cfg.sgprs_per_wf = 32;
      cfg.vgprs_per_wf = 32;
      auto cu = amdgpu::ComputeUnitCore::create("rdna_layered_image_cu", cfg, &memory, &l2);
      auto *wf = cu->dispatch_wf(0, 0, 32, 32);
      ASSERT_NE(wf, nullptr);
      wf->set_exec(1);

      const auto resource = rdna_image_descriptor(arch, {.type = type});
      for (uint32_t i = 0; i < resource.size(); ++i)
        cu->write_sgpr(wf->sgpr_alloc().base + i, resource[i]);
      cu->write_vgpr(wf->vgpr_alloc().base, 0, 0);
      cu->write_vgpr(wf->vgpr_alloc().base + 1, 0, 1);
      cu->write_vgpr(wf->vgpr_alloc().base + 2, 0, 0);

      amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
      state.is_load = true;
      EXPECT_TRUE(
          amdgpu::prepare_image_access(
              *wf, state, 0, amdgpu::contiguous_image_address_registers(0),
              {.dmask = 1,
               .dim = type == amdgpu::kImage3d ? amdgpu::kImageDim3d : amdgpu::kImageDim2dArray,
               .a16 = false,
               .d16 = false,
               .lwe = false,
               .da = true,
               .unorm = true})
              .failed());
      EXPECT_EQ(wf->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
      wf->halt();
    }
  }
}

TEST(ImageAccessTest, UnsupportedArchitectureIsRejected) {
  amdgpu::GpuMemory memory{"unsupported_image_memory"};
  amdgpu::L2Cache l2{"unsupported_image_l2"};
  l2.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA2;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 32;
  cfg.vgprs_per_wf = 32;
  auto cu = amdgpu::ComputeUnitCore::create("unsupported_image_cu", cfg, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0, 32, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
  state.is_load = true;
  EXPECT_TRUE(amdgpu::prepare_image_access(
                  *wf, state, 0, amdgpu::contiguous_image_address_registers(0),
                  {.dmask = 1, .a16 = false, .d16 = false, .lwe = false, .da = true, .unorm = true})
                  .failed());
  EXPECT_EQ(wf->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
  wf->halt();
}

} // namespace
