// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "gfx11_mimg_test_common.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
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
#include "rocjitsu/isa/arch/amdgpu/shared/image_access.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_footprint.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_resource.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/image_addressing.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
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
constexpr uint32_t kR32Uint = 20;
constexpr uint32_t kRgba32Uint = 61;

struct Image {
  uint32_t data_format = kDfmt32;
  uint32_t num_format = kNfmtUint;
  uint32_t width = 1, height = 1, depth = 1, pitch = 64;
  uint32_t type = amdgpu::kImage2d;
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

struct RdnaImageDescriptor {
  uint32_t type = amdgpu::kImage2d;
  uint32_t format = kR32Uint;
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

TEST_F(Cdna2ImageAccessTest, DecodedMipLoadAndStoreAccessLevelZero) {
  Image image;
  image.width = 64;
  image.height = 2;
  bind(image);
  wf->set_exec(1);
  write_coordinates(0, {1, 1, 0});
  constexpr uint64_t kAddress = kBase + (64 + 1) * sizeof(uint32_t);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_NE(decoder, nullptr);

  for (const bool load : {false, true}) {
    SCOPED_TRACE(testing::Message() << "load=" << load);
    memory.write32(kAddress, 0x12345678u);
    cu->l1_vector().invalidate_all();
    l2.invalidate_all();
    cu->write_vgpr(vb + 8, 0, 0xdeadbeefu);
    const auto words =
        cdna2::build_mimg(load ? cdna2::kImageLoadMipMimg : cdna2::kImageStoreMipMimg,
                          {.dmask = 1, .unorm = 1, .da = 1, .vaddr = 0, .vdata = 8, .srsrc = 0});
    auto decoded = decoder->decode(words.data());
    ASSERT_TRUE(decoded.succeeded());
    std::unique_ptr<Instruction> inst = std::move(decoded).value();
    ASSERT_NE(inst, nullptr);

    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
    cu->l1_vector().flush_all();
    l2.flush_all();
    EXPECT_EQ(memory.read32(kAddress), load ? 0x12345678u : 0xdeadbeefu);
    EXPECT_EQ(cu->read_vgpr(vb + 8, 0), load ? 0x12345678u : 0xdeadbeefu);
  }
}

TEST_F(Cdna2ImageAccessTest, ZeroSelectorsDoNotUseBufferStoreA8Routing) {
  wf->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  for (uint32_t format : {kDfmt32, kDfmt8888}) {
    SCOPED_TRACE(format);
    Image image;
    image.data_format = format;
    image.selectors = 0;
    bind(image);
    for (uint32_t reg = 0; reg < 4; ++reg)
      cu->write_vgpr(vb + 8 + reg, 0, format == kDfmt32 ? 0x12345678u : 0x11u * (reg + 1));
    memory.write32(kBase, 0xdeadbeef);
    ASSERT_TRUE(issue(false));
    EXPECT_EQ(memory.read32(kBase), 0u);
  }
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

TEST_F(Cdna2ImageAccessTest, CompressedDescriptorsStopBeforeImageMemoryAccess) {
  bind(Image{});
  cu->write_sgpr(sb + 6, cu->read_sgpr(sb + 6) | (1u << 21)); // COMPRESSION_ENABLE.
  wf->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  for (bool load : {false, true}) {
    SCOPED_TRACE(load);
    memory.write32(kBase, 0x12345678u);
    fill_destination(0xdeadbeefu);
    wf->clear_instruction_execution_error();
    EXPECT_FALSE(issue(load, 1));
    EXPECT_EQ(wf->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(memory.read32(kBase), 0x12345678u);
    EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0xdeadbeefu);
  }
}

TEST_F(Cdna2ImageAccessTest, UnsupportedSwizzleModeStopsBeforeMemoryAccess) {
  Image image;
  image.width = 64;
  image.swizzle_mode = 31; // Reserved swizzle mode.
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

TEST_F(Cdna2ImageAccessTest, SrgbLoadDecodesColorAndPreservesAlpha) {
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = 9; // GFX9 TYPE_SRGB.
  image.selectors = kIdentity;
  bind(image);
  memory.write32(kBase, 0x80402000);
  ASSERT_TRUE(issue(true));
  for (uint32_t lane : {0u, 2u}) {
    EXPECT_EQ(cu->read_vgpr(vb + 8, lane), 0u);
    EXPECT_NEAR(std::bit_cast<float>(cu->read_vgpr(vb + 9, lane)), 0.014443844f, 1e-7);
    EXPECT_NEAR(std::bit_cast<float>(cu->read_vgpr(vb + 10, lane)), 0.051269458f, 1e-7);
    EXPECT_FLOAT_EQ(std::bit_cast<float>(cu->read_vgpr(vb + 11, lane)), 128.0f / 255);
  }
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
  amdgpu::GpuMemory memory_{"rdna_image_memory"};
  amdgpu::L2Cache l2_{"rdna_image_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  std::unique_ptr<Decoder> decoder_;
  amdgpu::Wavefront *wf_ = nullptr;

  void SetUp() override {
    l2_.set_backing_memory(&memory_);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = GetParam();
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 32;
    cfg.vgprs_per_wf = 32;
    cu_ = amdgpu::ComputeUnitCore::create("rdna_image_cu", cfg, &memory_, &l2_);
    wf_ = cu_->dispatch_wf(0, 0, 32, 32);
    ASSERT_NE(wf_, nullptr);
    wf_->set_exec(5); // Lane 1 must not store or receive a load result.
    decoder_ = Decoder::create(GetParam());
    ASSERT_NE(decoder_, nullptr);

    bind_resource();
  }

  void bind_resource(const RdnaImageDescriptor &descriptor) {
    const auto resource = rdna_image_descriptor(GetParam(), descriptor);
    for (uint32_t i = 0; i < resource.size(); ++i)
      cu_->write_sgpr(wf_->sgpr_alloc().base + kResourceBase + i, resource[i]);
  }

  void bind_resource(uint32_t format = kR32Uint, uint32_t selectors = kXOne) {
    bind_resource(RdnaImageDescriptor{.format = format, .selectors = selectors});
  }

  void TearDown() override {
    if (wf_)
      wf_->halt();
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
          *decoder_, rdna3::build_mimg, load ? rdna3::kImageLoadMimg : rdna3::kImageStoreMimg,
          mimg);
    }
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5) {
      return test::decode_gfx11_mimg<rdna3_5::MimgBuilderFields>(
          *decoder_, rdna3_5::build_mimg, load ? rdna3_5::kImageLoadMimg : rdna3_5::kImageStoreMimg,
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
    return decoder_->decode(words.data());
  }

  DecodeResult decode_mip(bool load) {
    const test::Gfx11MimgEncoding mimg{.nsa = true,
                                       .dim = amdgpu::kImageDim2d,
                                       .unorm = true,
                                       .dmask = 1,
                                       .vaddr = 2,
                                       .vdata = 8,
                                       .srsrc = 2,
                                       .nsa_word = 6};
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3) {
      return test::decode_gfx11_mimg<rdna3::MimgBuilderFields>(
          *decoder_, rdna3::build_mimg, load ? rdna3::kImageLoadMipMimg : rdna3::kImageStoreMipMimg,
          mimg);
    }
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5) {
      return test::decode_gfx11_mimg<rdna3_5::MimgBuilderFields>(
          *decoder_, rdna3_5::build_mimg,
          load ? rdna3_5::kImageLoadMipMimg : rdna3_5::kImageStoreMipMimg, mimg);
    }
    const auto words =
        rdna4::build_vimage(load ? rdna4::kImageLoadMipVimage : rdna4::kImageStoreMipVimage,
                            {.dim = amdgpu::kImageDim2d,
                             .dmask = 1,
                             .vdata = 8,
                             .rsrc = kResourceBase,
                             .vaddr0 = 2,
                             .vaddr1 = 6,
                             .vaddr2 = 7});
    return decoder_->decode(words.data());
  }

  void issue(bool load) { issue(load, Encoding{}); }

  void issue(bool load, Encoding encoding) {
    auto decoded = decode(load, encoding);
    ASSERT_TRUE(decoded.succeeded());
    std::unique_ptr<Instruction> inst = std::move(decoded).value();
    ASSERT_NE(inst, nullptr);

    const InstDefUse def_use(*inst, wf_->wf_size());
    EXPECT_TRUE(def_use.uses.contains(RegisterRef{RegClass::VGPR, 2, 1}));
    EXPECT_TRUE(def_use.uses.contains(RegisterRef{RegClass::VGPR, 6, 1}));
    EXPECT_FALSE(def_use.uses.contains(RegisterRef{RegClass::VGPR, 3, 1}));
    EXPECT_TRUE(def_use.uses.contains(RegisterRef{RegClass::SGPR, kResourceBase, 8}));
    EXPECT_FALSE(def_use.uses.contains(RegisterRef{RegClass::SGPR, 2, 1}));

    ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    cu_->l1_vector().flush_all();
    l2_.flush_all();
  }
};

TEST_P(RdnaDecodedImageAccessTest, LinearR32DescriptorsPrepareTheExpectedMemoryAccess) {
  wf_->set_exec(1);
  bind_resource(RdnaImageDescriptor{.width = 64, .height = 1, .gfx12_bit_157 = true});
  const uint32_t vb = wf_->vgpr_alloc().base;
  cu_->write_vgpr(vb, 0, 0);
  cu_->write_vgpr(vb + 1, 0, 0);

  amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
  state.is_load = true;
  ASSERT_TRUE(amdgpu::prepare_image_access(*wf_, state, kResourceBase,
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
  EXPECT_EQ(state.buffer_format, kR32Uint);
}

TEST_P(RdnaDecodedImageAccessTest, DecodedLoadAndStorePrintNoncontiguousCoordinates) {
  for (const bool load : {false, true}) {
    SCOPED_TRACE(testing::Message() << "load=" << load);
    auto decoded = decode(load);
    ASSERT_TRUE(decoded.succeeded());
    const std::unique_ptr<Instruction> inst = std::move(decoded).value();
    ASSERT_NE(inst, nullptr);
    std::string expected =
        load ? "image_load v8, [v2, v6], s[8:15]" : "image_store v8, [v2, v6], s[8:15]";
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
      expected += " dmask:0x1 dim:SQ_RSRC_IMG_2D unorm";
    EXPECT_EQ(inst->disassemble(), expected);

    const InstDefUse def_use(*inst, wf_->wf_size());
    EXPECT_TRUE(def_use.uses.contains({RegClass::VGPR, 2, 1}));
    EXPECT_TRUE(def_use.uses.contains({RegClass::VGPR, 6, 1}));
    EXPECT_FALSE(def_use.uses.contains({RegClass::VGPR, 3, 1}));
  }
}

TEST(ImageAccessTest, Rdna3MipUsesNsaSelectorsAndAtomicUsesBaseAddressPrinter) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_NE(decoder, nullptr);
  const test::Gfx11MimgEncoding mimg{.nsa = true,
                                     .dim = amdgpu::kImageDim2d,
                                     .unorm = true,
                                     .dmask = 1,
                                     .vaddr = 2,
                                     .vdata = 8,
                                     .srsrc = 2,
                                     .nsa_word = 6};
  // TODO(https://github.com/ROCm/rocm-systems/issues/14684): The encoding-base
  // fallback prints a contiguous VADDR tuple rather than the exact NSA selectors.
  // Repair its disassembly together with the corresponding dependency metadata.
  struct Case {
    uint16_t opcode;
    const char *disassembly;
  };
  for (const Case test_case : {
           Case{rdna3::kImageLoadMipMimg, "image_load_mip v8, [v2, v6, v0], s[8:15]"},
           Case{rdna3::kImageAtomicAddMimg, "image_atomic_add v8, v[2:3], s[8:15]"},
       }) {
    SCOPED_TRACE(test_case.disassembly);
    auto decoded = test::decode_gfx11_mimg<rdna3::MimgBuilderFields>(*decoder, rdna3::build_mimg,
                                                                     test_case.opcode, mimg);
    ASSERT_TRUE(decoded.succeeded());
    const std::unique_ptr<Instruction> inst = std::move(decoded).value();
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->disassemble(), test_case.disassembly);
  }
}

TEST_P(RdnaDecodedImageAccessTest, CompressionControlsStopBeforeImageMemoryAccess) {
  const uint32_t sb = wf_->sgpr_alloc().base + kResourceBase;
  const uint32_t vb = wf_->vgpr_alloc().base;
  const uint32_t original = cu_->read_sgpr(sb + 6);
  wf_->set_exec(1);
  cu_->write_vgpr(vb + 2, 0, 0);
  cu_->write_vgpr(vb + 6, 0, 0);
  // WRITE_COMPRESS_ENABLE and COMPRESSION_ENABLE, individually and together.
  for (uint32_t flags : {0u, 1u << 20, 1u << 21, 3u << 20}) {
    cu_->write_sgpr(sb + 6, original | flags);
    for (bool load : {false, true}) {
      SCOPED_TRACE(testing::Message() << "flags=" << flags << " load=" << load);
      memory_.write32(kBase, 0x12345678u);
      cu_->write_vgpr(vb + 8, 0, 0xdeadbeefu);
      wf_->clear_instruction_execution_error();
      auto decoded = decode(load);
      ASSERT_TRUE(decoded.succeeded());
      std::unique_ptr<Instruction> inst = std::move(decoded).value();
      EXPECT_EQ(cu_->execute_instruction(inst.get(), *wf_).failed(), flags != 0);
      if (flags == 0) {
        EXPECT_NE(inst->data(), nullptr);
      } else {
        EXPECT_EQ(wf_->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnimplementedInstruction);
        EXPECT_EQ(inst->data(), nullptr);
      }
      EXPECT_EQ(memory_.read32(kBase), 0x12345678u);
      EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), 0xdeadbeefu);
    }
  }
}

TEST_P(RdnaDecodedImageAccessTest, MipLoadAndStoreAccessLevelZero) {
  wf_->set_exec(1);
  const uint32_t vb = wf_->vgpr_alloc().base;
  cu_->write_vgpr(vb + 2, 0, 1);
  cu_->write_vgpr(vb + 6, 0, 1);
  cu_->write_vgpr(vb + 7, 0, 0);
  constexpr uint64_t kAddress = kBase + (64 + 1) * sizeof(uint32_t);

  for (const bool load : {false, true}) {
    SCOPED_TRACE(testing::Message() << "load=" << load);
    memory_.write32(kAddress, 0x12345678u);
    cu_->write_vgpr(vb + 8, 0, 0xdeadbeefu);
    auto decoded = decode_mip(load);
    ASSERT_TRUE(decoded.succeeded());
    std::unique_ptr<Instruction> inst = std::move(decoded).value();
    ASSERT_NE(inst, nullptr);

    ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    cu_->l1_vector().flush_all();
    l2_.flush_all();
    EXPECT_EQ(memory_.read32(kAddress), load ? 0x12345678u : 0xdeadbeefu);
    EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), load ? 0x12345678u : 0xdeadbeefu);
  }
}

TEST_P(RdnaDecodedImageAccessTest,
       StoreThenLoadUsesPaddedRowsNonconsecutiveCoordinatesAndExecMask) {
  const uint32_t vb = wf_->vgpr_alloc().base;
  cu_->write_vgpr(vb + 2, 0, 2);
  cu_->write_vgpr(vb + 6, 0, 1);
  cu_->write_vgpr(vb + 2, 1, 0);
  cu_->write_vgpr(vb + 6, 1, 0);
  cu_->write_vgpr(vb + 2, 2, 1);
  cu_->write_vgpr(vb + 6, 2, 0);
  for (uint32_t lane = 0; lane < 3; ++lane)
    cu_->write_vgpr(vb + 3, lane, 0); // Consecutive-register bug reads this as Y.

  cu_->write_vgpr(vb + 8, 0, 0x11111111);
  cu_->write_vgpr(vb + 8, 1, 0x22222222);
  cu_->write_vgpr(vb + 8, 2, 0x33333333);
  issue(false);
  EXPECT_EQ(memory_.read32(kBase + 264), 0x11111111u);
  EXPECT_EQ(memory_.read32(kBase), 0u);
  EXPECT_EQ(memory_.read32(kBase + 4), 0x33333333u);

  for (uint32_t lane = 0; lane < 3; ++lane)
    cu_->write_vgpr(vb + 8, lane, 0xdeadbeef);
  issue(true);
  EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), 0x11111111u);
  EXPECT_EQ(cu_->read_vgpr(vb + 8, 1), 0xdeadbeefu);
  EXPECT_EQ(cu_->read_vgpr(vb + 8, 2), 0x33333333u);
}

TEST_P(RdnaDecodedImageAccessTest, NarrowRowPitchIsRejectedAndUnitWidthUsesQualifiedAlignment) {
  const uint32_t vb = wf_->vgpr_alloc().base;
  wf_->set_exec(1);
  cu_->write_vgpr(vb + 2, 0, 0);
  cu_->write_vgpr(vb + 6, 0, 1);
  memory_.write32(kBase, 0xaaaaaaaa);
  memory_.write32(kBase + 4, 0xbbbbbbbb);

  bind_resource(RdnaImageDescriptor{.width = 64, .height = 2, .pitch = 1});
  for (bool load : {false, true}) {
    SCOPED_TRACE(testing::Message() << "load=" << load);
    cu_->write_vgpr(vb + 8, 0, 0xdeadbeef);
    auto decoded = decode(load);
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).failed());
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(decoded.value()->data(), nullptr);
    EXPECT_EQ(memory_.read32(kBase + 4), 0xbbbbbbbbu);
    EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), 0xdeadbeefu);
  }

  bind_resource(RdnaImageDescriptor{.width = 64, .height = 2, .pitch = 64});
  cu_->write_vgpr(vb + 8, 0, 0x12345678);
  issue(false);
  EXPECT_EQ(memory_.read32(kBase + 64 * 4), 0x12345678u);
  cu_->write_vgpr(vb + 8, 0, 0);
  issue(true);
  EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), 0x12345678u);

  bind_resource(RdnaImageDescriptor{.width = 1, .height = 2, .pitch = 1});
  cu_->write_vgpr(vb + 8, 0, 0x87654321);
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5) {
    auto decoded = decode(false);
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).failed());
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
  } else {
    issue(false);
    // Physical gfx1100/gfx1201 captures establish the minimum row quantum.
    const uint32_t row_bytes = GetParam() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
    EXPECT_EQ(memory_.read32(kBase + row_bytes), 0x87654321u);
  }
  EXPECT_EQ(memory_.read32(kBase + 4), 0xbbbbbbbbu);
}

TEST_P(RdnaDecodedImageAccessTest, StoreDmaskExpandsPackedInputsAndAppliesQualifiedSelectors) {
  const uint32_t vb = wf_->vgpr_alloc().base;
  wf_->set_exec(1);
  cu_->write_vgpr(vb + 2, 0, 0);
  cu_->write_vgpr(vb + 6, 0, 0);
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
      memory_.write32(kBase + offset * 4, 0xdeadbeef);
      cu_->write_vgpr(vb + 8 + offset, 0, test.packed[offset]);
    }
    issue(false, {.dmask = test.dmask, .d16 = test.d16});
    const auto &expected = GetParam() == ROCJITSU_CODE_ARCH_RDNA4 ? test.gfx12 : test.gfx11;
    for (uint32_t offset = 0; offset < expected.size(); ++offset)
      EXPECT_EQ(memory_.read32(kBase + offset * 4), expected[offset]) << "offset=" << offset;
  }
}

TEST_P(RdnaDecodedImageAccessTest, PackedRgbStoresWriteConstantAlpha) {
  const uint32_t vb = wf_->vgpr_alloc().base;
  wf_->set_exec(1);
  cu_->write_vgpr(vb + 2, 0, 0);
  cu_->write_vgpr(vb + 6, 0, 0);
  constexpr uint32_t kRgba2101010Uint = 40;
  for (const auto &[selectors, expected] :
       {std::pair{kXyzOne, 0x11u | (0x22u << 10) | (0x33u << 20) | (1u << 30)},
        std::pair{kZyxOne, 0x33u | (0x22u << 10) | (0x11u << 20) | (1u << 30)}}) {
    SCOPED_TRACE(selectors);
    bind_resource(kRgba2101010Uint, selectors);
    for (uint32_t component = 0; component < 4; ++component)
      cu_->write_vgpr(vb + 8 + component, 0, 0x11 * (component + 1));
    issue(false, {.dmask = 15});
    EXPECT_EQ(memory_.read32(kBase), expected);
  }
}

TEST_P(RdnaDecodedImageAccessTest, UnmodeledStoreSelectorsReportUnimplemented) {
  const uint32_t vb = wf_->vgpr_alloc().base;
  wf_->set_exec(1);
  cu_->write_vgpr(vb + 2, 0, 0);
  cu_->write_vgpr(vb + 6, 0, 0);
  constexpr uint32_t kXyzz = 4 | (5 << 3) | (6 << 6) | (6 << 9);
  constexpr uint32_t kX000 = 4;
  for (const uint32_t selectors : {kXyzz, kX000}) {
    SCOPED_TRACE(selectors);
    bind_resource(kRgba32Uint, selectors);
    auto decoded = decode(false, {.dmask = 15});
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).failed());
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
  }

  // The single-channel X001 layout remains supported.
  bind_resource();
  auto decoded = decode(false);
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).succeeded());
}

TEST_P(RdnaDecodedImageAccessTest, D16LoadPreservesAndReadsOnlyThePartialDestination) {
  const uint32_t vb = wf_->vgpr_alloc().base;
  wf_->set_exec(1);
  cu_->write_vgpr(vb + 2, 0, 0);
  cu_->write_vgpr(vb + 6, 0, 0);
  memory_.write32(kBase, 0x1234);

  for (const bool d16 : {false, true}) {
    for (const uint8_t dmask : {1, 3, 7, 15}) {
      SCOPED_TRACE(testing::Message() << "d16=" << d16 << " dmask=" << unsigned{dmask});
      for (uint32_t reg = 8; reg < 12; ++reg)
        cu_->write_vgpr(vb + reg, 0, 0xabcdffff);

      auto decoded = decode(true, {.dmask = dmask, .d16 = d16});
      ASSERT_TRUE(decoded.succeeded());
      std::unique_ptr<Instruction> inst = std::move(decoded).value();
      ASSERT_NE(inst, nullptr);

      const uint32_t components = std::popcount(dmask);
      const uint32_t registers = d16 ? (components + 1) / 2 : components;
      const bool preserves_last = d16 && (components & 1u);
      const InstDefUse def_use(*inst, wf_->wf_size());
      EXPECT_TRUE(def_use.defs.contains({RegClass::VGPR, 8, static_cast<uint8_t>(registers)}));
      for (uint32_t offset = 0; offset < registers; ++offset) {
        const bool expected_use = preserves_last && offset == components / 2;
        EXPECT_EQ(def_use.uses.contains({RegClass::VGPR, static_cast<uint16_t>(8 + offset), 1}),
                  expected_use)
            << "offset=" << offset;
      }

      ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
      amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
      ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
      cu_->l1_vector().flush_all();
      l2_.flush_all();

      for (uint32_t offset = 0; offset < registers; ++offset) {
        uint32_t expected = offset == 0 ? 0x1234 : 0;
        if (!d16 && dmask == 15 && offset == 3)
          expected = 1; // Descriptor W selector is constant one.
        if (d16 && dmask == 15 && offset == 1)
          expected = 1u << 16;
        if (preserves_last && offset == components / 2)
          expected |= 0xabcd0000;
        EXPECT_EQ(cu_->read_vgpr(vb + 8 + offset, 0), expected) << "offset=" << offset;
      }
    }
  }
}

TEST_P(RdnaDecodedImageAccessTest, TfeLoadReportsUnimplementedBeforeProducingData) {
  const uint32_t vb = wf_->vgpr_alloc().base;
  wf_->set_exec(1);
  cu_->write_vgpr(vb + 2, 0, 0);
  cu_->write_vgpr(vb + 6, 0, 0);
  cu_->write_vgpr(vb + 8, 0, 0xaaaaaaaa);
  cu_->write_vgpr(vb + 9, 0, 0xdeadbeef);
  memory_.write32(kBase, 0x12345678);

  auto decoded = decode(true, {.tfe = true});
  ASSERT_TRUE(decoded.succeeded());
  std::unique_ptr<Instruction> inst = std::move(decoded).value();
  ASSERT_NE(inst, nullptr);
  const InstDefUse def_use(*inst, wf_->wf_size());
  EXPECT_TRUE(def_use.defs.contains({RegClass::VGPR, 8, 2}));

  EXPECT_TRUE(cu_->execute_instruction(inst.get(), *wf_).failed());
  EXPECT_EQ(wf_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
  EXPECT_EQ(inst->data(), nullptr);
  EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), 0xaaaaaaaau);
  EXPECT_EQ(cu_->read_vgpr(vb + 9, 0), 0xdeadbeefu);
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
          EXPECT_TRUE(cu_->execute_instruction(inst.get(), *wf_).failed());
          EXPECT_EQ(wf_->instruction_execution_error(),
                    amdgpu::InstructionExecutionError::UnimplementedInstruction);
          EXPECT_EQ(inst->data(), nullptr);
        }
      }
    }
  }
}

TEST_P(RdnaDecodedImageAccessTest, TwoDimensionalArrayDimUsesTwoDimensionalResourcePrefix) {
  const uint32_t vb = wf_->vgpr_alloc().base;
  wf_->set_exec(1);
  for (const bool a16 : {false, true}) {
    for (const bool nsa : {false, true}) {
      if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4 && nsa)
        continue;
      SCOPED_TRACE(testing::Message() << "a16=" << a16 << " nsa=" << nsa);
      const bool noncontiguous = GetParam() == ROCJITSU_CODE_ARCH_RDNA4 || nsa;
      cu_->write_vgpr(vb + 2, 0, a16 ? 2u | (1u << 16) : 2u);
      cu_->write_vgpr(vb + 3, 0, noncontiguous ? 0xdeadbeef : 1u);
      cu_->write_vgpr(vb + 6, 0, noncontiguous ? 1u : 0xdeadbeef);
      const uint32_t value = 0x12340000u | (uint32_t{a16} << 1) | uint32_t{nsa};
      cu_->write_vgpr(vb + 8, 0, value);

      for (const bool load : {false, true}) {
        auto decoded = decode(load, {.dim = amdgpu::kImageDim2dArray, .a16 = a16, .nsa = nsa});
        ASSERT_TRUE(decoded.succeeded());
        std::unique_ptr<Instruction> inst = std::move(decoded).value();
        ASSERT_NE(inst, nullptr);
        const InstDefUse def_use(*inst, wf_->wf_size());
        EXPECT_TRUE(def_use.uses.contains({RegClass::VGPR, 2, 1}));
        if (!a16) {
          EXPECT_TRUE(def_use.uses.contains(
              {RegClass::VGPR, static_cast<uint16_t>(noncontiguous ? 6 : 3), 1}));
        }

        ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
        ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
        cu_->l1_vector().flush_all();
        l2_.flush_all();
        if (!load) {
          EXPECT_EQ(memory_.read32(kBase + 264), value);
          cu_->write_vgpr(vb + 8, 0, 0xdeadbeef);
        } else {
          EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), value);
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
    const char *address;
  };
  for (const Case test :
       {Case{.dim = amdgpu::kImageDim1d, .a16 = false, .expected_bits = 32, .address = "v2"},
        Case{.dim = amdgpu::kImageDim1d, .a16 = true, .expected_bits = 32, .address = "v2"},
        Case{.dim = amdgpu::kImageDim2d, .a16 = false, .expected_bits = 64, .address = "v[2:3]"},
        Case{.dim = amdgpu::kImageDim2d, .a16 = true, .expected_bits = 32, .address = "v2"}}) {
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
      std::string expected =
          std::string(load ? "image_load v8, " : "image_store v8, ") + test.address + ", s[8:15]";
      if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5) {
        expected += std::string{" dmask:0x1 dim:SQ_RSRC_IMG_"} +
                    (test.dim == amdgpu::kImageDim1d ? "1D" : "2D") + " unorm";
        if (test.a16)
          expected += " a16";
      }
      EXPECT_EQ(inst->disassemble(), expected);
    }
  }
}

TEST_P(RdnaDecodedImageAccessTest, R128LoadAndStoreReportUnimplemented) {
  const uint32_t sb = wf_->sgpr_alloc().base;
  const uint32_t vb = wf_->vgpr_alloc().base;
  for (bool poison_trailing_words : {false, true}) {
    SCOPED_TRACE(testing::Message() << "poison_trailing_words=" << poison_trailing_words);
    bind_resource();
    if (poison_trailing_words)
      for (uint32_t reg = kResourceBase + 4; reg < kResourceBase + 8; ++reg)
        cu_->write_sgpr(sb + reg, 0xffffffffu);

    for (bool load : {false, true}) {
      SCOPED_TRACE(testing::Message() << "load=" << load);
      memory_.write32(kBase, 0x12345678u);
      cu_->write_vgpr(vb + 8, 0, 0xdeadbeefu);
      auto decoded = decode(load, {.r128 = true});
      ASSERT_TRUE(decoded.succeeded());
      std::unique_ptr<Instruction> inst = std::move(decoded).value();
      ASSERT_NE(inst, nullptr);
      EXPECT_TRUE(cu_->execute_instruction(inst.get(), *wf_).failed());
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
      EXPECT_EQ(inst->data(), nullptr);
      EXPECT_EQ(memory_.read32(kBase), 0x12345678u);
      EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), 0xdeadbeefu);
    }
  }
}

TEST_P(RdnaDecodedImageAccessTest, LayeredAnd3dLinearImagesUseImplicitRowPitch) {
  wf_->set_exec(1);
  const uint32_t vb = wf_->vgpr_alloc().base;
  for (uint32_t type : {amdgpu::kImage2dArray, amdgpu::kImage3d}) {
    SCOPED_TRACE(testing::Message() << "type=" << type);
    bind_resource(RdnaImageDescriptor{.type = type});
    cu_->write_vgpr(vb, 0, 0);
    cu_->write_vgpr(vb + 1, 0, 1);
    cu_->write_vgpr(vb + 2, 0, 0);

    amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
    state.is_load = true;
    EXPECT_TRUE(
        amdgpu::prepare_image_access(
            *wf_, state, kResourceBase, amdgpu::contiguous_image_address_registers(0),
            {.dmask = 1,
             .dim = type == amdgpu::kImage3d ? amdgpu::kImageDim3d : amdgpu::kImageDim2dArray,
             .a16 = false,
             .d16 = false,
             .lwe = false,
             .da = true,
             .unorm = true})
            .succeeded());
    const uint32_t row_bytes = GetParam() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
    EXPECT_EQ(state.per_lane_addr[0], kBase + row_bytes);
    EXPECT_EQ(state.lane_mask, 1u);
  }
}

INSTANTIATE_TEST_SUITE_P(RdnaArchitectures, RdnaDecodedImageAccessTest,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));

TEST(ImageAccessTest, UnsupportedArchitectureIsRejected) {
  amdgpu::GpuMemory memory{"unsupported_image_memory"};
  amdgpu::L2Cache l2{"unsupported_image_l2"};
  l2.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA3;
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
  if (image.type == amdgpu::kImage3d || image.type == amdgpu::kImage1dArray ||
      image.type == amdgpu::kImage2dArray)
    set(128, 13, image.depth - 1);
  else if (arch != ROCJITSU_CODE_ARCH_RDNA1)
    set(128, 14, 63); // A 256-byte row pitch.
  return words;
}

class AllImageAccessTest : public Cdna2ImageAccessTest,
                           public testing::WithParamInterface<rj_code_arch_t> {
protected:
  bool scattered = false;
  bool packed_coordinates = false;
  bool packed_data = false;
  uint32_t resource_selector = 0;
  uint32_t component_mask = 1;
  bool unnormalized = true;
  bool declare_array = true;
  std::optional<uint32_t> dimension_override;
  rj_code_arch_t arch() const override { return GetParam(); }

  void bind_image(uint32_t dimensions, uint32_t width = 3, uint32_t height = 2,
                  uint32_t swizzle = 0) {
    Image image;
    image.width = width;
    image.height = dimensions >= 2 ? height : 1;
    image.swizzle_mode = swizzle;
    image.depth = dimensions == 3 ? 2 : 1;
    image.type = dimensions == 1   ? amdgpu::kImage1d
                 : dimensions == 2 ? amdgpu::kImage2d
                                   : amdgpu::kImage3d;
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
    encoding.dmask = component_mask;
    encoding.a16 = packed_coordinates;
    encoding.d16 = packed_data;
    if constexpr (requires { encoding.srsrc; })
      encoding.srsrc = resource_selector;
    else
      encoding.rsrc = resource_selector * 4;
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
      encoding.da = declare_array;
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
  bool decoded_issue(bool load, uint32_t dimensions, bool mip = false) {
    auto inst = decoded_instruction(load, dimensions, mip);
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

TEST_P(AllImageAccessTest, ImplicitSampleUsesQuadDerivativesToSelectTheMip) {
  Image image;
  image.width = image.height = 64;
  image.swizzle_mode = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 1 : 0;
  auto resource =
      arch_is_cdna_4_or_lower(arch()) ? descriptor(image) : rdna_descriptor(image, arch());
  const bool gfx9 = arch_is_cdna_4_or_lower(arch()), gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  util::insert_bits_into_words<uint32_t>(resource.data(),
                                         gfx9                                 ? 188
                                         : gfx12                              ? 44
                                         : arch() <= ROCJITSU_CODE_ARCH_RDNA2 ? 164
                                                                              : 48,
                                         gfx12 ? 5 : 4, 3u);
  const auto bind_view = [&](uint32_t base, uint32_t last) {
    auto view = resource;
    util::insert_bits_into_words<uint32_t>(view.data(), gfx12 ? 57 : 108, gfx12 ? 5 : 4, base);
    util::insert_bits_into_words<uint32_t>(view.data(), gfx12 ? 111 : 112, gfx12 ? 5 : 4, last);
    for (uint32_t i = 0; i < view.size(); ++i)
      cu->write_sgpr(sb + i, view[i]);
  };
  const std::array<uint32_t, 4> sampler{2 | (2 << 3) | (2 << 6),
                                        gfx12 ? 0x1fffu << 13 : 0xfffu << 12, 1u << 26, 0};
  for (uint32_t i = 0; i < sampler.size(); ++i)
    cu->write_sgpr(sb + 8 + i, sampler[i]);
  std::array<uint32_t, 3> words{};
#define SAMPLE(ARCH, ENUM)                                                                         \
  case ENUM: {                                                                                     \
    auto f = fields<ARCH::MimgBuilderFields>(2);                                                   \
    f.unorm = 0;                                                                                   \
    f.ssamp = 2;                                                                                   \
    const auto encoded = ARCH::build_mimg(ARCH::kImageSampleMimg, f);                              \
    std::copy(encoded.begin(), encoded.end(), words.begin());                                      \
    break;                                                                                         \
  }
  switch (arch()) {
    SAMPLE(cdna1, ROCJITSU_CODE_ARCH_CDNA1)
    SAMPLE(cdna2, ROCJITSU_CODE_ARCH_CDNA2)
    SAMPLE(rdna1, ROCJITSU_CODE_ARCH_RDNA1)
    SAMPLE(rdna2, ROCJITSU_CODE_ARCH_RDNA2)
    SAMPLE(rdna3, ROCJITSU_CODE_ARCH_RDNA3)
    SAMPLE(rdna3_5, ROCJITSU_CODE_ARCH_RDNA3_5)
  case ROCJITSU_CODE_ARCH_RDNA4:
    words = rdna4::build_vsample(
        rdna4::kImageSampleVsample,
        {.dim = 1, .dmask = 1, .vdata = 8, .rsrc = 0, .samp = 8, .vaddr0 = 0, .vaddr1 = 1});
    break;
  default:
    FAIL();
  }
#undef SAMPLE
  struct QuadCase {
    uint32_t base;
    std::array<std::array<uint32_t, 2>, 4> offsets;
    uint32_t expected_level;
  };
  std::vector<QuadCase> cases;
  for (uint32_t base : {0u, 1u}) {
    for (uint32_t step : {0u, 1u, 2u, 4u, 8u}) {
      cases.push_back({base,
                       {{{0, 0}, {step, 0}, {0, step}, {step, step}}},
                       std::max(base, step <= 1 ? 0u : uint32_t(std::countr_zero(step)))});
    }
  }
  // Independent gfx1100/gfx1201 captures: the same coarse derivatives apply to
  // all four lanes. Moving lane 3 cannot affect them; moving only lane 1 does.
  cases.push_back({0, {{{0, 0}, {0, 0}, {0, 0}, {56, 56}}}, 0});
  cases.push_back({0, {{{0, 0}, {8, 0}, {0, 1}, {1, 1}}}, 3});
  for (const auto &test : cases) {
    wf->set_exec(15);
    SCOPED_TRACE(testing::Message() << test.base << ',' << test.expected_level);
    const uint32_t level = test.expected_level;
    bind_view(level, level);
    for (uint32_t lane = 0; lane < 4; ++lane) {
      const uint32_t edge = (64u >> level) - 1;
      write_coordinates(lane, {std::min(edge, (8 + test.offsets[lane][0]) >> level),
                               std::min(edge, (8 + test.offsets[lane][1]) >> level), 0});
      cu->write_vgpr(vb + 8, lane, 0xa0000000u + level);
    }
    ASSERT_TRUE(decoded_issue(false, 2));
    bind_view(test.base, 3);
    for (uint32_t lane = 0; lane < 4; ++lane) {
      cu->write_vgpr(vb, lane, std::bit_cast<uint32_t>((8.5f + test.offsets[lane][0]) / 64));
      cu->write_vgpr(vb + 1, lane, std::bit_cast<uint32_t>((8.5f + test.offsets[lane][1]) / 64));
      cu->write_vgpr(vb + 8, lane, 0xdeadbeef);
    }
    const auto issue_sample = [&]() {
      auto decoder = Decoder::create(arch());
      auto decoded = decoder->decode(words.data());
      EXPECT_TRUE(decoded.succeeded());
      auto inst = std::move(decoded).value();
      inst->execute(*inst, wf);
      if (!inst->data())
        return false;
      amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
      EXPECT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
      return true;
    };
    ASSERT_TRUE(issue_sample());
    for (uint32_t lane = 0; lane < 4; ++lane)
      EXPECT_EQ(cu->read_vgpr(vb + 8, lane), 0xa0000000u + level);
    EXPECT_TRUE(wf->wait_counters().empty());
    if (test.base == 0 && level == 0) {
      for (uint64_t mask : {1u, 7u}) {
        wf->set_exec(mask);
        wf->clear_instruction_execution_error();
        EXPECT_FALSE(issue_sample());
        EXPECT_EQ(wf->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnimplementedInstruction);
        wf->clear_instruction_execution_error();
        bind_view(0, 0);
        EXPECT_TRUE(issue_sample());
        EXPECT_EQ(wf->instruction_execution_error(), amdgpu::InstructionExecutionError::None);
        bind_view(0, 3);
      }
    }
  }
}

TEST_P(AllImageAccessTest, MipOpcodesUseViewRelativeLevelsForLoadsAndStores) {
  for (bool a16 : {false, true}) {
    for (bool nsa : {false, true}) {
      if (nsa && arch_is_cdna_4_or_lower(arch()))
        continue;
      packed_coordinates = a16;
      scattered = nsa;
      SCOPED_TRACE(testing::Message() << "a16=" << a16 << " nsa=" << nsa);
      const auto write_mip = [&](uint32_t level) {
        const uint32_t selector = a16 ? (nsa ? 10 : 1) : (nsa ? 12 : 2);
        cu->write_vgpr(vb + selector, 0, level);
      };
      Image image;
      image.width = image.height = 64;
      image.swizzle_mode = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 1 : 0;
      auto words =
          arch_is_cdna_4_or_lower(arch()) ? descriptor(image) : rdna_descriptor(image, arch());
      const bool gfx9 = arch_is_cdna_4_or_lower(arch()), gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
      const uint32_t max_bit = gfx9                                 ? 188
                               : gfx12                              ? 44
                               : arch() <= ROCJITSU_CODE_ARCH_RDNA2 ? 164
                                                                    : 48;
      util::insert_bits_into_words<uint32_t>(words.data(), max_bit, gfx12 ? 5 : 4, 1);
      const auto bind_level = [&](uint32_t base) {
        auto view = words;
        util::insert_bits_into_words<uint32_t>(view.data(), gfx12 ? 57 : 108, gfx12 ? 5 : 4, base);
        util::insert_bits_into_words<uint32_t>(view.data(), gfx12 ? 111 : 112, gfx12 ? 5 : 4, 1);
        for (uint32_t i = 0; i < view.size(); ++i)
          cu->write_sgpr(sb + i, view[i]);
      };
      wf->set_exec(1);
      for (uint32_t reg : {0u, 1u, 2u, 10u, 12u})
        cu->write_vgpr(vb + reg, 0, 0);
      // Seed the two physical levels through independently qualified plain opcodes.
      bind_level(0);
      cu->write_vgpr(vb + 8, 0, 0x11112222);
      ASSERT_TRUE(decoded_issue(false, 2));
      bind_level(1);
      cu->write_vgpr(vb + 8, 0, 0x33334444);
      ASSERT_TRUE(decoded_issue(false, 2));
      cu->write_vgpr(vb + 8, 0, 0xdeadbeef);
      cu->write_vgpr(vb + 8, 1, 0xdeadbeef);
      write_mip(0);
      ASSERT_TRUE(decoded_issue(true, 2, true));
      EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x33334444u);
      EXPECT_EQ(cu->read_vgpr(vb + 8, 1), 0xdeadbeefu);
      bind_level(0);
      write_mip(1);
      ASSERT_TRUE(decoded_issue(true, 2, true));
      EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x33334444u);
      cu->write_vgpr(vb + 8, 0, 0x77778888);
      ASSERT_TRUE(decoded_issue(false, 2, true));
      bind_level(1);
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x77778888u);
      bind_level(0);
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x11112222u);
      bind_level(1);
      for (uint32_t relative : {1u, 0xffffffffu}) {
        SCOPED_TRACE(relative);
        write_mip(relative);
        ASSERT_TRUE(decoded_issue(true, 2, true));
        EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0u);
        cu->write_vgpr(vb + 8, 0, 0xaaaaaaaa);
        ASSERT_TRUE(decoded_issue(false, 2, true)); // OOB cannot change either level.
      }
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x77778888u);
      write_mip(0);
      cu->write_vgpr(vb + 8, 0, 0x55556666);
      ASSERT_TRUE(decoded_issue(false, 2, true));
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x55556666u);
      bind_level(0);
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x11112222u);
    }
  }
}

TEST_P(AllImageAccessTest, DecodedLoadAndStoreRoundTripDimensionsAndExec) {
  for (uint32_t dimensions = 1; dimensions <= 3; ++dimensions) {
    SCOPED_TRACE(dimensions);
    for (bool array_dim : {false, true}) {
      if (array_dim && (dimensions != 3 || arch_is_cdna_4_or_lower(arch())))
        continue;
      SCOPED_TRACE(array_dim);
      dimension_override = array_dim ? amdgpu::kImageDim2dArray : dimensions - 1;
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

TEST_P(AllImageAccessTest, Rdna2RbPlusMatchesIndependentBackingOffsets) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA2)
    GTEST_SKIP() << "Navi21 RB+ qualification";
  // Independent Navi21 (revision 0x28), GB_ADDR_CONFIG=0x444 witnesses.
  constexpr uint32_t cases[][4] = {
      {2, 4, 0, 64},  {2, 33, 17, 17428}, {6, 4, 0, 64},  {6, 33, 17, 5140},
      {10, 4, 0, 64}, {10, 33, 17, 9236}, {22, 4, 0, 64}, {22, 33, 17, 4116},
      {26, 4, 0, 64}, {26, 33, 17, 8212}, {27, 4, 0, 64}, {27, 33, 17, 2836},
  };
  wf->set_exec(1);
  for (const auto &c : cases) {
    Image image;
    image.width = 256;
    image.height = 128;
    image.swizzle_mode = c[0];
    bind_image(image);
    write_coordinates(0, {c[1], c[2], 0});
    cu->write_vgpr(vb + 8, 0, 0x12340000u + c[3]);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory.read32(kBase + c[3]), 0x12340000u + c[3]);
    memory.write32(kBase + c[3], 0xabcd0000u + c[3]);
    cu->l1_vector().invalidate_all();
    l2.invalidate_all();
    ASSERT_TRUE(decoded_issue(true, 2));
    EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0xabcd0000u + c[3]);
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
  const std::array<uint32_t, 2> offsets = gfx9 ? std::array<uint32_t, 2>{116, 8308}
                                          : arch() == ROCJITSU_CODE_ARCH_RDNA2
                                              ? std::array<uint32_t, 2>{8308, 4212}
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

TEST_P(AllImageAccessTest, TiledPitchFieldMatchesPhysicalGpuBacking) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Physical tiled pitch captures cover gfx1100 and gfx1201";
  Image image;
  image.width = 300;
  image.height = 70;
  image.swizzle_mode = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 3 : 27;
  const uint32_t offset = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 156316 : 146588;
  wf->set_exec(1);
  // Full backing dumps are identical for these descriptor pitches on both GPUs.
  for (uint32_t pitch : {1u, 320u, 384u, 512u, 1024u}) {
    bind_image(image);
    cu->write_sgpr(sb + 4, pitch - 1);
    write_coordinates(0, {299, 69, 0});
    cu->write_vgpr(vb + 8, 0, 0x12340000u + pitch);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory.read32(kBase + offset), 0x12340000u + pitch);
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
  for (uint32_t type : {amdgpu::kImage2dArray, amdgpu::kImage3d}) {
    image.type = type;
    for (uint32_t dim : {type - amdgpu::kImage1d, uint32_t{amdgpu::kImageDim2dArray}}) {
      SCOPED_TRACE(dim);
      dimension_override = dim;
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
}

TEST_P(AllImageAccessTest, ArrayLayerPositionFollowsInstructionDimension) {
  if (arch_is_cdna_4_or_lower(arch()))
    GTEST_SKIP() << "GFX9 uses DA rather than DIM";
  Image image;
  image.width = 30;
  image.depth = 3;
  image.type = amdgpu::kImage1dArray;
  declare_array = false; // GFX10+ DIM supplies the coordinate inventory.
  bind_image(image);
  wf->set_exec(1);
  for (uint32_t dim : {amdgpu::kImageDim1dArray, amdgpu::kImageDim2dArray, amdgpu::kImageDim3d}) {
    dimension_override = dim;
    for (uint32_t layer : {1u, 2u}) {
      SCOPED_TRACE(testing::Message() << "dim=" << dim << " layer=" << layer);
      // Physical 1DA R32 captures put each layer on a 256-byte boundary.
      const uint32_t offset = layer * 256 + 5 * 4;
      write_coordinates(0, dim == amdgpu::kImageDim1dArray ? std::array<uint32_t, 3>{5, layer, 0}
                                                           : std::array<uint32_t, 3>{5, 0, layer});
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

TEST_P(AllImageAccessTest, ArrayResourcesRejectDimensionsWithoutALayerOperand) {
  if (arch_is_cdna_4_or_lower(arch()))
    GTEST_SKIP() << "GFX9 uses DA rather than DIM";
  Image image;
  image.width = 30;
  image.depth = 3;
  image.type = amdgpu::kImage1dArray;
  bind_image(image);
  wf->set_exec(1);
  write_coordinates(0, {5, 2, 1});
  for (uint32_t dim : {amdgpu::kImageDim1d, amdgpu::kImageDim2d})
    for (bool load : {false, true}) {
      dimension_override = dim;
      wf->clear_instruction_execution_error();
      EXPECT_FALSE(decoded_issue(load, 3));
      EXPECT_EQ(wf->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
    }
}

TEST_P(AllImageAccessTest, StubMimgInstructionsScaleDescriptorSelectors) {
  if (arch() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "GFX12 uses VIMAGE/VSAMPLE rather than MIMG";
  for (uint32_t operation = 0; operation < 3; ++operation) {
    std::array<uint32_t, 2> words{};
#define DECODE_STUB(ARCH, ENUM)                                                                    \
  case ENUM: {                                                                                     \
    auto encoding = fields<ARCH::MimgBuilderFields>(2);                                            \
    encoding.srsrc = 4;                                                                            \
    encoding.ssamp = 3;                                                                            \
    const uint16_t opcode = operation == 0   ? ARCH::kImageSampleMimg                              \
                            : operation == 1 ? ARCH::kImageLoadMipMimg                             \
                                             : ARCH::kImageAtomicAddMimg;                          \
    words = ARCH::build_mimg(opcode, encoding);                                                    \
    break;                                                                                         \
  }
    switch (arch()) {
      DECODE_STUB(cdna1, ROCJITSU_CODE_ARCH_CDNA1)
      DECODE_STUB(cdna2, ROCJITSU_CODE_ARCH_CDNA2)
      DECODE_STUB(rdna1, ROCJITSU_CODE_ARCH_RDNA1)
      DECODE_STUB(rdna2, ROCJITSU_CODE_ARCH_RDNA2)
      DECODE_STUB(rdna3, ROCJITSU_CODE_ARCH_RDNA3)
      DECODE_STUB(rdna3_5, ROCJITSU_CODE_ARCH_RDNA3_5)
    default:
      FAIL();
    }
#undef DECODE_STUB
    auto decoder = Decoder::create(arch());
    auto decoded = decoder->decode(words.data());
    ASSERT_TRUE(decoded.succeeded());
    InstDefUse uses(*decoded.value(), 64);
    EXPECT_TRUE(uses.uses.contains({RegClass::SGPR, 16, 8}));
    if (operation == 0) {
      EXPECT_TRUE(uses.uses.contains({RegClass::SGPR, 12, 4}));
    }
    EXPECT_FALSE(uses.uses.contains({RegClass::SGPR, 4, 1}));
  }
}

TEST_P(AllImageAccessTest, LinearOneDimensionalArraysUseIndependentLayerStride) {
  Image image;
  image.width = 30;
  image.depth = 3;
  image.type = amdgpu::kImage1dArray;
  dimension_override = image.type - amdgpu::kImage1d;
  bind_image(image);
  wf->set_exec(1);
  // Physical 1DA R32 captures put each layer on a 256-byte boundary.
  for (uint32_t layer : {1u, 2u}) {
    const uint32_t offset = layer * 256 + 5 * 4;
    write_coordinates(0, {5, layer, 0});
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

TEST_P(AllImageAccessTest, RdnaSrgbFormatsDecodeColorAndKeepLinearAlpha) {
  if (arch_is_cdna_4_or_lower(arch()))
    GTEST_SKIP() << "GFX9 encodes sRGB in NFMT, covered separately";
  const bool gfx10 = arch() == ROCJITSU_CODE_ARCH_RDNA1 || arch() == ROCJITSU_CODE_ARCH_RDNA2;
  // IEC sRGB transfer after F32 UNORM conversion. GFX11/12 point captures
  // round color channels to BF16; alpha retains its linear F32 UNORM value.
  const std::array<uint32_t, 4> expected =
      gfx10 ? std::array<uint32_t, 4>{0x3e5d0a8b, 0x3d51ffef, 0x3f800000, 0x3f008081}
            : std::array<uint32_t, 4>{0x3e5d0000, 0x3d520000, 0x3f800000, 0x3f008081};
  for (uint32_t format = 0; format < 3; ++format) {
    SCOPED_TRACE(format);
    Image image;
    auto words = rdna_descriptor(image, arch());
    util::insert_bits_into_words<uint32_t>(words.data(),
                                           arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 49 : 52,
                                           gfx10 ? 9 : 8, (gfx10 ? 128u : 64u) + format);
    util::insert_bits_into_words<uint32_t>(words.data(), 96, 12, kIdentity);
    util::insert_bits_into_words<uint32_t>(words.data(), 128, 14, (256u >> format) - 1);
    for (uint32_t i = 0; i < words.size(); ++i)
      cu->write_sgpr(sb + i, words[i]);
    const uint32_t components = 1u << format;
    component_mask = (1u << components) - 1;
    memory.write32(kBase, 0x80ff4080);
    wf->set_exec(1);
    write_coordinates(0, {0, 0, 0});
    ASSERT_TRUE(decoded_issue(true, 2));
    for (uint32_t component = 0; component < components; ++component)
      EXPECT_EQ(cu->read_vgpr(vb + 8 + component, 0), expected[component]);
    wf->clear_instruction_execution_error();
    EXPECT_FALSE(decoded_issue(false, 2));
    EXPECT_EQ(wf->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(memory.read32(kBase), 0x80ff4080u);
    EXPECT_TRUE(wf->wait_counters().empty());
    wf->clear_instruction_execution_error();
  }
}

TEST_P(AllImageAccessTest, UnsupportedImageAtomicsDoNotIssueMemory) {
  bind_image(2);
  wf->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  memory.write32(kBase, 0x12345678);
  cu->write_vgpr(vb + 8, 0, 1);
  std::array<uint32_t, 3> words{};
#define ATOMIC_CASE(ARCH, ENUM)                                                                    \
  case ENUM: {                                                                                     \
    const auto encoded =                                                                           \
        ARCH::build_mimg(ARCH::kImageAtomicAddMimg, fields<ARCH::MimgBuilderFields>(2));           \
    std::copy(encoded.begin(), encoded.end(), words.begin());                                      \
    break;                                                                                         \
  }
  switch (arch()) {
    ATOMIC_CASE(cdna1, ROCJITSU_CODE_ARCH_CDNA1)
    ATOMIC_CASE(cdna2, ROCJITSU_CODE_ARCH_CDNA2)
    ATOMIC_CASE(rdna1, ROCJITSU_CODE_ARCH_RDNA1)
    ATOMIC_CASE(rdna2, ROCJITSU_CODE_ARCH_RDNA2)
    ATOMIC_CASE(rdna3, ROCJITSU_CODE_ARCH_RDNA3)
    ATOMIC_CASE(rdna3_5, ROCJITSU_CODE_ARCH_RDNA3_5)
  case ROCJITSU_CODE_ARCH_RDNA4:
    words = rdna4::build_vimage(rdna4::kImageAtomicAddUintVimage,
                                fields<rdna4::VimageBuilderFields>(2));
    break;
  default:
    FAIL() << "Unexpected image architecture";
  }
#undef ATOMIC_CASE
  auto decoder = Decoder::create(arch());
  auto decoded = decoder->decode(words.data());
  ASSERT_TRUE(decoded.succeeded());
  auto instruction = std::move(decoded).value();
  instruction->execute(*instruction, wf);
  EXPECT_EQ(wf->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
  EXPECT_FALSE(instruction->data());
  EXPECT_EQ(memory.read32(kBase), 0x12345678u);
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 1u);
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST_P(AllImageAccessTest, MisalignedLinearPitchNeedsQualifiedRounding) {
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

TEST_P(AllImageAccessTest, CompressedDescriptorsStopBeforeAccessingOrdinaryBacking) {
  for (uint32_t swizzle : {0u, arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 2u : 22u}) {
    SCOPED_TRACE(swizzle);
    bind_image(2, 64, 64, swizzle);
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
  Image image;
  image.type = amdgpu::kImage3d;
  if (arch_is_cdna_4_or_lower(arch()))
    bind(image); // GFX9 selects the actual coordinate count from T#.
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

TEST_P(AllImageAccessTest, LivenessUsesScatteredCoordinatesAndScaledDescriptor) {
  scattered = !arch_is_cdna_4_or_lower(arch());
  resource_selector = 4;
  for (bool load : {false, true}) {
    for (bool packed : {false, true}) {
      SCOPED_TRACE(testing::Message() << "load=" << load << " packed=" << packed);
      packed_coordinates = packed;
      auto instruction = decoded_instruction(load, 3);
      ASSERT_NE(instruction, nullptr);
      const InstDefUse def_use(*instruction, wf->wf_size());
      EXPECT_TRUE(def_use.uses.contains({RegClass::SGPR, 16, 8}));
      EXPECT_FALSE(def_use.uses.contains({RegClass::SGPR, 4, 1}));
      for (uint16_t reg : {0, 1, 2, 3, 10, 12}) {
        const bool used =
            reg == 0 || reg == (scattered ? 10 : 1) || (!packed && reg == (scattered ? 12 : 2));
        EXPECT_EQ(def_use.uses.contains({RegClass::VGPR, reg, 1}), used) << reg;
      }
    }
  }
}

TEST_P(AllImageAccessTest, A16CoordinateWaitChecksIgnoreTheUnusedHighHalf) {
  bind_image(3);
  wf->set_exec(1);
  scattered = !arch_is_cdna_4_or_lower(arch());
  packed_coordinates = true;
  auto instruction = decoded_instruction(false, 3);
  ASSERT_NE(instruction, nullptr);
  auto &state = wf->ensure_memory_wait_scoreboard();
  for (uint8_t bytes : {0x3, 0xc}) {
    SCOPED_TRACE(bytes);
    state.clear();
    const auto counter = amdgpu::WaitCounterKind::Load;
    const uint16_t last = scattered ? 10 : 1;
    state.add({state.issue(counter), 0x100, 1, {RegClass::VGPR, last, 1}, counter, bytes});
    unsigned reports = 0;
    state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
    state.check_instruction(*instruction, *wf);
    EXPECT_EQ(reports, bytes == 0x3 ? 1u : 0u);
  }
}

TEST_P(AllImageAccessTest, D16LoadWaitOwnershipDistinguishesPreservedAndZeroedHighHalves) {
  wf->set_exec(1);
  packed_data = true;
  auto instruction = decoded_instruction(true, 3);
  ASSERT_NE(instruction, nullptr);
  const InstDefUse def_use(*instruction, wf->wf_size());
  EXPECT_EQ(def_use.uses.contains({RegClass::VGPR, 8, 1}), !cu->sram_ecc());
  auto &state = wf->ensure_memory_wait_scoreboard();
  for (uint8_t bytes : {0x3, 0xc}) {
    SCOPED_TRACE(bytes);
    state.clear();
    unsigned reports = 0;
    cu->track_memory_wait(*instruction, *wf);
    state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
    state.access({RegClass::VGPR, 8, 1}, 1, bytes, false);
    EXPECT_EQ(reports, bytes == 0x3 || cu->sram_ecc() ? 1u : 0u);
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
  // GFX12, and RDNA2 RB+ (Navi21 revision 0x28). Separate physical raw-backing tests qualify
  // GFX11/12 layouts.
  struct Case {
    uint32_t kind, dimensions, bytes, width, height, depth, levels, mip, x, y, z;
    std::array<uint64_t, 6> expected;
  };
  constexpr Case cases[] = {
      {1,
       2,
       1,
       130,
       65,
       3,
       1,
       0,
       129,
       64,
       2,
       {69633ull, 72961ull, 72961ull, 69633ull, 69633ull, 72961ull}},
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
       {344064ull, 345088ull, 706048ull, 628736ull, 671744ull, 327936ull}},
      {3,
       3,
       1,
       33,
       17,
       5,
       1,
       0,
       32,
       16,
       4,
       {18688ull, 264320ull, 264960ull, 263680ull, 18688ull, 265984ull}},
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
       {106498ull, 108290ull, 108290ull, 106498ull, 106498ull, 108290ull}},
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
       {720896ull, 721920ull, 755200ull, 723968ull, 720896ull, 737536ull}},
      {3,
       3,
       2,
       33,
       17,
       5,
       1,
       0,
       32,
       16,
       4,
       {19456ull, 268288ull, 264960ull, 265216ull, 19456ull, 265984ull}},
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
       {180228ull, 181764ull, 181764ull, 180228ull, 180228ull, 181764ull}},
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
       {1114112ull, 1115136ull, 1443328ull, 1442816ull, 1376256ull, 1130752ull}},
      {3,
       3,
       4,
       33,
       17,
       5,
       1,
       0,
       32,
       16,
       4,
       {37888ull, 264192ull, 264960ull, 268288ull, 37888ull, 265984ull}},
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
       {303112ull, 304136ull, 304136ull, 303112ull, 303112ull, 304136ull}},
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
       {1703936ull, 1704960ull, 3118080ull, 2885632ull, 2883584ull, 1704192ull}},
      {3,
       3,
       8,
       33,
       17,
       5,
       1,
       0,
       32,
       16,
       4,
       {77824ull, 280576ull, 265984ull, 274432ull, 77824ull, 265984ull}},
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
       {548880ull, 549904ull, 549904ull, 548880ull, 548880ull, 549904ull}},
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
       {2883584ull, 2884608ull, 4658176ull, 4457472ull, 4456448ull, 2886912ull}},
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
       {118784ull, 264192ull, 264448ull, 278528ull, 118784ull, 264448ull}},
      {1,
       2,
       4,
       37,
       19,
       3,
       6,
       0,
       36,
       18,
       2,
       {37984ull, 46240ull, 46176ull, 46176ull, 46176ull, 46176ull}},
      {1,
       2,
       4,
       37,
       19,
       3,
       6,
       1,
       17,
       8,
       2,
       {43268ull, 40196ull, 40196ull, 39172ull, 39172ull, 40196ull}},
      {1,
       2,
       4,
       37,
       19,
       3,
       6,
       2,
       8,
       3,
       2,
       {47696ull, 36400ull, 36392ull, 35368ull, 35368ull, 36400ull}},
      {1,
       2,
       4,
       37,
       19,
       3,
       6,
       3,
       3,
       1,
       2,
       {46620ull, 33308ull, 33308ull, 34332ull, 34332ull, 33308ull}},
      {1,
       2,
       4,
       37,
       19,
       3,
       6,
       4,
       1,
       0,
       2,
       {46340ull, 33028ull, 33028ull, 34052ull, 34052ull, 33028ull}},
      {1,
       2,
       4,
       37,
       19,
       3,
       6,
       5,
       0,
       0,
       2,
       {46080ull, 32768ull, 32768ull, 33792ull, 33792ull, 32768ull}},
      {0,
       1,
       4,
       257,
       1,
       3,
       6,
       0,
       256,
       0,
       2,
       {16384ull, 9728ull, 9728ull, 9728ull, 9728ull, 9728ull}},
      {0, 1, 4, 257, 1, 3, 6, 2, 63, 0, 2, {18172ull, 7676ull, 7676ull, 7676ull, 7676ull, 7676ull}},
      {0, 1, 4, 257, 1, 3, 6, 5, 7, 0, 2, {21788ull, 6684ull, 6684ull, 6684ull, 6684ull, 6684ull}},
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
       {1728514ull, 3309570ull, 4444162ull, 4429826ull, 1667074ull, 3279874ull}},
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
       {382974ull, 611582ull, 817150ull, 825342ull, 301054ull, 596478ull}},
      {3,
       3,
       2,
       130,
       129,
       17,
       6,
       4,
       7,
       7,
       0,
       {79486ull, 1150ull, 32894ull, 8318ull, 1662ull, 1150ull}},
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
    EXPECT_EQ(addressing->byte_offset(c.x, c.y, c.z),
              c.expected[arch() == ROCJITSU_CODE_ARCH_RDNA2 ? 5 : group]);
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

TEST_P(AllImageAccessTest, TiledBaseXorStopsBeforeImageMemoryAccess) {
  bind_image(2, 64, 64, arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 2 : 22);
  cu->write_sgpr(sb, (kBase + 256) >> 8);
  wf->set_exec(1);
  memory.write32(kBase + 256, 0x12345678);
  cu->write_vgpr(vb + 8, 0, 0xabcdef01);
  for (uint32_t x : {0u, 64u}) {
    write_coordinates(0, {x, 0, 0});
    for (bool load : {false, true}) {
      wf->clear_instruction_execution_error();
      EXPECT_FALSE(decoded_issue(load, 2));
      EXPECT_EQ(wf->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
      EXPECT_EQ(memory.read32(kBase + 256), 0x12345678u);
      EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0xabcdef01u);
    }
  }
}

TEST_P(AllImageAccessTest, BaseAlignmentDependsOnTheStorageBlock) {
  const auto config = amdgpu::default_image_address_config(arch());
  amdgpu::ImageLayout layout{.element_bytes = 4};
  auto addressing = amdgpu::ImageAddressing::prepare(config, layout);
  ASSERT_TRUE(addressing);
  EXPECT_TRUE(addressing->has_supported_base(kBase + 256));
  layout.swizzle_mode = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 2 : 22;
  addressing = amdgpu::ImageAddressing::prepare(config, layout);
  ASSERT_TRUE(addressing);
  EXPECT_TRUE(addressing->has_supported_base(kBase));
  EXPECT_FALSE(addressing->has_supported_base(kBase + 256));
}

TEST(ImageAddressingTest, Navi10ZModeMipTailMatchesIndependentOffsets) {
  // Navi10 revision 1, GB_ADDR_CONFIG=0x44; the small-format tail differs
  // from later RDNA devices. These offsets come from the independent oracle.
  const amdgpu::ImageAddressConfig config{ROCJITSU_CODE_ARCH_RDNA1, 0x44};
  for (const auto &[bytes, mip, offset] :
       {std::array{1u, 0u, 32768u}, std::array{2u, 1u, 18432u}, std::array{2u, 3u, 1024u}}) {
    const amdgpu::ImageLayout layout{
        .width = 67, .height = 45, .element_bytes = bytes, .swizzle_mode = 24, .mip_levels = 7};
    const auto addressing = amdgpu::ImageAddressing::prepare(config, layout, mip);
    ASSERT_TRUE(addressing);
    EXPECT_EQ(addressing->byte_offset(0, 0, 0), offset);
  }
}

TEST(ImageAddressingTest, UnqualifiedCdnaTilingRegistersAreRejected) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2}) {
    const amdgpu::ImageAddressConfig config{arch, 0x82004};
    const amdgpu::ImageLayout layout{.element_bytes = 4, .swizzle_mode = 22};
    EXPECT_FALSE(amdgpu::ImageAddressing::prepare(config, layout));
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

enum class SampleTestOperation { Zero, Lod, Gradients, Gather, Unsupported };

class ImageSamplerControlTest : public AllImageAccessTest {
protected:
  void bind_sample_image(Image image, bool floating = true, bool mip_view = false) {
    image.num_format = floating ? 7 : kNfmtUint;
    auto words =
        arch_is_cdna_4_or_lower(arch()) ? descriptor(image) : rdna_descriptor(image, arch());
    if (!arch_is_cdna_4_or_lower(arch()) && floating)
      util::insert_bits_into_words<uint32_t>(words.data(),
                                             arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 49 : 52, 8, 22u);
    if (mip_view)
      util::insert_bits_into_words<uint32_t>(
          words.data(), arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 111 : 112,
          arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 5 : 4, image.mip_levels - 1);
    for (uint32_t word = 0; word < words.size(); ++word)
      cu->write_sgpr(sb + word, words[word]);
  }

  void bind_sampler(uint32_t filter, int32_t bias = 0, uint32_t minimum = 0,
                    bool normalized = true) {
    const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
    const std::array<uint32_t, 4> sampler{2u | (2u << 3) | (2u << 6) | (normalized ? 0u : 1u << 15),
                                          minimum | (768u << (gfx12 ? 13 : 12)),
                                          filter | (static_cast<uint32_t>(bias) & 0x3fffu), 0};
    for (uint32_t word = 0; word < sampler.size(); ++word)
      cu->write_sgpr(sb + 8 + word, sampler[word]);
  }

  std::unique_ptr<Instruction> sample_instruction(SampleTestOperation operation, bool d16 = false,
                                                  uint32_t sampler = 8) {
    std::array<uint32_t, 3> words{};
#define SAMPLE_CONTROL(ARCH, ENUM)                                                                 \
  case ENUM: {                                                                                     \
    auto encoding = fields<ARCH::MimgBuilderFields>(2);                                            \
    encoding.unorm = 0;                                                                            \
    encoding.ssamp = sampler / 4;                                                                  \
    encoding.d16 = d16;                                                                            \
    const uint16_t opcode = operation == SampleTestOperation::Zero  ? ARCH::kImageSampleLzMimg     \
                            : operation == SampleTestOperation::Lod ? ARCH::kImageSampleLMimg      \
                            : operation == SampleTestOperation::Gradients                          \
                                ? ARCH::kImageSampleDMimg                                          \
                            : operation == SampleTestOperation::Gather ? ARCH::kImageGather4LzMimg \
                                                                       : ARCH::kImageSampleOMimg;  \
    const auto encoded = ARCH::build_mimg(opcode, encoding);                                       \
    std::copy(encoded.begin(), encoded.end(), words.begin());                                      \
    break;                                                                                         \
  }
    switch (arch()) {
      SAMPLE_CONTROL(cdna1, ROCJITSU_CODE_ARCH_CDNA1)
      SAMPLE_CONTROL(rdna1, ROCJITSU_CODE_ARCH_RDNA1)
      SAMPLE_CONTROL(rdna2, ROCJITSU_CODE_ARCH_RDNA2)
      SAMPLE_CONTROL(rdna3, ROCJITSU_CODE_ARCH_RDNA3)
      SAMPLE_CONTROL(rdna3_5, ROCJITSU_CODE_ARCH_RDNA3_5)
    case ROCJITSU_CODE_ARCH_CDNA2: {
      // CDNA2's public ISA input exposes only base IMAGE_SAMPLE. A single-level
      // sample without mip filtering has the same controls as the zero form.
      if (operation != SampleTestOperation::Zero)
        return {};
      auto encoding = fields<cdna2::MimgBuilderFields>(2);
      encoding.unorm = 0;
      encoding.ssamp = sampler / 4;
      const auto encoded = cdna2::build_mimg(cdna2::kImageSampleMimg, encoding);
      std::copy(encoded.begin(), encoded.end(), words.begin());
      break;
    }
    case ROCJITSU_CODE_ARCH_RDNA4: {
      auto encoding = rdna4::VsampleBuilderFields{};
      encoding.dim = dimension_override.value_or(1);
      encoding.dmask = 1;
      encoding.vdata = 8;
      encoding.rsrc = 0;
      encoding.samp = sampler;
      encoding.vaddr0 = 0;
      encoding.vaddr1 = 1;
      encoding.vaddr2 = 2;
      encoding.vaddr3 = 3;
      encoding.d16 = d16;
      const uint16_t opcode =
          operation == SampleTestOperation::Zero        ? rdna4::kImageSampleLzVsample
          : operation == SampleTestOperation::Lod       ? rdna4::kImageSampleLVsample
          : operation == SampleTestOperation::Gradients ? rdna4::kImageSampleDVsample
          : operation == SampleTestOperation::Gather    ? rdna4::kImageGather4LzVsample
                                                        : rdna4::kImageSampleOVsample;
      words = rdna4::build_vsample(opcode, encoding);
      break;
    }
    default:
      return {};
    }
#undef SAMPLE_CONTROL
    auto decoder = Decoder::create(arch());
    auto decoded = decoder->decode(words.data());
    EXPECT_TRUE(decoded.succeeded());
    return decoded.succeeded() ? std::move(decoded).value() : nullptr;
  }

  bool sample(SampleTestOperation operation, bool d16 = false, uint32_t sampler = 8) {
    auto instruction = sample_instruction(operation, d16, sampler);
    if (!instruction)
      return false;
    instruction->execute(*instruction, wf);
    if (!instruction->data())
      return false;
    amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
    EXPECT_EQ(pipeline.issue(instruction.release(), *wf), amdgpu::VmAccessOutcome::Complete);
    // Later cases seed backing storage directly, as in the base image fixture.
    cu->l1_vector().flush_all();
    l2.flush_all();
    return true;
  }
};

TEST_P(ImageSamplerControlTest, UnnormalizedWrapUsesClampAndMirrorOnce) {
  Image image;
  image.width = 16;
  bind_sample_image(image);
  for (uint32_t x = 0; x < 16; ++x)
    memory.write32(kBase + x * 4, std::bit_cast<uint32_t>(float(x)));
  wf->set_exec(1);
  for (uint32_t wrap : {0u, 1u})
    for (uint32_t filter : {0u, 1u}) {
      bind_sampler((filter << 20) | (filter << 22), 0, 0, false);
      cu->write_sgpr(sb + 8, (1u << 15) | wrap | (wrap << 3));
      for (const auto &[x, mirror] :
           {std::pair{-INFINITY, 15.0f}, std::pair{-16.5f, 15.0f}, std::pair{-1.5f, 1.0f},
            std::pair{-0.5f, 0.0f}, std::pair{0.0f, 0.0f}, std::pair{16.5f, 15.0f}}) {
        cu->write_vgpr(vb, 0, std::bit_cast<uint32_t>(x));
        cu->write_vgpr(vb + 1, 0, std::bit_cast<uint32_t>(0.5f));
        ASSERT_TRUE(sample(SampleTestOperation::Zero));
        EXPECT_EQ(cu->read_vgpr(vb + 8, 0), std::bit_cast<uint32_t>(wrap == 1 ? mirror
                                                                    : x < 1   ? 0.0f
                                                                              : 15.0f));
      }
    }
}

TEST_P(ImageSamplerControlTest, GatherPreservesFloatPayloadsThatPointSamplingCanonicalizes) {
  if (arch() == ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP() << "The public CDNA2 ISA omits gather forms";
  Image image;
  image.width = image.height = 2;
  bind_sample_image(image);
  bind_sampler(0, 0, 0, false);
  wf->set_exec(1);
  // Captured on gfx1100/gfx1201: gather preserves NaN payload and denormal bits.
  for (uint32_t payload : {0x7fc12345u, 0x7f800001u, 0xffc00001u, 1u, 0x80000001u, 0x007fffffu,
                           0x80000000u, 0x7f800000u}) {
    memory.write32(kBase, payload);
    cu->l1_vector().invalidate_all();
    l2.invalidate_all();
    cu->write_vgpr(vb, 0, std::bit_cast<uint32_t>(0.5f));
    cu->write_vgpr(vb + 1, 0, std::bit_cast<uint32_t>(0.5f));
    ASSERT_TRUE(sample(SampleTestOperation::Zero));
    const bool nan = (payload & 0x7fffffffu) > 0x7f800000u;
    const bool denormal = (payload & 0x7f800000u) == 0;
    const uint32_t point = nan ? 0xffc00000u : denormal ? payload & 0x80000000u : payload;
    EXPECT_EQ(cu->read_vgpr(vb + 8, 0), point);
    cu->write_vgpr(vb, 0, std::bit_cast<uint32_t>(1.0f));
    cu->write_vgpr(vb + 1, 0, std::bit_cast<uint32_t>(1.0f));
    ASSERT_TRUE(sample(SampleTestOperation::Gather));
    EXPECT_EQ(cu->read_vgpr(vb + 11, 0), payload);
  }
}

TEST_P(ImageSamplerControlTest, UnqualifiedLodControlsAreRejected) {
  const bool legacy = arch_is_cdna_4_or_lower(arch()) || arch() == ROCJITSU_CODE_ARCH_RDNA1 ||
                      arch() == ROCJITSU_CODE_ARCH_RDNA2;
  for (bool resource_control : {false, true}) {
    if (!resource_control && !legacy)
      continue;
    bind_sample_image(Image{});
    bind_sampler(1u << 26);
    if (resource_control) {
      const uint32_t bit = legacy ? 40 : arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 186 : 187;
      cu->write_sgpr(sb + bit / 32, cu->read_sgpr(sb + bit / 32) | (1u << (bit % 32)));
    } else {
      cu->write_sgpr(sb + 10, cu->read_sgpr(sb + 10) | (1u << 28));
    }
    wf->set_exec(1);
    wf->clear_instruction_execution_error();
    EXPECT_FALSE(sample(SampleTestOperation::Zero));
    EXPECT_EQ(wf->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
  }
}

TEST_P(ImageSamplerControlTest, OneDimensionalViewUsesZeroForTheMissingCoordinate) {
  if (arch_is_cdna_4_or_lower(arch()))
    GTEST_SKIP() << "GFX9 has no DIM operand";
  for (bool layered : {false, true}) {
    for (uint32_t layer = 0; layer < (layered ? 2u : 1u); ++layer) {
      Image image;
      image.width = 4;
      image.height = 2;
      image.type = layered ? amdgpu::kImage2dArray : amdgpu::kImage2d;
      image.depth = layered ? 2 : 1;
      dimension_override = layered ? amdgpu::kImageDim1dArray : amdgpu::kImageDim1d;
      bind_sample_image(image);
      // GFX12 array rows use 128-byte alignment, while allocation keeps
      // 256-byte rows for the layer stride (two rows = 512 bytes).
      const uint32_t row_bytes = layered && arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
      for (uint32_t i = 0; i < 4; ++i) {
        memory.write32(kBase + layer * 512 + i * 4, std::bit_cast<uint32_t>(1000.0f * layer + i));
        memory.write32(kBase + layer * 512 + row_bytes + i * 4,
                       std::bit_cast<uint32_t>(1000.0f * layer + 100.0f + i));
      }
      cu->l1_vector().invalidate_all();
      l2.invalidate_all();
      wf->set_exec(15);
      for (bool normalized : {false, true}) {
        for (bool linear : {false, true}) {
          for (uint32_t wrap : {0u, 1u, 2u, 6u}) {
            bind_sampler(linear ? (1u << 20) | (1u << 22) : 0, 0, 0, normalized);
            cu->write_sgpr(sb + 8, (cu->read_sgpr(sb + 8) & ~(7u << 3)) | (wrap << 3));
            for (uint32_t lane = 0; lane < 4; ++lane) {
              const float x = lane + (linear ? 0.25f : 0.5f);
              cu->write_vgpr(vb, lane, std::bit_cast<uint32_t>(normalized ? x / 4 : x));
              // The instruction has no Y input; a following live VGPR is irrelevant.
              cu->write_vgpr(vb + 1, lane, std::bit_cast<uint32_t>(layered ? float(layer) : 1.5f));
              cu->write_vgpr(vb + 2, lane, std::bit_cast<uint32_t>(1.5f));
            }
            ASSERT_TRUE(sample(SampleTestOperation::Zero));
            for (uint32_t lane = 0; lane < 4; ++lane) {
              float expected = 1000.0f * layer + (linear && lane ? lane - 0.25f : float(lane));
              if (linear && wrap == 6)
                expected *= 0.5f; // Half the footprint reads transparent black.
              else if (linear && normalized && wrap == 0)
                expected += 50.0f; // Half reads the last row, whose values are 100 higher.
              EXPECT_EQ(cu->read_vgpr(vb + 8, lane), std::bit_cast<uint32_t>(expected));
            }
          }
        }
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, UnqualifiedCornerAndCombinedPerformanceControlsAreRejected) {
  const bool gfx9 = arch_is_cdna_4_or_lower(arch());
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  for (uint32_t control = 0; control < 6; ++control) {
    if (control == 0 && gfx9)
      continue; // GFX9 has no CORNER_SAMPLES field.
    Image image;
    if (control >= 4) {
      image.type = amdgpu::kImage3d;
      image.depth = 2;
      dimension_override = amdgpu::kImageDim3d;
    }
    bind_sample_image(image);
    bind_sampler(0);
    if (control == 0)
      cu->write_sgpr(sb + 5, cu->read_sgpr(sb + 5) | (1u << 23));
    if (control == 1 || control == 3 || control == 5) {
      const uint32_t bit = gfx9 ? 92 : 180;
      cu->write_sgpr(sb + bit / 32, cu->read_sgpr(sb + bit / 32) | (1u << (bit % 32)));
    }
    if (control == 2 || control == 3) {
      if (gfx12)
        cu->write_sgpr(sb + 11, cu->read_sgpr(sb + 11) | 2u);
      else
        cu->write_sgpr(sb + 9, cu->read_sgpr(sb + 9) | (8u << 24));
    }
    if (control >= 4)
      cu->write_sgpr(sb + 9, cu->read_sgpr(sb + 9) | (15u << 28));
    wf->set_exec(1);
    wf->clear_instruction_execution_error();
    // The performance fields independently leave the qualified result unchanged.
    if (control == 1 || control == 2 || control == 4) {
      ASSERT_TRUE(sample(SampleTestOperation::Zero));
    } else {
      EXPECT_FALSE(sample(SampleTestOperation::Zero));
      EXPECT_EQ(wf->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
    }
  }
}

TEST_P(ImageSamplerControlTest, TiledSamplerBaseXorIsRejected) {
  Image image;
  image.width = image.height = 64;
  image.swizzle_mode = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 2 : 22;
  bind_sample_image(image);
  bind_sampler(0);
  cu->write_sgpr(sb, (kBase + 256) >> 8);
  wf->set_exec(1);
  EXPECT_FALSE(sample(SampleTestOperation::Zero));
  EXPECT_EQ(wf->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
}

TEST(ImageFootprintTest, CapturedFractionalLodCoversLogSegmentsAndVolumeProjections) {
  struct Case {
    std::array<double, 3> dx, dy;
    std::array<uint32_t, 3> extents;
    uint32_t dimensions;
    double expected;
  };
  // Independent matching gfx1100/gfx1201 captures. Constant-valued mip levels
  // and linear mip filtering expose the sampler's fractional LOD directly.
  const Case cases[] = {
      {{0x1.019d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1p-7},
      {{0x1.219d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.7p-3},
      {{0x1.419d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.5p-2},
      {{0x1.619d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.dcp-2},
      {{0x1.819d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.2ep-1},
      {{0x1.a19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.6ap-1},
      {{0x1.c19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.ap-1},
      {{0x1.e19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.d2p-1},
      {{0x1.019d000000000p-8, 0x1.019d000000000p-6, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.02p-1},
      {{0x1.219d000000000p-8, 0x1.219d000000000p-6, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.58p-1},
      {{0x1.419d000000000p-8, 0x1.419d000000000p-6, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.a6p-1},
      {{0x1.619d000000000p-8, 0x1.619d000000000p-6, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.ecp-1},
      {{0x1.819d000000000p-8, 0x1.819d000000000p-6, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.16p+0},
      {{0x1.a19d000000000p-8, 0x1.a19d000000000p-6, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.33p+0},
      {{0x1.c19d000000000p-8, 0x1.c19d000000000p-6, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.4fp+0},
      {{0x1.e19d000000000p-8, 0x1.e19d000000000p-6, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.68p+0},
      {{0x1.019d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x1.019d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.02p-1},
      {{0x1.219d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x1.219d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.58p-1},
      {{0x1.419d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x1.419d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.a6p-1},
      {{0x1.619d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x1.619d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.ecp-1},
      {{0x1.819d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x1.819d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.16p+0},
      {{0x1.a19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x1.a19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.33p+0},
      {{0x1.c19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x1.c19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.4fp+0},
      {{0x1.e19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x1.e19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.68p+0},
      {{-0x1.5bf1f8p-6, 0x1.208a3ep-6, -0x1.e9ea3ep-5},
       {0x0p+0, 0x0p+0, 0x0p+0},
       {37, 19, 11},
       3,
       0x1.cp-4},
      {{-0x1.e8bfccp-6, 0x1.954decp-6, -0x1.5815d2p-4},
       {0x0p+0, 0x0p+0, 0x0p+0},
       {37, 19, 11},
       3,
       0x1.34p-1},
      {{0x0p+0, 0x0p+0, 0x0p+0},
       {-0x1.bd65f2p-8, 0x1.a5ffcp-5, -0x1.1798d2p-5},
       {37, 19, 11},
       3,
       0x1.cp-4},
      {{0x0p+0, 0x0p+0, 0x0p+0},
       {-0x1.38d1c8p-7, 0x1.2862a6p-4, -0x1.88be08p-5},
       {37, 19, 11},
       3,
       0x1.34p-1},
      {{-0x1.26d1p-7, 0x1.fc4bc6p-7, -0x1.1ef782p-5},
       {-0x1.11b9dcp-7, 0x1.376612p-5, -0x1.12e478p-4},
       {37, 19, 11},
       3,
       0x1.cp-4},
      {{-0x1.9e1ee4p-7, 0x1.64febp-6, -0x1.931838p-5},
       {-0x1.807edcp-7, 0x1.b569d4p-5, -0x1.822252p-4},
       {37, 19, 11},
       3,
       0x1.32p-1},
      {{0x1.adaf82p-9, 0x1.306cf8p-5, -0x1.2bf50ep-4},
       {0x0p+0, 0x0p+0, 0x0p+0},
       {37, 19, 11},
       3,
       0x1.cp-4},
      {{0x1.2dc8aep-8, 0x1.ab9e5ap-5, -0x1.a55788p-4},
       {0x0p+0, 0x0p+0, 0x0p+0},
       {37, 19, 11},
       3,
       0x1.32p-1},
      {{0x0p+0, 0x0p+0, 0x0p+0},
       {-0x1.7487acp-8, -0x1.a5a39ap-5, 0x1.2b5c58p-5},
       {37, 19, 11},
       3,
       0x1.dp-4},
      {{0x0p+0, 0x0p+0, 0x0p+0},
       {-0x1.05a434p-7, -0x1.2821eep-4, 0x1.a48106p-5},
       {37, 19, 11},
       3,
       0x1.34p-1},
      {{0x1.13c47cp-11, -0x1.4ec31p-5, 0x1.1662e8p-4},
       {-0x1.4a8bd8p-6, 0x1.3f30aep-7, 0x1.429284p-5},
       {37, 19, 11},
       3,
       0x1.bp-4},
      {{0x1.835cfcp-11, -0x1.d63b2cp-5, 0x1.870ab8p-4},
       {-0x1.d04f3ap-6, 0x1.c05b9ap-7, 0x1.c51bd2p-5},
       {37, 19, 11},
       3,
       0x1.32p-1},
      {{0x1.f69754p-7, 0x1.28e23ep-7, 0x1.4f38eep-4},
       {0x0p+0, 0x0p+0, 0x0p+0},
       {37, 19, 11},
       3,
       0x1.cp-4},
      {{0x1.60fcf4p-6, 0x1.a1065p-7, 0x1.d6e0bep-4},
       {0x0p+0, 0x0p+0, 0x0p+0},
       {37, 19, 11},
       3,
       0x1.32p-1},
      {{0x0p+0, 0x0p+0, 0x0p+0},
       {0x1.69a6c8p-11, -0x1.d06f0ep-5, 0x1.9a7648p-9},
       {37, 19, 11},
       3,
       0x1.cp-4},
      {{0x0p+0, 0x0p+0, 0x0p+0},
       {0x1.fc008ep-11, -0x1.46305ap-4, 0x1.204852p-8},
       {37, 19, 11},
       3,
       0x1.34p-1},
      {{-0x1.13823ap-9, 0x1.4af7a6p-5, -0x1.f45ec6p-5},
       {0x1.daa314p-7, -0x1.dfee1ep-6, -0x1.17eee8p-4},
       {37, 19, 11},
       3,
       0x1.bp-4},
      {{-0x1.82ffe8p-9, 0x1.d0e6a6p-5, -0x1.5f6da4p-4},
       {0x1.4d5adcp-6, -0x1.51128ep-5, -0x1.8936f4p-4},
       {37, 19, 11},
       3,
       0x1.3p-1},
      {{-0x1.e410d6p-7, -0x1.90339ap-5, -0x1.feeacep-9},
       {0x0p+0, 0x0p+0, 0x0p+0},
       {37, 19, 11},
       3,
       0x1.dp-4},
      {{-0x1.53fa24p-6, -0x1.191384p-4, -0x1.66d5fap-8},
       {0x0p+0, 0x0p+0, 0x0p+0},
       {37, 19, 11},
       3,
       0x1.34p-1},
      {{0x0p+0, 0x0p+0, 0x0p+0},
       {0x1.a609ap-7, 0x1.7b5c7p-12, -0x1.680f6ep-4},
       {37, 19, 11},
       3,
       0x1.bp-4},
      {{0x0p+0, 0x0p+0, 0x0p+0},
       {0x1.286994p-6, 0x1.0a706ep-11, -0x1.f9c458p-4},
       {37, 19, 11},
       3,
       0x1.32p-1},
      {{-0x1.5443bep-6, 0x1.b3ee3p-11, 0x1.c54ep-8},
       {-0x1.25e2ep-6, 0x1.067ad4p-5, -0x1.b94152p-5},
       {37, 19, 11},
       3,
       0x1.cp-4},
      {{-0x1.ddf5e8p-6, 0x1.322b7cp-10, 0x1.3e5f5ap-7},
       {-0x1.9cd066p-6, 0x1.70b2c6p-5, -0x1.35e8dep-4},
       {37, 19, 11},
       3,
       0x1.32p-1},
  };
  for (const auto &test : cases) {
    const double lod = amdgpu::image_gradient_lod(test.dx, test.dy, test.extents, test.dimensions);
    EXPECT_EQ(std::clamp(lod, 0.0, 5.0), test.expected);
  }
}

TEST_P(ImageSamplerControlTest, DecodedGradientsSelectCapturedBoundaryMips) {
  if (arch() == ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP() << "The public CDNA2 ISA omits explicit gradient forms";
  Image image;
  image.width = 37;
  image.height = 19;
  image.mip_levels = 6;
  bind_sample_image(image, true, true);
  bind_sampler(1u << 26);
  cu->write_sgpr(sb + 9, 1280u << (arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 13 : 12));
  // Independent linear mip-atlas offsets for a 37x19 R32 allocation.
  // Storage rounds mip extents upward; the accessible view still rounds down.
  const std::array<uint32_t, 6> offsets =
      arch_is_cdna_4_or_lower(arch()) ? std::array<uint32_t, 6>{0, 4864, 7424, 8704, 9472, 9984}
                                      : std::array<uint32_t, 6>{5376, 2816, 1536, 768, 256, 0};
  const uint32_t pitch = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
  for (uint32_t mip = 0; mip < 6; ++mip)
    for (uint32_t y = 0; y < std::max(1u, 19u >> mip); ++y)
      for (uint32_t x = 0; x < std::max(1u, 37u >> mip); ++x)
        memory.write32(kBase + offsets[mip] + y * pitch + x * 4,
                       std::bit_cast<uint32_t>(float(mip)));
  struct Case {
    std::array<float, 4> gradients;
    float expected;
  };
  // Off-axis NPOT boundary captures; each gradient prefix is decoded and executed.
  const Case cases[] = {
      {{-0x1.85f86ap-6f, 0x1.4363e2p-6f, -0x1.467c04p-6f, 0x1.4f5242p-5f}, 0x0p+0f},
      {{-0x1.8a4898p-6f, 0x1.46f78cp-6f, -0x1.4a187p-6f, 0x1.5307b4p-5f}, 0x1p+0f},
      {{-0x1.85f86ap-5f, 0x1.4363e2p-5f, -0x1.467c04p-5f, 0x1.4f5242p-4f}, 0x1p+0f},
      {{-0x1.8a4898p-5f, 0x1.46f78cp-5f, -0x1.4a187p-5f, 0x1.5307b4p-4f}, 0x1p+1f},
      {{-0x1.85f86ap-4f, 0x1.4363e2p-4f, -0x1.467c04p-4f, 0x1.4f5242p-3f}, 0x1p+1f},
      {{-0x1.8a4898p-4f, 0x1.46f78cp-4f, -0x1.4a187p-4f, 0x1.5307b4p-3f}, 0x1.8p+1f},
      {{-0x1.85f86ap-3f, 0x1.4363e2p-3f, -0x1.467c04p-3f, 0x1.4f5242p-2f}, 0x1.8p+1f},
      {{-0x1.8a4898p-3f, 0x1.46f78cp-3f, -0x1.4a187p-3f, 0x1.5307b4p-2f}, 0x1p+2f},
      {{-0x1.ff83bep-8f, -0x1.8fff16p-9f, 0x1.6f4b42p-6f, -0x1.e72046p-5f}, 0x0p+0f},
      {{-0x1.02960ap-7f, -0x1.946ba8p-9f, 0x1.735b3ap-6f, -0x1.ec838ep-5f}, 0x1p+0f},
      {{-0x1.ff83bep-7f, -0x1.8fff16p-8f, 0x1.6f4b42p-5f, -0x1.e72046p-4f}, 0x1p+0f},
      {{-0x1.02960ap-6f, -0x1.946ba8p-8f, 0x1.735b3ap-5f, -0x1.ec838ep-4f}, 0x1p+1f},
      {{-0x1.ff83bep-6f, -0x1.8fff16p-7f, 0x1.6f4b42p-4f, -0x1.e72046p-3f}, 0x1p+1f},
      {{-0x1.02960ap-5f, -0x1.946ba8p-7f, 0x1.735b3ap-4f, -0x1.ec838ep-3f}, 0x1.8p+1f},
      {{-0x1.ff83bep-5f, -0x1.8fff16p-6f, 0x1.6f4b42p-3f, -0x1.e72046p-2f}, 0x1.8p+1f},
      {{-0x1.02960ap-4f, -0x1.946ba8p-6f, 0x1.735b3ap-3f, -0x1.ec838ep-2f}, 0x1p+2f},
      {{-0x1.982114p-7f, 0x1.46b13ap-7f, 0x1.edf4c2p-11f, -0x1.2bd04ap-4f}, 0x0p+0f},
      {{-0x1.9ca4acp-7f, 0x1.4a4e3ep-7f, 0x1.f36b5ep-11f, -0x1.2f2132p-4f}, 0x1p+0f},
      {{-0x1.982114p-6f, 0x1.46b13ap-6f, 0x1.edf4c2p-10f, -0x1.2bd04ap-3f}, 0x1p+0f},
      {{-0x1.9ca4acp-6f, 0x1.4a4e3ep-6f, 0x1.f36b5ep-10f, -0x1.2f2132p-3f}, 0x1p+1f},
      {{-0x1.982114p-5f, 0x1.46b13ap-5f, 0x1.edf4c2p-9f, -0x1.2bd04ap-2f}, 0x1p+1f},
      {{-0x1.9ca4acp-5f, 0x1.4a4e3ep-5f, 0x1.f36b5ep-9f, -0x1.2f2132p-2f}, 0x1.8p+1f},
      {{-0x1.982114p-4f, 0x1.46b13ap-4f, 0x1.edf4c2p-8f, -0x1.2bd04ap-1f}, 0x1.8p+1f},
      {{-0x1.9ca4acp-4f, 0x1.4a4e3ep-4f, 0x1.f36b5ep-8f, -0x1.2f2132p-1f}, 0x1p+2f},
  };
  wf->set_exec(1);
  for (const auto &test : cases) {
    for (uint32_t word = 0; word < 4; ++word)
      cu->write_vgpr(vb + word, 0, std::bit_cast<uint32_t>(test.gradients[word]));
    cu->write_vgpr(vb + 4, 0, std::bit_cast<uint32_t>(0.5f));
    cu->write_vgpr(vb + 5, 0, std::bit_cast<uint32_t>(0.5f));
    ASSERT_TRUE(sample(SampleTestOperation::Gradients));
    EXPECT_EQ(cu->read_vgpr(vb + 8, 0), std::bit_cast<uint32_t>(test.expected));
  }
}

TEST_P(ImageSamplerControlTest, ExplicitLodRoundsBeforeBiasAndSaturatesInfiniteInputs) {
  if (arch() == ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP() << "The public CDNA2 ISA and installed SDK omit explicit-LOD forms";
  Image image;
  image.width = image.height = 16;
  image.mip_levels = 4;
  bind_sample_image(image, true, true);
  const bool gfx9 = arch_is_cdna_4_or_lower(arch());
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  // Independent linear mip-atlas offsets; each mip contains its level number.
  const std::array<uint32_t, 4> offsets = gfx9 ? std::array<uint32_t, 4>{0, 4096, 6144, 7168}
                                               : std::array<uint32_t, 4>{3584, 1536, 512, 0};
  for (uint32_t level = 0; level < 4; ++level)
    for (uint32_t y = 0; y < (16u >> level); ++y)
      for (uint32_t x = 0; x < (16u >> level); ++x)
        memory.write32(kBase + offsets[level] + y * (gfx12 ? 128 : 256) + x * 4,
                       std::bit_cast<uint32_t>(static_cast<float>(level)));
  struct Case {
    bool zero;
    float lod;
    int32_t bias;
    uint32_t minimum;
    float expected;
  };
  // Exact captures on gfx1100/gfx1201. Odd bias distinguishes conversion before
  // bias from a superficially similar nearest-even conversion after bias.
  const Case cases[] = {{true, 0, 128, 0, 0.5f},
                        {false, 1.0f / 512, 0, 0, 0},
                        {false, 5.0f / 512, 0, 0, 2.0f / 256},
                        {false, 3.0f / 512, 0, 0, 2.0f / 256},
                        {false, 1.0f / 512, 1, 0, 1.0f / 256},
                        {false, -1.0f / 512, 1, 0, 1.0f / 256},
                        {false, -INFINITY, 256, 0, 0},
                        {false, INFINITY, -256, 0, 3},
                        {false, -INFINITY, 0x1fff, 256, 1},
                        {false, INFINITY, -0x2000, 256, 3},
                        {false, NAN, 256, 0, 1},
                        {false, -1.0e30f, 256, 0, 0},
                        {false, 1.0e30f, -256, 0, 3},
                        {false, 0, 0, 384, 1.5f}};
  wf->set_exec(1);
  for (const Case &test : cases)
    for (uint32_t mip_filter : {1u, 2u}) {
      SCOPED_TRACE(testing::Message() << test.zero << ',' << test.lod << ',' << test.bias << ','
                                      << test.minimum << ',' << mip_filter);
      bind_sampler(mip_filter << 26, test.bias, test.minimum);
      cu->write_vgpr(vb, 0, std::bit_cast<uint32_t>(0.5f));
      cu->write_vgpr(vb + 1, 0, std::bit_cast<uint32_t>(0.5f));
      cu->write_vgpr(vb + 2, 0, std::bit_cast<uint32_t>(test.lod));
      ASSERT_TRUE(sample(test.zero ? SampleTestOperation::Zero : SampleTestOperation::Lod));
      const float expected = mip_filter == 2 ? test.expected : std::floor(test.expected + 0.5f);
      EXPECT_EQ(cu->read_vgpr(vb + 8, 0), std::bit_cast<uint32_t>(expected));
    }
  // Independent GFX11/12 captures distinguish integer-LOD mip blending.
  for (uint32_t level = 0; level < 4; ++level)
    for (uint32_t y = 0; y < (16u >> level); ++y)
      for (uint32_t x = 0; x < (16u >> level); ++x)
        memory.write32(kBase + offsets[level] + y * (gfx12 ? 128 : 256) + x * 4, 0x80000000u);
  for (uint32_t spatial_filter : {0u, 1u})
    for (uint32_t mip_filter : {1u, 2u})
      for (float lod : {0.0f, 0.25f, 0.5f, 1.0f, 1.5f, 2.75f, 3.0f}) {
        bind_sampler((spatial_filter << 20) | (spatial_filter << 22) | (mip_filter << 26));
        cu->write_vgpr(vb, 0, 0u);
        cu->write_vgpr(vb + 1, 0, 0u);
        cu->write_vgpr(vb + 2, 0, std::bit_cast<uint32_t>(lod));
        ASSERT_TRUE(sample(SampleTestOperation::Lod));
        const bool blends = mip_filter == 2 && lod != std::floor(lod);
        EXPECT_EQ(cu->read_vgpr(vb + 8, 0), blends ? 0u : 0x80000000u);
      }
  struct BlendCase {
    uint64_t exec;
    uint32_t fractional_lane;
    float lod;
    uint64_t gfx12_positive;
    uint64_t other_positive;
  };
  // Captured quad contexts: a scalar witness cannot distinguish these rules.
  const BlendCase blends[] = {{15, 8, 0.25f, 0, 0},
                              {15, 3, 0.25f, 15, 8},
                              {7, 3, 0.25f, 0, 0},
                              {255, 4, 0.25f, 240, 16},
                              {15, 3, 1.0f + 1.0f / 512, 0, 0},
                              {15, 3, 3.5f, 0, 0},
                              {2, 8, 0.25f, 0, 0}};
  for (uint32_t spatial_filter : {0u, 1u})
    for (const BlendCase &test : blends) {
      SCOPED_TRACE(testing::Message()
                   << test.exec << ',' << test.fractional_lane << ',' << test.lod);
      wf->set_exec(test.exec);
      bind_sampler((spatial_filter << 20) | (spatial_filter << 22) | (2u << 26));
      for (uint32_t lane = 0; lane < 8; ++lane) {
        cu->write_vgpr(vb, lane, 0u);
        cu->write_vgpr(vb + 1, lane, 0u);
        cu->write_vgpr(vb + 2, lane,
                       std::bit_cast<uint32_t>(lane == test.fractional_lane ? test.lod : 1.0f));
        cu->write_vgpr(vb + 8, lane, 0xA5A5A5A5u);
      }
      ASSERT_TRUE(sample(SampleTestOperation::Lod));
      for (uint32_t lane = 0; lane < 8; ++lane) {
        const uint64_t bit = uint64_t{1} << lane;
        const uint64_t positive = gfx12 ? test.gfx12_positive : test.other_positive;
        const uint32_t expected = !(test.exec & bit) ? 0xA5A5A5A5u
                                  : positive & bit   ? 0u
                                                     : 0x80000000u;
        EXPECT_EQ(cu->read_vgpr(vb + 8, lane), expected);
      }
    }
}

TEST_P(ImageSamplerControlTest, NullGatherClearsEveryPackedAndUnpackedDestination) {
  if (arch() == ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP() << "The public CDNA2 ISA and installed SDK omit gather forms";
  wf->set_exec(5);
  for (bool d16 : {false, true})
    for (uint32_t invalid : {0u, 1u, 2u}) {
      SCOPED_TRACE(testing::Message() << d16 << ',' << invalid);
      Image image;
      image.width = 2;
      bind_sample_image(image, false);
      memory.write32(kBase, 0x11);
      memory.write32(kBase + 4, 0x33);
      if (invalid == 0)
        for (uint32_t word = 0; word < 8; ++word)
          cu->write_sgpr(sb + word, 0);
      if (invalid == 1) {
        cu->write_sgpr(sb + 1, 0);
      }
      bind_sampler(0);
      for (uint32_t lane : {0u, 1u, 2u}) {
        cu->write_vgpr(vb, lane, std::bit_cast<uint32_t>(0.5f));
        cu->write_vgpr(vb + 1, lane, std::bit_cast<uint32_t>(0.5f));
        for (uint32_t reg = 0; reg < 4; ++reg)
          cu->write_vgpr(vb + 8 + reg, lane, 0xdeadbeef);
      }
      ASSERT_TRUE(sample(SampleTestOperation::Gather, d16, invalid == 2 ? 104 : 8));
      for (uint32_t lane : {0u, 1u, 2u})
        for (uint32_t reg = 0; reg < 4; ++reg)
          EXPECT_EQ(cu->read_vgpr(vb + 8 + reg, lane),
                    lane != 1 && reg < (d16 ? 2u : 4u) ? 0u : 0xdeadbeefu);
    }
}

TEST_P(ImageSamplerControlTest, IntegerFilteringIsExplicitlyUnsupportedButPointSamplingWorks) {
  Image image;
  image.width = 2;
  bind_sample_image(image, false);
  memory.write32(kBase, 1);
  memory.write32(kBase + 4, 3);
  wf->set_exec(1);
  cu->write_vgpr(vb, 0, std::bit_cast<uint32_t>(0.5f));
  cu->write_vgpr(vb + 1, 0, std::bit_cast<uint32_t>(0.5f));
  for (uint32_t filter : {0u, (1u << 20) | (1u << 22), 2u << 26}) {
    bind_sampler(filter);
    wf->clear_instruction_execution_error();
    if (filter == 0) {
      ASSERT_TRUE(sample(SampleTestOperation::Zero));
      EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 3u);
    } else {
      EXPECT_FALSE(sample(SampleTestOperation::Zero));
      EXPECT_EQ(wf->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
    }
  }
}

TEST_P(ImageSamplerControlTest, ClampedLinearTapsPreserveNegativeZero) {
  Image image;
  image.width = 2;
  bind_sample_image(image);
  memory.write32(kBase, 0x80000000u);
  memory.write32(kBase + 4, 0x80000000u);
  wf->set_exec(1);
  bind_sampler((1u << 20) | (1u << 22), 0, 0, false);
  for (float x : {0.0f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f}) {
    cu->write_vgpr(vb, 0, std::bit_cast<uint32_t>(x));
    cu->write_vgpr(vb + 1, 0, std::bit_cast<uint32_t>(0.5f));
    ASSERT_TRUE(sample(SampleTestOperation::Zero));
    EXPECT_EQ(cu->read_vgpr(vb + 8, 0), x > 0.5f && x < 1.5f ? 0u : 0x80000000u);
  }
  struct Case {
    uint32_t width;
    uint32_t wrap;
    float x;
    uint32_t expected;
  };
  // Captured normalized mirror/repeat edge and distinct-texel controls.
  const Case cases[] = {{2, 1, 0, 0x80000000u},
                        {4, 1, 0.05f, 0x80000000u},
                        {4, 1, 1, 0x80000000u},
                        {1, 0, 0, 0x80000000u},
                        {1, 0, -0.3f, 0x80000000u},
                        {1, 1, 1.3f, 0x80000000u},
                        {2, 0, 0, 0},
                        {2, 1, 0.5f, 0},
                        {2, 6, 0, 0}};
  for (const Case &test : cases) {
    SCOPED_TRACE(testing::Message() << test.width << ',' << test.wrap << ',' << test.x);
    image.width = test.width;
    bind_sample_image(image);
    for (uint32_t x = 0; x < image.width; ++x)
      memory.write32(kBase + x * 4, 0x80000000u);
    bind_sampler((1u << 20) | (1u << 22));
    cu->write_sgpr(sb + 8, test.wrap | (2u << 3) | (2u << 6));
    cu->write_vgpr(vb, 0, std::bit_cast<uint32_t>(test.x));
    cu->write_vgpr(vb + 1, 0, std::bit_cast<uint32_t>(0.5f));
    ASSERT_TRUE(sample(SampleTestOperation::Zero));
    EXPECT_EQ(cu->read_vgpr(vb + 8, 0), test.expected);
  }
}

TEST_P(ImageSamplerControlTest, UnsupportedSampleFormsReportAnExecutionError) {
  if (arch() == ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP() << "The public CDNA2 ISA exposes only the base sample form";
  wf->clear_instruction_execution_error();
  EXPECT_FALSE(sample(SampleTestOperation::Unsupported));
  EXPECT_EQ(wf->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
}

TEST_P(ImageSamplerControlTest, DerivativeMetadataUsesTwoCubeComponents) {
  if (arch_is_cdna_4_or_lower(arch()) || arch() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "MIMG DIM encoding on GFX10/GFX11";
  for (uint32_t dim : {1u, 2u, 3u}) {
    dimension_override = dim;
    auto instruction = sample_instruction(SampleTestOperation::Gradients);
    ASSERT_TRUE(instruction);
    amdgpu::RegisterModifiers modifiers{};
    instruction->amdgpu_register_modifiers(modifiers);
    const uint32_t prefix = dim == 2 ? 6 : 4;
    const uint32_t coordinates = dim == 1 ? 2 : 3;
    EXPECT_EQ(modifiers.image_prefix_words, prefix);
    EXPECT_EQ(modifiers.image_coordinate_count, coordinates);
    InstDefUse uses(*instruction, 32);
    for (uint32_t reg = 0; reg < prefix + coordinates; ++reg)
      EXPECT_TRUE(uses.uses.contains({RegClass::VGPR, static_cast<uint16_t>(reg), 1}));
    EXPECT_FALSE(
        uses.uses.contains({RegClass::VGPR, static_cast<uint16_t>(prefix + coordinates), 1}));
  }
}

TEST_P(ImageSamplerControlTest, VimageMipTracksItsFifthIndependentSelector) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "VIMAGE is a GFX12 encoding";
  rdna4::VimageBuilderFields encoding{};
  encoding.dim = 7;
  encoding.dmask = 1;
  encoding.vdata = 8;
  encoding.rsrc = 0;
  encoding.vaddr0 = 1;
  encoding.vaddr1 = 9;
  encoding.vaddr2 = 16;
  encoding.vaddr3 = 23;
  encoding.vaddr4 = 30;
  const auto words = rdna4::build_vimage(rdna4::kImageLoadMipVimage, encoding);
  auto decoder = Decoder::create(arch());
  auto decoded = decoder->decode(words.data());
  ASSERT_TRUE(decoded.succeeded());
  amdgpu::RegisterModifiers modifiers{};
  decoded.value()->amdgpu_register_modifiers(modifiers);
  EXPECT_EQ(modifiers.image_coordinate_count, 5);
  for (uint32_t i = 0; i < 5; ++i)
    EXPECT_EQ(modifiers.image_coordinates[i], (std::array<uint32_t, 5>{1, 9, 16, 23, 30})[i]);
}

INSTANTIATE_TEST_SUITE_P(Architectures, ImageSamplerControlTest,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));

} // namespace
