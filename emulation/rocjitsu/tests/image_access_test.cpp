// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
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
  uint32_t base_level = 0;
  uint32_t last_level = 0;
  uint32_t rdna_format = kR32Uint;
  uint32_t min_lod = 0;
  uint32_t perf_mod = 0;
  bool corner_samples = false;
};

std::array<uint32_t, 8> descriptor(const Image &image) {
  std::array<uint32_t, 8> r{};
  const auto set = [&](uint32_t bit, uint32_t width, uint64_t value) {
    for (uint32_t i = 0; i < width; ++i)
      r[(bit + i) / 32] |= static_cast<uint32_t>((value >> i) & 1u) << ((bit + i) % 32);
  };
  set(0, 40, kBase >> 8);
  set(40, 12, image.min_lod);
  set(92, 3, image.perf_mod);
  set(52, 6, image.data_format);
  set(58, 4, image.num_format);
  set(64, 14, image.width - 1);
  set(78, 14, image.height - 1);
  set(96, 12, image.selectors);
  set(108, 4, image.base_level);
  set(112, 4, image.last_level);
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

class ImageAccessFixture : public testing::Test {
protected:
  amdgpu::GpuMemory memory_{"image_memory"};
  amdgpu::L2Cache l2_{"image_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  amdgpu::Wavefront *wf_ = nullptr;
  uint32_t sb_ = 0, vb_ = 0;

  virtual rj_code_arch_t arch() const = 0;
  void SetUp() override {
    l2_.set_backing_memory(&memory_);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch();
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 32;
    cfg.lds_size_kb = 64;
    ASSERT_NO_FATAL_FAILURE(
        rocjitsu::test::initialize_compute_unit(cu_, "image_cu", cfg, &memory_, &l2_));
    wf_ = cu_->dispatch_wf(0, 0, 106, 32);
    ASSERT_NE(wf_, nullptr);
    sb_ = wf_->sgpr_alloc().base;
    vb_ = wf_->vgpr_alloc().base;
    wf_->set_exec(5); // Lane 1 must not read, store or receive a load result.
  }
  void TearDown() override {
    if (wf_)
      wf_->halt();
  }

  void write_coordinates(uint32_t lane, std::array<uint32_t, 3> coords) {
    for (uint32_t i = 0; i < coords.size(); ++i)
      cu_->write_vgpr(vb_ + i, lane, coords[i]);
  }
  void fill_destination(uint32_t value) {
    for (uint32_t lane = 0; lane < 3; ++lane)
      for (uint32_t reg = 0; reg < 4; ++reg)
        cu_->write_vgpr(vb_ + 8 + reg, lane, value);
  }
};

class Cdna2ImageAccessTest : public ImageAccessFixture {
protected:
  /// @brief MIMG flags used by the access fixture.
  struct Encoding {
    bool a16 = false;
    bool d16 = false;
    bool da = true;
    bool unorm = true;
  };

  rj_code_arch_t arch() const override { return ROCJITSU_CODE_ARCH_CDNA2; }

  void bind(const Image &image) {
    const auto words = descriptor(image);
    for (uint32_t i = 0; i < words.size(); ++i)
      cu_->write_sgpr(sb_ + i, words[i]);
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
    inst->execute_impl(*wf_);
    if (!inst->data())
      return false;
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    EXPECT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    cu_->l1_vector().flush_all();
    l2_.flush_all();
    return true;
  }
};

TEST_F(Cdna2ImageAccessTest, StoreThenLoadRoundTripsActiveLanesOfAnR32Image) {
  Image image;
  image.width = 64;
  bind(image);
  for (uint32_t lane = 0; lane < 3; ++lane) {
    write_coordinates(lane, {lane * 16, 0, 0});
    cu_->write_vgpr(vb_ + 8, lane, 0x55000000u + lane);
  }
  ASSERT_TRUE(issue(false));
  EXPECT_EQ(memory_.read32(kBase), 0x55000000u);
  EXPECT_EQ(memory_.read32(kBase + 16 * 4), 0u);
  EXPECT_EQ(memory_.read32(kBase + 32 * 4), 0x55000002u);

  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true));
  for (uint32_t lane : {0u, 2u}) {
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, lane), 0x55000000u + lane);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 9, lane), 0u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 10, lane), 0u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 11, lane), 1u);
  }
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 1), 0xdeadbeef);
}

TEST_F(Cdna2ImageAccessTest, DecodedMipLoadAndStoreAccessLevelZero) {
  Image image;
  image.width = 64;
  image.height = 2;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {1, 1, 0});
  constexpr uint64_t kAddress = kBase + (64 + 1) * sizeof(uint32_t);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_NE(decoder, nullptr);

  for (const bool load : {false, true}) {
    SCOPED_TRACE(testing::Message() << "load=" << load);
    memory_.write32(kAddress, 0x12345678u);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeefu);
    const auto words =
        cdna2::build_mimg(load ? cdna2::kImageLoadMipMimg : cdna2::kImageStoreMipMimg,
                          {.dmask = 1, .unorm = 1, .da = 1, .vaddr = 0, .vdata = 8, .srsrc = 0});
    auto decoded = decoder->decode(words.data());
    ASSERT_TRUE(decoded.succeeded());
    std::unique_ptr<Instruction> inst = std::move(decoded).value();
    ASSERT_NE(inst, nullptr);

    ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    cu_->l1_vector().flush_all();
    l2_.flush_all();
    EXPECT_EQ(memory_.read32(kAddress), load ? 0x12345678u : 0xdeadbeefu);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), load ? 0x12345678u : 0xdeadbeefu);
  }
}

TEST_F(Cdna2ImageAccessTest, ZeroSelectorsDoNotUseBufferStoreA8Routing) {
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  for (uint32_t format : {kDfmt32, kDfmt8888}) {
    SCOPED_TRACE(format);
    Image image;
    image.data_format = format;
    image.selectors = 0;
    bind(image);
    for (uint32_t reg = 0; reg < 4; ++reg)
      cu_->write_vgpr(vb_ + 8 + reg, 0, format == kDfmt32 ? 0x12345678u : 0x11u * (reg + 1));
    memory_.write32(kBase, 0xdeadbeef);
    ASSERT_TRUE(issue(false));
    EXPECT_EQ(memory_.read32(kBase), 0u);
  }
}

TEST_F(Cdna2ImageAccessTest, ArrayLayersFollowRowsOfPitchElements) {
  Image image;
  image.type = amdgpu::kImage2dArray;
  image.width = 3;
  image.height = 2;
  image.depth = 2;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {2, 1, 1});
  cu_->write_vgpr(vb_ + 8, 0, 0x12345678u);
  ASSERT_TRUE(issue(false));
  EXPECT_EQ(memory_.read32(kBase + ((1 * 2 + 1) * 64 + 2) * 4), 0x12345678u);
}

TEST_F(Cdna2ImageAccessTest, OneDimensionalArraysTakeTheLayerFromTheSecondCoordinate) {
  Image image;
  image.type = amdgpu::kImage1dArray;
  image.width = 4;
  image.depth = 3;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {1, 2, 0});
  cu_->write_vgpr(vb_ + 8, 0, 0xabcdu);
  ASSERT_TRUE(issue(false));
  EXPECT_EQ(memory_.read32(kBase + (2 * 64 + 1) * 4), 0xabcdu);
}

TEST_F(Cdna2ImageAccessTest, ArrayWithoutDaUsesLayerZeroAndIgnoresTheNextRegister) {
  wf_->set_exec(1);
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
      memory_.write32(address, 0x12345678);
      memory_.write32(layer_one, 0x87654321);
      ASSERT_TRUE(issue(true, 1, {.da = false}));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x12345678u);
      cu_->write_vgpr(vb_ + 8, 0, 0x10203040);
      ASSERT_TRUE(issue(false, 1, {.da = false}));
      EXPECT_EQ(memory_.read32(address), 0x10203040u);
      EXPECT_EQ(memory_.read32(layer_one), 0x87654321u);
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
  wf_->set_exec(1);
  write_coordinates(0, {2 | (1u << 16), 1 | (0xffffu << 16), 0xffffffffu});
  cu_->write_vgpr(vb_ + 8, 0, 0x12345678);
  ASSERT_TRUE(issue(false, 1, {.a16 = true}));
  const uint64_t address = kBase + ((1 * 2 + 1) * image.pitch + 2) * 4;
  EXPECT_EQ(memory_.read32(address), 0x12345678u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true, 1, {.a16 = true}));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x12345678u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0xdeadbeefu);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 1), 0xdeadbeefu);
}

TEST_F(Cdna2ImageAccessTest, D16RoundTripsThreeComponentsAndPreservesTheUnusedHighHalf) {
  Image image;
  image.data_format = kDfmt8888;
  image.selectors = kIdentity;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  cu_->write_vgpr(vb_ + 8, 0, 0x00220011);
  cu_->write_vgpr(vb_ + 9, 0, 0xdead0033);
  ASSERT_TRUE(issue(false, 7, {.d16 = true}));
  EXPECT_EQ(memory_.read32(kBase), 0x00332211u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true, 7, {.d16 = true}));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x00220011u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), cu_->sram_ecc() ? 0x0033u : 0xdead0033u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 10, 0), 0xdeadbeefu);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 1), 0xdeadbeefu);
}

TEST_F(Cdna2ImageAccessTest, D16StoresAndLoadsSelectComponentsFromPackedRegisters) {
  Image image;
  image.data_format = kDfmt8888;
  image.selectors = kIdentity;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  // Preserve sparse component selection while consuming consecutive halves.
  // This follows the explicit RDNA2 Table 41 example; gfx90a is not hardware-tested.
  for (const auto &[mask, expected] :
       {std::pair{0b0011u, 0x00002211u}, std::pair{0b1001u, 0x22000011u},
        std::pair{0b1100u, 0x22110000u}}) {
    SCOPED_TRACE(mask);
    memory_.write32(kBase, 0xffffffff);
    cu_->write_vgpr(vb_ + 8, 0, 0x00220011);
    ASSERT_TRUE(issue(false, mask, {.d16 = true}));
    EXPECT_EQ(memory_.read32(kBase), expected);
  }
  memory_.write32(kBase, 0x44332211);
  ASSERT_TRUE(issue(true, 0b1001, {.d16 = true}));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x00440011u);
}

TEST_F(Cdna2ImageAccessTest, XyzOneStoresRetainCompactedSelectorBehavior) {
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = kNfmtUint;
  image.selectors = kXyzOne;
  bind(image);
  wf_->set_exec(1);
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
    memory_.write32(kBase, 0xdeadbeef);
    for (uint32_t component = 0; component < test.packed.size(); ++component)
      cu_->write_vgpr(vb_ + 8 + component, 0, test.packed[component]);
    ASSERT_TRUE(issue(false, test.dmask));
    EXPECT_EQ(memory_.read32(kBase), test.expected);
  }
}

TEST_F(Cdna2ImageAccessTest, ArgbStoreRetainsCompactedSelectorBehavior) {
  constexpr uint32_t kYzwx = 5 | (6 << 3) | (7 << 6) | (4 << 9);
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = kNfmtUint;
  image.selectors = kYzwx;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  for (uint32_t component = 0; component < 4; ++component)
    cu_->write_vgpr(vb_ + 8 + component, 0, 0x11 * (component + 1));

  ASSERT_TRUE(issue(false, 15));
  EXPECT_EQ(memory_.read32(kBase), 0x33221144u);
}

TEST_F(Cdna2ImageAccessTest, LoadWithoutUnormStillUsesIntegerCoordinates) {
  Image image;
  image.width = 4;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {2, 0, 0});
  memory_.write32(kBase + 8, 0x12345678);
  ASSERT_TRUE(issue(true, 1, {.unorm = false}));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x12345678u);
}

TEST_F(Cdna2ImageAccessTest, StoreWithoutRequiredUnormIsRejected) {
  bind(Image{});
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  cu_->write_vgpr(vb_ + 8, 0, 0x12345678);
  EXPECT_FALSE(issue(false, 1, {.unorm = false}));
  EXPECT_EQ(wf_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnsupportedOperandValue);
  EXPECT_EQ(memory_.read32(kBase), 0u);
}

TEST_F(Cdna2ImageAccessTest, DmaskPacksTheEnabledComponentsAfterTheChannelSelects) {
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = kNfmtUint;
  image.width = 4;
  image.selectors = kIdentity;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  memory_.write32(kBase, 0x44332211u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true, 0b1010));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x22u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0x44u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 10, 0), 0xdeadbeef);
}

TEST_F(Cdna2ImageAccessTest, UnormStoresConvertAndLoadsReturnFloats) {
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = kNfmtUnorm;
  image.width = 4;
  image.selectors = kIdentity;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {1, 0, 0});
  for (uint32_t i = 0; i < 4; ++i)
    cu_->write_vgpr(vb_ + 8 + i, 0, std::bit_cast<uint32_t>(i == 3 ? 1.0f : 0.0f));
  ASSERT_TRUE(issue(false));
  EXPECT_EQ(memory_.read32(kBase + 4), 0xff000000u);
  fill_destination(0);
  ASSERT_TRUE(issue(true));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 11, 0), std::bit_cast<uint32_t>(1.0f));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0u);
}

TEST_F(Cdna2ImageAccessTest, OutOfBoundsLanesLoadZeroAndDropTheirStores) {
  Image image;
  image.width = 2;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {2, 0, 0});
  memory_.write32(kBase + 2 * 4, 0x77777777u);
  cu_->write_vgpr(vb_ + 8, 0, 0x11111111u);
  ASSERT_TRUE(issue(false));
  EXPECT_EQ(memory_.read32(kBase + 2 * 4), 0x77777777u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0u);
}

TEST_F(Cdna2ImageAccessTest, NullDescriptorLoadsZero) {
  for (uint32_t i = 0; i < 8; ++i)
    cu_->write_sgpr(sb_ + i, 0);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  memory_.write32(kBase, 0x99999999u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true));
  for (uint32_t reg = 0; reg < 4; ++reg)
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + reg, 0), 0u);
}

TEST_F(Cdna2ImageAccessTest, CompressedDescriptorsStopBeforeImageMemoryAccess) {
  bind(Image{});
  cu_->write_sgpr(sb_ + 6, cu_->read_sgpr(sb_ + 6) | (1u << 21)); // COMPRESSION_ENABLE.
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  for (bool load : {false, true}) {
    SCOPED_TRACE(load);
    memory_.write32(kBase, 0x12345678u);
    fill_destination(0xdeadbeefu);
    wf_->clear_instruction_execution_error();
    EXPECT_FALSE(issue(load, 1));
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(memory_.read32(kBase), 0x12345678u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xdeadbeefu);
  }
}

TEST_F(Cdna2ImageAccessTest, UnsupportedSwizzleModeStopsBeforeMemoryAccess) {
  Image image;
  image.width = 64;
  image.swizzle_mode = 31; // Reserved swizzle mode.
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  EXPECT_FALSE(issue(true));
  EXPECT_EQ(wf_->instruction_execution_error(),
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
  wf_->set_exec(1);
  write_coordinates(0, {2, 1, 1});
  const uint64_t target = kBase + ((image.height + 1) * image.pitch + 2) * 4;
  const uint64_t other_slice = kBase + (image.pitch + 2) * 4;
  memory_.write32(target, 0xffffffff);
  memory_.write32(other_slice, 0xabcdef01);
  cu_->write_vgpr(vb_ + 8, 0, 0x11);
  cu_->write_vgpr(vb_ + 9, 0, 0x22);
  ASSERT_TRUE(issue(false, 0b1001, {.da = false}));
  EXPECT_EQ(memory_.read32(target), 0x22000011u);
  EXPECT_EQ(memory_.read32(other_slice), 0xabcdef01u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true, 0b1001, {.da = false}));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x11u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0x22u);
}

TEST_F(Cdna2ImageAccessTest, NullFormatIgnoresConstantOneSelector) {
  Image image;
  image.data_format = 0;
  image.num_format = kNfmtUnorm;
  bind(image);
  wf_->set_exec(1);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true));
  for (uint32_t reg = 0; reg < 4; ++reg)
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + reg, 0), 0u);
}

TEST_F(Cdna2ImageAccessTest, DecodedLoadTracksOnlyItsDestinationRegisters) {
  wf_->set_exec(1);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
  amdgpu::MemoryWaitScoreboard &state = wf_->ensure_memory_wait_scoreboard();
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
      cu_->track_memory_wait(*decoded.value(), *wf_);
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
  wf_->set_exec(1);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
  amdgpu::MemoryWaitScoreboard &state = wf_->ensure_memory_wait_scoreboard();
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
      state.check_instruction(*decoded.value(), *wf_);
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
  EXPECT_EQ(wf_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
}

TEST_F(Cdna2ImageAccessTest, SrgbLoadDecodesColorAndPreservesAlpha) {
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = 9; // GFX9 TYPE_SRGB.
  image.selectors = kIdentity;
  bind(image);
  memory_.write32(kBase, 0x80402000);
  ASSERT_TRUE(issue(true));
  for (uint32_t lane : {0u, 2u}) {
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, lane), 0u);
    EXPECT_NEAR(std::bit_cast<float>(cu_->read_vgpr(vb_ + 9, lane)), 0.014443844f, 1e-7);
    EXPECT_NEAR(std::bit_cast<float>(cu_->read_vgpr(vb_ + 10, lane)), 0.051269458f, 1e-7);
    EXPECT_FLOAT_EQ(std::bit_cast<float>(cu_->read_vgpr(vb_ + 11, lane)), 128.0f / 255);
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
    ASSERT_NO_FATAL_FAILURE(
        rocjitsu::test::initialize_compute_unit(cu_, "rdna_image_cu", cfg, &memory_, &l2_));
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
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  ASSERT_NO_FATAL_FAILURE(
      rocjitsu::test::initialize_compute_unit(cu, "unsupported_image_cu", cfg, &memory, &l2));
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
  set(gfx12 ? 49 : 52, 8, image.rdna_format);
  set(62, 14, image.width - 1);
  set(78, 14, image.height - 1);
  set(96, 12, kXOne);
  set(124, 4, image.type);
  set(gfx12 ? 57 : 108, gfx12 ? 5 : 4, image.base_level);
  set(gfx12 ? 111 : 112, gfx12 ? 5 : 4, image.last_level);
  const bool max_level_in_word5 =
      arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
  set(gfx12 ? 44 : max_level_in_word5 ? 164 : 48, gfx12 ? 5 : 4, image.mip_levels - 1);
  set(max_level_in_word5 ? 40 : gfx12 ? 186 : 187, gfx12 ? 13 : 12, image.min_lod);
  set(180, 3, image.perf_mod);
  set(183, 1, image.corner_samples);
  set(116, 5, image.swizzle_mode);
  if (image.type == amdgpu::kImage3d || image.type == amdgpu::kImage1dArray ||
      image.type == amdgpu::kImage2dArray)
    set(128, 13, image.depth - 1);
  else if (arch != ROCJITSU_CODE_ARCH_RDNA1)
    set(128, gfx12 ? 16 : 14, image.pitch - 1);
  return words;
}

template <typename Builder, typename Fields>
std::array<uint32_t, 3> image_instruction_words(Builder builder, uint16_t opcode,
                                                const Fields &fields) {
  std::array<uint32_t, 3> words{};
  std::ranges::copy(builder(opcode, fields), words.begin());
  return words;
}

template <typename Builder, typename Fields>
std::array<uint32_t, 3> stub_image_instruction_words(Builder builder, uint16_t opcode,
                                                     Fields fields) {
  fields.srsrc = 4;
  fields.ssamp = 3;
  return image_instruction_words(builder, opcode, fields);
}

class AllImageAccessTest : public ImageAccessFixture,
                           public testing::WithParamInterface<rj_code_arch_t> {
protected:
  bool scattered_ = false;
  bool packed_coordinates_ = false;
  bool packed_data_ = false;
  uint32_t resource_selector_ = 0;
  uint32_t component_mask_ = 1;
  bool unnormalized_ = true;
  bool declare_array_ = true;
  std::optional<uint32_t> dimension_override_;
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
    const std::array<uint32_t, 8> words =
        arch_is_cdna_4_or_lower(arch()) ? descriptor(image) : rdna_descriptor(image, arch());
    for (uint32_t index = 0; index < words.size(); ++index)
      cu_->write_sgpr(sb_ + index, words[index]);
  }

  template <typename Fields> Fields fields(uint32_t dimensions) {
    Fields encoding{};
    encoding.dmask = component_mask_;
    encoding.a16 = packed_coordinates_;
    encoding.d16 = packed_data_;
    if constexpr (requires { encoding.srsrc; })
      encoding.srsrc = resource_selector_;
    else
      encoding.rsrc = resource_selector_ * 4;
    encoding.vdata = 8;
    if constexpr (requires { encoding.vaddr; })
      encoding.vaddr = 0;
    else {
      encoding.vaddr0 = 0;
      encoding.vaddr1 = scattered_ ? 10 : 1;
      encoding.vaddr2 = scattered_ ? 12 : 2;
    }
    if constexpr (requires { encoding.nsa; })
      encoding.nsa = scattered_;
    if constexpr (requires { encoding.unorm; })
      encoding.unorm = unnormalized_;
    if constexpr (requires { encoding.da; })
      encoding.da = declare_array_;
    if constexpr (requires { encoding.dim; })
      encoding.dim = dimension_override_.value_or(dimensions - 1);
    return encoding;
  }
  template <typename Fields, typename Builder>
  std::array<uint32_t, 3> sample_mimg_words(Builder builder, uint16_t opcode, bool d16,
                                            uint32_t sampler) {
    Fields encoding = fields<Fields>(2);
    encoding.unorm = 0;
    encoding.ssamp = sampler / 4;
    encoding.d16 = d16;
    return image_instruction_words(builder, opcode, encoding);
  }

  std::unique_ptr<Instruction> decoded_instruction(bool load, uint32_t dimensions,
                                                   bool explicit_mip = false) {
    std::array<uint32_t, 3> words{};
    switch (arch()) {
    case ROCJITSU_CODE_ARCH_CDNA1:
      words = image_instruction_words(
          cdna1::build_mimg,
          load ? (explicit_mip ? cdna1::kImageLoadMipMimg : cdna1::kImageLoadMimg)
               : (explicit_mip ? cdna1::kImageStoreMipMimg : cdna1::kImageStoreMimg),
          fields<cdna1::MimgBuilderFields>(dimensions));
      break;
    case ROCJITSU_CODE_ARCH_CDNA2:
      words = image_instruction_words(
          cdna2::build_mimg,
          load ? (explicit_mip ? cdna2::kImageLoadMipMimg : cdna2::kImageLoadMimg)
               : (explicit_mip ? cdna2::kImageStoreMipMimg : cdna2::kImageStoreMimg),
          fields<cdna2::MimgBuilderFields>(dimensions));
      break;
    case ROCJITSU_CODE_ARCH_RDNA1:
      words = image_instruction_words(
          rdna1::build_mimg,
          load ? (explicit_mip ? rdna1::kImageLoadMipMimg : rdna1::kImageLoadMimg)
               : (explicit_mip ? rdna1::kImageStoreMipMimg : rdna1::kImageStoreMimg),
          fields<rdna1::MimgBuilderFields>(dimensions));
      break;
    case ROCJITSU_CODE_ARCH_RDNA2:
      words = image_instruction_words(
          rdna2::build_mimg,
          load ? (explicit_mip ? rdna2::kImageLoadMipMimg : rdna2::kImageLoadMimg)
               : (explicit_mip ? rdna2::kImageStoreMipMimg : rdna2::kImageStoreMimg),
          fields<rdna2::MimgBuilderFields>(dimensions));
      break;
    case ROCJITSU_CODE_ARCH_RDNA3:
      words = image_instruction_words(
          rdna3::build_mimg,
          load ? (explicit_mip ? rdna3::kImageLoadMipMimg : rdna3::kImageLoadMimg)
               : (explicit_mip ? rdna3::kImageStoreMipMimg : rdna3::kImageStoreMimg),
          fields<rdna3::MimgBuilderFields>(dimensions));
      break;
    case ROCJITSU_CODE_ARCH_RDNA3_5:
      words = image_instruction_words(
          rdna3_5::build_mimg,
          load ? (explicit_mip ? rdna3_5::kImageLoadMipMimg : rdna3_5::kImageLoadMimg)
               : (explicit_mip ? rdna3_5::kImageStoreMipMimg : rdna3_5::kImageStoreMimg),
          fields<rdna3_5::MimgBuilderFields>(dimensions));
      break;
    case ROCJITSU_CODE_ARCH_RDNA4:
      words = image_instruction_words(
          rdna4::build_vimage,
          load ? (explicit_mip ? rdna4::kImageLoadMipVimage : rdna4::kImageLoadVimage)
               : (explicit_mip ? rdna4::kImageStoreMipVimage : rdna4::kImageStoreVimage),
          fields<rdna4::VimageBuilderFields>(dimensions));
      break;
    default:
      return {};
    }
    if (scattered_ && arch() != ROCJITSU_CODE_ARCH_RDNA4)
      words[2] = 10 | (12 << 8);
    std::unique_ptr<Decoder> decoder = Decoder::create(arch());
    DecodeResult decoded = decoder->decode(words.data());
    EXPECT_TRUE(decoded.succeeded());
    if (decoded.failed())
      return {};
    return std::move(decoded).value();
  }
  bool decoded_issue(bool load, uint32_t dimensions, bool mip = false) {
    std::unique_ptr<Instruction> inst = decoded_instruction(load, dimensions, mip);
    if (!inst)
      return false;
    EXPECT_TRUE(inst->is_memory_op());
    inst->execute(*inst, wf_);
    if (!inst->data())
      return false;
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    EXPECT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    cu_->l1_vector().flush_all();
    l2_.flush_all();
    return true;
  }
};

TEST_P(AllImageAccessTest, ImplicitSampleUsesQuadDerivativesToSelectTheMip) {
  Image image;
  image.width = image.height = 64;
  image.swizzle_mode = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 1 : 0;
  image.mip_levels = 4;
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  const auto bind_view = [&](uint32_t base, uint32_t last) {
    image.base_level = base;
    image.last_level = last;
    bind_image(image);
  };
  const std::array<uint32_t, 4> sampler{2 | (2 << 3) | (2 << 6),
                                        gfx12 ? 0x1fffu << 13 : 0xfffu << 12, 1u << 26, 0};
  for (uint32_t i = 0; i < sampler.size(); ++i)
    cu_->write_sgpr(sb_ + 8 + i, sampler[i]);
  std::array<uint32_t, 3> words{};
  switch (arch()) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    words = sample_mimg_words<cdna1::MimgBuilderFields>(cdna1::build_mimg, cdna1::kImageSampleMimg,
                                                        false, 8);
    break;
  case ROCJITSU_CODE_ARCH_CDNA2:
    words = sample_mimg_words<cdna2::MimgBuilderFields>(cdna2::build_mimg, cdna2::kImageSampleMimg,
                                                        false, 8);
    break;
  case ROCJITSU_CODE_ARCH_RDNA1:
    words = sample_mimg_words<rdna1::MimgBuilderFields>(rdna1::build_mimg, rdna1::kImageSampleMimg,
                                                        false, 8);
    break;
  case ROCJITSU_CODE_ARCH_RDNA2:
    words = sample_mimg_words<rdna2::MimgBuilderFields>(rdna2::build_mimg, rdna2::kImageSampleMimg,
                                                        false, 8);
    break;
  case ROCJITSU_CODE_ARCH_RDNA3:
    words = sample_mimg_words<rdna3::MimgBuilderFields>(rdna3::build_mimg, rdna3::kImageSampleMimg,
                                                        false, 8);
    break;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    words = sample_mimg_words<rdna3_5::MimgBuilderFields>(rdna3_5::build_mimg,
                                                          rdna3_5::kImageSampleMimg, false, 8);
    break;
  case ROCJITSU_CODE_ARCH_RDNA4:
    words = rdna4::build_vsample(
        rdna4::kImageSampleVsample,
        {.dim = 1, .dmask = 1, .vdata = 8, .rsrc = 0, .samp = 8, .vaddr0 = 0, .vaddr1 = 1});
    break;
  default:
    FAIL();
  }
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
    wf_->set_exec(15);
    SCOPED_TRACE(testing::Message() << test.base << ',' << test.expected_level);
    const uint32_t level = test.expected_level;
    bind_view(level, level);
    for (uint32_t lane = 0; lane < 4; ++lane) {
      const uint32_t edge = (64u >> level) - 1;
      write_coordinates(lane, {std::min(edge, (8 + test.offsets[lane][0]) >> level),
                               std::min(edge, (8 + test.offsets[lane][1]) >> level), 0});
      cu_->write_vgpr(vb_ + 8, lane, 0xa0000000u + level);
    }
    ASSERT_TRUE(decoded_issue(false, 2));
    bind_view(test.base, 3);
    for (uint32_t lane = 0; lane < 4; ++lane) {
      cu_->write_vgpr(vb_, lane, std::bit_cast<uint32_t>((8.5f + test.offsets[lane][0]) / 64));
      cu_->write_vgpr(vb_ + 1, lane, std::bit_cast<uint32_t>((8.5f + test.offsets[lane][1]) / 64));
      cu_->write_vgpr(vb_ + 8, lane, 0xdeadbeef);
    }
    const auto issue_sample = [&]() {
      auto decoder = Decoder::create(arch());
      auto decoded = decoder->decode(words.data());
      EXPECT_TRUE(decoded.succeeded());
      auto inst = std::move(decoded).value();
      inst->execute(*inst, wf_);
      if (!inst->data())
        return false;
      amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
      EXPECT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
      return true;
    };
    ASSERT_TRUE(issue_sample());
    for (uint32_t lane = 0; lane < 4; ++lane)
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, lane), 0xa0000000u + level);
    EXPECT_TRUE(wf_->wait_counters().empty());
    if (test.base == 0 && level == 0) {
      for (uint64_t mask : {1u, 7u}) {
        wf_->set_exec(mask);
        wf_->clear_instruction_execution_error();
        EXPECT_FALSE(issue_sample());
        EXPECT_EQ(wf_->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnimplementedInstruction);
        wf_->clear_instruction_execution_error();
        bind_view(0, 0);
        EXPECT_TRUE(issue_sample());
        EXPECT_EQ(wf_->instruction_execution_error(), amdgpu::InstructionExecutionError::None);
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
      packed_coordinates_ = a16;
      scattered_ = nsa;
      SCOPED_TRACE(testing::Message() << "a16=" << a16 << " nsa=" << nsa);
      const auto write_mip = [&](uint32_t level) {
        const uint32_t selector = a16 ? (nsa ? 10 : 1) : (nsa ? 12 : 2);
        cu_->write_vgpr(vb_ + selector, 0, level);
      };
      Image image;
      image.width = image.height = 64;
      image.swizzle_mode = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 1 : 0;
      image.mip_levels = 2;
      const auto bind_level = [&](uint32_t base, uint32_t last = 1) {
        image.base_level = base;
        image.last_level = last;
        bind_image(image);
      };
      wf_->set_exec(1);
      for (uint32_t reg : {0u, 1u, 2u, 10u, 12u})
        cu_->write_vgpr(vb_ + reg, 0, 0);
      // Seed the two physical levels through independently qualified plain opcodes.
      bind_level(0);
      cu_->write_vgpr(vb_ + 8, 0, 0x11112222);
      ASSERT_TRUE(decoded_issue(false, 2));
      bind_level(1);
      cu_->write_vgpr(vb_ + 8, 0, 0x33334444);
      ASSERT_TRUE(decoded_issue(false, 2));
      cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeef);
      cu_->write_vgpr(vb_ + 8, 1, 0xdeadbeef);
      write_mip(0);
      ASSERT_TRUE(decoded_issue(true, 2, true));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x33334444u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 1), 0xdeadbeefu);
      bind_level(0);
      write_mip(1);
      ASSERT_TRUE(decoded_issue(true, 2, true));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x33334444u);
      cu_->write_vgpr(vb_ + 8, 0, 0x77778888);
      ASSERT_TRUE(decoded_issue(false, 2, true));
      bind_level(1);
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x77778888u);
      bind_level(0);
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x11112222u);
      bind_level(1);
      for (uint32_t relative : {1u, 0xffffffffu}) {
        SCOPED_TRACE(relative);
        write_mip(relative);
        ASSERT_TRUE(decoded_issue(true, 2, true));
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0u);
        cu_->write_vgpr(vb_ + 8, 0, 0xaaaaaaaa);
        ASSERT_TRUE(decoded_issue(false, 2, true)); // OOB cannot change either level.
      }
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x77778888u);
      write_mip(0);
      cu_->write_vgpr(vb_ + 8, 0, 0x55556666);
      ASSERT_TRUE(decoded_issue(false, 2, true));
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x55556666u);
      bind_level(0);
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x11112222u);
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
      dimension_override_ = array_dim ? amdgpu::kImageDim2dArray : dimensions - 1;
      bind_image(dimensions);
      for (uint32_t lane = 0; lane < 3; ++lane) {
        write_coordinates(lane, {lane, dimensions >= 2 ? 1u : 0u, dimensions == 3 ? 1u : 0u});
        cu_->write_vgpr(vb_ + 8, lane, 0x12345678u + lane);
      }
      ASSERT_TRUE(decoded_issue(false, dimensions));
      fill_destination(0xdeadbeef);
      ASSERT_TRUE(decoded_issue(true, dimensions));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x12345678u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 1), 0xdeadbeef);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 2), 0x1234567au);
    }
  }
}

TEST_P(AllImageAccessTest, ScaledViewsStopBeforeImageMemoryAccess) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  if (!gfx12 && arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP() << "GFX11 and GFX12 expose separate resource scales";
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  // Exercise each depth, height and width scale bit and GFX12 UAV3D.
  for (uint32_t bit_index = gfx12 ? 4 : 8; bit_index <= 19; ++bit_index) {
    SCOPED_TRACE(bit_index);
    bind_image(3, 64, 64);
    cu_->write_sgpr(sb_ + 5, 1u << bit_index);
    for (bool load : {false, true}) {
      SCOPED_TRACE(load);
      memory_.write32(kBase, 0x12345678);
      cu_->write_vgpr(vb_ + 8, 0, 0xabcdef01);
      wf_->clear_instruction_execution_error();
      EXPECT_FALSE(decoded_issue(load, 3));
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
      EXPECT_EQ(memory_.read32(kBase), 0x12345678u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcdef01u);
    }
  }
}

TEST_P(AllImageAccessTest, TiledAddressesMatchIndependentBackingOffsets) {
  // Direct ROCr AddrLib reference calls establish these R32 offsets in a
  // 64x64 image. The latter coordinates cross 256-byte and 4-KiB tile edges.
  const rj_code_arch_t architecture = arch();
  const bool gfx9 = arch_is_cdna_4_or_lower(architecture);
  const bool gfx10 =
      architecture == ROCJITSU_CODE_ARCH_RDNA1 || architecture == ROCJITSU_CODE_ARCH_RDNA2;
  const uint32_t swizzle = gfx9 ? 22 : architecture == ROCJITSU_CODE_ARCH_RDNA4 ? 1 : 2;
  bind_image(2, 64, 64, swizzle);
  wf_->set_exec(1);
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
    cu_->write_vgpr(vb_ + 8, 0, stored);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory_.read32(kBase + offset), stored);
    EXPECT_NE(memory_.read32(kBase + (y * 64 + x) * 4), stored);
    // Seed backing memory independently before loading, so a mutually wrong
    // store and load cannot satisfy the oracle by undoing each other.
    memory_.write32(kBase + offset, loaded);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    ASSERT_TRUE(decoded_issue(true, 2));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), loaded);
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
  wf_->set_exec(1);
  for (const auto &c : cases) {
    Image image;
    image.width = 256;
    image.height = 128;
    image.swizzle_mode = c[0];
    bind_image(image);
    write_coordinates(0, {c[1], c[2], 0});
    cu_->write_vgpr(vb_ + 8, 0, 0x12340000u + c[3]);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory_.read32(kBase + c[3]), 0x12340000u + c[3]);
    memory_.write32(kBase + c[3], 0xabcd0000u + c[3]);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    ASSERT_TRUE(decoded_issue(true, 2));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcd0000u + c[3]);
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
  wf_->set_exec(1);
  const std::array<std::array<uint32_t, 4>, 3> cases{{{1, 1, 12, 0xb5f93d83},
                                                      {21, 11, 2412, 0x2167abef},
                                                      {33, 27, gfx12 ? 5420u : 4396u, 0x85c90d53}}};
  for (const auto &[x, y, offset, value] : cases) {
    SCOPED_TRACE(testing::Message() << x << ',' << y);
    write_coordinates(0, {x, y, 0});
    cu_->write_vgpr(vb_ + 8, 0, value);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory_.read32(kBase + offset), value);
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
  wf_->set_exec(1);
  // Independent reference constants; the GFX11/12 layouts also match physical
  // full-allocation captures after each mip-level import.
  const std::array<uint32_t, 2> offsets = gfx9 ? std::array<uint32_t, 2>{116, 8308}
                                          : arch() == ROCJITSU_CODE_ARCH_RDNA2
                                              ? std::array<uint32_t, 2>{8308, 4212}
                                          : gfx10 ? std::array<uint32_t, 2>{8372, 4276}
                                                  : std::array<uint32_t, 2>{8300, 4204};
  for (uint32_t level = 0; level < offsets.size(); ++level) {
    SCOPED_TRACE(level);
    image.base_level = image.last_level = level;
    bind_image(image);
    write_coordinates(0, {5, 3, 0});
    cu_->write_vgpr(vb_ + 8, 0, 0x12340000u + level);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory_.read32(kBase + offsets[level]), 0x12340000u + level);
    memory_.write32(kBase + offsets[level], 0xabcd0000u + level);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    ASSERT_TRUE(decoded_issue(true, 2));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcd0000u + level);
  }
  write_coordinates(0, {image.width >> 1, 0, 0});
  cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeefu);
  ASSERT_TRUE(decoded_issue(true, 2));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0u);
}

TEST_P(AllImageAccessTest, LinearRowsMatchPhysicalGpuBackingAtExplicitPitches) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Physical linear captures are available for gfx1100 and gfx1201";
  Image image;
  image.width = 30;
  image.height = 5;
  wf_->set_exec(1);
  // Valid 256/512-byte R32 allocations and independently patched descriptors
  // establish row rounding: GFX11 uses 256-byte quanta, GFX12 uses 128.
  // A 140-byte pitch rounds to 256 on both; 128 stays 128 only on GFX12.
  for (uint32_t pitch : {32u, 35u, 64u, 128u}) {
    bind_image(image);
    cu_->write_sgpr(sb_ + 4, pitch - 1);
    for (uint32_t row : {1u, 3u}) {
      write_coordinates(0, {5, row, 0});
      const uint32_t captured_pitch = pitch == 128                                        ? 128
                                      : pitch == 32 && arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 32
                                                                                          : 64;
      const uint32_t offset = (row * captured_pitch + 5) * 4;
      cu_->write_vgpr(vb_ + 8, 0, 0x12340000u + offset);
      ASSERT_TRUE(decoded_issue(false, 2));
      EXPECT_EQ(memory_.read32(kBase + offset), 0x12340000u + offset);
      memory_.write32(kBase + offset, 0xabcd0000u + offset);
      cu_->l1_vector().invalidate_all();
      l2_.invalidate_all();
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcd0000u + offset);
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
  wf_->set_exec(1);
  // Full backing dumps are identical for these descriptor pitches on both GPUs.
  for (uint32_t pitch : {1u, 320u, 384u, 512u, 1024u}) {
    bind_image(image);
    cu_->write_sgpr(sb_ + 4, pitch - 1);
    write_coordinates(0, {299, 69, 0});
    cu_->write_vgpr(vb_ + 8, 0, 0x12340000u + pitch);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory_.read32(kBase + offset), 0x12340000u + pitch);
  }
}

TEST_P(AllImageAccessTest, LinearArrayAndVolumeRowsMatchPhysicalGpuBacking) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Physical linear captures are available for gfx1100 and gfx1201";
  Image image;
  image.width = 30;
  image.height = 5;
  image.depth = 2;
  wf_->set_exec(1);
  // Independent raw captures: GFX12 rows advance by 128 bytes, while slices
  // advance by 5 * 256 bytes. GFX11 rows and slice allocation use 256 bytes.
  const uint32_t row_pitch = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
  for (uint32_t type : {amdgpu::kImage2dArray, amdgpu::kImage3d}) {
    image.type = type;
    for (uint32_t dim : {type - amdgpu::kImage1d, uint32_t{amdgpu::kImageDim2dArray}}) {
      SCOPED_TRACE(dim);
      dimension_override_ = dim;
      bind_image(image);
      for (const std::array<uint32_t, 3> coordinates :
           {std::array<uint32_t, 3>{5, 1, 0}, std::array<uint32_t, 3>{5, 3, 1}}) {
        const uint32_t offset =
            coordinates[2] * 1280 + coordinates[1] * row_pitch + coordinates[0] * 4;
        write_coordinates(0, coordinates);
        cu_->write_vgpr(vb_ + 8, 0, 0x12340000u + offset);
        ASSERT_TRUE(decoded_issue(false, 3));
        EXPECT_EQ(memory_.read32(kBase + offset), 0x12340000u + offset);
        memory_.write32(kBase + offset, 0xabcd0000u + offset);
        cu_->l1_vector().invalidate_all();
        l2_.invalidate_all();
        ASSERT_TRUE(decoded_issue(true, 3));
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcd0000u + offset);
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
  declare_array_ = false; // GFX10+ DIM supplies the coordinate inventory.
  bind_image(image);
  wf_->set_exec(1);
  for (uint32_t dim : {amdgpu::kImageDim1dArray, amdgpu::kImageDim2dArray, amdgpu::kImageDim3d}) {
    dimension_override_ = dim;
    for (uint32_t layer : {1u, 2u}) {
      SCOPED_TRACE(testing::Message() << "dim=" << dim << " layer=" << layer);
      // Native R32 1D-array captures on gfx1100 (W7900) and gfx1201 (R9700)
      // place each layer on a 256-byte boundary.
      const uint32_t offset = layer * 256 + 5 * 4;
      write_coordinates(0, dim == amdgpu::kImageDim1dArray ? std::array<uint32_t, 3>{5, layer, 0}
                                                           : std::array<uint32_t, 3>{5, 0, layer});
      cu_->write_vgpr(vb_ + 8, 0, 0x12340000u + offset);
      ASSERT_TRUE(decoded_issue(false, 3));
      EXPECT_EQ(memory_.read32(kBase + offset), 0x12340000u + offset);
      memory_.write32(kBase + offset, 0xabcd0000u + offset);
      cu_->l1_vector().invalidate_all();
      l2_.invalidate_all();
      ASSERT_TRUE(decoded_issue(true, 3));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcd0000u + offset);
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
  wf_->set_exec(1);
  write_coordinates(0, {5, 2, 1});
  for (uint32_t dim : {amdgpu::kImageDim1d, amdgpu::kImageDim2d})
    for (bool load : {false, true}) {
      dimension_override_ = dim;
      wf_->clear_instruction_execution_error();
      EXPECT_FALSE(decoded_issue(load, 3));
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
    }
}

TEST_P(AllImageAccessTest, StubMimgInstructionsScaleDescriptorSelectors) {
  if (arch() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "GFX12 uses VIMAGE/VSAMPLE rather than MIMG";
  for (uint32_t operation = 0; operation < 3; ++operation) {
    std::array<uint32_t, 3> words{};
    switch (arch()) {
    case ROCJITSU_CODE_ARCH_CDNA1:
      words = stub_image_instruction_words(cdna1::build_mimg,
                                           operation == 0   ? cdna1::kImageSampleMimg
                                           : operation == 1 ? cdna1::kImageLoadMipMimg
                                                            : cdna1::kImageAtomicAddMimg,
                                           fields<cdna1::MimgBuilderFields>(2));
      break;
    case ROCJITSU_CODE_ARCH_CDNA2:
      words = stub_image_instruction_words(cdna2::build_mimg,
                                           operation == 0   ? cdna2::kImageSampleMimg
                                           : operation == 1 ? cdna2::kImageLoadMipMimg
                                                            : cdna2::kImageAtomicAddMimg,
                                           fields<cdna2::MimgBuilderFields>(2));
      break;
    case ROCJITSU_CODE_ARCH_RDNA1:
      words = stub_image_instruction_words(rdna1::build_mimg,
                                           operation == 0   ? rdna1::kImageSampleMimg
                                           : operation == 1 ? rdna1::kImageLoadMipMimg
                                                            : rdna1::kImageAtomicAddMimg,
                                           fields<rdna1::MimgBuilderFields>(2));
      break;
    case ROCJITSU_CODE_ARCH_RDNA2:
      words = stub_image_instruction_words(rdna2::build_mimg,
                                           operation == 0   ? rdna2::kImageSampleMimg
                                           : operation == 1 ? rdna2::kImageLoadMipMimg
                                                            : rdna2::kImageAtomicAddMimg,
                                           fields<rdna2::MimgBuilderFields>(2));
      break;
    case ROCJITSU_CODE_ARCH_RDNA3:
      words = stub_image_instruction_words(rdna3::build_mimg,
                                           operation == 0   ? rdna3::kImageSampleMimg
                                           : operation == 1 ? rdna3::kImageLoadMipMimg
                                                            : rdna3::kImageAtomicAddMimg,
                                           fields<rdna3::MimgBuilderFields>(2));
      break;
    case ROCJITSU_CODE_ARCH_RDNA3_5:
      words = stub_image_instruction_words(rdna3_5::build_mimg,
                                           operation == 0   ? rdna3_5::kImageSampleMimg
                                           : operation == 1 ? rdna3_5::kImageLoadMipMimg
                                                            : rdna3_5::kImageAtomicAddMimg,
                                           fields<rdna3_5::MimgBuilderFields>(2));
      break;
    default:
      FAIL();
    }
    std::unique_ptr<Decoder> decoder = Decoder::create(arch());
    DecodeResult decoded = decoder->decode(words.data());
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
  dimension_override_ = image.type - amdgpu::kImage1d;
  bind_image(image);
  wf_->set_exec(1);
  // Native R32 1D-array captures on gfx1100 (W7900) and gfx1201 (R9700)
  // place each layer on a 256-byte boundary.
  for (uint32_t layer : {1u, 2u}) {
    const uint32_t offset = layer * 256 + 5 * 4;
    write_coordinates(0, {5, layer, 0});
    cu_->write_vgpr(vb_ + 8, 0, 0x12340000u + offset);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory_.read32(kBase + offset), 0x12340000u + offset);
    memory_.write32(kBase + offset, 0xabcd0000u + offset);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    ASSERT_TRUE(decoded_issue(true, 2));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcd0000u + offset);
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
      cu_->write_sgpr(sb_ + i, words[i]);
    const uint32_t components = 1u << format;
    component_mask_ = (1u << components) - 1;
    memory_.write32(kBase, 0x80ff4080);
    wf_->set_exec(1);
    write_coordinates(0, {0, 0, 0});
    ASSERT_TRUE(decoded_issue(true, 2));
    for (uint32_t component = 0; component < components; ++component)
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + component, 0), expected[component]);
    wf_->clear_instruction_execution_error();
    EXPECT_FALSE(decoded_issue(false, 2));
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(memory_.read32(kBase), 0x80ff4080u);
    EXPECT_TRUE(wf_->wait_counters().empty());
    wf_->clear_instruction_execution_error();
  }
}

TEST_P(AllImageAccessTest, UnsupportedImageAtomicsDoNotIssueMemory) {
  bind_image(2);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  memory_.write32(kBase, 0x12345678);
  cu_->write_vgpr(vb_ + 8, 0, 1);
  std::array<uint32_t, 3> words{};
  switch (arch()) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    words = image_instruction_words(cdna1::build_mimg, cdna1::kImageAtomicAddMimg,
                                    fields<cdna1::MimgBuilderFields>(2));
    break;
  case ROCJITSU_CODE_ARCH_CDNA2:
    words = image_instruction_words(cdna2::build_mimg, cdna2::kImageAtomicAddMimg,
                                    fields<cdna2::MimgBuilderFields>(2));
    break;
  case ROCJITSU_CODE_ARCH_RDNA1:
    words = image_instruction_words(rdna1::build_mimg, rdna1::kImageAtomicAddMimg,
                                    fields<rdna1::MimgBuilderFields>(2));
    break;
  case ROCJITSU_CODE_ARCH_RDNA2:
    words = image_instruction_words(rdna2::build_mimg, rdna2::kImageAtomicAddMimg,
                                    fields<rdna2::MimgBuilderFields>(2));
    break;
  case ROCJITSU_CODE_ARCH_RDNA3:
    words = image_instruction_words(rdna3::build_mimg, rdna3::kImageAtomicAddMimg,
                                    fields<rdna3::MimgBuilderFields>(2));
    break;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    words = image_instruction_words(rdna3_5::build_mimg, rdna3_5::kImageAtomicAddMimg,
                                    fields<rdna3_5::MimgBuilderFields>(2));
    break;
  case ROCJITSU_CODE_ARCH_RDNA4:
    words = rdna4::build_vimage(rdna4::kImageAtomicAddUintVimage,
                                fields<rdna4::VimageBuilderFields>(2));
    break;
  default:
    FAIL() << "Unexpected image architecture";
  }
  auto decoder = Decoder::create(arch());
  auto decoded = decoder->decode(words.data());
  ASSERT_TRUE(decoded.succeeded());
  auto instruction = std::move(decoded).value();
  instruction->execute(*instruction, wf_);
  EXPECT_EQ(wf_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
  EXPECT_FALSE(instruction->data());
  EXPECT_EQ(memory_.read32(kBase), 0x12345678u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 1u);
  EXPECT_TRUE(wf_->wait_counters().empty());
}

TEST_P(AllImageAccessTest, CompressedDescriptorsStopBeforeAccessingOrdinaryBacking) {
  for (uint32_t swizzle : {0u, arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 2u : 22u}) {
    SCOPED_TRACE(swizzle);
    bind_image(2, 64, 64, swizzle);
    const uint32_t word6 = cu_->read_sgpr(sb_ + 6);
    for (uint32_t flag : {21u, 20u}) {
      if (flag == 20 && arch_is_cdna_4_or_lower(arch()))
        continue;
      cu_->write_sgpr(sb_ + 6, word6 | (1u << flag));
      for (bool load : {false, true}) {
        wf_->clear_instruction_execution_error();
        EXPECT_FALSE(decoded_issue(load, 2));
        EXPECT_EQ(wf_->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnimplementedInstruction);
      }
    }
  }
}

TEST_P(AllImageAccessTest, UnsupportedLayoutIsDiagnosedEvenWithoutInBoundsLanes) {
  bind_image(2, 3, 2, 8);
  for (uint64_t lanes : {uint64_t{0}, uint64_t{1}}) {
    wf_->clear_instruction_execution_error();
    wf_->set_exec(lanes);
    write_coordinates(0, {100, 100, 100});
    EXPECT_FALSE(decoded_issue(true, 2));
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
  }
}

TEST_P(AllImageAccessTest, StoreRequiresUnnormalizedCoordinates) {
  if (arch() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "VIMAGE has no UNORM control";
  bind_image(2);
  unnormalized_ = false;
  EXPECT_FALSE(decoded_issue(false, 2));
  EXPECT_EQ(wf_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnsupportedOperandValue);
}

TEST_P(AllImageAccessTest, LegalCubeAndMsaaDimensionsStopAsUnimplemented) {
  if (arch_is_cdna_4_or_lower(arch()))
    GTEST_SKIP() << "GFX9 has no DIM field";
  bind_image(2);
  for (uint32_t dimension : {3u, 6u, 7u}) {
    SCOPED_TRACE(dimension);
    wf_->clear_instruction_execution_error();
    dimension_override_ = dimension;
    EXPECT_FALSE(decoded_issue(true, 2));
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
  }
}

TEST_P(AllImageAccessTest, ScatteredAndPackedCoordinatesReachTheSameTexel) {
  if (arch_is_cdna_4_or_lower(arch()))
    GTEST_SKIP() << "GFX9 image coordinates are sequential";
  wf_->set_exec(1);
  bind_image(3);
  scattered_ = true;
  // Independent 3D linear layout: byte offset = slice + row + 2 DWORDs.
  const uint32_t byte_offset = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 648 : 776;
  for (bool packed : {false, true}) {
    packed_coordinates_ = packed;
    cu_->write_vgpr(vb_, 0, packed ? 2 | (1u << 16) : 2);
    cu_->write_vgpr(vb_ + 10, 0, 1);
    cu_->write_vgpr(vb_ + 12, 0, 1);
    cu_->write_vgpr(vb_ + 8, 0, 0x10203040u + packed);
    ASSERT_TRUE(decoded_issue(false, 3));
    EXPECT_EQ(memory_.read32(kBase + byte_offset), 0x10203040u + packed);
    const uint32_t load_value = 0xabcde000u + packed;
    memory_.write32(kBase + byte_offset, load_value);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    cu_->write_vgpr(vb_ + 8, 0, 0);
    ASSERT_TRUE(decoded_issue(true, 3));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), load_value);
  }
}

TEST_P(AllImageAccessTest, WaitChecksConsumeTheEncodedCoordinateRegisters) {
  Image image;
  image.type = amdgpu::kImage3d;
  if (arch_is_cdna_4_or_lower(arch()))
    bind_image(image); // GFX9 selects the actual coordinate count from T#.
  wf_->set_exec(1);
  amdgpu::MemoryWaitScoreboard &state = wf_->ensure_memory_wait_scoreboard();
  scattered_ = !arch_is_cdna_4_or_lower(arch());
  for (uint16_t pending : {0, 1, 2, 3, 10, 12}) {
    SCOPED_TRACE(pending);
    state.clear();
    const uint64_t sequence = state.issue(amdgpu::WaitCounterKind::Load);
    state.add(
        {sequence, 0x100, 1, {RegClass::VGPR, pending, 1}, amdgpu::WaitCounterKind::Load, 0xf});
    unsigned reports = 0;
    state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
    std::unique_ptr<Instruction> instruction = decoded_instruction(false, 3);
    ASSERT_NE(instruction, nullptr);
    state.check_instruction(*instruction, *wf_);
    EXPECT_EQ(reports, (scattered_ ? pending == 0 || pending == 10 || pending == 12 : pending < 3)
                           ? 1u
                           : 0u);
  }
}

TEST_P(AllImageAccessTest, LivenessUsesScatteredCoordinatesAndScaledDescriptor) {
  scattered_ = !arch_is_cdna_4_or_lower(arch());
  resource_selector_ = 4;
  for (bool load : {false, true}) {
    for (bool packed : {false, true}) {
      SCOPED_TRACE(testing::Message() << "load=" << load << " packed=" << packed);
      packed_coordinates_ = packed;
      std::unique_ptr<Instruction> instruction = decoded_instruction(load, 3);
      ASSERT_NE(instruction, nullptr);
      const InstDefUse def_use(*instruction, wf_->wf_size());
      EXPECT_TRUE(def_use.uses.contains({RegClass::SGPR, 16, 8}));
      EXPECT_FALSE(def_use.uses.contains({RegClass::SGPR, 4, 1}));
      for (uint16_t reg : {0, 1, 2, 3, 10, 12}) {
        const bool used =
            reg == 0 || reg == (scattered_ ? 10 : 1) || (!packed && reg == (scattered_ ? 12 : 2));
        EXPECT_EQ(def_use.uses.contains({RegClass::VGPR, reg, 1}), used) << reg;
      }
    }
  }
}

TEST_P(AllImageAccessTest, A16CoordinateWaitChecksIgnoreTheUnusedHighHalf) {
  bind_image(3);
  wf_->set_exec(1);
  scattered_ = !arch_is_cdna_4_or_lower(arch());
  packed_coordinates_ = true;
  std::unique_ptr<Instruction> instruction = decoded_instruction(false, 3);
  ASSERT_NE(instruction, nullptr);
  amdgpu::MemoryWaitScoreboard &state = wf_->ensure_memory_wait_scoreboard();
  for (uint8_t bytes : {0x3, 0xc}) {
    SCOPED_TRACE(bytes);
    state.clear();
    const auto counter = amdgpu::WaitCounterKind::Load;
    const uint16_t last = scattered_ ? 10 : 1;
    state.add({state.issue(counter), 0x100, 1, {RegClass::VGPR, last, 1}, counter, bytes});
    unsigned reports = 0;
    state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
    state.check_instruction(*instruction, *wf_);
    EXPECT_EQ(reports, bytes == 0x3 ? 1u : 0u);
  }
}

TEST_P(AllImageAccessTest, D16LoadWaitOwnershipDistinguishesPreservedAndZeroedHighHalves) {
  wf_->set_exec(1);
  packed_data_ = true;
  std::unique_ptr<Instruction> instruction = decoded_instruction(true, 3);
  ASSERT_NE(instruction, nullptr);
  const InstDefUse def_use(*instruction, wf_->wf_size());
  EXPECT_EQ(def_use.uses.contains({RegClass::VGPR, 8, 1}), !cu_->sram_ecc());
  amdgpu::MemoryWaitScoreboard &state = wf_->ensure_memory_wait_scoreboard();
  for (uint8_t bytes : {0x3, 0xc}) {
    SCOPED_TRACE(bytes);
    state.clear();
    unsigned reports = 0;
    cu_->track_memory_wait(*instruction, *wf_);
    state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
    state.access({RegClass::VGPR, 8, 1}, 1, bytes, false);
    EXPECT_EQ(reports, bytes == 0x3 || cu_->sram_ecc() ? 1u : 0u);
  }
}

TEST_P(AllImageAccessTest, TiledBaseXorStopsBeforeImageMemoryAccess) {
  bind_image(2, 64, 64, arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 2 : 22);
  cu_->write_sgpr(sb_, (kBase + 256) >> 8);
  wf_->set_exec(1);
  memory_.write32(kBase + 256, 0x12345678);
  cu_->write_vgpr(vb_ + 8, 0, 0xabcdef01);
  for (uint32_t x : {0u, 64u}) {
    write_coordinates(0, {x, 0, 0});
    for (bool load : {false, true}) {
      wf_->clear_instruction_execution_error();
      EXPECT_FALSE(decoded_issue(load, 2));
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
      EXPECT_EQ(memory_.read32(kBase + 256), 0x12345678u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcdef01u);
    }
  }
}

TEST_P(AllImageAccessTest, ThickDisplayVolumesStopBeforeImageMemoryAccess) {
  if (arch_is_cdna_4_or_lower(arch()) || arch() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "GFX9 display volumes are thin; GFX12 has different swizzle encodings";
  bind_image(3, 64, 64, 26);
  for (bool load : {false, true}) {
    wf_->clear_instruction_execution_error();
    EXPECT_FALSE(decoded_issue(load, 3));
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
  }
}

INSTANTIATE_TEST_SUITE_P(Architectures, AllImageAccessTest,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));

enum class SampleTestOperation { Zero, Lod, Gradients, Gather, Unsupported };

// Raw S# field encodings independent of the sampler decoder under test.
enum class Clamp : uint32_t { Repeat = 0, Mirror = 1, Edge = 2, Border = 6 };
enum class MipFilter : uint32_t { None = 0, Point = 1, Linear = 2 };

struct Sampler {
  Clamp clamp_x = Clamp::Edge, clamp_y = Clamp::Edge, clamp_z = Clamp::Edge;
  bool linear_xy = false; // Program the same XY_MAG_FILTER and XY_MIN_FILTER.
  MipFilter mip = MipFilter::None;
  uint32_t min_lod_fixed8 = 0, max_lod_fixed8 = 3 * 256;
  int32_t bias_fixed8 = 0;
  bool normalized = true;
  uint32_t perf_mip = 0, perf_z = 0;
};

class ImageSamplerControlTest : public AllImageAccessTest {
protected:
  void bind_sample_image(Image image, bool floating = true) {
    image.num_format = floating ? 7 : kNfmtUint;
    image.rdna_format = floating ? 22 : kR32Uint;
    bind_image(image);
  }

  void bind_sampler(const Sampler &sampler = {}) {
    const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
    const uint32_t xy_filter = sampler.linear_xy ? 1u : 0u;
    // Z_FILTER_NONE, zero anisotropy, transparent black border.
    const std::array<uint32_t, 4> words{
        static_cast<uint32_t>(sampler.clamp_x) | (static_cast<uint32_t>(sampler.clamp_y) << 3) |
            (static_cast<uint32_t>(sampler.clamp_z) << 6) | (sampler.normalized ? 0u : 1u << 15),
        sampler.min_lod_fixed8 | (sampler.max_lod_fixed8 << (gfx12 ? 13 : 12)) |
            (gfx12 ? 0u : sampler.perf_mip << 24) | (sampler.perf_z << 28),
        (static_cast<uint32_t>(sampler.bias_fixed8) & 0x3fffu) | (xy_filter << 20) |
            (xy_filter << 22) | (static_cast<uint32_t>(sampler.mip) << 26) |
            (gfx12 ? (sampler.perf_mip & 3) << 30 : 0u),
        gfx12 ? sampler.perf_mip >> 2 : 0u};
    for (uint32_t word = 0; word < words.size(); ++word)
      cu_->write_sgpr(sb_ + 8 + word, words[word]);
  }

  template <typename Fields, typename Builder>
  std::array<uint32_t, 3>
  sample_control_mimg_words(Builder builder, std::array<uint16_t, 5> opcodes,
                            SampleTestOperation operation, bool d16, uint32_t sampler) {
    return sample_mimg_words<Fields>(builder, opcodes[static_cast<size_t>(operation)], d16,
                                     sampler);
  }

  std::unique_ptr<Instruction> sample_instruction(SampleTestOperation operation, bool d16 = false,
                                                  uint32_t sampler = 8) {
    std::array<uint32_t, 3> words{};
    switch (arch()) {
    case ROCJITSU_CODE_ARCH_CDNA1:
      words = sample_control_mimg_words<cdna1::MimgBuilderFields>(
          cdna1::build_mimg,
          {cdna1::kImageSampleLzMimg, cdna1::kImageSampleLMimg, cdna1::kImageSampleDMimg,
           cdna1::kImageGather4LzMimg, cdna1::kImageSampleOMimg},
          operation, d16, sampler);
      break;
    case ROCJITSU_CODE_ARCH_RDNA1:
      words = sample_control_mimg_words<rdna1::MimgBuilderFields>(
          rdna1::build_mimg,
          {rdna1::kImageSampleLzMimg, rdna1::kImageSampleLMimg, rdna1::kImageSampleDMimg,
           rdna1::kImageGather4LzMimg, rdna1::kImageSampleOMimg},
          operation, d16, sampler);
      break;
    case ROCJITSU_CODE_ARCH_RDNA2:
      words = sample_control_mimg_words<rdna2::MimgBuilderFields>(
          rdna2::build_mimg,
          {rdna2::kImageSampleLzMimg, rdna2::kImageSampleLMimg, rdna2::kImageSampleDMimg,
           rdna2::kImageGather4LzMimg, rdna2::kImageSampleOMimg},
          operation, d16, sampler);
      break;
    case ROCJITSU_CODE_ARCH_RDNA3:
      words = sample_control_mimg_words<rdna3::MimgBuilderFields>(
          rdna3::build_mimg,
          {rdna3::kImageSampleLzMimg, rdna3::kImageSampleLMimg, rdna3::kImageSampleDMimg,
           rdna3::kImageGather4LzMimg, rdna3::kImageSampleOMimg},
          operation, d16, sampler);
      break;
    case ROCJITSU_CODE_ARCH_RDNA3_5:
      words = sample_control_mimg_words<rdna3_5::MimgBuilderFields>(
          rdna3_5::build_mimg,
          {rdna3_5::kImageSampleLzMimg, rdna3_5::kImageSampleLMimg, rdna3_5::kImageSampleDMimg,
           rdna3_5::kImageGather4LzMimg, rdna3_5::kImageSampleOMimg},
          operation, d16, sampler);
      break;
    case ROCJITSU_CODE_ARCH_CDNA2: {
      // CDNA2's public ISA input exposes only base IMAGE_SAMPLE. A single-level
      // sample without mip filtering has the same controls as the zero form.
      if (operation != SampleTestOperation::Zero)
        return {};
      words = sample_mimg_words<cdna2::MimgBuilderFields>(cdna2::build_mimg,
                                                          cdna2::kImageSampleMimg, d16, sampler);
      break;
    }
    case ROCJITSU_CODE_ARCH_RDNA4: {
      auto encoding = rdna4::VsampleBuilderFields{};
      encoding.dim = dimension_override_.value_or(1);
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
    std::unique_ptr<Decoder> decoder = Decoder::create(arch());
    DecodeResult decoded = decoder->decode(words.data());
    EXPECT_TRUE(decoded.succeeded());
    return decoded.succeeded() ? std::move(decoded).value() : nullptr;
  }

  bool sample(SampleTestOperation operation, bool d16 = false, uint32_t sampler = 8) {
    std::unique_ptr<Instruction> instruction = sample_instruction(operation, d16, sampler);
    if (!instruction)
      return false;
    instruction->execute(*instruction, wf_);
    if (!instruction->data())
      return false;
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    EXPECT_EQ(pipeline.issue(instruction.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    // Later cases seed backing storage directly, as in the base image fixture.
    cu_->l1_vector().flush_all();
    l2_.flush_all();
    return true;
  }
};

TEST_P(ImageSamplerControlTest, WaitChecksFollowDescriptorLodAndGradientOperands) {
  const bool gfx9 = arch() == ROCJITSU_CODE_ARCH_CDNA1;
  if (!gfx9 && arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Covers descriptor-selected GFX9 tuples and GFX12 VSAMPLE selectors";
  wf_->set_exec(1);
  std::unique_ptr<Decoder> decoder = Decoder::create(arch());
  amdgpu::MemoryWaitScoreboard &state = wf_->ensure_memory_wait_scoreboard();
  const auto has_hazard = [&](const Instruction &instruction, uint16_t reg, uint8_t bytes) {
    state.clear();
    const auto counter = amdgpu::WaitCounterKind::Load;
    state.add({state.issue(counter), 0x100, 1, {RegClass::VGPR, reg, 1}, counter, bytes});
    unsigned reports = 0;
    state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
    state.check_instruction(instruction, *wf_);
    return reports != 0;
  };
  for (uint32_t dimensions : {2u, 3u}) {
    bind_image(dimensions);
    for (bool packed : {false, true}) {
      SCOPED_TRACE(testing::Message() << dimensions << ':' << packed);
      if (gfx9) {
        packed_coordinates_ = packed;
        auto encoding = fields<cdna1::MimgBuilderFields>(dimensions);
        encoding.vdata = 28;
        encoding.ssamp = 2;
        auto words = image_instruction_words(cdna1::build_mimg, cdna1::kImageSampleLMimg, encoding);
        DecodeResult lod = decoder->decode(words.data());
        ASSERT_TRUE(lod.succeeded());
        const uint16_t lod_reg = packed ? dimensions / 2 : dimensions;
        EXPECT_TRUE(has_hazard(*lod.value(), lod_reg, 0x3));
        EXPECT_EQ(has_hazard(*lod.value(), lod_reg, 0xc), !packed || dimensions == 3);
        EXPECT_FALSE(has_hazard(*lod.value(), lod_reg + 1, 0xf));
        // Packed GFX9 gradients are excluded from execution. Score the
        // supported full-width gradient prefix and following coordinates.
        if (!packed) {
          words = image_instruction_words(cdna1::build_mimg, cdna1::kImageSampleDMimg, encoding);
          DecodeResult grad = decoder->decode(words.data());
          ASSERT_TRUE(grad.succeeded());
          EXPECT_TRUE(has_hazard(*grad.value(), 2 * dimensions - 1, 0xc));
          EXPECT_TRUE(has_hazard(*grad.value(), 3 * dimensions - 1, 0xc));
          EXPECT_FALSE(has_hazard(*grad.value(), 3 * dimensions, 0xf));
        }
      } else {
        rdna4::VsampleBuilderFields encoding{};
        encoding.dim = dimensions - 1;
        encoding.dmask = 1;
        encoding.vdata = 28;
        encoding.samp = 8;
        encoding.a16 = packed;
        encoding.vaddr0 = 1;
        encoding.vaddr1 = 12;
        encoding.vaddr2 = 14;
        encoding.vaddr3 = 20;
        const auto words = rdna4::build_vsample(rdna4::kImageSampleDVsample, encoding);
        DecodeResult grad = decoder->decode(words.data());
        ASSERT_TRUE(grad.succeeded());
        for (uint16_t reg : {1, 12, 14, 20})
          EXPECT_TRUE(has_hazard(*grad.value(), reg, 0xc)) << reg;
        EXPECT_FALSE(has_hazard(*grad.value(), 2, 0xf));
        const uint16_t first_coordinate = 21 + (dimensions == 3 ? 2 : 0);
        if (dimensions == 3) {
          EXPECT_TRUE(has_hazard(*grad.value(), first_coordinate - 1, 0xc));
        }
        const uint16_t last = first_coordinate + (packed ? (dimensions - 1) / 2 : dimensions - 1);
        EXPECT_TRUE(has_hazard(*grad.value(), last, 0x3));
        EXPECT_EQ(has_hazard(*grad.value(), last, 0xc), !packed || dimensions == 2);
        EXPECT_FALSE(has_hazard(*grad.value(), last + 1, 0xf));
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, UnnormalizedWrapUsesClampAndMirrorOnce) {
  Image image;
  image.width = 16;
  bind_sample_image(image);
  for (uint32_t x = 0; x < 16; ++x)
    memory_.write32(kBase + x * 4, std::bit_cast<uint32_t>(float(x)));
  wf_->set_exec(1);
  for (Clamp wrap : {Clamp::Repeat, Clamp::Mirror})
    for (bool linear : {false, true}) {
      bind_sampler({.clamp_x = wrap,
                    .clamp_y = wrap,
                    .clamp_z = Clamp::Repeat,
                    .linear_xy = linear,
                    .normalized = false});
      for (const auto &[x, mirror] :
           {std::pair{-INFINITY, 15.0f}, std::pair{-16.5f, 15.0f}, std::pair{-1.5f, 1.0f},
            std::pair{-0.5f, 0.0f}, std::pair{0.0f, 0.0f}, std::pair{16.5f, 15.0f}}) {
        cu_->write_vgpr(vb_, 0, std::bit_cast<uint32_t>(x));
        cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(0.5f));
        ASSERT_TRUE(sample(SampleTestOperation::Zero));
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), std::bit_cast<uint32_t>(wrap == Clamp::Mirror ? mirror
                                                                      : x < 1               ? 0.0f
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
  bind_sampler({.normalized = false});
  wf_->set_exec(1);
  // Captured on gfx1100/gfx1201: gather preserves NaN payload and denormal bits.
  for (uint32_t payload : {0x7fc12345u, 0x7f800001u, 0xffc00001u, 1u, 0x80000001u, 0x007fffffu,
                           0x80000000u, 0x7f800000u}) {
    memory_.write32(kBase, payload);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    cu_->write_vgpr(vb_, 0, std::bit_cast<uint32_t>(0.5f));
    cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(0.5f));
    ASSERT_TRUE(sample(SampleTestOperation::Zero));
    const bool nan = (payload & 0x7fffffffu) > 0x7f800000u;
    const bool denormal = (payload & 0x7f800000u) == 0;
    const uint32_t point = nan ? 0xffc00000u : denormal ? payload & 0x80000000u : payload;
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), point);
    cu_->write_vgpr(vb_, 0, std::bit_cast<uint32_t>(1.0f));
    cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(1.0f));
    ASSERT_TRUE(sample(SampleTestOperation::Gather));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 11, 0), payload);
  }
}

TEST_P(ImageSamplerControlTest, GatherReturnsAllFourTexelsInComponentOrder) {
  if (arch() == ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP() << "The public CDNA2 ISA omits gather forms";
  Image image;
  image.width = image.height = 2;
  bind_sample_image(image);
  bind_sampler({.normalized = false});
  wf_->set_exec(1);
  write_coordinates(0, {std::bit_cast<uint32_t>(1.0f), std::bit_cast<uint32_t>(1.0f), 0});
  for (uint32_t y = 0; y < 2; ++y)
    for (uint32_t x = 0; x < 2; ++x)
      memory_.write32(kBase + 256 * y + 4 * x, std::bit_cast<uint32_t>(float(1 + 2 * y + x)));

  // A row-major {1, 2, 3, 4} returned these F32 and D16 tuples on gfx1100 and gfx1201.
  ASSERT_TRUE(sample(SampleTestOperation::Gather));
  constexpr std::array<float, 4> expected{3, 4, 2, 1};
  for (size_t i = 0; i < expected.size(); ++i)
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + i, 0), std::bit_cast<uint32_t>(expected[i]));
  ASSERT_TRUE(sample(SampleTestOperation::Gather, true));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x44004200u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0x3c004000u);
}

TEST_P(ImageSamplerControlTest, LinearXyFilterBlendsVolumeSlicesWithZFilterNone) {
  Image image;
  image.type = amdgpu::kImage3d;
  image.width = image.depth = 2;
  dimension_override_ = amdgpu::kImageDim3d;
  bind_sample_image(image);
  bind_sampler({.linear_xy = true, .normalized = false});
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, std::bit_cast<uint32_t>(0.75f)});
  for (uint32_t x = 0; x < 2; ++x) {
    memory_.write32(kBase + 4 * x, 0);
    memory_.write32(kBase + 256 + 4 * x, std::bit_cast<uint32_t>(4.0f));
  }
  // gfx1100 and gfx1201 with ROCr's XY linear and Z_FILTER_NONE sampler
  // return 0.75 * slice(0) + 0.25 * slice(1) = 1 at this coordinate.
  ASSERT_TRUE(sample(SampleTestOperation::Zero));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), std::bit_cast<uint32_t>(1.0f));
}

TEST_P(ImageSamplerControlTest, UnqualifiedLodControlsAreRejected) {
  const bool legacy = arch_is_cdna_4_or_lower(arch()) || arch() == ROCJITSU_CODE_ARCH_RDNA1 ||
                      arch() == ROCJITSU_CODE_ARCH_RDNA2;
  for (bool resource_control : {false, true}) {
    if (!resource_control && !legacy)
      continue;
    Image image;
    image.min_lod = resource_control ? 1 : 0;
    bind_sample_image(image);
    bind_sampler({.mip = MipFilter::Point});
    if (!resource_control) {
      cu_->write_sgpr(sb_ + 10, cu_->read_sgpr(sb_ + 10) | (1u << 28));
    }
    wf_->set_exec(1);
    wf_->clear_instruction_execution_error();
    EXPECT_FALSE(sample(SampleTestOperation::Zero));
    EXPECT_EQ(wf_->instruction_execution_error(),
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
      dimension_override_ = layered ? amdgpu::kImageDim1dArray : amdgpu::kImageDim1d;
      bind_sample_image(image);
      // GFX12 array rows use 128-byte alignment, while allocation keeps
      // 256-byte rows for the layer stride (two rows = 512 bytes).
      const uint32_t row_bytes = layered && arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
      for (uint32_t i = 0; i < 4; ++i) {
        memory_.write32(kBase + layer * 512 + i * 4, std::bit_cast<uint32_t>(1000.0f * layer + i));
        memory_.write32(kBase + layer * 512 + row_bytes + i * 4,
                        std::bit_cast<uint32_t>(1000.0f * layer + 100.0f + i));
      }
      cu_->l1_vector().invalidate_all();
      l2_.invalidate_all();
      wf_->set_exec(15);
      for (bool normalized : {false, true}) {
        for (bool linear : {false, true}) {
          for (Clamp wrap : {Clamp::Repeat, Clamp::Mirror, Clamp::Edge, Clamp::Border}) {
            bind_sampler({.clamp_y = wrap, .linear_xy = linear, .normalized = normalized});
            for (uint32_t lane = 0; lane < 4; ++lane) {
              const float x = lane + (linear ? 0.25f : 0.5f);
              cu_->write_vgpr(vb_, lane, std::bit_cast<uint32_t>(normalized ? x / 4 : x));
              // The instruction has no Y input; a following live VGPR is irrelevant.
              cu_->write_vgpr(vb_ + 1, lane,
                              std::bit_cast<uint32_t>(layered ? float(layer) : 1.5f));
              cu_->write_vgpr(vb_ + 2, lane, std::bit_cast<uint32_t>(1.5f));
            }
            ASSERT_TRUE(sample(SampleTestOperation::Zero));
            for (uint32_t lane = 0; lane < 4; ++lane) {
              float expected = 1000.0f * layer + (linear && lane ? lane - 0.25f : float(lane));
              if (linear && wrap == Clamp::Border)
                expected *= 0.5f; // Half the footprint reads transparent black.
              else if (linear && normalized && wrap == Clamp::Repeat)
                expected += 50.0f; // Half reads the last row, whose values are 100 higher.
              EXPECT_EQ(cu_->read_vgpr(vb_ + 8, lane), std::bit_cast<uint32_t>(expected));
            }
          }
        }
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, GenerationSpecificUnsupportedSamplerControlsPreserveDestination) {
  const bool gfx9 = arch_is_cdna_4_or_lower(arch());
  if (arch() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "GFX12 has none of these controls";
  wf_->set_exec(1);
  for (bool filter_precision : {false, true}) {
    if (filter_precision && !gfx9)
      continue;
    Image image;
    image.width = image.height = 2;
    bind_sampler({.linear_xy = filter_precision, .mip = gfx9 ? MipFilter::None : MipFilter::Point});
    if (gfx9 && !filter_precision) {
      image.data_format = kDfmt8888;
      image.num_format = 9; // GFX9 sRGB.
      bind_image(image);
      memory_.write32(kBase, 0x80402020);
      cu_->write_sgpr(sb_ + 11, 1u << 12); // SKIP_DEGAMMA
    } else {
      bind_sample_image(image);
      if (filter_precision)
        cu_->write_sgpr(sb_ + 10, cu_->read_sgpr(sb_ + 10) | (1u << 30)); // FILTER_PREC_FIX
      else {
        cu_->write_sgpr(sb_ + 10, cu_->read_sgpr(sb_ + 10) | (1u << 31)); // DERIV_ADJUST_EN
        cu_->write_sgpr(sb_ + 11, 0x01234567);
      }
    }
    for (uint32_t reg = 0; reg < 6; ++reg)
      cu_->write_vgpr(vb_ + reg, 0, std::bit_cast<uint32_t>(0.25f));
    cu_->write_vgpr(vb_ + 8, 0, 0xabcdef01);
    wf_->clear_instruction_execution_error();
    const SampleTestOperation operation =
        gfx9 ? SampleTestOperation::Zero : SampleTestOperation::Gradients;
    EXPECT_FALSE(sample(operation));
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcdef01u);
    EXPECT_TRUE(wf_->wait_counters().empty());
  }
}

TEST_P(ImageSamplerControlTest, UnqualifiedSamplerFlagsPreserveDestination) {
  const bool gfx9 = arch_is_cdna_4_or_lower(arch());
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  bind_sample_image(Image{});
  bind_sampler();
  wf_->set_exec(1);
  memory_.write32(kBase, std::bit_cast<uint32_t>(0.75f));
  ASSERT_TRUE(sample(SampleTestOperation::Zero));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), std::bit_cast<uint32_t>(0.75f));

  struct Flag {
    uint32_t word, bit;
    bool present;
  };
  const Flag flags[] = {{0, 31, gfx9},                 // COMPAT_MODE
                        {2, gfx9 ? 29u : 30u, !gfx12}, // BLEND_ZERO_PRT
                        {2, gfx9 ? 31u : 29u, true}};  // ANISO_OVERRIDE
  for (const auto &flag : flags) {
    if (!flag.present)
      continue;
    SCOPED_TRACE(testing::Message() << flag.word << ':' << flag.bit);
    bind_sampler();
    cu_->write_sgpr(sb_ + 8 + flag.word, cu_->read_sgpr(sb_ + 8 + flag.word) | (1u << flag.bit));
    cu_->write_vgpr(vb_ + 8, 0, 0xabcdef01);
    wf_->clear_instruction_execution_error();
    EXPECT_FALSE(sample(SampleTestOperation::Zero));
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcdef01u);
    EXPECT_TRUE(wf_->wait_counters().empty());
  }
}

TEST_P(ImageSamplerControlTest, UnbackedSamplerChecksTheNonNullResourceFirst) {
  wf_->set_exec(1);
  bind_sampler();
  memory_.write32(kBase, std::bit_cast<uint32_t>(0.75f));
  for (uint32_t type : {amdgpu::kImage2d, amdgpu::kImageCube, amdgpu::kImage2dMsaa}) {
    Image image;
    image.type = type;
    bind_sample_image(image);
    dimension_override_ = type - amdgpu::kImage1d;
    for (uint32_t sampler : {8u, 104u}) {
      SCOPED_TRACE(testing::Message() << type << ':' << sampler);
      cu_->write_vgpr(vb_ + 8, 0, 0xabcdef01);
      wf_->clear_instruction_execution_error();
      if (type == amdgpu::kImage2d) {
        ASSERT_TRUE(sample(SampleTestOperation::Zero, false, sampler));
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), sampler == 8 ? std::bit_cast<uint32_t>(0.75f) : 0u);
      } else {
        EXPECT_FALSE(sample(SampleTestOperation::Zero, false, sampler));
        EXPECT_EQ(wf_->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnimplementedInstruction);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcdef01u);
        EXPECT_TRUE(wf_->wait_counters().empty());
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, LinkedResourcesStopBeforeSampleLoadOrStore) {
  const bool gfx9 = arch_is_cdna_4_or_lower(arch());
  bind_sample_image(Image{});
  bind_sampler();
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  constexpr uint32_t texel = std::bit_cast<uint32_t>(0.75f);
  constexpr uint32_t sentinel = 0xabcdef01;
  memory_.write32(kBase, texel);
  ASSERT_TRUE(sample(SampleTestOperation::Zero));
  ASSERT_EQ(cu_->read_vgpr(vb_ + 8, 0), texel);
  // Word 5 bit 24 selects a linked resource on GFX10/11/12. On GFX9 it
  // belongs to the metadata address, unused by this uncompressed image.
  cu_->write_sgpr(sb_ + 5, cu_->read_sgpr(sb_ + 5) | (1u << 24));
  enum class Operation { Sample, Load, Store };
  for (Operation op : {Operation::Sample, Operation::Load, Operation::Store}) {
    SCOPED_TRACE(static_cast<int>(op));
    wf_->clear_instruction_execution_error();
    cu_->write_vgpr(vb_ + 8, 0, sentinel);
    const bool issued = op == Operation::Sample ? sample(SampleTestOperation::Zero)
                                                : decoded_issue(op == Operation::Load, 2);
    EXPECT_EQ(issued, gfx9);
    if (!gfx9) {
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
    }
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), gfx9 && op != Operation::Store ? texel : sentinel);
    EXPECT_EQ(memory_.read32(kBase), gfx9 && op == Operation::Store ? sentinel : texel);
    EXPECT_TRUE(wf_->wait_counters().empty());
  }
}

TEST_P(ImageSamplerControlTest, InvalidFormatCompletesBeforeResourceTypeQualification) {
  Image image;
  image.type = amdgpu::kImageCube;
  image.data_format = image.rdna_format = 0; // INVALID
  bind_image(image);
  bind_sampler();
  wf_->set_exec(1);
  constexpr uint32_t texel = 0x12345678;
  constexpr uint32_t sentinel = 0xdeadbeef;
  memory_.write32(kBase, texel);
  cu_->write_vgpr(vb_ + 8, 0, sentinel);
  ASSERT_TRUE(sample(SampleTestOperation::Zero));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0u);
  cu_->write_vgpr(vb_ + 8, 0, sentinel);
  ASSERT_TRUE(decoded_issue(true, 2));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0u);
  cu_->write_vgpr(vb_ + 8, 0, sentinel);
  ASSERT_TRUE(decoded_issue(false, 2));
  EXPECT_EQ(memory_.read32(kBase), texel);
}

TEST_P(ImageSamplerControlTest, SrgbConversionPrecedesSpatialAndMipFiltering) {
  const bool gfx9 = arch_is_cdna_4_or_lower(arch());
  const bool gfx10 = arch() == ROCJITSU_CODE_ARCH_RDNA1 || arch() == ROCJITSU_CODE_ARCH_RDNA2;
  wf_->set_exec(15);
  for (uint32_t lane = 0; lane < 4; ++lane)
    write_coordinates(lane, {std::bit_cast<uint32_t>(0.5f), std::bit_cast<uint32_t>(0.5f),
                             std::bit_cast<uint32_t>(0.5f)});
  // gfx1100 and gfx1201 returned 0x1.284p-2 for both filters, versus
  // 0x1.010102p-1 with R8_UNORM. Earlier ISAs use the IEC sRGB transfer
  // of 64/255 and 192/255 followed by their arithmetic mean.
  const float expected = gfx9 || gfx10 ? 0.289192f : 0x1.284p-2f;
  for (bool mip : {false, true}) {
    SCOPED_TRACE(mip);
    Image image;
    image.width = image.height = 2;
    image.data_format = 1; // R8
    image.num_format = 9;  // sRGB
    image.rdna_format = gfx10 ? 128 : 64;
    image.pitch = 256;
    image.mip_levels = mip ? 2 : 1;
    image.last_level = image.mip_levels - 1;
    bind_image(image);
    // CDNA2 exposes the base SAMPLE form. Its MIN_LOD clamp supplies the
    // same fractional level as the explicit LOD operand on the other ISAs.
    const bool implicit_mip = mip && arch() == ROCJITSU_CODE_ARCH_CDNA2;
    bind_sampler({.linear_xy = !mip,
                  .mip = mip ? MipFilter::Linear : MipFilter::None,
                  .min_lod_fixed8 = implicit_mip ? 128u : 0u});
    const uint64_t base = kBase + (mip && !gfx9 ? 256 : 0);
    const uint32_t pitch = mip && arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
    for (uint32_t y = 0; y < 2; ++y)
      memory_.write32(base + y * pitch, 64 | ((mip ? 64u : 192u) << 8));
    if (mip)
      memory_.write32(kBase + (gfx9 ? 512 : 0), 192);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    ASSERT_TRUE(
        sample(mip && !implicit_mip ? SampleTestOperation::Lod : SampleTestOperation::Zero));
    EXPECT_NEAR(std::bit_cast<float>(cu_->read_vgpr(vb_ + 8, 0)), expected, 0.0025f);
  }
}

TEST_P(ImageSamplerControlTest, D16FloatFilteringMatchesGfx11AndGfx12Captures) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "D16 filtering captured on gfx1100 and gfx1201";
  // R32_FLOAT, blend exactly halfway between 1 and each high texel.
  // Physical gfx1100 and gfx1201 agree for spatial and mip filtering.
  // In particular 1 + 3/4096 returns 0x3c00, distinguishing truncation
  // from round-to-nearest-even (0x3c01).
  struct Case {
    float high;
    uint16_t expected;
  };
  constexpr Case cases[] = {
      {0x1.004p+0f, 0x3c00}, {0x1.006p+0f, 0x3c00}, {0x1.008p+0f, 0x3c01}, {0x1.00cp+0f, 0x3c01}};
  const uint32_t row_pitch = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
  wf_->set_exec(1);
  write_coordinates(0, {std::bit_cast<uint32_t>(0.5f), std::bit_cast<uint32_t>(0.5f),
                        std::bit_cast<uint32_t>(0.5f)});
  for (bool mip : {false, true}) {
    Image image;
    image.width = image.height = 2;
    image.pitch = row_pitch / sizeof(float);
    image.mip_levels = mip ? 2 : 1;
    image.last_level = mip ? 1 : 0;
    bind_sample_image(image);
    bind_sampler({.linear_xy = !mip, .mip = mip ? MipFilter::Linear : MipFilter::None});
    for (const Case &test : cases) {
      SCOPED_TRACE(testing::Message() << mip << ':' << test.high);
      const uint64_t base_level = kBase + (mip ? 256 : 0);
      for (uint32_t row = 0; row < 2; ++row) {
        memory_.write32(base_level + row * row_pitch, std::bit_cast<uint32_t>(1.0f));
        memory_.write32(base_level + row * row_pitch + 4,
                        std::bit_cast<uint32_t>(mip ? 1.0f : test.high));
      }
      if (mip)
        memory_.write32(kBase, std::bit_cast<uint32_t>(test.high));
      cu_->l1_vector().invalidate_all();
      l2_.invalidate_all();
      cu_->write_vgpr(vb_ + 8, 0, 0xabcdef01);
      ASSERT_TRUE(sample(mip ? SampleTestOperation::Lod : SampleTestOperation::Zero, true));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0) & 0xffffu, test.expected);
    }
  }
}

TEST_P(ImageSamplerControlTest, CornerAndPerformanceControlsQualifyTheSample) {
  struct Case {
    const char *name;
    Image image;
    Sampler sampler;
    bool accepted;
  };
  const Case cases[] = {
      {"CORNER_SAMPLES", {.corner_samples = true}, {}, false},
      {"PERF_MOD", {.perf_mod = 1}, {}, true},
      // GFX12 splits PERF_MIP over words 2 and 3. This value sets bits in both.
      {"PERF_MIP", {}, {.perf_mip = 11}, true},
      {"PERF_MOD + PERF_MIP", {.perf_mod = 1}, {.perf_mip = 8}, false},
      {"Volume PERF_Z", {.depth = 2, .type = amdgpu::kImage3d}, {.perf_z = 15}, true},
      {"Volume PERF_MOD + PERF_Z",
       {.depth = 2, .type = amdgpu::kImage3d, .perf_mod = 1},
       {.perf_z = 15},
       false},
  };
  constexpr uint32_t texel = std::bit_cast<uint32_t>(0.75f);
  constexpr uint32_t sentinel = 0xabcdef01;
  memory_.write32(kBase, texel);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  for (const Case &test : cases) {
    if (test.image.corner_samples && arch_is_cdna_4_or_lower(arch()))
      continue; // GFX9 has no CORNER_SAMPLES.
    SCOPED_TRACE(test.name);
    dimension_override_ =
        test.image.type == amdgpu::kImage3d ? amdgpu::kImageDim3d : amdgpu::kImageDim2d;
    bind_sample_image(test.image);
    bind_sampler(test.sampler);
    cu_->write_vgpr(vb_ + 8, 0, sentinel);
    wf_->clear_instruction_execution_error();
    EXPECT_EQ(sample(SampleTestOperation::Zero), test.accepted);
    if (!test.accepted) {
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
    }
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), test.accepted ? texel : sentinel);
  }
}

TEST_P(ImageSamplerControlTest, ScaledViewsStopBeforeSampling) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  if (!gfx12 && arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP() << "GFX11 and GFX12 expose separate resource scales";
  Image image;
  image.type = amdgpu::kImage3d;
  image.depth = 2;
  dimension_override_ = amdgpu::kImageDim3d;
  wf_->set_exec(1);
  for (uint32_t bit_index = gfx12 ? 4 : 8; bit_index <= 19; ++bit_index) {
    SCOPED_TRACE(bit_index);
    bind_sample_image(image);
    bind_sampler();
    cu_->write_sgpr(sb_ + 5, cu_->read_sgpr(sb_ + 5) | (1u << bit_index));
    for (SampleTestOperation operation :
         {SampleTestOperation::Zero, SampleTestOperation::Lod, SampleTestOperation::Gradients}) {
      SCOPED_TRACE(static_cast<uint32_t>(operation));
      memory_.write32(kBase, 0x12345678);
      cu_->write_vgpr(vb_ + 8, 0, 0xabcdef01);
      wf_->clear_instruction_execution_error();
      EXPECT_FALSE(sample(operation));
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
      EXPECT_EQ(memory_.read32(kBase), 0x12345678u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcdef01u);
      EXPECT_TRUE(wf_->wait_counters().empty());
    }
  }
}

TEST_P(ImageSamplerControlTest, TiledSamplerBaseXorIsRejected) {
  Image image;
  image.width = image.height = 64;
  image.swizzle_mode = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 2 : 22;
  bind_sample_image(image);
  bind_sampler();
  cu_->write_sgpr(sb_, (kBase + 256) >> 8);
  wf_->set_exec(1);
  EXPECT_FALSE(sample(SampleTestOperation::Zero));
  EXPECT_EQ(wf_->instruction_execution_error(),
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
      // This fractional LOD distinguishes slope 129 from 128 in the fourth segment.
      {{0x1.955d52p-5, 0x1.f7ce2ap-5, 0},
       {0x1.f82884p-5, -0x1.312724p-4, 0},
       {37, 19, 1},
       2,
       0x1.91p+0},
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
  image.last_level = image.mip_levels - 1;
  bind_sample_image(image);
  // Independent linear mip-atlas offsets for a 37x19 R32 allocation.
  // Storage rounds mip extents upward; the accessible view still rounds down.
  const std::array<uint32_t, 6> offsets =
      arch_is_cdna_4_or_lower(arch()) ? std::array<uint32_t, 6>{0, 4864, 7424, 8704, 9472, 9984}
                                      : std::array<uint32_t, 6>{5376, 2816, 1536, 768, 256, 0};
  const uint32_t pitch = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
  for (uint32_t mip = 0; mip < 6; ++mip)
    for (uint32_t y = 0; y < std::max(1u, 19u >> mip); ++y)
      for (uint32_t x = 0; x < std::max(1u, 37u >> mip); ++x)
        memory_.write32(kBase + offsets[mip] + y * pitch + x * 4,
                        std::bit_cast<uint32_t>(float(mip)));
  struct Case {
    std::array<float, 4> gradients;
    float expected;
    bool linear_mip = false;
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
      {{0x1.955d52p-5f, 0x1.f7ce2ap-5f, 0x1.f82884p-5f, -0x1.312724p-4f}, 0x1.91p+0f, true},
  };
  wf_->set_exec(1);
  for (const auto &test : cases) {
    // Constant-valued mips and linear blending expose the captured fractional LOD.
    bind_sampler(
        {.mip = test.linear_mip ? MipFilter::Linear : MipFilter::Point, .max_lod_fixed8 = 5 * 256});
    for (uint32_t word = 0; word < 4; ++word)
      cu_->write_vgpr(vb_ + word, 0, std::bit_cast<uint32_t>(test.gradients[word]));
    cu_->write_vgpr(vb_ + 4, 0, std::bit_cast<uint32_t>(0.5f));
    cu_->write_vgpr(vb_ + 5, 0, std::bit_cast<uint32_t>(0.5f));
    ASSERT_TRUE(sample(SampleTestOperation::Gradients));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), std::bit_cast<uint32_t>(test.expected));
  }
}

TEST_P(ImageSamplerControlTest, ExplicitLodRoundsBeforeBiasAndSaturatesInfiniteInputs) {
  if (arch() == ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP() << "The public CDNA2 ISA and installed SDK omit explicit-LOD forms";
  Image image;
  image.width = image.height = 16;
  image.mip_levels = 4;
  image.last_level = image.mip_levels - 1;
  bind_sample_image(image);
  const bool gfx9 = arch_is_cdna_4_or_lower(arch());
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  // Independent linear mip-atlas offsets; each mip contains its level number.
  const std::array<uint32_t, 4> offsets = gfx9 ? std::array<uint32_t, 4>{0, 4096, 6144, 7168}
                                               : std::array<uint32_t, 4>{3584, 1536, 512, 0};
  for (uint32_t level = 0; level < 4; ++level)
    for (uint32_t y = 0; y < (16u >> level); ++y)
      for (uint32_t x = 0; x < (16u >> level); ++x)
        memory_.write32(kBase + offsets[level] + y * (gfx12 ? 128 : 256) + x * 4,
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
  wf_->set_exec(1);
  for (const Case &test : cases)
    for (MipFilter mip_filter : {MipFilter::Point, MipFilter::Linear}) {
      SCOPED_TRACE(testing::Message() << test.zero << ',' << test.lod << ',' << test.bias << ','
                                      << test.minimum << ',' << static_cast<uint32_t>(mip_filter));
      bind_sampler({.mip = mip_filter, .min_lod_fixed8 = test.minimum, .bias_fixed8 = test.bias});
      cu_->write_vgpr(vb_, 0, std::bit_cast<uint32_t>(0.5f));
      cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(0.5f));
      cu_->write_vgpr(vb_ + 2, 0, std::bit_cast<uint32_t>(test.lod));
      ASSERT_TRUE(sample(test.zero ? SampleTestOperation::Zero : SampleTestOperation::Lod));
      const float expected =
          mip_filter == MipFilter::Linear ? test.expected : std::floor(test.expected + 0.5f);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), std::bit_cast<uint32_t>(expected));
    }
  // Independent GFX11/12 captures distinguish integer-LOD mip blending.
  for (uint32_t level = 0; level < 4; ++level)
    for (uint32_t y = 0; y < (16u >> level); ++y)
      for (uint32_t x = 0; x < (16u >> level); ++x)
        memory_.write32(kBase + offsets[level] + y * (gfx12 ? 128 : 256) + x * 4, 0x80000000u);
  for (bool linear : {false, true})
    for (MipFilter mip_filter : {MipFilter::Point, MipFilter::Linear})
      for (float lod : {0.0f, 0.25f, 0.5f, 1.0f, 1.5f, 2.75f, 3.0f}) {
        bind_sampler({.linear_xy = linear, .mip = mip_filter});
        cu_->write_vgpr(vb_, 0, 0u);
        cu_->write_vgpr(vb_ + 1, 0, 0u);
        cu_->write_vgpr(vb_ + 2, 0, std::bit_cast<uint32_t>(lod));
        ASSERT_TRUE(sample(SampleTestOperation::Lod));
        const bool blends = mip_filter == MipFilter::Linear && lod != std::floor(lod);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), blends ? 0u : 0x80000000u);
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
  for (bool linear : {false, true})
    for (const BlendCase &test : blends) {
      SCOPED_TRACE(testing::Message()
                   << test.exec << ',' << test.fractional_lane << ',' << test.lod);
      wf_->set_exec(test.exec);
      bind_sampler({.linear_xy = linear, .mip = MipFilter::Linear});
      for (uint32_t lane = 0; lane < 8; ++lane) {
        cu_->write_vgpr(vb_, lane, 0u);
        cu_->write_vgpr(vb_ + 1, lane, 0u);
        cu_->write_vgpr(vb_ + 2, lane,
                        std::bit_cast<uint32_t>(lane == test.fractional_lane ? test.lod : 1.0f));
        cu_->write_vgpr(vb_ + 8, lane, 0xA5A5A5A5u);
      }
      ASSERT_TRUE(sample(SampleTestOperation::Lod));
      for (uint32_t lane = 0; lane < 8; ++lane) {
        const uint64_t bit = uint64_t{1} << lane;
        const uint64_t positive = gfx12 ? test.gfx12_positive : test.other_positive;
        const uint32_t expected = !(test.exec & bit) ? 0xA5A5A5A5u
                                  : positive & bit   ? 0u
                                                     : 0x80000000u;
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, lane), expected);
      }
    }
}

TEST_P(ImageSamplerControlTest, NullGatherClearsEveryPackedAndUnpackedDestination) {
  if (arch() == ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP() << "The public CDNA2 ISA and installed SDK omit gather forms";
  wf_->set_exec(5);
  for (bool d16 : {false, true})
    for (uint32_t invalid : {0u, 1u, 2u}) {
      SCOPED_TRACE(testing::Message() << d16 << ',' << invalid);
      Image image;
      image.width = 2;
      bind_sample_image(image, false);
      memory_.write32(kBase, 0x11);
      memory_.write32(kBase + 4, 0x33);
      if (invalid == 0)
        for (uint32_t word = 0; word < 8; ++word)
          cu_->write_sgpr(sb_ + word, 0);
      if (invalid == 1) {
        cu_->write_sgpr(sb_ + 1, 0);
      }
      bind_sampler();
      for (uint32_t lane : {0u, 1u, 2u}) {
        cu_->write_vgpr(vb_, lane, std::bit_cast<uint32_t>(0.5f));
        cu_->write_vgpr(vb_ + 1, lane, std::bit_cast<uint32_t>(0.5f));
        for (uint32_t reg = 0; reg < 4; ++reg)
          cu_->write_vgpr(vb_ + 8 + reg, lane, 0xdeadbeef);
      }
      ASSERT_TRUE(sample(SampleTestOperation::Gather, d16, invalid == 2 ? 104 : 8));
      for (uint32_t lane : {0u, 1u, 2u})
        for (uint32_t reg = 0; reg < 4; ++reg)
          EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + reg, lane),
                    lane != 1 && reg < (d16 ? 2u : 4u) ? 0u : 0xdeadbeefu);
    }
}

TEST_P(ImageSamplerControlTest, IntegerFilteringIsExplicitlyUnsupportedButPointSamplingWorks) {
  Image image;
  image.width = 2;
  bind_sample_image(image, false);
  memory_.write32(kBase, 1);
  memory_.write32(kBase + 4, 3);
  wf_->set_exec(1);
  cu_->write_vgpr(vb_, 0, std::bit_cast<uint32_t>(0.5f));
  cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(0.5f));
  for (Sampler sampler :
       {Sampler{}, Sampler{.linear_xy = true}, Sampler{.mip = MipFilter::Linear}}) {
    bind_sampler(sampler);
    wf_->clear_instruction_execution_error();
    if (!sampler.linear_xy && sampler.mip == MipFilter::None) {
      ASSERT_TRUE(sample(SampleTestOperation::Zero));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 3u);
    } else {
      EXPECT_FALSE(sample(SampleTestOperation::Zero));
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
    }
  }
}

TEST_P(ImageSamplerControlTest, ClampedLinearTapsPreserveNegativeZero) {
  Image image;
  image.width = 2;
  bind_sample_image(image);
  memory_.write32(kBase, 0x80000000u);
  memory_.write32(kBase + 4, 0x80000000u);
  wf_->set_exec(1);
  bind_sampler({.linear_xy = true, .normalized = false});
  for (float x : {0.0f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f}) {
    cu_->write_vgpr(vb_, 0, std::bit_cast<uint32_t>(x));
    cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(0.5f));
    ASSERT_TRUE(sample(SampleTestOperation::Zero));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), x > 0.5f && x < 1.5f ? 0u : 0x80000000u);
  }
  struct Case {
    uint32_t width;
    Clamp wrap;
    float x;
    uint32_t expected;
  };
  // Captured normalized mirror/repeat edge and distinct-texel controls.
  const Case cases[] = {{2, Clamp::Mirror, 0, 0x80000000u},
                        {4, Clamp::Mirror, 0.05f, 0x80000000u},
                        {4, Clamp::Mirror, 1, 0x80000000u},
                        {1, Clamp::Repeat, 0, 0x80000000u},
                        {1, Clamp::Repeat, -0.3f, 0x80000000u},
                        {1, Clamp::Mirror, 1.3f, 0x80000000u},
                        {2, Clamp::Repeat, 0, 0},
                        {2, Clamp::Mirror, 0.5f, 0},
                        {2, Clamp::Border, 0, 0}};
  for (const Case &test : cases) {
    SCOPED_TRACE(testing::Message()
                 << test.width << ',' << static_cast<uint32_t>(test.wrap) << ',' << test.x);
    image.width = test.width;
    bind_sample_image(image);
    for (uint32_t x = 0; x < image.width; ++x)
      memory_.write32(kBase + x * 4, 0x80000000u);
    bind_sampler({.clamp_x = test.wrap, .linear_xy = true});
    cu_->write_vgpr(vb_, 0, std::bit_cast<uint32_t>(test.x));
    cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(0.5f));
    ASSERT_TRUE(sample(SampleTestOperation::Zero));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), test.expected);
  }
}

TEST_P(ImageSamplerControlTest, UnsupportedSampleFormsReportAnExecutionError) {
  if (arch() == ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP() << "The public CDNA2 ISA exposes only the base sample form";
  wf_->clear_instruction_execution_error();
  EXPECT_FALSE(sample(SampleTestOperation::Unsupported));
  EXPECT_EQ(wf_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
}

TEST_P(ImageSamplerControlTest, DerivativeMetadataUsesTwoCubeComponents) {
  if (arch_is_cdna_4_or_lower(arch()) || arch() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "MIMG DIM encoding on GFX10/GFX11";
  for (uint32_t dim : {1u, 2u, 3u}) {
    dimension_override_ = dim;
    std::unique_ptr<Instruction> instruction = sample_instruction(SampleTestOperation::Gradients);
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
  std::unique_ptr<Decoder> decoder = Decoder::create(arch());
  DecodeResult decoded = decoder->decode(words.data());
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
