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
    cu_ = amdgpu::ComputeUnitCore::create("image_cu", cfg, &memory_, &l2_);
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

TEST_F(Cdna2ImageAccessTest, DecodedMipLoadAndStoreReportUnimplemented) {
  Image image;
  image.width = 64;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {1, 1, 0});
  constexpr uint64_t kAddress = kBase + (64 + 1) * sizeof(uint32_t);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_NE(decoder, nullptr);

  for (const bool load : {false, true}) {
    SCOPED_TRACE(testing::Message() << "load=" << load);
    memory_.write32(kAddress, 0x12345678u);
    cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeefu);
    const auto words =
        cdna2::build_mimg(load ? cdna2::kImageLoadMipMimg : cdna2::kImageStoreMipMimg,
                          {.dmask = 1, .unorm = 1, .da = 1, .vaddr = 0, .vdata = 8, .srsrc = 0});
    auto decoded = decoder->decode(words.data());
    ASSERT_TRUE(decoded.succeeded());
    std::unique_ptr<Instruction> inst = std::move(decoded).value();
    ASSERT_NE(inst, nullptr);

    EXPECT_TRUE(cu_->execute_instruction(inst.get(), *wf_).failed());
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(inst->data(), nullptr);
    EXPECT_EQ(memory_.read32(kAddress), 0x12345678u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xdeadbeefu);
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
  auto &state = wf_->ensure_memory_wait_scoreboard();
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
  auto &state = wf_->ensure_memory_wait_scoreboard();
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

TEST_F(Cdna2ImageAccessTest, ImageOnlySrgbFormatStopsAsUnimplemented) {
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = 9; // GFX9 TYPE_SRGB, outside the buffer NFMT table.
  bind(image);
  EXPECT_FALSE(issue(true));
  EXPECT_EQ(wf_->instruction_execution_error(),
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

TEST(ImageAccessTest, Rdna3MipAndAtomicUseTheBaseAddressPrinter) {
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
           Case{rdna3::kImageLoadMipMimg, "image_load_mip v8, v[2:4], s[8:15]"},
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

TEST_P(RdnaDecodedImageAccessTest, MipLoadAndStoreReportUnimplemented) {
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

    EXPECT_TRUE(cu_->execute_instruction(inst.get(), *wf_).failed());
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(inst->data(), nullptr);
    EXPECT_EQ(memory_.read32(kAddress), 0x12345678u);
    EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), 0xdeadbeefu);
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
  const bool max_level_in_word5 =
      arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
  set(gfx12 ? 44 : max_level_in_word5 ? 164 : 48, gfx12 ? 5 : 4, image.mip_levels - 1);
  set(116, 5, image.swizzle_mode);
  if (image.type == amdgpu::kImage3d || image.type == amdgpu::kImage1dArray ||
      image.type == amdgpu::kImage2dArray)
    set(128, 13, image.depth - 1);
  else if (arch != ROCJITSU_CODE_ARCH_RDNA1)
    set(128, 14, 63); // A 256-byte row pitch.
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
    encoding.dmask = 1;
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
  bool decoded_issue(bool load, uint32_t dimensions) {
    std::unique_ptr<Instruction> inst = decoded_instruction(load, dimensions);
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

TEST_P(AllImageAccessTest, Gfx12ViewControlsStopBeforeImageMemoryAccess) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "GFX12 has separate UAV3D and DEPTH_SCALE controls";
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  // Word 5 bit 4 is UAV3D; bits 5-9 are all five DEPTH_SCALE bits.
  // Exercise each independently, including the scale bits above bit 7.
  for (uint32_t bit_index = 4; bit_index <= 9; ++bit_index) {
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
    image.level = level;
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

TEST_P(AllImageAccessTest, UnsupportedExplicitMipInstructionsReportAnExecutionError) {
  bind_image(2);
  for (bool load : {false, true}) {
    wf_->clear_instruction_execution_error();
    std::unique_ptr<Instruction> instruction = decoded_instruction(load, 2, true);
    ASSERT_TRUE(instruction);
    instruction->execute(*instruction, wf_);
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
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

} // namespace
