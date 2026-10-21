// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/mimg.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_resource.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
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
      const auto counter = WaitCounterKind::Ds;
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

} // namespace
