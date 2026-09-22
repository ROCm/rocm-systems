// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/graphics_draw.h"
#include "rocjitsu/vm/amdgpu/graphics_stage.h"
#include "rocjitsu/vm/amdgpu/image_address.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/raster_math.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

using namespace rocjitsu;

namespace {

class ExportCollector final : public amdgpu::GraphicsStage {
public:
  struct Export {
    uint32_t lane, target, mask;
    std::array<uint32_t, 4> values;
  };
  std::vector<Export> exports;
  void initialize(amdgpu::Wavefront &, uint32_t, uint32_t) override {}
  void export_lane(amdgpu::Wavefront &, uint32_t lane, uint32_t target, uint32_t mask,
                   const std::array<uint32_t, 4> &values) override {
    exports.push_back({lane, target, mask, values});
  }
};

class GraphicsExportTest : public testing::TestWithParam<rj_code_arch_t> {
protected:
  amdgpu::GpuMemory memory_{"graphics_memory"};
  amdgpu::L2Cache cache_{"graphics_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  std::unique_ptr<Decoder> decoder_;
  amdgpu::Wavefront *wave_ = nullptr;
  std::shared_ptr<ExportCollector> collector_ = std::make_shared<ExportCollector>();
  amdgpu::Lds lds_{4};

  void SetUp() override {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = GetParam();
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    cache_.set_backing_memory(&memory_);
    cu_ = amdgpu::ComputeUnitCore::create("graphics_export", config, &memory_, &cache_);
    decoder_ = Decoder::create(GetParam());
    wave_ = cu_->dispatch_wf(0, 0, 106, 16);
    wave_->set_graphics_stage(collector_);
    wave_->set_exec(3);
    wave_->set_lds(&lds_);
  }
  void TearDown() override { wave_->halt(); }

  template <size_t N> void run(const std::array<uint32_t, N> &words) {
    std::array<uint32_t, 4> padded{};
    std::copy(words.begin(), words.end(), padded.begin());
    auto decoded = decoder_->decode(padded.data());
    ASSERT_FALSE(decoded.failed());
    std::unique_ptr<Instruction> instruction(std::move(decoded).value());
    (void)cu_->execute_instruction(instruction.get(), *wave_);
  }

  void execute(uint8_t target, uint8_t mask, bool row = false) {
    std::array<uint32_t, 2> words;
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
      words = rdna4::build_vexport({.en = mask,
                                    .tgt = target,
                                    .done = 1,
                                    .row_en = uint8_t(row),
                                    .vsrc0 = 3,
                                    .vsrc1 = 7,
                                    .vsrc2 = 9,
                                    .vsrc3 = 255});
    else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
      words = rdna3_5::build_exp({.en = mask,
                                  .tgt = target,
                                  .done = 1,
                                  .row_en = uint8_t(row),
                                  .vsrc0 = 3,
                                  .vsrc1 = 7,
                                  .vsrc2 = 9,
                                  .vsrc3 = 255});
    else
      words = rdna3::build_exp({.en = mask,
                                .tgt = target,
                                .done = 1,
                                .row_en = uint8_t(row),
                                .vsrc0 = 3,
                                .vsrc1 = 7,
                                .vsrc2 = 9,
                                .vsrc3 = 255});
    run(words);
  }
};

TEST_P(GraphicsExportTest, HalfInterpolationResultsHonorRoundingMode) {
  constexpr uint16_t positive[] = {0x3c01, 0x3c01, 0x3c00, 0x3c00};
  constexpr uint16_t negative[] = {0xbc01, 0xbc00, 0xbc01, 0xbc00};
  wave_->set_exec(0x7fffffff);
  for (uint32_t mode = 0; mode < 4; ++mode) {
    wave_->set_mode_raw(mode << 2);
    for (bool high : {false, true}) {
      for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
        // Halfway between the half-precision midpoint and its upper neighbor.
        wave_->debug_write_vgpr(3, lane, 0x3f801800u | ((lane & 1) << 31));
        wave_->debug_write_vgpr(6, lane, 0x12345678);
      }
      const uint8_t opcode = high ? 34 : 33;
      if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
        run(rdna4::build_vop3p(opcode, {.vdst = 6, .src0 = 259, .src1 = 242, .src2 = 128}));
      else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
        run(rdna3_5::build_vop3p(opcode, {.vdst = 6, .src0 = 259, .src1 = 242, .src2 = 128}));
      else
        run(rdna3::build_vop3p(opcode, {.vdst = 6, .src0 = 259, .src1 = 242, .src2 = 128}));
      for (uint32_t lane = 0; lane < 31; ++lane) {
        const uint32_t half = lane & 1 ? negative[mode] : positive[mode];
        EXPECT_EQ(cu_->read_vgpr(wave_->vgpr_alloc().base + 6, lane),
                  high ? (half << 16) | 0x5678 : 0x12340000 | half)
            << "mode=" << mode << " high=" << high << " lane=" << lane;
      }
      EXPECT_EQ(cu_->read_vgpr(wave_->vgpr_alloc().base + 6, 31), 0x12345678u);
    }
  }
}

TEST(GraphicsRasterMathTest, InterpolationMatchesPhysicalRdna4QuadInputs) {
  // Raw float bits captured with GL_AMD_shader_explicit_vertex_parameter.
  // Both ordinary and near-edge quads exercise opposite gradient signs and
  // cancellation during the shared-exponent addition of two pixel offsets.
  struct Case {
    double area, edge_x, edge_y, x, y;
    std::array<uint32_t, 4> expected;
  };
  const Case cases[] = {
      {33600, -160, 0, -2.5, -149.5, {0x3c430c30, 0x3bea0ea0, 0x3c430c30, 0x3bea0ea0}},
      {33600, 0, -210, -2.5, -149.5, {0x3f6f3332, 0x3f6f3332, 0x3f6d9999, 0x3f6d9999}},
      {31500, 40, -210, 119.5, -29.5, {0x3eb26324, 0x3eb30994, 0x3eaef954, 0x3eaf9fc4}},
      {31500, 120, 157.5, 119.5, -29.5, {0x3e9d8fd7, 0x3e9f8329, 0x3ea01f33, 0x3ea21285}},
      {31500, 40, -210, 5.5, -1.5, {0x3c8b224a, 0x3c958956, 0x3c290a8e, 0x3c3dd8a6}},
      {31500, 120, 157.5, 5.5, -1.5, {0x3c5c675e, 0x3c8d68d5, 0x3c972971, 0x3cb65e97}},
      {29400, 160, -52.5, 115.5, 80.5, {0x3ef83a82, 0x3efb03d3, 0x3ef75074, 0x3efa19c5}},
      {29400, -80, 210, 115.5, 80.5, {0x3e857c56, 0x3e8417ae, 0x3e892490, 0x3e87bfe8}},
  };
  for (const auto &test : cases) {
    const float inverse_area = amdgpu::raster::truncate_float(1.0 / test.area);
    const amdgpu::raster::Plane plane{amdgpu::raster::truncate_float(test.edge_x * inverse_area),
                                      amdgpu::raster::truncate_float(test.edge_y * inverse_area)};
    for (uint32_t lane = 0; lane < 4; ++lane)
      EXPECT_EQ(std::bit_cast<uint32_t>(plane.at_quad(test.x, test.y, lane)), test.expected[lane]);
  }
}

TEST(GraphicsRasterMathTest, SubpixelQuantizationRoundsMidpointsToEven) {
  for (double value : {0.0, 17.0, -17.0}) {
    EXPECT_EQ(amdgpu::raster::round_subpixel(value + 0.5 / 256), value);
    EXPECT_EQ(amdgpu::raster::round_subpixel(value + 1.5 / 256), value + 2.0 / 256);
  }
}

TEST(GraphicsRasterMathTest, PerspectiveProductsMatchPhysicalRdna3AndRdna4) {
  // Raw barycentric captures with power-of-two plane gradients isolate the
  // interpolation multiplier from triangle setup and parameter interpolation.
  constexpr std::array<std::array<uint32_t, 3>, 4> cases{{
      {0x3d900000, 0x3f8b7034, 0x3d9cde3b},
      {0x3e580000, 0x3fa4a9cf, 0x3e8aef46},
      {0x3e600000, 0x3fa655c4, 0x3e918b0b},
      {0x3e400000, 0x3fa237c3, 0x3e7353a5},
  }};
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(amdgpu::raster::multiply_perspective(
                  std::bit_cast<float>(test[0]), std::bit_cast<float>(test[1]))),
              test[2]);
}

TEST_P(GraphicsExportTest, ParameterLoadUsesQuadMaskAndPrimitiveOffsets) {
  // Quad two starts a second primitive; attribute one follows both records of attr0.
  wave_->set_m0(128 | (1u << 17));
  wave_->set_exec((1u << 1) | (1u << 9));
  for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
    wave_->debug_write_vgpr(3, lane, 0xdeadbeef);
  for (uint32_t primitive = 0; primitive < 2; ++primitive)
    for (uint32_t coefficient = 0; coefficient < 3; ++coefficient)
      lds_.write32(128 + 96 + primitive * 48 + 24 + coefficient * 4,
                   100 + primitive * 10 + coefficient);
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    run(rdna4::build_vdsdir(0, {.vdst = 3, .attr_chan = 2, .attr = 1}));
  else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
    run(rdna3_5::build_ldsdir(0, {.vdst = 3, .attr_chan = 2, .attr = 1}));
  else
    run(rdna3::build_ldsdir(0, {.vdst = 3, .attr_chan = 2, .attr = 1}));
  EXPECT_FALSE(wave_->instruction_execution_failed());
  for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
    uint32_t expected = 0xdeadbeef;
    if (lane < 3)
      expected = 100 + lane;
    else if (lane >= 8 && lane < 11)
      expected = 110 + lane - 8;
    if (lane % 4 != 3) // The unused fourth coefficient is unspecified.
      EXPECT_EQ(wave_->debug_read_vgpr(3, lane), expected) << lane;
  }
}

TEST_P(GraphicsExportTest, RectangleCoverageAndUnsupportedRasterStates) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  enum class Outcome { Draw, Empty, Reject };
  struct Case {
    const char *name;
    Outcome outcome = Outcome::Draw;
    uint32_t clip_control = 0;
    uint32_t shader_control = 0;
    uint32_t color_control = 0xcc0010;
    uint32_t polygon_mode = 0;
    uint32_t samples = 0;
    std::array<float, 4> depth_bias{};
    bool depth_only = false;
    float depth = 0;
    bool first_vertex_depth_only = false;
    float extent = 1;
    bool full_scissor = false;
  };
  const Case cases[] = {
      {.name = "integer coverage"},
      {.name = "fractional coverage", .extent = 0.625f, .full_scissor = true},
      {.name = "rasterizer discard", .outcome = Outcome::Empty, .clip_control = 1u << 22},
      {.name = "fragment discard", .outcome = Outcome::Reject, .shader_control = 1u << 6},
      {.name = "fully clipped", .outcome = Outcome::Empty, .depth = 2},
      {.name = "partially clipped",
       .outcome = Outcome::Reject,
       .depth = 2,
       .first_vertex_depth_only = true},
      {.name = "logic op clear", .outcome = Outcome::Reject, .color_control = 0x10},
      {.name = "color disabled", .color_control = 0, .depth_only = true},
      {.name = "unsupported color mode", .outcome = Outcome::Reject, .color_control = 0xcc0020},
      {.name = "color degamma", .outcome = Outcome::Reject, .color_control = 0xcc0018},
      {.name = "line polygons",
       .outcome = Outcome::Reject,
       .polygon_mode = 8 | (1 << 5) | (1 << 8)},
      {.name = "point polygons", .outcome = Outcome::Reject, .polygon_mode = 8},
      {.name = "mixed polygon faces",
       .outcome = Outcome::Reject,
       .polygon_mode = 8 | (2 << 5) | (1 << 8)},
      {.name = "reserved polygon mode",
       .outcome = Outcome::Reject,
       .polygon_mode = 16 | (2 << 5) | (2 << 8)},
      {.name = "filled dual mode", .polygon_mode = 8 | (2 << 5) | (2 << 8)},
      {.name = "front depth bias",
       .outcome = Outcome::Reject,
       .polygon_mode = 1u << 11,
       .depth_bias = {1, 0, 0, 0}},
      {.name = "back depth bias",
       .outcome = Outcome::Reject,
       .polygon_mode = 1u << 12,
       .depth_bias = {0, 0, 1, 0}},
      {.name = "parallel depth bias",
       .outcome = Outcome::Reject,
       .polygon_mode = 1u << 13,
       .depth_bias = {0, 1, 0, 1}},
      {.name = "four samples", .outcome = Outcome::Reject, .samples = 2},
      {.name = "zero depth bias", .polygon_mode = 7u << 11, .depth_bias = {-0.0f, 0, 0, 0}},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.name);
    for (uint32_t y = 0; y < 4; ++y)
      for (uint32_t x = 0; x < 4; ++x) {
        const auto address = gfx12 ? amdgpu::gfx12_image_offset(x, y, 4, 4, 3)
                                   : amdgpu::gfx11_image_offset(x, y, 4, 4, 26);
        memory_.write32(0x100000 + *address, 0);
      }
    amdgpu::Pm4QueueState state;
    state.num_instances = 1;
    state.uconfig_registers[0x242] = 0x11;
    state.context_registers[0x3b0] = 10;
    state.context_registers[0x31e] = (3 << 16) | 3;
    state.context_registers[0x31f] = 3 << 15;
    state.context_registers[0x318] = 0x1000;
    state.context_registers[0x214] = 15;
    state.context_registers[0x195] = 4;
    state.context_registers[0x198] = 2;
    state.context_registers[0x2f9] = 0x2d;
    state.context_registers[0x205] = 0x43f;
    state.context_registers[0x10f] = state.context_registers[0x110] =
        state.context_registers[0x111] = state.context_registers[0x112] =
            std::bit_cast<uint32_t>(2.0f);
    state.context_registers[0x90] = 1 | (1 << 16);
    state.context_registers[0x91] = 3 | (3 << 16);
    if (!gfx12) {
      state.context_registers[0x31c] = 10;
      state.context_registers[0x3b0] = 3 | (3 << 14);
      state.context_registers[0x3b8] = 26 << 14;
      state.context_registers[0x31e] = 0;
      state.context_registers[0x8e] = 15;
      state.context_registers[0x1c5] = 4;
      state.context_registers[0x1b4] = 2;
      state.context_registers[0x206] = 0x43f;
      state.context_registers[0x205] = 0;
    }
    if (test.full_scissor) {
      state.context_registers[0x90] = 0;
      state.context_registers[0x91] = 4 | (4 << 16);
    }
    state.context_registers[0x204] = test.clip_control;
    state.context_registers[gfx12 ? 0x1b : 0x203] = test.shader_control;
    state.context_registers[gfx12 ? 0x216 : 0x202] = test.color_control;
    state.context_registers[gfx12 ? 0x207 : 0x205] = test.polygon_mode;
    state.context_registers[0x2f8] = test.samples;
    for (uint32_t i = 0; i < test.depth_bias.size(); ++i)
      state.context_registers[0x2e0 + i] = std::bit_cast<uint32_t>(test.depth_bias[i]);
    if (test.depth_only) {
      state.context_registers[gfx12 ? 0x1c : 0x200] = 6 | (7 << 4);
      state.context_registers[gfx12 ? 5 : 7] = (3 << 16) | 3;
      state.context_registers[gfx12 ? 6 : 0x10] = 3 | ((gfx12 ? 3 : 24) << 4);
      state.context_registers[gfx12 ? 8 : 0x12] = 0x2000;
      state.context_registers[gfx12 ? 10 : 0x14] = 0x2000;
    }
    const float extent = test.extent;
    auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
    for (uint32_t i = 0; i < 3; ++i) {
      const std::array<uint32_t, 4> position{
          std::bit_cast<uint32_t>(i == 2 ? extent : -extent),
          std::bit_cast<uint32_t>(i == 1 ? extent : -extent),
          std::bit_cast<uint32_t>(!test.first_vertex_depth_only || i == 0 ? test.depth : 0.0f),
          std::bit_cast<uint32_t>(1.0f)};
      draw->export_lane(*wave_, i, 12, 15, position);
    }
    draw->export_lane(*wave_, 0, 20, 1,
                      {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
    if (test.outcome == Outcome::Reject) {
      EXPECT_THROW(draw->advance(memory_, 0), std::runtime_error);
      continue;
    }
    if (test.outcome == Outcome::Empty) {
      EXPECT_FALSE(draw->advance(memory_, 0));
      EXPECT_EQ(memory_.read32(0x100000), 0);
      continue;
    }
    const auto dispatch = draw->advance(memory_, 0);
    ASSERT_TRUE(dispatch);
    ASSERT_EQ(dispatch->total_wgs, 1);
    wave_->set_wg_coord(0, 0, 0);
    wave_->set_graphics_stage(draw);
    draw->initialize(*wave_, 0, 0);
    EXPECT_EQ(std::popcount(wave_->exec()), 4);
    // Include helper lanes in exports; only the four covered fragments may write.
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
      draw->export_lane(*wave_, lane, 0, 3, {0x00003c00, 0x3c000000, 0, 0});
    EXPECT_FALSE(draw->advance(memory_, 0));
    for (uint32_t y = 0; y < 4; ++y)
      for (uint32_t x = 0; x < 4; ++x) {
        const auto address = gfx12 ? amdgpu::gfx12_image_offset(x, y, 4, 4, 3)
                                   : amdgpu::gfx11_image_offset(x, y, 4, 4, 26);
        ASSERT_TRUE(address);
        EXPECT_EQ(memory_.read32(0x100000 + *address),
                  !test.depth_only && x >= 1 && x < 3 && y >= 1 && y < 3 ? 0xff0000ffu : 0u)
            << x << "," << y;
      }
  }
}

TEST_P(GraphicsExportTest, IndexedDrawPreservesVertexIndicesAndLocalConnectivity) {
  amdgpu::Pm4QueueState state;
  state.num_instances = 1;
  state.uconfig_registers[0x242] = 4;
  amdgpu::GraphicsDraw draw(state, GetParam(), 3, {9, 4, 9});
  draw.initialize(*wave_, 0, 0);
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t vertex_id = gfx12 ? 3 : 5;
  EXPECT_EQ(wave_->debug_read_vgpr(vertex_id, 0), 9);
  EXPECT_EQ(wave_->debug_read_vgpr(vertex_id, 1), 4);
  EXPECT_EQ(wave_->debug_read_vgpr(vertex_id, 2), 9);
  const uint32_t bits = gfx12 ? 9 : 10;
  EXPECT_EQ(wave_->debug_read_vgpr(0, 0), (1u << bits) | (2u << (2 * bits)));
  state.uconfig_registers[0x24b] = 1;
  EXPECT_THROW(amdgpu::GraphicsDraw(state, GetParam(), 3, {9, 4, 9}), std::runtime_error);
}

TEST_P(GraphicsExportTest, MergedVertexUserCountExcludesSystemRingPair) {
  amdgpu::Pm4QueueState state;
  state.num_instances = 1;
  state.uconfig_registers[0x242] = 4;
  state.sh_registers[0x8b] = 6 << 1;
  for (uint32_t i = 0; i < 6; ++i)
    state.sh_registers[0x8c + i] = 100 + i;
  amdgpu::GraphicsDraw draw(state, GetParam(), 3);
  draw.initialize(*wave_, 0, 0);
  for (uint32_t i = 0; i < 6; ++i)
    EXPECT_EQ(wave_->debug_read_sgpr(8 + i), 100 + i);
}

TEST(GraphicsImageAddressTest, MatchesGfx12CoordinateBitsAndCrossesTileRows) {
  using amdgpu::gfx12_image_offset;
  EXPECT_EQ(gfx12_image_offset(1, 0, 256, 4, 3), 4);
  EXPECT_EQ(gfx12_image_offset(0, 1, 256, 4, 3), 8);
  EXPECT_EQ(gfx12_image_offset(8, 0, 256, 4, 3), 512);
  EXPECT_EQ(gfx12_image_offset(0, 8, 256, 4, 3), 256);
  EXPECT_EQ(gfx12_image_offset(128, 0, 256, 4, 3), 65536);
  EXPECT_EQ(gfx12_image_offset(0, 128, 256, 4, 3), 131072);
  EXPECT_EQ(gfx12_image_offset(2, 3, 64, 4, 0), 776);
  EXPECT_FALSE(gfx12_image_offset(64, 0, 64, 4, 3));
  EXPECT_FALSE(gfx12_image_offset(0, 0, 64, 3, 3));
  EXPECT_FALSE(gfx12_image_offset(0, 0, 64, 4, 5));
  EXPECT_EQ(amdgpu::gfx12_image_address(0x140100, 0, 8, 256, 4, 4), 0x140000);
  EXPECT_EQ(amdgpu::gfx12_image_address(0x140100, 255, 255, 256, 4, 4), 0x17fefc);
  EXPECT_EQ(amdgpu::gfx12_image_address(0x140100, 256, 0, 512, 4, 4), 0x180100);
}

TEST(GraphicsImageAddressTest, MatchesGfx11AddrLibWithPipeXorAndAlignedLinearPitch) {
  // AddrLib: GFX11, GB_ADDR_CONFIG=0x545, single-level 2D surfaces.
  struct Case {
    uint32_t swizzle, bytes, width, x, y;
    uint64_t offset;
  };
  constexpr Case cases[] = {
      {0, 4, 420, 419, 319, 573324},  {0, 4, 1024, 513, 777, 3184644},
      {2, 4, 420, 419, 319, 542652},  {2, 4, 1024, 513, 777, 3194892},
      {6, 4, 420, 419, 319, 570812},  {6, 4, 1024, 513, 777, 3211532},
      {10, 4, 420, 419, 319, 734652}, {10, 4, 1024, 513, 777, 3408140},
      {22, 4, 420, 419, 319, 571836}, {22, 4, 1024, 513, 777, 3211532},
      {24, 4, 420, 419, 319, 742332}, {24, 4, 1024, 513, 777, 3426316},
      {26, 4, 420, 419, 319, 730300}, {26, 4, 1024, 513, 777, 3408140},
      {27, 4, 420, 419, 319, 742332}, {27, 4, 1024, 513, 777, 3426316},
      {28, 4, 420, 419, 319, 873404}, {28, 4, 1024, 513, 777, 3721228},
      {30, 4, 420, 419, 319, 926908}, {30, 4, 1024, 513, 777, 3670284},
      {31, 4, 420, 419, 319, 873404}, {31, 4, 1024, 513, 777, 3721228},
  };
  for (const auto &c : cases)
    EXPECT_EQ(amdgpu::gfx11_image_offset(c.x, c.y, c.width, c.bytes, c.swizzle), c.offset)
        << c.swizzle << "," << c.x << "," << c.y;
  EXPECT_FALSE(amdgpu::gfx11_image_offset(0, 0, 64, 4, 18));
  EXPECT_FALSE(amdgpu::gfx11_image_offset(0, 0, 64, 16, 24));
  EXPECT_FALSE(amdgpu::gfx11_image_offset(64, 0, 64, 4, 26));
}

TEST_P(GraphicsExportTest, HardwareImageLoadReadsTiledUintChannelsAndPacksD16) {
  if (GetParam() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "RDNA4 image transfer encoding";
  const std::array<uint32_t, 8> descriptor{
      0x1000, (46u << 17) | (3u << 30), 3u << 14, (9u << 28) | (3u << 20) | 0xfac, 0, 0, 0, 0};
  for (uint32_t r = 0; r < descriptor.size(); ++r)
    wave_->debug_write_sgpr(8 + r, descriptor[r]);
  wave_->set_exec(5);
  for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
    wave_->debug_write_vgpr(0, lane, 0xdeadbeef);
    wave_->debug_write_vgpr(1, lane, 0xdeadbeef);
    wave_->debug_write_vgpr(2, lane, lane == 2 ? 4 : 1);
    wave_->debug_write_vgpr(3, lane, 2);
  }
  // Byte address for texel (1,2) in a 4-byte GFX12 64 KiB tile.
  memory_.write32(0x100000 + 36, 0x44332211);
  const std::array<uint32_t, 4> words{0xd3c00021, 0x00001000, 0x00000302, 0};
  auto decoded = decoder_->decode(words.data());
  ASSERT_FALSE(decoded.failed());
  auto instruction_owner = std::move(decoded).value();
  auto *instruction = instruction_owner.get();
  ASSERT_TRUE(instruction->is_memory_op());
  ASSERT_TRUE(cu_->execute_instruction(instruction, *wave_).succeeded());
  ASSERT_FALSE(wave_->instruction_execution_failed());
  ASSERT_NE(instruction->data(), nullptr);
  EXPECT_EQ(instruction->data_as<amdgpu::VectorMemState>()->wait_counter_type,
            amdgpu::WaitCounterType::LOADCNT);
  amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
  pipeline.issue(instruction_owner.release(), *wave_);
  EXPECT_EQ(wave_->debug_read_vgpr(0, 0), 0x00220011);
  EXPECT_EQ(wave_->debug_read_vgpr(1, 0), 0x00440033);
  EXPECT_EQ(wave_->debug_read_vgpr(0, 1), 0xdeadbeef);
  EXPECT_EQ(wave_->debug_read_vgpr(1, 1), 0xdeadbeef);
  EXPECT_EQ(wave_->debug_read_vgpr(0, 2), 0);
  EXPECT_EQ(wave_->debug_read_vgpr(1, 2), 0);
  wave_->debug_write_vgpr(0, 0, 0x00660055);
  wave_->debug_write_vgpr(1, 0, 0x00880077);
  const auto store = rdna4::build_vimage(
      6, {.dim = 1, .d16 = 1, .dmask = 15, .rsrc = 8, .vaddr0 = 2, .vaddr1 = 3});
  std::array<uint32_t, 4> store_words{};
  std::copy(store.begin(), store.end(), store_words.begin());
  auto decoded_store = decoder_->decode(store_words.data());
  ASSERT_FALSE(decoded_store.failed());
  auto store_instruction = std::move(decoded_store).value();
  ASSERT_TRUE(store_instruction->is_memory_op());
  ASSERT_TRUE(cu_->execute_instruction(store_instruction.get(), *wave_).succeeded());
  ASSERT_NE(store_instruction->data(), nullptr);
  EXPECT_EQ(store_instruction->data_as<amdgpu::VectorMemState>()->wait_counter_type,
            amdgpu::WaitCounterType::STORECNT);
  pipeline.issue(store_instruction.release(), *wave_);
  cu_->l1_vector().flush_all();
  cache_.flush_all();
  EXPECT_EQ(memory_.read32(0x100000 + 36), 0x88776655);
}

TEST_P(GraphicsExportTest, LinearImageLoadsRespectDefaultPitchAndArrayDescriptorFields) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    uint32_t width, word4, type, dim, offset;
  };
  const Case cases[] = {{2, 0, 9, 1, gfx12 ? 132u : 260u},
                        {2, 127, 9, 1, 516},
                        {128, 1, 13, 5, 516},
                        {128, 0, 13, 5, 516}};
  for (uint32_t i = 0; i < std::size(cases); ++i) {
    SCOPED_TRACE(i);
    const auto &c = cases[i];
    const uint32_t base = 0x100000 + i * 0x10000;
    const std::array<uint32_t, 8> descriptor{base >> 8,
                                             (46u << (gfx12 ? 17 : 20)) |
                                                 (((c.width - 1) & 3) << 30),
                                             ((c.width - 1) >> 2) | (1u << 14),
                                             (c.type << 28) | 0xfac,
                                             c.word4,
                                             0,
                                             0,
                                             0};
    for (uint32_t r = 0; r < descriptor.size(); ++r)
      wave_->debug_write_sgpr(8 + r, descriptor[r]);
    wave_->set_exec(1);
    wave_->debug_write_vgpr(2, 0, 1);
    wave_->debug_write_vgpr(3, 0, 1);
    wave_->debug_write_vgpr(4, 0, 0);
    memory_.write32(base + c.offset, 0x44332211);
    std::array<uint32_t, 4> words{};
    if (gfx12) {
      const auto inst = rdna4::build_vimage(0, {.dim = uint8_t(c.dim),
                                                .dmask = 15,
                                                .vdata = 8,
                                                .rsrc = 8,
                                                .vaddr0 = 2,
                                                .vaddr1 = 3,
                                                .vaddr2 = 4});
      std::copy(inst.begin(), inst.end(), words.begin());
    } else {
      const auto inst = rdna3::build_mimg(
          0, {.dim = uint8_t(c.dim), .dmask = 15, .vaddr = 2, .vdata = 8, .srsrc = 2});
      std::copy(inst.begin(), inst.end(), words.begin());
    }
    auto decoded = decoder_->decode(words.data());
    ASSERT_FALSE(decoded.failed());
    auto instruction = std::move(decoded).value();
    ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
    ASSERT_NE(instruction->data(), nullptr);
    EXPECT_EQ(instruction->data_as<amdgpu::VectorMemState>()->per_lane_addr[0], base + c.offset);
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
    pipeline.issue(instruction.release(), *wave_);
    for (uint32_t c = 0; c < 4; ++c)
      EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 0), 0x11u * (c + 1));
  }
}

TEST_P(GraphicsExportTest, UnsupportedComparisonSamplingReportsAnExecutionError) {
  std::array<uint32_t, 4> words{};
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
    const auto sample = rdna4::build_vsample(32, {.dmask = 1}); // IMAGE_SAMPLE_C
    std::copy(sample.begin(), sample.end(), words.begin());
  } else {
    const auto sample = rdna3::build_mimg(32, {.dmask = 1}); // IMAGE_SAMPLE_C
    std::copy(sample.begin(), sample.end(), words.begin());
  }
  run(words);
  EXPECT_EQ(wave_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
}

TEST_P(GraphicsExportTest, LinearMipLevelsUseReverseAllocationOrder) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint32_t offsets[] = {1024, 512, 256, 0};
  for (uint32_t level = 0; level < 4; ++level) {
    SCOPED_TRACE(level);
    std::array<uint32_t, 8> descriptor{0x3000, 0, 11, (8u << 28) | 0xfac, 0, 0, 0, 0};
    descriptor[1] = (63u << (gfx12 ? 17 : 20)) | (3u << 30) | (3u << (gfx12 ? 12 : 16));
    descriptor[2] = 11; // Full level-zero width is 48 texels.
    if (gfx12) {
      descriptor[1] |= level << 25;
      descriptor[3] |= level << 15;
    } else {
      descriptor[3] |= (level << 12) | (level << 16);
    }
    for (uint32_t r = 0; r < descriptor.size(); ++r)
      wave_->debug_write_sgpr(8 + r, descriptor[r]);
    wave_->set_exec(1);
    wave_->debug_write_vgpr(2, 0, 5);
    const uint32_t address = 0x300000 + offsets[level] + 5 * 16;
    for (uint32_t c = 0; c < 4; ++c)
      memory_.write32(address + 4 * c, std::bit_cast<uint32_t>(float(level + c)));
    std::array<uint32_t, 4> words{};
    if (gfx12) {
      const auto inst =
          rdna4::build_vimage(0, {.dim = 0, .dmask = 15, .vdata = 8, .rsrc = 8, .vaddr0 = 2});
      std::copy(inst.begin(), inst.end(), words.begin());
    } else {
      const auto inst =
          rdna3::build_mimg(0, {.dim = 0, .dmask = 15, .vaddr = 2, .vdata = 8, .srsrc = 2});
      std::copy(inst.begin(), inst.end(), words.begin());
    }
    auto decoded = decoder_->decode(words.data());
    ASSERT_FALSE(decoded.failed());
    auto instruction = std::move(decoded).value();
    ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
    ASSERT_NE(instruction->data(), nullptr);
    EXPECT_EQ(instruction->data_as<amdgpu::VectorMemState>()->per_lane_addr[0], address);
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
    pipeline.issue(instruction.release(), *wave_);
    for (uint32_t c = 0; c < 4; ++c)
      EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 0), std::bit_cast<uint32_t>(float(level + c)));
  }
}

TEST_P(GraphicsExportTest, HardwareSampleClampsCoordinatesAndConvertsSrgb) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const std::array<uint32_t, 8> descriptor{
      0x1000, (66u << (gfx12 ? 17 : 20)) | (1u << 30), 1u << 14, (9u << 28) | 0xfac, 0, 0, 0, 0};
  for (uint32_t r = 0; r < descriptor.size(); ++r)
    wave_->debug_write_sgpr(8 + r, descriptor[r]);
  for (uint32_t r = 4; r < 8; ++r)
    wave_->debug_write_sgpr(r, r == 4 ? 2 | (2 << 3) | (2 << 6) : 0);
  wave_->set_exec(5);
  for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
    wave_->debug_write_vgpr(8, lane, std::bit_cast<uint32_t>(lane == 2 ? 0.1f : 2.0f));
    wave_->debug_write_vgpr(9, lane, std::bit_cast<uint32_t>(lane == 2 ? -0.25f : 2.0f));
    wave_->debug_write_vgpr(10, lane, 0xdeadbeef);
  }
  memory_.write32(0x100000, 0xff000000);
  memory_.write32(gfx12 ? 0x100084 : 0x100104, 0x804080ff);
  // The cube shader aliases both coordinate VGPRs with the sampled result.
  std::array<uint32_t, 4> words{0xe7c6c001, 0x02001008, 0x00000809, 0};
  if (!gfx12) {
    const auto mimg = rdna3::build_mimg(
        31, {.nsa = 1, .dim = 1, .dmask = 15, .vaddr = 8, .vdata = 8, .srsrc = 2, .ssamp = 1});
    std::copy(mimg.begin(), mimg.end(), words.begin());
    words[2] = 9;
  }
  auto decoded = decoder_->decode(words.data());
  ASSERT_FALSE(decoded.failed());
  auto instruction = std::move(decoded).value();
  ASSERT_TRUE(instruction->is_memory_op());
  ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
  ASSERT_FALSE(wave_->instruction_execution_failed());
  ASSERT_NE(instruction->data(), nullptr);
  EXPECT_EQ(instruction->data_as<amdgpu::VectorMemState>()->wait_counter_type,
            gfx12 ? amdgpu::WaitCounterType::SAMPLECNT : amdgpu::WaitCounterType::LOADCNT);
  amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
  pipeline.issue(instruction.release(), *wave_);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(wave_->debug_read_vgpr(8, 0)), 1.0f);
  EXPECT_EQ(wave_->debug_read_vgpr(9, 0), 0x3e5d0000u);
  EXPECT_EQ(wave_->debug_read_vgpr(10, 0), 0x3d520000u);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(wave_->debug_read_vgpr(11, 0)), 128.0f / 255.0f);
  EXPECT_EQ(wave_->debug_read_vgpr(10, 1), 0xdeadbeef);
  EXPECT_EQ(wave_->debug_read_vgpr(8, 2), 0);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(wave_->debug_read_vgpr(11, 2)), 1.0f);
}

TEST_P(GraphicsExportTest, DepthClearAndComparisonsUseTiledD16AndD32) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  for (uint32_t bytes : {2u, 4u}) {
    for (uint32_t comparison = 0; comparison < 8; ++comparison) {
      amdgpu::Pm4QueueState state;
      state.num_instances = 1;
      state.uconfig_registers[0x242] = 17;
      state.context_registers[5] = (3 << 16) | 3;
      state.context_registers[6] = (bytes == 2 ? 1 : 3) | (3 << 4);
      state.context_registers[8] = state.context_registers[10] = 0x2000;
      state.context_registers[0x1c] = 6 | (comparison << 4);
      state.context_registers[0x198] = 2;
      state.context_registers[0x2f9] = 0x2d;
      state.context_registers[0x205] = 0x43f;
      state.context_registers[0x10f] = state.context_registers[0x110] =
          state.context_registers[0x111] = state.context_registers[0x112] =
              std::bit_cast<uint32_t>(2.0f);
      state.context_registers[0x113] = std::bit_cast<uint32_t>(1.0f);
      state.context_registers[0x90] = 1 | (1 << 16);
      state.context_registers[0x91] = 3 | (3 << 16);
      if (!gfx12) {
        state.context_registers[7] = (3 << 16) | 3;
        state.context_registers[0x10] = (bytes == 2 ? 1 : 3) | (24 << 4);
        state.context_registers[0x12] = state.context_registers[0x14] = 0x2000;
        state.context_registers[0x200] = 6 | (comparison << 4);
        state.context_registers[0x1c] = 0;
        state.context_registers[0x1b4] = 2;
        state.context_registers[0x206] = 0x43f;
        state.context_registers[0x205] = 0;
      }
      for (uint32_t y = 0; y < 4; ++y)
        for (uint32_t x = 0; x < 4; ++x) {
          const auto address = gfx12 ? amdgpu::gfx12_image_address(0x200000, x, y, 4, bytes, 3)
                                     : amdgpu::gfx11_image_address(0x200000, x, y, 4, bytes, 24);
          ASSERT_TRUE(address);
          const uint32_t value = bytes == 2 ? 65535 : std::bit_cast<uint32_t>(1.0f);
          memory_.write_block(*address, {reinterpret_cast<const uint8_t *>(&value), bytes}, 0);
        }
      auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
      for (uint32_t i = 0; i < 3; ++i)
        draw->export_lane(*wave_, i, 12, 15,
                          {std::bit_cast<uint32_t>(i == 2 ? 1.0f : -1.0f),
                           std::bit_cast<uint32_t>(i == 1 ? 1.0f : -1.0f), 0,
                           std::bit_cast<uint32_t>(1.0f)});
      draw->export_lane(*wave_, 0, 20, 1,
                        {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
      ASSERT_TRUE(draw->advance(memory_, 0));
      // Depth-only clears have no fragment exports; fixed-function Z still writes.
      EXPECT_FALSE(draw->advance(memory_, 0));
      const bool pass = comparison == 1 || comparison == 3 || comparison == 5 || comparison == 7;
      for (uint32_t y = 0; y < 4; ++y)
        for (uint32_t x = 0; x < 4; ++x) {
          const auto address = gfx12 ? amdgpu::gfx12_image_address(0x200000, x, y, 4, bytes, 3)
                                     : amdgpu::gfx11_image_address(0x200000, x, y, 4, bytes, 24);
          uint32_t actual = 0;
          ASSERT_EQ(
              memory_.read_block_exact(*address, {reinterpret_cast<uint8_t *>(&actual), bytes}, 0),
              amdgpu::AccessOutcome::Complete);
          const bool changed = pass && x >= 1 && x < 3 && y >= 1 && y < 3;
          EXPECT_EQ(actual, changed      ? 0
                            : bytes == 2 ? 65535
                                         : std::bit_cast<uint32_t>(1.0f))
              << bytes << "," << comparison << "," << x << "," << y;
        }
    }
  }
}

TEST_P(GraphicsExportTest, HardwareInterpolationEncodingUsesVgprSelectors) {
  // RADV's triangle fragment shader uses this word pair on both physical cards.
  const std::array<uint32_t, 4> words{0xcd000205, 0x040a0102, 0, 0};
  auto decoded = decoder_->decode(words.data());
  ASSERT_FALSE(decoded.failed());
  std::unique_ptr<Instruction> instruction(std::move(decoded).value());
  ASSERT_EQ(instruction->num_src_operands(), 3u);
  EXPECT_EQ(instruction->src_operand(0)->unified_vgpr_index(), 2u);
  EXPECT_EQ(instruction->src_operand(1)->unified_vgpr_index(), 0u);
  EXPECT_EQ(instruction->src_operand(2)->unified_vgpr_index(), 2u);
}

TEST_P(GraphicsExportTest, ParameterLoadOutOfRangeDestinationClearsExec) {
  wave_->set_m0(0);
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    run(rdna4::build_vdsdir(0, {.vdst = 255}));
  else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
    run(rdna3_5::build_ldsdir(0, {.vdst = 255}));
  else
    run(rdna3::build_ldsdir(0, {.vdst = 255}));
  EXPECT_EQ(wave_->exec(), 0u);
  EXPECT_FALSE(wave_->instruction_execution_failed());
}

TEST_P(GraphicsExportTest, SecondInterpolationStepUsesP20AndModifiers) {
  wave_->set_exec(15);
  wave_->set_mode_raw(0xf0);
  for (uint32_t lane = 0; lane < 4; ++lane) {
    wave_->debug_write_vgpr(0, lane, std::bit_cast<uint32_t>(4.0f));
    wave_->debug_write_vgpr(1, lane, std::bit_cast<uint32_t>(0.25f));
    wave_->debug_write_vgpr(2, lane, std::bit_cast<uint32_t>(float(lane)));
  }
  // Clamp(-P20 * J + tmp) = clamp(lane - 1); v0 aliases the quad coefficient.
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    run(rdna4::build_vinterp(
        1, {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .src2 = 258, .neg = 1}));
  else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
    run(rdna3_5::build_vinterp(
        1, {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .src2 = 258, .neg = 1}));
  else
    run(rdna3::build_vinterp(
        1, {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .src2 = 258, .neg = 1}));
  EXPECT_FALSE(wave_->instruction_execution_failed());
  for (uint32_t lane = 0; lane < 4; ++lane)
    EXPECT_EQ(wave_->debug_read_vgpr(0, lane), lane < 2 ? 0u : 0x3f800000u);
}

TEST_P(GraphicsExportTest, InterpolationSnapshotsQuadSourcesWhenDestinationAliases) {
  wave_->set_exec(0xfd); // Lane one is inactive but supplies P10 to its quad.
  wave_->set_mode_raw(0xf0);
  for (uint32_t lane = 0; lane < 8; ++lane) {
    float coefficient = lane % 4 == 0 ? 1.0f : lane % 4 == 1 ? 2.0f : 4.0f;
    if (lane >= 4)
      coefficient *= 2.0f;
    wave_->debug_write_vgpr(0, lane, std::bit_cast<uint32_t>(coefficient));
    wave_->debug_write_vgpr(1, lane, std::bit_cast<uint32_t>(0.25f));
  }
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    run(rdna4::build_vinterp(0, {.vdst = 0, .src0 = 256, .src1 = 257, .src2 = 256}));
  else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
    run(rdna3_5::build_vinterp(0, {.vdst = 0, .src0 = 256, .src1 = 257, .src2 = 256}));
  else
    run(rdna3::build_vinterp(0, {.vdst = 0, .src0 = 256, .src1 = 257, .src2 = 256}));
  EXPECT_FALSE(wave_->instruction_execution_failed());
  for (uint32_t lane = 0; lane < 8; ++lane) {
    const float expected = lane == 1 ? 2.0f : lane < 4 ? 1.5f : 3.0f;
    EXPECT_EQ(wave_->debug_read_vgpr(0, lane), std::bit_cast<uint32_t>(expected)) << lane;
  }
}

TEST_P(GraphicsExportTest, PreservesBitsAndSelectsActiveLanesAndComponents) {
  wave_->set_exec(uint64_t{1} | (uint64_t{1} << 31));
  wave_->debug_write_vgpr(3, 0, 0x7fc12345);
  wave_->debug_write_vgpr(9, 0, 0x80000000);
  wave_->debug_write_vgpr(3, 31, 0x3c003800);
  wave_->debug_write_vgpr(9, 31, 0xfedcba98);
  execute(12, 5);
  EXPECT_FALSE(wave_->instruction_execution_failed());
  ASSERT_EQ(collector_->exports.size(), 2u);
  EXPECT_EQ(collector_->exports[0].lane, 0u);
  EXPECT_EQ(collector_->exports[1].lane, 31u);
  EXPECT_EQ(collector_->exports[1].target, 12u);
  EXPECT_EQ(collector_->exports[1].mask, 5u);
  EXPECT_EQ(collector_->exports[0].values, (std::array<uint32_t, 4>{0x7fc12345, 0, 0x80000000, 0}));
  EXPECT_EQ(collector_->exports[1].values, (std::array<uint32_t, 4>{0x3c003800, 0, 0xfedcba98, 0}));
}

TEST_P(GraphicsExportTest, RejectsEnabledUnallocatedSourceBeforeExporting) {
  execute(0, 15);
  EXPECT_TRUE(wave_->instruction_execution_failed());
  EXPECT_TRUE(collector_->exports.empty());
}

TEST_P(GraphicsExportTest, UnsupportedRowModeFailsSubmission) {
  execute(12, 1, true);
  EXPECT_TRUE(wave_->instruction_execution_failed());
  EXPECT_TRUE(collector_->exports.empty());
}

TEST_P(GraphicsExportTest, MissingExportDestinationFailsExecution) {
  wave_->set_graphics_stage(nullptr);
  execute(0, 1);
  EXPECT_TRUE(wave_->instruction_execution_failed());
}

TEST_P(GraphicsExportTest, SkipExportAndEmptyExecDoNotAccessSources) {
  wave_->set_graphics_stage(nullptr);
  wave_->set_status_raw(wave_->status_raw() | (1u << 18));
  execute(12, 15, true);
  EXPECT_FALSE(wave_->instruction_execution_failed());
  wave_->set_status_raw(wave_->status_raw() & ~(1u << 18));
  wave_->set_exec(0);
  execute(12, 15, true);
  EXPECT_FALSE(wave_->instruction_execution_failed());
}

INSTANTIATE_TEST_SUITE_P(Rdna, GraphicsExportTest,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));

} // namespace
