// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"
#include "rocjitsu/vm/amdgpu/graphics_draw.h"
#include "rocjitsu/vm/amdgpu/graphics_stage.h"
#include "rocjitsu/vm/amdgpu/image_address.h"
#include "rocjitsu/vm/amdgpu/image_metadata.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/raster_math.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"

#include <cmath>

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
  void export_mask(amdgpu::Wavefront &, uint64_t) override {}
  void export_lane(amdgpu::Wavefront &, uint32_t lane, uint32_t target, uint32_t mask,
                   const std::array<uint32_t, 4> &values) override {
    exports.push_back({lane, target, mask, values});
  }
};

class GraphicsExportTest : public testing::TestWithParam<rj_code_arch_t> {
protected:
  amdgpu::GpuMemory memory_{"graphics_memory"};
  amdgpu::GpuVm vm_;
  std::optional<amdgpu::GpuVmAccess> access_;
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
    const auto address_space = vm_.register_address_space(
        0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(),
        std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory_), {}, true);
    access_ = vm_.snapshot(address_space);
    ASSERT_TRUE(access_);
    cu_->set_gpu_vm(&vm_);
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

TEST(GraphicsRasterMathTest, WeightedGradientsMatchPhysicalInterpolationPlanes) {
  // Gradients recovered from raw pull-model values across rotating cube faces.
  // The last two cases exercise cancellation in the reciprocal-W numerator.
  struct Case {
    double edge1, edge2, delta1, delta2;
    float inverse_area;
    uint32_t expected;
  };
  const Case cases[] = {
      {48.63671875, -55.8359375, 0x1.996342p-3, 0, -0x1.9c6a64p-12f, 0xbb7a9a10},
      {49.96484375, -54.25, 0x1.7b0c18p-3, 0, -0x1.7b9fap-12f, 0xbb5b698e},
      {49.96484375, -54.25, 0, 0x1.6edff4p-3, -0x1.7b9fap-12f, 0x3b66945d},
      {-28.57421875, 22.65234375, 0, 0x1.6e6aecp-3, -0x1.3dd3a6p-10f, 0xbba1033e},
      {42.9609375, 6.41796875, 0x1.b7dd2ap-3, 0, -0x1.413274p-12f, 0xbb393b23},
      {59.9140625, -67.0390625, 0, 0x1.683f6ap-3, -0x1.413274p-12f, 0x3b6cba88},
      {49.25, 1.0859375, -0x1.8a7f8p-9, -0x1.6ce818p-5, -0x1.2d795ap-12f, 0x386d1582},
      {48.63671875, -55.8359375, -0x1.52c248p-5, -0x1.35102p-4, -0x1.9c6a64p-12f, 0xba6304d6},
  };
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(amdgpu::raster::plane_gradient(
                  test.edge1, test.edge2, test.delta1, test.delta2, test.inverse_area)),
              test.expected);
}

TEST(GraphicsRasterMathTest, SubpixelQuantizationRoundsMidpointsToEven) {
  for (double value : {0.0, 17.0, -17.0}) {
    EXPECT_EQ(amdgpu::raster::round_subpixel(value + 0.5 / 256), value);
    EXPECT_EQ(amdgpu::raster::round_subpixel(value + 1.5 / 256), value + 2.0 / 256);
  }
}

TEST(GraphicsRasterMathTest, ViewportTruncationMatchesPhysicalRdna3AndRdna4) {
  // Controlled triangles distinguish truncating the multiply and the add from
  // a fused operation and from discarding aligned operand bits before adding.
  // Coordinates were recovered from raw hardware barycentric captures.
  struct Case {
    float position, w, scale, offset;
    int subpixel;
  };
  const Case cases[] = {
      {-0.8996930718421936f, 1, 210, 210, 5393},
      {-0.8993768692016602f, 1, 210, 210, 5410},
      {-0.8990606069564819f, 1, 210, 210, 5427},
      {-0.6084542870521545f, 1, 210, 210, 21049},
      {-0.6078218221664429f, 1, 210, 210, 21084},
      {-0.39026230573654175f, 1, 210, 210, 32780},
      {-0.3889974355697632f, 1, 210, 210, 32847},
      {0.0005859374068677425f, 1, 210, 210, 53791},
      {0.0012183779617771506f, 1, 210, 210, 53825},
      {0x1.719728p+3f, 0x1.2cded6p+4f, 210, 210, 86779},
      {0x1.4410a2p+4f, 0x1.4367dcp+4f, 160, 160, 82003},
      {0x1.ff166cp-1f, 0x1.0a0878p+2f, 64, 64, 20318},
  };
  for (const auto &test : cases)
    EXPECT_EQ(amdgpu::raster::viewport_coordinate(test.position, test.w, test.scale, test.offset),
              test.subpixel / 256.0);
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

TEST_P(GraphicsExportTest, RasterCoverageFragmentInputsAndUnsupportedStates) {
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
    uint32_t sample_coverage = 15;
    uint32_t inputs = 2;
    std::array<float, 4> depth_bias{};
    bool depth_only = false;
    float depth = 0;
    bool first_vertex_depth_only = false;
    float extent = 1;
    bool full_scissor = false;
    bool triangle = false;
    uint32_t expected_coverage = 0;
    bool passthrough = false;
    bool viewport_scissor = false;
  };
  const Case cases[] = {
      {.name = "integer coverage"},
      {.name = "viewport scissor", .expected_coverage = 0x440, .viewport_scissor = true},
      {.name = "unequal W and explicit vertex parameters", .inputs = 0x8f28, .passthrough = true},
      {.name = "position z and packed coordinates", .inputs = 0x8402},
      {.name = "unsupported line stipple input", .outcome = Outcome::Reject, .inputs = 0x80},
      {.name = "unsupported ancillary input", .outcome = Outcome::Reject, .inputs = 0x2000},
      {.name = "fractional coverage", .extent = 0.625f, .full_scissor = true},
      {.name = "rasterizer discard", .outcome = Outcome::Empty, .clip_control = 1u << 22},
      {.name = "fragment discard enabled", .shader_control = 1u << 6},
      {.name = "early fragment tests", .outcome = Outcome::Reject, .shader_control = 1u << 12},
      {.name = "sample disabled", .outcome = Outcome::Empty, .sample_coverage = 0},
      {.name = "sample at even x and y", .sample_coverage = 1},
      {.name = "sample at odd x and even y", .sample_coverage = 2},
      {.name = "sample at even x and odd y", .sample_coverage = 4},
      {.name = "sample at odd x and y", .sample_coverage = 8},
      {.name = "fully clipped", .outcome = Outcome::Empty, .depth = 2},
      {.name = "partially clipped",
       .outcome = Outcome::Reject,
       .depth = 2,
       .first_vertex_depth_only = true},
      {.name = "triangle far clipping",
       .depth = 2,
       .first_vertex_depth_only = true,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x136},
      {.name = "triangle near clipping",
       .depth = -2,
       .first_vertex_depth_only = true,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x136},
      {.name = "triangle D3D near boundary",
       .outcome = Outcome::Empty,
       .clip_control = 1u << 19,
       .depth = -2,
       .first_vertex_depth_only = true,
       .triangle = true},
      {.name = "triangle far clipping disabled",
       .clip_control = 1u << 27,
       .depth = 2,
       .first_vertex_depth_only = true,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x137},
      {.name = "triangle near clipping disabled",
       .clip_control = (1u << 19) | (1u << 26),
       .depth = -2,
       .first_vertex_depth_only = true,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x137},
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
    state.uconfig_registers[0x242] = test.triangle ? 4 : 0x11;
    state.context_registers[0x3b0] = 10;
    state.context_registers[0x31e] = (3 << 16) | 3;
    state.context_registers[0x31f] = 3 << 15;
    state.context_registers[0x318] = 0x1000;
    state.context_registers[0x214] = 15;
    state.context_registers[0x215] = 15;
    state.context_registers[0x195] = 4;
    state.context_registers[0x198] = test.inputs;
    state.context_registers[0x2f9] = 0x2d;
    state.context_registers[0x205] = 0x43f;
    state.context_registers[0x10f] = state.context_registers[0x110] =
        state.context_registers[0x111] = state.context_registers[0x112] =
            std::bit_cast<uint32_t>(2.0f);
    state.context_registers[0x90] = 1 | (1 << 16);
    state.context_registers[0x91] = (3 - gfx12) | ((3 - gfx12) << 16);
    if (!gfx12) {
      state.context_registers[0x31c] = 10;
      state.context_registers[0x3b0] = 3 | (3 << 14);
      state.context_registers[0x3b8] = 26 << 14;
      state.context_registers[0x31e] = 0;
      state.context_registers[0x8e] = 15;
      state.context_registers[0x8f] = 15;
      state.context_registers[0x1c5] = 4;
      state.context_registers[0x1b4] = test.inputs;
      state.context_registers[0x206] = 0x43f;
      state.context_registers[0x205] = 0;
    }
    if (test.full_scissor) {
      state.context_registers[0x90] = 0;
      state.context_registers[0x91] = (4 - gfx12) | ((4 - gfx12) << 16);
    }
    state.context_registers[0x204] = test.clip_control;
    state.context_registers[gfx12 ? 0x1b : 0x203] = test.shader_control;
    state.context_registers[gfx12 ? 0x216 : 0x202] = test.color_control;
    state.context_registers[gfx12 ? 0x207 : 0x205] = test.polygon_mode;
    state.context_registers[0x2f8] = test.samples;
    state.context_registers[0x30e] =
        (test.sample_coverage & 1) | ((test.sample_coverage & 2) << 15);
    state.context_registers[0x30f] =
        ((test.sample_coverage & 4) >> 2) | ((test.sample_coverage & 8) << 13);
    for (uint32_t i = 0; i < test.depth_bias.size(); ++i)
      state.context_registers[0x2e0 + i] = std::bit_cast<uint32_t>(test.depth_bias[i]);
    if (test.depth_only) {
      state.context_registers[gfx12 ? 0x1c : 0x200] = 6 | (7 << 4);
      state.context_registers[gfx12 ? 5 : 7] = (3 << 16) | 3;
      state.context_registers[gfx12 ? 6 : 0x10] = 3 | ((gfx12 ? 3 : 24) << 4);
      state.context_registers[gfx12 ? 8 : 0x12] = 0x2000;
      state.context_registers[gfx12 ? 10 : 0x14] = 0x2000;
    }
    if (test.passthrough) {
      state.sh_registers[gfx12 ? 0x84 : 0x88] = 0x300000;
      if (gfx12)
        state.sh_registers[0x31] = 1u << 11;
      else
        state.context_registers[0x1b6] = 1;
      state.context_registers[gfx12 ? 0x199 : 0x191] = 0x420;
      state.context_registers[gfx12 ? 0x197 : 0x1b3] = test.inputs;
      for (uint32_t c = 0; c < 4; ++c)
        memory_.write32(0x3000a0 + c * 4, c == 0 ? 0x400000 : 0);
      for (uint32_t k = 0; k < 3; ++k)
        for (uint32_t c = 0; c < 4; ++c)
          memory_.write32(0x400000 + k * 16 + c * 4,
                          std::bit_cast<uint32_t>(float(10 + k * 4 + c)));
    }
    if (test.viewport_scissor) {
      state.context_registers[0x292] = 2;
      state.context_registers[0x94] = 2 | (1 << 16);
      state.context_registers[0x95] = (3 - gfx12) | ((3 - gfx12) << 16);
    }
    const float extent = test.extent;
    auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
    for (uint32_t i = 0; i < 3; ++i) {
      const float w = test.passthrough ? float(1u << i) : 1.0f;
      const std::array<uint32_t, 4> position{
          std::bit_cast<uint32_t>((i == 2 ? extent : -extent) * w),
          std::bit_cast<uint32_t>((i == 1 ? extent : -extent) * w),
          std::bit_cast<uint32_t>(!test.first_vertex_depth_only || i == 0 ? test.depth : 0.0f),
          std::bit_cast<uint32_t>(w)};
      draw->export_lane(*wave_, i, 12, 15, position);
    }
    draw->export_lane(*wave_, 0, 20, 1,
                      {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
    if (test.outcome == Outcome::Reject) {
      EXPECT_THROW(draw->advance(*access_), std::runtime_error);
      continue;
    }
    if (test.outcome == Outcome::Empty) {
      EXPECT_FALSE(draw->advance(*access_));
      EXPECT_EQ(memory_.read32(0x100000), 0);
      continue;
    }
    const auto dispatch = draw->advance(*access_);
    ASSERT_TRUE(dispatch);
    ASSERT_EQ(dispatch->total_wgs, 1);
    wave_->set_wg_coord(0, 0, 0);
    wave_->set_graphics_stage(draw);
    draw->initialize(*wave_, 0, 0);
    EXPECT_EQ(std::popcount(wave_->exec()), (test.triangle || test.viewport_scissor)
                                                ? std::popcount(test.expected_coverage)
                                                : std::popcount(test.sample_coverage));
    uint32_t covered_index = 0;
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
      if (!(wave_->exec() & (uint64_t{1} << lane)))
        continue;
      if (test.inputs == 0x8f28) {
        // Pull I/W, J/W, 1/W; linear I/J; position XYZW; packed XY.
        const uint32_t x = 1 + covered_index % 2, y = 1 + covered_index / 2;
        // Screen vertices are (0,0), (0,4), (4,0), with W={1,2,4}.
        // Every intermediate below is dyadic and exactly representable.
        const float b1 = (y + 0.5f) / 4, b2 = (x + 0.5f) / 4;
        const float rw = 1 - b1 - b2 + b1 / 2 + b2 / 4;
        const float expected[] = {b1 / 2, b2 / 4, rw, b1, b2, x + 0.5f, y + 0.5f, 0, rw};
        for (uint32_t reg = 0; reg < 9; ++reg)
          EXPECT_EQ(wave_->debug_read_vgpr(reg, lane), std::bit_cast<uint32_t>(expected[reg]))
              << "register=" << reg << " lane=" << lane;
        EXPECT_EQ(wave_->debug_read_vgpr(9, lane), x | (y << 16));
        for (uint32_t c = 0; c < 4; ++c)
          for (uint32_t k = 0; k < 3; ++k)
            EXPECT_EQ(lds_.read32(wave_->lds_base() + (c * 3 + k) * 4),
                      std::bit_cast<uint32_t>(float(10 + k * 4 + c)));
      } else if (test.inputs == 0x8402) {
        EXPECT_EQ(wave_->debug_read_vgpr(2, lane), 0u);
        const uint32_t packed = wave_->debug_read_vgpr(3, lane);
        EXPECT_GE(packed & 0xffff, 1u);
        EXPECT_LT(packed & 0xffff, 3u);
        EXPECT_GE(packed >> 16, 1u);
        EXPECT_LT(packed >> 16, 3u);
      }
      ++covered_index;
    }
    // Include helper lanes in exports; only covered fragments may write.
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
      draw->export_lane(*wave_, lane, 0, 3, {0x00003c00, 0x3c000000, 0, 0});
    EXPECT_FALSE(draw->advance(*access_));
    for (uint32_t y = 0; y < 4; ++y)
      for (uint32_t x = 0; x < 4; ++x) {
        const auto address = gfx12 ? amdgpu::gfx12_image_offset(x, y, 4, 4, 3)
                                   : amdgpu::gfx11_image_offset(x, y, 4, 4, 26);
        ASSERT_TRUE(address);
        const bool sample_enabled = test.sample_coverage & (1u << ((x & 1) + 2 * (y & 1)));
        const bool covered = (test.triangle || test.viewport_scissor)
                                 ? (test.expected_coverage & (1u << (y * 4 + x)))
                                 : x >= 1 && x < 3 && y >= 1 && y < 3;
        EXPECT_EQ(memory_.read32(0x100000 + *address),
                  !test.depth_only && sample_enabled && covered ? 0xff0000ffu : 0u)
            << x << "," << y;
      }
  }
}

TEST_P(GraphicsExportTest, ArrayAttachmentViewsSelectProvokingVertexAndPreserveOtherLayers) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const auto image_address = gfx12 ? amdgpu::gfx12_image_address : amdgpu::gfx11_image_address;
  for (uint32_t target : {0u, 7u})
    for (uint32_t first : {0u, 2u})
      for (bool enabled : {false, true})
        for (bool last_provoking : {false, true})
          for (uint32_t exported : {1u, 3u}) {
            SCOPED_TRACE(testing::Message()
                         << "target=" << target << ", first=" << first << ", enabled=" << enabled
                         << ", last_provoking=" << last_provoking << ", exported=" << exported);
            amdgpu::Pm4QueueState state;
            state.num_instances = 1;
            state.uconfig_registers[0x242] = 4;
            auto &context = state.context_registers;

            const uint32_t block = 0x318 + (gfx12 ? 9 : 15) * target;
            context[gfx12 ? 0x3b0 + target : block + 4] = 10;
            context[gfx12 ? block + 6 : 0x3b0 + target] = gfx12 ? 15 | (15 << 16) : 15 | (15 << 14);
            context[gfx12 ? block + 7 : 0x3b8 + target] = (gfx12 ? 3u << 15 : 26u << 14) | 4;
            context[gfx12 ? block + 1 : block + 3] = first | ((first + 2) << (gfx12 ? 14 : 13));
            context[block] = 0x1000;
            context[gfx12 ? 0x214 : 0x8e] = 15u << (4 * target);
            context[gfx12 ? 0x215 : 0x8f] = 15u << (4 * target);
            context[gfx12 ? 0x195 : 0x1c5] = 9;
            context[gfx12 ? 0x198 : 0x1b4] = 2;
            context[0x2f9] = 0x2d;
            context[gfx12 ? 0x205 : 0x206] = 0x43f;
            context[gfx12 ? 0x206 : 0x207] = enabled ? 1u << 18 : 0;
            context[gfx12 ? 0x207 : 0x205] = last_provoking ? 1u << 19 : 0;
            context[gfx12 ? 0x216 : 0x202] = 0xcc0010;
            context[0x10f] = context[0x110] = context[0x111] = context[0x112] =
                std::bit_cast<uint32_t>(8.0f);
            context[0x91] = gfx12 ? 15 | (15 << 16) : 16 | (16 << 16);
            context[0x30e] = context[0x30f] = 0xffffffff;
            // One 64KiB block per slice on both chosen layouts; inspect every texel.
            for (uint32_t layer = 0; layer < 5; ++layer)
              for (uint32_t y = 0; y < 16; ++y)
                for (uint32_t x = 0; x < 16; ++x) {
                  const uint64_t base =
                      amdgpu::image_layer_base(gfx12, 0x100000, 65536, layer, 4, gfx12 ? 3 : 26);
                  const auto address = image_address(base, x, y, 16, 4, gfx12 ? 3 : 26);
                  memory_.write32(*address, 0xdeadbeef);
                }

            auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
            for (uint32_t i = 0; i < 3; ++i) {
              draw->export_lane(*wave_, i, 12, 15,
                                {std::bit_cast<uint32_t>(i == 2 ? 1.0f : -1.0f),
                                 std::bit_cast<uint32_t>(i == 1 ? 1.0f : -1.0f), 0,
                                 std::bit_cast<uint32_t>(1.0f)});
              draw->export_lane(*wave_, i, 13, 4, {0, 0, i == 2 ? exported : 0, 0});
            }
            draw->export_lane(*wave_, 0, 20, 1,
                              {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
            const uint32_t relative_layer = enabled && last_provoking ? exported : 0;
            auto dispatch = draw->advance(*access_);
            if (relative_layer > 2)
              EXPECT_FALSE(dispatch);
            else {
              ASSERT_TRUE(dispatch);
              EXPECT_GT(dispatch->grid_wgs_x, 1u);
              for (uint32_t workgroup = 0; workgroup < dispatch->grid_wgs_x; ++workgroup) {
                wave_->set_wg_coord(workgroup, 0, 0);
                wave_->set_graphics_stage(draw);
                draw->initialize(*wave_, workgroup, 0);
                for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
                  draw->export_lane(*wave_, lane, 0, 15, {0x3f800000, 0, 0, 0x3f800000});
              }
              EXPECT_FALSE(draw->advance(*access_));
            }
            uint32_t changed = 0;
            for (uint32_t layer = 0; layer < 5; ++layer)
              for (uint32_t y = 0; y < 16; ++y)
                for (uint32_t x = 0; x < 16; ++x) {
                  const uint64_t base =
                      amdgpu::image_layer_base(gfx12, 0x100000, 65536, layer, 4, gfx12 ? 3 : 26);
                  const auto address = image_address(base, x, y, 16, 4, gfx12 ? 3 : 26);
                  const auto value = memory_.read32(*address);
                  if (value != 0xdeadbeef) {
                    ++changed;
                    EXPECT_EQ(layer, first + relative_layer);
                    EXPECT_EQ(value, 0xff0000ffu);
                  }
                }
            if (relative_layer <= 2)
              EXPECT_GT(changed, 32u);
            else
              EXPECT_EQ(changed, 0u);
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

TEST_P(GraphicsExportTest, TriangleAndRectangleListsKeepTrailingVerticesWithoutCreatingPrimitives) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  for (uint32_t primitive : {4u, 17u})
    for (uint32_t vertices : {4u, 5u, 31u, 32u, 64u, 65u}) {
      SCOPED_TRACE(testing::Message() << "vertices=" << vertices << ", primitive="
                                      << (primitive == 4 ? "triangle list" : "rectangle list"));
      amdgpu::Pm4QueueState state;
      state.num_instances = 2;
      state.uconfig_registers[0x242] = primitive;
      auto &context = state.context_registers;
      context[gfx12 ? 0x2a6 : 0x2d5] = 1u << 22; // Wave32: groups of thirty vertices.
      context[gfx12 ? 0x31e : 0x3b0] = gfx12 ? 3 | (3 << 16) : 3 | (3 << 14);
      context[0x2f9] = 0x2d;
      context[gfx12 ? 0x205 : 0x206] = 0x43f;
      context[gfx12 ? 0x216 : 0x202] = 0xcc0010;
      context[0x10f] = context[0x110] = context[0x111] = context[0x112] =
          std::bit_cast<uint32_t>(2.0f);
      context[gfx12 ? 0x3b0 : 0x31c] = 10;
      context[0x318] = 0x1000;
      context[gfx12 ? 0x214 : 0x8e] = 15;
      context[gfx12 ? 0x215 : 0x8f] = 15;
      context[gfx12 ? 0x195 : 0x1c5] = 9;
      // Cull these primitives so advance returns the
      // next vertex group, allowing us to inspect every invocation and primitive.
      context[gfx12 ? 0x207 : 0x205] = 3;

      auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), vertices);
      uint32_t invocations = 0, primitives = 0, groups = 0;
      do {
        ASSERT_LT(groups++, 10u);
        draw->initialize(*wave_, 0, 0);
        const uint32_t count = wave_->debug_read_sgpr(3) & 255;
        const uint32_t primitive_count = (wave_->debug_read_sgpr(3) >> 8) & 255;
        for (uint32_t i = 0; i < count; ++i) {
          EXPECT_EQ(wave_->debug_read_vgpr(gfx12 ? 3 : 5, i), invocations % vertices);
          ++invocations;
          draw->export_lane(*wave_, i, 12, 15,
                            {std::bit_cast<uint32_t>(i % 3 == 2 ? 1.0f : -1.0f),
                             std::bit_cast<uint32_t>(i % 3 == 1 ? 1.0f : -1.0f), 0,
                             std::bit_cast<uint32_t>(1.0f)});
        }
        for (uint32_t i = 0; i < primitive_count; ++i)
          draw->export_lane(*wave_, i, 20, 1, {wave_->debug_read_vgpr(0, i), 0, 0, 0});
        primitives += primitive_count;
      } while (draw->advance(*access_));
      EXPECT_EQ(invocations, 2 * vertices);
      EXPECT_EQ(primitives, 2 * (vertices / 3));
      EXPECT_FALSE(wave_->instruction_execution_failed());
    }
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

TEST(GraphicsImageAddressTest, MipAddressesMatchAddrLibAtOddSizesAndTailTransitions) {
  // Independent AddrLib addresses of the last accessible texel of each view.
  struct Case {
    bool gfx12;
    uint32_t swizzle, bytes, width, height, levels, level;
    uint64_t expected;
  };
  constexpr Case cases[] = {
      {false, 0, 4, 129, 33, 8, 2, 4988},          {true, 0, 4, 129, 33, 8, 2, 4988},
      {false, 2, 8, 200, 180, 8, 2, 29960},        {true, 1, 8, 200, 180, 8, 2, 29960},
      {false, 22, 2, 200, 180, 8, 2, 9026},        {true, 2, 2, 200, 180, 8, 2, 11074},
      {false, 24, 1, 200, 180, 8, 3, 49508},       {false, 28, 2, 512, 512, 10, 3, 208382},
      {false, 27, 4, 200, 180, 8, 1, 121628},      {true, 3, 4, 200, 180, 8, 1, 124188},
      {false, 27, 4, 200, 180, 8, 2, 52868},       {true, 3, 4, 200, 180, 8, 2, 47492},
      {false, 31, 16, 4093, 2049, 12, 4, 1011168}, {true, 4, 16, 4093, 2049, 12, 4, 1048544},
      {false, 30, 4, 512, 512, 10, 9, 1536},       {true, 4, 4, 512, 512, 10, 9, 1536},
  };
  for (const auto &c : cases) {
    SCOPED_TRACE(testing::Message() << c.gfx12 << "," << c.swizzle << "," << c.level);
    const auto mip =
        amdgpu::image_mip_layout(c.gfx12, c.swizzle, c.bytes, c.width, c.height, c.levels, c.level);
    ASSERT_TRUE(mip);
    const auto image_address = c.gfx12 ? amdgpu::gfx12_image_address : amdgpu::gfx11_image_address;
    EXPECT_EQ(image_address(mip->offset, mip->tail_x + mip->width - 1,
                            mip->tail_y + mip->height - 1, mip->pitch, c.bytes, c.swizzle),
              c.expected);
  }
  for (bool gfx12 : {false, true}) {
    EXPECT_FALSE(amdgpu::image_mip_layout(gfx12, 0, 4, 64, 64, 0, 0));
    EXPECT_FALSE(amdgpu::image_mip_layout(gfx12, 0, 4, 64, 64, 8, 0));
    EXPECT_FALSE(amdgpu::image_mip_layout(gfx12, 0, 4, 64, 64, 7, 7));
    EXPECT_FALSE(amdgpu::image_mip_layout(gfx12, 0, 4, 0, 64, 1, 0));
    EXPECT_FALSE(amdgpu::image_mip_layout(gfx12, 0, 4, 65537, 64, 1, 0));
    EXPECT_FALSE(amdgpu::image_mip_layout(gfx12, 18, 4, 64, 64, 7, 0));
    EXPECT_FALSE(amdgpu::image_mip_layout(gfx12, 0, 3, 64, 64, 7, 0));
  }
}

TEST(GraphicsImageAddressTest, ArrayLayersUseMipChainStrideAndSliceXor) {
  struct Case {
    bool gfx12;
    uint32_t swizzle, bytes, levels, level, x, y, layer;
    uint64_t slice_size, address;
  };
  // Independent AddrLib offsets for 129x71 surfaces, including linear rows,
  // mip tails, different block sizes and layer-dependent pipe/bank selection.
  const Case cases[] = {
      {false, 0, 4, 7, 3, 12, 6, 3, 82432, 251440},
      {true, 0, 4, 7, 3, 12, 6, 3, 82432, 250672},
      {false, 22, 4, 7, 2, 19, 14, 35, 102400, 3593652},
      {false, 26, 8, 7, 2, 19, 14, 35, 393216, 13799848},
      {false, 27, 4, 7, 2, 19, 14, 3, 262144, 836788},
      {false, 27, 8, 7, 2, 19, 14, 7, 393216, 2776744},
      {false, 27, 16, 7, 2, 19, 14, 7, 655360, 4676816},
      {false, 31, 8, 7, 2, 19, 14, 31, 524288, 16428200},
      {false, 31, 16, 7, 2, 19, 14, 31, 1048576, 32562384},
      {true, 1, 4, 7, 2, 19, 14, 3, 57088, 175796},
      {true, 3, 8, 7, 2, 19, 14, 35, 393216, 13798824},
      {true, 4, 16, 7, 2, 19, 14, 255, 1048576, 267527632},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(testing::Message()
                 << test.gfx12 << ", swizzle=" << test.swizzle << ", bytes=" << test.bytes
                 << ", level=" << test.level << ", layer=" << test.layer);
    const auto mip = amdgpu::image_mip_layout(test.gfx12, test.swizzle, test.bytes, 129, 71,
                                              test.levels, test.level);
    ASSERT_TRUE(mip);
    EXPECT_EQ(mip->slice_size, test.slice_size);
    const uint64_t base = amdgpu::image_layer_base(test.gfx12, mip->offset, mip->slice_size,
                                                   test.layer, test.bytes, test.swizzle);
    const auto image_address =
        test.gfx12 ? amdgpu::gfx12_image_address : amdgpu::gfx11_image_address;
    const auto address = image_address(base, test.x + mip->tail_x, test.y + mip->tail_y, mip->pitch,
                                       test.bytes, test.swizzle);
    ASSERT_TRUE(address);
    EXPECT_EQ(*address, test.address);
  }
}

TEST_P(GraphicsExportTest, HardwareImageLoadReadsTiledUintChannelsAndPacksD16) {
  if (GetParam() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "RDNA4 image transfer encoding";
  for (bool a16 : {false, true}) {
    SCOPED_TRACE(a16);
    cu_->l1_vector().flush_all();
    cache_.flush_all();
    cu_->l1_vector().invalidate_all();
    cache_.invalidate_all();
    const std::array<uint32_t, 8> descriptor{
        0x1000, (46u << 17) | (3u << 30), 3u << 14, (9u << 28) | (3u << 20) | 0xfac, 0, 0, 0, 0};
    for (uint32_t r = 0; r < descriptor.size(); ++r)
      wave_->debug_write_sgpr(8 + r, descriptor[r]);
    wave_->set_exec(5);
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
      wave_->debug_write_vgpr(0, lane, 0xdeadbeef);
      wave_->debug_write_vgpr(1, lane, 0xdeadbeef);
      wave_->debug_write_vgpr(2, lane, (lane == 2 ? 4 : 1) | (a16 ? 2u << 16 : 0));
      wave_->debug_write_vgpr(3, lane, 2);
    }
    // Byte address for texel (1,2) in a 4-byte GFX12 64 KiB tile.
    memory_.write32(0x100000 + 36, 0x44332211);
    const std::array<uint32_t, 4> words{0xd3c00021u | (a16 ? 1u << 6 : 0), 0x00001000,
                                        a16 ? 0x0000ff02u : 0x00000302u, 0};
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
    const auto store = rdna4::build_vimage(6, {.dim = 1,
                                               .d16 = 1,
                                               .a16 = a16,
                                               .dmask = 15,
                                               .rsrc = 8,
                                               .vaddr0 = 2,
                                               .vaddr1 = static_cast<uint8_t>(a16 ? 255 : 3)});
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
}

TEST_P(GraphicsExportTest, LinearImageLoadsRespectDefaultPitchAndArrayDescriptorFields) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    const char *name;
    uint32_t width, word4, type, dim, offset;
    uint32_t layer = 0;
    bool out_of_bounds = false;
  };
  const Case cases[] = {{"default pitch", 2, 0, 9, 1, gfx12 ? 132u : 260u},
                        {"custom pitch", 2, 127, 9, 1, 516},
                        {"two array layers", 128, 1, 13, 5, 516},
                        {"one array layer", 128, 0, 13, 5, 516},
                        {"second array layer", 128, 1, 13, 5, 1540, 1},
                        {"nonzero array view", 128, (2 << 16) | 4, 13, 5, 3588, 1},
                        {"array view limit", 128, (2 << 16) | 4, 13, 5, 516, 3, true},
                        {"array index overflow", 128, (2 << 16) | 4, 13, 5, 516, 0xffffffff, true}};
  for (bool a16 : {false, true})
    for (uint32_t i = 0; i < std::size(cases); ++i) {
      const auto &c = cases[i];
      SCOPED_TRACE(c.name);
      SCOPED_TRACE(a16);
      cu_->l1_vector().flush_all();
      cache_.flush_all();
      cu_->l1_vector().invalidate_all();
      cache_.invalidate_all();
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
      wave_->debug_write_vgpr(2, 0, a16 ? 1 | (1 << 16) : 1);
      wave_->debug_write_vgpr(3, 0, a16 ? (c.layer & 0xffff) | 0xbeef0000 : 1);
      wave_->debug_write_vgpr(4, 0, c.layer);
      memory_.write32(base + c.offset, 0x44332211);
      std::array<uint32_t, 4> words{};
      if (gfx12) {
        const auto inst = rdna4::build_vimage(0, {.dim = uint8_t(c.dim),
                                                  .a16 = a16,
                                                  .dmask = 15,
                                                  .vdata = 8,
                                                  .rsrc = 8,
                                                  .vaddr0 = 2,
                                                  .vaddr1 = 3,
                                                  .vaddr2 = static_cast<uint8_t>(a16 ? 255 : 4)});
        std::copy(inst.begin(), inst.end(), words.begin());
      } else {
        const auto inst = rdna3::build_mimg(
            0,
            {.dim = uint8_t(c.dim), .dmask = 15, .a16 = a16, .vaddr = 2, .vdata = 8, .srsrc = 2});
        std::copy(inst.begin(), inst.end(), words.begin());
      }
      auto decoded = decoder_->decode(words.data());
      ASSERT_FALSE(decoded.failed());
      auto instruction = std::move(decoded).value();
      ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
      ASSERT_NE(instruction->data(), nullptr);
      const auto *transfer = instruction->data_as<amdgpu::VectorMemState>();
      EXPECT_EQ(transfer->lane_mask, c.out_of_bounds ? 0u : 1u);
      if (!c.out_of_bounds)
        EXPECT_EQ(transfer->per_lane_addr[0], base + c.offset);
      amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
      pipeline.issue(instruction.release(), *wave_);
      for (uint32_t component = 0; component < 4; ++component)
        EXPECT_EQ(wave_->debug_read_vgpr(8 + component, 0),
                  c.out_of_bounds ? 0 : 0x11u * (component + 1));
      EXPECT_EQ(memory_.read32(base + c.offset), 0x44332211u);
    }
}

TEST_P(GraphicsExportTest, CompressedImageTransfersMaterializeClearsAndPreserveStores) {
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "GFX12 metadata is allocation-managed";
  constexpr uint64_t base = 0x200000, metadata = 0x100700;
  constexpr uint32_t width = 17, height = 13;
  for (bool a16 : {false, true})
    for (bool array : {false, true}) {
      SCOPED_TRACE(a16);
      SCOPED_TRACE(array);
      cu_->l1_vector().flush_all();
      cache_.flush_all();
      cu_->l1_vector().invalidate_all();
      cache_.invalidate_all();
      // AddrLib: layer 3 of this R_X surface has stride 64KiB and XOR 0x600.
      const uint64_t selected_base = array ? base + 3 * 65536 + 0x600 : base;
      const auto preserved = *amdgpu::gfx11_image_address(base, 11, 10, width, 4, 27);
      memory_.write32(preserved, 0x12345678);
      const std::array<uint32_t, 8> descriptor{
          base >> 8,
          46u << 20,
          ((width - 1) >> 2) | ((height - 1) << 14),
          ((array ? 13u : 9u) << 28) | (27u << 20) | 0xfac,
          array ? (2u << 16) | 4 : 0,
          0,
          (1u << 21) | (1u << 19) | (static_cast<uint32_t>((metadata >> 8) & 255) << 24),
          metadata >> 16};
      for (uint32_t i = 0; i < descriptor.size(); ++i)
        wave_->debug_write_sgpr(8 + i, descriptor[i]);
      wave_->set_exec(5);
      for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
        wave_->debug_write_vgpr(2, lane, (lane == 2 ? width : 11) | (a16 ? 10u << 16 : 0));
        wave_->debug_write_vgpr(3, lane, a16 ? 0xbeef0001 : 10);
        wave_->debug_write_vgpr(4, lane, 1);
        for (uint32_t c = 0; c < 4; ++c)
          wave_->debug_write_vgpr(8 + c, lane, 0xdeadbeef);
      }
      const auto tag = *amdgpu::gfx11_metadata_address(metadata, 11, 10, width, height, 4, 27,
                                                       false, true, array ? 3 : 0);
      // Queue the clear tag through L2 to check that the image path observes dirty
      // metadata and does not overwrite it with a stale backing-memory value.
      const uint8_t key = 8;
      memory_.write8(tag, 2);
      cache_.write(tag, &key, 1);
      amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
      const auto transfer = [&](uint32_t opcode) {
        const auto encoding = rdna3::build_mimg(opcode, {.dim = static_cast<uint8_t>(array ? 5 : 1),
                                                         .dmask = 15,
                                                         .a16 = a16,
                                                         .vaddr = 2,
                                                         .vdata = 8,
                                                         .srsrc = 2});
        std::array<uint32_t, 4> words{};
        std::copy(encoding.begin(), encoding.end(), words.begin());
        auto decoded = decoder_->decode(words.data());
        ASSERT_FALSE(decoded.failed());
        auto instruction = std::move(decoded).value();
        ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
        pipeline.issue(instruction.release(), *wave_);
        ASSERT_FALSE(wave_->instruction_execution_failed());
      };
      transfer(0);
      for (uint32_t c = 0; c < 4; ++c) {
        EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 0), c == 3 ? 255 : 0);
        EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 1), 0xdeadbeef);
        EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 2), 0);
        wave_->debug_write_vgpr(8 + c, 0, 0x11u * (c + 1));
      }
      transfer(6); // IMAGE_STORE
      transfer(0);
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 0), 0x11u * (c + 1));
      cu_->l1_vector().flush_all();
      cache_.flush_all();
      EXPECT_EQ(memory_.read32(*amdgpu::gfx11_image_address(selected_base, 11, 10, width, 4, 27)),
                0x44332211u);
      EXPECT_EQ(memory_.read32(*amdgpu::gfx11_image_address(selected_base, 10, 10, width, 4, 27)),
                0xff000000u);
      EXPECT_EQ(memory_.read8(tag), 0xff);
      if (array)
        EXPECT_EQ(memory_.read32(preserved), 0x12345678u);
      // Filtering must also publish dirty texels before materializing metadata.
      const uint32_t updated = 0x66332211;
      cache_.write(*amdgpu::gfx11_image_address(selected_base, 11, 10, width, 4, 27),
                   reinterpret_cast<const uint8_t *>(&updated), 4);
      wave_->debug_write_sgpr(9, 42u << 20);
      wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6));
      wave_->debug_write_sgpr(5, 0);
      wave_->debug_write_sgpr(6, (1 << 20) | (1 << 22));
      wave_->debug_write_sgpr(7, 0);
      wave_->set_exec(1);
      wave_->debug_write_vgpr(2, 0, std::bit_cast<uint32_t>(11.0f / width));
      wave_->debug_write_vgpr(3, 0, std::bit_cast<uint32_t>(10.5f / height));
      wave_->debug_write_vgpr(4, 0, std::bit_cast<uint32_t>(1.0f));
      // IMAGE_SAMPLE_LZ
      const auto encoding = rdna3::build_mimg(31, {.dim = static_cast<uint8_t>(array ? 5 : 1),
                                                   .dmask = 15,
                                                   .vaddr = 2,
                                                   .vdata = 8,
                                                   .srsrc = 2,
                                                   .ssamp = 1});
      std::array<uint32_t, 4> words{};
      std::copy(encoding.begin(), encoding.end(), words.begin());
      auto decoded = decoder_->decode(words.data());
      ASSERT_FALSE(decoded.failed());
      auto instruction = std::move(decoded).value();
      ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
      ASSERT_FALSE(wave_->instruction_execution_failed());
      pipeline.issue(instruction.release(), *wave_);
      const float expected[] = {8.5f / 255, 17.0f / 255, 25.5f / 255, 178.5f / 255};
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 0), std::bit_cast<uint32_t>(expected[c]));
      // A subsequent dirty metadata clear must affect only the sampled layer.
      cache_.write(tag, &key, 1);
      auto cleared = decoder_->decode(words.data());
      ASSERT_FALSE(cleared.failed());
      instruction = std::move(cleared).value();
      ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
      pipeline.issue(instruction.release(), *wave_);
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 0), c == 3 ? 0x3f800000u : 0);
      if (array)
        EXPECT_EQ(memory_.read32(preserved), 0x12345678u);
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

TEST_P(GraphicsExportTest, UnsupportedMipImageLoadReportsAnExecutionError) {
  std::array<uint32_t, 4> words{};
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
    const auto load = rdna4::build_vimage(1, {.dmask = 1}); // IMAGE_LOAD_MIP
    std::copy(load.begin(), load.end(), words.begin());
  } else {
    const auto load = rdna3::build_mimg(1, {.dmask = 1}); // IMAGE_LOAD_MIP
    std::copy(load.begin(), load.end(), words.begin());
  }
  run(words);
  EXPECT_EQ(wave_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
}

TEST_P(GraphicsExportTest, LinearMipLevelsUseReverseAllocationOrder) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  // The odd-width offsets were also checked by mapping physical RDNA3/RDNA4
  // image backing after clears; a load/store round trip alone can hide them.
  struct Case {
    uint32_t width;
    std::array<uint32_t, 4> offsets;
  };
  const Case cases[] = {{48, {1024, 512, 256, 0}}, {129, {2560, 1280, 512, 0}}};
  for (const auto &test : cases) {
    SCOPED_TRACE(test.width);
    const uint32_t base = 0x300000 + test.width * 0x1000;
    for (uint32_t level = 0; level < 4; ++level) {
      SCOPED_TRACE(level);
      std::array<uint32_t, 8> descriptor{base >> 8, 0, 11, (8u << 28) | 0xfac, 0, 0, 0, 0};
      descriptor[1] =
          (63u << (gfx12 ? 17 : 20)) | (((test.width - 1) & 3u) << 30) | (3u << (gfx12 ? 12 : 16));
      descriptor[2] = (test.width - 1) >> 2;
      if (gfx12) {
        descriptor[1] |= level << 25;
        descriptor[3] |= 3u << 15;
      } else {
        descriptor[3] |= (level << 12) | (3u << 16);
      }
      for (uint32_t r = 0; r < descriptor.size(); ++r)
        wave_->debug_write_sgpr(8 + r, descriptor[r]);
      wave_->set_exec(1);
      wave_->debug_write_vgpr(2, 0, 5);
      const uint32_t address = base + test.offsets[level] + 5 * 16;
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
}

// AddrLib offsets for the last texel of all eight 200x180 RGBA8 mip levels.
constexpr uint32_t kMipLastTexelOffsets[2][8] = {
    // GFX11
    {333692, 121628, 52868, 12680, 26676, 1220, 536, 8960},
    // GFX12
    {365692, 124188, 47492, 20104, 9012, 4292, 2072, 1536},
};

TEST_P(GraphicsExportTest, TiledMipTransfersUseViewBoundsAndIndependentBackingOffsets) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint32_t base = 0x400000;
  for (uint32_t level = 0; level < 8; ++level) {
    SCOPED_TRACE(level);
    const uint32_t width = std::max(1u, 200u >> level), height = std::max(1u, 180u >> level);
    std::array<uint32_t, 8> descriptor{base >> 8,
                                       (46u << (gfx12 ? 17 : 20)) | (3u << 30) |
                                           (7u << (gfx12 ? 12 : 16)),
                                       49u | (179u << 14),
                                       (9u << 28) | ((gfx12 ? 3u : 27u) << 20) | 0xfac,
                                       0,
                                       0,
                                       0,
                                       0};
    if (gfx12) {
      descriptor[1] |= level << 25;
      descriptor[3] |= 7u << 15;
    } else {
      descriptor[3] |= (level << 12) | (7u << 16);
    }
    for (uint32_t r = 0; r < descriptor.size(); ++r)
      wave_->debug_write_sgpr(8 + r, descriptor[r]);
    wave_->set_exec(15);
    for (uint32_t lane = 0; lane < 4; ++lane) {
      wave_->debug_write_vgpr(2, lane, lane == 1 ? width : lane == 3 ? ~0u : width - 1);
      wave_->debug_write_vgpr(3, lane, lane == 2 ? height : height - 1);
    }
    const uint32_t address = base + kMipLastTexelOffsets[gfx12][level];
    memory_.write32(address, 0x44332211);
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
    for (bool store : {false, true}) {
      std::array<uint32_t, 4> words{};
      if (gfx12) {
        const auto inst = rdna4::build_vimage(
            store ? 6 : 0,
            {.dim = 1, .dmask = 15, .vdata = 8, .rsrc = 8, .vaddr0 = 2, .vaddr1 = 3});
        std::copy(inst.begin(), inst.end(), words.begin());
      } else {
        const auto inst = rdna3::build_mimg(
            store ? 6 : 0, {.dim = 1, .dmask = 15, .vaddr = 2, .vdata = 8, .srsrc = 2});
        std::copy(inst.begin(), inst.end(), words.begin());
      }
      auto decoded = decoder_->decode(words.data());
      ASSERT_FALSE(decoded.failed());
      auto instruction = std::move(decoded).value();
      ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
      ASSERT_FALSE(wave_->instruction_execution_failed());
      ASSERT_NE(instruction->data(), nullptr);
      const auto *transfer = instruction->data_as<amdgpu::VectorMemState>();
      EXPECT_EQ(transfer->per_lane_addr[0], address);
      EXPECT_EQ(transfer->lane_mask, 1u);
      pipeline.issue(instruction.release(), *wave_);
      if (!store) {
        for (uint32_t c = 0; c < 4; ++c) {
          EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 0), 0x11u * (c + 1));
          for (uint32_t lane = 1; lane < 4; ++lane)
            EXPECT_EQ(wave_->debug_read_vgpr(8 + c, lane), 0u);
          wave_->debug_write_vgpr(8 + c, 0, 0x80u + c);
        }
      }
    }
    cu_->l1_vector().flush_all();
    cache_.flush_all();
    EXPECT_EQ(memory_.read32(address), 0x83828180u);
  }
}

TEST_P(GraphicsExportTest, ColorBlendingPreservesMasksAndUsesSeparateAlpha) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    uint32_t blend, mask, expected;
    bool reject = false;
    bool constant_boundary = false;
    uint32_t number_format = 0, export_format = 9;
    uint32_t initial = 0xff40bf80;
    std::array<uint32_t, 4> exported{0x3f000000, 0x3e800000, 0x3e800000, 0x3f000000};
  };
  constexpr uint32_t enabled = 1u << 30, separate = 1u << 29;
  const Case cases[] = {
      {0, 5, 0xff40bf80},
      {enabled | 4 | (5 << 8), 15, 0xbf407f80},
      {enabled | separate | 4 | (5 << 8) | (1 << 16), 15, 0x80407f80},
      {enabled | (2 << 5), 15, 0x80404080},
      {enabled | (3 << 5), 15, 0xff40bf80},
      {enabled | 11, 15, 0x40201020},
      {enabled | 10, 15, 0x80000000},
      {.blend = enabled | 13, .mask = 15, .expected = 0, .reject = true},
      {.blend = enabled | (5 << 5), .mask = 15, .expected = 0, .reject = true},
      {.blend = enabled | 1, .mask = 15, .expected = 0, .reject = true, .number_format = 4},
      // FP32 sRGB exports and FP16 sRGB blending need separate qualification.
      {.blend = 0, .mask = 15, .expected = 0, .reject = true, .number_format = 6},
      {.blend = enabled | 1,
       .mask = 15,
       .expected = 0,
       .reject = true,
       .number_format = 6,
       .export_format = 4},
      {.blend = enabled | 11, .mask = 5, .expected = 0xff20bf20},
      {.blend = enabled | separate | 11 | (17 << 16),
       .mask = 15,
       .expected = 0x9f6a3500,
       .constant_boundary = true,
       .exported = {0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000}},
      // Physical RDNA3/RDNA4 witnesses distinguish twelve-bit destination
      // normalization from both full FP32 and FP16 precision.
      {.blend = enabled | separate | 4 | (5 << 8) | (1 << 16) | (1 << 24),
       .mask = 15,
       .expected = 0xc6c999a8,
       .initial = 0x10c60035,
       .exported = {0x3f57c000, 0x3f57c000, 0x3f4ac000, 0x3f36a000}},
      {.blend = enabled | separate | 4 | (5 << 8) | (1 << 16) | (1 << 24),
       .mask = 15,
       .expected = 0x8d5c155a,
       .initial = 0x20760005,
       .exported = {0x3f4bc000, 0x3e40c000, 0x3e60e000, 0x3edac000}},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.blend);
    amdgpu::Pm4QueueState state;
    state.num_instances = 1;
    state.uconfig_registers[0x242] = 17;
    auto &context = state.context_registers;
    context[gfx12 ? 0x3b0 : 0x31c] = 10 | (test.number_format << 8);
    context[gfx12 ? 0x31e : 0x3b0] = gfx12 ? 3 | (3 << 16) : 3 | (3 << 14);
    context[gfx12 ? 0x31f : 0x3b8] = gfx12 ? 3u << 15 : 26u << 14;
    context[0x318] = 0x1000;
    context[gfx12 ? 0x214 : 0x8e] = test.mask;
    context[gfx12 ? 0x215 : 0x8f] = 15;
    context[gfx12 ? 0x195 : 0x1c5] = test.export_format;
    context[gfx12 ? 0x198 : 0x1b4] = 2;
    context[0x2f9] = 0x2d;
    context[gfx12 ? 0x205 : 0x206] = 0x43f;
    context[gfx12 ? 0x216 : 0x202] = 0xcc0010;
    context[0x1e0] = test.blend;
    const float constants[] = {0.25f, 0.25f, 0.5f, 0.5f};
    const uint32_t boundary[] = {0x3b008081, 0x3e56d6d7, 0x3ed5d5d6, 0x3f202020};
    for (uint32_t c = 0; c < 4; ++c)
      context[0x105 + c] =
          test.constant_boundary ? boundary[c] : std::bit_cast<uint32_t>(constants[c]);
    context[0x10f] = context[0x110] = context[0x111] = context[0x112] =
        std::bit_cast<uint32_t>(2.0f);
    context[0x91] = gfx12 ? 0 : 1 | (1 << 16);
    context[0x30e] = context[0x30f] = 0xffffffff;
    memory_.write32(0x100000, test.initial);

    auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
    for (uint32_t i = 0; i < 3; ++i)
      draw->export_lane(*wave_, i, 12, 15,
                        {std::bit_cast<uint32_t>(i == 2 ? 1.0f : -1.0f),
                         std::bit_cast<uint32_t>(i == 1 ? 1.0f : -1.0f), 0,
                         std::bit_cast<uint32_t>(1.0f)});
    draw->export_lane(*wave_, 0, 20, 1,
                      {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
    if (test.reject) {
      EXPECT_THROW(draw->advance(*access_), std::runtime_error);
      EXPECT_EQ(memory_.read32(0x100000), test.initial);
      continue;
    }
    ASSERT_TRUE(draw->advance(*access_));
    wave_->set_wg_coord(0, 0, 0);
    wave_->set_graphics_stage(draw);
    draw->initialize(*wave_, 0, 0);
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
      draw->export_lane(*wave_, lane, 0, 15, test.exported);
    EXPECT_FALSE(draw->advance(*access_));
    EXPECT_EQ(memory_.read32(0x100000), test.expected);
  }
}

TEST_P(GraphicsExportTest, WideColorAttachmentsWriteFullTexelsAndPreserveMaskedChannels) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    uint32_t data_format, number_format, export_format, bytes, mask;
    std::array<uint32_t, 4> exported, expected;
  };
  const Case cases[] = {
      // FP16 sRGB exports captured on both physical RDNA3 and RDNA4.
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x00000000, 0x00000000},
       .expected = {0x00000000}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x41941234, 0x1234c70c},
       .expected = {0x0000ff03}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x2b132aab, 0x2aabd685},
       .expected = {0x0d004341}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x76c737ff, 0x37ff4471},
       .expected = {0x7fffffbb}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x78003800, 0x38004800},
       .expected = {0x80ffffbc}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x571c3bfc, 0x3bfc75c4},
       .expected = {0xffffffff}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x5c003c00, 0x3c008400},
       .expected = {0xff00ffff}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x9c007c00, 0x7c004400},
       .expected = {0xffff00ff}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x0e007e00, 0x7e006200},
       .expected = {0x00ff0100}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0xdc00bc00, 0xbc000400},
       .expected = {0x00000000}},
      {.data_format = 12,
       .number_format = 0,
       .export_format = 4,
       .bytes = 8,
       .mask = 3,
       .exported = {0x38003400, 0x3c003a00},
       .expected = {0x80004000, 0xffffbfff}},
      {.data_format = 12,
       .number_format = 7,
       .export_format = 4,
       .bytes = 8,
       .mask = 3,
       .exported = {0x38003400, 0x3c003a00},
       .expected = {0x38003400, 0x3c003a00}},
      {.data_format = 12,
       .number_format = 4,
       .export_format = 7,
       .bytes = 8,
       .mask = 3,
       .exported = {0x45670123, 0xcdef89ab},
       .expected = {0x45670123, 0xcdef89ab}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x3e800000, 0x3f000000, 0x3f400000, 0x3f800000},
       .expected = {0x3e800000, 0x3f000000, 0x3f400000, 0x3f800000}},
      {.data_format = 14,
       .number_format = 4,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x12345678, 0x87654321, 0x0fedcba9, 0x9abcdef0},
       .expected = {0x12345678, 0x87654321, 0x0fedcba9, 0x9abcdef0}},
  };
  for (const auto &test : cases) {
    for (uint32_t write_mask : {5u, 15u}) {
      SCOPED_TRACE(test.data_format);
      SCOPED_TRACE(test.number_format);
      SCOPED_TRACE(write_mask);
      amdgpu::Pm4QueueState state;
      state.num_instances = 1;
      state.uconfig_registers[0x242] = 17;
      auto &context = state.context_registers;
      context[gfx12 ? 0x3b0 : 0x31c] = test.data_format | (test.number_format << 8);
      context[gfx12 ? 0x31e : 0x3b0] = gfx12 ? 3 | (3 << 16) : 3 | (3 << 14);
      // Linear targets cover the RGBA32F linear-image clear CTS regression.
      context[gfx12 ? 0x31f : 0x3b8] = 0;
      context[0x318] = 0x1000;
      context[gfx12 ? 0x214 : 0x8e] = write_mask;
      context[gfx12 ? 0x215 : 0x8f] = 15;
      context[gfx12 ? 0x195 : 0x1c5] = test.export_format;
      context[gfx12 ? 0x198 : 0x1b4] = 2;
      context[0x2f9] = 0x2d;
      context[gfx12 ? 0x205 : 0x206] = 0x43f;
      context[gfx12 ? 0x216 : 0x202] = 0xcc0010;
      context[0x10f] = context[0x110] = context[0x111] = context[0x112] =
          std::bit_cast<uint32_t>(2.0f);
      context[0x91] = gfx12 ? 0 : 1 | (1 << 16);
      context[0x30e] = context[0x30f] = 0xffffffff;
      for (uint32_t offset = 0; offset <= test.bytes; offset += 4)
        memory_.write32(0x100000 + offset, 0xcccccccc);

      auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
      for (uint32_t i = 0; i < 3; ++i)
        draw->export_lane(*wave_, i, 12, 15,
                          {std::bit_cast<uint32_t>(i == 2 ? 1.0f : -1.0f),
                           std::bit_cast<uint32_t>(i == 1 ? 1.0f : -1.0f), 0,
                           std::bit_cast<uint32_t>(1.0f)});
      draw->export_lane(*wave_, 0, 20, 1,
                        {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
      ASSERT_TRUE(draw->advance(*access_));
      wave_->set_wg_coord(0, 0, 0);
      wave_->set_graphics_stage(draw);
      draw->initialize(*wave_, 0, 0);
      for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
        draw->export_lane(*wave_, lane, 0, test.mask, test.exported);
      EXPECT_FALSE(draw->advance(*access_));
      for (uint32_t byte = 0; byte < test.bytes; ++byte) {
        const uint32_t component = byte / (test.bytes / 4);
        const uint8_t expected =
            write_mask & (1u << component) ? test.expected[byte / 4] >> (8 * (byte % 4)) : 0xcc;
        uint8_t actual = 0;
        ASSERT_EQ(access_->read(0x100000 + byte, {reinterpret_cast<std::byte *>(&actual), 1}),
                  amdgpu::VmAccessOutcome::Complete);
        EXPECT_EQ(actual, expected) << byte;
      }
      EXPECT_EQ(memory_.read32(0x100000 + test.bytes), 0xcccccccc);
    }
  }
}

TEST_P(GraphicsExportTest, MultipleAttachmentsKeepFormatsMasksAndBlendStateIndependent) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  for (bool disabled_first : {false, true})
    for (bool sparse : {false, true}) {
      for (bool blend : {false, true}) {
        SCOPED_TRACE(testing::Message() << "sparse=" << sparse << " disabled_first="
                                        << disabled_first << " blend=" << blend);
        const std::array<uint32_t, 3> targets =
            sparse ? std::array<uint32_t, 3>{1, 3, 7} : std::array<uint32_t, 3>{0, 1, 2};
        const std::array<uint32_t, 3> bytes{4, 8, 16};
        const std::array<uint32_t, 3> formats{10u, 12u | (7u << 8), 14u | (7u << 8)};
        const std::array<uint32_t, 3> export_formats{4u, 4u, 9u};
        const std::array<uint32_t, 3> masks{disabled_first ? 0u : 15u, 5, 10};
        const std::array<std::array<uint32_t, 4>, 3> expected{
            std::array<uint32_t, 4>{blend || disabled_first ? 0xccccccccu : 0xff0000ffu},
            std::array<uint32_t, 4>{0xcccc3400, 0xcccc3a00},
            std::array<uint32_t, 4>{0xcccccccc, 0x40400000, 0xcccccccc, 0x40a00000}};
        amdgpu::Pm4QueueState state;
        state.num_instances = 1;
        state.uconfig_registers[0x242] = 17;
        auto &context = state.context_registers;
        for (uint32_t i = 0; i < targets.size(); ++i) {
          const uint32_t target = targets[i], block = 0x318 + (gfx12 ? 9 : 15) * target;
          context[block] = 0x1000 + 0x100 * i;
          context[gfx12 ? 0x3b0 + target : block + 4] = formats[i];
          context[gfx12 ? block + 6 : 0x3b0 + target] = gfx12 ? 3 | (3 << 16) : 3 | (3 << 14);
          context[gfx12 ? 0x214 : 0x8e] |= masks[i] << (4 * target);
          context[gfx12 ? 0x195 : 0x1c5] |= export_formats[i] << (4 * i);
          context[gfx12 ? 0x215 : 0x8f] |= 15u << (4 * target);
          // ZERO * source + ONE * destination on only the first target.
          context[0x1e0 + target] = blend && i == 0 ? (1u << 30) | (1 << 8) : 0;
          for (uint32_t offset = 0; offset <= bytes[i]; offset += 4)
            memory_.write32(0x100000 + 0x10000 * i + offset, 0xcccccccc);
        }
        context[gfx12 ? 0x198 : 0x1b4] = 2;
        context[0x2f9] = 0x2d;
        context[gfx12 ? 0x205 : 0x206] = 0x43f;
        context[gfx12 ? 0x216 : 0x202] = 0xcc0010;
        context[0x10f] = context[0x110] = context[0x111] = context[0x112] =
            std::bit_cast<uint32_t>(2.0f);
        context[0x91] = gfx12 ? 0 : 1 | (1 << 16);
        context[0x30e] = context[0x30f] = 0xffffffff;
        // LESS plus a depth write must run once, before all color attachments.
        context[gfx12 ? 0x1c : 0x200] = 6 | (1 << 4);
        context[gfx12 ? 5 : 7] = 3 | (3 << 16);
        context[gfx12 ? 6 : 0x10] = 3 | ((gfx12 ? 3 : 24) << 4);
        context[gfx12 ? 8 : 0x12] = context[gfx12 ? 10 : 0x14] = 0x2000;
        context[gfx12 ? 0x116 : 0xb5] = std::bit_cast<uint32_t>(1.0f);
        context[0x113] = std::bit_cast<uint32_t>(1.0f);
        memory_.write32(0x200000, std::bit_cast<uint32_t>(1.0f));
        auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
        for (uint32_t i = 0; i < 3; ++i)
          draw->export_lane(*wave_, i, 12, 15,
                            {std::bit_cast<uint32_t>(i == 2 ? 1.0f : -1.0f),
                             std::bit_cast<uint32_t>(i == 1 ? 1.0f : -1.0f),
                             std::bit_cast<uint32_t>(0.5f), std::bit_cast<uint32_t>(1.0f)});
        draw->export_lane(*wave_, 0, 20, 1,
                          {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
        ASSERT_TRUE(draw->advance(*access_));
        wave_->set_wg_coord(0, 0, 0);
        wave_->set_graphics_stage(draw);
        draw->initialize(*wave_, 0, 0);
        for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
          draw->export_lane(*wave_, lane, 0, 3, {0x00003c00, 0x3c000000});
          draw->export_lane(*wave_, lane, 1, 3, {0x38003400, 0x3c003a00});
          draw->export_lane(*wave_, lane, 2, 15, {0x40000000, 0x40400000, 0x40800000, 0x40a00000});
        }
        EXPECT_FALSE(draw->advance(*access_));
        EXPECT_EQ(memory_.read32(0x200000), std::bit_cast<uint32_t>(0.5f));
        for (uint32_t i = 0; i < targets.size(); ++i) {
          for (uint32_t word = 0; word < bytes[i] / 4; ++word)
            EXPECT_EQ(memory_.read32(0x100000 + 0x10000 * i + 4 * word), expected[i][word])
                << "target " << targets[i] << " word " << word;
          EXPECT_EQ(memory_.read32(0x100000 + 0x10000 * i + bytes[i]), 0xcccccccc);
        }
      }
    }
}

TEST_P(GraphicsExportTest, MultipleAttachmentViewsClipLayersIndependently) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  amdgpu::Pm4QueueState state;
  state.num_instances = 1;
  state.uconfig_registers[0x242] = 17;
  auto &context = state.context_registers;
  const std::array<uint32_t, 2> targets{0, 7};
  const std::array<uint32_t, 2> views{0, 1 | (3u << (gfx12 ? 14 : 13))};
  for (uint32_t i = 0; i < targets.size(); ++i) {
    const uint32_t target = targets[i], block = 0x318 + (gfx12 ? 9 : 15) * target;
    context[block] = (0x100000 + i * 0x100000) >> 8;
    context[gfx12 ? 0x3b0 + target : block + 4] = 10;
    context[gfx12 ? block + 6 : 0x3b0 + target] = gfx12 ? 3 | (3 << 16) : 3 | (3 << 14);
    context[gfx12 ? block + 7 : 0x3b8 + target] = (gfx12 ? 3u << 15 : 26u << 14) | 3;
    // Target 0 has one layer; target 7 starts at layer 1 and has three layers.
    context[gfx12 ? block + 1 : block + 3] = views[i];
    context[gfx12 ? 0x214 : 0x8e] |= 15u << (4 * target);
    context[gfx12 ? 0x215 : 0x8f] |= 15u << (4 * target);
    context[gfx12 ? 0x195 : 0x1c5] |= 4u << (4 * i);
    for (uint32_t layer = 0; layer < 4; ++layer)
      memory_.write32(
          amdgpu::image_layer_base(gfx12, 0x100000 + i * 0x100000, 65536, layer, 4, gfx12 ? 3 : 26),
          0xcccccccc);
  }
  context[gfx12 ? 0x198 : 0x1b4] = 2;
  context[0x2f9] = 0x2d;
  context[gfx12 ? 0x205 : 0x206] = 0x43f;
  context[gfx12 ? 0x206 : 0x207] = 1u << 18;
  context[gfx12 ? 0x216 : 0x202] = 0xcc0010;
  context[0x10f] = context[0x110] = context[0x111] = context[0x112] = std::bit_cast<uint32_t>(2.0f);
  context[0x91] = gfx12 ? 0 : 1 | (1 << 16);
  context[0x30e] = context[0x30f] = 0xffffffff;
  auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
  for (uint32_t i = 0; i < 3; ++i) {
    draw->export_lane(*wave_, i, 12, 15,
                      {std::bit_cast<uint32_t>(i == 2 ? 1.0f : -1.0f),
                       std::bit_cast<uint32_t>(i == 1 ? 1.0f : -1.0f), 0,
                       std::bit_cast<uint32_t>(1.0f)});
    draw->export_lane(*wave_, i, 13, 4, {0, 0, 1, 0});
  }
  draw->export_lane(*wave_, 0, 20, 1,
                    {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
  ASSERT_TRUE(draw->advance(*access_));
  wave_->set_wg_coord(0, 0, 0);
  wave_->set_graphics_stage(draw);
  draw->initialize(*wave_, 0, 0);
  for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
    draw->export_lane(*wave_, lane, 0, 3, {0x00003c00, 0x3c000000});
    draw->export_lane(*wave_, lane, 1, 3, {0x3c000000, 0x3c000000});
  }
  EXPECT_FALSE(draw->advance(*access_));
  for (uint32_t i = 0; i < targets.size(); ++i)
    for (uint32_t layer = 0; layer < 4; ++layer) {
      const uint64_t address =
          amdgpu::image_layer_base(gfx12, 0x100000 + i * 0x100000, 65536, layer, 4, gfx12 ? 3 : 26);
      EXPECT_EQ(memory_.read32(address), targets[i] == 7 && layer == 2 ? 0xff00ff00 : 0xcccccccc)
          << "target " << targets[i] << " layer " << layer;
    }
}

TEST_P(GraphicsExportTest, ColorAttachmentWritesSelectedMipAndPreservesOtherLevels) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint32_t base = 0x400000;
  for (uint32_t level = 0; level < 8; ++level) {
    SCOPED_TRACE(level);
    const uint32_t width = std::max(1u, 200u >> level), height = std::max(1u, 180u >> level);
    for (uint32_t offset : kMipLastTexelOffsets[gfx12])
      memory_.write32(base + offset, 0x12345678);
    amdgpu::Pm4QueueState state;
    state.num_instances = 1;
    state.uconfig_registers[0x242] = 17;
    auto &context = state.context_registers;
    context[gfx12 ? 0x3b0 : 0x31c] = 10;
    context[gfx12 ? 0x31e : 0x3b0] = gfx12 ? 179 | (199 << 16) : 179 | (199 << 14) | (7u << 28);
    context[gfx12 ? 0x31f : 0x3b8] = gfx12 ? (3u << 15) | (7u << 19) : 27u << 14;
    context[gfx12 ? 0x31a : 0x31b] = gfx12 ? level : level << 26;
    context[0x318] = base >> 8;
    context[gfx12 ? 0x214 : 0x8e] = 15;
    context[gfx12 ? 0x215 : 0x8f] = 15;
    context[gfx12 ? 0x195 : 0x1c5] = 4;
    context[gfx12 ? 0x198 : 0x1b4] = 2;
    context[0x2f9] = 0x2d;
    context[gfx12 ? 0x205 : 0x206] = 0x43f;
    context[gfx12 ? 0x216 : 0x202] = 0xcc0010;
    context[0x10f] = context[0x110] = std::bit_cast<uint32_t>(width / 2.0f);
    context[0x111] = context[0x112] = std::bit_cast<uint32_t>(height / 2.0f);
    context[0x113] = context[gfx12 ? 0x116 : 0xb5] = std::bit_cast<uint32_t>(1.0f);
    context[0x90] = (width - 1) | ((height - 1) << 16);
    context[0x91] = (width - gfx12) | ((height - gfx12) << 16);
    context[0x30e] = context[0x30f] = 0xffffffff;

    auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
    for (uint32_t i = 0; i < 3; ++i)
      draw->export_lane(*wave_, i, 12, 15,
                        {std::bit_cast<uint32_t>(i == 2 ? 1.0f : -1.0f),
                         std::bit_cast<uint32_t>(i == 1 ? 1.0f : -1.0f), 0,
                         std::bit_cast<uint32_t>(1.0f)});
    draw->export_lane(*wave_, 0, 20, 1,
                      {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
    ASSERT_TRUE(draw->advance(*access_));
    wave_->set_wg_coord(0, 0, 0);
    wave_->set_graphics_stage(draw);
    draw->initialize(*wave_, 0, 0);
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
      draw->export_lane(*wave_, lane, 0, 3, {0x00003c00, 0x3c000000, 0, 0});
    EXPECT_FALSE(draw->advance(*access_));
    for (uint32_t mip = 0; mip < 8; ++mip)
      EXPECT_EQ(memory_.read32(base + kMipLastTexelOffsets[gfx12][mip]),
                mip == level ? 0xff0000ffu : 0x12345678u);
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

TEST_P(GraphicsExportTest, HardwareSampleFiltersAndAddressesRgba8) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const std::array<uint32_t, 8> descriptor{
      0x1000, (42u << (gfx12 ? 17 : 20)) | (1u << 30), 1u << 14, (9u << 28) | 0xfac, 0, 0, 0, 0};
  for (uint32_t r = 0; r < descriptor.size(); ++r)
    wave_->debug_write_sgpr(8 + r, descriptor[r]);
  memory_.write32(0x100000, 0x40fa0b00);
  memory_.write32(0x100004, 0x80c94dff);
  memory_.write32(gfx12 ? 0x100080 : 0x100100, 0xc0618eaa);
  memory_.write32(gfx12 ? 0x100084 : 0x100104, 0xff02db55);
  constexpr std::array<uint32_t, 4> first{0, 0x3d30b0b1, 0x3f7afafb, 0x3e808081};
  constexpr std::array<uint32_t, 4> second{0x3f800000, 0x3e9a9a9b, 0x3f49c9ca, 0x3f008081};
  struct Case {
    const char *name;
    uint32_t wrap;
    bool linear, unnormalized;
    uint32_t border;
    float u, v;
    std::array<uint32_t, 4> expected;
    bool srgb = false;
  };
  // Filtering results were captured with the same 2x2 texture on physical
  // gfx1100 and gfx1201. Compare raw floats, including the fractional precision.
  const Case cases[] = {
      {"center", 2, true, false, 0, 0.5f, 0.5f, {0x3f000000, 0x3ee16161, 0x3f0a0a0a, 0x3f206060}},
      {"fractional",
       2,
       true,
       false,
       0,
       0.25f + 1.0f / 512,
       0.25f,
       {0x3b800000, 0x3d34d4d5, 0x3f7ac9ca, 0x3e810101}},
      {"two fractional axes",
       2,
       true,
       false,
       0,
       0.25f + 255.0f / 512,
       0.25f + 255.0f / 512,
       {0x3eaca808, 0x3f5b0008, 0x3c4a3e3e, 0x3f7f4141}},
      {"fraction rounds even down", 2, true, false, 0, 0.25f + 1.0f / 1024, 0.25f, first},
      {"unnormalized center",
       2,
       true,
       true,
       0,
       1,
       1,
       {0x3f000000, 0x3ee16161, 0x3f0a0a0a, 0x3f206060}},
      {"nearest repeat", 0, false, false, 0, 1.25f, 0.25f, first},
      {"linear repeat", 0, true, false, 0, 1.25f, 0.25f, first},
      {"negative repeat", 0, true, false, 0, -0.25f, 0.25f, second},
      {"mirror repeat", 1, true, false, 0, 1.25f, 0.25f, second},
      {"negative mirror repeat", 1, false, false, 0, -0.25f, 0.25f, first},
      {"mirror once", 3, true, false, 0, -0.75f, 0.25f, second},
      {"edge clamp", 2, true, false, 0, -0.25f, 0.25f, first},
      {"transparent border", 6, false, false, 0, -0.25f, 0.25f, {0, 0, 0, 0}},
      {"opaque black border", 6, true, false, 1, -0.25f, 0.25f, {0, 0, 0, 0x3f800000}},
      {"white border",
       6,
       true,
       false,
       2,
       1.5f,
       0.25f,
       {0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000}},
      {"partial border",
       6,
       true,
       false,
       0,
       0,
       0.25f,
       {0, std::bit_cast<uint32_t>(5.5f / 255), std::bit_cast<uint32_t>(125.0f / 255),
        std::bit_cast<uint32_t>(32.0f / 255)}},
      {"unorm normalization boundary",
       2,
       true,
       false,
       0,
       0.25f + 151.0f / 512,
       0.25f,
       {0x3f170000, 0x3e488080, 0x3f5df6f7, 0x3ecc4c4c}},
      {"unorm two-axis normalization boundary",
       2,
       true,
       false,
       0,
       0.25f + 5.0f / 512,
       0.25f + 1.0f / 512,
       {0x3cb48080, 0x3d4da121, 0x3f796a83, 0x3e8403f4}},
      {"srgb texel center",
       2,
       true,
       false,
       0,
       0.25f,
       0.25f,
       {0, 0x3b5b0000, 0x3f750000, 0x3e808081},
       true},
      {"srgb edge alignment",
       2,
       true,
       false,
       0,
       0.25f + 1.0f / 512,
       0.25f,
       {0x3b800000, 0x3b6c2600, 0x3f74a100, 0x3e810101},
       true},
      {"srgb four texels",
       2,
       true,
       false,
       0,
       0.5f,
       0.5f,
       {0x3ebf2000, 0x3e86e800, 0x3ed4e000, 0x3f206060},
       true},
      {"srgb unequal weights",
       2,
       true,
       false,
       0,
       0.25f + 255.0f / 512,
       0.25f + 255.0f / 512,
       {0x3dc3b982, 0x3f33ee5e, 0x3b54a080, 0x3f7f4141},
       true},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.name);
    wave_->debug_write_sgpr(9, ((test.srgb ? 66u : 42u) << (gfx12 ? 17 : 20)) | (1u << 30));
    wave_->debug_write_sgpr(4, test.wrap | (test.wrap << 3) | (2 << 6) |
                                   (uint32_t(test.unnormalized) << 15));
    wave_->debug_write_sgpr(5, 0);
    wave_->debug_write_sgpr(6, test.linear ? (1 << 20) | (1 << 22) : 0);
    wave_->debug_write_sgpr(7, test.border << 30);
    wave_->set_exec(5);
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
      wave_->debug_write_vgpr(8, lane, std::bit_cast<uint32_t>(test.u));
      wave_->debug_write_vgpr(9, lane, std::bit_cast<uint32_t>(test.v));
      wave_->debug_write_vgpr(10, lane, 0xdeadbeef);
      wave_->debug_write_vgpr(11, lane, 0xdeadbeef);
    }
    std::array<uint32_t, 4> words{0xe7c6c001, 0x02001008, 0x00000908, 0};
    if (!gfx12) {
      const auto mimg = rdna3::build_mimg(
          31, {.nsa = 1, .dim = 1, .dmask = 15, .vaddr = 8, .vdata = 8, .srsrc = 2, .ssamp = 1});
      std::copy(mimg.begin(), mimg.end(), words.begin());
      words[2] = 9;
    }
    auto decoded = decoder_->decode(words.data());
    ASSERT_FALSE(decoded.failed());
    auto instruction = std::move(decoded).value();
    ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
    ASSERT_FALSE(wave_->instruction_execution_failed());
    ASSERT_NE(instruction->data(), nullptr);
    EXPECT_EQ(instruction->data_as<amdgpu::VectorMemState>()->wait_counter_type,
              gfx12 ? amdgpu::WaitCounterType::SAMPLECNT : amdgpu::WaitCounterType::LOADCNT);
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
    pipeline.issue(instruction.release(), *wave_);
    for (uint32_t c = 0; c < 4; ++c) {
      EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 0), test.expected[c]) << c;
      EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 2), test.expected[c]) << c;
    }
    EXPECT_EQ(wave_->debug_read_vgpr(8, 1), std::bit_cast<uint32_t>(test.u));
    EXPECT_EQ(wave_->debug_read_vgpr(9, 1), std::bit_cast<uint32_t>(test.v));
    EXPECT_EQ(wave_->debug_read_vgpr(10, 1), 0xdeadbeefu);
    EXPECT_EQ(wave_->debug_read_vgpr(11, 1), 0xdeadbeefu);
  }
}

TEST_P(GraphicsExportTest, SampleLodUsesMipViewsDerivativesAndBias) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t colors[] = {0xff000000, 0xff0000ff, 0xff00ff00, 0xffff0000};
  for (uint32_t level = 0; level < 4; ++level) {
    const auto mip = amdgpu::image_mip_layout(gfx12, 0, 4, 8, 8, 4, level);
    ASSERT_TRUE(mip);
    for (uint32_t y = 0; y < mip->height; ++y)
      for (uint32_t x = 0; x < mip->width; ++x) {
        const auto address =
            gfx12 ? amdgpu::gfx12_image_address(0x100000 + mip->offset, x, y, mip->pitch, 4, 0)
                  : amdgpu::gfx11_image_address(0x100000 + mip->offset, x, y, mip->pitch, 4, 0);
        ASSERT_TRUE(address);
        memory_.write32(*address, colors[level]);
      }
  }
  struct Case {
    const char *name;
    uint32_t opcode, mip_filter, first_level;
    float lod, min_lod, max_lod, coordinate_step;
    std::array<float, 4> expected;
    bool a16 = false;
  };
  const Case cases[] = {
      {"explicit nearest", 29, 1, 0, 1.25f, 0, 3, 0, {1, 0, 0, 1}},
      {"explicit rounded up", 29, 1, 0, 1.75f, 0, 3, 0, {0, 1, 0, 1}},
      {"below nearest threshold", 29, 1, 0, 0.498046875f - 1.0f / 8192, 0, 3, 0, {0, 0, 0, 1}},
      {"rounded nearest threshold", 29, 1, 0, 0.498046875f, 0, 3, 0, {1, 0, 0, 1}},
      {"mip interpolation", 29, 2, 0, 1.5f, 0, 3, 0, {0.5f, 0.5f, 0, 1}},
      {"minimum LOD", 29, 1, 0, 0, 2, 3, 0, {0, 1, 0, 1}},
      {"maximum LOD", 29, 1, 0, 3, 0, 1, 0, {1, 0, 0, 1}},
      {"view base", 29, 1, 1, 0, 0, 3, 0, {1, 0, 0, 1}},
      {"view upper bound", 29, 1, 1, 3, 0, 3, 0, {0, 0, 1, 1}},
      {"explicit derivatives", 28, 1, 0, 0, 0, 3, 0.5f, {0, 1, 0, 1}},
      {"implicit derivatives", 27, 1, 0, 0, 0, 3, 0.5f, {0, 1, 0, 1}},
      {"shader bias", 30, 1, 0, 1, 0, 3, 0.25f, {0, 1, 0, 1}},
      {"forced zero", 31, 1, 0, 0, 0, 3, 0.5f, {0, 0, 0, 1}},
      {"packed explicit LOD", 29, 2, 0, 1.5f, 0, 3, 0, {0.5f, 0.5f, 0, 1}, true},
      {"packed coordinates with full derivatives", 28, 1, 0, 0, 0, 3, 0.5f, {0, 1, 0, 1}, true},
      {"packed implicit coordinates", 27, 1, 0, 0, 0, 3, 0.5f, {0, 1, 0, 1}, true},
      {"packed coordinates with full bias", 30, 1, 0, 1, 0, 3, 0.25f, {0, 1, 0, 1}, true},
      {"packed forced zero", 31, 1, 0, 0, 0, 3, 0.5f, {0, 0, 0, 1}, true},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.name);
    const std::array<uint32_t, 8> descriptor{
        0x1000,
        (42u << (gfx12 ? 17 : 20)) | (3u << 30) | (3u << (gfx12 ? 12 : 16)) |
            (gfx12 ? test.first_level << 25 : 0),
        1 | (7u << 14),
        (9u << 28) | 0xfac | (3u << (gfx12 ? 15 : 16)) | (gfx12 ? 0 : test.first_level << 12),
        0,
        0,
        0,
        0};
    for (uint32_t r = 0; r < descriptor.size(); ++r)
      wave_->debug_write_sgpr(8 + r, descriptor[r]);
    wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6));
    wave_->debug_write_sgpr(5, uint32_t(test.min_lod * 256) |
                                   (uint32_t(test.max_lod * 256) << (gfx12 ? 13 : 12)));
    wave_->debug_write_sgpr(6, test.mip_filter << 26);
    wave_->debug_write_sgpr(7, 0);
    wave_->set_exec(9);
    // Populate inactive quad lanes as well: implicit derivatives consume them.
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
      const float u = 0.25f + (lane & 1) * test.coordinate_step;
      const float v = 0.25f + ((lane >> 1) & 1) * test.coordinate_step;
      const std::array<uint32_t, 6> address_regs{6, 0, 4, 8, 9, 10};
      std::array<float, 6> values{u, v, test.lod};
      if (test.opcode == 28)
        values = {test.coordinate_step, 0, 0, test.coordinate_step, u, v};
      if (test.opcode == 30)
        values = {test.lod, u, v};
      for (uint32_t r = 0; r < values.size(); ++r)
        wave_->debug_write_vgpr(address_regs[r], lane, std::bit_cast<uint32_t>(values[r]));
      if (test.a16) {
        const uint32_t prefix = test.opcode == 28 ? 4 : test.opcode == 30 ? 1 : 0;
        wave_->debug_write_vgpr(address_regs[prefix], lane,
                                util::f32_to_f16(values[prefix]) |
                                    (uint32_t(util::f32_to_f16(values[prefix + 1])) << 16));
        if (test.opcode == 29)
          wave_->debug_write_vgpr(address_regs[1], lane, util::f32_to_f16(test.lod));
      }
    }
    std::array<uint32_t, 4> words{};
    const uint8_t second_address = test.a16 && (test.opcode == 27 || test.opcode == 31) ? 255 : 0;
    if (gfx12) {
      const auto encoded = rdna4::build_vsample(test.opcode, {.dim = 1,
                                                              .a16 = test.a16,
                                                              .dmask = 15,
                                                              .vdata = 6,
                                                              .rsrc = 8,
                                                              .samp = 4,
                                                              .vaddr0 = 6,
                                                              .vaddr1 = second_address,
                                                              .vaddr2 = 4,
                                                              .vaddr3 = 8});
      std::copy(encoded.begin(), encoded.end(), words.begin());
    } else {
      const auto encoded = rdna3::build_mimg(test.opcode, {.nsa = 1,
                                                           .dim = 1,
                                                           .dmask = 15,
                                                           .a16 = test.a16,
                                                           .vaddr = 6,
                                                           .vdata = 6,
                                                           .srsrc = 2,
                                                           .ssamp = 1});
      std::copy(encoded.begin(), encoded.end(), words.begin());
      words[2] = second_address | (4 << 8) | (8 << 16) | (9 << 24);
    }
    auto decoded = decoder_->decode(words.data());
    ASSERT_FALSE(decoded.failed());
    auto instruction = std::move(decoded).value();
    ASSERT_TRUE(instruction->is_memory_op());
    ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
    ASSERT_FALSE(wave_->instruction_execution_failed());
    ASSERT_NE(instruction->data(), nullptr);
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
    pipeline.issue(instruction.release(), *wave_);
    for (uint32_t lane : {0u, 3u})
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(6 + c, lane), std::bit_cast<uint32_t>(test.expected[c]))
            << lane << "," << c;
  }
}

TEST_P(GraphicsExportTest, SampledArrayViewsClampLayersAndKeepMipCoordinatesSeparate) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  std::optional<uint64_t> (*image_address)(uint64_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                           uint32_t) =
      gfx12 ? amdgpu::gfx12_image_address : amdgpu::gfx11_image_address;
  for (uint32_t layer = 0; layer < 5; ++layer)
    for (uint32_t level = 0; level < 4; ++level) {
      const auto mip = amdgpu::image_mip_layout(gfx12, 0, 4, 8, 8, 4, level);
      ASSERT_TRUE(mip);
      for (uint32_t y = 0; y < mip->height; ++y)
        for (uint32_t x = 0; x < mip->width; ++x)
          memory_.write32(*image_address(0x100000 + layer * mip->slice_size + mip->offset, x, y,
                                         mip->pitch, 4, 0),
                          0xff000000 | (layer << 8) | level);
    }
  for (bool a16 : {false, true})
    for (bool array : {false, true})
      for (const auto [opcode, name] : {std::pair{27u, "IMAGE_SAMPLE"},
                                        {28u, "IMAGE_SAMPLE_D"},
                                        {29u, "IMAGE_SAMPLE_L"},
                                        {30u, "IMAGE_SAMPLE_B"},
                                        {31u, "IMAGE_SAMPLE_LZ"}}) {
        SCOPED_TRACE(testing::Message()
                     << "a16=" << a16 << ", array=" << array << ", opcode=" << name);
        const std::array<uint32_t, 8> descriptor{0x1000,
                                                 (46u << (gfx12 ? 17 : 20)) | (3u << 30) |
                                                     (3u << (gfx12 ? 12 : 16)),
                                                 1 | (7u << 14),
                                                 (13u << 28) | 0xfac | (3u << (gfx12 ? 15 : 16)),
                                                 (2u << 16) | 4,
                                                 0,
                                                 0,
                                                 0};
        for (uint32_t r = 0; r < descriptor.size(); ++r)
          wave_->debug_write_sgpr(8 + r, descriptor[r]);
        wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6));
        wave_->debug_write_sgpr(5, 768u << (gfx12 ? 13 : 12));
        wave_->debug_write_sgpr(6, 1u << 26);
        wave_->debug_write_sgpr(7, 0);
        wave_->set_exec(15);
        const uint32_t prefix = opcode == 28 ? 4 : opcode == 30 ? 1 : 0;
        const std::array<uint32_t, 7> registers{6, 0, 4, 8, 9, 10, 11};
        for (uint32_t lane = 0; lane < 4; ++lane) {
          std::array<float, 7> values{};
          if (opcode == 28)
            values = {0.5f, 0, 0, 0.5f};
          if (opcode == 30)
            values[0] = 1;
          values[prefix] = (lane & 1) * 0.5f;
          values[prefix + 1] = ((lane >> 1) & 1) * 0.5f;
          if (array)
            values[prefix + 2] = lane == 0 ? -100.0f : lane == 3 ? 100.0f : 1.0f;
          if (opcode == 29)
            values[prefix + 2 + array] = 1.0f;
          for (uint32_t i = 0; i < values.size(); ++i)
            wave_->debug_write_vgpr(registers[i], lane, std::bit_cast<uint32_t>(values[i]));
          if (a16) {
            const uint32_t count = 2 + array + (opcode == 29);
            for (uint32_t i = 0; i < count; i += 2)
              wave_->debug_write_vgpr(
                  registers[prefix + i / 2], lane,
                  util::f32_to_f16(values[prefix + i]) |
                      (uint32_t(util::f32_to_f16(i + 1 < count ? values[prefix + i + 1] : 0))
                       << 16));
          }
        }
        std::array<uint32_t, 4> words{};
        if (gfx12) {
          const auto encoded = rdna4::build_vsample(opcode, {.dim = uint8_t(array ? 5 : 1),
                                                             .a16 = a16,
                                                             .dmask = 15,
                                                             .vdata = 12,
                                                             .rsrc = 8,
                                                             .samp = 4,
                                                             .vaddr0 = 6,
                                                             .vaddr1 = 0,
                                                             .vaddr2 = 4,
                                                             .vaddr3 = 8});
          std::copy(encoded.begin(), encoded.end(), words.begin());
        } else {
          const auto encoded = rdna3::build_mimg(opcode, {.nsa = 1,
                                                          .dim = uint8_t(array ? 5 : 1),
                                                          .dmask = 15,
                                                          .a16 = a16,
                                                          .vaddr = 6,
                                                          .vdata = 12,
                                                          .srsrc = 2,
                                                          .ssamp = 1});
          std::copy(encoded.begin(), encoded.end(), words.begin());
          words[2] = (4 << 8) | (8 << 16) | (9 << 24);
        }
        auto decoded = decoder_->decode(words.data());
        ASSERT_FALSE(decoded.failed());
        auto instruction = std::move(decoded).value();
        ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
        ASSERT_FALSE(wave_->instruction_execution_failed());
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
        pipeline.issue(instruction.release(), *wave_);
        for (uint32_t lane = 0; lane < 4; ++lane) {
          const uint32_t level = opcode == 31 ? 0 : opcode == 29 ? 1 : opcode == 30 ? 3 : 2;
          const uint32_t layer = !array || lane == 0 ? 2 : lane == 3 ? 4 : 3;
          EXPECT_EQ(wave_->debug_read_vgpr(12, lane), level);
          EXPECT_EQ(wave_->debug_read_vgpr(13, lane), layer);
          EXPECT_EQ(wave_->debug_read_vgpr(14, lane), 0u);
          EXPECT_EQ(wave_->debug_read_vgpr(15, lane), 255u);
        }
      }
}

TEST_P(GraphicsExportTest, TrilinearUsesDistinctWeightsForNonuniformMipLevels) {
  for (bool translated : {false, true}) {
    SCOPED_TRACE(translated);
    if (translated) {
      const auto address_space =
          vm_.register_address_space(1, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(),
                                     std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory_));
      ASSERT_TRUE(address_space);
      wave_->set_address_space(address_space);
    }
    for (bool srgb : {false, true}) {
      SCOPED_TRACE(srgb);
      // The host refills backing memory directly between image fixtures.
      cu_->l1_vector().invalidate_all();
      cache_.invalidate_all();
      const uint32_t width = srgb ? 7 : 13, height = srgb ? 5 : 9, levels = srgb ? 3 : 4;
      const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
      for (uint32_t level = 0; level < levels; ++level) {
        const auto mip = amdgpu::image_mip_layout(gfx12, 0, 4, width, height, levels, level);
        ASSERT_TRUE(mip);
        for (uint32_t y = 0; y < mip->height; ++y)
          for (uint32_t x = 0; x < mip->width; ++x) {
            const auto address =
                gfx12 ? amdgpu::gfx12_image_address(0x100000 + mip->offset, x, y, mip->pitch, 4, 0)
                      : amdgpu::gfx11_image_address(0x100000 + mip->offset, x, y, mip->pitch, 4, 0);
            ASSERT_TRUE(address);
            uint32_t color = (17 * x + 31 * y + 53 * level) |
                             ((13 * x + 47 * y + 19 * level) << 8) |
                             ((11 * x + 23 * y + 29 * level) << 16) | 0xff000000;
            if (!srgb) {
              color = (x + 13 * y + 117 * level) * 1664525u + 1013904223u;
              color ^= color >> 16;
              color *= 2246822519u;
            }
            memory_.write32(*address, color);
          }
      }
      const std::array<uint32_t, 8> descriptor{
          0x1000,
          ((srgb ? 66u : 42u) << (gfx12 ? 17 : 20)) | (((width - 1) & 3) << 30) |
              ((levels - 1) << (gfx12 ? 12 : 16)),
          ((width - 1) >> 2) | ((height - 1) << 14),
          (9u << 28) | 0xfac | ((levels - 1) << (gfx12 ? 15 : 16)),
          0,
          0,
          0,
          0};
      for (uint32_t r = 0; r < descriptor.size(); ++r)
        wave_->debug_write_sgpr(8 + r, descriptor[r]);
      wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6));
      wave_->debug_write_sgpr(5, ((levels - 1) * 256) << (gfx12 ? 13 : 12));
      wave_->debug_write_sgpr(6, (1 << 20) | (1 << 22) | (2 << 26));
      wave_->debug_write_sgpr(7, 0);
      wave_->set_exec(1);
      struct Case {
        uint32_t index;
        std::array<uint32_t, 4> expected;
      };
      // Physical gfx1100 and gfx1201 results for UV=(index%256,index/256)/256,
      // LOD=(index%513)/256. All eight taps contribute at the interior coordinates.
      const std::array<Case, 5> srgb_cases{{
          {343, {0x3da6ca8a, 0x3c5f2280, 0x3cd123dc, 0x3f800000}},
          {6764, {0x3d122bb4, 0x3c6a7780, 0x3c6a0f81, 0x3f800000}},
          {18374, {0x3e011439, 0x3cb9adb7, 0x3d1f1fdd, 0x3f800000}},
          {32467, {0x3e557300, 0x3e40d053, 0x3db0d1b7, 0x3f800000}},
          {49141, {0x3e2372b6, 0x3d62df4e, 0x3d58d6c0, 0x3f800000}},
      }};
      // Independent hardware captures use pseudorandom RGBA texels and exercise
      // rounding of each weighted mip before the final texel-value rounding.
      const std::array<Case, 5> unorm_cases{{
          {239, {0x3e83a808, 0x3eecb969, 0x3f113f6f, 0x3e9cd8d9}},
          {612, {0x3e86c6a7, 0x3ed3f7c8, 0x3ef3c9fa, 0x3ec74959}},
          {1344, {0x3f0acd1d, 0x3ef5b707, 0x3f3414c5, 0x3f18ab03}},
          {1769, {0x3eda8d2d, 0x3ebf60f1, 0x3f24469f, 0x3effae4e}},
          {2099, {0x3ed0b6e7, 0x3ee7aacb, 0x3ed90a0a, 0x3f18b8c1}},
      }};
      for (const auto &test : srgb ? srgb_cases : unorm_cases) {
        SCOPED_TRACE(test.index);
        wave_->debug_write_vgpr(0, 0, std::bit_cast<uint32_t>((test.index % 256) / 256.0f));
        wave_->debug_write_vgpr(1, 0, std::bit_cast<uint32_t>((test.index / 256) / 256.0f));
        wave_->debug_write_vgpr(2, 0, std::bit_cast<uint32_t>((test.index % 513) / 256.0f));
        if (!srgb) {
          uint32_t seed = test.index * 1664525u + 1013904223u;
          seed ^= seed >> 16;
          seed *= 2246822519u;
          wave_->debug_write_vgpr(0, 0, std::bit_cast<uint32_t>((seed & 1023) / 512.0f - 0.5f));
          wave_->debug_write_vgpr(1, 0,
                                  std::bit_cast<uint32_t>(((seed >> 10) & 1023) / 512.0f - 0.5f));
          wave_->debug_write_vgpr(2, 0, std::bit_cast<uint32_t>(((seed >> 20) % 2049) / 512.0f));
        }
        std::array<uint32_t, 4> words{};
        if (gfx12) {
          const auto encoded = rdna4::build_vsample(29, {.dim = 1,
                                                         .dmask = 15,
                                                         .vdata = 4,
                                                         .rsrc = 8,
                                                         .samp = 4,
                                                         .vaddr0 = 0,
                                                         .vaddr1 = 1,
                                                         .vaddr2 = 2});
          std::copy(encoded.begin(), encoded.end(), words.begin());
        } else {
          const auto encoded = rdna3::build_mimg(
              29, {.dim = 1, .dmask = 15, .vaddr = 0, .vdata = 4, .srsrc = 2, .ssamp = 1});
          std::copy(encoded.begin(), encoded.end(), words.begin());
        }
        auto decoded = decoder_->decode(words.data());
        ASSERT_FALSE(decoded.failed());
        auto instruction = std::move(decoded).value();
        ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
        ASSERT_FALSE(wave_->instruction_execution_failed());
        ASSERT_NE(instruction->data(), nullptr);
        const auto *state = instruction->data_as<amdgpu::VectorMemState>();
        ASSERT_NE(state->image_sample, nullptr);
        EXPECT_EQ(state->image_sample->tap_count, 8u);
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
        pipeline.issue(instruction.release(), *wave_);
        for (uint32_t c = 0; c < 4; ++c) {
          const auto actual = wave_->debug_read_vgpr(4 + c, 0);
          EXPECT_EQ(actual, test.expected[c]) << c;
        }
      }
    }
  }
}

TEST_P(GraphicsExportTest, DepthClearAndComparisonsUseTiledD16AndD32) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    const char *name;
    float vertex_z = 0, scale = 1, offset = 0, minimum = 0, maximum = 1;
    bool disable_clamp = false;
    float expected = 0;
    uint16_t expected_d16 = 0;
    bool disable_samples = false;
    bool fragment_export = false;
    uint64_t export_mask = ~uint64_t{0};
  };
  constexpr Case cases[] = {
      {"zero depth", 0, 1, 0, 0, 1, false, 0, 0},
      {"far depth clamped", 2, 0.5f, 0.25f, 0.25f, 0.75f, false, 0.75f, 49151},
      {"far depth unclamped", 2, 0.5f, 0.25f, 0.25f, 0.75f, true, 1.25f, 65535},
      {"near depth clamped", -2, 0.5f, 0.25f, 0.25f, 0.75f, false, 0.25f, 16384},
      {"near depth unclamped", -2, 0.5f, 0.25f, 0.25f, 0.75f, true, -0.75f, 0},
      {.name = "sample disabled", .disable_samples = true},
      {.name = "zero-component export", .fragment_export = true},
      {.name = "discard all pixels", .fragment_export = true, .export_mask = 0},
      {.name = "discard odd columns", .fragment_export = true, .export_mask = 0x5555555555555555},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.name);
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
        state.context_registers[0x30e] = state.context_registers[0x30f] =
            test.disable_samples ? 0 : 0xffffffffu;
        state.context_registers[0x205] = 0x43f;
        state.context_registers[0x10f] = state.context_registers[0x110] =
            state.context_registers[0x111] = state.context_registers[0x112] =
                std::bit_cast<uint32_t>(2.0f);
        state.context_registers[0x113] = std::bit_cast<uint32_t>(test.scale);
        state.context_registers[0x114] = std::bit_cast<uint32_t>(test.offset);
        state.context_registers[gfx12 ? 0x115 : 0xb4] = std::bit_cast<uint32_t>(test.minimum);
        state.context_registers[gfx12 ? 0x116 : 0xb5] = std::bit_cast<uint32_t>(test.maximum);
        state.context_registers[gfx12 ? 0x19 : 3] =
            test.disable_clamp ? (gfx12 ? 1u : 1u << 16) : 0;
        state.context_registers[0x204] = (1u << 26) | (1u << 27);
        state.context_registers[0x90] = 1 | (1 << 16);
        state.context_registers[0x91] = (3 - gfx12) | ((3 - gfx12) << 16);
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
            memory_.write_block(*address, {reinterpret_cast<const uint8_t *>(&value), bytes});
          }
        state.context_registers[gfx12 ? 0x215 : 0x8f] = 15;
        auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
        for (uint32_t i = 0; i < 3; ++i)
          draw->export_lane(*wave_, i, 12, 15,
                            {std::bit_cast<uint32_t>(i == 2 ? 1.0f : -1.0f),
                             std::bit_cast<uint32_t>(i == 1 ? 1.0f : -1.0f),
                             std::bit_cast<uint32_t>(test.vertex_z),
                             std::bit_cast<uint32_t>(1.0f)});
        draw->export_lane(*wave_, 0, 20, 1,
                          {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
        const auto dispatch = draw->advance(*access_);
        ASSERT_EQ(bool(dispatch), !test.disable_samples);
        // A zero-component export still carries pixel validity for late depth.
        if (dispatch) {
          if (test.fragment_export) {
            wave_->set_wg_coord(0, 0, 0);
            wave_->set_graphics_stage(draw);
            draw->initialize(*wave_, 0, 0);
            wave_->set_exec(wave_->exec() & test.export_mask);
            execute(0, 0);
          }
          EXPECT_FALSE(draw->advance(*access_));
        }
        const float expected = bytes == 2 ? test.expected_d16 / 65535.0f : test.expected;
        const bool comparisons[] = {false,        expected < 1,  expected == 1, expected <= 1,
                                    expected > 1, expected != 1, expected >= 1, true};
        const bool pass = !test.disable_samples && comparisons[comparison];
        const uint32_t expected_bits =
            bytes == 2 ? test.expected_d16 : std::bit_cast<uint32_t>(expected);
        for (uint32_t y = 0; y < 4; ++y)
          for (uint32_t x = 0; x < 4; ++x) {
            const auto address = gfx12 ? amdgpu::gfx12_image_address(0x200000, x, y, 4, bytes, 3)
                                       : amdgpu::gfx11_image_address(0x200000, x, y, 4, bytes, 24);
            uint32_t actual = 0;
            ASSERT_EQ(access_->read(*address, {reinterpret_cast<std::byte *>(&actual), bytes}),
                      amdgpu::VmAccessOutcome::Complete);
            const bool survives =
                !test.fragment_export || (test.export_mask & (uint64_t{1} << (x & 1)));
            const bool changed = pass && survives && x >= 1 && x < 3 && y >= 1 && y < 3;
            EXPECT_EQ(actual, changed      ? expected_bits
                              : bytes == 2 ? 65535
                                           : std::bit_cast<uint32_t>(1.0f))
                << bytes << "," << comparison << "," << x << "," << y;
          }
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

TEST_P(GraphicsExportTest, MetadataAttachmentsPreserveDrawsAndRejectedState) {
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP();
  struct Case {
    const char *name;
    bool valid_viewport = true;
    float min_depth = 0;
    float max_depth = 1;
    bool scratch = false;
    bool reject = false;
  };
  const Case cases[] = {
      {.name = "valid state"},
      {.name = "invalid viewport", .valid_viewport = false, .reject = true},
      {.name = "nonfinite depth range",
       .max_depth = std::bit_cast<float>(0x7fc00000u),
       .reject = true},
      {.name = "inverted depth range", .min_depth = 1, .max_depth = 0, .reject = true},
      {.name = "fragment scratch", .scratch = true, .reject = true},
  };
  for (uint32_t target : {0u, 7u})
    for (bool aligned : {true, false}) {
      for (const auto &test : cases) {
        SCOPED_TRACE(testing::Message() << "target=" << target << ", pipe_aligned=" << aligned);
        SCOPED_TRACE(test.name);
        constexpr uint32_t width = 17, height = 13;
        constexpr uint64_t color_base = 0x200000, dcc = 0x100000;
        constexpr uint64_t depth_base = 0x400000, htile = 0x300000;
        for (uint32_t i = 0; i < 16384; ++i)
          memory_.write8(dcc + i, 0xff);
        for (uint32_t y = 0; y < height; y += 8) {
          for (uint32_t x = 0; x < width; x += 8) {
            memory_.write8(
                *amdgpu::gfx11_metadata_address(dcc, x, y, width, height, 4, 27, false, aligned),
                8);
            memory_.write32(
                *amdgpu::gfx11_metadata_address(htile, x, y, width, height, 4, 24, true),
                0x55555550);
          }
        }
        for (uint32_t y = 0; y < height; ++y) {
          for (uint32_t x = 0; x < width; ++x) {
            memory_.write32(*amdgpu::gfx11_image_address(color_base, x, y, width, 4, 27),
                            0x12345678);
            memory_.write32(*amdgpu::gfx11_image_address(depth_base, x, y, width, 4, 24),
                            0x12345678);
          }
        }
        amdgpu::Pm4QueueState state;
        state.num_instances = 1;
        state.uconfig_registers[0x242] = 17;
        auto &context = state.context_registers;

        const uint32_t block = 0x318 + 15 * target;
        context[block + 4] = 10;
        context[0x3b0 + target] = (height - 1) | ((width - 1) << 14);
        context[0x3b8 + target] = (27 << 14) | (uint32_t(aligned) << 30);
        context[block + 6] = 1u << 22;
        context[block + 13] = dcc >> 8;
        context[block] = color_base >> 8;
        context[0x8e] = 15u << (4 * target);
        context[0x8f] = 15u << (4 * target);
        context[0x1c5] = 4;
        context[0x1b4] = 2;
        context[0x2f9] = 0x2d;
        context[0x206] = test.valid_viewport ? 0x43f : 0;
        context[0x202] = 0xcc0010;
        context[0x10f] = context[0x110] = std::bit_cast<uint32_t>(width / 2.0f);
        context[0x111] = context[0x112] = std::bit_cast<uint32_t>(height / 2.0f);
        context[0x113] = std::bit_cast<uint32_t>(1.0f);
        context[0xb4] = std::bit_cast<uint32_t>(test.min_depth);
        context[0xb5] = std::bit_cast<uint32_t>(test.max_depth);
        state.sh_registers[0xb] = test.scratch;
        context[0x90] = 16 | (8 << 16);
        context[0x91] = 17 | (9 << 16);
        context[0x30e] = context[0x30f] = 0xffffffff;
        context[0x200] = 6 | (1 << 4); // Depth test LESS and write enabled.
        context[7] = (width - 1) | ((height - 1) << 16);
        context[0x10] = 3 | (24 << 4) | (1u << 29); // D32 with HTILE.
        context[0x12] = context[0x14] = depth_base >> 8;
        context[0x2af] = 1u << 18;
        context[5] = htile >> 8;
        context[0xb] = std::bit_cast<uint32_t>(0.375f);

        auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
        for (uint32_t i = 0; i < 3; ++i)
          draw->export_lane(*wave_, i, 12, 15,
                            {std::bit_cast<uint32_t>(i == 2 ? 1.0f : -1.0f),
                             std::bit_cast<uint32_t>(i == 1 ? 1.0f : -1.0f),
                             std::bit_cast<uint32_t>(0.25f), std::bit_cast<uint32_t>(1.0f)});
        draw->export_lane(*wave_, 0, 20, 1, {(1u << 10) | (2u << 20), 0, 0, 0});
        const auto color_address = *amdgpu::gfx11_image_address(color_base, 16, 8, width, 4, 27);
        const auto depth_address = *amdgpu::gfx11_image_address(depth_base, 16, 8, width, 4, 24);
        const auto color_key =
            *amdgpu::gfx11_metadata_address(dcc, 16, 8, width, height, 4, 27, false, aligned);
        const auto depth_key =
            *amdgpu::gfx11_metadata_address(htile, 16, 8, width, height, 4, 24, true);
        if (test.reject) {
          EXPECT_THROW(draw->advance(*access_), std::runtime_error);
          EXPECT_EQ(memory_.read32(color_address), 0x12345678u);
          EXPECT_EQ(memory_.read32(depth_address), 0x12345678u);
          EXPECT_EQ(memory_.read8(color_key), 8);
          EXPECT_EQ(memory_.read32(depth_key), 0x55555550u);
          continue;
        }
        ASSERT_TRUE(draw->advance(*access_));
        EXPECT_EQ(memory_.read32(color_address), 0xff000000u);
        EXPECT_EQ(memory_.read32(depth_address), std::bit_cast<uint32_t>(0.375f));
        EXPECT_EQ(memory_.read8(color_key), 0xff);
        EXPECT_EQ(memory_.read32(depth_key), 0xfffc000fu);
        wave_->set_wg_coord(0, 0, 0);
        wave_->set_graphics_stage(draw);
        draw->initialize(*wave_, 0, 0);
        for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
          draw->export_lane(*wave_, lane, 0, 3, {0x00003c00, 0x3c000000, 0, 0});
        EXPECT_FALSE(draw->advance(*access_));
        EXPECT_EQ(memory_.read32(color_address), 0xff0000ffu);
        EXPECT_EQ(memory_.read32(depth_address), std::bit_cast<uint32_t>(0.25f));
        amdgpu::materialize_gfx11_dcc(*access_, color_base, dcc, 16, 8, width, height, 4, 27,
                                      aligned);
        amdgpu::materialize_gfx11_htile(*access_, depth_base, htile, 16, 8, width, height, 4, 24);
        EXPECT_EQ(memory_.read32(color_address), 0xff0000ffu);
        EXPECT_EQ(memory_.read32(depth_address), std::bit_cast<uint32_t>(0.25f));
      }
    }
}

INSTANTIATE_TEST_SUITE_P(Rdna, GraphicsExportTest,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));

} // namespace

TEST(GraphicsImageMetadataTest, AddressesMatchAddrLibAcrossMetadataBlocks) {
  struct Case {
    uint32_t swizzle, bytes, width, x, y, pipe_xor, expected;
    bool depth, pipe_aligned;
  };
  const Case cases[] = {
      {27, 1, 420, 419, 319, 7, 1102, false, false},
      {27, 1, 4093, 3001, 2049, 31, 43589, false, false},
      {27, 1, 420, 419, 319, 7, 5133, false, true},
      {27, 1, 4093, 3001, 2049, 31, 49173, false, true},
      {27, 2, 420, 419, 319, 7, 413, false, false},
      {27, 2, 4093, 3001, 2049, 31, 75146, false, false},
      {27, 2, 420, 419, 319, 7, 5147, false, true},
      {27, 2, 4093, 3001, 2049, 31, 81962, false, true},
      {27, 4, 420, 419, 319, 7, 2618, false, false},
      {27, 4, 4093, 3001, 2049, 31, 154133, false, false},
      {27, 4, 420, 419, 319, 7, 5174, false, true},
      {27, 4, 4093, 3001, 2049, 31, 163925, false, true},
      {27, 8, 420, 419, 319, 7, 7541, false, false},
      {27, 8, 4093, 3001, 2049, 31, 283946, false, false},
      {27, 8, 420, 419, 319, 7, 1142, false, true},
      {27, 8, 4093, 3001, 2049, 31, 295061, false, true},
      {27, 16, 420, 419, 319, 7, 13290, false, false},
      {27, 16, 4093, 3001, 2049, 31, 572244, false, false},
      {27, 16, 420, 419, 319, 7, 1206, false, true},
      {27, 16, 4093, 3001, 2049, 31, 606229, false, true},
      {31, 1, 420, 419, 319, 7, 1102, false, false},
      {31, 1, 4093, 3001, 2049, 31, 43589, false, false},
      {31, 1, 420, 419, 319, 7, 5133, false, true},
      {31, 1, 4093, 3001, 2049, 31, 49173, false, true},
      {31, 2, 420, 419, 319, 7, 413, false, false},
      {31, 2, 4093, 3001, 2049, 31, 75146, false, false},
      {31, 2, 420, 419, 319, 7, 5147, false, true},
      {31, 2, 4093, 3001, 2049, 31, 81962, false, true},
      {31, 4, 420, 419, 319, 7, 2618, false, false},
      {31, 4, 4093, 3001, 2049, 31, 154133, false, false},
      {31, 4, 420, 419, 319, 7, 5174, false, true},
      {31, 4, 4093, 3001, 2049, 31, 163925, false, true},
      {31, 8, 420, 419, 319, 7, 7541, false, false},
      {31, 8, 4093, 3001, 2049, 31, 283946, false, false},
      {31, 8, 420, 419, 319, 7, 5229, false, true},
      {31, 8, 4093, 3001, 2049, 31, 295082, false, true},
      {31, 16, 420, 419, 319, 7, 13290, false, false},
      {31, 16, 4093, 3001, 2049, 31, 572244, false, false},
      {31, 16, 420, 419, 319, 7, 5338, false, true},
      {31, 16, 4093, 3001, 2049, 31, 606292, false, true},
      {24, 4, 420, 419, 319, 7, 5336, true, true},
      {24, 4, 4093, 3001, 2049, 31, 663636, true, true},
      {28, 4, 420, 419, 319, 7, 5336, true, true},
      {28, 4, 4093, 3001, 2049, 31, 663636, true, true},
  };
  for (const auto &test : cases) {
    // These single-layer offsets only require height to include the sampled y.
    // Descriptor writers mask the pipe XOR to the metadata allocation alignment.
    const uint32_t alignment = test.depth ? 131072 : test.pipe_aligned ? 16384 : 4096;
    const auto actual = amdgpu::gfx11_metadata_address(
        0x100000 | ((test.pipe_xor << 8) & (alignment - 1)), test.x, test.y, test.width, test.y + 1,
        test.bytes, test.swizzle, test.depth, test.pipe_aligned);
    EXPECT_EQ(actual, 0x100000 + test.expected)
        << test.swizzle << "," << test.bytes << "," << test.x << "," << test.y;
  }
}

TEST(GraphicsImageMetadataTest, DccClearsPreserveNeighborBlocksAndLaterWrites) {
  amdgpu::GpuMemory memory{"dcc_clear_memory"};
  amdgpu::GpuVm vm;
  const auto address_space =
      vm.register_address_space(0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(),
                                std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory));
  const auto access = vm.snapshot(address_space);
  ASSERT_TRUE(access);
  constexpr uint64_t base = 0x200000, metadata = 0x100700;
  for (const auto [key, expected] : {std::pair{0u, 0u},
                                     {1u, 0x76543210u},
                                     {2u, 0xffffffffu},
                                     {4u, 0x3c003c00u},
                                     {6u, 0x3f800000u},
                                     {8u, 0xff000000u},
                                     {10u, 0x00ffffffu}}) {
    const auto tag = amdgpu::gfx11_metadata_address(metadata, 8, 8, 17, 17, 4, 27, false);
    const auto neighbor = amdgpu::gfx11_metadata_address(metadata, 16, 8, 17, 17, 4, 27, false);
    memory.write8(*tag, key);
    memory.write8(*neighbor, 2);
    memory.write32(*amdgpu::gfx11_image_address(base, 8, 8, 17, 4, 27), 0x76543210);
    amdgpu::materialize_gfx11_dcc(*access, base, metadata, 11, 10, 17, 13, 4, 27);
    for (uint32_t y = 8; y < 13; ++y)
      for (uint32_t x = 8; x < 16; ++x)
        EXPECT_EQ(memory.read32(*amdgpu::gfx11_image_address(base, x, y, 17, 4, 27)), expected);
    EXPECT_EQ(memory.read8(*tag), 0xff);
    EXPECT_EQ(memory.read8(*neighbor), 2);
    const auto changed = *amdgpu::gfx11_image_address(base, 11, 10, 17, 4, 27);
    memory.write32(changed, 0x10203040);
    amdgpu::materialize_gfx11_dcc(*access, base, metadata, 11, 10, 17, 13, 4, 27);
    EXPECT_EQ(memory.read32(changed), 0x10203040u);
    memory.write8(*tag, 3);
    EXPECT_THROW(amdgpu::materialize_gfx11_dcc(*access, base, metadata, 11, 10, 17, 13, 4, 27),
                 std::runtime_error);
  }
}

TEST(GraphicsImageMetadataTest, LayerDccClearPreservesOtherLayers) {
  amdgpu::GpuMemory memory{"layered_dcc_memory"};
  amdgpu::GpuVm vm;
  const auto address_space = vm.register_address_space(
      0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(),
      std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory), {}, true);
  const auto access = vm.snapshot(address_space);
  ASSERT_TRUE(access);
  // AddrLib: 129x71 RGBA8 R_X, layer 3 has data at 3*131072 ^ 0x600
  // and DCC at 3*16384 ^ 0x600. Layer 2 uses XOR 0x200.
  memory.write32(0x10c600, 2);
  memory.write32(0x108200, 2);
  memory.write32(0x460600, 0);
  memory.write32(0x440200, 0x12345678);
  amdgpu::materialize_gfx11_dcc(*access, 0x400000, 0x100000, 0, 0, 129, 71, 4, 27, true, 3, 131072);
  EXPECT_EQ(memory.read32(0x460600), 0xffffffffu);
  EXPECT_EQ(memory.read32(0x440200), 0x12345678u);
  EXPECT_EQ(memory.read32(0x10c600), 0xffu);
  EXPECT_EQ(memory.read32(0x108200), 2u);
}

TEST(GraphicsImageMetadataTest, HtileEndpointClearsAndExplicitClearRegister) {
  amdgpu::GpuMemory memory{"htile_clear_memory"};
  amdgpu::GpuVm vm;
  const auto address_space =
      vm.register_address_space(0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(),
                                std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory));
  const auto access = vm.snapshot(address_space);
  ASSERT_TRUE(access);
  constexpr uint64_t base = 0x200000, metadata = 0x100000;
  for (uint32_t bytes : {2u, 4u}) {
    const auto tag = *amdgpu::gfx11_metadata_address(metadata, 8, 8, 17, 17, bytes, 24, true);
    for (uint32_t key : {0u, 0xfffffff0u}) {
      memory.write32(tag, key);
      amdgpu::materialize_gfx11_htile(*access, base, metadata, 9, 9, 17, 13, bytes, 24);
      for (uint32_t y = 8; y < 13; ++y)
        for (uint32_t x = 8; x < 16; ++x) {
          const auto a = *amdgpu::gfx11_image_address(base, x, y, 17, bytes, 24);
          const uint32_t actual = bytes == 2 ? memory.read16(a) : memory.read32(a);
          EXPECT_EQ(actual, key == 0 ? 0 : bytes == 2 ? 65535u : 0x3f800000u);
        }
      EXPECT_EQ(memory.read32(tag), 0xfffc000fu);
    }
    memory.write32(tag, 0x80020000);
    EXPECT_THROW(amdgpu::materialize_gfx11_htile(*access, base, metadata, 9, 9, 17, 13, bytes, 24),
                 std::runtime_error);
    const uint32_t clear = bytes == 2 ? 32768 : 0x3f000000;
    amdgpu::materialize_gfx11_htile(*access, base, metadata, 9, 9, 17, 13, bytes, 24, clear);
    const auto a = *amdgpu::gfx11_image_address(base, 9, 9, 17, bytes, 24);
    EXPECT_EQ(bytes == 2 ? memory.read16(a) : memory.read32(a), clear);
    memory.write32(tag, 0xfffc0001);
    EXPECT_THROW(amdgpu::materialize_gfx11_htile(*access, base, metadata, 9, 9, 17, 13, bytes, 24),
                 std::runtime_error);
  }
}
