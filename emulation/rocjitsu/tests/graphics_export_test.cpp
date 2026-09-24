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
#include "rocjitsu/vm/amdgpu/image_filter.h"
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

  void sample(uint8_t opcode, uint8_t dim, bool a16 = false) {
    std::array<uint32_t, 4> words{};
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
      const auto encoded = rdna4::build_vsample(opcode, {.dim = dim,
                                                         .a16 = a16,
                                                         .dmask = 15,
                                                         .vdata = 12,
                                                         .rsrc = 8,
                                                         .samp = 4,
                                                         .vaddr0 = 0,
                                                         .vaddr1 = 1,
                                                         .vaddr2 = 2,
                                                         .vaddr3 = 3});
      std::copy(encoded.begin(), encoded.end(), words.begin());
    } else {
      const auto encoded = rdna3::build_mimg(
          opcode,
          {.dim = dim, .dmask = 15, .a16 = a16, .vaddr = 0, .vdata = 12, .srsrc = 2, .ssamp = 1});
      std::copy(encoded.begin(), encoded.end(), words.begin());
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

TEST(GraphicsRasterMathTest, AttributeDifferencesMatchPhysicalInterpolation) {
  // Raw RDNA3/4 outputs from an 8x8 right triangle. Independent controls cover
  // cancellation, operand alignment and normalization across both signs.
  struct Case {
    std::array<float, 3> values;
    int x, y;
    uint32_t expected;
  };
  const Case cases[] = {
      {{{0x1.f2addcp+15f, 0x1.8eee8ep-5f, -0x1.2bd482p+23f}}, 0, 0, 0xc908477fu},
      {{{0x1.c50a4ap+24f, 0x1.b9c84ep+16f, -0x1.f4596p+2f}}, 0, 1, 0x4ba9f1aau},
      {{{0x1.1f2552p-24f, -0x1.e7792ap-4f, 0x1.820c08p-38f}}, 0, 0, 0xbbf3bc16u},
      {{{0x1.b9a9c8p-11f, 0x1.d29e82p-34f, -0x1.b39ab6p+13f}}, 0, 0, 0xc459cd4eu},
      {{{0x1.2e2b76p+25f, 0x1.e556ap-25f, -0x1.98d974p+33f}}, 0, 0, 0xce442989u},
      {{{0x1.ff268ep-26f, 0x1.017be2p-25f, 0x1.018d58p-25f}}, 1, 0, 0x33000742u},
      {{{0x1.0086fcp-31f, 0x1.fd5f6ap-32f, 0x1.ff00c2p-32f}}, 0, 1, 0x30001c26u},
      {{{-0x1.ffe8fap-1f, -0x1.006c56p+0f, -0x1.01271p+0f}}, 0, 3, 0xbf80410du},
      {{{0x1.013ecep+62f, 0x1.ffba0ep+61f, 0x1.fe9378p+61f}}, 3, 0, 0x5e80425bu},
      {{{-0x1.d0093p+103f, 0x1.d0093p+103f, 0x1.d0092ep+103f}}, 3, 0, 0xe6000000u},
      {{{-0x1.e57b7p-4f, -0x1.e57b7p-28f, 0x1.e57b7p-28f}}, 0, 0, 0xbdd46602u},
      {{{-0x1.a69a9cp-111f, -0x1.ea8d5ep-106f, 0x1.ea7672p-104f}}, 0, 0, 0x09a0cd18u},
      {{{-0x1.0121ecp+97f, 0x1.0121ecp+97f, 0x1.0121eap+97f}}, 0, 0, 0xefc0d971u},
      {{{0x1.eabfp+80f, 0x1.eabfp+56f, -0x1.eabfp+56f}}, 1, 2, 0x67755f80u},
  };
  for (const auto &test : cases) {
    const float p10 = amdgpu::raster::attribute_difference(test.values[1], test.values[0]);
    const float p20 = amdgpu::raster::attribute_difference(test.values[2], test.values[0]);
    const float result =
        std::fma(p20, (test.y + 0.5f) / 8, std::fma(p10, (test.x + 0.5f) / 8, test.values[0]));
    EXPECT_EQ(std::bit_cast<uint32_t>(result), test.expected);
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

TEST(GraphicsRasterMathTest, VertexDifferencesMatchPhysicalReciprocalW) {
  // Raw pull-model W values captured on both physical RDNA3 and RDNA4.
  // Unequal vertex W values distinguish FP32 truncation of the differences
  // from both wide subtraction and rounding to nearest before weighting.
  // The last two cases straddle the 26-exponent operand cutoff.
  struct Case {
    std::array<std::array<float, 3>, 3> positions; // Homogeneous X, Y, W.
    uint32_t pixel, expected;
  };
  const Case cases[] = {
      {{{{-0x1.e40cp-1f, 0x1.33a78ep-1f, 0x1.841e4p+0f},
         {-0x1.03b21ap+9f, 0x1.caffdp+7f, 0x1.16f4aep+9f},
         {-0x1.ea0b2ep-3f, 0x1.9fd112p+0f, 0x1.d14486p-1f}}},
       12,
       0x3e9e2303u},
      {{{{0x1.deb3f4p-1f, -0x1.630bccp-1f, 0x1.03deaap+1f},
         {-0x1.503afcp-4f, -0x1.f27acep-6f, 0x1.dc85d2p-5f},
         {-0x1.aaa884p+4f, 0x1.066592p+2f, 0x1.119ebcp+4f}}},
       4,
       0x40ca85bcu},
      {{{{0x1.08077ap+3f, 0x1.871c58p+4f, 0x1.3ed0c2p+4f},
         {0x1.644c9cp+7f, -0x1.cf06acp+3f, 0x1.9d3fccp+6f},
         {-0x1.46fbf4p+5f, -0x1.035e62p+5f, 0x1.625e1ap+4f}}},
       1,
       0x3cfed2a4u},
      {{{{-0x1.17d4c4p-11f, 0x1.228112p-6f, 0x1.f1ec8ap-6f},
         {0x1.d51e56p-8f, 0x1.a18582p-6f, 0x1.c210c8p-7f},
         {0x1.843394p-12f, -0x1.0bc392p-14f, 0x1.208b8cp-12f}}},
       14,
       0x43e43e50u},
      {{{{-1, -1, 1}, {0x1.8p24f, -0x1.8p24f, 0x1.8p24f}, {-1, 1, 1}}}, 2, 0x3ec00001u},
      {{{{-1, -1, 1}, {0x1.8p25f, -0x1.8p25f, 0x1.8p25f}, {-1, 1, 1}}}, 2, 0x3ec00000u},
  };
  for (const auto &test : cases) {
    struct Vertex {
      double x, y;
      float w, reciprocal_w;
    };
    std::array<Vertex, 3> vertices;
    for (uint32_t i = 0; i < 3; ++i) {
      const auto &p = test.positions[i];
      vertices[i] = {amdgpu::raster::viewport_coordinate(p[0], p[2], 2, 2),
                     amdgpu::raster::viewport_coordinate(p[1], p[2], 2, 2), p[2],
                     amdgpu::raster::reciprocal(p[2])};
    }
    const auto first = std::min_element(vertices.begin(), vertices.end(),
                                        [](Vertex a, Vertex b) { return a.w < b.w; });
    std::rotate(vertices.begin(), first, vertices.end());
    const auto &[a, b, c] = vertices;
    const double area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    const float inverse_area = amdgpu::raster::truncate_float(1.0 / area);
    const float delta_b = amdgpu::raster::vertex_difference(b.reciprocal_w, a.reciprocal_w);
    const float delta_c = amdgpu::raster::vertex_difference(c.reciprocal_w, a.reciprocal_w);
    const amdgpu::raster::Plane plane{
        amdgpu::raster::plane_gradient(c.y - a.y, a.y - b.y, delta_b, delta_c, inverse_area),
        amdgpu::raster::plane_gradient(a.x - c.x, b.x - a.x, delta_b, delta_c, inverse_area),
        a.reciprocal_w};
    const uint32_t x = test.pixel % 4, y = test.pixel / 4;
    EXPECT_EQ(std::bit_cast<uint32_t>(plane.at_quad((x & ~1u) + 0.5 - a.x, (y & ~1u) + 0.5 - a.y,
                                                    (x & 1) + 2 * (y & 1))),
              test.expected);
  }
}

TEST(GraphicsRasterMathTest, DepthTilesMatchPhysicalRdna3AndRdna4) {
  // Raw gl_FragCoord.z witnesses exercise unequal slopes, tile boundaries,
  // and negative tile-center depths in otherwise visible triangles. Power-of-two
  // spans separate evaluation precision from reciprocal-area approximation.
  struct Case {
    std::array<float, 3> z;
    double origin_x, origin_y, span_x, span_y;
    int x, y;
    uint32_t expected;
  };
  const Case cases[] = {
      {{0x1.61f094p-10f, 0x1.80ed3p-6f, 0x1.b1d6fcp-1f}, 0, 0, 8, 8, 0, 0, 0x3d63c5fc},
      {{0x1.61f094p-10f, 0x1.80ed3p-6f, 0x1.b1d6fcp-1f}, 0, 0, 8, 8, 0, 2, 0x3e88c240},
      {{0x1.c218d2p-8f, 0x1.30ef8cp-4f, 0x1.f9df8p-1f}, 0, 0, 8, 8, 0, 0, 0x3d944e09},
      {{0x1.c218d2p-8f, 0x1.30ef8cp-4f, 0x1.f9df8p-1f}, 0, 0, 8, 8, 0, 2, 0x3ea2aa55},
      {{0x1.a1b06ep-8f, 0x1.1b8234p-6f, 0x1.130d44p-1f}, 0, 0, 8, 8, 0, 0, 0x3d24cc50},
      {{0x1.a1b06ep-8f, 0x1.1b8234p-6f, 0x1.130d44p-1f}, 0, 0, 8, 8, 0, 2, 0x3e311805},
      {{0x1.c589bcp-7f, 0x1.5c398cp-6f, 0x1.357a4ep-1f}, 0, 0, 8, 8, 0, 0, 0x3d51c91d},
      {{0x1.c589bcp-7f, 0x1.5c398cp-6f, 0x1.357a4ep-1f}, 0, 0, 8, 8, 0, 2, 0x3e4ba45a},
      {{0x1.28e624p-5f, 0x1.611d44p-1f, 0x1.56aff2p-7f}, 0, 0, 8, 8, 0, 0, 0x3d9a9058},
      {{0x1.28e624p-5f, 0x1.611d44p-1f, 0x1.56aff2p-7f}, 0, 0, 8, 8, 0, 2, 0x3d8d5cb6},
      {{0x1.efd10ap-5f, 0x1.9178ecp-1f, 0x1.aa8638p-4f}, 0, 0, 8, 8, 0, 0, 0x3dde2826},
      {{0x1.efd10ap-5f, 0x1.9178ecp-1f, 0x1.aa8638p-4f}, 0, 0, 8, 8, 0, 2, 0x3df47bdd},
      {{0x1.61f094p-10f, 0x1.80ed3p-6f, 0x1.b1d6fcp-1f}, 0, 0, 256, 256, 7, 7, 0x3cdb69da},
      {{0x1.61f094p-10f, 0x1.80ed3p-6f, 0x1.b1d6fcp-1f}, 0, 0, 256, 256, 8, 8, 0x3cf731a1},
      {{0x1.61f094p-10f, 0x1.80ed3p-6f, 0x1.b1d6fcp-1f}, 0, 0, 256, 256, 31, 32, 0x3de44db9},
      {{0x1.61f094p-10f, 0x1.80ed3p-6f, 0x1.b1d6fcp-1f}, 0, 0, 256, 256, 127, 127, 0x3ede10f3},
      {{0x1.c218d2p-8f, 0x1.30ef8cp-4f, 0x1.f9df8p-1f}, 0, 0, 256, 256, 7, 7, 0x3d19fafc},
      {{0x1.c218d2p-8f, 0x1.30ef8cp-4f, 0x1.f9df8p-1f}, 0, 0, 256, 256, 8, 8, 0x3d2ac2a4},
      {{0x1.c218d2p-8f, 0x1.30ef8cp-4f, 0x1.f9df8p-1f}, 0, 0, 256, 256, 31, 32, 0x3e0f1968},
      {{0x1.c218d2p-8f, 0x1.30ef8cp-4f, 0x1.f9df8p-1f}, 0, 0, 256, 256, 127, 127, 0x3f07791e},
      {{0x1.a1b06ep-8f, 0x1.1b8234p-6f, 0x1.130d44p-1f}, 0, 0, 256, 256, 7, 7, 0x3cb63c42},
      {{0x1.a1b06ep-8f, 0x1.1b8234p-6f, 0x1.130d44p-1f}, 0, 0, 256, 256, 8, 8, 0x3cc7926b},
      {{0x1.a1b06ep-8f, 0x1.1b8234p-6f, 0x1.130d44p-1f}, 0, 0, 256, 256, 31, 32, 0x3d99d32f},
      {{0x1.a1b06ep-8f, 0x1.1b8234p-6f, 0x1.130d44p-1f}, 0, 0, 256, 256, 127, 127, 0x3e8d69f8},
      {{0x1.2a4644p-1f, 0x1.344512p-17f, 0x1.a61d1ep-17f}, 0, -0.75, 1, 8, 0, 0, 0x3e4d1209},
      {{0x1.908346p-12f, 0x1.d4e288p-12f, 0x1.1a08bcp-16f}, -1.25, 0.25, 8, 1, 0, 0, 0x399fdfb6},
      {{0x1.4e9ceap-16f, 0x1.81919ep-19f, 0x1.5d1c5p-21f}, -1.25, 1.5, 4, 1, 1, 1, 0x3709b38c},
  };
  for (const auto &test : cases) {
    const amdgpu::raster::DepthPlane plane{
        amdgpu::raster::vertex_difference(test.z[1], test.z[0]) / test.span_x,
        amdgpu::raster::vertex_difference(test.z[2], test.z[0]) / test.span_y, test.z[0],
        test.origin_x, test.origin_y};
    EXPECT_EQ(std::bit_cast<uint32_t>(plane.at(test.x, test.y)), test.expected);
  }
}

TEST(GraphicsRasterMathTest, AreaReciprocalMatchesPhysicalInterpolationBoundaries) {
  // Each inverse is isolated by all three raw pull-model planes on both cards.
  // Exact division followed by truncation misses each boundary by one FP32 step.
  struct Case {
    double area;
    uint32_t expected;
  };
  const Case cases[] = {
      {-0x1.cd58p+2, 0xbe0e0df9u},       {0x1.31b7a8p+5, 0x3cd65e2fu},
      {-0x1.82db3p+4, 0xbd296818u},      {0x1.bc544p+5, 0x3c937e89u},
      {0x1.4e50ep+5, 0x3cc407b8u},       {0x1.62182p+3, 0x3db914a7u},
      {0x1.6bb84p+4, 0x3d342ec5u},       {-0x1.7895cp+3, 0xbdae06f2u},
      {-0x1.116bp+5, 0xbcefb10fu},       {-0x1.7bd4p+1, 0xbeac8a8bu},
      {0x1.29620eb378p+25, 0x32dc602fu}, {0x1.29dd006dep+19, 0x35dc0539u},
  };
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(
                  amdgpu::raster::truncate_float(amdgpu::raster::area_reciprocal(test.area))),
              test.expected);
}

TEST(GraphicsRasterMathTest, AreaReciprocalMatchesPhysicalDepthGrid) {
  // Raw gl_FragCoord.z digest from 32768 triangles captured on RDNA3/4.
  // The numerator is exactly one; the regular area grid spans every seed segment.
  uint64_t digest = 14695981039346656037ull;
  for (uint32_t index = 0; index < 32768; ++index) {
    const double area = (2048 + index / 16.0) * 8;
    const amdgpu::raster::DepthPlane plane{
        amdgpu::raster::depth_gradient(1, amdgpu::raster::area_reciprocal(area)), 0, 0, 0, 0};
    for (int x = 0; x < 128; ++x)
      digest = (digest ^ std::bit_cast<uint32_t>(plane.at(x, 0))) * 1099511628211ull;
  }
  EXPECT_EQ(digest, 0x7854ef00633a3c21ull);
}

TEST(GraphicsRasterMathTest, AreaReciprocalTruncatesWideInputsBeforeSeed) {
  // Raw RDNA3/4 depth values from large triangles with numerator exactly one.
  // The last four areas become seed boundaries after input truncation, so
  // discarded input bits must not contribute to the seed's sticky bit.
  struct Case {
    double area;
    int x;
    uint32_t expected;
  };
  const Case cases[] = {
      {0x1.3a3e5945e8ep+29, 0, 0x30508d37u},   {0x1.c6caa591adp+26, 104, 0x356b4a4au},
      {0x1.085e7be51d8p+28, 48, 0x343bdbccu},  {0x1.71b5e6b5afep+28, 63, 0x342fe0cdu},
      {0x1.bbf68001aap+27, 52, 0x34722eb2u},   {0x1.5eca8000e64p+29, 0, 0x303ad2d0u},
      {0x1.065e00017f38p+29, 65, 0x33ffa448u}, {0x1.17078000d15p+29, 123, 0x34629d3cu},
  };
  for (const auto &test : cases) {
    const amdgpu::raster::DepthPlane plane{
        amdgpu::raster::depth_gradient(1, amdgpu::raster::area_reciprocal(test.area)), 0, 0, 0, 0};
    EXPECT_EQ(std::bit_cast<uint32_t>(plane.at(test.x, 0)), test.expected);
  }
}

TEST(GraphicsRasterMathTest, DepthGradientMatchesPhysicalSignedAndStickyInputs) {
  // Raw positive/negative depth captures distinguish seed sticky bits, the
  // gradient product width, and truncation toward zero from rounding downward.
  struct Case {
    double area;
    float numerator;
    double base;
    int x;
    uint32_t expected;
  };
  const Case cases[] = {
      {0x1.1b09a00000000p+14, 0x1.0000000000000p+0f, 0x0.0p+0, 57, 0x3b500765u},
      {0x1.1c03000000000p+14, 0x1.0000000000000p+0f, 0x0.0p+0, 37, 0x3b0734a7u},
      {0x1.34a3600000000p+14, 0x1.0000000000000p+0f, 0x0.0p+0, 20, 0x3a8807a3u},
      {0x1.548fa00000000p+14, 0x1.0000000000000p+0f, 0x0.0p+0, 81, 0x3b750df0u},
      {0x1.680ba00000000p+14, 0x1.0000000000000p+0f, 0x0.0p+0, 3, 0x391f44cfu},
      {0x1.a04e200000000p+14, 0x1.0000000000000p+0f, 0x0.0p+0, 17, 0x3a2c2e6eu},
      {0x1.e3bd818000000p+16, 0x1.c93f360000000p+0f, 0x0.0p+0, 21, 0x39a29477u},
      {0x1.29ef6ec000000p+17, 0x1.5645080000000p-3f, 0x0.0p+0, 34, 0x381e88feu},
      {0x1.15f0bcc000000p+17, 0x1.3efa920000000p+5f, 0x0.0p+0, 65, 0x3c9657a6u},
      {0x1.6ecbb6b000000p+17, 0x1.2c06320000000p+5f, 0x0.0p+0, 39, 0x3c013cd6u},
      {0x1.3bb2c28000000p+16, 0x1.c225160000000p+3f, 0x0.0p+0, 66, 0x3c3da409u},
      {0x1.f684445000000p+17, 0x1.2053500000000p+5f, 0x0.0p+0, 105, 0x3c7220c1u},
      {0x1.aa767b8000000p+18, -0x1.0000020000000p+0f, 0x1.0000000000000p-1, 21, 0x3efff98cu},
      {0x1.9192488000000p+16, -0x1.00000e0000000p+0f, 0x1.0000000000000p-1, 110, 0x3eff731du},
      {0x1.452dcb0000000p+14, -0x1.0000080000000p+0f, 0x1.0000000000000p-1, 120, 0x3efd0915u},
      {0x1.800b140000000p+15, -0x1.ffffe20000000p-1f, 0x1.0000000000000p-1, 47, 0x3eff8159u},
      {0x1.e1d91a8000000p+14, -0x1.0000040000000p+0f, 0x1.0000000000000p-1, 38, 0x3eff5c5du},
      {0x1.f9d8780000000p+13, -0x1.fffffc0000000p-1f, 0x1.0000000000000p-1, 78, 0x3efd845cu},
  };
  for (const auto &test : cases) {
    const double inverse = amdgpu::raster::area_reciprocal(test.area);
    const amdgpu::raster::DepthPlane plane{amdgpu::raster::depth_gradient(test.numerator, inverse),
                                           0, test.base, 0, 0};
    EXPECT_EQ(std::bit_cast<uint32_t>(plane.at(test.x, 0)), test.expected);
  }
}

TEST(GraphicsRasterMathTest, NegativeDepthSlopesMatchPhysicalPacking) {
  // Raw RDNA3/4 depth captures bracket 27-bit midpoint boundaries and translate
  // two-slope triangles across tile positions. Keep the original setup slopes
  // for the center; only negative local slopes round, with ties away from zero.
  struct Case {
    amdgpu::raster::DepthPlane plane;
    int x, y;
    uint32_t expected;
  };
  const Case cases[] = {
      {{-0x1.0708105e00000p-5, 0x0.0p+0, 0x1.fea6a80000000p-4, 0x0.0p+0, 0x0.0p+0},
       0,
       0,
       0x3dde7251u},
      {{-0x1.c03c36de00000p-5, 0x0.0p+0, 0x1.ac19840000000p-3, 0x0.0p+0, 0x0.0p+0},
       3,
       0,
       0x3c8f934fu},
      {{-0x1.088ea01e00000p-4, 0x0.0p+0, 0x1.dc68d80000000p-3, 0x0.0p+0, 0x0.0p+0},
       0,
       0,
       0x3e4d2297u},
      {{-0x1.bf5a09a000000p-11, 0x0.0p+0, 0x1.ea9a000000000p-9, 0x0.0p+0, 0x0.0p+0},
       2,
       0,
       0x3ad301bau},
      {{-0x1.05284ee000000p-5, 0x0.0p+0, 0x1.64cbd20000000p-3, 0x0.0p+0, 0x0.0p+0},
       3,
       0,
       0x3d80488du},
      {{-0x1.35ead22000000p-2, 0x0.0p+0, 0x1.3bf86a0000000p-1, 0x0.0p+0, 0x0.0p+0},
       1,
       0,
       0x3e271099u},
      {{-0x1.029a972200000p-8, 0x0.0p+0, 0x1.b0dbc20000000p-7, 0x0.0p+0, 0x0.0p+0},
       0,
       0,
       0x3c381a8eu},
      {{-0x1.be34b06200000p-6, 0x0.0p+0, 0x1.7f05bc0000000p-4, 0x0.0p+0, 0x0.0p+0},
       0,
       0,
       0x3da39f93u},
      {{-0x1.276a5a6200000p-9, 0x0.0p+0, 0x1.c5cc200000000p-7, 0x0.0p+0, 0x0.0p+0},
       1,
       0,
       0x3c2b821fu},
      {{-0x1.03b33c2000000p-3, -0x1.3510a7e400000p-3, 0x1.59b10c0000000p-2, 0x1.2b00000000000p+3,
        0x1.f600000000000p+0},
       7,
       3,
       0x3eada23au},
  };
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(test.plane.at(test.x, test.y)), test.expected);
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

TEST(GraphicsRasterMathTest, ViewportScalePrecedesPerspectiveDivision) {
  // Exact vertex inputs and subpixel coordinates recovered independently on
  // physical RDNA3 and RDNA4. These include the conditional-rendering triangle
  // and the buffer-device-address edge, plus a sweep over positive W values.
  struct Case {
    uint32_t position, w;
    double expected;
  };
  const Case cases[] = {
      {0xc0148f6e, 0x410f9103, 155.67578125}, {0x408c7bd1, 0x40a00000, 394.3828125},
      {0xbea6cd92, 0x3f3e6086, 118.00390625}, {0xbe9e11f8, 0x3f295d42, 112.00390625},
      {0xc11cde0d, 0x4129ce7e, 16.00390625},  {0x401d634c, 0x409be672, 316.0},
      {0xbeebb3df, 0x3f11952c, 40.00390625},  {0x40289793, 0x405d463d, 370.00390625},
      {0xc0cba720, 0x41686f4e, 118.00390625}, {0xc042be5d, 0x404c7b98, 10.00390625},
      {0xc0b5cd8b, 0x41378df1, 106.00390625}, {0xbff25182, 0x400756bc, 22.00390625},
      {0xc07311dd, 0x409103cc, 34.00390625},  {0xbf60af9f, 0x3fb01000, 76.00390625},
  };
  for (const auto &test : cases)
    EXPECT_EQ(amdgpu::raster::viewport_coordinate(std::bit_cast<float>(test.position),
                                                  std::bit_cast<float>(test.w), 210, 210),
              test.expected);
}

TEST(GraphicsRasterMathTest, ViewportDepthMatchesPhysicalRdna3AndRdna4) {
  // Raw gl_FragCoord.z captures from triangles with constant Z and W isolate
  // viewport arithmetic from depth interpolation. Include reversed and narrow
  // depth ranges, which distinguish scaling before division from scaling NDC.
  struct Case {
    float z, w, scale, offset;
    uint32_t expected;
  };
  const Case cases[] = {
      {0x1.18caaep+24f, 0x1.504b1ap+26f, 1, 0, 0x3e55bffa},
      {0x1.45e0d2p-5f, 0x1.8489a6p-5f, 1, 0, 0x3f56b701},
      {0x1.754f2p-7f, 0x1.26cd38p-6f, 1, 0, 0x3f221650},
      {0x1.18caaep+24f, 0x1.504b1ap+26f, .75f, .125f, 0x3e9027fd},
      {0x1.45e0d2p-5f, 0x1.8489a6p-5f, .75f, .125f, 0x3f410940},
      {0x1.36f0f2p+3f, 0x1.45d762p+6f, .75f, .125f, 0x3e5b9c2e},
      {0x1.45e0d2p-5f, 0x1.8489a6p-5f, -.75f, 1, 0x3ebded80},
      {0x1.e36a64p+12f, 0x1.a9922cp+13f, -.75f, 1, 0x3f12f392},
      {0x1.12b2aap-24f, 0x1.4e3484p-23f, -.75f, 1, 0x3f3117e7},
      {0x1.18caaep+24f, 0x1.504b1ap+26f, 0x1.47aep-8f, .31f, 0x3e9f411e},
      {0x1.e36a64p+12f, 0x1.a9922cp+13f, 0x1.47aep-8f, .31f, 0x3ea02c89},
      {0x1.62a5fcp-6f, 0x1.6553e8p-5f, 0x1.47aep-8f, .31f, 0x3e9ffd8a},
  };
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(
                  amdgpu::raster::viewport_transform(test.z, test.w, test.scale, test.offset)),
              test.expected);
}

TEST(GraphicsRasterMathTest, ReciprocalMatchesPhysicalRdna3AndRdna4) {
  // Raw pull-model reciprocal-W captures cover sticky-bit boundaries, seed
  // segment boundaries, values differing from IEEE division, and exponents.
  constexpr std::array<std::array<uint32_t, 2>, 24> cases{{
      {0x3f800000u, 0x3f800000u}, {0x3f800001u, 0x3f7ffffeu}, {0x3f8003ffu, 0x3f7ff802u},
      {0x3f800400u, 0x3f7ff800u}, {0x3f800401u, 0x3f7ff7feu}, {0x3f81ffffu, 0x3f7c0fc3u},
      {0x3f820000u, 0x3f7c0fc1u}, {0x3f820001u, 0x3f7c0fbfu}, {0x3fb55555u, 0x3f34b4b5u},
      {0x3fc00000u, 0x3f2aaaabu}, {0x3fc00001u, 0x3f2aaaaau}, {0x3fffffffu, 0x3f000001u},
      {0x3f800556u, 0x3f7ff555u}, {0x3f8d547cu, 0x3f67dac1u}, {0x3f9bd297u, 0x3f524a58u},
      {0x3fabe70du, 0x3f3e9ea1u}, {0x3fbd7fffu, 0x3f2ceb11u}, {0x3fd1446du, 0x3f1c959du},
      {0x3fe7be07u, 0x3f0d6601u}, {0x3fffeaa9u, 0x3f000aadu}, {0x02000000u, 0x7d000000u},
      {0x30000000u, 0x4f000000u}, {0x60000000u, 0x1f000000u}, {0x7e000000u, 0x01000000u},
  }};
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(amdgpu::raster::reciprocal(std::bit_cast<float>(test[0]))),
              test[1])
        << std::hex << test[0];
}

TEST(GraphicsRasterMathTest, ReciprocalCompleteNormalizedHardwareDigest) {
  // FNV-style hash of raw reciprocal-W words captured independently on gfx1100 and gfx1201.
  uint64_t digest = 14695981039346656037ull;
  for (uint32_t mantissa = 0; mantissa < (1u << 23); ++mantissa) {
    const float input = std::bit_cast<float>(0x3f800000u | mantissa);
    digest =
        (digest ^ std::bit_cast<uint32_t>(amdgpu::raster::reciprocal(input))) * 1099511628211ull;
  }
  EXPECT_EQ(digest, 0xdb90a52d4704a833ull);
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
    if (lane % 4 != 3) { // The unused fourth coefficient is unspecified.
      EXPECT_EQ(wave_->debug_read_vgpr(3, lane), expected) << lane;
    }
  }
}

TEST_P(GraphicsExportTest, RasterCoverageFragmentInputsAndUnsupportedStates) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  enum class Outcome { Draw, Empty, Reject };
  struct FragmentInputWitness {
    uint32_t xy;
    std::vector<uint32_t> values;
  };
  struct Case {
    const char *name;
    Outcome outcome = Outcome::Draw;
    uint32_t clip_control = 0;
    uint32_t shader_control = 0;
    uint32_t color_control = 0xcc0010;
    uint32_t target_mask = 15;
    uint32_t shader_mask = 15;
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
    bool flat = false;
    bool viewport_scissor = false;
    bool wide_window_scissor = false;
    std::optional<std::array<std::array<float, 4>, 3>> positions{};
    std::optional<FragmentInputWitness> fragment_inputs{};
    float guard = 16382.5f;
  };
  // Additional coverage masks were captured on physical gfx1100/gfx1201.
  const Case cases[] = {
      {.name = "flat first provoking vertex", .inputs = 0xaf28, .flat = true},
      {.name = "flat last provoking vertex",
       .polygon_mode = 1u << 19,
       .inputs = 0xaf28,
       .flat = true},
      {.name = "fragment reciprocal and perspective reconstruction",
       .clip_control = 1u << 19,
       .inputs = 0x880a,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x777,
       .positions = std::array<std::array<float, 4>, 3>{{{-0x1.7ffb74p-1f, -0x1.fff0f4p-2f,
                                                          0x1.80047ap-17f, 0x1.80047ap-15f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}},
       // Physical RDNA3/4 inputs: perspective I/J, pull I/W, J/W, 1/W, and W.
       .fragment_inputs =
           FragmentInputWitness{
               1, {0x3eb9eebfu, 0x3de83a50u, 0x3f4316d8u, 0x3e73aa10u, 0x40064dbau, 0x3ef3fc08u}}},
      {.name = "first clipped fan plane",
       .clip_control = 1u << 19,
       .inputs = 0x8008,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x6660,
       .positions = std::array<std::array<float, 4>, 3>{{{-0x1.a68p-4f, 0x1.0308p+1f, 0.5f, 1.0f},
                                                         {-0x1.03p-1f, -0x1.5f4p-1f, 0.5f, 1.0f},
                                                         {0x1.98ap-1f, -0x1.5f4p-1f, 0.5f, 1.0f}}},
       .fragment_inputs = FragmentInputWitness{0x20001, {0x3eb0d289u, 0x3f109cb1u, 0x3f800000u}},
       .guard = 1.0f},
      {.name = "second clipped fan plane",
       .clip_control = 1u << 19,
       .inputs = 0x8008,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x6660,
       .positions = std::array<std::array<float, 4>, 3>{{{-0x1.a68p-4f, 0x1.0308p+1f, 0.5f, 1.0f},
                                                         {-0x1.03p-1f, -0x1.5f4p-1f, 0.5f, 1.0f},
                                                         {0x1.98ap-1f, -0x1.5f4p-1f, 0.5f, 1.0f}}},
       .fragment_inputs = FragmentInputWitness{0x20002, {0x3ef221ecu, 0x3eb0d28du, 0x3f800000u}},
       .guard = 1.0f},
      {.name = "large guard-band plane 77",
       .clip_control = 1u << 19,
       .inputs = 0x8008,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0xffff,
       .positions =
           std::array<std::array<float, 4>, 3>{
               {{0x1.4240f8p+12f, -0x1.5bd2d6p+12f, 0x1.9d4a52p-3f, 0x1.9d4a52p-2f},
                {-0x1.c59decp+3f, 0x1.0e3f8ep+7f, 0x1.01d442p-8f, 0x1.01d442p-7f},
                {-0x1.2ebf44p+10f, -0x1.5bea08p+10f, 0x1.9a73aap-5f, 0x1.9a73aap-4f}}},
       .fragment_inputs = FragmentInputWitness{0x0, {0x3f3f445du, 0x4262cbedu, 0x426fdb72u}}},
      {.name = "large guard-band plane 101",
       .clip_control = 1u << 19,
       .inputs = 0x8008,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0xffff,
       .positions =
           std::array<std::array<float, 4>, 3>{
               {{0x1.a14c92p+11f, -0x1.6fd38cp+11f, 0x1.a0f93p-4f, 0x1.a0f93p-3f},
                {0x1.3f3308p+4f, 0x1.5d5316p+7f, 0x1.674696p-8f, 0x1.674696p-7f},
                {-0x1.582426p+9f, -0x1.33a608p+9f, 0x1.88e304p-6f, 0x1.88e304p-5f}}},
       .fragment_inputs = FragmentInputWitness{0x0, {0x40d2b26eu, 0x3f8db90eu, 0x424624b3u}}},
      {.name = "large guard-band plane 651",
       .clip_control = 1u << 19,
       .inputs = 0x8008,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0xffff,
       .positions =
           std::array<std::array<float, 4>, 3>{
               {{-0x1.0e6bd6p+14f, -0x1.fc483p+13f, 0x1.222636p-1f, 0x1.222636p+0f},
                {0x1.f3d8eap+12f, -0x1.2deb56p+13f, 0x1.1a2406p-2f, 0x1.1a2406p-1f},
                {0x1.e1057p+6f, 0x1.595754p+9f, 0x1.4c13fep-6f, 0x1.4c13fep-5f}}},
       .fragment_inputs = FragmentInputWitness{0x0, {0x3ecb9e89u, 0x413d3646u, 0x4147d4b7u}}},
      {.name = "all vertices behind eye",
       .outcome = Outcome::Empty,
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x0,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.5f, -1.0f},
                                                         {0.75f, -0.5f, 0.5f, -1.0f},
                                                         {0.0f, 0.75f, 0.5f, -1.0f}}}},
      {.name = "top vertex behind eye",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0xff0,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, -0.5f}}}},
      {.name = "left vertex behind eye",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x777,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.25f, -0.5f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}}},
      {.name = "two vertices behind eye",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x1,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.75f, -0.5f, 0.25f, -0.5f},
                                                         {0.0f, 0.75f, 0.25f, -0.5f}}}},
      {.name = "mixed W outside viewport",
       .outcome = Outcome::Empty,
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x0,
       .positions = std::array<std::array<float, 4>, 3>{{{-2.0f, -2.0f, 0.25f, 1.0f},
                                                         {2.0f, -2.0f, 0.25f, 1.0f},
                                                         {0.0f, -2.0f, 0.25f, -1.0f}}}},
      {.name = "near plane and eye clipping",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x7773,
       .positions = std::array<std::array<float, 4>, 3>{{{-1.25f, -0.5f, -0.25f, -0.5f},
                                                         {1.25f, -0.5f, 0.5f, 2.0f},
                                                         {0.0f, 1.75f, 0.125f, 0.5f}}}},
      {.name = "unequal W across eye plane",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0xcc80,
       .positions = std::array<std::array<float, 4>, 3>{{{1.5f, 1.0f, 0.125f, -0.25f},
                                                         {-0.25f, 1.5f, 0.25f, 1.0f},
                                                         {0.75f, -1.0f, 0.5f, 2.0f}}}},
      {.name = "top vertex at zero W",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x7770,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 0.0f}}}},
      {.name = "left vertex at zero W",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x777,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.25f, 0.0f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}}},
      {.name = "near-zero W",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x777,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.25f, 1e-10f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}}},
      {.name = "overflowing initial projection",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x777,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.25f, 1e-39f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}}},
      {.name = "zero W and Z",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x777,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.0f, 0.0f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}}},
      {.name = "subnormal W and Z",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x777,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 1e-39f, 1e-39f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}}},
      {.name = "finite small W at zero Z",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x677,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.0f, 1e-10f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}}},
      {.name = "guard-band horizontal clipping",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0xff0,
       .positions = std::array<std::array<float, 4>, 3>{{{-100000000.0f, -0.5f, 0.25f, 1.0f},
                                                         {100000000.0f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}}},

      {.name = "integer coverage"},
      {.name = "viewport scissor", .expected_coverage = 0x440, .viewport_scissor = true},
      {.name = "unequal W and explicit vertex parameters", .inputs = 0xaf28, .passthrough = true},
      {.name = "position z and packed coordinates", .inputs = 0x8402},
      {.name = "unsupported line stipple input", .outcome = Outcome::Reject, .inputs = 0x80},
      {.name = "unsupported front-face input", .outcome = Outcome::Reject, .inputs = 0x1000},
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
      {.name = "pattern-dependent logic op", .outcome = Outcome::Reject, .color_control = 0x120010},
      {.name = "color disabled", .color_control = 0, .depth_only = true},
      {.name = "color mode disabled without depth", .color_control = 0},
      {.name = "all color writes masked", .target_mask = 0},
      {.name = "no fragment color export", .shader_mask = 0},
      {.name = "no attachment writes with viewport scissor",
       .target_mask = 0,
       .shader_mask = 0,
       .expected_coverage = 0x440,
       .viewport_scissor = true,
       .wide_window_scissor = true},
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
    state.context_registers[0x214] = test.target_mask;
    state.context_registers[0x215] = test.shader_mask;
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
      state.context_registers[0x8e] = test.target_mask;
      state.context_registers[0x8f] = test.shader_mask;
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
    const bool vertex_parameters = test.passthrough || test.flat;
    if (vertex_parameters) {
      state.sh_registers[gfx12 ? 0x84 : 0x88] = 0x300000;
      if (gfx12)
        state.sh_registers[0x31] = 1u << 11;
      else
        state.context_registers[0x1b6] = 1;
      state.context_registers[gfx12 ? 0x199 : 0x191] = test.flat ? 0x400 : 0x420;
      state.context_registers[gfx12 ? 0x197 : 0x1b3] = test.inputs;
      for (uint32_t c = 0; c < 4; ++c)
        memory_.write32(0x3000a0 + c * 4, c == 0 ? 0x400000 : 0);
      for (uint32_t k = 0; k < 3; ++k)
        for (uint32_t c = 0; c < 4; ++c)
          memory_.write32(0x400000 + k * 16 + c * 4,
                          test.flat ? 0x7fc00000u + k * 4 + c
                                    : std::bit_cast<uint32_t>(float(10 + k * 4 + c)));
    }
    if (test.viewport_scissor) {
      state.context_registers[0x292] = 2;
      state.context_registers[0x94] = 2 | (1 << 16);
      state.context_registers[0x95] = (3 - gfx12) | ((3 - gfx12) << 16);
    }
    if (test.wide_window_scissor)
      state.context_registers[0x91] = 0x3fff3fff;
    if (test.positions) {
      state.context_registers[gfx12 ? 0x10b : 0x2fa] =
          state.context_registers[gfx12 ? 0x10d : 0x2fc] = std::bit_cast<uint32_t>(test.guard);
    }
    const float extent = test.extent;
    auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
    for (uint32_t i = 0; i < 3; ++i) {
      const float w = vertex_parameters ? float(1u << i) : 1.0f;
      std::array<uint32_t, 4> position{
          std::bit_cast<uint32_t>((i == 2 ? extent : -extent) * w),
          std::bit_cast<uint32_t>((i == 1 ? extent : -extent) * w),
          std::bit_cast<uint32_t>(!test.first_vertex_depth_only || i == 0 ? test.depth : 0.0f),
          std::bit_cast<uint32_t>(w)};
      if (test.positions)
        for (uint32_t c = 0; c < 4; ++c)
          position[c] = std::bit_cast<uint32_t>((*test.positions)[i][c]);
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
    uint32_t covered_index = 0;
    bool witness_seen = false;
    for (uint32_t workgroup = 0; workgroup < dispatch->total_wgs; ++workgroup) {
      wave_->set_wg_coord(workgroup, 0, 0);
      wave_->set_graphics_stage(draw);
      draw->initialize(*wave_, workgroup, 0);
      for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
        if (!(wave_->exec() & (uint64_t{1} << lane)))
          continue;
        if (test.fragment_inputs && wave_->debug_read_vgpr(test.fragment_inputs->values.size(),
                                                           lane) == test.fragment_inputs->xy) {
          witness_seen = true;
          for (uint32_t reg = 0; reg < test.fragment_inputs->values.size(); ++reg)
            EXPECT_EQ(wave_->debug_read_vgpr(reg, lane), test.fragment_inputs->values[reg])
                << "register=" << reg << " lane=" << lane;
        }
        if (test.inputs == 0xaf28) {
          // Pull I/W, J/W, 1/W; linear I/J; position XYZW; packed XY.
          const uint32_t x = 1 + covered_index % 2, y = 1 + covered_index / 2;
          // Screen vertices are (0,0), (0,4), (4,0), with W={1,2,4}.
          // Barycentric and reciprocal-W values below are exactly representable.
          const float b1 = (y + 0.5f) / 4, b2 = (x + 0.5f) / 4;
          const float rw = 1 - b1 - b2 + b1 / 2 + b2 / 4;
          const float expected[] = {b1 / 2, b2 / 4, rw, b1, b2, x + 0.5f, y + 0.5f, 0, 1.0f / rw};
          for (uint32_t reg = 0; reg < 9; ++reg)
            EXPECT_EQ(wave_->debug_read_vgpr(reg, lane), std::bit_cast<uint32_t>(expected[reg]))
                << "register=" << reg << " lane=" << lane;
          EXPECT_EQ(wave_->debug_read_vgpr(9, lane), 0u);
          EXPECT_EQ(wave_->debug_read_vgpr(10, lane), x | (y << 16));
          for (uint32_t c = 0; c < 4; ++c)
            for (uint32_t k = 0; k < 3; ++k)
              EXPECT_EQ(
                  lds_.read32(wave_->lds_base() + (c * 3 + k) * 4),
                  test.flat
                      ? (k ? 0u : 0x7fc00000u + ((test.polygon_mode & (1u << 19)) ? 8u : 0u) + c)
                      : std::bit_cast<uint32_t>(float(10 + k * 4 + c)));
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
    }
    EXPECT_EQ(covered_index, (test.triangle || test.viewport_scissor)
                                 ? std::popcount(test.expected_coverage)
                                 : std::popcount(test.sample_coverage));
    if (test.fragment_inputs) {
      EXPECT_TRUE(witness_seen);
    }
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
                  !test.depth_only && test.color_control && test.target_mask && test.shader_mask &&
                          sample_enabled && covered
                      ? 0xff0000ffu
                      : 0u)
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
            context[gfx12 ? 0x198 : 0x1b4] = 2 | (1u << 13);
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
                EXPECT_EQ(wave_->debug_read_vgpr(2, 0), relative_layer << 16);
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
  EXPECT_NO_THROW(amdgpu::GraphicsDraw(state, GetParam(), 3, {9, 4, 9}));
  // Odd strip triangles retain the API-selected first or last provoking vertex.
  state.uconfig_registers[0x242] = 6;
  state.uconfig_registers[0x24b] = 0;
  for (bool last_provoking : {false, true}) {
    state.context_registers[gfx12 ? 0x207 : 0x205] = last_provoking ? 1u << 19 : 0;
    amdgpu::GraphicsDraw strip(state, GetParam(), 4, {3, 4, 5, 6});
    strip.initialize(*wave_, 0, 0);
    EXPECT_EQ(wave_->debug_read_vgpr(0, 1), last_provoking
                                                ? 2u | (1u << bits) | (3u << (2 * bits))
                                                : 1u | (3u << bits) | (2u << (2 * bits)));
  }
}

TEST_P(GraphicsExportTest, PrimitiveRestartResetsStripWindingAndDropsIncompleteSegments) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  for (bool last_provoking : {false, true})
    for (uint32_t primitive : {4u, 6u}) {
      for (const auto &[index_type, marker] :
           {std::pair{0u, 0xffffu}, {1u, 0xffffffffu}, {2u, 0xffu}}) {
        amdgpu::Pm4QueueState state;
        state.uconfig_registers[0x242] = primitive;
        state.uconfig_registers[0x24b] = 1;
        state.uconfig_registers[0x243] = index_type;
        state.context_registers[0x103] = 0xffffffff;
        state.context_registers[gfx12 ? 0x207 : 0x205] = last_provoking ? 1u << 19 : 0;
        const std::vector<uint32_t> input{marker, 99, marker, 3,  4,  5,      6,  marker,
                                          marker, 10, 11,     12, 13, marker, 90, 91};
        const std::vector<uint32_t> expected =
            primitive == 4   ? std::vector<uint32_t>{3, 4, 5, 10, 11, 12}
            : last_provoking ? std::vector<uint32_t>{3, 4, 5, 5, 4, 6, 10, 11, 12, 12, 11, 13}
                             : std::vector<uint32_t>{3, 4, 5, 4, 6, 5, 10, 11, 12, 11, 13, 12};
        amdgpu::GraphicsDraw draw(state, GetParam(), input.size(), input);
        draw.initialize(*wave_, 0, 0);
        EXPECT_EQ(wave_->debug_read_sgpr(3), expected.size() | ((expected.size() / 3) << 8));
        for (uint32_t i = 0; i < expected.size(); ++i)
          EXPECT_EQ(wave_->debug_read_vgpr(gfx12 ? 3 : 5, i), expected[i]);
        for (uint32_t i = 0; i < expected.size() / 3; ++i) {
          const uint32_t bits = gfx12 ? 9 : 10;
          EXPECT_EQ(wave_->debug_read_vgpr(0, i),
                    (3 * i) | ((3 * i + 1) << bits) | ((3 * i + 2) << (2 * bits)));
        }
        // MATCH_ALL_BITS keeps upper register bits significant for narrow indices.
        state.uconfig_registers[0x24b] |= 2;
        amdgpu::GraphicsDraw full_match(state, GetParam(), input.size(), input);
        full_match.initialize(*wave_, 0, 0);
        if (marker != 0xffffffff) {
          for (uint32_t i = 0; i < input.size(); ++i)
            EXPECT_EQ(wave_->debug_read_vgpr(gfx12 ? 3 : 5, i), input[i]);
        }
      }
    }
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
    for (const auto &[vertices, instances] :
         {std::pair{4u, 2u}, {5u, 2u}, {31u, 2u}, {32u, 2u}, {64u, 2u}, {65u, 2u}, {3u, 8192u}}) {
      SCOPED_TRACE(testing::Message() << "vertices=" << vertices << ", primitive="
                                      << (primitive == 4 ? "triangle list" : "rectangle list"));
      amdgpu::Pm4QueueState state;
      state.num_instances = instances;
      state.sh_registers[0x8b] = (gfx12 ? 1u : 3u) << 16;
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
        ASSERT_LT(groups++, instances * ((vertices + 29) / 30));
        draw->initialize(*wave_, 0, 0);
        const uint32_t count = wave_->debug_read_sgpr(3) & 255;
        const uint32_t primitive_count = (wave_->debug_read_sgpr(3) >> 8) & 255;
        for (uint32_t i = 0; i < count; ++i) {
          EXPECT_EQ(wave_->debug_read_vgpr(gfx12 ? 3 : 5, i), invocations % vertices);
          EXPECT_EQ(wave_->debug_read_vgpr(gfx12 ? 4 : 8, i), invocations / vertices);
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
      EXPECT_EQ(invocations, instances * vertices);
      EXPECT_EQ(primitives, instances * (vertices / 3));
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
                        {"cube face view", 128, (6 << 16) | 11, 11, 5, 7684, 1},
                        {"cube face view limit", 128, (6 << 16) | 11, 11, 5, 516, 6, true},
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
      if (!c.out_of_bounds) {
        EXPECT_EQ(transfer->per_lane_addr[0], base + c.offset);
      }
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
      if (array) {
        EXPECT_EQ(memory_.read32(preserved), 0x12345678u);
      }
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
      if (array) {
        EXPECT_EQ(memory_.read32(preserved), 0x12345678u);
      }
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
    uint32_t color_control = 0xcc0010;
    std::array<uint32_t, 4> exported{0x3f000000, 0x3e800000, 0x3e800000, 0x3f000000};
    std::array<float, 4> constants{0.25f, 0.25f, 0.5f, 0.5f};
  };
  constexpr uint32_t enabled = 1u << 30, separate = 1u << 29;
  const Case cases[] = {
      // Physical RDNA3/4 give blending precedence over XOR, including ZERO/ZERO.
      {.blend = enabled, .mask = 15, .expected = 0, .color_control = 0x660010},
      {.blend = enabled | 1, .mask = 15, .expected = 0x80404080, .color_control = 0x660010},
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
      // Physical HPP depth-peeling inputs: the red blend lies just above
      // the 141/142 midpoint, but rounding it to FP32 first gives 141.
      {.blend = enabled | separate | 4 | (3 << 8) | (1 << 16),
       .mask = 15,
       .expected = 0xef02968e,
       .initial = 0x01020135,
       .exported = {0x3ef36000, 0x3f1fa000, 0, 0x3f702000}},
      // Independent physical readbacks distinguish 23-bit product alignment
      // and inverse-factor decomposition from exact host-double arithmetic.
      {.blend = enabled | separate | 2 | (3 << 8) | (1 << 16),
       .mask = 15,
       .expected = 0x68a2a2a2,
       .initial = 0x25252525,
       .exported = {0x3f472000, 0x3f472000, 0x3f472000, 0x3ed0c000}},
      {.blend = enabled | separate | 11 | (12 << 8) | (1 << 16),
       .mask = 15,
       .expected = 0x2f628899,
       .initial = 0x99999999,
       .exported = {0x3ec4a000, 0x3ec4a000, 0x3ec4a000, 0x3e3e6000},
       .constants = {-0.75f, 0.3f, 1.25f, 0.65f}},
      {.blend = enabled | separate | 11 | (12 << 8) | (1 << 16),
       .mask = 15,
       .expected = 0xd453b3dd,
       .initial = 0xdddddddd,
       .exported = {0x3ea5e000, 0x3ea5e000, 0x3ea5e000, 0x3f546000},
       .constants = {-0.75f, 0.3f, 1.25f, 0.65f}},
      {.blend = enabled | separate | 10 | (1 << 8) | (1 << 16),
       .mask = 15,
       .expected = 0x90aeaeae,
       .initial = 0xa5a5a5a5,
       .exported = {0x3dd8a000, 0x3dd8a000, 0x3dd8a000, 0x3f10a000}},
      {.blend = enabled | separate | 10 | (1 << 8) | (1 << 16),
       .mask = 15,
       .expected = 0x73d0d0d0,
       .initial = 0xcccccccc,
       .exported = {0x3db44000, 0x3db44000, 0x3db44000, 0x3ee6a000}},
  };
  for (uint32_t swap = 0; swap < 4; ++swap) {
    const auto memory_word = [swap](uint32_t rgba) {
      const uint32_t r = rgba & 255, g = (rgba >> 8) & 255, b = (rgba >> 16) & 255, a = rgba >> 24;
      switch (swap) {
      case 1:
        return b | (g << 8) | (r << 16) | (a << 24);
      case 2:
        return a | (b << 8) | (g << 16) | (r << 24);
      case 3:
        return a | (r << 8) | (g << 16) | (b << 24);
      default:
        return rgba;
      }
    };
    SCOPED_TRACE(swap);
    for (const auto &test : cases) {
      SCOPED_TRACE(test.blend);
      amdgpu::Pm4QueueState state;
      state.num_instances = 1;
      state.uconfig_registers[0x242] = 17;
      auto &context = state.context_registers;
      context[gfx12 ? 0x3b0 : 0x31c] = 10 | (test.number_format << 8) | (swap << 11);
      context[gfx12 ? 0x31e : 0x3b0] = gfx12 ? 3 | (3 << 16) : 3 | (3 << 14);
      context[gfx12 ? 0x31f : 0x3b8] = gfx12 ? 3u << 15 : 26u << 14;
      context[0x318] = 0x1000;
      context[gfx12 ? 0x214 : 0x8e] = test.mask;
      context[gfx12 ? 0x215 : 0x8f] = 15;
      context[gfx12 ? 0x195 : 0x1c5] = test.export_format;
      context[gfx12 ? 0x198 : 0x1b4] = 2;
      context[0x2f9] = 0x2d;
      context[gfx12 ? 0x205 : 0x206] = 0x43f;
      context[gfx12 ? 0x216 : 0x202] = test.color_control;
      context[0x1e0] = test.blend;
      const uint32_t boundary[] = {0x3b008081, 0x3e56d6d7, 0x3ed5d5d6, 0x3f202020};
      for (uint32_t c = 0; c < 4; ++c)
        context[0x105 + c] =
            test.constant_boundary ? boundary[c] : std::bit_cast<uint32_t>(test.constants[c]);
      context[0x10f] = context[0x110] = context[0x111] = context[0x112] =
          std::bit_cast<uint32_t>(2.0f);
      context[0x91] = gfx12 ? 0 : 1 | (1 << 16);
      context[0x30e] = context[0x30f] = 0xffffffff;
      memory_.write32(0x100000, memory_word(test.initial));

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
        EXPECT_EQ(memory_.read32(0x100000), memory_word(test.initial));
        continue;
      }
      ASSERT_TRUE(draw->advance(*access_));
      wave_->set_wg_coord(0, 0, 0);
      wave_->set_graphics_stage(draw);
      draw->initialize(*wave_, 0, 0);
      for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
        draw->export_lane(*wave_, lane, 0, 15, test.exported);
      EXPECT_FALSE(draw->advance(*access_));
      EXPECT_EQ(memory_.read32(0x100000), memory_word(test.expected));
    }
  }
}

TEST_P(GraphicsExportTest, LogicOperationsUseConvertedBitsAndLogicalChannelMasks) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t sources[] = {0x3c695ac3, 0x3cc35a69, 0xc35a693c, 0x695ac33c};
  const uint32_t masks[] = {0x00ff00ff, 0x00ff00ff, 0xff00ff00, 0xff00ff00};
  for (uint32_t swap = 0; swap < 4; ++swap) {
    const uint32_t src = sources[swap], dst = 0xf3c596a5;
    const uint32_t results[] = {0,          ~(src | dst), ~src & dst, ~src,
                                src & ~dst, ~dst,         src ^ dst,  ~(src & dst),
                                src & dst,  ~(src ^ dst), dst,        ~src | dst,
                                src,        src | ~dst,   src | dst,  ~0u};
    for (uint32_t rop = 0; rop < 16; ++rop) {
      for (bool disable : {false, true})
        for (uint32_t mask : {5u, 15u}) {
          SCOPED_TRACE(swap);
          SCOPED_TRACE(rop);
          SCOPED_TRACE(mask);
          SCOPED_TRACE(disable);
          amdgpu::Pm4QueueState state;
          state.num_instances = 1;
          state.uconfig_registers[0x242] = 17;
          auto &context = state.context_registers;
          context[gfx12 ? 0x3b0 : 0x31c] = 10 | (4u << 8) | (swap << 11);
          context[gfx12 ? 0x31e : 0x3b0] = gfx12 ? 3 | (3 << 16) : 3 | (3 << 14);
          context[gfx12 ? 0x31f : 0x3b8] = gfx12 ? 3u << 15 : 26u << 14;
          context[0x318] = 0x1000;
          context[0x1e0] = disable ? 1u << 31 : 0;
          context[gfx12 ? 0x214 : 0x8e] = mask;
          context[gfx12 ? 0x215 : 0x8f] = 15;
          context[gfx12 ? 0x195 : 0x1c5] = 9u;
          context[gfx12 ? 0x198 : 0x1b4] = 2;
          context[0x2f9] = 0x2d;
          context[gfx12 ? 0x205 : 0x206] = 0x43f;
          context[gfx12 ? 0x216 : 0x202] = (rop * 0x110000) | 0x10;
          context[0x10f] = context[0x110] = context[0x111] = context[0x112] =
              std::bit_cast<uint32_t>(2.0f);
          context[0x91] = gfx12 ? 0 : 1 | (1 << 16);
          context[0x30e] = context[0x30f] = 0xffffffff;
          memory_.write32(0x100000, 0xf3c596a5);

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
            draw->export_lane(*wave_, lane, 0, 15, {0xc3, 0x5a, 0x69, 0x3c});
          EXPECT_FALSE(draw->advance(*access_));
          const uint32_t bits = mask == 15 ? ~0u : masks[swap];
          EXPECT_EQ(memory_.read32(0x100000),
                    ((disable ? src : results[rop]) & bits) | (dst & ~bits));
        }
    }
  }
}

TEST_P(GraphicsExportTest, ColorAttachmentsWriteFullTexelsAndPreserveMaskedChannels) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    uint32_t data_format, number_format, export_format, bytes, components = 4, mask;
    std::array<uint32_t, 4> exported, expected;
    uint32_t blend = 0, blend_opt = 0;
    std::array<uint32_t, 4> initial{0xcccccccc, 0xcccccccc, 0xcccccccc, 0xcccccccc};
    std::array<uint32_t, 4> constant_bits{0xbf400000, 0x3e99999a, 0x3fa00000, 0x3f266666};
  };
  const Case cases[] = {
      // A zero export bypasses arithmetic and preserves signaling NaN payloads.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
       .expected = {0x7f812345u, 0x7f812345u, 0x7f812345u, 0xff812345u},
       .blend = 0x61000101u,
       .blend_opt = 0x1100111u,
       .initial = {0x7f812345u, 0x7f812345u, 0x7f812345u, 0xff812345u}},
      // Subnormal exports do not trigger zero flags, then flush before arithmetic.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x00000001u, 0x00000001u, 0x00000001u, 0x00000000u},
       .expected = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
       .blend = 0x61000101u,
       .blend_opt = 0x1100111u,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u}},
      // Active NaN products canonicalize their result.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x00000000u},
       .expected = {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u},
       .blend = 0x61000101u,
       .blend_opt = 0x1100111u,
       .initial = {0x7f812345u, 0x7f812345u, 0x7f812345u, 0x7fc12345u}},
      // A positive-zero coefficient suppresses a NaN color.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x7fc12345u, 0x7fc12345u, 0x7fc12345u, 0x00000000u},
       .expected = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u},
       .blend = 0x61000604u,
       .blend_opt = 0x1100174u,
       .initial = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}},
      // Negative subnormal coefficients flush with their sign intact.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x80000001u},
       .expected = {0x80000000u, 0x80000000u, 0x80000000u, 0x00000000u},
       .blend = 0x61000604u,
       .blend_opt = 0x1100174u,
       .initial = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x80000000u}},
      // Saturation selects the numeric factor when the source alpha is NaN.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x7fc12345u},
       .expected = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u},
       .blend = 0x6100010au,
       .blend_opt = 0x1100176u,
       .initial = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}},
      // MIN selects the numeric operand instead of a source NaN.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x7f812345u, 0x7f812345u, 0x7f812345u, 0x7f800000u},
       .expected = {0xbf800000u, 0xbf800000u, 0xbf800000u, 0xff800000u},
       .blend = 0x61400141u,
       .blend_opt = 0x0u,
       .initial = {0xbf800000u, 0xbf800000u, 0xbf800000u, 0xff800000u}},
      // MAX normalizes zero signs and canonicalizes two NaNs.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x80000000u, 0x80000000u, 0x80000000u, 0x7f812345u},
       .expected = {0x00000000u, 0x00000000u, 0x00000000u, 0xffc00000u},
       .blend = 0x61600161u,
       .blend_opt = 0x0u,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0xffc12345u}},
      // Constant zero/one equations preserve raw destination bits.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x7ff339aeu, 0x3f800000u, 0x40000000u, 0xea97a4b8u},
       .expected = {0x7f839d69u, 0x40400000u, 0x40800000u, 0xff800000u},
       .blend = 0x61000c0bu,
       .blend_opt = 0x1100177u,
       .initial = {0x7f839d69u, 0x40400000u, 0x40800000u, 0xff800000u},
       .constant_bits = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      // A subnormal constant flushes without qualifying for raw destination bypass.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x7ff339aeu, 0x3f800000u, 0x40000000u, 0xea97a4b8u},
       .expected = {0xffc00000u, 0x40400000u, 0x40800000u, 0xff800000u},
       .blend = 0x61000c0bu,
       .blend_opt = 0x1100177u,
       .initial = {0x7f839d69u, 0x40400000u, 0x40800000u, 0xff800000u},
       .constant_bits = {0x00000001u, 0x00000001u, 0x00000001u, 0x00000001u}},
      // Signaling NaN constants produce canonical NaNs during arithmetic.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x7ff339aeu, 0x3f800000u, 0x40000000u, 0xea97a4b8u},
       .expected = {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xff800000u},
       .blend = 0x61000c0bu,
       .blend_opt = 0x1100177u,
       .initial = {0x7f839d69u, 0x40400000u, 0x40800000u, 0xff800000u},
       .constant_bits = {0x7f812345u, 0x7f812345u, 0x7f812345u, 0x7f812345u}},
      // An exact source replacement preserves raw source bits.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 1,
       .exported = {0x7ff339aeu, 0x3f800000u, 0x40000000u, 0xea97a4b8u},
       .expected = {0x7ff339aeu, 0x40400000u, 0x40800000u, 0xff800000u},
       .blend = 0x61000b0cu,
       .blend_opt = 0x1100177u,
       .initial = {0x7f839d69u, 0x40400000u, 0x40800000u, 0xff800000u},
       .constant_bits = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      // Source replacement also bypasses subtraction of a fixed zero.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 1,
       .exported = {0x7ff339aeu, 0x3f800000u, 0x40000000u, 0xea97a4b8u},
       .expected = {0x7ff339aeu, 0x40400000u, 0x40800000u, 0xff800000u},
       .blend = 0x61200b2cu,
       .blend_opt = 0x2100277u,
       .initial = {0x7f839d69u, 0x40400000u, 0x40800000u, 0xff800000u},
       .constant_bits = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      // Constant classification includes RGB channels absent from the export.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 1,
       .exported = {0x7ff339aeu, 0x3f800000u, 0x40000000u, 0xea97a4b8u},
       .expected = {0xffc00000u, 0x40400000u, 0x40800000u, 0xff800000u},
       .blend = 0x61000c0bu,
       .blend_opt = 0x1100177u,
       .initial = {0x7f839d69u, 0x40400000u, 0x40800000u, 0xff800000u},
       .constant_bits = {0x00000000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}},
      // Physical zero signs with SX bypass enabled, disabled, and partial exports.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0, 0, 0, 0},
       .expected = {0, 0, 0, 0},
       .blend = 0x61000604u,
       .blend_opt = 0x01100174u,
       .initial = {0, 0, 0, 0x80000000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x00000000u, 0x00000000u, 0x00000000u, 0x80000000u},
       .expected = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
       .blend = 0x61000604u,
       .blend_opt = 0x00000000u,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0x00000000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x00000000u, 0x00000000u, 0x00000000u, 0x80000000u},
       .expected = {0x80000000u, 0x80000000u, 0x80000000u, 0x00000000u},
       .blend = 0x61200624u,
       .blend_opt = 0x00000000u,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0x00000000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
       .expected = {0x80000000u, 0x80000000u, 0x80000000u, 0x00000000u},
       .blend = 0x61000101u,
       .blend_opt = 0x01100111u,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0x00000000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
       .expected = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
       .blend = 0x61000101u,
       .blend_opt = 0x00000000u,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0x00000000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x00000000u, 0x3f800000u, 0x40000000u, 0x00000000u},
       .expected = {0x00000000u, 0x40800000u, 0x40c00000u, 0x00000000u},
       .blend = 0x61000101u,
       .blend_opt = 0x01100111u,
       .initial = {0x80000000u, 0x40400000u, 0x40800000u, 0x00000000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 1,
       .exported = {0x00000000u, 0x3f800000u, 0x40000000u, 0x00000000u},
       .expected = {0x80000000u, 0x40400000u, 0x40800000u, 0x00000000u},
       .blend = 0x61000101u,
       .blend_opt = 0x01100111u,
       .initial = {0x80000000u, 0x40400000u, 0x40800000u, 0x00000000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x3f800000u, 0x3f800000u, 0x40000000u, 0x00000000u},
       .expected = {0x00000000u, 0x40400000u, 0x41000000u, 0x00000000u},
       .blend = 0x61000204u,
       .blend_opt = 0x01100124u,
       .initial = {0x80000000u, 0x40400000u, 0x40800000u, 0x00000000u}},
      // Saturation retains factor selection, tie behavior, and comparison precision.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u},
       .expected = {0, 0, 0, 0x3f800000u},
       .blend = 0x6100010au,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0x3f800000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0xbf653334u, 0x3fa10000u, 0xc1023334u, 0x3ebb3333u},
       .expected = {0xc014f348u, 0x3f35bb33u, 0x3cca8ee1u, 0x3f5d999au},
       .blend = 0x4000010au,
       .initial = {0xc0000000u, 0x3e800000u, 0x40400000u, 0x3f000000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x00800000u, 0x00800000u, 0x00800000u, 0xff7fffffu},
       .expected = {0xc03fffffu, 0xc03fffffu, 0xc03fffffu, 0x7f7fffffu},
       .blend = 0x6100010au,
       .initial = {0x3f800001u, 0x3f800001u, 0x3f800001u, 0x7f7fffffu}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x4b0b53f3u, 0x4b0b53f3u, 0x4b0b53f3u, 0x3f662b48u},
       .expected = {0x3ddb8000u, 0x3ddb8000u, 0x3ddb8000u, 0x3dcea5c0u},
       .blend = 0x6100010au,
       .initial = {0xcafa89f5u, 0xcafa89f5u, 0xcafa89f5u, 0x3dcea5c0u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x80000000u, 0x80000000u, 0x80000000u, 0x3f800000u},
       .expected = {0x00000000u, 0x00000000u, 0x00000000u, 0x00800000u},
       .blend = 0x6100010au,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0x00800000u}},
      // Physical constant-alpha and mixed inverse-factor cancellation.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0xaf0060a4u, 0xaf0060a4u, 0xaf0060a4u, 0x4052fde5u},
       .expected = {0x1d800000u, 0x1d800000u, 0x1d800000u, 0x3f7fe89au},
       .blend = 0x61000507u,
       .initial = {0xa8a37ba9u, 0xa8a37ba9u, 0xa8a37ba9u, 0x3f7fe89au}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x80000000u, 0x80000000u, 0x80000000u, 0},
       .expected = {0x80000000u, 0x80000000u, 0x80000000u, 0},
       .blend = 0x61000504u,
       .blend_opt = 0x01100154u,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x3f883334u, 0x3fb1cccdu, 0xbf066667u, 0x4058e666u},
       .expected = {0xbc08f5cfu, 0x3f7d8a3du, 0x3f3570a5u, 0x40182f5cu},
       .blend = 0x40001211u,
       .initial = {0xc0000000u, 0x3e800000u, 0x40400000u, 0x3f000000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x3ceec6c7u, 0x3ceec6c7u, 0x3ceec6c7u, 0xbe9368dbu},
       .expected = {0xaf2a0000u, 0xaf2a0000u, 0xaf2a0000u, 0x4056d022u},
       .blend = 0x61000705u,
       .initial = {0x3c828093u, 0x3c828093u, 0x3c828093u, 0x4056d022u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x395addbeu, 0x395addbeu, 0x395addbeu, 0xb07442afu},
       .expected = {0x2cb1e000u, 0x2cb1e000u, 0x2cb1e000u, 0xc0e932bbu},
       .blend = 0x61000705u,
       .initial = {0xb7d34666u, 0xb7d34666u, 0xb7d34666u, 0xc0e932bbu}},
      // Physical RDNA3/4 cancellation, factors, sticky rounding, and underflow.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0xbf800000u, 0xbf800000u, 0xbf800000u, 0x80000000u},
       .expected = {0xbf800000u, 0xbf800000u, 0xbf800000u, 0x80000000u},
       .blend = 0x61000104u,
       .blend_opt = 0x01100114u,
       .initial = {0xbf800000u, 0xbf800000u, 0xbf800000u, 0x80000000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x80800000u, 0x80800000u, 0x80800000u, 0x3f7fffffu},
       .expected = {0x80000000u, 0x80000000u, 0x80000000u, 0x3f800000u},
       .blend = 0x61000604u,
       .initial = {0, 0, 0, 0x3f800000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x00800001u, 0x00800001u, 0x00800001u, 0x3f7ffffeu},
       .expected = {0x00800000u, 0x00800000u, 0x00800000u, 0x3f800000u},
       .blend = 0x61000604u,
       .initial = {0, 0, 0, 0x3f800000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0xbf800000u, 0xbf800000u, 0xbf800000u, 0x40000000u},
       .expected = {0, 0, 0, 0xc0000000u},
       .blend = 0x61000000u,
       .initial = {0xc0000000u, 0xc0000000u, 0xc0000000u, 0xc0000000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x3ffbe98eu, 0x3ffbe98eu, 0x3ffbe98eu, 0x3fe52bfcu},
       .expected = {0x336b0000u, 0x336b0000u, 0x336b0000u, 0xc0618334u},
       .blend = 0x61000104u,
       .initial = {0xc0618334u, 0xc0618334u, 0xc0618334u, 0xc0618334u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x3ffbe98eu, 0x3ffbe98eu, 0x3ffbe98eu, 0x3fe52bfcu},
       .expected = {0x336b0000u, 0x336b0000u, 0x336b0000u, 0x3f800000u},
       .blend = 0x61000604u,
       .initial = {0xc0618334u, 0xc0618334u, 0xc0618334u, 0x3f800000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0x3fa44755u, 0x3fa44755u, 0x3fa44755u, 0x3fd24517u},
       .expected = {0x32c80000u, 0x32c80000u, 0x32c80000u, 0xc000141eu},
       .blend = 0x61000604u,
       .initial = {0x3f86d9a8u, 0x3f86d9a8u, 0x3f86d9a8u, 0xc000141eu}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0xbb33749eu, 0xbb33749eu, 0xbb33749eu, 0x94b0b5edu},
       .expected = {0xf27fbcf7u, 0xf27fbcf7u, 0xf27fbcf7u, 0xd9b9d285u},
       .blend = 0x61000604u,
       .initial = {0x583028fcu, 0x583028fcu, 0x583028fcu, 0xd9b9d285u}},

      // Physical FP16/FP32 blend captures include product precision and RTZ output.
      {.data_format = 12,
       .number_format = 7,
       .export_format = 4,
       .bytes = 8,
       .mask = 3,
       .exported = {0xbff0c3d8u, 0x2c00c7e8u},
       .expected = {0x2f10c03du, 0x379040a3u},
       .blend = 0x40000504,
       .initial = {0x3400c000u, 0x38004200u}},
      {.data_format = 12,
       .number_format = 7,
       .export_format = 4,
       .bytes = 8,
       .mask = 3,
       .exported = {0xbff0c3f8u, 0x0000c7f8u},
       .expected = {0xb6b8b818u, 0x319ac95bu},
       .blend = 0x40000c0b,
       .initial = {0x3400c000u, 0x38004200u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0xbf943334u, 0x3fb1cccdu, 0xc116999au, 0x3d666666u},
       .expected = {0xbff9efaeu, 0x3ea0ce14u, 0x401350a4u, 0x3ef33852u},
       .blend = 0x40000504,
       .initial = {0xc0000000u, 0x3e800000u, 0x40400000u, 0x3f000000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0xbf956667u, 0x3fb1cccdu, 0xc1173334u, 0x3d2cccccu},
       .expected = {0xc027f999u, 0x3f177ae2u, 0xc1490001u, 0x3e4f47afu},
       .blend = 0x40000c0b,
       .initial = {0xc0000000u, 0x3e800000u, 0x40400000u, 0x3f000000u}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 15,
       .exported = {0xbf653334u, 0x3fb1cccdu, 0xc105cccdu, 0x3ee66666u},
       .expected = {0x3f8d6666u, 0x3f91cccdu, 0xc135cccdu, 0xbd4cccd0u},
       .blend = 0x40000121,
       .initial = {0xc0000000u, 0x3e800000u, 0x40400000u, 0x3f000000u}},

      // Full-width float and integer values for R32/RG32 attachments.
      {.data_format = 4,
       .number_format = 7,
       .export_format = 1,
       .bytes = 4,
       .components = 1,
       .mask = 1,
       .exported = {0xbf812345, 0x3eabcdef},
       .expected = {0xbf812345, 0x3eabcdef}},
      {.data_format = 4,
       .number_format = 4,
       .export_format = 1,
       .bytes = 4,
       .components = 1,
       .mask = 1,
       .exported = {0x87654321, 0xfedcba98},
       .expected = {0x87654321, 0xfedcba98}},
      {.data_format = 4,
       .number_format = 5,
       .export_format = 1,
       .bytes = 4,
       .components = 1,
       .mask = 1,
       .exported = {0x80000001, 0x7ffffffe},
       .expected = {0x80000001, 0x7ffffffe}},
      {.data_format = 11,
       .number_format = 7,
       .export_format = 2,
       .bytes = 8,
       .components = 2,
       .mask = 3,
       .exported = {0xbf812345, 0x3eabcdef},
       .expected = {0xbf812345, 0x3eabcdef}},
      {.data_format = 11,
       .number_format = 4,
       .export_format = 2,
       .bytes = 8,
       .components = 2,
       .mask = 3,
       .exported = {0x87654321, 0xfedcba98},
       .expected = {0x87654321, 0xfedcba98}},
      {.data_format = 11,
       .number_format = 5,
       .export_format = 2,
       .bytes = 8,
       .components = 2,
       .mask = 3,
       .exported = {0x80000001, 0x7ffffffe},
       .expected = {0x80000001, 0x7ffffffe}},

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
      // Signed packed exports sign-extend integers and normalize signed endpoints.
      {.data_format = 12,
       .number_format = 1,
       .export_format = 6,
       .bytes = 8,
       .mask = 3,
       .exported = {0x80018000, 0x7fff0001},
       .expected = {0x80018001, 0x7fff0001}},
      {.data_format = 12,
       .number_format = 1,
       .export_format = 6,
       .bytes = 8,
       .mask = 3,
       .exported = {0xc001ffff, 0x40010000},
       .expected = {0xc001ffff, 0x40010000}},
      {.data_format = 12,
       .number_format = 5,
       .export_format = 8,
       .bytes = 8,
       .mask = 3,
       .exported = {0x80007fff, 0xffff0001},
       .expected = {0x80007fff, 0xffff0001}},
      {.data_format = 14,
       .number_format = 5,
       .export_format = 8,
       .bytes = 16,
       .mask = 3,
       .exported = {0x80007fff, 0xffff0001},
       .expected = {0x00007fff, 0xffff8000, 0x00000001, 0xffffffff}},
      // Packed UNORM16 exports retain every channel bit, including endpoints.
      {.data_format = 12,
       .number_format = 0,
       .export_format = 5,
       .bytes = 8,
       .mask = 3,
       .exported = {0x00010000, 0xfffffffe},
       .expected = {0x00010000, 0xfffffffe}},
      {.data_format = 12,
       .number_format = 0,
       .export_format = 5,
       .bytes = 8,
       .mask = 3,
       .exported = {0x7fff8000, 0xabc12345},
       .expected = {0x7fff8000, 0xabc12345}},
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
      // RGB-only and sparse component exports preserve the remaining channels.
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 7,
       .exported = {0x3e800000, 0x3f000000, 0x3f400000, 0x3f800000},
       .expected = {0x3e800000, 0x3f000000, 0x3f400000, 0xcccccccc}},
      {.data_format = 14,
       .number_format = 7,
       .export_format = 9,
       .bytes = 16,
       .mask = 10,
       .exported = {0x3e800000, 0x3f000000, 0x3f400000, 0x3f800000},
       .expected = {0xcccccccc, 0x3f000000, 0xcccccccc, 0x3f800000}},
      {.data_format = 12,
       .number_format = 7,
       .export_format = 4,
       .bytes = 8,
       .mask = 1,
       .exported = {0x38003400, 0x3c003a00},
       .expected = {0x38003400, 0xcccccccc}},
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
    for (uint32_t write_mask : {1u, 2u, 5u, 15u}) {
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
      context[gfx12 ? 0x31b : 0x31d] = test.components < 4 ? 1u << 2 : 0; // FORCE_DST_ALPHA_1
      context[0x318] = 0x1000;
      context[0x1e0] = test.blend;
      context[0x1d8] = test.blend_opt;
      for (uint32_t c = 0; c < 4; ++c)
        context[0x105 + c] = test.constant_bits[c];
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
        memory_.write32(0x100000 + offset,
                        offset < test.bytes ? test.initial[offset / 4] : 0xcccccccc);

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
        const uint32_t component = byte / (test.bytes / test.components);
        const uint8_t expected =
            ((write_mask & (1u << component)) ? test.expected[byte / 4] : test.initial[byte / 4]) >>
            (8 * (byte % 4));
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

TEST_P(GraphicsExportTest, HardwareSampleFiltersFp16AndMipLevels) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    const char *name;
    std::array<std::array<uint16_t, 4>, 4> texels;
    std::array<uint16_t, 4> mip;
    std::array<std::array<uint32_t, 4>, 6> expected;
  };
  // Raw outputs from the same 2x2 texture and 1x1 mip on physical gfx1100/gfx1201.
  const Case cases[] = {
      {.name = "positive finite",
       .texels = {{{0x0000, 0x3801, 0x3c01, 0x0001},
                   {0x3c00, 0x2001, 0x5bff, 0x03ff},
                   {0x3bff, 0x1001, 0x7bff, 0x0400},
                   {0x3401, 0x2c01, 0x4c10, 0x0801}}},
       .mip = {0x2c01, 0xbc01, 0x7801, 0x1001},
       .expected = {{{0x00000000, 0x3f002000, 0x3f802000, 0x33800000},
                     {0x3e7fe000, 0x3ec04000, 0x467fe000, 0x37806000},
                     {0x3ec7f400, 0x3e92f000, 0x4640a000, 0x38002000},
                     {0x3d002000, 0xbe802000, 0x46802100, 0x39802400},
                     {0x3e67fc00, 0xbeb6c800, 0x46b04800, 0x39882200},
                     {0x3d802000, 0xbf802000, 0x47002000, 0x3a002000}}}},
      {.name = "signed finite",
       .texels = {{{0x0000, 0xb801, 0x3c01, 0x8001},
                   {0xbc00, 0x2001, 0xdbff, 0x03ff},
                   {0xbbff, 0x1001, 0xfbff, 0x0400},
                   {0x3401, 0xac01, 0x4c10, 0x8801}}},
       .mip = {0x2c01, 0xbc01, 0x7801, 0x1001},
       .expected = {{{0x00000000, 0xbf002000, 0x3f802000, 0xb3800000},
                     {0xbe7fe000, 0xbec02000, 0xc67fe000, 0x377f4000},
                     {0xbeb7f400, 0xbe915800, 0xc6409800, 0x377f2000},
                     {0x3d002000, 0xbf403000, 0x46802100, 0x39801c00},
                     {0xbe17ec00, 0xbf247600, 0x461ff400, 0x39841c80},
                     {0x3d802000, 0xbf802000, 0x47002000, 0x3a002000}}}},
      {.name = "NaN and infinity",
       .texels = {{{0x7c00, 0xfc00, 0x7c01, 0x7e01},
                   {0x3c00, 0x3c00, 0x3c00, 0x3c00},
                   {0x4000, 0xfc00, 0x7c00, 0x0000},
                   {0x7c00, 0x7c00, 0xfc00, 0xfc01}}},
       .mip = {0xfc00, 0x7c00, 0x7e00, 0x3c00},
       .expected = {{{0x7f800000, 0xff800000, 0xffc00000, 0xffc00000},
                     {0x7f800000, 0xff800000, 0xffc00000, 0xffc00000},
                     {0x7f800000, 0xffc00000, 0xffc00000, 0xffc00000},
                     {0xffc00000, 0xffc00000, 0xffc00000, 0xffc00000},
                     {0xffc00000, 0xffc00000, 0xffc00000, 0xffc00000},
                     {0xff800000, 0x7f800000, 0xffc00000, 0x3f800000}}}},
      {.name = "mixed signed zeros",
       .texels = {{{0x0000, 0x8000, 0x0000, 0x8000},
                   {0x8000, 0x8000, 0x0000, 0x0000},
                   {0x0000, 0x0000, 0x0000, 0x8000},
                   {0x8000, 0x0000, 0x0000, 0x8000}}},
       .mip = {0x2c01, 0xbc01, 0x7801, 0x1001},
       .expected = {{{0x00000000, 0x80000000, 0x00000000, 0x80000000},
                     {0x00000000, 0x00000000, 0x00000000, 0x00000000},
                     {0x00000000, 0x00000000, 0x00000000, 0x00000000},
                     {0x3d002000, 0xbf002000, 0x46802000, 0x39802000},
                     {0x3d002000, 0xbf002000, 0x46802000, 0x39802000},
                     {0x3d802000, 0xbf802000, 0x47002000, 0x3a002000}}}},
      {.name = "subnormals",
       .texels = {{{0x0001, 0x0002, 0x0003, 0x03ff},
                   {0x0003, 0x003f, 0x0081, 0x0101},
                   {0x0011, 0x0012, 0x0103, 0x0201},
                   {0x0201, 0x0245, 0x0000, 0x8001}}},
       .mip = {0x2c01, 0xbc01, 0x7801, 0x1001},
       .expected = {{{0x33800000, 0x34000000, 0x34400000, 0x387fc000},
                     {0x34a00000, 0x34c00000, 0x36860000, 0x385fe000},
                     {0x36118000, 0x36528000, 0x3694e000, 0x3833f000},
                     {0x3d002008, 0xbf001fff, 0x46802000, 0x39901c00},
                     {0x3d002123, 0xbf001fe6, 0x46802000, 0x398b5f00},
                     {0x3d802000, 0xbf802000, 0x47002000, 0x3a002000}}}},
      {.name = "large cancellation",
       .texels = {{{0x7bff, 0xfbff, 0x7bff, 0x7001},
                   {0xfbff, 0x7bff, 0x03ff, 0xf001},
                   {0x0001, 0x4000, 0xfbff, 0x4001},
                   {0x4000, 0xc001, 0x3c01, 0xc001}}},
       .mip = {0x2c01, 0xbc01, 0x7801, 0x1001},
       .expected = {{{0x477fe000, 0xc77fe000, 0x477fe000, 0x46002000},
                     {0x473fe800, 0xc73fe800, 0x46ffe000, 0x45c03000},
                     {0x46bfe800, 0xc6bfe800, 0x46bfe800, 0x45403000},
                     {0x46ffe010, 0xc6ffe100, 0x47400000, 0x45802000},
                     {0x463fe820, 0xc63fea00, 0x46e01400, 0x44c03002},
                     {0x3d802000, 0xbf802000, 0x47002000, 0x3a002000}}}},
      {.name = "negative zeros",
       .texels = {{{0x8000, 0x8000, 0x8000, 0x8000},
                   {0x8000, 0x8000, 0x8000, 0x8000},
                   {0x8000, 0x8000, 0x8000, 0x8000},
                   {0x8000, 0x8000, 0x8000, 0x8000}}},
       .mip = {0x8000, 0x8000, 0x8000, 0x8000},
       .expected = {{{0x80000000, 0x80000000, 0x80000000, 0x80000000},
                     {0x00000000, 0x00000000, 0x00000000, 0x00000000},
                     {0x00000000, 0x00000000, 0x00000000, 0x00000000},
                     {0x00000000, 0x00000000, 0x00000000, 0x00000000},
                     {0x00000000, 0x00000000, 0x00000000, 0x00000000},
                     {0x80000000, 0x80000000, 0x80000000, 0x80000000}}}},
      {.name = "zero-weight NaN and infinity",
       .texels = {{{0x3c00, 0x8000, 0x3c00, 0x8000},
                   {0x7e01, 0x8000, 0x7c00, 0x8000},
                   {0x4200, 0x8000, 0x4200, 0x8000},
                   {0x7c00, 0x8000, 0xfc00, 0x8000}}},
       .mip = {0x2c01, 0xbc01, 0x7801, 0x1001},
       .expected = {{{0x3f800000, 0x80000000, 0x3f800000, 0x80000000},
                     {0x3fc00000, 0x00000000, 0x3fc00000, 0x00000000},
                     {0xffc00000, 0x00000000, 0xffc00000, 0x00000000},
                     {0x3f080200, 0xbf002000, 0x46802100, 0x39802000},
                     {0xffc00000, 0xbf002000, 0xffc00000, 0x39802000},
                     {0x3d802000, 0xbf802000, 0x47002000, 0x3a002000}}}},
  };
  constexpr struct Sample {
    const char *name;
    uint32_t index;
  } samples[] = {{"texel center", 0},
                 {"two contributing texels", 4},
                 {"four contributing texels", 5},
                 {"mip blend at texel centers", 32768},
                 {"mip and bilinear blend", 32773},
                 {"clamped mip endpoint", 65520}};
  for (uint32_t components : {1u, 2u, 4u}) {
    for (uint32_t pattern = 0; pattern < std::size(cases); ++pattern) {
      SCOPED_TRACE(components);
      const auto &test = cases[pattern];
      SCOPED_TRACE(test.name);
      const uint64_t base = 0x100000 + components * 0x100000 + pattern * 0x10000;
      const uint32_t bytes = components * 2;
      for (uint32_t level = 0; level < 2; ++level) {
        const auto mip = amdgpu::image_mip_layout(gfx12, 0, bytes, 2, 2, 2, level);
        ASSERT_TRUE(mip);
        for (uint32_t y = 0; y < mip->height; ++y)
          for (uint32_t x = 0; x < mip->width; ++x) {
            const auto address =
                gfx12 ? amdgpu::gfx12_image_address(base + mip->offset, x, y, mip->pitch, bytes, 0)
                      : amdgpu::gfx11_image_address(base + mip->offset, x, y, mip->pitch, bytes, 0);
            ASSERT_TRUE(address);
            const auto &texel = level ? test.mip : test.texels[y * 2 + x];
            ASSERT_EQ(access_->write(*address, std::as_bytes(std::span(texel).first(components))),
                      amdgpu::VmAccessOutcome::Complete);
          }
      }
      const uint32_t format = components == 1 ? 13 : components == 2 ? 29 : 57;
      // Missing Vulkan components are supplied by the descriptor selectors.
      const uint32_t selectors = components == 1 ? 0x204 : components == 2 ? 0x22c : 0xfac;
      const std::array<uint32_t, 8> descriptor{uint32_t(base >> 8),
                                               (format << (gfx12 ? 17 : 20)) | (1u << 30) |
                                                   (1u << (gfx12 ? 12 : 16)),
                                               1u << 14,
                                               (9u << 28) | selectors | (1u << (gfx12 ? 15 : 16)),
                                               0,
                                               0,
                                               0,
                                               0};
      for (uint32_t r = 0; r < descriptor.size(); ++r)
        wave_->debug_write_sgpr(8 + r, descriptor[r]);
      wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6));
      wave_->debug_write_sgpr(5, 256u << (gfx12 ? 13 : 12));
      wave_->debug_write_sgpr(6, (1 << 20) | (1 << 22) | (2 << 26));
      wave_->debug_write_sgpr(7, 0);
      wave_->set_exec(5);
      for (uint32_t sample = 0; sample < std::size(samples); ++sample) {
        SCOPED_TRACE(samples[sample].name);
        const uint32_t index = samples[sample].index;
        const float u = 0.25f + (index & 3u) / 8.0f;
        const float v = 0.25f + ((index >> 2) & 3u) / 8.0f;
        const float lod = (index >> 4) / 4096.0f;
        for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
          wave_->debug_write_vgpr(6, lane, std::bit_cast<uint32_t>(u));
          wave_->debug_write_vgpr(0, lane, std::bit_cast<uint32_t>(v));
          wave_->debug_write_vgpr(4, lane, std::bit_cast<uint32_t>(lod));
          for (uint32_t c = 0; c < 4; ++c)
            wave_->debug_write_vgpr(8 + c, lane, 0xdeadbeef);
        }
        std::array<uint32_t, 4> words{};
        if (gfx12) {
          const auto encoded = rdna4::build_vsample(29, {.dim = 1,
                                                         .dmask = 15,
                                                         .vdata = 8,
                                                         .rsrc = 8,
                                                         .samp = 4,
                                                         .vaddr0 = 6,
                                                         .vaddr1 = 0,
                                                         .vaddr2 = 4});
          std::copy(encoded.begin(), encoded.end(), words.begin());
        } else {
          const auto encoded = rdna3::build_mimg(
              29,
              {.nsa = 1, .dim = 1, .dmask = 15, .vaddr = 6, .vdata = 8, .srsrc = 2, .ssamp = 1});
          std::copy(encoded.begin(), encoded.end(), words.begin());
          words[2] = 4 << 8;
        }
        auto decoded = decoder_->decode(words.data());
        ASSERT_FALSE(decoded.failed());
        auto instruction = std::move(decoded).value();
        ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
        ASSERT_FALSE(wave_->instruction_execution_failed());
        ASSERT_NE(instruction->data(), nullptr);
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
        pipeline.issue(instruction.release(), *wave_);
        for (uint32_t c = 0; c < 4; ++c) {
          const uint32_t expected = c < components ? test.expected[sample][c]
                                    : c == 3       ? 0x3f800000u
                                                   : 0;
          for (uint32_t lane : {0u, 2u})
            EXPECT_EQ(wave_->debug_read_vgpr(8 + c, lane), expected) << c;
          EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 1), 0xdeadbeefu);
        }
      }
    }
  }
}

TEST_P(GraphicsExportTest, NearestSamplingAndImageLoadsUseDistinctFloatRules) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint32_t kR16Float = 13, kRg16Float = 29, kRgba16Float = 57;
  constexpr uint32_t kR32Float = 22, kRg32Float = 50, kRgba32Float = 63;
  constexpr uint32_t kR32Uint = 20, kRg32Uint = 48, kRgba32Uint = 61;
  constexpr uint32_t kImageLoad = 0, kImageSampleLz = 31;
  struct Case {
    const char *name;
    uint32_t format;
    std::array<uint32_t, 4> raw;
    std::array<uint32_t, 4> loaded;
    std::array<uint32_t, 4> sampled;
  };
  // Actual image-load and nearest-sample results from both physical RDNA3/4
  // cards, including half NaN payloads and single-precision subnormals.
  constexpr Case cases[] = {
      {.name = "FP16 zero and finite",
       .format = kRgba16Float,
       .raw = {0x00000000u, 0x00000089u, 0x00000112u, 0x0000019bu},
       .loaded = {0x00000000u, 0x37090000u, 0x37890000u, 0x37cd8000u},
       .sampled = {0x00000000u, 0x37090000u, 0x37890000u, 0x37cd8000u}},
      {.name = "FP16 positive subnormal",
       .format = kRgba16Float,
       .raw = {0x00000001u, 0x0000008au, 0x00000113u, 0x0000019cu},
       .loaded = {0x33800000u, 0x370a0000u, 0x37898000u, 0x37ce0000u},
       .sampled = {0x33800000u, 0x370a0000u, 0x37898000u, 0x37ce0000u}},
      {.name = "FP16 signaling NaN",
       .format = kRgba16Float,
       .raw = {0x00007c01u, 0x00007c8au, 0x00007d13u, 0x00007d9cu},
       .loaded = {0x7f802000u, 0x7f914000u, 0x7fa26000u, 0x7fb38000u},
       .sampled = {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
      {.name = "FP16 quiet NaN",
       .format = kRgba16Float,
       .raw = {0x00007e00u, 0x00007e89u, 0x00007f12u, 0x00007f9bu},
       .loaded = {0x7fc00000u, 0x7fd12000u, 0x7fe24000u, 0x7ff36000u},
       .sampled = {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
      {.name = "FP16 negative zero",
       .format = kRgba16Float,
       .raw = {0x00008000u, 0x00008089u, 0x00008112u, 0x0000819bu},
       .loaded = {0x80000000u, 0xb7090000u, 0xb7890000u, 0xb7cd8000u},
       .sampled = {0x80000000u, 0xb7090000u, 0xb7890000u, 0xb7cd8000u}},
      {.name = "FP16 negative signaling NaN",
       .format = kRgba16Float,
       .raw = {0x0000fc01u, 0x0000fc8au, 0x0000fd13u, 0x0000fd9cu},
       .loaded = {0xff802000u, 0xff914000u, 0xffa26000u, 0xffb38000u},
       .sampled = {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
      {.name = "FP32 zero and finite",
       .format = kRgba32Float,
       .raw = {0x00000000u, 0x00890000u, 0x01120000u, 0x019b0000u},
       .loaded = {0x00000000u, 0x00890000u, 0x01120000u, 0x019b0000u},
       .sampled = {0x00000000u, 0x00890000u, 0x01120000u, 0x019b0000u}},
      {.name = "FP32 positive subnormal",
       .format = kRgba32Float,
       .raw = {0x00010049u, 0x008a0049u, 0x01130049u, 0x019c0049u},
       .loaded = {0x00010049u, 0x008a0049u, 0x01130049u, 0x019c0049u},
       .sampled = {0x00000000u, 0x008a0049u, 0x01130049u, 0x019c0049u}},
      {.name = "FP32 signaling NaN",
       .format = kRgba32Float,
       .raw = {0x7f805b80u, 0x80095b80u, 0x80925b80u, 0x811b5b80u},
       .loaded = {0x7f805b80u, 0x80095b80u, 0x80925b80u, 0x811b5b80u},
       .sampled = {0xffc00000u, 0x80000000u, 0x80925b80u, 0x811b5b80u}},
      {.name = "FP32 quiet NaN",
       .format = kRgba32Float,
       .raw = {0x7fc06dc0u, 0x80496dc0u, 0x80d26dc0u, 0x815b6dc0u},
       .loaded = {0x7fc06dc0u, 0x80496dc0u, 0x80d26dc0u, 0x815b6dc0u},
       .sampled = {0xffc00000u, 0x80000000u, 0x80d26dc0u, 0x815b6dc0u}},
      {.name = "FP32 negative subnormal",
       .format = kRgba32Float,
       .raw = {0x80008000u, 0x80898000u, 0x81128000u, 0x819b8000u},
       .loaded = {0x80008000u, 0x80898000u, 0x81128000u, 0x819b8000u},
       .sampled = {0x80000000u, 0x80898000u, 0x81128000u, 0x819b8000u}},
      {.name = "FP32 negative subnormal payload",
       .format = kRgba32Float,
       .raw = {0x80018049u, 0x808a8049u, 0x81138049u, 0x819c8049u},
       .loaded = {0x80018049u, 0x808a8049u, 0x81138049u, 0x819c8049u},
       .sampled = {0x80000000u, 0x808a8049u, 0x81138049u, 0x819c8049u}},
      {.name = "FP32 negative signaling NaN",
       .format = kRgba32Float,
       .raw = {0xff80db80u, 0x0009db80u, 0x0092db80u, 0x011bdb80u},
       .loaded = {0xff80db80u, 0x0009db80u, 0x0092db80u, 0x011bdb80u},
       .sampled = {0xffc00000u, 0x00000000u, 0x0092db80u, 0x011bdb80u}},
      {.name = "FP32 negative quiet NaN",
       .format = kRgba32Float,
       .raw = {0xffc0edc0u, 0x0049edc0u, 0x00d2edc0u, 0x015bedc0u},
       .loaded = {0xffc0edc0u, 0x0049edc0u, 0x00d2edc0u, 0x015bedc0u},
       .sampled = {0xffc00000u, 0x00000000u, 0x00d2edc0u, 0x015bedc0u}},
      {.name = "UINT subnormal bits",
       .format = kRgba32Uint,
       .raw = {0x00010049u, 0x008a0049u, 0x01130049u, 0x019c0049u},
       .loaded = {0x00010049u, 0x008a0049u, 0x01130049u, 0x019c0049u},
       .sampled = {0x00010049u, 0x008a0049u, 0x01130049u, 0x019c0049u}},
      {.name = "UINT NaN bits",
       .format = kRgba32Uint,
       .raw = {0x7f805b80u, 0x80095b80u, 0x80925b80u, 0x811b5b80u},
       .loaded = {0x7f805b80u, 0x80095b80u, 0x80925b80u, 0x811b5b80u},
       .sampled = {0x7f805b80u, 0x80095b80u, 0x80925b80u, 0x811b5b80u}},
      {.name = "FP32 normal boundary",
       .format = kRgba32Float,
       .raw = {0x007fffffu, 0x00800000u, 0x807fffffu, 0x80800000u},
       .loaded = {0x007fffffu, 0x00800000u, 0x807fffffu, 0x80800000u},
       .sampled = {0x00000000u, 0x00800000u, 0x80000000u, 0x80800000u}},
      {.name = "FP32 infinities and signed zero",
       .format = kRgba32Float,
       .raw = {0x7f800000u, 0xff800000u, 0x00000000u, 0x80000000u},
       .loaded = {0x7f800000u, 0xff800000u, 0x00000000u, 0x80000000u},
       .sampled = {0x7f800000u, 0xff800000u, 0x00000000u, 0x80000000u}},
      {.name = "FP32 extreme NaN payloads",
       .format = kRgba32Float,
       .raw = {0x7f800001u, 0xff800001u, 0x7fffffffu, 0xffffffffu},
       .loaded = {0x7f800001u, 0xff800001u, 0x7fffffffu, 0xffffffffu},
       .sampled = {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
  };
  constexpr uint64_t base = 0x180000;
  wave_->set_exec(5);
  // Sampling flushes FP32 texel subnormals even when VALU preserves them.
  wave_->set_mode_raw(0xf0);
  for (uint32_t components : {1u, 2u, 4u}) {
    SCOPED_TRACE(components);
    for (const auto &test : cases) {
      SCOPED_TRACE(test.name);
      const bool fp16 = test.format == kRgba16Float;
      std::array<uint16_t, 4> halves{};
      for (uint32_t c = 0; c < 4; ++c)
        halves[c] = static_cast<uint16_t>(test.raw[c]);
      ASSERT_EQ(
          access_->write(
              base, fp16 ? std::as_bytes(std::span<const uint16_t>(halves).first(components))
                         : std::as_bytes(std::span<const uint32_t>(test.raw).first(components))),
          amdgpu::VmAccessOutcome::Complete);
      cache_.invalidate_all();
      cu_->l1_vector().invalidate_all();
      uint32_t format;
      if (fp16)
        format = components == 1 ? kR16Float : components == 2 ? kRg16Float : kRgba16Float;
      else if (test.format == kRgba32Float)
        format = components == 1 ? kR32Float : components == 2 ? kRg32Float : kRgba32Float;
      else
        format = components == 1 ? kR32Uint : components == 2 ? kRg32Uint : kRgba32Uint;
      // Missing Vulkan components are supplied by the descriptor selectors.
      const uint32_t selectors = components == 1 ? 0x204 : components == 2 ? 0x22c : 0xfac;
      const std::array<uint32_t, 8> descriptor{
          uint32_t(base >> 8), format << (gfx12 ? 17 : 20), 0, (9u << 28) | selectors, 0, 0, 0, 0};
      for (uint32_t r = 0; r < descriptor.size(); ++r)
        wave_->debug_write_sgpr(8 + r, descriptor[r]);
      wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6));
      for (uint32_t r = 5; r < 8; ++r)
        wave_->debug_write_sgpr(r, 0);
      for (bool sample : {false, true}) {
        SCOPED_TRACE(sample ? "IMAGE_SAMPLE_LZ" : "IMAGE_LOAD");
        for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
          wave_->debug_write_vgpr(0, lane, sample ? std::bit_cast<uint32_t>(0.5f) : 0);
          wave_->debug_write_vgpr(1, lane, sample ? std::bit_cast<uint32_t>(0.5f) : 0);
          for (uint32_t c = 0; c < 4; ++c)
            wave_->debug_write_vgpr(8 + c, lane, 0xdeadbeef);
        }
        std::array<uint32_t, 4> words{};
        if (gfx12) {
          if (sample) {
            const auto encoded = rdna4::build_vsample(kImageSampleLz, {.dim = 1,
                                                                       .dmask = 15,
                                                                       .vdata = 8,
                                                                       .rsrc = 8,
                                                                       .samp = 4,
                                                                       .vaddr0 = 0,
                                                                       .vaddr1 = 1});
            std::copy(encoded.begin(), encoded.end(), words.begin());
          } else {
            const auto encoded = rdna4::build_vimage(
                kImageLoad,
                {.dim = 1, .dmask = 15, .vdata = 8, .rsrc = 8, .vaddr0 = 0, .vaddr1 = 1});
            std::copy(encoded.begin(), encoded.end(), words.begin());
          }
        } else {
          const auto encoded = rdna3::build_mimg(
              sample ? kImageSampleLz : kImageLoad,
              {.dim = 1, .dmask = 15, .vaddr = 0, .vdata = 8, .srsrc = 2, .ssamp = 1});
          std::copy(encoded.begin(), encoded.end(), words.begin());
        }
        auto decoded = decoder_->decode(words.data());
        ASSERT_FALSE(decoded.failed());
        auto instruction = std::move(decoded).value();
        ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
        ASSERT_FALSE(wave_->instruction_execution_failed());
        ASSERT_NE(instruction->data(), nullptr);
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
        pipeline.issue(instruction.release(), *wave_);
        std::array<uint32_t, 4> expected = sample ? test.sampled : test.loaded;
        for (uint32_t c = components; c < 4; ++c)
          expected[c] = c == 3 ? (test.format == kRgba32Uint ? 1u : 0x3f800000u) : 0;
        for (uint32_t c = 0; c < 4; ++c) {
          for (uint32_t lane : {0u, 2u})
            EXPECT_EQ(wave_->debug_read_vgpr(8 + c, lane), expected[c]) << c;
          EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 1), 0xdeadbeefu);
        }
      }
    }
  }
}

TEST_P(GraphicsExportTest, SampleLodUsesMipViewsDerivativesAndBias) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t colors[] = {0xff000000, 0xff0000ff, 0xff00ff00, 0xffff0000};
  struct Case {
    const char *name;
    uint32_t opcode, mip_filter, first_level = 0;
    float lod, min_lod, max_lod, coordinate_step = 0;
    std::array<float, 4> expected;
    bool a16 = false;
    std::optional<std::array<float, 4>> gradient = std::nullopt;
    uint32_t width = 8, height = 8;
    uint32_t perf_mip = 0, perf_mod = 0;
    float sampler_bias = 0;
    int32_t secondary_bias = 0;
    uint32_t aniso_ratio = 0;
    float diagonal_offset = 0;
  };
  // Physical GFX11/12 mip weights, including a norm carry into the logarithm.
  const std::array rotated_gradient{0x1.f6cp-3f, -0x1.1ap-3f, -0x1.d48p-3f, 0x1.338p-4f};
  const std::array orthogonal_gradient{0.125f, 0.00250244140625f, -0.00250244140625f, 0.125f};
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
      {"half derivatives", 57, 1, 0, 0, 0, 3, 0.5f, {0, 1, 0, 1}},
      {"packed coordinates with half derivatives", 57, 1, 0, 0, 0, 3, 0.5f, {0, 1, 0, 1}, true},
      {"implicit derivatives", 27, 1, 0, 0, 0, 3, 0.5f, {0, 1, 0, 1}},
      // Hardware keeps coarse derivatives when the fourth lane's coordinates
      // are not affine. Fine derivatives would select the final blue mip.
      {.name = "coarse implicit derivatives",
       .opcode = 27,
       .mip_filter = 1,
       .lod = 0,
       .min_lod = 0,
       .max_lod = 3,
       .coordinate_step = 0.25f,
       .expected = {1, 0, 0, 1},
       .diagonal_offset = 2},
      {.name = "coarse packed implicit derivatives",
       .opcode = 27,
       .mip_filter = 1,
       .lod = 0,
       .min_lod = 0,
       .max_lod = 3,
       .coordinate_step = 0.25f,
       .expected = {1, 0, 0, 1},
       .a16 = true,
       .diagonal_offset = 2},
      {.name = "coarse biased derivatives",
       .opcode = 30,
       .mip_filter = 1,
       .lod = 1,
       .min_lod = 0,
       .max_lod = 3,
       .coordinate_step = 0.25f,
       .expected = {0, 1, 0, 1},
       .diagonal_offset = 2},
      {"shader bias", 30, 1, 0, 1, 0, 3, 0.25f, {0, 1, 0, 1}},
      {"forced zero", 31, 1, 0, 0, 0, 3, 0.5f, {0, 0, 0, 1}},
      {"packed explicit LOD", 29, 2, 0, 1.5f, 0, 3, 0, {0.5f, 0.5f, 0, 1}, true},
      {"packed coordinates with full derivatives", 28, 1, 0, 0, 0, 3, 0.5f, {0, 1, 0, 1}, true},
      {"packed implicit coordinates", 27, 1, 0, 0, 0, 3, 0.5f, {0, 1, 0, 1}, true},
      {"packed coordinates with full bias", 30, 1, 0, 1, 0, 3, 0.25f, {0, 1, 0, 1}, true},
      {"packed forced zero", 31, 1, 0, 0, 0, 3, 0.5f, {0, 0, 0, 1}, true},
      {"axis logarithm", 28, 2, 0, 0, 0, 3, 0x1.1e3cb6p-1f, {0, 0.8359375f, 0.1640625f, 1}},
      {"correlated derivatives",
       28,
       2,
       0,
       0,
       0,
       3,
       0,
       {0.83984375f, 0.16015625f, 0, 1},
       false,
       std::array{0.1875f, 0.0625f, 0.1875f, 0.0625f}},
      {"rotated derivatives",
       28,
       2,
       0,
       0,
       0,
       3,
       0,
       {0.4375f, 0.5625f, 0, 1},
       false,
       rotated_gradient},
      {"rotated half derivatives",
       57,
       2,
       0,
       0,
       0,
       3,
       0,
       {0.4375f, 0.5625f, 0, 1},
       false,
       rotated_gradient},
      {"rotated implicit derivatives",
       27,
       2,
       0,
       0,
       0,
       3,
       0,
       {0.4375f, 0.5625f, 0, 1},
       false,
       rotated_gradient},
      {"coordinate alignment", 28, 2, 0, 0, 0, 3, 0, {0, 0, 0, 1}, false, orthogonal_gradient},
      {"half coordinate alignment", 57, 2, 0, 0, 0, 3, 0, {0, 0, 0, 1}, false, orthogonal_gradient},
      {"implicit coordinate alignment",
       27,
       2,
       0,
       0,
       0,
       3,
       0,
       {0, 0, 0, 1},
       false,
       orthogonal_gradient},
      {"swapped coordinate exponents",
       28,
       2,
       0,
       0,
       0,
       3,
       0,
       {0, 0.53125f, 0.46875f, 1},
       false,
       std::array{0x1.90ef84p-4f, 0x1.30801p-1f, 0x1.30801p-1f, 0x1.90ef84p-4f}},
      {"negative coordinate alignment",
       28,
       2,
       0,
       0,
       0,
       3,
       0,
       {0.2734375f, 0.7265625f, 0, 1},
       false,
       std::array{-0.1253662109375f, -0.25f, -0.235595703125f, -0.2083740234375f}},
      {"non-power-of-two extents",
       28,
       2,
       0,
       0,
       0,
       3,
       0,
       {0, 0.70703125f, 0.29296875f, 1},
       false,
       std::array{-0x1.7b0788p-2f, 0x1.8f028cp-5f, -0x1.2460ecp-4f, -0x1.87502ap-2f},
       13,
       9},
      {"extent multiplication carry",
       28,
       2,
       0,
       0,
       0,
       3,
       0,
       {0, 0.3828125f, 0.6171875f, 1},
       false,
       std::array{0x1.47e36ep-5f, 0x1.91f32ep-2f, 0x1.c3679p-2f, 0x1.bc721p-3f},
       7,
       13},
      {"extent mantissa truncation",
       28,
       2,
       0,
       0,
       0,
       3,
       0,
       {0, 0.96875f, 0.03125f, 1},
       false,
       std::array{-0x1.119a6cp-13f, -0x1.35fd0ap-2f, -0x1.044334p-9f, 0x1.72db22p-5f},
       2051,
       9},
      {"zero coordinate exponent",
       28,
       2,
       0,
       0,
       0,
       3,
       0,
       {0.2265625f, 0.7734375f, 0, 1},
       false,
       std::array{0x1.5b3cd2p-10f, 0.0f, -0x1.0a3806p-10f, 0.0f},
       2051,
       9},
      {.name = "mip lower snap",
       .opcode = 29,
       .mip_filter = 2,
       .lod = 1.25f,
       .min_lod = 0,
       .max_lod = 3,
       .expected = {1, 0, 0, 1},
       .perf_mip = 10,
       .perf_mod = 4,
       .sampler_bias = 0,
       .secondary_bias = 0},
      {.name = "mip upper snap",
       .opcode = 29,
       .mip_filter = 2,
       .lod = 1.75f,
       .min_lod = 0,
       .max_lod = 3,
       .expected = {0, 1, 0, 1},
       .perf_mip = 10,
       .perf_mod = 4,
       .sampler_bias = 0,
       .secondary_bias = 0},
      {.name = "mip fraction truncation",
       .opcode = 29,
       .mip_filter = 2,
       .lod = 1.49609375f,
       .min_lod = 0,
       .max_lod = 3,
       .expected = {0.5078125f, 0.4921875f, 0, 1},
       .perf_mip = 7,
       .perf_mod = 4,
       .sampler_bias = 0,
       .secondary_bias = 0},
      {.name = "maximum mip gain",
       .opcode = 29,
       .mip_filter = 2,
       .lod = 1.50390625f,
       .min_lod = 0,
       .max_lod = 3,
       .expected = {0.453125f, 0.546875f, 0, 1},
       .perf_mip = 15,
       .perf_mod = 7,
       .sampler_bias = 0,
       .secondary_bias = 0},
      {.name = "disabled mip modulation",
       .opcode = 29,
       .mip_filter = 2,
       .lod = 1.50390625f,
       .min_lod = 0,
       .max_lod = 3,
       .expected = {0.49609375f, 0.50390625f, 0, 1},
       .perf_mip = 15,
       .perf_mod = 0,
       .sampler_bias = 0,
       .secondary_bias = 0},
      {.name = "explicit rounding before bias",
       .opcode = 29,
       .mip_filter = 2,
       .lod = 1.001953125f,
       .min_lod = 0,
       .max_lod = 3,
       .expected = {0.99609375f, 0.00390625f, 0, 1},
       .sampler_bias = 0.00390625f},
      {.name = "explicit sampler bias",
       .opcode = 29,
       .mip_filter = 2,
       .lod = 1,
       .min_lod = 0,
       .max_lod = 3,
       .expected = {0.75f, 0.25f, 0, 1},
       .perf_mip = 10,
       .perf_mod = 4,
       .sampler_bias = 0.375f,
       .secondary_bias = 0},
      {.name = "negative explicit sampler bias",
       .opcode = 29,
       .mip_filter = 2,
       .lod = 1,
       .min_lod = 0,
       .max_lod = 3,
       .expected = {0.75f, 0, 0, 1},
       .perf_mip = 10,
       .perf_mod = 4,
       .sampler_bias = -0.375f,
       .secondary_bias = 0},
      {.name = "zero sampler bias",
       .opcode = 31,
       .mip_filter = 2,
       .lod = 0,
       .min_lod = 0,
       .max_lod = 3,
       .expected = {0.25f, 0, 0, 1},
       .perf_mip = 10,
       .perf_mod = 4,
       .sampler_bias = 0.375f,
       .secondary_bias = 0},
      {.name = "bias bound before mip gain",
       .opcode = 31,
       .mip_filter = 2,
       .lod = 0,
       .min_lod = 0.375f,
       .max_lod = 3,
       .expected = {0.25f, 0, 0, 1},
       .perf_mip = 10,
       .perf_mod = 4,
       .sampler_bias = -0.375f,
       .secondary_bias = 0},
      {.name = "nearest mip ignores gain",
       .opcode = 29,
       .mip_filter = 1,
       .lod = 1,
       .min_lod = 0,
       .max_lod = 3,
       .expected = {0, 1, 0, 1},
       .perf_mip = 15,
       .perf_mod = 7,
       .sampler_bias = 0.625f,
       .secondary_bias = 0},
      {.name = "negative secondary bias floor",
       .opcode = 31,
       .mip_filter = 2,
       .lod = 0,
       .min_lod = 0,
       .max_lod = 3,
       .expected = {0.83203125f, 0.16796875f, 0, 1},
       .perf_mip = 10,
       .perf_mod = 1,
       .sampler_bias = 1.375f,
       .secondary_bias = -17},
      {.name = "positive secondary bias floor",
       .opcode = 31,
       .mip_filter = 2,
       .lod = 0,
       .min_lod = 0,
       .max_lod = 3,
       .expected = {0.109375f, 0.890625f, 0, 1},
       .perf_mip = 10,
       .perf_mod = 2,
       .sampler_bias = 1.375f,
       .secondary_bias = 23},
      {.name = "disabled secondary modulation",
       .opcode = 31,
       .mip_filter = 2,
       .lod = 0,
       .min_lod = 0,
       .max_lod = 3,
       .expected = {0.625f, 0.375f, 0, 1},
       .perf_mip = 10,
       .perf_mod = 0,
       .sampler_bias = 1.375f,
       .secondary_bias = 23},
      {.name = "negative minor-axis difference",
       .opcode = 28,
       .mip_filter = 2,
       .lod = 0,
       .min_lod = 0,
       .max_lod = 3,
       .expected = {0.49609375f, 0, 0, 1},
       .gradient = std::array{-0x1.15654ap-1f, 0x1.00d27ep-3f, 0x1.982c54p-3f, 0x1.24184ep-3f},
       .aniso_ratio = 2},
      {.name = "positive minor-axis difference",
       .opcode = 28,
       .mip_filter = 2,
       .lod = 0,
       .min_lod = 0,
       .max_lod = 3,
       .expected = {0.39453125f, 0.60546875f, 0, 1},
       .gradient = std::array{0x1.ea1552p-1f, -0x1.0b398ap-3f, -0x1.ad88b6p-1f, 0x1.53ed8ep-1f},
       .aniso_ratio = 2},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.name);
    cu_->l1_vector().invalidate_all();
    cache_.invalidate_all();
    for (uint32_t level = 0; level < 4; ++level) {
      const auto mip = amdgpu::image_mip_layout(gfx12, 0, 4, test.width, test.height, 4, level);
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
    const std::array<uint32_t, 8> descriptor{
        0x1000,
        (42u << (gfx12 ? 17 : 20)) | (((test.width - 1) & 3u) << 30) | (3u << (gfx12 ? 12 : 16)) |
            (gfx12 ? test.first_level << 25 : 0),
        ((test.width - 1) >> 2) | ((test.height - 1) << 14),
        (9u << 28) | 0xfac | (3u << (gfx12 ? 15 : 16)) | (gfx12 ? 0 : test.first_level << 12),
        0,
        test.perf_mod << 20,
        0,
        0};
    for (uint32_t r = 0; r < descriptor.size(); ++r)
      wave_->debug_write_sgpr(8 + r, descriptor[r]);
    wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6) | (test.aniso_ratio << 9));
    wave_->debug_write_sgpr(5, uint32_t(test.min_lod * 256) |
                                   (uint32_t(test.max_lod * 256) << (gfx12 ? 13 : 12)) |
                                   (gfx12 ? 0 : test.perf_mip << 24));
    wave_->debug_write_sgpr(6, (test.mip_filter << 26) | (test.aniso_ratio ? 15u << 20 : 0) |
                                   (gfx12 ? (test.perf_mip & 3) << 30 : 0) |
                                   (uint32_t(int32_t(test.sampler_bias * 256)) & 0x3fff) |
                                   ((uint32_t(test.secondary_bias) & 63) << 14));
    wave_->debug_write_sgpr(7, gfx12 ? test.perf_mip >> 2 : 0);
    wave_->set_exec(9);
    const auto gradient =
        test.gradient.value_or(std::array{test.coordinate_step, 0.0f, 0.0f, test.coordinate_step});
    // Populate inactive quad lanes as well: implicit derivatives consume them.
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
      const float offset = (lane & 3) == 3 ? test.diagonal_offset : 0;
      const float u = 0.25f + (lane & 1) * gradient[0] + ((lane >> 1) & 1) * gradient[2] + offset;
      const float v = 0.25f + (lane & 1) * gradient[1] + ((lane >> 1) & 1) * gradient[3] + offset;
      const std::array<uint32_t, 6> address_regs{6, 0, 4, 8, 9, 10};
      std::array<float, 6> values{u, v, test.lod};
      if (test.opcode == 28)
        values = {gradient[0], gradient[1], gradient[2], gradient[3], u, v};
      if (test.opcode == 57)
        values = {0, 0, u, v};
      if (test.opcode == 30)
        values = {test.lod, u, v};
      for (uint32_t r = 0; r < values.size(); ++r)
        wave_->debug_write_vgpr(address_regs[r], lane, std::bit_cast<uint32_t>(values[r]));
      if (test.opcode == 57) {
        wave_->debug_write_vgpr(address_regs[0], lane,
                                util::f32_to_f16(gradient[0]) |
                                    (uint32_t(util::f32_to_f16(gradient[1])) << 16));
        wave_->debug_write_vgpr(address_regs[1], lane,
                                util::f32_to_f16(gradient[2]) |
                                    (uint32_t(util::f32_to_f16(gradient[3])) << 16));
      }
      if (test.a16) {
        const uint32_t prefix = test.opcode == 28   ? 4
                                : test.opcode == 57 ? 2
                                : test.opcode == 30 ? 1
                                                    : 0;
        wave_->debug_write_vgpr(address_regs[prefix], lane,
                                util::f32_to_f16(values[prefix]) |
                                    (uint32_t(util::f32_to_f16(values[prefix + 1])) << 16));
        if (test.opcode == 29)
          wave_->debug_write_vgpr(address_regs[1], lane, util::f32_to_f16(test.lod));
      }
    }
    std::array<uint32_t, 4> words{};
    const uint8_t second_address = test.a16 && (test.opcode == 27 || test.opcode == 31) ? 255 : 0;
    const uint8_t fourth_address = test.a16 && test.opcode == 57 ? 255 : 8;
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
                                                              .vaddr3 = fourth_address});
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
      words[2] = second_address | (4 << 8) | (uint32_t(fourth_address) << 16) | (9 << 24);
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

TEST_P(GraphicsExportTest, PackedGradientComponentsSelectMipIndependentlyOfCoordinates) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  for (bool cube : {false, true}) {
    // The non-square 2D image distinguishes U/V halves. Cube neighbors select
    // opposite faces, so implicit derivatives would select a different mip.
    const uint32_t height = cube ? 8 : 4, layers = cube ? 6 : 1;
    cu_->l1_vector().invalidate_all();
    cache_.invalidate_all();
    for (uint32_t layer = 0; layer < layers; ++layer)
      for (uint32_t level = 0; level < 4; ++level) {
        const auto mip = amdgpu::image_mip_layout(gfx12, 0, 4, 8, height, 4, level);
        ASSERT_TRUE(mip);
        const uint64_t base = 0x100000 + layer * mip->slice_size + mip->offset;
        for (uint32_t y = 0; y < mip->height; ++y)
          for (uint32_t x = 0; x < mip->width; ++x) {
            const auto address = gfx12 ? amdgpu::gfx12_image_address(base, x, y, mip->pitch, 4, 0)
                                       : amdgpu::gfx11_image_address(base, x, y, mip->pitch, 4, 0);
            ASSERT_TRUE(address);
            memory_.write32(*address, 0xff000000 | (layer << 8) | level);
          }
      }
    const std::array<uint32_t, 8> descriptor{
        0x1000,
        (46u << (gfx12 ? 17 : 20)) | (3u << 30) | (3u << (gfx12 ? 12 : 16)),
        1 | ((height - 1) << 14),
        ((cube ? 11u : 9u) << 28) | 0xfac | (3u << (gfx12 ? 15 : 16)),
        layers - 1,
        0,
        0,
        0};
    for (uint32_t i = 0; i < descriptor.size(); ++i)
      wave_->debug_write_sgpr(8 + i, descriptor[i]);
    wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6));
    wave_->debug_write_sgpr(5, 768u << (gfx12 ? 13 : 12));
    wave_->debug_write_sgpr(6, 1u << 26);
    wave_->debug_write_sgpr(7, 0);
    wave_->set_exec(15);
    for (bool a16 : {false, true}) {
      SCOPED_TRACE(testing::Message() << "cube=" << cube << ", a16=" << a16);
      for (uint32_t lane = 0; lane < 4; ++lane) {
        // Each lane supplies exactly one nonzero explicit derivative. Body
        // coordinates are constant, so their quad differences cannot stand in.
        const uint32_t gradient = uint32_t(util::f32_to_f16(0.5f)) << (16 * (lane % 2));
        wave_->debug_write_vgpr(0, lane, lane < 2 ? gradient : 0);
        wave_->debug_write_vgpr(1, lane, lane >= 2 ? gradient : 0);
        const float coordinate = cube ? 1.25f : 0.25f;
        const uint32_t packed = util::f32_to_f16(coordinate);
        wave_->debug_write_vgpr(
            2, lane, a16 ? packed | (packed << 16) : std::bit_cast<uint32_t>(coordinate));
        wave_->debug_write_vgpr(
            3, lane, a16 ? util::f32_to_f16(float(lane)) : std::bit_cast<uint32_t>(coordinate));
        wave_->debug_write_vgpr(4, lane, std::bit_cast<uint32_t>(float(lane)));
      }
      wave_->debug_write_vgpr(12, 4, 0xdeadbeef);
      ASSERT_NO_FATAL_FAILURE(sample(57, cube ? 3 : 1, a16));
      for (uint32_t lane = 0; lane < 4; ++lane) {
        EXPECT_EQ(wave_->debug_read_vgpr(12, lane), cube || !(lane & 1) ? 2u : 1u) << lane;
        EXPECT_EQ(wave_->debug_read_vgpr(13, lane), cube ? lane : 0u) << lane;
      }
      EXPECT_EQ(wave_->debug_read_vgpr(12, 4), 0xdeadbeefu);
    }
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
  for (uint32_t type : {9u, 13u})
    for (bool a16 : {false, true})
      for (bool array : {false, true})
        for (const auto &[opcode, name] : {std::pair{27u, "IMAGE_SAMPLE"},
                                           {28u, "IMAGE_SAMPLE_D"},
                                           {57u, "IMAGE_SAMPLE_D_G16"},
                                           {29u, "IMAGE_SAMPLE_L"},
                                           {30u, "IMAGE_SAMPLE_B"},
                                           {31u, "IMAGE_SAMPLE_LZ"}}) {
          SCOPED_TRACE(testing::Message() << "type=" << type << ", a16=" << a16
                                          << ", array=" << array << ", opcode=" << name);
          const std::array<uint32_t, 8> descriptor{0x1000,
                                                   (46u << (gfx12 ? 17 : 20)) | (3u << 30) |
                                                       (3u << (gfx12 ? 12 : 16)),
                                                   1 | (7u << 14),
                                                   (type << 28) | 0xfac | (3u << (gfx12 ? 15 : 16)),
                                                   type == 13 ? (2u << 16) | 4 : 0,
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
          const uint32_t prefix = opcode == 28 ? 4 : opcode == 57 ? 2 : opcode == 30 ? 1 : 0;
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
            if (opcode == 57) {
              wave_->debug_write_vgpr(registers[0], lane, util::f32_to_f16(0.5f));
              wave_->debug_write_vgpr(registers[1], lane, uint32_t(util::f32_to_f16(0.5f)) << 16);
            }
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
          ASSERT_TRUE(instruction->is_memory_op());
          ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
          ASSERT_FALSE(wave_->instruction_execution_failed());
          amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
          pipeline.issue(instruction.release(), *wave_);
          for (uint32_t lane = 0; lane < 4; ++lane) {
            const uint32_t level = opcode == 31 ? 0 : opcode == 29 ? 1 : opcode == 30 ? 3 : 2;
            const uint32_t layer = type == 9 ? 0 : !array || lane == 0 ? 2 : lane == 3 ? 4 : 3;
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
        state.context_registers[0x1c] = 0x700780 | 6 | (comparison << 4);
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
          state.context_registers[0x200] = 0x700780 | 6 | (comparison << 4);
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

TEST_P(GraphicsExportTest, InterpolationPreservesHardwareNanPayloads) {
  wave_->set_exec(15);
  for (uint32_t ieee : {0u, 1u}) {
    wave_->set_mode_raw(ieee << 9);
    wave_->debug_write_vgpr(0, 0, 0xffc01234u);
    wave_->debug_write_vgpr(0, 1, 0x7f801abcu);
    for (uint32_t lane = 0; lane < 4; ++lane)
      wave_->debug_write_vgpr(1, lane, 0x3f800000u);
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
      run(rdna4::build_vinterp(0, {.vdst = 0, .src0 = 256, .src1 = 257, .src2 = 256}));
    else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
      run(rdna3_5::build_vinterp(0, {.vdst = 0, .src0 = 256, .src1 = 257, .src2 = 256}));
    else
      run(rdna3::build_vinterp(0, {.vdst = 0, .src0 = 256, .src1 = 257, .src2 = 256}));
    const uint32_t expected =
        ieee || GetParam() == ROCJITSU_CODE_ARCH_RDNA4 ? 0x7fc01abcu : 0x7f801abcu;
    for (uint32_t lane = 0; lane < 4; ++lane)
      EXPECT_EQ(wave_->debug_read_vgpr(0, lane), expected);
  }
}

TEST(GraphicsRasterMathTest, ParameterDifferencesPreserveReferenceNan) {
  const auto difference = [](uint32_t a, uint32_t b) {
    return std::bit_cast<uint32_t>(
        amdgpu::raster::attribute_difference(std::bit_cast<float>(a), std::bit_cast<float>(b)));
  };
  EXPECT_EQ(difference(0x7fc01234u, 0xff801abcu), 0xff801abcu);
  EXPECT_EQ(difference(0x3f800000u, 0x7f801abcu), 0x7f801abcu);
  EXPECT_EQ(difference(0xff801abcu, 0x3f800000u), 0xff801abcu);
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
  for (const auto &[key, expected] : {std::pair{0u, 0u},
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

TEST(GraphicsImageFilterTest, AnisotropicDescriptorControlsMatchPhysicalFilterCounts) {
  // FNV-1a digests of counts recovered from R32 impulse readbacks. The first
  // corpus covers every bias/PERF_MOD pair on GFX11; the independent threshold
  // corpus agrees byte-for-byte on physical GFX11 and GFX12.
  uint64_t digest = 14695981039346656037ull;
  for (uint32_t perf = 0; perf < 8; ++perf)
    for (uint32_t bias = 0; bias < 64; ++bias)
      for (uint32_t i = 0; i < 512; ++i) {
        const auto count = amdgpu::image_anisotropic_filter_count(
            amdgpu::image_footprint(1.0 + i / 32.0, 0, 0, 1, 1, 1), 16, 0, bias, perf);
        digest = (digest ^ count) * 1099511628211ull;
      }
  EXPECT_EQ(digest, 0x19dbaf1b0e5faf7cull);
  digest = 14695981039346656037ull;
  struct Control {
    uint32_t threshold, bias;
  };
  constexpr Control controls[] = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0},  {5, 0},
                                  {6, 0}, {7, 0}, {0, 4}, {2, 4}, {2, 32}, {2, 63}};
  for (const auto &[threshold, bias] : controls)
    for (uint32_t i = 0; i < 4096; ++i) {
      const auto count = amdgpu::image_anisotropic_filter_count(
          amdgpu::image_footprint(1.0 + i / 256.0, 0, 0, 1, 1, 1), 16, threshold, bias, 4);
      digest = (digest ^ count) * 1099511628211ull;
    }
  EXPECT_EQ(digest, 0x435635cf362ec9fcull);
}

TEST(GraphicsImageFilterTest, AnisotropicTwoAxisCountsMatchPhysicalBoundaries) {
  struct Witness {
    float major, minor;
    uint32_t max_anisotropy, threshold, bias, perf_mod, expected;
  };
  // Physical R32 impulse counts on both GFX11 and GFX12. Non-unit minor
  // axes expose the two precision reductions; small axes expose bias ordering.
  constexpr Witness witnesses[] = {
      {2.420099f, 1.03f, 16, 2, 4, 4, 2},    {5.9045086f, 1.33f, 16, 2, 4, 4, 4},
      {17.743763f, 2.07f, 16, 2, 4, 4, 8},   {22.065083f, 2.07f, 16, 2, 4, 4, 10},
      {26.418175f, 2.07f, 16, 2, 4, 4, 12},  {30.701893f, 2.07f, 16, 2, 4, 4, 14},
      {2.015625f, 0.484375f, 4, 0, 4, 4, 4}, {2.015625f, 0.5f, 4, 0, 4, 4, 2},
      {4.125f, 0.5f, 8, 0, 4, 4, 6},         {4.125f, 0.515625f, 8, 0, 4, 4, 4},
      {8.25f, 0.5f, 16, 0, 4, 4, 10},        {8.25f, 0.515625f, 16, 0, 4, 4, 8},
      {16.5f, 1, 16, 0, 32, 7, 10},
  };
  for (const auto &witness : witnesses)
    EXPECT_EQ(amdgpu::image_anisotropic_filter_count(
                  amdgpu::image_footprint(witness.major, 0, 0, witness.minor, 1, 1),
                  witness.max_anisotropy, witness.threshold, witness.bias, witness.perf_mod),
              witness.expected)
        << witness.major << "," << witness.minor << "," << witness.max_anisotropy;
}

TEST_P(GraphicsExportTest, AnisotropicDirectionMatchesPhysicalMatricesAndSignedTies) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const auto layout = amdgpu::image_mip_layout(gfx12, 0, 16, 64, 16, 1, 0);
  ASSERT_TRUE(layout);
  for (uint32_t y = 0; y < 16; ++y)
    for (uint32_t x = 0; x < 64; ++x) {
      const auto address = gfx12
                               ? amdgpu::gfx12_image_address(0x100000, x, y, layout->pitch, 16, 0)
                               : amdgpu::gfx11_image_address(0x100000, x, y, layout->pitch, 16, 0);
      ASSERT_TRUE(address);
      const float channels[] = {float(std::max(int(x) - 32, 0)), float(std::max(32 - int(x), 0)),
                                float(std::max(int(y) - 8, 0)), float(std::max(8 - int(y), 0))};
      for (uint32_t c = 0; c < 4; ++c)
        memory_.write32(*address + 4 * c, std::bit_cast<uint32_t>(channels[c]));
    }
  const std::array<uint32_t, 8> descriptor{0x1000,
                                           (63u << (gfx12 ? 17 : 20)) | (3u << 30),
                                           15 | (15u << 14),
                                           (9u << 28) | 0xfac,
                                           0,
                                           4u << 20,
                                           0,
                                           0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->debug_write_sgpr(4, 0x92 | (1u << 9) | (1u << 27));
  wave_->debug_write_sgpr(5, 0);
  wave_->debug_write_sgpr(6, (3u << 20) | (3u << 22));
  wave_->debug_write_sgpr(7, 0);
  struct Witness {
    std::array<uint32_t, 4> gradients;
    int32_t phase;
    std::array<uint32_t, 4> expected;
  };
  // Raw RGBA32F readbacks agree on physical GFX11 and GFX12. Independent
  // full-rank matrices exercise exponent alignment; signed rank-one inputs
  // distinguish positive and negative multiplication ties.
  constexpr Witness witnesses[] = {
      {{0x3ebb77fdu, 0xbf989c0du, 0x3f8c89d3u, 0xbf4c9a92u},
       -31,
       {0x3e1a0000u, 0x3e1e0000u, 0x3e4a0000u, 0x3e4e0000u}},
      {{0x3f522dc4u, 0x3fbd8ef4u, 0x3d8169d0u, 0x3eef87e8u},
       -15,
       {0x3dec0000u, 0x3df00000u, 0x3e620000u, 0x3e640000u}},
      {{0xbe1ce6acu, 0xbf01f41bu, 0x3fdeeb8au, 0xbdf3638du},
       0,
       {0x3e800000u, 0x3e800000u, 0x3c400000u, 0x3c400000u}},
      {{0x3f9519dau, 0xbfa705a3u, 0x3ebafd2bu, 0x3e82c808u},
       15,
       {0x3e2e0000u, 0x3e2c0000u, 0x3e3e0000u, 0x3e3c0000u}},
      {{0x3e403644u, 0x3ebd0003u, 0xbfd76762u, 0x3ef21fd3u},
       31,
       {0x3e7a0000u, 0x3e760000u, 0x3d880000u, 0x3d800000u}},
      {{0xbf2945d5u, 0xbfc61554u, 0x3eb195cdu, 0x3ebe1816u},
       -31,
       {0x3dd00000u, 0x3dd80000u, 0x3e660000u, 0x3e6a0000u}},
      {{0xbf436f54u, 0xbea8bed4u, 0x3fa85e58u, 0xbf898113u},
       -15,
       {0x3e560000u, 0x3e580000u, 0x3e0a0000u, 0x3e0c0000u}},
      {{0x3f95559au, 0xbe76914bu, 0xbf2d37e2u, 0x3fade987u},
       0,
       {0x3e320000u, 0x3e320000u, 0x3e380000u, 0x3e380000u}},
      {{0xbf9ec7cfu, 0x3f254c5fu, 0xbf95380bu, 0xbdfac4f5u},
       15,
       {0x3e7a0000u, 0x3e780000u, 0x3d800000u, 0x3d780000u}},
      {{0xbecd11b6u, 0xbf8f3667u, 0x3fa9d804u, 0x3f1f6a0du},
       31,
       {0x3e400000u, 0x3e3c0000u, 0x3e2c0000u, 0x3e280000u}},
      {{0x3f79806bu, 0x3f0e107fu, 0xbfb444cfu, 0x3f42b011u},
       -31,
       {0x3e760000u, 0x3e7a0000u, 0x3d680000u, 0x3d780000u}},
      {{0x3f9f622cu, 0xbf9a4822u, 0x3e80e2b9u, 0xbdb67449u},
       -15,
       {0x3e380000u, 0x3e3a0000u, 0x3e300000u, 0x3e300000u}},
      {{0x3f043c8cu, 0x3f8ee149u, 0xbfb354bau, 0xbed825efu},
       0,
       {0x3e500000u, 0x3e500000u, 0x3e140000u, 0x3e140000u}},
      {{0xbf93ae2eu, 0xbf9c6d8eu, 0xbde404d2u, 0xbf0e3d26u},
       15,
       {0x3e280000u, 0x3e260000u, 0x3e420000u, 0x3e420000u}},
      {{0xbe68e0cau, 0xbfc38c21u, 0x3f1c3262u, 0x3f34d3eau},
       31,
       {0x3d940000u, 0x3d8c0000u, 0x3e780000u, 0x3e740000u}},
      {{0x3f26eb77u, 0x3fa1be1du, 0xbf11fb53u, 0x3f9a8c91u},
       -31,
       {0x3c600000u, 0x3c800000u, 0x3e7e0000u, 0x3e810000u}},
      {{0x3fc00000u, 0x3d800000u, 0x00000000u, 0x00000000u},
       6,
       {0x3e800000u, 0x3e7e0000u, 0x3c400000u, 0x3c200000u}},
      {{0xbfc00000u, 0x3d800000u, 0x00000000u, 0x00000000u},
       6,
       {0x3e800000u, 0x3e7e0000u, 0x3c400000u, 0x3c200000u}},
      {{0x3fc00000u, 0xbd800000u, 0x00000000u, 0x00000000u},
       6,
       {0x3e800000u, 0x3e7e0000u, 0x3c400000u, 0x3c200000u}},
      {{0xbfc00000u, 0xbd800000u, 0x00000000u, 0x00000000u},
       6,
       {0x3e800000u, 0x3e7e0000u, 0x3c400000u, 0x3c200000u}},
  };
  wave_->set_exec((1u << std::size(witnesses)) - 1);
  for (uint32_t lane = 0; lane < std::size(witnesses); ++lane) {
    const auto &witness = witnesses[lane];
    const float shift = witness.phase / 8192.0f;
    const float values[] = {std::bit_cast<float>(witness.gradients[0]) / 64,
                            std::bit_cast<float>(witness.gradients[1]) / 16,
                            std::bit_cast<float>(witness.gradients[2]) / 64,
                            std::bit_cast<float>(witness.gradients[3]) / 16,
                            (32.5f + shift) / 64,
                            (8.5f + shift) / 16};
    for (uint32_t i = 0; i < std::size(values); ++i)
      wave_->debug_write_vgpr(i, lane, std::bit_cast<uint32_t>(values[i]));
  }
  wave_->debug_write_vgpr(12, 31, 0xdeadbeef);
  ASSERT_NO_FATAL_FAILURE(sample(28, 1));
  for (uint32_t lane = 0; lane < std::size(witnesses); ++lane)
    for (uint32_t c = 0; c < 4; ++c)
      EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane), witnesses[lane].expected[c])
          << lane << ',' << c;
  EXPECT_EQ(wave_->debug_read_vgpr(12, 31), 0xdeadbeefu);
}

TEST_P(GraphicsExportTest, AnisotropicCoordinatesMatchPhysicalExtentsAndAlignment) {
  struct Witness {
    uint32_t width, height;
    bool repeat;
    std::array<uint32_t, 6> values;
    std::array<uint32_t, 4> expected;
  };
  // Raw gradient, coordinate and RGBA32F bits captured on GFX11 and GFX12.
  // The controls cover exact-power norm reciprocals, 19x13 and 29x31 image
  // reciprocals, and positive/negative coordinates near zero and repeat edges.
  constexpr Witness witnesses[] = {
      {64,
       16,
       false,
       {0x3cd59b3du, 0xbca87c86u, 0x3c196de4u, 0x3d34550bu, 0x3f01fc80u, 0x3f07f200u},
       {0x3e7e0000u, 0x3e810000u, 0x3c400000u, 0x3c600000u}},
      {64,
       16,
       false,
       {0x3cd59b3du, 0xbca87c86u, 0x3c196de4u, 0x3d34550bu, 0x3f0200a0u, 0x3f080280u},
       {0x3e800000u, 0x3e800000u, 0x3c600000u, 0x3c400000u}},
      {19,
       13,
       false,
       {0xba537364u, 0x3cbe5c60u, 0xbdbb2f03u, 0x3cb61294u, 0x3effe86cu, 0x3effdd8au},
       {0x3e7a0000u, 0x3e7e0000u, 0x3d280000u, 0x3d380000u}},
      {19,
       13,
       false,
       {0xbd591efdu, 0xbcc709eeu, 0x3d6b9cfeu, 0xbdb019e3u, 0x3effe6bdu, 0x3effdb14u},
       {0x3e560000u, 0x3e5a0000u, 0x3e080000u, 0x3e0a0000u}},
      {29,
       31,
       false,
       {0x3d76f05fu, 0x3c463a5bu, 0xbc8d43d3u, 0x3c3ac217u, 0x3effee58u, 0x3effef7bu},
       {0x3e7a0000u, 0x3e7e0000u, 0x3d100000u, 0x3d200000u}},
      {29,
       31,
       false,
       {0x3d6bb420u, 0xbc2debe5u, 0x3ca94d1fu, 0x3cba263du, 0x3f00088du, 0x3f000800u},
       {0x3e810000u, 0x3e7e0000u, 0x3c600000u, 0x3c200000u}},
      {19,
       13,
       true,
       {0xbccb6817u, 0xbe0978ceu, 0x3a3d3244u, 0xbd096a13u, 0xbeffe5e5u, 0xbeffd9d8u},
       {0x3d800000u, 0x3d780000u, 0x3e7a0000u, 0x3e760000u}},
      {19,
       13,
       true,
       {0xbcc1e249u, 0xbde508a5u, 0x3d09b8a9u, 0xbd61e869u, 0xbefffe51u, 0xbefffd89u},
       {0x3cc00000u, 0x3cc00000u, 0x3e800000u, 0x3e800000u}},
      {19,
       13,
       true,
       {0x3dbc7413u, 0x3cec5947u, 0xbcd79d6au, 0x3cdeac58u, 0x3fbff943u, 0x3fbff628u},
       {0x3e7a0000u, 0x3e7e0000u, 0x3d100000u, 0x3d200000u}},
      {19,
       13,
       true,
       {0x3db3e10bu, 0xbccf5e2fu, 0x3d01341eu, 0x3d5df284u, 0x3fc00687u, 0x3fc0098au},
       {0x3e810000u, 0x3e7e0000u, 0x3c600000u, 0x3c200000u}},
      {19,
       13,
       true,
       {0x3d71969bu, 0x3d8f53d7u, 0xbd9169ccu, 0x3cb01940u, 0x30820000u, 0x30820000u},
       {0x40900000u, 0x40900000u, 0x40400000u, 0x40400000u}},
      {19,
       13,
       true,
       {0x3d054cc6u, 0x3b07f324u, 0xbd5f5b14u, 0x3dd9ca9cu, 0x30a00000u, 0x30a00000u},
       {0x40900000u, 0x40900000u, 0x40400000u, 0x40400000u}},
      {19,
       13,
       true,
       {0x3d71969bu, 0x3d8f53d7u, 0xbd9169ccu, 0x3cb01940u, 0xb0820000u, 0xb0820000u},
       {0x40900000u, 0x40900000u, 0x40400000u, 0x40400000u}},
      {19,
       13,
       true,
       {0x3d054cc6u, 0x3b07f324u, 0xbd5f5b14u, 0x3dd9ca9cu, 0xb0a00000u, 0xb0a00000u},
       {0x40900000u, 0x40900000u, 0x40400000u, 0x40400000u}},
  };
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  wave_->set_exec(1);
  for (const auto &witness : witnesses) {
    SCOPED_TRACE(::testing::Message()
                 << witness.width << ',' << witness.height << ',' << std::hex << witness.values[4]);
    const auto layout = amdgpu::image_mip_layout(gfx12, 0, 16, witness.width, witness.height, 1, 0);
    ASSERT_TRUE(layout);
    for (uint32_t y = 0; y < witness.height; ++y)
      for (uint32_t x = 0; x < witness.width; ++x) {
        const auto address =
            gfx12 ? amdgpu::gfx12_image_address(0x100000, x, y, layout->pitch, 16, 0)
                  : amdgpu::gfx11_image_address(0x100000, x, y, layout->pitch, 16, 0);
        ASSERT_TRUE(address);
        const int dx = int(x) - int(witness.width / 2);
        const int dy = int(y) - int(witness.height / 2);
        const float channels[] = {float(std::max(dx, 0)), float(std::max(-dx, 0)),
                                  float(std::max(dy, 0)), float(std::max(-dy, 0))};
        for (uint32_t c = 0; c < 4; ++c)
          memory_.write32(*address + 4 * c, std::bit_cast<uint32_t>(channels[c]));
      }
    const uint32_t width = witness.width - 1;
    const std::array<uint32_t, 8> descriptor{0x1000,
                                             (63u << (gfx12 ? 17 : 20)) | ((width & 3) << 30),
                                             (width >> 2) | ((witness.height - 1) << 14),
                                             (9u << 28) | 0xfac,
                                             0,
                                             4u << 20,
                                             0,
                                             0};
    for (uint32_t i = 0; i < descriptor.size(); ++i)
      wave_->debug_write_sgpr(8 + i, descriptor[i]);
    wave_->debug_write_sgpr(4, (witness.repeat ? 0 : 0x92) | (1u << 9) | (1u << 27));
    wave_->debug_write_sgpr(5, 0);
    wave_->debug_write_sgpr(6, (3u << 20) | (3u << 22));
    wave_->debug_write_sgpr(7, 0);
    for (uint32_t i = 0; i < witness.values.size(); ++i)
      wave_->debug_write_vgpr(i, 0, witness.values[i]);
    ASSERT_NO_FATAL_FAILURE(sample(28, 1));
    for (uint32_t c = 0; c < 4; ++c)
      EXPECT_EQ(wave_->debug_read_vgpr(12 + c, 0), witness.expected[c]) << c;
  }
}

TEST(GraphicsImageFilterTest, AnisotropicNormalizationMatchesAllPhysicalMantissas) {
  // Direction components inferred independently from cardinal phase captures
  // below and above two texels. Compare the complete finite mantissa domain.
  uint64_t digest = 14695981039346656037ull;
  for (uint32_t i = 0; i < 1024; ++i) {
    const double gradient = (1.0 + i / 1024.0) / 64;
    const auto direction = amdgpu::image_footprint(gradient, 0, 0, 0, 64, 16).direction();
    EXPECT_EQ(direction[1], 0);
    digest = (digest ^ std::bit_cast<uint64_t>(direction[0])) * 1099511628211ull;
  }
  EXPECT_EQ(digest, 0x282fbe27df287325ull);
}

TEST_P(GraphicsExportTest, AnisotropicSamplingUsesPhysicalNonuniformWeights) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const auto layout = amdgpu::image_mip_layout(gfx12, 0, 4, 64, 16, 1, 0);
  ASSERT_TRUE(layout);
  for (uint32_t y = 0; y < 16; ++y)
    for (uint32_t x = 0; x < 64; ++x) {
      const auto address = gfx12 ? amdgpu::gfx12_image_address(0x100000, x, y, layout->pitch, 4, 0)
                                 : amdgpu::gfx11_image_address(0x100000, x, y, layout->pitch, 4, 0);
      ASSERT_TRUE(address);
      memory_.write32(*address, x == 32 ? 0x3f800000 : 0);
    }
  const std::array<uint32_t, 8> descriptor{0x1000,
                                           (22u << (gfx12 ? 17 : 20)) | (3u << 30),
                                           15 | (15u << 14),
                                           (9u << 28) | 0x204,
                                           0,
                                           4u << 20,
                                           0,
                                           0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->debug_write_sgpr(4, 0x92 | (4u << 9) | (2u << 16) | (4u << 21) | (1u << 27));
  wave_->debug_write_sgpr(5, 0);
  wave_->debug_write_sgpr(6, (2u << 20) | (2u << 22));
  wave_->debug_write_sgpr(7, 0);
  struct Witness {
    uint32_t count;
    std::array<uint32_t, 14> expected;
  };
  // Raw R32 impulse outputs for nearest anisotropic filtering on both cards.
  constexpr Witness witnesses[] = {
      {6, {0x3e2a0000, 0x3e2b0000, 0x3e2b0000, 0x3e2b0000, 0x3e2b0000, 0x3e2a0000}},
      {10,
       {0x3dcc0000, 0x3dcd0000, 0x3dcd0000, 0x3dcd0000, 0x3dcd0000, 0x3dcd0000, 0x3dcd0000,
        0x3dcd0000, 0x3dcd0000, 0x3dcc0000}},
      {12,
       {0x3daa0000, 0x3daa0000, 0x3dab0000, 0x3dab0000, 0x3dab0000, 0x3dab0000, 0x3dab0000,
        0x3dab0000, 0x3dab0000, 0x3dab0000, 0x3daa0000, 0x3daa0000}},
      {14,
       {0x3d920000, 0x3d920000, 0x3d920000, 0x3d920000, 0x3d920000, 0x3d930000, 0x3d930000,
        0x3d930000, 0x3d930000, 0x3d920000, 0x3d920000, 0x3d920000, 0x3d920000, 0x3d920000}},
  };
  for (const auto &witness : witnesses) {
    wave_->set_exec((1u << witness.count) - 1);
    for (uint32_t lane = 0; lane < witness.count; ++lane) {
      const float offset = float(lane) + 0.125f - witness.count * 0.5f;
      const float values[] = {witness.count / 64.0f, 0,        0, 1.0f / 16,
                              (32.5f + offset) / 64, 8.5f / 16};
      for (uint32_t i = 0; i < std::size(values); ++i)
        wave_->debug_write_vgpr(i, lane, std::bit_cast<uint32_t>(values[i]));
    }
    wave_->debug_write_vgpr(12, 31, 0xdeadbeef);
    ASSERT_NO_FATAL_FAILURE(sample(28, 1)); // IMAGE_SAMPLE_D, 2D.
    for (uint32_t lane = 0; lane < witness.count; ++lane) {
      EXPECT_EQ(wave_->debug_read_vgpr(12, lane), witness.expected[lane])
          << witness.count << "," << lane;
      EXPECT_EQ(wave_->debug_read_vgpr(13, lane), 0u);
      EXPECT_EQ(wave_->debug_read_vgpr(14, lane), 0u);
      EXPECT_EQ(wave_->debug_read_vgpr(15, lane), 0x3f800000u);
    }
    EXPECT_EQ(wave_->debug_read_vgpr(12, 31), 0xdeadbeefu);
  }
  // A footprint above the sampler's maximum still applies bias before the
  // count cap. Both cards select ten filters here; pre-capping selects eight.
  wave_->debug_write_sgpr(13, 7u << 20);
  wave_->debug_write_sgpr(4, 0x92 | (4u << 9) | (32u << 21) | (1u << 27));
  wave_->set_exec(1);
  const float values[] = {16.5f / 64, 0, 0, 1.0f / 16, 33.75f / 64, 8.5f / 16};
  for (uint32_t i = 0; i < std::size(values); ++i)
    wave_->debug_write_vgpr(i, 0, std::bit_cast<uint32_t>(values[i]));
  ASSERT_NO_FATAL_FAILURE(sample(28, 1));
  EXPECT_EQ(wave_->debug_read_vgpr(12, 0), 0x3dcd0000u);
}

TEST_P(GraphicsExportTest, AnisotropicArraySamplingUsesIndependentLaneFootprints) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const auto layout = amdgpu::image_mip_layout(gfx12, 0, 4, 64, 16, 1, 0);
  ASSERT_TRUE(layout);
  for (uint32_t y = 0; y < 16; ++y)
    for (uint32_t x = 0; x < 64; ++x) {
      uint32_t value = x * 1664525u + y * 1013904223u;
      value ^= value >> 16;
      value = value * 2246822519u | 0xff000000u;
      const auto address = gfx12 ? amdgpu::gfx12_image_address(0x100000 + 2 * layout->slice_size, x,
                                                               y, layout->pitch, 4, 0)
                                 : amdgpu::gfx11_image_address(0x100000 + 2 * layout->slice_size, x,
                                                               y, layout->pitch, 4, 0);
      ASSERT_TRUE(address);
      memory_.write32(*address, value);
    }
  const std::array<uint32_t, 8> descriptor{0x1000,
                                           (42u << (gfx12 ? 17 : 20)) | (3u << 30),
                                           15 | (15u << 14),
                                           (13u << 28) | 0xfac,
                                           (2u << 16) | 2,
                                           4u << 20,
                                           0,
                                           0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->debug_write_sgpr(4, 0x92 | (4u << 9) | (2u << 16) | (4u << 21));
  wave_->debug_write_sgpr(5, 0);
  wave_->debug_write_sgpr(6, (3u << 20) | (3u << 22));
  wave_->debug_write_sgpr(7, 0);
  struct Footprint {
    float major, minor, u, v;
  };
  struct Witness {
    Footprint footprint;
    std::array<uint32_t, 4> expected;
  };
  // Exact raw outputs from 64x16 RGBA8 image samples on both physical targets.
  const Witness witnesses[] = {
      {{1.0f, 0.5f, 0.537109375f, 0.37f}, {0x3e7c3030u, 0x3f197dfeu, 0x3e8f1515u, 0x3f800000u}},
      {{2.0f, 0.5f, 0.537109375f, 0.37f}, {0x3e999091u, 0x3f019fe0u, 0x3ea60000u, 0x3f800000u}},
      {{4.0f, 0.5f, 0.537109375f, 0.37f}, {0x3eb72f2fu, 0x3ecd8505u, 0x3ee05adbu, 0x3f800000u}},
      {{8.0f, 0.5f, 0.537109375f, 0.37f}, {0x3eca9e5eu, 0x3ec755b6u, 0x3ec3de9fu, 0x3f800000u}},
      {{16.0f, 0.5f, 0.537109375f, 0.37f}, {0x3ee2acadu, 0x3ef34c2cu, 0x3ecb8707u, 0x3f800000u}},
      {{1.0f, 1.0f, 0.537109375f, 0.37f}, {0x3e7c3030u, 0x3f197dfeu, 0x3e8f1515u, 0x3f800000u}},
      {{2.0f, 1.0f, 0.537109375f, 0.37f}, {0x3e999091u, 0x3f019fe0u, 0x3ea60000u, 0x3f800000u}},
      {{4.0f, 1.0f, 0.537109375f, 0.37f}, {0x3eb72f2fu, 0x3ecd8505u, 0x3ee05adbu, 0x3f800000u}},
      {{8.0f, 1.0f, 0.537109375f, 0.37f}, {0x3eca9e5eu, 0x3ec755b6u, 0x3ec3de9fu, 0x3f800000u}},
      {{6.0f, 1.0f, 0.537109375f, 0.37f}, {0x3ec2abdcu, 0x3eb81151u, 0x3ed71adbu, 0x3f800000u}},
      {{16.0f, 1.0f, 0.537109375f, 0.37f}, {0x3ee2acadu, 0x3ef34c2cu, 0x3ecb8707u, 0x3f800000u}},
      {{1.0f, 2.0f, 0.537109375f, 0.37f}, {0x3e80b8b9u, 0x3f1c46c7u, 0x3e9e9515u, 0x3f800000u}},
      {{2.0f, 2.0f, 0.537109375f, 0.37f}, {0x3e7c3030u, 0x3f197dfeu, 0x3e8f1515u, 0x3f800000u}},
      {{4.0f, 2.0f, 0.537109375f, 0.37f}, {0x3ec3b9bau, 0x3ec38081u, 0x3eda5f5fu, 0x3f800000u}},
      {{8.0f, 2.0f, 0.537109375f, 0.37f}, {0x3ecbf373u, 0x3ead7afbu, 0x3ea282c3u, 0x3f800000u}},
      {{16.0f, 2.0f, 0.537109375f, 0.37f}, {0x3ee7f373u, 0x3eed5959u, 0x3eb10525u, 0x3f800000u}},
      // Additional phase sweeps isolate anisotropic UNORM accumulation and
      // normalization rounding for power-of-two and uneven filter counts.
      {{2.0f, 1.0f, 0.00121093751f, 0.0319140628f},
       {0x3ca9f3f4u, 0x3ca44040u, 0x3d1e1495u, 0x3f800000u}},
      {{4.0f, 1.0f, 0.00121093751f, 0.0319140628f},
       {0x3dc22c6cu, 0x3d9a6cedu, 0x3e45e3a4u, 0x3f800000u}},
      {{6.0f, 1.0f, 0.415273428f, 0.0319140628f},
       {0x3e9744b5u, 0x3eff4464u, 0x3ed18919u, 0x3f800000u}},
      {{6.0f, 1.0f, 0.458242178f, 0.0397265628f},
       {0x3e9894e5u, 0x3eb67737u, 0x3ed33e2eu, 0x3f800000u}},
      {{8.0f, 1.0f, 0.0168359373f, 0.0319140628f},
       {0x3e95eddeu, 0x3e98fd7du, 0x3e9a1e9fu, 0x3f800000u}},
      {{14.0f, 1.0f, 0.0441796891f, 0.0319140628f},
       {0x3eb64aabu, 0x3ec83d3du, 0x3eb293c4u, 0x3f800000u}},
      {{14.0f, 1.0f, 0.219960943f, 0.0319140628f},
       {0x3f12ca22u, 0x3f166d55u, 0x3f0ebe0eu, 0x3f800000u}},
      {{16.0f, 1.0f, 0.376210928f, 0.0319140628f},
       {0x3ed37e5eu, 0x3f04e8c9u, 0x3f059921u, 0x3f800000u}},
  };
  wave_->set_exec((1u << std::size(witnesses)) - 1);
  for (uint32_t lane = 0; lane < std::size(witnesses); ++lane) {
    const auto &footprint = witnesses[lane].footprint;
    const float values[] = {footprint.major / 64, 0,           0, footprint.minor / 16,
                            footprint.u,          footprint.v, 0};
    for (uint32_t i = 0; i < std::size(values); ++i)
      wave_->debug_write_vgpr(i, lane, std::bit_cast<uint32_t>(values[i]));
  }
  for (uint32_t c = 0; c < 4; ++c)
    wave_->debug_write_vgpr(12 + c, 31, 0xdeadbeef);
  ASSERT_NO_FATAL_FAILURE(sample(28, 5)); // IMAGE_SAMPLE_D, 2D array.
  for (uint32_t lane = 0; lane < std::size(witnesses); ++lane)
    for (uint32_t c = 0; c < 4; ++c)
      EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane), witnesses[lane].expected[c])
          << lane << "," << c;
  for (uint32_t c = 0; c < 4; ++c)
    EXPECT_EQ(wave_->debug_read_vgpr(12 + c, 31), 0xdeadbeefu);
}

TEST_P(GraphicsExportTest, CubeSamplingRemapsEdgesCornersAndArrayViewFaces) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const auto layout = amdgpu::image_mip_layout(gfx12, 0, 8, 4, 4, 1, 0);
  ASSERT_TRUE(layout);
  for (uint32_t face = 0; face < 6; ++face)
    for (uint32_t y = 0; y < 4; ++y)
      for (uint32_t x = 0; x < 4; ++x) {
        const auto address =
            gfx12 ? amdgpu::gfx12_image_address(0x100000 + (6 + face) * layout->slice_size, x, y,
                                                layout->pitch, 8, 0)
                  : amdgpu::gfx11_image_address(0x100000 + (6 + face) * layout->slice_size, x, y,
                                                layout->pitch, 8, 0);
        ASSERT_TRUE(address);
        for (uint32_t c = 0; c < 4; ++c)
          memory_.write16(*address + 2 * c, 0x2800 + face * 0x500 + x * 0x21 + y * 0x31 + c * 0xb);
      }
  const std::array<uint32_t, 8> descriptor{0x1000,
                                           (57u << (gfx12 ? 17 : 20)) | (3u << 30),
                                           3u << 14,
                                           (11u << 28) | 0xfac,
                                           (6u << 16) | 11,
                                           0,
                                           0,
                                           0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->debug_write_sgpr(4, 0x92);
  wave_->debug_write_sgpr(5, 0);
  wave_->debug_write_sgpr(6, (1u << 20) | (1u << 22));
  wave_->debug_write_sgpr(7, 0);
  struct Witness {
    uint32_t query;
    std::array<uint32_t, 4> expected;
  };
  // Face-edge and corner outputs captured from a 4x4 RGBA16F cube on both cards.
  const Witness witnesses[] = {
      {0, {0x3ee3e5e0u, 0x3ee5f93cu, 0x3ee81d19u, 0x3eea30f8u}},
      {15, {0x3e955181u, 0x3e9694d6u, 0x3e97f04au, 0x3e99345du}},
      {16, {0x3f0c7f20u, 0x3f0d815au, 0x3f0e9c34u, 0x3f0fceeeu}},
      {31, {0x3e14e50cu, 0x3e15804eu, 0x3e169c8bu, 0x3e1835ccu}},
      {32, {0x3fb5d6e3u, 0x3fb741a3u, 0x3fb8ac63u, 0x3fba2763u}},
      {47, {0x3fb4f27du, 0x3fb65d3du, 0x3fb7c7fdu, 0x3fb942fdu}},
      {48, {0x3f4a4587u, 0x3f4bf578u, 0x3f4dad47u, 0x3f4f5d38u}},
      {63, {0x3f4b7619u, 0x3f4d29d8u, 0x3f4eddd9u, 0x3f509198u}},
      {64, {0x3f1e3f9cu, 0x3f1fcc5du, 0x3f21611du, 0x3f22f5ddu}},
      {79, {0x3f1f5c34u, 0x3f20e543u, 0x3f227a03u, 0x3f240ec3u}},
      {80, {0x3fb20a03u, 0x3fb37f43u, 0x3fb50443u, 0x3fb67983u}},
      {95, {0x3fb2f69du, 0x3fb46bddu, 0x3fb5f0ddu, 0x3fb7661du}},
      {255, {0x3f14048eu, 0x3f152edcu, 0x3f1658e9u, 0x3f178b37u}},
      {256, {0x3ee03546u, 0x3ee26d36u, 0x3ee4b5a4u, 0x3ee6ed95u}},
      {32767, {0x3f127baeu, 0x3f13ee6eu, 0x3f15616eu, 0x3f16cc2eu}},
      {32768, {0x3e0f1870u, 0x3e1021d0u, 0x3e112b30u, 0x3e123490u}},
      {65280, {0x3f13270cu, 0x3f144fd4u, 0x3f1578ddu, 0x3f16a9e5u}},
      {65295, {0x3ed5f1abu, 0x3ed78295u, 0x3ed9143cu, 0x3edabd45u}},
      {65311, {0x3e94fa4du, 0x3e95c52cu, 0x3e96d088u, 0x3e97dbe4u}},
      {65327, {0x3f19c82cu, 0x3f1b4f7du, 0x3f1cdabdu, 0x3f1e65fdu}},
      {65343, {0x3fd8c65au, 0x3fda499bu, 0x3fdbd4dbu, 0x3fdd601bu}},
      {65359, {0x3f4fa087u, 0x3f516078u, 0x3f5318c7u, 0x3f54d8b8u}},
      {65520, {0x3facee23u, 0x3fae39fau, 0x3fafa932u, 0x3fb10c5au}},
      {65535, {0x3f8d244eu, 0x3f8e3680u, 0x3f8f5febu, 0x3f908115u}},
  };
  for (const auto &[opcode, name] :
       {std::pair{31u, "IMAGE_SAMPLE_LZ"}, {57u, "IMAGE_SAMPLE_D_G16"}})
    for (bool a16 : {false, true}) {
      SCOPED_TRACE(testing::Message() << "opcode=" << name << ", a16=" << a16);
      const uint32_t prefix = opcode == 57 ? 2 : 0;
      wave_->set_exec((1u << std::size(witnesses)) - 1);
      for (uint32_t lane = 0; lane < std::size(witnesses); ++lane) {
        const uint32_t q = witnesses[lane].query;
        const float values[] = {1 + ((q & 255) + 0.5f) / 256, 1 + ((q >> 8) + 0.5f) / 256,
                                float((q >> 4) % 6)};
        for (uint32_t i = 0; i < prefix; ++i)
          wave_->debug_write_vgpr(i, lane, 0);
        if (a16) {
          wave_->debug_write_vgpr(prefix, lane,
                                  util::f32_to_f16(values[0]) |
                                      (uint32_t(util::f32_to_f16(values[1])) << 16));
          wave_->debug_write_vgpr(prefix + 1, lane, util::f32_to_f16(values[2]));
        } else {
          for (uint32_t i = 0; i < 3; ++i)
            wave_->debug_write_vgpr(prefix + i, lane, std::bit_cast<uint32_t>(values[i]));
        }
      }
      wave_->debug_write_vgpr(12, 31, 0xdeadbeef);
      ASSERT_NO_FATAL_FAILURE(sample(opcode, 3, a16));
      for (uint32_t lane = 0; lane < std::size(witnesses); ++lane)
        for (uint32_t c = 0; c < 4; ++c)
          EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane), witnesses[lane].expected[c])
              << lane << "," << c;
      EXPECT_EQ(wave_->debug_read_vgpr(12, 31), 0xdeadbeefu);
    }
}

TEST_P(GraphicsExportTest, ImplicitCubeSamplingUnfoldsAdjacentFacesAndBoundsOppositeFaces) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  for (uint32_t level = 0; level < 3; ++level) {
    const auto mip = amdgpu::image_mip_layout(gfx12, 0, 8, 4, 4, 3, level);
    ASSERT_TRUE(mip);
    for (uint32_t face = 0; face < 6; ++face)
      for (uint32_t y = 0; y < mip->height; ++y)
        for (uint32_t x = 0; x < mip->width; ++x) {
          const uint64_t base = 0x100000 + mip->offset + face * mip->slice_size;
          const auto address = gfx12 ? amdgpu::gfx12_image_address(base, x, y, mip->pitch, 8, 0)
                                     : amdgpu::gfx11_image_address(base, x, y, mip->pitch, 8, 0);
          ASSERT_TRUE(address);
          for (uint32_t c = 0; c < 4; ++c)
            memory_.write16(*address + 2 * c, 0x3800 + level * 0x400 + face * 0x20);
        }
  }
  const std::array<uint32_t, 8> descriptor{
      0x1000,   (57u << (gfx12 ? 17 : 20)) | (3u << 30) | (2u << (gfx12 ? 12 : 16)),
      3u << 14, (11u << 28) | 0xfac | (2u << (gfx12 ? 15 : 16)),
      5,        0,
      0,        0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->debug_write_sgpr(4, 0x92);
  wave_->debug_write_sgpr(5, 512u << (gfx12 ? 13 : 12));
  wave_->debug_write_sgpr(6, 1u << 26);
  wave_->debug_write_sgpr(7, 0);
  // Physical RDNA3/4 captures distinguish each mip (0.5, 1, 2) and each face.
  // Include perpendicular and opposite faces, identical directions, and an edge.
  for (uint32_t opcode : {27u, 31u}) {
    struct Case {
      const char *name;
      float face, u, expected;
    };
    const Case cases[] = {{"perpendicular Y", 2.0f, 1.5f, 2.0f},
                          {"opposite X", 1.0f, 1.5f, 2.0f},
                          {"perpendicular Z", 4.0f, 1.5f, 2.0f},
                          {"identical directions", 0.0f, 1.5f, 0.5f},
                          {"adjacent Y toward shared edge", 2.0f, 1.75f, 2.0f},
                          {"adjacent Y away from shared edge", 2.0f, 1.25f, 2.0f},
                          {"shared edge", 2.0f, 2.0f, 1.0f}};
    for (const auto &[name, face, u, expected] : cases) {
      SCOPED_TRACE(opcode == 27 ? "IMAGE_SAMPLE" : "IMAGE_SAMPLE_LZ");
      SCOPED_TRACE(name);
      wave_->set_exec(15);
      for (uint32_t lane = 0; lane < 4; ++lane) {
        wave_->debug_write_vgpr(0, lane, std::bit_cast<uint32_t>(lane & 1 ? u : 1.5f));
        wave_->debug_write_vgpr(1, lane, std::bit_cast<uint32_t>(1.5f));
        wave_->debug_write_vgpr(2, lane, std::bit_cast<uint32_t>(lane & 1 ? face : 0.0f));
      }
      ASSERT_NO_FATAL_FAILURE(sample(opcode, 3));
      // A 0x20 FP16 significand step adds 1/32 of these power-of-two mip colors.
      for (uint32_t lane = 0; lane < 4; ++lane)
        for (uint32_t c = 0; c < 4; ++c)
          EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane),
                    std::bit_cast<uint32_t>((opcode == 31 ? 0.5f : expected) *
                                            (lane & 1 ? 1.0f + face / 32 : 1.0f)));
    }
  }
}

TEST_P(GraphicsExportTest, CubeSamplingMatchesPhysicalFootprints) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const uint16_t colors[] = {0, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600};
  const auto setup_cube = [&](uint32_t size) {
    // Keep the fixtures at separate addresses so previous texture-cache entries
    // cannot observe the direct backing-memory writes for the next image.
    const uint64_t image_base = size == 64 ? 0x100000 : 0x200000;
    for (uint32_t level = 0; level < std::size(colors); ++level) {
      const auto mip = amdgpu::image_mip_layout(gfx12, 0, 8, size, size, 7, level);
      ASSERT_TRUE(mip);
      for (uint32_t face = 0; face < 6; ++face)
        for (uint32_t y = 0; y < mip->height; ++y)
          for (uint32_t x = 0; x < mip->width; ++x) {
            const uint64_t base = image_base + mip->offset + face * mip->slice_size;
            const auto address = gfx12 ? amdgpu::gfx12_image_address(base, x, y, mip->pitch, 8, 0)
                                       : amdgpu::gfx11_image_address(base, x, y, mip->pitch, 8, 0);
            ASSERT_TRUE(address);
            for (uint32_t c = 0; c < 4; ++c)
              memory_.write16(*address + 2 * c, colors[level]);
          }
    }
    const std::array<uint32_t, 8> descriptor{
        static_cast<uint32_t>(image_base >> 8),
        (57u << (gfx12 ? 17 : 20)) | (((size - 1) & 3u) << 30) | (6u << (gfx12 ? 12 : 16)),
        ((size - 1) >> 2) | ((size - 1) << 14),
        (11u << 28) | 0xfac | (6u << (gfx12 ? 15 : 16)),
        5,
        4u << 20,
        0,
        0};
    for (uint32_t i = 0; i < descriptor.size(); ++i)
      wave_->debug_write_sgpr(8 + i, descriptor[i]);
    wave_->debug_write_sgpr(4, 0x92 | (4u << 9) | (2u << 16) | (4u << 21));
    wave_->debug_write_sgpr(5, (1536u << (gfx12 ? 13 : 12)) | (gfx12 ? 0 : 10u << 24));
    wave_->debug_write_sgpr(6, (3u << 20) | (3u << 22) | (2u << 26) | (gfx12 ? 2u << 30 : 0));
    wave_->debug_write_sgpr(7, gfx12 ? 2 : 0);
  };
  setup_cube(64);
  struct Witness {
    float dxu, dxv, dyu, dyv;
    uint32_t expected;
  };
  // Both physical cards return these mip colors with 16x anisotropy enabled.
  // The gradients describe cube directions (1, -v, -u); projected coordinates
  // scale them by one half. Cover narrow, square, degenerate and rotated quads.
  const Witness witnesses[] = {
      {0.5f, 0, 0, 0.03125f, 0x40800000u},
      {0.25f, 0, 0, 0.0625f, 0x40400000u},
      {0.0625f, 0, 0, 0.5f, 0x40800000u},
      {0.0625f, 0, 0, 0.0625f, 0x3f800000u},
      {0.03125f, 0, 0, 0.03125f, 0x00000000u},
      {0.5f, 0, 0, 0, 0x40800000u},
      {0.5f, 0.125f, 0, 0.03125f, 0x40800000u},
      {0.25f, 0.125f, 0.125f, 0.25f, 0x406b0000u},
      // The same-face and explicit-gradient controls retain magnitude rounding.
      {-0x1.3d50000000000p-8f, -0x1.2bb5a00000000p-3f, -0x1.531b800000000p-5f,
       -0x1.0aa0000000000p-3f, 0x40358000u},
      {0x1.09cb800000000p-5f, -0x1.93e0000000000p-5f, 0x1.4d64800000000p-5f, -0x1.4ced000000000p-5f,
       0x3fa50000u},
  };
  for (uint32_t opcode : {27u, 28u})
    for (const auto &witness : witnesses) {
      SCOPED_TRACE(testing::Message() << opcode << ", " << witness.dxu << ", " << witness.dyv);
      wave_->set_exec(15);
      wave_->debug_write_vgpr(12, 31, 0xdeadbeef);
      for (uint32_t lane = 0; lane < 4; ++lane) {
        const uint32_t offset = opcode == 28 ? 4 : 0;
        const float gradients[] = {witness.dxu, witness.dxv, witness.dyu, witness.dyv};
        for (uint32_t i = 0; i < offset; ++i)
          wave_->debug_write_vgpr(i, lane, std::bit_cast<uint32_t>(gradients[i] * 0.5f));
        const float u = 0.125f + (lane & 1) * witness.dxu + (lane >> 1) * witness.dyu;
        const float v = 0.25f + (lane & 1) * witness.dxv + (lane >> 1) * witness.dyv;
        wave_->debug_write_vgpr(offset, lane, std::bit_cast<uint32_t>(1.5f + u * 0.5f));
        wave_->debug_write_vgpr(offset + 1, lane, std::bit_cast<uint32_t>(1.5f + v * 0.5f));
        wave_->debug_write_vgpr(offset + 2, lane, 0);
      }
      ASSERT_NO_FATAL_FAILURE(sample(opcode, 3));
      for (uint32_t lane = 0; lane < 4; ++lane)
        for (uint32_t c = 0; c < 4; ++c)
          EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane), witness.expected);
      EXPECT_EQ(wave_->debug_read_vgpr(12, 31), 0xdeadbeefu);
    }

  // One quad spans +Y, +X and +Z. Both cards share the origin's footprint
  // across all lanes; unfolding onto lane 3's face instead gives mip 3.
  const float corner[4][3] = {{1.9375f, 1.96875f, 2.0f},
                              {1.90625f, 1.9375f, 2.0f},
                              {1.0625f, 1.03125f, 0.0f},
                              {1.9375f, 1.03125f, 4.0f}};
  wave_->set_exec(15);
  for (uint32_t lane = 0; lane < 4; ++lane)
    for (uint32_t c = 0; c < 3; ++c)
      wave_->debug_write_vgpr(c, lane, std::bit_cast<uint32_t>(corner[lane][c]));
  ASSERT_NO_FATAL_FAILURE(sample(27, 3));
  for (uint32_t lane = 0; lane < 4; ++lane)
    for (uint32_t c = 0; c < 4; ++c)
      EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane), 0x40390000u);
  // Captured halfway cases distinguish source-face rounding from magnitude
  // rounding and cover both signs of the unfolded coordinate transform.
  struct CornerWitness {
    uint32_t coordinates[4][3];
    uint32_t expected;
  };
  const CornerWitness corner_witnesses[] = {
      {{{0x3f859aefu, 0x3f82ceeau, 0x00000000u},
        {0x3ff9713du, 0x3ffab465u, 0x40000000u},
        {0x3ffa79eau, 0x3ffd0b48u, 0x40000000u},
        {0x3ffb9eacu, 0x3f856c1fu, 0x40800000u}},
       0x40350000u},
      {{{0x3ffd516du, 0x3f82e7cdu, 0x40400000u},
        {0x3fff6504u, 0x3fffc00du, 0x40800000u},
        {0x3fffec36u, 0x3f804df3u, 0x40400000u},
        {0x3fff33d9u, 0x3fff943eu, 0x40800000u}},
       0x3fa40000u},
      {{{0x3ffa6511u, 0x3f82ceeau, 0x3f800000u},
        {0x3f868ec3u, 0x3ffab465u, 0x40000000u},
        {0x3f858616u, 0x3ffd0b48u, 0x40000000u},
        {0x3f846154u, 0x3f856c1fu, 0x40800000u}},
       0x40358000u},
      {{{0x3f859aefu, 0x3ffd3116u, 0x00000000u},
        {0x3ff9713du, 0x3f854b9bu, 0x40400000u},
        {0x3ffa79eau, 0x3f82f4b8u, 0x40400000u},
        {0x3ffb9eacu, 0x3ffa93e1u, 0x40800000u}},
       0x40350000u},
      {{{0x3ffd516du, 0x3ffd1833u, 0x40400000u},
        {0x3f809afcu, 0x3fffc00du, 0x40a00000u},
        {0x3fffec36u, 0x3fffb20du, 0x40400000u},
        {0x3f80cc27u, 0x3fff943eu, 0x40a00000u}},
       0x3fa40000u},
      {{{0x3ffa6511u, 0x3ffd3116u, 0x00000000u},
        {0x3ff9713du, 0x3ffab465u, 0x40400000u},
        {0x3ffa79eau, 0x3ffd0b48u, 0x40400000u},
        {0x3f846154u, 0x3ffa93e1u, 0x40a00000u}},
       0x40350000u},
  };
  for (const auto &witness : corner_witnesses) {
    for (uint32_t lane = 0; lane < 4; ++lane)
      for (uint32_t c = 0; c < 3; ++c)
        wave_->debug_write_vgpr(c, lane, witness.coordinates[lane][c]);
    ASSERT_NO_FATAL_FAILURE(sample(27, 3));
    for (uint32_t lane = 0; lane < 4; ++lane)
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane), witness.expected);
  }
  // A non-power-of-two extent distinguishes the parallel-coordinate rule and
  // the reflected sum with 22 fractional bits before derivative conversion.
  setup_cube(127);
  const CornerWitness non_power_of_two[] = {
      {{{0x3ffad4f9u, 0x3ffe2842u, 0x40800000u},
        {0x3ff94b3bu, 0x3ff5b576u, 0x40800000u},
        {0x3f860d2au, 0x3ff95142u, 0x00000000u},
        {0x3ff7d038u, 0x3ffd6624u, 0x40800000u}},
       0x40710000u},
      {{{0x3ff98d25u, 0x3ff47332u, 0x40400000u},
        {0x3f897a5au, 0x3ff33712u, 0x40a00000u},
        {0x3ffa12a5u, 0x3ffe95a0u, 0x00000000u},
        {0x3f886537u, 0x3ffd4abau, 0x40a00000u}},
       0x40988000u},
      {{{0x3fff5614u, 0x3f8042ecu, 0x40400000u},
        {0x3f80c7d0u, 0x3fffebfau, 0x00000000u},
        {0x3ffedf04u, 0x3fff3680u, 0x40800000u},
        {0x3ffe93e6u, 0x3f80aeb5u, 0x40400000u}},
       0x3ef40000u},
      {{{0x3f80d7d5u, 0x3fff64bcu, 0x40400000u},
        {0x3f807792u, 0x3fff4c93u, 0x40400000u},
        {0x3f802f9bu, 0x3fffbd84u, 0x3f800000u},
        {0x3fffdf89u, 0x3fffbeddu, 0x40a00000u}},
       0x3d800000u},
  };
  for (const auto &witness : non_power_of_two) {
    for (uint32_t lane = 0; lane < 4; ++lane)
      for (uint32_t c = 0; c < 3; ++c)
        wave_->debug_write_vgpr(c, lane, witness.coordinates[lane][c]);
    ASSERT_NO_FATAL_FAILURE(sample(27, 3));
    for (uint32_t lane = 0; lane < 4; ++lane)
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane), witness.expected);
  }
}

TEST_P(GraphicsExportTest, Fp32FilteringMatchesPhysicalRoundingAndSpecialValues) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Witness {
    uint32_t pattern, query;
    std::array<uint32_t, 4> expected;
  };
  const Witness witnesses[] = {
      {0, 0, {0x3ee5747au, 0x3e0e5bc4u, 0x3e12bb23u, 0x3e379a0au}},
      {0, 1, {0x3ee570cbu, 0x3e0e5f92u, 0x3e12df73u, 0x3e37bad8u}},
      {0, 255, {0x3ee1c906u, 0x3e122661u, 0x3e36e6b6u, 0x3e58478eu}},
      {0, 256, {0x3ee56c9du, 0x3e0e8420u, 0x3e12c441u, 0x3e37df6au}},
      {0, 257, {0x3ee5686fu, 0x3e0e87cfu, 0x3e12e8cdu, 0x3e37fffbu}},
      {0, 32896, {0x3ebff983u, 0x3e1ca19eu, 0x3e3882c4u, 0x3e5b6214u}},
      {0, 65535, {0x3e383a88u, 0x3e1b5aa7u, 0x3e7bc5b7u, 0x3e609f98u}},
      {1, 0, {0xbee5747au, 0xbe0e5bc4u, 0xbe12bb23u, 0xbe379a0au}},
      {1, 1, {0xbee570cbu, 0xbe0e5f92u, 0xbe12df73u, 0xbe37bad8u}},
      {1, 255, {0xbee1c906u, 0xbe122661u, 0xbe36e6b6u, 0xbe58478eu}},
      {1, 256, {0xbee56c9du, 0xbe0e8420u, 0xbe12c441u, 0xbe37df6au}},
      {1, 257, {0xbee5686fu, 0xbe0e87cfu, 0xbe12e8cdu, 0xbe37fffbu}},
      {1, 32896, {0xbebff983u, 0xbe1ca19eu, 0xbe3882c4u, 0xbe5b6214u}},
      {1, 65535, {0xbe383a88u, 0xbe1b5aa7u, 0xbe7bc5b7u, 0xbe609f98u}},
      {2, 0, {0xc665747au, 0xc70e5bc4u, 0xc712bb23u, 0xc7379a0au}},
      {2, 1, {0xc6649d22u, 0xc70dd68bu, 0xc71233d9u, 0xc736eff6u}},
      {2, 255, {0xc46f3ad2u, 0xc51a7dbau, 0xc53f7fa7u, 0xc5630990u}},
      {2, 256, {0xc6648fe3u, 0xc70dce1fu, 0xc7122904u, 0xc736e36du}},
      {2, 257, {0xc663b962u, 0xc70d496au, 0xc711a240u, 0xc7363a03u}},
      {2, 32896, {0xc5749bf0u, 0xc6183ed3u, 0xc61ed772u, 0xc6462b92u}},
      {2, 65535, {0xc173fe02u, 0xc19ff7edu, 0xc1e1d1d3u, 0xc1e8c899u}},
      {3, 0, {0xffc00000u, 0x00000000u, 0x80000000u, 0xffc00000u}},
      {3, 1, {0xffc00000u, 0xbc000000u, 0x3b800000u, 0xffc00000u}},
      {3, 255, {0xffc00000u, 0xbfff0000u, 0x3f7f0000u, 0xffc00000u}},
      {3, 256, {0xffc00000u, 0x00000000u, 0xff800000u, 0xffc00000u}},
      {3, 257, {0xffc00000u, 0xbbff0000u, 0xffc00000u, 0xffc00000u}},
      {3, 32896, {0xffc00000u, 0xbf000000u, 0xffc00000u, 0xffc00000u}},
      {3, 65535, {0xffc00000u, 0xbbff0000u, 0xffc00000u, 0xffc00000u}},
  };
  for (uint32_t pattern = 0; pattern < 4; ++pattern) {
    cu_->l1_vector().invalidate_all();
    cache_.invalidate_all();
    for (uint32_t t = 0; t < 4; ++t)
      for (uint32_t c = 0; c < 4; ++c) {
        uint32_t q = t * 131 + c * 7 + 12345;
        q ^= q << 13;
        q ^= q >> 17;
        q ^= q << 5;
        uint32_t bits = 0x3e000000 + (q & 0xffffff);
        if (pattern == 1)
          bits |= q & 0x80000000;
        if (pattern == 2)
          bits = q & 0xff7fffff;
        if (pattern == 3) {
          constexpr uint32_t special[]{0,          0x80000000, 1,          0x80000001,
                                       0x7fffff,   0x807fffff, 0x3f800000, 0xbf800000,
                                       0x7f800000, 0xff800000, 0x7f800001, 0xff800001,
                                       0x7fc00001, 0xffc00001, 0x40000000, 0xc0000000};
          bits = special[q & 15];
        }
        const auto address = gfx12 ? amdgpu::gfx12_image_address(0x100000, t % 2, t / 2, 16, 16, 0)
                                   : amdgpu::gfx11_image_address(0x100000, t % 2, t / 2, 16, 16, 0);
        ASSERT_TRUE(address);
        memory_.write32(*address + 4 * c, bits);
      }
    const std::array<uint32_t, 8> descriptor{
        0x1000, (63u << (gfx12 ? 17 : 20)) | (1u << 30), 1u << 14, (9u << 28) | 0xfac, 15, 0, 0, 0};
    for (uint32_t i = 0; i < descriptor.size(); ++i)
      wave_->debug_write_sgpr(8 + i, descriptor[i]);
    wave_->debug_write_sgpr(4, 0x92);
    wave_->debug_write_sgpr(5, 0);
    wave_->debug_write_sgpr(6, (1u << 20) | (1u << 22));
    wave_->debug_write_sgpr(7, 0);
    wave_->set_exec(1);
    for (const auto &witness : witnesses) {
      if (witness.pattern != pattern)
        continue;
      wave_->debug_write_vgpr(0, 0,
                              std::bit_cast<uint32_t>(0.25f + (witness.query & 255) / 512.0f));
      wave_->debug_write_vgpr(1, 0, std::bit_cast<uint32_t>(0.25f + (witness.query >> 8) / 512.0f));
      ASSERT_NO_FATAL_FAILURE(sample(31, 1));
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(12 + c, 0), witness.expected[c])
            << pattern << "," << witness.query << "," << c;
    }
  }
}

TEST_P(GraphicsExportTest, AnisotropicCountsAndSpacingMatchPhysicalReadbacks) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const auto layout = amdgpu::image_mip_layout(gfx12, 0, 16, 64, 64, 1, 0);
  ASSERT_TRUE(layout);
  for (uint32_t y = 0; y < 64; ++y)
    for (uint32_t x = 0; x < 64; ++x) {
      const auto address = gfx12
                               ? amdgpu::gfx12_image_address(0x100000, x, y, layout->pitch, 16, 0)
                               : amdgpu::gfx11_image_address(0x100000, x, y, layout->pitch, 16, 0);
      ASSERT_TRUE(address);
      const float channels[] = {float(std::max(int(x) - 32, 0)), float(std::max(32 - int(x), 0)),
                                float(std::max(int(y) - 32, 0)), float(std::max(32 - int(y), 0))};
      for (uint32_t c = 0; c < 4; ++c)
        memory_.write32(*address + 4 * c, std::bit_cast<uint32_t>(channels[c]));
    }
  const std::array<uint32_t, 8> descriptor{0x1000,
                                           (63u << (gfx12 ? 17 : 20)) | (3u << 30),
                                           15 | (63u << 14),
                                           (9u << 28) | 0xfac,
                                           0,
                                           4u << 20,
                                           0,
                                           0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->debug_write_sgpr(5, 0);
  wave_->debug_write_sgpr(6, (3u << 20) | (3u << 22));
  wave_->debug_write_sgpr(7, 0);
  struct Witness {
    uint32_t max_anisotropy, bias, threshold;
    std::array<uint32_t, 4> gradients;
    int32_t phase;
    std::array<uint32_t, 4> expected;
    uint32_t mip_filter = 0;
  };
  // Identical raw RGBA32F readbacks on physical GFX11 and GFX12. Rotated
  // footprints cross a count boundary and exercise every even count, both
  // reconstruction tables, and bias redistribution with threshold controls.
  constexpr Witness witnesses[] = {
      {16,
       0,
       0,
       {0xbe940e97u, 0xbf7a0389u, 0x3e36996cu, 0xbf304c5fu},
       -26,
       {0x3c600000u, 0x3c800000u, 0x3dfc0000u, 0x3e000000u}},
      {16,
       0,
       0,
       {0x3e129a71u, 0xbfa7d059u, 0x3e9ec2fau, 0xbff798deu},
       -20,
       {0x3d280000u, 0x3d340000u, 0x3e960000u, 0x3e970000u}},
      {16,
       0,
       0,
       {0xbfdd0965u, 0x402879d4u, 0xc01d1f9bu, 0x3fcafaecu},
       -14,
       {0x3ebcc100u, 0x3ebcc100u, 0x3ec16a00u, 0x3ec21480u}},
      {16,
       0,
       0,
       {0x3e2c99a1u, 0x3ed1cb53u, 0xbfbc8651u, 0x40d63183u},
       -8,
       {0x3e3b0000u, 0x3e3d0000u, 0x3f574000u, 0x3f578000u}},
      {16,
       0,
       0,
       {0xc090a0f2u, 0xc0e23c12u, 0x4039b9cdu, 0x407fe00au},
       -2,
       {0x3f2c5d80u, 0x3f2c5d80u, 0x3f8265e0u, 0x3f82a5e0u}},
      {16,
       0,
       0,
       {0x403aa0e4u, 0xc0cae6ceu, 0xc03b9eecu, 0x40f3647bu},
       4,
       {0x3f046920u, 0x3f043e80u, 0x3f9eeaf0u, 0x3f9ed5a0u}},
      {16,
       0,
       0,
       {0xc0f7ff44u, 0x40d9d8f5u, 0xc0fa09a6u, 0x409b77a6u},
       10,
       {0x3fb077c0u, 0x3fb04a10u, 0x3f85a290u, 0x3f857df0u}},
      {16,
       0,
       0,
       {0xc03e060bu, 0x3cf68e78u, 0xc19962d1u, 0xc03d5fbeu},
       16,
       {0x401a0000u, 0x401a0000u, 0x3eb9e000u, 0x3eb8e000u}},
      {4,
       0,
       0,
       {0xbfa52c6cu, 0x3fe6900au, 0xbf8beb8du, 0xc033bf44u},
       -31,
       {0x3d140000u, 0x3d200000u, 0x3ed48000u, 0x3ed68000u}},
      {16,
       16,
       0,
       {0xc0f7ff44u, 0x40d9d8f5u, 0xc0fa09a6u, 0x409b77a6u},
       1,
       {0x3fafa560u, 0x3fafa560u, 0x3f84f3a0u, 0x3f84f3a0u}},
      {16,
       16,
       0,
       {0xbfdd0965u, 0x402879d4u, 0xc01d1f9bu, 0x3fcafaecu},
       7,
       {0x3ebd8000u, 0x3ebc8000u, 0x3ec20000u, 0x3ec20000u}},
      {16,
       16,
       0,
       {0xc06206b2u, 0x40ad7d2bu, 0xbfe6d7a8u, 0x3f9c226fu},
       14,
       {0x3efaac80u, 0x3ef9ac80u, 0x3f31c4c0u, 0x3f3144c0u}},
      {16,
       16,
       0,
       {0x3feb0040u, 0xc0b85fcau, 0x41312b68u, 0xc10a47e0u},
       3,
       {0x3faf2000u, 0x3faf2000u, 0x3fa0e000u, 0x3fa0a000u}},
      {16,
       16,
       0,
       {0xc089f1dbu, 0x4146bc0bu, 0xc083611du, 0x40e70d25u},
       8,
       {0x3f38a7e0u, 0x3f3874a0u, 0x3fe35280u, 0x3fe35280u}},
      {16,
       16,
       0,
       {0xbf7c5ad0u, 0x40026b1eu, 0xc00cf53fu, 0x3efd8518u},
       3,
       {0x3e900000u, 0x3e900000u, 0x3e660000u, 0x3e640000u}},
      {16,
       63,
       0,
       {0xc089f1dbu, 0x4146bc0bu, 0xc083611du, 0x40e70d25u},
       -1,
       {0x3f394240u, 0x3f394240u, 0x3fe3f3e0u, 0x3fe433e0u}},
      {16,
       63,
       7,
       {0xc089f1dbu, 0x4146bc0bu, 0xc083611du, 0x40e70d25u},
       -1,
       {0x3f36c300u, 0x3f36c300u, 0x3fe0f4e0u, 0x3fe134e0u}},
      // Linear mip filtering retains finer span precision, including when
      // both mip taps select this single-level image.
      {2,
       0,
       0,
       {0xbf82e328u, 0x40d08320u, 0x3ebc90a6u, 0xc0568304u},
       -26,
       {0x3e080000u, 0x3e0a0000u, 0x3f680000u, 0x3f690000u},
       2},
      {4,
       0,
       0,
       {0x407ea1c2u, 0xc03d4ca7u, 0x40211620u, 0xbfc9eb88u},
       15,
       {0x3f170000u, 0x3f168000u, 0x3ed70000u, 0x3ed60000u},
       2},
      {8,
       0,
       0,
       {0xc0ca76f8u, 0xc12aa528u, 0xbf84d27du, 0xc0a4fa45u},
       15,
       {0x3f47a000u, 0x3f472000u, 0x3fbca000u, 0x3fbc6000u},
       2},
      {16,
       0,
       0,
       {0x3fd0c0bbu, 0xbed1ef98u, 0xc0301859u, 0x411b09a0u},
       -7,
       {0x3eb88000u, 0x3eb90000u, 0x3f9b9000u, 0x3f9bb000u},
       2},
      {4,
       16,
       0,
       {0x407ea1c2u, 0xc03d4ca7u, 0x40211620u, 0xbfc9eb88u},
       30,
       {0x3f174000u, 0x3f164000u, 0x3ed70000u, 0x3ed58000u},
       2},
      {16,
       16,
       0,
       {0x40c2c47cu, 0xbf2cb363u, 0x41830fbcu, 0x406e7efdu},
       8,
       {0x400c0000u, 0x400c0000u, 0x3ed48000u, 0x3ed40000u},
       2},
      {16,
       0,
       7,
       {0x3fd0c0bbu, 0xbed1ef98u, 0xc0301859u, 0x411b09a0u},
       -7,
       {0x3eb88000u, 0x3eb90000u, 0x3f9b9000u, 0x3f9bb000u},
       2},
  };
  wave_->set_exec(1);
  for (const auto &witness : witnesses) {
    wave_->debug_write_sgpr(6, (3u << 20) | (3u << 22) | (witness.mip_filter << 26));
    wave_->debug_write_sgpr(4, 0x92 | (std::countr_zero(witness.max_anisotropy) << 9) |
                                   (witness.threshold << 16) | (witness.bias << 21) | (1u << 27));
    for (uint32_t i = 0; i < 4; ++i)
      wave_->debug_write_vgpr(
          i, 0, std::bit_cast<uint32_t>(std::bit_cast<float>(witness.gradients[i]) / 64));
    const float coordinate = (32.5f + witness.phase / 8192.0f) / 64;
    wave_->debug_write_vgpr(4, 0, std::bit_cast<uint32_t>(coordinate));
    wave_->debug_write_vgpr(5, 0, std::bit_cast<uint32_t>(coordinate));
    ASSERT_NO_FATAL_FAILURE(sample(28, 1));
    for (uint32_t c = 0; c < 4; ++c)
      EXPECT_EQ(wave_->debug_read_vgpr(12 + c, 0), witness.expected[c])
          << witness.max_anisotropy << ',' << witness.bias << ',' << witness.phase << ',' << c;
  }
}

TEST_P(GraphicsExportTest, AnisotropicFloatingAccumulationMatchesPhysicalReadbacks) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Witness {
    uint32_t format, max_anisotropy, bias;
    std::array<uint32_t, 4> gradients;
    int32_t phase;
    std::array<uint32_t, 4> expected;
  };
  // Raw physical GFX11/GFX12 results for a seeded texture. sRGB and FP16
  // expose intermediate accumulation; FP32 cancellation also distinguishes
  // operand alignment and retention of the accumulator exponent.
  constexpr Witness witnesses[] = {
      {66,
       16,
       0,
       {0xc129dff5u, 0xc01193ceu, 0xc0d91b8fu, 0xc0884878u},
       -29,
       {0x3e877469u, 0x3e679331u, 0x3e3cad84u, 0x3eac1192u}},
      {66,
       16,
       16,
       {0xc129dff5u, 0xc01193ceu, 0xc0d91b8fu, 0xc0884878u},
       -29,
       {0x3e877469u, 0x3e679331u, 0x3e3cad84u, 0x3eac1192u}},
      {57,
       2,
       0,
       {0xc12fda97u, 0x41a04872u, 0xc1221509u, 0x415ec7cau},
       1,
       {0xc2fd81d6u, 0x40e16010u, 0x41b1f62cu, 0xc193d850u}},
      {57,
       4,
       0,
       {0xc03e060bu, 0x3cf68e78u, 0xc19962d1u, 0xc03d5fbeu},
       -16,
       {0xc39c9d75u, 0x409e393eu, 0xc2c4018cu, 0x42e95d51u}},
      {57,
       16,
       16,
       {0xc03e060bu, 0x3cf68e78u, 0xc19962d1u, 0xc03d5fbeu},
       -29,
       {0xc1eb252cu, 0x42966696u, 0xc2c9e0beu, 0x41e5f944u}},
      {63,
       2,
       0,
       {0xbf6dd938u, 0xbe93684bu, 0xbeae4847u, 0xbe863b6bu},
       -29,
       {0xc267165du, 0xc25ed558u, 0xc6c43c87u, 0xc1edd99cu}},
      {63,
       2,
       0,
       {0xc0c6768eu, 0xc085be2cu, 0x41e515d2u, 0x40c52df9u},
       9,
       {0xc561eb8bu, 0x3c42e800u, 0xc38770b3u, 0xc5383b8au}},
      {63,
       4,
       0,
       {0x3e129a71u, 0xbfa7d059u, 0x3e9ec2fau, 0xbff798deu},
       9,
       {0xc2f836e8u, 0x43f87c95u, 0xc6259cecu, 0xc532fdb2u}},
      {63,
       4,
       0,
       {0x3f533b6fu, 0x3ec0655du, 0xbf806c3bu, 0xbed562d0u},
       20,
       {0x3bd67940u, 0xc3c113b0u, 0xc637d948u, 0xc44d624fu}},
      {63,
       16,
       0,
       {0x3e129a71u, 0xbfa7d059u, 0x3e9ec2fau, 0xbff798deu},
       9,
       {0xc2f836e8u, 0x43f87c95u, 0xc6259cecu, 0xc532fdb2u}},
      {63,
       16,
       0,
       {0xc112e8d2u, 0xc12b9ff7u, 0x40a16424u, 0x40ae589cu},
       3,
       {0xc3faa53eu, 0xc201b8f1u, 0x3ef1d500u, 0x42afe091u}},
      {63,
       16,
       16,
       {0x3e129a71u, 0xbfa7d059u, 0x3e9ec2fau, 0xbff798deu},
       9,
       {0xc2f836e8u, 0x43f87c95u, 0xc6259cecu, 0xc532fdb2u}},
      {63,
       16,
       16,
       {0xc112e8d2u, 0xc12b9ff7u, 0x40a16424u, 0x40ae589cu},
       3,
       {0xc3faa53eu, 0xc201b8f1u, 0x3ef1d500u, 0x42afe091u}},
  };
  for (uint32_t format : {66u, 57u, 63u}) {
    cu_->l1_vector().invalidate_all();
    cache_.invalidate_all();
    const uint32_t bytes = format == 66 ? 4 : format == 57 ? 8 : 16;
    const auto layout = amdgpu::image_mip_layout(gfx12, 0, bytes, 64, 64, 1, 0);
    ASSERT_TRUE(layout);
    uint32_t state = 0x7941332d;
    const auto next = [&]() {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      return state;
    };
    for (uint32_t y = 0; y < 64; ++y)
      for (uint32_t x = 0; x < 64; ++x) {
        const auto address =
            gfx12 ? amdgpu::gfx12_image_address(0x100000, x, y, layout->pitch, bytes, 0)
                  : amdgpu::gfx11_image_address(0x100000, x, y, layout->pitch, bytes, 0);
        ASSERT_TRUE(address);
        if (format == 66) {
          memory_.write32(*address, next());
          continue;
        }
        for (uint32_t c = 0; c < 4; ++c) {
          const uint32_t bits = next();
          if (format == 57)
            memory_.write16(*address + c * 2, (bits & 0x83ff) | (((bits >> 10) % 24 + 3) << 10));
          else
            memory_.write32(*address + c * 4,
                            (bits & 0x807fffff) | (((bits >> 23) % 32 + 111) << 23));
        }
      }
    const std::array<uint32_t, 8> descriptor{0x1000,
                                             (format << (gfx12 ? 17 : 20)) | (3u << 30),
                                             15 | (63u << 14),
                                             (9u << 28) | 0xfac,
                                             0,
                                             4u << 20,
                                             0,
                                             0};
    for (uint32_t i = 0; i < descriptor.size(); ++i)
      wave_->debug_write_sgpr(8 + i, descriptor[i]);
    wave_->debug_write_sgpr(5, 0);
    wave_->debug_write_sgpr(6, (3u << 20) | (3u << 22));
    wave_->debug_write_sgpr(7, 0);
    wave_->set_exec(1);
    for (const auto &witness : witnesses) {
      if (witness.format != format)
        continue;
      wave_->debug_write_sgpr(4, 0x92 | (std::countr_zero(witness.max_anisotropy) << 9) |
                                     (witness.bias << 21) | (1u << 27));
      for (uint32_t i = 0; i < 4; ++i)
        wave_->debug_write_vgpr(
            i, 0, std::bit_cast<uint32_t>(std::bit_cast<float>(witness.gradients[i]) / 64));
      const float coordinate = (32.5f + witness.phase / 8192.0f) / 64;
      wave_->debug_write_vgpr(4, 0, std::bit_cast<uint32_t>(coordinate));
      wave_->debug_write_vgpr(5, 0, std::bit_cast<uint32_t>(coordinate));
      ASSERT_NO_FATAL_FAILURE(sample(28, 1));
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(12 + c, 0), witness.expected[c])
            << format << ',' << witness.max_anisotropy << ',' << witness.bias << ','
            << witness.phase << ',' << c;
    }
  }
}

TEST_P(GraphicsExportTest, AnisotropicFloatingSpecialValuesMatchPhysicalReadbacks) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint32_t pairs[][2] = {
      {0x00000000u, 0x00000000u}, {0x80000000u, 0x80000000u}, {0x00000000u, 0x80000000u},
      {0x80000000u, 0x00000000u}, {0x3f800000u, 0xbf800000u}, {0x7f800000u, 0x3f800000u},
      {0xff800000u, 0x3f800000u}, {0x7f800000u, 0x7f800000u}, {0xff800000u, 0xff800000u},
      {0x7f800000u, 0xff800000u}, {0x7fc12345u, 0x3f800000u}, {0xffc12345u, 0x3f800000u},
      {0x7f812345u, 0x00000000u}, {0x00000001u, 0x00000000u}, {0x80000001u, 0x00000000u},
      {0x00800000u, 0x80800000u}, {0x3f800001u, 0xbf800000u}, {0x3f800000u, 0xbf800001u},
      {0x3f800000u, 0x33800000u}, {0xbf800000u, 0x33800000u}, {0x40000000u, 0xbfffffffu},
      {0x00800000u, 0x00000000u}, {0x80800000u, 0x00000000u}, {0x7f7fffffu, 0xff7fffffu},
  };
  // Linear and nearest anisotropic filters, captured on both physical targets.
  // Include signed zeros, infinities, NaNs, cancellation and output underflow.
  constexpr uint32_t expected[2][std::size(pairs)] = {
      {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x7f800000u,
       0xff800000u, 0x7f800000u, 0xff800000u, 0xffc00000u, 0xffc00000u, 0xffc00000u,
       0xffc00000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x33000000u, 0xb3000000u,
       0x3e800000u, 0xbe7fffffu, 0x33000000u, 0x00000000u, 0x80000000u, 0x00000000u},
      {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0xbf000000u, 0x3f000000u,
       0x3f000000u, 0x7f800000u, 0xff800000u, 0xff800000u, 0x3f000000u, 0x3f000000u,
       0x00000000u, 0x00000000u, 0x00000000u, 0x80000000u, 0xbf000000u, 0xbf000001u,
       0x33000000u, 0x33000000u, 0xbf7fffffu, 0x00000000u, 0x00000000u, 0xfeffffffu},
  };
  const auto layout = amdgpu::image_mip_layout(gfx12, 0, 16, 64, 64, 1, 0);
  ASSERT_TRUE(layout);
  for (uint32_t y = 0; y < 64; ++y)
    for (uint32_t x = 0; x < 64; ++x) {
      const auto address = gfx12
                               ? amdgpu::gfx12_image_address(0x100000, x, y, layout->pitch, 16, 0)
                               : amdgpu::gfx11_image_address(0x100000, x, y, layout->pitch, 16, 0);
      ASSERT_TRUE(address);
      for (uint32_t c = 0; c < 4; ++c) {
        const auto &pair = pairs[(y + c) % std::size(pairs)];
        const uint32_t value = x < 32   ? pair[0]
                               : x > 32 ? pair[1]
                                        : (pair[0] & pair[1] & 0x80000000u);
        memory_.write32(*address + c * 4, value);
      }
    }
  const std::array<uint32_t, 8> descriptor{0x1000,
                                           (63u << (gfx12 ? 17 : 20)) | (3u << 30),
                                           15 | (63u << 14),
                                           (9u << 28) | 0xfac,
                                           0,
                                           4u << 20,
                                           0,
                                           0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->debug_write_sgpr(4, 0x92 | (1u << 9) | (1u << 27));
  wave_->debug_write_sgpr(5, 0);
  wave_->debug_write_sgpr(7, 0);
  wave_->set_exec(1);
  const float gradients[] = {2.0f / 64, 0, 0, 1.0f / 64};
  for (uint32_t i = 0; i < 4; ++i)
    wave_->debug_write_vgpr(i, 0, std::bit_cast<uint32_t>(gradients[i]));
  wave_->debug_write_vgpr(4, 0, std::bit_cast<uint32_t>(32.5f / 64));
  for (uint32_t nearest : {0u, 1u}) {
    const uint32_t filter = nearest ? 2 : 3;
    wave_->debug_write_sgpr(6, (filter << 20) | (filter << 22));
    for (uint32_t row = 0; row < std::size(pairs); ++row) {
      wave_->debug_write_vgpr(5, 0, std::bit_cast<uint32_t>((row + 0.5f) / 64));
      ASSERT_NO_FATAL_FAILURE(sample(28, 1));
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(12 + c, 0),
                  expected[nearest][(row + c) % std::size(pairs)])
            << nearest << ',' << row << ',' << c;
    }
  }
}
