// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "counting_gpu_memory.h"
#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/mtbuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/mubuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/mtbuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/mubuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/mtbuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/mubuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/mtbuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/mubuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/mtbuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/mubuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/mtbuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/mubuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/mtbuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/mubuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/mtbuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/mubuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vbuffer.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/hwfloat/unorm.h"
#include "rocjitsu/vm/amdgpu/buffer_format.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <memory>

namespace {
using namespace rocjitsu;
constexpr uint32_t identity = 4 | (5 << 3) | (6 << 6) | (7 << 9);
uint32_t bits(float f) { return std::bit_cast<uint32_t>(f); }

TEST(BufferFormatTest, UnpacksNormalizedSignedHalfAndPackedChannels) {
  const std::array<uint8_t, 4> data{0x00, 0x80, 0xff, 0x7f};
  EXPECT_EQ(amdgpu::unpack_buffer_format(42, identity, data).value(),
            (std::array<uint32_t, 4>{bits(0), bits(128.0f / 255), bits(1), bits(127.0f / 255)}));
  EXPECT_EQ(amdgpu::unpack_buffer_format(47, identity, data).value(),
            (std::array<uint32_t, 4>{0, 0xffffff80, 0xffffffff, 127}));
  const std::array<uint8_t, 2> snorm{0x00, 0x80};
  EXPECT_EQ(amdgpu::unpack_buffer_format(8, identity, snorm).value()[0], bits(-1));
  const std::array<uint8_t, 4> half{0x00, 0x3c, 0x00, 0xc0};
  EXPECT_EQ(amdgpu::unpack_buffer_format(29, identity, half).value(),
            (std::array<uint32_t, 4>{bits(1), bits(-2), 0, 0}));
  // 11_11_10 unsigned float channels: 1, 2, 0.5.
  const uint32_t packed = (15u << 6) | ((16u << 6) << 11) | ((14u << 5) << 22);
  const auto raw = std::bit_cast<std::array<uint8_t, 4>>(packed);
  EXPECT_EQ(amdgpu::unpack_buffer_format(30, identity, raw).value(),
            (std::array<uint32_t, 4>{bits(1), bits(2), bits(0.5f), 0}));
}

TEST(BufferFormatTest, CountsStoredComponentsForNarrowAndPackedFormats) {
  for (uint32_t format : {20u, 21u, 22u})
    EXPECT_EQ(amdgpu::decode_buffer_format(format).value().component_count(), 1u);
  for (uint32_t format : {48u, 49u, 50u})
    EXPECT_EQ(amdgpu::decode_buffer_format(format).value().component_count(), 2u);
  EXPECT_EQ(amdgpu::decode_buffer_format(30).value().component_count(), 3u);
  for (uint32_t format : {42u, 43u, 44u, 45u, 46u, 47u})
    EXPECT_EQ(amdgpu::decode_buffer_format(format).value().component_count(), 4u);
  EXPECT_EQ(amdgpu::decode_buffer_format(0).value().component_count(), 0u);
}

TEST(BufferFormatTest, PacksConvertedComponentsAndReplicatesMissingShaderValues) {
  std::array<uint8_t, 4> bytes{};
  const std::array<uint32_t, 4> values{bits(-2), bits(0.5f), bits(1), bits(2)};
  ASSERT_TRUE(amdgpu::pack_buffer_format(42, identity, values, bytes).succeeded());
  EXPECT_EQ(bytes, (std::array<uint8_t, 4>{0, 128, 255, 255}));
  const std::array<uint32_t, 2> integers{0x12, 0x34};
  ASSERT_TRUE(amdgpu::pack_buffer_format(46, identity, integers, bytes).succeeded());
  EXPECT_EQ(bytes, (std::array<uint8_t, 4>{0x12, 0x34, 0x12, 0x12}));
  const std::array<uint32_t, 2> halves{bits(1), bits(-2)};
  ASSERT_TRUE(amdgpu::pack_buffer_format(29, identity, halves, bytes).succeeded());
  EXPECT_EQ(bytes, (std::array<uint8_t, 4>{0, 0x3c, 0, 0xc0}));
  const std::array<uint32_t, 3> floats{bits(1), bits(2), bits(0.5f)};
  ASSERT_TRUE(amdgpu::pack_buffer_format(30, identity, floats, bytes).succeeded());
  EXPECT_EQ(std::bit_cast<uint32_t>(bytes), (15u << 6) | ((16u << 6) << 11) | ((14u << 5) << 22));
}

TEST(BufferFormatTest, UnormBitsClampClassifyAndRoundQuantizationEdges) {
  constexpr std::array<uint32_t, 4> widths{2, 8, 10, 16};
  struct Case {
    uint32_t input;
    std::array<uint32_t, 4> expected;
  };
  constexpr Case cases[] = {
      {0x00000000, {0, 0, 0, 0}},          {0x80000000, {0, 0, 0, 0}},
      {0x00000001, {0, 0, 0, 0}},          {0x007fffff, {0, 0, 0, 0}},
      {0x00800000, {0, 0, 0, 0}},          {0xbf000000, {0, 0, 0, 0}},
      {0x7f800001, {0, 0, 0, 0}},          {0xff800001, {0, 0, 0, 0}},
      {0x7fc00000, {0, 0, 0, 0}},          {0xffffffff, {0, 0, 0, 0}},
      {0xff800000, {0, 0, 0, 0}},          {0x7f800000, {3, 255, 1023, 65535}},
      {0x3f7fffff, {3, 255, 1023, 65535}}, {0x3f800000, {3, 255, 1023, 65535}},
      {0x3f800001, {3, 255, 1023, 65535}}, {0x3effffff, {1, 127, 511, 32767}},
      {0x3f000000, {2, 128, 512, 32768}},  {0x3f000001, {2, 128, 512, 32768}},
      {0x3e2aaaaa, {0, 42, 170, 10922}},   {0x3e2aaaab, {1, 43, 171, 10923}},
      {0x3b008080, {0, 0, 2, 128}},        {0x3b008081, {0, 1, 2, 129}},
      {0x3a002008, {0, 0, 0, 32}},         {0x3a002009, {0, 0, 1, 32}},
      {0x37000080, {0, 0, 0, 0}},          {0x37000081, {0, 0, 0, 1}}};
  for (const auto &test : cases)
    for (size_t i = 0; i < widths.size(); ++i)
      EXPECT_EQ(amdgpu::hwfloat::unorm_from_f32(test.input, widths[i]), test.expected[i])
          << "input=" << std::hex << test.input << " width=" << std::dec << widths[i];
  EXPECT_FALSE(amdgpu::hwfloat::supports_unorm_width(11));
  EXPECT_FALSE(amdgpu::hwfloat::supports_unorm_width(32));
}

TEST(BufferFormatTest, UnormBitsMatchPreviousDoubleQuantization) {
  const amdgpu::fp_mode::detail::ScopedFenv environment(0);
  const auto reference = [](uint32_t input, uint32_t width) {
    const float value = std::bit_cast<float>(input);
    const double number = std::isnan(value) ? 0 : value;
    const double scaled = std::clamp(number, 0.0, 1.0) * ((1u << width) - 1);
    const double lo = std::floor(scaled), fraction = scaled - lo;
    return static_cast<uint32_t>(lo +
                                 (fraction > 0.5 || (fraction == 0.5 && std::fmod(lo, 2.0) != 0)));
  };
  for (uint32_t width : {2u, 8u, 10u, 16u}) {
    uint32_t random = 0x73b521e9;
    for (uint32_t i = 0; i < 8192; ++i) {
      random ^= random << 13;
      random ^= random >> 17;
      random ^= random << 5;
      ASSERT_EQ(amdgpu::hwfloat::unorm_from_f32(random, width), reference(random, width));
    }
    const uint32_t maximum = (1u << width) - 1;
    for (uint32_t integer = 0; integer < maximum; ++integer) {
      const uint32_t midpoint = bits(static_cast<float>((double(integer) + 0.5) / maximum));
      for (uint32_t input : {midpoint - 1, midpoint, midpoint + 1})
        ASSERT_EQ(amdgpu::hwfloat::unorm_from_f32(input, width), reference(input, width))
            << "input=" << std::hex << input << " width=" << std::dec << width;
    }
  }
}

TEST(BufferFormatTest, UnormStoresPreserveHostEnvironmentAndWiderFormatRounding) {
  const amdgpu::fp_mode::detail::ScopedFenv restore_environment(0);
  using E = amdgpu::BufferFormatEncoding;
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    for (uint32_t flush = 0; flush < 4; ++flush) {
      ASSERT_EQ(std::fesetround(mode), 0);
      ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
      ASSERT_EQ(std::feraiseexcept(FE_INVALID | FE_DIVBYZERO), 0);
#if defined(__x86_64__) || defined(__i386__)
      _mm_setcsr((_mm_getcsr() & ~((1u << 6) | (1u << 15))) | ((flush & 1) ? 1u << 6 : 0) |
                 ((flush & 2) ? 1u << 15 : 0));
      const auto before_mxcsr = _mm_getcsr();
#endif
      const std::array<uint32_t, 4> input{0x7f800001, 0x3f000000, 0x7f800000, 0x80000001};
      std::array<uint8_t, 8> bytes{};
      ASSERT_TRUE(
          amdgpu::pack_buffer_format(42, identity, input, std::span(bytes).first(4)).succeeded());
      EXPECT_EQ(bytes, (std::array<uint8_t, 8>{0, 128, 255, 0, 0, 0, 0, 0}));
      ASSERT_TRUE(amdgpu::pack_buffer_format(51, identity, input, bytes).succeeded());
      EXPECT_EQ(bytes, (std::array<uint8_t, 8>{0, 0, 0, 128, 255, 255, 0, 0}));
      constexpr uint32_t reverse = 7 | (6 << 3) | (5 << 6) | (4 << 9);
      ASSERT_TRUE(
          amdgpu::pack_buffer_format(36, reverse, input, std::span(bytes).first(4)).succeeded());
      EXPECT_EQ((std::array<uint8_t, 4>{bytes[0], bytes[1], bytes[2], bytes[3]}),
                (std::array<uint8_t, 4>{0, 252, 15, 32}));
      // Width 32 deliberately retains the existing double intermediate. Exact
      // integer quantization would differ here because of double rounding.
      const std::array<uint32_t, 1> wide{0x3f000001};
      ASSERT_TRUE(amdgpu::pack_buffer_format(4, identity, wide, std::span(bytes).first(4), E::Gfx9)
                      .succeeded());
      EXPECT_EQ((std::array<uint8_t, 4>{bytes[0], bytes[1], bytes[2], bytes[3]}),
                (std::array<uint8_t, 4>{0, 1, 0, 128}));
      EXPECT_EQ(std::fegetround(), mode);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID | FE_DIVBYZERO);
#if defined(__x86_64__) || defined(__i386__)
      EXPECT_EQ(_mm_getcsr(), before_mxcsr);
#endif
    }
  }
}

TEST(BufferFormatTest, SelectorsAndOobConstantsRespectNumericType) {
  constexpr uint32_t selectors = 6 | (5 << 3) | (4 << 6) | (1 << 9);
  const std::array<uint8_t, 4> bytes{11, 22, 33, 44};
  EXPECT_EQ(amdgpu::unpack_buffer_format(46, selectors, bytes).value(),
            (std::array<uint32_t, 4>{33, 22, 11, 1}));
  EXPECT_EQ(amdgpu::unpack_buffer_format(46, selectors, {}).value(),
            (std::array<uint32_t, 4>{0, 0, 0, 1}));
  EXPECT_EQ(amdgpu::unpack_buffer_format(42, selectors, {}).value(),
            (std::array<uint32_t, 4>{0, 0, 0, bits(1)}));
  std::array<uint8_t, 1> alpha{};
  const std::array<uint32_t, 4> channels{bits(0), bits(0), bits(0), bits(1)};
  ASSERT_TRUE(amdgpu::pack_buffer_format(1, 4 << 9, channels, alpha).succeeded());
  EXPECT_EQ(alpha[0], 255);
  std::array<uint8_t, 4> packed{};
  const std::array<uint32_t, 4> ten_bit{1, 2, 3, 2};
  ASSERT_TRUE(amdgpu::pack_buffer_format(40, identity, ten_bit, packed).succeeded());
  EXPECT_EQ(std::bit_cast<uint32_t>(packed), 1u | (2u << 10) | (3u << 20) | (2u << 30));
  EXPECT_EQ(amdgpu::buffer_format_bytes(5).value(), 1);
  EXPECT_EQ(amdgpu::buffer_format_bytes(29).value(), 4);
  EXPECT_EQ(amdgpu::buffer_format_bytes(60).value(), 12);
  EXPECT_EQ(amdgpu::buffer_format_bytes(63).value(), 16);
}

TEST(BufferFormatTest, LegacyFormatTablesDecodeTheirOwnNumericAndChannelFields) {
  using E = amdgpu::BufferFormatEncoding;
  const std::array<uint8_t, 4> bytes{0, 128, 255, 127};
  const std::array<uint32_t, 4> expected{bits(0), bits(128.0f / 255), bits(1), bits(127.0f / 255)};
  EXPECT_EQ(amdgpu::unpack_buffer_format(56, identity, bytes, E::Rdna1).value(), expected);
  EXPECT_EQ(amdgpu::unpack_buffer_format(56, identity, bytes, E::Rdna2).value(), expected);
  EXPECT_EQ(amdgpu::unpack_buffer_format(10, identity, bytes, E::Gfx9).value(), expected);
  const uint32_t packed = 1 | (2 << 11) | (3 << 22);
  const auto raw = std::bit_cast<std::array<uint8_t, 4>>(packed);
  const std::array<uint32_t, 4> integers{1, 2, 3, 0};
  EXPECT_EQ(amdgpu::unpack_buffer_format(34, identity, raw, E::Rdna1).value(), integers);
  EXPECT_EQ(amdgpu::unpack_buffer_format(6 | (4 << 4), identity, raw, E::Gfx9).value(), integers);
  EXPECT_EQ(amdgpu::buffer_format_bytes(77, E::Rdna1).value(), 16);
  EXPECT_EQ(amdgpu::buffer_format_bytes(74, E::Rdna2).value(), 12);
  EXPECT_EQ(amdgpu::buffer_format_bytes(13 | (7 << 4), E::Gfx9).value(), 12);
  EXPECT_TRUE(amdgpu::buffer_format_bytes(34, E::Rdna2).failed());
  EXPECT_TRUE(amdgpu::buffer_format_bytes(46, E::Rdna2).failed());
  EXPECT_TRUE(amdgpu::buffer_format_bytes(1 | (6 << 4), E::Gfx9).failed());
}

TEST(BufferFormatTest, UnsupportedFormatsReturnFailureWithoutModifyingStoreBytes) {
  using E = amdgpu::BufferFormatEncoding;
  for (const auto &[format, encoding] : std::array<std::pair<uint32_t, E>, 4>{
           {{64, E::Gfx11}, {78, E::Rdna1}, {34, E::Rdna2}, {0x61, E::Gfx9}}}) {
    std::array<uint8_t, 4> bytes{1, 2, 3, 4};
    const std::array<uint32_t, 1> components{0};
    EXPECT_TRUE(amdgpu::buffer_format_bytes(format, encoding).failed());
    EXPECT_TRUE(amdgpu::unpack_buffer_format(format, identity, bytes, encoding).failed());
    EXPECT_TRUE(amdgpu::pack_buffer_format(format, identity, components, bytes, encoding).failed());
    EXPECT_EQ(bytes, (std::array<uint8_t, 4>{1, 2, 3, 4}));
  }
}

TEST(BufferFormatTest, RdnaSwizzleUsesIndexOrLaneAndKeepsScalarOffsetLinear) {
  using amdgpu::addr_calc::rdna_buffer_address;
  constexpr uint32_t add_tid = 1u << 23;
  // The same address witnesses match physical gfx1100 and gfx1201.
  EXPECT_EQ(rdna_buffer_address(1u << 30, add_tid, 32, true, 1, 4, 4, 5).offset, 40);
  EXPECT_EQ(rdna_buffer_address(1u << 30, add_tid, 32, false, 0, 4, 4, 5).offset, 56);
  EXPECT_EQ(rdna_buffer_address(1u << 30, 0, 32, false, 0, 4, 4, 5).offset, 8);
  EXPECT_EQ(rdna_buffer_address(3u << 30, 0, 32, true, 1, 4, 4, 5).offset, 24);
  EXPECT_EQ(rdna_buffer_address(1u << 30, add_tid, 32, false, 0, 0, 0, 63).offset, 1820);
  EXPECT_EQ(rdna_buffer_address(0, add_tid, 32, true, 1, 4, 4, 5).offset, 40);
  EXPECT_EQ(rdna_buffer_address(0, add_tid, 32, false, 0, 4, 4, 5).offset, 168);
}

TEST(BufferFormatTest, Cdna4FormattedLoadCanWriteLdsWithoutClobberingVgprs) {
  amdgpu::GpuMemory memory("format_lds_memory");
  amdgpu::L2Cache l2("format_lds_l2");
  l2.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 32;
  cfg.vgprs_per_wf = 32;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("format_lds_cu", cfg, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0, 32, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);
  wf->set_m0(64);
  const auto sb = wf->sgpr_alloc().base, vb = wf->vgpr_alloc().base;
  cu->write_sgpr(sb + 4, 0x1000);
  cu->write_sgpr(sb + 5, 0);
  cu->write_sgpr(sb + 6, 4);
  cu->write_sgpr(sb + 7, identity | (2 << 15) | (4 << 12)); // 16_UINT.
  cu->write_vgpr(vb + 8, 0, 0xdeadbeef);
  memory.write32(0x1000, 0xabcd1234);
  cdna4::MubufMachineInst m{};
  m.srsrc = 1;
  m.soffset = 128;
  m.vdata = 8;
  m.lds = 1;
  auto *inst = new cdna4::BufferLoadFormatXMubuf(reinterpret_cast<const cdna4::MachineInst *>(&m));
  inst->execute_impl(*wf);
  ASSERT_NE(inst->data(), nullptr);
  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
  pipeline.issue(inst, *wf);
  uint32_t result = 0;
  static_cast<const amdgpu::Lds &>(wf->lds()).read(64u, reinterpret_cast<uint8_t *>(&result), 4);
  EXPECT_EQ(result, 0x1234);
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0xdeadbeef);
  wf->halt();
}

class BufferFormatExecutionTest : public ::testing::TestWithParam<rj_code_arch_t> {
protected:
  amdgpu::GpuMemory memory{"formatted_memory"};
  amdgpu::L2Cache l2{"formatted_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wf = nullptr;
  void SetUp() override {
    l2.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = GetParam();
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 32;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("formatted_cu", cfg, &memory, &l2);
    wf = cu->dispatch_wf(0, 0, 106, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(5); // Lane 1 must not read, store or receive a load result.
  }
  void TearDown() override {
    if (wf)
      wf->halt();
  }
  uint32_t encoded_format(uint32_t modern) {
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA1 || GetParam() == ROCJITSU_CODE_ARCH_RDNA2)
      return modern >= 36 ? modern + 14 : modern;
    if (arch_is_cdna_4_or_lower(GetParam())) {
      switch (modern) {
      case 0:
        return 0;
      case 11:
        return 2 | (4 << 4); // 16_UINT.
      case 20:
        return 4 | (4 << 4); // 32_UINT.
      case 42:
        return 10; // 8_8_8_8_UNORM.
      case 46:
        return 10 | (4 << 4); // 8_8_8_8_UINT.
      case 60:
        return 13 | (7 << 4); // 32_32_32_FLOAT.
      case 62:
        return 14 | (5 << 4); // 32_32_32_32_SINT.
      default:
        ADD_FAILURE() << "Missing test descriptor format " << modern;
        return 0;
      }
    }
    return modern;
  }
  void descriptor(uint32_t format, uint32_t stride, uint32_t selectors = identity) {
    format = encoded_format(format);
    const uint32_t format_word = arch_is_cdna_4_or_lower(GetParam())
                                     ? ((format & 15) << 15) | ((format >> 4) << 12)
                                     : format << 12;
    const uint32_t sb = wf->sgpr_alloc().base;
    cu->write_sgpr(sb + 4, 0x1000);
    cu->write_sgpr(sb + 5, stride << 16);
    cu->write_sgpr(sb + 6, 8);
    cu->write_sgpr(sb + 7, selectors | format_word | (1u << 28));
  }
  template <typename Load, typename Store, typename Machine>
  Instruction *execute(bool load, const Machine *raw) {
    if (load) {
      auto *inst = new Load(raw);
      inst->execute_impl(*wf);
      return inst;
    }
    auto *inst = new Store(raw);
    inst->execute_impl(*wf);
    return inst;
  }
  template <typename Raw, typename Machine, typename Load, typename Store, typename LoadD16,
            typename StoreD16>
  Instruction *legacy(bool load, bool d16, int format = -1, uint32_t soffset = 128) {
    Machine m{};
    m.srsrc = 1;
    m.soffset = soffset;
    m.idxen = 1;
    m.vdata = 8;
    if constexpr (requires { m.glc; })
      m.glc = m.slc = 1;
    else
      m.sc0 = m.sc1 = 1;
    if constexpr (requires { m.format; })
      m.format = format;
    if constexpr (requires { m.dfmt; }) {
      m.dfmt = format & 15;
      m.nfmt = format >> 4;
    }
    // Every legacy machine encoding is two dwords; each executor copies it.
    const auto *raw = reinterpret_cast<const Raw *>(&m);
    return d16 ? execute<LoadD16, StoreD16>(load, raw) : execute<Load, Store>(load, raw);
  }
  void issue_typed(uint32_t format, uint32_t soffset = 128) {
    Instruction *inst = nullptr;
    switch (GetParam()) {
    case ROCJITSU_CODE_ARCH_RDNA1:
      inst = legacy<rdna1::MachineInst, rdna1::MtbufMachineInst, rdna1::TbufferLoadFormatXyzwMtbuf,
                    rdna1::TbufferStoreFormatXMtbuf, rdna1::TbufferLoadFormatD16XyzMtbuf,
                    rdna1::TbufferStoreFormatD16XyzwMtbuf>(true, false, encoded_format(format),
                                                           soffset);
      break;
    case ROCJITSU_CODE_ARCH_RDNA2:
      inst = legacy<rdna2::MachineInst, rdna2::MtbufMachineInst, rdna2::TbufferLoadFormatXyzwMtbuf,
                    rdna2::TbufferStoreFormatXMtbuf, rdna2::TbufferLoadFormatD16XyzMtbuf,
                    rdna2::TbufferStoreFormatD16XyzwMtbuf>(true, false, encoded_format(format),
                                                           soffset);
      break;
    case ROCJITSU_CODE_ARCH_CDNA1:
      inst = legacy<cdna1::MachineInst, cdna1::MtbufMachineInst, cdna1::TbufferLoadFormatXyzwMtbuf,
                    cdna1::TbufferStoreFormatXMtbuf, cdna1::TbufferLoadFormatD16XyzMtbuf,
                    cdna1::TbufferStoreFormatD16XyzwMtbuf>(true, false, encoded_format(format),
                                                           soffset);
      break;
    case ROCJITSU_CODE_ARCH_CDNA2:
      inst = legacy<cdna2::MachineInst, cdna2::MtbufMachineInst, cdna2::TbufferLoadFormatXyzwMtbuf,
                    cdna2::TbufferStoreFormatXMtbuf, cdna2::TbufferLoadFormatD16XyzMtbuf,
                    cdna2::TbufferStoreFormatD16XyzwMtbuf>(true, false, encoded_format(format),
                                                           soffset);
      break;
    case ROCJITSU_CODE_ARCH_CDNA3:
      inst = legacy<cdna3::MachineInst, cdna3::MtbufMachineInst, cdna3::TbufferLoadFormatXyzwMtbuf,
                    cdna3::TbufferStoreFormatXMtbuf, cdna3::TbufferLoadFormatD16XyzMtbuf,
                    cdna3::TbufferStoreFormatD16XyzwMtbuf>(true, false, encoded_format(format),
                                                           soffset);
      break;
    case ROCJITSU_CODE_ARCH_CDNA4:
      inst = legacy<cdna4::MachineInst, cdna4::MtbufMachineInst, cdna4::TbufferLoadFormatXyzwMtbuf,
                    cdna4::TbufferStoreFormatXMtbuf, cdna4::TbufferLoadFormatD16XyzMtbuf,
                    cdna4::TbufferStoreFormatD16XyzwMtbuf>(true, false, encoded_format(format),
                                                           soffset);
      break;
    case ROCJITSU_CODE_ARCH_RDNA3:
      inst = legacy<rdna3::MachineInst, rdna3::MtbufMachineInst, rdna3::TbufferLoadFormatXyzwMtbuf,
                    rdna3::TbufferStoreFormatXMtbuf, rdna3::TbufferLoadD16FormatXyzMtbuf,
                    rdna3::TbufferStoreD16FormatXyzwMtbuf>(true, false, encoded_format(format),
                                                           soffset);
      break;
    case ROCJITSU_CODE_ARCH_RDNA3_5:
      inst = legacy<rdna3_5::MachineInst, rdna3_5::MtbufMachineInst,
                    rdna3_5::TbufferLoadFormatXyzwMtbuf, rdna3_5::TbufferStoreFormatXMtbuf,
                    rdna3_5::TbufferLoadD16FormatXyzMtbuf, rdna3_5::TbufferStoreD16FormatXyzwMtbuf>(
          true, false, encoded_format(format), soffset);
      break;
    case ROCJITSU_CODE_ARCH_RDNA4: {
      rdna4::VbufferMachineInst m{};
      m.rsrc = 4;
      m.soffset = rdna4::OPR_SREG_M0_NULL;
      m.idxen = 1;
      m.vdata = 8;
      m.scope = 3;
      m.format = format;
      inst = execute<rdna4::TbufferLoadFormatXyzwVbuffer, rdna4::TbufferStoreFormatXVbuffer>(
          true, reinterpret_cast<const rdna4::MachineInst *>(&m));
      break;
    }
    default:
      FAIL() << "No typed opcode selected";
    }
    amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
    ASSERT_EQ(pipeline.issue(inst, *wf), amdgpu::VmAccessOutcome::Complete);
  }
  void issue(bool load, bool d16 = false, uint32_t soffset = 128) {
    Instruction *inst;
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
      rdna4::VbufferMachineInst m{};
      m.rsrc = 4;
      m.soffset = rdna4::OPR_SREG_M0_NULL;
      m.idxen = 1;
      m.vdata = 8;
      m.scope = 3;
      const auto *raw = reinterpret_cast<const rdna4::MachineInst *>(&m);
      inst = d16 ? execute<rdna4::BufferLoadD16FormatXyzVbuffer,
                           rdna4::BufferStoreD16FormatXyzwVbuffer>(load, raw)
                 : execute<rdna4::BufferLoadFormatXyzwVbuffer, rdna4::BufferStoreFormatXVbuffer>(
                       load, raw);
    } else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3) {
      rdna3::MubufMachineInst m{};
      m.srsrc = 1;
      m.soffset = soffset;
      m.idxen = 1;
      m.vdata = 8;
      m.glc = m.slc = 1;
      const auto *raw = reinterpret_cast<const rdna3::MachineInst *>(&m);
      inst =
          d16 ? execute<rdna3::BufferLoadD16FormatXyzMubuf, rdna3::BufferStoreD16FormatXyzwMubuf>(
                    load, raw)
              : execute<rdna3::BufferLoadFormatXyzwMubuf, rdna3::BufferStoreFormatXMubuf>(load,
                                                                                          raw);
    } else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA1) {
      inst = legacy<rdna1::MachineInst, rdna1::MubufMachineInst, rdna1::BufferLoadFormatXyzwMubuf,
                    rdna1::BufferStoreFormatXMubuf, rdna1::BufferLoadFormatD16XyzMubuf,
                    rdna1::BufferStoreFormatD16XyzwMubuf>(load, d16, -1, soffset);
    } else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA2) {
      inst = legacy<rdna2::MachineInst, rdna2::MubufMachineInst, rdna2::BufferLoadFormatXyzwMubuf,
                    rdna2::BufferStoreFormatXMubuf, rdna2::BufferLoadFormatD16XyzMubuf,
                    rdna2::BufferStoreFormatD16XyzwMubuf>(load, d16, -1, soffset);
    } else if (GetParam() == ROCJITSU_CODE_ARCH_CDNA1) {
      inst = legacy<cdna1::MachineInst, cdna1::MubufMachineInst, cdna1::BufferLoadFormatXyzwMubuf,
                    cdna1::BufferStoreFormatXMubuf, cdna1::BufferLoadFormatD16XyzMubuf,
                    cdna1::BufferStoreFormatD16XyzwMubuf>(load, d16, -1, soffset);
    } else if (GetParam() == ROCJITSU_CODE_ARCH_CDNA2) {
      inst = legacy<cdna2::MachineInst, cdna2::MubufMachineInst, cdna2::BufferLoadFormatXyzwMubuf,
                    cdna2::BufferStoreFormatXMubuf, cdna2::BufferLoadFormatD16XyzMubuf,
                    cdna2::BufferStoreFormatD16XyzwMubuf>(load, d16, -1, soffset);
    } else if (GetParam() == ROCJITSU_CODE_ARCH_CDNA3) {
      inst = legacy<cdna3::MachineInst, cdna3::MubufMachineInst, cdna3::BufferLoadFormatXyzwMubuf,
                    cdna3::BufferStoreFormatXMubuf, cdna3::BufferLoadFormatD16XyzMubuf,
                    cdna3::BufferStoreFormatD16XyzwMubuf>(load, d16, -1, soffset);
    } else if (GetParam() == ROCJITSU_CODE_ARCH_CDNA4) {
      inst = legacy<cdna4::MachineInst, cdna4::MubufMachineInst, cdna4::BufferLoadFormatXyzwMubuf,
                    cdna4::BufferStoreFormatXMubuf, cdna4::BufferLoadFormatD16XyzMubuf,
                    cdna4::BufferStoreFormatD16XyzwMubuf>(load, d16, -1, soffset);
    } else {
      rdna3_5::MubufMachineInst m{};
      m.srsrc = 1;
      m.soffset = soffset;
      m.idxen = 1;
      m.vdata = 8;
      m.glc = m.slc = 1;
      const auto *raw = reinterpret_cast<const rdna3_5::MachineInst *>(&m);
      inst = d16 ? execute<rdna3_5::BufferLoadD16FormatXyzMubuf,
                           rdna3_5::BufferStoreD16FormatXyzwMubuf>(load, raw)
                 : execute<rdna3_5::BufferLoadFormatXyzwMubuf, rdna3_5::BufferStoreFormatXMubuf>(
                       load, raw);
    }
    ASSERT_NE(inst->data(), nullptr);
    amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
    ASSERT_EQ(pipeline.issue(inst, *wf), amdgpu::VmAccessOutcome::Complete);
  }
};

TEST(BufferFormatTest, Unorm8FirstDestinationSeesDecodeExceptions) {
  using Base = amdgpu::IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, rdna3::Isa>;
  class RecordingCu final : public Base {
  public:
    using Base::Base;
    int first_flags = -1;
    size_t first_chunks = 0;
    void write_vgpr(uint32_t reg, uint32_t lane, uint32_t value) override {
      if (first_flags == -1) {
        first_flags = std::fetestexcept(FE_ALL_EXCEPT);
        first_chunks = vgpr_file().materialized_chunk_count();
      }
      Base::write_vgpr(reg, lane, value);
    }
  };
  const amdgpu::fp_mode::detail::ScopedFenv restore_environment(0);
  amdgpu::GpuMemory memory("unorm_first_touch");
  amdgpu::L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_RDNA3;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 32;
  config.lds_size_kb = 64;
  RecordingCu cu("cu", config, &memory, &l2);
  auto *wf = cu.dispatch_wf(0, 0, 106, 32);
  ASSERT_NE(wf, nullptr);
  amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
  state.wf_size = 32;
  state.exec_mask = state.lane_mask = 1;
  state.elem_size = state.buffer_components = 4;
  state.buffer_selectors = identity;
  state.decoded_buffer_format = amdgpu::decode_buffer_format(42).value();
  state.image_sampling = true;
  state.dst_reg_base = wf->vgpr_alloc().base + 8;
  state.image_sample = std::make_unique<amdgpu::ImageSampleAccess>();
  auto &sample = *state.image_sample;
  sample.tap_count = sample.taps_per_filter = 4;
  sample.texels_per_tap = sample.filter_counts[0] = 1;
  sample.taps.resize(4);
  sample.filters.resize(1);
  for (auto &tap : sample.taps)
    tap.lane_mask = 1;
  sample.filters[0].fractions[0][0] = {0.0f, 0.0f};
  state.response_data.assign(4 * state.wf_size * state.elem_size, 1);
  ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
  amdgpu::complete_buffer_format_load(*wf, cu, state);
  EXPECT_EQ(cu.first_chunks, 0u);
  // Decoding 1/255 raises inexact before the first lazy allocation. The
  // completion's environment guard restores the caller only after all writes.
  EXPECT_EQ(cu.first_flags, FE_INEXACT);
  EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), 0);
  EXPECT_EQ(cu.read_vgpr(state.dst_reg_base, 0), 0x3b808081u);
  wf->halt();
}

TEST_P(BufferFormatExecutionTest, Unorm8FilteringMatchesWidenedAlphaReference) {
  // Widening A to UNORM16 forces the generic decoder while preserving every
  // UNORM8 input exactly: byte * 257 / 65535 == byte / 255. The filter still
  // uses eight-bit texel units because R remains eight bits.
  const amdgpu::fp_mode::detail::ScopedFenv restore_environment(0);
  uint32_t seed = 0x53a92017;
  const auto random = [&] { return seed = seed * 1664525u + 1013904223u; };
  for (uint32_t test = 0; test < 384; ++test) {
    SCOPED_TRACE(test);
    const uint32_t format_id = std::array{1u, 14u, 42u}[test % 3];
    const auto format = amdgpu::decode_buffer_format(format_id).value();
    const uint32_t channels = format.component_count();
    const uint32_t filters = std::array{1u, 3u, 16u}[(test / 3) % 3];
    const uint32_t sources = (test / 9) % 2 ? 3 : 1;
    const uint32_t levels = (test / 18) % 2 + 1;
    const uint32_t taps = filters * levels * 4 * sources;
    amdgpu::VectorMemState fast(amdgpu::GLOBAL_MEM), reference(amdgpu::GLOBAL_MEM);
    for (auto *state : {&fast, &reference}) {
      state->wf_size = 4;
      state->exec_mask = 0b1101;
      state->lane_mask = 0b0101; // Lane 3 completes with invalid texels.
      state->elem_size = channels;
      state->buffer_components = test % 4 + 1;
      state->buffer_d16 = test & 1;
      state->buffer_format = format_id;
      state->decoded_buffer_format = format;
      for (uint32_t c = 0; c < 4; ++c)
        state->buffer_selectors |= ((test / 4 + c) % 8) << (3 * c);
      state->image_sampling = true;
      state->dst_reg_base = wf->vgpr_alloc().base + 8;
      state->image_sample = std::make_unique<amdgpu::ImageSampleAccess>();
      auto &sample = *state->image_sample;
      sample.tap_count = taps;
      sample.texels_per_tap = sources;
      sample.taps_per_filter = levels * 4 * sources;
      sample.border_color = (test / 36) % 3;
      sample.taps.resize(taps);
      sample.filters.resize(filters);
      for (uint32_t lane = 0; lane < 4; ++lane) {
        sample.filter_counts[lane] = lane == 2 ? 1 : filters;
        sample.mip_fractions[lane] = test % 3 == 0 ? 0 : test % 3 == 1 ? 1 : 113.0f / 256;
      }
    }
    reference.decoded_buffer_format.widths[3] = 16;
    reference.elem_size += channels == 4 ? 1 : 2;
    for (uint32_t filter = 0; filter < filters; ++filter)
      for (uint32_t lane = 0; lane < 4; ++lane)
        for (uint32_t level = 0; level < levels; ++level) {
          auto &f = fast.image_sample->filters[filter];
          f.fractions[lane][level] = {(random() & 255) / 256.0f, (random() & 255) / 256.0f};
          f.cube_corners[lane][level] = sources == 3 ? random() & 15 : 0;
          reference.image_sample->filters[filter] = f;
        }
    fast.response_data.resize(taps * 4 * fast.elem_size);
    reference.response_data.resize(taps * 4 * reference.elem_size);
    for (uint32_t tap = 0; tap < taps; ++tap) {
      const uint64_t lane_mask = random() & 15;
      fast.image_sample->taps[tap].lane_mask = lane_mask;
      reference.image_sample->taps[tap].lane_mask = lane_mask;
      for (uint32_t lane = 0; lane < 4; ++lane) {
        const uint32_t original = (tap * 4 + lane) * fast.elem_size;
        const uint32_t widened = (tap * 4 + lane) * reference.elem_size;
        for (uint32_t c = 0; c < channels; ++c) {
          const uint8_t value = random() >> 24;
          fast.response_data[original + c] = value;
          reference.response_data[widened + c] = value;
          if (c == 3)
            reference.response_data[widened + c + 1] = value;
        }
      }
    }
    const uint32_t registers =
        fast.buffer_d16 ? (fast.buffer_components + 1) / 2 : fast.buffer_components;
    const int mode = std::array{FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}[test % 4];
    ASSERT_EQ(std::fesetround(mode), 0);
    ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    ASSERT_EQ(std::feraiseexcept(FE_INVALID), 0);
    std::array<std::array<uint32_t, 4>, 4> expected{};
    for (const auto *state : {&reference, &fast}) {
      for (uint32_t reg = 0; reg < registers; ++reg)
        for (uint32_t lane = 0; lane < 4; ++lane)
          cu->write_vgpr(state->dst_reg_base + reg, lane, 0xa57ec031);
      amdgpu::complete_buffer_format_load(*wf, *cu, *state);
      EXPECT_EQ(std::fegetround(), mode);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID);
      for (uint32_t reg = 0; reg < registers; ++reg)
        for (uint32_t lane = 0; lane < 4; ++lane) {
          const uint32_t value = cu->read_vgpr(state->dst_reg_base + reg, lane);
          if (state == &reference)
            expected[reg][lane] = value;
          else
            EXPECT_EQ(value, expected[reg][lane]) << "reg=" << reg << " lane=" << lane;
        }
    }
  }
}

TEST_P(BufferFormatExecutionTest, Unorm8SingleFilterKeepsQ8BoundariesAndFallback) {
  const amdgpu::fp_mode::detail::ScopedFenv restore_environment(0);
  // Include adjacent FP32 values around the Q8 lattice. Non-Q8 descriptors
  // must retain the general filter, including both mips' separate rounding.
  constexpr std::array<uint32_t, 16> fraction_bits{0,          0x80000000, 0x3a800000, 0x3b7fffff,
                                                   0x3b800000, 0x3b800001, 0x3c400000, 0x3c800000,
                                                   0x3cc00000, 0x3effffff, 0x3f000000, 0x3f000001,
                                                   0x3f7effff, 0x3f7f0000, 0x3f7fffff, 0x3f800000};
  for (uint32_t test = 0; test < fraction_bits.size() * 8; ++test) {
    SCOPED_TRACE(test);
    const uint32_t levels = test < 2 ? 2 : 1 + (test / fraction_bits.size()) % 2;
    const uint32_t taps = levels * 4;
    amdgpu::VectorMemState fast(amdgpu::GLOBAL_MEM), reference(amdgpu::GLOBAL_MEM);
    for (auto *state : {&fast, &reference}) {
      state->wf_size = 4;
      state->exec_mask = test >= 2 && (test & 1) ? 0b1010 : 0b1101;
      state->lane_mask = test >= 2 && (test & 1) ? 0b1010 : 0b0101;
      state->elem_size = 4;
      state->buffer_components = test < 2 ? 4 : 1 + (test / 4) % 4;
      state->buffer_d16 = (test / (fraction_bits.size() * 2)) & 1;
      state->d16_hi = state->buffer_d16 && (test & 2);
      state->lds_dst = (test / (fraction_bits.size() * 4)) & 1;
      state->lds_base = 256;
      state->buffer_selectors = identity;
      state->decoded_buffer_format = amdgpu::decode_buffer_format(42).value();
      state->image_sampling = true;
      state->dst_reg_base = wf->vgpr_alloc().base + 8;
      state->image_sample = std::make_unique<amdgpu::ImageSampleAccess>();
      auto &sample = *state->image_sample;
      sample.tap_count = sample.taps_per_filter = taps;
      sample.texels_per_tap = 1;
      sample.border_color = test % 3;
      sample.taps.resize(taps);
      sample.filters.resize(1);
      for (uint32_t lane = 0; lane < 4; ++lane) {
        sample.filter_counts[lane] = 1;
        sample.mip_fractions[lane] =
            std::bit_cast<float>(fraction_bits[(test + 3 * lane) % fraction_bits.size()]);
        for (uint32_t level = 0; level < levels; ++level)
          sample.filters[0].fractions[lane][level] = {
              std::bit_cast<float>(fraction_bits[(test + level) % fraction_bits.size()]),
              std::bit_cast<float>(fraction_bits[(test + lane + 5) % fraction_bits.size()])};
      }
      if (test < 2) {
        // Q16 values (4,5) with mip=1/256 round to Q19 (32,0),
        // then Q13 zero. Rounding their combined product would give one.
        // (12,11) rounds through (96,0), then Q13 two, not one.
        sample.mip_fractions[0] = 1.0f / 256;
        sample.filters[0].fractions[0][0] = {1.0f / 256, (test ? 12.0f : 4.0f) / 256};
        sample.filters[0].fractions[0][1] = {1.0f / 256, (test ? 11.0f : 5.0f) / 256};
      }
    }
    // Independent existing path: widening alpha prevents integer-texel
    // admission while byte * 257 / 65535 preserves the original UNORM8 value.
    reference.decoded_buffer_format.widths[3] = 16;
    reference.elem_size = 5;
    fast.response_data.resize(taps * 4 * fast.elem_size);
    reference.response_data.resize(taps * 4 * reference.elem_size);
    for (uint32_t tap = 0; tap < taps; ++tap) {
      const uint64_t lanes = tap % 3 == 1 ? 0b1001 : 0b1101;
      fast.image_sample->taps[tap].lane_mask = lanes;
      reference.image_sample->taps[tap].lane_mask = lanes;
      for (uint32_t lane = 0; lane < 4; ++lane)
        for (uint32_t c = 0; c < 4; ++c) {
          const uint8_t value = c == 0   ? 0
                                : c == 1 ? 255
                                : c == 2 ? (tap == 3 || tap == 7)
                                         : test * 19 + tap * 37;
          fast.response_data[(tap * 4 + lane) * 4 + c] = value;
          reference.response_data[(tap * 4 + lane) * 5 + c] = value;
          if (c == 3)
            reference.response_data[(tap * 4 + lane) * 5 + c + 1] = value;
        }
    }
    const uint32_t registers =
        fast.buffer_d16 ? (fast.buffer_components + 1) / 2 : fast.buffer_components;
    for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
      SCOPED_TRACE(mode);
      std::array<std::array<uint32_t, 4>, 4> expected{};
      for (const auto *state : {&reference, &fast}) {
        for (uint32_t reg = 0; reg < registers; ++reg)
          for (uint32_t lane = 0; lane < 4; ++lane) {
            constexpr uint32_t sentinel = 0xa57ec031;
            cu->write_vgpr(state->dst_reg_base + reg, lane, sentinel);
            wf->lds().write(state->lds_base + (lane * registers + reg) * 4,
                            reinterpret_cast<const uint8_t *>(&sentinel), 4);
          }
        ASSERT_EQ(std::fesetround(mode), 0);
        ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
        ASSERT_EQ(std::feraiseexcept(FE_INVALID | FE_DIVBYZERO), 0);
        errno = EILSEQ;
        amdgpu::complete_buffer_format_load(*wf, *cu, *state);
        const int saved_errno = errno;
        EXPECT_EQ(std::fegetround(), mode);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID | FE_DIVBYZERO);
        EXPECT_EQ(saved_errno, EILSEQ);
        for (uint32_t reg = 0; reg < registers; ++reg)
          for (uint32_t lane = 0; lane < 4; ++lane) {
            uint32_t value = 0;
            if (state->lds_dst)
              static_cast<const amdgpu::Lds &>(wf->lds()).read(
                  state->lds_base + (lane * registers + reg) * 4,
                  reinterpret_cast<uint8_t *>(&value), 4);
            else
              value = cu->read_vgpr(state->dst_reg_base + reg, lane);
            if (state == &reference)
              expected[reg][lane] = value;
            else
              EXPECT_EQ(value, expected[reg][lane]) << "reg=" << reg << " lane=" << lane;
          }
      }
    }
  }
}

TEST_P(BufferFormatExecutionTest, FixedUnormCompletionMatchesQualifiedScalingBoundaries) {
  if (GetParam() != ROCJITSU_CODE_ARCH_RDNA3 && GetParam() != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      GetParam() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Fixed texture filtering is qualified on RDNA3/4";
  const amdgpu::fp_mode::detail::ScopedFenv restore_environment(0);
  // Retain the previous scaling and final conversion as the reference. A
  // direct bit encoder must match zero, exponent boundaries and carries.
  const auto round_even = [](double value) {
    const double lo = std::floor(value), fraction = value - lo;
    return lo + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lo, 2.0) != 0));
  };
  const auto normalize = [&](double filtered, uint32_t width, uint32_t count) {
    const uint64_t rounded =
        count > 1 ? static_cast<uint64_t>(std::floor(std::ldexp(filtered, 13) + 0.5)) /
                        std::bit_floor(count)
                  : static_cast<uint64_t>(round_even(std::ldexp(filtered, 13)));
    const uint64_t numerator = rounded << (34 - 13 - width);
    uint64_t normalized = 0;
    for (uint32_t offset = 0; offset < 34; offset += width)
      normalized += numerator >> offset;
    const uint32_t significant = std::bit_width(normalized);
    const uint32_t shift = significant > 24 ? significant - 24 : 0;
    if (shift)
      normalized = (normalized + (uint64_t{1} << (shift - 1))) >> shift;
    return bits(static_cast<float>(
        std::ldexp(static_cast<double>(normalized), static_cast<int>(shift) - 34)));
  };
  constexpr std::array fractions{0u, 1u, 3u, 4u, 5u, 127u, 128u, 129u, 255u, 256u};
  for (const uint32_t format_id : {42u, 36u}) {
    const uint32_t width = format_id == 42 ? 8 : 10, maximum = (1u << width) - 1;
    for (uint32_t test = 0; test < 80; ++test) {
      SCOPED_TRACE(format_id);
      SCOPED_TRACE(test);
      const bool q13_boundary = test >= 16 && test < 24;
      const uint32_t count = q13_boundary ? 1 : std::array{1u, 3u, 16u}[test % 3];
      const uint32_t levels = q13_boundary ? 1 : 1 + (test / 3) % 2;
      const uint32_t sources = q13_boundary ? 1 : (test / 6) % 2 ? 3 : 1;
      const uint32_t taps = count * levels * 4 * sources;
      amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
      state.wf_size = 2;
      state.exec_mask = state.lane_mask = 1;
      state.elem_size = 4;
      state.buffer_components = 4;
      state.buffer_selectors = identity;
      state.decoded_buffer_format = amdgpu::decode_buffer_format(format_id).value();
      state.image_sampling = true;
      state.dst_reg_base = wf->vgpr_alloc().base + 8;
      state.image_sample = std::make_unique<amdgpu::ImageSampleAccess>();
      auto &sample = *state.image_sample;
      sample.tap_count = taps;
      sample.taps_per_filter = levels * 4 * sources;
      sample.texels_per_tap = sources;
      sample.filter_counts[0] = count;
      sample.mip_fractions[0] = fractions[(test / 2) % fractions.size()] / 256.0f;
      sample.filters.resize(count);
      sample.taps.resize(taps);
      state.response_data.resize(taps * state.wf_size * state.elem_size);
      std::vector<std::array<double, 4>> texels(taps);
      for (uint32_t tap = 0; tap < taps; ++tap) {
        sample.taps[tap].lane_mask = 1;
        const uint32_t raw = test < 2    ? test * maximum
                             : test < 12 ? std::min(maximum, 1u << (test - 2))
                             : test < 32 ? uint32_t(tap % (4 * sources) == 3 * sources)
                                         : (test * 131 + tap * 37) & maximum;
        const std::array channels{raw, raw, raw, width == 10 ? raw & 3 : raw};
        uint32_t packed = 0, offset = 0;
        for (uint32_t c = 0; c < 4; ++c) {
          const uint32_t channel_width = state.decoded_buffer_format.widths[c];
          packed |= channels[c] << offset;
          offset += channel_width;
          // Keep the old unpack-to-FP32 then recover-texel conversion,
          // including expansion of packed two-bit alpha to ten-bit units.
          texels[tap][c] =
              round_even(double(float(channels[c]) / ((1u << channel_width) - 1)) * maximum);
        }
        for (uint32_t byte = 0; byte < 4; ++byte)
          state.response_data[tap * state.wf_size * 4 + byte] = packed >> (8 * byte);
      }
      for (uint32_t filter = 0; filter < count; ++filter)
        for (uint32_t level = 0; level < levels; ++level) {
          auto &f = sample.filters[filter];
          f.fractions[0][level] = {fractions[(test + filter) % fractions.size()] / 256.0f,
                                   fractions[(test / 2 + level) % fractions.size()] / 256.0f};
          if (q13_boundary) {
            // The lone unit texel contributes Y/8 in Q13: both even/odd
            // midpoint ties and neighboring representable fractions.
            constexpr std::array y{3u, 4u, 5u, 11u, 12u, 13u, 19u, 20u};
            f.fractions[0][level] = {1.0f / 256, y[test - 16] / 256.0f};
          }
          f.cube_corners[0][level] = sources == 3 ? (test + filter + level) & 15 : 0;
        }
      std::array<uint32_t, 4> expected{};
      for (uint32_t c = 0; c < 4; ++c) {
        double sum = 0;
        for (uint32_t filter = 0; filter < count; ++filter) {
          std::array<double, 2> filtered{};
          for (uint32_t level = 0; level < levels; ++level) {
            const auto &f = sample.filters[filter];
            const double x = f.fractions[0][level][0], y = f.fractions[0][level][1];
            const std::array weights{(1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y};
            for (uint32_t tap = 0; tap < 4; ++tap) {
              const uint32_t first = filter * sample.taps_per_filter + (level * 4 + tap) * sources;
              double channel = texels[first][c];
              if (f.cube_corners[0][level] & (1u << tap))
                channel = (channel * 21846 + texels[first + 1][c] * 21845 +
                           texels[first + 2][c] * 21845) /
                          65536;
              filtered[level] += channel * weights[tap];
            }
          }
          double value = filtered[0];
          if (levels == 2) {
            const double fraction = sample.mip_fractions[0];
            value = (round_even(std::ldexp(value * (1 - fraction), 19)) +
                     round_even(std::ldexp(filtered[1] * fraction, 19))) /
                    std::ldexp(1.0, 19);
          }
          if (count > 1) {
            const uint32_t denominator = 128 * std::bit_ceil(count), residual = denominator % count;
            const uint32_t first = (count - residual) / 2;
            const uint32_t weight =
                denominator / count + (filter >= first && filter < first + residual);
            value = round_even(std::ldexp(
                        value * (double(weight) / denominator) * std::bit_floor(count), 19)) /
                    std::ldexp(1.0, 19);
          }
          sum += value;
        }
        expected[c] = normalize(sum, width, count);
      }
      for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        ASSERT_EQ(std::fesetround(mode), 0);
        ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
        ASSERT_EQ(std::feraiseexcept(FE_INVALID | FE_DIVBYZERO), 0);
        for (uint32_t c = 0; c < 4; ++c)
          cu->write_vgpr(state.dst_reg_base + c, 1, 0xa57ec031);
        amdgpu::complete_buffer_format_load(*wf, *cu, state);
        EXPECT_EQ(std::fegetround(), mode);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID | FE_DIVBYZERO);
        for (uint32_t c = 0; c < 4; ++c) {
          EXPECT_EQ(cu->read_vgpr(state.dst_reg_base + c, 0), expected[c]) << "channel=" << c;
          EXPECT_EQ(cu->read_vgpr(state.dst_reg_base + c, 1), 0xa57ec031);
        }
      }
      ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    }
  }
}

TEST_P(BufferFormatExecutionTest, NormalizedLoadsIgnoreAndRestoreHostRoundingMode) {
  const amdgpu::fp_mode::detail::ScopedFenv restore_environment(0);
  descriptor(42, 4);
  wf->set_exec(1);
  memory.write32(0x1000, 0x7fff8000);
  auto decoder = Decoder::create(GetParam());
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(mode), 0);
    ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    ASSERT_EQ(std::feraiseexcept(FE_INVALID), 0);
    std::unique_ptr<Instruction> inst;
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
      const auto words =
          rdna4::build_vbuffer(rdna4::kBufferLoadFormatXyzwVbuffer,
                               {.soffset = rdna4::OPR_SREG_M0_NULL, .vdata = 8, .rsrc = 4});
      inst.reset(decode_valid(*decoder, words.data()));
    } else {
      const std::array<uint32_t, 2> words{0xe00c0000u, 0x80010800u};
      inst.reset(decode_valid(*decoder, words.data()));
    }
    ASSERT_NE(inst, nullptr);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(std::fegetround(), mode);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID);
    const std::array<uint32_t, 4> expected{0, 0x3f008081, 0x3f800000, 0x3efefeff};
    for (uint32_t reg = 0; reg < 4; ++reg)
      EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + 8 + reg, 0), expected[reg]);
  }
}

TEST_P(BufferFormatExecutionTest, RdnaTransfersSwizzleEveryComponentAcrossUnitBoundaries) {
  if (GetParam() != ROCJITSU_CODE_ARCH_RDNA3 && GetParam() != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      GetParam() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "RDNA3+ swizzle geometry";
  wf->set_exec(1);
  auto decoder = Decoder::create(GetParam());
  const auto sb = wf->sgpr_alloc().base, vb = wf->vgpr_alloc().base;
  amdgpu::GpuVm vm;
  auto physical = std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory);
  const auto address_space = vm.register_address_space(
      0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(), physical);
  ASSERT_TRUE(address_space);
  for (bool translated : {false, true}) {
    cu->set_gpu_vm(translated ? &vm : nullptr);
    wf->set_address_space(translated ? address_space : amdgpu::AddressSpaceHandle{});
    for (uint32_t unit : {4u, 16u}) {
      const uint32_t stride = unit * 2, offset = unit - 4;
      constexpr uint32_t base = 0x1000, soffset = 3;
      // Index 1, group size 8, and two records. SOFFSET stays outside swizzling.
      cu->write_sgpr(sb, soffset);
      cu->write_sgpr(sb + 4, base);
      cu->write_sgpr(sb + 5, (stride << 16) | ((unit == 4 ? 1u : 3u) << 30));
      cu->write_sgpr(sb + 6, 2);
      cu->write_sgpr(sb + 7, identity | (48u << 12) | (1u << 28));
      cu->write_vgpr(vb, 0, 1);
      const uint64_t first = base + unit + offset + soffset;
      const uint64_t second = base + unit + unit * 8 + soffset;
      for (bool typed : {false, true}) {
        for (bool load : {true, false}) {
          SCOPED_TRACE(testing::Message() << "translated=" << translated << " unit=" << unit
                                          << " typed=" << typed << " load=" << load);
          cu->l1_vector().invalidate_all();
          l2.invalidate_all();
          memory.write32(first, load ? 0x11223344 : 0);
          memory.write32(second, load ? 0x55667788 : 0);
          memory.write32(first + 4, 0xdeadbeef);
          cu->write_vgpr(vb + 8, 0, load ? 0 : 0x11223344);
          cu->write_vgpr(vb + 9, 0, load ? 0 : 0x55667788);
          std::unique_ptr<Instruction> inst;
          if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
            const uint16_t op =
                typed ? (load ? rdna4::kTbufferLoadFormatXyVbuffer
                              : rdna4::kTbufferStoreFormatXyVbuffer)
                      : (load ? rdna4::kBufferLoadB64Vbuffer : rdna4::kBufferStoreB64Vbuffer);
            const auto words = rdna4::build_vbuffer(
                op,
                {.soffset = 0, .vdata = 8, .rsrc = 4, .format = 48, .idxen = 1, .ioffset = offset});
            inst.reset(decode_valid(*decoder, words.data()));
          } else {
            const auto words = typed ? rdna3::build_mtbuf(load ? rdna3::kTbufferLoadFormatXyMtbuf
                                                               : rdna3::kTbufferStoreFormatXyMtbuf,
                                                          {.offset = static_cast<uint16_t>(offset),
                                                           .format = 48,
                                                           .vdata = 8,
                                                           .srsrc = 1,
                                                           .idxen = 1})
                                     : rdna3::build_mubuf(load ? rdna3::kBufferLoadB64Mubuf
                                                               : rdna3::kBufferStoreB64Mubuf,
                                                          {.offset = static_cast<uint16_t>(offset),
                                                           .vdata = 8,
                                                           .srsrc = 1,
                                                           .idxen = 1});
            inst.reset(decode_valid(*decoder, words.data()));
          }
          ASSERT_NE(inst, nullptr);
          ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
          const auto &state = *inst->data_as<amdgpu::VectorMemState>();
          ASSERT_EQ(state.lane_mask, 1u);
          ASSERT_EQ(state.elem_size * state.num_elems, 8u);
          ASSERT_EQ(state.per_lane_addr[0], first);
          ASSERT_EQ(state.scratch_swizzle_unit, unit);
          ASSERT_EQ(state.scratch_addr_base_offset, soffset);
          amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
          ASSERT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
          cu->l1_vector().flush_all();
          l2.flush_all();
          EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0x11223344u);
          EXPECT_EQ(cu->read_vgpr(vb + 9, 0), 0x55667788u);
          EXPECT_EQ(memory.read32(first), 0x11223344u);
          EXPECT_EQ(memory.read32(second), 0x55667788u);
          EXPECT_EQ(memory.read32(first + 4), 0xdeadbeefu);
        }
      }
    }
  }
  wf->set_address_space({});
  cu->set_gpu_vm(nullptr);
}

TEST_P(BufferFormatExecutionTest, UnsupportedDescriptorFailsBeforeIssuingMemory) {
  if (GetParam() != ROCJITSU_CODE_ARCH_RDNA2 && GetParam() != ROCJITSU_CODE_ARCH_CDNA3)
    GTEST_SKIP() << "Reserved formats differ by architecture";
  descriptor(11, 4);
  cu->write_sgpr(
      wf->sgpr_alloc().base + 7,
      identity | (GetParam() == ROCJITSU_CODE_ARCH_RDNA2 ? 34u << 12 : (1u << 15) | (6u << 12)));
  auto decoder = Decoder::create(GetParam());
  // buffer_load_format_xyzw v[8:11], off, s[4:7], 0.
  const std::array<uint32_t, 2> words{0xe00c0000u, 0x80010800u};
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  cu->write_vgpr(wf->vgpr_alloc().base + 8, 0, 0xdeadbeef);
  EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).failed());
  EXPECT_EQ(inst->data(), nullptr);
  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + 8, 0), 0xdeadbeef);
}

TEST_P(BufferFormatExecutionTest, InlineScalarOffsetsUseTheirUnsignedBitsForTypedAndUntypedLoads) {
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "VBUFFER uses a seven-bit register selector instead of inline SOFFSET";
  const uint32_t lane = wf->wf_size() - 1;
  wf->set_exec(uint64_t{1} << lane);
  const uint32_t sb = wf->sgpr_alloc().base, vb = wf->vgpr_alloc().base;
  constexpr uint64_t address = 0x200001000ull;
  constexpr uint32_t expected = 0x92345678;
  memory.write32(address, expected);
  cu->write_vgpr(vb, lane, 0);
  constexpr uint32_t float_bits[] = {0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000, 0x40000000,
                                     0xc0000000, 0x40800000, 0xc0800000, 0x3e22f983};
  for (uint32_t selector = 128; selector <= 255; ++selector) {
    // Some older ISAs also permit dynamic special sources here. This regression
    // covers static constants and universally invalid encodings only.
    if ((selector >= 235 && selector <= 239) || (selector >= 251 && selector <= 253))
      continue;
    SCOPED_TRACE(selector);
    const bool valid = selector <= 208 || (selector >= 240 && selector <= 248);
    const uint32_t offset = selector <= 192   ? selector - 128
                            : selector <= 208 ? 192u - selector
                            : valid           ? float_bits[selector - 240]
                                              : 0;
    descriptor(20, 4); // 32_UINT, index-checked so large unsigned offsets remain in range.
    const uint64_t base = address - offset;
    cu->write_sgpr(sb + 4, uint32_t(base));
    cu->write_sgpr(sb + 5, uint32_t(base >> 32) | (4u << 16));
    for (bool typed : {false, true}) {
      SCOPED_TRACE(typed);
      cu->write_vgpr(vb + 8, lane, 0xdeadbeef);
      cu->write_vgpr(vb + 8, 0, 0xabcdef01);
      if (typed)
        issue_typed(20, selector);
      else
        issue(true, false, selector);
      EXPECT_EQ(cu->read_vgpr(vb + 8, lane), valid ? expected : 0);
      EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0xabcdef01);
    }
  }
}

TEST_P(BufferFormatExecutionTest, TypedFormatOverridesDescriptorAndExecIncludesTheHighestLane) {
  descriptor(11, 4); // Deliberately incompatible 16_UINT descriptor.
  const uint32_t lane = wf->wf_size() - 1;
  wf->set_exec(uint64_t{1} << lane);
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb, lane, 1);
  cu->write_vgpr(vb + 8, 0, 0xdeadbeef);
  memory.write32(0x1004, 0x7fff8000);
  issue_typed(42); // Explicit 8_8_8_8_UNORM, with identity selectors.
  EXPECT_EQ(cu->read_vgpr(vb + 8, lane), bits(0));
  EXPECT_EQ(cu->read_vgpr(vb + 9, lane), bits(128.0f / 255));
  EXPECT_EQ(cu->read_vgpr(vb + 10, lane), bits(1));
  EXPECT_EQ(cu->read_vgpr(vb + 11, lane), bits(127.0f / 255));
  EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0xdeadbeef);
}

TEST_P(BufferFormatExecutionTest, TypedFormatDoesNotBindAnInvalidResource) {
  descriptor(0, 4);
  memory.write32(0x1000, 0xffffffff);
  cu->write_vgpr(wf->vgpr_alloc().base, 0, 0);
  issue_typed(42);
  for (uint32_t i = 0; i < 4; ++i)
    EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + 8 + i, 0), 0);
}

TEST_P(BufferFormatExecutionTest, Rdna4TypedOpcodesConvertEveryLoadAndStoreWidth) {
  if (GetParam() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Other targets already expose these typed opcodes";
  descriptor(11, 4); // Explicit instruction format overrides 16_UINT.
  const uint32_t vb = wf->vgpr_alloc().base;
  const uint32_t lane = wf->wf_size() - 1;
  wf->set_exec(uint64_t{1} << lane);
  cu->write_vgpr(vb, lane, 1);
  auto decoder = Decoder::create(GetParam());
  constexpr uint16_t loads[] = {
      rdna4::kTbufferLoadFormatXVbuffer, rdna4::kTbufferLoadFormatXyVbuffer,
      rdna4::kTbufferLoadFormatXyzVbuffer, rdna4::kTbufferLoadFormatXyzwVbuffer};
  constexpr uint16_t stores[] = {
      rdna4::kTbufferStoreFormatXVbuffer, rdna4::kTbufferStoreFormatXyVbuffer,
      rdna4::kTbufferStoreFormatXyzVbuffer, rdna4::kTbufferStoreFormatXyzwVbuffer};
  for (uint32_t width = 1; width <= 4; ++width) {
    for (bool load : {false, true}) {
      SCOPED_TRACE(testing::Message() << "width=" << width << " load=" << load);
      const uint32_t values[] = {bits(0), bits(1), bits(0.5f), bits(0.25f)};
      for (uint32_t i = 0; i < 5; ++i) {
        cu->write_vgpr(vb + 8 + i, lane, i < width && !load ? values[i] : 0xdeadbeef);
        cu->write_vgpr(vb + 8 + i, 0, 0xdeadbeef);
      }
      const auto words = rdna4::build_vbuffer(load ? loads[width - 1] : stores[width - 1],
                                              {.soffset = rdna4::OPR_SREG_M0_NULL,
                                               .vdata = 8,
                                               .rsrc = 4,
                                               .scope = 3,
                                               .format = 42,
                                               .idxen = 1});
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
      ASSERT_NE(inst, nullptr);
      ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
      ASSERT_NE(inst->data(), nullptr);
      EXPECT_EQ(inst->data_as<amdgpu::VectorMemState>()->wait_counter_type,
                load ? amdgpu::WaitCounterType::LOADCNT : amdgpu::WaitCounterType::STORECNT);
      amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
      pipeline.issue(inst.release(), *wf);
      cu->l1_vector().flush_all();
      l2.flush_all();
      const uint32_t packed[] = {0, 0xff00, 0x80ff00, 0x4080ff00};
      EXPECT_EQ(memory.read32(0x1004), packed[width - 1]);
      if (load) {
        constexpr uint8_t bytes[] = {0, 255, 128, 64};
        for (uint32_t i = 0; i < width; ++i)
          EXPECT_EQ(cu->read_vgpr(vb + 8 + i, lane), bits(bytes[i] / 255.0f));
      }
      EXPECT_EQ(cu->read_vgpr(vb + 8 + width, lane), 0xdeadbeef);
      EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0xdeadbeef);
    }
  }
}

TEST_P(BufferFormatExecutionTest, RdnaBoundsClampRawDwordsButDropFormattedTransfersTogether) {
  if (!amdgpu::addr_calc::uses_rdna_buffer_range_check(GetParam()))
    GTEST_SKIP() << "RDNA1/2 and CDNA use their own range rules";
  const uint32_t sb = wf->sgpr_alloc().base, vb = wf->vgpr_alloc().base;
  const uint32_t last = wf->wf_size() - 1;
  wf->set_exec(uint64_t{1} << last);
  struct Case {
    uint32_t mode, stride, records, scalar, offset;
    std::array<uint32_t, 4> values;
  };
  // Matched on physical gfx1100 and gfx1201, including the partially valid final DWORD.
  constexpr Case cases[] = {
      {0, 8, 7, 0, 0, {1, 2, 0, 0}}, {0, 8, 7, 4, 4, {3, 0, 0, 0}},  {1, 8, 7, 4, 4, {3, 4, 5, 6}},
      {2, 8, 0, 0, 0, {0, 0, 0, 0}}, {2, 8, 1, 0, 12, {4, 5, 6, 7}}, {3, 0, 17, 4, 0, {2, 3, 4, 0}},
  };
  for (const auto &c : cases) {
    descriptor(20, c.stride);
    cu->write_sgpr(sb + 6, c.records);
    cu->write_sgpr(sb + 7, identity | (20u << 12) | (c.mode << 28));
    cu->write_sgpr(sb + 12, c.scalar);
    cu->write_vgpr(vb, last, c.offset);
    for (uint32_t i = 0; i < 16; ++i)
      memory.write32(0x1000 + i * 4, i + 1);
    for (uint32_t i = 0; i < 4; ++i)
      cu->write_vgpr(vb + 8 + i, 0, 0xdeadbeef);
    Instruction *inst;
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
      rdna4::VbufferMachineInst m{};
      m.rsrc = 4;
      m.soffset = 12;
      m.offen = 1;
      m.vdata = 8;
      m.scope = 3;
      inst = execute<rdna4::BufferLoadB128Vbuffer, rdna4::BufferStoreB128Vbuffer>(
          true, reinterpret_cast<const rdna4::MachineInst *>(&m));
    } else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3) {
      rdna3::MubufMachineInst m{};
      m.srsrc = 1;
      m.soffset = 12;
      m.offen = 1;
      m.vdata = 8;
      m.glc = m.slc = 1;
      inst = execute<rdna3::BufferLoadB128Mubuf, rdna3::BufferStoreB128Mubuf>(
          true, reinterpret_cast<const rdna3::MachineInst *>(&m));
    } else {
      rdna3_5::MubufMachineInst m{};
      m.srsrc = 1;
      m.soffset = 12;
      m.offen = 1;
      m.vdata = 8;
      m.glc = m.slc = 1;
      inst = execute<rdna3_5::BufferLoadB128Mubuf, rdna3_5::BufferStoreB128Mubuf>(
          true, reinterpret_cast<const rdna3_5::MachineInst *>(&m));
    }
    amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
    ASSERT_EQ(pipeline.issue(inst, *wf), amdgpu::VmAccessOutcome::Complete);
    for (uint32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(cu->read_vgpr(vb + 8 + i, last), c.values[i]);
      EXPECT_EQ(cu->read_vgpr(vb + 8 + i, 0), 0xdeadbeef);
    }
  }
  descriptor(60, 16); // 12-byte formatted transfer, only 8 bytes in range.
  cu->write_sgpr(sb + 6, 8);
  cu->write_sgpr(sb + 7, identity | (60u << 12) | (3u << 28));
  cu->write_vgpr(vb, last, 0);
  issue(true);
  for (uint32_t i = 0; i < 4; ++i)
    EXPECT_EQ(cu->read_vgpr(vb + 8 + i, last), 0);
  issue(false);
  cu->l1_vector().flush_all();
  l2.flush_all();
  EXPECT_EQ(memory.read32(0x1000), 1);
  EXPECT_EQ(memory.read32(0x1008), 3);
}

TEST_P(BufferFormatExecutionTest, RdnaSwizzledMode3BoundsIncludeScalarOffset) {
  if (!amdgpu::addr_calc::uses_rdna_buffer_range_check(GetParam()))
    GTEST_SKIP() << "RDNA3+ bounds modes";
  struct Case {
    uint32_t records, scalar, index, offset, valid_modes;
  };
  // RDNA3 ISA Table 46 reduces NUM_RECORDS by SOFFSET only in mode 3.
  // Include the last valid index, equality, saturation, and the stride boundary.
  constexpr Case cases[] = {
      {2, 0, 1, 0, 0xf}, {2, 1, 0, 0, 0xf}, {2, 1, 1, 0, 0x7},
      {2, 2, 0, 0, 0x7}, {2, 3, 0, 0, 0x7}, {2, UINT32_MAX, 0, 0, 0x7},
      {0, 0, 0, 0, 0x0}, {2, 0, 0, 4, 0xf}, {2, 0, 0, 8, 0x6},
  };
  const auto sb = wf->sgpr_alloc().base, vb = wf->vgpr_alloc().base;
  const uint32_t lane = wf->wf_size() - 1;
  wf->set_exec(uint64_t{1} << lane);
  auto decoder = Decoder::create(GetParam());
  amdgpu::GpuVm vm;
  auto physical = std::make_shared<test::CountingGpuMemory>(memory);
  const auto address_space = vm.register_address_space(
      0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(), physical);
  ASSERT_TRUE(address_space);
  cu->set_gpu_vm(&vm);
  wf->set_address_space(address_space);
  for (const auto &c : cases) {
    cu->write_sgpr(sb + 4, 0x1000);
    cu->write_sgpr(sb + 5, (8u << 16) | (1u << 30)); // Stride 8, 4-byte swizzle units.
    cu->write_sgpr(sb + 6, c.records);
    cu->write_sgpr(sb + 12, c.scalar);
    cu->write_vgpr(vb, lane, c.index);
    const uint64_t address = 0x1000ull + c.index * 4 + (c.offset / 4) * 32 + c.scalar;
    for (uint32_t mode = 0; mode < 4; ++mode) {
      const bool valid = c.valid_modes & (1u << mode);
      cu->write_sgpr(sb + 7, identity | (20u << 12) | (mode << 28));
      for (bool typed : {false, true}) {
        for (bool load : {false, true}) {
          SCOPED_TRACE(testing::Message()
                       << "records=" << c.records << " scalar=" << c.scalar << " index=" << c.index
                       << " offset=" << c.offset << " mode=" << mode << " typed=" << typed
                       << " load=" << load);
          memory.write32(address, 0x12345678);
          cu->write_vgpr(vb + 8, lane, 0xdeadbeef);
          cu->write_vgpr(vb + 8, 0, 0xfacefeed);
          std::unique_ptr<Instruction> inst;
          if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
            const auto words = rdna4::build_vbuffer(
                typed ? (load ? rdna4::kTbufferLoadFormatXVbuffer
                              : rdna4::kTbufferStoreFormatXVbuffer)
                      : (load ? rdna4::kBufferLoadB32Vbuffer : rdna4::kBufferStoreB32Vbuffer),
                {.soffset = 12,
                 .vdata = 8,
                 .rsrc = 4,
                 .format = 20,
                 .idxen = 1,
                 .ioffset = c.offset});
            inst.reset(decode_valid(*decoder, words.data()));
          } else {
            const auto words = typed
                                   ? rdna3::build_mtbuf(load ? rdna3::kTbufferLoadFormatXMtbuf
                                                             : rdna3::kTbufferStoreFormatXMtbuf,
                                                        {.offset = static_cast<uint16_t>(c.offset),
                                                         .format = 20,
                                                         .vdata = 8,
                                                         .srsrc = 1,
                                                         .idxen = 1,
                                                         .soffset = 12})
                                   : rdna3::build_mubuf(load ? rdna3::kBufferLoadB32Mubuf
                                                             : rdna3::kBufferStoreB32Mubuf,
                                                        {.offset = static_cast<uint16_t>(c.offset),
                                                         .vdata = 8,
                                                         .srsrc = 1,
                                                         .idxen = 1,
                                                         .soffset = 12});
            inst.reset(decode_valid(*decoder, words.data()));
          }
          ASSERT_NE(inst, nullptr);
          ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
          EXPECT_EQ(inst->data_as<amdgpu::VectorMemState>()->lane_mask, valid ? wf->exec() : 0);
          physical->reads = 0;
          amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
          ASSERT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
          if (load && valid) {
            EXPECT_GT(physical->reads, 0u); // An unaligned load can span two pages.
          } else {
            EXPECT_EQ(physical->reads, 0u);
          }
          EXPECT_EQ(cu->read_vgpr(vb + 8, lane), load ? (valid ? 0x12345678u : 0u) : 0xdeadbeefu);
          EXPECT_EQ(cu->read_vgpr(vb + 8, 0), 0xfacefeedu);
          EXPECT_EQ(memory.read32(address), !load && valid ? 0xdeadbeefu : 0x12345678u);
        }
      }
    }
  }
  wf->set_address_space({});
  cu->set_gpu_vm(nullptr);
}

TEST_P(BufferFormatExecutionTest, IndexedNarrowStoresPreserveNeighborsAndLoadsExpandToVgprs) {
  descriptor(11, 2, 4 | (1 << 9)); // 16_UINT, X001.
  memory.write32(0x1000, 0xaaaaaaaa);
  memory.write32(0x1004, 0xbbbbbbbb);
  const uint32_t vb = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < 3; ++lane) {
    cu->write_vgpr(vb, lane, lane + 1);
    cu->write_vgpr(vb + 8, lane, 0x1234 + lane);
    for (uint32_t i = 1; i < 4; ++i)
      cu->write_vgpr(vb + 8 + i, lane, 0xdeadbeef);
  }
  issue(false);
  cu->l1_vector().flush_all();
  l2.flush_all();
  EXPECT_EQ(memory.read32(0x1000), 0x1234aaaa);
  EXPECT_EQ(memory.read32(0x1004), 0x1236bbbb);
  issue(true);
  for (uint32_t lane : {0u, 2u}) {
    EXPECT_EQ(cu->read_vgpr(vb + 8, lane), 0x1234 + lane);
    EXPECT_EQ(cu->read_vgpr(vb + 9, lane), 0);
    EXPECT_EQ(cu->read_vgpr(vb + 10, lane), 0);
    EXPECT_EQ(cu->read_vgpr(vb + 11, lane), 1);
  }
  EXPECT_EQ(cu->read_vgpr(vb + 8, 1), 0x1235);
  EXPECT_EQ(cu->read_vgpr(vb + 11, 1), 0xdeadbeef);
}

TEST_P(BufferFormatExecutionTest, InvalidFormatZerosActiveDestinationsWithoutBackingReads) {
  amdgpu::GpuVm vm;
  auto physical = std::make_shared<test::CountingGpuMemory>(memory);
  const auto address_space = vm.register_address_space(
      0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(), physical);
  ASSERT_TRUE(address_space);
  cu->set_gpu_vm(&vm);
  wf->set_address_space(address_space);
  descriptor(0, 4);
  memory.write32(0x1000, 0xffffffff);
  for (bool faulting : {false, true}) {
    physical->fault_reads = faulting;
    for (uint32_t lane : {0u, 1u, 2u})
      for (uint32_t reg = 0; reg < 4; ++reg)
        cu->write_vgpr(wf->vgpr_alloc().base + 8 + reg, lane, 0xdeadbeef);
    issue(true);
    EXPECT_EQ(physical->reads, 0u);
    for (uint32_t lane : {0u, 1u, 2u})
      for (uint32_t reg = 0; reg < 4; ++reg)
        EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + 8 + reg, lane), lane == 1 ? 0xdeadbeef : 0);
  }
  wf->set_address_space({});
  cu->set_gpu_vm(nullptr);
}

TEST_P(BufferFormatExecutionTest, D16StoresExpandSignedHalvesAndOddLoadsPreserveTheUpperHalf) {
  descriptor(62, 16); // 32_32_32_32_SINT.
  const uint32_t vb = wf->vgpr_alloc().base;
  for (uint32_t lane : {0u, 2u}) {
    cu->write_vgpr(vb, lane, lane + 1);
    cu->write_vgpr(vb + 8, lane, 0x8000ffff);
    cu->write_vgpr(vb + 9, lane, 0x12345678);
  }
  issue(false, true);
  cu->l1_vector().flush_all();
  l2.flush_all();
  EXPECT_EQ(memory.read32(0x1010), 0xffffffff);
  EXPECT_EQ(memory.read32(0x1014), 0xffff8000);
  EXPECT_EQ(memory.read32(0x1018), 0x5678);
  EXPECT_EQ(memory.read32(0x101c), 0x1234);
  for (uint32_t lane : {0u, 2u}) {
    cu->write_vgpr(vb + 8, lane, 0);
    cu->write_vgpr(vb + 9, lane, 0xdeadbeef);
  }
  issue(true, true); // XYZ uses one and a half VGPRs.
  for (uint32_t lane : {0u, 2u}) {
    EXPECT_EQ(cu->read_vgpr(vb + 8, lane), 0x8000ffff);
    EXPECT_EQ(cu->read_vgpr(vb + 9, lane), (cu->sram_ecc() ? 0x5678u : 0xdead5678u));
  }
}

TEST_P(BufferFormatExecutionTest, D16Float32LoadsTruncateInsteadOfRoundingToNearest) {
  descriptor(60, 16);
  const uint32_t vb = wf->vgpr_alloc().base;
  for (uint32_t lane : {0u, 2u}) {
    cu->write_vgpr(vb, lane, lane);
    cu->write_vgpr(vb + 9, lane, 0xabcd0000);
    memory.write32(0x1000 + lane * 16, bits(1.0008f));
    memory.write32(0x1004 + lane * 16, bits(-2));
    memory.write32(0x1008 + lane * 16, bits(0.5f));
  }
  issue(true, true);
  for (uint32_t lane : {0u, 2u}) {
    EXPECT_EQ(cu->read_vgpr(vb + 8, lane), 0xc0003c00);
    EXPECT_EQ(cu->read_vgpr(vb + 9, lane), (cu->sram_ecc() ? 0x3800u : 0xabcd3800u));
  }
}

TEST_P(BufferFormatExecutionTest, IndexedAddressUsesOffsetAfterIndexAndScaledStride) {
  descriptor(20, 4);
  const uint32_t sb = wf->sgpr_alloc().base;
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_sgpr(sb + 12, 16);
  cu->write_vgpr(vb, 0, 3);
  cu->write_vgpr(vb + 1, 0, 32);
  cu->write_vgpr(vb, 2, 5);
  cu->write_vgpr(vb + 1, 2, 64);
  amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
  state.elem_size = 4;
  state.num_elems = 1;
  uint32_t stride = 4;
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
    stride *= 8;
    cu->write_sgpr(sb + 7, identity | (20 << 12) | (2 << 18) | (1u << 28));
    rdna4::VbufferMachineInst m{};
    m.rsrc = 4;
    m.soffset = 12;
    m.idxen = m.offen = 1;
    m.ioffset = 8;
    rdna4::mubuf_calculate_addresses(m, *wf, state);
  } else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3) {
    rdna3::MubufMachineInst m{};
    m.srsrc = 1;
    m.soffset = 12;
    m.idxen = m.offen = 1;
    m.offset = 8;
    rdna3::mubuf_calculate_addresses(m, *wf, state);
  } else {
    rdna3_5::MubufMachineInst m{};
    m.srsrc = 1;
    m.soffset = 12;
    m.idxen = m.offen = 1;
    m.offset = 8;
    amdgpu::addr_calc::mubuf_calculate_addresses(m, *wf, state);
  }
  EXPECT_EQ(state.lane_mask, 5);
  EXPECT_EQ(state.per_lane_addr[0], 0x1000 + 3 * stride + 32 + 8 + 16);
  EXPECT_EQ(state.per_lane_addr[1], 0);
  EXPECT_EQ(state.per_lane_addr[2], 0x1000 + 5 * stride + 64 + 8 + 16);
}

INSTANTIATE_TEST_SUITE_P(Targets, BufferFormatExecutionTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                           ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                           ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4,
                                           ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                           ROCJITSU_CODE_ARCH_RDNA4));

class RdnaLegacyFormatDecodeTest : public BufferFormatExecutionTest {};

TEST_P(RdnaLegacyFormatDecodeTest, SplitOpcodeBitSelectsD16BeforeExecution) {
  // MUBUF OPM is bit 25; MTBUF OPM is bit 53. Both store a half value of 1000.
  const std::array<std::array<uint32_t, 2>, 2> code{{
      {0xe2102000, 0x80010800},
      {0xe86c2000, 0x80210800},
  }};
  auto decoder = Decoder::create(GetParam());
  ASSERT_NE(decoder, nullptr);
  descriptor(13, 2); // 16_FLOAT.
  wf->set_exec(1);
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb, 0, 0);
  cu->write_vgpr(vb + 8, 0, 0x63d0);
  for (const auto &words : code) {
    memory.write32(0x1000, 0xaaaaaaaa);
    auto *inst = decode_valid(*decoder, words.data());
    ASSERT_NE(inst, nullptr);
    EXPECT_NE(inst->mnemonic().find("store_format_d16_x"), std::string::npos);
    ASSERT_TRUE(cu->execute_instruction(inst, *wf).succeeded());
    ASSERT_NE(inst->data(), nullptr);
    amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
    ASSERT_EQ(pipeline.issue(inst, *wf), amdgpu::VmAccessOutcome::Complete);
    cu->l1_vector().flush_all();
    l2.flush_all();
    EXPECT_EQ(memory.read32(0x1000), 0xaaaa63d0);
  }
}

INSTANTIATE_TEST_SUITE_P(Rdna, RdnaLegacyFormatDecodeTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2));
} // namespace
