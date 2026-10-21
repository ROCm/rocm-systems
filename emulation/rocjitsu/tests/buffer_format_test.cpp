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
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/mtbuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/mubuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
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

#include <array>
#include <bit>
#include <cfenv>
#include <cstdint>
#include <memory>

namespace {
using namespace rocjitsu;
constexpr uint32_t identity = 4 | (5 << 3) | (6 << 6) | (7 << 9);
constexpr uint32_t bgra = 6 | (5 << 3) | (4 << 6) | (7 << 9);
uint32_t bits(float f) { return std::bit_cast<uint32_t>(f); }

TEST(BufferFormatTest, Gfx11TwoBitSnormRoundsHalfAwayFromZero) {
  // Native gfx1100/gfx1201 agree for typed/untyped and ordinary/D16 stores.
  const std::array<float, 8> inputs{0.5f, -0.5f, 0.499f, -0.499f, 0.501f, -0.501f, 1, -1};
  const std::array<uint32_t, 8> red_two_bit{0x40100401, 0xc0300c03, 0x3fcff3fc, 0xc0701c04,
                                            0x40100401, 0xc0300c03, 0x7fdff7fd, 0x80601807};
  const std::array<uint32_t, 8> alpha_two_bit{0x50040100, 0xf00c0300, 0x0ff3fcff, 0x301c0701,
                                              0x50040100, 0xf00c0300, 0x5ff7fdff, 0xe0180601};
  for (uint32_t format : {33u, 37u, 39u})
    for (size_t i = 0; i < inputs.size(); ++i) {
      SCOPED_TRACE(testing::Message() << "format=" << format << " input=" << inputs[i]);
      const uint32_t value = bits(inputs[i]);
      const std::array<uint32_t, 4> source{value, value, value, value};
      std::array<uint8_t, 4> packed{};
      ASSERT_TRUE(amdgpu::pack_buffer_format(format, identity, source, packed,
                                             amdgpu::BufferFormatEncoding::Gfx11)
                      .succeeded());
      EXPECT_EQ(std::bit_cast<uint32_t>(packed), (format == 33 ? red_two_bit : alpha_two_bit)[i]);
    }
}

TEST(BufferFormatTest, ConstantStoreAlphaUsesTheMemoryNumericFormat) {
  constexpr uint32_t rgb = 4 | (5 << 3) | (6 << 6);
  struct Case {
    uint32_t format, width, one;
  };
  // Native gfx1100/gfx1201: UINT/SINT8/16/32, UNORM8/16, FLOAT16/32.
  constexpr std::array cases{
      Case{46, 1, 1},      Case{55, 2, 1},          Case{61, 4, 1},      Case{47, 1, 1},
      Case{56, 2, 1},      Case{62, 4, 1},          Case{42, 1, 0xff},   Case{51, 2, 0xffff},
      Case{57, 2, 0x3c00}, Case{63, 4, 0x3f800000}, Case{43, 1, 0x7f},   Case{52, 2, 0x7fff},
      Case{44, 1, 0xff},   Case{45, 1, 0x7f},       Case{53, 2, 0xffff}, Case{54, 2, 0x7fff}};
  for (auto test : cases)
    for (uint32_t alpha : {0u, 1u})
      for (auto missing : {amdgpu::BufferStoreMissingChannel::Zero,
                           amdgpu::BufferStoreMissingChannel::ReplicateFirst})
        for (uint32_t count = 1; count <= 4; ++count) {
          SCOPED_TRACE(testing::Message()
                       << "format=" << test.format << " alpha=" << alpha << " count=" << count);
          const std::array<uint32_t, 4> components{0x11, 0x22, 0x33, 0x44};
          std::array<uint8_t, 16> bytes{};
          ASSERT_TRUE(amdgpu::pack_buffer_format(test.format, rgb | (alpha << 9),
                                                 std::span(components).first(count),
                                                 std::span(bytes).first(test.width * 4),
                                                 amdgpu::BufferFormatEncoding::Gfx11, missing)
                          .succeeded());
          uint32_t actual = 0;
          for (uint32_t b = 0; b < test.width; ++b)
            actual |= uint32_t(bytes[test.width * 3 + b]) << (b * 8);
          EXPECT_EQ(actual, alpha ? test.one : 0);
        }
}

TEST(BufferFormatTest, PackedConstantAlphaAndScaledStoresMatchNativeGfx11) {
  constexpr uint32_t rgb1 = 4 | (5 << 3) | (6 << 6) | (1 << 9);
  const std::array<uint32_t, 4> values{bits(0.5f), bits(1), bits(-1), bits(-2)};
  for (uint32_t format : {32u, 33u, 34u, 35u, 36u, 37u, 38u, 39u, 40u, 41u}) {
    std::array<uint8_t, 4> bytes{};
    ASSERT_TRUE(amdgpu::pack_buffer_format(format, rgb1, values, bytes).succeeded());
    const uint32_t shift = format <= 35 ? 22 : 30;
    const uint32_t expected = format == 32                   ? 1023
                              : format == 33                 ? 511
                              : format == 36 || format == 38 ? 3
                                                             : 1;
    EXPECT_EQ(std::bit_cast<uint32_t>(bytes) >> shift, expected) << format;
  }
  // Cover every SCALED format, using canonical selectors for narrow formats.
  for (uint32_t format : {3u, 4u, 9u, 10u, 16u, 17u, 25u, 26u, 38u, 39u, 44u, 45u, 53u, 54u}) {
    const uint32_t channels = format <= 10 ? 1 : format <= 26 ? 2 : 4;
    uint32_t selectors = channels == 4 ? identity : 1u << 9;
    for (uint32_t i = 0; i < channels; ++i)
      selectors |= (i + 4) << (3 * i);
    std::array<uint8_t, 16> scaled{}, normalized{};
    ASSERT_TRUE(amdgpu::pack_buffer_format(format, selectors, values, scaled).succeeded());
    ASSERT_TRUE(amdgpu::pack_buffer_format(format - 2, selectors, values, normalized).succeeded());
    EXPECT_EQ(scaled, normalized) << format;
  }
  std::array<uint8_t, 4> legacy{}, modern{};
  ASSERT_TRUE(amdgpu::pack_buffer_format(10 | (2 << 4), identity, values, legacy,
                                         amdgpu::BufferFormatEncoding::Gfx9)
                  .succeeded());
  ASSERT_TRUE(amdgpu::pack_buffer_format(44, identity, values, modern).succeeded());
  EXPECT_EQ(legacy, (std::array<uint8_t, 4>{0, 1, 0, 0}));
  EXPECT_EQ(modern, (std::array<uint8_t, 4>{128, 255, 0, 0}));
}

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

TEST(BufferFormatTest, PacksConvertedComponentsWithExplicitMissingChannelPolicy) {
  std::array<uint8_t, 4> bytes{};
  const std::array<uint32_t, 4> values{bits(-2), bits(0.5f), bits(1), bits(2)};
  ASSERT_TRUE(amdgpu::pack_buffer_format(42, identity, values, bytes).succeeded());
  EXPECT_EQ(bytes, (std::array<uint8_t, 4>{0, 128, 255, 255}));
  const std::array<uint32_t, 2> integers{0x12, 0x34};
  ASSERT_TRUE(amdgpu::pack_buffer_format(46, identity, integers, bytes).succeeded());
  EXPECT_EQ(bytes, (std::array<uint8_t, 4>{0x12, 0x34, 0, 0}));
  ASSERT_TRUE(amdgpu::pack_buffer_format(46, identity, integers, bytes,
                                         amdgpu::BufferFormatEncoding::Gfx11,
                                         amdgpu::BufferStoreMissingChannel::ReplicateFirst)
                  .succeeded());
  EXPECT_EQ(bytes, (std::array<uint8_t, 4>{0x12, 0x34, 0x12, 0x12}));
  ASSERT_TRUE(amdgpu::pack_buffer_format(46, identity, {}, bytes,
                                         amdgpu::BufferFormatEncoding::Gfx11,
                                         amdgpu::BufferStoreMissingChannel::ReplicateFirst)
                  .failed());
  const std::array<uint32_t, 2> halves{bits(1), bits(-2)};
  // The ISA defines format 29 as 16_16_FLOAT. Use canonical RG01 selectors;
  // the expected bytes below are the IEEE binary16 encodings of 1 and -2.
  ASSERT_TRUE(amdgpu::pack_buffer_format(29, 4 | (5 << 3) | (1 << 9), halves, bytes).succeeded());
  EXPECT_EQ(bytes, (std::array<uint8_t, 4>{0, 0x3c, 0, 0xc0}));
  const std::array<uint32_t, 3> floats{bits(1), bits(2), bits(0.5f)};
  ASSERT_TRUE(amdgpu::pack_buffer_format(30, 4 | (5 << 3) | (6 << 6) | (1 << 9), floats, bytes)
                  .succeeded());
  EXPECT_EQ(std::bit_cast<uint32_t>(bytes), (15u << 6) | ((16u << 6) << 11) | ((14u << 5) << 22));
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
  wf->set_lds_size(128);
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
      case 5:
        return 1 | (4 << 4); // 8_UINT.
      case 18:
        return 3 | (4 << 4); // 8_8_UINT.
      case 58:
        return 13 | (4 << 4); // 32_32_32_UINT.
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
  void issue_gfx11_store(uint32_t format, bool typed, bool d16) {
    auto decoder = Decoder::create(GetParam());
    const uint16_t op = d16 ? 15 : 7;
    std::unique_ptr<Instruction> inst;
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
      const auto words =
          rdna4::build_vbuffer(op + (typed ? 128 : 0), {.soffset = rdna4::OPR_SREG_M0_NULL,
                                                        .vdata = 8,
                                                        .rsrc = 4,
                                                        .scope = 3,
                                                        .format = static_cast<uint8_t>(format),
                                                        .idxen = 1});
      inst.reset(decode_valid(*decoder, words.data()));
    } else if (typed) {
      const auto words = rdna3::build_mtbuf(op, {.format = static_cast<uint8_t>(format),
                                                 .vdata = 8,
                                                 .srsrc = 1,
                                                 .idxen = 1,
                                                 .soffset = 128});
      inst.reset(decode_valid(*decoder, words.data()));
    } else {
      const auto words =
          rdna3::build_mubuf(op, {.vdata = 8, .srsrc = 1, .idxen = 1, .soffset = 128});
      inst.reset(decode_valid(*decoder, words.data()));
    }
    ASSERT_NE(inst, nullptr);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
    cu->l1_vector().flush_all();
    l2.flush_all();
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

TEST_P(BufferFormatExecutionTest, MissingStoreChannelsFollowConfiguredPolicyForAllWidths) {
  // Native gfx1100/gfx1201 establish the RDNA3/RDNA4 expectations. The other
  // targets test the configured extrapolation, not hardware qualification.
  const auto arch = GetParam();
  auto decoder = Decoder::create(arch);
  const auto vb = wf->vgpr_alloc().base;
  const bool replicate = arch == ROCJITSU_CODE_ARCH_RDNA4;
  for (bool typed : {false, true})
    for (bool d16 : {false, true}) {
      for (uint32_t count = 1; count <= 4; ++count)
        for (uint32_t selectors :
             {identity, bgra, 7u | (6u << 3) | (5u << 6) | (4u << 9), 4u | (5u << 3) | (6u << 6),
              4u | (5u << 3) | (6u << 6) | (1u << 9)}) {
          SCOPED_TRACE(testing::Message() << "typed=" << typed << " d16=" << d16
                                          << " count=" << count << " selectors=" << selectors);
          const bool reverse = (selectors & 7) == 7;
          descriptor(46, 4, selectors); // 8_8_8_8_UINT, typed instructions use identity.
          for (uint32_t lane = 0; lane < 3; ++lane) {
            memory.write32(0x1000 + lane * 4, 0xa5a5a5a5);
            cu->write_vgpr(vb, lane, lane);
            for (uint32_t reg = 0; reg < 4; ++reg)
              cu->write_vgpr(vb + 8 + reg, lane, (reg + 1) * 0x11);
            if (d16) {
              cu->write_vgpr(vb + 8, lane, 0x00220011);
              cu->write_vgpr(vb + 9, lane, 0x00440033);
            }
          }
          const uint32_t op = (d16 ? 12 : 4) + count - 1;
          std::unique_ptr<Instruction> inst;
          if (arch == ROCJITSU_CODE_ARCH_RDNA4) {
            const auto words =
                rdna4::build_vbuffer(op + (typed ? 128 : 0), {.soffset = rdna4::OPR_SREG_M0_NULL,
                                                              .vdata = 8,
                                                              .rsrc = 4,
                                                              .scope = 3,
                                                              .format = 46,
                                                              .idxen = 1});
            inst.reset(decode_valid(*decoder, words.data()));
          } else if (arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2) {
            if (typed) {
              const auto words =
                  rdna1::build_mtbuf(4 + count - 1, {.idxen = 1,
                                                     .format = 60,
                                                     .vdata = 8,
                                                     .srsrc = 1,
                                                     .opm = static_cast<uint8_t>(d16),
                                                     .soffset = 128});
              inst.reset(decode_valid(*decoder, words.data()));
            } else {
              const auto words =
                  rdna1::build_mubuf(4 + count - 1, {.idxen = 1,
                                                     .opm = static_cast<uint8_t>(d16),
                                                     .vdata = 8,
                                                     .srsrc = 1,
                                                     .soffset = 128});
              inst.reset(decode_valid(*decoder, words.data()));
            }
          } else if (typed && arch_is_cdna_4_or_lower(arch)) {
            const auto words = cdna4::build_mtbuf(
                op, {.idxen = 1, .dfmt = 10, .nfmt = 4, .vdata = 8, .srsrc = 1, .soffset = 128});
            inst.reset(decode_valid(*decoder, words.data()));
          } else if (typed) {
            const auto words = rdna3::build_mtbuf(
                op, {.format = 46, .vdata = 8, .srsrc = 1, .idxen = 1, .soffset = 128});
            inst.reset(decode_valid(*decoder, words.data()));
          } else if (arch_is_cdna_4_or_lower(arch)) {
            const auto words =
                cdna4::build_mubuf(op, {.idxen = 1, .vdata = 8, .srsrc = 1, .soffset = 128});
            inst.reset(decode_valid(*decoder, words.data()));
          } else {
            const auto words =
                rdna3::build_mubuf(op, {.vdata = 8, .srsrc = 1, .idxen = 1, .soffset = 128});
            inst.reset(decode_valid(*decoder, words.data()));
          }
          ASSERT_NE(inst, nullptr);
          ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
          amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
          ASSERT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
          cu->l1_vector().flush_all();
          l2.flush_all();
          uint32_t expected = 0;
          for (uint32_t component = 0; component < 4; ++component) {
            const uint32_t shader = !typed && reverse                              ? 3 - component
                                    : !typed && selectors == bgra && component < 3 ? 2 - component
                                                                                   : component;
            const bool constant_alpha = !typed && component == 3 && (selectors >> 9) <= 1;
            const uint32_t value = constant_alpha   ? selectors >> 9
                                   : shader < count ? (shader + 1) * 0x11
                                   : replicate      ? 0x11
                                                    : 0;
            expected |= value << (component * 8);
          }
          EXPECT_EQ(memory.read32(0x1000), expected);
          EXPECT_EQ(memory.read32(0x1004), 0xa5a5a5a5u);
          EXPECT_EQ(memory.read32(0x1008), expected);
        }
    }
}

TEST_P(BufferFormatExecutionTest, Gfx11IntegerStoresSaturateForTypedAndD16Sources) {
  if (GetParam() != ROCJITSU_CODE_ARCH_RDNA3 && GetParam() != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      GetParam() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Older integer store conversions were not measured";
  struct Case {
    uint32_t format, bytes;
    uint64_t ordinary, d16;
  };
  // Measured on gfx1100 and gfx1201 for typed/untyped XYZW and D16_XYZW.
  // Cover UINT/SINT8, UINT/SINT16, both 2/10-bit layouts, and four-channel 8/16-bit formats.
  constexpr std::array cases{Case{5, 1, 0xff, 0xff},
                             Case{6, 1, 0x80, 0x80},
                             Case{11, 2, 0xffff, 0x8000},
                             Case{12, 2, 0x8000, 0x8000},
                             Case{34, 4, 0x403fffff, 0x403fffff},
                             Case{35, 4, 0x401ff803, 0x401ff803},
                             Case{40, 4, 0xffffffff, 0xffffffff},
                             Case{41, 4, 0x5ff803ff, 0x5ff803ff},
                             Case{46, 4, 0xffffffff, 0xffffffff},
                             Case{47, 4, 0x7f7f80ff, 0x7f7f80ff},
                             Case{55, 8, 0x0100ffffffffffff, 0x01007fff8000ffff},
                             Case{56, 8, 0x01007fff8000ffff, 0x01007fff8000ffff}};
  const auto vb = wf->vgpr_alloc().base;
  for (const auto &c : cases)
    for (bool typed : {false, true})
      for (bool d16 : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "format=" << c.format << " typed=" << typed << " d16=" << d16);
        descriptor(c.format, 16);
        for (uint32_t lane = 0; lane < 3; ++lane) {
          cu->write_vgpr(vb, lane, lane);
          const std::array<uint32_t, 4> values{0xffffffff, 0x80000000, 65536, 256};
          for (uint32_t i = 0; i < values.size(); ++i)
            cu->write_vgpr(vb + 8 + i, lane, values[i]);
          if (d16) {
            cu->write_vgpr(vb + 8, lane, 0x8000ffff);
            cu->write_vgpr(vb + 9, lane, 0x01007fff);
          }
          // Single-channel X must distinguish clamping from truncation. A D16
          // source fits a 16-bit destination, but still overflows an 8-bit one.
          if (c.bytes <= 2)
            cu->write_vgpr(vb + 8, lane, d16 ? 0x80008000 : 0x80000000);
          for (uint32_t i = 0; i < 16; ++i)
            memory.write8(0x1000 + lane * 16 + i, 0xa5);
        }
        ASSERT_NO_FATAL_FAILURE(issue_gfx11_store(c.format, typed, d16));
        const uint64_t expected = d16 ? c.d16 : c.ordinary;
        for (uint32_t lane = 0; lane < 3; ++lane)
          for (uint32_t i = 0; i < 16; ++i)
            EXPECT_EQ(memory.read8(0x1000 + lane * 16 + i),
                      lane != 1 && i < c.bytes ? (expected >> (i * 8)) & 0xff : 0xa5);
      }
}

TEST_P(BufferFormatExecutionTest, BgraStoresPackNormalizedAndIntegerChannels) {
  if (GetParam() != ROCJITSU_CODE_ARCH_RDNA3 && GetParam() != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      GetParam() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Native BGRA conversion witnesses use Gfx11 formats";
  struct Case {
    uint32_t format, expected;
    bool normalized;
  };
  // Native gfx1100/gfx1201: 10_10_10_2 and 8_8_8_8 UNORM/UINT, BGRA selectors.
  constexpr std::array cases{Case{36, 0xc0040200, true}, Case{40, 0xc1108833, false},
                             Case{42, 0xff004080, true}, Case{46, 0x44112233, false}};
  const auto vb = wf->vgpr_alloc().base;
  for (const auto &c : cases)
    for (bool d16 : {false, true}) {
      SCOPED_TRACE(testing::Message() << "format=" << c.format << " d16=" << d16);
      descriptor(c.format, 4, bgra);
      for (uint32_t lane = 0; lane < 3; ++lane) {
        cu->write_vgpr(vb, lane, lane);
        const auto values = c.normalized ? std::array{bits(0), bits(0.25f), bits(0.5f), bits(1)}
                                         : std::array{0x11u, 0x22u, 0x33u, 0x44u};
        for (uint32_t i = 0; i < values.size(); ++i)
          cu->write_vgpr(vb + 8 + i, lane, values[i]);
        if (d16) {
          cu->write_vgpr(vb + 8, lane, c.normalized ? 0x34000000 : 0x00220011);
          cu->write_vgpr(vb + 9, lane, c.normalized ? 0x3c003800 : 0x00440033);
        }
        memory.write32(0x1000 + lane * 4, 0xa5a5a5a5);
      }
      ASSERT_NO_FATAL_FAILURE(issue_gfx11_store(c.format, false, d16));
      EXPECT_EQ(memory.read32(0x1000), c.expected);
      EXPECT_EQ(memory.read32(0x1004), 0xa5a5a5a5u);
      EXPECT_EQ(memory.read32(0x1008), c.expected);
    }
}

TEST_P(BufferFormatExecutionTest, ScaledStoresNormalizeOnGfx11ForTypedAndD16Sources) {
  const auto arch = GetParam();
  if (arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      arch != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Older SCALED store conversions were not measured";
  auto decoder = Decoder::create(arch);
  wf->set_exec(1);
  const auto vb = wf->vgpr_alloc().base;
  for (uint32_t format : {44u, 45u})
    for (bool typed : {false, true})
      for (bool d16 : {false, true}) {
        descriptor(format, 4);
        cu->write_vgpr(vb, 0, 0);
        const std::array<uint32_t, 4> values{bits(0.5f), bits(1), bits(-1), bits(-2)};
        for (uint32_t i = 0; i < 4; ++i)
          cu->write_vgpr(vb + 8 + i, 0, values[i]);
        if (d16) {
          cu->write_vgpr(vb + 8, 0, 0x3c003800);
          cu->write_vgpr(vb + 9, 0, 0xc000bc00);
        }
        memory.write32(0x1000, 0xa5a5a5a5);
        const uint16_t op = d16 ? 15 : 7;
        std::unique_ptr<Instruction> inst;
        if (arch == ROCJITSU_CODE_ARCH_RDNA4) {
          const auto words =
              rdna4::build_vbuffer(op + (typed ? 128 : 0), {.soffset = rdna4::OPR_SREG_M0_NULL,
                                                            .vdata = 8,
                                                            .rsrc = 4,
                                                            .format = static_cast<uint8_t>(format),
                                                            .idxen = 1});
          inst.reset(decode_valid(*decoder, words.data()));
        } else if (typed) {
          const auto words = rdna3::build_mtbuf(op, {.format = static_cast<uint8_t>(format),
                                                     .vdata = 8,
                                                     .srsrc = 1,
                                                     .idxen = 1,
                                                     .soffset = 128});
          inst.reset(decode_valid(*decoder, words.data()));
        } else {
          const auto words =
              rdna3::build_mubuf(op, {.vdata = 8, .srsrc = 1, .idxen = 1, .soffset = 128});
          inst.reset(decode_valid(*decoder, words.data()));
        }
        ASSERT_NE(inst, nullptr);
        ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
        amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
        ASSERT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
        cu->l1_vector().flush_all();
        l2.flush_all();
        EXPECT_EQ(memory.read32(0x1000), format == 44 ? 0x0000ff80u : 0x81817f40u);
      }
}

TEST_P(BufferFormatExecutionTest, D16FloatRgb1ConvertsRgbAndFillsMissingChannels) {
  const auto arch = GetParam();
  if (arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      arch != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "This native witness uses Gfx11 format encodings";
  auto decoder = Decoder::create(arch);
  wf->set_exec(1);
  const auto vb = wf->vgpr_alloc().base;
  const std::array<uint32_t, 4> values{bits(0.5f), bits(1), bits(2), bits(3)};
  for (uint32_t count = 1; count <= 4; ++count) {
    descriptor(63, 16, 0x3ac); // 32_32_32_32_FLOAT with RGB1.
    cu->write_vgpr(vb, 0, 0);
    cu->write_vgpr(vb + 8, 0, 0x3c003800);
    cu->write_vgpr(vb + 9, 0, 0x42004000);
    for (uint32_t i = 0; i < 8; ++i)
      memory.write32(0x1000 + i * 4, 0xa5a5a5a5);
    std::unique_ptr<Instruction> inst;
    const uint16_t op = 12 + count - 1;
    if (arch == ROCJITSU_CODE_ARCH_RDNA4) {
      const auto words = rdna4::build_vbuffer(
          op, {.soffset = rdna4::OPR_SREG_M0_NULL, .vdata = 8, .rsrc = 4, .idxen = 1});
      inst.reset(decode_valid(*decoder, words.data()));
    } else {
      const auto words =
          rdna3::build_mubuf(op, {.vdata = 8, .srsrc = 1, .idxen = 1, .soffset = 128});
      inst.reset(decode_valid(*decoder, words.data()));
    }
    ASSERT_NE(inst, nullptr);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
    cu->l1_vector().flush_all();
    l2.flush_all();
    for (uint32_t i = 0; i < 3; ++i)
      EXPECT_EQ(memory.read32(0x1000 + i * 4), i < count                          ? values[i]
                                               : arch == ROCJITSU_CODE_ARCH_RDNA4 ? values[0]
                                                                                  : 0u);
    EXPECT_EQ(memory.read32(0x100c), bits(1));
    for (uint32_t i = 4; i < 8; ++i)
      EXPECT_EQ(memory.read32(0x1000 + i * 4), 0xa5a5a5a5u);
  }
}

TEST_P(BufferFormatExecutionTest, UnqualifiedStoreSelectorsAreRejectedBeforeMemoryIssue) {
  auto decoder = Decoder::create(GetParam());
  descriptor(46, 4, 4 | (4 << 3) | (4 << 6) | (4 << 9));
  memory.write32(0x1000, 0xa5a5a5a5u);
  std::unique_ptr<Instruction> inst;
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
    const auto words =
        rdna4::build_vbuffer(rdna4::kBufferStoreFormatXVbuffer,
                             {.soffset = rdna4::OPR_SREG_M0_NULL, .vdata = 8, .rsrc = 4});
    inst.reset(decode_valid(*decoder, words.data()));
  } else {
    const auto words = cdna4::build_mubuf(4, {.vdata = 8, .srsrc = 1, .soffset = 128});
    inst.reset(decode_valid(*decoder, words.data()));
  }
  ASSERT_NE(inst, nullptr);
  EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).failed());
  EXPECT_EQ(inst->data(), nullptr);
  EXPECT_EQ(memory.read32(0x1000), 0xa5a5a5a5u);
  for (uint32_t selectors :
       {4u | (5u << 3) | (0u << 6) | (1u << 9), 7u | (7u << 3) | (5u << 6) | (5u << 9),
        4u | (6u << 3) | (7u << 6) | (5u << 9)}) {
    descriptor(46, 4, selectors);
    amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
    state.is_load = false;
    EXPECT_TRUE(amdgpu::prepare_buffer_format(*wf, state, 4, -1, 1).failed());
  }
}

TEST_P(BufferFormatExecutionTest, NarrowStoreRoutingUsesQualifiedSelectorPatterns) {
  auto decoder = Decoder::create(GetParam());
  wf->set_exec(1);
  for (uint32_t i = 0; i < 4; ++i)
    cu->write_vgpr(wf->vgpr_alloc().base + 8 + i, 0, 0x11u * (i + 1));
  for (uint32_t format : {5u, 18u, 58u}) {
    const uint32_t channels = format == 5 ? 1 : format == 18 ? 2 : 3;
    uint32_t selectors = 1u << 9;
    for (uint32_t i = 0; i < channels; ++i)
      selectors |= (i + 4) << (3 * i);
    for (bool alpha : {false, true}) {
      if (alpha && channels != 1)
        continue;
      descriptor(format, 16, alpha ? 4u << 9 : selectors);
      cu->write_sgpr(wf->sgpr_alloc().base + 6, 64);
      memory.write32(0x1000, 0xa5a5a5a5u);
      std::unique_ptr<Instruction> inst;
      if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
        const auto words =
            rdna4::build_vbuffer(rdna4::kBufferStoreFormatXyzwVbuffer,
                                 {.soffset = rdna4::OPR_SREG_M0_NULL, .vdata = 8, .rsrc = 4});
        inst.reset(decode_valid(*decoder, words.data()));
      } else {
        const auto words = cdna4::build_mubuf(7, {.vdata = 8, .srsrc = 1, .soffset = 128});
        inst.reset(decode_valid(*decoder, words.data()));
      }
      ASSERT_NE(inst, nullptr);
      ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
      amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
      ASSERT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
      EXPECT_EQ(memory.read32(0x1000), channels == 1   ? (alpha ? 0xa5a5a544u : 0xa5a5a511u)
                                       : channels == 2 ? 0xa5a52211u
                                                       : 0x11u);
      if (channels == 3) {
        EXPECT_EQ(memory.read32(0x1004), 0x22u);
        EXPECT_EQ(memory.read32(0x1008), 0x33u);
      }
    }
  }
  for (uint32_t selectors :
       {identity, 4u | (5u << 3) | (4u << 6) | (5u << 9), 7u | (6u << 3) | (5u << 6) | (4u << 9)}) {
    descriptor(18, 16, selectors);
    amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
    state.is_load = false;
    EXPECT_TRUE(amdgpu::prepare_buffer_format(*wf, state, 4, -1, 4).failed());
    std::array<uint8_t, 2> bytes{0xa5, 0xa5};
    const std::array<uint32_t, 4> components{0x11, 0x22, 0x33, 0x44};
    EXPECT_TRUE(amdgpu::pack_buffer_format(18, selectors, components, bytes).failed());
    EXPECT_EQ(bytes, (std::array<uint8_t, 2>{0xa5, 0xa5}));
  }
}

TEST_P(BufferFormatExecutionTest, RdnaNarrowLoadSelectorsWrapOnlyForUntypedFormats) {
  if (GetParam() != ROCJITSU_CODE_ARCH_RDNA3 && GetParam() != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      GetParam() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Native qualification covers the Gfx11 formatted-load policy";
  auto decoder = Decoder::create(GetParam());
  wf->set_exec(1);
  for (uint32_t format : {5u, 18u, 58u, 59u, 30u}) {
    const uint32_t channels = format == 5 ? 1 : format == 18 ? 2 : 3;
    const bool fp = format == 30;
    const std::array<uint32_t, 3> physical =
        fp              ? std::array<uint32_t, 3>{0x3c220000, 0x44c80000, 0x3c400000}
        : channels == 3 ? std::array<uint32_t, 3>{0x44332211, 0x84736251, 0xc4b3a291}
                        : std::array<uint32_t, 3>{0x11, 0x22, 0};
    memory.write32(0x1000, 0x44332211u);
    memory.write32(0x1004, 0x84736251u);
    memory.write32(0x1008, 0xc4b3a291u);
    for (uint32_t selectors :
         {identity, 7u | (6u << 3) | (5u << 6) | (4u << 9), 6u | (5u << 3) | (1u << 9), 4u << 9})
      for (bool typed : {false, true})
        for (bool d16 : {false, true}) {
          SCOPED_TRACE(::testing::Message() << "format=" << format << " selectors=" << selectors
                                            << " typed=" << typed << " d16=" << d16);
          descriptor(format, 16, selectors);
          cu->write_sgpr(wf->sgpr_alloc().base + 6, 64);
          for (uint32_t i = 0; i < 4; ++i)
            cu->write_vgpr(wf->vgpr_alloc().base + 8 + i, 1, 0xdeadbeef);
          const uint16_t opcode = d16 ? 11 : 3;
          std::unique_ptr<Instruction> inst;
          if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
            const auto words = rdna4::build_vbuffer(
                opcode + (typed ? 128 : 0), {.soffset = rdna4::OPR_SREG_M0_NULL,
                                             .vdata = 8,
                                             .rsrc = 4,
                                             .format = static_cast<uint8_t>(typed ? format : 0)});
            inst.reset(decode_valid(*decoder, words.data()));
          } else if (typed) {
            const auto words = rdna3::build_mtbuf(
                opcode,
                {.format = static_cast<uint8_t>(format), .vdata = 8, .srsrc = 1, .soffset = 128});
            inst.reset(decode_valid(*decoder, words.data()));
          } else {
            const auto words = rdna3::build_mubuf(opcode, {.vdata = 8, .srsrc = 1, .soffset = 128});
            inst.reset(decode_valid(*decoder, words.data()));
          }
          ASSERT_NE(inst, nullptr);
          ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
          amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
          ASSERT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
          std::array<uint32_t, 4> expected{};
          for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t selector = typed ? i + 4 : (selectors >> (3 * i)) & 7;
            uint32_t value = selector == 1 ? (fp ? bits(1) : 1) : 0;
            if (selector >= 4 && (!typed || i < channels))
              value = physical[(selector - 4) % channels];
            if (d16) {
              // Native UINT32 values saturate to 0xffff; these SINT32 values
              // saturate to 0x7fff, 0x8000, 0x8000 respectively.
              uint16_t half = static_cast<uint16_t>(value);
              if (format == 58 && selector >= 4 && (!typed || i < channels))
                half = 0xffff;
              if (format == 59 && selector >= 4 && (!typed || i < channels))
                half = (selector - 4) % channels == 0 ? 0x7fff : 0x8000;
              if (fp)
                half = value == bits(1)      ? 0x3c00
                       : value == 0x3c220000 ? 0x2110
                       : value == 0x44c80000 ? 0x6640
                       : value == 0x3c400000 ? 0x2200
                                             : 0;
              expected[i / 2] |= static_cast<uint32_t>(half) << (16 * (i % 2));
            } else {
              expected[i] = value;
            }
          }
          for (uint32_t i = 0; i < (d16 ? 2u : 4u); ++i) {
            EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + 8 + i, 0), expected[i]);
            EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + 8 + i, 1), 0xdeadbeef);
          }
        }
  }
}

TEST_P(BufferFormatExecutionTest, RdnaNarrowTypedStoresBypassDescriptorSwizzle) {
  if (GetParam() != ROCJITSU_CODE_ARCH_RDNA3 && GetParam() != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      GetParam() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Native qualification covers the Gfx11 format table";
  auto decoder = Decoder::create(GetParam());
  wf->set_exec(1);
  for (bool d16 : {false, true}) {
    descriptor(18, 4, 0); // A typed format supplies identity despite this DST_SEL.
    cu->write_vgpr(wf->vgpr_alloc().base + 8, 0, d16 ? 0x00220011 : 0x11);
    cu->write_vgpr(wf->vgpr_alloc().base + 9, 0, 0x22);
    memory.write32(0x1000, 0xa5a5a5a5);
    std::unique_ptr<Instruction> inst;
    const uint16_t opcode = d16 ? 13 : 5;
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
      const auto words = rdna4::build_vbuffer(
          opcode + 128, {.soffset = rdna4::OPR_SREG_M0_NULL, .vdata = 8, .rsrc = 4, .format = 18});
      inst.reset(decode_valid(*decoder, words.data()));
    } else {
      const auto words =
          rdna3::build_mtbuf(opcode, {.format = 18, .vdata = 8, .srsrc = 1, .soffset = 128});
      inst.reset(decode_valid(*decoder, words.data()));
    }
    ASSERT_NE(inst, nullptr);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf), amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(memory.read32(0x1000), 0xa5a52211u);
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
  cu->write_sgpr(sb + 7, 4 | (5 << 3) | (6 << 6) | (60u << 12) | (3u << 28));
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
