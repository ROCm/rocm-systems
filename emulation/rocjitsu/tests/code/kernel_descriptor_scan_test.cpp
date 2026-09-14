// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kernel_descriptor_scan_test.cpp
/// @brief Unit tests for scan_kernel_descriptors -- the ELF walk shared by DBT
///        (translate_image) and DBI (instrumentor). Pins the discovery contract
///        directly, using the shared gfx950 fixture builder. Multi-kernel and
///        stripped/.dynsym discovery are exercised by tests/dbt/translate_test.cpp.

#include "rocjitsu/code/kernel_descriptor_scan.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/code_object.h"

#include "../dbi_test_util.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

using namespace rocjitsu::test;
using KD = rocr::llvm::amdhsa::kernel_descriptor_t;

// scan the fixture image using its own .text section coordinates.
std::vector<KernelDescriptorInfo> scan_via_text_section(const std::vector<uint8_t> &image) {
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_FALSE(obj.text_sections().empty());
  const Section *text = obj.text_sections().front();
  return scan_kernel_descriptors({image.data(), image.size()}, text->sectionOffset(), text->size());
}

// The single fixture kernel is located with name, file offset, .text-relative
// entry, and raw descriptor bytes all decoded correctly.
TEST(KernelDescriptorScan, SingleKernelDecodesAllFields) {
  const auto image = make_gfx950_kernel_elf({kMovV3V2, 0xbf810000u}, /*private_bytes=*/256);
  const auto found = scan_via_text_section(image);
  ASSERT_EQ(found.size(), 1u);
  EXPECT_EQ(found[0].kernel_name, "test_kernel");
  EXPECT_EQ(found[0].entry_text_offset, 0u); // fixture entry is at .text offset 0
  EXPECT_EQ(found[0].descriptor.private_segment_fixed_size, 256u);
  // The reported file offset points at the actual descriptor bytes.
  ASSERT_LE(found[0].descriptor_file_offset + sizeof(KD), image.size());
  KD at_offset{};
  std::memcpy(&at_offset, image.data() + found[0].descriptor_file_offset, sizeof(KD));
  EXPECT_EQ(at_offset.private_segment_fixed_size, 256u);
}

// A descriptor whose decoded entry lands outside .text is dropped. Rewrite the
// fixture descriptor's entry so it points at the descriptor's own (.rodata)
// address instead of .text, then confirm the walk excludes it.
TEST(KernelDescriptorScan, EntryOutsideTextIsDropped) {
  auto image = make_gfx950_kernel_elf({kMovV3V2, 0xbf810000u}, /*private_bytes=*/64);
  const auto located = scan_via_text_section(image);
  ASSERT_EQ(located.size(), 1u);

  KD desc{};
  std::memcpy(&desc, image.data() + located[0].descriptor_file_offset, sizeof(KD));
  desc.kernel_code_entry_byte_offset = 0; // entry now resolves to the .rodata KD vaddr
  std::memcpy(image.data() + located[0].descriptor_file_offset, &desc, sizeof(KD));

  EXPECT_TRUE(scan_via_text_section(image).empty());
}

// A too-small buffer is not a valid ELF: no descriptors, no crash.
TEST(KernelDescriptorScan, TooSmallImageReturnsEmpty) {
  std::vector<uint8_t> tiny(8, 0);
  EXPECT_TRUE(scan_kernel_descriptors({tiny.data(), tiny.size()}, 0x100, 0x8).empty());
}

// A descriptor symbol whose name runs to the end of its string table with no
// in-bounds NUL terminator is rejected, not read past the table boundary.
TEST(KernelDescriptorScan, UnterminatedDescriptorNameIsRejected) {
  const auto image =
      make_gfx950_unterminated_kd_name_elf({kMovV3V2, 0xbf810000u}, /*private_bytes=*/64);
  EXPECT_TRUE(scan_via_text_section(image).empty());
}

// .text (offset, size) from a pristine build, valid for a same-layout hostile
// variant whose own headers cannot be trusted to resolve them.
std::pair<uint64_t, uint64_t> clean_text_coords(const std::vector<uint32_t> &code) {
  const auto image = make_gfx950_kernel_elf(code, /*private_bytes=*/64);
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_FALSE(obj.text_sections().empty());
  const Section *text = obj.text_sections().front();
  return {text->sectionOffset(), text->size()};
}

// A section-header table whose declared extent overflows is rejected before any
// section pointer is formed: no descriptors and no out-of-bounds read.
TEST(KernelDescriptorScan, WrappingSectionHeaderTableIsRejected) {
  const std::vector<uint32_t> code = {kMovV3V2, 0xbf810000u};
  const auto [text_off, text_sz] = clean_text_coords(code);
  const auto image = make_gfx950_wrapping_shoff_elf(code, /*private_bytes=*/64);
  EXPECT_TRUE(scan_kernel_descriptors({image.data(), image.size()}, text_off, text_sz).empty());
}

// A symbol-table section whose declared extent overflows is skipped, not read past;
// the intact .text still lets discovery resolve the text base first.
TEST(KernelDescriptorScan, WrappingSymtabRangeIsRejected) {
  const std::vector<uint32_t> code = {kMovV3V2, 0xbf810000u};
  const auto [text_off, text_sz] = clean_text_coords(code);
  const auto image = make_gfx950_wrapping_symtab_elf(code, /*private_bytes=*/64);
  EXPECT_TRUE(scan_kernel_descriptors({image.data(), image.size()}, text_off, text_sz).empty());
}

// A `.kd` whose 64-byte descriptor extends past its owning section's sh_size into the
// adjacent section is rejected: the descriptor must be bounded by its section, not just
// the image, so it cannot be returned (and later mutated) across the section boundary.
TEST(KernelDescriptorScan, DescriptorCrossingOwningSectionIsRejected) {
  const std::vector<uint32_t> code = {kMovV3V2, 0xbf810000u};
  const auto [text_off, text_sz] = clean_text_coords(code);
  const auto image = make_gfx950_kd_crossing_section_elf(code, /*private_bytes=*/64);
  EXPECT_TRUE(scan_kernel_descriptors({image.data(), image.size()}, text_off, text_sz).empty());
}

// When no section matches the requested (text_offset, text_size), the walk cannot
// resolve .text's base address and returns nothing.
TEST(KernelDescriptorScan, NoMatchingTextSectionReturnsEmpty) {
  const auto image = make_gfx950_kernel_elf({kMovV3V2, 0xbf810000u}, /*private_bytes=*/64);
  const auto found = scan_kernel_descriptors({image.data(), image.size()},
                                             /*text_offset=*/0xDEAD, /*text_size=*/0x4);
  EXPECT_TRUE(found.empty());
}

// RDNA opts into Wave32 via COMPUTE_PGM_RSRC1's ENABLE_WAVEFRONT_SIZE32; a clear bit
// means Wave64. CDNA is always Wave64, gfx1250 always Wave32.
TEST(KernelWavefrontSize, Rdna4HonorsEnableWavefrontSize32Bit) {
  rocr::llvm::amdhsa::kernel_descriptor_t desc{};
  EXPECT_EQ(kernel_wavefront_size(ROCJITSU_CODE_ARCH_RDNA4, desc), 64); // bit clear
  AMDHSA_BITS_SET(desc.kernel_code_properties,
                  rocr::llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 1);
  EXPECT_EQ(kernel_wavefront_size(ROCJITSU_CODE_ARCH_RDNA4, desc), 32); // bit set
  EXPECT_EQ(kernel_wavefront_size(ROCJITSU_CODE_ARCH_CDNA4, desc), 64);
  EXPECT_EQ(kernel_wavefront_size(ROCJITSU_CODE_ARCH_CDNA5, desc), 32);
}

// The AMDHSA descriptor VGPR granule is wave-size dependent on RDNA: 8 for Wave32,
// 4 for Wave64. Using the Wave32 granule for a Wave64 kernel doubles the decoded
// allocation.
TEST(KernelDescriptorVgprGranule, RdnaIsWaveSizeDependent) {
  EXPECT_EQ(descriptor_vgpr_granularity_for_wavefront(ROCJITSU_CODE_ARCH_RDNA4, 32), 8u);
  EXPECT_EQ(descriptor_vgpr_granularity_for_wavefront(ROCJITSU_CODE_ARCH_RDNA4, 64), 4u);
  EXPECT_EQ(descriptor_vgpr_granularity_for_wavefront(ROCJITSU_CODE_ARCH_CDNA4, 64), 8u);
  EXPECT_EQ(descriptor_vgpr_granularity_for_wavefront(ROCJITSU_CODE_ARCH_CDNA1, 64), 4u);
  EXPECT_EQ(descriptor_vgpr_granularity_for_wavefront(ROCJITSU_CODE_ARCH_CDNA5, 32), 16u);
}

// Paired Wave32/Wave64 example: an RDNA4 descriptor with GRANULATED_WORKITEM_VGPR_COUNT=0
// declares v0:v3 (4 VGPRs) under Wave64 but v0:v7 (8) under Wave32. Decoding a Wave64
// kernel with the Wave32 granule would claim v4:v7 as allocated -- exactly the range a
// mis-decode could hand to the SGPR spill bridge.
TEST(KernelDescriptorVgprGranule, Rdna4Wave64ZeroGranulatedCountDeclaresFourVgprs) {
  constexpr uint32_t granulated = 0;

  rocr::llvm::amdhsa::kernel_descriptor_t wave64{}; // ENABLE_WAVEFRONT_SIZE32 clear
  const uint32_t g64 = descriptor_vgpr_granularity_for_wavefront(
      ROCJITSU_CODE_ARCH_RDNA4, kernel_wavefront_size(ROCJITSU_CODE_ARCH_RDNA4, wave64));
  EXPECT_EQ((granulated + 1) * g64, 4u);

  rocr::llvm::amdhsa::kernel_descriptor_t wave32{};
  AMDHSA_BITS_SET(wave32.kernel_code_properties,
                  rocr::llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 1);
  const uint32_t g32 = descriptor_vgpr_granularity_for_wavefront(
      ROCJITSU_CODE_ARCH_RDNA4, kernel_wavefront_size(ROCJITSU_CODE_ARCH_RDNA4, wave32));
  EXPECT_EQ((granulated + 1) * g32, 8u);
}

} // namespace
} // namespace rocjitsu
