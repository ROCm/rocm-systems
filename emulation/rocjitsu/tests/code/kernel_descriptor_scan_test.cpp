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

// When no section matches the requested (text_offset, text_size), the walk cannot
// resolve .text's base address and returns nothing.
TEST(KernelDescriptorScan, NoMatchingTextSectionReturnsEmpty) {
  const auto image = make_gfx950_kernel_elf({kMovV3V2, 0xbf810000u}, /*private_bytes=*/64);
  const auto found = scan_kernel_descriptors({image.data(), image.size()},
                                             /*text_offset=*/0xDEAD, /*text_size=*/0x4);
  EXPECT_TRUE(found.empty());
}

} // namespace
} // namespace rocjitsu
