// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kernel_descriptor_scan_test.cpp
/// @brief Unit tests for scan_kernel_descriptors -- the ELF walk shared by DBT
///        (translate_image) and DBI (instrumentor). Pins the discovery contract
///        directly, using the shared gfx950 fixture builder. Multi-kernel and
///        stripped/.dynsym discovery are exercised by tests/dbt/translate_test.cpp.

#include "rocjitsu/code/kernel_descriptor_scan.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/code_object.h"

#include "../dbi_test_util.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

using namespace rocjitsu::test;
using KD = rocjitsu::amdhsa::kernel_descriptor_t;

// scan the fixture image using its own .text section coordinates.
std::vector<KernelDescriptorInfo> scan_via_text_section(const std::vector<uint8_t> &image) {
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_FALSE(obj.text_sections().empty());
  const Section *text = obj.text_sections().front();
  return scan_kernel_descriptors({image.data(), image.size()}, text->sectionOffset(), text->size());
}

void rewrite_descriptor_entry(std::vector<uint8_t> &image, uint64_t descriptor_vaddr,
                              uint64_t text_vaddr, int64_t entry_offset) {
  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  std::vector<Elf64_Shdr> shdrs(ehdr.e_shnum);
  std::memcpy(shdrs.data(), image.data() + ehdr.e_shoff, shdrs.size() * sizeof(Elf64_Shdr));

  const std::vector<Elf64_Shdr>::iterator symtab =
      std::find_if(shdrs.begin(), shdrs.end(),
                   [](const Elf64_Shdr &shdr) { return shdr.sh_type == SHT_SYMTAB; });
  ASSERT_NE(symtab, shdrs.end());
  std::vector<Elf64_Sym> symbols(symtab->sh_size / sizeof(Elf64_Sym));
  std::memcpy(symbols.data(), image.data() + symtab->sh_offset, symbols.size() * sizeof(Elf64_Sym));
  ASSERT_GT(symbols.size(), 1u);
  Elf64_Sym &symbol = symbols[1];
  ASSERT_LT(symbol.st_shndx, shdrs.size());
  Elf64_Shdr &owner = shdrs[symbol.st_shndx];
  const uint64_t descriptor_offset = owner.sh_offset + symbol.st_value - owner.sh_addr;
  ASSERT_LE(descriptor_offset + sizeof(KD), image.size());

  const std::vector<Elf64_Shdr>::iterator text =
      std::find_if(shdrs.begin(), shdrs.end(), [](const Elf64_Shdr &shdr) {
        return (shdr.sh_flags & (SHF_ALLOC | SHF_EXECINSTR)) == (SHF_ALLOC | SHF_EXECINSTR);
      });
  ASSERT_NE(text, shdrs.end());
  text->sh_addr = text_vaddr;
  owner.sh_addr = descriptor_vaddr;
  symbol.st_value = descriptor_vaddr;

  KD descriptor{};
  std::memcpy(&descriptor, image.data() + descriptor_offset, sizeof(descriptor));
  descriptor.kernel_code_entry_byte_offset = entry_offset;
  std::memcpy(image.data() + descriptor_offset, &descriptor, sizeof(descriptor));
  std::memcpy(image.data() + symtab->sh_offset, symbols.data(), symbols.size() * sizeof(Elf64_Sym));
  std::memcpy(image.data() + ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
}

// The single fixture kernel is located with name, file offset, .text-relative
// entry, and raw descriptor bytes all decoded correctly.
TEST(KernelDescriptorScan, SingleKernelDecodesAllFields) {
  const std::vector<uint8_t> image =
      make_gfx950_kernel_elf({kMovV3V2, 0xbf810000u}, /*private_bytes=*/256);
  const std::vector<KernelDescriptorInfo> found = scan_via_text_section(image);
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
  std::vector<uint8_t> image =
      make_gfx950_kernel_elf({kMovV3V2, 0xbf810000u}, /*private_bytes=*/64);
  const std::vector<KernelDescriptorInfo> located = scan_via_text_section(image);
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
  const std::vector<uint8_t> image =
      make_gfx950_unterminated_kd_name_elf({kMovV3V2, 0xbf810000u}, /*private_bytes=*/64);
  EXPECT_TRUE(scan_via_text_section(image).empty());
}

// .text (offset, size) from a pristine build, valid for a same-layout hostile
// variant whose own headers cannot be trusted to resolve them.
std::pair<uint64_t, uint64_t> clean_text_coords(const std::vector<uint32_t> &code) {
  const std::vector<uint8_t> image = make_gfx950_kernel_elf(code, /*private_bytes=*/64);
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
  const std::vector<uint8_t> image = make_gfx950_wrapping_shoff_elf(code, /*private_bytes=*/64);
  EXPECT_TRUE(scan_kernel_descriptors({image.data(), image.size()}, text_off, text_sz).empty());
}

// A symbol-table section whose declared extent overflows is skipped, not read past;
// the intact .text still lets discovery resolve the text base first.
TEST(KernelDescriptorScan, WrappingSymtabRangeIsRejected) {
  const std::vector<uint32_t> code = {kMovV3V2, 0xbf810000u};
  const auto [text_off, text_sz] = clean_text_coords(code);
  const std::vector<uint8_t> image = make_gfx950_wrapping_symtab_elf(code, /*private_bytes=*/64);
  EXPECT_TRUE(scan_kernel_descriptors({image.data(), image.size()}, text_off, text_sz).empty());
}

TEST(KernelDescriptorScan, UnalignedImageSpanDoesNotRequireNativeElfAlignment) {
  const std::vector<uint32_t> code = {kMovV3V2, 0xbf810000u};
  const auto [text_off, text_sz] = clean_text_coords(code);
  const std::vector<uint8_t> image = make_gfx950_kernel_elf(code, 64);
  std::vector<uint8_t> storage(image.size() + 1);
  std::memcpy(storage.data() + 1, image.data(), image.size());
  const std::vector<KernelDescriptorInfo> found =
      scan_kernel_descriptors(std::span(storage).subspan(1), text_off, text_sz);
  ASSERT_EQ(found.size(), 1u);
  EXPECT_EQ(found.front().kernel_name, "test_kernel");
}

TEST(KernelDescriptorScan, UnalignedSectionAndSymbolTablesAreReadAsBytes) {
  const std::vector<uint32_t> code = {kMovV3V2, 0xbf810000u};
  const auto [text_off, text_sz] = clean_text_coords(code);
  std::vector<uint8_t> image = make_gfx950_kernel_elf(code, 64);
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  std::vector<Elf64_Shdr> sections(header.e_shnum);
  std::memcpy(sections.data(), image.data() + header.e_shoff, sections.size() * sizeof(Elf64_Shdr));
  size_t symbol_tables = 0;
  for (Elf64_Shdr &section : sections) {
    if (section.sh_type != SHT_SYMTAB)
      continue;
    const std::vector<uint8_t> symbols(image.begin() + section.sh_offset,
                                       image.begin() + section.sh_offset + section.sh_size);
    while (image.size() % 8 != 1)
      image.push_back(0);
    section.sh_offset = image.size();
    image.resize(image.size() + symbols.size());
    std::memcpy(image.data() + section.sh_offset, symbols.data(), symbols.size());
    EXPECT_NE(section.sh_offset % alignof(Elf64_Sym), 0u);
    ++symbol_tables;
  }
  ASSERT_EQ(symbol_tables, 1u);
  while (image.size() % 8 != 1)
    image.push_back(0);
  header.e_shoff = image.size();
  const std::span<const std::byte> bytes = std::as_bytes(std::span(sections));
  image.resize(image.size() + bytes.size());
  std::memcpy(image.data() + header.e_shoff, bytes.data(), bytes.size());
  EXPECT_NE(header.e_shoff % alignof(Elf64_Shdr), 0u);
  std::memcpy(image.data(), &header, sizeof(header));
  const std::vector<KernelDescriptorInfo> found = scan_kernel_descriptors(image, text_off, text_sz);
  ASSERT_EQ(found.size(), 1u);
  EXPECT_EQ(found.front().kernel_name, "test_kernel");
}

TEST(KernelDescriptorScan, EntryDisplacementOutsideTextIsRejected) {
  const std::vector<uint32_t> code = {kMovV3V2, 0xbf810000u};
  const auto [text_off, text_sz] = clean_text_coords(code);
  std::vector<uint8_t> image = make_gfx950_kernel_elf(code, 64);
  const std::vector<KernelDescriptorInfo> original =
      scan_kernel_descriptors(image, text_off, text_sz);
  ASSERT_EQ(original.size(), 1u);
  KD descriptor = original.front().descriptor;
  descriptor.kernel_code_entry_byte_offset = INT64_MAX;
  std::memcpy(image.data() + original.front().descriptor_file_offset, &descriptor,
              sizeof(descriptor));
  EXPECT_TRUE(scan_kernel_descriptors(image, text_off, text_sz).empty());
}

// A `.kd` whose 64-byte descriptor extends past its owning section's sh_size into the
// adjacent section is rejected: the descriptor must be bounded by its section, not just
// the image, so it cannot be returned (and later mutated) across the section boundary.
TEST(KernelDescriptorScan, DescriptorCrossingOwningSectionIsRejected) {
  const std::vector<uint32_t> code = {kMovV3V2, 0xbf810000u};
  const auto [text_off, text_sz] = clean_text_coords(code);
  const std::vector<uint8_t> image =
      make_gfx950_kd_crossing_section_elf(code, /*private_bytes=*/64);
  EXPECT_TRUE(scan_kernel_descriptors({image.data(), image.size()}, text_off, text_sz).empty());
}

// When no section matches the requested (text_offset, text_size), the walk cannot
// resolve .text's base address and returns nothing.
TEST(KernelDescriptorScan, NoMatchingTextSectionReturnsEmpty) {
  const std::vector<uint8_t> image =
      make_gfx950_kernel_elf({kMovV3V2, 0xbf810000u}, /*private_bytes=*/64);
  const std::vector<KernelDescriptorInfo> found =
      scan_kernel_descriptors({image.data(), image.size()},
                              /*text_offset=*/0xDEAD, /*text_size=*/0x4);
  EXPECT_TRUE(found.empty());
}

// Forming the magnitude of INT64_MIN must avoid signed negation overflow while
// still resolving the entry at the first .text byte.
TEST(KernelDescriptorScan, Int64MinEntryOffsetResolvesWithoutSignedOverflow) {
  std::vector<uint8_t> image =
      make_gfx950_kernel_elf({kMovV3V2, 0xbf810000u}, /*private_bytes=*/64);
  constexpr uint64_t kTextVaddr = 0x1000;
  constexpr uint64_t kDescriptorVaddr = (uint64_t{1} << 63) | kTextVaddr;
  rewrite_descriptor_entry(image, kDescriptorVaddr, kTextVaddr,
                           std::numeric_limits<int64_t>::min());

  const std::vector<KernelDescriptorInfo> found = scan_via_text_section(image);
  ASSERT_EQ(found.size(), 1u);
  EXPECT_EQ(found[0].entry_text_offset, 0u);
}

// Unsigned addition would wrap this descriptor entry precisely into .text, so
// only the explicit positive-overflow guard keeps the false descriptor out.
TEST(KernelDescriptorScan, PositiveEntryOffsetOverflowIsRejected) {
  std::vector<uint8_t> image =
      make_gfx950_kernel_elf({kMovV3V2, 0xbf810000u}, /*private_bytes=*/64);
  rewrite_descriptor_entry(image, std::numeric_limits<uint64_t>::max() - 0xfff, 0x1000, 0x2000);
  EXPECT_TRUE(scan_via_text_section(image).empty());
}

// Unsigned subtraction would wrap this descriptor entry precisely into .text,
// so only the explicit negative-underflow guard keeps the false descriptor out.
TEST(KernelDescriptorScan, NegativeEntryOffsetUnderflowIsRejected) {
  std::vector<uint8_t> image =
      make_gfx950_kernel_elf({kMovV3V2, 0xbf810000u}, /*private_bytes=*/64);
  rewrite_descriptor_entry(image, 0x1000, std::numeric_limits<uint64_t>::max() - 0xffff, -0x11000);
  EXPECT_TRUE(scan_via_text_section(image).empty());
}

// The raw descriptor scanner can read record prefixes from a larger stride,
// but the code-object parser rejects layouts its patcher cannot preserve.
TEST(KernelDescriptorScan, StridedSectionHeaderTableIsReadSafely) {
  std::vector<uint8_t> image =
      make_gfx950_kernel_elf({kMovV3V2, 0xbf810000u}, /*private_bytes=*/64);
  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  std::vector<Elf64_Shdr> shdrs(ehdr.e_shnum);
  std::memcpy(shdrs.data(), image.data() + ehdr.e_shoff, shdrs.size() * sizeof(Elf64_Shdr));

  constexpr uint16_t kStride = 2 * sizeof(Elf64_Shdr);
  const uint64_t strided_offset = image.size();
  image.resize(image.size() + static_cast<size_t>(ehdr.e_shnum) * kStride, 0);
  for (size_t section_index = 0; section_index < shdrs.size(); ++section_index) {
    std::memcpy(image.data() + strided_offset + section_index * kStride, &shdrs[section_index],
                sizeof(Elf64_Shdr));
  }
  ehdr.e_shoff = strided_offset;
  ehdr.e_shentsize = kStride;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  const auto [text_offset, text_size] = clean_text_coords({kMovV3V2, 0xbf810000u});
  EXPECT_EQ(scan_kernel_descriptors({image.data(), image.size()}, text_offset, text_size).size(),
            1u);
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

// Raw descriptor discovery handles larger symbol strides safely, while the
// code-object parser rejects tables its patcher cannot relocate.
TEST(KernelDescriptorScan, StridedSymbolTableIsReadSafely) {
  std::vector<uint8_t> image =
      make_gfx950_kernel_elf({kMovV3V2, 0xbf810000u}, /*private_bytes=*/64);
  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  std::vector<Elf64_Shdr> shdrs(ehdr.e_shnum);
  std::memcpy(shdrs.data(), image.data() + ehdr.e_shoff, shdrs.size() * sizeof(Elf64_Shdr));
  const std::vector<Elf64_Shdr>::iterator symbol_table =
      std::find_if(shdrs.begin(), shdrs.end(),
                   [](const Elf64_Shdr &shdr) { return shdr.sh_type == SHT_SYMTAB; });
  ASSERT_NE(symbol_table, shdrs.end());
  std::vector<Elf64_Sym> symbols(symbol_table->sh_size / symbol_table->sh_entsize);
  std::memcpy(symbols.data(), image.data() + symbol_table->sh_offset,
              symbols.size() * sizeof(Elf64_Sym));

  constexpr uint64_t kStride = 2 * sizeof(Elf64_Sym);
  symbol_table->sh_offset = image.size();
  symbol_table->sh_entsize = kStride;
  symbol_table->sh_size = symbols.size() * kStride;
  image.resize(image.size() + symbol_table->sh_size, 0);
  for (size_t symbol_index = 0; symbol_index < symbols.size(); ++symbol_index) {
    std::memcpy(image.data() + symbol_table->sh_offset + symbol_index * kStride,
                &symbols[symbol_index], sizeof(Elf64_Sym));
  }
  std::memcpy(image.data() + ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  const auto [text_offset, text_size] = clean_text_coords({kMovV3V2, 0xbf810000u});
  EXPECT_EQ(scan_kernel_descriptors({image.data(), image.size()}, text_offset, text_size).size(),
            1u);
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

// RDNA opts into Wave32 via COMPUTE_PGM_RSRC1's ENABLE_WAVEFRONT_SIZE32; a clear bit
// means Wave64. CDNA is always Wave64, gfx1250 always Wave32.
TEST(KernelWavefrontSize, Rdna4HonorsEnableWavefrontSize32Bit) {
  rocjitsu::amdhsa::kernel_descriptor_t desc{};
  EXPECT_EQ(kernel_wavefront_size(ROCJITSU_CODE_ARCH_RDNA4, desc), 64); // bit clear
  RJ_AMDHSA_BITS_SET(desc.kernel_code_properties,
                     rocjitsu::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 1);
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

  rocjitsu::amdhsa::kernel_descriptor_t wave64{}; // ENABLE_WAVEFRONT_SIZE32 clear
  const uint32_t g64 = descriptor_vgpr_granularity_for_wavefront(
      ROCJITSU_CODE_ARCH_RDNA4, kernel_wavefront_size(ROCJITSU_CODE_ARCH_RDNA4, wave64));
  EXPECT_EQ((granulated + 1) * g64, 4u);

  rocjitsu::amdhsa::kernel_descriptor_t wave32{};
  RJ_AMDHSA_BITS_SET(wave32.kernel_code_properties,
                     rocjitsu::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 1);
  const uint32_t g32 = descriptor_vgpr_granularity_for_wavefront(
      ROCJITSU_CODE_ARCH_RDNA4, kernel_wavefront_size(ROCJITSU_CODE_ARCH_RDNA4, wave32));
  EXPECT_EQ((granulated + 1) * g32, 8u);
}

// The kernarg pointer's user-SGPR index is the summed width of whichever of
// private_segment_buffer (4), dispatch_ptr (2) and queue_ptr (2) precede it.
// Every consumer that loads through the pointer needs this index, so a wrong sum
// reads an unrelated register rather than failing.
TEST(KernargSegmentPtr, SlotIsTheWidthOfTheEnabledPredecessors) {
  using namespace rocjitsu::amdhsa;

  EXPECT_EQ(kernarg_segment_ptr_slot(KD{}), 0);

  KD buffer{};
  RJ_AMDHSA_BITS_SET(buffer.kernel_code_properties,
                     KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER, 1);
  EXPECT_EQ(kernarg_segment_ptr_slot(buffer), 4);

  KD dispatch{};
  RJ_AMDHSA_BITS_SET(dispatch.kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR,
                     1);
  EXPECT_EQ(kernarg_segment_ptr_slot(dispatch), 2);

  KD queue{};
  RJ_AMDHSA_BITS_SET(queue.kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR, 1);
  EXPECT_EQ(kernarg_segment_ptr_slot(queue), 2);

  KD buffer_and_dispatch = buffer;
  RJ_AMDHSA_BITS_SET(buffer_and_dispatch.kernel_code_properties,
                     KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR, 1);
  EXPECT_EQ(kernarg_segment_ptr_slot(buffer_and_dispatch), 6);

  KD all_three = buffer_and_dispatch;
  RJ_AMDHSA_BITS_SET(all_three.kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR,
                     1);
  EXPECT_EQ(kernarg_segment_ptr_slot(all_three), 8);
}

// The slot query answers where the pointer would go; the sgpr query answers where
// it is. Keeping them distinct lets a caller inserting the pointer ask for the
// index before setting the bit, without suppressing a nullopt it knows is wrong.
TEST(KernargSegmentPtr, SgprIsEmptyUntilTheDescriptorEnablesThePointer) {
  using namespace rocjitsu::amdhsa;

  KD desc{};
  RJ_AMDHSA_BITS_SET(desc.kernel_code_properties,
                     KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER, 1);
  EXPECT_FALSE(has_kernarg_segment_ptr(desc));
  EXPECT_FALSE(kernarg_segment_ptr_sgpr(desc).has_value());
  EXPECT_EQ(kernarg_segment_ptr_slot(desc), 4); // still answers, and with the right slot

  RJ_AMDHSA_BITS_SET(desc.kernel_code_properties,
                     KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
  EXPECT_TRUE(has_kernarg_segment_ptr(desc));
  ASSERT_TRUE(kernarg_segment_ptr_sgpr(desc).has_value());
  EXPECT_EQ(*kernarg_segment_ptr_sgpr(desc), 4);
}

// Length and offset occupy one descriptor field. A decoder that confused them
// would report "preloading is on" for a kernel whose offset is non-zero and
// length zero, which is not a preloading kernel at all.
TEST(KernargPreload, LengthAndOffsetDecodeIndependently) {
  KD desc{};
  EXPECT_EQ(kernarg_preload_length(desc), 0u);
  EXPECT_EQ(kernarg_preload_offset(desc), 0u);

  RJ_AMDHSA_BITS_SET(desc.kernarg_preload, rocjitsu::amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 3);
  EXPECT_EQ(kernarg_preload_length(desc), 3u);
  EXPECT_EQ(kernarg_preload_offset(desc), 0u);

  RJ_AMDHSA_BITS_SET(desc.kernarg_preload, rocjitsu::amdhsa::KERNARG_PRELOAD_SPEC_OFFSET, 5);
  EXPECT_EQ(kernarg_preload_length(desc), 3u);
  EXPECT_EQ(kernarg_preload_offset(desc), 5u);
}

} // namespace

// The dense system-SGPR block following the user block. DBT uses it as the range
// to repair when it inserts a kernarg pointer; DBI as a floor for framework
// storage. Distinct values per field so a dropped or double-counted term shows.
TEST(KernelDescriptorScan, InitialSgprCountFoldsTheSystemBlockOntoTheUserBlock) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;

  KD none{};
  set_kernel_descriptor_user_sgpr_count(kArch, none, 6);
  EXPECT_EQ(kernel_descriptor_initial_sgpr_count(kArch, none), 6u);

  KD ids = none;
  RJ_AMDHSA_BITS_SET(ids.compute_pgm_rsrc2,
                     rocjitsu::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1);
  RJ_AMDHSA_BITS_SET(ids.compute_pgm_rsrc2,
                     rocjitsu::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z, 1);
  EXPECT_EQ(kernel_descriptor_initial_sgpr_count(kArch, ids), 8u);

  // WORKGROUP_INFO follows the enabled dimensions and is the term the DBI copy
  // of this walk was missing before it was shared.
  KD info = ids;
  RJ_AMDHSA_BITS_SET(info.compute_pgm_rsrc2,
                     rocjitsu::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO, 1);
  EXPECT_EQ(kernel_descriptor_initial_sgpr_count(kArch, info), 9u);

  // ENABLE_PRIVATE_SEGMENT is deliberately excluded.
  KD priv = info;
  RJ_AMDHSA_BITS_SET(priv.compute_pgm_rsrc2,
                     rocjitsu::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 1);
  EXPECT_EQ(kernel_descriptor_initial_sgpr_count(kArch, priv), 9u);
}

} // namespace rocjitsu
