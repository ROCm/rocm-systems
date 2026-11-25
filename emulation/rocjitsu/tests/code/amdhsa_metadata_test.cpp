// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../amdhsa_metadata_test_util.h"
#include "../dbi_test_util.h"

#include "rocjitsu/code/amdhsa_metadata.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

std::vector<KernelDescriptorInfo> descriptors(const std::vector<uint8_t> &image) {
  const AmdGpuCodeObject object(image.data(), image.size());
  if (!object.is_valid() || object.text_sections().size() != 1) {
    return {};
  }
  const Section &text = *object.text_sections().front();
  return scan_kernel_descriptors(image, text.sectionOffset(), text.size());
}

std::vector<uint8_t> plain_kernel() {
  return test::make_kernarg_kernel_elf({0xbf800000u, 0xbf810000u}, 4096,
                                       EF_AMDGPU_MACH_AMDGCN_GFX950, 20);
}

TEST(AmdhsaMetadata, UpdatesSelectedKernelDescriptorAndNoteTogether) {
  const std::vector<uint8_t> image = test::with_resource_metadata(
      test::make_gfx950_two_kernel_elf({0xbf800000u, 0xbf810000u}, 4096),
      {{"test_kernel0", 0, 16, 4}, {"test_kernel1", 0, 16, 4}});
  const std::vector<KernelDescriptorInfo> kernels = descriptors(image);
  ASSERT_EQ(kernels.size(), 2u);
  const AmdGpuCodeObject object(image.data(), image.size());
  CodeObjectPatcher patcher(object);
  amdhsa::kernel_descriptor_t descriptor = kernels.front().descriptor;
  descriptor.kernarg_size = 24;
  RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                     amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 5);
  RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                     amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 1);
  std::string error;
  ASSERT_TRUE(patcher.patch_kernel_resources(kernels.front(), descriptor, 40, 8, &error)) << error;
  const FailureOr<amdhsa::MetadataDocument> metadata =
      amdhsa::MetadataDocument::parse(patcher.image_bytes(), &error);
  ASSERT_TRUE(metadata.succeeded()) << error;
  const amdhsa::MetadataValue *entries = metadata.value().root().field("amdhsa.kernels");
  ASSERT_NE(entries, nullptr);
  ASSERT_EQ(entries->array.size(), 2u);
  EXPECT_EQ(entries->array[0].integer_field(".kernarg_segment_size"), 24u);
  EXPECT_EQ(entries->array[0].integer_field(".sgpr_count"), 40u);
  EXPECT_EQ(entries->array[0].integer_field(".vgpr_count"), 8u);
  EXPECT_EQ(entries->array[1].integer_field(".sgpr_count"), 16u);
  EXPECT_EQ(entries->array[1].integer_field(".kernarg_segment_size"), 0u);
  EXPECT_EQ(entries->array[0].integer_field(".private_segment_fixed_size"), 4096u);
  EXPECT_TRUE(metadata.value().root().text_field_is("vendor.extra", "unchanged"));
  const std::vector<uint8_t> updated = patcher.emit();
  const std::vector<KernelDescriptorInfo> changed = descriptors(updated);
  ASSERT_EQ(changed.size(), 2u);
  EXPECT_EQ(std::memcmp(&changed[0].descriptor, &descriptor, sizeof(descriptor)), 0);
  EXPECT_EQ(std::memcmp(&changed[1].descriptor, &kernels[1].descriptor, sizeof(descriptor)), 0);
}

TEST(AmdhsaMetadata, WrapperOnlyUpdateRetainsRegisterMetadata) {
  const std::vector<uint8_t> image =
      test::with_resource_metadata(plain_kernel(), {{"test_kernel", 20, 16, 4}});
  const std::vector<KernelDescriptorInfo> kernels = descriptors(image);
  ASSERT_EQ(kernels.size(), 1u);
  const AmdGpuCodeObject object(image.data(), image.size());
  CodeObjectPatcher patcher(object);
  amdhsa::kernel_descriptor_t descriptor = kernels.front().descriptor;
  descriptor.kernarg_size = 40;
  std::string error;
  ASSERT_TRUE(patcher.patch_kernel_resources(kernels.front(), descriptor, std::nullopt,
                                             std::nullopt, &error))
      << error;
  const FailureOr<amdhsa::MetadataDocument> metadata =
      amdhsa::MetadataDocument::parse(patcher.image_bytes(), &error);
  ASSERT_TRUE(metadata.succeeded()) << error;
  const amdhsa::MetadataValue *entries = metadata.value().root().field("amdhsa.kernels");
  ASSERT_NE(entries, nullptr);
  ASSERT_EQ(entries->array.size(), 1u);
  EXPECT_EQ(entries->array[0].integer_field(".kernarg_segment_size"), 40u);
  EXPECT_EQ(entries->array[0].integer_field(".sgpr_count"), 16u);
  EXPECT_EQ(entries->array[0].integer_field(".vgpr_count"), 4u);
}

TEST(AmdhsaMetadata, ScratchGrowthUpdatesTheDescriptorAndNoteTogether) {
  const std::vector<uint8_t> image =
      test::with_resource_metadata(test::make_gfx950_kernel_elf({0xbf800000u, 0xbf810000u}, 4096),
                                   {{"test_kernel", 0, 16, 4, 4096}});
  const std::vector<KernelDescriptorInfo> kernels = descriptors(image);
  ASSERT_EQ(kernels.size(), 1u);
  const AmdGpuCodeObject object(image.data(), image.size());
  CodeObjectPatcher patcher(object);
  amdhsa::kernel_descriptor_t descriptor = kernels.front().descriptor;
  descriptor.kernarg_size = 24;
  descriptor.private_segment_fixed_size = 8192;
  std::string error;
  ASSERT_TRUE(patcher.patch_kernel_resources(kernels.front(), descriptor, std::nullopt,
                                             std::nullopt, &error))
      << error;
  const FailureOr<amdhsa::MetadataDocument> metadata =
      amdhsa::MetadataDocument::parse(patcher.image_bytes(), &error);
  ASSERT_TRUE(metadata.succeeded()) << error;
  const amdhsa::MetadataValue *entries = metadata.value().root().field("amdhsa.kernels");
  ASSERT_NE(entries, nullptr);
  ASSERT_EQ(entries->array.size(), 1u);
  EXPECT_EQ(entries->array.front().integer_field(".private_segment_fixed_size"), 8192u);
  EXPECT_EQ(entries->array.front().integer_field(".kernarg_segment_size"), 24u);
  const std::vector<KernelDescriptorInfo> updated = descriptors(patcher.emit());
  ASSERT_EQ(updated.size(), 1u);
  EXPECT_EQ(updated.front().descriptor.private_segment_fixed_size, 8192u);
}

TEST(AmdhsaMetadata, ScratchWidthFailureLeavesBothRepresentationsUnchanged) {
  const std::vector<uint8_t> image = test::with_resource_metadata_padding(
      test::make_gfx950_kernel_elf({0xbf800000u, 0xbf810000u}, 127),
      {{"test_kernel", 0, 16, 4, 127}}, 0);
  const std::vector<KernelDescriptorInfo> kernels = descriptors(image);
  ASSERT_EQ(kernels.size(), 1u);
  const AmdGpuCodeObject object(image.data(), image.size());
  CodeObjectPatcher patcher(object);
  amdhsa::kernel_descriptor_t descriptor = kernels.front().descriptor;
  descriptor.kernarg_size = 24;
  descriptor.private_segment_fixed_size = 128;
  std::string error;
  EXPECT_FALSE(patcher.patch_kernel_resources(kernels.front(), descriptor, std::nullopt,
                                              std::nullopt, &error));
  EXPECT_NE(error.find("wider"), std::string::npos);
  EXPECT_TRUE(std::ranges::equal(patcher.image_bytes(), image));
}

TEST(AmdhsaMetadata, WiderIntegerWithoutPaddingRejectsBeforeNoteOrDescriptorChanges) {
  const std::vector<uint8_t> image =
      test::with_resource_metadata_padding(plain_kernel(), {{"test_kernel", 20, 16, 4}}, 0);
  const std::vector<KernelDescriptorInfo> kernels = descriptors(image);
  ASSERT_EQ(kernels.size(), 1u);
  const AmdGpuCodeObject object(image.data(), image.size());
  CodeObjectPatcher patcher(object);
  amdhsa::kernel_descriptor_t descriptor = kernels.front().descriptor;
  descriptor.kernarg_size = 40;
  std::string error;
  EXPECT_FALSE(patcher.patch_kernel_resources(kernels.front(), descriptor, 128, 8, &error));
  EXPECT_NE(error.find("wider"), std::string::npos);
  EXPECT_TRUE(std::ranges::equal(patcher.image_bytes(), image));
}

TEST(AmdhsaMetadata, IntegerGrowthUsesOnlyExistingNotePaddingAndRetainsOtherEntries) {
  for (const uint32_t padding : {0u, 1u, 2u, 3u}) {
    for (const uint32_t number : {128u, 256u, 65536u}) {
      SCOPED_TRACE(padding);
      SCOPED_TRACE(number);
      const uint32_t growth = number == 128 ? 1 : number == 256 ? 2 : 4;
      std::vector<uint8_t> image = test::with_resource_metadata_padding(
          test::make_gfx950_two_kernel_elf({0xbf800000u, 0xbf810000u}, 4096),
          {{"test_kernel0", 20, 16, 4}, {"test_kernel1", 20, 16, 4}}, padding);
      const std::vector<KernelDescriptorInfo> kernels = descriptors(image);
      ASSERT_EQ(kernels.size(), 2u);
      const FailureOr<amdhsa::MetadataDocument> original_metadata =
          amdhsa::MetadataDocument::parse(image);
      ASSERT_TRUE(original_metadata.succeeded());
      const size_t payload_offset = original_metadata.value().payload().data() - image.data();
      const size_t payload_end = payload_offset + original_metadata.value().payload().size();
      std::fill(image.begin() + payload_end, image.begin() + payload_end + padding, uint8_t{0xa5});
      const amdhsa::MetadataValue &original_entry =
          original_metadata.value().root().field("amdhsa.kernels")->array.front();
      const amdhsa::MetadataValue &field = *original_entry.field(".kernarg_segment_size");
      ASSERT_EQ(field.integer_bytes, 0u);
      const AmdGpuCodeObject object(image.data(), image.size());
      CodeObjectPatcher patcher(object);
      amdhsa::kernel_descriptor_t descriptor = kernels.front().descriptor;
      descriptor.kernarg_size = number;
      std::string error;
      const bool changed = patcher.patch_kernel_resources(kernels.front(), descriptor, std::nullopt,
                                                          std::nullopt, &error);
      ASSERT_EQ(changed, growth <= padding) << error;
      if (!changed) {
        EXPECT_NE(error.find("note padding permits"), std::string::npos);
        EXPECT_TRUE(std::ranges::equal(patcher.image_bytes(), image));
        continue;
      }
      EXPECT_EQ(patcher.image_bytes().size(), image.size());
      const FailureOr<amdhsa::MetadataDocument> metadata =
          amdhsa::MetadataDocument::parse(patcher.image_bytes());
      ASSERT_TRUE(metadata.succeeded());
      const amdhsa::MetadataValue &entries = *metadata.value().root().field("amdhsa.kernels");
      EXPECT_EQ(entries.array.front().integer_field(".kernarg_segment_size"), number);
      EXPECT_EQ(entries.array[1].integer_field(".kernarg_segment_size"), 20u);
      EXPECT_EQ(entries.array[1].integer_field(".sgpr_count"), 16u);
      EXPECT_EQ(entries.array[1].integer_field(".vgpr_count"), 4u);
      EXPECT_TRUE(metadata.value().root().text_field_is("vendor.extra", "unchanged"));
      const std::span<const uint8_t> before = original_metadata.value().payload();
      const std::span<const uint8_t> after = metadata.value().payload();
      EXPECT_EQ(after.size(), before.size() + growth);
      EXPECT_TRUE(std::ranges::equal(
          std::span(image).subspan(payload_end + growth, padding - growth),
          patcher.image_bytes().subspan(payload_end + growth, padding - growth)));
      EXPECT_TRUE(std::ranges::equal(before.first(field.offset), after.first(field.offset)));
      EXPECT_TRUE(std::ranges::equal(before.subspan(field.offset + 1),
                                     after.subspan(field.offset + 1 + growth)));
      // ELF tables, other sections and descriptor addresses never move.
      Elf64_Ehdr header = {};
      std::memcpy(&header, image.data(), sizeof(header));
      EXPECT_EQ(std::memcmp(image.data(), patcher.image_bytes().data(), sizeof(header)), 0);
      EXPECT_EQ(std::memcmp(image.data() + header.e_shoff,
                            patcher.image_bytes().data() + header.e_shoff,
                            header.e_shnum * header.e_shentsize),
                0);
    }
  }
}

TEST(AmdhsaMetadata, DescriptorOverlappingNoteFramingRejectsWithoutMutation) {
  std::vector<uint8_t> image =
      test::with_resource_metadata_padding(plain_kernel(), {{"test_kernel", 20, 16, 4}}, 1);
  const std::vector<uint8_t> original = image;
  const std::vector<KernelDescriptorInfo> kernels = descriptors(image);
  ASSERT_EQ(kernels.size(), 1u);
  const FailureOr<amdhsa::MetadataDocument> document = amdhsa::MetadataDocument::parse(image);
  ASSERT_TRUE(document.succeeded());
  const size_t payload = document.value().payload().data() - image.data();
  ASSERT_GE(payload, sizeof(amdhsa::kernel_descriptor_t));
  KernelDescriptorInfo forged = kernels.front();
  // End at the payload start: payload-only overlap checks miss the note size word.
  forged.descriptor_file_offset = payload - sizeof(amdhsa::kernel_descriptor_t);
  amdhsa::kernel_descriptor_t descriptor = forged.descriptor;
  descriptor.kernarg_size = 128;
  std::string error;
  EXPECT_TRUE(amdhsa::update_resource_metadata(image, forged, descriptor, std::nullopt,
                                               std::nullopt, &error)
                  .failed());
  EXPECT_NE(error.find("framing or padding overlaps"), std::string::npos) << error;
  EXPECT_EQ(image, original);
}

TEST(AmdhsaMetadata, MissingResourceFieldRejectsBeforeAnyWrites) {
  std::vector<uint8_t> image =
      test::with_resource_metadata(plain_kernel(), {{"test_kernel", 20, 16, 4}});
  constexpr std::string_view key = ".sgpr_count";
  const std::vector<uint8_t>::iterator field =
      std::search(image.begin(), image.end(), key.begin(), key.end());
  ASSERT_NE(field, image.end());
  *(field + key.size() - 1) = 'X';
  const std::vector<KernelDescriptorInfo> kernels = descriptors(image);
  ASSERT_EQ(kernels.size(), 1u);
  const AmdGpuCodeObject object(image.data(), image.size());
  CodeObjectPatcher patcher(object);
  amdhsa::kernel_descriptor_t descriptor = kernels.front().descriptor;
  descriptor.kernarg_size = 40;
  std::string error;
  EXPECT_FALSE(patcher.patch_kernel_resources(kernels.front(), descriptor, 40, 8, &error));
  EXPECT_NE(error.find("resource field"), std::string::npos);
  EXPECT_TRUE(std::ranges::equal(patcher.image_bytes(), image));
}

TEST(AmdhsaMetadata, MissingOrAmbiguousKernelMetadataRejectsWithoutMutation) {
  const std::vector<uint8_t> plain = plain_kernel();
  const std::vector<uint8_t> wrong =
      test::with_resource_metadata(plain, {{"other_kernel", 20, 16, 4}});
  const std::vector<uint8_t> duplicate =
      test::with_resource_metadata(plain, {{"test_kernel", 20, 16, 4}, {"test_kernel", 20, 16, 4}});
  for (const std::vector<uint8_t> &image : {plain, wrong, duplicate}) {
    const std::vector<KernelDescriptorInfo> kernels = descriptors(image);
    ASSERT_EQ(kernels.size(), 1u);
    const AmdGpuCodeObject object(image.data(), image.size());
    CodeObjectPatcher patcher(object);
    amdhsa::kernel_descriptor_t descriptor = kernels.front().descriptor;
    descriptor.kernarg_size = 40;
    std::string error;
    EXPECT_FALSE(patcher.patch_kernel_resources(kernels.front(), descriptor, 40, 8, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_TRUE(std::ranges::equal(patcher.image_bytes(), image));
  }
}

TEST(AmdhsaMetadata, DescriptorOverlapRejectsWithoutMutation) {
  const std::vector<uint8_t> image =
      test::with_resource_metadata(plain_kernel(), {{"test_kernel", 20, 16, 4}});
  const std::vector<KernelDescriptorInfo> kernels = descriptors(image);
  ASSERT_EQ(kernels.size(), 1u);
  const FailureOr<amdhsa::MetadataDocument> metadata = amdhsa::MetadataDocument::parse(image);
  ASSERT_TRUE(metadata.succeeded());
  KernelDescriptorInfo overlapping = kernels.front();
  overlapping.descriptor_file_offset = metadata.value().payload().data() - image.data();
  const AmdGpuCodeObject object(image.data(), image.size());
  CodeObjectPatcher patcher(object);
  std::string error;
  EXPECT_FALSE(
      patcher.patch_kernel_resources(overlapping, kernels.front().descriptor, 40, 8, &error));
  EXPECT_NE(error.find("overlaps"), std::string::npos);
  EXPECT_TRUE(std::ranges::equal(patcher.image_bytes(), image));
}

TEST(AmdhsaMetadata, DuplicateKeysAndTrailingPayloadAreRejected) {
  const std::vector<uint8_t> plain = plain_kernel();
  std::vector<uint8_t> duplicate = test::resource_metadata_payload({{"test_kernel", 20, 16, 4}});
  duplicate.front() = 0x84;
  test::append_metadata_text(duplicate, "amdhsa.version");
  duplicate.insert(duplicate.end(), {0x92, 1, 2});
  std::vector<uint8_t> trailing = test::resource_metadata_payload({{"test_kernel", 20, 16, 4}});
  trailing.push_back(0xc0);
  for (const std::vector<uint8_t> &payload : {duplicate, trailing}) {
    const std::vector<uint8_t> image = test::with_metadata_note(plain, payload);
    EXPECT_TRUE(amdhsa::MetadataDocument::parse(image).failed());
  }
}

TEST(AmdhsaMetadata, RecursiveValuesHaveABoundedDepth) {
  for (const unsigned depth : {31u, 32u}) {
    std::vector<uint8_t> payload{0x81};
    test::append_metadata_text(payload, "nested");
    payload.insert(payload.end(), depth, 0x91);
    payload.push_back(0xc0);
    const std::vector<uint8_t> image = test::with_metadata_note(plain_kernel(), payload);
    EXPECT_EQ(amdhsa::MetadataDocument::parse(image).succeeded(), depth == 31);
  }
}

} // namespace
} // namespace rocjitsu
