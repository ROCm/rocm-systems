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
  if (!object.is_valid() || object.text_sections().size() != 1)
    return {};
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
  const std::vector<uint8_t> image =
      test::with_resource_metadata(test::make_gfx950_kernel_elf({0xbf800000u, 0xbf810000u}, 127),
                                   {{"test_kernel", 0, 16, 4, 127}}, true);
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

TEST(AmdhsaMetadata, WiderIntegerRejectsBeforeNoteOrDescriptorChanges) {
  const std::vector<uint8_t> image =
      test::with_resource_metadata(plain_kernel(), {{"test_kernel", 20, 16, 4}}, true);
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
