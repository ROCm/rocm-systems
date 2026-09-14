// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amdgpu_elf_test_support.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/major_image_ownership.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <utility>

namespace rocjitsu {
namespace {
TEST(AmdGpuCodeObjectAccounting, MovedPatcherMeasuresDestinationImage) {
  const auto image = test_support::make_minimal_amdgpu_elf_with_descriptor_after_text();
  AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());

  major_image_ownership::ScopedMeasurement measurement;
  CodeObjectPatcher patcher(object);
  CodeObjectPatcher moved(std::move(patcher));
  const major_image_ownership::ScopedPhase phase(major_image_ownership::Phase::IncrementalPatch);
  const major_image_ownership::Measurement observed = measurement.snapshot();
  const auto &incremental = observed.phase(major_image_ownership::Phase::IncrementalPatch);
  EXPECT_FALSE(observed.bookkeeping_error);
  EXPECT_GE(incremental
                .bytes_at_peak[static_cast<size_t>(major_image_ownership::OwnerKind::PatcherImage)],
            moved.image_bytes().size());
}

TEST(AmdGpuCodeObjectPatcher, MovedPatcherRetainsSelectedTextSectionForReplacement) {
  const auto image = test_support::make_minimal_amdgpu_elf_with_descriptor_after_text();
  AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());

  CodeObjectPatcher patcher(object);
  std::vector<uint8_t> replacement(patcher.text_bytes().begin(), patcher.text_bytes().end());
  ASSERT_FALSE(replacement.empty());
  replacement.front() ^= 1u;

  CodeObjectPatcher moved(std::move(patcher));
  ASSERT_TRUE(moved.replace_text(replacement));
  EXPECT_TRUE(std::ranges::equal(moved.text_bytes(), replacement));

  AmdGpuCodeObject replaced(moved.image_bytes().data(), moved.image_bytes().size());
  ASSERT_TRUE(replaced.is_valid());
  ASSERT_EQ(replaced.text_sections().size(), 1u);
  const Section *text = replaced.text_sections().front();
  const std::span<const uint8_t> replaced_text(reinterpret_cast<const uint8_t *>(text->data()),
                                               text->size());
  EXPECT_TRUE(std::ranges::equal(replaced_text, replacement));
}

} // namespace
} // namespace rocjitsu
