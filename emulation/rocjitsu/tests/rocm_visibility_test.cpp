// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocm_visibility.h"

#include <gtest/gtest.h>
#include <optional>
#include <string_view>
#include <vector>

namespace {

std::vector<rocjitsu::cli::VisibleGpu> test_gpus() {
  return {{0, 100, 90402, 0x1111111111111111ULL},
          {1, 101, 90402, 0x2222222222222222ULL},
          {2, 102, 120001, 0x3333333333333333ULL}};
}

} // namespace

TEST(RocmVisibilityTest, UnsetAndEmptyRocrSelectorsDiffer) {
  const auto gpus = test_gpus();
  EXPECT_EQ(3u, rocjitsu::cli::filter_rocr_visible_gpus(gpus, std::nullopt).size());
  EXPECT_TRUE(rocjitsu::cli::filter_rocr_visible_gpus(gpus, std::string_view{}).empty());
}

TEST(RocmVisibilityTest, KfdEnumerationSkipsZeroGpuIdsAndCompactsOrdinals) {
  const std::vector<rocjitsu::cli::VisibleGpu> candidates{{0, 0, 90402, 0},
                                                          {1, 101, 90402, 0x2222222222222222ULL}};
  const auto gpus = rocjitsu::cli::enumerate_kfd_gpus(candidates);

  ASSERT_EQ(1u, gpus.size());
  EXPECT_EQ(0u, gpus[0].ordinal);
  EXPECT_EQ(101u, gpus[0].gpu_id);
}

TEST(RocmVisibilityTest, NumericAndUuidSelectorsReorderDevices) {
  const auto gpus = test_gpus();
  const auto numeric = rocjitsu::cli::filter_rocr_visible_gpus(gpus, "1,0");
  const auto uuid = rocjitsu::cli::filter_rocr_visible_gpus(gpus, "gpu-2222");

  ASSERT_EQ(2u, numeric.size());
  EXPECT_EQ(101u, numeric[0].gpu_id);
  EXPECT_EQ(100u, numeric[1].gpu_id);
  ASSERT_EQ(1u, uuid.size());
  EXPECT_EQ(101u, uuid[0].gpu_id);
}

TEST(RocmVisibilityTest, AmbiguousRocrUuidTerminatesAfterValidPrefix) {
  const std::vector<rocjitsu::cli::VisibleGpu> gpus{{0, 100, 90402, 0x1111111111111111ULL},
                                                    {1, 101, 90402, 0x2222222211111111ULL},
                                                    {2, 102, 90402, 0x2222222244444444ULL}};
  const auto selected = rocjitsu::cli::filter_rocr_visible_gpus(gpus, "0,GPU-2222,1");

  ASSERT_EQ(1u, selected.size());
  EXPECT_EQ(100u, selected[0].gpu_id);
}

TEST(RocmVisibilityTest, ClientUuidUsesCaseSensitiveFirstMatch) {
  const std::vector<rocjitsu::cli::VisibleGpu> gpus{{0, 100, 90402, 0x1111111111111111ULL},
                                                    {1, 101, 90402, 0x1111111111111122ULL}};
  const auto first = rocjitsu::cli::filter_client_visible_gpus(gpus, "GPU-1111");
  const auto lowercase = rocjitsu::cli::filter_client_visible_gpus(gpus, "gpu-1111");

  ASSERT_EQ(1u, first.size());
  EXPECT_EQ(100u, first[0].gpu_id);
  EXPECT_TRUE(lowercase.empty());
}

TEST(RocmVisibilityTest, InvalidRocrTokenPreservesValidPrefix) {
  const auto selected = rocjitsu::cli::filter_rocr_visible_gpus(test_gpus(), "1,invalid,0");
  ASSERT_EQ(1u, selected.size());
  EXPECT_EQ(101u, selected[0].gpu_id);
}

TEST(RocmVisibilityTest, MalformedSelectorsTerminateSelection) {
  EXPECT_TRUE(rocjitsu::cli::filter_rocr_visible_gpus(test_gpus(), "GPU-").empty());
  EXPECT_TRUE(rocjitsu::cli::filter_client_visible_gpus(test_gpus(), "01").empty());
  EXPECT_TRUE(rocjitsu::cli::filter_client_visible_gpus(test_gpus(), "00").empty());

  // The client UUID guard rejects malformed spellings rather than substring-matching them onto the
  // first agent: too short to carry a body, the ROCR no-UUID sentinel, and one hex digit past the
  // 20-char maximum. Each falls through to the numeric parse, which terminates selection.
  EXPECT_TRUE(rocjitsu::cli::filter_client_visible_gpus(test_gpus(), "GPU-").empty());
  EXPECT_TRUE(rocjitsu::cli::filter_client_visible_gpus(test_gpus(), "GPU-XX").empty());
  EXPECT_TRUE(
      rocjitsu::cli::filter_client_visible_gpus(test_gpus(), "GPU-11111111111111111").empty());

  const auto rocr_prefix = rocjitsu::cli::filter_rocr_visible_gpus(test_gpus(), "0,GPU-,1");
  ASSERT_EQ(1u, rocr_prefix.size());
  EXPECT_EQ(100u, rocr_prefix[0].gpu_id);

  const auto client_prefix = rocjitsu::cli::filter_client_visible_gpus(test_gpus(), "1,00,2");
  ASSERT_EQ(1u, client_prefix.size());
  EXPECT_EQ(101u, client_prefix[0].gpu_id);

  const auto client_uuid_prefix =
      rocjitsu::cli::filter_client_visible_gpus(test_gpus(), "1,GPU-,2");
  ASSERT_EQ(1u, client_uuid_prefix.size());
  EXPECT_EQ(101u, client_uuid_prefix[0].gpu_id);
}

TEST(RocmVisibilityTest, DuplicateAndReorderedSelectorsMatchRuntimeBehavior) {
  const auto rocr_duplicate = rocjitsu::cli::filter_rocr_visible_gpus(test_gpus(), "0,0,1");
  ASSERT_EQ(1u, rocr_duplicate.size());
  EXPECT_EQ(100u, rocr_duplicate[0].gpu_id);

  const auto client_duplicate = rocjitsu::cli::filter_client_visible_gpus(test_gpus(), "0,0,1");
  ASSERT_EQ(2u, client_duplicate.size());
  EXPECT_EQ(100u, client_duplicate[0].gpu_id);
  EXPECT_EQ(101u, client_duplicate[1].gpu_id);

  const auto client_reordered = rocjitsu::cli::filter_client_visible_gpus(test_gpus(), "2,0");
  ASSERT_EQ(2u, client_reordered.size());
  EXPECT_EQ(102u, client_reordered[0].gpu_id);
  EXPECT_EQ(100u, client_reordered[1].gpu_id);
}

TEST(RocmVisibilityTest, NegativeRocrOrdinalPreservesValidPrefix) {
  const auto selected = rocjitsu::cli::filter_rocr_visible_gpus(test_gpus(), "0,-1,1");
  ASSERT_EQ(1u, selected.size());
  EXPECT_EQ(100u, selected[0].gpu_id);
}

TEST(RocmVisibilityTest, HipSelectorUsesPostRocrOrdinals) {
  const auto selected =
      rocjitsu::cli::effective_visible_gpus(test_gpus(), "1,0", "1", std::nullopt);
  ASSERT_EQ(1u, selected.size());
  EXPECT_EQ(100u, selected[0].gpu_id);
}

TEST(RocmVisibilityTest, HipSelectorTakesPrecedenceOverCudaFallback) {
  const auto hip = rocjitsu::cli::effective_visible_gpus(test_gpus(), std::nullopt, "1", "2");
  const auto cuda =
      rocjitsu::cli::effective_visible_gpus(test_gpus(), std::nullopt, std::string_view{}, "2");

  ASSERT_EQ(1u, hip.size());
  EXPECT_EQ(101u, hip[0].gpu_id);
  ASSERT_EQ(1u, cuda.size());
  EXPECT_EQ(102u, cuda[0].gpu_id);
}

TEST(RocmVisibilityTest, NormalizesClientUuidToPostRocrOrdinal) {
  const auto normalized = rocjitsu::cli::normalized_client_visible_devices(
      test_gpus(), "1,0", "GPU-1111111111111111", std::nullopt);
  ASSERT_TRUE(normalized);
  EXPECT_EQ("HIP_VISIBLE_DEVICES", normalized->name);
  EXPECT_EQ("1", normalized->value);
}

TEST(RocmVisibilityTest, NormalizesClientSelectorAgainstExpandedRocrOrder) {
  const auto normalized =
      rocjitsu::cli::normalized_client_visible_devices(test_gpus(), "1,0,3", "1", std::nullopt);
  ASSERT_TRUE(normalized);
  EXPECT_EQ("HIP_VISIBLE_DEVICES", normalized->name);
  EXPECT_EQ("1", normalized->value);
}

TEST(RocmVisibilityTest, NormalizesSelectedDbtHostToClientDeviceZero) {
  const std::vector<rocjitsu::cli::VisibleGpu> heterogeneous_gpus{
      {0, 100, 110000, 0x1111111111111111ULL}, {1, 101, 90402, 0x2222222222222222ULL}};
  const auto normalized = rocjitsu::cli::normalized_client_visible_devices(
      heterogeneous_gpus, std::nullopt, "0,1", std::nullopt, 101);

  ASSERT_TRUE(normalized);
  EXPECT_EQ("HIP_VISIBLE_DEVICES", normalized->name);
  EXPECT_EQ("1,0", normalized->value);
}

TEST(RocmVisibilityTest, SynthesizesClientOrderForSelectedDbtHost) {
  const std::vector<rocjitsu::cli::VisibleGpu> heterogeneous_gpus{
      {0, 100, 110000, 0x1111111111111111ULL}, {1, 101, 90402, 0x2222222222222222ULL}};
  const auto normalized = rocjitsu::cli::normalized_client_visible_devices(
      heterogeneous_gpus, std::nullopt, std::nullopt, std::nullopt, 101);

  ASSERT_TRUE(normalized);
  EXPECT_EQ("HIP_VISIBLE_DEVICES", normalized->name);
  EXPECT_EQ("1,0", normalized->value);
}

TEST(RocmVisibilityTest, DoesNotSynthesizeClientOrderWithoutSelectedHost) {
  EXPECT_FALSE(rocjitsu::cli::normalized_client_visible_devices(
      test_gpus(), std::nullopt, std::nullopt, std::nullopt, std::nullopt));
}

TEST(RocmVisibilityTest, SelectsDbtHostFromClientVisibleGpusBeforeNormalization) {
  const auto visible =
      rocjitsu::cli::effective_visible_gpus(test_gpus(), std::nullopt, "1", std::nullopt);
  const auto automatic = rocjitsu::cli::select_host_gpu(visible, 0, 90402);
  ASSERT_EQ(rocjitsu::cli::HostSelectionStatus::Selected, automatic.status);
  EXPECT_EQ(101u, automatic.gpu_id);

  const auto normalized = rocjitsu::cli::normalized_client_visible_devices(
      test_gpus(), std::nullopt, "1", std::nullopt, automatic.gpu_id);
  ASSERT_TRUE(normalized);
  EXPECT_EQ("1", normalized->value);

  const auto hidden_explicit = rocjitsu::cli::select_host_gpu(visible, 100, 90402);
  EXPECT_EQ(rocjitsu::cli::HostSelectionStatus::ExplicitGpuHidden, hidden_explicit.status);
}

TEST(RocmVisibilityTest, ExpandsRocrSelectionWithGuestOrdinal) {
  const auto expanded = rocjitsu::cli::expanded_rocr_visible_devices(test_gpus(), "GPU-2222");
  ASSERT_TRUE(expanded);
  EXPECT_EQ("1,3", *expanded);
}

TEST(RocmVisibilityTest, DoesNotExpandEmptyRocrSelection) {
  EXPECT_FALSE(rocjitsu::cli::expanded_rocr_visible_devices(test_gpus(), "5"));
}

TEST(RocmVisibilityTest, SelectsFirstVisibleIsaMatch) {
  const rocjitsu::cli::HostSelection selection =
      rocjitsu::cli::select_host_gpu(test_gpus(), 0, 90402);
  EXPECT_EQ(rocjitsu::cli::HostSelectionStatus::Selected, selection.status);
  EXPECT_EQ(100u, selection.gpu_id);
}

TEST(RocmVisibilityTest, SelectsExplicitGpuWithMatchingIsa) {
  const rocjitsu::cli::HostSelection selection =
      rocjitsu::cli::select_host_gpu(test_gpus(), 101, 90402);
  EXPECT_EQ(rocjitsu::cli::HostSelectionStatus::Selected, selection.status);
  EXPECT_EQ(101u, selection.gpu_id);
}

TEST(RocmVisibilityTest, RejectsHiddenExplicitGpu) {
  const rocjitsu::cli::HostSelection selection =
      rocjitsu::cli::select_host_gpu({test_gpus()[0]}, 101, 90402);
  EXPECT_EQ(rocjitsu::cli::HostSelectionStatus::ExplicitGpuHidden, selection.status);
  EXPECT_EQ(101u, selection.gpu_id);
}

TEST(RocmVisibilityTest, RejectsExplicitGpuWithDifferentIsa) {
  const rocjitsu::cli::HostSelection selection =
      rocjitsu::cli::select_host_gpu(test_gpus(), 102, 90402);
  EXPECT_EQ(rocjitsu::cli::HostSelectionStatus::ExplicitGpuIsaMismatch, selection.status);
  EXPECT_EQ(102u, selection.gpu_id);
}

TEST(RocmVisibilityTest, ReportsMissingIsaMatch) {
  const rocjitsu::cli::HostSelection selection =
      rocjitsu::cli::select_host_gpu(test_gpus(), 0, 999999);
  EXPECT_EQ(rocjitsu::cli::HostSelectionStatus::NoIsaMatch, selection.status);
  EXPECT_EQ(0u, selection.gpu_id);
}
