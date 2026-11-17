// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/code/consan/image/facts.h"
#include "rocjitsu/code/consan/target.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>

namespace rocjitsu::consan {
namespace {

// Deliberately synthetic codec facts; these do not claim valid ELF/GPU evidence.
ImageFacts example_image() {
  ImageFacts image;
  image.original = {.sha256 = digest(std::array<uint8_t, 4096>{}), .bytes = 4096};
  image.prepared = {.sha256 = digest(std::array<uint8_t, 8192>{}), .bytes = 8192};
  image.target = 1250;
  image.wave_size = 32;
  image.code_object_abi = 5;
  image.original_text_bytes = 1024;
  image.prepared_text_bytes = 4096;
  for (uint32_t index = 0; index < 3; ++index) {
    ImageEntryFacts entry;
    entry.key = 2048 + 64 * index;
    entry.disposition = index < 2 ? EntryDisposition::Selected : EntryDisposition::Native;
    entry.original_sgprs = 16 + 8 * index;
    entry.original_vgprs = 8 + 8 * index;
    entry.binding = {.name = std::string("entry_") + std::to_string(index),
                     .original_descriptor_offset = entry.key,
                     .prepared_descriptor_offset = 5120 + 64 * index,
                     .original_entry_offset = 256 * index,
                     .prepared_entry_offset = 256 * index,
                     .original_kernarg_bytes = 16 + 8 * index,
                     .kernarg_alignment = 8,
                     .original_group_bytes = 256,
                     .prepared_group_bytes = 256,
                     .prepared_sgprs = 16 + 8 * index,
                     .prepared_vgprs = 8 + 8 * index};
    if (entry.disposition == EntryDisposition::Selected) {
      Manifest manifest;
      manifest.original = image.original;
      manifest.prepared = image.prepared;
      manifest.options = mvp_options_digest();
      manifest.target = image.target;
      manifest.wave_size = image.wave_size;
      manifest.code_object_abi = image.code_object_abi;
      manifest.original_text_bytes = image.original_text_bytes;
      manifest.prepared_text_bytes = image.prepared_text_bytes;
      manifest.entry = entry.binding;
      manifest.sites = {{.analysis = AnalysisKind::LdsRaceDetection,
                         .access = index == 0 ? AccessKind::Write : AccessKind::Read,
                         .width = 4,
                         .original_offset = 256 * index + 32,
                         .original_size = 8,
                         .prepared_guest_offset = 2048 + 256 * index,
                         .prepared_guest_size = 8,
                         .probe_offset = 2080 + 256 * index,
                         .probe_size = 16}};
      entry.selected = std::move(manifest);
    }
    image.entries.push_back(std::move(entry));
  }
  const std::array<EntryKey, 2> keys = {2048, 2112};
  image.selection_profile = selection_profile_digest(keys);
  return image;
}

ImageFacts qualified_image(const TargetProfile &target, uint32_t wave_size, ScopeKind scope,
                           uint32_t width = 4) {
  ImageFacts image = example_image();
  image.target = target.id;
  image.wave_size = wave_size;
  for (ImageEntryFacts &entry : image.entries) {
    entry.binding.workgroup_id_scalar =
        target.id == 1201 || target.id == 1250 ? kWorkgroupIdTtmp9Selector : 15;
    if (!entry.selected)
      continue;
    Manifest &manifest = *entry.selected;
    manifest.target = target.id;
    manifest.wave_size = wave_size;
    manifest.scope = scope;
    manifest.entry = entry.binding;
    manifest.sites.front().width = width;
    if (scope != ScopeKind::NativeLdsNoSynchronization) {
      const bool split = uses_split_workgroup_barrier(target.arch);
      const uint64_t original = entry.binding.original_entry_offset + 8;
      const uint64_t prepared = 1024 + (entry.key - 2048) * 2;
      manifest.barriers = {{.original_drain = original,
                            .original_signal = original + 4,
                            .original_wait = original + (split ? 8u : 4u),
                            .prepared_signal = prepared,
                            .prepared_wait = prepared + (split ? 4u : 0u),
                            .completion_probe = prepared + (split ? 8u : 4u),
                            .probe_size = 32}};
    }
  }
  const std::array<EntryKey, 2> keys = {2048, 2112};
  image.selection_profile = selection_profile_digest(keys, scope);
  return image;
}

void rehash(std::vector<uint8_t> &bytes) {
  const Digest checksum = digest(std::span(bytes).subspan(64));
  std::copy(checksum.begin(), checksum.end(), bytes.begin() + 32);
}

TEST(ConSanImageFacts, OwnsDistinctSelectedManifestsAndNativeDisposition) {
  const ImageFacts image = example_image();
  ContractError error;
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(image, &error);
  ASSERT_TRUE(encoded.succeeded()) << error.message;
  FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value(), &error);
  ASSERT_TRUE(decoded.succeeded()) << error.message;
  EXPECT_EQ(decoded.value(), image);
  ASSERT_NE(selected_manifest(decoded.value(), 2048), nullptr);
  ASSERT_NE(selected_manifest(decoded.value(), 2112), nullptr);
  EXPECT_EQ(selected_manifest(decoded.value(), 2176), nullptr);
  EXPECT_EQ(selected_manifest(decoded.value(), 1), nullptr);
  EXPECT_EQ(find_image_entry(decoded.value(), 2176)->disposition, EntryDisposition::Native);
  EXPECT_NE(selected_manifest(decoded.value(), 2048)->entry.original_kernarg_bytes,
            selected_manifest(decoded.value(), 2112)->entry.original_kernarg_bytes);
}

TEST(ConSanImageFacts, RejectsDuplicateKeysNamesDescriptorsAndSharedSites) {
  const ImageFacts baseline = example_image();
  for (uint32_t kind = 0; kind < 4; ++kind) {
    ImageFacts image = baseline;
    if (kind == 0)
      image.entries[1].key = image.entries[0].key;
    else if (kind == 1)
      image.entries[1].binding.name = image.entries[0].binding.name;
    else if (kind == 2)
      image.entries[1].binding.prepared_descriptor_offset =
          image.entries[0].binding.prepared_descriptor_offset;
    else
      image.entries[1].selected->sites[0].original_offset =
          image.entries[0].selected->sites[0].original_offset;
    EXPECT_TRUE(validate_image_facts(image).failed()) << kind;
  }
}

TEST(ConSanImageFacts, RejectsReboundEntryImageAndSelectionFacts) {
  const ImageFacts baseline = example_image();
  for (uint32_t kind = 0; kind < 4; ++kind) {
    ImageFacts image = baseline;
    if (kind == 0)
      image.entries[0].selected->entry.name = "rebound";
    else if (kind == 1)
      image.entries[1].selected->prepared.sha256[0] ^= 1;
    else if (kind == 2)
      image.selection_profile[0] ^= 1;
    else
      image.entries[1].selected->target = 1100;
    EXPECT_TRUE(validate_image_facts(image).failed()) << kind;
  }
}

TEST(ConSanImageFacts, NativeCannotClaimDetectorFactsOrGrownResources) {
  const ImageFacts baseline = example_image();
  for (uint32_t kind = 0; kind < 4; ++kind) {
    ImageFacts image = baseline;
    ImageEntryFacts &native = image.entries.back();
    if (kind == 0)
      native.selected = image.entries[0].selected;
    else if (kind == 1)
      ++native.binding.prepared_group_bytes;
    else if (kind == 2)
      ++native.binding.prepared_sgprs;
    else
      native.disposition = static_cast<EntryDisposition>(99);
    EXPECT_TRUE(validate_image_facts(image).failed()) << kind;
  }
}

TEST(ConSanImageFacts, RejectsUnselectedImageAndMissingSelectedManifest) {
  ImageFacts image = example_image();
  image.entries[0].selected.reset();
  EXPECT_TRUE(validate_image_facts(image).failed());
  image = example_image();
  for (ImageEntryFacts &entry : image.entries) {
    entry.disposition = EntryDisposition::Native;
    entry.selected.reset();
  }
  EXPECT_TRUE(validate_image_facts(image).failed());
}

TEST(ConSanImageFacts, BoundedAggregateMetadataAndSiteCounts) {
  const ImageFacts image = example_image();
  ContractLimits limits;
  limits.sites = 1;
  ContractError error;
  EXPECT_TRUE(encode_image_facts(image, &error, limits).failed());
  EXPECT_EQ(error.code, ContractErrorCode::ResourceLimit);
  limits = {};
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(image);
  ASSERT_TRUE(encoded.succeeded());
  limits.metadata_bytes = encoded.value().size() - 1;
  EXPECT_TRUE(encode_image_facts(image, &error, limits).failed());
  EXPECT_EQ(error.code, ContractErrorCode::ResourceLimit);
  EXPECT_TRUE(decode_image_facts(encoded.value(), &error, limits).failed());
}

TEST(ConSanImageFacts, DecoderRejectsTruncationCorruptionAndOversizedEntryClaims) {
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(example_image());
  ASSERT_TRUE(encoded.succeeded());
  for (size_t extent = 0; extent < encoded.value().size(); ++extent)
    EXPECT_TRUE(decode_image_facts(std::span(encoded.value()).first(extent)).failed()) << extent;
  std::vector<uint8_t> bytes = encoded.value();
  bytes.back() ^= 1;
  EXPECT_TRUE(decode_image_facts(bytes).failed());
  bytes = encoded.value();
  wire::write_u32(bytes, 64 + 140, std::numeric_limits<uint32_t>::max());
  rehash(bytes);
  ContractError error;
  EXPECT_TRUE(decode_image_facts(bytes, &error).failed());
  EXPECT_EQ(error.code, ContractErrorCode::ResourceLimit);
  bytes = encoded.value();
  bytes[8] ^= 1;
  EXPECT_TRUE(decode_image_facts(bytes, &error).failed());
  EXPECT_EQ(error.code, ContractErrorCode::IncompatibleContract);
}

TEST(ConSanImageFacts, ExpectedProfileIsIndependentOfInternallyValidFacts) {
  const ImageFacts image = example_image();
  const std::array<uint8_t, 4096> original{};
  const std::array<uint8_t, 8192> prepared{};
  EXPECT_TRUE(match_image_facts(image, original, prepared, image.selection_profile).succeeded());
  Digest wrong_profile = image.selection_profile;
  wrong_profile[0] ^= 1;
  EXPECT_TRUE(match_image_facts(image, original, prepared, wrong_profile).failed());
  std::array<uint8_t, 8192> changed = prepared;
  changed.back() = 1;
  EXPECT_TRUE(match_image_facts(image, original, changed, image.selection_profile).failed());
}

TEST(ConSanImageFacts, EveryQScopeWidthAndTargetWaveRoundTripsWorkgroupAndBarrierFacts) {
  for (const TargetProfile &target : kNativeLdsTargets)
    for (const uint32_t wave_size : {32u, 64u}) {
      if (!supports_wave(target, wave_size))
        continue;
      for (const ScopeKind scope :
           {ScopeKind::NativeLdsNoSynchronization, ScopeKind::NativeLdsWorkgroupBarrier,
            ScopeKind::NativeLdsWorkgroupBarrierLoop})
        for (const uint32_t width : {2u, 4u, 8u, 16u}) {
          SCOPED_TRACE(target.id);
          SCOPED_TRACE(wave_size);
          SCOPED_TRACE(static_cast<uint32_t>(scope));
          SCOPED_TRACE(width);
          const ImageFacts image = qualified_image(target, wave_size, scope, width);
          ContractError error;
          const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(image, &error);
          ASSERT_TRUE(encoded.succeeded()) << error.message;
          const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value(), &error);
          ASSERT_TRUE(decoded.succeeded()) << error.message;
          EXPECT_EQ(decoded.value(), image);
          EXPECT_EQ(selected_scope(decoded.value()), scope);
          for (const ImageEntryFacts &entry : decoded.value().entries) {
            if (!entry.selected)
              continue;
            const Manifest &manifest = *entry.selected;
            const uint32_t threads = 2 * wave_size;
            const FailureOr<LaunchLayout> layout =
                plan_launch(manifest, {{4 * threads, 1, 1}, {threads, 1, 1}}, 256, 0, &error);
            ASSERT_TRUE(layout.succeeded()) << error.message;
            EXPECT_EQ(layout.value().workgroup_count, 4u);
            EXPECT_EQ(layout.value().wave_count, 8u);
            EXPECT_EQ(layout.value().row_stride,
                      report_row_bytes(1, scope != ScopeKind::NativeLdsNoSynchronization));
          }
        }
    }
}

TEST(ConSanImageFacts, ScopeIdentityIsExplicitAndMixedSelectedScopesAreRejected) {
  const std::array<EntryKey, 2> keys = {2048, 2112};
  EXPECT_NE(selection_profile_digest(keys, ScopeKind::NativeLdsNoSynchronization),
            selection_profile_digest(keys, ScopeKind::NativeLdsWorkgroupBarrier));
  EXPECT_NE(selection_profile_digest(keys, ScopeKind::NativeLdsWorkgroupBarrier),
            selection_profile_digest(keys, ScopeKind::NativeLdsWorkgroupBarrierLoop));
  ImageFacts image =
      qualified_image(*target_profile(1250), 32, ScopeKind::NativeLdsWorkgroupBarrier);
  ASSERT_TRUE(validate_image_facts(image).succeeded());
  image.entries[1].selected->scope = ScopeKind::NativeLdsWorkgroupBarrierLoop;
  ASSERT_TRUE(validate_manifest(*image.entries[1].selected).succeeded());
  EXPECT_FALSE(selected_scope(image).has_value());
  ContractError error;
  EXPECT_TRUE(encode_image_facts(image, &error).failed());
  EXPECT_EQ(error.code, ContractErrorCode::InvalidFacts);
  EXPECT_NE(error.message.find("share one qualified scope"), std::string::npos);
}

TEST(ConSanImageFacts, BarrierBytesParticipateInTheInclusiveAggregateMetadataBudget) {
  const ImageFacts image =
      qualified_image(*target_profile(1250), 32, ScopeKind::NativeLdsWorkgroupBarrierLoop);
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(image);
  ASSERT_TRUE(encoded.succeeded());
  ContractLimits limits;
  limits.metadata_bytes = encoded.value().size();
  EXPECT_TRUE(encode_image_facts(image, nullptr, limits).succeeded());
  EXPECT_TRUE(decode_image_facts(encoded.value(), nullptr, limits).succeeded());
  --limits.metadata_bytes;
  ContractError error;
  EXPECT_TRUE(encode_image_facts(image, &error, limits).failed());
  EXPECT_EQ(error.code, ContractErrorCode::ResourceLimit);
}

TEST(ConSanImageFacts, SelectedEntriesCannotShareOriginalOrPreparedBarrierObservations) {
  const ImageFacts baseline =
      qualified_image(*target_profile(1250), 32, ScopeKind::NativeLdsWorkgroupBarrier);
  ASSERT_TRUE(validate_image_facts(baseline).succeeded());
  for (const bool original : {false, true}) {
    ImageFacts image = baseline;
    const BarrierFacts &first = image.entries[0].selected->barriers.front();
    BarrierFacts &second = image.entries[1].selected->barriers.front();
    if (original) {
      second.original_drain = first.original_drain;
      second.original_signal = first.original_signal;
      second.original_wait = first.original_wait;
    } else {
      second.prepared_signal = first.prepared_signal;
      second.prepared_wait = first.prepared_wait;
      second.completion_probe = first.completion_probe;
    }
    ASSERT_TRUE(validate_manifest(*image.entries[1].selected).succeeded());
    EXPECT_TRUE(validate_image_facts(image).failed());
  }
}

} // namespace
} // namespace rocjitsu::consan
