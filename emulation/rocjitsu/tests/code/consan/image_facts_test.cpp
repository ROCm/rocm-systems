// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/code/consan/image/facts.h"
#include "rocjitsu/code/consan/target.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>

namespace rocjitsu::consan {
namespace {

constexpr Digest kPriorOptions = {
    0x8e, 0x4e, 0x17, 0xc7, 0x9d, 0x00, 0x15, 0x6a, 0x93, 0x8a, 0xeb, 0xa2, 0x32, 0x06, 0x3d, 0xb0,
    0xa7, 0x24, 0x2c, 0x81, 0x8f, 0xe1, 0x9f, 0xa8, 0x3e, 0x6c, 0x37, 0x78, 0xd4, 0x3c, 0xff, 0x17,
};

// Deliberately synthetic codec facts; these do not claim valid ELF/GPU evidence.
ImageFacts example_image(const Digest &options = mvp_options_digest()) {
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
      manifest.options = options;
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
    } else {
      entry.native_code = {{.original_offset = entry.binding.original_entry_offset,
                            .prepared_offset = entry.binding.prepared_entry_offset,
                            .bytes = 4}};
    }
    image.entries.push_back(std::move(entry));
  }
  const std::array<EntryKey, 2> keys = {2048, 2112};
  image.selection_profile =
      selection_profile_digest(keys, ScopeKind::NativeLdsNoSynchronization, options);
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
    if (!entry.selected) {
      continue;
    }
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

TEST(ConSanImageFacts, EncodingPreservesKnownWireIdentity) {
  // Pin the image envelope, selected manifests and native entry for the
  // current producer contract and a historical options profile, including
  // its token, absent proof flags and empty boundary count in each nested LDS
  // manifest. Derived independently with struct/hashlib.
  const Digest expected = {0x86, 0x55, 0xe8, 0xf2, 0xbd, 0x68, 0x37, 0xf2, 0x07, 0xe9, 0x6f,
                           0x0a, 0xf8, 0x57, 0x5f, 0x62, 0xe8, 0x47, 0xda, 0x26, 0x87, 0xee,
                           0x6c, 0x82, 0x95, 0x92, 0x54, 0x03, 0xd1, 0x10, 0x60, 0x46};
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(example_image(kPriorOptions));
  ASSERT_TRUE(encoded.succeeded());
  EXPECT_EQ(encoded.value().size(), 1411u);
  EXPECT_EQ(digest(encoded.value()), expected);
}

TEST(ConSanImageFacts, ImageAndSingleEntryFormatsHaveDistinctMagic) {
  const ImageFacts image = example_image();
  const FailureOr<std::vector<uint8_t>> image_bytes = encode_image_facts(image);
  const FailureOr<std::vector<uint8_t>> manifest_bytes =
      encode_manifest(*image.entries[0].selected);
  ASSERT_TRUE(image_bytes.succeeded());
  ASSERT_TRUE(manifest_bytes.succeeded());
  const std::array<uint8_t, 8> image_magic = {'R', 'J', 'C', 'S', 'I', 'M', 'G', 'F'};
  const std::array<uint8_t, 8> manifest_magic = {'R', 'J', 'C', 'S', 'F', 'A', 'C', 'T'};
  EXPECT_TRUE(std::ranges::equal(std::span(image_bytes.value()).first(8), image_magic));
  EXPECT_TRUE(std::ranges::equal(std::span(manifest_bytes.value()).first(8), manifest_magic));
  util::StringDiagnostic error;
  EXPECT_TRUE(decode_manifest(image_bytes.value(), error.emitter()).failed());
  EXPECT_EQ(error.message(), "invalid static-facts framing");
  EXPECT_TRUE(decode_image_facts(manifest_bytes.value(), error.emitter()).failed());
  EXPECT_EQ(error.message(), "invalid image-facts framing");
}

TEST(ConSanImageFacts, OwnsDistinctSelectedManifestsAndNativeDisposition) {
  const ImageFacts image = example_image();
  util::StringDiagnostic error;
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(image, error.emitter());
  ASSERT_TRUE(encoded.succeeded()) << error.message();
  FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value(), error.emitter());
  ASSERT_TRUE(decoded.succeeded()) << error.message();
  EXPECT_EQ(decoded.value(), image);
  ASSERT_NE(selected_manifest(decoded.value(), 2048), nullptr);
  ASSERT_NE(selected_manifest(decoded.value(), 2112), nullptr);
  EXPECT_EQ(selected_manifest(decoded.value(), 2176), nullptr);
  EXPECT_EQ(selected_manifest(decoded.value(), 1), nullptr);
  EXPECT_EQ(find_image_entry(decoded.value(), 2176)->disposition, EntryDisposition::Native);
  EXPECT_NE(selected_manifest(decoded.value(), 2048)->entry.original_kernarg_bytes,
            selected_manifest(decoded.value(), 2112)->entry.original_kernarg_bytes);
}

TEST(ConSanImageFacts, NativeOwnershipRoundTripsAcrossAnotherEntry) {
  ImageFacts image = example_image();
  const ImageEntryFacts &selected = image.entries[1];
  ASSERT_EQ(selected.disposition, EntryDisposition::Selected);
  ASSERT_EQ(selected.binding.original_entry_offset, 256u);
  ASSERT_EQ(selected.binding.prepared_entry_offset, 256u);

  ImageEntryFacts &native = image.entries.back();
  native.binding.original_entry_offset = 128;
  native.binding.prepared_entry_offset = 128;
  native.native_code = {{.original_offset = 128, .prepared_offset = 128, .bytes = 16},
                        {.original_offset = 384, .prepared_offset = 384, .bytes = 16}};

  util::StringDiagnostic error;
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(image, error.emitter());
  ASSERT_TRUE(encoded.succeeded()) << error.message();
  const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value(), error.emitter());
  ASSERT_TRUE(decoded.succeeded()) << error.message();
  EXPECT_EQ(decoded.value(), image);
}

TEST(ConSanImageFacts, NativeNoProbesCertificateRoundTripsWithoutClaimingSelectedExecution) {
  ImageFacts image = example_image();
  ImageEntryFacts &native = image.entries.back();
  NoProbesNeededProof proof = {.instruction_count = 1, .code_bytes = 4};
  proof.code_sha256[0] = 1;
  native.no_probes_needed = proof;
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(image);
  ASSERT_TRUE(encoded.succeeded());
  const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value());
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value(), image);
  EXPECT_EQ(decoded.value().entries.back().disposition, EntryDisposition::Native);
  EXPECT_FALSE(decoded.value().entries.back().selected.has_value());
  ContractLimits limits;
  limits.metadata_bytes = encoded.value().size();
  EXPECT_TRUE(encode_image_facts(image, {}, limits).succeeded());
  --limits.metadata_bytes;
  EXPECT_TRUE(encode_image_facts(image, {}, limits).failed());

  // Recompute the framing checksum after lying about the certified decoded
  // instruction count. A valid envelope cannot make impossible proof facts valid.
  std::vector<uint8_t> forged = encoded.value();
  const size_t native_start = forged.size() - (128 + native.binding.name.size() + 24 + 56);
  const size_t proof_start = native_start + 16;
  ASSERT_EQ(wire::read_u32(forged, native_start + 12), 1u);
  ASSERT_EQ(wire::read_u64(forged, proof_start + 8), 1u);
  ASSERT_TRUE(wire::write_u64(forged, proof_start + 8, 2));
  rehash(forged);
  util::StringDiagnostic error;
  EXPECT_TRUE(decode_image_facts(forged, error.emitter()).failed());
  EXPECT_EQ(error.message(), "invalid no-probes-needed certificate");
}

TEST(ConSanImageFacts, NoProbesCertificateRequiresCompleteBoundedLdsNativeEvidence) {
  ImageFacts baseline = example_image();
  NoProbesNeededProof proof = {.instruction_count = 1, .code_bytes = 4};
  proof.code_sha256[0] = 1;
  baseline.entries.back().no_probes_needed = proof;
  ASSERT_TRUE(validate_image_facts(baseline).succeeded());
  for (uint32_t kind = 0; kind < 7; ++kind) {
    ImageFacts image = baseline;
    NoProbesNeededProof &changed = *image.entries.back().no_probes_needed;
    if (kind == 0)
      changed.version = 2;
    else if (kind == 1)
      changed.scope = ScopeKind::ExecutionTrace;
    else if (kind == 2)
      changed.instruction_count = 0;
    else if (kind == 3)
      changed.instruction_count = 2;
    else if (kind == 4)
      changed.code_bytes = 8;
    else if (kind == 5)
      changed.code_bytes = 0;
    else
      changed.code_sha256 = {};
    EXPECT_TRUE(validate_image_facts(image).failed()) << kind;
  }
  ImageFacts selected = example_image();
  selected.entries.front().no_probes_needed = proof;
  EXPECT_TRUE(validate_image_facts(selected).failed());
}

TEST(ConSanImageFacts, DecoderRejectsUnknownProofPresenceInsteadOfGuessingNativeCoverage) {
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(example_image());
  ASSERT_TRUE(encoded.succeeded());
  std::vector<uint8_t> bytes = encoded.value();
  wire::write_u32(bytes, 64 + 144 + 8 + 4, 2);
  rehash(bytes);
  util::StringDiagnostic error;
  EXPECT_TRUE(decode_image_facts(bytes, error.emitter()).failed());
  EXPECT_EQ(error.message(), "invalid or truncated image entry certificate flag");
}

TEST(ConSanImageFacts, NativeKernargAlignmentPreservesSmallPowersOfTwo) {
  for (const uint32_t alignment : {1u, 2u, 4u}) {
    SCOPED_TRACE(alignment);
    ImageFacts image = example_image();
    image.entries.back().binding.kernarg_alignment = alignment;
    util::StringDiagnostic error;
    EXPECT_TRUE(validate_image_facts(image, error.emitter()).succeeded()) << error.message();
    const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(image, error.emitter());
    EXPECT_TRUE(encoded.succeeded()) << error.message();
    if (encoded.failed())
      continue;
    const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value(), error.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error.message();
    EXPECT_EQ(decoded.value(), image);
    const FailureOr<std::vector<uint8_t>> reencoded = encode_image_facts(decoded.value());
    ASSERT_TRUE(reencoded.succeeded());
    EXPECT_EQ(reencoded.value(), encoded.value());
  }
}

TEST(ConSanImageFacts, NativeVgprsPreserveTheGfx1250AddressableRange) {
  for (const uint32_t vgprs : {384u, 1024u}) {
    SCOPED_TRACE(vgprs);
    ImageFacts image = example_image();
    ImageEntryFacts &native = image.entries.back();
    native.original_vgprs = vgprs;
    native.binding.prepared_vgprs = vgprs;
    util::StringDiagnostic diagnostic;
    ASSERT_TRUE(validate_image_facts(image, diagnostic.emitter()).succeeded())
        << diagnostic.message();
    const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(image, diagnostic.emitter());
    ASSERT_TRUE(encoded.succeeded()) << diagnostic.message();
    const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value(), diagnostic.emitter());
    ASSERT_TRUE(decoded.succeeded()) << diagnostic.message();
    EXPECT_EQ(decoded.value(), image);
    const FailureOr<std::vector<uint8_t>> reencoded = encode_image_facts(decoded.value());
    ASSERT_TRUE(reencoded.succeeded());
    EXPECT_EQ(reencoded.value(), encoded.value());
  }
}

TEST(ConSanImageFacts, NativeVgprsRejectCountsAboveTheTargetRange) {
  for (const TargetProfile &target : kNativeLdsTargets) {
    SCOPED_TRACE(target.name);
    ImageFacts image =
        qualified_image(target, target.default_wave_size, ScopeKind::NativeLdsNoSynchronization);
    ImageEntryFacts &native = image.entries.back();
    const uint32_t vgprs = target.id == 1250                      ? 1025
                           : target.id == 942 || target.id == 950 ? 513
                                                                  : 257;
    native.original_vgprs = vgprs;
    native.binding.prepared_vgprs = vgprs;
    EXPECT_TRUE(validate_image_facts(image).failed());
    EXPECT_TRUE(encode_image_facts(image).failed());
  }
}

TEST(ConSanImageFacts, SelectedVgprsRetainTheInstrumentationLimit) {
  ImageFacts image = example_image();
  ImageEntryFacts &selected = image.entries.front();
  selected.original_vgprs = 256;
  selected.binding.prepared_vgprs = 256;
  selected.selected->entry = selected.binding;
  ASSERT_TRUE(validate_manifest(*selected.selected).succeeded());
  ASSERT_TRUE(validate_image_facts(image).succeeded());
  selected.original_vgprs = 257;
  selected.binding.prepared_vgprs = 257;
  selected.selected->entry = selected.binding;
  EXPECT_TRUE(validate_manifest(*selected.selected).failed());
  EXPECT_TRUE(validate_image_facts(image).failed());
  EXPECT_TRUE(encode_image_facts(image).failed());
}

TEST(ConSanImageFacts, NativeCdnaVgprsPreserveTheUnifiedAllocationRange) {
  for (const char *name : {"gfx942", "gfx950"}) {
    SCOPED_TRACE(name);
    const TargetProfile &target = *target_profile(name);
    for (const uint32_t vgprs : {384u, 512u}) {
      SCOPED_TRACE(vgprs);
      ImageFacts image = qualified_image(target, 64, ScopeKind::NativeLdsNoSynchronization);
      ImageEntryFacts &native = image.entries.back();
      native.original_vgprs = vgprs;
      native.binding.prepared_vgprs = vgprs;
      util::StringDiagnostic diagnostic;
      ASSERT_TRUE(validate_image_facts(image, diagnostic.emitter()).succeeded())
          << diagnostic.message();
      const FailureOr<std::vector<uint8_t>> encoded =
          encode_image_facts(image, diagnostic.emitter());
      ASSERT_TRUE(encoded.succeeded()) << diagnostic.message();
      const FailureOr<ImageFacts> decoded =
          decode_image_facts(encoded.value(), diagnostic.emitter());
      ASSERT_TRUE(decoded.succeeded()) << diagnostic.message();
      EXPECT_EQ(decoded.value(), image);
    }
  }
}

TEST(ConSanImageFacts, NativeVgprsCannotGrowWithinTheTargetRange) {
  ImageFacts image = example_image();
  ImageEntryFacts &native = image.entries.back();
  native.original_vgprs = 384;
  native.binding.prepared_vgprs = 400;
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(validate_image_facts(image, diagnostic.emitter()).failed());
  EXPECT_EQ(diagnostic.message(), "native entry changes ABI or resources");
  EXPECT_TRUE(encode_image_facts(image).failed());
}

TEST(ConSanImageFacts, RejectsZeroAndNonPowerOfTwoKernargAlignment) {
  for (const uint32_t alignment : {0u, 3u}) {
    SCOPED_TRACE(alignment);
    for (const size_t index : {size_t{0}, size_t{2}}) {
      SCOPED_TRACE(index);
      ImageFacts image = example_image();
      ImageEntryFacts &entry = image.entries[index];
      entry.binding.kernarg_alignment = alignment;
      if (entry.selected)
        entry.selected->entry = entry.binding;
      EXPECT_TRUE(validate_image_facts(image).failed());
      EXPECT_TRUE(encode_image_facts(image).failed());
    }
  }
}

TEST(ConSanImageFacts, SelectedKernargAlignmentRetainsMinimumEight) {
  for (const uint32_t alignment : {1u, 2u, 4u}) {
    SCOPED_TRACE(alignment);
    ImageFacts image = example_image();
    ImageEntryFacts &selected = image.entries.front();
    selected.binding.kernarg_alignment = alignment;
    selected.selected->entry = selected.binding;
    EXPECT_TRUE(validate_manifest(*selected.selected).failed());
    EXPECT_TRUE(validate_image_facts(image).failed());
    EXPECT_TRUE(encode_image_facts(image).failed());
  }
}

TEST(ConSanImageFacts, SuccessfulValidationEmitsNoDiagnostic) {
  std::vector<std::string> messages;
  auto collect = [&messages](std::string_view message) { messages.emplace_back(message); };
  const util::DiagnosticEmitter diagnostic(collect);
  EXPECT_TRUE(validate_image_facts(example_image(), diagnostic).succeeded());
  EXPECT_TRUE(messages.empty());
}

TEST(ConSanImageFacts, NestedValidationEmitsOneDiagnosticAndFailsWithoutASink) {
  ImageFacts image = example_image();
  image.entries.front().selected->sites.front().width = 3;
  std::vector<std::string> messages;
  auto collect = [&messages](std::string_view message) { messages.emplace_back(message); };
  const util::DiagnosticEmitter diagnostic(collect);
  EXPECT_TRUE(validate_image_facts(image, diagnostic).failed());
  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages.front(), "invalid site identity, mapping or address group");
  EXPECT_TRUE(validate_image_facts(image).failed());
  EXPECT_TRUE(validate_image_facts(example_image(), diagnostic).succeeded());
  EXPECT_EQ(messages.size(), 1u);
}

TEST(ConSanImageFacts, RejectsDuplicateKeysNamesDescriptorsAndSharedSites) {
  const ImageFacts baseline = example_image();
  for (uint32_t kind = 0; kind < 4; ++kind) {
    ImageFacts image = baseline;
    if (kind == 0) {
      image.entries[1].key = image.entries[0].key;
      image.entries[1].binding.original_descriptor_offset = image.entries[1].key;
      const std::array<EntryKey, 2> keys = {image.entries[0].key, image.entries[1].key};
      image.selection_profile = selection_profile_digest(keys);
    } else if (kind == 1) {
      image.entries[1].binding.name = image.entries[0].binding.name;
    } else if (kind == 2) {
      image.entries[1].binding.prepared_descriptor_offset =
          image.entries[0].binding.prepared_descriptor_offset;
    } else {
      image.entries[1].selected->sites[0].original_offset =
          image.entries[0].selected->sites[0].original_offset;
    }
    image.entries[1].selected->entry = image.entries[1].binding;
    EXPECT_TRUE(validate_image_facts(image).failed()) << kind;
  }
}

TEST(ConSanImageFacts, NativeCodeRangesAreBoundedOrderedAndDisjoint) {
  const ImageFacts baseline = example_image();
  for (uint32_t kind = 0; kind < 6; ++kind) {
    SCOPED_TRACE(kind);
    ImageFacts image = baseline;
    ImageEntryFacts &native = image.entries.back();
    if (kind == 0) {
      native.native_code.clear();
    } else if (kind == 1) {
      native.native_code.front().bytes = 0;
    } else if (kind == 2) {
      native.native_code.front().bytes = image.original_text_bytes;
    } else if (kind == 3) {
      ++native.native_code.front().prepared_offset;
    } else if (kind == 4) {
      native.native_code.push_back(native.native_code.front());
    } else {
      native.native_code.front().original_offset =
          image.entries[0].selected->sites[0].original_offset;
      native.binding.original_entry_offset = native.native_code.front().original_offset;
    }
    EXPECT_TRUE(validate_image_facts(image).failed());
  }
  ImageFacts selected = baseline;
  selected.entries[0].native_code = baseline.entries.back().native_code;
  EXPECT_TRUE(validate_image_facts(selected).failed());
}

TEST(ConSanImageFacts, DecoderRejectsNativeRangeCountBeforeAllocatingClaimedStorage) {
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(example_image());
  ASSERT_TRUE(encoded.succeeded());
  std::vector<uint8_t> bytes = encoded.value();
  // The last native row has one 24-byte range followed by a zero manifest length.
  wire::write_u32(bytes, bytes.size() - 36, UINT32_MAX);
  rehash(bytes);
  util::StringDiagnostic error;
  EXPECT_TRUE(decode_image_facts(bytes, error.emitter()).failed());
  EXPECT_EQ(error.message(), "truncated native code ranges");
}

TEST(ConSanImageFacts, RejectsReboundEntryImageAndSelectionFacts) {
  const ImageFacts baseline = example_image();
  for (uint32_t kind = 0; kind < 4; ++kind) {
    ImageFacts image = baseline;
    if (kind == 0) {
      image.entries[0].selected->entry.name = "rebound";
    } else if (kind == 1) {
      image.entries[1].selected->prepared.sha256[0] ^= 1;
    } else if (kind == 2) {
      image.selection_profile[0] ^= 1;
    } else {
      image.entries[1].selected->target = 1100;
    }
    EXPECT_TRUE(validate_image_facts(image).failed()) << kind;
  }
}

TEST(ConSanImageFacts, NativeCannotClaimDetectorFactsOrGrownResources) {
  const ImageFacts baseline = example_image();
  for (uint32_t kind = 0; kind < 4; ++kind) {
    ImageFacts image = baseline;
    ImageEntryFacts &native = image.entries.back();
    if (kind == 0) {
      native.selected = image.entries[0].selected;
    } else if (kind == 1) {
      ++native.binding.prepared_group_bytes;
    } else if (kind == 2) {
      ++native.binding.prepared_sgprs;
    } else {
      native.disposition = static_cast<EntryDisposition>(99);
    }
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
    entry.native_code = {{.original_offset = entry.binding.original_entry_offset,
                          .prepared_offset = entry.binding.prepared_entry_offset,
                          .bytes = 4}};
  }
  image.selection_profile = selection_profile_digest({});
  util::StringDiagnostic error;
  EXPECT_TRUE(validate_image_facts(image, error.emitter()).failed());
  EXPECT_EQ(error.message(), "selected entries must share one qualified scope");
}

TEST(ConSanImageFacts, BoundedAggregateMetadataAndSiteCounts) {
  const ImageFacts image = example_image();
  ContractLimits limits;
  limits.sites = 1;
  util::StringDiagnostic error;
  EXPECT_TRUE(encode_image_facts(image, error.emitter(), limits).failed());
  EXPECT_EQ(error.message(), "aggregate image facts exceed budget");
  limits = {};
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(image);
  ASSERT_TRUE(encoded.succeeded());
  limits.metadata_bytes = encoded.value().size() - 1;
  EXPECT_TRUE(encode_image_facts(image, error.emitter(), limits).failed());
  EXPECT_EQ(error.message(), "native code ranges exceed metadata budget");
  EXPECT_TRUE(decode_image_facts(encoded.value(), error.emitter(), limits).failed());
}

TEST(ConSanImageFacts, DecoderRejectsTruncationCorruptionAndOversizedEntryClaims) {
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(example_image());
  ASSERT_TRUE(encoded.succeeded());
  for (size_t extent = 0; extent < encoded.value().size(); ++extent) {
    EXPECT_TRUE(decode_image_facts(std::span(encoded.value()).first(extent)).failed()) << extent;
  }
  std::vector<uint8_t> bytes = encoded.value();
  bytes.back() ^= 1;
  EXPECT_TRUE(decode_image_facts(bytes).failed());
  bytes = encoded.value();
  wire::write_u32(bytes, 64 + 140, std::numeric_limits<uint32_t>::max());
  rehash(bytes);
  util::StringDiagnostic error;
  EXPECT_TRUE(decode_image_facts(bytes, error.emitter()).failed());
  EXPECT_EQ(error.message(), "image entry count exceeds budget");
  bytes = encoded.value();
  bytes[8] ^= 1;
  EXPECT_TRUE(decode_image_facts(bytes, error.emitter()).failed());
  EXPECT_EQ(error.message(), "producer contract mismatch");
}

TEST(ConSanImageFacts, ReadEarlierWidthProfileWithoutAdmittingNewSitesUnderItsIdentity) {
  ImageFacts image = example_image(kPriorOptions);
  EXPECT_EQ(image_selection_profile_digest(image), image.selection_profile);
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(image);
  ASSERT_TRUE(encoded.succeeded());
  EXPECT_TRUE(decode_image_facts(encoded.value()).succeeded());
  image.entries.front().selected->sites.front().width = 12;
  EXPECT_TRUE(validate_image_facts(image).failed());
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
  for (const TargetProfile &target : kNativeLdsTargets) {
    for (const uint32_t wave_size : {32u, 64u}) {
      if (!supports_wave(target, wave_size)) {
        continue;
      }
      for (const ScopeKind scope :
           {ScopeKind::NativeLdsNoSynchronization, ScopeKind::NativeLdsWorkgroupBarrier,
            ScopeKind::NativeLdsWorkgroupBarrierLoop}) {
        for (const uint32_t width : {1u, 2u, 4u, 8u, 12u, 16u}) {
          SCOPED_TRACE(target.id);
          SCOPED_TRACE(wave_size);
          SCOPED_TRACE(static_cast<uint32_t>(scope));
          SCOPED_TRACE(width);
          const ImageFacts image = qualified_image(target, wave_size, scope, width);
          util::StringDiagnostic error;
          const FailureOr<std::vector<uint8_t>> encoded =
              encode_image_facts(image, error.emitter());
          ASSERT_TRUE(encoded.succeeded()) << error.message();
          const FailureOr<ImageFacts> decoded =
              decode_image_facts(encoded.value(), error.emitter());
          ASSERT_TRUE(decoded.succeeded()) << error.message();
          EXPECT_EQ(decoded.value(), image);
          EXPECT_EQ(selected_scope(decoded.value()), scope);
          for (const ImageEntryFacts &entry : decoded.value().entries) {
            if (!entry.selected) {
              continue;
            }
            const Manifest &manifest = *entry.selected;
            const uint32_t threads = 2 * wave_size;
            const FailureOr<LaunchLayout> layout = plan_launch(
                manifest, {{4 * threads, 1, 1}, {threads, 1, 1}}, 256, 0, error.emitter());
            ASSERT_TRUE(layout.succeeded()) << error.message();
            EXPECT_EQ(layout.value().workgroup_count, 4u);
            EXPECT_EQ(layout.value().wave_count, 8u);
            EXPECT_EQ(layout.value().row_stride,
                      report_row_bytes(1, scope != ScopeKind::NativeLdsNoSynchronization));
          }
        }
      }
    }
  }
}

TEST(ConSanImageFacts, MixedLdsScopesHaveAnExplicitPerEntrySelectionIdentity) {
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
  util::StringDiagnostic error;
  EXPECT_TRUE(encode_image_facts(image, error.emitter()).failed());
  EXPECT_EQ(error.message(), "image selection profile differs from entries");
  image.selection_profile = image_selection_profile_digest(image);
  EXPECT_NE(image.selection_profile,
            selection_profile_digest(keys, ScopeKind::NativeLdsWorkgroupBarrier));
  EXPECT_NE(image.selection_profile,
            selection_profile_digest(keys, ScopeKind::NativeLdsWorkgroupBarrierLoop));
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(image);
  ASSERT_TRUE(encoded.succeeded());
  const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value());
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value(), image);
  EXPECT_FALSE(selected_scope(decoded.value()));
  ImageFacts historical = image;
  for (ImageEntryFacts &entry : historical.entries) {
    if (entry.selected) {
      entry.selected->options = kPriorOptions;
    }
  }
  historical.selection_profile = image_selection_profile_digest(historical);
  EXPECT_NE(historical.selection_profile, image.selection_profile);
  const FailureOr<std::vector<uint8_t>> historical_bytes = encode_image_facts(historical);
  ASSERT_TRUE(historical_bytes.succeeded());
  const FailureOr<ImageFacts> historical_decoded = decode_image_facts(historical_bytes.value());
  ASSERT_TRUE(historical_decoded.succeeded());
  EXPECT_EQ(historical_decoded.value(), historical);
  historical.entries.front().selected->options = mvp_options_digest();
  historical.selection_profile = image_selection_profile_digest(historical);
  EXPECT_TRUE(encode_image_facts(historical, error.emitter()).failed());
  EXPECT_EQ(error.message(), "selected entries must share one options profile");
}

TEST(ConSanImageFacts, BarrierBytesParticipateInTheInclusiveAggregateMetadataBudget) {
  const ImageFacts image =
      qualified_image(*target_profile(1250), 32, ScopeKind::NativeLdsWorkgroupBarrierLoop);
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(image);
  ASSERT_TRUE(encoded.succeeded());
  ContractLimits limits;
  limits.metadata_bytes = encoded.value().size();
  EXPECT_TRUE(encode_image_facts(image, {}, limits).succeeded());
  EXPECT_TRUE(decode_image_facts(encoded.value(), {}, limits).succeeded());
  --limits.metadata_bytes;
  util::StringDiagnostic error;
  EXPECT_TRUE(encode_image_facts(image, error.emitter(), limits).failed());
  EXPECT_EQ(error.message(), "native code ranges exceed metadata budget");
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

TEST(ConSanImageFacts, RequiredShapesPreserveQScopesSourcesAndMultiWorkgroupRows) {
  for (const TargetProfile &target : kNativeLdsTargets)
    for (const uint32_t wave_size : {32u, 64u}) {
      if (!supports_wave(target, wave_size))
        continue;
      for (const ScopeKind scope :
           {ScopeKind::NativeLdsNoSynchronization, ScopeKind::NativeLdsWorkgroupBarrier,
            ScopeKind::NativeLdsWorkgroupBarrierLoop}) {
        SCOPED_TRACE(target.id);
        SCOPED_TRACE(wave_size);
        SCOPED_TRACE(static_cast<uint32_t>(scope));
        ImageFacts image = qualified_image(target, wave_size, scope, 16);
        const uint32_t threads = 2 * wave_size;
        for (ImageEntryFacts &entry : image.entries) {
          entry.binding.required_workgroup_size = {threads, 1, 1};
          if (entry.selected)
            entry.selected->entry = entry.binding;
        }
        util::StringDiagnostic error;
        const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(image, error.emitter());
        ASSERT_TRUE(encoded.succeeded()) << error.message();
        const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value(), error.emitter());
        ASSERT_TRUE(decoded.succeeded()) << error.message();
        EXPECT_EQ(decoded.value(), image);
        ContractLimits limits;
        limits.metadata_bytes = encoded.value().size();
        EXPECT_TRUE(encode_image_facts(image, error.emitter(), limits).succeeded())
            << error.message();
        --limits.metadata_bytes;
        EXPECT_TRUE(encode_image_facts(image, error.emitter(), limits).failed());
        EXPECT_EQ(error.message(), "native code ranges exceed metadata budget");
        for (const ImageEntryFacts &entry : decoded.value().entries) {
          if (!entry.selected)
            continue;
          const FailureOr<LaunchLayout> layout = plan_launch(
              *entry.selected, {{4 * threads, 1, 1}, {threads, 1, 1}}, 256, 0, error.emitter());
          ASSERT_TRUE(layout.succeeded()) << error.message();
          EXPECT_EQ(layout.value().workgroup_count, 4u);
          EXPECT_EQ(layout.value().wave_count, 8u);
          EXPECT_EQ(layout.value().row_stride,
                    report_row_bytes(1, scope != ScopeKind::NativeLdsNoSynchronization));
          EXPECT_TRUE(plan_launch(*entry.selected, {{4 * wave_size, 1, 1}, {wave_size, 1, 1}}, 256,
                                  0, error.emitter())
                          .failed());
          EXPECT_NE(error.message().find("required workgroup shape"), std::string::npos);
        }
      }
    }
}

TEST(ConSanImageFacts, NativePrivateSegmentsArePreservedWhileSelectedPrivateSegmentsReject) {
  ImageFacts image = example_image();
  ImageEntryFacts &native = image.entries.back();
  native.binding.original_private_bytes = 68;
  native.binding.prepared_private_bytes = 68;
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(image);
  ASSERT_TRUE(encoded.succeeded());
  const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value());
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value(), image);
  ++native.binding.prepared_private_bytes;
  EXPECT_TRUE(validate_image_facts(image).failed());
  image = example_image();
  image.entries.front().binding.original_private_bytes = 68;
  image.entries.front().binding.prepared_private_bytes = 68;
  image.entries.front().selected->entry = image.entries.front().binding;
  EXPECT_TRUE(validate_image_facts(image).failed());
}

TEST(ConSanImageFacts, TraceScopeKeepsBoundaryBudgetAndSelectionSeparateFromQLds) {
  ImageFacts image = example_image();
  for (ImageEntryFacts &entry : image.entries) {
    if (!entry.selected) {
      continue;
    }
    entry.binding.workgroup_id_scalar = kWorkgroupIdTtmp9Selector;
    entry.binding.original_group_bytes = entry.binding.prepared_group_bytes = 0;
    Manifest &manifest = *entry.selected;
    manifest.entry = entry.binding;
    manifest.scope = ScopeKind::ExecutionTrace;
    manifest.options = profile_options_digest(manifest.scope);
    manifest.sites.clear();
    const uint64_t probe = 2048 + 256 * ((entry.key - 2048) / 64);
    manifest.boundaries = {{.kind = BoundaryKind::Entry,
                            .original_offset = entry.binding.original_entry_offset,
                            .prepared_guest_offset = entry.binding.prepared_entry_offset,
                            .prepared_guest_size = 0,
                            .probe_offset = probe,
                            .probe_size = 16},
                           {.kind = BoundaryKind::Exit,
                            .original_offset = entry.binding.original_entry_offset + 64,
                            .prepared_guest_offset = probe + 64,
                            .prepared_guest_size = 4,
                            .probe_offset = probe + 32,
                            .probe_size = 16}};
  }
  const std::array<EntryKey, 2> keys = {2048, 2112};
  image.selection_profile = selection_profile_digest(keys, ScopeKind::ExecutionTrace);
  EXPECT_NE(image.selection_profile,
            selection_profile_digest(keys, ScopeKind::NativeLdsWorkgroupBarrier));
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(image);
  ASSERT_TRUE(encoded.succeeded());
  const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value());
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value(), image);
  ImageFacts with_lds_certificate = image;
  NoProbesNeededProof proof = {.instruction_count = 1, .code_bytes = 4};
  proof.code_sha256[0] = 1;
  with_lds_certificate.entries.back().no_probes_needed = proof;
  util::StringDiagnostic profile_error;
  EXPECT_TRUE(validate_image_facts(with_lds_certificate, profile_error.emitter()).failed());
  EXPECT_EQ(profile_error.message(),
            "LDS no-probes-needed certificate cannot qualify execution trace");
  ContractLimits limits;
  limits.metadata_bytes = encoded.value().size();
  EXPECT_TRUE(encode_image_facts(image, {}, limits).succeeded());
  --limits.metadata_bytes;
  util::StringDiagnostic error;
  EXPECT_TRUE(encode_image_facts(image, error.emitter(), limits).failed());
  EXPECT_EQ(error.message(), "native code ranges exceed metadata budget");
  limits = {};
  limits.sites = 3;
  EXPECT_TRUE(encode_image_facts(image, error.emitter(), limits).failed());
  EXPECT_EQ(error.message(), "aggregate image facts exceed budget");
  ImageFacts mixed = image;
  mixed.entries[1] = example_image().entries[1];
  ASSERT_TRUE(validate_manifest(*mixed.entries[1].selected).succeeded());
  EXPECT_TRUE(validate_image_facts(mixed, error.emitter()).failed());
  EXPECT_NE(error.message().find("trace and LDS entries cannot share"), std::string::npos);
  // A selected entry cannot reuse another entry's original or prepared boundary.
  ImageFacts aliased = image;
  aliased.entries[1].selected->boundaries.back().original_offset =
      aliased.entries[0].selected->boundaries.back().original_offset;
  ASSERT_TRUE(validate_manifest(*aliased.entries[1].selected).succeeded());
  EXPECT_TRUE(validate_image_facts(aliased).failed());
  aliased = image;
  aliased.entries[1].selected->boundaries.front().probe_offset =
      aliased.entries[0].selected->boundaries.front().probe_offset;
  ASSERT_TRUE(validate_manifest(*aliased.entries[1].selected).succeeded());
  EXPECT_TRUE(validate_image_facts(aliased).failed());
}

} // namespace
} // namespace rocjitsu::consan
