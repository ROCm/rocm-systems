// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/contract.h"
#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/code/consan/target.h"
#include "rocjitsu/code/patch/kernarg_extension.h"
#include "util/sha256.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string>

namespace rocjitsu::consan {
namespace {

std::string hex(const Digest &digest_bytes) {
  constexpr std::string_view kDigits = "0123456789abcdef";
  std::string result;
  for (uint8_t byte : digest_bytes) {
    result += kDigits[byte >> 4];
    result += kDigits[byte & 15];
  }
  return result;
}

TEST(ConSanDigest, OptionsIdentityNamesTheSoftwareAddressGroupReservation) {
  const std::string reservation =
      "address-groups-per-site=" + std::to_string(kAddressGroupsPerSite) + ";";
  EXPECT_NE(kMvpOptions.find(reservation), std::string_view::npos);
  EXPECT_EQ(hex(mvp_options_digest()),
            "c581d4da9c5d8ef58ba2c21b4955e4aed86ea84c72473c310d63113fb05fb437");
}

TEST(ConSanDigest, KnownAnswersIncludePaddingAndMultipleBlocks) {
  const std::array<std::pair<std::string_view, std::string_view>, 4> answers = {{
      {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
      {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
      {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
       "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
      {"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
       "ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
       "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"},
  }};
  for (const auto &[input, expected] : answers) {
    util::Sha256 hash;
    hash.update(input);
    EXPECT_EQ(hex(hash.finish()), expected) << input;
  }
}

TEST(ConSanDigest, MillionBytesAndIncrementalUnalignedChunks) {
  const std::string input(1000000, 'a');
  util::Sha256 hash;
  for (size_t offset = 0; offset < input.size();) {
    const size_t count = std::min(size_t{73}, input.size() - offset);
    hash.update(input.data() + offset, count);
    offset += count;
  }
  EXPECT_EQ(hex(hash.finish()), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  EXPECT_EQ(hex(digest({reinterpret_cast<const uint8_t *>(input.data()), input.size()})),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

// Synthetic static facts exercise the new handoff only. These bytes are not a
// prepared ELF, and no test here claims compiler or GPU execution coverage.
Manifest example() {
  Manifest manifest;
  manifest.original = {.sha256 = digest(std::array<uint8_t, 256>{}), .bytes = 256};
  manifest.prepared = {.sha256 = digest(std::array<uint8_t, 1024>{}), .bytes = 1024};
  manifest.options = mvp_options_digest();
  manifest.target = 1100;
  manifest.wave_size = 32;
  manifest.code_object_abi = 5;
  manifest.original_text_bytes = 128;
  manifest.prepared_text_bytes = 768;
  manifest.entry = {.id = 0,
                    .name = "lds_producer_consumer",
                    .original_descriptor_offset = 128,
                    .prepared_descriptor_offset = 800,
                    .original_entry_offset = 0,
                    .prepared_entry_offset = 512,
                    .original_kernarg_bytes = 16,
                    .kernarg_alignment = 8,
                    .original_group_bytes = 8,
                    .prepared_group_bytes = 8,
                    .prepared_sgprs = 40,
                    .prepared_vgprs = 16};
  manifest.sites = {{.id = 0,
                     .analysis = AnalysisKind::LdsRaceDetection,
                     .access = AccessKind::Write,
                     .width = 4,
                     .original_offset = 0,
                     .original_size = 8,
                     .prepared_guest_offset = 32,
                     .prepared_guest_size = 8,
                     .probe_offset = 64,
                     .probe_size = 64,
                     .first_slot = 0},
                    {.id = 1,
                     .analysis = AnalysisKind::LdsRaceDetection,
                     .access = AccessKind::Read,
                     .width = 4,
                     .original_offset = 16,
                     .original_size = 8,
                     .prepared_guest_offset = 256,
                     .prepared_guest_size = 8,
                     .probe_offset = 288,
                     .probe_size = 64,
                     .first_slot = 4}};
  return manifest;
}

TEST(ConSanBoundedLaunch, FourCompleteGroupsKeepAllPartialWaves) {
  Manifest manifest = example();
  manifest.target = 1250;
  manifest.entry.workgroup_id_scalar = 117;
  ContractError error;
  const FailureOr<LaunchLayout> layout =
      plan_launch(manifest, {{260, 1, 1}, {65, 1, 1}}, 8, 0, &error);
  ASSERT_TRUE(layout.succeeded()) << error.message;
  EXPECT_EQ(layout.value().wave_count, 12u);
  EXPECT_EQ(layout.value().report_bytes, 64u + 12u * 576u);
}

TEST(ConSanBoundedLaunch, AllNativeTargetsKeepBoundedWorkgroupRowsAndAdmission) {
  for (const TargetProfile &target : kNativeLdsTargets)
    for (const uint32_t wave_size : {32u, 64u}) {
      if (!supports_wave(target, wave_size))
        continue;
      SCOPED_TRACE(target.name);
      SCOPED_TRACE(wave_size);
      Manifest manifest = example();
      manifest.target = target.id;
      manifest.wave_size = wave_size;
      manifest.entry.workgroup_id_scalar = target.id == 1201 || target.id == 1250 ? 117 : 15;
      for (const uint32_t groups : {1u, 4u, 16u})
        for (const uint32_t threads : {2 * wave_size, 2 * wave_size + 1}) {
          SCOPED_TRACE(groups);
          SCOPED_TRACE(threads);
          const uint32_t waves = (threads + wave_size - 1) / wave_size;
          const LaunchDimensions dimensions = {{groups * threads, 1, 1}, {threads, 1, 1}};
          ContractError error;
          const FailureOr<LaunchLayout> layout = plan_launch(manifest, dimensions, 8, 0, &error);
          ASSERT_TRUE(layout.succeeded()) << error.message;
          EXPECT_EQ(layout.value().workgroup_count, groups);
          EXPECT_EQ(layout.value().waves_per_workgroup, waves);
          EXPECT_EQ(layout.value().wave_count, groups * waves);
          EXPECT_EQ(layout.value().report_bytes, 64u + groups * waves * 576u);
          const FailureOr<TransportImage> transport =
              initialize_transport(manifest, dimensions, 8, 0, 1, 0x10000, &error);
          ASSERT_TRUE(transport.succeeded()) << error.message;
          for (uint32_t row = 0; row < groups * waves; ++row) {
            const size_t offset = kReportHeaderBytes + row * layout.value().row_stride;
            EXPECT_EQ(wire::read_u32(transport.value().report, offset + wire::kRowWorkgroupX),
                      row / waves);
            EXPECT_EQ(wire::read_u32(transport.value().report, offset + wire::kRowWave),
                      row % waves);
          }
        }
      ContractError error;
      EXPECT_TRUE(plan_launch(manifest, {{17 * wave_size, 1, 1}, {wave_size, 1, 1}}, 8, 0, &error)
                      .failed());
      EXPECT_EQ(error.code, ContractErrorCode::UnsupportedLaunch);
      manifest.entry.workgroup_id_scalar = UINT32_MAX;
      EXPECT_TRUE(plan_launch(manifest, {{wave_size, 1, 1}, {wave_size, 1, 1}}, 8, 0).succeeded());
      EXPECT_TRUE(
          plan_launch(manifest, {{4 * wave_size, 1, 1}, {wave_size, 1, 1}}, 8, 0, &error).failed());
      EXPECT_EQ(error.code, ContractErrorCode::UnsupportedLaunch);
    }
}

TEST(ConSanBoundedLaunch, ValidateAllRowsDimensionsIdentityAndBudgetsBeforeAllocation) {
  Manifest manifest = example();
  manifest.target = 1250;
  manifest.entry.workgroup_id_scalar = 117;
  ContractError error;
  for (const uint32_t groups : {1u, 4u, 16u}) {
    const LaunchDimensions dimensions{{65 * groups, 1, 1}, {65, 1, 1}};
    const FailureOr<LaunchLayout> layout = plan_launch(manifest, dimensions, 8, 0, &error);
    ASSERT_TRUE(layout.succeeded()) << error.message;
    EXPECT_EQ(layout.value().workgroup_count, groups);
    EXPECT_EQ(layout.value().waves_per_workgroup, 3u);
    EXPECT_EQ(layout.value().wave_count, 3 * groups);
    const FailureOr<TransportImage> transport =
        initialize_transport(manifest, dimensions, 8, 0, 1, 0x10000, &error);
    ASSERT_TRUE(transport.succeeded()) << error.message;
    for (uint32_t row = 0; row < 3 * groups; ++row) {
      const size_t offset = kReportHeaderBytes + row * layout.value().row_stride;
      EXPECT_EQ(wire::read_u32(transport.value().report, offset + wire::kRowWave), row % 3);
      EXPECT_EQ(wire::read_u32(transport.value().report, offset + wire::kRowWorkgroupX), row / 3);
      EXPECT_EQ(wire::read_u32(transport.value().report, offset + wire::kRowStatus), 0u);
    }
    ContractLimits limits;
    limits.report_bytes = layout.value().report_bytes;
    EXPECT_TRUE(plan_launch(manifest, dimensions, 8, 0, &error, limits).succeeded());
    --limits.report_bytes;
    EXPECT_TRUE(plan_launch(manifest, dimensions, 8, 0, &error, limits).failed());
    EXPECT_EQ(error.code, ContractErrorCode::ResourceLimit);
  }
  for (const LaunchDimensions dimensions : {LaunchDimensions{{0, 1, 1}, {65, 1, 1}},
                                            {{65, 1, 1}, {0, 1, 1}},
                                            {{65 * 17, 1, 1}, {65, 1, 1}},
                                            {{261, 1, 1}, {65, 1, 1}},
                                            {{260, 2, 1}, {65, 1, 1}},
                                            {{260, 1, 2}, {65, 1, 1}},
                                            {{260, 1, 1}, {65, 2, 1}},
                                            {{260, 1, 1}, {65, 1, 2}},
                                            {{UINT32_MAX, 1, 1}, {65, 1, 1}},
                                            {{UINT32_MAX, 1, 1}, {UINT32_MAX, 1, 1}},
                                            {{1025, 1, 1}, {1025, 1, 1}}}) {
    EXPECT_TRUE(plan_launch(manifest, dimensions, 8, 0, &error).failed());
    EXPECT_EQ(error.code, ContractErrorCode::UnsupportedLaunch);
  }
  manifest.entry.workgroup_id_scalar = UINT32_MAX;
  EXPECT_TRUE(plan_launch(manifest, {{65, 1, 1}, {65, 1, 1}}, 8, 0).succeeded());
  EXPECT_TRUE(plan_launch(manifest, {{260, 1, 1}, {65, 1, 1}}, 8, 0, &error).failed());
  EXPECT_EQ(error.code, ContractErrorCode::UnsupportedLaunch);
  manifest.entry.workgroup_id_scalar = 117;
  manifest.target = 1201;
  EXPECT_TRUE(plan_launch(manifest, {{260, 1, 1}, {65, 1, 1}}, 8, 0, &error).succeeded());
}

TEST(ConSanBoundedLaunch, DefaultReportCapNeverReducesQuotasOrSamplesWorkgroups) {
  Manifest manifest = example();
  manifest.target = 1250;
  manifest.entry.workgroup_id_scalar = 117;
  manifest.original.bytes = 65536;
  manifest.original_text_bytes = 16384;
  manifest.entry.original_descriptor_offset = 32768;
  manifest.prepared.bytes = 1048576;
  manifest.prepared_text_bytes = 262144;
  manifest.entry.prepared_descriptor_offset = 524288;
  manifest.sites.clear();
  for (uint32_t site = 0; site < 1024; ++site)
    manifest.sites.push_back({.id = site,
                              .analysis = AnalysisKind::LdsRaceDetection,
                              .width = 4,
                              .original_offset = uint64_t{site} * 8,
                              .original_size = 8,
                              .prepared_guest_offset = uint64_t{site} * 128,
                              .prepared_guest_size = 8,
                              .probe_offset = uint64_t{site} * 128 + 16,
                              .probe_size = 64,
                              .first_slot = site * kAddressGroupsPerSite});
  ASSERT_TRUE(validate_manifest(manifest).succeeded());
  ContractError error;
  EXPECT_TRUE(plan_launch(manifest, {{16384, 1, 1}, {1024, 1, 1}}, 8, 0, &error).failed());
  EXPECT_EQ(error.code, ContractErrorCode::ResourceLimit);
  EXPECT_EQ(manifest.sites.size(), 1024u);
  EXPECT_EQ(kAddressGroupsPerSite, 4u);
}

TEST(ConSanBoundedLaunch, ProbeRowOffsetsCannotWrapWhenSoftwareBudgetsAreRaised) {
  Manifest manifest = example();
  manifest.target = 1250;
  manifest.entry.workgroup_id_scalar = 117;
  manifest.original.bytes = 2 * 1024 * 1024;
  manifest.original_text_bytes = 1024 * 1024;
  manifest.entry.original_descriptor_offset = 1024 * 1024;
  manifest.prepared.bytes = 32 * 1024 * 1024;
  manifest.prepared_text_bytes = 16 * 1024 * 1024;
  manifest.entry.prepared_descriptor_offset = 24 * 1024 * 1024;
  manifest.sites.clear();
  for (uint32_t site = 0; site < 65536; ++site)
    manifest.sites.push_back({.id = site,
                              .analysis = AnalysisKind::LdsRaceDetection,
                              .width = 4,
                              .original_offset = uint64_t{site} * 8,
                              .original_size = 8,
                              .prepared_guest_offset = uint64_t{site} * 128,
                              .prepared_guest_size = 8,
                              .probe_offset = uint64_t{site} * 128 + 16,
                              .probe_size = 64,
                              .first_slot = site * kAddressGroupsPerSite});
  ASSERT_TRUE(validate_manifest(manifest).succeeded());
  ContractLimits limits;
  limits.report_bytes = UINT64_MAX;
  ContractError error;
  EXPECT_TRUE(plan_launch(manifest, {{16384, 1, 1}, {1024, 1, 1}}, 8, 0, &error, limits).failed());
  EXPECT_EQ(error.code, ContractErrorCode::ResourceLimit);
}

class ConSanContract : public ::testing::Test {
protected:
  void SetUp() override {
    const FailureOr<std::vector<uint8_t>> encoded = encode_manifest(example(), &error_);
    ASSERT_TRUE(encoded.succeeded()) << error_.message;
    bytes_ = encoded.value();
  }

  static constexpr size_t kHeader = 64;
  static constexpr size_t kEntry = kHeader + 144;

  void put32(size_t offset, uint32_t value) {
    ASSERT_LE(offset + 4, bytes_.size());
    for (size_t index = 0; index < 4; ++index)
      bytes_[offset + index] = static_cast<uint8_t>(value >> (index * 8));
  }

  void rehash() {
    const Digest checksum = digest(std::span<const uint8_t>(bytes_).subspan(kHeader));
    std::copy(checksum.begin(), checksum.end(), bytes_.begin() + 32);
  }

  void reframe() {
    const uint64_t payload_bytes = bytes_.size() - kHeader;
    put32(24, static_cast<uint32_t>(payload_bytes));
    put32(28, static_cast<uint32_t>(payload_bytes >> 32));
    rehash();
  }

  size_t count_offset() const { return kEntry + 80 + example().entry.name.size(); }
  size_t site_offset() const { return count_offset() + 4; }

  ContractError error_;
  std::vector<uint8_t> bytes_;
};

TEST_F(ConSanContract, UnspecifiedSiteAnalysisCannotBecomeValidFacts) {
  Manifest manifest = example();
  const SiteFacts unspecified;
  EXPECT_EQ(unspecified.analysis, AnalysisKind::Invalid);
  manifest.sites.front().analysis = unspecified.analysis;
  EXPECT_TRUE(validate_manifest(manifest, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::InvalidFacts);
  EXPECT_TRUE(encode_manifest(manifest, &error_).failed());
}

TEST_F(ConSanContract, OwningFactsRoundTripIsCanonicalAndLittleEndian) {
  const Manifest original = example();
  const FailureOr<Manifest> decoded = decode_manifest(bytes_, &error_);
  ASSERT_TRUE(decoded.succeeded()) << error_.message;
  EXPECT_EQ(decoded.value(), original);
  const FailureOr<std::vector<uint8_t>> reencoded = encode_manifest(decoded.value());
  ASSERT_TRUE(reencoded.succeeded());
  EXPECT_EQ(reencoded.value(), bytes_);
  EXPECT_EQ(bytes_.size(), 64u + 144u + 80u + original.entry.name.size() + 4u + 2u * 68u + 4u);
  // gfx1100 is encoded as the integer 1100, after the two image identities
  // and the options digest. This also pins the independently specified layout.
  EXPECT_EQ(bytes_[64 + 112], 0x4c);
  EXPECT_EQ(bytes_[64 + 113], 0x04);
  EXPECT_EQ(bytes_[64 + 114], 0);
  bytes_.assign(bytes_.size(), 0xa5);
  EXPECT_EQ(decoded.value(), original);
}

TEST_F(ConSanContract, EveryTruncatedPrefixAndTrailingBytesAreRejected) {
  for (size_t size = 0; size < bytes_.size(); ++size)
    EXPECT_TRUE(decode_manifest(std::span<const uint8_t>(bytes_).first(size)).failed()) << size;
  bytes_.push_back(0);
  EXPECT_TRUE(decode_manifest(bytes_, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::InvalidEncoding);
}

TEST_F(ConSanContract, WrongContractAndDigestAreDistinctFailures) {
  bytes_[8] ^= 1;
  EXPECT_TRUE(decode_manifest(bytes_, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::IncompatibleContract);
  bytes_[8] ^= 1;
  bytes_.back() ^= 1;
  EXPECT_TRUE(decode_manifest(bytes_, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::IdentityMismatch);
}

TEST_F(ConSanContract, TruncatedPayloadsWithValidFramingStillFailAtTheFactsReader) {
  const std::vector<uint8_t> saved = bytes_;
  for (size_t size = kHeader; size < saved.size(); ++size) {
    bytes_.assign(saved.begin(), saved.begin() + size);
    reframe();
    EXPECT_TRUE(decode_manifest(bytes_, &error_).failed()) << size;
    EXPECT_EQ(error_.code, ContractErrorCode::InvalidEncoding) << size;
  }
}

TEST_F(ConSanContract, EnvelopeLengthsCannotWrapAndMagicIsRequired) {
  const std::vector<uint8_t> saved = bytes_;
  put32(24, std::numeric_limits<uint32_t>::max());
  put32(28, std::numeric_limits<uint32_t>::max());
  EXPECT_TRUE(decode_manifest(bytes_, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::InvalidEncoding);
  bytes_ = saved;
  bytes_[0] ^= 1;
  EXPECT_TRUE(decode_manifest(bytes_, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::InvalidEncoding);
}

TEST_F(ConSanContract, MetadataBudgetIsInclusiveAndCheckedBeforeParsing) {
  ContractLimits limits;
  limits.metadata_bytes = bytes_.size();
  EXPECT_TRUE(decode_manifest(bytes_, &error_, limits).succeeded());
  EXPECT_TRUE(encode_manifest(example(), &error_, limits).succeeded());
  --limits.metadata_bytes;
  EXPECT_TRUE(decode_manifest(bytes_, &error_, limits).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::ResourceLimit);
  EXPECT_TRUE(encode_manifest(example(), &error_, limits).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::ResourceLimit);
}

TEST_F(ConSanContract, CountAndStringLengthsCannotCauseUnboundedAllocation) {
  put32(count_offset(), std::numeric_limits<uint32_t>::max());
  rehash();
  EXPECT_TRUE(decode_manifest(bytes_, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::ResourceLimit);
  put32(count_offset(), 1);
  rehash();
  EXPECT_TRUE(decode_manifest(bytes_, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::InvalidEncoding);
  put32(kEntry + 4, std::numeric_limits<uint32_t>::max());
  rehash();
  EXPECT_TRUE(decode_manifest(bytes_, &error_).failed());
}

TEST_F(ConSanContract, ValidChecksumDoesNotBypassRequiredMappings) {
  // A prepared guest size of zero is missing; offset zero itself is legal.
  put32(site_offset() + 48, 0);
  rehash();
  EXPECT_TRUE(decode_manifest(bytes_, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::InvalidFacts);
}

TEST_F(ConSanContract, RejectsOverlappingPreparedBarrierMappings) {
  Manifest manifest = example();
  manifest.target = 1250;
  manifest.scope = ScopeKind::NativeLdsWorkgroupBarrier;
  manifest.sites.back().original_offset = 64;
  manifest.barriers = {{.original_drain = 8,
                        .original_signal = 12,
                        .original_wait = 16,
                        .prepared_signal = 128,
                        .prepared_wait = 132,
                        .completion_probe = 136,
                        .probe_size = 8},
                       {.original_drain = 32,
                        .original_signal = 36,
                        .original_wait = 40,
                        .prepared_signal = 144,
                        .prepared_wait = 164,
                        .completion_probe = 168,
                        .probe_size = 8}};
  // B1's signal starts exactly where B0's completion probe ends: adjacency
  // is legal, but sharing any instruction or probe bytes is not.
  ASSERT_TRUE(validate_manifest(manifest, &error_).succeeded()) << error_.message;
  const FailureOr<std::vector<uint8_t>> encoded = encode_manifest(manifest, &error_);
  ASSERT_TRUE(encoded.succeeded()) << error_.message;
  ASSERT_TRUE(decode_manifest(encoded.value(), &error_).succeeded()) << error_.message;

  struct Overlap {
    const char *name;
    uint64_t signal;
    uint64_t wait;
  };
  constexpr std::array<Overlap, 5> kOverlaps = {{
      {.name = "shared signal", .signal = 128, .wait = 164},
      {.name = "shared wait and callback", .signal = 144, .wait = 132},
      {.name = "shared signal, wait and callback", .signal = 128, .wait = 132},
      {.name = "signal inside another callback", .signal = 140, .wait = 164},
      {.name = "signal inside its own callback", .signal = 168, .wait = 164},
  }};
  for (const Overlap &overlap : kOverlaps) {
    SCOPED_TRACE(overlap.name);
    Manifest invalid = manifest;
    invalid.barriers.back().prepared_signal = overlap.signal;
    invalid.barriers.back().prepared_wait = overlap.wait;
    invalid.barriers.back().completion_probe = overlap.wait + 4;
    EXPECT_TRUE(validate_manifest(invalid, &error_).failed());
    EXPECT_EQ(error_.code, ContractErrorCode::InvalidFacts);
    EXPECT_TRUE(encode_manifest(invalid, &error_).failed());
    EXPECT_EQ(error_.code, ContractErrorCode::InvalidFacts);

    // Mutate the last encoded barrier directly, so the encoder cannot hide
    // a missing decoder check. Keep framing and checksum valid.
    bytes_ = encoded.value();
    constexpr size_t kBarrierBytes = 6 * sizeof(uint64_t) + 2 * sizeof(uint32_t);
    const size_t barrier = bytes_.size() - kBarrierBytes;
    wire::write_u64(bytes_, barrier + 24, overlap.signal);
    wire::write_u64(bytes_, barrier + 32, overlap.wait);
    wire::write_u64(bytes_, barrier + 40, overlap.wait + 4);
    rehash();
    EXPECT_TRUE(decode_manifest(bytes_, &error_).failed());
    EXPECT_EQ(error_.code, ContractErrorCode::InvalidFacts);
  }
}

TEST_F(ConSanContract, RejectsBarrierCallbacksOverlappingAccessEnvelopes) {
  Manifest manifest = example();
  manifest.target = 1250;
  manifest.scope = ScopeKind::NativeLdsWorkgroupBarrier;
  manifest.sites.back().original_offset = 64;
  manifest.barriers = {{.original_drain = 8,
                        .original_signal = 12,
                        .original_wait = 16,
                        .prepared_signal = 400,
                        .prepared_wait = 404,
                        .completion_probe = 408,
                        .probe_size = 64}};
  const FailureOr<std::vector<uint8_t>> encoded = encode_manifest(manifest, &error_);
  ASSERT_TRUE(encoded.succeeded()) << error_.message;

  struct Mapping {
    const char *name;
    uint64_t wait;
    uint32_t probe_size;
    bool overlaps;
  };
  constexpr std::array<Mapping, 5> kMappings = {{
      {.name = "identical access callback", .wait = 60, .probe_size = 64, .overlaps = true},
      {.name = "partial overlap at start", .wait = 44, .probe_size = 64, .overlaps = true},
      {.name = "partial overlap at end", .wait = 92, .probe_size = 64, .overlaps = true},
      {.name = "contained callback", .wait = 76, .probe_size = 16, .overlaps = true},
      {.name = "adjacent wait and callback", .wait = 128, .probe_size = 64, .overlaps = false},
  }};
  for (const Mapping &mapping : kMappings) {
    SCOPED_TRACE(mapping.name);
    Manifest changed = manifest;
    changed.barriers.front().prepared_wait = mapping.wait;
    changed.barriers.front().completion_probe = mapping.wait + 4;
    changed.barriers.front().probe_size = mapping.probe_size;
    EXPECT_EQ(validate_manifest(changed, &error_).failed(), mapping.overlaps);
    EXPECT_EQ(encode_manifest(changed, &error_).failed(), mapping.overlaps);

    // Independently exercise decoding with valid framing and checksum.
    bytes_ = encoded.value();
    constexpr size_t kBarrierBytes = 6 * sizeof(uint64_t) + 2 * sizeof(uint32_t);
    const size_t barrier = bytes_.size() - kBarrierBytes;
    ASSERT_TRUE(wire::write_u64(bytes_, barrier + 32, mapping.wait));
    ASSERT_TRUE(wire::write_u64(bytes_, barrier + 40, mapping.wait + 4));
    put32(barrier + 48, mapping.probe_size);
    rehash();
    EXPECT_EQ(decode_manifest(bytes_, &error_).failed(), mapping.overlaps);
    if (mapping.overlaps) {
      EXPECT_EQ(error_.code, ContractErrorCode::InvalidFacts);
    }
  }
}

TEST_F(ConSanContract, BarrierMaximumWorkgroupMustPermitTwoWaves) {
  Manifest manifest = example();
  manifest.target = 1250;
  manifest.scope = ScopeKind::NativeLdsWorkgroupBarrier;
  manifest.sites.back().original_offset = 64;
  manifest.barriers = {{.original_drain = 8,
                        .original_signal = 12,
                        .original_wait = 16,
                        .prepared_signal = 400,
                        .prepared_wait = 404,
                        .completion_probe = 408,
                        .probe_size = 64}};
  manifest.entry.maximum_workgroup_size = 33;
  const FailureOr<std::vector<uint8_t>> encoded = encode_manifest(manifest, &error_);
  ASSERT_TRUE(encoded.succeeded()) << error_.message;
  ASSERT_TRUE(decode_manifest(encoded.value(), &error_).succeeded()) << error_.message;
  EXPECT_TRUE(plan_launch(manifest, {{33, 1, 1}, {33, 1, 1}}, 8, 0, &error_).succeeded());

  for (const uint32_t maximum : {1u, 31u, 32u}) {
    SCOPED_TRACE(maximum);
    manifest.entry.maximum_workgroup_size = maximum;
    EXPECT_TRUE(validate_manifest(manifest, &error_).failed());
    EXPECT_EQ(error_.code, ContractErrorCode::InvalidFacts);
    EXPECT_TRUE(encode_manifest(manifest, &error_).failed());
    bytes_ = encoded.value();
    // workgroup_id_scalar follows maximum_workgroup_size before the site count.
    put32(count_offset() - 2 * sizeof(uint32_t), maximum);
    rehash();
    EXPECT_TRUE(decode_manifest(bytes_, &error_).failed());
    EXPECT_EQ(error_.code, ContractErrorCode::InvalidFacts);
    EXPECT_EQ(error_.message,
              "barrier profile requires a maximum workgroup size permitting two waves");
  }
  Manifest no_barrier = example();
  no_barrier.entry.maximum_workgroup_size = 32;
  EXPECT_TRUE(validate_manifest(no_barrier, &error_).succeeded()) << error_.message;
}

TEST_F(ConSanContract, BarrierJournalCapacityIsInclusiveForTypedAndEncodedFacts) {
  Manifest manifest = example();
  manifest.target = 1250;
  manifest.scope = ScopeKind::NativeLdsWorkgroupBarrier;
  manifest.original.bytes = 4096;
  manifest.prepared.bytes = 8192;
  manifest.original_text_bytes = 2048;
  manifest.prepared_text_bytes = 4096;
  manifest.entry.original_descriptor_offset = 2048;
  manifest.entry.prepared_descriptor_offset = 4096;
  const auto barrier_at = [](uint32_t group) {
    const uint64_t original = 64 + uint64_t{group} * 16;
    const uint64_t prepared = 1024 + uint64_t{group} * 32;
    return BarrierFacts{.original_drain = original,
                        .original_signal = original + 4,
                        .original_wait = original + 8,
                        .prepared_signal = prepared,
                        .prepared_wait = prepared + 4,
                        .completion_probe = prepared + 8,
                        .probe_size = 8};
  };
  for (uint32_t group = 0; group < kSyncCapacity; ++group)
    manifest.barriers.push_back(barrier_at(group));
  ASSERT_TRUE(validate_manifest(manifest, &error_).succeeded()) << error_.message;
  const FailureOr<std::vector<uint8_t>> encoded = encode_manifest(manifest, &error_);
  ASSERT_TRUE(encoded.succeeded()) << error_.message;
  const FailureOr<Manifest> decoded = decode_manifest(encoded.value(), &error_);
  ASSERT_TRUE(decoded.succeeded()) << error_.message;
  EXPECT_EQ(decoded.value(), manifest);

  const BarrierFacts extra = barrier_at(kSyncCapacity);
  manifest.barriers.push_back(extra);
  EXPECT_TRUE(validate_manifest(manifest, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::ResourceLimit);
  EXPECT_TRUE(encode_manifest(manifest, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::ResourceLimit);

  // Supply all seventeen well-formed records, not just an excessive count:
  // an extent or checksum error must not mask the decoder's capacity guard.
  bytes_ = encoded.value();
  constexpr size_t kBarrierBytes = 6 * sizeof(uint64_t) + 2 * sizeof(uint32_t);
  const size_t count = bytes_.size() - kSyncCapacity * kBarrierBytes - sizeof(uint32_t);
  const size_t tail = bytes_.size();
  bytes_.resize(tail + kBarrierBytes);
  put32(count, kSyncCapacity + 1);
  const std::array<uint64_t, 6> offsets = {extra.original_drain, extra.original_signal,
                                           extra.original_wait,  extra.prepared_signal,
                                           extra.prepared_wait,  extra.completion_probe};
  for (size_t index = 0; index < offsets.size(); ++index)
    ASSERT_TRUE(wire::write_u64(bytes_, tail + index * sizeof(uint64_t), offsets[index]));
  put32(tail + 48, extra.probe_size);
  put32(tail + 52, extra.identity);
  reframe();
  EXPECT_TRUE(decode_manifest(bytes_, &error_).failed());
  // InvalidEncoding identifies the count check before parsing the records;
  // the later typed validation would instead return ResourceLimit.
  EXPECT_EQ(error_.code, ContractErrorCode::InvalidEncoding);
}

TEST_F(ConSanContract, DuplicateIdentityAndInvalidOwnersAreRejectedAfterChecksum) {
  put32(site_offset() + 68, 0);
  rehash();
  EXPECT_TRUE(decode_manifest(bytes_, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::InvalidFacts);
  put32(site_offset() + 68, 1);
  put32(site_offset() + 4, 1);
  rehash();
  EXPECT_TRUE(decode_manifest(bytes_, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::InvalidFacts);
}

TEST_F(ConSanContract, UnknownAnalysisRecordAccessAndScopeNeverDecodeAsEmpty) {
  // Recompute the checksum so an unset on-wire analysis reaches validation.
  const std::vector<uint8_t> original = bytes_;
  put32(site_offset() + 8, 0);
  rehash();
  EXPECT_TRUE(decode_manifest(bytes_, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::InvalidFacts);
  bytes_ = original;
  for (size_t offset :
       {site_offset() + 8, site_offset() + 12, site_offset() + 20, size_t{kHeader + 124}}) {
    const std::vector<uint8_t> saved = bytes_;
    put32(offset, 99);
    rehash();
    EXPECT_TRUE(decode_manifest(bytes_, &error_).failed()) << offset;
    EXPECT_EQ(error_.code, ContractErrorCode::InvalidFacts);
    bytes_ = saved;
  }
}

TEST_F(ConSanContract, OutOfBoundsAndWrappingPhysicalMappingsFail) {
  Manifest manifest = example();
  manifest.entry.original_descriptor_offset = std::numeric_limits<uint64_t>::max() - 8;
  EXPECT_TRUE(validate_manifest(manifest, &error_).failed());
  manifest = example();
  manifest.sites[0].probe_offset = std::numeric_limits<uint64_t>::max() - 3;
  EXPECT_TRUE(validate_manifest(manifest, &error_).failed());
  manifest = example();
  manifest.sites[0].original_offset = 124;
  EXPECT_TRUE(validate_manifest(manifest, &error_).failed());
}

TEST_F(ConSanContract, GuestMappingsAndReservationsMustBeUnambiguous) {
  Manifest manifest = example();
  manifest.sites[1].prepared_guest_offset = manifest.sites[0].prepared_guest_offset;
  EXPECT_TRUE(validate_manifest(manifest, &error_).failed());
  manifest = example();
  manifest.sites[1].probe_offset = manifest.sites[0].probe_offset;
  EXPECT_TRUE(validate_manifest(manifest, &error_).failed());
  manifest = example();
  manifest.sites[0].probe_offset = manifest.sites[0].prepared_guest_offset + 4;
  EXPECT_TRUE(validate_manifest(manifest, &error_).failed());
  manifest = example();
  manifest.sites[1].first_slot = 0;
  EXPECT_TRUE(validate_manifest(manifest, &error_).failed());
  manifest = example();
  manifest.sites.clear();
  EXPECT_TRUE(validate_manifest(manifest, &error_).failed());
}

TEST_F(ConSanContract, UnsupportedTargetAbiAndOptionsAreRejected) {
  Manifest manifest = example();
  manifest.target = 1150;
  EXPECT_TRUE(validate_manifest(manifest, &error_).failed());
  manifest = example();
  manifest.wave_size = 16;
  EXPECT_TRUE(validate_manifest(manifest, &error_).failed());
  manifest = example();
  manifest.code_object_abi = 6;
  EXPECT_TRUE(validate_manifest(manifest, &error_).failed());
  manifest = example();
  manifest.options[0] ^= 1;
  EXPECT_TRUE(validate_manifest(manifest, &error_).failed());
}

TEST_F(ConSanContract, EntryPresenceResourcesAndImageBudgetsAreChecked) {
  Manifest manifest = example();
  manifest.entry.kernarg_alignment = 3;
  EXPECT_TRUE(validate_manifest(manifest).failed());
  manifest = example();
  manifest.entry.name = std::string("kernel\0alias", 12);
  EXPECT_TRUE(validate_manifest(manifest).failed());
  manifest = example();
  manifest.entry.prepared_vgprs = 257;
  EXPECT_TRUE(validate_manifest(manifest).failed());
  manifest = example();
  manifest.entry.prepared_group_bytes = 4;
  EXPECT_TRUE(validate_manifest(manifest).failed());
  manifest = example();
  ContractLimits limits;
  limits.image_bytes = manifest.prepared.bytes - 1;
  EXPECT_TRUE(validate_manifest(manifest, &error_, limits).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::ResourceLimit);
}

TEST_F(ConSanContract, ExactImageMatchingChecksBothSizeAndDigest) {
  std::array<uint8_t, 256> original{};
  std::array<uint8_t, 1024> prepared{};
  const Manifest manifest = example();
  EXPECT_TRUE(match_images(manifest, original, prepared, &error_).succeeded());
  original[1] = 1;
  EXPECT_TRUE(match_images(manifest, original, prepared, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::IdentityMismatch);
  original[1] = 0;
  prepared[1] = 1;
  EXPECT_TRUE(match_images(manifest, original, prepared, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::IdentityMismatch);
  prepared[1] = 0;
  EXPECT_TRUE(match_images(manifest, std::span(original).first(255), prepared, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::IdentityMismatch);
}

constexpr LaunchDimensions kLaunch{.grid = {64, 1, 1}, .workgroup = {64, 1, 1}};

TEST_F(ConSanContract, TwoChecksShareOneStaticReservationPerAccessSite) {
  const FailureOr<LaunchLayout> layout = plan_launch(example(), kLaunch, 8, 0, &error_);
  ASSERT_TRUE(layout.succeeded()) << error_.message;
  EXPECT_EQ(layout.value(), (LaunchLayout{.wrapper_bytes = 32,
                                          .original_pointer_offset = 16,
                                          .context_pointer_offset = 24,
                                          .row_stride = 576,
                                          .report_bytes = 1216,
                                          .group_bytes = 8,
                                          .private_bytes = 0,
                                          .wave_count = 2,
                                          .workgroup_count = 1,
                                          .waves_per_workgroup = 2}));
  Manifest manifest = example();
  manifest.sites.pop_back();
  const FailureOr<LaunchLayout> single = plan_launch(manifest, kLaunch, 8, 0);
  ASSERT_TRUE(single.succeeded());
  EXPECT_EQ(layout.value().report_bytes - single.value().report_bytes,
            2u * 4u * kAccessRecordBytes);
}

TEST_F(ConSanContract, WrapperPreservesIndependentWireLayoutAndOriginalArguments) {
  struct ExpectedLayout {
    uint32_t original_bytes;
    uint32_t original_pointer;
    uint32_t context_pointer;
    uint32_t wrapper_bytes;
  };
  constexpr std::array cases = {
      ExpectedLayout{1, 8, 16, 24},   ExpectedLayout{8, 8, 16, 24},
      ExpectedLayout{16, 16, 24, 32}, ExpectedLayout{17, 24, 32, 40},
      ExpectedLayout{64, 64, 72, 80}, ExpectedLayout{65536, 65536, 65544, 65552}};
  for (const ExpectedLayout &expected : cases) {
    Manifest manifest = example();
    manifest.entry.original_kernarg_bytes = expected.original_bytes;
    const FailureOr<LaunchLayout> planned = plan_launch(manifest, kLaunch, 8, 0);
    ASSERT_TRUE(planned.succeeded());
    EXPECT_EQ(planned.value().wrapper_bytes, expected.wrapper_bytes);
    EXPECT_EQ(planned.value().original_pointer_offset, expected.original_pointer);
    EXPECT_EQ(planned.value().context_pointer_offset, expected.context_pointer);
    const std::optional<KernargExtensionLayout> generic =
        make_kernarg_extension_layout(expected.original_bytes, kContextPointerPayloads);
    ASSERT_TRUE(generic);
    std::vector<uint8_t> wrapper(expected.wrapper_bytes, 0xff);
    std::vector<uint8_t> original(expected.original_bytes, 0xa5);
    const std::array<uint8_t, 8> pointer = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::array<KernargExtensionPayloadWrite, 1> writes = {
        {{.data = pointer.data(), .size = 8}}};
    ASSERT_TRUE(
        write_kernarg_extension_wrapper(wrapper, *generic, original.data(), 0x1234, writes));
    std::vector<uint8_t> expected_bytes(expected.wrapper_bytes);
    std::copy(original.begin(), original.end(), expected_bytes.begin());
    expected_bytes[expected.original_pointer] = 0x34;
    expected_bytes[expected.original_pointer + 1] = 0x12;
    std::copy(pointer.begin(), pointer.end(), expected_bytes.begin() + expected.context_pointer);
    EXPECT_EQ(wrapper, expected_bytes);
  }
}

TEST_F(ConSanContract, GeometryAndDynamicPrivateRequirementsAreExplicit) {
  for (const LaunchDimensions &dimensions :
       {LaunchDimensions{.grid = {128, 1, 1}, .workgroup = {64, 1, 1}},
        LaunchDimensions{.grid = {32, 2, 1}, .workgroup = {32, 2, 1}},
        LaunchDimensions{.grid = {64, 1, 1}, .workgroup = {32, 1, 1}}, LaunchDimensions{}}) {
    EXPECT_TRUE(plan_launch(example(), dimensions, 8, 0, &error_).failed());
    EXPECT_EQ(error_.code, ContractErrorCode::UnsupportedLaunch);
  }
  EXPECT_TRUE(plan_launch(example(), kLaunch, 8, 4, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::UnsupportedLaunch);
}

TEST_F(ConSanContract, ReportBudgetAndWrapperOverflowFailBeforeSubmission) {
  ContractLimits limits;
  limits.report_bytes = 1216;
  EXPECT_TRUE(plan_launch(example(), kLaunch, 8, 0, &error_, limits).succeeded());
  --limits.report_bytes;
  EXPECT_TRUE(plan_launch(example(), kLaunch, 8, 0, &error_, limits).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::ResourceLimit);
  Manifest manifest = example();
  manifest.entry.original_kernarg_bytes = std::numeric_limits<uint32_t>::max();
  EXPECT_TRUE(encode_manifest(manifest, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::ResourceLimit);
  EXPECT_TRUE(plan_launch(manifest, kLaunch, 8, 0, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::ResourceLimit);
}

TEST_F(ConSanContract, SegmentMinimaPreserveOriginalPacketExtents) {
  const FailureOr<LaunchLayout> larger = plan_launch(example(), kLaunch, 128, 0);
  ASSERT_TRUE(larger.succeeded());
  EXPECT_EQ(larger.value().group_bytes, 128);
  const FailureOr<LaunchLayout> smaller = plan_launch(example(), kLaunch, 0, 0);
  ASSERT_TRUE(smaller.succeeded());
  EXPECT_EQ(smaller.value().group_bytes, 8);
  EXPECT_TRUE(plan_launch(example(), kLaunch, 65537, 0).failed());
}

TEST_F(ConSanContract, TargetWaveFactsRoundTripAndRowsFollowActualGeometry) {
  for (const TargetProfile &target : kNativeLdsTargets) {
    for (const uint32_t wave_size : {32u, 64u}) {
      Manifest manifest = example();
      manifest.target = target.id;
      manifest.wave_size = wave_size;
      manifest.entry.workgroup_id_scalar = target.id == 1250 || target.id == 1201 ? 117 : 15;
      if (!supports_wave(target, wave_size)) {
        EXPECT_TRUE(validate_manifest(manifest).failed());
        continue;
      }
      const FailureOr<std::vector<uint8_t>> encoded = encode_manifest(manifest);
      ASSERT_TRUE(encoded.succeeded());
      const FailureOr<Manifest> decoded = decode_manifest(encoded.value());
      ASSERT_TRUE(decoded.succeeded());
      EXPECT_EQ(decoded.value(), manifest);
      for (const uint32_t threads : {1u, wave_size, wave_size * 3, wave_size * 3 + 1, 1024u}) {
        const LaunchDimensions dimensions{{threads, 1, 1}, {threads, 1, 1}};
        const FailureOr<LaunchLayout> layout = plan_launch(manifest, dimensions, 8, 0);
        ASSERT_TRUE(layout.succeeded());
        const uint32_t waves = (threads + wave_size - 1) / wave_size;
        EXPECT_EQ(layout.value().wave_count, waves);
        EXPECT_EQ(layout.value().report_bytes,
                  kReportHeaderBytes + waves * layout.value().row_stride);
        const FailureOr<TransportImage> transport =
            initialize_transport(manifest, dimensions, 8, 0, 17, 0x10000);
        ASSERT_TRUE(transport.succeeded());
        EXPECT_EQ(wire::read_u32(transport.value().report, wire::kReportRows), waves);
        EXPECT_EQ(wire::read_u32(transport.value().context, wire::kContextWorkgroup), threads);
        const size_t last_row = kReportHeaderBytes + (waves - 1) * layout.value().row_stride;
        EXPECT_EQ(wire::read_u32(transport.value().report, last_row + wire::kRowWave), waves - 1);
        EXPECT_EQ(wire::read_u32(transport.value().report, last_row + wire::kRowStatus), 0u);
      }
      manifest.entry.maximum_workgroup_size = wave_size;
      EXPECT_TRUE(
          plan_launch(manifest, {{wave_size + 1, 1, 1}, {wave_size + 1, 1, 1}}, 8, 0).failed());
    }
  }
}

TEST_F(ConSanContract, OldProducerCannotBeDecodedAsTheNewMaskLayout) {
  FailureOr<std::vector<uint8_t>> bytes = encode_manifest(example());
  ASSERT_TRUE(bytes.succeeded());
  constexpr std::array<uint8_t, 16> old = {0x48, 0x9b, 0x15, 0x5f, 0x6c, 0x1a, 0x40, 0xf8,
                                           0xa3, 0x07, 0x8e, 0x2d, 0x73, 0xb6, 0x09, 0xc4};
  std::copy(old.begin(), old.end(), bytes.value().begin() + 8);
  EXPECT_TRUE(decode_manifest(bytes.value(), &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::IncompatibleContract);
}

} // namespace
} // namespace rocjitsu::consan
