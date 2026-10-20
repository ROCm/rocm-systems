// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/code/consan/image/artifact.h"
#include "rocjitsu/code/consan/image/metadata.h"
#include "rocjitsu/code/patch/consan/prepare.h"

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>
#include <iterator>

namespace rocjitsu::consan {
namespace {

class ConSanPrepare : public ::testing::Test {
protected:
  void SetUp() override {
#ifndef CONSAN_FIXTURE_DIR
    GTEST_SKIP() << "gfx1100 assembler/linker fixtures unavailable";
#else
    original_ = load("race");
    ASSERT_FALSE(original_.empty());
#endif
  }
  static std::vector<uint8_t> load(std::string_view name) {
#ifdef CONSAN_FIXTURE_DIR
    std::ifstream file(std::string(CONSAN_FIXTURE_DIR) + "/" + std::string(name) + ".hsaco",
                       std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
#else
    (void)name;
    return {};
#endif
  }
  std::vector<uint8_t> original_;
  InputError error_;
};

TEST_F(ConSanPrepare, AllAdmittedCompanionsHaveOwningValidatedReplayableArtifacts) {
  for (const std::string_view name :
       {"race", "disjoint", "read_read", "collision", "single_lane", "alias", "immediate",
        "four_ranges", "exhausted", "empty", "branched"}) {
    const auto original = load(name);
    const auto artifact =
        prepare(original, *target_profile("gfx1100"), "lds_producer_consumer", error_);
    ASSERT_TRUE(artifact.succeeded()) << name << ": " << error_.message;
    EXPECT_TRUE(validate_prepared(original, artifact.value(), error_).succeeded())
        << error_.message;
    ContractError image_error;
    EXPECT_TRUE(verify_artifact_images(artifact.value().manifest, original, artifact.value().image,
                                       &image_error)
                    .succeeded())
        << name << ": " << image_error.message;
    const auto encoded = encode_manifest(artifact.value().manifest);
    ASSERT_TRUE(encoded.succeeded());
    const auto decoded = decode_manifest(encoded.value());
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_EQ(decoded.value(), artifact.value().manifest);
    const auto repeated =
        prepare(original, *target_profile("gfx1100"), "lds_producer_consumer", error_);
    ASSERT_TRUE(repeated.succeeded());
    EXPECT_EQ(repeated.value().image, artifact.value().image);
    EXPECT_EQ(repeated.value().manifest, artifact.value().manifest);
  }
}

TEST_F(ConSanPrepare, NonExecutableOriginalCannotProduceAPublishableArtifact) {
  Elf64_Ehdr header{};
  std::memcpy(&header, original_.data(), sizeof(header));
  bool changed = false;
  for (size_t index = 0; index < header.e_phnum; ++index) {
    const size_t offset = header.e_phoff + index * sizeof(Elf64_Phdr);
    Elf64_Phdr segment{};
    std::memcpy(&segment, original_.data() + offset, sizeof(segment));
    if (segment.p_type == PT_LOAD && (segment.p_flags & PF_X) != 0) {
      segment.p_flags &= ~PF_X;
      std::memcpy(original_.data() + offset, &segment, sizeof(segment));
      changed = true;
    }
  }
  ASSERT_TRUE(changed);
  ASSERT_TRUE(
      build_input_program(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_)
          .succeeded());
  EXPECT_TRUE(
      prepare(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_).failed());
  EXPECT_NE(error_.message.find("executable .text"), std::string::npos);
}

TEST_F(ConSanPrepare, EntrySelectionIsIndependentOfTheExampleKernelName) {
  const auto bytes = load("renamed");
  const auto artifact = prepare(bytes, *target_profile("gfx1100"), "renamed_input", error_);
  ASSERT_TRUE(artifact.succeeded()) << error_.message;
  EXPECT_EQ(artifact.value().manifest.entry.name, "renamed_input");
  EXPECT_EQ(artifact.value().manifest.sites.size(), 2u);
  EXPECT_TRUE(prepare(bytes, *target_profile("gfx1100"), "lds_producer_consumer", error_).failed());
}

TEST_F(ConSanPrepare, ModifiedPreparedIsaAndMappingsFailAfterRecomputingTheirImageDigest) {
  auto artifact = prepare(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_);
  ASSERT_TRUE(artifact.succeeded()) << error_.message;
  const auto input =
      build_input_program(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_);
  ASSERT_TRUE(input.succeeded());
  auto broken = artifact.value();
  broken.manifest.sites[0].probe_offset += 4;
  EXPECT_TRUE(validate_manifest(broken.manifest).succeeded());
  EXPECT_TRUE(validate_prepared(original_, broken, error_).failed());
  EXPECT_NE(error_.message.find("canonical preservation plan"), std::string::npos);
  broken = artifact.value();
  // Mutate an original guest operand, not a file checksum or framing byte.
  broken
      .image[input.value().text_file_offset + broken.manifest.sites[0].prepared_guest_offset + 4] ^=
      1;
  broken.manifest.prepared.sha256 = digest(broken.image);
  EXPECT_TRUE(match_images(broken.manifest, original_, broken.image).succeeded());
  EXPECT_TRUE(validate_prepared(original_, broken, error_).failed());
}

TEST_F(ConSanPrepare, RegisterPressureReturnsAResourceErrorInsteadOfAnInternalAssertion) {
  const FailureOr<InputProgram> input =
      build_input_program(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_);
  ASSERT_TRUE(input.succeeded()) << error_.message;
  std::string metadata_error;
  const FailureOr<MetadataResources> metadata =
      inspect_metadata(original_, input.value().kernel, &metadata_error);
  ASSERT_TRUE(metadata.succeeded()) << metadata_error;
  for (const uint32_t declared : {80u, 82u}) {
    std::vector<uint8_t> image = original_;
    MetadataResources resources = metadata.value();
    resources.sgprs = declared;
    ASSERT_TRUE(
        update_metadata(image, input.value().kernel, resources, &metadata_error).succeeded())
        << metadata_error;
    const FailureOr<InputProgram> qualified =
        build_input_program(image, *target_profile("gfx1100"), "lds_producer_consumer", error_);
    ASSERT_TRUE(qualified.succeeded()) << error_.message;
    EXPECT_EQ(qualified.value().original_sgprs, declared);
    const FailureOr<PreparedArtifact> artifact =
        prepare(image, *target_profile("gfx1100"), "lds_producer_consumer", error_);
    if (declared == 80) {
      ASSERT_TRUE(artifact.succeeded()) << error_.message;
      EXPECT_EQ(artifact.value().manifest.entry.prepared_sgprs, 102u);
    } else {
      EXPECT_TRUE(artifact.failed());
      EXPECT_EQ(error_.code, InputErrorCode::ResourceLimit);
      EXPECT_EQ(error_.message, "no bounded, disjoint probe register/report plan");
    }
  }
}

TEST_F(ConSanPrepare, DescriptorAndMetadataMismatchCannotProduceAnArtifact) {
  const auto input =
      build_input_program(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_);
  ASSERT_TRUE(input.succeeded());
  auto descriptor = input.value().kernel.descriptor;
  descriptor.kernarg_size += 8;
  std::memcpy(original_.data() + input.value().kernel.descriptor_file_offset, &descriptor,
              sizeof(descriptor));
  EXPECT_TRUE(
      prepare(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_).failed());
  EXPECT_NE(error_.message.find("metadata"), std::string::npos);
}

TEST_F(ConSanPrepare, MetadataGrowthFailureDoesNotPartiallyMutateTheImage) {
  const auto input =
      build_input_program(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_);
  ASSERT_TRUE(input.succeeded());
  const auto saved = original_;
  std::string message;
  EXPECT_TRUE(
      update_metadata(original_, input.value().kernel, {4096, 8, 28, 16}, &message).failed());
  EXPECT_EQ(original_, saved);
  EXPECT_NE(message.find("wider"), std::string::npos);
}

TEST_F(ConSanPrepare, ConcreteElfTargetMismatchFailsEvenWhenTheRequestedTargetIsGfx1100) {
  Elf64_Ehdr header{};
  std::memcpy(&header, original_.data(), sizeof(header));
  header.e_flags ^= 1;
  std::memcpy(original_.data(), &header, sizeof(header));
  EXPECT_TRUE(
      prepare(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_).failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedTarget);
}

TEST_F(ConSanPrepare, Gfx1100Wave32RequiresTheInputSgprPaddingWorkaround) {
  const auto input =
      build_input_program(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_);
  ASSERT_TRUE(input.succeeded());
  auto descriptor = input.value().kernel.descriptor;
  set_kernel_descriptor_user_sgpr_count(ROCJITSU_CODE_ARCH_RDNA3, descriptor, 2);
  std::memcpy(original_.data() + input.value().kernel.descriptor_file_offset, &descriptor,
              sizeof(descriptor));
  EXPECT_TRUE(
      prepare(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_).failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedEntry);
  EXPECT_NE(error_.message.find("SGPR padding"), std::string::npos);
}

TEST_F(ConSanPrepare, TransportStartsUnenteredAndOwnsExactParentAndGeometryBindings) {
  const auto artifact =
      prepare(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_);
  ASSERT_TRUE(artifact.succeeded()) << error_.message;
  const auto &manifest = artifact.value().manifest;
  const LaunchDimensions shape{{64, 1, 1}, {64, 1, 1}};
  const auto image = initialize_transport(manifest, shape, 8, 0, 101, 0x10000);
  ASSERT_TRUE(image.succeeded());
  EXPECT_EQ(wire::read_u64(image.value().context, wire::kContextSerial), 101u);
  EXPECT_EQ(wire::read_u64(image.value().context, wire::kContextReportPointer), 0x10000u);
  EXPECT_EQ(wire::read_u64(image.value().context, wire::kContextReportBytes),
            image.value().report.size());
  for (uint32_t wave = 0; wave < 2; ++wave) {
    const size_t row = kReportHeaderBytes + wave * 576;
    EXPECT_EQ(wire::read_u32(image.value().report, row + wire::kRowStatus), 0u);
    EXPECT_EQ(wire::read_u64(image.value().report, row + wire::kRowSerial), 101u);
    EXPECT_EQ(wire::read_u32(image.value().report, row + wire::kRowWave), wave);
  }
  EXPECT_TRUE(initialize_transport(manifest, shape, 8, 0, 0, 0x10000).failed());
  EXPECT_TRUE(initialize_transport(manifest, shape, 8, 0, 101, 0).failed());
  EXPECT_TRUE(initialize_transport(manifest, shape, 8, 0, 101, UINT64_MAX - 63).failed());
  EXPECT_FALSE(wire::read_u32({}, 0).has_value());
  EXPECT_FALSE(wire::read_u64({}, SIZE_MAX).has_value());
}

} // namespace
} // namespace rocjitsu::consan
