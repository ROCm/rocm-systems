// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "target_fixture.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/code/consan/image/artifact.h"
#include "rocjitsu/code/patch/consan/prepare.h"

#include <gtest/gtest.h>

#include <cstring>

namespace rocjitsu::consan {
namespace {

class ConSanTargetStatic : public ::testing::TestWithParam<test::TargetMode> {};

TEST_P(ConSanTargetStatic, AssemblyLoweringPreservesConcreteIdentityAndLoadableImages) {
  const auto mode = GetParam();
  const auto *target = target_profile(mode.target);
  ASSERT_NE(target, nullptr);
  for (const uint32_t fixture : {1u, 2u, 3u, 4u, 6u, 11u, 12u, 13u, 14u}) {
    SCOPED_TRACE(mode.target + std::string(" native-") + std::to_string(fixture));
    const auto original =
        test::read_fixture(test::target_fixture_path(mode, "native-" + std::to_string(fixture)));
    ASSERT_FALSE(original.empty());
    InputError error;
    const auto input =
        build_input_program(original, *target_profile(mode.target), "lds_producer_consumer", error);
    ASSERT_TRUE(input.succeeded()) << error.message;
    EXPECT_EQ(input.value().wave_size, mode.wave_size);
    ASSERT_FALSE(input.value().accesses.empty());
    const auto prepared =
        prepare(original, *target_profile(mode.target), "lds_producer_consumer", error);
    ASSERT_TRUE(prepared.succeeded()) << error.message;
    EXPECT_EQ(prepared.value().manifest.target, target->id);
    EXPECT_EQ(prepared.value().manifest.wave_size, mode.wave_size);
    ContractError image_error;
    EXPECT_TRUE(verify_artifact_images(prepared.value().manifest, original, prepared.value().image,
                                       &image_error)
                    .succeeded())
        << image_error.message;
    const auto encoded = encode_manifest(prepared.value().manifest);
    ASSERT_TRUE(encoded.succeeded());
    const auto decoded = decode_manifest(encoded.value());
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_EQ(decoded.value(), prepared.value().manifest);
  }
}

TEST_P(ConSanTargetStatic, GdsAndAccumulatorAliasingAreRejectedAtTheirActualBoundaries) {
  const auto mode = GetParam();
  const auto *target = target_profile(mode.target);
  auto original = test::read_fixture(test::target_fixture_path(mode, "native-1"));
  InputError error;
  const auto input =
      build_input_program(original, *target_profile(mode.target), "lds_producer_consumer", error);
  ASSERT_TRUE(input.succeeded()) << error.message;
  if (target->arch == ROCJITSU_CODE_ARCH_CDNA3 || target->arch == ROCJITSU_CODE_ARCH_CDNA4 ||
      target->arch == ROCJITSU_CODE_ARCH_RDNA3) {
    const auto offset = input.value().text_file_offset + input.value().accesses[0].original_offset;
    auto gds = original;
    const auto word = *wire::read_u32(gds, offset);
    ASSERT_TRUE(wire::write_u32(
        gds, offset, word | (1u << (mode.wave_size == 64 && target->id < 1000 ? 16 : 17))));
    EXPECT_TRUE(
        build_input_program(gds, *target_profile(mode.target), "lds_producer_consumer", error)
            .failed());
    EXPECT_EQ(error.code, InputErrorCode::UnsupportedAccess) << error.message;
  }
  if (target->id != 942 && target->id != 950)
    return;
  auto prepared = prepare(original, *target_profile(mode.target), "lds_producer_consumer", error);
  ASSERT_TRUE(prepared.succeeded()) << error.message;
  auto &artifact = prepared.value();
  AmdGpuCodeObject image(artifact.image.data(), artifact.image.size());
  const auto &text = *image.text_sections().front();
  const auto kernels = scan_kernel_descriptors(artifact.image, text.sectionOffset(), text.size(),
                                               text.sectionHeaderIndex());
  ASSERT_EQ(kernels.size(), 1u);
  auto descriptor = kernels[0].descriptor;
  namespace kd = rocjitsu::amdhsa;
  EXPECT_EQ(
      (RJ_AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET) +
       1) *
          4,
      artifact.manifest.entry.prepared_vgprs);
  RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 0);
  std::memcpy(artifact.image.data() + kernels[0].descriptor_file_offset, &descriptor,
              sizeof(descriptor));
  artifact.manifest.prepared.sha256 = digest(artifact.image);
  ContractError image_error;
  EXPECT_TRUE(
      verify_artifact_images(artifact.manifest, original, artifact.image, &image_error).failed());
}

TEST_P(ConSanTargetStatic, HipWorkloadsMustContainQualifiedNativeLdsSites) {
  const auto mode = GetParam();
  for (const uint32_t fixture : {1u, 2u, 3u}) {
    SCOPED_TRACE(mode.target + std::string(" HIP-") + std::to_string(fixture));
    const auto original =
        test::read_fixture(test::target_fixture_path(mode, "hip-" + std::to_string(fixture)));
    ASSERT_FALSE(original.empty());
    InputError error;
    const auto input =
        build_input_program(original, *target_profile(mode.target), "lds_producer_consumer", error);
    if (std::string_view(mode.target) == "gfx1250") {
      EXPECT_TRUE(input.failed());
      EXPECT_EQ(error.code, InputErrorCode::UnsupportedInstruction);
      EXPECT_NE(error.message.find("global_prefetch_b8"), std::string::npos) << error.message;
      continue;
    }
    ASSERT_TRUE(input.succeeded()) << error.message;
    EXPECT_EQ(input.value().wave_size, mode.wave_size);
    ASSERT_FALSE(input.value().accesses.empty());
    uint32_t reads = 0, writes = 0;
    for (const auto &site : input.value().accesses)
      (site.kind == AccessKind::Read ? reads : writes)++;
    EXPECT_GT(reads, 0u);
    EXPECT_EQ(writes != 0, fixture != 3);
    const auto prepared =
        prepare(original, *target_profile(mode.target), "lds_producer_consumer", error);
    ASSERT_TRUE(prepared.succeeded()) << error.message;
    EXPECT_TRUE(validate_prepared(original, prepared.value(), error).succeeded()) << error.message;
  }
}

INSTANTIATE_TEST_SUITE_P(ConSanTargets, ConSanTargetStatic, ::testing::ValuesIn(test::kTargetModes),
                         [](const auto &info) {
                           return std::string(info.param.target) + "Wave" +
                                  std::to_string(info.param.wave_size);
                         });

} // namespace
} // namespace rocjitsu::consan
