// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "target_fixture.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/code/consan/image/artifact.h"
#include "rocjitsu/code/patch/consan/prepare.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>

namespace rocjitsu::consan {
namespace {

class ConSanTargetStatic : public ::testing::TestWithParam<test::TargetMode> {};

TEST_P(ConSanTargetStatic, CallableArtifactsPreserveStaticSitesAndPrivateFrameAdmission) {
  const test::TargetMode mode = GetParam();
  const TargetProfile &target = *target_profile(mode.target);
  for (const std::string_view fixture :
       {"race", "disjoint", "read_read", "collision", "single_lane", "alias", "immediate",
        "four_ranges", "exhausted", "empty", "branched", "native-14"}) {
    SCOPED_TRACE(fixture);
    const std::vector<uint8_t> original =
        test::read_fixture(test::target_fixture_path(mode, fixture));
    ASSERT_FALSE(original.empty());
    InputError error;
    const FailureOr<PreparedArtifact> called =
        prepare(original, target, "lds_producer_consumer", error);
    ASSERT_TRUE(called.succeeded()) << error.message;
    const FailureOr<InputProgram> input =
        build_input_program(original, target, "lds_producer_consumer", error);
    ASSERT_TRUE(input.succeeded());
    const std::vector<SiteFacts> &sites = called.value().manifest.sites;
    ASSERT_EQ(sites.size(), input.value().accesses.size());
    for (size_t index = 0; index < sites.size(); ++index) {
      const NativeLdsAccess &access = input.value().accesses[index];
      EXPECT_EQ(sites[index].id, access.site);
      EXPECT_EQ(sites[index].original_offset, access.original_offset);
      EXPECT_EQ(sites[index].access, access.kind);
      EXPECT_EQ(sites[index].first_slot, access.site * kAddressGroupsPerSite);
    }
    AmdGpuCodeObject image(called.value().image.data(), called.value().image.size());
    const Section *text = image.text_sections().front();
    const std::vector<KernelDescriptorInfo> kernels =
        scan_kernel_descriptors(called.value().image, text->sectionOffset(), text->size());
    ASSERT_EQ(kernels.size(), 1u);
    std::vector<uint32_t> words(text->size() / sizeof(uint32_t));
    std::memcpy(words.data(), text->data(), text->size());
    std::unique_ptr<Decoder> decoder =
        Decoder::create(default_isa_target_registry(), target.public_id);
    ASSERT_TRUE(decoder);
    size_t calls = 0;
    for (size_t cursor = 0; cursor < words.size();) {
      FailureOr<std::unique_ptr<Instruction>> instruction =
          decoder->decode_window(std::span(words).subspan(cursor), cursor * 4);
      ASSERT_TRUE(instruction.succeeded());
      calls += (instruction.value()->flags() & INDIRECT_CALL) != 0;
      cursor += instruction.value()->size() / sizeof(uint32_t);
    }
    EXPECT_EQ(calls, sites.size() + input.value().exit_offsets.size() + 1);
    EXPECT_EQ(kernels.front().descriptor.private_segment_fixed_size, 0u);
    EXPECT_EQ(kernels.front().descriptor.group_segment_fixed_size,
              input.value().kernel.descriptor.group_segment_fixed_size);
    PreparedArtifact corrupted = called.value();
    corrupted.image[text->sectionOffset() + sites.front().probe_offset] ^= 1;
    corrupted.manifest.prepared.sha256 = digest(corrupted.image);
    EXPECT_TRUE(validate_prepared(original, corrupted, error).failed());
  }
}

TEST(ConSanNativeWidthsStatic, AdmitGfx1250B64WithExactWidths) {
  const std::vector<uint8_t> original =
      test::read_fixture(test::target_fixture_path({"gfx1250", 32}, "native-9"));
  ASSERT_FALSE(original.empty());
  InputError error;
  const FailureOr<PreparedArtifact> artifact =
      prepare(original, *target_profile("gfx1250"), "lds_producer_consumer", error);
  ASSERT_TRUE(artifact.succeeded()) << error.message;
  ASSERT_EQ(artifact.value().manifest.sites.size(), 2u);
  EXPECT_EQ(artifact.value().manifest.sites[0].width, 8u);
  EXPECT_EQ(artifact.value().manifest.sites[1].width, 4u);
}

TEST_P(ConSanTargetStatic, QualifyOnlyWholeRegisterSingleRangeForms) {
  const test::TargetMode mode = GetParam();
  for (const auto &[name, width] : std::array<std::pair<const char *, uint32_t>, 5>{
           {{"u16", 2}, {"i16", 2}, {"b64", 8}, {"b128", 16}, {"b128-alias", 16}}}) {
    SCOPED_TRACE(name);
    const std::vector<uint8_t> original =
        test::read_fixture(test::target_fixture_path(mode, "width-" + std::string(name)));
    ASSERT_FALSE(original.empty());
    InputError error;
    const FailureOr<PreparedArtifact> artifact =
        prepare(original, *target_profile(mode.target), "lds_producer_consumer", error);
    ASSERT_TRUE(artifact.succeeded()) << error.message;
    ASSERT_EQ(artifact.value().manifest.sites.size(), 2u);
    for (const SiteFacts &site : artifact.value().manifest.sites)
      EXPECT_EQ(site.width, width);
    EXPECT_TRUE(validate_prepared(original, artifact.value(), error).succeeded()) << error.message;
  }
  for (const char *name : {"excluded-b8", "excluded-b96", "excluded-partial"}) {
    SCOPED_TRACE(name);
    const std::vector<uint8_t> original =
        test::read_fixture(test::target_fixture_path(mode, "width-" + std::string(name)));
    ASSERT_FALSE(original.empty());
    InputError error;
    EXPECT_TRUE(
        prepare(original, *target_profile(mode.target), "lds_producer_consumer", error).failed());
    EXPECT_EQ(error.code, InputErrorCode::UnsupportedAccess) << error.message;
  }
}

TEST_P(ConSanTargetStatic, QualifiedBarriersPreserveOriginalOperations) {
  const test::TargetMode mode = GetParam();
  const TargetProfile &target = *target_profile(mode.target);
  const bool split =
      target.arch == ROCJITSU_CODE_ARCH_RDNA4 || target.arch == ROCJITSU_CODE_ARCH_CDNA5;
  for (const char *fixture : {"native-15", "native-16", "native-20", "native-21", "native-24"}) {
    SCOPED_TRACE(fixture);
    const std::vector<uint8_t> original =
        test::read_fixture(test::target_fixture_path(mode, fixture));
    ASSERT_FALSE(original.empty());
    InputError error;
    const FailureOr<PreparedArtifact> artifact =
        prepare(original, target, "lds_producer_consumer", error);
    ASSERT_TRUE(artifact.succeeded()) << error.message;
    EXPECT_EQ(artifact.value().manifest.scope, ScopeKind::NativeLdsWorkgroupBarrier);
    EXPECT_EQ(artifact.value().manifest.sites.size(),
              std::string_view(fixture) == "native-24" ? 3u : 2u);
    EXPECT_EQ(artifact.value().manifest.barriers.size(),
              std::string_view(fixture) == "native-24" ? 2u : 1u);
    ContractError contract_error;
    const FailureOr<std::vector<uint8_t>> encoded =
        encode_manifest(artifact.value().manifest, &contract_error);
    ASSERT_TRUE(encoded.succeeded()) << contract_error.message;
    const FailureOr<Manifest> decoded = decode_manifest(encoded.value(), &contract_error);
    ASSERT_TRUE(decoded.succeeded()) << contract_error.message;
    EXPECT_EQ(decoded.value(), artifact.value().manifest);
    for (const BarrierFacts &barrier : artifact.value().manifest.barriers) {
      EXPECT_EQ(barrier.completion_probe, barrier.prepared_wait + 4);
      if (split) {
        // Only the completing wait moves to the AfterInst trampoline.
        EXPECT_EQ(barrier.prepared_signal, barrier.original_signal);
        EXPECT_NE(barrier.prepared_wait - barrier.prepared_signal,
                  barrier.original_wait - barrier.original_signal);
      } else {
        // The single legacy barrier both signals and completes the group.
        EXPECT_EQ(barrier.original_signal, barrier.original_wait);
        EXPECT_EQ(barrier.prepared_signal, barrier.prepared_wait);
        EXPECT_NE(barrier.prepared_wait, barrier.original_wait);
      }
    }
    EXPECT_TRUE(validate_prepared(original, artifact.value(), error).succeeded()) << error.message;
  }
}

TEST_P(ConSanTargetStatic, BarrierFactsRequireQualifiedMappingsScopeAndLaunch) {
  const test::TargetMode mode = GetParam();
  const TargetProfile &target = *target_profile(mode.target);
  const std::vector<uint8_t> original =
      test::read_fixture(test::target_fixture_path(mode, "native-15"));
  ASSERT_FALSE(original.empty());
  InputError error;
  const FailureOr<PreparedArtifact> artifact =
      prepare(original, target, "lds_producer_consumer", error);
  ASSERT_TRUE(artifact.succeeded()) << error.message;
  const Manifest &manifest = artifact.value().manifest;
  ContractError contract_error;
  EXPECT_TRUE(plan_launch(manifest, {{mode.wave_size, 1, 1}, {mode.wave_size, 1, 1}}, 128, 0,
                          &contract_error)
                  .failed());
  EXPECT_EQ(contract_error.code, ContractErrorCode::UnsupportedLaunch);
  Manifest invalid = manifest;
  invalid.barriers.front().original_signal += 4;
  EXPECT_TRUE(validate_manifest(invalid).failed());
  invalid = manifest;
  invalid.scope = ScopeKind::NativeLdsNoSynchronization;
  EXPECT_TRUE(validate_manifest(invalid).failed());
  invalid = manifest;
  const bool split =
      target.arch == ROCJITSU_CODE_ARCH_RDNA4 || target.arch == ROCJITSU_CODE_ARCH_CDNA5;
  invalid.target = split ? 1100 : 1201;
  EXPECT_TRUE(validate_manifest(invalid).failed());
}

TEST_P(ConSanTargetStatic, RejectMismatchedMissingDrainAndNonconvergentGroups) {
  const test::TargetMode mode = GetParam();
  const TargetProfile &target = *target_profile(mode.target);
  for (const char *fixture : {"native-17", "native-18", "native-19", "native-22", "native-23"}) {
    SCOPED_TRACE(fixture);
    const std::vector<uint8_t> original =
        test::read_fixture(test::target_fixture_path(mode, fixture));
    ASSERT_FALSE(original.empty());
    InputError error;
    EXPECT_TRUE(prepare(original, target, "lds_producer_consumer", error).failed());
    EXPECT_EQ(error.code, InputErrorCode::UnsupportedSynchronization) << error.message;
  }
}

TEST_P(ConSanTargetStatic, OriginalDrainMustWaitForAllPriorLds) {
  const test::TargetMode mode = GetParam();
  const TargetProfile &target = *target_profile(mode.target);
  const bool split =
      target.arch == ROCJITSU_CODE_ARCH_RDNA4 || target.arch == ROCJITSU_CODE_ARCH_CDNA5;
  const std::vector<uint8_t> original =
      test::read_fixture(test::target_fixture_path(mode, "native-15"));
  ASSERT_FALSE(original.empty());
  InputError error;
  const FailureOr<InputProgram> input =
      build_input_program(original, target, "lds_producer_consumer", error);
  ASSERT_TRUE(input.succeeded()) << error.message;
  ASSERT_EQ(input.value().barriers.size(), 1u);
  const BarrierFacts &barrier = input.value().barriers.front();
  const uint64_t drain_offset = input.value().text_file_offset + barrier.original_drain;
  const uint32_t drain = *wire::read_u32(original, drain_offset);
  // The split DS counter is the immediate. Legacy LGKM starts at bit four on
  // RDNA3 and bit eight on CDNA3/4; increasing that threshold leaves LDS pending.
  const uint32_t pending =
      split ? 1u : (target.arch == ROCJITSU_CODE_ARCH_RDNA3 ? 1u << 4 : 1u << 8);
  std::vector<uint8_t> changed = original;
  ASSERT_TRUE(wire::write_u32(changed, drain_offset, drain | pending));
  EXPECT_TRUE(build_input_program(changed, target, "lds_producer_consumer", error).failed());
  EXPECT_EQ(error.code, InputErrorCode::UnsupportedSynchronization) << error.message;
  EXPECT_EQ(error.text_offset, barrier.original_signal);
  if (!split) {
    // Waiting for every counter is stronger than waiting only for LGKM.
    changed = original;
    ASSERT_TRUE(wire::write_u32(changed, drain_offset, drain & 0xffff0000u));
    EXPECT_TRUE(prepare(changed, target, "lds_producer_consumer", error).succeeded())
        << error.message;
  }
}

TEST_P(ConSanTargetStatic, MaximumWorkgroupMustPermitTwoWavesBeforePreparation) {
  const test::TargetMode mode = GetParam();
  const TargetProfile &target = *target_profile(mode.target);
  for (const uint32_t maximum : {mode.wave_size - 1, mode.wave_size, mode.wave_size + 1}) {
    SCOPED_TRACE(maximum);
    const std::string fixture = "native-barrier-max" + std::to_string(maximum);
    const std::vector<uint8_t> original =
        test::read_fixture(test::target_fixture_path(mode, fixture));
    ASSERT_FALSE(original.empty());
    InputError error;
    const FailureOr<InputProgram> input =
        build_input_program(original, target, "lds_producer_consumer", error);
    if (maximum <= mode.wave_size) {
      EXPECT_TRUE(input.failed());
      EXPECT_EQ(error.code, InputErrorCode::UnsupportedEntry) << error.message;
      EXPECT_TRUE(prepare(original, target, "lds_producer_consumer", error).failed());
      EXPECT_EQ(error.code, InputErrorCode::UnsupportedEntry) << error.message;
    } else {
      ASSERT_TRUE(input.succeeded()) << error.message;
      EXPECT_EQ(input.value().maximum_workgroup_size, maximum);
      const FailureOr<PreparedArtifact> artifact =
          prepare(original, target, "lds_producer_consumer", error);
      ASSERT_TRUE(artifact.succeeded()) << error.message;
      EXPECT_EQ(artifact.value().manifest.entry.maximum_workgroup_size, maximum);
      EXPECT_TRUE(plan_launch(artifact.value().manifest, {{maximum, 1, 1}, {maximum, 1, 1}}, 128, 0)
                      .succeeded());
    }
  }
  // The minimum only belongs to barrier scope: one-wave no-sync kernels remain valid.
  const std::vector<uint8_t> no_barrier = test::read_fixture(
      test::target_fixture_path(mode, "native-no-barrier-max" + std::to_string(mode.wave_size)));
  ASSERT_FALSE(no_barrier.empty());
  InputError error;
  EXPECT_TRUE(prepare(no_barrier, target, "lds_producer_consumer", error).succeeded())
      << error.message;
}

TEST(ConSanBarrierStatic, RejectUnmatchedSignalAndWaitOperations) {
  const std::vector<uint8_t> original =
      test::read_fixture(test::target_fixture_path({"gfx1250", 32}, "native-15"));
  ASSERT_FALSE(original.empty());
  const TargetProfile &target = *target_profile("gfx1250");
  InputError error;
  const FailureOr<InputProgram> baseline =
      build_input_program(original, target, "lds_producer_consumer", error);
  ASSERT_TRUE(baseline.succeeded()) << error.message;
  ASSERT_EQ(baseline.value().barriers.size(), 1u);
  const BarrierFacts &barrier = baseline.value().barriers.front();
  const uint64_t text = baseline.value().text_file_offset;
  const uint32_t wait = *wire::read_u32(original, text + barrier.original_wait);
  const FailureOr<uint32_t> nop = build_s_nop(0, target.arch);
  ASSERT_TRUE(nop.succeeded());
  struct Mutation {
    const char *name;
    uint64_t offset;
    uint32_t word;
  };
  const std::array<Mutation, 3> mutations = {{
      {.name = "signal without wait", .offset = barrier.original_wait, .word = nop.value()},
      {.name = "wait without signal", .offset = barrier.original_signal, .word = nop.value()},
      // Replace the scalar EXEC restore immediately after the valid pair;
      // every other instruction and its original offset remain unchanged.
      {.name = "extra wait after pair", .offset = barrier.original_wait + 4, .word = wait},
  }};
  for (const Mutation &mutation : mutations) {
    SCOPED_TRACE(mutation.name);
    std::vector<uint8_t> changed = original;
    ASSERT_TRUE(wire::write_u32(changed, text + mutation.offset, mutation.word));
    EXPECT_TRUE(build_input_program(changed, target, "lds_producer_consumer", error).failed());
    EXPECT_EQ(error.code, InputErrorCode::UnsupportedSynchronization) << error.message;
  }
}

TEST(ConSanBarrierStatic, DiagnoseInterveningBarrierWithoutMemoryOperations) {
  const std::vector<uint8_t> original =
      test::read_fixture(test::target_fixture_path({"gfx1250", 32}, "native-21"));
  ASSERT_FALSE(original.empty());
  const TargetProfile &target = *target_profile("gfx1250");
  InputError error;
  const FailureOr<InputProgram> baseline =
      build_input_program(original, target, "lds_producer_consumer", error);
  ASSERT_TRUE(baseline.succeeded()) << error.message;
  ASSERT_EQ(baseline.value().barriers.size(), 1u);
  const BarrierFacts &barrier = baseline.value().barriers.front();
  ASSERT_GT(barrier.original_wait, barrier.original_signal + 4);
  // Replace the scalar comparison between signal and wait, preserving the
  // original group and every instruction offset without introducing memory.
  const uint32_t signal =
      cdna5::build_sop1(cdna5::kSBarrierSignalSop1, {.ssrc0 = cdna5::OPR_SSRC_M0})[0];
  std::vector<uint8_t> changed = original;
  ASSERT_TRUE(wire::write_u32(
      changed, baseline.value().text_file_offset + barrier.original_wait - 4, signal));
  EXPECT_TRUE(prepare(changed, target, "lds_producer_consumer", error).failed());
  EXPECT_EQ(error.code, InputErrorCode::UnsupportedSynchronization);
  EXPECT_EQ(error.text_offset, barrier.original_signal);
  EXPECT_NE(error.message.find("without intervening memory operations or barriers"),
            std::string::npos)
      << error.message;
}

TEST_P(ConSanTargetStatic, PreserveRealLoopBackedgesAndRejectMultipleBlockCycles) {
  const test::TargetMode mode = GetParam();
  const TargetProfile &target = *target_profile(mode.target);
  for (const char *name :
       {"native-25", "native-26", "native-27", "native-30", "native-31", "native-40"}) {
    SCOPED_TRACE(name);
    std::vector<uint8_t> original = test::read_fixture(test::target_fixture_path(mode, name));
    InputError error;
    const FailureOr<InputProgram> input =
        build_input_program(original, target, "lds_producer_consumer", error);
    ASSERT_TRUE(input.succeeded()) << error.message;
    ASSERT_TRUE(input.value().repeated);
    const AmdGpuCodeObject object(original.data(), original.size());
    const std::unique_ptr<Decoder> decoder =
        Decoder::create(default_isa_target_registry(), target.public_id);
    ASSERT_TRUE(decoder);
    const std::array entries{input.value().kernel.entry_text_offset};
    const FailureOr<std::vector<std::unique_ptr<BasicBlock>>> blocks =
        BasicBlock::build_reachable(object, *decoder, target.arch, entries);
    ASSERT_TRUE(blocks.succeeded());
    const Instruction *backedge = nullptr;
    for (const std::unique_ptr<BasicBlock> &block : blocks.value())
      for (const Instruction &instruction : block->instructions()) {
        const std::optional<int64_t> branch = instruction.branch_offset_bytes();
        if (branch && *branch <= -static_cast<int64_t>(instruction.size())) {
          ASSERT_EQ(backedge, nullptr);
          backedge = &instruction;
        }
      }
    ASSERT_NE(backedge, nullptr);
    const FailureOr<PreparedArtifact> artifact =
        prepare(original, target, "lds_producer_consumer", error);
    ASSERT_TRUE(artifact.succeeded()) << error.message;
    EXPECT_EQ(artifact.value().manifest.scope, ScopeKind::NativeLdsWorkgroupBarrierLoop);
    EXPECT_TRUE(validate_prepared(original, artifact.value(), error).succeeded()) << error.message;
    const AmdGpuCodeObject prepared(artifact.value().image.data(), artifact.value().image.size());
    const Section &prepared_text = *prepared.text_sections().front();
    EXPECT_EQ(
        wire::read_u32(artifact.value().image, prepared_text.sectionOffset() + backedge->src_loc()),
        backedge->raw_encoding()[0]);
    // Redirect the real conditional backedge to the preheader. The resulting
    // multi-block cycle is outside the currently admitted single-block shape.
    const int64_t delta = (static_cast<int64_t>(input.value().kernel.entry_text_offset) -
                           static_cast<int64_t>(backedge->src_loc()) - 4) /
                          4;
    wire::write_u32(original, input.value().text_file_offset + backedge->src_loc(),
                    (backedge->raw_encoding()[0] & 0xffff0000u) | static_cast<uint16_t>(delta));
    EXPECT_TRUE(prepare(original, target, "lds_producer_consumer", error).failed());
    EXPECT_EQ(error.code, InputErrorCode::UnsupportedFlow) << error.message;
  }
}

TEST_P(ConSanTargetStatic, EquivalentMetadataEnvironmentsPreserveTargetAndAbiValidation) {
  const test::TargetMode mode = GetParam();
  const TargetProfile &target = *target_profile(mode.target);
  std::vector<uint8_t> original_text;
  for (const std::string_view fixture : {"native-1", "native-unknown-environment"}) {
    SCOPED_TRACE(fixture);
    const std::vector<uint8_t> original =
        test::read_fixture(test::target_fixture_path(mode, fixture));
    InputError error;
    const FailureOr<InputProgram> input =
        build_input_program(original, target, "lds_producer_consumer", error);
    ASSERT_TRUE(input.succeeded()) << error.message;
    const std::span<const uint8_t> text =
        std::span(original).subspan(input.value().text_file_offset, input.value().text_bytes);
    if (original_text.empty())
      original_text.assign(text.begin(), text.end());
    else
      EXPECT_TRUE(std::ranges::equal(original_text, text));
    const FailureOr<PreparedArtifact> prepared =
        prepare(original, target, "lds_producer_consumer", error);
    ASSERT_TRUE(prepared.succeeded()) << error.message;
    ContractError image_error;
    ASSERT_TRUE(verify_artifact_images(prepared.value().manifest, original, prepared.value().image,
                                       &image_error)
                    .succeeded())
        << image_error.message;
    if (fixture == "native-1")
      continue;

    const std::string expected = std::string("amdgcn-amd-amdhsa-unknown-") + mode.target;
    std::vector<std::string> invalid;
    for (const std::pair<std::string_view, std::string_view> replacement :
         {std::pair{"amdgcn", "xmdgcn"},
          {"amd", "xyz"},
          {"amdhsa", "amdxsa"},
          {"unknown", "invalid"}}) {
      std::string changed = expected;
      changed.replace(changed.find(replacement.first), replacement.first.size(),
                      replacement.second);
      // Keep the MessagePack length and ELF note framing intact.
      changed.resize(expected.size(), ' ');
      invalid.push_back(std::move(changed));
    }
    invalid.push_back(std::string("amdgcn-amd-amdhsa-unknown-") + (target.id == 942   ? "gfx950"
                                                                   : target.id == 950 ? "gfx942"
                                                                   : target.id == 1100
                                                                       ? "gfx1201"
                                                                       : "gfx1100"));
    invalid.push_back(std::string("amdgcn-amd-amdhsa--") + mode.target + ":xnack+");
    invalid.push_back(std::string("amdgcn-amd-amdhsa--") + mode.target + ":xnack-");
    invalid.push_back(expected.substr(0, expected.size() - 1) + '\0');
    for (const std::string &changed : invalid) {
      SCOPED_TRACE(changed);
      ASSERT_EQ(changed.size(), expected.size());
      for (const bool prepared_image : {false, true}) {
        std::vector<uint8_t> image = prepared_image ? prepared.value().image : original;
        const std::vector<uint8_t>::iterator position =
            std::search(image.begin(), image.end(), expected.begin(), expected.end());
        ASSERT_NE(position, image.end());
        ASSERT_EQ(
            std::search(position + expected.size(), image.end(), expected.begin(), expected.end()),
            image.end());
        std::copy(changed.begin(), changed.end(), position);
        if (prepared_image) {
          Manifest manifest = prepared.value().manifest;
          manifest.prepared.sha256 = digest(image);
          ASSERT_TRUE(match_images(manifest, original, image).succeeded());
          EXPECT_TRUE(verify_artifact_images(manifest, original, image, &image_error).failed());
        } else {
          EXPECT_TRUE(build_input_program(image, target, "lds_producer_consumer", error).failed());
          EXPECT_NE(error.message.find("metadata"), std::string::npos);
        }
      }
    }
  }
}

TEST_P(ConSanTargetStatic, PreparedCdnaDescriptorMustAllocateItsDeclaredOrdinarySgprs) {
  const test::TargetMode mode = GetParam();
  const TargetProfile &target = *target_profile(mode.target);
  if (target.id != 942 && target.id != 950)
    return;
  const std::vector<uint8_t> original =
      test::read_fixture(test::target_fixture_path(mode, "native-1"));
  InputError error;
  FailureOr<PreparedArtifact> prepared = prepare(original, target, "lds_producer_consumer", error);
  ASSERT_TRUE(prepared.succeeded()) << error.message;
  PreparedArtifact &artifact = prepared.value();
  const uint64_t offset = artifact.manifest.entry.prepared_descriptor_offset + 48;
  const uint32_t resource = *wire::read_u32(artifact.image, offset);
  const uint32_t required = artifact.manifest.entry.prepared_sgprs;
  namespace kd = rocjitsu::amdhsa;
  // Test every hardware allocation granule, including rounded allocations that
  // exceed the exact metadata count. Rehash so identity checks cannot mask this.
  for (uint32_t granules = 1; granules <= 16; ++granules) {
    uint32_t changed = resource;
    RJ_AMDHSA_BITS_SET(changed, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                       granules - 1);
    ASSERT_TRUE(wire::write_u32(artifact.image, offset, changed));
    artifact.manifest.prepared.sha256 = digest(artifact.image);
    ContractError image_error;
    ASSERT_TRUE(match_images(artifact.manifest, original, artifact.image).succeeded());
    EXPECT_EQ(verify_artifact_images(artifact.manifest, original, artifact.image, &image_error)
                  .succeeded(),
              granules * 8 >= required + 6) // Six ABI special-register tail slots.
        << "granules=" << granules << ": " << image_error.message;
  }
}

TEST_P(ConSanTargetStatic, UnreachableEntryOrDenseSitesFailBeforeArtifactPublication) {
  const test::TargetMode mode = GetParam();
  const TargetProfile &target = *target_profile(mode.target);
  for (const auto fixture : {"native-distant-entry", "native-dense-sites"}) {
    SCOPED_TRACE(fixture);
    const std::vector<uint8_t> original =
        test::read_fixture(test::target_fixture_path(mode, fixture));
    ASSERT_FALSE(original.empty());
    InputError error;
    ASSERT_TRUE(build_input_program(original, target, "lds_producer_consumer", error).succeeded())
        << error.message;
    EXPECT_TRUE(prepare(original, target, "lds_producer_consumer", error).failed());
    EXPECT_EQ(error.code, InputErrorCode::ResourceLimit);
    EXPECT_NE(error.message.find("branch"), std::string::npos) << error.message;
    const std::string_view reason = std::string_view(fixture) == "native-distant-entry"
                                        ? "entry-prologue stub is out of branch range"
                                        : "return branch exceeds s_branch simm16";
    EXPECT_NE(error.message.find(reason), std::string::npos) << error.message;
  }
}

TEST_P(ConSanTargetStatic, AssemblyLoweringPreservesConcreteIdentityAndLoadableImages) {
  const test::TargetMode mode = GetParam();
  const TargetProfile *target = target_profile(mode.target);
  ASSERT_NE(target, nullptr);
  for (const uint32_t fixture : {1u, 2u, 3u, 4u, 6u, 11u, 12u, 13u, 14u}) {
    SCOPED_TRACE(mode.target + std::string(" native-") + std::to_string(fixture));
    const std::vector<uint8_t> original =
        test::read_fixture(test::target_fixture_path(mode, "native-" + std::to_string(fixture)));
    ASSERT_FALSE(original.empty());
    InputError error;
    const FailureOr<InputProgram> input =
        build_input_program(original, *target_profile(mode.target), "lds_producer_consumer", error);
    ASSERT_TRUE(input.succeeded()) << error.message;
    EXPECT_EQ(input.value().wave_size, mode.wave_size);
    ASSERT_FALSE(input.value().accesses.empty());
    const FailureOr<PreparedArtifact> prepared =
        prepare(original, *target_profile(mode.target), "lds_producer_consumer", error);
    ASSERT_TRUE(prepared.succeeded()) << error.message;
    EXPECT_EQ(prepared.value().manifest.target, target->id);
    EXPECT_EQ(prepared.value().manifest.wave_size, mode.wave_size);
    ContractError image_error;
    EXPECT_TRUE(verify_artifact_images(prepared.value().manifest, original, prepared.value().image,
                                       &image_error)
                    .succeeded())
        << image_error.message;
    const FailureOr<std::vector<uint8_t>> encoded = encode_manifest(prepared.value().manifest);
    ASSERT_TRUE(encoded.succeeded());
    const FailureOr<Manifest> decoded = decode_manifest(encoded.value());
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_EQ(decoded.value(), prepared.value().manifest);
  }
}

TEST_P(ConSanTargetStatic, GdsAndAccumulatorAliasingAreRejectedAtTheirActualBoundaries) {
  const test::TargetMode mode = GetParam();
  const TargetProfile *target = target_profile(mode.target);
  std::vector<uint8_t> original = test::read_fixture(test::target_fixture_path(mode, "native-1"));
  InputError error;
  const FailureOr<InputProgram> input =
      build_input_program(original, *target_profile(mode.target), "lds_producer_consumer", error);
  ASSERT_TRUE(input.succeeded()) << error.message;
  if (target->arch == ROCJITSU_CODE_ARCH_CDNA3 || target->arch == ROCJITSU_CODE_ARCH_CDNA4 ||
      target->arch == ROCJITSU_CODE_ARCH_RDNA3) {
    const uint64_t offset =
        input.value().text_file_offset + input.value().accesses[0].original_offset;
    std::vector<uint8_t> gds = original;
    const uint32_t word = *wire::read_u32(gds, offset);
    ASSERT_TRUE(wire::write_u32(
        gds, offset, word | (1u << (mode.wave_size == 64 && target->id < 1000 ? 16 : 17))));
    EXPECT_TRUE(
        build_input_program(gds, *target_profile(mode.target), "lds_producer_consumer", error)
            .failed());
    EXPECT_EQ(error.code, InputErrorCode::UnsupportedAccess) << error.message;
  }
  if (target->id != 942 && target->id != 950)
    return;
  FailureOr<PreparedArtifact> prepared =
      prepare(original, *target_profile(mode.target), "lds_producer_consumer", error);
  ASSERT_TRUE(prepared.succeeded()) << error.message;
  PreparedArtifact &artifact = prepared.value();
  AmdGpuCodeObject image(artifact.image.data(), artifact.image.size());
  const Section &text = *image.text_sections().front();
  const std::vector<KernelDescriptorInfo> kernels = scan_kernel_descriptors(
      artifact.image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  ASSERT_EQ(kernels.size(), 1u);
  rocjitsu::amdhsa::kernel_descriptor_t descriptor = kernels[0].descriptor;
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
  const test::TargetMode mode = GetParam();
  for (const uint32_t fixture : {1u, 2u, 3u}) {
    SCOPED_TRACE(mode.target + std::string(" HIP-") + std::to_string(fixture));
    const std::vector<uint8_t> original =
        test::read_fixture(test::target_fixture_path(mode, "hip-" + std::to_string(fixture)));
    ASSERT_FALSE(original.empty());
    InputError error;
    const FailureOr<InputProgram> input =
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
    for (const NativeLdsAccess &site : input.value().accesses)
      (site.kind == AccessKind::Read ? reads : writes)++;
    EXPECT_GT(reads, 0u);
    EXPECT_EQ(writes != 0, fixture != 3);
    const FailureOr<PreparedArtifact> prepared =
        prepare(original, *target_profile(mode.target), "lds_producer_consumer", error);
    ASSERT_TRUE(prepared.succeeded()) << error.message;
    EXPECT_TRUE(validate_prepared(original, prepared.value(), error).succeeded()) << error.message;
  }
}

TEST(ConSanBoundedStatic, BindOriginalWorkgroupInputAndRejectForgedIdentityFacts) {
  for (const test::TargetMode mode : test::kTargetModes)
    for (const char *fixture : {"native-2", "native-no-workgroup-id"}) {
      SCOPED_TRACE(mode.target);
      SCOPED_TRACE(mode.wave_size);
      SCOPED_TRACE(fixture);
      const TargetProfile &target = *target_profile(mode.target);
      const std::vector<uint8_t> original =
          test::read_fixture(test::target_fixture_path(mode, fixture));
      ASSERT_FALSE(original.empty());
      InputError error;
      const FailureOr<PreparedArtifact> artifact =
          prepare(original, target, "lds_producer_consumer", error);
      ASSERT_TRUE(artifact.succeeded()) << error.message;
      const bool identity = std::string_view(fixture) == "native-2";
      const bool ttmp =
          target.arch == ROCJITSU_CODE_ARCH_RDNA4 || target.arch == ROCJITSU_CODE_ARCH_CDNA5;
      const uint32_t selector = ttmp ? 117u : 15u;
      EXPECT_EQ(artifact.value().manifest.entry.workgroup_id_scalar,
                identity ? selector : UINT32_MAX);
      ContractError contract_error;
      const uint32_t threads = 2 * mode.wave_size + 1;
      EXPECT_EQ(plan_launch(artifact.value().manifest, {{4 * threads, 1, 1}, {threads, 1, 1}}, 8, 0)
                    .succeeded(),
                identity);
      EXPECT_TRUE(plan_launch(artifact.value().manifest, {{threads, 1, 1}, {threads, 1, 1}}, 8, 0)
                      .succeeded());
      Manifest forged = artifact.value().manifest;
      forged.entry.workgroup_id_scalar = identity ? UINT32_MAX : selector;
      ASSERT_TRUE(match_images(forged, original, artifact.value().image).succeeded());
      EXPECT_TRUE(verify_artifact_images(forged, original, artifact.value().image, &contract_error)
                      .failed());
      EXPECT_EQ(contract_error.code, ContractErrorCode::IdentityMismatch);
    }
}

TEST_P(ConSanTargetStatic, BoundedWorkgroupFixturesRetainQualifiedLdsSites) {
  const test::TargetMode mode = GetParam();
  const TargetProfile &target = *target_profile(mode.target);
  for (const char *fixture : {"group-1", "group-3", "group-4", "group-6"}) {
    SCOPED_TRACE(fixture);
    const std::vector<uint8_t> original =
        test::read_fixture(test::target_fixture_path(mode, fixture));
    ASSERT_FALSE(original.empty());
    InputError error;
    const FailureOr<PreparedArtifact> artifact =
        prepare(original, target, "lds_producer_consumer", error);
    ASSERT_TRUE(artifact.succeeded()) << error.message;
    ASSERT_EQ(artifact.value().manifest.sites.size(), 2u);
    EXPECT_TRUE(artifact.value().manifest.barriers.empty());
    EXPECT_EQ(artifact.value().manifest.sites[0].access,
              std::string_view(fixture) == "group-4" ? AccessKind::Read : AccessKind::Write);
    EXPECT_EQ(artifact.value().manifest.sites[1].access, AccessKind::Read);
    const uint32_t threads = 2 * mode.wave_size + 1;
    const FailureOr<LaunchLayout> layout =
        plan_launch(artifact.value().manifest, {{16 * threads, 1, 1}, {threads, 1, 1}}, 128, 0);
    ASSERT_TRUE(layout.succeeded());
    EXPECT_EQ(layout.value().waves_per_workgroup, 3u);
    EXPECT_EQ(layout.value().wave_count, 48u);
    EXPECT_TRUE(verify_artifact_images(artifact.value().manifest, original, artifact.value().image)
                    .succeeded());
  }
}

INSTANTIATE_TEST_SUITE_P(ConSanTargets, ConSanTargetStatic, ::testing::ValuesIn(test::kTargetModes),
                         [](const auto &info) {
                           return std::string(info.param.target) + "Wave" +
                                  std::to_string(info.param.wave_size);
                         });

} // namespace
} // namespace rocjitsu::consan
