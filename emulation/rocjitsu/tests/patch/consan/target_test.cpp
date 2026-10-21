// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "target_fixture.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/code/consan/image/artifact.h"
#include "rocjitsu/code/patch/consan/prepare.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <bitset>
#include <cstring>

namespace rocjitsu::consan {
namespace {

class ConSanTargetStatic : public ::testing::TestWithParam<test::TargetMode> {};

TEST_P(ConSanTargetStatic, EquivalentMetadataEnvironmentsPreserveTargetAndAbiValidation) {
  const test::TargetMode mode = GetParam();
  const TargetProfile &target = *target_profile(mode.target);
  std::vector<OriginalInstruction> instructions;
  for (const std::string_view fixture : {"native-1", "native-unknown-environment"}) {
    SCOPED_TRACE(fixture);
    const std::vector<uint8_t> original =
        test::read_fixture(test::target_fixture_path(mode, fixture));
    InputError error;
    const FailureOr<InputProgram> input =
        build_input_program(original, target, "lds_producer_consumer", error);
    ASSERT_TRUE(input.succeeded()) << error.message;
    if (instructions.empty()) {
      instructions = input.value().instructions;
    } else {
      ASSERT_EQ(instructions.size(), input.value().instructions.size());
      for (size_t index = 0; index < instructions.size(); ++index)
        EXPECT_EQ(instructions[index].words, input.value().instructions[index].words);
    }
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
        const auto position =
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

TEST_P(ConSanTargetStatic, ProbeScalarDependenciesHaveTargetRequiredGuards) {
  const test::TargetMode mode = GetParam();
  const TargetProfile &target = *target_profile(mode.target);
  const std::vector<uint8_t> original =
      test::read_fixture(test::target_fixture_path(mode, "native-1"));
  InputError error;
  const FailureOr<InputProgram> program =
      build_input_program(original, target, "lds_producer_consumer", error);
  ASSERT_TRUE(program.succeeded()) << error.message;
  const FailureOr<LdsRaceProbePlan> plan = plan_lds_race_probes(program.value(), error);
  ASSERT_TRUE(plan.succeeded()) << error.message;
  std::vector<uint8_t> text;
  ASSERT_TRUE(emit_lds_race_entry(text, plan.value(), error).succeeded());
  // Repeated probes exercise scratch reuse across separate emitter instances.
  for (int repetition = 0; repetition < 2; ++repetition)
    for (const NativeLdsAccess &access : program.value().accesses)
      ASSERT_TRUE(emit_lds_race_access(text, plan.value(), access, error).succeeded());
  ASSERT_TRUE(emit_lds_race_exit(text, plan.value(), error).succeeded());
  std::vector<uint32_t> words(text.size() / 4);
  std::memcpy(words.data(), text.data(), text.size());
  std::unique_ptr<Decoder> decoder = Decoder::create(target.arch);
  struct Dependencies {
    std::bitset<128> salu;
    std::bitset<128> valu;
    std::bitset<64> tracked;
    bool vcc = false;
    bool reachable = false;
  };
  std::vector<Dependencies> incoming(words.size() + 1);
  incoming[0].reachable = true;
  size_t reads = 0;
  size_t vector_consumers = 0;
  size_t waits = 0;
  for (size_t offset = 0; offset < words.size();) {
    DecodeResult decoded = decoder->decode_window(std::span(words).subspan(offset), offset * 4);
    ASSERT_TRUE(decoded.succeeded()) << "offset=" << offset * 4;
    const Instruction &instruction = *decoded.value();
    const std::string_view name = instruction.mnemonic();
    const size_t next = offset + instruction.size() / 4;
    if (target.id == 942 || target.id == 950) {
      if (name.starts_with("v_readfirstlane_b32")) {
        ++reads;
        const std::optional<RegisterRef> destination =
            instruction.dst_operand(0)->to_register_ref();
        ASSERT_TRUE(destination);
        ASSERT_EQ(destination->cls, RegClass::SGPR);
        uint32_t wait_states = 0;
        // Check the emitted dependency distance, allowing independent
        // instructions as well as nop padding to satisfy the hardware rule.
        for (size_t cursor = next; cursor < words.size();) {
          DecodeResult following = decoder->decode_window(std::span(words).subspan(cursor));
          ASSERT_TRUE(following.succeeded());
          const Instruction &consumer = *following.value();
          bool reads_destination = false;
          bool overwrites_destination = false;
          for (int index = 0; index < consumer.num_src_operands(); ++index) {
            const std::optional<RegisterRef> source =
                consumer.src_operand(index)->to_register_ref();
            reads_destination |= source && source->cls == RegClass::SGPR &&
                                 source->index <= destination->index &&
                                 destination->index < source->index + source->width;
          }
          for (int index = 0; index < consumer.num_dst_operands(); ++index) {
            const std::optional<RegisterRef> written =
                consumer.dst_operand(index)->to_register_ref();
            overwrites_destination |= written && written->cls == RegClass::SGPR &&
                                      written->index <= destination->index &&
                                      destination->index < written->index + written->width;
          }
          if (reads_destination && consumer.mnemonic().starts_with("v_")) {
            EXPECT_GE(wait_states, 2u) << "consumer at " << cursor * 4;
            ++vector_consumers;
            break;
          }
          if (overwrites_destination)
            break;
          wait_states += consumer.mnemonic() == "s_nop" ? (words[cursor] & 0xffffu) + 1 : 1;
          cursor += consumer.size() / 4;
        }
      }
    } else if (target.id == 1201 && incoming[offset].reachable) {
      Dependencies state = incoming[offset];
      if (name == "s_wait_alu") {
        const uint32_t control = words[offset] & 0xffff;
        if ((control & 1) == 0)
          state.salu.reset();
        if ((control & 0xe00) == 0)
          state.valu.reset();
        if ((control & 2) == 0)
          state.vcc = false;
        ++waits;
      }
      const bool vector_alu = name.starts_with("v_");
      for (int operand_index = 0; operand_index < instruction.num_src_operands(); ++operand_index) {
        const std::optional<RegisterRef> reg =
            instruction.src_operand(operand_index)->to_register_ref();
        if (!reg)
          continue;
        if (reg->cls == RegClass::VCC) {
          EXPECT_FALSE(state.vcc) << "VCC consumer at " << offset * 4 << ": " << name;
        }
        if (reg->cls != RegClass::SGPR)
          continue;
        for (uint16_t lane = reg->index; lane < reg->index + reg->width; ++lane) {
          EXPECT_FALSE(state.salu[lane])
              << "SALU consumer s" << lane << " at " << offset * 4 << ": " << name;
          if (vector_alu) {
            EXPECT_FALSE(state.valu[lane]) << "VALU consumer s" << lane << " at " << offset * 4;
            state.tracked.set(lane / 2);
          }
        }
      }
      for (int operand_index = 0; operand_index < instruction.num_dst_operands(); ++operand_index) {
        const std::optional<RegisterRef> reg =
            instruction.dst_operand(operand_index)->to_register_ref();
        if (!reg || reg->cls != RegClass::SGPR)
          continue;
        for (uint16_t lane = reg->index; lane < reg->index + reg->width; ++lane)
          if (state.tracked[lane / 2])
            (vector_alu ? state.valu : state.salu).set(lane);
      }
      if (name.starts_with("v_cmp_"))
        state.vcc = true;
      const auto merge = [&](size_t destination) {
        ASSERT_LE(destination, words.size());
        incoming[destination].salu |= state.salu;
        incoming[destination].valu |= state.valu;
        incoming[destination].tracked |= state.tracked;
        incoming[destination].vcc |= state.vcc;
        incoming[destination].reachable = true;
      };
      if (name.starts_with("s_cbranch_")) {
        const int16_t displacement = static_cast<int16_t>(words[offset] & 0xffff);
        ASSERT_GE(displacement, 0);
        merge(offset + 1 + displacement);
      }
      merge(next);
    }
    offset = next;
  }
  if (target.id == 942 || target.id == 950) {
    EXPECT_GT(reads, 1u);
    EXPECT_GT(vector_consumers, 0u);
  }
  if (target.id == 1201) {
    EXPECT_GT(waits, 0u);
  }
}

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
