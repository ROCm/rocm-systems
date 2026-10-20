// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/input.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>

namespace rocjitsu::consan {
namespace {

constexpr const TargetProfile &kTarget = kNativeLdsTargets[2];

TEST(ConSanInput, TruncatedImageAndWrongSelectedTargetHaveDistinctErrors) {
  InputError error;
  EXPECT_TRUE(build_input_program({}, kTarget, "kernel", error).failed());
  EXPECT_EQ(error.code, InputErrorCode::InvalidImage);
  EXPECT_TRUE(build_input_program({},
                                  TargetProfile{.id = 1150,
                                                .name = "gfx1150",
                                                .arch = ROCJITSU_CODE_ARCH_INVALID,
                                                .public_id = ROCJITSU_CODE_TARGET_INVALID,
                                                .elf_machine = 0,
                                                .default_wave_size = 0,
                                                .wave64 = false},
                                  "kernel", error)
                  .failed());
  EXPECT_EQ(error.code, InputErrorCode::UnsupportedTarget);
}

class ConSanInputFixture : public ::testing::Test {
protected:
  void SetUp() override {
#ifndef CONSAN_FIXTURE_DIR
    GTEST_SKIP() << "gfx1100 assembler/linker fixtures unavailable";
#else
    image_ = load("race");
    ASSERT_FALSE(image_.empty());
    const FailureOr<InputProgram> baseline = inspect(image_);
    ASSERT_TRUE(baseline.succeeded()) << error_.message;
    descriptor_offset_ = baseline.value().kernel.descriptor_file_offset;
    text_offset_ = baseline.value().text_file_offset;
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

  FailureOr<InputProgram> inspect(std::span<const uint8_t> bytes,
                                  std::string_view entry = "lds_producer_consumer") {
    return build_input_program(bytes, kTarget, entry, error_);
  }

  void expect_value_operand(const InputProgram &program, const NativeLdsAccess &access,
                            uint16_t expected) {
    const auto original = std::ranges::find(program.instructions, access.original_offset,
                                            &OriginalInstruction::offset);
    ASSERT_NE(original, program.instructions.end());
    std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3);
    ASSERT_NE(decoder, nullptr);
    const DecodeResult decoded = decoder->decode_window(original->words);
    ASSERT_TRUE(decoded.succeeded());
    const Operand *operand = access.kind == AccessKind::Read ? decoded.value()->dst_operand(0)
                                                             : decoded.value()->src_operand(1);
    const std::optional<RegisterRef> reg = operand->decoded_vgpr();
    ASSERT_TRUE(reg.has_value());
    EXPECT_EQ(reg->index, expected);
  }

  template <typename Mutate> void mutate_descriptor(Mutate mutate) {
    rocjitsu::amdhsa::kernel_descriptor_t descriptor;
    ASSERT_LE(descriptor_offset_ + sizeof(descriptor), image_.size());
    std::memcpy(&descriptor, image_.data() + descriptor_offset_, sizeof(descriptor));
    mutate(descriptor);
    std::memcpy(image_.data() + descriptor_offset_, &descriptor, sizeof(descriptor));
  }

  void put_word(size_t file_offset, uint32_t word) {
    ASSERT_LE(file_offset + sizeof(word), image_.size());
    std::memcpy(image_.data() + file_offset, &word, sizeof(word));
  }

  InputError error_;
  std::vector<uint8_t> image_;
  size_t descriptor_offset_ = 0;
  size_t text_offset_ = 0;
};

TEST_F(ConSanInputFixture, DecoderFindsTheTwoPhysicalSitesAndRuntimeOperands) {
  const FailureOr<InputProgram> program = inspect(image_);
  ASSERT_TRUE(program.succeeded()) << error_.message;
  ASSERT_EQ(program.value().accesses.size(), 2u);
  const NativeLdsAccess &write = program.value().accesses[0];
  const NativeLdsAccess &read = program.value().accesses[1];
  EXPECT_EQ(write.site, 0u);
  EXPECT_EQ(read.site, 1u);
  EXPECT_LT(write.original_offset, read.original_offset);
  EXPECT_EQ(write.kind, AccessKind::Write);
  EXPECT_EQ(read.kind, AccessKind::Read);
  EXPECT_EQ(write.address_vgpr, 1u);
  expect_value_operand(program.value(), write, 2);
  EXPECT_EQ(read.address_vgpr, 1u);
  expect_value_operand(program.value(), read, 3);
  EXPECT_EQ(write.immediate_bytes, 0u);
  EXPECT_EQ(program.value().identity.sha256, digest(image_));
  EXPECT_EQ(program.value().original_vgprs, 8u);
  EXPECT_EQ(program.value().image, image_);
}

TEST_F(ConSanInputFixture, TemporaryProfileAndNameDoNotEscapeInputConstruction) {
  const FailureOr<InputProgram> program = [&] {
    std::string name = "gfx1100";
    TargetProfile local = kTarget;
    local.name = name;
    return build_input_program(image_, TargetProfile(local), "lds_producer_consumer", error_);
  }();
  ASSERT_TRUE(program.succeeded()) << error_.message;
  ASSERT_EQ(program.value().target, &kTarget);
  EXPECT_EQ(program.value().target->name, "gfx1100");
  EXPECT_EQ(program.value().wave_size, 32u);
}

TEST_F(ConSanInputFixture, InconsistentTypedProfilesCannotBypassTargetAdmission) {
  std::array<TargetProfile, 7> invalid;
  invalid.fill(kTarget);
  invalid[0].id = 1150;
  invalid[1].name = "gfx1201";
  invalid[2].arch = ROCJITSU_CODE_ARCH_INVALID;
  invalid[3].public_id = ROCJITSU_CODE_TARGET_GFX1201;
  invalid[4].elf_machine ^= 1;
  invalid[5].default_wave_size = 0;
  invalid[6].wave64 = false;
  for (const TargetProfile &profile : invalid) {
    EXPECT_TRUE(build_input_program(image_, profile, "lds_producer_consumer", error_).failed());
    EXPECT_EQ(error_.code, InputErrorCode::UnsupportedTarget);
  }
}

TEST_F(ConSanInputFixture, CompanionSitesDoNotDependOnFixtureNamesOrInstructionOffsets) {
  for (const std::string_view name : {"disjoint", "read_read", "alias", "immediate"}) {
    const FailureOr<InputProgram> program = inspect(load(name));
    ASSERT_TRUE(program.succeeded()) << name << ": " << error_.message;
    ASSERT_EQ(program.value().accesses.size(), 2u) << name;
    for (const NativeLdsAccess &access : program.value().accesses) {
      const auto found =
          std::find_if(program.value().instructions.begin(), program.value().instructions.end(),
                       [&](const OriginalInstruction &instruction) {
                         return instruction.offset == access.original_offset;
                       });
      ASSERT_NE(found, program.value().instructions.end());
      EXPECT_EQ(found->words.size(), 2u);
    }
    if (name == "read_read") {
      EXPECT_EQ(program.value().accesses[0].kind, AccessKind::Read);
    }
    if (name == "alias") {
      expect_value_operand(program.value(), program.value().accesses[1],
                           program.value().accesses[1].address_vgpr);
    }
    if (name == "immediate") {
      EXPECT_EQ(program.value().accesses[0].immediate_bytes, 4u);
    }
  }
  const FailureOr<InputProgram> race = inspect(load("race"));
  const FailureOr<InputProgram> disjoint = inspect(load("disjoint"));
  ASSERT_TRUE(race.succeeded());
  ASSERT_TRUE(disjoint.succeeded());
  EXPECT_NE(race.value().accesses[0].original_offset, disjoint.value().accesses[0].original_offset);
}

TEST_F(ConSanInputFixture, CollisionAndSingleLaneShareOnePhysicalStoreSite) {
  for (const std::string_view name : {"collision", "single_lane"}) {
    const FailureOr<InputProgram> program = inspect(load(name));
    ASSERT_TRUE(program.succeeded()) << error_.message;
    ASSERT_EQ(program.value().accesses.size(), 1u);
    EXPECT_EQ(program.value().accesses[0].kind, AccessKind::Write);
  }
}

TEST_F(ConSanInputFixture, BarrierAndWideAccessCannotBecomeNoSyncFacts) {
  EXPECT_TRUE(inspect(load("barrier")).failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedSynchronization);
  EXPECT_TRUE(error_.text_offset.has_value());
  EXPECT_TRUE(inspect(load("wide")).failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedAccess);
  EXPECT_TRUE(error_.text_offset.has_value());
}

TEST_F(ConSanInputFixture, GenericContextBootstrapUsesOwnedRegistersAndWrapperOffsets) {
  const FailureOr<InputProgram> program = inspect(image_);
  ASSERT_TRUE(program.succeeded()) << error_.message;
  const DbiEntryProloguePlan &plan = program.value().entry_plan;
  EXPECT_GE(plan.storage.persistent_base, 16u);
  EXPECT_EQ(plan.storage.entry_temp_base, plan.storage.persistent_base + 2);
  EXPECT_EQ(plan.prologue.payload_byte_offset, 24u);
  EXPECT_EQ(plan.prologue.original_kernarg_pointer_offset, 16u);
  std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_NE(decoder, nullptr);
  std::vector<std::string_view> names;
  for (size_t offset = 0; offset < plan.prologue.words.size();) {
    DecodeResult decoded = decoder->decode_window(std::span(plan.prologue.words).subspan(offset));
    ASSERT_TRUE(decoded.succeeded());
    names.push_back(decoded.value()->mnemonic());
    offset += decoded.value()->size() / 4;
  }
  EXPECT_EQ(names, (std::vector<std::string_view>{"s_load_b64", "s_load_b64", "s_waitcnt",
                                                  "s_mov_b32", "s_mov_b32"}));
}

TEST_F(ConSanInputFixture, WrongEntryTargetAbiAndEndianAreRejected) {
  EXPECT_TRUE(inspect(image_, "not_the_selected_entry").failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedEntry);
  image_[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V6;
  EXPECT_TRUE(inspect(image_).failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedTarget);
  image_[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  image_[EI_DATA] = 2;
  EXPECT_TRUE(inspect(image_).failed());
  EXPECT_EQ(error_.code, InputErrorCode::InvalidImage);
}

TEST_F(ConSanInputFixture, LdsValueOperandsRemainBoundedWithoutRetainingTheirIndices) {
  const FailureOr<InputProgram> program = inspect(image_);
  ASSERT_TRUE(program.succeeded()) << error_.message;
  const uint8_t outside = static_cast<uint8_t>(program.value().original_vgprs);
  for (const NativeLdsAccess &access : program.value().accesses) {
    const std::vector<uint8_t> saved = image_;
    const std::array<uint32_t, 2> words =
        access.kind == AccessKind::Read
            ? rdna3::build_ds(rdna3::kDsLoadB32Ds,
                              {.addr = static_cast<uint8_t>(access.address_vgpr), .vdst = outside})
            : rdna3::build_ds(
                  rdna3::kDsStoreB32Ds,
                  {.addr = static_cast<uint8_t>(access.address_vgpr), .data0 = outside});
    std::memcpy(image_.data() + text_offset_ + access.original_offset, words.data(), sizeof(words));
    EXPECT_TRUE(inspect(image_).failed());
    EXPECT_EQ(error_.code, InputErrorCode::UnsupportedAccess);
    EXPECT_EQ(error_.message, "LDS operands exceed allocation");
    EXPECT_EQ(error_.text_offset, access.original_offset);
    image_ = saved;
  }
}

TEST_F(ConSanInputFixture, PrivateFrameAndMissingKernargCannotUseBootstrap) {
  const std::vector<uint8_t> original = image_;
  mutate_descriptor([](auto &descriptor) { descriptor.private_segment_fixed_size = 4; });
  EXPECT_TRUE(inspect(image_).failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedEntry);
  image_ = original;
  mutate_descriptor([](auto &descriptor) { descriptor.kernel_code_properties = 0; });
  EXPECT_TRUE(inspect(image_).failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedEntry);
}

TEST_F(ConSanInputFixture, BackedgeFailsClosed) {
  // Replace the original entry with an unconditional self-loop. No GPU executes
  // this input; it tests the production preparation admission gate.
  put_word(text_offset_, build_s_branch(-1, ROCJITSU_CODE_ARCH_RDNA3));
  EXPECT_TRUE(inspect(image_).failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedFlow);
}

TEST_F(ConSanInputFixture, TextImageAndInstructionBudgetsAreEnforced) {
  InputLimits limits;
  limits.image_bytes = image_.size() - 1;
  EXPECT_TRUE(
      build_input_program(image_, kTarget, "lds_producer_consumer", error_, limits).failed());
  EXPECT_EQ(error_.code, InputErrorCode::ResourceLimit);
  limits = {};
  limits.text_bytes = 1;
  EXPECT_TRUE(
      build_input_program(image_, kTarget, "lds_producer_consumer", error_, limits).failed());
  EXPECT_EQ(error_.code, InputErrorCode::ResourceLimit);
  limits = {};
  limits.instructions = 1;
  EXPECT_TRUE(
      build_input_program(image_, kTarget, "lds_producer_consumer", error_, limits).failed());
  EXPECT_EQ(error_.code, InputErrorCode::ResourceLimit);
}

} // namespace
} // namespace rocjitsu::consan
