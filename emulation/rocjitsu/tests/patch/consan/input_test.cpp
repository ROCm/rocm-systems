// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "target_fixture.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include "../../dbi_test_util.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <utility>

namespace rocjitsu::consan {
namespace {

constexpr const TargetProfile &kTarget = kNativeLdsTargets[2];

TEST(ConSanInput, Gfx1100UsesConcreteModelOnlyDecoder) {
  const IsaTargetRegistry &registry = default_isa_target_registry();
  ASSERT_TRUE(registry.ok()) << registry.error();
  const IsaTargetDescriptor *provider = registry.find(ROCJITSU_CODE_TARGET_GFX1100);
  ASSERT_NE(provider, nullptr);
  EXPECT_EQ(provider, registry.find("gfx1100"));
  EXPECT_EQ(provider->architecture_id, ROCJITSU_CODE_ARCH_RDNA3);
  EXPECT_FALSE(provider->supports_execution);

  const IsaGpuTargetDescription *binding = registry.find_gpu_target(ROCJITSU_CODE_TARGET_GFX1100);
  ASSERT_NE(binding, nullptr);
  EXPECT_EQ(binding, registry.find_gpu_target_by_code_object_id("gfx1100"));
  EXPECT_EQ(binding, registry.find_gpu_target_by_elf_machine(EF_AMDGPU_MACH_AMDGCN_GFX1100));
  EXPECT_EQ(binding->gfx_target_version, 110000u);
  EXPECT_FALSE(binding->capabilities.execution_implemented);

  std::unique_ptr<Decoder> decoder = Decoder::create(registry, ROCJITSU_CODE_TARGET_GFX1100);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint32_t, 1> kNop = {0xbf800000u};
  const DecodeResult decoded = decoder->decode_window(kNop);
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value()->mnemonic(), "s_nop");
  EXPECT_EQ(decoded.value()->execute, nullptr);
}

TEST(ConSanInput, Gfx1100RequiresConcreteIdentityBeforeImageValidation) {
  const TargetProfile *profile = target_profile("gfx1100");
  ASSERT_NE(profile, nullptr);
  EXPECT_EQ(profile->public_id, ROCJITSU_CODE_TARGET_GFX1100);

  // An empty image distinguishes target admission from later ELF validation.
  InputError error;
  EXPECT_TRUE(build_input_program({}, *profile, "kernel", error).failed());
  EXPECT_EQ(error.code, InputErrorCode::InvalidImage);

  TargetProfile selected = *profile;
  selected.public_id = ROCJITSU_CODE_TARGET_INVALID;
  EXPECT_TRUE(build_input_program({}, selected, "kernel", error).failed());
  EXPECT_EQ(error.code, InputErrorCode::UnsupportedTarget);
}

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

TEST(ConSanInput, MissingEntryAndAdditionalKernelHaveDistinctDiagnostics) {
  std::vector<uint8_t> image = rocjitsu::test::make_gfx950_two_kernel_elf({0xbf810000u}, 0);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const TargetProfile &target = *target_profile("gfx950");
  InputError error;
  EXPECT_TRUE(build_input_program(image, target, "missing_kernel", error).failed());
  EXPECT_EQ(error.code, InputErrorCode::UnsupportedEntry);
  EXPECT_EQ(error.message, "requested kernel entry 'missing_kernel' was not found");
  EXPECT_TRUE(build_input_program(image, target, "test_kernel0", error).failed());
  EXPECT_EQ(error.code, InputErrorCode::UnsupportedEntry);
  EXPECT_EQ(error.message, "ConSan requires a single-kernel code object; found 2 kernels");
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
    std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3);
    ASSERT_NE(decoder, nullptr);
    std::array<uint32_t, 2> words{};
    ASSERT_LE(program.text_file_offset + access.original_offset + sizeof(words),
              program.image.size());
    std::memcpy(words.data(),
                program.image.data() + program.text_file_offset + access.original_offset,
                sizeof(words));
    const DecodeResult decoded = decoder->decode_window(words);
    ASSERT_TRUE(decoded.succeeded());
    ASSERT_EQ(decoded.value()->size(), 8u);
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

  void make_instruction_detour(uint64_t anchor, bool returns_to_anchor,
                               std::vector<uint8_t> &image) {
    const AmdGpuCodeObject source(image_.data(), image_.size());
    const Section &text = *source.text_sections().front();
    const std::span<const uint32_t> source_words(reinterpret_cast<const uint32_t *>(text.data()),
                                                 text.size() / sizeof(uint32_t));
    std::unique_ptr<Decoder> decoder =
        Decoder::create(default_isa_target_registry(), kTarget.public_id);
    ASSERT_NE(decoder, nullptr);
    ASSERT_LT(anchor / 4, source_words.size());
    const DecodeResult first = decoder->decode_window(source_words.subspan(anchor / 4), anchor);
    ASSERT_TRUE(first.succeeded());
    ASSERT_FALSE(first.value()->branch_offset_bytes());
    ASSERT_EQ(first.value()->flags() & (INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR), 0u);
    const uint32_t first_size = first.value()->size();
    ASSERT_GE(first_size, 4u);
    ASSERT_LT(anchor + first_size, text.size());
    const uint64_t cave = text.size();
    const std::optional<int16_t> forward = compute_sopp_branch_simm16(anchor, cave);
    const uint64_t continuation = returns_to_anchor ? anchor : anchor + first_size;
    const std::optional<int16_t> backward =
        compute_sopp_branch_simm16(cave + first_size, continuation);
    ASSERT_TRUE(forward);
    ASSERT_TRUE(backward);
    std::vector<uint8_t> patched(text.data(), text.data() + text.size());
    patched.insert(patched.end(), text.data() + anchor, text.data() + anchor + first_size);
    const util::FailureOr<uint32_t> maybe_return_branch = build_s_branch(*backward, kTarget.arch);
    ASSERT_TRUE(maybe_return_branch.succeeded());
    const uint32_t return_branch = maybe_return_branch.value();
    const uint8_t *return_bytes = reinterpret_cast<const uint8_t *>(&return_branch);
    patched.insert(patched.end(), return_bytes, return_bytes + sizeof(return_branch));
    const util::FailureOr<uint32_t> maybe_anchor_branch = build_s_branch(*forward, kTarget.arch);
    ASSERT_TRUE(maybe_anchor_branch.succeeded());
    const uint32_t anchor_branch = maybe_anchor_branch.value();
    std::memcpy(patched.data() + anchor, &anchor_branch, sizeof(anchor_branch));
    const util::FailureOr<uint32_t> maybe_nop = build_s_nop(0, kTarget.arch);
    ASSERT_TRUE(maybe_nop.succeeded());
    const uint32_t nop = maybe_nop.value();
    for (uint32_t offset = sizeof(uint32_t); offset < first_size; offset += sizeof(uint32_t))
      std::memcpy(patched.data() + anchor + offset, &nop, sizeof(nop));
    CodeObjectPatcher patcher(source);
    ASSERT_TRUE(patcher.replace_text(patched));
    image = std::move(patcher).emit();
  }

  void make_entry_detour(bool returns_to_entry, std::vector<uint8_t> &image) {
    const FailureOr<InputProgram> pristine = inspect(image_);
    ASSERT_TRUE(pristine.succeeded()) << error_.message;
    make_instruction_detour(pristine.value().kernel.entry_text_offset, returns_to_entry, image);
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

TEST_F(ConSanInputFixture, ReplacementMutantReceivesFreshInputAnalysis) {
  const std::vector<uint8_t> pristine_image = image_;
  std::vector<uint8_t> mutant;
  Digest pristine_digest = {};
  NativeLdsAccess original_read = {};
  uint32_t pristine_sgprs = 0;
  uint32_t pristine_vgprs = 0;
  uint64_t pristine_entry = 0;
  {
    const FailureOr<InputProgram> original =
        build_input_program(pristine_image, kTarget, "lds_producer_consumer", error_);
    ASSERT_TRUE(original.succeeded()) << error_.message;
    ASSERT_EQ(original.value().accesses.size(), 2u);
    const NativeLdsAccess &write = original.value().accesses[0];
    original_read = original.value().accesses[1];
    ASSERT_EQ(write.kind, AccessKind::Write);
    ASSERT_EQ(original_read.kind, AccessKind::Read);
    pristine_digest = original.value().identity.sha256;
    pristine_sgprs = original.value().original_sgprs;
    pristine_vgprs = original.value().original_vgprs;
    pristine_entry = original.value().kernel.entry_text_offset;

    const AmdGpuCodeObject source(original.value().image.data(), original.value().image.size());
    ASSERT_TRUE(source.is_valid());
    const uint64_t write_file_offset = original.value().text_file_offset + write.original_offset;
    std::array<uint32_t, 2> words = {};
    ASSERT_LE(write_file_offset + sizeof(words), original.value().image.size());
    std::memcpy(words.data(), original.value().image.data() + write_file_offset, sizeof(words));
    std::unique_ptr<Decoder> decoder =
        Decoder::create(default_isa_target_registry(), kTarget.public_id);
    ASSERT_NE(decoder, nullptr);
    const DecodeResult decoded = decoder->decode_window(words, write.original_offset);
    ASSERT_TRUE(decoded.succeeded());
    ASSERT_EQ(decoded.value()->size(), 8u);

    const util::FailureOr<uint32_t> maybe_nop = build_s_nop(0, kTarget.arch);
    ASSERT_TRUE(maybe_nop.succeeded());
    const uint32_t nop = maybe_nop.value();
    const std::array<uint32_t, 2> nops = {nop, nop};
    InstructionReplacement replacement = {.expected_bytes = std::vector<uint8_t>(sizeof(words)),
                                          .replacement_bytes = std::vector<uint8_t>(sizeof(nops))};
    std::memcpy(replacement.expected_bytes.data(), words.data(), sizeof(words));
    std::memcpy(replacement.replacement_bytes.data(), nops.data(), sizeof(nops));
    Instrumentor instrumentor(source, kTarget.arch);
    instrumentor.add_point({.anchor_offset = write.original_offset,
                            .kind = InstrumentationKind::ReplaceWithInst,
                            .probe_symbol = {},
                            .probe_args = {},
                            .replacement = std::move(replacement)});
    InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    ASSERT_EQ(result.sites.size(), 1u);
    ASSERT_EQ(result.patches.size(), 1u);
    const InstrumentedSite &site = result.sites.front();
    EXPECT_EQ(site.kind, InstrumentationKind::ReplaceWithInst);
    EXPECT_EQ(site.original_offset, write.original_offset);
    EXPECT_EQ(site.original_size, 8u);
    EXPECT_EQ(site.guest_size, sizeof(nops));
    EXPECT_EQ(site.instrumentation_size, 0u);
    const AmdGpuCodeObject emitted(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(emitted.is_valid());
    EXPECT_EQ(emitted.target_id(), source.target_id());
    const Section &source_text = *source.text_sections().front();
    const Section &emitted_text = *emitted.text_sections().front();
    ASSERT_GT(emitted_text.size(), source_text.size());
    const std::span<const uint32_t> emitted_words(
        reinterpret_cast<const uint32_t *>(emitted_text.data()), emitted_text.size() / 4);
    const DecodeResult forward = decoder->decode_window(
        emitted_words.subspan(write.original_offset / 4), write.original_offset);
    ASSERT_TRUE(forward.succeeded());
    ASSERT_EQ(forward.value()->mnemonic(), "s_branch");
    ASSERT_TRUE(forward.value()->branch_offset_bytes());
    const int64_t destination = static_cast<int64_t>(write.original_offset) +
                                forward.value()->size() + *forward.value()->branch_offset_bytes();
    ASSERT_GE(destination, 0);
    const uint64_t replacement_offset = static_cast<uint64_t>(destination);
    ASSERT_GE(replacement_offset, source_text.size());
    ASSERT_LE(replacement_offset + sizeof(nops) + 4, emitted_text.size());
    EXPECT_EQ(site.guest_offset, replacement_offset);
    EXPECT_EQ(site.instrumentation_offset, replacement_offset);
    EXPECT_EQ(result.patches.front().trampoline_offset, replacement_offset);
    EXPECT_EQ(emitted_words[write.original_offset / 4 + 1], nop);
    EXPECT_EQ(std::memcmp(emitted_text.data() + replacement_offset, nops.data(), sizeof(nops)), 0);
    for (size_t index = 0; index < nops.size(); ++index) {
      const uint64_t offset = replacement_offset + index * sizeof(uint32_t);
      const DecodeResult replaced =
          decoder->decode_window(emitted_words.subspan(offset / 4), offset);
      ASSERT_TRUE(replaced.succeeded());
      EXPECT_EQ(replaced.value()->mnemonic(), "s_nop");
      EXPECT_EQ(replaced.value()->size(), 4u);
    }
    const uint64_t return_offset = replacement_offset + sizeof(nops);
    const DecodeResult returned =
        decoder->decode_window(emitted_words.subspan(return_offset / 4), return_offset);
    ASSERT_TRUE(returned.succeeded());
    ASSERT_EQ(returned.value()->mnemonic(), "s_branch");
    ASSERT_TRUE(returned.value()->branch_offset_bytes());
    EXPECT_EQ(static_cast<int64_t>(return_offset) + returned.value()->size() +
                  *returned.value()->branch_offset_bytes(),
              static_cast<int64_t>(write.original_offset + sizeof(words)));
    EXPECT_EQ(std::memcmp(source_text.data(), emitted_text.data(), write.original_offset), 0);
    const uint64_t unchanged = write.original_offset + sizeof(words);
    EXPECT_EQ(std::memcmp(source_text.data() + unchanged, emitted_text.data() + unchanged,
                          source_text.size() - unchanged),
              0);
    EXPECT_EQ(original.value().image, pristine_image);
    EXPECT_EQ(std::memcmp(source.image_data(), pristine_image.data(), pristine_image.size()), 0);
    mutant = std::move(result.elf_bytes);
  }

  // The original program, decoder, code object, and Instrumentor are gone. ConSan
  // must discover the remaining read from the mutant's bytes and new identity.
  const FailureOr<InputProgram> fresh =
      build_input_program(mutant, kTarget, "lds_producer_consumer", error_);
  ASSERT_TRUE(fresh.succeeded()) << error_.message;
  ASSERT_EQ(fresh.value().accesses.size(), 1u);
  const NativeLdsAccess &remaining = fresh.value().accesses.front();
  EXPECT_EQ(remaining.site, 0u);
  EXPECT_EQ(remaining.kind, AccessKind::Read);
  EXPECT_EQ(remaining.original_offset, original_read.original_offset);
  EXPECT_EQ(remaining.address_vgpr, original_read.address_vgpr);
  EXPECT_EQ(remaining.immediate_bytes, original_read.immediate_bytes);
  expect_value_operand(fresh.value(), remaining, 3);
  EXPECT_NE(fresh.value().identity.sha256, pristine_digest);
  EXPECT_EQ(fresh.value().identity.sha256, digest(mutant));
  EXPECT_GT(fresh.value().identity.bytes, pristine_image.size());
  EXPECT_EQ(fresh.value().image, mutant);
  EXPECT_EQ(fresh.value().original_sgprs, pristine_sgprs);
  EXPECT_EQ(fresh.value().original_vgprs, pristine_vgprs);
  EXPECT_EQ(fresh.value().kernel.entry_text_offset, pristine_entry);
  EXPECT_EQ(image_, pristine_image);
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
      EXPECT_LE(access.original_offset + 8, program.value().text_bytes);
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
  const std::optional<DbiEntryProloguePlan> planned = test::bootstrap_plan(program.value());
  ASSERT_TRUE(planned);
  const DbiEntryProloguePlan &plan = *planned;
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

TEST_F(ConSanInputFixture, AcyclicBackwardEdgeIsAccepted) {
  const std::vector<uint8_t> pristine = image_;
  const FailureOr<InputProgram> original = inspect(image_);
  ASSERT_TRUE(original.succeeded()) << error_.message;
  std::vector<uint8_t> detour;
  make_entry_detour(false, detour);
  ASSERT_FALSE(detour.empty());
  const FailureOr<InputProgram> program = inspect(detour);
  ASSERT_TRUE(program.succeeded()) << error_.message;
  ASSERT_EQ(program.value().accesses.size(), original.value().accesses.size());
  for (size_t index = 0; index < program.value().accesses.size(); ++index) {
    const NativeLdsAccess &actual = program.value().accesses[index];
    const NativeLdsAccess &expected = original.value().accesses[index];
    EXPECT_EQ(actual.original_offset, expected.original_offset);
    EXPECT_EQ(actual.kind, expected.kind);
    EXPECT_EQ(actual.address_vgpr, expected.address_vgpr);
    EXPECT_EQ(actual.immediate_bytes, expected.immediate_bytes);
  }
  EXPECT_EQ(program.value().identity.sha256, digest(detour));
  EXPECT_NE(program.value().identity.sha256, original.value().identity.sha256);
  EXPECT_EQ(image_, pristine);
}

TEST_F(ConSanInputFixture, AcyclicBackwardLdsAccessOrderFailsClosed) {
  const std::vector<uint8_t> pristine = image_;
  const FailureOr<InputProgram> original = inspect(image_);
  ASSERT_TRUE(original.succeeded()) << error_.message;
  ASSERT_EQ(original.value().accesses.size(), 2u);
  const NativeLdsAccess &write = original.value().accesses[0];
  const NativeLdsAccess &read = original.value().accesses[1];
  ASSERT_EQ(write.kind, AccessKind::Write);
  ASSERT_EQ(read.kind, AccessKind::Read);
  ASSERT_LT(write.original_offset, read.original_offset);
  std::vector<uint8_t> detour;
  // Move the write to the cave, then return after its original source extent.
  // This graph is acyclic, but executes the high-PC write before a low-PC read.
  ASSERT_NO_FATAL_FAILURE(make_instruction_detour(write.original_offset, false, detour));
  ASSERT_FALSE(detour.empty());
  const AmdGpuCodeObject output(detour.data(), detour.size());
  ASSERT_TRUE(output.is_valid());
  const Section &text = *output.text_sections().front();
  const uint64_t cave = original.value().text_bytes;
  ASSERT_LE(cave + 12, text.size());
  EXPECT_EQ(std::memcmp(text.data() + cave,
                        pristine.data() + original.value().text_file_offset + write.original_offset,
                        8),
            0);
  EXPECT_TRUE(inspect(detour).failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedFlow);
  EXPECT_NE(error_.message.find("LDS access order"), std::string::npos);
  EXPECT_EQ(error_.text_offset, read.original_offset);
  EXPECT_EQ(image_, pristine);
}

TEST_F(ConSanInputFixture, LdsAccessOrderAtAJoinIncludesEveryPredecessor) {
  const std::vector<uint8_t> pristine = image_;
  const FailureOr<InputProgram> original = inspect(image_);
  ASSERT_TRUE(original.succeeded()) << error_.message;
  ASSERT_EQ(original.value().accesses.size(), 2u);
  ASSERT_EQ(original.value().accesses[0].kind, AccessKind::Write);
  ASSERT_EQ(original.value().accesses[1].kind, AccessKind::Read);
  ASSERT_EQ(kTarget.arch, ROCJITSU_CODE_ARCH_RDNA3);
  const AmdGpuCodeObject source(image_.data(), image_.size());
  const Section &text = *source.text_sections().front();
  const uint64_t entry = original.value().kernel.entry_text_offset;
  ASSERT_EQ(entry % 4, 0u);
  const util::FailureOr<uint32_t> conditional =
      build_sopp_encoding(kTarget.arch, rdna3::kSCbranchScc1Sopp, 9);
  ASSERT_TRUE(conditional.succeeded());
  const util::FailureOr<uint32_t> nop = build_s_nop(0, kTarget.arch);
  const util::FailureOr<uint32_t> low_branch = build_s_branch(0, kTarget.arch);
  const util::FailureOr<uint32_t> join_branch = build_s_branch(1, kTarget.arch);
  const util::FailureOr<uint32_t> backward_branch = build_s_branch(-7, kTarget.arch);
  const util::FailureOr<uint32_t> terminator = build_s_endpgm(kTarget.arch);
  ASSERT_TRUE(nop.succeeded());
  ASSERT_TRUE(low_branch.succeeded());
  ASSERT_TRUE(join_branch.succeeded());
  ASSERT_TRUE(backward_branch.succeeded());
  ASSERT_TRUE(terminator.succeeded());
  for (const bool high_access : {false, true}) {
    SCOPED_TRACE(high_access);
    // Relative layout: branch at 0 chooses low write at 8 or high arm at 40.
    // Both arms join at read 24. The high arm's backward edge is harmless only
    // when that arm has no LDS access; the low arm's write 8 cannot erase it.
    std::vector<uint32_t> words(std::max(text.size() / 4, entry / 4 + 13), nop.value());
    const size_t first = entry / 4;
    words[first] = conditional.value();
    words[first + 1] = low_branch.value();
    const uint64_t write = original.value().accesses[0].original_offset;
    const uint64_t read = original.value().accesses[1].original_offset;
    ASSERT_LE(write + 8, text.size());
    ASSERT_LE(read + 8, text.size());
    std::memcpy(words.data() + first + 2, text.data() + write, 8);
    words[first + 4] = join_branch.value();
    std::memcpy(words.data() + first + 6, text.data() + read, 8);
    words[first + 8] = terminator.value();
    if (high_access)
      std::memcpy(words.data() + first + 10, text.data() + write, 8);
    words[first + 12] = backward_branch.value();
    CodeObjectPatcher patcher(source);
    ASSERT_TRUE(patcher.replace_text(
        {reinterpret_cast<const uint8_t *>(words.data()), words.size() * sizeof(uint32_t)}));
    const std::vector<uint8_t> joined = std::move(patcher).emit();
    const FailureOr<InputProgram> program = inspect(joined);
    if (high_access) {
      EXPECT_TRUE(program.failed());
      EXPECT_EQ(error_.code, InputErrorCode::UnsupportedFlow);
      EXPECT_NE(error_.message.find("LDS access order"), std::string::npos);
      EXPECT_EQ(error_.text_offset, entry + 24);
    } else {
      ASSERT_TRUE(program.succeeded()) << error_.message;
      ASSERT_EQ(program.value().accesses.size(), 2u);
      EXPECT_EQ(program.value().accesses[0].original_offset, entry + 8);
      EXPECT_EQ(program.value().accesses[1].original_offset, entry + 24);
    }
  }
  EXPECT_EQ(image_, pristine);
}

TEST_F(ConSanInputFixture, MultiBlockCycleFailsClosed) {
  const std::vector<uint8_t> pristine = image_;
  std::vector<uint8_t> cycle;
  make_entry_detour(true, cycle);
  ASSERT_FALSE(cycle.empty());
  EXPECT_TRUE(inspect(cycle).failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedFlow);
  EXPECT_EQ(image_, pristine);
}

TEST_F(ConSanInputFixture, BackedgeFailsClosed) {
  // Replace the original entry with an unconditional self-loop. No GPU executes
  // this input; it tests the production preparation admission gate.
  constexpr util::FailureOr<uint32_t> branch = build_s_branch(-1, ROCJITSU_CODE_ARCH_RDNA3);
  static_assert(branch.succeeded());
  put_word(text_offset_, branch.value());
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
