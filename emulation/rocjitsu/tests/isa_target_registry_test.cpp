// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna1/target_provider.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/target_provider.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/target_provider.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace rocjitsu {
namespace {

class FixtureDecoder final : public Decoder {
public:
  Instruction *decode(const rj_code_binary_inst_t *) override { return nullptr; }
};

std::unique_ptr<Decoder> create_fixture_decoder() { return std::make_unique<FixtureDecoder>(); }

constexpr IsaTargetDescriptor
fixture_target(std::string_view id, std::span<const std::string_view> aliases = {},
               rj_code_arch_t architecture_id = ROCJITSU_CODE_ARCH_INVALID,
               std::span<const IsaGpuTargetDescription> gpu_targets = {}) {
  return {
      .id = id,
      .aliases = aliases,
      .architecture_id = architecture_id,
      .gpu_targets = gpu_targets,
      .decoder_factory = &create_fixture_decoder,
  };
}

constexpr IsaGpuTargetDescription fixture_gpu_target(rj_code_target_id_t public_id,
                                                     std::string_view code_object_id,
                                                     uint32_t elf_machine) {
  return {
      .public_id = public_id,
      .code_object_id = code_object_id,
      .elf_machine = elf_machine,
  };
}

template <size_t Size>
void expect_registry_error(const std::array<IsaTargetDescriptor, Size> &targets,
                           std::string_view expected) {
  const IsaTargetRegistry registry{targets};
  ASSERT_FALSE(registry.ok());
  EXPECT_NE(registry.error().find(expected), std::string_view::npos)
      << "registry error was: " << registry.error();
  EXPECT_TRUE(registry.targets().empty());
}

void first_execute(Instruction &, void *) {}
void second_execute(Instruction &, void *) {}

static_assert(!std::is_default_constructible_v<IsaTargetRegistry>);
static_assert(!std::is_move_constructible_v<IsaTargetRegistry>);
static_assert(!std::is_copy_constructible_v<IsaTargetRegistry>);
static_assert(std::is_trivially_copyable_v<IsaTargetDescriptor>);
using FixtureTargetArray = std::array<IsaTargetDescriptor, 1>;
static_assert(std::is_constructible_v<IsaTargetRegistry, const FixtureTargetArray &>);
static_assert(!std::is_constructible_v<IsaTargetRegistry, FixtureTargetArray &>);
static_assert(!std::is_constructible_v<IsaTargetRegistry, FixtureTargetArray &&>);
static_assert(!std::is_constructible_v<IsaTargetRegistry, const FixtureTargetArray &&>);
using FixtureTargetCArray = IsaTargetDescriptor[1];
static_assert(std::is_constructible_v<IsaTargetRegistry, const FixtureTargetCArray &>);
static_assert(!std::is_constructible_v<IsaTargetRegistry, FixtureTargetCArray &>);
static_assert(!std::is_constructible_v<IsaTargetRegistry, FixtureTargetCArray &&>);
static_assert(!std::is_constructible_v<IsaTargetRegistry, const FixtureTargetCArray &&>);
static_assert(cdna1::kTargetDescriptor.aliases.empty());
static_assert(cdna1::kTargetDescriptor.decoder_factory == &cdna1::create_target_decoder);
static_assert(cdna2::kTargetDescriptor.id == "cdna2");
static_assert(cdna2::kTargetDescriptor.aliases.size() == 1);
static_assert(cdna2::kTargetDescriptor.aliases.front() == "gfx90a");
static_assert(cdna2::kTargetDescriptor.architecture_id == ROCJITSU_CODE_ARCH_CDNA2);
static_assert(cdna2::kTargetDescriptor.gpu_targets.front().public_id ==
              ROCJITSU_CODE_TARGET_GFX90A);
static_assert(rdna4::kTargetDescriptor.aliases.size() == 2);
static_assert(rdna4::kTargetDescriptor.aliases[0] == "gfx1200");
static_assert(rdna4::kTargetDescriptor.aliases[1] == "gfx1201");
static_assert(static_cast<int>(ROCJITSU_CODE_TARGET_GFX1250) == 5);
static_assert(static_cast<int>(ROCJITSU_CODE_TARGET_INVALID) == 6);

TEST(IsaTargetRegistryTest, PreservesStaticDescriptorOrder) {
  static constexpr std::array targets = {
      fixture_target("zeta"),
      fixture_target("alpha"),
  };
  const IsaTargetRegistry registry{targets};

  ASSERT_TRUE(registry.ok()) << registry.error();
  ASSERT_EQ(registry.targets().size(), 2u);
  EXPECT_EQ(registry.targets()[0].id, "zeta");
  EXPECT_EQ(registry.targets()[1].id, "alpha");
  EXPECT_EQ(registry.find("missing"), nullptr);
}

TEST(IsaTargetRegistryTest, RejectsConflictingCanonicalIdentities) {
  static constexpr std::array<std::string_view, 1> self_alias{"self"};
  static constexpr std::array self_alias_targets = {
      fixture_target("self", self_alias),
  };
  expect_registry_error(self_alias_targets, "duplicate ISA target ID");

  static constexpr std::array<std::string_view, 1> shared_alias{"shared"};
  static constexpr std::array duplicate_id_targets = {
      fixture_target("first", shared_alias),
      fixture_target("shared"),
  };
  expect_registry_error(duplicate_id_targets, "duplicate ISA target ID");

  static constexpr std::array<std::string_view, 2> duplicate_aliases{"twice", "twice"};
  static constexpr std::array duplicate_alias_targets = {
      fixture_target("second", duplicate_aliases),
  };
  expect_registry_error(duplicate_alias_targets, "duplicate ISA target ID");
}

TEST(IsaTargetRegistryTest, RejectsConflictingPublicEnumKeys) {
  static constexpr std::array first_gpu_targets = {
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX90A, "first", EF_AMDGPU_MACH_AMDGCN_GFX90A),
  };
  static constexpr IsaTargetDescriptor first =
      fixture_target("first", {}, ROCJITSU_CODE_ARCH_CDNA1, first_gpu_targets);

  static constexpr std::array duplicate_architecture_targets = {
      first,
      fixture_target("duplicate-architecture", {}, ROCJITSU_CODE_ARCH_CDNA1),
  };
  expect_registry_error(duplicate_architecture_targets, "duplicate ISA target architecture");

  static constexpr std::array duplicate_gpu = {
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX90A, "duplicate-gpu-target",
                         EF_AMDGPU_MACH_AMDGCN_GFX942),
  };
  static constexpr std::array duplicate_gpu_targets = {
      first,
      fixture_target("duplicate-gpu-target", {}, ROCJITSU_CODE_ARCH_CDNA2, duplicate_gpu),
  };
  expect_registry_error(duplicate_gpu_targets, "duplicate ISA target GPU target");

  static constexpr std::array duplicate_elf = {
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX942, "duplicate-elf-machine",
                         EF_AMDGPU_MACH_AMDGCN_GFX90A),
  };
  static constexpr std::array duplicate_elf_targets = {
      first,
      fixture_target("duplicate-elf-machine", {}, ROCJITSU_CODE_ARCH_CDNA2, duplicate_elf),
  };
  expect_registry_error(duplicate_elf_targets, "duplicate ISA target GPU ELF machine");
}

TEST(IsaTargetRegistryTest, RejectsInvalidTargetDescriptors) {
  static constexpr std::array empty_id = {fixture_target("")};
  expect_registry_error(empty_id, "canonical ID must not be empty");

  static constexpr std::array missing_factory = {
      IsaTargetDescriptor{.id = "missing-factory"},
  };
  expect_registry_error(missing_factory, "no decoder factory");

  static constexpr std::array<std::string_view, 1> empty_alias_value{""};
  static constexpr std::array empty_alias = {
      fixture_target("empty-alias", empty_alias_value),
  };
  expect_registry_error(empty_alias, "empty alias");

  static constexpr auto invalid_architecture =
      static_cast<rj_code_arch_t>(static_cast<int>(ROCJITSU_CODE_ARCH_NUM_ARCHS) + 1);
  static constexpr std::array invalid_enums = {
      fixture_target("invalid-enums", {}, invalid_architecture),
  };
  expect_registry_error(invalid_enums, "unallocated architecture");

  static constexpr std::array invalid_gpu_value = {
      fixture_gpu_target(ROCJITSU_CODE_TARGET_INVALID, "invalid-gpu-target",
                         EF_AMDGPU_MACH_AMDGCN_GFX90A),
  };
  static constexpr std::array invalid_gpu_target = {
      fixture_target("invalid-gpu-target", {}, ROCJITSU_CODE_ARCH_CDNA1, invalid_gpu_value),
  };
  expect_registry_error(invalid_gpu_target, "unallocated GPU target");

  static constexpr std::array<std::string_view, 1> duplicate_enum_alias{"duplicate-enums-alias"};
  static constexpr std::array duplicate_enum_values = {
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX90A, "duplicate-enums",
                         EF_AMDGPU_MACH_AMDGCN_GFX90A),
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX90A, "duplicate-enums-alias",
                         EF_AMDGPU_MACH_AMDGCN_GFX942),
  };
  static constexpr std::array duplicate_enums = {
      fixture_target("duplicate-enums", duplicate_enum_alias, ROCJITSU_CODE_ARCH_CDNA1,
                     duplicate_enum_values),
  };
  expect_registry_error(duplicate_enums, "duplicate ISA target GPU target");

  static constexpr std::array duplicate_code_object_values = {
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX90A, "duplicate-code-object-id",
                         EF_AMDGPU_MACH_AMDGCN_GFX90A),
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX942, "duplicate-code-object-id",
                         EF_AMDGPU_MACH_AMDGCN_GFX942),
  };
  static constexpr std::array duplicate_code_object_id = {
      fixture_target("duplicate-code-object-id", {}, ROCJITSU_CODE_ARCH_CDNA1,
                     duplicate_code_object_values),
  };
  expect_registry_error(duplicate_code_object_id, "duplicate ISA target GPU code-object ID");

  static constexpr std::array empty_code_object_value = {
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX90A, "", EF_AMDGPU_MACH_AMDGCN_GFX90A),
  };
  static constexpr std::array empty_code_object_id = {
      fixture_target("empty-code-object-id", {}, ROCJITSU_CODE_ARCH_CDNA1, empty_code_object_value),
  };
  expect_registry_error(empty_code_object_id, "empty GPU code-object ID");

  static constexpr std::array unknown_code_object_value = {
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX90A, "not-an-alias", EF_AMDGPU_MACH_AMDGCN_GFX90A),
  };
  static constexpr std::array unknown_code_object_id = {
      fixture_target("unknown-code-object-id", {}, ROCJITSU_CODE_ARCH_CDNA1,
                     unknown_code_object_value),
  };
  expect_registry_error(unknown_code_object_id, "not an ID or alias");

  static constexpr std::array empty_elf_value = {
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX90A, "empty-elf-machine", 0),
  };
  static constexpr std::array empty_elf_machine = {
      fixture_target("empty-elf-machine", {}, ROCJITSU_CODE_ARCH_CDNA1, empty_elf_value),
  };
  expect_registry_error(empty_elf_machine, "empty GPU ELF machine");

  static constexpr std::array missing_gpu_architecture_value = {
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX90A, "missing-gpu-architecture",
                         EF_AMDGPU_MACH_AMDGCN_GFX90A),
  };
  static constexpr std::array missing_gpu_architecture = {
      fixture_target("missing-gpu-architecture", {}, ROCJITSU_CODE_ARCH_INVALID,
                     missing_gpu_architecture_value),
  };
  expect_registry_error(missing_gpu_architecture, "must have an architecture ID");
}

TEST(IsaTargetRegistryTest, InvalidCompositionFailsClosed) {
  static constexpr std::array invalid_targets = {
      IsaTargetDescriptor{.id = "target"},
  };
  const IsaTargetRegistry registry{invalid_targets};

  EXPECT_FALSE(registry.ok());
  EXPECT_TRUE(registry.targets().empty());
  EXPECT_EQ(registry.find("target"), nullptr);
  EXPECT_EQ(registry.find(ROCJITSU_CODE_ARCH_CDNA1), nullptr);
  EXPECT_EQ(registry.find(ROCJITSU_CODE_TARGET_GFX90A), nullptr);
  EXPECT_EQ(registry.find_gpu_target_by_elf_machine(EF_AMDGPU_MACH_AMDGCN_GFX90A), nullptr);
  EXPECT_EQ(registry.find_gpu_target_by_code_object_id("gfx9999"), nullptr);
}

TEST(IsaTargetRegistryTest, ExecutionBackendScopesNestAndRestore) {
  constexpr std::array<Instruction::ExecuteFn, 1> outer_callbacks = {&first_execute};
  constexpr std::array<Instruction::ExecuteFn, 1> inner_callbacks = {&second_execute};
  constexpr int outer_operand_backend = 1;
  constexpr int inner_operand_backend = 2;
  const IsaExecutionBackend outer{
      .instruction_callbacks = outer_callbacks.data(),
      .instruction_callback_count = outer_callbacks.size(),
      .operand_backend = &outer_operand_backend,
  };
  const IsaExecutionBackend inner{
      .instruction_callbacks = inner_callbacks.data(),
      .instruction_callback_count = inner_callbacks.size(),
      .operand_backend = &inner_operand_backend,
  };

  EXPECT_EQ(current_instruction_execute(0), nullptr);
  EXPECT_EQ(current_isa_operand_backend(), nullptr);
  {
    ScopedIsaExecutionBackend outer_scope(&outer);
    EXPECT_EQ(current_instruction_execute(0), &first_execute);
    EXPECT_EQ(current_instruction_execute(1), nullptr);
    EXPECT_EQ(current_isa_operand_backend(), &outer_operand_backend);
    {
      ScopedIsaExecutionBackend inner_scope(&inner);
      EXPECT_EQ(current_instruction_execute(0), &second_execute);
      EXPECT_EQ(current_isa_operand_backend(), &inner_operand_backend);
    }
    EXPECT_EQ(current_instruction_execute(0), &first_execute);
    EXPECT_EQ(current_isa_operand_backend(), &outer_operand_backend);
  }
  EXPECT_EQ(current_instruction_execute(0), nullptr);
  EXPECT_EQ(current_isa_operand_backend(), nullptr);
}

TEST(IsaTargetRegistryTest, SourceIntegratedDescriptorUsesCanonicalIdsWithoutExtendingEnums) {
  static constexpr std::array<std::string_view, 1> aliases{"vendor-next"};
  static constexpr std::array npi_targets = {
      fixture_target("gfx9999", aliases),
  };
  const IsaTargetRegistry npi_registry{npi_targets};

  ASSERT_TRUE(npi_registry.ok()) << npi_registry.error();
  EXPECT_NE(npi_registry.find("gfx9999"), nullptr);
  EXPECT_NE(npi_registry.find("vendor-next"), nullptr);
  EXPECT_NE(Decoder::create(npi_registry, "gfx9999"), nullptr);

  static constexpr std::array unrelated_targets = {
      fixture_target("unrelated"),
  };
  const IsaTargetRegistry unrelated{unrelated_targets};
  EXPECT_EQ(unrelated.find("gfx9999"), nullptr);
}

TEST(IsaTargetRegistryTest, BuiltinRegistryUsesDescriptorOwnedPublicEnumBindings) {
  const IsaTargetRegistry &registry = default_isa_target_registry();
  ASSERT_TRUE(registry.ok()) << registry.error();
  const std::vector<std::string> expected = {
      "cdna1", "cdna2",   "cdna3", "cdna4",   "rdna1",  "rdna2",
      "rdna3", "rdna3_5", "rdna4", "gfx1250", "risc-v",
  };
  std::vector<std::string> actual;
  for (const IsaTargetDescriptor &target : registry.targets())
    actual.emplace_back(target.id);
  EXPECT_EQ(actual, expected);

  const IsaTargetDescriptor *gfx1201 = registry.find("gfx1201");
  ASSERT_NE(gfx1201, nullptr);
  EXPECT_EQ(gfx1201->id, "rdna4");
  const IsaTargetDescriptor *rv64i = registry.find("rv64i");
  ASSERT_NE(rv64i, nullptr);
  EXPECT_EQ(rv64i->id, "risc-v");
  const IsaTargetDescriptor *cdna3 = registry.find(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(cdna3, nullptr);
  EXPECT_EQ(cdna3->id, "cdna3");
  const IsaTargetDescriptor *gfx1201_enum = registry.find(ROCJITSU_CODE_TARGET_GFX1201);
  ASSERT_NE(gfx1201_enum, nullptr);
  EXPECT_EQ(gfx1201_enum->id, "rdna4");
  EXPECT_NE(Decoder::create(registry, "gfx942"), nullptr);
  EXPECT_NE(Decoder::create(registry, ROCJITSU_CODE_ARCH_CDNA3), nullptr);
  EXPECT_EQ(registry.find("rv32i"), nullptr);
  EXPECT_EQ(Decoder::create(registry, ROCJITSU_CODE_ARCH_RV32I), nullptr);
  std::unique_ptr<Decoder> risc_v_decoder = Decoder::create(registry, ROCJITSU_CODE_ARCH_RV64I);
  ASSERT_NE(risc_v_decoder, nullptr);
  constexpr rj_code_binary_inst_t kAddiX1X0One = 0x00100093;
  std::unique_ptr<Instruction> risc_v_instruction(risc_v_decoder->decode(&kAddiX1X0One));
  ASSERT_NE(risc_v_instruction, nullptr);
  EXPECT_NE(risc_v_instruction->execute, nullptr);
}

TEST(IsaTargetRegistryTest, PublicCEntryPointAcceptsCanonicalTargetIds) {
  rj_code_decoder_t *decoder = nullptr;
  EXPECT_EQ(rj_code_decoder_create_for_target("gfx942", &decoder), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(decoder, nullptr);
  rj_code_decoder_destroy(decoder);

  decoder = nullptr;
  EXPECT_EQ(rj_code_decoder_create_for_target("rv64i", &decoder), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(decoder, nullptr);
  rj_code_decoder_destroy(decoder);

  decoder = nullptr;
  EXPECT_EQ(rj_code_decoder_create_for_target("vendor-not-linked", &decoder),
            ROCJITSU_STATUS_ERROR);
  EXPECT_EQ(decoder, nullptr);
  EXPECT_EQ(rj_code_decoder_create_for_target("", &decoder), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_code_decoder_create_for_target(nullptr, &decoder), ROCJITSU_STATUS_INVALID_ARGUMENT);

  EXPECT_EQ(rj_code_decoder_create(ROCJITSU_CODE_ARCH_INVALID, &decoder),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(decoder, nullptr);

  constexpr auto unnamed_architecture =
      static_cast<rj_code_arch_t>(ROCJITSU_CODE_ARCH_NUM_ARCHS + 1);
  EXPECT_EQ(rj_code_decoder_create(unnamed_architecture, &decoder),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(decoder, nullptr);

  EXPECT_EQ(rj_code_decoder_create(ROCJITSU_CODE_ARCH_RV32I, &decoder), ROCJITSU_STATUS_ERROR);
  EXPECT_EQ(decoder, nullptr);
}

} // namespace
} // namespace rocjitsu
