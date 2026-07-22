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

std::unique_ptr<Decoder> create_fixture_decoder(const IsaExecutionBackend *) {
  return std::make_unique<FixtureDecoder>();
}

IsaTargetDescriptor fixture_target(std::string id) {
  IsaTargetDescriptor descriptor;
  descriptor.id = std::move(id);
  descriptor.decoder_factory = &create_fixture_decoder;
  return descriptor;
}

IsaGpuTargetBinding fixture_gpu_target(rj_code_target_id_t public_id, std::string code_object_id,
                                       uint32_t elf_machine) {
  return {
      .public_id = public_id,
      .code_object_id = std::move(code_object_id),
      .elf_machine = elf_machine,
  };
}

void expect_registry_error(IsaTargetRegistryError error, std::string_view expected) {
  ASSERT_TRUE(error.has_value()) << "expected registry error containing: " << expected;
  EXPECT_NE(error->find(expected), std::string::npos) << "registry error was: " << *error;
}

IsaTargetRegistryError register_npi_target(IsaTargetRegistry &registry) {
  IsaTargetDescriptor descriptor = fixture_target("gfx9999");
  descriptor.aliases = {"vendor-next"};
  return registry.add(std::move(descriptor));
}

IsaTargetRegistryError register_failing_target(IsaTargetRegistry &) {
  return std::string{"fixture provider failed"};
}

IsaTargetRegistryError register_empty_error(IsaTargetRegistry &) { return std::string{}; }

void first_execute(Instruction &, void *) {}
void second_execute(Instruction &, void *) {}

static_assert(std::is_move_constructible_v<IsaTargetRegistry>);
static_assert(!std::is_move_assignable_v<IsaTargetRegistry>);
static_assert(cdna1::target_description.aliases.empty());
static_assert(cdna2::target_description.id == "cdna2");
static_assert(cdna2::target_description.aliases.size() == 1);
static_assert(cdna2::target_description.aliases.front() == "gfx90a");
static_assert(cdna2::target_description.architecture_id == ROCJITSU_CODE_ARCH_CDNA2);
static_assert(cdna2::target_description.gpu_targets.front().public_id ==
              ROCJITSU_CODE_TARGET_GFX90A);
static_assert(rdna4::target_description.aliases.size() == 2);
static_assert(rdna4::target_description.aliases[0] == "gfx1200");
static_assert(rdna4::target_description.aliases[1] == "gfx1201");
static_assert(static_cast<int>(ROCJITSU_CODE_TARGET_GFX1250) == 5);
static_assert(static_cast<int>(ROCJITSU_CODE_TARGET_INVALID) == 6);

TEST(IsaTargetRegistryTest, FreezesIntoDeterministicCanonicalOrder) {
  IsaTargetRegistry registry;
  EXPECT_EQ(registry.add(fixture_target("zeta")), std::nullopt);
  EXPECT_EQ(registry.add(fixture_target("alpha")), std::nullopt);
  ASSERT_EQ(registry.freeze(), std::nullopt);

  ASSERT_EQ(registry.targets().size(), 2u);
  EXPECT_EQ(registry.targets()[0].id, "alpha");
  EXPECT_EQ(registry.targets()[1].id, "zeta");
  EXPECT_EQ(registry.find("missing"), nullptr);
  EXPECT_TRUE(registry.add(fixture_target("late")).has_value());
  EXPECT_TRUE(registry.freeze().has_value());
}

TEST(IsaTargetRegistryTest, RejectsConflictingCanonicalIdentities) {
  IsaTargetRegistry registry;
  IsaTargetDescriptor first = fixture_target("first");
  first.aliases = {"shared"};
  ASSERT_EQ(registry.add(std::move(first)), std::nullopt);

  IsaTargetDescriptor duplicate_id = fixture_target("shared");
  expect_registry_error(registry.add(std::move(duplicate_id)), "duplicate ISA target ID");

  IsaTargetDescriptor duplicate_local_alias = fixture_target("second");
  duplicate_local_alias.aliases = {"twice", "twice"};
  expect_registry_error(registry.add(std::move(duplicate_local_alias)), "duplicate ISA target ID");

  EXPECT_EQ(registry.add(fixture_target("still-usable")), std::nullopt);
}

TEST(IsaTargetRegistryTest, RejectsConflictingPublicEnumKeys) {
  IsaTargetRegistry registry;
  IsaTargetDescriptor first = fixture_target("first");
  first.architecture_id = ROCJITSU_CODE_ARCH_CDNA1;
  first.gpu_targets = {
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX90A, "first", EF_AMDGPU_MACH_AMDGCN_GFX90A)};
  ASSERT_EQ(registry.add(std::move(first)), std::nullopt);

  IsaTargetDescriptor duplicate_architecture = fixture_target("duplicate-architecture");
  duplicate_architecture.architecture_id = ROCJITSU_CODE_ARCH_CDNA1;
  expect_registry_error(registry.add(std::move(duplicate_architecture)),
                        "duplicate ISA target architecture");

  IsaTargetDescriptor duplicate_gpu_target = fixture_target("duplicate-gpu-target");
  duplicate_gpu_target.architecture_id = ROCJITSU_CODE_ARCH_CDNA2;
  duplicate_gpu_target.gpu_targets = {fixture_gpu_target(
      ROCJITSU_CODE_TARGET_GFX90A, "duplicate-gpu-target", EF_AMDGPU_MACH_AMDGCN_GFX942)};
  expect_registry_error(registry.add(std::move(duplicate_gpu_target)),
                        "duplicate ISA target GPU target");

  IsaTargetDescriptor duplicate_elf_machine = fixture_target("duplicate-elf-machine");
  duplicate_elf_machine.architecture_id = ROCJITSU_CODE_ARCH_CDNA2;
  duplicate_elf_machine.gpu_targets = {fixture_gpu_target(
      ROCJITSU_CODE_TARGET_GFX942, "duplicate-elf-machine", EF_AMDGPU_MACH_AMDGCN_GFX90A)};
  expect_registry_error(registry.add(std::move(duplicate_elf_machine)),
                        "duplicate ISA target GPU ELF machine");
}

TEST(IsaTargetRegistryTest, RejectsInvalidTargetDescriptors) {
  IsaTargetRegistry registry;
  expect_registry_error(registry.add(fixture_target("")), "canonical ID must not be empty");

  IsaTargetDescriptor missing_factory = fixture_target("missing-factory");
  missing_factory.decoder_factory = nullptr;
  expect_registry_error(registry.add(std::move(missing_factory)), "no decoder factory");

  IsaTargetDescriptor empty_alias = fixture_target("empty-alias");
  empty_alias.aliases = {""};
  expect_registry_error(registry.add(std::move(empty_alias)), "empty alias");

  IsaTargetDescriptor invalid_enums = fixture_target("invalid-enums");
  invalid_enums.architecture_id =
      static_cast<rj_code_arch_t>(static_cast<int>(ROCJITSU_CODE_ARCH_NUM_ARCHS) + 1);
  expect_registry_error(registry.add(std::move(invalid_enums)), "unallocated architecture");

  IsaTargetDescriptor invalid_gpu_target = fixture_target("invalid-gpu-target");
  invalid_gpu_target.gpu_targets = {fixture_gpu_target(
      ROCJITSU_CODE_TARGET_INVALID, "invalid-gpu-target", EF_AMDGPU_MACH_AMDGCN_GFX90A)};
  expect_registry_error(registry.add(std::move(invalid_gpu_target)), "unallocated GPU target");

  IsaTargetDescriptor duplicate_enums = fixture_target("duplicate-enums");
  duplicate_enums.aliases = {"duplicate-enums-alias"};
  duplicate_enums.gpu_targets = {
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX90A, "duplicate-enums",
                         EF_AMDGPU_MACH_AMDGCN_GFX90A),
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX90A, "duplicate-enums-alias",
                         EF_AMDGPU_MACH_AMDGCN_GFX942),
  };
  expect_registry_error(registry.add(std::move(duplicate_enums)),
                        "duplicate ISA target GPU target");

  IsaTargetDescriptor duplicate_code_object_id = fixture_target("duplicate-code-object-id");
  duplicate_code_object_id.gpu_targets = {
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX90A, "duplicate-code-object-id",
                         EF_AMDGPU_MACH_AMDGCN_GFX90A),
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX942, "duplicate-code-object-id",
                         EF_AMDGPU_MACH_AMDGCN_GFX942),
  };
  expect_registry_error(registry.add(std::move(duplicate_code_object_id)),
                        "duplicate ISA target GPU code-object ID");

  IsaTargetDescriptor empty_code_object_id = fixture_target("empty-code-object-id");
  empty_code_object_id.gpu_targets = {
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX90A, "", EF_AMDGPU_MACH_AMDGCN_GFX90A)};
  expect_registry_error(registry.add(std::move(empty_code_object_id)), "empty GPU code-object ID");

  IsaTargetDescriptor unknown_code_object_id = fixture_target("unknown-code-object-id");
  unknown_code_object_id.gpu_targets = {fixture_gpu_target(
      ROCJITSU_CODE_TARGET_GFX90A, "not-an-alias", EF_AMDGPU_MACH_AMDGCN_GFX90A)};
  expect_registry_error(registry.add(std::move(unknown_code_object_id)), "not an ID or alias");

  IsaTargetDescriptor empty_elf_machine = fixture_target("empty-elf-machine");
  empty_elf_machine.gpu_targets = {
      fixture_gpu_target(ROCJITSU_CODE_TARGET_GFX90A, "empty-elf-machine", 0)};
  expect_registry_error(registry.add(std::move(empty_elf_machine)), "empty GPU ELF machine");

  IsaTargetDescriptor missing_gpu_architecture = fixture_target("missing-gpu-architecture");
  missing_gpu_architecture.gpu_targets = {fixture_gpu_target(
      ROCJITSU_CODE_TARGET_GFX90A, "missing-gpu-architecture", EF_AMDGPU_MACH_AMDGCN_GFX90A)};
  expect_registry_error(registry.add(std::move(missing_gpu_architecture)),
                        "must have an architecture ID");

  EXPECT_EQ(registry.add(fixture_target("valid-after-errors")), std::nullopt);
}

TEST(IsaTargetRegistryTest, LookupsFailClosedBeforeFreezeAndNullProvidersReportErrors) {
  IsaTargetRegistry registry;
  ASSERT_EQ(registry.add(fixture_target("target")), std::nullopt);
  EXPECT_TRUE(registry.targets().empty());
  EXPECT_EQ(registry.find("target"), nullptr);
  EXPECT_EQ(registry.find(ROCJITSU_CODE_ARCH_CDNA1), nullptr);
  EXPECT_EQ(registry.find(ROCJITSU_CODE_TARGET_GFX90A), nullptr);
  EXPECT_EQ(registry.find_gpu_target_by_elf_machine(EF_AMDGPU_MACH_AMDGCN_GFX90A), nullptr);
  EXPECT_EQ(registry.find_gpu_target_by_code_object_id("gfx9999"), nullptr);

  constexpr std::array<IsaTargetProvider, 1> providers = {nullptr};
  IsaTargetRegistry failed_registry = make_isa_target_registry(providers);
  EXPECT_FALSE(failed_registry.ok());
  EXPECT_FALSE(failed_registry.frozen());
  EXPECT_FALSE(failed_registry.error().empty());
  EXPECT_TRUE(failed_registry.targets().empty());
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

TEST(IsaTargetRegistryTest, SourceIntegratedProviderUsesCanonicalIdsWithoutExtendingEnums) {
  constexpr IsaTargetProvider providers[] = {&register_npi_target};
  IsaTargetRegistry npi_registry = make_isa_target_registry(providers);

  ASSERT_TRUE(npi_registry.ok()) << npi_registry.error();
  ASSERT_TRUE(npi_registry.frozen());
  EXPECT_NE(npi_registry.find("gfx9999"), nullptr);
  EXPECT_NE(npi_registry.find("vendor-next"), nullptr);
  EXPECT_NE(Decoder::create(npi_registry, "gfx9999"), nullptr);

  IsaTargetRegistry unrelated;
  ASSERT_EQ(unrelated.add(fixture_target("unrelated")), std::nullopt);
  ASSERT_EQ(unrelated.freeze(), std::nullopt);
  EXPECT_EQ(unrelated.find("gfx9999"), nullptr);
}

TEST(IsaTargetRegistryTest, ProviderCompositionReturnsItsDiagnostic) {
  constexpr IsaTargetProvider providers[] = {&register_npi_target, &register_failing_target};
  IsaTargetRegistry registry = make_isa_target_registry(providers);

  EXPECT_FALSE(registry.ok());
  EXPECT_FALSE(registry.frozen());
  EXPECT_EQ(registry.error(), "fixture provider failed");
  EXPECT_TRUE(registry.targets().empty());
  EXPECT_EQ(registry.find("gfx9999"), nullptr);
}

TEST(IsaTargetRegistryTest, ProviderCompositionTracksAnEmptyErrorValue) {
  constexpr IsaTargetProvider providers[] = {&register_empty_error};
  IsaTargetRegistry registry = make_isa_target_registry(providers);

  EXPECT_FALSE(registry.ok());
  EXPECT_FALSE(registry.frozen());
  EXPECT_TRUE(registry.error().empty());
  EXPECT_TRUE(registry.targets().empty());
}

TEST(IsaTargetRegistryTest, BuiltinRegistryUsesProviderOwnedPublicEnumBindings) {
  const IsaTargetRegistry &registry = default_isa_target_registry();
  ASSERT_TRUE(registry.ok()) << registry.error();
  const std::vector<std::string> expected = {
      "cdna1", "cdna2", "cdna3",   "cdna4", "gfx1250", "rdna1",
      "rdna2", "rdna3", "rdna3_5", "rdna4", "risc-v",
  };
  std::vector<std::string> actual;
  for (const IsaTargetDescriptor &target : registry.targets())
    actual.push_back(target.id);
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
  auto risc_v_decoder = Decoder::create(registry, ROCJITSU_CODE_ARCH_RV64I);
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
