// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

#include "rocjitsu/code/patch/consan/consan_identity_contracts.h"
#include "rocjitsu/code/patch/consan/consan_internal.h"
#include "rocjitsu/code/patch/consan/consan_physical_site_alias.h"
#include "rocjitsu/code/patch/consan/targets/consan_target_lds_ops.h"

namespace rocjitsu::consan {
namespace {

struct PhysicalAliasTestCandidate {
  uint64_t file_offset = 0;
  std::string container_name;
  uint32_t semantics = 0;
  std::vector<uint64_t> owners;
  std::optional<uint64_t> descriptor;

  bool operator==(const PhysicalAliasTestCandidate &) const = default;
};

auto make_physical_alias_test_canonicalizer(std::vector<std::string> &errors,
                                            size_t expected_candidate_count = 0) {
  return detail::make_physical_site_alias_canonicalizer<PhysicalAliasTestCandidate>(
      errors, "ConSan incremental test site",
      [](const PhysicalAliasTestCandidate &candidate) { return candidate.file_offset; },
      [](const PhysicalAliasTestCandidate &candidate) -> std::string_view {
        return candidate.container_name;
      },
      [](const PhysicalAliasTestCandidate &lhs, const PhysicalAliasTestCandidate &rhs) {
        return lhs.semantics == rhs.semantics;
      },
      [](PhysicalAliasTestCandidate &retained, const PhysicalAliasTestCandidate &alias) {
        retained.owners.insert(retained.owners.end(), alias.owners.begin(), alias.owners.end());
        retained.descriptor.reset();
      },
      expected_candidate_count);
}

TEST(ConSan, SelectableVgprBankStateDispatchesOnlyToOwningTarget) {
  constexpr uint32_t kSetVgprBankModeFour = 0xBF860004u;
  std::array<uint8_t, sizeof(kSetVgprBankModeFour)> bytes{};
  std::memcpy(bytes.data(), &kSetVgprBankModeFour, sizeof(kSetVgprBankModeFour));

  EXPECT_EQ(selectable_vgpr_bank_mode_at(ROCJITSU_CODE_ARCH_CDNA5, bytes, 0u, 0u, bytes.size()),
            4u);
  EXPECT_FALSE(selectable_vgpr_bank_mode_at(ROCJITSU_CODE_ARCH_RDNA4, bytes, 0u, 0u, bytes.size()));
  EXPECT_FALSE(selectable_vgpr_bank_mode_at(ROCJITSU_CODE_ARCH_CDNA5, bytes, 1u, bytes.size(),
                                            bytes.size()));
  EXPECT_TRUE(
      selectable_vgpr_bank_transition_in_range(ROCJITSU_CODE_ARCH_CDNA5, bytes, 0u, bytes.size()));
  EXPECT_FALSE(
      selectable_vgpr_bank_transition_in_range(ROCJITSU_CODE_ARCH_RDNA4, bytes, 0u, bytes.size()));
  EXPECT_FALSE(
      selectable_vgpr_bank_transition_in_range(ROCJITSU_CODE_ARCH_CDNA5, bytes, bytes.size(), 0u));
}

TEST(ConSan, SplitTwoAddressLdsRecipeDispatchesOnlyToOwningTarget) {
  const SplitTwoAddressLdsRequest request{
      .first_byte_offset = 4u,
      .second_byte_offset = 8u,
      .element_dwords = 1u,
      .address_vgpr = 2u,
      .first_data_vgpr = 3u,
      .second_data_vgpr = 4u,
      .adjusted_address_vgpr = std::nullopt,
      .load = true,
  };

  const auto gfx1250 = build_split_two_address_lds_pair(request, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(gfx1250);
  EXPECT_EQ(gfx1250->size(), 4u);
  EXPECT_FALSE(build_split_two_address_lds_pair(request, ROCJITSU_CODE_ARCH_RDNA4));

  SplitTwoAddressLdsRequest large_offset = request;
  large_offset.second_byte_offset = UINT16_MAX + 1u;
  EXPECT_FALSE(build_split_two_address_lds_pair(large_offset, ROCJITSU_CODE_ARCH_CDNA5));
  large_offset.adjusted_address_vgpr = 5u;
  const auto adjusted = build_split_two_address_lds_pair(large_offset, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(adjusted);
  EXPECT_EQ(adjusted->size(), 6u);
}

TEST(ConSan, OperatingPointEqualityCoversOwnerAssignments) {
  TransientSgprAssignment owner_state;
  owner_state.descriptor_file_offset = 64u;
  owner_state.exec_save_sgpr = 8u;
  owner_state.dispatch_id_sgpr = 10u;
  PersistentVgprAssignment persistent_state;
  persistent_state.descriptor_file_offset = 64u;
  persistent_state.owner_epoch_vgprs = OwnerEpochRegisters{16u, 17u};
  OperatingPoint allocation;
  allocation.automatic_private_epoch = true;
  allocation.exec_save_sgpr = 2u;
  allocation.dispatch_sgpr.set(4u);
  allocation.owner_persistent_vgprs = {persistent_state};
  allocation.owner_transient_sgprs = {owner_state};

  EXPECT_EQ(OperatingPoint{}, OperatingPoint{});
  EXPECT_EQ(allocation, OperatingPoint(allocation));

  OperatingPoint changed = allocation;
  changed.automatic_private_epoch = false;
  EXPECT_NE(changed, allocation);
  changed = allocation;
  changed.owner_persistent_vgprs.clear();
  EXPECT_NE(changed, allocation);
  changed = allocation;
  changed.exec_save_sgpr.reset();
  EXPECT_NE(changed, allocation);
  changed = allocation;
  changed.owner_transient_sgprs.clear();
  EXPECT_NE(changed, allocation);
  changed = allocation;
  changed.dispatch_sgpr.reset();
  EXPECT_NE(changed, allocation);
}

TEST(ConSan, PersistentVgprStateHasOneProjectionForEveryAllocationScope) {
  OperatingPoint point;
  point.set_owner_epoch_vgprs(3u, 4u);
  point.exact_workgroup_vgprs = PersistentWorkgroupRegisters(8u, 9u, 10u, 11u);

  PersistentVgprAssignment assignment{
      .descriptor_file_offset = 64u,
      .owner_epoch_vgprs = {3u, 4u},
      .exact_workgroup_vgprs = PersistentWorkgroupRegisters(8u, 9u, 10u, 11u),
  };
  const auto point_state = detail::persistent_vgpr_state_view(point);
  EXPECT_EQ(point_state, detail::persistent_vgpr_state_view(assignment));

  std::vector<std::pair<uint16_t, uint16_t>> ranges;
  point_state.for_each_range([&](std::optional<uint16_t> base, uint16_t width) {
    if (base)
      ranges.emplace_back(*base, width);
  });
  EXPECT_EQ(ranges, (std::vector<std::pair<uint16_t, uint16_t>>{
                        {3u, 1u}, {4u, 1u}, {8u, 1u}, {9u, 1u}, {10u, 1u}, {11u, 1u}}));
}

TEST(ConSan, PersistentSgprStateOwnsItsCompleteDescriptorExtent) {
  OperatingPoint point;
  point.persistent_sgprs.set_owner_epoch(20u, 21u);
  point.persistent_sgprs.exact_workgroup = PersistentWorkgroupRegisters(23u, 24u, 25u, 26u);
  detail::ResolvedScratchPlan resources;
  resources.owner_descriptor_file_offsets = {64u};
  detail::DescriptorSgprRequirements requirements;

  detail::note_sgpr_requirements(requirements, resources, Request{}, BoundRuntimeResources{}, point,
                                 ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_EQ(requirements.size(), 1u);
  EXPECT_EQ(requirements.at(64u), 27u);
}

TEST(ConSan, ExactEntryWorkgroupCaptureHasOneExplicitStorageDomain) {
  OperatingPoint point;
  const PersistentWorkgroupPrivateOffsets private_offsets{32u, 36u, 40u};

  EXPECT_FALSE(detail::has_exact_entry_workgroup_capture(point));
  EXPECT_TRUE(detail::has_exact_entry_workgroup_capture(point, &private_offsets));
  EXPECT_TRUE(detail::exact_entry_workgroup_capture_is_unambiguous(point, &private_offsets));

  point.exact_workgroup_vgprs = PersistentWorkgroupRegisters{26u, 27u, 28u};
  EXPECT_TRUE(detail::has_exact_entry_workgroup_capture(point));
  EXPECT_FALSE(detail::exact_entry_workgroup_capture_is_unambiguous(point, &private_offsets));
}

TEST(ConSan, OperatingPointEqualityCoversEveryCodeObjectWideSelection) {
  OperatingPoint state{
      .initialize_owner_epoch = true,
      .exec_save_sgpr = 2u,
      .owner_sgpr = {},
      .owner_epoch_vgprs = OwnerEpochRegisters{4u, 5u},
      .automatic_persistent_vgprs = true,
      .automatic_private_epoch = true,
      .automatic_partial_exec_save_sgprs = true,
      .automatic_scalar_spill_layout = ScalarSpillLayout::Compact,
      .exec_save_sgprs_persistent = true,
      .dynamic_stack_spill = true,
      .scalar_spill_setup =
          ScalarSpillSetup{
              .temporaries = ScalarSpillTemporaries{2u, 7u},
          },
      .branch_only_spill = BranchOnlyScalarSpill{10u},
      .dispatch_sgpr = {},
      .persistent_sgprs = {},
      .exact_workgroup_vgprs = PersistentWorkgroupRegisters{26u, 27u, 28u},
      .owner_persistent_vgprs = {},
      .owner_transient_sgprs = {},
  };
  state.owner_sgpr.set(3u, true);
  state.dispatch_sgpr.set(16u, true);
  state.persistent_sgprs.set_owner_epoch(20u, 21u);
  state.persistent_sgprs.exact_workgroup = PersistentWorkgroupRegisters{23u, 24u, 25u};

  EXPECT_EQ(OperatingPoint{}, OperatingPoint{});
  EXPECT_EQ(state, OperatingPoint(state));

  auto expect_field_participates = [&](auto mutate) {
    OperatingPoint changed = state;
    mutate(changed);
    EXPECT_NE(changed, state);
  };
  expect_field_participates([](auto &value) { value.initialize_owner_epoch = false; });
  expect_field_participates([](auto &value) { value.exec_save_sgpr.reset(); });
  expect_field_participates([](auto &value) { value.owner_sgpr.reset(); });
  expect_field_participates([](auto &value) { value.owner_epoch_vgprs.reset_owner_epoch(); });
  expect_field_participates([](auto &value) { value.owner_epoch_vgprs.set_owner_epoch(5u, 5u); });
  expect_field_participates([](auto &value) { value.owner_epoch_vgprs.set_owner_epoch(4u, 6u); });
  expect_field_participates([](auto &value) { value.automatic_persistent_vgprs = false; });
  expect_field_participates([](auto &value) { value.automatic_private_epoch = false; });
  expect_field_participates([](auto &value) { value.automatic_partial_exec_save_sgprs = false; });
  expect_field_participates(
      [](auto &value) { value.automatic_scalar_spill_layout = ScalarSpillLayout::None; });
  expect_field_participates([](auto &value) { value.exec_save_sgprs_persistent = false; });
  expect_field_participates([](auto &value) { value.dynamic_stack_spill = false; });
  expect_field_participates([](auto &value) { value.owner_sgpr.set(3u, false); });
  expect_field_participates([](auto &value) { value.dispatch_sgpr.set(16u, false); });
  expect_field_participates([](auto &value) { value.scalar_spill_setup.reset(); });
  expect_field_participates([](auto &value) { value.branch_only_spill.reset(); });
  expect_field_participates([](auto &value) { value.dispatch_sgpr.reset(); });
  expect_field_participates([](auto &value) { value.persistent_sgprs = {}; });
  expect_field_participates([](auto &value) { value.exact_workgroup_vgprs = {}; });
}

TEST(ConSan, OptionsSeedsSelectedRegistersWithoutMutatingCallerInput) {
  Options input;
  input.requested_exec_save_sgpr = 2u;
  input.requested_owner_sgpr = 3u;
  input.requested_owner_vgpr = 4u;
  input.requested_epoch_vgpr = 5u;

  TestOptions attempt(input);
  EXPECT_EQ(attempt.exec_save_sgpr, 2u);
  EXPECT_EQ(attempt.owner_sgpr.base(), 3u);
  EXPECT_EQ(attempt.owner_epoch_vgprs.owner(), 4u);
  EXPECT_EQ(attempt.owner_epoch_vgprs.epoch(), 5u);

  attempt.exec_save_sgpr = 12u;
  attempt.owner_sgpr.reset();
  attempt.set_owner_epoch_vgprs(14u, 15u);

  EXPECT_EQ(input.requested_exec_save_sgpr, 2u);
  EXPECT_EQ(input.requested_owner_sgpr, 3u);
  EXPECT_EQ(input.requested_owner_vgpr, 4u);
  EXPECT_EQ(input.requested_epoch_vgpr, 5u);
  const TestOptions fresh_attempt(input);
  EXPECT_EQ(fresh_attempt.exec_save_sgpr, 2u);
  EXPECT_EQ(fresh_attempt.owner_sgpr.base(), 3u);
  EXPECT_EQ(fresh_attempt.owner_epoch_vgprs.owner(), 4u);
  EXPECT_EQ(fresh_attempt.owner_epoch_vgprs.epoch(), 5u);
}

TEST(ConSan, ExecSaveRequirementProjectsOnlyScalarAbiFacts) {
  Request request;
  request.track_atomics = true;
  request.runtime_sample_stride = 7u;
  BoundRuntimeResources resources;
  resources.report_buffer_address = 0x1000u;
  resources.report_layout = ReportBufferLayout{};
  OperatingPoint point;
  point.automatic_scalar_spill_layout = ScalarSpillLayout::Compact;
  point.dynamic_stack_spill = true;
  EXPECT_EQ(resolve_exec_save_requirement(request, resources, point),
            (ExecSaveRequirement{
                .has_report_buffer = true,
                .track_atomics = true,

                .scalar_spill = true,
                .dynamic_stack_spill = true,

            }));
}

TEST(ConSan, OwnerEpochInitializationIsAnOperatingPointDecision) {
  Options options;
  EXPECT_FALSE(initial_operating_point(options, options).initialize_owner_epoch);

  options.init_owner_epoch = true;
  OperatingPoint point = initial_operating_point(options, options);
  EXPECT_TRUE(point.initialize_owner_epoch);

  options.init_owner_epoch = false;
  EXPECT_TRUE(point.initialize_owner_epoch);
  point.initialize_owner_epoch = false;
  EXPECT_FALSE(point.initialize_owner_epoch);
  EXPECT_FALSE(options.init_owner_epoch);
}

TEST(ConSan, ResourceProblemBindsImmutableSolverInputs) {
  const std::array<uint8_t, 4> image{1u, 2u, 3u, 4u};
  Request request;
  BoundRuntimeResources resources;
  resources.report_buffer_address = 0x2000u;
  ProgramInventory inventory;
  ObservationPlan observation_plan;
  const ProgramSite site;
  const std::array<Candidate, 1> candidates{Candidate(site)};
  const ResourceProblem problem(image, ROCJITSU_CODE_ARCH_CDNA5, request, resources, inventory,
                                observation_plan, candidates);
  EXPECT_EQ(problem.image().data(), image.data());
  EXPECT_EQ(problem.image().size(), image.size());
  EXPECT_EQ(problem.arch(), ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(&problem.request(), &request);
  EXPECT_EQ(&problem.resources(), &resources);
  EXPECT_EQ(&problem.inventory(), &inventory);
  EXPECT_EQ(&problem.observation_plan(), &observation_plan);
  EXPECT_EQ(problem.candidates().data(), candidates.data());
  EXPECT_EQ(problem.candidates().size(), candidates.size());
}

TEST(ConSan, ExecSaveRequirementOwnsTargetAndFallbackSizing) {
  ExecSaveRequirement requirement{};
  EXPECT_EQ(exec_save_sgpr_count(requirement, ROCJITSU_CODE_ARCH_RDNA3), 0u);

  requirement = {.has_report_buffer = true};
  EXPECT_EQ(exec_save_sgpr_count(requirement, ROCJITSU_CODE_ARCH_RDNA3), 7u);
  EXPECT_EQ(exec_save_sgpr_count(requirement, ROCJITSU_CODE_ARCH_CDNA5), 8u);
  requirement.track_atomics = true;
  EXPECT_EQ(exec_save_sgpr_count(requirement, ROCJITSU_CODE_ARCH_RDNA3), 8u);
  requirement = {.has_report_buffer = true, .dynamic_stack_spill = true};
  EXPECT_EQ(exec_save_sgpr_count(requirement, ROCJITSU_CODE_ARCH_CDNA5), 9u);

  requirement = {.has_report_buffer = true, .scalar_spill = true};
  EXPECT_EQ(exec_save_sgpr_count(requirement, ROCJITSU_CODE_ARCH_CDNA5), 8u);
}

TEST(ConSan, ResourcePlanningResultSeparatesStructuralFailureFromUnsupportedSites) {
  ResourcePlanningResult planning;
  planning.attempted_operating_point.exec_save_sgpr = 12u;
  CandidateResourcePlan unsupported;
  unsupported.source = RegisterAllocationSource::Unsupported;
  unsupported.reason = RegisterPlanReason::NoLegalWindow;
  planning.plans.push_back(unsupported);
  planning.selected_fallback = FallbackKind::DynamicStackScalarSpill;
  planning.diagnostics.emplace_back("accepted attempt diagnostic");
  EXPECT_TRUE(planning.success());

  ResourcePlanningResult accepted_attempt = planning;
  const auto accepted = std::move(accepted_attempt).accept();
  ASSERT_TRUE(accepted);
  EXPECT_EQ(accepted->operating_point.exec_save_sgpr, 12u);
  ASSERT_EQ(accepted->site_plans.size(), 1u);
  EXPECT_EQ(accepted->site_plans.front().source, RegisterAllocationSource::Unsupported);
  EXPECT_EQ(accepted->selected_fallback, FallbackKind::DynamicStackScalarSpill);
  ASSERT_EQ(accepted->diagnostics.size(), 1u);
  EXPECT_EQ(accepted->diagnostics.front(), "accepted attempt diagnostic");

  planning.errors.emplace_back("inconsistent physical alias");
  EXPECT_FALSE(planning.success());
  EXPECT_EQ(planning.plans.size(), 1u);
  EXPECT_FALSE(std::move(planning).accept());
}

TEST(ConSan, OperatingPointAttemptPublishesOnlyTypedAcceptedFallbacks) {
  OperatingPointAttempt rejected;
  rejected.attempted_operating_point.exec_save_sgpr = 42u;
  EXPECT_FALSE(rejected.accepted());
  EXPECT_FALSE(rejected.accepted_fallback.has_value());

  OperatingPointAttempt accepted;
  accepted.attempted_operating_point.exec_save_sgpr = 44u;
  accepted.accepted_fallback = FallbackKind::LiteralDispatchId;
  accepted.diagnostics.emplace_back("rendered only after acceptance");
  ASSERT_TRUE(accepted.accepted());
  EXPECT_EQ(*accepted.accepted_fallback, FallbackKind::LiteralDispatchId);
  EXPECT_EQ(accepted.attempted_operating_point.exec_save_sgpr, 44u);
  EXPECT_EQ(accepted.diagnostics.size(), 1u);
}

TEST(ConSan, OperatingPointUpdateCarriesPointDiagnosticsAndTypedRejectionTogether) {
  OperatingPointUpdate accepted;
  accepted.attempted_operating_point.dispatch_sgpr.set(40u);
  accepted.diagnostics.emplace_back("accepted placement");
  EXPECT_TRUE(accepted.accepted());
  EXPECT_EQ(accepted.attempted_operating_point.dispatch_sgpr.base(), 40u);
  EXPECT_EQ(accepted.diagnostics, std::vector<std::string>{"accepted placement"});

  OperatingPointUpdate rejected = accepted;
  rejected.rejection = PlacementRejection::PersistentStateUnavailable;
  EXPECT_FALSE(rejected.accepted());
  EXPECT_EQ(rejected.rejection, PlacementRejection::PersistentStateUnavailable);
}

TEST(ConSan, PersistentPlacementUpdateKeepsEntryScratchWithItsOperatingPoint) {
  PersistentPlacementUpdate accepted;
  accepted.attempted_operating_point.set_owner_epoch_vgprs(40u, 41u);
  accepted.prologue_scratch_assignments.push_back(
      {.descriptor_file_offset = 96u, .scratch_vgpr = 42u});
  EXPECT_TRUE(accepted.accepted());
  ASSERT_EQ(accepted.prologue_scratch_assignments.size(), 1u);
  EXPECT_EQ(accepted.prologue_scratch_assignments.front().descriptor_file_offset, 96u);
  EXPECT_EQ(accepted.prologue_scratch_assignments.front().scratch_vgpr, 42u);

  PersistentPlacementUpdate rejected = accepted;
  rejected.rejection = PlacementRejection::PersistentStateUnavailable;
  EXPECT_FALSE(rejected.accepted());
}

TEST(ConSan, PersistentScalarStateRequiresTheOwnerEpochPair) {
  PersistentSgprState state;
  EXPECT_FALSE(state.complete());
  EXPECT_FALSE(state.owner());
  EXPECT_FALSE(state.epoch());

  state.set_owner_epoch(12u, 13u);
  EXPECT_TRUE(state.complete());
  EXPECT_EQ(state.owner(), 12u);
  EXPECT_EQ(state.epoch(), 13u);

  state.reset_owner_epoch();
  EXPECT_FALSE(state.complete());
  EXPECT_FALSE(state.owner());
  EXPECT_FALSE(state.epoch());
}

TEST(ConSan, OperatingPointPersistentOwnerEpochVgprsAreAllOrNothing) {
  OperatingPoint point;
  EXPECT_FALSE(point.owner_epoch_vgprs.complete());
  EXPECT_FALSE(point.owner_epoch_vgprs.owner());
  EXPECT_FALSE(point.owner_epoch_vgprs.epoch());

  point.set_owner_epoch_vgprs(12u, 13u);
  ASSERT_TRUE(point.owner_epoch_vgprs.complete());
  EXPECT_EQ(point.owner_epoch_vgprs.owner(), 12u);
  EXPECT_EQ(point.owner_epoch_vgprs.epoch(), 13u);

  point.reset_owner_epoch_vgprs();
  EXPECT_FALSE(point.owner_epoch_vgprs.complete());
  EXPECT_FALSE(point.owner_epoch_vgprs.owner());
  EXPECT_FALSE(point.owner_epoch_vgprs.epoch());
}

TEST(ConSan, SiteSourcesRemainDistinctFromAcceptedOwnerEpochPair) {
  OperatingPoint accepted;
  accepted.set_owner_epoch_vgprs(12u, 13u);
  const OwnerEpochVgprSources materialized{.owner = std::nullopt, .epoch = 31u};

  EXPECT_FALSE(materialized.owner);
  EXPECT_EQ(materialized.epoch, 31u);
  EXPECT_EQ(accepted.owner_epoch_vgprs.owner(), 12u);
  EXPECT_EQ(accepted.owner_epoch_vgprs.epoch(), 13u);
}

TEST(ConSan, PersistentWorkgroupTupleIsAllOrNothing) {
  PersistentWorkgroupRegisters registers;
  EXPECT_TRUE(registers.empty());
  EXPECT_FALSE(registers.complete());
  EXPECT_EQ(registers.values(), (std::array<std::optional<uint16_t>, 4>{
                                    std::nullopt, std::nullopt, std::nullopt, std::nullopt}));

  registers = PersistentWorkgroupRegisters{12u, 13u, 14u, 15u};
  EXPECT_FALSE(registers.empty());
  EXPECT_TRUE(registers.complete());
  EXPECT_EQ(registers.values(), (std::array<std::optional<uint16_t>, 4>{12u, 13u, 14u, 15u}));

  registers.reset();
  EXPECT_TRUE(registers.empty());
  EXPECT_FALSE(registers.complete());
  EXPECT_EQ(registers.values(), (std::array<std::optional<uint16_t>, 4>{
                                    std::nullopt, std::nullopt, std::nullopt, std::nullopt}));
}

TEST(ConSan, PhysicalSiteAliasCanonicalizationRetainsOrderAndRunsTypedMerge) {
  std::vector<PhysicalAliasTestCandidate> candidates{
      {.file_offset = 24u,
       .container_name = "first",
       .semantics = 3u,
       .owners = {1u},
       .descriptor = 1u},
      {.file_offset = 8u,
       .container_name = "independent",
       .semantics = 7u,
       .owners = {9u},
       .descriptor = 9u},
      {.file_offset = 24u,
       .container_name = "alias",
       .semantics = 3u,
       .owners = {2u},
       .descriptor = 2u},
      {.file_offset = 24u,
       .container_name = "second-alias",
       .semantics = 3u,
       .owners = {3u},
       .descriptor = 3u},
  };
  std::vector<std::string> errors;

  ASSERT_TRUE(detail::canonicalize_physical_site_aliases(
      candidates, errors, "ConSan test site",
      [](const PhysicalAliasTestCandidate &candidate) { return candidate.file_offset; },
      [](const PhysicalAliasTestCandidate &candidate) -> std::string_view {
        return candidate.container_name;
      },
      [](const PhysicalAliasTestCandidate &lhs, const PhysicalAliasTestCandidate &rhs) {
        return lhs.semantics == rhs.semantics;
      },
      [](PhysicalAliasTestCandidate &retained, const PhysicalAliasTestCandidate &alias) {
        retained.owners.insert(retained.owners.end(), alias.owners.begin(), alias.owners.end());
        retained.descriptor.reset();
      }));

  EXPECT_TRUE(errors.empty());
  ASSERT_EQ(candidates.size(), 2u);
  EXPECT_EQ(candidates[0].file_offset, 24u);
  EXPECT_EQ(candidates[0].container_name, "first");
  EXPECT_EQ(candidates[0].owners, (std::vector<uint64_t>{1u, 2u, 3u}));
  EXPECT_FALSE(candidates[0].descriptor);
  EXPECT_EQ(candidates[1].file_offset, 8u);
  EXPECT_EQ(candidates[1].owners, (std::vector<uint64_t>{9u}));
  EXPECT_EQ(candidates[1].descriptor, 9u);
}

TEST(ConSan, PhysicalSiteAliasCanonicalizationReportsExactTypedConflict) {
  std::vector<PhysicalAliasTestCandidate> candidates{
      {.file_offset = 24u,
       .container_name = "first",
       .semantics = 3u,
       .owners = {},
       .descriptor = std::nullopt},
      {.file_offset = 24u,
       .container_name = "conflict",
       .semantics = 4u,
       .owners = {},
       .descriptor = std::nullopt},
  };
  const std::vector<PhysicalAliasTestCandidate> original = candidates;
  std::vector<std::string> errors;

  EXPECT_FALSE(detail::canonicalize_physical_site_aliases(
      candidates, errors, "ConSan test site",
      [](const PhysicalAliasTestCandidate &candidate) { return candidate.file_offset; },
      [](const PhysicalAliasTestCandidate &candidate) -> std::string_view {
        return candidate.container_name;
      },
      [](const PhysicalAliasTestCandidate &lhs, const PhysicalAliasTestCandidate &rhs) {
        return lhs.semantics == rhs.semantics;
      },
      [](PhysicalAliasTestCandidate &, const PhysicalAliasTestCandidate &) {}));

  ASSERT_EQ(errors.size(), 1u);
  EXPECT_EQ(errors.front(),
            "ConSan test site at file offset 24 was decoded inconsistently through aliases "
            "'first' and 'conflict'");
  EXPECT_EQ(candidates, original);
}

TEST(ConSan, IncrementalPhysicalSiteAliasConflictIsFailFastAndTransactional) {
  std::vector<std::string> errors;
  auto canonicalizer = make_physical_alias_test_canonicalizer(errors, 4u);
  ASSERT_TRUE(canonicalizer.insert({
      .file_offset = 24u,
      .container_name = "first",
      .semantics = 3u,
      .owners = {1u},
      .descriptor = std::nullopt,
  }));
  ASSERT_TRUE(canonicalizer.insert({
      .file_offset = 8u,
      .container_name = "independent",
      .semantics = 7u,
      .owners = {9u},
      .descriptor = std::nullopt,
  }));
  const std::vector<PhysicalAliasTestCandidate> accepted = canonicalizer.candidates();

  EXPECT_FALSE(canonicalizer.insert({
      .file_offset = 24u,
      .container_name = "conflict",
      .semantics = 4u,
      .owners = {2u},
      .descriptor = std::nullopt,
  }));
  EXPECT_TRUE(canonicalizer.failed());
  EXPECT_EQ(canonicalizer.candidates(), accepted);
  ASSERT_EQ(errors.size(), 1u);
  EXPECT_EQ(errors.front(),
            "ConSan incremental test site at file offset 24 was decoded inconsistently through "
            "aliases 'first' and 'conflict'");

  // A conflict is sticky: later aliases cannot mutate the accepted inventory
  // or create a cascade of secondary diagnostics.
  EXPECT_FALSE(canonicalizer.insert({
      .file_offset = 24u,
      .container_name = "later-alias",
      .semantics = 3u,
      .owners = {3u},
      .descriptor = std::nullopt,
  }));
  EXPECT_TRUE(canonicalizer.failed());
  EXPECT_EQ(canonicalizer.candidates(), accepted);
  EXPECT_EQ(errors.size(), 1u);

  // A failed inventory cannot be mistaken for a complete one.
  EXPECT_FALSE(std::move(canonicalizer).take());
  EXPECT_TRUE(canonicalizer.candidates().empty());
}

TEST(ConSan, IncrementalPhysicalSiteAliasMergeRetainsOrderAndClosesAfterTake) {
  std::vector<std::string> errors;
  auto canonicalizer = make_physical_alias_test_canonicalizer(errors, 4u);
  ASSERT_TRUE(canonicalizer.insert({
      .file_offset = 24u,
      .container_name = "first",
      .semantics = 3u,
      .owners = {1u},
      .descriptor = 1u,
  }));
  ASSERT_TRUE(canonicalizer.insert({
      .file_offset = 8u,
      .container_name = "independent",
      .semantics = 7u,
      .owners = {9u},
      .descriptor = 9u,
  }));
  ASSERT_TRUE(canonicalizer.insert({
      .file_offset = 24u,
      .container_name = "alias",
      .semantics = 3u,
      .owners = {2u},
      .descriptor = 2u,
  }));
  ASSERT_TRUE(canonicalizer.insert({
      .file_offset = 24u,
      .container_name = "second-alias",
      .semantics = 3u,
      .owners = {3u},
      .descriptor = 3u,
  }));

  EXPECT_FALSE(canonicalizer.failed());
  std::optional<std::vector<PhysicalAliasTestCandidate>> canonical =
      std::move(canonicalizer).take();
  ASSERT_TRUE(canonical);
  EXPECT_TRUE(errors.empty());
  ASSERT_EQ(canonical->size(), 2u);
  EXPECT_EQ((*canonical)[0].container_name, "first");
  EXPECT_EQ((*canonical)[0].owners, (std::vector<uint64_t>{1u, 2u, 3u}));
  EXPECT_FALSE((*canonical)[0].descriptor);
  EXPECT_EQ((*canonical)[1].container_name, "independent");
  EXPECT_EQ((*canonical)[1].owners, (std::vector<uint64_t>{9u}));
  EXPECT_EQ((*canonical)[1].descriptor, 9u);
  EXPECT_TRUE(canonicalizer.candidates().empty());

  // Extraction permanently closes the object; later use fails safely.
  EXPECT_FALSE(canonicalizer.insert({
      .file_offset = 24u,
      .container_name = "after-take",
      .semantics = 3u,
      .owners = {5u},
      .descriptor = std::nullopt,
  }));
  EXPECT_FALSE(std::move(canonicalizer).take());
  EXPECT_TRUE(canonicalizer.candidates().empty());
}

TEST(ConSan, DisabledModeDoesNotParseCodeObject) {
  const std::vector<uint8_t> bytes = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  Options options;
  options.mode = Mode::None;
  options.supercollider_delay_nops = 32;

  const auto result = test_lower_consan(bytes, options);

  EXPECT_FALSE(result.program_inventory.code_object_parsed());
  EXPECT_FALSE(result.modified());
  EXPECT_EQ(result.outcome, TransformOutcome::Unchanged);
  EXPECT_EQ(result.program_inventory.code_object_id(), make_code_object_id(bytes));
  EXPECT_TRUE(result.replacement.empty());
  EXPECT_TRUE(result.errors.empty());
  EXPECT_TRUE(result.warnings.empty());
  EXPECT_EQ(result.program_inventory.target(), ROCJITSU_CODE_TARGET_INVALID);
  EXPECT_TRUE(result.program_inventory.kernels().empty());
}

TEST(ConSan, StagedModificationStateCannotOverwriteFailureOutcome) {
  TransformArtifacts artifacts;
  EXPECT_FALSE(artifacts.modified());

  artifacts.mark_modified();
  EXPECT_TRUE(artifacts.modified());
  EXPECT_EQ(artifacts.outcome, TransformOutcome::ModifiedValid);

  for (const TransformOutcome failure :
       {TransformOutcome::Unsupported, TransformOutcome::Invalid}) {
    artifacts.outcome = failure;
    artifacts.mark_modified();
    EXPECT_FALSE(artifacts.modified());
    EXPECT_EQ(artifacts.outcome, failure);
  }

  artifacts.patches.emplace_back();
  EXPECT_TRUE(artifacts.modified());

  artifacts.replacement = {1u, 2u, 3u};
  artifacts.warnings.emplace_back("retained diagnostic");
  artifacts.outcome = TransformOutcome::Unsupported;
  artifacts.mutation.fault.planned = 2u;
  artifacts.discard_candidate_modification();
  EXPECT_TRUE(artifacts.replacement.empty());
  EXPECT_TRUE(artifacts.patches.empty());
  EXPECT_EQ(artifacts.outcome, TransformOutcome::Unsupported);
  EXPECT_EQ(artifacts.warnings, std::vector<std::string>{"retained diagnostic"});
  EXPECT_EQ(artifacts.mutation.fault.planned, 2u);
  EXPECT_FALSE(artifacts.modified());
}

TEST(ConSan, PatchedImageGrowthPolicyPreservesAbsoluteDefault) {
  PatchedImageGrowthLimit policy;
  EXPECT_EQ(policy.kind, PatchedImageGrowthLimitKind::AbsoluteBytes);
  EXPECT_EQ(policy.absolute_bytes, kDefaultMaxPatchedImageGrowthBytes);
  EXPECT_EQ(patched_image_growth_limit_bytes(policy, 17u), kDefaultMaxPatchedImageGrowthBytes);
}

TEST(ConSan, RelativePatchedImageGrowthPolicyRoundsDownWithoutOverflow) {
  PatchedImageGrowthLimit policy;
  policy.kind = PatchedImageGrowthLimitKind::InputPercent;
  policy.input_percent = 25u;
  EXPECT_EQ(patched_image_growth_limit_bytes(policy, 1003u), 250u);

  policy.input_percent = 200u;
  EXPECT_EQ(patched_image_growth_limit_bytes(policy, std::numeric_limits<size_t>::max()),
            std::numeric_limits<size_t>::max());
}

TEST(ConSan, PatchedImageGrowthPolicyRejectsInvalidKind) {
  PatchedImageGrowthLimit policy;
  policy.kind = static_cast<PatchedImageGrowthLimitKind>(255u);
  EXPECT_FALSE(patched_image_growth_limit_bytes(policy, 1003u));
}

TEST(ConSan, PatchedImageGrowthBudgetIsSharedAcrossStages) {
  PatchedImageGrowthLimit policy{
      .kind = PatchedImageGrowthLimitKind::InputPercent,
      .input_percent = 25u,
  };
  const auto budget = patched_image_growth_budget(policy, 1000u, 1100u);
  ASSERT_TRUE(budget);
  EXPECT_EQ(budget->total_limit_bytes, 250u);
  EXPECT_EQ(budget->existing_growth_bytes, 100u);
  EXPECT_EQ(budget->remaining_growth_bytes, 150u);
  EXPECT_FALSE(budget->already_exceeded);

  const auto exceeded = patched_image_growth_budget(policy, 1000u, 1300u);
  ASSERT_TRUE(exceeded);
  EXPECT_EQ(exceeded->remaining_growth_bytes, 0u);
  EXPECT_TRUE(exceeded->already_exceeded);
}

TEST(ConSan, SharedDiagnosticVocabularyUsesStableNames) {
  EXPECT_STREQ(delay_mode_name(SuperColliderDelayMode::SleepVar), "sleep_var");
  EXPECT_STREQ(barrier_operand_source_name(BarrierSite::OperandSource::StaticM0Literal32),
               "static-m0-literal32");
  EXPECT_STREQ(barrier_scope_name(BarrierSite::Scope::Workgroup), "workgroup");
}

TEST(ConSan, SynchronizationConfidenceCompositionNeverStrengthensInputs) {
  constexpr std::array confidences = {
      SemanticConfidence::Exact,
      SemanticConfidence::Conservative,
      SemanticConfidence::Ambiguous,
      SemanticConfidence::Unsupported,
  };
  for (size_t lhs = 0; lhs < confidences.size(); ++lhs) {
    for (size_t rhs = 0; rhs < confidences.size(); ++rhs) {
      const SemanticConfidence combined =
          combine_sync_confidence(confidences[lhs], confidences[rhs]);
      EXPECT_EQ(combined, confidences[std::max(lhs, rhs)]);
      EXPECT_EQ(combined, combine_sync_confidence(confidences[rhs], confidences[lhs]));
    }
  }
}

TEST(ConSan, SynchronizationConsumerContractRequiresUniqueAcceptableSequence) {
  EXPECT_TRUE(sync_confidence_meets(SemanticConfidence::Exact, SemanticConfidence::Conservative));
  EXPECT_TRUE(
      sync_confidence_meets(SemanticConfidence::Conservative, SemanticConfidence::Conservative));
  EXPECT_FALSE(
      sync_confidence_meets(SemanticConfidence::Ambiguous, SemanticConfidence::Conservative));
  EXPECT_FALSE(
      sync_confidence_meets(SemanticConfidence::Unsupported, SemanticConfidence::Ambiguous));

  TransformArtifacts result;
  ProgramInventoryBuilder inventory;
  SyncEvent event_a;
  event_a.identity = "event-a";
  event_a.semantic_id.physical.original_text_offset = 4;
  event_a.semantic_id.domain = SemanticSiteDomain::SynchronizationEvent;
  SyncEvent event_b;
  event_b.identity = "event-b";
  event_b.semantic_id.physical.original_text_offset = 8;
  event_b.semantic_id.domain = SemanticSiteDomain::SynchronizationEvent;
  inventory.synchronization().sync_events = {event_a, event_b};

  SyncSequence sequence;
  sequence.identity = "sequence-a";
  sequence.member_event_ids = {{0}, {1}};
  inventory.synchronization().sync_sequences.push_back(sequence);
  result.program_inventory = inventory.view();
  ASSERT_NE(result.program_inventory.sync().find_unique_sequence_containing("event-b"), nullptr);
  EXPECT_EQ(result.program_inventory.sync().find_unique_sequence_containing("missing"), nullptr);

  sequence.identity = "sequence-b";
  sequence.member_event_ids = {{1}};
  inventory.synchronization().sync_sequences.push_back(sequence);
  result.program_inventory = inventory.view();
  EXPECT_EQ(result.program_inventory.sync().find_unique_sequence_containing("event-b"), nullptr);
}

TEST(ConSan, EnabledModeRejectsInvalidCodeObject) {
  const std::vector<uint8_t> bytes = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  Options options;
  options.mode = Mode::SuperCollider;

  const auto result = test_lower_consan(bytes, options);

  EXPECT_FALSE(result.program_inventory.code_object_parsed());
  EXPECT_FALSE(result.modified());
  EXPECT_EQ(result.outcome, TransformOutcome::Invalid);
  EXPECT_EQ(result.program_inventory.code_object_id(), make_code_object_id(bytes));
  EXPECT_TRUE(result.replacement.empty());
  EXPECT_FALSE(result.errors.empty());
  EXPECT_EQ(result.program_inventory.target(), ROCJITSU_CODE_TARGET_INVALID);
  EXPECT_TRUE(result.program_inventory.kernels().empty());
}

TEST(ConSan, RejectsTargetsOutsideDocumentedSupport) {
  const std::array<uint32_t, 1> text_words = {build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4)};
  struct UnsupportedTarget {
    uint32_t machine;
    rj_code_target_id_t target;
  };
  constexpr std::array unsupported_targets = {
      UnsupportedTarget{EF_AMDGPU_MACH_AMDGCN_GFX90A, ROCJITSU_CODE_TARGET_GFX90A},
      UnsupportedTarget{EF_AMDGPU_MACH_AMDGCN_GFX1200, ROCJITSU_CODE_TARGET_GFX1200},
      UnsupportedTarget{0x1234u, ROCJITSU_CODE_TARGET_INVALID},
  };

  for (const UnsupportedTarget &unsupported : unsupported_targets) {
    SCOPED_TRACE(unsupported.machine);
    EXPECT_EQ(arch_for_target(unsupported.target), ROCJITSU_CODE_ARCH_INVALID);
    EXPECT_EQ(capability_disposition(unsupported.target, Mode::SuperCollider,
                                     CapabilityForm::NativeLdsAccess),
              CapabilityDisposition::OutOfContract);
    std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
    mutate_elf_header(bytes,
                      [unsupported](Elf64_Ehdr &header) { header.e_flags = unsupported.machine; });
    Options options;
    options.mode = Mode::SuperCollider;

    const TransformArtifacts result = test_lower_consan(bytes, options);

    EXPECT_EQ(result.outcome, TransformOutcome::Unsupported);
    EXPECT_TRUE(result.program_inventory.code_object_parsed());
    EXPECT_FALSE(result.modified());
    EXPECT_TRUE(result.errors.empty());
    EXPECT_FALSE(result.program_inventory.semantic_arch_required());
    EXPECT_TRUE(result.program_inventory.has_resolved_semantic_arch());
    EXPECT_EQ(result.program_inventory.target(), unsupported.target);
    EXPECT_TRUE(std::ranges::any_of(result.warnings, [&](const std::string &warning) {
      return warning == "ConSan does not support target '" +
                            std::string(rj_code_target_name(unsupported.target)) + "'";
    })) << testing::PrintToString(result.warnings);
  }
}

TEST(ConSan, ProgramInventoryOwnsSemanticArchitectureResolutionGate) {
  TransformArtifacts parse_only;
  parse_only.outcome = TransformOutcome::Unsupported;
  ProgramInventoryBuilder inventory;
  inventory.text_sections().push_back({});
  inventory.add_kernel({});
  inventory.add_function({});
  parse_only.program_inventory = inventory.view();
  EXPECT_TRUE(parse_only.program_inventory.has_resolved_semantic_arch());

  ProgramInventoryBuilder required_inventory(parse_only.program_inventory);
  required_inventory.set_semantic_arch_required(true);
  parse_only.program_inventory = required_inventory.view();
  EXPECT_FALSE(parse_only.program_inventory.has_resolved_semantic_arch());

  ProgramInventoryBuilder resolved_inventory(parse_only.program_inventory);
  resolved_inventory.set_code_object_facts(false, 0u, ROCJITSU_CODE_ARCH_RDNA4,
                                           ROCJITSU_CODE_TARGET_GFX1201);
  parse_only.program_inventory = resolved_inventory.view();
  EXPECT_TRUE(parse_only.program_inventory.has_resolved_semantic_arch());
}

TEST(ConSan, StubRejectsEmptyCodeObject) {
  const std::vector<uint8_t> bytes;
  Options options;
  options.mode = Mode::SuperCollider;

  const auto result = test_lower_consan(bytes, options);

  EXPECT_FALSE(result.program_inventory.code_object_parsed());
  EXPECT_FALSE(result.modified());
  EXPECT_EQ(result.outcome, TransformOutcome::Invalid);
  EXPECT_EQ(result.program_inventory.code_object_id(), make_code_object_id(bytes));
  EXPECT_TRUE(result.replacement.empty());
  EXPECT_FALSE(result.errors.empty());
}

TEST(ConSan, MalformedCodeObjectsNeverProduceReplacementBytes) {
  const std::array<uint32_t, 13> text_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBFB00000u,
  };
  const std::vector<uint8_t> valid = make_rdna4_lds_code_object(text_words);
  std::vector<std::pair<std::string, std::vector<uint8_t>>> cases;

  const std::array<uint32_t, 1> truncated_instruction_words = {0xD8340000u};
  cases.emplace_back("truncated eight-byte instruction",
                     make_rdna4_lds_code_object(truncated_instruction_words));

  cases.emplace_back("truncated ELF header",
                     std::vector<uint8_t>(valid.begin(), valid.begin() + sizeof(Elf64_Ehdr) - 1));
  std::vector<uint8_t> truncated_sections = valid;
  Elf64_Ehdr valid_header{};
  std::memcpy(&valid_header, valid.data(), sizeof(valid_header));
  truncated_sections.resize(valid_header.e_shoff + sizeof(Elf64_Shdr) - 1);
  cases.emplace_back("truncated section table", std::move(truncated_sections));

  std::vector<uint8_t> zero_section_count = valid;
  mutate_elf_header(zero_section_count, [](Elf64_Ehdr &header) { header.e_shnum = 0; });
  cases.emplace_back("zero section count", std::move(zero_section_count));

  std::vector<uint8_t> bad_section_offset = valid;
  mutate_elf_header(bad_section_offset,
                    [&](Elf64_Ehdr &header) { header.e_shoff = valid.size() + 64u; });
  cases.emplace_back("section table outside image", std::move(bad_section_offset));

  std::vector<uint8_t> bad_section_size = valid;
  mutate_elf_section(bad_section_size, 1,
                     [&](Elf64_Shdr &section) { section.sh_size = valid.size(); });
  cases.emplace_back("text range outside image", std::move(bad_section_size));

  std::vector<uint8_t> bad_symtab_entry_size = valid;
  mutate_elf_section(bad_symtab_entry_size, 3, [](Elf64_Shdr &section) { section.sh_entsize = 0; });
  cases.emplace_back("zero symbol entry size", std::move(bad_symtab_entry_size));

  std::vector<uint8_t> bad_symbol_section = valid;
  mutate_elf_symbol(bad_symbol_section, 1, [](Elf64_Sym &symbol) { symbol.st_shndx = 0xfffeu; });
  cases.emplace_back("kernel symbol has invalid section", std::move(bad_symbol_section));

  std::vector<uint8_t> oversized_symbol = valid;
  mutate_elf_symbol(oversized_symbol, 1, [](Elf64_Sym &symbol) { symbol.st_size = UINT64_MAX; });
  cases.emplace_back("kernel symbol range overflows", std::move(oversized_symbol));

  std::vector<uint8_t> short_descriptor = valid;
  mutate_elf_section(short_descriptor, 2, [](Elf64_Shdr &section) { section.sh_size = 4; });
  cases.emplace_back("kernel descriptor is truncated", std::move(short_descriptor));

  std::vector<uint8_t> misaligned_descriptor = valid;
  mutate_elf_section(misaligned_descriptor, 2, [](Elf64_Shdr &section) { section.sh_offset += 4; });
  cases.emplace_back("kernel descriptor has a misaligned file offset",
                     std::move(misaligned_descriptor));

  std::vector<uint8_t> bad_entry_offset = valid;
  mutate_first_kernel_descriptor(bad_entry_offset, [](KD &descriptor) {
    descriptor.kernel_code_entry_byte_offset = INT64_MAX;
  });
  cases.emplace_back("kernel descriptor entry is outside text", std::move(bad_entry_offset));

  const std::array<uint32_t, 4> function_words = {0xBF800000u, 0xBF800000u, 0xBF800000u,
                                                  0xBFB00000u};
  std::vector<uint8_t> overlapping_symbols =
      make_rdna4_code_object_with_local_function(function_words, function_words);
  mutate_elf_symbol(overlapping_symbols, 2, [](Elf64_Sym &symbol) { symbol.st_value = 0x1104u; });
  cases.emplace_back("partially overlapping function symbols", std::move(overlapping_symbols));

  for (const auto &profile : all_transform_profiles()) {
    for (const auto &[name, bytes] : cases) {
      SCOPED_TRACE(::testing::Message() << profile.name << ": " << name);
      const TransformArtifacts result = test_lower_consan(bytes, profile.options);
      EXPECT_NE(result.outcome, TransformOutcome::ModifiedValid);
      EXPECT_FALSE(result.modified());
      EXPECT_NE(result.outcome, TransformOutcome::ModifiedValid);
      EXPECT_TRUE(result.replacement.empty());
      EXPECT_TRUE(result.patches.empty());
    }
  }
}

TEST(ConSan, RejectsCodeObjectWithMalformedKernelMetadataNote) {
  const std::array<uint32_t, 4> text_words = {
      build_v_mov_b32_e32(/*vdst=*/11, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4),
      0xD8340000u,
      0x00000000u, // ds_store_b32
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  constexpr std::array<uint8_t, 3> kRequiredWorkgroupSize{2u, 4u, 8u};
  const auto make_bytes = [&] {
    std::vector<uint8_t> bytes =
        make_rdna4_lds_code_object(text_words, "malformed_kernel_metadata");
    append_kernel_metadata_note(bytes, "malformed_kernel_metadata",
                                /*uses_dynamic_stack=*/true, /*sgpr_count=*/24u,
                                /*private_segment_fixed_size=*/0u, kRequiredWorkgroupSize,
                                /*has_dynamic_lds=*/true);
    return bytes;
  };

  struct MetadataDamageCase {
    std::string_view description;
    std::vector<uint8_t> bytes;
    size_t malformed_note_count = 0;
    std::string expected_error;
  };
  std::vector<MetadataDamageCase> cases;
  auto malformed_payload = make_bytes();
  Elf64_Ehdr header{};
  std::memcpy(&header, malformed_payload.data(), sizeof(header));
  Elf64_Phdr note_segment{};
  std::memcpy(&note_segment, malformed_payload.data() + header.e_phoff, sizeof(note_segment));
  malformed_payload[note_segment.p_offset + sizeof(Elf64_Nhdr) + 8u] = 0xc1u;
  cases.push_back({.description = "payload",
                   .bytes = std::move(malformed_payload),
                   .malformed_note_count = 1u,
                   .expected_error =
                       "ConSan cannot safely transform a code object with 1 malformed AMDGPU "
                       "kernel metadata note"});

  auto malformed_framing = make_bytes();
  std::memcpy(&header, malformed_framing.data(), sizeof(header));
  std::memcpy(&note_segment, malformed_framing.data() + header.e_phoff, sizeof(note_segment));
  Elf64_Nhdr note_header{};
  std::memcpy(&note_header, malformed_framing.data() + note_segment.p_offset, sizeof(note_header));
  note_header.n_descsz += 64u;
  std::memcpy(malformed_framing.data() + note_segment.p_offset, &note_header, sizeof(note_header));
  cases.push_back({.description = "framing",
                   .bytes = std::move(malformed_framing),
                   .malformed_note_count = 1u,
                   .expected_error =
                       "ConSan cannot safely transform a code object with 1 malformed AMDGPU "
                       "kernel metadata note"});

  auto incomplete_scan = make_bytes();
  std::memcpy(&header, incomplete_scan.data(), sizeof(header));
  std::memcpy(&note_segment, incomplete_scan.data() + header.e_phoff, sizeof(note_segment));
  note_segment.p_filesz = incomplete_scan.size();
  std::memcpy(incomplete_scan.data() + header.e_phoff, &note_segment, sizeof(note_segment));
  cases.push_back(
      {.description = "out-of-range note segment",
       .bytes = std::move(incomplete_scan),
       .expected_error =
           "ConSan cannot safely transform a code object with incomplete AMDGPU kernel metadata"});

  // A note that claims metadata but cannot be read is rejected by every mode.
  for (const MetadataDamageCase &damage : cases) {
    AmdGpuCodeObject malformed(damage.bytes.data(), damage.bytes.size());
    ASSERT_TRUE(malformed.is_valid());
    ASSERT_FALSE(malformed.kernel_metadata_is_trustworthy());
    ASSERT_EQ(malformed.malformed_kernel_metadata_note_count(), damage.malformed_note_count);

    for (const auto &profile : all_transform_profiles()) {
      SCOPED_TRACE(::testing::Message() << damage.description << ": " << profile.name);
      const TransformArtifacts result = test_lower_consan(damage.bytes, profile.options);
      EXPECT_EQ(result.outcome, TransformOutcome::Invalid);
      EXPECT_FALSE(result.modified());
      EXPECT_NE(result.outcome, TransformOutcome::ModifiedValid);
      EXPECT_TRUE(result.replacement.empty());
      EXPECT_TRUE(result.patches.empty());
      EXPECT_FALSE(result.program_inventory.kernel_metadata_trustworthy());
      EXPECT_EQ(result.program_inventory.malformed_kernel_metadata_note_count(),
                damage.malformed_note_count);
      EXPECT_EQ(result.program_inventory.target(), ROCJITSU_CODE_TARGET_GFX1201);
      ASSERT_EQ(result.errors.size(), 1u);
      EXPECT_EQ(result.errors.front(), damage.expected_error);
    }
  }
}

TEST(ConSan, ReportsMultipleMalformedKernelMetadataNotes) {
  const std::array<uint32_t, 1> text_words = {
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "multiple_malformed_metadata");
  append_kernel_metadata_note(bytes, "multiple_malformed_metadata",
                              /*uses_dynamic_stack=*/true, /*sgpr_count=*/24u);

  Elf64_Ehdr header{};
  std::memcpy(&header, bytes.data(), sizeof(header));
  Elf64_Phdr note_segment{};
  std::memcpy(&note_segment, bytes.data() + header.e_phoff, sizeof(note_segment));
  const std::vector<uint8_t> duplicate = first_note_segment_bytes(bytes);
  ASSERT_FALSE(duplicate.empty());
  const uint64_t second_note_offset = note_segment.p_offset + duplicate.size();
  bytes.insert(bytes.end(), duplicate.begin(), duplicate.end());
  note_segment.p_filesz += duplicate.size();
  note_segment.p_memsz = note_segment.p_filesz;
  std::memcpy(bytes.data() + header.e_phoff, &note_segment, sizeof(note_segment));
  bytes[note_segment.p_offset + sizeof(Elf64_Nhdr) + 8u] = 0xc1u;
  bytes[second_note_offset + sizeof(Elf64_Nhdr) + 8u] = 0xc1u;

  AmdGpuCodeObject malformed(bytes.data(), bytes.size());
  ASSERT_TRUE(malformed.is_valid());
  ASSERT_EQ(malformed.malformed_kernel_metadata_note_count(), 2u);
  ASSERT_FALSE(malformed.kernel_metadata_is_trustworthy());

  Options options;
  options.mode = Mode::SuperCollider;
  const TransformArtifacts result = test_lower_consan(bytes, options);
  EXPECT_EQ(result.outcome, TransformOutcome::Invalid);
  EXPECT_FALSE(result.program_inventory.kernel_metadata_trustworthy());
  EXPECT_EQ(result.program_inventory.malformed_kernel_metadata_note_count(), 2u);
  ASSERT_EQ(result.errors.size(), 1u);
  EXPECT_NE(result.errors.front().find("2 malformed AMDGPU kernel metadata notes"),
            std::string::npos);
}

TEST(ConSan, InfersZeroSizedKernelFunctionThroughTextEnd) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBFB00000u, // s_endpgm
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_elf_symbol(bytes, 1, [](Elf64_Sym &symbol) { symbol.st_size = 0; });

  Options options;
  options.mode = Mode::SuperCollider;
  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.kernels().size(), 1u);
  EXPECT_TRUE(result.program_inventory.kernels().front().has_text_range);
  EXPECT_TRUE(result.program_inventory.kernels().front().decoded);
  EXPECT_EQ(result.program_inventory.kernels().front().code_size,
            text_words.size() * sizeof(uint32_t));
  EXPECT_EQ(result.program_inventory.kernels().front().stats.lds_write_count, 1u);
}

TEST(ConSan, UsesExplicitAliasedFunctionSizeForZeroSizedKernelSymbol) {
  const std::array<uint32_t, 3> kernel_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 3> trailing_padding = {};
  std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, trailing_padding);
  mutate_elf_symbol(bytes, 1, [](Elf64_Sym &symbol) { symbol.st_size = 0; });
  mutate_elf_symbol(bytes, 2, [](Elf64_Sym &symbol) { symbol.st_value = 0x1100u; });

  Options options;
  options.mode = Mode::SuperCollider;
  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.program_inventory.kernels().size(), 1u);
  EXPECT_TRUE(result.program_inventory.kernels().front().has_text_range);
  EXPECT_TRUE(result.program_inventory.kernels().front().decoded);
  EXPECT_EQ(result.program_inventory.kernels().front().code_size,
            kernel_words.size() * sizeof(uint32_t));
  EXPECT_FALSE(result.program_inventory.kernels().front().code_size_inferred_from_zero);
  EXPECT_EQ(result.program_inventory.kernels().front().stats.lds_write_count, 1u);
}

TEST(ConSan, ConflictingAliasedFunctionSizesFallBackToNextDistinctEntry) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.entry_nop_words = 2u;
  std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  const auto first =
      std::ranges::find(original.kernels(), "shared_owner_0", &AmdGpuKernelInfo::name);
  const auto unrelated =
      std::ranges::find(original.kernels(), "unrelated_kernel", &AmdGpuKernelInfo::name);
  ASSERT_NE(first, original.kernels().end());
  ASSERT_NE(unrelated, original.kernels().end());
  ASSERT_FALSE(original.text_sections().empty());
  const uint64_t first_entry_address =
      original.text_sections().front()->vaddr() + first->entry_text_offset;

  mutate_elf_symbol_by_name(bytes, "shared_owner_0",
                            [](Elf64_Sym &symbol) { symbol.st_size = 0u; });
  mutate_elf_symbol_by_name(bytes, "shared_owner_1", [&](Elf64_Sym &symbol) {
    symbol.st_value = first_entry_address;
    symbol.st_size = first->code_size;
  });
  mutate_elf_symbol_by_name(bytes, "shared_lds_helper", [&](Elf64_Sym &symbol) {
    symbol.st_value = first_entry_address;
    symbol.st_size = first->code_size + sizeof(uint32_t);
  });

  AmdGpuCodeObject conflicted(bytes.data(), bytes.size());
  ASSERT_TRUE(conflicted.is_valid());
  const auto inferred =
      std::ranges::find(conflicted.kernels(), "shared_owner_0", &AmdGpuKernelInfo::name);
  const auto next =
      std::ranges::find(conflicted.kernels(), "unrelated_kernel", &AmdGpuKernelInfo::name);
  ASSERT_NE(inferred, conflicted.kernels().end());
  ASSERT_NE(next, conflicted.kernels().end());
  EXPECT_TRUE(inferred->code_size_inferred_from_zero);
  EXPECT_EQ(inferred->code_size, next->entry_text_offset - inferred->entry_text_offset);
}

TEST(ConSan, ElfFixtureAcceptsEmptyInstructionSpans) {
  const std::array<uint32_t, 1> end = {0xBFB00000u};
  for (const auto kernel : {std::span<const uint32_t>{}, std::span<const uint32_t>{end}}) {
    const auto bytes = make_rdna4_code_object_with_local_function(kernel, {}, {});
    AmdGpuCodeObject object(bytes.data(), bytes.size());
    ASSERT_TRUE(object.is_valid());
    ASSERT_EQ(object.text_sections().size(), 1u);
    EXPECT_EQ(object.text_sections().front()->size(), kernel.size_bytes());
  }
}

TEST(ConSan, SkipsEmptyTargetSelectionKernelAtTextEnd) {
  const std::array<uint32_t, 3> kernel_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 0> empty_specialization = {};
  const std::vector<uint8_t> bytes = make_rdna4_code_object_with_local_function(
      kernel_words, empty_specialization, {}, kRdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);

  Options options;
  options.mode = Mode::SuperCollider;
  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.program_inventory.kernels().size(), 2u);
  const auto empty =
      std::ranges::find(result.program_inventory.kernels(), "lds_helper", &ProgramContainer::name);
  ASSERT_NE(empty, result.program_inventory.kernels().end());
  EXPECT_TRUE(empty->has_text_range);
  EXPECT_EQ(empty->code_size, 0u);
  EXPECT_TRUE(empty->decoded);
  EXPECT_EQ(empty->preflight_action, PreflightAction::Skip);
  EXPECT_EQ(empty->stats.instruction_count, 0u);
}

TEST(ConSan, ExcessiveAllocatedSectionAlignmentCannotDriveTextGrowthAllocation) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBFB00000u,
  };
  const std::vector<uint8_t> valid = make_rdna4_lds_code_object(text_words);
  std::vector<std::pair<std::string, std::vector<uint8_t>>> cases;

  std::vector<uint8_t> excessive_alignment = valid;
  mutate_elf_section(excessive_alignment, 2,
                     [](Elf64_Shdr &section) { section.sh_addralign = uint64_t{1} << 58u; });
  cases.emplace_back("excessive allocated-section alignment", std::move(excessive_alignment));

  std::vector<uint8_t> overflowing_program_headers = valid;
  mutate_elf_header(overflowing_program_headers, [](Elf64_Ehdr &header) {
    header.e_phoff = UINT64_MAX;
    header.e_phentsize = sizeof(Elf64_Phdr);
    header.e_phnum = 1;
  });
  cases.emplace_back("overflowing program-header table", std::move(overflowing_program_headers));

  for (const auto &profile : all_replacement_profiles()) {
    for (const auto &[name, bytes] : cases) {
      SCOPED_TRACE(::testing::Message() << profile.name << ": " << name);
      const TransformArtifacts result = test_lower_consan(bytes, profile.options);
      EXPECT_NE(result.outcome, TransformOutcome::ModifiedValid);
      EXPECT_FALSE(result.modified());
      EXPECT_NE(result.outcome, TransformOutcome::ModifiedValid);
      EXPECT_TRUE(result.replacement.empty());
      EXPECT_TRUE(result.patches.empty());
    }
  }
}

TEST(ConSan, RelocationRejectsAlignmentGrowthBeforeAllocation) {
  const std::array<uint32_t, 3> text_words = {0xD8340000u, 0x00000102u, 0xBFB00000u};
  auto bytes = make_rdna4_lds_code_object(text_words);
  mutate_elf_section(bytes, 2,
                     [](Elf64_Shdr &section) { section.sh_addralign = uint64_t{1} << 58u; });
  for (const auto &profile : all_replacement_profiles()) {
    SCOPED_TRACE(profile.name);
    const auto result = test_lower_consan(bytes, profile.options);
    EXPECT_FALSE(result.modified());
    EXPECT_TRUE(result.replacement.empty());
    EXPECT_TRUE(result.patches.empty());
    EXPECT_EQ(result.transform_failure_cause, TransformFailureCause::PatchedImageGrowthLimit);
  }
}

TEST(ConSan, BoundedElfMutationsOnlyProduceValidatedReplacementOrOriginal) {
  // The patched-image growth budget must reject oversized ELF insertions before
  // allocation, including under ASan, whose shadow mapping precludes RLIMIT_AS.

  const std::array<uint32_t, 13> text_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBFB00000u,
  };
  const std::vector<uint8_t> valid = make_rdna4_lds_code_object(text_words);
  auto expect_transactional_result = [&](std::span<const uint8_t> input,
                                         const TransformProfile &profile) {
    const TransformArtifacts result = test_lower_consan(input, profile.options);
    EXPECT_EQ(result.program_inventory.code_object_id(), make_code_object_id(input));
    if (result.modified()) {
      EXPECT_TRUE(result.modified());
      EXPECT_EQ(result.outcome, TransformOutcome::ModifiedValid);
      EXPECT_FALSE(result.replacement.empty());
      EXPECT_FALSE(result.patches.empty());
      EXPECT_TRUE(validate_modified_elf(input, result).empty());
    } else {
      EXPECT_FALSE(result.modified());
      EXPECT_NE(result.outcome, TransformOutcome::ModifiedValid);
      EXPECT_TRUE(result.replacement.empty());
      EXPECT_TRUE(result.patches.empty());
    }
  };

  for (const auto &profile : all_transform_profiles()) {
    for (size_t size = 0; size <= valid.size(); ++size) {
      SCOPED_TRACE(::testing::Message() << profile.name << ": truncation size " << size);
      expect_transactional_result(std::span<const uint8_t>(valid.data(), size), profile);
    }
    for (size_t offset = 0; offset < valid.size(); ++offset) {
      SCOPED_TRACE(::testing::Message()
                   << profile.name << ": single-byte mutation offset " << offset);
      std::vector<uint8_t> mutated = valid;
      mutated[offset] ^= static_cast<uint8_t>(0xA5u ^ (offset & 0xFFu));
      expect_transactional_result(mutated, profile);
    }
  }
}

TEST(ConSan, FinalStructuralValidationRediscoversReplacementIdentity) {
  const std::array<uint32_t, 13> text_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  for (const auto &profile : all_replacement_profiles()) {
    SCOPED_TRACE(profile.name);
    const TransformArtifacts valid = test_lower_consan(bytes, profile.options);
    ASSERT_EQ(valid.outcome, TransformOutcome::ModifiedValid);
    EXPECT_EQ(valid.outcome, TransformOutcome::ModifiedValid);
    EXPECT_TRUE(validate_modified_elf(bytes, valid).empty());

    TransformArtifacts wrong_target = valid;
    mutate_elf_header(wrong_target.replacement, [](Elf64_Ehdr &header) { header.e_flags = 0; });
    EXPECT_FALSE(validate_modified_elf(bytes, wrong_target).empty());

    TransformArtifacts stale_text = valid;
    mutate_elf_section(stale_text.replacement, 1, [&](Elf64_Shdr &section) {
      section.sh_size = stale_text.replacement.size();
    });
    EXPECT_FALSE(validate_modified_elf(bytes, stale_text).empty());

    TransformArtifacts unaccounted_change = valid;
    unaccounted_change.replacement[0x100u + 48u] ^= 1u;
    const std::vector<std::string> unaccounted_errors =
        validate_modified_elf(bytes, unaccounted_change);
    ASSERT_FALSE(unaccounted_errors.empty());
    EXPECT_TRUE(std::ranges::any_of(unaccounted_errors, [](const std::string &error) {
      return error.find("unaccounted executable byte change") != std::string::npos;
    }));

    TransformArtifacts overlapping_inventory = valid;
    PatchInfo overlap = overlapping_inventory.patches.front();
    overlap.anchor_offset += sizeof(uint32_t);
    overlapping_inventory.patches.push_back(overlap);
    const std::vector<std::string> overlap_errors =
        validate_modified_elf(bytes, overlapping_inventory);
    ASSERT_FALSE(overlap_errors.empty());
    EXPECT_TRUE(std::ranges::any_of(overlap_errors, [](const std::string &error) {
      return error.find("partially overlapping patch ranges") != std::string::npos;
    }));

    TransformArtifacts stale_inventory = valid;
    stale_inventory.patches.front().anchor_offset = UINT64_MAX;
    const std::vector<std::string> stale_inventory_errors =
        validate_modified_elf(bytes, stale_inventory);
    ASSERT_FALSE(stale_inventory_errors.empty());
    EXPECT_TRUE(std::ranges::any_of(stale_inventory_errors, [](const std::string &error) {
      return error.find("stale or unaligned patch range") != std::string::npos;
    }));

    TransformArtifacts undecodable_patch = valid;
    const uint32_t invalid_instruction = 0xffffffffu;
    std::memcpy(undecodable_patch.replacement.data() + 0x100u + valid.patches.front().anchor_offset,
                &invalid_instruction, sizeof(invalid_instruction));
    const std::vector<std::string> decode_errors = validate_modified_elf(bytes, undecodable_patch);
    ASSERT_FALSE(decode_errors.empty());
    EXPECT_TRUE(std::ranges::any_of(decode_errors, [](const std::string &error) {
      return error.find("re-decode patch anchor") != std::string::npos;
    }));
  }
}

TEST(ConSan, FinalValidationScalesAcrossManyDisjointPatchRanges) {
  constexpr size_t kPatchCount = 8192u;
  std::vector<uint32_t> text_words(kPatchCount + 1u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "many_disjoint_patch_ranges");

  TransformArtifacts result;
  result.mark_modified();
  result.replacement = bytes;
  AmdGpuCodeObject replacement(result.replacement.data(), result.replacement.size());
  ASSERT_EQ(replacement.text_sections().size(), 1u);
  const uint64_t text_file_offset = replacement.text_sections().front()->sectionOffset();
  const uint32_t replacement_word = build_s_nop(1, ROCJITSU_CODE_ARCH_RDNA4);
  result.patches.reserve(kPatchCount);
  for (size_t index = 0; index < kPatchCount; ++index) {
    const uint64_t anchor = index * sizeof(uint32_t);
    std::memcpy(result.replacement.data() + text_file_offset + anchor, &replacement_word,
                sizeof(replacement_word));
    PatchInfo patch;
    patch.kind = PatchKind::InlineBarrierNopRewrite;
    patch.anchor_offset = anchor;
    patch.original_size = sizeof(uint32_t);
    result.patches.push_back(std::move(patch));
  }

  EXPECT_TRUE(validate_modified_elf(bytes, result).empty());
}

} // namespace
} // namespace rocjitsu::consan
