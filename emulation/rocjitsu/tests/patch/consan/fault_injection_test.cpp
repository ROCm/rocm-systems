// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/consan/consan_fault_selection.h"
#include "rocjitsu/code/patch/consan/targets/consan_fault_target_ops.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"

namespace rocjitsu::consan {
namespace {

TEST(ConSan, Rdna4Cdna5AtomicFaultTargetOperationsOwnRawAddressAndScopeRewrites) {
  const auto bytes_of = []<typename Raw>(Raw &raw) {
    return std::span<uint8_t>(reinterpret_cast<uint8_t *>(&raw), sizeof(raw));
  };

  rdna4::VflatMachineInst flat{};
  flat.ioffset = 0xfffffcu; // -4
  flat.scope = 3u;
  auto rewrite =
      rewrite_atomic_fault_address(bytes_of(flat), AtomicFaultEncoding::FlatLike, 32u, 8u);
  ASSERT_TRUE(rewrite.rewritten());
  EXPECT_EQ(flat.ioffset, 4u);
  rewrite = rewrite_atomic_fault_scope_to_wave(bytes_of(flat), AtomicFaultEncoding::FlatLike);
  ASSERT_TRUE(rewrite.rewritten());
  EXPECT_EQ(rewrite.previous_value, 3u);
  EXPECT_EQ(flat.scope, 0u);
  EXPECT_EQ(
      rewrite_atomic_fault_scope_to_wave(bytes_of(flat), AtomicFaultEncoding::FlatLike).status,
      AtomicFaultRewriteStatus::AlreadyWaveScope);

  rdna4::VbufferMachineInst buffer{};
  buffer.ioffset = 0x7ffffcu;
  buffer.scope = 2u;
  const rdna4::VbufferMachineInst original_buffer = buffer;
  rewrite = rewrite_atomic_fault_address(bytes_of(buffer), AtomicFaultEncoding::Buffer, 32u, 4u);
  EXPECT_EQ(rewrite.status, AtomicFaultRewriteStatus::OffsetOverflow);
  EXPECT_EQ(std::memcmp(&buffer, &original_buffer, sizeof(buffer)), 0);

  rdna4::VdsMachineInst ds{};
  ds.offset0 = 4u;
  rewrite = rewrite_atomic_fault_address(bytes_of(ds), AtomicFaultEncoding::Ds, 32u, 4u);
  ASSERT_TRUE(rewrite.rewritten());
  EXPECT_EQ(ds.offset0, 8u);
  ds.offset0 = 0u;
  rewrite = rewrite_atomic_fault_address(bytes_of(ds), AtomicFaultEncoding::Ds, 64u, 4u);
  EXPECT_EQ(rewrite.status, AtomicFaultRewriteStatus::MisalignedOffset);
  EXPECT_EQ(ds.offset0, 0u);
  ds.offset0 = 252u;
  rewrite = rewrite_atomic_fault_address(bytes_of(ds), AtomicFaultEncoding::Ds, 32u, 4u);
  EXPECT_EQ(rewrite.status, AtomicFaultRewriteStatus::OffsetOverflow);
  EXPECT_EQ(ds.offset0, 252u);

  EXPECT_EQ(
      rewrite_atomic_fault_address(bytes_of(flat), AtomicFaultEncoding::CdnaFlat, 32u, 4u).status,
      AtomicFaultRewriteStatus::InvalidEncoding);
}

TEST(ConSan, ExactBarrierDropIssuesAreTypedAndRenderEstablishedDiagnostics) {
  using PairIssue = ExactBarrierDropPairIssue;
  const std::array pair_messages = {
      std::pair{PairIssue::None, std::string_view{""}},
      std::pair{PairIssue::MissingExactIdentity,
                std::string_view{
                    "an exact site identity and logical sequence identity are both required"}},
      std::pair{PairIssue::SequenceNotFound,
                std::string_view{"the exact logical sequence identity was not found"}},
      std::pair{PairIssue::SequenceNotQualified,
                std::string_view{
                    "the exact sequence is not an owned complete conservative two-member barrier"}},
      std::pair{
          PairIssue::PrimaryNotMember,
          std::string_view{"the exact site is not a member of the requested logical barrier"}},
      std::pair{PairIssue::MemberSiteMissing,
                std::string_view{"a logical barrier member has no exact owned patch site"}},
      std::pair{PairIssue::MemberSiteAmbiguous,
                std::string_view{"a logical barrier member maps to multiple patch sites"}},
      std::pair{PairIssue::InvalidPairGeometry,
                std::string_view{
                    "the exact logical barrier does not have two distinct one-word patch sites"}},
  };
  static_assert(pair_messages.size() == static_cast<size_t>(PairIssue::Count));
  for (size_t index = 0; index < pair_messages.size(); ++index) {
    EXPECT_EQ(static_cast<size_t>(pair_messages[index].first), index);
    EXPECT_EQ(exact_barrier_drop_pair_issue_message(pair_messages[index].first),
              pair_messages[index].second);
  }
  EXPECT_EQ(exact_barrier_drop_pair_issue_message(PairIssue::Count),
            "invalid exact barrier-drop pair issue");
  EXPECT_EQ(exact_barrier_drop_pair_issue_message(static_cast<PairIssue>(255u)),
            "invalid exact barrier-drop pair issue");

  using GroupIssue = ExactBarrierDropGroupIssue;
  struct ExpectedGroupMessage {
    GroupIssue issue;
    PairIssue member_issue;
    std::string_view message;
  };
  const std::array group_messages = {
      ExpectedGroupMessage{GroupIssue::None, PairIssue::None, ""},
      ExpectedGroupMessage{GroupIssue::MissingCompanionIdentity, PairIssue::None,
                           "a grouped drop requires an exact companion site and sequence identity"},
      ExpectedGroupMessage{
          GroupIssue::FirstPairRejected, PairIssue::MemberSiteMissing,
          "group member zero is invalid: a logical barrier member has no exact owned patch site"},
      ExpectedGroupMessage{
          GroupIssue::SecondPairRejected, PairIssue::MemberSiteAmbiguous,
          "group member one is invalid: a logical barrier member maps to multiple patch sites"},
      ExpectedGroupMessage{
          GroupIssue::PairsOverlapOrUnordered, PairIssue::None,
          "the two exact pairs are duplicate, overlapping, or not strictly ordered"},
      ExpectedGroupMessage{
          GroupIssue::PairsHaveDifferentOwners, PairIssue::None,
          "the two exact pairs do not have identical container and execution owners"},
  };
  static_assert(group_messages.size() == static_cast<size_t>(GroupIssue::Count));
  for (size_t index = 0; index < group_messages.size(); ++index) {
    const auto &[issue, member_issue, expected] = group_messages[index];
    EXPECT_EQ(static_cast<size_t>(issue), index);
    EXPECT_EQ(exact_barrier_drop_group_issue_message(issue, member_issue), expected);
  }
  EXPECT_EQ(exact_barrier_drop_group_issue_message(GroupIssue::Count),
            "invalid exact barrier-drop group issue");
  EXPECT_EQ(exact_barrier_drop_group_issue_message(static_cast<GroupIssue>(255u)),
            "invalid exact barrier-drop group issue");
}

std::vector<uint8_t> make_lds_address_fault_code_object(rj_code_arch_t arch) {
  const auto access =
      instrumentation::build_ds_store_b32(/*vaddr=*/2u, /*vdata=*/3u, /*byte_offset=*/4u, arch);
  if (!access)
    return {};
  std::vector<uint32_t> words(access->begin(), access->end());
  words.push_back(build_s_endpgm(arch));
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return make_cdna3_lds_code_object(words, "lds_address_fault");
  case ROCJITSU_CODE_ARCH_CDNA4:
    return make_cdna4_lds_code_object(words, "lds_address_fault");
  case ROCJITSU_CODE_ARCH_RDNA4:
    return make_rdna4_lds_code_object(words, "lds_address_fault");
  case ROCJITSU_CODE_ARCH_CDNA5:
    return make_gfx1250_code_object(words, "lds_address_fault");
  default:
    return {};
  }
}

TEST(ConSan, FaultApplicationCommitsOneCandidateAcrossIndependentMechanisms) {
  const auto lds_store = instrumentation::build_ds_store_b32(
      /*vaddr=*/2u, /*vdata=*/3u, /*byte_offset=*/4u, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(lds_store.has_value());
  const auto atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/6u, /*vsrc=*/8u, /*vdst=*/4u, /*return_old_value=*/true,
      /*scope=*/2u, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic.has_value());
  std::vector<uint32_t> words(lds_store->begin(), lds_store->end());
  words.insert(words.end(), atomic->begin(), atomic->end());
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(words, "independent_fault_transaction");

  Options options;
  options.mode = Mode::SuperCollider;
  options.fault_lds_wrong_address = true;
  options.fault_lds_address_vgpr = 12u;
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_address_delta = 4u;

  const TransformArtifacts result = test_lower_consan(bytes, options);
  ASSERT_EQ(result.outcome, TransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors);
  EXPECT_EQ(result.mutation.fault.requested, 2u);
  EXPECT_EQ(result.mutation.fault.planned, 2u);
  EXPECT_EQ(result.mutation.fault.applied, 2u);
  EXPECT_EQ(
      std::ranges::count(result.patches, PatchKind::InlineAtomicAddressRewrite, &PatchInfo::kind),
      1u);
  EXPECT_EQ(
      std::ranges::count(result.patches, PatchKind::InlineLdsAddressRewrite, &PatchInfo::kind), 1u);
}

TEST(ConSan, FaultInventoryProvesDirectSharedHelperOwnersAndFiltersExactDispatch) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_ordered_atomic = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  Options inventory_options;
  inventory_options.mode = Mode::Default;
  inventory_options.fault_dry_run = true;
  const TransformArtifacts inventory = test_lower_consan(bytes, inventory_options);
  const auto site = std::ranges::find_if(inventory.fault_sites, [&](const FaultSite &item) {
    const ProgramSite *source = test_fault_source(inventory, item);
    return item.kind == FaultSiteKind::Atomic && source != nullptr &&
           test_program_container_name(inventory, *source) == "shared_lds_helper";
  });
  ASSERT_NE(site, inventory.fault_sites.end());
  const FaultSiteDiagnostic diagnostic = test_fault_diagnostic(inventory, *site);
  ASSERT_EQ(diagnostic.execution_owners.size(), 2u);
  for (const ExecutionOwner &owner : diagnostic.execution_owners)
    EXPECT_EQ(owner.proof, OwnerProofKind::DirectCall);

  for (std::string_view owner_name : {"shared_owner_0", "shared_owner_1"}) {
    Options options = inventory_options;
    options.fault_atomic_wrong_address = true;
    options.fault_site_identity = site->identity;
    options.test_kernel_name_filter = owner_name;
    const TransformArtifacts selected = test_lower_consan(bytes, options);
    ASSERT_EQ(selected.fault_plans.size(), 1u) << testing::PrintToString(selected.warnings);
    EXPECT_EQ(selected.fault_plans.front().primary_identity, site->identity);
  }

  for (std::string_view rejected_filter : {"unrelated_kernel", "shared_owner_"}) {
    Options options = inventory_options;
    options.fault_atomic_wrong_address = true;
    options.fault_site_identity = site->identity;
    options.test_kernel_name_filter = rejected_filter;
    const TransformArtifacts rejected = test_lower_consan(bytes, options);
    EXPECT_TRUE(rejected.fault_plans.empty());
  }
}

TEST(ConSan, FaultInventoryProvesRecoveredIndirectSharedHelperOwners) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_ordered_atomic = true;
  fixture.use_indirect_calls = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  Options options = test_options();
  options.fault_dry_run = true;
  const TransformArtifacts result = test_lower_consan(bytes, options);
  const auto site = std::ranges::find_if(result.fault_sites, [&](const FaultSite &item) {
    const ProgramSite *source = test_fault_source(result, item);
    return item.kind == FaultSiteKind::Atomic && source != nullptr &&
           test_program_container_name(result, *source) == "shared_lds_helper";
  });
  ASSERT_NE(site, result.fault_sites.end());
  const FaultSiteDiagnostic diagnostic = test_fault_diagnostic(result, *site);
  ASSERT_EQ(diagnostic.execution_owners.size(), 2u);
  for (const ExecutionOwner &owner : diagnostic.execution_owners)
    EXPECT_EQ(owner.proof, OwnerProofKind::RecoveredIndirectCall);
}

TEST(ConSan, FaultInventoryMarksKernelLocalOwner) {
  const std::vector<uint8_t> bytes = make_rdna4_global_atomic_code_object();
  Options options = test_options();
  options.fault_dry_run = true;
  const TransformArtifacts result = test_lower_consan(bytes, options);
  ASSERT_EQ(result.fault_sites.size(), 1u);
  const FaultSiteDiagnostic diagnostic = test_fault_diagnostic(result, result.fault_sites.front());
  ASSERT_EQ(diagnostic.execution_owners.size(), 1u);
  EXPECT_EQ(diagnostic.execution_owners.front().proof, OwnerProofKind::KernelLocal);
}

TEST(ConSan, FaultLoadSelectorSelectsOneStableOneBasedOccurrence) {
  FaultLoadSelector selector(/*requested_occurrence=*/2);
  const FaultLoadSelection first = selector.observe();
  const FaultLoadSelection second = selector.observe();
  const FaultLoadSelection third = selector.observe();
  EXPECT_EQ(first.occurrence, 1u);
  EXPECT_FALSE(first.selected);
  EXPECT_EQ(second.occurrence, 2u);
  EXPECT_TRUE(second.selected);
  EXPECT_EQ(third.occurrence, 3u);
  EXPECT_FALSE(third.selected);
  EXPECT_EQ(selector.observed(), 3u);
  EXPECT_EQ(selector.selected(), 1u);
  EXPECT_TRUE(selector.accepted());
}

TEST(ConSan, FaultLoadSelectorFailsClosedWhenOccurrenceIsAbsent) {
  FaultLoadSelector selector(/*requested_occurrence=*/2);
  EXPECT_FALSE(selector.accepted());
  EXPECT_FALSE(selector.observe().selected);
  EXPECT_FALSE(selector.accepted());
}

TEST(ConSan, FaultMutationCardinalityReportsZeroAndEnforcesGuard) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_atomic_release_acquire_code_object();
  Options options;
  options.mode = Mode::SuperCollider;
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_index = 99;

  const TransformArtifacts unguarded = test_lower_consan(bytes, options);
  EXPECT_EQ(unguarded.outcome, TransformOutcome::Unchanged);
  EXPECT_EQ(unguarded.mutation.fault.requested, 1u);
  EXPECT_EQ(unguarded.mutation.fault.applied, 0u);

  options.fault_require_exactly_one = true;
  const TransformArtifacts guarded = test_lower_consan(bytes, options);
  EXPECT_EQ(guarded.outcome, TransformOutcome::Invalid);
  EXPECT_EQ(guarded.mutation.fault.requested, 1u);
  EXPECT_EQ(guarded.mutation.fault.applied, 0u);
  EXPECT_TRUE(std::ranges::any_of(guarded.errors, [](const std::string &error) {
    return error.find("required exactly one applied mutation, got 0") != std::string::npos;
  }));
}

TEST(ConSan, FaultMutationCardinalityExcludesRetiredThOrderMutation) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_release_acquire_code_object();
  Options options = test_options();
  options.track_atomics = true;
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(32);
  options.max_patches = 8;
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_weaken_order = true;
  options.fault_atomic_index = 1;

  const TransformArtifacts unguarded = test_lower_consan(bytes, options);
  EXPECT_EQ(unguarded.outcome, TransformOutcome::ModifiedValid);
  EXPECT_EQ(unguarded.mutation.fault.requested, 2u);
  EXPECT_EQ(unguarded.mutation.fault.applied, 2u);
  EXPECT_TRUE(std::ranges::any_of(unguarded.warnings, [](const std::string &warning) {
    return warning.find("removed associated global_inv") != std::string::npos;
  }));

  options.fault_require_exactly_one = true;
  const TransformArtifacts guarded = test_lower_consan(bytes, options);
  EXPECT_EQ(guarded.outcome, TransformOutcome::Invalid);
  EXPECT_EQ(guarded.mutation.fault.requested, 2u);
  EXPECT_EQ(guarded.mutation.fault.applied, 2u);
}

TEST(ConSan, LdsAddressFaultInventoryAndExactMutationAreTargetNeutral) {
  constexpr std::array targets = {
      ROCJITSU_CODE_ARCH_CDNA3,
      ROCJITSU_CODE_ARCH_CDNA4,
      ROCJITSU_CODE_ARCH_RDNA4,
      ROCJITSU_CODE_ARCH_CDNA5,
  };
  for (rj_code_arch_t arch : targets) {
    SCOPED_TRACE(static_cast<uint32_t>(arch));
    const std::vector<uint8_t> bytes = make_lds_address_fault_code_object(arch);
    ASSERT_FALSE(bytes.empty());

    Options inventory_options;
    inventory_options.mode = Mode::SuperCollider;
    inventory_options.probe_lds_check_trap = true;
    inventory_options.scratch_vgpr = 8u;
    inventory_options.supercollider_delay_nops = 1u;
    inventory_options.fault_lds_wrong_address = true;
    inventory_options.fault_lds_address_vgpr = 6u;
    inventory_options.fault_dry_run = true;
    const TransformArtifacts inventory = test_lower_consan(bytes, inventory_options);
    const auto site =
        std::ranges::find(inventory.fault_sites, FaultSiteKind::LdsAccess, &FaultSite::kind);
    ASSERT_NE(site, inventory.fault_sites.end());
    const ProgramSite *source = test_fault_source(inventory, *site);
    ASSERT_NE(source, nullptr);
    EXPECT_EQ(source->operands.address_vgpr, 2u);
    const FaultSiteDiagnostic diagnostic = test_fault_diagnostic(inventory, *site);
    EXPECT_EQ(diagnostic.semantic_role, "lds-write");
    ASSERT_EQ(diagnostic.execution_owners.size(), 1u);
    ASSERT_EQ(inventory.fault_plans.size(), 1u);
    EXPECT_EQ(inventory.fault_plans.front().kind, FaultMutationKind::LdsWrongAddress);
    EXPECT_EQ(inventory.fault_plans.front().primary_identity, site->identity);

    Options live_options = inventory_options;
    live_options.fault_dry_run = false;
    live_options.fault_require_exactly_one = true;
    live_options.fault_site_identity = site->identity;
    const TransformArtifacts result = test_lower_consan(bytes, live_options);
    ASSERT_EQ(result.outcome, TransformOutcome::ModifiedValid)
        << testing::PrintToString(result.errors) << testing::PrintToString(result.warnings);
    EXPECT_EQ(result.mutation.fault.requested, 1u);
    EXPECT_EQ(result.mutation.fault.planned, 1u);
    EXPECT_EQ(result.mutation.fault.applied, 1u);
    EXPECT_EQ(result.mutation.applied_fault_logical_identity, site->identity);
    const auto mutation = std::ranges::find_if(result.patches, [](const PatchInfo &patch) {
      return patch.phase == PatchPhase::Mutation &&
             patch.kind == PatchKind::InlineLdsAddressRewrite;
    });
    ASSERT_NE(mutation, result.patches.end());
    EXPECT_EQ(mutation->fault_primary_identity, site->identity);
    EXPECT_EQ(mutation->fault_original_address_vgpr, 2u);
    EXPECT_EQ(mutation->fault_target_address_vgpr, 6u);
    EXPECT_TRUE(validate_modified_elf(bytes, result).empty());
  }
}

TEST(ConSan, LdsAddressFaultRejectsSameRegisterAndFinalProofRejectsOtherFieldDrift) {
  const std::vector<uint8_t> bytes = make_lds_address_fault_code_object(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_FALSE(bytes.empty());
  Options options;
  options.mode = Mode::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 8u;
  options.supercollider_delay_nops = 1u;
  options.fault_lds_wrong_address = true;
  options.fault_lds_address_vgpr = 2u;
  options.fault_require_exactly_one = true;
  const TransformArtifacts rejected = test_lower_consan(bytes, options);
  EXPECT_EQ(rejected.outcome, TransformOutcome::Invalid);
  EXPECT_EQ(rejected.mutation.fault.applied, 0u);
  EXPECT_TRUE(std::ranges::any_of(rejected.errors, [](const std::string &error) {
    return error.find("requires a distinct 8-bit replacement VGPR") != std::string::npos;
  }));

  options.fault_lds_address_vgpr = 6u;
  const TransformArtifacts valid = test_lower_consan(bytes, options);
  ASSERT_EQ(valid.outcome, TransformOutcome::ModifiedValid) << testing::PrintToString(valid.errors);
  ASSERT_EQ(valid.program_inventory.text_sections().size(), 1u);
  const auto mutation = std::ranges::find_if(valid.patches, [](const PatchInfo &patch) {
    return patch.phase == PatchPhase::Mutation && patch.kind == PatchKind::InlineLdsAddressRewrite;
  });
  ASSERT_NE(mutation, valid.patches.end());
  const auto instrumentation_patch =
      std::ranges::find_if(valid.patches, [&](const PatchInfo &patch) {
        return patch.phase == PatchPhase::Instrumentation &&
               patch.anchor_offset == mutation->anchor_offset &&
               patch.relocated_guest_instruction_offset.has_value();
      });
  ASSERT_NE(instrumentation_patch, valid.patches.end());

  TransformArtifacts corrupted = valid;
  const uint64_t word1_file_offset = valid.program_inventory.text_sections().front().file_offset +
                                     *instrumentation_patch->relocated_guest_instruction_offset +
                                     sizeof(uint32_t);
  uint32_t word1 = 0;
  std::memcpy(&word1, corrupted.replacement.data() + word1_file_offset, sizeof(word1));
  word1 ^= 1u << 8u;
  std::memcpy(corrupted.replacement.data() + word1_file_offset, &word1, sizeof(word1));
  const std::vector<std::string> errors = validate_modified_elf(bytes, corrupted);
  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("fields other than the selected LDS address VGPR") != std::string::npos;
  })) << testing::PrintToString(errors);
}

TEST(ConSan, LdsAddressFaultRequiresExplicitAllocatedReplacement) {
  const auto access = instrumentation::build_ds_store_b32(
      /*vaddr=*/2u, /*vdata=*/3u, /*byte_offset=*/4u, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(access);
  std::vector<uint32_t> words(access->begin(), access->end());
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(words, "lds_address_allocation", /*vgpr_granulated=*/0u);
  Options options;
  options.mode = Mode::SuperCollider;
  options.fault_lds_wrong_address = true;
  options.fault_require_exactly_one = true;

  const TransformArtifacts missing = test_lower_consan(bytes, options);
  EXPECT_EQ(missing.outcome, TransformOutcome::Invalid);
  EXPECT_TRUE(std::ranges::any_of(missing.errors, [](const std::string &error) {
    return error.find("requires an explicit replacement VGPR") != std::string::npos;
  })) << testing::PrintToString(missing.errors);

  // A zero-granulated RDNA4 wave64 descriptor allocates v0..v3. The boundary
  // register v4 must be rejected rather than read as undefined input.
  options.fault_lds_address_vgpr = 4u;
  const TransformArtifacts outside_allocation = test_lower_consan(bytes, options);
  EXPECT_EQ(outside_allocation.outcome, TransformOutcome::Invalid);
  EXPECT_EQ(outside_allocation.mutation.fault.applied, 0u);
  EXPECT_TRUE(std::ranges::any_of(outside_allocation.errors, [](const std::string &error) {
    return error.find("outside an execution owner's allocated VGPR window") != std::string::npos;
  })) << testing::PrintToString(outside_allocation.errors);
}

TEST(ConSan, Gfx1250TwoAddressLdsAccessIsNotAdvertisedForExactAddressRewrite) {
  constexpr auto load = cdna5::build_vds(cdna5::kDsLoad2addrStride64B32Vds, {.addr = 0, .vdst = 1});
  const std::array<uint32_t, 3> words = {load[0], load[1],
                                         build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5)};
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(words, "two_address_fault_inventory");
  Options options;
  options.mode = Mode::SuperCollider;
  options.fault_lds_wrong_address = true;
  options.fault_lds_address_vgpr = 4u;
  options.fault_dry_run = true;

  const TransformArtifacts result = test_lower_consan(bytes, options);
  EXPECT_TRUE(std::ranges::none_of(result.fault_sites, [](const FaultSite &site) {
    return site.kind == FaultSiteKind::LdsAccess;
  }));
  EXPECT_TRUE(result.fault_plans.empty());
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("found no site for the requested mutation") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSan, LdsAddressFaultComposesWithAccess) {
  const std::vector<uint8_t> bytes = make_lds_address_fault_code_object(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_FALSE(bytes.empty());

  Options options = test_options();
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(32);
  options.max_patches = 4u;
  options.fault_lds_wrong_address = true;
  options.fault_lds_address_vgpr = 6u;
  options.fault_require_exactly_one = true;

  const TransformArtifacts result = test_lower_consan(bytes, options);
  ASSERT_EQ(result.outcome, TransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.outcome, TransformOutcome::ModifiedValid);
  EXPECT_EQ(result.mutation.fault.applied, 1u);
  const auto mutation =
      std::ranges::find(result.patches, PatchKind::InlineLdsAddressRewrite, &PatchInfo::kind);
  ASSERT_NE(mutation, result.patches.end());
  const auto relocated = std::ranges::find_if(result.patches, [&](const PatchInfo &patch) {
    return patch.phase == PatchPhase::Instrumentation &&
           patch.anchor_offset == mutation->anchor_offset &&
           patch.relocated_guest_instruction_offset.has_value();
  });
  ASSERT_NE(relocated, result.patches.end());
  EXPECT_TRUE(validate_modified_elf(bytes, result).empty());
}

TEST(ConSan, FinalValidationRejectsUnprovenBarrierMutation) {
  const std::array<uint32_t, 2> text_words = {
      0xBF940000u, // s_barrier_wait
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  Options options;
  options.mode = Mode::SuperCollider;
  options.fault_drop_barrier = true;
  const TransformArtifacts valid = test_lower_consan(bytes, options);
  ASSERT_EQ(valid.outcome, TransformOutcome::ModifiedValid);
  ASSERT_EQ(valid.program_inventory.text_sections().size(), 1u);

  TransformArtifacts corrupted = valid;
  std::memcpy(corrupted.replacement.data() +
                  valid.program_inventory.text_sections().front().file_offset,
              text_words.data(), sizeof(uint32_t));
  const std::vector<std::string> errors = validate_modified_elf(bytes, corrupted);

  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("mutation proof did not replace the selected barrier") != std::string::npos;
  }));
}

TEST(ConSan, FinalValidationRejectsWrongAtomicMutationDisplacement) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_atomic_release_acquire_code_object();
  Options options;
  options.mode = Mode::SuperCollider;
  options.fault_atomic_wrong_address = true;
  const TransformArtifacts valid = test_lower_consan(bytes, options);
  ASSERT_EQ(valid.outcome, TransformOutcome::ModifiedValid);
  ASSERT_EQ(valid.patches.size(), 1u);
  ASSERT_EQ(valid.program_inventory.text_sections().size(), 1u);

  TransformArtifacts corrupted = valid;
  const size_t word2_file_offset = valid.program_inventory.text_sections().front().file_offset +
                                   valid.patches.front().anchor_offset + 2 * sizeof(uint32_t);
  uint32_t word2 = 0;
  std::memcpy(&word2, corrupted.replacement.data() + word2_file_offset, sizeof(word2));
  word2 = (word2 & 0xffu) | (2u << 8u);
  std::memcpy(corrupted.replacement.data() + word2_file_offset, &word2, sizeof(word2));
  const std::vector<std::string> errors = validate_modified_elf(bytes, corrupted);

  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("mutation proof found the wrong atomic address displacement") !=
           std::string::npos;
  }));
}

TEST(ConSan, FinalValidationRejectsScopeMutationThatChangesTh) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_release_acquire_code_object();
  Options options = test_options();
  options.track_atomics = true;
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(32);
  options.max_patches = 8;
  options.fault_atomic_weaken_scope = true;
  options.fault_atomic_index = 0;
  const TransformArtifacts valid = test_lower_consan(bytes, options);
  ASSERT_EQ(valid.outcome, TransformOutcome::ModifiedValid);
  EXPECT_EQ(valid.mutation.fault.requested, 1u);
  EXPECT_EQ(valid.mutation.fault.planned, 1u);
  EXPECT_EQ(valid.mutation.fault.applied, 1u);
  ASSERT_EQ(valid.fault_plans.size(), 1u);
  EXPECT_EQ(valid.fault_plans.front().kind, FaultMutationKind::AtomicWeakenScope);
  ASSERT_EQ(valid.program_inventory.text_sections().size(), 1u);
  const auto scope_patch = std::ranges::find_if(valid.patches, [](const PatchInfo &patch) {
    return patch.phase == PatchPhase::Mutation && patch.kind == PatchKind::InlineAtomicScopeRewrite;
  });
  ASSERT_NE(scope_patch, valid.patches.end());

  TransformArtifacts corrupted = valid;
  const size_t word1_file_offset = valid.program_inventory.text_sections().front().file_offset +
                                   scope_patch->anchor_offset + sizeof(uint32_t);
  uint32_t word1 = 0;
  std::memcpy(&word1, corrupted.replacement.data() + word1_file_offset, sizeof(word1));
  word1 ^= 1u << 16u;
  std::memcpy(corrupted.replacement.data() + word1_file_offset, &word1, sizeof(word1));
  const std::vector<std::string> errors = validate_modified_elf(bytes, corrupted);

  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("changed fields other than the selected atomic scope") != std::string::npos;
  }));
}

TEST(ConSan, FinalValidationRejectsCorruptedDsAtomicAddressMutation) {
  const std::vector<uint8_t> bytes = make_rdna4_ds_atomic_code_object();
  Options options = test_options();
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_address_delta = 4;
  const TransformArtifacts valid = test_lower_consan(bytes, options);
  ASSERT_EQ(valid.outcome, TransformOutcome::ModifiedValid);
  ASSERT_EQ(valid.patches.size(), 1u);

  TransformArtifacts corrupted = valid;
  const size_t word1_file_offset =
      valid.program_inventory.text_sections().front().file_offset + sizeof(uint32_t);
  uint32_t word1 = 0;
  std::memcpy(&word1, corrupted.replacement.data() + word1_file_offset, sizeof(word1));
  word1 ^= 1u;
  std::memcpy(corrupted.replacement.data() + word1_file_offset, &word1, sizeof(word1));
  const std::vector<std::string> errors = validate_modified_elf(bytes, corrupted);

  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("wrong atomic address displacement") != std::string::npos;
  }));
}

} // namespace
} // namespace rocjitsu::consan
