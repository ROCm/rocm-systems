// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/major_image_ownership.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_transform_memory.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>

namespace rocjitsu::consan {
namespace {

static_assert(std::same_as<decltype(retry_patch_from_inventory(
                               std::declval<RetryInventory>(), std::declval<TestOptions>(),
                               std::declval<std::span<const uint8_t>>())),
                           TransformArtifacts>);

[[nodiscard]] constexpr uint64_t ownership_mask(major_image_ownership::OwnerKind kind) {
  return uint64_t{1} << static_cast<size_t>(kind);
}

[[nodiscard]] TestOptions release_last_options(bool with_atomic_fault) {
  Options input = test_options();
  input.track_atomics = true;
  input.max_patches = 8;
  input.scratch_vgpr = 8;
  input.requested_owner_vgpr = 40;
  input.requested_epoch_vgpr = 41;
  TestOptions options(input);
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(4);
  options.fault_atomic_wrong_address = with_atomic_fault;
  options.fault_atomic_address_delta = 4;
  options.fault_require_exactly_one = with_atomic_fault;
  return options;
}

TEST(ConSan, FaultRetryRejectsDryRunFaultRequest) {
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_store_and_release_wait_no_return_bitwise_code_object();
  TestOptions inventory_options = release_last_options(/*with_atomic_fault=*/false);
  inventory_options.report_buffer_address.reset();
  inventory_options.report_buffer_size = 0;
  TransformArtifacts inventory = test_lower_consan(bytes, inventory_options);
  ASSERT_TRUE(inventory.errors.empty()) << testing::PrintToString(inventory.errors);

  TestOptions dry_run = release_last_options(/*with_atomic_fault=*/true);
  dry_run.fault_dry_run = true;
  const TransformArtifacts retried =
      test_retry_from_inventory(std::move(inventory), dry_run, bytes);

  EXPECT_EQ(retried.outcome, TransformOutcome::Invalid);
  EXPECT_FALSE(retried.modified());
  EXPECT_EQ(retried.mutation.fault.applied, 0u);
  EXPECT_TRUE(std::ranges::any_of(retried.errors, [](const std::string &error) {
    return error.find("only a live late-bound fault selection") != std::string::npos;
  })) << testing::PrintToString(retried.errors);
}

TEST(ConSan, AtomicWrongAddressRetryFromInventoryMatchesFreshLiveTransform) {
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_store_and_release_wait_no_return_bitwise_code_object();
  const TestOptions live = release_last_options(/*with_atomic_fault=*/true);
  TestOptions inventory_options = live;
  inventory_options.report_buffer_address.reset();
  inventory_options.report_buffer_size = 0;
  inventory_options.fault_atomic_wrong_address = false;
  inventory_options.fault_require_exactly_one = false;
  TransformArtifacts inventory = test_lower_consan(bytes, inventory_options);
  ASSERT_TRUE(inventory.errors.empty()) << testing::PrintToString(inventory.errors);
  ASSERT_FALSE(inventory.modified());

  // A sizing outcome is not part of the immutable retry product.
  inventory.outcome = TransformOutcome::Unsupported;
  const TransformArtifacts fresh = test_lower_consan(bytes, live);
  const TransformArtifacts retried = test_retry_from_inventory(std::move(inventory), live, bytes);

  ASSERT_EQ(retried.outcome, fresh.outcome)
      << testing::PrintToString(retried.errors) << testing::PrintToString(retried.warnings);
  EXPECT_EQ(retried.modified(), fresh.modified());
  EXPECT_EQ(retried.mutation.fault.applied, fresh.mutation.fault.applied);
  EXPECT_EQ(retried.mutation.applied_fault_logical_identity,
            fresh.mutation.applied_fault_logical_identity);
  EXPECT_EQ(retried.replacement, fresh.replacement);
  ASSERT_EQ(retried.patches.size(), fresh.patches.size());
  for (size_t index = 0; index < fresh.patches.size(); ++index) {
    EXPECT_EQ(retried.patches[index].kind, fresh.patches[index].kind);
    EXPECT_EQ(retried.patches[index].anchor_offset, fresh.patches[index].anchor_offset);
    EXPECT_EQ(retried.patches[index].trampoline_offset, fresh.patches[index].trampoline_offset);
  }
  EXPECT_EQ(retried.warnings, fresh.warnings);
}

TEST(ConSan, UnsatisfiedLateFaultRetryMatchesFreshRejection) {
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_store_and_release_wait_no_return_bitwise_code_object();
  TestOptions inventory_options = release_last_options(/*with_atomic_fault=*/false);
  inventory_options.report_buffer_address.reset();
  inventory_options.report_buffer_size = 0;
  TransformArtifacts inventory = test_lower_consan(bytes, inventory_options);
  ASSERT_TRUE(inventory.errors.empty()) << testing::PrintToString(inventory.errors);

  TestOptions live = release_last_options(/*with_atomic_fault=*/true);
  live.fault_site_identity = "missing-site";
  const TransformArtifacts fresh = test_lower_consan(bytes, live);
  const TransformArtifacts retried = test_retry_from_inventory(std::move(inventory), live, bytes);

  EXPECT_EQ(retried.outcome, fresh.outcome);
  EXPECT_EQ(retried.mutation.fault.applied, 0u);
  EXPECT_EQ(retried.replacement, fresh.replacement);
  EXPECT_EQ(retried.errors, fresh.errors);
  EXPECT_EQ(retried.warnings, fresh.warnings);
}

TEST(ConSanBenchmark, LiveFaultInventoryRetryFromObject) {
  const char *path = std::getenv("RJ_CONSAN_BENCHMARK_OBJECT");
  if (path == nullptr)
    GTEST_SKIP() << "set RJ_CONSAN_BENCHMARK_OBJECT to an AMDGPU code object";
  std::ifstream input(path, std::ios::binary);
  ASSERT_TRUE(input) << path;
  const std::vector<uint8_t> bytes(std::istreambuf_iterator<char>(input), {});
  ASSERT_FALSE(bytes.empty());

  const auto timed = [](auto &&action) {
    const auto begin = std::chrono::steady_clock::now();
    auto result = action();
    return std::pair{
        std::move(result),
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count(),
    };
  };
  TestOptions live = test_options();
  live.track_barriers = true;
  live.track_atomics = true;
  live.max_patches_is_expert_limit = false;
  live.fault_atomic_wrong_address = true;
  live.fault_atomic_address_delta = 4;
  live.fault_require_exactly_one = true;

  TestOptions probe_options = live;
  probe_options.fault_dry_run = true;
  probe_options.fault_require_exactly_one = false;
  const auto [probe, probe_ms] = timed([&] { return test_lower_consan(bytes, probe_options); });
  ASSERT_TRUE(probe.errors.empty()) << testing::PrintToString(probe.errors);
  ASSERT_EQ(probe.mutation.fault.planned, 1u)
      << "benchmark object needs one default atomic wrong-address target";

  TestOptions inventory_options = live;
  inventory_options.fault_atomic_wrong_address = false;
  inventory_options.fault_require_exactly_one = false;
  const auto [inventory, inventory_ms] =
      timed([&] { return test_lower_consan(bytes, inventory_options); });
  ASSERT_TRUE(inventory.errors.empty()) << testing::PrintToString(inventory.errors);
  ASSERT_FALSE(inventory.modified());
  const AutoReportInventory capacity = plan_test_evidence_inventory(inventory, inventory_options);
  const AutoReportPlan plan = plan_auto_report(capacity);
  ASSERT_TRUE(plan.complete());
  const auto layout = plan.complete_layout();
  ASSERT_TRUE(layout);
  live.report_buffer_address = 0x123456780000ull;
  live.report_buffer_size = plan.required_bytes;
  live.report_layout = *layout;

  const auto [fresh, fresh_ms] = timed([&] { return test_lower_consan(bytes, live); });
  ASSERT_EQ(fresh.outcome, TransformOutcome::ModifiedValid)
      << testing::PrintToString(fresh.errors) << testing::PrintToString(fresh.warnings);
  TransformArtifacts retry_inventory = inventory;
  const auto [retried, retry_ms] =
      timed([&] { return test_retry_from_inventory(std::move(retry_inventory), live, bytes); });
  ASSERT_EQ(retried.outcome, TransformOutcome::ModifiedValid)
      << testing::PrintToString(retried.errors) << testing::PrintToString(retried.warnings);
  EXPECT_EQ(retried.replacement, fresh.replacement);
  EXPECT_EQ(retried.mutation.fault.applied, 1u);
  const AutoReportInventory required = plan_test_evidence_inventory(retried, live);
  EXPECT_TRUE(auto_report_inventory_covers(capacity, required));

  std::cout << "live_fault_retry_benchmark bytes=" << bytes.size() << " probe_ms=" << probe_ms
            << " inventory_ms=" << inventory_ms << " fresh_live_ms=" << fresh_ms
            << " retry_ms=" << retry_ms << " old_total_ms=" << probe_ms + inventory_ms + fresh_ms
            << " new_total_ms=" << probe_ms + inventory_ms + retry_ms << '\n';
}

TEST(ConSanBenchmark, ReportInventoryRetryFromObject) {
  const char *path = std::getenv("RJ_CONSAN_BENCHMARK_OBJECT");
  if (path == nullptr)
    GTEST_SKIP() << "set RJ_CONSAN_BENCHMARK_OBJECT to an AMDGPU code object";
  std::ifstream input(path, std::ios::binary);
  ASSERT_TRUE(input) << path;
  const std::vector<uint8_t> bytes(std::istreambuf_iterator<char>(input), {});
  ASSERT_FALSE(bytes.empty());

  const auto timed = [](auto &&action) {
    const auto begin = std::chrono::steady_clock::now();
    auto result = action();
    return std::pair{
        std::move(result),
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count(),
    };
  };
  TestOptions inventory_options = test_options();
  // Match the standard hook profile used for large E2E objects. The unit-test
  // defaults deliberately select only one site and would make this benchmark
  // measure a different transformation.
  inventory_options.track_barriers = true;
  inventory_options.track_atomics = true;
  inventory_options.max_patches = 65'536;
  inventory_options.max_patches_is_expert_limit = false;
  inventory_options.runtime_sample_stride = 256u;
  const auto [inventory, inventory_ms] =
      timed([&] { return test_lower_consan(bytes, inventory_options); });
  ASSERT_TRUE(inventory.errors.empty()) << testing::PrintToString(inventory.errors);
  ASSERT_FALSE(inventory.modified());
  const AutoReportInventory capacity = plan_test_evidence_inventory(inventory, inventory_options);
  const AutoReportPlan plan = plan_auto_report(capacity);
  ASSERT_TRUE(plan.complete());
  const auto layout = plan.complete_layout();
  ASSERT_TRUE(layout);

  TestOptions live = inventory_options;
  live.report_buffer_address = 0x123456780000ull;
  live.report_buffer_size = plan.required_bytes;
  live.report_layout = *layout;
  TransformArtifacts retry_inventory = inventory;
  const auto [fresh, fresh_ms] = timed([&] { return test_lower_consan(bytes, live); });
  const auto [retried, retry_ms] =
      timed([&] { return test_retry_from_inventory(std::move(retry_inventory), live, bytes); });

  ASSERT_EQ(fresh.outcome, TransformOutcome::ModifiedValid)
      << testing::PrintToString(fresh.errors) << testing::PrintToString(fresh.warnings);
  ASSERT_EQ(retried.outcome, TransformOutcome::ModifiedValid)
      << testing::PrintToString(retried.errors) << testing::PrintToString(retried.warnings);
  EXPECT_EQ(retried.replacement, fresh.replacement);
  const AutoReportInventory required = plan_test_evidence_inventory(retried, live);
  EXPECT_TRUE(auto_report_inventory_covers(capacity, required));

  std::cout << "report_retry_benchmark bytes=" << bytes.size() << " inventory_ms=" << inventory_ms
            << " fresh_live_ms=" << fresh_ms << " retry_ms=" << retry_ms
            << " patches=" << retried.patches.size()
            << " output_bytes=" << retried.replacement.size() << '\n';
}

TEST(ConSan, AtomicWrongAddressComposesWithReleaseLastProbe) {
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_store_and_release_wait_no_return_bitwise_code_object();
  const TestOptions options = release_last_options(/*with_atomic_fault=*/true);
  const TransformArtifacts valid = test_lower_consan(bytes, options);

  ASSERT_EQ(valid.outcome, TransformOutcome::ModifiedValid)
      << testing::PrintToString(valid.errors) << testing::PrintToString(valid.warnings);
  EXPECT_EQ(valid.outcome, TransformOutcome::ModifiedValid);
  EXPECT_EQ(valid.mutation.fault.applied, 1u);
  ASSERT_GE(valid.replacement.size(), bytes.size());
  const auto mutation =
      std::ranges::find(valid.patches, PatchKind::InlineAtomicAddressRewrite, &PatchInfo::kind);
  const auto record =
      std::ranges::find(valid.patches, PatchKind::TrampolineSyncMetadata, &PatchInfo::kind);
  ASSERT_NE(mutation, valid.patches.end());
  ASSERT_NE(record, valid.patches.end());
  ASSERT_TRUE(record->relocated_guest_instruction_offset);
  // The relocated release is contained in the probe; a return branch may follow it.
  EXPECT_GE(*record->relocated_guest_instruction_offset, record->trampoline_offset);
  EXPECT_LE(*record->relocated_guest_instruction_offset + mutation->original_size,
            record->trampoline_offset + record->trampoline_size);

  AmdGpuCodeObject replacement(valid.replacement.data(), valid.replacement.size());
  ASSERT_TRUE(replacement.is_valid());
  const uint64_t text_file_offset = replacement.text_sections().front()->sectionOffset();
  uint32_t relocated_word2 = 0;
  std::memcpy(&relocated_word2,
              valid.replacement.data() + text_file_offset +
                  *record->relocated_guest_instruction_offset + 2u * sizeof(uint32_t),
              sizeof(relocated_word2));
  EXPECT_EQ((relocated_word2 >> 8u) & 0xffffffu, 4u);

  TransformArtifacts negative_drift = valid;
  relocated_word2 = (relocated_word2 & 0xffu) | (0xfffffcu << 8u);
  std::memcpy(negative_drift.replacement.data() + text_file_offset +
                  *record->relocated_guest_instruction_offset + 2u * sizeof(uint32_t),
              &relocated_word2, sizeof(relocated_word2));
  const std::vector<std::string> errors = validate_modified_elf(bytes, negative_drift);
  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("mutation proof found the wrong atomic address displacement") !=
           std::string::npos;
  })) << testing::PrintToString(errors);
}

TEST(ConSanOwnership, CompositePeakFitsAdmissionAcrossAllPhases) {
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_store_and_release_wait_no_return_bitwise_code_object();
  const TestOptions options = release_last_options(/*with_atomic_fault=*/true);

  major_image_ownership::ScopedMeasurement measurement;
  const TransformArtifacts result = test_lower_consan(bytes, options);
  const major_image_ownership::Measurement observed = measurement.snapshot();

  ASSERT_EQ(result.outcome, TransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors) << testing::PrintToString(result.warnings);
  ASSERT_GE(result.replacement.size(), bytes.size());
  const PatchedImageGrowthLimit exact_growth = {
      .kind = PatchedImageGrowthLimitKind::AbsoluteBytes,
      .absolute_bytes = result.replacement.size() - bytes.size(),
  };
  const auto ownership_estimate =
      hook::transform_major_image_reservation(bytes.size(), exact_growth);
  ASSERT_TRUE(ownership_estimate);
  const auto composite_phase = std::ranges::find(
      hook::kTransformOwnershipPhases, hook::TransformOwnershipPhase::CompositeIncrementalPatch,
      &hook::TransformOwnership::phase);
  ASSERT_NE(composite_phase, hook::kTransformOwnershipPhases.end());
  const auto composite_reservation = hook::transform_phase_reservation_bytes(
      *composite_phase, bytes.size(), result.replacement.size());
  ASSERT_TRUE(composite_reservation);
  EXPECT_EQ(*composite_reservation, bytes.size() + 12u * result.replacement.size())
      << "modified composite transforms must retain their explicit parser-complete phase";
  EXPECT_GE(ownership_estimate->reservation_bytes, *composite_reservation);

  struct MeasuredPhase {
    major_image_ownership::Phase measured;
    hook::TransformOwnershipPhase admitted;
  };
  constexpr std::array measured_phases{
      MeasuredPhase{major_image_ownership::Phase::IncrementalPatch,
                    hook::TransformOwnershipPhase::IncrementalPatch},
      MeasuredPhase{major_image_ownership::Phase::CompositeIncrementalPatch,
                    hook::TransformOwnershipPhase::CompositeIncrementalPatch},
      MeasuredPhase{major_image_ownership::Phase::FinalValidation,
                    hook::TransformOwnershipPhase::FinalValidation},
  };
  EXPECT_FALSE(observed.overflowed);
  EXPECT_FALSE(observed.bookkeeping_error);
  // Admission is a conservative all-input upper bound, not a fixture-specific
  // utilization target. Record the representative peaks without imposing a
  // lower-bound ratio that would couple production coefficients to this ELF.
  for (const MeasuredPhase &phase : measured_phases) {
    const auto admitted = std::ranges::find(hook::kTransformOwnershipPhases, phase.admitted,
                                            &hook::TransformOwnership::phase);
    ASSERT_NE(admitted, hook::kTransformOwnershipPhases.end());
    const auto reservation =
        hook::transform_phase_reservation_bytes(*admitted, bytes.size(), result.replacement.size());
    ASSERT_TRUE(reservation);
    const auto &observation = observed.phase(phase.measured);
    EXPECT_GT(observation.peak_bytes, 0u);
    EXPECT_LE(observation.peak_bytes, *reservation)
        << hook::transform_ownership_phase_name(phase.admitted);
    EXPECT_NE(observation.observed_owner_mask &
                  ownership_mask(major_image_ownership::OwnerKind::Parser),
              0u);
  }
  const auto &incremental = observed.phase(major_image_ownership::Phase::IncrementalPatch);
  EXPECT_NE(incremental.observed_owner_mask &
                ownership_mask(major_image_ownership::OwnerKind::PatcherImage),
            0u);
  const auto &composite = observed.phase(major_image_ownership::Phase::CompositeIncrementalPatch);
  EXPECT_NE(composite.observed_owner_mask &
                ownership_mask(major_image_ownership::OwnerKind::CompositeImage),
            0u);
  // Stable kernel handles eliminate the former pair of descriptor-remapping
  // hash tables from staged composition.
  EXPECT_EQ(composite.observed_owner_mask &
                ownership_mask(major_image_ownership::OwnerKind::CompactIndex),
            0u);
  EXPECT_EQ(composite.observed_owner_mask &
                ownership_mask(major_image_ownership::OwnerKind::ReplacementBytes),
            0u);
  EXPECT_EQ(composite.observed_owner_mask &
                ownership_mask(major_image_ownership::OwnerKind::TransactionImage),
            0u);
  const auto &validation = observed.phase(major_image_ownership::Phase::FinalValidation);
  EXPECT_NE(validation.observed_owner_mask &
                ownership_mask(major_image_ownership::OwnerKind::ResultImage),
            0u);
  RecordProperty("composite_peak_bytes", composite.peak_bytes);
  RecordProperty("final_validation_peak_bytes", validation.peak_bytes);
  RecordProperty("input_image_bytes", bytes.size());
  RecordProperty("replacement_image_bytes", result.replacement.size());
}

TEST(ConSanOwnership, OrdinaryIncrementalPeakFitsAdmission) {
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_store_and_release_wait_no_return_bitwise_code_object();
  const TestOptions options = release_last_options(/*with_atomic_fault=*/false);

  major_image_ownership::ScopedMeasurement measurement;
  const TransformArtifacts result = test_lower_consan(bytes, options);
  const major_image_ownership::Measurement observed = measurement.snapshot();

  ASSERT_EQ(result.outcome, TransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors) << testing::PrintToString(result.warnings);
  const auto phase = std::ranges::find(hook::kTransformOwnershipPhases,
                                       hook::TransformOwnershipPhase::IncrementalPatch,
                                       &hook::TransformOwnership::phase);
  ASSERT_NE(phase, hook::kTransformOwnershipPhases.end());
  const auto reservation =
      hook::transform_phase_reservation_bytes(*phase, bytes.size(), result.replacement.size());
  ASSERT_TRUE(reservation);
  const auto &incremental = observed.phase(major_image_ownership::Phase::IncrementalPatch);
  EXPECT_FALSE(observed.overflowed);
  EXPECT_FALSE(observed.bookkeeping_error);
  EXPECT_GT(incremental.peak_bytes, 0u);
  EXPECT_LE(incremental.peak_bytes, *reservation);
  for (const major_image_ownership::OwnerKind kind : {
           major_image_ownership::OwnerKind::InputImage,
           major_image_ownership::OwnerKind::Parser,
           major_image_ownership::OwnerKind::ResultImage,
           major_image_ownership::OwnerKind::PatcherImage,
       }) {
    EXPECT_NE(incremental.observed_owner_mask & ownership_mask(kind), 0u)
        << static_cast<unsigned>(kind);
  }
  EXPECT_EQ(incremental.observed_owner_mask &
                ownership_mask(major_image_ownership::OwnerKind::ReplacementBytes),
            0u);
  EXPECT_EQ(incremental.observed_owner_mask &
                ownership_mask(major_image_ownership::OwnerKind::TransactionImage),
            0u);
  EXPECT_EQ(observed.phase(major_image_ownership::Phase::CompositeIncrementalPatch).peak_bytes, 0u);
  RecordProperty("incremental_peak_bytes", incremental.peak_bytes);
  RecordProperty("input_image_bytes", bytes.size());
  RecordProperty("replacement_image_bytes", result.replacement.size());
}

TEST(ConSan, AtomicWrongAddressComposesWithRetainedProbe) {
  const std::vector<uint8_t> bytes =
      make_rdna4_ordered_global_atomic_release_acquire_code_object(/*include_lds=*/true);
  TestOptions options = test_options();
  options.track_atomics = true;
  options.scratch_vgpr = 8;
  options.exec_save_sgpr = 80;
  options.set_owner_epoch_vgprs(40, 41);
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(32);
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_address_delta = 4;
  options.fault_require_exactly_one = true;
  options.max_patches = 3;

  const TransformArtifacts valid = test_lower_consan(bytes, options);

  ASSERT_EQ(valid.outcome, TransformOutcome::ModifiedValid)
      << testing::PrintToString(valid.errors) << testing::PrintToString(valid.warnings);
  EXPECT_EQ(valid.outcome, TransformOutcome::ModifiedValid);
  EXPECT_EQ(valid.mutation.fault.applied, 1u);
  const auto mutation =
      std::ranges::find(valid.patches, PatchKind::InlineAtomicAddressRewrite, &PatchInfo::kind);
  ASSERT_NE(mutation, valid.patches.end());
  const auto probe = std::ranges::find_if(valid.patches, [&](const PatchInfo &patch) {
    return patch.kind == PatchKind::TrampolineSyncMetadata &&
           patch.anchor_offset == mutation->anchor_offset &&
           patch.original_size == mutation->original_size;
  });
  const auto any =
      std::ranges::find(valid.patches, PatchKind::TrampolineSyncMetadata, &PatchInfo::kind);
  ASSERT_NE(probe, valid.patches.end())
      << "mutation anchor=" << mutation->anchor_offset
      << " first ConSan anchor=" << (any == valid.patches.end() ? UINT64_MAX : any->anchor_offset);
  ASSERT_TRUE(probe->relocated_guest_instruction_offset);
  EXPECT_GE(*probe->relocated_guest_instruction_offset, probe->trampoline_offset);
  EXPECT_LE(*probe->relocated_guest_instruction_offset + mutation->original_size,
            probe->trampoline_offset + probe->trampoline_size);

  AmdGpuCodeObject replacement(valid.replacement.data(), valid.replacement.size());
  ASSERT_TRUE(replacement.is_valid());
  const uint64_t text_file_offset = replacement.text_sections().front()->sectionOffset();
  uint32_t relocated_word2 = 0;
  std::memcpy(&relocated_word2,
              valid.replacement.data() + text_file_offset +
                  *probe->relocated_guest_instruction_offset + 2u * sizeof(uint32_t),
              sizeof(relocated_word2));
  EXPECT_EQ((relocated_word2 >> 8u) & 0xffffffu, 4u);

  TransformArtifacts negative_drift = valid;
  relocated_word2 = (relocated_word2 & 0xffu) | (0xfffffcu << 8u);
  std::memcpy(negative_drift.replacement.data() + text_file_offset +
                  *probe->relocated_guest_instruction_offset + 2u * sizeof(uint32_t),
              &relocated_word2, sizeof(relocated_word2));
  const std::vector<std::string> errors = validate_modified_elf(bytes, negative_drift);
  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("mutation proof found the wrong atomic address displacement") !=
           std::string::npos;
  })) << testing::PrintToString(errors);
}

TEST(ConSan, PristineAutoReportInventoryCoversLiveBarrierMoveComposition) {
  const std::array<uint32_t, 7> text_words = {
      0xD8340000u, 0x00000000u, // earlier ds_store_b32 destination
      0xBE804EC1u,              // s_barrier_signal -1
      0xBF94FFFFu,              // s_barrier_wait -1
      0xD8D80000u, 0x00000000u, // later ds_load_b32 destination
      0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  TestOptions selection_options;
  selection_options.mode = Mode::SuperCollider;
  const TransformArtifacts selection = test_barrier_move_inventory(bytes, selection_options);
  ASSERT_EQ(std::ranges::count(selection.fault_sites, FaultSiteKind::Barrier, &FaultSite::kind),
            2u);
  const auto destination = std::ranges::find(selection.barrier_move_destinations, 0u,
                                             &BarrierMoveDestination::text_offset);
  ASSERT_NE(destination, selection.barrier_move_destinations.end());

  TestOptions live_options = test_options();
  live_options.track_barriers = true;
  live_options.track_atomics = true;
  live_options.report_buffer_address = 0x123456780000ull;
  live_options.report_buffer_size = 64u * 1024u * 1024u;
  live_options.max_patches = 16;
  live_options.fault_move_barrier = true;
  live_options.fault_require_exactly_one = true;
  live_options.fault_site_identity = selection.fault_sites.front().identity;
  live_options.fault_barrier_move_direction = BarrierMoveDirection::Earlier;
  live_options.fault_barrier_destination_identity = destination->identity;

  TestOptions pristine_options = live_options;
  pristine_options.fault_move_barrier = false;
  pristine_options.fault_require_exactly_one = false;
  pristine_options.report_buffer_address.reset();
  pristine_options.report_buffer_size = 0;
  const TransformArtifacts pristine = test_lower_consan(bytes, pristine_options);
  const AutoReportInventory pristine_inventory =
      plan_test_evidence_inventory(pristine, pristine_options);

  const TransformArtifacts live = test_lower_consan(bytes, live_options);
  ASSERT_EQ(live.outcome, TransformOutcome::ModifiedValid)
      << testing::PrintToString(live.errors) << testing::PrintToString(live.warnings);
  ASSERT_EQ(live.mutation.fault.applied, 1u);
  const TransformArtifacts retried = test_retry_from_inventory(pristine, live_options, bytes);
  ASSERT_EQ(retried.outcome, TransformOutcome::ModifiedValid)
      << testing::PrintToString(retried.errors) << testing::PrintToString(retried.warnings);
  EXPECT_EQ(retried.mutation.fault.applied, 1u);
  EXPECT_EQ(retried.replacement, live.replacement);
  EXPECT_EQ(retried.outcome, live.outcome);
  const AutoReportInventory live_inventory = plan_test_evidence_inventory(retried, live_options);

  EXPECT_TRUE(auto_report_inventory_covers(pristine_inventory, live_inventory));
  EXPECT_EQ(pristine_inventory.access_range_count, live_inventory.access_range_count);
  EXPECT_GE(pristine_inventory.barrier_event_count, live_inventory.barrier_event_count);
  EXPECT_GE(pristine_inventory.atomic_event_count, live_inventory.atomic_event_count);
}

TEST(ConSan, LateUnqualifiedBarrierDropRetryMatchesFreshSafety) {
  std::array<uint32_t, 17> text_words{};
  text_words[0] = 0xBE804EC1u; // s_barrier_signal -1
  std::fill(text_words.begin() + 1, text_words.begin() + 15,
            build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[15] = 0xBF94FFFFu; // s_barrier_wait -1
  text_words[16] = 0xBFB00000u;
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "retry_extended_barrier_pair");

  TestOptions live = test_options();
  live.track_barriers = true;
  live.report_buffer_address = 0x123456780000ull;
  live.report_buffer_size = 64u * 1024u * 1024u;
  live.max_patches = 16;
  live.fault_drop_barrier = true;
  const TransformArtifacts selection = test_lower_consan(bytes, [&] {
    TestOptions options = live;
    options.fault_dry_run = true;
    return options;
  }());
  const auto sequence = std::ranges::find(selection.program_inventory.sync().sync_sequences,
                                          SyncOperation::BarrierFull, &SyncSequence::operation);
  ASSERT_NE(sequence, selection.program_inventory.sync().sync_sequences.end());
  const auto primary = std::ranges::find_if(selection.fault_sites, [&](const auto &site) {
    const ProgramSite *source = test_fault_source(selection, site);
    return test_sync_sequence(selection, site) == &*sequence && source != nullptr &&
           source->mnemonic_view() == "s_barrier_signal";
  });
  ASSERT_NE(primary, selection.fault_sites.end());
  live.fault_site_identity = primary->identity;

  TestOptions inventory_options = live;
  inventory_options.report_buffer_address.reset();
  inventory_options.report_buffer_size = 0;
  inventory_options.fault_drop_barrier = false;
  TransformArtifacts inventory = test_lower_consan(bytes, inventory_options);
  ASSERT_EQ(std::ranges::count(inventory.program_inventory.sync().sync_sequences,
                               SyncOperation::BarrierFull, &SyncSequence::operation),
            1u);
  const TransformArtifacts fresh = test_lower_consan(bytes, live);
  const TransformArtifacts retried = test_retry_from_inventory(std::move(inventory), live, bytes);

  EXPECT_EQ(fresh.outcome, TransformOutcome::Unchanged);
  EXPECT_EQ(retried.outcome, fresh.outcome);
  EXPECT_EQ(retried.mutation.fault.applied, 0u);
  EXPECT_EQ(retried.replacement, fresh.replacement);
  EXPECT_EQ(retried.errors, fresh.errors);
  EXPECT_EQ(retried.warnings, fresh.warnings);
}

TEST(ConSan, LateExactBarrierDropRetryMatchesFreshTransform) {
  std::array<uint32_t, 17> text_words{};
  text_words[0] = 0xBE804EC1u; // s_barrier_signal -1
  std::fill(text_words.begin() + 1, text_words.begin() + 15,
            build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[15] = 0xBF94FFFFu; // s_barrier_wait -1
  text_words[16] = 0xBFB00000u;
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "retry_fallback_barrier_pair");

  TestOptions selection_options = test_options();
  selection_options.fault_drop_barrier = true;
  selection_options.fault_dry_run = true;
  const TransformArtifacts selection = test_lower_consan(bytes, selection_options);
  const auto sequence = std::ranges::find(selection.program_inventory.sync().sync_sequences,
                                          SyncOperation::BarrierFull, &SyncSequence::operation);
  ASSERT_NE(sequence, selection.program_inventory.sync().sync_sequences.end());
  const auto primary = std::ranges::find_if(selection.fault_sites, [&](const auto &site) {
    const ProgramSite *source = test_fault_source(selection, site);
    return test_sync_sequence(selection, site) == &*sequence && source != nullptr &&
           source->mnemonic_view() == "s_barrier_signal";
  });
  ASSERT_NE(primary, selection.fault_sites.end());

  TestOptions live = test_options();
  live.track_barriers = true;
  live.report_buffer_address = 0x123456780000ull;
  live.report_buffer_size = 64u * 1024u * 1024u;
  live.max_patches = 16;
  live.fault_drop_barrier = true;
  live.fault_require_exactly_one = true;
  live.fault_site_identity = primary->identity;
  live.fault_barrier_sequence_identity = sequence->identity;
  TestOptions inventory_options = live;
  inventory_options.report_buffer_address.reset();
  inventory_options.report_buffer_size = 0;
  inventory_options.fault_drop_barrier = false;
  inventory_options.fault_require_exactly_one = false;
  TransformArtifacts inventory = test_lower_consan(bytes, inventory_options);
  ASSERT_EQ(std::ranges::count(inventory.program_inventory.sync().sync_sequences,
                               SyncOperation::BarrierFull, &SyncSequence::operation),
            1u);

  const TransformArtifacts fresh = test_lower_consan(bytes, live);
  const TransformArtifacts retried = test_retry_from_inventory(std::move(inventory), live, bytes);
  ASSERT_EQ(fresh.outcome, TransformOutcome::ModifiedValid) << testing::PrintToString(fresh.errors);
  EXPECT_EQ(retried.outcome, fresh.outcome);
  EXPECT_EQ(retried.mutation.fault.applied, 1u);
  EXPECT_EQ(retried.replacement, fresh.replacement);
  EXPECT_EQ(retried.errors, fresh.errors);
  EXPECT_EQ(retried.warnings, fresh.warnings);
}

TEST(ConSan, FaultBarrierMarkerlessUncoveredLocalCaveComposesWithAccess) {
  const std::array<uint32_t, 9> kernel_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xBE804EC1u,              // s_barrier_signal -1
      0xBF94FFFFu,              // s_barrier_wait -1
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xD8340000u, 0x00000000u, // ds_store_b32 after the move return
      0xBFB00000u,              // s_endpgm
  };
  const std::array<uint32_t, 1> tail_function_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 5> uncovered_cave_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4), build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4), build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_code_object_with_local_function(
      kernel_words, tail_function_words, uncovered_cave_words, kRdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);

  TestOptions inventory_options;
  inventory_options.mode = Mode::SuperCollider;
  inventory_options.test_kernel_name_filter = "lds_probe";
  const TransformArtifacts inventory = test_barrier_move_inventory(bytes, inventory_options);
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, FaultSiteKind::Barrier, &FaultSite::kind),
            2u);
  const auto destination = std::ranges::find(inventory.barrier_move_destinations, 16u,
                                             &BarrierMoveDestination::text_offset);
  ASSERT_NE(destination, inventory.barrier_move_destinations.end());

  TestOptions options = test_options();
  options.track_barriers = true;
  options.report_buffer_address = 0x100000000ull;
  options.report_buffer_size = direct_report_bytes(32);
  options.max_patches = 16;
  options.test_kernel_name_filter = "lds_probe";
  options.fault_move_barrier = true;
  options.fault_require_exactly_one = true;
  options.fault_site_identity = inventory.fault_sites[0].identity;
  options.fault_barrier_move_direction = BarrierMoveDirection::Later;
  options.fault_barrier_destination_identity = destination->identity;

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_EQ(result.outcome, TransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.modified());
  EXPECT_EQ(result.outcome, TransformOutcome::ModifiedValid);
  EXPECT_EQ(result.mutation.fault.applied, 1u);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const PatchInfo &patch) {
    return patch.phase == PatchPhase::Mutation &&
           patch.kind == PatchKind::InlineBarrierMoveTargetRewrite;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const PatchInfo &patch) {
    return patch.phase == PatchPhase::Instrumentation &&
           patch.kind == PatchKind::TrampolineWatchpointStore;
  }));
  // The moved pair in this uncovered cave lacks a qualified sync sequence.
  // Access instrumentation must still compose with the mutation and retain
  // correct ownership both inside its trampoline and after the return.
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("no qualified complete barrier sequence") != std::string::npos;
  }));
  const std::vector<ProgramSite> candidates = test_admitted_accesses(result);
  const auto post_return_candidate =
      std::ranges::find_if(candidates, [](const ProgramSite &candidate) {
        return candidate.physical_id.original_text_offset == 24u;
      });
  ASSERT_NE(post_return_candidate, candidates.end());
  EXPECT_EQ(post_return_candidate->execution_owners.size(), 1u);

  AmdGpuCodeObject original(bytes.data(), bytes.size());
  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_EQ(original.kernels().size(), 2u);
  ASSERT_EQ(patched.kernels().size(), 2u);
  const auto original_owner =
      std::ranges::find(original.kernels(), "lds_probe", &AmdGpuKernelInfo::name);
  const auto original_tail =
      std::ranges::find(original.kernels(), "lds_helper", &AmdGpuKernelInfo::name);
  const auto patched_owner =
      std::ranges::find(patched.kernels(), "lds_probe", &AmdGpuKernelInfo::name);
  const auto patched_tail =
      std::ranges::find(patched.kernels(), "lds_helper", &AmdGpuKernelInfo::name);
  ASSERT_NE(original_owner, original.kernels().end());
  ASSERT_NE(original_tail, original.kernels().end());
  ASSERT_NE(patched_owner, patched.kernels().end());
  ASSERT_NE(patched_tail, patched.kernels().end());
  EXPECT_GT(patched_owner->code_size, original_owner->code_size)
      << "whole-object relocation must expand the owner symbol over its owned local cave";

  const auto mutation_target = std::ranges::find_if(result.patches, [](const auto &patch) {
    return patch.phase == PatchPhase::Mutation &&
           patch.kind == PatchKind::InlineBarrierMoveTargetRewrite;
  });
  ASSERT_NE(mutation_target, result.patches.end());
  const auto nested_instrumentation =
      std::ranges::find_if(result.patches, [&](const PatchInfo &patch) {
        return patch.phase == PatchPhase::Instrumentation &&
               patch.anchor_offset >= mutation_target->trampoline_offset &&
               patch.anchor_offset <
                   mutation_target->trampoline_offset + mutation_target->trampoline_size;
      });
  ASSERT_NE(nested_instrumentation, result.patches.end());
  size_t nested_patch_count = 0;
  for (const PatchInfo &patch : result.patches) {
    if (patch.phase != PatchPhase::Instrumentation ||
        patch.anchor_offset < mutation_target->trampoline_offset ||
        patch.anchor_offset >=
            mutation_target->trampoline_offset + mutation_target->trampoline_size)
      continue;
    ++nested_patch_count;
    EXPECT_NE(std::ranges::find(patch.owner_descriptor_file_offsets,
                                original_owner->descriptor_file_offset),
              patch.owner_descriptor_file_offsets.end());
    EXPECT_EQ(std::ranges::find(patch.owner_descriptor_file_offsets,
                                original_tail->descriptor_file_offset),
              patch.owner_descriptor_file_offsets.end());
  }
  EXPECT_EQ(nested_patch_count, 1u);
  const IntentCoverageEntry *post_return_coverage = access_coverage_at(result, 24u);
  ASSERT_NE(post_return_coverage, nullptr);
  EXPECT_EQ(post_return_coverage->lowering, LoweringOutcomeKind::Instrumented)
      << post_return_coverage->detail;
  const auto prologue = std::ranges::find_if(result.patches, [](const PatchInfo &patch) {
    return patch.kind == PatchKind::KernelEntryOwnerEpochPrologue;
  });
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const PatchInfo &patch) {
                                    return patch.kind == PatchKind::KernelEntryOwnerEpochPrologue;
                                  }),
            1);
  EXPECT_EQ(prologue->anchor_offset, original_owner->entry_text_offset);

  TransformArtifacts corrupted = result;
  AmdGpuCodeObject composed(corrupted.replacement.data(), corrupted.replacement.size());
  ASSERT_EQ(composed.text_sections().size(), 1u);
  const uint64_t text_file_offset = composed.text_sections().front()->sectionOffset();
  ASSERT_TRUE(nested_instrumentation->relocated_guest_instruction_offset);
  const uint32_t invalid_opcode = 0;
  std::memcpy(corrupted.replacement.data() + text_file_offset +
                  *nested_instrumentation->relocated_guest_instruction_offset,
              &invalid_opcode, sizeof(invalid_opcode));
  EXPECT_FALSE(validate_modified_elf(bytes, corrupted).empty());
}

TEST(ConSan, Rdna4DenseRelaysRespectPreappliedBarrierMoveContinuation) {
  constexpr uint32_t kBarrierCount = 10u;
  constexpr uint32_t kEarlyAccessCount = 9u;
  constexpr size_t kFirstAccessWord = 32'900u;
  std::vector<uint32_t> words(
      33'000u, build_s_mov_b32(/*sdst=*/100, /*ssrc0=*/100, ROCJITSU_CODE_ARCH_RDNA4));
  for (uint32_t index = 0u; index < kEarlyAccessCount; ++index) {
    words[32u + 2u * index] = 0xD8340000u | index * sizeof(uint32_t);
    words[33u + 2u * index] = 0x00000000u;
  }
  size_t cursor = kFirstAccessWord;
  words[cursor++] = 0xD8340000u;
  words[cursor++] = 0x00000000u; // ds_store_b32
  words[cursor++] = 0xBE804EC1u; // s_barrier_signal -1
  words[cursor++] = 0xBF94FFFFu; // s_barrier_wait -1
  words[cursor++] = 0xD8D80000u;
  words[cursor++] = 0x00000000u; // ds_load_b32 move destination
  for (uint32_t index = 1u; index < kBarrierCount; ++index) {
    words[cursor++] = 0xD8340000u | index * sizeof(uint32_t);
    words[cursor++] = 0x00000000u; // ds_store_b32 v0, v0 offset:index*4
    words[cursor++] = 0xBE804EC1u; // s_barrier_signal -1
    words[cursor++] = 0xBF94FFFFu; // s_barrier_wait -1
  }
  // The long owner forces retained CFG construction across a realistic dense
  // kernel, while the tail-local group keeps the live mutation cave reachable.
  words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(words, "rdna4_dense_move_composition");

  TestOptions inventory_options;
  inventory_options.mode = Mode::SuperCollider;
  const TransformArtifacts inventory = test_barrier_move_inventory(bytes, inventory_options);
  const auto sequence = std::ranges::find(inventory.program_inventory.sync().sync_sequences,
                                          SyncOperation::BarrierFull, &SyncSequence::operation);
  ASSERT_NE(sequence, inventory.program_inventory.sync().sync_sequences.end());
  const auto primary = std::ranges::find_if(inventory.fault_sites, [&](const FaultSite &site) {
    const ProgramSite *source = test_fault_source(inventory, site);
    return test_sync_sequence(inventory, site) == &*sequence && source != nullptr &&
           source->mnemonic_view() == "s_barrier_signal";
  });
  ASSERT_NE(primary, inventory.fault_sites.end());
  const uint64_t destination_offset = (kFirstAccessWord + 4u) * sizeof(uint32_t);
  const auto destination =
      std::ranges::find(inventory.barrier_move_destinations, destination_offset,
                        &BarrierMoveDestination::text_offset);
  ASSERT_NE(destination, inventory.barrier_move_destinations.end());
  ASSERT_TRUE(destination->suitable())
      << barrier_move_destination_issue_message(destination->issue, destination->issue_detail);

  TestOptions options = test_options();
  options.track_barriers = true;
  options.scratch_vgpr = 8;
  options.exec_save_sgpr = 80;
  options.dispatch_sgpr.set(70);
  options.set_owner_epoch_vgprs(40, 41);
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = 64u * 1024u * 1024u;
  options.runtime_sample_stride = 16'384;
  options.max_patches = 6u * kBarrierCount;
  options.fault_move_barrier = true;
  options.fault_require_exactly_one = true;
  options.fault_site_identity = primary->identity;
  options.fault_barrier_move_direction = BarrierMoveDirection::Later;
  options.fault_barrier_destination_identity = destination->identity;

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_EQ(result.outcome, TransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.outcome, TransformOutcome::ModifiedValid);
  EXPECT_EQ(result.mutation.fault.applied, 1u);
  const PatchKind barrier_patch = PatchKind::TrampolineSyncMetadata;
  EXPECT_GE(std::ranges::count(result.patches, barrier_patch, &PatchInfo::kind), 8u);
  const auto mutation = std::ranges::find_if(result.patches, [](const PatchInfo &patch) {
    return patch.phase == PatchPhase::Mutation &&
           patch.kind == PatchKind::InlineBarrierMoveTargetRewrite;
  });
  ASSERT_NE(mutation, result.patches.end());
  ASSERT_TRUE(result.text_relocation);
}

TEST(ConSan, Rdna4DenseBarrierHostFailurePreservesIndependentAccessPatches) {
  constexpr uint32_t kBarrierCount = 10u;
  std::vector<uint32_t> words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xD8D80000u,
      0x00000000u, // ds_load_b32 move destination
      build_s_branch(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  for (uint32_t index = 1u; index < kBarrierCount; ++index) {
    words.push_back(0xD8340000u | index * sizeof(uint32_t));
    words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
    words.push_back(0xBE804EC1u); // s_barrier_signal -1
    words.push_back(0xBF94FFFFu); // s_barrier_wait -1
    // A branch to the immediately following instruction preserves execution
    // while bounding every basic block below the dense host footprint.
    words.push_back(build_s_branch(0, ROCJITSU_CODE_ARCH_RDNA4));
  }
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(words, "rdna4_dense_move_no_host");

  TestOptions inventory_options;
  inventory_options.mode = Mode::SuperCollider;
  const TransformArtifacts inventory = test_barrier_move_inventory(bytes, inventory_options);
  const auto sequence = std::ranges::find(inventory.program_inventory.sync().sync_sequences,
                                          SyncOperation::BarrierFull, &SyncSequence::operation);
  ASSERT_NE(sequence, inventory.program_inventory.sync().sync_sequences.end());
  const auto primary = std::ranges::find_if(inventory.fault_sites, [&](const FaultSite &site) {
    const ProgramSite *source = test_fault_source(inventory, site);
    return test_sync_sequence(inventory, site) == &*sequence && source != nullptr &&
           source->mnemonic_view() == "s_barrier_signal";
  });
  ASSERT_NE(primary, inventory.fault_sites.end());
  const auto destination = std::ranges::find(inventory.barrier_move_destinations, 16u,
                                             &BarrierMoveDestination::text_offset);
  ASSERT_NE(destination, inventory.barrier_move_destinations.end());
  ASSERT_TRUE(destination->suitable())
      << barrier_move_destination_issue_message(destination->issue, destination->issue_detail);

  TestOptions options = test_options();
  options.track_barriers = true;
  options.scratch_vgpr = 8;
  options.exec_save_sgpr = 80;
  options.dispatch_sgpr.set(70);
  options.set_owner_epoch_vgprs(40, 41);
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(2u * kBarrierCount);
  options.runtime_sample_stride = 64;
  options.max_patches = 4u * kBarrierCount;
  options.fault_move_barrier = true;
  options.fault_require_exactly_one = true;
  options.fault_site_identity = primary->identity;
  options.fault_barrier_move_direction = BarrierMoveDirection::Later;
  options.fault_barrier_destination_identity = destination->identity;

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_EQ(result.outcome, TransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.outcome, TransformOutcome::ModifiedValid);
  EXPECT_EQ(result.mutation.fault.applied, 1u);
  EXPECT_EQ(std::ranges::count(result.patches, PatchKind::TrampolineSyncMetadata, &PatchInfo::kind),
            kBarrierCount);
  EXPECT_EQ(std::ranges::count_if(result.coverage_ledger.lowering_commits(),
                                  [&](const CommittedLowering &commit) {
                                    return committed_lowering_has_intent_kind(
                                               result, commit, ProbeIntentKind::BarrierEpoch) &&
                                           committed_semantic_sites(result, commit).size() == 2u;
                                  }),
            kBarrierCount);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const PatchInfo &patch) {
    return patch.kind == PatchKind::InlineWatchpointStore ||
           patch.kind == PatchKind::TrampolineWatchpointStore;
  }));
}

TEST(ConSan, Gfx1250DenseHostPreservesPreappliedBarrierDrop) {
  constexpr uint32_t kAccessCount = 9u;
  constexpr size_t kLargeTextWords = 33000u;
  const uint32_t filler = build_s_mov_b32(/*sdst=*/100, /*ssrc0=*/100, ROCJITSU_CODE_ARCH_CDNA5);
  std::vector<uint32_t> text_words(kLargeTextWords, filler);
  size_t cursor = 32u;
  const size_t dropped_pair_word = cursor;
  text_words[cursor++] = *build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_CDNA5);
  text_words[cursor++] = *build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_CDNA5);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_CDNA5);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/1, /*ssrc0=*/1, ROCJITSU_CODE_ARCH_CDNA5);
  text_words[cursor++] = *build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_CDNA5);
  text_words[cursor++] = *build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_CDNA5);
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words[cursor++] = 0xD8340000u | index * sizeof(uint32_t);
    text_words[cursor++] = 0x00000000u; // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5);
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "gfx1250_dense_fault");

  TestOptions inventory_options;
  inventory_options.mode = Mode::SuperCollider;
  inventory_options.fault_drop_barrier = true;
  inventory_options.fault_dry_run = true;
  const TransformArtifacts inventory = test_lower_consan(bytes, inventory_options);
  ASSERT_TRUE(inventory.errors.empty()) << testing::PrintToString(inventory.errors);
  ASSERT_EQ(inventory.program_inventory.sync().sync_sequences.size(), 2u);
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, FaultSiteKind::Barrier, &FaultSite::kind),
            4u);

  TestOptions options = test_options();
  options.scratch_vgpr = 82;
  options.set_owner_epoch_vgprs(80, 81);
  options.exec_save_sgpr = 60;
  options.report_buffer_address = 0x100000000ull;
  options.report_buffer_size = direct_report_bytes(32);
  options.track_barriers = true;
  options.track_atomics = false;
  options.max_patches = 64;
  options.fault_drop_barrier = true;
  options.fault_require_exactly_one = true;
  options.fault_site_identity = inventory.fault_sites.front().identity;
  options.fault_barrier_sequence_identity =
      inventory.program_inventory.sync().sync_sequences.front().identity;

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_EQ(result.outcome, TransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.outcome, TransformOutcome::ModifiedValid);
  EXPECT_EQ(result.mutation.fault.applied, 1u);
  EXPECT_EQ(
      std::ranges::count(result.patches, PatchKind::InlineBarrierNopRewrite, &PatchInfo::kind), 2u);
  EXPECT_EQ(
      std::ranges::count(result.patches, PatchKind::TrampolineWatchpointStore, &PatchInfo::kind),
      kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, PatchKind::TrampolineSyncMetadata, &PatchInfo::kind),
            1u);

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  std::array<uint32_t, 2> dropped{};
  std::memcpy(dropped.data(),
              patched.text_sections().front()->data() + dropped_pair_word * sizeof(uint32_t),
              sizeof(dropped));
  EXPECT_EQ(dropped[0], build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5));
  EXPECT_EQ(dropped[1], build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5));
}

} // namespace
} // namespace rocjitsu::consan
