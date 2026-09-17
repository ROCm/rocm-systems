// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

#include <dlfcn.h>

#include "hsa/hsa_api_trace_minimal.h"
#include "patch/consan/consan_model_test_support.h"
#include "patch/consan/lowering_commit_test_support.h"
#include "patch/consan/transform_result_test_access.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/virtual_lds.h"
#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/kernarg_extension.h"
#include "rocjitsu/code/patch/sidecar_metadata.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_hook_internal.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_process_byte_budget.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_registry_lifecycle.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_renderer.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_sync.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_transform_memory.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "scoped_temp.h"
#include "waitcheck_fixture.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

extern "C" bool OnLoad(HsaApiTable *table, uint64_t runtime_version, uint64_t failed_tool_count,
                       const char *const *failed_tool_names);
extern "C" void OnUnload();
extern "C" void rj_hsa_dbt_set_topology_nodes_root_for_test(const char *root);

namespace rocjitsu::consan::hook {
namespace {

template <typename Vocabulary, typename NameFunction>
void expect_hook_vocabulary(const Vocabulary &vocabulary, NameFunction name,
                            std::initializer_list<std::string_view> expected_names) {
  ASSERT_EQ(vocabulary.size(), expected_names.size());
  size_t index = 0;
  for (std::string_view expected : expected_names) {
    const auto value = vocabulary[index++];
    EXPECT_EQ(vocabulary.name(value), expected);
    EXPECT_EQ(std::string_view{name(value)}, expected);
  }
  using Enum = decltype(vocabulary[0]);
  const auto invalid = static_cast<Enum>(255);
  EXPECT_EQ(vocabulary.name(invalid), "unknown");
  EXPECT_EQ(std::string_view{name(invalid)}, "unknown");
}

using ExpectedQueueInterceptPacketWriter = void (*)(const void *, uint64_t);
using ExpectedQueueInterceptHandler = void (*)(const void *, uint64_t, uint64_t, void *,
                                               ExpectedQueueInterceptPacketWriter);
using ExpectedQueueInterceptCreate = hsa_status_t(HSA_API *)(
    hsa_agent_t, uint32_t, hsa_queue_type32_t, void (*)(hsa_status_t, hsa_queue_t *, void *),
    void *, uint32_t, uint32_t, hsa_queue_t **);
using ExpectedQueueInterceptRegister = hsa_status_t(HSA_API *)(hsa_queue_t *,
                                                               ExpectedQueueInterceptHandler,
                                                               void *);
using ExpectedAmdQueueCreate = hsa_status_t(HSA_API *)(hsa_agent_t, hsa_amd_queue_create_desc_t *,
                                                       uint32_t);

static_assert(
    std::is_same_v<hsa_amd_queue_intercept_packet_writer_t, ExpectedQueueInterceptPacketWriter>);
static_assert(std::is_same_v<hsa_amd_queue_intercept_handler_t, ExpectedQueueInterceptHandler>);
static_assert(std::is_same_v<hsa_amd_queue_intercept_create_fn_t, ExpectedQueueInterceptCreate>);
static_assert(
    std::is_same_v<hsa_amd_queue_intercept_register_fn_t, ExpectedQueueInterceptRegister>);
static_assert(std::is_same_v<decltype(AmdExtTable::hsa_amd_queue_intercept_create_fn),
                             ExpectedQueueInterceptCreate>);
static_assert(std::is_same_v<decltype(AmdExtTable::hsa_amd_queue_intercept_register_fn),
                             ExpectedQueueInterceptRegister>);
static_assert(
    std::is_same_v<decltype(AmdExtTable::hsa_amd_queue_create_fn), ExpectedAmdQueueCreate>);

template <typename Configure>
void install_consan_test_program_inventory(TransformArtifacts &result, Configure configure) {
  ProgramInventoryBuilder builder = result.program_inventory.code_object_id().valid()
                                        ? ProgramInventoryBuilder(result.program_inventory)
                                        : ProgramInventoryBuilder(std::span<const uint8_t>{});
  configure(builder);
  builder.publish_decoded_accesses({});
  result.program_inventory = builder.view();
}

void install_consan_test_program_identity(TransformArtifacts &result, rj_code_arch_t arch,
                                          rj_code_target_id_t target,
                                          bool semantic_arch_required = false) {
  install_consan_test_program_inventory(result, [&](ProgramInventoryBuilder &builder) {
    builder.set_code_object_facts(false, 0u, arch, target);
    builder.set_semantic_arch_required(semantic_arch_required);
  });
}

[[nodiscard]] RuntimeCapabilities complete_consan_runtime_capabilities() {
  return {
      .backend = RuntimeBackend::PhysicalHsa,
      .host_device_visible_memory = true,
      .host_device_coherent_memory = true,
      .device_atomic_publication = true,
      .max_report_allocation_bytes = 512u * 1024u * 1024u,
      .max_workgroup_lds_bytes = 64u * 1024u,
      .executable_binding = true,
      .dispatch_segment_binding = true,
  };
}

[[nodiscard]] RuntimePolicy enabled_consan_runtime_policy() {
  RuntimePolicy policy;
  policy.enabled = true;
  return policy;
}

static_assert(std::is_base_of_v<Request, rocjitsu::consan::hook::HookConfig>);
static_assert(std::is_base_of_v<TransformPolicy, rocjitsu::consan::hook::HookConfig>);
static_assert(std::is_base_of_v<RuntimePolicy, rocjitsu::consan::hook::HookConfig>);
static_assert(std::is_base_of_v<DebugOverrides, rocjitsu::consan::hook::HookConfig>);
static_assert(std::is_base_of_v<MutationRequest, rocjitsu::consan::hook::HookConfig>);
static_assert(std::is_base_of_v<BoundRuntimeResources, rocjitsu::consan::hook::HookConfig>);

TEST(HsaHooksUnitTest, QueueInterceptionEntriesUsePublicAbiSignatures) {
  EXPECT_TRUE((std::is_same_v<decltype(AmdExtTable::hsa_amd_queue_intercept_create_fn),
                              ExpectedQueueInterceptCreate>));
  EXPECT_TRUE((std::is_same_v<decltype(AmdExtTable::hsa_amd_queue_intercept_register_fn),
                              ExpectedQueueInterceptRegister>));
  EXPECT_TRUE(
      (std::is_same_v<decltype(AmdExtTable::hsa_amd_queue_create_fn), ExpectedAmdQueueCreate>));
}

TEST(HsaHooksUnitTest, ConSanHookConfigConstructsSeparatedSlice2Contracts) {
  const rocjitsu::consan::hook::HookConfig config;
  EXPECT_FALSE(static_cast<const RuntimePolicy &>(config).enabled);
  EXPECT_FALSE(static_cast<const Request &>(config).mode.has_value());
  EXPECT_EQ(static_cast<const TransformPolicy &>(config).max_patches,
            rocjitsu::consan::hook::kAllSupportedPatchBudget);
  EXPECT_FALSE(static_cast<const TransformPolicy &>(config).max_patches_is_expert_limit);
  EXPECT_FALSE(static_cast<const MutationRequest &>(config).has_mutation());
  EXPECT_FALSE(static_cast<const BoundRuntimeResources &>(config).bound());
}

TEST(HsaHooksUnitTest, ConSanHookDiagnosticVocabulariesPreserveEveryStableSpelling) {

  expect_hook_vocabulary(kFaultSiteKinds, fault_site_kind_name,
                         {"barrier", "atomic", "lds-access", "ordinary-memory"});
  expect_hook_vocabulary(kOrdinaryMemorySupportReasons, ordinary_memory_support_reason_name,
                         {"not-applicable", "supported", "supported-synchronization-only",
                          "unsupported-architecture", "unsupported-encoding-size",
                          "malformed-encoding", "missing-address-vgpr", "missing-destination-vgpr",
                          "missing-value-vgpr"});
  expect_hook_vocabulary(
      kFaultMutationKinds, fault_mutation_kind_name,
      {"drop-barrier", "move-barrier-pair", "barrier-id-scope", "barrier-participant-count",
       "atomic-wrong-address", "atomic-weaken-order", "atomic-weaken-scope", "lds-wrong-address",
       "ordinary-weaken-order", "ordinary-wrong-address", "ordinary-weaken-scope"});
  expect_hook_vocabulary(kBarrierMoveDirections, barrier_move_direction_name,
                         {"legacy-marker", "earlier", "later"});
  expect_hook_vocabulary(
      kBarrierMoveCfgContracts, barrier_move_cfg_contract_name,
      {"same-block", "completing-structured-diamond", "destructive-structured-exec-diamond"});
  expect_hook_vocabulary(kSyncSequenceKinds, sync_sequence_kind_name,
                         {"barrier", "fence", "atomic", "ordinary-memory"});
  expect_hook_vocabulary(kSyncOperations, sync_operation_name,
                         {"unknown", "barrier-signal", "barrier-wait", "barrier-full",
                          "barrier-init", "barrier-join", "barrier-leave", "barrier-wakeup",
                          "barrier-state-query", "fence", "atomic-rmw", "atomic-compare-exchange",
                          "ordinary-load", "ordinary-store"});
  expect_hook_vocabulary(kSyncAddressSources, sync_address_source_name,
                         {"not-applicable", "unknown", "lds-vector", "flat-vector",
                          "global-scalar-vector", "buffer-resource", "scratch-vector"});
  expect_hook_vocabulary(
      kSyncMemoryRoles, sync_memory_role_name,
      {"unknown", "none", "acquire", "release", "acquire-release", "sequentially-consistent"});
  expect_hook_vocabulary(
      kSyncRmwOutcomes, sync_rmw_outcome_name,
      {"not-applicable", "unknown", "no-return", "returns-old-value", "compare-exchange"});
  expect_hook_vocabulary(kSyncConfidences, sync_confidence_name,
                         {"exact", "conservative", "ambiguous", "unsupported"});
  expect_hook_vocabulary(kOwnerProofs, owner_proof_name,
                         {"kernel-local", "direct-call", "recovered-indirect-call"});
  expect_hook_vocabulary(kPatchedImageGrowthLimitKinds, patched_image_growth_limit_kind_name,
                         {"absolute-bytes", "input-percent"});
  expect_hook_vocabulary(kOwnerSources, owner_source_name, {"automatic", "workitem_id", "hw_id"});
  expect_hook_vocabulary(kFlatProvenanceModes, flat_provenance_mode_name, {"likely", "strict"});
  expect_hook_vocabulary(kCheckTrapModes, check_trap_mode_name, {"all", "lds", "flat"});
}

TEST(ProcessByteBudgetTest, PlansCommitsRefundsAndTracksPeak) {
  rocjitsu::consan::hook::ProcessByteBudget budget;

  const auto first = budget.plan_charge(4, 8);
  ASSERT_TRUE(first);
  EXPECT_EQ(first.live_bytes, 0u);
  EXPECT_EQ(first.required_bytes, 4u);
  budget.commit_charge(first);

  const auto rejected = budget.plan_charge(5, 8);
  EXPECT_EQ(rejected.outcome,
            rocjitsu::consan::hook::ProcessByteBudget::ChargeOutcome::LimitExceeded);
  EXPECT_EQ(rejected.live_bytes, 4u);
  EXPECT_EQ(rejected.required_bytes, 9u);
  EXPECT_TRUE(budget.refund(4));
  EXPECT_EQ(budget.summary().live_bytes, 0u);
  EXPECT_EQ(budget.summary().peak_bytes, 4u);

  const auto next_interval = budget.plan_charge(2, std::nullopt);
  ASSERT_TRUE(next_interval);
  budget.commit_charge(next_interval);
  budget.reset_peak_to_live();
  EXPECT_EQ(budget.summary().live_bytes, 2u);
  EXPECT_EQ(budget.summary().peak_bytes, 2u);
  EXPECT_TRUE(budget.refund(2));
}

TEST(ProcessByteBudgetTest, ReportsOverflowAndRecoversFromInvalidRefund) {
  rocjitsu::consan::hook::ProcessByteBudget budget;
  const auto maximum = budget.plan_charge(std::numeric_limits<uint64_t>::max(), std::nullopt);
  ASSERT_TRUE(maximum);
  budget.commit_charge(maximum);

  const auto overflow = budget.plan_charge(1, std::nullopt);
  EXPECT_EQ(overflow.outcome,
            rocjitsu::consan::hook::ProcessByteBudget::ChargeOutcome::AccountingOverflow);
  EXPECT_FALSE(overflow.required_bytes);

  EXPECT_TRUE(budget.refund(std::numeric_limits<uint64_t>::max()));
  const auto small = budget.plan_charge(4, std::nullopt);
  ASSERT_TRUE(small);
  budget.commit_charge(small);
  EXPECT_FALSE(budget.refund(5));
  EXPECT_EQ(budget.summary().live_bytes, 0u);
}

TEST(ConSanTransformMemoryTest, ReportsGoverningFinalValidationPhase) {
  const PatchedImageGrowthLimit absolute = {
      .kind = PatchedImageGrowthLimitKind::AbsoluteBytes,
      .absolute_bytes = 4,
  };
  const auto estimate = rocjitsu::consan::hook::transform_major_image_reservation(8, absolute);
  ASSERT_TRUE(estimate);
  ASSERT_TRUE(estimate->ownership);
  EXPECT_EQ(estimate->reservation_bytes, 172u);
  EXPECT_EQ(estimate->maximum_image_bytes, 12u);
  EXPECT_EQ(estimate->ownership->phase,
            rocjitsu::consan::hook::TransformOwnershipPhase::FinalValidation);
  EXPECT_EQ(estimate->ownership->input_image_copies, 8u);
  EXPECT_EQ(estimate->ownership->maximum_image_copies, 9u);
  EXPECT_STREQ(rocjitsu::consan::hook::transform_ownership_phase_name(estimate->ownership->phase),
               "final-validation");
}

TEST(ConSanTransformMemoryTest, MatchesAbsoluteReservationUnderEquivalentInputPercent) {
  const PatchedImageGrowthLimit relative = {
      .kind = PatchedImageGrowthLimitKind::InputPercent,
      .input_percent = 50,
  };
  const auto estimate = rocjitsu::consan::hook::transform_major_image_reservation(8, relative);
  ASSERT_TRUE(estimate);
  EXPECT_EQ(estimate->reservation_bytes, 172u);
}

TEST(ConSanTransformMemoryTest, PinsDefaultPolicyReservationMagnitude) {
  const PatchedImageGrowthLimit default_policy = {
      .kind = PatchedImageGrowthLimitKind::AbsoluteBytes,
      .absolute_bytes = kDefaultMaxPatchedImageGrowthBytes,
  };
  const auto estimate =
      rocjitsu::consan::hook::transform_major_image_reservation(8, default_policy);
  ASSERT_TRUE(estimate);
  EXPECT_EQ(estimate->reservation_bytes, 11274289256u);
}

TEST(ConSanTransformMemoryTest, ReportsGoverningCompositePhaseForDefaultGrowth) {
  const PatchedImageGrowthLimit default_policy = {
      .kind = PatchedImageGrowthLimitKind::AbsoluteBytes,
      .absolute_bytes = kDefaultMaxPatchedImageGrowthBytes,
  };
  const auto estimate =
      rocjitsu::consan::hook::transform_major_image_reservation(8, default_policy);
  ASSERT_TRUE(estimate);
  ASSERT_TRUE(estimate->ownership);
  EXPECT_EQ(estimate->ownership->phase,
            rocjitsu::consan::hook::TransformOwnershipPhase::CompositeIncrementalPatch);
  EXPECT_EQ(estimate->ownership->input_image_copies, 1u);
  EXPECT_EQ(estimate->ownership->maximum_image_copies, 12u);
}

TEST(ConSanTransformMemoryTest, PinsParserAndPhaseOwnershipCoefficients) {
  using rocjitsu::consan::hook::TransformOwnershipPhase;
  const auto &phases = rocjitsu::consan::hook::kTransformOwnershipPhases;

  EXPECT_EQ(rocjitsu::kAmdGpuCodeObjectRetainedMajorImageUnits, 7u);
  EXPECT_EQ(phases[0].phase, TransformOwnershipPhase::IncrementalPatch);
  EXPECT_EQ(phases[1].phase, TransformOwnershipPhase::CompositeIncrementalPatch);
  EXPECT_EQ(phases[2].phase, TransformOwnershipPhase::FinalValidation);
  EXPECT_EQ(phases[0].input_image_copies, 1u);
  EXPECT_EQ(phases[0].maximum_image_copies, 11u);
  EXPECT_EQ(rocjitsu::consan::hook::transform_max_maximum_image_copies(), 12u);
  EXPECT_EQ(rocjitsu::consan::hook::transform_max_total_copies(), 17u);
}

TEST(ConSanTransformMemoryTest, FloorsSubUnitPercentGrowthToZeroExtraBytes) {
  // Any percentage below 100 floors to zero extra bytes for a one-byte image.
  for (const uint32_t percent : {1u, 37u, 99u}) {
    const PatchedImageGrowthLimit policy = {
        .kind = PatchedImageGrowthLimitKind::InputPercent,
        .input_percent = percent,
    };
    const auto estimate = rocjitsu::consan::hook::transform_major_image_reservation(1, policy);
    ASSERT_TRUE(estimate);
    EXPECT_EQ(estimate->maximum_image_bytes, 1u);
    EXPECT_EQ(estimate->reservation_bytes, 17u);
  }
}

TEST(ConSanTransformMemoryTest, ReportsNoGoverningPhaseForZeroReservation) {
  const PatchedImageGrowthLimit no_growth = {
      .kind = PatchedImageGrowthLimitKind::AbsoluteBytes,
      .absolute_bytes = 0,
  };
  const auto estimate = rocjitsu::consan::hook::transform_major_image_reservation(0, no_growth);
  ASSERT_TRUE(estimate);
  EXPECT_EQ(estimate->maximum_image_bytes, 0u);
  EXPECT_EQ(estimate->reservation_bytes, 0u);
  EXPECT_FALSE(estimate->ownership);
}

TEST(ConSanTransformMemoryTest, RejectsInputPlusGrowthOverflow) {
  const PatchedImageGrowthLimit sum_overflow = {
      .kind = PatchedImageGrowthLimitKind::AbsoluteBytes,
      .absolute_bytes = std::numeric_limits<uint64_t>::max(),
  };
  EXPECT_FALSE(rocjitsu::consan::hook::transform_major_image_reservation(8, sum_overflow));
}

TEST(ConSanTransformMemoryTest, RejectsMaximumImagePhaseMultiplyOverflow) {
  const PatchedImageGrowthLimit multiply_overflow = {
      .kind = PatchedImageGrowthLimitKind::AbsoluteBytes,
      .absolute_bytes = std::numeric_limits<uint64_t>::max() /
                        rocjitsu::consan::hook::transform_max_maximum_image_copies(),
  };
  EXPECT_FALSE(rocjitsu::consan::hook::transform_major_image_reservation(8, multiply_overflow));
}

TEST(ConSanTransformMemoryTest, RejectsInputImagePhaseMultiplyOverflow) {
  constexpr rocjitsu::consan::hook::TransformOwnership final_validation =
      rocjitsu::consan::hook::kTransformOwnershipPhases[2];
  static_assert(final_validation.phase ==
                rocjitsu::consan::hook::TransformOwnershipPhase::FinalValidation);
  EXPECT_FALSE(rocjitsu::consan::hook::transform_phase_reservation_bytes(
      final_validation,
      std::numeric_limits<uint64_t>::max() / final_validation.input_image_copies + 1, 0));
}

TEST(ConSanTransformMemoryTest, RejectsPhaseReservationSumOverflow) {
  const PatchedImageGrowthLimit no_growth = {
      .kind = PatchedImageGrowthLimitKind::AbsoluteBytes,
      .absolute_bytes = 0,
  };
  constexpr uint64_t maximum_total_copies = rocjitsu::consan::hook::transform_max_total_copies();
  EXPECT_FALSE(rocjitsu::consan::hook::transform_major_image_reservation(
      std::numeric_limits<uint64_t>::max() / maximum_total_copies + 1, no_growth));
}

TEST(ConSanTransformMemoryTest, ReturnsExactLargestNoGrowthReservation) {
  const PatchedImageGrowthLimit no_growth = {
      .kind = PatchedImageGrowthLimitKind::AbsoluteBytes,
      .absolute_bytes = 0,
  };
  constexpr uint64_t maximum_total_copies = rocjitsu::consan::hook::transform_max_total_copies();
  const uint64_t input = std::numeric_limits<uint64_t>::max() / maximum_total_copies;
  const auto estimate = rocjitsu::consan::hook::transform_major_image_reservation(input, no_growth);
  ASSERT_TRUE(estimate);
  EXPECT_EQ(estimate->reservation_bytes, input * maximum_total_copies);
}

TEST(ConSanTransformMemoryTest, RejectsUnknownGrowthPolicyKind) {
  const PatchedImageGrowthLimit invalid = {
      .kind = static_cast<PatchedImageGrowthLimitKind>(255),
  };
  EXPECT_FALSE(rocjitsu::consan::hook::transform_major_image_reservation(8, invalid));
}

constexpr hsa_agent_t kGuestAgent{1};
constexpr hsa_agent_t kHostAgent{2};
// An agent that is neither the guest nor the guest's execution host. Its queues
// are tracked for doorbell forwarding but never rewritten, and host_lds_bytes
// (derived from the guest target arch) does not apply to them.
constexpr hsa_agent_t kUnrelatedAgent{3};
// A CPU agent. Enumeration must keep publishing it even when the guest/host
// mapping is unresolved: host fine-grained pools, hsa_amd_memory_lock, and
// host-side copies all need it, and it cannot execute a guest kernel.
constexpr hsa_agent_t kCpuAgent{4};
constexpr hsa_isa_t kGuestIsa{950};
constexpr hsa_isa_t kHostIsa{1201};
constexpr hsa_amd_memory_pool_t kGuestPool{10};
constexpr hsa_amd_memory_pool_t kHostPool{20};
constexpr hsa_amd_memory_pool_t kHostKernargPool{21};
constexpr hsa_amd_memory_pool_t kCpuFineGrainedPool{30};
constexpr uint32_t kGuestNodeId = 100;
constexpr uint32_t kHostNodeId = 200;
constexpr uint32_t kCpuNodeId = 0;
constexpr uint32_t kVirtualLdsWrapperStateOffsetForTest = 8;
constexpr uint32_t kVirtualLdsWrapperSizeForTest = 32;
constexpr uint16_t kVirtualLdsWrapperFlagsForTest =
    rocjitsu::kVirtualLdsFlagRuntimeStateBlock | rocjitsu::kVirtualLdsFlagWorkgroupIdX;

std::mutex g_pool_mutex;
std::condition_variable g_pool_cv;
bool g_block_guest_pool_iteration = false;
bool g_guest_pool_iteration_entered = false;
bool g_release_guest_pool_iteration = false;
bool g_fail_guest_pool_iteration_once = false;
std::mutex g_agent_mutex;
std::condition_variable g_agent_cv;
bool g_block_agent_iteration = false;
bool g_agent_iteration_entered = false;
bool g_release_agent_iteration = false;
// Plain globals rather than g_agent_mutex-guarded state: only the
// single-threaded discovery-retry tests touch them, and the blocking multi-
// threaded test must not contend on them.
bool g_fail_agent_iteration = false;
int g_fake_iterate_agents_calls = 0;
int g_fake_shutdown_calls = 0;
hsa_amd_memory_pool_t g_last_allocate_pool{};
hsa_agent_t g_last_agent_memory_pool_agent{};
hsa_amd_memory_pool_t g_last_agent_memory_pool{};
int g_agent_memory_pool_get_info_calls = 0;
int g_fake_allocation_storage = 0;
hsa_agent_t g_pointer_info_accessible[2] = {};
std::vector<uint64_t> g_last_batch_src_agents;
std::vector<uint64_t> g_last_batch_dst_agents;
std::vector<uint64_t> g_last_memory_lock_agents;
std::vector<uint64_t> g_last_memory_lock_to_pool_agents;
std::vector<uint64_t> g_last_vmem_access_agents;
hsa_amd_memory_pool_t g_last_memory_lock_to_pool_pool{};
int g_code_object_reader_create_calls = 0;
bool g_fail_replacement_reader_create = false;
bool g_fail_core_memory_allocate = false;
bool g_offer_fine_report_region = true;
bool g_offer_coarse_report_region = false;
bool g_offer_coarse_report_region_first = false;
int g_core_memory_allocate_calls = 0;
int g_core_memory_free_calls = 0;
int g_core_memory_runtime_reclaim_calls = 0;
std::vector<size_t> g_core_memory_allocation_sizes;
std::vector<uint64_t> g_core_memory_allocation_regions;
std::vector<void *> g_core_memory_allocations;
// Used only by the rollover integration test to force exact-address reuse
// with dirty contents rather than relying on the system allocator's policy.
bool g_reuse_core_memory = false;
void *g_recycled_core_memory = nullptr;
size_t g_recycled_core_memory_size = 0;
std::vector<ReportHeader> g_core_memory_headers_at_free;
std::vector<uint32_t> g_sc_markers_at_free;
std::vector<std::vector<uint8_t>> g_code_object_reader_inputs;
struct FakeMemoryReader {
  uint64_t handle = 0;
  const uint8_t *bytes = nullptr;
  size_t size = 0;
  bool replacement = false;
};
std::vector<FakeMemoryReader> g_memory_code_object_readers;
std::vector<uint64_t> g_destroyed_code_object_readers;
std::vector<uint64_t> g_destroyed_executables;
std::vector<std::pair<uint64_t, bool>> g_replacement_storage_valid_by_executable;
std::vector<uint64_t> g_loaded_code_object_readers;
std::vector<std::pair<uint64_t, uint64_t>> g_loaded_executable_readers;
TransformArtifacts g_transform_override_result;
std::deque<TransformArtifacts> g_transform_override_results;
size_t g_log_sink_write_count = 0;
size_t g_log_sink_max_write_size = 0;
bool g_log_sink_writes_end_in_newline = true;
std::string g_log_sink_bytes;
std::vector<Mode> g_transform_override_flavors;
std::vector<std::vector<std::string>> g_transform_override_kernel_allowlists;
std::vector<bool> g_transform_override_abort_unmatched_waits;
std::vector<bool> g_transform_override_track_barriers;
std::vector<bool> g_transform_override_track_atomics;
std::vector<bool> g_transform_override_fault_drop_barriers;
std::vector<bool> g_transform_override_fault_mutations;
std::vector<bool> g_transform_override_fault_dry_runs;
std::vector<PatchedImageGrowthLimit> g_transform_override_patched_image_growth_limits;
bool g_transform_override_models_fault_application = false;
size_t g_transform_override_actual_fault_applications = 1;
std::optional<TransformArtifacts> g_transform_override_live_fault_result;
std::mutex g_transform_observation_mutex;
std::mutex g_transform_block_mutex;
std::condition_variable g_transform_block_cv;
bool g_block_first_transform = false;
bool g_first_transform_entered = false;
bool g_release_first_transform = false;
std::mutex g_fault_application_block_mutex;
std::condition_variable g_fault_application_block_cv;
bool g_block_first_fault_application = false;
bool g_first_fault_application_entered = false;
bool g_release_first_fault_application = false;
std::optional<TransformArtifacts> g_first_fault_application_result;
std::mutex g_loader_block_mutex;
std::condition_variable g_loader_block_cv;
bool g_block_first_loader_call = false;
std::optional<uint64_t> g_block_loader_reader;
bool g_first_loader_call_entered = false;
bool g_release_first_loader_call = false;
std::optional<uint64_t> g_fail_loader_once_for_reader;
std::function<hsa_status_t()> g_reentrant_fault_load;
std::optional<hsa_status_t> g_reentrant_fault_load_status;
std::vector<uint32_t> g_transform_override_runtime_sample_strides;
std::vector<uint64_t> g_transform_override_report_sizes;
std::vector<std::optional<uint64_t>> g_transform_override_sc_report_addresses;
std::vector<std::optional<ReportBufferLayout>> g_transform_override_report_layouts;
bool g_seed_auto_report_on_load = false;
bool g_seed_auto_report_succeeded = false;
bool g_seed_auto_pending_release_scale = false;
bool g_seed_auto_pending_identity_collision = false;
bool g_seed_auto_conflict_pair = false;
bool g_seed_auto_distinct_dispatches = false;
bool g_seed_auto_distinct_clusters = false;
std::vector<std::vector<uint8_t>> g_fake_allocations;
std::vector<hsa_amd_memory_pool_t> g_fake_allocation_pools;
std::vector<size_t> g_fake_allocation_sizes;
std::vector<void *> g_fake_freed_allocations;
std::array<hsa_kernel_dispatch_packet_t, 4> g_fake_queue_packets{};
hsa_queue_t g_fake_queue{};

void reset_queue_fakes();

void capture_log_sink(const char *bytes, size_t size) {
  ++g_log_sink_write_count;
  g_log_sink_max_write_size = std::max(g_log_sink_max_write_size, size);
  g_log_sink_writes_end_in_newline &= size != 0u && bytes[size - 1u] == '\n';
  g_log_sink_bytes.append(bytes, size);
}
std::array<hsa_kernel_dispatch_packet_t, 4> g_fake_batch_queue_packets{};
hsa_queue_t g_fake_batch_queue{};
int g_fake_amd_queue_create_calls = 0;
hsa_agent_t g_last_queue_create_agent{};
hsa_queue_t *g_last_destroyed_queue = nullptr;
int g_fake_signal_store_relaxed_calls = 0;
int g_fake_signal_store_screlease_calls = 0;
hsa_signal_t g_last_signal_store_signal{};
hsa_signal_value_t g_last_signal_store_value = 0;
uint64_t g_next_fake_signal_handle = 10000;
std::vector<hsa_signal_t> g_fake_created_signals;
std::vector<hsa_signal_t> g_fake_destroyed_signals;
struct FakeSignalValue {
  uint64_t handle = 0;
  hsa_signal_value_t value = 0;
};
std::vector<FakeSignalValue> g_fake_signal_values;
hsa_queue_t *g_last_intercept_registered_queue = nullptr;
hsa_amd_queue_intercept_handler_t g_fake_intercept_handler = nullptr;
void *g_fake_intercept_user_data = nullptr;
std::vector<hsa_kernel_dispatch_packet_t> g_last_intercept_written_packets;
uint64_t g_fake_symbol_kernel_object = 0;
uint32_t g_fake_symbol_group_segment_size = 0;
uint32_t g_fake_symbol_private_segment_size = 0;
std::string g_fake_symbol_name = "oversized_kernel.kd";
// Records the arguments the hook forwards to the original agent-code-object
// loader, so a test can assert a non-guest load reaches the loader unchanged.
int g_fake_load_agent_calls = 0;
hsa_agent_t g_last_load_agent{};
hsa_code_object_reader_t g_last_load_reader{};
constexpr hsa_executable_t kFakeExecutable{123};
constexpr hsa_executable_symbol_t kFakeKernelSymbol{500};
constexpr uint32_t kResolvedHostGpuId = 8716;

const char *isa_name(hsa_isa_t isa) {
  if (isa.handle == kGuestIsa.handle)
    return "amdgcn-amd-amdhsa--gfx950";
  if (isa.handle == kHostIsa.handle)
    return "amdgcn-amd-amdhsa--gfx1201";
  return "";
}

hsa_status_t HSA_API fake_iterate_agents(hsa_status_t (*callback)(hsa_agent_t, void *),
                                         void *data) {
  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  ++g_fake_iterate_agents_calls;
  if (g_fail_agent_iteration)
    return HSA_STATUS_ERROR;

  {
    std::unique_lock lock(g_agent_mutex);
    if (g_block_agent_iteration) {
      g_agent_iteration_entered = true;
      g_agent_cv.notify_all();
      g_agent_cv.wait(lock, [] { return g_release_agent_iteration; });
    }
  }

  hsa_status_t status = callback(kGuestAgent, data);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  return callback(kHostAgent, data);
}

hsa_status_t HSA_API fake_iterate_agents_host_first(hsa_status_t (*callback)(hsa_agent_t, void *),
                                                    void *data) {
  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  ++g_fake_iterate_agents_calls;
  hsa_status_t status = callback(kHostAgent, data);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  return callback(kGuestAgent, data);
}

hsa_status_t HSA_API fake_iterate_agents_with_cpu(hsa_status_t (*callback)(hsa_agent_t, void *),
                                                  void *data) {
  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  ++g_fake_iterate_agents_calls;
  hsa_status_t status = callback(kCpuAgent, data);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  status = callback(kGuestAgent, data);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  return callback(kHostAgent, data);
}

// Only the physical host is present -- the synthetic guest node never appeared.
// Paired with fake_agent_iterate_isas_overlapping this is the guest/host role
// collision: one agent satisfies both predicates and no distinct guest remains.
hsa_status_t HSA_API fake_iterate_agents_host_only(hsa_status_t (*callback)(hsa_agent_t, void *),
                                                   void *data) {
  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  ++g_fake_iterate_agents_calls;
  return callback(kHostAgent, data);
}

hsa_status_t HSA_API fake_shut_down() {
  ++g_fake_shutdown_calls;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute,
                                         void *value) {
  if (value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (attribute == HSA_AGENT_INFO_DEVICE) {
    *static_cast<hsa_device_type_t *>(value) =
        agent.handle == kCpuAgent.handle ? HSA_DEVICE_TYPE_CPU : HSA_DEVICE_TYPE_GPU;
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_AGENT_INFO_ISA) {
    *static_cast<hsa_isa_t *>(value) = agent.handle == kGuestAgent.handle ? kGuestIsa : kHostIsa;
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID)) {
    if (agent.handle == kCpuAgent.handle)
      *static_cast<uint32_t *>(value) = kCpuNodeId;
    else
      *static_cast<uint32_t *>(value) =
          agent.handle == kGuestAgent.handle ? kGuestNodeId : kHostNodeId;
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t HSA_API fake_agent_iterate_isas(hsa_agent_t agent,
                                             hsa_status_t (*callback)(hsa_isa_t, void *),
                                             void *data) {
  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (agent.handle == kGuestAgent.handle)
    return callback(kGuestIsa, data);
  if (agent.handle == kHostAgent.handle)
    return callback(kHostIsa, data);
  return HSA_STATUS_ERROR_INVALID_AGENT;
}

// Every GPU advertises both the guest and the host ISA. This is the shape of a
// same-ISA DBT config (guest_isa == host_isa, the gfx1250 B0-on-A0 revision
// profile): both predicates match every agent, so only the host's node-id
// constraint can tell the two roles apart.
hsa_status_t HSA_API fake_agent_iterate_isas_overlapping(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_isa_t, void *), void *data) {
  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (agent.handle != kGuestAgent.handle && agent.handle != kHostAgent.handle)
    return HSA_STATUS_ERROR_INVALID_AGENT;

  hsa_status_t status = callback(kGuestIsa, data);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  return callback(kHostIsa, data);
}

hsa_status_t HSA_API fake_isa_get_info_alt(hsa_isa_t isa, hsa_isa_info_t attribute, void *value) {
  if (value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  const char *name = isa_name(isa);
  if (name[0] == '\0')
    return HSA_STATUS_ERROR_INVALID_ISA;

  if (attribute == HSA_ISA_INFO_NAME_LENGTH) {
    *static_cast<uint32_t *>(value) = static_cast<uint32_t>(std::strlen(name));
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_ISA_INFO_NAME) {
    std::strcpy(static_cast<char *>(value), name);
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t HSA_API fake_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                       void (*)(hsa_status_t, hsa_queue_t *, void *), void *,
                                       uint32_t, uint32_t, hsa_queue_t **queue) {
  if (queue == nullptr || size == 0 || size > g_fake_queue_packets.size())
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  g_last_queue_create_agent = agent;
  g_fake_queue_packets = {};
  g_fake_queue = {};
  g_fake_queue.type = type;
  g_fake_queue.features = HSA_QUEUE_FEATURE_KERNEL_DISPATCH;
  g_fake_queue.base_address = g_fake_queue_packets.data();
  g_fake_queue.doorbell_signal = hsa_signal_t{77};
  g_fake_queue.size = size;
  g_fake_queue.id = 1234;
  *queue = &g_fake_queue;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_queue_destroy(hsa_queue_t *queue) {
  g_last_destroyed_queue = queue;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_queue_create(hsa_agent_t agent, hsa_amd_queue_create_desc_t *descs,
                                           uint32_t num_descs) {
  ++g_fake_amd_queue_create_calls;
  if (descs == nullptr || num_descs != 1u)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  hsa_amd_queue_create_desc_t &desc = descs[0];
  desc.queue = nullptr;
  if (desc.version != HSA_AMD_QUEUE_CREATE_DESC_VERSION || desc.queue_size_bytes == 0u)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  g_last_queue_create_agent = agent;
  g_fake_batch_queue_packets = {};
  g_fake_batch_queue = {};
  g_fake_batch_queue.type = desc.engine.compute.type;
  g_fake_batch_queue.features = desc.engine_type == HSA_AMD_QUEUE_ENGINE_COMPUTE
                                    ? static_cast<uint32_t>(HSA_QUEUE_FEATURE_KERNEL_DISPATCH)
                                    : 0u;
  g_fake_batch_queue.base_address = g_fake_batch_queue_packets.data();
  g_fake_batch_queue.doorbell_signal = hsa_signal_t{78};
  g_fake_batch_queue.size = desc.engine_type == HSA_AMD_QUEUE_ENGINE_COMPUTE
                                ? desc.queue_size_bytes / sizeof(hsa_kernel_dispatch_packet_t)
                                : desc.queue_size_bytes;
  g_fake_batch_queue.id = 1235;
  desc.queue = &g_fake_batch_queue;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_queue_intercept_create(
    hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t, hsa_queue_t *, void *), void *data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t **queue) {
  return fake_queue_create(agent, size, type, callback, data, private_segment_size,
                           group_segment_size, queue);
}

hsa_status_t HSA_API fake_amd_queue_intercept_register(hsa_queue_t *queue,
                                                       hsa_amd_queue_intercept_handler_t callback,
                                                       void *user_data) {
  g_last_intercept_registered_queue = queue;
  g_fake_intercept_handler = callback;
  g_fake_intercept_user_data = user_data;
  return HSA_STATUS_SUCCESS;
}

void fake_intercept_packet_writer(const void *pkts, uint64_t pkt_count) {
  g_last_intercept_written_packets.clear();
  if (pkts == nullptr || pkt_count == 0)
    return;

  const auto *packets = static_cast<const hsa_kernel_dispatch_packet_t *>(pkts);
  g_last_intercept_written_packets.assign(packets, packets + pkt_count);
}

void HSA_API fake_signal_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
  ++g_fake_signal_store_relaxed_calls;
  g_last_signal_store_signal = signal;
  g_last_signal_store_value = value;
}

void HSA_API fake_signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
  ++g_fake_signal_store_screlease_calls;
  g_last_signal_store_signal = signal;
  g_last_signal_store_value = value;
}

void set_fake_signal_value(hsa_signal_t signal, hsa_signal_value_t value) {
  for (FakeSignalValue &entry : g_fake_signal_values) {
    if (entry.handle == signal.handle) {
      entry.value = value;
      return;
    }
  }
  g_fake_signal_values.push_back(FakeSignalValue{.handle = signal.handle, .value = value});
}

hsa_status_t HSA_API fake_signal_create(hsa_signal_value_t initial_value, uint32_t,
                                        const hsa_agent_t *, hsa_signal_t *signal) {
  if (signal == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *signal = hsa_signal_t{g_next_fake_signal_handle++};
  g_fake_created_signals.push_back(*signal);
  set_fake_signal_value(*signal, initial_value);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_signal_destroy(hsa_signal_t signal) {
  g_fake_destroyed_signals.push_back(signal);
  return HSA_STATUS_SUCCESS;
}

hsa_signal_value_t HSA_API fake_signal_load_scacquire(hsa_signal_t signal) {
  for (const FakeSignalValue &entry : g_fake_signal_values) {
    if (entry.handle == signal.handle)
      return entry.value;
  }
  return 1;
}

hsa_signal_value_t HSA_API fake_signal_wait_relaxed(hsa_signal_t signal, hsa_signal_condition_t,
                                                    hsa_signal_value_t, uint64_t,
                                                    hsa_wait_state_t) {
  return fake_signal_load_scacquire(signal);
}

hsa_signal_value_t HSA_API fake_signal_wait_scacquire(hsa_signal_t signal, hsa_signal_condition_t,
                                                      hsa_signal_value_t, uint64_t,
                                                      hsa_wait_state_t) {
  return fake_signal_load_scacquire(signal);
}

hsa_status_t HSA_API
fake_code_object_reader_create_from_file(hsa_file_t, hsa_code_object_reader_t *code_object_reader) {
  if (code_object_reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  code_object_reader->handle = 1;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_code_object_reader_create_from_memory(
    const void *bytes, size_t size, hsa_code_object_reader_t *code_object_reader) {
  if (code_object_reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  ++g_code_object_reader_create_calls;
  if (g_code_object_reader_create_calls > 1 && g_fail_replacement_reader_create)
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  const auto *begin = static_cast<const uint8_t *>(bytes);
  g_code_object_reader_inputs.emplace_back(begin, begin == nullptr ? begin : begin + size);
  code_object_reader->handle = 100u + static_cast<uint64_t>(g_code_object_reader_create_calls);
  const bool replacement = !g_transform_override_result.replacement.empty() &&
                           g_transform_override_result.replacement.size() == size &&
                           std::equal(g_transform_override_result.replacement.begin(),
                                      g_transform_override_result.replacement.end(), begin);
  g_memory_code_object_readers.push_back({code_object_reader->handle, begin, size, replacement});
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_code_object_reader_destroy(hsa_code_object_reader_t reader) {
  g_destroyed_code_object_readers.push_back(reader.handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_executable_destroy(hsa_executable_t executable) {
  g_destroyed_executables.push_back(executable.handle);
  bool saw_replacement = false;
  bool all_replacements_valid = true;
  for (const auto &[loaded_executable, loaded_reader] : g_loaded_executable_readers) {
    if (loaded_executable != executable.handle)
      continue;
    const auto replacement =
        std::ranges::find(g_memory_code_object_readers, loaded_reader, &FakeMemoryReader::handle);
    if (replacement == g_memory_code_object_readers.end() || !replacement->replacement)
      continue;
    saw_replacement = true;
    const size_t index = static_cast<size_t>(replacement - g_memory_code_object_readers.begin());
    const bool valid = index < g_code_object_reader_inputs.size() &&
                       replacement->bytes != nullptr &&
                       replacement->size == g_code_object_reader_inputs[index].size() &&
                       std::equal(g_code_object_reader_inputs[index].begin(),
                                  g_code_object_reader_inputs[index].end(), replacement->bytes);
    all_replacements_valid = all_replacements_valid && valid;
  }
  if (saw_replacement)
    g_replacement_storage_valid_by_executable.emplace_back(executable.handle,
                                                           all_replacements_valid);
  return HSA_STATUS_SUCCESS;
}

bool replacement_storage_valid_at_destroy(uint64_t executable) {
  const auto result = std::ranges::find(g_replacement_storage_valid_by_executable, executable,
                                        &std::pair<uint64_t, bool>::first);
  return result != g_replacement_storage_valid_by_executable.end() && result->second;
}

hsa_status_t HSA_API fake_system_get_extension_table(uint16_t, uint16_t, uint16_t, void *) {
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t HSA_API fake_system_get_major_extension_table(uint16_t, uint16_t, size_t, void *) {
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t HSA_API fake_queue_intercept_create(hsa_agent_t, uint32_t, hsa_queue_type32_t,
                                                 void (*)(hsa_status_t, hsa_queue_t *, void *),
                                                 void *, uint32_t, uint32_t, hsa_queue_t **) {
  return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
}

hsa_status_t HSA_API fake_queue_intercept_register(hsa_queue_t *, hsa_amd_queue_intercept_handler_t,
                                                   void *) {
  return HSA_STATUS_ERROR_INVALID_QUEUE;
}

hsa_status_t HSA_API fake_agent_iterate_regions(hsa_agent_t agent,
                                                hsa_status_t (*callback)(hsa_region_t, void *),
                                                void *data) {
  if (agent.handle != kHostAgent.handle || callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_AGENT;
  if (g_offer_coarse_report_region && g_offer_coarse_report_region_first) {
    const hsa_status_t coarse_status = callback(hsa_region_t{31}, data);
    if (coarse_status != HSA_STATUS_SUCCESS)
      return coarse_status;
  }
  if (g_offer_fine_report_region) {
    const hsa_status_t fine_status = callback(hsa_region_t{30}, data);
    if (fine_status != HSA_STATUS_SUCCESS)
      return fine_status;
  }
  if (g_offer_coarse_report_region && !g_offer_coarse_report_region_first)
    return callback(hsa_region_t{31}, data);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_guest_agent_iterate_regions(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_region_t, void *), void *data) {
  if (agent.handle != kGuestAgent.handle || callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_AGENT;
  return callback(hsa_region_t{30}, data);
}

hsa_status_t HSA_API fake_region_get_info(hsa_region_t region, hsa_region_info_t attribute,
                                          void *value) {
  if ((region.handle != 30 && region.handle != 31) || value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  switch (attribute) {
  case HSA_REGION_INFO_SEGMENT:
    *static_cast<hsa_region_segment_t *>(value) = HSA_REGION_SEGMENT_GLOBAL;
    return HSA_STATUS_SUCCESS;
  case HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED:
    *static_cast<bool *>(value) = true;
    return HSA_STATUS_SUCCESS;
  case HSA_REGION_INFO_ALLOC_MAX_SIZE:
    *static_cast<size_t *>(value) = kAutoReportProcessCeilingBytes;
    return HSA_STATUS_SUCCESS;
  case HSA_REGION_INFO_GLOBAL_FLAGS:
    *static_cast<uint32_t *>(value) = region.handle == 30 ? HSA_REGION_GLOBAL_FLAG_FINE_GRAINED
                                                          : HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED;
    return HSA_STATUS_SUCCESS;
  default:
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

hsa_status_t HSA_API fake_core_memory_allocate(hsa_region_t region, size_t size, void **ptr) {
  ++g_core_memory_allocate_calls;
  if ((region.handle != 30 && region.handle != 31) || ptr == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (g_fail_core_memory_allocate)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  void *allocation = nullptr;
  if (g_reuse_core_memory && g_recycled_core_memory != nullptr &&
      g_recycled_core_memory_size == size)
    allocation = std::exchange(g_recycled_core_memory, nullptr);
  else
    allocation = std::malloc(size);
  if (allocation == nullptr)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  *ptr = allocation;
  g_core_memory_allocation_sizes.push_back(size);
  g_core_memory_allocation_regions.push_back(region.handle);
  g_core_memory_allocations.push_back(allocation);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_core_memory_free(void *ptr) {
  if (ptr == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  ++g_core_memory_free_calls;
  const auto it = std::ranges::find(g_core_memory_allocations, ptr);
  if (it == g_core_memory_allocations.end())
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  const size_t index = static_cast<size_t>(it - g_core_memory_allocations.begin());
  const size_t freed_size = g_core_memory_allocation_sizes[index];
  if (g_core_memory_allocation_sizes[index] >= sizeof(ReportHeader))
    g_core_memory_headers_at_free.push_back(*static_cast<const ReportHeader *>(ptr));
  else
    g_sc_markers_at_free.push_back(*static_cast<const uint32_t *>(ptr));
  g_core_memory_allocation_sizes.erase(g_core_memory_allocation_sizes.begin() + index);
  g_core_memory_allocations.erase(it);
  if (g_reuse_core_memory) {
    std::free(g_recycled_core_memory);
    g_recycled_core_memory = ptr;
    g_recycled_core_memory_size = freed_size;
    std::memset(ptr, 0xa5, freed_size);
  } else {
    std::free(ptr);
  }
  return HSA_STATUS_SUCCESS;
}

void fake_runtime_reclaim_core_memory() {
  g_core_memory_runtime_reclaim_calls += static_cast<int>(g_core_memory_allocations.size());
  for (void *allocation : g_core_memory_allocations)
    std::free(allocation);
  g_core_memory_allocations.clear();
  g_core_memory_allocation_sizes.clear();
  g_core_memory_allocation_regions.clear();
}

hsa_status_t HSA_API fake_memory_assign_agent(void *, hsa_agent_t agent, hsa_access_permission_t) {
  return agent.handle == kHostAgent.handle ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_INVALID_AGENT;
}

TransformResult transform_override(std::span<const uint8_t> bytes, const Request &request,
                                   const TransformPolicy &transform_policy,
                                   const RuntimePolicy &runtime_policy, const DebugOverrides &debug,
                                   const MutationRequest &mutation,
                                   const RuntimeCapabilities &capabilities,
                                   const BoundRuntimeResources &resources) {
  const bool fault_mutation_enabled = mutation.has_fault_mutation();
  std::optional<TransformArtifacts> queued_result;
  {
    std::lock_guard lock(g_transform_observation_mutex);
    g_transform_override_flavors.push_back(*request.mode);
    g_transform_override_kernel_allowlists.push_back(request.kernel_name_allowlist);
    g_transform_override_abort_unmatched_waits.push_back(debug.abort_unmatched_barrier_wait);
    g_transform_override_track_barriers.push_back(request.track_barriers);
    g_transform_override_track_atomics.push_back(request.track_atomics);
    g_transform_override_fault_drop_barriers.push_back(mutation.fault_drop_barrier);
    g_transform_override_fault_mutations.push_back(fault_mutation_enabled);
    g_transform_override_fault_dry_runs.push_back(mutation.fault_dry_run);
    g_transform_override_patched_image_growth_limits.push_back(
        transform_policy.patched_image_growth_limit);
    g_transform_override_runtime_sample_strides.push_back(request.runtime_sample_stride);
    g_transform_override_sc_report_addresses.push_back(
        resources.supercollider_report_buffer_address);
    g_transform_override_report_sizes.push_back(resources.report_buffer_size);
    g_transform_override_report_layouts.push_back(resources.report_layout);
    if (!g_transform_override_results.empty()) {
      queued_result = std::move(g_transform_override_results.front());
      g_transform_override_results.pop_front();
    }
  }
  {
    std::unique_lock lock(g_transform_block_mutex);
    if (g_block_first_transform && !g_first_transform_entered) {
      g_first_transform_entered = true;
      g_transform_block_cv.notify_all();
      g_transform_block_cv.wait(lock, [] { return g_release_first_transform; });
    }
  }
  TransformArtifacts result =
      queued_result ? std::move(*queued_result) : g_transform_override_result;
  if (g_transform_override_models_fault_application && fault_mutation_enabled &&
      !mutation.fault_dry_run) {
    if (g_transform_override_live_fault_result)
      result = *g_transform_override_live_fault_result;
    if (g_reentrant_fault_load) {
      auto load = std::move(g_reentrant_fault_load);
      g_reentrant_fault_load = {};
      g_reentrant_fault_load_status = load();
    }
    std::unique_lock lock(g_fault_application_block_mutex);
    if (g_block_first_fault_application && !g_first_fault_application_entered) {
      g_first_fault_application_entered = true;
      g_fault_application_block_cv.notify_all();
      g_fault_application_block_cv.wait(lock, [] { return g_release_first_fault_application; });
      if (g_first_fault_application_result)
        result = *g_first_fault_application_result;
    }
  }
  if (g_transform_override_models_fault_application) {
    result.mutation.fault.planned = fault_mutation_enabled ? 1u : 0u;
    result.mutation.fault.applied = fault_mutation_enabled && !mutation.fault_dry_run
                                        ? g_transform_override_actual_fault_applications
                                        : 0u;
  }
  return TransformResultTestAccess::publish(bytes, request, transform_policy, runtime_policy, debug,
                                            mutation, capabilities, resources, std::move(result));
}

hsa_status_t HSA_API fake_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t reader, const char *,
    hsa_loaded_code_object_t *loaded_code_object) {
  {
    std::lock_guard lock(g_transform_observation_mutex);
    ++g_fake_load_agent_calls;
    g_last_load_agent = agent;
    g_last_load_reader = reader;
    g_loaded_code_object_readers.push_back(reader.handle);
  }
  {
    std::unique_lock lock(g_loader_block_mutex);
    const bool should_block = g_block_first_loader_call ||
                              (g_block_loader_reader && *g_block_loader_reader == reader.handle);
    if (should_block && !g_first_loader_call_entered) {
      g_first_loader_call_entered = true;
      g_loader_block_cv.notify_all();
      g_loader_block_cv.wait(lock, [] { return g_release_first_loader_call; });
    }
  }
  std::lock_guard observation_lock(g_transform_observation_mutex);
  if (g_fail_loader_once_for_reader == reader.handle) {
    g_fail_loader_once_for_reader.reset();
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  if (g_seed_auto_report_on_load) {
    g_seed_auto_report_on_load = false;
    if (g_core_memory_allocations.empty() || g_transform_override_report_layouts.empty() ||
        !g_transform_override_report_layouts.back()) {
      return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    }
    const ReportBufferLayout &layout = *g_transform_override_report_layouts.back();
    const uint32_t visible_count =
        g_seed_auto_pending_release_scale || g_seed_auto_conflict_pair ? 2u : 1u;
    if (layout.watchpoint_capacity < visible_count ||
        layout.causal_window_capacity < visible_count) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    auto *const report = static_cast<uint8_t *>(g_core_memory_allocations.back());
    auto *const header = reinterpret_cast<ReportHeader *>(report);
    header->causal_window_count = visible_count;
    auto *const windows = reinterpret_cast<CausalWindow *>(report + layout.causal_windows_offset);
    auto *const watchpoints = reinterpret_cast<uint64_t *>(report + layout.watchpoints_offset);
    for (uint32_t index = 0; index < visible_count; ++index) {
      windows[index] = {
          .generation = header->generation,
          .dispatch_id = 0x1122334455667788ull + (g_seed_auto_distinct_dispatches ? index : 0u),
          .workgroup_x = 3u,
          .workgroup_y = 4u,
          .workgroup_z = 5u,
          .epoch = 2u,
          .first_entry = index,
          .entry_count = 1u,
          .publication_state = static_cast<uint32_t>(CausalPublicationState::Ready),
          .cluster_workgroup_id = g_seed_auto_distinct_clusters ? index : 0u,
      };
      watchpoints[index] = pack_watchpoint_entry(
          ShadowAccessKind::Write,
          /*owner_id=*/7u + (g_seed_auto_conflict_pair ? index : 0u), /*epoch=*/2u,
          static_cast<uint32_t>(header->generation), /*start_cell=*/9u, /*cell_count=*/2u);
    }
    if (g_seed_auto_pending_release_scale) {
      const uint32_t owner_bank_count = pending_acquire_owner_bank_count(
          layout.pending_acquire_capacity, layout.causal_window_capacity);
      constexpr uint32_t kAcquireSlot = 1u;
      if (owner_bank_count == 0u)
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
      const uint32_t pending_index =
          kAcquireSlot * owner_bank_count + (7u & (owner_bank_count - 1u));
      if (pending_index >= layout.pending_acquire_capacity)
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
      const auto encoded = encode_sync_metadata({
          .address = 0x123456780000ull,
          .byte_count = 4u,
          .kind = SyncMetadataKind::Atomic,
          .role = SyncRole::RmwAcquireRelease,
          .scope = SyncScope::Agent,
          .outcome = SyncOutcome::RmwReturnsOld,
          .epoch_before = 2u,
          .epoch_after = 2u,
      });
      if (encoded.classification != SyncClassification::Valid)
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      auto *const pending =
          reinterpret_cast<PendingAcquireSlot *>(report + layout.pending_acquires_offset);
      pending[pending_index] = {
          .version = 2u,
          .selected_slot = kAcquireSlot,
          .generation = header->generation,
          .dispatch_id = 0x1122334455667788ull,
          .workgroup_x = 3u,
          .workgroup_y = 4u,
          .workgroup_z = 5u,
          .owner_id = 7u,
          .source_epoch = 2u,
          .reserved = 1u,
          .metadata = encoded.packed,
      };
      header->pending_acquire_count = 1u;
    }
    if (g_seed_auto_pending_identity_collision) {
      const uint32_t owner_bank_count = pending_acquire_owner_bank_count(
          layout.pending_acquire_capacity, layout.causal_window_capacity);
      if (owner_bank_count == 0u)
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
      const uint32_t pending_index = 7u & (owner_bank_count - 1u);
      if (pending_index >= layout.pending_acquire_capacity)
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
      const auto encoded = encode_sync_metadata({
          .address = 0x123456780000ull,
          .byte_count = 4u,
          .kind = SyncMetadataKind::Atomic,
          .role = SyncRole::RmwAcquire,
          .scope = SyncScope::Agent,
          .outcome = SyncOutcome::RmwReturnsOld,
          .epoch_before = 2u,
          .epoch_after = 2u,
      });
      if (encoded.classification != SyncClassification::Valid)
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      auto *const pending =
          reinterpret_cast<PendingAcquireSlot *>(report + layout.pending_acquires_offset);
      pending[pending_index] = {
          .version = 2u,
          .selected_slot = 0u,
          .generation = header->generation,
          .dispatch_id = 0x1122334455667788ull,
          // Same hash slot and owner bank, but a different workgroup: this is
          // an expected table collision, not malformed evidence for window 0.
          .workgroup_x = 99u,
          .workgroup_y = 4u,
          .workgroup_z = 5u,
          .owner_id = 7u,
          .source_epoch = 2u,
          .metadata = encoded.packed,
      };
      header->pending_acquire_count = 1u;
    }
    g_seed_auto_report_succeeded = true;
  }
  if (loaded_code_object != nullptr)
    loaded_code_object->handle = 77;
  g_loaded_executable_readers.emplace_back(executable.handle, reader.handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_executable_get_symbol_by_name(hsa_executable_t, const char *symbol_name,
                                                        const hsa_agent_t *,
                                                        hsa_executable_symbol_t *symbol) {
  if (symbol_name == nullptr || symbol == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (symbol_name != g_fake_symbol_name)
    return HSA_STATUS_ERROR_INVALID_SYMBOL_NAME;
  *symbol = kFakeKernelSymbol;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_executable_get_symbol(hsa_executable_t, const char *,
                                                const char *symbol_name, hsa_agent_t, int32_t,
                                                hsa_executable_symbol_t *symbol) {
  if (symbol_name == nullptr || symbol == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (symbol_name != g_fake_symbol_name)
    return HSA_STATUS_ERROR_INVALID_SYMBOL_NAME;
  *symbol = kFakeKernelSymbol;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_executable_iterate_symbols(
    hsa_executable_t executable,
    hsa_status_t (*callback)(hsa_executable_t, hsa_executable_symbol_t, void *), void *data) {
  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  return callback(executable, kFakeKernelSymbol, data);
}

hsa_status_t HSA_API fake_executable_iterate_agent_symbols(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t, void *),
    void *data) {
  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  return callback(executable, agent, kFakeKernelSymbol, data);
}

hsa_status_t HSA_API fake_executable_symbol_get_info(hsa_executable_symbol_t symbol,
                                                     hsa_executable_symbol_info_t attribute,
                                                     void *value) {
  if (symbol.handle != kFakeKernelSymbol.handle || value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH) {
    const uint32_t name_length = static_cast<uint32_t>(g_fake_symbol_name.size());
    std::memcpy(value, &name_length, sizeof(name_length));
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_NAME) {
    std::memcpy(value, g_fake_symbol_name.data(), g_fake_symbol_name.size());
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT) {
    std::memcpy(value, &g_fake_symbol_kernel_object, sizeof(g_fake_symbol_kernel_object));
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE) {
    std::memcpy(value, &g_fake_symbol_group_segment_size, sizeof(g_fake_symbol_group_segment_size));
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE) {
    std::memcpy(value, &g_fake_symbol_private_segment_size,
                sizeof(g_fake_symbol_private_segment_size));
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size,
                                                   uint32_t, void **ptr) {
  g_last_allocate_pool = memory_pool;
  if (ptr != nullptr) {
    g_fake_allocations.emplace_back(size == 0 ? 1 : size);
    g_fake_allocation_pools.push_back(memory_pool);
    g_fake_allocation_sizes.push_back(size);
    *ptr = g_fake_allocations.back().data();
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_memory_pool_free(void *ptr) {
  g_fake_freed_allocations.push_back(ptr);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_agents_allow_access(uint32_t, const hsa_agent_t *, const uint32_t *,
                                                  const void *) {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_memory_async_batch_copy(const hsa_amd_memory_copy_op_t *copy_ops,
                                                      uint32_t num_copy_ops, uint32_t,
                                                      const hsa_signal_t *) {
  g_last_batch_src_agents.clear();
  g_last_batch_dst_agents.clear();
  if (copy_ops == nullptr && num_copy_ops != 0)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  for (uint32_t op_idx = 0; op_idx < num_copy_ops; ++op_idx) {
    const hsa_amd_memory_copy_op_t &op = copy_ops[op_idx];
    switch (static_cast<hsa_amd_memory_copy_op_type_t>(op.type)) {
    case HSA_AMD_MEMORY_COPY_OP_LINEAR:
    case HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP:
    case HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRC:
    case HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_DST:
    case HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRCDST:
      if (op.num_entries == 0) {
        g_last_batch_src_agents.push_back(op.src_agent.handle);
        g_last_batch_dst_agents.push_back(op.dst_agent.handle);
        continue;
      }
      if (op.dst_agent_list == nullptr)
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      for (uint16_t entry_idx = 0; entry_idx < op.num_entries; ++entry_idx) {
        g_last_batch_src_agents.push_back(op.src_agent.handle);
        g_last_batch_dst_agents.push_back(op.dst_agent_list[entry_idx].handle);
      }
      continue;
    case HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST:
      if (op.num_entries == 0 || op.dst_agent_list == nullptr)
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      for (uint16_t entry_idx = 0; entry_idx < op.num_entries; ++entry_idx) {
        g_last_batch_src_agents.push_back(op.src_agent.handle);
        g_last_batch_dst_agents.push_back(op.dst_agent_list[entry_idx].handle);
      }
      continue;
    }
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_memory_lock(void *, size_t, hsa_agent_t *agents, int num_agent,
                                          void **) {
  g_last_memory_lock_agents.clear();
  if (agents == nullptr && num_agent != 0)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (int i = 0; i < num_agent; ++i)
    g_last_memory_lock_agents.push_back(agents[i].handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_memory_lock_to_pool(void *, size_t, hsa_agent_t *agents,
                                                  int num_agent, hsa_amd_memory_pool_t pool,
                                                  uint32_t, void **) {
  g_last_memory_lock_to_pool_pool = pool;
  g_last_memory_lock_to_pool_agents.clear();
  if (agents == nullptr && num_agent != 0)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (int i = 0; i < num_agent; ++i)
    g_last_memory_lock_to_pool_agents.push_back(agents[i].handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_vmem_set_access(void *, size_t,
                                              const hsa_amd_memory_access_desc_t *desc,
                                              size_t desc_cnt) {
  g_last_vmem_access_agents.clear();
  if (desc == nullptr && desc_cnt != 0)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (size_t i = 0; i < desc_cnt; ++i)
    g_last_vmem_access_agents.push_back(desc[i].agent_handle.handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_pointer_info(const void *, hsa_amd_pointer_info_t *info,
                                           void *(*)(size_t), uint32_t *num_agents_accessible,
                                           hsa_agent_t **accessible) {
  if (info != nullptr) {
    info->size = sizeof(hsa_amd_pointer_info_t);
    info->agentOwner = kHostAgent;
  }
  if (num_agents_accessible != nullptr && accessible != nullptr) {
    g_pointer_info_accessible[0] = kHostAgent;
    g_pointer_info_accessible[1] = kGuestAgent;
    *num_agents_accessible = 2;
    *accessible = g_pointer_info_accessible;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_agent_iterate_memory_pools(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t, void *), void *data) {
  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (agent.handle == kGuestAgent.handle) {
    if (g_fail_guest_pool_iteration_once) {
      g_fail_guest_pool_iteration_once = false;
      return HSA_STATUS_ERROR;
    }
    std::unique_lock lock(g_pool_mutex);
    if (g_block_guest_pool_iteration) {
      g_guest_pool_iteration_entered = true;
      g_pool_cv.notify_all();
      g_pool_cv.wait(lock, [] { return g_release_guest_pool_iteration; });
    }
    lock.unlock();
    return callback(kGuestPool, data);
  }
  if (agent.handle == kHostAgent.handle) {
    hsa_status_t status = callback(kHostPool, data);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    return callback(kHostKernargPool, data);
  }
  if (agent.handle == kCpuAgent.handle)
    return callback(kCpuFineGrainedPool, data);
  return HSA_STATUS_ERROR_INVALID_AGENT;
}

hsa_status_t HSA_API fake_amd_memory_pool_get_info(hsa_amd_memory_pool_t pool,
                                                   hsa_amd_memory_pool_info_t attribute,
                                                   void *value) {
  if (value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (attribute == HSA_AMD_MEMORY_POOL_INFO_SEGMENT) {
    *static_cast<uint32_t *>(value) = 0;
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS) {
    *static_cast<uint32_t *>(value) = pool.handle == kHostKernargPool.handle ? 1u : 2u;
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED) {
    *static_cast<bool *>(value) = true;
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_AMD_MEMORY_POOL_INFO_LOCATION) {
    *static_cast<uint32_t *>(value) = 0;
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t HSA_API fake_amd_agent_memory_pool_get_info(hsa_agent_t agent,
                                                         hsa_amd_memory_pool_t memory_pool,
                                                         hsa_amd_agent_memory_pool_info_t attribute,
                                                         void *value) {
  ++g_agent_memory_pool_get_info_calls;
  g_last_agent_memory_pool_agent = agent;
  g_last_agent_memory_pool = memory_pool;

  if (value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (memory_pool.handle == 0)
    return HSA_STATUS_SUCCESS;
  if (attribute == HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS) {
    *static_cast<uint32_t *>(value) = 0;
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_agent_t g_last_agent_preload_agent{};
hsa_agent_t g_last_async_scratch_limit_agent{};

hsa_status_t HSA_API fake_amd_agent_preload(hsa_agent_t agent, uint64_t) {
  g_last_agent_preload_agent = agent;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_agent_set_async_scratch_limit(hsa_agent_t agent, size_t) {
  g_last_async_scratch_limit_agent = agent;
  return HSA_STATUS_SUCCESS;
}

struct FakeApiTable {
  CoreApiTable core{};
  AmdExtTable amd{};
  HsaApiTable table{};

  FakeApiTable() {
    core.version.minor_id = sizeof(CoreApiTable);
    amd.version.minor_id = sizeof(AmdExtTable);
    table.version.minor_id = sizeof(HsaApiTable);
    table.core_ = &core;
    table.amd_ext_ = &amd;

    core.hsa_shut_down_fn = fake_shut_down;
    core.hsa_iterate_agents_fn = fake_iterate_agents;
    core.hsa_agent_get_info_fn = fake_agent_get_info;
    core.hsa_agent_iterate_isas_fn = fake_agent_iterate_isas;
    core.hsa_isa_get_info_alt_fn = fake_isa_get_info_alt;
    core.hsa_queue_create_fn = fake_queue_create;
    core.hsa_queue_destroy_fn = fake_queue_destroy;
    core.hsa_signal_create_fn = fake_signal_create;
    core.hsa_signal_destroy_fn = fake_signal_destroy;
    core.hsa_signal_load_scacquire_fn = fake_signal_load_scacquire;
    core.hsa_signal_store_relaxed_fn = fake_signal_store_relaxed;
    core.hsa_signal_store_screlease_fn = fake_signal_store_screlease;
    core.hsa_signal_wait_relaxed_fn = fake_signal_wait_relaxed;
    core.hsa_signal_wait_scacquire_fn = fake_signal_wait_scacquire;
    core.hsa_code_object_reader_create_from_file_fn = fake_code_object_reader_create_from_file;
    core.hsa_code_object_reader_create_from_memory_fn = fake_code_object_reader_create_from_memory;
    core.hsa_code_object_reader_destroy_fn = fake_code_object_reader_destroy;
    core.hsa_system_get_extension_table_fn = fake_system_get_extension_table;
    core.hsa_system_get_major_extension_table_fn = fake_system_get_major_extension_table;
    core.hsa_executable_load_agent_code_object_fn = fake_executable_load_agent_code_object;
    core.hsa_executable_destroy_fn = fake_executable_destroy;
    core.hsa_executable_get_symbol_fn = fake_executable_get_symbol;
    core.hsa_executable_get_symbol_by_name_fn = fake_executable_get_symbol_by_name;
    core.hsa_executable_symbol_get_info_fn = fake_executable_symbol_get_info;
    core.hsa_agent_iterate_regions_fn = fake_agent_iterate_regions;
    core.hsa_region_get_info_fn = fake_region_get_info;
    core.hsa_memory_allocate_fn = fake_core_memory_allocate;
    core.hsa_memory_free_fn = fake_core_memory_free;
    core.hsa_memory_assign_agent_fn = fake_memory_assign_agent;
    core.hsa_executable_iterate_symbols_fn = fake_executable_iterate_symbols;
    core.hsa_executable_iterate_agent_symbols_fn = fake_executable_iterate_agent_symbols;
    amd.hsa_amd_agent_iterate_memory_pools_fn = fake_amd_agent_iterate_memory_pools;
    amd.hsa_amd_memory_pool_get_info_fn = fake_amd_memory_pool_get_info;
    amd.hsa_amd_agent_memory_pool_get_info_fn = fake_amd_agent_memory_pool_get_info;
    amd.hsa_amd_memory_pool_allocate_fn = fake_amd_memory_pool_allocate;
    amd.hsa_amd_memory_async_batch_copy_fn = fake_amd_memory_async_batch_copy;
    amd.hsa_amd_memory_lock_fn = fake_amd_memory_lock;
    amd.hsa_amd_memory_lock_to_pool_fn = fake_amd_memory_lock_to_pool;
    amd.hsa_amd_pointer_info_fn = fake_amd_pointer_info;
    amd.hsa_amd_vmem_set_access_fn = fake_amd_vmem_set_access;
    amd.hsa_amd_queue_intercept_create_fn = fake_queue_intercept_create;
    amd.hsa_amd_queue_intercept_register_fn = fake_queue_intercept_register;
    amd.hsa_amd_queue_create_fn = fake_amd_queue_create;
    amd.hsa_amd_memory_pool_free_fn = fake_amd_memory_pool_free;
    amd.hsa_amd_agents_allow_access_fn = fake_amd_agents_allow_access;
    amd.hsa_amd_agent_preload_fn = fake_amd_agent_preload;
    amd.hsa_amd_agent_set_async_scratch_limit_fn = fake_amd_agent_set_async_scratch_limit;
  }
};

void reset_core_memory_observations();

TEST(HsaHooksUnitTest, SharedAutoReportAllocationReleasesFailedAgentAssignment) {
  reset_core_memory_observations();
  FakeApiTable api;
  api.core.hsa_agent_iterate_regions_fn = fake_guest_agent_iterate_regions;

  const rocjitsu::consan::hook::detail::AutoReportAllocation allocation =
      rocjitsu::consan::hook::detail::allocate_auto_report_memory(api.table.core_, kGuestAgent,
                                                                  sizeof(uint32_t));

  EXPECT_FALSE(allocation);
  EXPECT_STREQ(allocation.failure_reason, "hsa_memory_assign_agent");
  EXPECT_EQ(allocation.status, HSA_STATUS_ERROR_INVALID_AGENT);
  EXPECT_EQ(g_core_memory_allocate_calls, 1);
  EXPECT_EQ(g_core_memory_free_calls, 1);
  EXPECT_TRUE(g_core_memory_allocations.empty());
}

void write_runtime_config_path(const std::string &runtime_dir,
                               bool create_host_topology_node = true) {
  setenv("ROCJITSU_RUNTIME_DIR", runtime_dir.c_str(), 1);

  const std::filesystem::path topology_root =
      std::filesystem::path(runtime_dir) / "topology" / "nodes";
  std::filesystem::create_directories(topology_root);
  if (create_host_topology_node) {
    const std::filesystem::path host_node = topology_root / std::to_string(kHostNodeId);
    std::filesystem::create_directories(host_node);
    std::ofstream(host_node / "gpu_id") << kResolvedHostGpuId << '\n';
  }
  rj_hsa_dbt_set_topology_nodes_root_for_test(topology_root.c_str());

  std::ofstream config_path(rocjitsu::rpc_default_config_file_path());
  config_path << RJ_HOOK_UNIT_CONFIG_PATH << '\n' << kResolvedHostGpuId << '\n';
}

class InstalledHook {
public:
  explicit InstalledHook(FakeApiTable &api, bool create_host_topology_node = true)
      : runtime_dir_("rocjitsu-hsa-hooks-unit-") {
    OnUnload();
    write_runtime_config_path(runtime_dir_.path(), create_host_topology_node);
    installed_ = OnLoad(&api.table, 0, 0, nullptr);
  }
  ~InstalledHook() {
    OnUnload();
    rj_hsa_dbt_set_topology_nodes_root_for_test(nullptr);
  }

  [[nodiscard]] bool installed() const { return installed_; }

private:
  rocjitsu::test::ScopedTempDirectory runtime_dir_;
  bool installed_ = false;
};

class ScopedEnvVar {
public:
  ScopedEnvVar(const char *name, const char *value) : name_(name) {
    if (const char *old = std::getenv(name); old != nullptr)
      old_ = old;
    if (value != nullptr)
      setenv(name, value, 1);
    else
      unsetenv(name);
  }
  ~ScopedEnvVar() {
    if (old_)
      setenv(name_.c_str(), old_->c_str(), 1);
    else
      unsetenv(name_.c_str());
  }

private:
  std::string name_;
  std::optional<std::string> old_;
};

class InstalledDbiHook {
public:
  explicit InstalledDbiHook(FakeApiTable &api) : runtime_dir_("rocjitsu-hsa-dbi-hooks-unit-") {
    write_runtime_config_path(runtime_dir_.path());
    const std::filesystem::path executable = std::filesystem::canonical("/proc/self/exe");
    const std::filesystem::path library =
        executable.parent_path().parent_path() /
        "lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so";
    library_ = dlopen(library.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (library_ == nullptr) {
      error_ = dlerror();
      return;
    }
    on_load_ = reinterpret_cast<OnLoadFn>(dlsym(library_, "OnLoad"));
    on_unload_ = reinterpret_cast<OnUnloadFn>(dlsym(library_, "OnUnload"));
    set_override_ = reinterpret_cast<SetOverrideFn>(
        dlsym(library_, "rj_dbi_test_set_consan_transform_override"));
    set_log_sink_override_ = reinterpret_cast<SetLogSinkOverrideFn>(
        dlsym(library_, "rj_dbi_test_set_log_sink_override"));
    retry_count_ =
        reinterpret_cast<RetryCountFn>(dlsym(library_, "rj_dbi_test_consan_retry_count"));
    instrumentation_nanoseconds_ = reinterpret_cast<InstrumentationNanosecondsFn>(
        dlsym(library_, "rj_dbi_consan_instrumentation_nanoseconds"));
    checkpoint_after_device_synchronize_ = reinterpret_cast<CheckpointAfterDeviceSynchronizeFn>(
        dlsym(library_, "rj_dbi_consan_checkpoint_after_device_synchronize"));
    begin_epoch_analysis_window_ = reinterpret_cast<EpochAnalysisWindowFn>(
        dlsym(library_, "rj_dbi_consan_begin_epoch_analysis_window"));
    end_epoch_analysis_window_ = reinterpret_cast<EpochAnalysisWindowFn>(
        dlsym(library_, "rj_dbi_consan_end_epoch_analysis_window"));
    if (on_load_ == nullptr || on_unload_ == nullptr || set_override_ == nullptr ||
        set_log_sink_override_ == nullptr || retry_count_ == nullptr ||
        instrumentation_nanoseconds_ == nullptr ||
        checkpoint_after_device_synchronize_ == nullptr ||
        begin_epoch_analysis_window_ == nullptr || end_epoch_analysis_window_ == nullptr) {
      error_ = dlerror();
      return;
    }
    on_unload_();
    set_override_(transform_override);
    installed_ = on_load_(&api.table, 0, 0, nullptr);
    needs_unload_ = true;
    if (!installed_)
      error_ = "DBI OnLoad returned false";
  }
  ~InstalledDbiHook() {
    unload();
    if (set_log_sink_override_ != nullptr)
      set_log_sink_override_(nullptr);
    if (set_override_ != nullptr)
      set_override_(nullptr);
    if (library_ != nullptr)
      dlclose(library_);
    rj_hsa_dbt_set_topology_nodes_root_for_test(nullptr);
  }

  [[nodiscard]] bool installed() const { return installed_; }
  [[nodiscard]] const std::string &error() const { return error_; }
  [[nodiscard]] size_t retry_count() const { return retry_count_(); }
  [[nodiscard]] uint64_t instrumentation_nanoseconds() const {
    return instrumentation_nanoseconds_();
  }
  [[nodiscard]] uint32_t checkpoint_after_device_synchronize() const {
    return checkpoint_after_device_synchronize_();
  }
  [[nodiscard]] uint32_t begin_epoch_analysis_window() const {
    return begin_epoch_analysis_window_();
  }
  [[nodiscard]] uint32_t end_epoch_analysis_window() const { return end_epoch_analysis_window_(); }
  void use_production_transform() { set_override_(nullptr); }
  void set_log_sink_override(rocjitsu::consan::hook::LogSinkOverride sink) {
    set_log_sink_override_(sink);
  }
  void unload() {
    if (on_unload_ != nullptr && needs_unload_) {
      on_unload_();
      // A real HSA runtime reclaims any remaining runtime allocations after
      // the tool's OnUnload callback returns. Model that separately from the
      // callable hsa_memory_free API so shutdown tests can detect re-entry.
      fake_runtime_reclaim_core_memory();
      needs_unload_ = false;
      installed_ = false;
    }
  }
  void invoke_unload_again_for_test() {
    if (on_unload_ != nullptr)
      on_unload_();
  }
  [[nodiscard]] bool reload(FakeApiTable &api) {
    if (on_load_ == nullptr || set_override_ == nullptr || needs_unload_)
      return false;
    error_.clear();
    set_override_(transform_override);
    installed_ = on_load_(&api.table, 0, 0, nullptr);
    needs_unload_ = true;
    if (!installed_)
      error_ = "DBI OnLoad returned false";
    return installed_;
  }

private:
  using OnLoadFn = bool (*)(HsaApiTable *, uint64_t, uint64_t, const char *const *);
  using OnUnloadFn = void (*)();
  using SetOverrideFn = void (*)(rocjitsu::consan::hook::TransformOverride);
  using SetLogSinkOverrideFn = void (*)(rocjitsu::consan::hook::LogSinkOverride);
  using RetryCountFn = size_t (*)();
  using InstrumentationNanosecondsFn = uint64_t (*)();
  using CheckpointAfterDeviceSynchronizeFn = uint32_t (*)();
  using EpochAnalysisWindowFn = uint32_t (*)();
  rocjitsu::test::ScopedTempDirectory runtime_dir_;
  void *library_ = nullptr;
  OnLoadFn on_load_ = nullptr;
  OnUnloadFn on_unload_ = nullptr;
  SetOverrideFn set_override_ = nullptr;
  SetLogSinkOverrideFn set_log_sink_override_ = nullptr;
  RetryCountFn retry_count_ = nullptr;
  InstrumentationNanosecondsFn instrumentation_nanoseconds_ = nullptr;
  CheckpointAfterDeviceSynchronizeFn checkpoint_after_device_synchronize_ = nullptr;
  EpochAnalysisWindowFn begin_epoch_analysis_window_ = nullptr;
  EpochAnalysisWindowFn end_epoch_analysis_window_ = nullptr;
  bool installed_ = false;
  bool needs_unload_ = false;
  std::string error_;
};

struct ConSanHookProfile {
  const char *name;
  const char *mode;
  Mode expected_flavor;
};

constexpr std::array kConSanHookProfiles = {
    ConSanHookProfile{"supercollider", "supercollider", Mode::SuperCollider},
    ConSanHookProfile{"default", "default", Mode::Default},
};

void expect_transform_profile(const ConSanHookProfile &profile, size_t expected_calls = 1u);

void reset_code_object_observations() {
  g_code_object_reader_create_calls = 0;
  g_fail_replacement_reader_create = false;
  g_code_object_reader_inputs.clear();
  g_memory_code_object_readers.clear();
  g_destroyed_code_object_readers.clear();
  g_destroyed_executables.clear();
  g_replacement_storage_valid_by_executable.clear();
  g_loaded_code_object_readers.clear();
  g_loaded_executable_readers.clear();
  g_transform_override_flavors.clear();
  g_transform_override_kernel_allowlists.clear();
  g_transform_override_abort_unmatched_waits.clear();
  g_transform_override_track_barriers.clear();
  g_transform_override_track_atomics.clear();
  g_transform_override_fault_drop_barriers.clear();
  g_transform_override_fault_mutations.clear();
  g_transform_override_fault_dry_runs.clear();
  g_transform_override_patched_image_growth_limits.clear();
  g_transform_override_models_fault_application = false;
  g_transform_override_actual_fault_applications = 1;
  g_transform_override_live_fault_result.reset();
  {
    std::lock_guard lock(g_transform_block_mutex);
    g_block_first_transform = false;
    g_first_transform_entered = false;
    g_release_first_transform = false;
  }
  {
    std::lock_guard lock(g_fault_application_block_mutex);
    g_block_first_fault_application = false;
    g_first_fault_application_entered = false;
    g_release_first_fault_application = false;
    g_first_fault_application_result.reset();
  }
  {
    std::lock_guard lock(g_loader_block_mutex);
    g_block_first_loader_call = false;
    g_block_loader_reader.reset();
    g_first_loader_call_entered = false;
    g_release_first_loader_call = false;
  }
  g_fail_loader_once_for_reader.reset();
  g_reentrant_fault_load = {};
  g_reentrant_fault_load_status.reset();
  g_transform_override_runtime_sample_strides.clear();
  g_transform_override_sc_report_addresses.clear();
  g_transform_override_report_sizes.clear();
  g_transform_override_report_layouts.clear();
  g_transform_override_results.clear();
  g_seed_auto_report_on_load = false;
  g_seed_auto_report_succeeded = false;
  g_seed_auto_pending_release_scale = false;
  g_seed_auto_pending_identity_collision = false;
  g_seed_auto_conflict_pair = false;
  g_seed_auto_distinct_dispatches = false;
  g_seed_auto_distinct_clusters = false;
  g_transform_override_result = {};
}

void reset_core_memory_observations() {
  ASSERT_TRUE(g_core_memory_allocations.empty());
  g_fail_core_memory_allocate = false;
  g_offer_fine_report_region = true;
  g_offer_coarse_report_region = false;
  g_offer_coarse_report_region_first = false;
  g_core_memory_allocate_calls = 0;
  g_core_memory_free_calls = 0;
  g_core_memory_runtime_reclaim_calls = 0;
  g_core_memory_allocation_sizes.clear();
  g_core_memory_allocation_regions.clear();
  g_core_memory_headers_at_free.clear();
  g_sc_markers_at_free.clear();
}

void configure_consan_profile(const ConSanHookProfile &profile, bool fail_closed) {
  setenv("RJ_CONSAN_MODE", profile.mode, 1);
  unsetenv("RJ_CONSAN_POLICY");
  setenv("RJ_CONSAN_FAIL_CLOSED", fail_closed ? "1" : "0", 1);
  unsetenv("RJ_CONSAN_ABORT_UNMATCHED_BARRIER_WAIT");
  unsetenv("RJ_CONSAN_TRACK_BARRIERS");
  unsetenv("RJ_CONSAN_TRACK_ATOMICS");
  unsetenv("RJ_CONSAN_SC_REPORT_MODE");
  unsetenv("RJ_CONSAN_SC_REPORT_BUFFER");
  unsetenv("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES");
  unsetenv("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT");
  unsetenv("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES");
  unsetenv("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES");
  unsetenv("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES");
  if (profile.expected_flavor == Mode::Default) {
    setenv("RJ_CONSAN_REPORT_BUFFER", "4096", 1);
    setenv("RJ_CONSAN_REPORT_BUFFER_SIZE", "65536", 1);
    setenv("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "0", 1);
  } else {
    unsetenv("RJ_CONSAN_REPORT_BUFFER");
    unsetenv("RJ_CONSAN_REPORT_BUFFER_SIZE");
    unsetenv("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE");
  }
}

uint64_t transform_test_reservation_bytes(uint64_t input_bytes,
                                          const PatchedImageGrowthLimit &growth_limit) {
  const auto estimate =
      rocjitsu::consan::hook::transform_major_image_reservation(input_bytes, growth_limit);
  if (!estimate) {
    ADD_FAILURE() << "test transform reservation unexpectedly overflowed";
    return 0;
  }
  return estimate->reservation_bytes;
}

uint64_t absolute_transform_test_reservation_bytes(uint64_t input_bytes, uint64_t growth_bytes) {
  const PatchedImageGrowthLimit growth_limit = {
      .kind = PatchedImageGrowthLimitKind::AbsoluteBytes,
      .absolute_bytes = growth_bytes,
  };
  return transform_test_reservation_bytes(input_bytes, growth_limit);
}

void install_test_access_coverage(TransformArtifacts &result, size_t site_count,
                                  SiteDecisionKind decision_kind, AccessPolicyReason reason,
                                  LoweringOutcomeKind lowering = LoweringOutcomeKind::Pending,
                                  Mode mode = Mode::Default,
                                  ProbeIntentKind intent_kind = ProbeIntentKind::Access) {
  ObservationPlan plan;
  plan.mode = mode;
  plan.site_decisions.reserve(site_count);
  if (decision_kind == SiteDecisionKind::Admitted)
    plan.probe_intents.reserve(site_count);
  for (size_t index = 0; index < site_count; ++index) {
    const PhysicalSiteId physical{
        .code_object = result.program_inventory.code_object_id(),
        .original_text_offset = index * sizeof(uint32_t),
    };
    const SemanticSiteId semantic{
        .physical = physical,
        .domain = SemanticSiteDomain::Access,
        .member_ordinal = 0u,
        .range_ordinal = 0u,
    };
    if (decision_kind == SiteDecisionKind::Admitted) {
      const ProbeIntentId id{static_cast<uint32_t>(plan.probe_intents.size())};
      plan.probe_intents.push_back({
          .id = id,
          .mode = mode,
          .source_site = {static_cast<uint32_t>(index)},
          .physical_site = physical,
          .covered_semantic_sites = {semantic},
          .kind = intent_kind,
          .position = ProbePosition::Before,
          .synchronization_association = std::nullopt,
          .dynamic_result = DynamicResultRequirement::None,
          .atomic_lowering_form = std::nullopt,
      });
    }
    plan.site_decisions.push_back({
        .semantic_site = semantic,
        .kind = decision_kind,
        .reason = reason,
    });
  }
  ASSERT_TRUE(plan.valid());
  result.coverage_ledger = CoverageLedger(std::move(plan));
  for (const ProbeIntent &intent : result.observation_plan().probe_intents) {
    ASSERT_TRUE(publish_test_lowering_outcome(result.coverage_ledger, intent.id, lowering));
  }
}

TransformArtifacts process_growth_replacement_result(size_t replacement_size = 12) {
  assert(replacement_size >= 4);
  TransformArtifacts result;
  install_consan_test_program_identity(result, ROCJITSU_CODE_ARCH_CDNA3,
                                       ROCJITSU_CODE_TARGET_GFX942);
  result.outcome = TransformOutcome::ModifiedValid;
  result.replacement.resize(replacement_size);
  std::ranges::copy(std::array<uint8_t, 4>{0x7f, 'E', 'L', 'F'}, result.replacement.begin());
  ObservationPlan plan;
  plan.mode = Mode::Default;
  assert(plan.valid());
  result.coverage_ledger = CoverageLedger(std::move(plan));
  return result;
}

TEST(HsaHooksUnitTest, ConSanProcessExitFinalizesWithoutRuntimeUnload) {
  EXPECT_EXIT(
      {
        ScopedEnvVar mode("RJ_CONSAN_MODE", "supercollider");
        FakeApiTable api;
        const int saved_stderr = dup(STDERR_FILENO);
        FILE *quiet = std::fopen("/dev/null", "w");
        dup2(fileno(quiet), STDERR_FILENO);
        InstalledDbiHook hook(api);
        std::fflush(stderr);
        dup2(saved_stderr, STDERR_FILENO);
        close(saved_stderr);
        std::fclose(quiet);
        if (!hook.installed())
          std::_Exit(2);
        // std::exit deliberately skips the fixture's automatic OnUnload.
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "ConSan analysis verdict");
}

TEST(HsaHooksUnitTest, ConSanExitFinalizationRetainsCachedRuntimeCallbacks) {
  EXPECT_EXIT(
      {
        ScopedEnvVar mode("RJ_CONSAN_MODE", "supercollider");
        static FakeApiTable *api = nullptr;
        static InstalledDbiHook *hook = nullptr;
        // Run fixture cleanup after both the tool and the cached HIP callback.
        // Keeping these objects alive until then must not leak them at exit.
        std::atexit([] {
          delete hook;
          delete api;
          hook = nullptr;
          api = nullptr;
        });
        // Model a HIP exit handler registered before HSA loads the tool.
        static decltype(hsa_executable_destroy) *cached_destroy = nullptr;
        std::atexit([] {
          if (cached_destroy(hsa_executable_t{7}) != HSA_STATUS_SUCCESS)
            std::_Exit(3);
        });
        api = new FakeApiTable;
        hook = new InstalledDbiHook(*api);
        if (!hook->installed())
          std::_Exit(2);
        cached_destroy = api->core.hsa_executable_destroy_fn;
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "ConSan analysis verdict");
}

TEST(HsaHooksUnitTest, ConSanRepeatedUnloadDoesNotEmitAnotherVerdict) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "supercollider");
  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();
  testing::internal::CaptureStderr();
  hook.unload();
  hook.invoke_unload_again_for_test();
  const std::string log = testing::internal::GetCapturedStderr();
  const auto first = log.find("ConSan analysis verdict");
  ASSERT_NE(first, std::string::npos) << log;
  EXPECT_EQ(log.find("ConSan analysis verdict", first + 1), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, ConSanLoadedWithoutConfigurationDefaultsTo) {
  for (const char *selection : {static_cast<const char *>(nullptr), ""}) {
    SCOPED_TRACE(selection ? "empty mode" : "unset mode");
    ScopedEnvVar log_level("RJ_CONSAN_LOG", "3");
    ScopedEnvVar mode("RJ_CONSAN_MODE", selection);
    ScopedEnvVar policy("RJ_CONSAN_POLICY", nullptr);
    ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
    ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
    ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "0");
    ScopedEnvVar absolute_growth_limit("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES", nullptr);
    ScopedEnvVar relative_growth_limit("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT", nullptr);
    ScopedEnvVar process_transform_limit("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES",
                                         nullptr);
    ScopedEnvVar process_image_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES", nullptr);
    ScopedEnvVar process_growth_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES", nullptr);

    reset_code_object_observations();
    TransformArtifacts unchanged;
    unchanged.outcome = TransformOutcome::Unchanged;
    g_transform_override_result = unchanged;

    FakeApiTable api;
    const auto original_load = api.core.hsa_executable_load_agent_code_object_fn;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    EXPECT_NE(api.core.hsa_executable_load_agent_code_object_fn, original_load);
    EXPECT_EQ(hook.instrumentation_nanoseconds(), 0u);

    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    EXPECT_GT(hook.instrumentation_nanoseconds(), 0u);
    ASSERT_EQ(g_transform_override_flavors.size(), 1u);
    EXPECT_EQ(g_transform_override_flavors.front(), Mode::Default);

    ASSERT_EQ(g_transform_override_runtime_sample_strides.size(), 1u);
    EXPECT_EQ(g_transform_override_runtime_sample_strides.front(), 256u);

    testing::internal::CaptureStderr();
    hook.unload();
    const std::string unload_log = testing::internal::GetCapturedStderr();
    EXPECT_NE(unload_log.find("ConSan instrumentation timing total_ns="), std::string::npos)
        << unload_log;
  }
}

TEST(HsaHooksUnitTest, ConSanThreadsAbsoluteAndRelativePatchedImageGrowthLimits) {
  struct GrowthCase {
    const char *absolute;
    const char *relative;
    PatchedImageGrowthLimit expected;
  };
  constexpr std::array cases = {
      GrowthCase{"4096",
                 nullptr,
                 {.kind = PatchedImageGrowthLimitKind::AbsoluteBytes,
                  .absolute_bytes = 4096u,
                  .input_percent = 0u}},
      GrowthCase{"0100",
                 nullptr,
                 {.kind = PatchedImageGrowthLimitKind::AbsoluteBytes,
                  .absolute_bytes = 100u,
                  .input_percent = 0u}},
      GrowthCase{nullptr,
                 "37",
                 {.kind = PatchedImageGrowthLimitKind::InputPercent,
                  .absolute_bytes = kDefaultMaxPatchedImageGrowthBytes,
                  .input_percent = 37u}},
      GrowthCase{nullptr,
                 "0100",
                 {.kind = PatchedImageGrowthLimitKind::InputPercent,
                  .absolute_bytes = kDefaultMaxPatchedImageGrowthBytes,
                  .input_percent = 100u}},
  };

  for (const GrowthCase &test : cases) {
    SCOPED_TRACE(test.absolute != nullptr ? "absolute" : "relative");
    SCOPED_TRACE(test.absolute != nullptr ? test.absolute : test.relative);
    configure_consan_profile(kConSanHookProfiles[1], false);
    ScopedEnvVar absolute("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES", test.absolute);
    ScopedEnvVar relative("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT", test.relative);
    reset_code_object_observations();
    g_transform_override_result.outcome = TransformOutcome::Unchanged;

    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);

    ASSERT_EQ(g_transform_override_patched_image_growth_limits.size(), 1u);
    const auto &observed = g_transform_override_patched_image_growth_limits.front();
    EXPECT_EQ(observed, test.expected);
  }
}

TEST(HsaHooksUnitTest, ConSanRejectsAmbiguousPatchedImageGrowthLimits) {
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar absolute("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES", "4096");
  ScopedEnvVar relative("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT", "37");

  reset_code_object_observations();
  FakeApiTable api;
  const auto original_load = api.core.hsa_executable_load_agent_code_object_fn;
  InstalledDbiHook hook(api);
  EXPECT_FALSE(hook.installed());
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn, original_load);
}

TEST(HsaHooksUnitTest, ConSanRejectsMalformedPatchedImageGrowthLimits) {
  constexpr std::array cases = {
      std::pair{"-1", static_cast<const char *>(nullptr)},
      std::pair{" -1", static_cast<const char *>(nullptr)},
      std::pair{"\t-1", static_cast<const char *>(nullptr)},
      std::pair{"+5", static_cast<const char *>(nullptr)},
      std::pair{static_cast<const char *>(nullptr), "-1"},
      std::pair{static_cast<const char *>(nullptr), " -1"},
      std::pair{static_cast<const char *>(nullptr), "\t-1"},
      std::pair{static_cast<const char *>(nullptr), "+5"},
  };
  for (const auto &[absolute_value, relative_value] : cases) {
    SCOPED_TRACE(absolute_value != nullptr ? "absolute" : "relative");
    SCOPED_TRACE(absolute_value != nullptr ? absolute_value : relative_value);
    configure_consan_profile(kConSanHookProfiles[1], false);
    ScopedEnvVar absolute("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES", absolute_value);
    ScopedEnvVar relative("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT", relative_value);

    reset_code_object_observations();
    FakeApiTable api;
    const auto original_load = api.core.hsa_executable_load_agent_code_object_fn;
    InstalledDbiHook hook(api);
    EXPECT_FALSE(hook.installed());
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn, original_load);
  }
}

TEST(HsaHooksUnitTest, ConSanRejectsMalformedProcessMemoryLimits) {
  constexpr std::array names = {
      "RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES",
      "RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES",
      "RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES",
  };
  constexpr std::array values = {
      "-1", " 1", "\t1", "+5", "0x10", "1x", "18446744073709551616",
  };
  for (const char *name : names) {
    for (const char *value : values) {
      SCOPED_TRACE(name);
      SCOPED_TRACE(value);
      configure_consan_profile(kConSanHookProfiles[1], false);
      ScopedEnvVar process_limit(name, value);

      reset_code_object_observations();
      FakeApiTable api;
      const auto original_load = api.core.hsa_executable_load_agent_code_object_fn;
      InstalledDbiHook hook(api);
      EXPECT_FALSE(hook.installed());
      EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn, original_load);
    }
  }
}

TEST(HsaHooksUnitTest, ConSanAcceptsBoundaryProcessMemoryLimits) {
  struct Limit {
    const char *name;
    const char *log_name;
  };
  constexpr std::array limits = {
      Limit{"RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES",
            "process_concurrent_transform_limit_bytes"},
      Limit{"RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES", "process_patched_image_limit_bytes"},
      Limit{"RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES",
            "process_patched_image_growth_limit_bytes"},
  };
  constexpr std::array values = {
      "",
      "0",
      "18446744073709551615",
  };
  for (const Limit &limit : limits) {
    for (const char *value : values) {
      SCOPED_TRACE(limit.name);
      SCOPED_TRACE(value);
      configure_consan_profile(kConSanHookProfiles[1], false);
      ScopedEnvVar process_limit(limit.name, value);
      ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
      reset_code_object_observations();

      testing::internal::CaptureStderr();
      {
        FakeApiTable api;
        InstalledDbiHook hook(api);
        ASSERT_TRUE(hook.installed()) << hook.error();
      }
      const std::string log = testing::internal::GetCapturedStderr();
      const std::string expected =
          std::string(limit.log_name) + "=" + (value[0] == '\0' ? "unlimited" : value);
      EXPECT_NE(log.find(expected), std::string::npos) << log;
    }
  }
}

TEST(HsaHooksUnitTest, ConSanAcceptsRuntimeSamplingOverrideForEveryEngine) {
  ScopedEnvVar runtime_stride("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE", "1");
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  for (size_t profile_index : {1u}) {
    SCOPED_TRACE(kConSanHookProfiles[profile_index].name);
    configure_consan_profile(kConSanHookProfiles[profile_index], false);
    reset_code_object_observations();

    testing::internal::CaptureStderr();
    {
      FakeApiTable api;
      InstalledDbiHook hook(api);
      ASSERT_TRUE(hook.installed()) << hook.error();
    }
    const std::string log = testing::internal::GetCapturedStderr();
    EXPECT_NE(log.find("runtime_sample_stride=1 runtime_sample_stride_source=expert-override"),
              std::string::npos)
        << log;
    EXPECT_EQ(log.find("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE is ignored"), std::string::npos) << log;
  }
}

TEST(HsaHooksUnitTest, ConSanAcceptsEpochAnalysisPolicies) {
  constexpr std::array policies = {
      std::pair{"every", "every"},
      std::pair{"EVERY", "every"},
      std::pair{"nth:7", "nth:7"},
      std::pair{"periodic:9", "periodic:9:9"},
      std::pair{"periodic:9:2", "periodic:9:2"},
      std::pair{"manual", "manual"},
  };
  for (const auto &[value, normalized] : policies) {
    SCOPED_TRACE(value);
    configure_consan_profile(kConSanHookProfiles[1], false);
    ScopedEnvVar epoch_analysis("RJ_CONSAN_EPOCH_ANALYSIS", value);
    ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
    reset_code_object_observations();

    testing::internal::CaptureStderr();
    {
      FakeApiTable api;
      InstalledDbiHook hook(api);
      ASSERT_TRUE(hook.installed()) << hook.error();
    }
    const std::string log = testing::internal::GetCapturedStderr();
    EXPECT_NE(log.find("epoch_analysis=" + std::string(normalized)), std::string::npos) << log;
  }
}

TEST(HsaHooksUnitTest, ConSanRejectsMalformedEpochAnalysisPolicies) {
  constexpr std::array values = {
      "none",           "nth",        "nth:0",       "nth:-1",       "nth:1:2",
      "periodic",       "periodic:0", "periodic:2:", "periodic:2:0", "periodic:2:3",
      "periodic:2:1:1", "manual:1",   " every",      "periodic: 2",  "18446744073709551616",
  };
  for (const char *value : values) {
    SCOPED_TRACE(value);
    configure_consan_profile(kConSanHookProfiles[1], false);
    ScopedEnvVar epoch_analysis("RJ_CONSAN_EPOCH_ANALYSIS", value);
    reset_code_object_observations();

    FakeApiTable api;
    InstalledDbiHook hook(api);
    EXPECT_FALSE(hook.installed());
  }
}

TEST(HsaHooksUnitTest, ConSanWarnsWhenTransformLimitCannotAdmitAnyNonemptyObject) {
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar object_growth_limit("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES", "4");
  const uint64_t smallest_reservation = absolute_transform_test_reservation_bytes(1, 4);
  ASSERT_GT(smallest_reservation, 0u);
  const std::string transform_limit_value = std::to_string(smallest_reservation - 1);
  ScopedEnvVar transform_limit("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES",
                               transform_limit_value.c_str());
  reset_code_object_observations();

  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
  }
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(
      log.find("cannot admit any nonempty code object; the smallest possible reservation is " +
               std::to_string(smallest_reservation) + " bytes"),
      std::string::npos)
      << log;
  EXPECT_NE(log.find("phase=composite-incremental-patch: 1 * input bytes + 12 * "
                     "(input bytes + maximum growth bytes)"),
            std::string::npos)
      << log;
}

TEST(HsaHooksUnitTest, ConSanWarnsWhenRelativeGrowthTransformLimitCannotAdmitAnyObject) {
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar object_growth_limit("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT", "37");
  const PatchedImageGrowthLimit growth_limit = {
      .kind = PatchedImageGrowthLimitKind::InputPercent,
      .input_percent = 37,
  };
  const uint64_t smallest_reservation = transform_test_reservation_bytes(1, growth_limit);
  ASSERT_GT(smallest_reservation, 0u);
  const std::string transform_limit_value = std::to_string(smallest_reservation - 1);
  ScopedEnvVar transform_limit("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES",
                               transform_limit_value.c_str());
  reset_code_object_observations();

  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
  }
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(
      log.find("cannot admit any nonempty code object; the smallest possible reservation is " +
               std::to_string(smallest_reservation) + " bytes"),
      std::string::npos)
      << log;
  EXPECT_NE(log.find("phase=final-validation: 8 * input bytes + 9 * "
                     "(input bytes + maximum growth bytes)"),
            std::string::npos)
      << log;
}

TEST(HsaHooksUnitTest, ConSanConcurrentTransformLimitFailsOpenBeforeTransform) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar object_growth_limit("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES", "4");
  const uint64_t reservation_bytes = absolute_transform_test_reservation_bytes(8, 4);
  ASSERT_GT(reservation_bytes, 0u);
  const std::string transform_limit_value = std::to_string(reservation_bytes - 1);
  ScopedEnvVar transform_limit("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES",
                               transform_limit_value.c_str());

  g_transform_override_result = process_growth_replacement_result();

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  hook.unload();
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(g_transform_override_flavors.empty());
  EXPECT_EQ(g_loaded_code_object_readers, (std::vector<uint64_t>{reader.handle}));
  const std::string expected_limit_log =
      "process concurrent transform limit exceeded: reader=101 input_image=8 live=0 reservation=" +
      std::to_string(reservation_bytes) + " required=" + std::to_string(reservation_bytes) +
      " limit=" + transform_limit_value;
  EXPECT_NE(log.find(expected_limit_log), std::string::npos) << log;
  EXPECT_NE(log.find("analysis_complete=false static_complete=false "
                     "dynamic_complete=true applicable_code_objects=1 "
                     "incomplete_code_objects=1 access=0/0"),
            std::string::npos)
      << log;
}

TEST(HsaHooksUnitTest, ConSanConcurrentTransformLimitFailsClosedBeforeTransform) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], true);
  ScopedEnvVar object_growth_limit("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES", "4");
  const uint64_t reservation_bytes = absolute_transform_test_reservation_bytes(8, 4);
  ASSERT_GT(reservation_bytes, 0u);
  const std::string transform_limit_value = std::to_string(reservation_bytes - 1);
  ScopedEnvVar transform_limit("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES",
                               transform_limit_value.c_str());

  g_transform_override_result = process_growth_replacement_result();

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_ERROR_OUT_OF_RESOURCES);
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("reason=process-concurrent-transform-limit"), std::string::npos) << log;
  EXPECT_TRUE(g_transform_override_flavors.empty());
  EXPECT_TRUE(g_loaded_code_object_readers.empty());
}

TEST(HsaHooksUnitTest, ConSanConcurrentTransformLimitTracksAndRefundsReservations) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar object_growth_limit("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES", "4");
  const uint64_t reservation_bytes = absolute_transform_test_reservation_bytes(8, 4);
  ASSERT_GT(reservation_bytes, 0u);
  const std::string transform_limit_value = std::to_string(reservation_bytes);
  ScopedEnvVar transform_limit("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES",
                               transform_limit_value.c_str());

  g_transform_override_result = process_growth_replacement_result();
  g_block_first_transform = true;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  std::array<hsa_code_object_reader_t, 3> readers{};
  for (hsa_code_object_reader_t &reader : readers) {
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
  }

  testing::internal::CaptureStderr();
  auto first_load = std::async(std::launch::async, [&] {
    return api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                             readers[0], nullptr, nullptr);
  });
  {
    std::unique_lock lock(g_transform_block_mutex);
    if (!g_transform_block_cv.wait_for(lock, std::chrono::seconds(2),
                                       [] { return g_first_transform_entered; })) {
      g_release_first_transform = true;
      lock.unlock();
      g_transform_block_cv.notify_all();
      (void)first_load.get();
      (void)testing::internal::GetCapturedStderr();
      FAIL() << "first transform did not reach the blocked transform boundary";
      return;
    }
  }

  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{8}, kHostAgent,
                                                              readers[1], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  {
    std::lock_guard lock(g_transform_block_mutex);
    g_release_first_transform = true;
  }
  g_transform_block_cv.notify_all();
  EXPECT_EQ(first_load.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{9}, kHostAgent,
                                                              readers[2], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_transform_override_flavors.size(), 2u);
  EXPECT_EQ(g_loaded_code_object_readers, (std::vector<uint64_t>{102u, 104u, 105u}));

  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{8}), HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{9}), HSA_STATUS_SUCCESS);
  hook.unload();
  const std::string log = testing::internal::GetCapturedStderr();
  const std::string expected_rejection = "live=" + std::to_string(reservation_bytes) +
                                         " reservation=" + std::to_string(reservation_bytes) +
                                         " required=" + std::to_string(2 * reservation_bytes) +
                                         " limit=" + transform_limit_value;
  EXPECT_NE(log.find(expected_rejection), std::string::npos) << log;
  EXPECT_NE(log.find("transform admission memory live_bytes=0 peak_reserved_bytes=" +
                     std::to_string(reservation_bytes) +
                     " process_ceiling=" + transform_limit_value),
            std::string::npos)
      << log;
}

TEST(HsaHooksUnitTest, ConSanConcurrentTransformTracksUnlimitedPeak) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar object_growth_limit("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES", "4");
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  g_transform_override_result = process_growth_replacement_result();

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  hook.unload();
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_EQ(g_transform_override_flavors.size(), 1u);
  const uint64_t reservation_bytes = absolute_transform_test_reservation_bytes(8, 4);
  EXPECT_NE(log.find("ConSan transform admission request reader=101 input_image=8 reservation=" +
                     std::to_string(reservation_bytes) +
                     " phase=final-validation phase_input_copies=8 phase_maximum_copies=9"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("transform admission memory live_bytes=0 peak_reserved_bytes=" +
                     std::to_string(reservation_bytes) + " process_ceiling=unlimited"),
            std::string::npos)
      << log;
}

TEST(HsaHooksUnitTest, ConSanTransformReservationOverflowHasDistinctFailure) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], true);
  ScopedEnvVar object_growth_limit("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES",
                                   "18446744073709551615");
  ScopedEnvVar transform_limit("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES",
                               "18446744073709551615");
  g_transform_override_result = process_growth_replacement_result();

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_ERROR_OUT_OF_RESOURCES);
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("transform reservation accounting overflow: reader=101 input_image=8"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("reason=process-concurrent-transform-accounting"), std::string::npos) << log;
  EXPECT_TRUE(g_transform_override_flavors.empty());
}

TEST(HsaHooksUnitTest, ConSanUnlimitedTransformReservationOverflowStillTransforms) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar object_growth_limit("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES",
                                   "18446744073709551615");
  g_transform_override_result = process_growth_replacement_result();

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  hook.unload();
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_EQ(g_transform_override_flavors.size(), 1u);
  EXPECT_NE(log.find("continuing without a process reservation"), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, ConSanReleasesTransformReservationBeforeOriginalLoad) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar object_growth_limit("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES", "4");
  const std::string transform_limit_value =
      std::to_string(absolute_transform_test_reservation_bytes(8, 4));
  ScopedEnvVar transform_limit("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES",
                               transform_limit_value.c_str());
  g_transform_override_result.outcome = TransformOutcome::Unchanged;
  g_block_first_loader_call = true;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  std::array<hsa_code_object_reader_t, 2> readers{};
  for (hsa_code_object_reader_t &reader : readers) {
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
  }

  auto first_load = std::async(std::launch::async, [&] {
    return api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                             readers[0], nullptr, nullptr);
  });
  bool loader_entered = false;
  {
    std::unique_lock lock(g_loader_block_mutex);
    loader_entered = g_loader_block_cv.wait_for(lock, std::chrono::seconds(2),
                                                [] { return g_first_loader_call_entered; });
  }
  if (!loader_entered) {
    {
      std::lock_guard lock(g_loader_block_mutex);
      g_release_first_loader_call = true;
    }
    g_loader_block_cv.notify_all();
    (void)first_load.get();
    FAIL() << "first load did not reach the blocked loader boundary";
    return;
  }
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{8}, kHostAgent,
                                                              readers[1], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  {
    std::lock_guard lock(g_loader_block_mutex);
    g_release_first_loader_call = true;
  }
  g_loader_block_cv.notify_all();
  EXPECT_EQ(first_load.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_transform_override_flavors.size(), 2u);
}

TEST(HsaHooksUnitTest, ConSanReleasesTransformReservationBeforeRetentionFallback) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar object_growth_limit("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES", "4");
  const std::string transform_limit_value =
      std::to_string(absolute_transform_test_reservation_bytes(8, 4));
  ScopedEnvVar transform_limit("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES",
                               transform_limit_value.c_str());
  ScopedEnvVar process_image_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES", "0");
  g_transform_override_result = process_growth_replacement_result();
  g_block_first_loader_call = true;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  std::array<hsa_code_object_reader_t, 2> readers{};
  for (hsa_code_object_reader_t &reader : readers) {
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
  }

  auto first_load = std::async(std::launch::async, [&] {
    return api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                             readers[0], nullptr, nullptr);
  });
  bool loader_entered = false;
  {
    std::unique_lock lock(g_loader_block_mutex);
    loader_entered = g_loader_block_cv.wait_for(lock, std::chrono::seconds(2),
                                                [] { return g_first_loader_call_entered; });
  }
  if (!loader_entered) {
    {
      std::lock_guard lock(g_loader_block_mutex);
      g_release_first_loader_call = true;
    }
    g_loader_block_cv.notify_all();
    (void)first_load.get();
    FAIL() << "first retention fallback did not reach the blocked loader boundary";
    return;
  }
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{8}, kHostAgent,
                                                              readers[1], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  {
    std::lock_guard lock(g_loader_block_mutex);
    g_release_first_loader_call = true;
  }
  g_loader_block_cv.notify_all();
  EXPECT_EQ(first_load.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_transform_override_flavors.size(), 2u);
}

TEST(HsaHooksUnitTest, ConSanReleasesTransformReservationOnEarlyRejection) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar object_growth_limit("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES", "4");
  const std::string transform_limit_value =
      std::to_string(absolute_transform_test_reservation_bytes(8, 4));
  ScopedEnvVar transform_limit("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES",
                               transform_limit_value.c_str());
  TransformArtifacts unresolved = process_growth_replacement_result();
  install_consan_test_program_identity(unresolved, ROCJITSU_CODE_ARCH_INVALID,
                                       ROCJITSU_CODE_TARGET_GFX1201,
                                       /*semantic_arch_required=*/true);
  g_transform_override_results.push_back(std::move(unresolved));
  g_transform_override_results.push_back(process_growth_replacement_result());

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  std::array<hsa_code_object_reader_t, 2> readers{};
  for (hsa_code_object_reader_t &reader : readers) {
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
  }

  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              readers[0], nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{8}, kHostAgent,
                                                              readers[1], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_transform_override_flavors.size(), 2u);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{8}), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, ConSanPreservesLiveTransformReservationAcrossUnloadAndReload) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar object_growth_limit("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES", "4");
  const uint64_t reservation_bytes = absolute_transform_test_reservation_bytes(8, 4);
  const std::string transform_limit_value = std::to_string(reservation_bytes);
  ScopedEnvVar transform_limit("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES",
                               transform_limit_value.c_str());
  g_transform_override_result = process_growth_replacement_result();
  g_block_first_transform = true;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  std::array<hsa_code_object_reader_t, 3> readers{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &readers[0]),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  auto first_load = std::async(std::launch::async, [&] {
    return api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                             readers[0], nullptr, nullptr);
  });
  bool transform_entered = false;
  {
    std::unique_lock lock(g_transform_block_mutex);
    transform_entered = g_transform_block_cv.wait_for(lock, std::chrono::seconds(2),
                                                      [] { return g_first_transform_entered; });
  }
  if (!transform_entered) {
    {
      std::lock_guard lock(g_transform_block_mutex);
      g_release_first_transform = true;
    }
    g_transform_block_cv.notify_all();
    (void)first_load.get();
    (void)testing::internal::GetCapturedStderr();
    FAIL() << "first transform did not reach the blocked transform boundary";
    return;
  }

  hook.unload();
  const bool reloaded = hook.reload(api);
  if (reloaded) {
    EXPECT_GT(hook.instrumentation_nanoseconds(), 0u);
  }
  hsa_status_t second_reader_status = HSA_STATUS_ERROR;
  hsa_status_t third_reader_status = HSA_STATUS_ERROR;
  if (reloaded) {
    second_reader_status = api.core.hsa_code_object_reader_create_from_memory_fn(
        original.data(), original.size(), &readers[1]);
    third_reader_status = api.core.hsa_code_object_reader_create_from_memory_fn(
        original.data(), original.size(), &readers[2]);
  }
  if (!reloaded || second_reader_status != HSA_STATUS_SUCCESS ||
      third_reader_status != HSA_STATUS_SUCCESS) {
    {
      std::lock_guard lock(g_transform_block_mutex);
      g_release_first_transform = true;
    }
    g_transform_block_cv.notify_all();
    (void)first_load.get();
    (void)testing::internal::GetCapturedStderr();
    FAIL() << "failed to reinstall the hook and recreate readers: " << hook.error();
    return;
  }
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{8}, kHostAgent,
                                                              readers[1], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  {
    std::lock_guard lock(g_transform_block_mutex);
    g_release_first_transform = true;
  }
  g_transform_block_cv.notify_all();
  EXPECT_EQ(first_load.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{9}, kHostAgent,
                                                              readers[2], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_transform_override_flavors.size(), 2u);

  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{8}), HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{9}), HSA_STATUS_SUCCESS);
  hook.unload();
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("live_bytes=" + std::to_string(reservation_bytes) +
                     " peak_reserved_bytes=" + std::to_string(reservation_bytes) +
                     " process_ceiling=" + transform_limit_value),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("live=" + std::to_string(reservation_bytes) +
                     " reservation=" + std::to_string(reservation_bytes) + " required=" +
                     std::to_string(2 * reservation_bytes) + " limit=" + transform_limit_value),
            std::string::npos)
      << log;
  EXPECT_EQ(log.find("transform reservation refund exceeded the live total"), std::string::npos)
      << log;
}

TEST(HsaHooksUnitTest, ConSanProcessPatchedImageLimitIsAtomicWithGrowthBudget) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar process_growth_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES", "8");
  ScopedEnvVar process_image_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES", "12");

  g_transform_override_result = process_growth_replacement_result();

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  std::array<hsa_code_object_reader_t, 3> readers{};
  for (hsa_code_object_reader_t &reader : readers) {
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
  }

  testing::internal::CaptureStderr();
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              readers[0], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{8}, kHostAgent,
                                                              readers[1], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{9}, kHostAgent,
                                                              readers[2], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_loaded_code_object_readers, (std::vector<uint64_t>{104u, 102u, 105u}));

  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{8}), HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{9}), HSA_STATUS_SUCCESS);
  hook.unload();
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("process patched-image limit exceeded: "
                     "live=12 replacement_image=12 replacement_growth=4 "
                     "required=24 limit=12"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("patched-image memory live_bytes=0 "
                     "peak_image_bytes=12 process_ceiling=12"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("patched-image growth memory live_bytes=0 "
                     "peak_growth_bytes=4 process_ceiling=8"),
            std::string::npos)
      << log;
}

TEST(HsaHooksUnitTest, ConSanProcessPatchedImageLimitRejectsNoGrowthReplacement) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], true);
  ScopedEnvVar process_image_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES", "0");

  g_transform_override_result = process_growth_replacement_result(8);

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_ERROR_OUT_OF_RESOURCES);
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("reason=process-patched-image-limit"), std::string::npos) << log;
  EXPECT_TRUE(g_loaded_code_object_readers.empty());
}

TEST(HsaHooksUnitTest, ConSanProcessPatchedImageGrowthLimitTracksLiveReplacements) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar process_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES", "4");

  g_transform_override_result = process_growth_replacement_result();

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  std::array<hsa_code_object_reader_t, 3> readers{};
  for (hsa_code_object_reader_t &reader : readers) {
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
  }

  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              readers[0], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  testing::internal::CaptureStderr();
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{8}, kHostAgent,
                                                              readers[1], nullptr, nullptr),
            HSA_STATUS_SUCCESS);

  ASSERT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{9}, kHostAgent,
                                                              readers[2], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_loaded_code_object_readers, (std::vector<uint64_t>{104u, 102u, 105u}));

  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{8}), HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{9}), HSA_STATUS_SUCCESS);
  hook.unload();
  const std::string limit_log = testing::internal::GetCapturedStderr();
  EXPECT_NE(limit_log.find("process patched-image growth limit exceeded: "
                           "live=4 replacement_growth=4 replacement_image=12 "
                           "required=8 limit=4"),
            std::string::npos)
      << limit_log;
  EXPECT_NE(limit_log.find("patched-image growth memory live_bytes=0 "
                           "peak_growth_bytes=4 process_ceiling=4"),
            std::string::npos)
      << limit_log;
}

TEST(HsaHooksUnitTest, ConSanProcessPatchedImageGrowthLimitFailsClosed) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], true);
  ScopedEnvVar process_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES", "0");

  g_transform_override_result = process_growth_replacement_result();

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_ERROR_OUT_OF_RESOURCES);
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("reason=process-patched-image-growth-limit"), std::string::npos) << log;
  EXPECT_TRUE(g_loaded_code_object_readers.empty());
}

TEST(HsaHooksUnitTest, ConSanProcessPatchedImageGrowthLimitZeroAdmitsNoGrowthReplacement) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar process_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES", "0");

  g_transform_override_result = process_growth_replacement_result(8);

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_loaded_code_object_readers, (std::vector<uint64_t>{102u}));
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, ConSanProcessPatchedImageGrowthLimitRefundsFailedReaderCreation) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar process_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES", "4");

  g_transform_override_result = process_growth_replacement_result();

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  std::array<hsa_code_object_reader_t, 2> readers{};
  for (hsa_code_object_reader_t &reader : readers) {
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
  }

  g_fail_replacement_reader_create = true;
  g_block_loader_reader = readers[0].handle;
  auto first_load = std::async(std::launch::async, [&] {
    return api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                             readers[0], nullptr, nullptr);
  });
  {
    std::unique_lock lock(g_loader_block_mutex);
    if (!g_loader_block_cv.wait_for(lock, std::chrono::seconds(2),
                                    [] { return g_first_loader_call_entered; })) {
      g_release_first_loader_call = true;
      lock.unlock();
      g_loader_block_cv.notify_all();
      (void)first_load.get();
      FAIL() << "reader-creation fallback did not reach the blocked original loader";
      return;
    }
  }
  g_fail_replacement_reader_create = false;
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{8}, kHostAgent,
                                                              readers[1], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  {
    std::lock_guard lock(g_loader_block_mutex);
    g_release_first_loader_call = true;
  }
  g_loader_block_cv.notify_all();
  EXPECT_EQ(first_load.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_loaded_code_object_readers, (std::vector<uint64_t>{101u, 104u}));

  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{8}), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, ConSanProcessPatchedImageGrowthLimitRefundsExactFailedReplacementLoad) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar process_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES", "12");

  g_transform_override_results.push_back(process_growth_replacement_result(12));
  g_transform_override_results.push_back(process_growth_replacement_result(16));
  g_transform_override_results.push_back(process_growth_replacement_result(16));

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  std::array<hsa_code_object_reader_t, 3> readers{};
  for (hsa_code_object_reader_t &reader : readers) {
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
  }

  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              readers[0], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  g_fail_loader_once_for_reader = 105u;
  g_block_loader_reader = readers[1].handle;
  auto failed_replacement_load = std::async(std::launch::async, [&] {
    return api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                             readers[1], nullptr, nullptr);
  });
  {
    std::unique_lock lock(g_loader_block_mutex);
    if (!g_loader_block_cv.wait_for(lock, std::chrono::seconds(2),
                                    [] { return g_first_loader_call_entered; })) {
      g_release_first_loader_call = true;
      lock.unlock();
      g_loader_block_cv.notify_all();
      (void)failed_replacement_load.get();
      FAIL() << "failed-replacement fallback did not reach the blocked original loader";
      return;
    }
  }
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{8}, kHostAgent,
                                                              readers[2], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  {
    std::lock_guard lock(g_loader_block_mutex);
    g_release_first_loader_call = true;
  }
  g_loader_block_cv.notify_all();
  EXPECT_EQ(failed_replacement_load.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_loaded_code_object_readers, (std::vector<uint64_t>{104u, 105u, 102u, 106u}));

  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{8}), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, ConSanProcessPatchedImageGrowthLimitRefundsEveryObjectOnExecutableDestroy) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar process_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES", "8");

  g_transform_override_result = process_growth_replacement_result();

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  std::array<hsa_code_object_reader_t, 3> readers{};
  for (hsa_code_object_reader_t &reader : readers) {
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
  }

  for (size_t i = 0; i < 2; ++i) {
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                readers[i], nullptr, nullptr),
              HSA_STATUS_SUCCESS);
  }
  ASSERT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{8}, kHostAgent,
                                                              readers[2], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_loaded_code_object_readers, (std::vector<uint64_t>{104u, 105u, 106u}));
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{8}), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, ConSanProcessPatchedImageGrowthLimitCountsInFlightReplacementLoads) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar process_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES", "4");

  g_transform_override_result = process_growth_replacement_result();
  g_block_first_loader_call = true;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  std::array<hsa_code_object_reader_t, 2> readers{};
  for (hsa_code_object_reader_t &reader : readers) {
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
  }

  auto first_load = std::async(std::launch::async, [&] {
    return api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                             readers[0], nullptr, nullptr);
  });
  {
    std::unique_lock lock(g_loader_block_mutex);
    if (!g_loader_block_cv.wait_for(lock, std::chrono::seconds(2),
                                    [] { return g_first_loader_call_entered; })) {
      g_release_first_loader_call = true;
      lock.unlock();
      g_loader_block_cv.notify_all();
      (void)first_load.get();
      FAIL() << "first replacement load did not reach the blocked loader";
      return;
    }
  }

  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{8}, kHostAgent,
                                                              readers[1], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  {
    std::lock_guard lock(g_loader_block_mutex);
    g_release_first_loader_call = true;
  }
  g_loader_block_cv.notify_all();
  EXPECT_EQ(first_load.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_loaded_code_object_readers, (std::vector<uint64_t>{103u, 102u}));

  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{8}), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, ConSanPreservesRetainedReplacementAcrossUnloadAndReload) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar process_growth_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES", "4");
  ScopedEnvVar process_image_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES", "12");
  g_transform_override_result = process_growth_replacement_result();
  g_block_first_loader_call = true;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  std::array<hsa_code_object_reader_t, 3> readers{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &readers[0]),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  auto first_load = std::async(std::launch::async, [&] {
    return api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                             readers[0], nullptr, nullptr);
  });
  bool loader_entered = false;
  {
    std::unique_lock lock(g_loader_block_mutex);
    loader_entered = g_loader_block_cv.wait_for(lock, std::chrono::seconds(2),
                                                [] { return g_first_loader_call_entered; });
  }
  if (!loader_entered) {
    {
      std::lock_guard lock(g_loader_block_mutex);
      g_release_first_loader_call = true;
    }
    g_loader_block_cv.notify_all();
    (void)first_load.get();
    (void)testing::internal::GetCapturedStderr();
    FAIL() << "replacement load did not reach the blocked loader boundary";
    return;
  }

  hook.unload();
  const std::string unload_log = testing::internal::GetCapturedStderr();
  EXPECT_NE(unload_log.find("patched-image memory live_bytes=12 peak_image_bytes=12 "
                            "process_ceiling=12"),
            std::string::npos)
      << unload_log;
  testing::internal::CaptureStderr();
  if (!hook.reload(api)) {
    {
      std::lock_guard lock(g_loader_block_mutex);
      g_release_first_loader_call = true;
    }
    g_loader_block_cv.notify_all();
    (void)first_load.get();
    (void)testing::internal::GetCapturedStderr();
    FAIL() << "failed to reinstall the hook: " << hook.error();
    return;
  }
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &readers[1]),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &readers[2]),
            HSA_STATUS_SUCCESS);

  // The first executable's retained replacement still owns the entire
  // process budget, so the reloaded hook must fail open to the second original.
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{8}, kHostAgent,
                                                              readers[1], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  {
    std::lock_guard lock(g_loader_block_mutex);
    g_release_first_loader_call = true;
  }
  g_loader_block_cv.notify_all();
  EXPECT_EQ(first_load.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_loaded_code_object_readers, (std::vector<uint64_t>{102u, 103u}));

  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  EXPECT_TRUE(replacement_storage_valid_at_destroy(7u));

  // Destroying the first executable releases both retained charges, allowing
  // a subsequent replacement under the reloaded hook.
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{9}, kHostAgent,
                                                              readers[2], nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_loaded_code_object_readers, (std::vector<uint64_t>{102u, 103u, 105u}));
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{8}), HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{9}), HSA_STATUS_SUCCESS);

  hook.unload();
  const std::string final_log = testing::internal::GetCapturedStderr();
  EXPECT_NE(final_log.find("patched-image memory live_bytes=0 peak_image_bytes=12 "
                           "process_ceiling=12"),
            std::string::npos)
      << final_log;
  EXPECT_NE(final_log.find("patched-image growth memory live_bytes=0 peak_growth_bytes=4 "
                           "process_ceiling=4"),
            std::string::npos)
      << final_log;
  EXPECT_NE(final_log.find("process patched-image growth limit exceeded: "
                           "live=4 replacement_growth=4 replacement_image=12"),
            std::string::npos)
      << final_log;
}

TEST(HsaHooksUnitTest, ConSanKeepsRetainedChargeForDestroyInsideUnloadedWindow) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar process_growth_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES", "4");
  ScopedEnvVar process_image_limit("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES", "12");
  g_transform_override_result = process_growth_replacement_result();

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);
  constexpr hsa_executable_t executable{17};
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(executable, kHostAgent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_SUCCESS);

  hook.unload();
  // This call is outside the hook lifetime and reaches only the runtime's
  // original function, so the retained replacement cannot be reconciled.
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(executable), HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  ASSERT_TRUE(hook.reload(api)) << hook.error();
  hook.unload();
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("patched-image memory live_bytes=12 peak_image_bytes=12 "
                     "process_ceiling=12"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("patched-image growth memory live_bytes=4 peak_growth_bytes=4 "
                     "process_ceiling=4"),
            std::string::npos)
      << log;

  // A later observed destroy can still reconcile the synthetic test handle;
  // release it so this process-wide registry cannot affect following tests.
  ASSERT_TRUE(hook.reload(api)) << hook.error();
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(executable), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, ConSanReportTrustEvaluationOwnsDynamicCompleteness) {
  using Summary = rocjitsu::consan::hook::ReportSummary;
  using rocjitsu::consan::hook::evaluate_report_trust;
  Summary summary;
  const auto missing = evaluate_report_trust(summary, true);
  EXPECT_EQ(missing.visible_evidence_count, 0u);
  EXPECT_TRUE(missing.required_records_missing);
  EXPECT_FALSE(missing.dynamic_complete);
  EXPECT_TRUE(evaluate_report_trust(summary, false).dynamic_complete);
  summary.visible_watchpoint_count = 2;
  EXPECT_TRUE(evaluate_report_trust(summary, true).dynamic_complete);
  for (auto counter : {&Summary::allocation_failure_count, &Summary::cleanup_failure_count,
                       &Summary::dropped_window_count, &Summary::stale_snapshot_count,
                       &Summary::incomplete_snapshot_count, &Summary::changed_snapshot_count,
                       &Summary::malformed_snapshot_count, &Summary::unsupported_sync_count,
                       &Summary::malformed_sync_count}) {
    auto incomplete = summary;
    incomplete.*counter = 1;
    const auto evaluated = evaluate_report_trust(incomplete, true);
    EXPECT_EQ(evaluated.visible_evidence_count, 2u);
    EXPECT_EQ(evaluated.dynamic_incomplete_count, 1u);
    EXPECT_FALSE(evaluated.required_records_missing);
    EXPECT_FALSE(evaluated.dynamic_complete);
    EXPECT_FALSE(evaluated.has_diagnostics);
  }
  for (auto counter : {&Summary::conflict_count, &Summary::immediate_conflict_count}) {
    auto conflict = summary;
    conflict.*counter = 1;
    const auto evaluated = evaluate_report_trust(conflict, true);
    EXPECT_TRUE(evaluated.dynamic_complete);
    EXPECT_TRUE(evaluated.has_diagnostics);
  }
}

TEST(HsaHooksUnitTest, ConSanAnalyzerOwnsConflicts) {
  using AccessKind = ShadowAccessKind;
  std::array<rocjitsu::consan::hook::Evidence, 2> sampled{};
  for (uint32_t index = 0; index < sampled.size(); ++index) {
    sampled[index].index = index;
    sampled[index].generation = 7;
    sampled[index].dispatch_id = 11;
    sampled[index].epoch = 3;
    sampled[index].entry = {
        .valid = true,
        .kind = AccessKind::Write,
        .owner_id = index + 1u,
        .epoch = 3,
        .generation = 7,
        .start_byte = 16,
        .byte_count = 4,
    };
  }
  const auto conflict = rocjitsu::consan::hook::analyze_conflicts(sampled, false);
  EXPECT_EQ(conflict.conflict_count, 1u);
  ASSERT_EQ(conflict.examples.size(), 1u);
  EXPECT_EQ(conflict.examples.front().first.index, 0u);
  EXPECT_EQ(conflict.examples.front().second.index, 1u);

  std::array<rocjitsu::consan::hook::AccessStaticMapping, 2> mappings{};
  mappings[0].owner_kernel_ids = {0x100};
  mappings[1].owner_kernel_ids = {0x200};
  mappings[0].owner_provenance_complete = true;
  mappings[1].owner_provenance_complete = true;
  sampled[0].static_mapping = &mappings[0];
  sampled[1].static_mapping = &mappings[1];
  EXPECT_EQ(rocjitsu::consan::hook::analyze_conflicts(sampled, false).conflict_count, 0u);
}

TEST(HsaHooksUnitTest, AutoReportSnapshotOwnsVisibilityAndCopyFailures) {
  AutoReportInventory inventory;
  inventory.access_range_count = 2;
  inventory.barrier_event_count = 16;
  inventory.atomic_event_count = 8;
  inventory.range_bank_count = 2;
  inventory.watchpoint_count = 2;
  const auto report = plan_auto_report(inventory);
  ASSERT_TRUE(report.complete());

  std::vector<uint8_t> source(report.required_bytes, 0xa5);
  auto header = make_report_header_for_layout(1, 2, report.layout);
  std::memcpy(source.data(), &header, sizeof(header));
  const rocjitsu::consan::hook::ReportSnapshotRequest request{
      .source = source.data(),
      .size = source.size(),
      .fine_grained = true,
  };

  const auto snapshot = rocjitsu::consan::hook::capture_report_snapshot(request);
  ASSERT_TRUE(snapshot.complete());
  EXPECT_EQ(snapshot.copied_bytes, source.size());
  EXPECT_EQ(snapshot.bytes, source);

  source[0] = 0;
  const auto malformed = rocjitsu::consan::hook::capture_report_snapshot(request);
  ASSERT_TRUE(malformed.complete());
  EXPECT_EQ(malformed.copied_bytes, source.size());
  EXPECT_EQ(malformed.bytes, source);

  auto coarse_request = request;
  coarse_request.fine_grained = false;
  const auto unavailable = rocjitsu::consan::hook::capture_report_snapshot(coarse_request);
  EXPECT_EQ(unavailable.failure, rocjitsu::consan::hook::ReportSnapshotFailure::CopyUnavailable);

  const auto failing_copy = [](void *, void *, const void *, size_t, int32_t *status) {
    *status = 17;
    return false;
  };
  const auto failed =
      rocjitsu::consan::hook::capture_report_snapshot(coarse_request, failing_copy, nullptr);
  EXPECT_EQ(failed.failure, rocjitsu::consan::hook::ReportSnapshotFailure::CopyFailed);
  EXPECT_EQ(failed.copy_status, 17);
}

TEST(HsaHooksUnitTest, AutoReportDecoderRejectsStaleFullGenerationAcrossTagRollover) {
  AutoReportInventory inventory;
  inventory.access_range_count = inventory.range_bank_count = inventory.watchpoint_count = 1;
  const auto report = plan_auto_report(inventory);
  ASSERT_TRUE(report.complete());
  constexpr uint64_t old_generation = 7;
  constexpr uint64_t generation = old_generation + (uint64_t{1} << watchpoint::generation_bits);
  ReportPipelineInput input{.size = static_cast<size_t>(report.required_bytes),
                            .layout = report.layout,
                            .input_fingerprint = {},
                            .expected_generation = generation};
  ReportSnapshot snapshot;
  snapshot.bytes.resize(report.required_bytes);
  auto header = make_report_header_for_layout(old_generation, 11, report.layout);
  header.causal_window_count = 1;
  CausalWindow window{.generation = old_generation,
                      .dispatch_id = 11,
                      .first_entry = 0,
                      .entry_count = 1,
                      .publication_state = static_cast<uint32_t>(CausalPublicationState::Ready)};
  const uint64_t packed =
      pack_watchpoint_entry(ShadowAccessKind::Write, 0, 0, old_generation, 0x10800, 2);
  const auto store = [&] {
    std::memcpy(snapshot.bytes.data(), &header, sizeof(header));
    std::memcpy(snapshot.bytes.data() + report.layout.causal_windows_offset, &window,
                sizeof(window));
    std::memcpy(snapshot.bytes.data() + report.layout.watchpoints_offset, &packed, sizeof(packed));
  };
  store();
  EXPECT_EQ(decode_report(input, snapshot, {}).failure, ReportDecodeFailure::GenerationMismatch);
  header.generation = generation;
  store();
  auto decoded = decode_report(input, snapshot, {});
  EXPECT_TRUE(decoded.complete());
  EXPECT_TRUE(decoded.records.evidence.empty());
  EXPECT_EQ(decoded.summary.malformed_snapshot_count, 1u);
  window.generation = generation;
  window.publication_state = static_cast<uint32_t>(CausalPublicationState::Publishing);
  store();
  decoded = decode_report(input, snapshot, {});
  EXPECT_TRUE(decoded.records.evidence.empty());
  EXPECT_EQ(decoded.summary.incomplete_snapshot_count, 1u);
  window.publication_state = static_cast<uint32_t>(CausalPublicationState::Ready);
  store();
  decoded = decode_report(input, snapshot, {});
  ASSERT_EQ(decoded.records.evidence.size(), 1u);
  EXPECT_EQ(decoded.records.evidence[0].generation, generation);
  EXPECT_EQ(decoded.records.evidence[0].entry.start_byte, 0x10800u);
  EXPECT_EQ(decoded.records.evidence[0].entry.byte_count, 2u);
  // Atomic metadata must not attach to an old window with an aliased tag.
  const AtomicAttachmentKey key{.generation = generation, .dispatch_id = 11};
  EXPECT_TRUE(atomic_attachment_matches(window, packed, 0, key));
  window.generation = old_generation;
  EXPECT_FALSE(atomic_attachment_matches(window, packed, 0, key));
}

TEST(HsaHooksUnitTest, AutoReportDecoderProducesTypedEventsFailuresAndLoss) {

  AutoReportInventory inventory;
  inventory.access_range_count = 1;
  inventory.range_bank_count = 1;
  inventory.watchpoint_count = 1;
  const auto report = plan_auto_report(inventory);
  ASSERT_TRUE(report.complete());
  ReportSnapshot snapshot;
  snapshot.bytes.resize(report.required_bytes);
  auto header = make_report_header_for_layout(7, 11, report.layout);
  header.causal_window_count = 1;
  header.dropped_window_count = 2;
  std::memcpy(snapshot.bytes.data(), &header, sizeof(header));
  const CausalWindow window{
      .generation = 7,
      .dispatch_id = 11,
      .first_entry = 0,
      .entry_count = 1,
      .publication_state = static_cast<uint32_t>(CausalPublicationState::Ready),
  };
  std::memcpy(snapshot.bytes.data() + report.layout.causal_windows_offset, &window, sizeof(window));
  const uint64_t packed = pack_watchpoint_entry(ShadowAccessKind::Write, 3, 0, 7, 16, 4);
  std::memcpy(snapshot.bytes.data() + report.layout.watchpoints_offset, &packed, sizeof(packed));
  ReportSummary initial_summary;
  initial_summary.allocation_failure_count = 1;
  ReportPipelineInput input;
  input.reader = 101;
  input.source_address = 0x1000;
  input.size = snapshot.bytes.size();
  input.layout = report.layout;
  const auto decoded = decode_report(input, snapshot, initial_summary);
  ASSERT_TRUE(decoded.complete());
  ASSERT_EQ(decoded.records.evidence.size(), 1u);
  EXPECT_EQ(decoded.records.evidence.front().entry.owner_id, 3u);
  EXPECT_EQ(decoded.summary.visible_watchpoint_count, 1u);
  EXPECT_EQ(decoded.summary.dropped_window_count, 2u);
  EXPECT_EQ(decoded.summary.allocation_failure_count, 1u);
  auto mismatched = input;
  ++mismatched.layout.watchpoint_capacity;
  EXPECT_EQ(decode_report(mismatched, snapshot, {}).failure, ReportDecodeFailure::LayoutMismatch);
  snapshot.bytes[0] = 0;
  EXPECT_EQ(decode_report(input, snapshot, {}).failure, ReportDecodeFailure::InvalidHeader);
  snapshot.bytes.resize(sizeof(header) - 1);
  EXPECT_EQ(decode_report(input, snapshot, {}).failure, ReportDecodeFailure::SnapshotTooSmall);
}

TEST(HsaHooksUnitTest, AutoReportPipelineCarriesOptionalMetadata) {
  rocjitsu::consan::hook::ReportPipelineInput input;
  EXPECT_EQ(input.static_metadata, nullptr);
  rocjitsu::consan::hook::AccessStaticMetadata metadata =
      rocjitsu::consan::hook::AccessStaticMetadata{.mappings = {}, .malformed = true};
  input.static_metadata = &metadata;
  EXPECT_TRUE(input.static_metadata->malformed);
  input.static_metadata = nullptr;
  EXPECT_EQ(input.static_metadata, nullptr);
}

TEST(HsaHooksUnitTest, AutoReportRendererConsumesOnlyTypedResultsAndPreservesDiagnostics) {
  rocjitsu::consan::hook::ReportPipelineInput input;
  input.reader = 101;
  input.source_address = 0x2000;
  input.size = 64;
  input.input_fingerprint = "fixture";

  rocjitsu::consan::hook::DecodedReport invalid;
  invalid.failure = rocjitsu::consan::hook::ReportDecodeFailure::InvalidHeader;
  invalid.header.magic = 0x1234;
  invalid.header.abi_version = 9;
  invalid.header.header_size = 7;
  const auto failure =
      rocjitsu::consan::hook::render_report({input, invalid, invalid.summary, nullptr});
  ASSERT_EQ(failure.size(), 1u);
  EXPECT_EQ(failure.front().kind, rocjitsu::consan::hook::ReportDiagnosticKind::Failure);
  EXPECT_EQ(failure.front().text,
            "ConSan auto report reader=101 has invalid header magic=0x00001234 abi=9 "
            "header_size=7");

  rocjitsu::consan::hook::DecodedReport decoded;
  decoded.header.generation = 5;
  rocjitsu::consan::hook::DecodedEvidence evidence;
  evidence.issues.push_back({
      .reason = rocjitsu::consan::hook::EvidenceReason::MalformedWatchpoint,
      .index = 3,
      .words = {0x11, 0x22, 4},
  });
  evidence.evidence.emplace_back();
  decoded.records = std::move(evidence);
  rocjitsu::consan::hook::ConflictAnalysis conflict_analysis =
      rocjitsu::consan::hook::ConflictAnalysis{};
  const auto rendered =
      rocjitsu::consan::hook::render_report({input, decoded, decoded.summary, &conflict_analysis});
  ASSERT_EQ(rendered.size(), 3u);
  EXPECT_EQ(rendered[0].kind, ReportDiagnosticKind::Evidence);
  EXPECT_EQ(rendered[1].kind, ReportDiagnosticKind::Summary);
  EXPECT_EQ(rendered[2].kind, ReportDiagnosticKind::Detail);
  EXPECT_TRUE(std::ranges::any_of(rendered, [](const auto &diagnostic) {
    return diagnostic.kind == rocjitsu::consan::hook::ReportDiagnosticKind::Evidence &&
           diagnostic.text == "ConSan malformed packed watchpoint index=3 low=0x00000011 "
                              "high=0x00000022 epoch=4";
  }));
  EXPECT_TRUE(std::ranges::any_of(rendered, [](const auto &diagnostic) {
    return diagnostic.kind == rocjitsu::consan::hook::ReportDiagnosticKind::Summary &&
           diagnostic.text.starts_with("ConSan auto report reader=101 addr=0x2000 bytes=64 ");
  }));
}

TEST(HsaHooksUnitTest, AutoReportRendererEmitsSummary) {

  ReportPipelineInput input;
  input.reader = 17;
  DecodedReport decoded;
  ConflictAnalysis analysis;
  const auto rendered = render_report({input, decoded, decoded.summary, &analysis});
  const auto summary = std::ranges::find_if(rendered, [](const auto &diagnostic) {
    return diagnostic.kind == ReportDiagnosticKind::Summary;
  });
  ASSERT_NE(summary, rendered.end());
  EXPECT_NE(summary->text.find("watchpoints="), std::string::npos);
}

TEST(HsaHooksUnitTest, AutoReportDetailLoggingIsBoundedIndependentlyOfTraceSize) {
  ReportPipelineInput input;
  input.reader = 17;
  DecodedReport decoded;
  decoded.records.evidence.resize(65);
  ConflictAnalysis analysis;
  const auto rendered = render_report({input, decoded, decoded.summary, &analysis});
  ASSERT_EQ(rendered.size(), 66u); // Summary, 64 access details, and omission notice.
  EXPECT_EQ(rendered.front().kind, ReportDiagnosticKind::Summary);
  for (size_t index = 1; index <= 64; ++index) {
    EXPECT_EQ(rendered[index].kind, ReportDiagnosticKind::Detail);
    EXPECT_TRUE(rendered[index].text.starts_with("ConSan access reader=17 index="));
  }
  EXPECT_EQ(rendered.back().text, "ConSan access reader=17 omitted=1 after log limit=64");
}

TEST(HsaHooksUnitTest, ConSanRejectsInvalidMode) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "magic");

  reset_code_object_observations();
  FakeApiTable api;
  const auto original_load = api.core.hsa_executable_load_agent_code_object_fn;
  InstalledDbiHook hook(api);
  EXPECT_FALSE(hook.installed());
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn, original_load);
}

TEST(HsaHooksUnitTest, ConSanParsesAndNormalizesExactKernelAllowlist) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar allowlist("RJ_CONSAN_KERNEL_ALLOWLIST",
                         " selected_kernel.kd,second_kernel,selected_kernel ");
  g_transform_override_result.outcome = rocjitsu::consan::TransformOutcome::Unchanged;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(g_transform_override_kernel_allowlists.size(), 1u);
  EXPECT_EQ(g_transform_override_kernel_allowlists.front(),
            (std::vector<std::string>{"selected_kernel", "second_kernel"}));
}

TEST(HsaHooksUnitTest, ConSanParsesExactKernelAllowlistFileWithoutCommaAmbiguity) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  rocjitsu::test::ScopedTempFile names("rocjitsu-consan-kernel-allowlist-");
  names.write("void templated_kernel<int, float>(int).kd\nsecond_kernel.kd\n");
  ScopedEnvVar inline_allowlist("RJ_CONSAN_KERNEL_ALLOWLIST", nullptr);
  ScopedEnvVar allowlist_file("RJ_CONSAN_KERNEL_ALLOWLIST_FILE", names.path().c_str());
  g_transform_override_result.outcome = rocjitsu::consan::TransformOutcome::Unchanged;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(g_transform_override_kernel_allowlists.size(), 1u);
  EXPECT_EQ(g_transform_override_kernel_allowlists.front(),
            (std::vector<std::string>{"void templated_kernel<int, float>(int)", "second_kernel"}));
}

TEST(HsaHooksUnitTest, ConSanAllowlistSkipsUnmatchedObjectBeforeWaitcheckAndTransform) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  ScopedEnvVar allowlist("RJ_CONSAN_KERNEL_ALLOWLIST", "selected_kernel");
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  g_transform_override_result.outcome = rocjitsu::consan::TransformOutcome::Unchanged;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  const std::vector<uint8_t> original =
      rocjitsu::waitcheck_test::make_gfx1201_multi_kernel_code_object(
          {{"large_unrelated_kernel", {0xBFB00000u}}});
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
      hsa_executable_t{7}, kHostAgent, reader, nullptr, nullptr);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  EXPECT_TRUE(g_transform_override_flavors.empty());
  EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{reader.handle});
  EXPECT_NE(log.find("ConSan kernel allowlist prefilter reader="), std::string::npos) << log;
  EXPECT_NE(log.find("outcome=skipped reason=no-matching-entry"), std::string::npos) << log;
  EXPECT_EQ(log.find("ConSan waitcheck timing"), std::string::npos) << log;
  EXPECT_EQ(log.find("ConSan patch begin"), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, ConSanAllowlistRetainsMatchedObjectWaitcheckAndTransform) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  ScopedEnvVar allowlist("RJ_CONSAN_KERNEL_ALLOWLIST", "selected_kernel.kd");
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  g_transform_override_result.outcome = rocjitsu::consan::TransformOutcome::Unchanged;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  std::vector<uint32_t> hazardous_unrelated_kernel;
  rocjitsu::waitcheck_test::append_inst(hazardous_unrelated_kernel,
                                        rocjitsu::waitcheck_test::global_load_b32(0));
  rocjitsu::waitcheck_test::append_inst(hazardous_unrelated_kernel,
                                        rocjitsu::waitcheck_test::v_mov_b32(1, 0));
  const std::vector<uint8_t> original =
      rocjitsu::waitcheck_test::make_gfx1201_multi_kernel_code_object(
          {{"selected_kernel", {0xBFB00000u}},
           {"hazardous_unrelated_kernel", hazardous_unrelated_kernel}});
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
      hsa_executable_t{7}, kHostAgent, reader, nullptr, nullptr);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  expect_transform_profile(kConSanHookProfiles[1]);
  ASSERT_EQ(g_transform_override_kernel_allowlists.size(), 1u);
  EXPECT_EQ(g_transform_override_kernel_allowlists.front(),
            std::vector<std::string>{"selected_kernel"});
  EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{reader.handle});
  EXPECT_EQ(log.find("outcome=skipped reason=no-matching-entry"), std::string::npos) << log;
  EXPECT_NE(log.find("outcome=passed"), std::string::npos) << log;
  EXPECT_NE(log.find("kernels=1/2"), std::string::npos) << log;
  EXPECT_EQ(log.find("reason=wait-hazard"), std::string::npos) << log;
  EXPECT_NE(log.find("ConSan waitcheck timing"), std::string::npos) << log;
  EXPECT_NE(log.find("ConSan patch begin"), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, ConSanAllowlistMatchesDemangledProfilerKernelName) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  ScopedEnvVar allowlist("RJ_CONSAN_KERNEL_ALLOWLIST", "selected_kernel()");
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  g_transform_override_result.outcome = rocjitsu::consan::TransformOutcome::Unchanged;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  const std::vector<uint8_t> original =
      rocjitsu::waitcheck_test::make_gfx1201_multi_kernel_code_object(
          {{"_Z15selected_kernelv", {0xBFB00000u}}});
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
      hsa_executable_t{7}, kHostAgent, reader, nullptr, nullptr);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  expect_transform_profile(kConSanHookProfiles[1]);
  EXPECT_EQ(log.find("outcome=skipped reason=no-matching-entry"), std::string::npos) << log;
  EXPECT_NE(log.find("ConSan patch begin"), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, ConSanRejectsEmptyKernelAllowlistEntry) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar allowlist("RJ_CONSAN_KERNEL_ALLOWLIST", "selected_kernel,,second_kernel");

  FakeApiTable api;
  const auto original_load = api.core.hsa_executable_load_agent_code_object_fn;
  InstalledDbiHook hook(api);
  EXPECT_FALSE(hook.installed());
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn, original_load);
}

TEST(HsaHooksUnitTest, ConSanRejectsZeroFaultReservationTimeout) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar reservation_timeout("RJ_CONSAN_FAULT_RESERVATION_TIMEOUT_MS", "0");

  reset_code_object_observations();
  FakeApiTable api;
  const auto original_load = api.core.hsa_executable_load_agent_code_object_fn;
  InstalledDbiHook hook(api);
  EXPECT_FALSE(hook.installed());
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn, original_load);
}

TEST(HsaHooksUnitTest, ConSanRejectsInvalidPolicy) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "fatal-races");

  reset_code_object_observations();
  FakeApiTable api;
  const auto original_load = api.core.hsa_executable_load_agent_code_object_fn;
  InstalledDbiHook hook(api);
  EXPECT_FALSE(hook.installed());
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn, original_load);
}

TEST(HsaHooksUnitTest, ConSanStrictPolicyRequiresCompleteInstrumentationButNotCleanDiagnostics) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "strict");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", nullptr);
  ScopedEnvVar require_patch("RJ_CONSAN_REQUIRE_PATCH", nullptr);
  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", nullptr);
  ScopedEnvVar forbid_diagnostics("RJ_CONSAN_FORBID_DIAGNOSTICS", nullptr);
  ScopedEnvVar forbid_overflow("RJ_CONSAN_FORBID_OVERFLOW", nullptr);

  ASSERT_EXIT(
      {
        FakeApiTable api;
        InstalledDbiHook hook(api);
        if (!hook.installed())
          std::_Exit(1);
      },
      testing::ExitedWithCode(86),
      "installed ConSan hook.*policy=strict.*fail_closed=true require_patch=true.*"
      "require_records=true.*forbid_diagnostics=false.*forbid_overflow=true");
}

TEST(HsaHooksUnitTest, ConSanStrictPolicyTerminatesAtRejectedCodeObjectLoad) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[0], false);
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "strict");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", nullptr);
  ScopedEnvVar require_patch("RJ_CONSAN_REQUIRE_PATCH", nullptr);
  g_transform_override_result.outcome = rocjitsu::consan::TransformOutcome::Unsupported;

  ASSERT_EXIT(([] {
                FakeApiTable api;
                InstalledDbiHook hook(api);
                if (!hook.installed())
                  std::_Exit(1);
                std::array<uint8_t, 8> original{};
                original[0] = 0x7f;
                original[1] = 'E';
                original[2] = 'L';
                original[3] = 'F';
                hsa_code_object_reader_t reader{};
                if (api.core.hsa_code_object_reader_create_from_memory_fn(
                        original.data(), original.size(), &reader) != HSA_STATUS_SUCCESS)
                  std::_Exit(2);
                (void)api.core.hsa_executable_load_agent_code_object_fn(
                    hsa_executable_t{7}, kHostAgent, reader, nullptr, nullptr);
                std::_Exit(3);
              }()),
              testing::ExitedWithCode(92),
              "ConSan load rejection.*reason=non-installable-transform-outcome.*"
              "policy=strict action=terminate exit_code=92");
}

void expect_transform_profile(const ConSanHookProfile &profile, size_t expected_calls) {
  ASSERT_EQ(g_transform_override_flavors.size(), expected_calls);
  ASSERT_EQ(g_transform_override_abort_unmatched_waits.size(), expected_calls);
  EXPECT_TRUE(std::ranges::all_of(g_transform_override_flavors, [&](const auto mode) {
    return mode == profile.expected_flavor;
  }));
  EXPECT_TRUE(std::ranges::none_of(g_transform_override_abort_unmatched_waits,
                                   [](bool value) { return value; }));
  ASSERT_EQ(g_transform_override_track_barriers.size(), expected_calls);
  ASSERT_EQ(g_transform_override_track_atomics.size(), expected_calls);
  ASSERT_EQ(g_transform_override_runtime_sample_strides.size(), expected_calls);
  ASSERT_EQ(g_transform_override_patched_image_growth_limits.size(), expected_calls);
  const bool expected_sync_defaults = profile.expected_flavor == rocjitsu::consan::Mode::Default;
  EXPECT_EQ(g_transform_override_track_barriers.front(), expected_sync_defaults);
  EXPECT_EQ(g_transform_override_track_atomics.front(), expected_sync_defaults);
  const uint32_t expected_runtime_sample_stride =
      profile.expected_flavor == rocjitsu::consan::Mode::Default ? 256u : 1u;
  EXPECT_EQ(g_transform_override_runtime_sample_strides.front(), expected_runtime_sample_stride);
  const auto &growth = g_transform_override_patched_image_growth_limits.front();
  EXPECT_EQ(growth.kind, rocjitsu::consan::PatchedImageGrowthLimitKind::AbsoluteBytes);
  EXPECT_EQ(growth.absolute_bytes, rocjitsu::consan::kDefaultMaxPatchedImageGrowthBytes);
}

void run_hook_load_case(const ConSanHookProfile &profile, bool fail_closed,
                        rocjitsu::consan::TransformArtifacts transform_result,
                        hsa_status_t expected_load_status, uint64_t expected_loaded_reader,
                        std::span<const uint8_t> expected_replacement = {},
                        bool fail_replacement_reader_create = false, bool use_auto_report = false,
                        hook::LogSinkOverride log_sink_override = nullptr,
                        size_t expected_transform_calls = 1u) {
  reset_code_object_observations();
  g_fail_replacement_reader_create = fail_replacement_reader_create;
  configure_consan_profile(profile, fail_closed);
  std::optional<ScopedEnvVar> report_buffer;
  std::optional<ScopedEnvVar> report_buffer_size;
  std::optional<ScopedEnvVar> auto_report_buffer_size;
  if (use_auto_report) {
    report_buffer.emplace("RJ_CONSAN_REPORT_BUFFER", nullptr);
    report_buffer_size.emplace("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
    auto_report_buffer_size.emplace("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "65536");
  }
  g_transform_override_result = std::move(transform_result);
  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << profile.name << ": " << hook.error();
  hook.set_log_sink_override(log_sink_override);

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t original_reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &original_reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(original_reader.handle, 101u);

  const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
      hsa_executable_t{7}, kHostAgent, original_reader, nullptr, nullptr);
  EXPECT_EQ(status, expected_load_status) << profile.name;
  expect_transform_profile(profile, expected_transform_calls);
  if (use_auto_report) {
    ASSERT_EQ(g_transform_override_report_sizes.size(), 1u);
    EXPECT_EQ(g_transform_override_report_sizes.front(), 0u);
  }
  if (expected_loaded_reader == 0) {
    EXPECT_TRUE(g_loaded_code_object_readers.empty()) << profile.name;
  } else {
    ASSERT_EQ(g_loaded_code_object_readers.size(), 1u) << profile.name;
    EXPECT_EQ(g_loaded_code_object_readers.front(), expected_loaded_reader) << profile.name;
  }
  if (!expected_replacement.empty()) {
    ASSERT_EQ(g_code_object_reader_inputs.size(), 2u) << profile.name;
    EXPECT_EQ(g_code_object_reader_inputs.back(),
              std::vector<uint8_t>(expected_replacement.begin(), expected_replacement.end()))
        << profile.name;
    EXPECT_EQ(g_destroyed_code_object_readers, std::vector<uint64_t>{102u}) << profile.name;
  }
  if (fail_replacement_reader_create) {
    EXPECT_EQ(g_code_object_reader_create_calls, 2) << profile.name;
    ASSERT_EQ(g_code_object_reader_inputs.size(), 1u) << profile.name;
    EXPECT_TRUE(g_destroyed_code_object_readers.empty()) << profile.name;
  }
  ASSERT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_destroyed_executables, std::vector<uint64_t>{7u}) << profile.name;
  if (!expected_replacement.empty()) {
    EXPECT_TRUE(replacement_storage_valid_at_destroy(7u)) << profile.name;
  }
}

TEST(HsaHooksUnitTest, ConSanTransformRejectionReportsStableTypedCause) {
  for (const auto [cause, expected] : std::array{
           std::pair{rocjitsu::consan::TransformFailureCause::PatchedImageGrowthLimit,
                     std::string_view{"patched-image-growth-limit"}},
           std::pair{rocjitsu::consan::TransformFailureCause::OverlappingPatchRanges,
                     std::string_view{"overlapping-patch-ranges"}},
       }) {
    SCOPED_TRACE(expected);
    rocjitsu::consan::TransformArtifacts result;
    result.outcome = rocjitsu::consan::TransformOutcome::Invalid;
    result.errors.emplace_back("synthetic typed transform failure");
    result.transform_failure_cause = cause;

    testing::internal::CaptureStderr();
    run_hook_load_case(kConSanHookProfiles[0], /*fail_closed=*/true, std::move(result),
                       HSA_STATUS_ERROR_INVALID_CODE_OBJECT, /*expected_loaded_reader=*/0u);
    const std::string log = testing::internal::GetCapturedStderr();
    EXPECT_NE(log.find("ConSan load rejection reader=101 reason=transform-error status=4112 "
                       "policy=default action=return-error exit_code=none cause=" +
                       std::string(expected)),
              std::string::npos)
        << log;
  }
}

void reset_pool_blocker(bool enabled) {
  std::lock_guard lock(g_pool_mutex);
  g_block_guest_pool_iteration = enabled;
  g_guest_pool_iteration_entered = false;
  g_release_guest_pool_iteration = false;
  g_fail_guest_pool_iteration_once = false;
}

void release_pool_blocker() {
  {
    std::lock_guard lock(g_pool_mutex);
    g_release_guest_pool_iteration = true;
  }
  g_pool_cv.notify_all();
}

void reset_agent_blocker(bool enabled) {
  std::lock_guard lock(g_agent_mutex);
  g_block_agent_iteration = enabled;
  g_agent_iteration_entered = false;
  g_release_agent_iteration = false;
  g_fail_agent_iteration = false;
  g_fake_iterate_agents_calls = 0;
}

hsa_status_t HSA_API collect_agent_handles(hsa_agent_t agent, void *data) {
  static_cast<std::vector<uint64_t> *>(data)->push_back(agent.handle);
  return HSA_STATUS_SUCCESS;
}

void release_agent_blocker() {
  {
    std::lock_guard lock(g_agent_mutex);
    g_release_agent_iteration = true;
  }
  g_agent_cv.notify_all();
}

void expect_batch_copy_forwarding(const hsa_amd_memory_copy_op_t &op,
                                  const std::vector<uint64_t> &expected_src_agents,
                                  const std::vector<uint64_t> &expected_dst_agents) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  g_last_batch_src_agents.clear();
  g_last_batch_dst_agents.clear();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_memory_async_batch_copy_fn, fake_amd_memory_async_batch_copy);

  EXPECT_EQ(api.amd.hsa_amd_memory_async_batch_copy_fn(&op, 1, 0, nullptr), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_batch_src_agents, expected_src_agents);
  EXPECT_EQ(g_last_batch_dst_agents, expected_dst_agents);
}

rocjitsu::consan::SyncDecodeResult atomic(rocjitsu::consan::SyncRole role,
                                          rocjitsu::consan::SyncScope scope,
                                          rocjitsu::consan::SyncOutcome outcome,
                                          uint64_t address = 0x1000, uint32_t byte_count = 4,
                                          uint32_t epoch = 7) {
  return {
      .metadata =
          {
              .address = address,
              .byte_count = byte_count,
              .kind = rocjitsu::consan::SyncMetadataKind::Atomic,
              .role = role,
              .scope = scope,
              .outcome = outcome,
              .epoch_before = epoch,
              .epoch_after = epoch,
          },
      .classification = rocjitsu::consan::SyncClassification::Valid,
  };
}

TEST(HsaHooksUnitTest, ConSanLoaderHonorsAllTypedOutcomesAcrossAllProfiles) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", nullptr);
  ScopedEnvVar policy("RJ_CONSAN_POLICY", nullptr);
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", nullptr);
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", nullptr);

  for (const ConSanHookProfile &profile : kConSanHookProfiles) {
    SCOPED_TRACE(profile.name);

    rocjitsu::consan::TransformArtifacts unchanged;
    unchanged.outcome = rocjitsu::consan::TransformOutcome::Unchanged;
    run_hook_load_case(profile, false, unchanged, HSA_STATUS_SUCCESS, 101u);
    run_hook_load_case(profile, true, unchanged, HSA_STATUS_SUCCESS, 101u);

    rocjitsu::consan::TransformArtifacts unsupported;
    unsupported.outcome = rocjitsu::consan::TransformOutcome::Unsupported;
    run_hook_load_case(profile, false, unsupported, HSA_STATUS_SUCCESS, 101u);
    run_hook_load_case(profile, true, unsupported, HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u);

    rocjitsu::consan::TransformArtifacts invalid;
    invalid.outcome = rocjitsu::consan::TransformOutcome::Invalid;
    run_hook_load_case(profile, false, invalid, HSA_STATUS_SUCCESS, 101u);
    run_hook_load_case(profile, true, invalid, HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u);

    const std::array<uint8_t, 7> replacement = {'p', 'a', 't', 'c', 'h', 'e', 'd'};
    rocjitsu::consan::TransformArtifacts modified;
    install_consan_test_program_identity(modified, ROCJITSU_CODE_ARCH_CDNA3,
                                         ROCJITSU_CODE_TARGET_GFX942);
    modified.outcome = rocjitsu::consan::TransformOutcome::ModifiedValid;
    modified.replacement.assign(replacement.begin(), replacement.end());
    if (profile.expected_flavor == rocjitsu::consan::Mode::SuperCollider) {
      rocjitsu::consan::ObservationPlan plan;
      plan.mode = rocjitsu::consan::Mode::SuperCollider;
      ASSERT_TRUE(plan.valid());
      modified.coverage_ledger = rocjitsu::consan::CoverageLedger(std::move(plan));
    }
    run_hook_load_case(profile, false, modified, HSA_STATUS_SUCCESS, 102u, replacement);
    run_hook_load_case(profile, true, modified, HSA_STATUS_SUCCESS, 102u, replacement);
    run_hook_load_case(profile, false, modified, HSA_STATUS_SUCCESS, 101u, {}, true);
    run_hook_load_case(profile, true, modified, HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u, {}, true);

    rocjitsu::consan::TransformArtifacts corrupt = modified;
    corrupt.replacement.clear();
    run_hook_load_case(profile, false, corrupt, HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u);
    run_hook_load_case(profile, true, corrupt, HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u);
  }
}

TEST(HsaHooksUnitTest, ConSanLogsSharedNamesForParsedTypedIdentities) {
  struct TargetCase {
    rj_code_target_id_t target;
    rj_code_arch_t semantic_arch;
    const char *target_name;
    const char *display_arch_name;
  };
  constexpr std::array cases = {
      TargetCase{ROCJITSU_CODE_TARGET_GFX942, ROCJITSU_CODE_ARCH_CDNA3, "gfx942", "cdna3"},
      TargetCase{ROCJITSU_CODE_TARGET_GFX950, ROCJITSU_CODE_ARCH_CDNA4, "gfx950", "cdna4"},
      TargetCase{ROCJITSU_CODE_TARGET_GFX1201, ROCJITSU_CODE_ARCH_RDNA4, "gfx1201", "rdna4"},
      TargetCase{ROCJITSU_CODE_TARGET_GFX1250, ROCJITSU_CODE_ARCH_CDNA5, "gfx1250", "cdna5"},
      TargetCase{ROCJITSU_CODE_TARGET_GFX1200, ROCJITSU_CODE_ARCH_INVALID, "gfx1200", "rdna4"},
      TargetCase{ROCJITSU_CODE_TARGET_INVALID, ROCJITSU_CODE_ARCH_INVALID, "invalid", "invalid"},
  };
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");

  for (const TargetCase &target_case : cases) {
    SCOPED_TRACE(target_case.target_name);
    rocjitsu::consan::TransformArtifacts result;
    install_consan_test_program_identity(result, target_case.semantic_arch, target_case.target);
    result.outcome = target_case.semantic_arch == ROCJITSU_CODE_ARCH_INVALID
                         ? rocjitsu::consan::TransformOutcome::Unsupported
                         : rocjitsu::consan::TransformOutcome::Unchanged;

    testing::internal::CaptureStderr();
    run_hook_load_case(kConSanHookProfiles[1], false, result, HSA_STATUS_SUCCESS, 101u);
    const std::string log = testing::internal::GetCapturedStderr();
    const std::string expected =
        "target=" + std::string(target_case.target_name) + " arch=" + target_case.display_arch_name;
    EXPECT_NE(log.find(expected), std::string::npos) << log;
  }
}

TEST(HsaHooksUnitTest, ConSanProductionUnsupportedTargetPassesThroughWhenFailOpen) {
  const std::vector<uint8_t> unsupported =
      rocjitsu::waitcheck_test::make_gfx1200_code_object({0xBFB00000u});
  rocjitsu::consan::Request request;
  request.mode = rocjitsu::consan::Mode::Default;
  const TransformResult direct =
      transform(unsupported, request, TransformPolicy{}, enabled_consan_runtime_policy(),
                rocjitsu::consan::DebugOverrides{}, complete_consan_runtime_capabilities(), {});
  ASSERT_EQ(direct.outcome, rocjitsu::consan::TransformOutcome::Unsupported);
  ASSERT_TRUE(direct.program_inventory.code_object_parsed());
  ASSERT_EQ(direct.program_inventory.arch(), ROCJITSU_CODE_ARCH_INVALID);
  ASSERT_FALSE(direct.program_inventory.text_sections().empty());
  ASSERT_FALSE(direct.program_inventory.kernels().empty());
  ASSERT_FALSE(direct.program_inventory.semantic_arch_required());
  ASSERT_TRUE(direct.program_inventory.has_resolved_semantic_arch());

  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  for (const ConSanHookProfile &profile : kConSanHookProfiles) {
    SCOPED_TRACE(profile.name);
    reset_code_object_observations();
    configure_consan_profile(profile, /*fail_closed=*/false);
    FakeApiTable api;
    testing::internal::CaptureStderr();
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << profile.name << ": " << hook.error();
    hook.use_production_transform();

    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(unsupported.data(),
                                                                    unsupported.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{101u});
    const std::string log = testing::internal::GetCapturedStderr();
    EXPECT_NE(log.find("target=gfx1200 arch=rdna4"), std::string::npos) << log;
  }
}

TEST(HsaHooksUnitTest, ConSanProductionTransformUsesDerivedMajorImageAdmission) {
  const std::vector<uint32_t> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = rocjitsu::waitcheck_test::make_gfx1201_code_object(text_words);
  rocjitsu::consan::Request request;
  request.mode = rocjitsu::consan::Mode::SuperCollider;
  request.probe_lds_check_trap = true;
  request.supercollider_delay_nops = 2;
  request.supercollider_evidence_mode = rocjitsu::consan::SuperColliderEvidenceMode::TrapOnly;
  rocjitsu::consan::DebugOverrides debug;
  debug.scratch_vgpr = 3;
  TransformPolicy transform_policy;
  transform_policy.patched_image_growth_limit.absolute_bytes = 0;
  const TransformResult direct =
      transform(bytes, request, transform_policy, enabled_consan_runtime_policy(), debug,
                complete_consan_runtime_capabilities(), {});
  ASSERT_EQ(direct.outcome, rocjitsu::consan::TransformOutcome::ModifiedValid)
      << "errors=" << testing::PrintToString(direct.errors)
      << " warnings=" << testing::PrintToString(direct.warnings)
      << " intents=" << direct.observation_plan().probe_intents.size()
      << " patches=" << transform_diagnostic_report(direct).patches.size();
  ASSERT_EQ(direct.replacement.size(), bytes.size());
  const auto direct_report = transform_diagnostic_report(direct);
  ASSERT_EQ(direct_report.patches.size(), 1u);
  EXPECT_EQ(direct_report.patches.front().trampoline_size, 0u);
  EXPECT_GT(direct_report.patches.front().original_size, 2u * sizeof(uint32_t));

  const auto estimate = hook::transform_major_image_reservation(
      bytes.size(), transform_policy.patched_image_growth_limit);
  ASSERT_TRUE(estimate);
  ASSERT_GT(estimate->reservation_bytes, 0u);

  {
    reset_code_object_observations();
    configure_consan_profile(kConSanHookProfiles[0], false);
    const std::string limit = std::to_string(estimate->reservation_bytes - 1u);
    ScopedEnvVar probe("RJ_CONSAN_PROBE_LDS_CHECK_TRAP", "1");
    ScopedEnvVar scratch("RJ_CONSAN_TMP_VGPR", "3");
    ScopedEnvVar delay("RJ_CONSAN_SC_DELAY", "2");
    ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", "trap");
    ScopedEnvVar growth("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES", "0");
    ScopedEnvVar process_limit("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES", limit.c_str());
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    hook.use_production_transform();
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(
        api.core.hsa_code_object_reader_create_from_memory_fn(bytes.data(), bytes.size(), &reader),
        HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{reader.handle});
  }

  {
    reset_code_object_observations();
    configure_consan_profile(kConSanHookProfiles[0], false);
    const std::string limit = std::to_string(estimate->reservation_bytes);
    ScopedEnvVar probe("RJ_CONSAN_PROBE_LDS_CHECK_TRAP", "1");
    ScopedEnvVar scratch("RJ_CONSAN_TMP_VGPR", "3");
    ScopedEnvVar delay("RJ_CONSAN_SC_DELAY", "2");
    ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", "trap");
    ScopedEnvVar growth("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES", "0");
    ScopedEnvVar process_limit("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES", limit.c_str());
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    hook.use_production_transform();
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(
        api.core.hsa_code_object_reader_create_from_memory_fn(bytes.data(), bytes.size(), &reader),
        HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{102u});
    EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  }
}

TEST(HsaHooksUnitTest, ConSanWaitcheckReportsHazardBeforeTransformRegardlessOfWaitcheckEnv) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  ScopedEnvVar waitcheck_enable("ROCJITSU_WAITCHECK", "0");
  ScopedEnvVar waitcheck_mode("ROCJITSU_WAITCHECK_MODE", "dispatch");
  ScopedEnvVar waitcheck_fail("ROCJITSU_WAITCHECK_FAIL", "0");
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  g_transform_override_result.outcome = rocjitsu::consan::TransformOutcome::Unchanged;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  std::vector<uint32_t> clean_kernel;
  rocjitsu::waitcheck_test::append_inst(clean_kernel, rocjitsu::waitcheck_test::global_load_b32(0));
  rocjitsu::waitcheck_test::append_inst(clean_kernel, rocjitsu::waitcheck_test::s_wait_loadcnt(0));
  rocjitsu::waitcheck_test::append_inst(clean_kernel, rocjitsu::waitcheck_test::v_mov_b32(1, 0));
  std::vector<uint32_t> hazardous_kernel;
  rocjitsu::waitcheck_test::append_inst(hazardous_kernel,
                                        rocjitsu::waitcheck_test::global_load_b32(0));
  rocjitsu::waitcheck_test::append_inst(hazardous_kernel,
                                        rocjitsu::waitcheck_test::v_mov_b32(1, 0));
  const std::vector<uint8_t> original =
      rocjitsu::waitcheck_test::make_gfx1201_multi_kernel_code_object(
          {{"clean_kernel", clean_kernel}, {"hazardous_kernel", hazardous_kernel}});
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
      hsa_executable_t{7}, kHostAgent, reader, nullptr, nullptr);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  expect_transform_profile(kConSanHookProfiles[1]);
  EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{101u});
  const size_t waitcheck_pos = log.find("ConSan preflight reported reader=101 target=gfx1201 "
                                        "reason=wait-hazard diagnostics=1");
  const size_t consan_pos = log.find("ConSan patch begin reader=101");
  EXPECT_NE(waitcheck_pos, std::string::npos) << log;
  EXPECT_NE(consan_pos, std::string::npos) << log;
  EXPECT_LT(waitcheck_pos, consan_pos) << log;
}

TEST(HsaHooksUnitTest, ConSanWaitcheckReportsAnalysisFailureBeforeTransform) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  g_transform_override_result.outcome = rocjitsu::consan::TransformOutcome::Unchanged;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  const std::vector<uint8_t> original =
      rocjitsu::waitcheck_test::make_gfx1201_invalid_instruction_code_object();
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
      hsa_executable_t{7}, kHostAgent, reader, nullptr, nullptr);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  expect_transform_profile(kConSanHookProfiles[1]);
  EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{101u});
  const size_t waitcheck_pos = log.find("ConSan preflight reported reader=101 target=gfx1201 "
                                        "reason=analysis-failed");
  const size_t consan_pos = log.find("ConSan patch begin reader=101");
  EXPECT_NE(waitcheck_pos, std::string::npos) << log;
  EXPECT_NE(consan_pos, std::string::npos) << log;
  EXPECT_LT(waitcheck_pos, consan_pos) << log;
}

TEST(HsaHooksUnitTest, ConSanWaitcheckPassesBeforeTransformForCleanCodeObject) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  g_transform_override_result.outcome = rocjitsu::consan::TransformOutcome::Unchanged;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  const std::vector<uint8_t> original =
      rocjitsu::waitcheck_test::make_gfx1201_correct_wait_code_object();
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
      hsa_executable_t{7}, kHostAgent, reader, nullptr, nullptr);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  expect_transform_profile(kConSanHookProfiles[1]);
  EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{101u});
  const size_t waitcheck_pos = log.find("waitcheck preflight reader=101 target=gfx1201 "
                                        "outcome=passed");
  const size_t consan_pos = log.find("ConSan patch begin reader=101");
  EXPECT_NE(waitcheck_pos, std::string::npos) << log;
  EXPECT_NE(consan_pos, std::string::npos) << log;
  EXPECT_LT(waitcheck_pos, consan_pos) << log;
}

TEST(HsaHooksUnitTest, ConSanRequirePatchUsesTypedCoverageLedger) {
  ScopedEnvVar require_patch("RJ_CONSAN_REQUIRE_PATCH", "1");
  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", nullptr);

  rocjitsu::consan::TransformArtifacts structural_only;
  install_consan_test_program_identity(structural_only, ROCJITSU_CODE_ARCH_RDNA4,
                                       ROCJITSU_CODE_TARGET_GFX1201);
  structural_only.outcome = rocjitsu::consan::TransformOutcome::ModifiedValid;
  structural_only.replacement = {0x7f, 'E', 'L', 'F', 'p', 'r', 'o', 'l'};
  rocjitsu::consan::PatchInfo prologue_patch;
  prologue_patch.phase = rocjitsu::consan::PatchPhase::Instrumentation;
  prologue_patch.kind = rocjitsu::consan::PatchKind::KernelEntryOwnerEpochPrologue;
  structural_only.patches.push_back(prologue_patch);

  for (size_t i = 1; i < kConSanHookProfiles.size(); ++i) {
    SCOPED_TRACE(kConSanHookProfiles[i].name);
    const auto capability_engine = enabled_mode(kConSanHookProfiles[i].expected_flavor);
    ASSERT_TRUE(capability_engine.has_value());
    constexpr auto intent_kind = rocjitsu::consan::ProbeIntentKind::Access;

    rocjitsu::consan::TransformArtifacts pending = structural_only;
    install_test_access_coverage(pending, 1u, rocjitsu::consan::SiteDecisionKind::Admitted,
                                 rocjitsu::consan::AccessPolicyReason::None,
                                 rocjitsu::consan::LoweringOutcomeKind::Pending, *capability_engine,
                                 intent_kind);
    run_hook_load_case(kConSanHookProfiles[i], false, pending, HSA_STATUS_ERROR_INVALID_CODE_OBJECT,
                       0u);

    rocjitsu::consan::TransformArtifacts resource_rejected = structural_only;
    install_test_access_coverage(
        resource_rejected, 1u, rocjitsu::consan::SiteDecisionKind::Admitted,
        rocjitsu::consan::AccessPolicyReason::None,
        rocjitsu::consan::LoweringOutcomeKind::ResourceRejected, *capability_engine, intent_kind);
    run_hook_load_case(kConSanHookProfiles[i], false, resource_rejected,
                       HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u);
  }

  rocjitsu::consan::TransformArtifacts site_patched = structural_only;
  install_test_access_coverage(site_patched, 1u, rocjitsu::consan::SiteDecisionKind::Admitted,
                               rocjitsu::consan::AccessPolicyReason::None,
                               rocjitsu::consan::LoweringOutcomeKind::Instrumented);
  rocjitsu::consan::PatchInfo site_patch;
  site_patch.phase = rocjitsu::consan::PatchPhase::Instrumentation;
  site_patch.kind = rocjitsu::consan::PatchKind::TrampolineSyncMetadata;
  site_patched.patches.push_back(site_patch);
  run_hook_load_case(kConSanHookProfiles[1], false, site_patched, HSA_STATUS_SUCCESS, 102u,
                     site_patched.replacement);
}

TEST(HsaHooksUnitTest, ConSanRequirePatchUsesTypedSuperColliderCoverageLedger) {
  ScopedEnvVar require_patch("RJ_CONSAN_REQUIRE_PATCH", "1");
  ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", "trap");
  rocjitsu::consan::TransformArtifacts structural_only;
  install_consan_test_program_identity(structural_only, ROCJITSU_CODE_ARCH_CDNA4,
                                       ROCJITSU_CODE_TARGET_GFX950);
  structural_only.outcome = rocjitsu::consan::TransformOutcome::ModifiedValid;
  structural_only.replacement = {0x7f, 'E', 'L', 'F', 'd', '1', '6'};
  install_test_access_coverage(structural_only, 1u, rocjitsu::consan::SiteDecisionKind::Admitted,
                               rocjitsu::consan::AccessPolicyReason::None,
                               rocjitsu::consan::LoweringOutcomeKind::Pending,
                               rocjitsu::consan::Mode::SuperCollider,
                               rocjitsu::consan::ProbeIntentKind::RedundantAccessObservation);
  rocjitsu::consan::PatchInfo structural_patch;
  structural_patch.phase = rocjitsu::consan::PatchPhase::Instrumentation;
  structural_patch.kind = rocjitsu::consan::PatchKind::TrampolineSuperColliderPerturbation;
  structural_only.patches.push_back(structural_patch);

  run_hook_load_case(kConSanHookProfiles[0], false, structural_only,
                     HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u, {}, false, false, nullptr, 2u);

  rocjitsu::consan::TransformArtifacts placement_rejected = structural_only;
  ASSERT_TRUE(
      publish_test_lowering_outcome(placement_rejected.coverage_ledger, {0u},
                                    rocjitsu::consan::LoweringOutcomeKind::PlacementRejected));
  run_hook_load_case(kConSanHookProfiles[0], false, placement_rejected,
                     HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u, {}, false, false, nullptr, 2u);

  rocjitsu::consan::TransformArtifacts unresolved = structural_only;
  unresolved.coverage_ledger = {};
  run_hook_load_case(kConSanHookProfiles[0], false, unresolved,
                     HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u);

  rocjitsu::consan::TransformArtifacts valid_empty = structural_only;
  rocjitsu::consan::ObservationPlan valid_empty_plan;
  valid_empty_plan.mode = rocjitsu::consan::Mode::SuperCollider;
  ASSERT_TRUE(valid_empty_plan.valid());
  valid_empty.coverage_ledger = rocjitsu::consan::CoverageLedger(std::move(valid_empty_plan));
  run_hook_load_case(kConSanHookProfiles[0], false, valid_empty, HSA_STATUS_SUCCESS, 102u,
                     valid_empty.replacement);

  rocjitsu::consan::TransformArtifacts resource_rejected = structural_only;
  ASSERT_TRUE(
      publish_test_lowering_outcome(resource_rejected.coverage_ledger, {0u},
                                    rocjitsu::consan::LoweringOutcomeKind::ResourceRejected));
  run_hook_load_case(kConSanHookProfiles[0], false, resource_rejected, HSA_STATUS_SUCCESS, 102u,
                     resource_rejected.replacement, false, false, nullptr, 2u);
}

rocjitsu::consan::TransformArtifacts fault_test_result_for_arch(rj_code_arch_t arch) {
  rocjitsu::consan::TransformArtifacts result;
  install_consan_test_program_identity(result, arch, ROCJITSU_CODE_TARGET_GFX950,
                                       /*semantic_arch_required=*/true);
  result.outcome = rocjitsu::consan::TransformOutcome::ModifiedValid;
  result.replacement = {0x7f, 'E', 'L', 'F', 'f', 'a', 'u', 'l', 't'};
  rocjitsu::consan::ObservationPlan plan;
  plan.mode = rocjitsu::consan::Mode::Default;
  assert(plan.valid());
  result.coverage_ledger = rocjitsu::consan::CoverageLedger(std::move(plan));
  rocjitsu::consan::PatchInfo prologue_patch;
  prologue_patch.phase = rocjitsu::consan::PatchPhase::Instrumentation;
  prologue_patch.kind = rocjitsu::consan::PatchKind::KernelEntryOwnerEpochPrologue;
  result.patches.push_back(prologue_patch);
  return result;
}

TEST(HsaHooksUnitTest, ConSanLoadRejectsArchitectureDependentResultWithoutResolvedArch) {
  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", nullptr);

  testing::internal::CaptureStderr();
  run_hook_load_case(kConSanHookProfiles[1], false,
                     fault_test_result_for_arch(ROCJITSU_CODE_ARCH_INVALID),
                     HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u);
  const std::string direct_error = testing::internal::GetCapturedStderr();
  EXPECT_NE(direct_error.find("ConSan internal invariant violation"), std::string::npos);
  EXPECT_NE(direct_error.find("reason=internal-semantic-arch-missing"), std::string::npos);

  testing::internal::CaptureStderr();
  run_hook_load_case(kConSanHookProfiles[1], false,
                     fault_test_result_for_arch(ROCJITSU_CODE_ARCH_INVALID),
                     HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u,
                     /*expected_replacement=*/{},
                     /*fail_replacement_reader_create=*/false,
                     /*use_auto_report=*/true);
  const std::string inventory_error = testing::internal::GetCapturedStderr();
  EXPECT_NE(inventory_error.find("reason=internal-semantic-arch-missing"), std::string::npos);

  rocjitsu::consan::TransformArtifacts empty;
  empty.outcome = rocjitsu::consan::TransformOutcome::Unchanged;
  run_hook_load_case(kConSanHookProfiles[1], false, empty, HSA_STATUS_SUCCESS, 101u);
}

TEST(HsaHooksUnitTest, ConSanStrictRejectionFlushesDiscardedFaultInstallationEvidence) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "strict");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", nullptr);
  ScopedEnvVar require_patch("RJ_CONSAN_REQUIRE_PATCH", nullptr);
  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", nullptr);
  ScopedEnvVar drop_barrier("RJ_CONSAN_FAULT_DROP_BARRIER", "1");
  ScopedEnvVar require_exactly_one("RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE", "1");
  g_transform_override_result = fault_test_result_for_arch(ROCJITSU_CODE_ARCH_RDNA4);
  install_test_access_coverage(g_transform_override_result, 1u,
                               rocjitsu::consan::SiteDecisionKind::Admitted,
                               rocjitsu::consan::AccessPolicyReason::None);
  g_transform_override_models_fault_application = true;

  ASSERT_EXIT(([] {
                FakeApiTable api;
                InstalledDbiHook hook(api);
                if (!hook.installed())
                  std::_Exit(1);
                constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
                hsa_code_object_reader_t reader{};
                if (api.core.hsa_code_object_reader_create_from_memory_fn(
                        original.data(), original.size(), &reader) != HSA_STATUS_SUCCESS)
                  std::_Exit(2);
                (void)api.core.hsa_executable_load_agent_code_object_fn(
                    hsa_executable_t{7}, kHostAgent, reader, nullptr, nullptr);
                std::_Exit(3);
              }()),
              testing::ExitedWithCode(92), "ConSan fault install.*applied=1 installed=false");
}

TEST(HsaHooksUnitTest, ConSanFaultReservationIsReleasedAfterRejectedTransform) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar drop_barrier("RJ_CONSAN_FAULT_DROP_BARRIER", "1");
  ScopedEnvVar require_exactly_one("RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE", "1");
  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", nullptr);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");

  g_transform_override_result = fault_test_result_for_arch(ROCJITSU_CODE_ARCH_INVALID);
  g_transform_override_models_fault_application = true;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  testing::internal::CaptureStderr();
  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  for (size_t load = 0; load < 2; ++load) {
    SCOPED_TRACE(load);
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  }

  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_EQ(g_transform_override_fault_drop_barriers, (std::vector<bool>{true, true, true, true}));
  EXPECT_NE(log.find("applied=1 installed=false"), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, ConSanFaultReservationIsReleasedWhenReplacementIsNotInstalled) {
  reset_code_object_observations();
  ScopedEnvVar drop_barrier("RJ_CONSAN_FAULT_DROP_BARRIER", "1");
  ScopedEnvVar require_exactly_one("RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE", "1");
  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", nullptr);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  configure_consan_profile(kConSanHookProfiles[1], false);

  g_transform_override_result = fault_test_result_for_arch(ROCJITSU_CODE_ARCH_CDNA3);
  g_transform_override_models_fault_application = true;
  g_fail_replacement_reader_create = true;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();
  testing::internal::CaptureStderr();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t first_reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &first_reader),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              first_reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);

  g_fail_replacement_reader_create = false;
  hsa_code_object_reader_t second_reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &second_reader),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              second_reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);

  EXPECT_EQ(g_transform_override_fault_drop_barriers, (std::vector<bool>{true, true, true, true}));
  EXPECT_EQ(g_loaded_code_object_readers, (std::vector<uint64_t>{101u, 104u}));
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("applied=1 installed=false"), std::string::npos) << log;
  EXPECT_NE(log.find("applied=1 installed=true"), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, ConSanFaultReservationIsRetainedAfterReplacementLoads) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar drop_barrier("RJ_CONSAN_FAULT_DROP_BARRIER", "1");
  ScopedEnvVar require_exactly_one("RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE", "1");
  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", nullptr);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "0");
  g_transform_override_result = fault_test_result_for_arch(ROCJITSU_CODE_ARCH_CDNA3);
  g_transform_override_models_fault_application = true;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  testing::internal::CaptureStderr();
  for (size_t load = 0; load < 2; ++load) {
    SCOPED_TRACE(load);
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
  }

  EXPECT_EQ(g_transform_override_fault_drop_barriers, (std::vector<bool>{true, true, true, false}));
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  hook.unload();
  hook.invoke_unload_again_for_test();
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("reservation=reserved"), std::string::npos) << log;
  EXPECT_NE(log.find("reservation=mutation-already-installed"), std::string::npos) << log;
  EXPECT_NE(log.find("ConSan fault install process="), std::string::npos) << log;
  EXPECT_NE(log.find("ConSan fault reservation summary process="), std::string::npos) << log;
  EXPECT_EQ(log.find("ConSan fault reservation summary process=",
                     log.find("ConSan fault reservation summary process=") + 1),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("attempts=2 reserved=1 mutation_already_installed=1 "
                     "contention_timeout=0 reentrant_contention=0 mutation_installed=true "
                     "active=false complete=true"),
            std::string::npos)
      << log;
}

TEST(HsaHooksUnitTest, ConSanFaultReservationRetriesAfterConcurrentRejectedOwner) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar drop_barrier("RJ_CONSAN_FAULT_DROP_BARRIER", "1");
  ScopedEnvVar require_exactly_one("RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE", "1");
  ScopedEnvVar require_patch("RJ_CONSAN_REQUIRE_PATCH", "1");
  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", nullptr);
  ScopedEnvVar reservation_timeout("RJ_CONSAN_FAULT_RESERVATION_TIMEOUT_MS", "30000");

  g_transform_override_result = fault_test_result_for_arch(ROCJITSU_CODE_ARCH_RDNA4);
  rocjitsu::consan::PatchInfo site_patch;
  site_patch.phase = rocjitsu::consan::PatchPhase::Instrumentation;
  site_patch.kind = rocjitsu::consan::PatchKind::TrampolineWatchpointStore;
  g_transform_override_result.patches.push_back(site_patch);
  g_transform_override_models_fault_application = true;
  g_block_first_fault_application = true;
  g_first_fault_application_result = fault_test_result_for_arch(ROCJITSU_CODE_ARCH_RDNA4);
  install_test_access_coverage(*g_first_fault_application_result, 1u,
                               rocjitsu::consan::SiteDecisionKind::Admitted,
                               rocjitsu::consan::AccessPolicyReason::None);

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> first = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  constexpr std::array<uint8_t, 8> second = {0x7f, 'E', 'L', 'F', 5, 6, 7, 8};
  hsa_code_object_reader_t first_reader{};
  hsa_code_object_reader_t second_reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(first.data(), first.size(),
                                                                  &first_reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(second.data(), second.size(),
                                                                  &second_reader),
            HSA_STATUS_SUCCESS);

  auto load = [&](hsa_code_object_reader_t reader) {
    return api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                             reader, nullptr, nullptr);
  };
  std::future<hsa_status_t> first_load = std::async(std::launch::async, load, first_reader);
  {
    std::unique_lock lock(g_fault_application_block_mutex);
    if (!g_fault_application_block_cv.wait_for(lock, std::chrono::seconds(2),
                                               [] { return g_first_fault_application_entered; })) {
      g_release_first_fault_application = true;
      lock.unlock();
      g_fault_application_block_cv.notify_all();
      (void)first_load.get();
      FAIL() << "first fault application did not reach the blocked transform";
      return;
    }
  }

  std::future<hsa_status_t> second_load = std::async(std::launch::async, load, second_reader);
  EXPECT_EQ(second_load.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
  {
    std::lock_guard lock(g_fault_application_block_mutex);
    g_release_first_fault_application = true;
  }
  g_fault_application_block_cv.notify_all();

  EXPECT_EQ(first_load.get(), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_EQ(second_load.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{103u});
  EXPECT_TRUE(std::ranges::all_of(g_transform_override_fault_drop_barriers,
                                  [](bool enabled) { return enabled; }));
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, ConSanFaultReservationTimesOutWithoutApplyingASecondMutation) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar drop_barrier("RJ_CONSAN_FAULT_DROP_BARRIER", "1");
  ScopedEnvVar require_exactly_one("RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE", "1");
  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", nullptr);
  ScopedEnvVar reservation_timeout("RJ_CONSAN_FAULT_RESERVATION_TIMEOUT_MS", "20");
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");

  g_transform_override_result = fault_test_result_for_arch(ROCJITSU_CODE_ARCH_CDNA3);
  g_transform_override_models_fault_application = true;
  g_block_first_fault_application = true;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> first = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  constexpr std::array<uint8_t, 8> second = {0x7f, 'E', 'L', 'F', 5, 6, 7, 8};
  hsa_code_object_reader_t first_reader{};
  hsa_code_object_reader_t second_reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(first.data(), first.size(),
                                                                  &first_reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(second.data(), second.size(),
                                                                  &second_reader),
            HSA_STATUS_SUCCESS);

  auto load = [&](hsa_code_object_reader_t reader) {
    return api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                             reader, nullptr, nullptr);
  };
  testing::internal::CaptureStderr();
  std::future<hsa_status_t> first_load = std::async(std::launch::async, load, first_reader);
  {
    std::unique_lock lock(g_fault_application_block_mutex);
    ASSERT_TRUE(g_fault_application_block_cv.wait_for(
        lock, std::chrono::seconds(2), [] { return g_first_fault_application_entered; }));
  }

  std::future<hsa_status_t> second_load = std::async(std::launch::async, load, second_reader);
  EXPECT_EQ(second_load.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_EQ(second_load.get(), HSA_STATUS_SUCCESS);
  {
    std::lock_guard lock(g_fault_application_block_mutex);
    g_release_first_fault_application = true;
  }
  g_fault_application_block_cv.notify_all();
  EXPECT_EQ(first_load.get(), HSA_STATUS_SUCCESS);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_NE(log.find("outcome=contention-timeout"), std::string::npos) << log;
  EXPECT_NE(std::ranges::find(g_transform_override_fault_drop_barriers, false),
            g_transform_override_fault_drop_barriers.end());
  EXPECT_EQ(std::ranges::count(g_transform_override_fault_drop_barriers, false), 1);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, ConSanFaultReservationTimesOutWhileOwnerIsInLoader) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar drop_barrier("RJ_CONSAN_FAULT_DROP_BARRIER", "1");
  ScopedEnvVar require_exactly_one("RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE", "1");
  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", nullptr);
  ScopedEnvVar reservation_timeout("RJ_CONSAN_FAULT_RESERVATION_TIMEOUT_MS", "20");
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");

  g_transform_override_result = fault_test_result_for_arch(ROCJITSU_CODE_ARCH_CDNA3);
  g_transform_override_models_fault_application = true;
  g_block_first_loader_call = true;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> first = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  constexpr std::array<uint8_t, 8> second = {0x7f, 'E', 'L', 'F', 5, 6, 7, 8};
  hsa_code_object_reader_t first_reader{};
  hsa_code_object_reader_t second_reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(first.data(), first.size(),
                                                                  &first_reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(second.data(), second.size(),
                                                                  &second_reader),
            HSA_STATUS_SUCCESS);

  auto load = [&](hsa_code_object_reader_t reader) {
    return api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                             reader, nullptr, nullptr);
  };
  testing::internal::CaptureStderr();
  std::future<hsa_status_t> first_load = std::async(std::launch::async, load, first_reader);
  {
    std::unique_lock lock(g_loader_block_mutex);
    if (!g_loader_block_cv.wait_for(lock, std::chrono::seconds(2),
                                    [] { return g_first_loader_call_entered; })) {
      g_release_first_loader_call = true;
      lock.unlock();
      g_loader_block_cv.notify_all();
      (void)first_load.get();
      FAIL() << "first fault application did not reach the blocked loader";
      return;
    }
  }

  std::future<hsa_status_t> second_load = std::async(std::launch::async, load, second_reader);
  EXPECT_EQ(second_load.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_EQ(second_load.get(), HSA_STATUS_SUCCESS);
  {
    std::lock_guard lock(g_loader_block_mutex);
    g_release_first_loader_call = true;
  }
  g_loader_block_cv.notify_all();
  EXPECT_EQ(first_load.get(), HSA_STATUS_SUCCESS);
  EXPECT_NE(std::ranges::find(g_transform_override_fault_drop_barriers, false),
            g_transform_override_fault_drop_barriers.end());
  EXPECT_EQ(std::ranges::count(g_transform_override_fault_drop_barriers, false), 1);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  hook.unload();
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_NE(log.find("outcome=contention-timeout"), std::string::npos) << log;
  EXPECT_NE(log.find("attempts=2 reserved=1 mutation_already_installed=0 "
                     "contention_timeout=1 reentrant_contention=0 mutation_installed=true "
                     "active=false complete=true"),
            std::string::npos)
      << log;
}

TEST(HsaHooksUnitTest, ConSanFaultReservationRejectsSameThreadReentry) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar drop_barrier("RJ_CONSAN_FAULT_DROP_BARRIER", "1");
  ScopedEnvVar require_exactly_one("RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE", "1");
  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", nullptr);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");

  g_transform_override_result = fault_test_result_for_arch(ROCJITSU_CODE_ARCH_CDNA3);
  g_transform_override_models_fault_application = true;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> first = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  constexpr std::array<uint8_t, 8> second = {0x7f, 'E', 'L', 'F', 5, 6, 7, 8};
  hsa_code_object_reader_t first_reader{};
  hsa_code_object_reader_t second_reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(first.data(), first.size(),
                                                                  &first_reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(second.data(), second.size(),
                                                                  &second_reader),
            HSA_STATUS_SUCCESS);
  g_reentrant_fault_load = [&] {
    return api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                             second_reader, nullptr, nullptr);
  };

  testing::internal::CaptureStderr();
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              first_reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_TRUE(g_reentrant_fault_load_status.has_value());
  EXPECT_EQ(*g_reentrant_fault_load_status, HSA_STATUS_SUCCESS);
  EXPECT_NE(std::ranges::find(g_transform_override_fault_drop_barriers, false),
            g_transform_override_fault_drop_barriers.end());
  EXPECT_EQ(std::ranges::count(g_transform_override_fault_drop_barriers, false), 1);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  hook.unload();
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_NE(log.find("outcome=reentrant-contention"), std::string::npos) << log;
  EXPECT_NE(log.find("attempts=2 reserved=1 mutation_already_installed=0 "
                     "contention_timeout=0 reentrant_contention=1 mutation_installed=true "
                     "active=false complete=true"),
            std::string::npos)
      << log;
}

TEST(HsaHooksUnitTest, ConSanExactOneRejectsMultipleAppliedMutationsBeforeInstallation) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  ScopedEnvVar drop_barrier("RJ_CONSAN_FAULT_DROP_BARRIER", "1");
  ScopedEnvVar require_exactly_one("RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE", "1");
  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", nullptr);

  g_transform_override_result = fault_test_result_for_arch(ROCJITSU_CODE_ARCH_CDNA3);
  g_transform_override_models_fault_application = true;
  g_transform_override_actual_fault_applications = 2;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();
  testing::internal::CaptureStderr();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t first_reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &first_reader),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              first_reader, nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_TRUE(g_loaded_code_object_readers.empty());

  g_transform_override_actual_fault_applications = 1;
  hsa_code_object_reader_t second_reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &second_reader),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              second_reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(g_loaded_code_object_readers.size(), 1u);
  EXPECT_EQ(g_loaded_code_object_readers.front(), 103u);

  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("reason=multiple-fault-mutations-applied"), std::string::npos) << log;
  EXPECT_NE(log.find("applied=2 installed=false"), std::string::npos) << log;
  EXPECT_NE(log.find("applied=1 installed=true"), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, ConSanSynchronizationDefaultsRemainExplicitlyOverridable) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar track_barriers("RJ_CONSAN_TRACK_BARRIERS", "0");
  ScopedEnvVar track_atomics("RJ_CONSAN_TRACK_ATOMICS", "0");
  ScopedEnvVar abort_unmatched("RJ_CONSAN_ABORT_UNMATCHED_BARRIER_WAIT", "1");

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();
  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(g_transform_override_track_barriers.size(), 1u);
  ASSERT_EQ(g_transform_override_track_atomics.size(), 1u);
  ASSERT_EQ(g_transform_override_abort_unmatched_waits.size(), 1u);
  EXPECT_FALSE(g_transform_override_track_barriers.front());
  EXPECT_FALSE(g_transform_override_track_atomics.front());
  EXPECT_TRUE(g_transform_override_abort_unmatched_waits.front());
}

rocjitsu::consan::TransformArtifacts diagnostic_coverage_transform_result() {
  rocjitsu::consan::TransformArtifacts result;
  install_consan_test_program_identity(result, ROCJITSU_CODE_ARCH_RDNA4,
                                       ROCJITSU_CODE_TARGET_GFX1201);
  result.outcome = rocjitsu::consan::TransformOutcome::ModifiedValid;
  result.replacement = {0x7f, 'E', 'L', 'F', 'p', 'a', 't', 'c', 'h'};

  const auto semantic_site = [&](uint64_t offset, rocjitsu::consan::SemanticSiteDomain domain) {
    return SemanticSiteId{
        .physical = {.code_object = result.program_inventory.code_object_id(),
                     .original_text_offset = offset},
        .domain = domain,
        .member_ordinal = 0u,
        .range_ordinal = 0u,
    };
  };
  const SemanticSiteId access = semantic_site(0x10u, rocjitsu::consan::SemanticSiteDomain::Access);
  const SemanticSiteId barrier =
      semantic_site(0x20u, rocjitsu::consan::SemanticSiteDomain::SynchronizationEvent);
  const SemanticSiteId atomic =
      semantic_site(0x30u, rocjitsu::consan::SemanticSiteDomain::SynchronizationEvent);
  const SemanticSiteId fence =
      semantic_site(0x40u, rocjitsu::consan::SemanticSiteDomain::SynchronizationEvent);
  const rocjitsu::consan::SynchronizationAssociationId atomic_association{"test-atomic"};
  install_consan_test_program_inventory(result, [&](ProgramInventoryBuilder &builder) {
    const auto add_source = [&](uint64_t offset, std::string name) {
      rocjitsu::consan::ProgramContainer container{rocjitsu::consan::ProgramContainerKind::Kernel};
      container.name = std::move(name);
      const rocjitsu::consan::ProgramContainerId container_id =
          builder.add_kernel(std::move(container)).id;
      rocjitsu::consan::ProgramSite site;
      site.physical_id.original_text_offset = offset;
      site.container = container_id;
      builder.add_semantic_site(std::move(site));
    };
    add_source(0x10u, "unsupported_kernel");
    add_source(0x20u, "barrier_helper");
    add_source(0x30u, "atomic_kernel");
    add_source(0x40u, "fence_kernel");
  });
  rocjitsu::consan::ObservationPlan plan =
      {
          .mode = rocjitsu::consan::Mode::Default,
          .site_decisions = {{
              .semantic_site = access,
              .kind = rocjitsu::consan::SiteDecisionKind::Unsupported,
              .reason = rocjitsu::consan::AccessPolicyReason::UnsupportedMnemonic,
          }},
          .barrier_site_decisions = {{
              .semantic_site = barrier,
              .kind = rocjitsu::consan::SiteDecisionKind::Admitted,
              .reason = rocjitsu::consan::BarrierPolicyReason::None,
          }},
          .atomic_site_decisions = {{
              .semantic_site = atomic,
              .kind = rocjitsu::consan::SiteDecisionKind::Admitted,
              .capability = rocjitsu::consan::CapabilityDisposition::Supported,
              .reason = rocjitsu::consan::AtomicPolicyReason::None,
          }},
          .fence_site_decisions = {{
              .semantic_site = fence,
              .kind = rocjitsu::consan::SiteDecisionKind::Admitted,
              .capability = rocjitsu::consan::CapabilityDisposition::Supported,
              .reason = rocjitsu::consan::FencePolicyReason::None,
              .inventory_association = rocjitsu::consan::FenceAssociation::Qualified,
          }},
          .probe_intents =
              {
                  {.id = {0u},
                   .mode = rocjitsu::consan::Mode::Default,
                   .source_site = {0u},
                   .physical_site = barrier.physical,
                   .covered_semantic_sites = {barrier},
                   .kind = rocjitsu::consan::ProbeIntentKind::BarrierEpoch,
                   .position = rocjitsu::consan::ProbePosition::After,
                   .synchronization_association = std::nullopt,
                   .dynamic_result = rocjitsu::consan::DynamicResultRequirement::None,
                   .atomic_lowering_form = std::nullopt},
                  {.id = {1u},
                   .mode = rocjitsu::consan::Mode::Default,
                   .source_site = {1u},
                   .physical_site = atomic.physical,
                   .covered_semantic_sites = {atomic, fence},
                   .kind = rocjitsu::consan::ProbeIntentKind::AtomicOrdering,
                   .position = rocjitsu::consan::ProbePosition::After,
                   .synchronization_association = atomic_association,
                   .dynamic_result = rocjitsu::consan::DynamicResultRequirement::None,
                   .atomic_lowering_form = std::nullopt},

              },
      };
  EXPECT_TRUE(plan.valid());
  result.coverage_ledger = rocjitsu::consan::CoverageLedger(std::move(plan));
  EXPECT_TRUE(publish_test_lowering_outcome(
      result.coverage_ledger, {0u}, rocjitsu::consan::LoweringOutcomeKind::ResourceRejected));
  EXPECT_TRUE(publish_test_lowering_outcome(
      result.coverage_ledger, {1u}, rocjitsu::consan::LoweringOutcomeKind::PlacementRejected));
  return result;
}

rocjitsu::consan::TransformArtifacts typed_coverage_transform_result() {
  rocjitsu::consan::TransformArtifacts result;
  install_consan_test_program_identity(result, ROCJITSU_CODE_ARCH_RDNA4,
                                       ROCJITSU_CODE_TARGET_GFX1201);
  result.outcome = rocjitsu::consan::TransformOutcome::ModifiedValid;
  result.replacement = {0x7f, 'E', 'L', 'F', 'p', 'a', 't', 'c', 'h'};

  const PhysicalSiteId physical{
      .code_object = result.program_inventory.code_object_id(),
      .original_text_offset = 0x10u,
  };
  const SemanticSiteId semantic{
      .physical = physical,
      .domain = rocjitsu::consan::SemanticSiteDomain::Access,
      .member_ordinal = 0u,
      .range_ordinal = 0u,
  };
  install_consan_test_program_inventory(result, [&](ProgramInventoryBuilder &builder) {
    rocjitsu::consan::ProgramContainer container{rocjitsu::consan::ProgramContainerKind::Kernel};
    container.name = "typed_kernel";
    const rocjitsu::consan::ProgramContainerId container_id =
        builder.add_kernel(std::move(container)).id;
    rocjitsu::consan::ProgramSite site;
    site.physical_id.original_text_offset = 0x10u;
    site.container = container_id;
    builder.add_semantic_site(std::move(site));
  });
  rocjitsu::consan::ObservationPlan plan = {
      .mode = rocjitsu::consan::Mode::Default,
      .site_decisions = {{
          .semantic_site = semantic,
          .kind = rocjitsu::consan::SiteDecisionKind::Admitted,
          .reason = rocjitsu::consan::AccessPolicyReason::None,
      }},
      .barrier_site_decisions = {},
      .atomic_site_decisions = {},
      .fence_site_decisions = {},
      .probe_intents = {{
          .id = {0u},
          .mode = rocjitsu::consan::Mode::Default,
          .source_site = {0u},
          .physical_site = physical,
          .covered_semantic_sites = {semantic},
          .kind = rocjitsu::consan::ProbeIntentKind::Access,
          .position = rocjitsu::consan::ProbePosition::Before,
          .synchronization_association = std::nullopt,
          .dynamic_result = rocjitsu::consan::DynamicResultRequirement::None,
          .atomic_lowering_form = std::nullopt,
      }},
  };
  result.coverage_ledger = rocjitsu::consan::CoverageLedger(std::move(plan));
  EXPECT_TRUE(publish_test_lowering_outcome(result.coverage_ledger, {0u},
                                            rocjitsu::consan::LoweringOutcomeKind::Instrumented));
  return result;
}

TEST(HsaHooksUnitTest, ConSanCoverageUsesTypedPipelineLedgerAfterPublishingMechanismResult) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  const ConSanHookProfile &profile = kConSanHookProfiles[1];
  const rocjitsu::consan::TransformArtifacts result = typed_coverage_transform_result();

  testing::internal::CaptureStderr();
  run_hook_load_case(profile, false, result, HSA_STATUS_SUCCESS, 102u, result.replacement);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_NE(log.find("ConSan coverage reader=101 mode=default "
                     "analysis_complete=true"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("access_discovered=1 access_supported=1 access_selected=1 "
                     "access_patched=1"),
            std::string::npos)
      << log;
}

TEST(HsaHooksUnitTest, ConSanCoverageSiteDiagnosticsRetainStableReasonsAndSourceLocations) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "3");
  const ConSanHookProfile &profile = kConSanHookProfiles[1];
  const rocjitsu::consan::TransformArtifacts result = diagnostic_coverage_transform_result();

  testing::internal::CaptureStderr();
  run_hook_load_case(profile, false, result, HSA_STATUS_SUCCESS, 102u, result.replacement);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_NE(log.find("ConSan coverage_site reader=101 kind=access disposition=unsupported "
                     "reason=unsupported_mnemonic outcome=unsupported "
                     "lowering_reason=semantic_unsupported resource_reason=none "
                     "container=unsupported_kernel scope=kernel text=0x10 mnemonic=unknown"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("ConSan coverage_site reader=101 kind=barrier disposition=supported "
                     "reason=none outcome=resource_failed "
                     "lowering_reason=unsupported_resource_plan resource_reason=invalid_request "
                     "container=barrier_helper scope=kernel text=0x20 mnemonic=unknown"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("ConSan coverage_site reader=101 kind=atomic disposition=supported "
                     "reason=none outcome=placement_or_lowering_failed "
                     "lowering_reason=instrumentation_patch_missing resource_reason=none "
                     "container=atomic_kernel scope=kernel text=0x30 mnemonic=unknown"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("ConSan coverage_site reader=101 kind=fence disposition=supported "
                     "reason=none outcome=placement_or_lowering_failed "
                     "lowering_reason=instrumentation_patch_missing resource_reason=none "
                     "container=fence_kernel scope=kernel text=0x40 mnemonic=unknown"),
            std::string::npos)
      << log;
}

TEST(HsaHooksUnitTest, ConSanHighCardinalityTypedDiagnosticsUseBoundedBatchedWrites) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "3");
  const ConSanHookProfile &profile = kConSanHookProfiles[1];
  rocjitsu::consan::TransformArtifacts result = diagnostic_coverage_transform_result();
  constexpr size_t kRecordCount = 4096;

  install_test_access_coverage(result, kRecordCount,
                               rocjitsu::consan::SiteDecisionKind::Unsupported,
                               rocjitsu::consan::AccessPolicyReason::UnsupportedMnemonic);
  result.outcome = rocjitsu::consan::TransformOutcome::Unchanged;
  result.replacement.clear();
  const auto count_records = [](std::string_view log, std::string_view marker) {
    size_t count = 0;
    for (size_t offset = 0; (offset = log.find(marker, offset)) != std::string_view::npos;
         offset += marker.size()) {
      ++count;
    }
    return count;
  };

  g_log_sink_write_count = 0;
  g_log_sink_max_write_size = 0;
  g_log_sink_writes_end_in_newline = true;
  g_log_sink_bytes.clear();
  run_hook_load_case(profile, false, result, HSA_STATUS_SUCCESS, 101u, {}, false, false,
                     capture_log_sink);

  EXPECT_EQ(count_records(g_log_sink_bytes, "ConSan coverage_site "), kRecordCount);
  EXPECT_TRUE(g_log_sink_writes_end_in_newline);
  EXPECT_LE(g_log_sink_max_write_size, 1024u * 1024u);
  // This is a structural regression check for the reported per-record write
  // path, not a machine-dependent stopwatch assertion. The old implementation
  // performed more than two writes per record; bounded batching needs only a
  // handful of writes per MiB plus low-cardinality setup diagnostics.
  EXPECT_LT(g_log_sink_write_count, kRecordCount / 16u);
}

TEST(HsaHooksUnitTest, ConSanResourcePlanFallbackTelemetryIsVisibleAtQualificationLogLevel) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  const ConSanHookProfile &profile = kConSanHookProfiles[1];
  rocjitsu::consan::TransformArtifacts result;
  install_consan_test_program_identity(result, ROCJITSU_CODE_ARCH_RDNA4,
                                       ROCJITSU_CODE_TARGET_GFX1201);
  result.outcome = rocjitsu::consan::TransformOutcome::ModifiedValid;
  result.replacement = {0x7f, 'E', 'L', 'F', 'p', 'a', 't', 'c', 'h'};

  rocjitsu::consan::CandidateResourcePlan access_plan;
  access_plan.site_kind = rocjitsu::consan::ResourceSiteKind::Access;
  access_plan.candidate_index = 0;
  access_plan.text_offset = 0x20;
  access_plan.source = rocjitsu::consan::RegisterAllocationSource::SpillRequired;
  access_plan.alternatives = {
      {.kind = rocjitsu::consan::ResourcePlanAlternativeKind::GuestOperandOverlapSpill,
       .source = rocjitsu::consan::RegisterAllocationSource::SpillRequired,
       .scratch_vgpr_count = 17,
       .outcome = rocjitsu::consan::ResourcePlanAlternativeOutcome::Superseded},
      {.kind = rocjitsu::consan::ResourcePlanAlternativeKind::SpillBackedOperandRecovery,
       .source = rocjitsu::consan::RegisterAllocationSource::SpillRequired,
       .scratch_vgpr_count = 16,
       .outcome = rocjitsu::consan::ResourcePlanAlternativeOutcome::Selected},
      {.kind = rocjitsu::consan::ResourcePlanAlternativeKind::GuestOperandOverlapSpill,
       .source = rocjitsu::consan::RegisterAllocationSource::SpillRequired,
       .scratch_vgpr_count = 16,
       .outcome = rocjitsu::consan::ResourcePlanAlternativeOutcome::Contributed},
  };
  result.resource_plans.push_back(std::move(access_plan));

  rocjitsu::consan::CandidateResourcePlan atomic_plan;
  atomic_plan.site_kind = rocjitsu::consan::ResourceSiteKind::Atomic;
  atomic_plan.candidate_index = 1;
  atomic_plan.text_offset = 0x40;
  atomic_plan.source = rocjitsu::consan::RegisterAllocationSource::Unsupported;
  atomic_plan.reason = rocjitsu::consan::RegisterPlanReason::ForbiddenOverlap;
  atomic_plan.scratch_vgpr_count = 10;
  atomic_plan.current_vgpr_count = 64;
  atomic_plan.max_referenced_vgpr_count = 61;
  atomic_plan.ordinary_vgpr_limit = 256;
  atomic_plan.required_vgpr_count = 64;
  atomic_plan.owner_kernel_ids = {{0}};
  atomic_plan.alternatives = {
      {.kind = rocjitsu::consan::ResourcePlanAlternativeKind::GuestOperandOverlapSpill,
       .source = rocjitsu::consan::RegisterAllocationSource::SpillRequired,
       .scratch_vgpr_count = 10,
       .outcome = rocjitsu::consan::ResourcePlanAlternativeOutcome::Selected},
  };
  result.resource_plans.push_back(std::move(atomic_plan));

  rocjitsu::consan::CandidateResourcePlan fence_plan;
  fence_plan.site_kind = rocjitsu::consan::ResourceSiteKind::Fence;
  fence_plan.candidate_index = 2;
  fence_plan.text_offset = 0x60;
  fence_plan.source = rocjitsu::consan::RegisterAllocationSource::Unsupported;
  fence_plan.reason = rocjitsu::consan::RegisterPlanReason::NoLegalWindow;
  fence_plan.scratch_vgpr_count = 8;
  fence_plan.current_vgpr_count = 256;
  fence_plan.max_referenced_vgpr_count = 256;
  fence_plan.ordinary_vgpr_limit = 256;
  fence_plan.required_vgpr_count = 256;
  fence_plan.owner_kernel_ids = {{1}, {2}};
  fence_plan.has_indirect_vgpr_access = true;
  fence_plan.alternatives = {
      {.kind = rocjitsu::consan::ResourcePlanAlternativeKind::GuestOperandOverlapSpill,
       .source = rocjitsu::consan::RegisterAllocationSource::Unsupported,
       .reason = rocjitsu::consan::RegisterPlanReason::NoLegalWindow,
       .scratch_vgpr_count = 8,
       .outcome = rocjitsu::consan::ResourcePlanAlternativeOutcome::Rejected},
  };
  result.resource_plans.push_back(std::move(fence_plan));

  testing::internal::CaptureStderr();
  run_hook_load_case(profile, false, result, HSA_STATUS_SUCCESS, 102u, result.replacement);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_NE(log.find("alternative_attempts=5 alternative_selected=1 "
                     "alternative_rejected=1 alternative_superseded=1 "
                     "alternative_contributed=1 alternative_vetoed=1"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("resource-failure reader=101 site=atomic reason=forbidden_overlap count=1 "
                     "scratch_vgprs=10..10 current_vgprs=64..64 "
                     "max_referenced_vgprs=61..61 ordinary_vgpr_limit=256..256 "
                     "required_vgprs=64..64 owners=1..1 indirect_vgprs=false"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("resource-failure reader=101 site=fence reason=no_legal_window count=1 "
                     "scratch_vgprs=8..8 current_vgprs=256..256 "
                     "max_referenced_vgprs=256..256 ordinary_vgpr_limit=256..256 "
                     "required_vgprs=256..256 owners=2..2 indirect_vgprs=true"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("attempt=0 kind=guest_operand_overlap_spill scratch_count=17 "
                     "source=spill reason=none outcome=superseded"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("attempt=1 kind=spill_backed_operand_recovery scratch_count=16 "
                     "source=spill reason=none outcome=selected"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("attempt=2 kind=guest_operand_overlap_spill scratch_count=16 "
                     "source=spill reason=none outcome=contributed"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("site=atomic candidate=1 text_offset=0x40 attempt=0 "
                     "kind=guest_operand_overlap_spill scratch_count=10 "
                     "source=spill reason=none outcome=vetoed"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("site=fence candidate=2 text_offset=0x60 attempt=0 "
                     "kind=guest_operand_overlap_spill scratch_count=8 "
                     "source=unsupported reason=no_legal_window outcome=rejected"),
            std::string::npos)
      << log;
}

TEST(HsaHooksUnitTest, ConSanCoverageDoesNotResurrectNotApplicableResourcePlan) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  const ConSanHookProfile &profile = kConSanHookProfiles[1];
  rocjitsu::consan::TransformArtifacts result;
  install_consan_test_program_identity(result, ROCJITSU_CODE_ARCH_RDNA4,
                                       ROCJITSU_CODE_TARGET_GFX1201);
  result.outcome = rocjitsu::consan::TransformOutcome::ModifiedValid;
  result.replacement = {0x7f, 'E', 'L', 'F', 'p', 'a', 't', 'c', 'h'};
  const SemanticSiteId atomic_site{
      .physical = {.code_object = result.program_inventory.code_object_id(),
                   .original_text_offset = 0x30u},
      .domain = rocjitsu::consan::SemanticSiteDomain::SynchronizationEvent,
      .member_ordinal = 0u,
      .range_ordinal = 0u,
  };
  rocjitsu::consan::ObservationPlan plan = {
      .mode = rocjitsu::consan::Mode::Default,
      .site_decisions = {},
      .barrier_site_decisions = {},
      .atomic_site_decisions = {{
          .semantic_site = atomic_site,
          .kind = rocjitsu::consan::SiteDecisionKind::NotApplicable,
          .capability = rocjitsu::consan::CapabilityDisposition::OutOfContract,
          .reason = rocjitsu::consan::AtomicPolicyReason::UnqualifiedSyncSequence,
      }},
      .fence_site_decisions = {},
      .probe_intents = {},
  };
  ASSERT_TRUE(plan.valid());
  result.coverage_ledger = rocjitsu::consan::CoverageLedger(std::move(plan));
  rocjitsu::consan::CandidateResourcePlan atomic_plan;
  atomic_plan.site_kind = rocjitsu::consan::ResourceSiteKind::Atomic;
  atomic_plan.text_offset = 0x30;
  result.resource_plans.push_back(std::move(atomic_plan));
  result.outcome = rocjitsu::consan::TransformOutcome::Unchanged;
  result.replacement.clear();

  testing::internal::CaptureStderr();
  run_hook_load_case(profile, false, result, HSA_STATUS_SUCCESS, 101u);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_NE(log.find("atomic_discovered=0 atomic_supported=0 atomic_selected=0 "
                     "atomic_patched=0 atomic_unsupported=0 atomic_resource_failed=0 "
                     "atomic_placement_or_lowering_failed=0 atomic_expert_limit_omitted=0"),
            std::string::npos)
      << log;
  EXPECT_EQ(log.find("coverage_site reader=101 kind=atomic"), std::string::npos) << log;
}

rocjitsu::consan::TransformArtifacts auto_report_atomic_transform_result() {
  rocjitsu::consan::TransformArtifacts result;
  install_consan_test_program_identity(result, ROCJITSU_CODE_ARCH_RDNA4,
                                       ROCJITSU_CODE_TARGET_GFX1201);
  result.outcome = rocjitsu::consan::TransformOutcome::ModifiedValid;
  result.replacement = {0x7f, 'E', 'L', 'F', 'p', 'a', 't', 'c', 'h'};
  rocjitsu::consan::CandidateResourcePlan atomic_plan;
  atomic_plan.site_kind = rocjitsu::consan::ResourceSiteKind::Atomic;
  atomic_plan.source = rocjitsu::consan::RegisterAllocationSource::LivenessDead;
  result.resource_plans.push_back(atomic_plan);
  install_consan_test_program_inventory(result, [](ProgramInventoryBuilder &builder) {
    auto &kernel = builder.add_kernel();
    kernel.name = "auto_report_atomic";
    kernel.entry_text_offset = 0u;
    kernel.code_size = 4u;
    kernel.has_text_range = true;
    auto site = make_program_site(kernel.id, rocjitsu::consan::AtomicSite{});
    site.execution_owners = {{.kernel = kernel.id}};
    builder.add_semantic_site(std::move(site));
  });
  const PhysicalSiteId physical{
      .code_object = result.program_inventory.code_object_id(),
      .original_text_offset = 0u,
  };
  const SemanticSiteId semantic{
      .physical = physical,
      .domain = rocjitsu::consan::SemanticSiteDomain::SynchronizationEvent,
  };
  rocjitsu::consan::ObservationPlan plan = {
      .mode = rocjitsu::consan::Mode::Default,
      .site_decisions = {},
      .barrier_site_decisions = {},
      .atomic_site_decisions = {},
      .fence_site_decisions = {},
      .probe_intents = {{
          .id = {0u},
          .mode = rocjitsu::consan::Mode::Default,
          .source_site = {0u},
          .physical_site = physical,
          .covered_semantic_sites = {semantic},
          .kind = rocjitsu::consan::ProbeIntentKind::AtomicOrdering,
          .position = rocjitsu::consan::ProbePosition::After,
          .synchronization_association =
              rocjitsu::consan::SynchronizationAssociationId{"auto-report-atomic"},
          .dynamic_result = rocjitsu::consan::DynamicResultRequirement::None,
          .atomic_lowering_form = std::nullopt,
      }},
  };
  EXPECT_TRUE(plan.valid());
  result.coverage_ledger = rocjitsu::consan::CoverageLedger(std::move(plan));
  EXPECT_TRUE(publish_test_lowering_outcome(result.coverage_ledger, {0u},
                                            rocjitsu::consan::LoweringOutcomeKind::Instrumented));
  return result;
}

void install_auto_report_access_coverage(rocjitsu::consan::TransformArtifacts &result,
                                         rocjitsu::consan::Mode mode,
                                         rocjitsu::consan::ProbeIntentKind intent_kind,
                                         std::span<const uint64_t> instruction_offsets) {
  rocjitsu::consan::ObservationPlan plan;
  plan.mode = mode;
  for (uint64_t instruction_offset : instruction_offsets) {
    const PhysicalSiteId physical{
        .code_object = result.program_inventory.code_object_id(),
        .original_text_offset = instruction_offset,
    };
    const SemanticSiteId semantic{
        .physical = physical,
        .domain = rocjitsu::consan::SemanticSiteDomain::Access,
    };
    const rocjitsu::consan::ProbeIntentId intent_id{
        static_cast<uint32_t>(plan.probe_intents.size())};
    plan.site_decisions.push_back({
        .semantic_site = semantic,
        .kind = rocjitsu::consan::SiteDecisionKind::Admitted,
        .reason = rocjitsu::consan::AccessPolicyReason::None,
    });
    plan.probe_intents.push_back({
        .id = intent_id,
        .mode = mode,
        .source_site = {intent_id.value},
        .physical_site = physical,
        .covered_semantic_sites = {semantic},
        .kind = intent_kind,
        .position = rocjitsu::consan::ProbePosition::Before,
        .synchronization_association = std::nullopt,
        .dynamic_result = rocjitsu::consan::DynamicResultRequirement::None,
        .atomic_lowering_form = std::nullopt,
    });
  }
  ASSERT_TRUE(plan.valid());
  result.coverage_ledger = rocjitsu::consan::CoverageLedger(std::move(plan));
}

rocjitsu::consan::StaticAccessAttribution auto_report_static_access_attribution(
    const rocjitsu::consan::TransformArtifacts &result, size_t intent_index,
    std::vector<rocjitsu::consan::ProgramContainerId> owners, bool owner_provenance_complete) {
  const rocjitsu::consan::ProbeIntent &intent =
      result.observation_plan().probe_intents[intent_index];
  return {
      .intent_ids = {intent.id},
      .original_site = intent.physical_site,
      .original_semantic_sites = intent.covered_semantic_sites,
      .execution_owner_kernel_ids = std::move(owners),
      .owner_provenance_complete = owner_provenance_complete,
  };
}

enum class ReportOwnerScope {
  SingleMapping,
  SharedOwnerPair,
  DisjointOwnerPair,
  UnknownOwnerPair,
};

rocjitsu::consan::TransformArtifacts
auto_report_transform_result(bool malformed_mapping = false,
                             ReportOwnerScope owner_scope = ReportOwnerScope::SingleMapping) {
  rocjitsu::consan::TransformArtifacts result = auto_report_atomic_transform_result();
  std::vector<uint64_t> instruction_offsets = {0x120u};
  if (owner_scope != ReportOwnerScope::SingleMapping)
    instruction_offsets.push_back(0x140u);
  install_auto_report_access_coverage(result, rocjitsu::consan::Mode::Default,
                                      rocjitsu::consan::ProbeIntentKind::Access,
                                      instruction_offsets);
  for (size_t index = 0; index < instruction_offsets.size(); ++index) {
    const bool owner_provenance_complete =
        !(index == 1u && owner_scope == ReportOwnerScope::UnknownOwnerPair);
    std::vector<rocjitsu::consan::ProgramContainerId> owners;
    if (owner_provenance_complete) {
      owners.push_back(
          {index == 1u && owner_scope == ReportOwnerScope::DisjointOwnerPair ? 0x200u : 0x100u});
    }
    rocjitsu::consan::StaticAccessMappings runtime_mapping =
        rocjitsu::consan::StaticAccessMappings{{
            .access = auto_report_static_access_attribution(result, index, std::move(owners),
                                                            owner_provenance_complete),
            .first_slot = malformed_mapping && index == 0u ? std::numeric_limits<uint32_t>::max()
                                                           : static_cast<uint32_t>(index),
            .range_count = 1u,
            .bank_count = 1u,
            .emitted_probe_text_offset = 0x440u + index * 0x20u,
            .relocated_guest_text_offset = 0x448u + index * 0x20u,
            .scratch_vgpr = 12u,
        }};
    const rocjitsu::consan::ProbeIntent &intent = result.observation_plan().probe_intents[index];
    const std::array intent_ids = {intent.id};
    const std::array locations = {rocjitsu::consan::CommittedLoweringLocation{
        .original_site = intent.physical_site,
        .emitted_text_offset = 0x440u + index * 0x20u,
        .emitted_size = 4u,
        .relocated_guest_text_offset = 0x448u + index * 0x20u,
    }};
    auto commit = make_committed_lowering(result.observation_plan(), intent_ids, locations,
                                          rocjitsu::consan::LoweringOutcomeKind::Instrumented, {},
                                          std::move(runtime_mapping));
    EXPECT_TRUE(commit.has_value());
    if (commit) {
      EXPECT_TRUE(result.coverage_ledger.publish_lowering_commit(std::move(*commit)));
    }
  }
  return result;
}

TEST(HsaHooksUnitTest, ConSanAutoReportNeverReconstructsMissingEvidenceFromMechanismTelemetry) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "16777216");

  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result();
  g_transform_override_result.coverage_ledger = {};

  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{101u});
    EXPECT_TRUE(g_core_memory_allocations.empty());
    EXPECT_EQ(g_core_memory_allocate_calls, 0);
  }
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("missing or invalid typed evidence requirements"), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, ConSanAutoReportDoesNotAllocateForValidEmptyTypedEvidence) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "16777216");

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result();
  rocjitsu::consan::ObservationPlan valid_empty_plan;
  valid_empty_plan.mode = rocjitsu::consan::Mode::Default;
  ASSERT_TRUE(valid_empty_plan.valid());
  g_transform_override_result.coverage_ledger =
      rocjitsu::consan::CoverageLedger(std::move(valid_empty_plan));

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();
  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_core_memory_allocate_calls, 0);
  EXPECT_TRUE(g_core_memory_allocations.empty());
  EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{102u});
}

TEST(HsaHooksUnitTest, ConSanAutoReportLiveFaultUsesPristineSizingAndLateBoundLiveOptions) {
  constexpr std::array fault_environments = {
      "RJ_CONSAN_FAULT_DROP_BARRIER",
      "RJ_CONSAN_FAULT_MOVE_BARRIER",
      "RJ_CONSAN_FAULT_MUTATE_BARRIER_ID_SCOPE",
      "RJ_CONSAN_FAULT_MUTATE_BARRIER_PARTICIPANTS",
      "RJ_CONSAN_FAULT_ATOMIC_WRONG_ADDRESS",
      "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER",
      "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_SCOPE",
      "RJ_CONSAN_FAULT_LDS_WRONG_ADDRESS",
      "RJ_CONSAN_FAULT_ORDINARY_WRONG_ADDRESS",
      "RJ_CONSAN_FAULT_ORDINARY_WEAKEN_ORDER",
      "RJ_CONSAN_FAULT_ORDINARY_WEAKEN_SCOPE",
  };
  for (const char *fault_environment : fault_environments) {
    SCOPED_TRACE(fault_environment);
    ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
    ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "1");
    ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
    ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
    ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "16777216");

    ScopedEnvVar selected_fault(fault_environment, "1");
    ScopedEnvVar require_exactly_one("RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE", "1");
    ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);
    ScopedEnvVar barrier_sequence("RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY", "sequence");
    ScopedEnvVar barrier_target_id("RJ_CONSAN_FAULT_BARRIER_TARGET_ID", "1");
    ScopedEnvVar participant_count("RJ_CONSAN_FAULT_BARRIER_TARGET_PARTICIPANT_COUNT", "64");
    ScopedEnvVar atomic_address_delta("RJ_CONSAN_FAULT_ATOMIC_VALID_ADDRESS_DELTA", "4");
    ScopedEnvVar lds_address_vgpr("RJ_CONSAN_FAULT_LDS_ADDRESS_VGPR", "6");
    ScopedEnvVar ordinary_address_delta("RJ_CONSAN_FAULT_ORDINARY_VALID_ADDRESS_DELTA", "4");

    reset_code_object_observations();
    reset_core_memory_observations();
    g_transform_override_result = auto_report_transform_result();
    g_transform_override_models_fault_application = true;
    {
      FakeApiTable api;
      InstalledDbiHook hook(api);
      ASSERT_TRUE(hook.installed()) << hook.error();

      constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
      hsa_code_object_reader_t reader{};
      ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                      original.size(), &reader),
                HSA_STATUS_SUCCESS);
      EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                  reader, nullptr, nullptr),
                HSA_STATUS_SUCCESS);

      EXPECT_EQ(g_transform_override_fault_dry_runs, (std::vector<bool>{true, false, false}));
      EXPECT_EQ(g_transform_override_fault_mutations, (std::vector<bool>{true, false, true}));
      EXPECT_EQ(hook.retry_count(), 1u);
      ASSERT_EQ(g_transform_override_report_layouts.size(), 3u);
      EXPECT_FALSE(g_transform_override_report_layouts[0]);
      EXPECT_FALSE(g_transform_override_report_layouts[1]);
      EXPECT_TRUE(g_transform_override_report_layouts[2]);
      EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{102u});
    }

    EXPECT_TRUE(g_core_memory_allocations.empty());
    EXPECT_EQ(g_core_memory_free_calls, 0);
    EXPECT_EQ(g_core_memory_runtime_reclaim_calls, 1);
  }
}

TEST(HsaHooksUnitTest, ConSanAutoReportRejectsLiveFaultInventoryGrowth) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", nullptr);
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "16777216");

  ScopedEnvVar drop_barrier("RJ_CONSAN_FAULT_DROP_BARRIER", "1");
  ScopedEnvVar require_exactly_one("RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE", "1");
  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result();
  g_transform_override_live_fault_result = g_transform_override_result;
  g_transform_override_live_fault_result->resource_plans.push_back(
      g_transform_override_live_fault_result->resource_plans.front());
  install_test_access_coverage(
      *g_transform_override_live_fault_result, 2u, rocjitsu::consan::SiteDecisionKind::Admitted,
      rocjitsu::consan::AccessPolicyReason::None,
      rocjitsu::consan::LoweringOutcomeKind::Instrumented, rocjitsu::consan::Mode::Default,
      rocjitsu::consan::ProbeIntentKind::Access);
  g_transform_override_models_fault_application = true;

  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();

    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_ERROR_OUT_OF_RESOURCES);
    EXPECT_TRUE(g_loaded_code_object_readers.empty());
  }
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_NE(log.find("live fault transform grew the automatic ConSan report inventory"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("reason=report-live-inventory-growth"), std::string::npos) << log;
  EXPECT_NE(log.find("ConSan fault install"), std::string::npos) << log;
  EXPECT_NE(log.find("applied=1 installed=false"), std::string::npos) << log;
  EXPECT_TRUE(g_core_memory_allocations.empty());
  EXPECT_EQ(g_core_memory_free_calls, 1);
}

TEST(HsaHooksUnitTest, ConSanAutoReportFallbacksStillExecuteLiveFaultTransform) {
  struct Case {
    const char *name;
    bool has_report_sites;
    bool fail_allocation;
  };
  constexpr std::array cases = {
      Case{"no-report-sites", false, false},
      Case{"report-allocation-failure", true, true},
  };
  for (const Case &test : cases) {
    SCOPED_TRACE(test.name);
    ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
    ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", nullptr);
    ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
    ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
    ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "262144");
    ScopedEnvVar drop_barrier("RJ_CONSAN_FAULT_DROP_BARRIER", "1");
    ScopedEnvVar require_exactly_one("RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE", "1");
    ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);

    reset_code_object_observations();
    reset_core_memory_observations();
    g_fail_core_memory_allocate = test.fail_allocation;
    if (test.has_report_sites) {
      g_transform_override_result = auto_report_transform_result();
    } else {
      install_consan_test_program_identity(g_transform_override_result, ROCJITSU_CODE_ARCH_RDNA4,
                                           ROCJITSU_CODE_TARGET_GFX1201);
      g_transform_override_result.outcome = rocjitsu::consan::TransformOutcome::ModifiedValid;
      g_transform_override_result.replacement = {0x7f, 'E', 'L', 'F', 'n', 'o'};
      rocjitsu::consan::ObservationPlan plan;
      plan.mode = rocjitsu::consan::Mode::Default;
      ASSERT_TRUE(plan.valid());
      g_transform_override_result.coverage_ledger =
          rocjitsu::consan::CoverageLedger(std::move(plan));
    }
    g_transform_override_models_fault_application = true;
    {
      FakeApiTable api;
      InstalledDbiHook hook(api);
      ASSERT_TRUE(hook.installed()) << hook.error();

      constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
      hsa_code_object_reader_t reader{};
      ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                      original.size(), &reader),
                HSA_STATUS_SUCCESS);
      EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                  reader, nullptr, nullptr),
                HSA_STATUS_SUCCESS);

      EXPECT_EQ(g_transform_override_fault_dry_runs, (std::vector<bool>{true, false, false}));
      EXPECT_EQ(g_transform_override_fault_mutations, (std::vector<bool>{true, false, true}));
      EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{102u});
    }

    EXPECT_TRUE(g_core_memory_allocations.empty());
  }
}

rocjitsu::consan::TransformArtifacts auto_sc_transform_result() {
  rocjitsu::consan::TransformArtifacts result;
  install_consan_test_program_identity(result, ROCJITSU_CODE_ARCH_CDNA3,
                                       ROCJITSU_CODE_TARGET_GFX942);
  result.outcome = rocjitsu::consan::TransformOutcome::ModifiedValid;
  result.replacement = {0x7f, 'E', 'L', 'F', 's', 'c'};
  rocjitsu::consan::PatchInfo patch;
  patch.phase = rocjitsu::consan::PatchPhase::Instrumentation;
  patch.kind = rocjitsu::consan::PatchKind::LdsStoreCheckTrap;
  result.patches.push_back(patch);
  install_test_access_coverage(result, 1u, rocjitsu::consan::SiteDecisionKind::Admitted,
                               rocjitsu::consan::AccessPolicyReason::None,
                               rocjitsu::consan::LoweringOutcomeKind::Instrumented,
                               rocjitsu::consan::Mode::SuperCollider,
                               rocjitsu::consan::ProbeIntentKind::RedundantAccessObservation);
  return result;
}

TEST(HsaHooksUnitTest, ConSanScAutoReportNeverReconstructsMissingEvidenceFromMechanismTelemetry) {
  for (const bool fail_closed : {false, true}) {
    SCOPED_TRACE(fail_closed);
    ScopedEnvVar mode("RJ_CONSAN_MODE", "supercollider");
    ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", nullptr);
    ScopedEnvVar report_buffer("RJ_CONSAN_SC_REPORT_BUFFER", nullptr);
    ScopedEnvVar fail_closed_value("RJ_CONSAN_FAIL_CLOSED", fail_closed ? "1" : "0");
    ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);
    ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");

    reset_code_object_observations();
    reset_core_memory_observations();
    g_transform_override_result = auto_sc_transform_result();
    g_transform_override_result.coverage_ledger = {};

    testing::internal::CaptureStderr();
    {
      FakeApiTable api;
      InstalledDbiHook hook(api);
      ASSERT_TRUE(hook.installed()) << hook.error();
      constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
      hsa_code_object_reader_t reader{};
      ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                      original.size(), &reader),
                HSA_STATUS_SUCCESS);
      EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                  reader, nullptr, nullptr),
                fail_closed ? HSA_STATUS_ERROR_OUT_OF_RESOURCES : HSA_STATUS_SUCCESS);
      EXPECT_EQ(g_core_memory_allocate_calls, 0);
      EXPECT_TRUE(g_core_memory_allocations.empty());
      if (fail_closed)
        EXPECT_TRUE(g_loaded_code_object_readers.empty());
      else
        EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{reader.handle});
    }
    const std::string log = testing::internal::GetCapturedStderr();
    EXPECT_NE(log.find("missing or invalid typed evidence requirements"), std::string::npos) << log;
  }
}

TEST(HsaHooksUnitTest, ConSanOnUnloadDefersLiveReportFreeToRuntime) {
  struct Case {
    const char *mode;
    bool supercollider;
    bool sampled;
  };
  constexpr std::array cases = {
      Case{"supercollider", true, false},
      Case{"default", false, false},
      Case{"default", false, true},
  };

  for (const Case &test : cases) {
    SCOPED_TRACE(test.mode);
    ScopedEnvVar mode("RJ_CONSAN_MODE", test.mode);
    ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
    ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
    ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
    ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "16777216");

    ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);

    reset_code_object_observations();
    reset_core_memory_observations();
    g_transform_override_result = test.supercollider ? auto_sc_transform_result()
                                  : test.sampled     ? auto_report_transform_result()
                                                     : auto_report_transform_result();
    {
      FakeApiTable api;
      InstalledDbiHook hook(api);
      ASSERT_TRUE(hook.installed()) << hook.error();
      constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
      hsa_code_object_reader_t reader{};
      ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                      original.size(), &reader),
                HSA_STATUS_SUCCESS);
      ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                  reader, nullptr, nullptr),
                HSA_STATUS_SUCCESS);
      ASSERT_EQ(g_core_memory_allocations.size(), 1u);
      EXPECT_EQ(g_core_memory_free_calls, 0);
      EXPECT_EQ(g_core_memory_runtime_reclaim_calls, 0);
    }

    EXPECT_TRUE(g_core_memory_allocations.empty());
    EXPECT_EQ(g_core_memory_free_calls, 0);
    EXPECT_EQ(g_core_memory_runtime_reclaim_calls, 1);
  }
}

TEST(HsaHooksUnitTest, ConSanScAutoReportUsesMarkerAndCleansUpWithoutTrapFallback) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "supercollider");
  ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", nullptr);
  ScopedEnvVar report_buffer("RJ_CONSAN_SC_REPORT_BUFFER", nullptr);
  ScopedEnvVar foreign_report_buffer("RJ_CONSAN_REPORT_BUFFER", "4096");
  ScopedEnvVar foreign_report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", "65536");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  ScopedEnvVar drop_barrier("RJ_CONSAN_FAULT_DROP_BARRIER", "1");

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_sc_transform_result();
  g_transform_override_models_fault_application = true;
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(g_core_memory_allocations.size(), 1u);
    ASSERT_EQ(g_core_memory_allocation_sizes.back(), sizeof(uint32_t));
    *static_cast<uint32_t *>(g_core_memory_allocations.front()) = 1;
    ASSERT_EQ(g_transform_override_sc_report_addresses.size(), 2u);
    EXPECT_FALSE(g_transform_override_sc_report_addresses[0]);
    EXPECT_TRUE(g_transform_override_sc_report_addresses[1]);
    EXPECT_EQ(g_transform_override_fault_mutations, (std::vector<bool>{false, true}));
    ASSERT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
    EXPECT_TRUE(g_core_memory_allocations.empty());
    EXPECT_EQ(g_core_memory_free_calls, 1);
  }
  EXPECT_TRUE(g_core_memory_allocations.empty());
  EXPECT_EQ(g_core_memory_free_calls, 1);
  EXPECT_EQ(g_sc_markers_at_free, std::vector<uint32_t>{1});
}

TEST(HsaHooksUnitTest, ConSanScAutoReportReclaimsEveryExecutableAcrossRegistryChurn) {
  for (const char *policy : {"default", "strict"}) {
    SCOPED_TRACE(policy);
    ScopedEnvVar mode("RJ_CONSAN_MODE", "supercollider");
    ScopedEnvVar selected_policy("RJ_CONSAN_POLICY", policy);
    ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", nullptr);
    ScopedEnvVar report_buffer("RJ_CONSAN_SC_REPORT_BUFFER", nullptr);
    ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", nullptr);
    ScopedEnvVar require_patch("RJ_CONSAN_REQUIRE_PATCH", nullptr);
    ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);

    reset_code_object_observations();
    reset_core_memory_observations();
    g_transform_override_result = auto_sc_transform_result();
    {
      FakeApiTable api;
      InstalledDbiHook hook(api);
      ASSERT_TRUE(hook.installed()) << hook.error();
      constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
      hsa_code_object_reader_t reader{};
      ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                      original.size(), &reader),
                HSA_STATUS_SUCCESS);
      constexpr size_t kLoadCount = 300;
      for (size_t load = 0; load < kLoadCount; ++load) {
        const hsa_executable_t executable{1000u + load};
        ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(executable, kHostAgent, reader,
                                                                    nullptr, nullptr),
                  HSA_STATUS_SUCCESS)
            << load;
        ASSERT_EQ(g_core_memory_allocations.size(), 1u) << load;
        ASSERT_EQ(api.core.hsa_executable_destroy_fn(executable), HSA_STATUS_SUCCESS) << load;
        ASSERT_TRUE(g_core_memory_allocations.empty()) << load;
      }
      EXPECT_EQ(g_core_memory_allocate_calls, static_cast<int>(kLoadCount));
      EXPECT_EQ(g_core_memory_free_calls, static_cast<int>(kLoadCount));
    }
    EXPECT_TRUE(g_core_memory_allocations.empty());
  }
}

TEST(HsaHooksUnitTest, ConSanScTrapIsExplicitAndAllocationFailureDoesNotFallBack) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "supercollider");
  ScopedEnvVar report_buffer("RJ_CONSAN_SC_REPORT_BUFFER", nullptr);
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_sc_transform_result();
  {
    ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", "trap");
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(g_core_memory_allocate_calls, 0);
    ASSERT_EQ(g_transform_override_sc_report_addresses.size(), 1u);
    EXPECT_FALSE(g_transform_override_sc_report_addresses.front());
  }

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_sc_transform_result();
  {
    ScopedEnvVar explicit_report_buffer("RJ_CONSAN_SC_REPORT_BUFFER", "4096");
    ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", nullptr);
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 5, 6, 7, 8};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(g_core_memory_allocate_calls, 0);
    ASSERT_EQ(g_transform_override_sc_report_addresses.size(), 1u);
    EXPECT_EQ(g_transform_override_sc_report_addresses.front(), 4096u);
  }

  reset_code_object_observations();
  reset_core_memory_observations();
  g_fail_core_memory_allocate = true;
  g_transform_override_result = auto_sc_transform_result();
  {
    ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", nullptr);
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 5, 6, 7, 8};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(g_core_memory_allocate_calls, 1);
    ASSERT_EQ(g_transform_override_sc_report_addresses.size(), 1u);
    EXPECT_FALSE(g_transform_override_sc_report_addresses.front());
    EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{reader.handle});
  }
  g_fail_core_memory_allocate = false;
  EXPECT_TRUE(g_core_memory_allocations.empty());
  EXPECT_EQ(g_core_memory_free_calls, 0);
}

TEST(HsaHooksUnitTest, ConSanScAutoReportAllocationFailureRejectsWhenFailClosed) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "supercollider");
  ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", nullptr);
  ScopedEnvVar report_buffer("RJ_CONSAN_SC_REPORT_BUFFER", nullptr);
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "1");
  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);

  reset_code_object_observations();
  reset_core_memory_observations();
  g_fail_core_memory_allocate = true;
  g_transform_override_result = auto_sc_transform_result();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_ERROR_OUT_OF_RESOURCES);
    EXPECT_TRUE(g_loaded_code_object_readers.empty());
  }
  g_fail_core_memory_allocate = false;
}

TEST(HsaHooksUnitTest, ReportsHonorTypedVisibilityAndPreferFineRegion) {
  struct Case {
    const char *mode;
    bool offer_fine_region;
    bool offer_coarse_region;
    bool offer_coarse_region_first;
    hsa_status_t expected_status;
    std::optional<uint64_t> expected_region;
  };
  constexpr std::array cases = {
      Case{"default", true, false, false, HSA_STATUS_SUCCESS, 30u},
      Case{"default", true, true, false, HSA_STATUS_SUCCESS, 30u},
      Case{"default", true, true, true, HSA_STATUS_SUCCESS, 30u},
      Case{"default", false, true, false, HSA_STATUS_ERROR_OUT_OF_RESOURCES, std::nullopt},

  };
  for (const Case &test : cases) {
    SCOPED_TRACE(test.mode);
    ScopedEnvVar mode("RJ_CONSAN_MODE", test.mode);
    ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "1");
    ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
    ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
    ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "67108864");

    ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);

    reset_code_object_observations();
    reset_core_memory_observations();
    g_offer_fine_report_region = test.offer_fine_region;
    g_offer_coarse_report_region = test.offer_coarse_region;
    g_offer_coarse_report_region_first = test.offer_coarse_region_first;
    g_transform_override_result =
        std::string_view(test.mode) == "default"   ? auto_report_transform_result()
        : std::string_view(test.mode) == "default" ? auto_report_transform_result()
                                                   : auto_report_transform_result();
    {
      FakeApiTable api;
      InstalledDbiHook hook(api);
      ASSERT_TRUE(hook.installed()) << hook.error();
      constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
      hsa_code_object_reader_t reader{};
      ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                      original.size(), &reader),
                HSA_STATUS_SUCCESS);
      EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                  reader, nullptr, nullptr),
                test.expected_status);
      if (test.expected_region) {
        ASSERT_EQ(g_core_memory_allocation_regions.size(), 1u);
        EXPECT_EQ(g_core_memory_allocation_regions.front(), *test.expected_region);
      } else {
        EXPECT_TRUE(g_core_memory_allocation_regions.empty());
      }
    }
    EXPECT_TRUE(g_core_memory_allocations.empty());
  }
  g_offer_fine_report_region = true;
  g_offer_coarse_report_region = false;
  g_offer_coarse_report_region_first = false;
}

TEST(HsaHooksUnitTest, ConSanAutoReportUsesExactLayoutAcrossTwoLiveCodeObjectsAndCleansUp) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "1");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", nullptr);

  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();

    constexpr std::array<uint8_t, 8> first = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    constexpr std::array<uint8_t, 8> second = {0x7f, 'E', 'L', 'F', 5, 6, 7, 8};
    hsa_code_object_reader_t first_reader{};
    hsa_code_object_reader_t second_reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(first.data(), first.size(),
                                                                    &first_reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(second.data(), second.size(),
                                                                    &second_reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                first_reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                second_reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);

    ASSERT_EQ(g_core_memory_allocation_sizes.size(), 2u);
    EXPECT_EQ(g_core_memory_allocations.size(), 2u);
    EXPECT_EQ(g_core_memory_free_calls, 0);
    for (size_t size : g_core_memory_allocation_sizes) {
      EXPECT_GE(size, sizeof(rocjitsu::consan::ReportHeader));
      EXPECT_LE(size, static_cast<size_t>(rocjitsu::consan::kOrdinaryAutoReportBufferCeilingBytes));
    }
    ASSERT_EQ(g_transform_override_report_sizes.size(), 4u);
    ASSERT_EQ(g_transform_override_report_layouts.size(), 4u);
    for (size_t inventory_index : {0u, 2u}) {
      EXPECT_EQ(g_transform_override_report_sizes[inventory_index], 0u);
      EXPECT_FALSE(g_transform_override_report_layouts[inventory_index]);
      const size_t patch_index = inventory_index + 1u;
      EXPECT_EQ(g_transform_override_report_sizes[patch_index],
                g_core_memory_allocation_sizes[inventory_index / 2u]);
      ASSERT_TRUE(g_transform_override_report_layouts[patch_index]);
      EXPECT_EQ(g_transform_override_report_layouts[patch_index]->required_bytes,
                g_core_memory_allocation_sizes[inventory_index / 2u]);
    }
    ASSERT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
    EXPECT_TRUE(g_core_memory_allocations.empty());
    EXPECT_EQ(g_core_memory_free_calls, 2);
  }

  EXPECT_TRUE(g_core_memory_allocations.empty());
  EXPECT_EQ(g_core_memory_free_calls, 2);
  ASSERT_EQ(g_core_memory_headers_at_free.size(), 2u);
  for (const auto &header : g_core_memory_headers_at_free) {
    EXPECT_TRUE(report_header_is_current(header));
    EXPECT_GT(header.sync_metadata_capacity, 0u);
    EXPECT_GT(header.causal_window_capacity, 0u);
  }
}

TEST(HsaHooksUnitTest, ConSanAutoReportReclaimsEveryExecutableAcrossProcessBudgetChurn) {
  for (const char *policy : {"default", "strict"}) {
    SCOPED_TRACE(policy);
    ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
    ScopedEnvVar selected_policy("RJ_CONSAN_POLICY", policy);
    ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", nullptr);
    ScopedEnvVar require_patch("RJ_CONSAN_REQUIRE_PATCH", nullptr);
    ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", "0");
    ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
    ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
    ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", nullptr);

    ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);

    reset_code_object_observations();
    reset_core_memory_observations();
    g_transform_override_result = auto_report_transform_result();
    {
      FakeApiTable api;
      InstalledDbiHook hook(api);
      ASSERT_TRUE(hook.installed()) << hook.error();
      constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
      hsa_code_object_reader_t reader{};
      ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                      original.size(), &reader),
                HSA_STATUS_SUCCESS);
      constexpr size_t kLoadCount = 300;
      for (size_t load = 0; load < kLoadCount; ++load) {
        const hsa_executable_t executable{2000u + load};
        ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(executable, kHostAgent, reader,
                                                                    nullptr, nullptr),
                  HSA_STATUS_SUCCESS)
            << load;
        ASSERT_EQ(g_core_memory_allocations.size(), 1u) << load;
        ASSERT_EQ(api.core.hsa_executable_destroy_fn(executable), HSA_STATUS_SUCCESS) << load;
        ASSERT_TRUE(g_core_memory_allocations.empty()) << load;
      }
      EXPECT_EQ(g_core_memory_allocate_calls, static_cast<int>(kLoadCount));
      EXPECT_EQ(g_core_memory_free_calls, static_cast<int>(kLoadCount));
    }
    EXPECT_TRUE(g_core_memory_allocations.empty());
  }
}

TEST(HsaHooksUnitTest, ConSanConcurrentReportLoadsRetainIndependentAllocationIdentities) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "default");
  ScopedEnvVar logging("RJ_CONSAN_LOG", "0");
  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", "0");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", nullptr);
  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result();
  g_block_first_loader_call = true;
  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();
  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);
  auto pending = std::async(std::launch::async, [&] {
    return api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                             reader, nullptr, nullptr);
  });
  struct ReleaseLoader {
    static void release() {
      {
        std::lock_guard lock(g_loader_block_mutex);
        g_release_first_loader_call = true;
      }
      g_loader_block_cv.notify_all();
    }
    ~ReleaseLoader() { release(); }
  } release_loader;
  {
    std::unique_lock lock(g_loader_block_mutex);
    ASSERT_TRUE(g_loader_block_cv.wait_for(lock, std::chrono::seconds(2),
                                           [] { return g_first_loader_call_entered; }));
  }
  ASSERT_EQ(g_core_memory_allocations.size(), 1u);
  auto *const first = static_cast<ReportHeader *>(g_core_memory_allocations.front());
  const uint64_t generation = first->generation;
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{8}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(g_core_memory_allocations.size(), 2u);
  auto *const second = static_cast<ReportHeader *>(g_core_memory_allocations.back());
  EXPECT_NE(first, second);
  EXPECT_EQ(second->generation, generation + 1u);
  ReleaseLoader::release();
  EXPECT_EQ(pending.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(first->generation, generation);
  EXPECT_EQ(hook.checkpoint_after_device_synchronize(), 0u);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{8}), HSA_STATUS_SUCCESS);
  ASSERT_EQ(g_core_memory_allocations.size(), 1u);
  EXPECT_EQ(g_core_memory_allocations.front(), first);
  EXPECT_EQ(first->generation, generation);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  EXPECT_TRUE(g_core_memory_allocations.empty());
}

TEST(HsaHooksUnitTest, ConSanReportGenerationRolloverWithLiveReportsAndDirtyAddressReuse) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "default");
  ScopedEnvVar logging("RJ_CONSAN_LOG", "0");
  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", "0");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", nullptr);
  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result();
  struct ReuseGuard {
    ReuseGuard() { g_reuse_core_memory = true; }
    ~ReuseGuard() {
      g_reuse_core_memory = false;
      std::free(std::exchange(g_recycled_core_memory, nullptr));
      g_recycled_core_memory_size = 0;
    }
  } reuse_guard;
  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();
  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);
  const auto load = [&](uint64_t executable) {
    return api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{executable},
                                                             kHostAgent, reader, nullptr, nullptr);
  };
  ASSERT_EQ(load(7), HSA_STATUS_SUCCESS);
  auto *const first = static_cast<ReportHeader *>(g_core_memory_allocations.front());
  const uint64_t first_generation = first->generation;
  void *reused_address = nullptr;
  constexpr uint64_t period = uint64_t{1} << watchpoint::generation_bits;
  // Actual production allocations, not a counter setter. Transform work is
  // mocked, so a full tag cycle is inexpensive. Keep generation G alive
  // while creating G+period, whose compact tag is identical.
  for (uint64_t index = 1; index <= period; ++index) {
    ASSERT_EQ(load(8), HSA_STATUS_SUCCESS) << index;
    ASSERT_EQ(g_core_memory_allocations.size(), 2u);
    auto *const current = static_cast<ReportHeader *>(g_core_memory_allocations.back());
    if (reused_address) {
      ASSERT_EQ(current, reused_address);
    }
    reused_address = current;
    ASSERT_NE(current, first);
    ASSERT_EQ(current->generation, first_generation + index);
    const size_t size = g_core_memory_allocation_sizes.back();
    const auto *bytes = reinterpret_cast<const uint8_t *>(current);
    ASSERT_TRUE(std::ranges::all_of(std::span(bytes + sizeof(*current), size - sizeof(*current)),
                                    [](uint8_t byte) { return byte == 0; }));
    if (index == period) {
      EXPECT_EQ(current->generation & watchpoint::max_generation,
                first_generation & watchpoint::max_generation);
      // Even an internally consistent stale header must not override the
      // allocation identity held in the host registry.
      current->generation = first_generation;
      EXPECT_NE(hook.checkpoint_after_device_synchronize(), 0u);
      EXPECT_EQ(current->generation, first_generation); // failure is transactional
      current->generation = first_generation + index;
      EXPECT_EQ(hook.checkpoint_after_device_synchronize(), 0u);
      EXPECT_EQ(first->generation, first_generation);
      EXPECT_EQ(current->generation, first_generation + index);
    }
    ASSERT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{8}), HSA_STATUS_SUCCESS);
    ASSERT_EQ(g_core_memory_allocations.size(), 1u);
    // The fake loader keeps history for other tests' lifetime assertions.
    // Retired transient readers are irrelevant here; retaining them makes
    // repeated destruction of handle 8 scan dangling historical pointers.
    // Preserve only the original reader and the still-live executable 7.
    g_loaded_executable_readers.resize(1);
    g_memory_code_object_readers.resize(2);
    g_code_object_reader_inputs.resize(2);
  }
  ASSERT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{7}), HSA_STATUS_SUCCESS);
  EXPECT_TRUE(g_core_memory_allocations.empty());
  hook.unload();
  ASSERT_TRUE(hook.reload(api));
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(load(9), HSA_STATUS_SUCCESS);
  EXPECT_EQ(static_cast<ReportHeader *>(g_core_memory_allocations.front())->generation,
            first_generation + period + 1u);
  EXPECT_EQ(hook.checkpoint_after_device_synchronize(), 0u);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(hsa_executable_t{9}), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, ConSanAutoReportAllocationFailureFailsClosedWithoutLeakingBudget) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "1");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", nullptr);

  reset_code_object_observations();
  reset_core_memory_observations();
  g_fail_core_memory_allocate = true;
  g_transform_override_result = auto_report_atomic_transform_result();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_ERROR_OUT_OF_RESOURCES);
    EXPECT_EQ(g_core_memory_allocate_calls, 1);
    EXPECT_TRUE(g_core_memory_allocations.empty());
    ASSERT_EQ(g_transform_override_report_sizes.size(), 1u);
    EXPECT_EQ(g_transform_override_report_sizes.front(), 0u);
  }
  g_fail_core_memory_allocate = false;
  EXPECT_EQ(g_core_memory_free_calls, 0);
  EXPECT_TRUE(g_core_memory_allocations.empty());
}

TEST(HsaHooksUnitTest, ConSanEpochCheckpointRecyclesInPlace) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "67108864");

  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", "0");

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(g_core_memory_allocations.size(), 1u);
    ASSERT_EQ(g_core_memory_allocation_sizes.size(), 1u);

    auto *const bytes = static_cast<uint8_t *>(g_core_memory_allocations.front());
    const size_t size = g_core_memory_allocation_sizes.front();
    ASSERT_GT(size, sizeof(rocjitsu::consan::ReportHeader));
    auto *const header = reinterpret_cast<rocjitsu::consan::ReportHeader *>(bytes);
    const rocjitsu::consan::ReportHeader original_header = *header;
    header->event_counter = 17u;
    bytes[size - 1u] = 0xa5u;
    void *const allocation = bytes;

    EXPECT_EQ(hook.checkpoint_after_device_synchronize(), 0u);
    EXPECT_EQ(g_core_memory_allocations.size(), 1u);
    EXPECT_EQ(g_core_memory_allocations.front(), allocation);
    EXPECT_EQ(header->generation, original_header.generation);
    EXPECT_EQ(header->dispatch_id, original_header.dispatch_id);
    EXPECT_EQ(header->event_counter, 0u);
    EXPECT_EQ(header->flags, 0u);
    EXPECT_TRUE(std::ranges::all_of(
        std::span<const uint8_t>(bytes + sizeof(*header), size - sizeof(*header)),
        [](uint8_t byte) { return byte == 0u; }));

    // Empty epochs are legal and recycling never reallocates the report.
    EXPECT_EQ(hook.checkpoint_after_device_synchronize(), 0u);
    EXPECT_EQ(g_core_memory_allocations.front(), allocation);
  }
  EXPECT_TRUE(g_core_memory_allocations.empty());
}

TEST(HsaHooksUnitTest, ConSanEpochCheckpointAccumulatesEvidenceAcrossEpochs) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "67108864");
  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", "0");

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result();
  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(g_core_memory_allocations.size(), 1u);
    auto *const header =
        static_cast<rocjitsu::consan::ReportHeader *>(g_core_memory_allocations.front());
    ASSERT_TRUE(g_transform_override_report_layouts.back());
    const auto &layout = *g_transform_override_report_layouts.back();
    const auto publish = [&](uint32_t count) {
      ASSERT_GE(layout.causal_window_capacity, count);
      ASSERT_GE(layout.watchpoint_capacity, count);
      auto *bytes = reinterpret_cast<uint8_t *>(header);
      for (uint32_t index = 0; index < count; ++index) {
        const rocjitsu::consan::CausalWindow window{
            .generation = header->generation,
            .dispatch_id = header->dispatch_id,
            .first_entry = index,
            .entry_count = 1,
            .publication_state =
                static_cast<uint32_t>(rocjitsu::consan::CausalPublicationState::Ready),
        };
        std::memcpy(bytes + layout.causal_windows_offset + index * sizeof(window), &window,
                    sizeof(window));
        const uint64_t packed = pack_watchpoint_entry(rocjitsu::consan::ShadowAccessKind::Write, 0,
                                                      0, header->generation, 16, 4);
        std::memcpy(bytes + layout.watchpoints_offset + index * sizeof(packed), &packed,
                    sizeof(packed));
      }
      header->causal_window_count = count;
    };
    publish(3u);
    ASSERT_EQ(hook.checkpoint_after_device_synchronize(), 0u);
    publish(5u);
    ASSERT_EQ(hook.checkpoint_after_device_synchronize(), 0u);
  }
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("completed_epochs=2"), std::string::npos) << log;
  EXPECT_NE(log.find("visible_evidence=8"), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, ConSanEpochCheckpointSustainsManyRepeatedRuns) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "67108864");
  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", "0");

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result();
  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(g_core_memory_allocations.size(), 1u);
    void *const allocation = g_core_memory_allocations.front();
    auto *const header = static_cast<rocjitsu::consan::ReportHeader *>(allocation);
    for (uint32_t epoch = 0; epoch < 256u; ++epoch) {
      header->event_counter = epoch + 1u;
      ASSERT_EQ(hook.checkpoint_after_device_synchronize(), 0u) << epoch;
      ASSERT_EQ(g_core_memory_allocations.size(), 1u) << epoch;
      EXPECT_EQ(g_core_memory_allocations.front(), allocation) << epoch;
      EXPECT_EQ(header->event_counter, 0u) << epoch;
    }
  }
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("completed_epochs=256"), std::string::npos) << log;
  EXPECT_EQ(g_core_memory_allocate_calls, 1);
}

TEST(HsaHooksUnitTest, ConSanEpochCheckpointPreservesPriorLossAndClearsNextEpoch) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "16777216");

  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", "0");

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result();
  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(g_core_memory_allocations.size(), 1u);
    auto *const header =
        static_cast<rocjitsu::consan::ReportHeader *>(g_core_memory_allocations.front());
    header->dropped_window_count = 1u;
    ASSERT_EQ(hook.checkpoint_after_device_synchronize(), 0u);
    EXPECT_EQ(header->dropped_window_count, 0u);
    header->event_counter = 1u;
    ASSERT_EQ(hook.checkpoint_after_device_synchronize(), 0u);
    EXPECT_EQ(header->event_counter, 0u);
  }
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("completed_epochs=2"), std::string::npos) << log;
  EXPECT_NE(log.find("dynamic_incomplete=1"), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, ConSanEpochCheckpointIsTransactionalAcrossLiveReports) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "16777216");

  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", "0");

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> first = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    constexpr std::array<uint8_t, 8> second = {0x7f, 'E', 'L', 'F', 5, 6, 7, 8};
    hsa_code_object_reader_t first_reader{};
    hsa_code_object_reader_t second_reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(first.data(), first.size(),
                                                                    &first_reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(second.data(), second.size(),
                                                                    &second_reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                first_reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                second_reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(g_core_memory_allocations.size(), 2u);
    auto *const first_header =
        static_cast<rocjitsu::consan::ReportHeader *>(g_core_memory_allocations[0]);
    auto *const second_header =
        static_cast<rocjitsu::consan::ReportHeader *>(g_core_memory_allocations[1]);
    const rocjitsu::consan::ReportHeader valid_second_header = *second_header;
    first_header->event_counter = 11u;
    second_header->magic = 0u;

    EXPECT_EQ(hook.checkpoint_after_device_synchronize(), 3u);
    EXPECT_EQ(first_header->event_counter, 11u);
    EXPECT_EQ(second_header->magic, 0u);

    *second_header = valid_second_header;
    second_header->event_counter = 13u;
    EXPECT_EQ(hook.checkpoint_after_device_synchronize(), 0u);
    EXPECT_EQ(first_header->event_counter, 0u);
    EXPECT_EQ(second_header->event_counter, 0u);
  }
}

TEST(HsaHooksUnitTest, ConSanEpochCheckpointIsNoOpOutside) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "supercollider");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
  ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", nullptr);
  ScopedEnvVar report_buffer("RJ_CONSAN_SC_REPORT_BUFFER", nullptr);
  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_sc_transform_result();
  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();
  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(g_core_memory_allocations.size(), 1u);
  auto *const marker = static_cast<uint32_t *>(g_core_memory_allocations.front());
  *marker = 1u;
  EXPECT_EQ(hook.checkpoint_after_device_synchronize(), 2u);
  EXPECT_EQ(*marker, 1u);
  hook.unload();
  EXPECT_EQ(hook.checkpoint_after_device_synchronize(), 1u);
}

TEST(HsaHooksUnitTest, ConSanAutomaticallyCheckpointsAtTrackedGlobalQuiescence) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "16777216");

  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", "0");

  reset_code_object_observations();
  reset_core_memory_observations();
  reset_queue_fakes();
  g_fake_symbol_name = "auto_report_atomic.kd";
  g_transform_override_result = auto_report_atomic_transform_result();
  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
    api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    EXPECT_NE(api.core.hsa_signal_wait_relaxed_fn, fake_signal_wait_relaxed);
    EXPECT_NE(api.core.hsa_signal_wait_scacquire_fn, fake_signal_wait_scacquire);

    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kHostAgent, reader,
                                                                nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(g_core_memory_allocations.size(), 1u);

    hsa_queue_t *queue = nullptr;
    ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr,
                                           0, 0, &queue),
              HSA_STATUS_SUCCESS);
    ASSERT_NE(g_fake_intercept_handler, nullptr);

    g_fake_symbol_kernel_object = 0x12345678u;
    hsa_executable_symbol_t symbol{};
    ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                  kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
              HSA_STATUS_SUCCESS);

    constexpr hsa_signal_t first_signal{7001u};
    constexpr hsa_signal_t second_signal{7002u};
    set_fake_signal_value(first_signal, 1);
    set_fake_signal_value(second_signal, 1);
    std::array<hsa_kernel_dispatch_packet_t, 2> packets{};
    for (auto &packet : packets) {
      packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
      packet.kernel_object = g_fake_symbol_kernel_object;
    }
    packets[0].completion_signal = first_signal;
    packets[1].completion_signal = second_signal;
    g_fake_intercept_handler(packets.data(), packets.size(), 0u, g_fake_intercept_user_data,
                             fake_intercept_packet_writer);

    auto *const header =
        static_cast<rocjitsu::consan::ReportHeader *>(g_core_memory_allocations.front());
    header->event_counter = 7u;
    set_fake_signal_value(first_signal, 0);
    EXPECT_EQ(api.core.hsa_signal_wait_scacquire_fn(first_signal, HSA_SIGNAL_CONDITION_LT, 1u,
                                                    UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
              0);
    EXPECT_EQ(header->event_counter, 7u);

    set_fake_signal_value(second_signal, 0);
    EXPECT_EQ(api.core.hsa_signal_wait_relaxed_fn(second_signal, HSA_SIGNAL_CONDITION_LT, 1u,
                                                  UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
              0);
    EXPECT_EQ(header->event_counter, 0u);

    constexpr hsa_signal_t third_signal{7003u};
    set_fake_signal_value(third_signal, 1);
    packets[0].completion_signal = third_signal;
    g_fake_intercept_handler(packets.data(), 1u, 2u, g_fake_intercept_user_data,
                             fake_intercept_packet_writer);
    header->event_counter = 9u;
    set_fake_signal_value(third_signal, 0);
    EXPECT_EQ(api.core.hsa_signal_wait_scacquire_fn(third_signal, HSA_SIGNAL_CONDITION_LT, 1u,
                                                    UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
              0);
    EXPECT_EQ(header->event_counter, 0u);
    EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
  }
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("completed_epochs=2"), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, ConSanAutomaticEpochAnalysisSelectionDiscardsUnselectedEpochs) {
  struct Case {
    const char *policy;
    size_t epoch_count;
    size_t window_epoch;
    uint64_t analyzed_reports;
    uint64_t discarded_reports;
  };
  constexpr size_t kNeverArm = std::numeric_limits<size_t>::max();
  constexpr std::array cases = {
      Case{"nth:2", 3, kNeverArm, 1, 2},
      Case{"periodic:3:2", 7, kNeverArm, 2, 5},
      Case{"manual", 3, 1, 1, 2},
  };

  for (const Case &test : cases) {
    SCOPED_TRACE(test.policy);
    ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
    ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
    ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
    ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
    ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "16777216");

    ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", "0");
    ScopedEnvVar epoch_analysis("RJ_CONSAN_EPOCH_ANALYSIS", test.policy);

    reset_code_object_observations();
    reset_core_memory_observations();
    reset_queue_fakes();
    g_fake_symbol_name = "auto_report_atomic.kd";
    g_transform_override_result = auto_report_atomic_transform_result();
    testing::internal::CaptureStderr();
    {
      FakeApiTable api;
      api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
      api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
      InstalledDbiHook hook(api);
      ASSERT_TRUE(hook.installed()) << hook.error();

      constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
      hsa_code_object_reader_t reader{};
      ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                      original.size(), &reader),
                HSA_STATUS_SUCCESS);
      ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kHostAgent,
                                                                  reader, nullptr, nullptr),
                HSA_STATUS_SUCCESS);
      ASSERT_EQ(g_core_memory_allocations.size(), 1u);

      hsa_queue_t *queue = nullptr;
      ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr,
                                             nullptr, 0, 0, &queue),
                HSA_STATUS_SUCCESS);
      g_fake_symbol_kernel_object = 0x12345678u;
      hsa_executable_symbol_t symbol{};
      ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                    kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
                HSA_STATUS_SUCCESS);
      auto *const header =
          static_cast<rocjitsu::consan::ReportHeader *>(g_core_memory_allocations.front());

      for (size_t epoch = 0; epoch < test.epoch_count; ++epoch) {
        if (epoch == test.window_epoch) {
          ASSERT_EQ(hook.begin_epoch_analysis_window(), 0u);
        }
        const hsa_signal_t signal{7100u + epoch};
        set_fake_signal_value(signal, 1);
        hsa_kernel_dispatch_packet_t packet{};
        packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
        packet.kernel_object = g_fake_symbol_kernel_object;
        packet.completion_signal = signal;
        g_fake_intercept_handler(&packet, 1u, epoch, g_fake_intercept_user_data,
                                 fake_intercept_packet_writer);
        header->event_counter = epoch + 1u;
        set_fake_signal_value(signal, 0);
        EXPECT_EQ(api.core.hsa_signal_wait_scacquire_fn(signal, HSA_SIGNAL_CONDITION_LT, 1u,
                                                        UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
                  0);
        EXPECT_EQ(header->event_counter, 0u);
        if (epoch == test.window_epoch) {
          ASSERT_EQ(hook.end_epoch_analysis_window(), 0u);
        }
      }
      EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
    }
    const std::string log = testing::internal::GetCapturedStderr();
    EXPECT_NE(log.find("completed_epochs=" + std::to_string(test.analyzed_reports)),
              std::string::npos)
        << log;
    EXPECT_NE(log.find("discarded_epochs=" + std::to_string(test.discarded_reports)),
              std::string::npos)
        << log;
  }
}

TEST(HsaHooksUnitTest, ConSanAutomaticEpochSelectionRetriesAfterTransactionalFailure) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "16777216");

  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", "0");
  ScopedEnvVar epoch_analysis("RJ_CONSAN_EPOCH_ANALYSIS", "nth:1");
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");

  reset_code_object_observations();
  reset_core_memory_observations();
  reset_queue_fakes();
  g_fake_symbol_name = "auto_report_atomic.kd";
  g_transform_override_result = auto_report_atomic_transform_result();
  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
    api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();

    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kHostAgent, reader,
                                                                nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(g_core_memory_allocations.size(), 1u);

    hsa_queue_t *queue = nullptr;
    ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr,
                                           0, 0, &queue),
              HSA_STATUS_SUCCESS);
    g_fake_symbol_kernel_object = 0x12345678u;
    hsa_executable_symbol_t symbol{};
    ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                  kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
              HSA_STATUS_SUCCESS);
    auto *const header =
        static_cast<rocjitsu::consan::ReportHeader *>(g_core_memory_allocations.front());
    const uint32_t expected_magic = header->magic;

    const hsa_signal_t signal{7199u};
    set_fake_signal_value(signal, 1);
    hsa_kernel_dispatch_packet_t packet{};
    packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
    packet.kernel_object = g_fake_symbol_kernel_object;
    packet.completion_signal = signal;
    g_fake_intercept_handler(&packet, 1u, 0u, g_fake_intercept_user_data,
                             fake_intercept_packet_writer);
    header->event_counter = 1u;
    header->magic = 0u;
    set_fake_signal_value(signal, 0);
    EXPECT_EQ(api.core.hsa_signal_wait_scacquire_fn(signal, HSA_SIGNAL_CONDITION_LT, 1u, UINT64_MAX,
                                                    HSA_WAIT_STATE_BLOCKED),
              0);
    EXPECT_EQ(header->event_counter, 1u);

    header->magic = expected_magic;
    EXPECT_EQ(api.core.hsa_signal_wait_scacquire_fn(signal, HSA_SIGNAL_CONDITION_LT, 1u, UINT64_MAX,
                                                    HSA_WAIT_STATE_BLOCKED),
              0);
    EXPECT_EQ(header->event_counter, 0u);
    EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
  }
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("outcome=snapshot-failed"), std::string::npos) << log;
  EXPECT_NE(log.find("outcome=complete automatic_epoch=1"), std::string::npos) << log;
  EXPECT_EQ(log.find("automatic_epoch=2"), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, ConSanManualEpochAnalysisWindowHasStrictLifecycle) {
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar epoch_analysis("RJ_CONSAN_EPOCH_ANALYSIS", "every");
  reset_code_object_observations();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    EXPECT_EQ(hook.begin_epoch_analysis_window(), 3u);
    EXPECT_EQ(hook.end_epoch_analysis_window(), 3u);
  }

  ScopedEnvVar manual_epoch_analysis("RJ_CONSAN_EPOCH_ANALYSIS", "manual");
  reset_code_object_observations();
  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();
  EXPECT_EQ(hook.end_epoch_analysis_window(), 3u);
  EXPECT_EQ(hook.begin_epoch_analysis_window(), 0u);
  EXPECT_EQ(hook.begin_epoch_analysis_window(), 3u);
  EXPECT_EQ(hook.end_epoch_analysis_window(), 0u);
  EXPECT_EQ(hook.end_epoch_analysis_window(), 3u);
}

TEST(HsaHooksUnitTest, ConSanUsesOrderedBarrierAsCompletionProxyForSignalLessDispatch) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "16777216");

  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", "0");

  reset_code_object_observations();
  reset_core_memory_observations();
  reset_queue_fakes();
  g_fake_symbol_name = "auto_report_atomic.kd";
  g_transform_override_result = auto_report_atomic_transform_result();
  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
    api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();

    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kHostAgent, reader,
                                                                nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(g_core_memory_allocations.size(), 1u);
    hsa_queue_t *queue = nullptr;
    ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr,
                                           0, 0, &queue),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(g_fake_intercept_user_data, queue);

    g_fake_symbol_kernel_object = 0x12345678u;
    hsa_executable_symbol_t symbol{};
    ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                  kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
              HSA_STATUS_SUCCESS);

    auto submit_epoch = [&](hsa_signal_t proxy_signal, bool ordered) {
      set_fake_signal_value(proxy_signal, 1);
      std::array<hsa_kernel_dispatch_packet_t, 2> packets{};
      packets[0].header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
      packets[0].kernel_object = g_fake_symbol_kernel_object;
      packets[0].completion_signal = {};
      auto *const barrier = reinterpret_cast<hsa_barrier_and_packet_t *>(&packets[1]);
      barrier->header =
          static_cast<uint16_t>((HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                                (static_cast<uint16_t>(ordered) << HSA_PACKET_HEADER_BARRIER));
      barrier->completion_signal = proxy_signal;
      g_fake_intercept_handler(packets.data(), packets.size(), 0u, g_fake_intercept_user_data,
                               fake_intercept_packet_writer);
    };

    auto *const header =
        static_cast<rocjitsu::consan::ReportHeader *>(g_core_memory_allocations.front());
    constexpr hsa_signal_t unordered_signal{7020u};
    submit_epoch(unordered_signal, false);
    header->event_counter = 5u;
    set_fake_signal_value(unordered_signal, 0);
    EXPECT_EQ(api.core.hsa_signal_wait_scacquire_fn(unordered_signal, HSA_SIGNAL_CONDITION_LT, 1u,
                                                    UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
              0);
    EXPECT_EQ(header->event_counter, 5u);

    // An ordered barrier without its own completion signal is safe but cannot
    // yet prove completion to the host. A later ordered barrier can still
    // cover the whole prefix and restore automatic tracking.
    hsa_barrier_and_packet_t unobservable_barrier{};
    unobservable_barrier.header =
        static_cast<uint16_t>((HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                              (1u << HSA_PACKET_HEADER_BARRIER));
    g_fake_intercept_handler(&unobservable_barrier, 1u, 2u, g_fake_intercept_user_data,
                             fake_intercept_packet_writer);
    EXPECT_EQ(header->event_counter, 5u);

    // An observable ordered barrier on the same queue completes every
    // preceding packet, including the signal-less dispatch above.
    constexpr hsa_signal_t first_proxy{7021u};
    set_fake_signal_value(first_proxy, 1);
    hsa_barrier_and_packet_t first_barrier{};
    first_barrier.header =
        static_cast<uint16_t>((HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                              (1u << HSA_PACKET_HEADER_BARRIER));
    first_barrier.completion_signal = first_proxy;
    g_fake_intercept_handler(&first_barrier, 1u, 3u, g_fake_intercept_user_data,
                             fake_intercept_packet_writer);
    set_fake_signal_value(first_proxy, 0);
    EXPECT_EQ(api.core.hsa_signal_wait_relaxed_fn(first_proxy, HSA_SIGNAL_CONDITION_LT, 1u,
                                                  UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
              0);
    EXPECT_EQ(header->event_counter, 0u);

    constexpr hsa_signal_t second_proxy{7022u};
    submit_epoch(second_proxy, true);
    header->event_counter = 9u;
    set_fake_signal_value(second_proxy, 0);
    EXPECT_EQ(api.core.hsa_signal_wait_scacquire_fn(second_proxy, HSA_SIGNAL_CONDITION_LT, 1u,
                                                    UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
              0);
    EXPECT_EQ(header->event_counter, 0u);
    EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
  }
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("completed_epochs=2"), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, ConSanExplicitCheckpointRecoversUntrackableDispatchEpoch) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "16777216");

  ScopedEnvVar require_records("RJ_CONSAN_REQUIRE_RECORDS", "0");

  reset_code_object_observations();
  reset_core_memory_observations();
  reset_queue_fakes();
  g_fake_symbol_name = "auto_report_atomic.kd";
  g_transform_override_result = auto_report_atomic_transform_result();
  FakeApiTable api;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kHostAgent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(g_core_memory_allocations.size(), 1u);
  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  g_fake_symbol_kernel_object = 0x12345678u;
  hsa_executable_symbol_t symbol{};
  ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
            HSA_STATUS_SUCCESS);

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = g_fake_symbol_kernel_object;
  packet.completion_signal = {};
  ASSERT_NE(g_fake_intercept_handler, nullptr);
  g_fake_intercept_handler(&packet, 1u, 0u, g_fake_intercept_user_data,
                           fake_intercept_packet_writer);
  auto *const header =
      static_cast<rocjitsu::consan::ReportHeader *>(g_core_memory_allocations.front());
  header->event_counter = 5u;
  constexpr hsa_signal_t unrelated_signal{7010u};
  set_fake_signal_value(unrelated_signal, 0);
  EXPECT_EQ(api.core.hsa_signal_wait_scacquire_fn(unrelated_signal, HSA_SIGNAL_CONDITION_LT, 1u,
                                                  UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
            0);
  EXPECT_EQ(header->event_counter, 5u);

  // The caller has established device-wide quiescence out of band.
  EXPECT_EQ(hook.checkpoint_after_device_synchronize(), 0u);
  EXPECT_EQ(header->event_counter, 0u);
  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, AtomicPairAcceptsCompleteReleaseToAcquireEvidence) {
  using Outcome = rocjitsu::consan::SyncOutcome;
  using Role = rocjitsu::consan::SyncRole;
  using Scope = rocjitsu::consan::SyncScope;

  const auto release = atomic(Role::Release, Scope::Workgroup, Outcome::NotApplicable);
  const auto acquire = atomic(Role::Acquire, Scope::System, Outcome::NotApplicable);
  EXPECT_TRUE(atomic_pair_orders_same_workgroup(release, acquire));

  const auto successful_cas = atomic(Role::RmwAcquireRelease, Scope::Agent, Outcome::CasSuccess);
  const auto failed_acquire = atomic(Role::RmwAcquire, Scope::Workgroup, Outcome::CasFailure);
  EXPECT_TRUE(atomic_pair_orders_same_workgroup(successful_cas, failed_acquire));
}

TEST(HsaHooksUnitTest, AtomicPairRejectsIncompleteDirectionRangeScopeEpochAndOutcome) {
  using Outcome = rocjitsu::consan::SyncOutcome;
  using Role = rocjitsu::consan::SyncRole;
  using Scope = rocjitsu::consan::SyncScope;

  const auto release = atomic(Role::RmwRelease, Scope::Agent, Outcome::RmwReturnsOld);
  const auto acquire = atomic(Role::RmwAcquire, Scope::Agent, Outcome::RmwReturnsOld);
  EXPECT_TRUE(atomic_pair_orders_same_workgroup(acquire, release));
  EXPECT_FALSE(atomic_pair_orders_same_workgroup(
      release, atomic(Role::Release, Scope::System, Outcome::NotApplicable)));
  EXPECT_FALSE(atomic_pair_orders_same_workgroup(
      acquire, atomic(Role::Acquire, Scope::System, Outcome::NotApplicable)));
  EXPECT_FALSE(atomic_pair_orders_same_workgroup(
      release, atomic(Role::RmwAcquire, Scope::Agent, Outcome::RmwReturnsOld, 0x1004)));
  EXPECT_FALSE(atomic_pair_orders_same_workgroup(
      release, atomic(Role::RmwAcquire, Scope::Agent, Outcome::RmwReturnsOld, 0x1000, 8)));
  EXPECT_FALSE(atomic_pair_orders_same_workgroup(
      atomic(Role::RmwRelease, Scope::Wavefront, Outcome::RmwReturnsOld), acquire));
  EXPECT_FALSE(atomic_pair_orders_same_workgroup(
      release, atomic(Role::RmwAcquire, Scope::Agent, Outcome::RmwReturnsOld, 0x1000, 4, 8)));
  EXPECT_FALSE(atomic_pair_orders_same_workgroup(
      atomic(Role::RmwRelease, Scope::Agent, Outcome::CasFailure), acquire));
}

TEST(HsaHooksUnitTest, AtomicPairFailsClosedOnMissingMalformedAndCollidingHalves) {
  using Classification = rocjitsu::consan::SyncClassification;
  using Outcome = rocjitsu::consan::SyncOutcome;
  using Role = rocjitsu::consan::SyncRole;
  using Scope = rocjitsu::consan::SyncScope;

  const auto release = atomic(Role::Release, Scope::Agent, Outcome::NotApplicable);
  const auto acquire = atomic(Role::Acquire, Scope::Agent, Outcome::NotApplicable);
  auto malformed = acquire;
  malformed.classification = Classification::Malformed;
  EXPECT_FALSE(atomic_pair_orders_same_workgroup(release, malformed));
  EXPECT_FALSE(atomic_pair_orders_same_workgroup(rocjitsu::consan::SyncDecodeResult{}, acquire));

  EXPECT_TRUE(sync_report_is_complete(0, 0, 0));
  EXPECT_FALSE(sync_report_is_complete(1, 0, 0));
  EXPECT_FALSE(sync_report_is_complete(0, 1, 0));
  EXPECT_FALSE(sync_report_is_complete(0, 0, 1));
  EXPECT_FALSE(sync_report_is_complete(0, 0, 0, 1, 0));
  EXPECT_FALSE(sync_report_is_complete(0, 0, 0, 0, 1));
}

TEST(HsaHooksUnitTest, AutoReportMetadataMatchesReaderAndGeneration) {
  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  const std::string fingerprint = "code_object=" + make_code_object_id(original).fingerprint;
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "1");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "16777216");

  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result();
  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();

    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    for (size_t load = 0; load < 2; ++load) {
      SCOPED_TRACE(load);
      EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                  reader, nullptr, nullptr),
                HSA_STATUS_SUCCESS);
    }
  }
  const std::string log = testing::internal::GetCapturedStderr();

  const size_t first_fingerprint = log.find(fingerprint);
  ASSERT_NE(first_fingerprint, std::string::npos) << log;
  const size_t second_fingerprint = log.find(fingerprint, first_fingerprint + fingerprint.size());
  ASSERT_NE(second_fingerprint, std::string::npos) << log;
  EXPECT_EQ(log.find(fingerprint, second_fingerprint + fingerprint.size()), std::string::npos)
      << log;
  EXPECT_EQ(log.find("code_object=missing"), std::string::npos) << log;
  EXPECT_TRUE(g_core_memory_allocations.empty());
  EXPECT_EQ(g_core_memory_free_calls, 0);
  EXPECT_EQ(g_core_memory_runtime_reclaim_calls, 2);
}

TEST(HsaHooksUnitTest, AutoReportLogsStaticMappingProvenance) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "1");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "4194304");
  ScopedEnvVar runtime_stride("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE", "1");

  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result();
  g_seed_auto_report_on_load = true;

  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
  }
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(g_seed_auto_report_succeeded) << log;
  const size_t detail = log.find("ConSan access reader=101");
  ASSERT_NE(detail, std::string::npos) << log;
  for (std::string_view field :
       {"dispatch=0x1122334455667788", "workgroup=(3,4,5)", "instruction=0x120", "trampoline=0x440",
        "relocated_guest=0x448", "scratch_vgpr=12", "range=0", "bank=0", "mapped=true"}) {
    EXPECT_NE(log.find(field, detail), std::string::npos) << field << "\n" << log;
  }
  // Access-only ConSan reports cannot publish pending atomic acquires. Even
  // with a visible watchpoint, report teardown must not scan the capacity-sized
  // pending table once for every visible entry.
  EXPECT_NE(log.find("pending_acquires=0 pending_acquire_contention=0 "), std::string::npos) << log;
  EXPECT_NE(log.find("pending_release_slots_examined=0"), std::string::npos) << log;
  EXPECT_NE(log.find("watchpoint_slots_examined=1"), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, AutoReportDoesNotInferAttributionFromRawPatchTelemetry) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "1");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "4194304");
  ScopedEnvVar runtime_stride("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE", "1");

  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result();
  g_transform_override_result.coverage_ledger.discard_instrumented_lowerings();
  rocjitsu::consan::PatchInfo raw_patch_telemetry;
  raw_patch_telemetry.kind = rocjitsu::consan::PatchKind::TrampolineWatchpointStore;
  raw_patch_telemetry.anchor_offset = 0x120u;
  raw_patch_telemetry.trampoline_offset = 0x440u;
  raw_patch_telemetry.relocated_guest_instruction_offset = 0x448u;
  raw_patch_telemetry.scratch_vgpr = 12u;
  raw_patch_telemetry.owner_descriptor_file_offsets = {0x100u};
  g_transform_override_result.patches.push_back(std::move(raw_patch_telemetry));
  g_seed_auto_report_on_load = true;

  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
  }
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(g_seed_auto_report_succeeded) << log;
  EXPECT_EQ(log.find("ConSan diagnostic map reader=101"), std::string::npos) << log;
  const size_t detail = log.find("ConSan access reader=101");
  ASSERT_NE(detail, std::string::npos) << log;
  EXPECT_NE(log.find("instruction=0x0", detail), std::string::npos) << log;
  EXPECT_NE(log.find("mapped=false", detail), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, AutoPendingReleaseScansEachRelevantOwnerBankOnce) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "1");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "4194304");
  ScopedEnvVar runtime_stride("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE", "1");

  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result();
  g_seed_auto_report_on_load = true;
  g_seed_auto_pending_release_scale = true;

  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
  }
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(g_seed_auto_report_succeeded) << log;
  ASSERT_FALSE(g_transform_override_report_layouts.empty());
  ASSERT_TRUE(g_transform_override_report_layouts.back());
  const uint32_t capacity = g_transform_override_report_layouts.back()->causal_window_capacity;
  ASSERT_NE(capacity, 0u);
  EXPECT_NE(log.find("visible=2"), std::string::npos) << log;
  EXPECT_NE(log.find("visible_sync=2"), std::string::npos) << log;
  EXPECT_NE(log.find("watchpoint_slots_examined=2"), std::string::npos) << log;
  EXPECT_NE(log.find("pending_acquires=1"), std::string::npos) << log;
  EXPECT_NE(log.find("pending_release_slots_examined=" + std::to_string(capacity)),
            std::string::npos)
      << log;
  EXPECT_EQ(log.find("pending_release_slots_examined=" + std::to_string(2u * capacity)),
            std::string::npos)
      << "report teardown must scan the relevant bank once, not once per visible access\n"
      << log;
}

TEST(HsaHooksUnitTest, AutoIgnoresPendingAcquireIdentityCollision) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "1");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "4194304");
  ScopedEnvVar runtime_stride("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE", "1");

  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result();
  g_seed_auto_report_on_load = true;
  g_seed_auto_pending_identity_collision = true;

  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
  }
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(g_seed_auto_report_succeeded) << log;
  EXPECT_NE(log.find("visible=1"), std::string::npos) << log;
  EXPECT_NE(log.find("pending_acquires=1"), std::string::npos) << log;
  EXPECT_NE(log.find("malformed_sync=0"), std::string::npos) << log;
  EXPECT_NE(log.find("dynamic_complete=true"), std::string::npos) << log;
}

TEST(HsaHooksUnitTest, AutoConflictRequiresSameDispatchClusterAndKernelOwnerScope) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "1");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "4194304");
  ScopedEnvVar runtime_stride("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE", "1");

  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");

  struct Case {
    std::string_view name;
    ReportOwnerScope owner_scope;
    bool distinct_dispatches = false;
    bool distinct_clusters = false;
    uint32_t expected_conflicts = 0;
  };
  constexpr std::array cases = {
      Case{"same dispatch, cluster, and owner", ReportOwnerScope::SharedOwnerPair, false, false,
           1u},
      Case{"disjoint kernel owners", ReportOwnerScope::DisjointOwnerPair, false, false, 0u},
      Case{"missing kernel-owner provenance remains conservative",
           ReportOwnerScope::UnknownOwnerPair, false, false, 1u},
      Case{"different dispatches", ReportOwnerScope::SharedOwnerPair, true, false, 0u},
      Case{"different cluster workgroups", ReportOwnerScope::SharedOwnerPair, false, true, 0u},
  };
  for (const Case &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    reset_code_object_observations();
    reset_core_memory_observations();
    g_transform_override_result =
        auto_report_transform_result(/*malformed_mapping=*/false, test_case.owner_scope);
    g_seed_auto_report_on_load = true;
    g_seed_auto_conflict_pair = true;
    g_seed_auto_distinct_dispatches = test_case.distinct_dispatches;
    g_seed_auto_distinct_clusters = test_case.distinct_clusters;

    testing::internal::CaptureStderr();
    {
      FakeApiTable api;
      InstalledDbiHook hook(api);
      ASSERT_TRUE(hook.installed()) << hook.error();
      constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
      hsa_code_object_reader_t reader{};
      ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                      original.size(), &reader),
                HSA_STATUS_SUCCESS);
      EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                  reader, nullptr, nullptr),
                HSA_STATUS_SUCCESS);
    }
    const std::string log = testing::internal::GetCapturedStderr();

    EXPECT_TRUE(g_seed_auto_report_succeeded) << log;
    EXPECT_NE(log.find("visible=2"), std::string::npos) << log;
    EXPECT_NE(log.find("conflicts=" + std::to_string(test_case.expected_conflicts)),
              std::string::npos)
        << log;
  }
}

TEST(HsaHooksUnitTest, AutoExampleBudgetSpansReportsWithoutSuppressingCounts) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "1");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "4194304");
  ScopedEnvVar runtime_stride("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE", "1");
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  ScopedEnvVar example_limit("RJ_CONSAN_CONFLICT_LIMIT", "1");
  ScopedEnvVar total_limit("RJ_CONSAN_TOTAL_CONFLICT_LIMIT", "2");
  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result =
      auto_report_transform_result(false, ReportOwnerScope::SharedOwnerPair);
  g_seed_auto_conflict_pair = true;
  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    for (uint8_t i = 0; i < 3; ++i) {
      g_seed_auto_report_on_load = true;
      g_seed_auto_report_succeeded = false;
      const std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, i};
      hsa_code_object_reader_t reader{};
      ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                      original.size(), &reader),
                HSA_STATUS_SUCCESS);
      ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(
                    hsa_executable_t{7u + i}, kHostAgent, reader, nullptr, nullptr),
                HSA_STATUS_SUCCESS);
      EXPECT_TRUE(g_seed_auto_report_succeeded);
    }
  }
  const auto log = testing::internal::GetCapturedStderr();
  const auto occurrences = [&](std::string_view needle) {
    size_t count = 0, position = 0;
    while ((position = log.find(needle, position)) != std::string::npos) {
      ++count;
      position += needle.size();
    }
    return count;
  };
  EXPECT_EQ(occurrences("effective_banks_min=1 effective_banks_max=1"), 3u) << log;
  EXPECT_EQ(occurrences("conflicts=1 "), 3u) << log;
  EXPECT_EQ(occurrences("ConSan conflict reader="), 2u) << log;
  EXPECT_EQ(occurrences("conflict_examples=0 conflict_pairs_without_example=1"), 1u) << log;
}

TEST(HsaHooksUnitTest, ConSanResolvesIndependentSelectorsAndRejectsAmbiguity) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar legacy_stride("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE", nullptr);
  ScopedEnvVar legacy_offset("RJ_CONSAN_RUNTIME_SAMPLE_OFFSET", nullptr);
  ScopedEnvVar workgroup_stride("RJ_CONSAN_WORKGROUP_SAMPLE_STRIDE", "1");
  ScopedEnvVar workgroup_offset("RJ_CONSAN_WORKGROUP_SAMPLE_OFFSET", nullptr);
  ScopedEnvVar cell_stride("RJ_CONSAN_CELL_SAMPLE_STRIDE", "4");
  ScopedEnvVar cell_offset("RJ_CONSAN_CELL_SAMPLE_OFFSET", "3");
  ScopedEnvVar banks("RJ_CONSAN_WATCHPOINT_BANKS", "2");
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  const auto install = [] {
    reset_code_object_observations();
    testing::internal::CaptureStderr();
    bool installed = false;
    {
      FakeApiTable api;
      InstalledDbiHook hook(api);
      installed = hook.installed();
    }
    return std::pair{installed, testing::internal::GetCapturedStderr()};
  };
  const auto [installed, log] = install();
  ASSERT_TRUE(installed) << log;
  EXPECT_NE(log.find("workgroup_stride=1 workgroup_offset=0 cell_stride=4 cell_offset=3"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("selection=independent requested_banks=2"), std::string::npos) << log;
  {
    ScopedEnvVar ambiguous("RJ_CONSAN_RUNTIME_SAMPLE_OFFSET", "0");
    EXPECT_FALSE(install().first);
  }
  {
    ScopedEnvVar unspecified_stride("RJ_CONSAN_CELL_SAMPLE_STRIDE", nullptr);
    const auto [default_installed, default_log] = install();
    ASSERT_TRUE(default_installed) << default_log;
    EXPECT_NE(default_log.find("cell_stride=256 cell_offset=3"), std::string::npos) << default_log;
  }
  {
    ScopedEnvVar wrong_mode("RJ_CONSAN_MODE", "supercollider");
    EXPECT_FALSE(install().first);
  }
}

TEST(HsaHooksUnitTest, ConSanUniformLdsStoresRequireExplicitOptIn) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  for (const char *mode : {"default"}) {
    ScopedEnvVar selected_mode("RJ_CONSAN_MODE", mode);
    for (const char *value : {static_cast<const char *>(nullptr), "0", "1", "invalid"}) {
      ScopedEnvVar policy("RJ_CONSAN_ALLOW_PROVABLY_SAME_VALUE_WRITE_RACES", value);
      reset_code_object_observations();
      testing::internal::CaptureStderr();
      bool installed;
      {
        FakeApiTable api;
        InstalledDbiHook hook(api);
        installed = hook.installed();
      }
      const std::string log = testing::internal::GetCapturedStderr();
      const bool invalid = value && std::string_view(value) == "invalid";
      EXPECT_EQ(installed, !invalid) << log;
      if (!invalid) {
        const bool enabled = value && std::string_view(value) == "1";
        EXPECT_NE(
            log.find(enabled ? "allow_uniform_lds_stores=true" : "allow_uniform_lds_stores=false"),
            std::string::npos)
            << log;
      }
    }
  }
}

TEST(HsaHooksUnitTest, ConSanPresetsResolveAndAllowExplicitOverrides) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", nullptr);
  ScopedEnvVar legacy_stride("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE", nullptr);
  ScopedEnvVar legacy_offset("RJ_CONSAN_RUNTIME_SAMPLE_OFFSET", nullptr);
  ScopedEnvVar workgroup_stride("RJ_CONSAN_WORKGROUP_SAMPLE_STRIDE", nullptr);
  ScopedEnvVar workgroup_offset("RJ_CONSAN_WORKGROUP_SAMPLE_OFFSET", nullptr);
  ScopedEnvVar cell_stride("RJ_CONSAN_CELL_SAMPLE_STRIDE", nullptr);
  ScopedEnvVar cell_offset("RJ_CONSAN_CELL_SAMPLE_OFFSET", nullptr);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  const auto install = [] {
    reset_code_object_observations();
    testing::internal::CaptureStderr();
    bool installed;
    {
      FakeApiTable api;
      InstalledDbiHook hook(api);
      installed = hook.installed();
    }
    return std::pair{installed, testing::internal::GetCapturedStderr()};
  };
  for (const auto &[preset, selection] : std::array{
           std::pair{static_cast<const char *>(nullptr),
                     "workgroup_stride=256 workgroup_offset=0 cell_stride=256 cell_offset=0"},
           std::pair{"", "workgroup_stride=256 workgroup_offset=0 cell_stride=256 cell_offset=0"},
           std::pair{"default",
                     "workgroup_stride=256 workgroup_offset=0 cell_stride=256 cell_offset=0"},
           std::pair{"low",
                     "workgroup_stride=1024 workgroup_offset=0 cell_stride=1024 cell_offset=0"},
           std::pair{"high", "workgroup_stride=16 workgroup_offset=0 cell_stride=16 cell_offset=0"},
           std::pair{"higher", "workgroup_stride=1 workgroup_offset=0 cell_stride=4 cell_offset=0"},
           std::pair{"HiGhEr", "workgroup_stride=1 workgroup_offset=0 cell_stride=4 cell_offset=0"},
           std::pair{"max", "workgroup_stride=1 workgroup_offset=0 cell_stride=1 cell_offset=0"}}) {
    ScopedEnvVar selected("RJ_CONSAN_PRESET", preset);
    const auto [ok, log] = install();
    ASSERT_TRUE(ok) << log;
    EXPECT_NE(log.find(selection), std::string::npos) << log;
    EXPECT_NE(log.find("requested_banks=auto"), std::string::npos) << log;
    EXPECT_NE(log.find("epoch_analysis=every"), std::string::npos) << log;
  }
  ScopedEnvVar high("RJ_CONSAN_PRESET", "high");
  {
    ScopedEnvVar explicit_cell("RJ_CONSAN_CELL_SAMPLE_STRIDE", "8");
    ScopedEnvVar explicit_offset("RJ_CONSAN_CELL_SAMPLE_OFFSET", "7");
    const auto [ok, log] = install();
    ASSERT_TRUE(ok) << log;
    EXPECT_NE(log.find("workgroup_stride=16 workgroup_offset=0 cell_stride=8 cell_offset=7"),
              std::string::npos)
        << log;
    ScopedEnvVar mixed("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE", "2");
    EXPECT_FALSE(install().first);
  }
  {
    ScopedEnvVar explicit_legacy("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE", "2");
    const auto [ok, log] = install();
    ASSERT_TRUE(ok) << log;
    EXPECT_NE(log.find("workgroup_stride=2 workgroup_offset=0 cell_stride=2 cell_offset=0"),
              std::string::npos)
        << log;
  }
  {
    ScopedEnvVar bad_offset("RJ_CONSAN_CELL_SAMPLE_OFFSET", "16");
    EXPECT_FALSE(install().first);
  }
  {
    ScopedEnvVar invalid("RJ_CONSAN_PRESET", "highest");
    const auto [ok, log] = install();
    EXPECT_FALSE(ok);
    EXPECT_NE(log.find("expected low|default|high|higher|max"), std::string::npos) << log;
  }
  for (const char *other : {"supercollider"}) {
    ScopedEnvVar other_mode("RJ_CONSAN_MODE", other);
    EXPECT_FALSE(install().first);
  }
}

TEST(HsaHooksUnitTest, ConSanRejectsUnboundedExampleLimits) {
  for (const auto &[name, value] : std::array{std::pair{"RJ_CONSAN_CONFLICT_LIMIT", "1025"},
                                              std::pair{"RJ_CONSAN_TOTAL_CONFLICT_LIMIT", "65537"},
                                              std::pair{"RJ_CONSAN_CONFLICT_LIMIT", "-1"}}) {
    ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
    ScopedEnvVar limit(name, value);
    reset_code_object_observations();
    FakeApiTable api;
    InstalledDbiHook hook(api);
    EXPECT_FALSE(hook.installed()) << name;
  }
}

TEST(HsaHooksUnitTest, AutoEmptyReportSkipsCapacityScanAndSurfacesMalformedStaticMapping) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "default");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "1");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "4194304");
  ScopedEnvVar runtime_stride("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE", "1");

  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_transform_result(/*malformed_mapping=*/true);

  testing::internal::CaptureStderr();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
  }
  const std::string log = testing::internal::GetCapturedStderr();

  const size_t mapping = log.find("ConSan diagnostic map reader=101 entries=1 mappings=0 ");
  ASSERT_NE(mapping, std::string::npos) << log;
  EXPECT_NE(log.find("malformed=true", mapping), std::string::npos) << log;
  EXPECT_NE(log.find("static_mapping_malformed=1"), std::string::npos) << log;
  EXPECT_NE(log.find("watchpoint_slots_examined=0"), std::string::npos) << log;
}

void reset_queue_fakes() {
  g_fake_queue_packets = {};
  g_fake_queue = {};
  g_fake_batch_queue_packets = {};
  g_fake_batch_queue = {};
  g_fake_amd_queue_create_calls = 0;
  g_last_queue_create_agent = {};
  g_last_destroyed_queue = nullptr;
  g_fake_allocations.clear();
  g_fake_allocation_pools.clear();
  g_fake_allocation_sizes.clear();
  g_fake_freed_allocations.clear();
  g_fake_signal_store_relaxed_calls = 0;
  g_fake_signal_store_screlease_calls = 0;
  g_last_signal_store_signal = {};
  g_last_signal_store_value = 0;
  g_next_fake_signal_handle = 10000;
  g_fake_created_signals.clear();
  g_fake_destroyed_signals.clear();
  g_fake_signal_values.clear();
  g_last_intercept_registered_queue = nullptr;
  g_fake_intercept_handler = nullptr;
  g_fake_intercept_user_data = nullptr;
  g_last_intercept_written_packets.clear();
  g_fake_symbol_kernel_object = 0;
  g_fake_symbol_group_segment_size = 0;
  g_fake_symbol_private_segment_size = 0;
  g_fake_symbol_name = "oversized_kernel.kd";
  g_fake_load_agent_calls = 0;
  g_last_load_agent = {};
  g_last_load_reader = {};
}

void write_bytes(std::vector<uint8_t> &image, size_t offset, const void *src, size_t size) {
  if (image.size() < offset + size)
    image.resize(offset + size);
  std::memcpy(image.data() + offset, src, size);
}

template <typename T>
void write_struct(std::vector<uint8_t> &image, size_t offset, const T &value) {
  write_bytes(image, offset, &value, sizeof(T));
}

size_t align_up(size_t value, size_t alignment) {
  if (alignment <= 1)
    return value;
  const size_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

rocjitsu::Elf64_Ehdr make_amdgpu_elf_header(uint32_t mach) {
  rocjitsu::Elf64_Ehdr header{};
  std::memcpy(header.e_ident, rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE);
  header.e_ident[rocjitsu::EI_CLASS] = rocjitsu::ELFCLASS64;
  header.e_ident[rocjitsu::EI_DATA] = 1;
  header.e_ident[rocjitsu::EI_VERSION] = 1;
  header.e_ident[rocjitsu::EI_OSABI] = rocjitsu::ELFOSABI_AMDGPU_HSA;
  header.e_ident[rocjitsu::EI_ABIVERSION] = rocjitsu::ELFABIVERSION_AMDGPU_HSA_V5;
  header.e_type = rocjitsu::ET_DYN;
  header.e_machine = rocjitsu::EM_AMDGPU;
  header.e_version = 1;
  header.e_flags = mach;
  header.e_ehsize = sizeof(rocjitsu::Elf64_Ehdr);
  return header;
}

struct VirtualLdsMetadataForTest {
  std::string kernel_name;
  uint64_t normal_descriptor_vaddr = 0;
  uint64_t virtual_descriptor_vaddr = 0;
  uint32_t static_lds_bytes = 0;
  uint32_t normal_private_segment_size = 0;
  uint32_t virtual_private_segment_size = 0;
  uint32_t kernarg_size = 0;
  uint32_t backing_pointer_kernarg_offset = 0;
  uint16_t virtual_lds_base_sgpr = 0;
  uint16_t flags = 0;
};

std::vector<uint8_t>
make_translated_metadata_elf(uint32_t mach, const std::vector<VirtualLdsMetadataForTest> &records) {
  std::vector<rocjitsu::SidecarVariantMetadata> sidecars;
  std::vector<rocjitsu::KernargExtensionMetadata> kernarg_extensions;
  std::vector<rocjitsu::VirtualLdsKernelMetadata> virtual_lds;
  for (const VirtualLdsMetadataForTest &record : records) {
    sidecars.push_back({
        .kernel_name = record.kernel_name,
        .variant_name = std::string(rocjitsu::kVirtualLdsSidecarVariantName),
        .normal_descriptor_vaddr = record.normal_descriptor_vaddr,
        .variant_descriptor_vaddr = record.virtual_descriptor_vaddr,
    });
    kernarg_extensions.push_back({
        .kernel_name = record.kernel_name,
        .variant_name = std::string(rocjitsu::kVirtualLdsSidecarVariantName),
        .original_kernarg_size = record.kernarg_size,
        .payloads = {{
            .size = rocjitsu::kVirtualLdsRuntimeStateBytes,
            .alignment = alignof(uint64_t),
            .name = std::string(rocjitsu::kVirtualLdsRuntimeStatePayloadName),
        }},
    });
    const rocjitsu::KernargExtensionPayloadLayout payload{
        .size = rocjitsu::kVirtualLdsRuntimeStateBytes,
        .alignment = alignof(uint64_t),
    };
    const auto layout =
        rocjitsu::make_kernarg_extension_layout(record.kernarg_size, std::span{&payload, 1});
    EXPECT_TRUE(layout.has_value());
    if (layout) {
      EXPECT_EQ(layout->payload_offsets.front(), record.backing_pointer_kernarg_offset);
    }
    virtual_lds.push_back({
        .kernel_name = record.kernel_name,
        .sidecar_variant_name = std::string(rocjitsu::kVirtualLdsSidecarVariantName),
        .static_lds_bytes = record.static_lds_bytes,
        .normal_private_segment_size = record.normal_private_segment_size,
        .virtual_private_segment_size = record.virtual_private_segment_size,
        .virtual_lds_base_sgpr = record.virtual_lds_base_sgpr,
        .flags = record.flags,
    });
  }

  const std::array metadata = {
      rocjitsu::serialize_sidecar_metadata(sidecars),
      rocjitsu::serialize_kernarg_extension_metadata(kernarg_extensions),
      rocjitsu::serialize_virtual_lds_metadata(virtual_lds),
  };
  const std::array<std::string_view, 3> metadata_names = {
      rocjitsu::kSidecarMetadataSectionName,
      rocjitsu::kKernargExtensionMetadataSectionName,
      rocjitsu::kVirtualLdsMetadataSectionName,
  };

  std::string shstr(1, '\0');
  std::array<uint32_t, 3> metadata_name_offsets{};
  for (size_t i = 0; i < metadata_names.size(); ++i) {
    metadata_name_offsets[i] = static_cast<uint32_t>(shstr.size());
    shstr.append(metadata_names[i]);
    shstr.push_back('\0');
  }
  const uint32_t shstrtab_name = static_cast<uint32_t>(shstr.size());
  shstr.append(".shstrtab");
  shstr.push_back('\0');

  auto header = make_amdgpu_elf_header(mach);
  std::vector<uint8_t> image(sizeof(header));
  std::array<size_t, 3> metadata_offsets{};
  for (size_t i = 0; i < metadata.size(); ++i) {
    metadata_offsets[i] = image.size();
    write_bytes(image, metadata_offsets[i], metadata[i].data(), metadata[i].size());
  }
  const size_t shstrtab_offset = image.size();
  write_bytes(image, shstrtab_offset, shstr.data(), shstr.size());

  const size_t section_header_offset = align_up(image.size(), alignof(rocjitsu::Elf64_Shdr));
  image.resize(section_header_offset);

  std::array<rocjitsu::Elf64_Shdr, 5> sections{};
  for (size_t i = 0; i < metadata.size(); ++i) {
    sections[i + 1].sh_name = metadata_name_offsets[i];
    sections[i + 1].sh_type = rocjitsu::SHT_PROGBITS;
    sections[i + 1].sh_offset = metadata_offsets[i];
    sections[i + 1].sh_size = metadata[i].size();
  }
  sections[4].sh_name = shstrtab_name;
  sections[4].sh_type = rocjitsu::SHT_STRTAB;
  sections[4].sh_offset = shstrtab_offset;
  sections[4].sh_size = shstr.size();
  write_bytes(image, section_header_offset, sections.data(), sizeof(sections));

  header.e_shoff = section_header_offset;
  header.e_shentsize = sizeof(rocjitsu::Elf64_Shdr);
  header.e_shnum = sections.size();
  header.e_shstrndx = 4;
  write_struct(image, 0, header);
  return image;
}

struct VirtualLdsRegistrationForTest {};

VirtualLdsRegistrationForTest register_virtual_lds_kernel_for_test(
    FakeApiTable &api, const rocr::llvm::amdhsa::kernel_descriptor_t &normal_descriptor,
    const rocr::llvm::amdhsa::kernel_descriptor_t &virtual_descriptor, uint32_t static_lds_bytes,
    uint32_t kernarg_size = 0,
    uint32_t backing_pointer_kernarg_offset = kVirtualLdsWrapperStateOffsetForTest,
    uint16_t flags = kVirtualLdsWrapperFlagsForTest, bool resolve_symbol_by_name = true,
    bool request_loaded_code_object = true) {
  const auto normal_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  const auto virtual_object = reinterpret_cast<uintptr_t>(&virtual_descriptor);
  const int64_t descriptor_delta =
      static_cast<int64_t>(virtual_object) - static_cast<int64_t>(normal_object);
  constexpr uint64_t kNormalDescriptorVaddr = 0x100000000ull;
  const uint64_t virtual_descriptor_vaddr =
      static_cast<uint64_t>(static_cast<int64_t>(kNormalDescriptorVaddr) + descriptor_delta);

  g_fake_symbol_kernel_object = normal_object;
  g_fake_symbol_group_segment_size = normal_descriptor.group_segment_fixed_size;
  g_fake_symbol_private_segment_size = normal_descriptor.private_segment_fixed_size;

  const std::vector<VirtualLdsMetadataForTest> metadata = {{
      .kernel_name = "oversized_kernel",
      .normal_descriptor_vaddr = kNormalDescriptorVaddr,
      .virtual_descriptor_vaddr = virtual_descriptor_vaddr,
      .static_lds_bytes = static_lds_bytes,
      .normal_private_segment_size = normal_descriptor.private_segment_fixed_size,
      .virtual_private_segment_size = virtual_descriptor.private_segment_fixed_size,
      .kernarg_size = kernarg_size,
      .backing_pointer_kernarg_offset = backing_pointer_kernarg_offset,
      .virtual_lds_base_sgpr = 8,
      .flags = flags,
  }};
  // Load a target-matching object with the DBT metadata section prebuilt. This
  // exercises the same hook registry path as translated code objects without
  // depending on the production translator in this unit-test helper.
  const auto code_object =
      make_translated_metadata_elf(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201, metadata);

  hsa_code_object_reader_t reader{};
  EXPECT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(code_object.data(),
                                                                  code_object.size(), &reader),
            HSA_STATUS_SUCCESS);

  hsa_loaded_code_object_t loaded{};
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(
                kFakeExecutable, kGuestAgent, reader, nullptr,
                request_loaded_code_object ? &loaded : nullptr),
            HSA_STATUS_SUCCESS);

  if (!resolve_symbol_by_name)
    return {};

  hsa_executable_symbol_t symbol{};
  EXPECT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
            HSA_STATUS_SUCCESS);

  uint64_t kernel_object = 0;
  EXPECT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(kernel_object, normal_object);

  return {};
}

struct IteratedSymbolForTest {
  hsa_agent_t agent{};
  hsa_executable_symbol_t symbol{};
};

struct IteratedLegacySymbolForTest {
  hsa_executable_symbol_t symbol{};
};

hsa_status_t HSA_API capture_iterated_symbol_for_test(hsa_executable_t, hsa_agent_t agent,
                                                      hsa_executable_symbol_t symbol, void *data) {
  auto *captured = static_cast<IteratedSymbolForTest *>(data);
  captured->agent = agent;
  captured->symbol = symbol;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API capture_iterated_legacy_symbol_for_test(hsa_executable_t,
                                                             hsa_executable_symbol_t symbol,
                                                             void *data) {
  static_cast<IteratedLegacySymbolForTest *>(data)->symbol = symbol;
  return HSA_STATUS_SUCCESS;
}

TEST(HsaHooksUnitTest, IterateAgentsDropsGuestOwnSlotWhenGuestAppearsFirst) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_iterate_agents_fn, fake_iterate_agents);

  std::vector<uint64_t> seen;
  hsa_status_t status = api.core.hsa_iterate_agents_fn(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        static_cast<std::vector<uint64_t> *>(data)->push_back(agent.handle);
        return HSA_STATUS_SUCCESS;
      },
      &seen);

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(seen, std::vector<uint64_t>{kGuestAgent.handle});
}

TEST(HsaHooksUnitTest, IterateAgentsDropsGuestOwnSlotWhenHostAppearsFirst) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  api.core.hsa_iterate_agents_fn = fake_iterate_agents_host_first;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_iterate_agents_fn, fake_iterate_agents_host_first);

  std::vector<uint64_t> seen;
  hsa_status_t status = api.core.hsa_iterate_agents_fn(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        static_cast<std::vector<uint64_t> *>(data)->push_back(agent.handle);
        return HSA_STATUS_SUCCESS;
      },
      &seen);

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(seen, std::vector<uint64_t>{kGuestAgent.handle});
}

// The configured host gpu_id has no topology node here, so the hook cannot tell
// which physical agent the guest should execute on. Every GPU must be suppressed
// rather than falling through to the raw agent list: publishing the physical host
// would let an application using the default device run untranslated on it.
// GuestKfd already rejects the same unresolved host with ENODEV. The status stays
// SUCCESS so a GPU client sees an empty device list instead of the opaque error
// HIP/CLR reports as a fatal ROCR init failure.
TEST(HsaHooksUnitTest, MissingResolvedHostSuppressesGpuAgents) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  InstalledHook hook(api, /*create_host_topology_node=*/false);
  ASSERT_TRUE(hook.installed());
  // Without this anchor the expectations below are indistinguishable from an
  // unpatched table forwarding straight to a fake that emits only GPU agents.
  ASSERT_NE(api.core.hsa_iterate_agents_fn, fake_iterate_agents);

  std::vector<uint64_t> seen;
  hsa_status_t status = api.core.hsa_iterate_agents_fn(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        static_cast<std::vector<uint64_t> *>(data)->push_back(agent.handle);
        return HSA_STATUS_SUCCESS;
      },
      &seen);

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  EXPECT_TRUE(seen.empty());
}

// Suppression is GPU-only. A CPU agent cannot run a guest kernel, and callers
// that need one -- host fine-grained pool lookup, hsa_amd_memory_lock, host-side
// copies -- must keep working while the mapping is unresolved.
TEST(HsaHooksUnitTest, MissingResolvedHostStillPublishesCpuAgents) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  api.core.hsa_iterate_agents_fn = fake_iterate_agents_with_cpu;
  InstalledHook hook(api, /*create_host_topology_node=*/false);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_iterate_agents_fn, fake_iterate_agents_with_cpu);

  std::vector<uint64_t> seen;
  EXPECT_EQ(api.core.hsa_iterate_agents_fn(collect_agent_handles, &seen), HSA_STATUS_SUCCESS);
  EXPECT_EQ(seen, std::vector<uint64_t>{kCpuAgent.handle});
}

// The GPU-only suppression must not have disturbed the success-path shadow
// callback: CPU agents still pass through untouched, and the guest is still
// emitted in the selected host's ordinal slot.
TEST(HsaHooksUnitTest, MappedEnumerationStillPublishesCpuAgents) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  api.core.hsa_iterate_agents_fn = fake_iterate_agents_with_cpu;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_iterate_agents_fn, fake_iterate_agents_with_cpu);

  std::vector<uint64_t> seen;
  EXPECT_EQ(api.core.hsa_iterate_agents_fn(collect_agent_handles, &seen), HSA_STATUS_SUCCESS);
  EXPECT_EQ(seen, (std::vector<uint64_t>{kCpuAgent.handle, kGuestAgent.handle}));
}

// An unopenable topology root is an I/O failure, not proof the configured host
// gpu_id is absent. Latching it would poison enumeration for the process: every
// later hsa_iterate_agents would keep suppressing every GPU, even once the
// redirected topology tree became readable.
TEST(HsaHooksUnitTest, AgentMapperRetriesAfterUnreadableTopologyRoot) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_iterate_agents_fn, fake_iterate_agents);

  // Redirect at a root that does not exist yet, so opendir fails outright.
  rocjitsu::test::ScopedTempDirectory late("rocjitsu-late-topology-");
  const std::filesystem::path nodes = std::filesystem::path(late.path()) / "nodes";
  rj_hsa_dbt_set_topology_nodes_root_for_test(nodes.c_str());

  std::vector<uint64_t> first;
  EXPECT_EQ(api.core.hsa_iterate_agents_fn(collect_agent_handles, &first), HSA_STATUS_SUCCESS);
  // The fake publishes only GPU agents, so a suppressed list is an empty list.
  EXPECT_TRUE(first.empty());

  const std::filesystem::path host_node = nodes / std::to_string(kHostNodeId);
  std::filesystem::create_directories(host_node);
  std::ofstream(host_node / "gpu_id") << kResolvedHostGpuId << '\n';

  std::vector<uint64_t> second;
  EXPECT_EQ(api.core.hsa_iterate_agents_fn(collect_agent_handles, &second), HSA_STATUS_SUCCESS);
  EXPECT_EQ(second, std::vector<uint64_t>{kGuestAgent.handle});
}

// A readable topology root whose node directory has no readable gpu_id is the other
// half of the same rule: the skipped node may be the one being searched for, so the
// scan cannot prove the configured gpu_id is absent. The verdict must stay retryable
// rather than latching as a permanent "absent" that suppresses every GPU for good.
TEST(HsaHooksUnitTest, AgentMapperRetriesAfterUnreadableTopologyNode) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_iterate_agents_fn, fake_iterate_agents);

  // Unlike the root test above, this root opens fine; it is the per-node gpu_id read
  // that fails, because the node directory exists with no gpu_id file in it.
  rocjitsu::test::ScopedTempDirectory partial("rocjitsu-partial-topology-");
  const std::filesystem::path nodes = std::filesystem::path(partial.path()) / "nodes";
  const std::filesystem::path host_node = nodes / std::to_string(kHostNodeId);
  std::filesystem::create_directories(host_node);
  ASSERT_FALSE(std::filesystem::exists(host_node / "gpu_id"));
  rj_hsa_dbt_set_topology_nodes_root_for_test(nodes.c_str());

  std::vector<uint64_t> first;
  EXPECT_EQ(api.core.hsa_iterate_agents_fn(collect_agent_handles, &first), HSA_STATUS_SUCCESS);
  EXPECT_TRUE(first.empty());

  std::ofstream(host_node / "gpu_id") << kResolvedHostGpuId << '\n';

  std::vector<uint64_t> second;
  EXPECT_EQ(api.core.hsa_iterate_agents_fn(collect_agent_handles, &second), HSA_STATUS_SUCCESS);
  EXPECT_EQ(second, std::vector<uint64_t>{kGuestAgent.handle});
}

// The retry must be budgeted. An application that polls the device list while
// ROCR stays broken would otherwise pay a sysfs topology walk plus a full agent
// iteration on every query.
TEST(HsaHooksUnitTest, AgentMapperStopsRediscoveringAfterTransientRetryBudget) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_iterate_agents_fn, fake_iterate_agents);

  constexpr int kEnumerationCalls = 64;
  g_fail_agent_iteration = true;
  g_fake_iterate_agents_calls = 0;
  for (int i = 0; i < kEnumerationCalls; ++i) {
    std::vector<uint64_t> seen;
    EXPECT_EQ(api.core.hsa_iterate_agents_fn(collect_agent_handles, &seen), HSA_STATUS_ERROR);
  }
  // Each refused enumeration still forwards to the original iterator once, to
  // filter GPU agents out of the published list, so kEnumerationCalls of these
  // are pass-throughs rather than rediscovery. A range rather than the exact
  // bound, so the test does not pin kMaxTransientAttempts: at least one retry
  // happened on top of the first attempt, and the mapper latched well before
  // rediscovering on every call.
  EXPECT_GE(g_fake_iterate_agents_calls, kEnumerationCalls + 2);
  EXPECT_LT(g_fake_iterate_agents_calls, kEnumerationCalls * 2);
  g_fail_agent_iteration = false;
}

// Only the enumeration entry point opts into rediscovery. Everything reached
// per copy, per dispatch, or per allocation stays on the cached verdict.
TEST(HsaHooksUnitTest, HotPathAgentMappingDoesNotRerunDiscovery) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_memory_pool_allocate_fn, fake_amd_memory_pool_allocate);

  g_fail_agent_iteration = true;
  g_fake_iterate_agents_calls = 0;
  for (int i = 0; i < 16; ++i) {
    void *ptr = nullptr;
    EXPECT_EQ(api.amd.hsa_amd_memory_pool_allocate_fn(kGuestPool, 4096, 0, &ptr),
              HSA_STATUS_SUCCESS);
  }
  // hsa_amd_memory_pool_allocate reaches AgentMapper through
  // MemoryPoolMapper::ensure_discovered, which retries pool discovery on every
  // call; none of those retries may rerun agent discovery.
  EXPECT_EQ(g_fake_iterate_agents_calls, 1);
  g_fail_agent_iteration = false;
}

// Enumeration publishing no GPU is not by itself fail-closed: a handle cached
// before discovery failed, or one held by a co-loaded tool, still reaches the
// execution entry points. Each of the five that can put code on -- or dispatch
// to -- real silicon must refuse while the mapping is unusable.
TEST(HsaHooksUnitTest, UnresolvedMappingRefusesQueueCreate) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api, /*create_host_topology_node=*/false);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_queue_create_fn, fake_queue_create);

  hsa_queue_t *queue = nullptr;
  EXPECT_EQ(api.core.hsa_queue_create_fn(kHostAgent, 4, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_ERROR_INVALID_AGENT);
  EXPECT_EQ(queue, nullptr);
  EXPECT_EQ(g_last_queue_create_agent.handle, 0u);
}

TEST(HsaHooksUnitTest, UnresolvedMappingRefusesInterceptQueueCreate) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledHook hook(api, /*create_host_topology_node=*/false);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_queue_intercept_create_fn, fake_amd_queue_intercept_create);

  hsa_queue_t *queue = nullptr;
  EXPECT_EQ(api.amd.hsa_amd_queue_intercept_create_fn(kHostAgent, 4, HSA_QUEUE_TYPE_MULTI, nullptr,
                                                      nullptr, 0, 0, &queue),
            HSA_STATUS_ERROR_INVALID_AGENT);
  EXPECT_EQ(queue, nullptr);
}

// The regression test for the load path specifically: is_guest() is false
// whenever the mapping failed, so without the guard this load would take the
// "not the guest, forward verbatim" branch and put an untranslated guest code
// object on the raw host.
TEST(HsaHooksUnitTest, UnresolvedMappingRefusesAgentCodeObjectLoad) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api, /*create_host_topology_node=*/false);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_executable_load_agent_code_object_fn,
            fake_executable_load_agent_code_object);

  std::vector<uint8_t> image(sizeof(rocjitsu::Elf64_Ehdr), 0);
  write_struct(image, 0, make_amdgpu_elf_header(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX950));
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(image.data(), image.size(), &reader),
      HSA_STATUS_SUCCESS);

  hsa_loaded_code_object_t loaded{};
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kHostAgent, reader,
                                                              nullptr, &loaded),
            HSA_STATUS_ERROR_INVALID_AGENT);
  EXPECT_EQ(g_fake_load_agent_calls, 0);
}

TEST(HsaHooksUnitTest, UnresolvedMappingRefusesAgentPreload) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  InstalledHook hook(api, /*create_host_topology_node=*/false);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_agent_preload_fn, fake_amd_agent_preload);

  g_last_agent_preload_agent = {};
  EXPECT_EQ(api.amd.hsa_amd_agent_preload_fn(kHostAgent, 0), HSA_STATUS_ERROR_INVALID_AGENT);
  EXPECT_EQ(g_last_agent_preload_agent.handle, 0u);
}

TEST(HsaHooksUnitTest, UnresolvedMappingRefusesAsyncScratchLimit) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  InstalledHook hook(api, /*create_host_topology_node=*/false);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_agent_set_async_scratch_limit_fn,
            fake_amd_agent_set_async_scratch_limit);

  g_last_async_scratch_limit_agent = {};
  EXPECT_EQ(api.amd.hsa_amd_agent_set_async_scratch_limit_fn(kHostAgent, 4096),
            HSA_STATUS_ERROR_INVALID_AGENT);
  EXPECT_EQ(g_last_async_scratch_limit_agent.handle, 0u);
}

// The guard against over-applying the refusal. Query and allocation hooks
// legitimately take CPU agents and cannot execute a kernel; blanket-guarding
// them would undo the CPU pass-through that enumeration deliberately preserves.
TEST(HsaHooksUnitTest, UnresolvedMappingStillAllowsCpuPoolIteration) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  InstalledHook hook(api, /*create_host_topology_node=*/false);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_agent_iterate_memory_pools_fn, fake_amd_agent_iterate_memory_pools);

  std::vector<uint64_t> pools;
  EXPECT_EQ(api.amd.hsa_amd_agent_iterate_memory_pools_fn(
                kCpuAgent,
                [](hsa_amd_memory_pool_t pool, void *data) -> hsa_status_t {
                  static_cast<std::vector<uint64_t> *>(data)->push_back(pool.handle);
                  return HSA_STATUS_SUCCESS;
                },
                &pools),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(pools, std::vector<uint64_t>{kCpuFineGrainedPool.handle});
}

// The refusal is conditional on the mapping being unusable, not on the agent
// being unrelated to DBT: a resolved mapping must leave other GPUs alone.
TEST(HsaHooksUnitTest, ResolvedMappingStillAllowsUnrelatedAgentQueueCreate) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_queue_create_fn, fake_queue_create);

  hsa_queue_t *queue = nullptr;
  EXPECT_EQ(api.core.hsa_queue_create_fn(kUnrelatedAgent, 4, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr,
                                         0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(g_last_queue_create_agent.handle, kUnrelatedAgent.handle);
}

// Only the host predicate carries the node-id constraint, so when both agents
// advertise both ISAs -- the shape of a same-ISA config -- the guest predicate
// must not be allowed to claim the physical host. Enumeration still resolves a
// distinct pair, whichever side ROCR emits first.
TEST(HsaHooksUnitTest, SameIsaGuestAndHostSelectDistinctAgentsWhenGuestAppearsFirst) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  api.core.hsa_agent_iterate_isas_fn = fake_agent_iterate_isas_overlapping;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_iterate_agents_fn, fake_iterate_agents);

  std::vector<uint64_t> seen;
  EXPECT_EQ(api.core.hsa_iterate_agents_fn(collect_agent_handles, &seen), HSA_STATUS_SUCCESS);
  EXPECT_EQ(seen, std::vector<uint64_t>{kGuestAgent.handle});
}

TEST(HsaHooksUnitTest, SameIsaGuestAndHostSelectDistinctAgentsWhenHostAppearsFirst) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  api.core.hsa_agent_iterate_isas_fn = fake_agent_iterate_isas_overlapping;
  api.core.hsa_iterate_agents_fn = fake_iterate_agents_host_first;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_iterate_agents_fn, fake_iterate_agents_host_first);

  std::vector<uint64_t> seen;
  EXPECT_EQ(api.core.hsa_iterate_agents_fn(collect_agent_handles, &seen), HSA_STATUS_SUCCESS);
  EXPECT_EQ(seen, std::vector<uint64_t>{kGuestAgent.handle});
}

// One agent satisfies both roles and no distinct guest exists. The mapping is a
// failure, not a pass-through: aliasing guest onto host would translate nothing
// and load guest code objects verbatim on real silicon.
TEST(HsaHooksUnitTest, SameAgentForGuestAndHostFailsClosed) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  api.core.hsa_agent_iterate_isas_fn = fake_agent_iterate_isas_overlapping;
  api.core.hsa_iterate_agents_fn = fake_iterate_agents_host_only;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_iterate_agents_fn, fake_iterate_agents_host_only);

  std::vector<uint64_t> seen;
  EXPECT_EQ(api.core.hsa_iterate_agents_fn(collect_agent_handles, &seen), HSA_STATUS_SUCCESS);
  EXPECT_TRUE(seen.empty());

  hsa_queue_t *queue = nullptr;
  EXPECT_EQ(api.core.hsa_queue_create_fn(kHostAgent, 4, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_ERROR_INVALID_AGENT);
  EXPECT_EQ(queue, nullptr);
}

TEST(HsaHooksUnitTest, BatchCopyMapsScalarSourceAndDestinationAgents) {
  hsa_amd_memory_copy_op_t op{};
  op.type = HSA_AMD_MEMORY_COPY_OP_LINEAR;
  op.src_agent = kGuestAgent;
  op.dst_agent = kGuestAgent;
  op.size = 64;

  expect_batch_copy_forwarding(op, {kHostAgent.handle}, {kHostAgent.handle});
}

TEST(HsaHooksUnitTest, BatchCopyMapsMultiLinearScalarSourceAndDestinationList) {
  int src0 = 0;
  int src1 = 0;
  int dst0 = 0;
  int dst1 = 0;
  void *src_list[] = {&src0, &src1};
  void *dst_list[] = {&dst0, &dst1};
  hsa_agent_t dst_agents[] = {kGuestAgent, kHostAgent};
  size_t sizes[] = {64, 128};

  hsa_amd_memory_copy_op_t op{};
  op.type = HSA_AMD_MEMORY_COPY_OP_LINEAR;
  op.num_entries = 2;
  op.src_list = src_list;
  op.src_agent = kGuestAgent;
  op.dst_agent_list = dst_agents;
  op.dst_list = dst_list;
  op.size_list = sizes;

  expect_batch_copy_forwarding(op, {kHostAgent.handle, kHostAgent.handle},
                               {kHostAgent.handle, kHostAgent.handle});
}

TEST(HsaHooksUnitTest, BatchCopyMapsBroadcastScalarSourceAndDestinationList) {
  int src = 0;
  int dst0 = 0;
  int dst1 = 0;
  void *dst_list[] = {&dst0, &dst1};
  hsa_agent_t dst_agents[] = {kGuestAgent, kHostAgent};

  hsa_amd_memory_copy_op_t op{};
  op.type = HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST;
  op.num_entries = 2;
  op.src = &src;
  op.src_agent = kGuestAgent;
  op.dst_agent_list = dst_agents;
  op.dst_list = dst_list;
  op.size = 64;

  expect_batch_copy_forwarding(op, {kHostAgent.handle, kHostAgent.handle},
                               {kHostAgent.handle, kHostAgent.handle});
}

TEST(HsaHooksUnitTest, BatchCopyMapsMultiIndirectScalarSourceAndDestinationList) {
  int src0 = 0;
  int src1 = 0;
  int dst0 = 0;
  int dst1 = 0;
  void *src_list[] = {&src0, &src1};
  void *dst_list[] = {&dst0, &dst1};
  hsa_agent_t dst_agents[] = {kGuestAgent, kHostAgent};
  size_t sizes[] = {64, 128};

  hsa_amd_memory_copy_op_t op{};
  op.type = HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRCDST;
  op.num_entries = 2;
  op.src_list = src_list;
  op.src_agent = kGuestAgent;
  op.dst_agent_list = dst_agents;
  op.dst_list = dst_list;
  op.size_list = sizes;

  expect_batch_copy_forwarding(op, {kHostAgent.handle, kHostAgent.handle},
                               {kHostAgent.handle, kHostAgent.handle});
}

TEST(HsaHooksUnitTest, PointerInfoReportsGuestIdentityOnce) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_pointer_info_fn, fake_amd_pointer_info);

  hsa_amd_pointer_info_t info{};
  uint32_t accessible_count = 0;
  hsa_agent_t *accessible = nullptr;
  hsa_status_t status = api.amd.hsa_amd_pointer_info_fn(&g_fake_allocation_storage, &info, nullptr,
                                                        &accessible_count, &accessible);

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(info.agentOwner.handle, kGuestAgent.handle);
  ASSERT_NE(accessible, nullptr);
  ASSERT_EQ(accessible_count, 1u);
  EXPECT_EQ(accessible[0].handle, kGuestAgent.handle);
}

TEST(HsaHooksUnitTest, AgentMemoryPoolGetInfoRejectsNullPoolBeforeForwarding) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  g_agent_memory_pool_get_info_calls = 0;
  g_last_agent_memory_pool_agent = {};
  g_last_agent_memory_pool = {};
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_agent_memory_pool_get_info_fn, fake_amd_agent_memory_pool_get_info);

  uint32_t access = 0;
  hsa_status_t status = api.amd.hsa_amd_agent_memory_pool_get_info_fn(
      kGuestAgent, hsa_amd_memory_pool_t{}, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);

  EXPECT_EQ(status, HSA_STATUS_ERROR_INVALID_MEMORY_POOL);
  EXPECT_EQ(g_agent_memory_pool_get_info_calls, 0);
}

TEST(HsaHooksUnitTest, PoolMapperRetriesAfterTransientPoolIterationFailure) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  g_last_allocate_pool = {};
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_memory_pool_allocate_fn, fake_amd_memory_pool_allocate);

  g_fail_guest_pool_iteration_once = true;
  void *ptr = nullptr;
  EXPECT_EQ(api.amd.hsa_amd_memory_pool_allocate_fn(kGuestPool, 4096, 0, &ptr), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_allocate_pool.handle, kGuestPool.handle);

  ptr = nullptr;
  EXPECT_EQ(api.amd.hsa_amd_memory_pool_allocate_fn(kGuestPool, 4096, 0, &ptr), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_allocate_pool.handle, kHostPool.handle);
}

TEST(HsaHooksUnitTest, MemoryLockDeduplicatesAgentsAfterGuestMapping) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  g_last_memory_lock_agents.clear();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_memory_lock_fn, fake_amd_memory_lock);

  hsa_agent_t agents[] = {kGuestAgent, kHostAgent};
  int storage = 0;
  void *agent_ptr = nullptr;
  EXPECT_EQ(api.amd.hsa_amd_memory_lock_fn(&storage, sizeof(storage), agents, 2, &agent_ptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_memory_lock_agents, std::vector<uint64_t>{kHostAgent.handle});
}

TEST(HsaHooksUnitTest, MemoryLockToPoolMapsPoolAndDeduplicatesAgents) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  g_last_memory_lock_to_pool_agents.clear();
  g_last_memory_lock_to_pool_pool = {};
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_memory_lock_to_pool_fn, fake_amd_memory_lock_to_pool);

  hsa_agent_t agents[] = {kGuestAgent, kHostAgent};
  int storage = 0;
  void *agent_ptr = nullptr;
  EXPECT_EQ(api.amd.hsa_amd_memory_lock_to_pool_fn(&storage, sizeof(storage), agents, 2, kGuestPool,
                                                   0, &agent_ptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_memory_lock_to_pool_pool.handle, kHostPool.handle);
  EXPECT_EQ(g_last_memory_lock_to_pool_agents, std::vector<uint64_t>{kHostAgent.handle});
}

TEST(HsaHooksUnitTest, VmemSetAccessDeduplicatesDescriptorsAfterGuestMapping) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  g_last_vmem_access_agents.clear();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_vmem_set_access_fn, fake_amd_vmem_set_access);

  hsa_amd_memory_access_desc_t desc[] = {
      {.permissions = HSA_ACCESS_PERMISSION_RW, .agent_handle = kGuestAgent},
      {.permissions = HSA_ACCESS_PERMISSION_RW, .agent_handle = kHostAgent},
  };
  int storage = 0;
  EXPECT_EQ(api.amd.hsa_amd_vmem_set_access_fn(&storage, sizeof(storage), desc, 2),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_vmem_access_agents, std::vector<uint64_t>{kHostAgent.handle});
}

TEST(HsaHooksUnitTest, PoolAllocateWaitsForAgentDiscoveryPublication) {
  reset_pool_blocker(false);
  reset_agent_blocker(true);
  g_last_allocate_pool = {};
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_iterate_agents_fn, fake_iterate_agents);
  ASSERT_NE(api.amd.hsa_amd_memory_pool_allocate_fn, fake_amd_memory_pool_allocate);

  std::vector<uint64_t> seen;
  hsa_status_t iterate_status = HSA_STATUS_ERROR;
  std::thread iterate_thread([&] {
    iterate_status = api.core.hsa_iterate_agents_fn(
        [](hsa_agent_t agent, void *data) -> hsa_status_t {
          static_cast<std::vector<uint64_t> *>(data)->push_back(agent.handle);
          return HSA_STATUS_SUCCESS;
        },
        &seen);
  });

  bool mapper_entered_agent_iteration = false;
  {
    std::unique_lock lock(g_agent_mutex);
    mapper_entered_agent_iteration = g_agent_cv.wait_for(lock, std::chrono::seconds(1),
                                                         [] { return g_agent_iteration_entered; });
  }
  if (!mapper_entered_agent_iteration) {
    release_agent_blocker();
    iterate_thread.join();
    ADD_FAILURE() << "agent mapper did not enter discovery iteration";
    return;
  }

  std::atomic_bool allocate_done = false;
  hsa_status_t allocate_status = HSA_STATUS_ERROR;
  std::thread allocate_thread([&] {
    void *ptr = nullptr;
    allocate_status = api.amd.hsa_amd_memory_pool_allocate_fn(kGuestPool, 4096, 0, &ptr);
    allocate_done.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(allocate_done.load());

  release_agent_blocker();
  iterate_thread.join();
  allocate_thread.join();

  EXPECT_EQ(iterate_status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(seen, std::vector<uint64_t>{kGuestAgent.handle});
  EXPECT_EQ(allocate_status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_allocate_pool.handle, kHostPool.handle);
  reset_agent_blocker(false);
}

TEST(HsaHooksUnitTest, UninstallDoesNotWaitForPoolMapperDiscoveryLock) {
  reset_pool_blocker(true);
  reset_agent_blocker(false);
  g_last_allocate_pool = {};
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_memory_pool_allocate_fn, fake_amd_memory_pool_allocate);

  hsa_status_t allocate_status = HSA_STATUS_ERROR;
  std::thread mapper_thread([&] {
    void *ptr = nullptr;
    allocate_status = api.amd.hsa_amd_memory_pool_allocate_fn(kGuestPool, 4096, 0, &ptr);
  });

  bool mapper_entered_pool_iteration = false;
  {
    std::unique_lock lock(g_pool_mutex);
    mapper_entered_pool_iteration = g_pool_cv.wait_for(
        lock, std::chrono::seconds(1), [] { return g_guest_pool_iteration_entered; });
  }
  if (!mapper_entered_pool_iteration) {
    release_pool_blocker();
    mapper_thread.join();
    ADD_FAILURE() << "mapper thread did not enter guest pool discovery";
    return;
  }

  bool uninstall_done = false;
  std::thread uninstall_thread([&] {
    OnUnload();
    std::lock_guard lock(g_pool_mutex);
    uninstall_done = true;
    g_pool_cv.notify_all();
  });

  bool completed_without_pool_release = false;
  {
    std::unique_lock lock(g_pool_mutex);
    completed_without_pool_release =
        g_pool_cv.wait_for(lock, std::chrono::seconds(1), [&] { return uninstall_done; });
  }

  release_pool_blocker();
  uninstall_thread.join();
  mapper_thread.join();

  EXPECT_TRUE(completed_without_pool_release);
  EXPECT_EQ(allocate_status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_allocate_pool.handle, kHostPool.handle);
  reset_pool_blocker(false);
}

TEST(HsaHooksUnitTest, GuestShutdownKeepsHookInstalledForProcessLifetime) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  g_fake_shutdown_calls = 0;
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  auto *patched_shutdown = api.core.hsa_shut_down_fn;
  ASSERT_NE(patched_shutdown, fake_shut_down);

  EXPECT_EQ(patched_shutdown(), HSA_STATUS_SUCCESS);

  EXPECT_EQ(g_fake_shutdown_calls, 0);
  EXPECT_EQ(api.core.hsa_shut_down_fn, patched_shutdown);
  EXPECT_NE(api.core.hsa_shut_down_fn, fake_shut_down);
}

TEST(HsaHooksUnitTest, VirtualLdsSymbolInfoReportsNormalDescriptorUntilPacketFallback) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();

  kernel_descriptor_t normal_descriptor{};
  normal_descriptor.group_segment_fixed_size = 108288;
  g_fake_symbol_kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  g_fake_symbol_group_segment_size = normal_descriptor.group_segment_fixed_size;
  g_fake_symbol_private_segment_size = 16;

  const std::vector<VirtualLdsMetadataForTest> metadata = {{
      .kernel_name = "oversized_kernel",
      .normal_descriptor_vaddr = 0x1000,
      .virtual_descriptor_vaddr = 0x2000,
      .static_lds_bytes = normal_descriptor.group_segment_fixed_size,
      .normal_private_segment_size = 16,
      .virtual_private_segment_size = 16,
      .kernarg_size = 0,
      .backing_pointer_kernarg_offset = kVirtualLdsWrapperStateOffsetForTest,
      .virtual_lds_base_sgpr = 8,
      .flags = kVirtualLdsWrapperFlagsForTest,
  }};
  const auto code_object =
      make_translated_metadata_elf(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201, metadata);

  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(code_object.data(),
                                                                  code_object.size(), &reader),
            HSA_STATUS_SUCCESS);

  hsa_loaded_code_object_t loaded{};
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGuestAgent, reader,
                                                              nullptr, &loaded),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(loaded.handle, 77u);

  hsa_executable_symbol_t symbol{};
  ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(symbol.handle, kFakeKernelSymbol.handle);

  uint64_t kernel_object = 0;
  ASSERT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(kernel_object, reinterpret_cast<uintptr_t>(&normal_descriptor));

  uint32_t group_segment_size = 0;
  ASSERT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_segment_size),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(group_segment_size, normal_descriptor.group_segment_fixed_size);
}

TEST(HsaHooksUnitTest, VirtualLdsRegistryKeepsFittingDispatchOnNormalDescriptor) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();

  struct Descriptors {
    kernel_descriptor_t normal{};
    kernel_descriptor_t virtual_sidecar{};
  } descriptors;
  descriptors.normal.group_segment_fixed_size = 32 * 1024;
  descriptors.virtual_sidecar.private_segment_fixed_size = 96;
  ASSERT_GT(reinterpret_cast<uintptr_t>(&descriptors.virtual_sidecar),
            reinterpret_cast<uintptr_t>(&descriptors.normal));

  g_fake_symbol_kernel_object = reinterpret_cast<uintptr_t>(&descriptors.normal);
  g_fake_symbol_group_segment_size = descriptors.normal.group_segment_fixed_size;
  g_fake_symbol_private_segment_size = 12;

  const uint64_t normal_descriptor_vaddr = 0x4000;
  const uint64_t sidecar_descriptor_delta =
      reinterpret_cast<uintptr_t>(&descriptors.virtual_sidecar) -
      reinterpret_cast<uintptr_t>(&descriptors.normal);
  const std::vector<VirtualLdsMetadataForTest> metadata = {{
      .kernel_name = "oversized_kernel",
      .normal_descriptor_vaddr = normal_descriptor_vaddr,
      .virtual_descriptor_vaddr = normal_descriptor_vaddr + sidecar_descriptor_delta,
      .static_lds_bytes = descriptors.normal.group_segment_fixed_size,
      .normal_private_segment_size = 12,
      .virtual_private_segment_size = descriptors.virtual_sidecar.private_segment_fixed_size,
      .kernarg_size = 0,
      .backing_pointer_kernarg_offset = kVirtualLdsWrapperStateOffsetForTest,
      .virtual_lds_base_sgpr = 8,
      .flags = kVirtualLdsWrapperFlagsForTest,
  }};
  const auto code_object =
      make_translated_metadata_elf(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201, metadata);

  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(code_object.data(),
                                                                  code_object.size(), &reader),
            HSA_STATUS_SUCCESS);

  hsa_loaded_code_object_t loaded{};
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGuestAgent, reader,
                                                              nullptr, &loaded),
            HSA_STATUS_SUCCESS);

  hsa_executable_symbol_t symbol{};
  ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
            HSA_STATUS_SUCCESS);
  uint64_t kernel_object = 0;
  ASSERT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernel_object, reinterpret_cast<uintptr_t>(&descriptors.normal));

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&descriptors.normal);
  packet.private_segment_size = 12;
  packet.group_segment_size = 64 * 1024;
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  // This uses the same load-time `.rocjitsu.lds` registry path as real DBT
  // code objects. Even when a virtual sidecar exists, a packet whose total LDS
  // request still fits CDNA3 must remain on the normal descriptor.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&descriptors.normal));
  EXPECT_EQ(packet.group_segment_size, 64u * 1024u);
  EXPECT_EQ(packet.private_segment_size, 12u);
  EXPECT_EQ(packet.reserved2, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsRegistryResolvesKernelObjectFromIteratedSymbol) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000;
  normal_descriptor.private_segment_fixed_size = 40;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size,
      /*kernarg_size=*/0, kVirtualLdsWrapperStateOffsetForTest, kVirtualLdsWrapperFlagsForTest,
      /*resolve_symbol_by_name=*/false);
  (void)registration;

  IteratedSymbolForTest iterated{};
  ASSERT_EQ(api.core.hsa_executable_iterate_agent_symbols_fn(
                kFakeExecutable, kGuestAgent, capture_iterated_symbol_for_test, &iterated),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(iterated.agent.handle, kGuestAgent.handle);
  ASSERT_EQ(iterated.symbol.handle, kFakeKernelSymbol.handle);

  uint64_t kernel_object = 0;
  ASSERT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                iterated.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernel_object, reinterpret_cast<uintptr_t>(&normal_descriptor));

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  // The packet carries 80 bytes of dynamic private memory above the normal
  // descriptor's fixed 40 bytes. The sidecar must retain those 80 bytes.
  packet.private_segment_size = 120;
  packet.group_segment_size = 71024;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  // This path intentionally never calls hsa_executable_get_symbol_by_name().
  // The iterate wrapper must record the symbol name before the client asks for
  // KERNEL_OBJECT, otherwise the packet scanner cannot associate the normal
  // descriptor with the virtual-LDS metadata loaded from `.rocjitsu.lds`.
  ASSERT_EQ(g_fake_allocation_sizes.size(), 2u);
  EXPECT_EQ(g_fake_allocation_pools[0].handle, kHostPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[0], static_cast<size_t>(71024u * 4u));
  EXPECT_EQ(g_fake_allocation_pools[1].handle, kHostKernargPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[1], kVirtualLdsWrapperSizeForTest);
  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&virtual_descriptor));
  EXPECT_EQ(packet.group_segment_size, 0u);
  EXPECT_EQ(packet.private_segment_size, 176u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsRegistryRejectsSidecarDescriptorAsPacketInput) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();

  struct Descriptors {
    kernel_descriptor_t normal{};
    kernel_descriptor_t virtual_sidecar{};
  } descriptors;
  descriptors.normal.group_segment_fixed_size = 108288;
  descriptors.virtual_sidecar.group_segment_fixed_size = 0;
  descriptors.virtual_sidecar.private_segment_fixed_size = 96;
  ASSERT_GT(reinterpret_cast<uintptr_t>(&descriptors.virtual_sidecar),
            reinterpret_cast<uintptr_t>(&descriptors.normal));

  g_fake_symbol_kernel_object = reinterpret_cast<uintptr_t>(&descriptors.normal);
  g_fake_symbol_group_segment_size = descriptors.normal.group_segment_fixed_size;
  g_fake_symbol_private_segment_size = 12;

  const uint64_t normal_descriptor_vaddr = 0x5000;
  const uint64_t sidecar_descriptor_delta =
      reinterpret_cast<uintptr_t>(&descriptors.virtual_sidecar) -
      reinterpret_cast<uintptr_t>(&descriptors.normal);
  const std::vector<VirtualLdsMetadataForTest> metadata = {{
      .kernel_name = "oversized_kernel",
      .normal_descriptor_vaddr = normal_descriptor_vaddr,
      .virtual_descriptor_vaddr = normal_descriptor_vaddr + sidecar_descriptor_delta,
      .static_lds_bytes = descriptors.normal.group_segment_fixed_size,
      .normal_private_segment_size = 12,
      .virtual_private_segment_size = descriptors.virtual_sidecar.private_segment_fixed_size,
      .kernarg_size = 0,
      .backing_pointer_kernarg_offset = kVirtualLdsWrapperStateOffsetForTest,
      .virtual_lds_base_sgpr = 8,
      .flags = kVirtualLdsWrapperFlagsForTest,
  }};
  const auto code_object =
      make_translated_metadata_elf(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201, metadata);

  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(code_object.data(),
                                                                  code_object.size(), &reader),
            HSA_STATUS_SUCCESS);

  hsa_loaded_code_object_t loaded{};
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGuestAgent, reader,
                                                              nullptr, &loaded),
            HSA_STATUS_SUCCESS);

  hsa_executable_symbol_t symbol{};
  ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
            HSA_STATUS_SUCCESS);
  uint64_t kernel_object = 0;
  ASSERT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernel_object, reinterpret_cast<uintptr_t>(&descriptors.normal));

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&descriptors.virtual_sidecar);
  packet.private_segment_size = descriptors.virtual_sidecar.private_segment_fixed_size;
  packet.group_segment_size = 0;
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  // The sidecar descriptor is write-only from rocjitsu's perspective: it may be
  // installed into a packet after the fallback threshold check, but it must not
  // be accepted as a lookup key for taking the fallback again.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&descriptors.virtual_sidecar));
  EXPECT_EQ(packet.group_segment_size, 0u);
  EXPECT_EQ(packet.private_segment_size, 96u);
  EXPECT_EQ(packet.reserved2, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, QueueDoorbellSignalStoreIsForwardedAfterTrackedQueueScan) {
  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_queue_create_fn, fake_queue_create);
  ASSERT_NE(api.core.hsa_signal_store_relaxed_fn, fake_signal_store_relaxed);

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(g_last_queue_create_agent.handle, kHostAgent.handle);

  g_fake_queue_packets[0].header = HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;
  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  EXPECT_EQ(g_fake_signal_store_relaxed_calls, 1);
  EXPECT_EQ(g_last_signal_store_signal.handle, queue->doorbell_signal.handle);
  EXPECT_EQ(g_last_signal_store_value, 0);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_destroyed_queue, queue);
}

TEST(HsaHooksUnitTest, QueueDoorbellRaisesPacketPrivateSizeFromDescriptor) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t descriptor{};
  descriptor.private_segment_fixed_size = 40;
  g_fake_symbol_kernel_object = reinterpret_cast<uintptr_t>(&descriptor);
  g_fake_symbol_private_segment_size = descriptor.private_segment_fixed_size;

  // Production kernel objects are GPU virtual addresses and cannot safely be
  // dereferenced by the packet hook. Exercise the real symbol-query path that
  // caches the runtime-reported private size before dispatch.
  hsa_executable_symbol_t symbol{};
  ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
            HSA_STATUS_SUCCESS);
  uint64_t kernel_object = 0;
  ASSERT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object),
            HSA_STATUS_SUCCESS);

  // Frameworks can rediscover the same symbol through another lookup path
  // after querying its object. This must not erase the cached private size.
  hsa_executable_symbol_t repeated_symbol{};
  ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &repeated_symbol),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(repeated_symbol.handle, symbol.handle);

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = kernel_object;
  packet.private_segment_size = 0;
  packet.group_segment_size = 0;
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  // Semantic DBT rules may add flat-scratch spills to a descriptor whose source
  // private segment was zero. ROCR can still hand us the original packet value,
  // so the queue scanner must raise the dispatch metadata before hardware sees
  // the packet.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&descriptor));
  EXPECT_EQ(packet.private_segment_size, 40u);
  EXPECT_EQ(packet.group_segment_size, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, ConSanInterceptsDescriptorCreatedComputeQueue) {
  reset_code_object_observations();
  reset_queue_fakes();
  configure_consan_profile(kConSanHookProfiles[0], false);

  FakeApiTable api;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();
  ASSERT_NE(api.amd.hsa_amd_queue_create_fn, fake_amd_queue_create);

  hsa_amd_queue_create_desc_t desc{};
  desc.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
  desc.queue_size_bytes = 4u * sizeof(hsa_kernel_dispatch_packet_t);
  desc.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
  desc.engine_type = HSA_AMD_QUEUE_ENGINE_COMPUTE;
  desc.engine.compute.type = HSA_QUEUE_TYPE_SINGLE;
  desc.engine.compute.private_segment_size = HSA_AMD_PRIVATE_SEGMENT_SIZE_DEFAULT;

  ASSERT_EQ(api.amd.hsa_amd_queue_create_fn(kGuestAgent, &desc, 1u), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_fake_amd_queue_create_calls, 1);
  EXPECT_EQ(g_last_destroyed_queue, &g_fake_batch_queue);
  ASSERT_EQ(desc.queue, &g_fake_queue);
  EXPECT_EQ(g_last_intercept_registered_queue, desc.queue);
  EXPECT_NE(g_fake_intercept_handler, nullptr);

  hook.unload();
  EXPECT_EQ(api.amd.hsa_amd_queue_create_fn, fake_amd_queue_create);
}

TEST(HsaHooksUnitTest, ConSanDynamicStackDispatchAddsMaximumFrameAboveRuntimePrivateSize) {
  reset_code_object_observations();
  reset_queue_fakes();
  configure_consan_profile(kConSanHookProfiles[1], false);

  g_transform_override_result.outcome = TransformOutcome::ModifiedValid;
  install_consan_test_program_identity(g_transform_override_result, ROCJITSU_CODE_ARCH_CDNA3,
                                       ROCJITSU_CODE_TARGET_GFX942);
  g_transform_override_result.replacement = {0x7f, 'E', 'L', 'F', 'd', 'y', 'n'};
  install_consan_test_program_inventory(g_transform_override_result,
                                        [](ProgramInventoryBuilder &builder) {
                                          auto &kernel = builder.add_kernel();
                                          kernel.name = "oversized_kernel";
                                          kernel.descriptor_file_offset = 64u;
                                          kernel.uses_dynamic_stack = true;
                                        });

  PatchInfo first_patch;
  first_patch.phase = PatchPhase::Instrumentation;
  first_patch.kind = PatchKind::TrampolineSyncMetadata;
  first_patch.required_private_segment_size = 48u;
  first_patch.dynamic_private_segment_addend = 32u;
  first_patch.owner_descriptor_file_offsets = {64u};
  g_transform_override_result.patches.push_back(first_patch);

  PatchInfo second_patch = first_patch;
  second_patch.required_private_segment_size = 64u;
  second_patch.dynamic_private_segment_addend = 16u;
  g_transform_override_result.patches.push_back(second_patch);

  FakeApiTable api;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGuestAgent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_SUCCESS);

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  g_fake_symbol_kernel_object = 0x12345678u;
  g_fake_symbol_private_segment_size = 16u;
  hsa_executable_symbol_t symbol{};
  ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
            HSA_STATUS_SUCCESS);
  uint32_t symbol_private_bytes = 0;
  ASSERT_EQ(
      api.core.hsa_executable_symbol_get_info_fn(
          symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &symbol_private_bytes),
      HSA_STATUS_SUCCESS);
  // Symbol metadata exposes the absolute descriptor minimum. The dynamic
  // addend is applied only after the launch has selected its runtime depth.
  EXPECT_EQ(symbol_private_bytes, 64u);

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = g_fake_symbol_kernel_object;
  packet.private_segment_size = 1024u;
  packet.workgroup_size_x = 64u;
  packet.workgroup_size_y = 1u;
  packet.workgroup_size_z = 1u;
  packet.grid_size_x = 64u;
  packet.grid_size_y = 1u;
  packet.grid_size_z = 1u;

  ASSERT_NE(g_fake_intercept_handler, nullptr);
  g_fake_intercept_handler(&packet, 1u, 0u, g_fake_intercept_user_data,
                           fake_intercept_packet_writer);

  // Alternative site-local frames cannot overlap, so the per-kernel addend is
  // their maximum (32), not their sum (48).
  ASSERT_EQ(g_last_intercept_written_packets.size(), 1u);
  EXPECT_EQ(g_last_intercept_written_packets.front().private_segment_size, 1056u);
  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, ConSanDynamicPrivateReplacementRequiresDispatchPacketInterception) {
  for (const bool fail_closed : {false, true}) {
    SCOPED_TRACE(fail_closed ? "fail-closed" : "fail-open");
    reset_code_object_observations();
    configure_consan_profile(kConSanHookProfiles[0], fail_closed);
    ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", "trap");

    g_transform_override_result.outcome = TransformOutcome::ModifiedValid;
    install_consan_test_program_identity(g_transform_override_result, ROCJITSU_CODE_ARCH_CDNA3,
                                         ROCJITSU_CODE_TARGET_GFX942);
    install_consan_test_program_inventory(g_transform_override_result,
                                          [](ProgramInventoryBuilder &builder) {
                                            auto &kernel = builder.add_kernel();
                                            kernel.name = "oversized_kernel";
                                            kernel.descriptor_file_offset = 64u;
                                            kernel.entry_text_offset = 0u;
                                            kernel.code_size = sizeof(uint32_t);
                                            kernel.has_text_range = true;
                                            kernel.uses_dynamic_stack = true;
                                          });
    g_transform_override_result.replacement = {0x7f, 'E', 'L', 'F', 'd', 'y', 'n'};
    PatchInfo patch;
    patch.phase = PatchPhase::Instrumentation;
    patch.kind = PatchKind::FlatLoadCheckTrap;
    patch.required_private_segment_size = 32u;
    patch.dynamic_private_segment_addend = 32u;
    patch.owner_descriptor_file_offsets = {64u};
    g_transform_override_result.patches.push_back(patch);

    FakeApiTable api;
    api.amd.hsa_amd_queue_intercept_create_fn = nullptr;
    api.amd.hsa_amd_queue_intercept_register_fn = nullptr;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();

    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
        kFakeExecutable, kGuestAgent, reader, nullptr, nullptr);

    if (fail_closed) {
      EXPECT_EQ(status, HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
      EXPECT_TRUE(g_loaded_code_object_readers.empty());
    } else {
      EXPECT_EQ(status, HSA_STATUS_SUCCESS);
      EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{reader.handle});
    }
    EXPECT_EQ(g_code_object_reader_create_calls, 1);
    EXPECT_TRUE(g_destroyed_code_object_readers.empty());
  }
}

void configure_consan_symbol_binding_case() {
  reset_code_object_observations();
  reset_queue_fakes();
  configure_consan_profile(kConSanHookProfiles[1], false);

  g_transform_override_result.outcome = TransformOutcome::ModifiedValid;
  install_consan_test_program_identity(g_transform_override_result, ROCJITSU_CODE_ARCH_CDNA3,
                                       ROCJITSU_CODE_TARGET_GFX942);
  g_transform_override_result.replacement = {0x7f, 'E', 'L', 'F', 's', 'y', 'm'};
  install_consan_test_program_inventory(g_transform_override_result,
                                        [](ProgramInventoryBuilder &builder) {
                                          auto &kernel = builder.add_kernel();
                                          kernel.name = "oversized_kernel";
                                          kernel.descriptor_file_offset = 64u;
                                          kernel.entry_text_offset = 0u;
                                          kernel.code_size = sizeof(uint32_t);
                                          kernel.has_text_range = true;
                                        });

  PatchInfo patch;
  patch.phase = PatchPhase::Instrumentation;
  patch.kind = PatchKind::TrampolineSyncMetadata;
  patch.required_private_segment_size = 64u;
  patch.owner_descriptor_file_offsets = {64u};
  g_transform_override_result.patches.push_back(std::move(patch));
}

void load_consan_symbol_binding_case(FakeApiTable &api) {
  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGuestAgent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_SUCCESS);
}

void expect_consan_symbol_segment_requirements(FakeApiTable &api, hsa_executable_symbol_t symbol) {
  uint32_t private_bytes = 0;
  ASSERT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_bytes),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(private_bytes, 64u);
  uint32_t group_bytes = 0;
  ASSERT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_bytes),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(group_bytes, g_fake_symbol_group_segment_size);
}

TEST(HsaHooksUnitTest, ConSanLegacyGetSymbolBindsDispatchSegmentRequirements) {
  configure_consan_symbol_binding_case();
  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();
  ASSERT_NE(api.core.hsa_executable_get_symbol_fn, fake_executable_get_symbol);
  load_consan_symbol_binding_case(api);

  g_fake_symbol_kernel_object = 0x12345678u;
  g_fake_symbol_private_segment_size = 16u;
  g_fake_symbol_group_segment_size = 32u;
  hsa_executable_symbol_t symbol{};
  ASSERT_EQ(api.core.hsa_executable_get_symbol_fn(
                kFakeExecutable, nullptr, g_fake_symbol_name.c_str(), kGuestAgent, 0, &symbol),
            HSA_STATUS_SUCCESS);
  expect_consan_symbol_segment_requirements(api, symbol);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(kFakeExecutable), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, ConSanLegacyIterateSymbolsBindsDispatchSegmentRequirements) {
  configure_consan_symbol_binding_case();
  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();
  ASSERT_NE(api.core.hsa_executable_iterate_symbols_fn, fake_executable_iterate_symbols);
  load_consan_symbol_binding_case(api);

  g_fake_symbol_kernel_object = 0x12345678u;
  g_fake_symbol_private_segment_size = 16u;
  g_fake_symbol_group_segment_size = 32u;
  IteratedLegacySymbolForTest iterated{};
  ASSERT_EQ(api.core.hsa_executable_iterate_symbols_fn(
                kFakeExecutable, capture_iterated_legacy_symbol_for_test, &iterated),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(iterated.symbol.handle, kFakeKernelSymbol.handle);
  expect_consan_symbol_segment_requirements(api, iterated.symbol);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(kFakeExecutable), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, ConSanExecutableDestroyDropsReusedSymbolDispatchMetadata) {
  configure_consan_symbol_binding_case();
  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();
  load_consan_symbol_binding_case(api);

  g_fake_symbol_kernel_object = 0x12345678u;
  g_fake_symbol_private_segment_size = 16u;
  g_fake_symbol_group_segment_size = 32u;
  hsa_executable_symbol_t symbol{};
  ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
            HSA_STATUS_SUCCESS);
  expect_consan_symbol_segment_requirements(api, symbol);

  ASSERT_EQ(api.core.hsa_executable_destroy_fn(kFakeExecutable), HSA_STATUS_SUCCESS);

  hsa_executable_symbol_t reused_symbol{};
  ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &reused_symbol),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(reused_symbol.handle, symbol.handle);
  uint32_t private_bytes = 0;
  ASSERT_EQ(
      api.core.hsa_executable_symbol_get_info_fn(
          reused_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_bytes),
      HSA_STATUS_SUCCESS);
  EXPECT_EQ(private_bytes, 16u);
  uint32_t group_bytes = 0;
  ASSERT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                reused_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_bytes),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(group_bytes, 32u);
}

void configure_consan_zero_record_case() {
  reset_code_object_observations();
  reset_queue_fakes();
  unsetenv("RJ_CONSAN_REPORT_BUFFER");
  unsetenv("RJ_CONSAN_REPORT_BUFFER_SIZE");
  setenv("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", "1048576", 1);
  g_transform_override_result.outcome = TransformOutcome::ModifiedValid;
  install_consan_test_program_identity(g_transform_override_result, ROCJITSU_CODE_ARCH_CDNA3,
                                       ROCJITSU_CODE_TARGET_GFX942);
  g_transform_override_result.replacement = {0x7f, 'E', 'L', 'F', 's', 'a', 'm', 'p'};
  install_consan_test_program_inventory(g_transform_override_result,
                                        [](ProgramInventoryBuilder &builder) {
                                          auto &kernel = builder.add_kernel();
                                          kernel.name = "oversized_kernel";
                                          kernel.descriptor_file_offset = 64u;
                                          kernel.entry_text_offset = 0u;
                                          kernel.code_size = sizeof(uint32_t);
                                          kernel.has_text_range = true;
                                        });

  install_test_access_coverage(g_transform_override_result, 1u, SiteDecisionKind::Admitted,
                               AccessPolicyReason::None, LoweringOutcomeKind::Instrumented);

  PatchInfo patch;
  patch.phase = PatchPhase::Instrumentation;
  patch.kind = PatchKind::TrampolineSyncMetadata;
  patch.owner_descriptor_file_offsets = {64u};
  g_transform_override_result.patches.push_back(patch);
}

void run_consan_zero_record_case(bool iterate_symbol, bool dispatch_kernel) {
  FakeApiTable api;
  api.core.hsa_agent_iterate_regions_fn = fake_guest_agent_iterate_regions;
  api.core.hsa_memory_assign_agent_fn = nullptr;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledDbiHook hook(api);
  if (!hook.installed())
    std::_Exit(1);

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  if (api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                            &reader) != HSA_STATUS_SUCCESS)
    std::_Exit(2);
  if (api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGuestAgent, reader,
                                                        nullptr, nullptr) != HSA_STATUS_SUCCESS)
    std::_Exit(3);

  g_fake_symbol_kernel_object = 0x12345678u;
  hsa_executable_symbol_t symbol{};
  if (iterate_symbol) {
    IteratedSymbolForTest iterated{};
    if (api.core.hsa_executable_iterate_agent_symbols_fn(kFakeExecutable, kGuestAgent,
                                                         capture_iterated_symbol_for_test,
                                                         &iterated) != HSA_STATUS_SUCCESS)
      std::_Exit(4);
    symbol = iterated.symbol;
  } else if (api.core.hsa_executable_get_symbol_by_name_fn(kFakeExecutable,
                                                           g_fake_symbol_name.c_str(), &kGuestAgent,
                                                           &symbol) != HSA_STATUS_SUCCESS) {
    std::_Exit(5);
  }
  if (!dispatch_kernel)
    return;

  hsa_queue_t *queue = nullptr;
  if (api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0,
                                   &queue) != HSA_STATUS_SUCCESS)
    std::_Exit(6);
  hsa_kernel_dispatch_packet_t dispatch{};
  dispatch.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  dispatch.kernel_object = g_fake_symbol_kernel_object;
  dispatch.workgroup_size_x = 64u;
  dispatch.workgroup_size_y = 1u;
  dispatch.workgroup_size_z = 1u;
  dispatch.grid_size_x = 64u;
  dispatch.grid_size_y = 1u;
  dispatch.grid_size_z = 1u;
  if (g_fake_intercept_handler == nullptr)
    std::_Exit(7);
  g_fake_intercept_handler(&dispatch, 1u, 0u, g_fake_intercept_user_data,
                           fake_intercept_packet_writer);
}

TEST(HsaHooksUnitTest, ConSanZeroRecordDiagnosticReportsRuntimeDispatch) {
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "strict");
  configure_consan_zero_record_case();

  ASSERT_EXIT(
      ([] {
        run_consan_zero_record_case(/*iterate_symbol=*/true, /*dispatch_kernel=*/true);
        std::_Exit(8);
      }()),
      testing::ExitedWithCode(86),
      "zero visible records after 1 instrumented dispatch packet.*runtime sampling may have "
      "selected no workgroups.*stride=256 offset=0");
}

TEST(HsaHooksUnitTest, ConSanZeroRecordDiagnosticRecognizesIndependentCellSampling) {
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "strict");
  ScopedEnvVar legacy_stride("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE", nullptr);
  ScopedEnvVar legacy_offset("RJ_CONSAN_RUNTIME_SAMPLE_OFFSET", nullptr);
  ScopedEnvVar workgroup_stride("RJ_CONSAN_WORKGROUP_SAMPLE_STRIDE", "1");
  ScopedEnvVar cell_stride("RJ_CONSAN_CELL_SAMPLE_STRIDE", "256");
  configure_consan_zero_record_case();
  g_transform_override_result.patches.front().kind = PatchKind::TrampolineWatchpointStore;
  install_test_access_coverage(g_transform_override_result, 1u, SiteDecisionKind::Admitted,
                               AccessPolicyReason::None, LoweringOutcomeKind::Instrumented,
                               Mode::Default, ProbeIntentKind::Access);
  ASSERT_EXIT(([] {
                run_consan_zero_record_case(/*iterate_symbol=*/true, /*dispatch_kernel=*/true);
                std::_Exit(8);
              }()),
              testing::ExitedWithCode(86),
              "independent sampling may have selected no workgroups or LDS cells "
              ".*workgroup_stride=1 workgroup_offset=0 cell_stride=256 cell_offset=0");
}

TEST(HsaHooksUnitTest, ConSanZeroRecordDiagnosticReportsNoDispatch) {
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "strict");
  configure_consan_zero_record_case();

  ASSERT_EXIT(([] {
                run_consan_zero_record_case(/*iterate_symbol=*/false, /*dispatch_kernel=*/false);
                std::_Exit(8);
              }()),
              testing::ExitedWithCode(86),
              "zero visible records and no kernel dispatch packet was observed");
}

TEST(HsaHooksUnitTest, ConSanAllowlistReportsInstrumentedEntryThatNeverDispatched) {
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "strict");
  ScopedEnvVar allowlist("RJ_CONSAN_KERNEL_ALLOWLIST", "oversized_kernel.kd");
  configure_consan_zero_record_case();

  ASSERT_EXIT(([] {
                run_consan_zero_record_case(/*iterate_symbol=*/false, /*dispatch_kernel=*/false);
                std::_Exit(8);
              }()),
              testing::ExitedWithCode(86),
              "ConSan kernel allowlist entry name=oversized_kernel loaded=true instrumented=true "
              "dispatches=0 visible_records=0 status=instrumented-not-dispatched");
}

TEST(HsaHooksUnitTest, ConSanAllowlistReportsLoadedEntryThatWasNotInstrumented) {
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "strict");
  ScopedEnvVar require_patch("RJ_CONSAN_REQUIRE_PATCH", "0");
  ScopedEnvVar allowlist("RJ_CONSAN_KERNEL_ALLOWLIST", "oversized_kernel");
  configure_consan_zero_record_case();
  install_test_access_coverage(g_transform_override_result, 1u, SiteDecisionKind::Admitted,
                               AccessPolicyReason::None, LoweringOutcomeKind::PlacementRejected);

  ASSERT_EXIT(([] {
                run_consan_zero_record_case(/*iterate_symbol=*/false, /*dispatch_kernel=*/false);
                std::_Exit(8);
              }()),
              testing::ExitedWithCode(86),
              "ConSan kernel allowlist entry name=oversized_kernel loaded=true instrumented=false "
              "dispatches=0 visible_records=0 status=loaded-not-instrumented");
}

TEST(HsaHooksUnitTest, ConSanAllowlistDistinguishesDispatchedEntryWithZeroRecords) {
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "strict");
  ScopedEnvVar allowlist("RJ_CONSAN_KERNEL_ALLOWLIST", "oversized_kernel");
  configure_consan_zero_record_case();

  ASSERT_EXIT(([] {
                run_consan_zero_record_case(/*iterate_symbol=*/false, /*dispatch_kernel=*/true);
                std::_Exit(8);
              }()),
              testing::ExitedWithCode(86),
              "ConSan kernel allowlist entry name=oversized_kernel loaded=true instrumented=true "
              "dispatches=1 visible_records=0 status=instrumented-dispatched");
}

TEST(HsaHooksUnitTest, ConSanAllowlistReportsEntryThatWasNeverLoaded) {
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "strict");
  ScopedEnvVar allowlist("RJ_CONSAN_KERNEL_ALLOWLIST", "missing_kernel");
  configure_consan_zero_record_case();

  ASSERT_EXIT(([] {
                run_consan_zero_record_case(/*iterate_symbol=*/false, /*dispatch_kernel=*/false);
                std::_Exit(8);
              }()),
              testing::ExitedWithCode(86),
              "ConSan kernel allowlist entry name=missing_kernel loaded=false instrumented=false "
              "dispatches=0 visible_records=0 status=not-loaded");
}

TEST(HsaHooksUnitTest, ConSanZeroRecordDiagnosticReportsDensePathGap) {
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "strict");
  ScopedEnvVar runtime_stride("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE", "1");
  configure_consan_zero_record_case();

  ASSERT_EXIT(
      ([] {
        run_consan_zero_record_case(/*iterate_symbol=*/false, /*dispatch_kernel=*/true);
        std::_Exit(8);
      }()),
      testing::ExitedWithCode(86),
      "zero visible records after 1 instrumented dispatch packet.*dense record path produced no "
      "evidence");
}

TEST(HsaHooksUnitTest, MultiProducerDoorbellRewritesEarlierPublishedPacket) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  // Fallback (non-intercept) multi-producer path: a producer publishes two ready
  // packets and rings once with the FINAL packet id. Packet 0 needs virtual-LDS
  // rewriting, packet 1 does not. The doorbell must rewrite the whole published
  // range [next_packet_id, id], not just the named packet -- otherwise packet 0
  // reaches the command processor as an oversized (host-faulting) launch and the
  // frontier advances past it so the scanner skips it too.
  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);
  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000; // exceeds host LDS -> sidecar
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  // Packet 0: the oversized virtual-LDS dispatch that must be rewritten.
  auto &oversized = g_fake_queue_packets[0];
  oversized.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  oversized.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  oversized.group_segment_size = normal_descriptor.group_segment_fixed_size;
  oversized.workgroup_size_x = 64;
  oversized.workgroup_size_y = 1;
  oversized.workgroup_size_z = 1;
  oversized.grid_size_x = 64;
  oversized.grid_size_y = 1;
  oversized.grid_size_z = 1;

  // Packet 1: an ordinary below-threshold dispatch (no rewrite needed).
  auto &ordinary = g_fake_queue_packets[1];
  ordinary.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  ordinary.kernel_object = 0; // no registered virtual-LDS metadata
  ordinary.group_segment_size = 0;
  ordinary.workgroup_size_x = 64;
  ordinary.grid_size_x = 64;

  // Ring once with the FINAL packet id (1).
  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 1);

  // Packet 0 was rewritten to the virtual descriptor (range covered), not left
  // on its oversized normal descriptor.
  EXPECT_EQ(oversized.kernel_object, reinterpret_cast<uintptr_t>(&virtual_descriptor));
  EXPECT_EQ(oversized.group_segment_size, 0u);
  EXPECT_FALSE(g_fake_allocation_sizes.empty());

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, MultiProducerHoleCloseAdvancesFrontierAcrossReadySuffix) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  // Regression for the stranded-frontier bug: on a size-4 multi-producer queue an
  // out-of-order ring publishes a ready suffix ABOVE an unready hole, the hole
  // later closes, and a subsequent batch must not skip a packet.
  //
  //   1. next_packet_id = 0. Slot 0 (packet 0) is INVALID (hole); packets 1..3 are
  //      ready. A producer rings with id 3. The range [0,4) rewrites 1..3, but the
  //      cursor cannot pass the hole at 0 -- it must REMEMBER 1..3 as ready.
  //   2. Packet 0 closes (becomes ready) and rings with id 0. The cursor must now
  //      catch up across the remembered 1..3 -> next_packet_id = 4, NOT 1.
  //   3. Packets 4 and 5 are published and a producer rings once with id 5. Only
  //      when the cursor is at 4 does 5 - 4 = 1 < size take the range path and
  //      rewrite packet 4. If the cursor were stranded at 1, 5 - 1 = 4 == size
  //      would fall to the single-packet path and packet 4 (needing a virtual-LDS
  //      rewrite) would reach the command processor as an oversized launch.
  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000; // exceeds host LDS -> sidecar
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  const auto make_oversized = [&](uint32_t slot) {
    auto &packet = g_fake_queue_packets[slot];
    packet = {};
    packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
    packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
    packet.group_segment_size = normal_descriptor.group_segment_fixed_size;
    packet.workgroup_size_x = 64;
    packet.workgroup_size_y = 1;
    packet.workgroup_size_z = 1;
    packet.grid_size_x = 64;
    packet.grid_size_y = 1;
    packet.grid_size_z = 1;
  };
  const auto make_ordinary = [&](uint32_t slot) {
    auto &packet = g_fake_queue_packets[slot];
    packet = {};
    packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
    packet.kernel_object = 0; // no registered virtual-LDS metadata
    packet.workgroup_size_x = 64;
    packet.workgroup_size_y = 1;
    packet.workgroup_size_z = 1;
    packet.grid_size_x = 64;
    packet.grid_size_y = 1;
    packet.grid_size_z = 1;
  };
  const auto make_invalid = [&](uint32_t slot) {
    auto &packet = g_fake_queue_packets[slot];
    packet = {};
    packet.header = HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;
  };

  // Phase 1: slot 0 (packet 0) is an unready hole; packets 1..3 are ready ordinary
  // dispatches. Ring with the final published id (3).
  make_invalid(0);
  make_ordinary(1);
  make_ordinary(2);
  make_ordinary(3);
  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 3);
  // The hole blocks the cursor, so no virtual-LDS rewrite happened yet.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());

  // Phase 2: the hole closes (packet 0 becomes a ready ordinary dispatch) and its
  // producer rings with id 0. The cursor must catch up across the remembered ready
  // suffix 1..3 and land at 4.
  make_ordinary(0);
  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);
  EXPECT_TRUE(g_fake_allocation_sizes.empty());

  // Phase 3: packets 4 (slot 0) and 5 (slot 1) are published; packet 4 is the
  // oversized virtual-LDS dispatch. Ring once with the final id (5).
  make_oversized(0); // packet 4 -> slot 4 % 4 == 0
  make_ordinary(1);  // packet 5 -> slot 5 % 4 == 1
  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 5);

  // Packet 4 was rewritten to the virtual descriptor -- it was NOT skipped.
  EXPECT_EQ(g_fake_queue_packets[0].kernel_object,
            reinterpret_cast<uintptr_t>(&virtual_descriptor));
  EXPECT_EQ(g_fake_queue_packets[0].group_segment_size, 0u);
  EXPECT_FALSE(g_fake_allocation_sizes.empty());

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsRewriteWorksWithoutLoadedCodeObjectOutput) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size, 0,
      kVirtualLdsWrapperStateOffsetForTest, kVirtualLdsWrapperFlagsForTest, true, false);
  (void)registration;

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  constexpr uint32_t kStaticLds = 70000;
  constexpr uint32_t kDynamicLds = 1024;
  constexpr uint32_t kRequestedLds = kStaticLds + kDynamicLds;
  // HSA packets report the total group-segment allocation. Static LDS is kept
  // separately in rocjitsu metadata only so symbol-time virtual descriptors,
  // which advertise zero hardware LDS, can still allocate the minimum backing
  // store when a packet arrives with a zero group size.
  packet.group_segment_size = kRequestedLds;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  constexpr uint32_t kGroupsX = 4;
  ASSERT_EQ(g_fake_allocation_sizes.size(), 2u);
  EXPECT_EQ(g_fake_allocation_pools[0].handle, kHostPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[0], static_cast<size_t>(kRequestedLds * kGroupsX));
  EXPECT_EQ(g_fake_allocation_pools[1].handle, kHostKernargPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[1], kVirtualLdsWrapperSizeForTest);

  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&virtual_descriptor));
  EXPECT_EQ(packet.group_segment_size, 0u);
  // The normal descriptor has no fixed private allocation, so the packet's 12
  // bytes are entirely dynamic and must be added above the sidecar's 96 bytes.
  EXPECT_EQ(packet.private_segment_size, 108u);
  EXPECT_EQ(packet.reserved2, 0u);
  EXPECT_EQ(packet.kernarg_address, g_fake_allocations[1].data());

  struct RuntimeState {
    uint64_t backing_base = 0;
    uint32_t stride_x = 0;
    uint32_t stride_y = 0;
    uint32_t stride_z = 0;
    uint32_t reserved = 0;
  } state{};
  static_assert(sizeof(RuntimeState) == 24);
  std::memcpy(&state, g_fake_allocations[1].data() + kVirtualLdsWrapperStateOffsetForTest,
              sizeof(state));
  EXPECT_EQ(state.backing_base, reinterpret_cast<uintptr_t>(g_fake_allocations[0].data()));
  EXPECT_EQ(state.stride_x, kRequestedLds);
  EXPECT_EQ(state.stride_y, 0u);
  EXPECT_EQ(state.stride_z, 0u);
  EXPECT_EQ(state.reserved, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsScannerKeepsDestroyedSignalSlotUntilReuse) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  constexpr hsa_signal_t kApplicationSignal{4243};
  set_fake_signal_value(kApplicationSignal, 1);

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 71024;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;
  packet.completion_signal = kApplicationSignal;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);
  ASSERT_EQ(g_fake_allocation_sizes.size(), 2u);
  ASSERT_TRUE(g_fake_freed_allocations.empty());
  void *first_wrapper = packet.kernarg_address;

  set_fake_signal_value(kApplicationSignal, 0);
  EXPECT_EQ(api.core.hsa_signal_destroy_fn(kApplicationSignal), HSA_STATUS_SUCCESS);

  // The scanner still needs the slot record to recognize this as the already
  // rewritten packet. Ringing the same packet again must neither free its memory
  // while the stale packet points at it nor perform a second rewrite.
  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);
  EXPECT_EQ(packet.kernarg_address, first_wrapper);
  EXPECT_EQ(g_fake_allocation_sizes.size(), 2u);
  EXPECT_TRUE(g_fake_freed_allocations.empty());

  // Packet id 2 reuses slot 0. At this point the old packet is no longer visible,
  // so its explicitly-completed buffers can be retired without loading the
  // destroyed signal, before the replacement dispatch receives new buffers.
  packet = {};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 71024;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;
  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 2);

  EXPECT_EQ(g_fake_allocation_sizes.size(), 4u);
  EXPECT_EQ(g_fake_freed_allocations.size(), 2u);
  EXPECT_NE(packet.kernarg_address, first_wrapper);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsRewriteCopiesOriginalKernargIntoWrapperPrefix) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000;
  virtual_descriptor.private_segment_fixed_size = 96;

  constexpr uint32_t kSourceKernargSize = 16;
  constexpr uint32_t kOriginalPointerOffset = 16;
  constexpr uint32_t kRuntimeStateOffset = 24;
  constexpr uint32_t kWrapperSize = 48;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size,
      kSourceKernargSize, kRuntimeStateOffset);
  (void)registration;

  const std::array<uint8_t, kSourceKernargSize> original_kernarg = {
      0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87,
      0x98, 0xA9, 0xBA, 0xCB, 0xDC, 0xED, 0xFE, 0x0F,
  };

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.kernarg_address = const_cast<uint8_t *>(original_kernarg.data());
  packet.private_segment_size = 12;
  packet.group_segment_size = normal_descriptor.group_segment_fixed_size;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  ASSERT_EQ(g_fake_allocation_sizes.size(), 2u);
  EXPECT_EQ(g_fake_allocation_pools[1].handle, kHostKernargPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[1], kWrapperSize);
  ASSERT_EQ(packet.kernarg_address, g_fake_allocations[1].data());
  const auto *wrapper = g_fake_allocations[1].data();
  EXPECT_EQ(std::memcmp(wrapper, original_kernarg.data(), original_kernarg.size()), 0);

  uint64_t copied_original_pointer = 0;
  std::memcpy(&copied_original_pointer, wrapper + kOriginalPointerOffset,
              sizeof(copied_original_pointer));
  EXPECT_EQ(copied_original_pointer, reinterpret_cast<uintptr_t>(original_kernarg.data()));

  struct RuntimeState {
    uint64_t backing_base = 0;
    uint32_t stride_x = 0;
    uint32_t stride_y = 0;
    uint32_t stride_z = 0;
    uint32_t reserved = 0;
  } state{};
  static_assert(sizeof(RuntimeState) == 24);
  std::memcpy(&state, wrapper + kRuntimeStateOffset, sizeof(state));
  EXPECT_EQ(state.backing_base, reinterpret_cast<uintptr_t>(g_fake_allocations[0].data()));
  EXPECT_EQ(state.stride_x, normal_descriptor.group_segment_fixed_size);
  EXPECT_EQ(state.stride_y, 0u);
  EXPECT_EQ(state.stride_z, 0u);
  EXPECT_EQ(state.reserved, 0u);
  EXPECT_EQ(packet.reserved2, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsRewriteKeepsBelowThresholdDispatchOnNormalDescriptor) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 0;
  normal_descriptor.private_segment_fixed_size = 40;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 16384;
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 16;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  // Virtual LDS metadata may be present for a dynamic-LDS overflow fallback, but
  // a launch that fits in host hardware must keep the normal descriptor and
  // hardware LDS allocation untouched.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&normal_descriptor));
  EXPECT_EQ(packet.group_segment_size, 16384u);
  EXPECT_EQ(packet.private_segment_size, 40u);
  EXPECT_EQ(packet.reserved2, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsRewriteKeepsExactHardwareLimitOnNormalDescriptor) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 64 * 1024;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 0;
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  // Exactly 64 KiB still fits CDNA3 hardware LDS, so the virtual descriptor is
  // a fallback candidate only and must not be selected for this launch.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&normal_descriptor));
  EXPECT_EQ(packet.group_segment_size, 0u);
  EXPECT_EQ(packet.private_segment_size, 12u);
  EXPECT_EQ(packet.reserved2, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsRewriteKeepsStaticPlusDynamicLimitOnNormalDescriptor) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 32 * 1024;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 64 * 1024;
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  // HSA packets carry total LDS, not a dynamic-only tail. A 32 KiB static
  // descriptor plus 32 KiB dynamic request is a 64 KiB packet and still fits, so
  // the virtual sidecar is only a fallback candidate and must not replace it.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&normal_descriptor));
  EXPECT_EQ(packet.group_segment_size, 64u * 1024u);
  EXPECT_EQ(packet.private_segment_size, 12u);
  EXPECT_EQ(packet.reserved2, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsInterceptRewritePublishesWrapperKernarg) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(g_last_queue_create_agent.handle, kHostAgent.handle);
  EXPECT_EQ(g_last_intercept_registered_queue, queue);
  ASSERT_NE(g_fake_intercept_handler, nullptr);
  EXPECT_EQ(g_fake_intercept_user_data, queue);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000;
  normal_descriptor.private_segment_fixed_size = 40;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  // The intercept adapter must preserve 80 dynamic bytes above normal fixed.
  packet.private_segment_size = 120;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  constexpr uint32_t kStaticLds = 70000;
  constexpr uint32_t kDynamicLds = 1024;
  constexpr uint32_t kRequestedLds = kStaticLds + kDynamicLds;
  // HSA packets report total LDS, so dynamic LDS has already been added to the
  // packet value before rocjitsu scans or intercepts the dispatch.
  packet.group_segment_size = kRequestedLds;

  constexpr uint64_t kPacketIndex = 2;
  g_fake_queue_packets[kPacketIndex] = packet;
  g_fake_intercept_handler(&packet, 1, kPacketIndex, g_fake_intercept_user_data,
                           fake_intercept_packet_writer);

  constexpr uint32_t kGroupsX = 4;
  ASSERT_EQ(g_fake_allocation_sizes.size(), 2u);
  EXPECT_EQ(g_fake_allocation_pools[0].handle, kHostPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[0], static_cast<size_t>(kRequestedLds * kGroupsX));
  EXPECT_EQ(g_fake_allocation_pools[1].handle, kHostKernargPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[1], kVirtualLdsWrapperSizeForTest);

  ASSERT_EQ(g_last_intercept_written_packets.size(), 1u);
  const hsa_kernel_dispatch_packet_t &written = g_last_intercept_written_packets[0];
  EXPECT_EQ(written.kernel_object, reinterpret_cast<uintptr_t>(&virtual_descriptor));
  EXPECT_EQ(written.group_segment_size, 0u);
  EXPECT_EQ(written.private_segment_size, 176u);
  EXPECT_EQ(written.reserved2, 0u);
  EXPECT_EQ(written.kernarg_address, g_fake_allocations[1].data());
  EXPECT_EQ(g_fake_queue_packets[kPacketIndex].reserved2, 0u);

  struct RuntimeState {
    uint64_t backing_base = 0;
    uint32_t stride_x = 0;
    uint32_t stride_y = 0;
    uint32_t stride_z = 0;
    uint32_t reserved = 0;
  } state{};
  static_assert(sizeof(RuntimeState) == 24);
  std::memcpy(&state, g_fake_allocations[1].data() + kVirtualLdsWrapperStateOffsetForTest,
              sizeof(state));
  EXPECT_EQ(state.backing_base, reinterpret_cast<uintptr_t>(g_fake_allocations[0].data()));
  EXPECT_EQ(state.stride_x, kRequestedLds);
  EXPECT_EQ(state.stride_y, 0u);
  EXPECT_EQ(state.stride_z, 0u);
  EXPECT_EQ(state.reserved, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsInterceptReleasesCompletedRetiredBuffers) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);
  ASSERT_NE(g_fake_intercept_handler, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 71024;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  packet.completion_signal = {};
  g_fake_queue_packets[1] = packet;
  g_fake_intercept_handler(&packet, 1, 1, g_fake_intercept_user_data, fake_intercept_packet_writer);
  ASSERT_EQ(g_fake_allocation_sizes.size(), 2u);
  ASSERT_TRUE(g_fake_freed_allocations.empty());
  ASSERT_EQ(g_fake_created_signals.size(), 1u);
  ASSERT_EQ(g_last_intercept_written_packets.size(), 1u);
  EXPECT_EQ(g_last_intercept_written_packets[0].completion_signal.handle,
            g_fake_created_signals[0].handle);
  void *first_backing = g_fake_allocations[0].data();
  void *first_wrapper = g_fake_allocations[1].data();
  const hsa_signal_t first_signal = g_fake_created_signals[0];

  // Intercept callbacks retain virtual-LDS buffers after writing packets to
  // ROCR. Real framework packets are often fire-and-forget, so rocjitsu adds a
  // private completion signal and uses it as a fence. When a later callback
  // observes that signal at zero, the old backing/wrapper allocations can be
  // returned before allocating the next oversized dispatch.
  set_fake_signal_value(first_signal, 0);
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 71024;
  packet.reserved2 = 0;
  packet.completion_signal = {};
  g_fake_queue_packets[2] = packet;
  g_fake_intercept_handler(&packet, 1, 2, g_fake_intercept_user_data, fake_intercept_packet_writer);

  ASSERT_EQ(g_fake_allocation_sizes.size(), 4u);
  ASSERT_EQ(g_fake_created_signals.size(), 2u);
  ASSERT_EQ(g_fake_freed_allocations.size(), 2u);
  EXPECT_EQ(g_fake_freed_allocations[0], first_wrapper);
  EXPECT_EQ(g_fake_freed_allocations[1], first_backing);
  ASSERT_EQ(g_fake_destroyed_signals.size(), 1u);
  EXPECT_EQ(g_fake_destroyed_signals[0].handle, first_signal.handle);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsInterceptReleasesBorrowedSignalBeforeDestroy) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_signal_destroy_fn, fake_signal_destroy);

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);
  ASSERT_NE(g_fake_intercept_handler, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  constexpr hsa_signal_t kApplicationSignal{4242};
  set_fake_signal_value(kApplicationSignal, 1);

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 71024;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;
  packet.completion_signal = kApplicationSignal;

  g_fake_intercept_handler(&packet, 1, 1, g_fake_intercept_user_data, fake_intercept_packet_writer);
  ASSERT_EQ(g_fake_allocation_sizes.size(), 2u);
  ASSERT_TRUE(g_fake_freed_allocations.empty());
  ASSERT_EQ(g_last_intercept_written_packets.size(), 1u);
  EXPECT_EQ(g_last_intercept_written_packets[0].completion_signal.handle,
            kApplicationSignal.handle);

  // Once the application observes zero, it owns the right to destroy its
  // signal immediately. The destroy wrapper must release the already-completed
  // intercept buffers before the original runtime invalidates the handle; a
  // later dispatch must never need to load from that signal again.
  set_fake_signal_value(kApplicationSignal, 0);
  EXPECT_EQ(api.core.hsa_signal_destroy_fn(kApplicationSignal), HSA_STATUS_SUCCESS);

  ASSERT_EQ(g_fake_freed_allocations.size(), 2u);
  ASSERT_EQ(g_fake_destroyed_signals.size(), 1u);
  EXPECT_EQ(g_fake_destroyed_signals[0].handle, kApplicationSignal.handle);

  // Queue destruction must not release the same backing and kernarg twice after
  // signal destruction removed them from the retired list.
  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_fake_freed_allocations.size(), 2u);
}

TEST(HsaHooksUnitTest, VirtualLdsInterceptKeepsBelowThresholdPacketOnNormalDescriptor) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);
  ASSERT_NE(g_fake_intercept_handler, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 0;
  normal_descriptor.private_segment_fixed_size = 40;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 16384;
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 16;
  packet.grid_size_z = 1;

  constexpr uint64_t kPacketIndex = 2;
  g_fake_queue_packets[kPacketIndex] = packet;
  g_fake_intercept_handler(&packet, 1, kPacketIndex, g_fake_intercept_user_data,
                           fake_intercept_packet_writer);

  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  ASSERT_EQ(g_last_intercept_written_packets.size(), 1u);
  const hsa_kernel_dispatch_packet_t &written = g_last_intercept_written_packets[0];
  EXPECT_EQ(written.kernel_object, reinterpret_cast<uintptr_t>(&normal_descriptor));
  EXPECT_EQ(written.group_segment_size, 16384u);
  EXPECT_EQ(written.private_segment_size, 40u);
  EXPECT_EQ(written.reserved2, 0u);
  EXPECT_EQ(g_fake_queue_packets[kPacketIndex].reserved2, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsInterceptKeepsStaticPlusDynamicLimitOnNormalDescriptor) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);
  ASSERT_NE(g_fake_intercept_handler, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 32 * 1024;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 64 * 1024;
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  constexpr uint64_t kPacketIndex = 2;
  g_fake_queue_packets[kPacketIndex] = packet;
  g_fake_intercept_handler(&packet, 1, kPacketIndex, g_fake_intercept_user_data,
                           fake_intercept_packet_writer);

  // Queue-intercept dispatches use the same fallback rule as direct queue
  // scanning: if total LDS still fits CDNA3 hardware, leave the packet on the
  // normal descriptor and let real LDS handle the DS traffic.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  ASSERT_EQ(g_last_intercept_written_packets.size(), 1u);
  const hsa_kernel_dispatch_packet_t &written = g_last_intercept_written_packets[0];
  EXPECT_EQ(written.kernel_object, reinterpret_cast<uintptr_t>(&normal_descriptor));
  EXPECT_EQ(written.group_segment_size, 64u * 1024u);
  EXPECT_EQ(written.private_segment_size, 12u);
  EXPECT_EQ(written.reserved2, 0u);
  EXPECT_EQ(g_fake_queue_packets[kPacketIndex].reserved2, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, DoorbellForwardsOversizedPacketOnUnrelatedAgentQueueUnchanged) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  // A queue on an agent that is neither the guest nor the guest's execution host.
  // host_lds_bytes is derived from the configured host target (gfx1201 -> 64 KiB),
  // so it does not describe this agent. Its dispatches must be forwarded unchanged
  // even when the group segment exceeds that host limit -- the fail-close for
  // oversized-no-metadata launches applies only to the guest's own queue.
  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kUnrelatedAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr,
                                         nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t descriptor{};
  descriptor.group_segment_fixed_size = 96 * 1024;

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 96 * 1024; // exceeds the guest-derived 64 KiB limit
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&descriptor));
  EXPECT_EQ(packet.group_segment_size, 96u * 1024u);
  EXPECT_EQ(packet.private_segment_size, 12u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, InterceptForwardsOversizedPacketOnUnrelatedAgentQueueUnchanged) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kUnrelatedAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr,
                                         nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);
  ASSERT_NE(g_fake_intercept_handler, nullptr);

  kernel_descriptor_t descriptor{};
  descriptor.group_segment_fixed_size = 96 * 1024;

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 96 * 1024; // exceeds the guest-derived 64 KiB limit
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  constexpr uint64_t kPacketIndex = 2;
  g_fake_queue_packets[kPacketIndex] = packet;
  g_fake_intercept_handler(&packet, 1, kPacketIndex, g_fake_intercept_user_data,
                           fake_intercept_packet_writer);

  // The unrelated-agent intercept queue forwards the oversized packet untouched:
  // no allocation, no rewrite, no abort.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  ASSERT_EQ(g_last_intercept_written_packets.size(), 1u);
  const hsa_kernel_dispatch_packet_t &written = g_last_intercept_written_packets[0];
  EXPECT_EQ(written.kernel_object, reinterpret_cast<uintptr_t>(&descriptor));
  EXPECT_EQ(written.group_segment_size, 96u * 1024u);
  EXPECT_EQ(written.private_segment_size, 12u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, LoadOnUnrelatedAgentForwardsDifferentTargetImageUnchanged) {
  // A code object for an unrelated GPU (a different, non-guest target ISA) loaded
  // on a non-guest agent must be forwarded verbatim to the original loader. The
  // hook must NOT reinterpret it as the guest ISA, translate it, or run
  // rocjitsu metadata validation on it. Even though the reader's bytes ARE
  // registered (memory-backed), the load is gated out by the non-guest early
  // return before any reader lookup or target detection.
  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  // A minimal, valid AMDGPU ELF for gfx942 (neither the guest gfx950 nor the
  // configured host gfx1201). Content is irrelevant: the load must pass through
  // untouched.
  std::vector<uint8_t> image(sizeof(rocjitsu::Elf64_Ehdr), 0);
  const auto ehdr = make_amdgpu_elf_header(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX942);
  write_struct(image, 0, ehdr);

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(image.data(), image.size(), &reader),
      HSA_STATUS_SUCCESS);

  hsa_loaded_code_object_t loaded{};
  const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
      kFakeExecutable, kUnrelatedAgent, reader, nullptr, &loaded);
  EXPECT_EQ(status, HSA_STATUS_SUCCESS);

  // The original loader saw the unrelated agent and reader verbatim (no remap to
  // the guest execution host, no translated reader substituted).
  ASSERT_EQ(g_fake_load_agent_calls, 1);
  EXPECT_EQ(g_last_load_agent.handle, kUnrelatedAgent.handle);
  EXPECT_EQ(g_last_load_reader.handle, reader.handle);
}

TEST(HsaHooksUnitTest, LoadOnUnrelatedAgentForwardsMalformedMetadataImageUnchanged) {
  // A non-guest load must reach the original loader even if its bytes are not a
  // parseable code object / carry malformed rocjitsu metadata: the hook only
  // validates the guest's own loads. Garbage bytes that would fail
  // parse_virtual_lds_hook_metadata must still pass through unchanged.
  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  const std::vector<uint8_t> image(64, 0xAB); // not a valid ELF
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(image.data(), image.size(), &reader),
      HSA_STATUS_SUCCESS);

  hsa_loaded_code_object_t loaded{};
  const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
      kFakeExecutable, kUnrelatedAgent, reader, nullptr, &loaded);
  EXPECT_EQ(status, HSA_STATUS_SUCCESS);

  ASSERT_EQ(g_fake_load_agent_calls, 1);
  EXPECT_EQ(g_last_load_agent.handle, kUnrelatedAgent.handle);
  EXPECT_EQ(g_last_load_reader.handle, reader.handle);
}

TEST(HsaHooksUnitDeathTest, InterceptGuestUnregisteredOversizedDispatchAborts) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  // fork-based death test: the guest queue's intercept handler runs synchronously
  // (no scanner jthread on the intercept path), so aborting inside it is safe to
  // observe. A dispatch on the GUEST queue whose kernel object has no virtual-LDS
  // metadata and whose group segment exceeds the host LDS limit has no sidecar to
  // fall back to, so it must fail closed rather than submit a host-faulting launch.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        reset_pool_blocker(false);
        reset_queue_fakes();
        FakeApiTable api;
        api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
        api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
        InstalledHook hook(api);

        hsa_queue_t *queue = nullptr;
        api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0,
                                     &queue);

        kernel_descriptor_t descriptor{};
        descriptor.group_segment_fixed_size = 96 * 1024;

        hsa_kernel_dispatch_packet_t packet{};
        packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
        packet.kernel_object = reinterpret_cast<uintptr_t>(&descriptor);
        packet.group_segment_size = 96 * 1024; // exceeds the guest 64 KiB limit
        packet.workgroup_size_x = 256;
        packet.workgroup_size_y = 1;
        packet.workgroup_size_z = 1;
        packet.grid_size_x = 256;
        packet.grid_size_y = 1;
        packet.grid_size_z = 1;

        constexpr uint64_t kPacketIndex = 2;
        g_fake_queue_packets[kPacketIndex] = packet;
        g_fake_intercept_handler(&packet, 1, kPacketIndex, g_fake_intercept_user_data,
                                 fake_intercept_packet_writer);
      },
      "no virtual-LDS variant but its group segment exceeds the host LDS limit");
}

TEST(HsaHooksUnitTest, UntrackedSignalStoreScreleaseIsForwarded) {
  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_signal_store_screlease_fn, fake_signal_store_screlease);

  api.core.hsa_signal_store_screlease_fn(hsa_signal_t{999}, 42);

  EXPECT_EQ(g_fake_signal_store_screlease_calls, 1);
  EXPECT_EQ(g_last_signal_store_signal.handle, 999u);
  EXPECT_EQ(g_last_signal_store_value, 42);
}

} // namespace

} // namespace rocjitsu::consan::hook
