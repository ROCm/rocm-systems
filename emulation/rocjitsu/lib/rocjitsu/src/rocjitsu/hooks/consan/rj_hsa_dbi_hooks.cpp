// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hsa_dbi_hooks.cpp
/// @brief HSA tools load-time hook for opt-in rocJITsu DBI instrumentation.
///
/// @details ROCR loads this shared library through `HSA_TOOLS_LIB` during
/// `hsa_init()`. This initial DBI hook only parses configuration, installs the
/// code-object reader/load wrappers, logs observed loads when requested, and
/// routes memory-backed reader bytes through the selected ConSan DBI mode.

#include "hsa/hsa_api_trace_minimal.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_hook_internal.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_registry_lifecycle.h"

#include "rocjitsu/checked_byte_budget.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/waitcheck.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_pipeline.h"
#include "rocjitsu/code/patch/consan/consan_transform_diagnostics.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_process_byte_budget.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_sync.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_transform_memory.h"
#include "rocjitsu/hooks/hsa_api_function_patch.h"
#include "rocjitsu/hooks/hsa_code_object_file_snapshot.h"
#include "rocjitsu/hooks/hsa_code_object_reader_registry.h"
#include "rocjitsu/hooks/hsa_tool_lifetime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>

namespace rocjitsu::consan::hook {

using LoaderCreateFromFileWithOffsetSize = hsa_status_t (*)(hsa_file_t, size_t, size_t,
                                                            hsa_code_object_reader_t *);

struct AmdLoaderExtTable102 {
  std::array<void *, 5> preceding_functions;
  LoaderCreateFromFileWithOffsetSize create_from_file_with_offset_size;
};

struct AmdLoaderExtTable103 : AmdLoaderExtTable102 {
  void *iterate_executables;
};

static_assert(offsetof(AmdLoaderExtTable102, create_from_file_with_offset_size) == 40);
static_assert(sizeof(AmdLoaderExtTable102) == 48);
static_assert(sizeof(AmdLoaderExtTable103) == 56);

std::atomic<int> g_log_level{kLogDisabled};
std::atomic<uint64_t> g_dump_sequence{0};
std::atomic<LogSinkOverride> g_test_log_sink_override{nullptr};

std::mutex &log_mutex();

std::atomic<TransformOverride> g_test_transform_override{nullptr};
std::atomic<size_t> g_test_retry_count{0};

[[nodiscard]] std::string
epoch_analysis_policy_name(const HookConfig::EpochAnalysisPolicy &policy) {
  switch (policy.kind) {
  case HookConfig::EpochAnalysisKind::Every:
    return "every";
  case HookConfig::EpochAnalysisKind::Nth:
    return "nth:" + std::to_string(policy.value);
  case HookConfig::EpochAnalysisKind::Periodic:
    return "periodic:" + std::to_string(policy.value) + ":" + std::to_string(policy.offset);
  case HookConfig::EpochAnalysisKind::Manual:
    return "manual";
  }
  return "invalid";
}

/// Process-wide union of time intervals spent preparing instrumented code
/// objects. Loads may transform concurrently, so summing per-load durations
/// would overstate the wall-clock startup cost seen by the application.
class InstrumentationClock {
public:
  static InstrumentationClock &instance() {
    static InstrumentationClock clock;
    return clock;
  }

  void begin() {
    std::lock_guard lock(mutex_);
    if (active_++ == 0u)
      interval_begin_ = std::chrono::steady_clock::now();
  }

  void end() {
    std::lock_guard lock(mutex_);
    assert(active_ != 0u);
    if (--active_ == 0u) {
      elapsed_ += std::chrono::steady_clock::now() - interval_begin_;
    }
  }

  [[nodiscard]] uint64_t elapsed_nanoseconds() const {
    std::lock_guard lock(mutex_);
    auto elapsed = elapsed_;
    if (active_ != 0u)
      elapsed += std::chrono::steady_clock::now() - interval_begin_;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
  }

  void reset() {
    std::lock_guard lock(mutex_);
    // Reload may overlap an old transform or replacement load. Its RAII
    // timer must still be able to end(), but only post-reset time belongs
    // to the new measurement interval.
    if (active_ != 0u)
      interval_begin_ = std::chrono::steady_clock::now();
    elapsed_ = {};
  }

private:
  mutable std::mutex mutex_;
  size_t active_ = 0;
  std::chrono::steady_clock::time_point interval_begin_{};
  std::chrono::steady_clock::duration elapsed_{};
};

class ScopedInstrumentationTimer {
public:
  ScopedInstrumentationTimer() { InstrumentationClock::instance().begin(); }
  ScopedInstrumentationTimer(const ScopedInstrumentationTimer &) = delete;
  ScopedInstrumentationTimer &operator=(const ScopedInstrumentationTimer &) = delete;
  ~ScopedInstrumentationTimer() { stop(); }

  void stop() {
    if (!active_)
      return;
    active_ = false;
    InstrumentationClock::instance().end();
  }

private:
  bool active_ = true;
};

/// Invoke the production transformer or a test double through the same typed
/// contract. The hook never reconstructs prototype option or result values.
TransformResult run_transform(std::span<const uint8_t> bytes, const Request &request,
                              const TransformPolicy &transform_policy,
                              const RuntimePolicy &runtime_policy, const DebugOverrides &debug,
                              const MutationRequest &mutation,
                              const RuntimeCapabilities &capabilities,
                              const BoundRuntimeResources &resources) {
  if (const TransformOverride override =
          g_test_transform_override.load(std::memory_order_acquire)) {
    return override(bytes, request, transform_policy, runtime_policy, debug, mutation, capabilities,
                    resources);
  }
  return mutation.has_mutation()
             ? transform_with_mutation(bytes, request, transform_policy, runtime_policy, debug,
                                       mutation, capabilities, resources)
             : transform(bytes, request, transform_policy, runtime_policy, debug, capabilities,
                         resources);
}

/// Test executor injected into the library-owned automatic transaction. The
/// resource state distinguishes preparation from resume without exposing the
/// library's selected retry strategy to the hook.
TransformResult run_automatic_test_executor(std::span<const uint8_t> bytes, const Request &request,
                                            const TransformPolicy &transform_policy,
                                            const RuntimePolicy &runtime_policy,
                                            const DebugOverrides &debug,
                                            const MutationRequest &mutation,
                                            const RuntimeCapabilities &capabilities,
                                            const BoundRuntimeResources &resources) {
  const TransformOverride override = g_test_transform_override.load(std::memory_order_acquire);
  if (override == nullptr)
    return run_transform(bytes, request, transform_policy, runtime_policy, debug, mutation,
                         capabilities, resources);
  if (resources.bound() && request.mode == Mode::Default)
    g_test_retry_count.fetch_add(1, std::memory_order_relaxed);
  return override(bytes, request, transform_policy, runtime_policy, debug, mutation, capabilities,
                  resources);
}

AutomaticTransformPreparation
run_automatic_prepare(std::span<const uint8_t> bytes, const Request &request,
                      const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
                      const DebugOverrides &debug, const MutationRequest &mutation,
                      const RuntimeCapabilities &capabilities) {
  const TransformExecutor executor =
      g_test_transform_override.load(std::memory_order_acquire) != nullptr
          ? run_automatic_test_executor
          : nullptr;
  return prepare_automatic_transform(bytes, request, transform_policy, runtime_policy, debug,
                                     mutation, capabilities, executor);
}

/// Accumulator used by the HSA runtime-capability query adapter. It owns no HSA
/// handles; the callback only normalizes region facts into the shared typed
/// contract.
struct RuntimeCapabilityRegionSearch {
  CoreApiTable *core = nullptr;
  RuntimeCapabilities *capabilities = nullptr;
};

hsa_status_t HSA_API collect_runtime_capability_region(hsa_region_t region, void *data) {
  auto *search = static_cast<RuntimeCapabilityRegionSearch *>(data);
  hsa_region_segment_t segment{};
  hsa_status_t status =
      search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_SEGMENT, &segment);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  if (segment == HSA_REGION_SEGMENT_GROUP) {
    size_t size_bytes = 0;
    status = search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_SIZE, &size_bytes);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    if (size_bytes != 0 && size_bytes <= std::numeric_limits<uint32_t>::max())
      search->capabilities->max_workgroup_lds_bytes = static_cast<uint32_t>(size_bytes);
    return HSA_STATUS_SUCCESS;
  }
  if (segment != HSA_REGION_SEGMENT_GLOBAL)
    return HSA_STATUS_SUCCESS;

  bool allocation_allowed = false;
  status = search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
                                                &allocation_allowed);
  if (status != HSA_STATUS_SUCCESS || !allocation_allowed)
    return status;
  size_t max_size = 0;
  status = search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_ALLOC_MAX_SIZE, &max_size);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  uint32_t flags = 0;
  status = search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  const bool fine_grained = (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0;
  const bool coarse_grained = (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) != 0;
  if (!fine_grained && !coarse_grained)
    return HSA_STATUS_SUCCESS;
  search->capabilities->host_device_visible_memory = true;
  search->capabilities->device_atomic_publication = true;
  search->capabilities->host_device_coherent_memory |= fine_grained;
  const uint64_t prior = search->capabilities->max_report_allocation_bytes.value_or(0);
  search->capabilities->max_report_allocation_bytes =
      std::max<uint64_t>(prior, static_cast<uint64_t>(max_size));
  return HSA_STATUS_SUCCESS;
}

[[nodiscard]] RuntimeCapabilities query_hsa_runtime_capabilities(CoreApiTable *core,
                                                                 hsa_agent_t agent) {
  RuntimeCapabilities capabilities;
  capabilities.backend = RuntimeBackend::PhysicalHsa;
  if (core == nullptr || core->hsa_agent_iterate_regions_fn == nullptr ||
      core->hsa_region_get_info_fn == nullptr) {
    return capabilities;
  }
  capabilities.executable_binding = core->hsa_code_object_reader_create_from_memory_fn != nullptr &&
                                    core->hsa_executable_load_agent_code_object_fn != nullptr;
  capabilities.dispatch_segment_binding = true;
  RuntimeCapabilityRegionSearch search{.core = core, .capabilities = &capabilities};
  if (core->hsa_agent_iterate_regions_fn(agent, collect_runtime_capability_region, &search) !=
      HSA_STATUS_SUCCESS) {
    RuntimeCapabilities unavailable;
    unavailable.backend = RuntimeBackend::PhysicalHsa;
    unavailable.executable_binding = capabilities.executable_binding;
    unavailable.dispatch_segment_binding = capabilities.dispatch_segment_binding;
    return unavailable;
  }
  return capabilities;
}

enum class WaitcheckPreflightOutcome { NotApplicable, Passed, HazardReported, AnalysisFailed };

void print_waitcheck_issue(uint64_t reader, const AmdGpuCodeObject &code_object,
                           const rocjitsu::WaitcheckReport &report) {
  std::lock_guard lock(log_mutex());
  if (!report.supported) {
    std::fprintf(stderr,
                 "rocjitsu-waitcheck: ConSan preflight reported reader=%llu target=%s "
                 "reason=analysis-failed action=continue",
                 static_cast<unsigned long long>(reader),
                 rj_code_target_name(code_object.target_id()));
    if (!report.analysis_error.empty())
      std::fprintf(stderr, ": %s", report.analysis_error.c_str());
    std::fprintf(stderr, "\n");
    return;
  }

  std::fprintf(stderr,
               "rocjitsu-waitcheck: ConSan preflight reported reader=%llu target=%s "
               "reason=wait-hazard diagnostics=%zu action=continue\n",
               static_cast<unsigned long long>(reader),
               rj_code_target_name(code_object.target_id()), report.diagnostics_observed);
  constexpr size_t kMaxDiagnostics = 32;
  const size_t limit = std::min(kMaxDiagnostics, report.diagnostics.size());
  for (size_t i = 0; i < limit; ++i) {
    const rocjitsu::WaitcheckDiagnostic &diagnostic = report.diagnostics[i];
    std::fprintf(stderr, "rocjitsu-waitcheck: %s+0x%llx: %s; producer %s+0x%llx: %s\n",
                 diagnostic.section_name.c_str(),
                 static_cast<unsigned long long>(diagnostic.section_offset),
                 diagnostic.message.c_str(), diagnostic.section_name.c_str(),
                 static_cast<unsigned long long>(diagnostic.producer_section_offset),
                 diagnostic.producer_instruction.c_str());
    std::fprintf(stderr, "rocjitsu-waitcheck:   consumer: %s\n", diagnostic.instruction.c_str());
  }
  if (report.diagnostics.size() > limit) {
    std::fprintf(stderr, "rocjitsu-waitcheck: omitted %zu additional diagnostic(s)\n",
                 report.diagnostics.size() - limit);
  } else if (report.diagnostics_truncated) {
    std::fprintf(stderr, "rocjitsu-waitcheck: omitted additional diagnostic(s) after limit\n");
  }
}

void print_waitcheck_exception(uint64_t reader, const std::exception *error) {
  std::lock_guard lock(log_mutex());
  std::fprintf(stderr,
               "rocjitsu-waitcheck: ConSan preflight reported reader=%llu "
               "reason=analysis-failed action=continue",
               static_cast<unsigned long long>(reader));
  if (error != nullptr)
    std::fprintf(stderr, ": %s", error->what());
  std::fprintf(stderr, "\n");
}

[[nodiscard]] WaitcheckPreflightOutcome
run_waitcheck_preflight(std::span<const uint8_t> bytes, uint64_t reader,
                        std::span<const std::string> kernel_name_allowlist = {}) {
  bool recognized_code_object = false;
  try {
    AmdGpuCodeObject code_object(bytes.data(), bytes.size());
    if (!code_object.is_valid()) {
      log_message(kLogVerbose,
                  "waitcheck preflight reader=%llu outcome=not-applicable reason=invalid-object",
                  static_cast<unsigned long long>(reader));
      return WaitcheckPreflightOutcome::NotApplicable;
    }

    recognized_code_object = true;
    const rj_code_arch_t arch = rocjitsu::waitcheck_arch_for_target(code_object.target_id());
    if (arch == ROCJITSU_CODE_ARCH_INVALID) {
      log_message(kLogVerbose,
                  "waitcheck preflight reader=%llu target=%s outcome=not-applicable "
                  "reason=unsupported-target",
                  static_cast<unsigned long long>(reader),
                  rj_code_target_name(code_object.target_id()));
      return WaitcheckPreflightOutcome::NotApplicable;
    }

    rocjitsu::WaitcheckOptions options;
    options.max_diagnostics = 32;
    size_t kernels_discovered = 0;
    rocjitsu::WaitcheckReport report;
    if (kernel_name_allowlist.empty()) {
      report = rocjitsu::analyze_waitcnts(code_object, arch, options);
      kernels_discovered = report.kernels_discovered;
    } else {
      const std::vector<rocjitsu::WaitcheckKernelInfo> kernels =
          rocjitsu::waitcheck_kernels(code_object);
      kernels_discovered = kernels.size();
      std::vector<rocjitsu::WaitcheckKernelInfo> selected_kernels;
      std::ranges::copy_if(kernels, std::back_inserter(selected_kernels), [&](const auto &kernel) {
        return std::ranges::any_of(kernel_name_allowlist, [&](const std::string &allowlisted) {
          return rocjitsu::kernel_symbol_names_match(kernel.name, allowlisted);
        });
      });

      // Kernel-name indexing and waitcheck descriptor discovery intentionally
      // use independent ELF readers. Preserve the conservative full-object
      // preflight if their views ever disagree instead of silently skipping it.
      report = selected_kernels.empty() ? rocjitsu::analyze_waitcnts(code_object, arch, options)
                                        : rocjitsu::analyze_waitcnts_for_kernels(
                                              code_object, arch, selected_kernels, options);
      if (selected_kernels.empty())
        kernels_discovered = report.kernels_discovered;
    }
    if (!report.supported) {
      print_waitcheck_issue(reader, code_object, report);
      return WaitcheckPreflightOutcome::AnalysisFailed;
    }
    if (!report.passed()) {
      print_waitcheck_issue(reader, code_object, report);
      return WaitcheckPreflightOutcome::HazardReported;
    }

    log_message(kLogInfo,
                "waitcheck preflight reader=%llu target=%s outcome=passed instructions=%zu "
                "memory_events=%zu kernels=%zu/%zu",
                static_cast<unsigned long long>(reader),
                rj_code_target_name(code_object.target_id()), report.instructions_analyzed,
                report.memory_events_tracked, report.kernels_analyzed, kernels_discovered);
    return WaitcheckPreflightOutcome::Passed;
  } catch (const std::exception &error) {
    if (!recognized_code_object) {
      log_message(kLogVerbose,
                  "waitcheck preflight reader=%llu outcome=not-applicable reason=parse-failed",
                  static_cast<unsigned long long>(reader));
      return WaitcheckPreflightOutcome::NotApplicable;
    }
    print_waitcheck_exception(reader, &error);
  } catch (...) {
    if (!recognized_code_object) {
      log_message(kLogVerbose,
                  "waitcheck preflight reader=%llu outcome=not-applicable reason=parse-failed",
                  static_cast<unsigned long long>(reader));
      return WaitcheckPreflightOutcome::NotApplicable;
    }
    print_waitcheck_exception(reader, nullptr);
  }
  return WaitcheckPreflightOutcome::AnalysisFailed;
}

[[nodiscard]] std::vector<ProbeIntentId> intent_ids_covering(const TransformResult &result,
                                                             const SemanticSiteId &semantic_site) {
  return result.coverage_ledger.intent_ids_covering(semantic_site);
}

[[nodiscard]] bool require_patch_applies_to(const TransformResult &result,
                                            const HookConfig &config) {
  const bool has_selected_access = config.probe_lds_check_trap || config.probe_flat_check_trap;
  if (!has_selected_access)
    return false;
  if (result.coverage_ledger.site_decisions().empty())
    return !result.observation_plan().valid();
  return std::ranges::any_of(
      result.coverage_ledger.site_decisions(), [&](const SiteDecision &decision) {
        if (decision.kind != SiteDecisionKind::Admitted)
          return false;
        const auto intent_ids = intent_ids_covering(result, decision.semantic_site);
        return std::ranges::any_of(intent_ids, [&](ProbeIntentId id) {
          const IntentCoverageEntry *entry = result.coverage_ledger.intent_entry(id);
          return entry == nullptr || entry->lowering != LoweringOutcomeKind::ResourceRejected;
        });
      });
}

[[nodiscard]] bool require_patch_applies_to(const TransformResult &result) {
  return !result.observation_plan().probe_intents.empty();
}

[[nodiscard]] bool has_instrumented_site(const TransformResult &result) {
  return std::ranges::any_of(result.coverage_ledger.intent_entries(),
                             [](const IntentCoverageEntry &entry) {
                               return entry.lowering == LoweringOutcomeKind::Instrumented;
                             });
}

struct StaticCoverageKind {
  uint64_t discovered = 0;
  uint64_t supported = 0;
  uint64_t selected = 0;
  uint64_t patched = 0;
  uint64_t unsupported = 0;
  uint64_t resource_failed = 0;
  uint64_t placement_or_lowering_failed = 0;
  uint64_t expert_limit_omitted = 0;
};

struct StaticCoverage {
  StaticCoverageKind access;
  StaticCoverageKind barrier;
  StaticCoverageKind atomic;
  StaticCoverageKind fence;
  bool complete = false;
  bool expert_limit = false;
};

void mark_static_coverage_uninstrumented(StaticCoverage &coverage) {
  coverage.access.patched = 0;
  coverage.barrier.patched = 0;
  coverage.atomic.patched = 0;
  coverage.fence.patched = 0;
  coverage.complete = false;
}

[[nodiscard]] StaticCoverageKind &coverage_kind(StaticCoverage &coverage, ResourceSiteKind kind) {
  switch (kind) {
  case ResourceSiteKind::Access:
    return coverage.access;
  case ResourceSiteKind::Barrier:
    return coverage.barrier;
  case ResourceSiteKind::Atomic:
    return coverage.atomic;
  case ResourceSiteKind::Fence:
    return coverage.fence;
  }
  return coverage.access;
}

void finalize_coverage_kind(StaticCoverageKind &kind, const HookConfig &config) {
  kind.unsupported = kind.discovered > kind.supported ? kind.discovered - kind.supported : 0;
  kind.selected = kind.supported;
  if (config.max_patches_explicit && kind.selected > config.max_patches) {
    kind.expert_limit_omitted = kind.selected - config.max_patches;
    kind.selected = config.max_patches;
  }
  const uint64_t accounted = kind.patched + kind.resource_failed + kind.expert_limit_omitted;
  kind.placement_or_lowering_failed = kind.supported > accounted ? kind.supported - accounted : 0;
}

[[nodiscard]] StaticCoverage compute_static_coverage(const CoverageLedger &ledger,
                                                     const HookConfig &config) {
  StaticCoverage coverage;
  coverage.expert_limit = config.max_patches_explicit;
  struct TypedSiteCoverage {
    ResourceSiteKind kind = ResourceSiteKind::Access;
    uint64_t text_offset = 0;
    bool supported = false;
    std::vector<ProbeIntentId> intents;
  };
  std::vector<TypedSiteCoverage> sites;
  constexpr size_t kResourceKindCount = 4u;
  std::array<std::unordered_map<uint64_t, size_t>, kResourceKindCount> site_indices;
  const auto append_decisions = [&](const auto &decisions, ResourceSiteKind kind) {
    const size_t kind_index = static_cast<size_t>(kind);
    if (kind_index >= site_indices.size())
      return;
    for (const auto &decision : decisions) {
      if (decision.kind == SiteDecisionKind::NotApplicable)
        continue;
      const uint64_t text_offset = decision.semantic_site.physical.original_text_offset;
      const auto [indexed, inserted] =
          site_indices[kind_index].try_emplace(text_offset, sites.size());
      if (inserted) {
        sites.push_back({
            .kind = kind,
            .text_offset = text_offset,
            .supported = false,
            .intents = {},
        });
      }
      TypedSiteCoverage &site = sites[indexed->second];
      if (decision.kind != SiteDecisionKind::Admitted)
        continue;
      site.supported = true;
      for (const ProbeIntentId id : ledger.intent_ids_covering(decision.semantic_site))
        if (std::ranges::find(site.intents, id) == site.intents.end())
          site.intents.push_back(id);
    }
  };
  append_decisions(ledger.site_decisions(), ResourceSiteKind::Access);
  append_decisions(ledger.barrier_site_decisions(), ResourceSiteKind::Barrier);
  append_decisions(ledger.atomic_site_decisions(), ResourceSiteKind::Atomic);
  append_decisions(ledger.fence_site_decisions(), ResourceSiteKind::Fence);
  for (const TypedSiteCoverage &site : sites) {
    StaticCoverageKind &kind = coverage_kind(coverage, site.kind);
    ++kind.discovered;
    if (!site.supported)
      continue;
    ++kind.supported;
    bool all_instrumented = !site.intents.empty();
    bool resource_rejected = false;
    for (ProbeIntentId id : site.intents) {
      const IntentCoverageEntry *entry = ledger.intent_entry(id);
      if (entry == nullptr || entry->lowering != LoweringOutcomeKind::Instrumented)
        all_instrumented = false;
      if (entry != nullptr && entry->lowering == LoweringOutcomeKind::ResourceRejected)
        resource_rejected = true;
    }
    if (all_instrumented)
      ++kind.patched;
    else if (resource_rejected)
      ++kind.resource_failed;
    else
      ++kind.placement_or_lowering_failed;
  }
  finalize_coverage_kind(coverage.access, config);
  finalize_coverage_kind(coverage.barrier, config);
  finalize_coverage_kind(coverage.atomic, config);
  finalize_coverage_kind(coverage.fence, config);
  const auto complete_kind = [](const StaticCoverageKind &kind) {
    return kind.unsupported == 0 && kind.supported == kind.patched && kind.resource_failed == 0 &&
           kind.placement_or_lowering_failed == 0 && kind.expert_limit_omitted == 0;
  };
  coverage.complete = complete_kind(coverage.access) && complete_kind(coverage.barrier) &&
                      complete_kind(coverage.atomic) && complete_kind(coverage.fence);
  return coverage;
}

class StaticCoverageRegistry {
public:
  struct Summary {
    uint64_t applicable_code_objects = 0;
    uint64_t incomplete_code_objects = 0;
    uint64_t supported_access = 0;
    uint64_t patched_access = 0;
    uint64_t supported_barrier = 0;
    uint64_t patched_barrier = 0;
    uint64_t supported_atomic = 0;
    uint64_t patched_atomic = 0;
    uint64_t supported_fence = 0;
    uint64_t patched_fence = 0;

    [[nodiscard]] bool complete() const {
      return applicable_code_objects != 0 && incomplete_code_objects == 0;
    }
  };

  static StaticCoverageRegistry &instance() {
    static StaticCoverageRegistry registry;
    return registry;
  }

  void record(const StaticCoverage &coverage) {
    const uint64_t discovered = coverage.access.discovered + coverage.barrier.discovered +
                                coverage.atomic.discovered + coverage.fence.discovered;
    if (discovered == 0)
      return;
    std::lock_guard lock(mutex_);
    ++summary_.applicable_code_objects;
    if (!coverage.complete)
      ++summary_.incomplete_code_objects;
    summary_.supported_access += coverage.access.supported;
    summary_.patched_access += coverage.access.patched;
    summary_.supported_barrier += coverage.barrier.supported;
    summary_.patched_barrier += coverage.barrier.patched;
    summary_.supported_atomic += coverage.atomic.supported;
    summary_.patched_atomic += coverage.atomic.patched;
    summary_.supported_fence += coverage.fence.supported;
    summary_.patched_fence += coverage.fence.patched;
  }

  /// Records a code object whose applicability could not be inventoried.
  ///
  /// Resource admission can intentionally reject a transform before semantic
  /// discovery. Treat that object as conservatively applicable and incomplete
  /// so fail-open execution cannot produce an apparently complete verdict.
  void record_unclassified_incomplete_code_object() {
    std::lock_guard lock(mutex_);
    ++summary_.applicable_code_objects;
    ++summary_.incomplete_code_objects;
  }

  Summary summarize_and_clear() {
    std::lock_guard lock(mutex_);
    const Summary result = summary_;
    summary_ = {};
    return result;
  }

  void clear() {
    std::lock_guard lock(mutex_);
    summary_ = {};
  }

private:
  std::mutex mutex_;
  Summary summary_;
};

/// Visit a well-formed ConSan evidence contract without admitting SuperCollider or
/// malformed requirements through the ConSan report-allocation path. The hook
/// uses this narrow adapter only to bind an already-planned contract; it never
/// reconstructs evidence requirements from mechanism telemetry.
template <typename Callback>
[[nodiscard]] bool
visit_evidence_requirements(const std::optional<EvidenceRequirements> &evidence_requirements,
                            Callback &&callback) {
  if (!evidence_requirements)
    return false;
  const auto visit_if_present = [&](const auto *requirements) {
    if (!requirements || !requirements->well_formed())
      return false;
    callback(*requirements);
    return true;
  };
  return visit_if_present(std::get_if<ReportRequirements>(&*evidence_requirements));
}

template <typename Callback>
[[nodiscard]] bool visit_evidence_requirements(const TransformResult &result, Callback &&callback) {
  return visit_evidence_requirements(result.evidence_requirements,
                                     std::forward<Callback>(callback));
}

std::mutex &log_mutex() {
  static std::mutex mutex;
  return mutex;
}

enum class ProcessFaultReservationOutcome : uint8_t {
  Reserved,
  MutationAlreadyInstalled,
  ContentionTimeout,
  ReentrantContention,
};

struct ProcessFaultReservationSummary {
  uint64_t reserved = 0;
  uint64_t mutation_already_installed = 0;
  uint64_t contention_timeout = 0;
  uint64_t reentrant_contention = 0;
  bool mutation_installed = false;
  bool reservation_active = false;

  [[nodiscard]] uint64_t attempts() const {
    uint64_t total = 0;
    for (const uint64_t count :
         {reserved, mutation_already_installed, contention_timeout, reentrant_contention}) {
      if (count > std::numeric_limits<uint64_t>::max() - total)
        return std::numeric_limits<uint64_t>::max();
      total += count;
    }
    return total;
  }

  [[nodiscard]] bool complete() const { return !reservation_active; }
};

struct ProcessFaultApplicationState {
  std::mutex mutex;
  std::condition_variable changed;
  bool reservation_active = false;
  bool mutation_installed = false;
  std::thread::id reservation_owner;
  ProcessFaultReservationSummary summary;
  bool exactly_one_requested = false;
  bool summary_taken = false;
};

struct ProcessFaultApplicationSnapshot {
  ProcessFaultReservationSummary reservation;
  bool exactly_one_requested = false;
};

ProcessFaultApplicationState &process_fault_application_state() {
  // Match the automatic-report registry's process-lifetime storage. A
  // reservation can still unwind through a runtime-owned load callback while
  // shared-library finalizers are running, so destroying its synchronization
  // primitives during ordinary static teardown is unsafe.
  static auto *state = new ProcessFaultApplicationState;
  return *state;
}

void reset_process_fault_application_state() {
  ProcessFaultApplicationState &state = process_fault_application_state();
  {
    std::lock_guard lock(state.mutex);
    state.reservation_active = false;
    state.mutation_installed = false;
    state.reservation_owner = {};
    state.summary = {};
    state.exactly_one_requested = false;
    state.summary_taken = false;
  }
  state.changed.notify_all();
}

[[nodiscard]] std::optional<ProcessFaultApplicationSnapshot>
take_process_fault_application_snapshot() {
  ProcessFaultApplicationState &state = process_fault_application_state();
  std::lock_guard lock(state.mutex);
  if (state.summary_taken)
    return std::nullopt;
  state.summary_taken = true;
  ProcessFaultApplicationSnapshot result;
  result.reservation = state.summary;
  result.reservation.mutation_installed = state.mutation_installed;
  result.reservation.reservation_active = state.reservation_active;
  result.exactly_one_requested = state.exactly_one_requested;
  return result;
}

void observe_process_fault_requirement(bool require_exactly_one) {
  if (!require_exactly_one)
    return;
  ProcessFaultApplicationState &state = process_fault_application_state();
  std::lock_guard lock(state.mutex);
  state.exactly_one_requested = true;
}

// Requires state.mutex to be held by the caller.
void record_process_fault_reservation_outcome(ProcessFaultApplicationState &state,
                                              ProcessFaultReservationOutcome outcome) {
  const auto increment = [](uint64_t &count) {
    if (count != std::numeric_limits<uint64_t>::max())
      ++count;
  };
  switch (outcome) {
  case ProcessFaultReservationOutcome::Reserved:
    increment(state.summary.reserved);
    return;
  case ProcessFaultReservationOutcome::MutationAlreadyInstalled:
    increment(state.summary.mutation_already_installed);
    return;
  case ProcessFaultReservationOutcome::ContentionTimeout:
    increment(state.summary.contention_timeout);
    return;
  case ProcessFaultReservationOutcome::ReentrantContention:
    increment(state.summary.reentrant_contention);
    return;
  }
}

[[nodiscard]] constexpr std::string_view
process_fault_reservation_outcome_name(ProcessFaultReservationOutcome outcome) {
  using E = ProcessFaultReservationOutcome;
  constexpr auto vocabulary =
      make_enum_vocabulary("unknown", enum_entry(E::Reserved, "reserved"),
                           enum_entry(E::MutationAlreadyInstalled, "mutation-already-installed"),
                           enum_entry(E::ContentionTimeout, "contention-timeout"),
                           enum_entry(E::ReentrantContention, "reentrant-contention"));
  return vocabulary.name(outcome);
}

class ProcessFaultApplicationReservation {
public:
  ProcessFaultApplicationReservation() = default;
  ProcessFaultApplicationReservation(const ProcessFaultApplicationReservation &) = delete;
  ProcessFaultApplicationReservation &
  operator=(const ProcessFaultApplicationReservation &) = delete;

  ~ProcessFaultApplicationReservation() { release(); }

  [[nodiscard]] ProcessFaultReservationOutcome reserve(std::chrono::milliseconds wait,
                                                       size_t *prior_applications) {
    ProcessFaultApplicationState &state = process_fault_application_state();
    std::unique_lock lock(state.mutex);
    *prior_applications = state.mutation_installed ? 1u : 0u;
    if (state.reservation_active && state.reservation_owner == std::this_thread::get_id()) {
      constexpr ProcessFaultReservationOutcome outcome =
          ProcessFaultReservationOutcome::ReentrantContention;
      record_process_fault_reservation_outcome(state, outcome);
      return outcome;
    }
    // This hook must not create an unbounded process-wide blocking edge inside
    // an interposed loader call. On timeout the contender loads unmodified and
    // the fault harness attributes any resulting zero-application trial.
    if (!state.changed.wait_for(lock, wait, [&] { return !state.reservation_active; })) {
      *prior_applications = state.mutation_installed ? 1u : 0u;
      constexpr ProcessFaultReservationOutcome outcome =
          ProcessFaultReservationOutcome::ContentionTimeout;
      record_process_fault_reservation_outcome(state, outcome);
      return outcome;
    }
    *prior_applications = state.mutation_installed ? 1u : 0u;
    if (state.mutation_installed) {
      constexpr ProcessFaultReservationOutcome outcome =
          ProcessFaultReservationOutcome::MutationAlreadyInstalled;
      record_process_fault_reservation_outcome(state, outcome);
      return outcome;
    }
    state.reservation_active = true;
    state.reservation_owner = std::this_thread::get_id();
    reserved_ = true;
    constexpr ProcessFaultReservationOutcome outcome = ProcessFaultReservationOutcome::Reserved;
    record_process_fault_reservation_outcome(state, outcome);
    return outcome;
  }

  void commit_applied_mutation() {
    if (!reserved_)
      return;
    ProcessFaultApplicationState &state = process_fault_application_state();
    {
      std::lock_guard lock(state.mutex);
      state.reservation_active = false;
      state.mutation_installed = true;
      state.reservation_owner = {};
      reserved_ = false;
    }
    state.changed.notify_all();
  }

private:
  void release() {
    if (!reserved_)
      return;
    ProcessFaultApplicationState &state = process_fault_application_state();
    {
      std::lock_guard lock(state.mutex);
      state.reservation_active = false;
      state.reservation_owner = {};
      reserved_ = false;
    }
    state.changed.notify_all();
  }

  bool reserved_ = false;
};

[[nodiscard]] bool fault_mutations_enabled(const MutationRequest &request) {
  return request.has_fault_mutation();
}

void disable_fault_mutations(MutationRequest *request) {
  *request = without_fault_mutations(*request);
}

namespace {

constexpr std::string_view kLogPrefix = "[rocjitsu-dbi-hooks] ";
constexpr size_t kDetailedLogBatchBytes = 1024u * 1024u;

void write_log_bytes_locked(std::string_view bytes) {
  if (const LogSinkOverride sink = g_test_log_sink_override.load(std::memory_order_acquire)) {
    sink(bytes.data(), bytes.size());
    return;
  }
  std::fwrite(bytes.data(), 1, bytes.size(), stderr);
}

[[nodiscard]] std::string format_log_line(const char *format, va_list args) {
  va_list measure_args;
  va_copy(measure_args, args);
  const int payload_size = std::vsnprintf(nullptr, 0, format, measure_args);
  va_end(measure_args);
  if (payload_size < 0)
    return std::string(kLogPrefix) + "log formatting failed\n";

  std::string line(kLogPrefix);
  const size_t payload_offset = line.size();
  line.resize(payload_offset + static_cast<size_t>(payload_size) + 1u);
  std::vsnprintf(line.data() + payload_offset, static_cast<size_t>(payload_size) + 1u, format,
                 args);
  line.back() = '\n';
  return line;
}

class ScopedDetailedLogBatch;
thread_local ScopedDetailedLogBatch *g_active_detailed_log_batch = nullptr;

class ScopedDetailedLogBatch {
public:
  ScopedDetailedLogBatch() : previous_(g_active_detailed_log_batch) {
    if (previous_ != nullptr)
      previous_->flush();
    g_active_detailed_log_batch = this;
  }

  ScopedDetailedLogBatch(const ScopedDetailedLogBatch &) = delete;
  ScopedDetailedLogBatch &operator=(const ScopedDetailedLogBatch &) = delete;

  ~ScopedDetailedLogBatch() {
    flush();
    g_active_detailed_log_batch = previous_;
  }

  void append(std::string_view line) {
    if (line.size() > kDetailedLogBatchBytes) {
      flush();
      std::lock_guard lock(log_mutex());
      write_log_bytes_locked(line);
      return;
    }
    if (buffer_.size() + line.size() > kDetailedLogBatchBytes)
      flush();
    buffer_.append(line);
  }

  void flush() {
    if (buffer_.empty())
      return;
    std::lock_guard lock(log_mutex());
    write_log_bytes_locked(buffer_);
    buffer_.clear();
  }

private:
  ScopedDetailedLogBatch *previous_ = nullptr;
  std::string buffer_;
};

void flush_detailed_log_batch() {
  if (g_active_detailed_log_batch != nullptr)
    g_active_detailed_log_batch->flush();
}

} // namespace

void log_message(int required_level, const char *format, ...) {
  if (g_log_level.load(std::memory_order_relaxed) < required_level)
    return;

  va_list args;
  va_start(args, format);
  std::string line = format_log_line(format, args);
  va_end(args);

  if (g_active_detailed_log_batch != nullptr) {
    g_active_detailed_log_batch->append(line);
    return;
  }
  std::lock_guard lock(log_mutex());
  write_log_bytes_locked(line);
}

void emit_evidence_message(const char *format, ...) {
  flush_detailed_log_batch();
  std::lock_guard lock(log_mutex());
  std::fprintf(stderr, "[rocjitsu-dbi-hooks] ");

  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);

  std::fprintf(stderr, "\n");
  std::fflush(stderr);
}

void emit_fault_summary_message(bool required_evidence, const char *format, ...) {
  if (!required_evidence && g_log_level.load(std::memory_order_relaxed) < kLogInfo)
    return;
  flush_detailed_log_batch();
  std::lock_guard lock(log_mutex());
  std::fprintf(stderr, "[rocjitsu-dbi-hooks] ");

  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);

  std::fprintf(stderr, "\n");
  if (required_evidence)
    std::fflush(stderr);
}

void emit_process_fault_reservation_summary(const ProcessFaultReservationSummary &summary) {
  emit_evidence_message(
      "ConSan fault reservation summary process=%llu attempts=%llu reserved=%llu "
      "mutation_already_installed=%llu contention_timeout=%llu reentrant_contention=%llu "
      "mutation_installed=%s active=%s complete=%s",
      static_cast<unsigned long long>(::getpid()),
      static_cast<unsigned long long>(summary.attempts()),
      static_cast<unsigned long long>(summary.reserved),
      static_cast<unsigned long long>(summary.mutation_already_installed),
      static_cast<unsigned long long>(summary.contention_timeout),
      static_cast<unsigned long long>(summary.reentrant_contention),
      summary.mutation_installed ? "true" : "false", summary.reservation_active ? "true" : "false",
      summary.complete() ? "true" : "false");
}

class FaultInstallationEvidence {
public:
  explicit FaultInstallationEvidence(uint64_t reader) : reader_(reader) {}
  FaultInstallationEvidence(const FaultInstallationEvidence &) = delete;
  FaultInstallationEvidence &operator=(const FaultInstallationEvidence &) = delete;

  ~FaultInstallationEvidence() { emit(); }

  void emit() {
    if (emitted_)
      return;
    emitted_ = true;
    if (applied_ == 0)
      return;
    emit_evidence_message("ConSan fault install process=%llu reader=%llu applied=%zu installed=%s",
                          static_cast<unsigned long long>(::getpid()),
                          static_cast<unsigned long long>(reader_), applied_,
                          installed_ ? "true" : "false");
    std::fflush(stderr);
  }

  void record_applied_mutations(size_t applied) { applied_ = applied; }
  void mark_installed() { installed_ = true; }

private:
  uint64_t reader_ = 0;
  size_t applied_ = 0;
  bool installed_ = false;
  bool emitted_ = false;
};

void dump_code_object_bytes(const HookConfig &config, uint64_t dump_id, uint64_t reader,
                            std::string_view tag, std::span<const uint8_t> bytes) {
  if (config.dump_dir.empty() || bytes.empty())
    return;

  if (::mkdir(config.dump_dir.c_str(), 0755) != 0 && errno != EEXIST) {
    log_message(kLogInfo, "failed to create RJ_CONSAN_DUMP_DIR='%s': %s", config.dump_dir.c_str(),
                std::strerror(errno));
    return;
  }

  std::array<char, 4096> path{};
  const int written = std::snprintf(
      path.data(), path.size(), "%s/rj-dbi-%06llu-reader-%llu-%.*s.hsaco", config.dump_dir.c_str(),
      static_cast<unsigned long long>(dump_id), static_cast<unsigned long long>(reader),
      static_cast<int>(tag.size()), tag.data());
  if (written < 0 || static_cast<size_t>(written) >= path.size()) {
    log_message(kLogInfo, "RJ_CONSAN_DUMP_DIR path is too long: %s", config.dump_dir.c_str());
    return;
  }

  FILE *file = std::fopen(path.data(), "wb");
  if (file == nullptr) {
    log_message(kLogInfo, "failed to open DBI dump '%s': %s", path.data(), std::strerror(errno));
    return;
  }
  const size_t stored = std::fwrite(bytes.data(), 1, bytes.size(), file);
  const int close_status = std::fclose(file);
  if (stored != bytes.size() || close_status != 0) {
    log_message(kLogInfo, "failed to write complete DBI dump '%s'", path.data());
    return;
  }

  log_message(kLogInfo, "dumped DBI %.*s code object reader=%llu bytes=%zu path=%s",
              static_cast<int>(tag.size()), tag.data(), static_cast<unsigned long long>(reader),
              bytes.size(), path.data());
}

using rocjitsu::hooks::HsaCodeObjectReaderRegistry;

/// ConSan owns its reader namespace independently from any co-loaded HSA tool.
HsaCodeObjectReaderRegistry &code_object_reader_registry() {
  static HsaCodeObjectReaderRegistry registry;
  return registry;
}

/// Admits conservative major image working-set bounds for concurrent transforms.
///
/// The reservation is acquired before the first semantic inventory pass and
/// remains live until the final replacement bytes are either retained by the
/// executable registry or discarded. Multiple inventory/retry passes for one
/// reader share one reservation rather than each receiving a fresh allowance.
class ProcessTransformAdmissionRegistry {
public:
  enum class AdmissionOutcome : uint8_t {
    Admitted,
    LimitExceeded,
    AccountingOverflow,
  };

  struct AdmissionResult {
    AdmissionOutcome outcome = AdmissionOutcome::AccountingOverflow;
    uint64_t live_bytes = 0;
    uint64_t reservation_bytes = 0;
    std::optional<uint64_t> required_bytes;
    std::optional<uint64_t> limit_bytes;

    [[nodiscard]] explicit operator bool() const { return outcome == AdmissionOutcome::Admitted; }
  };

  static ProcessTransformAdmissionRegistry &instance() {
    // A reservation can unwind through a runtime-owned load callback while
    // shared-library finalizers are running. Keep the synchronization state
    // alive for the process lifetime.
    static auto *registry = new ProcessTransformAdmissionRegistry;
    return *registry;
  }

  [[nodiscard]] AdmissionResult admit(uint64_t reservation_bytes,
                                      std::optional<uint64_t> limit_bytes) {
    std::lock_guard lock(mutex_);
    const ProcessByteBudget::ChargePlan plan = budget_.plan_charge(reservation_bytes, limit_bytes);
    AdmissionOutcome outcome;
    switch (plan.outcome) {
    case ProcessByteBudget::ChargeOutcome::WithinLimit:
      outcome = AdmissionOutcome::Admitted;
      break;
    case ProcessByteBudget::ChargeOutcome::LimitExceeded:
      outcome = AdmissionOutcome::LimitExceeded;
      break;
    case ProcessByteBudget::ChargeOutcome::AccountingOverflow:
      outcome = AdmissionOutcome::AccountingOverflow;
      break;
    }
    const AdmissionResult result = {
        .outcome = outcome,
        .live_bytes = plan.live_bytes,
        .reservation_bytes = reservation_bytes,
        .required_bytes = plan.required_bytes,
        .limit_bytes = limit_bytes,
    };
    if (result)
      budget_.commit_charge(plan);
    return result;
  }

  void release(uint64_t reservation_bytes) {
    std::lock_guard lock(mutex_);
    if (!budget_.refund(reservation_bytes)) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] ConSan internal invariant violation: "
                           "transform reservation refund exceeded the live total\n");
    }
  }

  [[nodiscard]] ProcessByteBudget::Summary summarize_and_rollover() {
    std::lock_guard lock(mutex_);
    const ProcessByteBudget::Summary summary = budget_.summary();
    // OnUnload does not quiesce runtime-owned load callbacks. Preserve their
    // live charges across reinstall while starting a fresh peak interval.
    budget_.reset_peak_to_live();
    return summary;
  }

private:
  std::mutex mutex_;
  ProcessByteBudget budget_;
};

class ProcessTransformReservation {
public:
  ProcessTransformReservation() = default;
  ProcessTransformReservation(const ProcessTransformReservation &) = delete;
  ProcessTransformReservation &operator=(const ProcessTransformReservation &) = delete;

  ~ProcessTransformReservation() { release(); }

  [[nodiscard]] ProcessTransformAdmissionRegistry::AdmissionResult
  acquire(uint64_t reservation_bytes, std::optional<uint64_t> limit_bytes) {
    release();
    ProcessTransformAdmissionRegistry &registry = ProcessTransformAdmissionRegistry::instance();
    const ProcessTransformAdmissionRegistry::AdmissionResult result =
        registry.admit(reservation_bytes, limit_bytes);
    if (result) {
      registry_ = &registry;
      reservation_bytes_ = reservation_bytes;
    }
    return result;
  }

  void release() {
    if (registry_ == nullptr)
      return;
    registry_->release(reservation_bytes_);
    registry_ = nullptr;
    reservation_bytes_ = 0;
  }

  void discard_image_and_release(std::vector<uint8_t> &image) {
    // clear() and initializer-list assignment may retain capacity. Swap with an
    // explicit empty vector so the backing allocation is gone before its
    // accounting reservation is refunded.
    std::vector<uint8_t>{}.swap(image);
    release();
  }

private:
  ProcessTransformAdmissionRegistry *registry_ = nullptr;
  uint64_t reservation_bytes_ = 0;
};

/// Couples every load-scoped major-image owner to its admission reservation.
///
/// The reservation is declared before every load-scoped owner so reverse member
/// destruction refunds it only after those owners are destroyed. Explicit
/// release paths likewise discard relevant image storage before refunding.
class TransformLoadState {
private:
  // Declared first so reverse member destruction releases this reservation
  // after every present and future load-scoped owner below.
  ProcessTransformReservation reservation_;

public:
  TransformLoadState() = default;
  TransformLoadState(const TransformLoadState &) = delete;
  TransformLoadState &operator=(const TransformLoadState &) = delete;

  [[nodiscard]] ProcessTransformAdmissionRegistry::AdmissionResult
  acquire(uint64_t reservation_bytes, std::optional<uint64_t> limit_bytes) {
    return reservation_.acquire(reservation_bytes, limit_bytes);
  }

  void release() { reservation_.release(); }

  void discard_image_and_release(std::vector<uint8_t> &image) {
    reservation_.discard_image_and_release(image);
  }

  std::shared_ptr<const std::vector<uint8_t>> replacement_storage;
  std::optional<TransformResult> patch_result_storage;
  std::optional<StaticCoverage> static_coverage_storage;
  std::optional<DeferredBinding> deferred_binding;
  std::optional<AutoReportInventory> live_fault_auto_report_capacity_inventory;
};

/// Owns replacement memory for as long as the executable can expose it.
///
/// HSA loaded-code-object introspection reports the original memory base and
/// size even after the temporary reader used for loading has been destroyed.
/// ROCProfiler consumes that storage when the executable is frozen, which can
/// happen after hsa_executable_load_agent_code_object() returns. Therefore the
/// hook must not tie replacement storage to the load call's transform result. The
/// registry reserves configured process growth and full-image budgets in the
/// same critical section that retains the storage, and refunds both when the
/// storage is released.
class ReplacementCodeObjectStorageRegistry {
public:
  enum class RetainOutcome : uint8_t {
    Retained,
    ProcessGrowthLimitExceeded,
    ProcessImageLimitExceeded,
    GrowthAccountingOverflow,
    ImageAccountingOverflow,
    AllocationFailure,
  };

  struct RetainResult {
    RetainOutcome outcome = RetainOutcome::AllocationFailure;
    uint64_t live_growth_bytes = 0;
    uint64_t live_image_bytes = 0;
    uint64_t replacement_growth_bytes = 0;
    uint64_t replacement_image_bytes = 0;
    std::optional<uint64_t> required_total_growth_bytes;
    std::optional<uint64_t> required_total_image_bytes;
    std::optional<uint64_t> growth_limit_bytes;
    std::optional<uint64_t> image_limit_bytes;

    [[nodiscard]] explicit operator bool() const { return outcome == RetainOutcome::Retained; }
  };

  struct Summary {
    ProcessByteBudget::Summary growth;
    ProcessByteBudget::Summary image;
  };

  static ReplacementCodeObjectStorageRegistry &instance() {
    // The HSA runtime may call OnUnload from a shared-library finalizer after
    // ordinary function-local statics have already been destroyed. Keep both
    // the registry and retained executable storage alive for the process
    // lifetime; executable destruction observed while the hook is active
    // remains the ownership release point.
    static auto *registry = new ReplacementCodeObjectStorageRegistry;
    return *registry;
  }

  [[nodiscard]] RetainResult retain(hsa_executable_t executable,
                                    std::shared_ptr<const std::vector<uint8_t>> storage,
                                    uint64_t replacement_growth_bytes, uint64_t replacement_size,
                                    std::optional<uint64_t> process_growth_limit_bytes,
                                    std::optional<uint64_t> process_image_limit_bytes) {
    std::lock_guard lock(mutex_);
    const ProcessByteBudget::ChargePlan growth_plan =
        growth_budget_.plan_charge(replacement_growth_bytes, process_growth_limit_bytes);
    const ProcessByteBudget::ChargePlan image_plan =
        image_budget_.plan_charge(replacement_size, process_image_limit_bytes);
    RetainOutcome outcome;
    switch (growth_plan.outcome) {
    case ProcessByteBudget::ChargeOutcome::WithinLimit:
      outcome = RetainOutcome::Retained;
      break;
    case ProcessByteBudget::ChargeOutcome::LimitExceeded:
      outcome = RetainOutcome::ProcessGrowthLimitExceeded;
      break;
    case ProcessByteBudget::ChargeOutcome::AccountingOverflow:
      outcome = process_growth_limit_bytes ? RetainOutcome::ProcessGrowthLimitExceeded
                                           : RetainOutcome::GrowthAccountingOverflow;
      break;
    }
    if (outcome == RetainOutcome::Retained) {
      switch (image_plan.outcome) {
      case ProcessByteBudget::ChargeOutcome::WithinLimit:
        break;
      case ProcessByteBudget::ChargeOutcome::LimitExceeded:
        outcome = RetainOutcome::ProcessImageLimitExceeded;
        break;
      case ProcessByteBudget::ChargeOutcome::AccountingOverflow:
        outcome = process_image_limit_bytes ? RetainOutcome::ProcessImageLimitExceeded
                                            : RetainOutcome::ImageAccountingOverflow;
        break;
      }
    }
    const RetainResult result = {
        .outcome = outcome,
        .live_growth_bytes = growth_plan.live_bytes,
        .live_image_bytes = image_plan.live_bytes,
        .replacement_growth_bytes = replacement_growth_bytes,
        .replacement_image_bytes = replacement_size,
        .required_total_growth_bytes = growth_plan.required_bytes,
        .required_total_image_bytes = image_plan.required_bytes,
        .growth_limit_bytes = process_growth_limit_bytes,
        .image_limit_bytes = process_image_limit_bytes,
    };
    if (result.outcome != RetainOutcome::Retained)
      return result;

    try {
      auto entry = std::ranges::find(entries_, executable.handle, &Entry::executable);
      if (entry == entries_.end()) {
        Entry new_entry{.executable = executable.handle, .objects = {}};
        new_entry.objects.push_back({.storage = std::move(storage),
                                     .growth_bytes = replacement_growth_bytes,
                                     .image_bytes = replacement_size});
        entries_.push_back(std::move(new_entry));
      } else {
        entry->objects.push_back({.storage = std::move(storage),
                                  .growth_bytes = replacement_growth_bytes,
                                  .image_bytes = replacement_size});
      }
    } catch (const std::bad_alloc &) {
      RetainResult failure = result;
      failure.outcome = RetainOutcome::AllocationFailure;
      return failure;
    } catch (const std::length_error &) {
      RetainResult failure = result;
      failure.outcome = RetainOutcome::AllocationFailure;
      return failure;
    }
    growth_budget_.commit_charge(growth_plan);
    image_budget_.commit_charge(image_plan);
    return result;
  }

  void release(hsa_executable_t executable, const std::vector<uint8_t> *storage) {
    std::lock_guard lock(mutex_);
    const auto entry = std::ranges::find(entries_, executable.handle, &Entry::executable);
    if (entry == entries_.end())
      return;
    const auto object = std::ranges::find(entry->objects, storage,
                                          [](const Object &value) { return value.storage.get(); });
    if (object != entry->objects.end()) {
      refund_object(*object);
      entry->objects.erase(object);
    }
    if (entry->objects.empty())
      entries_.erase(entry);
  }

  void remove(hsa_executable_t executable) {
    std::lock_guard lock(mutex_);
    const auto entry = std::ranges::find(entries_, executable.handle, &Entry::executable);
    if (entry != entries_.end()) {
      for (const Object &object : entry->objects)
        refund_object(object);
      entries_.erase(entry);
    }
  }

  [[nodiscard]] Summary summarize_and_rollover() {
    std::lock_guard lock(mutex_);
    const Summary summary = {
        .growth = growth_budget_.summary(),
        .image = image_budget_.summary(),
    };
    // OnUnload does not quiesce runtime-owned load callbacks or invalidate
    // executables that already refer to these bytes. Preserve ownership and
    // live charges across reinstall while starting a fresh peak interval. New
    // HSA calls made after OnUnload are outside the hook lifetime; a destroy
    // only in that interval cannot be observed and leaves this process-lifetime
    // storage charged until exit.
    growth_budget_.reset_peak_to_live();
    image_budget_.reset_peak_to_live();
    return summary;
  }

private:
  struct Object {
    std::shared_ptr<const std::vector<uint8_t>> storage;
    uint64_t growth_bytes = 0;
    uint64_t image_bytes = 0;
  };

  struct Entry {
    uint64_t executable = 0;
    std::vector<Object> objects;
  };

  void refund_object(const Object &object) {
    if (!growth_budget_.refund(object.growth_bytes)) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] ConSan internal invariant violation: "
                           "replacement growth refund exceeded the live total\n");
    }
    if (!image_budget_.refund(object.image_bytes)) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] ConSan internal invariant violation: "
                           "replacement image refund exceeded the live total\n");
    }
  }

  std::mutex mutex_;
  std::vector<Entry> entries_;
  ProcessByteBudget growth_budget_;
  ProcessByteBudget image_budget_;
};

class AutoSuperColliderReportBufferRegistry {
public:
  struct Summary {
    uint64_t buffer_count = 0;
    uint64_t mismatch_count = 0;
    uint64_t allocation_failure_count = 0;
    uint64_t read_failure_count = 0;
    uint64_t cleanup_failure_count = 0;

    [[nodiscard]] bool complete() const {
      return allocation_failure_count == 0 && read_failure_count == 0 && cleanup_failure_count == 0;
    }
  };

  static AutoSuperColliderReportBufferRegistry &instance() {
    static auto *registry = new AutoSuperColliderReportBufferRegistry;
    return *registry;
  }

  [[nodiscard]] bool allocate(CoreApiTable *core, hsa_agent_t agent, uint64_t reader,
                              uint64_t *address, uint64_t *registered_generation) {
    std::lock_guard lock(mutex_);
    if (entry_count_ >= entries_.size()) {
      ++allocation_failure_count_;
      log_message(kLogDebug,
                  "ConSan SuperCollider auto report allocation reader=%llu outcome=failed "
                  "reason=registry_full",
                  static_cast<unsigned long long>(reader));
      return false;
    }
    const detail::AutoReportAllocation allocation =
        detail::allocate_auto_report_memory(core, agent, sizeof(uint32_t));
    if (!allocation) {
      ++allocation_failure_count_;
      log_message(kLogInfo,
                  "ConSan SuperCollider auto report allocation reader=%llu outcome=failed "
                  "reason=%s status=%d",
                  static_cast<unsigned long long>(reader), allocation.failure_reason,
                  static_cast<int>(allocation.status));
      return false;
    }
    void *ptr = allocation.data;
    const uint64_t generation = next_generation_.fetch_add(1, std::memory_order_relaxed) + 1u;
    entries_[entry_count_++] = Entry{reader, generation, ptr, allocation.fine_grained};
    *address = reinterpret_cast<uint64_t>(ptr);
    if (registered_generation != nullptr)
      *registered_generation = generation;
    log_message(kLogInfo,
                "ConSan SuperCollider auto report buffer reader=%llu addr=0x%llx bytes=%zu "
                "allocation_outcome=allocated fine_grained=%s",
                static_cast<unsigned long long>(reader), static_cast<unsigned long long>(*address),
                sizeof(uint32_t), allocation.fine_grained ? "true" : "false");
    return true;
  }

  void bind_to_executable(uint64_t reader, uint64_t generation, hsa_executable_t executable) {
    std::lock_guard lock(mutex_);
    detail::bind_auto_report_entry(entries_, entry_count_, reader, generation, executable.handle);
  }

  void discard(CoreApiTable *core, uint64_t reader, uint64_t generation) {
    std::lock_guard lock(mutex_);
    detail::discard_auto_report_entry(
        entries_, entry_count_, reader, generation, [&](Entry &entry) {
          return release_entry(core, entry, /*allow_runtime_reclaimed=*/false);
        });
  }

  void retire(CoreApiTable *core, hsa_executable_t executable) {
    std::lock_guard lock(mutex_);
    detail::retire_auto_report_entries(
        entries_, entry_count_, executable.handle, retired_summary_,
        [&](const Entry &entry) { return summarize_entry(core, entry); },
        [&](Entry &entry) { return release_entry(core, entry, /*allow_runtime_reclaimed=*/false); },
        accumulate_summary);
  }

  Summary summarize_and_clear(CoreApiTable *core) {
    std::lock_guard lock(mutex_);
    Summary summary = retired_summary_;
    summary.allocation_failure_count = allocation_failure_count_;
    for (size_t index = 0; index < entry_count_; ++index) {
      Entry &entry = entries_[index];
      accumulate_summary(summary, summarize_entry(core, entry));
      if (!release_entry(core, entry, /*allow_runtime_reclaimed=*/true))
        ++summary.cleanup_failure_count;
      entry = {};
    }
    entry_count_ = 0;
    allocation_failure_count_ = 0;
    retired_summary_ = {};
    return summary;
  }

private:
  struct Entry {
    uint64_t reader = 0;
    uint64_t generation = 0;
    void *ptr = nullptr;
    bool fine_grained = false;
    uint64_t executable = 0;
    bool executable_bound = false;
  };

  [[nodiscard]] Summary summarize_entry(CoreApiTable *core, const Entry &entry) const {
    Summary summary{.buffer_count = 1};
    uint32_t marker = 0;
    bool readable = false;
    if (entry.fine_grained) {
      std::memcpy(&marker, entry.ptr, sizeof(marker));
      readable = true;
    } else if (core != nullptr && core->hsa_memory_copy_fn != nullptr) {
      const hsa_status_t status = core->hsa_memory_copy_fn(&marker, entry.ptr, sizeof(marker));
      readable = status == HSA_STATUS_SUCCESS;
    }
    if (!readable) {
      ++summary.read_failure_count;
      log_message(
          kLogInfo,
          "ConSan SuperCollider auto report reader=%llu outcome=unreadable mismatch=unknown",
          static_cast<unsigned long long>(entry.reader));
    } else {
      summary.mismatch_count = marker != 0;
      log_message(
          kLogInfo,
          "ConSan SuperCollider auto report reader=%llu outcome=complete marker=%u mismatch=%s",
          static_cast<unsigned long long>(entry.reader), marker, marker != 0 ? "true" : "false");
    }
    return summary;
  }

  [[nodiscard]] bool release_entry(CoreApiTable *core, const Entry &entry,
                                   bool allow_runtime_reclaimed) const {
    bool freed = entry.ptr == nullptr;
    hsa_status_t free_status = HSA_STATUS_SUCCESS;
    if (!freed && allow_runtime_reclaimed) {
      // OnUnload is a callback from inside ROCR shutdown. Do not re-enter an
      // HSA allocation API while the runtime may hold its shutdown locks;
      // ROCR reclaims the allocation after the callback returns.
      freed = true;
      log_message(kLogInfo,
                  "ConSan SuperCollider auto report cleanup reader=%llu "
                  "outcome=runtime-reclaimed",
                  static_cast<unsigned long long>(entry.reader));
    } else if (!freed && (core == nullptr || core->hsa_memory_free_fn == nullptr)) {
      // Non-shutdown cleanup must retain the entry for a later retry.
    } else if (!freed) {
      free_status = core->hsa_memory_free_fn(entry.ptr);
      freed = free_status == HSA_STATUS_SUCCESS ||
              free_status == HSA_STATUS_ERROR_INVALID_ALLOCATION ||
              free_status == HSA_STATUS_ERROR_NOT_INITIALIZED;
    }
    if (!freed) {
      log_message(kLogInfo,
                  "ConSan SuperCollider auto report cleanup reader=%llu outcome=failed status=%d",
                  static_cast<unsigned long long>(entry.reader), static_cast<int>(free_status));
    }
    return freed;
  }

  static void accumulate_summary(Summary &total, const Summary &entry) {
    total.buffer_count += entry.buffer_count;
    total.mismatch_count += entry.mismatch_count;
    total.read_failure_count += entry.read_failure_count;
    total.cleanup_failure_count += entry.cleanup_failure_count;
  }

  std::mutex mutex_;
  std::array<Entry, 256> entries_{};
  size_t entry_count_ = 0;
  uint64_t allocation_failure_count_ = 0;
  Summary retired_summary_;
  std::atomic<uint64_t> next_generation_{0};
};

class AutoReportLoadGuard {
public:
  explicit AutoReportLoadGuard(CoreApiTable *core) : core_(core) {}
  AutoReportLoadGuard(const AutoReportLoadGuard &) = delete;
  AutoReportLoadGuard &operator=(const AutoReportLoadGuard &) = delete;

  ~AutoReportLoadGuard() {
    if (supercollider_)
      AutoSuperColliderReportBufferRegistry::instance().discard(core_, supercollider_->reader,
                                                                supercollider_->generation);
    if (report_)
      discard_report_buffer(core_, report_->reader, report_->generation);
  }

  void note_sc(uint64_t reader, uint64_t generation) {
    supercollider_ = Registration{reader, generation};
  }
  void note(uint64_t reader, uint64_t generation) { report_ = Registration{reader, generation}; }

  void bind_to_executable(hsa_executable_t executable) {
    if (supercollider_) {
      AutoSuperColliderReportBufferRegistry::instance().bind_to_executable(
          supercollider_->reader, supercollider_->generation, executable);
      supercollider_.reset();
    }
    if (report_) {
      bind_report_buffer_to_executable(report_->reader, report_->generation, executable);
      report_.reset();
    }
  }

private:
  struct Registration {
    uint64_t reader = 0;
    uint64_t generation = 0;
  };

  CoreApiTable *core_ = nullptr;
  std::optional<Registration> supercollider_;
  std::optional<Registration> report_;
};

class KernelPrivateDispatchRegistry {
public:
  struct DispatchRequirements {
    uint32_t required_private_bytes = 0;
    uint32_t dynamic_private_addend = 0;
    uint32_t required_group_bytes = 0;
    bool instrumented = false;

    [[nodiscard]] bool has_segment_requirement() const {
      return required_private_bytes != 0u || dynamic_private_addend != 0u ||
             required_group_bytes != 0u;
    }
  };

  struct DispatchSummary {
    uint64_t packet_count = 0;
    uint64_t instrumented_packet_count = 0;
  };

  struct AllowlistEntrySummary {
    std::string kernel_name;
    bool loaded = false;
    bool instrumented = false;
    uint64_t dispatch_count = 0;
  };

  static KernelPrivateDispatchRegistry &instance() {
    // The HSA runtime may call OnUnload from a shared-library finalizer after
    // ordinary function-local statics have already been destroyed. Keep the
    // registry alive for the process lifetime and clear its contents explicitly
    // when the hook layer is uninstalled.
    static auto *registry = new KernelPrivateDispatchRegistry;
    return *registry;
  }

  void configure_allowlist(std::span<const std::string> kernel_names) {
    std::lock_guard lock(mutex_);
    allowlist_.clear();
    allowlist_.reserve(kernel_names.size());
    for (const std::string &kernel_name : kernel_names)
      allowlist_.push_back({kernel_name});
  }

  void note_code_object(const TransformResult &result) {
    std::lock_guard lock(mutex_);
    for (AllowlistEntry &entry : allowlist_) {
      entry.loaded |=
          std::ranges::any_of(result.program_inventory.kernels(), [&](const auto &kernel) {
            return rocjitsu::kernel_symbol_names_match(kernel.name, entry.kernel_name);
          });
    }
  }

  void note_requirements(hsa_executable_t executable,
                         const consan::DispatchRequirements &requirements) {
    std::lock_guard lock(mutex_);
    for (const KernelDispatchRequirement &requirement : requirements.kernels) {
      const auto allowlisted = std::ranges::find_if(allowlist_, [&](const AllowlistEntry &entry) {
        return rocjitsu::kernel_symbol_names_match(entry.kernel_name, requirement.kernel_name);
      });
      if (allowlisted != allowlist_.end())
        allowlisted->instrumented |= requirement.has_instrumented_probe;
      const auto pending = std::ranges::find_if(pending_, [&](const Pending &candidate) {
        return candidate.executable == executable.handle &&
               candidate.kernel_name == requirement.kernel_name;
      });
      if (pending == pending_.end()) {
        pending_.push_back({executable.handle, requirement.kernel_name,
                            requirement.required_private_bytes, requirement.dynamic_private_addend,
                            requirement.required_group_bytes, requirement.has_instrumented_probe});
        continue;
      }
      pending->required_private_bytes =
          std::max(pending->required_private_bytes, requirement.required_private_bytes);
      pending->dynamic_private_addend =
          std::max(pending->dynamic_private_addend, requirement.dynamic_private_addend);
      pending->required_group_bytes =
          std::max(pending->required_group_bytes, requirement.required_group_bytes);
      pending->has_instrumented_probe |= requirement.has_instrumented_probe;
    }
  }

  void bind_symbol(hsa_executable_t executable, std::string_view symbol_name,
                   hsa_executable_symbol_t symbol,
                   decltype(hsa_executable_symbol_get_info) *original_get_info) {
    if (original_get_info == nullptr)
      return;
    std::lock_guard lock(mutex_);
    const auto pending = std::ranges::find_if(pending_, [&](const Pending &candidate) {
      return candidate.executable == executable.handle &&
             rocjitsu::kernel_symbol_names_match(candidate.kernel_name, symbol_name);
    });
    const auto allowlisted = std::ranges::find_if(allowlist_, [&](const AllowlistEntry &entry) {
      return rocjitsu::kernel_symbol_names_match(entry.kernel_name, symbol_name);
    });
    if (pending == pending_.end() && allowlisted == allowlist_.end())
      return;

    uint64_t kernel_object = 0;
    if (original_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object) !=
        HSA_STATUS_SUCCESS) {
      return;
    }
    const Bound observed{
        .executable = executable.handle,
        .symbol = symbol.handle,
        .kernel_object = kernel_object,
        .required_private_bytes = pending == pending_.end() ? 0u : pending->required_private_bytes,
        .dynamic_private_addend = pending == pending_.end() ? 0u : pending->dynamic_private_addend,
        .required_group_bytes = pending == pending_.end() ? 0u : pending->required_group_bytes,
        .has_instrumented_probe = pending != pending_.end() && pending->has_instrumented_probe,
        .allowlisted_kernel_name =
            allowlisted == allowlist_.end() ? std::string() : allowlisted->kernel_name,
    };
    const auto bound = std::ranges::find_if(
        bound_, [&](const Bound &candidate) { return candidate.symbol == symbol.handle; });
    if (bound == bound_.end()) {
      bound_.push_back(observed);
    } else if (bound->executable != executable.handle) {
      *bound = observed;
    } else {
      bound->kernel_object = kernel_object;
      if (pending != pending_.end()) {
        bound->required_private_bytes =
            std::max(bound->required_private_bytes, pending->required_private_bytes);
        bound->dynamic_private_addend =
            std::max(bound->dynamic_private_addend, pending->dynamic_private_addend);
        bound->required_group_bytes =
            std::max(bound->required_group_bytes, pending->required_group_bytes);
        bound->has_instrumented_probe |= pending->has_instrumented_probe;
      }
      if (allowlisted != allowlist_.end())
        bound->allowlisted_kernel_name = allowlisted->kernel_name;
    }
    if (pending != pending_.end() &&
        (pending->required_private_bytes != 0u || pending->dynamic_private_addend != 0u ||
         pending->required_group_bytes != 0u)) {
      log_message(
          kLogInfo,
          "ConSan dispatch-segment binding executable=%llu symbol=%llu kernel_object=0x%llx "
          "private_bytes=%u dynamic_private_addend=%u group_bytes=%u",
          static_cast<unsigned long long>(executable.handle),
          static_cast<unsigned long long>(symbol.handle),
          static_cast<unsigned long long>(kernel_object), pending->required_private_bytes,
          pending->dynamic_private_addend, pending->required_group_bytes);
    }
  }

  void erase_executable(hsa_executable_t executable) {
    std::lock_guard lock(mutex_);
    std::erase_if(pending_, [&](const Pending &candidate) {
      return candidate.executable == executable.handle;
    });
    std::erase_if(
        bound_, [&](const Bound &candidate) { return candidate.executable == executable.handle; });
  }

  [[nodiscard]] std::optional<uint32_t> required_for_symbol(hsa_executable_symbol_t symbol) const {
    std::lock_guard lock(mutex_);
    const auto bound = std::ranges::find_if(
        bound_, [&](const Bound &candidate) { return candidate.symbol == symbol.handle; });
    return bound == bound_.end() || bound->required_private_bytes == 0u
               ? std::nullopt
               : std::optional<uint32_t>(bound->required_private_bytes);
  }

  [[nodiscard]] std::optional<uint32_t>
  required_group_for_symbol(hsa_executable_symbol_t symbol) const {
    std::lock_guard lock(mutex_);
    const auto bound = std::ranges::find_if(
        bound_, [&](const Bound &candidate) { return candidate.symbol == symbol.handle; });
    return bound == bound_.end() || bound->required_group_bytes == 0u
               ? std::nullopt
               : std::optional<uint32_t>(bound->required_group_bytes);
  }

  [[nodiscard]] DispatchRequirements note_and_query_dispatch(uint64_t kernel_object) {
    std::lock_guard lock(mutex_);
    if (dispatch_packet_count_ != std::numeric_limits<uint64_t>::max())
      ++dispatch_packet_count_;
    const auto bound = std::ranges::find_if(
        bound_, [&](const Bound &candidate) { return candidate.kernel_object == kernel_object; });
    if (bound == bound_.end())
      return {};
    if (!bound->allowlisted_kernel_name.empty()) {
      const auto allowlisted = std::ranges::find(allowlist_, bound->allowlisted_kernel_name,
                                                 &AllowlistEntry::kernel_name);
      if (allowlisted != allowlist_.end() &&
          allowlisted->dispatch_count != std::numeric_limits<uint64_t>::max()) {
        ++allowlisted->dispatch_count;
      }
    }
    if (bound->has_instrumented_probe &&
        instrumented_dispatch_count_ != std::numeric_limits<uint64_t>::max())
      ++instrumented_dispatch_count_;
    return {
        .required_private_bytes = bound->required_private_bytes,
        .dynamic_private_addend = bound->dynamic_private_addend,
        .required_group_bytes = bound->required_group_bytes,
        .instrumented = bound->has_instrumented_probe,
    };
  }

  [[nodiscard]] DispatchSummary dispatch_summary() const {
    std::lock_guard lock(mutex_);
    return {
        .packet_count = dispatch_packet_count_,
        .instrumented_packet_count = instrumented_dispatch_count_,
    };
  }

  [[nodiscard]] std::vector<AllowlistEntrySummary> allowlist_summary() const {
    std::lock_guard lock(mutex_);
    std::vector<AllowlistEntrySummary> summary;
    summary.reserve(allowlist_.size());
    for (const AllowlistEntry &entry : allowlist_) {
      summary.push_back(
          {entry.kernel_name, entry.loaded, entry.instrumented, entry.dispatch_count});
    }
    return summary;
  }

  void clear() {
    std::lock_guard lock(mutex_);
    pending_.clear();
    bound_.clear();
    allowlist_.clear();
    dispatch_packet_count_ = 0;
    instrumented_dispatch_count_ = 0;
  }

private:
  struct AllowlistEntry {
    std::string kernel_name;
    bool loaded = false;
    bool instrumented = false;
    uint64_t dispatch_count = 0;
  };
  struct Pending {
    uint64_t executable = 0;
    std::string kernel_name;
    uint32_t required_private_bytes = 0;
    uint32_t dynamic_private_addend = 0;
    uint32_t required_group_bytes = 0;
    bool has_instrumented_probe = false;
  };
  struct Bound {
    uint64_t executable = 0;
    uint64_t symbol = 0;
    uint64_t kernel_object = 0;
    uint32_t required_private_bytes = 0;
    uint32_t dynamic_private_addend = 0;
    uint32_t required_group_bytes = 0;
    bool has_instrumented_probe = false;
    std::string allowlisted_kernel_name;
  };

  mutable std::mutex mutex_;
  std::vector<Pending> pending_;
  std::vector<Bound> bound_;
  std::vector<AllowlistEntry> allowlist_;
  uint64_t dispatch_packet_count_ = 0;
  uint64_t instrumented_dispatch_count_ = 0;
};

/// Detects host-observable points at which every instrumented dispatch known
/// to the packet interceptor has completed. Submission and recycling share one
/// gate so a new report writer cannot appear between the quiescence check and
/// the report reset.
class ReportEpochRegistry {
public:
  static ReportEpochRegistry &instance() {
    static auto *registry = new ReportEpochRegistry;
    return *registry;
  }

  [[nodiscard]] std::unique_lock<std::mutex> lock_submission() {
    return std::unique_lock<std::mutex>(mutex_);
  }

  void note_instrumented_dispatch_locked(void *queue, hsa_signal_t signal,
                                         decltype(hsa_signal_load_scacquire) *load_signal) {
    if (signal.handle == 0u) {
      note_queue_awaiting_completion_proxy_locked(queue);
      return;
    }
    if (!note_completion_signal_locked(signal, load_signal))
      note_queue_awaiting_completion_proxy_locked(queue);
  }

  void note_ordering_barrier_locked(void *queue, bool orders_prior_packets,
                                    hsa_signal_t completion_signal,
                                    decltype(hsa_signal_load_scacquire) *load_signal) {
    if (queue == nullptr || !orders_prior_packets)
      return;
    const auto awaiting = std::ranges::find(queues_awaiting_completion_proxy_, queue);
    if (awaiting == queues_awaiting_completion_proxy_.end())
      return;
    if (completion_signal.handle == 0u ||
        !note_completion_signal_locked(completion_signal, load_signal))
      return;
    queues_awaiting_completion_proxy_.erase(awaiting);
  }

  void note_queue_awaiting_completion_proxy_locked(void *queue) {
    if (queue == nullptr) {
      untrackable_dispatch_ = true;
    } else if (std::ranges::find(queues_awaiting_completion_proxy_, queue) ==
               queues_awaiting_completion_proxy_.end()) {
      queues_awaiting_completion_proxy_.push_back(queue);
    }
  }

  [[nodiscard]] bool
  note_completion_signal_locked(hsa_signal_t signal,
                                decltype(hsa_signal_load_scacquire) *load_signal) {
    if (load_signal == nullptr)
      return false;
    const hsa_signal_value_t current = load_signal(signal);
    if (current == std::numeric_limits<hsa_signal_value_t>::min())
      return false;
    auto pending = std::ranges::find(pending_, signal.handle, &Pending::signal_handle);
    if (pending == pending_.end()) {
      pending_.push_back({signal.handle, current - 1});
    } else if (current <= pending->target_value) {
      // The signal was completed and then reused without an intervening wait
      // visible to the hook. Start tracking its new countdown from the value
      // observed at this submission.
      pending->target_value = current - 1;
    } else if (pending->target_value == std::numeric_limits<hsa_signal_value_t>::min()) {
      return false;
    } else {
      --pending->target_value;
    }
    return true;
  }

  template <typename Checkpoint>
  bool checkpoint_if_quiescent(decltype(hsa_signal_load_scacquire) *load_signal,
                               Checkpoint &&checkpoint) {
    std::lock_guard lock(mutex_);
    if (untrackable_dispatch_ || !queues_awaiting_completion_proxy_.empty() || pending_.empty() ||
        load_signal == nullptr)
      return false;
    if (!std::ranges::all_of(pending_, [&](const Pending &pending) {
          return load_signal(hsa_signal_t{pending.signal_handle}) <= pending.target_value;
        })) {
      return false;
    }
    const ReportCheckpointResult result = checkpoint();
    if (result.status != EpochCheckpointStatus::Complete)
      return false;
    log_message(kLogVerbose,
                "ConSan automatic ConSan epoch checkpoint outcome=complete dispatch_signals=%zu "
                "reports=%llu",
                pending_.size(), static_cast<unsigned long long>(result.report_count));
    pending_.clear();
    return true;
  }

  void forget_signal(hsa_signal_t signal, decltype(hsa_signal_load_scacquire) *load_signal) {
    std::lock_guard lock(mutex_);
    const auto pending = std::ranges::find(pending_, signal.handle, &Pending::signal_handle);
    if (pending == pending_.end())
      return;
    if (load_signal == nullptr || load_signal(signal) > pending->target_value)
      untrackable_dispatch_ = true;
    pending_.erase(pending);
  }

  void clear_after_explicit_checkpoint() {
    std::lock_guard lock(mutex_);
    pending_.clear();
    queues_awaiting_completion_proxy_.clear();
    untrackable_dispatch_ = false;
  }

  void clear() {
    std::lock_guard lock(mutex_);
    pending_.clear();
    queues_awaiting_completion_proxy_.clear();
    untrackable_dispatch_ = false;
  }

private:
  struct Pending {
    uint64_t signal_handle = 0;
    hsa_signal_value_t target_value = 0;
  };

  std::mutex mutex_;
  std::vector<Pending> pending_;
  std::vector<void *> queues_awaiting_completion_proxy_;
  bool untrackable_dispatch_ = false;
};

hsa_status_t HSA_API rj_dbi_code_object_reader_create_from_memory(
    const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader);
hsa_status_t HSA_API rj_dbi_code_object_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t *code_object_reader);
hsa_status_t HSA_API rj_dbi_loader_code_object_reader_create_from_file_with_offset_size(
    hsa_file_t file, size_t offset, size_t size, hsa_code_object_reader_t *code_object_reader);
hsa_status_t HSA_API rj_dbi_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader);
hsa_status_t HSA_API rj_dbi_signal_destroy(hsa_signal_t signal);
hsa_signal_value_t HSA_API rj_dbi_signal_wait_relaxed(hsa_signal_t signal,
                                                      hsa_signal_condition_t condition,
                                                      hsa_signal_value_t compare_value,
                                                      uint64_t timeout_hint,
                                                      hsa_wait_state_t wait_state_hint);
hsa_signal_value_t HSA_API rj_dbi_signal_wait_scacquire(hsa_signal_t signal,
                                                        hsa_signal_condition_t condition,
                                                        hsa_signal_value_t compare_value,
                                                        uint64_t timeout_hint,
                                                        hsa_wait_state_t wait_state_hint);
hsa_status_t HSA_API rj_dbi_system_get_extension_table(uint16_t extension, uint16_t version_major,
                                                       uint16_t version_minor, void *table);
hsa_status_t HSA_API rj_dbi_system_get_major_extension_table(uint16_t extension,
                                                             uint16_t version_major,
                                                             size_t table_length, void *table);
hsa_status_t HSA_API rj_dbi_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader,
    const char *options, hsa_loaded_code_object_t *loaded_code_object);
hsa_status_t HSA_API rj_dbi_executable_destroy(hsa_executable_t executable);
hsa_status_t HSA_API rj_dbi_executable_get_symbol(hsa_executable_t executable,
                                                  const char *module_name, const char *symbol_name,
                                                  hsa_agent_t agent, int32_t call_convention,
                                                  hsa_executable_symbol_t *symbol);
hsa_status_t HSA_API rj_dbi_executable_get_symbol_by_name(hsa_executable_t executable,
                                                          const char *symbol_name,
                                                          const hsa_agent_t *agent,
                                                          hsa_executable_symbol_t *symbol);
hsa_status_t HSA_API rj_dbi_executable_iterate_symbols(
    hsa_executable_t executable,
    hsa_status_t (*callback)(hsa_executable_t, hsa_executable_symbol_t, void *), void *data);
hsa_status_t HSA_API rj_dbi_executable_iterate_agent_symbols(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t, void *),
    void *data);
hsa_status_t HSA_API rj_dbi_executable_symbol_get_info(hsa_executable_symbol_t symbol,
                                                       hsa_executable_symbol_info_t attribute,
                                                       void *value);
hsa_status_t HSA_API rj_dbi_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                         void (*callback)(hsa_status_t, hsa_queue_t *, void *),
                                         void *data, uint32_t private_segment_size,
                                         uint32_t group_segment_size, hsa_queue_t **queue);
hsa_status_t HSA_API rj_dbi_amd_queue_create(hsa_agent_t agent, hsa_amd_queue_create_desc_t *descs,
                                             uint32_t num_descs);

#define RJ_DBI_HSA_CORE_FUNCTIONS(X)                                                               \
  X(create_from_file, hsa_code_object_reader_create_from_file_fn,                                  \
    rj_dbi_code_object_reader_create_from_file,                                                    \
    decltype(hsa_code_object_reader_create_from_file) *, true)                                     \
  X(create_from_memory, hsa_code_object_reader_create_from_memory_fn,                              \
    rj_dbi_code_object_reader_create_from_memory,                                                  \
    decltype(hsa_code_object_reader_create_from_memory) *, true)                                   \
  X(destroy, hsa_code_object_reader_destroy_fn, rj_dbi_code_object_reader_destroy,                 \
    decltype(hsa_code_object_reader_destroy) *, true)                                              \
  X(signal_destroy, hsa_signal_destroy_fn, rj_dbi_signal_destroy, decltype(hsa_signal_destroy) *,  \
    intercept_dispatch_packets_)                                                                   \
  X(signal_wait_relaxed, hsa_signal_wait_relaxed_fn, rj_dbi_signal_wait_relaxed,                   \
    decltype(hsa_signal_wait_relaxed) *, intercept_dispatch_packets_)                              \
  X(signal_wait_scacquire, hsa_signal_wait_scacquire_fn, rj_dbi_signal_wait_scacquire,             \
    decltype(hsa_signal_wait_scacquire) *, intercept_dispatch_packets_)                            \
  X(get_extension_table, hsa_system_get_extension_table_fn, rj_dbi_system_get_extension_table,     \
    decltype(hsa_system_get_extension_table) *, true)                                              \
  X(get_major_extension_table, hsa_system_get_major_extension_table_fn,                            \
    rj_dbi_system_get_major_extension_table, decltype(hsa_system_get_major_extension_table) *,     \
    true)                                                                                          \
  X(load_agent_code_object, hsa_executable_load_agent_code_object_fn,                              \
    rj_dbi_executable_load_agent_code_object, decltype(hsa_executable_load_agent_code_object) *,   \
    true)                                                                                          \
  X(executable_destroy, hsa_executable_destroy_fn, rj_dbi_executable_destroy,                      \
    decltype(hsa_executable_destroy) *, true)                                                      \
  X(get_symbol, hsa_executable_get_symbol_fn, rj_dbi_executable_get_symbol,                        \
    decltype(hsa_executable_get_symbol) *,                                                         \
    intercept_dispatch_segments_ &&get_symbol_.original() != nullptr)                              \
  X(get_symbol_by_name, hsa_executable_get_symbol_by_name_fn,                                      \
    rj_dbi_executable_get_symbol_by_name, decltype(hsa_executable_get_symbol_by_name) *,           \
    intercept_dispatch_segments_)                                                                  \
  X(iterate_symbols, hsa_executable_iterate_symbols_fn, rj_dbi_executable_iterate_symbols,         \
    decltype(hsa_executable_iterate_symbols) *,                                                    \
    intercept_dispatch_segments_ &&iterate_symbols_.original() != nullptr)                         \
  X(iterate_agent_symbols, hsa_executable_iterate_agent_symbols_fn,                                \
    rj_dbi_executable_iterate_agent_symbols, decltype(hsa_executable_iterate_agent_symbols) *,     \
    intercept_dispatch_segments_)                                                                  \
  X(symbol_get_info, hsa_executable_symbol_get_info_fn, rj_dbi_executable_symbol_get_info,         \
    decltype(hsa_executable_symbol_get_info) *, intercept_dispatch_segments_)                      \
  X(queue_create, hsa_queue_create_fn, rj_dbi_queue_create, decltype(hsa_queue_create) *,          \
    intercept_dispatch_packets_)

class RjDbiHsaLayer {
public:
  bool install(HsaApiTable *table, HookConfig config) {
    std::lock_guard lock(mutex_);
    if (active_) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] OnLoad called while hook is already active\n");
      return false;
    }
    if (!validate_table(table))
      return false;

    table_ = table;
    core_ = table->core_;
    amd_ext_ = table->amd_ext_;
    g_log_level.store(config.log_level, std::memory_order_relaxed);
    reset_process_fault_application_state();
    config_ = config;
    require_records_ = config.require_records;
    require_diagnostics_ = config.require_diagnostics;
    forbid_diagnostics_ = config.forbid_diagnostics;
    forbid_overflow_ = config.forbid_overflow;
    runtime_sample_stride_ = config.runtime_sample_stride;
    runtime_sample_offset_ = config.runtime_sample_offset;
    configure_epoch_analysis(config.epoch_analysis, config.conflict_limit,
                             config.total_conflict_limit, config.allow_uniform_lds_stores);
    log_message(kLogInfo, "ConSan allow_uniform_lds_stores=%s",
                config.allow_uniform_lds_stores ? "true" : "false");
    if (config.mode == Mode::Default) {
      const auto cell = config.cell_selector();
      log_message(kLogInfo,
                  "ConSan configuration static_stride=%u static_offset=%u "
                  "workgroup_stride=%u workgroup_offset=%u cell_stride=%u cell_offset=%u "
                  "selection=%s requested_banks=%s report_ceiling=%llu epoch_analysis=%s "
                  "per_report_limit=%u session_limit=%u preset=%s",
                  config.sample_stride, config.sample_offset, config.runtime_sample_stride,
                  config.runtime_sample_offset, cell.stride, cell.offset,
                  config.cell_selection ? "independent" : "legacy-coupled",
                  config.watchpoint_banks == 0 ? "auto"
                                               : std::to_string(config.watchpoint_banks).c_str(),
                  static_cast<unsigned long long>(config.auto_report_buffer_size),
                  epoch_analysis_policy_name(config.epoch_analysis).c_str(), config.conflict_limit,
                  config.total_conflict_limit, config.preset);
    }
    fault_load_selector_.reset();
    if (config.fault_load_occurrence)
      fault_load_selector_.emplace(*config.fault_load_occurrence);
#define RJ_DBI_CAPTURE_CORE(name, field, wrapper, type, install_if) name##_.capture(&core_->field);
    RJ_DBI_HSA_CORE_FUNCTIONS(RJ_DBI_CAPTURE_CORE)
#undef RJ_DBI_CAPTURE_CORE
    const bool amd_queue_create_table_valid =
        amd_ext_ != nullptr &&
        amd_ext_->version.minor_id >= offsetof(AmdExtTable, hsa_amd_queue_create_fn) +
                                          sizeof(AmdExtTable::hsa_amd_queue_create_fn);
    if (amd_queue_create_table_valid)
      amd_queue_create_.capture(&amd_ext_->hsa_amd_queue_create_fn);
    intercept_dispatch_segments_ = config.mode.value_or(Mode::None) != Mode::None;
    const bool require_dispatch_packets = config.mode.value_or(Mode::None) == Mode::Default;
    const bool amd_intercept_table_valid =
        amd_ext_ != nullptr &&
        amd_ext_->version.minor_id >= offsetof(AmdExtTable, hsa_amd_queue_intercept_register_fn) +
                                          sizeof(AmdExtTable::hsa_amd_queue_intercept_register_fn);
    intercept_dispatch_packets_ =
        intercept_dispatch_segments_ && queue_create_.original() != nullptr &&
        amd_intercept_table_valid && amd_ext_->hsa_amd_queue_intercept_create_fn != nullptr &&
        amd_ext_->hsa_amd_queue_intercept_register_fn != nullptr;

    if (create_from_file_.original() == nullptr || create_from_memory_.original() == nullptr ||
        destroy_.original() == nullptr || get_extension_table_.original() == nullptr ||
        get_major_extension_table_.original() == nullptr ||
        load_agent_code_object_.original() == nullptr ||
        executable_destroy_.original() == nullptr ||
        (intercept_dispatch_segments_ && (get_symbol_by_name_.original() == nullptr ||
                                          iterate_agent_symbols_.original() == nullptr ||
                                          symbol_get_info_.original() == nullptr)) ||
        (intercept_dispatch_packets_ &&
         (signal_destroy_.original() == nullptr || signal_wait_relaxed_.original() == nullptr ||
          signal_wait_scacquire_.original() == nullptr ||
          core_->hsa_signal_load_scacquire_fn == nullptr)) ||
        (require_dispatch_packets && !intercept_dispatch_packets_)) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] HSA API table lacks a required DBI entry\n");
      clear_unlocked();
      return false;
    }

#define RJ_DBI_INSTALL_CORE(name, field, wrapper, type, install_if)                                \
  if (install_if)                                                                                  \
    name##_.install(wrapper);
    RJ_DBI_HSA_CORE_FUNCTIONS(RJ_DBI_INSTALL_CORE)
#undef RJ_DBI_INSTALL_CORE
    if (intercept_dispatch_packets_ && amd_queue_create_.original() != nullptr)
      amd_queue_create_.install(rj_dbi_amd_queue_create);
    KernelPrivateDispatchRegistry::instance().configure_allowlist(config.kernel_name_allowlist);
    ReportEpochRegistry::instance().clear();
    active_ = true;
    StaticCoverageRegistry::instance().clear();

    log_message(
        kLogInfo,
        "installed ConSan hook mode=%s policy=%s profile=%s supercollider_delay_nops=%u "
        "fail_closed=%s "
        "require_patch=%s "
        "check_trap_mode=%s supercollider_report_mode=%s probe_lds_check_trap=%s "
        "probe_flat_check_trap=%s "
        "fault_drop_barrier=%s fault_reservation_timeout_ms=%u "
        "init_owner_epoch=%s track_barriers=%s "
        "track_atomics=%s require_records=%s "
        "require_diagnostics=%s forbid_diagnostics=%s "
        "forbid_overflow=%s "
        "fault_barrier_index=%u "
        "supercollider_delay_mode=%s supercollider_delay_var_ssrc=%u "
        "patched_image_growth_limit_kind=%s patched_image_growth_limit_value=%llu "
        "process_concurrent_transform_limit_bytes=%s "
        "process_patched_image_limit_bytes=%s "
        "process_patched_image_growth_limit_bytes=%s "
        "max_patches=%u max_patches_source=%s tmp_vgpr=%s exec_save_sgpr=%s "
        "owner_source=%s flat_provenance=%s owner_sgpr=%s owner_vgpr=%s "
        "epoch_vgpr=%s "
        "runtime_sample_stride=%u runtime_sample_stride_source=%s "
        "epoch_analysis=%s "
        "report_buffer=%s report_buffer_size=%llu "
        "auto_report_buffer_size=%llu auto_report_buffer_size_source=%s mode=%s",
        mode_name(config.mode.value_or(Mode::None)), hook_policy_name(config.policy),
        config.mode == Mode::Default ? kStandardProfile.data() : "none",
        config.supercollider_delay_nops, config.fail_closed ? "true" : "false",
        config.require_patch ? "true" : "false", check_trap_mode_name(config.check_trap_mode),
        supercollider_report_mode_name(config.supercollider_report_mode),
        config.probe_lds_check_trap ? "true" : "false",
        config.probe_flat_check_trap ? "true" : "false",
        config.fault_drop_barrier ? "true" : "false", config.fault_reservation_timeout_ms,
        config.init_owner_epoch ? "true" : "false", config.track_barriers ? "true" : "false",
        config.track_atomics ? "true" : "false", config.require_records ? "true" : "false",
        config.require_diagnostics ? "true" : "false", config.forbid_diagnostics ? "true" : "false",
        config.forbid_overflow ? "true" : "false", config.fault_barrier_index,
        delay_mode_name(config.supercollider_delay_mode), config.supercollider_delay_var_ssrc,
        patched_image_growth_limit_kind_name(config.patched_image_growth_limit.kind),
        static_cast<unsigned long long>(
            patched_image_growth_limit_value(config.patched_image_growth_limit)),
        config.process_concurrent_transform_limit_bytes
            ? std::to_string(*config.process_concurrent_transform_limit_bytes).c_str()
            : "unlimited",
        config.process_patched_image_limit_bytes
            ? std::to_string(*config.process_patched_image_limit_bytes).c_str()
            : "unlimited",
        config.process_patched_image_growth_limit_bytes
            ? std::to_string(*config.process_patched_image_growth_limit_bytes).c_str()
            : "unlimited",
        config.max_patches, config.max_patches_explicit ? "expert-limit" : "all-supported-default",
        config.scratch_vgpr ? std::to_string(*config.scratch_vgpr).c_str() : "auto",
        config.requested_exec_save_sgpr ? std::to_string(*config.requested_exec_save_sgpr).c_str()
                                        : "unset",
        owner_source_name(config.owner_source),
        flat_provenance_mode_name(config.flat_provenance_mode),
        config.requested_owner_sgpr ? std::to_string(*config.requested_owner_sgpr).c_str()
                                    : "unset",
        config.requested_owner_vgpr ? std::to_string(*config.requested_owner_vgpr).c_str()
                                    : "unset",
        config.requested_epoch_vgpr ? std::to_string(*config.requested_epoch_vgpr).c_str()
                                    : "unset",
        config.runtime_sample_stride,
        config.runtime_sample_stride_explicit          ? "expert-override"
        : std::string_view(config.preset) != "default" ? "preset"
                                                       : "standard-profile",
        epoch_analysis_policy_name(config.epoch_analysis).c_str(),
        config.report_buffer_address ? std::to_string(*config.report_buffer_address).c_str()
                                     : "disabled",
        static_cast<unsigned long long>(config.report_buffer_size),
        static_cast<unsigned long long>(config.auto_report_buffer_size),
        config.auto_report_buffer_size_explicit ? "explicit_cap" : "inventory_ceiling",
        config.fault_drop_barrier
            ? (config.probe_lds_check_trap && config.probe_flat_check_trap
                   ? "proof-check-trap-all+fault-drop-barrier"
               : config.probe_lds_check_trap  ? "proof-lds-check-trap+fault-drop-barrier"
               : config.probe_flat_check_trap ? "proof-flat-check-trap+fault-drop-barrier"
                                              : "fault-drop-barrier")
        : config.probe_lds_check_trap && config.probe_flat_check_trap ? "proof-check-trap-all"
        : config.probe_lds_check_trap                                 ? "proof-lds-check-trap"
        : config.probe_flat_check_trap                                ? "proof-flat-check-trap"
                                                                      : "pass-through");
    if (config.fault_allow_destructive_incomplete_barrier_drop) {
      log_message(kLogInfo, "ConSan destructive control incomplete_barrier_drop=true "
                            "containment=external-runner-required");
    }
    if (!config.dump_dir.empty())
      log_message(kLogInfo, "DBI code-object dumps enabled dir=%s", config.dump_dir.c_str());
    if (!config.kernel_name_allowlist.empty())
      log_message(kLogInfo, "ConSan kernel allowlist enabled entries=%zu",
                  config.kernel_name_allowlist.size());
    return true;
  }

  [[nodiscard]] EpochCheckpointStatus checkpoint_after_device_synchronize() {
    std::lock_guard lock(mutex_);
    if (!active_ || core_ == nullptr || !config_)
      return EpochCheckpointStatus::Inactive;
    if (config_->mode != Mode::Default)
      return EpochCheckpointStatus::ModeDoesNotUseReports;
    const EpochCheckpointStatus status =
        checkpoint_report_buffers_after_device_synchronize(core_).status;
    if (status == EpochCheckpointStatus::Complete)
      ReportEpochRegistry::instance().clear_after_explicit_checkpoint();
    return status;
  }

  [[nodiscard]] EpochCheckpointStatus set_epoch_analysis_window(bool open) {
    std::lock_guard lock(mutex_);
    if (!active_ || !config_)
      return EpochCheckpointStatus::Inactive;
    if (config_->mode != Mode::Default)
      return EpochCheckpointStatus::ModeDoesNotUseReports;
    if (config_->epoch_analysis.kind != HookConfig::EpochAnalysisKind::Manual)
      return EpochCheckpointStatus::ReportSnapshotFailed;
    const bool changed = open ? begin_epoch_analysis_window() : end_epoch_analysis_window();
    return changed ? EpochCheckpointStatus::Complete : EpochCheckpointStatus::ReportSnapshotFailed;
  }

  void uninstall(bool process_exit = false) {
    std::lock_guard lock(mutex_);
    // Runtime unload and the process-exit fallback must finalize exactly once.
    if (!active_)
      return;
    // Opaque benchmark clients rely on this record even when the runtime
    // retains an HSA reference and only the process-exit fallback runs.
    log_message(
        kLogInfo, "ConSan instrumentation timing total_ns=%llu",
        static_cast<unsigned long long>(InstrumentationClock::instance().elapsed_nanoseconds()));
    const bool supercollider_active = config_ && config_->mode == Mode::SuperCollider;
    // Preserve the preset-only recommendation before clear_unlocked resets
    // config_. Explicit sampling controls belong to expert workflows: do not
    // suggest a preset that those controls would override.
    const char *next_preset = nullptr;
    if (config_ && config_->mode == Mode::Default && !config_->sampling_controls_explicit) {
      const std::string_view preset = config_->preset;
      if (preset == "low")
        next_preset = "default";
      else if (preset == "default")
        next_preset = "high";
      else if (preset == "high")
        next_preset = "higher";
      else if (preset == "higher")
        next_preset = "max";
    }
    const std::string process_concurrent_transform_ceiling =
        config_ && config_->process_concurrent_transform_limit_bytes
            ? std::to_string(*config_->process_concurrent_transform_limit_bytes)
            : "unlimited";
    const std::string process_patched_image_ceiling =
        config_ && config_->process_patched_image_limit_bytes
            ? std::to_string(*config_->process_patched_image_limit_bytes)
            : "unlimited";
    const std::string process_patched_image_growth_ceiling =
        config_ && config_->process_patched_image_growth_limit_bytes
            ? std::to_string(*config_->process_patched_image_growth_limit_bytes)
            : "unlimited";
    if (active_ && core_ != nullptr) {
#define RJ_DBI_RESTORE_CORE(name, field, wrapper, type, install_if) name##_.restore();
      RJ_DBI_HSA_CORE_FUNCTIONS(RJ_DBI_RESTORE_CORE)
#undef RJ_DBI_RESTORE_CORE
    }
    if (active_)
      amd_queue_create_.restore();

    const AutoSuperColliderReportBufferRegistry::Summary supercollider_report_summary =
        AutoSuperColliderReportBufferRegistry::instance().summarize_and_clear(core_);
    const ReportSummary report_summary = summarize_and_clear_report_buffers(core_);
    const StaticCoverageRegistry::Summary static_coverage_summary =
        StaticCoverageRegistry::instance().summarize_and_clear();
    const ProcessByteBudget::Summary transform_admission_summary =
        ProcessTransformAdmissionRegistry::instance().summarize_and_rollover();
    const ReplacementCodeObjectStorageRegistry::Summary patched_image_summary =
        ReplacementCodeObjectStorageRegistry::instance().summarize_and_rollover();
    const bool require_records = require_records_;
    const bool require_diagnostics = require_diagnostics_;
    const bool forbid_diagnostics = forbid_diagnostics_;
    const bool forbid_overflow = forbid_overflow_;
    const uint32_t runtime_sample_stride = runtime_sample_stride_;
    const uint32_t runtime_sample_offset = runtime_sample_offset_;
    // Snapshot initialized scalars before clear_unlocked() resets config_.
    // GCC 13 otherwise loses the optional engagement proof across this function.
    bool cell_selection_configured = false;
    SampleSelector cell_selection;
    if (config_ && config_->cell_selection) {
      cell_selection_configured = true;
      cell_selection = *config_->cell_selection;
    }
    const KernelPrivateDispatchRegistry::DispatchSummary dispatch_summary =
        KernelPrivateDispatchRegistry::instance().dispatch_summary();
    const std::vector<KernelPrivateDispatchRegistry::AllowlistEntrySummary> allowlist_summary =
        KernelPrivateDispatchRegistry::instance().allowlist_summary();
    const bool fault_require_exactly_one = config_ && config_->fault_require_exactly_one;
    const std::optional<ProcessFaultApplicationSnapshot> fault_application_snapshot =
        take_process_fault_application_snapshot();
    const std::optional<FaultLoadSelector> fault_load_selector = fault_load_selector_;
    code_object_reader_registry().clear();
    KernelPrivateDispatchRegistry::instance().clear();
    ReportEpochRegistry::instance().clear();
    if (fault_application_snapshot &&
        (fault_require_exactly_one || fault_application_snapshot->exactly_one_requested)) {
      emit_process_fault_reservation_summary(fault_application_snapshot->reservation);
    }
    if (fault_load_selector) {
      log_message(kLogInfo,
                  "ConSan fault load summary requested_occurrence=%llu observed=%llu "
                  "selected=%llu overflow=%s accepted=%s",
                  static_cast<unsigned long long>(fault_load_selector->requested_occurrence()),
                  static_cast<unsigned long long>(fault_load_selector->observed()),
                  static_cast<unsigned long long>(fault_load_selector->selected()),
                  fault_load_selector->overflow() ? "true" : "false",
                  fault_load_selector->accepted() ? "true" : "false");
      if (!fault_load_selector->accepted()) {
        std::fprintf(stderr, "[rocjitsu-dbi-hooks] ConSan fault load selection failed closed: "
                             "requested occurrence was absent, ambiguous, or overflowed\n");
        std::fflush(stderr);
        std::_Exit(91);
      }
    }
    clear_unlocked(process_exit);
    if (supercollider_active || supercollider_report_summary.buffer_count != 0 ||
        !supercollider_report_summary.complete()) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] ConSan SuperCollider report summary buffers=%llu mismatches=%llu "
          "allocation_failures=%llu read_failures=%llu cleanup_failures=%llu "
          "complete=%s\n",
          static_cast<unsigned long long>(supercollider_report_summary.buffer_count),
          static_cast<unsigned long long>(supercollider_report_summary.mismatch_count),
          static_cast<unsigned long long>(supercollider_report_summary.allocation_failure_count),
          static_cast<unsigned long long>(supercollider_report_summary.read_failure_count),
          static_cast<unsigned long long>(supercollider_report_summary.cleanup_failure_count),
          supercollider_report_summary.complete() ? "true" : "false");
      std::fflush(stderr);
    }
    const ReportTrustEvaluation trust = evaluate_report_trust(report_summary, require_records);
    for (const KernelPrivateDispatchRegistry::AllowlistEntrySummary &entry : allowlist_summary) {
      const char *status = !entry.loaded                ? "not-loaded"
                           : !entry.instrumented        ? "loaded-not-instrumented"
                           : entry.dispatch_count == 0u ? "instrumented-not-dispatched"
                                                        : "instrumented-dispatched";
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] ConSan kernel allowlist entry name=%s loaded=%s "
                   "instrumented=%s dispatches=%llu visible_records=%llu status=%s\n",
                   entry.kernel_name.c_str(), entry.loaded ? "true" : "false",
                   entry.instrumented ? "true" : "false",
                   static_cast<unsigned long long>(entry.dispatch_count),
                   static_cast<unsigned long long>(trust.visible_evidence_count), status);
    }
    std::fprintf(
        stderr,
        "[rocjitsu-dbi-hooks] ConSan report memory required_bytes=%llu "
        "allocated_bytes=%llu completed_epochs=%llu discarded_epochs=%llu "
        "live_before_cleanup=%llu "
        "live_after_cleanup=%llu "
        "peak_live_bytes=%llu per_buffer_ceiling=%llu process_ceiling=%llu "
        "allocation_failures=%llu capacity_failures=%llu cleanup_failures=%llu "
        "fine_grained_snapshot_bytes=%llu coarse_grained_snapshot_bytes=%llu\n",
        static_cast<unsigned long long>(report_summary.required_report_bytes),
        static_cast<unsigned long long>(report_summary.allocated_report_bytes),
        static_cast<unsigned long long>(report_summary.completed_epoch_count),
        static_cast<unsigned long long>(report_summary.discarded_epoch_count),
        static_cast<unsigned long long>(report_summary.current_live_report_bytes),
        static_cast<unsigned long long>(report_summary.current_live_report_bytes_after_cleanup),
        static_cast<unsigned long long>(report_summary.peak_live_report_bytes),
        static_cast<unsigned long long>(kOrdinaryAutoReportBufferCeilingBytes),
        static_cast<unsigned long long>(kAutoReportProcessCeilingBytes),
        static_cast<unsigned long long>(report_summary.allocation_failure_count),
        static_cast<unsigned long long>(report_summary.capacity_failure_count),
        static_cast<unsigned long long>(report_summary.cleanup_failure_count),
        static_cast<unsigned long long>(report_summary.fine_grained_snapshot_bytes),
        static_cast<unsigned long long>(report_summary.coarse_grained_snapshot_bytes));
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] ConSan transform admission memory "
                 "live_bytes=%llu peak_reserved_bytes=%llu process_ceiling=%s\n",
                 static_cast<unsigned long long>(transform_admission_summary.live_bytes),
                 static_cast<unsigned long long>(transform_admission_summary.peak_bytes),
                 process_concurrent_transform_ceiling.c_str());
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] ConSan patched-image memory live_bytes=%llu "
                 "peak_image_bytes=%llu process_ceiling=%s\n",
                 static_cast<unsigned long long>(patched_image_summary.image.live_bytes),
                 static_cast<unsigned long long>(patched_image_summary.image.peak_bytes),
                 process_patched_image_ceiling.c_str());
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] ConSan patched-image growth memory live_bytes=%llu "
                 "peak_growth_bytes=%llu process_ceiling=%s\n",
                 static_cast<unsigned long long>(patched_image_summary.growth.live_bytes),
                 static_cast<unsigned long long>(patched_image_summary.growth.peak_bytes),
                 process_patched_image_growth_ceiling.c_str());
    const uint64_t dynamic_incomplete_count =
        supercollider_report_summary.allocation_failure_count +
        supercollider_report_summary.read_failure_count +
        supercollider_report_summary.cleanup_failure_count + trust.dynamic_incomplete_count;
    const bool static_complete = static_coverage_summary.complete();
    const bool dynamic_complete = dynamic_incomplete_count == 0 && !trust.required_records_missing;
    const bool analysis_complete = static_complete && dynamic_complete;
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] ConSan analysis verdict applicable=%s analysis_complete=%s "
                 "static_complete=%s dynamic_complete=%s applicable_code_objects=%llu "
                 "incomplete_code_objects=%llu access=%llu/%llu barrier=%llu/%llu "
                 "atomic=%llu/%llu fence=%llu/%llu visible_evidence=%llu dynamic_incomplete=%llu "
                 "\n",
                 static_coverage_summary.applicable_code_objects != 0 ? "true" : "false",
                 analysis_complete ? "true" : "false", static_complete ? "true" : "false",
                 dynamic_complete ? "true" : "false",
                 static_cast<unsigned long long>(static_coverage_summary.applicable_code_objects),
                 static_cast<unsigned long long>(static_coverage_summary.incomplete_code_objects),
                 static_cast<unsigned long long>(static_coverage_summary.patched_access),
                 static_cast<unsigned long long>(static_coverage_summary.supported_access),
                 static_cast<unsigned long long>(static_coverage_summary.patched_barrier),
                 static_cast<unsigned long long>(static_coverage_summary.supported_barrier),
                 static_cast<unsigned long long>(static_coverage_summary.patched_atomic),
                 static_cast<unsigned long long>(static_coverage_summary.supported_atomic),
                 static_cast<unsigned long long>(static_coverage_summary.patched_fence),
                 static_cast<unsigned long long>(static_coverage_summary.supported_fence),
                 static_cast<unsigned long long>(trust.visible_evidence_count),
                 static_cast<unsigned long long>(dynamic_incomplete_count));
    std::fflush(stderr);
    if (!analysis_complete) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] ConSan analysis incomplete: no clean result is "
                   "available; inspect coverage and runtime evidence (applicable=%s). "
                   "Use RJ_CONSAN_POLICY=strict to reject ineffective instrumentation.\n",
                   static_coverage_summary.applicable_code_objects != 0 ? "true" : "false");
    }

    if (trust.dropped_record_count != 0) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] ConSan report overflow: "
                   "windows=%llu "
                   "across %llu auto report "
                   "buffer(s)\n",
                   static_cast<unsigned long long>(report_summary.dropped_window_count),
                   static_cast<unsigned long long>(report_summary.buffer_count));
      std::fflush(stderr);
      if (forbid_overflow)
        std::_Exit(90);
    }

    if (report_summary.unusable_snapshot_count() != 0) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] ConSan snapshot incomplete: stale=%llu "
          "incomplete=%llu changed=%llu malformed=%llu across %llu auto report buffer(s)\n",
          static_cast<unsigned long long>(report_summary.stale_snapshot_count),
          static_cast<unsigned long long>(report_summary.incomplete_snapshot_count),
          static_cast<unsigned long long>(report_summary.changed_snapshot_count),
          static_cast<unsigned long long>(report_summary.malformed_snapshot_count),
          static_cast<unsigned long long>(report_summary.buffer_count));
      std::fflush(stderr);
      if (forbid_overflow)
        std::_Exit(90);
    }

    if (trust.required_records_missing) {
      if (dispatch_summary.packet_count == 0u) {
        std::fprintf(
            stderr,
            "[rocjitsu-dbi-hooks] RJ_CONSAN_REQUIRE_RECORDS requested, but %llu auto "
            "ConSan report buffer(s) contained zero visible records and no kernel dispatch "
            "packet was observed\n",
            static_cast<unsigned long long>(report_summary.buffer_count));
      } else if (dispatch_summary.instrumented_packet_count == 0u) {
        std::fprintf(
            stderr,
            "[rocjitsu-dbi-hooks] RJ_CONSAN_REQUIRE_RECORDS requested, but %llu auto "
            "ConSan report buffer(s) contained zero visible records; none of %llu observed "
            "kernel dispatch packet(s) was attributed to a bound ConSan-instrumented "
            "kernel object\n",
            static_cast<unsigned long long>(report_summary.buffer_count),
            static_cast<unsigned long long>(dispatch_summary.packet_count));
      } else if (cell_selection_configured &&
                 (runtime_sample_stride > 1u || cell_selection.stride > 1u)) {
        std::fprintf(stderr,
                     "[rocjitsu-dbi-hooks] RJ_CONSAN_REQUIRE_RECORDS requested, but %llu auto "
                     "ConSan report buffer(s) contained zero visible records after %llu "
                     "instrumented dispatch packet(s); independent sampling may have selected no "
                     "workgroups or LDS cells (workgroup_stride=%u workgroup_offset=%u "
                     "cell_stride=%u cell_offset=%u). Set RJ_CONSAN_WORKGROUP_SAMPLE_STRIDE=1 "
                     "and RJ_CONSAN_CELL_SAMPLE_STRIDE=1 with both offsets zero to "
                     "distinguish sampling gaps from dense-path gaps\n",
                     static_cast<unsigned long long>(report_summary.buffer_count),
                     static_cast<unsigned long long>(dispatch_summary.instrumented_packet_count),
                     runtime_sample_stride, runtime_sample_offset, cell_selection.stride,
                     cell_selection.offset);
      } else if (runtime_sample_stride > 1u) {
        std::fprintf(stderr,
                     "[rocjitsu-dbi-hooks] RJ_CONSAN_REQUIRE_RECORDS requested, but %llu auto "
                     "ConSan report buffer(s) contained zero visible records after %llu "
                     "instrumented dispatch packet(s); runtime sampling may have selected no "
                     "workgroups, or selected workgroups may not have executed an instrumented "
                     "site (stride=%u offset=%u). Set RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=1 "
                     "and RJ_CONSAN_RUNTIME_SAMPLE_OFFSET=0 to "
                     "distinguish sampling gaps from dense-path gaps\n",
                     static_cast<unsigned long long>(report_summary.buffer_count),
                     static_cast<unsigned long long>(dispatch_summary.instrumented_packet_count),
                     runtime_sample_stride, runtime_sample_offset);
      } else {
        std::fprintf(stderr,
                     "[rocjitsu-dbi-hooks] RJ_CONSAN_REQUIRE_RECORDS requested, but %llu auto "
                     "ConSan report buffer(s) contained zero visible records after %llu "
                     "instrumented dispatch packet(s); the dense record path produced no "
                     "evidence\n",
                     static_cast<unsigned long long>(report_summary.buffer_count),
                     static_cast<unsigned long long>(dispatch_summary.instrumented_packet_count));
      }
      std::fflush(stderr);
      std::_Exit(86);
    }

    if (require_diagnostics && !trust.has_diagnostics) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] RJ_CONSAN_REQUIRE_DIAGNOSTICS requested, but "
                   "%llu auto ConSan report buffer(s) produced zero diagnostics or sampled "
                   "conflicts "
                   "("
                   "sampled=%llu, conflicts=%llu "
                   "immediate_conflicts=%llu)\n",
                   static_cast<unsigned long long>(report_summary.buffer_count),
                   static_cast<unsigned long long>(report_summary.visible_watchpoint_count),
                   static_cast<unsigned long long>(report_summary.conflict_count),
                   static_cast<unsigned long long>(report_summary.immediate_conflict_count));
      std::fflush(stderr);
      std::_Exit(88);
    }
    if (forbid_diagnostics && trust.has_diagnostics) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] RJ_CONSAN_FORBID_DIAGNOSTICS requested, but "
                   "%llu auto ConSan report buffer(s) produced diagnostics or sampled "
                   "conflicts "
                   "("
                   "sampled=%llu, conflicts=%llu "
                   "immediate_conflicts=%llu)\n",
                   static_cast<unsigned long long>(report_summary.buffer_count),
                   static_cast<unsigned long long>(report_summary.visible_watchpoint_count),
                   static_cast<unsigned long long>(report_summary.conflict_count),
                   static_cast<unsigned long long>(report_summary.immediate_conflict_count));
      std::fflush(stderr);
      std::_Exit(89);
    }
    if (next_preset != nullptr && analysis_complete && !trust.has_diagnostics &&
        report_summary.static_mapping_malformed_count == 0) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] ConSan coverage hint: %s "
                   "For increased coverage, retry with RJ_CONSAN_PRESET=%s. "
                   "No reports does not establish race freedom.\n",
                   trust.visible_evidence_count == 0
                       ? "No runtime evidence was collected under sampled coverage."
                       : "No races were reported under sampled coverage.",
                   next_preset);
      std::fflush(stderr);
    }
  }

  [[nodiscard]] std::optional<HookConfig> config() const {
    std::lock_guard lock(mutex_);
    return config_;
  }

  [[nodiscard]] std::optional<FaultLoadSelection> observe_fault_load_match() {
    std::lock_guard lock(mutex_);
    if (!fault_load_selector_)
      return std::nullopt;
    return fault_load_selector_->observe();
  }

#define RJ_DBI_CORE_GETTER(name, field, wrapper, type, install_if)                                 \
  [[nodiscard]] type name() const {                                                                \
    std::lock_guard lock(mutex_);                                                                  \
    return name##_.original();                                                                     \
  }
  RJ_DBI_HSA_CORE_FUNCTIONS(RJ_DBI_CORE_GETTER)
#undef RJ_DBI_CORE_GETTER

  void set_loader_create_from_file_with_offset_size(LoaderCreateFromFileWithOffsetSize function) {
    std::lock_guard lock(mutex_);
    original_loader_create_from_file_with_offset_size_ = function;
  }

  [[nodiscard]] LoaderCreateFromFileWithOffsetSize
  loader_create_from_file_with_offset_size() const {
    std::lock_guard lock(mutex_);
    return original_loader_create_from_file_with_offset_size_;
  }

  [[nodiscard]] hsa_amd_queue_create_fn_t amd_queue_create() const {
    std::lock_guard lock(mutex_);
    return amd_queue_create_.original();
  }

  [[nodiscard]] decltype(hsa_queue_destroy) *queue_destroy() const {
    std::lock_guard lock(mutex_);
    return core_ == nullptr ? nullptr : core_->hsa_queue_destroy_fn;
  }

  [[nodiscard]] hsa_amd_queue_set_priority_fn_t queue_set_priority() const {
    std::lock_guard lock(mutex_);
    return amd_ext_ == nullptr ? nullptr : amd_ext_->hsa_amd_queue_set_priority_fn;
  }

  [[nodiscard]] hsa_amd_queue_cu_set_mask_fn_t queue_cu_set_mask() const {
    std::lock_guard lock(mutex_);
    return amd_ext_ == nullptr ? nullptr : amd_ext_->hsa_amd_queue_cu_set_mask_fn;
  }

  [[nodiscard]] bool dispatch_packet_interception_enabled() const {
    std::lock_guard lock(mutex_);
    return intercept_dispatch_packets_;
  }

  [[nodiscard]] hsa_amd_queue_intercept_create_fn_t queue_intercept_create() const {
    std::lock_guard lock(mutex_);
    return amd_ext_ == nullptr ? nullptr : amd_ext_->hsa_amd_queue_intercept_create_fn;
  }

  [[nodiscard]] hsa_amd_queue_intercept_register_fn_t queue_intercept_register() const {
    std::lock_guard lock(mutex_);
    return amd_ext_ == nullptr ? nullptr : amd_ext_->hsa_amd_queue_intercept_register_fn;
  }

  [[nodiscard]] CoreApiTable *core_table() const {
    std::lock_guard lock(mutex_);
    return core_;
  }

private:
  [[nodiscard]] static bool validate_table(HsaApiTable *table) {
    if (table == nullptr || table->core_ == nullptr) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid HSA API table passed to OnLoad\n");
      return false;
    }

    constexpr size_t required_size =
        std::max({offsetof(CoreApiTable, hsa_executable_load_agent_code_object_fn) +
                      sizeof(CoreApiTable::hsa_executable_load_agent_code_object_fn),
                  offsetof(CoreApiTable, hsa_executable_destroy_fn) +
                      sizeof(CoreApiTable::hsa_executable_destroy_fn),
                  offsetof(CoreApiTable, hsa_executable_get_symbol_by_name_fn) +
                      sizeof(CoreApiTable::hsa_executable_get_symbol_by_name_fn),
                  offsetof(CoreApiTable, hsa_executable_iterate_agent_symbols_fn) +
                      sizeof(CoreApiTable::hsa_executable_iterate_agent_symbols_fn),
                  offsetof(CoreApiTable, hsa_executable_symbol_get_info_fn) +
                      sizeof(CoreApiTable::hsa_executable_symbol_get_info_fn)});
    if (table->core_->version.minor_id < required_size) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] HSA core table too small: got %u bytes, need %zu bytes\n",
                   table->core_->version.minor_id, required_size);
      return false;
    }
    return true;
  }

  void clear_unlocked(bool retain_runtime_functions = false) {
    active_ = false;
    g_log_level.store(kLogDisabled, std::memory_order_relaxed);
    if (!retain_runtime_functions) {
      table_ = nullptr;
      core_ = nullptr;
      amd_ext_ = nullptr;
    }
    config_.reset();
    require_records_ = false;
    require_diagnostics_ = false;
    forbid_diagnostics_ = false;
    forbid_overflow_ = false;
    runtime_sample_stride_ = 1u;
    runtime_sample_offset_ = 0u;
    fault_load_selector_.reset();
    // HIP can cache wrapper pointers and invoke them in later exit handlers.
    // Keep their original runtime functions callable after exit finalization.
    if (!retain_runtime_functions) {
#define RJ_DBI_CLEAR_CORE(name, field, wrapper, type, install_if) name##_.clear();
      RJ_DBI_HSA_CORE_FUNCTIONS(RJ_DBI_CLEAR_CORE)
#undef RJ_DBI_CLEAR_CORE
      original_loader_create_from_file_with_offset_size_ = nullptr;
      amd_queue_create_.clear();
    }
    intercept_dispatch_segments_ = false;
    intercept_dispatch_packets_ = false;
  }

  mutable std::mutex mutex_;
  HsaApiTable *table_ = nullptr;
  CoreApiTable *core_ = nullptr;
  AmdExtTable *amd_ext_ = nullptr;
  std::optional<HookConfig> config_;
  // Unload acceptance gates are immutable process-level policy. Keep them
  // separate from the per-load report-address copy of HookConfig.
  bool require_records_ = false;
  bool require_diagnostics_ = false;
  bool forbid_diagnostics_ = false;
  bool forbid_overflow_ = false;
  uint32_t runtime_sample_stride_ = 1u;
  uint32_t runtime_sample_offset_ = 0u;
  std::optional<FaultLoadSelector> fault_load_selector_;
  bool active_ = false;
#define RJ_DBI_DECLARE_CORE(name, field, wrapper, type, install_if)                                \
  rocjitsu::hooks::HsaApiFunctionPatch<type> name##_;
  RJ_DBI_HSA_CORE_FUNCTIONS(RJ_DBI_DECLARE_CORE)
#undef RJ_DBI_DECLARE_CORE
  LoaderCreateFromFileWithOffsetSize original_loader_create_from_file_with_offset_size_ = nullptr;
  rocjitsu::hooks::HsaApiFunctionPatch<hsa_amd_queue_create_fn_t> amd_queue_create_;
  bool intercept_dispatch_segments_ = false;
  bool intercept_dispatch_packets_ = false;
};

#undef RJ_DBI_HSA_CORE_FUNCTIONS

RjDbiHsaLayer &layer() {
  // Cached runtime callbacks may outlive C++ static destruction ordering.
  static auto *state = new RjDbiHsaLayer;
  return *state;
}

void try_automatic_epoch_checkpoint() {
  CoreApiTable *const core = layer().core_table();
  if (core == nullptr)
    return;
  ReportEpochRegistry::instance().checkpoint_if_quiescent(
      core->hsa_signal_load_scacquire_fn,
      [core] { return checkpoint_report_buffers_automatically(core); });
}

hsa_status_t HSA_API rj_dbi_signal_destroy(hsa_signal_t signal) {
  auto *const original = layer().signal_destroy();
  if (original == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  try_automatic_epoch_checkpoint();
  CoreApiTable *const core = layer().core_table();
  ReportEpochRegistry::instance().forget_signal(
      signal, core == nullptr ? nullptr : core->hsa_signal_load_scacquire_fn);
  return original(signal);
}

hsa_signal_value_t HSA_API rj_dbi_signal_wait_relaxed(hsa_signal_t signal,
                                                      hsa_signal_condition_t condition,
                                                      hsa_signal_value_t compare_value,
                                                      uint64_t timeout_hint,
                                                      hsa_wait_state_t wait_state_hint) {
  auto *const original = layer().signal_wait_relaxed();
  if (original == nullptr)
    return 0;
  const hsa_signal_value_t value =
      original(signal, condition, compare_value, timeout_hint, wait_state_hint);
  try_automatic_epoch_checkpoint();
  return value;
}

hsa_signal_value_t HSA_API rj_dbi_signal_wait_scacquire(hsa_signal_t signal,
                                                        hsa_signal_condition_t condition,
                                                        hsa_signal_value_t compare_value,
                                                        uint64_t timeout_hint,
                                                        hsa_wait_state_t wait_state_hint) {
  auto *const original = layer().signal_wait_scacquire();
  if (original == nullptr)
    return 0;
  const hsa_signal_value_t value =
      original(signal, condition, compare_value, timeout_hint, wait_state_hint);
  try_automatic_epoch_checkpoint();
  return value;
}

void intercept_loader_extension_table(size_t table_length, void *table) {
  constexpr size_t required_length =
      offsetof(AmdLoaderExtTable102, create_from_file_with_offset_size) +
      sizeof(AmdLoaderExtTable102::create_from_file_with_offset_size);
  if (table == nullptr || table_length < required_length)
    return;
  auto *loader = static_cast<AmdLoaderExtTable102 *>(table);
  auto *original = loader->create_from_file_with_offset_size;
  if (original == nullptr ||
      original == rj_dbi_loader_code_object_reader_create_from_file_with_offset_size)
    return;
  layer().set_loader_create_from_file_with_offset_size(original);
  loader->create_from_file_with_offset_size =
      rj_dbi_loader_code_object_reader_create_from_file_with_offset_size;
}

hsa_status_t HSA_API rj_dbi_system_get_extension_table(uint16_t extension, uint16_t version_major,
                                                       uint16_t version_minor, void *table) {
  auto *original = layer().get_extension_table();
  if (original == nullptr)
    return HSA_STATUS_ERROR;
  const hsa_status_t status = original(extension, version_major, version_minor, table);
  if (status == HSA_STATUS_SUCCESS && extension == HSA_EXTENSION_AMD_LOADER && version_major == 1 &&
      version_minor >= 2) {
    const size_t table_length =
        version_minor >= 3 ? sizeof(AmdLoaderExtTable103) : sizeof(AmdLoaderExtTable102);
    intercept_loader_extension_table(table_length, table);
  }
  return status;
}

hsa_status_t HSA_API rj_dbi_system_get_major_extension_table(uint16_t extension,
                                                             uint16_t version_major,
                                                             size_t table_length, void *table) {
  auto *original = layer().get_major_extension_table();
  if (original == nullptr)
    return HSA_STATUS_ERROR;
  const hsa_status_t status = original(extension, version_major, table_length, table);
  if (status == HSA_STATUS_SUCCESS && extension == HSA_EXTENSION_AMD_LOADER && version_major == 1)
    intercept_loader_extension_table(table_length, table);
  return status;
}

hsa_status_t HSA_API rj_dbi_code_object_reader_create_from_memory(
    const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader) {
  auto *original = layer().create_from_memory();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original(code_object, size, code_object_reader);
  if (status == HSA_STATUS_SUCCESS && code_object_reader != nullptr && code_object != nullptr) {
    if (!code_object_reader_registry().store(code_object_reader->handle,
                                             static_cast<const uint8_t *>(code_object), size)) {
      if (auto *original_destroy = layer().destroy(); original_destroy != nullptr)
        (void)original_destroy(*code_object_reader);
      *code_object_reader = {};
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] failed to track memory-backed code-object reader\n");
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    log_message(kLogDebug, "registered memory reader=%llu bytes=%zu",
                static_cast<unsigned long long>(code_object_reader->handle), size);
  }
  return status;
}

hsa_status_t HSA_API rj_dbi_code_object_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t *code_object_reader) {
  auto *original = layer().create_from_file();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original(file, code_object_reader);
  if (status == HSA_STATUS_SUCCESS && code_object_reader != nullptr) {
    const auto bytes = rocjitsu::snapshot_code_object_file(file);
    if (!bytes) {
      log_message(kLogInfo, "could not snapshot file-backed reader=%llu",
                  static_cast<unsigned long long>(code_object_reader->handle));
    } else if (!code_object_reader_registry().store(code_object_reader->handle, bytes->data(),
                                                    bytes->size(), bytes)) {
      if (auto *original_destroy = layer().destroy(); original_destroy != nullptr)
        (void)original_destroy(*code_object_reader);
      *code_object_reader = {};
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] failed to track file-backed code-object reader\n");
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    } else {
      log_message(kLogDebug, "registered file reader=%llu bytes=%zu",
                  static_cast<unsigned long long>(code_object_reader->handle), bytes->size());
    }
  }
  return status;
}

hsa_status_t HSA_API rj_dbi_loader_code_object_reader_create_from_file_with_offset_size(
    hsa_file_t file, size_t offset, size_t size, hsa_code_object_reader_t *code_object_reader) {
  auto *original = layer().loader_create_from_file_with_offset_size();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original(file, offset, size, code_object_reader);
  if (status != HSA_STATUS_SUCCESS || code_object_reader == nullptr)
    return status;

  const auto bytes = rocjitsu::snapshot_code_object_file_range(file, offset, size);
  if (!bytes) {
    log_message(kLogInfo, "could not snapshot ranged file-backed reader=%llu offset=%zu bytes=%zu",
                static_cast<unsigned long long>(code_object_reader->handle), offset, size);
  } else if (!code_object_reader_registry().store(code_object_reader->handle, bytes->data(),
                                                  bytes->size(), bytes)) {
    if (auto *original_destroy = layer().destroy(); original_destroy != nullptr)
      (void)original_destroy(*code_object_reader);
    *code_object_reader = {};
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] failed to track ranged file-backed code-object reader\n");
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  } else {
    log_message(kLogDebug, "registered ranged file reader=%llu offset=%zu bytes=%zu",
                static_cast<unsigned long long>(code_object_reader->handle), offset, bytes->size());
  }
  return status;
}

hsa_status_t HSA_API
rj_dbi_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader) {
  code_object_reader_registry().remove(code_object_reader.handle);

  auto *original = layer().destroy();
  if (original == nullptr)
    return HSA_STATUS_ERROR;
  return original(code_object_reader);
}

hsa_status_t HSA_API rj_dbi_executable_destroy(hsa_executable_t executable) {
  auto *original = layer().executable_destroy();
  if (original == nullptr)
    return HSA_STATUS_ERROR;
  const hsa_status_t status = original(executable);
  if (status == HSA_STATUS_SUCCESS) {
    AutoSuperColliderReportBufferRegistry::instance().retire(layer().core_table(), executable);
    retire_report_buffers(layer().core_table(), executable);
    ReplacementCodeObjectStorageRegistry::instance().remove(executable);
    KernelPrivateDispatchRegistry::instance().erase_executable(executable);
  }
  return status;
}

struct alignas(8) InterceptPacket {
  std::array<uint8_t, sizeof(hsa_kernel_dispatch_packet_t)> bytes{};
};

// The minimal HSA tool headers intentionally omit the extended-dispatch
// definition. Keep this binary view local to the interceptor and assert every
// field that ConSan reads or rewrites against the ordinary dispatch ABI.
struct AmdExtKernelDispatchPacket {
  uint16_t header = 0;
  uint8_t amd_format = 0;
  uint8_t setup = 0;
  uint16_t workgroup_size_x = 0;
  uint16_t workgroup_size_y = 0;
  uint16_t workgroup_size_z = 0;
  uint16_t reserved0 = 0;
  uint32_t cluster_count_x = 0;
  uint16_t cluster_count_y = 0;
  uint16_t cluster_count_z = 0;
  uint8_t cluster_size_x = 0;
  uint8_t cluster_size_y = 0;
  uint8_t cluster_size_z = 0;
  uint8_t perf_hint = 0;
  uint32_t private_segment_size = 0;
  uint32_t group_segment_size = 0;
  uint64_t kernel_object = 0;
  void *kernarg_address = nullptr;
  hsa_signal_t dep_signal{};
  hsa_signal_t completion_signal{};
};

constexpr uint8_t kAmdExtKernelDispatchFormat = 3;
static_assert(sizeof(InterceptPacket) == sizeof(hsa_kernel_dispatch_packet_t));
static_assert(sizeof(InterceptPacket) == 64);
static_assert(sizeof(AmdExtKernelDispatchPacket) == sizeof(hsa_kernel_dispatch_packet_t));
static_assert(offsetof(hsa_kernel_dispatch_packet_t, private_segment_size) ==
              offsetof(AmdExtKernelDispatchPacket, private_segment_size));
static_assert(offsetof(hsa_kernel_dispatch_packet_t, group_segment_size) ==
              offsetof(AmdExtKernelDispatchPacket, group_segment_size));
static_assert(offsetof(hsa_kernel_dispatch_packet_t, kernel_object) ==
              offsetof(AmdExtKernelDispatchPacket, kernel_object));
static_assert(offsetof(hsa_kernel_dispatch_packet_t, kernarg_address) ==
              offsetof(AmdExtKernelDispatchPacket, kernarg_address));

void rj_dbi_queue_write_interceptor(const void *packets, uint64_t packet_count,
                                    uint64_t user_packet_index, void *data,
                                    hsa_amd_queue_intercept_packet_writer_t writer) {
  (void)user_packet_index;
  (void)user_packet_index;
  if (packets == nullptr || writer == nullptr || packet_count == 0) {
    if (writer != nullptr)
      writer(packets, packet_count);
    return;
  }
  const auto total_packet_bytes =
      byte_accounting::checked_allocation_charge(0, packet_count, sizeof(InterceptPacket));
  if (!total_packet_bytes) {
    writer(packets, packet_count);
    return;
  }

  CoreApiTable *const core = layer().core_table();
  auto submission_lock = ReportEpochRegistry::instance().lock_submission();
  std::vector<InterceptPacket> rewritten(static_cast<size_t>(packet_count));
  std::memcpy(rewritten.data(), packets, static_cast<size_t>(*total_packet_bytes));
  for (InterceptPacket &packet_bytes : rewritten) {
    auto *packet = reinterpret_cast<hsa_kernel_dispatch_packet_t *>(packet_bytes.bytes.data());
    const uint16_t type =
        static_cast<uint16_t>((packet->header >> HSA_PACKET_HEADER_TYPE) &
                              ((uint16_t{1} << HSA_PACKET_HEADER_WIDTH_TYPE) - 1u));
    const auto *extended_packet = reinterpret_cast<const AmdExtKernelDispatchPacket *>(packet);
    const bool is_extended_dispatch =
        extended_packet->amd_format == kAmdExtKernelDispatchFormat &&
        (type == HSA_PACKET_TYPE_VENDOR_SPECIFIC || type == HSA_PACKET_TYPE_INVALID);
    if (type == HSA_PACKET_TYPE_BARRIER_AND || type == HSA_PACKET_TYPE_BARRIER_OR) {
      const auto *barrier =
          reinterpret_cast<const hsa_barrier_and_packet_t *>(packet_bytes.bytes.data());
      const uint16_t barrier_bit =
          static_cast<uint16_t>((barrier->header >> HSA_PACKET_HEADER_BARRIER) & 1u);
      ReportEpochRegistry::instance().note_ordering_barrier_locked(
          data, barrier_bit != 0u, barrier->completion_signal,
          core == nullptr ? nullptr : core->hsa_signal_load_scacquire_fn);
    }
    if (type != HSA_PACKET_TYPE_KERNEL_DISPATCH && !is_extended_dispatch)
      continue;
    const KernelPrivateDispatchRegistry::DispatchRequirements requirements =
        KernelPrivateDispatchRegistry::instance().note_and_query_dispatch(packet->kernel_object);
    if (requirements.instrumented) {
      ReportEpochRegistry::instance().note_instrumented_dispatch_locked(
          data,
          is_extended_dispatch ? extended_packet->completion_signal : packet->completion_signal,
          core == nullptr ? nullptr : core->hsa_signal_load_scacquire_fn);
    }
    if (requirements.has_segment_requirement()) {
      if (is_extended_dispatch) {
        log_message(kLogInfo,
                    "ConSan extended dispatch geometry kernel_object=0x%llx "
                    "workgroup=%ux%ux%u clusters=%ux%ux%u cluster_size=%ux%ux%u "
                    "private_bytes=%u group_bytes=%u",
                    static_cast<unsigned long long>(extended_packet->kernel_object),
                    extended_packet->workgroup_size_x, extended_packet->workgroup_size_y,
                    extended_packet->workgroup_size_z, extended_packet->cluster_count_x,
                    extended_packet->cluster_count_y, extended_packet->cluster_count_z,
                    extended_packet->cluster_size_x, extended_packet->cluster_size_y,
                    extended_packet->cluster_size_z, extended_packet->private_segment_size,
                    extended_packet->group_segment_size);
      } else {
        log_message(kLogInfo,
                    "ConSan dispatch geometry kernel_object=0x%llx workgroup=%ux%ux%u "
                    "grid=%ux%ux%u private_bytes=%u group_bytes=%u",
                    static_cast<unsigned long long>(packet->kernel_object),
                    packet->workgroup_size_x, packet->workgroup_size_y, packet->workgroup_size_z,
                    packet->grid_size_x, packet->grid_size_y, packet->grid_size_z,
                    packet->private_segment_size, packet->group_segment_size);
      }
    }
    const uint32_t runtime_private_bytes = packet->private_segment_size;
    uint32_t target_private_bytes =
        std::max(runtime_private_bytes, requirements.required_private_bytes);
    if (requirements.dynamic_private_addend != 0u) {
      const uint64_t dynamic_target = static_cast<uint64_t>(runtime_private_bytes) +
                                      static_cast<uint64_t>(requirements.dynamic_private_addend);
      if (dynamic_target > std::numeric_limits<uint32_t>::max()) {
        target_private_bytes = std::numeric_limits<uint32_t>::max();
        log_message(kLogInfo,
                    "ConSan dispatch-private saturated kernel_object=0x%llx "
                    "runtime_bytes=%u dynamic_addend=%u",
                    static_cast<unsigned long long>(packet->kernel_object), runtime_private_bytes,
                    requirements.dynamic_private_addend);
      } else {
        target_private_bytes =
            std::max(target_private_bytes, static_cast<uint32_t>(dynamic_target));
      }
    }
    if (target_private_bytes > packet->private_segment_size) {
      log_message(kLogInfo,
                  "ConSan dispatch-private grow kernel_object=0x%llx private_bytes=%u->%u",
                  static_cast<unsigned long long>(packet->kernel_object),
                  packet->private_segment_size, target_private_bytes);
      packet->private_segment_size = target_private_bytes;
    }
    if (requirements.required_group_bytes > packet->group_segment_size) {
      log_message(kLogInfo, "ConSan dispatch-group grow kernel_object=0x%llx group_bytes=%u->%u",
                  static_cast<unsigned long long>(packet->kernel_object),
                  packet->group_segment_size, requirements.required_group_bytes);
      packet->group_segment_size = requirements.required_group_bytes;
    }
  }
  writer(rewritten.data(), packet_count);
}

hsa_status_t HSA_API rj_dbi_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                         void (*callback)(hsa_status_t, hsa_queue_t *, void *),
                                         void *data, uint32_t private_segment_size,
                                         uint32_t group_segment_size, hsa_queue_t **queue) {
  auto *intercept_create = layer().queue_intercept_create();
  auto *intercept_register = layer().queue_intercept_register();
  if (intercept_create == nullptr || intercept_register == nullptr)
    return HSA_STATUS_ERROR;
  const hsa_status_t create_status = intercept_create(
      agent, size, type, callback, data, private_segment_size, group_segment_size, queue);
  if (create_status != HSA_STATUS_SUCCESS || queue == nullptr || *queue == nullptr)
    return create_status;
  const hsa_status_t register_status =
      intercept_register(*queue, rj_dbi_queue_write_interceptor, *queue);
  if (register_status != HSA_STATUS_SUCCESS) {
    CoreApiTable *core = layer().core_table();
    if (core != nullptr && core->hsa_queue_destroy_fn != nullptr)
      (void)core->hsa_queue_destroy_fn(*queue);
    *queue = nullptr;
  }
  return register_status;
}

hsa_status_t HSA_API rj_dbi_amd_queue_create(hsa_agent_t agent, hsa_amd_queue_create_desc_t *descs,
                                             uint32_t num_descs) {
  auto *original_create = layer().amd_queue_create();
  if (original_create == nullptr)
    return HSA_STATUS_ERROR;

  hsa_status_t status = original_create(agent, descs, num_descs);
  if (descs == nullptr)
    return status;

  hsa_status_t replacement_error = HSA_STATUS_SUCCESS;
  auto note_replacement_error = [&](hsa_status_t error) {
    if (replacement_error == HSA_STATUS_SUCCESS)
      replacement_error = error;
  };
  auto destroy_queue = [&](hsa_queue_t *queue) {
    auto *destroy = layer().queue_destroy();
    return queue == nullptr || destroy == nullptr ? HSA_STATUS_ERROR : destroy(queue);
  };

  for (uint32_t index = 0; index < num_descs; ++index) {
    hsa_amd_queue_create_desc_t &desc = descs[index];
    hsa_queue_t *plain_queue = desc.queue;
    if (plain_queue == nullptr || desc.engine_type != HSA_AMD_QUEUE_ENGINE_COMPUTE)
      continue;

    // Descriptor queues are ordinary hardware queues and cannot accept a
    // ROCr packet interceptor. Replace each successfully created compute queue
    // with an equivalent InterceptQueue so ConSan can grow the dispatch packet
    // segment sizes when instrumentation grows the kernel descriptor.
    const hsa_amd_compute_queue_params_t &compute = desc.engine.compute;
    const uint32_t packet_count =
        desc.queue_size_bytes / static_cast<uint32_t>(sizeof(hsa_kernel_dispatch_packet_t));
    hsa_queue_t *intercept_queue = nullptr;
    hsa_status_t intercept_status =
        rj_dbi_queue_create(agent, packet_count, compute.type, desc.callback, desc.callback_data,
                            compute.private_segment_size, UINT32_MAX, &intercept_queue);

    if (intercept_status == HSA_STATUS_SUCCESS && desc.priority != HSA_AMD_QUEUE_PRIORITY_NORMAL) {
      auto *set_priority = layer().queue_set_priority();
      intercept_status = set_priority == nullptr ? HSA_STATUS_ERROR_INVALID_QUEUE_CREATION
                                                 : set_priority(intercept_queue, desc.priority);
    }
    if (intercept_status == HSA_STATUS_SUCCESS && compute.cu_mask_count != 0u) {
      auto *set_cu_mask = layer().queue_cu_set_mask();
      intercept_status = set_cu_mask == nullptr
                             ? HSA_STATUS_ERROR_INVALID_QUEUE_CREATION
                             : set_cu_mask(intercept_queue, compute.cu_mask_count, compute.cu_mask);
    }

    if (intercept_status == HSA_STATUS_SUCCESS) {
      const hsa_status_t destroy_status = destroy_queue(plain_queue);
      if (destroy_status == HSA_STATUS_SUCCESS) {
        desc.queue = intercept_queue;
        if (desc.flags != HSA_AMD_QUEUE_CREATE_SYSTEM_MEM) {
          log_message(kLogInfo,
                      "ConSan descriptor queue interception index=%u flags=0x%x "
                      "ring-placement=system-memory",
                      index, desc.flags);
        }
        continue;
      }
      intercept_status = destroy_status;
    }

    // Never return a successful but uninstrumented compute queue: ConSan may
    // have increased its scratch or LDS requirements after CLR cached the
    // original metadata. Preserve batch partial-success semantics by clearing
    // only the descriptor whose replacement failed.
    if (intercept_queue != nullptr)
      (void)destroy_queue(intercept_queue);
    (void)destroy_queue(plain_queue);
    desc.queue = nullptr;
    note_replacement_error(intercept_status);
  }

  return status != HSA_STATUS_SUCCESS ? status : replacement_error;
}

[[nodiscard]] std::optional<std::string>
query_dbi_executable_symbol_name(hsa_executable_symbol_t symbol) {
  auto *original = layer().symbol_get_info();
  if (original == nullptr)
    return std::nullopt;
  uint32_t name_length = 0;
  if (original(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_length) !=
          HSA_STATUS_SUCCESS ||
      name_length == 0u) {
    return std::nullopt;
  }
  std::string name(name_length, '\0');
  if (original(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data()) != HSA_STATUS_SUCCESS)
    return std::nullopt;
  return name;
}

struct DbiIterateAgentSymbolsData {
  hsa_status_t (*callback)(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t,
                           void *) = nullptr;
  void *data = nullptr;
};

struct DbiIterateSymbolsData {
  hsa_status_t (*callback)(hsa_executable_t, hsa_executable_symbol_t, void *) = nullptr;
  void *data = nullptr;
};

hsa_status_t HSA_API rj_dbi_iterate_symbols_callback(hsa_executable_t executable,
                                                     hsa_executable_symbol_t symbol, void *data) {
  auto *wrapped = static_cast<DbiIterateSymbolsData *>(data);
  if (const auto name = query_dbi_executable_symbol_name(symbol)) {
    KernelPrivateDispatchRegistry::instance().bind_symbol(executable, *name, symbol,
                                                          layer().symbol_get_info());
  }
  return wrapped->callback(executable, symbol, wrapped->data);
}

hsa_status_t HSA_API rj_dbi_executable_iterate_symbols(
    hsa_executable_t executable,
    hsa_status_t (*callback)(hsa_executable_t, hsa_executable_symbol_t, void *), void *data) {
  auto *original = layer().iterate_symbols();
  if (original == nullptr)
    return HSA_STATUS_ERROR;
  if (callback == nullptr)
    return original(executable, callback, data);
  DbiIterateSymbolsData wrapped{callback, data};
  return original(executable, rj_dbi_iterate_symbols_callback, &wrapped);
}

hsa_status_t HSA_API rj_dbi_iterate_agent_symbols_callback(hsa_executable_t executable,
                                                           hsa_agent_t agent,
                                                           hsa_executable_symbol_t symbol,
                                                           void *data) {
  auto *wrapped = static_cast<DbiIterateAgentSymbolsData *>(data);
  if (const auto name = query_dbi_executable_symbol_name(symbol)) {
    KernelPrivateDispatchRegistry::instance().bind_symbol(executable, *name, symbol,
                                                          layer().symbol_get_info());
  }
  return wrapped->callback(executable, agent, symbol, wrapped->data);
}

hsa_status_t HSA_API rj_dbi_executable_iterate_agent_symbols(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t, void *),
    void *data) {
  auto *original = layer().iterate_agent_symbols();
  if (original == nullptr)
    return HSA_STATUS_ERROR;
  if (callback == nullptr)
    return original(executable, agent, callback, data);
  DbiIterateAgentSymbolsData wrapped{callback, data};
  return original(executable, agent, rj_dbi_iterate_agent_symbols_callback, &wrapped);
}

hsa_status_t HSA_API rj_dbi_executable_get_symbol_by_name(hsa_executable_t executable,
                                                          const char *symbol_name,
                                                          const hsa_agent_t *agent,
                                                          hsa_executable_symbol_t *symbol) {
  auto *original = layer().get_symbol_by_name();
  if (original == nullptr)
    return HSA_STATUS_ERROR;
  const hsa_status_t status = original(executable, symbol_name, agent, symbol);
  if (status == HSA_STATUS_SUCCESS && symbol_name != nullptr && symbol != nullptr) {
    KernelPrivateDispatchRegistry::instance().bind_symbol(executable, symbol_name, *symbol,
                                                          layer().symbol_get_info());
  }
  return status;
}

hsa_status_t HSA_API rj_dbi_executable_get_symbol(hsa_executable_t executable,
                                                  const char *module_name, const char *symbol_name,
                                                  hsa_agent_t agent, int32_t call_convention,
                                                  hsa_executable_symbol_t *symbol) {
  auto *original = layer().get_symbol();
  if (original == nullptr)
    return HSA_STATUS_ERROR;
  const hsa_status_t status =
      original(executable, module_name, symbol_name, agent, call_convention, symbol);
  if (status == HSA_STATUS_SUCCESS && symbol_name != nullptr && symbol != nullptr) {
    KernelPrivateDispatchRegistry::instance().bind_symbol(executable, symbol_name, *symbol,
                                                          layer().symbol_get_info());
  }
  return status;
}

hsa_status_t HSA_API rj_dbi_executable_symbol_get_info(hsa_executable_symbol_t symbol,
                                                       hsa_executable_symbol_info_t attribute,
                                                       void *value) {
  auto *original = layer().symbol_get_info();
  if (original == nullptr)
    return HSA_STATUS_ERROR;
  const hsa_status_t status = original(symbol, attribute, value);
  if (status != HSA_STATUS_SUCCESS || value == nullptr)
    return status;

  auto &registry = KernelPrivateDispatchRegistry::instance();
  if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE) {
    if (const auto required = registry.required_for_symbol(symbol); required) {
      auto *private_bytes = static_cast<uint32_t *>(value);
      *private_bytes = std::max(*private_bytes, *required);
    }
  } else if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE) {
    if (const auto required = registry.required_group_for_symbol(symbol); required) {
      auto *group_bytes = static_cast<uint32_t *>(value);
      *group_bytes = std::max(*group_bytes, *required);
    }
  }
  return status;
}

constexpr int kStrictLoadRejectionExitCode = 92;

[[nodiscard]] hsa_status_t reject_code_object_load(
    const HookConfig &config, hsa_status_t status, uint64_t reader, std::string_view reason,
    FaultInstallationEvidence &fault_installation_evidence, std::string_view cause = {}) {
  const bool terminate = config.policy == HookPolicy::Strict;
  // Strict policy exits without unwinding this stack. Flush any applied
  // mutation evidence before the process terminates so fault qualification can
  // distinguish a rejected replacement from an installed one.
  fault_installation_evidence.emit();
  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] ConSan load rejection reader=%llu reason=%.*s "
               "status=%d policy=%s action=%s exit_code=%s",
               static_cast<unsigned long long>(reader), static_cast<int>(reason.size()),
               reason.data(), static_cast<int>(status), hook_policy_name(config.policy),
               terminate ? "terminate" : "return-error", terminate ? "92" : "none");
  if (!cause.empty())
    std::fprintf(stderr, " cause=%.*s", static_cast<int>(cause.size()), cause.data());
  std::fputc('\n', stderr);
  std::fflush(stderr);
  // A caller that ignores the HSA code-object load error can retain a null
  // kernel symbol and crash later during launch. HIP does this for some
  // precompiled PyTorch fat objects. Strict policy promises fail-closed
  // execution, so stop at the attributable loader failure instead of handing
  // an unusable executable back to a client that may continue regardless.
  if (terminate)
    std::_Exit(kStrictLoadRejectionExitCode);
  return status;
}

[[nodiscard]] hsa_status_t
reject_unresolved_semantic_arch(const HookConfig &config, uint64_t reader,
                                FaultInstallationEvidence &fault_installation_evidence) {
  // Reject unconditionally rather than turning an internal semantic mismatch
  // into an apparently successful, uninstrumented sanitizer run. The original
  // code object is intact, but this result cannot safely support either an
  // instrumented replacement or a coverage verdict.
  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] ConSan internal invariant violation: transform returned "
               "semantic inventory without a resolved target architecture\n");
  return reject_code_object_load(config, HSA_STATUS_ERROR_INVALID_CODE_OBJECT, reader,
                                 "internal-semantic-arch-missing", fault_installation_evidence);
}

hsa_status_t HSA_API rj_dbi_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader,
    const char *options, hsa_loaded_code_object_t *loaded_code_object) {
  auto *original_load = layer().load_agent_code_object();
  if (original_load == nullptr)
    return HSA_STATUS_ERROR;

  auto config = layer().config();
  if (!config) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] DBI hook layer is inactive during load\n");
    return HSA_STATUS_ERROR;
  }
  if (!refresh_report_config_from_env(&*config))
    return HSA_STATUS_ERROR;
  observe_process_fault_requirement(config->fault_require_exactly_one);

  TransformLoadState transform_state;
  AutoReportLoadGuard auto_report_guard(layer().core_table());
  auto &replacement_storage = transform_state.replacement_storage;
  auto &patch_result_storage = transform_state.patch_result_storage;
  auto &static_coverage_storage = transform_state.static_coverage_storage;
  auto &deferred_binding = transform_state.deferred_binding;
  auto &live_fault_auto_report_capacity_inventory =
      transform_state.live_fault_auto_report_capacity_inventory;
  hsa_code_object_reader_t reader_to_load = code_object_reader;
  hsa_code_object_reader_t replacement_reader{};
  bool using_replacement_reader = false;
  bool replacement_storage_retained = false;
  bool replacement_instrumentation_selected = false;
  std::optional<ScopedInstrumentationTimer> instrumentation_timer;
  const auto release_replacement_storage = [&] {
    if (!replacement_storage_retained) {
      replacement_storage.reset();
      return;
    }
    const std::vector<uint8_t> *storage_key = replacement_storage.get();
    const std::weak_ptr<const std::vector<uint8_t>> storage_lifetime = replacement_storage;
    // Relinquish the load call's owner while the registry still owns and
    // charges the allocation. Registry release then destroys the final owner
    // before another admission can enter its locked accounting domain.
    replacement_storage.reset();
    ReplacementCodeObjectStorageRegistry::instance().release(executable, storage_key);
    assert(storage_lifetime.expired() &&
           "replacement allocation outlived its retained-image accounting");
    replacement_storage_retained = false;
  };
  InstallAction install_action = InstallAction::LoadOriginal;
  ProcessFaultApplicationReservation process_fault_application_reservation;
  ProcessFaultReservationOutcome process_fault_reservation_outcome =
      ProcessFaultReservationOutcome::MutationAlreadyInstalled;
  bool process_fault_reservation_attempted = false;
  FaultInstallationEvidence fault_installation_evidence(code_object_reader.handle);
  const auto record_static_coverage = [&](bool replacement_installed) {
    if (!static_coverage_storage)
      return;
    if (replacement_instrumentation_selected && !replacement_installed) {
      mark_static_coverage_uninstrumented(*static_coverage_storage);
    }
    StaticCoverageRegistry::instance().record(*static_coverage_storage);
    static_coverage_storage.reset();
  };

  const HsaCodeObjectReaderRegistry::ReaderBytes reader_bytes =
      code_object_reader_registry().lookup(code_object_reader.handle);
  if (reader_bytes) {
    const uint8_t *bytes = reader_bytes.bytes;
    const size_t size = reader_bytes.size;
    if (!config->kernel_name_allowlist.empty()) {
      log_message(kLogInfo, "ConSan kernel allowlist prefilter reader=%llu bytes=%zu outcome=begin",
                  static_cast<unsigned long long>(code_object_reader.handle), size);
      const rocjitsu::KernelNameIndexMatch match = rocjitsu::match_kernel_name_index(
          std::span<const uint8_t>(bytes, size), config->kernel_name_allowlist);
      if (match == rocjitsu::KernelNameIndexMatch::NoMatch) {
        log_message(kLogInfo,
                    "ConSan kernel allowlist prefilter reader=%llu bytes=%zu outcome=skipped "
                    "reason=no-matching-entry",
                    static_cast<unsigned long long>(code_object_reader.handle), size);
        return original_load(executable, agent, code_object_reader, options, loaded_code_object);
      }
      if (match == rocjitsu::KernelNameIndexMatch::Indeterminate) {
        log_message(kLogVerbose,
                    "ConSan kernel allowlist prefilter reader=%llu bytes=%zu outcome=deferred "
                    "reason=indeterminate-kernel-index",
                    static_cast<unsigned long long>(code_object_reader.handle), size);
      }
    }
    instrumentation_timer.emplace();
    // Reader handles may be destroyed and reused by the HSA runtime. Keep a
    // process-local identity for this particular load so retained coverage can
    // be joined to the correspondingly numbered captured object without
    // conflating two lifetimes of the same opaque handle.
    const uint64_t load_id = g_dump_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint64_t dump_id = config->dump_dir.empty() ? 0 : load_id;
    dump_code_object_bytes(*config, dump_id, code_object_reader.handle, "original",
                           std::span<const uint8_t>(bytes, size));

    const auto waitcheck_begin = std::chrono::steady_clock::now();
    (void)run_waitcheck_preflight(std::span<const uint8_t>(bytes, size), code_object_reader.handle,
                                  config->kernel_name_allowlist);
    log_message(kLogInfo, "ConSan waitcheck timing reader=%llu elapsed_ms=%.3f",
                static_cast<unsigned long long>(code_object_reader.handle),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                          waitcheck_begin)
                    .count());

    const RuntimeCapabilities runtime_capabilities =
        query_hsa_runtime_capabilities(layer().core_table(), agent);
    const auto &request = static_cast<const Request &>(*config);
    const auto &transform_policy = static_cast<const TransformPolicy &>(*config);
    const auto &runtime_policy = static_cast<const RuntimePolicy &>(*config);
    const auto &debug_overrides = static_cast<const DebugOverrides &>(*config);
    const auto &configured_runtime_resources = static_cast<const BoundRuntimeResources &>(*config);
    BoundRuntimeResources runtime_resources;
    // Parsing retains ignored, foreign-mode settings so it can diagnose
    // them. Do not let those settings masquerade as a binding of the selected
    // mode's typed evidence contract.
    if (request.mode == Mode::SuperCollider) {
      runtime_resources.supercollider_report_buffer_address =
          configured_runtime_resources.supercollider_report_buffer_address;
      if (runtime_resources.supercollider_report_buffer_address)
        runtime_resources.scope = configured_runtime_resources.scope;
    } else if (request.mode == Mode::Default) {
      runtime_resources = configured_runtime_resources;
      runtime_resources.supercollider_report_buffer_address.reset();
      if (!runtime_resources.report_buffer_address)
        runtime_resources.scope = RuntimeResourceScope::Unbound;
    }
    MutationRequest mutation_request = static_cast<const MutationRequest &>(*config);
    // Exact fault selection is process-scoped in the HSA hook: workloads can
    // load runtime-helper code objects before the one containing the reviewed
    // site.  The process-wide atomic reservation below admits only the first match,
    // while retained fault telemetry lets the validation driver reject zero
    // matches.  Enforcing this inside each code-object transform would reject
    // every preceding nonmatching object.
    mutation_request.fault_require_exactly_one = false;
    if (request.mode != Mode::None) {
      const std::optional<TransformReservationEstimate> reservation =
          transform_major_image_reservation(size, transform_policy.patched_image_growth_limit);
      const bool relative_growth = transform_policy.patched_image_growth_limit.kind ==
                                   PatchedImageGrowthLimitKind::InputPercent;
      log_message(
          kLogInfo,
          "ConSan transform admission request reader=%llu input_image=%zu reservation=%s "
          "phase=%s phase_input_copies=%llu phase_maximum_copies=%llu "
          "growth_policy=%s growth_value=%llu",
          static_cast<unsigned long long>(code_object_reader.handle), size,
          reservation ? std::to_string(reservation->reservation_bytes).c_str() : "uint64-overflow",
          reservation ? reservation->phase_name() : "unavailable",
          static_cast<unsigned long long>(reservation ? reservation->input_image_copies() : 0),
          static_cast<unsigned long long>(reservation ? reservation->maximum_image_copies() : 0),
          relative_growth ? "input-percent" : "absolute-bytes",
          static_cast<unsigned long long>(
              relative_growth ? transform_policy.patched_image_growth_limit.input_percent
                              : transform_policy.patched_image_growth_limit.absolute_bytes));
      std::optional<ProcessTransformAdmissionRegistry::AdmissionResult> admission;
      if (reservation) {
        admission = transform_state.acquire(reservation->reservation_bytes,
                                            config->process_concurrent_transform_limit_bytes);
      }
      const bool accounting_overflow =
          !reservation ||
          (admission &&
           admission->outcome ==
               ProcessTransformAdmissionRegistry::AdmissionOutcome::AccountingOverflow);
      const bool limit_exceeded =
          admission &&
          admission->outcome == ProcessTransformAdmissionRegistry::AdmissionOutcome::LimitExceeded;
      if (accounting_overflow && !config->process_concurrent_transform_limit_bytes) {
        std::fprintf(
            stderr,
            "[rocjitsu-dbi-hooks] warning: ConSan transform reservation accounting overflow: "
            "reader=%llu input_image=%zu live=%llu reservation=%s process_ceiling=unlimited; "
            "continuing without a process reservation\n",
            static_cast<unsigned long long>(code_object_reader.handle), size,
            static_cast<unsigned long long>(admission ? admission->live_bytes : 0),
            reservation ? std::to_string(reservation->reservation_bytes).c_str()
                        : "uint64-overflow");
      } else if (accounting_overflow || limit_exceeded) {
        StaticCoverageRegistry::instance().record_unclassified_incomplete_code_object();
        const char *rejection_reason = "process-concurrent-transform-accounting";
        if (limit_exceeded) {
          rejection_reason = "process-concurrent-transform-limit";
          std::fprintf(
              stderr,
              "[rocjitsu-dbi-hooks] ConSan process concurrent transform limit exceeded: "
              "reader=%llu input_image=%zu live=%llu reservation=%llu required=%llu limit=%llu\n",
              static_cast<unsigned long long>(code_object_reader.handle), size,
              static_cast<unsigned long long>(admission->live_bytes),
              static_cast<unsigned long long>(admission->reservation_bytes),
              static_cast<unsigned long long>(*admission->required_bytes),
              static_cast<unsigned long long>(*admission->limit_bytes));
        } else {
          std::fprintf(
              stderr,
              "[rocjitsu-dbi-hooks] ConSan transform reservation accounting overflow: "
              "reader=%llu input_image=%zu live=%llu reservation=%s process_ceiling=%llu\n",
              static_cast<unsigned long long>(code_object_reader.handle), size,
              static_cast<unsigned long long>(admission ? admission->live_bytes : 0),
              reservation ? std::to_string(reservation->reservation_bytes).c_str()
                          : "uint64-overflow",
              static_cast<unsigned long long>(*config->process_concurrent_transform_limit_bytes));
        }
        if (config->fail_closed || config->require_patch) {
          return reject_code_object_load(*config, HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                                         code_object_reader.handle, rejection_reason,
                                         fault_installation_evidence);
        }
        log_message(kLogInfo, "ConSan transform admission failed; loading original reader=%llu",
                    static_cast<unsigned long long>(code_object_reader.handle));
        return original_load(executable, agent, code_object_reader, options, loaded_code_object);
      } else if (admission && *admission) {
        log_message(kLogInfo,
                    "ConSan transform admission reader=%llu input_image=%zu reservation=%llu "
                    "phase=%s phase_input_copies=%llu phase_maximum_copies=%llu "
                    "live_before=%llu required=%llu process_ceiling=%s",
                    static_cast<unsigned long long>(code_object_reader.handle), size,
                    static_cast<unsigned long long>(admission->reservation_bytes),
                    reservation->phase_name(),
                    static_cast<unsigned long long>(reservation->input_image_copies()),
                    static_cast<unsigned long long>(reservation->maximum_image_copies()),
                    static_cast<unsigned long long>(admission->live_bytes),
                    static_cast<unsigned long long>(*admission->required_bytes),
                    admission->limit_bytes ? std::to_string(*admission->limit_bytes).c_str()
                                           : "unlimited");
      }
    }
    size_t process_prior_fault_applications = 0;
    if ((config->fault_require_exactly_one && !config->fault_dry_run) ||
        config->fault_load_occurrence) {
      MutationRequest probe_mutation = mutation_request;
      BoundRuntimeResources probe_resources = runtime_resources;
      // Keep the configured mode so internal lowering performs dry-run
      // planning; mode=None intentionally skips every ConSan planning step.
      // The probe result is discarded, so retaining the mode cannot install
      // instrumentation but does let the exact fault resolver identify this
      // dynamic code-object load.
      probe_mutation.fault_dry_run = true;
      probe_mutation.fault_require_exactly_one = false;
      probe_resources.report_buffer_address.reset();
      probe_resources.report_buffer_size = 0;
      probe_resources.report_layout.reset();
      probe_resources.report_generation = 0;
      probe_resources.report_dispatch_id = 0;
      probe_resources.scope = probe_resources.supercollider_report_buffer_address
                                  ? RuntimeResourceScope::CodeObject
                                  : RuntimeResourceScope::Unbound;
      const TransformResult probe_transform = run_transform(
          std::span<const uint8_t>(bytes, size), request, transform_policy, runtime_policy,
          debug_overrides, probe_mutation, runtime_capabilities, probe_resources);
      const MutationTally &probe_fault = probe_transform.mutation.fault;
      const bool matched = probe_fault.has_plan();
      if (probe_fault.has_ambiguous_plan()) {
        std::fprintf(stderr,
                     "[rocjitsu-dbi-hooks] ConSan fault load selection is ambiguous: "
                     "one reader planned %zu mutations\n",
                     probe_fault.planned);
        return reject_code_object_load(*config, HSA_STATUS_ERROR_INVALID_CODE_OBJECT,
                                       code_object_reader.handle, "ambiguous-fault-load-selection",
                                       fault_installation_evidence);
      }
      bool selected = matched;
      if (config->fault_load_occurrence) {
        std::optional<FaultLoadSelection> selection;
        if (matched)
          selection = layer().observe_fault_load_match();
        selected = selection && selection->selected;
        log_message(kLogInfo,
                    "ConSan fault load selection reader=%llu site=%s matched=%s "
                    "requested_occurrence=%u observed_occurrence=%llu selected=%s overflow=%s",
                    static_cast<unsigned long long>(code_object_reader.handle),
                    config->fault_site_identity.c_str(), matched ? "true" : "false",
                    *config->fault_load_occurrence,
                    static_cast<unsigned long long>(selection ? selection->occurrence : 0),
                    selected ? "true" : "false",
                    selection && selection->overflow ? "true" : "false");
        if (selection && selection->overflow)
          return reject_code_object_load(*config, HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                                         code_object_reader.handle, "fault-load-selection-overflow",
                                         fault_installation_evidence);
      }
      if (selected && config->fault_require_exactly_one && !config->fault_dry_run) {
        process_fault_reservation_attempted = true;
        process_fault_reservation_outcome = process_fault_application_reservation.reserve(
            std::chrono::milliseconds(config->fault_reservation_timeout_ms),
            &process_prior_fault_applications);
        selected = process_fault_reservation_outcome == ProcessFaultReservationOutcome::Reserved;
        if (!selected) {
          const std::string_view outcome =
              process_fault_reservation_outcome_name(process_fault_reservation_outcome);
          if (process_fault_reservation_outcome ==
                  ProcessFaultReservationOutcome::ContentionTimeout ||
              process_fault_reservation_outcome ==
                  ProcessFaultReservationOutcome::ReentrantContention) {
            std::fprintf(stderr,
                         "[rocjitsu-dbi-hooks] ConSan fault reservation warning reader=%llu "
                         "outcome=%.*s process_prior_applied=%zu\n",
                         static_cast<unsigned long long>(code_object_reader.handle),
                         static_cast<int>(outcome.size()), outcome.data(),
                         process_prior_fault_applications);
          } else {
            log_message(kLogInfo,
                        "ConSan fault reservation reader=%llu outcome=%.*s "
                        "process_prior_applied=%zu",
                        static_cast<unsigned long long>(code_object_reader.handle),
                        static_cast<int>(outcome.size()), outcome.data(),
                        process_prior_fault_applications);
          }
        }
      }
      if (!selected)
        disable_fault_mutations(&mutation_request);
    }
    if (request.mode == Mode::SuperCollider &&
        config->supercollider_report_mode == SuperColliderReportMode::Auto &&
        !runtime_resources.supercollider_report_buffer_address) {
      AutomaticTransformPreparation preparation = run_automatic_prepare(
          std::span<const uint8_t>(bytes, size), request, transform_policy, runtime_policy,
          debug_overrides, mutation_request, runtime_capabilities);
      if (auto *completed = std::get_if<TransformResult>(&preparation)) {
        patch_result_storage = std::move(*completed);
        const auto *requirements = patch_result_storage->evidence_requirements
                                       ? std::get_if<SuperColliderEvidenceRequirements>(
                                             &*patch_result_storage->evidence_requirements)
                                       : nullptr;
        if (patch_result_storage->outcome == TransformOutcome::ModifiedValid &&
            (!requirements || !requirements->well_formed())) {
          log_message(kLogInfo,
                      "ConSan SuperCollider automatic report rejected reader=%llu: missing or "
                      "invalid typed "
                      "evidence requirements",
                      static_cast<unsigned long long>(code_object_reader.handle));
          patch_result_storage->discard_replacement(
              "SuperCollider automatic report requires typed evidence requirements");
          if (config->fail_closed)
            return reject_code_object_load(
                *config, HSA_STATUS_ERROR_OUT_OF_RESOURCES, code_object_reader.handle,
                "supercollider-report-missing-evidence-requirements", fault_installation_evidence);
        }
      } else {
        DeferredBinding deferred = std::move(std::get<DeferredBinding>(preparation));
        const auto *supercollider_evidence_requirements =
            deferred.evidence_requirements()
                ? std::get_if<SuperColliderEvidenceRequirements>(&*deferred.evidence_requirements())
                : nullptr;
        if (!supercollider_evidence_requirements ||
            !supercollider_evidence_requirements->well_formed()) {
          log_message(kLogInfo,
                      "ConSan SuperCollider automatic report rejected reader=%llu: missing or "
                      "invalid typed "
                      "evidence requirements",
                      static_cast<unsigned long long>(code_object_reader.handle));
          patch_result_storage = cancel_automatic_transform(
              std::move(deferred),
              "SuperCollider automatic report requires typed evidence requirements");
          if (config->fail_closed)
            return reject_code_object_load(
                *config, HSA_STATUS_ERROR_OUT_OF_RESOURCES, code_object_reader.handle,
                "supercollider-report-missing-evidence-requirements", fault_installation_evidence);
        } else {
          uint64_t auto_report_address = 0;
          uint64_t auto_report_generation = 0;
          const ContractIssue capability_issue = validate_runtime_capabilities(
              runtime_capabilities, supercollider_evidence_requirements->runtime_requirements);
          const bool allocated = capability_issue == ContractIssue::None &&
                                 AutoSuperColliderReportBufferRegistry::instance().allocate(
                                     layer().core_table(), agent, code_object_reader.handle,
                                     &auto_report_address, &auto_report_generation);
          if (!allocated) {
            if (capability_issue != ContractIssue::None) {
              const std::string_view reason = contract_issue_name(capability_issue);
              log_message(kLogInfo,
                          "ConSan SuperCollider automatic report rejected by runtime capabilities "
                          "reader=%llu reason=%.*s",
                          static_cast<unsigned long long>(code_object_reader.handle),
                          static_cast<int>(reason.size()), reason.data());
            }
            std::fprintf(stderr,
                         "[rocjitsu-dbi-hooks] ConSan SuperCollider automatic non-trapping report "
                         "allocation failed; analysis incomplete, refusing trap fallback\n");
            patch_result_storage = cancel_automatic_transform(
                std::move(deferred),
                "SuperCollider automatic report allocation failed; original code loaded "
                "without instrumentation");
            if (config->fail_closed || config->require_patch)
              return reject_code_object_load(
                  *config, HSA_STATUS_ERROR_OUT_OF_RESOURCES, code_object_reader.handle,
                  "supercollider-report-allocation", fault_installation_evidence);
          } else {
            runtime_resources.supercollider_report_buffer_address = auto_report_address;
            runtime_resources.scope = RuntimeResourceScope::Executable;
            auto_report_guard.note_sc(code_object_reader.handle, auto_report_generation);
            deferred_binding = std::move(deferred);
          }
        }
      }
    }
    if (request.mode == Mode::Default && runtime_resources.report_buffer_address &&
        !runtime_resources.report_dispatch_id) {
      // Explicit report buffers are commonly installed immediately before a
      // fixture's code object is loaded. Give their emitted records the same
      // nonzero code-object identity used by automatic report buffers.
      runtime_resources.report_dispatch_id = code_object_reader.handle;
    }
    std::optional<uint64_t> registered_report_generation;
    if (request.mode == Mode::Default && !runtime_resources.report_buffer_address &&
        request.auto_report_buffer_size != 0) {
      const bool live_fault_transform =
          fault_mutations_enabled(mutation_request) && !mutation_request.fault_dry_run;
      log_message(kLogInfo, "ConSan inventory begin reader=%llu bytes=%zu",
                  static_cast<unsigned long long>(code_object_reader.handle), size);
      const auto inventory_begin = std::chrono::steady_clock::now();
      AutomaticTransformPreparation preparation = run_automatic_prepare(
          std::span<const uint8_t>(bytes, size), request, transform_policy, runtime_policy,
          debug_overrides, mutation_request, runtime_capabilities);
      std::optional<DeferredBinding> deferred_inventory;
      if (auto *completed = std::get_if<TransformResult>(&preparation)) {
        patch_result_storage = std::move(*completed);
        const bool typed_evidence_available =
            visit_evidence_requirements(*patch_result_storage, [](const auto &) {});
        if (patch_result_storage->outcome == TransformOutcome::ModifiedValid &&
            !typed_evidence_available) {
          log_message(kLogInfo,
                      "ConSan auto report rejected reader=%llu: missing or invalid typed "
                      "evidence requirements",
                      static_cast<unsigned long long>(code_object_reader.handle));
          reject_report_plan(code_object_reader.handle, /*required_size=*/0,
                             config->auto_report_buffer_size, "missing_evidence_requirements");
          patch_result_storage->discard_replacement(
              "ConSan automatic report requires typed evidence requirements");
          if (config->fail_closed)
            return reject_code_object_load(
                *config, HSA_STATUS_ERROR_OUT_OF_RESOURCES, code_object_reader.handle,
                "report-missing-evidence-requirements", fault_installation_evidence);
        }
      } else {
        deferred_inventory = std::move(std::get<DeferredBinding>(preparation));
      }
      log_message(kLogInfo, "ConSan inventory end reader=%llu elapsed_ms=%.3f",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                            inventory_begin)
                      .count());
      if (deferred_inventory &&
          !deferred_inventory->program_inventory().has_resolved_semantic_arch())
        return reject_unresolved_semantic_arch(*config, code_object_reader.handle,
                                               fault_installation_evidence);
      uint64_t auto_report_address = 0;
      uint64_t auto_report_size = 0;
      uint64_t auto_report_generation = 0;
      uint64_t required_report_size = 0;
      uint64_t requested_report_size = 0;
      ReportBufferLayout report_layout;
      std::optional<ReportBufferLayout> planned_report_layout;
      std::optional<AutoReportInventory> planned_report_inventory;
      std::optional<RuntimeCapabilityRequirements> evidence_runtime_requirements;
      AutoReportInventory report_inventory;
      AutoReportPlan report_plan;
      bool inventory_has_semantic_observations = false;
      const bool evidence_available =
          deferred_inventory &&
          visit_evidence_requirements(
              deferred_inventory->evidence_requirements(), [&](const auto &requirements) {
                report_inventory = requirements.sizing_inventory;
                report_plan = requirements.abi_plan;
                evidence_runtime_requirements = requirements.runtime_requirements;
                inventory_has_semantic_observations =
                    requirements.sizing_inventory.has_semantic_observations();
              });
      bool auto_report_plan_available = evidence_available && inventory_has_semantic_observations;
      if (deferred_inventory && !evidence_available) {
        log_message(kLogInfo,
                    "ConSan auto report rejected reader=%llu: missing or invalid typed "
                    "evidence requirements",
                    static_cast<unsigned long long>(code_object_reader.handle));
        reject_report_plan(code_object_reader.handle, /*required_size=*/0,
                           config->auto_report_buffer_size, "missing_evidence_requirements");
        if (config->fail_closed)
          return reject_code_object_load(
              *config, HSA_STATUS_ERROR_OUT_OF_RESOURCES, code_object_reader.handle,
              "report-missing-evidence-requirements", fault_installation_evidence);
        if (deferred_inventory) {
          patch_result_storage = cancel_automatic_transform(
              std::move(*deferred_inventory),
              "ConSan automatic report requires typed evidence requirements");
          deferred_inventory.reset();
        }
      } else if (deferred_inventory && !inventory_has_semantic_observations) {
        log_message(kLogInfo,
                    "ConSan auto report buffer skipped reader=%llu: no ConSan report sites",
                    static_cast<unsigned long long>(code_object_reader.handle));
      }

      if (auto_report_plan_available && !patch_result_storage) {
        planned_report_inventory = report_inventory;
        required_report_size = report_plan.required_bytes;
        requested_report_size = report_plan.required_bytes;
        report_layout = report_plan.layout;
        planned_report_layout = report_plan.complete_layout();
        log_message(kLogInfo,
                    "ConSan auto report plan reader=%llu outcome=%s reason=%s "
                    "required_bytes=%llu cap_bytes=%llu per_buffer_ceiling=%llu "
                    "process_ceiling=%llu access_ranges=%llu barriers=%llu atomics=%llu "
                    "watchpoint_banks=%llu watchpoints=%llu",
                    static_cast<unsigned long long>(code_object_reader.handle),
                    auto_report_plan_outcome_name(report_plan.outcome).data(),
                    auto_report_plan_reason_name(report_plan.reason).data(),
                    static_cast<unsigned long long>(report_plan.required_bytes),
                    static_cast<unsigned long long>(config->auto_report_buffer_size),
                    static_cast<unsigned long long>(report_plan.ceiling_bytes),
                    static_cast<unsigned long long>(kAutoReportProcessCeilingBytes),
                    static_cast<unsigned long long>(report_inventory.access_range_count),
                    static_cast<unsigned long long>(report_inventory.barrier_event_count),
                    static_cast<unsigned long long>(report_inventory.atomic_event_count),
                    static_cast<unsigned long long>(report_inventory.range_bank_count),
                    static_cast<unsigned long long>(report_inventory.watchpoint_count));
      }
      bool report_runtime_capabilities_available = true;
      if (auto_report_plan_available && !patch_result_storage) {
        RuntimeCapabilityRequirements report_requirements;
        if (evidence_runtime_requirements) {
          report_requirements = *evidence_runtime_requirements;
        } else {
          report_requirements.host_device_visible_memory = true;
          report_requirements.device_atomic_publication = true;
          report_requirements.minimum_report_allocation_bytes = requested_report_size;
          report_requirements.executable_binding = true;
        }
        const ContractIssue capability_issue =
            validate_runtime_capabilities(runtime_capabilities, report_requirements);
        report_runtime_capabilities_available = capability_issue == ContractIssue::None;
        if (!report_runtime_capabilities_available) {
          const std::string_view reason = contract_issue_name(capability_issue);
          log_message(kLogInfo,
                      "ConSan auto report rejected by runtime capabilities "
                      "reader=%llu required_bytes=%llu reason=%.*s",
                      static_cast<unsigned long long>(code_object_reader.handle),
                      static_cast<unsigned long long>(requested_report_size),
                      static_cast<int>(reason.size()), reason.data());
          reject_report_plan(code_object_reader.handle, required_report_size,
                             config->auto_report_buffer_size, reason);
        }
      }
      if (auto_report_plan_available && report_runtime_capabilities_available &&
          !patch_result_storage &&
          allocate_report_buffer(
              layer().core_table(), agent, code_object_reader.handle, required_report_size,
              requested_report_size, config->auto_report_buffer_size, report_layout,
              config->track_barriers, config->track_atomics, &auto_report_address,
              &auto_report_size, &auto_report_generation)) {
        runtime_resources.scope = RuntimeResourceScope::Executable;
        runtime_resources.report_buffer_address = auto_report_address;
        runtime_resources.report_buffer_size = auto_report_size;
        runtime_resources.report_layout = planned_report_layout;
        runtime_resources.report_generation = auto_report_generation;
        registered_report_generation = auto_report_generation;
        auto_report_guard.note(code_object_reader.handle, auto_report_generation);
        runtime_resources.report_dispatch_id = code_object_reader.handle;
        if (live_fault_transform && planned_report_inventory)
          live_fault_auto_report_capacity_inventory = *planned_report_inventory;
        if (deferred_inventory) {
          deferred_binding = std::move(*deferred_inventory);
          deferred_inventory.reset();
        }
      } else if (auto_report_plan_available && !patch_result_storage && config->fail_closed) {
        return reject_code_object_load(*config, HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                                       code_object_reader.handle, "report-allocation",
                                       fault_installation_evidence);
      } else if (auto_report_plan_available && !patch_result_storage) {
        if (!live_fault_transform && deferred_inventory) {
          patch_result_storage = cancel_automatic_transform(
              std::move(*deferred_inventory), "ConSan automatic report allocation failed");
          deferred_inventory.reset();
        }
      }
    }

    log_message(kLogInfo, "ConSan patch begin reader=%llu bytes=%zu",
                static_cast<unsigned long long>(code_object_reader.handle), size);
    const auto patch_begin = std::chrono::steady_clock::now();
    if (!patch_result_storage) {
      if (deferred_binding) {
        patch_result_storage = resume_automatic_transform(
            std::span<const uint8_t>(bytes, size), runtime_resources, std::move(*deferred_binding));
      } else {
        patch_result_storage = run_transform(
            std::span<const uint8_t>(bytes, size), request, transform_policy, runtime_policy,
            debug_overrides, mutation_request, runtime_capabilities, runtime_resources);
      }
    }
    const TransformResult &transform_result = *patch_result_storage;
    const TransformDiagnosticReport transform_diagnostics =
        transform_diagnostic_report(transform_result);
    const MutationOutcome &mutation = transform_result.mutation;
    fault_installation_evidence.record_applied_mutations(mutation.fault.applied);
    if (live_fault_auto_report_capacity_inventory) {
      AutoReportInventory live_requirement;
      const bool live_evidence_available =
          visit_evidence_requirements(transform_result, [&](const auto &requirements) {
            live_requirement = requirements.sizing_inventory;
          });
      if (!live_evidence_available ||
          !auto_report_inventory_covers(*live_fault_auto_report_capacity_inventory,
                                        live_requirement)) {
        std::fprintf(stderr, "[rocjitsu-dbi-hooks] ConSan internal invariant violation: live fault "
                             "transform grew the automatic ConSan report inventory\n");
        reject_report_plan(code_object_reader.handle, runtime_resources.report_buffer_size,
                           config->auto_report_buffer_size, "live_fault_inventory_growth");
        return reject_code_object_load(*config, HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                                       code_object_reader.handle, "report-live-inventory-growth",
                                       fault_installation_evidence);
      }
    }
    if (!transform_result.program_inventory.has_resolved_semantic_arch())
      return reject_unresolved_semantic_arch(*config, code_object_reader.handle,
                                             fault_installation_evidence);
    if (registered_report_generation)
      register_report_metadata(code_object_reader.handle, *registered_report_generation,
                               transform_result.code_object.fingerprint,
                               transform_result.runtime_static_mapping());
    install_action = transform_result.install_action(config->fail_closed);
    replacement_instrumentation_selected = install_action == InstallAction::LoadReplacement;
    log_message(
        kLogInfo,
        "ConSan patch end reader=%llu modified=%s outcome=%s errors=%zu "
        "warnings=%zu patches=%zu patch_ms=%.3f",
        static_cast<unsigned long long>(code_object_reader.handle),
        transform_result.outcome == TransformOutcome::ModifiedValid ? "true" : "false",
        transform_outcome_name(transform_result.outcome), transform_result.errors.size(),
        transform_result.warnings.size(), transform_diagnostics.patches.size(),
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - patch_begin)
            .count());
    for (const std::string &warning : transform_result.warnings)
      log_message(kLogVerbose, "%s", warning.c_str());
    if (!transform_result.errors.empty()) {
      for (const std::string &error : transform_result.errors)
        std::fprintf(stderr, "[rocjitsu-dbi-hooks] %s\n", error.c_str());
      if (config->fail_closed)
        return reject_code_object_load(
            *config, HSA_STATUS_ERROR_INVALID_CODE_OBJECT, code_object_reader.handle,
            "transform-error", fault_installation_evidence,
            transform_result.transform_failure_cause
                ? transform_failure_cause_name(*transform_result.transform_failure_cause)
                : std::string_view{});
    }
    if (install_action == InstallAction::Reject) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] ConSan outcome %s is not installable "
                   "(fail_closed=%s)\n",
                   transform_outcome_name(transform_result.outcome),
                   config->fail_closed ? "true" : "false");
      return reject_code_object_load(*config, HSA_STATUS_ERROR_INVALID_CODE_OBJECT,
                                     code_object_reader.handle, "non-installable-transform-outcome",
                                     fault_installation_evidence);
    }
    const bool requires_dynamic_private_dispatch_adjustment =
        replacement_instrumentation_selected &&
        transform_result.dispatch_requirements.requires_packet_interception();
    if (requires_dynamic_private_dispatch_adjustment &&
        !layer().dispatch_packet_interception_enabled()) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] ConSan replacement requires dispatch-packet "
                           "interception to extend dynamic private segments\n");
      if (config->fail_closed) {
        return reject_code_object_load(
            *config, HSA_STATUS_ERROR_INVALID_CODE_OBJECT, code_object_reader.handle,
            "dynamic-private-dispatch-interception-unavailable", fault_installation_evidence);
      }
      log_message(kLogInfo,
                  "ConSan dispatch-packet interception unavailable; loading original reader=%llu",
                  static_cast<unsigned long long>(code_object_reader.handle));
      install_action = InstallAction::LoadOriginal;
    }

    // A large production object may produce hundreds of thousands of detailed
    // inventory, coverage, static-mapping, and diagnostic records below.
    // Preserve the stable line-oriented format while amortizing the stderr
    // lock and write cost.
    ScopedDetailedLogBatch detailed_log_batch;
    log_message(
        kLogInfo,
        "ConSan inventory reader=%llu mode=%s bytes=%zu modified=%s "
        "supercollider_delay_nops=%u fail_closed=%s check_trap_mode=%s probe_lds_check_trap=%s "
        "probe_flat_check_trap=%s fault_drop_barrier=%s "
        "init_owner_epoch=%s track_barriers=%s track_atomics=%s "
        "fault_barrier_index=%u "
        "supercollider_delay_mode=%s supercollider_delay_var_ssrc=%u "
        "patched_image_growth_limit_kind=%s patched_image_growth_limit_value=%llu "
        "process_concurrent_transform_limit_bytes=%s "
        "process_patched_image_limit_bytes=%s "
        "process_patched_image_growth_limit_bytes=%s "
        "max_patches=%u max_patches_source=%s tmp_vgpr=%s exec_save_sgpr=%s "
        "owner_source=%s owner_sgpr=%s owner_vgpr=%s epoch_vgpr=%s "
        "report_buffer=%s supercollider_report_marker=%u "
        "report_buffer=%s report_buffer_size=%llu "
        "auto_report_buffer_size=%llu require_patch=%s",
        static_cast<unsigned long long>(code_object_reader.handle),
        mode_name(request.mode.value_or(Mode::None)), transform_result.code_object.byte_size,
        transform_result.outcome == TransformOutcome::ModifiedValid ? "true" : "false",
        config->supercollider_delay_nops, config->fail_closed ? "true" : "false",
        check_trap_mode_name(config->check_trap_mode),
        config->probe_lds_check_trap ? "true" : "false",
        config->probe_flat_check_trap ? "true" : "false",
        config->fault_drop_barrier ? "true" : "false", config->init_owner_epoch ? "true" : "false",
        config->track_barriers ? "true" : "false", config->track_atomics ? "true" : "false",
        config->fault_barrier_index, delay_mode_name(config->supercollider_delay_mode),
        config->supercollider_delay_var_ssrc,
        patched_image_growth_limit_kind_name(transform_policy.patched_image_growth_limit.kind),
        static_cast<unsigned long long>(
            patched_image_growth_limit_value(transform_policy.patched_image_growth_limit)),
        config->process_concurrent_transform_limit_bytes
            ? std::to_string(*config->process_concurrent_transform_limit_bytes).c_str()
            : "unlimited",
        config->process_patched_image_limit_bytes
            ? std::to_string(*config->process_patched_image_limit_bytes).c_str()
            : "unlimited",
        config->process_patched_image_growth_limit_bytes
            ? std::to_string(*config->process_patched_image_growth_limit_bytes).c_str()
            : "unlimited",
        config->max_patches,
        config->max_patches_explicit ? "expert-limit" : "all-supported-default",
        config->scratch_vgpr ? std::to_string(*config->scratch_vgpr).c_str() : "auto",
        config->requested_exec_save_sgpr ? std::to_string(*config->requested_exec_save_sgpr).c_str()
                                         : "unset",
        owner_source_name(config->owner_source),
        config->requested_owner_sgpr ? std::to_string(*config->requested_owner_sgpr).c_str()
                                     : "unset",
        config->requested_owner_vgpr ? std::to_string(*config->requested_owner_vgpr).c_str()
                                     : "unset",
        config->requested_epoch_vgpr ? std::to_string(*config->requested_epoch_vgpr).c_str()
                                     : "unset",
        config->supercollider_report_buffer_address
            ? std::to_string(*config->supercollider_report_buffer_address).c_str()
            : "disabled",
        config->supercollider_report_marker,
        runtime_resources.report_buffer_address
            ? std::to_string(*runtime_resources.report_buffer_address).c_str()
            : "disabled",
        static_cast<unsigned long long>(runtime_resources.report_buffer_size),
        static_cast<unsigned long long>(config->auto_report_buffer_size),
        config->require_patch ? "true" : "false");
    log_message(
        kLogInfo,
        "ConSan code-object reader=%llu target=%s arch=%s text_sections=%zu "
        "kernels=%zu functions=%zu",
        static_cast<unsigned long long>(code_object_reader.handle),
        rj_code_target_name(transform_result.program_inventory.target()),
        rj_code_arch_name(rj_code_arch_for_target(transform_result.program_inventory.target())),
        transform_result.program_inventory.text_sections().size(),
        transform_result.program_inventory.kernels().size(),
        transform_result.program_inventory.functions().size());
    for (const FaultSiteDiagnostic &site : transform_diagnostics.fault_sites) {
      const OwnerLogFields owners =
          owner_log_fields(site.execution_owners, transform_result.program_inventory.kernels());
      log_message(kLogVerbose,
                  "ConSan fault site reader=%llu identity=%s kind=%s container=%s "
                  "container_kind=%s occurrence=%u text_offset=0x%llx file_offset=0x%llx "
                  "size=%u width_bits=%u mnemonic=%s role=%s operands=%s sync_event=%s "
                  "sync_sequence=%s sync_confidence=%s sync_memory_role=%s "
                  "ordinary_memory_support=%s owners=%zu owner_names=%s owner_proofs=%s",
                  static_cast<unsigned long long>(code_object_reader.handle), site.identity.c_str(),
                  fault_site_kind_name(site.kind), site.container_name.c_str(),
                  site.in_kernel ? "kernel" : "function", site.occurrence,
                  static_cast<unsigned long long>(site.text_offset),
                  static_cast<unsigned long long>(site.file_offset), site.size, site.width_bits,
                  site.mnemonic.c_str(), site.semantic_role.c_str(), site.decoded_operands.c_str(),
                  site.sync_event_identity ? site.sync_event_identity->c_str() : "-",
                  site.sync_sequence_identity ? site.sync_sequence_identity->c_str() : "-",
                  sync_confidence_name(site.sync_confidence),
                  sync_memory_role_name(site.sync_memory_role),
                  ordinary_memory_support_reason_name(site.ordinary_memory_support_reason),
                  site.execution_owners.size(), owners.names.c_str(), owners.proofs.c_str());
    }
    for (const BarrierMoveDestinationDiagnostic &destination :
         transform_diagnostics.barrier_move_destinations) {
      const OwnerLogFields owners = owner_log_fields(destination.execution_owners,
                                                     transform_result.program_inventory.kernels());
      std::string reason =
          barrier_move_destination_issue_message(destination.issue, destination.issue_detail);
      if (reason.empty())
        reason = "-";
      std::ranges::replace(reason, ' ', '-');
      const std::string structured_guard_block =
          destination.structured_guard_block_index
              ? std::to_string(*destination.structured_guard_block_index)
              : "-";
      const std::string structured_source_block =
          destination.structured_source_block_index
              ? std::to_string(*destination.structured_source_block_index)
              : "-";
      log_message(kLogVerbose,
                  "ConSan barrier destination reader=%llu identity=%s container=%s "
                  "container_kind=%s block=%u text_offset=0x%llx file_offset=0x%llx size=%u "
                  "mnemonic=%s memory_operation=%s suitable=%s reason=%s cfg_contract=%s "
                  "structured_guard_block=%s structured_source_block=%s "
                  "structured_guard_offset=0x%llx structured_source_offset=0x%llx "
                  "owners=%zu owner_names=%s owner_proofs=%s",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  destination.identity.c_str(), destination.container_name.c_str(),
                  destination.in_kernel ? "kernel" : "function", destination.basic_block_index,
                  static_cast<unsigned long long>(destination.text_offset),
                  static_cast<unsigned long long>(destination.file_offset), destination.size,
                  destination.mnemonic.c_str(), destination.memory_operation ? "true" : "false",
                  destination.suitable() ? "true" : "false", reason.c_str(),
                  barrier_move_cfg_contract_name(destination.cfg_contract),
                  structured_guard_block.c_str(), structured_source_block.c_str(),
                  static_cast<unsigned long long>(destination.structured_guard_offset.value_or(0)),
                  static_cast<unsigned long long>(destination.structured_source_offset.value_or(0)),
                  destination.execution_owners.size(), owners.names.c_str(), owners.proofs.c_str());
    }
    for (const FaultMutationDiagnostic &plan : transform_diagnostics.fault_mutations) {
      std::string members;
      for (const std::string &identity : plan.ordered_member_identities) {
        if (!members.empty())
          members += ',';
        members += identity;
      }
      log_message(kLogInfo,
                  "ConSan fault plan reader=%llu dry_run=%s mutation=%s primary=%s "
                  "companion=%s logical_sequence=%s members=%s destination=%s direction=%s "
                  "cfg_contract=%s "
                  "original_barrier_id=%s target_barrier_id=%s original_barrier_scope=%s "
                  "target_barrier_scope=%s",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  config->fault_dry_run ? "true" : "false", fault_mutation_kind_name(plan.kind),
                  plan.primary_identity.c_str(),
                  plan.companion_identity ? plan.companion_identity->c_str() : "-",
                  plan.logical_sequence_identity ? plan.logical_sequence_identity->c_str() : "-",
                  members.empty() ? "-" : members.c_str(),
                  plan.destination_identity ? plan.destination_identity->c_str() : "-",
                  barrier_move_direction_name(plan.barrier_move_direction),
                  barrier_move_cfg_contract_name(plan.barrier_move_cfg_contract),
                  plan.original_barrier_id ? std::to_string(*plan.original_barrier_id).c_str()
                                           : "-",
                  plan.target_barrier_id ? std::to_string(*plan.target_barrier_id).c_str() : "-",
                  barrier_scope_name(plan.original_barrier_scope),
                  barrier_scope_name(plan.target_barrier_scope));
    }
    emit_fault_summary_message(
        config->fault_require_exactly_one,
        "ConSan fault summary process=%llu reader=%llu requested=%zu planned=%zu "
        "applied=%zu process_prior_applied=%zu "
        "reservation=%.*s "
        "require_exactly_one=%s destructive_incomplete_barrier_drop=%s "
        "completing_conditional_barrier_move=%s "
        "destructive_divergent_barrier_move=%s",
        static_cast<unsigned long long>(::getpid()),
        static_cast<unsigned long long>(code_object_reader.handle), mutation.fault.requested,
        mutation.fault.planned, mutation.fault.applied, process_prior_fault_applications,
        static_cast<int>(
            (process_fault_reservation_attempted
                 ? process_fault_reservation_outcome_name(process_fault_reservation_outcome)
                 : std::string_view{"not-requested"})
                .size()),
        (process_fault_reservation_attempted
             ? process_fault_reservation_outcome_name(process_fault_reservation_outcome)
             : std::string_view{"not-requested"})
            .data(),
        config->fault_require_exactly_one ? "true" : "false",
        config->fault_allow_destructive_incomplete_barrier_drop ? "true" : "false",
        config->fault_allow_completing_conditional_barrier_move ? "true" : "false",
        config->fault_allow_destructive_divergent_barrier_move ? "true" : "false");
    if (config->fault_require_exactly_one && !config->fault_dry_run &&
        mutation.fault.applied > 1u) {
      return reject_code_object_load(*config, HSA_STATUS_ERROR_INVALID_CODE_OBJECT,
                                     code_object_reader.handle, "multiple-fault-mutations-applied",
                                     fault_installation_evidence);
    }
    log_message(kLogInfo,
                "ConSan SuperCollider perturb summary reader=%llu requested=%zu planned=%zu "
                "applied=%zu max=%u "
                "required=%u sleep=%u",
                static_cast<unsigned long long>(code_object_reader.handle),
                mutation.supercollider_perturbation.requested,
                mutation.supercollider_perturbation.planned,
                mutation.supercollider_perturbation.applied, config->supercollider_perturb_max,
                config->supercollider_perturb_required_count, config->supercollider_perturb_sleep);
    const SynchronizationInventoryView sync = transform_result.program_inventory.sync();
    for (const SyncSequence &sequence : sync.sync_sequences) {
      const std::vector<ExecutionOwner> sequence_owners = sync.execution_owners(sequence);
      const ProgramContainer *sequence_container = sync.container(sequence);
      const OwnerLogFields owners =
          owner_log_fields(sequence_owners, transform_result.program_inventory.kernels());
      std::string reason = sequence.confidence_reason;
      std::ranges::replace(reason, ' ', '-');
      std::string members;
      const ProgramSite *presentation_source = nullptr;
      size_t source_count = 0;
      size_t addressed_source_count = 0;
      for (const SyncEventId member_id : sequence.member_event_ids) {
        const SyncEvent *event = sync.find_event(member_id);
        if (event == nullptr)
          continue;
        if (!members.empty())
          members += ',';
        members += event->identity;
        const ProgramSite *source = sync.source(*event);
        if (source == nullptr)
          continue;
        ++source_count;
        const bool addressed_source = source->get_if<OrdinaryMemorySite>() != nullptr ||
                                      source->get_if<AtomicSite>() != nullptr;
        if (addressed_source) {
          ++addressed_source_count;
          presentation_source = source;
        } else if (presentation_source == nullptr) {
          presentation_source = source;
        }
      }
      if (addressed_source_count > 1u || (addressed_source_count == 0u && source_count != 1u))
        presentation_source = nullptr;
      const OrdinaryMemorySite *ordinary = presentation_source == nullptr
                                               ? nullptr
                                               : presentation_source->get_if<OrdinaryMemorySite>();
      const AtomicSite *atomic =
          presentation_source == nullptr ? nullptr : presentation_source->get_if<AtomicSite>();
      const DecodedMemorySite *memory = ordinary;
      if (memory == nullptr)
        memory = atomic;
      const BarrierSite *barrier =
          presentation_source == nullptr ? nullptr : presentation_source->get_if<BarrierSite>();
      const std::string block =
          sequence.basic_block_index ? std::to_string(*sequence.basic_block_index) : "-";
      const std::string static_offset =
          memory != nullptr && memory->raw_ioffset ? std::to_string(*memory->raw_ioffset) : "-";
      const std::string raw_scope =
          memory != nullptr && memory->raw_scope ? std::to_string(*memory->raw_scope) : "-";
      const std::string participant_count =
          sequence.participant_count ? std::to_string(*sequence.participant_count) : "-";
      const std::string participant_mask =
          sequence.participant_mask ? std::to_string(*sequence.participant_mask) : "-";
      const std::string barrier_id =
          sequence.barrier_id ? std::to_string(*sequence.barrier_id) : "-";
      const std::string barrier_raw_selector = barrier != nullptr && barrier->raw_operand_selector
                                                   ? std::to_string(*barrier->raw_operand_selector)
                                                   : "-";
      const std::string barrier_literal_width = barrier != nullptr && barrier->literal_width_bits
                                                    ? std::to_string(*barrier->literal_width_bits)
                                                    : "-";
      const std::string barrier_literal_value = barrier != nullptr && barrier->literal_value
                                                    ? std::to_string(*barrier->literal_value)
                                                    : "-";
      const std::string barrier_raw_simm16 =
          barrier != nullptr && barrier->raw_simm16 ? std::to_string(*barrier->raw_simm16) : "-";
      std::array<char, 32> release_wait_offset{};
      if (sequence.release_wait_text_offset) {
        std::snprintf(release_wait_offset.data(), release_wait_offset.size(), "0x%llx",
                      static_cast<unsigned long long>(*sequence.release_wait_text_offset));
      } else {
        release_wait_offset[0] = '-';
      }
      log_message(
          kLogVerbose,
          "ConSan sync sequence reader=%llu identity=%s kind=%s operation=%s "
          "address_source=%s memory_role=%s memory_role_confidence=%s rmw_outcome=%s "
          "confidence=%s reason=%s "
          "container=%s container_kind=%s block=%s begin_text_offset=0x%llx "
          "end_text_offset=0x%llx width_bits=%u static_offset=%s raw_scope=%s "
          "barrier_id=%s barrier_operand_source=%s barrier_raw_selector=%s "
          "barrier_literal_width_bits=%s barrier_literal_value=%s "
          "barrier_raw_simm16=%s barrier_scope=%s "
          "release_wait_text_offset=%s "
          "participant_count=%s participant_mask=%s members=%s "
          "owners=%zu owner_names=%s owner_proofs=%s",
          static_cast<unsigned long long>(code_object_reader.handle), sequence.identity.c_str(),
          sync_sequence_kind_name(sequence.kind), sync_operation_name(sequence.operation),
          sync_address_source_name(sequence.address_source),
          sync_memory_role_name(sequence.memory_role),
          sync_confidence_name(sequence.memory_role_confidence),
          sync_rmw_outcome_name(sequence.rmw_outcome), sync_confidence_name(sequence.confidence),
          reason.c_str(),
          sequence_container != nullptr ? sequence_container->name.c_str() : "<unresolved>",
          sequence_container != nullptr && sequence_container->is_kernel() ? "kernel" : "function",
          block.c_str(), static_cast<unsigned long long>(sequence.begin_text_offset),
          static_cast<unsigned long long>(sequence.end_text_offset),
          memory != nullptr ? memory->width_bits : 0u, static_offset.c_str(), raw_scope.c_str(),
          barrier_id.c_str(), barrier_operand_source_name(sequence.barrier_operand_source),
          barrier_raw_selector.c_str(), barrier_literal_width.c_str(),
          barrier_literal_value.c_str(), barrier_raw_simm16.c_str(),
          barrier_scope_name(sequence.barrier_scope), release_wait_offset.data(),
          participant_count.c_str(), participant_mask.c_str(), members.c_str(),
          sequence_owners.size(), owners.names.c_str(), owners.proofs.c_str());
    }
    if (request.mode == Mode::Default) {
      const ResourcePlanSummary &resource_summary = transform_diagnostics.resource_summary;
      log_message(kLogInfo,
                  "ConSan resources reader=%llu explicit=%zu dead=%zu "
                  "descriptor_growth=%zu spill=%zu unsupported=%zu "
                  "planned_spill_slot_bytes=%zu emitted_spill_patches=%zu "
                  "emitted_spill_slot_bytes=%zu alternative_attempts=%zu "
                  "alternative_selected=%zu alternative_rejected=%zu "
                  "alternative_superseded=%zu alternative_contributed=%zu "
                  "alternative_vetoed=%zu",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  resource_summary.explicit_plans, resource_summary.dead_plans,
                  resource_summary.descriptor_growth_plans, resource_summary.spill_plans,
                  resource_summary.unsupported_plans, resource_summary.planned_spill_slot_bytes,
                  resource_summary.emitted_spill_patches, resource_summary.emitted_spill_slot_bytes,
                  resource_summary.alternative_attempts, resource_summary.alternative_selected,
                  resource_summary.alternative_rejected, resource_summary.alternative_superseded,
                  resource_summary.alternative_contributed, resource_summary.alternative_vetoed);
      for (const ResourceFailureDiagnostic &failure : transform_diagnostics.resource_failures) {
        log_message(kLogInfo,
                    "ConSan resource-failure reader=%llu site=%s reason=%s count=%zu "
                    "scratch_vgprs=%u..%u current_vgprs=%u..%u "
                    "max_referenced_vgprs=%u..%u ordinary_vgpr_limit=%u..%u "
                    "required_vgprs=%u..%u owners=%zu..%zu indirect_vgprs=%s",
                    static_cast<unsigned long long>(code_object_reader.handle),
                    resource_site_kind_name(failure.site_kind),
                    register_plan_reason_name(failure.reason), failure.count,
                    failure.min_scratch_vgprs, failure.max_scratch_vgprs, failure.min_current_vgprs,
                    failure.max_current_vgprs, failure.min_max_referenced_vgprs,
                    failure.max_max_referenced_vgprs, failure.min_ordinary_vgpr_limit,
                    failure.max_ordinary_vgpr_limit, failure.min_required_vgprs,
                    failure.max_required_vgprs, failure.min_owners, failure.max_owners,
                    failure.has_indirect_vgpr_access ? "true" : "false");
      }
      for (const ResourceAlternativeDiagnostic &alternative :
           transform_diagnostics.resource_alternatives) {
        log_message(
            kLogInfo,
            "ConSan resource-alternative reader=%llu site=%s candidate=%zu "
            "text_offset=0x%llx attempt=%zu kind=%s scratch_count=%u "
            "source=%s reason=%s outcome=%s",
            static_cast<unsigned long long>(code_object_reader.handle),
            resource_site_kind_name(alternative.site_kind), alternative.candidate_index,
            static_cast<unsigned long long>(alternative.text_offset), alternative.attempt_index,
            resource_plan_alternative_kind_name(alternative.kind), alternative.scratch_vgpr_count,
            register_allocation_source_name(alternative.source),
            register_plan_reason_name(alternative.reason),
            resource_plan_alternative_outcome_name(alternative.outcome));
      }
    }
    size_t candidate_kernel_count = 0;
    size_t skipped_kernel_count = 0;
    size_t blocked_kernel_count = 0;
    size_t supported_lds_site_count = 0;
    size_t flat_site_count = 0;
    size_t flat_group_hint_count = 0;
    size_t flat_private_hint_count = 0;
    size_t flat_maybe_group_hint_count = 0;
    size_t flat_maybe_private_hint_count = 0;
    size_t flat_global_hint_count = 0;
    size_t flat_unknown_hint_count = 0;
    size_t function_lds_site_count = 0;
    size_t function_supported_lds_site_count = 0;
    size_t function_flat_site_count = 0;
    size_t function_flat_group_hint_count = 0;
    size_t function_flat_private_hint_count = 0;
    size_t function_flat_maybe_group_hint_count = 0;
    size_t function_flat_maybe_private_hint_count = 0;
    size_t function_flat_global_hint_count = 0;
    size_t function_flat_unknown_hint_count = 0;
    for (const ProgramContainer &kernel : transform_result.program_inventory.kernels()) {
      switch (kernel.preflight_action) {
      case PreflightAction::Candidate:
        ++candidate_kernel_count;
        break;
      case PreflightAction::Skip:
        ++skipped_kernel_count;
        break;
      case PreflightAction::Blocked:
        ++blocked_kernel_count;
        break;
      case PreflightAction::NotRun:
        break;
      }
      flat_group_hint_count += kernel.stats.flat_group_hint_count;
      flat_private_hint_count += kernel.stats.flat_private_hint_count;
      flat_maybe_group_hint_count += kernel.stats.flat_maybe_group_hint_count;
      flat_maybe_private_hint_count += kernel.stats.flat_maybe_private_hint_count;
      flat_global_hint_count += kernel.stats.flat_global_hint_count;
      flat_unknown_hint_count += kernel.stats.flat_unknown_hint_count;
    }
    for (const ProgramContainer &function : transform_result.program_inventory.functions()) {
      function_flat_group_hint_count += function.stats.flat_group_hint_count;
      function_flat_private_hint_count += function.stats.flat_private_hint_count;
      function_flat_maybe_group_hint_count += function.stats.flat_maybe_group_hint_count;
      function_flat_maybe_private_hint_count += function.stats.flat_maybe_private_hint_count;
      function_flat_global_hint_count += function.stats.flat_global_hint_count;
      function_flat_unknown_hint_count += function.stats.flat_unknown_hint_count;
    }
    for (const ProgramSite &site : transform_result.program_inventory.access_sites()) {
      const ProgramContainer *container =
          transform_result.program_inventory.container(site.container);
      const bool function =
          container != nullptr && container->kind == ProgramContainerKind::Function;
      if (site.origin == AccessOrigin::Flat) {
        ++(function ? function_flat_site_count : flat_site_count);
      } else {
        if (function)
          ++function_lds_site_count;
        if (site.lowering.replay_guest_access.available())
          ++(function ? function_supported_lds_site_count : supported_lds_site_count);
      }
    }
    log_message(kLogInfo,
                "ConSan summary reader=%llu kernels=%zu candidates=%zu skips=%zu "
                "blocked=%zu supported_lds_sites=%zu flat_sites=%zu flat_group_hints=%zu "
                "flat_private_hints=%zu flat_maybe_group_hints=%zu "
                "flat_maybe_private_hints=%zu flat_global_hints=%zu "
                "flat_unknown_hints=%zu functions=%zu function_lds_sites=%zu "
                "function_supported_lds_sites=%zu function_flat_sites=%zu "
                "function_flat_group_hints=%zu function_flat_private_hints=%zu "
                "function_flat_maybe_group_hints=%zu function_flat_maybe_private_hints=%zu "
                "function_flat_global_hints=%zu function_flat_unknown_hints=%zu patches=%zu "
                "modified=%s",
                static_cast<unsigned long long>(code_object_reader.handle),
                transform_result.program_inventory.kernels().size(), candidate_kernel_count,
                skipped_kernel_count, blocked_kernel_count, supported_lds_site_count,
                flat_site_count, flat_group_hint_count, flat_private_hint_count,
                flat_maybe_group_hint_count, flat_maybe_private_hint_count, flat_global_hint_count,
                flat_unknown_hint_count, transform_result.program_inventory.functions().size(),
                function_lds_site_count, function_supported_lds_site_count,
                function_flat_site_count, function_flat_group_hint_count,
                function_flat_private_hint_count, function_flat_maybe_group_hint_count,
                function_flat_maybe_private_hint_count, function_flat_global_hint_count,
                function_flat_unknown_hint_count, transform_diagnostics.patches.size(),
                transform_result.outcome == TransformOutcome::ModifiedValid ? "true" : "false");
    static_coverage_storage = compute_static_coverage(transform_result.coverage_ledger, *config);
    const StaticCoverage &static_coverage = *static_coverage_storage;
    log_message(
        kLogInfo,
        "ConSan coverage reader=%llu mode=%s "
        "analysis_complete=%s expert_limit=%s "
        "access_discovered=%llu access_supported=%llu access_selected=%llu "
        "access_patched=%llu access_unsupported=%llu access_resource_failed=%llu "
        "access_placement_or_lowering_failed=%llu access_expert_limit_omitted=%llu "
        "barrier_discovered=%llu barrier_supported=%llu barrier_selected=%llu "
        "barrier_patched=%llu barrier_unsupported=%llu barrier_resource_failed=%llu "
        "barrier_placement_or_lowering_failed=%llu barrier_expert_limit_omitted=%llu "
        "atomic_discovered=%llu atomic_supported=%llu atomic_selected=%llu "
        "atomic_patched=%llu atomic_unsupported=%llu atomic_resource_failed=%llu "
        "atomic_placement_or_lowering_failed=%llu atomic_expert_limit_omitted=%llu "
        "fence_discovered=%llu fence_supported=%llu fence_selected=%llu "
        "fence_patched=%llu fence_unsupported=%llu fence_resource_failed=%llu "
        "fence_placement_or_lowering_failed=%llu fence_expert_limit_omitted=%llu load=%llu",
        static_cast<unsigned long long>(code_object_reader.handle), mode_name(*request.mode),
        static_coverage.complete ? "true" : "false",
        static_coverage.expert_limit ? "true" : "false",
        static_cast<unsigned long long>(static_coverage.access.discovered),
        static_cast<unsigned long long>(static_coverage.access.supported),
        static_cast<unsigned long long>(static_coverage.access.selected),
        static_cast<unsigned long long>(static_coverage.access.patched),
        static_cast<unsigned long long>(static_coverage.access.unsupported),
        static_cast<unsigned long long>(static_coverage.access.resource_failed),
        static_cast<unsigned long long>(static_coverage.access.placement_or_lowering_failed),
        static_cast<unsigned long long>(static_coverage.access.expert_limit_omitted),
        static_cast<unsigned long long>(static_coverage.barrier.discovered),
        static_cast<unsigned long long>(static_coverage.barrier.supported),
        static_cast<unsigned long long>(static_coverage.barrier.selected),
        static_cast<unsigned long long>(static_coverage.barrier.patched),
        static_cast<unsigned long long>(static_coverage.barrier.unsupported),
        static_cast<unsigned long long>(static_coverage.barrier.resource_failed),
        static_cast<unsigned long long>(static_coverage.barrier.placement_or_lowering_failed),
        static_cast<unsigned long long>(static_coverage.barrier.expert_limit_omitted),
        static_cast<unsigned long long>(static_coverage.atomic.discovered),
        static_cast<unsigned long long>(static_coverage.atomic.supported),
        static_cast<unsigned long long>(static_coverage.atomic.selected),
        static_cast<unsigned long long>(static_coverage.atomic.patched),
        static_cast<unsigned long long>(static_coverage.atomic.unsupported),
        static_cast<unsigned long long>(static_coverage.atomic.resource_failed),
        static_cast<unsigned long long>(static_coverage.atomic.placement_or_lowering_failed),
        static_cast<unsigned long long>(static_coverage.atomic.expert_limit_omitted),
        static_cast<unsigned long long>(static_coverage.fence.discovered),
        static_cast<unsigned long long>(static_coverage.fence.supported),
        static_cast<unsigned long long>(static_coverage.fence.selected),
        static_cast<unsigned long long>(static_coverage.fence.patched),
        static_cast<unsigned long long>(static_coverage.fence.unsupported),
        static_cast<unsigned long long>(static_coverage.fence.resource_failed),
        static_cast<unsigned long long>(static_coverage.fence.placement_or_lowering_failed),
        static_cast<unsigned long long>(static_coverage.fence.expert_limit_omitted),
        static_cast<unsigned long long>(load_id));
    std::unordered_map<uint64_t, std::vector<std::string_view>> source_containers_by_text_offset;
    if (g_log_level.load(std::memory_order_relaxed) >= kLogDebug) {
      for (const ProgramSite &site : transform_result.program_inventory.program_sites()) {
        const ProgramContainer *container =
            transform_result.program_inventory.container(site.container);
        if (container == nullptr)
          continue;
        auto &names = source_containers_by_text_offset[site.text_offset()];
        if (std::ranges::find(names, container->name) == names.end())
          names.push_back(container->name);
      }
      for (auto &entry : source_containers_by_text_offset)
        std::ranges::sort(entry.second);
    }
    const auto log_typed_coverage_sites = [&](const auto &decisions, ResourceSiteKind kind,
                                              const auto &reason_name) {
      if (g_log_level.load(std::memory_order_relaxed) < kLogDebug)
        return;
      for (const auto &decision : decisions) {
        std::string_view disposition = "not_applicable";
        std::string_view outcome = "not_applicable";
        std::string_view lowering_reason = "semantic_not_applicable";
        std::string_view resource_reason = "none";
        if (decision.kind == SiteDecisionKind::Unsupported) {
          disposition = "unsupported";
          outcome = "unsupported";
          lowering_reason = "semantic_unsupported";
        } else if (decision.kind == SiteDecisionKind::Admitted) {
          disposition = "supported";
          const auto intent_ids = intent_ids_covering(transform_result, decision.semantic_site);
          bool all_instrumented = !intent_ids.empty();
          bool resource_rejected = false;
          bool placement_rejected = false;
          for (ProbeIntentId id : intent_ids) {
            const IntentCoverageEntry *entry = transform_result.coverage_ledger.intent_entry(id);
            all_instrumented &=
                entry != nullptr && entry->lowering == LoweringOutcomeKind::Instrumented;
            resource_rejected |=
                entry != nullptr && entry->lowering == LoweringOutcomeKind::ResourceRejected;
            placement_rejected |=
                entry != nullptr && entry->lowering == LoweringOutcomeKind::PlacementRejected;
          }
          if (all_instrumented) {
            outcome = "patched";
            lowering_reason = "none";
          } else if (resource_rejected) {
            outcome = "resource_failed";
            lowering_reason = "unsupported_resource_plan";
            resource_reason = "invalid_request";
            for (ProbeIntentId id : intent_ids) {
              const IntentCoverageEntry *entry = transform_result.coverage_ledger.intent_entry(id);
              if (entry != nullptr && entry->resource_rejection_reason) {
                resource_reason = register_plan_reason_name(*entry->resource_rejection_reason);
                break;
              }
            }
          } else if (placement_rejected) {
            outcome = "placement_or_lowering_failed";
            lowering_reason = "instrumentation_patch_missing";
          } else {
            outcome = "pending";
            lowering_reason = "none";
          }
        }
        const auto source_containers = source_containers_by_text_offset.find(
            decision.semantic_site.physical.original_text_offset);
        const std::string_view source =
            source_containers == source_containers_by_text_offset.end() ||
                    source_containers->second.empty()
                ? std::string_view{"<none>"}
                : source_containers->second.front();
        std::string reason(reason_name(decision.reason));
        std::ranges::replace(reason, '-', '_');
        log_message(
            kLogDebug,
            "ConSan coverage_site reader=%llu kind=%s disposition=%s reason=%s "
            "outcome=%s lowering_reason=%s resource_reason=%s "
            "container=%.*s scope=kernel text=0x%llx mnemonic=unknown load=%llu",
            static_cast<unsigned long long>(code_object_reader.handle),
            resource_site_kind_name(kind), disposition.data(), reason.c_str(), outcome.data(),
            lowering_reason.data(), resource_reason.data(), static_cast<int>(source.size()),
            source.data(),
            static_cast<unsigned long long>(decision.semantic_site.physical.original_text_offset),
            static_cast<unsigned long long>(load_id));
      }
    };
    log_typed_coverage_sites(
        transform_result.coverage_ledger.site_decisions(), ResourceSiteKind::Access,
        [](AccessPolicyReason reason) { return access_policy_reason_name(reason); });
    log_typed_coverage_sites(
        transform_result.coverage_ledger.barrier_site_decisions(), ResourceSiteKind::Barrier,
        [](BarrierPolicyReason reason) { return barrier_policy_reason_name(reason); });
    log_typed_coverage_sites(
        transform_result.coverage_ledger.atomic_site_decisions(), ResourceSiteKind::Atomic,
        [](AtomicPolicyReason reason) { return atomic_policy_reason_name(reason); });
    log_typed_coverage_sites(
        transform_result.coverage_ledger.fence_site_decisions(), ResourceSiteKind::Fence,
        [](FencePolicyReason reason) { return fence_policy_reason_name(reason); });
    for (const PatchDiagnostic &patch : transform_diagnostics.patches) {
      if (g_log_level.load(std::memory_order_relaxed) < kLogDebug)
        break;
      const std::string scratch_vgpr =
          patch.scratch_vgpr ? std::to_string(*patch.scratch_vgpr) : "-";
      const std::string private_epoch_offset =
          patch.persistent_epoch_private_offset
              ? std::to_string(*patch.persistent_epoch_private_offset)
              : "-";
      const auto *scalar_vcc_spill = patch.scalar_vcc_spill ? &*patch.scalar_vcc_spill : nullptr;
      const std::string scalar_vcc_spill_vgpr =
          scalar_vcc_spill ? std::to_string(scalar_vcc_spill->reservoir_vgpr) : "-";
      const std::string scalar_vcc_spill_sgpr =
          scalar_vcc_spill ? std::to_string(scalar_vcc_spill->vcc_save_sgpr) : "-";
      log_message(kLogDebug,
                  "ConSan proof patch reader=%llu kind=%s anchor=0x%llx "
                  "trampoline=0x%llx original_size=%u trampoline_size=%u scratch_vgpr=%s "
                  "scalar_vcc_spill_sgpr=%s scalar_vcc_spill_vgpr=%s "
                  "scalar_vcc_spill_vgpr_count=%u "
                  "private_epoch_offset=%s spilled_vgprs=%u "
                  "private_bytes=%u dynamic_private_addend=%u",
                  static_cast<unsigned long long>(code_object_reader.handle), patch.kind.c_str(),
                  static_cast<unsigned long long>(patch.anchor_offset),
                  static_cast<unsigned long long>(patch.trampoline_offset), patch.original_size,
                  patch.trampoline_size, scratch_vgpr.c_str(), scalar_vcc_spill_sgpr.c_str(),
                  scalar_vcc_spill_vgpr.c_str(),
                  scalar_vcc_spill ? scalar_vcc_spill->reservoir_vgpr_count() : 0u,
                  private_epoch_offset.c_str(), patch.spilled_vgpr_count,
                  patch.required_private_segment_size, patch.dynamic_private_segment_addend);
    }
    detailed_log_batch.flush();
    if (config->require_patch && !config->fault_dry_run &&
        !has_instrumented_site(transform_result)) {
      const bool required =
          (request.mode == Mode::SuperCollider &&
           require_patch_applies_to(transform_result, *config)) ||
          (request.mode == Mode::Default && require_patch_applies_to(transform_result));
      if (required) {
        std::fprintf(stderr,
                     "[rocjitsu-dbi-hooks] RJ_CONSAN_REQUIRE_PATCH requested, but no relevant "
                     "access, barrier, atomic, or fence patch was applied to a code object with "
                     "supported ConSan sites\n");
        record_static_coverage(false);
        return reject_code_object_load(
            *config, HSA_STATUS_ERROR_INVALID_CODE_OBJECT, code_object_reader.handle,
            "required-instrumentation-missing", fault_installation_evidence);
      }
    }

    if (install_action == InstallAction::LoadReplacement) {
      dump_code_object_bytes(*config, dump_id, code_object_reader.handle, "patched",
                             std::span<const uint8_t>(transform_result.replacement.data(),
                                                      transform_result.replacement.size()));
    } else {
      // All metadata consumers above are finished. Discard the unused final
      // image and its reservation together before invoking the original
      // loader, which may block or re-enter the hook.
      transform_state.discard_image_and_release(patch_result_storage->replacement);
    }
  } else {
    log_message(kLogInfo, "ConSan pass-through reader=%llu bytes=unavailable",
                static_cast<unsigned long long>(code_object_reader.handle));
  }

  if (patch_result_storage && install_action == InstallAction::LoadReplacement) {
    auto *original_create = layer().create_from_memory();
    if (original_create == nullptr) {
      record_static_coverage(false);
      return HSA_STATUS_ERROR;
    }

    const size_t input_size = patch_result_storage->code_object.byte_size;
    const size_t replacement_size = patch_result_storage->replacement.size();
    const uint64_t replacement_growth_bytes =
        replacement_size > input_size ? static_cast<uint64_t>(replacement_size - input_size) : 0;
    try {
      replacement_storage = std::make_shared<const std::vector<uint8_t>>(
          std::move(patch_result_storage->replacement));
    } catch (const std::bad_alloc &) {
      replacement_storage.reset();
    }
    std::optional<ReplacementCodeObjectStorageRegistry::RetainResult> retain_result;
    if (replacement_storage) {
      retain_result = ReplacementCodeObjectStorageRegistry::instance().retain(
          executable, replacement_storage, replacement_growth_bytes,
          static_cast<uint64_t>(replacement_size), config->process_patched_image_growth_limit_bytes,
          config->process_patched_image_limit_bytes);
    }
    const bool process_growth_limit_exceeded =
        retain_result &&
        retain_result->outcome ==
            ReplacementCodeObjectStorageRegistry::RetainOutcome::ProcessGrowthLimitExceeded;
    const bool process_image_limit_exceeded =
        retain_result &&
        retain_result->outcome ==
            ReplacementCodeObjectStorageRegistry::RetainOutcome::ProcessImageLimitExceeded;
    if (!replacement_storage || !retain_result || !*retain_result) {
      const char *rejection_reason = "replacement-storage-retention";
      if (process_growth_limit_exceeded) {
        rejection_reason = "process-patched-image-growth-limit";
        if (retain_result->required_total_growth_bytes) {
          std::fprintf(stderr,
                       "[rocjitsu-dbi-hooks] ConSan process patched-image growth limit exceeded: "
                       "live=%llu replacement_growth=%llu replacement_image=%llu "
                       "required=%llu limit=%llu\n",
                       static_cast<unsigned long long>(retain_result->live_growth_bytes),
                       static_cast<unsigned long long>(retain_result->replacement_growth_bytes),
                       static_cast<unsigned long long>(retain_result->replacement_image_bytes),
                       static_cast<unsigned long long>(*retain_result->required_total_growth_bytes),
                       static_cast<unsigned long long>(*retain_result->growth_limit_bytes));
        } else {
          std::fprintf(stderr,
                       "[rocjitsu-dbi-hooks] ConSan process patched-image growth limit exceeded: "
                       "live=%llu replacement_growth=%llu replacement_image=%llu "
                       "required=uint64-overflow limit=%llu\n",
                       static_cast<unsigned long long>(retain_result->live_growth_bytes),
                       static_cast<unsigned long long>(retain_result->replacement_growth_bytes),
                       static_cast<unsigned long long>(retain_result->replacement_image_bytes),
                       static_cast<unsigned long long>(*retain_result->growth_limit_bytes));
        }
      } else if (process_image_limit_exceeded) {
        rejection_reason = "process-patched-image-limit";
        if (retain_result->required_total_image_bytes) {
          std::fprintf(stderr,
                       "[rocjitsu-dbi-hooks] ConSan process patched-image limit exceeded: "
                       "live=%llu replacement_image=%llu replacement_growth=%llu "
                       "required=%llu limit=%llu\n",
                       static_cast<unsigned long long>(retain_result->live_image_bytes),
                       static_cast<unsigned long long>(retain_result->replacement_image_bytes),
                       static_cast<unsigned long long>(retain_result->replacement_growth_bytes),
                       static_cast<unsigned long long>(*retain_result->required_total_image_bytes),
                       static_cast<unsigned long long>(*retain_result->image_limit_bytes));
        } else {
          std::fprintf(stderr,
                       "[rocjitsu-dbi-hooks] ConSan process patched-image limit exceeded: "
                       "live=%llu replacement_image=%llu replacement_growth=%llu "
                       "required=uint64-overflow limit=%llu\n",
                       static_cast<unsigned long long>(retain_result->live_image_bytes),
                       static_cast<unsigned long long>(retain_result->replacement_image_bytes),
                       static_cast<unsigned long long>(retain_result->replacement_growth_bytes),
                       static_cast<unsigned long long>(*retain_result->image_limit_bytes));
        }
      } else if (retain_result && retain_result->outcome ==
                                      ReplacementCodeObjectStorageRegistry::RetainOutcome::
                                          GrowthAccountingOverflow) {
        rejection_reason = "process-patched-image-growth-accounting";
        std::fprintf(
            stderr,
            "[rocjitsu-dbi-hooks] ConSan process patched-image growth accounting overflow: "
            "live=%llu replacement_growth=%llu replacement_image=%llu\n",
            static_cast<unsigned long long>(retain_result->live_growth_bytes),
            static_cast<unsigned long long>(retain_result->replacement_growth_bytes),
            static_cast<unsigned long long>(retain_result->replacement_image_bytes));
      } else if (retain_result &&
                 retain_result->outcome ==
                     ReplacementCodeObjectStorageRegistry::RetainOutcome::ImageAccountingOverflow) {
        rejection_reason = "process-patched-image-accounting";
        std::fprintf(stderr,
                     "[rocjitsu-dbi-hooks] ConSan process patched-image accounting overflow: "
                     "live=%llu replacement_image=%llu replacement_growth=%llu\n",
                     static_cast<unsigned long long>(retain_result->live_image_bytes),
                     static_cast<unsigned long long>(retain_result->replacement_image_bytes),
                     static_cast<unsigned long long>(retain_result->replacement_growth_bytes));
      } else {
        std::fprintf(stderr,
                     "[rocjitsu-dbi-hooks] failed to retain replacement code-object storage\n");
      }
      replacement_storage.reset();
      transform_state.discard_image_and_release(patch_result_storage->replacement);
      if (config->fail_closed) {
        record_static_coverage(false);
        return reject_code_object_load(*config, HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                                       code_object_reader.handle, rejection_reason,
                                       fault_installation_evidence);
      }
      log_message(kLogInfo,
                  "ConSan replacement storage retention failed reason=%s; "
                  "loading original reader=%llu",
                  rejection_reason, static_cast<unsigned long long>(code_object_reader.handle));
    } else {
      replacement_storage_retained = true;
      transform_state.release();
    }

    const hsa_status_t reader_status =
        replacement_storage_retained
            ? original_create(replacement_storage->data(), replacement_storage->size(),
                              &replacement_reader)
            : HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    if (!replacement_storage_retained) {
      // Fail-open policy already selected the untouched reader above.
    } else if (reader_status != HSA_STATUS_SUCCESS) {
      release_replacement_storage();
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] failed to create replacement patched reader: %d\n",
                   static_cast<int>(reader_status));
      if (config->fail_closed) {
        record_static_coverage(false);
        return reject_code_object_load(*config, reader_status, code_object_reader.handle,
                                       "replacement-reader-creation", fault_installation_evidence);
      }
      log_message(kLogInfo,
                  "ConSan replacement reader creation failed; loading original reader=%llu",
                  static_cast<unsigned long long>(code_object_reader.handle));
    } else {
      reader_to_load = replacement_reader;
      using_replacement_reader = true;
      log_message(kLogInfo, "ConSan replacement reader=%llu original_reader=%llu bytes=%zu",
                  static_cast<unsigned long long>(replacement_reader.handle),
                  static_cast<unsigned long long>(code_object_reader.handle),
                  replacement_storage->size());
    }
  }

  // Loading and binding a replacement is part of its one-off startup cost.
  // Loading an untouched object is native runtime work and must not be charged
  // to ConSan merely because semantic inspection preceded it.
  if (instrumentation_timer && !using_replacement_reader)
    instrumentation_timer->stop();
  hsa_status_t load_status =
      original_load(executable, agent, reader_to_load, options, loaded_code_object);
  if (load_status != HSA_STATUS_SUCCESS && using_replacement_reader && !config->fail_closed) {
    auto *original_destroy = layer().destroy();
    if (original_destroy != nullptr)
      (void)original_destroy(replacement_reader);
    using_replacement_reader = false;
    release_replacement_storage();
    if (loaded_code_object != nullptr)
      *loaded_code_object = {};
    log_message(kLogInfo,
                "ConSan replacement load failed status=%d; retrying untouched original reader=%llu",
                static_cast<int>(load_status),
                static_cast<unsigned long long>(code_object_reader.handle));
    if (instrumentation_timer)
      instrumentation_timer->stop();
    load_status = original_load(executable, agent, code_object_reader, options, loaded_code_object);
  }
  if (load_status == HSA_STATUS_SUCCESS && using_replacement_reader && patch_result_storage) {
    auto_report_guard.bind_to_executable(executable);
    if (patch_result_storage->mutation.fault.has_application()) {
      fault_installation_evidence.mark_installed();
      process_fault_application_reservation.commit_applied_mutation();
    }
    KernelPrivateDispatchRegistry::instance().note_requirements(
        executable, patch_result_storage->dispatch_requirements);
  }
  if (using_replacement_reader) {
    auto *original_destroy = layer().destroy();
    if (original_destroy != nullptr)
      (void)original_destroy(replacement_reader);
  }
  if (load_status != HSA_STATUS_SUCCESS)
    release_replacement_storage();
  if (load_status == HSA_STATUS_SUCCESS && patch_result_storage)
    KernelPrivateDispatchRegistry::instance().note_code_object(*patch_result_storage);
  record_static_coverage(load_status == HSA_STATUS_SUCCESS && using_replacement_reader);
  if (instrumentation_timer)
    instrumentation_timer->stop();
  return load_status;
}

#if defined(__GNUC__) || defined(__clang__)
#define RJ_HOOK_EXPORT __attribute__((visibility("default")))
#else
#define RJ_HOOK_EXPORT
#endif

extern "C" RJ_HOOK_EXPORT bool OnLoad(HsaApiTable *table, uint64_t runtime_version,
                                      uint64_t failed_tool_count,
                                      const char *const *failed_tool_names) {
  (void)runtime_version;
  (void)failed_tool_count;
  (void)failed_tool_names;

  if (!rocjitsu::hooks::retain_hsa_tool_dso())
    return false;

  auto config = parse_config();
  if (!config) {
    report_config_rejection();
    return false;
  }

  g_log_level.store(config->log_level, std::memory_order_relaxed);
  InstrumentationClock::instance().reset();
  if (!config->enabled) {
    log_message(kLogInfo, "ConSan is disabled; not installing wrappers");
    return true;
  }

  if (!layer().install(table, *config))
    return false;
  // Some applications retain an HSA reference until process exit, so ROCR
  // never invokes OnUnload. The layer and DSO remain available for cached
  // runtime callbacks in later process-exit handlers.
  static const bool exit_handler_registered =
      std::atexit([] { layer().uninstall(/*process_exit=*/true); }) == 0;
  if (!exit_handler_registered) {
    layer().uninstall();
    return false;
  }
  return true;
}

extern "C" RJ_HOOK_EXPORT void OnUnload() { layer().uninstall(); }

extern "C" RJ_HOOK_EXPORT void
rj_dbi_test_set_consan_transform_override(TransformOverride override) {
  g_test_transform_override.store(override, std::memory_order_release);
  g_test_retry_count.store(0, std::memory_order_relaxed);
}

extern "C" RJ_HOOK_EXPORT void rj_dbi_test_set_log_sink_override(LogSinkOverride override) {
  g_test_log_sink_override.store(override, std::memory_order_release);
}

extern "C" RJ_HOOK_EXPORT size_t rj_dbi_test_consan_retry_count() {
  return g_test_retry_count.load(std::memory_order_relaxed);
}

extern "C" RJ_HOOK_EXPORT uint64_t rj_dbi_consan_instrumentation_nanoseconds() {
  return InstrumentationClock::instance().elapsed_nanoseconds();
}

extern "C" RJ_HOOK_EXPORT bool rj_dbi_test_advance_report_generation(uint64_t generation) {
  return advance_report_generation_for_test(generation);
}

extern "C" RJ_HOOK_EXPORT uint32_t rj_dbi_consan_checkpoint_after_device_synchronize() {
  return static_cast<uint32_t>(layer().checkpoint_after_device_synchronize());
}

extern "C" RJ_HOOK_EXPORT uint32_t rj_dbi_consan_begin_epoch_analysis_window() {
  return static_cast<uint32_t>(layer().set_epoch_analysis_window(true));
}

extern "C" RJ_HOOK_EXPORT uint32_t rj_dbi_consan_end_epoch_analysis_window() {
  return static_cast<uint32_t>(layer().set_epoch_analysis_window(false));
}

} // namespace rocjitsu::consan::hook
