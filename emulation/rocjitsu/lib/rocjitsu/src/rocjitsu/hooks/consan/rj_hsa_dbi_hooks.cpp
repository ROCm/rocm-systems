// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hsa_dbi_hooks.cpp
/// @brief HSA tools load-time hook for opt-in rocJITsu DBI instrumentation.
///
/// @details ROCR loads this shared library through `HSA_TOOLS_LIB` during
/// `hsa_init()`. This initial DBI hook only parses configuration, installs the
/// code-object reader/load wrappers, logs observed loads when requested, and
/// routes memory-backed reader bytes through the selected ConSan DBI flavor.

#include "hsa/hsa_api_trace_minimal.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_hook_internal.h"

#include "rocjitsu/analysis/waitcheck.h"
#include "rocjitsu/checked_byte_budget.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_process_byte_budget.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_replay_provenance.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_sampled_sync.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_transform_memory.h"
#include "rocjitsu/hooks/hsa_tool_lifetime.h"
#include "util/arena_alloc.h"
#include "util/intrusive_list.h"

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
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace rocjitsu::consan_hook {

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

[[nodiscard]] bool consan_log_level_enabled(int required_level) {
  return g_log_level.load(std::memory_order_relaxed) >= required_level;
}

std::mutex &log_mutex();

using ConSanTransformOverride = rocjitsu::ConSanResult (*)(std::span<const uint8_t>,
                                                           const rocjitsu::ConSanOptions &);
std::atomic<ConSanTransformOverride> g_test_consan_transform_override{nullptr};
std::atomic<size_t> g_test_consan_moi_retry_count{0};

rocjitsu::ConSanResult run_consan_transform(std::span<const uint8_t> bytes,
                                            const rocjitsu::ConSanOptions &options) {
  if (const ConSanTransformOverride override =
          g_test_consan_transform_override.load(std::memory_order_acquire))
    return override(bytes, options);
  return rocjitsu::try_patch_consan(bytes, options);
}

rocjitsu::ConSanResult retry_consan_moi_transform(std::span<const uint8_t> bytes,
                                                  const rocjitsu::ConSanOptions &options,
                                                  rocjitsu::ConSanResult inventory) {
  // Unit tests replace the whole transform boundary and expect both phases to
  // flow through that seam. Production reuses the immutable semantic
  // inventory and reruns only MOI planning/lowering with the allocated buffer.
  if (const ConSanTransformOverride override =
          g_test_consan_transform_override.load(std::memory_order_acquire)) {
    g_test_consan_moi_retry_count.fetch_add(1, std::memory_order_relaxed);
    return override(bytes, options);
  }
  const rocjitsu::ConSanMoiInventoryRetryConfig retry{
      .report =
          {
              .buffer_address = options.moi_report_buffer_address,
              .buffer_size = options.moi_report_buffer_size,
              .layout = options.moi_report_layout,
              .generation = options.moi_report_generation,
              .dispatch_id = options.moi_report_dispatch_id,
          },
      .fault = rocjitsu::ConSanFaultMutationRetryConfig::from_options(options),
  };
  return rocjitsu::retry_patch_consan_moi_from_inventory(std::move(inventory), retry, bytes);
}

struct GroupSegmentRegionSearch {
  CoreApiTable *core = nullptr;
  std::optional<uint32_t> size_bytes;
};

hsa_status_t HSA_API select_group_segment_region(hsa_region_t region, void *data) {
  auto *search = static_cast<GroupSegmentRegionSearch *>(data);
  hsa_region_segment_t segment{};
  hsa_status_t status =
      search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_SEGMENT, &segment);
  if (status != HSA_STATUS_SUCCESS || segment != HSA_REGION_SEGMENT_GROUP)
    return status;
  size_t size_bytes = 0;
  status = search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_SIZE, &size_bytes);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  if (size_bytes == 0 || size_bytes > std::numeric_limits<uint32_t>::max())
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  search->size_bytes = static_cast<uint32_t>(size_bytes);
  return HSA_STATUS_INFO_BREAK;
}

[[nodiscard]] std::optional<uint32_t> runtime_group_segment_size_bytes(CoreApiTable *core,
                                                                       hsa_agent_t agent) {
  if (core == nullptr || core->hsa_agent_iterate_regions_fn == nullptr ||
      core->hsa_region_get_info_fn == nullptr) {
    return std::nullopt;
  }
  GroupSegmentRegionSearch search{.core = core, .size_bytes = std::nullopt};
  const hsa_status_t status =
      core->hsa_agent_iterate_regions_fn(agent, select_group_segment_region, &search);
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
    return std::nullopt;
  return search.size_bytes;
}

enum class WaitcheckPreflightOutcome { NotApplicable, Passed, HazardReported, AnalysisFailed };

void print_waitcheck_issue(uint64_t reader, const rocjitsu::AmdGpuCodeObject &code_object,
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

[[nodiscard]] WaitcheckPreflightOutcome run_waitcheck_preflight(std::span<const uint8_t> bytes,
                                                                uint64_t reader) {
  bool recognized_code_object = false;
  try {
    rocjitsu::AmdGpuCodeObject code_object(bytes.data(), bytes.size());
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
    const rocjitsu::WaitcheckReport report = rocjitsu::analyze_waitcnts(code_object, arch, options);
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
                report.memory_events_tracked, report.kernels_analyzed, report.kernels_discovered);
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

[[nodiscard]] bool require_patch_applies_to(const rocjitsu::ConSanResult &result,
                                            const HookConfig &config) {
  const bool has_selected_access = config.probe_lds_check_trap || config.probe_flat_check_trap;
  if (!has_selected_access)
    return false;
  if (!result.sc_access_coverage_resolved)
    return result.flavor == rocjitsu::ConSanFlavor::SuperCollider;
  return std::ranges::any_of(result.sc_access_coverage_sites, [&](const auto &site) {
    return sc_access_coverage_kind_enabled(site.kind, config) &&
           (!site.evaluated || site.supported);
  });
}

[[nodiscard]] bool
is_supported_require_patch_moi_candidate(const rocjitsu::ConSanMoiCandidate &candidate,
                                         rj_code_arch_t arch) {
  if (candidate.kind != rocjitsu::ConSanLdsAccessKind::Read &&
      candidate.kind != rocjitsu::ConSanLdsAccessKind::Write)
    return false;
  if (candidate.size == 0 || candidate.size % sizeof(uint32_t) != 0)
    return false;
  if (candidate.width_bits == 0 || candidate.width_bits % 8u != 0)
    return false;
  if (!candidate.addr_vgpr)
    return false;

  if (candidate.source == rocjitsu::ConSanMoiCandidateSource::NativeLds) {
    return rocjitsu::consan_moi_supports_native_lds_mnemonic(candidate.mnemonic, arch);
  }

  if (candidate.source != rocjitsu::ConSanMoiCandidateSource::FlatGroup &&
      candidate.source != rocjitsu::ConSanMoiCandidateSource::FlatMaybeGroup)
    return false;
  if (candidate.size != 2u * sizeof(uint32_t) && candidate.size != 3u * sizeof(uint32_t))
    return false;
  if (!candidate.raw_ioffset || *candidate.raw_ioffset != 0)
    return false;
  if (*candidate.addr_vgpr >= 255)
    return false;
  return rocjitsu::consan_moi_supports_flat_access_mnemonic(candidate.mnemonic);
}

[[nodiscard]] bool require_moi_patch_applies_to(const rocjitsu::ConSanResult &result) {
  if (std::ranges::any_of(result.site_dispositions,
                          [](const rocjitsu::ConSanSiteDispositionRecord &site) {
                            return site.disposition == rocjitsu::ConSanSiteDisposition::Supported;
                          }))
    return true;
  if (!result.resource_plans.empty())
    return true;
  return std::ranges::any_of(result.moi_candidates, [&](const auto &candidate) {
    return is_supported_require_patch_moi_candidate(candidate, result.arch);
  });
}

struct ConSanStaticCoverageKind {
  uint64_t discovered = 0;
  uint64_t supported = 0;
  uint64_t selected = 0;
  uint64_t patched = 0;
  uint64_t unsupported = 0;
  uint64_t resource_failed = 0;
  uint64_t placement_or_lowering_failed = 0;
  uint64_t expert_limit_omitted = 0;
};

struct ConSanStaticCoverage {
  ConSanStaticCoverageKind access;
  ConSanStaticCoverageKind barrier;
  ConSanStaticCoverageKind atomic;
  ConSanStaticCoverageKind fence;
  bool complete = false;
  bool expert_limit = false;
};

void mark_consan_static_coverage_uninstrumented(ConSanStaticCoverage &coverage) {
  coverage.access.patched = 0;
  coverage.barrier.patched = 0;
  coverage.atomic.patched = 0;
  coverage.fence.patched = 0;
  coverage.complete = false;
}

[[nodiscard]] bool is_consan_access_instrumentation_patch(rocjitsu::ConSanPatchKind kind) {
  switch (kind) {
  case rocjitsu::ConSanPatchKind::InlineLdsLoadCheckTrap:
  case rocjitsu::ConSanPatchKind::InlineLdsStoreCheckTrap:
  case rocjitsu::ConSanPatchKind::LocalCaveLdsLoadCheckTrap:
  case rocjitsu::ConSanPatchKind::LocalCaveLdsStoreCheckTrap:
  case rocjitsu::ConSanPatchKind::InlineFlatLoadCheckTrap:
  case rocjitsu::ConSanPatchKind::InlineFlatStoreCheckTrap:
  case rocjitsu::ConSanPatchKind::LocalCaveFlatLoadCheckTrap:
  case rocjitsu::ConSanPatchKind::LocalCaveFlatStoreCheckTrap:
  case rocjitsu::ConSanPatchKind::InlineMoiAccessRecordStore:
  case rocjitsu::ConSanPatchKind::TrampolineMoiAccessRecordStore:
  case rocjitsu::ConSanPatchKind::InlineMoiExactShadowStore:
  case rocjitsu::ConSanPatchKind::TrampolineMoiExactShadowStore:
  case rocjitsu::ConSanPatchKind::InlineMoiSampledWatchpointStore:
  case rocjitsu::ConSanPatchKind::TrampolineMoiSampledWatchpointStore:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] std::optional<rocjitsu::ConSanResourceSiteKind>
consan_patch_resource_site_kind(const rocjitsu::ConSanPatchInfo &patch,
                                const rocjitsu::ConSanResult &result) {
  if (is_consan_access_instrumentation_patch(patch.kind))
    return rocjitsu::ConSanResourceSiteKind::Access;
  if (patch.kind == rocjitsu::ConSanPatchKind::TrampolineMoiBarrierRecord ||
      patch.kind == rocjitsu::ConSanPatchKind::TrampolineMoiInlineEpochBarrier ||
      patch.kind == rocjitsu::ConSanPatchKind::InlineMalformedBarrierAbort)
    return rocjitsu::ConSanResourceSiteKind::Barrier;
  if (patch.kind == rocjitsu::ConSanPatchKind::TrampolineMoiInlineAtomicOrdering ||
      patch.kind == rocjitsu::ConSanPatchKind::TrampolineMoiAtomicRecord)
    return rocjitsu::ConSanResourceSiteKind::Atomic;
  if (patch.kind == rocjitsu::ConSanPatchKind::TrampolineMoiFenceRecord)
    return rocjitsu::ConSanResourceSiteKind::Fence;
  if (patch.kind != rocjitsu::ConSanPatchKind::TrampolineMoiSampledSyncMetadata)
    return std::nullopt;
  for (const rocjitsu::ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.text_offset == patch.anchor_offset &&
        (plan.site_kind == rocjitsu::ConSanResourceSiteKind::Barrier ||
         plan.site_kind == rocjitsu::ConSanResourceSiteKind::Atomic))
      return plan.site_kind;
  }
  return std::nullopt;
}

[[nodiscard]] bool has_consan_site_instrumentation_patch(const rocjitsu::ConSanResult &result) {
  return std::ranges::any_of(result.patches, [&](const rocjitsu::ConSanPatchInfo &patch) {
    return patch.phase == rocjitsu::ConSanPatchPhase::Instrumentation &&
           consan_patch_resource_site_kind(patch, result).has_value();
  });
}

[[nodiscard]] ConSanStaticCoverageKind &
consan_coverage_kind(ConSanStaticCoverage &coverage, rocjitsu::ConSanResourceSiteKind kind) {
  switch (kind) {
  case rocjitsu::ConSanResourceSiteKind::Access:
    return coverage.access;
  case rocjitsu::ConSanResourceSiteKind::Barrier:
    return coverage.barrier;
  case rocjitsu::ConSanResourceSiteKind::Atomic:
    return coverage.atomic;
  case rocjitsu::ConSanResourceSiteKind::Fence:
    return coverage.fence;
  }
  return coverage.access;
}

void finalize_consan_coverage_kind(ConSanStaticCoverageKind &kind, const HookConfig &config) {
  kind.unsupported = kind.discovered > kind.supported ? kind.discovered - kind.supported : 0;
  kind.selected = kind.supported;
  if (config.max_patches_explicit && kind.selected > config.max_patches) {
    kind.expert_limit_omitted = kind.selected - config.max_patches;
    kind.selected = config.max_patches;
  }
  const uint64_t accounted = kind.patched + kind.resource_failed + kind.expert_limit_omitted;
  kind.placement_or_lowering_failed = kind.supported > accounted ? kind.supported - accounted : 0;
}

[[nodiscard]] ConSanStaticCoverage
compute_consan_static_coverage(const rocjitsu::ConSanResult &result, const HookConfig &config) {
  ConSanStaticCoverage coverage;
  coverage.expert_limit = config.max_patches_explicit;
  bool has_durable_moi_lowering_outcomes = false;
  if (result.flavor == rocjitsu::ConSanFlavor::Moi) {
    if (!result.site_dispositions.empty()) {
      has_durable_moi_lowering_outcomes = std::ranges::none_of(
          result.site_dispositions, [](const rocjitsu::ConSanSiteDispositionRecord &site) {
            return site.disposition == rocjitsu::ConSanSiteDisposition::Supported &&
                   site.lowering_outcome == rocjitsu::ConSanSiteLoweringOutcome::Pending;
          });
      for (const rocjitsu::ConSanSiteDispositionRecord &site : result.site_dispositions) {
        if (site.disposition == rocjitsu::ConSanSiteDisposition::NotApplicable)
          continue;
        ConSanStaticCoverageKind &kind = consan_coverage_kind(coverage, site.site_kind);
        ++kind.discovered;
        if (site.disposition == rocjitsu::ConSanSiteDisposition::Supported) {
          ++kind.supported;
          if (has_durable_moi_lowering_outcomes) {
            switch (site.lowering_outcome) {
            case rocjitsu::ConSanSiteLoweringOutcome::Patched:
              ++kind.patched;
              break;
            case rocjitsu::ConSanSiteLoweringOutcome::ResourceFailed:
              ++kind.resource_failed;
              break;
            case rocjitsu::ConSanSiteLoweringOutcome::PlacementOrLoweringFailed:
              ++kind.placement_or_lowering_failed;
              break;
            default:
              break;
            }
          }
        }
      }
    } else {
      coverage.access.discovered = result.moi_candidates.size();
      coverage.access.supported = static_cast<uint64_t>(std::count_if(
          result.moi_candidates.begin(), result.moi_candidates.end(), [&](const auto &candidate) {
            return is_supported_require_patch_moi_candidate(candidate, result.arch);
          }));
    }
    const auto has_disposition_kind = [&](rocjitsu::ConSanResourceSiteKind wanted) {
      return std::ranges::any_of(result.site_dispositions,
                                 [&](const rocjitsu::ConSanSiteDispositionRecord &site) {
                                   return site.site_kind == wanted;
                                 });
    };
    for (const rocjitsu::ConSanCandidateResourcePlan &plan : result.resource_plans) {
      ConSanStaticCoverageKind &kind = consan_coverage_kind(coverage, plan.site_kind);
      // A semantic disposition ledger is authoritative even when every site
      // of a kind is NotApplicable. Resource plans are created before final
      // semantic pruning and must not resurrect those sites in the coverage
      // denominator.
      if (!has_disposition_kind(plan.site_kind)) {
        ++kind.discovered;
        ++kind.supported;
      }
      if (!has_durable_moi_lowering_outcomes &&
          plan.source == rocjitsu::ConSanRegisterAllocationSource::Unsupported)
        ++kind.resource_failed;
    }
    // Access candidates and access resource plans describe the same sites.
    // Keep the semantic candidate inventory as the authoritative access count.
    if (result.site_dispositions.empty()) {
      coverage.access.discovered = result.moi_candidates.size();
      coverage.access.supported = static_cast<uint64_t>(std::count_if(
          result.moi_candidates.begin(), result.moi_candidates.end(), [&](const auto &candidate) {
            return is_supported_require_patch_moi_candidate(candidate, result.arch);
          }));
    }
    const bool has_inline_exact_patch =
        std::ranges::any_of(result.patches, [](const rocjitsu::ConSanPatchInfo &patch) {
          return patch.phase == rocjitsu::ConSanPatchPhase::Instrumentation &&
                 (patch.kind == rocjitsu::ConSanPatchKind::InlineMoiExactShadowStore ||
                  patch.kind == rocjitsu::ConSanPatchKind::TrampolineMoiExactShadowStore);
        });
    const bool has_sampled_access_patch =
        std::ranges::any_of(result.patches, [](const rocjitsu::ConSanPatchInfo &patch) {
          return patch.phase == rocjitsu::ConSanPatchPhase::Instrumentation &&
                 (patch.kind == rocjitsu::ConSanPatchKind::InlineMoiSampledWatchpointStore ||
                  patch.kind == rocjitsu::ConSanPatchKind::TrampolineMoiSampledWatchpointStore);
        });
    if (!has_durable_moi_lowering_outcomes &&
        config.moi_engine == rocjitsu::ConSanMoiEngine::Sampled && config.moi_track_barriers &&
        has_sampled_access_patch) {
      // Sampled barrier metadata advances only causal windows selected before
      // the sequence. Earlier raw barriers cannot order a supported sampled
      // LDS access and are outside this engine's applicable synchronization
      // denominator. The transform publishes that denominator before resource
      // selection so any later lowering failure still fails closed.
      coverage.barrier = {};
      coverage.barrier.discovered = result.sampled_barrier_applicable_event_count;
      coverage.barrier.supported = result.sampled_barrier_applicable_event_count;
    }
    if (!has_durable_moi_lowering_outcomes &&
        config.moi_engine == rocjitsu::ConSanMoiEngine::InlineShadow && config.moi_track_barriers &&
        has_inline_exact_patch) {
      // Inline epoch probes consume the semantic barrier inventory directly;
      // unlike record-producing engines they do not need a scratch-resource
      // plan per site. Make that inventory authoritative so successfully
      // emitted epoch bodies cannot be reported as 28 patches for 0 sites.
      coverage.barrier = {};
      const auto count_barriers = [&](const auto &container) {
        coverage.barrier.discovered += container.barrier_sites.size();
        coverage.barrier.supported += static_cast<uint64_t>(std::count_if(
            container.barrier_sites.begin(), container.barrier_sites.end(),
            [](const rocjitsu::ConSanBarrierSite &site) { return site.size == sizeof(uint32_t); }));
      };
      for (const auto &kernel : result.kernels)
        count_barriers(kernel);
      if (!result.moi_persistent_vgprs_automatic && !result.moi_private_epoch_automatic) {
        for (const auto &function : result.functions)
          count_barriers(function);
      }
    }
  } else if (result.flavor == rocjitsu::ConSanFlavor::SuperCollider) {
    const ConSanSuperColliderAccessCoverage access =
        compute_consan_supercollider_access_coverage(result, config);
    coverage.access.discovered = access.discovered;
    coverage.access.supported = access.supported;
  }
  if (!has_durable_moi_lowering_outcomes) {
    for (const rocjitsu::ConSanPatchInfo &patch : result.patches) {
      if (patch.phase != rocjitsu::ConSanPatchPhase::Instrumentation)
        continue;
      const auto kind = consan_patch_resource_site_kind(patch, result);
      if (kind) {
        uint64_t covered_sites = 1;
        if (patch.kind == rocjitsu::ConSanPatchKind::TrampolineMoiSampledSyncMetadata &&
            *kind == rocjitsu::ConSanResourceSiteKind::Barrier &&
            patch.covered_sync_event_count != 0) {
          covered_sites = patch.covered_sync_event_count;
        }
        consan_coverage_kind(coverage, *kind).patched += covered_sites;
      }
    }
  }
  finalize_consan_coverage_kind(coverage.access, config);
  finalize_consan_coverage_kind(coverage.barrier, config);
  finalize_consan_coverage_kind(coverage.atomic, config);
  finalize_consan_coverage_kind(coverage.fence, config);
  const auto complete_kind = [](const ConSanStaticCoverageKind &kind) {
    return kind.unsupported == 0 && kind.supported == kind.patched && kind.resource_failed == 0 &&
           kind.placement_or_lowering_failed == 0 && kind.expert_limit_omitted == 0;
  };
  coverage.complete = complete_kind(coverage.access) && complete_kind(coverage.barrier) &&
                      complete_kind(coverage.atomic) && complete_kind(coverage.fence);
  return coverage;
}

class ConSanStaticCoverageRegistry {
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

  static ConSanStaticCoverageRegistry &instance() {
    static ConSanStaticCoverageRegistry registry;
    return registry;
  }

  void record(const ConSanStaticCoverage &coverage) {
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

[[nodiscard]] bool moi_inventory_needs_report_buffer(const rocjitsu::ConSanResult &result,
                                                     const HookConfig &config) {
  if (!result.moi_candidates.empty())
    return true;
  const auto container_needs_buffer = [&](const auto &container) {
    return (config.moi_track_barriers && !container.barrier_sites.empty()) ||
           (config.moi_track_atomics && !container.atomic_sites.empty());
  };
  return std::ranges::any_of(result.kernels, container_needs_buffer) ||
         std::ranges::any_of(result.functions, container_needs_buffer);
}

[[nodiscard]] bool sc_inventory_needs_report_buffer(const rocjitsu::ConSanResult &result) {
  return std::ranges::any_of(result.patches, [](const rocjitsu::ConSanPatchInfo &patch) {
    switch (patch.kind) {
    case rocjitsu::ConSanPatchKind::InlineLdsLoadCheckTrap:
    case rocjitsu::ConSanPatchKind::InlineLdsStoreCheckTrap:
    case rocjitsu::ConSanPatchKind::LocalCaveLdsLoadCheckTrap:
    case rocjitsu::ConSanPatchKind::LocalCaveLdsStoreCheckTrap:
    case rocjitsu::ConSanPatchKind::InlineFlatLoadCheckTrap:
    case rocjitsu::ConSanPatchKind::InlineFlatStoreCheckTrap:
    case rocjitsu::ConSanPatchKind::LocalCaveFlatLoadCheckTrap:
    case rocjitsu::ConSanPatchKind::LocalCaveFlatStoreCheckTrap:
      return patch.phase == rocjitsu::ConSanPatchPhase::Instrumentation;
    default:
      return false;
    }
  });
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
  switch (outcome) {
  case ProcessFaultReservationOutcome::Reserved:
    return "reserved";
  case ProcessFaultReservationOutcome::MutationAlreadyInstalled:
    return "mutation-already-installed";
  case ProcessFaultReservationOutcome::ContentionTimeout:
    return "contention-timeout";
  case ProcessFaultReservationOutcome::ReentrantContention:
    return "reentrant-contention";
  }
  return "unknown";
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

[[nodiscard]] bool fault_mutations_enabled(const rocjitsu::ConSanOptions &options) {
  return rocjitsu::consan_fault_mutations_enabled(options);
}

void disable_fault_mutations(rocjitsu::ConSanOptions *options) {
  rocjitsu::disable_consan_fault_mutations(*options);
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

/// @brief Process-local map from HSA code-object reader handles to ELF bytes.
///
/// @details `hsa_executable_load_agent_code_object()` receives only an opaque
/// reader handle. The create wrapper records memory-backed reader bytes here so
/// the load wrapper can later hand those bytes to the DBI patcher. Session 2
/// only logs and passes through, but this registry is the Session 3 handoff.
class CodeObjectReaderRegistry {
public:
  static CodeObjectReaderRegistry &instance() {
    static CodeObjectReaderRegistry registry;
    return registry;
  }

  struct ReaderBytes {
    const uint8_t *bytes = nullptr;
    size_t size = 0;
    std::shared_ptr<const std::vector<uint8_t>> owned;

    [[nodiscard]] explicit operator bool() const { return bytes != nullptr; }
  };

  [[nodiscard]] bool store(hsa_code_object_reader_t reader, const uint8_t *bytes, size_t size,
                           std::shared_ptr<const std::vector<uint8_t>> owned = {}) {
    std::unique_lock lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      auto *entry = static_cast<Entry *>(it.node_pointer());
      if (entry->handle == reader.handle) {
        entry->bytes = bytes;
        entry->size = size;
        entry->owned = std::move(owned);
        return true;
      }
    }

    void *storage = entry_pool_.try_allocate(sizeof(Entry));
    if (storage == nullptr)
      return false;
    auto *entry = new (storage) Entry(reader.handle, bytes, size, std::move(owned));
    entries_.push_front(*entry);
    return true;
  }

  ReaderBytes lookup(hsa_code_object_reader_t reader) {
    std::shared_lock lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      auto *entry = static_cast<Entry *>(it.node_pointer());
      if (entry->handle == reader.handle) {
        return ReaderBytes{entry->bytes, entry->size, entry->owned};
      }
    }
    return {};
  }

  void remove(hsa_code_object_reader_t reader) {
    std::unique_lock lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end();) {
      auto *entry = static_cast<Entry *>(it.node_pointer());
      if (entry->handle == reader.handle) {
        it = entries_.erase(it);
        destroy_entry(entry);
        return;
      }
      ++it;
    }
  }

  void clear() {
    std::unique_lock lock(mutex_);
    while (!entries_.empty()) {
      auto it = entries_.begin();
      auto *entry = static_cast<Entry *>(it.node_pointer());
      entries_.erase(it);
      destroy_entry(entry);
    }
  }

private:
  struct Entry : util::IListNode<Entry> {
    Entry(uint64_t h, const uint8_t *b, size_t s, std::shared_ptr<const std::vector<uint8_t>> o)
        : handle(h), bytes(b), size(s), owned(std::move(o)) {}

    uint64_t handle = 0;
    const uint8_t *bytes = nullptr;
    size_t size = 0;
    std::shared_ptr<const std::vector<uint8_t>> owned;
  };

  void destroy_entry(Entry *entry) {
    entry->~Entry();
    entry_pool_.deallocate(entry);
  }

  mutable std::shared_mutex mutex_;
  util::ArenaAlloc<sizeof(Entry), 256, alignof(Entry)> entry_pool_;
  util::IntrusiveList<Entry> entries_;
};

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
  std::optional<rocjitsu::ConSanResult> patch_result_storage;
  std::optional<ConSanStaticCoverage> static_coverage_storage;
  std::optional<rocjitsu::ConSanResult> reusable_moi_inventory;
  std::optional<rocjitsu::ConSanMoiAutoReportInventory> live_fault_auto_report_capacity_inventory;
};

/// Owns replacement memory for as long as the executable can expose it.
///
/// HSA loaded-code-object introspection reports the original memory base and
/// size even after the temporary reader used for loading has been destroyed.
/// ROCProfiler consumes that storage when the executable is frozen, which can
/// happen after hsa_executable_load_agent_code_object() returns. Therefore the
/// hook must not tie replacement storage to the load call's ConSanResult. The
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

class AutoScReportBufferRegistry {
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

  static AutoScReportBufferRegistry &instance() {
    static auto *registry = new AutoScReportBufferRegistry;
    return *registry;
  }

  [[nodiscard]] bool allocate(CoreApiTable *core, hsa_agent_t agent, uint64_t reader,
                              uint64_t *address, uint64_t *registered_generation) {
    std::lock_guard lock(mutex_);
    if (entry_count_ >= entries_.size()) {
      ++allocation_failure_count_;
      log_message(kLogDebug,
                  "ConSan SC auto report allocation reader=%llu outcome=failed "
                  "reason=registry_full",
                  static_cast<unsigned long long>(reader));
      return false;
    }
    if (core == nullptr || core->hsa_agent_iterate_regions_fn == nullptr ||
        core->hsa_region_get_info_fn == nullptr || core->hsa_memory_allocate_fn == nullptr ||
        core->hsa_memory_free_fn == nullptr) {
      ++allocation_failure_count_;
      log_message(kLogInfo,
                  "ConSan SC auto report allocation reader=%llu outcome=failed "
                  "reason=hsa_allocation_api_unavailable",
                  static_cast<unsigned long long>(reader));
      return false;
    }

    RegionSearch search{.core = core};
    const hsa_status_t iterate_status =
        core->hsa_agent_iterate_regions_fn(agent, select_region, &search);
    if ((iterate_status != HSA_STATUS_SUCCESS && iterate_status != HSA_STATUS_INFO_BREAK) ||
        !search.found) {
      ++allocation_failure_count_;
      log_message(kLogInfo,
                  "ConSan SC auto report allocation reader=%llu outcome=failed "
                  "reason=no_global_region status=%d",
                  static_cast<unsigned long long>(reader), static_cast<int>(iterate_status));
      return false;
    }

    void *ptr = nullptr;
    const hsa_status_t allocate_status =
        core->hsa_memory_allocate_fn(search.region, sizeof(uint32_t), &ptr);
    if (allocate_status != HSA_STATUS_SUCCESS || ptr == nullptr) {
      ++allocation_failure_count_;
      log_message(kLogInfo,
                  "ConSan SC auto report allocation reader=%llu outcome=failed "
                  "reason=hsa_memory_allocate status=%d",
                  static_cast<unsigned long long>(reader), static_cast<int>(allocate_status));
      return false;
    }
    std::memset(ptr, 0, sizeof(uint32_t));
    if (core->hsa_memory_assign_agent_fn != nullptr) {
      const hsa_status_t assign_status =
          core->hsa_memory_assign_agent_fn(ptr, agent, HSA_ACCESS_PERMISSION_RW);
      if (assign_status != HSA_STATUS_SUCCESS) {
        (void)core->hsa_memory_free_fn(ptr);
        ++allocation_failure_count_;
        log_message(kLogInfo,
                    "ConSan SC auto report allocation reader=%llu outcome=failed "
                    "reason=hsa_memory_assign_agent status=%d",
                    static_cast<unsigned long long>(reader), static_cast<int>(assign_status));
        return false;
      }
    }
    const uint64_t generation = next_generation_.fetch_add(1, std::memory_order_relaxed) + 1u;
    entries_[entry_count_++] = Entry{reader, generation, ptr, search.fine_grained};
    *address = reinterpret_cast<uint64_t>(ptr);
    if (registered_generation != nullptr)
      *registered_generation = generation;
    log_message(kLogInfo,
                "ConSan SC auto report buffer reader=%llu addr=0x%llx bytes=%zu "
                "allocation_outcome=allocated fine_grained=%s",
                static_cast<unsigned long long>(reader), static_cast<unsigned long long>(*address),
                sizeof(uint32_t), search.fine_grained ? "true" : "false");
    return true;
  }

  void bind_to_executable(uint64_t reader, uint64_t generation, hsa_executable_t executable) {
    std::lock_guard lock(mutex_);
    const auto entry =
        std::find_if(entries_.begin(), entries_.begin() + entry_count_,
                     [reader, generation](const Entry &candidate) {
                       return candidate.reader == reader && candidate.generation == generation;
                     });
    if (entry == entries_.begin() + entry_count_)
      return;
    entry->executable = executable.handle;
    entry->executable_bound = true;
  }

  void discard(CoreApiTable *core, uint64_t reader, uint64_t generation) {
    std::lock_guard lock(mutex_);
    const auto entry =
        std::find_if(entries_.begin(), entries_.begin() + entry_count_,
                     [reader, generation](const Entry &candidate) {
                       return candidate.reader == reader && candidate.generation == generation;
                     });
    if (entry == entries_.begin() + entry_count_)
      return;
    const size_t index = static_cast<size_t>(entry - entries_.begin());
    if (release_entry(core, *entry, /*allow_runtime_reclaimed=*/false))
      erase_entry(index);
  }

  void retire(CoreApiTable *core, hsa_executable_t executable) {
    std::lock_guard lock(mutex_);
    size_t index = 0;
    while (index < entry_count_) {
      Entry &entry = entries_[index];
      if (!entry.executable_bound || entry.executable != executable.handle) {
        ++index;
        continue;
      }
      const Summary entry_summary = summarize_entry(core, entry);
      if (!release_entry(core, entry, /*allow_runtime_reclaimed=*/false)) {
        ++index;
        continue;
      }
      accumulate_summary(retired_summary_, entry_summary);
      erase_entry(index);
    }
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
  struct RegionSearch {
    CoreApiTable *core = nullptr;
    hsa_region_t region{};
    bool found = false;
    bool fine_grained = false;
  };

  struct Entry {
    uint64_t reader = 0;
    uint64_t generation = 0;
    void *ptr = nullptr;
    bool fine_grained = false;
    uint64_t executable = 0;
    bool executable_bound = false;
  };

  void erase_entry(size_t index) {
    for (size_t next = index + 1; next < entry_count_; ++next)
      entries_[next - 1] = entries_[next];
    entries_[--entry_count_] = {};
  }

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
      log_message(kLogInfo, "ConSan SC auto report reader=%llu outcome=unreadable mismatch=unknown",
                  static_cast<unsigned long long>(entry.reader));
    } else {
      summary.mismatch_count = marker != 0;
      log_message(
          kLogInfo, "ConSan SC auto report reader=%llu outcome=complete marker=%u mismatch=%s",
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
                  "ConSan SC auto report cleanup reader=%llu "
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
      log_message(kLogInfo, "ConSan SC auto report cleanup reader=%llu outcome=failed status=%d",
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

  static hsa_status_t HSA_API select_region(hsa_region_t region, void *data) {
    auto *search = static_cast<RegionSearch *>(data);
    hsa_region_segment_t segment{};
    hsa_status_t status =
        search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_SEGMENT, &segment);
    if (status != HSA_STATUS_SUCCESS || segment != HSA_REGION_SEGMENT_GLOBAL)
      return status;
    bool alloc_allowed = false;
    status = search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
                                                  &alloc_allowed);
    if (status != HSA_STATUS_SUCCESS || !alloc_allowed)
      return status;
    size_t max_size = 0;
    status =
        search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_ALLOC_MAX_SIZE, &max_size);
    if (status != HSA_STATUS_SUCCESS || max_size < sizeof(uint32_t))
      return status;
    uint32_t flags = 0;
    status = search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    const bool fine_grained = (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0;
    const bool coarse_grained = (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) != 0;
    if (fine_grained) {
      search->region = region;
      search->found = true;
      search->fine_grained = true;
      return HSA_STATUS_INFO_BREAK;
    }
    if (!search->found && coarse_grained) {
      search->region = region;
      search->found = true;
    }
    return HSA_STATUS_SUCCESS;
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
    if (sc_)
      AutoScReportBufferRegistry::instance().discard(core_, sc_->reader, sc_->generation);
    if (moi_)
      discard_auto_moi_report_buffer(core_, moi_->reader, moi_->generation);
  }

  void note_sc(uint64_t reader, uint64_t generation) { sc_ = Registration{reader, generation}; }
  void note_moi(uint64_t reader, uint64_t generation) { moi_ = Registration{reader, generation}; }

  void bind_to_executable(hsa_executable_t executable) {
    if (sc_) {
      AutoScReportBufferRegistry::instance().bind_to_executable(sc_->reader, sc_->generation,
                                                                executable);
      sc_.reset();
    }
    if (moi_) {
      bind_auto_moi_report_buffer_to_executable(moi_->reader, moi_->generation, executable);
      moi_.reset();
    }
  }

private:
  struct Registration {
    uint64_t reader = 0;
    uint64_t generation = 0;
  };

  CoreApiTable *core_ = nullptr;
  std::optional<Registration> sc_;
  std::optional<Registration> moi_;
};

class KernelPrivateDispatchRegistry {
public:
  struct DispatchRequirements {
    uint32_t required_private_bytes = 0;
    uint32_t dynamic_private_addend = 0;
    uint32_t required_group_bytes = 0;

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
      allowlist_.push_back({std::string(normalize_kernel_name(kernel_name))});
  }

  void note_code_object(const rocjitsu::ConSanResult &result) {
    std::lock_guard lock(mutex_);
    for (AllowlistEntry &entry : allowlist_) {
      entry.loaded |= std::ranges::any_of(result.kernels, [&](const auto &kernel) {
        return normalize_kernel_name(kernel.name) == normalize_kernel_name(entry.kernel_name);
      });
    }
  }

  void note_patch_requirements(hsa_executable_t executable, const rocjitsu::ConSanResult &result) {
    std::lock_guard lock(mutex_);
    for (const rocjitsu::ConSanPatchInfo &patch : result.patches) {
      const bool has_segment_requirement = patch.required_private_segment_size != 0 ||
                                           patch.dynamic_private_segment_addend != 0 ||
                                           patch.required_group_segment_size != 0;
      const bool records_moi_site = patch.phase == rocjitsu::ConSanPatchPhase::Instrumentation &&
                                    consan_patch_resource_site_kind(patch, result).has_value();
      if (!has_segment_requirement && !records_moi_site) {
        continue;
      }
      const auto note_kernel = [&](const auto &kernel) {
        if (records_moi_site) {
          const auto allowlisted =
              std::ranges::find_if(allowlist_, [&](const AllowlistEntry &entry) {
                return normalize_kernel_name(entry.kernel_name) ==
                       normalize_kernel_name(kernel.name);
              });
          if (allowlisted != allowlist_.end())
            allowlisted->instrumented = true;
        }
        const auto pending = std::ranges::find_if(pending_, [&](const Pending &candidate) {
          return candidate.executable == executable.handle && candidate.kernel_name == kernel.name;
        });
        if (pending == pending_.end()) {
          pending_.push_back({executable.handle, kernel.name, patch.required_private_segment_size,
                              patch.dynamic_private_segment_addend,
                              patch.required_group_segment_size, records_moi_site});
        } else {
          pending->required_private_bytes =
              std::max(pending->required_private_bytes, patch.required_private_segment_size);
          pending->dynamic_private_addend =
              std::max(pending->dynamic_private_addend, patch.dynamic_private_segment_addend);
          pending->required_group_bytes =
              std::max(pending->required_group_bytes, patch.required_group_segment_size);
          pending->records_moi_site |= records_moi_site;
        }
      };

      if (!patch.owner_descriptor_file_offsets.empty()) {
        for (uint64_t descriptor_offset : patch.owner_descriptor_file_offsets) {
          const auto kernel = std::ranges::find_if(result.kernels, [&](const auto &candidate) {
            return candidate.descriptor_file_offset == descriptor_offset;
          });
          if (kernel != result.kernels.end())
            note_kernel(*kernel);
        }
        continue;
      }

      // Legacy single-owner patches predate explicit owner lists. Preserve the
      // anchor-range fallback for those kernel-local sites.
      const auto kernel = std::ranges::find_if(result.kernels, [&](const auto &candidate) {
        return candidate.has_text_range && patch.anchor_offset >= candidate.entry_text_offset &&
               patch.anchor_offset - candidate.entry_text_offset < candidate.code_size;
      });
      if (kernel != result.kernels.end())
        note_kernel(*kernel);
    }
  }

  void bind_symbol(hsa_executable_t executable, std::string_view symbol_name,
                   hsa_executable_symbol_t symbol,
                   decltype(hsa_executable_symbol_get_info) *original_get_info) {
    if (original_get_info == nullptr)
      return;
    std::lock_guard lock(mutex_);
    const std::string_view normalized = normalize_kernel_name(symbol_name);
    const auto pending = std::ranges::find_if(pending_, [&](const Pending &candidate) {
      return candidate.executable == executable.handle &&
             normalize_kernel_name(candidate.kernel_name) == normalized;
    });
    const auto allowlisted = std::ranges::find_if(allowlist_, [&](const AllowlistEntry &entry) {
      return normalize_kernel_name(entry.kernel_name) == normalized;
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
        .records_moi_site = pending != pending_.end() && pending->records_moi_site,
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
        bound->records_moi_site |= pending->records_moi_site;
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
          allowlisted->dispatch_count != std::numeric_limits<uint64_t>::max())
        ++allowlisted->dispatch_count;
    }
    if (bound->records_moi_site &&
        instrumented_dispatch_count_ != std::numeric_limits<uint64_t>::max())
      ++instrumented_dispatch_count_;
    return {
        .required_private_bytes = bound->required_private_bytes,
        .dynamic_private_addend = bound->dynamic_private_addend,
        .required_group_bytes = bound->required_group_bytes,
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
    bool records_moi_site = false;
  };
  struct Bound {
    uint64_t executable = 0;
    uint64_t symbol = 0;
    uint64_t kernel_object = 0;
    uint32_t required_private_bytes = 0;
    uint32_t dynamic_private_addend = 0;
    uint32_t required_group_bytes = 0;
    bool records_moi_site = false;
    std::string allowlisted_kernel_name;
  };

  [[nodiscard]] static std::string_view normalize_kernel_name(std::string_view name) {
    if (name.ends_with(".kd"))
      name.remove_suffix(3);
    return name;
  }

  mutable std::mutex mutex_;
  std::vector<Pending> pending_;
  std::vector<Bound> bound_;
  std::vector<AllowlistEntry> allowlist_;
  uint64_t dispatch_packet_count_ = 0;
  uint64_t instrumented_dispatch_count_ = 0;
};

hsa_status_t HSA_API rj_dbi_code_object_reader_create_from_memory(
    const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader);
hsa_status_t HSA_API rj_dbi_code_object_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t *code_object_reader);
hsa_status_t HSA_API rj_dbi_loader_code_object_reader_create_from_file_with_offset_size(
    hsa_file_t file, size_t offset, size_t size, hsa_code_object_reader_t *code_object_reader);
hsa_status_t HSA_API rj_dbi_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader);
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
    moi_require_records_ = config.moi_require_records;
    moi_require_diagnostics_ = config.moi_require_diagnostics;
    moi_forbid_diagnostics_ = config.moi_forbid_diagnostics;
    moi_require_replay_conflict_ = config.moi_require_replay_conflict;
    moi_forbid_overflow_ = config.moi_forbid_overflow;
    moi_runtime_sample_stride_ = config.moi_runtime_sample_stride;
    moi_runtime_sample_offset_ = config.moi_runtime_sample_offset;
    fault_load_selector_.reset();
    if (config.fault_load_occurrence)
      fault_load_selector_.emplace(*config.fault_load_occurrence);
    original_create_from_file_ = core_->hsa_code_object_reader_create_from_file_fn;
    original_create_from_memory_ = core_->hsa_code_object_reader_create_from_memory_fn;
    original_destroy_ = core_->hsa_code_object_reader_destroy_fn;
    original_get_extension_table_ = core_->hsa_system_get_extension_table_fn;
    original_get_major_extension_table_ = core_->hsa_system_get_major_extension_table_fn;
    original_load_agent_code_object_ = core_->hsa_executable_load_agent_code_object_fn;
    original_executable_destroy_ = core_->hsa_executable_destroy_fn;
    original_get_symbol_ = core_->hsa_executable_get_symbol_fn;
    original_get_symbol_by_name_ = core_->hsa_executable_get_symbol_by_name_fn;
    original_iterate_symbols_ = core_->hsa_executable_iterate_symbols_fn;
    original_iterate_agent_symbols_ = core_->hsa_executable_iterate_agent_symbols_fn;
    original_symbol_get_info_ = core_->hsa_executable_symbol_get_info_fn;
    original_queue_create_ = core_->hsa_queue_create_fn;
    const bool amd_queue_create_table_valid =
        amd_ext_ != nullptr &&
        amd_ext_->version.minor_id >= offsetof(AmdExtTable, hsa_amd_queue_create_fn) +
                                          sizeof(AmdExtTable::hsa_amd_queue_create_fn);
    original_amd_queue_create_ =
        amd_queue_create_table_valid ? amd_ext_->hsa_amd_queue_create_fn : nullptr;
    intercept_dispatch_segments_ =
        config.flavor.value_or(rocjitsu::ConSanFlavor::None) != rocjitsu::ConSanFlavor::None;
    const bool require_dispatch_packets =
        config.flavor.value_or(rocjitsu::ConSanFlavor::None) == rocjitsu::ConSanFlavor::Moi;
    const bool amd_intercept_table_valid =
        amd_ext_ != nullptr &&
        amd_ext_->version.minor_id >= offsetof(AmdExtTable, hsa_amd_queue_intercept_register_fn) +
                                          sizeof(AmdExtTable::hsa_amd_queue_intercept_register_fn);
    intercept_dispatch_packets_ = intercept_dispatch_segments_ &&
                                  original_queue_create_ != nullptr && amd_intercept_table_valid &&
                                  amd_ext_->hsa_amd_queue_intercept_create_fn != nullptr &&
                                  amd_ext_->hsa_amd_queue_intercept_register_fn != nullptr;

    if (original_create_from_file_ == nullptr || original_create_from_memory_ == nullptr ||
        original_destroy_ == nullptr || original_get_extension_table_ == nullptr ||
        original_get_major_extension_table_ == nullptr ||
        original_load_agent_code_object_ == nullptr || original_executable_destroy_ == nullptr ||
        (intercept_dispatch_segments_ &&
         (original_get_symbol_by_name_ == nullptr || original_iterate_agent_symbols_ == nullptr ||
          original_symbol_get_info_ == nullptr)) ||
        (require_dispatch_packets && !intercept_dispatch_packets_)) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] HSA API table lacks a required DBI entry\n");
      clear_unlocked();
      return false;
    }

    core_->hsa_code_object_reader_create_from_file_fn = rj_dbi_code_object_reader_create_from_file;
    core_->hsa_code_object_reader_create_from_memory_fn =
        rj_dbi_code_object_reader_create_from_memory;
    core_->hsa_code_object_reader_destroy_fn = rj_dbi_code_object_reader_destroy;
    core_->hsa_system_get_extension_table_fn = rj_dbi_system_get_extension_table;
    core_->hsa_system_get_major_extension_table_fn = rj_dbi_system_get_major_extension_table;
    core_->hsa_executable_load_agent_code_object_fn = rj_dbi_executable_load_agent_code_object;
    core_->hsa_executable_destroy_fn = rj_dbi_executable_destroy;
    if (intercept_dispatch_segments_) {
      if (original_get_symbol_ != nullptr)
        core_->hsa_executable_get_symbol_fn = rj_dbi_executable_get_symbol;
      core_->hsa_executable_get_symbol_by_name_fn = rj_dbi_executable_get_symbol_by_name;
      if (original_iterate_symbols_ != nullptr)
        core_->hsa_executable_iterate_symbols_fn = rj_dbi_executable_iterate_symbols;
      core_->hsa_executable_iterate_agent_symbols_fn = rj_dbi_executable_iterate_agent_symbols;
      core_->hsa_executable_symbol_get_info_fn = rj_dbi_executable_symbol_get_info;
    }
    if (intercept_dispatch_packets_) {
      core_->hsa_queue_create_fn = rj_dbi_queue_create;
      if (original_amd_queue_create_ != nullptr)
        amd_ext_->hsa_amd_queue_create_fn = rj_dbi_amd_queue_create;
    }
    KernelPrivateDispatchRegistry::instance().configure_allowlist(config.kernel_name_allowlist);
    active_ = true;
    ConSanStaticCoverageRegistry::instance().clear();

    log_message(
        kLogInfo,
        "installed ConSan hook flavor=%s moi_engine=%s policy=%s moi_profile=%s delay_nops=%u "
        "fail_closed=%s "
        "require_patch=%s "
        "probe_nop=%s probe_trampoline_nop=%s probe_endpgm=%s probe_lds_endpgm=%s "
        "check_trap_mode=%s sc_report_mode=%s probe_lds_check_trap=%s "
        "probe_flat_check_trap=%s probe_flat_trap=%s "
        "fault_drop_barrier=%s fault_reservation_timeout_ms=%u "
        "moi_init_owner_epoch=%s moi_track_barriers=%s "
        "moi_track_atomics=%s moi_dynamic_access_records=%s moi_require_records=%s "
        "moi_require_diagnostics=%s moi_forbid_diagnostics=%s "
        "moi_require_replay_conflict=%s moi_forbid_overflow=%s "
        "fault_barrier_index=%u "
        "delay_mode=%s delay_var_ssrc=%u "
        "patched_image_growth_limit_kind=%s patched_image_growth_limit_value=%llu "
        "process_concurrent_transform_limit_bytes=%s "
        "process_patched_image_limit_bytes=%s "
        "process_patched_image_growth_limit_bytes=%s "
        "max_patches=%u max_patches_source=%s tmp_vgpr=%s moi_exec_save_sgpr=%s "
        "moi_owner_source=%s flat_provenance=%s moi_owner_sgpr=%s moi_owner_vgpr=%s "
        "moi_epoch_vgpr=%s "
        "moi_runtime_sample_stride=%u moi_runtime_sample_stride_source=%s "
        "moi_report_buffer=%s moi_report_buffer_size=%llu "
        "moi_auto_report_buffer_size=%llu moi_auto_report_buffer_size_source=%s mode=%s",
        flavor_name(config.flavor.value_or(rocjitsu::ConSanFlavor::None)),
        rocjitsu::consan_moi_engine_name(config.moi_engine), hook_policy_name(config.policy),
        config.flavor == rocjitsu::ConSanFlavor::Moi ? kMoiStandardProfile.data() : "none",
        config.delay_nops, config.fail_closed ? "true" : "false",
        config.require_patch ? "true" : "false", config.probe_nop ? "true" : "false",
        config.probe_trampoline_nop ? "true" : "false", config.probe_endpgm ? "true" : "false",
        config.probe_lds_endpgm ? "true" : "false", check_trap_mode_name(config.check_trap_mode),
        sc_report_mode_name(config.sc_report_mode), config.probe_lds_check_trap ? "true" : "false",
        config.probe_flat_check_trap ? "true" : "false", config.probe_flat_trap ? "true" : "false",
        config.fault_drop_barrier ? "true" : "false", config.fault_reservation_timeout_ms,
        config.moi_init_owner_epoch ? "true" : "false",
        config.moi_track_barriers ? "true" : "false", config.moi_track_atomics ? "true" : "false",
        config.moi_dynamic_access_records ? "true" : "false",
        config.moi_require_records ? "true" : "false",
        config.moi_require_diagnostics ? "true" : "false",
        config.moi_forbid_diagnostics ? "true" : "false",
        config.moi_require_replay_conflict ? "true" : "false",
        config.moi_forbid_overflow ? "true" : "false", config.fault_barrier_index,
        delay_mode_name(config.delay_mode), config.delay_var_ssrc,
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
        config.moi_exec_save_sgpr ? std::to_string(*config.moi_exec_save_sgpr).c_str() : "unset",
        owner_source_name(config.moi_owner_source),
        flat_provenance_mode_name(config.flat_provenance_mode),
        config.moi_owner_sgpr ? std::to_string(*config.moi_owner_sgpr).c_str() : "unset",
        config.moi_owner_vgpr ? std::to_string(*config.moi_owner_vgpr).c_str() : "unset",
        config.moi_epoch_vgpr ? std::to_string(*config.moi_epoch_vgpr).c_str() : "unset",
        config.moi_runtime_sample_stride,
        config.moi_runtime_sample_stride_explicit ? "expert-override" : "standard-profile",
        config.moi_report_buffer_address ? std::to_string(*config.moi_report_buffer_address).c_str()
                                         : "disabled",
        static_cast<unsigned long long>(config.moi_report_buffer_size),
        static_cast<unsigned long long>(config.moi_auto_report_buffer_size),
        config.moi_auto_report_buffer_size_explicit ? "explicit_cap" : "inventory_ceiling",
        config.fault_drop_barrier
            ? (config.probe_lds_check_trap && config.probe_flat_check_trap
                   ? "proof-check-trap-all+fault-drop-barrier"
               : config.probe_lds_check_trap  ? "proof-lds-check-trap+fault-drop-barrier"
               : config.probe_flat_check_trap ? "proof-flat-check-trap+fault-drop-barrier"
                                              : "fault-drop-barrier")
        : config.probe_lds_check_trap && config.probe_flat_check_trap ? "proof-check-trap-all"
        : config.probe_lds_check_trap                                 ? "proof-lds-check-trap"
        : config.probe_flat_check_trap
            ? "proof-flat-check-trap"
            : (config.probe_flat_trap
                   ? "proof-flat-trap"
                   : (config.probe_lds_endpgm
                          ? "proof-lds-endpgm"
                          : (config.probe_endpgm
                                 ? "proof-endpgm"
                                 : (config.probe_trampoline_nop
                                        ? "proof-trampoline-nop"
                                        : (config.probe_nop ? "proof-nop" : "pass-through"))))));
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

  void uninstall() {
    std::lock_guard lock(mutex_);
    const bool supercollider_active =
        config_ && config_->flavor == rocjitsu::ConSanFlavor::SuperCollider;
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
      if (core_->hsa_code_object_reader_create_from_file_fn ==
          rj_dbi_code_object_reader_create_from_file)
        core_->hsa_code_object_reader_create_from_file_fn = original_create_from_file_;
      if (core_->hsa_code_object_reader_create_from_memory_fn ==
          rj_dbi_code_object_reader_create_from_memory)
        core_->hsa_code_object_reader_create_from_memory_fn = original_create_from_memory_;
      if (core_->hsa_code_object_reader_destroy_fn == rj_dbi_code_object_reader_destroy)
        core_->hsa_code_object_reader_destroy_fn = original_destroy_;
      if (core_->hsa_system_get_extension_table_fn == rj_dbi_system_get_extension_table)
        core_->hsa_system_get_extension_table_fn = original_get_extension_table_;
      if (core_->hsa_system_get_major_extension_table_fn == rj_dbi_system_get_major_extension_table)
        core_->hsa_system_get_major_extension_table_fn = original_get_major_extension_table_;
      if (core_->hsa_executable_load_agent_code_object_fn ==
          rj_dbi_executable_load_agent_code_object)
        core_->hsa_executable_load_agent_code_object_fn = original_load_agent_code_object_;
      if (core_->hsa_executable_destroy_fn == rj_dbi_executable_destroy)
        core_->hsa_executable_destroy_fn = original_executable_destroy_;
      if (core_->hsa_executable_get_symbol_fn == rj_dbi_executable_get_symbol)
        core_->hsa_executable_get_symbol_fn = original_get_symbol_;
      if (core_->hsa_executable_get_symbol_by_name_fn == rj_dbi_executable_get_symbol_by_name)
        core_->hsa_executable_get_symbol_by_name_fn = original_get_symbol_by_name_;
      if (core_->hsa_executable_iterate_symbols_fn == rj_dbi_executable_iterate_symbols)
        core_->hsa_executable_iterate_symbols_fn = original_iterate_symbols_;
      if (core_->hsa_executable_iterate_agent_symbols_fn == rj_dbi_executable_iterate_agent_symbols)
        core_->hsa_executable_iterate_agent_symbols_fn = original_iterate_agent_symbols_;
      if (core_->hsa_executable_symbol_get_info_fn == rj_dbi_executable_symbol_get_info)
        core_->hsa_executable_symbol_get_info_fn = original_symbol_get_info_;
      if (core_->hsa_queue_create_fn == rj_dbi_queue_create)
        core_->hsa_queue_create_fn = original_queue_create_;
    }
    if (active_ && amd_ext_ != nullptr && original_amd_queue_create_ != nullptr &&
        amd_ext_->hsa_amd_queue_create_fn == rj_dbi_amd_queue_create) {
      amd_ext_->hsa_amd_queue_create_fn = original_amd_queue_create_;
    }

    const AutoScReportBufferRegistry::Summary sc_report_summary =
        AutoScReportBufferRegistry::instance().summarize_and_clear(core_);
    const AutoMoiReportSummary moi_report_summary =
        summarize_and_clear_auto_moi_report_buffers(core_);
    const ConSanStaticCoverageRegistry::Summary static_coverage_summary =
        ConSanStaticCoverageRegistry::instance().summarize_and_clear();
    const ProcessByteBudget::Summary transform_admission_summary =
        ProcessTransformAdmissionRegistry::instance().summarize_and_rollover();
    const ReplacementCodeObjectStorageRegistry::Summary patched_image_summary =
        ReplacementCodeObjectStorageRegistry::instance().summarize_and_rollover();
    const bool moi_require_records = moi_require_records_;
    const bool moi_require_diagnostics = moi_require_diagnostics_;
    const bool moi_forbid_diagnostics = moi_forbid_diagnostics_;
    const bool moi_require_replay_conflict = moi_require_replay_conflict_;
    const bool moi_forbid_overflow = moi_forbid_overflow_;
    const uint32_t moi_runtime_sample_stride = moi_runtime_sample_stride_;
    const uint32_t moi_runtime_sample_offset = moi_runtime_sample_offset_;
    const KernelPrivateDispatchRegistry::DispatchSummary dispatch_summary =
        KernelPrivateDispatchRegistry::instance().dispatch_summary();
    const std::vector<KernelPrivateDispatchRegistry::AllowlistEntrySummary> allowlist_summary =
        KernelPrivateDispatchRegistry::instance().allowlist_summary();
    const rocjitsu::ConSanMoiEngine moi_engine =
        config_ ? config_->moi_engine : rocjitsu::ConSanMoiEngine::RecordReplay;
    const bool fault_require_exactly_one = config_ && config_->fault_require_exactly_one;
    const std::optional<ProcessFaultApplicationSnapshot> fault_application_snapshot =
        take_process_fault_application_snapshot();
    const std::optional<rocjitsu::ConSanFaultLoadSelector> fault_load_selector =
        fault_load_selector_;
    CodeObjectReaderRegistry::instance().clear();
    KernelPrivateDispatchRegistry::instance().clear();
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
    clear_unlocked();
    if (supercollider_active || sc_report_summary.buffer_count != 0 ||
        !sc_report_summary.complete()) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] ConSan SC report summary buffers=%llu mismatches=%llu "
                   "allocation_failures=%llu read_failures=%llu cleanup_failures=%llu "
                   "complete=%s\n",
                   static_cast<unsigned long long>(sc_report_summary.buffer_count),
                   static_cast<unsigned long long>(sc_report_summary.mismatch_count),
                   static_cast<unsigned long long>(sc_report_summary.allocation_failure_count),
                   static_cast<unsigned long long>(sc_report_summary.read_failure_count),
                   static_cast<unsigned long long>(sc_report_summary.cleanup_failure_count),
                   sc_report_summary.complete() ? "true" : "false");
      std::fflush(stderr);
    }
    const uint64_t visible_evidence_count = moi_report_summary.visible_access_record_count +
                                            moi_report_summary.visible_barrier_record_count +
                                            moi_report_summary.visible_atomic_record_count +
                                            moi_report_summary.visible_fence_record_count +
                                            moi_report_summary.visible_diagnostic_record_count +
                                            moi_report_summary.visible_inline_publication_count +
                                            moi_report_summary.visible_exact_shadow_entry_count +
                                            moi_report_summary.visible_inline_atomic_release_count +
                                            moi_report_summary.visible_inline_acquired_token_count +
                                            moi_report_summary.visible_sampled_watchpoint_count;
    const bool required_records_missing = moi_require_records && visible_evidence_count == 0;
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
                   static_cast<unsigned long long>(visible_evidence_count), status);
    }
    std::fprintf(
        stderr,
        "[rocjitsu-dbi-hooks] ConSan MOI report memory required_bytes=%llu "
        "allocated_bytes=%llu live_before_cleanup=%llu live_after_cleanup=%llu "
        "peak_live_bytes=%llu per_buffer_ceiling=%llu process_ceiling=%llu "
        "allocation_failures=%llu capacity_failures=%llu cleanup_failures=%llu "
        "fine_grained_snapshot_bytes=%llu coarse_grained_snapshot_bytes=%llu\n",
        static_cast<unsigned long long>(moi_report_summary.required_report_bytes),
        static_cast<unsigned long long>(moi_report_summary.allocated_report_bytes),
        static_cast<unsigned long long>(moi_report_summary.current_live_report_bytes),
        static_cast<unsigned long long>(moi_report_summary.current_live_report_bytes_after_cleanup),
        static_cast<unsigned long long>(moi_report_summary.peak_live_report_bytes),
        static_cast<unsigned long long>(
            rocjitsu::consan_moi_auto_report_buffer_ceiling_bytes(moi_engine)),
        static_cast<unsigned long long>(rocjitsu::kConSanMoiAutoReportProcessCeilingBytes),
        static_cast<unsigned long long>(moi_report_summary.allocation_failure_count),
        static_cast<unsigned long long>(moi_report_summary.capacity_failure_count),
        static_cast<unsigned long long>(moi_report_summary.cleanup_failure_count),
        static_cast<unsigned long long>(moi_report_summary.fine_grained_snapshot_bytes),
        static_cast<unsigned long long>(moi_report_summary.coarse_grained_snapshot_bytes));
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
        sc_report_summary.allocation_failure_count + sc_report_summary.read_failure_count +
        sc_report_summary.cleanup_failure_count + moi_report_summary.allocation_failure_count +
        moi_report_summary.cleanup_failure_count + moi_report_summary.dropped_access_record_count +
        moi_report_summary.dropped_barrier_record_count +
        moi_report_summary.dropped_atomic_record_count +
        moi_report_summary.dropped_fence_record_count +
        moi_report_summary.dropped_diagnostic_record_count +
        moi_report_summary.record_replay_bank_saturation_count +
        moi_report_summary.record_replay_invalid_site_token_count +
        moi_report_summary.sampled_dropped_window_count +
        moi_report_summary.sampled_unusable_snapshot_count() +
        moi_report_summary.exact_unusable_snapshot_count() +
        moi_report_summary.release_unusable_snapshot_count() +
        moi_report_summary.token_unusable_snapshot_count() +
        moi_report_summary.inline_undercoverage_count + moi_report_summary.inline_overflow_count +
        moi_report_summary.inline_unsupported_count + moi_report_summary.inline_malformed_count +
        moi_report_summary.sampled_unsupported_sync_count +
        moi_report_summary.sampled_malformed_sync_count +
        moi_report_summary.replay_dropped_access_count +
        moi_report_summary.replay_dropped_barrier_count +
        moi_report_summary.replay_unsupported_access_count +
        moi_report_summary.replay_unsupported_atomic_count +
        moi_report_summary.replay_unsupported_fence_count +
        moi_report_summary.replay_metadata_full_count +
        moi_report_summary.replay_diagnostic_capacity_exhausted_count;
    const bool static_complete = static_coverage_summary.complete();
    const bool dynamic_complete = dynamic_incomplete_count == 0 && !required_records_missing;
    const bool analysis_complete = static_complete && dynamic_complete;
    std::fprintf(
        stderr,
        "[rocjitsu-dbi-hooks] ConSan analysis verdict applicable=%s analysis_complete=%s "
        "static_complete=%s dynamic_complete=%s applicable_code_objects=%llu "
        "incomplete_code_objects=%llu access=%llu/%llu barrier=%llu/%llu "
        "atomic=%llu/%llu fence=%llu/%llu visible_evidence=%llu dynamic_incomplete=%llu "
        "record_replay_bank_saturation=%llu "
        "record_replay_invalid_site_tokens=%llu "
        "replay_unsupported_access=%llu replay_unsupported_atomics=%llu "
        "replay_unsupported_fences=%llu replay_metadata_full=%llu\n",
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
        static_cast<unsigned long long>(visible_evidence_count),
        static_cast<unsigned long long>(dynamic_incomplete_count),
        static_cast<unsigned long long>(moi_report_summary.record_replay_bank_saturation_count),
        static_cast<unsigned long long>(moi_report_summary.record_replay_invalid_site_token_count),
        static_cast<unsigned long long>(moi_report_summary.replay_unsupported_access_count),
        static_cast<unsigned long long>(moi_report_summary.replay_unsupported_atomic_count),
        static_cast<unsigned long long>(moi_report_summary.replay_unsupported_fence_count),
        static_cast<unsigned long long>(moi_report_summary.replay_metadata_full_count));
    std::fflush(stderr);
    const uint64_t moi_dropped_record_count = moi_report_summary.dropped_access_record_count +
                                              moi_report_summary.dropped_barrier_record_count +
                                              moi_report_summary.dropped_atomic_record_count +
                                              moi_report_summary.dropped_fence_record_count +
                                              moi_report_summary.dropped_diagnostic_record_count +
                                              moi_report_summary.sampled_dropped_window_count;
    if (moi_dropped_record_count != 0) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] ConSan MOI report overflow: dropped access=%llu "
          "barrier=%llu atomic=%llu fence=%llu diagnostic=%llu sampled_windows=%llu "
          "across %llu auto report "
          "buffer(s)\n",
          static_cast<unsigned long long>(moi_report_summary.dropped_access_record_count),
          static_cast<unsigned long long>(moi_report_summary.dropped_barrier_record_count),
          static_cast<unsigned long long>(moi_report_summary.dropped_atomic_record_count),
          static_cast<unsigned long long>(moi_report_summary.dropped_fence_record_count),
          static_cast<unsigned long long>(moi_report_summary.dropped_diagnostic_record_count),
          static_cast<unsigned long long>(moi_report_summary.sampled_dropped_window_count),
          static_cast<unsigned long long>(moi_report_summary.buffer_count));
      std::fflush(stderr);
      if (moi_forbid_overflow)
        std::_Exit(90);
    }
    if (moi_report_summary.record_replay_bank_saturation_count != 0) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] ConSan MOI Record/Replay dispatch-directory/access-table "
          "saturation: "
          "%llu auto report buffer(s) exhausted a bounded capture probe\n",
          static_cast<unsigned long long>(moi_report_summary.record_replay_bank_saturation_count));
      std::fflush(stderr);
      if (moi_forbid_overflow)
        std::_Exit(90);
    }
    if (moi_report_summary.record_replay_invalid_site_token_count != 0) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] ConSan MOI Record/Replay malformed access evidence: "
                   "invalid_site_tokens=%llu across %llu auto report buffer(s)\n",
                   static_cast<unsigned long long>(
                       moi_report_summary.record_replay_invalid_site_token_count),
                   static_cast<unsigned long long>(moi_report_summary.buffer_count));
      std::fflush(stderr);
      if (moi_forbid_overflow)
        std::_Exit(90);
    }
    if (moi_report_summary.sampled_unusable_snapshot_count() != 0) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] ConSan MOI sampled snapshot incomplete: stale=%llu "
          "incomplete=%llu changed=%llu malformed=%llu across %llu auto report buffer(s)\n",
          static_cast<unsigned long long>(moi_report_summary.sampled_stale_snapshot_count),
          static_cast<unsigned long long>(moi_report_summary.sampled_incomplete_snapshot_count),
          static_cast<unsigned long long>(moi_report_summary.sampled_changed_snapshot_count),
          static_cast<unsigned long long>(moi_report_summary.sampled_malformed_snapshot_count),
          static_cast<unsigned long long>(moi_report_summary.buffer_count));
      std::fflush(stderr);
      if (moi_forbid_overflow)
        std::_Exit(90);
    }
    if (moi_report_summary.exact_unusable_snapshot_count() != 0) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] ConSan MOI exact snapshot unusable: incomplete=%llu "
          "changed=%llu malformed=%llu across %llu auto report buffer(s)\n",
          static_cast<unsigned long long>(moi_report_summary.exact_incomplete_snapshot_count),
          static_cast<unsigned long long>(moi_report_summary.exact_changed_snapshot_count),
          static_cast<unsigned long long>(moi_report_summary.exact_malformed_snapshot_count),
          static_cast<unsigned long long>(moi_report_summary.buffer_count));
      std::fflush(stderr);
      if (moi_forbid_overflow)
        std::_Exit(90);
    }
    if (moi_report_summary.release_unusable_snapshot_count() != 0) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] ConSan MOI release snapshot unusable: incomplete=%llu "
          "changed=%llu overflow=%llu source_incomplete=%llu malformed=%llu across %llu auto "
          "report buffer(s)\n",
          static_cast<unsigned long long>(moi_report_summary.release_incomplete_snapshot_count),
          static_cast<unsigned long long>(moi_report_summary.release_changed_snapshot_count),
          static_cast<unsigned long long>(moi_report_summary.release_overflow_snapshot_count),
          static_cast<unsigned long long>(
              moi_report_summary.release_source_incomplete_snapshot_count),
          static_cast<unsigned long long>(moi_report_summary.release_malformed_snapshot_count),
          static_cast<unsigned long long>(moi_report_summary.buffer_count));
      std::fflush(stderr);
      if (moi_forbid_overflow)
        std::_Exit(90);
    }
    if (moi_report_summary.token_unusable_snapshot_count() != 0) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] ConSan MOI acquired token snapshot unusable: "
          "incomplete=%llu changed=%llu malformed=%llu across %llu auto report buffer(s)\n",
          static_cast<unsigned long long>(moi_report_summary.token_incomplete_snapshot_count),
          static_cast<unsigned long long>(moi_report_summary.token_changed_snapshot_count),
          static_cast<unsigned long long>(moi_report_summary.token_malformed_snapshot_count),
          static_cast<unsigned long long>(moi_report_summary.buffer_count));
      std::fflush(stderr);
      if (moi_forbid_overflow)
        std::_Exit(90);
    }
    const uint64_t inline_coverage_loss =
        moi_report_summary.inline_undercoverage_count + moi_report_summary.inline_overflow_count +
        moi_report_summary.inline_unsupported_count + moi_report_summary.inline_malformed_count;
    if (inline_coverage_loss != 0) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] ConSan MOI inline coverage loss: undercoverage=%llu "
          "overflow=%llu unsupported=%llu malformed=%llu across %llu auto report buffer(s)\n",
          static_cast<unsigned long long>(moi_report_summary.inline_undercoverage_count),
          static_cast<unsigned long long>(moi_report_summary.inline_overflow_count),
          static_cast<unsigned long long>(moi_report_summary.inline_unsupported_count),
          static_cast<unsigned long long>(moi_report_summary.inline_malformed_count),
          static_cast<unsigned long long>(moi_report_summary.buffer_count));
      std::fflush(stderr);
      if (moi_forbid_overflow)
        std::_Exit(90);
    }
    if (required_records_missing) {
      if (dispatch_summary.packet_count == 0u) {
        std::fprintf(stderr,
                     "[rocjitsu-dbi-hooks] RJ_CONSAN_MOI_REQUIRE_RECORDS requested, but %llu auto "
                     "MOI report buffer(s) contained zero visible records and no kernel dispatch "
                     "packet was observed\n",
                     static_cast<unsigned long long>(moi_report_summary.buffer_count));
      } else if (dispatch_summary.instrumented_packet_count == 0u) {
        std::fprintf(stderr,
                     "[rocjitsu-dbi-hooks] RJ_CONSAN_MOI_REQUIRE_RECORDS requested, but %llu auto "
                     "MOI report buffer(s) contained zero visible records; none of %llu observed "
                     "kernel dispatch packet(s) was attributed to a bound MOI-instrumented "
                     "kernel object\n",
                     static_cast<unsigned long long>(moi_report_summary.buffer_count),
                     static_cast<unsigned long long>(dispatch_summary.packet_count));
      } else if (moi_runtime_sample_stride > 1u) {
        std::fprintf(stderr,
                     "[rocjitsu-dbi-hooks] RJ_CONSAN_MOI_REQUIRE_RECORDS requested, but %llu auto "
                     "MOI report buffer(s) contained zero visible records after %llu "
                     "instrumented dispatch packet(s); runtime sampling may have selected no "
                     "workgroups, or selected workgroups may not have executed an instrumented "
                     "site (stride=%u offset=%u). Set RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE=1 to "
                     "distinguish sampling gaps from dense-path gaps\n",
                     static_cast<unsigned long long>(moi_report_summary.buffer_count),
                     static_cast<unsigned long long>(dispatch_summary.instrumented_packet_count),
                     moi_runtime_sample_stride, moi_runtime_sample_offset);
      } else {
        std::fprintf(stderr,
                     "[rocjitsu-dbi-hooks] RJ_CONSAN_MOI_REQUIRE_RECORDS requested, but %llu auto "
                     "MOI report buffer(s) contained zero visible records after %llu "
                     "instrumented dispatch packet(s); the dense record path produced no "
                     "evidence\n",
                     static_cast<unsigned long long>(moi_report_summary.buffer_count),
                     static_cast<unsigned long long>(dispatch_summary.instrumented_packet_count));
      }
      std::fflush(stderr);
      std::_Exit(86);
    }
    if (moi_require_replay_conflict && moi_report_summary.replay_conflict_count == 0) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT requested, but "
          "%llu auto MOI report buffer(s) produced zero replay conflicts "
          "(visible access=%llu barrier=%llu atomic=%llu fence=%llu diagnostics=%llu sampled=%llu, "
          "replay diagnostics=%llu)\n",
          static_cast<unsigned long long>(moi_report_summary.buffer_count),
          static_cast<unsigned long long>(moi_report_summary.visible_access_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_barrier_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_atomic_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_fence_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_diagnostic_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_sampled_watchpoint_count),
          static_cast<unsigned long long>(moi_report_summary.replay_diagnostic_count));
      std::fflush(stderr);
      std::_Exit(87);
    }
    if (moi_report_summary.inline_undercoverage_count != 0) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] ConSan MOI inline undercoverage observed: "
                   "%llu lane-site publication(s) were not represented in exact shadow\n",
                   static_cast<unsigned long long>(moi_report_summary.inline_undercoverage_count));
      std::fflush(stderr);
    }
    const bool moi_has_diagnostics = moi_report_summary.visible_diagnostic_record_count +
                                         moi_report_summary.replay_diagnostic_count +
                                         moi_report_summary.sampled_conflict_count +
                                         moi_report_summary.sampled_immediate_conflict_count >
                                     0;
    if (moi_require_diagnostics && !moi_has_diagnostics) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS requested, but "
          "%llu auto MOI report buffer(s) produced zero visible/replay diagnostics or sampled "
          "conflicts "
          "(visible access=%llu barrier=%llu atomic=%llu diagnostics=%llu "
          "exact-shadow=%llu sampled=%llu, replay diagnostics=%llu sampled_conflicts=%llu "
          "sampled_immediate_conflicts=%llu)\n",
          static_cast<unsigned long long>(moi_report_summary.buffer_count),
          static_cast<unsigned long long>(moi_report_summary.visible_access_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_barrier_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_atomic_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_diagnostic_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_exact_shadow_entry_count),
          static_cast<unsigned long long>(moi_report_summary.visible_sampled_watchpoint_count),
          static_cast<unsigned long long>(moi_report_summary.replay_diagnostic_count),
          static_cast<unsigned long long>(moi_report_summary.sampled_conflict_count),
          static_cast<unsigned long long>(moi_report_summary.sampled_immediate_conflict_count));
      std::fflush(stderr);
      std::_Exit(88);
    }
    if (moi_forbid_diagnostics && moi_has_diagnostics) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] RJ_CONSAN_MOI_FORBID_DIAGNOSTICS requested, but "
          "%llu auto MOI report buffer(s) produced visible/replay diagnostics or sampled "
          "conflicts "
          "(visible access=%llu barrier=%llu atomic=%llu diagnostics=%llu "
          "exact-shadow=%llu sampled=%llu, replay diagnostics=%llu sampled_conflicts=%llu "
          "sampled_immediate_conflicts=%llu)\n",
          static_cast<unsigned long long>(moi_report_summary.buffer_count),
          static_cast<unsigned long long>(moi_report_summary.visible_access_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_barrier_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_atomic_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_diagnostic_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_exact_shadow_entry_count),
          static_cast<unsigned long long>(moi_report_summary.visible_sampled_watchpoint_count),
          static_cast<unsigned long long>(moi_report_summary.replay_diagnostic_count),
          static_cast<unsigned long long>(moi_report_summary.sampled_conflict_count),
          static_cast<unsigned long long>(moi_report_summary.sampled_immediate_conflict_count));
      std::fflush(stderr);
      std::_Exit(89);
    }
  }

  [[nodiscard]] std::optional<HookConfig> config() const {
    std::lock_guard lock(mutex_);
    return config_;
  }

  [[nodiscard]] std::optional<rocjitsu::ConSanFaultLoadSelection> observe_fault_load_match() {
    std::lock_guard lock(mutex_);
    if (!fault_load_selector_)
      return std::nullopt;
    return fault_load_selector_->observe();
  }

  [[nodiscard]] decltype(hsa_code_object_reader_create_from_file) *create_from_file() const {
    std::lock_guard lock(mutex_);
    return original_create_from_file_;
  }

  [[nodiscard]] decltype(hsa_code_object_reader_create_from_memory) *create_from_memory() const {
    std::lock_guard lock(mutex_);
    return original_create_from_memory_;
  }

  [[nodiscard]] decltype(hsa_code_object_reader_destroy) *destroy() const {
    std::lock_guard lock(mutex_);
    return original_destroy_;
  }

  [[nodiscard]] decltype(hsa_system_get_extension_table) *get_extension_table() const {
    std::lock_guard lock(mutex_);
    return original_get_extension_table_;
  }

  [[nodiscard]] decltype(hsa_system_get_major_extension_table) *get_major_extension_table() const {
    std::lock_guard lock(mutex_);
    return original_get_major_extension_table_;
  }

  void set_loader_create_from_file_with_offset_size(LoaderCreateFromFileWithOffsetSize function) {
    std::lock_guard lock(mutex_);
    original_loader_create_from_file_with_offset_size_ = function;
  }

  [[nodiscard]] LoaderCreateFromFileWithOffsetSize
  loader_create_from_file_with_offset_size() const {
    std::lock_guard lock(mutex_);
    return original_loader_create_from_file_with_offset_size_;
  }

  [[nodiscard]] decltype(hsa_executable_load_agent_code_object) *load_agent_code_object() const {
    std::lock_guard lock(mutex_);
    return original_load_agent_code_object_;
  }

  [[nodiscard]] decltype(hsa_executable_destroy) *executable_destroy() const {
    std::lock_guard lock(mutex_);
    return original_executable_destroy_;
  }

  [[nodiscard]] decltype(hsa_executable_get_symbol) *get_symbol() const {
    std::lock_guard lock(mutex_);
    return original_get_symbol_;
  }

  [[nodiscard]] decltype(hsa_executable_get_symbol_by_name) *get_symbol_by_name() const {
    std::lock_guard lock(mutex_);
    return original_get_symbol_by_name_;
  }

  [[nodiscard]] decltype(hsa_executable_iterate_symbols) *iterate_symbols() const {
    std::lock_guard lock(mutex_);
    return original_iterate_symbols_;
  }

  [[nodiscard]] decltype(hsa_executable_iterate_agent_symbols) *iterate_agent_symbols() const {
    std::lock_guard lock(mutex_);
    return original_iterate_agent_symbols_;
  }

  [[nodiscard]] decltype(hsa_executable_symbol_get_info) *symbol_get_info() const {
    std::lock_guard lock(mutex_);
    return original_symbol_get_info_;
  }

  [[nodiscard]] decltype(hsa_queue_create) *queue_create() const {
    std::lock_guard lock(mutex_);
    return original_queue_create_;
  }

  [[nodiscard]] hsa_amd_queue_create_fn_t amd_queue_create() const {
    std::lock_guard lock(mutex_);
    return original_amd_queue_create_;
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

  void clear_unlocked() {
    active_ = false;
    g_log_level.store(kLogDisabled, std::memory_order_relaxed);
    table_ = nullptr;
    core_ = nullptr;
    amd_ext_ = nullptr;
    config_.reset();
    moi_require_records_ = false;
    moi_require_diagnostics_ = false;
    moi_forbid_diagnostics_ = false;
    moi_require_replay_conflict_ = false;
    moi_forbid_overflow_ = false;
    moi_runtime_sample_stride_ = 1u;
    moi_runtime_sample_offset_ = 0u;
    fault_load_selector_.reset();
    original_create_from_file_ = nullptr;
    original_create_from_memory_ = nullptr;
    original_destroy_ = nullptr;
    original_get_extension_table_ = nullptr;
    original_get_major_extension_table_ = nullptr;
    original_loader_create_from_file_with_offset_size_ = nullptr;
    original_load_agent_code_object_ = nullptr;
    original_executable_destroy_ = nullptr;
    original_get_symbol_ = nullptr;
    original_get_symbol_by_name_ = nullptr;
    original_iterate_symbols_ = nullptr;
    original_iterate_agent_symbols_ = nullptr;
    original_symbol_get_info_ = nullptr;
    original_queue_create_ = nullptr;
    original_amd_queue_create_ = nullptr;
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
  bool moi_require_records_ = false;
  bool moi_require_diagnostics_ = false;
  bool moi_forbid_diagnostics_ = false;
  bool moi_require_replay_conflict_ = false;
  bool moi_forbid_overflow_ = false;
  uint32_t moi_runtime_sample_stride_ = 1u;
  uint32_t moi_runtime_sample_offset_ = 0u;
  std::optional<rocjitsu::ConSanFaultLoadSelector> fault_load_selector_;
  bool active_ = false;
  decltype(hsa_code_object_reader_create_from_file) *original_create_from_file_ = nullptr;
  decltype(hsa_code_object_reader_create_from_memory) *original_create_from_memory_ = nullptr;
  decltype(hsa_code_object_reader_destroy) *original_destroy_ = nullptr;
  decltype(hsa_system_get_extension_table) *original_get_extension_table_ = nullptr;
  decltype(hsa_system_get_major_extension_table) *original_get_major_extension_table_ = nullptr;
  LoaderCreateFromFileWithOffsetSize original_loader_create_from_file_with_offset_size_ = nullptr;
  decltype(hsa_executable_load_agent_code_object) *original_load_agent_code_object_ = nullptr;
  decltype(hsa_executable_destroy) *original_executable_destroy_ = nullptr;
  decltype(hsa_executable_get_symbol) *original_get_symbol_ = nullptr;
  decltype(hsa_executable_get_symbol_by_name) *original_get_symbol_by_name_ = nullptr;
  decltype(hsa_executable_iterate_symbols) *original_iterate_symbols_ = nullptr;
  decltype(hsa_executable_iterate_agent_symbols) *original_iterate_agent_symbols_ = nullptr;
  decltype(hsa_executable_symbol_get_info) *original_symbol_get_info_ = nullptr;
  decltype(hsa_queue_create) *original_queue_create_ = nullptr;
  hsa_amd_queue_create_fn_t original_amd_queue_create_ = nullptr;
  bool intercept_dispatch_segments_ = false;
  bool intercept_dispatch_packets_ = false;
};

RjDbiHsaLayer &layer() {
  static RjDbiHsaLayer state;
  return state;
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
    if (!CodeObjectReaderRegistry::instance().store(
            *code_object_reader, static_cast<const uint8_t *>(code_object), size)) {
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

std::shared_ptr<const std::vector<uint8_t>>
snapshot_code_object_file_range(hsa_file_t file, size_t offset, size_t size) {
  struct stat file_stat {};
  if (fstat(file, &file_stat) != 0 || file_stat.st_size <= 0 || size == 0 ||
      static_cast<uintmax_t>(file_stat.st_size) > std::numeric_limits<size_t>::max() ||
      offset > static_cast<size_t>(file_stat.st_size) ||
      size > static_cast<size_t>(file_stat.st_size) - offset)
    return {};

  std::shared_ptr<std::vector<uint8_t>> bytes;
  try {
    bytes = std::make_shared<std::vector<uint8_t>>(size);
  } catch (const std::bad_alloc &) {
    return {};
  }

  size_t read_size = 0;
  while (read_size < size) {
    const ssize_t result = pread(file, bytes->data() + read_size, size - read_size,
                                 static_cast<off_t>(offset + read_size));
    if (result > 0) {
      read_size += static_cast<size_t>(result);
      continue;
    }
    if (result < 0 && errno == EINTR)
      continue;
    return {};
  }
  return bytes;
}

std::shared_ptr<const std::vector<uint8_t>> snapshot_code_object_file(hsa_file_t file) {
  struct stat file_stat {};
  if (fstat(file, &file_stat) != 0 || file_stat.st_size <= 0 ||
      static_cast<uintmax_t>(file_stat.st_size) > std::numeric_limits<size_t>::max())
    return {};
  return snapshot_code_object_file_range(file, 0, static_cast<size_t>(file_stat.st_size));
}

hsa_status_t HSA_API rj_dbi_code_object_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t *code_object_reader) {
  auto *original = layer().create_from_file();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original(file, code_object_reader);
  if (status == HSA_STATUS_SUCCESS && code_object_reader != nullptr) {
    const auto bytes = snapshot_code_object_file(file);
    if (!bytes) {
      log_message(kLogInfo, "could not snapshot file-backed reader=%llu",
                  static_cast<unsigned long long>(code_object_reader->handle));
    } else if (!CodeObjectReaderRegistry::instance().store(*code_object_reader, bytes->data(),
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

  const auto bytes = snapshot_code_object_file_range(file, offset, size);
  if (!bytes) {
    log_message(kLogInfo, "could not snapshot ranged file-backed reader=%llu offset=%zu bytes=%zu",
                static_cast<unsigned long long>(code_object_reader->handle), offset, size);
  } else if (!CodeObjectReaderRegistry::instance().store(*code_object_reader, bytes->data(),
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
  CodeObjectReaderRegistry::instance().remove(code_object_reader);

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
    AutoScReportBufferRegistry::instance().retire(layer().core_table(), executable);
    retire_auto_moi_report_buffers(layer().core_table(), executable);
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
  (void)data;
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
    if (type != HSA_PACKET_TYPE_KERNEL_DISPATCH && !is_extended_dispatch)
      continue;
    const KernelPrivateDispatchRegistry::DispatchRequirements requirements =
        KernelPrivateDispatchRegistry::instance().note_and_query_dispatch(packet->kernel_object);
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
      intercept_register(*queue, rj_dbi_queue_write_interceptor, nullptr);
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
  auto &reusable_moi_inventory = transform_state.reusable_moi_inventory;
  auto &live_fault_auto_report_capacity_inventory =
      transform_state.live_fault_auto_report_capacity_inventory;
  hsa_code_object_reader_t reader_to_load = code_object_reader;
  hsa_code_object_reader_t replacement_reader{};
  bool using_replacement_reader = false;
  bool replacement_storage_retained = false;
  bool replacement_instrumentation_selected = false;
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
  rocjitsu::ConSanInstallAction install_action = rocjitsu::ConSanInstallAction::LoadOriginal;
  ProcessFaultApplicationReservation process_fault_application_reservation;
  ProcessFaultReservationOutcome process_fault_reservation_outcome =
      ProcessFaultReservationOutcome::MutationAlreadyInstalled;
  bool process_fault_reservation_attempted = false;
  FaultInstallationEvidence fault_installation_evidence(code_object_reader.handle);
  const auto record_static_coverage = [&](bool replacement_installed) {
    if (!static_coverage_storage)
      return;
    if (replacement_instrumentation_selected && !replacement_installed) {
      mark_consan_static_coverage_uninstrumented(*static_coverage_storage);
    }
    ConSanStaticCoverageRegistry::instance().record(*static_coverage_storage);
    static_coverage_storage.reset();
  };

  const CodeObjectReaderRegistry::ReaderBytes reader_bytes =
      CodeObjectReaderRegistry::instance().lookup(code_object_reader);
  if (reader_bytes) {
    const uint8_t *bytes = reader_bytes.bytes;
    const size_t size = reader_bytes.size;
    if (!config->kernel_name_allowlist.empty()) {
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
    // Reader handles may be destroyed and reused by the HSA runtime. Keep a
    // process-local identity for this particular load so retained coverage can
    // be joined to the correspondingly numbered captured object without
    // conflating two lifetimes of the same opaque handle.
    const uint64_t load_id = g_dump_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint64_t dump_id = config->dump_dir.empty() ? 0 : load_id;
    dump_code_object_bytes(*config, dump_id, code_object_reader.handle, "original",
                           std::span<const uint8_t>(bytes, size));

    const auto waitcheck_begin = std::chrono::steady_clock::now();
    (void)run_waitcheck_preflight(std::span<const uint8_t>(bytes, size), code_object_reader.handle);
    log_message(kLogInfo, "ConSan waitcheck timing reader=%llu elapsed_ms=%.3f",
                static_cast<unsigned long long>(code_object_reader.handle),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                          waitcheck_begin)
                    .count());

    rocjitsu::ConSanOptions patch_options;
    patch_options.flavor = config->flavor.value_or(rocjitsu::ConSanFlavor::None);
    patch_options.moi_engine = config->moi_engine;
    patch_options.moi_owner_source = config->moi_owner_source;
    patch_options.flat_provenance_mode = config->flat_provenance_mode;
    patch_options.fail_closed = config->fail_closed;
    patch_options.probe_nop = config->probe_nop;
    patch_options.probe_trampoline_nop = config->probe_trampoline_nop;
    patch_options.probe_endpgm = config->probe_endpgm;
    patch_options.probe_lds_endpgm = config->probe_lds_endpgm;
    patch_options.probe_lds_check_trap = config->probe_lds_check_trap;
    patch_options.probe_flat_check_trap = config->probe_flat_check_trap;
    patch_options.probe_flat_trap = config->probe_flat_trap;
    patch_options.abort_unmatched_barrier_wait = config->abort_unmatched_barrier_wait;
    patch_options.fault_drop_barrier = config->fault_drop_barrier;
    patch_options.fault_allow_destructive_incomplete_barrier_drop =
        config->fault_allow_destructive_incomplete_barrier_drop;
    patch_options.fault_move_barrier = config->fault_move_barrier;
    patch_options.collect_barrier_move_destinations = config->fault_move_barrier;
    patch_options.fault_allow_completing_conditional_barrier_move =
        config->fault_allow_completing_conditional_barrier_move;
    patch_options.fault_allow_destructive_divergent_barrier_move =
        config->fault_allow_destructive_divergent_barrier_move;
    patch_options.fault_mutate_barrier_id_scope = config->fault_mutate_barrier_id_scope;
    patch_options.fault_mutate_barrier_participants = config->fault_mutate_barrier_participants;
    patch_options.fault_barrier_move_direction = config->fault_barrier_move_direction;
    patch_options.fault_barrier_destination_identity = config->fault_barrier_destination_identity;
    patch_options.fault_barrier_sequence_identity = config->fault_barrier_sequence_identity;
    patch_options.fault_barrier_companion_site_identity =
        config->fault_barrier_companion_site_identity;
    patch_options.fault_barrier_companion_sequence_identity =
        config->fault_barrier_companion_sequence_identity;
    patch_options.fault_barrier_target_id = config->fault_barrier_target_id;
    patch_options.fault_barrier_target_participant_count =
        config->fault_barrier_target_participant_count;
    patch_options.fault_barrier_target_participant_mask =
        config->fault_barrier_target_participant_mask;
    patch_options.fault_atomic_wrong_address = config->fault_atomic_wrong_address;
    patch_options.fault_atomic_weaken_order = config->fault_atomic_weaken_order;
    patch_options.fault_atomic_order_edge = config->fault_atomic_order_edge;
    patch_options.fault_atomic_weaken_scope = config->fault_atomic_weaken_scope;
    patch_options.fault_lds_wrong_address = config->fault_lds_wrong_address;
    patch_options.fault_ordinary_wrong_address = config->fault_ordinary_wrong_address;
    patch_options.fault_ordinary_weaken_order = config->fault_ordinary_weaken_order;
    patch_options.fault_ordinary_weaken_scope = config->fault_ordinary_weaken_scope;
    patch_options.fault_atomic_address_delta = config->fault_atomic_address_delta;
    patch_options.fault_lds_address_vgpr = config->fault_lds_address_vgpr;
    patch_options.fault_ordinary_address_delta = config->fault_ordinary_address_delta;
    patch_options.fault_dry_run = config->fault_dry_run;
    // Exact fault selection is process-scoped in the HSA hook: workloads can
    // load runtime-helper code objects before the one containing the reviewed
    // site.  The process-wide atomic reservation below admits only the first match,
    // while retained fault telemetry lets the validation driver reject zero
    // matches.  Enforcing this inside each code-object transform would reject
    // every preceding nonmatching object.
    patch_options.fault_require_exactly_one = false;
    patch_options.sc_perturb_kind = config->sc_perturb_kind;
    patch_options.sc_perturb_edge = config->sc_perturb_edge;
    patch_options.sc_perturb_identity = config->sc_perturb_identity;
    patch_options.sc_perturb_index = config->sc_perturb_index;
    patch_options.sc_perturb_max = config->sc_perturb_max;
    patch_options.sc_perturb_sleep = config->sc_perturb_sleep;
    patch_options.sc_perturb_required_count = config->sc_perturb_required_count;
    patch_options.moi_init_owner_epoch = config->moi_init_owner_epoch;
    patch_options.moi_track_barriers = config->moi_track_barriers;
    patch_options.moi_track_atomics = config->moi_track_atomics;
    patch_options.moi_dynamic_access_records = config->moi_dynamic_access_records;
    patch_options.moi_inline_workgroup_shadow =
        config->moi_engine == rocjitsu::ConSanMoiEngine::InlineShadow;
    patch_options.moi_sampled_check = config->moi_sampled_check;
    patch_options.moi_partition_mask_debug = config->moi_partition_mask_debug;
    patch_options.force_vgpr_spill = config->test_force_vgpr_spill;
    patch_options.force_private_epoch = config->test_force_private_epoch;
    patch_options.test_kernel_name_filter = config->test_kernel_name_filter;
    patch_options.kernel_name_allowlist = config->kernel_name_allowlist;
    patch_options.fault_barrier_index = config->fault_barrier_index;
    patch_options.fault_atomic_index = config->fault_atomic_index;
    patch_options.fault_lds_index = config->fault_lds_index;
    patch_options.fault_ordinary_index = config->fault_ordinary_index;
    patch_options.fault_site_identity = config->fault_site_identity;
    patch_options.delay_mode = config->delay_mode;
    patch_options.delay_var_ssrc = config->delay_var_ssrc;
    patch_options.scratch_vgpr = config->scratch_vgpr;
    patch_options.moi_exec_save_sgpr = config->moi_exec_save_sgpr;
    patch_options.moi_owner_sgpr = config->moi_owner_sgpr;
    patch_options.moi_owner_vgpr = config->moi_owner_vgpr;
    patch_options.moi_epoch_vgpr = config->moi_epoch_vgpr;
    patch_options.report_buffer_address = config->report_buffer_address;
    patch_options.moi_report_buffer_address = config->moi_report_buffer_address;
    patch_options.moi_report_buffer_size = config->moi_report_buffer_size;
    patch_options.moi_max_workgroup_lds_bytes =
        runtime_group_segment_size_bytes(layer().core_table(), agent);
    patch_options.delay_nops = config->delay_nops;
    patch_options.patched_image_growth_limit = config->patched_image_growth_limit;
    patch_options.max_patches = config->max_patches;
    patch_options.max_patches_is_expert_limit = config->max_patches_explicit;
    patch_options.moi_sample_stride = config->moi_sample_stride;
    patch_options.moi_sample_offset = config->moi_sample_offset;
    patch_options.moi_runtime_sample_stride = config->moi_runtime_sample_stride;
    patch_options.moi_runtime_sample_offset = config->moi_runtime_sample_offset;
    patch_options.report_marker = config->report_marker;
    if (patch_options.flavor != rocjitsu::ConSanFlavor::None) {
      const std::optional<ConSanTransformReservationEstimate> reservation =
          consan_transform_major_image_reservation(size, patch_options.patched_image_growth_limit);
      const bool relative_growth = patch_options.patched_image_growth_limit.kind ==
                                   rocjitsu::ConSanPatchedImageGrowthLimitKind::InputPercent;
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
              relative_growth ? patch_options.patched_image_growth_limit.input_percent
                              : patch_options.patched_image_growth_limit.absolute_bytes));
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
        ConSanStaticCoverageRegistry::instance().record_unclassified_incomplete_code_object();
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
      rocjitsu::ConSanOptions probe_options = patch_options;
      // Keep the configured flavor: try_patch_consan intentionally skips all
      // ConSan planning for flavor=None, including dry-run fault planning.
      // The probe result is discarded, so retaining the flavor cannot install
      // instrumentation but does let the exact fault resolver identify this
      // dynamic code-object load.
      probe_options.fault_dry_run = true;
      probe_options.fault_require_exactly_one = false;
      probe_options.moi_report_buffer_address.reset();
      probe_options.moi_report_buffer_size = 0;
      const rocjitsu::ConSanResult probe =
          run_consan_transform(std::span<const uint8_t>(bytes, size), probe_options);
      const bool matched = probe.planned_fault_mutations != 0;
      if (probe.planned_fault_mutations > 1) {
        std::fprintf(stderr,
                     "[rocjitsu-dbi-hooks] ConSan fault load selection is ambiguous: "
                     "one reader planned %zu mutations\n",
                     probe.planned_fault_mutations);
        return reject_code_object_load(*config, HSA_STATUS_ERROR_INVALID_CODE_OBJECT,
                                       code_object_reader.handle, "ambiguous-fault-load-selection",
                                       fault_installation_evidence);
      }
      bool selected = matched;
      if (config->fault_load_occurrence) {
        std::optional<rocjitsu::ConSanFaultLoadSelection> selection;
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
        disable_fault_mutations(&patch_options);
    }
    if (patch_options.flavor == rocjitsu::ConSanFlavor::SuperCollider &&
        config->sc_report_mode == ScReportMode::Auto && !patch_options.report_buffer_address) {
      rocjitsu::ConSanOptions inventory_options = patch_options;
      const bool live_fault_transform =
          fault_mutations_enabled(patch_options) && !patch_options.fault_dry_run;
      if (live_fault_transform)
        disable_fault_mutations(&inventory_options);
      rocjitsu::ConSanResult inventory =
          run_consan_transform(std::span<const uint8_t>(bytes, size), inventory_options);
      if (!sc_inventory_needs_report_buffer(inventory)) {
        log_message(kLogInfo,
                    "ConSan SC auto report buffer skipped reader=%llu: no selected check sites",
                    static_cast<unsigned long long>(code_object_reader.handle));
        if (!live_fault_transform)
          patch_result_storage = std::move(inventory);
      } else {
        uint64_t auto_report_address = 0;
        uint64_t auto_report_generation = 0;
        if (!AutoScReportBufferRegistry::instance().allocate(
                layer().core_table(), agent, code_object_reader.handle, &auto_report_address,
                &auto_report_generation)) {
          std::fprintf(stderr, "[rocjitsu-dbi-hooks] ConSan SC automatic non-trapping report "
                               "allocation failed; analysis incomplete, refusing trap fallback\n");
          for (rocjitsu::ConSanSiteDispositionRecord &site : inventory.site_dispositions) {
            if (site.lowering_outcome == rocjitsu::ConSanSiteLoweringOutcome::Patched) {
              site.lowering_outcome = rocjitsu::ConSanSiteLoweringOutcome::ResourceFailed;
              site.lowering_reason = rocjitsu::ConSanSiteLoweringReason::UnsupportedResourcePlan;
              site.resource_reason = rocjitsu::ConSanRegisterPlanReason::InvalidRequest;
            }
          }
          inventory.outcome = rocjitsu::ConSanTransformOutcome::Unsupported;
          inventory.modified = false;
          inventory.final_validation_passed = false;
          inventory.elf_bytes.clear();
          inventory.patches.clear();
          inventory.warnings.emplace_back(
              "SuperCollider automatic report allocation failed; original code loaded "
              "without instrumentation");
          patch_result_storage = std::move(inventory);
          if (config->fail_closed || config->require_patch)
            return reject_code_object_load(
                *config, HSA_STATUS_ERROR_OUT_OF_RESOURCES, code_object_reader.handle,
                "supercollider-report-allocation", fault_installation_evidence);
        } else {
          patch_options.report_buffer_address = auto_report_address;
          auto_report_guard.note_sc(code_object_reader.handle, auto_report_generation);
        }
      }
    }
    if (patch_options.flavor == rocjitsu::ConSanFlavor::Moi &&
        patch_options.moi_report_buffer_address && !patch_options.moi_report_dispatch_id) {
      // Explicit report buffers are commonly installed immediately before a
      // fixture's code object is loaded. Give their emitted records the same
      // nonzero code-object identity used by automatic report buffers.
      patch_options.moi_report_dispatch_id = code_object_reader.handle;
    }
    std::optional<uint64_t> registered_auto_moi_report_generation;
    if (patch_options.flavor == rocjitsu::ConSanFlavor::Moi &&
        !patch_options.moi_report_buffer_address && config->moi_auto_report_buffer_size != 0) {
      rocjitsu::ConSanOptions inventory_options = patch_options;
      inventory_options.moi_report_buffer_size = 0;
      const bool live_fault_transform =
          fault_mutations_enabled(patch_options) && !patch_options.fault_dry_run;
      // A report-sizing pass is pristine MOI inventory, never the live
      // mutation stage. fault_dry_run returns after fault planning and before
      // MOI candidate/resource planning, so disable the mutation members while
      // retaining the complete MOI inventory pass. Retry binds the selected
      // mutation and runtime layout to this pristine inventory.
      if (live_fault_transform) {
        inventory_options.qualify_extended_barrier_pairs =
            rocjitsu::consan_qualifies_extended_barrier_pairs(patch_options);
        disable_fault_mutations(&inventory_options);
      }
      log_message(kLogInfo, "ConSan MOI inventory begin reader=%llu bytes=%zu",
                  static_cast<unsigned long long>(code_object_reader.handle), size);
      const auto inventory_begin = std::chrono::steady_clock::now();
      rocjitsu::ConSanResult inventory =
          run_consan_transform(std::span<const uint8_t>(bytes, size), inventory_options);
      log_message(kLogInfo, "ConSan MOI inventory end reader=%llu elapsed_ms=%.3f",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                            inventory_begin)
                      .count());
      if (!rocjitsu::consan_result_has_resolved_semantic_arch(inventory))
        return reject_unresolved_semantic_arch(*config, code_object_reader.handle,
                                               fault_installation_evidence);
      const bool inventory_requires_report_buffer =
          moi_inventory_needs_report_buffer(inventory, *config);
      bool auto_report_plan_available = inventory_requires_report_buffer;
      if (!inventory_requires_report_buffer) {
        log_message(kLogInfo,
                    "ConSan MOI auto report buffer skipped reader=%llu: no MOI report sites",
                    static_cast<unsigned long long>(code_object_reader.handle));
        if (!live_fault_transform)
          patch_result_storage = std::move(inventory);
      }

      uint64_t auto_report_address = 0;
      uint64_t auto_report_size = 0;
      uint64_t auto_report_generation = 0;
      uint64_t required_report_size = 0;
      uint64_t requested_report_size = 0;
      rocjitsu::ConSanMoiReportBufferLayout report_layout;
      std::optional<rocjitsu::ConSanMoiReportLayoutOverride> report_layout_override;
      std::optional<rocjitsu::ConSanMoiAutoReportInventory> planned_report_inventory;
      if (inventory_requires_report_buffer && !patch_result_storage &&
          patch_options.moi_dynamic_access_records) {
        if (!config->moi_auto_report_buffer_size_explicit) {
          log_message(kLogInfo,
                      "ConSan MOI auto report plan reader=%llu outcome="
                      "insufficient_report_capacity reason="
                      "dynamic_replay_requires_explicit_cap cap_bytes=%llu",
                      static_cast<unsigned long long>(code_object_reader.handle),
                      static_cast<unsigned long long>(config->moi_auto_report_buffer_size));
          reject_auto_moi_report_plan(code_object_reader.handle, /*required_size=*/0,
                                      config->moi_auto_report_buffer_size,
                                      "dynamic_replay_requires_explicit_cap");
          auto_report_plan_available = false;
          if (!live_fault_transform)
            patch_result_storage = std::move(inventory);
        } else {
          required_report_size = config->moi_auto_report_buffer_size;
          requested_report_size = config->moi_auto_report_buffer_size;
          report_layout = rocjitsu::consan_moi_report_buffer_layout_for_bytes(
              requested_report_size, config->moi_track_barriers, config->moi_track_atomics,
              config->moi_track_atomics);
          log_message(kLogInfo,
                      "ConSan MOI auto report plan reader=%llu outcome=complete "
                      "reason=bounded_expert_dynamic_replay required_bytes=%llu cap_bytes=%llu",
                      static_cast<unsigned long long>(code_object_reader.handle),
                      static_cast<unsigned long long>(required_report_size),
                      static_cast<unsigned long long>(config->moi_auto_report_buffer_size));
        }
      } else if (inventory_requires_report_buffer && !patch_result_storage) {
        const rocjitsu::ConSanMoiAutoReportInventory report_inventory =
            rocjitsu::inventory_consan_moi_auto_report(inventory, inventory_options,
                                                       std::span<const uint8_t>(bytes, size),
                                                       config->moi_auto_report_buffer_size);
        planned_report_inventory = report_inventory;
        const rocjitsu::ConSanMoiAutoReportPlan report_plan = rocjitsu::plan_consan_moi_auto_report(
            report_inventory, config->moi_auto_report_buffer_size);
        required_report_size = report_plan.required_bytes;
        requested_report_size = report_plan.required_bytes;
        report_layout = report_plan.layout;
        report_layout_override = rocjitsu::consan_moi_auto_report_layout_override(report_plan);
        log_message(
            kLogInfo,
            "ConSan MOI auto report plan reader=%llu outcome=%s reason=%s "
            "required_bytes=%llu cap_bytes=%llu per_buffer_ceiling=%llu "
            "process_ceiling=%llu access_ranges=%llu barriers=%llu atomics=%llu fences=%llu "
            "dispatch_banks=%llu owner_banks=%llu address_group_headroom=%llu "
            "diagnostics=%llu sampled_banks=%llu sampled_watchpoints=%llu inline_lds_bytes=%llu "
            "inline_releases=%llu inline_snapshots=%llu inline_tokens=%llu",
            static_cast<unsigned long long>(code_object_reader.handle),
            rocjitsu::consan_moi_auto_report_plan_outcome_name(report_plan.outcome).data(),
            rocjitsu::consan_moi_auto_report_plan_reason_name(report_plan.reason).data(),
            static_cast<unsigned long long>(report_plan.required_bytes),
            static_cast<unsigned long long>(config->moi_auto_report_buffer_size),
            static_cast<unsigned long long>(report_plan.ceiling_bytes),
            static_cast<unsigned long long>(rocjitsu::kConSanMoiAutoReportProcessCeilingBytes),
            static_cast<unsigned long long>(report_inventory.access_range_count),
            static_cast<unsigned long long>(report_inventory.barrier_event_count),
            static_cast<unsigned long long>(report_inventory.atomic_event_count),
            static_cast<unsigned long long>(report_inventory.fence_event_count),
            static_cast<unsigned long long>(
                report_inventory.record_replay_access_dispatch_bank_count),
            static_cast<unsigned long long>(report_inventory.record_replay_access_owner_bank_count),
            static_cast<unsigned long long>(report_inventory.record_replay_address_group_headroom),
            static_cast<unsigned long long>(report_inventory.diagnostic_count),
            static_cast<unsigned long long>(report_inventory.sampled_range_bank_count),
            static_cast<unsigned long long>(report_inventory.sampled_watchpoint_count),
            static_cast<unsigned long long>(report_inventory.inline_lds_bytes),
            static_cast<unsigned long long>(report_inventory.inline_atomic_release_count),
            static_cast<unsigned long long>(report_inventory.inline_causal_snapshot_count),
            static_cast<unsigned long long>(report_inventory.inline_acquired_epoch_token_count));
      }
      if (auto_report_plan_available && !patch_result_storage &&
          allocate_auto_moi_report_buffer(
              layer().core_table(), agent, code_object_reader.handle, required_report_size,
              requested_report_size, config->moi_auto_report_buffer_size, report_layout,
              config->moi_engine, config->moi_track_barriers, config->moi_track_atomics,
              config->test_seed_inline_exact_odd, &auto_report_address, &auto_report_size,
              &auto_report_generation)) {
        patch_options.moi_report_buffer_address = auto_report_address;
        patch_options.moi_report_buffer_size = auto_report_size;
        patch_options.moi_report_layout = report_layout_override;
        patch_options.moi_report_generation = auto_report_generation;
        registered_auto_moi_report_generation = auto_report_generation;
        auto_report_guard.note_moi(code_object_reader.handle, auto_report_generation);
        patch_options.moi_report_dispatch_id = code_object_reader.handle;
        if (live_fault_transform && planned_report_inventory)
          live_fault_auto_report_capacity_inventory = *planned_report_inventory;
        if (!live_fault_transform ||
            patch_options.sc_perturb_kind == rocjitsu::ConSanPerturbationKind::None)
          reusable_moi_inventory = std::move(inventory);
      } else if (auto_report_plan_available && !patch_result_storage && config->fail_closed) {
        return reject_code_object_load(*config, HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                                       code_object_reader.handle, "moi-report-allocation",
                                       fault_installation_evidence);
      } else if (auto_report_plan_available && !patch_result_storage) {
        if (!live_fault_transform)
          patch_result_storage = std::move(inventory);
      }
    }

    log_message(kLogInfo, "ConSan patch begin reader=%llu bytes=%zu",
                static_cast<unsigned long long>(code_object_reader.handle), size);
    const auto patch_begin = std::chrono::steady_clock::now();
    if (!patch_result_storage) {
      if (reusable_moi_inventory) {
        patch_result_storage =
            retry_consan_moi_transform(std::span<const uint8_t>(bytes, size), patch_options,
                                       std::move(*reusable_moi_inventory));
      } else {
        patch_result_storage =
            run_consan_transform(std::span<const uint8_t>(bytes, size), patch_options);
      }
    }
    const rocjitsu::ConSanResult &patch_result = *patch_result_storage;
    fault_installation_evidence.record_applied_mutations(patch_result.applied_fault_mutations);
    if (live_fault_auto_report_capacity_inventory) {
      const rocjitsu::ConSanMoiAutoReportInventory live_requirement =
          rocjitsu::inventory_consan_moi_auto_report(patch_result, patch_options,
                                                     std::span<const uint8_t>(bytes, size),
                                                     config->moi_auto_report_buffer_size);
      if (!rocjitsu::consan_moi_auto_report_inventory_covers(
              *live_fault_auto_report_capacity_inventory, live_requirement)) {
        std::fprintf(stderr, "[rocjitsu-dbi-hooks] ConSan internal invariant violation: live fault "
                             "transform grew the automatic MOI report inventory\n");
        reject_auto_moi_report_plan(code_object_reader.handle, patch_options.moi_report_buffer_size,
                                    config->moi_auto_report_buffer_size,
                                    "live_fault_inventory_growth");
        return reject_code_object_load(
            *config, HSA_STATUS_ERROR_OUT_OF_RESOURCES, code_object_reader.handle,
            "moi-report-live-inventory-growth", fault_installation_evidence);
      }
    }
    if (!rocjitsu::consan_result_has_resolved_semantic_arch(patch_result))
      return reject_unresolved_semantic_arch(*config, code_object_reader.handle,
                                             fault_installation_evidence);
    if (registered_auto_moi_report_generation)
      register_auto_moi_report_metadata(code_object_reader.handle,
                                        *registered_auto_moi_report_generation, patch_result);
    install_action = rocjitsu::consan_install_action(patch_result, config->fail_closed);
    replacement_instrumentation_selected =
        install_action == rocjitsu::ConSanInstallAction::LoadReplacement;
    log_message(
        kLogInfo,
        "ConSan patch end reader=%llu visited=%s modified=%s outcome=%s errors=%zu "
        "warnings=%zu patches=%zu patch_ms=%.3f",
        static_cast<unsigned long long>(code_object_reader.handle),
        patch_result.visited_code_object ? "true" : "false",
        patch_result.modified ? "true" : "false",
        rocjitsu::consan_transform_outcome_name(patch_result.outcome), patch_result.errors.size(),
        patch_result.warnings.size(), patch_result.patches.size(),
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - patch_begin)
            .count());
    if (consan_log_level_enabled(kLogInfo) && patch_result.flat_selection_telemetry) {
      const rocjitsu::ConSanFlatSelectionTelemetry &selection =
          *patch_result.flat_selection_telemetry;
      const bool has_discarded_branch_work =
          selection.discarded_branch_only_placement_failure_count != 0u ||
          !rocjitsu::branch_only_relay_telemetry_is_empty(
              selection.discarded_branch_only_routing) ||
          !rocjitsu::branch_only_reservoir_telemetry_is_empty(
              selection.discarded_branch_only_reservoir_telemetry);
      const rocjitsu::ConSanBranchOnlyReservoirTelemetry &flat_reservoirs =
          selection.branch_only_reservoir_telemetry;
      const std::string flat_routing =
          rocjitsu::format_consan_branch_only_routing_telemetry(selection.branch_only_routing);
      log_message(
          kLogInfo,
          "ConSan SC flat selection reader=%llu supported=%zu target=%zu selected=%zu "
          "branch_only_candidates=%zu branch_only_selected=%zu placement_failed=%zu "
          "%s "
          "reservoirs_planned=%zu reservoirs_used=%zu reservoirs_unused=%zu "
          "reservoir_planned_appended_bytes=%zu reservoir_used_appended_bytes=%zu "
          "reservoir_unused_appended_bytes=%zu "
          "discarded_branch_work=%s "
          "fixed_stack=%zu dynamic_stack=%zu mixed_stack=%zu missing_vcc_save=%zu "
          "missing_scratch=%zu spill_backed=%zu mixed_stack_spill_rejected=%zu "
          "dynamic_stack_spill_failed=%zu private_spill_failed=%zu original_relays=%zu "
          "selected_anchor_relays=%zu",
          static_cast<unsigned long long>(code_object_reader.handle),
          selection.supported_candidate_count, selection.selection_target,
          selection.selected_candidate_count, selection.branch_only_candidate_count,
          selection.branch_only_selected_count, selection.branch_only_placement_failure_count,
          flat_routing.c_str(), flat_reservoirs.planned_reservoir_count,
          flat_reservoirs.used_reservoir_count, flat_reservoirs.unused_reservoir_count,
          flat_reservoirs.planned_appended_bytes, flat_reservoirs.used_appended_bytes,
          flat_reservoirs.unused_appended_bytes, has_discarded_branch_work ? "true" : "false",
          selection.fixed_stack_candidate_count, selection.dynamic_stack_candidate_count,
          selection.mixed_stack_candidate_count, selection.missing_vcc_save_candidate_count,
          selection.missing_scratch_candidate_count, selection.spill_backed_candidate_count,
          selection.mixed_stack_spill_rejection_count,
          selection.dynamic_stack_spill_encoding_failure_count,
          selection.private_spill_encoding_failure_count, selection.original_nop_relay_slot_count,
          selection.selected_anchor_relay_slot_count);
      const rocjitsu::ConSanBranchOnlyRoutingTelemetry &discarded_routing =
          selection.discarded_branch_only_routing;
      const rocjitsu::ConSanBranchOnlyReservoirTelemetry &discarded_reservoirs =
          selection.discarded_branch_only_reservoir_telemetry;
      if (has_discarded_branch_work) {
        const std::string discarded_routing_fields =
            rocjitsu::format_consan_branch_only_routing_telemetry(discarded_routing);
        log_message(
            kLogInfo,
            "ConSan SC flat discarded branch routing reader=%llu placement_failed=%zu "
            "%s "
            "reservoirs_planned=%zu reservoirs_used=%zu reservoirs_unused=%zu "
            "reservoir_planned_appended_bytes=%zu reservoir_used_appended_bytes=%zu "
            "reservoir_unused_appended_bytes=%zu",
            static_cast<unsigned long long>(code_object_reader.handle),
            selection.discarded_branch_only_placement_failure_count,
            discarded_routing_fields.c_str(), discarded_reservoirs.planned_reservoir_count,
            discarded_reservoirs.used_reservoir_count, discarded_reservoirs.unused_reservoir_count,
            discarded_reservoirs.planned_appended_bytes, discarded_reservoirs.used_appended_bytes,
            discarded_reservoirs.unused_appended_bytes);
      }
    }
    const rocjitsu::ConSanBranchOnlyRoutingTelemetry &lds_routing =
        patch_result.lds_branch_only_routing_telemetry;
    const rocjitsu::ConSanBranchOnlyReservoirTelemetry &lds_reservoirs =
        patch_result.lds_relay_reservoir_telemetry;
    if (consan_log_level_enabled(kLogInfo) &&
        (!rocjitsu::branch_only_relay_telemetry_is_empty(lds_routing) ||
         !rocjitsu::branch_only_reservoir_telemetry_is_empty(lds_reservoirs))) {
      const std::string lds_routing_fields =
          rocjitsu::format_consan_branch_only_routing_telemetry(lds_routing);
      log_message(kLogInfo,
                  "ConSan SC LDS branch routing reader=%llu %s "
                  "lds_replay_limit_reached=%zu "
                  "lds_relay_reservoirs_planned=%zu lds_relay_reservoirs_used=%zu "
                  "lds_relay_reservoirs_unused=%zu "
                  "lds_relay_reservoir_planned_appended_bytes=%zu "
                  "lds_relay_reservoir_used_appended_bytes=%zu "
                  "lds_relay_reservoir_unused_appended_bytes=%zu",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  lds_routing_fields.c_str(), lds_reservoirs.lds_replay_limit_reached_count,
                  lds_reservoirs.planned_reservoir_count, lds_reservoirs.used_reservoir_count,
                  lds_reservoirs.unused_reservoir_count, lds_reservoirs.planned_appended_bytes,
                  lds_reservoirs.used_appended_bytes, lds_reservoirs.unused_appended_bytes);
    }
    const rocjitsu::ConSanBranchOnlyRoutingTelemetry &moi_routing =
        patch_result.moi_branch_only_routing_telemetry;
    const rocjitsu::ConSanBranchOnlyReservoirTelemetry &moi_reservoirs =
        patch_result.moi_branch_only_reservoir_telemetry;
    if (consan_log_level_enabled(kLogInfo) &&
        (patch_result.moi_branch_only_placement_failure_count != 0u ||
         !rocjitsu::branch_only_relay_telemetry_is_empty(moi_routing) ||
         !rocjitsu::branch_only_reservoir_telemetry_is_empty(moi_reservoirs))) {
      const std::string moi_routing_fields =
          rocjitsu::format_consan_branch_only_routing_telemetry(moi_routing);
      log_message(kLogInfo,
                  "ConSan MOI branch routing reader=%llu placement_failed=%zu "
                  "%s "
                  "reservoirs_planned=%zu reservoirs_used=%zu reservoirs_unused=%zu "
                  "reservoir_planned_appended_bytes=%zu reservoir_used_appended_bytes=%zu "
                  "reservoir_unused_appended_bytes=%zu",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  patch_result.moi_branch_only_placement_failure_count, moi_routing_fields.c_str(),
                  moi_reservoirs.planned_reservoir_count, moi_reservoirs.used_reservoir_count,
                  moi_reservoirs.unused_reservoir_count, moi_reservoirs.planned_appended_bytes,
                  moi_reservoirs.used_appended_bytes, moi_reservoirs.unused_appended_bytes);
    }
    const rocjitsu::ConSanPlanningWorkTelemetry &planning_work =
        patch_result.planning_work_telemetry;
    if (!rocjitsu::consan_planning_work_telemetry_is_empty(planning_work)) {
      log_message(kLogInfo,
                  "ConSan planning work reader=%llu sopp_relay_work=%zu "
                  "sopp_relay_exhaustions=%zu direct_reservoir_work=%zu "
                  "direct_reservoir_exhaustions=%zu lds_relay_layout_work=%zu "
                  "lds_relay_layout_exhaustions=%zu lds_convergence_work=%zu "
                  "lds_convergence_exhaustions=%zu",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  planning_work.sopp_relay_work_count, planning_work.sopp_relay_exhaustion_count,
                  planning_work.direct_reservoir_work_count,
                  planning_work.direct_reservoir_exhaustion_count,
                  planning_work.lds_relay_layout_work_count,
                  planning_work.lds_relay_layout_exhaustion_count,
                  planning_work.lds_convergence_work_count,
                  planning_work.lds_convergence_exhaustion_count);
    }

    for (const std::string &warning : patch_result.warnings)
      log_message(kLogVerbose, "%s", warning.c_str());
    if (!patch_result.errors.empty()) {
      for (const std::string &error : patch_result.errors)
        std::fprintf(stderr, "[rocjitsu-dbi-hooks] %s\n", error.c_str());
      if (config->fail_closed)
        return reject_code_object_load(
            *config, HSA_STATUS_ERROR_INVALID_CODE_OBJECT, code_object_reader.handle,
            "transform-error", fault_installation_evidence,
            patch_result.transform_failure_cause ? rocjitsu::consan_transform_failure_cause_name(
                                                       *patch_result.transform_failure_cause)
                                                 : std::string_view{});
    }
    if (install_action == rocjitsu::ConSanInstallAction::Reject) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] ConSan outcome %s is not installable "
                   "(fail_closed=%s)\n",
                   rocjitsu::consan_transform_outcome_name(patch_result.outcome),
                   config->fail_closed ? "true" : "false");
      return reject_code_object_load(*config, HSA_STATUS_ERROR_INVALID_CODE_OBJECT,
                                     code_object_reader.handle, "non-installable-transform-outcome",
                                     fault_installation_evidence);
    }
    const bool requires_dynamic_private_dispatch_adjustment =
        replacement_instrumentation_selected &&
        std::ranges::any_of(patch_result.patches, [](const rocjitsu::ConSanPatchInfo &patch) {
          return patch.dynamic_private_segment_addend != 0u;
        });
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
      install_action = rocjitsu::ConSanInstallAction::LoadOriginal;
    }

    // A large production object may produce hundreds of thousands of detailed
    // inventory, coverage, and patch-proof records below. Preserve the stable
    // line-oriented format while amortizing the stderr lock and write cost.
    ScopedDetailedLogBatch detailed_log_batch;
    log_message(
        kLogInfo,
        "ConSan inventory reader=%llu flavor=%s moi_engine=%s bytes=%zu visited=%s modified=%s "
        "delay_nops=%u fail_closed=%s probe_nop=%s probe_trampoline_nop=%s "
        "probe_endpgm=%s probe_lds_endpgm=%s check_trap_mode=%s probe_lds_check_trap=%s "
        "probe_flat_check_trap=%s probe_flat_trap=%s fault_drop_barrier=%s "
        "moi_init_owner_epoch=%s moi_track_barriers=%s moi_track_atomics=%s "
        "moi_dynamic_access_records=%s "
        "fault_barrier_index=%u "
        "delay_mode=%s delay_var_ssrc=%u "
        "patched_image_growth_limit_kind=%s patched_image_growth_limit_value=%llu "
        "process_concurrent_transform_limit_bytes=%s "
        "process_patched_image_limit_bytes=%s "
        "process_patched_image_growth_limit_bytes=%s "
        "max_patches=%u max_patches_source=%s tmp_vgpr=%s moi_exec_save_sgpr=%s "
        "moi_owner_source=%s moi_owner_sgpr=%s moi_owner_vgpr=%s moi_epoch_vgpr=%s "
        "report_buffer=%s report_marker=%u "
        "moi_report_buffer=%s moi_report_buffer_size=%llu "
        "moi_auto_report_buffer_size=%llu require_patch=%s",
        static_cast<unsigned long long>(code_object_reader.handle),
        flavor_name(patch_options.flavor),
        rocjitsu::consan_moi_engine_name(patch_options.moi_engine), patch_result.input_size,
        patch_result.visited_code_object ? "true" : "false",
        patch_result.modified ? "true" : "false", config->delay_nops,
        config->fail_closed ? "true" : "false", config->probe_nop ? "true" : "false",
        config->probe_trampoline_nop ? "true" : "false", config->probe_endpgm ? "true" : "false",
        config->probe_lds_endpgm ? "true" : "false", check_trap_mode_name(config->check_trap_mode),
        config->probe_lds_check_trap ? "true" : "false",
        config->probe_flat_check_trap ? "true" : "false",
        config->probe_flat_trap ? "true" : "false", config->fault_drop_barrier ? "true" : "false",
        config->moi_init_owner_epoch ? "true" : "false",
        config->moi_track_barriers ? "true" : "false", config->moi_track_atomics ? "true" : "false",
        config->moi_dynamic_access_records ? "true" : "false", config->fault_barrier_index,
        delay_mode_name(config->delay_mode), config->delay_var_ssrc,
        patched_image_growth_limit_kind_name(patch_options.patched_image_growth_limit.kind),
        static_cast<unsigned long long>(
            patched_image_growth_limit_value(patch_options.patched_image_growth_limit)),
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
        config->moi_exec_save_sgpr ? std::to_string(*config->moi_exec_save_sgpr).c_str() : "unset",
        owner_source_name(config->moi_owner_source),
        config->moi_owner_sgpr ? std::to_string(*config->moi_owner_sgpr).c_str() : "unset",
        config->moi_owner_vgpr ? std::to_string(*config->moi_owner_vgpr).c_str() : "unset",
        config->moi_epoch_vgpr ? std::to_string(*config->moi_epoch_vgpr).c_str() : "unset",
        config->report_buffer_address ? std::to_string(*config->report_buffer_address).c_str()
                                      : "disabled",
        config->report_marker,
        patch_options.moi_report_buffer_address
            ? std::to_string(*patch_options.moi_report_buffer_address).c_str()
            : "disabled",
        static_cast<unsigned long long>(patch_options.moi_report_buffer_size),
        static_cast<unsigned long long>(config->moi_auto_report_buffer_size),
        config->require_patch ? "true" : "false");
    if (patch_result.parsed_code_object) {
      log_message(kLogInfo,
                  "ConSan code-object reader=%llu target=%s arch=%s text_sections=%zu "
                  "kernels=%zu functions=%zu",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  rj_code_target_name(patch_result.target),
                  rj_code_arch_name(rj_code_arch_for_target(patch_result.target)),
                  patch_result.text_sections.size(), patch_result.kernels.size(),
                  patch_result.functions.size());
    }
    log_message(kLogInfo, "ConSan fault inventory reader=%llu sites=%zu",
                static_cast<unsigned long long>(code_object_reader.handle),
                patch_result.fault_sites.size());
    for (const rocjitsu::ConSanFaultSite &site : patch_result.fault_sites) {
      const OwnerLogFields owners = owner_log_fields(site.execution_owners, patch_result.kernels);
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
    log_message(kLogInfo, "ConSan barrier destination inventory reader=%llu destinations=%zu",
                static_cast<unsigned long long>(code_object_reader.handle),
                patch_result.barrier_move_destinations.size());
    for (const rocjitsu::ConSanBarrierMoveDestination &destination :
         patch_result.barrier_move_destinations) {
      const OwnerLogFields owners =
          owner_log_fields(destination.execution_owners, patch_result.kernels);
      std::string reason =
          destination.rejection_reason.empty() ? "-" : destination.rejection_reason;
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
                  destination.suitable ? "true" : "false", reason.c_str(),
                  barrier_move_cfg_contract_name(destination.cfg_contract),
                  structured_guard_block.c_str(), structured_source_block.c_str(),
                  static_cast<unsigned long long>(destination.structured_guard_offset.value_or(0)),
                  static_cast<unsigned long long>(destination.structured_source_offset.value_or(0)),
                  destination.execution_owners.size(), owners.names.c_str(), owners.proofs.c_str());
    }
    for (const rocjitsu::ConSanFaultMutationPlan &plan : patch_result.fault_plans) {
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
        static_cast<unsigned long long>(code_object_reader.handle),
        patch_result.requested_fault_mutations, patch_result.planned_fault_mutations,
        patch_result.applied_fault_mutations, process_prior_fault_applications,
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
        patch_result.applied_fault_mutations > 1u) {
      return reject_code_object_load(*config, HSA_STATUS_ERROR_INVALID_CODE_OBJECT,
                                     code_object_reader.handle, "multiple-fault-mutations-applied",
                                     fault_installation_evidence);
    }
    log_message(kLogInfo, "ConSan SC perturb inventory reader=%llu candidates=%zu",
                static_cast<unsigned long long>(code_object_reader.handle),
                patch_result.perturbation_candidates.size());
    for (const rocjitsu::ConSanPerturbationCandidate &candidate :
         patch_result.perturbation_candidates) {
      log_message(kLogVerbose,
                  "ConSan SC perturb candidate reader=%llu identity=%s sequence=%s kind=%s "
                  "edge=%s container=%s container_kind=%s block=%u anchor=%s "
                  "anchor_text_offset=0x%llx anchor_size=%u eligible=%s reason=%s",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  candidate.identity.c_str(), candidate.sequence_identity.c_str(),
                  perturbation_kind_name(candidate.kind), perturbation_edge_name(candidate.edge),
                  candidate.container_name.c_str(), candidate.in_kernel ? "kernel" : "function",
                  candidate.basic_block_index, candidate.anchor_event_identity.c_str(),
                  static_cast<unsigned long long>(candidate.anchor_text_offset),
                  candidate.anchor_size, candidate.eligible ? "true" : "false",
                  candidate.rejection_reason.empty() ? "-" : candidate.rejection_reason.c_str());
    }
    for (const rocjitsu::ConSanPerturbationPlan &plan : patch_result.perturbation_plans) {
      log_message(kLogInfo,
                  "ConSan SC perturb plan reader=%llu dry_run=%s identity=%s sequence=%s "
                  "kind=%s edge=%s anchor=%s anchor_text_offset=0x%llx anchor_size=%u sleep=%u",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  config->fault_dry_run ? "true" : "false", plan.candidate_identity.c_str(),
                  plan.sequence_identity.c_str(), perturbation_kind_name(plan.kind),
                  perturbation_edge_name(plan.edge), plan.anchor_event_identity.c_str(),
                  static_cast<unsigned long long>(plan.anchor_text_offset), plan.anchor_size,
                  plan.sleep_imm);
    }
    log_message(
        kLogInfo,
        "ConSan SC perturb summary reader=%llu requested=%zu planned=%zu applied=%zu max=%u "
        "required=%u sleep=%u",
        static_cast<unsigned long long>(code_object_reader.handle),
        patch_result.requested_perturbations, patch_result.planned_perturbations,
        patch_result.applied_perturbations, config->sc_perturb_max,
        config->sc_perturb_required_count, config->sc_perturb_sleep);
    for (const rocjitsu::ConSanAccessPlan &plan : patch_result.access_plans) {
      log_message(kLogVerbose,
                  "ConSan access plan reader=%llu dry_run=true identity=%s container=%s "
                  "container_kind=%s kind=%s planned=1 text_offset=0x%llx",
                  static_cast<unsigned long long>(code_object_reader.handle), plan.identity.c_str(),
                  plan.container_name.c_str(), plan.in_kernel ? "kernel" : "function",
                  plan.kind.c_str(), static_cast<unsigned long long>(plan.text_offset));
    }
    if (patch_result.composite_proof) {
      const rocjitsu::ConSanCompositeProof &proof = *patch_result.composite_proof;
      log_message(
          kLogInfo,
          "ConSan composite proof reader=%llu staged_composition_validated=true "
          "pristine_identity=%s pristine_sequence=%s pristine_container=%s "
          "pristine_container_kind=%s pristine_owner_descriptor=0x%llx "
          "pristine_edge=%s pristine_anchor=%s pristine_anchor_text_offset=0x%llx "
          "pristine_anchor_size=%u translated_identity=%s translated_edge=%s "
          "translated_anchor_text_offset=0x%llx translated_anchor_size=%u "
          "anchor_relation=%s cache_companion_identity=%s atomic_overlap=%s "
          "removed_cache_boundary=%s removed_cache_non_resurrection_validated=%s "
          "atomic_mutation_anchor_text_offset=0x%llx",
          static_cast<unsigned long long>(code_object_reader.handle),
          proof.pristine_identity.c_str(), proof.pristine_sequence.c_str(),
          proof.pristine_container.c_str(), proof.pristine_in_kernel ? "kernel" : "function",
          static_cast<unsigned long long>(proof.pristine_owner_descriptor),
          perturbation_edge_name(proof.pristine_edge), proof.pristine_anchor.c_str(),
          static_cast<unsigned long long>(proof.pristine_anchor_text_offset),
          proof.pristine_anchor_size, proof.translated_identity.c_str(),
          perturbation_edge_name(proof.translated_edge),
          static_cast<unsigned long long>(proof.translated_anchor_text_offset),
          proof.translated_anchor_size, proof.anchor_relation.c_str(),
          proof.cache_companion_identity.c_str(), proof.atomic_overlap ? "true" : "false",
          proof.removed_cache_boundary ? "true" : "false",
          proof.removed_cache_non_resurrection_applicable
              ? (proof.removed_cache_non_resurrection_validated ? "true" : "false")
              : "not-applicable",
          static_cast<unsigned long long>(proof.atomic_mutation_anchor_text_offset.value_or(0)));
    }
    log_message(kLogInfo, "ConSan sync inventory reader=%llu events=%zu",
                static_cast<unsigned long long>(code_object_reader.handle),
                patch_result.sync_events.size());
    for (const rocjitsu::ConSanSyncEvent &event : patch_result.sync_events) {
      const OwnerLogFields owners = owner_log_fields(event.execution_owners, patch_result.kernels);
      std::string reason = event.confidence_reason;
      std::ranges::replace(reason, ' ', '-');
      const std::string static_offset =
          event.static_byte_offset ? std::to_string(*event.static_byte_offset) : "-";
      const std::string raw_scope = event.raw_scope ? std::to_string(*event.raw_scope) : "-";
      const std::string participant_count =
          event.participant_count ? std::to_string(*event.participant_count) : "-";
      const std::string participant_mask =
          event.participant_mask ? std::to_string(*event.participant_mask) : "-";
      const std::string barrier_id = event.barrier_id ? std::to_string(*event.barrier_id) : "-";
      const std::string barrier_raw_selector =
          event.barrier_raw_operand_selector ? std::to_string(*event.barrier_raw_operand_selector)
                                             : "-";
      const std::string barrier_literal_width =
          event.barrier_literal_width_bits ? std::to_string(*event.barrier_literal_width_bits)
                                           : "-";
      const std::string barrier_literal_value =
          event.barrier_literal_value ? std::to_string(*event.barrier_literal_value) : "-";
      const std::string barrier_raw_simm16 =
          event.barrier_raw_simm16 ? std::to_string(*event.barrier_raw_simm16) : "-";
      log_message(
          kLogVerbose,
          "ConSan sync event reader=%llu identity=%s kind=%s operation=%s "
          "address_source=%s memory_role=%s memory_role_confidence=%s rmw_outcome=%s "
          "confidence=%s reason=%s "
          "container=%s container_kind=%s occurrence=%u text_offset=0x%llx "
          "file_offset=0x%llx size=%u width_bits=%u mnemonic=%s static_offset=%s "
          "raw_scope=%s barrier_id=%s barrier_operand_source=%s barrier_raw_selector=%s "
          "barrier_literal_width_bits=%s barrier_literal_value=%s barrier_raw_simm16=%s "
          "barrier_scope=%s "
          "participant_count=%s participant_mask=%s "
          "owners=%zu owner_names=%s owner_proofs=%s",
          static_cast<unsigned long long>(code_object_reader.handle), event.identity.c_str(),
          sync_event_kind_name(event.kind), sync_operation_name(event.operation),
          sync_address_source_name(event.address_source), sync_memory_role_name(event.memory_role),
          sync_confidence_name(event.memory_role_confidence),
          sync_rmw_outcome_name(event.rmw_outcome), sync_confidence_name(event.confidence),
          reason.c_str(), event.container_name.c_str(), event.in_kernel ? "kernel" : "function",
          event.occurrence, static_cast<unsigned long long>(event.text_offset),
          static_cast<unsigned long long>(event.file_offset), event.size, event.width_bits,
          event.mnemonic.c_str(), static_offset.c_str(), raw_scope.c_str(), barrier_id.c_str(),
          barrier_operand_source_name(event.barrier_operand_source), barrier_raw_selector.c_str(),
          barrier_literal_width.c_str(), barrier_literal_value.c_str(), barrier_raw_simm16.c_str(),
          barrier_scope_name(event.barrier_scope), participant_count.c_str(),
          participant_mask.c_str(), event.execution_owners.size(), owners.names.c_str(),
          owners.proofs.c_str());
    }
    log_message(kLogInfo, "ConSan sync sequence inventory reader=%llu sequences=%zu",
                static_cast<unsigned long long>(code_object_reader.handle),
                patch_result.sync_sequences.size());
    for (const rocjitsu::ConSanSyncSequence &sequence : patch_result.sync_sequences) {
      const OwnerLogFields owners =
          owner_log_fields(sequence.execution_owners, patch_result.kernels);
      std::string reason = sequence.confidence_reason;
      std::ranges::replace(reason, ' ', '-');
      std::string members;
      for (const std::string &identity : sequence.member_event_identities) {
        if (!members.empty())
          members += ',';
        members += identity;
      }
      const std::string block =
          sequence.basic_block_index ? std::to_string(*sequence.basic_block_index) : "-";
      const std::string static_offset =
          sequence.static_byte_offset ? std::to_string(*sequence.static_byte_offset) : "-";
      const std::string raw_scope = sequence.raw_scope ? std::to_string(*sequence.raw_scope) : "-";
      const std::string participant_count =
          sequence.participant_count ? std::to_string(*sequence.participant_count) : "-";
      const std::string participant_mask =
          sequence.participant_mask ? std::to_string(*sequence.participant_mask) : "-";
      const std::string barrier_id =
          sequence.barrier_id ? std::to_string(*sequence.barrier_id) : "-";
      const std::string barrier_raw_selector =
          sequence.barrier_raw_operand_selector
              ? std::to_string(*sequence.barrier_raw_operand_selector)
              : "-";
      const std::string barrier_literal_width =
          sequence.barrier_literal_width_bits ? std::to_string(*sequence.barrier_literal_width_bits)
                                              : "-";
      const std::string barrier_literal_value =
          sequence.barrier_literal_value ? std::to_string(*sequence.barrier_literal_value) : "-";
      const std::string barrier_raw_simm16 =
          sequence.barrier_raw_simm16 ? std::to_string(*sequence.barrier_raw_simm16) : "-";
      std::array<char, 32> release_wait_offset{};
      if (sequence.release_wait_text_offset) {
        std::snprintf(release_wait_offset.data(), release_wait_offset.size(), "0x%llx",
                      static_cast<unsigned long long>(*sequence.release_wait_text_offset));
      } else {
        release_wait_offset[0] = '-';
      }
      log_message(kLogVerbose,
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
                  static_cast<unsigned long long>(code_object_reader.handle),
                  sequence.identity.c_str(), sync_sequence_kind_name(sequence.kind),
                  sync_operation_name(sequence.operation),
                  sync_address_source_name(sequence.address_source),
                  sync_memory_role_name(sequence.memory_role),
                  sync_confidence_name(sequence.memory_role_confidence),
                  sync_rmw_outcome_name(sequence.rmw_outcome),
                  sync_confidence_name(sequence.confidence), reason.c_str(),
                  sequence.container_name.c_str(), sequence.in_kernel ? "kernel" : "function",
                  block.c_str(), static_cast<unsigned long long>(sequence.begin_text_offset),
                  static_cast<unsigned long long>(sequence.end_text_offset), sequence.width_bits,
                  static_offset.c_str(), raw_scope.c_str(), barrier_id.c_str(),
                  barrier_operand_source_name(sequence.barrier_operand_source),
                  barrier_raw_selector.c_str(), barrier_literal_width.c_str(),
                  barrier_literal_value.c_str(), barrier_raw_simm16.c_str(),
                  barrier_scope_name(sequence.barrier_scope), release_wait_offset.data(),
                  participant_count.c_str(), participant_mask.c_str(), members.c_str(),
                  sequence.execution_owners.size(), owners.names.c_str(), owners.proofs.c_str());
    }
    log_message(kLogInfo, "ConSan barrier lifecycle inventory reader=%llu groups=%zu",
                static_cast<unsigned long long>(code_object_reader.handle),
                patch_result.barrier_lifecycle_groups.size());
    for (const rocjitsu::ConSanBarrierLifecycleGroup &group :
         patch_result.barrier_lifecycle_groups) {
      std::string members;
      for (const std::string &identity : group.member_event_identities) {
        if (!members.empty())
          members += ',';
        members += identity;
      }
      std::string reason = group.rejection_reason.empty() ? "-" : group.rejection_reason;
      std::ranges::replace(reason, ' ', '-');
      log_message(kLogVerbose,
                  "ConSan barrier lifecycle reader=%llu identity=%s container=%s "
                  "container_kind=%s block=%s begin_text_offset=0x%llx "
                  "end_text_offset=0x%llx barrier_id=%s barrier_scope=%s admissible=%s "
                  "confidence=%s reason=%s members=%s",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  group.identity.c_str(), group.container_name.c_str(),
                  group.in_kernel ? "kernel" : "function",
                  group.basic_block_index ? std::to_string(*group.basic_block_index).c_str() : "-",
                  static_cast<unsigned long long>(group.begin_text_offset),
                  static_cast<unsigned long long>(group.end_text_offset),
                  group.barrier_id ? std::to_string(*group.barrier_id).c_str() : "-",
                  barrier_scope_name(group.barrier_scope), group.admissible ? "true" : "false",
                  sync_confidence_name(group.confidence), reason.c_str(),
                  members.empty() ? "-" : members.c_str());
    }
    if (patch_options.flavor == rocjitsu::ConSanFlavor::Moi) {
      log_message(kLogInfo, "ConSan MOI inventory reader=%llu candidates=%zu",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  patch_result.moi_candidates.size());
      for (const rocjitsu::ConSanMoiCandidate &candidate : patch_result.moi_candidates) {
        log_message(kLogDebug,
                    "ConSan MOI candidate reader=%llu source=%s container=%s container_kind=%s "
                    "kind=%s mnemonic=%s text_offset=0x%llx file_offset=0x%llx size=%u "
                    "width_bits=%u dst_vgpr=%s addr_vgpr=%s data_vgpr=%s flat_hint=%s "
                    "raw_saddr=%s raw_vaddr=%s raw_vsrc=%s raw_vdst=%s raw_ioffset=%s "
                    "raw_scope=%s raw_th=%s",
                    static_cast<unsigned long long>(code_object_reader.handle),
                    moi_candidate_source_name(candidate.source), candidate.container_name.c_str(),
                    candidate.in_kernel ? "kernel" : "function",
                    lds_access_kind_name(candidate.kind), candidate.mnemonic.c_str(),
                    static_cast<unsigned long long>(candidate.text_offset),
                    static_cast<unsigned long long>(candidate.file_offset), candidate.size,
                    candidate.width_bits,
                    candidate.dst_vgpr ? std::to_string(*candidate.dst_vgpr).c_str() : "-",
                    candidate.addr_vgpr ? std::to_string(*candidate.addr_vgpr).c_str() : "-",
                    candidate.data_vgpr ? std::to_string(*candidate.data_vgpr).c_str() : "-",
                    flat_address_space_hint_name(candidate.flat_address_space_hint),
                    candidate.raw_saddr ? std::to_string(*candidate.raw_saddr).c_str() : "-",
                    candidate.raw_vaddr ? std::to_string(*candidate.raw_vaddr).c_str() : "-",
                    candidate.raw_vsrc ? std::to_string(*candidate.raw_vsrc).c_str() : "-",
                    candidate.raw_vdst ? std::to_string(*candidate.raw_vdst).c_str() : "-",
                    candidate.raw_ioffset ? std::to_string(*candidate.raw_ioffset).c_str() : "-",
                    candidate.raw_scope ? std::to_string(*candidate.raw_scope).c_str() : "-",
                    candidate.raw_th ? std::to_string(*candidate.raw_th).c_str() : "-");
      }
      const rocjitsu::ConSanResourcePlanSummary &resource_summary =
          patch_result.resource_plan_summary;
      log_message(kLogInfo,
                  "ConSan MOI resources reader=%llu explicit=%zu dead=%zu "
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
      struct ResourceFailureSummary {
        size_t count = 0;
        uint16_t min_scratch_vgprs = std::numeric_limits<uint16_t>::max();
        uint16_t max_scratch_vgprs = 0;
        uint16_t min_current_vgprs = std::numeric_limits<uint16_t>::max();
        uint16_t max_current_vgprs = 0;
        uint16_t min_max_referenced_vgprs = std::numeric_limits<uint16_t>::max();
        uint16_t max_max_referenced_vgprs = 0;
        uint16_t min_ordinary_vgpr_limit = std::numeric_limits<uint16_t>::max();
        uint16_t max_ordinary_vgpr_limit = 0;
        uint16_t min_required_vgprs = std::numeric_limits<uint16_t>::max();
        uint16_t max_required_vgprs = 0;
        size_t min_owners = std::numeric_limits<size_t>::max();
        size_t max_owners = 0;
        bool has_indirect_vgpr_access = false;
      };
      constexpr size_t kResourceSiteKindCount =
          static_cast<size_t>(rocjitsu::ConSanResourceSiteKind::Fence) + 1u;
      constexpr size_t kRegisterPlanReasonCount =
          static_cast<size_t>(rocjitsu::ConSanRegisterPlanReason::DynamicStack) + 1u;
      std::array<std::array<ResourceFailureSummary, kRegisterPlanReasonCount>,
                 kResourceSiteKindCount>
          failure_summaries{};
      for (const rocjitsu::ConSanCandidateResourcePlan &plan : patch_result.resource_plans) {
        if (plan.source != rocjitsu::ConSanRegisterAllocationSource::Unsupported)
          continue;
        ResourceFailureSummary &summary = failure_summaries[static_cast<size_t>(plan.site_kind)]
                                                           [static_cast<size_t>(plan.reason)];
        ++summary.count;
        summary.min_scratch_vgprs = std::min(summary.min_scratch_vgprs, plan.scratch_vgpr_count);
        summary.max_scratch_vgprs = std::max(summary.max_scratch_vgprs, plan.scratch_vgpr_count);
        summary.min_current_vgprs = std::min(summary.min_current_vgprs, plan.current_vgpr_count);
        summary.max_current_vgprs = std::max(summary.max_current_vgprs, plan.current_vgpr_count);
        summary.min_max_referenced_vgprs =
            std::min(summary.min_max_referenced_vgprs, plan.max_referenced_vgpr_count);
        summary.max_max_referenced_vgprs =
            std::max(summary.max_max_referenced_vgprs, plan.max_referenced_vgpr_count);
        summary.min_ordinary_vgpr_limit =
            std::min(summary.min_ordinary_vgpr_limit, plan.ordinary_vgpr_limit);
        summary.max_ordinary_vgpr_limit =
            std::max(summary.max_ordinary_vgpr_limit, plan.ordinary_vgpr_limit);
        summary.min_required_vgprs = std::min(summary.min_required_vgprs, plan.required_vgpr_count);
        summary.max_required_vgprs = std::max(summary.max_required_vgprs, plan.required_vgpr_count);
        summary.min_owners =
            std::min(summary.min_owners, plan.owner_descriptor_file_offsets.size());
        summary.max_owners =
            std::max(summary.max_owners, plan.owner_descriptor_file_offsets.size());
        summary.has_indirect_vgpr_access |= plan.has_indirect_vgpr_access;
      }
      for (size_t site_index = 0; site_index < kResourceSiteKindCount; ++site_index) {
        for (size_t reason_index = 0; reason_index < kRegisterPlanReasonCount; ++reason_index) {
          const ResourceFailureSummary &summary = failure_summaries[site_index][reason_index];
          if (summary.count == 0)
            continue;
          log_message(kLogInfo,
                      "ConSan MOI resource-failure reader=%llu site=%s reason=%s count=%zu "
                      "scratch_vgprs=%u..%u current_vgprs=%u..%u "
                      "max_referenced_vgprs=%u..%u ordinary_vgpr_limit=%u..%u "
                      "required_vgprs=%u..%u owners=%zu..%zu indirect_vgprs=%s",
                      static_cast<unsigned long long>(code_object_reader.handle),
                      moi_resource_site_kind_name(
                          static_cast<rocjitsu::ConSanResourceSiteKind>(site_index)),
                      rocjitsu::consan_register_plan_reason_name(
                          static_cast<rocjitsu::ConSanRegisterPlanReason>(reason_index)),
                      summary.count, summary.min_scratch_vgprs, summary.max_scratch_vgprs,
                      summary.min_current_vgprs, summary.max_current_vgprs,
                      summary.min_max_referenced_vgprs, summary.max_max_referenced_vgprs,
                      summary.min_ordinary_vgpr_limit, summary.max_ordinary_vgpr_limit,
                      summary.min_required_vgprs, summary.max_required_vgprs, summary.min_owners,
                      summary.max_owners, summary.has_indirect_vgpr_access ? "true" : "false");
        }
      }
      for (const rocjitsu::ConSanCandidateResourcePlan &plan : patch_result.resource_plans) {
        constexpr size_t kMaxLoggedResourceOwners = 8;
        std::string owner_names;
        size_t logged_owners = 0;
        for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
          if (logged_owners == kMaxLoggedResourceOwners)
            break;
          const auto kernel = std::ranges::find_if(
              patch_result.kernels, [descriptor_offset](const rocjitsu::ConSanKernelInfo &item) {
                return item.descriptor_file_offset == descriptor_offset;
              });
          if (kernel == patch_result.kernels.end())
            continue;
          if (!owner_names.empty())
            owner_names += ',';
          owner_names += kernel->name;
          ++logged_owners;
        }
        if (plan.owner_descriptor_file_offsets.size() > logged_owners) {
          if (!owner_names.empty())
            owner_names += ',';
          owner_names +=
              "+" + std::to_string(plan.owner_descriptor_file_offsets.size() - logged_owners);
        }
        if (owner_names.empty())
          owner_names = "-";
        log_message(
            kLogDebug,
            "ConSan MOI resource reader=%llu site=%s candidate=%zu text_offset=0x%llx "
            "source=%s reason=%s owners=%zu owner_names=%s scratch_vgpr=%s scratch_count=%u "
            "current_vgprs=%u max_referenced_vgprs=%u required_vgprs=%u "
            "current_sgprs=%u max_referenced_sgprs=%u scalar_tail_floor=%u "
            "indirect_sgprs=%s sgpr_reference_coverage=%s "
            "private_bytes=%u",
            static_cast<unsigned long long>(code_object_reader.handle),
            moi_resource_site_kind_name(plan.site_kind), plan.candidate_index,
            static_cast<unsigned long long>(plan.text_offset),
            moi_resource_source_name(plan.source),
            rocjitsu::consan_register_plan_reason_name(plan.reason),
            plan.owner_descriptor_file_offsets.size(), owner_names.c_str(),
            plan.scratch_vgpr ? std::to_string(*plan.scratch_vgpr).c_str() : "-",
            plan.scratch_vgpr_count, plan.current_vgpr_count, plan.max_referenced_vgpr_count,
            plan.required_vgpr_count, plan.current_sgpr_count, plan.max_referenced_sgpr_count,
            plan.scalar_tail_floor, plan.has_indirect_sgpr_access ? "true" : "false",
            plan.sgpr_reference_coverage_complete ? "complete" : "open",
            plan.original_private_segment_size);
        for (size_t alternative_index = 0; alternative_index < plan.alternatives.size();
             ++alternative_index) {
          const rocjitsu::ConSanResourcePlanAlternative &alternative =
              plan.alternatives[alternative_index];
          log_message(kLogInfo,
                      "ConSan MOI resource-alternative reader=%llu site=%s candidate=%zu "
                      "text_offset=0x%llx attempt=%zu kind=%s scratch_count=%u "
                      "source=%s reason=%s outcome=%s",
                      static_cast<unsigned long long>(code_object_reader.handle),
                      moi_resource_site_kind_name(plan.site_kind), plan.candidate_index,
                      static_cast<unsigned long long>(plan.text_offset), alternative_index,
                      rocjitsu::consan_resource_plan_alternative_kind_name(alternative.kind),
                      alternative.scratch_vgpr_count, moi_resource_source_name(alternative.source),
                      rocjitsu::consan_register_plan_reason_name(alternative.reason),
                      rocjitsu::consan_resource_plan_alternative_outcome_name(
                          rocjitsu::consan_resource_plan_alternative_outcome(plan, alternative)));
        }
      }
      if (patch_result.resolved_moi_owner_vgpr || patch_result.resolved_moi_epoch_vgpr ||
          patch_result.resolved_moi_workgroup_key_vgpr ||
          !patch_result.resolved_moi_record_replay_workgroup_vgprs.empty() ||
          patch_result.resolved_moi_exec_save_sgpr || patch_result.resolved_moi_owner_sgpr ||
          patch_result.resolved_moi_persistent_owner_sgpr ||
          patch_result.resolved_moi_persistent_epoch_sgpr ||
          patch_result.resolved_moi_persistent_workgroup_key_sgpr ||
          !patch_result.resolved_moi_record_replay_workgroup_sgprs.empty() ||
          patch_result.moi_private_epoch_automatic) {
        log_message(
            kLogInfo,
            "ConSan MOI persistent reader=%llu owner_vgpr=%s epoch_vgpr=%s "
            "workgroup_key_vgpr=%s rr_workgroup_vgprs=%s/%s/%s "
            "exec_save_sgpr=%s owner_sgpr=%s scalar_owner_sgpr=%s "
            "scalar_epoch_sgpr=%s scalar_workgroup_key_sgpr=%s "
            "rr_workgroup_sgprs=%s/%s/%s automatic_vgprs=%s "
            "automatic_private_epoch=%s automatic_exec_save=%s "
            "automatic_owner_sgpr=%s",
            static_cast<unsigned long long>(code_object_reader.handle),
            patch_result.resolved_moi_owner_vgpr
                ? std::to_string(*patch_result.resolved_moi_owner_vgpr).c_str()
                : "-",
            patch_result.resolved_moi_epoch_vgpr
                ? std::to_string(*patch_result.resolved_moi_epoch_vgpr).c_str()
                : "-",
            patch_result.resolved_moi_workgroup_key_vgpr
                ? std::to_string(*patch_result.resolved_moi_workgroup_key_vgpr).c_str()
                : "-",
            patch_result.resolved_moi_record_replay_workgroup_vgprs.x
                ? std::to_string(*patch_result.resolved_moi_record_replay_workgroup_vgprs.x).c_str()
                : "-",
            patch_result.resolved_moi_record_replay_workgroup_vgprs.y
                ? std::to_string(*patch_result.resolved_moi_record_replay_workgroup_vgprs.y).c_str()
                : "-",
            patch_result.resolved_moi_record_replay_workgroup_vgprs.z
                ? std::to_string(*patch_result.resolved_moi_record_replay_workgroup_vgprs.z).c_str()
                : "-",
            patch_result.resolved_moi_exec_save_sgpr
                ? std::to_string(*patch_result.resolved_moi_exec_save_sgpr).c_str()
                : "-",
            patch_result.resolved_moi_owner_sgpr
                ? std::to_string(*patch_result.resolved_moi_owner_sgpr).c_str()
                : "-",
            patch_result.resolved_moi_persistent_owner_sgpr
                ? std::to_string(*patch_result.resolved_moi_persistent_owner_sgpr).c_str()
                : "-",
            patch_result.resolved_moi_persistent_epoch_sgpr
                ? std::to_string(*patch_result.resolved_moi_persistent_epoch_sgpr).c_str()
                : "-",
            patch_result.resolved_moi_persistent_workgroup_key_sgpr
                ? std::to_string(*patch_result.resolved_moi_persistent_workgroup_key_sgpr).c_str()
                : "-",
            patch_result.resolved_moi_record_replay_workgroup_sgprs.x
                ? std::to_string(*patch_result.resolved_moi_record_replay_workgroup_sgprs.x).c_str()
                : "-",
            patch_result.resolved_moi_record_replay_workgroup_sgprs.y
                ? std::to_string(*patch_result.resolved_moi_record_replay_workgroup_sgprs.y).c_str()
                : "-",
            patch_result.resolved_moi_record_replay_workgroup_sgprs.z
                ? std::to_string(*patch_result.resolved_moi_record_replay_workgroup_sgprs.z).c_str()
                : "-",
            patch_result.moi_persistent_vgprs_automatic ? "true" : "false",
            patch_result.moi_private_epoch_automatic ? "true" : "false",
            patch_result.moi_exec_save_sgprs_automatic ? "true" : "false",
            patch_result.moi_owner_sgpr_automatic ? "true" : "false");
      }
    }
    size_t candidate_kernel_count = 0;
    size_t skipped_kernel_count = 0;
    size_t rejected_kernel_count = 0;
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
    for (const rocjitsu::ConSanKernelInfo &kernel : patch_result.kernels) {
      switch (kernel.preflight_action) {
      case rocjitsu::ConSanPreflightAction::Candidate:
        ++candidate_kernel_count;
        break;
      case rocjitsu::ConSanPreflightAction::Skip:
        ++skipped_kernel_count;
        break;
      case rocjitsu::ConSanPreflightAction::Reject:
        ++rejected_kernel_count;
        break;
      case rocjitsu::ConSanPreflightAction::NotRun:
        break;
      }
      for (const rocjitsu::ConSanLdsSite &site : kernel.lds_sites) {
        if (site.supported_mvp)
          ++supported_lds_site_count;
      }
      flat_site_count += kernel.flat_sites.size();
      flat_group_hint_count += kernel.stats.flat_group_hint_count;
      flat_private_hint_count += kernel.stats.flat_private_hint_count;
      flat_maybe_group_hint_count += kernel.stats.flat_maybe_group_hint_count;
      flat_maybe_private_hint_count += kernel.stats.flat_maybe_private_hint_count;
      flat_global_hint_count += kernel.stats.flat_global_hint_count;
      flat_unknown_hint_count += kernel.stats.flat_unknown_hint_count;
    }
    for (const rocjitsu::ConSanFunctionInfo &function : patch_result.functions) {
      function_lds_site_count += function.lds_sites.size();
      for (const rocjitsu::ConSanLdsSite &site : function.lds_sites) {
        if (site.supported_mvp)
          ++function_supported_lds_site_count;
      }
      function_flat_site_count += function.flat_sites.size();
      function_flat_group_hint_count += function.stats.flat_group_hint_count;
      function_flat_private_hint_count += function.stats.flat_private_hint_count;
      function_flat_maybe_group_hint_count += function.stats.flat_maybe_group_hint_count;
      function_flat_maybe_private_hint_count += function.stats.flat_maybe_private_hint_count;
      function_flat_global_hint_count += function.stats.flat_global_hint_count;
      function_flat_unknown_hint_count += function.stats.flat_unknown_hint_count;
    }
    log_message(kLogInfo,
                "ConSan summary reader=%llu kernels=%zu candidates=%zu skips=%zu "
                "rejects=%zu supported_lds_sites=%zu flat_sites=%zu flat_group_hints=%zu "
                "flat_private_hints=%zu flat_maybe_group_hints=%zu "
                "flat_maybe_private_hints=%zu flat_global_hints=%zu "
                "flat_unknown_hints=%zu functions=%zu function_lds_sites=%zu "
                "function_supported_lds_sites=%zu function_flat_sites=%zu "
                "function_flat_group_hints=%zu function_flat_private_hints=%zu "
                "function_flat_maybe_group_hints=%zu function_flat_maybe_private_hints=%zu "
                "function_flat_global_hints=%zu function_flat_unknown_hints=%zu patches=%zu "
                "modified=%s",
                static_cast<unsigned long long>(code_object_reader.handle),
                patch_result.kernels.size(), candidate_kernel_count, skipped_kernel_count,
                rejected_kernel_count, supported_lds_site_count, flat_site_count,
                flat_group_hint_count, flat_private_hint_count, flat_maybe_group_hint_count,
                flat_maybe_private_hint_count, flat_global_hint_count, flat_unknown_hint_count,
                patch_result.functions.size(), function_lds_site_count,
                function_supported_lds_site_count, function_flat_site_count,
                function_flat_group_hint_count, function_flat_private_hint_count,
                function_flat_maybe_group_hint_count, function_flat_maybe_private_hint_count,
                function_flat_global_hint_count, function_flat_unknown_hint_count,
                patch_result.patches.size(), patch_result.modified ? "true" : "false");
    static_coverage_storage = compute_consan_static_coverage(patch_result, *config);
    const ConSanStaticCoverage &static_coverage = *static_coverage_storage;
    log_message(
        kLogInfo,
        "ConSan coverage reader=%llu flavor=%s engine=%s "
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
        static_cast<unsigned long long>(code_object_reader.handle),
        flavor_name(patch_result.flavor),
        patch_result.flavor == rocjitsu::ConSanFlavor::SuperCollider
            ? "supercollider"
            : rocjitsu::consan_moi_engine_name(patch_result.moi_engine),
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
    for (const rocjitsu::ConSanSiteDispositionRecord &site : patch_result.site_dispositions) {
      if (site.lowering_outcome == rocjitsu::ConSanSiteLoweringOutcome::NotApplicable)
        continue;
      log_message(kLogDebug,
                  "ConSan coverage_site reader=%llu kind=%s disposition=%s reason=%s "
                  "outcome=%s lowering_reason=%s resource_reason=%s "
                  "container=%s scope=%s text=0x%llx mnemonic=%s load=%llu",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  rocjitsu::consan_resource_site_kind_name(site.site_kind),
                  rocjitsu::consan_site_disposition_name(site.disposition),
                  rocjitsu::consan_site_disposition_reason_name(site.reason),
                  rocjitsu::consan_site_lowering_outcome_name(site.lowering_outcome),
                  rocjitsu::consan_site_lowering_reason_name(site.lowering_reason),
                  rocjitsu::consan_register_plan_reason_name(site.resource_reason),
                  site.container_name.c_str(), site.in_kernel ? "kernel" : "function",
                  static_cast<unsigned long long>(site.text_offset), site.mnemonic.c_str(),
                  static_cast<unsigned long long>(load_id));
    }
    for (const rocjitsu::ConSanTextSection &text : patch_result.text_sections) {
      log_message(kLogVerbose, "ConSan text reader=%llu name=%s file=0x%llx vaddr=0x%llx size=%llu",
                  static_cast<unsigned long long>(code_object_reader.handle), text.name.c_str(),
                  static_cast<unsigned long long>(text.file_offset),
                  static_cast<unsigned long long>(text.virtual_address),
                  static_cast<unsigned long long>(text.size));
    }
    for (const rocjitsu::ConSanKernelInfo &kernel : patch_result.kernels) {
      if (kernel.has_text_range) {
        log_message(
            kLogDebug,
            "ConSan kernel reader=%llu name=%s kd_file=0x%llx "
            "text_file=0x%llx entry_text=0x%llx code_size=%llu decoded=%s "
            "dynamic_stack=%s "
            "insts=%llu lds_reads=%llu lds_writes=%llu lds_atomics=%llu ds_other=%llu "
            "flat_reads=%llu flat_writes=%llu flat_atomics=%llu flat_group_hints=%llu "
            "flat_private_hints=%llu flat_maybe_group_hints=%llu "
            "flat_maybe_private_hints=%llu flat_global_hints=%llu "
            "flat_unknown_hints=%llu global_mem=%llu scratch_mem=%llu barriers=%llu "
            "waits=%llu fences=%llu decode_errors=%llu "
            "preflight=%s",
            static_cast<unsigned long long>(code_object_reader.handle), kernel.name.c_str(),
            static_cast<unsigned long long>(kernel.descriptor_file_offset),
            static_cast<unsigned long long>(kernel.text_file_offset),
            static_cast<unsigned long long>(kernel.entry_text_offset),
            static_cast<unsigned long long>(kernel.code_size), kernel.decoded ? "true" : "false",
            kernel.uses_dynamic_stack ? (*kernel.uses_dynamic_stack ? "true" : "false") : "unknown",
            static_cast<unsigned long long>(kernel.stats.instruction_count),
            static_cast<unsigned long long>(kernel.stats.lds_read_count),
            static_cast<unsigned long long>(kernel.stats.lds_write_count),
            static_cast<unsigned long long>(kernel.stats.lds_atomic_count),
            static_cast<unsigned long long>(kernel.stats.ds_other_count),
            static_cast<unsigned long long>(kernel.stats.flat_read_count),
            static_cast<unsigned long long>(kernel.stats.flat_write_count),
            static_cast<unsigned long long>(kernel.stats.flat_atomic_count),
            static_cast<unsigned long long>(kernel.stats.flat_group_hint_count),
            static_cast<unsigned long long>(kernel.stats.flat_private_hint_count),
            static_cast<unsigned long long>(kernel.stats.flat_maybe_group_hint_count),
            static_cast<unsigned long long>(kernel.stats.flat_maybe_private_hint_count),
            static_cast<unsigned long long>(kernel.stats.flat_global_hint_count),
            static_cast<unsigned long long>(kernel.stats.flat_unknown_hint_count),
            static_cast<unsigned long long>(kernel.stats.global_memory_count),
            static_cast<unsigned long long>(kernel.stats.scratch_memory_count),
            static_cast<unsigned long long>(kernel.stats.barrier_count),
            static_cast<unsigned long long>(kernel.stats.wait_count),
            static_cast<unsigned long long>(kernel.stats.fence_like_count),
            static_cast<unsigned long long>(kernel.stats.decode_error_count),
            preflight_action_name(kernel.preflight_action));
      } else {
        log_message(kLogDebug,
                    "ConSan kernel reader=%llu name=%s kd_file=0x%llx "
                    "text_range=unavailable decoded=%s dynamic_stack=%s decode_errors=%llu "
                    "preflight=%s",
                    static_cast<unsigned long long>(code_object_reader.handle), kernel.name.c_str(),
                    static_cast<unsigned long long>(kernel.descriptor_file_offset),
                    kernel.decoded ? "true" : "false",
                    kernel.uses_dynamic_stack ? (*kernel.uses_dynamic_stack ? "true" : "false")
                                              : "unknown",
                    static_cast<unsigned long long>(kernel.stats.decode_error_count),
                    preflight_action_name(kernel.preflight_action));
      }
      for (const std::string &reason : kernel.preflight_reasons) {
        log_message(kLogInfo, "ConSan preflight reader=%llu kernel=%s action=%s reason=%s",
                    static_cast<unsigned long long>(code_object_reader.handle), kernel.name.c_str(),
                    preflight_action_name(kernel.preflight_action), reason.c_str());
      }
      for (const rocjitsu::ConSanLdsSite &site : kernel.lds_sites) {
        log_message(kLogVerbose,
                    "ConSan lds-site reader=%llu kernel=%s kind=%s supported=%s "
                    "mnemonic=%s text=0x%llx file=0x%llx size=%u width=%u "
                    "dst_vgpr=%s addr_vgpr=%s data_vgpr=%s",
                    static_cast<unsigned long long>(code_object_reader.handle), kernel.name.c_str(),
                    lds_access_kind_name(site.kind), site.supported_mvp ? "true" : "false",
                    site.mnemonic.c_str(), static_cast<unsigned long long>(site.text_offset),
                    static_cast<unsigned long long>(site.file_offset), site.size, site.width_bits,
                    site.dst_vgpr ? std::to_string(*site.dst_vgpr).c_str() : "-",
                    site.addr_vgpr ? std::to_string(*site.addr_vgpr).c_str() : "-",
                    site.data_vgpr ? std::to_string(*site.data_vgpr).c_str() : "-");
      }
      for (const rocjitsu::ConSanFlatSite &site : kernel.flat_sites) {
        log_message(kLogVerbose,
                    "ConSan flat-site reader=%llu kernel=%s kind=%s "
                    "mnemonic=%s text=0x%llx file=0x%llx size=%u width=%u "
                    "dst_vgpr=%s addr_vgpr=%s data_vgpr=%s addr_hint=%s raw_saddr=%s "
                    "raw_vaddr=%s raw_vsrc=%s raw_vdst=%s raw_ioffset=%s "
                    "raw_scope=%s raw_th=%s",
                    static_cast<unsigned long long>(code_object_reader.handle), kernel.name.c_str(),
                    lds_access_kind_name(site.kind), site.mnemonic.c_str(),
                    static_cast<unsigned long long>(site.text_offset),
                    static_cast<unsigned long long>(site.file_offset), site.size, site.width_bits,
                    site.dst_vgpr ? std::to_string(*site.dst_vgpr).c_str() : "-",
                    site.addr_vgpr ? std::to_string(*site.addr_vgpr).c_str() : "-",
                    site.data_vgpr ? std::to_string(*site.data_vgpr).c_str() : "-",
                    flat_address_space_hint_name(site.address_space_hint),
                    site.raw_saddr ? std::to_string(*site.raw_saddr).c_str() : "-",
                    site.raw_vaddr ? std::to_string(*site.raw_vaddr).c_str() : "-",
                    site.raw_vsrc ? std::to_string(*site.raw_vsrc).c_str() : "-",
                    site.raw_vdst ? std::to_string(*site.raw_vdst).c_str() : "-",
                    site.raw_ioffset ? std::to_string(*site.raw_ioffset).c_str() : "-",
                    site.raw_scope ? std::to_string(*site.raw_scope).c_str() : "-",
                    site.raw_th ? std::to_string(*site.raw_th).c_str() : "-");
      }
    }
    for (const rocjitsu::ConSanFunctionInfo &function : patch_result.functions) {
      log_message(kLogVerbose,
                  "ConSan function reader=%llu name=%s text_file=0x%llx "
                  "entry_text=0x%llx code_size=%llu decoded=%s insts=%llu lds_reads=%llu "
                  "lds_writes=%llu lds_atomics=%llu ds_other=%llu flat_reads=%llu "
                  "flat_writes=%llu flat_atomics=%llu flat_group_hints=%llu "
                  "flat_private_hints=%llu flat_maybe_group_hints=%llu "
                  "flat_maybe_private_hints=%llu flat_global_hints=%llu "
                  "flat_unknown_hints=%llu global_mem=%llu scratch_mem=%llu barriers=%llu "
                  "waits=%llu fences=%llu decode_errors=%llu",
                  static_cast<unsigned long long>(code_object_reader.handle), function.name.c_str(),
                  static_cast<unsigned long long>(function.text_file_offset),
                  static_cast<unsigned long long>(function.entry_text_offset),
                  static_cast<unsigned long long>(function.code_size),
                  function.decoded ? "true" : "false",
                  static_cast<unsigned long long>(function.stats.instruction_count),
                  static_cast<unsigned long long>(function.stats.lds_read_count),
                  static_cast<unsigned long long>(function.stats.lds_write_count),
                  static_cast<unsigned long long>(function.stats.lds_atomic_count),
                  static_cast<unsigned long long>(function.stats.ds_other_count),
                  static_cast<unsigned long long>(function.stats.flat_read_count),
                  static_cast<unsigned long long>(function.stats.flat_write_count),
                  static_cast<unsigned long long>(function.stats.flat_atomic_count),
                  static_cast<unsigned long long>(function.stats.flat_group_hint_count),
                  static_cast<unsigned long long>(function.stats.flat_private_hint_count),
                  static_cast<unsigned long long>(function.stats.flat_maybe_group_hint_count),
                  static_cast<unsigned long long>(function.stats.flat_maybe_private_hint_count),
                  static_cast<unsigned long long>(function.stats.flat_global_hint_count),
                  static_cast<unsigned long long>(function.stats.flat_unknown_hint_count),
                  static_cast<unsigned long long>(function.stats.global_memory_count),
                  static_cast<unsigned long long>(function.stats.scratch_memory_count),
                  static_cast<unsigned long long>(function.stats.barrier_count),
                  static_cast<unsigned long long>(function.stats.wait_count),
                  static_cast<unsigned long long>(function.stats.fence_like_count),
                  static_cast<unsigned long long>(function.stats.decode_error_count));
      for (const rocjitsu::ConSanLdsSite &site : function.lds_sites) {
        log_message(kLogVerbose,
                    "ConSan function-lds-site reader=%llu function=%s kind=%s "
                    "supported=%s mnemonic=%s text=0x%llx file=0x%llx size=%u width=%u "
                    "dst_vgpr=%s addr_vgpr=%s data_vgpr=%s",
                    static_cast<unsigned long long>(code_object_reader.handle),
                    function.name.c_str(), lds_access_kind_name(site.kind),
                    site.supported_mvp ? "true" : "false", site.mnemonic.c_str(),
                    static_cast<unsigned long long>(site.text_offset),
                    static_cast<unsigned long long>(site.file_offset), site.size, site.width_bits,
                    site.dst_vgpr ? std::to_string(*site.dst_vgpr).c_str() : "-",
                    site.addr_vgpr ? std::to_string(*site.addr_vgpr).c_str() : "-",
                    site.data_vgpr ? std::to_string(*site.data_vgpr).c_str() : "-");
      }
      for (const rocjitsu::ConSanFlatSite &site : function.flat_sites) {
        log_message(kLogVerbose,
                    "ConSan function-flat-site reader=%llu function=%s kind=%s "
                    "mnemonic=%s text=0x%llx file=0x%llx size=%u width=%u "
                    "dst_vgpr=%s addr_vgpr=%s data_vgpr=%s addr_hint=%s raw_saddr=%s "
                    "raw_vaddr=%s raw_vsrc=%s raw_vdst=%s raw_ioffset=%s raw_scope=%s "
                    "raw_th=%s",
                    static_cast<unsigned long long>(code_object_reader.handle),
                    function.name.c_str(), lds_access_kind_name(site.kind), site.mnemonic.c_str(),
                    static_cast<unsigned long long>(site.text_offset),
                    static_cast<unsigned long long>(site.file_offset), site.size, site.width_bits,
                    site.dst_vgpr ? std::to_string(*site.dst_vgpr).c_str() : "-",
                    site.addr_vgpr ? std::to_string(*site.addr_vgpr).c_str() : "-",
                    site.data_vgpr ? std::to_string(*site.data_vgpr).c_str() : "-",
                    flat_address_space_hint_name(site.address_space_hint),
                    site.raw_saddr ? std::to_string(*site.raw_saddr).c_str() : "-",
                    site.raw_vaddr ? std::to_string(*site.raw_vaddr).c_str() : "-",
                    site.raw_vsrc ? std::to_string(*site.raw_vsrc).c_str() : "-",
                    site.raw_vdst ? std::to_string(*site.raw_vdst).c_str() : "-",
                    site.raw_ioffset ? std::to_string(*site.raw_ioffset).c_str() : "-",
                    site.raw_scope ? std::to_string(*site.raw_scope).c_str() : "-",
                    site.raw_th ? std::to_string(*site.raw_th).c_str() : "-");
      }
    }
    for (const rocjitsu::ConSanPatchInfo &patch : patch_result.patches) {
      const std::string scratch_vgpr =
          patch.scratch_vgpr ? std::to_string(*patch.scratch_vgpr) : "-";
      const std::string private_epoch_offset =
          patch.persistent_epoch_private_offset
              ? std::to_string(*patch.persistent_epoch_private_offset)
              : "-";
      const std::string scalar_vcc_spill_vgpr =
          patch.scalar_vcc_spill_vgpr ? std::to_string(*patch.scalar_vcc_spill_vgpr) : "-";
      const std::string scalar_vcc_spill_sgpr =
          patch.scalar_vcc_spill_sgpr ? std::to_string(*patch.scalar_vcc_spill_sgpr) : "-";
      log_message(kLogDebug,
                  "ConSan proof patch reader=%llu kind=%s anchor=0x%llx "
                  "trampoline=0x%llx original_size=%u trampoline_size=%u scratch_vgpr=%s "
                  "scalar_vcc_spill_sgpr=%s scalar_vcc_spill_vgpr=%s "
                  "scalar_vcc_spill_vgpr_count=%u "
                  "private_epoch_offset=%s spilled_vgprs=%u "
                  "private_bytes=%u dynamic_private_addend=%u "
                  "workgroup_shadow_base=%u workgroup_shadow_bytes=%u group_bytes=%u "
                  "sampled_first_slot=%u sampled_window_banks=%u sampled_access_kind=%u",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  patch_kind_name(patch.kind), static_cast<unsigned long long>(patch.anchor_offset),
                  static_cast<unsigned long long>(patch.trampoline_offset), patch.original_size,
                  patch.trampoline_size, scratch_vgpr.c_str(), scalar_vcc_spill_sgpr.c_str(),
                  scalar_vcc_spill_vgpr.c_str(), patch.scalar_vcc_spill_vgpr_count,
                  private_epoch_offset.c_str(), patch.spilled_vgpr_count,
                  patch.required_private_segment_size, patch.dynamic_private_segment_addend,
                  patch.workgroup_shadow_base, patch.workgroup_shadow_size,
                  patch.required_group_segment_size, patch.sampled_first_slot,
                  patch.sampled_window_bank_count,
                  static_cast<uint32_t>(patch.sampled_access_kind));
    }
    detailed_log_batch.flush();
    if (config->require_patch && !config->fault_dry_run &&
        !has_consan_site_instrumentation_patch(patch_result)) {
      const bool required = (patch_options.flavor == rocjitsu::ConSanFlavor::SuperCollider &&
                             require_patch_applies_to(patch_result, *config)) ||
                            (patch_options.flavor == rocjitsu::ConSanFlavor::Moi &&
                             require_moi_patch_applies_to(patch_result));
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

    if (install_action == rocjitsu::ConSanInstallAction::LoadReplacement) {
      dump_code_object_bytes(
          *config, dump_id, code_object_reader.handle, "patched",
          std::span<const uint8_t>(patch_result.elf_bytes.data(), patch_result.elf_bytes.size()));
    } else {
      // All metadata consumers above are finished. Discard the unused final
      // image and its reservation together before invoking the original
      // loader, which may block or re-enter the hook.
      transform_state.discard_image_and_release(patch_result_storage->elf_bytes);
    }
  } else {
    log_message(kLogInfo, "ConSan pass-through reader=%llu bytes=unavailable",
                static_cast<unsigned long long>(code_object_reader.handle));
  }

  if (patch_result_storage && install_action == rocjitsu::ConSanInstallAction::LoadReplacement) {
    auto *original_create = layer().create_from_memory();
    if (original_create == nullptr) {
      record_static_coverage(false);
      return HSA_STATUS_ERROR;
    }

    const size_t input_size = patch_result_storage->input_size;
    const size_t replacement_size = patch_result_storage->elf_bytes.size();
    const uint64_t replacement_growth_bytes =
        replacement_size > input_size ? static_cast<uint64_t>(replacement_size - input_size) : 0;
    try {
      replacement_storage =
          std::make_shared<const std::vector<uint8_t>>(std::move(patch_result_storage->elf_bytes));
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
      transform_state.discard_image_and_release(patch_result_storage->elf_bytes);
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
    load_status = original_load(executable, agent, code_object_reader, options, loaded_code_object);
  }
  if (load_status == HSA_STATUS_SUCCESS && using_replacement_reader && patch_result_storage) {
    auto_report_guard.bind_to_executable(executable);
    if (patch_result_storage->applied_fault_mutations != 0u) {
      fault_installation_evidence.mark_installed();
      process_fault_application_reservation.commit_applied_mutation();
    }
    KernelPrivateDispatchRegistry::instance().note_patch_requirements(executable,
                                                                      *patch_result_storage);
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
  return load_status;
}

} // namespace rocjitsu::consan_hook

using namespace rocjitsu::consan_hook;

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
  if (!config)
    return false;

  g_log_level.store(config->log_level, std::memory_order_relaxed);
  if (!config->enabled) {
    log_message(kLogInfo, "ConSan is disabled; not installing wrappers");
    return true;
  }

  return layer().install(table, *config);
}

extern "C" RJ_HOOK_EXPORT void OnUnload() { layer().uninstall(); }

extern "C" RJ_HOOK_EXPORT void
rj_dbi_test_set_consan_transform_override(ConSanTransformOverride override) {
  g_test_consan_transform_override.store(override, std::memory_order_release);
  g_test_consan_moi_retry_count.store(0, std::memory_order_relaxed);
}

extern "C" RJ_HOOK_EXPORT void rj_dbi_test_set_log_sink_override(LogSinkOverride override) {
  g_test_log_sink_override.store(override, std::memory_order_release);
}

extern "C" RJ_HOOK_EXPORT size_t rj_dbi_test_consan_moi_retry_count() {
  return g_test_consan_moi_retry_count.load(std::memory_order_relaxed);
}
