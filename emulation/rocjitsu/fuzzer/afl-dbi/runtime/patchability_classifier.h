#pragma once

#include "rocjitsu_fuzzer/afl_dbi_plan.h"

#include <optional>
#include <span>
#include <string_view>

namespace rocjitsu::fuzzer::afl_dbi {

struct KernelPatchabilityFilters {
  const char *include = nullptr;
  const char *exclude = nullptr;
};

struct KernelPatchability {
  bool instrumentable = true;
  bool entry_probe_safe = true;
  bool self_contained_probe_safe = true;
  bool branch_probe_safe = true;
  bool prefers_self_contained_edge_probes = false;
  bool prefers_fixed_branch_counters = false;
  const char *instrumentation_reason = "instrumentable";
  const char *entry_probe_reason = "entry-probe-safe";
  const char *self_contained_probe_reason = "self-contained-probe-safe";
  const char *branch_probe_reason = "branch-probe-safe";
  const char *self_contained_strategy_reason = nullptr;
  const char *fixed_branch_strategy_reason = nullptr;
};

struct LoaderPatchability {
  bool prefers_self_contained_edge_probes = false;
  bool prefers_fixed_branch_counters = false;
  bool allows_vgpr_scratch_spills = true;
  const char *self_contained_strategy_reason = "default-entry-preferred";
  const char *fixed_branch_strategy_reason = nullptr;
  const char *vgpr_scratch_spill_reason = "loader-context-scratch-spills-supported";
};

KernelPatchability classify_kernel_patchability(
    std::string_view name, const KernelPatchabilityFilters &filters = {});

LoaderPatchability classify_loader_patchability(std::string_view context);

std::optional<std::string_view> code_object_self_contained_edge_probe_reason(
    std::span<const KernelSite> sites, const KernelPatchabilityFilters &filters = {});

} // namespace rocjitsu::fuzzer::afl_dbi
