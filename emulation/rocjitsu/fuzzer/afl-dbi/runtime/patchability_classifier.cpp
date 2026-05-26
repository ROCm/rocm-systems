#include "patchability_classifier.h"

namespace rocjitsu::fuzzer::afl_dbi {
namespace {

bool kernel_name_matches_filter(std::string_view name, const char *filter) {
  return filter == nullptr || filter[0] == '\0' || name.find(filter) != std::string_view::npos;
}

bool is_runtime_internal_kernel(std::string_view name) {
  return name.rfind("__amd_rocclr_", 0) == 0;
}

bool has_prior_invalid_dispatch(std::string_view name) {
  // TODO(rocfuzz): rocFFT's generated twiddle kernels have direct branch sites,
  // but fixed high-register probes produced HSA invalid-dispatch failures.
  // Retrying after loader-visible scratch support still produced a GPU memory
  // fault: block-entry scratch remained blocked, and branch scratch/fallback
  // still selected no scratch-spill points because no allocated saved-EXEC SGPR
  // pair was liveness-safe. Keep twiddle out of automatic coverage until
  // block-entry scratch, SGPR spill support, or a twiddle-specific policy is
  // proven.
  return name.find("twiddle_gen_") != std::string_view::npos;
}

bool has_prior_entry_redirection_failure(std::string_view name) {
  // Tensile GEMM kernels have useful self-contained branch sites, but entry
  // redirection has been unsafe in default coverage runs. This is prior
  // patchability knowledge, not a user-visible mode choice.
  if (name.rfind("Cijk_", 0) == 0)
    return true;

  // MIOpen activation kernels faulted with HSA illegal-instruction errors even
  // after branch terminators were left unpatched. That points at entry
  // redirection rather than branch probes as the unsafe operation.
  return name.rfind("MIOpenActive", 0) == 0;
}

bool prefers_fixed_branch_counters(std::string_view name) {
  // Until previous-BB state liveness and entry interactions are proven for
  // MIOpen activation kernels, keep their self-contained branch probes on the
  // smaller fixed-counter sequence.
  return name.rfind("MIOpenActive", 0) == 0;
}

} // namespace

KernelPatchability classify_kernel_patchability(
    std::string_view name, const KernelPatchabilityFilters &filters) {
  KernelPatchability decision;

  if (is_runtime_internal_kernel(name)) {
    decision.instrumentable = false;
    decision.entry_probe_safe = false;
    decision.self_contained_probe_safe = false;
    decision.branch_probe_safe = false;
    decision.instrumentation_reason = "runtime-internal-kernel";
    decision.entry_probe_reason = "not-instrumented";
    decision.self_contained_probe_reason = "not-instrumented";
    decision.branch_probe_reason = "not-instrumented";
    return decision;
  }

  if (!kernel_name_matches_filter(name, filters.include)) {
    decision.instrumentable = false;
    decision.entry_probe_safe = false;
    decision.self_contained_probe_safe = false;
    decision.branch_probe_safe = false;
    decision.instrumentation_reason = "kernel-include-filter";
    decision.entry_probe_reason = "not-instrumented";
    decision.self_contained_probe_reason = "not-instrumented";
    decision.branch_probe_reason = "not-instrumented";
    return decision;
  }

  if (filters.exclude != nullptr && filters.exclude[0] != '\0' &&
      name.find(filters.exclude) != std::string_view::npos) {
    decision.instrumentable = false;
    decision.entry_probe_safe = false;
    decision.self_contained_probe_safe = false;
    decision.branch_probe_safe = false;
    decision.instrumentation_reason = "kernel-exclude-filter";
    decision.entry_probe_reason = "not-instrumented";
    decision.self_contained_probe_reason = "not-instrumented";
    decision.branch_probe_reason = "not-instrumented";
    return decision;
  }

  if (has_prior_invalid_dispatch(name)) {
    decision.instrumentable = false;
    decision.entry_probe_safe = false;
    decision.self_contained_probe_safe = false;
    decision.branch_probe_safe = false;
    decision.instrumentation_reason = "prior-invalid-dispatch-kernel";
    decision.entry_probe_reason = "prior-invalid-dispatch-kernel";
    decision.self_contained_probe_reason = "prior-invalid-dispatch-kernel";
    decision.branch_probe_reason = "prior-invalid-dispatch-kernel";
    return decision;
  }

  if (has_prior_entry_redirection_failure(name)) {
    decision.entry_probe_safe = false;
    decision.prefers_self_contained_edge_probes = true;
    decision.entry_probe_reason = "prior-entry-redirection-unsafe";
    decision.self_contained_strategy_reason = "classifier-entry-redirection-unsafe";
  }

  if (prefers_fixed_branch_counters(name)) {
    decision.prefers_fixed_branch_counters = true;
    decision.fixed_branch_strategy_reason = "classifier-fixed-branch-preferred";
  }

  return decision;
}

LoaderPatchability classify_loader_patchability(std::string_view context) {
  LoaderPatchability decision;

  if (context.find("lazy CCOB") != std::string_view::npos) {
    decision.prefers_self_contained_edge_probes = true;
    decision.allows_vgpr_scratch_spills = false;
    decision.self_contained_strategy_reason = "loader-context-self-contained";
    decision.vgpr_scratch_spill_reason = "loader-context-scratch-spills-unproven";
  }

  if (context.find("runtime KPACK shadow") != std::string_view::npos) {
    decision.prefers_self_contained_edge_probes = true;
    decision.prefers_fixed_branch_counters = true;
    decision.allows_vgpr_scratch_spills = false;
    decision.self_contained_strategy_reason = "loader-context-self-contained";
    decision.fixed_branch_strategy_reason = "loader-context-fixed-branch-preferred";
    decision.vgpr_scratch_spill_reason = "loader-context-scratch-spills-unproven";
  }

  if (context.find("HSA memory reader") != std::string_view::npos) {
    decision.prefers_fixed_branch_counters = true;
    decision.allows_vgpr_scratch_spills = false;
    decision.fixed_branch_strategy_reason = "loader-context-fixed-branch-preferred";
    decision.vgpr_scratch_spill_reason = "loader-context-scratch-spills-unproven";
  }

  return decision;
}

std::optional<std::string_view> code_object_self_contained_edge_probe_reason(
    std::span<const KernelSite> sites, const KernelPatchabilityFilters &filters) {
  bool saw_instrumentable = false;
  bool all_instrumentable_entries_unsafe = true;
  std::optional<std::string_view> first_self_contained_reason;

  for (const KernelSite &site : sites) {
    const KernelPatchability decision = classify_kernel_patchability(site.name, filters);
    if (!decision.instrumentable)
      continue;

    saw_instrumentable = true;
    if (decision.prefers_self_contained_edge_probes &&
        decision.self_contained_strategy_reason != nullptr) {
      if (!first_self_contained_reason)
        first_self_contained_reason = std::string_view(decision.self_contained_strategy_reason);
    }
    if (decision.entry_probe_safe)
      all_instrumentable_entries_unsafe = false;
  }

  if (saw_instrumentable && all_instrumentable_entries_unsafe) {
    if (first_self_contained_reason)
      return first_self_contained_reason;
    return std::string_view("all-instrumentable-entries-unsafe");
  }
  return std::nullopt;
}

} // namespace rocjitsu::fuzzer::afl_dbi
