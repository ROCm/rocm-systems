#include "patchability_classifier.h"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace afl_dbi = rocjitsu::fuzzer::afl_dbi;

namespace {

void check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

afl_dbi::KernelSite kernel(std::string name) {
  afl_dbi::KernelSite site;
  site.name = std::move(name);
  return site;
}

} // namespace

int main() {
  using afl_dbi::KernelPatchabilityFilters;
  using afl_dbi::classify_kernel_patchability;
  using afl_dbi::classify_loader_patchability;
  using afl_dbi::code_object_self_contained_edge_probe_reason;

  const auto ordinary = classify_kernel_patchability("ordinary_kernel");
  check(ordinary.instrumentable, "ordinary kernel should be instrumentable");
  check(ordinary.entry_probe_safe, "ordinary kernel should allow entry probes");
  check(ordinary.self_contained_probe_safe,
        "ordinary kernel should allow self-contained probes");
  check(ordinary.branch_probe_safe, "ordinary kernel should allow branch probes");
  check(!ordinary.prefers_self_contained_edge_probes,
        "ordinary kernel should not force self-contained planning");

  const auto runtime_internal =
      classify_kernel_patchability("__amd_rocclr_fillBuffer");
  check(!runtime_internal.instrumentable,
        "runtime internal kernel should be filtered out");
  check(std::string_view(runtime_internal.instrumentation_reason) ==
            "runtime-internal-kernel",
        "runtime internal reason changed");

  const auto include_filtered =
      classify_kernel_patchability("kernel_a", KernelPatchabilityFilters{"kernel_b", nullptr});
  check(!include_filtered.instrumentable, "include filter should reject non-matches");
  check(std::string_view(include_filtered.instrumentation_reason) ==
            "kernel-include-filter",
        "include filter reason changed");

  const auto exclude_filtered =
      classify_kernel_patchability("kernel_a", KernelPatchabilityFilters{nullptr, "kernel"});
  check(!exclude_filtered.instrumentable, "exclude filter should reject matches");
  check(std::string_view(exclude_filtered.instrumentation_reason) ==
            "kernel-exclude-filter",
        "exclude filter reason changed");

  const auto twiddle =
      classify_kernel_patchability("twiddle_gen_radices_sp");
  check(!twiddle.instrumentable, "known unsafe twiddle kernel should not be patched");
  check(!twiddle.entry_probe_safe, "twiddle entry probe should be unsafe");
  check(!twiddle.self_contained_probe_safe,
        "twiddle self-contained probe should be unsafe");
  check(!twiddle.branch_probe_safe, "twiddle branch probe should be unsafe");
  check(std::string_view(twiddle.instrumentation_reason) ==
            "prior-invalid-dispatch-kernel",
        "twiddle reason changed");

  const auto tensile = classify_kernel_patchability("Cijk_Ailk_Bljk_SB_MT64x64");
  check(tensile.instrumentable, "Tensile kernel should remain instrumentable");
  check(!tensile.entry_probe_safe, "Tensile entry redirection should be unsafe");
  check(tensile.self_contained_probe_safe,
        "Tensile self-contained branch probes should be allowed");
  check(tensile.branch_probe_safe, "Tensile branch probes should be allowed");
  check(tensile.prefers_self_contained_edge_probes,
        "Tensile kernel should prefer self-contained planning");
  check(std::string_view(tensile.self_contained_strategy_reason) ==
            "classifier-entry-redirection-unsafe",
        "Tensile self-contained reason changed");

  const auto miopen = classify_kernel_patchability("MIOpenActiveFwdLite");
  check(miopen.instrumentable, "MIOpen activation kernel should remain instrumentable");
  check(!miopen.entry_probe_safe,
        "MIOpen activation entry redirection should be unsafe");
  check(miopen.self_contained_probe_safe,
        "MIOpen activation self-contained probes should remain generally allowed");
  check(miopen.branch_probe_safe, "MIOpen activation branch probes should be allowed");
  check(miopen.prefers_self_contained_edge_probes,
        "MIOpen activation should prefer self-contained planning");
  check(miopen.prefers_fixed_branch_counters,
        "MIOpen activation should prefer fixed branch counters");
  check(std::string_view(miopen.self_contained_strategy_reason) ==
            "classifier-entry-redirection-unsafe",
        "MIOpen activation self-contained reason changed");
  check(std::string_view(miopen.fixed_branch_strategy_reason) ==
            "classifier-fixed-branch-preferred",
        "MIOpen activation fixed branch reason changed");

  const auto lazy_ccob = classify_loader_patchability("lazy CCOB module");
  check(lazy_ccob.prefers_self_contained_edge_probes,
        "lazy CCOB should prefer self-contained probes");
  check(!lazy_ccob.prefers_fixed_branch_counters,
        "lazy CCOB should not force fixed counters");
  check(!lazy_ccob.allows_vgpr_scratch_spills,
        "lazy CCOB scratch-backed probes are not proven safe yet");
  check(std::string_view(lazy_ccob.vgpr_scratch_spill_reason) ==
            "loader-context-scratch-spills-unproven",
        "lazy CCOB scratch-spill reason changed");

  const auto hsa_reader = classify_loader_patchability("HSA memory reader");
  check(!hsa_reader.prefers_self_contained_edge_probes,
        "HSA reader should not force self-contained before branch preflight");
  check(hsa_reader.prefers_fixed_branch_counters,
        "HSA reader should prefer fixed branch counters");
  check(!hsa_reader.allows_vgpr_scratch_spills,
        "HSA reader scratch-backed probes are not proven safe yet");
  check(std::string_view(hsa_reader.vgpr_scratch_spill_reason) ==
            "loader-context-scratch-spills-unproven",
        "HSA reader scratch-spill reason changed");

  const auto kpack_shadow = classify_loader_patchability("runtime KPACK shadow");
  check(kpack_shadow.prefers_self_contained_edge_probes,
        "runtime KPACK shadow should prefer self-contained probes");
  check(kpack_shadow.prefers_fixed_branch_counters,
        "runtime KPACK shadow should prefer fixed branch counters");
  check(!kpack_shadow.allows_vgpr_scratch_spills,
        "runtime KPACK scratch-backed probes are not proven safe yet");
  check(std::string_view(kpack_shadow.vgpr_scratch_spill_reason) ==
            "loader-context-scratch-spills-unproven",
        "runtime KPACK scratch-spill reason changed");

  std::vector<afl_dbi::KernelSite> sites;
  sites.push_back(kernel("ordinary_kernel"));
  check(!code_object_self_contained_edge_probe_reason(sites),
        "ordinary code object should not force self-contained planning");

  sites.push_back(kernel("Cijk_Ailk_Bljk_SB_MT64x64"));
  check(!code_object_self_contained_edge_probe_reason(sites),
        "mixed entry-safe/entry-unsafe code object should use per-kernel planning");

  std::vector<afl_dbi::KernelSite> unsafe_sites;
  unsafe_sites.push_back(kernel("Cijk_Ailk_Bljk_SB_MT64x64"));
  unsafe_sites.push_back(kernel("MIOpenActiveFwdLite"));
  const std::optional<std::string_view> tensile_reason =
      code_object_self_contained_edge_probe_reason(unsafe_sites);
  check(tensile_reason &&
            *tensile_reason == "classifier-entry-redirection-unsafe",
        "all-entry-unsafe code object self-contained reason changed");

  std::vector<afl_dbi::KernelSite> filtered_sites;
  filtered_sites.push_back(kernel("twiddle_gen_radices_sp"));
  check(!code_object_self_contained_edge_probe_reason(filtered_sites),
        "fully skipped code object should not force a self-contained strategy");

  return 0;
}
