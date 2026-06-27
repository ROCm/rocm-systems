//===- hotswap_loader_policy.cpp - HotSwap loader decision policy ---------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap_loader_policy.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>

namespace rocr::hotswap {

bool gate_allows_hotswap(const AgentGfxRevision &gfx) {
  return gfx.revision_valid && gfx.gfx_target == "gfx1250" &&
         gfx.asic_revision == 0; // A0
}

namespace {

bool is_gfx12_5_target(const std::string &gfx_target) {
  constexpr char Gfx125Prefix[] = "gfx125";
  constexpr size_t Gfx125PrefixLen = sizeof(Gfx125Prefix) - 1;
  if (gfx_target == "gfx12-5-generic") {
    return true;
  }
  if (gfx_target.size() <= Gfx125PrefixLen ||
      gfx_target.compare(0, Gfx125PrefixLen, Gfx125Prefix) != 0) {
    return false;
  }
  return std::all_of(gfx_target.begin() + Gfx125PrefixLen, gfx_target.end(),
                     [](unsigned char c) { return std::isdigit(c); });
}

std::string with_gfx1250_stepping_feature(const std::string &isa_name,
                                          bool is_b0) {
  if (extract_gfx_target(isa_name) != "gfx1250" ||
      isa_name.find(":gfx1250-b0-specific+") != std::string::npos ||
      isa_name.find(":gfx1250-b0-specific-") != std::string::npos) {
    return isa_name;
  }
  return isa_name + (is_b0 ? ":gfx1250-b0-specific+" : ":gfx1250-b0-specific-");
}

} // namespace

bool has_candidate_hotswap_rewrite(const AgentGfxRevision &gfx,
                                   const RewriteOptions &options) {
  return gate_allows_hotswap(gfx) ||
         (options.gfx12_5_rewrite_requested &&
          is_gfx12_5_target(gfx.gfx_target));
}

std::optional<RewriteDecision>
decide_hotswap_rewrite(const AgentGfxRevision &gfx,
                       const std::string &source_isa,
                       const std::string &target_isa,
                       const RewriteOptions &options) {
  if (source_isa.empty() || target_isa.empty()) {
    return std::nullopt;
  }

  std::string source_gfx = extract_gfx_target(source_isa);
  std::string target_gfx = extract_gfx_target(target_isa);

  if (gate_allows_hotswap(gfx) && source_gfx == "gfx1250" &&
      target_gfx == "gfx1250") {
    return RewriteDecision{with_gfx1250_stepping_feature(source_isa, true),
                           with_gfx1250_stepping_feature(target_isa, false)};
  }

  if (!options.gfx12_5_rewrite_requested ||
      !is_gfx12_5_target(gfx.gfx_target) || !is_gfx12_5_target(source_gfx)) {
    return std::nullopt;
  }

  // ROCm/rocm-systems#7581 established the loader-side invariant that this
  // opt-in path uses the code object's processor, not a source->agent retarget.
  RewriteDecision decision{source_isa, source_isa};

  if (source_gfx == "gfx1250") {
    decision.source_isa = with_gfx1250_stepping_feature(source_isa, true);
    decision.target_isa = with_gfx1250_stepping_feature(source_isa, true);
  }

  return decision;
}

} // namespace rocr::hotswap
