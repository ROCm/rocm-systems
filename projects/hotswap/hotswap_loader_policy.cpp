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

namespace {

constexpr char Gfx1250[] = "gfx1250";
constexpr char Gfx12_5Generic[] = "gfx12-5-generic";
constexpr char Gfx125Prefix[] = "gfx125";
constexpr char Gfx1250B0Feature[] = ":gfx1250-b0-specific+";
constexpr char Gfx1250A0Feature[] = ":gfx1250-b0-specific-";

enum class Gfx1250Stepping {
  B0,
  A0,
};

const char *gfx1250_stepping_feature(Gfx1250Stepping stepping) {
  return stepping == Gfx1250Stepping::B0 ? Gfx1250B0Feature : Gfx1250A0Feature;
}

bool is_gfx12_5_target(const std::string &gfx_target) {
  constexpr size_t Gfx125PrefixLen = sizeof(Gfx125Prefix) - 1;
  if (gfx_target == Gfx12_5Generic) {
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
                                          Gfx1250Stepping stepping) {
  if (extract_gfx_target(isa_name) != Gfx1250 ||
      isa_name.find(Gfx1250B0Feature) != std::string::npos ||
      isa_name.find(Gfx1250A0Feature) != std::string::npos) {
    return isa_name;
  }
  return isa_name + gfx1250_stepping_feature(stepping);
}

} // namespace

bool gate_allows_hotswap(const AgentGfxRevision &gfx) {
  return gfx.revision_valid && gfx.gfx_target == Gfx1250 &&
         gfx.asic_revision == 0; // A0
}

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

  if (gate_allows_hotswap(gfx) && source_gfx == Gfx1250 &&
      extract_gfx_target(target_isa) == Gfx1250) {
    return RewriteDecision{
        with_gfx1250_stepping_feature(source_isa, Gfx1250Stepping::B0),
        with_gfx1250_stepping_feature(target_isa, Gfx1250Stepping::A0)};
  }

  if (!options.gfx12_5_rewrite_requested ||
      !is_gfx12_5_target(gfx.gfx_target) || !is_gfx12_5_target(source_gfx)) {
    return std::nullopt;
  }

  // ROCm/rocm-systems#7581 established the loader-side invariant that this
  // opt-in path uses the code object's processor, not a source->agent retarget.
  RewriteDecision decision{source_isa, source_isa};

  if (source_gfx == Gfx1250) {
    decision.source_isa =
        with_gfx1250_stepping_feature(source_isa, Gfx1250Stepping::B0);
    decision.target_isa =
        with_gfx1250_stepping_feature(source_isa, Gfx1250Stepping::B0);
  }

  return decision;
}

} // namespace rocr::hotswap
